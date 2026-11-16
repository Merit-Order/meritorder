"""How fast each route answers, learned from the gateway's own traffic.

The same model often costs the same everywhere (Claude on Anthropic, Bedrock, Foundry and OpenRouter), so price
alone can't choose between those routes. Each route (backend + upstream model) keeps a moving average of its time to
first token and of how long each output token takes after that, so the router can put the faster one first, and
teams that choose speed over cost can pay a little more for it (``routing.priority``).

Timings start when the route was called, not when the gateway received the request, so a route that took over
after another one failed isn't charged for the failed attempt. Each gateway process learns from its own requests.
"""

from __future__ import annotations

from dataclasses import dataclass

ALPHA = 0.2  # weight of the newest sample: recent behavior matters, one slow reply doesn't decide
MIN_SAMPLES = 2  # a route's speed is trusted after this many replies
STALE_SECONDS = 600.0  # a route not measured for this long is checked again now and then
DEFAULT_SECONDS_PER_TOKEN = 0.01  # 100 tokens/s, for a route whose streams haven't been timed yet
MIN_TIMED_TOKENS = 16  # below this, a reply is too short to time the writing speed


def _ewma(old: float | None, new: float) -> float:
    return new if old is None else old + ALPHA * (new - old)


@dataclass
class RouteSpeed:
    first_token: float | None = None  # seconds until the first token
    per_token: float | None = None  # seconds per output token after the first (from streamed replies)
    samples: int = 0
    last: float = 0.0  # wall-clock time of the latest sample

    @property
    def measured(self) -> bool:
        return self.samples >= MIN_SAMPLES

    def stale(self, now: float) -> bool:
        return not self.measured or now - self.last > STALE_SECONDS

    def expected(self, output_tokens: int) -> float:
        """Seconds for a reply of ``output_tokens``: the wait for the first token, then the writing."""
        per = self.per_token if self.per_token is not None else DEFAULT_SECONDS_PER_TOKEN
        return (self.first_token or 0.0) + output_tokens * per

    def tokens_per_second(self) -> float | None:
        return 1 / self.per_token if self.per_token else None


class Speeds:
    """Per-route speed for this gateway process. It outlives configuration reloads (the router doesn't)."""

    def __init__(self) -> None:
        self.routes: dict[tuple[str, str], RouteSpeed] = {}
        self.decisions = 0  # counts routing decisions that could check a route's speed (see Router)

    def record(self, backend: str, model: str, seconds: float, first_token_seconds: float | None, output_tokens: int,
               now: float) -> None:
        """One successful reply: ``seconds`` from calling the route to its last byte; ``first_token_seconds``
        to its first content (streams only)."""
        if seconds <= 0:
            return
        s = self.routes.setdefault((backend, model), RouteSpeed())
        if first_token_seconds is not None:
            first = max(first_token_seconds, 0.0)
            if output_tokens >= MIN_TIMED_TOKENS and seconds > first:
                s.per_token = _ewma(s.per_token, (seconds - first) / (output_tokens - 1))
        else:
            # Not streamed: the whole reply arrived at once. Take out the writing time at the route's known
            # speed (or a typical one) to estimate the wait before it.
            per = s.per_token if s.per_token is not None else DEFAULT_SECONDS_PER_TOKEN
            first = max(seconds - output_tokens * per, 0.0)
        s.first_token = _ewma(s.first_token, first)
        s.samples += 1
        s.last = now

    def get(self, backend: str, model: str) -> RouteSpeed | None:
        return self.routes.get((backend, model))

    def expected(self, backend: str, model: str, output_tokens: int) -> float | None:
        """Expected seconds for a reply on this route, or None until it has been measured."""
        s = self.routes.get((backend, model))
        return s.expected(output_tokens) if s and s.measured else None
