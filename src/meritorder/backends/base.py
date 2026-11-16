from __future__ import annotations

from dataclasses import dataclass, field
from typing import AsyncIterator, Awaitable, Callable

from ..config import BackendConfig, batch_api
from ..formats.sse import SSEEvent

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}
CAPACITY_STATUS = {429, 503, 529}  # this model is busy here; others on the same backend usually aren't
# Failures that say something about one model on a backend (its capacity, its deployment) rather than the
# backend as a whole (its credentials, its network). Only these skip just that model.
MODEL_SCOPED_STATUS = CAPACITY_STATUS | {404, 408, 409, 425, 500}
MAX_CAPACITY_WAIT = 300.0


class BackendError(Exception):
    def __init__(self, status: int, message: str, body: dict | None = None, retryable: bool = False,
                 retry_after: float | None = None, unreachable: bool = False):
        super().__init__(message)
        self.status = status
        self.message = message
        self.body = body
        self.retryable = retryable
        self.retry_after = retry_after  # seconds, from the provider's Retry-After header
        self.unreachable = unreachable  # no connection could be made (nothing was sent)


def describe(status: int) -> str:
    """A failure in words, for logs, alerts, and error messages."""
    return {429: "rate limited", 529: "overloaded", 503: "unavailable", 404: "model or deployment not found",
            401: "credentials rejected", 403: "access denied", 504: "timed out", 502: "unreachable or bad response",
            500: "server error", 408: "request timeout"}.get(status, f"HTTP {status}")


@dataclass
class Outage:
    """Why a backend, or one model on it, is being skipped and until when."""

    until: float
    status: int
    message: str
    capacity: bool  # busy rather than broken: still worth trying when nothing else can serve the request


@dataclass
class UpstreamResponse:
    status: int
    headers: dict[str, str] = field(default_factory=dict)
    body: dict | None = None
    events: AsyncIterator[SSEEvent] | None = None
    _close: Callable[[], Awaitable[None]] | None = None
    # A provider's Responses API asked to store the response (formats.responses_upstream.PROVIDER_CHAIN): the id it gave
    # it (known once a stream has started), and whether this turn went as the previous one's continuation there.
    provider_response_id: str | None = None
    provider_chained: bool = False

    async def aclose(self) -> None:
        if self._close is not None:
            await self._close()
            self._close = None


class Backend:
    """A place a model runs. Subclasses implement ``send``."""

    def __init__(self, cfg: BackendConfig):
        self.cfg = cfg
        self.name = cfg.name
        self.format = cfg.format
        self.kind = cfg.kind
        self._failures = 0
        self._open_until = 0.0
        self._model_failures: dict[str, int] = {}
        self._model_open: dict[str, Outage] = {}
        self.outage: Outage | None = None  # the whole backend's, while it's being skipped
        # Self-hosted deployments managed by the controller: set from published state.
        self.available = not (cfg.kind == "self_hosted" and cfg.serve is not None)
        self.draining = False
        self.unavailable_reason: str | None = None  # shown when a request has nowhere else to go
        self.live_cost_per_hour: float | None = None  # per replica, from the controller (SkyPilot catalog) when known
        self.replicas: int | None = None  # running replicas, from the controller
        self.reported_waiting: int | None = None  # requests waiting in vLLM at the controller's last sample
        self.deploy: dict | None = None  # the controller's published record for this deployment
        # Its replicas (affinity.Replica), from that record, and the ones this process couldn't connect to lately
        # (replica id -> skip until). With serve.replica_affinity, the router picks among them per session.
        self.replica_set: list = []
        self.unreachable: dict[str, float] = {}
        self.reports_cached = False  # a reply here has said how many prompt tokens came from the cache
        # The provider's batch API (batches/providers.py), or None and why not: batches are routed only where there's one.
        self.batch_api, self.batch_why = batch_api(cfg)

    def addressable(self, now: float) -> list:
        """Replicas a session can be sent to directly: replica affinity is on, and they're ready and reachable."""
        serve = self.cfg.serve
        if serve is None or not serve.replica_affinity:
            return []
        return [r for r in self.replica_set if r.ready and self.unreachable.get(r.id, 0.0) <= now]

    # Circuit breaker, per backend and per model on it.
    #   capacity (429/503/529): skip that model here for Retry-After, else 10 s doubling to 5 min
    #   model or deployment missing (404): skip that model here for 5 minutes, doubling to an hour while it stays missing
    #   credentials rejected (401/403): skip the backend for 1 minute doubling to 10
    #   anything else (5xx, timeouts, network): after 3 in a row, skip the backend for 5 s doubling to 60 s
    def healthy(self, now: float, model: str | None = None) -> bool:
        if now < self._open_until:
            return False
        if model is None:
            return True
        o = self._model_open.get(model)
        return o is None or now >= o.until

    def skipped(self, now: float, model: str | None = None) -> Outage | None:
        """The outage this backend (or model on it) is being skipped for right now, if any."""
        if now < self._open_until:
            return self.outage or Outage(self._open_until, 0, "repeated failures", capacity=False)
        o = self._model_open.get(model) if model else None
        return o if o and now < o.until else None

    def cooling_models(self, now: float) -> dict[str, Outage]:
        return {m: o for m, o in self._model_open.items() if now < o.until}

    def record_failure(self, now: float, model: str | None = None, status: int = 0, retry_after: float | None = None,
                       message: str = "") -> Outage | None:
        """Count a failure. Returns the outage if this one started skipping the backend or model (for alerts)."""
        scoped = model is not None and status in MODEL_SCOPED_STATUS
        n = (self._model_failures.get(model, 0) if scoped else self._failures) + 1
        if scoped:
            self._model_failures[model] = n
        else:
            self._failures = n
        if status in CAPACITY_STATUS or (retry_after and status in RETRYABLE_STATUS):
            wait = min(retry_after if retry_after else 10.0 * 2 ** (n - 1), MAX_CAPACITY_WAIT)
        elif scoped and status == 404:
            wait = min(300.0 * 2 ** (n - 1), 3600.0)  # not deployed there yet: check back less and less often
        elif status in (401, 403):
            wait = min(60.0 * 2 ** (n - 1), 600.0)
        elif n >= 3:
            wait = min(60.0, 5.0 * 2 ** (n - 3))
        else:
            return None
        was_healthy = self.healthy(now, model if scoped else None)
        outage = Outage(now + wait, status, message, capacity=status in CAPACITY_STATUS)
        if scoped:
            self._model_open[model] = outage
        else:
            self._open_until, self.outage = now + wait, outage
        return outage if was_healthy else None

    def record_success(self, model: str | None = None) -> bool:
        """A request worked. Returns True if the backend or model was being skipped (it has recovered)."""
        recovered = self._failures >= 3 or self.outage is not None or (model is not None and model in self._model_open)
        self._failures = 0
        self._open_until = 0.0
        self.outage = None
        if model is not None:
            self._model_failures.pop(model, None)
            self._model_open.pop(model, None)
        return recovered

    async def send(self, body: dict, stream: bool, headers: dict[str, str], base_url: str | None = None,
                   session_tag: str | None = None) -> UpstreamResponse:
        """``base_url``: one replica's address instead of the backend's (replica affinity). ``session_tag``: the
        session's id for the server (vLLM's ``X-Session-ID``, which it puts on its KV events)."""
        raise NotImplementedError

    async def count_tokens(self, body: dict, headers: dict[str, str]) -> dict | None:
        return None

    async def close(self) -> None:
        return None
