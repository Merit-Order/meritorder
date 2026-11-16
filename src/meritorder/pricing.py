"""Token usage, cost, and savings attribution.

Savings are split by cause and computed so they add up exactly:

    baseline_cost - actual_cost = routing + caching + self_hosting + batch

* routing       = baseline model's cost - served model's cost, both without proxy-added caching
* caching       = served model without proxy-added caching - served model as actually billed
* self_hosting  = served model's API cost - our GPU chargeback (0 for API backends); for batch work on a GPU, its
                  batch API price - our GPU chargeback
* batch         = served model's API cost - the same tokens at its batch API price (0 for live requests)

Caching the client set up itself would have happened without Merit Order, so it
is counted in every term alike and never shows up as savings.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from .config import Price


@dataclass
class Usage:
    input_tokens: int = 0  # uncached input
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    # The part of cache_write_tokens Claude kept for an hour (a "ttl": "1h" marker), which costs more than a 5-minute
    # write. None: the backend didn't say (see proxy.Gateway.costs for how that's settled).
    cache_write_1h_tokens: int | None = None
    web_search_requests: int = 0  # Anthropic server-side web searches, billed per search
    billed_usd: float | None = None  # what the backend says it charged, when it says (OpenRouter)
    # The part of output_tokens that was reasoning, where the backend says (OpenAI's reasoning_tokens, Claude's
    # thinking_tokens, Gemini's thoughts). Already in output_tokens: it's reported, never priced on its own.
    reasoning_tokens: int = 0

    @property
    def prompt_tokens(self) -> int:
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.output_tokens

    def merge(self, other: "Usage") -> None:
        """Take the larger of each field (stream events repeat cumulative counts)."""
        self.input_tokens = max(self.input_tokens, other.input_tokens)
        self.cache_read_tokens = max(self.cache_read_tokens, other.cache_read_tokens)
        self.cache_write_tokens = max(self.cache_write_tokens, other.cache_write_tokens)
        self.output_tokens = max(self.output_tokens, other.output_tokens)
        if other.cache_write_1h_tokens is not None:
            self.cache_write_1h_tokens = max(self.cache_write_1h_tokens or 0, other.cache_write_1h_tokens)
        self.web_search_requests = max(self.web_search_requests, other.web_search_requests)
        self.reasoning_tokens = max(self.reasoning_tokens, other.reasoning_tokens)
        if other.billed_usd is not None:
            self.billed_usd = max(self.billed_usd or 0.0, other.billed_usd)


def _server_tools(price: Price, u: Usage) -> float:
    return u.web_search_requests * price.web_search_per_1k / 1000


def cost(price: Price, u: Usage) -> float:
    price = price.at(u.prompt_tokens)  # a long prompt can cost more per token (Claude Haiku 5.5 above 100K, GPT-6.x above 272K)
    hour = min(u.cache_write_1h_tokens or 0, u.cache_write_tokens)
    return (
        u.input_tokens * price.input
        + u.cache_read_tokens * price.read()
        + (u.cache_write_tokens - hour) * price.write()
        + hour * price.write_1h()
        + u.output_tokens * price.output
    ) / 1e6 + _server_tools(price, u)


def compression_savings(price: Price, new_tokens: int, history_tokens: int) -> float:
    """What tokens compression removed would have cost: new ones as input, ones from earlier turns (mostly cached by
    now) as cache reads. The gateway's ledger and the demo data both use this."""
    return (new_tokens * price.input + history_tokens * price.read()) / 1e6


def uncached_cost(price: Price, u: Usage) -> float:
    price = price.at(u.prompt_tokens)
    return (u.prompt_tokens * price.input + u.output_tokens * price.output) / 1e6 + _server_tools(price, u)


@dataclass
class Costs:
    actual: float
    served_api: float
    baseline: float
    list_price: float
    savings_routing: float
    savings_caching: float
    savings_self_hosting: float
    savings_compression: float = 0.0  # what the rules took out of tool results
    savings_response_cache: float = 0.0
    savings_compression_model: float = 0.0  # what model-based compression took out on top of the rules
    savings_batch: float = 0.0  # what the provider's batch price took off its live price

    @property
    def savings_total(self) -> float:
        return (self.savings_routing + self.savings_caching + self.savings_self_hosting + self.savings_batch
                + self.savings_compression + self.savings_compression_model + self.savings_response_cache)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["savings_total"] = self.savings_total
        return d

    def extra_attempt(self) -> "Costs":
        """These costs as an attempt the client never got (a refusal, retried on another model): what it cost, with no
        baseline of its own. Without Merit Order the person would have sent the prompt to the baseline model once, and
        the answer's row carries that baseline; counting it here too would count it twice. Its whole cost is a loss to
        routing, so baseline - actual = the sum of the causes still holds on the row and on the request."""
        return Costs(actual=self.actual, served_api=self.served_api, baseline=0.0, list_price=self.list_price,
                     savings_routing=-self.actual, savings_caching=0.0, savings_self_hosting=0.0)


def removed_cost(price: Price, u: Usage, new_tokens: int, history_tokens: int) -> float:
    """What tokens taken out of the prompt would have cost on top of ``u``: new ones as input, older ones as cache reads.
    At the price for the prompt they would have made, so a prompt that compression kept under a long-prompt tier is
    credited with the whole tier's difference."""
    if not new_tokens and not history_tokens:
        return 0.0
    more = replace(u, input_tokens=u.input_tokens + new_tokens, cache_read_tokens=u.cache_read_tokens + history_tokens)
    return cost(price, more) - cost(price, u)


def chargeback(gpu_cost_per_hour: float, capacity_tokens_per_second: float, u: Usage) -> float:
    """Fair share of GPU time for this request's tokens."""
    if capacity_tokens_per_second <= 0:
        return 0.0
    seconds = u.total_tokens / capacity_tokens_per_second
    return gpu_cost_per_hour * seconds / 3600.0


def attribute(
    usage: Usage,
    served_price: Price,
    baseline_price: Price | None,
    proxy_cache: bool,
    self_hosted_cost: float | None = None,
    billed: float | None = None,
    batch_price: Price | None = None,
) -> Costs:
    """``billed``: what the backend reported charging. It replaces the catalog price of the served model,
    and the served model's counterfactuals are scaled by the same factor, so a provider that charged more
    or less than the catalog says doesn't show up as routing or caching savings.

    ``batch_price``: the request went through the provider's batch API at this price. Everything up to the served model's
    API cost is counted as for a live request; what the batch price takes off that is the batch cause. With
    ``self_hosted_cost`` too, it was batch work on a GPU (batches/gpu.py): the batch price is what a provider's batch API
    would have charged, so the batch cause is the same, and what the GPU took off the batch price is self-hosting's."""
    served_api = cost(served_price, usage)
    served_no_proxy_cache = uncached_cost(served_price, usage) if proxy_cache else served_api
    list_price = uncached_cost(served_price, usage)
    if billed is not None and self_hosted_cost is None:
        if served_api > 0:
            scale = billed / served_api
            served_no_proxy_cache *= scale
            list_price *= scale
        else:
            served_no_proxy_cache = list_price = billed
        served_api = billed
    if baseline_price is None:
        baseline = served_no_proxy_cache
    elif proxy_cache:
        baseline = uncached_cost(baseline_price, usage)
    else:
        baseline = cost(baseline_price, usage)
    batched = cost(batch_price, usage) if batch_price is not None else None
    actual = self_hosted_cost if self_hosted_cost is not None else batched if batched is not None else served_api
    if batched is None:  # live: an API's (0) or a GPU's
        self_hosting, batch_cut = served_api - actual, 0.0
    elif self_hosted_cost is None:  # a provider's batch API
        self_hosting, batch_cut = 0.0, served_api - batched
    else:  # batch work on a GPU
        self_hosting, batch_cut = batched - actual, served_api - batched
    return Costs(
        actual=actual,
        served_api=served_api,
        baseline=baseline,
        list_price=list_price,
        savings_routing=baseline - served_no_proxy_cache,
        savings_caching=served_no_proxy_cache - served_api,
        savings_self_hosting=self_hosting,
        savings_batch=batch_cut,
    )
