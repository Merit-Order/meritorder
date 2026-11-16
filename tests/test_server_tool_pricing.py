"""Server-side tool fees (Anthropic web search) count toward cost, budgets, and the baseline."""

from meritorder.config import Price
from meritorder.formats.translate import UsageTracker, usage_from_anthropic
from meritorder.formats.sse import SSEEvent
from meritorder.pricing import Usage, attribute, cost


def test_web_searches_are_charged_per_search():
    u = usage_from_anthropic({"input_tokens": 1000, "output_tokens": 100, "server_tool_use": {"web_search_requests": 3}})
    assert u.web_search_requests == 3
    p = Price(input=3, output=15)
    assert abs(cost(p, u) - (1000 * 3 + 100 * 15) / 1e6 - 0.03) < 1e-12
    c = attribute(u, p, Price(input=3, output=15), proxy_cache=False)
    assert abs(c.actual - cost(p, u)) < 1e-12 and abs(c.savings_routing) < 1e-12  # same searches in the baseline
    assert cost(p, Usage(input_tokens=10)) == 30 / 1e6


def test_web_searches_read_from_the_stream():
    t = UsageTracker("anthropic")
    t.feed(SSEEvent(event="message_start", data='{"type":"message_start","message":{"usage":{"input_tokens":50,"output_tokens":1}}}'))
    t.feed(SSEEvent(event="message_delta",
                    data='{"type":"message_delta","usage":{"output_tokens":40,"server_tool_use":{"web_search_requests":2}}}'))
    assert t.usage.web_search_requests == 2 and t.usage.input_tokens == 50 and t.usage.output_tokens == 40


def test_anthropic_beta_allowlist():
    from meritorder.backends.http import AnthropicBackend
    from meritorder.backends.auth import BackendAuthProvider
    from meritorder.config import BackendConfig

    sent = "interleaved-thinking-2025-05-14, context-1m-2025-08-07,fine-grained-tool-streaming-2025-05-14"
    open_ = AnthropicBackend(BackendConfig(name="a", format="anthropic", base_url="https://x"), BackendAuthProvider())
    assert open_._headers({"anthropic-beta": sent})["anthropic-beta"] == sent
    cfg = BackendConfig(name="a", format="anthropic", base_url="https://x", anthropic_betas=["interleaved-thinking", "fine-grained"])
    b = AnthropicBackend(cfg, BackendAuthProvider())
    assert b._headers({"anthropic-beta": sent})["anthropic-beta"] == "interleaved-thinking-2025-05-14,fine-grained-tool-streaming-2025-05-14"
    cfg.anthropic_betas = []
    assert "anthropic-beta" not in b._headers({"anthropic-beta": sent})
