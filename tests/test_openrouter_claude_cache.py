"""Claude through OpenRouter caches only when asked (found live: three turns of a 17.5k-token prompt paid full
price each time). The gateway turns on automatic caching there, and the router prices cache writes on every route."""

import copy
import json

import httpx
from conftest import TEST_CONFIG
from test_http_backend import OPENAI_OK, Upstream, post

from meritorder.app import build_gateway
from meritorder.backends.mock import MockBackend
from meritorder.cache_shaping import shape_openai
from meritorder.config import BackendConfig, Price, build_config
from meritorder.formats.translate import anthropic_to_openai_request
from meritorder.router import Router
from meritorder.state import MemoryState
from meritorder.store import MemoryStore

OR = BackendConfig(name="or", format="openai", base_url="https://openrouter.ai/api/v1")


def test_claude_on_openrouter_gets_the_automatic_caching_marker():
    body = {"model": "anthropic/claude-haiku-5.5", "messages": [{"role": "user", "content": "hi"}]}
    assert shape_openai(body, OR, "sid") is True and body["cache_control"] == {"type": "ephemeral"}
    # A backend set to an hour: 5 minutes on a session's first turn, an hour once it has come back.
    first, later = {"model": "anthropic/claude-sonnet-5.5", "messages": []}, {"model": "anthropic/claude-sonnet-5.5", "messages": []}
    assert shape_openai(first, OR.model_copy(update={"cache_ttl_seconds": 3600}), "sid") is True
    assert first["cache_control"] == {"type": "ephemeral"}
    assert shape_openai(later, OR.model_copy(update={"cache_ttl_seconds": 3600}), "sid", later_turn=True) is True
    assert later["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


def test_other_models_and_providers_and_client_markers_are_left_alone():
    for backend, model in ((OR, "moonshotai/kimi-k3.1"), (OR.model_copy(update={"base_url": "https://api.openai.com/v1"}), "anthropic/x")):
        body = {"model": model, "messages": []}
        assert shape_openai(body, backend, "sid") is False and "cache_control" not in body
    own = {"model": "anthropic/claude-haiku-5.5", "messages": [{"role": "system", "content": [
        {"type": "text", "text": "rules", "cache_control": {"type": "ephemeral"}}]}]}
    assert shape_openai(own, OR, "sid") is False and "cache_control" not in own


def gateway(monkeypatch, upstream: Upstream):
    monkeypatch.setenv("MO_TEST_OR", "key-or")
    raw = copy.deepcopy(TEST_CONFIG)
    raw["backends"] = {"or": {"kind": "api", "format": "openai", "base_url": "https://openrouter.ai/api/v1",
                              "auth": {"type": "api_key", "value_env": "MO_TEST_OR"}}}
    raw["models"] = {"c": {"quality": {"coding": 4, "general": 4}, "routes": [
        {"backend": "or", "model": "anthropic/claude-x", "price": {"input": 1, "output": 5, "cache_read": 0.1, "cache_write": 1.25}}]}}
    raw["baseline_model"] = "c"
    gw = build_gateway(build_config(raw, quiet=True), store=MemoryStore(), state=MemoryState())
    gw.backends["or"].client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    return gw


async def test_anthropic_client_on_openrouter_claude_is_cached_and_credited(monkeypatch):
    up = Upstream(httpx.Response(200, json={**OPENAI_OK, "usage": {
        "prompt_tokens": 1000, "completion_tokens": 2, "prompt_tokens_details": {"cached_tokens": 900}}}))
    gw = gateway(monkeypatch, up)
    r = await post(gw, {"model": "c", "max_tokens": 20, "system": "rules", "messages": [{"role": "user", "content": "hi"}]},
                   path="/v1/messages")
    assert r.status_code == 200 and json.loads(up.requests[0].content)["cache_control"] == {"type": "ephemeral"}
    await gw.drain()
    row = (await gw.store.recent(1))[0]
    assert row["proxy_cache"] is True and row["cache_read_tokens"] == 900 and row["savings_caching"] > 0


async def test_claude_code_markers_reach_openrouter_claude_where_they_were_put(monkeypatch):
    up = Upstream(httpx.Response(200, json=OPENAI_OK))
    gw = gateway(monkeypatch, up)
    hour = {"type": "ephemeral", "ttl": "1h"}
    body = {"model": "c", "max_tokens": 20, "messages": [
        {"role": "user", "content": [{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}]},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "f", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "out", "cache_control": hour}]}],
        "system": [{"type": "text", "text": "rules", "cache_control": hour}, {"type": "text", "text": "more"}]}
    r = await post(gw, body, path="/v1/messages")
    sent = json.loads(up.requests[0].content)
    assert r.status_code == 200 and "cache_control" not in sent  # no gateway marker of its own, with its own lifetime
    msgs = sent["messages"]
    assert msgs[0]["content"] == [{"type": "text", "text": "rules", "cache_control": hour}, {"type": "text", "text": "more"}]
    assert msgs[1]["content"] == [{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}]
    assert msgs[3] == {"role": "tool", "tool_call_id": "t1", "content": [{"type": "text", "text": "out", "cache_control": hour}]}
    await gw.drain()
    assert (await gw.store.recent(1))[0]["proxy_cache"] is False  # the client asked for caching itself


def test_cache_markers_are_left_out_where_the_backend_does_not_read_them():
    body = {"messages": [{"role": "user", "content": [{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}]}],
            "system": [{"type": "text", "text": "a", "cache_control": {"type": "ephemeral"}}, {"type": "text", "text": "b"}]}
    out = anthropic_to_openai_request(body)
    assert out["messages"] == [{"role": "system", "content": "a\n\nb"}, {"role": "user", "content": "hi"}]


def test_router_charges_cache_writes_only_where_caching_is_explicit(cfg):
    router = Router(cfg, {})
    backend = MockBackend(BackendConfig(name="or", kind="api", format="openai", base_url="https://openrouter.ai/api/v1"))
    plain = router.estimate(Price(input=1, output=5), backend, 20_000, None, warm=False, turns=3, route_model="anthropic/claude-x")
    premium = router.estimate(Price(input=1, output=5, cache_write=1.25), backend, 20_000, None, warm=False, turns=3, route_model="anthropic/claude-x")
    assert premium > plain  # Claude through OpenRouter caches when asked, and the write costs 1.25x
    # Gemini, GPT and most open models cache automatically: new tokens cost the input price, whatever a price list
    # says about writes (OpenRouter lists Gemini 4 Carbon's at $0.19 against $2 input).
    listed = router.estimate(Price(input=2, output=10, cache_read=0.1, cache_write=0.1875), backend, 20_000, None, warm=False, turns=3, route_model="google/gemini-4-carbon")
    assert listed == router.estimate(Price(input=2, output=10, cache_read=0.1), backend, 20_000, None, warm=False, turns=3, route_model="google/gemini-4-carbon")
    direct = MockBackend(BackendConfig(name="an", kind="api", format="anthropic", base_url="https://api.anthropic.com"))
    assert router.estimate(Price(input=1, output=5, cache_write=1.25), direct, 20_000, None, warm=False, turns=3) > \
        router.estimate(Price(input=1, output=5), direct, 20_000, None, warm=False, turns=3)
