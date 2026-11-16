"""Billed cost: OpenRouter returns what it charged (usage.cost), which can differ from the catalog price because
it picks among providers. The ledger records that amount, and savings stay consistent with it."""

import copy

import httpx
import pytest
from conftest import TEST_CONFIG
from test_http_backend import Upstream, post, stream_response

from meritorder.app import build_gateway
from meritorder.config import BackendConfig, Price, build_config, uses_reported_cost
from meritorder.formats.translate import usage_from_openai
from meritorder.pricing import Usage, attribute
from meritorder.state import MemoryState
from meritorder.store import MemoryStore

PRICE = Price(input=1, output=2)  # $ per million tokens
BASELINE = Price(input=10, output=20)


def usage(**kw) -> Usage:
    return Usage(**{"input_tokens": 1000, "output_tokens": 500, **kw})


def balanced(c) -> bool:
    return c.baseline - c.actual == pytest.approx(c.savings_routing + c.savings_caching + c.savings_self_hosting)


def test_billed_amount_replaces_the_catalog_price_and_savings_still_add_up():
    c = attribute(usage(), PRICE, BASELINE, proxy_cache=False, billed=0.003)  # catalog says 0.002
    assert c.actual == c.served_api == 0.003 and c.list_price == pytest.approx(0.003)
    assert c.baseline == pytest.approx(0.02) and c.savings_routing == pytest.approx(0.017) and c.savings_caching == 0
    assert balanced(c)


def test_a_pricier_provider_is_not_counted_as_routing_savings_or_losses_for_a_named_model():
    c = attribute(usage(), PRICE, None, proxy_cache=False, billed=0.005)
    assert c.actual == 0.005 and c.baseline == pytest.approx(0.005) and c.savings_routing == pytest.approx(0)
    assert balanced(c)


def test_caching_savings_scale_with_the_billed_amount():
    u = usage(input_tokens=200, cache_read_tokens=800)
    catalog = attribute(u, PRICE, BASELINE, proxy_cache=True)
    c = attribute(u, PRICE, BASELINE, proxy_cache=True, billed=catalog.actual * 2)
    assert c.actual == pytest.approx(catalog.actual * 2)
    assert c.savings_caching == pytest.approx(catalog.savings_caching * 2) and c.savings_caching > 0
    assert balanced(c)


def test_billed_amount_with_no_catalog_price_and_free_models():
    c = attribute(usage(), Price(), None, proxy_cache=False, billed=0.004)
    assert c.actual == c.served_api == c.list_price == c.baseline == 0.004 and balanced(c)
    free = attribute(usage(), PRICE, BASELINE, proxy_cache=False, billed=0.0)
    assert free.actual == 0 and free.list_price == 0 and balanced(free)


def test_self_hosted_chargeback_ignores_a_billed_amount():
    c = attribute(usage(), PRICE, BASELINE, proxy_cache=False, self_hosted_cost=0.0007, billed=9.0)
    assert c.actual == 0.0007 and c.served_api == pytest.approx(0.002) and balanced(c)


def test_parsing_openrouter_usage_cost():
    assert usage_from_openai({"prompt_tokens": 10, "completion_tokens": 2, "cost": 8.87e-05}).billed_usd == 8.87e-05
    assert usage_from_openai({"prompt_tokens": 10, "completion_tokens": 2}).billed_usd is None
    # Your own provider key: usage.cost is only OpenRouter's fee; the provider billed your key separately.
    byok = {"cost": 0.0001, "is_byok": True, "cost_details": {"upstream_inference_cost": 0.002}}
    assert usage_from_openai(byok).billed_usd == pytest.approx(0.0021)
    for junk in (True, "0.1", -1, None, {"x": 1}):
        assert usage_from_openai({"cost": junk}).billed_usd is None
    u = Usage()
    u.merge(usage_from_openai({"prompt_tokens": 5}))
    u.merge(usage_from_openai({"prompt_tokens": 5, "completion_tokens": 3, "cost": 0.01}))
    u.merge(usage_from_openai({"prompt_tokens": 5}))
    assert u.billed_usd == 0.01 and u.output_tokens == 3


@pytest.mark.parametrize(("kind", "url", "setting", "expected"), [
    ("api", "https://openrouter.ai/api/v1", "auto", True),
    ("api", "https://api.deepseek.com/v1", "auto", False),
    ("self_hosted", "http://openrouter.ai.internal:8000/v1", "auto", False),
    ("api", "https://openrouter.ai/api/v1", "catalog", False),
    ("api", "https://my-litellm.example/v1", "reported", True),
])
def test_which_backends_use_the_reported_cost(kind, url, setting, expected):
    assert uses_reported_cost(BackendConfig(kind=kind, format="openai", base_url=url, billed_cost=setting)) is expected


def gateway(monkeypatch, upstream: Upstream):
    monkeypatch.setenv("MO_TEST_OR", "key-or")
    raw = copy.deepcopy(TEST_CONFIG)
    raw["baseline_model"] = "base"
    raw["backends"] = {
        "or": {"kind": "api", "format": "openai", "base_url": "https://openrouter.ai/api/v1",
               "auth": {"type": "api_key", "value_env": "MO_TEST_OR"}},
        "plain": {"kind": "api", "format": "openai", "base_url": "https://plain.example/v1",
                  "auth": {"type": "api_key", "value_env": "MO_TEST_OR"}},
    }
    raw["models"] = {
        "m": {"quality": {"coding": 4, "general": 4}, "routes": [{"backend": "or", "model": "m-or", "price": {"input": 1, "output": 2}}]},
        "p": {"quality": {"coding": 4, "general": 4}, "routes": [{"backend": "plain", "model": "p-1", "price": {"input": 1, "output": 2}}]},
        "base": {"quality": {"coding": 5, "general": 5}, "routes": [{"backend": "plain", "model": "b-1", "price": {"input": 10, "output": 20}}]},
    }
    gw = build_gateway(build_config(raw, quiet=True), store=MemoryStore(), state=MemoryState())
    for name in ("or", "plain"):
        gw.backends[name].client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    return gw


def reply(cost: float, model: str = "m-or") -> httpx.Response:
    return httpx.Response(200, json={
        "id": "gen-1", "object": "chat.completion", "created": 1, "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500, "cost": cost}})


async def test_ledger_and_cost_header_use_what_openrouter_billed(monkeypatch):
    gw = gateway(monkeypatch, Upstream(reply(0.003)))
    r = await post(gw, {"model": "m", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200 and float(r.headers["x-meritorder-cost-usd"]) == 0.003
    await gw.drain()
    row = (await gw.store.recent(1))[0]
    assert row["actual_cost"] == pytest.approx(0.003) and row["served_api_cost"] == pytest.approx(0.003)
    assert row["baseline_cost"] == pytest.approx(0.02) and row["savings_routing"] == pytest.approx(0.017)


async def test_a_backend_that_is_not_trusted_for_billing_is_priced_from_the_catalog(monkeypatch):
    gw = gateway(monkeypatch, Upstream(reply(99.0, "p-1")))
    r = await post(gw, {"model": "p", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    await gw.drain()
    assert (await gw.store.recent(1))[0]["actual_cost"] == pytest.approx(0.002)


async def test_streamed_anthropic_client_on_openrouter_records_the_billed_amount(monkeypatch):
    sse = (b'data: {"id":"gen-2","choices":[{"index":0,"delta":{"role":"assistant","content":"hi"}}]}\n\n'
           b'data: {"id":"gen-2","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
           b'data: {"id":"gen-2","choices":[],"usage":{"prompt_tokens":1000,"completion_tokens":500,"cost":0.0031}}\n\n'
           b"data: [DONE]\n\n")
    gw = gateway(monkeypatch, Upstream(lambda req: stream_response(sse)))
    r = await post(gw, {"model": "m", "max_tokens": 50, "stream": True, "messages": [{"role": "user", "content": "hi"}]},
                   path="/v1/messages")
    assert r.status_code == 200
    await gw.drain()
    row = (await gw.store.recent(1))[0]
    assert row["actual_cost"] == pytest.approx(0.0031) and row["stream"] is True
