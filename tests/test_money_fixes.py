"""Money: an unknown price is never served as free, a refused attempt doesn't count the baseline twice, and the savings
counterfactuals use the prices that would really have applied (long-prompt tiers, a provider's own batch discount, no
response-cache savings for evals)."""

from __future__ import annotations

import copy

import httpx
import pytest
import yaml
from conftest import TEST_CONFIG, auth, ledger

from meritorder import alerts
from meritorder.app import build_gateway, create_app
from meritorder.config import ModelConfig, Price, PriceTier, Route, load_config, needs_price
from meritorder.pricing import Usage, cost, removed_cost
from meritorder.proxy import api_batch_price, known_api_price
from meritorder.state import MemoryState
from meritorder.store import MemoryStore, month_start


def identity(row: dict) -> float:
    return row["baseline_cost"] - row["actual_cost"] - row["savings_total"]


async def test_a_refused_attempt_counts_the_baseline_once(client, gateway):
    """One client request, two ledger rows (the refused attempt and the answer): the baseline is the answer's alone, and
    the refusal's cost is a loss to routing, so baseline - actual = the sum of causes per row and for the request."""
    body = {"model": "auto", "messages": [{"role": "user", "content": "explain this exploit [refuse:small-1]"}]}
    r = await client.post("/v1/chat/completions", headers={**auth(), "x-route-floor": "general:3"}, json=body)
    assert r.status_code == 200 and r.headers["x-meritorder-refused-by"] == "small"
    rows = await ledger(gateway)
    assert len(rows) == 2
    refused = next(x for x in rows if x["error"] and "refused" in x["error"])
    served = next(x for x in rows if not x["error"])
    assert refused["actual_cost"] > 0 and refused["baseline_cost"] == 0
    assert refused["savings_routing"] == pytest.approx(-refused["actual_cost"])
    spent = refused["actual_cost"] + served["actual_cost"]
    without = served["baseline_cost"]
    assert refused["baseline_cost"] + served["baseline_cost"] == pytest.approx(without)
    assert refused["savings_total"] + served["savings_total"] == pytest.approx(without - spent)
    for row in rows:
        assert identity(row) == pytest.approx(0, abs=1e-12)


def unpriced_gateway(tmp_path, **models):
    raw = copy.deepcopy(TEST_CONFIG)
    raw["models"].update(models)
    raw["teams"]["default"]["user_budget_usd_month"] = 0.000001
    path = tmp_path / "meritorder.yaml"
    path.write_text(yaml.safe_dump(raw))
    gw = build_gateway(load_config(path), store=MemoryStore(), state=MemoryState())
    return gw, httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(gateway=gw)), base_url="http://test")


async def test_a_named_model_with_no_price_is_refused_not_served_free(tmp_path):
    """A model whose only route has no price (the live list hasn't been fetched, or doesn't list it) isn't served at $0,
    where no budget would ever bind: the request gets a 503 that says why."""
    gw, client = unpriced_gateway(tmp_path, unpriced={"quality": {"general": 3}, "routes": [{"backend": "mock-openai", "model": "unpriced-1"}]})
    async with client:
        r = await client.post("/v1/chat/completions", headers=auth(), json={"model": "unpriced", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 503
    assert "no price known for this route yet; set `price:` or wait for the price list" in r.json()["error"]["message"]
    await gw.drain()
    assert await ledger(gw) == []


async def test_a_named_model_goes_to_its_priced_route_and_never_its_unpriced_one(tmp_path):
    gw, client = unpriced_gateway(tmp_path, half={"quality": {"general": 3}, "routes": [
        {"backend": "mock-anthropic", "model": "half-free"},  # no price
        {"backend": "mock-openai", "model": "half-1", "price": {"input": 1, "output": 2}}]})
    async with client:
        for _ in range(3):
            r = await client.post("/v1/chat/completions", headers=auth(), json={"model": "half", "messages": [{"role": "user", "content": "hello " * 50}]})
            assert r.status_code == 200 and r.headers["x-meritorder-backend"] == "mock-openai"
    await gw.drain()
    assert await gw.store.spend_since("user_id", "alice", month_start()) > 0


async def test_embeddings_on_a_route_with_no_price_are_refused(tmp_path):
    gw, client = unpriced_gateway(tmp_path, **{"embed-unpriced": {"kind": "embedding", "routes": [{"backend": "mock-openai", "model": "e-1"}]}})
    async with client:
        r = await client.post("/v1/embeddings", headers=auth(), json={"model": "embed-unpriced", "input": "hello"})
    assert r.status_code == 503 and "no price known for this route yet" in r.text
    await gw.drain()
    assert await ledger(gw) == []


async def test_a_batch_for_a_model_with_no_price_is_refused(tmp_path):
    gw, client = unpriced_gateway(tmp_path, unpriced={"quality": {"general": 3}, "routes": [{"backend": "mock-openai", "model": "unpriced-1"}]})
    async with client:
        r = await client.post("/v1/messages/batches", headers=auth(),
                              json={"requests": [{"custom_id": "a", "params": {"model": "unpriced", "max_tokens": 10,
                                                                               "messages": [{"role": "user", "content": "hi"}]}}]})
    assert r.status_code == 400 and "no price known for this route yet" in r.text


def test_only_routes_charged_per_token_need_a_price():
    """GPUs are charged by their time, so a named request may use one without a price; an API or a mock may not."""
    bare = Route(backend="x", model="m")
    assert needs_price(bare, "api") and needs_price(bare, "mock") and not needs_price(bare, "self_hosted")
    assert not needs_price(Route(backend="x", model="m", price=Price(input=1)), "api")


async def test_each_route_with_no_price_raises_a_critical_alert_once_a_day(tmp_path):
    gw, client = unpriced_gateway(tmp_path, unpriced={"quality": {"general": 3}, "routes": [{"backend": "mock-openai", "model": "unpriced-1"}]})
    await client.aclose()
    state = MemoryState()
    assert await alerts.check_unpriced(gw.cfg, state) == ["unpriced@mock-openai"]
    assert await alerts.check_unpriced(gw.cfg, state) == []
    alert = (await state.recent_alerts())[0]
    assert alert["level"] == "critical" and "no price known" in alert["text"] and alert["page"] == "models"


async def test_alerts_are_sent_once_however_many_workers_check_at_once():
    import asyncio

    state = MemoryState()
    got = await asyncio.gather(*[alerts._once(state, "alert:x", 60) for _ in range(20)])
    assert got.count(True) == 1


async def test_response_cache_hits_for_eval_traffic_claim_no_savings(client, gateway):
    """Evals and judged labels get no routing or caching savings; an answer from the response cache is caching too."""
    body = {"model": "small", "messages": [{"role": "user", "content": "what is 2 + 2?"}]}
    for tags in ("eval=run-1", "project=web"):
        headers = {**auth(), "x-meritorder-cache": "on", "x-meritorder-tags": tags}
        first = await client.post("/v1/chat/completions", headers=headers, json={**body, "temperature": 0.1 if "eval" in tags else 0.2})
        assert first.status_code == 200
        await gateway.drain()
        again = await client.post("/v1/chat/completions", headers=headers, json={**body, "temperature": 0.1 if "eval" in tags else 0.2})
        assert again.headers["x-meritorder-backend"] == "response-cache"
    await gateway.drain()
    hits = {(r["tags"] or {}).get("eval", "people"): r for r in await ledger(gateway) if r["backend"] == "response-cache"}
    assert hits["run-1"]["savings_response_cache"] == 0 and hits["run-1"]["baseline_cost"] == 0
    assert hits["people"]["savings_response_cache"] > 0


def test_compression_savings_are_at_the_long_prompt_tier_the_tokens_would_have_made():
    """Tokens taken out of a prompt that would otherwise have crossed a long-prompt tier would have cost the tier's price,
    on every token of the prompt."""
    price = Price(input=1.0, output=5.0, tiers=[PriceTier(above=100_000, input=2.0, output=10.0)])
    sent = Usage(input_tokens=90_000, output_tokens=1_000)
    saved = removed_cost(price, sent, 20_000, 0)
    assert saved == pytest.approx(cost(price, Usage(input_tokens=110_000, output_tokens=1_000)) - cost(price, sent))
    assert saved > 20_000 * price.input / 1e6  # more than the removed tokens at the base input price
    assert removed_cost(price, sent, 0, 0) == 0


def test_the_gpu_batch_counterfactual_is_the_providers_own_batch_price(tmp_path):
    """Batch work on a GPU is measured against what a provider's batch API would have charged: xAI takes 20% off, not
    the usual half; a listed batch price wins over any discount."""
    raw = copy.deepcopy(TEST_CONFIG)
    raw["backends"]["xai"] = {"format": "openai", "base_url": "https://api.x.ai/v1", "auth": {"type": "api_key", "value": "k"}}
    raw["models"]["grok"] = {"quality": {"general": 3}, "routes": [{"backend": "xai", "model": "grok-1", "price": {"input": 1.0, "output": 4.0}}]}
    raw["models"]["listed"] = {"quality": {"general": 3}, "routes": [
        {"backend": "xai", "model": "l-1", "price": {"input": 1.0, "output": 4.0, "batch": {"input": 0.3, "output": 1.0}}}]}
    path = tmp_path / "meritorder.yaml"
    path.write_text(yaml.safe_dump(raw))
    cfg = load_config(path)
    grok = api_batch_price(cfg, cfg.model("grok"))
    assert (grok.input, grok.output) == pytest.approx((0.8, 3.2))
    listed = api_batch_price(cfg, cfg.model("listed"))
    assert (listed.input, listed.output) == pytest.approx((0.3, 1.0))
    gpu_only = ModelConfig(name="g", routes=[Route(backend="gpu", model="g", price=Price(input=1.0, output=2.0))])
    assert api_batch_price(cfg, gpu_only).input == pytest.approx(0.5)  # no API route: the usual half


def test_a_model_with_no_price_has_no_counterfactual(tmp_path):
    path = tmp_path / "meritorder.yaml"
    path.write_text(yaml.safe_dump(TEST_CONFIG))
    cfg = load_config(path)
    assert known_api_price(cfg, ModelConfig(name="n", routes=[Route(backend="mock-openai", model="n")])) is None
    gpu = ModelConfig(name="g", routes=[Route(backend="mock-openai", model="g"), Route(backend="gpu", model="g", price=Price(input=0.1))])
    assert known_api_price(cfg, gpu).input == 0.1  # a priced route of any kind before none
