"""Live prices: from OpenRouter's catalog and the providers' list prices (LiteLLM's list), long-prompt tiers, refreshed
by the worker, with alerts when prices change, the lists disagree, or one can't be fetched."""

import copy

import httpx
import pytest
from conftest import TEST_CONFIG

from meritorder import catalog, live_prices
from meritorder.config import BackendConfig, Price, PriceTier, build_config
from meritorder.pricing import Usage, cost, uncached_cost
from meritorder.state import MemoryState
from meritorder.store import MemoryStore

HAIKU = Price(input=0.1, output=0.5, cache_read=0.01, cache_write=0.125,
              tiers=[PriceTier(above=100_000, input=0.5, output=2.5, cache_read=0.05, cache_write=0.625)])


def test_long_prompts_use_their_tier():
    assert HAIKU.at(100_000) is HAIKU and HAIKU.at(0).input == 0.1
    long = HAIKU.at(100_001)
    assert (long.input, long.output, long.read(), long.write(), long.tiers) == (0.5, 2.5, 0.05, 0.625, [])
    stepped = Price(input=2, output=10, cache_read=0.2, tiers=[PriceTier(above=272_000, input=4, output=15), PriceTier(above=128_000, input=3, output=12)])
    assert stepped.at(200_000).input == 3 and stepped.at(300_000).input == 4
    assert stepped.at(300_000).read() == pytest.approx(0.4)  # no tier cache price: the base one, scaled with input
    assert Price(input=1, output=2, tiers=[PriceTier(above=10, input=2, output=4)]).at(11).read() == pytest.approx(0.2)

    short = Usage(input_tokens=50_000, output_tokens=1000)
    assert cost(HAIKU, short) == pytest.approx((50_000 * 0.1 + 1000 * 0.5) / 1e6)
    big = Usage(input_tokens=20_000, cache_read_tokens=130_000, output_tokens=1000)  # the cache counts toward the prompt's length
    assert cost(HAIKU, big) == pytest.approx((20_000 * 0.5 + 130_000 * 0.05 + 1000 * 2.5) / 1e6)
    assert uncached_cost(HAIKU, big) == pytest.approx((150_000 * 0.5 + 1000 * 2.5) / 1e6)


def test_each_backend_knows_where_its_prices_come_from():
    def b(**kw):
        return BackendConfig(name="x", **{"format": "openai", **kw})
    assert catalog.price_source(b(base_url="https://openrouter.ai/api/v1")) == ("openrouter", "")
    assert catalog.price_source(b(format="anthropic", base_url="https://api.anthropic.com")) == ("litellm", "")
    assert catalog.price_source(b(format="anthropic", base_url="https://bedrock-mantle.us-east-1.api.aws/anthropic")) == ("litellm", "")
    assert catalog.price_source(b(format="anthropic", flavor="vertex")) == ("litellm", "vertex_ai/")
    assert catalog.price_source(b(format="anthropic", base_url="https://x.services.ai.azure.com/anthropic")) == ("litellm", "azure_ai/")
    assert catalog.price_source(b(base_url="https://x.openai.azure.com/openai/v1")) == ("litellm", "azure/")
    assert catalog.price_source(b(base_url="https://api.openai.com/v1")) == ("litellm", "")
    assert catalog.price_source(b(base_url="https://api.mistral.ai/v1")) == ("litellm", "mistral/")
    assert catalog.price_source(b(kind="self_hosted", base_url="http://gpu:8000/v1")) == ("none", "")
    assert catalog.price_source(b(kind="mock")) == ("none", "")
    assert catalog.price_source(b(kind="mock", base_url="https://openrouter.ai/api/v1")) == ("openrouter", "")
    assert catalog.price_source(b(base_url="https://llm.example.com/v1")) == ("none", "")
    assert catalog.price_source(b(base_url="https://llm.example.com/v1", price_prefix="together_ai/")) == ("litellm", "together_ai/")
    assert catalog.price_source(b(base_url="https://openrouter.ai/api/v1", price_source="none")) == ("none", "")


def test_litellm_entries_become_prices_with_tiers():
    entry = {"input_cost_per_token": 1e-7, "output_cost_per_token": 5e-7, "cache_read_input_token_cost": 1e-8, "cache_creation_input_token_cost": 1.25e-7,
             "input_cost_per_token_above_100k_tokens": 5e-7, "output_cost_per_token_above_100k_tokens": 2.5e-6,
             "cache_read_input_token_cost_above_100k_tokens": 5e-8, "input_cost_per_token_above_100k_tokens_batches": 2.5e-7,
             "input_cost_per_token_batches": 5e-8, "output_cost_per_token_above_200k_tokens": 9e-6, "max_output_tokens": 128000}
    p = catalog.litellm_price(entry)
    assert (p["input"], p["output"], p["cache_read"], p["cache_write"]) == (0.1, 0.5, 0.01, 0.125)
    assert p["tiers"] == [{"above": 100_000, "input": 0.5, "output": 2.5, "cache_read": 0.05}]  # 200K has no input price: left out
    assert catalog.litellm_price({"mode": "chat"}) is None and "tiers" not in catalog.litellm_price({"input_cost_per_token": 1e-6})


def live_config():
    raw = copy.deepcopy(TEST_CONFIG)
    raw["backends"] = {**raw["backends"],
                       "direct": {"kind": "api", "format": "anthropic", "base_url": "https://api.anthropic.com"},
                       "azure": {"kind": "api", "format": "openai", "base_url": "https://x.openai.azure.com/openai/v1"},
                       "openrouter": {"kind": "api", "format": "openai", "base_url": "https://openrouter.ai/api/v1"}}
    raw["models"] = {**raw["models"], "haiku": {"routes": [{"backend": "direct", "model": "claude-haiku-5-5"},
                                                            {"backend": "openrouter", "model": "anthropic/claude-haiku-5.5"}]},
                     "terra": {"routes": [{"backend": "azure", "model": "my-deployment", "price_key": "gpt-6.1-terra"},
                                          {"backend": "direct", "model": "nope"}]},
                     "deal": {"routes": [{"backend": "direct", "model": "claude-sonnet-5-5", "price": {"input": 1.5, "output": 7.5}}]}}
    return build_config(raw, prices_data={}, quiet=True)


LITELLM = {"claude-haiku-5-5": {"input_cost_per_token": 1e-7, "output_cost_per_token": 5e-7, "cache_read_input_token_cost": 1e-8,
                                "input_cost_per_token_above_100k_tokens": 5e-7, "output_cost_per_token_above_100k_tokens": 2.5e-6,
                                "max_output_tokens": 128000, "max_input_tokens": 1_000_000},
           "azure/gpt-6.1-terra": {"input_cost_per_token": 1e-6, "output_cost_per_token": 5e-6}}
OPENROUTER = {"data": [{"id": "anthropic/claude-haiku-5.5", "pricing": {"prompt": "0.0000001", "completion": "0.0000005"}, "context_length": 1_000_000}]}


def test_refresh_prices_from_both_lists_and_compare_them():
    cfg = live_config()
    _, n, missing, data = catalog.refresh_prices(cfg, OPENROUTER, write=False, litellm=LITELLM)
    p = data["prices"]
    assert n == 3 and p["direct:claude-haiku-5-5"]["source"] == "litellm" and p["direct:claude-haiku-5-5"]["tiers"][0]["above"] == 100_000
    assert p["azure:my-deployment"] == {"input": 1.0, "output": 5.0, "source": "litellm"}  # found by its price_key
    assert p["openrouter:anthropic/claude-haiku-5.5"]["source"] == "openrouter"
    assert missing == ["terra@direct (nope)"] and "direct:claude-sonnet-5-5" not in p  # an override needs no list price
    assert data["max_output"]["haiku"] == 128000 and data["sources"]["litellm"]["ok"] and data["disagreements"] == []

    dearer = {**OPENROUTER, "data": [{**OPENROUTER["data"][0], "pricing": {"prompt": "0.0000002", "completion": "0.0000005"}}]}
    _, _, _, data = catalog.refresh_prices(cfg, dearer, write=False, litellm=LITELLM)
    assert data["disagreements"] == [{"model": "haiku", "backend": "direct", "field": "input", "listed": 0.1, "openrouter": 0.2}]

    loaded = build_config(copy.deepcopy(TEST_CONFIG), prices_data={}, quiet=True)
    assert all(r.price.source == "config" for m in loaded.models for r in m.routes)  # written into the config: an override


def test_a_list_that_cant_be_fetched_is_reported(monkeypatch):
    def down():
        raise httpx.ConnectError("no route to host")
    monkeypatch.setattr(catalog, "fetch_litellm", down)
    monkeypatch.setattr(catalog.httpx, "get", lambda *a, **k: (_ for _ in ()).throw(httpx.ConnectError("offline")))
    _, n, _, data = catalog.refresh_prices(live_config(), None, write=False)
    assert n == 0 and not data["sources"]["litellm"]["ok"] and "no route to host" in data["sources"]["litellm"]["error"]
    assert not data["sources"]["openrouter"]["ok"]


async def test_the_worker_keeps_prices_live_and_says_what_changed():
    cfg, store, state = live_config(), MemoryStore(), MemoryState()
    lists = {"or": OPENROUTER, "ll": LITELLM}

    def fetch(c, path, write):
        return catalog.refresh_prices(c, lists["or"], write=False, litellm=lists["ll"])
    first = await live_prices.refresh_once(cfg, store, state, fetch)
    stored = await store.get_setting("prices")
    assert await state.get_config_token() == f"file:{stored['updated']}:"  # gateways see it and reload
    assert first["priced_routes"] == 3 and stored["prices"]["direct:claude-haiku-5-5"]["input"] == 0.1 and not await state.recent_alerts()

    await live_prices.refresh_once(cfg, store, state, fetch)
    assert (await store.get_setting("prices"))["updated"] == stored["updated"] or (await store.get_setting("prices"))["prices"] == stored["prices"]
    assert not await state.recent_alerts()  # nothing changed: nothing to say

    lists["ll"] = {**LITELLM, "azure/gpt-6.1-terra": {"input_cost_per_token": 2e-6, "output_cost_per_token": 1e-5}}
    status = await live_prices.refresh_once(cfg, store, state, fetch)
    assert status["changed"] == ["azure:my-deployment: $1/$5 -> $2/$10"]
    assert (await store.get_setting("prices"))["prices"]["azure:my-deployment"]["input"] == 2.0
    assert any(a["title"] == "Prices changed" for a in await state.recent_alerts())

    def litellm_down(c, path, write):
        _, n, missing, data = catalog.refresh_prices(c, lists["or"], write=False, litellm={})
        data["sources"]["litellm"] = {"ok": False, "error": "HTTP 503"}
        data["prices"] = {k: v for k, v in data["prices"].items() if v["source"] != "litellm"}
        return None, n, missing, data
    status = await live_prices.refresh_once(cfg, store, state, litellm_down)
    kept = (await store.get_setting("prices"))["prices"]
    assert kept["azure:my-deployment"]["input"] == 2.0 and kept["direct:claude-haiku-5-5"]["source"] == "litellm"  # last good prices
    assert status["changed"] == [] and any(a["title"] == "Price list unavailable: litellm" for a in await state.recent_alerts())

    lists["or"] = {"data": [{**OPENROUTER["data"][0], "pricing": {"prompt": "0.0000002", "completion": "0.0000005"}}]}
    await live_prices.refresh_once(cfg, store, state, fetch)
    await live_prices.refresh_once(cfg, store, state, fetch)
    titles = [a["title"] for a in await state.recent_alerts()]
    assert titles.count("Prices disagree for haiku") == 1  # once a day, not every refresh

    refresher, calls = live_prices.PriceRefresher(), []

    def counting(c, path, write):
        calls.append(1)
        return fetch(c, path, write)
    assert await refresher.maybe(cfg, store, state, 1000.0, counting) is not None
    assert await refresher.maybe(cfg, store, state, 1000.0 + 60, counting) is None  # not due for 15 minutes
    assert await refresher.maybe(cfg, store, state, 1000.0 + 15 * 60, counting) is not None and len(calls) == 2

    def broken(c, path, write):
        raise RuntimeError("bug")
    assert await live_prices.PriceRefresher().maybe(cfg, store, state, 5.0, broken) is None  # logged; prices stay as they were


def test_changes_are_described():
    old = {"a": {"input": 1, "output": 2}, "b": {"input": 1, "output": 2, "tiers": []}}
    new = {"a": {"input": 1, "output": 2}, "b": {"input": 1, "output": 2, "tiers": [{"above": 1}]}, "c": {"input": 3, "output": 4}}
    assert live_prices.changes(old, new) == ["b: $1/$2 -> $1/$2 (long-prompt prices changed)", "c: none -> $3/$4"]
