"""Claude's 1-hour prompt cache: asked for from a session's second turn on a backend set to an hour, priced at its own
write price (2x input, against 1.25x for 5 minutes), and kept as warm as the markers actually sent (decision 53)."""

import copy
import json
import time

import httpx
import pytest
from conftest import TEST_CONFIG
from test_http_backend import ANTHROPIC_OK, Upstream, post

from meritorder.app import build_gateway
from meritorder.backends.base import Backend
from meritorder.backends.mock import MockBackend
from meritorder.cache_shaping import marker_ttls, shape_anthropic
from meritorder.catalog import litellm_price
from meritorder.config import BackendConfig, Config, ModelConfig, Price, PriceTier, Route, TeamConfig, build_config
from meritorder.formats.translate import UsageTracker, anthropic_usage, usage_from_anthropic
from meritorder.formats.sse import SSEEvent
from meritorder.pricing import Usage, cost
from meritorder.router import Router
from meritorder.sessions import Session
from meritorder.state import MemoryState
from meritorder.store import MemoryStore

PRICE = {"input": 4, "output": 20, "cache_read": 0.2, "cache_write": 5, "cache_write_1h": 8}
HOUR_WRITES = {"input_tokens": 10, "cache_read_input_tokens": 1000, "cache_creation_input_tokens": 500,
               "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 500}, "output_tokens": 2}


def test_one_hour_writes_cost_their_own_price():
    p = Price(**PRICE)
    u = Usage(input_tokens=10, cache_read_tokens=1000, cache_write_tokens=500, cache_write_1h_tokens=200, output_tokens=2)
    assert cost(p, u) == (10 * 4 + 1000 * 0.2 + 300 * 5 + 200 * 8 + 2 * 20) / 1e6
    assert cost(p, Usage(cache_write_tokens=500)) == 500 * 5 / 1e6  # not split by the backend: the 5-minute price
    assert Price(input=3).write_1h() == 6  # unlisted: Anthropic's 2x input, on every platform
    negotiated = Price(input=1.76, output=8.8, cache_write=2.2)
    assert negotiated.write_1h() == 3.52  # moves with a negotiated input price


def test_one_hour_prices_follow_long_prompt_tiers_and_batches():
    p = Price(input=1, output=5, cache_write=1.25, cache_write_1h=2, tiers=[PriceTier(above=100_000, input=2, output=10, cache_write_1h=5)])
    assert p.at(200_000).write_1h() == 5 and p.at(10).write_1h() == 2
    assert Price(input=1, output=5, cache_write_1h=2, tiers=[PriceTier(above=100_000, input=2, output=10)]).at(200_000).write_1h() == 4
    assert Price(input=1, output=5, cache_write_1h=2).for_batch(0.5).write_1h() == 1


def test_litellm_one_hour_write_prices_are_read():
    entry = {"input_cost_per_token": 1e-07, "output_cost_per_token": 5e-07, "cache_creation_input_token_cost": 1.25e-07,
             "cache_creation_input_token_cost_above_1hr": 2e-07, "cache_creation_input_token_cost_above_100k_tokens": 6.25e-07,
             "cache_creation_input_token_cost_above_1hr_above_100k_tokens": 1e-06, "input_cost_per_token_above_100k_tokens": 5e-07,
             "output_cost_per_token_above_100k_tokens": 2.5e-06}
    price = litellm_price(entry)
    assert price["cache_write"] == 0.125 and price["cache_write_1h"] == 0.2
    assert price["tiers"] == [{"above": 100_000, "input": 0.5, "output": 2.5, "cache_write": 0.625, "cache_write_1h": 1.0}]


def test_usage_split_by_lifetime_is_read_and_written_back():
    u = usage_from_anthropic(HOUR_WRITES)
    assert u.cache_write_tokens == 500 and u.cache_write_1h_tokens == 500
    assert anthropic_usage(u)["cache_creation"] == {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 500}
    plain = usage_from_anthropic({"input_tokens": 1, "cache_creation_input_tokens": 5})
    assert plain.cache_write_1h_tokens is None and "cache_creation" not in anthropic_usage(plain)
    tracker = UsageTracker("anthropic")
    tracker.feed(SSEEvent(event="message_start", data=json.dumps({"type": "message_start", "message": {"usage": HOUR_WRITES}})))
    tracker.feed(SSEEvent(event="message_delta", data=json.dumps({"type": "message_delta", "usage": {"output_tokens": 9}})))
    assert tracker.usage.cache_write_1h_tokens == 500 and tracker.usage.output_tokens == 9


def test_marker_lifetimes_are_read_in_prompt_order():
    body = {"tools": [{"name": "t", "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
            "system": [{"type": "text", "text": "s", "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}]}]}
    assert marker_ttls(body) == [3600, 3600, 300]
    assert marker_ttls({"messages": [], "cache_control": {"type": "ephemeral", "ttl": "1h"}}) == [3600]
    assert marker_ttls({"messages": [{"role": "user", "content": "hi"}]}) == []


def test_a_first_turn_gets_five_minutes_even_on_a_backend_set_to_an_hour():
    hour = BackendConfig(name="an", format="anthropic", base_url="https://api.anthropic.com", cache_ttl_seconds=3600)
    first, later, five = {"messages": []}, {"messages": []}, {"messages": []}
    assert shape_anthropic(first, hour) and first["cache_control"] == {"type": "ephemeral"}
    assert shape_anthropic(later, hour, later_turn=True) and later["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert shape_anthropic(five, hour.model_copy(update={"cache_ttl_seconds": 300}), later_turn=True)
    assert five["cache_control"] == {"type": "ephemeral"}


def test_router_prices_writes_at_the_hour_where_the_backend_caches_for_one():
    router = Router(build_config(copy.deepcopy(TEST_CONFIG), quiet=True), {})
    price = Price(**PRICE)
    five = MockBackend(BackendConfig(name="an", kind="api", format="anthropic", base_url="https://api.anthropic.com"))
    hour = MockBackend(BackendConfig(name="an", kind="api", format="anthropic", base_url="https://api.anthropic.com", cache_ttl_seconds=3600))
    assert router.estimate(price, hour, 20_000, None, warm=False, turns=3) > router.estimate(price, five, 20_000, None, warm=False, turns=3)


HAIKU = Price(input=0.1, output=0.5, cache_read=0.01, cache_write=0.125)  # Claude Haiku 5.5's list price; 1-hour write 2x input


def test_a_new_session_is_priced_as_the_gateway_sends_it():
    """A session's first turn on a backend set to an hour writes at the 5-minute price (rebuild included), the turns after
    it at the hour's; once the session has come back, every write is an hour's."""
    router = Router(build_config(copy.deepcopy(TEST_CONFIG), quiet=True), {})
    five = MockBackend(BackendConfig(name="an", kind="api", format="anthropic", base_url="https://api.anthropic.com"))
    hour = MockBackend(BackendConfig(name="an", kind="api", format="anthropic", base_url="https://api.anthropic.com", cache_ttl_seconds=3600))
    new = router.cfg.routing.est_new_tokens_per_turn
    for session in (None, Session(id="s", turns=0)):
        one = [router.estimate(HAIKU, b, 30_000, session, warm=False, turns=1) for b in (hour, five)]
        assert one[0] == one[1]
        gap = router.estimate(HAIKU, hour, 30_000, session, warm=False, turns=3) - router.estimate(HAIKU, five, 30_000, session, warm=False, turns=3)
        assert gap == pytest.approx(2 * new * (HAIKU.write_1h() - HAIKU.write()) / 1e6)
    back = Session(id="s", turns=4, prefix_tokens=28_000)
    rebuild = router.estimate(HAIKU, hour, 30_000, back, warm=False, turns=1) - router.estimate(HAIKU, hour, 30_000, back, warm=True, turns=1)
    assert rebuild == pytest.approx(28_000 * (0.2 - 0.01) / 1e6)  # a session that came back rebuilds at the hour's price
    assert router.estimate(HAIKU, hour, 30_000, None, warm=False, turns=0) == pytest.approx(28_000 * (0.125 - 0.01) / 1e6)


def haiku_router():
    bcs = [BackendConfig(name="anthropic", format="anthropic", base_url="https://api.anthropic.com", cache_ttl_seconds=3600),
           BackendConfig(name="openrouter", format="openai", base_url="https://openrouter.ai/api/v1")]
    haiku = ModelConfig(name="claude-haiku", quality={"coding": 3, "general": 3},
                        routes=[Route(backend="openrouter", model="anthropic/claude-haiku-5.5", price=HAIKU),
                                Route(backend="anthropic", model="claude-haiku-5-5", price=HAIKU)])
    cfg = Config(backends=bcs, models=[haiku], teams=[TeamConfig(name="default")])
    return Router(cfg, {b.name: Backend(b) for b in bcs}), cfg


def test_a_new_haiku_session_is_not_priced_off_the_direct_backends():
    """Before, the direct backends priced a new session's rebuild at the hour's write price: 44% dearer than OpenRouter
    over three turns, beyond even the balanced priority's 20%. Now the rebuild and first turn cost the same on both, and
    only the hour's premium on the later turns remains (about 5% when sessions never pause), so a faster direct route
    wins under `balanced`."""
    router, cfg = haiku_router()
    router.pauses.share = lambda team: 0.0  # sessions that never pause between 5 and 60 minutes
    now = time.time()
    for _ in range(2):
        router.speeds.record("anthropic", "claude-haiku-5-5", 2.0, 0.4, 600, now)
        router.speeds.record("openrouter", "anthropic/claude-haiku-5.5", 4.0, 1.0, 600, now)
    body = {"messages": [{"role": "user", "content": "fix this bug in app.py"}]}
    d = router.decide("claude-haiku", cfg.team("default"), body, {"x-meritorder-priority": "balanced"}, None, 30_000, True, {}, now)
    cost = {c.backend.name: c.cost for c in d.candidates}
    assert d.candidates[0].backend.name == "anthropic"
    assert 1 < cost["anthropic"] / cost["openrouter"] < 1.06


def test_pauses_credit_the_hour_cache_so_a_new_haiku_session_goes_direct_at_the_cost_priority():
    """Decision 60: each later turn after a 5-60 minute pause rebuilds a 5-minute cache and not an hour's. At the prior
    (one turn in ten), that outweighs the hour's write premium: a new session goes to Anthropic directly at `cost`."""
    router, cfg = haiku_router()
    body = {"messages": [{"role": "user", "content": "fix this bug in app.py"}]}
    team = cfg.team("default")

    def costs() -> tuple[str, dict[str, float]]:
        d = router.decide("claude-haiku", team, body, {}, None, 30_000, True, {}, time.time())
        return d.candidates[0].backend.name, {c.backend.name: c.cost for c in d.candidates}

    assert router.pauses.share("default") == pytest.approx(0.10)  # no data yet: the prior
    first, cost = costs()
    assert first == "anthropic" and cost["anthropic"] < cost["openrouter"]
    # Three turns: the hour's premium on two later turns' 2,000 new tokens, against one rebuild of 28,000 tokens a tenth
    # of the time on each of them at 5-minute prices.
    premium = 2 * 2_000 * (HAIKU.write_1h() - HAIKU.write()) / 1e6
    saved = 2 * 0.10 * 28_000 * (HAIKU.write() - HAIKU.read()) / 1e6
    assert cost["openrouter"] - cost["anthropic"] == pytest.approx(saved - premium)
    # A team whose sessions never pause: the hour costs more for nothing, and OpenRouter is cheaper again.
    for _ in range(2_000):
        router.pauses.observe("default", 20.0)
    assert router.pauses.share("default") < 0.01
    first, cost = costs()
    assert first == "openrouter" and 1 < cost["anthropic"] / cost["openrouter"] < 1.06


def gateway(monkeypatch, upstream: Upstream, ttl: int = 3600):
    monkeypatch.setenv("MO_TEST_AN", "key-an")
    raw = copy.deepcopy(TEST_CONFIG)
    raw["backends"] = {"an": {"kind": "api", "format": "anthropic", "base_url": "https://api.anthropic.com", "cache_ttl_seconds": ttl,
                              "auth": {"type": "api_key", "header": "x-api-key", "value_env": "MO_TEST_AN"}}}
    raw["models"] = {"c": {"quality": {"coding": 4, "general": 4}, "routes": [{"backend": "an", "model": "claude-x", "price": PRICE}]}}
    raw["baseline_model"] = "c"
    gw = build_gateway(build_config(raw, quiet=True), store=MemoryStore(), state=MemoryState())
    gw.backends["an"].client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    return gw


def warm_for(gw) -> float:
    """Seconds from now that the one session's cache on backend "an" is expected to stay warm."""
    sessions = [v for k, (v, _) in gw.state._kv.items() if k.startswith("sess:")]
    assert len(sessions) == 1
    return sessions[0]["warm_until"]["an"] - time.time()


BODY = {"model": "c", "max_tokens": 20, "system": "rules", "messages": [{"role": "user", "content": "hi"}]}


async def test_a_session_caches_for_an_hour_from_its_second_turn(monkeypatch):
    up = Upstream(httpx.Response(200, json=ANTHROPIC_OK), httpx.Response(200, json={**ANTHROPIC_OK, "usage": HOUR_WRITES}))
    gw = gateway(monkeypatch, up)
    for _ in range(2):
        assert (await post(gw, BODY, path="/v1/messages", **{"x-session-id": "s1"})).status_code == 200
    await gw.drain()
    assert [json.loads(r.content)["cache_control"] for r in up.requests] == [{"type": "ephemeral"}, {"type": "ephemeral", "ttl": "1h"}]
    assert 3500 < warm_for(gw) <= 3600
    row = (await gw.store.recent(1))[0]
    assert row["cache_write_tokens"] == 500 and abs(row["actual_cost"] - (10 * 4 + 1000 * 0.2 + 500 * 8 + 2 * 20) / 1e6) < 1e-12
    assert row["proxy_cache"] is True and row["savings_caching"] > 0


async def test_a_clients_own_five_minute_markers_stay_five_minutes(monkeypatch):
    """Claude Code sets its own markers (an hour on its system prompt, 5 minutes on the conversation): left as they are,
    and the conversation, which the last marker covers, is warm for 5 minutes."""
    up = Upstream(httpx.Response(200, json=ANTHROPIC_OK))
    gw = gateway(monkeypatch, up)
    body = {**BODY, "system": [{"type": "text", "text": "rules", "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}]}]}
    for _ in range(2):
        assert (await post(gw, body, path="/v1/messages", **{"x-session-id": "s2"})).status_code == 200
    await gw.drain()
    assert all("cache_control" not in json.loads(r.content) for r in up.requests)
    assert 200 < warm_for(gw) <= 300
    assert (await gw.store.recent(1))[0]["proxy_cache"] is False


async def test_writes_a_backend_doesnt_split_are_priced_from_the_markers(monkeypatch):
    """A platform that reports only cache_creation_input_tokens: every marker asked for an hour, so every write was one."""
    unsplit = {"input_tokens": 10, "cache_read_input_tokens": 1000, "cache_creation_input_tokens": 500, "output_tokens": 2}
    up = Upstream(httpx.Response(200, json={**ANTHROPIC_OK, "usage": unsplit}))
    gw = gateway(monkeypatch, up)
    for _ in range(2):
        assert (await post(gw, BODY, path="/v1/messages", **{"x-session-id": "s3"})).status_code == 200
    await gw.drain()
    second, first = await gw.store.recent(2)
    assert abs(first["actual_cost"] - (10 * 4 + 1000 * 0.2 + 500 * 5 + 2 * 20) / 1e6) < 1e-12  # 5-minute marker
    assert abs(second["actual_cost"] - (10 * 4 + 1000 * 0.2 + 500 * 8 + 2 * 20) / 1e6) < 1e-12  # 1-hour marker


async def test_eval_traffic_is_cached_but_counts_no_caching_savings(monkeypatch):
    """Without Merit Order an eval wouldn't run at all, so the caching the gateway adds to it saves nobody anything."""
    up = Upstream(httpx.Response(200, json={**ANTHROPIC_OK, "usage": HOUR_WRITES}))
    gw = gateway(monkeypatch, up, ttl=300)
    r = await post(gw, BODY, path="/v1/messages", **{"x-meritorder-tags": "eval=run-1"})
    assert r.status_code == 200 and json.loads(up.requests[0].content)["cache_control"] == {"type": "ephemeral"}
    await gw.drain()
    row = (await gw.store.recent(1))[0]
    assert row["proxy_cache"] is True and row["savings_caching"] == 0 and row["savings_routing"] == 0
    assert row["baseline_cost"] == row["actual_cost"]
