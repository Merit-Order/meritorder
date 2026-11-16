"""Routing by speed: routes that cost about the same go to the faster one, priorities (cost, balanced, speed)
decide how much more a faster route may cost, and the alternatives keep being measured."""

import time

import httpx
import pytest
from pydantic import ValidationError
from test_http_backend import CHAT, OPENAI_OK, Upstream, http_gateway, post
from test_pricing_router import _router, coding_body

from meritorder.backends.base import Backend
from meritorder.config import BackendConfig, Config, ModelConfig, Price, Route, RoutingConfig, TeamConfig
from meritorder.router import RoutingError
from meritorder.sessions import Session
from meritorder.speed import DEFAULT_SECONDS_PER_TOKEN, STALE_SECONDS, Speeds
from meritorder.router import Router


def timed(speeds, backend, model, seconds, first=None, n=2, now=None, out=600):
    """n identical replies: ``seconds`` in total, the first token after ``first``."""
    for _ in range(n):
        speeds.record(backend, model, seconds, first, out, now or time.time())


def first_backends(d):
    return [c.backend.name for c in d.candidates]


def test_speed_from_streamed_and_whole_replies():
    s = Speeds()
    now = time.time()
    s.record("a", "m", 3.0, 0.5, 601, now)  # 0.5s to the first token, then 600 tokens in 2.5s
    sp = s.get("a", "m")
    assert sp.first_token == 0.5 and sp.per_token == pytest.approx(2.5 / 600) and not sp.measured
    assert s.expected("a", "m", 600) is None  # one reply isn't enough to trust
    s.record("a", "m", 3.0, 1.5, 601, now)
    assert sp.measured and sp.first_token == pytest.approx(0.7)  # moving average: 20% toward the new reply
    assert s.expected("a", "m", 600) == pytest.approx(0.7 + 600 * sp.per_token)
    assert sp.tokens_per_second() == pytest.approx(1 / sp.per_token)

    # A reply that wasn't streamed: the writing time is taken out at the route's speed, or a typical one.
    s.record("b", "m", 2.0, None, 100, now)
    assert s.get("b", "m").first_token == pytest.approx(2.0 - 100 * DEFAULT_SECONDS_PER_TOKEN) and s.get("b", "m").per_token is None
    s.record("c", "m", 1.0, 0.2, 5, now)  # too short to time the writing
    assert s.get("c", "m").per_token is None
    s.record("d", "m", 0, None, 10, now)
    assert s.get("d", "m") is None
    assert not sp.stale(now) and sp.stale(now + STALE_SECONDS + 1)


def test_equal_cost_routes_go_to_the_faster_one():
    router, cfg = _router()
    now = time.time()
    d = router.decide("frontier", cfg.team("default"), coding_body(), {}, None, 5_000, True, {}, now)
    assert first_backends(d) == ["api-a", "api-b"]  # nothing measured: config order, as before
    timed(router.speeds, "api-b", "f", 3.0, 0.5, now=now)
    d = router.decide("frontier", cfg.team("default"), coding_body(), {}, None, 5_000, True, {}, now)
    assert first_backends(d) == ["api-b", "api-a"]  # measured beats unknown
    timed(router.speeds, "api-a", "f", 9.0, 3.0, now=now)
    d = router.decide("frontier", cfg.team("default"), coding_body(), {}, None, 5_000, True, {}, now)
    assert first_backends(d) == ["api-b", "api-a"] and d.priority == "cost"
    assert "cost priority: api-b (~3.0s) over api-a (~9.0s)" in d.reason


def test_a_warm_cache_still_beats_a_faster_route():
    router, cfg = _router()
    now = time.time()
    timed(router.speeds, "api-a", "f", 2.0, 0.3, now=now)
    timed(router.speeds, "api-b", "f", 9.0, 3.0, now=now)
    sess = Session(id="s", model="frontier", backend="api-b", turns=10, prefix_tokens=100_000, warm_until={"api-b": now + 200})
    for priority in ("cost", "speed"):
        d = router.decide("frontier", cfg.team("default"), coding_body(), {"x-meritorder-priority": priority}, sess, 102_000, False, {}, now)
        assert d.candidates[0].backend.name == "api-b"  # moving would rebuild 100k tokens of cache


def _priced(*multipliers, teams=()):
    names = [f"r{i}" for i in range(len(multipliers))]
    bcs = [BackendConfig(name=n, format="openai", base_url=f"https://{n}") for n in names]
    routes = [Route(backend=n, model="m", price=Price(input=3 * k, output=15 * k)) for n, k in zip(names, multipliers)]
    cfg = Config(backends=bcs, models=[ModelConfig(name="m", quality={"coding": 4, "general": 4}, routes=routes)],
                 teams=[TeamConfig(name="default"), *teams])
    return Router(cfg, {b.name: Backend(b) for b in bcs}), cfg


def test_priorities_decide_how_much_more_a_faster_route_may_cost():
    router, cfg = _priced(1, 1.15, 1.9, 2.5, teams=[TeamConfig(name="fast", priority="speed")])
    now = time.time()
    for name, seconds in (("r0", 10.0), ("r1", 5.0), ("r2", 3.0), ("r3", 1.0)):
        timed(router.speeds, name, "m", seconds, seconds / 5, now=now)

    def order(team="default", **headers):
        return first_backends(router.decide("m", cfg.team(team), CHAT, headers, None, 2_000, True, {}, now))

    assert order() == ["r0", "r1", "r2", "r3"]  # cost: only near-ties compete on speed
    assert order(**{"x-meritorder-priority": "balanced"}) == ["r1", "r0", "r2", "r3"]  # up to 20% more
    assert order(**{"x-meritorder-priority": "speed"}) == ["r2", "r1", "r0", "r3"]  # up to double; r3 is 2.5x
    assert order("fast") == ["r2", "r1", "r0", "r3"]  # the team's priority
    assert order("fast", **{"x-meritorder-priority": "Cost "}) == ["r0", "r1", "r2", "r3"]  # the request's wins
    d = router.decide("m", cfg.team("default"), CHAT, {"x-meritorder-priority": "balanced"}, None, 2_000, True, {}, now)
    assert "balanced priority: r1 (~5.0s) over r0 (~10.0s) for 15% more" in d.reason
    with pytest.raises(RoutingError) as e:
        router.decide("m", cfg.team("default"), CHAT, {"x-meritorder-priority": "fastest"}, None, 2_000, True, {}, now)
    assert e.value.status == 400 and "cost, balanced, speed" in e.value.message


def test_the_alternatives_keep_being_measured():
    router, cfg = _router()
    now = time.time()
    timed(router.speeds, "api-a", "f", 3.0, 0.5, now=now)

    def first(session=None):
        d = router.decide("frontier", cfg.team("default"), coding_body(), {}, session, 5_000, True, {}, now)
        return d.candidates[0].backend.name, d.reason

    runs = [first() for _ in range(20)]
    assert [name for name, _ in runs].count("api-b") == 2 and runs[9][0] == runs[19][0] == "api-b"
    assert "checking api-b's speed (not measured yet)" in runs[9][1]
    # Out of date counts too, and the check goes to the route measured longest ago.
    timed(router.speeds, "api-b", "f", 9.0, 3.0, now=now - STALE_SECONDS - 60)
    runs = [first() for _ in range(10)]
    assert runs[9][0] == "api-b" and "checking api-b's speed (~9.0s)" in runs[9][1]
    # Never in the middle of a session: its cache is on its current route.
    sess = Session(id="s", model="frontier", backend="api-a", turns=3)
    assert {first(sess)[0] for _ in range(30)} == {"api-a"}
    router.cfg.routing.speed_check_every = 0
    assert {first()[0] for _ in range(30)} == {"api-a"}


def test_premium_config_keeps_the_defaults_and_rejects_unknown_priorities():
    assert RoutingConfig(priority_premium_pct={"balanced": 30}).priority_premium_pct == {"cost": 2.0, "balanced": 30, "speed": 100.0}
    with pytest.raises(ValidationError):
        RoutingConfig(priority_premium_pct={"fastest": 10})
    with pytest.raises(ValidationError):
        RoutingConfig(priority_premium_pct={"speed": -1})
    with pytest.raises(ValidationError):
        TeamConfig(name="t", priority="fastest")


async def test_the_gateway_times_replies_from_when_the_route_was_called(monkeypatch):
    a = Upstream(httpx.Response(503, json={"error": {"message": "busy"}}))
    gw = http_gateway(monkeypatch, a, Upstream(httpx.Response(200, json=OPENAI_OK)))
    r = await post(gw, CHAT)
    assert r.headers["x-meritorder-backend"] == "up-b"
    await gw.drain()
    assert gw.router.speeds.get("up-a", "m-a") is None  # failures aren't speeds
    sp = gw.router.speeds.get("up-b", "m-b")
    assert sp.samples == 1 and sp.first_token is not None and sp.first_token < 1
    # A configuration reload keeps what was measured.
    await gw.apply_config(gw.cfg, None)
    assert gw.router.speeds.get("up-b", "m-b").samples == 1


async def test_priority_header_through_the_gateway(monkeypatch):
    gw = http_gateway(monkeypatch, Upstream(httpx.Response(200, json=OPENAI_OK)), Upstream(httpx.Response(200, json=OPENAI_OK)))
    gw.cfg.routing.speed_check_every = 0
    timed(gw.router.speeds, "up-a", "m-a", 8.0, 2.0)
    timed(gw.router.speeds, "up-b", "m-b", 2.0, 0.4)  # twice the price, four times as fast
    assert (await post(gw, CHAT)).headers["x-meritorder-backend"] == "up-a"
    assert (await post(gw, CHAT, **{"x-meritorder-priority": "speed"})).headers["x-meritorder-backend"] == "up-b"
    r = await post(gw, CHAT, **{"x-meritorder-priority": "now"})
    assert r.status_code == 400 and "X-MeritOrder-Priority" in r.text


async def test_console_shows_measured_speeds(monkeypatch):
    from conftest import ADMIN_KEY as admin

    from meritorder.app import create_app

    gw = http_gateway(monkeypatch, Upstream(httpx.Response(200, json=OPENAI_OK)), Upstream(httpx.Response(200, json=OPENAI_OK)))
    gw.router.speeds.record("up-a", "m-a", 2.0, 0.25, 401, time.time() - 60)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(gateway=gw)), base_url="http://t") as c:
        backends = {b["name"]: b for b in (await c.get("/v1/console/backends", headers={"authorization": f"Bearer {admin}"})).json()["backends"]}
    speeds = backends["up-a"]["speeds"]["m-a"]
    assert 60 <= speeds.pop("seconds_ago") < 90  # measured a minute ago, by the real clock: a slow runner adds a little
    assert speeds == {"first_token_ms": 250, "tokens_per_second": 228.6, "samples": 1}
    assert backends["up-b"]["speeds"] == {}
