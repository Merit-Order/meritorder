"""Exploration (decision 59): a small random share of a team's `auto` task starts one level down or up, decided where no
cache is lost, never for a named model, a client's floor, a capped side request or past the team's policy or budgets;
canaries; the level map recommendations write; and what the ledger and route_features keep."""

import time

import pytest
from test_learned_router import chat, features_rows, shadow_gw  # noqa: F401 (a fixture)

from meritorder.auth import Principal
from meritorder.backends.base import Backend
from meritorder.config import TeamConfig, build_config
from meritorder.proxy import _month_key, headroom
from meritorder.router import Router
from meritorder.sessions import Session

SCORES = {"one": 1.0, "two": 2.0, "three": 3.0, "four": 4.0, "five": 5.0}
PRICE = {"one": 0.05, "two": 0.1, "three": 0.3, "four": 1.0, "five": 3.0}


def build(explore: dict | None = None, learned: str = "shadow", level_map: dict | None = None, allowed=("*",), canaries=None):
    backends = {n: {"format": "openai", "base_url": f"https://{n}.example/v1"} for n in [*SCORES, "new"]}
    models = {n: {"quality": {"coding": s, "general": s}, "routes": [{"backend": n, "model": n, "price": {"input": PRICE[n], "output": 5 * PRICE[n]}}]}
              for n, s in SCORES.items()}
    models["new"] = {"quality": {"coding": 3.2, "general": 3.2}, "routes": [{"backend": "new", "model": "new", "price": {"input": 0.2, "output": 1}}]}
    raw = {"backends": backends, "models": models, "quality": {"source": "config"},
           "routing": {"learned": {"mode": learned}, "level_map": level_map or {}},
           "teams": {"default": {"exploration": {"enabled": True, **(explore or {})}, "allowed_models": list(allowed)}}}
    catalog = {"exploration": {"canaries": canaries or []}}
    cfg = build_config(raw, prices_data=catalog, quiet=True)
    return Router(cfg, {b.name: Backend(b) for b in cfg.backends}), cfg


CODING3 = {"messages": [{"role": "user", "content": "add a --verbose flag"}], "tools": [{"name": "edit"}]}


def decide(r: Router, u: float, body=None, headers=None, session=None, up=True, model="auto"):
    r.rand = lambda: u
    return r.decide(model, r.cfg.team("default"), body or CODING3, headers or {}, session, 1_000, True, {}, time.time(),
                    explore=True, explore_up=up)


def test_a_small_share_goes_one_level_down_or_up_and_says_so():
    r, _ = build()
    down, up, served = decide(r, 0.01), decide(r, 0.03), decide(r, 0.5)
    assert str(down.floor) == "coding:2" and down.candidates[0].model.name == "two"
    assert down.explore == {"from": 3, "share": 0.04, "arm": "down", "level": 2, "p": 0.02}
    assert "exploring: one level down from coding:3 (2.0% of task starts)" in down.reason
    assert str(up.floor) == "coding:4" and up.explore["arm"] == "up" and up.candidates[0].model.name == "four"
    assert str(served.floor) == "coding:3" and served.explore == {"from": 3, "share": 0.04, "arm": "served", "level": 3, "p": 0.96}


def test_at_the_ends_and_near_a_budget_it_only_goes_the_one_way():
    r, _ = build()
    hard = {"messages": [{"role": "user", "content": "design a new scheduler architecture"}], "tools": [{"name": f"t{i}"} for i in range(5)]}
    d = decide(r, 0.03, body=hard)  # coding:5: all of the share goes down
    assert str(d.floor) == "coding:4" and d.explore["arm"] == "down" and d.explore["p"] == pytest.approx(0.04)
    d = decide(r, 0.01, up=False)  # a budget nearly spent: never up, and down keeps its own half
    assert d.explore["arm"] == "down" and d.explore["p"] == pytest.approx(0.02)
    assert decide(r, 0.03, up=False).explore["arm"] == "served"


@pytest.mark.parametrize("case", ["named", "floor", "capped", "warm", "learned off", "not opted in", "not live"])
def test_where_it_never_explores(case):
    r, cfg = build(learned="off" if case == "learned off" else "shadow", explore={"enabled": case != "not opted in"})
    headers = {"x-route-floor": "coding:3"} if case == "floor" else {"x-claude-code-request-class": "compaction"} if case == "capped" else {}
    session = Session(id="s", turns=2, warm_until={"two": time.time() + 300}) if case == "warm" else None
    r.rand = lambda: 0.0
    d = r.decide("three" if case == "named" else "auto", cfg.team("default"), CODING3, headers, session, 1_000, True, {}, time.time(),
                 explore=case != "not live")
    assert d.explore is None and "exploring" not in d.reason


def test_a_session_whose_cache_is_gone_can_explore_at_its_next_task():
    r, _ = build()
    cold = Session(id="s", turns=4, model="three", backend="three", warm_until={"three": time.time() - 1})
    assert decide(r, 0.01, session=cold).explore["arm"] == "down"


def test_never_past_the_teams_allowed_models():
    r, _ = build(allowed=["three", "four", "five", "new"])  # nothing below level 3 it may use: level 2 is served by three
    d = decide(r, 0.01)
    assert d.explore["arm"] == "down" and d.candidates[0].model.name == "new" and {c.model.name for c in d.candidates} <= {"three", "four", "five", "new"}


def test_a_canary_takes_its_share_and_nothing_else_until_its_tried():
    r, _ = build(canaries=[{"task": "coding", "level": 3, "model": "new", "since": "2026-11-15", "reason": "newly cheapest"}])
    c = decide(r, 0.1)
    assert c.explore["arm"] == "canary" and c.explore["p"] == pytest.approx(0.2) and [x.model.name for x in c.candidates] == ["new"]
    assert "canary new at coding:3" in c.reason
    s = decide(r, 0.9)
    assert s.explore["arm"] == "served" and "new" not in [x.model.name for x in s.candidates]  # cheaper, but held back
    assert s.explore["p"] == pytest.approx(1 - 0.2 - 0.8 * 0.04)
    d = decide(r, 0.2 + 0.01)
    assert d.explore["arm"] == "down" and d.explore["p"] == pytest.approx(0.8 * 0.02)


def test_the_level_map_moves_a_level_but_never_a_clients_floor():
    r, _ = build(explore={"enabled": False}, level_map={"coding:3": 2})
    d = decide(r, 0.5)
    assert str(d.floor) == "coding:2" and "coding:3 moved to 2 by routing.level_map" in d.reason
    assert str(decide(r, 0.5, headers={"x-route-floor": "coding:3"}).floor) == "coding:3"
    with pytest.raises(ValueError, match="level_map"):
        build(level_map={"coding:9": 2})


def test_headroom_is_every_budget_under_ninety_percent():
    p = Principal(user_id="u", team="t", role="user", key_id="k")
    team = TeamConfig(name="t", budget_usd_month=100.0)
    _, cfg = build()
    month = _month_key()
    assert headroom({"spend": {f"{month}:team:t": 89.0}}, p, team, cfg)
    assert not headroom({"spend": {f"{month}:team:t": 90.0}}, p, team, cfg)
    assert not headroom({"spend": {}, "session_spend": 9.5}, p, TeamConfig(name="t", max_session_usd=10.0), cfg)


async def test_the_ledger_and_the_tasks_row_keep_the_arm(shadow_gw):  # noqa: F811
    gw, client = shadow_gw
    gw.cfg.team("default").exploration.enabled = True
    gw.router.rand = lambda: 0.0  # the first arm: down
    msgs = [{"role": "user", "content": "write a python function that parses dates"}]  # coding:3
    r = await chat(client, "", {"x-session-id": "x-1"}, messages=msgs)
    assert r.status_code == 200
    await gw.drain()
    row = (await gw.store.recent(1))[0]
    assert row["tags"]["explore"] == "down" and row["tags"]["explore_from"] == "coding:3" and row["floor"] == "coding:2"
    feats = (await features_rows(gw))[0]
    assert feats["arm"] == "down" and feats["arm_from"] == 3 and feats["level"] == 2 and feats["propensity"] == pytest.approx(0.02)
    # Its next turn, mid-task, keeps the tags; a client can't set them.
    more = msgs + [{"role": "assistant", "content": None, "tool_calls": [{"id": "a", "type": "function", "function": {"name": "bash", "arguments": "{}"}}]},
                   {"role": "tool", "tool_call_id": "a", "content": "ok"}]
    r = await chat(client, "", {"x-session-id": "x-1", "x-meritorder-tags": "explore=up"}, messages=more)
    await gw.drain()
    row = (await gw.store.recent(1))[0]
    assert row["tags"]["explore"] == "down" and row["floor"] == "coding:2"


def test_nothing_the_team_may_use_at_the_level_above_means_no_exploring():
    r, _ = build(allowed=["one", "two", "three", "four"])
    agent = {"messages": [{"role": "user", "content": "fix the race condition"}], "tools": [{"name": f"t{i}"} for i in range(5)]}
    d = decide(r, 0.03, body=agent)  # coding:4, drawn up: nothing at 5
    assert d.explore is None and str(d.floor) == "coding:4"


async def test_exploring_up_is_never_why_a_request_is_refused(shadow_gw):  # noqa: F811
    gw, client = shadow_gw
    gw.cfg.team("default").exploration.enabled = True
    gw.cfg.safeguards.max_request_usd = 0.005  # big's reply could cost more; small's can't
    gw.router.rand = lambda: 0.03  # up: coding:4, which only big meets
    msgs = [{"role": "user", "content": "write a python function that parses dates"}]
    r = await chat(client, "", {"x-session-id": "x-2"}, messages=msgs, max_tokens=1000)
    assert r.status_code == 200 and r.headers["x-meritorder-model"] == "small"
    await gw.drain()
    row = (await gw.store.recent(1))[0]
    assert "explore" not in row["tags"] and row["floor"] == "coding:3"
