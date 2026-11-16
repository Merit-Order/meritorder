"""The two-sided outcome report (decision 59): success rates by exploration arm, weighted by each arm's chance, "too high"
and "too low" estimates, recommendations once evidence is enough, applying them to routing.level_map, and the worker's
watcher: catalog events, canaries, and alerts."""

import copy
import random
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from conftest import ADMIN_KEY, LEAD_KEY, TEST_CONFIG, auth

from meritorder import benchmark as bm
from meritorder.config import build_config
from meritorder.state import MemoryState
from meritorder.store import MemoryStore

NOW = datetime.now(UTC)


def cfg_with(models: dict[str, tuple[float, float]] | None = None, routing: dict | None = None, teams: dict | None = None):
    models = models or {"cheap": (2.0, 0.1), "mid": (3.0, 0.5), "top": (5.0, 3.0)}
    raw = {"backends": {n: {"format": "openai", "base_url": f"https://{n}.example/v1"} for n in models},
           "models": {n: {"quality": {"coding": q, "general": q}, "routes": [{"backend": n, "model": n, "price": {"input": p, "output": 5 * p}}]}
                      for n, (q, p) in models.items()},
           "quality": {"source": "config"}, "routing": {"learned": {"mode": "shadow"}, **(routing or {})},
           "teams": teams or {"default": {"exploration": {"enabled": True}}}}
    return build_config(raw, prices_data={}, quiet=True), raw


def rows(arm: str, n: int, fail: float, p: float, level_from: int = 3, team: str = "default", rng=None, learned_level=None) -> list[dict]:
    """``n`` task starts on one arm, a share ``fail`` of them followed by a correction (a 0.7 signal)."""
    rng = rng or random.Random(1)
    out = []
    for _ in range(n):
        bad = rng.random() < fail
        out.append({"request_id": uuid.uuid4(), "ts": NOW - timedelta(hours=1), "team": team, "task": "coding", "rules_level": level_from,
                    "status": 200, "retried": False, "floor_raised": None, "arm": arm, "arm_from": level_from, "propensity": p,
                    "level": level_from - (arm == "down") + (arm == "up"), "served_model": "mid",
                    "coding": {"correction": "wrong"} if bad else {"tests": 1}, "learned": {"level": learned_level} if learned_level else None})
    return out


DAILY = [{"day": NOW.date().isoformat(), "floor": "coding:3", "arm": "", "requests": 900, "cost": 9.0},
         {"day": NOW.date().isoformat(), "floor": "coding:2", "arm": "down", "requests": 50, "cost": 0.1},
         {"day": NOW.date().isoformat(), "floor": "general:2", "arm": "", "requests": 50, "cost": 0.9}]


def test_a_level_below_that_does_as_well_is_recommended_lower():
    cfg, _ = cfg_with()
    data = rows("served", 3000, 0.10, 0.96) + rows("down", 400, 0.10, 0.02, rng=random.Random(2))
    rep = bm.outcomes_report(data, DAILY, cfg)
    row = next(r for r in rep["levels"] if (r["task"], r["level"]) == ("coding", 3))
    assert row["traffic_share"] == 0.9 and row["spend_share"] == pytest.approx(0.9)
    served, down = row["arms"]["served"], row["arms"]["down"]
    assert served["n"] == 3000 and served["rate"] == pytest.approx(1 - 0.7 * 0.10, abs=0.02)
    assert row["too_high"]["enough"] and row["too_high"]["clear"]
    rec = rep["recommendations"][0]
    assert rec["to"] == 2 and rec["direction"] == "lower" and not rec["applied"]
    assert rec["text"].startswith("coding level 3: level 2 does as well in ") and "(n=400 explored, 3,000 served); consider lowering" in rec["text"]


def test_no_recommendation_without_enough_outcomes_or_when_the_level_below_fails_more():
    cfg, _ = cfg_with()
    few = bm.outcomes_report(rows("served", 3000, 0.1, 0.96) + rows("down", 50, 0.1, 0.02), DAILY, cfg)
    assert not few["recommendations"] and not few["levels"][1]["too_high"]["enough"]
    worse = bm.outcomes_report(rows("served", 3000, 0.05, 0.96) + rows("down", 400, 0.40, 0.02), DAILY, cfg)
    assert not worse["recommendations"] and worse["levels"][1]["too_high"]["ratio"] < 0.9


def test_a_level_above_that_clearly_succeeds_more_is_recommended_higher():
    cfg, _ = cfg_with()
    data = rows("served", 3000, 0.40, 0.96) + rows("up", 400, 0.02, 0.02, rng=random.Random(3))
    rec = bm.outcomes_report(data, DAILY, cfg)["recommendations"][0]
    assert rec["direction"] == "raise" and rec["to"] == 4 and "consider raising" in rec["text"]


def test_teams_exploring_at_different_shares_pool_by_inverse_propensity():
    """Team a explores at 4%, b at 40%, and b's work fails more on every arm: an unweighted mean of the down arm would be
    mostly b's; weighting each row by 1 / its chance gives each team's rows the weight of the traffic they stand for."""
    cfg, _ = cfg_with()
    a = rows("served", 960, 0.05, 0.96, team="a") + rows("down", 20, 0.05, 0.02, team="a")
    b = rows("served", 60, 0.50, 0.60, team="b", rng=random.Random(4)) + rows("down", 20, 0.50, 0.20, team="b", rng=random.Random(5))
    arms = bm.outcomes_report(a + b, DAILY, cfg)["levels"][1]["arms"]
    plain = sum(1 - 0.7 * bool(r["coding"].get("correction")) for r in a + b if r["arm"] == "down") / 40
    assert arms["down"]["rate"] > plain + 0.05  # a's 20 rows stand for 1,000 task starts, b's for 100
    assert arms["down"]["rate"] == pytest.approx(arms["served"]["rate"], abs=0.06)  # the same work on both arms
    assert arms["down"]["n_eff"] < arms["down"]["n"]


def test_a_dismissed_recommendation_stays_dismissed_until_its_evidence_changes():
    cfg, _ = cfg_with()
    data = rows("served", 3000, 0.10, 0.96) + rows("down", 400, 0.10, 0.02, rng=random.Random(2))  # explored an hour ago
    at = lambda delta: (NOW + delta).isoformat()  # noqa: E731
    held = bm.outcomes_report(data, DAILY, cfg, dismissed={"coding:3": {"direction": "lower", "to": 2, "at": at(timedelta(0)), "by": "root"}})
    row = next(r for r in held["levels"] if (r["task"], r["level"]) == ("coding", 3))
    assert held["recommendations"] == [] and row["recommendation"]["dismissed"] == {"at": at(timedelta(0)), "by": "root", "new_outcomes": 0}
    # 400 task starts explored on level 2 since it was dismissed (min_outcomes is 200): it's back.
    stale = bm.outcomes_report(data, DAILY, cfg, dismissed={"coding:3": {"direction": "lower", "to": 2, "at": at(timedelta(hours=-2))}})
    assert stale["recommendations"][0]["dismissed"] is None
    # A dismissal of something else (another direction), or one with no usable time, doesn't hold.
    other = bm.outcomes_report(data, DAILY, cfg, dismissed={"coding:3": {"direction": "raise", "to": 4, "at": at(timedelta(0))}})
    assert other["recommendations"] and not other["recommendations"][0]["dismissed"]
    assert bm.outcomes_report(data, DAILY, cfg, dismissed={"coding:3": {"direction": "lower", "to": 2}})["recommendations"]
    naive = (NOW + timedelta(minutes=1)).replace(tzinfo=None).isoformat()
    assert not bm.outcomes_report(data, DAILY, cfg, dismissed={"coding:3": {"direction": "lower", "to": 2, "at": naive}})["recommendations"]


async def test_the_watcher_leaves_a_dismissed_recommendation_alone():
    store, state = MemoryStore(), MemoryState()
    cfg, raw = cfg_with()
    for r in rows("served", 3000, 0.1, 0.96) + rows("down", 400, 0.1, 0.02, rng=random.Random(2)):
        await store.record_route_features({**r, "fv": "x", "text_features": [], "tags": []})
    await store.set_setting(bm.DISMISSED, {"coding:3": {"direction": "lower", "to": 2, "at": NOW.isoformat()}})
    out = await bm.Watcher().run(cfg, store, state, NOW.timestamp(), raw)
    assert out["recommendations"] == [] and not any(a["title"].startswith("Level recommendation") for a in await state.recent_alerts())


def test_applying_moves_the_level_and_what_was_already_moved_to_it():
    raw = {"routing": {"level_map": {"coding:4": 3}}}
    doc, changed = bm.apply(raw, "coding", 3, 2)
    assert doc["routing"]["level_map"] == {"coding:4": 2, "coding:3": 2} and changed == ["coding:4 3→2", "coding:3 3→2"]
    back, changed = bm.apply(doc, "coding", 3, 3)
    assert back["routing"]["level_map"] == {"coding:4": 2} and changed == ["coding:3 2→3"]
    assert bm.apply({"routing": {}}, "coding", 3, 3) == ({"routing": {}}, [])


def test_snapshots_say_what_changed_and_make_canaries():
    cfg, _ = cfg_with()
    old = bm.snapshot(cfg)
    assert old["cheapest"]["coding:2"] == "cheap" and old["eligible"]["coding:3"] == ["mid", "top"]
    newer, _ = cfg_with({"cheap": (2.0, 0.1), "mid": (3.0, 0.5), "top": (5.0, 3.0), "fresh": (3.4, 0.3)})
    events, canaries = bm.changes(old, bm.snapshot(newer), "2026-11-15T00:00:00+00:00")
    kinds = [e["kind"] for e in events]
    assert "catalog" in kinds and {"kind": "cheapest", "key": "coding:3"}.items() <= next(e for e in events if e["kind"] == "cheapest").items()
    assert {"task": "coding", "level": 3, "model": "fresh"} in [{k: c[k] for k in ("task", "level", "model")} for c in canaries]
    assert not any(c["model"] == "fresh" and c["level"] == 1 and "cheapest" in c["reason"] for c in canaries)  # cheap stays cheapest there
    rescored, _ = cfg_with({"cheap": (2.4, 0.1), "mid": (3.0, 0.5), "top": (5.0, 3.0)})
    events, _ = bm.changes(old, bm.snapshot(rescored), "t")
    assert any(e["kind"] == "scores" and "cheap coding 2.0→2.4" in e["text"] for e in events)


async def test_the_watcher_alerts_keeps_canaries_until_tried_and_publishes_them():
    store, state = MemoryStore(), MemoryState()
    await store.set_setting("prices", {"prices": {}})
    cfg, _ = cfg_with()
    w = bm.Watcher()
    await w.run(cfg, store, state, NOW.timestamp())
    newer, _ = cfg_with({"cheap": (2.0, 0.1), "mid": (3.0, 0.5), "top": (5.0, 3.0), "fresh": (3.4, 0.3)})
    out = await w.run(newer, store, state, NOW.timestamp() + 1000)
    assert {(c["model"], c["level"]) for c in out["canaries"]} >= {("fresh", 3)}
    alerts = await state.recent_alerts()
    assert any(a["title"] == "New cheapest model at coding:3" for a in alerts)
    prices = await store.get_setting("prices")
    assert prices["exploration"]["canaries"] == out["canaries"]
    # Enough outcomes on it at its level: done.
    for r in rows("canary", 30, 0.0, 0.2):
        await store.record_route_features({**r, "ts": NOW + timedelta(seconds=1500), "served_model": "fresh", "level": 3, "fv": "x", "text_features": [], "tags": []})
    out = await w.run(newer, store, state, NOW.timestamp() + 2000)
    left = {(c["task"], c["model"], c["level"]) for c in out["canaries"]}
    assert ("coding", "fresh", 3) not in left and ("general", "fresh", 3) in left  # the general one has no outcomes yet
    doc = await store.get_setting(bm.SETTING)
    assert any(e["kind"] == "canary" and "fresh" in e["text"] for e in doc["events"])
    assert await w.maybe(newer, store, state, NOW.timestamp() + 2001) is not None  # the first one runs
    assert await w.maybe(newer, store, state, NOW.timestamp() + 2002) is None  # throttled after it
    assert await w.maybe(newer, store, state, NOW.timestamp() + 2001 + bm.WATCH_SECONDS) is not None


async def test_the_watcher_alerts_on_a_recommendation_once_and_can_apply_it():
    store, state = MemoryStore(), MemoryState()
    cfg, raw = cfg_with(routing={"benchmark": {"auto_apply": True}})
    for r in rows("served", 3000, 0.1, 0.96, learned_level=2) + rows("down", 400, 0.1, 0.02, rng=random.Random(2)):
        await store.record_route_features({**r, "fv": "x", "text_features": [], "tags": []})
    w = bm.Watcher()
    out = await w.run(cfg, store, state, NOW.timestamp(), raw)
    assert out["recommendations"] and await store.active_config() is None  # no model in use that beat the rules: not applied
    assert sum(a["title"].startswith("Level recommendation") for a in await state.recent_alerts()) == 1
    w.reported = 0
    await w.run(cfg, store, state, NOW.timestamp() + 10, raw)
    assert sum(a["title"].startswith("Level recommendation") for a in await state.recent_alerts()) == 1  # once
    await store.save_router_model({"members": []}, {"better": True}, "test", active=True)
    w.reported = 0
    await w.run(cfg, store, state, NOW.timestamp() + 20, raw)
    saved = await store.active_config()
    assert saved["body"]["routing"]["level_map"] == {"coding:3": 2} and "outcome report" in saved["comment"]
    assert any(a["title"] == "Level moved by the outcome report" for a in await state.recent_alerts())


async def test_the_report_api_and_applying_from_the_console(gateway, client, monkeypatch):
    chat = {"model": "auto", "messages": [{"role": "user", "content": "hi"}]}
    assert (await client.post("/v1/chat/completions", json=chat, headers=auth(LEAD_KEY))).status_code == 200
    await gateway.drain()
    r = await client.get("/v1/reports/levels/outcomes", headers=auth(ADMIN_KEY))
    assert r.status_code == 200 and r.json()["recording"] is False and r.json()["requests"] >= 1
    row = r.json()["levels"][0]
    assert row["requests"] >= 1 and row["arms"]["served"]["n"] == 0 and r.json()["trend"][0]["requests"] >= 1
    lead = await client.get("/v1/reports/levels/outcomes", headers=auth(LEAD_KEY))
    assert lead.status_code == 200 and lead.json()["team"] == "default" and lead.json()["recommendations"] == []
    assert (await client.get("/v1/reports/levels/outcomes")).status_code in (401, 403)
    # Dismissing: admins only, kept as a setting (not a config version), and undone the same way.
    dismiss = {"task": "coding", "level": 3, "to": 2, "direction": "lower"}
    assert (await client.post("/v1/admin/levels/dismiss", json=dismiss, headers=auth(LEAD_KEY))).status_code in (401, 403)
    r = await client.post("/v1/admin/levels/dismiss", json=dismiss, headers=auth(ADMIN_KEY))
    assert r.status_code == 200 and r.json()["dismissed"]["coding:3"]["to"] == 2 and r.json()["dismissed"]["coding:3"]["by"]
    assert (await gateway.store.get_setting(bm.DISMISSED))["coding:3"]["direction"] == "lower"
    r = await client.post("/v1/admin/levels/dismiss", json={**dismiss, "undo": True}, headers=auth(ADMIN_KEY))
    assert r.status_code == 200 and r.json()["dismissed"] == {}
    if gateway.store.kind != "memory":
        return  # saving a version: the in-memory store is enough
    body = {"task": "coding", "level": 3, "to": 2}
    assert (await client.post("/v1/admin/levels/apply", json=body, headers=auth(ADMIN_KEY))).status_code == 409  # needs Postgres
    monkeypatch.setenv("T_ADMIN", ADMIN_KEY)
    doc = copy.deepcopy(TEST_CONFIG)
    doc["auth"]["static_keys"] = [{"key_env": "T_ADMIN", "user": "root", "team": "ops", "role": "admin"}]
    gateway.file_raw = doc
    gateway.store.kind = "postgres"
    try:
        r = await client.post("/v1/admin/levels/apply", json=body, headers=auth(ADMIN_KEY))
        assert r.status_code == 200, r.text
        assert r.json()["level_map"] == {"coding:3": 2} and r.json()["changed"] == ["coding:3 3→2"]
        assert (await client.post("/v1/admin/levels/apply", json=body, headers=auth(LEAD_KEY))).status_code in (401, 403)
    finally:
        gateway.store.kind = "memory"
