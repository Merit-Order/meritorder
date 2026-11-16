"""Contract tests: the in-memory and the Redis/Postgres implementations behave the same.

Each test runs once per backing (see conftest); the real ones run when
MERITORDER_TEST_REDIS_URL and MERITORDER_TEST_DATABASE_URL are set.
"""

import asyncio
import time
import uuid
from datetime import UTC, datetime, timedelta

from conftest import ledger_row

# ===================================================================== state


async def test_sessions_spend_and_json(state):
    assert await state.ping()
    assert await state.get_session("s1") is None
    await state.put_session("s1", {"model": "big", "turns": 2}, 60)
    assert await state.get_session("s1") == {"model": "big", "turns": 2}

    assert await state.get_spend("m:user:a") is None
    await state.seed_spend("m:user:a", 1.5)
    await state.add_spend(["m:user:a", "m:team:t"], 0.25)
    assert await state.get_spend("m:user:a") == 1.75 and await state.get_spend("m:team:t") == 0.25

    assert await state.get_json("k") is None
    await state.set_json("k", {"a": [1, 2]})
    await state.set_json("short", 1, ttl=1)
    assert await state.get_json("k") == {"a": [1, 2]} and await state.get_json("short") == 1

    assert await state.get_config_token() is None
    await state.set_config_token("v7")
    assert await state.get_config_token() == "v7"


async def test_admit_reads_everything_in_one_call(state):
    await state.seed_spend("m:org:all", 3.0)
    await state.put_session("s1", {"model": "small"}, 60)
    await state.add_spend(["session:s1"], 0.5)
    await state.set_utilization("gpu", 0.7)
    await state.publish_inflight("gw-1", {"gpu": 2})
    rates = [("rpm:user:a", 60, 1), ("tpm:user:a", 60, 500)]
    pre = await state.admit(["m:org:all", "m:user:a"], rates, "s1")
    assert pre["spend"] == {"m:org:all": 3.0, "m:user:a": None}
    assert pre["rates"] == {"rpm:user:a": 1, "tpm:user:a": 500}
    assert pre["session"] == {"model": "small"} and pre["session_spend"] == 0.5
    assert pre["util"] == {"gpu": 0.7} and pre["inflight"] == {"gw-1": {"gpu": 2}}
    again = await state.admit([], rates, None)
    assert again["rates"] == {"rpm:user:a": 2, "tpm:user:a": 1000} and again["session"] is None and again["session_spend"] is None
    assert await state.rate_hit("rpm:user:a", 60) == 3


async def test_record_request_feeds_events_health_and_spend(state):
    await state.seed_spend("m:user:a", 1.0)
    await state.record_request({"id": "r1", "model": "big"}, ("anthropic", True, 120, 40), ["m:user:a", "m:team:t", "session:s"], 0.01)
    await state.record_request({"id": "r2"}, ("anthropic", False, 300, None), ["m:user:a"], 0.0)
    await state.record_request(None, None, [], 0.0)
    assert [e["id"] for e in await state.recent_events(10)] == ["r2", "r1"]
    h = (await state.backend_health(["anthropic", "none"]))["anthropic"]
    assert h["requests"] == 2 and h["errors"] == 1
    assert await state.get_spend("m:user:a") == 1.01 and await state.get_spend("session:s") == 0.01
    assert await state.get_spend("m:team:t") is None  # a month's counter it didn't find is seeded from the ledger instead


async def test_events_alerts_health_and_utilization(state):
    for i in range(3):
        await state.push_event({"i": i})
        await state.push_alert({"title": f"a{i}"})
    assert [e["i"] for e in await state.recent_events(2)] == [2, 1]
    assert [a["title"] for a in await state.recent_alerts(5)] == ["a2", "a1", "a0"]
    await state.record_backend("b", True, 100, 20)
    await state.record_backend("b", True, 300, None)
    h = await state.backend_health(["b"])
    assert h["b"]["requests"] == 2 and h["b"]["errors"] == 0
    await state.set_utilization("gpu", 0.4)
    assert await state.utilization() == {"gpu": 0.4}


async def test_gateways_heartbeat_and_inflight(state):
    await state.heartbeat("gw-a")
    await state.publish_inflight("gw-b", {"gpu": 1})
    assert await state.live_gateways() == 2
    assert (await state.admit([], [], None))["inflight"] == {"gw-b": {"gpu": 1}}
    await state.publish_inflight("gw-b", {"gpu": 0})
    assert (await state.admit([], [], None))["inflight"] == {}


# ===================================================================== store


async def test_users_and_keys(store):
    assert await store.ping()
    await store.upsert_user("alice", "eng", "user", email="Alice@Example.com", display_name="Alice")
    await store.upsert_user("bob", "ops", "team_lead")
    assert (await store.get_user("alice"))["team"] == "eng"
    assert (await store.find_user_by_email("alice@example.com"))["id"] == "alice"
    await store.update_user("alice", team="ops", managed="admin")
    await store.upsert_user("alice", "eng", "user")  # an IdP login doesn't undo an admin's change
    assert (await store.get_user("alice"))["team"] == "ops"
    await store.update_user("bob", disabled=True)
    assert (await store.get_user("bob"))["disabled_at"] is not None
    assert {u["id"] for u in await store.list_users(team="ops")} == {"alice", "bob"}
    assert await store.update_user("nobody", team="x") is None

    key_id, key = await store.create_key("alice", "ops", "user", name="laptop", tags={"project": "x"})
    from meritorder.store import hash_key

    found = await store.find_key(hash_key(key))
    assert found["user_id"] == "alice" and found["tags"] == {"project": "x"}
    assert [k["id"] for k in await store.list_keys(user_id="alice")] == [key_id]
    assert [k["id"] for k in await store.list_keys(team="ops")] == [key_id]
    assert (await store.get_key(key_id))["name"] == "laptop"
    assert await store.revoke_key(key_id)
    assert (await store.find_key(hash_key(key)))["revoked_at"] is not None  # auth refuses it


async def test_scim_groups(store):
    await store.save_scim_group({"id": "g1", "display_name": "Eng", "members": ["alice"]})
    assert (await store.scim_group("g1"))["display_name"] == "Eng"
    assert [g["id"] for g in await store.scim_groups()] == ["g1"]
    assert await store.delete_scim_group("g1") and not await store.delete_scim_group("g1")
    assert await store.scim_group("g1") is None


async def test_ledger_spend_recent_and_export(store):
    now = datetime.now(UTC)
    await store.record(ledger_row(user_id="a", team="eng", served_model="big", actual_cost=1.0, ts=now - timedelta(minutes=2)))
    await store.record(ledger_row(user_id="b", team="eng", served_model="small", actual_cost=0.5, ts=now - timedelta(minutes=1)))
    await store.record(ledger_row(user_id="c", team="ops", served_model="big", actual_cost=2.0, status=500, ts=now))
    hour_ago = now - timedelta(hours=1)
    assert await store.spend_since("team", "eng", hour_ago) == 1.5
    assert await store.spend_since("user_id", "c", hour_ago) == 2.0
    assert await store.spend_all_since(hour_ago) == 3.5
    assert [r["user_id"] for r in await store.recent(10)] == ["c", "b", "a"]
    assert [r["user_id"] for r in await store.recent(10, team="eng")] == ["b", "a"]
    assert [r["user_id"] for r in await store.recent(10, user_id="a")] == ["a"]
    rows = [r async for r in store.export_rows(hour_ago, now + timedelta(seconds=1))] \
        if hasattr(store.export_rows(hour_ago, now), "__aiter__") else await store.export_rows(hour_ago, now + timedelta(seconds=1))
    assert len(rows) == 3


async def test_summary_groupings_raw_and_rollup(store):
    now = datetime.now(UTC)
    for user, model, cost, tags in [("a", "big", 1.0, {"project": "x"}), ("a", "small", 0.25, {}), ("b", "big", 2.0, {"project": "y"})]:
        await store.record(ledger_row(user_id=user, team="eng", served_model=model, backend="mock", actual_cost=cost,
                                      baseline_cost=cost * 2, savings_routing=cost, tags=tags, ts=now))
    start, end = now - timedelta(days=1), now + timedelta(hours=1)
    by_user = {r["key"]: r for r in await store.summary(start, end, "user")}
    assert by_user["a"]["requests"] == 2 and by_user["a"]["actual_cost"] == 1.25 and by_user["b"]["actual_cost"] == 2.0
    assert by_user["a"]["savings_total"] == 1.25
    by_tag = {r["key"]: r["requests"] for r in await store.summary(start, end, "tag:project")}
    assert by_tag == {"x": 1, "y": 1, "(untagged)": 1}
    assert [r["requests"] for r in await store.summary(start, end, "none")] == [3]
    assert sum(r["requests"] for r in await store.summary(start, end, "day")) == 3
    assert {r["key"] for r in await store.summary(start, end, "model")} == {"big", "small"}
    assert [r["key"] for r in await store.summary(start, end, "user", user_id="b")] == ["b"]
    # Ranges over a week read the hourly rollup (Postgres) once the worker has refreshed it.
    await store.rollup()
    long = {r["key"]: r["actual_cost"] for r in await store.summary(now - timedelta(days=30), end, "team")}
    assert long == {"eng": 3.25}
    assert await store.rollup(full=True) >= 1 or store.kind == "memory"


async def test_a_full_rollup_covers_every_month(store):
    """The full rollup (an empty usage_hourly, or `worker --backfill`) runs a month at a time: every month's rows are
    there, the newest month's and a row stamped ahead of this clock included."""
    now = datetime.now(UTC)
    for days, cost in ((95, 1.0), (40, 2.0), (0, 4.0), (-2 / 24, 8.0)):
        await store.record(ledger_row(team="eng", served_model="big", backend="mock", actual_cost=cost, ts=now - timedelta(days=days)))
    rows = await store.rollup(full=True)
    assert rows == 4 or store.kind == "memory"
    by_month = {str(r["key"]): r["actual_cost"] for r in await store.summary(now - timedelta(days=100), now + timedelta(days=1), "month")}
    assert sum(by_month.values()) == 15.0 and len(by_month) >= 3
    assert await store.rollup(full=True) == rows  # again: the same hours, updated in place


async def test_demand_queries(store):
    now = datetime.now(UTC)
    await store.record(ledger_row(served_model="big", served_api_cost=1.0, input_tokens=1000, output_tokens=100, ts=now))
    await store.record(ledger_row(served_model="big", served_api_cost=5.0, upgraded=True, ts=now))  # free upgrade: not demand
    await store.record(ledger_row(served_model="big", served_api_cost=7.0, status=500, ts=now))  # failed
    since = now - timedelta(hours=1)
    assert await store.needed_demand("big", since) >= 1.0
    rows = await store.demand_rows(since)
    assert sum(r["cost"] for r in rows) == 1.0 and sum(r["tokens"] for r in rows) == 1100
    by_hour = await store.demand_by_hour(since)
    assert by_hour and sum(r["cost"] for r in by_hour) == 1.0


async def test_config_versions_and_settings(store):
    assert await store.active_config() is None
    v1 = await store.save_config({"teams": {}}, "root", "first")
    v2 = await store.save_config({"teams": {"eng": {}}}, "root", None)
    assert (v1, v2) == (1, 2)
    assert (await store.active_config())["id"] == 2
    assert [v["id"] for v in await store.config_versions()] == [2, 1]
    assert (await store.get_config(1))["body"] == {"teams": {}} and await store.get_config(99) is None
    assert await store.get_setting("gpu_controls") is None
    await store.set_setting("gpu_controls", {"paused": True})
    await store.set_setting("gpu_controls", {"paused": False})
    assert await store.get_setting("gpu_controls") == {"paused": False}


async def test_gpu_usage_and_events(store):
    hour = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    await store.add_gpu_usage("gpu-a", hour, 1800, 4.0)
    await store.add_gpu_usage("gpu-a", hour, 1800, 4.0)  # same hour adds up
    await store.add_gpu_usage("gpu-b", hour - timedelta(days=40), 3600, 9.0)
    assert await store.gpu_spend_since(hour - timedelta(days=1)) == {"gpu-a": 8.0}
    daily = await store.gpu_daily(hour - timedelta(days=1))
    assert sum(d["cost"] for d in daily) == 8.0
    await store.gpu_event("stop", "gpu-a", "root", {"reason": "cost"})
    await asyncio.sleep(0.01)
    await store.gpu_event("start", "gpu-b", "controller")
    events = await store.gpu_events(10)
    assert [e["kind"] for e in events] == ["start", "stop"] and events[1]["detail"] == {"reason": "cost"}
    assert [e["kind"] for e in await store.gpu_events(10, backend="gpu-a")] == ["stop"]


async def test_ledger_rows_get_defaults(store):
    """A partial row (e.g. written by a job, not a request) still lands, with schema defaults."""
    await store.record({"user_id": "a", "team": "eng", "status": 200, "actual_cost": 0.1})
    [row] = await store.recent(1)
    assert row["input_tokens"] == 0 and row["tags"] == {} and row["id"] and row["ts"]
    uuid.UUID(str(row["id"]))
    assert abs(datetime.fromisoformat(row["ts"]).timestamp() - time.time()) < 60


async def test_shutdown_closes_retired_backends_without_waiting(gateway):
    """After a config change the old backends stay open 5 minutes for running streams; shutdown mustn't wait for that."""
    closed = []
    for b in gateway.backends.values():
        b.close = (lambda name: (lambda: closed.append(name) or asyncio.sleep(0)))(b.name)
    old = set(gateway.backends)
    await gateway.apply_config(gateway.cfg, 2)
    t0 = time.monotonic()
    await asyncio.wait_for(gateway.drain(), 5)
    assert time.monotonic() - t0 < 1 and set(closed) == old
