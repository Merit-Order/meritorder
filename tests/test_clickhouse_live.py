"""The ClickHouse copy of the ledger against real Postgres and ClickHouse: every report gives the same answer from both, the
copy is idempotent and repairs itself, the schema follows the ledger, and reports fall back to Postgres.

Runs when MERITORDER_TEST_CLICKHOUSE_URL and MERITORDER_TEST_DATABASE_URL are set (CI runs both services; set
MERITORDER_TEST_CLICKHOUSE_USER and MERITORDER_TEST_CLICKHOUSE_PASSWORD too if the server needs them). Each test uses a
ClickHouse database of its own and drops it afterwards. Locally these skip; test_clickhouse.py runs everywhere.
"""

from __future__ import annotations

import asyncio
import copy
import json
import math
import os
import random
import uuid
from datetime import UTC, datetime, timedelta

import httpx
import pytest
import yaml
from conftest import ADMIN_KEY, DATABASE_URL, LEAD_KEY, TEST_CONFIG, TEST_KEY, auth, real_store

from meritorder import cli
from meritorder.app import build_gateway, create_app
from meritorder.clickhouse import STATE_KEY, ClickHouse, Copier, column_types
from meritorder.config import ClickHouseConfig
from meritorder.state import MemoryState
from meritorder.store import LEDGER_FIELDS, SAVINGS_FIELDS, SUM_FIELDS, collect_answers, complete_row

CH_URL = os.environ.get("MERITORDER_TEST_CLICKHOUSE_URL")
pytestmark = pytest.mark.skipif(not (CH_URL and DATABASE_URL),
                                reason="set MERITORDER_TEST_CLICKHOUSE_URL and MERITORDER_TEST_DATABASE_URL")
NOW = datetime.now(UTC)


def ch_config(**kw) -> ClickHouseConfig:
    return ClickHouseConfig(url=CH_URL or "http://localhost:8123", database=f"motest_{uuid.uuid4().hex[:10]}",
                            user=os.environ.get("MERITORDER_TEST_CLICKHOUSE_USER", "default"),
                            auth={"value_env": "MERITORDER_TEST_CLICKHOUSE_PASSWORD"}, **kw)


async def drop(cfg: ClickHouseConfig) -> None:
    ch = ClickHouse(cfg)
    await ch.command(f"DROP DATABASE IF EXISTS `{cfg.database}`")
    await ch.close()


@pytest.fixture
async def pg():
    store = await real_store()
    yield store
    await store.close()


@pytest.fixture
async def ch_cfg():
    cfg = ch_config()
    yield cfg
    await drop(cfg)


def seeded_rows(n: int = 600, seed: int = 47) -> list[dict]:
    """Ledger rows over the last 40 days, with every kind of value a report groups or filters on: missing models, floors
    and upstream ids, empty and missing tags, errors, free upgrades, unknown statuses, odd characters in names, batch rows."""
    rnd = random.Random(seed)
    rows = []
    people = [("alice", "default"), ("bob", "default"), ("lee", "default"), ("carol", "ops"), ("root", "ops"), ("d\\a'n\tx", "ops")]
    for i in range(n):
        user, team = rnd.choice(people)
        rows.append(complete_row({
            "id": uuid.uuid4(), "ts": NOW - timedelta(minutes=rnd.randint(1, 40 * 24 * 60), microseconds=rnd.randint(0, 999_999)),
            "user_id": user, "team": team, "key_id": rnd.choice(["key_a", None]), "session_id": rnd.choice(["s1", "s2", None]),
            "tags": rnd.choice([{}, {"project": "apollo"}, {"project": ""}, {"project": "gemini", "ticket": "PLAT-1"}]),
            "client_format": rnd.choice(["openai", "anthropic"]), "requested_model": rnd.choice(["auto", "big", None]),
            "floor": rnd.choice(["coding:3", "general:2", None]), "route_reason": rnd.choice(["classified coding:3", None]),
            "served_model": rnd.choice(["big", "small", "small", None]), "upstream_model": rnd.choice(["big-1", "small-1", None]),
            "backend": rnd.choice(["mock-anthropic", "mock-openai"]), "backend_kind": "mock",
            "status": rnd.choice([200, 200, 200, 429, 500, None]), "error": rnd.choice([None, "upstream failed"]),
            "stream": rnd.choice([True, False]), "upgraded": rnd.choice([True, False, False, None]), "proxy_cache": rnd.choice([True, False]),
            "input_tokens": rnd.randint(0, 200_000), "cache_read_tokens": rnd.randint(0, 50_000), "cache_write_tokens": rnd.randint(0, 9000),
            "output_tokens": rnd.randint(0, 9000), "latency_ms": rnd.choice([rnd.randint(50, 90_000), None]),
            "ttft_ms": rnd.choice([rnd.randint(50, 9000), None]),
            **({"service_tier": "batch", "batch_id": f"batch_{i % 2}"} if i % 3 == 0 else {}),  # batch rows too
            **{f: rnd.random() * rnd.choice([0.001, 0.1, 3.0])
               for f in ("actual_cost", "served_api_cost", "baseline_cost", "list_cost", *SAVINGS_FIELDS)},  # every cause
        }))
    return rows


async def seed(pg, rows) -> None:
    for r in rows:
        await pg.record(r)
    await pg.rollup(full=True)


def same(a, b, path="") -> list[str]:
    """Where two report answers differ. Floats that the API doesn't round (GPU what-if, break-even) may differ in their
    last bits, because the two databases add the same numbers in a different order."""
    if isinstance(a, float) or isinstance(b, float):
        ok = isinstance(a, (int, float)) and isinstance(b, (int, float)) and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
        return [] if ok else [f"{path}: {a!r} != {b!r}"]
    if isinstance(a, dict) and isinstance(b, dict):
        return [d for k in sorted(set(a) | set(b)) for d in same(a.get(k), b.get(k), f"{path}.{k}")]
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: {len(a)} items != {len(b)}"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in same(x, y, f"{path}[{i}]")]
    if a == b:
        return []
    if isinstance(a, str) and isinstance(b, str):  # a JSON line or CSV: where they part
        at = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        return [f"{path}: …{a[max(0, at - 60):at + 60]!r} != …{b[max(0, at - 60):at + 60]!r}"]
    return [f"{path}: {a!r} != {b!r}"]


async def copy_all(pg, cfg: ClickHouseConfig):
    pg.use_clickhouse(cfg)
    copier = Copier(pg, pg.clickhouse)
    start = (NOW - timedelta(days=41)).replace(hour=0, minute=0, second=0, microsecond=0)
    res, note = await copier.backfill(start, NOW + timedelta(days=1))
    assert note is None
    await copier.run_once()
    return res


# ---------------------------------------------------------------------------- the same answers


async def test_every_ledger_query_matches_postgres(pg, ch_cfg):
    rows = seeded_rows()
    await seed(pg, rows)
    res = await copy_all(pg, ch_cfg)
    assert res.copied == len(rows) and not res.repaired and not res.ahead
    ledger = pg.clickhouse.ledger
    pg.clickhouse, reporting = None, pg.clickhouse  # Postgres alone answers `pg.*` below

    problems = []
    for days in (2, 6, 8, 31, 45):
        start, end = NOW - timedelta(days=days), NOW + timedelta(hours=1)
        for group_by in ("user", "team", "model", "backend", "day", "hour", "month", "none", "tag:project", "tag:ticket", "tag:nope"):
            for scope in ({}, {"user_id": "alice"}, {"team": "ops"}, {"user_id": "d\\a'n\tx", "team": "ops"}):
                want = await pg.summary(start, end, group_by, **scope)
                got = await ledger.summary(start, end, group_by, **scope)
                problems += same(got, want, f"summary {days}d {group_by} {scope}")
        backends = ["mock-anthropic", "mock-openai", "nope"]
        problems += same(sorted_rows(await ledger.ledger_costs(start, end, backends)), sorted_rows(await pg.ledger_costs(start, end, backends)),
                         f"ledger_costs {days}d")
        problems += same(sorted_rows(await ledger.demand_by_hour(start)), sorted_rows(await pg.demand_by_hour(start)), f"demand_by_hour {days}d")
        for team in (None, "default"):
            want = [r async for r in pg.export_rows(start, end, team)]
            got = [r async for r in ledger.export_rows(start, end, team)]
            problems += same([json.dumps(r, default=str) for r in got], [json.dumps(r, default=str) for r in want], f"export {days}d {team}")
    assert not problems, f"{len(problems)} differences, the first: " + "; ".join(problems[:10])
    await reporting.close()


def sorted_rows(rows: list[dict]) -> list[dict]:
    """Rows of a query without an order, sorted by everything but their floats (which may differ in the last bit)."""
    return sorted(rows, key=lambda r: json.dumps({k: v for k, v in r.items() if not isinstance(v, float)}, default=str, sort_keys=True))


async def test_every_report_the_api_serves_matches_postgres(pg, ch_cfg, cfg, monkeypatch):
    """The whole reports API, once with Postgres answering (reads: none) and once with ClickHouse answering everything it
    can (reads: all): the same JSON, CSV and JSON lines, apart from which store answered."""
    from meritorder import gpu_sizing, reports

    frozen = datetime.now(UTC)

    class Frozen(datetime):  # both passes see the same "now": the same windows ("last 30 days") and averages
        @classmethod
        def now(cls, tz=None):
            return frozen if tz else frozen.replace(tzinfo=None)

    monkeypatch.setattr(reports, "datetime", Frozen)
    await seed(pg, seeded_rows(400, seed=7))
    await copy_all(pg, ch_cfg)
    gw = build_gateway(cfg, store=pg, state=MemoryState())
    gw.file_raw = copy.deepcopy(TEST_CONFIG)
    fresh = datetime.now(UTC).isoformat(timespec="seconds")
    await pg.set_setting("gpu_catalog", {"offers": [{"cloud": "aws", "accelerator": "L4", "count": 1,
                                                     "on_demand": {"usd_per_hour": 0.8, "region": "us-east-1", "instance_type": "g6"}}],
                                         "fetched_at": fresh, "catalog_updated": {"aws": fresh}, "skypilot": "0.14.2", "clouds": ["aws"]})
    await pg.set_setting("model_sizes", {"fetched_at": fresh, "version": gpu_sizing.SIZES_VERSION, "models": {}})
    fin = (await pg.create_key("fin", "ops", "finance", name="finance"))[1]
    month = NOW.replace(day=1).date()
    calls = []
    for key in (ADMIN_KEY, LEAD_KEY, TEST_KEY, fin):
        for start, end in (((NOW - timedelta(days=3)).date(), None), ((NOW - timedelta(days=30)).date(), None), (month, None),
                           ((NOW - timedelta(days=40)).date(), (NOW - timedelta(days=20)).date())):
            dates = {"start": start.isoformat(), **({"end": end.isoformat()} if end else {})}
            for group_by in ("user", "team", "model", "backend", "day", "hour", "month", "none", "tag:project"):
                calls.append(("/v1/reports/summary", key, {**dates, "group_by": group_by}))
            calls.append(("/v1/reports/summary", key, {**dates, "group_by": "user", "team": "ops"}))
            calls.append(("/v1/reports/summary.csv", key, {**dates, "group_by": "tag:project"}))
            calls.append(("/v1/reports/export.jsonl", key, dates))
            calls.append(("/v1/reports/bills", key, {**dates, "by": "month"}))
        calls += [("/v1/reports/me", key, {}), ("/v1/reports/requests", key, {"limit": 1000}), ("/v1/reports/levels", key, {}),
                  ("/v1/reports/gpu", key, {}), ("/v1/reports/gpu/whatif", key, {"model": "small", "gpu_usd_per_hour": 2, "days": 30}),
                  ("/v1/reports/gpu/breakeven", key, {"days": 30}), ("/v1/reports/router", key, {})]

    async def answers(reads: str) -> list:
        pg.use_clickhouse(ch_cfg.model_copy(update={"reads": reads}))
        out = []
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(gateway=gw)), base_url="http://test") as client:
            for path, key, params in calls:
                r = await client.get(path, params=params, headers=auth(key))
                body = r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text
                who = r.headers.get("x-meritorder-answered-by")
                if isinstance(body, dict):
                    who = body.pop("answered_by", who)
                out.append((r.status_code, body, who))
        return out

    pg_answers, ch_answers = await answers("none"), await answers("all")
    problems, read_ch = [], set()
    for (path, key, params), (s1, b1, w1), (s2, b2, w2) in zip(calls, pg_answers, ch_answers):
        assert s1 == s2, (path, params, s1, s2)
        assert w1 in (None, "postgres"), (path, w1)
        if w2 and "clickhouse" in w2:
            read_ch.add(path)
        problems += same(b2, b1, f"{path} {key[-9:]} {params}")
    assert not problems, f"{len(problems)} differences, the first: " + "; ".join(problems[:10])
    # Every ledger sum was compared, each cause of savings among them (savings_compression_model too): none is missing or zero.
    summary = next(b for (path, key, _), (_, b, _) in zip(calls, pg_answers) if path == "/v1/reports/summary" and key == ADMIN_KEY)
    assert all(any(r[f] for r in summary["rows"]) for f in SUM_FIELDS), summary["rows"][:1]
    # Everything that reads ledger rows read them from ClickHouse; the newest rows and budgets come from Postgres.
    assert read_ch >= {"/v1/reports/summary", "/v1/reports/summary.csv", "/v1/reports/export.jsonl", "/v1/reports/bills",
                       "/v1/reports/me", "/v1/reports/gpu/whatif", "/v1/reports/gpu/breakeven"}
    assert "/v1/reports/requests" not in read_ch
    assert "/v1/reports/router" not in read_ch  # the learned router's request features stay in Postgres only
    await gw.drain()


async def test_large_reports_read_clickhouse_by_default(pg, ch_cfg):
    await seed(pg, seeded_rows(50))
    await copy_all(pg, ch_cfg)
    seen = collect_answers()
    await pg.summary(NOW - timedelta(days=30), NOW, "team")
    assert seen == {"clickhouse"}
    seen = collect_answers()
    await pg.summary(NOW - timedelta(days=3), NOW, "team")  # a short range: Postgres, which has the newest rows
    assert seen == {"postgres"}
    seen = collect_answers()
    await pg.summary(NOW - timedelta(days=3), NOW, "tag:project")  # a tag breakdown scans rows: ClickHouse
    assert seen == {"clickhouse"}
    # Before the range ClickHouse holds, Postgres answers.
    await pg.patch_setting(STATE_KEY, {"from": (NOW - timedelta(days=10)).isoformat()})
    pg.clickhouse.forget_state()
    seen = collect_answers()
    await pg.summary(NOW - timedelta(days=30), NOW, "team")
    assert seen == {"postgres"}


# ---------------------------------------------------------------------------- the copy


async def test_copy_is_idempotent_repairs_duplicates_and_follows_new_rows(pg, ch_cfg):
    rows = seeded_rows(80)
    await seed(pg, rows)
    res = await copy_all(pg, ch_cfg)
    rep = pg.clickhouse
    copier = Copier(pg, rep)
    assert res.copied == 80
    again, _ = await copier.backfill(NOW - timedelta(days=41), NOW + timedelta(days=1))
    assert again.copied == 0 and again.repaired == [] and again.ahead == []

    async def counts():
        r = (await rep.client.query(f"SELECT count() AS n, uniqExact(id) AS u FROM `{ch_cfg.database}`.requests"))[0]
        roll = (await rep.client.query(f"SELECT sum(requests) AS n FROM `{ch_cfg.database}`.usage_hourly"))[0]
        return int(r["n"]), int(r["u"]), int(roll["n"])

    assert await counts() == (80, 80, 80)
    # New rows, one landing late (stamped two hours ago): the next cycle copies both.
    late = [complete_row({"id": uuid.uuid4(), "ts": NOW - timedelta(minutes=m), "user_id": "alice", "team": "default", "backend": "mock-openai",
                          "status": 200, "actual_cost": 0.5}) for m in (1, 120)]
    for r in late:
        await pg.record(r)
    assert (await copier.run_once()).copied == 2
    assert await counts() == (82, 82, 82)
    # A row sent twice (an insert that timed out but landed): its hour is copied again whole.
    await rep.ledger.insert([late[0]])
    assert (await counts())[:2] == (83, 82)
    res = await copier.run_once()
    assert len(res.repaired) == 1 and await counts() == (82, 82, 82)
    # A row Postgres no longer has: ClickHouse keeps it, and says so.
    async with pg.pool.acquire() as c:
        await c.execute("DELETE FROM requests WHERE id = $1", late[1]["id"])
    res = await copier.run_once()
    assert len(res.ahead) == 1 and res.copied == 0
    state = await pg.get_setting(STATE_KEY)
    assert state["from"] and state["through"] and state["error"] is None and state["ahead"] == res.ahead


async def test_a_second_copy_skips_hours_the_first_holds(pg, ch_cfg):
    await seed(pg, [complete_row({"ts": NOW - timedelta(minutes=5), "user_id": "alice", "team": "default", "status": 200})])
    pg.use_clickhouse(ch_cfg)
    rep = pg.clickhouse
    await rep.ensure_schema()
    hour = NOW.replace(minute=0, second=0, microsecond=0) - (timedelta(hours=1) if NOW.minute < 5 else timedelta())
    async with pg.ledger_hour(hour, wait=False) as held:
        assert held is not None
        res = await Copier(pg, rep).sync(hour, hour + timedelta(hours=1))
        assert res.busy == [hour.isoformat()] and res.copied == 0
    assert (await Copier(pg, rep).sync(hour, hour + timedelta(hours=1))).copied == 1


async def test_a_new_clickhouse_doesnt_inherit_the_progress_of_the_last_one(pg, ch_cfg):
    """The copy's progress is in Postgres; the ClickHouse it described may be gone (a new server, another database, a lost
    volume). The first cycle against an empty one starts over, so long reports keep reading Postgres."""
    await seed(pg, seeded_rows(30))
    await pg.set_setting(STATE_KEY, {"from": (NOW - timedelta(days=60)).isoformat(), "through": NOW.isoformat()})
    pg.use_clickhouse(ch_cfg)  # a database with no tables yet
    await Copier(pg, pg.clickhouse).run_once()
    pg.clickhouse.forget_state()
    seen = collect_answers()
    rows = await pg.summary(NOW - timedelta(days=41), NOW + timedelta(days=1), "team")
    assert seen == {"postgres"} and sum(r["requests"] for r in rows) == 30


async def test_schema_follows_the_ledger_and_the_ttls(ch_cfg, pg):
    pg.use_clickhouse(ch_cfg.model_copy(update={"ttl_days": 90, "rollup_ttl_days": 800}))
    rep = pg.clickhouse
    await rep.ensure_schema()
    cols = await rep.client.query("SELECT name, type FROM system.columns WHERE database = {db:String} AND table = 'requests'",
                                  {"db": ch_cfg.database})
    assert [(c["name"], c["type"]) for c in cols] == column_types()
    tables = {t["name"]: t["create_table_query"] for t in await rep.client.query(
        "SELECT name, create_table_query FROM system.tables WHERE database = {db:String}", {"db": ch_cfg.database})}
    assert "TTL toDateTime(ts) + toIntervalDay(90)" in tables["requests"] and "toIntervalDay(800)" in tables["usage_hourly"]
    assert "PARTITION BY toYYYYMM(ts)" in tables["requests"] and rep.ledger.mv_name() in tables
    # A table made by an older version lacks a newer column: it's added. And the TTLs can be removed again.
    await rep.client.command(f"ALTER TABLE `{ch_cfg.database}`.requests DROP COLUMN route_reason")
    pg.use_clickhouse(ch_cfg)
    changes = await pg.clickhouse.ensure_schema()
    assert {"added requests.route_reason", "requests keeps everything", "usage_hourly keeps everything"} <= set(changes)
    assert await pg.clickhouse.ledger.migrate() == []  # nothing left to do


async def test_reports_fall_back_to_postgres_when_clickhouse_is_down(pg):
    await seed(pg, seeded_rows(30))
    cfg = ch_config().model_copy(update={"url": "http://127.0.0.1:9", "timeout_seconds": 2})
    pg.use_clickhouse(cfg)
    await pg.set_setting(STATE_KEY, {"from": (NOW - timedelta(days=60)).isoformat(), "through": NOW.isoformat()})
    seen = collect_answers()
    rows = await pg.summary(NOW - timedelta(days=41), NOW + timedelta(days=1), "team")
    assert seen == {"postgres"} and sum(r["requests"] for r in rows) == 30
    assert "ClickHouse failed just now" in await pg.clickhouse.why_not(NOW - timedelta(days=30))
    exported = [r async for r in pg.export_rows(NOW - timedelta(days=41), NOW + timedelta(days=1))]
    assert len(exported) == 30
    d = await pg.clickhouse.describe()
    assert d["ok"] is False and "can't reach ClickHouse" in d["error"] and d["answering"] is False
    with pytest.raises(Exception):
        await Copier(pg, pg.clickhouse).run_once()
    assert "can't reach ClickHouse" in (await pg.get_setting(STATE_KEY))["error"]


# ---------------------------------------------------------------------------- the worker, the CLI, the console


async def test_worker_copies_new_rows(pg, ch_cfg, cfg, monkeypatch):
    from meritorder import worker

    class Idle:  # no live prices, scores or bills here: nothing leaves the machine
        async def maybe(self, *args, **kw):
            return None

    for name in ("PriceRefresher", "QualityRefresher", "BillRefresher"):
        monkeypatch.setattr(worker, name, Idle)
    await seed(pg, seeded_rows(20))
    monkeypatch.setenv("MERITORDER_DATABASE_URL", DATABASE_URL)
    monkeypatch.delenv("MERITORDER_REDIS_URL", raising=False)
    cfg.reporting.clickhouse = ch_cfg
    await worker.run_worker(once=True, cfg=cfg)
    state = await pg.get_setting(STATE_KEY)
    assert state["from"] and state["through"] and state["error"] is None  # the last 3 hours were checked


def test_cli_backfill_and_status(ch_cfg, tmp_path, monkeypatch, capsys):
    rows = seeded_rows(40)

    async def prepare():
        store = await real_store()
        await seed(store, rows)
        await store.close()

    asyncio.run(prepare())
    raw = copy.deepcopy(TEST_CONFIG)
    raw["reporting"] = {"clickhouse": ch_cfg.model_dump(exclude_none=True)}
    path = tmp_path / "meritorder.yaml"
    path.write_text(yaml.safe_dump(raw))
    monkeypatch.setenv("MERITORDER_DATABASE_URL", DATABASE_URL)
    cli.main(["clickhouse", "backfill", "--config", str(path)])
    out = capsys.readouterr().out
    assert "done: 40 rows copied" in out
    cli.main(["clickhouse", "backfill", "--config", str(path)])
    assert "done: 0 rows copied" in capsys.readouterr().out
    cli.main(["clickhouse", "status", "--config", str(path)])
    out = capsys.readouterr().out
    assert "reachable: version" in out and "ledger rows: 40" in out and "copy: every row in the ledger" in out  # from its first day


async def test_console_shows_the_reporting_backend(pg, ch_cfg, cfg):
    await seed(pg, seeded_rows(10))
    await copy_all(pg, ch_cfg)
    cfg.reporting.clickhouse = ch_cfg
    gw = build_gateway(cfg, store=pg, state=MemoryState())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(gateway=gw)), base_url="http://test") as client:
        d = (await client.get("/v1/admin/reporting", headers=auth(ADMIN_KEY))).json()
    assert d["store"] == "postgres" and d["postgres"]["ok"] and d["configured"]
    c = d["clickhouse"]
    assert c["ok"] and c["version"] and c["tables"]["requests"]["rows"] == 10 and c["answering"] and c["copy"]["behind_seconds"] < 60
    await gw.drain()


def test_every_ledger_column_has_a_clickhouse_type():
    assert [f for f, _ in column_types()] == LEDGER_FIELDS and set(SUM_FIELDS) <= set(LEDGER_FIELDS)
