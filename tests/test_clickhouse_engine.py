"""The ClickHouse SQL against a real ClickHouse engine, in-process: chdb (embedded ClickHouse) answers the HTTP client's
requests, so the schema, the materialized view, the report queries and the copy run for real without a server.

chdb isn't a dependency (it's large); run these with `uv run --with chdb python -m pytest tests/test_clickhouse_engine.py`. They
skip without it. test_clickhouse_live.py runs the same code against real Postgres and a ClickHouse server, and compares
every report with Postgres's; here the Postgres side is the in-memory ledger (test_clickhouse.Source), compared where its
answers are the same as Postgres's.
"""

from __future__ import annotations

import random
import re
import uuid
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from test_clickhouse import Source

from meritorder.clickhouse import STATE_KEY, Copier, Reporting
from meritorder.config import ClickHouseConfig
from meritorder.store import LEDGER_FIELDS, SAVINGS_FIELDS, SUM_FIELDS, complete_row

chdb_session = pytest.importorskip("chdb.session", reason="embedded ClickHouse: uv run --with chdb python -m pytest tests/test_clickhouse_engine.py")
NOW = datetime.now(UTC)


INSERT = re.compile(r"^INSERT INTO (`(\w+)`\.(\w+)) \((.*)\) FORMAT RowBinary$")


class Chdb(httpx.AsyncBaseTransport):
    """ClickHouse's HTTP interface, answered by chdb: the SQL in the body and the parameters. chdb takes SQL as text, so a
    RowBinary insert's body goes through a file: INSERT ... SELECT FROM file(..., RowBinary) with the table's own types."""

    def __init__(self, session, folder):
        self.session, self.folder = session, folder

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        body = await request.aread()
        values = {k[6:]: v for k, v in params.items() if k.startswith("param_")}
        sql, fmt = (params.get("query") or body.decode()), "CSV"
        if sql.endswith("\nFORMAT JSONEachRow"):
            sql, fmt = sql[: -len("\nFORMAT JSONEachRow")], "JSONEachRow"
        try:
            if m := INSERT.match(sql):
                table, db, name, cols = m.groups()
                types = {r.split("\t")[0]: r.split("\t")[1] for r in str(self.session.query(
                    f"SELECT name, type FROM system.columns WHERE database = '{db}' AND table = '{name}'", "TSVRaw")).splitlines()}
                path = self.folder / f"{uuid.uuid4().hex}.bin"
                path.write_bytes(body)
                structure = ", ".join(f"{c} {types[c.strip('`')]}" for c in cols.split(", ")).replace("'", "\\'")
                sql = f"INSERT INTO {table} ({cols}) SELECT * FROM file('{path}', 'RowBinary', '{structure}')"
            res = self.session.query(sql, fmt, params=values or None)
        except Exception as e:  # noqa: BLE001 (as ClickHouse's HTTP interface reports it)
            return httpx.Response(500, text=str(e))
        return httpx.Response(200, content=res.bytes() if hasattr(res, "bytes") else str(res).encode())


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    session = chdb_session.Session(str(tmp_path_factory.mktemp("chdb")))
    yield session, tmp_path_factory.mktemp("inserts")
    session.close()


@pytest.fixture
def reporting(engine):
    src = Source()
    rep = Reporting(ClickHouseConfig(url="http://chdb", database=f"mo_{uuid.uuid4().hex[:8]}", batch_rows=50), src, transport=Chdb(*engine))
    src.clickhouse = rep
    return src, rep


def rows(n: int, seed: int = 3, plain: bool = True) -> list[dict]:
    """`plain`: no empty tag values and no missing names, where the in-memory ledger answers differently from Postgres."""
    rnd = random.Random(seed)
    out = []
    for i in range(n):
        out.append(complete_row({
            "id": uuid.uuid4(), "ts": NOW - timedelta(minutes=rnd.randint(0, 40 * 24 * 60), microseconds=rnd.randint(0, 999_999)),
            "user_id": rnd.choice(["alice", "bob", "d\\a'n\tx"]), "team": rnd.choice(["eng", "ops"]),
            "served_model": rnd.choice(["big", "small"] if plain else ["big", None]), "backend": rnd.choice(["mock-a", "mock-b"]),
            "status": rnd.choice([200, 200, 500, None]), "upgraded": rnd.choice([True, False, None]), "floor": rnd.choice(["coding:3", None]),
            "requested_model": rnd.choice(["auto", "big"]), "upstream_model": rnd.choice(["big-1", None]), "route_reason": "why",
            "tags": rnd.choice([{}, {"project": "x"}, {"project": "y", "ticket": "T-1"}] + ([] if plain else [{"project": ""}])),
            "input_tokens": rnd.randint(0, 99_999), "output_tokens": rnd.randint(0, 9999), "cache_read_tokens": rnd.randint(0, 9999),
            "latency_ms": rnd.choice([rnd.randint(1, 900), None]),
            **({"service_tier": "batch", "batch_id": "batch_a"} if i % 3 == 0 else {}),  # left out of demand, split out in bills
            **{f: rnd.random() * rnd.choice([0.001, 1.0]) for f in ("actual_cost", "served_api_cost", "baseline_cost", "list_cost",
                                                                     *SAVINGS_FIELDS)},
        }))
    return out


async def load(src, rep, data) -> None:
    for r in data:
        await src.record(r)
    start = (NOW - timedelta(days=41)).replace(hour=0, minute=0, second=0, microsecond=0)
    res, note = await Copier(src, rep).backfill(start, NOW + timedelta(days=1))
    assert res.copied == len(data) and note is None


def like_postgres(rows: list[dict], group_by: str) -> dict:
    """The in-memory summary's keys as Postgres gives them (a day or month as the time it starts)."""
    out = {}
    for r in rows:
        k = r["key"]
        if group_by == "day":
            k = f"{k}T00:00:00+00:00"
        elif group_by == "month":
            k = f"{k}-01T00:00:00+00:00"
        out[k] = {f: round(v, 6) if isinstance(v, float) else v for f, v in r.items() if f != "key"}
    return out


def by_key(rows: list[dict]) -> dict:
    return {r["key"]: {f: v for f, v in r.items() if f != "key"} for r in rows}


async def test_summaries_match_on_both_paths(reporting):
    src, rep = reporting
    await load(src, rep, rows(400))
    for days in (3, 30):  # the ledger's rows, and the hourly rollup the materialized view keeps
        start, end = NOW - timedelta(days=days), NOW + timedelta(hours=1)
        for group_by in ("user", "team", "model", "backend", "day", "hour", "month", "none", "tag:project", "tag:ticket"):
            for scope in ({}, {"user_id": "d\\a'n\tx"}, {"team": "eng"}):
                got = by_key(await rep.ledger.summary(start, end, group_by, **scope))
                want = like_postgres(await src.summary(start, end, group_by, **scope), group_by)
                assert got == want, (days, group_by, scope)


async def test_postgres_semantics_where_memory_differs(reporting):
    src, rep = reporting
    data = rows(60, seed=9, plain=False)
    await load(src, rep, data)
    start, end = NOW - timedelta(days=30), NOW + timedelta(hours=1)
    # An empty tag value is a value, as `coalesce(tags->>'project', '(untagged)')` has it; a missing one is untagged.
    tags = {r["key"]: r["requests"] for r in await rep.ledger.summary(start, end, "tag:project")}
    assert tags == {k: sum(1 for r in data if start <= r["ts"] < end and r["tags"].get("project", "(untagged)") == k) for k in tags}
    assert "" in tags
    # A missing model: NULL on the ledger's rows, '' in the hourly rollup, as in Postgres's usage_hourly.
    assert None in {r["key"] for r in await rep.ledger.summary(NOW - timedelta(days=3), end, "model")}
    assert "" in {r["key"] for r in await rep.ledger.summary(start, end, "model")}


async def test_exports_demand_and_bill_comparisons_match(reporting):
    src, rep = reporting
    await load(src, rep, rows(300, seed=5))
    start, end = NOW - timedelta(days=41), NOW + timedelta(days=1)
    for team in (None, "eng"):
        got = [r async for r in rep.ledger.export_rows(start, end, team)]
        want = sorted([r async for r in src.export_rows(start, end, team)], key=lambda r: (r["ts"], str(r["id"])))
        assert got == want and list(got[0]) == LEDGER_FIELDS

    def key(r):
        return tuple(str(r[k]) for k in r if not isinstance(r[k], (int, float)) or isinstance(r[k], bool))

    def rounded(rs):
        return sorted(({k: round(v, 9) if isinstance(v, float) else v for k, v in r.items()} for r in rs), key=key)

    since = NOW - timedelta(days=30)
    assert rounded(await rep.ledger.demand_by_hour(since)) == rounded(await src.demand_by_hour(since))
    assert rounded(await rep.ledger.ledger_costs(start, end, ["mock-a", "x"])) == rounded(await src.ledger_costs(start, end, ["mock-a", "x"]))


async def test_the_copy_on_a_real_engine(reporting):
    src, rep = reporting
    await load(src, rep, rows(120, seed=11))
    copier = Copier(src, rep)
    again, _ = await copier.backfill(NOW - timedelta(days=41), NOW + timedelta(days=1))
    assert again.copied == 0 and again.repaired == []
    fresh = [complete_row({"id": uuid.uuid4(), "ts": NOW - timedelta(minutes=m), "user_id": "alice", "team": "eng", "status": 200})
             for m in (1, 2, 130)]
    for r in fresh:
        await src.record(r)
    assert (await copier.run_once()).copied == 3
    await rep.ledger.insert([fresh[0]])  # a row twice: the hour is copied again whole
    res = await copier.run_once()
    hour = fresh[0]["ts"].replace(minute=0, second=0, microsecond=0)
    assert res.repaired == [hour.isoformat()]
    counts = await rep.ledger.minute_counts(hour, hour + timedelta(hours=1))
    assert all(n == u for n, u in counts.values()) and sum(n for n, _ in counts.values()) == len(
        [r for r in src.ledger if hour <= r["ts"] < hour + timedelta(hours=1)])
    assert (await rep.ledger.hour_rollups(hour, hour + timedelta(hours=1)))[hour] == sum(n for n, _ in counts.values())
    state = await src.get_setting(STATE_KEY)
    assert state["through"] and state["error"] is None
    d = await rep.describe()
    assert d["ok"] and d["tables"]["requests"]["rows"] >= 123 and d["answering"]


async def test_schema_changes_on_a_real_engine(reporting):
    src, rep = reporting
    assert await rep.ledger.migrate()
    assert await rep.ledger.migrate() == []
    rep.cfg = rep.ledger.cfg = rep.cfg.model_copy(update={"ttl_days": 90, "rollup_ttl_days": 800})
    assert await rep.ledger.migrate() == ["requests keeps 90 days", "usage_hourly keeps 800 days"]
    assert await rep.ledger.migrate() == []
    await rep.client.command(f"ALTER TABLE `{rep.cfg.database}`.requests DROP COLUMN route_reason")
    await rep.client.command(f"ALTER TABLE `{rep.cfg.database}`.usage_hourly DROP COLUMN savings_compression")
    rep.cfg = rep.ledger.cfg = rep.cfg.model_copy(update={"ttl_days": None, "rollup_ttl_days": None})
    assert set(await rep.ledger.migrate()) == {"added requests.route_reason", "added usage_hourly.savings_compression",
                                              "requests keeps everything", "usage_hourly keeps everything"}
    cols = await rep.client.query("SELECT name FROM system.columns WHERE database = {db:String} AND table = 'usage_hourly'",
                                  {"db": rep.cfg.database})
    assert {c["name"] for c in cols} >= set(SUM_FIELDS)
