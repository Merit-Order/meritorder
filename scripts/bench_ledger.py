"""Load a synthetic ledger and time the reports on Postgres and on ClickHouse (docs/08-reporting.md, "Measured at scale").

    # a throwaway Postgres migrated with `meritorder migrate`, and a ClickHouse
    uv run python scripts/bench_ledger.py seed --rows 20000000 --days 365      # into Postgres, a day at a time
    uv run meritorder worker --backfill                                        # Postgres's hourly rollup
    time uv run meritorder clickhouse backfill                                 # the copy: rows per second
    uv run python scripts/bench_ledger.py grow --database meritorder_x3 --times 3   # a bigger copy, in ClickHouse only
    uv run python scripts/bench_ledger.py reports [--database meritorder_x3]   # every report at 7, 30 and 365 days
    uv run python scripts/bench_ledger.py sizes

It reads MERITORDER_DATABASE_URL, MERITORDER_CLICKHOUSE_URL, _USER and _PASSWORD. The rows look like a year of a large
organization's traffic: 2,000 people in 40 teams with a few heavy users, ten models on ten backends, weekday and office-hour
peaks, growth over the year, agent sessions with cache reads, two tags on two thirds of the rows, errors and batches.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import time
from datetime import UTC, datetime, timedelta

from meritorder.clickhouse import ClickHouse, Ledger, Reporting
from meritorder.config import ClickHouseConfig
from meritorder.store import PostgresStore

# served model, upstream model, backend, backend kind, $ per million input and output tokens, share of traffic (%)
MODELS = [
    ("claude-sonnet", "claude-sonnet-5-5", "anthropic", "api", 3.0, 15.0, 20),
    ("claude-sonnet", "anthropic.claude-sonnet-5-5", "bedrock", "api", 3.0, 15.0, 10),
    ("claude-haiku", "claude-haiku-5-5", "anthropic", "api", 1.0, 5.0, 15),
    ("claude-opus", "claude-opus-5-5", "anthropic", "api", 5.0, 25.0, 7),
    ("gpt-5.6-mini", "gpt-5.6-mini", "openai", "api", 0.25, 2.0, 10),
    ("gpt-5.6", "gpt-5.6", "openai", "api", 1.25, 10.0, 5),
    ("gemini-3.5-flash", "gemini-3.5-flash", "vertex-gemini", "api", 0.3, 2.5, 8),
    ("kimi-k3.1", "moonshotai/kimi-k3.1", "openrouter", "api", 0.6, 2.5, 8),
    ("qwen3-coder-next", "qwen3-coder-next", "gpu-qwen-coder", "self_hosted", 0.4, 1.6, 9),
    ("deepseek-v4.1-flash", "deepseek-v4.1-flash", "deepseek", "api", 0.28, 0.42, 5),
    ("glm-5.3", "glm-5.3", "zai", "api", 0.6, 2.2, 3),
]
PICK = [i for i, m in enumerate(MODELS, 1) for _ in range(m[6])]  # 100 slots, a model's share each
assert len(PICK) == 100
COLS = [m[:6] for m in MODELS]

# One day of rows from random draws (r: the draws, c: the costs worked out from them). Postgres 16 has random_normal().
SEED_SQL = """
INSERT INTO requests (id, ts, user_id, team, key_id, session_id, tags, client_format, requested_model, floor, route_reason,
    served_model, upstream_model, backend, backend_kind, status, error, stream, upgraded, proxy_cache,
    input_tokens, cache_read_tokens, cache_write_tokens, output_tokens, latency_ms, ttft_ms,
    actual_cost, served_api_cost, baseline_cost, list_cost, savings_routing, savings_caching, savings_self_hosting,
    savings_compression, savings_compression_model, savings_response_cache, service_tier, batch_id, savings_batch)
SELECT gen_random_uuid(), ts, 'user' || u || '@example.com', 'team-' || lpad((u % 40)::text, 2, '0'), 'key_' || u,
       's-' || u || '-' || (g / 30), tags, CASE WHEN fmt < 0.6 THEN 'anthropic' ELSE 'openai' END,
       CASE WHEN auto THEN 'auto' ELSE model END, 'coding:' || lvl, CASE WHEN auto THEN 'cheapest route meeting coding:' || lvl END,
       model, upstream, backend, kind, status, CASE WHEN status >= 400 THEN 'upstream ' || status || ': try again later' END,
       stream, upgraded, proxy_cache, input, cache_read, cache_write, output, ttft + output * 12, ttft,
       actual, api, baseline, api, greatest(baseline - api - caching, 0), caching,
       CASE WHEN kind = 'self_hosted' THEN api - actual ELSE 0 END, compression, 0, CASE WHEN proxy_cache THEN api ELSE 0 END,
       CASE WHEN batch THEN 'batch' END, CASE WHEN batch THEN 'batch_' || u || '_' || (g / 500) END, CASE WHEN batch THEN actual ELSE 0 END
FROM (
    SELECT *, CASE WHEN kind = 'self_hosted' THEN api * 0.45 WHEN batch THEN api * 0.5 ELSE api END AS actual,
           ((input + cache_read + cache_write) * 3.0 + output * 15.0) / 1e6 AS baseline,
           cache_read * pin * 0.9 / 1e6 AS caching,
           CASE WHEN compressed THEN input * pin * 0.15 / 1e6 ELSE 0 END AS compression
    FROM (
        SELECT *, (input * pin + cache_read * pin * 0.1 + cache_write * pin * 1.25 + output * pout) / 1e6 AS api
        FROM (
            SELECT g, $1::timestamptz + make_interval(secs => CASE WHEN random() < 0.75 THEN 28800 + random() * 36000
                                                                   ELSE random() * 86400 END) AS ts,
                   floor(2000 * random() ^ 3)::int AS u, random() AS fmt, random() < 0.7 AS auto, 1 + floor(random() * 5)::int AS lvl,
                   (($3::text[])[k]) AS model, (($4::text[])[k]) AS upstream, (($5::text[])[k]) AS backend, (($6::text[])[k]) AS kind,
                   (($7::float8[])[k]) AS pin, (($8::float8[])[k]) AS pout,
                   CASE WHEN st < 0.98 THEN 200 WHEN st < 0.99 THEN 429 ELSE 529 END AS status,
                   random() < 0.8 AS stream, random() < 0.03 AS upgraded, random() < 0.02 AS proxy_cache, random() < 0.05 AS compressed,
                   st < 0.98 AND random() < 0.02 AS batch,
                   least(greatest(exp(7.6 + 1.3 * random_normal()), 20), 400000)::bigint AS input,
                   CASE WHEN random() < 0.6 THEN (exp(9.5 + 1.0 * random_normal()))::bigint ELSE 0 END AS cache_read,
                   CASE WHEN random() < 0.3 THEN (exp(7.0 + 1.0 * random_normal()))::bigint ELSE 0 END AS cache_write,
                   least((exp(6.0 + 1.0 * random_normal())), 32000)::bigint AS output,
                   (300 + random() * 1700)::int AS ttft,
                   CASE WHEN random() < 0.65 THEN jsonb_build_object('project', 'p' || floor(200 * random() ^ 2)::int,
                                                                      'env', CASE WHEN random() < 0.8 THEN 'prod' ELSE 'dev' END)
                        ELSE '{}'::jsonb END AS tags
            FROM (SELECT g, ($9::int[])[1 + floor(random() * 100)::int] AS k, random() AS st FROM generate_series(1, $2::int) g) d
        ) r
    ) a
) c
"""


def per_day(rows: int, days: int) -> list[int]:
    """Rows per day: weekdays full, weekends a quarter, and traffic growing from 60% to 140% over the range."""
    weights = []
    start = datetime.now(UTC).date() - timedelta(days=days)
    for i in range(days):
        day = start + timedelta(days=i)
        weights.append((0.25 if day.weekday() >= 5 else 1.0) * (0.6 + 0.8 * i / max(days - 1, 1)))
    total = sum(weights)
    return [round(rows * w / total) for w in weights]


async def ensure_months(store: PostgresStore, first: datetime) -> None:
    """Monthly partitions back to `first`, as the worker would have made them over the year (not the default partition)."""
    month = first.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    async with store.pool.acquire() as c:
        while month < datetime.now(UTC):
            nxt = (month + timedelta(days=32)).replace(day=1)
            name = f"requests_{month:%Y_%m}"
            if not await c.fetchval("SELECT to_regclass($1)", name):
                await c.execute(f"CREATE TABLE {name} PARTITION OF requests FOR VALUES FROM ('{month.isoformat()}') TO ('{nxt.isoformat()}')")
            month = nxt


async def seed(args) -> None:
    store = PostgresStore(os.environ["MERITORDER_DATABASE_URL"])
    await store.init()
    today = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    first = today - timedelta(days=args.days)
    await ensure_months(store, first)
    cols = [list(x) for x in zip(*COLS)]
    t0, done = time.monotonic(), 0
    async with store.pool.acquire() as c:
        for i, n in enumerate(per_day(args.rows, args.days)):
            if n:
                await c.execute(SEED_SQL, first + timedelta(days=i), n, *cols, PICK)
            done += n
            if i % 30 == 29 or i == args.days - 1:
                rate = done / (time.monotonic() - t0)
                print(f"  {first + timedelta(days=i):%Y-%m-%d}: {done:,} rows, {rate:,.0f} rows/s", flush=True)
    await store.close()


def ch_config(database: str | None = None) -> ClickHouseConfig:
    return ClickHouseConfig(url=os.environ["MERITORDER_CLICKHOUSE_URL"], user=os.environ.get("MERITORDER_CLICKHOUSE_USER", "default"),
                            database=database or "meritorder", timeout_seconds=600)


async def grow(args) -> None:
    """A bigger copy in ClickHouse only: the copied rows `times` times over, each time with new ids and times moved by up to
    half an hour, a month at a time. The tables and the rollup view are Merit Order's own (Ledger.migrate)."""
    src, dst = ch_config(), ch_config(args.database)
    ch = ClickHouse(dst)
    await Ledger(ch, dst).migrate()
    months = [r["m"] for r in await ch.query(f"SELECT DISTINCT toYYYYMM(ts) AS m FROM `{src.database}`.requests ORDER BY m")]
    t0, rows = time.monotonic(), 0
    for i in range(args.times):
        for m in months:
            await ch.command(f"INSERT INTO `{dst.database}`.requests SELECT * REPLACE (generateUUIDv4() AS id, "
                             f"ts + toIntervalMillisecond(rand({i}) % 3600000 - 1800000) AS ts) FROM `{src.database}`.requests "
                             f"WHERE toYYYYMM(ts) = {int(m)}",
                             # blocks of 100,000 rows, not a million: the rollup view aggregates each block in memory
                             settings={"max_threads": 2, "max_insert_threads": 1, "min_insert_block_size_rows": 100_000})
        rows = (await ch.query(f"SELECT count() AS n FROM `{dst.database}`.requests"))[0]["n"]
        print(f"  copy {i + 1}: {rows:,} rows, {rows / (time.monotonic() - t0):,.0f} rows/s", flush=True)
    await ch.close()


def days_ago(n: int) -> tuple[datetime, datetime]:
    end = datetime.now(UTC).replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    return end - timedelta(days=n), end


async def timed(call, runs: int) -> tuple[float, int]:
    """The median wall time of `runs` calls, after one to warm caches when runs > 1, and how many rows the result had."""
    times, n = [], 0
    for _ in range(runs + 1 if runs > 1 else 1):
        t = time.monotonic()
        out = await call()
        times.append(time.monotonic() - t)
        n = len(out) if isinstance(out, list) else out
    return statistics.median(times[1:] or times), n


async def count_export(gen) -> int:
    n = 0
    async for _ in gen:
        n += 1
    return n


async def reports(args) -> None:
    store = PostgresStore(os.environ["MERITORDER_DATABASE_URL"])
    await store.init()
    cfg = ch_config(args.database)
    rep = Reporting(cfg, store)
    ledger = rep.ledger
    team = "team-07"
    stores = {"postgres": store, "clickhouse": ledger}
    if args.only:
        stores = {k: v for k, v in stores.items() if k == args.only}
    cases = {
        "summary by team": lambda s, a, b: s.summary(a, b, "team"),
        "summary by model": lambda s, a, b: s.summary(a, b, "model"),
        "summary by user": lambda s, a, b: s.summary(a, b, "user"),
        "summary by day": lambda s, a, b: s.summary(a, b, "day"),
        "team lead: one team by user": lambda s, a, b: s.summary(a, b, "user", team=team),
        "tag breakdown (tag:project)": lambda s, a, b: s.summary(a, b, "tag:project"),
        "bills: ledger side": lambda s, a, b: s.ledger_costs(a, b, ["anthropic", "openai", "bedrock"]),
        "GPU what-if (demand by hour)": lambda s, a, b: s.demand_by_hour(a),
        "export, one team": lambda s, a, b: count_export(s.export_rows(a, b, team)),
    }
    if args.full_export:
        cases["export, everything"] = lambda s, a, b: count_export(s.export_rows(a, b))
    print(f"{'report':32} {'days':>4} " + " ".join(f"{k:>18}" for k in stores))
    for name, call in cases.items():
        for n in args.days:
            a, b = days_ago(n)
            cells = []
            for s in stores.values():  # the store has no ClickHouse attached: Postgres answers; ClickHouse through its Ledger
                secs, rows = await timed(lambda: call(s, a, b), args.runs)
                cells.append(f"{secs * 1000:>10,.0f} ms {rows:>5}" if rows < 100000 else f"{secs * 1000:>8,.0f} ms {rows / 1e6:.1f}M")
            print(f"{name:32} {n:>4} " + " ".join(f"{c:>18}" for c in cells), flush=True)
    await rep.close()
    await store.close()


async def sizes(args) -> None:
    store = PostgresStore(os.environ["MERITORDER_DATABASE_URL"])
    await store.init()
    async with store.pool.acquire() as c:
        rows = await c.fetchval("SELECT count(*) FROM requests")
        heap = await c.fetchval("SELECT sum(pg_table_size(inhrelid))::bigint FROM pg_inherits WHERE inhparent = 'requests'::regclass")
        idx = await c.fetchval("SELECT sum(pg_indexes_size(inhrelid))::bigint FROM pg_inherits WHERE inhparent = 'requests'::regclass")
        hourly = await c.fetchval("SELECT pg_total_relation_size('usage_hourly')")
        hourly_rows = await c.fetchval("SELECT count(*) FROM usage_hourly")
    print(f"Postgres: {rows:,} ledger rows, table {heap / 1e9:.2f} GB + indexes {idx / 1e9:.2f} GB "
          f"= {(heap + idx) / rows:.0f} MB per million rows; usage_hourly {hourly_rows:,} rows, {hourly / 1e9:.2f} GB")
    for db in [args.database] if args.database else ["meritorder"]:
        ch = ClickHouse(ch_config(db))
        for r in await ch.query("SELECT table, sum(rows) AS rows, sum(bytes_on_disk) AS disk, sum(data_uncompressed_bytes) AS raw, "
                                "count() AS parts FROM system.parts WHERE database = {db:String} AND active GROUP BY table", {"db": db}):
            per = r["disk"] / max(r["rows"], 1)  # bytes a row: MB per million rows
            print(f"ClickHouse {db}.{r['table']}: {r['rows']:,} rows in {r['parts']} parts, {r['disk'] / 1e9:.2f} GB on disk "
                  f"({per:.1f} MB per million rows; {r['raw'] / max(r['disk'], 1):.1f}x compression)")
        await ch.close()
    await store.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("seed")
    s.add_argument("--rows", type=int, default=1_000_000)
    s.add_argument("--days", type=int, default=365)
    g = sub.add_parser("grow")
    g.add_argument("--database", required=True)
    g.add_argument("--times", type=int, default=3)
    r = sub.add_parser("reports")
    r.add_argument("--database")
    r.add_argument("--days", type=int, nargs="+", default=[7, 30, 365])
    r.add_argument("--runs", type=int, default=3)
    r.add_argument("--only", choices=["postgres", "clickhouse"])
    r.add_argument("--full-export", action="store_true")
    z = sub.add_parser("sizes")
    z.add_argument("--database")
    a = p.parse_args()
    asyncio.run({"seed": seed, "grow": grow, "reports": reports, "sizes": sizes}[a.cmd](a))


if __name__ == "__main__":
    main()
