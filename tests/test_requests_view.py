"""The Requests view (GET /v1/reports/requests, requests.csv): keyset paging with no duplicates or gaps while rows arrive,
every filter, sorting, and role scoping that no filter can widen. Each test runs on the memory store and, in CI, on
Postgres, with the same assertions: the two answer alike."""

import csv
import io
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from conftest import ADMIN_KEY, LEAD_KEY, TEST_KEY, auth, ledger_row

from meritorder.reports import SORTS
from meritorder.store import SAVINGS_CAUSES, RequestFilter

BASE = datetime.now(UTC).replace(microsecond=0) - timedelta(hours=2)


def row(i: int, **kw) -> dict:
    """The i-th request, a second apart (every fifth shares its time with the one before, so ids break the ties)."""
    return ledger_row(ts=BASE + timedelta(seconds=i - (1 if i % 5 == 0 else 0)), **kw)


def varied() -> list[dict]:
    """Requests that differ on every field a filter reads."""
    out = []
    for i in range(40):
        out.append(row(
            i, user_id=["alice", "bob", "carol"][i % 3], team="ops" if i % 3 == 2 else "default", key_id=f"key_{i % 4}",
            session_id=f"s{i % 5}", requested_model="auto" if i % 2 else "big", served_model=["small", "big"][i % 2],
            backend=["mock-openai", "mock-anthropic", "response-cache"][i % 3], backend_kind="cache" if i % 3 == 2 else "mock",
            client_format=["openai", "anthropic"][i % 2], status=[200, 200, 429, 500, 200][i % 5],
            error="boom" if i % 5 in (2, 3) else None, floor=[None, "coding:3", "general:1", "coding:5"][i % 4],
            route_reason=["classified coding:3 (code fence)", "Short Question", None, "named model"][i % 4],
            tags={"project": ["apollo", "zeus"][i % 2], **({"explore": "down"} if i % 7 == 0 else {})},
            actual_cost=round(0.01 * (i % 9), 4), baseline_cost=0.2, savings_routing=round(0.005 * (i % 6), 4),
            savings_caching=0.01 if i % 4 == 0 else 0.0, latency_ms=None if i % 11 == 0 else 100 + 37 * (i % 13),
            stream=i % 2 == 0, service_tier="batch" if i % 6 == 0 else None, cache_read_tokens=50 if i % 8 == 0 else 0,
            failed_over="up-a/m-a 529" if i % 9 == 0 else None))
    return out


async def seed(gateway, rows):
    for r in rows:
        await gateway.store.record(r)
    return rows


async def page(client, who=ADMIN_KEY, **params) -> dict:
    r = await client.get("/v1/reports/requests", params=params, headers=auth(who))
    assert r.status_code == 200, r.text
    return r.json()


async def every_page(client, who=ADMIN_KEY, **params) -> list[str]:
    ids, cursor = [], None
    for _ in range(200):
        d = await page(client, who, **params, **({"cursor": cursor} if cursor else {}))
        ids += [r["id"] for r in d["rows"]]
        cursor = d["next"]
        if not cursor:
            return ids
    raise AssertionError("paging never ended")


def order_key(r: dict, sort: str):
    savings = 0.0
    for c in SAVINGS_CAUSES:
        savings += r.get(f"savings_{c}") or 0
    v = {"time": r["ts"], "cost": r["actual_cost"], "savings": savings,
         "latency": -1 if r.get("latency_ms") is None else r["latency_ms"]}[sort]
    return v, r["ts"], uuid.UUID(str(r["id"]))


async def test_pages_have_no_duplicates_or_gaps_while_new_rows_arrive(client, gateway):
    rows = await seed(gateway, [row(i) for i in range(57)])
    expected = [str(r["id"]) for r in sorted(rows, key=lambda r: order_key(r, "time"), reverse=True)]
    seen, cursor, n = [], None, 0
    while True:
        d = await page(client, limit=10, **({"cursor": cursor} if cursor else {}))
        assert d["total"] == 57 + n and d["total_exact"] is True and d["limit"] == 10
        seen += [r["id"] for r in d["rows"]]
        # New requests arrive between pages: newer than everything, they're on the first page, never on later ones.
        await gateway.store.record(ledger_row(ts=datetime.now(UTC) + timedelta(seconds=n)))
        n += 1
        if not d["next"]:
            break
        cursor = d["next"]
    assert seen == expected  # every row once, in order, none of the new ones

    # And back: each previous page is the one before it.
    back, prev = [r["id"] for r in d["rows"]], d["prev"]
    while prev:
        d = await page(client, limit=10, cursor=prev)
        back = [r["id"] for r in d["rows"]] + back
        prev = d["prev"]
    assert set(expected) <= set(back) and back[-7:] == expected[-7:]
    first = await page(client, limit=10)
    assert d["rows"] == first["rows"] and d["prev"] is None  # back at the start: the first page, whole (new rows and all)


async def test_previous_from_the_second_page_is_the_first(client, gateway):
    await seed(gateway, [row(i) for i in range(25)])
    one = await page(client, limit=10)
    two = await page(client, limit=10, cursor=one["next"])
    assert one["prev"] is None and two["prev"] and two["next"]
    assert {r["id"] for r in one["rows"]}.isdisjoint(r["id"] for r in two["rows"])
    assert (await page(client, limit=10, cursor=two["prev"]))["rows"] == one["rows"]
    three = await page(client, limit=10, cursor=two["next"])
    assert len(three["rows"]) == 5 and three["next"] is None
    assert (await page(client, limit=10, cursor=three["prev"]))["rows"] == two["rows"]


@pytest.mark.parametrize("sort", ["time", "cost", "savings", "latency"])
@pytest.mark.parametrize("order", ["desc", "asc"])
async def test_every_sort_pages_through_every_row_in_order(client, gateway, sort, order):
    rows = await seed(gateway, varied())
    params = {"sort": sort, "order": order, "limit": 7, "start": (BASE - timedelta(hours=1)).isoformat()}
    got = await every_page(client, **params)
    want = [str(r["id"]) for r in sorted(rows, key=lambda r: order_key(r, sort), reverse=order == "desc")]
    assert got == want


async def test_sorting_by_anything_but_time_needs_a_short_range(client, gateway):
    await seed(gateway, varied())
    for params in ({"sort": "cost"}, {"sort": "latency", "start": (BASE - timedelta(days=8)).date().isoformat()},
                   {"sort": "savings", "start": "2026-01-01", "end": "2026-01-09"}):
        r = await client.get("/v1/reports/requests", params=params, headers=auth(ADMIN_KEY))
        assert r.status_code == 400 and "7 days or less" in r.text
    assert (await page(client, sort="cost", start="2026-01-01", end="2026-01-08"))["rows"] == []
    assert len((await page(client, sort="time"))["rows"]) == 40  # time sorts any range
    assert set(SORTS.__args__) == {"time", "cost", "savings", "latency"}


# (query parameters, which rows they match)
FILTERS = [
    ({"user": "bob"}, lambda r: r["user_id"] == "bob"),
    ({"team": "ops"}, lambda r: r["team"] == "ops"),
    ({"key": "key_1"}, lambda r: r["key_id"] == "key_1"),
    ({"session": "s3"}, lambda r: r["session_id"] == "s3"),
    ({"requested_model": "auto"}, lambda r: r["requested_model"] == "auto"),
    ({"served_model": "big"}, lambda r: r["served_model"] == "big"),
    ({"backend": "response-cache"}, lambda r: r["backend"] == "response-cache"),
    ({"format": "anthropic"}, lambda r: r["client_format"] == "anthropic"),
    ({"status": "error"}, lambda r: r["status"] >= 400),
    ({"status": "success"}, lambda r: r["status"] < 400),
    ({"status": "429"}, lambda r: r["status"] == 429),
    ({"status": "5xx"}, lambda r: 500 <= r["status"] < 600),
    ({"task": "coding"}, lambda r: (r["floor"] or "").startswith("coding:")),
    ({"level": 1}, lambda r: r["floor"] == "general:1"),
    ({"task": "coding", "level": 5}, lambda r: r["floor"] == "coding:5"),
    ({"reason": "short QUESTION"}, lambda r: r["route_reason"] == "Short Question"),
    ({"reason": "coding:3 ("}, lambda r: (r["route_reason"] or "").startswith("classified coding:3")),
    ({"reason": "%"}, lambda r: False),  # LIKE's wildcards are only themselves
    ({"tag": "project=apollo"}, lambda r: r["tags"]["project"] == "apollo"),
    ({"tag": "explore"}, lambda r: "explore" in r["tags"]),
    ({"tag": ["project=zeus", "explore=down"]}, lambda r: r["tags"]["project"] == "zeus" and "explore" in r["tags"]),
    ({"tag": "judged_label"}, lambda r: False),
    ({"min_cost": 0.05}, lambda r: r["actual_cost"] >= 0.05),
    ({"min_cost": 0.02, "max_cost": 0.04}, lambda r: 0.02 <= r["actual_cost"] <= 0.04),
    ({"min_savings": 0.02}, lambda r: r["savings_routing"] + r["savings_caching"] >= 0.02),
    ({"max_savings": 0}, lambda r: r["savings_routing"] + r["savings_caching"] <= 0),
    ({"savings_cause": "caching"}, lambda r: r["savings_caching"] != 0),
    ({"savings_cause": "batch"}, lambda r: False),
    ({"min_latency": 300}, lambda r: r["latency_ms"] is not None and r["latency_ms"] >= 300),
    ({"max_latency": 200}, lambda r: r["latency_ms"] is not None and r["latency_ms"] <= 200),
    ({"stream": "true"}, lambda r: r["stream"]),
    ({"stream": "false"}, lambda r: not r["stream"]),
    ({"batch": "true"}, lambda r: r["service_tier"] == "batch"),
    ({"batch": "false"}, lambda r: r["service_tier"] != "batch"),
    ({"cache_hit": "true"}, lambda r: r["cache_read_tokens"] > 0 or r["backend_kind"] == "cache"),
    ({"cache_hit": "false"}, lambda r: not (r["cache_read_tokens"] > 0 or r["backend_kind"] == "cache")),
    ({"failover": "true"}, lambda r: bool(r["failed_over"])),
    ({"failover": "false"}, lambda r: not r["failed_over"]),
    ({"start": (BASE + timedelta(seconds=10)).isoformat(), "end": (BASE + timedelta(seconds=20)).isoformat().replace("+00:00", "Z")},
     lambda r: BASE + timedelta(seconds=10) <= r["ts"] < BASE + timedelta(seconds=20)),
    ({"user": "alice", "served_model": "small", "status": "success"},
     lambda r: r["user_id"] == "alice" and r["served_model"] == "small" and r["status"] < 400),
]


@pytest.mark.parametrize(("params", "match"), FILTERS, ids=[str(p) for p, _ in FILTERS])
async def test_every_filter(client, gateway, params, match):
    rows = await seed(gateway, varied())
    want = {str(r["id"]) for r in rows if match(r)}
    got = await every_page(client, limit=8, **params)
    assert len(got) == len(set(got)) and set(got) == want
    assert (await page(client, **params))["total"] == len(want)
    r = await client.get("/v1/reports/requests.csv", params=params, headers=auth(ADMIN_KEY))
    assert r.status_code == 200 and {x["id"] for x in csv.DictReader(io.StringIO(r.text))} == want


async def test_bad_filters_are_refused(client, gateway):
    for params, why in (({"status": "nope"}, "status must be"), ({"tag": "bad tag"}, "tag filters"),
                        ({"start": "yesterday"}, "start must be"), ({"start": "2026-02-01", "end": "2026-01-01"}, "end must be after"),
                        ({"cursor": "not-a-cursor"}, "cursor"), ({"savings_cause": "luck"}, ""), ({"level": 6}, "")):
        r = await client.get("/v1/reports/requests", params=params, headers=auth(ADMIN_KEY))
        assert r.status_code in (400, 422) and why in r.text, (params, r.text)
    for cause in SAVINGS_CAUSES:  # the API's list of causes is the store's
        assert (await client.get("/v1/reports/requests", params={"savings_cause": cause}, headers=auth(ADMIN_KEY))).status_code == 200
    await seed(gateway, [row(i) for i in range(5)])
    d = await page(client, limit=2)
    r = await client.get("/v1/reports/requests", params={"limit": 2, "cursor": d["next"], "sort": "cost", "start": BASE.isoformat()},
                         headers=auth(ADMIN_KEY))
    assert r.status_code == 400 and "this sort and order" in r.text  # a cursor belongs to its sort


async def test_scoping_holds_whatever_the_filters_ask_for(client, gateway):
    from test_reports_api import finance_key

    rows = await seed(gateway, varied())
    mine = {str(r["id"]) for r in rows if r["user_id"] == "alice"}
    team = {str(r["id"]) for r in rows if r["team"] == "default"}
    carol_key = next(r["key_id"] for r in rows if r["user_id"] == "carol")
    carol_session = next(r["session_id"] for r in rows if r["user_id"] == "carol")
    # A user sees their own requests: asking for another person, team, key or session never shows anyone else's.
    for params in ({}, {"user": "bob"}, {"team": "ops"}, {"key": carol_key}, {"session": carol_session}, {"user": "carol", "team": "ops"},
                   {"tag": "project=apollo"}, {"status": "error"}):
        got = set(await every_page(client, TEST_KEY, **params))
        assert got <= mine, params
        d = await page(client, TEST_KEY, **params)
        assert d["user"] == "alice" and d["team"] is None
        csv_ids = {x["id"] for x in csv.DictReader(io.StringIO((await client.get("/v1/reports/requests.csv", params=params,
                                                                                    headers=auth(TEST_KEY))).text))}
        assert csv_ids == got
    assert set(await every_page(client, TEST_KEY)) == mine
    # A team lead sees their team; another team is clamped to their own, and carol (ops) is never in it.
    assert set(await every_page(client, LEAD_KEY, team="ops")) == team
    assert await every_page(client, LEAD_KEY, user="carol") == []
    assert set(await every_page(client, LEAD_KEY, user="bob")) == {str(r["id"]) for r in rows if r["user_id"] == "bob"}
    # Admins see everyone and filter as they like.
    assert set(await every_page(client, ADMIN_KEY, team="ops")) == {str(r["id"]) for r in rows if r["team"] == "ops"}
    # Finance sees aggregates only: no per-request rows, as a page or as CSV.
    fin = await finance_key(gateway)
    for path in ("/v1/reports/requests", "/v1/reports/requests.csv"):
        r = await client.get(path, params={"team": "ops"}, headers=auth(fin))
        assert r.status_code == 403 and "aggregates only" in r.text


async def test_csv_has_every_ledger_column_in_the_views_order(client, gateway):
    rows = await seed(gateway, varied())
    r = await client.get("/v1/reports/requests.csv", params={"sort": "cost", "order": "asc", "start": (BASE - timedelta(hours=1)).isoformat()},
                         headers=auth(ADMIN_KEY))
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv") and "x-meritorder-answered-by" in r.headers
    got = list(csv.DictReader(io.StringIO(r.text)))
    assert [x["id"] for x in got] == [str(x["id"]) for x in sorted(rows, key=lambda x: order_key(x, "cost"))]
    first = next(x for x in got if x["id"] == str(rows[9]["id"]))
    assert first["failed_over"] == "up-a/m-a 529" and first["tags"] == '{"project":"zeus"}' and float(first["savings_total"]) >= 0
    empty = await client.get("/v1/reports/requests.csv", params={"user": "nobody"}, headers=auth(ADMIN_KEY))
    assert empty.text.startswith("id,ts,") and len(empty.text.splitlines()) == 1


async def test_counts_are_exact_up_to_the_cap_then_estimated(gateway):
    await seed(gateway, [row(i) for i in range(12)])
    store = gateway.store
    assert await store.requests_count(RequestFilter()) == (12, True)
    n, exact = await store.requests_count(RequestFilter(), cap=5)
    assert n >= 6 and exact is (store.kind == "memory")  # Postgres stops counting at the cap and estimates the rest
    assert await store.requests_count(RequestFilter(user_id="nobody"), cap=5) == (0, True)


async def test_a_page_past_the_end_still_goes_back(client, gateway):
    rows = await seed(gateway, [row(i) for i in range(6)])
    d = await page(client, limit=3)
    later = await page(client, limit=3, cursor=d["next"])
    assert len(later["rows"]) == 3 and later["next"] is None
    # The oldest rows went (a partition dropped): the link now lands past the end, and Previous still works.
    oldest = min(rows, key=lambda r: order_key(r, "time"))
    from meritorder.reports import _cursor

    past = _cursor((None, (oldest["ts"] - timedelta(days=1)).isoformat(), uuid.uuid4()), "time", "desc", True)
    empty = await page(client, limit=3, cursor=past)
    assert empty["rows"] == [] and empty["next"] is None and empty["prev"]
    assert len((await page(client, limit=3, cursor=empty["prev"]))["rows"]) == 3
