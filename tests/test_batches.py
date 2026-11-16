"""Batches end to end through the official OpenAI and Anthropic SDKs, on the mock backends' batch API: routing each request,
splitting, the budget reservation, the worker settling results into the ledger, joining them back, cancelling, and keeping
content out of the gateway."""

import io
import json

import anthropic
import httpx
import httpx2
import openai
import pytest
from conftest import ADMIN_KEY, DATABASE_URL, LEAD_KEY, TEST_KEY, auth, ledger, ledger_row, real_store

from meritorder.app import create_app
from meritorder.batches.poller import BatchPoller
from meritorder.batches.store import PostgresBatches


@pytest.fixture(autouse=True)
def batch_key(monkeypatch):
    """Uploads are held encrypted: a gateway on Postgres takes them only with a key."""
    monkeypatch.setenv("MERITORDER_BATCH_KEY", "test-batch-key")


@pytest.fixture
async def app_client(gateway):
    app = create_app(gateway=gateway)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        c.app = app
        yield c


def oai(client: httpx.AsyncClient, key: str = TEST_KEY) -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(api_key=key, base_url="http://test/v1", http_client=client, max_retries=0)


def ant(client: httpx.AsyncClient, key: str = TEST_KEY) -> anthropic.AsyncAnthropic:
    """The Anthropic SDK brings its own HTTP client package (httpx2), over the same app."""
    http = httpx2.AsyncClient(transport=httpx2.ASGITransport(app=client.app), base_url="http://test")
    return anthropic.AsyncAnthropic(api_key=key, base_url="http://test", http_client=http, max_retries=0)


def jsonl(rows: list[dict]) -> bytes:
    return b"".join(json.dumps(r).encode() + b"\n" for r in rows)


def chat_line(cid: str, model: str = "auto", text: str = "hello", **body) -> dict:
    return {"custom_id": cid, "method": "POST", "url": "/v1/chat/completions",
            "body": {"model": model, "messages": [{"role": "user", "content": text}], **body}}


async def settle(gateway) -> None:
    """What the worker does: check every due job and settle what ended."""
    for j in list(gateway.store._jobs.values()) if hasattr(gateway.store, "_jobs") else []:
        j["next_poll_at"] = j["created_at"]
    if gateway.store.kind == "postgres":
        async with gateway.store.pool.acquire() as c:
            await c.execute("UPDATE batch_jobs SET next_poll_at = now() - interval '1 second'")
    await BatchPoller(gateway).run_once()


async def test_openai_sdk_batch_round_trip(app_client, gateway):
    client = oai(app_client)
    f = await client.files.create(file=("in.jsonl", io.BytesIO(jsonl([chat_line("a", "small"), chat_line("b", "big", "write a python function")]))),
                                  purpose="batch")
    assert f.purpose == "batch" and f.bytes > 0
    b = await client.batches.create(input_file_id=f.id, endpoint="/v1/chat/completions", completion_window="24h", metadata={"job": "nightly"})
    assert b.status == "in_progress" and b.request_counts.total == 2 and b.output_file_id is None
    assert b.model_extra["meritorder"]["providers"] == {"mock-openai": 1, "mock-anthropic": 1}
    assert b.model_extra["meritorder"]["estimated_cost_usd"] > 0
    # The input file's content is gone once the batch is with the providers.
    with pytest.raises(openai.NotFoundError):
        await client.files.content(f.id)
    await settle(gateway)
    b = await client.batches.retrieve(b.id)
    assert b.status == "completed" and b.request_counts.completed == 2 and b.metadata == {"job": "nightly"}
    out = (await client.files.content(b.output_file_id)).text
    rows = {json.loads(x)["custom_id"]: json.loads(x) for x in out.splitlines()}
    assert rows["a"]["response"]["status_code"] == 200
    assert rows["a"]["response"]["body"]["choices"][0]["message"]["content"].startswith("[small-1]")
    # "big" runs on an Anthropic-format backend: translated back to Chat Completions.
    assert rows["b"]["response"]["body"]["object"] == "chat.completion" and rows["b"]["response"]["body"]["model"] == "big"
    assert (await client.files.content(b.error_file_id)).text == ""
    page = await client.batches.list(limit=10)
    assert [x.id for x in page.data] == [b.id]
    rows = [r for r in await ledger(gateway) if r.get("batch_id") == b.id]
    assert len(rows) == 2 and all(r["service_tier"] == "batch" for r in rows)
    assert all(r["savings_batch"] > 0 for r in rows)
    assert all(r["actual_cost"] == pytest.approx(r["served_api_cost"] / 2, abs=1e-6) for r in rows)  # half off


async def test_anthropic_sdk_batch_round_trip(app_client, gateway):
    client = ant(app_client)
    b = await client.messages.batches.create(requests=[
        {"custom_id": "one", "params": {"model": "big", "max_tokens": 50, "messages": [{"role": "user", "content": "hi"}]}},
        {"custom_id": "two", "params": {"model": "small", "max_tokens": 50, "messages": [{"role": "user", "content": "yo"}]}},
    ])
    assert b.processing_status == "in_progress" and b.request_counts.processing == 2 and b.results_url is None
    await settle(gateway)
    b = await client.messages.batches.retrieve(b.id)
    assert b.processing_status == "ended" and b.request_counts.succeeded == 2 and b.results_url.endswith(f"/v1/messages/batches/{b.id}/results")
    results = {r.custom_id: r async for r in await client.messages.batches.results(b.id)}
    assert results["one"].result.type == "succeeded" and results["one"].result.message.content[0].text.startswith("[big-1]")
    # "small" runs on an OpenAI-format backend: translated to an Anthropic message.
    assert results["two"].result.message.type == "message" and results["two"].result.message.model == "small"
    listed = [x.id async for x in client.messages.batches.list()]
    assert listed == [b.id]
    deleted = await client.messages.batches.delete(b.id)
    assert deleted.id == b.id
    with pytest.raises(anthropic.NotFoundError):
        await client.messages.batches.retrieve(b.id)


def test_migration_11_sits_between_10_and_12():
    """Versions are applied by membership, in list order, so the list stays sorted and each version appears once."""
    from meritorder.store import MIGRATIONS

    versions = [v for v, _ in MIGRATIONS]
    assert versions == sorted(set(versions)) and versions.index(11) == versions.index(10) + 1 == versions.index(12) - 1
    assert versions.index(13) == versions.index(12) + 1 == versions.index(14) - 1  # decision 49's, between 12 and 14 (decision 51's)
    assert versions.index(14) == versions.index(15) - 1


@pytest.mark.skipif(not DATABASE_URL, reason="set MERITORDER_TEST_DATABASE_URL")
async def test_migration_11_runs_after_12_and_15_on_a_database_that_has_them():
    """A database migrated before batches has 12 (the learned router) and 15 (model compression) but not 11 or 13 (decision
    49's, which adds to 11's tables). Migrations go by membership, so 11 and then 13 run after them: they need nothing 12 and
    15 made and touch nothing of theirs."""
    store = await real_store()
    try:
        async with store.pool.acquire() as c:
            await c.execute("DROP TABLE IF EXISTS batch_files, batch_lines, batch_jobs, batches, batch_blobs CASCADE")
            await c.execute("ALTER TABLE usage_hourly DROP COLUMN IF EXISTS savings_batch")
            await c.execute("ALTER TABLE requests DROP COLUMN IF EXISTS service_tier, DROP COLUMN IF EXISTS batch_id, "
                            "DROP COLUMN IF EXISTS savings_batch")
            await c.execute("DELETE FROM schema_migrations WHERE version IN (11, 13)")
            assert {12, 15} <= {r["version"] for r in await c.fetch("SELECT version FROM schema_migrations")}
        await store.migrate()
        async with store.pool.acquire() as c:
            order = [r["version"] for r in await c.fetch("SELECT version FROM schema_migrations ORDER BY applied_at, version")]
            cols = {r["column_name"] for r in await c.fetch(
                "SELECT column_name FROM information_schema.columns WHERE table_name = 'requests'")}
            tables = {r["tablename"] for r in await c.fetch("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")}
            added = {(r["table_name"], r["column_name"]) for r in await c.fetch(
                "SELECT table_name, column_name FROM information_schema.columns WHERE table_name IN ('batches', 'batch_lines', 'batch_files')")}
        assert {("batches", "savings_self_hosting"), ("batch_lines", "tries"), ("batch_files", "storage"), ("batch_files", "sealed_with")} <= added
        assert order[-2:] == [11, 13] and {12, 15} <= set(order)
        assert {"service_tier", "batch_id", "savings_batch", "savings_compression_model"} <= cols
        assert {"batches", "batch_jobs", "batch_lines", "batch_files", "batch_blobs", "route_features", "router_models"} <= tables
        row = ledger_row(service_tier="batch", batch_id="batch_m", savings_batch=0.1, savings_compression_model=0.2)
        await store.record(row)
        got = next(r for r in await store.recent(10) if r.get("batch_id") == "batch_m")
        assert got["savings_batch"] == pytest.approx(0.1) and got["savings_compression_model"] == pytest.approx(0.2)
        await store.migrate()  # nothing left to apply
    finally:
        await store.close()


async def test_batch_tables_on_either_store(store):
    """The store's side of batches, on memory and (in CI) Postgres: leases, lines, settling, reservations, retention, files."""
    from datetime import UTC, datetime, timedelta

    now = datetime.now(UTC)
    b = {"id": "batch_t1", "created_at": now - timedelta(minutes=1), "user_id": "alice", "team": "default", "client_format": "openai",
         "endpoint": "/v1/chat/completions", "status": "in_progress", "request_count": 2, "estimated_cost": 0.5, "reserved_cost": 0.5,
         "reserve_month": now.strftime("%Y-%m"), "tags": {"project": "x"}, "metadata": {"k": "v"}}
    lines = [{"idx": 0, "custom_id": "a", "job_id": "bj_1", "kind": "chat", "model": "small", "backend": "mock-openai", "estimated_cost": 0.3,
              "status": "pending"},
             {"idx": 1, "custom_id": "b", "job_id": "bj_1", "kind": "chat", "estimated_cost": 0.2, "status": "pending"}]
    jobs = [{"id": "bj_1", "batch_id": "batch_t1", "backend": "mock-openai", "api": "mock", "status": "running", "request_count": 2,
             "next_poll_at": now - timedelta(seconds=1), "provider": {"format": "openai"}, "prices": {"small|small-1": {"standard": {"input": 1}}}}]
    created = await store.create_batch(b, lines, jobs)
    assert created["id"] == "batch_t1" and created["tags"] == {"project": "x"} and created["counts"] == {}
    await store.create_batch({**b, "id": "batch_t2", "created_at": now, "user_id": "bob"}, [], [])
    assert [x["id"] for x in await store.list_batches(user_id="alice")] == ["batch_t1"]
    assert [x["id"] for x in await store.list_batches(limit=1)] == ["batch_t2"]
    assert [x["id"] for x in await store.list_batches(after="batch_t2", client_format="openai")] == ["batch_t1"]
    assert await store.list_batches(team="default", start=now + timedelta(seconds=1)) == []
    assert len(await store.list_batches(end=now)) == 1
    month = b["reserve_month"]
    assert await store.batch_reserved(month) == 1.0 and await store.batch_reserved(month, "user_id", "alice") == 0.5
    assert await store.batch_reserved(month, "team", "default") == 1.0
    with pytest.raises(ValueError):
        await PostgresBatches.batch_reserved(store, "x", "key_id", "k")

    claimed = await store.claim_batch_jobs("w1", 5, 60)
    assert [j["id"] for j in claimed] == ["bj_1"] and claimed[0]["lease_owner"] == "w1" and claimed[0]["prices"]["small|small-1"]
    assert await store.claim_batch_jobs("w2") == []
    assert await store.update_batch_job("bj_1", "w2", polls=3) is None
    j = await store.update_batch_job("bj_1", "w1", release=True, polls=3, counts={"succeeded": 1})
    assert j["polls"] == 3 and j["lease_owner"] is None and j["counts"] == {"succeeded": 1}
    assert (await store.update_batch_job("bj_1"))["id"] == "bj_1" and await store.get_batch_job("nope") is None
    await store.add_batch_jobs([{"id": "bj_2", "batch_id": "batch_t1", "backend": "b", "api": "anthropic", "status": "failed"}])
    await store.add_batch_jobs([])
    assert [x["id"] for x in await store.batch_jobs("batch_t1")] == ["bj_1", "bj_2"]

    await store.update_batch_lines("batch_t1", [{"idx": 1, "job_id": None, "status": "refused", "error": "no", "error_code": 403},
                                                {"idx": 0, "upstream_model": "small-1"}, {"idx": 0}])
    got = await store.batch_lines("batch_t1")
    assert got[1]["status"] == "refused" and got[1]["error_code"] == 403 and got[0]["upstream_model"] == "small-1"
    assert [x["idx"] for x in await store.batch_lines("batch_t1", "bj_1")] == [0]

    row = ledger_row(actual_cost=0.1, baseline_cost=0.4, savings_batch=0.1, savings_routing=0.2, service_tier="batch", batch_id="batch_t1")
    released, actual, written = await store.settle_batch_lines("batch_t1", [{"idx": 0, "status": "succeeded", "row": row}])
    assert (released, actual) == (pytest.approx(0.3), pytest.approx(0.1)) and [r["batch_id"] for r in written] == ["batch_t1"]
    assert await store.settle_batch_lines("batch_t1", [{"idx": 0, "status": "succeeded", "row": row}]) == (0.0, 0.0, [])  # once a line
    t = await store.get_batch("batch_t1")
    assert t["reserved_cost"] == pytest.approx(0.2) and t["actual_cost"] == pytest.approx(0.1) and t["savings_total"] == pytest.approx(0.3)
    await store.add_batch_reserved("batch_t1", 0.05)  # by a difference, on top of what settling released
    assert (await store.get_batch("batch_t1"))["reserved_cost"] == pytest.approx(0.25)
    await store.add_batch_reserved("batch_t1", -1.0)
    await store.add_batch_reserved("nope", 1.0)
    assert (await store.get_batch("batch_t1"))["reserved_cost"] == 0
    await store.add_batch_reserved("batch_t1", 0.2)
    rows = [r for r in await store.recent(10) if r.get("batch_id") == "batch_t1"]
    assert len(rows) == 1 and rows[0]["service_tier"] == "batch"
    assert (await store.update_batch("batch_t1"))["id"] == "batch_t1" and await store.update_batch("nope", status="x") is None

    await store.update_batch("batch_t1", status="ended", ended_at=now - timedelta(days=9))
    assert [x["id"] for x in await store.batches_to_purge(now - timedelta(days=7))] == ["batch_t1"]
    await store.update_batch("batch_t1", purged_at=now)
    assert await store.batches_to_purge(now) == []

    f = await store.create_batch_file({"id": "file-1", "user_id": "alice", "team": "default", "purpose": "batch", "kind": "input",
                                       "bytes": 3, "filename": "in.jsonl", "created_at": now - timedelta(days=8)}, [b"ab", b"c"])
    assert f["id"] == "file-1" and await store.batch_file_chunks("file-1") == [b"ab", b"c"]
    assert [x["id"] for x in await store.list_batch_files("alice", "batch")] == ["file-1"]
    assert [x["id"] for x in await store.stale_batch_files(now - timedelta(days=7))] == ["file-1"]
    await store.drop_batch_file("file-1")
    assert await store.batch_file_chunks("file-1") == [] and (await store.get_batch_file("file-1"))["deleted_at"]
    assert await store.list_batch_files("alice") == [] and await store.get_batch_file("nope") is None
    g = await store.create_batch_file({"id": "file-2", "user_id": "alice", "team": "default", "purpose": "batch", "kind": "input",
                                       "storage": "s3://b/p", "sealed_with": "abcd"}, [])
    assert g["storage"] == "s3://b/p" and [(x["id"], x["sealed_with"]) for x in await store.held_batch_files()] == [("file-2", "abcd")]

    # GPU work held sealed (content.py): blobs by name, listed and deleted by prefix (a job's folder)
    await store.put_batch_blob("batch_t1/bj_1/requests/0", [b"x", b"y"])
    await store.put_batch_blob("batch_t1/bj_1/results/0", [b"z"])
    await store.put_batch_blob("batch_t1/bj_1/results/0", [b"z2"])  # written again: replaced
    await store.put_batch_blob("batch_t2/bj_9/requests/0", [b"w"])
    assert await store.get_batch_blob("batch_t1/bj_1/requests/0") == [b"x", b"y"] and await store.get_batch_blob("nope") is None
    assert await store.get_batch_blob("batch_t1/bj_1/results/0") == [b"z2"]
    assert await store.batch_blob_names("batch_t1/") == ["batch_t1/bj_1/requests/0", "batch_t1/bj_1/results/0"]
    await store.delete_batch_blobs("batch_t1/bj_1/")
    assert await store.batch_blob_names("batch_t") == ["batch_t2/bj_9/requests/0"]
