"""Data regressions: stored mode's retention and key rotation, the export's file handling, and the in-memory store
answering as Postgres does. Tests on the ``store`` fixture also run on Postgres in CI."""

from __future__ import annotations

import copy
import types
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from conftest import TEST_CONFIG, ledger_row

from meritorder import cli
from meritorder.app import build_gateway
from meritorder.config import build_config
from meritorder.sealing import Keyring
from meritorder.state import MemoryState
from meritorder.store import MemoryStore
from meritorder.stored.service import StoredResponses


def stored_gateway(tmp_path, teams: dict):
    raw = copy.deepcopy(TEST_CONFIG)
    raw["teams"]["default"]["stored_responses"] = {"enabled": True, "retention_days": 30}
    raw["teams"].update(teams)
    gw = build_gateway(build_config(raw, tmp_path, quiet=True), store=MemoryStore(), state=MemoryState())
    return gw, StoredResponses(gw)


def stored_row(rid: str, team: str, when: datetime, **kw) -> dict:
    return {"id": rid, "user_id": "bob", "team": team, "thread_id": rid, "status": "completed", "seal_key": "k", "content": b"sealed",
            "touched_at": when, "created_at": when, **kw}


# ---------------------------------------------------------------- stored mode: retention of a team that turned it off


async def test_purge_deletes_everything_of_a_configured_team_whose_stored_mode_is_off(tmp_path, monkeypatch):
    """A configured team with stored mode off gets a cutoff of now: what it kept goes at the next purge, providers' copies too,
    rather than staying for the default team's retention."""
    monkeypatch.setenv("MERITORDER_RESPONSES_KEY", "k")
    gw, s = stored_gateway(tmp_path, {"eng": {"stored_responses": {"enabled": False}}})
    yesterday = datetime.now(UTC) - timedelta(days=1)
    await gw.store.save_stored_response(stored_row("resp_eng", "eng", yesterday, backend="openai", upstream_id="resp_up"))
    await gw.store.save_stored_response(stored_row("resp_default", "default", yesterday))
    assert s.expired(await gw.store.get_stored_response("resp_eng"))
    copies = []

    async def delete_copies(rows, backends=None):
        copies.extend(r["id"] for r in rows)
        return len(rows)

    monkeypatch.setattr(s, "delete_provider_copies", delete_copies)
    deleted, _ = await s.purge({})
    assert deleted == 1 and copies == ["resp_eng"]
    assert await gw.store.get_stored_response("resp_eng") is None
    assert await gw.store.get_stored_response("resp_default") is not None  # within the default team's 30 days


async def test_purge_gives_teams_not_in_the_config_the_default_teams_retention(tmp_path, monkeypatch):
    monkeypatch.setenv("MERITORDER_RESPONSES_KEY", "k")
    gw, s = stored_gateway(tmp_path, {})
    now = datetime.now(UTC)
    await gw.store.save_stored_response(stored_row("resp_recent", "unlisted", now - timedelta(days=1)))
    await gw.store.save_stored_response(stored_row("resp_old", "unlisted", now - timedelta(days=31)))
    deleted, _ = await s.purge({})
    assert deleted == 1
    assert await gw.store.get_stored_response("resp_recent") is not None
    assert await gw.store.get_stored_response("resp_old") is None


# ---------------------------------------------------------------- stored mode: resealing past values no key opens


async def test_reseal_gets_past_values_it_cannot_open_and_alerts_once_per_unknown_key(tmp_path, monkeypatch):
    """Values sealed with a key that's not on the ring are left out of the query, so they don't take the front of every
    pass: values an old key on the ring sealed are still sealed again. Each unknown key raises one critical alert."""
    monkeypatch.setenv("MERITORDER_RESPONSES_KEY", "new")
    monkeypatch.setenv("MERITORDER_RESPONSES_OLD_KEYS", "old")
    gw, s = stored_gateway(tmp_path, {})
    gone, old = Keyring("gone"), Keyring("old")
    for rid, ring in (("resp_a", gone), ("resp_b", old)):  # one sealed with a key nobody has any more, one with the old key
        row = {"id": rid, "user_id": "bob", "team": "default", "thread_id": rid, "status": "completed"}
        blob = ring.seal(b'{"response": {}, "input": []}', s.response_aad(row).encode())
        await gw.store.save_stored_response({**row, "seal_key": ring.current, "content": blob, "bytes": len(blob)})
    for _ in range(5):
        await s.reseal(limit=1)
    assert (await gw.store.get_stored_response("resp_b"))["seal_key"] == s.keyring().current
    assert (await gw.store.get_stored_response("resp_a"))["seal_key"] == gone.current  # left as it was
    alerts = [a for a in await gw.state.recent_alerts() if gone.current in a["title"]]
    assert len(alerts) == 1 and alerts[0]["level"] == "critical"
    assert await s.unopenable_keys(s.keyring()) == [gone.current]


async def test_values_to_reseal_are_those_an_old_key_on_the_ring_sealed(store):
    for rid, key in (("resp_new", "cur"), ("resp_old", "old"), ("resp_gone", "gone")):
        await store.save_stored_response({"id": rid, "user_id": "bob", "team": "default", "thread_id": rid, "status": "completed",
                                          "seal_key": key, "content": b"x", "bytes": 1})
    assert [v["id"] for v in await store.stored_to_reseal(["old"], 10)] == ["resp_old"]
    assert await store.stored_seal_keys() == {"cur", "old", "gone"}


# ---------------------------------------------------------------- export: a failed export keeps the file that was there


def _export(monkeypatch, out, lines):
    monkeypatch.setattr(cli, "_gateway", lambda args: "http://gw")
    monkeypatch.setattr(cli, "_key", lambda args: "k")

    class Resp:
        status_code = 200

        def iter_lines(self):
            yield from lines()

    class Stream:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return Resp()

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(cli.httpx, "stream", Stream)
    cli.cmd_export(types.SimpleNamespace(format="jsonl", start=None, end=None, out=str(out)))


def test_a_failed_export_leaves_the_file_that_was_there(tmp_path, monkeypatch):
    out = tmp_path / "october.jsonl"
    out.write_text('{"id": "kept"}\n')

    def lines():
        yield '{"id": "1"}'
        raise httpx.ReadError("connection lost")

    with pytest.raises(httpx.ReadError):
        _export(monkeypatch, out, lines)
    assert out.read_text() == '{"id": "kept"}\n'
    assert not (tmp_path / "october.jsonl.partial").exists()


def test_a_whole_export_replaces_the_file(tmp_path, monkeypatch):
    out = tmp_path / "october.jsonl"
    out.write_text('{"id": "old"}\n')
    _export(monkeypatch, out, lambda: iter(['{"id": "1"}', '{"id": "2"}']))
    assert out.read_text() == '{"id": "1"}\n{"id": "2"}\n'
    assert not (tmp_path / "october.jsonl.partial").exists()


# ---------------------------------------------------------------- the in-memory store answers as Postgres does


async def test_recent_is_newest_first_whatever_the_order_rows_were_written(store):
    now = datetime.now(UTC)
    await store.record(ledger_row(user_id="newest", ts=now))
    await store.record(ledger_row(user_id="older", ts=now - timedelta(minutes=5)))
    assert [r["user_id"] for r in await store.recent(10)] == ["newest", "older"]
    assert [r["user_id"] for r in await store.recent(1)] == ["newest"]


async def test_summary_time_keys_are_the_utc_timestamp_each_period_starts_at(store):
    ts = datetime.now(UTC).replace(hour=13, minute=20, second=0, microsecond=0) - timedelta(days=1)
    await store.record(ledger_row(ts=ts))
    start, end = ts - timedelta(hours=2), ts + timedelta(hours=2)
    assert (await store.summary(start, end, "hour"))[0]["key"] == ts.replace(minute=0).isoformat()
    assert (await store.summary(start, end, "day"))[0]["key"] == ts.replace(hour=0, minute=0).isoformat()
    assert (await store.summary(start, end, "month"))[0]["key"] == ts.replace(day=1, hour=0, minute=0).isoformat()


async def test_summary_puts_a_missing_key_last(store):
    now = datetime.now(UTC)
    await store.record(ledger_row(served_model=None, ts=now))
    await store.record(ledger_row(served_model="big", ts=now))
    rows = await store.summary(now - timedelta(hours=1), now + timedelta(hours=1), "model")
    assert [r["key"] for r in rows] == ["big", None]


async def test_summary_over_the_rollup_keeps_a_missing_key_missing(store):
    """The hourly rollup stores a missing model as '' (it's in the primary key); a long range reads the rollup, and
    must answer None, as a short one over the raw ledger does."""
    now = datetime.now(UTC)
    await store.record(ledger_row(served_model=None, ts=now - timedelta(days=2)))
    await store.rollup(full=True)
    long = await store.summary(now - timedelta(days=30), now + timedelta(hours=1), "model")
    short = await store.summary(now - timedelta(days=3), now + timedelta(hours=1), "model")
    assert [r["key"] for r in long] == [r["key"] for r in short] == [None]


async def test_summary_by_tag_keeps_an_empty_value_apart_from_untagged(store):
    now = datetime.now(UTC)
    await store.record(ledger_row(tags={"project": ""}, ts=now))
    await store.record(ledger_row(ts=now))
    rows = await store.summary(now - timedelta(hours=1), now + timedelta(hours=1), "tag:project")
    assert sorted(r["key"] for r in rows) == ["", "(untagged)"]


async def test_spend_by_tag_counts_a_tag_that_is_there_with_an_empty_value(store):
    now = datetime.now(UTC)
    await store.record(ledger_row(user_id="judge", tags={"judged": ""}, actual_cost=1.0, ts=now))
    await store.record(ledger_row(user_id="judge", tags={"judged": "replay"}, actual_cost=2.0, ts=now))
    await store.record(ledger_row(user_id="judge", actual_cost=4.0, ts=now))
    assert await store.spend_by_tag(["judge"], "judged", now - timedelta(hours=1)) == {"": 1.0, "replay": 2.0}


async def test_a_page_after_an_unknown_cursor_is_empty_on_every_store(store):
    await store.create_batch({"id": "batch_1", "user_id": "a", "team": "t", "client_format": "openai", "status": "ended",
                              "created_at": datetime.now(UTC)}, [], [])
    assert await store.list_batches(after="batch_gone") == []
    assert [b["id"] for b in await store.list_batches()] == ["batch_1"]
