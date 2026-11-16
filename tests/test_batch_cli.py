"""`meritorder batch`: submit, watch, list, fetch and cancel, against a real gateway on the mock backends."""

import asyncio
import json
from datetime import UTC, datetime, timedelta

import pytest
from conftest import TEST_KEY
from test_cli import LiveGateway

from meritorder import cli
from meritorder.batches.poller import BatchPoller


@pytest.fixture
def live(cfg):
    with LiveGateway(cfg) as g:
        yield g


def settle(gw) -> None:
    """The worker's pass, from the test's thread (the stores are in memory)."""
    for j in gw.store._jobs.values():
        j["next_poll_at"] = datetime.now(UTC) - timedelta(seconds=1)
    asyncio.run(BatchPoller(gw).run_once())


def run(live, *args) -> None:
    cli.main(["batch", *args, "--gateway", live.url, "--key", TEST_KEY])


def test_submit_watch_and_fetch_an_openai_batch(live, tmp_path, capsys, monkeypatch):
    f = tmp_path / "in.jsonl"
    f.write_text("\n".join(json.dumps({"custom_id": f"q{i}", "method": "POST", "url": "/v1/chat/completions",
                                       "body": {"model": "small", "messages": [{"role": "user", "content": "hi"}]}}) for i in range(2)) + "\n")
    monkeypatch.setattr(cli.time, "sleep", lambda s: settle(live.gw))  # each wait in watch is a worker pass
    run(live, "submit", str(f), "--watch", "--interval", "1")
    out = capsys.readouterr().out
    bid = out.split()[0]
    assert bid.startswith("batch_") and "in_progress" in out and "[mock-openai 2]" in out and "estimated $" in out
    assert "completed  total 2 completed 2 failed 0" in out and f"meritorder batch results {bid}" in out and "batch discount $" in out
    dest = tmp_path / "out.jsonl"
    run(live, "results", bid, "--out", str(dest))
    rows = [json.loads(x) for x in dest.read_text().splitlines()]
    assert capsys.readouterr().out.startswith("2 results written") and {r["custom_id"] for r in rows} == {"q0", "q1"}
    run(live, "results", bid, "--errors-only")
    assert capsys.readouterr().out == ""
    run(live, "list")
    assert bid in capsys.readouterr().out
    run(live, "cancel", bid)  # already ended: it stays completed
    assert "completed" in capsys.readouterr().out


def test_submit_an_anthropic_batch_and_cancel_it(live, tmp_path, capsys):
    f = tmp_path / "in.jsonl"
    f.write_text(json.dumps({"custom_id": "a", "params": {"model": "big", "max_tokens": 20, "messages": [{"role": "user", "content": "hi"}]}}))
    run(live, "submit", str(f))
    bid = capsys.readouterr().out.split()[0]
    assert bid.startswith("msgbatch_")
    with pytest.raises(SystemExit, match="hasn't finished yet"):
        run(live, "results", bid)
    run(live, "cancel", bid)
    assert "canceling" in capsys.readouterr().out
    settle(live.gw)
    run(live, "watch", bid)
    assert "Done." in capsys.readouterr().out
    run(live, "results", bid)
    assert json.loads(capsys.readouterr().out)["custom_id"] == "a"


def test_batch_command_errors(live, tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n")
    with pytest.raises(SystemExit, match="has no JSON lines"):
        run(live, "submit", str(empty))
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"custom_id": "a", "params": {"model": "big", "messages": []}}))
    with pytest.raises(SystemExit, match="^400"):
        run(live, "submit", str(bad))
    with pytest.raises(SystemExit, match="^404"):
        run(live, "watch", "batch_nope")
    with pytest.raises(SystemExit, match="^404"):
        run(live, "cancel", "msgbatch_nope")
    ok = tmp_path / "ok.jsonl"
    ok.write_text(json.dumps({"custom_id": "a", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "small", "messages": []}}))
    live.gw.cfg.batches.enabled = False
    with pytest.raises(SystemExit, match="upload failed: 403"):
        run(live, "submit", str(ok))
