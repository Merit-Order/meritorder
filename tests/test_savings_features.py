import uuid

from conftest import LEAD_KEY, auth, last_row, ledger

from meritorder.compression import CompressionConfig, compress_text, compress_tool_results
from meritorder.response_cache import ResponseCacheConfig, cache_key
from meritorder.tokens import text_tokens

CFG = CompressionConfig(enabled=True, min_chars=200, max_chars=2000)


def test_compress_text_is_deterministic_and_bounded():
    log = "\x1b[32mok\x1b[0m\n" + "Downloading... 50%\n" * 400 + "".join(f"line {i}\n" for i in range(3000))
    a, b = compress_text(log, CFG), compress_text(log, CFG)
    assert a == b
    assert "\x1b[" not in a
    assert "repeated 399 more times" in a
    assert "omitted by Merit Order" in a and len(a) < 2200
    assert compress_text("short", CFG) == "short"


def test_rules_output_is_remembered_per_text_and_settings(monkeypatch):
    from meritorder import compression

    log = "Downloading... 50%\n" * 400 + "".join(f"line {i}\n" for i in range(3000))
    monkeypatch.setattr(compression, "_remembered", {})
    monkeypatch.setattr(compression, "_remembered_chars", 0)
    a = compress_text(log, CFG)
    assert a == compression._apply_rules(log, CFG) and compression._remembered_chars == len(log) + len(a)
    assert compress_text(log, CFG) is a  # the next turn: remembered, not worked out again
    other = CompressionConfig(enabled=True, min_chars=200, max_chars=4000)
    assert compress_text(log, other) == compression._apply_rules(log, other) != a  # other settings, other output
    monkeypatch.setattr(compression, "REMEMBERED_CHARS", len(log) + 2500)  # room for about one: the oldest goes
    compress_text(log + "x", CFG)
    assert len(compression._remembered) == 1 and compression._remembered_chars <= len(log) + 2500


def test_compress_only_tool_results_and_tracks_new_vs_history():
    big = "x" * 50 + "\n" + "data row\n" * 1000
    body = {"messages": [
        {"role": "user", "content": big},  # user text is never touched
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "f", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": big}]},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t2", "name": "f", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t2", "content": [{"type": "text", "text": big}]}]},
    ]}
    out, res = compress_tool_results(body, "anthropic", CFG)
    assert out["messages"][0]["content"] == big
    assert len(out["messages"][2]["content"][0]["content"]) < len(big)
    assert res.removed_chars_history > 0 and res.removed_chars_new > 0
    assert body["messages"][2]["content"][0]["content"] == big  # original not mutated
    again, _ = compress_tool_results(body, "anthropic", CFG)
    assert again == out  # same history every turn -> provider cache keeps hitting
    # Removed tokens are estimated the way the gateway counts prompts (meritorder.tokens), for savings and calibration.
    removed = text_tokens(big) - text_tokens(out["messages"][4]["content"][0]["content"][0]["text"])
    assert res.removed_tokens_new == round(removed) and res.removed_tokens == res.removed_tokens_new + res.removed_tokens_history
    small = {"messages": [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "ok"},
                                                       {"type": "tool_result", "tool_use_id": "u", "content": None}]}]}
    assert compress_tool_results(small, "anthropic", CFG) == (small, compress_tool_results(small, "anthropic", CFG)[1])
    assert compress_tool_results(small, "anthropic", CFG)[1].removed_tokens == 0


def test_cache_key_scoping():
    body = {"model": "m", "messages": [{"role": "user", "content": "q"}], "stream": True}
    team = ResponseCacheConfig(scope="team")
    user = ResponseCacheConfig(scope="user")
    assert cache_key("openai", body, "t", "alice", team) == cache_key("openai", {**body, "stream": False}, "t", "bob", team)
    assert cache_key("openai", body, "t", "alice", user) != cache_key("openai", body, "t", "bob", user)
    assert cache_key("openai", body, "t1", "alice", team) != cache_key("openai", body, "t2", "alice", team)


async def test_compression_savings_in_ledger(client, gateway):
    gateway.cfg.team("default").compression = CompressionConfig(enabled=True, min_chars=200, max_chars=1000)
    big = "result line\n" * 2000
    body = {"model": "small", "messages": [
        {"role": "user", "content": "run it"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "sh", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": big},
    ]}
    r = await client.post("/v1/chat/completions", headers=auth(), json=body)
    assert r.status_code == 200
    await gateway.drain()
    row = (await last_row(gateway))
    assert row["savings_compression"] > 0
    assert row["baseline_cost"] - row["actual_cost"] > 0


async def test_response_cache_hit(client, gateway):
    body = {"model": "small", "messages": [{"role": "user", "content": "what is our refund policy?"}]}
    headers = {**auth(), "x-meritorder-cache": "on"}
    r1 = await client.post("/v1/chat/completions", headers=headers, json=body)
    await gateway.drain()
    r2 = await client.post("/v1/chat/completions", headers=headers, json=body)
    assert r2.headers["x-meritorder-backend"] == "response-cache"
    assert r2.json()["choices"] == r1.json()["choices"]
    r3 = await client.post("/v1/chat/completions", headers=headers, json={**body, "stream": True})
    assert r3.headers["x-meritorder-backend"] == "response-cache"
    assert r3.text.strip().endswith("data: [DONE]")
    # another team member hits the team-scoped cache; bypass with "off"
    r4 = await client.post("/v1/chat/completions", headers={**auth(LEAD_KEY), "x-meritorder-cache": "on"}, json=body)
    assert r4.headers["x-meritorder-backend"] == "response-cache"
    r5 = await client.post("/v1/chat/completions", headers={**auth(), "x-meritorder-cache": "off"}, json=body)
    assert r5.headers["x-meritorder-backend"] == "mock-openai"
    await gateway.drain()
    hits = [r for r in await ledger(gateway) if r["backend"] == "response-cache"]
    assert len(hits) == 3 and all(h["savings_response_cache"] > 0 and h["actual_cost"] == 0 for h in hits)


async def test_metrics_endpoint(client, gateway):
    await client.post("/v1/chat/completions", headers=auth(), json={"model": "small", "messages": [{"role": "user", "content": "hi"}]})
    await gateway.drain()
    text = (await client.get("/metrics")).text
    assert 'meritorder_requests_total{backend="mock-openai",model="small",status="200"} 1' in text
    assert "meritorder_request_duration_seconds_bucket" in text


async def test_alerts(gateway):
    from datetime import UTC, datetime

    from meritorder.alerts import check_budgets, check_spikes, weekly_digest

    now = datetime.now(UTC)
    for _ in range(12):
        await gateway.store.record({"id": uuid.uuid4(), "tags": {}, "ts": now, "user_id": "alice", "team": "default", "actual_cost": 1.0, "baseline_cost": 2.0,
                                    "savings_routing": 1.0, "status": 200})
    await gateway.store.rollup()  # the worker refreshes the rollup before checking budgets
    budget = await check_budgets(gateway.cfg, gateway.store, gateway.state)
    assert any("alice" in m and "120%" in m for m in budget)  # $12 of a $10 user budget
    assert await check_budgets(gateway.cfg, gateway.store, gateway.state) == []  # deduped
    spikes = await check_spikes(gateway.store, gateway.state)
    assert spikes and "alice" in spikes[0]
    digest = await weekly_digest(gateway.cfg, gateway.store, gateway.state, force=True)
    assert "Saved $12.00" in digest[0]
