import json

from conftest import LEAD_KEY, auth, last_row, ledger, ledger_row

from meritorder.sessions import session_key

from meritorder.proxy import _month_key


def sse_events(text: str) -> list[tuple[str | None, str]]:
    out = []
    for block in text.strip().split("\n\n"):
        name, data = None, []
        for line in block.split("\n"):
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                data.append(line[6:])
        out.append((name, "\n".join(data)))
    return out


async def test_requires_key(client):
    r = await client.post("/v1/messages", json={"model": "big", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 401
    assert r.json()["type"] == "error"


async def test_anthropic_passthrough(client, gateway):
    r = await client.post("/v1/messages", headers=auth(), json={"model": "claude-big-latest", "max_tokens": 10,
                                                                 "messages": [{"role": "user", "content": "hello"}]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["type"] == "message"
    assert body["model"] == "big-1"
    assert r.headers["x-meritorder-backend"] == "mock-anthropic"
    await gateway.drain()
    row = (await last_row(gateway))
    assert row["proxy_cache"] is True  # automatic caching was added
    assert row["served_model"] == "big"


async def test_openai_client_on_anthropic_backend_streaming(client, gateway):
    body = {"model": "big", "stream": True, "stream_options": {"include_usage": True},
            "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "please use a tool"}],
            "tools": [{"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}]}
    r = await client.post("/v1/chat/completions", headers=auth(), json=body)
    assert r.status_code == 200
    events = sse_events(r.text)
    assert events[-1][1] == "[DONE]"
    chunks = [json.loads(d) for _, d in events[:-1]]
    calls = [tc for c in chunks if c["choices"] for tc in c["choices"][0]["delta"].get("tool_calls", [])]
    assert calls[0]["function"]["name"] == "search"
    assert chunks[-1]["usage"]["prompt_tokens"] > 0


async def test_anthropic_client_on_openai_backend_streaming(client):
    body = {"model": "small", "max_tokens": 100, "stream": True, "messages": [{"role": "user", "content": "hi there"}]}
    r = await client.post("/v1/messages", headers=auth(), json=body)
    assert r.status_code == 200
    names = [n for n, _ in sse_events(r.text)]
    assert names[0] == "message_start" and names[-1] == "message_stop"
    assert "content_block_delta" in names


async def test_openai_passthrough_hides_usage_chunk_unless_asked(client):
    r = await client.post("/v1/chat/completions", headers=auth(),
                          json={"model": "small", "stream": True, "messages": [{"role": "user", "content": "x" * 2000}]})
    datas = [d for _, d in sse_events(r.text)]
    assert datas[-1] == "[DONE]"
    assert all("usage" not in json.loads(d) or json.loads(d)["choices"] for d in datas[:-1])


async def test_caching_savings_on_second_turn(client, gateway):
    convo = [{"role": "user", "content": "long context " * 400}]
    headers = {**auth(), "x-session-id": "sess-1"}
    r1 = await client.post("/v1/messages", headers=headers, json={"model": "big", "max_tokens": 50, "messages": convo})
    assert r1.status_code == 200
    convo += [{"role": "assistant", "content": r1.json()["content"]}, {"role": "user", "content": "and now?"}]
    r2 = await client.post("/v1/messages", headers=headers, json={"model": "big", "max_tokens": 50, "messages": convo})
    usage = r2.json()["usage"]
    assert usage["cache_read_input_tokens"] > 0
    await gateway.drain()
    second = (await last_row(gateway))
    assert second["savings_caching"] > 0
    session = await gateway.state.get_session(session_key("alice", "sess-1"))
    assert session["turns"] == 2 and "mock-anthropic" in session["warm_until"]


async def test_auto_routing_and_reports(client, gateway):
    r = await client.post("/v1/chat/completions", headers=auth(),
                          json={"model": "auto", "messages": [{"role": "user", "content": "what is 2+2?"}]})
    assert r.status_code == 200
    assert r.headers["x-meritorder-model"] == "small"  # trivial question -> cheap model
    assert float(r.headers["x-meritorder-savings-usd"]) > 0  # baseline is the big model
    await gateway.drain()
    await gateway.store.rollup()  # on Postgres, month-long reports read the hourly rollup the worker keeps current
    me = (await client.get("/v1/reports/me", headers=auth())).json()
    assert me["requests"] == 1 and me["savings_usd"] > 0
    summary = (await client.get("/v1/reports/summary?group_by=user", headers=auth(LEAD_KEY))).json()
    assert summary["rows"][0]["key"] == "alice"
    # a regular user only ever sees their own rows, whatever they ask for
    mine = (await client.get("/v1/reports/summary?group_by=user&user=lee", headers=auth())).json()
    assert {row["key"] for row in mine["rows"]} == {"alice"}
    csv = await client.get("/v1/reports/summary.csv?group_by=model", headers=auth())
    assert csv.text.startswith("key,requests")


async def test_budget_enforced(client, gateway):
    await gateway.state.seed_spend(f"{_month_key()}:user:alice", 10.0)
    r = await client.post("/v1/messages", headers=auth(), json={"model": "big", "max_tokens": 5, "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 429
    assert "budget" in r.json()["error"]["message"]


async def test_budget_still_binds_after_redis_loses_the_spend_counters(client, gateway):
    """Redis can lose the month's counters (a flush, a failover, eviction: they carry a TTL). A request that finishes
    then must not start them again from its own cost, or the next one reads that instead of the ledger's total."""
    from meritorder.state import MemoryState

    await gateway.store.record(ledger_row(actual_cost=9.999))  # alice's month so far; her limit is $10
    mock = gateway.backends["mock-anthropic"]
    send = mock.send

    async def lose_counters_then_send(*a, **kw):
        if isinstance(gateway.state, MemoryState):
            gateway.state._spend.clear()
        else:
            await gateway.state.r.flushdb()
        return await send(*a, **kw)

    mock.send = lose_counters_then_send
    body = {"model": "big", "max_tokens": 20, "messages": [{"role": "user", "content": "word " * 3000}]}
    r = await client.post("/v1/messages", headers=auth(), json=body)
    assert r.status_code == 200 and float(r.headers["x-meritorder-cost-usd"]) > 0.001  # the ledger is now over $10
    await gateway.drain()
    mock.send = send
    r = await client.post("/v1/messages", headers=auth(), json=body)
    assert r.status_code == 429 and "monthly budget of $10.00 reached for alice" in r.json()["error"]["message"]


async def test_unknown_model(client):
    r = await client.post("/v1/chat/completions", headers=auth(), json={"model": "gpt-nope", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 404
    assert r.json()["error"]["type"] == "not_found_error"


async def test_models_list(client):
    r = await client.get("/v1/models", headers=auth())
    models = {m["id"]: m.get("type") for m in r.json()["data"]}
    assert list(models) == ["auto", "big", "embed-small", "small"]
    assert models["embed-small"] == "embedding" and models["big"] == "chat"


async def test_failed_backend_is_skipped_for_other_models(client, gateway):
    from meritorder.backends.base import BackendError

    calls = []
    big = gateway.backends["mock-anthropic"]

    async def broken(body, stream, headers):
        calls.append(body["model"])
        raise BackendError(502, "mock-anthropic: credentials expired", retryable=True)

    big.send = broken
    gateway.cfg.model("small").quality = {"coding": 5, "general": 5}  # make both candidates for auto
    gateway.cfg.model("big").routes.append(gateway.cfg.model("big").routes[0].model_copy(update={"model": "big-2"}))
    r = await client.post("/v1/chat/completions", headers={**auth(), "x-route-floor": "general:5"},
                          json={"model": "auto", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    assert r.headers["x-meritorder-backend"] == "mock-openai"
    assert len(calls) <= 1  # never retried the same broken backend


async def test_rate_limits(client, gateway):
    from meritorder.config import RateLimits

    gateway.cfg.team("default").rate_limits = RateLimits(user_rpm=2)
    body = {"model": "small", "messages": [{"role": "user", "content": "hi"}]}
    codes = [(await client.post("/v1/chat/completions", headers=auth(), json=body)).status_code for _ in range(3)]
    assert codes[:2] == [200, 200] and codes[2] == 429
    r = await client.post("/v1/chat/completions", headers=auth(), json=body)
    assert int(r.headers["retry-after"]) >= 1
    assert r.json()["error"]["type"] == "rate_limit_error"
    gateway.cfg.team("default").rate_limits = RateLimits(team_tpm=50)
    big = {"model": "small", "messages": [{"role": "user", "content": "x" * 400}]}
    assert (await client.post("/v1/chat/completions", headers=auth(LEAD_KEY), json=big)).status_code == 429


async def test_admin_keys(client, gateway):
    from meritorder.store import MemoryStore

    gateway.store.kind = "postgres"  # the admin API requires a durable store; MemoryStore behaves the same here
    try:
        r = await client.post("/v1/admin/keys", headers=auth(LEAD_KEY), json={"user": "newbie@example.com"})
        assert r.status_code == 201, r.text
        secret, key_id = r.json()["api_key"], r.json()["id"]
        assert (await client.get("/v1/reports/me", headers=auth(secret))).json()["user"] == "newbie@example.com"
        # team leads can't mint admins or keys for other teams
        assert (await client.post("/v1/admin/keys", headers=auth(LEAD_KEY), json={"user": "x", "role": "admin"})).status_code == 403
        assert (await client.post("/v1/admin/keys", headers=auth(LEAD_KEY), json={"user": "x", "team": "other"})).status_code == 403
        # regular users can't use the admin API at all
        assert (await client.get("/v1/admin/keys", headers=auth())).status_code == 403
        keys = (await client.get("/v1/admin/keys", headers=auth(LEAD_KEY))).json()["keys"]
        assert any(k["id"] == key_id and k["status"] == "active" for k in keys)
        assert all("key_hash" not in k for k in keys)
        assert (await client.delete(f"/v1/admin/keys/{key_id}", headers=auth(LEAD_KEY))).json()["revoked"] is True
        assert (await client.get("/v1/reports/me", headers=auth(secret))).status_code == 401  # revoked immediately
    finally:
        gateway.store.kind = MemoryStore.kind


async def test_embeddings(client, gateway):
    r = await client.post("/v1/embeddings", headers=auth(), json={"model": "embed-small", "input": ["hello", "world"]})
    assert r.status_code == 200, r.text
    data = r.json()
    assert len(data["data"]) == 2 and data["model"] == "embed-small"
    assert r.headers["x-meritorder-backend"] == "mock-openai"
    again = (await client.post("/v1/embeddings", headers=auth(), json={"model": "embed-small", "input": "hello"})).json()
    assert again["data"][0]["embedding"] == data["data"][0]["embedding"]
    await gateway.drain()
    row = (await last_row(gateway))
    assert row["client_format"] == "embeddings" and row["input_tokens"] > 0 and row["actual_cost"] is not None
    # chat models aren't embedding models and vice versa; auto never picks an embedding model
    assert (await client.post("/v1/embeddings", headers=auth(), json={"model": "small", "input": "x"})).status_code == 404
    r = await client.post("/v1/chat/completions", headers=auth(), json={"model": "embed-small", "messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 400


async def test_session_spend_cap(client, gateway):
    gateway.cfg.team("default").max_session_usd = 0.0005
    headers = {**auth(), "x-session-id": "looping-agent"}
    body = {"model": "big", "max_tokens": 50, "messages": [{"role": "user", "content": "go " * 200}]}
    codes = []
    for _ in range(6):
        codes.append((await client.post("/v1/messages", headers=headers, json=body)).status_code)
        await gateway.drain()
    assert codes[0] == 200 and 429 in codes
    other = await client.post("/v1/messages", headers={**auth(), "x-session-id": "fresh"}, json=body)
    assert other.status_code == 200  # only that session is stopped


async def test_auto_effort_is_lowered_and_session_stable(client, gateway):
    sent = []
    backend = gateway.backends["mock-anthropic"]
    original = backend.send

    async def spy(body, stream, headers):
        sent.append((body.get("output_config") or {}).get("effort"))
        return await original(body, stream, headers)

    backend.send = spy
    gateway.cfg.model("big").effort_levels = ["low", "medium", "high"]
    gateway.cfg.model("small").quality = {"coding": 0, "general": 0}  # make `big` the only candidate
    headers = {**auth(), "x-session-id": "effort-sess"}
    convo = [{"role": "user", "content": "hi"}]
    await client.post("/v1/messages", headers=headers, json={"model": "auto", "max_tokens": 20, "messages": convo})
    await gateway.drain()
    # a harder follow-up in the same session keeps the same effort (changing it would break the cache)
    convo += [{"role": "assistant", "content": "hello"}, {"role": "user", "content": "now design the architecture of a distributed system " * 5}]
    await client.post("/v1/messages", headers=headers, json={"model": "auto", "max_tokens": 20, "messages": convo})
    # the client's own choice always wins
    await client.post("/v1/messages", headers=auth(), json={"model": "auto", "max_tokens": 20, "output_config": {"effort": "high"},
                                                             "messages": [{"role": "user", "content": "hi"}]})
    # an explicitly named model is never touched
    await client.post("/v1/messages", headers=auth(), json={"model": "big", "max_tokens": 20, "messages": [{"role": "user", "content": "hi"}]})
    assert sent == ["low", "low", "high", None]


async def test_ledger_export(client, gateway):
    import json as _json

    await client.post("/v1/chat/completions", headers=auth(), json={"model": "small", "messages": [{"role": "user", "content": "hi"}]})
    await gateway.drain()
    assert (await client.get("/v1/reports/export.jsonl", headers=auth())).status_code == 403  # users can't export
    r = await client.get("/v1/reports/export.jsonl", headers=auth(LEAD_KEY))
    rows = [_json.loads(line) for line in r.text.splitlines()]
    assert rows and rows[-1]["user_id"] == "alice" and "savings_routing" in rows[-1]
    assert not any(k in rows[-1] for k in ("messages", "content", "prompt"))


async def test_refusal_fallback(client, gateway):
    # small (cheapest) refuses; auto retries on big; the refused attempt is still recorded (it cost tokens)
    body = {"model": "auto", "messages": [{"role": "user", "content": "explain this exploit [refuse:small-1]"}]}
    headers = {**auth(), "x-route-floor": "general:3"}
    r = await client.post("/v1/chat/completions", headers=headers, json=body)
    assert r.status_code == 200 and r.headers["x-meritorder-model"] == "big"
    assert r.headers["x-meritorder-refused-by"] == "small"
    assert r.json()["choices"][0]["message"]["content"]
    r = await client.post("/v1/chat/completions", headers=headers, json={**body, "stream": True})
    assert r.headers["x-meritorder-model"] == "big" and "content_filter" not in r.text
    await gateway.drain()
    refused_rows = [x for x in await ledger(gateway) if x.get("error", "") and "refused" in x["error"]]
    assert len(refused_rows) == 2 and all(x["served_model"] == "small" for x in refused_rows)
    # the prompt was billed either way: a stream's usage comes after its finish_reason, and is read too
    assert all(x["input_tokens"] > 0 and x["actual_cost"] > 0 for x in refused_rows)
    # a named model gets its own answer, refusal included
    r = await client.post("/v1/chat/completions", headers=auth(), json={"model": "small", "messages": body["messages"]})
    assert r.json()["choices"][0]["finish_reason"] == "content_filter"
    # teams can turn it off
    gateway.cfg.team("default").refusal_fallback = False
    r = await client.post("/v1/chat/completions", headers=headers, json=body)
    assert r.headers["x-meritorder-model"] == "small"
