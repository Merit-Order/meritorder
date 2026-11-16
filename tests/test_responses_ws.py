"""The Responses API's WebSocket transport (decision 54): the same pipeline as POST /v1/responses, chaining on the
connection without stored mode, stored mode, generate: false, streams (stream_id), interrupts, errors, and the official
SDK's client (client.responses.connect) against a real server on localhost. On the mock backends."""

from __future__ import annotations

import asyncio
import json

import openai
import pytest
import uvicorn
from conftest import TEST_KEY, ledger
from starlette.testclient import TestClient

from meritorder.app import build_gateway, create_app
from meritorder.config import StoredResponsesConfig
from meritorder.responses_ws import MAX_LANES
from meritorder.state import MemoryState
from meritorder.store import MemoryStore

TERMINAL = ("response.completed", "response.incomplete", "response.failed", "error")


@pytest.fixture(autouse=True)
def responses_key(monkeypatch):
    monkeypatch.setenv("MERITORDER_RESPONSES_KEY", "test-responses-key")


class Socket:
    """A WebSocket client driving the ASGI app directly, in the test's own event loop (so Postgres and Redis work)."""

    def __init__(self, app, headers: dict[str, str]):
        self.app, self.inbox, self.outbox = app, asyncio.Queue(), asyncio.Queue()
        self.scope = {"type": "websocket", "asgi": {"version": "3.0"}, "path": "/v1/responses", "raw_path": b"/v1/responses",
                      "root_path": "", "query_string": b"", "scheme": "ws", "server": ("test", 80), "client": ("127.0.0.1", 1),
                      "subprotocols": [], "extensions": {"websocket.http.response": {}},
                      "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()]}
        self.accepted: dict | None = None
        self.denied: dict | None = None

    async def open(self) -> "Socket":
        self.task = asyncio.create_task(self.app(self.scope, self.inbox.get, self.outbox.put))
        await self.inbox.put({"type": "websocket.connect"})
        first = await asyncio.wait_for(self.outbox.get(), 5)
        if first["type"] == "websocket.accept":
            self.accepted = first
        elif first["type"] == "websocket.http.response.start":
            body = await asyncio.wait_for(self.outbox.get(), 5)
            self.denied = {"status": first["status"], "body": json.loads(body["body"])}
        else:
            self.denied = first
        return self

    async def send(self, value) -> None:
        await self.inbox.put({"type": "websocket.receive", "text": value if isinstance(value, str) else json.dumps(value)})

    async def recv(self) -> dict:
        msg = await asyncio.wait_for(self.outbox.get(), 5)
        assert msg["type"] == "websocket.send", msg
        return json.loads(msg["text"])

    async def until_done(self) -> list[dict]:
        events = []
        while not events or events[-1]["type"] not in TERMINAL:
            events.append(await self.recv())
        return events

    async def turn(self, **body) -> list[dict]:
        await self.send({"type": "response.create", **body})
        return await self.until_done()

    async def close(self) -> None:
        await self.inbox.put({"type": "websocket.disconnect", "code": 1000})
        await asyncio.wait_for(self.task, 5)


def connect(gateway, key: str = TEST_KEY, **headers) -> Socket:
    return Socket(create_app(gateway=gateway), {"authorization": f"Bearer {key}", **headers})


def spy(gateway, backend: str = "mock-openai") -> list[dict]:
    sent: list[dict] = []
    b = gateway.backends[backend]
    original = b.send

    async def send(body, stream, headers, **kw):
        sent.append(body)
        return await original(body, stream, headers, **kw)

    b.send = send
    return sent


def said(body: dict) -> list[str]:
    return [m["content"] for m in body["messages"] if m["role"] == "user"]


async def test_turns_chain_on_the_connection_without_stored_mode(gateway):
    sent = spy(gateway)
    ws = await connect(gateway, **{"session-id": "codex-1"}).open()
    assert ws.accepted is not None and dict(ws.accepted["headers"])[b"x-codex-websocket-max-message-bytes"] == str(32 * 1024 * 1024).encode()
    first = await ws.turn(model="small", input="first question", store=False, stream=True)
    kinds = [e["type"] for e in first]
    assert kinds[0] == "response.created" and kinds[-1] == "response.completed" and "response.output_text.delta" in kinds
    assert [e["sequence_number"] for e in first] == list(range(1, len(first) + 1))
    rid = first[-1]["response"]["id"]
    assert first[-1]["response"]["store"] is False
    second = await ws.turn(model="small", previous_response_id=rid, input=[{"role": "user", "content": "second question"}])
    assert second[-1]["type"] == "response.completed"
    assert second[-1]["response"]["previous_response_id"] == rid  # what the client chained from
    # The backend got the whole conversation: the first question, its answer, then the new one.
    assert said(sent[1]) == ["first question", "second question"]
    assert any(m["role"] == "assistant" for m in sent[1]["messages"])
    # Only the latest response of a stream can be continued without stored mode.
    stale = await ws.turn(model="small", previous_response_id=rid, input="again")
    assert stale == [{"type": "error", "status": 400, "error": {"type": "invalid_request_error", "code": "previous_response_not_found",
                                                               "message": stale[0]["error"]["message"], "param": "previous_response_id"}}]
    await ws.close()
    rows = await ledger(gateway)
    assert len(rows) == 2 and all(r["status"] == 200 for r in rows) and {r["session_id"] for r in rows} == {"codex-1"}
    counted = gateway.metrics.counters["meritorder_responses_websocket_total"]
    assert counted[(("event", "local_chain"),)] == 1 and counted[(("event", "response"),)] == 1


async def test_another_connection_has_no_memory_of_it(gateway):
    a = await connect(gateway).open()
    rid = (await a.turn(model="small", input="hi", store=False))[-1]["response"]["id"]
    await a.close()
    b = await connect(gateway).open()
    out = await b.turn(model="small", input="next", previous_response_id=rid)
    assert out[0]["error"]["code"] == "previous_response_not_found"
    await b.close()


async def test_generate_false_prepares_without_a_model_call(gateway):
    sent = spy(gateway)
    ws = await connect(gateway).open()
    warm = await ws.turn(model="small", input="the whole history", instructions="be brief", generate=False, stream_id="main")
    assert [e["type"] for e in warm] == ["response.created", "response.completed"]
    assert all(e["stream_id"] == "main" for e in warm) and warm[-1]["response"]["output"] == [] and not sent
    rid = warm[-1]["response"]["id"]
    # A warm-up can itself chain (Codex warms the next turn too).
    again = await ws.turn(model="small", input="more history", previous_response_id=rid, generate=False, stream_id="main")
    assert again[-1]["response"]["previous_response_id"] == rid
    out = await ws.turn(model="small", input="now answer", previous_response_id=again[-1]["response"]["id"], stream_id="main")
    assert out[-1]["type"] == "response.completed" and all(e["stream_id"] == "main" for e in out)
    assert said(sent[0]) == ["the whole history", "more history", "now answer"]
    await ws.close()
    assert len(await ledger(gateway)) == 1  # warm-ups cost nothing and leave no row


async def test_streams_and_their_limits(gateway):
    ws = await connect(gateway).open()
    await ws.send({"type": "response.create", "model": "small", "input": "a", "stream_id": "one"})
    await ws.send({"type": "response.create", "model": "small", "input": "b", "stream_id": "two"})
    done: dict[str, str] = {}
    while len(done) < 2:
        e = await ws.recv()
        if e["type"] == "response.completed":
            done[e["stream_id"]] = e["response"]["id"]
    # A new stream can fork from another stream's response.
    fork = await ws.turn(model="small", input="fork", previous_response_id=done["one"], stream_id="three")
    assert fork[-1]["type"] == "response.completed"
    bad = await ws.turn(model="small", input="x", stream_id="no spaces")
    assert bad[0]["error"]["code"] == "invalid_stream_id"
    for i in range(MAX_LANES - 3):
        await ws.turn(model="small", input="x", stream_id=f"lane{i}", generate=False)
    over = await ws.turn(model="small", input="x", stream_id="one-too-many")
    assert over[0]["error"]["code"] == "websocket_stream_limit_reached" and over[0]["stream_id"] == "one-too-many"
    await ws.close()


async def test_stored_mode_chains_across_connections(gateway):
    gateway.cfg.team("default").stored_responses = StoredResponsesConfig(enabled=True, retention_days=30)
    sent = spy(gateway)
    a = await connect(gateway).open()
    first = (await a.turn(model="small", input="remember this"))[-1]["response"]
    assert first["store"] is True
    await a.close()
    b = await connect(gateway).open()
    second = await b.turn(model="small", input="what was it?", previous_response_id=first["id"])
    assert second[-1]["type"] == "response.completed" and said(sent[1]) == ["remember this", "what was it?"]
    missing = await b.turn(model="small", input="x", previous_response_id="resp_nope")
    assert missing[0]["status"] == 404 and missing[0]["error"]["code"] == "previous_response_not_found"
    # store: false in stored mode chains on the connection, as without it.
    local = (await b.turn(model="small", input="not kept", store=False))[-1]["response"]
    assert (await b.turn(model="small", input="next", previous_response_id=local["id"], store=False))[-1]["type"] == "response.completed"
    await b.close()


async def test_refusals_and_errors(gateway):
    ws = await connect(gateway, key="mo-wrong").open()
    assert ws.denied == {"status": 401, "body": {"error": {"message": "invalid API key", "type": "authentication_error", "param": None,
                                                           "code": 401}}}
    ws = await connect(gateway).open()
    await ws.send("not json")
    assert (await ws.recv())["error"]["code"] == "invalid_json"
    await ws.send({"type": "response.steer", "previous_response_id": "resp_1", "input": []})
    assert (await ws.recv())["error"]["code"] == "unsupported"
    await ws.send({"type": "session.update", "stream_id": "main"})
    unknown = await ws.recv()
    assert unknown["error"]["code"] == "unknown_event" and unknown["stream_id"] == "main"
    assert (await ws.turn(model="small", input="x", background=True))[0]["error"]["param"] == "background"
    assert (await ws.turn(model="small", input=["not an item"]))[0]["status"] == 400
    assert (await ws.turn(model="small", input="x", conversation="conv_1"))[0]["status"] == 400  # stored mode is off
    assert (await ws.turn(model="nope", input="x"))[0]["status"] in (400, 404)
    assert (await ws.turn(model="small", input="x", generate=False, conversation="conv_1"))[0]["status"] == 400
    assert (await ws.turn(model="small", input=[{"type": "compaction_trigger"}], generate=False))[0]["status"] == 400
    await ws.close()


async def test_a_revoked_key_stops_at_the_next_turn(gateway, monkeypatch):
    from meritorder.auth import AuthError

    ws = await connect(gateway).open()
    assert (await ws.turn(model="small", input="hi"))[-1]["type"] == "response.completed"

    async def refuse(headers):
        raise AuthError(401, "API key has been revoked")

    monkeypatch.setattr(gateway.authn, "authenticate", refuse)
    out = await ws.turn(model="small", input="again")
    assert out[0]["status"] == 401 and out[0]["error"]["type"] == "authentication_error"
    await ws.close()


async def test_interrupt_stops_the_running_response(gateway):
    gateway.backends["mock-openai"].cfg.mock_latency_ms = 0
    original = gateway.backends["mock-openai"].send
    release = asyncio.Event()

    async def slow(body, stream, headers, **kw):
        resp = await original(body, stream, headers, **kw)
        events = resp.events

        async def held():
            async for ev in events:
                yield ev
                if any((c.get("delta") or {}).get("content") for c in (ev.json() or {}).get("choices") or []):
                    await release.wait()  # after the first text, so the stream has started

        resp.events = held()
        return resp

    gateway.backends["mock-openai"].send = slow
    ws = await connect(gateway).open()
    await ws.send({"type": "response.create", "model": "small", "input": "a long answer"})
    created = await ws.recv()
    assert created["type"] == "response.created"
    await ws.send({"type": "response.interrupt", "response_id": created["response"]["id"], "mode": "discard_partial_items"})
    while (e := await ws.recv())["type"] != "response.interrupt.accepted":
        pass
    assert e["response_id"] == created["response"]["id"]
    await ws.send({"type": "response.interrupt", "response_id": created["response"]["id"]})
    assert (await ws.recv())["type"] == "response.interrupt.failed"
    release.set()
    await ws.close()
    rows = await ledger(gateway)
    assert rows[-1]["status"] == 499


async def test_disconnect_mid_response_is_recorded(gateway):
    original = gateway.backends["mock-openai"].send
    hold = asyncio.Event()

    async def slow(body, stream, headers, **kw):
        resp = await original(body, stream, headers, **kw)
        events = resp.events

        async def held():
            async for ev in events:
                yield ev
                if any((c.get("delta") or {}).get("content") for c in (ev.json() or {}).get("choices") or []):
                    await hold.wait()  # after the first text, so the stream has started

        resp.events = held()
        return resp

    gateway.backends["mock-openai"].send = slow
    ws = await connect(gateway).open()
    await ws.send({"type": "response.create", "model": "small", "input": "hi"})
    assert (await ws.recv())["type"] == "response.created"
    await ws.close()
    assert (await ledger(gateway))[-1]["status"] == 499


async def test_edges(gateway, monkeypatch):
    # A server without the denial extension: the handshake is closed as a policy violation.
    ws = connect(gateway, key="mo-wrong")
    ws.scope["extensions"] = {}
    await ws.open()
    assert ws.denied == {"type": "websocket.close", "code": 1008, "reason": "invalid API key"}

    # A failed continuation drops what it chained from: the client sends everything next.
    ws = await connect(gateway).open()
    rid = (await ws.turn(model="small", input="hi", store=False))[-1]["response"]["id"]
    assert (await ws.turn(model="nope", input="x", previous_response_id=rid))[0]["type"] == "error"
    assert (await ws.turn(model="small", input="x", previous_response_id=rid))[0]["error"]["code"] == "previous_response_not_found"

    # A bug in one turn is an error event; the connection goes on.
    async def broken(*a, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(gateway, "respond", broken)
    out = await ws.turn(model="small", input="x", stream_id="s")
    assert out == [{"type": "error", "status": 500, "stream_id": "s", "error": {"type": "api_error", "code": "server_error",
                                                                                 "message": "the gateway failed while making this response",
                                                                                 "param": None}}]
    monkeypatch.undo()
    assert (await ws.turn(model="small", input="x", stream_id="s"))[-1]["type"] == "response.completed"
    await ws.close()


async def test_generate_false_in_stored_mode(gateway):
    gateway.cfg.team("default").stored_responses = StoredResponsesConfig(enabled=True, retention_days=30)
    sent = spy(gateway)
    ws = await connect(gateway).open()
    first = (await ws.turn(model="small", input="kept turn"))[-1]["response"]
    warm = await ws.turn(model="small", input="warm", previous_response_id=first["id"], generate=False)
    assert warm[-1]["type"] == "response.completed"
    await ws.turn(model="small", input="go", previous_response_id=warm[-1]["response"]["id"])
    assert said(sent[1]) == ["kept turn", "warm", "go"]
    missing = await ws.turn(model="small", input="x", previous_response_id="resp_gone", generate=False)
    assert missing[0]["error"]["code"] == "previous_response_not_found"
    await ws.close()


def test_starlette_test_client(cfg):
    """FastAPI's own test client, on its own event loop (memory storage)."""
    gw = build_gateway(cfg, store=MemoryStore(), state=MemoryState())
    with TestClient(create_app(gateway=gw)) as client:
        with client.websocket_connect("/v1/responses", headers={"authorization": f"Bearer {TEST_KEY}"}) as ws:
            ws.send_json({"type": "response.create", "model": "small", "input": "hello"})
            events = []
            while not events or events[-1]["type"] not in TERMINAL:
                events.append(ws.receive_json())
            rid = events[-1]["response"]["id"]
            ws.send_json({"type": "response.create", "model": "small", "input": "and then?", "previous_response_id": rid})
            events = []
            while not events or events[-1]["type"] not in TERMINAL:
                events.append(ws.receive_json())
            assert events[-1]["type"] == "response.completed"


async def test_official_sdk_client(gateway):
    """openai-python's client.responses.connect(), over a real WebSocket to a server on localhost."""
    server = uvicorn.Server(uvicorn.Config(create_app(gateway=gateway), host="127.0.0.1", port=0, log_level="warning", lifespan="off"))
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        client = openai.AsyncOpenAI(api_key=TEST_KEY, base_url=f"http://127.0.0.1:{port}/v1", max_retries=0)
        async with client.responses.connect() as conn:
            await conn.response.create(model="small", input="hello there", store=False)
            async for event in conn:
                if event.type in TERMINAL:
                    break
            assert event.type == "response.completed" and event.response.output_text
            await conn.response.create(model="small", input="go on", previous_response_id=event.response.id, store=False)
            async for event in conn:
                if event.type in TERMINAL:
                    break
            assert event.type == "response.completed" and event.response.previous_response_id
        bad = openai.AsyncOpenAI(api_key="mo-wrong", base_url=f"http://127.0.0.1:{port}/v1", max_retries=0)
        with pytest.raises(Exception):
            async with bad.responses.connect() as conn:
                await conn.recv()
    finally:
        server.should_exit = True
        await task
    rows = await ledger(gateway)
    assert len(rows) == 2 and rows[1]["input_tokens"] > rows[0]["input_tokens"]

