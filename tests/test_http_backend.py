"""The real-provider path: HTTP backends against a fake upstream, SSE parsing from wire bytes, failover."""

import copy
import json
import uuid

import httpx
import pytest
from conftest import TEST_CONFIG, auth

from meritorder.app import build_gateway, create_app
from meritorder.backends.auth import BackendAuthProvider
from meritorder.backends.base import RETRYABLE_STATUS, BackendError
from meritorder.backends.http import AnthropicBackend, OpenAIBackend, _error_message
from meritorder.config import BackendConfig, build_config
from meritorder.formats.assemble import plain_content
from meritorder.formats.sse import SSEEvent, event, from_list, iter_sse
from meritorder.sessions import session_key
from meritorder.state import MemoryState
from meritorder.store import MemoryStore

OPENAI_OK = {"id": "c1", "object": "chat.completion", "created": 1, "model": "m-1",
             "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
             "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}}
ANTHROPIC_OK = {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-x",
                "content": [{"type": "text", "text": "hello"}], "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 2}}


class Upstream:
    """A fake provider: records requests and answers with whatever the test queues up."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        r = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(r, Exception):
            raise r
        if callable(r):
            return r(request)
        return r

    def body(self, i: int = -1) -> dict:
        return json.loads(self.requests[i].content)


def wire(cfg: BackendConfig, upstream: Upstream, auth_provider=None):
    cls = AnthropicBackend if cfg.format == "anthropic" else OpenAIBackend
    b = cls(cfg, auth_provider or BackendAuthProvider())
    b.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    return b


def chunks(*parts: bytes):
    async def gen():
        for p in parts:
            yield p
    return gen()


def stream_response(*parts: bytes, die: Exception | None = None) -> httpx.Response:
    async def gen():
        for p in parts:
            yield p
        if die is not None:
            raise die
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=gen())


async def collect(it) -> list[SSEEvent]:
    return [e async for e in it]


# ------------------------------------------------------------------ SSE parsing


async def test_sse_events_split_across_chunks_and_multibyte_characters():
    data = 'data: {"text": "héllo ✓"}\n\n'.encode()
    # Split inside the multi-byte characters and inside the blank-line terminator.
    parts = [data[:12], data[12:15], data[15:-1], data[-1:]]
    [ev] = await collect(iter_sse(chunks(*parts)))
    assert ev.json() == {"text": "héllo ✓"} and ev.event is None


async def test_sse_named_events_multiline_data_comments_crlf_and_done():
    raw = (b": keepalive\r\n\r\n"
           b"event: message_start\r\ndata: {\"a\": 1}\r\n\r\n"
           b"data: line one\ndata: line two\n\n"
           b"id: 7\nretry: 100\n\n"  # fields we don't use: no event
           b"event: ping\n\n"  # a named event with no data still counts
           b"data:no-space\n\n"
           b"data: [DONE]\n\n")
    evs = await collect(iter_sse(chunks(raw[:20], raw[20:21], raw[21:])))
    assert [(e.event, e.data) for e in evs] == [
        ("message_start", '{"a": 1}'), (None, "line one\nline two"), ("ping", ""), (None, "no-space"), (None, "[DONE]")]
    assert evs[0].json() == {"a": 1} and evs[1].json() is None and evs[-1].json() is None


async def test_sse_trailing_event_without_blank_line_and_crlf_split_between_chunks():
    evs = await collect(iter_sse(chunks(b"data: {\"x\": 1}\r", b"\n\r\n", b"data: tail")))
    assert [e.data for e in evs] == ['{"x": 1}', "tail"]
    assert await collect(iter_sse(chunks(b"  \n"))) == []


def test_sse_event_json_and_encode_round_trip():
    assert SSEEvent(data="[1, 2]").json() is None  # not an object
    assert SSEEvent(data="").json() is None
    ev = event({"type": "x", "text": "é"}, "content_block_delta")
    assert ev.encode() == 'event: content_block_delta\ndata: {"type":"x","text":"é"}\n\n'.encode()
    assert SSEEvent(data="a\nb").encode() == b"data: a\ndata: b\n\n"


async def test_sse_encode_then_parse_is_lossless():
    evs = [event({"n": i}, "e" if i % 2 else None) for i in range(5)] + [SSEEvent(data="multi\nline")]
    wire_bytes = b"".join(e.encode() for e in evs)
    back = await collect(iter_sse(chunks(*[wire_bytes[i:i + 7] for i in range(0, len(wire_bytes), 7)])))
    assert back == evs
    assert await collect(from_list(evs)) == evs


# ------------------------------------------------------------------ HTTP backends


def openai_cfg(**kw) -> BackendConfig:
    return BackendConfig(name="up", kind="api", format="openai", base_url="https://up.example/v1/", **kw)


def anthropic_cfg(**kw) -> BackendConfig:
    return BackendConfig(name="anth", kind="api", format="anthropic", base_url="https://anth.example", **kw)


async def test_openai_json_request_shape_and_response():
    up = Upstream(httpx.Response(200, json=OPENAI_OK, headers={"x-request-id": "r1"}))
    b = wire(openai_cfg(max_tokens_field="max_completion_tokens", extra_headers={"x-extra": "1"}), up)
    resp = await b.send({"model": "m-1", "messages": [], "max_tokens": 5, "stream": False, "stream_options": {"x": 1}}, False, {})
    assert resp.status == 200 and resp.body == OPENAI_OK and resp.headers["x-request-id"] == "r1" and resp.events is None
    req = up.requests[0]
    assert str(req.url) == "https://up.example/v1/chat/completions" and req.headers["x-extra"] == "1"
    assert req.headers["content-type"] == "application/json"
    sent = up.body()
    assert sent["max_completion_tokens"] == 5 and "max_tokens" not in sent and "stream" not in sent and "stream_options" not in sent
    await b.close()


async def test_openai_stream_asks_for_usage_and_parses_wire_bytes():
    sse = (b'data: {"choices":[{"delta":{"content":"he"}}]}\n\n'
           b": ping\n\n"
           b'data: {"choices":[{"delta":{"content":"llo"}}]}\r\n\r\n'
           b'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2}}\n\n'
           b"data: [DONE]\n\n")
    up = Upstream(stream_response(sse[:30], sse[30:31], sse[31:90], sse[90:]))
    b = wire(openai_cfg(), up)
    resp = await b.send({"model": "m-1", "messages": [], "stream_options": {"keep": True}}, True, {})
    assert up.body()["stream"] is True and up.body()["stream_options"] == {"keep": True, "include_usage": True}
    evs = await collect(resp.events)
    await resp.aclose()
    await resp.aclose()  # closing twice is harmless
    assert [e.data for e in evs][-1] == "[DONE]" and len(evs) == 4
    assert "".join((e.json() or {}).get("choices", [{}])[0].get("delta", {}).get("content", "") for e in evs[:2]) == "hello"


async def test_mistral_content_chunks_become_plain_text():
    """Mistral's API sends content as typed chunks; clients get OpenAI's string, the thinking as reasoning_content."""
    chunked = [{"type": "thinking", "thinking": [{"type": "text", "text": "Need OK."}], "closed": True}, {"type": "text", "text": "OK"}]
    reply = {**OPENAI_OK, "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "tool_calls": None, "content": chunked}}]}
    resp = await wire(openai_cfg(), Upstream(httpx.Response(200, json=reply))).send({"model": "m", "messages": []}, False, {})
    assert resp.body["choices"][0]["message"] == {"role": "assistant", "tool_calls": None, "content": "OK", "reasoning_content": "Need OK."}
    other = [{"type": "text", "text": "see"}, {"type": "reference", "reference_ids": [1]}]  # any other kind of chunk: left alone
    reply = {**OPENAI_OK, "choices": [{"index": 0, "message": {"role": "assistant", "content": other}}]}
    resp = await wire(openai_cfg(), Upstream(httpx.Response(200, json=reply))).send({"model": "m", "messages": []}, False, {})
    assert resp.body["choices"][0]["message"]["content"] == other
    tool = {"choices": [{"message": {"content": [{"type": "thinking", "thinking": "hm"}], "tool_calls": [{"id": "t"}]}}]}
    assert plain_content(tool)["choices"][0]["message"] == {"content": None, "tool_calls": [{"id": "t"}], "reasoning_content": "hm"}

    sse = (b'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}\n\n'
           b'data: {"choices":[{"index":0,"delta":{"content":[{"type":"thinking","thinking":[{"type":"text","text":"Need"}]}]}}],"p":"abc"}\n\n'
           b'data: {"choices":[{"index":0,"delta":{"content":[{"type":"text","text":"OK"}]}}]}\n\n'
           b"data: [DONE]\n\n")
    resp = await wire(openai_cfg(), Upstream(stream_response(sse))).send({"model": "m", "messages": []}, True, {})
    deltas = [e.json()["choices"][0]["delta"] for e in await collect(resp.events) if e.json()]
    assert deltas == [{"role": "assistant", "content": ""}, {"reasoning_content": "Need"}, {"content": "OK"}]


async def test_upstream_dying_mid_stream_raises_from_the_event_iterator():
    up = Upstream(stream_response(b'data: {"choices":[{"delta":{"content":"he"}}]}\n\n', die=httpx.ReadError("connection reset")))
    resp = await wire(openai_cfg(), up).send({"model": "m", "messages": []}, True, {})
    it = resp.events.__aiter__()
    assert (await it.__anext__()).json()["choices"][0]["delta"]["content"] == "he"
    with pytest.raises(httpx.ReadError):
        await it.__anext__()


@pytest.mark.parametrize("status", sorted(RETRYABLE_STATUS))
async def test_retryable_statuses(status):
    up = Upstream(httpx.Response(status, json={"error": {"message": "busy", "type": "x"}}))
    with pytest.raises(BackendError) as e:
        await wire(openai_cfg(), up).send({"model": "m", "messages": []}, False, {})
    assert e.value.status == status and e.value.retryable and e.value.message == "up: busy"
    assert e.value.body == {"error": {"message": "busy", "type": "x"}}


@pytest.mark.parametrize("status", [400, 401, 403, 404, 413, 422])
async def test_client_and_auth_errors_are_not_retryable(status):
    up = Upstream(httpx.Response(status, text="plain failure"))
    with pytest.raises(BackendError) as e:
        await wire(openai_cfg(), up).send({"model": "m", "messages": []}, True, {})  # errors are read even when streaming
    assert e.value.status == status and not e.value.retryable and e.value.body is None and "plain failure" in e.value.message


def test_error_message_shapes():
    assert _error_message({"error": {"message": "a"}}, b"") == "a"
    assert _error_message({"error": "b"}, b"") == "b"
    assert _error_message({"message": "c"}, b"") == "c"
    assert _error_message({"other": 1}, b"raw text") == "raw text"
    assert _error_message(None, b"") == "upstream error"
    assert len(_error_message(None, b"x" * 2000)) == 500


@pytest.mark.parametrize("exc, status", [
    (httpx.ReadTimeout("slow"), 504),
    (httpx.ConnectTimeout("slow connect"), 504),
    (httpx.ConnectError("refused"), 502),
    (httpx.RemoteProtocolError("bad"), 502),
])
async def test_transport_failures_are_retryable(exc, status):
    with pytest.raises(BackendError) as e:
        await wire(openai_cfg(), Upstream(exc)).send({"model": "m", "messages": []}, False, {})
    assert e.value.status == status and e.value.retryable


async def test_non_json_success_body_and_credential_errors_are_502():
    with pytest.raises(BackendError) as e:
        await wire(openai_cfg(), Upstream(httpx.Response(200, text="<html>gateway</html>"))).send({"model": "m"}, False, {})
    assert e.value.status == 502 and e.value.retryable and "non-JSON" in e.value.message

    class Broken(BackendAuthProvider):
        async def apply(self, method, url, headers, body):
            raise RuntimeError("token expired")

    up = Upstream(httpx.Response(200, json=OPENAI_OK))
    with pytest.raises(BackendError) as e:
        await wire(openai_cfg(), up, Broken()).send({"model": "m"}, False, {})
    assert e.value.status == 502 and "RuntimeError: token expired" in e.value.message and not up.requests


async def test_auth_provider_headers_reach_the_upstream():
    class Signs(BackendAuthProvider):
        async def apply(self, method, url, headers, body):
            assert method == "POST" and url.endswith("/chat/completions") and json.loads(body)["model"] == "m"
            return {**headers, "authorization": "Bearer signed"}

    up = Upstream(httpx.Response(200, json=OPENAI_OK))
    await wire(openai_cfg(), up, Signs()).send({"model": "m"}, False, {})
    assert up.requests[0].headers["authorization"] == "Bearer signed"


async def test_openai_embeddings():
    up = Upstream(httpx.Response(200, json={"data": [{"embedding": [0.1]}], "usage": {"prompt_tokens": 1}}))
    resp = await wire(openai_cfg(), up).embed({"model": "e", "input": "hi"})
    assert str(up.requests[0].url) == "https://up.example/v1/embeddings" and resp.body["data"][0]["embedding"] == [0.1]


async def test_anthropic_headers_version_and_beta_forwarding():
    up = Upstream(httpx.Response(200, json=ANTHROPIC_OK))
    b = wire(anthropic_cfg(), up)
    resp = await b.send({"model": "claude-x", "messages": [], "stream": False}, False,
                        {"anthropic-version": "2024-01-01", "anthropic-beta": "context-1m"})
    req = up.requests[0]
    assert str(req.url) == "https://anth.example/v1/messages" and resp.body["content"][0]["text"] == "hello"
    assert req.headers["anthropic-version"] == "2024-01-01" and req.headers["anthropic-beta"] == "context-1m"
    assert "stream" not in up.body() and up.body()["model"] == "claude-x"

    up2 = Upstream(httpx.Response(200, json=ANTHROPIC_OK))
    await wire(anthropic_cfg(anthropic_version="2023-06-01", forward_anthropic_beta=False), up2).send(
        {"model": "c"}, False, {"anthropic-version": "2099-01-01", "anthropic-beta": "x"})
    assert up2.requests[0].headers["anthropic-version"] == "2023-06-01" and "anthropic-beta" not in up2.requests[0].headers

    up3 = Upstream(httpx.Response(200, json=ANTHROPIC_OK))
    await wire(anthropic_cfg(), up3).send({"model": "c"}, False, {})
    assert up3.requests[0].headers["anthropic-version"] == "2023-06-01"  # default


async def test_anthropic_stream_named_events():
    sse = (b"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":5}}}\n\n"
           b"event: ping\ndata: {\"type\": \"ping\"}\n\n"
           b"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n"
           b"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
    up = Upstream(stream_response(*[sse[i:i + 13] for i in range(0, len(sse), 13)]))
    resp = await wire(anthropic_cfg(), up).send({"model": "c", "messages": []}, True, {})
    assert up.body()["stream"] is True
    evs = await collect(resp.events)
    assert [e.event for e in evs] == ["message_start", "ping", "content_block_delta", "message_stop"]
    assert evs[2].json()["delta"]["text"] == "hi"


async def test_anthropic_vertex_urls_and_body():
    up = Upstream(httpx.Response(200, json=ANTHROPIC_OK))
    cfg = BackendConfig(name="vx", kind="api", format="anthropic", flavor="vertex", project="proj", region="us-east5")
    b = wire(cfg, up)
    await b.send({"model": "claude-x@1", "messages": []}, False, {"anthropic-beta": "b1"})
    req = up.requests[0]
    assert str(req.url) == ("https://us-east5-aiplatform.googleapis.com/v1/projects/proj/locations/us-east5"
                            "/publishers/anthropic/models/claude-x@1:rawPredict")
    sent = up.body()
    assert "model" not in sent and sent["anthropic_version"] == "vertex-2023-10-16"
    assert "anthropic-version" not in req.headers and req.headers["anthropic-beta"] == "b1"
    assert await b.count_tokens({"model": "c"}, {}) is None  # not offered on Vertex

    up2 = Upstream(stream_response(b"event: message_stop\ndata: {}\n\n"))
    g = wire(BackendConfig(name="vx", kind="api", format="anthropic", flavor="vertex", project="p"), up2)
    await g.send({"model": "c"}, True, {})
    assert str(up2.requests[0].url) == ("https://aiplatform.googleapis.com/v1/projects/p/locations/global"
                                        "/publishers/anthropic/models/c:streamRawPredict")


async def test_anthropic_count_tokens_keeps_only_countable_fields():
    up = Upstream(httpx.Response(200, json={"input_tokens": 42}))
    out = await wire(anthropic_cfg(), up).count_tokens(
        {"model": "c", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 9, "stream": True, "system": "s"}, {})
    assert out == {"input_tokens": 42} and str(up.requests[0].url) == "https://anth.example/v1/messages/count_tokens"
    assert set(up.body()) == {"model", "messages", "system"}


# ------------------------------------------------------------------ through the gateway


def http_gateway(monkeypatch, primary: Upstream, secondary: Upstream, primary_format: str = "openai"):
    """A named model 'm' on two HTTP backends: 'up-a' (cheaper, tried first) and 'up-b'."""
    monkeypatch.setenv("MO_TEST_UP_A", "key-a")
    monkeypatch.setenv("MO_TEST_UP_B", "key-b")
    raw = copy.deepcopy(TEST_CONFIG)
    raw["baseline_model"] = "m"
    raw["backends"] = {
        "up-a": {"kind": "api", "format": primary_format, "base_url": "https://a.example" + ("/v1" if primary_format == "openai" else ""),
                 "auth": {"type": "api_key", "value_env": "MO_TEST_UP_A", **({"header": "x-api-key"} if primary_format == "anthropic" else {})}},
        "up-b": {"kind": "api", "format": "openai", "base_url": "https://b.example/v1",
                 "auth": {"type": "api_key", "value_env": "MO_TEST_UP_B"}},
    }
    raw["models"] = {"m": {"quality": {"coding": 4, "general": 4}, "routes": [
        {"backend": "up-a", "model": "m-a", "price": {"input": 1, "output": 2}},
        {"backend": "up-b", "model": "m-b", "price": {"input": 2, "output": 4}},
    ]}}
    cfg = build_config(raw, quiet=True)
    gw = build_gateway(cfg, store=MemoryStore(), state=MemoryState())
    for name, up in (("up-a", primary), ("up-b", secondary)):
        gw.backends[name].client = httpx.AsyncClient(transport=httpx.MockTransport(up))
    return gw


async def post(gw, body, path="/v1/chat/completions", **headers):
    app = create_app(gateway=gw)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        return await c.post(path, json=body, headers={**auth(), "x-session-id": uuid.uuid4().hex, **headers})


CHAT = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}


async def test_half_an_emoji_in_a_request_is_sent_and_its_session_kept(monkeypatch):
    """A lone surrogate (half an emoji, from a tool's output cut mid-character) is valid JSON but not UTF-8. The request
    still goes to the provider, escaped; its session's routing state is saved; and the response cache can key it."""
    a = Upstream(httpx.Response(200, json=OPENAI_OK))
    gw = http_gateway(monkeypatch, a, Upstream(httpx.Response(200, json=OPENAI_OK)))
    raw = b'{"model": "m", "messages": [{"role": "user", "content": "half an emoji \\ud83d here"}]}'
    app = create_app(gateway=gw)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test") as c:
        headers = {**auth(), "content-type": "application/json", "x-session-id": "s-half"}
        r = await c.post("/v1/chat/completions", content=raw, headers=headers)
        assert r.status_code == 200 and b"half an emoji \\ud83d here" in a.requests[-1].content
        assert (await gw.state.get_session(session_key("alice", "s-half")))["turns"] == 1
        r = await c.post("/v1/chat/completions", content=raw, headers={**headers, "x-meritorder-cache": "on"})
        assert r.status_code == 200


@pytest.mark.parametrize("failure", [httpx.Response(503, json={"error": {"message": "overloaded"}}),
                                     httpx.Response(401, json={"error": {"message": "bad key"}}),
                                     httpx.ConnectError("refused")])
async def test_gateway_fails_over_to_the_next_backend(monkeypatch, failure):
    a, b = Upstream(failure), Upstream(httpx.Response(200, json={**OPENAI_OK, "model": "m-b"}))
    gw = http_gateway(monkeypatch, a, b)
    r = await post(gw, CHAT)
    assert r.status_code == 200 and r.headers["x-meritorder-backend"] == "up-b" and r.json()["choices"][0]["message"]["content"] == "hello"
    assert len(a.requests) == 1 and len(b.requests) == 1
    assert a.requests[0].headers["authorization"] == "Bearer key-a" and b.requests[0].headers["authorization"] == "Bearer key-b"
    assert json.loads(a.requests[0].content)["model"] == "m-a" and json.loads(b.requests[0].content)["model"] == "m-b"
    await gw.drain()
    row = (await gw.store.recent(1))[0]
    assert row["backend"] == "up-b" and row["status"] == 200 and row["served_model"] == "m" and row["output_tokens"] == 2


async def test_circuit_breaker_opens_after_three_failures(monkeypatch):
    a, b = Upstream(httpx.Response(500, text="boom")), Upstream(httpx.Response(200, json=OPENAI_OK))
    gw = http_gateway(monkeypatch, a, b)
    for _ in range(3):
        assert (await post(gw, CHAT)).headers["x-meritorder-backend"] == "up-b"
    now = __import__("time").time()
    # A 500 counts against the model on that backend, not the whole backend (other models there keep working).
    assert len(a.requests) == 3 and not gw.backends["up-a"].healthy(now, "m-a") and gw.backends["up-a"].healthy(now)
    r = await post(gw, CHAT)
    assert r.status_code == 200 and len(a.requests) == 3 and len(b.requests) == 4  # up-a skipped while open
    assert gw.backends["up-a"].record_success("m-a") is True  # it was being skipped: recovered
    a.responses = [httpx.Response(200, json=OPENAI_OK)]
    assert (await post(gw, CHAT)).headers["x-meritorder-backend"] == "up-a"


async def test_client_error_is_returned_without_failover(monkeypatch):
    a = Upstream(httpx.Response(400, json={"error": {"message": "messages: field required", "type": "invalid_request_error"}}))
    b = Upstream(httpx.Response(200, json=OPENAI_OK))
    gw = http_gateway(monkeypatch, a, b)
    r = await post(gw, CHAT)
    assert r.status_code == 400 and "field required" in r.json()["error"]["message"] and not b.requests
    await gw.drain()
    row = (await gw.store.recent(1))[0]
    assert row["status"] == 400 and "field required" in row["error"]


async def test_every_backend_failing_returns_the_last_error(monkeypatch):
    gw = http_gateway(monkeypatch, Upstream(httpx.Response(503, text="a down")), Upstream(httpx.ReadTimeout("slow")))
    r = await post(gw, CHAT)
    assert r.status_code == 504 and "timed out" in r.json()["error"]["message"]
    await gw.drain()
    assert (await gw.store.recent(1))[0]["status"] == 504


async def test_gateway_streams_real_wire_bytes_and_records_usage(monkeypatch):
    sse = (b'data: {"id":"c","object":"chat.completion.chunk","model":"m-a","choices":[{"index":0,"delta":{"role":"assistant","content":"he"}}]}\n\n'
           b": keepalive\n\n"
           b'data: {"id":"c","object":"chat.completion.chunk","model":"m-a","choices":[{"index":0,"delta":{"content":"llo"},"finish_reason":"stop"}]}\r\n\r\n'
           b'data: {"id":"c","object":"chat.completion.chunk","model":"m-a","choices":[],"usage":{"prompt_tokens":7,"completion_tokens":2}}\n\n'
           b"data: [DONE]\n\n")
    a = Upstream(lambda req: stream_response(*[sse[i:i + 11] for i in range(0, len(sse), 11)]))
    gw = http_gateway(monkeypatch, a, Upstream(httpx.Response(500)))
    r = await post(gw, {**CHAT, "stream": True, "stream_options": {"include_usage": True}})
    assert r.status_code == 200 and r.headers["x-meritorder-backend"] == "up-a"
    evs = await collect(iter_sse(chunks(r.content)))
    text = "".join(c["delta"].get("content", "") for e in evs if e.json() for c in e.json().get("choices", []))
    assert text == "hello" and evs[-1].data == "[DONE]"
    await gw.drain()
    row = (await gw.store.recent(1))[0]
    assert row["input_tokens"] == 7 and row["output_tokens"] == 2 and row["stream"] is True and row["status"] == 200
    assert gw.inflight.count("up-a") == 0


async def test_anthropic_client_streaming_from_openai_backend(monkeypatch):
    sse = (b'data: {"id":"c","choices":[{"index":0,"delta":{"role":"assistant","content":"hi"}}]}\n\n'
           b'data: {"id":"c","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
           b'data: {"id":"c","choices":[],"usage":{"prompt_tokens":3,"completion_tokens":1}}\n\n'
           b"data: [DONE]\n\n")
    gw = http_gateway(monkeypatch, Upstream(lambda req: stream_response(sse[:50], sse[50:])), Upstream(httpx.Response(500)))
    r = await post(gw, {"model": "m", "max_tokens": 50, "stream": True, "messages": [{"role": "user", "content": "hi"}]}, path="/v1/messages")
    assert r.status_code == 200
    evs = await collect(iter_sse(chunks(r.content)))
    kinds = [e.event for e in evs]
    assert kinds[0] == "message_start" and kinds[-1] == "message_stop" and "content_block_delta" in kinds
    assert any((e.json() or {}).get("delta", {}).get("text") == "hi" for e in evs)


async def test_upstream_dying_mid_stream_sends_an_error_event_and_logs_502(monkeypatch):
    a = Upstream(lambda req: stream_response(b'data: {"id":"c","choices":[{"index":0,"delta":{"content":"par"}}]}\n\n',
                                             die=httpx.ReadError("connection reset")))
    gw = http_gateway(monkeypatch, a, Upstream(httpx.Response(500)))
    r = await post(gw, {**CHAT, "stream": True})
    assert r.status_code == 200  # headers were already sent
    evs = await collect(iter_sse(chunks(r.content)))
    assert "par" in evs[0].data and "upstream stream failed" in evs[-1].data
    await gw.drain()
    row = (await gw.store.recent(1))[0]
    assert row["status"] == 502 and "ReadError" in row["error"]
    assert gw.inflight.count("up-a") == 0


async def test_anthropic_backend_through_the_gateway_for_an_openai_client(monkeypatch):
    a = Upstream(httpx.Response(200, json={**ANTHROPIC_OK, "model": "m-a"}))
    gw = http_gateway(monkeypatch, a, Upstream(httpx.Response(500)), primary_format="anthropic")
    r = await post(gw, CHAT)
    assert r.status_code == 200 and r.json()["choices"][0]["message"]["content"] == "hello"
    req = a.requests[0]
    assert str(req.url) == "https://a.example/v1/messages" and req.headers["x-api-key"] == "key-a"
    assert "mo-test-key" not in str(req.headers.raw)  # the client's gateway key never reaches a provider
