"""Switching to another source when a provider fails: errors that arrive after a 200, streams that never
start, Retry-After, busy models versus broken backends, last resorts, alerts, and what people see."""

import asyncio
import json
import time

import httpx
from conftest import ADMIN_KEY
from test_http_backend import CHAT, OPENAI_OK, Upstream, chunks, collect, http_gateway, post, stream_response

from meritorder.backends.base import Backend
from meritorder.backends.http import retry_after
from meritorder.config import BackendConfig, ModelConfig, Route
from meritorder.failover import error_in
from meritorder.formats.sse import iter_sse

OK_STREAM = (b'data: {"id":"c","choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}\n\n'
             b'data: {"id":"c","choices":[{"index":0,"delta":{"content":"from b"}}]}\n\n'
             b'data: {"id":"c","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2}}\n\n'
             b"data: [DONE]\n\n")
OVERLOADED = (b'event: message_start\ndata: {"type":"message_start","message":{"id":"m","type":"message","role":"assistant",'
              b'"content":[],"model":"m-a","usage":{"input_tokens":3,"output_tokens":0}}}\n\n'
              b'event: error\ndata: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}\n\n')


async def stream_text(r) -> str:
    out = ""
    for e in await collect(iter_sse(chunks(r.content))):
        d = e.json() or {}
        for c in d.get("choices") or []:
            out += (c.get("delta") or {}).get("content") or ""
    return out


def second_model(gw, name="y", route_model="y-a"):
    gw.cfg.models.append(ModelConfig(name=name, quality={"coding": 4, "general": 4},
                                     routes=[Route(backend="up-a", model=route_model, price={"input": 1, "output": 2})]))


async def test_anthropic_overload_inside_a_200_stream_goes_to_the_next_source(monkeypatch):
    a = Upstream(lambda req: stream_response(OVERLOADED))
    b = Upstream(lambda req: stream_response(OK_STREAM))
    gw = http_gateway(monkeypatch, a, b, primary_format="anthropic")
    r = await post(gw, {**CHAT, "stream": True})
    assert r.status_code == 200 and r.headers["x-meritorder-backend"] == "up-b"
    assert r.headers["x-meritorder-failed-over"] == "up-a/m-a 529"
    assert await stream_text(r) == "from b"
    await gw.drain()
    assert (await gw.store.recent(1))[0]["failed_over"] == "up-a/m-a 529"  # the ledger keeps it, for the Requests view
    now = time.time()
    assert not gw.backends["up-a"].healthy(now, "m-a") and gw.backends["up-a"].healthy(now)  # only that model is skipped
    await gw.drain()
    alerts = await gw.state.recent_alerts()
    assert alerts and "up-a (m-a) is failing: overloaded" in alerts[0]["text"] and "Requests go to up-b meanwhile" in alerts[0]["text"]
    assert gw.inflight.count("up-a") == 0


async def test_openrouter_style_error_chunk_and_error_body_fail_over(monkeypatch):
    chunk = b'data: {"id":"c","error":{"code":502,"message":"Provider returned error"},"choices":[{"index":0,"delta":{},"finish_reason":"error"}]}\n\n'
    a = Upstream(lambda req: stream_response(b'data: {"id":"c","choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}\n\n', chunk))
    gw = http_gateway(monkeypatch, a, Upstream(lambda req: stream_response(OK_STREAM)))
    r = await post(gw, {**CHAT, "stream": True})
    assert r.headers["x-meritorder-backend"] == "up-b" and await stream_text(r) == "from b"

    a2 = Upstream(httpx.Response(200, json={"error": {"message": "upstream timed out", "code": 504}}))
    gw = http_gateway(monkeypatch, a2, Upstream(httpx.Response(200, json=OPENAI_OK)))
    r = await post(gw, CHAT)
    assert r.status_code == 200 and r.headers["x-meritorder-backend"] == "up-b" and r.headers["x-meritorder-failed-over"] == "up-a/m-a 504"


async def test_a_failed_responses_api_stream_fails_over_as_busy(monkeypatch):
    failed = (b'event: response.created\ndata: {"type":"response.created","response":{"id":"r","model":"m-a","created_at":1}}\n\n'
              b'event: response.failed\ndata: {"type":"response.failed","response":{"status":"failed",'
              b'"error":{"code":"rate_limit_exceeded","message":"Rate limit reached"}}}\n\n')
    gw = http_gateway(monkeypatch, Upstream(lambda req: stream_response(failed)), Upstream(lambda req: stream_response(OK_STREAM)))
    gw.backends["up-a"].cfg = gw.backends["up-a"].cfg.model_copy(update={"api": "responses"})
    r = await post(gw, {**CHAT, "stream": True})
    assert r.headers["x-meritorder-backend"] == "up-b" and await stream_text(r) == "from b"
    assert r.headers["x-meritorder-failed-over"] == "up-a/m-a 429"
    assert gw.backends["up-a"].skipped(time.time(), "m-a").capacity


async def test_a_stream_that_never_starts_is_abandoned(monkeypatch):
    async def hang():
        await asyncio.sleep(30)
        yield b""

    a = Upstream(lambda req: httpx.Response(200, headers={"content-type": "text/event-stream"}, content=hang()))
    gw = http_gateway(monkeypatch, a, Upstream(lambda req: stream_response(OK_STREAM)))
    gw.backends["up-a"].cfg = gw.backends["up-a"].cfg.model_copy(update={"first_byte_timeout_seconds": 0.2})
    t = time.monotonic()
    r = await post(gw, {**CHAT, "stream": True})
    assert r.headers["x-meritorder-backend"] == "up-b" and time.monotonic() - t < 5
    assert r.headers["x-meritorder-failed-over"] == "up-a/m-a 504"
    assert gw.inflight.count("up-a") == 0


async def test_a_stream_that_breaks_before_content_fails_over_but_one_with_content_does_not(monkeypatch):
    a = Upstream(lambda req: stream_response(b'data: {"id":"c","choices":[{"index":0,"delta":{"role":"assistant"}}]}\n\n',
                                             die=httpx.ReadError("reset")))
    gw = http_gateway(monkeypatch, a, Upstream(lambda req: stream_response(OK_STREAM)))
    r = await post(gw, {**CHAT, "stream": True})
    assert r.headers["x-meritorder-backend"] == "up-b"
    # Errors after content has started are passed through: switching then would duplicate output.
    late = b'data: {"id":"c","choices":[{"index":0,"delta":{"content":"par"}}]}\n\ndata: {"error":{"message":"oops","code":500}}\n\n'
    gw = http_gateway(monkeypatch, Upstream(lambda req: stream_response(late)), Upstream(lambda req: stream_response(OK_STREAM)))
    r = await post(gw, {**CHAT, "stream": True})
    assert r.headers["x-meritorder-backend"] == "up-a" and "oops" in r.text


async def test_an_error_after_content_has_started_is_recorded_as_a_failure(monkeypatch):
    """The client gets the provider's error in the stream; the ledger, the backend's health and the learned router's
    outcomes must see a failure too, not a 200."""
    late = (b'data: {"id":"c","choices":[{"index":0,"delta":{"content":"par"}}]}\n\n'
            b'data: {"error":{"message":"oops","code":500}}\n\n')
    gw = http_gateway(monkeypatch, Upstream(lambda req: stream_response(late)), Upstream(lambda req: stream_response(OK_STREAM)))
    r = await post(gw, {**CHAT, "stream": True})
    assert r.status_code == 200 and "oops" in r.text
    await gw.drain()
    row = (await gw.store.recent(1))[0]
    assert row["status"] == 500 and "oops" in row["error"]

    overloaded = (b'event: message_start\ndata: {"type":"message_start","message":{"id":"m","type":"message","role":"assistant",'
                  b'"content":[],"model":"m-a","usage":{"input_tokens":30,"output_tokens":1}}}\n\n'
                  b'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n'
                  b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hel"}}\n\n'
                  b'event: error\ndata: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}\n\n')
    gw = http_gateway(monkeypatch, Upstream(lambda req: stream_response(overloaded)), Upstream(lambda req: stream_response(OK_STREAM)),
                      primary_format="anthropic")
    r = await post(gw, {**CHAT, "stream": True})
    assert r.headers["x-meritorder-backend"] == "up-a" and "Overloaded" in r.text
    await gw.drain()
    row = (await gw.store.recent(1))[0]
    assert row["status"] == 529 and row["error"] == "up-a: Overloaded" and row["input_tokens"] == 30  # still billed for what it read


async def test_a_stream_that_breaks_while_checked_for_a_refusal_is_handled_and_recorded(monkeypatch):
    """`auto` reads a stream up to its first text to see whether the model refuses. Reasoning comes before text (Gemini
    always sends its thought summaries), so a stream can break during that check: it must end like any broken stream,
    with an error the client can read, a ledger row, and its in-flight slot released."""
    thinking = b'data: {"id":"c","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"Let me think"}}]}\n\n'
    gw = http_gateway(monkeypatch, Upstream(lambda req: stream_response(thinking, die=httpx.ReadError("reset"))),
                      Upstream(lambda req: stream_response(OK_STREAM)))
    second_model(gw)
    r = await post(gw, {**CHAT, "model": "auto", "stream": True})
    assert r.status_code == 200 and r.headers["x-meritorder-backend"] == "up-a"
    assert "Let me think" in r.text and "upstream stream failed: ReadError: reset" in r.text
    await gw.drain()
    row = (await gw.store.recent(1))[0]
    assert row["status"] == 502 and "ReadError" in row["error"]
    assert gw.inflight.count("up-a") == 0


async def test_a_refusal_is_answered_when_the_only_other_model_already_failed(monkeypatch):
    """`auto` retries a refused request on another model, but only one still to be tried: the model it would retry on
    was busy a moment ago, so retrying would only answer with that old 429 and throw the refusal away."""
    refusal = {**OPENAI_OK, "choices": [{"index": 0, "message": {"role": "assistant", "content": None, "refusal": "I can't help"},
                                          "finish_reason": "content_filter"}]}

    def a(req: httpx.Request) -> httpx.Response:
        if json.loads(req.content)["model"] == "m-a":
            return httpx.Response(429, json={"error": {"message": "slow down"}})
        return httpx.Response(200, json=refusal)

    gw = http_gateway(monkeypatch, Upstream(a), Upstream(httpx.Response(200, json=OPENAI_OK)))
    gw.cfg.model("m").routes = gw.cfg.model("m").routes[:1]  # m only on up-a, where it's busy
    second_model(gw)
    r = await post(gw, {**CHAT, "model": "auto"})
    assert r.status_code == 200 and r.headers["x-meritorder-model"] == "y"
    assert r.json()["choices"][0]["finish_reason"] == "content_filter" and r.headers["x-meritorder-failed-over"] == "up-a/m-a 429"


def test_retry_after_header_forms():
    assert retry_after(httpx.Headers({"retry-after": "42"})) == 42
    assert retry_after(httpx.Headers({"retry-after-ms": "1500"})) == 1.5
    date = time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.gmtime(time.time() + 120))
    assert 100 < retry_after(httpx.Headers({"retry-after": date})) <= 121
    assert retry_after(httpx.Headers({})) is None and retry_after(httpx.Headers({"retry-after": "soon"})) is None


async def test_retry_after_decides_how_long_a_busy_model_is_skipped(monkeypatch):
    a = Upstream(httpx.Response(429, headers={"retry-after": "42"}, json={"error": {"message": "slow down"}}),
                 httpx.Response(200, json=OPENAI_OK))
    gw = http_gateway(monkeypatch, a, Upstream(httpx.Response(200, json=OPENAI_OK)))
    second_model(gw)
    assert (await post(gw, CHAT)).headers["x-meritorder-backend"] == "up-b"
    out = gw.backends["up-a"].skipped(time.time(), "m-a")
    assert out and out.capacity and 40 <= out.until - time.time() <= 42
    r = await post(gw, {**CHAT, "model": "y"})  # another model on the same backend isn't affected
    assert r.status_code == 200 and r.headers["x-meritorder-backend"] == "up-a"


async def test_rejected_credentials_skip_the_whole_backend_with_a_clear_message(monkeypatch):
    a = Upstream(httpx.Response(401, json={"error": {"message": "invalid x-api-key"}}))
    gw = http_gateway(monkeypatch, a, Upstream(httpx.Response(200, json=OPENAI_OK)))
    second_model(gw)
    assert (await post(gw, CHAT)).headers["x-meritorder-backend"] == "up-b"
    r = await post(gw, {**CHAT, "model": "y"})  # only on up-a, which is skipped
    assert r.status_code == 503 and 50 <= int(r.headers["retry-after"]) <= 60
    assert "up-a: credentials rejected, trying again in" in r.json()["error"]["message"]
    r = await post(gw, CHAT, **{"authorization": f"Bearer {ADMIN_KEY}", "x-meritorder-backend": "up-a"})
    assert r.status_code == 503 and "is being skipped" in r.text and "invalid x-api-key" in r.text and r.headers["retry-after"]
    assert len(a.requests) == 1


async def test_a_busy_only_route_is_still_tried_and_recovery_is_announced(monkeypatch):
    a = Upstream(httpx.Response(529, headers={"retry-after": "3"}, json={"error": {"message": "Overloaded"}}),
                 httpx.Response(200, json=OPENAI_OK))
    gw = http_gateway(monkeypatch, a, Upstream(httpx.Response(200, json=OPENAI_OK)))
    second_model(gw)
    r = await post(gw, {**CHAT, "model": "y"})
    assert r.status_code == 529 and r.headers["retry-after"] == "3"  # nothing else serves it: the provider's answer
    r = await post(gw, {**CHAT, "model": "y"})  # skipped, but the only route: tried anyway rather than refused
    assert r.status_code == 200 and len(a.requests) == 2
    await gw.drain()
    texts = [x["text"] for x in await gw.state.recent_alerts()]
    assert texts[0] == "up-a (y-a) is working again." and "No other live backend serves it" in texts[1]


async def test_a_missing_deployment_is_not_retried_for_five_minutes(monkeypatch):
    a = Upstream(httpx.Response(404, json={"error": {"message": "The API deployment y-a does not exist"}}))
    gw = http_gateway(monkeypatch, a, Upstream(httpx.Response(200, json=OPENAI_OK)))
    second_model(gw)
    assert (await post(gw, {**CHAT, "model": "y"})).status_code == 404
    r = await post(gw, {**CHAT, "model": "y"})
    assert r.status_code == 503 and "model or deployment not found" in r.text and 290 <= int(r.headers["retry-after"]) <= 300
    assert len(a.requests) == 1


async def test_an_alert_lists_where_requests_go_even_when_the_failing_request_was_pinned(monkeypatch):
    from conftest import ADMIN_KEY

    a = Upstream(httpx.Response(404, json={"error": {"message": "The model 'm-a' does not exist"}}))
    gw = http_gateway(monkeypatch, a, Upstream(httpx.Response(200, json=OPENAI_OK)))
    r = await post(gw, CHAT, authorization=f"Bearer {ADMIN_KEY}", **{"x-meritorder-backend": "up-a"})
    assert r.status_code == 404  # pinned: only up-a may answer
    await gw.drain()
    text = (await gw.state.recent_alerts())[0]["text"]
    assert "up-a (m-a) is failing" in text and "Requests go to up-b meanwhile." in text  # the model's other live route
    # Unpinned, the same request goes straight to the other route.
    assert (await post(gw, CHAT)).headers["x-meritorder-backend"] == "up-b"


def test_a_model_that_stays_missing_is_checked_less_and_less_often():
    b = Backend(BackendConfig(name="x", format="openai", base_url="https://x"))
    waits = []
    for i in range(7):
        now = 1000.0 * (i + 1) * 10
        b.record_failure(now, "m", 404, None, "does not exist")
        waits.append(b.skipped(now, "m").until - now)
    assert waits == [300, 600, 1200, 2400, 3600, 3600, 3600]
    b.record_success("m")
    b.record_failure(0.0, "m", 404, None, "does not exist")
    assert b.skipped(0.0, "m").until == 300  # found again: back to the start


async def test_console_shows_what_is_skipped_and_why(monkeypatch):
    from conftest import ADMIN_KEY as admin

    a = Upstream(httpx.Response(429, headers={"retry-after": "30"}, json={"error": {"message": "slow down"}}))
    gw = http_gateway(monkeypatch, a, Upstream(httpx.Response(200, json=OPENAI_OK)))
    await post(gw, CHAT)
    from meritorder.app import create_app

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(gateway=gw)), base_url="http://t") as c:
        h = {"authorization": f"Bearer {admin}"}
        backends = {b["name"]: b for b in (await c.get("/v1/console/backends", headers=h)).json()["backends"]}
        attention = (await c.get("/v1/console/attention", headers=h)).json()["items"]
    skipped = backends["up-a"]["skipped_models"]["m-a"]
    assert skipped["capacity"] and skipped["status"] == 429 and "rate limited" in skipped["why"] and backends["up-a"]["skipped"] is None
    assert any(i["title"] == "up-a: m-a skipped" for i in attention)
    assert 'meritorder_failovers_total{backend="up-a",status="429"} 1' in gw.metrics.render()


def test_breaker_rules():
    b = Backend(BackendConfig(name="x", format="openai"))
    now = 1000.0
    assert b.record_failure(now, "m", 500) is None and b.record_failure(now, "m", 500) is None  # 2 server errors: still tried
    assert b.record_failure(now, "m", 500).until == now + 5 and not b.healthy(now, "m") and b.healthy(now)
    assert b.record_failure(now, "m", 500) is None  # already skipped: no second alert
    assert b.record_failure(now, "n", 502) is None and b.record_failure(now, "n", 502) is None
    assert b.record_failure(now, "n", 502).status == 502  # network-level, whatever the model: the whole backend
    assert not b.healthy(now) and not b.healthy(now, "z") and b.skipped(now).status == 502
    assert b.record_success("m") is True and b.healthy(now, "m") and b.healthy(now)
    assert b.record_success("m") is False  # nothing was being skipped
    c = Backend(BackendConfig(name="y", format="openai"))
    assert c.record_failure(now, "m", 429).until == now + 10 and c.record_failure(now, "m", 429) is None
    assert c.skipped(now, "m").until == now + 20  # doubling
    assert c.record_failure(now, "m", 429, retry_after=9999) is None and c.skipped(now, "m").until == now + 300  # capped
    assert c.record_failure(now, "m", 401).until == now + 60 and not c.healthy(now)  # credentials: the backend


def test_first_byte_timeout_defaults():
    assert BackendConfig(name="a", format="openai").first_byte_timeout() == 120
    assert BackendConfig(name="g", format="openai", kind="self_hosted").first_byte_timeout() is None  # vLLM may queue on purpose
    assert BackendConfig(name="g", format="openai", kind="self_hosted", first_byte_timeout_seconds=300).first_byte_timeout() == 300
    assert BackendConfig(name="n", format="openai", first_byte_timeout_seconds=None).first_byte_timeout() is None


def test_error_shapes():
    assert error_in({"type": "error", "error": {"type": "rate_limit_error", "message": "x"}}, "anthropic") == (429, "x")
    assert error_in({"type": "message_start"}, "anthropic") is None
    assert error_in({"error": "plain text"}, "openai") == (502, "plain text")
    assert error_in({"error": {"code": "503", "message": "busy"}}, "openai") == (503, "busy")
    assert error_in({"error": {"type": "insufficient_quota", "message": "q"}}, "openai") == (429, "q")
    assert error_in({"error": {"type": "tokens", "code": "rate_limit_exceeded", "message": "r"}}, "openai") == (429, "r")
    assert error_in({"error": {"type": "api_error", "code": "rate_limit_exceeded", "message": "r"}}, "openai") == (429, "r")
    assert error_in({"choices": [{"delta": {"content": "hi"}}]}, "openai") is None
    assert error_in({"data": [{"embedding": [0.1]}], "error": None}, "openai") is None
    assert error_in(None, "openai") is None
