"""Output limits: a request without max_tokens may write as much as its model can (less what won't fit in the context
window or under the per-request cost limit), and a non-streaming request that may run long is streamed from the
provider and assembled, so no reply is cut short by a limit the model doesn't have."""

import json

import httpx
import pytest
from conftest import ADMIN_KEY, TEST_CONFIG, auth
from test_http_backend import ANTHROPIC_OK, CHAT, OPENAI_OK, Upstream, http_gateway, post, stream_response, wire

from meritorder import catalog
from meritorder.backends.base import Backend, BackendError
from meritorder.backends.http import LONG_REPLY_TOKENS, runs_long
from meritorder.config import BackendConfig, Config, ModelConfig, Price, Route, Safeguards, build_config
from meritorder.formats.assemble import anthropic_message, openai_completion
from meritorder.proxy import Gateway
from meritorder.router import Candidate


def candidate(most=128_000, window=1_000_000, fmt="anthropic", kind="api", price=Price(input=8, output=40)):
    bc = BackendConfig(name="b", format=fmt, kind=kind, base_url="https://b.example")
    m = ModelConfig(name="m", max_output_tokens=most, context_window=window, routes=[Route(backend="b", model="m-1", price=price)])
    return Candidate(model=m, route=m.routes[0], backend=Backend(bc), cost=0, warm=False, quality=5), bc, m


def gateway_for(bc, m, max_usd=None):
    return Gateway(Config(backends=[bc], models=[m], safeguards=Safeguards(max_request_usd=max_usd)), None, None, {}, None, None)


def test_allowance_is_the_models_limit_unless_something_smaller_applies():
    cand, bc, m = candidate()
    gw = gateway_for(bc, m)
    assert gw.output_allowance({}, cand, 1000) == (128_000, "model")
    assert gw.output_allowance({"max_tokens": 4000}, cand, 1000) == (4000, "client")
    assert gw.output_allowance({"max_completion_tokens": 500_000}, cand, 1000) == (128_000, "model-capped")
    # The prompt estimate runs low on dense text, so a third more is allowed for it.
    assert gw.output_allowance({}, cand, 600_000) == (128_000, "model")
    assert gw.output_allowance({}, cand, 700_000) == (1_000_000 - 700_000 * 4 // 3, "context")
    assert gw.output_allowance({}, cand, 900_000) == (1, "context")  # won't fit anyway: the provider says so

    capped = gateway_for(bc, m, max_usd=3.0)  # Fable-like prices: 128,000 tokens would cost $5.12
    assert capped.output_allowance({}, cand, 4000) == (int((3.0 - 4000 * 8 / 1e6) * 1e6 / 40), "cost limit")
    assert capped.output_allowance({"max_tokens": 128_000}, cand, 4000) == (128_000, "client")  # refused later, not trimmed
    assert capped.output_allowance({}, cand, 400_000) == (0, "cost limit")  # the input alone is over the limit

    hosted, hbc, hm = candidate(kind="self_hosted")
    assert gateway_for(hbc, hm, max_usd=0.01).output_allowance({}, hosted, 4000) == (128_000, "model")  # paid by the hour

    unknown, ubc, um = candidate(most=None, window=None)
    assert gateway_for(ubc, um).output_allowance({}, unknown, 10) == (16_000, "default")


def test_upstream_requests_carry_the_allowance():
    cand, bc, m = candidate()
    gw = gateway_for(bc, m)
    up, _ = gw.build_upstream({"messages": [{"role": "user", "content": "hi"}]}, "openai", cand, "s")
    assert up["max_tokens"] == 128_000

    oa, obc, om = candidate(fmt="openai")
    gw = gateway_for(obc, om, max_usd=1.0)
    up, _ = gw.build_upstream({"messages": [{"role": "user", "content": "hi"}], "max_completion_tokens": 900}, "openai", oa, "s")
    assert up["max_completion_tokens"] == 900 and "max_tokens" not in up  # the client's own choice is left alone
    up, _ = gw.build_upstream({"messages": [{"role": "user", "content": "hi"}]}, "openai", oa, "s", prompt_tokens=10)
    assert up["max_tokens"] == int((1.0 - 10 * 8 / 1e6) * 1e6 / 40)  # what fits under $1
    up, _ = gateway_for(obc, om).build_upstream({"messages": []}, "openai", oa, "s")
    assert up["max_tokens"] == 128_000  # some providers default lower, so the model's maximum is asked for

    hosted, hbc, hm = candidate(fmt="openai", kind="self_hosted")
    up, _ = gateway_for(hbc, hm).build_upstream({"messages": []}, "openai", hosted, "s")
    assert "max_tokens" not in up  # vLLM already allows the rest of the context window


async def test_cost_limit_trims_an_open_ended_reply_and_refuses_a_promised_one(client, gateway):
    big = next(m for m in gateway.cfg.models if m.name == "big")
    big.max_output_tokens = 128_000
    gateway.cfg.safeguards = Safeguards(max_request_usd=0.05)  # $15/M output: 128,000 tokens would be $1.92
    body = {"model": "big", "messages": [{"role": "user", "content": "hi"}]}
    r = await client.post("/v1/chat/completions", json=body, headers=auth())
    assert r.status_code == 200 and r.headers["x-meritorder-max-tokens-source"] == "cost limit"
    assert 3000 < int(r.headers["x-meritorder-max-tokens"]) < 3334
    r = await client.post("/v1/chat/completions", json={**body, "max_tokens": 10_000}, headers=auth())
    assert r.status_code == 400 and "per-request limit" in r.json()["error"]["message"]
    gateway.cfg.safeguards = Safeguards(max_request_usd=0.000001)
    r = await client.post("/v1/chat/completions", json=body, headers=auth())
    assert r.status_code == 400 and "before any output" in r.json()["error"]["message"]
    gateway.cfg.safeguards = Safeguards()
    r = await client.post("/v1/chat/completions", json=body, headers=auth())
    assert r.headers["x-meritorder-max-tokens"] == "128000" and r.headers["x-meritorder-max-tokens-source"] == "model"


async def test_eval_run_keys_may_choose_a_backend_and_no_one_else_can_mint_one(client, gateway):
    body = {"model": "small", "messages": [{"role": "user", "content": "hi"}]}
    pin = {"x-meritorder-backend": "mock-openai"}
    key_id, secret = await gateway.store.create_key("root", "ops", "user", tags={"eval_run": "ev_x"})
    assert (await client.post("/v1/chat/completions", json=body, headers={**auth(secret), **pin})).status_code == 200
    _, plain = await gateway.store.create_key("alice", "default", "user")
    assert (await client.post("/v1/chat/completions", json=body, headers={**auth(plain), **pin})).status_code == 403
    r = await client.post("/v1/admin/keys", headers=auth(ADMIN_KEY), json={"user": "bob", "tags": {"eval_run": "ev_y"}})
    assert r.status_code == 422 and "reserved" in r.json()["detail"]


# ------------------------------------------------------------------ long replies stream from the provider

def sse(*events: dict) -> list[bytes]:
    return [f"event: {e['type']}\ndata: {json.dumps(e)}\n\n".encode() if "type" in e else f"data: {json.dumps(e)}\n\n".encode()
            for e in events]


ANTHROPIC_EVENTS = [
    {"type": "message_start", "message": {"id": "msg_9", "type": "message", "role": "assistant", "model": "claude-x", "content": [],
                                          "usage": {"input_tokens": 12, "output_tokens": 1, "cache_read_input_tokens": 100}}},
    {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "Let me "}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "think."}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig=="}},
    {"type": "content_block_stop", "index": 0},
    {"type": "ping"},
    {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Hello, "}},
    {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "world"}},
    {"type": "content_block_stop", "index": 1},
    {"type": "content_block_start", "index": 2, "content_block": {"type": "tool_use", "id": "toolu_1", "name": "run", "input": {}}},
    {"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": '{"cmd": "ls'}},
    {"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": ' -la"}'}},
    {"type": "content_block_stop", "index": 2},
    {"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": None}, "usage": {"output_tokens": 4321}},
    {"type": "message_stop"},
]


def test_assembling_an_anthropic_stream():
    msg = anthropic_message(ANTHROPIC_EVENTS)
    assert msg["id"] == "msg_9" and msg["stop_reason"] == "tool_use"
    assert msg["content"] == [{"type": "thinking", "thinking": "Let me think.", "signature": "sig=="}, {"type": "text", "text": "Hello, world"},
                              {"type": "tool_use", "id": "toolu_1", "name": "run", "input": {"cmd": "ls -la"}}]
    assert msg["usage"] == {"input_tokens": 12, "output_tokens": 4321, "cache_read_input_tokens": 100}
    cut = anthropic_message(ANTHROPIC_EVENTS[:13])  # broken off inside the tool call's arguments
    assert cut["content"][2]["input"] == {}
    cited = anthropic_message([
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "citations_delta", "citation": {"cited_text": "x"}}},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "t", "name": "f", "input": {}}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{not json"}},
        {"type": "content_block_stop", "index": 1}])
    assert cited["content"][0]["citations"] == [{"cited_text": "x"}] and cited["content"][1]["input"] == {}
    assert cited["type"] == "message" and cited["role"] == "assistant"


def test_assembling_an_openai_stream():
    chunks = [
        {"id": "c9", "created": 5, "model": "m-1", "choices": [{"index": 0, "delta": {"role": "assistant", "reasoning_content": "hmm"}}]},
        {"id": "c9", "choices": [{"index": 0, "delta": {"content": "Hi "}}]},
        {"id": "c9", "choices": [{"index": 0, "delta": {"content": "there", "tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "run", "arguments": '{"a"'}}]}}]},
        {"id": "c9", "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": ": 1}"}}]}, "finish_reason": "length"}]},
        {"id": "c9", "choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 9, "cost": 0.01}},
    ]
    out = openai_completion(chunks)
    assert out["id"] == "c9" and out["model"] == "m-1" and out["usage"] == {"prompt_tokens": 7, "completion_tokens": 9, "cost": 0.01}
    [choice] = out["choices"]
    assert choice["finish_reason"] == "length"
    assert choice["message"] == {"role": "assistant", "content": "Hi there", "reasoning_content": "hmm",
                                 "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "run", "arguments": '{"a": 1}'}}]}
    assert openai_completion([])["choices"][0]["message"]["content"] == ""
    details = openai_completion([
        {"choices": [{"index": 0, "delta": {"reasoning_details": [{"type": "reasoning.text", "text": "Let ", "index": 0, "format": "google-gemini-v1"}]}}]},
        {"choices": [{"index": 0, "delta": {"reasoning_details": [{"type": "reasoning.text", "text": "me see", "index": 0, "signature": None}]}}]},
        {"choices": [{"index": 0, "delta": {"reasoning_details": [{"type": "reasoning.encrypted", "data": "AbC", "index": 1, "id": "tool_1"}]}}]},
        {"choices": [{"index": 0, "delta": {"reasoning_details": [{"type": "reasoning.text", "text": "", "index": 0, "signature": "sig"}]}, "finish_reason": "stop"}]}])
    assert details["choices"][0]["message"]["reasoning_details"] == [  # whole again, so they can be sent back as they were
        {"type": "reasoning.text", "text": "Let me see", "index": 0, "format": "google-gemini-v1", "signature": "sig"},
        {"type": "reasoning.encrypted", "data": "AbC", "index": 1, "id": "tool_1"}]


async def test_long_non_streaming_requests_stream_from_the_provider():
    assert runs_long(None) and runs_long(LONG_REPLY_TOKENS + 1) and not runs_long(LONG_REPLY_TOKENS) and runs_long("lots")
    up = Upstream(stream_response(*sse(*ANTHROPIC_EVENTS)))
    b = wire(BackendConfig(name="an", format="anthropic", base_url="https://an.example"), up)
    resp = await b.send({"model": "claude-x", "max_tokens": 128_000, "messages": []}, False, {})
    assert up.body()["stream"] is True and resp.events is None and resp.body["content"][1]["text"] == "Hello, world"

    short = Upstream(httpx.Response(200, json=ANTHROPIC_OK))
    b = wire(BackendConfig(name="an", format="anthropic", base_url="https://an.example"), short)
    resp = await b.send({"model": "claude-x", "max_tokens": 4000, "messages": []}, False, {})
    assert "stream" not in short.body() and resp.body == ANTHROPIC_OK

    plain = Upstream(httpx.Response(200, json=ANTHROPIC_OK))  # a provider that answers a stream request with JSON
    b = wire(BackendConfig(name="an", format="anthropic", base_url="https://an.example"), plain)
    assert (await b.send({"model": "claude-x", "max_tokens": 64_000, "messages": []}, False, {})).body == ANTHROPIC_OK

    overloaded = Upstream(stream_response(*sse(ANTHROPIC_EVENTS[0], {"type": "error", "error": {"type": "overloaded_error", "message": "busy"}})))
    b = wire(BackendConfig(name="an", format="anthropic", base_url="https://an.example"), overloaded)
    with pytest.raises(BackendError) as e:
        await b.send({"model": "claude-x", "max_tokens": 64_000, "messages": []}, False, {})
    assert e.value.status == 529 and e.value.retryable

    oa = Upstream(stream_response(*sse({"id": "c", "choices": [{"index": 0, "delta": {"content": "hey"}, "finish_reason": "stop"}]},
                                       {"id": "c", "choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 1}}), b"data: [DONE]\n\n"))
    b = wire(BackendConfig(name="oa", format="openai", base_url="https://oa.example/v1", max_tokens_field="max_completion_tokens"), oa)
    resp = await b.send({"model": "gpt-x", "messages": []}, False, {})
    assert oa.body()["stream"] is True and oa.body()["stream_options"] == {"include_usage": True}
    assert resp.body["choices"][0]["message"]["content"] == "hey" and resp.body["usage"]["completion_tokens"] == 1

    broken = Upstream(stream_response(*sse(ANTHROPIC_EVENTS[0]), die=httpx.ReadError("reset")))
    b = wire(BackendConfig(name="an", format="anthropic", base_url="https://an.example"), broken)
    with pytest.raises(BackendError) as e:
        await b.send({"model": "claude-x", "max_tokens": 64_000, "messages": []}, False, {})
    assert e.value.status == 502 and "broke off" in e.value.message and e.value.retryable

    hosted = Upstream(httpx.Response(200, json=OPENAI_OK))
    b = wire(BackendConfig(name="gpu", kind="self_hosted", format="openai", base_url="https://gpu.example/v1"), hosted)
    await b.send({"model": "qwen", "messages": []}, False, {})
    assert "stream" not in hosted.body()


async def test_a_long_reply_reaches_a_non_streaming_client_whole(monkeypatch):
    a = Upstream(stream_response(*sse({"id": "c", "model": "m-a", "choices": [{"index": 0, "delta": {"content": "long answer"}, "finish_reason": "stop"}]},
                                      {"id": "c", "choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 50_000}})))
    gw = http_gateway(monkeypatch, a, Upstream(httpx.Response(200, json=OPENAI_OK)))
    r = await post(gw, {**CHAT, "max_tokens": 100_000})
    assert r.status_code == 200 and r.json()["choices"][0]["message"]["content"] == "long answer"
    assert a.body()["stream"] is True and r.headers["x-meritorder-max-tokens"] == "100000"
    await gw.drain()
    assert (await gw.store.recent(1))[0]["output_tokens"] == 50_000


async def test_a_provider_that_cant_cover_the_whole_reply_gets_what_it_can(monkeypatch):
    """OpenRouter reserves credit for max_tokens up front and refuses (402) what the account can't cover, saying how much
    it can: the same route is asked again for exactly that (up to three times, as other requests reserve credit meanwhile).
    A 402 with nothing to go on moves to the next backend."""
    short = httpx.Response(402, json={"error": {"code": 402, "message": "This request requires more credits, or fewer max_tokens. "
                                                "You requested up to 16000 tokens, but can only afford 9000."}})
    a = Upstream(short, httpx.Response(200, json={**OPENAI_OK, "model": "m-a"}))
    gw = http_gateway(monkeypatch, a, Upstream(httpx.Response(200, json=OPENAI_OK)))
    r = await post(gw, CHAT)
    assert r.status_code == 200 and r.headers["x-meritorder-backend"] == "up-a" and "x-meritorder-failed-over" not in r.headers
    assert [json.loads(q.content).get("max_tokens") for q in a.requests] == [None, 9000]  # unknown limit: none sent, then what's affordable
    assert r.headers["x-meritorder-max-tokens"] == "9000" and r.headers["x-meritorder-max-tokens-source"] == "provider credit"

    def afford(n):
        return httpx.Response(402, json={"error": {"code": 402, "message": f"... but can only afford {n}."}})
    a = Upstream(short, afford(7000), afford(5000), httpx.Response(200, json={**OPENAI_OK, "model": "m-a"}))  # credit other requests reserve
    gw = http_gateway(monkeypatch, a, Upstream(httpx.Response(200, json=OPENAI_OK)))
    r = await post(gw, CHAT)
    assert r.headers["x-meritorder-backend"] == "up-a" and [json.loads(q.content).get("max_tokens") for q in a.requests] == [None, 9000, 7000, 5000]
    a = Upstream(short, afford(7000), afford(5000), afford(3000))  # three tries at what's affordable, then the next backend
    gw = http_gateway(monkeypatch, a, Upstream(httpx.Response(200, json=OPENAI_OK)))
    r = await post(gw, CHAT)
    assert r.headers["x-meritorder-backend"] == "up-b" and len(a.requests) == 4

    a = Upstream(short)  # still refused at what it said it could afford: another backend takes it
    gw = http_gateway(monkeypatch, a, Upstream(httpx.Response(200, json=OPENAI_OK)))
    r = await post(gw, CHAT)
    assert r.status_code == 200 and r.headers["x-meritorder-backend"] == "up-b" and len(a.requests) == 2

    broke = Upstream(httpx.Response(402, json={"error": {"message": "Insufficient credits"}}))
    gw = http_gateway(monkeypatch, broke, Upstream(httpx.Response(200, json=OPENAI_OK)))
    r = await post(gw, CHAT)
    assert r.status_code == 200 and r.headers["x-meritorder-failed-over"] == "up-a/m-a 402" and len(broke.requests) == 1

    asked = Upstream(short, httpx.Response(200, json=OPENAI_OK))  # the client's own max_tokens is never changed
    gw = http_gateway(monkeypatch, asked, Upstream(httpx.Response(200, json=OPENAI_OK)))
    r = await post(gw, {**CHAT, "max_tokens": 12000})
    assert r.headers["x-meritorder-backend"] == "up-b" and len(asked.requests) == 1


def test_output_limits_come_from_prices_json(monkeypatch):
    raw = {**TEST_CONFIG, "backends": {**TEST_CONFIG["backends"], "openrouter": {"kind": "mock", "format": "openai", "base_url": "https://openrouter.ai/api/v1"}}}
    raw["models"] = {**TEST_CONFIG["models"], "big": {**TEST_CONFIG["models"]["big"], "routes": [
        *TEST_CONFIG["models"]["big"]["routes"], {"backend": "openrouter", "model": "vendor/big", "price": {"input": 3, "output": 15}}]}}
    cfg = build_config(raw, prices_data={"max_output": {"big": 64_000}, "context": {"big": 200_000}}, quiet=True)
    big = next(m for m in cfg.models if m.name == "big")
    assert big.max_output_tokens == 64_000 and big.context_window == 200_000

    catalog_data = {"data": [{"id": "vendor/big", "pricing": {"prompt": "0.000003", "completion": "0.000015"}, "context_length": 200_000,
                              "top_provider": {"max_completion_tokens": 64_000}}]}
    _, n, missing, data = catalog.refresh_prices(cfg, catalog_data, write=False)
    assert n == 1 and data["max_output"] == {"big": 64_000}
