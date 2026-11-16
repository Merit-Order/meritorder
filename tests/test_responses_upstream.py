"""OpenAI backends with `api: responses`: Chat Completions requests go to /responses and come back as Chat
Completions. Found live: GPT-5.6 rejects function tools with reasoning on /chat/completions."""

import copy
import json

import httpx
from conftest import TEST_CONFIG
from test_http_backend import Upstream, chunks, collect, post, stream_response

from meritorder.app import build_gateway
from meritorder.config import build_config
from meritorder.formats.responses_upstream import ResponsesToChatStream, chat_to_responses_request, responses_to_chat_completion
from meritorder.formats.sse import event, iter_sse
from meritorder.state import MemoryState
from meritorder.store import MemoryStore

TOOL = {"type": "function", "function": {"name": "get_weather", "description": "Weather",
                                         "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}


def test_request_conversion_covers_roles_tools_limits_and_effort():
    body = {"model": "gpt-x", "max_completion_tokens": 300, "reasoning_effort": "high", "temperature": 0.2, "prompt_cache_key": "s1",
            "tools": [TOOL], "tool_choice": {"type": "function", "function": {"name": "get_weather"}}, "stream": True,
            "stop": ["x"], "messages": [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": [{"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]},
                {"role": "assistant", "content": "checking", "tool_calls": [{"id": "call_1", "type": "function",
                                                                              "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}}]},
                {"role": "tool", "tool_call_id": "call_1", "content": [{"type": "text", "text": "sunny"}]},
            ]}
    out = chat_to_responses_request(body)
    assert out["store"] is False and out["stream"] is True and out["max_output_tokens"] == 300
    assert out["reasoning"] == {"effort": "high"} and out["temperature"] == 0.2 and out["prompt_cache_key"] == "s1" and "stop" not in out
    assert out["tools"] == [{"type": "function", "name": "get_weather", "description": "Weather",
                             "parameters": TOOL["function"]["parameters"], "strict": False}]
    assert out["tool_choice"] == {"type": "function", "name": "get_weather"}
    assert out["input"] == [
        {"type": "message", "role": "developer", "content": "be brief"},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "look"},
                                                        {"type": "input_image", "image_url": "data:image/png;base64,AA", "detail": "auto"}]},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "checking"}]},
        {"type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": "{\"city\":\"Paris\"}"},
        {"type": "function_call_output", "call_id": "call_1", "output": "sunny"},
    ]
    schema = chat_to_responses_request({"messages": [], "max_tokens": 5, "response_format": {"type": "json_schema", "json_schema": {"name": "r", "schema": {"type": "object"}, "strict": True}}})
    assert schema["max_output_tokens"] == 5 and schema["text"] == {"format": {"type": "json_schema", "name": "r", "schema": {"type": "object"}, "strict": True}}
    assert chat_to_responses_request({"messages": [], "response_format": {"type": "json_object"}})["text"] == {"format": {"type": "json_object"}}


USAGE = {"input_tokens": 100, "input_tokens_details": {"cached_tokens": 60}, "output_tokens": 20,
         "output_tokens_details": {"reasoning_tokens": 12}, "total_tokens": 120}


def test_response_conversion_text_calls_refusals_and_limits():
    resp = {"id": "resp_1", "created_at": 5, "model": "gpt-x", "status": "completed", "usage": USAGE, "output": [
        {"type": "reasoning", "summary": []},
        {"type": "message", "content": [{"type": "output_text", "text": "hi "}, {"type": "output_text", "text": "there"}]},
        {"type": "function_call", "id": "fc_1", "call_id": "call_9", "name": "get_weather", "arguments": "{}"}]}
    out = responses_to_chat_completion(resp)
    choice = out["choices"][0]
    assert out["object"] == "chat.completion" and out["id"] == "resp_1" and choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] == "hi there" and choice["message"]["tool_calls"][0]["id"] == "call_9"
    assert out["usage"] == {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
                            "prompt_tokens_details": {"cached_tokens": 60}, "completion_tokens_details": {"reasoning_tokens": 12}}
    cut = responses_to_chat_completion({"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}, "output": []})
    assert cut["choices"][0]["finish_reason"] == "length" and cut["choices"][0]["message"]["content"] is None and "usage" not in cut
    no = responses_to_chat_completion({"status": "completed", "output": [{"type": "message", "content": [{"type": "refusal", "refusal": "no"}]}]})
    assert no["choices"][0]["message"]["refusal"] == "no" and no["choices"][0]["finish_reason"] == "stop"


def ev(d):
    return event(d, d["type"])


def test_stream_conversion_text_then_tool_call_then_usage():
    conv = ResponsesToChatStream()
    out = []
    for d in [{"type": "response.created", "response": {"id": "resp_1", "model": "gpt-x", "created_at": 5}},
              {"type": "response.reasoning_summary_text.delta", "delta": "thinking"},
              {"type": "response.output_text.delta", "delta": "he"}, {"type": "response.output_text.delta", "delta": "llo"},
              {"type": "response.output_item.added", "output_index": 1, "item": {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "get_weather"}},
              {"type": "response.function_call_arguments.delta", "item_id": "fc_1", "delta": "{\"city\":"},
              {"type": "response.function_call_arguments.delta", "item_id": "fc_1", "delta": "\"Paris\"}"},
              {"type": "response.function_call_arguments.delta", "item_id": "unknown", "delta": "x"},
              {"type": "response.completed", "response": {"status": "completed", "usage": USAGE}},
              {"type": "response.output_text.delta", "delta": "after the end"}]:
        out += conv.feed(ev(d))
    data = [e.json() for e in out if e.data != "[DONE]"]
    assert out[-1].data == "[DONE]" and all(d["id"] == "resp_1" for d in data)
    assert data[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    assert "".join(d["choices"][0]["delta"].get("content") or "" for d in data if d["choices"]) == "hello"
    args = "".join(tc["function"].get("arguments") or "" for d in data if d["choices"] for tc in d["choices"][0]["delta"].get("tool_calls") or [])
    assert args == "{\"city\":\"Paris\"}"
    assert data[-2]["choices"][0]["finish_reason"] == "tool_calls" and data[-1]["usage"]["prompt_tokens_details"]["cached_tokens"] == 60


def test_stream_errors_become_an_error_chunk():
    conv = ResponsesToChatStream()
    out = conv.feed(ev({"type": "response.failed", "response": {"error": {"message": "overloaded"}}}))
    assert out[0].json() == {"error": {"message": "overloaded", "type": "api_error"}} and out[1].data == "[DONE]"
    assert ResponsesToChatStream().feed(ev({"type": "error", "message": "bad"}))[0].json()["error"]["message"] == "bad"
    assert ResponsesToChatStream().feed(event("not json")) == []


def gateway(monkeypatch, upstream: Upstream):
    monkeypatch.setenv("MO_TEST_OAI", "key-oai")
    raw = copy.deepcopy(TEST_CONFIG)
    raw["backends"] = {"oai": {"kind": "api", "format": "openai", "base_url": "https://api.openai.example/v1", "api": "responses",
                               "max_tokens_field": "max_completion_tokens", "auth": {"type": "api_key", "value_env": "MO_TEST_OAI"}}}
    raw["models"] = {"g": {"quality": {"coding": 4, "general": 4}, "routes": [{"backend": "oai", "model": "gpt-x", "price": {"input": 1, "output": 8}}]}}
    raw["baseline_model"] = "g"
    gw = build_gateway(build_config(raw, quiet=True), store=MemoryStore(), state=MemoryState())
    gw.backends["oai"].client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    return gw


async def test_gateway_sends_chat_requests_to_responses_and_records_usage(monkeypatch):
    up = Upstream(httpx.Response(200, json={"id": "resp_1", "status": "completed", "model": "gpt-x", "usage": USAGE, "output": [
        {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}]}))
    gw = gateway(monkeypatch, up)
    r = await post(gw, {"model": "g", "max_tokens": 64, "tools": [TOOL], "messages": [{"role": "user", "content": "weather?"}]})
    assert r.status_code == 200 and r.json()["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "get_weather"
    sent = up.body()
    assert str(up.requests[0].url).endswith("/v1/responses") and sent["max_output_tokens"] == 64 and sent["tools"][0]["strict"] is False
    assert "messages" not in sent and "max_completion_tokens" not in sent
    await gw.drain()
    row = (await gw.store.recent(1))[0]
    assert row["input_tokens"] == 40 and row["cache_read_tokens"] == 60 and row["output_tokens"] == 20


async def test_claude_code_style_stream_through_responses(monkeypatch):
    sse = "".join(f"event: {d['type']}\ndata: {json.dumps(d)}\n\n" for d in [
        {"type": "response.created", "response": {"id": "resp_2", "model": "gpt-x", "created_at": 1}},
        {"type": "response.output_item.added", "output_index": 0, "item": {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "Write"}},
        {"type": "response.function_call_arguments.delta", "item_id": "fc_1", "delta": "{\"path\":\"a.py\"}"},
        {"type": "response.completed", "response": {"status": "completed", "usage": USAGE}}]).encode()
    up = Upstream(lambda req: stream_response(sse[:40], sse[40:]))
    gw = gateway(monkeypatch, up)
    r = await post(gw, {"model": "g", "max_tokens": 64, "stream": True, "messages": [{"role": "user", "content": "write it"}],
                        "tools": [{"name": "Write", "description": "w", "input_schema": {"type": "object"}}]}, path="/v1/messages")
    assert r.status_code == 200 and up.body()["stream"] is True
    evs = await collect(iter_sse(chunks(r.content)))
    kinds = [e.event for e in evs]
    assert kinds[0] == "message_start" and kinds[-1] == "message_stop"
    starts = [e.json() for e in evs if e.event == "content_block_start"]
    assert starts and starts[0]["content_block"]["type"] == "tool_use" and starts[0]["content_block"]["name"] == "Write"
    assert any((e.json() or {}).get("delta", {}).get("stop_reason") == "tool_use" for e in evs)
    await gw.drain()
    assert (await gw.store.recent(1))[0]["cache_read_tokens"] == 60
