import json

from meritorder.formats.sse import SSEEvent, event
from meritorder.formats.translate import (
    AnthropicToOpenAIStream,
    OpenAIToAnthropicStream,
    anthropic_to_openai_request,
    anthropic_to_openai_response,
    openai_to_anthropic_request,
    openai_to_anthropic_response,
    usage_from_openai,
)


def test_openai_request_to_anthropic():
    body = {
        "model": "x",
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "weather in Paris?"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "weather", "arguments": '{"city":"Paris"}'}},
                {"id": "call_2", "type": "function", "function": {"name": "time", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
            {"role": "tool", "tool_call_id": "call_2", "content": "noon"},
        ],
        "tools": [{"type": "function", "function": {"name": "weather", "parameters": {"type": "object"}}}],
        "tool_choice": "required",
        "stop": "END",
        "reasoning_effort": "high",
    }
    out = openai_to_anthropic_request(body, 1000)
    assert out["system"] == [{"type": "text", "text": "be brief"}]
    assert out["max_tokens"] == 1000
    assert "temperature" not in out
    assert out["stop_sequences"] == ["END"]
    assert out["tool_choice"] == {"type": "any"}  # a forced choice stays forced where the model takes it
    assert out["output_config"] == {"effort": "high"}
    assert out["tools"][0] == {"name": "weather", "input_schema": {"type": "object"}}
    roles = [m["role"] for m in out["messages"]]
    assert roles == ["user", "assistant", "user"]  # both tool results merged into one user turn
    assert [b["type"] for b in out["messages"][1]["content"]] == ["tool_use", "tool_use"]
    assert out["messages"][1]["content"][0]["input"] == {"city": "Paris"}
    results = out["messages"][2]["content"]
    assert [r["tool_use_id"] for r in results] == ["call_1", "call_2"]


def test_anthropic_request_to_openai():
    body = {
        "model": "x",
        "max_tokens": 50,
        "system": [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}],
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "hmm", "signature": "abc"},
                {"type": "text", "text": "let me check"},
                {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"q": 1}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": [{"type": "text", "text": "42"}]},
                {"type": "text", "text": "thanks"},
            ]},
        ],
        "tools": [{"name": "lookup", "input_schema": {"type": "object"}}, {"type": "web_search_20260209", "name": "web_search"}],
        "tool_choice": {"type": "any"},
        "stream": True,
    }
    out = anthropic_to_openai_request(body, "max_completion_tokens")
    assert out["max_completion_tokens"] == 50
    assert out["messages"][0] == {"role": "system", "content": "sys"}
    assistant = out["messages"][2]
    assert assistant["content"] == "let me check"
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {"q": 1}
    # tool results must directly follow the assistant tool call, before the user's text
    assert out["messages"][3] == {"role": "tool", "tool_call_id": "toolu_1", "content": "42"}
    assert out["messages"][4] == {"role": "user", "content": "thanks"}
    assert len(out["tools"]) == 1  # server tools dropped
    assert out["tool_choice"] == "required"
    assert out["stream_options"] == {"include_usage": True}


def test_responses_round_trip():
    a = {
        "id": "msg_1", "type": "message", "role": "assistant", "model": "m",
        "content": [{"type": "text", "text": "hello"}, {"type": "tool_use", "id": "t1", "name": "f", "input": {"a": 1}}],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 10, "cache_read_input_tokens": 90, "cache_creation_input_tokens": 0, "output_tokens": 5},
    }
    o = anthropic_to_openai_response(a, "m")
    assert o["choices"][0]["finish_reason"] == "tool_calls"
    assert o["usage"]["prompt_tokens"] == 100
    assert o["usage"]["prompt_tokens_details"]["cached_tokens"] == 90
    back = openai_to_anthropic_response(o, "m")
    assert back["stop_reason"] == "tool_use"
    assert back["content"][1]["input"] == {"a": 1}
    assert back["usage"]["cache_read_input_tokens"] == 90
    assert back["usage"]["input_tokens"] == 10


def test_usage_from_openai_cached():
    u = usage_from_openai({"prompt_tokens": 1000, "completion_tokens": 20, "prompt_tokens_details": {"cached_tokens": 800}})
    assert (u.input_tokens, u.cache_read_tokens, u.output_tokens) == (200, 800, 20)


def _anthropic_events():
    return [
        event({"type": "message_start", "message": {"usage": {"input_tokens": 5, "cache_read_input_tokens": 50, "output_tokens": 1}}}, "message_start"),
        event({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}, "content_block_start"),
        event({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi"}}, "content_block_delta"),
        event({"type": "content_block_stop", "index": 0}, "content_block_stop"),
        event({"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "t1", "name": "f", "input": {}}}, "content_block_start"),
        event({"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"a":'}}, "content_block_delta"),
        event({"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "1}"}}, "content_block_delta"),
        event({"type": "content_block_stop", "index": 1}, "content_block_stop"),
        event({"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 7}}, "message_delta"),
        event({"type": "message_stop"}, "message_stop"),
    ]


def test_anthropic_stream_to_openai():
    conv = AnthropicToOpenAIStream("m", include_usage=True)
    out = [e for ev in _anthropic_events() for e in conv.feed(ev)]
    assert out[-1].data == "[DONE]"
    chunks = [json.loads(e.data) for e in out[:-1]]
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"])
    assert text == "Hi"
    args = "".join(
        tc["function"].get("arguments", "")
        for c in chunks if c["choices"] for tc in c["choices"][0]["delta"].get("tool_calls", [])
    )
    assert json.loads(args) == {"a": 1}
    finish = [c["choices"][0]["finish_reason"] for c in chunks if c["choices"] and c["choices"][0]["finish_reason"]]
    assert finish == ["tool_calls"]
    assert chunks[-1]["usage"] == {"prompt_tokens": 55, "completion_tokens": 7, "total_tokens": 62, "prompt_tokens_details": {"cached_tokens": 50},
                                    "completion_tokens_details": {"reasoning_tokens": 0}}


def test_openai_stream_to_anthropic():
    def chunk(delta, finish=None):
        return event({"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})

    events = [
        chunk({"role": "assistant", "content": ""}),
        chunk({"content": "Hel"}),
        chunk({"content": "lo"}),
        chunk({"tool_calls": [{"index": 0, "id": "c1", "type": "function", "function": {"name": "f", "arguments": ""}}]}),
        chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"x": 2}'}}]}),
        chunk({}, "tool_calls"),
        event({"choices": [], "usage": {"prompt_tokens": 30, "completion_tokens": 4, "prompt_tokens_details": {"cached_tokens": 20}}}),
        SSEEvent(data="[DONE]"),
    ]
    conv = OpenAIToAnthropicStream("m")
    out = [e for ev in events for e in conv.feed(ev)]
    types = [e.event for e in out]
    assert types[0] == "message_start"
    assert types[-2:] == ["message_delta", "message_stop"]
    assert types.count("content_block_start") == 2 and types.count("content_block_stop") == 2
    payloads = [json.loads(e.data) for e in out]
    text = "".join(p["delta"]["text"] for p in payloads if p.get("type") == "content_block_delta" and p["delta"]["type"] == "text_delta")
    assert text == "Hello"
    args = "".join(p["delta"]["partial_json"] for p in payloads if p.get("type") == "content_block_delta" and p["delta"]["type"] == "input_json_delta")
    assert json.loads(args) == {"x": 2}
    delta = payloads[-2]
    assert delta["delta"]["stop_reason"] == "tool_use"
    assert delta["usage"] == {"input_tokens": 10, "cache_read_input_tokens": 20, "cache_creation_input_tokens": 0, "output_tokens": 4}
