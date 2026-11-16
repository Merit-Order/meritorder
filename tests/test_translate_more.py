"""Format translation edge cases: images, tool results, thinking, stop reasons, usage, errors, and stream sequencing."""

import json

import pytest

from meritorder.formats.sse import SSEEvent, event
from meritorder.formats.translate import (
    STOP_A2O,
    STOP_O2A,
    AnthropicToOpenAIStream,
    OpenAIToAnthropicStream,
    UsageTracker,
    _loads,
    anthropic_to_openai_request,
    anthropic_to_openai_response,
    openai_to_anthropic_request,
    openai_to_anthropic_response,
    text_of,
    usage_from_anthropic,
    usage_from_openai,
)


def payloads(events):
    return [e.json() for e in events if e.json()]


def test_argument_and_text_helpers():
    assert _loads({"a": 1}) == {"a": 1} and _loads("") == {} and _loads(None) == {}
    assert _loads('{"city":"Paris"}') == {"city": "Paris"}
    assert _loads("not json") == {"arguments": "not json"}  # a model's broken JSON is kept, not lost
    assert _loads("[1,2]") == {"value": [1, 2]} and _loads("3") == {"value": 3}
    assert text_of(None) == "" and text_of("hi") == "hi"
    assert text_of([{"type": "text", "text": "a"}, "b", {"type": "image_url"}, {"type": "output_text", "text": "c"}]) == "abc"


# ------------------------------------------------------------------ OpenAI request -> Anthropic


def test_openai_request_parts_images_and_options():
    body = {
        "model": "m",
        "max_tokens": 50, "max_completion_tokens": 99,
        "stop": "END", "stream": True, "user": 42,
        "reasoning_effort": "minimal",
        "parallel_tool_calls": False,
        "response_format": {"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "object"}}},
        "messages": [
            {"role": "developer", "content": [{"type": "text", "text": "rules", "cache_control": {"type": "ephemeral"}}]},
            {"role": "user", "content": [
                {"type": "text", "text": "look"}, {"type": "text", "text": ""},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR"}},
                {"type": "image_url", "image_url": {"url": "https://img.test/cat.jpg"}},
                {"type": "image_url", "image_url": {}},
            ]},
            {"role": "user", "content": "and this"},  # consecutive user turns merge
            {"role": "assistant", "content": [{"type": "text", "text": "ok"}], "tool_calls": [
                {"type": "function", "function": {"name": "f", "arguments": "{bad"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": [{"type": "text", "text": "result"}]},
        ],
        "tools": [
            {"type": "function", "function": {"name": "f", "description": "does f", "strict": True}},
            {"type": "web_search"},  # not a function: dropped
        ],
    }
    out = openai_to_anthropic_request(body, 1000)
    assert out["max_tokens"] == 99 and out["stop_sequences"] == ["END"] and out["stream"] is True
    assert out["system"] == [{"type": "text", "text": "rules", "cache_control": {"type": "ephemeral"}}]
    user = out["messages"][0]
    assert user["role"] == "user" and [b["type"] for b in user["content"]] == ["text", "image", "image", "text"]
    assert user["content"][1]["source"] == {"type": "base64", "media_type": "image/png", "data": "iVBOR"}
    assert user["content"][2]["source"] == {"type": "url", "url": "https://img.test/cat.jpg"}
    tool_use = out["messages"][1]["content"][1]
    assert tool_use["type"] == "tool_use" and tool_use["id"].startswith("toolu_") and tool_use["input"] == {"arguments": "{bad"}
    assert out["messages"][2]["content"][0] == {"type": "tool_result", "tool_use_id": "c1",
                                                "content": [{"type": "text", "text": "result"}]}
    assert out["tools"] == [{"name": "f", "input_schema": {"type": "object", "properties": {}}, "description": "does f", "strict": True}]
    assert out["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
    assert out["output_config"] == {"effort": "low", "format": {"type": "json_schema", "schema": {"type": "object"}}}
    assert out["metadata"] == {"user_id": "42"} and "temperature" not in out


def test_openai_tool_choice_none_and_defaults():
    base = {"messages": [{"role": "user", "content": "x"}], "tools": [{"type": "function", "function": {"name": "f"}}]}
    assert openai_to_anthropic_request({**base, "tool_choice": "none"}, 10)["tool_choice"] == {"type": "none"}
    out = openai_to_anthropic_request(base, 10)
    assert "tool_choice" not in out and out["max_tokens"] == 10 and "output_config" not in out
    out = openai_to_anthropic_request({**base, "stop": ["a", "b"], "reasoning_effort": "bogus"}, 10)
    assert out["stop_sequences"] == ["a", "b"] and "output_config" not in out
    assert openai_to_anthropic_request({"messages": [{"role": "user", "content": None}]}, 1)["messages"] == []


# ------------------------------------------------------------------ Anthropic response -> OpenAI


@pytest.mark.parametrize("stop, finish", list(STOP_A2O.items()) + [(None, "stop"), ("brand_new_reason", "stop")])
def test_anthropic_stop_reasons(stop, finish):
    r = anthropic_to_openai_response({"content": [{"type": "text", "text": "x"}], "stop_reason": stop}, "m")
    assert r["choices"][0]["finish_reason"] == finish


def test_anthropic_response_thinking_tools_and_usage():
    resp = {"id": "abc", "stop_reason": "tool_use",
            "content": [{"type": "thinking", "thinking": "step 1", "signature": "sig"}, {"type": "thinking", "thinking": ""},
                        {"type": "redacted_thinking", "data": "xyz"},
                        {"type": "tool_use", "id": "toolu_1", "name": "f", "input": {"a": 1}}],
            "usage": {"input_tokens": 10, "cache_read_input_tokens": 90, "cache_creation_input_tokens": 5, "output_tokens": 7}}
    r = anthropic_to_openai_response(resp, "m")
    msg = r["choices"][0]["message"]
    assert r["id"] == "chatcmpl-abc" and msg["content"] is None and msg["reasoning_content"] == "step 1"
    assert msg["tool_calls"] == [{"id": "toolu_1", "type": "function", "function": {"name": "f", "arguments": '{"a":1}'}}]
    assert r["usage"] == {"prompt_tokens": 105, "completion_tokens": 7, "total_tokens": 112, "prompt_tokens_details": {"cached_tokens": 90},
                          "completion_tokens_details": {"reasoning_tokens": 0}}
    empty = anthropic_to_openai_response({}, "m")
    assert empty["choices"][0]["message"]["content"] == "" and empty["usage"]["total_tokens"] == 0


# ------------------------------------------------------------------ Anthropic stream -> OpenAI


def a_ev(d):
    return event(d, d.get("type"))


def test_anthropic_stream_text_start_thinking_unknowns_and_usage():
    s = AnthropicToOpenAIStream("m", include_usage=True)
    out = []
    for d in [
        {"type": "message_start", "message": {"usage": {"input_tokens": 3, "cache_read_input_tokens": 100}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "hmm"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "s"}},  # dropped
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": ""}},  # dropped
        {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": "Hel"}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "lo"}},
        {"type": "content_block_start", "index": 2, "content_block": {"type": "server_tool_use"}},  # no equivalent
        {"type": "ping"},
        {"type": "message_delta", "delta": {"stop_reason": "max_tokens"}, "usage": {"output_tokens": 9}},
        {"type": "message_stop"},
    ]:
        out += s.feed(a_ev(d))
    out += s.feed(SSEEvent(data=""))  # empty and non-JSON events are ignored
    out += s.feed(SSEEvent(data="garbage"))
    out += s.finish_events()  # already finished: nothing more
    p = payloads(out)
    deltas = [c["choices"][0]["delta"] for c in p if c.get("choices")]
    assert {"reasoning_content": "hmm"} in deltas and {"content": "Hel"} in deltas and {"content": "lo"} in deltas
    finals = [c for c in p if c.get("choices") and c["choices"][0]["finish_reason"]]
    assert [c["choices"][0]["finish_reason"] for c in finals] == ["length"]
    usage = [c for c in p if c.get("usage")][0]["usage"]
    assert usage["prompt_tokens"] == 103 and usage["completion_tokens"] == 9 and usage["prompt_tokens_details"]["cached_tokens"] == 100
    assert out[-1].data == "[DONE]" and sum(e.data == "[DONE]" for e in out) == 1


def test_anthropic_stream_parallel_tools_and_error():
    s = AnthropicToOpenAIStream("m", include_usage=False)
    out = []
    for d in [
        {"type": "message_start", "message": {}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "t1", "name": "a"}},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "t2", "name": "b"}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"y":'}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": '{"x":1}'}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "2}"}},
        {"type": "error", "error": {"type": "overloaded_error", "message": "busy"}},
    ]:
        out += s.feed(a_ev(d))
    p = payloads(out)
    calls = [c["choices"][0]["delta"]["tool_calls"][0] for c in p if c.get("choices") and "tool_calls" in c["choices"][0]["delta"]]
    args = {}
    for c in calls:
        args.setdefault(c["index"], "")
        args[c["index"]] += (c.get("function") or {}).get("arguments", "")
    assert args == {0: '{"x":1}', 1: '{"y":2}'}  # each Anthropic block keeps its own OpenAI index
    assert p[-1] == {"error": {"message": "busy", "type": "overloaded_error"}} and out[-1].data == "[DONE]"
    assert s.finish_events() == []  # the error ended the stream: no finish chunk saying it completed


# ------------------------------------------------------------------ Anthropic request -> OpenAI


def test_anthropic_request_to_openai_everything():
    body = {
        "model": "m", "max_tokens": 64, "stop_sequences": ["##"], "temperature": 0.3, "top_p": 0.9, "stream": True,
        "system": [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}],
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "late system"}]},
            {"role": "user", "content": [
                {"type": "text", "text": "see "},
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "AAA"}},
                {"type": "image", "source": {"type": "url", "url": "https://img.test/x.png"}},
                {"type": "document", "source": {"type": "text", "data": "doc body"}},
                {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "PDF"}},
            ]},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "private", "signature": "s"},
                {"type": "text", "text": "calling"},
                {"type": "tool_use", "id": "t1", "name": "f", "input": {"q": 1}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": [{"type": "text", "text": "bad input"}], "is_error": True},
                {"type": "tool_result", "tool_use_id": "t2", "content": "fine"},
            ]},
            {"role": "assistant", "content": [{"type": "thinking", "thinking": "only thinking"}]},  # nothing to send
            {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
        ],
        "tools": [{"name": "f", "description": "fn", "input_schema": {"type": "object"}},
                  {"type": "web_search_20260209", "name": "web_search"}],
        "tool_choice": {"type": "tool", "name": "f", "disable_parallel_tool_use": True},
        "output_config": {"format": {"type": "json_schema", "schema": {"type": "object"}}},
    }
    out = anthropic_to_openai_request(body, "max_completion_tokens")
    msgs = out["messages"]
    assert msgs[0] == {"role": "system", "content": "sys"} and msgs[1] == {"role": "system", "content": "late system"}
    user = msgs[2]["content"]
    assert [p["type"] for p in user] == ["text", "image_url", "image_url", "text", "file"]
    assert user[1]["image_url"]["url"] == "data:image/jpeg;base64,AAA" and user[3]["text"] == "doc body"
    assert user[4]["file"] == {"filename": "document.pdf", "file_data": "data:application/pdf;base64,PDF"}
    assert msgs[3] == {"role": "assistant", "content": "calling", "tool_calls": [
        {"id": "t1", "type": "function", "function": {"name": "f", "arguments": '{"q":1}'}}]}
    assert msgs[4] == {"role": "tool", "tool_call_id": "t1", "content": "Error: bad input"}
    assert msgs[5] == {"role": "tool", "tool_call_id": "t2", "content": "fine"}
    assert msgs[6] == {"role": "user", "content": "a\n\nb"} and len(msgs) == 7  # thinking-only turn dropped; text parts joined
    assert out["max_completion_tokens"] == 64 and "max_tokens" not in out
    assert out["stop"] == ["##"] and out["temperature"] == 0.3 and out["top_p"] == 0.9
    assert out["stream_options"] == {"include_usage": True}
    assert out["tools"] == [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}, "description": "fn"}}]
    assert out["tool_choice"] == {"type": "function", "function": {"name": "f"}} and out["parallel_tool_calls"] is False
    assert out["response_format"]["json_schema"]["schema"] == {"type": "object"}


@pytest.mark.parametrize("choice, expected", [({"type": "any"}, "required"), ({"type": "none"}, "none"),
                                              ({"type": "auto"}, None), (None, None)])
def test_anthropic_tool_choice(choice, expected):
    body = {"messages": [{"role": "user", "content": "x"}], "tools": [{"name": "f", "input_schema": {}}]}
    if choice:
        body["tool_choice"] = choice
    assert anthropic_to_openai_request(body).get("tool_choice") == expected


def test_server_tools_only_means_no_tools():
    out = anthropic_to_openai_request({"messages": [], "tools": [{"type": "bash_20250124", "name": "bash"}], "tool_choice": {"type": "any"}})
    assert "tools" not in out and "tool_choice" not in out


# ------------------------------------------------------------------ OpenAI response -> Anthropic


@pytest.mark.parametrize("finish, stop", list(STOP_O2A.items()) + [(None, "end_turn"), ("weird", "end_turn")])
def test_openai_finish_reasons(finish, stop):
    r = openai_to_anthropic_response({"choices": [{"message": {"content": "x"}, "finish_reason": finish}]}, "m")
    assert r["stop_reason"] == stop


def test_openai_response_tools_usage_and_empty():
    resp = {"id": "c1", "choices": [{"finish_reason": "tool_calls", "message": {"content": [{"type": "text", "text": "hi"}], "tool_calls": [
        {"function": {"name": "f", "arguments": '{"a":1}'}}, {"id": "x", "function": {"name": "g", "arguments": None}}]}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 5, "prompt_tokens_details": {"cached_tokens": 60, "cache_write_tokens": 10}}}
    r = openai_to_anthropic_response(resp, "m")
    assert r["id"] == "msg_c1" and r["content"][0] == {"type": "text", "text": "hi"}
    assert r["content"][1]["id"].startswith("toolu_") and r["content"][1]["input"] == {"a": 1} and r["content"][2]["input"] == {}
    assert r["usage"] == {"input_tokens": 30, "cache_read_input_tokens": 60, "cache_creation_input_tokens": 10, "output_tokens": 5}
    empty = openai_to_anthropic_response({}, "m")
    assert empty["content"] == [] and empty["stop_reason"] == "end_turn"


def test_usage_helpers_never_go_negative():
    assert usage_from_openai({"prompt_tokens": 5, "prompt_tokens_details": {"cached_tokens": 9}}).input_tokens == 0
    assert usage_from_openai(None).total_tokens == 0 and usage_from_anthropic(None).total_tokens == 0


def test_usage_tracker_both_formats():
    a = UsageTracker("anthropic")
    for d in ({"type": "message_start", "message": {"usage": {"input_tokens": 4}}}, {"type": "message_delta", "usage": {"output_tokens": 6}},
              {"type": "content_block_delta"}):
        a.feed(event(d))
    a.feed(SSEEvent(data="[DONE]"))
    assert (a.usage.input_tokens, a.usage.output_tokens) == (4, 6)
    o = UsageTracker("openai")
    o.feed(event({"choices": []}))
    o.feed(event({"choices": [], "usage": {"prompt_tokens": 8, "completion_tokens": 2}}))
    assert (o.usage.input_tokens, o.usage.output_tokens) == (8, 2)


# ------------------------------------------------------------------ OpenAI stream -> Anthropic


def o_ev(d):
    return event(d)


def test_openai_stream_text_then_tools_blocks_close_in_order():
    s = OpenAIToAnthropicStream("m")
    out = []
    for d in [
        {"choices": [{"delta": {"role": "assistant"}}]},
        {"choices": [{"delta": {"content": "Let me "}}]},
        {"choices": [{"delta": {"content": "check."}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "a", "arguments": ""}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"x":'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 1, "function": {"name": "b", "arguments": "{}"}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "1}"}}]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        {"choices": [], "usage": {"prompt_tokens": 50, "completion_tokens": 12, "prompt_tokens_details": {"cached_tokens": 40}}},
    ]:
        out += s.feed(o_ev(d))
    out += s.feed(SSEEvent(data="[DONE]"))
    out += s.feed(SSEEvent(data="[DONE]"))  # a second [DONE] adds nothing
    p = payloads(out)
    kinds = [x["type"] for x in p]
    assert kinds[0] == "message_start" and kinds[-2:] == ["message_delta", "message_stop"] and kinds.count("message_start") == 1
    starts = [x for x in p if x["type"] == "content_block_start"]
    assert [(x["index"], x["content_block"]["type"]) for x in starts] == [(0, "text"), (1, "tool_use"), (2, "tool_use")]
    assert starts[1]["content_block"]["id"] == "c1" and starts[2]["content_block"]["id"].startswith("toolu_")
    # Every opened block is closed before the next opens, and the last one before message_delta.
    opened = set()
    for x in p:
        if x["type"] == "content_block_start":
            assert not opened
            opened.add(x["index"])
        elif x["type"] == "content_block_stop":
            opened.discard(x["index"])
        elif x["type"] == "message_delta":
            assert not opened
    by_index = {}
    for x in p:
        if x["type"] == "content_block_delta" and x["delta"]["type"] == "input_json_delta":
            by_index.setdefault(x["index"], "")
            by_index[x["index"]] += x["delta"]["partial_json"]
    assert by_index == {1: '{"x":1}', 2: "{}"}
    delta = next(x for x in p if x["type"] == "message_delta")
    assert delta["delta"]["stop_reason"] == "tool_use"
    assert delta["usage"] == {"input_tokens": 10, "cache_read_input_tokens": 40, "cache_creation_input_tokens": 0, "output_tokens": 12}


def test_openai_stream_errors_empty_and_finish_without_content():
    s = OpenAIToAnthropicStream("m")
    first = s.feed(o_ev({"error": {"message": "rate limited", "code": 429}}))
    assert [x["type"] for x in payloads(first)] == ["message_start", "error"]
    assert payloads(first)[1]["error"]["message"] == "rate limited"
    assert s.feed(o_ev({"choices": [{"delta": {"content": "late"}}]})) == [] and s.finish_events() == []  # the error ended it
    plain = OpenAIToAnthropicStream("m")
    assert plain.feed(o_ev({"error": "plain string"}))[-1].json()["error"]["message"] == "plain string"
    assert plain.feed(SSEEvent(data="")) == [] and plain.feed(SSEEvent(data="nope")) == []

    quiet = OpenAIToAnthropicStream("m")
    fin = payloads(quiet.finish_events())  # upstream ended with no chunks at all
    assert [x["type"] for x in fin] == ["message_start", "message_delta", "message_stop"]
    assert fin[1]["delta"]["stop_reason"] == "end_turn"
    assert quiet.finish_events() == []


def test_openai_stream_length_finish_and_text_after_tool():
    s = OpenAIToAnthropicStream("m")
    out = []
    for d in [{"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c", "function": {"name": "f"}}]}}]},
              {"choices": [{"delta": {"content": "after"}, "finish_reason": "length"}]}]:
        out += s.feed(o_ev(d))
    out += s.finish_events()
    p = payloads(out)
    starts = [x["content_block"]["type"] for x in p if x["type"] == "content_block_start"]
    assert starts == ["tool_use", "text"]  # text after a tool call opens a new block
    assert next(x for x in p if x["type"] == "message_delta")["delta"]["stop_reason"] == "max_tokens"
    assert all(json.loads(e.data) for e in out)
