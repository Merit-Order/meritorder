"""Responses API conversion edge cases (Codex CLI and other stateless clients)."""

import json

import pytest
from conftest import auth

from meritorder.formats.responses import ChatToResponsesStream, ResponsesError, chat_to_responses, responses_to_chat
from meritorder.formats.sse import SSEEvent, event


def test_input_items_tools_and_options():
    body = {
        "model": "m", "instructions": "be terse", "max_output_tokens": 64, "temperature": 0.1, "top_p": 0.5,
        "parallel_tool_calls": False, "user": "u1", "prompt_cache_key": "k", "reasoning": {"effort": "high"},
        "stream": True,
        "input": [
            {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "dev rules"}]},
            {"role": "user", "content": [{"type": "input_text", "text": "see "},
                                         {"type": "input_image", "image_url": "https://img.test/a.png"},
                                         {"type": "input_image", "file_id": "file_1"},
                                         {"type": "input_image"}]},
            {"type": "reasoning", "summary": []},  # no Chat equivalent: dropped
            {"type": "function_call", "call_id": "c1", "name": "ls", "arguments": '{"p":"."}'},
            {"type": "function_call", "id": "c2", "name": "pwd"},
            {"type": "function_call_output", "call_id": "c1", "output": {"files": ["a"]}},
            {"type": "function_call_output", "call_id": "c2", "output": "/home"},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "done"}]},
        ],
        "tools": [{"type": "function", "name": "ls", "description": "list", "parameters": {"type": "object"}, "strict": True},
                  {"type": "web_search"}],
        "tool_choice": {"type": "function", "name": "ls"},
        "text": {"format": {"type": "json_schema", "name": "out", "schema": {"type": "object"}, "strict": True}},
    }
    out = responses_to_chat(body)
    m = out["messages"]
    assert m[0] == {"role": "system", "content": "be terse"} and m[1] == {"role": "system", "content": "dev rules"}
    assert [p["type"] for p in m[2]["content"]] == ["text", "image_url", "image_url"]
    assert m[2]["content"][2]["image_url"]["url"] == "file:file_1"
    assert m[3] == {"role": "assistant", "content": None, "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "ls", "arguments": '{"p":"."}'}},
        {"id": "c2", "type": "function", "function": {"name": "pwd", "arguments": "{}"}}]}
    assert m[4] == {"role": "tool", "tool_call_id": "c1", "content": json.dumps({"files": ["a"]})}
    assert m[5]["content"] == "/home" and m[6] == {"role": "assistant", "content": "done"}
    assert out["max_tokens"] == 64 and out["reasoning_effort"] == "high" and out["prompt_cache_key"] == "k"
    assert out["parallel_tool_calls"] is False and out["user"] == "u1" and out["temperature"] == 0.1
    assert out["tools"] == [{"type": "function", "function": {"name": "ls", "description": "list", "parameters": {"type": "object"},
                                                              "strict": True}}]
    assert out["tool_choice"] == {"type": "function", "function": {"name": "ls"}}
    assert out["response_format"]["json_schema"] == {"name": "out", "schema": {"type": "object"}, "strict": True}
    assert out["stream_options"] == {"include_usage": True}


def test_string_input_trailing_calls_and_simple_tool_choice():
    out = responses_to_chat({"input": "hi", "reasoning": {"effort": "turbo"}})
    assert out["messages"] == [{"role": "user", "content": "hi"}] and "reasoning_effort" not in out
    out = responses_to_chat({"input": [{"type": "function_call", "call_id": "c", "name": "f", "arguments": "{}"}],
                             "tools": [{"type": "function", "name": "f"}], "tool_choice": "required"})
    assert out["messages"][-1]["tool_calls"][0]["id"] == "c" and out["tool_choice"] == "required"
    with pytest.raises(ResponsesError, match="previous_response_id"):
        responses_to_chat({"previous_response_id": "resp_1", "input": "x"})


def test_chat_to_responses_tools_and_truncation():
    r = chat_to_responses({"choices": [{"finish_reason": "length", "message": {"content": "partial", "tool_calls": [
        {"function": {"name": "f"}}]}}], "usage": {"prompt_tokens": 10, "completion_tokens": 3,
                                                   "prompt_tokens_details": {"cached_tokens": 4}}}, "m")
    assert r["status"] == "incomplete" and r["incomplete_details"] == {"reason": "max_output_tokens"}
    assert [o["type"] for o in r["output"]] == ["message", "function_call"]
    assert r["output"][1]["arguments"] == "{}" and r["output"][1]["call_id"].startswith("call_")
    assert r["usage"]["input_tokens"] == 10 and r["usage"]["input_tokens_details"]["cached_tokens"] == 4
    empty = chat_to_responses({}, "m")
    assert empty["output"] == [] and empty["status"] == "completed"


def kinds(events):
    return [e.json()["type"] for e in events if e.json()]


def test_stream_text_tool_text_and_sequence_numbers():
    s = ChatToResponsesStream("m")
    out = []
    for d in [{"choices": [{"delta": {"content": "Look"}}]},
              {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "ls", "arguments": '{"a"'}}]}}]},
              {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": ":1}"}}]}}]},
              {"choices": [{"delta": {"content": "Then"}}]},
              {"choices": [{"delta": {}, "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 5, "completion_tokens": 2}}]:
        out += s.feed(event(d))
    out += s.feed(SSEEvent(data="[DONE]"))
    out += s.feed(SSEEvent(data="[DONE]"))
    k = kinds(out)
    assert k[:2] == ["response.created", "response.in_progress"] and k[-1] == "response.completed"
    assert k.count("response.output_item.added") == 3  # message, function call, second message
    seqs = [e.json()["sequence_number"] for e in out]
    assert seqs == list(range(1, len(seqs) + 1))
    final = out[-1].json()["response"]
    assert [o["type"] for o in final["output"]] == ["message", "function_call", "message"]
    assert final["output"][1]["arguments"] == '{"a":1}' and all(o["status"] == "completed" for o in final["output"])
    assert final["usage"]["total_tokens"] == 7


def test_stream_errors_truncation_and_empty_upstream():
    s = ChatToResponsesStream("m")
    out = s.feed(event({"error": {"type": "rate_limit_error", "message": "slow down"}}))
    # The error, then the response failed with it; nothing after it, and no response.completed at [DONE].
    assert kinds(out) == ["response.created", "response.in_progress", "error", "response.failed"]
    assert out[2].json()["code"] == "rate_limit_error" and out[-1].json()["response"]["error"]["message"] == "slow down"
    assert s.feed(event({"choices": [{"delta": {"content": "late"}}]})) == [] and s.feed(SSEEvent(data="[DONE]")) == []
    assert s.feed(SSEEvent(data="")) == []
    plain = ChatToResponsesStream("m")
    assert plain.feed(event({"error": "plain"}))[2].json()["message"] == "plain"

    t = ChatToResponsesStream("m")
    t.feed(event({"choices": [{"delta": {"content": "x"}, "finish_reason": "length"}]}))
    fin = t.finish_events()
    assert kinds(fin)[-1] == "response.incomplete" and fin[-1].json()["response"]["incomplete_details"]["reason"] == "max_output_tokens"

    quiet = ChatToResponsesStream("m")
    assert kinds(quiet.finish_events()) == ["response.created", "response.in_progress", "response.completed"]
    assert quiet.finish_events() == []


async def test_responses_endpoint_tool_call_round_trip_through_anthropic_backend(client):
    body = {"model": "big", "input": [{"role": "user", "content": "use the tool [tool:ls]"}],
            "tools": [{"type": "function", "name": "ls", "parameters": {"type": "object", "properties": {}}}]}
    r = await client.post("/v1/responses", json=body, headers=auth())
    assert r.status_code == 200 and r.json()["object"] == "response"
    r = await client.post("/v1/responses", json={"model": "big", "input": "x", "previous_response_id": "resp_1"}, headers=auth())
    assert r.status_code == 400 and "previous_response_id" in r.text
