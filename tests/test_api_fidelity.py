"""API fidelity across formats, checked against the official openai and anthropic SDKs: what each client gets back is what
its own provider would have sent (streams the SDKs accumulate, objects they validate), nothing a request carries is lost in
translation without a word, and the ledger records what the provider billed even when a stream breaks off."""

from __future__ import annotations

import copy
import json
import time

import httpx
import pytest
from anthropic._models import construct_type as a_construct
from anthropic.lib.streaming._messages import accumulate_event
from anthropic.types import Message as AMessage
from anthropic.types import RawMessageStreamEvent
from conftest import TEST_CONFIG, auth, last_row, ledger
from openai._models import construct_type as o_construct
from openai.lib.streaming.responses._responses import ResponseStreamState
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.responses import Response as OResponse
from openai.types.responses import ResponseStreamEvent
from test_http_backend import OPENAI_OK, Upstream, post, stream_response

from meritorder.app import build_gateway
from meritorder.config import build_config
from meritorder.formats.responses import FILTERED, ChatToResponsesStream, chat_to_responses, request_echo, responses_to_chat, responses_usage
from meritorder.formats.responses_upstream import chat_to_responses_request
from meritorder.formats.sse import SSEEvent, event
from meritorder.formats.translate import (
    AnthropicToOpenAIStream,
    OpenAIToAnthropicStream,
    TranslationError,
    UsageTracker,
    anthropic_to_openai_request,
    anthropic_to_openai_response,
    openai_to_anthropic_request,
    openai_to_anthropic_response,
    server_tools,
    text_of,
    usage_from_anthropic,
    usage_from_openai,
)
from meritorder.pricing import Usage
from meritorder.state import MemoryState
from meritorder.store import MemoryStore
from meritorder.tokens import estimate

PDF = "JVBERi0xLjQKJcOkw7zDtsOfCjIgMCBvYmoKPDwvTGVuZ3RoIDMgMCBSPj4Kc3RyZWFtCnic"  # a few base64 bytes stand in for a PDF
ECHO = {**request_echo({"model": "m", "input": "x"}), "store": False}


def chunk(delta: dict | None = None, finish: str | None = None, **extra) -> dict:
    return {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "m",
            "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}], **extra}


def chat_chunks(text="Hello", tool=None, finish="stop", tool_finish="tool_calls"):
    out = [chunk({"role": "assistant", "content": ""})]
    out += [chunk({"content": w + " "}) for w in text.split(" ") if w]
    if tool:
        out.append(chunk({"tool_calls": [{"index": 0, "id": "call_1", "type": "function", "function": {"name": tool, "arguments": ""}}]}))
        out.append(chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"q":1}'}}]}))
        finish = tool_finish
    out.append(chunk({}, finish))
    out.append({**chunk(), "choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 7, "total_tokens": 107,
                                                     "prompt_tokens_details": {"cached_tokens": 40}}})
    return [event(d) for d in out] + [SSEEvent(data="[DONE]")]


def accumulate_anthropic(events: list[SSEEvent]):
    snapshot = None
    for ev in events:
        snapshot = accumulate_event(event=a_construct(type_=RawMessageStreamEvent, value=ev.json()), current_snapshot=snapshot, json_bufs={})
    AMessage.model_validate(snapshot.model_dump())
    return snapshot


def responses_state(events: list[SSEEvent]) -> ResponseStreamState:
    from openai._types import omit
    state = ResponseStreamState(input_tools=omit, text_format=omit)
    for ev in events:
        state.handle_event(o_construct(type_=ResponseStreamEvent, value=ev.json()))
    return state


def feed_all(conv, events: list[SSEEvent]) -> list[SSEEvent]:
    out: list[SSEEvent] = []
    for ev in events:
        out += conv.feed(ev)
    return out


# ---------------------------------------------------------------- streams the SDKs accumulate, objects they validate


def test_openai_stream_as_anthropic_accumulates_in_the_sdk():
    snapshot = accumulate_anthropic(feed_all(OpenAIToAnthropicStream("m"), chat_chunks(tool="search")))
    assert [b.type for b in snapshot.content] == ["text", "tool_use"]
    assert snapshot.content[1].input == {"q": 1} and snapshot.stop_reason == "tool_use"
    assert snapshot.usage.input_tokens == 60 and snapshot.usage.cache_read_input_tokens == 40 and snapshot.usage.output_tokens == 7


def test_anthropic_stream_as_openai_chunks_validate_in_the_sdk():
    conv = AnthropicToOpenAIStream("m", include_usage=True)
    evs = [
        {"type": "message_start", "message": {"id": "msg_1", "usage": {"input_tokens": 10, "output_tokens": 1, "cache_read_input_tokens": 5}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "hmm"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "hi"}},
        {"type": "content_block_stop", "index": 1},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 3, "output_tokens_details": {"thinking_tokens": 2}}},
        {"type": "message_stop"},
    ]
    out = feed_all(conv, [event(d, d["type"]) for d in evs])
    chunks = [ChatCompletionChunk.model_validate(e.json()) for e in out if e.data != "[DONE]"]
    assert chunks[-1].usage.prompt_tokens == 15 and chunks[-1].usage.completion_tokens == 3
    assert chunks[-1].usage.completion_tokens_details.reasoning_tokens == 2  # Claude's thinking tokens, as OpenAI reports them
    assert out[-1].data == "[DONE]"


def test_chat_stream_as_responses_accumulates_in_the_sdk():
    conv = ChatToResponsesStream("m", extra=ECHO)
    final = responses_state(feed_all(conv, chat_chunks(tool="search")))._completed_response
    assert [o.type for o in final.output] == ["message", "function_call"] and final.output[1].arguments == '{"q":1}'
    OResponse.model_validate(json.loads(json.dumps(conv.final)))


def test_chat_stream_as_responses_custom_tool_accumulates_in_the_sdk():
    from meritorder.formats.responses_tools import ToolSet
    ts = ToolSet()
    ts.add({"type": "custom", "name": "apply_patch", "format": {"type": "grammar", "syntax": "lark", "definition": "start: x"}})
    conv = ChatToResponsesStream("m", extra=ECHO, tools=ts.registry())
    raw = json.dumps({"input": "*** Begin Patch\n+x\n*** End Patch\n"})
    events = [event(chunk({"tool_calls": [{"index": 0, "id": "call_1", "type": "function", "function": {"name": "apply_patch", "arguments": ""}}]}))]
    events += [event(chunk({"tool_calls": [{"index": 0, "function": {"arguments": raw[i:i + 7]}}]})) for i in range(0, len(raw), 7)]
    events += [event(chunk({}, "tool_calls")), SSEEvent(data="[DONE]")]
    final = responses_state(feed_all(conv, events))._completed_response
    assert final.output[0].type == "custom_tool_call" and final.output[0].input == "*** Begin Patch\n+x\n*** End Patch\n"


def test_translated_objects_validate_in_the_sdks():
    a = {"id": "msg_1", "type": "message", "role": "assistant", "model": "m", "stop_reason": "tool_use",
         "content": [{"type": "text", "text": "x"}, {"type": "tool_use", "id": "toolu_1", "name": "f", "input": {"a": 1}}],
         "usage": {"input_tokens": 10, "output_tokens": 2, "cache_read_input_tokens": 3, "cache_creation_input_tokens": 4}}
    ChatCompletion.model_validate(anthropic_to_openai_response(a, "m"))
    o = {"id": "c", "object": "chat.completion", "created": 1, "model": "m",
         "choices": [{"index": 0, "message": {"role": "assistant", "content": None, "tool_calls": [
             {"id": "call_1", "type": "function", "function": {"name": "f", "arguments": '{"a":1}'}}]}, "finish_reason": "tool_calls"}],
         "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}}
    AMessage.model_validate(openai_to_anthropic_response(o, "m"))
    OResponse.model_validate(chat_to_responses(o, "m", extra=ECHO))


def test_responses_usage_carries_cache_writes_and_reasoning_tokens():
    u = usage_from_openai({"prompt_tokens": 100, "completion_tokens": 30, "prompt_tokens_details": {"cached_tokens": 40, "cache_write_tokens": 10},
                           "completion_tokens_details": {"reasoning_tokens": 25}})
    assert u.reasoning_tokens == 25 and u.cache_write_tokens == 10
    r = responses_usage(u)
    assert r["input_tokens_details"] == {"cached_tokens": 40, "cache_write_tokens": 10} and r["output_tokens_details"] == {"reasoning_tokens": 25}
    assert usage_from_anthropic({"output_tokens": 9, "output_tokens_details": {"thinking_tokens": 6}}).reasoning_tokens == 6
    merged = Usage(reasoning_tokens=3)
    merged.merge(Usage(reasoning_tokens=8))
    assert merged.reasoning_tokens == 8


# ---------------------------------------------------------------- PDFs and files, in every direction


def test_pdf_document_from_anthropic_client_reaches_openai_backend_as_a_file():
    body = {"model": "m", "max_tokens": 10, "messages": [{"role": "user", "content": [
        {"type": "document", "title": "a.pdf", "source": {"type": "base64", "media_type": "application/pdf", "data": PDF}},
        {"type": "document", "source": {"type": "url", "url": "https://x.test/b.pdf"}},
        {"type": "text", "text": "Summarize these"}]}]}
    parts = anthropic_to_openai_request(body)["messages"][0]["content"]
    assert parts[0] == {"type": "file", "file": {"filename": "a.pdf", "file_data": f"data:application/pdf;base64,{PDF}"}}
    assert parts[1] == {"type": "file", "file": {"filename": "b.pdf", "file_data": "https://x.test/b.pdf"}}
    # The same PDF counts the same either way, so routing and budgets see one size.
    assert estimate(body).media == estimate({"messages": [{"role": "user", "content": parts}]}).media


def test_pdf_file_from_openai_client_reaches_anthropic_backend_as_a_document():
    body = {"model": "m", "messages": [{"role": "user", "content": [
        {"type": "file", "file": {"filename": "a.pdf", "file_data": f"data:application/pdf;base64,{PDF}"}},
        {"type": "file", "file": {"filename": "notes.txt", "file_data": "data:text/plain;base64,aGVsbG8="}},
        {"type": "file", "file": {"filename": "c.pdf", "file_data": "https://x.test/c.pdf"}},
        {"type": "file", "file": {"filename": "d.pdf", "file_data": PDF}},  # bare base64, named as a PDF
        {"type": "text", "text": "Summarize this"}]}]}
    blocks = openai_to_anthropic_request(body, 100)["messages"][0]["content"]
    assert blocks[0] == {"type": "document", "title": "a.pdf", "source": {"type": "base64", "media_type": "application/pdf", "data": PDF}}
    assert blocks[1]["source"] == {"type": "text", "media_type": "text/plain", "data": "hello"}
    assert blocks[2]["source"] == {"type": "url", "url": "https://x.test/c.pdf"}
    assert blocks[3]["source"]["data"] == PDF and blocks[4] == {"type": "text", "text": "Summarize this"}


@pytest.mark.parametrize("file, says", [({"file_id": "file-abc"}, "uploaded to OpenAI"),
                                        ({"filename": "x.png", "file_data": "data:image/png;base64,AAAA"}, "not image/png"),
                                        ({"filename": "x.txt", "file_data": "data:text/plain;base64,@@@"}, "base64 UTF-8")])
def test_a_file_claude_cannot_read_is_refused_not_dropped(file, says):
    with pytest.raises(TranslationError, match=says):
        openai_to_anthropic_request({"messages": [{"role": "user", "content": [{"type": "file", "file": file}]}]}, 100)


@pytest.mark.parametrize("block", [{"type": "document", "source": {"type": "file", "file_id": "file_1"}},
                                   {"type": "image", "source": {"type": "file", "file_id": "file_2"}},
                                   {"type": "document", "source": {"type": "weird"}}])
def test_an_anthropic_file_another_provider_cannot_open_is_refused(block):
    with pytest.raises(TranslationError):
        anthropic_to_openai_request({"messages": [{"role": "user", "content": [block]}]})


def test_document_with_content_source_becomes_its_parts():
    body = {"messages": [{"role": "user", "content": [
        {"type": "document", "source": {"type": "content", "content": [{"type": "text", "text": "one"}, {"type": "text", "text": "two"}]}},
        {"type": "document", "source": {"type": "content", "content": "three"}}]}]}
    assert anthropic_to_openai_request(body)["messages"][0]["content"] == "one\n\ntwo\n\nthree"


def test_pdf_input_file_on_responses_api_reaches_chat_and_claude_and_responses_backends():
    body = {"model": "m", "input": [{"role": "user", "content": [
        {"type": "input_file", "filename": "a.pdf", "file_data": f"data:application/pdf;base64,{PDF}"},
        {"type": "input_file", "filename": "b.pdf", "file_url": "https://x.test/b.pdf"},
        {"type": "input_text", "text": "Summarize this"}]}]}
    chat = responses_to_chat(body)
    parts = chat["messages"][0]["content"]
    assert parts[0] == {"type": "file", "file": {"filename": "a.pdf", "file_data": f"data:application/pdf;base64,{PDF}"}}
    assert parts[1] == {"type": "file", "file": {"filename": "b.pdf", "file_data": "https://x.test/b.pdf"}}
    claude = openai_to_anthropic_request(chat, 100)["messages"][0]["content"]
    assert claude[0]["source"]["data"] == PDF and claude[1]["source"] == {"type": "url", "url": "https://x.test/b.pdf"}
    native = chat_to_responses_request(chat)["input"][0]["content"]
    assert native[0] == {"type": "input_file", "file_data": f"data:application/pdf;base64,{PDF}", "filename": "a.pdf"}
    assert native[1] == {"type": "input_file", "file_url": "https://x.test/b.pdf", "filename": "b.pdf"}
    assert chat_to_responses_request({"messages": [{"role": "user", "content": [{"type": "file", "file": {"file_id": "file-1"}}]}]}
                                     )["input"][0]["content"] == [{"type": "input_file", "file_id": "file-1"}]
    # Counted alike in every shape: an input_file, the file part it becomes, and the document Claude gets.
    one = {"type": "input_file", "file_data": f"data:application/pdf;base64,{PDF}"}
    assert estimate({"messages": [{"role": "user", "content": [one]}]}).media == estimate({"messages": [{"role": "user", "content": [parts[0]]}]}).media
    assert estimate({"messages": [{"role": "user", "content": [claude[0]]}]}).media == estimate({"messages": [{"role": "user", "content": [one]}]}).media


def test_files_returned_by_a_responses_tool_follow_the_tool_messages():
    body = {"model": "m", "input": [
        {"type": "function_call", "call_id": "c1", "name": "read", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": [
            {"type": "input_text", "text": "here"}, {"type": "input_file", "filename": "r.pdf", "file_data": "data:application/pdf;base64,QQ=="}]}]}
    msgs = responses_to_chat(body)["messages"]
    assert msgs[1] == {"role": "tool", "tool_call_id": "c1", "content": "here\n\n(1 attachment from this result follows below.)"}
    assert msgs[2]["content"][0]["text"] == "Attachments returned by tool call c1:" and msgs[2]["content"][1]["type"] == "file"


async def test_a_file_the_backend_cannot_take_is_a_400_for_the_client(gateway):
    principal = await gateway.authn.authenticate(auth())
    body = {"model": "big", "messages": [{"role": "user", "content": [{"type": "file", "file": {"file_id": "file-abc"}}]}]}
    r = await gateway.respond(principal, body, auth(), "openai", time.monotonic())
    assert r.status_code == 400 and "file-abc" in json.loads(r.body)["error"]["message"]


# ---------------------------------------------------------------- tool results with images, forced tools, server tools


def test_image_in_tool_result_from_anthropic_client_follows_the_tool_messages():
    body = {"model": "m", "max_tokens": 10, "messages": [
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "shot.png"}},
                                          {"type": "tool_use", "id": "t2", "name": "Read", "input": {"file_path": "b.txt"}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": [
                {"type": "text", "text": "a screenshot"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="}}]},
            {"type": "tool_result", "tool_use_id": "t2", "content": "plain", "is_error": True},
            {"type": "text", "text": "what now?"}]}]}
    msgs = anthropic_to_openai_request(body)["messages"]
    assert [m["role"] for m in msgs] == ["assistant", "tool", "tool", "user", "user"]
    assert msgs[1]["content"] == "a screenshot\n\n(1 image from this result follows below.)" and msgs[2]["content"] == "Error: plain"
    assert msgs[3]["content"] == [{"type": "text", "text": "Images returned by tool call t1:"},
                                  {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}]
    assert msgs[4] == {"role": "user", "content": "what now?"}


def test_tool_result_text_parts_keep_their_marker_where_markers_are_kept():
    hour = {"type": "ephemeral", "ttl": "1h"}
    body = {"messages": [{"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t1", "content": ["stray", {"type": "text", "text": "a"}, {"type": "text", "text": "b", "cache_control": hour}]},
        {"type": "search_result", "source": "x", "title": "t", "content": []}]}]}  # nothing Chat Completions has
    assert anthropic_to_openai_request(body, keep_cache_control=True)["messages"] == [
        {"role": "tool", "tool_call_id": "t1", "content": [{"type": "text", "text": "a\nb", "cache_control": hour}]}]
    assert anthropic_to_openai_request(body)["messages"] == [{"role": "tool", "tool_call_id": "t1", "content": "a\nb"}]


def test_forced_tool_choice_from_openai_client_is_kept_on_claude():
    body = {"model": "m", "messages": [{"role": "user", "content": "extract"}],
            "tools": [{"type": "function", "function": {"name": "extract", "parameters": {"type": "object"}}}]}
    assert openai_to_anthropic_request({**body, "tool_choice": "required"}, 100)["tool_choice"] == {"type": "any"}
    named = {**body, "tool_choice": {"type": "function", "function": {"name": "extract"}}, "parallel_tool_calls": False}
    out = openai_to_anthropic_request(named, 100, "claude-opus-4-8")
    assert out["tool_choice"] == {"type": "tool", "name": "extract", "disable_parallel_tool_use": True} and "system" not in out


@pytest.mark.parametrize("model", ["claude-opus-5-5", "anthropic.claude-sonnet-5-5", "claude-fable-5-1", "claude-mythos-5-2"])
def test_forced_tool_choice_becomes_an_instruction_on_claude_models_that_refuse_it(model):
    body = {"model": "m", "messages": [{"role": "system", "content": "be brief"}, {"role": "user", "content": "extract"}],
            "tools": [{"type": "function", "function": {"name": "extract", "parameters": {"type": "object"}}}],
            "tool_choice": {"type": "function", "function": {"name": "extract"}}, "parallel_tool_calls": False}
    out = openai_to_anthropic_request(body, 100, model)
    assert out["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
    assert out["system"][0] == {"type": "text", "text": "be brief"} and "call the extract tool" in out["system"][-1]["text"]
    anyone = openai_to_anthropic_request({**body, "tool_choice": "required", "messages": body["messages"][1:]}, 100, model)
    assert "call one of the tools" in anyone["system"][-1]["text"] and anyone["tool_choice"]["type"] == "auto"


@pytest.mark.parametrize("model", ["claude-opus-5", "claude-haiku-5-5", "claude-sonnet-5", "claude-opus-4-8"])
def test_claude_models_that_take_forced_tools_get_them(model):
    body = {"messages": [{"role": "user", "content": "x"}], "tools": [{"type": "function", "function": {"name": "f"}}], "tool_choice": "required"}
    assert openai_to_anthropic_request(body, 100, model)["tool_choice"] == {"type": "any"}


async def test_anthropic_server_tools_dropped_on_an_openai_backend_are_named(monkeypatch, caplog):
    up = Upstream(httpx.Response(200, json=OPENAI_OK))
    gw = fake_gateway(monkeypatch, chat=up)
    body = {"model": "kimi", "max_tokens": 20, "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "web_search_20260209", "name": "web_search"}, {"name": "f", "input_schema": {"type": "object"}}]}
    with caplog.at_level("INFO", logger="meritorder.proxy"):
        r = await post(gw, body, path="/v1/messages")
    assert r.status_code == 200 and r.headers["x-meritorder-dropped-tools"] == "web_search"
    assert "goes without web_search (an Anthropic server tool)" in caplog.text
    assert [t["function"]["name"] for t in json.loads(up.requests[0].content)["tools"]] == ["f"]
    assert server_tools({"tools": [{"name": "f", "input_schema": {}}, {"type": "custom", "name": "g", "input_schema": {}}]}) == []


# ---------------------------------------------------------------- refusals and stop reasons


def test_refusal_from_a_chat_backend_is_a_refusal_part_in_the_responses_object():
    o = {"id": "c", "object": "chat.completion", "created": 1, "model": "m",
         "choices": [{"index": 0, "message": {"role": "assistant", "content": None, "refusal": "I can't help with that."}, "finish_reason": "stop"}],
         "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}}
    r = chat_to_responses(o, "m", extra=ECHO)
    assert r["output"][0]["content"] == [{"type": "refusal", "refusal": "I can't help with that."}] and r["status"] == "completed"
    OResponse.model_validate(r)
    filtered = chat_to_responses({**o, "choices": [{"index": 0, "message": {"role": "assistant", "content": None}, "finish_reason": "content_filter"}]},
                                 "m", extra=ECHO)
    assert filtered["output"][0]["content"] == [{"type": "refusal", "refusal": FILTERED}]
    assert filtered["status"] == "incomplete" and filtered["incomplete_details"] == {"reason": "content_filter"}
    OResponse.model_validate(filtered)


def test_refusal_streams_as_refusal_events_the_sdk_accumulates():
    conv = ChatToResponsesStream("m", extra=ECHO)
    out = feed_all(conv, [event(chunk({"role": "assistant", "content": "Sorry, "})), event(chunk({"refusal": "I can't "})),
                          event(chunk({"refusal": "help."})), event(chunk({}, "stop")), SSEEvent(data="[DONE]")])
    kinds = [e.event for e in out]
    assert kinds.count("response.refusal.delta") == 2 and "response.refusal.done" in kinds
    final = responses_state(out)._completed_response
    assert [p.type for p in final.output[0].content] == ["output_text", "refusal"]
    assert final.output[0].content[1].refusal == "I can't help." and final.output[0].content[0].text == "Sorry, "


def test_content_filter_with_nothing_said_streams_a_refusal():
    conv = ChatToResponsesStream("m", extra=ECHO)
    out = feed_all(conv, [event(chunk({}, "content_filter")), SSEEvent(data="[DONE]")])
    assert [e.event for e in out][-1] == "response.incomplete"
    assert conv.final["output"][0]["content"] == [{"type": "refusal", "refusal": FILTERED}]
    assert conv.final["incomplete_details"] == {"reason": "content_filter"}
    OResponse.model_validate(conv.final)


def test_openai_refusal_reaches_anthropic_client_as_a_refusal():
    o = {"choices": [{"message": {"content": None, "refusal": "No."}, "finish_reason": "stop"}]}
    r = openai_to_anthropic_response(o, "m")
    assert r["content"] == [{"type": "text", "text": "No."}] and r["stop_reason"] == "refusal"
    snapshot = accumulate_anthropic(feed_all(OpenAIToAnthropicStream("m"), [event(chunk({"refusal": "No."})), event(chunk({}, "stop")),
                                                                              SSEEvent(data="[DONE]")]))
    assert snapshot.stop_reason == "refusal" and snapshot.content[0].text == "No."


def test_tool_calls_ending_with_stop_still_say_tool_use():
    snapshot = accumulate_anthropic(feed_all(OpenAIToAnthropicStream("m"), chat_chunks(text="", tool="search", tool_finish="stop")))
    assert snapshot.stop_reason == "tool_use"
    o = {"choices": [{"message": {"tool_calls": [{"id": "c", "function": {"name": "f", "arguments": "{}"}}]}, "finish_reason": "stop"}]}
    assert openai_to_anthropic_response(o, "m")["stop_reason"] == "tool_use"
    assert openai_to_anthropic_response({**o, "choices": [{**o["choices"][0], "finish_reason": "length"}]}, "m")["stop_reason"] == "max_tokens"


def test_system_blocks_keep_apart_when_joined():
    body = {"system": [{"type": "text", "text": "You are Claude."}, {"type": "text", "text": "Be brief."}], "messages": []}
    assert anthropic_to_openai_request(body)["messages"][0] == {"role": "system", "content": "You are Claude.\n\nBe brief."}
    assert text_of([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "ab"  # elsewhere unchanged


# ---------------------------------------------------------------- errors mid-stream end the stream as each API does


def test_openai_error_chunk_ends_the_anthropic_stream_with_the_error():
    conv = OpenAIToAnthropicStream("m")
    out = conv.feed(event(chunk({"content": "par"})))
    out += conv.feed(event({"error": {"message": "overloaded", "type": "server_error", "code": 503}}))
    out += conv.feed(SSEEvent(data="[DONE]")) + conv.finish_events()
    kinds = [e.event for e in out]
    assert kinds[-1] == "error" and "message_stop" not in kinds and "message_delta" not in kinds


async def test_claude_overload_after_content_reaches_openai_client_as_an_error_and_a_failure(monkeypatch):
    up = Upstream(lambda req: stream_response(sse(
        {"type": "message_start", "message": {"id": "msg_s", "usage": {"input_tokens": 900, "output_tokens": 0}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Working"}},
        {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}})))
    gw = fake_gateway(monkeypatch, claude=up)
    r = await post(gw, {"model": "claude", "stream": True, "messages": [{"role": "user", "content": "hi"}]})
    evs = parse_sse(r.content)
    datas = [json.loads(d) for _, d in evs if d != "[DONE]"]
    i = next(i for i, d in enumerate(datas) if "error" in d)
    assert not any(d.get("choices") and d["choices"][0].get("finish_reason") for d in datas[i + 1:])
    assert evs[-1][1] == "[DONE]"
    row = (await ledger(gw))[-1]
    assert row["status"] >= 400 and row["input_tokens"] == 900  # the provider's own count, not an estimate
    assert "usage" not in (row["tags"] or {})


async def test_openai_error_after_content_ends_the_anthropic_clients_stream(monkeypatch):
    body = "".join(f"data: {json.dumps(d)}\n\n" for d in [
        chunk({"role": "assistant", "content": "Wor"}), {"error": {"message": "upstream provider error", "type": "server_error", "code": 502}},
    ]) + "data: [DONE]\n\n"
    gw = fake_gateway(monkeypatch, chat=Upstream(lambda req: stream_response(body.encode())))
    r = await post(gw, {"model": "kimi", "max_tokens": 50, "stream": True, "messages": [{"role": "user", "content": "hi"}]}, path="/v1/messages")
    names = [n for n, _ in parse_sse(r.content)]
    assert names[-1] == "error" and "message_stop" not in names


async def test_chat_error_after_content_fails_the_responses_stream(monkeypatch):
    body = "".join(f"data: {json.dumps(d)}\n\n" for d in [chunk({"role": "assistant", "content": "Wor"}),
                                                          {"error": {"message": "boom", "type": "server_error"}}]) + "data: [DONE]\n\n"
    gw = fake_gateway(monkeypatch, chat=Upstream(lambda req: stream_response(body.encode())))
    r = await post(gw, {"model": "kimi", "stream": True, "input": "hi"}, path="/v1/responses")
    names = [n for n, _ in parse_sse(r.content)]
    assert names[-2:] == ["error", "response.failed"] and "response.completed" not in names


# ---------------------------------------------------------------- the ledger when a stream breaks off


async def disconnect_after(gateway, body: dict, fmt: str, chunks: int) -> dict:
    principal = await gateway.authn.authenticate(auth())
    resp = await gateway.respond(principal, body, {**auth(), "x-session-id": f"dc-{fmt}"}, fmt, time.monotonic())
    it = resp.body_iterator
    got = 0
    async for _ in it:
        got += 1
        if got >= chunks:
            break
    await it.aclose()  # the client went away
    return await last_row(gateway)


async def test_disconnect_on_openai_backend_records_the_estimated_prompt_and_output(gateway):
    body = {"model": "small", "stream": True, "messages": [{"role": "user", "content": "tell me a long story " * 50}]}
    row = await disconnect_after(gateway, body, "openai", 3)
    assert row["status"] == 499 and row["tags"]["usage"] == "estimated"
    # The provider billed the whole prompt and what it wrote: the ledger mustn't say the request was free.
    assert row["input_tokens"] >= estimate(body).tokens * 0.5 and row["output_tokens"] > 0 and row["actual_cost"] > 0
    # Budgets count it: the spend counters moved by what the row says it cost.
    rows = await ledger(gateway)
    assert await gateway.spend("user", "alice") == pytest.approx(sum(r["actual_cost"] for r in rows), abs=1e-6) and row["actual_cost"] > 0


async def test_disconnect_on_anthropic_backend_counts_the_streamed_output(gateway):
    body = {"model": "big", "max_tokens": 200, "stream": True, "messages": [{"role": "user", "content": "tell me a long story " * 50}]}
    row = await disconnect_after(gateway, body, "anthropic", 6)
    assert row["status"] == 499 and row["tags"]["usage"] == "estimated"
    assert row["input_tokens"] + row["cache_read_tokens"] + row["cache_write_tokens"] > 0
    assert row["output_tokens"] > 1  # several words streamed before the disconnect, not message_start's 1


async def test_stream_that_breaks_is_estimated_and_a_count_the_provider_sent_is_kept(monkeypatch):
    parts = [f"data: {json.dumps(chunk({'content': 'word ' * 40}))}\n\n".encode()]
    gw = fake_gateway(monkeypatch, chat=Upstream(lambda req: stream_response(*parts, die=httpx.ReadError("connection reset"))))
    r = await post(gw, {"model": "kimi", "stream": True, "messages": [{"role": "user", "content": "x " * 400}]})
    assert r.status_code == 200
    row = (await ledger(gw))[-1]
    assert row["status"] == 502 and row["tags"]["usage"] == "estimated" and row["input_tokens"] > 100 and row["output_tokens"] >= 30
    assert row["actual_cost"] > 0 and row["baseline_cost"] >= row["actual_cost"]  # savings see the estimate too

    # Usage that came before the break is the provider's: it stands, and nothing is added on top of it.
    reported = [f"data: {json.dumps(chunk({'content': 'word ' * 40}))}\n\n".encode(),
                f"data: {json.dumps({**chunk(), 'choices': [], 'usage': {'prompt_tokens': 77, 'completion_tokens': 5}})}\n\n".encode()]
    gw = fake_gateway(monkeypatch, chat=Upstream(lambda req: stream_response(*reported, die=httpx.ReadError("reset"))))
    await post(gw, {"model": "kimi", "stream": True, "messages": [{"role": "user", "content": "x " * 400}]})
    row = (await ledger(gw))[-1]
    assert row["status"] == 502 and (row["input_tokens"], row["output_tokens"]) == (77, 5) and "usage" not in (row["tags"] or {})


def test_tracker_keeps_what_the_model_wrote_in_either_format():
    t = UsageTracker("anthropic")
    for d in ({"type": "content_block_start", "content_block": {"type": "tool_use", "name": "Read"}},
              {"type": "content_block_delta", "delta": {"type": "input_json_delta", "partial_json": '{"a":'}},
              {"type": "content_block_delta", "delta": {"type": "thinking_delta", "thinking": "hm"}},
              {"type": "message_delta", "usage": {"output_tokens": 4}}):
        t.feed(event(d, d["type"]))
    assert t.streamed == 'Read{"a":hm' and t.output_reported
    o = UsageTracker("openai")
    o.feed(event(chunk({"reasoning_content": "r", "tool_calls": [{"function": {"name": "f", "arguments": "{}"}}]}, usage={"prompt_tokens": 9})))
    assert o.streamed == "rf{}" and not o.output_reported and o.usage.input_tokens == 9


# ---------------------------------------------------------------- count_tokens


async def test_count_tokens_refuses_an_unknown_model_like_messages_does(client):
    body = {"model": "auto", "messages": [{"role": "user", "content": "hello " * 100}]}
    r = await client.post("/v1/messages/count_tokens", headers=auth(), json=body)
    assert r.status_code == 200 and r.json()["input_tokens"] > 0
    r = await client.post("/v1/messages/count_tokens", headers=auth(), json={**body, "model": "no-such-model"})
    assert r.status_code == 404 and r.json()["error"]["type"] == "not_found_error"


# ---------------------------------------------------------------- a gateway with fake providers


def fake_gateway(monkeypatch, claude=None, chat=None):
    monkeypatch.setenv("MO_TEST_KEY", "k")
    key = {"type": "api_key", "value_env": "MO_TEST_KEY"}
    raw = copy.deepcopy(TEST_CONFIG)
    raw["backends"] = {
        "claude": {"kind": "api", "format": "anthropic", "base_url": "https://api.anthropic.example", "auth": key},
        "kimi": {"kind": "api", "format": "openai", "base_url": "https://api.moonshot.example/v1", "auth": key},
    }
    raw["models"] = {
        "claude": {"quality": {"coding": 5, "general": 5}, "routes": [{"backend": "claude", "model": "claude-x", "price": {"input": 3, "output": 15}}]},
        "kimi": {"quality": {"coding": 4, "general": 4}, "routes": [{"backend": "kimi", "model": "kimi-x", "price": {"input": 1, "output": 4}}]},
    }
    raw["baseline_model"] = "claude"
    gw = build_gateway(build_config(raw, quiet=True), store=MemoryStore(), state=MemoryState())
    for name, up in (("claude", claude), ("kimi", chat)):
        if up is not None:
            gw.backends[name].client = httpx.AsyncClient(transport=httpx.MockTransport(up))
    return gw


def sse(*events: dict) -> bytes:
    return "".join(f"event: {d['type']}\ndata: {json.dumps(d)}\n\n" for d in events).encode()


def parse_sse(content: bytes) -> list[tuple[str | None, str]]:
    out = []
    for block in content.decode().strip().split("\n\n"):
        name, data = None, []
        for line in block.split("\n"):
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                data.append(line[6:])
        out.append((name, "\n".join(data)))
    return out
