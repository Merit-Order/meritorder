"""Codex CLI's tools on any backend (decision 57): custom (freeform) tools such as apply_patch, namespaced tools, tool
search, local shell and hosted tools, translated for Chat Completions and Claude backends and back, over HTTP and the
WebSocket transport, streaming or not. The request bodies are shaped like codex-rs's (main of 15 November 2026). Every
backend is mocked."""

from __future__ import annotations

import copy
import json

import httpx
import pytest
from conftest import TEST_CONFIG, ledger
from test_http_backend import Upstream, post, stream_response
from test_responses_ws import connect

from meritorder.app import build_gateway
from meritorder.config import build_config
from meritorder.formats.responses import ChatToResponsesStream, chat_to_responses, responses_to_chat
from meritorder.formats.responses_tools import (TOOLS_KEY, InputStream, ToolSet, chat_name, custom_input, dropped_tools, fit_upstream,
                                                local_shell_action, place_hosted, tool_output)
from meritorder.formats.responses_upstream import ResponsesToChatStream, chat_to_responses_request, responses_to_chat_completion
from meritorder.formats.sse import event
from meritorder.formats.translate import AnthropicToOpenAIStream, openai_to_anthropic_request
from meritorder.state import MemoryState
from meritorder.store import MemoryStore

# codex-rs/core/assets/tools/apply_patch.lark
LARK = """start: begin_patch hunk+ end_patch
begin_patch: "*** Begin Patch" LF
end_patch: "*** End Patch" LF?

hunk: add_hunk | delete_hunk | update_hunk
add_hunk: "*** Add File: " filename LF add_line+
delete_hunk: "*** Delete File: " filename LF
update_hunk: "*** Update File: " filename LF change_move? change?

filename: /(.+)/
add_line: "+" /(.*)/ LF -> line

change_move: "*** Move to: " filename LF
change: (change_context | change_line)+ eof_line?
change_context: ("@@" | "@@ " /(.+)/) LF
change_line: ("+" | "-" | " ") /(.*)/ LF
eof_line: "*** End of File" LF

%import common.LF
"""
APPLY_PATCH = {"type": "custom", "name": "apply_patch",
               "description": "The `apply_patch` tool can be used to edit files. This is a FREEFORM tool, so do not wrap the patch in JSON.",
               "format": {"type": "grammar", "syntax": "lark", "definition": LARK}}
EXEC_COMMAND = {"type": "function", "name": "exec_command", "strict": False,
                "description": "Runs a command in a PTY, returning output or a session ID for ongoing interaction.",
                "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}, "workdir": {"type": "string"},
                                                                "yield_time_ms": {"type": "number"}},
                               "required": ["cmd"], "additionalProperties": False}}
VIEW_IMAGE = {"type": "function", "name": "view_image", "strict": False,
              "description": "View a local image file from the filesystem when visual inspection is needed.",
              "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"], "additionalProperties": False}}
COLLABORATION = {"type": "namespace", "name": "collaboration", "description": "Tools for spawning and managing sub-agents.",
                 "tools": [{"type": "function", "name": "spawn_agent", "strict": False, "description": "Spawn a sub-agent.",
                            "parameters": {"type": "object", "properties": {"task": {"type": "string"}}, "required": ["task"]}}]}
MCP = {"type": "namespace", "name": "mcp__github__", "description": "Tools in the mcp__github__ namespace.",
       "tools": [{"type": "function", "name": "create_issue", "strict": False, "defer_loading": True,
                  "parameters": {"type": "object", "properties": {"title": {"type": "string"}}}}]}
TOOL_SEARCH = {"type": "tool_search", "execution": "client", "description": "# Tool discovery\n\nSearches over deferred tool metadata.",
               "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "number"}}, "required": ["query"]}}
WEB_SEARCH = {"type": "web_search", "external_web_access": False}  # Codex's default: OpenAI's cached index only
LIVE_SEARCH = {"type": "web_search", "external_web_access": True, "filters": {"allowed_domains": ["docs.python.org"]},
               "user_location": {"type": "approximate", "country": "US", "city": "Austin"}}
TOOLS = [EXEC_COMMAND, APPLY_PATCH, VIEW_IMAGE, COLLABORATION, MCP, TOOL_SEARCH, WEB_SEARCH]

PATCH = '*** Begin Patch\n*** Update File: app.py\n@@\n-print("hi")\n+print("hello, \\u00e9 é \U0001f600")\n*** End Patch\n'


def dev(text: str) -> dict:
    return {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": text}]}


def user(text: str) -> dict:
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}


def codex_body(input_items: list, tools=TOOLS, model: str = "auto", **kw) -> dict:
    """A turn as codex-rs builds it for a custom provider (ResponsesApiRequest): base instructions as developer input."""
    return {"model": model, "input": input_items, "tools": tools, "tool_choice": "auto", "parallel_tool_calls": True,
            "reasoning": {"effort": "medium"}, "store": False, "stream": True, "include": ["reasoning.encrypted_content"],
            "prompt_cache_key": "019a-codex-session", "text": {"verbosity": "low"}, **kw}


FIRST = [dev("You are Codex, a coding agent."), user("<environment_context>cwd=/repo</environment_context>"), user("Fix the greeting.")]
CALL = {"type": "custom_tool_call", "id": "ctc_1", "status": "completed", "call_id": "call_patch", "name": "apply_patch", "input": PATCH}
RESULT = {"type": "custom_tool_call_output", "call_id": "call_patch", "output": "Success. Updated the following files:\nM app.py\n"}


# ---------------------------------------------------------------- requests


def test_codex_tools_become_function_tools():
    chat = responses_to_chat(codex_body(FIRST))
    tools = {t["function"]["name"]: t["function"] for t in chat["tools"]}
    assert list(tools) == ["exec_command", "apply_patch", "view_image", "collaboration__spawn_agent", "mcp__github__create_issue", "tool_search"]
    patch = tools["apply_patch"]
    assert patch["parameters"]["required"] == ["input"] and patch["parameters"]["properties"]["input"]["type"] == "string"
    assert patch["description"].startswith(APPLY_PATCH["description"]) and LARK.strip() in patch["description"] and "(lark)" in patch["description"]
    assert tools["exec_command"] == {k: EXEC_COMMAND[k] for k in ("name", "description", "parameters", "strict")}
    assert tools["collaboration__spawn_agent"]["description"].startswith("Tools for spawning and managing sub-agents.")
    assert "defer_loading" not in json.dumps(chat["tools"])  # every backend gets every tool up front
    reg = chat[TOOLS_KEY]
    assert reg["hosted"] == [WEB_SEARCH] and reg["native"] == TOOLS
    assert reg["entries"]["apply_patch"] == {"kind": "custom", "name": "apply_patch", "namespace": None}
    assert reg["entries"]["mcp__github__create_issue"] == {"kind": "function", "name": "create_issue", "namespace": "mcp__github__"}
    assert "exec_command" not in reg["entries"]
    # Only plain functions: nothing to carry, and the body is what it was before custom tools were translated.
    assert TOOLS_KEY not in responses_to_chat(codex_body(FIRST, tools=[EXEC_COMMAND]))
    named = responses_to_chat(codex_body(FIRST, tool_choice={"type": "custom", "name": "apply_patch"}))
    assert named["tool_choice"] == {"type": "function", "function": {"name": "apply_patch"}}


def test_chat_names():
    assert chat_name(None, "x.y") == "x.y" and chat_name("functions", "exec") == "exec"
    assert chat_name("mcp__fs__", "read") == "mcp__fs__read" and chat_name("my.server", "do it") == "my_server__do_it"
    long = chat_name("n" * 40, "t" * 40)
    assert len(long) == 64 and long != chat_name("n" * 40, "t" * 41)


def test_history_items_convert_both_ways():
    turn = codex_body([
        *FIRST, CALL, RESULT,
        {"type": "function_call", "call_id": "call_spawn", "namespace": "collaboration", "name": "spawn_agent", "arguments": '{"task":"t"}'},
        {"type": "function_call_output", "call_id": "call_spawn", "output": "spawned"},
        {"type": "local_shell_call", "id": "lsh_1", "call_id": "call_sh", "status": "completed",
         "action": {"type": "exec", "command": ["ls"], "timeout_ms": None, "working_directory": None, "env": None, "user": None}},
        {"type": "function_call_output", "call_id": "call_sh", "output": "app.py"},
        {"type": "tool_search_call", "call_id": "call_ts", "status": "completed", "execution": "client", "arguments": {"query": "calendar"}},
        {"type": "tool_search_output", "call_id": "call_ts", "status": "completed", "execution": "client",
         "tools": [{"type": "function", "name": "create_event", "parameters": {"type": "object", "properties": {}}}]},
        {"type": "web_search_call", "id": "ws_1", "status": "completed", "action": {"type": "search", "query": "x"}},
        {"type": "reasoning", "summary": [], "encrypted_content": "gAAA"},
        user("Thanks."),
    ])
    chat = responses_to_chat(turn)
    msgs = chat["messages"]
    calls = [m["tool_calls"][0] for m in msgs if m.get("tool_calls")]
    assert [c["function"]["name"] for c in calls] == ["apply_patch", "collaboration__spawn_agent", "local_shell", "tool_search"]
    assert json.loads(calls[0]["function"]["arguments"]) == {"input": PATCH} and calls[0]["id"] == "call_patch"
    assert json.loads(calls[2]["function"]["arguments"])["command"] == ["ls"]
    tool_msgs = [m for m in msgs if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["call_patch", "call_spawn", "call_sh", "call_ts"]
    assert tool_msgs[0]["content"] == RESULT["output"]
    assert "create_event" in [t["function"]["name"] for t in chat["tools"]]  # found by the search: callable from here on
    assert "create_event" not in json.dumps(chat[TOOLS_KEY]["native"])  # a Responses backend gets it with the search's output
    assert msgs[-1] == {"role": "user", "content": "Thanks."}

    # Back again, for a backend that speaks the Responses API: the same items, as the API takes them.
    native = chat_to_responses_request(chat)
    kinds = [i["type"] for i in native["input"]]
    assert kinds[-9:] == ["custom_tool_call", "custom_tool_call_output", "function_call", "function_call_output", "local_shell_call",
                          "function_call_output", "tool_search_call", "tool_search_output", "message"]
    ctc = native["input"][kinds.index("custom_tool_call")]
    assert ctc == {"type": "custom_tool_call", "call_id": "call_patch", "name": "apply_patch", "input": PATCH}
    spawn = native["input"][kinds.index("function_call")]
    assert spawn["namespace"] == "collaboration" and spawn["name"] == "spawn_agent"
    out = native["input"][kinds.index("tool_search_output")]
    assert out["tools"][0]["name"] == "create_event"
    assert native["tools"] == [{k: v for k, v in t.items() if v is not None} for t in TOOLS]


def test_tool_outputs_with_images_follow_the_results():
    image = [{"type": "input_text", "text": "the screenshot"}, {"type": "input_image", "image_url": "data:image/png;base64,AA", "detail": "high"}]
    chat = responses_to_chat(codex_body([
        *FIRST,
        {"type": "function_call", "call_id": "c1", "name": "view_image", "arguments": '{"path":"a.png"}'},
        {"type": "function_call", "call_id": "c2", "name": "exec_command", "arguments": '{"cmd":"ls"}'},
        {"type": "function_call_output", "call_id": "c1", "output": image},
        {"type": "function_call_output", "call_id": "c2", "output": "a.png"},
        {"type": "function_call", "call_id": "c3", "name": "exec_command", "arguments": '{"cmd":"pwd"}'},
    ], tools=[EXEC_COMMAND, VIEW_IMAGE]))
    roles = [m["role"] for m in chat["messages"]]
    assert roles[-5:] == ["assistant", "tool", "tool", "user", "assistant"]
    first = chat["messages"][-4]
    assert first["content"].startswith("the screenshot") and "1 image" in first["content"]
    pic = chat["messages"][-2]["content"]
    assert pic[1] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA", "detail": "high"}}
    # On Claude the tool results come first in their turn, then the image.
    claude = openai_to_anthropic_request(chat, 1024)["messages"][-2]["content"]
    assert [b["type"] for b in claude] == ["tool_result", "tool_result", "text", "image"]
    assert tool_output({"exit": 0}, "x") == ('{"exit": 0}', None)
    assert tool_output([{"type": "input_text", "text": "a"}, {"type": "input_text", "text": "b"}, "junk"], "x") == ("a\nb", None)
    assert tool_output([{"type": "input_image", "file_id": "f1"}], "x")[1]["content"][1]["image_url"]["url"] == "file:f1"


def test_additional_tools_items_are_tools_not_messages():
    """Codex's Responses lite mode (GPT-5.6 and GPT-6 in its catalog): the tools as the first input item."""
    lite = codex_body([{"type": "additional_tools", "role": "developer", "tools": [
        {"type": "namespace", "name": "functions", "description": "", "tools": [EXEC_COMMAND, APPLY_PATCH]}, TOOL_SEARCH]}, *FIRST,
        {"type": "additional_tools", "role": "developer", "tools": [
            {"type": "namespace", "name": "functions", "description": "", "tools": [{**APPLY_PATCH, "description": "Edit files."}]}]}],
        tools=[WEB_SEARCH], model="gpt-5.6-terra")
    chat = responses_to_chat(lite)
    assert [m["role"] for m in chat["messages"]] == ["system", "user", "user"]
    names = [t["function"]["name"] for t in chat["tools"]]
    assert names == ["exec_command", "apply_patch", "tool_search"]  # the functions namespace keeps plain names
    assert chat["tools"][1]["function"]["description"].startswith("Edit files.")  # the later definition wins
    assert [t["type"] for t in chat[TOOLS_KEY]["native"]] == ["web_search", "namespace", "tool_search"]


# ---------------------------------------------------------------- replies


def test_reply_calls_become_the_clients_items():
    reg = responses_to_chat(codex_body(FIRST, tools=[*TOOLS, {"type": "local_shell"}]))[TOOLS_KEY]
    reply = {"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "content": None, "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "apply_patch", "arguments": json.dumps({"input": PATCH})}},
        {"id": "c2", "type": "function", "function": {"name": "collaboration__spawn_agent", "arguments": '{"task":"x"}'}},
        {"id": "c3", "type": "function", "function": {"name": "local_shell", "arguments": '{"command":"ls"}'}},
        {"id": "c4", "type": "function", "function": {"name": "tool_search", "arguments": '{"query":"cal"}'}},
        {"id": "c5", "type": "function", "function": {"name": "exec_command", "arguments": '{"cmd":"ls"}'}}]}}]}
    out = chat_to_responses(reply, "auto", tools=reg)["output"]
    assert [o["type"] for o in out] == ["custom_tool_call", "function_call", "local_shell_call", "tool_search_call", "function_call"]
    assert out[0]["input"] == PATCH and out[0]["call_id"] == "c1" and out[0]["id"].startswith("ctc_") and "namespace" not in out[0]
    assert out[1]["name"] == "spawn_agent" and out[1]["namespace"] == "collaboration"
    assert out[2]["action"] == {"type": "exec", "command": ["ls"], "timeout_ms": None, "working_directory": None, "env": None, "user": None}
    assert out[3]["arguments"] == {"query": "cal"} and out[3]["execution"] == "client"
    assert out[4]["name"] == "exec_command" and "namespace" not in out[4]
    # Without the registry (a request with plain functions only), calls stay function calls.
    assert chat_to_responses(reply, "auto")["output"][0]["type"] == "function_call"


def test_custom_input_from_whatever_the_model_wrote():
    assert custom_input(json.dumps({"input": "x"})) == "x"
    assert custom_input(json.dumps({"patch": "y"})) == "y"  # one string under another name
    assert custom_input("*** Begin Patch\n") == "*** Begin Patch\n"  # the raw text, not JSON
    assert custom_input('"quoted"') == "quoted"
    assert custom_input('{"a": 1, "b": 2}') == '{"a": 1, "b": 2}'


@pytest.mark.parametrize("size", [1, 2, 3, 5, 7, 64])
def test_input_streams_as_text_whatever_the_chunking(size):
    raw = json.dumps({"input": PATCH})  # ASCII escapes, a surrogate pair among them
    dec = InputStream()
    text = "".join(dec.feed(raw[i:i + size]) for i in range(0, len(raw), size))
    assert text == PATCH and dec.state == "end"
    plain = json.dumps({"input": PATCH}, ensure_ascii=False)
    dec = InputStream()
    assert "".join(dec.feed(plain[i:i + size]) for i in range(0, len(plain), size)) == PATCH
    other = InputStream()
    assert other.feed('{"patch": "x"}') == "" and other.state == "off"


def chat_chunks(*args: str, name: str = "apply_patch") -> list:
    first = {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "call_9", "type": "function",
                                                                  "function": {"name": name, "arguments": ""}}]}}]}
    rest = [{"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": a}}]}}]} for a in args]
    end = {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 5, "completion_tokens": 3}}
    return [event(d) for d in [first, *rest, end]]


def test_stream_of_a_custom_tool_call():
    reg = responses_to_chat(codex_body(FIRST))[TOOLS_KEY]
    conv = ChatToResponsesStream("auto", tools=reg)
    raw = json.dumps({"input": PATCH})
    evs = [e.json() for ch in chat_chunks(raw[:9], raw[9:30], raw[30:]) for e in conv.feed(ch)] + [e.json() for e in conv.finish_events()]
    kinds = [e["type"] for e in evs]
    added = evs[kinds.index("response.output_item.added")]["item"]
    assert added["type"] == "custom_tool_call" and added["input"] == "" and added["status"] == "in_progress"
    assert "".join(e["delta"] for e in evs if e["type"] == "response.custom_tool_call_input.delta") == PATCH
    assert evs[kinds.index("response.custom_tool_call_input.done")]["input"] == PATCH
    done = evs[kinds.index("response.output_item.done")]["item"]
    assert done == {"type": "custom_tool_call", "id": added["id"], "call_id": "call_9", "name": "apply_patch", "input": PATCH, "status": "completed"}
    assert evs[-1]["type"] == "response.completed" and evs[-1]["response"]["output"] == [done]
    assert "response.function_call_arguments.delta" not in kinds

    # Arguments of another shape: nothing streams as text, and the input comes whole at the end.
    conv = ChatToResponsesStream("auto", tools=reg)
    evs = [e.json() for ch in chat_chunks('{"patch": ', json.dumps(PATCH) + "}") for e in conv.feed(ch)] + [e.json() for e in conv.finish_events()]
    assert [e["delta"] for e in evs if e["type"] == "response.custom_tool_call_input.delta"] == [PATCH]

    # A local shell call and a namespaced function stream as whole items.
    conv = ChatToResponsesStream("auto", tools=responses_to_chat(codex_body(FIRST, tools=[{"type": "local_shell"}, COLLABORATION]))[TOOLS_KEY])
    evs = [e.json() for ch in chat_chunks('{"command":["ls"]}', name="local_shell") for e in conv.feed(ch)] + [e.json() for e in conv.finish_events()]
    done = [e["item"] for e in evs if e["type"] == "response.output_item.done"]
    assert done[0]["type"] == "local_shell_call" and done[0]["action"]["command"] == ["ls"]
    conv = ChatToResponsesStream("auto", tools=responses_to_chat(codex_body(FIRST, tools=[COLLABORATION]))[TOOLS_KEY])
    evs = [e.json() for ch in chat_chunks('{"task":"a"}', name="collaboration__spawn_agent") for e in conv.feed(ch)] + [e.json() for e in conv.finish_events()]
    assert [e["item"] for e in evs if e["type"] == "response.output_item.done"][0]["namespace"] == "collaboration"
    assert "".join(e["delta"] for e in evs if e["type"] == "response.function_call_arguments.delta") == '{"task":"a"}'


def test_responses_backend_replies_with_custom_calls():
    resp = {"status": "completed", "output": [
        {"type": "custom_tool_call", "id": "ctc_1", "call_id": "c1", "name": "apply_patch", "input": PATCH},
        {"type": "function_call", "id": "fc_1", "call_id": "c2", "namespace": "collaboration", "name": "spawn_agent", "arguments": "{}"},
        {"type": "web_search_call", "id": "ws_1", "status": "completed"}]}
    calls = responses_to_chat_completion(resp)["choices"][0]["message"]["tool_calls"]
    assert [c["function"]["name"] for c in calls] == ["apply_patch", "collaboration__spawn_agent"]
    assert json.loads(calls[0]["function"]["arguments"]) == {"input": PATCH}

    conv = ResponsesToChatStream()
    out = []
    for d in [{"type": "response.created", "response": {"id": "r", "model": "gpt", "created_at": 1}},
              {"type": "response.output_item.added", "output_index": 0, "item": {"type": "custom_tool_call", "id": "ctc_1", "call_id": "c1",
                                                                                 "name": "apply_patch", "input": ""}},
              {"type": "response.custom_tool_call_input.delta", "item_id": "ctc_1", "delta": PATCH[:20]},
              {"type": "response.custom_tool_call_input.delta", "item_id": "ctc_1", "delta": PATCH[20:30]},
              {"type": "response.output_item.done", "output_index": 0, "item": {"type": "custom_tool_call", "id": "ctc_1", "call_id": "c1",
                                                                                "name": "apply_patch", "input": PATCH}},
              {"type": "response.output_item.added", "output_index": 1, "item": {"type": "local_shell_call", "id": "lsh_1", "call_id": "c3"}},
              {"type": "response.output_item.done", "output_index": 1, "item": {"type": "local_shell_call", "id": "lsh_1", "call_id": "c3",
                                                                                "action": {"type": "exec", "command": ["ls"]}}},
              {"type": "response.output_item.done", "output_index": 2, "item": {"type": "local_shell_call", "id": "unknown"}},
              {"type": "response.custom_tool_call_input.delta", "item_id": "unknown", "delta": "x"},
              {"type": "response.completed", "response": {"status": "completed"}}]:
        out += conv.feed(event(d))
    chunks = [e.json() for e in out if e.data != "[DONE]"]
    args: dict[int, str] = {}
    names: dict[int, str] = {}
    for c in chunks:
        for tc in (c["choices"][0]["delta"].get("tool_calls") or []) if c["choices"] else []:
            args[tc["index"]] = args.get(tc["index"], "") + (tc["function"].get("arguments") or "")
            names.setdefault(tc["index"], tc["function"].get("name"))
    assert names == {0: "apply_patch", 1: "local_shell"}
    assert json.loads(args[0]) == {"input": PATCH} and json.loads(args[1]) == {"command": ["ls"]}
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


# ---------------------------------------------------------------- hosted tools


def test_hosted_tools_per_backend():
    reg = {"hosted": [WEB_SEARCH, LIVE_SEARCH, {"type": "image_generation"}]}
    assert place_hosted(reg, "openai", "responses") == ([], [])
    added, dropped = place_hosted(reg, "anthropic", base_url="https://api.anthropic.com")
    assert added == [{"type": "web_search_20250305", "name": "web_search", "allowed_domains": ["docs.python.org"],
                      "user_location": {"type": "approximate", "city": "Austin", "country": "US"}}]
    assert dropped == [("web_search", "cached-only web search runs at OpenAI alone"),
                       ("image_generation", "runs at OpenAI, and this backend has no equivalent")]
    assert [n for n, _ in place_hosted(reg, "anthropic", base_url="https://bedrock-mantle.us-east-1.api.aws")[1]] == [
        "web_search", "web_search", "image_generation"]  # Bedrock has no web search
    assert place_hosted({"hosted": [LIVE_SEARCH]}, "anthropic", names={"web_search"})[1][0][1] == "a function tool is already named web_search"
    assert [n for n, _ in place_hosted(reg, "openai")[1]] == ["web_search", "web_search", "image_generation"]
    assert place_hosted(None, "openai") == ([], []) and dropped_tools({}, "openai") == []

    body = responses_to_chat(codex_body(FIRST, tools=[EXEC_COMMAND, LIVE_SEARCH]))
    up = openai_to_anthropic_request(body, 1024)
    assert fit_upstream(up, body, "anthropic") == [] and up["tools"][-1]["type"] == "web_search_20250305"
    chat = dict(body)
    assert [n for n, _ in fit_upstream(chat, body, "openai")] == ["web_search"] and TOOLS_KEY not in chat
    kept = dict(body)
    assert fit_upstream(kept, body, "openai", "responses") == [] and TOOLS_KEY in kept
    assert fit_upstream({}, {"messages": []}, "openai") == []


def test_claude_server_tool_input_never_reaches_the_client():
    conv = AnthropicToOpenAIStream("m", include_usage=False)
    evs = [{"type": "message_start", "message": {"usage": {"input_tokens": 1}}},
           {"type": "content_block_start", "index": 0, "content_block": {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search"}},
           {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": '{"query":"x"}'}},
           {"type": "content_block_start", "index": 1, "content_block": {"type": "web_search_tool_result", "content": []}},
           {"type": "content_block_start", "index": 2, "content_block": {"type": "tool_use", "id": "t1", "name": "exec_command"}},
           {"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": "{}"}}]
    out = [c.json() for e in evs for c in conv.feed(event(e))]
    calls = [tc for c in out for tc in c["choices"][0]["delta"].get("tool_calls") or []]
    assert [tc.get("function", {}).get("arguments") for tc in calls] == ["", "{}"] and calls[0]["function"]["name"] == "exec_command"


# ---------------------------------------------------------------- through the gateway


def gateway(monkeypatch, claude: Upstream | None = None, chat: Upstream | None = None, oai: Upstream | None = None):
    monkeypatch.setenv("MO_TEST_KEY", "k")
    key = {"type": "api_key", "value_env": "MO_TEST_KEY"}
    raw = copy.deepcopy(TEST_CONFIG)
    raw["backends"] = {
        "claude": {"kind": "api", "format": "anthropic", "base_url": "https://api.anthropic.example", "auth": key},
        "kimi": {"kind": "api", "format": "openai", "base_url": "https://api.moonshot.example/v1", "auth": key},
        "oai": {"kind": "api", "format": "openai", "base_url": "https://api.openai.example/v1", "api": "responses",
                "max_tokens_field": "max_completion_tokens", "auth": key},
    }
    raw["models"] = {
        "claude": {"quality": {"coding": 5, "general": 5}, "routes": [{"backend": "claude", "model": "claude-x", "price": {"input": 3, "output": 15}}]},
        "kimi": {"quality": {"coding": 4, "general": 4}, "routes": [{"backend": "kimi", "model": "kimi-x", "price": {"input": 1, "output": 4}}]},
        "gpt": {"quality": {"coding": 5, "general": 5}, "routes": [{"backend": "oai", "model": "gpt-x", "price": {"input": 1, "output": 8}}]},
    }
    raw["baseline_model"] = "claude"
    gw = build_gateway(build_config(raw, quiet=True), store=MemoryStore(), state=MemoryState())
    for name, up in (("claude", claude), ("kimi", chat), ("oai", oai)):
        if up is not None:
            gw.backends[name].client = httpx.AsyncClient(transport=httpx.MockTransport(up))
    return gw


CLAUDE_CALL = {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-x", "stop_reason": "tool_use",
               "content": [{"type": "text", "text": "Patching."}, {"type": "tool_use", "id": "toolu_1", "name": "apply_patch", "input": {"input": PATCH}}],
               "usage": {"input_tokens": 900, "output_tokens": 40}}
CLAUDE_DONE = {"id": "msg_2", "type": "message", "role": "assistant", "model": "claude-x", "stop_reason": "end_turn",
               "content": [{"type": "text", "text": "Done."}], "usage": {"input_tokens": 950, "output_tokens": 3, "cache_read_input_tokens": 800}}


def sse(*events: dict) -> bytes:
    return "".join(f"event: {d['type']}\ndata: {json.dumps(d)}\n\n" for d in events).encode()


def claude_stream(patch: str) -> bytes:
    raw = json.dumps({"input": patch})
    return sse({"type": "message_start", "message": {"id": "msg_s", "usage": {"input_tokens": 900, "output_tokens": 0}}},
               {"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "toolu_9", "name": "apply_patch", "input": {}}},
               *[{"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": raw[i:i + 11]}}
                 for i in range(0, len(raw), 11)],
               {"type": "content_block_stop", "index": 0},
               {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 40}},
               {"type": "message_stop"})


def parse_sse(content: bytes) -> list[dict]:
    return [json.loads(line[6:]) for line in content.decode().splitlines() if line.startswith("data: ") and line[6:] != "[DONE]"]


def unmarked(value):
    """A request without its cache markers, which move to the newest turn each time by design."""
    if isinstance(value, dict):
        return {k: unmarked(v) for k, v in value.items() if k != "cache_control"}
    if isinstance(value, list):
        return [unmarked(v) for v in value]
    return value


async def test_apply_patch_round_trip_on_claude_over_http(monkeypatch):
    up = Upstream(httpx.Response(200, json=CLAUDE_CALL), httpx.Response(200, json=CLAUDE_DONE))
    gw = gateway(monkeypatch, claude=up)
    r = await post(gw, codex_body(FIRST, model="claude", stream=False), path="/v1/responses", **{"session-id": "codex-1"})
    assert r.status_code == 200, r.text
    assert r.headers["x-meritorder-dropped-tools"] == "web_search"  # cached-only search: no equivalent on Claude
    out = r.json()["output"]
    assert [o["type"] for o in out] == ["message", "custom_tool_call"]
    call = out[1]
    assert call["name"] == "apply_patch" and call["input"] == PATCH and call["call_id"] == "toolu_1"
    sent = up.body(0)
    assert TOOLS_KEY not in sent and "web_search" not in json.dumps(sent["tools"])
    names = [t["name"] for t in sent["tools"]]
    assert names == ["exec_command", "apply_patch", "view_image", "collaboration__spawn_agent", "mcp__github__create_issue", "tool_search"]
    assert sent["tools"][1]["input_schema"]["required"] == ["input"]

    # Codex runs the patch and sends the next turn: everything again, with the call and its result.
    second = [*FIRST, {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Patching."}]}, call,
              {"type": "custom_tool_call_output", "call_id": call["call_id"], "output": "Success."}]
    r2 = await post(gw, codex_body(second, model="claude", stream=False), path="/v1/responses", **{"session-id": "codex-1"})
    assert r2.status_code == 200 and r2.json()["output"][0]["content"][0]["text"] == "Done."
    sent2 = up.body(1)
    turn = sent2["messages"][-2:]
    assert turn[0]["content"][-1] == {"type": "tool_use", "id": "toolu_1", "name": "apply_patch", "input": {"input": PATCH}}
    assert turn[1]["content"][0]["type"] == "tool_result" and turn[1]["content"][0]["tool_use_id"] == "toolu_1"
    # The prefix the cache reads is the same bytes: tools, system, and the first turn's messages.
    assert json.dumps(unmarked(sent2["tools"])) == json.dumps(unmarked(sent["tools"]))
    assert json.dumps(unmarked(sent2["system"])) == json.dumps(unmarked(sent["system"]))
    assert json.dumps(unmarked(sent2["messages"][:len(sent["messages"])])) == json.dumps(unmarked(sent["messages"]))
    rows = await ledger(gw)
    assert len(rows) == 2 and rows[1]["cache_read_tokens"] == 800


async def test_live_web_search_is_claudes_own(monkeypatch):
    up = Upstream(httpx.Response(200, json={**CLAUDE_DONE, "usage": {"input_tokens": 10, "output_tokens": 3, "server_tool_use": {"web_search_requests": 2}},
                                            "content": [{"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {"query": "py"}},
                                                        {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1", "content": []},
                                                        {"type": "text", "text": "Python 3.14 is out."}]}))
    gw = gateway(monkeypatch, claude=up)
    r = await post(gw, codex_body(FIRST, tools=[EXEC_COMMAND, LIVE_SEARCH], model="claude", stream=False), path="/v1/responses")
    assert r.status_code == 200 and "x-meritorder-dropped-tools" not in r.headers
    assert r.json()["output"][0]["content"][0]["text"] == "Python 3.14 is out."
    assert up.body()["tools"][-1] == {"type": "web_search_20250305", "name": "web_search", "allowed_domains": ["docs.python.org"],
                                      "user_location": {"type": "approximate", "city": "Austin", "country": "US"}}
    rows = await ledger(gw)
    assert rows[0]["actual_cost"] > (10 * 3 + 3 * 15) / 1e6  # two searches at $10 per 1,000 on top of tokens


async def test_apply_patch_streams_from_claude(monkeypatch):
    up = Upstream(lambda req: stream_response(claude_stream(PATCH)))
    gw = gateway(monkeypatch, claude=up)
    r = await post(gw, codex_body(FIRST, model="claude"), path="/v1/responses")
    assert r.status_code == 200 and r.headers["x-meritorder-dropped-tools"] == "web_search"
    evs = parse_sse(r.content)
    kinds = [e["type"] for e in evs]
    assert "".join(e["delta"] for e in evs if e["type"] == "response.custom_tool_call_input.delta") == PATCH
    done = evs[kinds.index("response.output_item.done")]["item"]
    assert done["type"] == "custom_tool_call" and done["input"] == PATCH and done["call_id"] == "toolu_9"
    assert kinds[-1] == "response.completed"


async def test_apply_patch_streams_from_a_chat_completions_backend(monkeypatch):
    raw = json.dumps({"input": PATCH})
    body = "".join(f"data: {json.dumps(e.json())}\n\n" for e in chat_chunks(*[raw[i:i + 13] for i in range(0, len(raw), 13)])) + "data: [DONE]\n\n"
    up = Upstream(lambda req: stream_response(body.encode()))
    gw = gateway(monkeypatch, chat=up)
    r = await post(gw, codex_body(FIRST, model="kimi"), path="/v1/responses")
    assert r.status_code == 200 and r.headers["x-meritorder-dropped-tools"] == "web_search"
    sent = up.body()
    assert TOOLS_KEY not in sent and [t["function"]["name"] for t in sent["tools"]][1] == "apply_patch"
    evs = parse_sse(r.content)
    done = [e["item"] for e in evs if e["type"] == "response.output_item.done"]
    assert done == [{"type": "custom_tool_call", "id": done[0]["id"], "call_id": "call_9", "name": "apply_patch", "input": PATCH, "status": "completed"}]


async def test_responses_backend_gets_the_tools_as_they_came(monkeypatch):
    up = Upstream(httpx.Response(200, json={"id": "resp_1", "status": "completed", "model": "gpt-x", "output": [
        {"type": "custom_tool_call", "id": "ctc_up", "call_id": "call_up", "name": "apply_patch", "input": PATCH}],
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}}))
    gw = gateway(monkeypatch, oai=up)
    turn = [*FIRST, CALL, RESULT]
    r = await post(gw, codex_body(turn, model="gpt", stream=False), path="/v1/responses")
    assert r.status_code == 200 and "x-meritorder-dropped-tools" not in r.headers
    sent = up.body()
    assert sent["tools"] == TOOLS and TOOLS_KEY not in sent
    assert {"type": "custom_tool_call", "call_id": "call_patch", "name": "apply_patch", "input": PATCH} in sent["input"]
    assert {"type": "custom_tool_call_output", "call_id": "call_patch", "output": RESULT["output"]} in sent["input"]
    out = r.json()["output"]
    assert out == [{"type": "custom_tool_call", "id": out[0]["id"], "call_id": "call_up", "name": "apply_patch", "input": PATCH, "status": "completed"}]


async def test_codex_session_over_websocket_on_claude(monkeypatch):
    """Two Codex turns on one connection: the second chains with previous_response_id and sends only the patch's result."""
    monkeypatch.setenv("MERITORDER_RESPONSES_KEY", "test-responses-key")
    up = Upstream(lambda req: stream_response(claude_stream(PATCH)), lambda req: stream_response(sse(
        {"type": "message_start", "message": {"id": "msg_t", "usage": {"input_tokens": 950, "cache_read_input_tokens": 800}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Done."}},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 2}}, {"type": "message_stop"})))
    gw = gateway(monkeypatch, claude=up)
    ws = await connect(gw, **{"session-id": "codex-ws"}).open()
    body = codex_body(FIRST, model="claude")
    first = await ws.turn(**body)
    assert first[-1]["type"] == "response.completed", first[-1]
    call = [e["item"] for e in first if e["type"] == "response.output_item.done"][0]
    assert call["type"] == "custom_tool_call" and call["input"] == PATCH
    second = await ws.turn(**{**body, "previous_response_id": first[-1]["response"]["id"],
                              "input": [{"type": "custom_tool_call_output", "call_id": call["call_id"], "output": "Success."}]})
    assert second[-1]["type"] == "response.completed" and second[-1]["response"]["output"][0]["content"][0]["text"] == "Done."
    await ws.close()
    a, b = up.body(0), up.body(1)
    assert json.dumps(unmarked(a["tools"])) == json.dumps(unmarked(b["tools"]))
    assert json.dumps(unmarked(b["messages"][:len(a["messages"])])) == json.dumps(unmarked(a["messages"]))
    assert b["messages"][-2]["content"] == [{"type": "tool_use", "id": "toolu_9", "name": "apply_patch", "input": {"input": PATCH}}]
    assert b["messages"][-1]["content"] == [{"type": "tool_result", "tool_use_id": "toolu_9", "content": "Success."}]


def test_odd_shapes_dont_break_the_translation():
    tools = ToolSet()
    tools.add("junk")
    tools.add({"type": "namespace", "name": "n", "tools": ["junk", {"type": "file_search"}]})
    assert tools.tools() == [] and tools.registry() is None
    assert local_shell_action("not json")["command"] == [] and local_shell_action('{"command":"ls"}')["command"] == ["ls"]
    chat = responses_to_chat(codex_body(FIRST, tool_choice={"type": "custom", "name": "apply_patch"}))
    chat["messages"] += [{"role": "assistant", "content": None, "tool_calls": [
        {"id": "ts", "type": "function", "function": {"name": "tool_search", "arguments": "not json"}}]},
        {"role": "tool", "tool_call_id": "ts", "content": "not json"}]
    native = chat_to_responses_request(chat)
    assert native["tool_choice"] == {"type": "custom", "name": "apply_patch"}
    assert native["input"][-2]["arguments"] == {} and native["input"][-1]["tools"] == []
