"""Sending Chat Completions requests to a backend's Responses API (``/v1/responses``).

OpenAI's GPT-5.6 models reject function tools with reasoning on Chat Completions ("use /v1/responses or set
reasoning_effort to 'none'"), so an OpenAI backend with ``api: responses`` speaks the Responses API while the
rest of the gateway keeps seeing Chat Completions: requests are converted on the way out, and replies and
streams are converted back.

Stateless (``store: false``) unless the request carries a ``PROVIDER_CHAIN`` hint, which only a turn of a team's stored
mode on a backend with ``responses_store`` gets (decision 51). Then the provider stores the response too (``store:
true``), the leading system messages go as ``instructions`` (which a provider's stored response doesn't carry over to
the next turn, so they're never held twice), and a turn that follows one the provider stored on the same model is sent
as that response's ``previous_response_id`` plus only the messages after what it holds, which keeps its reasoning items.
Otherwise reasoning items aren't carried between turns, since the Chat Completions history the gateway holds has no
place for them.
"""

from __future__ import annotations

import json
from typing import Any

from .responses_tools import (DEFAULT_NAMESPACE, INPUT, LOCAL_SHELL, TOOL_SEARCH, TOOLS_KEY, custom_input, dumps, entry_for,
                              history_call, local_shell_action)
from .sse import SSEEvent, event

_PASS_THROUGH = ("temperature", "top_p", "parallel_tool_calls", "user", "prompt_cache_key", "safety_identifier",
                 "service_tier", "metadata")
# Set by the gateway on a stored-mode turn sent to a backend with responses_store (proxy.Gateway.build_upstream):
# {"previous_response_id": the provider's id of the turn before, or None, "held": the index of the first message the provider
# doesn't hold yet}. The leading system messages are always sent, as instructions. Never sent as it is.
PROVIDER_CHAIN = "_meritorder_provider_chain"


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content or [] if isinstance(p, dict) and p.get("type") == "text")


def _user_content(content: Any) -> Any:
    if isinstance(content, str) or content is None:
        return content or ""
    parts = []
    for p in content:
        t = p.get("type")
        if t == "text":
            parts.append({"type": "input_text", "text": p.get("text", "")})
        elif t == "image_url":
            url = (p.get("image_url") or {}).get("url") if isinstance(p.get("image_url"), dict) else p.get("image_url")
            if url:
                parts.append({"type": "input_image", "image_url": url, "detail": (p.get("image_url") or {}).get("detail", "auto")
                              if isinstance(p.get("image_url"), dict) else "auto"})
        elif t == "file":
            f = p.get("file") or {}
            data = str(f.get("file_data") or "")
            item: dict[str, Any] = {"type": "input_file"}
            if data.startswith(("https://", "http://")):
                item["file_url"] = data  # a URL in file_data (input_file_part, translate._document_parts): the Responses API's own field
            elif data:
                item["file_data"] = data
            if f.get("file_id"):
                item["file_id"] = f["file_id"]
            if f.get("filename"):
                item["filename"] = f["filename"]
            parts.append(item)
    return parts


def _json(text: str) -> Any:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


def _native_call(entry: dict | None, call_id: str, name: str, arguments: str) -> dict:
    """A Chat Completions tool call as the Responses item it stands for (responses_tools.TOOLS_KEY says which)."""
    kind = (entry or {}).get("kind")
    ns = (entry or {}).get("namespace")
    if kind == "custom":
        item = {"type": "custom_tool_call", "call_id": call_id, "name": entry["name"], "input": custom_input(arguments)}
    elif kind == LOCAL_SHELL:
        return {"type": "local_shell_call", "call_id": call_id, "status": "completed", "action": local_shell_action(arguments)}
    elif kind == TOOL_SEARCH:
        args = _json(arguments)
        return {"type": "tool_search_call", "call_id": call_id, "status": "completed", "execution": "client",
                "arguments": args if args is not None else {}}
    else:
        item = {"type": "function_call", "call_id": call_id, "name": (entry or {}).get("name") or name, "arguments": arguments}
    if ns and ns != DEFAULT_NAMESPACE:
        item["namespace"] = ns
    return item


def chat_to_responses_request(body: dict) -> dict:
    """``body[TOOLS_KEY]``, set when the request came in on the Responses API with tools other than plain functions:
    those tools go as they came (custom tools, namespaces, tool search, hosted tools), and so do their calls."""
    items: list[dict] = []
    registry = body.get(TOOLS_KEY)
    kinds: dict[str, str] = {}  # call id -> the kind of tool it called
    chain = body.get(PROVIDER_CHAIN)
    messages = list(body.get("messages") or [])
    instructions = None
    if chain is not None:
        lead = 0
        while lead < len(messages) and messages[lead].get("role") in ("system", "developer"):
            lead += 1
        instructions = "\n\n".join(t for t in (_text_of(m.get("content")) for m in messages[:lead]) if t) or None
        start = max(lead, int(chain.get("held") or 0)) if chain.get("previous_response_id") else lead
        messages = messages[start:]
    for m in messages:
        role = m.get("role")
        if role in ("system", "developer"):
            items.append({"type": "message", "role": "developer", "content": _text_of(m.get("content"))})
        elif role == "user":
            items.append({"type": "message", "role": "user", "content": _user_content(m.get("content"))})
        elif role == "assistant":
            text = _text_of(m.get("content"))
            if text:
                items.append({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]})
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                entry = entry_for(registry, fn.get("name", ""))
                if entry is None and registry is not None and fn.get("name") in (LOCAL_SHELL, TOOL_SEARCH):
                    entry = {"kind": fn["name"], "name": fn["name"]}  # a call from the history to a tool no longer declared
                kinds[tc.get("id", "")] = (entry or {}).get("kind", "function")
                items.append(_native_call(entry, tc.get("id", ""), fn.get("name", ""), fn.get("arguments") or "{}"))
        elif role == "tool":
            call_id, output = m.get("tool_call_id", ""), _text_of(m.get("content"))
            kind = kinds.get(call_id)
            if kind == "custom":
                items.append({"type": "custom_tool_call_output", "call_id": call_id, "output": output})
            elif kind == TOOL_SEARCH:
                found = _json(output)
                items.append({"type": "tool_search_output", "call_id": call_id, "status": "completed", "execution": "client",
                              "tools": found if isinstance(found, list) else []})
            else:
                items.append({"type": "function_call_output", "call_id": call_id, "output": output})

    out: dict[str, Any] = {"model": body.get("model"), "input": items, "store": chain is not None}
    if instructions:
        out["instructions"] = instructions
    if chain is not None and chain.get("previous_response_id"):
        out["previous_response_id"] = chain["previous_response_id"]
    limit =body.get("max_completion_tokens") or body.get("max_tokens")
    if limit:
        out["max_output_tokens"] = limit
    for k in _PASS_THROUGH:
        if body.get(k) is not None:
            out[k] = body[k]
    if body.get("reasoning_effort"):
        out["reasoning"] = {"effort": body["reasoning_effort"]}
    tools = []
    for t in body.get("tools") or []:
        if t.get("type") != "function":
            continue
        fn = t.get("function") or {}
        # Responses makes function tools strict unless told otherwise; most clients' schemas aren't strict-ready.
        tools.append({"type": "function", "name": fn.get("name", ""), "description": fn.get("description"),
                      "parameters": fn.get("parameters") or {"type": "object", "properties": {}}, "strict": bool(fn.get("strict", False))})
    if registry is not None:
        # The client's own Responses tools, as it sent them (``additional_tools`` items' among them: the public API takes
        # tools in ``tools``); the tools a tool search found come with its output item instead.
        tools = list(registry.get("native") or [])
    if tools:
        out["tools"] = [{k: v for k, v in t.items() if v is not None} for t in tools]
        tc = body.get("tool_choice")
        if isinstance(tc, str):
            out["tool_choice"] = tc
        elif isinstance(tc, dict) and tc.get("type") == "function":
            name = (tc.get("function") or {}).get("name")
            entry = entry_for(registry, name or "")
            if entry is not None and entry["kind"] in ("custom", "function"):
                out["tool_choice"] = {"type": entry["kind"], "name": entry["name"]}
            else:
                out["tool_choice"] = {"type": "function", "name": name}
    rf = body.get("response_format") or {}
    if rf.get("type") == "json_schema":
        js = rf.get("json_schema") or {}
        out["text"] = {"format": {"type": "json_schema", "name": js.get("name", "output"), "schema": js.get("schema") or {},
                                  "strict": bool(js.get("strict", False))}}
    elif rf.get("type") == "json_object":
        out["text"] = {"format": {"type": "json_object"}}
    if body.get("stream"):
        out["stream"] = True
    return out


def chat_usage(u: dict | None) -> dict | None:
    if not u:
        return None
    prompt, completion = u.get("input_tokens") or 0, u.get("output_tokens") or 0
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": u.get("total_tokens") or prompt + completion,
            "prompt_tokens_details": {"cached_tokens": (u.get("input_tokens_details") or {}).get("cached_tokens") or 0},
            "completion_tokens_details": {"reasoning_tokens": (u.get("output_tokens_details") or {}).get("reasoning_tokens") or 0}}


def _finish(resp: dict, has_calls: bool) -> str:
    if has_calls:
        return "tool_calls"
    if resp.get("status") == "incomplete":
        reason = (resp.get("incomplete_details") or {}).get("reason")
        return "content_filter" if reason == "content_filter" else "length"
    return "stop"


def responses_to_chat_completion(resp: dict) -> dict:
    text, refusal, calls = "", "", []
    for item in resp.get("output") or []:
        if item.get("type") == "message":
            for part in item.get("content") or []:
                if part.get("type") == "output_text":
                    text += part.get("text", "")
                elif part.get("type") == "refusal":
                    refusal += part.get("refusal", "")
        else:
            call = history_call(item)  # function, custom tool, local shell and tool search calls
            if call is not None:
                calls.append(call)
    msg: dict[str, Any] = {"role": "assistant", "content": text or None}
    if refusal:
        msg["refusal"] = refusal
    if calls:
        msg["tool_calls"] = calls
    out = {"id": resp.get("id", ""), "object": "chat.completion", "created": resp.get("created_at") or 0,
           "model": resp.get("model", ""), "choices": [{"index": 0, "message": msg, "finish_reason": _finish(resp, bool(calls))}]}
    usage = chat_usage(resp.get("usage"))
    if usage:
        out["usage"] = usage
    return out


_CALLS = ("function_call", "custom_tool_call", "local_shell_call", "tool_search_call")


def _escaped(text: str) -> str:
    """Text as it goes inside a JSON string (each character escapes on its own, so pieces join into the whole)."""
    return dumps(text)[1:-1]


class ResponsesToChatStream:
    """Turns Responses API stream events into Chat Completions chunks, ending with a usage chunk and [DONE]."""

    def __init__(self) -> None:
        self.id, self.model, self.created = "", "", 0
        self.calls: dict[str, int] = {}  # call item id -> chat tool index
        self.custom: dict[str, str] = {}  # custom_tool_call item id -> its input so far
        self.done = False

    def _chunk(self, delta: dict | None = None, finish: str | None = None, usage: dict | None = None) -> SSEEvent:
        d: dict[str, Any] = {"id": self.id, "object": "chat.completion.chunk", "created": self.created, "model": self.model,
                             "choices": [] if delta is None else [{"index": 0, "delta": delta, "finish_reason": finish}]}
        if usage is not None:
            d["usage"] = usage
        return event(d)

    def feed(self, ev: SSEEvent) -> list[SSEEvent]:
        if self.done:
            return []
        d = ev.json()
        if not d:
            return []
        t = d.get("type") or ev.event
        if t == "response.created":
            r = d.get("response") or {}
            self.id, self.model, self.created = r.get("id", ""), r.get("model", ""), r.get("created_at") or 0
            return [self._chunk({"role": "assistant", "content": ""})]
        if t == "response.output_text.delta":
            return [self._chunk({"content": d.get("delta", "")})]
        if t == "response.refusal.delta":
            return [self._chunk({"refusal": d.get("delta", "")})]
        item = d.get("item") or {}
        if t == "response.output_item.added" and item.get("type") in _CALLS:
            key = item.get("id") or str(d.get("output_index"))
            i = self.calls.setdefault(key, len(self.calls))
            call = history_call(item) or {}
            name = (call.get("function") or {}).get("name", "")
            if item["type"] == "function_call":
                args = item.get("arguments") or ""
            elif item["type"] == "custom_tool_call":
                # The input streams as text: here it goes out as the string in {"input": "..."}, escaped as it comes.
                self.custom[key] = item.get("input") or ""
                args = '{"' + INPUT + '":"' + _escaped(self.custom[key])
            else:
                args = ""  # local shell and tool search calls come whole with output_item.done
            return [self._chunk({"tool_calls": [{"index": i, "id": item.get("call_id") or item.get("id"), "type": "function",
                                                 "function": {"name": name, "arguments": args}}]})]
        if t in ("response.function_call_arguments.delta", "response.custom_tool_call_input.delta"):
            key = d.get("item_id") or str(d.get("output_index"))
            i = self.calls.get(key)
            if i is None:
                return []
            delta = d.get("delta", "")
            if key in self.custom:
                self.custom[key] += delta
                delta = _escaped(delta)
            return [self._chunk({"tool_calls": [{"index": i, "function": {"arguments": delta}}]})]
        if t == "response.output_item.done" and item.get("type") in _CALLS and item["type"] != "function_call":
            key = item.get("id") or str(d.get("output_index"))
            i = self.calls.get(key)
            if i is None:
                return []
            if key in self.custom:
                full, sent = item.get("input") or "", self.custom.pop(key)
                rest = _escaped(full[len(sent):]) if full.startswith(sent) else ""
                return [self._chunk({"tool_calls": [{"index": i, "function": {"arguments": rest + '"}'}}]})]
            call = history_call(item) or {}
            return [self._chunk({"tool_calls": [{"index": i, "function": {"arguments": (call.get("function") or {}).get("arguments", "")}}]})]
        if t in ("response.completed", "response.incomplete"):
            self.done = True
            r = d.get("response") or {}
            usage = chat_usage(r.get("usage"))
            out = [self._chunk({}, _finish(r, bool(self.calls)))]
            if usage:
                out.append(self._chunk(None, usage=usage))
            return out + [SSEEvent(data="[DONE]")]
        if t in ("response.failed", "error"):
            self.done = True
            err = (d.get("response") or {}).get("error") or d.get("error") or d
            err = err if isinstance(err, dict) else {"message": str(err)}
            out_err = {"message": err.get("message") or "upstream error", "type": "api_error"}
            if err.get("code"):  # e.g. rate_limit_exceeded: lets the gateway treat it as busy rather than broken
                out_err["code"] = err["code"]
            return [event({"error": out_err}), SSEEvent(data="[DONE]")]
        return []


async def to_chat_stream(events):
    conv = ResponsesToChatStream()
    async for ev in events:
        for out in conv.feed(ev):
            yield out
    if not conv.done:
        yield SSEEvent(data="[DONE]")

