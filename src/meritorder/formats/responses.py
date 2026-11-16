"""OpenAI Responses API (``/v1/responses``) on top of the Chat Completions pipeline.

Requests are converted to Chat Completions, routed like any other request (to
OpenAI-format or Anthropic-format backends), and the result is converted back.

This module is stateless: it converts what it's given. Clients that send the full input
each turn (Codex CLI, ``store: false``) need nothing else. Chaining with
``previous_response_id`` or a ``conversation`` needs the team's stored mode
(stored/service.py), or, over WebSocket, the connection's memory (responses_ws.py),
which rebuild the full input and hand it here. Reasoning items from earlier turns are
dropped; the gateway's own compaction items become the summary they hold. Tools other than
plain functions (Codex's custom tools, namespaces, tool search, local shell, hosted tools)
are translated by responses_tools.py.
"""

from __future__ import annotations

import base64
import binascii
import json
import time
import uuid
from typing import Any

from ..pricing import Usage
from .responses_tools import (ITEM_PREFIX, TOOLS_KEY, InputStream, ToolSet, chat_name, dumps, entry_for, history_call, input_file_part,
                              output_item, tool_output)
from .sse import SSEEvent, event
from .translate import EFFORTS, usage_from_openai

CALL_ITEMS = ("function_call", "custom_tool_call", "local_shell_call", "tool_search_call")
OUTPUT_ITEMS = ("function_call_output", "custom_tool_call_output", "local_shell_call_output", "tool_search_output")


class ResponsesError(Exception):
    pass


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


# The gateway's own compaction items (POST /v1/responses/compact, responses_compact.py): the summary, encoded, not encrypted
# (it's the client's own conversation). The prefix tells them from OpenAI's, which only OpenAI can read.
COMPACTION_PREFIX = "mo1."
SUMMARY_HEADER = "Summary of the conversation so far (earlier turns were compacted):\n\n"


def compaction_item(item_id: str, summary: str) -> dict:
    raw = json.dumps({"summary": summary}, ensure_ascii=False, separators=(",", ":")).encode()
    return {"type": "compaction", "id": item_id, "encrypted_content": COMPACTION_PREFIX + base64.urlsafe_b64encode(raw).decode().rstrip("=")}


def compaction_summary(content: Any) -> str | None:
    """The summary in one of the gateway's compaction items; None for anything else (OpenAI's, or a damaged one)."""
    if not isinstance(content, str) or not content.startswith(COMPACTION_PREFIX):
        return None
    data = content[len(COMPACTION_PREFIX):]
    try:
        value = json.loads(base64.urlsafe_b64decode(data + "=" * (-len(data) % 4)))
    except (ValueError, binascii.Error):
        return None
    summary = value.get("summary") if isinstance(value, dict) else None
    return summary if isinstance(summary, str) else None


def _compaction_message(item: dict) -> dict:
    """A compaction item as the message it stands for: the summary, as the user's, wherever the turn goes."""
    summary = compaction_summary(item.get("encrypted_content"))
    if summary is None:
        # Dropping it would lose every turn it replaced without a word.
        raise ResponsesError("this compaction item wasn't made by this gateway (OpenAI's are encrypted for OpenAI alone), so it "
                             "can't be read here: send the conversation itself, or compact it with this gateway's POST "
                             "/v1/responses/compact")
    return {"role": "user", "content": SUMMARY_HEADER + summary}


def _content_to_chat(content: Any) -> Any:
    if isinstance(content, str):
        return content
    parts = []
    for p in content or []:
        t = p.get("type")
        if t in ("input_text", "output_text", "text"):
            parts.append({"type": "text", "text": p.get("text", "")})
        elif t == "input_image":
            url = p.get("image_url") or (p.get("file_id") and f"file:{p['file_id']}")
            if url:
                parts.append({"type": "image_url", "image_url": {"url": url}})
        elif t == "input_file" and (file := input_file_part(p)):
            parts.append(file)
    if parts and all(p["type"] == "text" for p in parts):
        return "".join(p["text"] for p in parts)
    return parts


def responses_to_chat(body: dict) -> dict:
    """A Responses API request as Chat Completions. ``previous_response_id`` and ``conversation`` must already have been
    resolved into ``input`` (the stored mode does that); here they're refused, as in batches, which don't chain."""
    if body.get("previous_response_id") or body.get("conversation"):
        raise ResponsesError("previous_response_id and conversation can't be used here: send the full conversation in "
                             "`input` (store: false).")
    messages: list[dict] = []
    if body.get("instructions"):
        messages.append({"role": "system", "content": body["instructions"]})
    items = body.get("input")
    if isinstance(items, str):
        items = [{"role": "user", "content": items}]
    pending_calls: list[dict] = []
    pending_images: list[dict] = []  # tool results' images, sent after the run of results (responses_tools.tool_output)
    toolset = ToolSet()
    for spec in body.get("tools") or []:
        toolset.add(spec)

    def flush_images() -> None:
        messages.extend(pending_images)
        pending_images.clear()

    def flush_calls(images: bool = True) -> None:
        if images:
            flush_images()
        if pending_calls:
            messages.append({"role": "assistant", "content": None, "tool_calls": list(pending_calls)})
            pending_calls.clear()

    for item in items or []:
        t = item.get("type", "message")
        if t == "item_reference":
            # Dropping it would lose part of the conversation without a word (the Vercel AI SDK sends earlier items this way
            # unless store is false).
            raise ResponsesError("item_reference needs stored responses, which keep the items it names: send the items "
                                 "themselves (store: false), or ask an admin to turn on stored responses for the team")
        if t == "compaction":
            flush_calls()
            messages.append(_compaction_message(item))
        elif t == "compaction_trigger":
            # Dropped, the model would answer instead of compacting, and the client would wait for a compaction item.
            raise ResponsesError("compaction_trigger isn't supported here: compact with POST /v1/responses/compact, or let the "
                                 "client summarize (Codex CLI does for a custom provider unless capabilities.remote_compaction is set)")
        elif t == "additional_tools":
            # Codex's tool declarations inside the conversation ("Responses lite", incremental tools): tools, not a message.
            for spec in item.get("tools") or []:
                toolset.add(spec)
        elif t == "message":
            flush_calls()
            role = item.get("role", "user")
            role = "system" if role == "developer" else role
            messages.append({"role": role, "content": _content_to_chat(item.get("content"))})
        elif t in CALL_ITEMS:
            flush_images()  # a new turn's calls: the images of the last turn's results go before them
            pending_calls.append(history_call(item))
        elif t in OUTPUT_ITEMS:
            flush_calls(images=False)  # images wait for the run of results to end
            call_id = item.get("call_id") or item.get("id") or ""
            if t == "tool_search_output":
                for spec in item.get("tools") or []:
                    toolset.add(spec, native=False)  # the tools it found can be called from here on
                content, images = dumps(item.get("tools") or []), None
            else:
                content, images = tool_output(item.get("output"), call_id)
            messages.append({"role": "tool", "tool_call_id": call_id, "content": content})
            if images:
                pending_images.append(images)
        # reasoning, web_search_call, image_generation_call, etc.: no Chat Completions equivalent; dropped
    flush_calls()

    out: dict[str, Any] = {"model": body.get("model"), "messages": messages}
    if body.get("max_output_tokens"):
        out["max_tokens"] = body["max_output_tokens"]
    for k in ("temperature", "top_p", "parallel_tool_calls", "user", "prompt_cache_key"):
        if body.get(k) is not None:
            out[k] = body[k]
    effort = (body.get("reasoning") or {}).get("effort")
    if effort in EFFORTS:
        out["reasoning_effort"] = effort
    tools = toolset.tools()
    if tools:
        out["tools"] = tools
        tc = body.get("tool_choice")
        if isinstance(tc, str):
            out["tool_choice"] = tc
        elif isinstance(tc, dict) and tc.get("type") in ("function", "custom"):
            out["tool_choice"] = {"type": "function", "function": {"name": chat_name(tc.get("namespace"), tc.get("name") or "")}}
    registry = toolset.registry()
    if registry is not None:
        out[TOOLS_KEY] = registry
    fmt = (body.get("text") or {}).get("format") or {}
    if fmt.get("type") == "json_schema":
        out["response_format"] = {"type": "json_schema", "json_schema": {"name": fmt.get("name", "output"), "schema": fmt.get("schema"),
                                                                          "strict": fmt.get("strict", False)}}
    if body.get("stream"):
        out["stream"] = True
        out["stream_options"] = {"include_usage": True}
    return out


def responses_usage(u: Usage) -> dict:
    return {
        "input_tokens": u.prompt_tokens,
        "input_tokens_details": {"cached_tokens": u.cache_read_tokens, "cache_write_tokens": u.cache_write_tokens},
        "output_tokens": u.output_tokens,
        "output_tokens_details": {"reasoning_tokens": u.reasoning_tokens},
        "total_tokens": u.total_tokens,
    }


def request_echo(body: dict) -> dict:
    """What a response object repeats from the request that made it (OpenAI's Response fields), so a client reading it
    back, or chaining from it, sees the settings it was made with."""
    return {"instructions": body.get("instructions"), "metadata": body.get("metadata") or {}, "tools": body.get("tools") or [],
            "tool_choice": body.get("tool_choice") or "auto", "parallel_tool_calls": body.get("parallel_tool_calls", True),
            "temperature": body.get("temperature"), "top_p": body.get("top_p"), "max_output_tokens": body.get("max_output_tokens"),
            "truncation": body.get("truncation") or "disabled", "text": body.get("text") or {"format": {"type": "text"}},
            "reasoning": body.get("reasoning") or {"effort": None, "summary": None}, "user": body.get("user"),
            "safety_identifier": body.get("safety_identifier"), "prompt_cache_key": body.get("prompt_cache_key"),
            "previous_response_id": body.get("previous_response_id"), "background": bool(body.get("background"))}


def _response_obj(rid: str, model: str, created: int, status: str, output: list, usage: dict | None, extra: dict | None = None) -> dict:
    return {"id": rid, "object": "response", "created_at": created, "status": status, "model": model, "output": output,
            "usage": usage, "error": None, "incomplete_details": None, "parallel_tool_calls": True, "tool_choice": "auto",
            "tools": [], "store": False, **(extra or {})}


def chat_to_responses(resp: dict, model: str, rid: str | None = None, extra: dict | None = None, tools: dict | None = None) -> dict:
    """``rid``: the response's id (a stored response keeps the one it's saved under); ``extra``: what the response object
    carries from the request (request_echo, ``store``, ``conversation``); ``tools``: the request's TOOLS_KEY, which says
    which calls go back as custom tool, local shell or tool search calls, or with a namespace."""
    choice = (resp.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    finish = choice.get("finish_reason")
    output: list[dict] = []
    parts: list[dict] = []
    if msg.get("content"):
        parts.append({"type": "output_text", "text": msg["content"], "annotations": []})
    refusal = msg.get("refusal") if isinstance(msg.get("refusal"), str) else None
    if finish == "content_filter" and not refusal and not msg.get("content") and not msg.get("tool_calls"):
        refusal = FILTERED
    if refusal:
        parts.append({"type": "refusal", "refusal": refusal})
    if parts:
        output.append({"type": "message", "id": _id("msg"), "status": "completed", "role": "assistant", "content": parts})
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        entry = entry_for(tools, fn.get("name", ""))
        output.append(output_item(entry, _id(ITEM_PREFIX[(entry or {}).get("kind", "function")]), tc.get("id") or _id("call"),
                                  fn.get("name", ""), fn.get("arguments") or "{}"))
    return _finished(_response_obj(rid or _id("resp"), model, int(time.time()), "completed", output,
                                   responses_usage(usage_from_openai(resp.get("usage"))), extra), finish)


# What a refusal part says when the provider's content filter stopped the reply before it said anything.
FILTERED = "The provider's content filter stopped this response."
INCOMPLETE = {"length": "max_output_tokens", "content_filter": "content_filter"}


def _finished(obj: dict, finish: str | None) -> dict:
    """The response object's status for a Chat Completions finish_reason: incomplete when the reply was cut off by
    the output limit or a content filter (the reason OpenAI gives for each), else completed."""
    reason = INCOMPLETE.get(finish or "")
    obj["status"] = "incomplete" if reason else "completed"
    obj["completed_at"] = None if reason else int(time.time())
    if reason:
        obj["incomplete_details"] = {"reason": reason}
    return obj


class ChatToResponsesStream:
    """Converts Chat Completions chunks into Responses API streaming events."""

    def __init__(self, model: str, rid: str | None = None, extra: dict | None = None, tools: dict | None = None):
        """``tools``: the request's TOOLS_KEY (see chat_to_responses)."""
        self.rid = rid or _id("resp")
        self.extra = extra or {}
        self.tools = tools
        self.model = model
        self.created = int(time.time())
        self.seq = 0
        self.started = False
        self.output: list[dict] = []
        self.msg: dict | None = None  # open message item
        self.calls: dict[int, dict] = {}  # chat tool index -> the call's item
        self.raw: dict[int, tuple[dict | None, str, InputStream]] = {}  # chat tool index -> (entry, the function's name, arguments)
        self.usage = Usage()
        self.finish: str | None = None
        self.error: str | None = None  # the provider's error in the stream: the response ends failed with it
        self.done = False
        self.final: dict | None = None  # the finished response object, once finish_events has made it

    def _ev(self, type_: str, **data) -> SSEEvent:
        self.seq += 1
        return event({"type": type_, "sequence_number": self.seq, **data}, type_)

    def _start(self) -> list[SSEEvent]:
        self.started = True
        obj = _response_obj(self.rid, self.model, self.created, "in_progress", [], None, self.extra)
        return [self._ev("response.created", response=obj), self._ev("response.in_progress", response=obj)]

    def _write(self, kind: str, text: str) -> list[SSEEvent]:
        """``text`` added to the open message's part of this kind (``output_text``, or ``refusal``), opening the message
        and the part as needed."""
        out: list[SSEEvent] = []
        if self.msg is None:
            self.msg = {"type": "message", "id": _id("msg"), "status": "in_progress", "role": "assistant", "content": []}
            self.output.append(self.msg)
            out.append(self._ev("response.output_item.added", output_index=len(self.output) - 1, item={**self.msg, "content": []}))
        m = self.msg
        idx = self.output.index(m)
        part = next((p for p in m["content"] if p["type"] == kind), None)
        if part is None:
            part = {"type": "output_text", "text": "", "annotations": []} if kind == "output_text" else {"type": "refusal", "refusal": ""}
            m["content"].append(part)
            out.append(self._ev("response.content_part.added", item_id=m["id"], output_index=idx, content_index=len(m["content"]) - 1,
                                part=dict(part)))
        key = "text" if kind == "output_text" else "refusal"
        part[key] += text
        delta = "response.output_text.delta" if kind == "output_text" else "response.refusal.delta"
        out.append(self._ev(delta, item_id=m["id"], output_index=idx, content_index=m["content"].index(part), delta=text))
        return out

    def _close_msg(self) -> list[SSEEvent]:
        if not self.msg:
            return []
        m, self.msg = self.msg, None
        idx = self.output.index(m)
        m["status"] = "completed"
        out = []
        for i, part in enumerate(m["content"]):
            if part["type"] == "output_text":
                out.append(self._ev("response.output_text.done", item_id=m["id"], output_index=idx, content_index=i, text=part["text"]))
            else:
                out.append(self._ev("response.refusal.done", item_id=m["id"], output_index=idx, content_index=i, refusal=part["refusal"]))
            out.append(self._ev("response.content_part.done", item_id=m["id"], output_index=idx, content_index=i, part=part))
        return out + [self._ev("response.output_item.done", output_index=idx, item=m)]

    def fail_events(self, message: str, code: str = "server_error") -> list[SSEEvent]:
        """The stream broke off after it started: the response as far as it got, failed (``response.failed``)."""
        if self.done:
            return []
        self.done = True
        out = [] if self.started else self._start()
        obj = _response_obj(self.rid, self.model, self.created, "failed", self.output, responses_usage(self.usage), self.extra)
        obj["error"] = {"code": code, "message": message}
        self.final = obj
        return out + [self._ev("response.failed", response=obj)]

    def feed(self, ev: SSEEvent) -> list[SSEEvent]:
        if ev.data.strip() == "[DONE]":
            return self.finish_events()
        d = ev.json()
        if not d:
            return []
        if self.error is not None:
            return []  # the response failed with the error; what comes after it isn't part of it
        out = [] if self.started else self._start()
        if d.get("error"):
            err = d["error"] if isinstance(d["error"], dict) else {"message": str(d["error"])}
            self.error = str(err.get("message") or "upstream error")
            out.append(self._ev("error", code=err.get("type", "api_error"), message=self.error, param=None))
            return out + self.fail_events(self.error)
        if d.get("usage"):
            self.usage.merge(usage_from_openai(d["usage"]))
        for ch in d.get("choices") or []:
            delta = ch.get("delta") or {}
            if delta.get("content"):
                out += self._write("output_text", delta["content"])
            if isinstance(delta.get("refusal"), str) and delta["refusal"]:
                out += self._write("refusal", delta["refusal"])
            for tc in delta.get("tool_calls") or []:
                i = tc.get("index", 0)
                fn = tc.get("function") or {}
                if i not in self.calls:
                    out += self._close_msg()
                    entry = entry_for(self.tools, fn.get("name", ""))
                    kind = (entry or {}).get("kind", "function")
                    item = output_item(entry, _id(ITEM_PREFIX[kind]), tc.get("id") or _id("call"), fn.get("name", ""), "", "in_progress")
                    if kind == "custom":
                        item["input"] = ""
                    elif kind == "function":
                        item["arguments"] = ""
                    self.calls[i] = item
                    self.raw[i] = (entry, fn.get("name", ""), InputStream())
                    self.output.append(item)
                    out.append(self._ev("response.output_item.added", output_index=len(self.output) - 1, item=dict(item)))
                item = self.calls[i]
                if fn.get("arguments"):
                    entry, _, args = self.raw[i]
                    text = args.feed(fn["arguments"])
                    kind = (entry or {}).get("kind", "function")
                    if kind == "function":
                        item["arguments"] += fn["arguments"]
                        out.append(self._ev("response.function_call_arguments.delta", item_id=item["id"],
                                            output_index=self.output.index(item), delta=fn["arguments"]))
                    elif kind == "custom" and text:
                        item["input"] += text
                        out.append(self._ev("response.custom_tool_call_input.delta", item_id=item["id"],
                                            output_index=self.output.index(item), delta=text))
            if ch.get("finish_reason"):
                self.finish = ch["finish_reason"]
        return out

    def finish_events(self) -> list[SSEEvent]:
        if self.done:
            return []
        self.done = True
        out = [] if self.started else self._start()
        if self.finish == "content_filter" and not self.output:
            out += self._write("refusal", FILTERED)  # the filter stopped it before it said anything: say so
        out += self._close_msg()
        for i, item in self.calls.items():
            idx = self.output.index(item)
            entry, name, args = self.raw[i]
            done = output_item(entry, item["id"], item["call_id"], name, args.raw)
            streamed = item.get("input") or ""
            if done["type"] == "custom_tool_call" and len(done["input"]) > len(streamed) and done["input"].startswith(streamed):
                # The rest of an input that didn't stream as text (arguments of another shape). Where what streamed
                # isn't the input at all, the done event below carries the right one.
                out.append(self._ev("response.custom_tool_call_input.delta", item_id=item["id"], output_index=idx,
                                    delta=done["input"][len(streamed):]))
            item.clear()
            item.update(done)
            if item["type"] == "function_call":
                out.append(self._ev("response.function_call_arguments.done", item_id=item["id"], output_index=idx, arguments=item["arguments"]))
            elif item["type"] == "custom_tool_call":
                out.append(self._ev("response.custom_tool_call_input.done", item_id=item["id"], output_index=idx, input=item["input"]))
            out.append(self._ev("response.output_item.done", output_index=idx, item=item))
        obj = _finished(_response_obj(self.rid, self.model, self.created, "completed", self.output, responses_usage(self.usage),
                                      self.extra), self.finish)
        status = obj["status"]
        self.final = obj
        out.append(self._ev("response.completed" if status == "completed" else "response.incomplete", response=obj))
        return out
