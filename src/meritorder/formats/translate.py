"""Translation between OpenAI Chat Completions and Anthropic Messages.

Only used when the client and the backend speak different formats. When they
match, requests pass through untouched (see proxy.py), which keeps provider
fields like ``cache_control`` and thinking signatures intact.
"""

from __future__ import annotations

import base64
import binascii
import json
import re
import time
import uuid
from typing import Any

from ..pricing import Usage
from .sse import SSEEvent, event

STOP_A2O = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "model_context_window_exceeded": "length",
    "tool_use": "tool_calls",
    "pause_turn": "stop",
    "refusal": "content_filter",
}
STOP_O2A = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "refusal",
}
EFFORTS = {"minimal": "low", "low": "low", "medium": "medium", "high": "high", "xhigh": "xhigh", "max": "max"}


def stop_o2a(finish: str | None, tool_calls: bool, refusal: bool = False) -> str:
    """Anthropic's stop_reason for a Chat Completions finish_reason. Some providers (Gemini's OpenAI endpoint, some vLLM
    parsers) end a turn that called tools with "stop"; an Anthropic client reads tool_use as the signal to run them."""
    if refusal:
        return "refusal"
    stop = STOP_O2A.get(finish or "", "end_turn")
    return "tool_use" if tool_calls and stop == "end_turn" else stop


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


def _loads(args: Any) -> dict:
    if isinstance(args, dict):
        return args
    if not args:
        return {}
    try:
        value = json.loads(args)
    except (json.JSONDecodeError, TypeError):
        return {"arguments": args}
    return value if isinstance(value, dict) else {"value": value}


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def text_of(content: Any, sep: str = "") -> str:
    """Plain text from either format's content (string or list of parts). ``sep`` goes between parts: a system prompt
    in blocks is separate sections, and run together they would read as one ("...Claude.You are...")."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for p in content:
        if isinstance(p, dict) and p.get("type") in ("text", "input_text", "output_text"):
            parts.append(p.get("text", ""))
        elif isinstance(p, str):
            parts.append(p)
    return sep.join(parts)


# What a coding harness adds to a person's message: Claude Code's system reminders (CLAUDE.md, todo lists, notices) and
# local command output, Codex CLI's environment context and AGENTS.md instructions. They're the same on every task,
# so wording read from them would say nothing about the task.
HARNESS_TEXT = re.compile(r"<(system-reminder|environment_context|user_instructions|INSTRUCTIONS|local-command-stdout|"
                          r"local-command-stderr|local-command-caveat)>.*?</\1>|^# AGENTS\.md instructions for .*$", re.S | re.M)
PERSON_TEXT_MAX = 200_000  # characters of a message read for wrappers: a long paste isn't scanned past this


def person_text(content: Any) -> str:
    """What a person wrote in a message: its text (text_of) without what the harness added (HARNESS_TEXT). Empty when
    the message is only tool results, images or the harness's own text."""
    text = text_of(content)[:PERSON_TEXT_MAX]
    return HARNESS_TEXT.sub("", text) if "<" in text or "# AGENTS" in text else text


# ---------------------------------------------------------------- usage


def usage_from_anthropic(u: dict | None) -> Usage:
    u = u or {}
    by_ttl = u.get("cache_creation")  # {"ephemeral_5m_input_tokens": ..., "ephemeral_1h_input_tokens": ...}, where reported
    hour = by_ttl.get("ephemeral_1h_input_tokens") if isinstance(by_ttl, dict) else None
    return Usage(
        input_tokens=u.get("input_tokens") or 0,
        cache_read_tokens=u.get("cache_read_input_tokens") or 0,
        cache_write_tokens=u.get("cache_creation_input_tokens") or 0,
        cache_write_1h_tokens=int(hour) if isinstance(hour, (int, float)) and not isinstance(hour, bool) else None,
        output_tokens=u.get("output_tokens") or 0,
        web_search_requests=int((u.get("server_tool_use") or {}).get("web_search_requests") or 0),
        reasoning_tokens=int((u.get("output_tokens_details") or {}).get("thinking_tokens") or 0),
    )


def usage_from_openai(u: dict | None) -> Usage:
    u = u or {}
    details = u.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens") or 0
    written = details.get("cache_write_tokens") or 0
    prompt = u.get("prompt_tokens") or 0
    return Usage(
        input_tokens=max(prompt - cached - written, 0),
        cache_read_tokens=cached,
        cache_write_tokens=written,
        output_tokens=u.get("completion_tokens") or 0,
        reasoning_tokens=int((u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0),
        billed_usd=_billed(u),
    )


def _billed(u: dict) -> float | None:
    """OpenRouter's ``usage.cost``. With your own provider key (BYOK) it is only OpenRouter's fee, and the
    provider's charge to your key is in ``cost_details.upstream_inference_cost``."""
    c = u.get("cost")
    if isinstance(c, bool) or not isinstance(c, (int, float)) or c < 0:
        return None
    if u.get("is_byok"):
        up = (u.get("cost_details") or {}).get("upstream_inference_cost")
        if not isinstance(up, bool) and isinstance(up, (int, float)) and up > 0:
            c += up
    return float(c)


def anthropic_usage(u: Usage) -> dict:
    out = {
        "input_tokens": u.input_tokens,
        "cache_read_input_tokens": u.cache_read_tokens,
        "cache_creation_input_tokens": u.cache_write_tokens,
        "output_tokens": u.output_tokens,
    }
    if u.cache_write_1h_tokens is not None:
        hour = min(u.cache_write_1h_tokens, u.cache_write_tokens)
        out["cache_creation"] = {"ephemeral_5m_input_tokens": u.cache_write_tokens - hour, "ephemeral_1h_input_tokens": hour}
    return out


def openai_usage(u: Usage) -> dict:
    return {
        "prompt_tokens": u.prompt_tokens,
        "completion_tokens": u.output_tokens,
        "total_tokens": u.total_tokens,
        "prompt_tokens_details": {"cached_tokens": u.cache_read_tokens},
        "completion_tokens_details": {"reasoning_tokens": u.reasoning_tokens},
    }


class UsageTracker:
    """Reads usage from upstream stream events without changing them. It also keeps what the model wrote (text,
    reasoning, tool calls), so a stream that ends before its usage arrives can be estimated (proxy.estimate_usage)."""

    def __init__(self, fmt: str):
        self.fmt = fmt
        self.usage = Usage()
        self.written: list[str] = []
        self.output_reported = False  # the provider has said how much it wrote (Claude's message_delta, a usage chunk)

    @property
    def streamed(self) -> str:
        return "".join(self.written)

    def feed(self, ev: SSEEvent) -> None:
        d = ev.json()
        if not d:
            return
        if self.fmt == "anthropic":
            t = d.get("type")
            if t == "message_start":
                self.usage.merge(usage_from_anthropic((d.get("message") or {}).get("usage")))
            elif t == "message_delta":
                self.usage.merge(usage_from_anthropic(d.get("usage")))
                self.output_reported = self.output_reported or bool(d.get("usage"))
            elif t == "content_block_delta":
                delta = d.get("delta") or {}
                self.written += [delta[k] for k in ("text", "thinking", "partial_json") if isinstance(delta.get(k), str)]
            elif t == "content_block_start":
                block = d.get("content_block") or {}
                self.written += [block[k] for k in ("text", "name") if isinstance(block.get(k), str)]
            return
        if d.get("usage"):
            self.usage.merge(usage_from_openai(d["usage"]))
            # Some providers send the prompt's count first, with completion_tokens 0: that doesn't count the output.
            self.output_reported = self.output_reported or bool(isinstance(d["usage"], dict) and d["usage"].get("completion_tokens"))
        for ch in d.get("choices") or []:
            delta = (ch.get("delta") or {}) if isinstance(ch, dict) else {}
            self.written += [delta[k] for k in ("content", "reasoning_content", "reasoning", "refusal") if isinstance(delta.get(k), str)]
            for tc in delta.get("tool_calls") or []:
                fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
                self.written += [fn[k] for k in ("name", "arguments") if isinstance(fn.get(k), str)]


# ---------------------------------------------------------------- files, documents and what tools return


class TranslationError(ValueError):
    """Content one format carries that the backend's can't: the request is refused (400), never sent without it."""


def _data_url(url: str) -> tuple[str, str] | None:
    if url.startswith("data:") and ";base64," in url:
        media_type, data = url[5:].split(";base64,", 1)
        return media_type, data
    return None


def file_to_document(f: dict) -> dict:
    """A Chat Completions file part's ``file`` as an Anthropic document block. Claude reads PDFs and plain text; a file
    uploaded to OpenAI (``file_id``) only OpenAI can open."""
    data = str(f.get("file_data") or "")
    name = f.get("filename")
    parsed = _data_url(data)
    block: dict[str, Any]
    if parsed and parsed[0] == "application/pdf":
        block = {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": parsed[1]}}
    elif parsed and parsed[0].startswith("text/"):
        try:
            text = base64.b64decode(parsed[1], validate=True).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError) as e:
            raise TranslationError(f"file {name or ''} isn't base64 UTF-8 text") from e
        block = {"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": text}}
    elif data.startswith(("https://", "http://")):
        block = {"type": "document", "source": {"type": "url", "url": data}}
    elif data and not parsed and str(name or "").lower().endswith(".pdf"):
        block = {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": data}}
    elif f.get("file_id") and not data:
        raise TranslationError(f"file_id {f['file_id']} names a file uploaded to OpenAI, which Claude can't open: send the "
                               "file itself as file_data (a data: URL)")
    else:
        kind = parsed[0] if parsed else "this file"
        raise TranslationError(f"Claude reads PDFs and plain text as documents, not {kind}" + (f" ({name})" if name else ""))
    if name:
        block["title"] = str(name)
    return block


def _file_id_error(src: dict, what: str) -> TranslationError:
    return TranslationError(f"file_id {src.get('file_id')} names a file in Anthropic's Files API, which only Claude can open: "
                            f"send the {what} itself (base64 or a URL)")


def _document_parts(b: dict) -> list[dict]:
    """An Anthropic document block as Chat Completions parts: a PDF as a file part whose ``file_data`` is a data: URL
    (or the PDF's URL, which OpenRouter fetches), text as text."""
    src = b.get("source") or {}
    kind = src.get("type")
    title = b.get("title")
    if kind == "text":
        return [{"type": "text", "text": src.get("data", "")}]
    if kind == "content":
        content = src.get("content")
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        return [p for c in content or [] if isinstance(c, dict) for p in media_parts(c)]
    if kind == "base64":
        media_type = src.get("media_type") or "application/pdf"
        name = title or ("document.pdf" if media_type == "application/pdf" else "document")
        return [{"type": "file", "file": {"filename": str(name), "file_data": f"data:{media_type};base64,{src.get('data', '')}"}}]
    if kind == "url" and src.get("url"):
        url = str(src["url"])
        return [{"type": "file", "file": {"filename": str(title or url.rsplit("/", 1)[-1] or "document.pdf"), "file_data": url}}]
    if kind == "file":
        raise _file_id_error(src, "document")
    raise TranslationError(f"a document with source type {kind!r} can't be sent to this model")


def media_parts(b: dict) -> list[dict]:
    """One Anthropic content block (text, image or document) as Chat Completions parts; [] for any other block."""
    t = b.get("type")
    if t == "text":
        return [{"type": "text", "text": b.get("text", "")}]
    if t == "image":
        src = b.get("source") or {}
        if src.get("type") == "file":
            raise _file_id_error(src, "image")
        url = f"data:{src.get('media_type')};base64,{src.get('data')}" if src.get("type") == "base64" else src.get("url", "")
        return [{"type": "image_url", "image_url": {"url": url}}] if url else []
    if t == "document":
        return _document_parts(b)
    return []


def tool_media(text: str, call_id: str, media: list[dict]) -> tuple[str, dict | None]:
    """A tool result as a Chat Completions tool message's text, and a user message holding its images and files.

    Tool messages take text only, and Claude wants its tool results before anything else in a turn, so the media
    follows the run of tool results as a user message of its own, and the tool message says that it does."""
    if not media:
        return text, None
    n = len(media)
    images = all(p.get("type") == "image_url" for p in media)
    what = "image" if images else "attachment"
    note = f"({n} {what}{'' if n == 1 else 's'} from this result follow{'s' if n == 1 else ''} below.)"
    label = "Images" if images else "Attachments"
    return (f"{text}\n\n{note}" if text else note,
            {"role": "user", "content": [{"type": "text", "text": f"{label} returned by tool call {call_id}:"}, *media]})


# Claude models that refuse a forced tool choice (``any``, or a named tool) with a 400 on every request (Anthropic's
# "Thinking" docs, November 2026). The others accept it, and answer it without thinking first.
FORCED_TOOLS_REFUSED = re.compile(r"claude-(?:(?:fable|mythos)-(?:5-[1-9]|[6-9])|(?:opus|sonnet)-(?:5-[5-9]|[6-9]))")


def server_tools(body: dict) -> list[str]:
    """An Anthropic request's server tools (web search, bash, code execution...). Claude runs them and nothing else
    can, so a request translated for another provider goes without them (``x-meritorder-dropped-tools``)."""
    return [str(t.get("name") or t.get("type")) for t in body.get("tools") or []
            if isinstance(t, dict) and "input_schema" not in t and t.get("type") not in (None, "custom")]


# ---------------------------------------------------------------- OpenAI -> Anthropic


def _openai_parts_to_blocks(content: Any) -> list[dict]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    blocks = []
    for p in content:
        t = p.get("type")
        if t in ("text", "input_text"):
            if p.get("text"):
                block = {"type": "text", "text": p["text"]}
                if p.get("cache_control"):
                    block["cache_control"] = p["cache_control"]
                blocks.append(block)
        elif t == "image_url":
            url = (p.get("image_url") or {}).get("url", "")
            if url.startswith("data:") and ";base64," in url:
                media_type, data = url[5:].split(";base64,", 1)
                blocks.append({"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}})
            elif url:
                blocks.append({"type": "image", "source": {"type": "url", "url": url}})
        elif t == "file":
            blocks.append(file_to_document(p.get("file") or {}))
    return blocks


def openai_to_anthropic_request(body: dict, default_max_tokens: int, model: str = "") -> dict:
    """``model``: the Claude model it goes to, which decides whether a forced tool choice can be sent as one
    (FORCED_TOOLS_REFUSED); without it, it is."""
    system: list[dict] = []
    messages: list[dict] = []

    def push(role: str, blocks: list[dict]) -> None:
        if not blocks:
            return
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"].extend(blocks)
        else:
            messages.append({"role": role, "content": blocks})

    for m in body.get("messages") or []:
        role = m.get("role")
        content = m.get("content")
        if role in ("system", "developer"):
            system.extend(_openai_parts_to_blocks(content))
        elif role == "user":
            push("user", _openai_parts_to_blocks(content))
        elif role == "assistant":
            blocks = []
            text = text_of(content)
            if text:
                blocks.append({"type": "text", "text": text})
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                blocks.append(
                    {"type": "tool_use", "id": tc.get("id") or _id("toolu"), "name": fn.get("name", ""), "input": _loads(fn.get("arguments"))}
                )
            push("assistant", blocks)
        elif role == "tool":
            result: Any = content if isinstance(content, str) else _openai_parts_to_blocks(content)
            push("user", [{"type": "tool_result", "tool_use_id": m.get("tool_call_id", ""), "content": result}])

    out: dict[str, Any] = {
        "model": body.get("model"),
        "messages": messages,
        "max_tokens": body.get("max_completion_tokens") or body.get("max_tokens") or default_max_tokens,
    }
    if system:
        out["system"] = system
    stop = body.get("stop")
    if stop:
        out["stop_sequences"] = [stop] if isinstance(stop, str) else list(stop)
    if body.get("stream"):
        out["stream"] = True
    # temperature / top_p are deliberately dropped: current Claude models reject them.

    tools = []
    for t in body.get("tools") or []:
        if t.get("type") != "function":
            continue
        fn = t.get("function") or {}
        tool = {"name": fn.get("name", ""), "input_schema": fn.get("parameters") or {"type": "object", "properties": {}}}
        if fn.get("description"):
            tool["description"] = fn["description"]
        if fn.get("strict"):
            tool["strict"] = True
        tools.append(tool)
    if tools:
        out["tools"] = tools
        choice = body.get("tool_choice")
        named = (choice.get("function") or {}).get("name") if isinstance(choice, dict) and choice.get("type") == "function" else None
        tc: dict[str, Any]
        if choice == "none":
            tc = {"type": "none"}
        elif choice == "required":
            tc = {"type": "any"}
        elif named:
            tc = {"type": "tool", "name": named}
        else:
            tc = {"type": "auto"}
        thinking = (out.get("thinking") or {}).get("type") == "enabled"  # manual thinking refuses forced tools too
        if tc["type"] in ("any", "tool") and (thinking or FORCED_TOOLS_REFUSED.search(model or "")):
            # This model takes "auto" only: the choice becomes an instruction at the end of the system prompt, which is
            # what Anthropic suggests for these models (with strict tools, which the request's own tools carry over).
            ask = f"call the {named} tool" if named else "call one of the tools"
            system.append({"type": "text", "text": f"In this reply you must {ask}; answer with the tool call."})
            out["system"] = system
            tc = {"type": "auto"}
        if body.get("parallel_tool_calls") is False and tc["type"] != "none":
            tc["disable_parallel_tool_use"] = True
        if choice is not None or "disable_parallel_tool_use" in tc:
            out["tool_choice"] = tc

    output_config: dict[str, Any] = {}
    effort = body.get("reasoning_effort")
    if effort in EFFORTS:
        output_config["effort"] = EFFORTS[effort]
    rf = body.get("response_format") or {}
    if rf.get("type") == "json_schema" and (rf.get("json_schema") or {}).get("schema"):
        output_config["format"] = {"type": "json_schema", "schema": rf["json_schema"]["schema"]}
    if output_config:
        out["output_config"] = output_config
    if body.get("user"):
        out["metadata"] = {"user_id": str(body["user"])}
    return out


def anthropic_to_openai_response(resp: dict, model: str) -> dict:
    texts: list[str] = []
    tool_calls: list[dict] = []
    reasoning: list[str] = []
    for b in resp.get("content") or []:
        t = b.get("type")
        if t == "text":
            texts.append(b.get("text", ""))
        elif t == "tool_use":
            tool_calls.append(
                {"id": b.get("id"), "type": "function", "function": {"name": b.get("name"), "arguments": _dumps(b.get("input") or {})}}
            )
        elif t == "thinking" and b.get("thinking"):
            reasoning.append(b["thinking"])
    message: dict[str, Any] = {"role": "assistant", "content": "".join(texts) if (texts or not tool_calls) else None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    if reasoning:
        message["reasoning_content"] = "\n".join(reasoning)
    return {
        "id": f"chatcmpl-{resp.get('id') or uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": STOP_A2O.get(resp.get("stop_reason") or "", "stop")}],
        "usage": openai_usage(usage_from_anthropic(resp.get("usage"))),
    }


class AnthropicToOpenAIStream:
    """Converts an Anthropic Messages event stream into OpenAI chunks."""

    def __init__(self, model: str, include_usage: bool):
        self.id = f"chatcmpl-{uuid.uuid4().hex}"
        self.created = int(time.time())
        self.model = model
        self.include_usage = include_usage
        self.tool_index: dict[int, int] = {}
        self.usage = Usage()
        self.finish: str | None = None
        self.done = False

    def _chunk(self, delta: dict, finish: str | None = None) -> SSEEvent:
        return event(
            {
                "id": self.id,
                "object": "chat.completion.chunk",
                "created": self.created,
                "model": self.model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
        )

    def feed(self, ev: SSEEvent) -> list[SSEEvent]:
        d = ev.json()
        if not d:
            return []
        t = d.get("type") or ev.event
        if t == "message_start":
            self.usage.merge(usage_from_anthropic((d.get("message") or {}).get("usage")))
            return [self._chunk({"role": "assistant", "content": ""})]
        if t == "content_block_start":
            b = d.get("content_block") or {}
            if b.get("type") == "tool_use":
                oi = len(self.tool_index)
                self.tool_index[d.get("index", 0)] = oi
                return [
                    self._chunk(
                        {"tool_calls": [{"index": oi, "id": b.get("id"), "type": "function", "function": {"name": b.get("name"), "arguments": ""}}]}
                    )
                ]
            if b.get("type") == "text" and b.get("text"):
                return [self._chunk({"content": b["text"]})]
            return []
        if t == "content_block_delta":
            delta = d.get("delta") or {}
            dt = delta.get("type")
            if dt == "text_delta":
                return [self._chunk({"content": delta.get("text", "")})]
            if dt == "input_json_delta":
                oi = self.tool_index.get(d.get("index", 0))
                if oi is None:
                    return []  # a server tool's input (web search's query): Claude runs it, the client never sees it
                return [self._chunk({"tool_calls": [{"index": oi, "function": {"arguments": delta.get("partial_json", "")}}]})]
            if dt == "thinking_delta" and delta.get("thinking"):
                return [self._chunk({"reasoning_content": delta["thinking"]})]
            return []
        if t == "message_delta":
            self.finish = STOP_A2O.get((d.get("delta") or {}).get("stop_reason") or "", "stop")
            self.usage.merge(usage_from_anthropic(d.get("usage")))
            return []
        if t == "message_stop":
            return self.finish_events()
        if t == "error":
            # The stream ends here, as OpenAI's own does after an error: no finish chunk saying it completed (the SDK
            # raises on the error), and [DONE] for clients that wait for it.
            err = d.get("error") or {}
            self.done = True
            return [event({"error": {"message": err.get("message", "upstream error"), "type": err.get("type", "api_error")}}),
                    SSEEvent(data="[DONE]")]
        return []

    def finish_events(self) -> list[SSEEvent]:
        if self.done:
            return []
        self.done = True
        out = [self._chunk({}, self.finish or "stop")]
        if self.include_usage:
            out.append(
                event(
                    {
                        "id": self.id,
                        "object": "chat.completion.chunk",
                        "created": self.created,
                        "model": self.model,
                        "choices": [],
                        "usage": openai_usage(self.usage),
                    }
                )
            )
        out.append(SSEEvent(data="[DONE]"))
        return out


# ---------------------------------------------------------------- Anthropic -> OpenAI


def _marked(parts: list[dict], block: dict, keep: bool) -> list[dict]:
    """``parts`` (one block's), with the block's cache marker on the last of them when markers are kept."""
    if keep and parts and isinstance(block.get("cache_control"), dict):
        parts[-1]["cache_control"] = block["cache_control"]
    return parts


def _joined(parts: list[dict], sep: str) -> Any:
    """Parts as one string when they're all text and carry no cache marker; else the parts as they are."""
    if all(p["type"] == "text" and "cache_control" not in p for p in parts):
        return sep.join(p["text"] for p in parts)
    return parts


def _tool_result(block: dict, keep: bool) -> tuple[Any, dict | None]:
    """An Anthropic tool_result as a tool message's content, and the user message that carries its images and
    documents, if any (tool_media)."""
    content = block.get("content")
    marker = block.get("cache_control") if keep and isinstance(block.get("cache_control"), dict) else None
    texts: list[str] = []
    media: list[dict] = []
    if isinstance(content, str):
        texts.append(content)
    for c in content if isinstance(content, list) else []:
        if not isinstance(c, dict):
            continue
        if c.get("type") == "text":
            texts.append(c.get("text", ""))
            if keep and isinstance(c.get("cache_control"), dict):
                marker = c["cache_control"]
        else:
            media += media_parts(c)
    text = "\n".join(texts)
    text, extra = tool_media(f"Error: {text}" if block.get("is_error") else text, block.get("tool_use_id", ""), media)
    return ([{"type": "text", "text": text, "cache_control": marker}] if marker else text), extra


def anthropic_to_openai_request(body: dict, max_tokens_field: str = "max_tokens", keep_cache_control: bool = False) -> dict:
    """``keep_cache_control``: the backend reads Anthropic cache markers on message parts (Claude through OpenRouter), so
    the client's own markers go on the parts they were on, with the lifetimes the client chose."""
    keep = keep_cache_control
    msgs: list[dict] = []
    system = body.get("system")
    if isinstance(system, str) and system:
        msgs.append({"role": "system", "content": system})
    elif isinstance(system, list) and system:
        parts = [p for b in system if isinstance(b, dict) and b.get("type") == "text"
                 for p in _marked([{"type": "text", "text": b.get("text", "")}], b, keep)]
        if parts:
            msgs.append({"role": "system", "content": _joined(parts, "\n\n")})

    for m in body.get("messages") or []:
        role = m.get("role")
        content = m.get("content")
        if role == "system":
            msgs.append({"role": "system", "content": text_of(content, "\n\n")})
            continue
        if isinstance(content, str):
            msgs.append({"role": role, "content": content})
            continue
        content = content or []
        if role == "user":
            parts = []
            attached: list[dict] = []  # tool results' images and documents, after the run of tool messages
            for b in content:
                if b.get("type") == "tool_result":
                    result, extra = _tool_result(b, keep)
                    msgs.append({"role": "tool", "tool_call_id": b.get("tool_use_id", ""), "content": result})
                    if extra:
                        attached.append(extra)
                else:
                    parts += _marked(media_parts(b), b, keep)
            msgs += attached
            if parts:
                msgs.append({"role": "user", "content": _joined(parts, "\n\n")})
        elif role == "assistant":
            texts = [p for b in content if b.get("type") == "text"
                     for p in _marked([{"type": "text", "text": b.get("text", "")}], b, keep)]
            tool_calls = [
                {"id": b.get("id"), "type": "function", "function": {"name": b.get("name"), "arguments": _dumps(b.get("input") or {})}}
                for b in content
                if b.get("type") == "tool_use"
            ]
            if any(p["text"] for p in texts) or tool_calls:
                msg: dict[str, Any] = {"role": "assistant", "content": _joined(texts, "") or None}
                if tool_calls:
                    msg["tool_calls"] = tool_calls
                msgs.append(msg)

    out: dict[str, Any] = {"model": body.get("model"), "messages": msgs}
    if body.get("max_tokens"):
        out[max_tokens_field] = body["max_tokens"]
    if body.get("stop_sequences"):
        out["stop"] = body["stop_sequences"]
    for k in ("temperature", "top_p"):
        if body.get(k) is not None:
            out[k] = body[k]
    if body.get("stream"):
        out["stream"] = True
        out["stream_options"] = {"include_usage": True}

    tools = []
    for t in body.get("tools") or []:
        if "input_schema" not in t:
            continue  # Anthropic server tools (web search, bash, ...) have no OpenAI equivalent
        fn: dict[str, Any] = {"name": t.get("name", ""), "parameters": t["input_schema"]}
        if t.get("description"):
            fn["description"] = t["description"]
        tools.append({"type": "function", "function": fn})
    if tools:
        out["tools"] = tools
        tc = body.get("tool_choice") or {}
        kind = tc.get("type")
        if kind == "any":
            out["tool_choice"] = "required"
        elif kind == "tool":
            out["tool_choice"] = {"type": "function", "function": {"name": tc.get("name", "")}}
        elif kind == "none":
            out["tool_choice"] = "none"
        if tc.get("disable_parallel_tool_use"):
            out["parallel_tool_calls"] = False

    fmt = (body.get("output_config") or {}).get("format") or {}
    if fmt.get("type") == "json_schema" and fmt.get("schema"):
        out["response_format"] = {"type": "json_schema", "json_schema": {"name": "output", "schema": fmt["schema"]}}
    return out


def openai_to_anthropic_response(resp: dict, model: str) -> dict:
    choice = (resp.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content: list[dict] = []
    text = text_of(msg.get("content"))
    if text:
        content.append({"type": "text", "text": text})
    refusal = msg.get("refusal") if isinstance(msg.get("refusal"), str) else None
    if refusal:
        content.append({"type": "text", "text": refusal})
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        content.append({"type": "tool_use", "id": tc.get("id") or _id("toolu"), "name": fn.get("name", ""), "input": _loads(fn.get("arguments"))})
    return {
        "id": f"msg_{resp.get('id') or uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_o2a(choice.get("finish_reason"), bool(msg.get("tool_calls")), bool(refusal)),
        "stop_sequence": None,
        "usage": anthropic_usage(usage_from_openai(resp.get("usage"))),
    }


class OpenAIToAnthropicStream:
    """Converts OpenAI chat completion chunks into an Anthropic Messages event stream."""

    def __init__(self, model: str):
        self.msg_id = _id("msg")
        self.model = model
        self.started = False
        self.block: int | None = None
        self.block_kind: str | None = None
        self.next_index = 0
        self.tool_blocks: dict[int, int] = {}
        self.stop: str | None = None  # the Chat Completions finish_reason
        self.refused = False
        self.usage = Usage()
        self.done = False

    def _start(self) -> SSEEvent:
        self.started = True
        return event(
            {
                "type": "message_start",
                "message": {
                    "id": self.msg_id,
                    "type": "message",
                    "role": "assistant",
                    "model": self.model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            },
            "message_start",
        )

    def _close(self) -> list[SSEEvent]:
        if self.block is None:
            return []
        ev = event({"type": "content_block_stop", "index": self.block}, "content_block_stop")
        self.block = None
        self.block_kind = None
        return [ev]

    def _open(self, kind: str, block: dict) -> tuple[list[SSEEvent], int]:
        out = self._close()
        idx = self.next_index
        self.next_index += 1
        self.block, self.block_kind = idx, kind
        out.append(event({"type": "content_block_start", "index": idx, "content_block": block}, "content_block_start"))
        return out, idx

    def feed(self, ev: SSEEvent) -> list[SSEEvent]:
        if ev.data.strip() == "[DONE]":
            return self.finish_events()
        d = ev.json()
        if not d:
            return []
        if self.done:
            return []  # after an error: Anthropic's stream ends with it
        out: list[SSEEvent] = [] if self.started else [self._start()]
        if d.get("error"):
            err = d["error"]
            message = err.get("message") if isinstance(err, dict) else str(err)
            out.append(event({"type": "error", "error": {"type": "api_error", "message": message}}, "error"))
            self.done = True  # nothing follows an error event, so the SDK doesn't take the reply for a whole one
            return out
        if d.get("usage"):
            self.usage.merge(usage_from_openai(d["usage"]))
        for ch in d.get("choices") or []:
            delta = ch.get("delta") or {}
            text = delta.get("content")
            if isinstance(delta.get("refusal"), str) and delta["refusal"]:
                text = (text or "") + delta["refusal"]  # Claude says a refusal as text, with stop_reason "refusal"
                self.refused = True
            if text:
                if self.block_kind != "text":
                    opened, _ = self._open("text", {"type": "text", "text": ""})
                    out += opened
                out.append(event({"type": "content_block_delta", "index": self.block, "delta": {"type": "text_delta", "text": text}}, "content_block_delta"))
            for tc in delta.get("tool_calls") or []:
                oi = tc.get("index", 0)
                fn = tc.get("function") or {}
                if oi not in self.tool_blocks:
                    opened, idx = self._open("tool", {"type": "tool_use", "id": tc.get("id") or _id("toolu"), "name": fn.get("name", ""), "input": {}})
                    out += opened
                    self.tool_blocks[oi] = idx
                if fn.get("arguments"):
                    out.append(
                        event(
                            {"type": "content_block_delta", "index": self.tool_blocks[oi], "delta": {"type": "input_json_delta", "partial_json": fn["arguments"]}},
                            "content_block_delta",
                        )
                    )
            if ch.get("finish_reason"):
                self.stop = ch["finish_reason"]
        return out

    def finish_events(self) -> list[SSEEvent]:
        if self.done:
            return []
        self.done = True
        out = [] if self.started else [self._start()]
        out += self._close()
        out.append(
            event(
                {"type": "message_delta", "delta": {"stop_reason": stop_o2a(self.stop, bool(self.tool_blocks), self.refused),
                                                    "stop_sequence": None}, "usage": anthropic_usage(self.usage)},
                "message_delta",
            )
        )
        out.append(event({"type": "message_stop"}, "message_stop"))
        return out
