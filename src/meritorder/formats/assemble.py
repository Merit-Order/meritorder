"""Turning a provider's event stream back into the single response a non-streaming request expects.

A long reply can take longer than an HTTP request should stay silent: Anthropic's SDKs refuse to send a
non-streaming request that could run past 10 minutes, and networks drop idle connections. So when a
non-streaming request may run long, the gateway streams from the provider and assembles the reply here.
"""

from __future__ import annotations

import json
from typing import Any


def _flatten(content: Any) -> tuple[str, str] | None:
    """(text, thinking) from content sent as typed chunks, or None if it isn't only text and thinking chunks."""
    if not isinstance(content, list) or not content:
        return None
    text, thinking = [], []
    for c in content:
        if not isinstance(c, dict) or c.get("type") not in ("text", "thinking"):
            return None
        if c["type"] == "text":
            text.append(str(c.get("text") or ""))
            continue
        parts = c.get("thinking")
        for t in parts if isinstance(parts, list) else [parts]:
            thinking.append(str(t.get("text") or "") if isinstance(t, dict) else str(t or ""))
    return "".join(text), "".join(thinking)


def plain_content(body: dict) -> dict:
    """OpenAI's format has a message's content (and a stream delta's) as a string. Mistral's API sends typed chunks
    instead (``[{"type": "thinking", "thinking": [...]}, {"type": "text", "text": "OK"}]``), which clients don't expect:
    the text becomes the content and the thinking ``reasoning_content``, as the gateway does for Claude's. Content
    with any other kind of chunk is left as it is. Changes ``body`` in place and returns it."""
    for choice in body.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        for key in ("message", "delta"):
            msg = choice.get(key)
            found = _flatten(msg.get("content")) if isinstance(msg, dict) else None
            if found is None:
                continue
            text, thinking = found
            if key == "delta" and not text:
                msg.pop("content")
            else:
                msg["content"] = text if text or not msg.get("tool_calls") else None
            if thinking:
                msg["reasoning_content"] = (msg.get("reasoning_content") or "") + thinking
    return body


def anthropic_message(events: list[dict]) -> dict:
    """A Messages API response from its stream events (message_start, content_block_*, message_delta)."""
    msg: dict[str, Any] = {}
    blocks: dict[int, dict] = {}
    partial: dict[int, str] = {}
    for d in events:
        kind = d.get("type")
        if kind == "message_start":
            msg = dict(d.get("message") or {})
            msg["usage"] = dict(msg.get("usage") or {})
        elif kind == "content_block_start":
            i = int(d.get("index", len(blocks)))
            blocks[i] = dict(d.get("content_block") or {})
        elif kind == "content_block_delta":
            i = int(d.get("index", 0))
            b = blocks.setdefault(i, {})
            delta = d.get("delta") or {}
            dt = delta.get("type")
            if dt == "text_delta":
                b["text"] = b.get("text", "") + delta.get("text", "")
            elif dt == "thinking_delta":
                b["thinking"] = b.get("thinking", "") + delta.get("thinking", "")
            elif dt == "signature_delta":
                b["signature"] = b.get("signature", "") + delta.get("signature", "")
            elif dt == "input_json_delta":
                partial[i] = partial.get(i, "") + delta.get("partial_json", "")
            elif dt == "citations_delta" and delta.get("citation") is not None:
                b.setdefault("citations", []).append(delta["citation"])
        elif kind == "content_block_stop":
            i = int(d.get("index", 0))
            if i in partial:
                try:
                    blocks[i]["input"] = json.loads(partial.pop(i) or "{}")
                except json.JSONDecodeError:
                    blocks[i]["input"] = {}
        elif kind == "message_delta":
            for k, v in (d.get("delta") or {}).items():
                msg[k] = v
            msg.setdefault("usage", {}).update({k: v for k, v in (d.get("usage") or {}).items() if v is not None})
    for i, raw in partial.items():  # a stream cut off inside a tool call
        try:
            blocks[i]["input"] = json.loads(raw or "{}")
        except json.JSONDecodeError:
            blocks[i]["input"] = {}
    msg["content"] = [blocks[i] for i in sorted(blocks)]
    msg.setdefault("type", "message")
    msg.setdefault("role", "assistant")
    return msg


def _merge_details(parts: list[dict]) -> list[dict]:
    """OpenRouter's reasoning_details arrive in pieces (one per chunk, sharing an index and type); a reply carries
    them whole. Gemini needs them sent back unchanged with the next turn's tool results, signatures included."""
    out: dict[tuple, dict] = {}
    for d in parts:
        key = (d.get("index", 0), d.get("type"))
        cur = out.setdefault(key, {})
        for k, v in d.items():
            if k in ("text", "summary", "data") and isinstance(v, str):
                cur[k] = cur.get(k, "") + v
            elif v is not None:
                cur[k] = v
    return list(out.values())


def openai_completion(chunks: list[dict]) -> dict:
    """A Chat Completions response from its stream chunks (content, reasoning and tool-call deltas, usage)."""
    out: dict[str, Any] = {"object": "chat.completion", "choices": []}
    choices: dict[int, dict] = {}
    for c in chunks:
        for k in ("id", "created", "model", "system_fingerprint", "provider", "service_tier"):
            if c.get(k) is not None:
                out[k] = c[k]
        if c.get("usage"):
            out["usage"] = c["usage"]
        for ch in c.get("choices") or []:
            i = int(ch.get("index", 0))
            slot = choices.setdefault(i, {"index": i, "message": {"role": "assistant", "content": None}, "finish_reason": None, "_calls": {}})
            msg = slot["message"]
            delta = ch.get("delta") or {}
            for k in ("content", "reasoning_content", "reasoning", "refusal"):
                if isinstance(delta.get(k), str) and delta[k]:
                    msg[k] = (msg.get(k) or "") + delta[k]
            if delta.get("reasoning_details"):
                msg.setdefault("reasoning_details", []).extend(delta["reasoning_details"])
            for tc in delta.get("tool_calls") or []:
                j = int(tc.get("index", len(slot["_calls"])))
                call = slot["_calls"].setdefault(j, {"id": None, "type": "function", "function": {"name": "", "arguments": ""}})
                if tc.get("id"):
                    call["id"] = tc["id"]
                if tc.get("extra_content"):  # Gemini's thought signature, which must go back with the call
                    call["extra_content"] = tc["extra_content"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    call["function"]["name"] += fn["name"]
                if fn.get("arguments"):
                    call["function"]["arguments"] += fn["arguments"]
            if ch.get("finish_reason"):
                slot["finish_reason"] = ch["finish_reason"]
    for i in sorted(choices):
        slot = choices[i]
        calls = slot.pop("_calls")
        if slot["message"].get("reasoning_details"):
            slot["message"]["reasoning_details"] = _merge_details(slot["message"]["reasoning_details"])
        if calls:
            slot["message"]["tool_calls"] = [calls[j] for j in sorted(calls)]
        elif slot["message"]["content"] is None:
            slot["message"]["content"] = ""
        out["choices"].append(slot)
    if not out["choices"]:
        out["choices"] = [{"index": 0, "message": {"role": "assistant", "content": ""}, "finish_reason": None}]
    return out
