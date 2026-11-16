"""Exact-match response cache (opt-in).

Useful for chatbots, RAG, evals, and repeated pipeline calls; rarely hits for
coding agents, where every turn is unique. This is the one feature that stores
response content: entries live in Redis with a TTL, scoped to a team or a user
so cached answers never cross that boundary.

Enable per team (``response_cache: {enabled: true}``) or per request with
``X-MeritOrder-Cache: on``; ``off`` always bypasses it.
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel

from .formats.sse import SSEEvent, event

_IGNORED = {"stream", "stream_options", "metadata", "user"}


class ResponseCacheConfig(BaseModel):
    enabled: bool = False
    ttl_seconds: int = 3600
    scope: Literal["team", "user"] = "team"


def wanted(cfg: ResponseCacheConfig, headers: dict[str, str]) -> bool:
    h = (headers.get("x-meritorder-cache") or "").lower()
    if h == "off":
        return False
    return h == "on" or cfg.enabled


def cache_key(fmt: str, body: dict, team: str, user: str, cfg: ResponseCacheConfig) -> str:
    scope = team if cfg.scope == "team" else f"{team}/{user}"
    canon = {k: v for k, v in body.items() if k not in _IGNORED}
    raw = json.dumps([fmt, scope, canon], sort_keys=True, ensure_ascii=False, default=str)
    digest = hashlib.sha256(raw.encode("utf-8", "surrogatepass")).hexdigest()  # half an emoji is hashed too
    return f"rc:{digest}"


def replay_stream(fmt: str, resp: dict) -> list[SSEEvent]:
    """Turn a stored complete response into a stream in the client's format."""
    if fmt == "anthropic":
        message = {**resp, "content": []}
        events = [event({"type": "message_start", "message": message}, "message_start")]
        for i, block in enumerate(resp.get("content") or []):
            if block.get("type") == "text":
                events.append(event({"type": "content_block_start", "index": i, "content_block": {"type": "text", "text": ""}}, "content_block_start"))
                events.append(event({"type": "content_block_delta", "index": i, "delta": {"type": "text_delta", "text": block.get("text", "")}}, "content_block_delta"))
            elif block.get("type") == "tool_use":
                events.append(event({"type": "content_block_start", "index": i, "content_block": {**block, "input": {}}}, "content_block_start"))
                events.append(event({"type": "content_block_delta", "index": i, "delta": {"type": "input_json_delta", "partial_json": json.dumps(block.get("input") or {})}}, "content_block_delta"))
            else:
                continue
            events.append(event({"type": "content_block_stop", "index": i}, "content_block_stop"))
        events.append(event({"type": "message_delta", "delta": {"stop_reason": resp.get("stop_reason"), "stop_sequence": None},
                             "usage": {"output_tokens": (resp.get("usage") or {}).get("output_tokens", 0)}}, "message_delta"))
        events.append(event({"type": "message_stop"}, "message_stop"))
        return events
    choice = (resp.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    base = {"id": resp.get("id"), "object": "chat.completion.chunk", "created": resp.get("created"), "model": resp.get("model")}
    delta: dict = {"role": "assistant", "content": msg.get("content") or ""}
    if msg.get("tool_calls"):
        delta["tool_calls"] = [{**tc, "index": i} for i, tc in enumerate(msg["tool_calls"])]
    return [
        event({**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}),
        event({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": choice.get("finish_reason") or "stop"}]}),
        SSEEvent(data="[DONE]"),
    ]
