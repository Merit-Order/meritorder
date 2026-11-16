"""Prompt shaping that makes provider cache hits likely without rewriting history.

Rules (see docs/04-caching-and-sessions.md):
* Never change earlier turns. Anything we add goes at the top level or the end.
* If the client already placed cache breakpoints, leave caching to the client.
* A backend set to an hour (``cache_ttl_seconds`` 3600 or more) gets Claude's 1-hour caching from a session's second turn
  on. A first turn gets 5 minutes: a 1-hour write costs 2x the input price instead of 1.25x, and a one-off request
  never reads it back (decision 53).
"""

from __future__ import annotations

import hashlib
from typing import Any

from .config import BackendConfig


def _has_cc(value: Any) -> bool:
    if isinstance(value, dict):
        if "cache_control" in value:
            return True
        return any(_has_cc(v) for k, v in value.items() if k in ("content", "system", "tools", "messages"))
    if isinstance(value, list):
        return any(_has_cc(v) for v in value)
    return False


def has_cache_control(body: dict) -> bool:
    if body.get("cache_control"):
        return True
    return _has_cc(body.get("tools")) or _has_cc(body.get("system")) or _has_cc(body.get("messages"))


HOUR = 3600
FIVE_MINUTES = 300


def _marker(backend: BackendConfig, later_turn: bool) -> dict[str, Any]:
    cc: dict[str, Any] = {"type": "ephemeral"}
    if later_turn and backend.cache_ttl_seconds >= HOUR:
        cc["ttl"] = "1h"
    return cc


def _ttls(value: Any, out: list[int]) -> None:
    if isinstance(value, dict):
        cc = value.get("cache_control")
        if isinstance(cc, dict):
            out.append(HOUR if cc.get("ttl") == "1h" else FIVE_MINUTES)
        for k, v in value.items():
            if k in ("content", "system", "tools", "messages"):
                _ttls(v, out)
    elif isinstance(value, list):
        for v in value:
            _ttls(v, out)


def marker_ttls(body: dict) -> list[int]:
    """How long each cache marker in a request asks Claude to keep its prefix, in prompt order (tools, system,
    messages; a top-level marker applies to the last block, so it comes last)."""
    out: list[int] = []
    for k in ("tools", "system", "messages"):
        _ttls(body.get(k), out)
    if isinstance(body.get("cache_control"), dict):
        _ttls({"cache_control": body["cache_control"]}, out)
    return out


def needs_cache_marker(backend: BackendConfig, upstream_model: str) -> bool:
    """Claude through OpenRouter's OpenAI-style API caches only when asked, unlike the automatic caching
    of OpenAI and most open-model providers. OpenRouter accepts the same top-level ``cache_control``."""
    return "openrouter.ai" in backend.base_url.lower() and upstream_model.lower().startswith("anthropic/")


def shape_anthropic(body: dict, backend: BackendConfig, later_turn: bool = False) -> bool:
    """Turn on automatic prompt caching when the client hasn't set any breakpoints. ``later_turn``: the session has
    had a turn before, so a 1-hour cache is worth its higher write price where the backend is set to one.

    Returns True when Merit Order added caching (counts toward caching savings).
    Automatic caching places one breakpoint on the last cacheable block and moves
    it forward as the conversation grows, which is exactly the agent-loop pattern.
    """
    if has_cache_control(body):
        return False
    body["cache_control"] = _marker(backend, later_turn)
    return True


def shape_openai(body: dict, backend: BackendConfig, session_id: str, later_turn: bool = False) -> bool:
    """OpenAI-style caching is automatic; a stable cache key improves hit rates. Claude through OpenRouter
    is the exception: it gets the same automatic-caching marker as the Anthropic API.

    Returns True only when Merit Order turned caching on; automatic caching would happen anyway.
    """
    if backend.supports_prompt_cache_key and "prompt_cache_key" not in body:
        # OpenAI allows 64 characters. A longer id (a long session id plus a subagent's) is hashed, not cut,
        # so two conversations never share a key.
        body["prompt_cache_key"] = session_id if len(session_id) <= 64 else hashlib.sha256(session_id.encode()).hexdigest()
    if needs_cache_marker(backend, str(body.get("model") or "")) and not has_cache_control(body):
        body["cache_control"] = _marker(backend, later_turn)
        return True
    return False
