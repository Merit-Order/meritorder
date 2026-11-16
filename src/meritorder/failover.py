"""Catching provider failures that arrive after a 200, so the request can still go to another source.

Anthropic reports overload inside a stream that has already started (``event: error`` with
``overloaded_error``), and OpenAI-compatible providers (OpenRouter in particular) send ``{"error": ...}``
chunks, or a 200 JSON body that is only an error. Nothing has reached the client at that point, so the
gateway reads the start of each stream (up to its first content) before committing to it, and turns these
into a :class:`BackendError` the fallback loop handles like any other failure.
"""

from __future__ import annotations

import asyncio
import time

import httpx

from .backends.base import BackendError, UpstreamResponse
from .formats.sse import SSEEvent

_TYPE_STATUS = {"overloaded_error": 529, "rate_limit_error": 429, "api_error": 500, "server_error": 500, "timeout_error": 504,
                "service_unavailable": 503, "service_unavailable_error": 503, "insufficient_quota": 429, "not_found_error": 404,
                "authentication_error": 401, "permission_error": 403, "invalid_request_error": 400,
                "rate_limit_exceeded": 429, "model_not_found": 404}
_PEEK_LIMIT = 200  # events; a stream that hasn't shown content by then is committed to anyway


def _status(err: dict) -> int:
    code = err.get("code")
    if isinstance(code, int) and 400 <= code < 600:
        return code
    if isinstance(code, str) and code.isdigit() and 400 <= int(code) < 600:
        return int(code)
    for key in ("code", "type"):  # OpenAI's code is the more specific one: type "tokens", code "rate_limit_exceeded"
        status = _TYPE_STATUS.get(str(err.get(key) or ""))
        if status:
            return status
    return 502


def error_in(data: dict | None, fmt: str) -> tuple[int, str] | None:
    """The provider's error in a stream event or a 200 body, as (status, message); None if it isn't one."""
    if not isinstance(data, dict):
        return None
    if fmt == "anthropic":
        if data.get("type") != "error":
            return None
        err = data.get("error") if isinstance(data.get("error"), dict) else {}
        return _status(err), str(err.get("message") or "upstream error")
    err = data.get("error")
    if err and not data.get("choices") and not data.get("data"):
        err = err if isinstance(err, dict) else {"message": str(err)}
        return _status(err), str(err.get("message") or "upstream error")
    choice = (data.get("choices") or [{}])[0] if isinstance(data.get("choices"), list) and data.get("choices") else {}
    if choice.get("finish_reason") == "error":  # OpenRouter: the upstream provider failed mid-request
        err = data.get("error") if isinstance(data.get("error"), dict) else {}
        return _status(err), str(err.get("message") or "upstream provider error")
    return None


def _started(ev: SSEEvent, fmt: str) -> bool:
    """Content (or the end) has arrived: from here on, the stream is the client's."""
    if ev.data.strip() == "[DONE]":
        return True
    d = ev.json() or {}
    if fmt == "anthropic":
        return d.get("type") in ("content_block_delta", "message_delta", "message_stop")
    choices = d.get("choices") or []
    if not choices:
        return bool(d.get("usage"))
    c = choices[0] or {}
    delta = c.get("delta") or {}
    return bool(c.get("finish_reason") or any(delta.get(k) for k in ("content", "tool_calls", "reasoning", "reasoning_content",
                                                                      "refusal", "function_call")))


def check_body(resp: UpstreamResponse, fmt: str, backend: str) -> None:
    found = error_in(resp.body, fmt)
    if found:
        status, message = found
        raise BackendError(status, f"{backend}: {message}", body=resp.body, retryable=status >= 500 or status in (408, 429))


async def start_stream(resp: UpstreamResponse, fmt: str, backend: str, first_event_timeout: float | None) -> UpstreamResponse:
    """Read a stream up to its first content. An error before then, or no event at all within
    ``first_event_timeout`` seconds, raises BackendError (after closing the upstream stream)."""
    assert resp.events is not None
    it = resp.events.__aiter__()
    buffered: list[SSEEvent] = []
    deadline = time.monotonic() + first_event_timeout if first_event_timeout else None
    try:
        while len(buffered) < _PEEK_LIMIT:
            try:
                if deadline is not None and not buffered:
                    ev = await asyncio.wait_for(it.__anext__(), max(0.001, deadline - time.monotonic()))
                else:
                    ev = await it.__anext__()
            except StopAsyncIteration:
                break
            buffered.append(ev)
            found = error_in(ev.json(), fmt)
            if found:
                status, message = found
                raise BackendError(status, f"{backend}: {message}", retryable=status >= 500 or status in (408, 429))
            if _started(ev, fmt):
                break
    except asyncio.TimeoutError:
        await resp.aclose()
        raise BackendError(504, f"{backend}: no response within {first_event_timeout:.0f}s", retryable=True)
    except httpx.HTTPError as e:
        await resp.aclose()
        raise BackendError(502, f"{backend}: stream broke before any content: {type(e).__name__}: {e}", retryable=True)
    except BackendError:
        await resp.aclose()
        raise

    async def replay():
        for ev in buffered:
            yield ev
        async for ev in it:
            yield ev

    resp.events = replay()
    return resp
