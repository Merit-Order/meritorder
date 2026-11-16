"""The Responses API's WebSocket transport (``wss://<gateway>/v1/responses``), as OpenAI's WebSocket mode guide and event
reference define it (read 15 November 2026, openai-python 3.42 to 3.51) and as Codex CLI uses it (decision 54).

A client opens one connection, authenticated by its handshake headers like any request, and sends ``response.create``
events: the body of ``POST /v1/responses``, flat, with ``type``. Each one runs through the same pipeline as the HTTP
endpoint (proxy.Gateway.respond: auth, budgets and caps, routing, sessions, compression, the ledger, translation to
any backend, stored mode), streamed, and its events come back as the SSE stream's, one per message, with the request's
``stream_id`` on each when it named one.

**Chaining on the connection.** A turn may send ``previous_response_id`` and only its new input. Without the team's
stored mode, the gateway keeps, for each stream (lane) of the connection, its latest response's whole conversation (the
input it was made from and its output) in memory, for as long as the connection is open, and nowhere else: a turn
chaining from it is rebuilt from that, as if the client had sent everything, so routing, caching and the ledger see the
same request a client resending its conversation sends. That is what OpenAI keeps for ``store: false`` on a connection,
and what Codex CLI relies on (it always sends ``store: false``). Anything else (an older response, another connection's)
gets ``previous_response_not_found``, and Codex then sends the whole conversation. In stored mode, responses kept by
the gateway chain as over HTTP, from any connection.

**Upstream is HTTP.** The connection ends at the gateway. Each turn is routed on its own and sent the way the chosen
backend takes it (OpenAI's included: its HTTP Responses API, with ``responses_store`` continuing its own stored
response), so ``auto`` may move a turn to another provider, a failover resends everything, and every turn is metered.

Also as OpenAI has it: ``generate: false`` prepares a turn without a model call (Codex's warm-up: no backend, no cost, no
ledger row); ``background`` is refused; up to 32 named streams, each first in, first out, while streams run side by
side; no limit on how long a connection lasts. Codex's ``response.interrupt`` stops the running response it names.
Steering (``response.steer``) isn't supported (error ``unsupported``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time

from fastapi import WebSocket
from fastapi.responses import JSONResponse, StreamingResponse

from .auth import AuthError, Principal
from .formats.responses import request_echo
from .formats.sse import iter_sse

log = logging.getLogger("meritorder.responses_ws")

LANE = re.compile(r"^[A-Za-z0-9_.-]{1,256}$")
MAX_LANES = 32  # OpenAI's own limit of named streams on a connection (the default stream doesn't count)
# response.create messages waiting on one stream behind the one it's running. A client that sends faster than its
# responses finish gets an error event for each one past this, instead of the gateway holding them all in memory.
MAX_QUEUED = 64
ENVELOPE = ("type", "stream_id", "generate")  # WebSocket-only fields, not part of the request body


class ResponsesSocket:
    """One client connection."""

    def __init__(self, gw, ws: WebSocket, max_bytes: int | None = None):
        self.gw, self.ws = gw, ws
        self.headers = {k.lower(): v for k, v in ws.headers.items()}
        self.max_bytes = max_bytes
        # Per stream: its latest response's id and the whole conversation through it, for a turn chaining from it.
        self.latest: dict[str | None, tuple[str, list[dict]]] = {}
        self.queues: dict[str | None, asyncio.Queue] = {}
        self.workers: list[asyncio.Task] = []
        self.jobs: dict[str | None, asyncio.Task] = {}  # the response each stream is making now
        self.running: dict[str | None, str] = {}  # and its id, once it has one (response.interrupt names it)
        self.lock = asyncio.Lock()

    # ---------------------------------------------------------------- the connection

    async def serve(self) -> None:
        try:
            await self.gw.authn.authenticate(self.headers)
        except AuthError as e:
            return await self.deny(e.status, e.message)
        # Codex CLI reads the largest message the server takes, and sends larger ones over HTTP instead.
        await self.ws.accept(headers=[(b"x-codex-websocket-max-message-bytes", str(self.max_bytes).encode())] if self.max_bytes else None)
        self.gw.metrics.inc("meritorder_responses_websocket_total", {"event": "connected"})
        try:
            while True:
                msg = await self.ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                await self.dispatch(msg.get("text") if msg.get("text") is not None else (msg.get("bytes") or b"").decode(errors="replace"))
        finally:
            # The client went away: what's still running stops, recorded as a disconnect (499), and its memory goes.
            for task in self.workers:
                task.cancel()
            await asyncio.gather(*self.workers, return_exceptions=True)
            self.latest.clear()

    async def deny(self, status: int, message: str) -> None:
        body = {"error": {"message": message, "type": "authentication_error", "param": None, "code": status}}
        self.gw.metrics.inc("meritorder_rejected_requests_total", {"status": str(status), "reason": "authentication_error"})
        try:
            await self.ws.send_denial_response(JSONResponse(body, status_code=status))
        except RuntimeError:  # a server without the denial extension: a policy-violation close instead
            await self.ws.close(code=1008, reason=message[:120])

    async def send(self, event: dict) -> None:
        async with self.lock:
            await self.ws.send_text(json.dumps(event, ensure_ascii=False, separators=(",", ":")))

    async def error(self, lane: str | None, status: int, message: str, kind: str = "invalid_request_error", code: str | None = None,
                    param: str | None = None) -> None:
        """OpenAI's error event (ResponseWsError): the status, and the error as the HTTP endpoint would give it."""
        event = {"type": "error", "status": status, "error": {"type": kind, "code": code, "message": message, "param": param}}
        if lane is not None:
            event["stream_id"] = lane
        await self.send(event)

    async def dispatch(self, text: str) -> None:
        try:
            msg = json.loads(text)
        except ValueError:
            msg = None
        if not isinstance(msg, dict):
            return await self.error(None, 400, "each message must be a JSON object with a `type`", code="invalid_json")
        kind = msg.get("type")
        if kind == "response.create":
            lane = msg.get("stream_id")
            if lane is not None and not (isinstance(lane, str) and LANE.match(lane)):
                return await self.error(None, 400, "stream_id must be 1 to 256 letters, digits, '_', '.' or '-'", code="invalid_stream_id",
                                        param="stream_id")
            if lane not in self.queues:
                if lane is not None and sum(k is not None for k in self.queues) >= MAX_LANES:
                    return await self.error(lane, 400, f"a connection takes up to {MAX_LANES} named streams", code="websocket_stream_limit_reached",
                                            param="stream_id")
                self.queues[lane] = asyncio.Queue(maxsize=MAX_QUEUED)
                self.workers.append(asyncio.create_task(self.lane(lane, self.queues[lane])))
            try:
                self.queues[lane].put_nowait(msg)
            except asyncio.QueueFull:
                return await self.error(lane, 429, f"{MAX_QUEUED} responses are already waiting on this stream; send the next one "
                                                   "when one finishes", kind="rate_limit_error", code="websocket_queue_full")
        elif kind == "response.interrupt":
            await self.interrupt(str(msg.get("response_id") or ""))
        elif kind == "response.steer":
            await self.error(None, 400, "steering isn't supported by this gateway: send the input with the next response.create",
                             code="unsupported")
        else:
            await self.error(msg.get("stream_id") if isinstance(msg.get("stream_id"), str) else None, 400,
                             f"unknown event type {str(kind)[:64]!r}: send response.create", code="unknown_event", param="type")

    async def lane(self, lane: str | None, queue: asyncio.Queue) -> None:
        """A stream's responses, one at a time, in the order they came (other streams run meanwhile)."""
        while True:
            msg = await queue.get()
            job = self.jobs[lane] = asyncio.create_task(self.create(lane, msg))
            try:
                await asyncio.wait({job})  # an interrupted job ends here; the stream goes on
            except asyncio.CancelledError:
                job.cancel()
                await asyncio.gather(job, return_exceptions=True)
                raise
            finally:
                self.jobs.pop(lane, None)
                self.running.pop(lane, None)

    async def interrupt(self, rid: str) -> None:
        job = next((self.jobs.get(k) for k, v in self.running.items() if v == rid), None)
        if job is None or job.done():
            return await self.send({"type": "response.interrupt.failed", "response_id": rid,
                                    "error": {"message": f"no response with id '{rid}' is running on this connection"}})
        job.cancel()
        await asyncio.gather(job, return_exceptions=True)
        await self.send({"type": "response.interrupt.accepted", "response_id": rid})

    # ---------------------------------------------------------------- one response

    async def create(self, lane: str | None, msg: dict) -> None:
        try:
            await self._create(lane, msg)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 (a broken connection, or a bug: the connection's other streams go on)
            log.exception("a Responses WebSocket turn failed")
            try:
                await self.error(lane, 500, "the gateway failed while making this response", "api_error", code="server_error")
            except Exception:  # noqa: BLE001
                pass

    def cached(self, rid: str) -> list[dict] | None:
        return next((items for k, (r, items) in self.latest.items() if r == rid), None)

    def evict(self, rid: str | None) -> None:
        """A continuation that failed drops what it chained from, as OpenAI does: the client sends everything next."""
        for k, (r, _) in list(self.latest.items()):
            if r == rid:
                del self.latest[k]

    async def _create(self, lane: str | None, msg: dict) -> None:
        from .stored.service import StoredError, normalize_input

        t0 = time.monotonic()
        body = {k: v for k, v in msg.items() if k not in ENVELOPE}
        if body.get("background"):
            return await self.error(lane, 400, "background isn't supported over WebSocket: use POST /v1/responses", param="background")
        try:
            # Again for each turn: a revoked key or a disabled person stops at the next one, not when the connection ends.
            principal = await self.gw.authn.authenticate(self.headers)
        except AuthError as e:
            return await self.error(lane, e.status, e.message, "authentication_error")
        prev = body.get("previous_response_id")
        history = self.cached(prev) if isinstance(prev, str) else None
        team = self.gw.cfg.team(principal.team)
        if prev and history is None and not team.stored_responses.enabled:
            return await self.error(lane, 400, f"Previous response with id '{prev}' not found. Without stored responses (off for team "
                                               f"'{team.name}'), a turn continues only the latest response of its stream on the same "
                                               "connection: send the whole conversation in `input`.",
                                    code="previous_response_not_found", param="previous_response_id")
        try:
            items = normalize_input(body.get("input"))
        except StoredError as e:
            return await self.error(lane, e.status, e.message, e.kind, param=e.param)
        if history is not None:
            # Rebuilt from the connection's memory: from here on it's a turn that sends its whole conversation.
            body.pop("previous_response_id")
            items = history + items
        body["input"], body["stream"] = items, True
        if msg.get("generate") is False:
            return await self.prepare(lane, principal, body, prev if history is not None else None)
        resp = await self.gw.respond(principal, body, self.headers, "openai", t0, wire="responses")
        if not isinstance(resp, StreamingResponse):  # refused before anything was sent upstream, or every route failed
            err = (json.loads(bytes(resp.body)).get("error") or {}) if resp.body else {}
            # A stored chain that isn't there (any 404 of a turn chaining from what the gateway stored): Codex then resends all.
            code = "previous_response_not_found" if resp.status_code == 404 and prev and history is None else None
            self.evict(prev)
            return await self.error(lane, resp.status_code, err.get("message") or f"HTTP {resp.status_code}", err.get("type") or "api_error",
                                    code=err.get("code") if isinstance(err.get("code"), str) else code, param=err.get("param"))
        self.gw.metrics.inc("meritorder_responses_websocket_total", {"event": "local_chain" if history is not None else "response"})
        final = None
        events = resp.body_iterator
        try:
            async for ev in iter_sse(events):
                data = ev.json()
                if data is None:
                    continue
                obj = data.get("response")
                if isinstance(obj, dict):
                    if history is not None:
                        obj["previous_response_id"] = prev  # what the client chained from, though it went whole
                    if data.get("type") == "response.created":
                        self.running[lane] = obj.get("id")
                    if data.get("type") in ("response.completed", "response.incomplete"):
                        final = obj
                if lane is not None:
                    data["stream_id"] = lane
                if data.get("type") in ("response.completed", "response.incomplete", "response.failed"):
                    self.remember(lane, final, items)  # before the client sees the end: it may chain at once
                await self.send(data)
        finally:
            await events.aclose()  # type: ignore[attr-defined]

    def remember(self, lane: str | None, final: dict | None, items: list[dict]) -> None:
        """Keep a finished response for the next turn of its stream, unless the gateway stored it (stored mode chains it
        from what it stored, anywhere). A response that failed isn't continued."""
        self.latest.pop(lane, None)
        if final is not None and final.get("store") is False:
            self.latest[lane] = (final["id"], items + list(final.get("output") or []))

    async def prepare(self, lane: str | None, principal: Principal, body: dict, chained: str | None) -> None:
        """``generate: false``: the turn's state, ready to chain from, without a model call. What it builds on is checked as
        for a real turn; it's kept on this connection only."""
        from .formats.responses import ResponsesError, responses_to_chat
        from .stored.service import StoredError, new_id

        items = body["input"]
        try:
            turn = await self.gw.stored.begin(principal, {**body, "store": False})
            if turn is not None:
                items = turn.history + turn.input
            responses_to_chat({**body, "input": items, "previous_response_id": None, "conversation": None})
        except StoredError as e:
            return await self.error(lane, e.status, e.message, e.kind,
                                    code="previous_response_not_found" if e.status == 404 and e.param == "previous_response_id" else None,
                                    param=e.param)
        except ResponsesError as e:
            return await self.error(lane, 400, str(e))
        obj = {"id": new_id("resp"), "object": "response", "created_at": int(time.time()), "status": "in_progress",
               "model": str(body.get("model") or ""), "output": [], "usage": None, "error": None, "incomplete_details": None,
               **request_echo(body), "store": False}
        if chained:
            obj["previous_response_id"] = chained
        done = {**obj, "status": "completed", "completed_at": obj["created_at"],
                "usage": {"input_tokens": 0, "input_tokens_details": {"cached_tokens": 0}, "output_tokens": 0,
                          "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 0}}
        self.remember(lane, done, items)
        self.gw.metrics.inc("meritorder_responses_websocket_total", {"event": "prepared"})
        for seq, (kind, value) in enumerate((("response.created", obj), ("response.completed", done)), 1):
            event = {"type": kind, "sequence_number": seq, "response": value}
            if lane is not None:
                event["stream_id"] = lane
            await self.send(event)
