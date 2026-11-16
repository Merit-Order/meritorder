"""The Responses API's stored mode in the gateway (decision 51): who may read what, rebuilding a turn's whole input from
what was stored, saving responses and conversation items sealed, background responses, and deleting it all again.

A turn that chains (``previous_response_id``) or belongs to a ``conversation`` is rebuilt here into a stateless request:
the stored items before it (each earlier turn's input and output, or the conversation's items) followed by its own input.
The rest of the pipeline sees only that rebuilt request, so routing, budgets, the ledger, sessions, prompt caching,
compression and the learned router treat it as they treat a client that sends the whole conversation every turn, and
``auto`` may send it to another model or provider than the turn before. The client's input is stored as it was sent,
before compression, so rebuilding the next turn gives the same history and compression's stored decisions apply to it
the same way.

When the next turn lands again on a provider's Responses API that stored the previous turn (a backend with
``responses_store``, the same model), it goes as that response's ``previous_response_id`` with only the new input
(proxy.Gateway.build_upstream), keeping the provider's reasoning items; the gateway still holds the whole conversation, so
any other route, or a provider that has lost it, gets everything.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import secrets
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from ..auth import Principal
from ..config import TeamConfig
from ..formats.responses import ResponsesError, request_echo, responses_to_chat
from ..sealing import Keyring, SealError
from .store import RUNNING

log = logging.getLogger("meritorder.stored")

DONE = ("completed", "incomplete")  # responses another turn can chain from
RESEAL_BATCH = 500  # values sealed again with the current key per pass of the worker
UNKNOWN_KEY_ALERT_SECONDS = 40 * 86400  # one alert per unknown seal key (again after this, if it's still there)
PURGE_SECONDS = 60  # how often a gateway without a worker (in-memory storage) deletes what's past its retention


class StoredError(Exception):
    def __init__(self, status: int, message: str, kind: str = "invalid_request_error", param: str | None = None):
        super().__init__(message)
        self.status, self.message, self.kind, self.param = status, message, kind, param


def new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(24)}"


ITEM_PREFIX = {"message": "msg", "function_call": "fc", "function_call_output": "fco", "reasoning": "rs"}


def normalize_input(value: Any) -> list[dict]:
    """A request's ``input`` as the items OpenAI stores: a string becomes a user message, every item gets a type and an
    id, and message text becomes content parts, so input items read back the way OpenAI returns them."""
    items = [{"role": "user", "content": value}] if isinstance(value, str) else list(value or [])
    out = []
    for raw in items:
        if not isinstance(raw, dict):
            raise StoredError(400, "each input item must be an object", param="input")
        item = copy.deepcopy(raw)
        kind = item.setdefault("type", "message")
        if kind == "message":
            role = item.get("role", "user")
            if isinstance(item.get("content"), str):
                part = "output_text" if role == "assistant" else "input_text"
                item["content"] = [{"type": part, "text": item["content"], **({"annotations": []} if role == "assistant" else {})}]
            item.setdefault("status", "completed")
        item.setdefault("id", new_id(ITEM_PREFIX.get(kind, "item")))
        out.append(item)
    return out


@dataclass
class StoredTurn:
    """One /v1/responses request in stored mode, from when it's read until its response is saved."""

    principal: Principal
    team: TeamConfig
    rid: str
    store: bool  # keep this response (the client's `store`, default true)
    background: bool
    input: list[dict]  # this turn's input items, as sent (with ids)
    history: list[dict] = field(default_factory=list)  # the items before them: the chain's or the conversation's
    echo: dict = field(default_factory=dict)  # what the response object repeats from the request (request_echo)
    previous: dict | None = None  # the previous response's row (no content)
    conversation: dict | None = None  # the conversation's row (no content)
    conversation_upto: int | None = None  # the conversation's last item before this turn
    session_id: str | None = None  # the chain's or conversation's session, unless the client names one
    thread_id: str = ""
    # The provider's copy of the previous turn ({backend, model, id}) and where the new input starts in the rebuilt
    # messages: a turn that lands on that backend and model again goes as its continuation there.
    provider_previous: dict | None = None
    held: int = 0
    created: int = field(default_factory=lambda: int(time.time()))
    model: str = ""

    @property
    def extra(self) -> dict:
        """What the response object carries besides its output: the request's settings, store, conversation."""
        out = {**self.echo, "store": self.store}
        if self.conversation is not None:
            out["conversation"] = {"id": self.conversation["id"]}
        return out


class StoredResponses:
    def __init__(self, gw):
        self.gw = gw
        self._running: dict[str, asyncio.Task] = {}  # background responses this process is working on

    # ---------------------------------------------------------------- keys and access

    @property
    def store(self):
        return self.gw.store

    def keyring(self) -> Keyring:
        rs = self.gw.cfg.response_store
        return Keyring(rs.key(), rs.old_keys(), self.store.kind == "memory")

    def _sealer(self) -> Keyring:
        ring = self.keyring()
        if not ring.usable:
            raise StoredError(503, f"stored responses need a key: set {self.gw.cfg.response_store.key_env} (the same on every gateway "
                                   "and the worker) to keep or read them", "api_error")
        return ring

    def seal(self, value: Any, aad: str) -> tuple[bytes, str]:
        ring = self._sealer()
        return ring.seal(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode(), aad.encode()), ring.current or ""

    def open(self, blob: bytes, aad: str) -> Any:
        try:
            return json.loads(self._sealer().open(bytes(blob), aad.encode()))
        except SealError as e:
            raise StoredError(500, f"a stored value can't be read: {e}", "api_error") from None

    @staticmethod
    def response_aad(row: dict) -> str:
        return f"response:{row['id']}:{row['user_id']}:{row['team']}"

    @staticmethod
    def conversation_aad(row: dict) -> str:
        return f"conversation:{row['id']}:{row['user_id']}:{row['team']}"

    @staticmethod
    def item_aad(conv: dict, item_id: str) -> str:
        return f"item:{conv['id']}:{item_id}:{conv['user_id']}:{conv['team']}"

    def retention(self, team: str) -> timedelta | None:
        """How long the team keeps stored content after a chain's last turn; None while its stored mode is off (then
        nothing of it is kept: what's there is deleted at the next purge)."""
        s = self.gw.cfg.team(team).stored_responses
        return timedelta(days=s.retention_days) if s.enabled and s.retention_days else None

    def expired(self, row: dict, now: datetime | None = None) -> bool:
        keep, now = self.retention(row["team"]), now or datetime.now(UTC)
        return keep is None or row["touched_at"] < now - keep or (row.get("expires_at") is not None and row["expires_at"] < now)

    def owns(self, p: Principal, row: dict) -> bool:
        return p.user_id == row["user_id"] and p.team == row["team"]

    def may_read(self, p: Principal, row: dict) -> bool:
        """The person and team that made it; besides them only roles the team lists in ``readers`` (admins from any
        team, the others on the same team). Admins aren't readers unless listed."""
        if self.owns(p, row):
            return True
        readers = self.gw.cfg.team(row["team"]).stored_responses.readers
        return p.role in readers and (p.role == "admin" or p.team == row["team"])

    async def response_row(self, p: Principal, rid: str, owner: bool = False) -> dict:
        """A stored response the caller may read (``owner``: one they made), or 404, as OpenAI says it."""
        row = await self.store.get_stored_response(rid) if rid else None
        if row is None or self.expired(row) or not (self.owns(p, row) if owner else self.may_read(p, row)):
            raise StoredError(404, f"Response with id '{rid}' not found.", "not_found_error")
        return row

    async def conversation_row(self, p: Principal, cid: str, owner: bool = False) -> dict:
        row = await self.store.get_conversation(cid) if cid else None
        if row is None or self.expired(row) or not (self.owns(p, row) if owner else self.may_read(p, row)):
            raise StoredError(404, f"Conversation with id '{cid}' not found.", "not_found_error")
        return row

    def require_on(self, team: TeamConfig, what: str) -> None:
        if not team.stored_responses.enabled:
            raise StoredError(400, f"{what} needs stored responses, which are off for team '{team.name}': send the whole "
                                   "conversation in `input` with store: false, or ask an admin to turn on stored responses for the "
                                   "team (Config → Teams)", param=None)

    # ---------------------------------------------------------------- reading what's stored

    def response_object(self, row: dict, content: dict | None = None) -> dict:
        """The response object as the client got it, with its status as the gateway last recorded it (a background
        response cancelled, or failed because the gateway running it stopped)."""
        content = content if content is not None else self.open(row["content"], self.response_aad(row))
        obj = content["response"]
        if row["status"] != obj.get("status"):
            obj["status"] = row["status"]
            if row["status"] == "failed" and not obj.get("error"):
                obj["error"] = {"code": "server_error", "message": "the gateway running this background response stopped before it finished"}
        return obj

    async def chain_items(self, p: Principal, rid: str) -> tuple[list[dict], list[dict]]:
        """The items a turn chaining from ``rid`` builds on (every response of the chain's input then output, oldest
        first), and the chain's rows, newest first. Every response in it must still be stored and readable."""
        rows = await self.store.stored_chain(rid)
        if not rows or rows[0]["id"] != rid:
            raise StoredError(404, f"Previous response with id '{rid}' not found.", "not_found_error", param="previous_response_id")
        items: list[dict] = []
        for i, row in enumerate(rows):
            # A background response sent with store: false is kept only to be polled, never to chain from.
            if self.expired(row) or not self.may_read(p, row) or row.get("expires_at") is not None:
                what = f"Previous response with id '{rid}'" if i == 0 else f"An earlier response in this chain ('{row['id']}')"
                raise StoredError(404, f"{what} not found.", "not_found_error", param="previous_response_id")
            if i == 0 and row["status"] not in DONE:
                raise StoredError(400, f"Previous response with id '{rid}' is {row['status']}: chain from a completed response.",
                                  param="previous_response_id")
            content = self.open(row["content"], self.response_aad(row))
            items[:0] = list(content.get("input") or []) + list(self.response_object(row, content).get("output") or [])
        last = rows[-1]
        if last["previous_response_id"]:  # the chain goes further back than what's stored: something in it was deleted
            raise StoredError(404, f"An earlier response in this chain ('{last['previous_response_id']}') was deleted or has expired.",
                              "not_found_error", param="previous_response_id")
        return items, rows

    async def input_items(self, p: Principal, row: dict) -> list[dict]:
        """Everything a stored response was made from, oldest first, as OpenAI lists it: the earlier turns of its chain (or
        its conversation's items before it) and then its own input. What has been deleted since is left out."""
        content = self.open(row["content"], self.response_aad(row))
        before: list[dict] = []
        if row["previous_response_id"]:
            for r in await self.store.stored_chain(row["previous_response_id"]):
                if self.expired(r) or not self.may_read(p, r):
                    break
                c = self.open(r["content"], self.response_aad(r))
                before[:0] = list(c.get("input") or []) + list(self.response_object(r, c).get("output") or [])
        elif row["conversation_id"] and row["conversation_upto"]:
            conv = await self.store.get_conversation(row["conversation_id"])
            if conv is not None and not self.expired(conv) and self.may_read(p, conv):
                rows = await self.store.conversation_items(conv["id"], upto=row["conversation_upto"])
                before = [self.open(r["content"], self.item_aad(conv, r["item_id"])) for r in rows]
        return before + list(content.get("input") or [])

    # ---------------------------------------------------------------- a turn

    async def begin(self, p: Principal, body: dict) -> StoredTurn | None:
        """Read a /v1/responses request: None for a stateless one (stored mode off or ``store: false``, and nothing to
        chain from), else the turn with the history it builds on."""
        team = self.gw.cfg.team(p.team)
        prev_id, conv_ref = body.get("previous_response_id"), body.get("conversation")
        cid = conv_ref.get("id") if isinstance(conv_ref, dict) else conv_ref
        background, store = bool(body.get("background")), body.get("store") is not False
        if prev_id and cid:
            raise StoredError(400, "previous_response_id and conversation can't be used together.", param="conversation")
        if prev_id:
            self.require_on(team, "previous_response_id")
        if cid:
            self.require_on(team, "conversation")
        if background:
            self.require_on(team, "background")  # with store: false it's kept only until it has been polled
            if body.get("stream"):
                raise StoredError(400, "background responses can't be streamed here: poll GET /v1/responses/{id} instead.", param="stream")
        refs = isinstance(body.get("input"), list) and any(isinstance(i, dict) and i.get("type") == "item_reference" for i in body["input"])
        if refs:
            self.require_on(team, "item_reference")
        if not (prev_id or cid or background or refs) and not (store and team.stored_responses.enabled):
            return None
        self._sealer()  # no key: refuse now, not after the model has answered
        turn = StoredTurn(p, team, new_id("resp"), store, background, normalize_input(body.get("input")), echo=request_echo(body),
                          model=str(body.get("model") or ""))
        if prev_id:
            history, rows = await self.chain_items(p, str(prev_id))
            prev = rows[0]
            turn.previous, turn.history, turn.thread_id, turn.session_id = prev, history, prev["thread_id"], prev["session_id"]
            if prev.get("upstream_id"):
                turn.provider_previous = {"backend": prev["backend"], "model": prev["upstream_model"], "id": prev["upstream_id"]}
        elif cid:
            conv = await self.conversation_row(p, str(cid))
            items = await self.store.conversation_items(conv["id"])
            turn.conversation, turn.session_id = conv, conv["session_id"]
            turn.conversation_upto = items[-1]["seq"] if items else 0
            turn.history = [self.open(r["content"], self.item_aad(conv, r["item_id"])) for r in items]
        if refs:
            turn.input = await self.resolve(p, turn.input, turn.history)
        turn.thread_id = turn.thread_id or turn.rid
        return turn

    async def resolve(self, p: Principal, items: list[dict], history: list[dict]) -> list[dict]:
        """Replace ``item_reference`` items with the items they name: from the history this turn builds on, else from what
        the person has stored (the Vercel AI SDK sends earlier turns' items this way when store is on)."""
        known = {h.get("id"): h for h in history if h.get("id")}
        missing = [i["id"] for i in items if i.get("type") == "item_reference" and i.get("id") not in known]
        if missing:
            responses, conv_items = await self.store.stored_with_items(p.user_id, missing)
            for row in responses:
                if self.expired(row) or not self.may_read(p, row):
                    continue
                content = self.open(row["content"], self.response_aad(row))
                for item in list(content.get("input") or []) + list(self.response_object(row, content).get("output") or []):
                    known.setdefault(item.get("id"), item)
            for row in conv_items:
                conv = row["conversation"]
                if not self.expired(conv) and self.may_read(p, conv):
                    known.setdefault(row["item_id"], self.open(row["content"], self.item_aad(conv, row["item_id"])))
        out = []
        for i in items:
            if i.get("type") != "item_reference":
                out.append(i)
            elif i.get("id") in known:
                out.append(known[i["id"]])
            else:
                raise StoredError(404, f"Item with id '{i.get('id')}' not found.", "not_found_error", param="input")
        return out

    def rebuilt(self, turn: StoredTurn, body: dict) -> dict:
        """The stateless request a turn amounts to: its history and its own input, in one ``input``."""
        out = {k: v for k, v in body.items() if k not in ("previous_response_id", "conversation", "background", "store")}
        out["input"] = turn.history + turn.input
        chat = responses_to_chat(out)
        if turn.history:
            self.gw.metrics.inc("meritorder_stored_responses_total", {"event": "rebuilt"})
        if turn.provider_previous is not None:
            # Where this turn's own input starts among the messages, for a provider that holds everything before it. Only
            # when the history alone converts to exactly those first messages (a tool call can't merge across the edge).
            alone = responses_to_chat({**out, "input": turn.history})["messages"]
            if chat["messages"][:len(alone)] == alone:
                turn.held = len(alone)
            else:
                turn.provider_previous = None
        return chat

    async def save(self, turn: StoredTurn, obj: dict, ctx, upstream_id: str | None = None) -> bool:
        """Keep a finished turn: the response (when ``store``) and, in a conversation, its input and output items. Done
        before the client sees the end of the response, since it may send the next turn at once. False if it couldn't
        be kept (logged; the response itself still goes to the client)."""
        try:
            await self._save(turn, obj, ctx, upstream_id)
        except Exception:  # noqa: BLE001 (the answer was paid for: give it to the client even if keeping it failed)
            log.exception("could not keep stored response %s", turn.rid)
            self.gw.metrics.inc("meritorder_stored_responses_total", {"event": "failed"})
            return False
        if ctx is not None and ctx.provider_chained:
            self.gw.metrics.inc("meritorder_stored_responses_total", {"event": "provider_chain"})
        self.gw.metrics.inc("meritorder_stored_responses_total", {"event": "stored"})
        return True

    async def _save(self, turn: StoredTurn, obj: dict, ctx, upstream_id: str | None) -> None:
        p, now = turn.principal, datetime.now(UTC)
        output = list(obj.get("output") or [])
        if turn.conversation is not None and obj.get("status") in DONE:
            conv = turn.conversation
            rows = []
            for item in turn.input + output:
                blob, kid = self.seal(item, self.item_aad(conv, item["id"]))
                rows.append({"item_id": item["id"], "response_id": turn.rid, "seal_key": kid, "content": blob, "bytes": len(blob)})
            await self.store.add_conversation_items(conv["id"], rows)
        if not turn.store and not turn.background:
            return
        row = {"id": turn.rid, "user_id": p.user_id, "team": p.team, "key_id": p.key_id, "created_at": datetime.fromtimestamp(turn.created, UTC),
               "touched_at": now, "thread_id": turn.thread_id, "previous_response_id": turn.previous["id"] if turn.previous else None,
               "conversation_id": turn.conversation["id"] if turn.conversation else None, "conversation_upto": turn.conversation_upto,
               "session_id": ctx.session.id if ctx is not None else turn.session_id, "status": obj.get("status") or "completed",
               "background": turn.background, "model": obj.get("model") or turn.model,
               "backend": ctx.cand.backend.name if ctx is not None else None, "upstream_model": ctx.cand.route.model if ctx is not None else None,
               "upstream_id": upstream_id, "request_id": ctx.request_id if ctx is not None else None,
               "input_items": len(turn.input), "output_items": len(output), "gateway": self.gw.gateway_id,
               "item_ids": [i["id"] for i in turn.input + output if i.get("id")], "expires_at": self.unstored_until(turn)}
        blob, kid = self.seal({"response": obj, "input": turn.input}, self.response_aad(row))
        row.update(content=blob, seal_key=kid, bytes=len(blob))
        if turn.background:
            # Cancelled, or given up on as failed (its gateway seemed to have stopped): what clients were told stands.
            if await self.store.update_stored_response(turn.rid, unless_status=("cancelled", "failed"), **row) is None:
                log.info("background response %s was cancelled or failed before it finished; its answer isn't kept", turn.rid)
                if upstream_id and ctx is not None:
                    self.gw.spawn(self.delete_provider_copies([row]))
                return
        else:
            await self.store.save_stored_response(row)
        if turn.previous is not None:
            await self.store.touch_stored_thread(turn.thread_id, now)  # the chain's retention counts from its last turn

    # ---------------------------------------------------------------- background

    def unstored_until(self, turn: StoredTurn) -> datetime | None:
        """A background response sent with store: false is kept only to be polled, for a while after it ends."""
        if turn.store:
            return None
        return datetime.now(UTC) + timedelta(seconds=self.gw.cfg.response_store.background_unstored_seconds)

    async def start_background(self, turn: StoredTurn, run) -> dict:
        """Keep the response as queued and answer at once; ``run()`` makes it, in this process, and returns the gateway's
        reply. Clients poll GET /v1/responses/{id}; POST .../cancel stops it."""
        obj = {"id": turn.rid, "object": "response", "created_at": turn.created, "status": "queued", "model": turn.model, "output": [],
               "usage": None, "error": None, "incomplete_details": None, **turn.extra}
        p = turn.principal
        row = {"id": turn.rid, "user_id": p.user_id, "team": p.team, "key_id": p.key_id, "created_at": datetime.fromtimestamp(turn.created, UTC),
               "thread_id": turn.thread_id, "previous_response_id": turn.previous["id"] if turn.previous else None,
               "conversation_id": turn.conversation["id"] if turn.conversation else None, "conversation_upto": turn.conversation_upto,
               "session_id": turn.session_id, "status": "queued", "background": True, "model": turn.model,
               "input_items": len(turn.input), "gateway": self.gw.gateway_id, "item_ids": [i["id"] for i in turn.input],
               # store: false: kept while it runs (at most the background timeout) and a while after, never longer
               "expires_at": None if turn.store else datetime.now(UTC) + timedelta(
                   seconds=self.gw.cfg.response_store.background_timeout_seconds + self.gw.cfg.response_store.background_unstored_seconds)}
        blob, kid = self.seal({"response": obj, "input": turn.input}, self.response_aad(row))
        await self.store.save_stored_response({**row, "content": blob, "seal_key": kid, "bytes": len(blob)})

        async def work() -> None:
            alive = None
            try:
                if await self.store.update_stored_response(turn.rid, unless_status=("cancelled",), status="in_progress") is None:
                    return
                alive = asyncio.create_task(self._keep_alive(turn))
                resp = await run()
                if resp.status_code >= 400:  # refused before or by every provider: nothing was kept
                    await self._fail(turn, obj, resp)
            except asyncio.CancelledError:
                await self.store.update_stored_response(turn.rid, unless_status=DONE, status="cancelled")
                raise
            except Exception as e:  # noqa: BLE001 (the client polls for the outcome)
                log.exception("background response %s failed", turn.rid)
                await self._fail(turn, obj, None, f"{type(e).__name__}: {e}")
            finally:
                if alive is not None:
                    alive.cancel()
                self._running.pop(turn.rid, None)

        task = self._running[turn.rid] = asyncio.create_task(work())
        self.gw._tasks.add(task)  # a gateway shutting down finishes what it was running (Gateway.drain)
        task.add_done_callback(self.gw._tasks.discard)
        return obj

    async def _keep_alive(self, turn: StoredTurn) -> None:
        """While this gateway works on a background response, touch its row (``updated_at``) a few times per
        background_timeout_seconds, so the worker's stale_background fails only the ones whose gateway stopped, never one
        that's still running (a long reasoning turn can take longer than the timeout). With store: false, the time it's
        kept for polling moves along too."""
        rs = self.gw.cfg.response_store
        every = min(60.0, rs.background_timeout_seconds / 4)
        while True:
            await asyncio.sleep(every)
            fields = {} if turn.store else {"expires_at": datetime.now(UTC) + timedelta(
                seconds=rs.background_timeout_seconds + rs.background_unstored_seconds)}
            if await self.store.update_stored_response(turn.rid, unless_status=(*DONE, "cancelled", "failed"), **fields) is None:
                return  # finished, cancelled or given up on: nothing to keep alive

    async def _fail(self, turn: StoredTurn, obj: dict, resp, message: str | None = None) -> None:
        if resp is not None:
            try:
                err = json.loads(resp.body).get("error") or {}
            except (ValueError, AttributeError):
                err = {}
            message = err.get("message") or f"HTTP {resp.status_code}"
            code = err.get("type") or "server_error"
        else:
            code = "server_error"
        failed = {**obj, "status": "failed", "error": {"code": code, "message": message}}
        row = await self.store.get_stored_response(turn.rid)
        if row is None:
            return
        blob, kid = self.seal({"response": failed, "input": turn.input}, self.response_aad(row))
        await self.store.update_stored_response(turn.rid, unless_status=("cancelled",), status="failed", content=blob, seal_key=kid,
                                                bytes=len(blob), expires_at=self.unstored_until(turn))

    async def cancel(self, p: Principal, rid: str) -> dict:
        row = await self.response_row(p, rid, owner=True)
        if not row["background"]:
            raise StoredError(400, "Only background responses can be cancelled.", param="response_id")
        if row["status"] in RUNNING:
            await self.store.update_stored_response(rid, unless_status=(*DONE, "failed"), status="cancelled", cancel_requested=True)
            task = self._running.get(rid)
            if task is not None:
                task.cancel()
            row = await self.store.get_stored_response(rid) or row
        return self.response_object(row)

    # ---------------------------------------------------------------- deleting

    async def delete_provider_copies(self, rows: list[dict], backends: dict | None = None) -> int:
        """Delete what providers stored for these responses (``responses_store``). Best effort: a provider that can't be
        reached keeps its copy until its own retention (OpenAI: 30 days)."""
        backends = self.gw.backends if backends is None else backends
        n = 0
        for row in rows:
            b = backends.get(row.get("backend") or "")
            if not row.get("upstream_id") or b is None or not hasattr(b, "delete_response"):
                continue
            try:
                await b.delete_response(row["upstream_id"])
                n += 1
            except Exception as e:  # noqa: BLE001
                log.warning("could not delete %s's copy of %s: %s", row["backend"], row["id"], e)
        return n

    async def delete(self, p: Principal, rid: str) -> dict:
        row = await self.response_row(p, rid, owner=True)
        task = self._running.get(rid)
        if task is not None:
            task.cancel()
        gone = await self.store.delete_stored_response(rid)
        if gone:
            await self.delete_provider_copies([gone])
        return {"id": rid, "object": "response.deleted", "deleted": True}

    async def forget_user(self, user_id: str) -> int:
        """Everything a person stored, deleted: when they're removed through SCIM or disabled on the People page."""
        gone = await self.store.delete_stored_for_user(user_id)
        for row in gone:
            task = self._running.get(row["id"])
            if task is not None:
                task.cancel()
        if gone:
            self.gw.spawn(self.delete_provider_copies(gone))
        return len(gone)

    async def purge(self, backends: dict | None = None, now: datetime | None = None) -> tuple[int, int]:
        """Delete chains and conversations past their team's retention, and everything of teams whose stored mode is
        off. Returns how many responses and conversations went."""
        now = now or datetime.now(UTC)
        # Every configured team has its own cutoff: its retention, or now while its stored mode is off (everything goes).
        # Only teams that aren't configured fall back to the default team's settings, as cfg.team() does.
        cutoffs = {t.name: now - keep if (keep := self.retention(t.name)) is not None else now for t in self.gw.cfg.teams}
        default = self.retention("default")
        gone, convs = await self.store.purge_stored(cutoffs, now - default if default is not None else now)
        if gone:
            await self.delete_provider_copies(gone, backends)
        return len(gone), convs

    async def maintain(self, backends: dict | None = None) -> dict:
        """The worker's pass (and a gateway's without one): retention, background responses whose gateway stopped, and
        values sealed with a key that's been rotated out."""
        responses, conversations = await self.purge(backends)
        stale = await self.store.stale_background(datetime.now(UTC) - timedelta(seconds=self.gw.cfg.response_store.background_timeout_seconds))
        resealed = await self.reseal()
        if responses or conversations or stale or resealed:
            log.info("stored responses: %d responses and %d conversations past retention deleted, %d background responses failed, "
                     "%d values sealed again", responses, conversations, len(stale), resealed)
        return {"deleted_responses": responses, "deleted_conversations": conversations, "failed_background": len(stale), "resealed": resealed}

    async def reseal(self, limit: int = RESEAL_BATCH) -> int:
        """Seal again with the current key what an older one sealed, so the old key can be dropped."""
        ring = self.keyring()
        if not ring.usable or len(ring.keys) < 2:
            return 0
        await self.unopenable_keys(ring)
        n = 0
        for value in await self.store.stored_to_reseal(sorted(k for k in ring.keys if k != ring.current), limit):
            if value["kind"] == "response":
                aad = f"response:{value['id']}:{value['user_id']}:{value['team']}"
            elif value["kind"] == "conversation":
                aad = f"conversation:{value['id']}:{value['user_id']}:{value['team']}"
            else:
                aad = f"item:{value['conversation_id']}:{value['id']}:{value['user_id']}:{value['team']}"
            try:
                data = ring.open(bytes(value["content"]), aad.encode())
            except SealError as e:
                log.warning("stored %s %s can't be opened to seal it again: %s", value["kind"], value["id"], e)
                continue
            n += await self.store.reseal_stored(value, ring.seal(data, aad.encode()), ring.current)
        return n

    async def unopenable_keys(self, ring: Keyring) -> list[str]:
        """Ids of keys that sealed something stored and aren't on the ring. Those values can't be opened, so they're left
        out of resealing (and stay unreadable until they expire); each such key is alerted once."""
        from .. import alerts

        missing = sorted((await self.store.stored_seal_keys()) - set(ring.keys))
        if not missing:
            return []
        rs = self.gw.cfg.response_store
        # Set-if-absent with a token of our own: of several workers (or gateways) only the one whose token lands alerts.
        token = secrets.token_hex(8)
        won = await self.gw.state.set_json_once({f"alert:stored:unknown-key:{kid}": token for kid in missing}, UNKNOWN_KEY_ALERT_SECONDS)
        for kid in missing:
            if won.get(f"alert:stored:unknown-key:{kid}") == token:
                await alerts.notify(f"Stored values sealed with key {kid} can't be opened: it isn't {rs.key_env} or one of "
                                    f"{rs.old_keys_env}. They aren't sealed again and stay unreadable until their retention deletes "
                                    "them. Put the key back in the old keys to read them and move them to the current key.",
                                    state=self.gw.state, title=f"Stored values sealed with an unknown key ({kid})", level="critical", page="teams")
        return missing

    async def sizes(self, team: str | None = None) -> list[dict]:
        ring = self.keyring()
        rows = await self.store.stored_sizes(ring.current)
        return [r for r in rows if team is None or r["team"] == team]
