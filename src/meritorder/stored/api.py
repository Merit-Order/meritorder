"""The stored-mode endpoints, as OpenAI's API reference defines them (openai-python 3.42, November 2026), so its SDK works
unchanged:

* ``GET /v1/responses/{id}``, ``DELETE /v1/responses/{id}``, ``POST /v1/responses/{id}/cancel`` (background responses),
  ``GET /v1/responses/{id}/input_items``;
* ``POST /v1/conversations``, ``GET``/``POST``/``DELETE /v1/conversations/{id}``, ``GET``/``POST /v1/conversations/{id}/items``,
  ``GET``/``DELETE /v1/conversations/{id}/items/{item_id}``.

A response or conversation belongs to the person and team that made it. Nobody else sees it (404), admins included,
unless the team lists their role in ``stored_responses.readers``; only its owner deletes or changes it.
"""

from __future__ import annotations

import functools
import secrets

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from ..auth import AuthError, Principal
from .service import StoredError, normalize_input

MAX_METADATA = 16  # OpenAI's limits for metadata: 16 pairs, keys up to 64 characters, values up to 512


def _error(e: StoredError) -> JSONResponse:
    return JSONResponse({"error": {"message": e.message, "type": e.kind, "param": e.param, "code": e.status}}, status_code=e.status)


def page(items: list[dict], after: str | None, limit: int, order: str) -> dict:
    """OpenAI's cursor list: items in ``order`` (asc: oldest first), after the item whose id is ``after``."""
    rows = list(reversed(items)) if order == "desc" else list(items)
    if after:
        ids = [i.get("id") for i in rows]
        rows = rows[ids.index(after) + 1:] if after in ids else []
    data = rows[:limit]
    return {"object": "list", "data": data, "first_id": data[0].get("id") if data else None,
            "last_id": data[-1].get("id") if data else None, "has_more": len(rows) > limit}


def metadata_of(value) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict) or len(value) > MAX_METADATA or any(
            not isinstance(k, str) or len(k) > 64 or not isinstance(v, str) or len(v) > 512 for k, v in value.items()):
        raise StoredError(400, "metadata takes up to 16 string pairs: keys up to 64 characters, values up to 512.", param="metadata")
    return value


def build_router(gw) -> APIRouter:
    router = APIRouter(tags=["responses"])

    def guarded(fn):
        @functools.wraps(fn)
        async def inner(*a, **kw):
            try:
                return await fn(*a, **kw)
            except StoredError as e:
                return _error(e)
        return inner

    async def who(request: Request) -> Principal:
        try:
            return await gw.authn.authenticate({k.lower(): v for k, v in request.headers.items()})
        except AuthError as e:
            raise StoredError(e.status, e.message, "authentication_error")

    async def json_body(request: Request) -> dict:
        try:
            body = await request.json()
        except ValueError:
            raise StoredError(400, "request body must be valid JSON")
        if not isinstance(body, dict):
            raise StoredError(400, "request body must be a JSON object")
        return body

    s = gw.stored

    # ---------------------------------------------------------------- responses

    @router.get("/v1/responses/{response_id}")
    @guarded
    async def get_response(request: Request, response_id: str, stream: bool = False):
        p = await who(request)
        row = await s.response_row(p, response_id)
        if stream:
            raise StoredError(400, "streaming a stored response again isn't supported here: GET it without stream.", param="stream")
        return s.response_object(row)

    @router.delete("/v1/responses/{response_id}")
    @guarded
    async def delete_response(request: Request, response_id: str):
        return await s.delete(await who(request), response_id)

    @router.post("/v1/responses/{response_id}/cancel")
    @guarded
    async def cancel_response(request: Request, response_id: str):
        return await s.cancel(await who(request), response_id)

    @router.get("/v1/responses/{response_id}/input_items")
    @guarded
    async def input_items(request: Request, response_id: str, after: str | None = None, limit: int = Query(20, ge=1, le=100),
                          order: str = Query("desc", pattern="^(asc|desc)$")):
        p = await who(request)
        row = await s.response_row(p, response_id)
        return page(await s.input_items(p, row), after, limit, order)

    # ---------------------------------------------------------------- conversations

    def conversation_object(row: dict) -> dict:
        meta = s.open(row["content"], s.conversation_aad(row)).get("metadata") or {}
        return {"id": row["id"], "object": "conversation", "created_at": int(row["created_at"].timestamp()), "metadata": meta}

    def item_rows(conv: dict, items: list[dict]) -> list[dict]:
        out = []
        for item in items:
            blob, kid = s.seal(item, s.item_aad(conv, item["id"]))
            out.append({"item_id": item["id"], "response_id": None, "seal_key": kid, "content": blob, "bytes": len(blob)})
        return out

    @router.post("/v1/conversations")
    @guarded
    async def create_conversation(request: Request):
        p = await who(request)
        body = await json_body(request)
        s.require_on(gw.cfg.team(p.team), "a conversation")
        cid = f"conv_{secrets.token_hex(24)}"
        row = {"id": cid, "user_id": p.user_id, "team": p.team, "key_id": p.key_id, "session_id": f"conv-{cid[5:29]}"}
        blob, kid = s.seal({"metadata": metadata_of(body.get("metadata"))}, s.conversation_aad(row))
        row.update(content=blob, seal_key=kid, bytes=len(blob))
        items = normalize_input(body.get("items") or [])
        conv, _ = await gw.store.create_conversation(row, item_rows(row, items))
        return conversation_object(conv)

    @router.get("/v1/conversations/{conversation_id}")
    @guarded
    async def get_conversation(request: Request, conversation_id: str):
        return conversation_object(await s.conversation_row(await who(request), conversation_id))

    @router.post("/v1/conversations/{conversation_id}")
    @guarded
    async def update_conversation(request: Request, conversation_id: str):
        p = await who(request)
        conv = await s.conversation_row(p, conversation_id, owner=True)
        body = await json_body(request)
        blob, kid = s.seal({"metadata": metadata_of(body.get("metadata"))}, s.conversation_aad(conv))
        return conversation_object(await gw.store.update_conversation(conversation_id, content=blob, seal_key=kid, bytes=len(blob)))

    @router.delete("/v1/conversations/{conversation_id}")
    @guarded
    async def delete_conversation(request: Request, conversation_id: str):
        await s.conversation_row(await who(request), conversation_id, owner=True)
        await gw.store.delete_conversation(conversation_id)
        return {"id": conversation_id, "object": "conversation.deleted", "deleted": True}

    @router.get("/v1/conversations/{conversation_id}/items")
    @guarded
    async def list_items(request: Request, conversation_id: str, after: str | None = None, limit: int = Query(20, ge=1, le=100),
                         order: str = Query("desc", pattern="^(asc|desc)$")):
        conv = await s.conversation_row(await who(request), conversation_id)
        cursor = None
        if after:
            at = await gw.store.get_conversation_item(conversation_id, after)
            if at is None:
                return page([], None, limit, order)
            cursor = at["seq"]
        rows = await gw.store.conversation_items(conversation_id, after=cursor, limit=limit + 1, desc=order == "desc")
        data = [s.open(r["content"], s.item_aad(conv, r["item_id"])) for r in rows[:limit]]
        return {"object": "list", "data": data, "first_id": data[0]["id"] if data else None, "last_id": data[-1]["id"] if data else None,
                "has_more": len(rows) > limit}

    @router.post("/v1/conversations/{conversation_id}/items")
    @guarded
    async def add_items(request: Request, conversation_id: str):
        p = await who(request)
        conv = await s.conversation_row(p, conversation_id, owner=True)
        body = await json_body(request)
        if not isinstance(body.get("items"), list) or not body["items"]:
            raise StoredError(400, "items must be a non-empty list of input items.", param="items")
        items = normalize_input(body["items"])
        known = {i["id"] for i in items}
        if len(known) < len(items) or any([await gw.store.get_conversation_item(conversation_id, i) for i in known]):
            raise StoredError(400, "an item with that id is already in this conversation.", param="items")
        await gw.store.add_conversation_items(conversation_id, item_rows(conv, items))
        return {"object": "list", "data": items, "first_id": items[0]["id"], "last_id": items[-1]["id"], "has_more": False}

    @router.get("/v1/conversations/{conversation_id}/items/{item_id}")
    @guarded
    async def get_item(request: Request, conversation_id: str, item_id: str):
        conv = await s.conversation_row(await who(request), conversation_id)
        row = await gw.store.get_conversation_item(conversation_id, item_id)
        if row is None:
            raise StoredError(404, f"Item with id '{item_id}' not found.", "not_found_error")
        return s.open(row["content"], s.item_aad(conv, item_id))

    @router.delete("/v1/conversations/{conversation_id}/items/{item_id}")
    @guarded
    async def delete_item(request: Request, conversation_id: str, item_id: str):
        conv = await s.conversation_row(await who(request), conversation_id, owner=True)
        if not await gw.store.delete_conversation_item(conversation_id, item_id):
            raise StoredError(404, f"Item with id '{item_id}' not found.", "not_found_error")
        return conversation_object(conv)

    return router
