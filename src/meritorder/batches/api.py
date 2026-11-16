"""The batch endpoints, as OpenAI and Anthropic define them, so their SDKs work unchanged:

* OpenAI: ``POST /v1/files`` (purpose ``batch``), ``GET /v1/files[/{id}[/content]]``, ``DELETE /v1/files/{id}``;
  ``POST /v1/batches``, ``GET /v1/batches[/{id}]``, ``POST /v1/batches/{id}/cancel``. Lines may call
  ``/v1/chat/completions``, ``/v1/responses`` or ``/v1/embeddings``.
* Anthropic: ``POST /v1/messages/batches``, ``GET /v1/messages/batches[/{id}]``, ``POST .../{id}/cancel``,
  ``GET .../{id}/results``, ``DELETE .../{id}``.

A batch is its owner's: other people get 404, except admins, who may also read and cancel it, but not read or delete its results.
"""

from __future__ import annotations

import functools
import re

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from ..auth import AuthError, Principal
from ..proxy import error_body
from .service import BatchError, anthropic_object, file_object, openai_object, parse_anthropic, parse_openai_file

ANTHROPIC = "anthropic"
OPENAI = "openai"


def parse_multipart(body: bytes, content_type: str) -> dict[str, tuple[str | None, bytes]]:
    """The parts of a multipart/form-data body: name -> (filename, content). Enough for the Files API's one file and a
    few fields, without another dependency."""
    m = re.search(r'boundary="?([^";]+)"?', content_type or "")
    if not m or "multipart/form-data" not in (content_type or "").lower():
        raise BatchError(400, "upload the file as multipart/form-data with file and purpose fields")
    sep = b"--" + m.group(1).encode()
    out: dict[str, tuple[str | None, bytes]] = {}
    for part in body.split(sep)[1:]:
        if part.startswith(b"--"):
            break
        head, _, data = part.removeprefix(b"\r\n").partition(b"\r\n\r\n")
        name = re.search(rb'\bname="([^"]*)"', head)
        if name:
            filename = re.search(rb'\bfilename="([^"]*)"', head)
            out[name.group(1).decode(errors="replace")] = (filename.group(1).decode(errors="replace") if filename else None,
                                                          data.removesuffix(b"\r\n"))
    return out


def base_url(request: Request) -> str:
    """Where clients reach this gateway (results_url must point back here): the request's own Host header and scheme.
    X-Forwarded-* headers aren't read here, since any client can send them; uvicorn applies X-Forwarded-Proto only from
    the proxies FORWARDED_ALLOW_IPS trusts (``meritorder serve`` runs it with proxy_headers on), so behind a proxy the
    scheme is right as long as the proxy passes the Host header through."""
    return str(request.base_url).rstrip("/")


def build_router(gw) -> APIRouter:
    router = APIRouter(tags=["batches"])

    def fail(fmt: str, e: BatchError) -> JSONResponse:
        resp = JSONResponse(error_body(fmt, e.status, e.message, e.kind), status_code=e.status)
        resp.headers.update(e.headers)
        return resp

    async def who(request: Request) -> Principal:
        try:
            return await gw.authn.authenticate({k.lower(): v for k, v in request.headers.items()})
        except AuthError as e:
            raise BatchError(e.status, e.message, "authentication_error")

    async def json_body(request: Request) -> dict:
        try:
            body = await request.json()
        except ValueError:
            raise BatchError(400, "request body must be valid JSON")
        if not isinstance(body, dict):
            raise BatchError(400, "request body must be a JSON object")
        return body

    def guarded(fmt: str):
        """Turn BatchErrors into the API's own error shape."""
        def wrap(fn):
            @functools.wraps(fn)
            async def inner(*a, **kw):
                try:
                    return await fn(*a, **kw)
                except BatchError as e:
                    return fail(fmt, e)
            return inner
        return wrap

    async def batch_of(request: Request, batch_id: str, fmt: str) -> tuple[Principal, dict]:
        p = await who(request)
        b = gw.batches.visible(p, await gw.store.get_batch(batch_id), fmt)
        if fmt == ANTHROPIC and (b.get("metadata") or {}).get("deleted"):
            raise BatchError(404, "no such batch", "not_found_error")
        return p, b

    # ---------------------------------------------------------------- OpenAI: files

    async def file_of(p: Principal, file_id: str) -> dict:
        f = await gw.store.get_batch_file(file_id)
        if f is None or f["user_id"] != p.user_id:
            raise BatchError(404, f"no such file: {file_id}", "not_found_error")
        return f

    @router.post("/v1/files")
    @guarded(OPENAI)
    async def upload(request: Request):
        p = await who(request)
        parts = parse_multipart(await request.body(), request.headers.get("content-type", ""))
        if "file" not in parts:
            raise BatchError(400, "the upload needs a file field")
        filename, data = parts["file"]
        purpose = parts.get("purpose", (None, b""))[1].decode(errors="replace").strip()
        return file_object(await gw.batches.upload(p, filename or "batch.jsonl", purpose, data))

    @router.get("/v1/files")
    @guarded(OPENAI)
    async def list_files(request: Request, purpose: str | None = None, limit: int = Query(10000, ge=1, le=10000)):
        p = await who(request)
        rows = [file_object(f) for f in await gw.store.list_batch_files(p.user_id, purpose if purpose != "batch_output" else None, limit)
                if purpose != "batch_output" or f["kind"] != "input"]
        return {"object": "list", "data": rows, "first_id": rows[0]["id"] if rows else None, "last_id": rows[-1]["id"] if rows else None,
                "has_more": False}

    @router.get("/v1/files/{file_id}")
    @guarded(OPENAI)
    async def get_file(request: Request, file_id: str):
        f = await file_of(await who(request), file_id)
        if f.get("deleted_at") and f["kind"] != "input":
            raise BatchError(404, f"no such file: {file_id}", "not_found_error")
        return file_object(f)

    @router.get("/v1/files/{file_id}/content")
    @guarded(OPENAI)
    async def file_content(request: Request, file_id: str):
        f = await file_of(await who(request), file_id)
        if f["kind"] == "input":
            return Response(await gw.batches.file_content(f), media_type="application/jsonl")
        if f.get("deleted_at"):
            raise BatchError(404, f"no such file: {file_id}", "not_found_error")
        b = await gw.store.get_batch(f["batch_id"])
        if b is None or b["status"] not in ("finalizing", "ended"):
            raise BatchError(400, "the batch hasn't finished yet", "invalid_request_error")
        stream = await gw.batches.results(b, "succeeded" if f["kind"] == "output" else "failed")
        return StreamingResponse(stream, media_type="application/jsonl")

    @router.delete("/v1/files/{file_id}")
    @guarded(OPENAI)
    async def delete_file(request: Request, file_id: str):
        f = await file_of(await who(request), file_id)
        if f["kind"] != "input":
            b = await gw.store.get_batch(f["batch_id"])
            if b is not None and b["status"] == "ended":
                await gw.batches.purge(b)  # the output and error files are read from the providers: deleting one deletes them there
        await gw.batches.drop_file(f)
        return {"id": file_id, "object": "file", "deleted": True}

    # ---------------------------------------------------------------- OpenAI: batches

    @router.post("/v1/batches")
    @guarded(OPENAI)
    async def create_openai(request: Request):
        p = await who(request)
        body = await json_body(request)
        if (body.get("completion_window") or "24h") != "24h":
            raise BatchError(400, "completion_window must be 24h, the only window the providers offer")
        f = await file_of(p, str(body.get("input_file_id") or ""))
        if f["kind"] != "input":
            raise BatchError(400, f"{f['id']} isn't an input file")
        endpoint = str(body.get("endpoint") or "")
        lines = parse_openai_file(await gw.batches.file_content(f), endpoint)
        metadata = body.get("metadata") if isinstance(body.get("metadata"), dict) else None
        headers = {k.lower(): v for k, v in request.headers.items()}
        b = await gw.batches.create(p, headers, OPENAI, lines, endpoint, f, metadata)
        return JSONResponse(openai_object(b), headers={"x-meritorder-estimated-cost-usd": f"{b['estimated_cost']:.6f}"})

    @router.get("/v1/batches")
    @guarded(OPENAI)
    async def list_openai(request: Request, after: str | None = None, limit: int = Query(20, ge=1, le=100)):
        p = await who(request)
        rows = await gw.store.list_batches(user_id=p.user_id, client_format=OPENAI, limit=limit + 1, after=after)
        data = [openai_object(b) for b in rows[:limit]]
        return {"object": "list", "data": data, "first_id": data[0]["id"] if data else None, "last_id": data[-1]["id"] if data else None,
                "has_more": len(rows) > limit}

    @router.get("/v1/batches/{batch_id}")
    @guarded(OPENAI)
    async def get_openai(request: Request, batch_id: str):
        return openai_object((await batch_of(request, batch_id, OPENAI))[1])

    @router.post("/v1/batches/{batch_id}/cancel")
    @guarded(OPENAI)
    async def cancel_openai(request: Request, batch_id: str):
        p, b = await batch_of(request, batch_id, OPENAI)
        return openai_object(await gw.batches.cancel(b, p.user_id))

    # ---------------------------------------------------------------- Anthropic: message batches

    @router.post("/v1/messages/batches")
    @guarded(ANTHROPIC)
    async def create_anthropic(request: Request):
        p = await who(request)
        body = await json_body(request)
        headers = {k.lower(): v for k, v in request.headers.items()}
        b = await gw.batches.create(p, headers, ANTHROPIC, parse_anthropic(body.get("requests")))
        if b["status"] == "ended" and b.get("outcome") == "failed":
            why = "; ".join(e["message"] for e in ((b.get("errors") or {}).get("data") or [])[:3])
            raise BatchError(502, f"no provider accepted the batch: {why}", "api_error")
        return JSONResponse(anthropic_object(b, base_url(request)), headers={"x-meritorder-estimated-cost-usd": f"{b['estimated_cost']:.6f}"})

    @router.get("/v1/messages/batches")
    @guarded(ANTHROPIC)
    async def list_anthropic(request: Request, limit: int = Query(20, ge=1, le=1000), before_id: str | None = None, after_id: str | None = None):
        p = await who(request)
        rows = [b for b in await gw.store.list_batches(user_id=p.user_id, client_format=ANTHROPIC, limit=None)
                if not (b.get("metadata") or {}).get("deleted")]
        ids = [b["id"] for b in rows]  # newest first: after_id pages on to older batches, before_id back to newer ones
        if before_id in ids:
            end = ids.index(before_id)
            rows, more = rows[max(0, end - limit):end], end > limit
        else:
            start = ids.index(after_id) + 1 if after_id in ids else 0
            rows, more = rows[start:start + limit], len(rows) > start + limit
        data = [anthropic_object(b, base_url(request)) for b in rows]
        return {"data": data, "has_more": more, "first_id": data[0]["id"] if data else None, "last_id": data[-1]["id"] if data else None}

    @router.get("/v1/messages/batches/{batch_id}")
    @guarded(ANTHROPIC)
    async def get_anthropic(request: Request, batch_id: str):
        return anthropic_object((await batch_of(request, batch_id, ANTHROPIC))[1], base_url(request))

    @router.post("/v1/messages/batches/{batch_id}/cancel")
    @guarded(ANTHROPIC)
    async def cancel_anthropic(request: Request, batch_id: str):
        p, b = await batch_of(request, batch_id, ANTHROPIC)
        return anthropic_object(await gw.batches.cancel(b, p.user_id), base_url(request))

    @router.get("/v1/messages/batches/{batch_id}/results")
    @guarded(ANTHROPIC)
    async def results_anthropic(request: Request, batch_id: str):
        p, b = await batch_of(request, batch_id, ANTHROPIC)
        if b["user_id"] != p.user_id:
            raise BatchError(403, "only the person who made a batch can read its results", "permission_error")
        if b["status"] not in ("finalizing", "ended"):
            raise BatchError(400, "the batch hasn't ended yet; its results_url is set once it has")
        return StreamingResponse(await gw.batches.results(b), media_type="application/binary")

    @router.delete("/v1/messages/batches/{batch_id}")
    @guarded(ANTHROPIC)
    async def delete_anthropic(request: Request, batch_id: str):
        p, b = await batch_of(request, batch_id, ANTHROPIC)
        if b["user_id"] != p.user_id:  # admins may see and cancel it, but its results are its owner's to delete
            raise BatchError(403, "only the person who made a batch can delete it", "permission_error")
        if b["status"] != "ended":
            raise BatchError(400, "a batch can be deleted once it has ended; cancel it first")
        await gw.batches.purge(b, deleted=True)
        return {"id": batch_id, "type": "message_batch_deleted"}

    return router
