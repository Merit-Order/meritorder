"""Batches in the gateway: create one from a client's requests, read it in the client's format, cancel it, and join its
results back from the providers (docs/02-request-flow.md#batches-work-that-can-wait).

Creating a batch:

1. Every request is routed on its own (Router.decide with ``batch=True``): a named model goes to its cheapest route with a
   batch API, ``auto`` gets a level and the cheapest model at it among those routes, at batch prices. Team policies,
   allow and deny lists and the per-request cost limit apply as for live requests; a request none of them allow is
   refused on its own, with why, and the rest go ahead.
2. The requests are grouped into provider jobs: one per backend, and per model where the provider takes one model a job,
   within the provider's limits (and above Bedrock's minimum, else they move to their next route).
3. The batch's estimated cost (each request's prompt at its batch price, plus its max_tokens, or the router's usual
   reply size when it sets none) must fit every budget it counts against; it is reserved there until the results settle.
4. The jobs go to the providers. A job a provider refuses moves to its requests' next routes once; what still can't go
   is refused with the reasons. An uploaded input file's content is deleted as soon as its batch is with the providers.

Results stay with the providers (or in the operator's bucket) and are read again, converted to the client's format,
whenever the client asks for them. Nothing of a request's content is stored by the gateway past submission, except for
batch work on self-hosted GPUs (gpu.py): there's no provider to hold it, so the gateway holds it sealed (content.py) until
each request runs, and its result until the retention period ends.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import secrets
from collections.abc import AsyncIterator
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta

from ..auth import Principal
from ..backends import BackendError
from ..formats.responses import ResponsesError, chat_to_responses, responses_to_chat
from ..formats.translate import anthropic_to_openai_response, openai_to_anthropic_response
from ..proxy import api_batch_price, kind_for_status, parse_tags
from ..router import Candidate, Decision, RoutingError
from ..sessions import Session
from ..tokens import Estimate, count, estimate
from .content import Content, ContentCache, SealError
from .providers import APIS, BatchAPI, Outcome, batch_api_for

log = logging.getLogger("meritorder.batches")

ENDPOINTS = {"/v1/chat/completions": "chat", "/v1/responses": "responses", "/v1/embeddings": "embeddings"}
CUSTOM_ID = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")  # Anthropic's rule for custom_id
WINDOW = timedelta(hours=24)  # every provider's completion window
SUBMIT_LIMIT = timedelta(minutes=20)  # one job's submission, then the provider counts as refusing it (inside poller.STUCK)
SPENDING_LIMIT = {"x-meritorder-limit": "budget"}
OUTCOME_STATUS = {"succeeded": 200, "canceled": 499, "expired": 504}
EXPIRED = "This request could not be executed before the completion window expired."
CANCELLED = "This request was not executed because the batch was cancelled."


class BatchError(Exception):
    def __init__(self, status: int, message: str, kind: str = "invalid_request_error", headers: dict | None = None):
        super().__init__(message)
        self.status, self.message, self.kind, self.headers = status, message, kind, headers or {}


def new_id(prefix: str) -> str:
    return f"{prefix}{secrets.token_hex(12)}"


def month_key(now: datetime | None = None) -> str:
    return (now or datetime.now(UTC)).strftime("%Y-%m")


def spend_keys(user_id: str, team: str, month: str) -> list[str]:
    """The budget counters a batch counts against (no session: a batch isn't one)."""
    return [f"{month}:user:{user_id}", f"{month}:team:{team}", f"{month}:org:all"]


# ---------------------------------------------------------------- client formats


def counts_of(b: dict) -> dict[str, int]:
    c = b.get("counts") or {}
    return {k: int(c.get(k) or 0) for k in ("processing", "succeeded", "errored", "canceled", "expired")}


def _ts(d: datetime | None) -> int | None:
    return int(d.timestamp()) if d else None


def _iso(d: datetime | None) -> str | None:
    return d.astimezone(UTC).isoformat().replace("+00:00", "Z") if d else None


def openai_status(b: dict) -> str:
    if b["status"] == "ended":
        return {"completed": "completed", "cancelled": "cancelled", "expired": "expired", "failed": "failed"}.get(b.get("outcome") or "", "completed")
    if b.get("cancel_requested_at"):
        return "cancelling"
    return "finalizing" if b["status"] == "finalizing" else "in_progress"


def summary(b: dict) -> dict:
    """What Merit Order adds to a batch object (as ``meritorder``): where its requests went and what they cost."""
    return {"estimated_cost_usd": round(b.get("estimated_cost") or 0, 6), "reserved_usd": round(b.get("reserved_cost") or 0, 6),
            "cost_usd": round(b.get("actual_cost") or 0, 6), "savings_usd": round(b.get("savings_total") or 0, 6),
            "batch_discount_usd": round(b.get("savings_batch") or 0, 6),
            # Batch work on GPUs: what running there took off the batch price (the batch discount is counted as above).
            "self_hosting_usd": round(b.get("savings_self_hosting") or 0, 6), "providers": b.get("providers") or {}}


def openai_object(b: dict) -> dict:
    status, c = openai_status(b), counts_of(b)
    ended = b["status"] == "ended"
    out = {"id": b["id"], "object": "batch", "endpoint": b.get("endpoint") or "/v1/chat/completions", "errors": b.get("errors"),
           "input_file_id": b.get("input_file_id") or "", "completion_window": "24h", "status": status,
           "output_file_id": b.get("output_file_id") if ended else None, "error_file_id": b.get("error_file_id") if ended else None,
           "created_at": _ts(b["created_at"]), "in_progress_at": _ts(b["created_at"]) if status != "failed" else None,
           "expires_at": _ts(b.get("expires_at")), "finalizing_at": _ts(b.get("finalizing_at")),
           "completed_at": _ts(b.get("ended_at")) if status == "completed" else None,
           "failed_at": _ts(b.get("ended_at")) if status == "failed" else None,
           "expired_at": _ts(b.get("ended_at")) if status == "expired" else None,
           "cancelling_at": _ts(b.get("cancel_requested_at")), "cancelled_at": _ts(b.get("ended_at")) if status == "cancelled" else None,
           "request_counts": {"total": b.get("request_count") or 0, "completed": c["succeeded"],
                              "failed": c["errored"] + c["canceled"] + c["expired"]},
           "metadata": b.get("metadata") or None, "meritorder": summary(b)}
    return out


def anthropic_object(b: dict, base_url: str) -> dict:
    ended = b["status"] in ("ended", "finalizing")
    status = "ended" if ended else "canceling" if b.get("cancel_requested_at") else "in_progress"
    return {"id": b["id"], "type": "message_batch", "processing_status": status, "request_counts": counts_of(b),
            "ended_at": _iso(b.get("ended_at") or b.get("finalizing_at")) if ended else None, "created_at": _iso(b["created_at"]),
            "expires_at": _iso(b.get("expires_at")), "archived_at": _iso(b.get("purged_at")), "cancel_initiated_at": _iso(b.get("cancel_requested_at")),
            "results_url": f"{base_url.rstrip('/')}/v1/messages/batches/{b['id']}/results" if ended and not b.get("purged_at") else None,
            "meritorder": summary(b)}


def file_object(f: dict) -> dict:
    return {"id": f["id"], "object": "file", "bytes": int(f.get("bytes") or 0), "created_at": _ts(f["created_at"]),
            "expires_at": _ts(f.get("expires_at")), "filename": f.get("filename") or f"{f['id']}.jsonl",
            "purpose": "batch_output" if f.get("kind") in ("output", "error") else f.get("purpose") or "batch",
            "status": "processed", "status_details": None}


def label(line: dict) -> str:
    """The model name a reply carries: the one asked for, or for ``auto`` the one that answered (as live replies do)."""
    asked = line.get("requested_model") or ""
    return line.get("model") or asked if asked in ("", "auto") else asked


def client_body(b: dict, line: dict, fmt: str, body: dict) -> dict:
    """A provider's reply (in the backend's format ``fmt``) as the client's endpoint returns it."""
    kind = line["kind"]
    if kind == "embeddings":
        return {**body, "model": label(line)}
    if b["client_format"] == "anthropic":
        return body if fmt == "anthropic" else openai_to_anthropic_response(body, label(line))
    chat = body if fmt == "openai" else anthropic_to_openai_response(body, label(line))
    return chat_to_responses(chat, label(line)) if kind == "responses" else chat


def render(b: dict, line: dict, status: str, body: dict | None = None, code: int = 0, kind: str = "", message: str = "") -> dict:
    """One line of the client's results: OpenAI's output or error file, or Anthropic's results."""
    cid = line["custom_id"]
    if b["client_format"] == "anthropic":
        if status == "succeeded":
            return {"custom_id": cid, "result": {"type": "succeeded", "message": body}}
        if status in ("canceled", "expired"):
            return {"custom_id": cid, "result": {"type": status}}
        return {"custom_id": cid, "result": {"type": "errored", "error": {"type": "error", "error": {"type": kind or "api_error", "message": message},
                                                                           "request_id": None}}}
    rid = f"batch_req_{hashlib.sha256(f'{b['id']}:{line['idx']}'.encode()).hexdigest()[:24]}"
    if status == "succeeded":
        return {"id": rid, "custom_id": cid, "response": {"status_code": 200, "request_id": rid, "body": body}, "error": None}
    if status in ("canceled", "expired"):
        return {"id": rid, "custom_id": cid, "response": None,
                "error": {"code": "batch_expired" if status == "expired" else "batch_cancelled", "message": EXPIRED if status == "expired" else CANCELLED}}
    return {"id": rid, "custom_id": cid, "response": {"status_code": code or 500, "request_id": rid,
                                                      "body": {"error": {"message": message, "type": kind or "api_error", "param": None, "code": None}}},
            "error": None}


# ---------------------------------------------------------------- planning


@dataclass
class Line:
    """One request while its batch is planned. ``fmt`` is the format the pipeline sees it in (Responses requests are
    converted to Chat Completions, as live ones are)."""

    idx: int
    custom_id: str
    kind: str  # chat, responses, embeddings (OpenAI), messages (Anthropic)
    fmt: str
    body: dict
    requested: str = ""
    est: Estimate | None = None
    decision: Decision | None = None
    cands: list[Candidate] = field(default_factory=list)
    up: dict | None = None
    estimate: float = 0.0
    error: tuple[int, str, str] | None = None  # (status, kind, message): refused before any provider saw it
    job: str | None = None
    tried: list[str] = field(default_factory=list)  # "backend: why" for routes whose provider refused the job
    # On a GPU: the route its request goes to if it hasn't run there in time, and the request as that provider takes it.
    fallback: Candidate | None = None
    fallback_up: dict | None = None

    @property
    def cand(self) -> Candidate | None:
        return self.cands[0] if self.cands else None


def parse_openai_file(data: bytes, endpoint: str) -> list[Line]:
    """The lines of an OpenAI batch input file, checked as OpenAI checks them before it accepts a batch."""
    kind = ENDPOINTS.get(endpoint)
    if kind is None:
        raise BatchError(400, f"endpoint must be one of {', '.join(ENDPOINTS)} (got {endpoint!r})")
    lines, seen = [], set()
    for n, raw in enumerate(data.decode("utf-8-sig", errors="replace").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            d = json.loads(raw)
        except ValueError:
            raise BatchError(400, f"line {n} of the input file isn't valid JSON")
        if not isinstance(d, dict) or not isinstance(d.get("body"), dict):
            raise BatchError(400, f"line {n} needs custom_id, method, url and a body object")
        cid = d.get("custom_id")
        if not isinstance(cid, str) or not cid:
            raise BatchError(400, f"line {n} has no custom_id")
        if cid in seen:
            raise BatchError(400, f"line {n}: custom_id {cid!r} is used twice; each request's must be unique")
        seen.add(cid)
        if str(d.get("method") or "POST").upper() != "POST":
            raise BatchError(400, f"line {n}: method must be POST")
        if (d.get("url") or endpoint) != endpoint:
            raise BatchError(400, f"line {n} calls {d.get('url')}, but the batch's endpoint is {endpoint}: every line must call it")
        body = dict(d["body"])
        if body.get("stream"):
            raise BatchError(400, f"line {n}: streaming isn't available in a batch (remove stream)")
        if kind == "responses":
            try:
                body = {**responses_to_chat(body), "model": body.get("model")}
            except ResponsesError as e:
                raise BatchError(400, f"line {n}: {e}")
        elif kind == "chat" and not isinstance(body.get("messages"), list):
            raise BatchError(400, f"line {n}: a chat completion needs a messages list")
        elif kind == "embeddings" and "input" not in body:
            raise BatchError(400, f"line {n}: an embeddings request needs input")
        lines.append(Line(len(lines), cid, kind, "openai", body, str(body.get("model") or "")))
    if not lines:
        raise BatchError(400, "the input file has no requests")
    return lines


def parse_anthropic(requests) -> list[Line]:
    if not isinstance(requests, list) or not requests:
        raise BatchError(400, "requests must be a non-empty list")
    lines, seen = [], set()
    for n, r in enumerate(requests):
        if not isinstance(r, dict) or not isinstance(r.get("params"), dict):
            raise BatchError(400, f"requests.{n}: needs custom_id and params")
        cid = r.get("custom_id")
        if not isinstance(cid, str) or not CUSTOM_ID.match(cid):
            raise BatchError(400, f"requests.{n}.custom_id: must be 1 to 64 letters, digits, - or _")
        if cid in seen:
            raise BatchError(400, f"requests.{n}.custom_id: {cid!r} is used twice; each request's must be unique")
        seen.add(cid)
        p = dict(r["params"])
        if p.get("stream"):
            raise BatchError(400, f"requests.{n}.params.stream: streaming isn't available in a batch")
        if not isinstance(p.get("messages"), list):
            raise BatchError(400, f"requests.{n}.params.messages: required")
        if not isinstance(p.get("max_tokens"), int) or p["max_tokens"] < 1:
            raise BatchError(400, f"requests.{n}.params.max_tokens: required, at least 1")
        lines.append(Line(n, cid, "messages", "anthropic", p, str(p.get("model") or "")))
    return lines


class Batches:
    """The gateway's side of batches. ``gw`` is the Gateway (cfg, backends, router, store, state)."""

    def __init__(self, gw):
        self.gw = gw
        self.contents = ContentCache()
        self._content: Content | None = None

    @property
    def settings(self):
        return self.gw.cfg.batches

    async def content(self) -> Content:
        """What the gateway holds for batches, sealed, and where (content.py); its keyring read again each time."""
        self._content = await self.contents.get(self.gw.cfg, self.gw.store)
        return self._content

    def api(self, backend_name: str) -> BatchAPI | None:
        b = self.gw.backends.get(backend_name)
        return batch_api_for(b, self.settings, self.gw.store, self._content) if b is not None else None

    def job_api(self, j: dict) -> BatchAPI | None:
        """A job's batch API. A GPU job's held work can be read and deleted even when its deployment isn't a backend here."""
        if j.get("api") == "gpu":
            from .gpu import GpuBatches

            b = self.gw.backends.get(j["backend"])
            return GpuBatches(b, self.settings, self.gw.store, self._content) if b is not None else GpuBatches.detached(self.settings, self._content)
        return self.api(j["backend"])

    # ------------------------------------------------------------ files

    async def upload(self, p: Principal, filename: str, purpose: str, data: bytes) -> dict:
        if purpose != "batch":
            raise BatchError(400, f"purpose must be batch (the gateway takes files for batches only), not {purpose!r}")
        if not self.settings.enabled:
            raise BatchError(403, "batches are turned off (batches.enabled)", "permission_error")
        content = await self.content()
        if content.ring.key is None:
            raise BatchError(503, "uploaded batch files are held encrypted until they're submitted: set MERITORDER_BATCH_KEY (any long "
                                  "random string, the same on every gateway) to accept them", "api_error")
        fid = new_id("file-")
        now = datetime.now(UTC)
        meta = {"id": fid, "created_at": now, "user_id": p.user_id, "team": p.team, "purpose": "batch", "filename": filename[:200],
                "bytes": len(data), "kind": "input", "expires_at": now + timedelta(days=self.settings.retention_days)}
        try:
            return await content.put_file(meta, data)
        except BackendError as e:
            raise BatchError(502, f"couldn't hold the file in {content.where}: {e.message}", "api_error")

    async def file_content(self, f: dict) -> bytes:
        if f.get("deleted_at"):
            raise BatchError(404, f"the content of {f['id']} was deleted once its batch was submitted (or its retention ended)",
                             "not_found_error")
        content = await self.content()
        try:
            return await content.file(f)
        except SealError as e:
            raise BatchError(500, f"{f['id']} can't be read: {e}", "api_error")
        except BackendError as e:
            raise BatchError(502, f"couldn't read {f['id']} from {f.get('storage')}: {e.message}", "api_error")

    async def drop_file(self, f: dict) -> None:
        """Delete an upload's content wherever it's held; its row stays, so the batch that used it can still name it."""
        content = await self.content()
        try:
            await content.drop_file(f)
        except (SealError, BackendError) as e:
            # The row is marked deleted anyway; the worker's retention pass tries the bucket again.
            log.warning("couldn't delete %s's content from %s: %s", f["id"], f.get("storage"), e)
            await self.gw.store.drop_batch_file(f["id"])

    # ------------------------------------------------------------ routing one request

    def _route(self, line: Line, team, headers: dict, now: float, util: dict[str, float] | None = None) -> None:
        cfg = self.gw.cfg
        if line.kind == "embeddings":
            return self._route_embedding(line, team)
        line.est = estimate(line.body)
        try:
            # A new task each, leveled as a live one would be (the learned router's level too, in route mode). It gets no
            # route_features row: no retry, raised floor or stopped stream can follow a batch line (decision 45). GPUs are
            # priced at their shadow price for their load now (``util``).
            d = self.gw.router.decide(requested=line.requested, team=team, body=line.body, headers=headers, session=None,
                                      prompt_tokens=line.est, boundary=True, utilization=util or {}, now=now,
                                      client="responses" if line.kind == "responses" else line.fmt, batch=True)
        except RoutingError as e:
            line.error = (e.status, e.kind, e.message)
            return
        line.decision = d
        cands = d.candidates
        max_usd = cfg.safeguards.max_request_usd
        if max_usd is not None:
            def tokens(c: Candidate) -> int:
                return c.prompt_tokens or self.gw.tokens.count(c.model.name, line.est)

            def worst(c: Candidate) -> float:
                price = c.route.batch_price(c.backend.cfg).at(tokens(c))
                out = self.gw.output_allowance(line.body, c, tokens(c), c.route.batch_price(c.backend.cfg))[0]
                return (tokens(c) * price.input + out * price.output) / 1e6

            within = [c for c in cands if worst(c) <= max_usd and self.gw.output_allowance(
                line.body, c, tokens(c), c.route.batch_price(c.backend.cfg))[0] > 0]
            if not within and cands:
                cheapest = min(cands, key=worst)
                asked = line.body.get("max_tokens") or line.body.get("max_completion_tokens")
                what = f"up to {int(asked):,} output" if asked else "before any output"
                line.error = (400, "invalid_request_error", f"this request could cost ${worst(cheapest):,.2f} at batch prices "
                              f"({tokens(cheapest):,} input tokens, {what}), above the ${max_usd:,.2f} per-request limit")
                return
            cands = within
        line.cands = cands

    def _route_embedding(self, line: Line, team) -> None:
        m = self.gw.cfg.model(line.requested)
        if m is None or m.kind != "embedding":
            line.error = (404, "not_found_error", f"embedding model '{line.requested}' is not configured")
            return
        if not team.allows_model(m.name):
            line.error = (403, "permission_error", f"model '{m.name}' is not allowed for team '{team.name}'")
            return
        tokens = count(line.body.get("input"))
        cands = []
        for r in m.routes:
            b = self.gw.backends.get(r.backend)
            if (b is None or not getattr(APIS.get(b.batch_api or ""), "embeddings", False) or not team.allows_backend(r.backend)
                    or not b.healthy(0)):
                continue
            cands.append(Candidate(model=m, route=r, backend=b, cost=tokens * r.batch_price(b.cfg).input / 1e6, warm=False, quality=0,
                                   prompt_tokens=tokens))
        if not cands:
            line.error = (400, "invalid_request_error", f"no route for '{m.name}' has a batch API that takes embeddings "
                                                        "(OpenAI's, Mistral's, OpenRouter's and the Gemini API's do; Azure's, "
                                                        "Anthropic's, xAI's and the GPUs' don't)")
            return
        cands.sort(key=lambda c: (c.backend.kind == "mock", c.cost))
        line.cands = cands

    def _upstream(self, line: Line, c: Candidate, batch_id: str) -> dict:
        if line.kind == "embeddings":
            return {**line.body, "model": c.route.batch_model or c.route.model}
        return self.gw.build_upstream(line.body, line.fmt, c, batch_id, line.decision, Session(id=batch_id),
                                      c.prompt_tokens or self.gw.tokens.count(c.model.name, line.est), batch=True)[0]

    def _shape(self, line: Line, batch_id: str) -> None:
        """The request as the chosen route's provider takes it, and its estimated cost. On a GPU, also its fallback: the
        next route with a provider batch API that takes a job of any size (not Bedrock's, which needs 100 requests) and can
        be cancelled (not OpenRouter's: a job handed on late is cancelled at the batch's deadline), shaped for it now, so the
        worker can hand the request on without routing it again. What's reserved is the fallback's estimate, since the
        request may end up there."""
        c = line.cand
        assert c is not None
        line.up = self._upstream(line, c, batch_id)
        line.estimate = c.cost
        line.fallback = line.fallback_up = None
        if c.backend.batch_api == "gpu":
            fb = next((x for x in line.cands[1:] if x.backend.batch_api != "gpu" and _can_take_handoff(self.api(x.backend.name) or _NONE)), None)
            if fb is not None:
                line.fallback, line.fallback_up = fb, self._upstream(line, fb, batch_id)
                line.estimate = max(c.cost, fb.cost)

    def _groups(self, lines: list[Line]) -> dict[tuple, list[Line]]:
        groups: dict[tuple, list[Line]] = {}
        for ln in lines:
            if ln.error is None and ln.job is None:
                api = self.api(ln.cand.backend.name)
                groups.setdefault((ln.cand.backend.name, ln.up["model"] if api.one_model else "", ln.kind == "embeddings"), []).append(ln)
        return groups

    def _jobs(self, batch_id: str, lines: list[Line]) -> list[dict]:
        """Group routed requests into provider jobs: per backend (and model, for providers that take one a job), within each
        provider's limits. A group below a provider's minimum (Bedrock: 100) moves to its requests' next routes."""
        groups = self._groups(lines)
        for _ in range(4):
            small = [(k, g) for k, g in groups.items() if len(g) < self.api(k[0]).min_requests]
            if not small:
                break
            for (backend, _, _), g in small:
                api = self.api(backend)
                for ln in g:
                    ln.tried.append(f"{backend}: a {api.name} batch job takes at least {api.min_requests} requests")
                    ln.cands = [c for c in ln.cands if c.backend.name != backend]
                    if ln.cands:
                        self._shape(ln, batch_id)
                    else:
                        ln.error = (400, "invalid_request_error", "no route could take it: " + "; ".join(ln.tried))
            groups = self._groups(lines)
        jobs = []
        for (backend, model, embeddings), group in groups.items():
            api = self.api(backend)
            if len(group) < api.min_requests:
                for ln in group:
                    ln.error = (400, "invalid_request_error", f"too few requests for {backend}'s batch API (at least {api.min_requests})")
                continue
            chunk: list[Line] = []
            size = 0
            for ln in group:
                n = api.size(f"r{ln.idx}", ln.up)
                if chunk and ((api.max_requests and len(chunk) >= api.max_requests) or (api.max_bytes and size + n > api.max_bytes)):
                    jobs.append(self._job(batch_id, backend, api, model, embeddings, chunk))
                    chunk, size = [], 0
                chunk.append(ln)
                size += n
            if chunk:
                jobs.append(self._job(batch_id, backend, api, model, embeddings, chunk))
        return jobs

    def _job(self, batch_id: str, backend: str, api: BatchAPI, model: str, embeddings: bool, lines: list[Line]) -> dict:
        jid = new_id("bj_")
        prices = {}
        fallback = {}
        for ln in lines:
            ln.job = jid
            c = ln.cand
            if api.name == "gpu":
                # Priced as live GPU work is (by the GPU time it takes, when it runs), against the model's API price, and
                # the batch price a provider would have charged: the fallback's, else the API price at the usual discount.
                standard = self.gw.api_price(c.model)
                batch = ln.fallback.route.batch_price(ln.fallback.backend.cfg) if ln.fallback else api_batch_price(self.gw.cfg, c.model)
                prices[f"{c.model.name}|{ln.up['model']}"] = {"standard": standard.model_dump(), "batch": batch.model_dump(), "gpu": True}
                if ln.fallback is not None:
                    fb = ln.fallback
                    fallback[f"r{ln.idx}"] = {"backend": fb.backend.name, "model": fb.model.name, "upstream_model": ln.fallback_up["model"],
                                              "format": fb.backend.format, "body": ln.fallback_up, "estimate": fb.cost,
                                              "prices": {"standard": fb.route.cost_price().model_dump(),
                                                         "batch": fb.route.batch_price(fb.backend.cfg).model_dump()}}
                continue
            prices[f"{c.model.name}|{ln.up['model']}"] = {"standard": c.route.cost_price().model_dump(),
                                                          "batch": c.route.batch_price(c.backend.cfg).model_dump()}
        return {"id": jid, "batch_id": batch_id, "backend": backend, "api": api.name, "model": model or None,
                "endpoint": "embeddings" if embeddings else "chat", "status": "submitting", "request_count": len(lines),
                "estimated_cost": sum(ln.estimate for ln in lines), "prices": prices,
                "provider": {"format": self.gw.backends[backend].format}, "_lines": lines, "_fallback": fallback}

    async def _check_budgets(self, p: Principal, team, amount: float, n: int, counted: bool = False) -> None:
        """Refuse a batch whose estimate doesn't fit a budget it counts against, saying which. ``counted``: the spend
        counters already hold this batch's reservation."""
        gw = self.gw
        spent = {k: await gw.spend(k, ident) - (amount if counted else 0.0) for k, ident in (("org", "all"), ("user", p.user_id), ("team", p.team))}
        if gw.cfg.safeguards.org_budget_usd_month is not None:
            spent["org"] += await gw.gpu_idle()  # the organization's budget counts GPU time no request paid for (decision 64)
        what = f"this batch's estimated cost of ${amount:,.2f} ({n:,} requests at batch prices)"
        for kind, cap, who in (("org", gw.cfg.safeguards.org_budget_usd_month, "the organization's monthly LLM budget"),
                               ("user", p.monthly_limit(team.user_budget_usd_month), f"{p.user_id}'s monthly budget"),
                               ("team", team.budget_usd_month, f"team '{p.team}''s monthly budget")):
            if cap is not None and spent[kind] + amount > cap:
                raise BatchError(429, f"{what} doesn't fit in {who} of ${cap:,.2f}: ${spent[kind]:,.2f} is already spent or reserved "
                                      "this month. Split the batch, lower max_tokens, or ask an admin to raise the budget.",
                                 "rate_limit_error", SPENDING_LIMIT)

    async def _reserve(self, b: dict, amount: float) -> None:
        if amount:
            await self.gw.state.adjust_spend(spend_keys(b["user_id"], b["team"], b["reserve_month"]), amount)

    # ------------------------------------------------------------ creating

    async def create(self, p: Principal, headers: dict, client_fmt: str, lines: list[Line], endpoint: str | None = None,
                     input_file: dict | None = None, metadata: dict | None = None) -> dict:
        gw = self.gw
        if not self.settings.enabled:
            raise BatchError(403, "batches are turned off (batches.enabled)", "permission_error")
        team = gw.cfg.team(p.team)
        bid = new_id("msgbatch_" if client_fmt == "anthropic" else "batch_")
        now = datetime.now(UTC)
        clock = now.timestamp()
        await self.content()
        util = await self.gw.gpu_utilization() if self.settings.gpu.enabled else {}
        for ln in lines:
            self._route(ln, team, headers, clock, util)
            if ln.error is None:
                self._shape(ln, bid)
        jobs = self._jobs(bid, lines)
        planned = sum(ln.estimate for ln in lines if ln.error is None)
        if not jobs:
            first = next((ln.error for ln in lines if ln.error), (400, "invalid_request_error", "no request could be routed"))
            raise BatchError(first[0], f"no request in this batch can go to a provider with a batch API: {first[2]}", first[1])
        routed = sum(1 for ln in lines if ln.error is None)
        await self._check_budgets(p, team, planned, routed)
        tags = {**parse_tags(headers.get("x-meritorder-tags", "")), **p.tags}
        b = {"id": bid, "created_at": now, "user_id": p.user_id, "team": p.team, "key_id": p.key_id, "tags": tags, "client_format": client_fmt,
             "endpoint": endpoint, "status": "in_progress", "expires_at": now + WINDOW, "metadata": metadata or {},
             "input_file_id": input_file["id"] if input_file else None, "request_count": len(lines), "estimated_cost": planned,
             "reserved_cost": planned, "reserve_month": month_key(now),
             "output_file_id": new_id("file-") if client_fmt == "openai" else None, "error_file_id": new_id("file-") if client_fmt == "openai" else None}
        await self._reserve(b, planned)  # the budget check seeded the counters, so this adds to the month's spend
        # Another batch may have reserved since that check: look again with this one counted, and refuse it if it no longer fits.
        try:
            await self._check_budgets(p, team, planned, routed, counted=True)
        except BatchError:
            await self._reserve(b, -planned)
            raise
        try:
            await gw.store.create_batch(b, [self._line_row(ln) for ln in lines], [_public_job(j) for j in jobs])
        except BaseException:
            # No batch row, so nothing would ever release the reservation: take it back now. (A gateway that dies right
            # here leaves it to the worker, which corrects spend counters to the ledger and reservations each cycle.)
            await self._reserve(b, -planned)
            raise
        if client_fmt == "openai":  # the output and error files: rows without content, joined from the providers when read
            for kind in ("output", "error"):
                await gw.store.create_batch_file({"id": b[f"{kind}_file_id"], "created_at": now, "user_id": p.user_id, "team": p.team,
                                                  "purpose": "batch_output", "filename": f"{bid}_{kind}.jsonl", "kind": kind, "batch_id": bid}, [])

        failed = await self._submit(jobs)
        if failed:  # their requests' next routes, once
            for ln in [ln for j in failed for ln in j["_lines"]]:
                ln.cands = [c for c in ln.cands if c.backend.name not in {j["backend"] for j in failed}]
                ln.job = None
                if ln.cands:
                    self._shape(ln, bid)
                else:
                    ln.error = (502, "api_error", "every route that could take it failed: " + "; ".join(ln.tried))
            again = self._jobs(bid, [ln for j in failed for ln in j["_lines"]])
            await gw.store.add_batch_jobs([_public_job(j) for j in again])
            for j in await self._submit(again):
                for ln in j["_lines"]:
                    ln.error = (502, "api_error", "every route that could take it failed: " + "; ".join(ln.tried))
                    ln.job = None
            await gw.store.update_batch_lines(bid, [self._line_row(ln) for j in failed + again for ln in j["_lines"]])
        final = sum(ln.estimate for ln in lines if ln.error is None)
        await self._reserve(b, final - planned)
        if final != planned:  # by the difference, not set: the worker may already have settled (and released) a job that ended
            await gw.store.add_batch_reserved(bid, final - planned)
        refused = sum(1 for ln in lines if ln.error)
        providers: dict[str, int] = {}
        for ln in lines:
            if ln.error is None:
                providers[ln.cand.backend.name] = providers.get(ln.cand.backend.name, 0) + 1
        update = {"estimated_cost": final, "providers": providers,
                  "counts": {"processing": len(lines) - refused, "succeeded": 0, "errored": refused, "canceled": 0, "expired": 0}}
        if refused == len(lines):
            update.update(status="ended", outcome="failed", ended_at=datetime.now(UTC), reserved_cost=0.0,
                          errors={"object": "list", "data": [{"code": "submission_failed", "line": None, "param": None,
                                                              "message": f"{ln.custom_id}: {ln.error[2]}"[:500]} for ln in lines[:10]]})
        b = await gw.store.update_batch(bid, **update)
        if input_file and refused < len(lines):
            await self.drop_file(input_file)  # it's with the providers now (or held for the GPUs, on its own)
        log.info("batch %s: %d requests to %s, %d refused, estimated $%.4f", bid, len(lines) - refused,
                 ", ".join(f"{k} ({v})" for k, v in providers.items()) or "nowhere", refused, final)
        return b

    def _reason(self, ln: Line) -> str | None:
        why = ln.decision.reason if ln.decision else ""
        c = ln.cand if ln.error is None else None
        if c is not None and c.backend.batch_api == "gpu":
            hours = self.settings.gpu.handoff_hours
            why += (f"; on {c.backend.name} at its shadow price (~${c.cost:.6f}), in its headroom at low priority"
                    + (f", else to {ln.fallback.backend.name}'s batch API after {hours:g} h" if ln.fallback else
                       ", with no other batch route: it waits for the GPU until the batch's deadline"))
        return why[:1000] or None

    def _line_row(self, ln: Line) -> dict:
        c = ln.cand if ln.error is None else None
        return {"idx": ln.idx, "custom_id": ln.custom_id, "job_id": ln.job if c else None, "kind": ln.kind, "requested_model": ln.requested or None,
                "model": c.model.name if c else None, "upstream_model": ln.up["model"] if c and ln.up else None, "backend": c.backend.name if c else None,
                "floor": str(ln.decision.floor) if ln.decision and ln.decision.floor else None,
                "route_reason": self._reason(ln), "upgraded": bool(c and c.upgraded),
                "estimated_cost": ln.estimate if c else 0.0, "status": "pending" if c else "refused",
                "error": (ln.error[2][:500] if ln.error else None), "error_code": ln.error[0] if ln.error else None, "settled": c is None}

    async def _submit(self, jobs: list[dict]) -> list[dict]:
        """Hand each job to its provider (a few at once). Returns the ones that failed, marked failed."""
        gate = asyncio.Semaphore(4)
        failed = []

        async def one(j: dict) -> None:
            api = self.api(j["backend"])
            async with gate:
                # The worker takes a job still "submitting" poller.STUCK after its created_at for one whose gateway stopped, and
                # fails it. So the clock starts as the job goes to its provider, not while it waited its turn, and the
                # provider gets SUBMIT_LIMIT (well inside STUCK) to take it: the worker and this can't both decide.
                await self.gw.store.update_batch_job(j["id"], created_at=datetime.now(UTC))
                try:
                    sub = await asyncio.wait_for(api.submit(j, [(f"r{ln.idx}", ln.up) for ln in j["_lines"]]), SUBMIT_LIMIT.total_seconds())
                except (BackendError, TimeoutError) as e:
                    why = e.message if isinstance(e, BackendError) else (
                        f"{j['backend']}: the provider didn't take the job within {SUBMIT_LIMIT.total_seconds() / 60:g} minutes")
                    log.warning("batch job %s on %s refused: %s", j["id"], j["backend"], why)
                    for ln in j["_lines"]:
                        ln.tried.append(f"{j['backend']}: {why}"[:300])
                    await self.gw.store.update_batch_job(j["id"], status="failed", error=why[:500], ended_at=datetime.now(UTC))
                    failed.append(j)
                    return
            first = 0 if j["api"] == "gpu" else self.settings.poll_seconds  # GPU work starts at the worker's next pass
            await self.gw.store.update_batch_job(j["id"], status="running", provider_id=sub.provider_id,
                                                 provider={**j["provider"], **sub.provider},
                                                 next_poll_at=datetime.now(UTC) + timedelta(seconds=first))

        await asyncio.gather(*(one(j) for j in jobs))
        return failed

    # ------------------------------------------------------------ reading, cancelling

    def visible(self, p: Principal, b: dict | None, fmt: str | None = None) -> dict:
        """The batch, if this caller may see it (its owner, or an admin) and it was made through this API."""
        if b is None or (fmt and b["client_format"] != fmt) or not (b["user_id"] == p.user_id or p.at_least("admin")):
            raise BatchError(404, "no such batch", "not_found_error")
        return b

    async def cancel(self, b: dict, by: str) -> dict:
        if b["status"] in ("ended", "finalizing") or b.get("cancel_requested_at"):
            return b  # finalizing: every job has ended, so there's nothing left to stop
        b = await self.gw.store.update_batch(b["id"], cancel_requested_at=datetime.now(UTC), cancelled_by=by,
                                             status="cancelling" if b["status"] == "in_progress" else b["status"])
        await self.content()
        for j in await self.gw.store.batch_jobs(b["id"]):
            if j["status"] != "running":
                continue
            api = self.job_api(j)
            try:
                if api is None:
                    raise BackendError(503, f"{j['backend']} isn't enabled here")
                await api.cancel(j)
                await self.gw.store.update_batch_job(j["id"], cancel_sent=True, next_poll_at=datetime.now(UTC))
            except BackendError as e:
                log.warning("couldn't cancel batch job %s on %s (the worker will try again): %s", j["id"], j["backend"], e.message)
        log.info("batch %s cancelled by %s", b["id"], by)
        return b

    async def purge(self, b: dict, deleted: bool = False) -> dict:
        """Delete what the providers (and buckets) hold for an ended batch now, rather than after the retention period.
        ``deleted``: the client deleted the batch (Anthropic's DELETE); it's no longer listed, though its costs stay."""
        await self.content()
        for j in await self.gw.store.batch_jobs(b["id"]):
            if j.get("purged_at") or not j.get("provider_id"):
                continue
            api = self.job_api(j)
            try:
                if api is None:
                    raise BackendError(503, f"{j['backend']} isn't enabled here")
                await api.purge(j)
            except BackendError as e:
                raise BatchError(502, f"couldn't delete the results held by {j['backend']}: {e.message}", "api_error")
            await self.gw.store.update_batch_job(j["id"], purged_at=datetime.now(UTC))
        fields = {"purged_at": b.get("purged_at") or datetime.now(UTC)}
        if deleted:
            fields["metadata"] = {**(b.get("metadata") or {}), "deleted": True}
        return await self.gw.store.update_batch(b["id"], **fields)

    async def results(self, b: dict, want: str | None = None) -> AsyncIterator[bytes]:
        """The batch's results in the client's format, read from the providers as they're sent. ``want``: "succeeded"
        (OpenAI's output file) or "failed" (its error file); None for Anthropic's, which has every request."""
        if b.get("purged_at"):
            raise BatchError(410, f"this batch's results were deleted {self.settings.retention_days:g} days after it ended "
                                  "(batches.retention_days)", "not_found_error")
        lines = await self.gw.store.batch_lines(b["id"])
        by_key = {f"r{ln['idx']}": ln for ln in lines}
        seen: set[str] = set()
        jobs = await self.gw.store.batch_jobs(b["id"])
        await self.content()
        return self._stream(b, by_key, seen, jobs, want)

    async def _stream(self, b: dict, by_key: dict, seen: set, jobs: list[dict], want: str | None) -> AsyncIterator[bytes]:
        def out(d: dict) -> bytes:
            return json.dumps(d, ensure_ascii=False, separators=(",", ":")).encode() + b"\n"

        for j in jobs:
            if j["status"] not in ("ended", "settled") or not j.get("provider_id"):
                continue
            api = self.job_api(j)
            if api is None:
                continue  # its results stay where they are; the lines say so below
            fmt = (j.get("provider") or {}).get("format") or "openai"
            try:
                async for o in api.results(j, want):
                    line = by_key.get(o.key)
                    if line is None or o.key in seen or line["job_id"] != j["id"]:
                        continue
                    seen.add(o.key)
                    if o.status == "canceled" and line["status"] == "expired":  # cancelled at the batch's deadline (poller.py)
                        o = replace(o, status="expired")
                    if want == "succeeded" and o.status != "succeeded" or want == "failed" and o.status == "succeeded":
                        continue
                    yield out(self._render(b, line, o, fmt))
            except BackendError as e:
                log.warning("couldn't read results of batch job %s from %s: %s", j["id"], j["backend"], e.message)
                for key, line in by_key.items():
                    if line["job_id"] == j["id"] and key not in seen:
                        seen.add(key)
                        if want != "succeeded":
                            yield out(render(b, line, "errored", code=502, kind="api_error",
                                             message=f"the result couldn't be read from {j['backend']}: {e.message}"[:500]))
        if want == "succeeded":
            return
        ended = {j["id"] for j in jobs if j["status"] in ("ended", "settled")}
        for key, line in by_key.items():
            if key in seen:
                continue
            if want == "failed" and (line["status"] == "succeeded" or (line["status"] == "pending" and line["job_id"] in ended)):
                continue  # a result from a file this read skipped (OpenAI's output file, when only errors were asked for)
            status = line["status"] if line["status"] in ("canceled", "expired") else "errored"
            message = line.get("error") or ("the provider returned no result for this request" if line["job_id"] else "refused")
            code = line.get("error_code") or (502 if line["job_id"] else 400)
            yield out(render(b, line, status, code=code, kind=kind_for_status(code), message=message))

    def _render(self, b: dict, line: dict, o: Outcome, fmt: str) -> dict:
        if o.status == "succeeded":
            try:
                return render(b, line, "succeeded", client_body(b, line, fmt, o.body or {}))
            except Exception as e:  # noqa: BLE001 (a reply that doesn't convert is that request's error, not the whole file's)
                return render(b, line, "errored", code=502, kind="api_error", message=f"the reply couldn't be converted: {type(e).__name__}: {e}")
        return render(b, line, o.status, code=o.code, kind=o.kind, message=o.message)


def _public_job(j: dict) -> dict:
    return {k: v for k, v in j.items() if not k.startswith("_")}


class _NoApi:
    min_requests = 1
    cancels = True


_NONE = _NoApi()


def _can_take_handoff(api) -> bool:
    """A GPU's fallback: a job of any size, and one the gateway can stop at the batch's deadline."""
    return api.min_requests <= 1 and api.cancels
