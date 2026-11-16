"""Each provider's batch API, behind one interface: submit a job, poll it, read its results, cancel it, delete what's left.

A job holds requests already shaped for the backend (proxy.Gateway.build_upstream: Anthropic Messages or Chat Completions,
or an embeddings request), each with a key (``r<line>``) that comes back with its result. Results come back the same way:
an Anthropic message or a Chat Completions reply (embeddings: the embeddings reply), whatever the provider's own format.

Written from the providers' API references as of November 2026 (docs/02-request-flow.md#batches-work-that-can-wait lists what was followed
and what still needs checking with a real account):

* Anthropic Message Batches (``/v1/messages/batches``): requests inline; models may mix; results as JSON lines.
* OpenAI Batch API (``/v1/files`` then ``/v1/batches``), and Azure OpenAI's global and data-zone batch on its v1 API
  (``/openai/v1/...``): one model (Azure: one batch deployment) a file; output and error files.
* Moonshot's (Kimi), OpenAI's shape with its own window and fewer sampling fields.
* Mistral's (``/v1/files`` then ``/v1/batch/jobs``): one model a job; output and error files. Chat and embeddings.
* xAI's (``/v1/batches``, then its requests added in pieces, results read in pages): models may mix; chat only.
* OpenRouter's (``/v1/batches`` with the requests inline, results on the batch once it's done): one model a batch.
* The Gemini API's batch mode (``models/{model}:batchGenerateContent``, and for embeddings
  ``models/{model}:asyncBatchEmbedContent``, each with an uploaded JSONL file): one model a batch.
* Bedrock batch inference (``CreateModelInvocationJob``): input and output in an S3 bucket the operator gives; Claude only.
* Vertex AI batch prediction (``batchPredictionJobs``): input and output in a Cloud Storage bucket; Gemini and Claude.

The in-process mock backend has one too, so batches work with no keys (its "provider" keeps results in the shared state),
and so do self-hosted GPUs (gpu.py), where the gateway runs the requests itself.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from urllib.parse import quote

import httpx

from ..backends.auth import BackendAuthProvider, SigV4Auth
from ..backends.base import Backend, BackendError
from ..config import BatchesConfig
from ..formats import gemini_upstream as gemini
from ..formats.responses_upstream import chat_to_responses_request, responses_to_chat_completion
from .buckets import Bucket, GCSBucket, S3Bucket, _AwsSigner, call, json_lines, split_bucket  # noqa: F401 (split_bucket: re-exported)

MB = 1024 * 1024
KEY = re.compile(r"^r\d+$")


@dataclass
class Submitted:
    provider_id: str
    provider: dict = field(default_factory=dict)  # file ids, bucket paths: whatever polling, results and purging need


@dataclass
class Polled:
    ended: bool
    provider_status: str
    counts: dict[str, int] = field(default_factory=dict)  # processing, succeeded, errored, canceled, expired (as far as known)
    provider: dict = field(default_factory=dict)  # merged into the job's provider record (result files, output folders)
    expired: bool = False  # the provider gave up on what hadn't run when the window closed
    failed: str | None = None  # the provider failed the job as a whole (it rejected the input), and why


@dataclass
class Outcome:
    key: str
    status: str  # succeeded, errored, canceled, expired
    body: dict | None = None  # succeeded: the reply, in the backend's format
    code: int = 200  # errored: the status the provider gave the request
    kind: str = ""  # errored: the provider's error type or code
    message: str = ""


def _counts(processing=0, succeeded=0, errored=0, canceled=0, expired=0) -> dict[str, int]:
    return {"processing": int(processing or 0), "succeeded": int(succeeded or 0), "errored": int(errored or 0),
            "canceled": int(canceled or 0), "expired": int(expired or 0)}


def _jsonl(rows: list[dict]) -> bytes:
    return b"".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")).encode() + b"\n" for r in rows)


def _status_kind(code: int) -> str:
    return {400: "invalid_request_error", 401: "authentication_error", 403: "permission_error", 404: "not_found_error",
            413: "request_too_large", 429: "rate_limit_error", 529: "overloaded_error"}.get(code, "api_error")


def _chat_outcome(key: str, body: dict | None) -> Outcome:
    """A converted reply that is an error (Gemini's blocked or malformed replies come back as {"error": ...})."""
    err = (body or {}).get("error")
    if body is None or err:
        err = err if isinstance(err, dict) else {"message": str(err or "no reply")}
        code = err.get("code") if isinstance(err.get("code"), int) else 500
        return Outcome(key, "errored", code=code, kind=str(err.get("type") or err.get("status") or _status_kind(code)),
                       message=str(err.get("message") or "the provider returned an error"))
    return Outcome(key, "succeeded", body)


class BatchAPI:
    """One backend's batch API. ``one_model``: a job may hold one model's requests only."""

    name = ""
    one_model = True
    embeddings = False  # it takes embeddings requests too
    max_requests: int | None = None
    max_bytes: int | None = None
    min_requests = 1
    # The gateway can stop a job (cancel). A GPU hands its work only to an API that can: a provider job started late gets a
    # fresh window, and the gateway cancels it at the batch's own deadline so the client's 24 hours hold.
    cancels = True

    def __init__(self, backend: Backend, settings: BatchesConfig, store=None, content=None):
        self.backend = backend
        self.cfg = backend.cfg
        self.settings = settings
        self.store = store  # only the mock's "provider" keeps anything, and it keeps it here
        self.content = content  # what the gateway holds, sealed (content.Content): only the GPUs' "provider" keeps anything there
        b = backend.cfg.batch
        self.max_requests = b.max_requests or self.max_requests
        self.max_bytes = b.max_bytes or self.max_bytes
        self.min_requests = b.min_requests or self.min_requests

    @property
    def retention_seconds(self) -> int:
        return int(self.settings.retention_days * 86400)

    # --- HTTP

    async def _request(self, method: str, url: str, *, body: dict | None = None, content: bytes | None = None,
                       headers: dict | None = None, auth: BackendAuthProvider | None = None, what: str = "") -> httpx.Response:
        data = content if content is not None else json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode() if body is not None else b""
        hdrs = {**({"content-type": "application/json"} if body is not None else {}), **(headers or {})}
        return await call(self.backend.client, auth or self.backend.auth, self.backend.name, method, url, content=data, headers=hdrs, what=what)

    async def _json(self, method: str, url: str, **kw) -> dict:
        r = await self._request(method, url, **kw)
        try:
            out = r.json() if r.content else {}
        except ValueError:
            raise BackendError(502, f"{self.backend.name}: {kw.get('what') or 'batch'}: the provider's reply isn't JSON", retryable=True)
        return out if isinstance(out, dict) else {}

    def _lines(self, url: str, headers: dict | None = None, auth: BackendAuthProvider | None = None, what: str = "") -> AsyncIterator[dict]:
        """A JSON-lines download, read line by line: results pass through the gateway without being kept."""
        return json_lines(self.backend.client, auth or self.backend.auth, self.backend.name, url, headers, what)

    # --- the interface

    def size(self, key: str, body: dict) -> int:
        """Roughly what one request adds to the job's upload, for splitting by ``max_bytes``."""
        return len(json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode()) + len(key) + 64

    async def submit(self, job: dict, requests: list[tuple[str, dict]]) -> Submitted:
        raise NotImplementedError

    async def poll(self, job: dict) -> Polled:
        raise NotImplementedError

    def results(self, job: dict, want: str | None = None) -> AsyncIterator[Outcome]:
        """Every result of an ended job. ``want``: "succeeded" or "failed" when only those are needed (skips a download)."""
        raise NotImplementedError

    async def cancel(self, job: dict) -> None:
        raise NotImplementedError

    async def purge(self, job: dict) -> None:
        """Delete what the provider (or the bucket) still holds for the job: its input, its results."""
        raise NotImplementedError


# ====================================================================== Anthropic


class AnthropicBatches(BatchAPI):
    name = "anthropic"
    one_model = False  # each request names its model
    max_requests = 100_000
    max_bytes = 256 * MB

    def _base(self) -> str:
        return self.cfg.base_url.rstrip("/")

    def _headers(self) -> dict:
        return {"anthropic-version": self.cfg.anthropic_version or "2023-06-01"}

    async def submit(self, job, requests):
        reqs = [{"custom_id": key, "params": {k: v for k, v in body.items() if k != "stream"}} for key, body in requests]
        d = await self._json("POST", f"{self._base()}/v1/messages/batches", body={"requests": reqs}, headers=self._headers(),
                             what="creating a message batch")
        return Submitted(str(d.get("id")), {"results_url": d.get("results_url")})

    async def poll(self, job):
        d = await self._json("GET", f"{self._base()}/v1/messages/batches/{job['provider_id']}", headers=self._headers(), what="checking a batch")
        c = d.get("request_counts") or {}
        status = str(d.get("processing_status") or "")
        return Polled(status == "ended", status, _counts(**{k: c.get(k) for k in ("processing", "succeeded", "errored", "canceled", "expired")}),
                      {"results_url": d.get("results_url")} if d.get("results_url") else {}, expired=bool(c.get("expired")))

    async def results(self, job, want=None):
        url = (job.get("provider") or {}).get("results_url") or f"{self._base()}/v1/messages/batches/{job['provider_id']}/results"
        async for d in self._lines(url, headers=self._headers(), what="reading batch results"):
            key, res = str(d.get("custom_id") or ""), d.get("result") or {}
            kind = res.get("type")
            if kind == "succeeded":
                yield Outcome(key, "succeeded", res.get("message") or {})
            elif kind in ("canceled", "expired"):
                yield Outcome(key, kind)
            else:
                err = ((res.get("error") or {}).get("error") or {})
                kind = str(err.get("type") or "api_error")
                code = {"invalid_request_error": 400, "authentication_error": 401, "permission_error": 403, "not_found_error": 404,
                        "request_too_large": 413, "rate_limit_error": 429, "overloaded_error": 529}.get(kind, 500)
                yield Outcome(key, "errored", code=code, kind=kind, message=str(err.get("message") or "the request failed"))

    async def cancel(self, job):
        await self._request("POST", f"{self._base()}/v1/messages/batches/{job['provider_id']}/cancel", headers=self._headers(),
                            what="cancelling a batch")

    async def purge(self, job):
        await self._request("DELETE", f"{self._base()}/v1/messages/batches/{job['provider_id']}", headers=self._headers(),
                            what="deleting a batch")


# ====================================================================== OpenAI and Azure OpenAI


def _openai_outcome(d: dict) -> Outcome:
    key, resp, err = str(d.get("custom_id") or ""), d.get("response"), d.get("error")
    if isinstance(resp, dict) and resp.get("status_code") is not None:
        code, body = int(resp.get("status_code") or 500), resp.get("body")
        if code < 400 and isinstance(body, dict) and not body.get("error"):
            return Outcome(key, "succeeded", body)
        e = (body or {}).get("error") if isinstance(body, dict) else None
        e = e if isinstance(e, dict) else {}
        return Outcome(key, "errored", code=code, kind=str(e.get("type") or e.get("code") or _status_kind(code)),
                       message=str(e.get("message") or "the request failed"))
    e = err if isinstance(err, dict) else {}
    code = str(e.get("code") or "")
    if code == "batch_expired":
        return Outcome(key, "expired")
    if code in ("batch_cancelled", "batch_canceled"):
        return Outcome(key, "canceled")
    return Outcome(key, "errored", code=500, kind=code or "api_error", message=str(e.get("message") or "no result"))


class OpenAIBatches(BatchAPI):
    """OpenAI's Batch API. Azure OpenAI's v1 API (``{endpoint}/openai/v1``) takes the same calls, with a global-batch or
    data-zone-batch deployment as the model, ``/chat/completions`` as the batch's endpoint, and dotted form fields."""

    name = "openai"
    embeddings = True
    max_requests = 50_000
    max_bytes = 200 * MB
    expires: tuple[int, int] | None = (3600, 2_592_000)  # what output_expires_after and expires_after accept, in seconds
    window = "24h"

    def _base(self) -> str:
        return self.cfg.base_url.rstrip("/")

    def path(self, job: dict) -> str:
        """The endpoint each line calls: the job's work, on the API the backend uses."""
        if job.get("endpoint") == "embeddings":
            return "/v1/embeddings"
        return "/v1/responses" if self.cfg.api == "responses" else "/v1/chat/completions"

    def endpoint(self, job: dict) -> str:
        return self.path(job)

    def expires_field(self, name: str) -> str:
        return f"{name}[{{}}]"

    def shape(self, job: dict, body: dict) -> dict:
        body = {k: v for k, v in body.items() if k not in ("stream", "stream_options")}
        if job.get("endpoint") == "embeddings":
            return body
        if self.cfg.api == "responses":
            return chat_to_responses_request(body)
        f = self.cfg.max_tokens_field
        if f != "max_tokens" and "max_tokens" in body:
            body[f] = body.pop("max_tokens")
        return body

    def reply(self, job: dict, body: dict) -> dict:
        if job.get("endpoint") != "embeddings" and self.cfg.api == "responses":
            return responses_to_chat_completion(body)
        return body

    def _seconds(self) -> int:
        lo, hi = self.expires or (0, 0)
        return max(lo, min(hi, self.retention_seconds))

    def line(self, job: dict, key: str, body: dict) -> dict:
        return {"custom_id": key, "method": "POST", "url": self.path(job), "body": self.shape(job, body)}

    def upload_fields(self) -> list[tuple[str, str]]:
        if not self.expires:
            return [("purpose", "batch")]
        ex = self.expires_field("expires_after")
        return [("purpose", "batch"), (ex.format("anchor"), "created_at"), (ex.format("seconds"), str(self._seconds()))]

    async def upload(self, job: dict, data: bytes) -> str:
        """The job's JSONL through the Files API (multipart, ``purpose: batch``). Returns the file's id."""
        boundary = uuid.uuid4().hex
        form = b"".join(f'--{boundary}\r\nContent-Disposition: form-data; name="{n}"\r\n\r\n{v}\r\n'.encode() for n, v in self.upload_fields())
        form += (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{job["id"]}.jsonl"\r\n'
                 f"Content-Type: application/jsonl\r\n\r\n").encode() + data + f"\r\n--{boundary}--\r\n".encode()
        f = await self._json("POST", f"{self._base()}/files", content=form, headers={"content-type": f"multipart/form-data; boundary={boundary}"},
                             what="uploading a batch file")
        return str(f.get("id") or "")

    def create_body(self, job: dict, file_id: str) -> dict:
        body = {"input_file_id": file_id, "endpoint": self.endpoint(job), "completion_window": self.window,
                "metadata": {"meritorder_job": job["id"], "meritorder_batch": job["batch_id"]}}
        if self.expires:
            body["output_expires_after"] = {"anchor": "created_at", "seconds": self._seconds()}
        return body

    def create_url(self) -> str:
        return f"{self._base()}/batches"

    async def submit(self, job, requests):
        file_id = await self.upload(job, _jsonl([self.line(job, key, body) for key, body in requests]))
        try:
            b = await self._json("POST", self.create_url(), what="creating a batch", body=self.create_body(job, file_id))
        except BackendError:
            try:
                await self._request("DELETE", f"{self._base()}/files/{file_id}", what="deleting a batch file")
            except BackendError:
                pass
            raise
        return Submitted(str(b.get("id")), {"input_file_id": file_id})

    async def poll(self, job):
        d = await self._json("GET", f"{self._base()}/batches/{job['provider_id']}", what="checking a batch")
        status = str(d.get("status") or "")
        c = d.get("request_counts") or {}
        total, done, failed = int(c.get("total") or 0), int(c.get("completed") or 0), int(c.get("failed") or 0)
        ended = status in ("completed", "failed", "expired", "cancelled")
        files = {k: d[k] for k in ("output_file_id", "error_file_id") if d.get(k)}
        why = None
        if status == "failed":
            errs = ((d.get("errors") or {}).get("data") or [])
            why = "; ".join(str(e.get("message") or e.get("code")) for e in errs[:3]) or "the provider failed the batch"
        return Polled(ended, status, _counts(max(total - done - failed, 0) if not ended else 0, done, failed), files,
                      expired=status == "expired", failed=why)

    async def results(self, job, want=None):
        p = job.get("provider") or {}
        files = [] if want == "failed" else [p.get("output_file_id")]
        files += [] if want == "succeeded" else [p.get("error_file_id")]
        for file_id in files:
            if not file_id:
                continue
            async for d in self._lines(f"{self._base()}/files/{file_id}/content", what="reading batch results"):
                o = _openai_outcome(d)
                if o.status == "succeeded":
                    o.body = self.reply(job, o.body or {})
                yield o

    async def cancel(self, job):
        await self._request("POST", f"{self._base()}/batches/{job['provider_id']}/cancel", what="cancelling a batch")

    async def purge(self, job):
        p = job.get("provider") or {}
        for key in ("input_file_id", "output_file_id", "error_file_id"):
            if p.get(key):
                try:
                    await self._request("DELETE", f"{self._base()}/files/{p[key]}", what="deleting a batch file")
                except BackendError as e:
                    if e.status != 404:
                        raise


class AzureBatches(OpenAIBatches):
    """Azure OpenAI's global and data-zone batch. No embeddings: Azure lists no embedding model for its batch deployment
    types (November 2026), so embeddings batches go to the other routes."""

    name = "azure"
    embeddings = False
    max_requests = 100_000
    expires = (1_209_600, 2_592_000)  # Azure takes 14 to 30 days

    def endpoint(self, job: dict) -> str:
        return self.path(job).removeprefix("/v1")

    def expires_field(self, name: str) -> str:
        return f"{name}.{{}}"


class MoonshotBatches(OpenAIBatches):
    """Moonshot's (Kimi) batch API: OpenAI's calls, with a window written in days, no file expiry fields, chat only, and
    no sampling fields (temperature, top_p, n, the penalties), which it refuses in a batch."""

    name = "moonshot"
    embeddings = False
    max_requests = None
    max_bytes = 100 * MB
    expires = None
    window = "1d"
    REFUSED = ("temperature", "top_p", "n", "presence_penalty", "frequency_penalty")

    def path(self, job: dict) -> str:
        return "/v1/chat/completions"

    def shape(self, job: dict, body: dict) -> dict:
        return {k: v for k, v in super().shape(job, body).items() if k not in self.REFUSED}


class MistralBatches(OpenAIBatches):
    """Mistral's batch API: the JSONL through its Files API, then ``/v1/batch/jobs`` for one model; output and error files
    like OpenAI's. Chat and embeddings."""

    name = "mistral"
    max_requests = 1_000_000
    max_bytes = 512 * MB
    expires = None
    ENDED = ("SUCCESS", "FAILED", "TIMEOUT_EXCEEDED", "CANCELLED")

    def path(self, job: dict) -> str:
        return "/v1/embeddings" if job.get("endpoint") == "embeddings" else "/v1/chat/completions"

    def line(self, job: dict, key: str, body: dict) -> dict:
        return {"custom_id": key, "body": self.shape(job, body)}

    def create_url(self) -> str:
        return f"{self._base()}/batch/jobs"

    def create_body(self, job: dict, file_id: str) -> dict:
        return {"input_files": [file_id], "endpoint": self.path(job), "model": job.get("model"), "timeout_hours": 24,
                "metadata": {"meritorder_job": job["id"], "meritorder_batch": job["batch_id"]}}

    async def poll(self, job):
        d = await self._json("GET", f"{self._base()}/batch/jobs/{job['provider_id']}", what="checking a batch job")
        status = str(d.get("status") or "").upper()
        total, ok, bad = (int(d.get(k) or 0) for k in ("total_requests", "succeeded_requests", "failed_requests"))
        ended = status in self.ENDED
        files = {k: d[f] for k, f in (("output_file_id", "output_file"), ("error_file_id", "error_file")) if d.get(f)}
        why = None
        if status == "FAILED":
            why = "; ".join(str(e.get("message")) for e in (d.get("errors") or [])[:3] if isinstance(e, dict)) or "the provider failed the job"
        return Polled(ended, status.lower(), _counts(max(total - ok - bad, 0) if not ended else 0, ok, bad), files,
                      expired=status == "TIMEOUT_EXCEEDED", failed=why)

    async def cancel(self, job):
        await self._request("POST", f"{self._base()}/batch/jobs/{job['provider_id']}/cancel", what="cancelling a batch job")

    async def purge(self, job):
        await super().purge(job)
        try:  # the job's record (in Mistral's SDK, not yet its API reference)
            await self._request("DELETE", f"{self._base()}/batch/jobs/{job['provider_id']}", what="deleting a batch job")
        except BackendError as e:
            if e.status not in (404, 405):
                raise


class OpenRouterBatches(BatchAPI):
    """OpenRouter's batch API: the requests inline in one call (``requests`` last: it reads the body as a stream), the
    results on the batch once it's done. Only models with a batch variant take it (others are refused, and their requests
    move to their next route). It has no cancel: a cancelled batch runs to its end, and what ran is billed."""

    name = "openrouter"
    embeddings = True
    cancels = False
    max_bytes = 200 * MB

    def _base(self) -> str:
        return self.cfg.base_url.rstrip("/")

    def path(self, job: dict) -> str:
        return "/v1/embeddings" if job.get("endpoint") == "embeddings" else "/v1/chat/completions"

    async def submit(self, job, requests):
        reqs = [{"custom_id": key, "body": {k: v for k, v in body.items() if k not in ("stream", "stream_options")}} for key, body in requests]
        d = await self._json("POST", f"{self._base()}/batches", what="creating a batch", body={
            "endpoint": self.path(job), "model": job.get("model"), "completion_window": "24h", "requests": reqs})
        return Submitted(str(d.get("id")), {})

    async def _batch(self, job: dict) -> dict:
        return await self._json("GET", f"{self._base()}/batches/{job['provider_id']}", what="checking a batch")

    async def poll(self, job):
        d = await self._batch(job)
        status = str(d.get("status") or "")
        c = d.get("request_counts") or {}
        total, done, failed = int(c.get("total") or 0), int(c.get("completed") or 0), int(c.get("failed") or 0)
        ended = status in ("completed", "failed", "expired", "cancelled")
        why = str((d.get("error") or {}).get("message") or "the provider failed the batch") if status == "failed" else None
        return Polled(ended, status, _counts(max(total - done - failed, 0) if not ended else 0, done, failed), expired=status == "expired",
                      failed=why)

    async def results(self, job, want=None):
        for r in (await self._batch(job)).get("results") or []:
            if not isinstance(r, dict):
                continue
            if isinstance(r.get("response"), dict):
                yield _openai_outcome(r)
                continue
            e = r.get("error") if isinstance(r.get("error"), dict) else {}
            kind = str(e.get("type") or e.get("error_type") or "api_error")
            if kind in ("expired", "batch_expired"):
                yield Outcome(str(r.get("custom_id") or ""), "expired")
                continue
            code = {"invalid_request_error": 400, "authentication_error": 401, "permission_error": 403, "not_found_error": 404,
                    "rate_limit_error": 429}.get(kind, 500)
            yield Outcome(str(r.get("custom_id") or ""), "errored", code=code, kind=kind, message=str(e.get("message") or "the request failed"))

    async def cancel(self, job):
        return None  # no cancel at OpenRouter: the batch runs to its end

    async def purge(self, job):
        try:
            await self._request("DELETE", f"{self._base()}/batches/{job['provider_id']}", what="deleting a batch")
        except BackendError as e:
            if e.status not in (404, 410):
                raise


class XaiBatches(BatchAPI):
    """xAI's batch API: a named batch, its requests added in pieces, their results read in pages. Models may mix; chat
    only. It takes 20% off for the models it lists (others pay the standard price: set their routes' ``price.batch``)."""

    name = "xai"
    one_model = False
    PIECE = 500  # requests a call adds
    PAGE = 1000  # results a page reads (its most)

    def _base(self) -> str:
        return self.cfg.base_url.rstrip("/")

    async def submit(self, job, requests):
        d = await self._json("POST", f"{self._base()}/batches", body={"name": job["id"]}, what="creating a batch")
        bid = str(d.get("batch_id") or d.get("id") or "")
        try:
            for i in range(0, len(requests), self.PIECE):
                rows = [{"batch_request_id": key, "batch_request": {"chat_get_completion": {k: v for k, v in body.items() if k != "stream"}}}
                        for key, body in requests[i:i + self.PIECE]]
                await self._json("POST", f"{self._base()}/batches/{bid}/requests", body={"batch_requests": rows}, what="adding batch requests")
        except BackendError:
            try:
                await self._request("POST", f"{self._base()}/batches/{bid}:cancel", body={}, what="cancelling a batch")
            except BackendError:
                pass
            raise
        return Submitted(bid, {})

    async def poll(self, job):
        d = await self._json("GET", f"{self._base()}/batches/{job['provider_id']}", what="checking a batch")
        st = d.get("state") or {}
        total, pending = int(st.get("num_requests") or 0), int(st.get("num_pending") or 0)
        ok, bad, cancelled = int(st.get("num_success") or 0), int(st.get("num_error") or 0), int(st.get("num_cancelled") or 0)
        ended = pending == 0 and total > 0 or bool(d.get("cancel_time")) and pending == 0
        status = "ended" if ended else "cancelling" if d.get("cancel_time") else "running"
        return Polled(ended, status, _counts(pending, ok, bad, cancelled))

    async def results(self, job, want=None):
        token = None
        while True:
            url = f"{self._base()}/batches/{job['provider_id']}/results?limit={self.PAGE}" + (f"&pagination_token={quote(token, safe='')}" if token else "")
            d = await self._json("GET", url, what="reading batch results")
            for r in d.get("results") or []:
                key, res = str(r.get("batch_request_id") or ""), r.get("batch_result") or {}
                reply = (res.get("response") or {}).get("chat_get_completion")
                if isinstance(reply, dict):
                    yield _chat_outcome(key, reply)
                else:
                    err = res.get("error")
                    message = err.get("message") if isinstance(err, dict) else err
                    yield Outcome(key, "errored", code=500, kind="api_error", message=str(message or "the request failed"))
            token = d.get("pagination_token")
            if not token:
                return

    async def cancel(self, job):
        await self._request("POST", f"{self._base()}/batches/{job['provider_id']}:cancel", body={}, what="cancelling a batch")

    async def purge(self, job):
        return None  # xAI has no call to delete a batch: its results stay with xAI under its own retention


# ====================================================================== Gemini API


def _gemini_state(value) -> str:
    return re.sub(r"^(BATCH_STATE_|JOB_STATE_)", "", str(value or "")).upper()


class GeminiBatches(BatchAPI):
    """The Gemini API's batch mode: the requests go up as a JSONL file through the Files API, and come back as one. Chat
    goes to ``batchGenerateContent``; embeddings to ``asyncBatchEmbedContent``, one Gemini request per input (a Gemini
    request gives one embedding), joined back into one embeddings reply per request."""

    name = "gemini"
    embeddings = True

    def _base(self) -> str:
        return self.cfg.base_url.rstrip("/")

    def _host(self) -> str:
        m = re.match(r"^(https?://[^/]+)(/.*)?$", self._base())
        return m.group(1) if m else self._base()

    def _version(self) -> str:
        return self._base()[len(self._host()):] or "/v1beta"

    async def submit(self, job, requests):
        if job.get("endpoint") == "embeddings":
            rows = [{"key": f"{key}:{j}:{len(texts)}", "request": gemini.embed_request(text, body)}
                    for key, body in requests for texts in [gemini.embed_inputs(body)] for j, text in enumerate(texts)]
            data, method = _jsonl(rows), "asyncBatchEmbedContent"
        else:
            data, method = _jsonl([{"key": key, "request": gemini.chat_to_gemini_request(body)} for key, body in requests]), "batchGenerateContent"
        start = await self._request("POST", f"{self._host()}/upload{self._version()}/files", body={"file": {"display_name": job["id"]}},
                                    headers={"x-goog-upload-protocol": "resumable", "x-goog-upload-command": "start",
                                             "x-goog-upload-header-content-length": str(len(data)),
                                             "x-goog-upload-header-content-type": "application/jsonl"}, what="starting a file upload")
        url = start.headers.get("x-goog-upload-url")
        if not url:
            raise BackendError(502, f"{self.backend.name}: the Files API gave no upload URL", retryable=True)
        up = await self._json("POST", url, content=data, headers={"x-goog-upload-offset": "0", "x-goog-upload-command": "upload, finalize",
                                                                    "content-type": "application/jsonl"}, what="uploading a batch file")
        name = str((up.get("file") or {}).get("name") or "")
        op = await self._json("POST", f"{self._base()}/models/{gemini.model_id(job['model'])}:{method}", what="creating a batch",
                              body={"batch": {"display_name": job["id"], "input_config": {"file_name": name}}})
        return Submitted(str(op.get("name") or (op.get("metadata") or {}).get("name")), {"input_file": name})

    async def poll(self, job):
        d = await self._json("GET", f"{self._base()}/{job['provider_id']}", what="checking a batch")
        meta = d.get("metadata") or d
        state = _gemini_state(meta.get("state"))
        s = meta.get("batchStats") or {}
        done, failed = int(s.get("successfulRequestCount") or 0), int(s.get("failedRequestCount") or 0)
        out = meta.get("output") or (d.get("response") or {}).get("output") or d.get("response") or {}
        files = {"responses_file": out["responsesFile"]} if out.get("responsesFile") else {}
        ended = state in ("SUCCEEDED", "FAILED", "CANCELLED", "EXPIRED")
        return Polled(ended, state.lower(), _counts(s.get("pendingRequestCount"), done, failed), files, expired=state == "EXPIRED",
                      failed=str((d.get("error") or {}).get("message") or "the provider failed the batch") if state == "FAILED" and not done else None)

    async def results(self, job, want=None):
        f = (job.get("provider") or {}).get("responses_file")
        if not f:
            return
        lines = self._lines(f"{self._host()}/download{self._version()}/{f}:download?alt=media", what="reading batch results")
        if job.get("endpoint") == "embeddings":
            async for o in gemini_embeddings(lines, job.get("model") or ""):
                yield o
            return
        async for d in lines:
            key = str(d.get("key") or (d.get("metadata") or {}).get("key") or "")
            if d.get("error") and not d.get("response"):
                e = d["error"] if isinstance(d["error"], dict) else {"message": str(d["error"])}
                code = e.get("code") if isinstance(e.get("code"), int) else 500
                yield Outcome(key, "errored", code=code, kind=str(e.get("status") or _status_kind(code)), message=str(e.get("message") or ""))
                continue
            yield _chat_outcome(key, gemini.gemini_to_chat_completion(d.get("response") or {}, job.get("model") or ""))

    async def cancel(self, job):
        await self._request("POST", f"{self._base()}/{job['provider_id']}:cancel", body={}, what="cancelling a batch")

    async def purge(self, job):
        p = job.get("provider") or {}
        for path in (job.get("provider_id"), p.get("input_file"), p.get("responses_file")):
            if path:
                try:
                    await self._request("DELETE", f"{self._base()}/{path}", what="deleting a batch")
                except BackendError as e:
                    if e.status != 404:
                        raise


async def gemini_embeddings(lines: AsyncIterator[dict], model: str) -> AsyncIterator[Outcome]:
    """Gemini's embedding results (one per input, keyed ``r<line>:<input>:<inputs>``) joined into one embeddings reply per
    request, as OpenAI's API gives it. A request any of whose inputs failed fails."""
    parts: dict[str, dict[int, dict]] = {}
    counts: dict[str, int] = {}
    errors: dict[str, Outcome] = {}

    def done(key: str) -> Outcome:
        got = parts.pop(key)
        if key in errors:
            return errors.pop(key)
        return Outcome(key, "succeeded", gemini.embeddings_reply([got[j] for j in sorted(got)], model))

    async for d in lines:
        raw = str(d.get("key") or (d.get("metadata") or {}).get("key") or "")
        key, _, rest = raw.partition(":")
        j, _, n = rest.partition(":")
        idx, total = (int(j), int(n)) if j.isdigit() and n.isdigit() else (0, 1)
        counts[key] = total
        got = parts.setdefault(key, {})
        if d.get("error") and not d.get("response"):
            e = d["error"] if isinstance(d["error"], dict) else {"message": str(d["error"])}
            code = e.get("code") if isinstance(e.get("code"), int) else 500
            errors.setdefault(key, Outcome(key, "errored", code=code, kind=str(e.get("status") or _status_kind(code)),
                                           message=str(e.get("message") or "")))
        got[idx] = d.get("response") or {}
        if len(got) >= total:
            yield done(key)
    for key in list(parts):
        if key in errors:
            yield errors.pop(key)
        else:
            yield Outcome(key, "errored", code=502, kind="api_error", message=f"the provider returned {len(parts[key])} of "
                                                                                  f"{counts[key]} embeddings for this request")


# ====================================================================== buckets (Bedrock: S3; Vertex AI: Cloud Storage)


class BucketBatches(BatchAPI):
    """A batch API that reads its input from a bucket and writes its results there (buckets.py). Each job gets its own folder."""

    bucket_class: type[Bucket] = Bucket

    def objects(self) -> Bucket:
        return self.bucket_class(self.cfg.batch.bucket or "", self.backend.client, self.storage_auth(), self.backend.name, self.region())

    def storage_auth(self) -> BackendAuthProvider:
        return self.backend.auth

    def region(self) -> str:
        return ""

    def folder(self, job: dict) -> str:
        return self.objects().name(job["batch_id"], job["id"])

    def bucket(self) -> str:
        return self.objects().bucket

    async def put(self, name: str, data: bytes) -> None:
        await self.objects().put(name, data, "application/jsonl")

    async def list(self, prefix: str) -> list[str]:
        return await self.objects().list(prefix)

    def read(self, name: str) -> AsyncIterator[dict]:
        return self.objects().lines(name)

    async def delete(self, name: str) -> None:
        await self.objects().delete(name)

    async def purge(self, job):
        for name in await self.list(self.folder(job) + "/"):
            await self.delete(name)


# ====================================================================== Bedrock


class BedrockBatches(BucketBatches):
    """Bedrock batch inference for Claude: a JSONL file of InvokeModel bodies in S3, results next to it."""

    name = "bedrock"
    bucket_class = S3Bucket
    max_requests = 100_000
    max_bytes = 1024 * MB
    min_requests = 100  # Bedrock's default quota: fewer records and the job is refused

    def region(self) -> str:
        return self.cfg.batch.region or getattr(self.backend.auth, "region", None) or self.cfg.auth.region or "us-east-1"

    def signer(self, service: str) -> BackendAuthProvider:
        a = self.backend.auth
        return _AwsSigner(a, service, self.region()) if isinstance(a, SigV4Auth) else a

    def storage_auth(self) -> BackendAuthProvider:
        return self.signer("s3")

    def _control(self) -> str:
        return f"https://bedrock.{self.region()}.amazonaws.com"

    async def submit(self, job, requests):
        folder = self.folder(job)
        rows = []
        for key, body in requests:
            body = {k: v for k, v in body.items() if k not in ("model", "stream")}
            rows.append({"recordId": key, "modelInput": {"anthropic_version": "bedrock-2023-05-31", **body}})
        await self.put(f"{folder}/input.jsonl", _jsonl(rows))
        name = re.sub(r"[^a-zA-Z0-9-]", "-", f"mo-{job['id']}")[:63]
        d = await self._json("POST", f"{self._control()}/model-invocation-job", auth=self.signer("bedrock"), what="creating a batch job", body={
            "jobName": name, "roleArn": self.cfg.batch.role_arn, "modelId": job["model"], "clientRequestToken": job["id"],
            "inputDataConfig": {"s3InputDataConfig": {"s3Uri": f"s3://{self.bucket()}/{folder}/input.jsonl", "s3InputFormat": "JSONL"}},
            "outputDataConfig": {"s3OutputDataConfig": {"s3Uri": f"s3://{self.bucket()}/{folder}/output/"}},
            "timeoutDurationInHours": 24})
        return Submitted(str(d.get("jobArn")), {"folder": folder})

    async def poll(self, job):
        d = await self._json("GET", f"{self._control()}/model-invocation-job/{quote(job['provider_id'], safe='')}", auth=self.signer("bedrock"),
                             what="checking a batch job")
        status = str(d.get("status") or "")
        ended = status in ("Completed", "PartiallyCompleted", "Failed", "Stopped", "Expired")
        total, ok, bad = (int(d.get(k) or 0) for k in ("totalRecordCount", "successRecordCount", "errorRecordCount"))
        return Polled(ended, status, _counts(max(total - ok - bad, 0) if not ended else 0, ok, bad), expired=status == "Expired",
                      failed=str(d.get("message") or "the provider failed the job") if status == "Failed" and not ok else None)

    async def results(self, job, want=None):
        folder = (job.get("provider") or {}).get("folder") or self.folder(job)
        for name in await self.list(f"{folder}/output/"):
            if not name.endswith(".jsonl.out"):
                continue  # manifest.json.out holds totals only
            async for d in self.read(name):
                key = str(d.get("recordId") or "")
                if d.get("modelOutput") is not None and not d.get("error"):
                    yield Outcome(key, "succeeded", d["modelOutput"])
                else:
                    e = d.get("error") or {}
                    code = int(e.get("errorCode") or 500) if str(e.get("errorCode") or "").isdigit() else 500
                    yield Outcome(key, "errored", code=code, kind=_status_kind(code), message=str(e.get("errorMessage") or "the request failed"))

    async def cancel(self, job):
        await self._request("POST", f"{self._control()}/model-invocation-job/{quote(job['provider_id'], safe='')}/stop",
                            auth=self.signer("bedrock"), what="stopping a batch job")


# ====================================================================== Vertex AI


class VertexBatches(BucketBatches):
    """Vertex AI batch prediction: Claude (Messages bodies) or Gemini (GenerateContent bodies) in a JSONL file in Cloud Storage.
    Gemini's output echoes each request rather than carrying an id, so the key rides along as a request label."""

    name = "vertex"
    bucket_class = GCSBucket
    max_requests = 200_000
    max_bytes = 1024 * MB

    @property
    def claude(self) -> bool:
        return self.cfg.format == "anthropic"

    def region(self) -> str:
        if self.cfg.batch.region:
            return self.cfg.batch.region
        own = self.cfg.region or "global"
        return "us-central1" if self.claude and own == "global" else own  # partner models have no global batch prediction

    def _host(self) -> str:
        r = self.region()
        return "https://aiplatform.googleapis.com" if r == "global" else f"https://{r}-aiplatform.googleapis.com"

    def _model(self, model: str) -> str:
        return f"publishers/{'anthropic' if self.claude else 'google'}/models/{gemini.model_id(model) if not self.claude else model}"

    def _line(self, key: str, body: dict) -> dict:
        if self.claude:
            body = {k: v for k, v in body.items() if k not in ("model", "stream")}
            return {"custom_id": key, "request": {"anthropic_version": "vertex-2023-10-16", **body}}
        req = gemini.chat_to_gemini_request(body)
        req["labels"] = {**(req.get("labels") or {}), "meritorder_key": key}
        return {"request": req}

    async def submit(self, job, requests):
        folder = self.folder(job)
        await self.put(f"{folder}/input.jsonl", _jsonl([self._line(k, b) for k, b in requests]))
        parent = f"projects/{self.cfg.project}/locations/{self.region()}"
        d = await self._json("POST", f"{self._host()}/v1/{parent}/batchPredictionJobs", what="creating a batch prediction job", body={
            "displayName": job["id"], "model": self._model(job["model"]),
            "inputConfig": {"instancesFormat": "jsonl", "gcsSource": {"uris": [f"gs://{self.bucket()}/{folder}/input.jsonl"]}},
            "outputConfig": {"predictionsFormat": "jsonl", "gcsDestination": {"outputUriPrefix": f"gs://{self.bucket()}/{folder}/output"}}})
        return Submitted(str(d.get("name")), {"folder": folder})

    async def poll(self, job):
        d = await self._json("GET", f"{self._host()}/v1/{job['provider_id']}", what="checking a batch prediction job")
        state = str(d.get("state") or "").removeprefix("JOB_STATE_")
        s = d.get("completionStats") or {}
        ok, bad, left = (int(s.get(k) or 0) for k in ("successfulCount", "failedCount", "incompleteCount"))
        ended = state in ("SUCCEEDED", "FAILED", "CANCELLED", "EXPIRED", "PARTIALLY_SUCCEEDED")
        out = (d.get("outputInfo") or {}).get("gcsOutputDirectory")
        return Polled(ended, state.lower(), _counts(left if not ended else 0, ok, bad), {"output": out} if out else {},
                      expired=state == "EXPIRED",
                      failed=str((d.get("error") or {}).get("message") or "the provider failed the job") if state == "FAILED" and not ok else None)

    async def results(self, job, want=None):
        p = job.get("provider") or {}
        out = p.get("output")
        prefix = split_bucket(out)[1] + "/" if out else f"{p.get('folder') or self.folder(job)}/output"
        for name in await self.list(prefix):
            if not name.endswith(".jsonl") or "/input.jsonl" in name:
                continue
            async for d in self.read(name):
                req = d.get("request") or {}
                key = str(d.get("custom_id") or (req.get("labels") or {}).get("meritorder_key") or "")
                resp, status = d.get("response"), d.get("status")
                if status or not resp:
                    yield Outcome(key, "errored", code=500, kind="api_error", message=str(status or "the request failed"))
                elif self.claude:
                    yield Outcome(key, "succeeded", resp)
                else:
                    yield _chat_outcome(key, gemini.gemini_to_chat_completion(resp, job.get("model") or ""))

    async def cancel(self, job):
        await self._request("POST", f"{self._host()}/v1/{job['provider_id']}:cancel", body={}, what="cancelling a batch prediction job")

    async def purge(self, job):
        await super().purge(job)
        try:
            await self._request("DELETE", f"{self._host()}/v1/{job['provider_id']}", what="deleting a batch prediction job")
        except BackendError as e:
            if e.status != 404:
                raise


# ====================================================================== mock


class MockBatches(BatchAPI):
    """The mock backend's batch API: it answers every request at once, as the live mock would, and keeps the replies in
    the store (Postgres in a real deployment), where any gateway or worker finds them, until the batch's results are deleted.
    Like a provider would, and only for the mock: it's for trying batches out with no keys."""

    name = "mock"
    one_model = False
    embeddings = True

    def _key(self, job: dict) -> str:
        return f"mockbatch:{job['provider_id']}"

    async def submit(self, job, requests):
        pid = f"mockbatch_{uuid.uuid4().hex[:20]}"
        out = []
        for key, body in requests:
            if job.get("endpoint") == "embeddings":
                resp = await self.backend.embed(body)
            else:
                resp = await self.backend.send({k: v for k, v in body.items() if k != "stream"}, False, {})
            out.append({"key": key, "body": resp.body})
        await self.store.set_setting(f"mockbatch:{pid}", {"results": out, "at": time.time()})
        return Submitted(pid, {})

    async def poll(self, job):
        held = await self.store.get_setting(self._key(job))
        if not held:
            return Polled(True, "ended", _counts(), failed="the mock batch is gone")
        return Polled(True, "ended", _counts(succeeded=len(held["results"])))

    async def results(self, job, want=None):
        held = await self.store.get_setting(self._key(job)) or {}
        for r in held.get("results") or []:
            yield Outcome(r["key"], "succeeded", r["body"])

    async def cancel(self, job):
        return None

    async def purge(self, job):
        await self.store.set_setting(self._key(job), {})


APIS: dict[str, type[BatchAPI]] = {"anthropic": AnthropicBatches, "openai": OpenAIBatches, "azure": AzureBatches, "moonshot": MoonshotBatches,
                                   "mistral": MistralBatches, "openrouter": OpenRouterBatches, "xai": XaiBatches, "gemini": GeminiBatches,
                                   "bedrock": BedrockBatches, "vertex": VertexBatches, "mock": MockBatches}


def batch_api_for(backend: Backend, settings: BatchesConfig, store=None, content=None) -> BatchAPI | None:
    api = getattr(backend, "batch_api", None)
    if api == "gpu":
        from .gpu import GpuBatches

        return GpuBatches(backend, settings, store, content) if settings.gpu.enabled else None
    return APIS[api](backend, settings, store, content) if api in APIS else None
