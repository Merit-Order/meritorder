"""The worker's side of batches: ask the providers how each job is doing until it ends, then settle it; and run the
batch work that went to self-hosted GPUs (gpu.py).

Jobs are claimed from Postgres with a lease (``FOR UPDATE SKIP LOCKED``), as eval runs are, so several workers share them
and a worker that dies mid-job hands it on when its lease runs out. A job is asked again after ``batches.poll_seconds``,
the wait doubling while nothing changes (and after errors) up to ``max_poll_seconds``. A GPU job is worked on a slice at a
time instead, and asked again within seconds while it has requests waiting and the deployment has room.

A job a GPU handed on (gpu.py) got a fresh 24-hour window at its provider when it was submitted, which ends after the
batch's own. So it's asked again no later than the batch's deadline (``expires_at``), cancelled if it's still running
then, and what the provider hadn't run settles as expired: the client's 24 hours hold.

Settling reads the job's results once, as they stream from the provider, and keeps none of their content: each request
becomes one ledger row (``service_tier = 'batch'``, priced at the batch price the job was submitted at, with the discount
as its own savings cause), its line is marked settled in the same transaction, and its estimate moves off the batch's
reservation. Reading again after a crash skips what's settled. When every job has settled, the batch has ended.

Whoever settles a request counts it in its Prometheus metrics (``meritorder_batch_*``, metrics.py): the worker, which
serves them on ``MERITORDER_WORKER_METRICS_PORT``, or without Postgres the gateway. Only the rows a settling call newly
wrote are counted, so a request is counted once whatever retries or workers do, and never in the live counters.

After ``batches.retention_days``, what the providers (and the buckets, and the gateway's own storage for GPU work) still
hold for an ended batch is deleted, and so is any uploaded input file no batch used.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import os
import socket
import uuid
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta

from .. import alerts
from ..backends import BackendError, build_backends
from ..config import BATCH_DISCOUNT, Config, Price
from ..formats.translate import usage_from_anthropic, usage_from_openai
from ..metrics import Metrics, cost_kind
from ..pricing import Costs, Usage, attribute
from .content import Content, ContentCache
from .gpu import GpuBatches, GpuRunner
from .providers import BatchAPI, Outcome, batch_api_for
from .service import OUTCOME_STATUS, month_key, spend_keys

log = logging.getLogger("meritorder.batches")

SETTLE_CHUNK = 500
LEASE_SECONDS = 900
# A job still "submitting" this long after its submission started: the gateway stopped part way (it gives a provider
# service.SUBMIT_LIMIT to take a job, and restarts the job's clock as it goes).
STUCK = timedelta(minutes=30)


def handed_on(job: dict) -> bool:
    """A provider job a GPU handed its requests to: started after the batch, so its provider's window ends after the batch's."""
    return bool((job.get("provider") or {}).get("handed_from"))


def past_deadline(b: dict, job: dict, now: datetime | None = None) -> bool:
    """A handed-on job still out when its batch's deadline came: the gateway cancels it there."""
    return handed_on(job) and bool(b.get("expires_at")) and (now or datetime.now(UTC)) >= b["expires_at"]


def cut_at_deadline(b: dict, job: dict) -> bool:
    """The gateway cancelled this job at its batch's deadline (not the client): what it hadn't run expired."""
    return handed_on(job) and bool(job.get("cancel_sent")) and not b.get("cancel_requested_at")


def usage_of(fmt: str, kind: str, body: dict) -> Usage:
    u = body.get("usage") or {}
    if kind == "embeddings":
        return Usage(input_tokens=int(u.get("prompt_tokens") or u.get("total_tokens") or 0))
    return usage_from_anthropic(u) if fmt == "anthropic" else usage_from_openai(u)


def line_costs(cfg: Config, b: dict, line: dict, prices: dict, usage: Usage, self_cost: float | None = None) -> Costs:
    """A batch request's costs: the batch price it was submitted at, against the team's baseline at live API prices.
    ``self_cost``: it ran on a self-hosted GPU and cost this much GPU time; then the batch price is what a provider's batch
    API would have charged, so the batch discount stays its own cause and the rest is self-hosting's."""
    from ..proxy import api_price

    p = prices.get(f"{line['model']}|{line['upstream_model']}") or {}
    standard = Price.model_validate(p.get("standard") or {})
    batch = Price.model_validate(p["batch"]) if p.get("batch") else standard.for_batch(BATCH_DISCOUNT)
    base = cfg.model(cfg.team(b["team"]).baseline_model or cfg.baseline_model)
    # Embeddings and eval traffic would have run on the same model without Merit Order: no routing savings (or losses).
    baseline = api_price(cfg, base) if base and base.routes and line["kind"] != "embeddings" and "eval" not in (b.get("tags") or {}) else None
    return attribute(usage, standard, baseline, False, self_hosted_cost=self_cost, batch_price=batch)


def ledger_row(cfg: Config, b: dict, job: dict, line: dict, o: Outcome, ts: datetime,
               self_cost: Callable[[Usage], float] | None = None) -> dict:
    fmt = (job.get("provider") or {}).get("format") or "openai"
    usage = usage_of(fmt, line["kind"], o.body or {}) if o.status == "succeeded" else Usage()
    gpu = self_cost(usage) if self_cost is not None and usage.total_tokens else None
    costs = line_costs(cfg, b, line, job.get("prices") or {}, usage, gpu) if usage.total_tokens else Costs(0, 0, 0, 0, 0, 0, 0)
    bc = cfg.backend(job["backend"])
    error = None if o.status == "succeeded" else (o.message or {"canceled": "cancelled before it ran", "expired": "the provider didn't run it in time"}
                                                   .get(o.status, "failed"))
    return {"id": uuid.uuid4(), "ts": ts, "user_id": b["user_id"], "team": b["team"], "key_id": b.get("key_id"), "session_id": None,
            "tags": b.get("tags") or {}, "client_format": "embeddings" if line["kind"] == "embeddings" else b["client_format"],
            "requested_model": line.get("requested_model"), "floor": line.get("floor"), "route_reason": line.get("route_reason"),
            "served_model": line.get("model"), "upstream_model": line.get("upstream_model"), "backend": job["backend"],
            "backend_kind": bc.kind if bc else "api", "status": OUTCOME_STATUS.get(o.status, o.code or 500), "error": (error or "")[:500] or None,
            "stream": False, "upgraded": bool(line.get("upgraded")), "proxy_cache": False,
            "input_tokens": usage.input_tokens, "cache_read_tokens": usage.cache_read_tokens, "cache_write_tokens": usage.cache_write_tokens,
            "output_tokens": usage.output_tokens, "latency_ms": None, "ttft_ms": None,
            "actual_cost": costs.actual, "served_api_cost": costs.served_api, "baseline_cost": costs.baseline, "list_cost": costs.list_price,
            "savings_routing": costs.savings_routing, "savings_caching": costs.savings_caching, "savings_self_hosting": costs.savings_self_hosting,
            "savings_compression": costs.savings_compression, "savings_response_cache": costs.savings_response_cache,
            "savings_batch": costs.savings_batch, "service_tier": "batch", "batch_id": b["id"]}


class BatchPoller:
    """Polls and settles provider jobs, and runs GPU jobs. ``ctx`` has cfg, backends, store, state and metrics: a
    Gateway, or the worker's own (WorkerContext)."""

    def __init__(self, ctx, owner: str | None = None):
        self.ctx = ctx
        self.owner = owner or f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:6]}"
        self.purged_at = 0.0
        self.contents = ContentCache()
        self.content: Content | None = None

    @property
    def settings(self):
        return self.ctx.cfg.batches

    def api(self, backend: str) -> BatchAPI | None:
        b = self.ctx.backends.get(backend)
        return batch_api_for(b, self.settings, self.ctx.store, self.content) if b is not None else None

    def _later(self, n: int, retry_after: float | None = None, b: dict | None = None, job: dict | None = None) -> datetime:
        wait = min(self.settings.max_poll_seconds, self.settings.poll_seconds * 2 ** min(n, 20))
        at = datetime.now(UTC) + timedelta(seconds=max(wait, retry_after or 0))
        if b is not None and job is not None and handed_on(job) and b.get("expires_at") and not job.get("cancel_sent"):
            at = min(at, max(b["expires_at"], datetime.now(UTC)))  # asked again at the batch's deadline at the latest
        return at

    def count(self, rows: list[dict], api: str = "") -> None:
        """Settled rows into this process's batch metrics (each row is a line settled by this call, so once)."""
        m: Metrics | None = getattr(self.ctx, "metrics", None)
        if m is not None:
            for r in rows:
                m.record_batch(r, api, cost_kind(self.ctx.cfg, r.get("backend")))

    async def run_once(self, limit: int = 20) -> int:
        """One pass: every due job this worker can claim. Returns how many it handled."""
        store = self.ctx.store
        self.content = await self.contents.get(self.ctx.cfg, store)
        jobs = await store.claim_batch_jobs(self.owner, limit, LEASE_SECONDS)
        for job in jobs:
            try:
                await self.process(job)
            except Exception:  # noqa: BLE001 (one bad job must not stop the others; it's tried again later)
                log.exception("batch job %s failed to process", job["id"])
                await store.update_batch_job(job["id"], self.owner, release=True, failures=job["failures"] + 1,
                                             next_poll_at=self._later(job["failures"] + 1))
        if datetime.now(UTC).timestamp() - self.purged_at > 3600:
            self.purged_at = datetime.now(UTC).timestamp()
            await self.purge()
        return len(jobs)

    async def process(self, job: dict) -> None:
        store = self.ctx.store
        if self.content is None:
            self.content = await self.contents.get(self.ctx.cfg, store)
        b = await store.get_batch(job["batch_id"])
        if b is None:
            await store.update_batch_job(job["id"], self.owner, release=True, status="failed", error="its batch is gone")
            return
        if job["status"] == "submitting":
            if datetime.now(UTC) - job["created_at"] < STUCK:
                await store.update_batch_job(job["id"], self.owner, release=True, next_poll_at=self._later(0))
                return
            # The gateway stopped before it heard back from the provider: whatever the provider made isn't tracked. The job
            # is stored as failed once its requests are settled (settle).
            job = {**job, "status": "failed", "ended_at": datetime.now(UTC), "error": "the gateway stopped before the provider confirmed the job"}
            await self.settle(b, job, None)
            await self.finish(b["id"])
            return
        if job["api"] == "gpu" and job["status"] == "running":
            # Its own runner: the deployment may be gone from this process's backends (stopped, removed), and then its
            # requests go to their fallbacks rather than wait for it.
            backend = self.ctx.backends.get(job["backend"])
            api = GpuBatches(backend, self.settings, store, self.content) if backend is not None else GpuBatches.detached(self.settings, self.content)
            await GpuRunner(self, api).run(b, job)
            return
        api = self.api(job["backend"])
        if api is None:
            # The backend was switched off or removed since: keep the job until it's back, and say so.
            if await alerts._once(self.ctx.state, f"alert:batch-backend:{job['id']}", 6 * 3600):
                await alerts.notify(f"Batch job {job['id']} (batch {b['id']}) is on {job['backend']}, which isn't enabled now: its results "
                                    "can't be read until it is.", state=self.ctx.state, title="Batch job can't be checked", page="reports")
            await store.update_batch_job(job["id"], self.owner, release=True, failures=job["failures"] + 1,
                                         next_poll_at=self._later(job["failures"] + 1))
            return
        if job["status"] == "running":
            late = past_deadline(b, job)
            if (b.get("cancel_requested_at") or late) and not job["cancel_sent"]:
                try:
                    await api.cancel(job)
                    job["cancel_sent"] = True
                    if late:
                        log.info("batch job %s on %s, handed on from a GPU, is still running at batch %s's deadline: cancelled, "
                                 "and what it hasn't run expires", job["id"], job["backend"], b["id"])
                except BackendError as e:
                    log.warning("couldn't cancel batch job %s: %s", job["id"], e.message)
            try:
                polled = await api.poll(job)
            except BackendError as e:
                gone = e.status == 404
                m: Metrics | None = getattr(self.ctx, "metrics", None)
                if m is not None:
                    m.inc("meritorder_batch_poll_errors_total", {"backend": job["backend"], "status": str(e.status)})
                if gone or job["failures"] + 1 >= 5:
                    log.warning("batch job %s on %s: %s", job["id"], job["backend"], e.message)
                if not gone:
                    if job["failures"] + 1 >= 5 and await alerts._once(self.ctx.state, f"alert:batch-poll:{job['id']}", 6 * 3600):
                        await alerts.notify(f"Batch job {job['id']} on {job['backend']} can't be checked ({e.message}); the worker keeps trying.",
                                            state=self.ctx.state, title="Batch job can't be checked", page="reports")
                    await store.update_batch_job(job["id"], self.owner, release=True, failures=job["failures"] + 1,
                                                 cancel_sent=job["cancel_sent"],
                                                 next_poll_at=self._later(job["failures"] + 1, e.retry_after, b, job))
                    return
                polled = None  # failed: stored as such once its requests are settled (settle)
                job = {**job, "status": "failed", "ended_at": datetime.now(UTC), "error": f"the provider no longer has this job: {e.message}"[:500]}
            if polled is not None:
                changed = polled.counts != (job.get("counts") or {}) or polled.provider_status != job.get("provider_status")
                fields = {"counts": polled.counts, "provider_status": polled.provider_status, "polls": job["polls"] + 1, "failures": 0,
                          "provider": {**(job.get("provider") or {}), **polled.provider}, "cancel_sent": job["cancel_sent"]}
                failed = None
                if polled.failed and not polled.counts.get("succeeded"):
                    failed = {"status": "failed", "ended_at": datetime.now(UTC), "error": polled.failed[:500]}  # stored once settled
                elif polled.ended:
                    fields.update(status="ended", ended_at=datetime.now(UTC))
                else:
                    unchanged = 0 if changed else job["unchanged"] + 1
                    fields.update(unchanged=unchanged, next_poll_at=self._later(unchanged, b=b, job=job))
                job = await store.update_batch_job(job["id"], self.owner, release=fields.get("status") is None and failed is None, **fields)
                await self.progress(b["id"])
                if job is None or (job["status"] == "running" and failed is None):
                    return
                job = {**job, **(failed or {})}
        if job["status"] in ("ended", "failed"):
            await self.settle(b, job, api if job["status"] == "ended" else None)
            await self.finish(b["id"])

    async def progress(self, batch_id: str) -> None:
        """The batch's counts while it runs: what its jobs' providers report (for GPU jobs, how their requests ended), plus
        the requests refused at submission."""
        store = self.ctx.store
        jobs = await store.batch_jobs(batch_id)
        b = await store.get_batch(batch_id)
        lines = await store.batch_lines(batch_id)
        keys = ("processing", "succeeded", "errored", "canceled", "expired")
        counts = {k: sum(int((j.get("counts") or {}).get(k) or 0) for j in jobs if j["status"] != "failed" and j["api"] != "gpu") for k in keys}
        gpu = {j["id"] for j in jobs if j["api"] == "gpu"}
        for ln in lines:
            if ln["status"] == "refused":
                counts["errored"] += 1
            elif ln["job_id"] in gpu and ln["settled"]:
                counts[ln["status"] if ln["status"] in keys else "errored"] += 1
        counts["processing"] = max(0, (b["request_count"] if b else 0) - sum(v for k, v in counts.items() if k != "processing"))
        await store.update_batch(batch_id, counts=counts)

    async def settle_lines(self, b: dict, job: dict, done: list[tuple[dict, Outcome]],
                           self_cost: Callable[[Usage], float] | None = None) -> None:
        """Write these lines' ledger rows (once each), move their estimates off the reservation and count what they cost."""
        if not done:
            return
        store, state = self.ctx.store, self.ctx.state
        # Stamped when written, not when the job ended (settling can come much later): the month's spend counters count the
        # rows now, and the rollups and the ClickHouse copy look again only at the last few hours.
        ts = datetime.now(UTC)
        for i in range(0, len(done), SETTLE_CHUNK):
            chunk = [{"idx": line["idx"], "status": o.status, "error": (o.message or None) if o.status != "succeeded" else None,
                      "row": ledger_row(self.ctx.cfg, b, job, line, o, ts, self_cost)} for line, o in done[i:i + SETTLE_CHUNK]]
            released, actual, written = await store.settle_batch_lines(b["id"], chunk)
            self.count(written, job["api"])
            now_month = month_key(ts)
            reserved_month = b.get("reserve_month") or now_month
            if reserved_month != now_month:  # its reservation counts in each month it's outstanding (batch_reserved)
                await state.adjust_spend(spend_keys(b["user_id"], b["team"], reserved_month), -released)
            await state.adjust_spend(spend_keys(b["user_id"], b["team"], now_month), actual - released)

    async def settle(self, b: dict, job: dict, api: BatchAPI | None) -> None:
        """Read the job's results once and write them into the ledger (see the module's docstring)."""
        store = self.ctx.store
        lines = {f"r{ln['idx']}": ln for ln in await store.batch_lines(b["id"], job["id"]) if not ln["settled"]}
        chunk: list[tuple[dict, Outcome]] = []
        seen: set[str] = set()
        cut = cut_at_deadline(b, job)
        if api is not None and lines:
            async for o in api.results(job):
                line = lines.get(o.key)
                if line is None or o.key in seen:
                    continue
                seen.add(o.key)
                if cut and o.status == "canceled":
                    o = replace(o, status="expired", message="not run by the batch's 24-hour deadline")
                chunk.append((line, o))
                if len(chunk) >= SETTLE_CHUNK:
                    await self.settle_lines(b, job, chunk)
                    chunk = []
        missing = ("expired" if cut or (job.get("provider_status") or "").lower() == "expired"
                   else "canceled" if b.get("cancel_requested_at") else "errored")
        for key, line in lines.items():
            if key not in seen:
                why = job.get("error") or "the provider returned no result for this request"
                chunk.append((line, Outcome(key, "errored" if job["status"] == "failed" else missing, code=502, kind="api_error", message=why)))
                if len(chunk) >= SETTLE_CHUNK:
                    await self.settle_lines(b, job, chunk)
                    chunk = []
        await self.settle_lines(b, job, chunk)
        if job["status"] == "ended":
            await store.update_batch_job(job["id"], self.owner, release=True, status="settled")
        else:  # failed: stored only now, as "settled" is, so an interrupted settle leaves the job to be tried again
            await store.update_batch_job(job["id"], self.owner, release=True, status="failed", ended_at=job.get("ended_at"), error=job.get("error"))

    async def finish(self, batch_id: str) -> dict | None:
        """End the batch once every job has settled (or failed): final counts, outcome, and the rest of its reservation back."""
        store = self.ctx.store
        b = await store.get_batch(batch_id)
        jobs = await store.batch_jobs(batch_id)
        if b is None or b["status"] == "ended":
            return b
        if any(j["status"] in ("submitting", "running") for j in jobs):
            return b
        if any(j["status"] == "ended" for j in jobs):  # the providers are done; results are being settled
            return await store.update_batch(batch_id, status="finalizing", finalizing_at=b.get("finalizing_at") or datetime.now(UTC))
        lines = await store.batch_lines(batch_id)
        counts = {k: 0 for k in ("processing", "succeeded", "errored", "canceled", "expired")}
        for ln in lines:
            counts[ln["status"] if ln["status"] in counts else "errored"] += 1
        outcome = ("cancelled" if b.get("cancel_requested_at") else "failed" if not counts["succeeded"] and all(j["status"] == "failed" for j in jobs)
                   else "expired" if counts["expired"] else "completed")
        if b.get("reserved_cost"):
            for month in dict.fromkeys((b.get("reserve_month") or month_key(), month_key())):  # outstanding in both (batch_reserved)
                await self.ctx.state.adjust_spend(spend_keys(b["user_id"], b["team"], month), -b["reserved_cost"])
        log.info("batch %s %s: %s", batch_id, outcome, ", ".join(f"{v} {k}" for k, v in counts.items() if v))
        return await store.update_batch(batch_id, status="ended", outcome=outcome, counts=counts, reserved_cost=0.0, ended_at=datetime.now(UTC),
                                        finalizing_at=b.get("finalizing_at") or datetime.now(UTC))

    async def purge(self) -> int:
        """Delete what's past ``batches.retention_days``: the providers' copies of ended batches, what the gateway holds for
        GPU work, and unused uploads."""
        store = self.ctx.store
        self.content = await self.contents.get(self.ctx.cfg, store)
        before = datetime.now(UTC) - timedelta(days=self.settings.retention_days)
        n = 0
        for f in await store.stale_batch_files(before):
            try:
                await self.content.drop_file(f)
            except Exception as e:  # noqa: BLE001 (a bucket that's unreachable now is tried again next time)
                log.warning("couldn't delete uploaded batch file %s (tried again later): %s", f["id"], e)
        for b in await store.batches_to_purge(before):
            ok = True
            for j in await store.batch_jobs(b["id"]):
                if j.get("purged_at") or not j.get("provider_id"):
                    continue
                api = GpuBatches.detached(self.settings, self.content) if j["api"] == "gpu" else self.api(j["backend"])
                try:
                    if api is None:
                        raise BackendError(503, f"{j['backend']} isn't enabled")
                    await api.purge(j)
                    await store.update_batch_job(j["id"], purged_at=datetime.now(UTC))
                except BackendError as e:
                    ok = False
                    log.warning("couldn't delete batch job %s's results from %s (tried again later): %s", j["id"], j["backend"], e.message)
            if ok:
                await store.update_batch(b["id"], purged_at=datetime.now(UTC))
                n += 1
        return n


class WorkerContext:
    """What the poller needs, in a process without a gateway: the configuration and its backends, rebuilt when it changes,
    the GPU deployments' state as the controller publishes it (for batch work on them), and its own metrics."""

    def __init__(self, store, state):
        self.store, self.state = store, state
        self.cfg: Config | None = None
        self.backends: dict = {}
        self.metrics = Metrics()

    async def use(self, cfg: Config) -> None:
        if cfg is not self.cfg:
            old = self.backends
            self.cfg, self.backends = cfg, build_backends(cfg)
            for b in old.values():
                await b.close()
        from ..proxy import refresh_deployments

        await refresh_deployments(self.backends, self.state, self.store)

    async def close(self) -> None:
        for b in self.backends.values():
            await b.close()


async def serve_metrics(metrics: Metrics, port: int, host: str = "0.0.0.0") -> asyncio.base_events.Server:
    """``GET /metrics`` for the worker's batch counters, in Prometheus' text format, on its own port: a few lines of HTTP
    rather than a web framework in the worker. ``MERITORDER_METRICS_TOKEN`` guards it as it guards the gateway's."""

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=10)
            request, *fields = head.decode("latin-1").split("\r\n")
            method, path = (request.split(" ") + ["", ""])[:2]
            headers = {k.strip().lower(): v.strip() for k, _, v in (f.partition(":") for f in fields if f)}
            token = os.environ.get("MERITORDER_METRICS_TOKEN")
            if method != "GET" or path.split("?")[0] not in ("/metrics", "/healthz"):
                status, body, ctype = "404 Not Found", b"not found\n", "text/plain"
            elif path.startswith("/healthz"):
                status, body, ctype = "200 OK", b'{"status": "ok"}', "application/json"
            elif token and not hmac.compare_digest(headers.get("authorization", "").encode(), f"Bearer {token}".encode()):
                status, body, ctype = "401 Unauthorized", b"metrics token required\n", "text/plain"
            else:
                status, body, ctype = "200 OK", metrics.render().encode(), "text/plain; version=0.0.4"
            writer.write(f"HTTP/1.1 {status}\r\ncontent-type: {ctype}\r\ncontent-length: {len(body)}\r\nconnection: close\r\n\r\n".encode() + body)
            await writer.drain()
        except (asyncio.TimeoutError, asyncio.IncompleteReadError, ConnectionError, asyncio.LimitOverrunError):
            pass
        finally:
            writer.close()

    return await asyncio.start_server(handle, host, port)


async def batch_loop(store, state, follower, interval: float = 15.0, ctx: WorkerContext | None = None) -> None:
    """In the worker: poll due batch jobs every ``interval`` seconds, following the stored configuration."""
    ctx = ctx or WorkerContext(store, state)
    poller = BatchPoller(ctx)
    try:
        while True:
            try:
                await ctx.use(await follower.current())
                await poller.run_once()
            except Exception:  # noqa: BLE001 (keep polling after a database or provider hiccup)
                log.exception("batch poll failed")
            await asyncio.sleep(interval)
    finally:
        await ctx.close()
        await poller.contents.close()
