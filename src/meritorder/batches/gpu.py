"""Batch work on self-hosted GPUs (decision 49, docs/02-request-flow.md#batch-work-on-your-gpus).

A GPU deployment that's running is already paid for by the hour, so a request that can wait costs it almost nothing while
the deployment has room: the router prices a batch request there at the deployment's shadow price (router.py), which is
the route's price times 2% below the shadow price's knee, against the providers' batch prices (about half). When the GPU
is cheapest it gets the request, and the request's next route that has a batch API is kept as its fallback.

The requests can't wait at a provider, so the gateway holds them, sealed (content.py), until they run; and their results
until the retention period ends. The worker runs them (``GpuRunner``), a slice at a time while it holds the job's lease:

* **Only in the headroom.** Batch requests fill the deployment up to the shadow price's knee (``routing.shadow_knee``, 60%
  of ``max_concurrency`` x replicas) and never past it, counting the gateways' live requests and other workers' batch
  requests (both published to Redis every couple of seconds), so batch work never makes live traffic pay more than the
  floor of the shadow price. A deployment without ``max_concurrency`` takes no batch work: its slots can't be counted.
* **Live traffic first.** Each request goes to vLLM with ``priority`` ``batches.gpu.priority`` (20; lower runs first), after
  free upgrades (10) and live requests (0), so vLLM's scheduler preempts batch work when live requests need the room.
* **Before the deadline.** What hasn't run ``batches.gpu.handoff_hours`` (12) after the batch was made goes to its fallback
  provider's batch API. So does everything still waiting when the deployment stops or starts draining, and a request the
  GPU fails twice. The provider's job gets a fresh 24-hour window, which would end after the client's, so the poller
  cancels a handed-on job still running at the batch's own deadline (``expires_at``) and what it hadn't run expires, as at a
  provider whose window closed. A fallback is therefore always a batch API the gateway can cancel (not OpenRouter's). A
  request with no fallback waits for the GPU until the batch's deadline, then expires; nothing is handed on after it.
* **No GPU starts for it.** Batch demand isn't counted by the break-even controller (decision 49): it never starts or
  keeps a deployment, it only uses one that live demand pays for.

Each finished request is settled at once: its ledger row (priced as live GPU work is, by the GPU time it took) and its
estimate off the batch's reservation, after its result is held.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from datetime import UTC, datetime, timedelta

from ..backends import BackendError
from ..pricing import chargeback
from .providers import BatchAPI, Outcome, Polled, Submitted, _jsonl

log = logging.getLogger("meritorder.batches")

HELD_PIECE = 500  # requests per held piece (each sealed on its own, so a slice reads only what it runs)
SLICE_SECONDS = 60.0  # how long one slice keeps starting requests (the lease is 15 minutes)
TRIES = 2  # a request the GPU fails this often goes to its fallback (or fails, without one)
BATCH = "~batch"  # published in-flight counts: "<backend>~batch" is batch work, which live routing leaves out


def held_prefix(job: dict) -> str:
    return f"{job['batch_id']}/{job['id']}/"


def _rows(data: bytes | None) -> list[dict]:
    return [json.loads(x) for x in (data or b"").decode().splitlines() if x.strip()]


class GpuBatches(BatchAPI):
    """A self-hosted GPU deployment as a batch API: the "provider" is the gateway itself, holding the requests (sealed)
    until the worker runs them, and their results until the retention period ends."""

    name = "gpu"
    one_model = False  # a deployment serves its one model

    @classmethod
    def detached(cls, settings, content) -> "GpuBatches":
        """For a job whose deployment isn't among this process's backends any more (stopped, switched off, removed): what
        it holds can still be handed on, read and deleted."""
        api = cls.__new__(cls)
        api.backend, api.cfg, api.settings, api.store, api.content = None, None, settings, None, content
        return api

    async def submit(self, job, requests):
        if self.content is None or self.content.ring.key is None:
            raise BackendError(503, f"{self.backend.name}: batch work on a GPU is held sealed until it runs: set MERITORDER_BATCH_KEY")
        fallback = job.get("_fallback") or {}
        rows = [{"key": key, "body": body, "fallback": fallback.get(key)} for key, body in requests]
        pieces = 0
        for i in range(0, len(rows), HELD_PIECE):
            await self.content.put(f"{held_prefix(job)}requests/{i // HELD_PIECE:06d}", _jsonl(rows[i:i + HELD_PIECE]))
            pieces += 1
        return Submitted(f"gpu_{job['id']}", {"pieces": pieces})

    async def poll(self, job):  # the runner does the work and ends the job; this only reports it
        return Polled(job.get("status") in ("ended", "settled"), job.get("provider_status") or "running", job.get("counts") or {})

    async def held(self, job: dict, keys: set[str] | None = None) -> dict[str, dict]:
        """The held requests of a job (only those with these keys), by key."""
        out: dict[str, dict] = {}
        for name in await self.content.names(f"{held_prefix(job)}requests/"):
            for row in _rows(await self.content.get(name)):
                if keys is None or row["key"] in keys:
                    out[row["key"]] = row
            if keys is not None and len(out) >= len(keys):
                break
        return out

    async def keep_results(self, job: dict, rows: list[dict]) -> None:
        if rows:
            name = f"{held_prefix(job)}results/{int(time.time() * 1000):015d}-{uuid.uuid4().hex[:8]}"
            await self.content.put(name, _jsonl(rows))

    async def results(self, job, want=None):
        if self.content is None:
            return
        for name in await self.content.names(f"{held_prefix(job)}results/"):
            for r in _rows(await self.content.get(name)):
                yield Outcome(r["key"], r["status"], r.get("body"), int(r.get("code") or 200), str(r.get("kind") or ""),
                              str(r.get("message") or ""))

    async def cancel(self, job):
        return None  # the runner stops at its next slice, when it sees the batch was cancelled

    async def purge(self, job):
        if self.content is not None:
            await self.content.delete(held_prefix(job))


class GpuRunner:
    """Runs one slice of a GPU job, for the poller (``ctx``: cfg, backends, store, state, metrics) while it holds the lease."""

    def __init__(self, poller, api: GpuBatches):
        self.poller, self.api = poller, api
        self.ctx = poller.ctx
        self.owner = f"batch-{poller.owner}"

    @property
    def cfg(self):
        return self.ctx.cfg

    def _metric(self, name: str, labels: dict, value: float = 1.0) -> None:
        m = getattr(self.ctx, "metrics", None)
        if m is not None:
            m.inc(name, labels, value)

    async def headroom(self, backend) -> tuple[int, str]:
        """How many more batch requests the deployment takes now: up to the shadow price's knee of its slots, less the live
        requests every gateway has open there and the batch requests other workers do. And how loaded it is, for the log."""
        slots = (backend.cfg.max_concurrency or 0) * max(1, backend.replicas or 1)
        if not slots:
            return 0, "no max_concurrency"
        knee = self.cfg.routing.shadow_knee
        snap = await self.ctx.state.gpu_load()
        live = sum(c.get(backend.name, 0) for c in snap["inflight"].values())
        batch = sum(c.get(backend.name + BATCH, 0) for gw, c in snap["inflight"].items() if gw != self.owner)
        sampled = float(snap["util"].get(backend.name) or 0.0)  # vLLM's own load at the controller's last sample
        ours = self.running
        # The sample counts batch work too (this worker's and others'): what's left of it is live load.
        live_load = max(live / slots, sampled - (batch + ours) / slots)
        free = int(slots * knee) - live - batch - ours
        if live_load >= knee:
            free = 0
        return max(0, free), f"{live} live and {batch + ours} batch of {slots} slots, {live_load:.0%} live load"

    async def publish(self, backend) -> None:
        try:
            await self.ctx.state.publish_inflight(self.owner, {backend.name + BATCH: self.running})
        except Exception:  # noqa: BLE001 (the count is advisory; the next one corrects it)
            log.debug("could not publish batch in-flight counts", exc_info=True)

    async def run(self, b: dict, job: dict) -> None:
        """One slice: settle what's done, hand on what can't wait, run what fits."""
        store = self.ctx.store
        lines = [ln for ln in await store.batch_lines(b["id"], job["id"]) if not ln["settled"]]
        now = datetime.now(UTC)
        backend = self.ctx.backends.get(job["backend"])
        if not lines:
            return await self.end(b, job)
        if b.get("cancel_requested_at"):
            await self.poller.settle_lines(b, job, [(ln, Outcome(f"r{ln['idx']}", "canceled")) for ln in lines])
            return await self.end(b, job)
        if b.get("expires_at") and now >= b["expires_at"]:
            # The client's deadline has passed: what hasn't run expires here, rather than start again at a provider.
            await self.poller.settle_lines(b, job, [(ln, Outcome(f"r{ln['idx']}", "expired")) for ln in lines])
            return await self.end(b, job)
        handoff_at = b["created_at"] + timedelta(hours=self.cfg.batches.gpu.handoff_hours)
        why = ("stopped" if backend is None or not backend.available or backend.draining
               else "deadline" if now >= handoff_at else None)
        if why:
            lines = await self.handoff(b, job, lines, why)
            if not lines:
                return await self.end(b, job)
            if why == "stopped":  # no fallback: wait for the deployment to come back, until the batch's deadline
                return await self.later(job, self.cfg.batches.poll_seconds, "waiting for the deployment to run again")
        await self.work(b, job, backend, lines)

    async def work(self, b: dict, job: dict, backend, lines: list[dict]) -> None:
        self.running = 0
        free, load = await self.headroom(backend)
        if free <= 0:
            return await self.later(job, 15, f"no headroom ({load})")
        held = await self.api.held(job, {f"r{ln['idx']}" for ln in lines})
        queue = [ln for ln in lines if f"r{ln['idx']}" in held]
        lost = [ln for ln in lines if f"r{ln['idx']}" not in held]
        if lost:  # its content is gone (deleted from the bucket by hand, say): nothing can run it
            await self.fail(b, job, lost, lambda ln: "its request is no longer held, so it can't run", 500)
        done: list[tuple[dict, dict, Outcome]] = []  # (line, its held row, outcome)
        failed: list[dict] = []  # lines that go to their fallback now
        retry: list[dict] = []
        rate = (backend.live_cost_per_hour or backend.cfg.gpu_cost_per_hour, backend.cfg.capacity_tokens_per_second)
        priority = self.cfg.batches.gpu.priority
        tasks: set[asyncio.Task] = set()

        async def one(ln: dict) -> None:
            row = held[f"r{ln['idx']}"]
            body = {**row["body"], "priority": priority}
            body.pop("stream", None)
            try:
                resp = await backend.send(body, False, {})
                reply = resp.body or {}
                if reply.get("error"):
                    raise BackendError(502, str((reply["error"] or {}).get("message") if isinstance(reply["error"], dict) else reply["error"]))
                done.append((ln, row, Outcome(row["key"], "succeeded", reply)))
                self._metric("meritorder_batch_gpu_requests_total", {"backend": backend.name, "result": "ok"})
            except BackendError as e:
                ln["error"], ln["error_code"] = e.message[:500], e.status
                ln["tries"] = int(ln.get("tries") or 0) + 1
                gone = not e.retryable or ln["tries"] >= TRIES
                (failed if gone else retry).append(ln)
                self._metric("meritorder_batch_gpu_requests_total", {"backend": backend.name, "result": "handed_off" if gone else "retry"})
            finally:
                self.running -= 1

        deadline = time.monotonic() + SLICE_SECONDS
        next_check = 0.0
        while queue and time.monotonic() < deadline:
            if time.monotonic() >= next_check:
                free, load = await self.headroom(backend)
                await self.publish(backend)
                next_check = time.monotonic() + 2
            if free <= 0:
                if not tasks:
                    break
                await asyncio.wait(tasks, timeout=2, return_when=asyncio.FIRST_COMPLETED)
                next_check = 0.0  # one finished (or two seconds passed): count the room again
                continue
            ln = queue.pop(0)
            self.running += 1
            free -= 1
            t = asyncio.create_task(one(ln))
            tasks.add(t)
            t.add_done_callback(tasks.discard)
        if tasks:
            await asyncio.gather(*tasks)
        await self.publish(backend)

        await self.api.keep_results(job, [{"key": o.key, "status": o.status, "body": o.body} for _, _, o in done])
        await self.poller.settle_lines(b, job, [(ln, o) for ln, _, o in done],
                                       self_cost=lambda body_usage: chargeback(rate[0], rate[1], body_usage))
        if retry:
            await store_update_tries(self.ctx.store, b["id"], retry)
        if failed:
            left = await self.handoff(b, job, failed, "failed")
            await self.fail(b, job, left, lambda ln: ln.get("error") or "the GPU couldn't run it")  # no fallback: the GPU's error is its
        await self.poller.progress(b["id"])
        remaining = [ln for ln in await self.ctx.store.batch_lines(b["id"], job["id"]) if not ln["settled"]]
        if not remaining:
            return await self.end(b, job)
        log.info("batch job %s on %s: %d ran, %d to try again, %d left (%s)", job["id"], backend.name, len(done), len(retry), len(remaining), load)
        await self.later(job, 1 if done else 15, f"{len(remaining)} waiting")

    async def fail(self, b: dict, job: dict, lines: list[dict], why, code: int | None = None) -> None:
        """These requests can't run anywhere: each one's result is its error (``why(line)``), at no cost."""
        if not lines:
            return
        outs = [Outcome(f"r{ln['idx']}", "errored", code=code or ln.get("error_code") or 502, kind="api_error", message=why(ln)[:500])
                for ln in lines]
        await self.api.keep_results(job, [{"key": o.key, "status": o.status, "code": o.code, "kind": o.kind, "message": o.message} for o in outs])
        await self.poller.settle_lines(b, job, list(zip(lines, outs)))

    async def handoff(self, b: dict, job: dict, lines: list[dict], why: str) -> list[dict]:
        """Move these requests to their fallback routes' batch APIs. Returns the ones without a fallback: they stay. One whose
        fallback provider refuses the job fails, as a request every route refused does when a batch is made."""
        held = await self.api.held(job, {f"r{ln['idx']}" for ln in lines})
        groups: dict[tuple[str, str], list[tuple[dict, dict]]] = {}
        stay, refused = [], []
        for ln in lines:
            row = held.get(f"r{ln['idx']}")
            fb = (row or {}).get("fallback")
            api = self.poller.api(fb["backend"]) if fb else None
            if row is None:
                refused.append(ln)
                ln["error"] = "its request is no longer held, so it can't run"
            elif fb is None or api is None:
                stay.append(ln)
            else:
                groups.setdefault((fb["backend"], fb["upstream_model"] if api.one_model else ""), []).append((ln, fb))
        moved: dict[str, int] = {}
        for (backend, model), group in groups.items():
            api = self.poller.api(backend)
            for chunk in _split(api, group):
                jid = f"bj_{uuid.uuid4().hex[:24]}"
                new = {"id": jid, "batch_id": b["id"], "backend": backend, "api": api.name, "model": model or None, "endpoint": "chat",
                       "status": "submitting", "request_count": len(chunk), "estimated_cost": sum(fb.get("estimate") or 0 for _, fb in chunk),
                       "prices": {f"{fb['model']}|{fb['upstream_model']}": fb["prices"] for _, fb in chunk},
                       "provider": {"format": fb_format(chunk), "handed_from": job["id"]}}
                await self.ctx.store.add_batch_jobs([new])
                try:
                    sub = await api.submit(new, [(f"r{ln['idx']}", fb["body"]) for ln, fb in chunk])
                except BackendError as e:
                    log.warning("batch job %s: handing %d requests from %s to %s failed: %s", job["id"], len(chunk), job["backend"], backend, e.message)
                    await self.ctx.store.update_batch_job(jid, status="failed", error=e.message[:500], ended_at=datetime.now(UTC))
                    for ln, _ in chunk:
                        ln["error"] = f"{job['backend']}: {ln.get('error') or why}; {backend} refused the batch job: {e.message}"[:500]
                        ln["error_code"] = e.status
                    refused += [ln for ln, _ in chunk]
                    continue
                await self.ctx.store.update_batch_job(jid, status="running", provider_id=sub.provider_id, provider={**new["provider"], **sub.provider},
                                                      next_poll_at=datetime.now(UTC) + timedelta(seconds=self.cfg.batches.poll_seconds))
                await self.ctx.store.update_batch_lines(b["id"], [{"idx": ln["idx"], "job_id": jid, "backend": backend, "model": fb["model"],
                                                                   "upstream_model": fb["upstream_model"], "estimated_cost": fb.get("estimate") or 0,
                                                                   "route_reason": f"{ln.get('route_reason') or ''}; handed from {job['backend']} "
                                                                                   f"to {backend} ({why})"[:1000]} for ln, fb in chunk])
                moved[backend] = moved.get(backend, 0) + len(chunk)
                self._metric("meritorder_batch_handoffs_total", {"backend": job["backend"], "why": why}, len(chunk))
        if moved:
            providers = dict(b.get("providers") or {})
            providers[job["backend"]] = providers.get(job["backend"], 0) - sum(moved.values())
            for backend, n in moved.items():
                providers[backend] = providers.get(backend, 0) + n
            b["providers"] = {k: v for k, v in providers.items() if v > 0}
            await self.ctx.store.update_batch(b["id"], providers=b["providers"])
            log.info("batch %s: %d requests handed from %s to %s (%s)", b["id"], sum(moved.values()), job["backend"],
                     ", ".join(f"{k} ({v})" for k, v in moved.items()), why)
        await self.fail(b, job, refused, lambda ln: ln.get("error") or why)
        return stay

    async def later(self, job: dict, seconds: float, status: str) -> None:
        await self.ctx.store.update_batch_job(job["id"], self.poller.owner, release=True, provider_status=status[:200],
                                              next_poll_at=datetime.now(UTC) + timedelta(seconds=seconds))

    async def end(self, b: dict, job: dict) -> None:
        await self.ctx.store.update_batch_job(job["id"], self.poller.owner, release=True, status="settled", ended_at=datetime.now(UTC),
                                              provider_status="ended")
        await self.poller.finish(b["id"])

    running = 0


def fb_format(chunk: list[tuple[dict, dict]]) -> str:
    return chunk[0][1].get("format") or "openai"


def _split(api: BatchAPI, group: list[tuple[dict, dict]]) -> list[list[tuple[dict, dict]]]:
    """A fallback group as provider jobs, within the provider's limits."""
    out, chunk, size = [], [], 0
    for item in group:
        n = api.size(f"r{item[0]['idx']}", item[1]["body"])
        if chunk and ((api.max_requests and len(chunk) >= api.max_requests) or (api.max_bytes and size + n > api.max_bytes)):
            out.append(chunk)
            chunk, size = [], 0
        chunk.append(item)
        size += n
    if chunk:
        out.append(chunk)
    return out


async def store_update_tries(store, batch_id: str, lines: list[dict]) -> None:
    await store.update_batch_lines(batch_id, [{"idx": ln["idx"], "tries": ln["tries"], "error": ln.get("error"), "error_code": ln.get("error_code")}
                                              for ln in lines])
