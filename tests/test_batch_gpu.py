"""Batch work on self-hosted GPUs (decision 49): routed there at the shadow price, held sealed until it runs, run in the
headroom at low priority, handed to a provider's batch API before the deadline, settled as GPU work, counted in metrics."""

import asyncio
import copy
import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from conftest import ADMIN_KEY, TEST_KEY, auth

from meritorder.app import build_gateway, create_app
from meritorder.batches.gpu import BATCH, GpuBatches, GpuRunner
from meritorder.batches.poller import BatchPoller, WorkerContext, serve_metrics
from meritorder.config import build_config
from meritorder.state import MemoryState
from meritorder.store import MemoryStore

CONFIG = {
    "baseline_model": "big",
    "backends": {
        "gpu": {"kind": "self_hosted", "format": "openai", "base_url": "http://gpu.test/v1", "max_concurrency": 5,
                "gpu_cost_per_hour": 3.6, "capacity_tokens_per_second": 1000},
        "mock-openai": {"kind": "mock", "format": "openai", "mock_latency_ms": 0},
        "mock-anthropic": {"kind": "mock", "format": "anthropic", "mock_latency_ms": 0},
    },
    "models": {
        "big": {"quality": {"coding": 5, "general": 5}, "routes": [{"backend": "mock-anthropic", "model": "big-m", "price": {"input": 3, "output": 15}}]},
        "open": {"quality": {"coding": 3, "general": 3}, "routes": [
            {"backend": "gpu", "model": "qwen-gpu", "price": {"input": 0.4, "output": 1.6}},
            {"backend": "mock-openai", "model": "qwen-api", "price": {"input": 0.4, "output": 1.6}}]},
        "gpu-only": {"quality": {"coding": 2, "general": 2}, "routes": [{"backend": "gpu", "model": "tiny-gpu", "price": {"input": 0.1, "output": 0.2}}]},
    },
    "teams": {"default": {}},
    "auth": {"static_keys": [{"key": TEST_KEY, "user": "alice", "team": "default", "role": "user"},
                             {"key": ADMIN_KEY, "user": "root", "team": "ops", "role": "admin"}]},
}
REPLY = {"id": "c1", "object": "chat.completion", "created": 1, "model": "qwen-gpu",
         "choices": [{"index": 0, "message": {"role": "assistant", "content": "done"}, "finish_reason": "stop"}],
         "usage": {"prompt_tokens": 1000, "completion_tokens": 200, "total_tokens": 1200}}


class Vllm:
    """A vLLM server: answers chat completions, or what ``fail`` says, and records each request's priority."""

    def __init__(self):
        self.bodies: list[dict] = []
        self.fail: list[int] = []  # statuses to answer with, in turn, before succeeding
        self.reply = REPLY

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        if self.fail:
            status = self.fail.pop(0)
            return httpx.Response(status, json={"error": {"message": f"vllm says {status}"}})
        return httpx.Response(200, json=self.reply)


@pytest.fixture
def setup(tmp_path):
    def make(**changes):
        raw = copy.deepcopy(CONFIG)
        for path, value in changes.items():
            *head, last = path.split("__")
            node = raw
            for k in head:
                node = node.setdefault(k, {})
            node[last] = value
        cfg = build_config(raw, tmp_path)
        gw = build_gateway(cfg, store=MemoryStore(), state=MemoryState())
        vllm = Vllm()
        gw.backends["gpu"].client = httpx.AsyncClient(transport=httpx.MockTransport(vllm))
        client = httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(gateway=gw)), base_url="http://test")
        return gw, vllm, client
    return make


def lines(n: int, model: str = "open") -> list[dict]:
    return [{"custom_id": f"q{i}", "params": {"model": model, "max_tokens": 200, "messages": [{"role": "user", "content": f"task {i} " * 50}]}}
            for i in range(n)]


async def run(gw, passes: int = 1) -> None:
    poller = BatchPoller(gw)
    for _ in range(passes):
        for j in gw.store._jobs.values():
            j["next_poll_at"] = datetime.now(UTC) - timedelta(seconds=1)
        await poller.run_once()


async def test_batch_work_runs_on_a_running_gpu_in_its_headroom_at_low_priority(setup):
    gw, vllm, c = setup()
    r = await c.post("/v1/messages/batches", headers=auth(), json={"requests": lines(5)})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["meritorder"]["providers"] == {"gpu": 5}  # near free on a GPU with room, against half price elsewhere
    ln = (await gw.store.batch_lines(b["id"]))[0]
    assert "on gpu at its shadow price" in ln["route_reason"] and "else to mock-openai's batch API after 12 h" in ln["route_reason"]
    # What's reserved is the fallback's estimate (it may end up there), not the GPU's near-zero one.
    job = next(iter(gw.store._jobs.values()))
    assert job["api"] == "gpu" and ln["estimated_cost"] > 10 * 0.02 * job["estimated_cost"] / 5
    # The requests are held, sealed, until they run: no content in the clear.
    held = gw.store._blobs
    assert list(held) == [f"{b['id']}/{job['id']}/requests/000000"] and b"task 0" not in b"".join(held[next(iter(held))])
    await run(gw)
    assert len(vllm.bodies) == 5 and {x["priority"] for x in vllm.bodies} == {20} and all(x["model"] == "qwen-gpu" for x in vllm.bodies)
    done = await gw.store.get_batch(b["id"])
    assert done["status"] == "ended" and done["outcome"] == "completed" and done["counts"]["succeeded"] == 5 and done["reserved_cost"] == 0
    row = next(r for r in gw.store.ledger if r["batch_id"] == b["id"])
    # Priced as GPU work: 1,200 tokens at 1,000 a second of a $3.60/hour replica = $0.0012.
    assert row["backend_kind"] == "self_hosted" and row["service_tier"] == "batch" and row["actual_cost"] == pytest.approx(0.0012)
    api_cost = (1000 * 0.4 + 200 * 1.6) / 1e6
    assert row["savings_batch"] == pytest.approx(api_cost / 2)  # what a provider's batch API would have taken off
    assert row["savings_self_hosting"] == pytest.approx(api_cost / 2 - 0.0012)  # and the GPU the rest (here it costs more)
    assert done["savings_self_hosting"] == pytest.approx(5 * row["savings_self_hosting"])
    out = await c.get(f"/v1/messages/batches/{b['id']}/results", headers=auth())
    got = [json.loads(x) for x in out.text.splitlines()]
    assert len(got) == 5 and got[0]["result"]["type"] == "succeeded" and got[0]["result"]["message"]["content"][0]["text"] == "done"
    m = gw.metrics.render()
    assert 'meritorder_batch_requests_total{api="gpu",backend="gpu",model="open",status="200"} 5' in m
    assert 'meritorder_batch_gpu_requests_total{backend="gpu",result="ok"} 5' in m
    assert "meritorder_requests_total" not in m  # batch requests never in the live counters
    # Purged after the retention period: the held requests and results go too.
    gw.store._batches[b["id"]]["ended_at"] -= timedelta(days=8)
    await BatchPoller(gw).purge()
    assert not gw.store._blobs


async def test_no_headroom_no_batch_work_and_other_workers_counted(setup):
    gw, vllm, c = setup()
    b = (await c.post("/v1/messages/batches", headers=auth(), json={"requests": lines(3)})).json()
    await gw.state.publish_inflight("gw-1", {"gpu": 3})  # live requests: 3 of 5 slots, the knee is at 3
    await run(gw)
    assert vllm.bodies == [] and "no headroom" in next(iter(gw.store._jobs.values()))["provider_status"]
    await gw.state.publish_inflight("gw-1", {"gpu": 1})
    await gw.state.publish_inflight("batch-elsewhere", {"gpu" + BATCH: 1})  # another worker's batch work
    poller = BatchPoller(gw)
    job = next(iter(gw.store._jobs.values()))
    runner = GpuRunner(poller, GpuBatches(gw.backends["gpu"], gw.cfg.batches, gw.store, None))
    free, load = await runner.headroom(gw.backends["gpu"])
    assert free == 1 and "1 live and 1 batch of 5 slots" in load
    await gw.state.set_utilization("gpu", 0.9)  # vLLM's own sample says it's busy: no batch work, whatever the counts
    assert (await runner.headroom(gw.backends["gpu"]))[0] == 0
    await gw.state.set_utilization("gpu", 0.0)
    await gw.state.publish_inflight("batch-elsewhere", {"gpu" + BATCH: 0})
    await run(gw, 2)
    assert len(vllm.bodies) == 3 and (await gw.store.get_batch(b["id"]))["status"] == "ended"
    assert job["id"] in gw.store._jobs


async def test_what_hasnt_run_goes_to_a_provider_batch_before_the_deadline(setup):
    gw, vllm, c = setup()
    b = (await c.post("/v1/messages/batches", headers=auth(), json={"requests": lines(4)})).json()
    gw.store._batches[b["id"]]["created_at"] -= timedelta(hours=13)  # past batches.gpu.handoff_hours (12)
    await run(gw)
    assert vllm.bodies == []
    after = await gw.store.get_batch(b["id"])
    assert after["providers"] == {"mock-openai": 4}
    jobs = list(gw.store._jobs.values())
    moved = next(j for j in jobs if j["api"] == "mock")
    assert moved["provider"]["handed_from"] and moved["request_count"] == 4
    lns = await gw.store.batch_lines(b["id"])
    assert {ln["backend"] for ln in lns} == {"mock-openai"} and "handed from gpu to mock-openai (deadline)" in lns[0]["route_reason"]
    await run(gw, 2)
    done = await gw.store.get_batch(b["id"])
    assert done["status"] == "ended" and done["counts"]["succeeded"] == 4
    rows = [r for r in gw.store.ledger if r["batch_id"] == b["id"]]
    assert {r["backend"] for r in rows} == {"mock-openai"} and rows[0]["savings_batch"] > 0 and rows[0]["savings_self_hosting"] == 0
    assert 'meritorder_batch_handoffs_total{backend="gpu",why="deadline"} 4' in gw.metrics.render()


async def test_a_stopped_gpu_hands_on_at_once_and_failures_go_to_the_fallback(setup):
    gw, vllm, c = setup()
    b = (await c.post("/v1/messages/batches", headers=auth(), json={"requests": lines(3)})).json()
    vllm.fail = [400, 503, 503]  # one the GPU refuses outright, two it may take on another try
    await run(gw)
    lns = {ln["custom_id"]: ln for ln in await gw.store.batch_lines(b["id"])}
    moved = [ln for ln in lns.values() if ln["backend"] == "mock-openai"]
    assert len(moved) == 1 and "failed" in moved[0]["route_reason"]  # the 400: the provider may take it
    retried = [ln for ln in lns.values() if ln["tries"] == 1 and ln["backend"] == "gpu"]
    assert len(retried) == 2  # 503s: tried again, not handed on yet
    gw.backends["gpu"].available = False  # the deployment stopped
    await run(gw, 3)
    done = await gw.store.get_batch(b["id"])
    assert done["status"] == "ended" and done["counts"]["succeeded"] == 3 and done["providers"] == {"mock-openai": 3}
    assert 'why="stopped"' in gw.metrics.render()


async def test_without_a_fallback_a_request_waits_for_the_gpu_until_the_deadline(setup):
    gw, vllm, c = setup()
    b = (await c.post("/v1/messages/batches", headers=auth(), json={"requests": lines(2, "gpu-only")})).json()
    ln = (await gw.store.batch_lines(b["id"]))[0]
    assert "it waits for the GPU until the batch's deadline" in ln["route_reason"]
    gw.backends["gpu"].available = False
    await run(gw)
    assert (await gw.store.get_batch(b["id"]))["status"] == "in_progress"
    assert "waiting for the deployment" in next(iter(gw.store._jobs.values()))["provider_status"]
    gw.store._batches[b["id"]]["expires_at"] = datetime.now(UTC) - timedelta(minutes=1)
    await run(gw)
    done = await gw.store.get_batch(b["id"])
    assert done["status"] == "ended" and done["outcome"] == "expired" and done["counts"]["expired"] == 2
    # A request the GPU keeps failing, with nowhere else to go, fails with the GPU's error.
    gw.backends["gpu"].available = True
    b2 = (await c.post("/v1/messages/batches", headers=auth(), json={"requests": lines(1, "gpu-only")})).json()
    vllm.fail = [422]
    await run(gw)
    out = [json.loads(x) for x in (await c.get(f"/v1/messages/batches/{b2['id']}/results", headers=auth())).text.splitlines()]
    assert out[0]["result"]["type"] == "errored" and "vllm says 422" in out[0]["result"]["error"]["error"]["message"]


async def test_cancelling_stops_gpu_work(setup):
    gw, vllm, c = setup()
    b = (await c.post("/v1/messages/batches", headers=auth(), json={"requests": lines(2)})).json()
    assert (await c.post(f"/v1/messages/batches/{b['id']}/cancel", headers=auth())).status_code == 200
    await run(gw)
    done = await gw.store.get_batch(b["id"])
    assert vllm.bodies == [] and done["outcome"] == "cancelled" and done["counts"]["canceled"] == 2


async def test_gpu_routing_follows_the_shadow_price_and_the_deployment(setup):
    gw, vllm, c = setup()
    await gw.state.set_utilization("gpu", 0.99)  # nearly full: dearer than half price
    b = (await c.post("/v1/messages/batches", headers=auth(), json={"requests": lines(1)})).json()
    assert b["meritorder"]["providers"] == {"mock-openai": 1}
    await gw.state.set_utilization("gpu", 0.0)
    gw.backends["gpu"].draining = True
    assert (await c.post("/v1/messages/batches", headers=auth(), json={"requests": lines(1)})).json()["meritorder"]["providers"] == {"mock-openai": 1}
    gw.backends["gpu"].draining = False
    gw.backends["gpu"].available = False  # not running: batch work never starts one
    r = await c.post("/v1/messages/batches", headers=auth(), json={"requests": lines(1, "gpu-only")})
    assert r.status_code == 400 and "aren't running" in r.text
    gw.backends["gpu"].available = True
    gw.cfg.batches.gpu.enabled = False
    r = await c.post("/v1/messages/batches", headers=auth(), json={"requests": lines(1, "gpu-only")})
    assert r.status_code == 400 and "batches.gpu.enabled" in r.text


async def test_without_a_batch_key_gpu_work_goes_to_the_providers(setup, monkeypatch):
    gw, vllm, c = setup()
    monkeypatch.setattr(gw.store, "kind", "postgres")  # a database-backed gateway without MERITORDER_BATCH_KEY holds nothing
    b = (await c.post("/v1/messages/batches", headers=auth(), json={"requests": lines(2)})).json()
    assert b["meritorder"]["providers"] == {"mock-openai": 2} and not gw.store._blobs


async def test_a_job_handed_on_late_is_cancelled_at_the_clients_deadline(setup, monkeypatch):
    """The provider gives a handed-on job a fresh 24 hours, which would end after the client's: the worker asks again by the
    batch's deadline, cancels the job if it's still running, and what the provider hadn't run expires."""
    from meritorder.batches.providers import MockBatches, Outcome, Polled, _counts

    gw, vllm, c = setup()
    b = (await c.post("/v1/messages/batches", headers=auth(), json={"requests": lines(3)})).json()
    gw.store._batches[b["id"]]["created_at"] -= timedelta(hours=13)
    await run(gw)
    moved = next(j for j in gw.store._jobs.values() if j["api"] == "mock")
    cancelled: list[str] = []

    async def poll(self, job):
        if cancelled:
            return Polled(True, "ended", _counts(succeeded=1, canceled=2))
        return Polled(False, "in_progress", _counts(processing=3))

    async def cancel(self, job):
        cancelled.append(job["id"])

    async def results(self, job, want=None):
        for i, ln in enumerate(await gw.store.batch_lines(b["id"], job["id"])):
            key = f"r{ln['idx']}"
            yield Outcome(key, "succeeded", REPLY) if i == 0 else Outcome(key, "canceled")

    monkeypatch.setattr(MockBatches, "poll", poll)
    monkeypatch.setattr(MockBatches, "cancel", cancel)
    monkeypatch.setattr(MockBatches, "results", results)
    deadline = datetime.now(UTC) + timedelta(seconds=20)
    gw.store._batches[b["id"]]["expires_at"] = deadline
    await run(gw)
    job = gw.store._jobs[moved["id"]]
    assert job["status"] == "running" and not cancelled and job["next_poll_at"] <= deadline  # asked again by the deadline
    gw.store._batches[b["id"]]["expires_at"] = datetime.now(UTC) - timedelta(seconds=1)
    await run(gw, 2)
    assert cancelled == [moved["id"]]
    done = await gw.store.get_batch(b["id"])
    assert done["status"] == "ended" and done["outcome"] == "expired"
    assert done["counts"]["succeeded"] == 1 and done["counts"]["expired"] == 2 and done["counts"]["canceled"] == 0
    out = [json.loads(x) for x in (await c.get(f"/v1/messages/batches/{b['id']}/results", headers=auth())).text.splitlines()]
    assert sorted(r["result"]["type"] for r in out) == ["expired", "expired", "succeeded"]


async def test_past_the_deadline_nothing_is_handed_on_and_openrouter_is_never_a_fallback(setup):
    gw, vllm, c = setup()
    b = (await c.post("/v1/messages/batches", headers=auth(), json={"requests": lines(2)})).json()
    gw.store._batches[b["id"]]["created_at"] -= timedelta(hours=25)
    gw.store._batches[b["id"]]["expires_at"] = datetime.now(UTC) - timedelta(hours=1)
    await run(gw)
    done = await gw.store.get_batch(b["id"])
    assert done["status"] == "ended" and done["counts"]["expired"] == 2 and done["providers"] == {"gpu": 2}
    assert not [j for j in gw.store._jobs.values() if j["api"] != "gpu"]

    from meritorder.batches.service import _NONE, _can_take_handoff
    from meritorder.batches.providers import OpenRouterBatches

    assert _can_take_handoff(_NONE) and not OpenRouterBatches.cancels
    assert not _can_take_handoff(OpenRouterBatches(gw.backends["mock-openai"], gw.cfg.batches))


async def test_a_gpu_job_whose_backend_is_gone_is_handed_on(setup):
    gw, vllm, c = setup()
    b = (await c.post("/v1/messages/batches", headers=auth(), json={"requests": lines(2)})).json()
    del gw.backends["gpu"]  # switched off or removed from the configuration
    await run(gw, 2)
    done = await gw.store.get_batch(b["id"])
    assert done["status"] == "ended" and done["providers"] == {"mock-openai": 2}


async def test_a_refused_handoff_and_lost_content_fail_their_requests(setup, monkeypatch):
    from meritorder.backends import BackendError
    from meritorder.batches import gpu as gpu_mod
    from meritorder.batches.providers import MockBatches

    gw, vllm, c = setup()
    b = (await c.post("/v1/messages/batches", headers=auth(), json={"requests": lines(3)})).json()
    job = next(iter(gw.store._jobs.values()))
    assert (await GpuBatches(gw.backends["gpu"], gw.cfg.batches, gw.store, None).poll(job)).provider_status == "running"

    async def refuse(self, job, requests):
        raise BackendError(400, "mock-openai: no batch for you")

    monkeypatch.setattr(MockBatches, "submit", refuse)
    gw.store._batches[b["id"]]["created_at"] -= timedelta(hours=13)
    monkeypatch.setattr(gpu_mod, "HELD_PIECE", 1)
    await run(gw)
    done = await gw.store.get_batch(b["id"])
    assert done["status"] == "ended" and done["counts"]["errored"] == 3 and done["actual_cost"] == 0
    out = [json.loads(x) for x in (await c.get(f"/v1/messages/batches/{b['id']}/results", headers=auth())).text.splitlines()]
    assert "no batch for you" in out[0]["result"]["error"]["error"]["message"]
    monkeypatch.undo()

    # Held requests gone (deleted from the bucket by hand): they can't run.
    b2 = (await c.post("/v1/messages/batches", headers=auth(), json={"requests": lines(2)})).json()
    gw.store._blobs.clear()
    await run(gw)
    out = [json.loads(x) for x in (await c.get(f"/v1/messages/batches/{b2['id']}/results", headers=auth())).text.splitlines()]
    assert len(out) == 2 and "no longer held" in out[0]["result"]["error"]["error"]["message"] and vllm.bodies == []


async def test_a_reply_with_an_error_goes_to_the_fallback_and_limits_split_it(setup):
    from meritorder.batches.gpu import _split

    gw, vllm, c = setup()
    b = (await c.post("/v1/messages/batches", headers=auth(), json={"requests": lines(1)})).json()
    vllm.reply = {"error": {"message": "bad tool"}}  # a 200 that's an error: the provider may do better
    await run(gw)
    assert (await gw.store.batch_lines(b["id"]))[0]["backend"] == "mock-openai"
    api = BatchPoller(gw).api("mock-openai")
    api.max_requests = 2
    group = [({"idx": i}, {"body": {"x": i}}) for i in range(5)]
    assert [len(c_) for c_ in _split(api, group)] == [2, 2, 1]
    gw.backends["gpu"].cfg.max_concurrency = None
    runner = GpuRunner(BatchPoller(gw), GpuBatches(gw.backends["gpu"], gw.cfg.batches, gw.store, None))
    assert await runner.headroom(gw.backends["gpu"]) == (0, "no max_concurrency")

    async def broken(*a, **kw):
        raise RuntimeError("redis down")

    gw.state.publish_inflight = broken
    await runner.publish(gw.backends["gpu"])  # advisory: a failure is logged, not raised


async def test_the_worker_follows_the_gpus_and_serves_its_metrics(setup, monkeypatch):
    gw, vllm, c = setup(**{"backends__gpu__serve": {"provisioner": "static", "model": "open", "static_url": "http://gpu.test/v1"}})
    ctx = WorkerContext(gw.store, gw.state)
    await gw.state.set_json("deploy:gpu", {"state": "warm", "endpoint": "http://gpu.test/v1", "replicas": 2, "burn_usd_per_hour": 7.2})
    await ctx.use(gw.cfg)
    assert ctx.backends["gpu"].available and ctx.backends["gpu"].replicas == 2 and ctx.backends["gpu"].live_cost_per_hour == 3.6
    ctx.metrics.inc("meritorder_batch_cost_usd_total", {"model": "m", "backend": "b"}, 0.5)
    server = await serve_metrics(ctx.metrics, 0, "127.0.0.1")
    port = server.sockets[0].getsockname()[1]

    async def get(path: str, headers: str = "") -> bytes:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(f"GET {path} HTTP/1.1\r\nhost: x\r\n{headers}\r\n".encode())
        await writer.drain()
        data = await reader.read()
        writer.close()
        return data

    try:
        assert b'meritorder_batch_cost_usd_total{backend="b",model="m"} 0.5' in await get("/metrics")
        assert b"200 OK" in await get("/healthz") and b"404" in await get("/nope")
        monkeypatch.setenv("MERITORDER_METRICS_TOKEN", "tok")
        assert b"401" in await get("/metrics")
        assert b"200 OK" in await get("/metrics", "authorization: Bearer tok\r\n")
    finally:
        server.close()
        await ctx.close()
