"""GPU lifecycle and cost safety: what the controller and provisioners once got wrong, each as the behaviour they now
have. A GPU that may bill is never left running unseen, uncounted or uncapped."""

from __future__ import annotations

import copy
import json
import time
import types
from datetime import UTC, datetime

import httpx
import pytest
from pydantic import ValidationError
from test_kubernetes import DEPLOYMENTS, ISVC, FakeCluster, k8s_config, pod, probe, provisioner  # noqa: F401 (probe: a fixture)
from test_skypilot import IDENT, fake_sky, replicas, settle, sky_config  # noqa: F401 (fixtures)
from test_worker_controller import pg_memory_store

from meritorder import controller, kube, skypilot
from meritorder.config import (BackendConfig, Config, KubernetesConfig, KubernetesServe, ModelConfig, NodeType, Route, Safeguards,
                               ServeConfig)
from meritorder.controller import billed_seconds, carry_over, deployments, end_pin, keep_lock, tick
from meritorder.gpu import API_ONLY, DRAINING, WARM, Demand, Deployment, Status, desired_replicas
from meritorder.kube import KubeError, Sweeper
from meritorder.skypilot import Fleet, Identity, SkyError, SkyPilotProvisioner
from meritorder.state import MemoryState
from meritorder.store import MemoryStore

MODEL = ModelConfig(name="ow", quality={"coding": 4}, hf_id="o/m", routes=[Route(backend="gpu", model="m")])
PIN = {"deployments": {"gpu": {"mode": "on", "replicas": 1, "by": "ann", "at": 1.0}}}


class Prov:
    """A provisioner that answers what it's told and records every call."""

    def __init__(self, ready=False, usd_per_hour=None):
        self.calls, self.status_calls, self.ready, self.usd_per_hour = [], 0, ready, usd_per_hour

    async def status(self):
        self.status_calls += 1
        return Status(ready=self.ready, endpoint="http://127.0.0.1:9/v1" if self.ready else None, replicas=1 if self.ready else 0,
                      replicas_info=[{"id": 1, "status": "READY" if self.ready else "STARTING", "endpoint": "http://127.0.0.1:9/v1"}],
                      usd_per_hour=self.usd_per_hour)

    async def up(self, n):
        self.calls.append(("up", n))

    async def scale(self, n):
        self.calls.append(("scale", n))

    async def down(self):
        self.calls.append(("down",))


def k8s_backend(**kw) -> BackendConfig:
    return BackendConfig(name="gpu", kind="self_hosted", format="openai",
                         serve=ServeConfig(model="ow", provisioner="kubernetes", scale_up_after_minutes=0,
                                           kubernetes=KubernetesServe(cpu="8", memory="64Gi")), **kw)


@pytest.fixture(autouse=True)
def quick_stops(monkeypatch):
    monkeypatch.setattr(kube, "DOWN_POLL_SECONDS", 0)


# ------------------------------------------------------------------ 1. a GPU with no price, or no demand, isn't run


def test_no_demand_wants_no_replica_whatever_the_price():
    serve = ServeConfig(model="ow")
    assert desired_replicas(serve, 0.0, 1000.0, Demand(0.0, 0.0)) == 0  # nobody asked for anything
    assert desired_replicas(serve, 2.0, 1000.0, Demand(0.0, 0.0), running=1) == 0
    assert desired_replicas(ServeConfig(model="ow", min_replicas=1), 0.0, 1000.0, Demand(0.0, 0.0)) == 1  # yours to set
    assert desired_replicas(serve, 0.0, 1000.0, Demand(0.0, 50.0)) == 1  # a free server still runs while it's used


def test_a_kubernetes_deployment_with_no_price_is_refused_at_load():
    with pytest.raises(ValidationError, match="runs GPUs on Kubernetes with no price"):
        Config(backends=[k8s_backend()], models=[MODEL])
    priced = {"g6": NodeType(usd_per_hour=4.6, gpus=4)}
    with pytest.raises(ValidationError, match="no price"):  # a node type that isn't in the table is no price either
        Config(backends=[k8s_backend()], models=[MODEL], kubernetes=KubernetesConfig(node_types=priced))
    ok = k8s_backend()
    ok.serve.kubernetes.node_type = "g6"
    Config(backends=[ok], models=[MODEL], kubernetes=KubernetesConfig(node_types=priced))
    Config(backends=[k8s_backend(gpu_cost_per_hour=3.0)], models=[MODEL])
    Config(backends=[k8s_backend(enabled=False)], models=[MODEL])  # switched off: nothing to price


async def test_a_billed_deployment_with_no_price_never_starts_even_pinned():
    # SkyPilot whose catalog gave no price, and no gpu_cost_per_hour to fall back on: no cap or budget could hold it.
    backend = BackendConfig(name="gpu", kind="self_hosted", format="openai", serve=ServeConfig(model="ow", scale_up_after_minutes=0))
    cfg = Config(backends=[backend], models=[MODEL], safeguards=Safeguards(max_gpu_usd_per_hour=1.0, gpu_budget_usd_month=1.0))
    store, state, prov = MemoryStore(), MemoryState(), Prov()
    await store.set_setting("gpu_controls", copy.deepcopy(PIN))
    deps = [Deployment(backend=backend, model=MODEL, provisioner=prov)]
    [rec] = await tick(cfg, store, state, deps)
    assert prov.calls == [] and rec["state"] == API_ONLY and rec["planned"] == 0 and "no price for a replica" in rec["note"]


async def test_an_unpriced_server_you_run_yourself_still_serves_while_used(monkeypatch):
    backend = BackendConfig(name="gpu", kind="self_hosted", format="openai",
                            serve=ServeConfig(model="ow", provisioner="static", static_url="http://gpu.test/v1", min_replicas=1))
    cfg = Config(backends=[backend], models=[MODEL])
    prov = Prov(ready=True)
    [rec] = await tick(cfg, MemoryStore(), MemoryState(), [Deployment(backend=backend, model=MODEL, provisioner=prov)])
    assert rec["state"] == WARM  # no cloud bill: the price rule doesn't apply


# ------------------------------------------------------------------ 2. Kubernetes deployments are stopped and swept


async def test_a_kubernetes_deployment_switched_off_in_config_is_stopped():
    cfg = Config(backends=[k8s_backend(gpu_cost_per_hour=10.0)], models=[MODEL])
    store, state, prov = MemoryStore(), MemoryState(), Prov(ready=True)
    await store.set_setting("gpu_controls", {"deployments": {"gpu": {"mode": "on", "replicas": 1, "by": "ann", "at": time.time()}}})
    deps = [Deployment(backend=cfg.backends[0], model=MODEL, provisioner=prov)]
    [rec] = await tick(cfg, store, state, deps)
    assert rec["state"] == WARM and prov.calls == [("up", 1)]
    off = copy.deepcopy(cfg)
    off.backends[0].enabled = False  # the same happens when it's removed, or its model renamed
    deps = await carry_over(deps, deployments(off), store, state)
    assert deps == [] and prov.calls[-1] == ("down",)
    rec = await state.get_json("deploy:gpu")
    assert rec["state"] == API_ONLY and rec["reason"] == "stopped: no longer in the config" and rec["last_run"]
    assert any(e["kind"] == "state" and e["detail"]["to"] == API_ONLY for e in await store.gpu_events())


async def test_a_deployment_that_moved_is_stopped_where_it_ran(monkeypatch):
    cfg = k8s_config()
    old = kube.KubernetesProvisioner(cfg.backends[0], cfg)
    moved = copy.deepcopy(cfg)
    moved.backends[0].serve.kubernetes.namespace = "elsewhere"
    calls = []

    async def down():
        calls.append("down")

    old.down = down
    new = deployments(moved)
    await carry_over([Deployment(cfg.backends[0], cfg.models[0], old)], new)
    assert calls == ["down"]
    calls.clear()
    same = deployments(cfg)
    await carry_over([Deployment(cfg.backends[0], cfg.models[0], old)], same)
    assert calls == []  # the same resource: the new provisioner takes over


async def test_stopping_a_dropped_deployment_that_fails_is_logged_and_left_to_the_sweep(caplog):
    store, state = MemoryStore(), MemoryState()
    await state.set_json("deploy:gpu", {"state": WARM, "replicas": 1})
    prov = Prov()

    async def down():
        raise KubeError("the API is unreachable")

    prov.down = down
    cfg = Config(backends=[k8s_backend(gpu_cost_per_hour=1.0)], models=[MODEL])
    await carry_over([Deployment(cfg.backends[0], MODEL, prov)], [], store, state)
    assert (await state.get_json("deploy:gpu"))["state"] == WARM  # never reported off while it may bill
    assert [e["kind"] for e in await store.gpu_events()] == ["error"] and "unreachable" in caplog.text


def owned(name: str, install: str | None, kind: str = "InferenceService", **meta) -> dict:
    labels = {"app.kubernetes.io/managed-by": "meritorder", kube.MANAGED: name, **({kube.INSTALL: install} if install else {})}
    return {"kind": kind, "metadata": {"name": name, "labels": labels, "resourceVersion": "1", **meta}}


async def test_the_kubernetes_sweep_deletes_only_this_installs_unowned_resources(probe):
    cfg, cluster = k8s_config(), FakeCluster()
    prov = provisioner(cfg, cluster)
    prov.install = "ab12"
    api = prov.api
    cluster.objects = {
        f"{ISVC}/meritorder-gpu": owned("meritorder-gpu", "ab12"),  # the running deployment's: kept
        f"{ISVC}/meritorder-old": owned("meritorder-old", "ab12"),  # a backend removed while the controller was down
        f"{ISVC}/meritorder-theirs": owned("meritorder-theirs", "zz99"),  # another install's, in the same namespace
        f"{DEPLOYMENTS}/meritorder-old-llmd": owned("meritorder-old-llmd", "ab12", kind="Deployment"),
        # KServe's predictor carries the InferenceService's labels, and an owner: never ours to delete
        f"{DEPLOYMENTS}/meritorder-gpu-predictor": owned("meritorder-gpu-predictor", "ab12", kind="Deployment",
                                                          ownerReferences=[{"kind": "InferenceService"}]),
    }
    store = MemoryStore()
    sweeper = Sweeper("ab12", api)
    deps = [Deployment(cfg.backends[0], cfg.models[0], prov)]
    done = await sweeper.sweep(cfg, store, deps, {"gpu": WARM})
    assert sorted(done) == [f"{DEPLOYMENTS}/meritorder-old-llmd", f"{ISVC}/meritorder-old"] and sweeper.leftovers
    assert f"{ISVC}/meritorder-theirs" in cluster.objects and f"{DEPLOYMENTS}/meritorder-gpu-predictor" in cluster.objects
    assert sorted(e["kind"] for e in await store.gpu_events()) == ["sweep", "sweep"]
    # A record that says api_only owns nothing: what's left of a stop that failed goes too.
    assert await sweeper.sweep(cfg, store, deps, {"gpu": API_ONLY}) == [f"{ISVC}/meritorder-gpu"]
    assert await sweeper.sweep(cfg, store, deps, {"gpu": API_ONLY}) == [] and not sweeper.leftovers
    # Nothing on Kubernetes configured and nothing found last time: the cluster isn't asked at all.
    calls = len(cluster.calls)
    assert await sweeper.sweep(sky_config(), store, [], {}) == [] and len(cluster.calls) == calls


async def test_the_kubernetes_sweep_survives_errors(probe, caplog):
    cfg, cluster = k8s_config(), FakeCluster()
    prov = provisioner(cfg, cluster)
    cluster.served = {"deployments"}  # no KServe: its list is a 404, skipped
    cluster.objects = {f"{DEPLOYMENTS}/meritorder-old": owned("meritorder-old", "ab12", kind="Deployment")}
    cluster.fault("DELETE", "meritorder-old", httpx.Response(500, json={"message": "etcd is unhappy"}))
    store = MemoryStore()
    sweeper = Sweeper("ab12", prov.api)
    assert await sweeper.sweep(cfg, store, [Deployment(cfg.backends[0], cfg.models[0], prov)], {}) == []
    assert [e["kind"] for e in await store.gpu_events()] == ["error"] and sweeper.leftovers
    cluster.fault("GET", "deployments", httpx.ConnectError("refused"))
    assert await sweeper.sweep(cfg, store, [], {}) == [] and "can't list" in caplog.text

    def nowhere(*a):
        raise KubeError("no kubeconfig", reason="Config")

    lost = Sweeper("ab12")
    lost.leftovers = True
    kube.KubeAPI.from_environment, real = staticmethod(nowhere), kube.KubeAPI.from_environment
    try:
        assert await lost.sweep(cfg, store, [], {}) == [] and "can't reach the cluster" in caplog.text
    finally:
        kube.KubeAPI.from_environment = real


async def test_the_controller_sweeps_kubernetes_every_cycle(probe):
    cfg, cluster = k8s_config(), FakeCluster()
    prov = provisioner(cfg, cluster)
    cluster.objects = {f"{ISVC}/meritorder-old": owned("meritorder-old", "ab12")}
    await tick(cfg, MemoryStore(), MemoryState(), [Deployment(cfg.backends[0], cfg.models[0], prov)], ksweep=Sweeper("ab12", prov.api))
    assert ("DELETE", f"{ISVC}/meritorder-old") in cluster.calls


async def test_two_installs_in_one_namespace_never_take_each_others_resource(probe):
    cfg, cluster = k8s_config(), FakeCluster()
    prov = provisioner(cfg, cluster)
    prov.install = "ab12"
    assert prov.labels()[kube.INSTALL] == "ab12"
    cluster.objects[f"{ISVC}/meritorder-gpu"] = owned("meritorder-gpu", "zz99")
    with pytest.raises(KubeError, match="belongs to another Merit Order install"):
        await prov.up(1)
    assert not [c for c in cluster.calls if c[0] == "PUT"]
    from meritorder.gpu import make_provisioner

    assert make_provisioner(cfg.backends[0], cfg, Fleet(IDENT)).install == "ab12"


async def test_a_kubernetes_stop_waits_until_its_pods_are_gone(probe, monkeypatch):
    cfg, cluster = k8s_config(), FakeCluster()
    prov = provisioner(cfg, cluster)
    await prov.up(1)
    cluster.pods = [pod("meritorder-gpu-predictor-0", {kube.KSERVE_LABEL: "meritorder-gpu"}, node="n-0", ready=True)]
    cluster.lingering = 3  # the pod takes three more looks to finish its grace period
    await prov.down()
    assert cluster.pods == [] and cluster.lingering == 0
    await prov.up(1)  # a start right after isn't a 409 any more
    cluster.pods = [pod("meritorder-gpu-predictor-0", {kube.KSERVE_LABEL: "meritorder-gpu"}, node="n-0", ready=True)]
    cluster.lingering = 10**6
    monkeypatch.setattr(kube, "DOWN_WAIT_SECONDS", 0)
    prov.k.termination_grace_seconds = 0
    with pytest.raises(KubeError, match="still being deleted"):
        await prov.down()


async def test_the_time_a_stop_takes_is_billed(monkeypatch):
    backend = BackendConfig(name="gpu", kind="self_hosted", format="openai", gpu_cost_per_hour=36.0, serve=ServeConfig(model="ow"))
    cfg = Config(backends=[backend], models=[MODEL])
    store, state, prov = MemoryStore(), MemoryState(), Prov(ready=True)
    t = time.time()
    await state.set_json("deploy:gpu", {"state": DRAINING, "replicas": 1, "since": t - 3600, "accrued_at": t, "burn_usd_per_hour": 36.0})
    clock = [0.0]

    async def slow_down():
        clock[0] += 100  # its pods take 100 s to finish
        prov.calls.append(("down",))

    prov.down = slow_down
    monkeypatch.setattr(controller, "time", types.SimpleNamespace(time=time.time, monotonic=lambda: clock[0]))
    [rec] = await tick(cfg, store, state, [Deployment(backend, MODEL, prov)], now=t)
    assert rec["state"] == API_ONLY
    assert sum((await store.gpu_spend_since(datetime.fromtimestamp(t - 60, UTC))).values()) == pytest.approx(1.0)


# ------------------------------------------------------------------ 3. ending a pin keeps what was saved meanwhile


async def test_ending_a_pin_keeps_a_stop_all_saved_during_the_cycle(store):
    state = MemoryState()
    await store.set_setting("gpu_controls", copy.deepcopy(PIN))
    controls = copy.deepcopy(await store.get_setting("gpu_controls"))  # what tick() read at its start
    # The cycle runs on (SkyPilot calls take minutes); an admin presses Stop all GPUs (controls.stop_all).
    await store.set_setting("gpu_controls", {**copy.deepcopy(PIN), "paused": True, "paused_by": "bob", "paused_at": 2.0, "drain": False})
    await end_pin(store, state, controls, "gpu", "the start failed")
    saved = await store.get_setting("gpu_controls")
    assert saved["paused"] is True and saved["deployments"] == {}, saved
    assert (await state.get_json("gpu:controls")) == saved  # the gateways' copy too
    # Pinned again meanwhile (to retry at once): that newer pin stays.
    again = {"deployments": {"gpu": {**PIN["deployments"]["gpu"], "at": 5.0}}}
    await store.set_setting("gpu_controls", again)
    await end_pin(store, state, copy.deepcopy(PIN), "gpu", "the start failed")
    assert (await store.get_setting("gpu_controls")) == again
    assert not await store.drop_setting_entry("gpu_controls", "deployments", "nobody")
    assert not await store.drop_setting_entry("missing", "deployments", "gpu")


# ------------------------------------------------------------------ 4. the burn cap counts what replicas really burn


async def test_the_fleet_burn_cap_counts_the_actual_burn_rate():
    backend = BackendConfig(name="gpu", kind="self_hosted", format="openai", gpu_cost_per_hour=1.0,
                            serve=ServeConfig(model="ow", scale_up_after_minutes=0))
    cfg = Config(backends=[backend], models=[MODEL], safeguards=Safeguards(max_gpu_usd_per_hour=2.0))
    store, state = MemoryStore(), MemoryState()
    await store.set_setting("gpu_controls", {"deployments": {"gpu": {"mode": "on", "replicas": 1, "by": "ann", "at": time.time()}}})
    prov = Prov(ready=True, usd_per_hour=5.0)  # SkyPilot placed it in a pricier region (or the spot price moved): $5/h
    deps = [Deployment(backend=backend, model=MODEL, provisioner=prov)]
    [rec] = await tick(cfg, store, state, deps)
    assert rec["state"] == WARM and rec["burn_usd_per_hour"] == 5.0
    [rec] = await tick(cfg, store, state, deps, now=time.time() + 60)
    assert rec["planned"] == 0 and rec["state"] != WARM and "fleet limit" in rec["note"]


# ------------------------------------------------------------------ 5. a hung SkyPilot doesn't cut the leases


async def test_the_last_cluster_list_stands_in_after_a_hung_skypilot(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(skypilot, "time", types.SimpleNamespace(monotonic=lambda: clock[0], time=time.time))
    calls = []

    async def sky_call(req, timeout=skypilot.SKY_TIMEOUT):
        calls.append((req["op"], timeout))
        if len(calls) == 1:
            return [{"name": "mo-ab12-gpu-1", "status": "UP", "head_ip": "10.0.0.9"}]
        clock[0] += timeout  # the API server hangs; sky_call gives up after `timeout` seconds
        raise SkyError(f"SkyPilot didn't answer within {timeout:.0f}s")

    monkeypatch.setattr(skypilot, "sky_call", sky_call)
    fleet = Fleet(Identity("ab12", "s"))
    assert len(await fleet.clusters()) == 1
    clock[0] += 60  # the next cycle
    assert [c["name"] for c in await fleet.clusters()] == ["mo-ab12-gpu-1"]
    assert [c["name"] for c in await fleet.clusters()] == ["mo-ab12-gpu-1"] and len(calls) == 2  # the next deployment doesn't wait
    assert calls[1] == ("clusters", skypilot.STATUS_TIMEOUT) and skypilot.STATUS_TIMEOUT * 3 <= 3 * 60  # well within the shortest lease
    clock[0] += skypilot.STALE_SECONDS
    with pytest.raises(SkyError):  # too old by now: each replica stops when its lease runs out
        await fleet.clusters()


# ------------------------------------------------------------------ 6. leases are renewed while the stores are down


async def test_leases_are_renewed_even_when_postgres_is_down():
    backend = BackendConfig(name="gpu", kind="self_hosted", format="openai", gpu_cost_per_hour=1.0, serve=ServeConfig(model="ow"))
    cfg = Config(backends=[backend], models=[MODEL])
    store, state, prov = MemoryStore(), MemoryState(), Prov(ready=True)
    await state.set_json("deploy:gpu", {"state": WARM, "replicas": 1, "since": 0, "warm_at": 0})

    async def broken(since):
        raise RuntimeError("the database is unreachable")

    store.demand_rows = broken
    deps = [Deployment(backend=backend, model=MODEL, provisioner=prov)]
    with pytest.raises(RuntimeError):
        await tick(cfg, store, state, deps)
    assert prov.status_calls == 1  # status() is where leases are renewed
    store.demand_rows = MemoryStore().demand_rows
    await tick(cfg, store, state, deps)
    assert prov.status_calls == 2  # read once a cycle: the start of the cycle's answer is used throughout


async def test_leases_are_renewed_when_redis_is_down_too():
    backend = BackendConfig(name="gpu", kind="self_hosted", format="openai", gpu_cost_per_hour=1.0, serve=ServeConfig(model="ow"))
    prov, idle = Prov(ready=True), Prov(ready=True)
    prov.want, idle.want = 1, None  # asked for one replica / nothing known since this process started

    class Down(MemoryState):
        async def get_json(self, key):
            raise ConnectionError("Redis is down")

    out = await controller.renew_first(Down(), [Deployment(backend, MODEL, prov), Deployment(backend, MODEL, idle)])
    assert prov.status_calls == 1 and idle.status_calls == 0 and list(out) == ["gpu"]


async def test_a_status_that_fails_first_is_read_again_later(caplog):
    backend = BackendConfig(name="gpu", kind="self_hosted", format="openai", gpu_cost_per_hour=1.0, serve=ServeConfig(model="ow"))
    state, prov = MemoryState(), Prov(ready=True)
    await state.set_json("deploy:gpu", {"state": DRAINING, "replicas": 1})
    prov.relaunch = True

    async def boom():
        raise RuntimeError("flaky")

    prov.status = boom
    assert await controller.renew_first(state, [Deployment(backend, MODEL, prov)]) == {}
    assert prov.relaunch is False and "status at the start of the cycle failed" in caplog.text


# ------------------------------------------------------------------ 7. a vLLM that dies after it was ready


async def test_a_vllm_that_dies_after_it_was_ready_is_torn_down_and_backs_off_when_it_does_it_again(fake_sky, replicas, caplog):
    cfg = sky_config()
    prov = SkyPilotProvisioner(cfg.backends[0], cfg, Fleet(IDENT))
    await prov.up(1)
    await settle(prov)
    assert (await prov.status()).ready
    replicas.vllm, replicas.exit_code = "exited", 1
    replicas.gate_status = {"vllm": "exited", "exit_code": 1, "tail": ["CUDA error: out of memory"]}
    st = await prov.status()
    assert st.replicas_info[0]["status"] == "SHUTTING_DOWN" and "  | CUDA error: out of memory" in caplog.text  # its last output
    await settle(prov)
    assert fake_sky.names() == set() and not prov._failed  # torn down now, not when autostop gets to it
    replicas.vllm, replicas.exit_code, replicas.gate_status = "running", None, None
    await prov.status()  # a first crash: launched again at once
    await settle(prov)
    assert len(fake_sky.calls("launch")) == 2 and (await prov.status()).ready
    replicas.vllm, replicas.exit_code = "exited", 1
    replicas.gate_status = {"vllm": "exited", "exit_code": 1, "tail": ["CUDA error: out of memory"]}
    await prov.status()
    await settle(prov)
    st = await prov.status()  # the second within an hour: it waits, as after a failed launch
    assert len(fake_sky.calls("launch")) == 2 and "2 crashes within 60 min" in st.message and st.failed


async def test_a_draining_deployment_doesnt_launch_a_replica_that_went_away(fake_sky, replicas):
    cfg = sky_config()
    prov = SkyPilotProvisioner(cfg.backends[0], cfg, Fleet(IDENT))
    await prov.up(1)
    await settle(prov)
    assert (await prov.status()).ready
    st = fake_sky.state()
    st["clusters"].clear()  # preempted
    fake_sky.path.write_text(json.dumps(st))
    prov.fleet.forget()
    prov.relaunch = False  # the controller says it's draining
    await prov.status()
    assert len(fake_sky.calls("launch")) == 1 and not prov._launching.get("mo-ab12-gpu-2")
    assert prov.renewed_at and prov.renewed_before == prov.renewed_at  # renewed last cycle, nothing to renew this one


# ------------------------------------------------------------------ accrual across a long gap


def test_a_long_gap_is_billed_in_full_where_the_gpus_run_on():
    k8s = Deployment(k8s_config().backends[0], MODEL, Prov())
    assert billed_seconds(k8s, {}, 0.0, 3 * 86400.0) == 3 * 86400  # Kubernetes pods run on without the controller
    sky = Deployment(sky_config().backends[0], MODEL, Prov())
    assert billed_seconds(sky, {}, 0.0, 3 * 86400.0) == 3 * 86400  # no renewal known: all of it
    # Renewed at t=100: the replicas stopped themselves lease_minutes later, and were deleted autostop_minutes after that.
    assert billed_seconds(sky, {"renewed_at": 100.0}, 0.0, 86400.0) == 100 + (15 + 5) * 60
    sky.provisioner.renewed_before = 50_000.0  # this process renewed them since
    assert billed_seconds(sky, {"renewed_at": 100.0}, 0.0, 86400.0) == 50_000 + 20 * 60


# ------------------------------------------------------------------ the controller lock, and Redis errors


class LockPool:
    """A connection that loses its lock: the check fails, and a new connection finds the lock taken (or free)."""

    def __init__(self, retake: bool):
        self.retake, self.acquired, self.released = retake, 0, 0
        self.held: object = True

    async def acquire(self):
        self.acquired += 1
        return self

    async def release(self, conn):
        self.released += 1

    async def fetchval(self, sql, *args):
        if sql == controller.LOCK_HELD:
            if isinstance(self.held, Exception):
                raise self.held
            return self.held
        return self.retake if self.acquired > 1 else True


async def test_a_controller_that_lost_its_lock_stops_acting(monkeypatch, caplog):
    for retake, ticks in ((False, 0), (True, 1)):
        store, state = pg_memory_store(), MemoryState()
        store.pool = LockPool(retake)
        store.pool.held = ConnectionError("the connection was closed")
        monkeypatch.setattr(controller, "make_store", lambda url: store)
        monkeypatch.setattr(controller, "make_state", lambda url: state)
        ran = []

        async def fake_tick(*a, **kw):
            ran.append(1)
            return []

        async def nothing(*a, **kw):
            return None

        monkeypatch.setattr(controller, "tick", fake_tick)
        monkeypatch.setattr(controller, "publish_gpu_prices", nothing)
        await controller.run_controller(Config(backends=[], models=[]), once=True)
        assert len(ran) == ticks and store.pool.acquired == 2
    assert "no longer holds the controller lock" in caplog.text


async def test_keep_lock_when_postgres_cant_be_reached():
    class Gone:
        async def acquire(self):
            raise OSError("connection refused")

        async def release(self, conn):
            raise OSError("connection refused")

    store = types.SimpleNamespace(pool=Gone())
    conn = LockPool(True)
    conn.held = False
    assert await keep_lock(store, conn) == (None, False)
    assert await keep_lock(store, None) == (None, False)


async def test_redis_errors_dont_end_the_controller(monkeypatch, caplog):
    class Flaky(MemoryState):
        async def get_json(self, key):
            if key == "controller:kick":
                raise ConnectionError("Redis is down")
            return await super().get_json(key)

        async def set_json(self, key, value, ttl=0):
            if key == "controller:heartbeat":
                raise ConnectionError("Redis is down")
            await super().set_json(key, value, ttl)

    store, state = pg_memory_store(), Flaky()
    monkeypatch.setattr(controller, "make_store", lambda url: store)
    monkeypatch.setattr(controller, "make_state", lambda url: state)

    async def nothing(*a, **kw):
        return None

    monkeypatch.setattr(controller, "publish_gpu_prices", nothing)  # its failure would hold later tests' fetches back
    await controller.run_controller(Config(backends=[], models=[]), once=True)
    assert "couldn't write the controller's heartbeat" in caplog.text and "can't read the controller's kick" in caplog.text
    clock = [0.0]

    async def sleep(seconds):
        clock[0] += seconds

    monkeypatch.setattr(controller, "asyncio", types.SimpleNamespace(sleep=sleep))
    monkeypatch.setattr(controller, "time", types.SimpleNamespace(monotonic=lambda: clock[0], time=time.time))
    assert await controller.wait_or_kick(state, 10, "old") == "old" and clock[0] == 10  # slept through, no early wake-up
