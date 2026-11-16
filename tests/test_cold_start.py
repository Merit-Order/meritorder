"""Cold starts: the gate's and the controller's marks of a replica's start, the phases and the rolling average, the
cold start in the break-even, the faster-start settings, and the weights cache volume (made before a launch, counted
in GPU spend, never swept, deleted only by `meritorder gpu volumes delete`). Everything external is a stand-in."""

from __future__ import annotations

import asyncio
import sys
import time
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from test_skypilot import IDENT, REAL_ASYNC_CLIENT, UPSTREAM, fake_sky, free_port, replicas, settle, sky_config, wait_for_port  # noqa: F401

from meritorder import cli, controller, gpu, replica_gate, sky_clouds, skypilot
from meritorder.config import BackendConfig, Config, KubernetesServe, ModelConfig, Route, ServeConfig, _warn_skypilot_clouds
from meritorder.gpu import API_ONLY, PROVISIONING, WARM, Demand, Deployment, Status
from meritorder.skypilot import Fleet, SkyPilotProvisioner
from meritorder.state import MemoryState
from meritorder.store import MemoryStore

RUNPOD = {"clouds": ["runpod"], "weights_cache": "volume", "weights_cache_infra": "runpod/CA/CA-MTL-1", "weights_cache_gb": 50}

# ------------------------------------------------------------------ the gate's marks


def test_machine_and_container_start_from_proc(tmp_path, monkeypatch):
    proc = tmp_path / "proc"
    (proc / "1").mkdir(parents=True)
    (proc / "uptime").write_text("600.00 1200.00\n")
    (proc / "1" / "comm").write_text("bash\n")
    fields = ["S"] + ["0"] * 18 + ["30000"]  # field 22 (starttime) is the 20th after the name
    (proc / "1" / "stat").write_text("1 (bash (x)) " + " ".join(fields) + " 0 0\n")
    monkeypatch.setattr(replica_gate.os, "sysconf", lambda name: 100)
    t = replica_gate.machine_times(str(proc), now=10_000.0)
    assert t == {"machine_booted": 9400.0, "container_started": 9700.0}  # booted 600 s ago; PID 1 started 300 s after boot
    (proc / "1" / "comm").write_text("systemd\n")  # the host's PID namespace: PID 1 is the machine's init
    assert replica_gate.machine_times(str(proc), now=10_000.0) == {"machine_booted": 9400.0, "container_started": None}
    (proc / "1" / "comm").unlink()
    assert replica_gate.machine_times(str(proc), now=10_000.0)["container_started"] is None
    assert replica_gate.machine_times(str(tmp_path / "nowhere")) == {"machine_booted": None, "container_started": None}


def test_gate_notes_vllms_own_marks_once():
    gate = replica_gate.Gate(["true"], "0" * 64, 60, 1, 2, gpu=lambda: {}, times=lambda: {"machine_booted": 1.0, "container_started": 2.0})
    assert gate.startup["machine_booted"] == 1.0 and gate.startup["gate_started"] > 0 and gate.startup["weights_loaded"] is None
    gate.mark("INFO weight_utils.py: Time spent downloading weights for openai/gpt-oss-20b: 95.123456 seconds")
    gate.mark("INFO default_loader.py: Loading weights took 21.37 seconds")
    first = gate.startup["weights_loaded"]
    gate.mark("INFO gpu_model_runner.py: Model loading took 13.7 GiB memory and 24.501234 seconds")  # the first said it already
    assert gate.startup["download_seconds"] == 95.123456 and gate.startup["load_seconds"] == 21.37 and gate.startup["weights_loaded"] == first
    other = replica_gate.Gate(["true"], "0" * 64, 60, 1, 2, gpu=lambda: {}, times=lambda: {})
    other.mark("Model loading took 13.7 GiB memory and 24.501234 seconds")
    assert other.startup["load_seconds"] == 24.501234
    assert "startup" in other.status()


async def test_gate_reports_its_start_on_the_lease_and_the_status():
    key = IDENT.api_key("gpu")
    port, upstream = free_port(), free_port()
    gate = replica_gate.Gate([sys.executable, "-c", UPSTREAM, str(upstream)], IDENT.key_sha256("gpu"), 30, port, upstream, grace_seconds=2,
                             host="127.0.0.1", gpu=lambda: {}, preflight=lambda: [], times=lambda: {"machine_booted": 5.0})
    run = asyncio.ensure_future(gate.run())
    await wait_for_port(port)
    async with REAL_ASYNC_CLIENT(base_url=f"http://127.0.0.1:{port}", timeout=10) as c:
        for _ in range(100):
            lease = (await c.post(replica_gate.LEASE_PATH, headers={"x-api-key": key})).json()
            if lease["startup"]["vllm_listening"]:
                break
            await asyncio.sleep(0.1)
        st = (await c.get(replica_gate.STATUS_PATH, headers={"x-api-key": key})).json()
    s = lease["startup"]
    assert s["machine_booted"] == 5.0 and s["gate_started"] <= s["vllm_started"] <= s["vllm_listening"]
    assert st["startup"]["vllm_listening"] == s["vllm_listening"]
    gate.expires = time.monotonic()
    assert await asyncio.wait_for(run, 10) == 0


# ------------------------------------------------------------------ phases


def test_phases_between_the_marks_that_are_known():
    marks = {"launch_requested": 0, "machine_booted": 100, "container_started": 400, "cluster_up": 470, "gate_started": 465,
             "vllm_started": 480, "weights_loaded": 600, "vllm_listening": 640, "ready": 660, "load_seconds": 20.0}
    s = skypilot.startup_phases(marks)
    assert [(p["phase"], p["seconds"]) for p in s["phases"]] == [
        ("machine boots", 100), ("image pull", 300), ("SkyPilot setup", 70), ("task start", 0),  # two clocks: clamped at zero
        ("GPU check", 10), ("weights download + weights load", 120), ("engine start", 40), ("first ready probe", 20)]
    assert s["total_seconds"] == 660 and s["load_seconds"] == 20.0
    assert sum(p["seconds"] for p in s["phases"]) == 660  # the phases add up to the whole, whatever the clamping
    partial = skypilot.startup_phases({"gate_started": 10, "vllm_started": 12})
    assert partial == {"phases": [{"phase": "GPU check", "seconds": 2}], "total_seconds": None}
    assert skypilot._minutes(None) == "?" and skypilot._minutes(30) == "30s" and skypilot._minutes(660) == "11.0 min"


async def test_a_ready_replica_says_how_its_start_went(fake_sky, replicas, monkeypatch):
    cfg = sky_config()
    prov = SkyPilotProvisioner(cfg.backends[0], cfg, Fleet(IDENT))
    replicas.ready = False
    await prov.up(1)
    await settle(prov)
    marks = prov._marks["mo-ab12-gpu-1"]
    assert marks["cluster_up"] >= marks["launch_requested"]
    st = await prov.status()
    assert st.replicas_info[0]["status"] == "STARTING" and st.replicas_info[0]["startup"]["total_seconds"] is None
    assert prov.drain_events() == []
    replicas.ready = True
    st = await prov.status()
    s = st.replicas_info[0]["startup"]
    assert s["total_seconds"] is not None and s["phases"][-1]["phase"].endswith("first ready probe")
    [(kind, ev)] = prov.drain_events()
    assert kind == "replica_ready" and ev["cluster"] == "mo-ab12-gpu-1" and ev["where"] == "lambda us-east-1 gpu_1x_a10"
    await prov.status()
    assert prov.drain_events() == []  # once per start
    # After a restart nothing knows when it was asked for: no phases are made up.
    fresh = SkyPilotProvisioner(cfg.backends[0], cfg, Fleet(IDENT))
    fresh.want = 1
    st = await fresh.status()
    assert "startup" not in st.replicas_info[0] and fresh.drain_events() == []


# ------------------------------------------------------------------ the rolling average and the break-even


def test_the_cold_start_is_the_average_of_the_last_starts():
    serve = ServeConfig(model="ow", cold_start_minutes=12)
    assert gpu.expected_cold_start(serve, {}) == (12.0, "configured")
    assert gpu.expected_cold_start(ServeConfig(model="ow", provisioner="static"), {}) == (0.0, "static")
    rec: dict = {}
    for s in [600, 720, 660]:
        gpu.record_cold_start(rec, s, {"cluster": "c"}, now=1.0)
    assert rec["cold_start"]["avg_seconds"] == 660 and rec["cold_start"]["last"] == {"cluster": "c", "total_seconds": 660, "at": 1.0}
    assert gpu.expected_cold_start(serve, rec) == (11.0, "measured")
    for _ in range(20):
        gpu.record_cold_start(rec, 60)
    assert len(rec["cold_start"]["samples"]) == gpu.COLD_START_SAMPLES and rec["cold_start"]["avg_seconds"] == 60


def test_a_start_must_pay_back_its_cold_start():
    serve = ServeConfig(model="ow", hold_minutes=30, margin=0.2, max_replicas=4)
    assert gpu.start_factor(serve, 12) == pytest.approx(42 / 30)
    burst = Demand(usd_per_hour=26, tokens_per_second=10)  # pays $20/h + 20%, not with 12 min paid and idle on top
    assert gpu.desired_replicas(serve, 20.0, 1000, burst) == 1  # without the cold start: one replica
    assert gpu.desired_replicas(serve, 20.0, 1000, burst, cold_start_minutes=12) == 0  # with it: not worth starting
    assert gpu.desired_replicas(serve, 20.0, 1000, Demand(34, 10), cold_start_minutes=12) == 1  # 24 x 1.4 = 33.6
    # A running replica's cold start is paid: it stays while demand covers its price and the margin.
    assert gpu.desired_replicas(serve, 20.0, 1000, burst, running=1, cold_start_minutes=12) == 1
    # One more must pay back its own start: $50/h keeps one ($24) but $26 left doesn't start another ($33.60).
    assert gpu.desired_replicas(serve, 20.0, 1000, Demand(50, 5000), running=1, cold_start_minutes=12) == 1
    assert gpu.desired_replicas(serve, 20.0, 1000, Demand(58, 5000), running=1, cold_start_minutes=12) == 2
    assert gpu.desired_replicas(serve, 20.0, 1000, Demand(30, 5000), running=2, cold_start_minutes=12) == 1  # scale down as before
    assert gpu.desired_replicas(serve, 0.0, 1000, Demand(1, 5000), cold_start_minutes=12) == 4  # free GPUs


def test_demand_that_comes_back_after_a_quiet_cycle_starts_the_wait():
    serve = ServeConfig(model="ow", scale_up_after_minutes=10)
    rec, _ = gpu.decide({}, serve, 0, Status(), 0.0)
    assert rec["over_since"] is None
    rec, act = gpu.decide(rec, serve, 1, Status(), 60.0)  # used to fail: None - a time
    assert rec["over_since"] == 60.0 and act is None
    rec, act = gpu.decide(rec, serve, 1, Status(), 60.0 + 601)
    assert act.kind == "up"


class Prov:
    def __init__(self, events=()):
        self.events, self.ready, self.calls = list(events), False, []

    def drain_events(self):
        out, self.events = self.events, []
        return out

    async def status(self):
        return Status(ready=self.ready, endpoint="http://127.0.0.1:9/v1" if self.ready else None, replicas=int(self.ready))

    async def up(self, n):
        self.calls.append(("up", n))

    async def scale(self, n):
        self.calls.append(("scale", n))

    async def down(self):
        self.calls.append(("down",))


def deployment(provisioner: str = "kubernetes", **serve) -> tuple[Config, BackendConfig, ModelConfig]:
    model = ModelConfig(name="ow", quality={"coding": 4}, hf_id="o/m", routes=[Route(backend="gpu", model="m")])
    kw = {"kubernetes": KubernetesServe(cpu="8", memory="32Gi")} if provisioner == "kubernetes" else {}
    backend = BackendConfig(name="gpu", kind="self_hosted", format="openai", gpu_cost_per_hour=10.0,
                            serve=ServeConfig(model="ow", provisioner=provisioner, scale_up_after_minutes=0, **kw, **serve))
    return Config(backends=[backend], models=[model]), backend, model


async def demand(store, dollars: int) -> None:
    now = datetime.now(UTC)
    for _ in range(dollars):
        await store.record({"ts": now, "floor": "coding:3", "served_model": "sonnet", "requested_model": "auto", "served_api_cost": 1.0,
                            "actual_cost": 1.0, "status": 200, "upgraded": False, "input_tokens": 1000, "output_tokens": 100})


async def test_the_controller_counts_the_cold_start_before_it_starts(monkeypatch):
    cfg, backend, model = deployment(cold_start_minutes=30, hold_minutes=30)
    store, state = MemoryStore(), MemoryState()
    await demand(store, 5)  # $5 in the last 15 minutes is $20/h, which pays $10/h + 20%, but not twice that (30 min paid without serving, over a 30 min hold)
    prov = Prov()
    deps = [Deployment(backend=backend, model=model, provisioner=prov)]
    [rec] = await controller.tick(cfg, store, state, deps)
    assert rec["state"] == API_ONLY and rec["desired"] == 0 and prov.calls == []
    assert rec["start_needs"] == {"usd_per_hour": 24.0, "cold_start_minutes": 30.0, "basis": "configured", "hold_minutes": 30}
    await demand(store, 2)  # $28/h
    [rec] = await controller.tick(cfg, store, state, deps)
    assert rec["state"] == PROVISIONING and prov.calls == [("up", 1)]
    # On a cluster, the start is timed as a whole: from the controller's start to the first ready replica.
    prov.ready = True
    [rec] = await controller.tick(cfg, store, state, deps, now=rec["started_at"] + 300)
    assert rec["state"] == WARM and rec["cold_start"]["samples"] == [300.0]
    [ev] = [e for e in await store.gpu_events() if e["kind"] == "replica_ready"]
    assert ev["detail"] == {"total_seconds": 300.0, "phases": []}
    [rec] = await controller.tick(cfg, store, state, deps, now=rec["started_at"] + 400)
    assert rec["start_needs"]["basis"] == "measured" and rec["start_needs"]["cold_start_minutes"] == 5.0
    assert len([e for e in await store.gpu_events() if e["kind"] == "replica_ready"]) == 1


async def test_replica_ready_events_go_into_the_average():
    cfg, backend, model = deployment("skypilot")
    store = MemoryStore()
    rec: dict = {}
    prov = Prov([("replica_ready", {"cluster": "c-1", "total_seconds": 660.0, "phases": [{"phase": "image pull", "seconds": 300}]}),
                 ("replica_ready", {"cluster": "c-2", "total_seconds": None, "phases": []})])
    await controller.write_events(store, Deployment(backend=backend, model=model, provisioner=prov), rec, now=5.0)
    assert rec["cold_start"]["samples"] == [660.0] and rec["cold_start"]["last"]["phases"][0]["phase"] == "image pull"
    assert [e["kind"] for e in await store.gpu_events()] == ["replica_ready", "replica_ready"]


# ------------------------------------------------------------------ faster starts


def test_every_replica_gets_the_fast_download_and_the_load_format_asked_for():
    cfg = sky_config(load_format="runai_streamer")
    prov = SkyPilotProvisioner(cfg.backends[0], cfg, Fleet(IDENT))
    task = prov.task_config(prov.cluster(1))
    assert task["envs"]["HF_XET_HIGH_PERFORMANCE"] == "1" and "volumes" not in task and "HF_HOME" not in task["envs"]
    assert "--load-format runai_streamer" in task["run"]
    plain = sky_config()
    assert "--load-format" not in " ".join(gpu.vllm_command(plain.backends[0], plain))  # auto: vLLM's own default


def test_a_registry_login_goes_with_the_launch_only_when_set(monkeypatch):
    for k in skypilot.DOCKER_LOGIN:
        monkeypatch.delenv(k, raising=False)
    assert skypilot.launch_secrets() == ["HF_TOKEN"]
    monkeypatch.setenv("SKYPILOT_DOCKER_SERVER", "123.dkr.ecr.us-east-1.amazonaws.com")
    assert skypilot.launch_secrets() == ["HF_TOKEN", *skypilot.DOCKER_LOGIN]


# ------------------------------------------------------------------ the weights cache volume


def test_volumes_only_where_skypilot_has_them():
    def problem(**kw):
        base = {"provisioner": "skypilot", "infra": "runpod/CA/CA-MTL-1", "size_gb": 50, "clouds": ["runpod"], "regions": []}
        base.update(kw)
        return sky_clouds.volume_problem(**base)

    assert problem() is None and problem(infra="kubernetes", clouds=["k8s"]) is None
    assert "only SkyPilot" in problem(provisioner="kubernetes")
    assert "must say where" in problem(infra=None)
    assert "not on lambda" in problem(infra="lambda/us-east-1", clouds=["lambda"])
    assert "one data center" in problem(infra="runpod/CA")
    assert "how big" in problem(size_gb=None)
    assert "at least 10 GB" in problem(size_gb=5)
    assert "set clouds to [runpod]" in problem(clouds=[]) and "set clouds" in problem(clouds=["runpod", "aws"])
    assert "set regions to [CA]" in problem(regions=["US"]) and problem(regions=["CA"]) is None
    with pytest.raises(ValueError, match="weights_cache is volume, but SkyPilot 0.14.2 has volumes only on kubernetes and runpod"):
        ServeConfig(model="ow", clouds=["lambda"], weights_cache="volume", weights_cache_infra="lambda/us-east-1", weights_cache_gb=50)
    assert ServeConfig(model="ow", **RUNPOD).weights_cache == "volume"


def test_what_a_volume_costs_a_month(caplog):
    assert sky_clouds.volume_usd_per_month("runpod/CA/CA-MTL-1", 50, None) == 3.5
    assert sky_clouds.volume_usd_per_month("runpod/CA/CA-MTL-1", 1500, None) == 95.0  # $0.05 a GB beyond the first TB
    assert sky_clouds.volume_usd_per_month("kubernetes", 100, None) is None
    assert sky_clouds.volume_usd_per_month("kubernetes", 100, 0.1) == 10.0
    assert sky_clouds.volume_usd_per_month("kubernetes", None, 0.1) is None
    caplog.set_level("INFO", "meritorder.config")
    _warn_skypilot_clouds(sky_config(**RUNPOD))
    _warn_skypilot_clouds(sky_config(clouds=["kubernetes"], weights_cache="volume", weights_cache_infra="kubernetes", weights_cache_gb=50))
    assert "a 50 GB volume on runpod/CA/CA-MTL-1, about $3.50 a month" in caplog.text
    assert "at your storage class's price" in caplog.text


def test_the_task_mounts_the_volume_and_puts_the_caches_on_it():
    cfg = sky_config(**RUNPOD, regions=["CA"])
    prov = SkyPilotProvisioner(cfg.backends[0], cfg, Fleet(IDENT))
    task = prov.task_config(prov.cluster(1), [["runpod", "CA"]])
    assert task["volumes"] == {skypilot.CACHE_MOUNT: "mo-ab12-gpu-hf"}
    assert task["envs"]["HF_HOME"] == f"{skypilot.CACHE_MOUNT}/huggingface" and task["envs"]["VLLM_CACHE_ROOT"] == f"{skypilot.CACHE_MOUNT}/vllm"
    spec = prov.volume_spec()
    assert spec["type"] == "runpod-network-volume" and spec["size"] == "50GB" and spec["config"] == {}
    k8s = sky_config(clouds=["kubernetes"], weights_cache="volume", weights_cache_infra="kubernetes", weights_cache_gb=50, max_replicas=1)
    assert SkyPilotProvisioner(k8s.backends[0], k8s, Fleet(IDENT)).volume_spec()["config"] == {"access_mode": "ReadWriteOnce"}
    k8s = sky_config(clouds=["kubernetes"], weights_cache="volume", weights_cache_infra="kubernetes", weights_cache_gb=50, max_replicas=3)
    assert SkyPilotProvisioner(k8s.backends[0], k8s, Fleet(IDENT)).volume_spec()["config"] == {"access_mode": "ReadWriteMany"}


async def test_the_volume_is_made_before_the_first_launch_counted_and_never_swept(fake_sky, replicas, monkeypatch):
    real = skypilot.sky_call
    asked: list[dict] = []

    async def sky_call(req, timeout=skypilot.SKY_TIMEOUT):
        asked.append(req)
        if req["op"] == "volume_apply":
            return {"created": len([r for r in asked if r["op"] == "volume_apply"]) == 1}
        if req["op"] == "volumes":
            return [{"name": "mo-ab12-gpu-hf", "usedby_clusters": []}]
        return await real(req, timeout)

    monkeypatch.setattr(skypilot, "sky_call", sky_call)
    cfg = sky_config(**RUNPOD)
    prov = SkyPilotProvisioner(cfg.backends[0], cfg, Fleet(IDENT))
    await prov.up(2)
    await settle(prov)
    applies = [r for r in asked if r["op"] == "volume_apply"]
    assert len(applies) == 1 and applies[0]["name"] == "mo-ab12-gpu-hf"  # once per process, before the launches
    assert fake_sky.calls("launch")[0][6]["volumes"] == {skypilot.CACHE_MOUNT: "mo-ab12-gpu-hf"}
    store, state = MemoryStore(), MemoryState()
    dep = Deployment(backend=cfg.backends[0], model=cfg.models[0], provisioner=prov)
    await controller.write_events(store, dep, {}, now=1000.0)
    assert (await store.get_setting(gpu.VOLUMES_SETTING))["mo-ab12-gpu-hf"] == {
        "deployment": "gpu", "infra": "runpod/CA/CA-MTL-1", "size_gb": 50, "usd_per_month": 3.5, "since": 1000.0}
    assert [e["kind"] for e in await store.gpu_events()] == ["volume"]
    # A later process finds it there: recorded again, not logged again.
    again = SkyPilotProvisioner(cfg.backends[0], cfg, Fleet(IDENT))
    await again._ensure_volume()
    await controller.write_events(store, Deployment(backend=cfg.backends[0], model=cfg.models[0], provisioner=again), {}, now=2000.0)
    assert len(await store.gpu_events()) == 1 and (await store.get_setting(gpu.VOLUMES_SETTING))["mo-ab12-gpu-hf"]["since"] == 1000.0
    # Its monthly price is counted in GPU spend, every hour it exists, running or not.
    assert await controller.accrue_volumes(store, state, 10_000.0) == 3.5
    assert sum((await store.gpu_spend_since(datetime(2000, 1, 1, tzinfo=UTC))).values()) == 0
    await controller.accrue_volumes(store, state, 10_000.0 + 730 * 36)
    assert sum((await store.gpu_spend_since(datetime(1970, 1, 1, tzinfo=UTC))).values()) == pytest.approx(3.5 * 730 * 36 / (730 * 3600))
    # The sweep tears down clusters no deployment wants, and never asks SkyPilot about volumes.
    asked.clear()
    prov.want = 0
    await prov.fleet.sweep(set(), store)
    assert fake_sky.names() == set() and not [r for r in asked if r["op"].startswith("volume")]


async def test_a_failed_volume_fails_the_launch(fake_sky, replicas, monkeypatch):
    async def sky_call(req, timeout=skypilot.SKY_TIMEOUT):
        if req["op"] == "volume_apply":
            raise skypilot.SkyError("RunPod DataCenterId is required")
        if req["op"] == "logs":
            return {"lines": []}
        return {}

    monkeypatch.setattr(skypilot, "sky_call", sky_call)
    cfg = sky_config(**RUNPOD)
    prov = SkyPilotProvisioner(cfg.backends[0], cfg, Fleet(IDENT))
    prov.want = 1
    prov._launch(prov.cluster(1), None)
    await settle(prov)
    assert "DataCenterId" in prov._failed[prov.cluster(1)][1] and prov._volume_ready is None


async def test_only_meritorder_gpu_deletes_a_volume(monkeypatch, capsys):
    store = MemoryStore()
    await store.set_setting("gpu_install_id", "ab12")
    await store.set_setting(gpu.VOLUMES_SETTING, {"mo-ab12-gpu-hf": {"deployment": "gpu", "infra": "runpod/CA/CA-MTL-1", "size_gb": 50,
                                                                      "usd_per_month": 3.5, "since": 1.0},
                                                   "mo-ab12-old-hf": {"deployment": "old", "infra": "kubernetes", "size_gb": 10,
                                                                      "usd_per_month": None, "since": 1.0}})
    monkeypatch.delenv(skypilot.INSTALL_ENV, raising=False)
    found = [{"name": "mo-ab12-gpu-hf", "cloud": "runpod", "usedby_clusters": ["mo-ab12-gpu-1"]}]
    deleted = []

    async def sky_call(req, timeout=skypilot.SKY_TIMEOUT):
        if req["op"] == "volumes":
            return found
        deleted.append(req["name"])
        return {}

    monkeypatch.setattr(skypilot, "sky_call", sky_call)
    listed = await skypilot.list_volumes(store)
    assert [(v["name"], v["in_skypilot"]) for v in listed] == [("mo-ab12-gpu-hf", True), ("mo-ab12-old-hf", False)]
    with pytest.raises(ValueError, match="in use by mo-ab12-gpu-1"):
        await skypilot.delete_volume(store, "mo-ab12-gpu-hf")
    with pytest.raises(ValueError, match="isn't one of this install's"):
        await skypilot.delete_volume(store, "mo-zz99-gpu-hf")
    found[0]["usedby_clusters"] = []
    await skypilot.delete_volume(store, "mo-ab12-gpu-hf")
    await skypilot.delete_volume(store, "mo-ab12-old-hf")  # gone from SkyPilot already: forgotten here too
    assert deleted == ["mo-ab12-gpu-hf"] and await store.get_setting(gpu.VOLUMES_SETTING) == {}
    assert [(e["kind"], e["backend"], e["actor"]) for e in await store.gpu_events()] == [("volume_deleted", "old", "cli"),
                                                                                         ("volume_deleted", "gpu", "cli")]


def test_the_command_lists_and_deletes_only_with_yes(monkeypatch, capsys):
    store = MemoryStore()
    asyncio.run(store.set_setting("gpu_install_id", "ab12"))
    asyncio.run(store.set_setting(gpu.VOLUMES_SETTING, {"mo-ab12-new-hf": {"deployment": "new", "infra": "runpod/CA/CA-MTL-1", "size_gb": 10,
                                                                           "usd_per_month": 0.7, "since": 1.0}}))
    monkeypatch.delenv(skypilot.INSTALL_ENV, raising=False)
    found = [{"name": "mo-ab12-new-hf", "cloud": "runpod", "size": "10", "usedby_clusters": []}]

    async def sky_call(req, timeout=skypilot.SKY_TIMEOUT):
        return found if req["op"] == "volumes" else {}

    monkeypatch.setattr(skypilot, "sky_call", sky_call)
    monkeypatch.setattr(cli, "_store", lambda: store)
    cli.cmd_gpu(SimpleNamespace(volumes_cmd="list", name=None, yes=False))
    assert "mo-ab12-new-hf  new  runpod/CA/CA-MTL-1  10 GB  about $0.70/month  used by nothing" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="--yes"):
        cli.cmd_gpu(SimpleNamespace(volumes_cmd="delete", name="mo-ab12-new-hf", yes=False))
    with pytest.raises(SystemExit, match="say which volume"):
        cli.cmd_gpu(SimpleNamespace(volumes_cmd="delete", name=None, yes=True))
    with pytest.raises(SystemExit, match="isn't one of this install's"):
        cli.cmd_gpu(SimpleNamespace(volumes_cmd="delete", name="nope", yes=True))
    cli.cmd_gpu(SimpleNamespace(volumes_cmd="delete", name="mo-ab12-new-hf", yes=True))
    assert "deleted mo-ab12-new-hf" in capsys.readouterr().out
    found.clear()
    cli.cmd_gpu(SimpleNamespace(volumes_cmd="list", name=None, yes=False))
    assert "no weights cache volumes" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli.main(["gpu", "volumes", "--help"])
    assert "--yes" in capsys.readouterr().out
