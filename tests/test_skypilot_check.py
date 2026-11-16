"""scripts/skypilot_check.py, without a cloud: it picks the cheapest GPU that holds the model on the cloud named, refuses
clouds a replica can't run on, says how to set the deployment up, and checks the cloud's own list of instances through
SkyPilot's provisioner (all against stand-ins)."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import httpx
import pytest

from meritorder import gpu, gpu_sizing

SCRIPT = Path(__file__).parent.parent / "scripts" / "skypilot_check.py"
REAL_CLIENT = httpx.Client


@pytest.fixture
def check(monkeypatch):
    spec = importlib.util.spec_from_file_location("skypilot_check", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.failed.clear()
    return mod


def offer(cloud, acc, count, price, region="us-east-1", it="x"):
    return {"cloud": cloud, "accelerator": acc, "count": count, "spot": None,
            "on_demand": {"usd_per_hour": price, "region": region, "instance_type": it}}


OFFERS = {"lambda": [offer("lambda", "H100", 1, 2.49), offer("lambda", "A10", 1, 1.29, it="gpu_1x_a10")],
          "runpod": [offer("runpod", "L4", 1, 0.43, "EU-RO-1", "1x_L4_SECURE"), offer("runpod", "A100-80GB", 1, 1.64)]}


class FakeSky:
    def __init__(self):
        self.left: dict = {}
        self.fail: Exception | None = None

    def offers(self, cloud):
        return OFFERS.get(cloud, [])

    def instances(self, cluster):
        if self.fail:
            raise self.fail
        return {"cloud": "Lambda", "instances": self.left}


def gateway(check, monkeypatch, deployment: dict, hf_id: str | None = "openai/gpt-oss-20b"):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/reports/gpu/candidates":
            return httpx.Response(200, json={"models": [{"id": "gpt-oss-20b", "hf_id": hf_id, "quality": {}}] if hf_id else []})
        if request.url.path == "/v1/reports/gpu":
            return httpx.Response(200, json={"deployments": [deployment], "controller": {"alive": True}})
        return httpx.Response(404)

    monkeypatch.setattr(httpx, "Client", lambda **kw: REAL_CLIENT(transport=httpx.MockTransport(handler), **kw))
    return check.Gateway("http://gw", "k")


@pytest.fixture
def sized(monkeypatch):
    async def size(hf_id, client):
        return {"weight_bytes": 13_000_000_000, "bits": 4, "active_params": 3.6e9}

    monkeypatch.setattr(gpu_sizing, "fetch_model_size", size)


def test_picks_the_cheapest_gpu_that_holds_the_model(check, monkeypatch, sized):
    gw = gateway(check, monkeypatch, {})
    sky = FakeSky()
    assert check.cheapest_fit(gw, sky, "gpt-oss-20b", "lambda", False) == {
        "accelerators": "A10:1", "usd_per_hour": 1.29, "instance_type": "gpu_1x_a10", "region": "us-east-1", "spot": False}
    assert check.cheapest_fit(gw, sky, "gpt-oss-20b", "runpod", False)["accelerators"] == "L4:1"
    assert "no offer" in check.cheapest_fit(gw, sky, "gpt-oss-20b", "fluidstack", False)["error"]
    gw = gateway(check, monkeypatch, {}, hf_id=None)
    assert "--accelerators" in check.cheapest_fit(gw, sky, "gpt-oss-20b", "lambda", False)["error"]


def run_main(check, monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["skypilot_check.py", "--key", "k", *args])
    monkeypatch.setattr(check, "Sky", lambda where: FakeSky())
    return check.main()


def test_main_refuses_clouds_a_replica_cant_run_on(check, monkeypatch):
    with pytest.raises(SystemExit, match="oci: SkyPilot 0.14.2 can't run a Docker image"):
        run_main(check, monkeypatch, "--cloud", "oci")
    with pytest.raises(SystemExit, match="not a cloud SkyPilot"):
        run_main(check, monkeypatch, "--cloud", "nowhere")


def test_main_says_how_to_set_the_deployment_up_and_spends_nothing_without_yes(check, monkeypatch, sized, capsys):
    dep = {"backend": "gpu-small", "provisioner": "skypilot", "enabled": True, "model": "gpt-oss-20b",
           "spec": {"accelerators": "A10:1", "clouds": ["lambda"], "use_spot": False, "lease_minutes": 15, "autostop_minutes": 5}}
    gateway(check, monkeypatch, dep)
    with pytest.raises(SystemExit, match='accelerators: "L4:1"\n  clouds: \\[runpod\\]\nand restart the stack \\(one replica: about \\$0.43 an hour'):
        run_main(check, monkeypatch, "--cloud", "RunPod")
    monkeypatch.setenv("HOME", str(Path(os.devnull).parent))  # no credential files: no notes
    assert run_main(check, monkeypatch, "--cloud", "lambda") == 2
    out = capsys.readouterr().out
    assert "The cheapest GPU on lambda that holds gpt-oss-20b: A10:1 (gpu_1x_a10, us-east-1) at $1.29/h" in out
    assert "about $0.97 to $1.61" in out and "Nothing started" in out
    gateway(check, monkeypatch, {**dep, "enabled": False})
    with pytest.raises(SystemExit, match="isn't an enabled SkyPilot deployment"):
        run_main(check, monkeypatch, "--cloud", "lambda", "--accelerators", "A10:1")


def test_checks_the_clouds_own_list(check, capsys):
    sky = FakeSky()
    check.gone_from_cloud(sky, "mo-ab12-gpu-small-1")
    assert check.failed == [] and "no instance of mo-ab12-gpu-small-1 left in Lambda's own list" in capsys.readouterr().out
    sky.left = {"inst-1": "UP"}
    check.gone_from_cloud(sky, "mo-ab12-gpu-small-1")
    assert check.failed and "inst-1 (UP)" in capsys.readouterr().out
    check.failed.clear()
    sky.fail = RuntimeError("SkyPilot here has no record")
    check.gone_from_cloud(sky, "mo-ab12-gpu-small-1")
    check.gone_from_cloud(sky, None)
    assert check.failed == [] and "can't be checked from here" in capsys.readouterr().out


def test_sky_runs_scripts_in_skypilots_python_and_reads_the_answer(check, monkeypatch):
    seen = []

    def run(cmd, **kw):
        seen.append(cmd)
        req = json.loads(cmd[-1])
        if req.get("op") == "offers":
            return subprocess.CompletedProcess(cmd, 0, stdout='noise\n{"offers": [1]}\n', stderr="")
        if req.get("op") == "boom":
            return subprocess.CompletedProcess(cmd, 1, stdout='MERITORDER_RESULT {"error": "no record"}\n', stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout='log line\nMERITORDER_RESULT {"cloud": "Lambda"}\n', stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(gpu, "sky_python", lambda: "/opt/skypilot/bin/python")
    local, compose = check.Sky("local"), check.Sky("compose")
    assert local.offers("lambda") == [1] and seen[-1][0] == "/opt/skypilot/bin/python"
    assert compose.remember("mo-ab12-x-1") == {"cloud": "Lambda"} and seen[-1][:5] == ["docker", "compose", "exec", "-T", "controller"]
    compose.controller_up = False
    compose.instances("mo-ab12-x-1")
    assert seen[-1][:4] == ["docker", "compose", "run", "--rm"]
    with pytest.raises(RuntimeError, match="no record"):
        compose.run(check.CLOUD_SCRIPT, {"op": "boom"})


def test_cloud_script_remembers_and_asks_the_cloud(check, tmp_path, monkeypatch):
    """The script that runs in SkyPilot's Python, against stand-ins for its state and provisioner."""
    monkeypatch.setenv("HOME", str(tmp_path))
    handle = types.SimpleNamespace(launched_resources=types.SimpleNamespace(cloud="Lambda"), cluster_name_on_cloud="mo-ab12-x-1-2ea4",
                                   cluster_yaml="/x.yml")
    state = types.ModuleType("sky.global_user_state")
    state.get_handle_from_cluster_name = lambda name: handle if name == "mo-ab12-x-1" else None
    state.get_cluster_yaml_dict = lambda path: {"provider": {"region": "us-east-1"}}
    provision = types.ModuleType("sky.provision")
    asked = []

    def query_instances(cloud, cluster, on_cloud, provider, non_terminated_only):
        asked.append((cloud, cluster, on_cloud, provider, non_terminated_only))
        return {"i-1": (types.SimpleNamespace(value="UP"), None)}

    provision.query_instances = query_instances
    sky = types.ModuleType("sky")
    sky.global_user_state, sky.provision = state, provision
    monkeypatch.setitem(sys.modules, "sky", sky)
    monkeypatch.setitem(sys.modules, "sky.global_user_state", state)
    monkeypatch.setitem(sys.modules, "sky.provision", provision)
    ns: dict = {"__name__": "cloud_script"}
    exec(check.CLOUD_SCRIPT, ns)
    assert ns["remember"]({"cluster": "mo-ab12-x-1"}) == {"cloud": "'Lambda'", "on_cloud": "mo-ab12-x-1-2ea4"}
    ref = tmp_path / ".sky" / "meritorder-check" / "mo-ab12-x-1.json"
    assert oct(ref.stat().st_mode & 0o777) == "0o600"
    assert ns["instances"]({"cluster": "mo-ab12-x-1"}) == {"cloud": "'Lambda'", "instances": {"i-1": "UP"}}
    assert asked == [("'Lambda'", "mo-ab12-x-1", "mo-ab12-x-1-2ea4", {"region": "us-east-1"}, True)]
    with pytest.raises(ValueError, match="no record"):
        ns["remember"]({"cluster": "mo-ab12-y-1"})
    ns["forget"]({"cluster": "mo-ab12-x-1"})
    ns["forget"]({"cluster": "mo-ab12-x-1"})
    assert not ref.exists()


def test_a_failed_start_prints_why_each_replica_failed(check, monkeypatch, capsys):
    events = [{"id": 3, "ts": "2026-11-15T10:06:00+00:00", "kind": "state", "backend": "gpu-small", "detail": {}},
              {"id": 2, "ts": "2026-11-15T10:05:00+00:00", "kind": "replica_failed", "backend": "gpu-small", "detail": {
                  "cluster": "mo-ab12-gpu-small-1", "reason": "vLLM exited with status 1 before it was ready: the NVIDIA driver is too old",
                  "driver_version": "535.129.03", "cuda_version": "12.2", "image_cuda": "13.0.2", "gpus": ["NVIDIA A10"],
                  "source": "gate", "tail": ["RuntimeError: driver too old"]}},
              {"id": 1, "ts": "2026-11-15T09:00:00+00:00", "kind": "replica_failed", "backend": "gpu-small", "detail": {"cluster": "old"}}]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/admin/gpu/events" and request.url.params["backend"] == "gpu-small":
            return httpx.Response(200, json={"events": events})
        return httpx.Response(500)

    monkeypatch.setattr(httpx, "Client", lambda **kw: REAL_CLIENT(transport=httpx.MockTransport(handler), **kw))
    gw = check.Gateway("http://gw", "k")
    since = check.datetime.fromisoformat("2026-11-15T10:00:00+00:00").timestamp()
    check.show_failures(gw, "gpu-small", since)
    out = capsys.readouterr().out
    assert "mo-ab12-gpu-small-1: vLLM exited with status 1 before it was ready: the NVIDIA driver is too old" in out and "old:" not in out
    assert "NVIDIA driver 535.129.03, CUDA 12.2 at most; image CUDA 13.0.2; GPUs NVIDIA A10" in out
    assert "its last 1 lines (gate):\n    | RuntimeError: driver too old" in out
    check.show_failures(gw, "gpu-small", since + 3600)
    assert "no failed replica" in capsys.readouterr().out
    events[1]["detail"] = {"cluster": "c", "reason": "r", "tail": [], "logs_error": "its gate is gone"}
    check.show_failures(gw, "gpu-small", since)
    assert "no output: its gate is gone" in capsys.readouterr().out
    check.show_failures(gw, "other", since)  # a 500
    assert "can't be read" in capsys.readouterr().out
    done = check.warm_or_given_up(since)
    assert not done({"state": "api_only"})  # not started yet
    assert not done({"state": "provisioning"}) and done({"state": "api_only", "reason": "up failed"})
    assert check.warm_or_given_up(since)({"state": "api_only", "cooldown_until": since + 1800})  # failed within one poll
    assert not check.warm_or_given_up(since)({"state": "api_only", "cooldown_until": since - 1})  # an earlier failure
    assert not check.warm_or_given_up(since)({"state": "api_only", "cooldown_until": since + 600})  # an earlier run's, still cooling down
    assert check.warm_or_given_up(since)({"state": "warm", "replicas_info": [{"status": "READY"}]})


def test_a_failed_start_stops_the_check_at_once_and_unpins(check, monkeypatch, sized, capsys):
    import time

    dep = {"backend": "gpu-small", "provisioner": "skypilot", "enabled": True, "model": "gpt-oss-20b", "state": "api_only",
           "spec": {"accelerators": "A10:1", "clouds": ["lambda"], "use_spot": False, "lease_minutes": 15, "autostop_minutes": 5}}
    controls: list[dict] = []
    failure = {"id": 1, "ts": "2999-01-01T00:00:00+00:00", "kind": "replica_failed", "backend": "gpu-small",
               "detail": {"cluster": "mo-ab12-gpu-small-1", "reason": "vLLM exited with status 1 before it was ready", "source": "gate",
                          "tail": ["RuntimeError: driver too old"]}}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/reports/gpu/candidates":
            return httpx.Response(200, json={"models": [{"id": "gpt-oss-20b", "hf_id": "openai/gpt-oss-20b", "quality": {}}]})
        if request.url.path == "/v1/reports/gpu":
            # The controller tried, failed within one poll and cooled down: back to API only.
            now = {**dep, "cooldown_until": time.time() + 1800, "reason": "provisioning failed (replica 1: vLLM exited)"} if controls else dep
            return httpx.Response(200, json={"deployments": [now], "controller": {"alive": True}})
        if request.url.path.endswith("/control"):
            controls.append(json.loads(request.content))
            return httpx.Response(200, json={})
        if request.url.path == "/v1/admin/gpu/events":
            return httpx.Response(200, json={"events": [failure]})
        return httpx.Response(404)

    monkeypatch.setattr(httpx, "Client", lambda **kw: REAL_CLIENT(transport=httpx.MockTransport(handler), **kw))
    assert run_main(check, monkeypatch, "--cloud", "lambda", "--yes-spend") == 1
    out = capsys.readouterr().out
    assert "FAIL the check stopped early: RuntimeError: provisioning failed: provisioning failed (replica 1: vLLM exited)" in out
    assert "    | RuntimeError: driver too old" in out and "--- 2." not in out
    assert [c["mode"] for c in controls] == ["on", "off"]  # the pin is gone in the finally


def test_prints_the_start_phase_by_phase_at_warm(check, capsys):
    check.show_startup({"startup": {"phases": [{"phase": "image pull", "seconds": 300}, {"phase": "weights download + weights load",
                                                                                         "seconds": 150}],
                                    "total_seconds": 660, "download_seconds": 95.2, "load_seconds": 21.4}})
    out = capsys.readouterr().out
    assert "11.0 min from the launch request to ready" in out and "image pull" in out and "  5.0 min" in out
    assert "vLLM says the weights downloaded in 95s and loaded in 21s" in out
    check.show_startup({"startup": {"phases": [{"phase": "GPU check", "seconds": 2}], "total_seconds": None}})
    check.show_startup({})
    out = capsys.readouterr().out
    assert "start:" in out and "not known" in out
