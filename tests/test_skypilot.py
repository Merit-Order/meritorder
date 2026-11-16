"""SkyPilot as plain clusters: the replica gate and its lease, keys, cluster names, the task, the SDK calls (against a
stand-in SkyPilot package), the provisioner's reconcile loop, the sweep, and the controller's handling of failed starts."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import socket
import subprocess
import sys
import time

import httpx
import pytest
import yaml

from meritorder import controller, gpu, replica_gate, skypilot
from meritorder.backends import build_backends
from meritorder.backends.auth import ReplicaKeyAuth
from meritorder.config import BackendConfig, Config, ModelConfig, Route, ServeConfig
from meritorder.gpu import API_ONLY, PROVISIONING, WARM, Deployment
from meritorder.skypilot import Fleet, Identity, SkyError, SkyPilotProvisioner
from meritorder.state import MemoryState
from meritorder.store import MemoryStore

REAL_ASYNC_CLIENT = httpx.AsyncClient
IDENT = Identity("ab12", "test-secret")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def sky_config(name: str = "gpu", **serve) -> Config:
    model = ModelConfig(name="ow", quality={"coding": 4}, hf_id="o/m", routes=[Route(backend=name, model="m")])
    backend = BackendConfig(name=name, kind="self_hosted", format="openai", gpu_cost_per_hour=2.0,
                            serve=ServeConfig(model="ow", provisioner="skypilot", accelerators="A10:1", **serve))
    return Config(backends=[backend], models=[model])


# ------------------------------------------------------------------ the gate


UPSTREAM = r'''
import http.server, sys
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = ("vllm saw " + self.path + " auth=" + str(self.headers.get("authorization"))).encode()
        self.send_response(200); self.send_header("content-length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def log_message(self, *a): pass
http.server.HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
'''


async def wait_for_port(port: int, timeout: float = 10.0) -> None:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        try:
            _, w = await asyncio.open_connection("127.0.0.1", port)
            w.close()
            return
        except OSError:
            await asyncio.sleep(0.05)
    raise TimeoutError(port)


async def test_gate_lets_in_only_the_key_proxies_and_holds_a_lease():
    key = IDENT.api_key("gpu")
    port, upstream = free_port(), free_port()
    gate = replica_gate.Gate([sys.executable, "-c", UPSTREAM, str(upstream)], IDENT.key_sha256("gpu"), 30, port, upstream, grace_seconds=2,
                             host="127.0.0.1")
    run = asyncio.ensure_future(gate.run())
    await wait_for_port(port)
    async with REAL_ASYNC_CLIENT(base_url=f"http://127.0.0.1:{port}", timeout=10) as c:
        assert (await c.get("/v1/models")).status_code == 401  # no key
        assert (await c.get("/metrics", headers={"authorization": "Bearer nope"})).status_code == 401  # every path, not just /v1
        r = await c.post(replica_gate.LEASE_PATH, headers={"authorization": f"Bearer {key}"})
        assert r.status_code == 200 and r.json()["expires_in"] == 30 and gate.renewed == 1
        await wait_for_port(upstream)
        r = await c.get("/v1/models", headers={"authorization": f"Bearer {key}"})
        assert r.status_code == 200 and r.text == f"vllm saw /v1/models auth=Bearer {key}"  # passed through untouched
        r = await c.get("/metrics", headers={"x-api-key": key})
        assert r.status_code == 200 and r.text.startswith("vllm saw /metrics")
        lease = (await c.get(replica_gate.LEASE_PATH, headers={"x-api-key": key})).json()
        assert lease["vllm"] == "running" and gate.renewed == 1  # a GET reads it without renewing
    gate.expires = time.monotonic()  # the lease runs out: vLLM is stopped and the gate exits, which ends the task
    assert await asyncio.wait_for(run, 10) == 0
    assert gate.proc.returncode is not None and gate.stopping.startswith("lease not renewed")


CRASHING_VLLM = r'''
import sys, time
print("INFO loading model token=hf_abcdefghijklmnopqrstuvwxyz", flush=True)
print("Loading shards:  50%\rLoading shards: 100%", flush=True)
print("INFO Received request chatcmpl-1: prompt: 'my secret prompt'", flush=True)
time.sleep(1.5)
print("RuntimeError: The NVIDIA driver on your system is too old (found version 12020)", file=sys.stderr, flush=True)
raise SystemExit(3)
'''
OLD_DRIVER = {"driver_version": "535.129.03", "cuda_version": "12.2", "image_cuda": "13.0.2", "gpus": ["NVIDIA A10"],
              "nvidia_smi_error": None, "driver_problem": replica_gate.driver_problem("535.129.03", "12.2", "13.0.2")}


async def test_gate_keeps_vllms_last_output_until_the_controller_has_read_it(capsys):
    port, key = free_port(), IDENT.api_key("gpu")
    gate = replica_gate.Gate([sys.executable, "-c", CRASHING_VLLM], IDENT.key_sha256("gpu"), 60, port, free_port(), host="127.0.0.1",
                             linger_seconds=30, gpu=lambda: OLD_DRIVER)
    run = asyncio.ensure_future(gate.run())
    await wait_for_port(port)
    async with REAL_ASYNC_CLIENT(base_url=f"http://127.0.0.1:{port}", timeout=10) as c:
        r = await c.get("/v1/models", headers={"authorization": f"Bearer {key}"})
        assert r.status_code == 503 and "isn't listening" in r.text
        assert (await c.get(replica_gate.STATUS_PATH)).status_code == 401  # the deployment's key, as for the lease
        st = (await c.get(replica_gate.STATUS_PATH, headers={"x-api-key": key})).json()
        assert st["vllm"] == "running" and st["exit_code"] is None and st["uptime_seconds"] >= 0 and not gate.read_after_exit
        for _ in range(100):  # vLLM crashes; the gate stays to say why
            lease = (await c.get(replica_gate.LEASE_PATH, headers={"x-api-key": key})).json()
            if lease["vllm"] == "exited":
                break
            await asyncio.sleep(0.1)
        assert lease["exit_code"] == 3 and not run.done()
        st = (await c.get(replica_gate.STATUS_PATH, headers={"x-api-key": key})).json()
    assert await asyncio.wait_for(run, 10) == 3  # gone soon after it was read: the task fails with vLLM's status
    assert st["vllm"] == "exited" and st["exit_code"] == 3 and st["driver_version"] == "535.129.03" and st["cuda_version"] == "12.2"
    assert "CUDA 13 images need driver 580.65.06 or newer" in st["driver_problem"] and st["gpus"] == ["NVIDIA A10"]
    assert st["tail"] == ["INFO loading model token=[redacted]", "Loading shards: 100%", "[a request log line, not kept]",
                          "RuntimeError: The NVIDIA driver on your system is too old (found version 12020)"]
    assert "my secret prompt" not in json.dumps(st)
    assert "too old for this image's CUDA 13.0.2" in capsys.readouterr().out  # said at start, and vLLM's output passes through


async def test_gate_lingers_only_so_long_without_a_reader():
    port = free_port()
    gate = replica_gate.Gate([sys.executable, "-c", "raise SystemExit(0)"], IDENT.key_sha256("gpu"), 60, port, free_port(),
                             host="127.0.0.1", linger_seconds=0.5, gpu=lambda: {})
    t0 = time.monotonic()
    assert await asyncio.wait_for(gate.run(), 10) == 0 and time.monotonic() - t0 < 5
    assert gate.status()["vllm"] == "exited" and gate.status()["tail"] == []


def test_gate_reads_the_driver_and_the_images_cuda(tmp_path, monkeypatch):
    smi = ("| NVIDIA-SMI 535.129.03   Driver Version: 535.129.03   CUDA Version: 12.2     |\n")

    def run(cmd, **kw):
        out = "NVIDIA A10\n" if "--query-gpu=name" in cmd else smi
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")

    monkeypatch.setenv("CUDA_VERSION", "13.0.2")
    info = replica_gate.gpu_info(run)
    assert (info["driver_version"], info["cuda_version"], info["image_cuda"], info["gpus"]) == ("535.129.03", "12.2", "13.0.2", ["NVIDIA A10"])
    assert "too old" in info["driver_problem"] and "v0.32.0-cu129" in info["driver_problem"]
    monkeypatch.delenv("CUDA_VERSION")
    monkeypatch.setenv("CUDA_HOME", str(tmp_path))
    assert replica_gate.image_cuda() is None
    (tmp_path / "version.json").write_text(json.dumps({"cuda": {"name": "CUDA SDK", "version": "12.9.1"}}))
    info = replica_gate.gpu_info(run)
    assert info["image_cuda"] == "12.9.1" and info["driver_problem"] is None  # 12.9 runs on a 12.2 driver (minor version compatibility)
    failing = replica_gate.gpu_info(lambda cmd, **kw: subprocess.CompletedProcess(cmd, 9, stdout="", stderr="NVML: Driver/library mismatch"))
    assert failing["nvidia_smi_error"] == "NVML: Driver/library mismatch" and failing["driver_version"] is None and failing["gpus"] == []

    def missing(cmd, **kw):
        raise FileNotFoundError("nvidia-smi")

    assert replica_gate.gpu_info(missing)["nvidia_smi_error"].startswith("FileNotFoundError")
    assert replica_gate.driver_problem("580.65.06", "13.0", "13.0.2") is None and replica_gate.driver_problem(None, None, "13.0") is None
    assert "need driver" not in replica_gate.driver_problem("x", "13.0", "14.1")  # no known floor for CUDA 14 yet


def test_gate_uses_nvidia_forward_compatibility_when_the_driver_is_older(monkeypatch, tmp_path):
    """Lambda's driver 570 reports CUDA 12.8; vLLM's 12.9.1 image failed there to load torchcodec until the image's
    forward-compatibility libraries came first on LD_LIBRARY_PATH."""
    assert replica_gate.older_driver("12.8", "12.9.1") and replica_gate.older_driver("12.8", "13.0.2")
    assert not replica_gate.older_driver("12.9", "12.9.1") and not replica_gate.older_driver(None, "12.9") and not replica_gate.older_driver("13.0", "12.9")
    monkeypatch.setenv("CUDA_HOME", str(tmp_path))
    monkeypatch.setenv("CUDA_VERSION", "12.9.1")
    smi = "| NVIDIA-SMI 570.148.08   Driver Version: 570.148.08   CUDA Version: 12.8 |\n"

    def run(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 0, stdout="NVIDIA A10\n" if "--query-gpu=name" in cmd else smi, stderr="")

    assert replica_gate.compat_dir() is None and replica_gate.gpu_info(run)["cuda_compat"] is None  # not in this image
    (tmp_path / "compat").mkdir()
    (tmp_path / "compat" / "libcuda.so.575.57.08").write_text("")
    info = replica_gate.gpu_info(run)
    assert info["cuda_compat"] == str(tmp_path / "compat") and info["driver_problem"] is None
    monkeypatch.setenv("CUDA_VERSION", "12.8.0")
    assert replica_gate.gpu_info(run)["cuda_compat"] is None  # the driver already runs it


async def test_gate_starts_vllm_with_the_compat_libraries_first(monkeypatch, tmp_path):
    seen = {}

    async def fake_exec(*cmd, **kw):
        seen.update(kw)
        raise RuntimeError("stop here")

    monkeypatch.setattr(replica_gate.asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/local/nvidia/lib64")
    gate = replica_gate.Gate(["vllm", "serve", "m"], "k" * 64, 900, free_port(), free_port(), 30,
                             gpu=lambda: {"driver_version": "570.148.08", "cuda_version": "12.8", "image_cuda": "12.9.1",
                                          "cuda_compat": "/usr/local/cuda/compat", "driver_problem": None})
    with pytest.raises(RuntimeError, match="stop here"):
        await gate.run()
    assert seen["env"]["LD_LIBRARY_PATH"] == "/usr/local/cuda/compat:/usr/local/nvidia/lib64"
    assert seen["env"]["VLLM_ENABLE_CUDA_COMPATIBILITY"] == "1"


def test_gate_redacts_what_looks_like_a_secret():
    k = replica_gate.keepable
    assert k("Authorization: Bearer abc.def") == "Authorization: Bearer [redacted]"
    assert k('{"api_key": "xyz", "aws_secret_access_key": "a/b"}') == '{"api_key": "[redacted]", "aws_secret_access_key": "[redacted]"}'
    assert k("key " + IDENT.api_key("gpu")) == "key [redacted]" and k("AKIAABCDEFGHIJKLMNOP sk-proj-abcdefghij") == "[redacted] [redacted]"
    for line in ("max_num_batched_tokens=8192, max_model_len: 4096", "cluster mo-ab12-gpu-small-1",
                 'File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core_client.py", line 12'):
        assert k(line) == line  # vLLM's own config, cluster names and tracebacks stay readable
    assert k("x" * 5000) == "x" * replica_gate.LINE_MAX and k("Generated response chatcmpl-1: output: 'hi'") == "[a request log line, not kept]"


async def test_gate_lease_expiry_stops_a_stubborn_vllm():
    port = free_port()
    stubborn = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"
    gate = replica_gate.Gate([sys.executable, "-c", stubborn], IDENT.key_sha256("gpu"), 0.5, port, free_port(), grace_seconds=0.5,
                             host="127.0.0.1")
    t0 = time.monotonic()
    await asyncio.wait_for(gate.run(), 15)
    assert gate.proc.returncode is not None and time.monotonic() - t0 < 10  # killed after the grace period


def test_gate_parsing_and_usage(capsys):
    head = b"GET / HTTP/1.1\r\nHost: x\r\nAuthorization: bearer abc\r\n\r\n"
    assert replica_gate.token_of(head) == "abc" and replica_gate.token_of(b"GET / HTTP/1.1\r\nx-api-key: k\r\n\r\n") == "k"
    assert replica_gate.token_of(b"GET / HTTP/1.1\r\nAuthorization: Basic x\r\n\r\n") is None
    assert replica_gate.allowed(head, hashlib.sha256(b"abc").hexdigest()) and not replica_gate.allowed(head, "0" * 64)
    assert replica_gate.main(["--"]) == 2 and "usage" in capsys.readouterr().err


def test_task_writes_the_gate_and_runs_vllm_behind_it(tmp_path):
    cfg = sky_config(vllm_args="--max-model-len 8192")
    prov = SkyPilotProvisioner(cfg.backends[0], cfg, Fleet(IDENT))
    task = prov.task_config(prov.cluster(1))
    target = tmp_path / "gate.py"
    script = task["run"].replace("/tmp/meritorder_gate.py", str(target)).split("exec python3")[0]
    subprocess.run(["bash", "-c", script], check=True)
    assert target.read_text() == (gpu.Path(replica_gate.__file__)).read_text()  # the heredoc carries the gate exactly
    assert "exec python3 /tmp/meritorder_gate.py -- vllm serve o/m --host 127.0.0.1 --port 8001" in task["run"]
    assert task["run"].rstrip().endswith("--max-model-len 8192")


# ------------------------------------------------------------------ keys, names, the task


def test_keys_are_per_deployment_and_only_their_hash_reaches_the_machine(monkeypatch):
    other = Identity("cd34", "another-secret")
    assert IDENT.api_key("gpu") == IDENT.api_key("gpu") != IDENT.api_key("gpu-2") != other.api_key("gpu")
    assert IDENT.key_sha256("gpu") == hashlib.sha256(IDENT.api_key("gpu").encode()).hexdigest()
    cfg = sky_config()
    task = SkyPilotProvisioner(cfg.backends[0], cfg, Fleet(IDENT)).task_config("mo-ab12-gpu-1")
    dumped = yaml.safe_dump(task)
    assert IDENT.api_key("gpu") not in dumped and IDENT.secret not in dumped and IDENT.key_sha256("gpu") in dumped
    monkeypatch.setenv(skypilot.INSTALL_ENV, "Not Valid")
    with pytest.raises(ValueError, match="lowercase"):
        Identity.from_env()
    monkeypatch.setenv(skypilot.INSTALL_ENV, "x9")
    monkeypatch.setenv(skypilot.SECRET_ENV, "s")
    assert Identity.from_env() == Identity("x9", "s")


async def test_identity_comes_from_the_store_once(monkeypatch):
    monkeypatch.delenv(skypilot.INSTALL_ENV, raising=False)
    monkeypatch.delenv(skypilot.SECRET_ENV, raising=False)
    store = MemoryStore()
    a, b = await skypilot.load_identity(store), await skypilot.load_identity(store)
    assert a == b and len(a.install_id) == 8 and a.secret  # made once; every process reads the same
    monkeypatch.setenv(skypilot.SECRET_ENV, "from-env")
    assert (await skypilot.load_identity(store)).secret == "from-env"
    await store.set_setting(skypilot.INSTALL_SETTING, "BAD ID")
    with pytest.raises(ValueError):
        await skypilot.load_identity(store)
    monkeypatch.setattr(skypilot, "_known", None)
    assert await skypilot.ensure_identity(store) is None  # logged; requests to replicas fail until it loads
    await store.set_setting(skypilot.INSTALL_SETTING, "ok1")
    found = await skypilot.ensure_identity(store)
    assert found.install_id == "ok1" and skypilot.known_identity() is found


def test_cluster_names_never_collide():
    assert skypilot.slug("gpu-qwen") == "gpu-qwen"
    assert skypilot.slug("gpu_qwen") != skypilot.slug("gpu.qwen") and skypilot.slug("gpu_qwen").startswith("gpu-qwen-")
    cfg = sky_config("gpu-qwen")
    prov = SkyPilotProvisioner(cfg.backends[0], cfg, Fleet(IDENT))
    assert prov.cluster(3) == "mo-ab12-gpu-qwen-3" and prov.index_of("mo-ab12-gpu-qwen-3") == 3
    assert prov.index_of("mo-ab12-gpu-qwen-2-1") is None and prov.index_of("mo-zz99-gpu-qwen-3") is None  # another deployment, install
    assert skypilot.deployment_of("mo-ab12-gpu-qwen-3", IDENT) == "gpu-qwen" and skypilot.deployment_of("mo-zz99-x-1", IDENT) is None
    assert len(SkyPilotProvisioner(cfg.backends[0], cfg, Fleet(IDENT)).cluster(10)) <= 57  # Lambda's limit


async def test_task_pairs_regions_only_with_their_own_cloud(monkeypatch):
    cfg = sky_config(clouds=["aws", "gcp", "lambda"], regions=["us-east-1"], kv_events=True, kv_events_port=8001)
    calls = []

    def placements(acc, clouds, regions, spot):
        calls.append((acc, clouds, regions, spot))
        return {"best": {"usd_per_hour": 1.29, "cloud": "Lambda"}, "pairs": [["aws", "us-east-1"], ["lambda", "us-east-1"]]}

    monkeypatch.setattr(gpu, "catalog_placements", placements)
    prov = SkyPilotProvisioner(cfg.backends[0], cfg, Fleet(IDENT))
    task = prov.task_config(prov.cluster(1), await prov._pairs())
    res = task["resources"]
    ports = [8000, 8001]
    assert res["any_of"] == [{"infra": "aws/us-east-1", "ports": ports}, {"infra": "lambda/us-east-1", "ports": ports}]  # no gcp/us-east-1
    assert await prov.expected_price() == {"usd_per_hour": 1.29, "cloud": "Lambda"} and len(calls) == 1  # one lookup for both
    assert res["autostop"] == {"idle_minutes": 5, "down": True, "wait_for": "jobs"} and "ports" not in res  # with each cloud
    assert task["envs"]["MERITORDER_UPSTREAM_PORT"] == "8002" and task["envs"]["MERITORDER_LEASE_SECONDS"] == "900"  # clear of KV events
    assert res["image_id"] == "docker:vllm/vllm-openai:v0.32.0-cu129" and res["labels"][skypilot.INSTALL_LABEL] == "ab12"
    monkeypatch.setattr(gpu, "catalog_placements", lambda *a: None)
    prov._expected = None
    with pytest.raises(SkyError, match="which cloud each region belongs to"):
        await prov._pairs()
    anywhere = SkyPilotProvisioner(sky_config().backends[0], sky_config(), Fleet(IDENT)).task_config("x")["resources"]
    assert "any_of" not in anywhere and anywhere["ports"] == [8000]


def test_task_asks_for_ports_only_where_skypilot_can_open_them():
    cfg = sky_config(clouds=["aws", "vast", "mithril", "RunPod"])
    res = SkyPilotProvisioner(cfg.backends[0], cfg, Fleet(IDENT)).task_config("x")["resources"]
    # Ports on Vast or Mithril would rule them out: their replicas are reached over SSH.
    assert res["any_of"] == [{"infra": "aws", "ports": [8000]}, {"infra": "vast"}, {"infra": "mithril"}, {"infra": "runpod", "ports": [8000]}]
    assert res["autostop"]["down"] is True and res["image_id"].startswith("docker:")


def test_config_rules():
    with pytest.raises(ValueError, match="replica_affinity"):
        ServeConfig(model="ow", provisioner="skypilot", replica_affinity=False)
    with pytest.raises(ValueError, match="oci: SkyPilot 0.14.2 can't run a Docker image.*hyperbolic: .*delete itself"):
        ServeConfig(model="ow", clouds=["aws", "oci", "hyperbolic"])
    with pytest.raises(ValueError, match="clouds: \\[kubernetes\\]"):
        ServeConfig(model="ow", clouds=["coreweave"])
    with pytest.raises(ValueError, match="opens no ports on vast"):
        ServeConfig(model="ow", clouds=["vast"], ssh_tunnel="never")
    assert ServeConfig(model="ow", provisioner="static", clouds=["oci"]).clouds == ["oci"]  # only SkyPilot launches there
    assert ServeConfig(model="ow", clouds=["vast", "kubernetes", "do"]).clouds
    with pytest.raises(ValueError, match=r"request logging \(--enable-log-requests\)"):
        ServeConfig(model="ow", vllm_args="--max-model-len 8192 --enable-log-requests")  # prompts would reach the logs
    assert "--enable-log-requests" not in " ".join(gpu.vllm_command(sky_config().backends[0], sky_config()))
    with pytest.raises(ValueError, match="must differ"):
        ServeConfig(model="ow", kv_events=True, kv_events_port=8000)
    with pytest.raises(ValueError):
        ServeConfig(model="ow", lease_minutes=1)


# ------------------------------------------------------------------ a stand-in SkyPilot


FAKE_SKY = r'''
import functools, json, os, threading

_lock = threading.RLock()  # the provisioner calls it from several threads at once


def _locked(fn):
    @functools.wraps(fn)
    def run(*a, **kw):
        with _lock:
            return fn(*a, **kw)
    return run


def _load():
    try:
        with open(os.environ["FAKE_SKY_STATE"]) as f:
            return json.load(f)
    except FileNotFoundError:
        return {"clusters": {}, "calls": []}


def _save(st):
    tmp = os.environ["FAKE_SKY_STATE"] + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f)
    os.replace(tmp, os.environ["FAKE_SKY_STATE"])


class StatusRefreshMode:
    NONE, AUTO = "none", "auto"


class _Status:
    def __init__(self, v):
        self.value = v


class _Res:
    def __init__(self, d):
        self.cloud, self.region, self.zone = d["cloud"], d["region"], d.get("zone")
        self.instance_type, self.use_spot = d["instance_type"], d.get("spot", False)


class _Handle:
    def __init__(self, d):
        self.head_ip = d.get("head_ip")
        self.launched_resources = _Res(d)


class Task:
    @staticmethod
    def from_yaml_config(cfg):
        t = Task()
        t.cfg, t.secrets = cfg, {}
        return t

    def update_secrets(self, s):
        self.secrets.update(s)
        return self


def get(x):
    return x


@_locked
def status(refresh=None, cluster_names=None):
    st = _load()
    st["calls"].append(["status", refresh])
    _save(st)
    if os.environ.get("FAKE_SKY_FAIL") == "status":
        raise RuntimeError("API server unhappy")
    return [{"name": n, "status": _Status(c["status"]), "handle": _Handle(c), "launched_at": 1700000000, "autostop": c.get("autostop", 5),
             "to_down": c.get("to_down", True), "resources_str": "1x(gpus=A10:1, gpu_1x_a10)"} for n, c in st["clusters"].items()]


@_locked
def launch(task, cluster_name, idle_minutes_to_autostop, down, wait_for, retry_until_up):
    st = _load()
    st["calls"].append(["launch", cluster_name, idle_minutes_to_autostop, down, str(wait_for), sorted(task.secrets), task.cfg])
    if os.environ.get("FAKE_SKY_FAIL") == "launch":
        st["clusters"][cluster_name] = {"status": "INIT", "cloud": "lambda", "region": "us-east-1", "instance_type": "gpu_1x_a10"}
        _save(st)
        raise RuntimeError("no capacity anywhere")
    st["clusters"][cluster_name] = {"status": "UP", "cloud": os.environ.get("FAKE_SKY_CLOUD", "lambda"),
                                    "region": os.environ.get("FAKE_SKY_REGION", "us-east-1"), "instance_type": "gpu_1x_a10",
                                    "head_ip": os.environ.get("FAKE_SKY_IP", "10.0.0.1")}
    _save(st)
    return 1, None


@_locked
def endpoints(cluster, port=None):
    st = _load()
    st["calls"].append(["endpoints", cluster, port])
    _save(st)
    if os.environ.get("FAKE_SKY_FAIL") == "endpoints":
        raise ValueError("Port 8000 not exposed yet")
    found = st["clusters"][cluster].get("endpoint")
    return {port: found} if found else {}


@_locked
def autostop(cluster_name, idle_minutes, wait_for=None, down=False):
    st = _load()
    st["calls"].append(["autostop", cluster_name, idle_minutes, str(wait_for), down])
    if os.environ.get("FAKE_SKY_FAIL") == "autostop":
        _save(st)
        raise RuntimeError("skylet unreachable")
    st["clusters"][cluster_name].update(autostop=idle_minutes, to_down=down)
    _save(st)


@_locked
def tail_logs(cluster_name, job_id, follow, tail=0, output_stream=None):
    st = _load()
    st["calls"].append(["tail_logs", cluster_name, job_id, follow, tail])
    _save(st)
    if os.environ.get("FAKE_SKY_FAIL") == "logs":
        from sky import exceptions
        raise exceptions.ClusterDoesNotExist(cluster_name)
    output_stream.write("(task, pid=1) INFO starting\n(task, pid=1) RuntimeError: no kernel image token=abc123\n(task, pid=1) bye\n")
    return 100


@_locked
def down(name):
    st = _load()
    st["calls"].append(["down", name])
    if os.environ.get("FAKE_SKY_FAIL") == "down":
        _save(st)
        raise RuntimeError("cloud API timed out")
    if name not in st["clusters"]:
        _save(st)
        from sky import exceptions
        raise exceptions.ClusterDoesNotExist(name)
    del st["clusters"][name]
    _save(st)
'''


class FakeSky:
    def __init__(self, path):
        self.path = path

    def state(self) -> dict:
        try:
            return json.loads(self.path.read_text())
        except FileNotFoundError:
            return {"clusters": {}, "calls": []}

    def calls(self, kind: str | None = None) -> list:
        return [c for c in self.state()["calls"] if kind is None or c[0] == kind]

    def add(self, name: str, status: str = "UP", **kw) -> None:
        st = self.state()
        st["clusters"][name] = {"status": status, "cloud": "lambda", "region": "us-east-1", "instance_type": "gpu_1x_a10",
                                "head_ip": "10.0.0.9", **kw}
        self.path.write_text(json.dumps(st))

    def names(self) -> set[str]:
        return set(self.state()["clusters"])


@pytest.fixture
def fake_sky(tmp_path, monkeypatch):
    """SkyPilot's SDK as a tiny package keeping its clusters in a JSON file, importable in-process."""
    if __import__("importlib").util.find_spec("sky") is not None:
        pytest.skip("SkyPilot is installed here")
    site = tmp_path / "site"
    pkg = site / "sky"
    (pkg / "skylet").mkdir(parents=True)
    (pkg / "__init__.py").write_text(FAKE_SKY)
    (pkg / "exceptions.py").write_text("class ClusterDoesNotExist(ValueError):\n    pass\n")
    (pkg / "skylet" / "__init__.py").write_text("")
    (pkg / "skylet" / "autostop_lib.py").write_text("import enum\nclass AutostopWaitFor(enum.Enum):\n    JOBS = 'jobs'\n")
    monkeypatch.syspath_prepend(str(site))
    monkeypatch.setenv("FAKE_SKY_STATE", str(tmp_path / "sky.json"))
    for k in ("FAKE_SKY_FAIL", "FAKE_SKY_CLOUD", "FAKE_SKY_REGION", "FAKE_SKY_IP"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(gpu, "catalog_price", lambda it, cloud, region, zone, spot: 1.29 if it == "gpu_1x_a10" else None)
    monkeypatch.setattr(gpu, "catalog_placements", lambda *a: {"best": {"usd_per_hour": 1.29, "cloud": "Lambda"}, "pairs": [["lambda", "us-east-1"]]})
    yield FakeSky(tmp_path / "sky.json")
    for name in [m for m in sys.modules if m == "sky" or m.startswith("sky.")]:
        del sys.modules[name]


class Replicas:
    """The replicas' gates and vLLM as httpx answers: lease renewals and /v1/models, by host."""

    def __init__(self):
        self.vllm = "running"
        self.ready = True
        self.down: dict[str, Exception] = {}
        self.seen: list[tuple[str, str, str | None]] = []
        self.exit_code: int | None = None
        self.gate_status: dict | None = None  # what /meritorder/status answers; None: 404

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.seen.append((request.method, str(request.url), request.headers.get("authorization")))
        if request.url.host in self.down:
            raise self.down[request.url.host]
        if request.url.path == skypilot.LEASE_PATH:
            return httpx.Response(200, json={"expires_in": 900, "vllm": self.vllm, "exit_code": self.exit_code})
        if request.url.path == skypilot.STATUS_PATH:
            return httpx.Response(200, json=self.gate_status) if self.gate_status is not None else httpx.Response(404)
        if request.url.path == "/v1/models":
            return httpx.Response(200 if self.ready else 503, json={"data": []})
        if request.url.path == "/metrics":
            return httpx.Response(200, text="vllm:kv_cache_usage_perc 0.3\n")
        return httpx.Response(404)


@pytest.fixture
def replicas(monkeypatch):
    r = Replicas()
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: REAL_ASYNC_CLIENT(transport=httpx.MockTransport(r.handler), **kw))
    return r


async def settle(prov: SkyPilotProvisioner) -> None:
    """Wait for the provisioner's background launches and teardowns."""
    await asyncio.gather(*(launch.task for launch in prov._launching.values()), *prov._removing.values(), return_exceptions=True)
    prov.fleet.forget()


# ------------------------------------------------------------------ the SDK calls


async def test_sky_calls_in_process_and_in_skypilots_own_python(fake_sky, monkeypatch):
    fake_sky.add("mo-ab12-gpu-1")
    fake_sky.add("mo-zz99-gpu-1")  # another install's: never listed
    [c] = await skypilot.sky_call({"op": "clusters", "prefix": "mo-ab12-", "refresh": True})
    assert (c["name"], c["status"], c["cloud"], c["instance_type"], c["head_ip"]) == ("mo-ab12-gpu-1", "UP", "lambda", "gpu_1x_a10", "10.0.0.9")
    assert await skypilot.sky_call({"op": "down", "cluster": "mo-ab12-gpu-1"}) == {"gone": False}
    assert await skypilot.sky_call({"op": "down", "cluster": "mo-ab12-gpu-1"}) == {"gone": True}
    monkeypatch.setenv("FAKE_SKY_FAIL", "status")
    with pytest.raises(SkyError, match="API server unhappy"):
        await skypilot.sky_call({"op": "clusters", "prefix": "mo-", "refresh": False})
    monkeypatch.delenv("FAKE_SKY_FAIL")
    # The same through SkyPilot's own interpreter (the -gpu image's /opt/skypilot), with this process's environment.
    monkeypatch.setattr(skypilot, "_in_process", lambda: False)
    monkeypatch.setenv("MERITORDER_SKY_PYTHON", sys.executable)
    monkeypatch.setenv("PYTHONPATH", str(fake_sky.path.parent / "site"))
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    out = await skypilot.sky_call({"op": "launch", "cluster": "mo-ab12-gpu-2", "task": {"name": "x"}, "autostop_minutes": 5,
                                   "secrets": ["HF_TOKEN", "MISSING"]})
    assert out == {"job_id": 1}
    [launch] = fake_sky.calls("launch")
    assert launch[1:6] == ["mo-ab12-gpu-2", 5, True, "AutostopWaitFor.JOBS", ["HF_TOKEN"]]  # down, jobs only, the token as a secret
    monkeypatch.setenv("FAKE_SKY_FAIL", "launch")
    with pytest.raises(SkyError, match="no capacity anywhere"):
        await skypilot.sky_call({"op": "launch", "cluster": "mo-ab12-gpu-3", "task": {}, "autostop_minutes": 5})
    monkeypatch.setenv("MERITORDER_SKY_PYTHON", sys.executable)
    with pytest.raises(SkyError, match="KeyError"):
        await skypilot.sky_call({"op": "nope"})
    monkeypatch.setattr(gpu, "sky_python", lambda: None)
    with pytest.raises(SkyError, match="isn't installed"):
        await skypilot.sky_call({"op": "clusters", "prefix": "mo-"})


async def test_fleet_lists_once_and_stands_in_while_skypilot_is_down(fake_sky, monkeypatch):
    fleet = Fleet(IDENT)
    fake_sky.add("mo-ab12-gpu-1")
    assert [c["name"] for c in await fleet.clusters()] == ["mo-ab12-gpu-1"]
    await fleet.clusters()
    assert len(fake_sky.calls("status")) == 1  # listed once a cycle for every deployment
    monkeypatch.setenv("FAKE_SKY_FAIL", "status")
    fleet.forget()
    assert [c["name"] for c in await fleet.clusters()] == ["mo-ab12-gpu-1"]  # the last list, while it's fresh enough
    with pytest.raises(SkyError):
        await fleet.clusters(fresh=True)
    with pytest.raises(ValueError, match="isn't one of this install's"):
        await fleet.down("mo-zz99-gpu-1")


# ------------------------------------------------------------------ the provisioner


async def test_up_scale_down_with_leases_keys_and_prices(fake_sky, replicas, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    cfg = sky_config()
    prov = SkyPilotProvisioner(cfg.backends[0], cfg, Fleet(IDENT))
    await prov.up(2)
    st = await prov.status()
    assert {r["status"] for r in st.replicas_info} == {"PROVISIONING"} and not st.ready and st.usd_per_hour == 2 * 1.29
    await settle(prov)
    assert fake_sky.names() == {"mo-ab12-gpu-1", "mo-ab12-gpu-2"}
    launch = fake_sky.calls("launch")[0]
    assert launch[5] == ["HF_TOKEN"] and launch[6]["resources"]["autostop"]["down"] is True
    st = await prov.status()
    assert st.ready and st.replicas == 2 and st.endpoint == "http://10.0.0.1:8000/v1" and st.usd_per_hour == 2.58 and not st.failed
    rep = st.replicas_info[0]
    assert (rep["id"], rep["cluster"], rep["cloud"], rep["instance_type"], rep["usd_per_hour"], rep["lease_expires_in"]) == (
        1, "mo-ab12-gpu-1", "lambda", "gpu_1x_a10", 1.29, 900)
    leases = [s for s in replicas.seen if s[1].endswith(skypilot.LEASE_PATH)]
    assert len(leases) == 2 and all(s[0] == "POST" and s[2] == f"Bearer {IDENT.api_key('gpu')}" for s in leases)
    assert prov.wanted() == {"mo-ab12-gpu-1", "mo-ab12-gpu-2"} and prov.headers()["authorization"].startswith("Bearer mo-")

    await prov.scale(1)
    await settle(prov)
    assert fake_sky.names() == {"mo-ab12-gpu-1"} and prov.wanted() == {"mo-ab12-gpu-1"}
    replicas.ready = False
    st = await prov.status()
    assert not st.ready and st.replicas_info[0]["status"] == "STARTING" and "loading" in st.message
    replicas.vllm = "exited"
    assert (await prov.status()).replicas_info[0]["status"] == "SHUTTING_DOWN"
    await prov.down()
    assert fake_sky.names() == set() and prov.want == 0


async def test_failed_launches_are_torn_down_and_retried_later(fake_sky, replicas, monkeypatch):
    cfg = sky_config()
    prov = SkyPilotProvisioner(cfg.backends[0], cfg, Fleet(IDENT))
    monkeypatch.setenv("FAKE_SKY_FAIL", "launch")
    await prov.up(1)
    await settle(prov)
    assert fake_sky.names() == set()  # the half-made cluster went too
    st = await prov.status()
    assert st.failed and st.replicas_info[0]["status"] == "FAILED" and "no capacity" in st.message
    assert len(fake_sky.calls("launch")) == 1  # not tried again every cycle
    monkeypatch.delenv("FAKE_SKY_FAIL")
    prov._failed[prov.cluster(1)] = (time.time() - skypilot.LAUNCH_RETRY_SECONDS, "old")
    await prov.status()
    await settle(prov)
    assert len(fake_sky.calls("launch")) == 2 and (await prov.status()).ready


async def test_a_replica_that_dies_before_it_was_ready_waits(fake_sky, replicas):
    cfg = sky_config()
    prov = SkyPilotProvisioner(cfg.backends[0], cfg, Fleet(IDENT))
    replicas.ready = False
    await prov.up(1)
    await settle(prov)
    await prov.status()
    st = fake_sky.state()
    st["clusters"].clear()  # autostop deleted it after vLLM crashed
    fake_sky.path.write_text(json.dumps(st))
    prov.fleet.forget()
    st = await prov.status()
    assert st.failed and "stopped before vLLM was ready" in st.message and len(fake_sky.calls("launch")) == 1


async def test_a_replica_whose_vllm_exits_before_ready_says_why_and_is_torn_down(fake_sky, replicas, caplog):
    cfg = sky_config()
    prov = SkyPilotProvisioner(cfg.backends[0], cfg, Fleet(IDENT))
    replicas.ready = False
    await prov.up(1)
    await settle(prov)
    assert (await prov.status()).replicas_info[0]["status"] == "STARTING"
    tail = [f"line {i}" for i in range(60)] + ["RuntimeError: The NVIDIA driver on your system is too old", "Bearer abc"]
    replicas.vllm, replicas.exit_code = "exited", 1
    replicas.gate_status = {**OLD_DRIVER, "vllm": "exited", "exit_code": 1, "uptime_seconds": 41.0, "tail": tail}
    st = await prov.status()
    [rep] = st.replicas_info
    assert rep["status"] == "SHUTTING_DOWN" and rep["message"].startswith("vLLM exited with status 1 before it was ready: the NVIDIA driver")
    [(kind, ev)] = prov.drain_events()
    assert kind == "replica_failed" and ev["source"] == "gate" and ev["cluster"] == "mo-ab12-gpu-1" and ev["exit_code"] == 1
    assert len(ev["tail"]) == skypilot.TAIL_KEPT and ev["tail"][-1] == "Bearer [redacted]" and ev["driver_version"] == "535.129.03"
    assert "CUDA 13 images need driver 580.65.06" in ev["reason"] and prov.drain_events() == []
    assert "its last 40 lines (gate)" in caplog.text and "  | RuntimeError: The NVIDIA driver" in caplog.text  # the controller's log
    await settle(prov)
    assert fake_sky.names() == set()  # torn down at once, after it said why
    await prov.status()
    st = await prov.status()
    assert st.failed and "the NVIDIA driver" in st.message and len(fake_sky.calls("launch")) == 1  # waits, as after a failed launch
    assert len([s for s in replicas.seen if s[1].endswith(skypilot.STATUS_PATH)]) == 1 and prov.drain_events() == []


async def test_a_replica_that_exits_after_it_was_ready_is_logged_too(fake_sky, replicas):
    cfg = sky_config()
    prov = SkyPilotProvisioner(cfg.backends[0], cfg, Fleet(IDENT))
    await prov.up(1)
    await settle(prov)
    assert (await prov.status()).ready
    replicas.vllm, replicas.exit_code = "exited", -9
    replicas.gate_status = {"vllm": "exited", "exit_code": -9, "tail": ["INFO serving", "Killed"]}
    st = await prov.status()
    assert st.replicas_info[0]["message"] == "vLLM has exited; tearing the machine down" and not prov._failed
    [ready, (_, ev)] = prov.drain_events()
    assert ready[0] == "replica_ready"
    assert ev["reason"] == "vLLM exited with status -9 (last output: Killed)"
    await settle(prov)
    assert fake_sky.names() == set()  # torn down after it said why (test_gpu_safety has the relaunch and its backoff)


async def test_without_the_gate_sky_logs_says_why_within_the_budget(fake_sky, replicas, monkeypatch):
    cfg = sky_config()
    prov = SkyPilotProvisioner(cfg.backends[0], cfg, Fleet(IDENT))
    monkeypatch.setenv("FAKE_SKY_FAIL", "launch")
    await prov.up(1)
    await settle(prov)
    [(_, ev)] = prov.drain_events()  # a failed launch: there's no gate to ask, so SkyPilot's log of the task
    assert ev["source"] == "sky logs" and ev["tail"][-2:] == ["(task, pid=1) RuntimeError: no kernel image token=[redacted]", "(task, pid=1) bye"]
    assert ev["reason"].startswith("launch failed: RuntimeError: no capacity anywhere (last output: (task, pid=1) RuntimeError: no kernel image")
    assert fake_sky.calls("tail_logs")[0] == ["tail_logs", "mo-ab12-gpu-1", None, False, skypilot.TAIL_KEPT]
    assert fake_sky.names() == set()
    # SkyPilot can't give it either: said, and the teardown goes on.
    monkeypatch.setenv("FAKE_SKY_FAIL", "logs")
    ev = await prov._diagnose("mo-ab12-gpu-1", "it failed")
    assert ev["tail"] == [] and "ClusterDoesNotExist" in ev["logs_error"] and ev["reason"] == "it failed"
    # Neither answers in time: the teardown waits LOGS_BUDGET seconds at most.

    async def hang(req, timeout=0):
        await asyncio.sleep(60)

    monkeypatch.setattr(skypilot, "sky_call", hang)
    monkeypatch.setattr(skypilot, "LOGS_BUDGET", 0.2)
    t0 = time.monotonic()
    ev = await prov._diagnose("mo-ab12-gpu-1", "it failed", "http://10.0.0.1:8000/v1")  # the gate 404s: SkyPilot is asked next
    assert ev["logs_error"] == "no answer within 0.2s" and time.monotonic() - t0 < 2


async def test_down_reads_why_a_failed_replica_failed_first(fake_sky, replicas):
    cfg = sky_config()
    prov = SkyPilotProvisioner(cfg.backends[0], cfg, Fleet(IDENT))
    replicas.ready = False
    await prov.up(1)
    await settle(prov)
    await prov.status()  # its gate's address is known from here
    prov._failed[prov.cluster(1)] = (time.time(), "it won't load")
    replicas.gate_status = {**OLD_DRIVER, "vllm": "running", "exit_code": None, "tail": ["still trying"]}
    await prov.down()
    [(_, ev)] = prov.drain_events()
    assert ev["source"] == "gate" and ev["reason"].startswith("it won't load: the NVIDIA driver") and fake_sky.names() == set()
    replicas.gate_status = None
    assert await prov._gate_status("http://10.0.0.1:8000/v1") is None  # a 404: no status


async def test_the_controller_writes_the_failure_to_the_gpu_activity_log():
    cfg = sky_config()
    store = MemoryStore()

    class Prov:
        def drain_events(self):
            return [("replica_failed", {"cluster": "mo-ab12-gpu-1", "reason": "vLLM exited", "tail": ["x"]})]

    await controller.write_events(store, Deployment(backend=cfg.backends[0], model=cfg.models[0], provisioner=Prov()))
    await controller.write_events(store, Deployment(backend=cfg.backends[0], model=cfg.models[0], provisioner=object()))
    [ev] = await store.gpu_events()
    assert (ev["kind"], ev["backend"], ev["actor"], ev["detail"]["tail"]) == ("replica_failed", "gpu", "controller", ["x"])


async def test_restarts_and_config_changes_keep_what_runs(fake_sky, replicas):
    cfg = sky_config()
    fake_sky.add("mo-ab12-gpu-1", head_ip="10.0.0.1")
    fresh = SkyPilotProvisioner(cfg.backends[0], cfg, Fleet(IDENT))
    fresh.restore({"state": WARM, "replicas": 1})
    assert fresh.wanted() == {"mo-ab12-gpu-1"}
    fresh.restore({"state": API_ONLY})  # only the first record counts
    assert (await fresh.status()).ready and fake_sky.calls("launch") == []
    later = SkyPilotProvisioner(cfg.backends[0], cfg, Fleet(IDENT))
    later.adopt(fresh)
    assert later.want == 1 and later._ever_ready == {"mo-ab12-gpu-1"}
    idle = SkyPilotProvisioner(cfg.backends[0], cfg, Fleet(IDENT))
    idle.restore({"state": API_ONLY, "replicas": 0})
    assert idle.wanted() == set()


async def test_ssh_tunnel_where_the_port_cant_be_reached(fake_sky, replicas, monkeypatch):
    cfg = sky_config()
    prov = SkyPilotProvisioner(cfg.backends[0], cfg, Fleet(IDENT))
    opened = []

    async def ensure(self):
        opened.append(self.cluster)
        self.port = 40001
        return self.url()

    monkeypatch.setattr(skypilot.Tunnel, "ensure", ensure)
    monkeypatch.setenv(skypilot.TUNNEL_HOST_ENV, "controller")
    fake_sky.add("mo-ab12-gpu-1", head_ip="10.0.0.1")
    fake_sky.add("mo-ab12-gpu-2", head_ip="10.0.0.2", region="us-south-1")  # Lambda opens no ports there
    replicas.down["10.0.0.1"] = httpx.ConnectTimeout("dropped")  # a firewall that drops the controller's packets
    prov.restore({"state": WARM, "replicas": 2})
    for _ in range(skypilot.TUNNEL_AFTER):
        st = await prov.status()
        prov.fleet.forget()
    by_id = {r["id"]: r for r in st.replicas_info}
    assert by_id[2]["tunnel"] and by_id[2]["endpoint"] == "http://controller:40001/v1" and by_id[2]["status"] == "READY"
    assert by_id[1]["status"] == "STARTING"
    st = await prov.status()
    assert {r["id"]: r.get("tunnel") for r in st.replicas_info} == {1: True, 2: True} and st.replicas == 2
    assert controller.kv_endpoints(ServeConfig(model="ow"), [{**r, "url": r["endpoint"]} for r in st.replicas_info]) == {}
    cfg.backends[0].serve.ssh_tunnel = "never"
    assert not prov._tunnel_for({"name": "mo-ab12-gpu-2", "cloud": "lambda", "region": "us-south-1"})
    cfg.backends[0].serve.ssh_tunnel = "always"
    assert prov._tunnel_for({"name": "mo-ab12-gpu-9", "cloud": "aws", "region": "us-east-1"})
    await prov.down()
    assert prov._tunnels == {}


async def test_each_cloud_is_reached_its_own_way(fake_sky, replicas, monkeypatch):
    """Head IP and port on most clouds; the address SkyPilot mapped the port to on Kubernetes and RunPod (looked up once);
    only SSH where SkyPilot opens no ports."""
    cfg = sky_config()
    prov = SkyPilotProvisioner(cfg.backends[0], cfg, Fleet(IDENT))
    opened = []

    async def ensure(self):
        opened.append(self.cluster)
        self.port = 40002
        return self.url()

    monkeypatch.setattr(skypilot.Tunnel, "ensure", ensure)
    fake_sky.add("mo-ab12-gpu-1", cloud="kubernetes", region="my-context", head_ip="10.8.0.4", endpoint="203.0.113.7:8000")
    fake_sky.add("mo-ab12-gpu-2", cloud="runpod", head_ip="10.0.0.5", endpoint="http://198.51.100.9:41234/")
    fake_sky.add("mo-ab12-gpu-3", cloud="vast", head_ip="10.0.0.6")
    fake_sky.add("mo-ab12-gpu-4", cloud="aws", head_ip="10.0.0.7")
    prov.restore({"state": WARM, "replicas": 4})
    st = await prov.status()
    by_id = {r["id"]: r for r in st.replicas_info}
    assert by_id[1]["endpoint"] == "http://203.0.113.7:8000/v1" and by_id[2]["endpoint"] == "http://198.51.100.9:41234/v1"
    assert by_id[3]["tunnel"] and opened == ["mo-ab12-gpu-3"] and by_id[4]["endpoint"] == "http://10.0.0.7:8000/v1"
    assert st.replicas == 4
    prov.fleet.forget()
    await prov.status()
    assert len(fake_sky.calls("endpoints")) == 2  # once per cluster
    # No address yet (a Kubernetes LoadBalancer still pending), or none at all: the replica waits.
    fake_sky.add("mo-ab12-gpu-5", cloud="kubernetes", head_ip="10.8.0.9")
    prov.want = 5
    prov.fleet.forget()
    assert {r["id"]: r for r in (await prov.status()).replicas_info}[5]["status"] == "STARTING"
    monkeypatch.setenv("FAKE_SKY_FAIL", "endpoints")
    fake_sky.add("mo-ab12-gpu-6", cloud="runpod", head_ip="10.0.0.9")
    prov.restore({"state": WARM, "replicas": 6})
    prov.want = 6
    prov.fleet.forget()
    by_id = {r["id"]: r for r in (await prov.status()).replicas_info}
    assert by_id[5]["status"] == by_id[6]["status"] == "STARTING" and "no address" in by_id[6]["message"]
    cfg.backends[0].serve.ssh_tunnel = "never"
    prov.fleet.forget()
    by_id = {r["id"]: r for r in (await prov.status()).replicas_info}
    assert by_id[3]["status"] == "STARTING" and "only through an SSH tunnel" in by_id[3]["message"]
    await settle(prov)


async def test_a_cluster_without_autodown_gets_it_back(fake_sky, replicas, monkeypatch):
    """A launch cut short after the machine came up but before SkyPilot set autostop, or someone's `sky autostop -i -1`:
    the controller sets autodown again, so the lease still ends with the machine deleted."""
    cfg = sky_config()
    prov = SkyPilotProvisioner(cfg.backends[0], cfg, Fleet(IDENT))
    fake_sky.add("mo-ab12-gpu-1", autostop=-1, to_down=False)
    fake_sky.add("mo-ab12-gpu-2", autostop=5, to_down=False)  # would stop, not delete
    fake_sky.add("mo-ab12-gpu-3")
    prov.restore({"state": WARM, "replicas": 3})
    await prov.status()
    await asyncio.gather(*(t for _, t in prov._autodown.values()))
    calls = fake_sky.calls("autostop")
    assert sorted(c[1] for c in calls) == ["mo-ab12-gpu-1", "mo-ab12-gpu-2"] and all(c[2:] == [5, "AutostopWaitFor.JOBS", True] for c in calls)
    await prov.status()
    assert len(fake_sky.calls("autostop")) == 2  # set: nothing more to do
    monkeypatch.setenv("FAKE_SKY_FAIL", "autostop")
    fake_sky.add("mo-ab12-gpu-3", autostop=-1, to_down=False)
    prov.fleet.forget()
    await prov.status()
    await asyncio.gather(*(t for _, t in prov._autodown.values()))
    prov.fleet.forget()
    await prov.status()
    assert len(fake_sky.calls("autostop")) == 3  # failed, and not tried again for AUTODOWN_RETRY_SECONDS
    prov._autodown["mo-ab12-gpu-3"] = (time.time() - skypilot.AUTODOWN_RETRY_SECONDS, prov._autodown["mo-ab12-gpu-3"][1])
    prov.fleet.forget()
    await prov.status()
    await asyncio.gather(*(t for _, t in prov._autodown.values()))
    assert len(fake_sky.calls("autostop")) == 4


async def test_tunnel_runs_ssh_with_skypilots_config(tmp_path, monkeypatch):
    monkeypatch.setattr(skypilot.Path, "home", lambda: tmp_path)
    t = skypilot.Tunnel("mo-ab12-gpu-1", 8000)
    assert await t.ensure() is None and t.url() is None  # no SSH config from SkyPilot for it
    (tmp_path / ".sky" / "generated" / "ssh").mkdir(parents=True)
    (tmp_path / ".sky" / "generated" / "ssh" / "mo-ab12-gpu-1").write_text("Host mo-ab12-gpu-1\n")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "ssh").write_text(f"#!/bin/sh\necho \"$@\" > {tmp_path / 'ssh.args'}\nexec sleep 30\n")
    (bindir / "ssh").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv(skypilot.TUNNEL_HOST_ENV, "controller")
    url = await t.ensure()
    assert url == f"http://controller:{t.port}/v1" and await t.ensure() == url  # one ssh while it runs
    for _ in range(50):
        if (tmp_path / "ssh.args").exists():
            break
        await asyncio.sleep(0.05)
    args = (tmp_path / "ssh.args").read_text()
    assert f"-L 0.0.0.0:{t.port}:127.0.0.1:8000 mo-ab12-gpu-1" in args and "ExitOnForwardFailure=yes" in args
    t.close()
    assert t.proc is None


async def test_lease_renewal_failures(monkeypatch):
    cfg = sky_config()
    prov = SkyPilotProvisioner(cfg.backends[0], cfg, Fleet(IDENT))
    answers = iter([httpx.Response(401), httpx.Response(200, text="nope"), httpx.ConnectError("refused")])

    def handler(request):
        a = next(answers)
        if isinstance(a, Exception):
            raise a
        return a

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: REAL_ASYNC_CLIENT(transport=httpx.MockTransport(handler), **kw))
    assert await prov._renew("http://r/v1") == (None, "the gate answered 401")
    assert await prov._renew("http://r/v1") == (None, "the gate's answer wasn't JSON")
    assert await prov._renew("http://r/v1") == (None, "ConnectError")


async def test_status_without_skypilot(monkeypatch):
    cfg = sky_config()
    monkeypatch.setattr(skypilot, "_in_process", lambda: False)
    monkeypatch.setattr(gpu, "sky_python", lambda: None)
    prov = SkyPilotProvisioner(cfg.backends[0], cfg, Fleet(IDENT))
    st = await prov.status()
    assert st.failed and "isn't installed" in st.message
    with pytest.raises(SkyError, match="isn't installed"):
        await prov.up(1)


# ------------------------------------------------------------------ the sweep and the controller


async def test_sweep_tears_down_what_no_deployment_wants(fake_sky, replicas, monkeypatch):
    monkeypatch.setattr(gpu, "sky_available", lambda: True)
    cfg = sky_config()
    store, state = MemoryStore(), MemoryState()
    fleet = Fleet(IDENT)
    deps = controller.deployments(cfg, fleet)
    fake_sky.add("mo-ab12-gpu-1")  # wanted once the record says so
    fake_sky.add("mo-ab12-old-name-1")  # a backend since renamed
    fake_sky.add("mo-ab12-gpu-3")  # left over from a scale cut short
    fake_sky.add("mo-zz99-gpu-1")  # another install's: never touched
    await state.set_json("deploy:gpu", {"state": WARM, "replicas": 1, "since": time.time(), "warm_at": time.time()})
    await store.set_setting("gpu_controls", {"deployments": {"gpu": {"mode": "on", "replicas": 1, "by": "root"}}})
    [rec] = await controller.tick(cfg, store, state, deps, fleet=fleet)
    await settle(deps[0].provisioner)
    assert rec["state"] == WARM and fake_sky.names() == {"mo-ab12-gpu-1", "mo-zz99-gpu-1"}
    events = [e for e in await store.gpu_events() if e["kind"] == "sweep"]
    # gpu-3 is above the count asked for, so the deployment tears it down itself; the sweep takes what nobody wants.
    assert [e["detail"]["cluster"] for e in events] == ["mo-ab12-old-name-1"] and events[0]["detail"]["reason"] == "no deployment wants it"
    assert rec["burn_usd_per_hour"] == 2.58  # each cluster at its own catalog price, gpu-3 too while it was there
    # The backend removed from the config: its cluster goes too, on the next pass.
    cfg.backends.clear()
    assert await controller.sweep(cfg, store, [], fleet) == ["mo-ab12-gpu-1"]
    assert await controller.sweep(cfg, store, [], fleet) == []
    assert not fleet.leftovers and await controller.sweep(cfg, store, [], fleet) == []  # nothing configured, nothing left: no call


async def test_sweep_skips_without_skypilot_and_survives_its_errors(fake_sky, monkeypatch, caplog):
    cfg = sky_config()
    fleet = Fleet(IDENT)
    monkeypatch.setattr(gpu, "sky_available", lambda: False)
    assert await controller.sweep(cfg, MemoryStore(), [], fleet) == [] and "isn't installed" in caplog.text and fleet.swept_once
    assert await controller.sweep(cfg, MemoryStore(), [], None) == []
    monkeypatch.setattr(gpu, "sky_available", lambda: True)
    monkeypatch.setenv("FAKE_SKY_FAIL", "status")
    assert await controller.sweep(cfg, MemoryStore(), [], fleet) == [] and "couldn't list" in caplog.text
    monkeypatch.setenv("FAKE_SKY_FAIL", "down")
    fake_sky.add("mo-ab12-stray-1")
    store = MemoryStore()
    assert await fleet.sweep(set(), store) == [] and fleet.leftovers
    assert [e["detail"]["action"] for e in await store.gpu_events() if e["kind"] == "error"] == ["sweep"]


class Flaky:
    """A provisioner whose start fails, and whose teardown fails while ``down_fails``."""

    def __init__(self):
        self.calls, self.down_fails = [], True

    async def status(self):
        return gpu.Status()

    async def up(self, n):
        self.calls.append(("up", n))
        raise RuntimeError("quota exceeded")

    async def scale(self, n):
        self.calls.append(("scale", n))
        raise RuntimeError("API refused")

    async def down(self):
        self.calls.append(("down",))
        if self.down_fails:
            raise RuntimeError("timed out")


async def test_a_failed_start_is_torn_down_or_kept_tracked_until_it_is():
    cfg = sky_config()
    cfg.backends[0].serve.provisioner = "kubernetes"  # any provisioner: the controller's handling is the same
    store, state = MemoryStore(), MemoryState()
    prov = Flaky()
    deps = [Deployment(backend=cfg.backends[0], model=cfg.models[0], provisioner=prov)]
    await store.set_setting("gpu_controls", {"deployments": {"gpu": {"mode": "on", "replicas": 2, "by": "root"}}})
    t = time.time()
    [rec] = await controller.tick(cfg, store, state, deps, now=t)
    assert prov.calls == [("up", 2), ("down",)]  # a start that failed is taken down at once
    assert rec["state"] == PROVISIONING and rec["stop_pending"] and rec["replicas"] == 2  # ...and, failing that, still tracked
    assert "stop failed, will retry" in rec["reason"]
    [rec] = await controller.tick(cfg, store, state, deps, now=t + 60)
    assert prov.calls[-1] == ("down",) and rec["state"] == PROVISIONING and rec["stop_pending"]  # still billing: still shown
    assert (await store.gpu_spend_since(gpu_month(t)))  # and still accrued
    prov.down_fails = False
    [rec] = await controller.tick(cfg, store, state, deps, now=t + 120)
    assert rec["state"] == API_ONLY and "stop_pending" not in rec and rec["cooldown_until"] >= t + 1800
    # The pin ended with the failed start: it doesn't launch the same broken deployment again after every cooldown.
    assert "gpu" not in ((await store.get_setting("gpu_controls")).get("deployments") or {})
    assert (await state.get_json("gpu:controls")) == await store.get_setting("gpu_controls")
    ended = [e for e in await store.gpu_events() if e["kind"] == "control"]
    assert ended[0]["detail"]["mode"] == "auto" and ended[0]["detail"]["reason"].startswith("pin ended: the start failed (up failed")
    assert ended[0]["detail"]["pinned_by"] == "root" and ended[0]["actor"] == "controller"
    # Pinned again, a start that fails and whose teardown works: off at once, cooling down, and the pin ends again.
    await store.set_setting("gpu_controls", {"deployments": {"gpu": {"mode": "on", "replicas": 2, "by": "root"}}})
    await state.set_json("deploy:gpu", {})
    [rec] = await controller.tick(cfg, store, state, deps, now=t + 180)
    assert rec["state"] == API_ONLY and rec["reason"].startswith("up failed: quota exceeded") and "stop_pending" not in rec
    assert not (await store.get_setting("gpu_controls"))["deployments"]


async def test_a_failed_scale_keeps_what_runs_and_tries_again():
    cfg = sky_config()
    store, state = MemoryStore(), MemoryState()
    prov = Flaky()

    async def ready():
        return gpu.Status(ready=True, endpoint="http://r/v1", replicas=1)

    prov.status = ready
    deps = [Deployment(backend=cfg.backends[0], model=cfg.models[0], provisioner=prov)]
    t = time.time()
    await state.set_json("deploy:gpu", {"state": WARM, "replicas": 1, "since": t, "warm_at": t, "endpoint": "http://r/v1"})
    await store.set_setting("gpu_controls", {"deployments": {"gpu": {"mode": "on", "replicas": 2, "by": "root"}}})
    [rec] = await controller.tick(cfg, store, state, deps, now=t)
    assert prov.calls == [("scale", 2)] and rec["state"] == WARM and rec["replicas"] == 1 and "scale failed, will retry" in rec["reason"]
    [rec] = await controller.tick(cfg, store, state, deps, now=t + 60)
    assert prov.calls == [("scale", 2), ("scale", 2)]


def gpu_month(t: float):
    from datetime import UTC, datetime

    from meritorder.store import month_start

    return month_start(datetime.fromtimestamp(t, UTC))


async def test_run_controller_warns_when_the_lease_is_too_short_for_its_cycle(monkeypatch, caplog):
    from test_worker_controller import pg_memory_store

    cfg = sky_config(lease_minutes=3)
    store = pg_memory_store()
    monkeypatch.setattr(controller, "make_store", lambda url: store)
    monkeypatch.setattr(controller, "make_state", lambda url: MemoryState())

    async def no_tick(*a, **kw):
        return []

    monkeypatch.setattr(controller, "tick", no_tick)
    monkeypatch.setattr(controller, "publish_gpu_prices", no_tick)
    await controller.run_controller(cfg, once=True, interval=120)
    assert "renews SkyPilot leases every 120s" in caplog.text
    assert await store.get_setting(skypilot.INSTALL_SETTING)  # this install's id, made once for every process


# ------------------------------------------------------------------ gateways


async def test_gateways_send_the_deployment_key(monkeypatch):
    cfg = sky_config()
    [backend] = build_backends(cfg).values()
    assert isinstance(backend.auth, ReplicaKeyAuth)
    monkeypatch.setattr(skypilot, "_known", None)
    assert await backend.auth.apply("POST", "http://10.0.0.1:8000/v1/chat/completions", {}, b"") == {}  # not loaded yet
    from meritorder.proxy import refresh_deployments

    store = MemoryStore()
    await refresh_deployments({"gpu": backend}, MemoryState(), store)
    ident = skypilot.known_identity()
    hdrs = await backend.auth.apply("POST", "http://10.0.0.1:8000/v1/chat/completions", {}, b"")
    assert hdrs == {"authorization": f"Bearer {ident.api_key('gpu')}"} and ident == await skypilot.load_identity(store)
    monkeypatch.setattr(skypilot, "_known", None)


def test_heredoc_marker_is_not_in_the_gate():
    assert "MERITORDER_GATE_EOF" not in (gpu.Path(replica_gate.__file__)).read_text()
    assert os.path.basename(replica_gate.__file__) == "replica_gate.py"


def test_gate_uninstalls_an_optional_package_the_image_ships_broken():
    """vLLM v0.32.0-cu129 bundles a CUDA 13 torchcodec next to torch+cu129: importing it raises an OSError that vLLM
    doesn't catch. Without it vLLM starts, with no audio input."""
    calls = []

    def broken(cmd, **kw):
        calls.append(cmd)
        if "-c" in cmd:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="Traceback ...\nOSError: Could not load this library: libtorchcodec_image.so")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    dropped = replica_gate.drop_broken_optional_packages(broken, python="py")
    assert dropped == ["torchcodec (OSError: Could not load this library: libtorchcodec_image.so)"]
    assert calls[-1] == ["py", "-m", "pip", "uninstall", "-y", "-q", "torchcodec"]
    absent = lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, stdout="", stderr="ModuleNotFoundError: No module named 'torchcodec'")
    fine = lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    assert replica_gate.drop_broken_optional_packages(absent) == [] and replica_gate.drop_broken_optional_packages(fine) == []

    def stuck(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, 300)

    assert replica_gate.drop_broken_optional_packages(stuck) == []
