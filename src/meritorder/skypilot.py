"""SkyPilot: one plain cluster per replica, which Merit Order runs itself (docs/05-gpu-autoscaling.md#how-skypilot-runs-a-deployment).

SkyPilot launches each replica (finding capacity across the allowed clouds and regions, setting up the image, port, disk
and labels), prices it from its catalog, and tears it down. Everything else is Merit Order's: the controller decides how
many replicas run, the gateways pick the replica for each request, and health is probed here.

Nothing runs unwatched:

- **A key per deployment.** Every replica's only open port is its gate (replica_gate.py), which lets in only callers
  with the deployment's key: an HMAC of this install's secret and the deployment's name, which the controller and every
  gateway work out for themselves. Only its SHA-256 goes to the machine.
- **A lease.** The controller renews each wanted replica's lease every cycle. A gate whose lease runs out stops vLLM, the
  task ends, and SkyPilot's autostop (always ``down``, waiting for jobs only) deletes the cluster ``autostop_minutes``
  later. That happens on the machine, whatever became of the controller.
- **A sweep.** Every cycle, and on the first one after a start, every cluster with this install's prefix that no
  deployment wants (a renamed or removed backend, a failed teardown, a crash halfway through a scale) is torn down and
  logged in the GPU activity log.
- **Why it failed, kept.** Before a replica that failed is torn down, its gate's ``/meritorder/status`` (vLLM's last
  output, exit status, the NVIDIA driver and CUDA versions) is read, or, when the gate doesn't answer, ``sky logs
  --tail``. The last ``TAIL_KEPT`` lines go in the controller's log and in a ``replica_failed`` GPU activity log entry.
  This never holds a teardown up by more than ``LOGS_BUDGET`` seconds.

Cluster names are ``mo-<install id>-<deployment>-<n>``, so two installs on one cloud account never touch each other's.

Clouds differ in how a replica is reached (its IP and port, a port SkyPilot maps, or only SSH) and in what's left on the
machine to delete it; sky_clouds.py has the differences, and docs/05-gpu-autoscaling.md#which-clouds the audit.

SkyPilot runs in its own Python (the -gpu image's /opt/skypilot), so the calls go through a small script run there, or
in-process when SkyPilot is importable here: the SDK's ``status``, ``launch``, ``down``, ``endpoints``, ``autostop`` and
``tail_logs`` (SkyPilot 0.14.2).
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import shlex
import shutil
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from . import gpu, replica_gate, sky_clouds
from .billing.sources import GPU_LABEL, label_value
from .config import BackendConfig, Config

log = logging.getLogger("meritorder.skypilot")

SECRET_ENV, SECRET_SETTING = "MERITORDER_GPU_KEY", "gpu_replica_secret"  # the secret replica keys are derived from
INSTALL_ENV, INSTALL_SETTING = "MERITORDER_INSTALL_ID", "gpu_install_id"  # what this install's cluster names start with
INSTALL_LABEL = "meritorder-install"
TUNNEL_HOST_ENV = "MERITORDER_TUNNEL_HOST"  # the address gateways reach this controller's SSH tunnels at
LEASE_PATH, STATUS_PATH = replica_gate.LEASE_PATH, replica_gate.STATUS_PATH
TAIL_KEPT = 40  # lines of a failed replica's last output kept in the GPU activity log
LOGS_BUDGET = 25  # seconds a teardown may wait for a failed replica's last output, at most
LAUNCH_RETRY_SECONDS = 600  # a replica whose launch failed is tried again after this long
SKY_TIMEOUT = 600  # a teardown; a launch gets provision_timeout_minutes on top
# Listing clusters, and a replica's address: every cycle, before leases are renewed, so a hung API server must give up
# well within a lease (3 minutes at the least).
STATUS_TIMEOUT = 60
PRICE_TTL = 6 * 3600
# How long the last cluster list stands in while SkyPilot can't be asked (counted from the last good list to when the
# failed call started, so a call that hung counts as when it was made): leases are still renewed meanwhile.
STALE_SECONDS = 3600
CRASH_WINDOW = 3600  # a replica whose vLLM exits after it was ready a second time within this long waits LAUNCH_RETRY_SECONDS
TUNNEL_AFTER = 3  # cycles a replica's port times out from here before an SSH tunnel is used instead (ssh_tunnel: auto)
AUTODOWN_RETRY_SECONDS = 600  # how often autodown is set again on a cluster that shows none
NO_FIREWALL = {("lambda", "us-south-1")}  # SkyPilot 0.14.2 skips opening ports there: Lambda has no firewall rules in it
_INSTALL_ID = re.compile(r"^[a-z0-9]{1,12}$")
CACHE_MOUNT = "/mnt/meritorder-cache"  # where the weights cache volume is mounted in the container
# Set on every replica (docs/05, "Faster starts"). The Hub serves weights through Xet now, and huggingface_hub 1.x ignores
# HF_HUB_ENABLE_HF_TRANSFER (the vLLM image has no hf_transfer either); Xet's high-performance mode is its successor:
# every core and the whole network link for the download. vLLM 0.32 turns it on itself when the variable is unset
# (model_executor/model_loader/weight_utils.py); it's set here so it holds for any image.
FAST_START_ENV: dict[str, str] = {"HF_XET_HIGH_PERFORMANCE": "1"}


# ------------------------------------------------------------------ identity and keys


@dataclass(frozen=True)
class Identity:
    """Who this install is to its clusters: the id in their names, and the secret their keys come from."""

    install_id: str
    secret: str

    def api_key(self, backend: str) -> str:
        """The key a deployment's replicas let in, the same in every process of this install."""
        digest = hmac.new(self.secret.encode(), b"meritorder vllm key\0" + backend.encode(), hashlib.sha256).digest()
        return "mo-" + base64.urlsafe_b64encode(digest).decode().rstrip("=")

    def key_sha256(self, backend: str) -> str:
        """What goes to the machine: the key's hash. The gate checks callers against it."""
        return hashlib.sha256(self.api_key(backend).encode()).hexdigest()

    @property
    def prefix(self) -> str:
        return f"mo-{self.install_id}-"

    @classmethod
    def from_env(cls) -> Identity:
        """From MERITORDER_INSTALL_ID and MERITORDER_GPU_KEY; what's missing is made up for this process alone."""
        install = os.environ.get(INSTALL_ENV) or secrets.token_hex(4)
        if not _INSTALL_ID.match(install):
            raise ValueError(f"{INSTALL_ENV} must be 1 to 12 lowercase letters and digits, not {install!r}")
        return cls(install, os.environ.get(SECRET_ENV) or secrets.token_urlsafe(32))


async def load_identity(store) -> Identity:
    """This install's identity: the environment's, else the one kept in the store's settings (the first process to get
    here makes it; every other reads it). The controller and every gateway must agree on it."""
    install = os.environ.get(INSTALL_ENV)
    if not install:
        await store.add_setting(INSTALL_SETTING, secrets.token_hex(4))
        install = await store.get_setting(INSTALL_SETTING)
    secret = os.environ.get(SECRET_ENV)
    if not secret:
        await store.add_setting(SECRET_SETTING, secrets.token_urlsafe(32))
        secret = await store.get_setting(SECRET_SETTING)
    if not _INSTALL_ID.match(str(install)):
        raise ValueError(f"the install id must be 1 to 12 lowercase letters and digits, not {install!r}")
    return Identity(str(install), str(secret))


_known: Identity | None = None


def known_identity() -> Identity | None:
    """This process's identity once ``ensure_identity`` has loaded it (gateways add replica keys with it)."""
    return _known


async def ensure_identity(store) -> Identity | None:
    """Load this install's identity once per process; None (and a log line) while the store can't give it."""
    global _known
    if _known is None:
        try:
            _known = await load_identity(store)
        except Exception as e:  # noqa: BLE001 (the gateway goes on; requests to SkyPilot replicas fail until it's loaded)
            log.warning("couldn't load the GPU replica key: %s", e)
    return _known


def slug(backend: str) -> str:
    """A backend name as part of a cluster name: lowercase letters, digits and -, at most 24, with a short hash of the
    name when that changed it, so two backends never share one."""
    s = re.sub(r"[^a-z0-9-]+", "-", backend.lower()).strip("-")[:24].strip("-") or "gpu"
    return s if s == backend else f"{s}-{hashlib.sha256(backend.encode()).hexdigest()[:4]}"


# ------------------------------------------------------------------ calls into SkyPilot

# Runs in SkyPilot's Python. Prints one line, MERITORDER_RESULT <json>: the answer, or {"error": ...}.
_SKY_SCRIPT = r'''
import json, os, sys


def _get(r, key, default=None):
    try:
        v = r[key]
    except Exception:
        v = getattr(r, key, default)
    return default if v is None else v


def clusters(req):
    import sky
    mode = sky.StatusRefreshMode.AUTO if req.get("refresh") else sky.StatusRefreshMode.NONE
    out = []
    for r in sky.get(sky.status(refresh=mode)) or []:
        name = _get(r, "name", "")
        if not name.startswith(req["prefix"]):
            continue
        handle = _get(r, "handle")
        res = getattr(handle, "launched_resources", None)
        status = _get(r, "status")
        cloud = getattr(res, "cloud", None) if res is not None else None
        out.append({"name": name, "status": getattr(status, "value", None) or str(status or ""),
                    "cloud": str(cloud).lower() if cloud is not None else (_get(r, "cloud") or None),
                    "region": getattr(res, "region", None) or _get(r, "region"), "zone": getattr(res, "zone", None),
                    "instance_type": getattr(res, "instance_type", None), "spot": bool(getattr(res, "use_spot", False)),
                    "head_ip": getattr(handle, "head_ip", None), "launched_at": _get(r, "launched_at"),
                    "autostop": _get(r, "autostop"), "to_down": bool(_get(r, "to_down", False)),
                    "resources": _get(r, "resources_str")})
    return out


def launch(req):
    import sky
    from sky.skylet.autostop_lib import AutostopWaitFor
    task = sky.Task.from_yaml_config(req["task"])
    # A registry login's three variables go together, an empty one too (ECR's password, which the machine gets itself).
    found = {k: os.environ[k] for k in req.get("secrets", []) if os.environ.get(k) or (k.startswith("SKYPILOT_DOCKER_") and k in os.environ)}
    if found:
        task.update_secrets(found)  # read from the controller's environment, never written to a file
    job_id, _ = sky.get(sky.launch(task, cluster_name=req["cluster"], idle_minutes_to_autostop=req["autostop_minutes"], down=True,
                                   wait_for=AutostopWaitFor.JOBS, retry_until_up=False))
    return {"job_id": job_id}


def down(req):
    import sky
    from sky import exceptions
    try:
        sky.get(sky.down(req["cluster"]))  # terminates: SkyPilot's down never stops a machine
    except exceptions.ClusterDoesNotExist:
        return {"gone": True}
    return {"gone": False}


def endpoint(req):
    import sky
    found = sky.get(sky.endpoints(req["cluster"], req["port"])) or {}
    return {"url": found.get(req["port"]) or found.get(str(req["port"]))}


def autostop(req):
    import sky
    from sky.skylet.autostop_lib import AutostopWaitFor
    sky.get(sky.autostop(req["cluster"], req["autostop_minutes"], wait_for=AutostopWaitFor.JOBS, down=True))
    return {}


def logs(req):
    import io, sky
    out = io.StringIO()
    code = sky.tail_logs(req["cluster"], None, False, tail=req["tail"], output_stream=out)  # the latest job's last lines
    return {"lines": out.getvalue().splitlines()[-req["tail"]:], "exit_code": code}


def volumes(req):
    import sky
    from sky import volumes as sv
    out = []
    for v in sky.get(sv.ls()) or []:
        name = _get(v, "name", "")
        if name.startswith(req["prefix"]):
            out.append({k: _get(v, k) for k in ("name", "type", "cloud", "region", "zone", "size", "status", "launched_at",
                                                "usedby_clusters", "usedby_pods")})
    return out


def volume_apply(req):
    import sky
    from sky import volumes as sv
    if any(_get(v, "name") == req["name"] for v in sky.get(sv.ls()) or []):
        return {"created": False}
    sky.get(sv.apply(sv.Volume(name=req["name"], type=req["type"], infra=req["infra"], size=req["size"], labels=req.get("labels"),
                               config=req.get("config") or {})))
    return {"created": True}


def volume_delete(req):
    import sky
    from sky import volumes as sv
    sky.get(sv.delete([req["name"]]))
    return {}


OPS = {"clusters": clusters, "launch": launch, "down": down, "endpoint": endpoint, "autostop": autostop, "logs": logs,
       "volumes": volumes, "volume_apply": volume_apply, "volume_delete": volume_delete}

if __name__ == "__main__":
    req = json.loads(sys.argv[1])
    try:
        result = OPS[req["op"]](req)
    except Exception as e:
        print("MERITORDER_RESULT " + json.dumps({"error": f"{type(e).__name__}: {e}"[-2000:]}))
        sys.exit(1)
    print("MERITORDER_RESULT " + json.dumps(result))
'''


class SkyError(RuntimeError):
    """A SkyPilot call that failed, with SkyPilot's own reason."""


def _in_process() -> bool:
    try:
        import sky  # noqa: F401
        return True
    except ImportError:
        return False


async def sky_call(req: dict, timeout: float = SKY_TIMEOUT):
    """Run one SkyPilot operation (``clusters``, ``launch``, ``down``, ``endpoint``, ``autostop``): in-process when SkyPilot is importable here, else
    in SkyPilot's own Python with this process's environment (its cloud credentials)."""
    if _in_process():
        ns: dict = {"__name__": "meritorder_sky"}
        exec(_SKY_SCRIPT, ns)  # our own constant script

        def run():
            return ns["OPS"][req["op"]](req)

        try:
            return await asyncio.wait_for(asyncio.to_thread(run), timeout)
        except asyncio.TimeoutError:
            raise SkyError(f"SkyPilot didn't answer within {timeout:.0f}s") from None
        except Exception as e:  # noqa: BLE001 (SkyPilot's own errors carry the reason)
            raise SkyError(f"{type(e).__name__}: {e}") from e
    py = gpu.sky_python()
    if not py:
        raise SkyError("SkyPilot isn't installed where the controller runs (the -gpu image has it in /opt/skypilot)")
    try:
        proc = await asyncio.create_subprocess_exec(py, "-c", _SKY_SCRIPT, json.dumps(req), stdout=asyncio.subprocess.PIPE,
                                                    stderr=asyncio.subprocess.PIPE)
    except OSError as e:
        raise SkyError(f"couldn't run SkyPilot's Python ({py}): {e}") from e
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout)
    except asyncio.TimeoutError:
        proc.kill()
        raise SkyError(f"SkyPilot didn't answer within {timeout:.0f}s") from None
    lines = [line for line in out.decode(errors="replace").splitlines() if line.startswith("MERITORDER_RESULT ")]
    result = json.loads(lines[-1][len("MERITORDER_RESULT "):]) if lines else None
    if isinstance(result, dict) and "error" in result:
        raise SkyError(result["error"])
    if proc.returncode or not lines:
        last = next((x for x in reversed(err.decode(errors="replace").strip().splitlines()) if x.strip()), "")
        raise SkyError(f"SkyPilot exited with status {proc.returncode}" + (f": {last.strip()}" if last else ""))
    return result


# ------------------------------------------------------------------ the fleet: this install's clusters


class Fleet:
    """This install's SkyPilot clusters, listed once a cycle for every deployment, and the sweep that tears down any that
    no deployment wants."""

    def __init__(self, identity: Identity, list_ttl: float = 15.0):
        self.identity = identity
        self.list_ttl = list_ttl
        self._listed: tuple[float, list[dict]] | None = None
        self._good: tuple[float, list[dict]] | None = None  # the last list SkyPilot gave
        self._lock = asyncio.Lock()
        self.swept_once = False
        self.leftovers = False  # the last sweep found clusters to tear down: keep sweeping even with no SkyPilot deployment

    @classmethod
    def from_env(cls) -> Fleet:
        return cls(Identity.from_env())

    async def clusters(self, fresh: bool = False) -> list[dict]:
        """Every cluster of this install, from SkyPilot (refreshed from the clouds, since each has autostop set). When
        SkyPilot can't be asked, the last list for up to STALE_SECONDS, so leases are still renewed meanwhile."""
        async with self._lock:
            if fresh or self._listed is None or time.monotonic() - self._listed[0] > self.list_ttl:
                asked = time.monotonic()
                try:
                    found = await sky_call({"op": "clusters", "prefix": self.identity.prefix, "refresh": True}, timeout=STATUS_TIMEOUT)
                except SkyError as e:
                    if self._good is None or fresh or asked - self._good[0] > STALE_SECONDS:
                        raise
                    log.warning("SkyPilot can't list clusters right now (%s): using the list from %.0fs ago", e, asked - self._good[0])
                    self._listed = (time.monotonic(), self._good[1])  # the other deployments this cycle don't wait again
                    return list(self._good[1])
                self._listed = self._good = (time.monotonic(), found or [])
            return list(self._listed[1])

    def forget(self) -> None:
        """List again next time (after a launch or a teardown)."""
        self._listed = None

    async def down(self, name: str) -> bool:
        """Tear down one cluster; True when it was already gone."""
        if not name.startswith(self.identity.prefix):
            raise ValueError(f"{name} isn't one of this install's clusters")
        try:
            out = await sky_call({"op": "down", "cluster": name})
        finally:
            self.forget()
        return bool((out or {}).get("gone"))

    async def sweep(self, wanted: set[str], store=None) -> list[str]:
        """Tear down every cluster of this install that isn't in ``wanted``, logging each one. Returns their names."""
        found = await self.clusters()
        extra = [c for c in found if c["name"] not in wanted]
        self.swept_once, self.leftovers = True, bool(extra)
        done = []
        for c in extra:
            where = " ".join(x for x in (c.get("cloud"), c.get("region"), c.get("instance_type")) if x)
            log.warning("sweep: %s (%s, %s) isn't wanted by any deployment: tearing it down", c["name"], c.get("status"), where or "?")
            detail = {"cluster": c["name"], "status": c.get("status"), "where": where or None}
            try:
                await self.down(c["name"])
            except Exception as e:  # noqa: BLE001 (tried again next cycle)
                log.error("sweep: couldn't tear down %s: %s", c["name"], e)
                if store is not None:
                    await store.gpu_event("error", deployment_of(c["name"], self.identity) or "-", "controller",
                                          {**detail, "action": "sweep", "error": str(e)[:500]})
                continue
            done.append(c["name"])
            if store is not None:
                await store.gpu_event("sweep", deployment_of(c["name"], self.identity) or "-", "controller",
                                      {**detail, "reason": "no deployment wants it"})
        return done


def deployment_of(cluster: str, identity: Identity) -> str | None:
    """The slug part of one of this install's cluster names ("mo-ab12-gpu-qwen-2" -> "gpu-qwen")."""
    if not cluster.startswith(identity.prefix):
        return None
    rest, _, n = cluster[len(identity.prefix):].rpartition("-")
    return rest if n.isdigit() and rest else None


# ------------------------------------------------------------------ SSH tunnels (where a replica's port can't be reached)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]



TUNNEL_BIND_ENV = "MERITORDER_TUNNEL_BIND"  # the local address the tunnels listen on, if not the one MERITORDER_TUNNEL_HOST names


async def tunnel_bind(host: str) -> str:
    """Where a tunnel listens: only the address gateways reach it at. Loopback for 127.0.0.1 or localhost; else the
    address ``host`` resolves to on this machine (in Compose, the controller container's own address), not every interface.
    MERITORDER_TUNNEL_BIND overrides it; if the name doesn't resolve, every interface, with a warning."""
    if os.environ.get(TUNNEL_BIND_ENV):
        return os.environ[TUNNEL_BIND_ENV]
    if host in ("127.0.0.1", "localhost", "::1"):
        return "127.0.0.1"
    try:
        return await asyncio.to_thread(socket.gethostbyname, host)
    except OSError as e:
        log.warning("couldn't resolve %s=%s (%s); the SSH tunnels listen on every interface. Set %s to the address to use",
                    TUNNEL_HOST_ENV, host, e, TUNNEL_BIND_ENV)
        return "0.0.0.0"

class Tunnel:
    """``ssh -N -L`` from the controller to one replica's gate, through the SSH config SkyPilot writes for each cluster.
    Gateways then reach the replica at MERITORDER_TUNNEL_HOST (default 127.0.0.1) on the tunnel's port."""

    def __init__(self, cluster: str, remote_port: int):
        self.cluster, self.remote_port = cluster, remote_port
        self.proc: asyncio.subprocess.Process | None = None
        self.port: int | None = None

    def url(self) -> str | None:
        if self.port is None:
            return None
        return f"http://{os.environ.get(TUNNEL_HOST_ENV) or '127.0.0.1'}:{self.port}/v1"

    async def ensure(self) -> str | None:
        if self.proc is not None and self.proc.returncode is None:
            return self.url()
        config = Path.home() / ".sky" / "generated" / "ssh" / self.cluster
        if not shutil.which("ssh") or not config.exists():
            return None
        host = os.environ.get(TUNNEL_HOST_ENV) or "127.0.0.1"
        bind = await tunnel_bind(host)
        self.port = self.port or _free_port()
        self.proc = await asyncio.create_subprocess_exec(
            "ssh", "-F", str(config), "-N", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3",
            "-o", "BatchMode=yes", "-L", f"{bind}:{self.port}:127.0.0.1:{self.remote_port}", self.cluster,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
        log.info("%s: SSH tunnel to its port %s on %s:%s", self.cluster, self.remote_port, host, self.port)
        return self.url()

    def close(self) -> None:
        if self.proc is not None and self.proc.returncode is None:
            self.proc.terminate()
        self.proc = None


# ------------------------------------------------------------------ a replica's start, phase by phase

# The marks of a replica's start, in order: the controller's (launch_requested, cluster_up when SkyPilot's launch
# returned, ready when its probe first saw vLLM answer) and the gate's (replica_gate.Gate.startup). Each phase is named
# after the mark that ends it, and runs from the last mark before it that's known.
MARKS = ("launch_requested", "machine_booted", "container_started", "cluster_up", "gate_started", "vllm_started",
         "weights_downloaded", "weights_loaded", "vllm_listening", "ready")
PHASE_NAMES = {
    "machine_booted": "machine boots",  # the cloud finds and boots it
    "container_started": "image pull",  # SkyPilot installs Docker's GPU runtime, pulls the image and starts the container
    "cluster_up": "SkyPilot setup",  # SkyPilot's runtime set up in the container, the task submitted
    "gate_started": "task start",
    "vllm_started": "GPU check",  # the gate reads the driver and drops broken optional packages
    "weights_downloaded": "weights download",
    "weights_loaded": "weights load",
    "vllm_listening": "engine start",  # memory profiling, torch.compile, CUDA graphs
    "ready": "first ready probe",  # up to one controller cycle
}


def startup_phases(marks: dict) -> dict:
    """A replica's start as phases: ``phases`` (name, seconds, in order) between the marks that are known, and
    ``total_seconds`` from the launch request to ready. The controller's marks and the machine's are on two clocks
    (both normally kept by NTP), so a phase across them is clamped at zero."""
    known = [(m, float(marks[m])) for m in MARKS if marks.get(m) is not None]
    phases, last = [], None
    for m, t in known:
        if last is not None:
            skipped = [PHASE_NAMES[x] for x in MARKS[MARKS.index(last[0]) + 1:MARKS.index(m) + 1]]
            phases.append({"phase": " + ".join(skipped), "seconds": round(max(0.0, t - last[1]), 1)})
        last = (m, max(t, last[1]) if last else t)
    ends = (marks.get("launch_requested"), marks.get("ready"))
    total = float(ends[1]) - float(ends[0]) if None not in ends else None
    return {"phases": phases, "total_seconds": round(total, 1) if total is not None else None,
            **{k: marks.get(k) for k in ("download_seconds", "load_seconds") if marks.get(k) is not None}}


def _minutes(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    return f"{seconds:.0f}s" if seconds < 90 else f"{seconds / 60:.1f} min"


# A registry SkyPilot logs in to before it pulls the image (SkyPilot 0.14.2, sky/provision/docker_utils.py): set all
# three where the controller runs to pull from a copy of the image near the GPUs (ECR or Artifact Registry in the same
# region) instead of Docker Hub. For ECR, the password may be empty: the machine logs in with its own IAM role.
DOCKER_LOGIN = ("SKYPILOT_DOCKER_USERNAME", "SKYPILOT_DOCKER_PASSWORD", "SKYPILOT_DOCKER_SERVER")


def launch_secrets() -> list[str]:
    """The variables a launch takes from the controller's environment as SkyPilot secrets (never written to a file)."""
    return ["HF_TOKEN", *(DOCKER_LOGIN if os.environ.get("SKYPILOT_DOCKER_SERVER") else ())]


# ------------------------------------------------------------------ the provisioner


@dataclass
class _Launch:
    task: asyncio.Task
    started: float = field(default_factory=time.time)


class SkyPilotProvisioner:
    """One SkyPilot cluster per replica. Needs SkyPilot where the controller runs, with cloud credentials (`sky check`)."""

    def __init__(self, backend: BackendConfig, cfg: Config, fleet: Fleet):
        self.backend, self.cfg, self.fleet = backend, cfg, fleet
        self.serve = backend.serve
        assert self.serve is not None
        self.slug = slug(backend.name)
        self.want: int | None = None  # replicas the controller asked for (None: not told yet since this process started)
        self._launching: dict[str, _Launch] = {}
        self._failed: dict[str, tuple[float, str]] = {}  # cluster -> (when its launch failed, why)
        self._removing: dict[str, asyncio.Task] = {}
        self._timeouts: dict[str, int] = {}  # cluster -> cycles in a row its port timed out from here
        self._tunnels: dict[str, Tunnel] = {}
        self._expected: tuple[float, dict | None] | None = None
        self._prices: dict[tuple, tuple[float, float | None]] = {}
        self._listed_before: set[str] = set()
        self._ever_ready: set[str] = set()
        self._endpoints: dict[tuple, str] = {}  # (cluster, launched_at) -> the address SkyPilot mapped the gate's port to
        self._autodown: dict[str, tuple[float, asyncio.Task]] = {}  # cluster -> when autodown was last set again, and how
        self._urls: dict[str, str] = {}  # cluster -> its gate's address, as last reached
        self._diagnosed: set[str] = set()  # clusters whose last output has been read since they were launched
        self._events: list[tuple[str, dict]] = []  # GPU activity log entries for the controller to write (drain_events)
        # cluster -> the controller's marks of its start (launch_requested, cluster_up), for the phases at ready. Only a
        # launch this process asked for is timed: after a restart, when it was asked for isn't known.
        self._marks: dict[str, dict] = {}
        self._volume_ready: str | None = None  # the weights cache volume this process has made sure of (its name)
        self._crashes: dict[str, list[float]] = {}  # cluster -> when its vLLM exited after it was ready (within CRASH_WINDOW)
        # False while the deployment drains: a replica that went away isn't launched again (the controller sets it).
        self.relaunch = True
        # When a lease was last renewed (unix seconds), and the last renewal before this cycle's: a replica runs on for
        # lease_minutes after it, which bounds what a gap in the controller's cycles is billed (controller.billed_seconds).
        self.renewed_at: float | None = None
        self.renewed_before: float | None = None

    # --- names, keys, task

    def cluster(self, index: int) -> str:
        return f"{self.fleet.identity.prefix}{self.slug}-{index}"

    def index_of(self, name: str) -> int | None:
        head = f"{self.fleet.identity.prefix}{self.slug}-"
        n = name[len(head):] if name.startswith(head) else ""
        return int(n) if n.isdigit() else None

    def wanted(self) -> set[str]:
        """The clusters this deployment takes care of now, which the sweep leaves alone: every replica up to the count asked
        for, and the ones it's tearing down itself (a scale down)."""
        removing = {n for n, t in self._removing.items() if not t.done()}
        return {self.cluster(i) for i in range(1, (self.want or 0) + 1)} | removing

    def headers(self) -> dict:
        return {"authorization": f"Bearer {self.fleet.identity.api_key(self.backend.name)}"}

    async def placements(self) -> dict | None:
        """The cheapest catalog offer and where launches may go (cached for 6 hours)."""
        if self._expected and time.time() - self._expected[0] < PRICE_TTL:
            return self._expected[1]
        s = self.serve
        found = await asyncio.to_thread(gpu.catalog_placements, s.accelerators, s.clouds, s.regions, s.use_spot)
        self._expected = (time.time(), found)
        return found

    async def expected_price(self) -> dict | None:
        """The cheapest catalog price for one replica where this deployment may run: the same places a launch may go."""
        found = await self.placements()
        return (found or {}).get("best")

    def task_config(self, name: str, pairs: list[list[str]] | None = None) -> dict:
        """The SkyPilot task for one replica. ``pairs`` are the (cloud, region) pairs it may launch in (from the catalog,
        so each region goes only with its own cloud); without regions, the clouds alone; with neither, anywhere."""
        s = self.serve
        upstream = s.port + 1 if s.port + 1 != s.kv_events_port else s.port + 2
        args = gpu.vllm_command(self.backend, self.cfg, port=upstream, host="127.0.0.1")
        # vllm_args are arguments, never shell: each token is quoted, so `;` or `$(...)` reach vLLM as text.
        vllm = " ".join(shlex.quote(a) for a in [*args, *shlex.split(s.vllm_args or "")])
        gate =(Path(__file__).parent / "replica_gate.py").read_text()
        run = ("cat > /tmp/meritorder_gate.py <<'MERITORDER_GATE_EOF'\n" + gate + "MERITORDER_GATE_EOF\n"
               f"exec python3 /tmp/meritorder_gate.py -- {vllm}\n")
        ports = [s.port, s.kv_events_port] if s.kv_events else [s.port]  # the gate (and the KV events, when asked)
        resources: dict = {
            "accelerators": s.accelerators,
            "use_spot": s.use_spot,
            "disk_size": s.disk_size_gb,
            "image_id": f"docker:{s.image}",
            # AWS tags and GCP labels on every instance: bills are matched to the deployment (billing/), and the install
            # says whose cluster it is.
            "labels": {GPU_LABEL: label_value(self.backend.name), INSTALL_LABEL: self.fleet.identity.install_id},
            # Deleted once nothing has run on it for this long: after the gate stops vLLM (the lease ran out), or after a
            # task that failed. Waits for jobs only, so an SSH session (a tunnel) doesn't keep it alive.
            "autostop": {"idle_minutes": s.autostop_minutes, "down": True, "wait_for": "jobs"},
        }
        # The ports go with each cloud that can open them: asking for ports where SkyPilot can't open any (Vast, Mithril)
        # would rule that cloud out, and its replicas are reached over SSH instead. With no clouds named, SkyPilot picks
        # among those that can.
        if s.regions and pairs:
            places = [(c, f"{c}/{r}") for c, r in pairs]
        else:
            places = [(c.lower(), c.lower()) for c in s.clouds]
        if places:
            resources["any_of"] = [{"infra": infra, **({"ports": ports} if sky_clouds.opens_ports(c) else {})} for c, infra in places]
        else:
            resources["ports"] = ports
        envs = {**FAST_START_ENV, "MERITORDER_KEY_SHA256": self.fleet.identity.key_sha256(self.backend.name),
                "MERITORDER_LEASE_SECONDS": str(s.lease_minutes * 60), "MERITORDER_PORT": str(s.port),
                "MERITORDER_UPSTREAM_PORT": str(upstream)}
        task = {"name": name, "resources": resources, "envs": envs, "run": run}
        if s.weights_cache == "volume":
            # The Hugging Face cache, and vLLM's compile cache, on the deployment's volume: the first start fills them.
            task["volumes"] = {CACHE_MOUNT: self.volume_name()}
            envs.update(HF_HOME=f"{CACHE_MOUNT}/huggingface", VLLM_CACHE_ROOT=f"{CACHE_MOUNT}/vllm")
        return task

    # --- the weights cache volume (serve.weights_cache: volume)

    def volume_name(self) -> str:
        """This deployment's weights cache volume: ``mo-<install id>-<deployment>-hf``. The sweep never touches it."""
        return f"{self.fleet.identity.prefix}{self.slug}-hf"

    def volume_spec(self) -> dict:
        s = self.serve
        cloud, _, _ = sky_clouds.volume_infra(s.weights_cache_infra)
        config = {}
        if cloud == "kubernetes":
            # Every replica mounts it at once when there can be more than one: the storage class must allow that.
            config["access_mode"] = "ReadWriteMany" if s.max_replicas > 1 else "ReadWriteOnce"
        return {"name": self.volume_name(), "type": sky_clouds.VOLUME_TYPES[cloud], "infra": s.weights_cache_infra,
                "size": f"{s.weights_cache_gb}GB", "config": config,
                "labels": {GPU_LABEL: label_value(self.backend.name), INSTALL_LABEL: self.fleet.identity.install_id}}

    async def _ensure_volume(self) -> None:
        """Before a launch: make the weights cache volume if it isn't there (once per process; SkyPilot keeps it after).
        It's recorded through the GPU activity log, so its monthly cost is counted (controller.accrue_volumes)."""
        s = self.serve
        if s.weights_cache != "volume" or self._volume_ready == self.volume_name():
            return
        spec = self.volume_spec()
        out = await sky_call({"op": "volume_apply", **spec})
        self._volume_ready = spec["name"]
        monthly = sky_clouds.volume_usd_per_month(s.weights_cache_infra, s.weights_cache_gb, s.weights_cache_usd_per_gb_month)
        detail = {"volume": spec["name"], "infra": s.weights_cache_infra, "size_gb": s.weights_cache_gb,
                  "usd_per_month": monthly, "created": bool((out or {}).get("created"))}
        if detail["created"]:
            log.info("%s: made the weights cache volume %s (%s GB on %s)", self.backend.name, spec["name"], s.weights_cache_gb,
                     s.weights_cache_infra)
        self._events.append(("volume", detail))

    async def _pairs(self) -> list[list[str]] | None:
        if not self.serve.regions:
            return None
        found = await self.placements()
        pairs = (found or {}).get("pairs")
        if not pairs:
            why = gpu.catalog_error() or f"no {self.serve.accelerators} offer in regions {', '.join(self.serve.regions)}"
            raise SkyError(f"can't tell which cloud each region belongs to: {why}")
        return pairs

    # --- launching and tearing down

    def _launch_running(self, name: str) -> bool:
        return name in self._launching and not self._launching[name].task.done()

    def _launch(self, name: str, pairs: list[list[str]] | None) -> None:
        if self._launch_running(name):
            return
        self._ever_ready.discard(name)
        self._diagnosed.discard(name)
        self._urls.pop(name, None)
        self._removing.pop(name, None)
        self._failed.pop(name, None)
        task_cfg = self.task_config(name, pairs)
        timeout = SKY_TIMEOUT + self.serve.provision_timeout_minutes * 60
        marks = self._marks[name] = {"launch_requested": round(time.time(), 1)}

        async def go() -> None:
            log.info("%s: launching %s", self.backend.name, name)
            try:
                await self._ensure_volume()
                await sky_call({"op": "launch", "cluster": name, "task": task_cfg, "autostop_minutes": self.serve.autostop_minutes,
                                "secrets": launch_secrets()}, timeout=timeout)
                marks["cluster_up"] = round(time.time(), 1)
                log.info("%s: %s launched (%.0fs)", self.backend.name, name, marks["cluster_up"] - marks["launch_requested"])
            except Exception as e:  # noqa: BLE001 (shown on the deployment; tried again after LAUNCH_RETRY_SECONDS)
                self._failed[name] = (time.time(), str(e)[-500:])
                log.error("%s: launching %s failed: %s", self.backend.name, name, e)
                await self._diagnose(name, f"launch failed: {str(e)[-300:]}")
                try:  # a launch can fail after the machine exists (an image that won't pull): don't leave it there
                    await self.fleet.down(name)
                except Exception as e2:  # noqa: BLE001 (the sweep and the next cycle try again)
                    log.warning("%s: couldn't tear down %s after its failed launch: %s", self.backend.name, name, e2)
            finally:
                self.fleet.forget()

        self._launching[name] = _Launch(asyncio.ensure_future(go()))

    def _remove(self, name: str) -> None:
        if name in self._removing and not self._removing[name].done():
            return
        if self._launch_running(name):
            self._launching[name].task.cancel()  # if SkyPilot finishes it anyway, the next cycle tears it down
        self._close_tunnel(name)

        async def go() -> None:
            try:
                await self.fleet.down(name)
                log.info("%s: %s torn down", self.backend.name, name)
            except Exception as e:  # noqa: BLE001 (still listed, so the next cycle tries again)
                log.error("%s: tearing down %s failed: %s", self.backend.name, name, e)

        self._removing[name] = asyncio.ensure_future(go())

    def _close_tunnel(self, name: str) -> None:
        t = self._tunnels.pop(name, None)
        if t is not None:
            t.close()

    async def up(self, replicas: int) -> None:
        await self.scale(replicas)

    async def scale(self, replicas: int) -> None:
        """Launch the replicas missing up to ``replicas``, and tear down the ones above it. Launches run in the background;
        status() shows how they're doing."""
        if not _in_process() and not gpu.sky_python():
            raise SkyError("SkyPilot isn't installed where the controller runs (the -gpu image has it in /opt/skypilot)")
        pairs = await self._pairs()
        self.want = replicas
        existing = {c["name"] for c in await self.fleet.clusters()}
        for i in range(1, replicas + 1):
            name = self.cluster(i)
            if name not in existing:
                self._launch(name, pairs)
        for name in existing | set(self._launching):
            i = self.index_of(name)
            if i is not None and i > replicas:
                self._remove(name)

    async def down(self) -> None:
        """Tear down every cluster of this deployment and wait for it. Raises if one couldn't be."""
        self.want = 0
        for launch in self._launching.values():
            launch.task.cancel()  # the SkyPilot request may still finish: the teardown below and the sweep cover it
        self._launching.clear()
        mine = [c["name"] for c in await self.fleet.clusters(fresh=True) if self.index_of(c["name"]) is not None]
        # The replicas that failed: why, before their machines (and vLLM's output on them) are deleted. Bounded inside.
        failed = [n for n in mine if n in self._failed and n not in self._diagnosed]
        if failed:
            await asyncio.gather(*(self._diagnose(n, self._failed[n][1]) for n in failed))
        for name in list(self._tunnels):
            self._close_tunnel(name)
        results = await asyncio.gather(*(self.fleet.down(n) for n in mine), return_exceptions=True)
        errors = [f"{n}: {r}" for n, r in zip(mine, results) if isinstance(r, Exception)]
        self._failed.clear()
        self._listed_before = set()
        if errors:
            raise SkyError("; ".join(errors)[:800])

    def adopt(self, old: SkyPilotProvisioner) -> None:
        """Take over from this deployment's provisioner under the previous config: its launches, teardowns, failures and
        tunnels, and what it was asked for."""
        for attr in ("want", "_launching", "_failed", "_removing", "_timeouts", "_tunnels", "_listed_before", "_ever_ready", "_prices",
                     "_endpoints", "_autodown", "_urls", "_diagnosed", "_events", "_marks", "_crashes", "renewed_at", "renewed_before"):
            setattr(self, attr, getattr(old, attr))

    def restore(self, rec: dict) -> None:
        """After a restart: what the controller last asked for, from its published record, so this process wants the same
        clusters (and the sweep leaves them alone)."""
        if self.want is None:
            self.want = int(rec.get("replicas") or 0) if rec.get("state", gpu.API_ONLY) != gpu.API_ONLY else 0

    # --- every cycle

    async def _renew(self, url: str) -> tuple[dict | None, str | None]:
        """Renew a replica's lease at its gate. Returns (the gate's answer, why not)."""
        base = url[:-3] if url.endswith("/v1") else url
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(10, connect=5)) as c:
                r = await c.post(base + LEASE_PATH, headers=self.headers())
        except httpx.ConnectTimeout:
            return None, "timeout"
        except httpx.HTTPError as e:
            return None, f"{type(e).__name__}"
        if r.status_code != 200:
            return None, f"the gate answered {r.status_code}"
        try:
            return r.json(), None
        except ValueError:
            return None, "the gate's answer wasn't JSON"

    async def _gate_status(self, url: str) -> dict | None:
        """The gate's /meritorder/status (vLLM's last output, its exit status, the driver and CUDA versions), or None."""
        base = url[:-3] if url.endswith("/v1") else url
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(10, connect=5)) as c:
                r = await c.get(base + STATUS_PATH, headers=self.headers())
            found = r.json() if r.status_code == 200 else None
        except (httpx.HTTPError, ValueError):
            return None
        return found if isinstance(found, dict) else None

    async def _diagnose(self, name: str, why: str, url: str | None = None) -> dict:
        """Why a replica failed, read before its machine is deleted: its gate's status, or else ``sky logs --tail``,
        within LOGS_BUDGET seconds in all. Logged here, and queued for the GPU activity log. Returns the entry, whose
        ``reason`` is ``why`` with the driver problem or the last error line added."""
        self._diagnosed.add(name)
        url = url or self._urls.get(name)
        diag: dict = {"cluster": name, "source": None, "tail": []}

        async def fetch() -> None:
            got = await self._gate_status(url) if url else None
            if got is not None:
                diag.update({k: got.get(k) for k in ("exit_code", "uptime_seconds", "driver_version", "cuda_version", "image_cuda",
                                                     "gpus", "driver_problem", "nvidia_smi_error")},
                            source="gate", tail=got.get("tail") or [])
                return
            try:
                out = await sky_call({"op": "logs", "cluster": name, "tail": TAIL_KEPT}, timeout=LOGS_BUDGET - 5)
                diag.update(source="sky logs", tail=(out or {}).get("lines") or [])
            except SkyError as e:
                diag["logs_error"] = str(e)[:300]

        try:
            await asyncio.wait_for(fetch(), LOGS_BUDGET)
        except asyncio.TimeoutError:
            diag["logs_error"] = f"no answer within {LOGS_BUDGET}s"
        tail = diag["tail"] = [replica_gate.keepable(str(x)) for x in diag["tail"] if str(x).strip()][-TAIL_KEPT:]
        reason = why
        if diag.get("driver_problem"):
            reason += f": {diag['driver_problem']}"
        elif tail:
            last = next((x for x in reversed(tail) if re.search(r"Error|Exception|error:|failed|FATAL", x)), tail[-1])
            reason += f" (last output: {last.strip()[:300]})"
        diag["reason"] = reason[:500]
        drivers = f"driver {diag.get('driver_version')}, CUDA {diag.get('cuda_version')}, image CUDA {diag.get('image_cuda')}"
        log.error("%s: %s failed: %s [%s]%s", self.backend.name, name, diag["reason"], drivers,
                  f"\nits last {len(tail)} lines ({diag['source']}):\n" + "\n".join("  | " + x for x in tail) if tail
                  else f" (no output could be read: {diag.get('logs_error') or 'its gate is gone'})")
        self._events.append(("replica_failed", diag))
        return diag

    def drain_events(self) -> list[tuple[str, dict]]:
        """GPU activity log entries since the last call (a failed replica's last output), for the controller to write."""
        out, self._events[:] = list(self._events), []
        return out

    async def _price(self, c: dict) -> float | None:
        key = (c.get("instance_type"), c.get("cloud"), c.get("region"), c.get("zone"), bool(c.get("spot")))
        hit = self._prices.get(key)
        if hit and time.time() - hit[0] < PRICE_TTL:
            return hit[1]
        price = None
        if c.get("instance_type") and c.get("cloud"):
            price = await asyncio.to_thread(gpu.catalog_price, c["instance_type"], c["cloud"], c.get("region"), c.get("zone"), bool(c.get("spot")))
        self._prices[key] = (time.time(), price)
        return price

    def _tunnel_for(self, c: dict) -> bool:
        mode = self.serve.ssh_tunnel
        if mode == "never":
            return False
        if mode == "always" or sky_clouds.reach(c.get("cloud")) == "tunnel" or (str(c.get("cloud")), str(c.get("region"))) in NO_FIREWALL:
            return True
        return self._timeouts.get(c["name"], 0) >= TUNNEL_AFTER

    async def _direct_url(self, c: dict) -> str | None:
        """The gate's address without a tunnel: the head IP and port, or where SkyPilot mapped the port (a Kubernetes
        service, a RunPod port mapping), looked up once per cluster."""
        if sky_clouds.reach(c.get("cloud")) != "endpoint":
            return f"http://{c['head_ip']}:{self.serve.port}/v1"
        key = (c["name"], c.get("launched_at"))
        if key not in self._endpoints:
            try:
                found = (await sky_call({"op": "endpoint", "cluster": c["name"], "port": self.serve.port}, timeout=STATUS_TIMEOUT)
                         or {}).get("url")
            except SkyError as e:
                log.info("%s: SkyPilot has no address for its port %s yet: %s", c["name"], self.serve.port, e)
                return None
            if not found:
                return None
            self._endpoints[key] = (found if "://" in found else f"http://{found}").rstrip("/") + "/v1"
        return self._endpoints[key]

    def _keep_autodown(self, c: dict) -> None:
        """A cluster that's up without autodown (a launch cut short before SkyPilot set it, or someone's `sky autostop`)
        wouldn't delete itself when its lease runs out: set it again, in the background, at most every 10 minutes."""
        stop, down = c.get("autostop"), c.get("to_down")
        if stop is None or (isinstance(stop, (int, float)) and stop >= 0 and down):
            return
        last = self._autodown.get(c["name"])
        if last and (not last[1].done() or time.time() - last[0] < AUTODOWN_RETRY_SECONDS):
            return
        log.warning("%s: up without autodown (autostop %s, down %s): setting it again", c["name"], stop, down)

        async def go() -> None:
            try:
                await sky_call({"op": "autostop", "cluster": c["name"], "autostop_minutes": self.serve.autostop_minutes})
            except Exception as e:  # noqa: BLE001 (tried again later; the sweep and the controller still cover it)
                log.error("%s: couldn't set autodown: %s", c["name"], e)
            finally:
                self.fleet.forget()

        self._autodown[c["name"]] = (time.time(), asyncio.ensure_future(go()))

    async def _replica(self, c: dict, wanted: bool = True) -> dict:
        """One cluster as a replica entry: where it runs, its price, its address, its lease, and whether vLLM answers.
        Only a wanted one has its lease renewed."""
        i = self.index_of(c["name"])
        info = {"id": i, "cluster": c["name"], "status": "PROVISIONING", "cloud": c.get("cloud"), "region": c.get("region"),
                "zone": c.get("zone"), "infra": "/".join(x for x in (c.get("cloud"), c.get("region")) if x) or None,
                "instance_type": c.get("instance_type"), "resources": c.get("resources") or c.get("instance_type"),
                "spot": bool(c.get("spot")), "launched_at": c.get("launched_at"), "endpoint": None,
                "usd_per_hour": await self._price(c)}
        if not wanted:
            info["status"] = "SHUTTING_DOWN"
            return info
        if c.get("status") != "UP" or not c.get("head_ip"):
            if c.get("status") == "STOPPED":
                info["status"] = "STOPPED"
            return info
        if not self._launch_running(c["name"]):
            self._keep_autodown(c)
        tunneled = self._tunnel_for(c)
        if tunneled:
            tunnel = self._tunnels.setdefault(c["name"], Tunnel(c["name"], self.serve.port))
            url = await tunnel.ensure()
            info["tunnel"] = True
            why = "its port can't be reached from here and there's no SSH config for a tunnel"
        elif sky_clouds.reach(c.get("cloud")) == "tunnel":
            url, why = None, f"SkyPilot opens no ports on {c.get('cloud')}, so it's reached only through an SSH tunnel (ssh_tunnel is never)"
        else:
            url, why = await self._direct_url(c), "SkyPilot has no address for its port yet"
        if not url:
            info.update(status="STARTING", message=why)
            return info
        info["endpoint"] = url
        lease, why = await self._renew(url)
        if lease is None:
            self._timeouts[c["name"]] = self._timeouts.get(c["name"], 0) + 1 if why == "timeout" and not tunneled else 0
            info.update(status="STARTING", message=f"the gate doesn't answer yet ({why})")
            return info
        self._timeouts.pop(c["name"], None)
        self._urls[c["name"]] = url
        self.renewed_at = time.time()
        info["lease_expires_in"] = lease.get("expires_in")
        if lease.get("startup") or c["name"] in self._marks:
            info["startup"] = startup_phases({**(lease.get("startup") or {}), **self._marks.get(c["name"], {})})
        if lease.get("vllm") != "running":
            name, message = c["name"], "vLLM has exited; tearing the machine down"
            if name not in self._diagnosed:
                ever = name in self._ever_ready
                why = f"vLLM exited with status {lease.get('exit_code')}" + ("" if ever else " before it was ready")
                diag = await self._diagnose(name, why, url)
                now = time.time()
                crashes = self._crashes[name] = [t for t in self._crashes.get(name, []) if now - t < CRASH_WINDOW] + ([now] if ever else [])
                # Torn down now, not when autostop gets to it (an idle machine still bills). Launched again at once after a
                # first crash; one that never got ready, or crashed twice within CRASH_WINDOW, waits LAUNCH_RETRY_SECONDS.
                if not ever or len(crashes) >= 2:
                    self._failed[name] = (now, diag["reason"] + ("" if not ever else f" ({len(crashes)} crashes within "
                                                                                     f"{CRASH_WINDOW // 60} min: waiting before the next launch)"))
                self._remove(name)
            if name in self._failed:
                message = self._failed[name][1]
            info.update(status="SHUTTING_DOWN", message=message)
            return info
        ready = await gpu._probe(url, self.headers())
        info["status"] = "READY" if ready else "STARTING"
        if not ready:
            info["message"] = "vLLM is loading the model"
        marks = self._marks.get(c["name"])
        if ready and marks is not None and "ready" not in marks:
            marks["ready"] = round(time.time(), 1)  # the first time this process sees it answer
            info["startup"] = startup_phases({**(lease.get("startup") or {}), **marks})
            self._ready_event(c, info)
        return info

    def _ready_event(self, c: dict, info: dict) -> None:
        """A replica is ready: how long its start took, phase by phase, in the log and the GPU activity log."""
        s = info["startup"]
        where = " ".join(x for x in (c.get("cloud"), c.get("region"), c.get("instance_type")) if x)
        log.info("%s: %s ready after %s: %s", self.backend.name, c["name"], _minutes(s.get("total_seconds")),
                 ", ".join(f"{p['phase']} {_minutes(p['seconds'])}" for p in s["phases"]) or "no phases known")
        self._events.append(("replica_ready", {"cluster": c["name"], "where": where or None, **s}))

    async def status(self) -> gpu.Status:
        self.renewed_before = self.renewed_at
        try:
            found = await self.fleet.clusters()
        except SkyError as e:
            # SkyPilot can't be asked, and hasn't been for a while: nothing to renew. Each replica stops when its lease runs out.
            return gpu.Status(failed=not _in_process() and not gpu.sky_python(), message=f"SkyPilot: {e}"[:500])
        mine = {c["name"]: c for c in found if self.index_of(c["name"]) is not None}
        # A cluster that went away by itself before vLLM was ever ready (vLLM crashed, the image wouldn't start, the gate
        # exited): wait before launching it again, as after a failed launch, rather than paying for a crash loop.
        for name in self._listed_before - set(mine):
            if name not in self._ever_ready and name not in self._removing and not self._launch_running(name):
                self._failed.setdefault(name, (time.time(), "it stopped before vLLM was ready (`sky logs` shows its last run)"))
        self._listed_before = set(mine)
        want = self.want or 0
        now = time.time()
        # Bring what runs in line with what's wanted: launch what's missing (a failed launch after a pause), and tear down
        # what's above the count or stopped.
        pairs = None
        for i in range(1, want + 1):
            name = self.cluster(i)
            launching = self._launch_running(name)
            failed = self._failed.get(name)
            if not self.relaunch:
                continue  # draining: a replica that went away (a crash, a preemption) is not launched again
            if (name not in mine or mine[name].get("status") == "INIT") and not launching and (not failed or now - failed[0] >= LAUNCH_RETRY_SECONDS):
                try:
                    pairs = pairs if pairs is not None else await self._pairs()
                except SkyError as e:
                    self._failed[name] = (now, str(e))
                    continue
                self._launch(name, pairs)
        for name, c in mine.items():
            i = self.index_of(name) or 0
            if i > want or c.get("status") == "STOPPED":
                self._remove(name)
        for name in [n for n in self._tunnels if n not in mine]:
            self._close_tunnel(name)

        reps = list(await asyncio.gather(*(self._replica(c, (self.index_of(n) or 0) <= want) for n, c in mine.items())))
        self._ever_ready |= {r["cluster"] for r in reps if r["status"] == "READY"}
        for i in range(1, want + 1):  # wanted replicas with no cluster yet: launching, or failed
            name = self.cluster(i)
            if name in mine:
                continue
            failed = self._failed.get(name)
            if self._launch_running(name):
                reps.append({"id": i, "cluster": name, "status": "PROVISIONING", "endpoint": None, "message": "SkyPilot is launching it"})
            elif failed:
                reps.append({"id": i, "cluster": name, "status": "FAILED", "endpoint": None, "message": failed[1]})
        reps.sort(key=lambda r: r["id"] or 0)
        ready = [r for r in reps if r["status"] == "READY"]
        failures = [self._failed[self.cluster(i)][1] for i in range(1, want + 1) if self.cluster(i) in self._failed]
        alive = [r for r in reps if r["status"] not in ("FAILED",)]
        # Everything wanted failed to launch, and nothing is up or on the way: the controller gives up and cools down.
        failed = want > 0 and not alive and len(failures) == want
        expected = await self.expected_price() if any(r.get("usd_per_hour") is None for r in reps) else None
        cost = 0.0
        for r in reps:
            if r["status"] == "FAILED":
                continue
            price = r.get("usd_per_hour")
            cost += price if price is not None else float((expected or {}).get("usd_per_hour") or self.backend.gpu_cost_per_hour)
        message = "; ".join(f"replica {r['id']}: {r['message']}" for r in reps if r.get("message") and r["status"] != "READY")[:500] or None
        return gpu.Status(ready=bool(ready), endpoint=ready[0]["endpoint"] if ready else None, failed=failed, replicas=len(ready),
                          replicas_info=reps, message=message, usd_per_hour=round(cost, 4))


# ------------------------------------------------------------------ weights cache volumes, for `meritorder gpu volumes`


async def list_volumes(store) -> list[dict]:
    """This install's weights cache volumes: what SkyPilot has, with what the controller recorded (deployment, size,
    monthly cost), and those recorded that SkyPilot no longer lists (``in_skypilot`` false)."""
    identity = await load_identity(store)
    recorded = await store.get_setting(gpu.VOLUMES_SETTING) or {}
    found = await sky_call({"op": "volumes", "prefix": identity.prefix}) or []
    names = {v["name"] for v in found}
    out = [{**(recorded.get(v["name"]) or {}), **v, "in_skypilot": True} for v in found]
    out += [{"name": n, **r, "in_skypilot": False} for n, r in recorded.items() if n not in names]
    return sorted(out, key=lambda v: v["name"])


async def delete_volume(store, name: str, by: str = "cli") -> None:
    """Delete one of this install's weights cache volumes: the only way one goes (the sweep never touches volumes).
    Refused while a replica uses it; the next start of its deployment makes it again, empty."""
    identity = await load_identity(store)
    if not name.startswith(identity.prefix) or not name.endswith("-hf"):
        raise ValueError(f"{name} isn't one of this install's weights cache volumes ({identity.prefix}<deployment>-hf)")
    found = {v["name"]: v for v in await sky_call({"op": "volumes", "prefix": identity.prefix}) or []}
    used = (found.get(name) or {}).get("usedby_clusters") or []
    if used:
        raise ValueError(f"{name} is in use by {', '.join(used)}: stop that deployment first")
    if name in found:
        await sky_call({"op": "volume_delete", "name": name})
    recorded = dict(await store.get_setting(gpu.VOLUMES_SETTING) or {})
    gone = recorded.pop(name, None) or {}
    await store.set_setting(gpu.VOLUMES_SETTING, recorded)
    await store.gpu_event("volume_deleted", gone.get("deployment") or "-", by, {"volume": name, "in_skypilot": name in found})
