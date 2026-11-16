"""GPU tier: provisioners, vLLM metrics, and the break-even state machine.

Lifecycle per self-hosted deployment (docs/05-gpu-autoscaling.md):

    api_only -> provisioning -> warm <-> draining -> api_only

The decision function is pure (``decide``) so it can be tested without clouds.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from .classifier import parse_floor
from .config import BackendConfig, Config, ModelConfig, Safeguards, ServeConfig

log = logging.getLogger("meritorder.gpu")

API_ONLY, PROVISIONING, WARM, DRAINING = "api_only", "provisioning", "warm", "draining"


# ------------------------------------------------------------------ demand


@dataclass
class Demand:
    usd_per_hour: float  # what addressable traffic costs today, per hour
    tokens_per_second: float  # its throughput (peak of the 1h and 15m windows)


def could_serve(model: ModelConfig, row: dict, cfg: Config) -> bool:
    """Whether this model could have served a ledger row's traffic: its quality meets the floor, or it was asked for."""
    floor = parse_floor(row.get("floor") or "")
    if floor is not None:
        return model.q(floor.task) >= floor.level
    requested = cfg.model(row.get("requested_model") or "")
    return (requested is not None and requested.name == model.name) or row.get("served_model") == model.name


def addressable(model: ModelConfig, rows: list[dict], cfg: Config) -> tuple[float, int]:
    """Spend and tokens of traffic this model could have served (its quality meets the floor)."""
    mine = [r for r in rows if could_serve(model, r, cfg)]
    return sum(r["cost"] for r in mine), sum(r["tokens"] for r in mine)


# ------------------------------------------------------------------ decisions


@dataclass
class Status:
    ready: bool = False
    endpoint: str | None = None
    failed: bool = False
    replicas: int = 0
    # Where each replica runs: cloud, region, zone, resources, spot, status, launched_at (unix seconds), endpoint.
    replicas_info: list[dict] = field(default_factory=list)
    message: str | None = None  # what the provisioner says is holding it up, or why it failed (shown and logged)
    # What its machines cost an hour right now, when the provisioner counts them itself (SkyPilot: every cluster that
    # exists or is being launched, at its catalog price). None: the controller works it out from the replicas.
    usd_per_hour: float | None = None


@dataclass
class Action:
    kind: str  # "up" | "scale" | "down"
    replicas: int = 0


VOLUMES_SETTING = "gpu_volumes"  # the weights cache volumes the controller made, kept until `meritorder gpu volumes delete`
COLD_START_SAMPLES = 10 # the expected cold start is the average of a deployment's last this many starts


def start_factor(serve: ServeConfig, cold_start_minutes: float) -> float:
    """What a replica that isn't running yet costs per hour it serves, as a multiple of its price: it's paid from the
    launch, through ``cold_start_minutes`` without serving, and then kept at least ``hold_minutes``, over which that
    cold start must pay back."""
    hold = max(serve.hold_minutes, 1)
    return (max(cold_start_minutes, 0.0) + hold) / hold


def expected_cold_start(serve: ServeConfig, rec: dict | None) -> tuple[float, str]:
    """Minutes a start is expected to take, and where that comes from: the average of this deployment's last starts
    (``cold_start`` in its record), else ``serve.cold_start_minutes``. A server you run yourself has none."""
    if serve.provisioner == "static":
        return 0.0, "static"
    cs = (rec or {}).get("cold_start") or {}
    if cs.get("avg_seconds"):
        return float(cs["avg_seconds"]) / 60, "measured"
    return float(serve.cold_start_minutes), "configured"


def record_cold_start(rec: dict, seconds: float, detail: dict | None = None, now: float | None = None) -> dict:
    """Add one start's time to a deployment's record: the last COLD_START_SAMPLES and their average."""
    cs = dict(rec.get("cold_start") or {})
    samples = [float(x) for x in cs.get("samples") or []][-(COLD_START_SAMPLES - 1):] + [round(float(seconds), 1)]
    cs.update(samples=samples, avg_seconds=round(sum(samples) / len(samples), 1), last={**(detail or {}), "total_seconds": round(seconds, 1),
                                                                                       "at": now or time.time()})
    rec["cold_start"] = cs
    return cs


def desired_replicas(serve: ServeConfig, gpu_cost_per_hour: float, capacity_tps: float, demand: Demand, running: int = 0,
                     cold_start_minutes: float = 0.0) -> int:
    """Replicas worth running: enough for throughput, but each one must pay for itself. The ``running`` ones (warm or
    starting: their cold start is paid) must cover their price plus the margin; each one more must also pay back its
    cold start over hold_minutes (start_factor), so a burst too short to cover a start doesn't start one. With no demand
    at all, none (min_replicas aside), whatever the price: a free replica is worth running only while someone asks."""
    if demand.usd_per_hour <= 0 and demand.tokens_per_second <= 0:
        return max(serve.min_replicas, 0)
    by_load = math.ceil(demand.tokens_per_second / max(capacity_tps * serve.target_utilization, 1e-9)) if demand.tokens_per_second else 0
    per = gpu_cost_per_hour * (1 + serve.margin)
    if per > 0:
        kept = min(running, math.floor(demand.usd_per_hour / per))
        extra = math.floor(max(demand.usd_per_hour - kept * per, 0) / (per * start_factor(serve, cold_start_minutes)))
        affordable = kept + extra if kept == running else kept
    else:
        affordable = serve.max_replicas
    want = min(max(by_load, 1 if affordable >= 1 else 0), affordable)
    return max(serve.min_replicas, min(serve.max_replicas, want))


def decide(rec: dict, serve: ServeConfig, desired: int, status: Status, now: float,
           force: str | None = None, drain: bool = True, why: str | None = None) -> tuple[dict, Action | None]:
    """Advance one deployment's state. Returns (new record, action to take).

    ``force="on"`` runs ``desired`` replicas now, skipping the demand timers and any cooldown.
    ``force="off"`` stops: through draining when ``drain``, otherwise immediately.
    """
    rec = dict(rec or {})
    state = rec.setdefault("state", API_ONLY)
    rec.setdefault("since", now)
    action: Action | None = None

    def move(new_state: str, reason: str) -> None:
        if new_state == PROVISIONING and state == API_ONLY:
            rec.update(started_at=now, run_cost=0.0, warm_at=None)
        if new_state == WARM and not rec.get("warm_at"):
            rec["warm_at"] = now
        if new_state == API_ONLY and rec.get("started_at") is not None:
            rec["last_run"] = {"started_at": rec["started_at"], "ended_at": now, "cost": round(rec.get("run_cost") or 0, 2)}
            rec.update(started_at=None, warm_at=None, run_cost=0.0, replicas_info=[])
        rec.update(state=new_state, since=now, reason=reason)
        log.info("deployment -> %s: %s", new_state, reason)

    def stop_now(reason: str) -> None:
        nonlocal action
        action = Action("down")
        rec.update(endpoint=None, replicas=0, under_since=None, over_since=None)
        move(API_ONLY, reason)

    if force == "off":
        if state == API_ONLY:
            rec["over_since"] = None
        elif state == PROVISIONING or (state in (WARM, DRAINING) and not drain):
            stop_now(why or "stopped")
        elif state == WARM:
            move(DRAINING, why or "stopping after warm sessions finish")
        elif now - rec["since"] >= serve.drain_minutes * 60:
            stop_now(why or "drained")
        return rec, action

    if state == API_ONLY:
        if desired >= 1 and (force == "on" or now >= rec.get("cooldown_until", 0)):
            if rec.get("over_since") is None:  # a cycle without enough demand leaves it None, not absent
                rec["over_since"] = now
            if force == "on" or serve.min_replicas > 0 or now - rec["over_since"] >= serve.scale_up_after_minutes * 60:
                action = Action("up", desired)
                rec.update(replicas=desired, over_since=None)
                move(PROVISIONING, why or f"break-even for {desired} replica(s)")
        else:
            rec["over_since"] = None

    elif state == PROVISIONING:
        if status.ready and status.endpoint:
            rec["endpoint"] = status.endpoint
            move(WARM, "replicas ready")
        elif status.failed or now - rec["since"] > serve.provision_timeout_minutes * 60:
            action = Action("down")
            rec.update(endpoint=None, cooldown_until=now + 1800, replicas=0)
            what = "provisioning failed" if status.failed else f"provisioning timed out after {serve.provision_timeout_minutes} min"
            move(API_ONLY, what + (f" ({status.message})" if status.message else "") + "; cooling down 30 min")

    elif state == WARM:
        if not status.ready:
            rec["endpoint"] = None
            move(PROVISIONING, "lost readiness")
        elif desired >= 1:
            rec["under_since"] = None
            rec["endpoint"] = status.endpoint
            if desired != rec.get("replicas"):
                action = Action("scale", desired)
                rec["replicas"] = desired
        else:
            rec.setdefault("under_since", None)
            rec["under_since"] = rec["under_since"] or now
            held = now - (rec.get("warm_at") or rec["since"]) >= serve.hold_minutes * 60
            under = now - rec["under_since"] >= serve.scale_up_after_minutes * 60
            if held and under:
                move(DRAINING, "demand below break-even")

    elif state == DRAINING:
        if desired >= 1 and status.ready:
            rec["under_since"] = None
            move(WARM, "demand returned")
        elif now - rec["since"] >= serve.drain_minutes * 60:
            action = Action("down")
            rec.update(endpoint=None, replicas=0, under_since=None)
            move(API_ONLY, "drained")
    return rec, action


@dataclass
class Plan:
    """What the controller should run for one deployment after controls and safeguards."""

    desired: int
    force: str | None = None  # "on" | "off"
    drain: bool = True
    note: str | None = None  # why this differs from what demand alone would do


def active_control(controls: dict, backend: str, now: float) -> dict | None:
    c = (controls.get("deployments") or {}).get(backend)
    if not c or c.get("mode", "auto") == "auto":
        return None
    if c.get("until") and c["until"] <= now:
        return None
    return c


def plan_fleet(items: list[dict], controls: dict, guards: Safeguards, gpu_spent_month: float, now: float | None = None) -> dict[str, Plan]:
    """Apply admin controls and safeguards to demand-based sizing, for every deployment at once.

    ``items``: one dict per deployment with ``name``, ``desired`` (from demand), ``price`` (per replica-hour: what the
    caps count), ``max_replicas``, ``value`` (addressable demand in $/h, used to rank who gets capacity under a cap),
    and ``billed`` (False for a server you run yourself, which has no cloud bill). A billed deployment with no price
    (0 or unknown) is never started, pinned or not: no budget or cap could hold it, and nothing it costs is counted.
    """
    now = now or time.time()
    plans: dict[str, Plan] = {}
    for it in items:
        c = active_control(controls, it["name"], now)
        if controls.get("paused"):
            who = controls.get("paused_by") or "an admin"
            plans[it["name"]] = Plan(0, "off", bool(controls.get("drain", False)),
                                     f"all GPUs stopped by {who}" + (f": {controls['reason']}" if controls.get("reason") else ""))
        elif c and c["mode"] == "off":
            plans[it["name"]] = Plan(0, "off", bool(c.get("drain", True)), f"stopped by {c.get('by') or 'an admin'}" + (f": {c['reason']}" if c.get("reason") else ""))
        elif c and c["mode"] == "on":
            n = max(1, min(int(c.get("replicas") or 1), it["max_replicas"]))
            plans[it["name"]] = Plan(n, "on", note=f"pinned to {n} by {c.get('by') or 'an admin'}" + (f": {c['reason']}" if c.get("reason") else ""))
        else:
            plans[it["name"]] = Plan(it["desired"])
        if it.get("billed", True) and not (it["price"] or 0) > 0 and plans[it["name"]].force != "off":
            plans[it["name"]] = Plan(0, None, True, "no price for a replica (0 or unknown), so no GPU budget or cap could hold "
                                                    "it: not started (set gpu_cost_per_hour, or a priced node type)")

    if guards.gpu_budget_usd_month is not None and gpu_spent_month >= guards.gpu_budget_usd_month:
        for name, p in plans.items():
            if p.force != "off":
                plans[name] = Plan(0, "off", True, f"GPU budget of ${guards.gpu_budget_usd_month:,.2f} this month is used up")
        return plans

    if guards.max_gpu_replicas is None and guards.max_gpu_usd_per_hour is None:
        return plans
    replicas_left = guards.max_gpu_replicas if guards.max_gpu_replicas is not None else 10**9
    usd_left = guards.max_gpu_usd_per_hour if guards.max_gpu_usd_per_hour is not None else float("inf")
    # Pins first, then the deployments whose demand best covers their cost.
    order = sorted(items, key=lambda it: (plans[it["name"]].force != "on", -(it["value"] / it["price"] if it["price"] else float("inf"))))
    for it in order:
        p = plans[it["name"]]
        if p.desired <= 0 or p.force == "off":
            continue
        afford = int(usd_left // it["price"]) if it["price"] > 0 and usd_left != float("inf") else p.desired
        grant = max(0, min(p.desired, replicas_left, afford))
        if grant < p.desired:
            note = f"held to {grant} of {p.desired} replica(s) by the fleet limit"
            plans[it["name"]] = Plan(grant, p.force if grant else "off", True, note) if grant else Plan(0, "off", True, note)
        replicas_left -= grant
        usd_left -= grant * it["price"]
    return plans


# ------------------------------------------------------------------ provisioners


def served_name(backend: BackendConfig, cfg: Config) -> str:
    serve = backend.serve
    assert serve is not None
    if serve.served_model_name:
        return serve.served_model_name
    model = cfg.model(serve.model)
    for r in model.routes if model else []:
        if r.backend == backend.name:
            return r.model
    return model.hf_id if model and model.hf_id else serve.model


def vllm_command(backend: BackendConfig, cfg: Config, port: int = 8000, host: str = "0.0.0.0") -> list[str]:
    """`vllm serve` for a deployment, the same on every provisioner: prefix caching and priority scheduling always on
    (free upgrades are sent at a lower priority), tensor parallel across the replica's GPUs. ``vllm_args`` come after.
    On SkyPilot it listens on the loopback address only, behind the replica's gate (replica_gate.py)."""
    serve = backend.serve
    assert serve is not None
    model = cfg.model(serve.model)
    if model is None or not model.hf_id:
        raise ValueError(f"model {serve.model!r} needs an hf_id to self-host")
    args = [
        "vllm", "serve", model.hf_id,
        "--host", host, "--port", str(port),
        "--served-model-name", served_name(backend, cfg),
        "--tensor-parallel-size", str(gpu_count(serve.accelerators)),
        "--enable-prefix-caching",
        "--scheduling-policy", "priority",
        # Replies say how many prompt tokens came from the cache, so the console can check the warm-cache estimates.
        "--enable-prompt-tokens-details",
    ]
    if serve.load_format != "auto":
        args += ["--load-format", serve.load_format]  # docs/05, "Faster starts": not every model loads every way
    if serve.kv_events:
        # Publish block stored / removed events (tagged with the session the gateway sends in X-Session-ID) for
        # the controller's KV-event subscriber (kv_events.py).
        args += ["--kv-events-config", json.dumps({"enable_kv_cache_events": True, "publisher": "zmq",
                                                    "endpoint": f"tcp://*:{serve.kv_events_port}"}, separators=(",", ":"))]
    return args


def gpu_count(accelerators: str) -> int:
    """'H100:4' -> 4; 'L4' -> 1."""
    _, _, count = accelerators.partition(":")
    return int(count) if count.strip().isdigit() and int(count) > 0 else 1


async def _probe(endpoint: str, headers: dict | None = None) -> bool:
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.get(endpoint.rstrip("/") + "/models", headers=headers)
            return r.status_code == 200
    except httpx.HTTPError:
        return False


def replica_url(endpoint: str | None) -> str | None:
    """A replica's OpenAI-compatible base URL from what a provisioner or the config gives ("3.4.5.6:8000",
    "http://host:8000", "http://host:8000/v1"): with a scheme and ending in /v1."""
    if not endpoint or endpoint in ("-", "None"):
        return None
    url = endpoint.strip().rstrip("/")
    if "://" not in url:
        url = f"http://{url}"
    return url if url.endswith("/v1") else f"{url}/v1"


class StaticProvisioner:
    """A server you run yourself. Up/down are no-ops; readiness is probed.

    ``static_urls`` lists replicas the gateway can reach one by one (for replica affinity); ``static_url`` is the
    address in front of them, if any. Each is probed; the deployment is ready while any answers."""

    def __init__(self, backend: BackendConfig, cfg: Config):
        self.backend = backend
        serve = backend.serve
        self.url = (serve.static_url if serve else None) or (None if serve and serve.static_urls else backend.base_url)
        self.urls = [u for u in (replica_url(x) for x in (serve.static_urls if serve else [])) if u]

    async def status(self) -> Status:
        if not self.url and not self.urls:
            return Status(failed=True)
        serve = self.backend.serve
        where = (serve.location if serve else None) or "your server"
        res = serve.accelerators if serve else None
        if not self.urls:
            ready = await _probe(self.url)
            info = [{"id": 1, "status": "READY" if ready else "NOT_READY", "infra": where, "resources": res, "spot": False, "endpoint": self.url}]
            return Status(ready=ready, endpoint=self.url if ready else None, replicas=1 if ready else 0, replicas_info=info)
        up = await asyncio.gather(*(_probe(u) for u in self.urls))
        info = [{"id": i + 1, "status": "READY" if ok else "NOT_READY", "infra": where, "resources": res, "spot": False, "endpoint": u}
                for i, (u, ok) in enumerate(zip(self.urls, up))]
        live = [u for u, ok in zip(self.urls, up) if ok]
        front = self.url if self.url and await _probe(self.url) else None
        endpoint = front or (live[0] if live else None)
        return Status(ready=bool(endpoint), endpoint=endpoint, replicas=len(live), replicas_info=info)

    async def expected_price(self) -> dict | None:
        return None  # your own hardware: the configured gpu_cost_per_hour is the price

    async def up(self, replicas: int) -> None:
        log.info("%s: static deployment, nothing to start", self.backend.name)

    async def scale(self, replicas: int) -> None:
        return None

    async def down(self) -> None:
        log.info("%s: static deployment, nothing to stop", self.backend.name)


# Catalog lookups run in SkyPilot's Python: in-process when it's installed alongside Merit Order, otherwise
# in SkyPilot's own environment (MERITORDER_SKY_PYTHON, or the interpreter behind the `sky` command), which
# keeps SkyPilot's many dependencies out of the gateway's. Prices are SkyPilot's catalog list prices.
_CATALOG_SCRIPT = r'''
import json, math, sys


def lookup(req):
    from sky import catalog
    if req["op"] == "price":
        return float(catalog.get_hourly_cost(req["instance"], req["spot"], req["region"], req["zone"], clouds=req["cloud"].lower()))
    name, _, count = req["accelerators"].partition(":")
    found = catalog.list_accelerators(name_filter=name, quantity_filter=int(count or 1),
                                      clouds=[c.lower() for c in req["clouds"]] or None, all_regions=True, case_sensitive=False)
    best, pairs = None, []
    for offers in found.values():
        for o in offers:
            if req["regions"] and o.region not in req["regions"]:
                continue
            p = o.spot_price if req["spot"] else o.price
            if p is None or (isinstance(p, float) and math.isnan(p)) or p <= 0:
                continue
            if [str(o.cloud).lower(), o.region] not in pairs:
                pairs.append([str(o.cloud).lower(), o.region])
            if best is None or p < best["usd_per_hour"]:
                best = {"usd_per_hour": float(p), "cloud": str(o.cloud), "region": o.region, "instance_type": o.instance_type, "spot": req["spot"]}
    # Where a launch may go: the (cloud, region) pairs the price comes from, so each region goes only with its own cloud.
    return {"best": best, "pairs": sorted(pairs)} if req["op"] == "placements" else best


def offers(req):
    """Every GPU offer in these clouds, reduced to the cheapest on-demand and spot price per (cloud, GPU, count)."""
    import os
    from datetime import datetime, timezone
    from sky import catalog
    from sky.catalog import common
    clouds = [c.lower() for c in req["clouds"]]
    if req.get("refresh"):
        for c in clouds:  # re-download the cloud's catalog now instead of on SkyPilot's own schedule
            try:
                common.read_catalog(c + "/vms.csv", pull_frequency_hours=0)._load_df()
            except Exception:
                pass
    best = {}
    for name, found in catalog.list_accelerators(gpus_only=True, clouds=clouds, all_regions=True, require_price=True).items():
        for o in found:
            count = float(o.accelerator_count or 0)
            if count < 1 or count != int(count) or count > 8:
                continue
            key = (str(o.cloud).lower(), name, int(count))
            offer = best.setdefault(key, {"cloud": key[0], "accelerator": name, "count": key[2], "on_demand": None, "spot": None})
            for kind, price in (("on_demand", o.price), ("spot", o.spot_price)):
                if price is None or price != price or price <= 0:
                    continue
                if offer[kind] is None or price < offer[kind]["usd_per_hour"]:
                    text = lambda v: v if isinstance(v, str) and v else None  # catalogs' blanks arrive as NaN
                    offer[kind] = {"usd_per_hour": round(float(price), 4), "region": text(o.region), "instance_type": text(o.instance_type)}
    updated = {}
    for c in clouds:
        path = os.path.join(common._ABSOLUTE_VERSIONED_CATALOG_DIR, c, "vms.csv")
        if os.path.exists(path):
            updated[c] = datetime.fromtimestamp(os.path.getmtime(path), timezone.utc).isoformat(timespec="seconds")
    import sky
    return {"offers": [o for o in best.values() if o["on_demand"] or o["spot"]], "catalog_updated": updated,
            "skypilot": getattr(sky, "__version__", None)}


if __name__ == "__main__":
    req = json.loads(sys.argv[1])
    print(json.dumps(offers(req) if req["op"] == "offers" else lookup(req)))
'''


def sky_python() -> str | None:
    """The Python interpreter SkyPilot is installed in, if we can find one."""
    explicit = os.environ.get("MERITORDER_SKY_PYTHON")
    if explicit:
        return explicit
    sky = shutil.which("sky")
    if not sky:
        return None
    # The interpreter next to the `sky` executable: venvs, uv tools, pipx, and our image's /opt/skypilot.
    # Check the link's own directory and, if it's a symlink (e.g. ~/.local/bin/sky), the target's.
    for d in dict.fromkeys([Path(sky).parent, Path(os.path.realpath(sky)).parent]):
        for name in ("python", "python3"):
            if os.access(d / name, os.X_OK):
                return str(d / name)
    try:
        head = Path(sky).read_text(errors="replace").splitlines()[:3]
    except OSError:
        return None
    return _shebang_python(head)


def _shebang_python(lines: list[str]) -> str | None:
    """The interpreter a console script launches, from its first lines.

    Handles ``#!/path/python``, ``#!/usr/bin/env python3``, and the ``/bin/sh`` launcher pip writes
    when the interpreter's path contains spaces (``'\'\'\'exec' "/path with space/python" "$0" "$@"``).
    """
    if not lines or not lines[0].startswith("#!"):
        return None
    words = lines[0][2:].split()
    if words and words[0].endswith("/env") and len(words) > 1:
        return shutil.which(words[-1])
    if words and "python" in Path(words[0]).name:
        return words[0]
    for line in lines[1:]:
        m = re.search(r"exec'?\s+\"([^\"]+)\"", line)
        if m:
            return m.group(1)
    return None


# Catalog lookups never need cloud credentials, and must not pick up whatever account the shell happens to have:
# no AWS variable but the region reaches SkyPilot, and no AWS config or credentials file.


def _catalog_env() -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("AWS_") or k in ("AWS_REGION", "AWS_DEFAULT_REGION")}
    return {**env, "AWS_CONFIG_FILE": os.devnull, "AWS_SHARED_CREDENTIALS_FILE": os.devnull}


def sky_available() -> bool:
    """Whether SkyPilot can be reached for catalog lookups (in-process or in its own environment)."""
    try:
        import sky  # noqa: F401
        return True
    except ImportError:
        py = sky_python()
        return bool(py and os.access(py, os.X_OK))


CATALOG_TIMEOUT_SECONDS = 600  # a lookup alone takes about a minute (two with downloads); this only stops a hung SkyPilot
_last_catalog_error: str | None = None


def catalog_error() -> str | None:
    """Why the last catalog lookup failed, for the message that says it did (None after one that worked)."""
    return _last_catalog_error


def _failed(reason: str) -> None:
    global _last_catalog_error
    _last_catalog_error = reason[-300:]
    log.warning("SkyPilot catalog lookup failed: %s", reason[-500:])


def _catalog(req: dict):
    global _last_catalog_error
    try:
        import sky  # noqa: F401  (installed alongside Merit Order)
    except ImportError:
        py = sky_python()
        if not py:
            return None
        import subprocess

        try:
            out = subprocess.run([py, "-c", _CATALOG_SCRIPT, json.dumps(req)], capture_output=True, text=True,
                                 timeout=CATALOG_TIMEOUT_SECONDS, env=_catalog_env())
        except subprocess.TimeoutExpired:
            return _failed(f"it took longer than {CATALOG_TIMEOUT_SECONDS // 60} minutes")
        except OSError as e:
            return _failed(f"couldn't run SkyPilot's Python ({py}): {e}")
        if out.returncode:
            last = next((line for line in reversed(out.stderr.strip().splitlines()) if line.strip()), "")
            return _failed(f"SkyPilot exited with status {out.returncode}" + (f": {last.strip()}" if last else ""))
        lines = out.stdout.strip().splitlines()
        _last_catalog_error = None
        return json.loads(lines[-1]) if lines else None
    ns: dict = {"__name__": "meritorder_sky_catalog"}
    try:
        exec(_CATALOG_SCRIPT, ns)  # our own constant script, run in-process when SkyPilot is importable
        found = ns["offers"](req) if req["op"] == "offers" else ns["lookup"](req)
    except Exception as e:  # noqa: BLE001 (SkyPilot's own errors: the caller says the lookup failed, and why)
        return _failed(f"{type(e).__name__}: {e}")
    _last_catalog_error = None
    return found


def catalog_price(instance: str, cloud: str, region: str | None, zone: str | None, spot: bool) -> float | None:
    """Hourly list price of one instance from SkyPilot's catalog, or None when SkyPilot isn't available."""
    return _catalog({"op": "price", "instance": instance, "cloud": cloud, "region": region, "zone": zone, "spot": spot})


def catalog_offers(clouds: list[str], refresh: bool = False) -> dict | None:
    """Every GPU offer in these clouds from SkyPilot's catalog: the cheapest on-demand and spot price per
    (cloud, GPU, count), with when SkyPilot last downloaded each cloud's catalog. ``refresh`` downloads them first."""
    return _catalog({"op": "offers", "clouds": clouds, "refresh": refresh})


def catalog_cheapest(accelerators: str, clouds: list[str], regions: list[str], spot: bool) -> dict | None:
    """Cheapest catalog offer for one replica's accelerators across the allowed clouds and regions."""
    return _catalog({"op": "cheapest", "accelerators": accelerators, "clouds": clouds, "regions": regions, "spot": spot})


def catalog_placements(accelerators: str, clouds: list[str], regions: list[str], spot: bool) -> dict | None:
    """The cheapest offer (``best``) and every (cloud, region) pair with a price for these accelerators (``pairs``), so a
    launch goes only where the price came from, and a region only with the cloud it belongs to."""
    return _catalog({"op": "placements", "accelerators": accelerators, "clouds": clouds, "regions": regions, "spot": spot})


def make_provisioner(backend: BackendConfig, cfg: Config, fleet=None):
    """The provisioner for a deployment. ``fleet`` (skypilot.Fleet) is this install's SkyPilot clusters, shared by every
    SkyPilot deployment so they're listed once a cycle; without one, a fleet from the environment is made."""
    if backend.serve is None:
        raise ValueError(f"backend {backend.name} has no serve config")
    if backend.serve.provisioner == "static":
        return StaticProvisioner(backend, cfg)
    if backend.serve.provisioner == "kubernetes":
        from .kube import KubernetesProvisioner

        return KubernetesProvisioner(backend, cfg, install=fleet.identity.install_id if fleet is not None else None)
    from .skypilot import Fleet, SkyPilotProvisioner

    return SkyPilotProvisioner(backend, cfg, fleet or Fleet.from_env())


# ------------------------------------------------------------------ vLLM metrics

_METRIC = re.compile(r"^(vllm:[a-z_]+)(?:\{[^}]*\})?\s+([0-9.eE+-]+)$", re.MULTILINE)
_CACHE_CONFIG = re.compile(r"^vllm:cache_config_info\{([^}]*)\}", re.MULTILINE)
_LABEL = re.compile(r'(\w+)="([^"]*)"')
_STARTED = re.compile(r"^process_start_time_seconds\s+([0-9.eE+-]+)$", re.MULTILINE)


def parse_vllm_metrics(text: str) -> float:
    """Utilization in [0, 1]: KV-cache usage, pushed up when requests are queueing."""
    return vllm_load(text)["utilization"]


def vllm_load(text: str) -> dict:
    """Utilization plus the running and waiting request counts, summed over the engines behind the endpoint.

    When vLLM reports them, also what its KV cache can hold and how much it has written, which is what pushes old
    prefixes out (affinity.estimated_share):

    - ``kv_capacity_tokens``: GPU blocks times block size (``vllm:cache_config_info``);
    - ``kv_usage``: the share of blocks held by running requests (the rest are free, cached or not);
    - ``kv_written_tokens``: tokens written into the cache since the server started, a counter: prompt tokens that
      missed the prefix cache plus generated tokens (``vllm:prompt_tokens``, ``vllm:prefix_cache_hits``,
      ``vllm:generation_tokens``);
    - ``prefix_cache_hits``, ``prefix_cache_queries``: prompt tokens found in the cache, and looked up (counters);
    - ``started_at``: when the server process started, so a restart (an empty cache) is noticed."""
    values: dict[str, float] = {}
    for name, value in _METRIC.findall(text):
        try:
            values[name] = values.get(name, 0.0) + float(value)
        except ValueError:
            continue
    kv = values.get("vllm:kv_cache_usage_perc", values.get("vllm:gpu_cache_usage_perc", 0.0))
    waiting = values.get("vllm:num_requests_waiting", 0.0)
    util = min(kv, 1.0)
    if waiting > 0:
        util = max(util, min(1.0, 0.85 + 0.015 * waiting))
    out: dict = {"utilization": util, "waiting": int(waiting), "running": int(values.get("vllm:num_requests_running", 0.0))}
    capacity = 0
    for labels in _CACHE_CONFIG.findall(text):
        found = dict(_LABEL.findall(labels))
        try:
            capacity += int(found.get("num_gpu_blocks") or 0) * int(found.get("block_size") or 0)
        except ValueError:
            continue
    if capacity:
        out.update(kv_capacity_tokens=capacity, kv_usage=round(min(kv, 1.0), 4))

    def counter(*names: str) -> float | None:
        for n in names:  # Prometheus adds _total to counters; older vLLM named the prefix-cache ones gpu_*
            for v in (values.get(f"{n}_total"), values.get(n)):
                if v is not None:
                    return v
        return None

    prompt, generated = counter("vllm:prompt_tokens"), counter("vllm:generation_tokens")
    hits, queries = counter("vllm:prefix_cache_hits", "vllm:gpu_prefix_cache_hits"), counter("vllm:prefix_cache_queries", "vllm:gpu_prefix_cache_queries")
    if prompt is not None and generated is not None:
        out["kv_written_tokens"] = max(prompt - (hits or 0.0), 0.0) + generated
    if hits is not None and queries is not None:
        out.update(prefix_cache_hits=hits, prefix_cache_queries=queries)
    started = _STARTED.search(text)
    if started:
        out["started_at"] = float(started.group(1))
    return out


async def fetch_load(endpoint: str, headers: dict | None = None) -> dict | None:
    base = endpoint.rstrip("/")
    base = base[:-3] if base.endswith("/v1") else base
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.get(f"{base}/metrics", headers=headers)
        return vllm_load(r.text) if r.status_code == 200 else None
    except httpx.HTTPError:
        return None


async def fetch_utilization(endpoint: str, headers: dict | None = None) -> float | None:
    load = await fetch_load(endpoint, headers)
    return load["utilization"] if load else None


@dataclass
class Deployment:
    backend: BackendConfig
    model: ModelConfig
    provisioner: object
    record: dict = field(default_factory=dict)


# ------------------------------------------------------------------ what-if


def simulate(model: ModelConfig, cfg: Config, hourly_rows: list[dict], gpu_cost_per_hour: float,
             capacity_tps: float, serve: ServeConfig | None = None) -> dict:
    """Replay hourly demand: what would self-hosting this model have cost and saved?

    Hour by hour, the traffic the model could serve (its quality meets the floor) is
    priced at API rates; GPUs run in the hours where they pay for themselves. Ignores
    hysteresis and cold starts, so treat the result as an upper bound.
    """
    serve = serve or ServeConfig(model=model.name)
    by_hour: dict = {}
    for r in hourly_rows:
        by_hour.setdefault(r["hour"], []).append(r)
    hours = sorted(by_hour)
    api_cost = gpu_cost = saved = 0.0
    hours_on = peak = tokens_all = tokens_on = 0
    for h in hours:
        cost, tokens = addressable(model, by_hour[h], cfg)
        api_cost += cost
        tokens_all += tokens
        replicas = desired_replicas(serve, gpu_cost_per_hour, capacity_tps, Demand(cost, tokens / 3600))
        if replicas:
            hours_on += 1
            peak = max(peak, replicas)
            gpu_cost += replicas * gpu_cost_per_hour
            saved += cost - replicas * gpu_cost_per_hour
            tokens_on += tokens
    span_hours = (hours[-1] - hours[0]).total_seconds() / 3600 + 1 if hours else 1.0
    return {
        "model": model.name,
        "hours_with_traffic": len(hours),
        "hours_gpus_would_run": hours_on,
        "peak_replicas": peak,
        "addressable_api_cost": round(api_cost, 4),
        "gpu_cost": round(gpu_cost, 4),
        "net_savings": round(saved, 4),
        "projected_monthly_savings": round(saved / span_hours * 24 * 30, 2),
        "share_of_tokens_on_gpu": round(tokens_on / tokens_all, 3) if tokens_all else 0.0,
    }


# ------------------------------------------------------------------ break-even

# Shares of each kind of token, used until there's traffic to measure: agent work, mostly prompt read from the cache.
ASSUMED_MIX = {"input": 0.1, "cache_read": 0.8, "cache_write": 0.05, "output": 0.05}
MIX = tuple(ASSUMED_MIX)
MIX_MIN_REQUESTS = 20  # a level or model sets its own token mix from this many requests; fewer use everyone's


def token_mix(rows: list[dict], min_requests: int = 1) -> dict | None:
    """Shares of input, cache reads, cache writes and output in these rows, with tokens per request; None with no
    tokens or fewer than ``min_requests`` requests."""
    totals = {f: sum(r.get(f) or 0 for r in rows) for f in MIX}
    tokens = sum(totals.values())
    requests = sum(r.get("requests") or 0 for r in rows)
    if not tokens or requests < min_requests:
        return None
    return {**{f: v / tokens for f, v in totals.items()}, "tokens_per_request": tokens / requests if requests else None}


def blended(price, mix: dict) -> float:
    """Dollars per million tokens of this mix at this price (long-prompt tiers aside)."""
    return mix["input"] * price.input + mix["cache_read"] * price.read() + mix["cache_write"] * price.write() + mix["output"] * price.output


CLOSE_ENOUGH = 1.1  # when no GPU pays, a cheaper machine this close to the lowest cost per token shows instead


def need(gpu_usd_per_hour: float, capacity_tps: float, utilization: float, margin: float, usd_per_million: float | None,
         tokens_per_request: float | None) -> dict:
    """The traffic one replica needs to pay for itself the way the controller decides it: what that traffic costs at
    API prices must reach the replica's price plus the margin. ``pays`` is False when even a full replica (at its
    target utilization) can't get there: per token, the GPU costs more than the API."""
    full = capacity_tps * 3600 * utilization
    out = {"capacity_tokens_per_minute": round(full / 60), "gpu_usd_per_million": round(gpu_usd_per_hour / full * 1e6, 4) if full else None,
           "api_usd_per_million": round(usd_per_million, 4) if usd_per_million is not None else None}
    if not usd_per_million or usd_per_million <= 0:
        return {**out, "tokens_per_minute": None, "requests_per_hour": None, "pays": False}
    per_hour = gpu_usd_per_hour * (1 + margin) / (usd_per_million / 1e6)
    return {**out, "tokens_per_minute": round(per_hour / 60), "pays": per_hour <= full,
            "requests_per_hour": round(per_hour / tokens_per_request, 1) if tokens_per_request else None}


def _usage(rows: list[dict], hours: float) -> dict:
    """Average and busiest-hour traffic in tokens per minute, and requests per hour on average."""
    by_hour: dict = {}
    for r in rows:
        by_hour[r["hour"]] = by_hour.get(r["hour"], 0) + r["tokens"]
    tokens = sum(by_hour.values())
    return {"tokens_per_minute": round(tokens / (hours * 60), 1), "peak_tokens_per_minute": round(max(by_hour.values(), default=0) / 60, 1),
            "requests_per_hour": round(sum(r.get("requests") or 0 for r in rows) / hours, 2), "usd_per_hour": round(sum(r["cost"] for r in rows) / hours, 4)}


def _api_routes(cfg: Config, live: set[str]):
    """(model, route, backend) for every priced, live API route: what `auto` would otherwise pay. GPUs, mocks and
    free tiers (rate-limited: they can't take sustained traffic) are left out."""
    kinds = {b.name: b.kind for b in cfg.backends}
    for m in cfg.models:
        if m.kind != "chat":
            continue
        for r in m.routes:
            if r.backend in live and kinds.get(r.backend) not in ("self_hosted", "mock") and r.price and (r.price.input or r.price.output):
                yield m, r


def deployment_for(cfg: Config, model: ModelConfig) -> BackendConfig | None:
    """The self-hosted deployment that serves this model, if one is configured: enabled first, then ones the controller
    starts (SkyPilot, then Kubernetes), then servers you run."""
    deps = [b for b in cfg.backends if b.kind == "self_hosted" and (
        (b.serve and b.serve.model == model.name) or any(r.backend == b.name for r in model.routes))]
    order = {"skypilot": 0, "kubernetes": 1}
    deps.sort(key=lambda b: (not b.enabled, order.get(b.serve.provisioner if b.serve else "", 2)))
    return deps[0] if deps else None


def breakeven(cfg: Config, live: set[str], rows: list[dict], hours: float, plans: dict[str, dict], utilization: float = 0.7,
              margin: float = 0.2, efficiency: float = 0.1) -> dict:
    """How much traffic a GPU replica needs to pay for itself, by quality level and by open-weight model.

    A replica displaces what `auto` would otherwise pay for the traffic it can take: at each level, the cheapest
    API route good enough for it. So for a level, the price to beat is that route's, blended with the level's own
    mix of input, cached and output tokens, and the GPU is the open model (and machine) that qualifies and pays
    from the least traffic, or if none can, the one cheapest per token; for a model, it's what the traffic it could
    serve actually cost, on its own replica.
    ``rows`` are ``demand_by_hour`` rows over ``hours``. ``plans`` are each open model's replica
    (``gpu_sizing.replica_plans``, with the machines it could have); one without a measured capacity gets an
    estimate (``efficiency`` of peak).
    ``hostable`` is, per task type, the best open model that has a replica: the highest level you could serve on GPUs."""
    from .gpu_sizing import BITS_NAME, capacity_tps

    def weights(size: dict | None) -> dict:
        """The precision a replica serves the weights in, and a warning when it's below the published one: quality
        scores are for the weights as published."""
        if not size or "bits" not in size:
            return {"weights": None, "published_weights": None, "quality_note": None}
        now, was = size["bits"], size.get("published_bits", size["bits"])
        note = (f"Quality scores are for the {BITS_NAME[was]} weights as published; at {BITS_NAME[now]} it may score lower, "
                "so measure it with an eval before routing on it") if now < was else None
        return {"weights": BITS_NAME[now], "published_weights": BITS_NAME[was], "quality_note": note}

    overall = token_mix([r for r in rows if r.get("floor")]) or token_mix(rows)
    mix_source = "traffic" if overall else "assumed"
    overall = overall or {**ASSUMED_MIX, "tokens_per_request": None}
    routes = list(_api_routes(cfg, live))
    tasks = sorted({t for m in cfg.models if m.kind == "chat" for t, v in m.quality.items() if v > 0})

    def cheapest(task: str, level: int, mix: dict):
        best = None
        for m, r in routes:
            if m.q(task) >= level:
                price = blended(r.cost_price(), mix)
                if best is None or price < best[0]:
                    best = (price, m.name, r.backend)
        return best

    def gpu(name: str, mix: dict, price: float | None) -> dict | None:
        """A model's replica for this mix, against API work at ``price`` per million tokens: price per hour, tokens
        per second, and cost per million tokens when full. Of the machines it could have, the cheapest per hour that
        pays (the least traffic to pay for); if none can, the cheapest per token (the closest to paying)."""
        plan = plans.get(name) or {}
        if "replica" not in plan:
            return None

        def on(replica: dict) -> dict:
            estimated = capacity_tps(replica, plan["size"], efficiency, mix["cache_read"]) if plan.get("size") and replica.get("tflops") else None
            cap = plan.get("capacity_tps") or estimated
            usd = replica["usd_per_hour"]
            return {"model": name, "deployment": plan.get("deployment"), "usd_per_hour": usd, "capacity_tps": cap,
                    "estimated_capacity_tps": round(estimated) if estimated else None,
                    "capacity_basis": "configured" if plan.get("capacity_tps") else "estimated",
                    "usd_per_million": usd / (cap * 3600 * utilization) * 1e6, "replica": replica}

        choices = [on(r) for r in plan.get("options") or [plan["replica"]]]
        return first(choices, price) if price else choices[0]

    def first(options: list[dict], price: float | None) -> dict | None:
        """The one that pays from the least traffic (the cheapest per hour that pays); if none can, the cheapest per
        token, or a cheaper machine within 10% of it (the same verdict, without eight GPUs for a model that fits on one)."""
        paying = [g for g in options if price and need(g["usd_per_hour"], g["capacity_tps"], utilization, margin, price, None)["pays"]]
        if paying:
            return min(paying, key=lambda g: g["usd_per_hour"])
        best = min((g["usd_per_million"] for g in options), default=None)
        return min((g for g in options if g["usd_per_million"] <= best * CLOSE_ENOUGH), key=lambda g: g["usd_per_hour"]) if options else None

    def summary(g: dict) -> dict:
        r = g["replica"]
        return {"model": g["model"], "deployment": g["deployment"], "usd_per_hour": g["usd_per_hour"],
                "capacity_tokens_per_second": round(g["capacity_tps"]), "estimated_capacity_tokens_per_second": g["estimated_capacity_tps"],
                "capacity_basis": g["capacity_basis"],
                **{k: r.get(k) for k in ("accelerator", "count", "memory_gb", "cloud", "region", "instance_type", "spot", "source",
                                         "on_demand_usd_per_hour", "spot_usd_per_hour", "computes_in")},
                **weights((plans.get(g["model"]) or {}).get("size"))}

    groups = []
    for task in tasks:
        for level in range(1, 6):
            floor = f"{task}:{level}"
            mine = [r for r in rows if r.get("floor") == floor]
            mix = token_mix(mine, MIX_MIN_REQUESTS) or overall
            best = cheapest(task, level, mix)
            qualify = sorted(m.name for m in cfg.models if m.kind == "chat" and m.hf_id and m.q(task) >= level)
            options = [g for g in (gpu(n, mix, best[0] if best else None) for n in qualify) if g]
            pick = first(options, best[0] if best else None)
            why = None if pick else ("no open-weight model scores high enough" if not qualify else
                                     "none of the open models that qualify can be run: "
                                     + "; ".join(f"{n} {(plans.get(n) or {}).get('error', 'not sized')}" for n in qualify))
            row = {"task": task, "level": level, "floor": floor, "cheapest": {"model": best[1], "backend": best[2]} if best else None,
                   "open_models": qualify, "gpu": summary(pick) if pick else None, "why": why, "traffic": _usage(mine, hours)}
            if pick:
                row.update(need(pick["usd_per_hour"], pick["capacity_tps"], utilization, margin, best[0] if best else None,
                                mix.get("tokens_per_request") or overall.get("tokens_per_request")))
            else:
                row.update({"capacity_tokens_per_minute": None, "gpu_usd_per_million": None, "api_usd_per_million": round(best[0], 4) if best else None,
                            "tokens_per_minute": None, "requests_per_hour": None, "pays": False})
            groups.append(row)

    models = []
    for m in cfg.models:
        if m.kind != "chat" or not m.hf_id:
            continue
        plan = plans.get(m.name) or {"error": "not sized"}
        mine = [r for r in rows if could_serve(m, r, cfg)]
        tokens = sum(r["tokens"] for r in mine)
        mix = token_mix(mine, MIX_MIN_REQUESTS) or overall
        own = min((blended(r.cost_price(), mix) for mm, r in routes if mm.name == m.name), default=None)
        if tokens:
            value, basis = sum(r["cost"] for r in mine) / tokens * 1e6, "what the traffic it could serve cost"
        else:
            # No traffic yet: the hardest work it could take, at the cheapest API price for that level.
            tops = [(cheapest(t, int(m.q(t)), mix), t, int(m.q(t))) for t in tasks if m.q(t) >= 1]
            top = max((x for x in tops if x[0]), key=lambda x: x[0][0], default=None)
            value = top[0][0] if top else None
            basis = (f"no traffic yet: the cheapest API model at {top[1]}:{top[2]}" if top
                     else "auto doesn't pick it (no quality score), so there's no traffic for it to take" if not any(m.q(t) > 0 for t in tasks)
                     else "no API route good enough to compare with")
        g = gpu(m.name, mix, value)
        size = plan.get("size")
        row = {"model": m.name, "hf_id": m.hf_id, "quality": {t: m.q(t) for t in tasks}, "basis": basis,
               "size": {**{k: size[k] for k in ("weight_bytes", "total_params", "active_params", "basis")}, **weights(size)} if size else None,
               "gpu": summary(g) if g else None, "why": None if g else plan.get("error"),
               "own_api_usd_per_million": round(own, 4) if own is not None else None, "traffic": _usage(mine, hours)}
        if g:
            row.update(need(g["usd_per_hour"], g["capacity_tps"], utilization, margin, value,
                            mix.get("tokens_per_request") or overall.get("tokens_per_request")))
        else:
            row.update({"capacity_tokens_per_minute": None, "gpu_usd_per_million": None, "api_usd_per_million": round(value, 4) if value else None,
                        "tokens_per_minute": None, "requests_per_hour": None, "pays": False})
        models.append(row)
    models.sort(key=lambda x: (x["tokens_per_minute"] is None, not x["pays"], x["tokens_per_minute"] or 0))
    hostable = {}
    for task in tasks:
        runnable = [m for m in models if m["gpu"] and m["quality"].get(task, 0) >= 1]
        best = max(runnable, key=lambda m: (m["quality"][task], -(m["gpu_usd_per_million"] or 0)), default=None)
        unrunnable = max((m for m in models if not m["gpu"] and m["quality"].get(task, 0) > (best["quality"][task] if best else 0)),
                         key=lambda m: m["quality"][task], default=None)
        hostable[task] = {"level": int(best["quality"][task]) if best else None, "model": best["model"] if best else None,
                          "quality": best["quality"][task] if best else None, "gpu": best["gpu"] if best else None,
                          "better_unrunnable": {"model": unrunnable["model"], "quality": unrunnable["quality"][task], "why": unrunnable["why"]}
                          if unrunnable else None}
    return {"assumptions": {"utilization": utilization, "margin": margin, "efficiency": efficiency}, "hostable": hostable,
            "mix": {**{f: round(overall[f], 4) for f in MIX}, "tokens_per_request": round(overall["tokens_per_request"]) if overall.get("tokens_per_request") else None,
                    "source": mix_source},
            "hours": hours, "groups": groups, "models": models}
