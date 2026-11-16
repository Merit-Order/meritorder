"""Capacity controller (docs/05-gpu-autoscaling.md).

Every interval, for each self-hosted backend with a ``serve:`` section:

1. Measure addressable demand: what traffic this model could serve costs per hour
   at API prices, and its token throughput. Free upgrades are excluded, so idle
   capacity can't keep GPUs alive by attracting traffic.
2. Size replicas: enough for throughput, but each replica must pay for itself.
   Then apply admin controls (stop, pin, stop all) and safeguards (GPU budget, fleet caps).
3. Advance the state machine (gpu.decide) and act through the provisioner.
4. Publish state, endpoint, where replicas run, and vLLM utilization to Redis for the gateways: per replica when
   their addresses are known, with the KV-cache figures gateways estimate warm caches from (affinity.py), and
   follow the replicas' KV events where a deployment asks for it (kv_events.py).
5. Accrue GPU cost (replicas x hourly price) and log state changes, and what holds a start up, to the GPU activity log.
6. On SkyPilot, renew each wanted replica's lease (in the provisioner's status, read first thing in the cycle so an
   outage of Postgres can't cut it), and sweep: tear down every cluster of this install that no deployment wants
   (skypilot.py). On Kubernetes, delete this install's resources that no running deployment owns (kube.Sweeper).

A start or scale that fails is never forgotten: a failed scale keeps the replicas that ran and tries again, and a failed
start is torn down before the cooldown, or stays tracked as running, and stopped every cycle, while that fails.

Only one controller is active at a time (Postgres advisory lock, confirmed every cycle: keep_lock).
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import time
from datetime import UTC, datetime, timedelta
from urllib.parse import urlparse

from . import kv_events
from .alerts import notify
from .config import Config
from .configstore import ConfigFollower
from .gpu import (API_ONLY, DRAINING, PROVISIONING, VOLUMES_SETTING, WARM, Deployment, Demand, Plan, Status, active_control,
                  addressable, decide, desired_replicas, expected_cold_start, fetch_load, make_provisioner, plan_fleet,
                  record_cold_start, replica_url, start_factor)
from .state import make_state
from .store import make_store, month_start

log = logging.getLogger("meritorder.controller")
LOCK_ID = 0x4D45524954  # "MERIT"
PLACEMENT_TTL = 6 * 3600  # where a session sat on the GPUs is kept as long as the session itself
KV_FIELDS = ("kv_capacity_tokens", "kv_usage", "kv_written_tokens", "started_at")


VOLUMES_ACCRUED = "gpu_volumes:accrued_at"


async def write_events(store, dep: Deployment, rec: dict | None = None, now: float | None = None) -> None:
    """The provisioner's own GPU activity log entries since the last cycle (SkyPilot): a failed replica's last output
    and the driver and CUDA versions it ran on; a ready replica's start, phase by phase, which goes into the
    deployment's cold-start average (``rec``); and the weights cache volume, which is kept in the store so its monthly
    cost is counted (accrue_volumes) for as long as it exists."""
    drain = getattr(dep.provisioner, "drain_events", None)
    for kind, detail in drain() if drain is not None else []:
        if kind == "volume":
            volumes = dict(await store.get_setting(VOLUMES_SETTING) or {})
            known = volumes.get(detail["volume"]) or {}
            volumes[detail["volume"]] = {**known, "deployment": dep.backend.name, "infra": detail.get("infra"), "size_gb": detail.get("size_gb"),
                                         "usd_per_month": detail.get("usd_per_month"), "since": known.get("since") or now or time.time()}
            await store.set_setting(VOLUMES_SETTING, volumes)
            if not detail.get("created"):
                continue  # already there: nothing to log
        if kind == "replica_ready" and rec is not None and detail.get("total_seconds") is not None:
            record_cold_start(rec, detail["total_seconds"], {"cluster": detail.get("cluster"), "phases": detail.get("phases")}, now)
        await store.gpu_event(kind, dep.backend.name, "controller", detail)


async def accrue_volumes(store, state, now: float) -> float:
    """Count what the weights cache volumes cost since the last cycle in GPU spend (so the GPU budget sees it): they
    bill every hour they exist, running or not. A volume without a known price (a Kubernetes PVC with no
    weights_cache_usd_per_gb_month) isn't counted. Returns what all of them cost a month."""
    volumes = await store.get_setting(VOLUMES_SETTING) or {}
    last = await state.get_json(VOLUMES_ACCRUED)
    await state.set_json(VOLUMES_ACCRUED, now)
    elapsed = max(now - float(last), 0) if last else 0  # all of it, however long the controller was away: volumes bill on
    monthly = 0.0
    for v in volumes.values():
        usd = v.get("usd_per_month")
        if usd is None:
            continue
        monthly += usd
        if elapsed:
            await store.add_gpu_usage(v.get("deployment") or "-", datetime.fromtimestamp(now, UTC), 0, usd * elapsed / (730 * 3600))
    return round(monthly, 2)


async def end_pin(store, state, controls: dict, backend: str, why: str) -> None:
    """A pinned start failed: the pin ends, and the deployment follows demand again after the 30-minute cooldown. A pin
    that outlived its failure would launch the same broken deployment (an image the driver can't run, a missing
    secret) again after every cooldown, for as long as nobody looks. Pinning it again retries at once.

    Only this pin is removed, from the controls as they are now: a control saved while the cycle ran (Stop all GPUs, a
    pin on another deployment, this one pinned again) stays. ``controls`` is what the cycle started with."""
    pin = (controls.get("deployments") or {}).get(backend)
    await store.drop_setting_entry("gpu_controls", "deployments", backend, expect=pin)
    await state.set_json("gpu:controls", await store.get_setting("gpu_controls") or {})
    log.warning("%s: the pin (%s replica(s), by %s) ends: its start failed: %s", backend, (pin or {}).get("replicas"),
                (pin or {}).get("by") or "an admin", why)
    await store.gpu_event("control", backend, "controller", {"mode": "auto", "reason": f"pin ended: the start failed ({why})"[:500],
                                                             "pinned_by": (pin or {}).get("by")})


async def sample_replicas(reps: list[dict], before: list[dict], now: float, headers: dict | None = None) -> None:
    """Read each ready replica's vLLM metrics into its entry: load, and the KV-cache figures (capacity, use, tokens
    written, and how fast, from the previous sample) that gateways estimate warm caches from. ``headers``: the
    deployment's key, where its replicas need one (SkyPilot)."""
    prev = {str(r.get("id")): dict(r) for r in before or []}  # copies: the entries may be the ones updated below

    async def one(r: dict) -> None:
        r["url"] = replica_url(r.get("endpoint"))
        if not r["url"] or r.get("status") != "READY":
            return
        load = await fetch_load(r["url"], headers=headers)
        if load is None:
            return
        r.update(utilization=round(load["utilization"], 3), waiting=load["waiting"], running=load["running"], sampled_at=now,
                 **{k: load[k] for k in KV_FIELDS if k in load})
        old = prev.get(str(r.get("id"))) or {}
        same = old.get("started_at") == r.get("started_at") and old.get("sampled_at")
        if same and r.get("kv_written_tokens") is not None and old.get("kv_written_tokens") is not None and now > old["sampled_at"]:
            r["kv_written_per_second"] = round(max(r["kv_written_tokens"] - old["kv_written_tokens"], 0) / (now - old["sampled_at"]), 1)
        if same and "prefix_cache_queries" in load and old.get("prefix_cache_queries") is not None:
            asked = load["prefix_cache_queries"] - old["prefix_cache_queries"]
            if asked > 0:
                r["prefix_hit_rate"] = round(max(load["prefix_cache_hits"] - old["prefix_cache_hits"], 0) / asked, 3)
        if "prefix_cache_queries" in load:
            r.update(prefix_cache_queries=load["prefix_cache_queries"], prefix_cache_hits=load["prefix_cache_hits"])

    await asyncio.gather(*(one(r) for r in reps))


def kv_endpoints(serve, reps: list[dict]) -> dict[str, str]:
    """Where each ready replica publishes its KV events: its host, on the configured port."""
    out = {}
    for r in reps:
        host = urlparse(r.get("url") or "").hostname
        if host and r.get("status") == "READY" and not r.get("tunnel"):  # a tunnel carries the replica's gate alone
            out[str(r["id"])] = f"tcp://{host}:{serve.kv_events_port}"
    return out


def cache_signal(serve, rec: dict, following: set[str]) -> dict:
    """What the gateways know a session's warm cache from on this deployment, for the console."""
    reps = [r for r in rec.get("replicas_info") or [] if r.get("status") == "READY"]
    affinity = serve.replica_affinity and any(r.get("url") for r in reps)
    if following:
        source = "events"
    elif any(r.get("kv_capacity_tokens") and r.get("kv_written_tokens") is not None for r in reps):
        source = "estimate"
    else:
        source = "ttl"
    why = None
    if serve.kv_events and not following:
        why = kv_events.available() or "no replica is publishing KV events yet"
    return {"affinity": bool(affinity), "source": source, "kv_events": serve.kv_events, "kv_events_problem": why}


def deployments(cfg: Config, fleet=None) -> list[Deployment]:
    """The deployments to control. ``fleet`` (skypilot.Fleet): this install's SkyPilot clusters, shared by them all."""
    out = []
    for b in cfg.backends:
        if b.kind != "self_hosted" or b.serve is None or not b.enabled:
            continue
        model = cfg.model(b.serve.model)
        if model is None:
            log.warning("backend %s serves unknown model %r; skipping", b.name, b.serve.model)
            continue
        out.append(Deployment(backend=b, model=model, provisioner=make_provisioner(b, cfg, fleet)))
    return out


async def sweep(cfg: Config, store, deps: list[Deployment], fleet) -> list[str]:
    """Tear down every SkyPilot cluster of this install that no deployment wants now: one of a backend that was renamed,
    removed or switched off, a teardown that failed, or a scale cut short by a crash. Each is logged in the GPU activity
    log. Runs on the first cycle (whatever the config says), then while a SkyPilot deployment is configured or the last
    sweep found leftovers."""
    from .gpu import sky_available
    from .skypilot import SkyPilotProvisioner

    mine = [d.provisioner for d in deps if isinstance(d.provisioner, SkyPilotProvisioner)]
    configured = any(b.serve and b.serve.provisioner == "skypilot" for b in cfg.backends)
    if fleet is None or not (mine or configured or not fleet.swept_once or fleet.leftovers):
        return []
    if not sky_available():
        if configured and not fleet.swept_once:
            log.warning("SkyPilot deployments are configured, but SkyPilot isn't installed here: their clusters can't be checked")
        fleet.swept_once = True
        return []
    wanted = set().union(*(p.wanted() for p in mine)) if mine else set()
    try:
        return await fleet.sweep(wanted, store)
    except Exception as e:  # noqa: BLE001 (tried again next cycle; each replica stops by itself when its lease runs out)
        log.warning("couldn't list SkyPilot clusters for the sweep: %s", e)
        return []


def _same_place(old, new) -> bool:
    """Whether a deployment's new provisioner runs what its old one ran: the same kind, and on Kubernetes the same resource."""
    if not isinstance(old, type(new)):
        return False
    where = getattr(old, "resource_path", None)
    return where is None or where() == new.resource_path()


async def carry_over(old: list[Deployment], new: list[Deployment], store=None, state=None, now: float | None = None) -> list[Deployment]:
    """After a config change, a deployment's new provisioner takes over what its old one was doing (launches under way,
    failures it's waiting out), so a launch isn't started twice. A deployment that's gone from the new config (switched
    off, removed, its model renamed), or moved (another provisioner, namespace or KServe/llm-d), is stopped through its
    old provisioner, now: nothing else would stop a Kubernetes one. One that can't be stopped is logged; the Kubernetes
    and SkyPilot sweeps try again every cycle."""
    before = {d.backend.name: d.provisioner for d in old}
    kept = set()
    for d in new:
        prev = before.get(d.backend.name)
        adopt = getattr(d.provisioner, "adopt", None)
        if prev is not None and _same_place(prev, d.provisioner):
            kept.add(d.backend.name)
            if adopt is not None:
                adopt(prev)
    for d in old:
        if d.backend.name in kept or d.backend.serve.provisioner == "static":
            continue
        rec = (await state.get_json(f"deploy:{d.backend.name}") or {}) if state is not None else {}
        if rec.get("state", API_ONLY) == API_ONLY and d.backend.serve.provisioner != "kubernetes" and not getattr(d.provisioner, "want", 0):
            continue  # nothing running (a SkyPilot one the sweep would catch anyway)
        log.warning("%s: no longer in the config as it ran (switched off, removed, renamed or moved): stopping it", d.backend.name)
        try:
            await d.provisioner.down()
        except Exception as e:  # noqa: BLE001 (the sweeps try again every cycle)
            log.error("%s: stopping it after the config change failed: %s", d.backend.name, e)
            if store is not None:
                await store.gpu_event("error", d.backend.name, "controller", {"action": "down", "error": str(e)[:500]})
            continue
        if state is not None and rec.get("state", API_ONLY) != API_ONLY:
            rec, _ = decide(rec, d.backend.serve, 0, Status(), now or time.time(), force="off", drain=False,
                            why="stopped: no longer in the config")
            await state.set_json(f"deploy:{d.backend.name}", rec)
            if store is not None:
                await store.gpu_event("state", d.backend.name, "controller", {"from": "running", "to": API_ONLY, "reason": rec["reason"]})
    return new


async def renew_first(state, deps: list[Deployment]) -> dict[str, Status]:
    """Each running deployment's status, read first thing in the cycle: on SkyPilot that's where the leases are renewed,
    so a Postgres outage (the ledger, the controls) never lets them run out while the controller is up. Each deployment
    on its own: one that fails is read again later in the cycle. When Redis can't say what runs either, only what this
    process already asked for is renewed (after a restart nothing is known: renewing nothing stops nothing that's wanted).
    A draining SkyPilot deployment doesn't launch again a replica that went away."""
    out: dict[str, Status] = {}
    for dep in deps:
        if dep.backend.serve.provisioner == "static":
            continue  # nothing to renew: it's probed later in the cycle
        try:
            try:
                rec = await state.get_json(f"deploy:{dep.backend.name}") or {}
            except Exception as e:  # noqa: BLE001 (Redis is down: what this process asked for stands in)
                log.warning("%s: can't read its record (%s): renewing what this process last asked for", dep.backend.name, e)
                if not getattr(dep.provisioner, "want", None):
                    continue
                rec = None
            if rec is not None:
                if rec.get("state", API_ONLY) == API_ONLY:
                    continue
                restore = getattr(dep.provisioner, "restore", None)
                if restore is not None:
                    restore(rec)
                if hasattr(dep.provisioner, "relaunch"):
                    dep.provisioner.relaunch = rec.get("state") != DRAINING
            out[dep.backend.name] = await dep.provisioner.status()
        except Exception as e:  # noqa: BLE001 (read again later in the cycle)
            log.warning("%s: status at the start of the cycle failed: %s", dep.backend.name, e)
    return out


def cap_price(rec: dict, expected: float) -> float:
    """What one replica counts against the fleet's burn-rate cap: the expected (cheapest) price, or, while it runs, what
    each replica really burns when that's more (SkyPilot placed it in a pricier region, a spot price moved)."""
    running = rec.get("replicas") or 0 if rec.get("state", API_ONLY) != API_ONLY else 0
    burn = rec.get("burn_usd_per_hour")
    return max(expected, burn / running) if running and burn else expected


def billed_seconds(dep: Deployment, rec: dict, last: float, now: float) -> float:
    """Seconds a running deployment billed since its last accrual. All of them, however long the gap (a controller that
    was down for a day): a Kubernetes deployment, or a server you run, runs on without the controller. A SkyPilot replica
    doesn't: it stops itself ``lease_minutes`` after its last lease renewal, and its machine is deleted
    ``autostop_minutes`` later, so a gap is counted up to then (when the last renewal is known)."""
    elapsed = max(now - last, 0)
    s = dep.backend.serve
    renewed = max(float(rec.get("renewed_at") or 0), float(getattr(dep.provisioner, "renewed_before", None) or 0))
    if s.provisioner == "skypilot" and renewed:
        elapsed = min(elapsed, max(renewed + (s.lease_minutes + s.autostop_minutes) * 60 - last, 0))
    return elapsed


async def tick(cfg: Config, store, state, deps: list[Deployment], now: float | None = None,
               kv: kv_events.Subscriber | None = None, fleet=None, ksweep=None) -> list[dict]:
    """One cycle for every deployment. ``ksweep`` (kube.Sweeper): after it, delete this install's Kubernetes resources
    that no running deployment owns."""
    now = now or time.time()
    wall = datetime.fromtimestamp(now, UTC)
    early = await renew_first(state, deps)  # before anything that needs Postgres: an outage there mustn't cut the leases
    rows_1h = await store.demand_rows(wall - timedelta(hours=1))
    rows_15m = await store.demand_rows(wall - timedelta(minutes=15))
    controls = await store.get_setting("gpu_controls") or {}
    await state.set_json("gpu:controls", controls)  # gateways read it to stop routing to stopped GPUs right away
    guards = cfg.safeguards

    # Accrue what running replicas cost since the last cycle (you pay while provisioning and draining too).
    recs: dict[str, dict] = {}
    for dep in deps:
        b = dep.backend
        rec = await state.get_json(f"deploy:{b.name}") or {}
        last = rec.get("accrued_at") or now
        elapsed = billed_seconds(dep, rec, last, now)
        if rec.get("state", API_ONLY) != API_ONLY and rec.get("replicas") and elapsed:
            burn = rec.get("burn_usd_per_hour")
            burn = burn if burn is not None else rec["replicas"] * b.gpu_cost_per_hour
            cost = burn * elapsed / 3600
            await store.add_gpu_usage(b.name, wall, rec["replicas"] * elapsed, cost)
            rec["run_cost"] = (rec.get("run_cost") or 0) + cost
        rec["accrued_at"] = now
        recs[b.name] = rec
    await accrue_volumes(store, state, now)
    gpu_spent = sum((await store.gpu_spend_since(month_start(wall))).values())

    sized = []
    for dep in deps:
        b, serve = dep.backend, dep.backend.serve
        cost_1h, tok_1h = addressable(dep.model, rows_1h, cfg)
        cost_15, tok_15 = addressable(dep.model, rows_15m, cfg)
        demand = Demand(usd_per_hour=max(cost_1h, cost_15 * 4), tokens_per_second=max(tok_1h / 3600, tok_15 / 900))
        price = await replica_price(dep)
        rec = recs[b.name]
        running = (rec.get("replicas") or 0) if rec.get("state", API_ONLY) != API_ONLY else 0
        cold, basis = expected_cold_start(serve, rec)
        desired = desired_replicas(serve, price["usd_per_hour"], b.capacity_tokens_per_second, demand, running=running, cold_start_minutes=cold)
        # What one more replica needs to start: its price and the margin, with its cold start paid back over hold_minutes.
        rec["start_needs"] = {"usd_per_hour": round(price["usd_per_hour"] * (1 + serve.margin) * start_factor(serve, cold), 4),
                              "cold_start_minutes": round(cold, 1), "basis": basis, "hold_minutes": serve.hold_minutes}
        sized.append((dep, demand, desired, price))
    plans = plan_fleet([{"name": d.backend.name, "desired": want, "price": cap_price(recs[d.backend.name], price["usd_per_hour"]),
                         "max_replicas": d.backend.serve.max_replicas, "value": dem.usd_per_hour,
                         "billed": d.backend.serve.provisioner != "static"} for d, dem, want, price in sized],
                       controls, guards, gpu_spent, now)

    published = []
    for dep, demand, desired, price in sized:
        b, serve = dep.backend, dep.backend.serve
        plan = plans[b.name]
        key = f"deploy:{b.name}"
        rec = recs[b.name]
        before = rec.get("state", API_ONLY)
        pin = active_control(controls, b.name, now)
        if (plan.force == "on" and before == API_ONLY and now < rec.get("cooldown_until", 0)
                and (pin or {}).get("at", 0) < rec.get("since", 0)):
            # Pinned before the start that failed (a missing Secret, an image that isn't there): starting again every other
            # cycle would only fail and alert again. The cooldown applies; pinning it again retries at once. (A failed start
            # ends its pin, end_pin below; this covers a pin written back with its old time by a control saved meanwhile.)
            plan = Plan(0, None, plan.drain, f"{plan.note}; cooling down after a failed start (pin it again to retry now)")
        restore = getattr(dep.provisioner, "restore", None)
        if restore is not None:
            restore(rec)  # after a restart: want what the last controller asked for, so nothing is swept that should run
        if rec.get("stop_pending"):
            # A start that failed, whose teardown failed too: stop it before anything else, every cycle until it's down.
            plan = Plan(0, "off", False, rec.get("reason") or "stopping after a failed start")
        if b.name in early:
            status = early[b.name]  # read (and leases renewed) at the start of the cycle
        elif before == API_ONLY and serve.provisioner != "static":
            status = Status()  # don't poll the cloud for something that isn't running
        else:
            status = await dep.provisioner.status()
        new, action = decide(rec, serve, plan.desired, status, now, force=plan.force, drain=plan.drain, why=plan.note)
        if action is not None:
            began = time.monotonic()
            try:
                if action.kind == "up":
                    await dep.provisioner.up(action.replicas)
                elif action.kind == "scale":
                    await dep.provisioner.scale(action.replicas)
                elif action.kind == "down":
                    await dep.provisioner.down()
            except Exception as e:
                log.error("%s: %s failed: %s", b.name, action.kind, e)
                await store.gpu_event("error", b.name, "controller", {"action": action.kind, "error": str(e)[:500]})
                if action.kind == "down":
                    # Keep trying to stop it: never report a GPU as off when it may still be billing.
                    new = {**rec, "reason": f"stop failed, will retry: {e}"[:300]}
                elif action.kind == "scale":
                    # What ran before still runs and is still tracked (and billed); the scale is tried again next cycle.
                    new.update(replicas=rec.get("replicas"), reason=f"scale failed, will retry: {e}"[:300])
                else:
                    # A start can fail halfway (some machines launched, a resource half made): take down whatever is there
                    # before calling it off, and if that fails too, keep it tracked as running and stop it next cycle.
                    try:
                        await dep.provisioner.down()
                    except Exception as e2:
                        log.error("%s: stop after the failed start failed too: %s", b.name, e2)
                        await store.gpu_event("error", b.name, "controller", {"action": "down", "error": str(e2)[:500]})
                        new.update(stop_pending=True, cooldown_until=now + 1800,
                                   reason=f"up failed: {e}; stop failed, will retry: {e2}"[:300])
                    else:
                        new.update(state=API_ONLY, since=now, endpoint=None, replicas=0, cooldown_until=now + 1800,
                                   reason=f"{action.kind} failed: {e}"[:300])
            else:
                if action.kind == "down":
                    new.pop("stop_pending", None)
                    # A stop returns once the machines are gone (on Kubernetes its pods, after their grace period): that
                    # time was billed too.
                    took, burn = time.monotonic() - began, rec.get("burn_usd_per_hour") or 0
                    if took >= 1 and burn:
                        await store.add_gpu_usage(b.name, wall, (rec.get("replicas") or 0) * took, burn * took / 3600)
                if action.kind == "scale":
                    await store.gpu_event("scale", b.name, "controller", {"replicas": action.replicas, "reason": plan.note or "demand"})
                if action.kind == "up":
                    # A server that is already running (static) or comes up fast goes warm this cycle.
                    status = await dep.provisioner.status()
                    if status.ready:
                        new, _ = decide(new, serve, plan.desired, status, now, force=plan.force, drain=plan.drain, why=plan.note)
        await write_events(store, dep, new, now)
        if (before == PROVISIONING and new.get("state") == WARM and not rec.get("warm_at") and new.get("started_at")
                and serve.provisioner == "kubernetes"):
            # On a cluster the start is timed as a whole, from the controller's start to the first ready replica.
            took = now - float(new["started_at"])
            record_cold_start(new, took, None, now)
            await store.gpu_event("replica_ready", b.name, "controller", {"total_seconds": round(took, 1), "phases": []})
        # A start that failed this cycle (whether its teardown worked or not) sets a new cooldown.
        if pin and pin.get("mode") == "on" and new.get("cooldown_until", 0) > rec.get("cooldown_until", 0):
            await end_pin(store, state, controls, b.name, new.get("reason") or "the start failed")
        if new.get("state") != API_ONLY and status.replicas_info:
            new["replicas_info"] = status.replicas_info
        # What the provisioner says is holding it up (no GPU node free, an image that won't pull): shown and logged once.
        message = status.message if new.get("state", API_ONLY) != API_ONLY else None
        if message and message != rec.get("status_message"):
            await store.gpu_event("status", b.name, "controller", {"message": message[:500]})
        new["status_message"] = message
        # What it's costing right now: each replica's own catalog price when known, else the expected price.
        running = (new.get("replicas") or 0) if new.get("state", API_ONLY) != API_ONLY else 0
        priced = [r.get("usd_per_hour") for r in new.get("replicas_info") or [] if r.get("usd_per_hour") is not None]
        if running and status.usd_per_hour is not None:  # the provisioner counted its machines itself (SkyPilot)
            new["burn_usd_per_hour"] = round(status.usd_per_hour, 4)
        elif running and len(priced) >= running:
            new["burn_usd_per_hour"] = round(sum(sorted(priced)[:running]), 4)
        else:
            new["burn_usd_per_hour"] = round(running * price["usd_per_hour"], 4)
        new["price"] = price
        renewed = getattr(dep.provisioner, "renewed_at", None)
        if renewed:
            new["renewed_at"] = renewed  # a SkyPilot lease's last renewal: how long its replicas outlive this controller
        live = new.get("state") in (WARM, DRAINING)
        headers = dep.provisioner.headers() if hasattr(dep.provisioner, "headers") else None
        if live and new.get("replicas_info"):
            await sample_replicas(new["replicas_info"], rec.get("replicas_info") or [], now, headers)
        sampled = [r for r in new.get("replicas_info") or [] if live and r.get("sampled_at") == now]
        load = None
        if sampled:  # the replicas' own figures: the endpoint in front of them would answer for one of them only
            load = {"utilization": sum(r["utilization"] for r in sampled) / len(sampled), "waiting": sum(r["waiting"] for r in sampled)}
        elif new.get("endpoint"):
            load = await fetch_load(new["endpoint"], headers=headers)
        if load is not None:
            await state.set_utilization(b.name, load["utilization"])
            new["utilization"] = round(load["utilization"], 3)
            new["waiting"] = load["waiting"]  # gateways bound the last-resort queue with this
            new["sampled_at"] = now
        following: set[str] = set()
        if kv is not None and serve.kv_events:
            kv.sync(b.name, kv_endpoints(serve, new.get("replicas_info") or []) if live else {})
            following = kv.following(b.name)
            for r in new.get("replicas_info") or []:
                r["kv_events"] = str(r.get("id")) in following
        new["cache"] = cache_signal(serve, new if live else {}, following)
        await prune_placements(state, b.name, now)
        new.update(
            backend=b.name,
            model=dep.model.name,
            desired=desired,
            planned=plan.desired,
            note=plan.note,
            demand_usd_per_hour=round(demand.usd_per_hour, 4),
            demand_tokens_per_second=round(demand.tokens_per_second, 1),
            gpu_usd_per_hour=price["usd_per_hour"],
            updated=wall.isoformat(timespec="seconds"),
        )
        if new.get("state") != before:
            await store.gpu_event("state", b.name, "controller", {"from": before, "to": new["state"], "reason": new.get("reason", "")})
            await notify(f"GPU tier: {b.name} ({dep.model.name}) {before} -> {new['state']}: {new.get('reason', '')}. "
                         f"Demand ${demand.usd_per_hour:,.2f}/h at API prices vs ${price['usd_per_hour']:,.2f}/h per replica.",
                         state=state, title=f"GPU {b.name}: {new['state'].replace('_', ' ')}", level="info", page="gpus")
        await state.set_json(key, new)
        published.append(new)
        log.info("%s (%s): %s, demand $%.2f/h at API prices, %.0f tok/s, desired %d replica(s) at $%.2f/h each%s",
                 b.name, dep.model.name, new["state"], demand.usd_per_hour, demand.tokens_per_second, desired, price["usd_per_hour"],
                 f" ({plan.note})" if plan.note else "")
    await sweep(cfg, store, deps, fleet)
    if ksweep is not None:
        await ksweep.sweep(cfg, store, deps, {r["backend"]: r.get("state", API_ONLY) for r in published})
    return published


async def prune_placements(state, backend: str, now: float) -> None:
    """Forget where sessions sat on this deployment once the sessions themselves have expired."""
    placed = await state.hgetall_json(f"placements:{backend}")
    old = [sid for sid, p in placed.items() if (p or {}).get("at", 0) < now - PLACEMENT_TTL]
    if old:
        await state.hdel(f"placements:{backend}", old)


async def replica_price(dep: Deployment) -> dict:
    """Price of one replica for sizing decisions: what the provisioner expects (SkyPilot's catalog, or the configured
    price of a Kubernetes node type, which says so in ``source``), else the deployment's gpu_cost_per_hour."""
    b = dep.backend
    offer = None
    getter = getattr(dep.provisioner, "expected_price", None)
    if getter is not None:
        try:
            offer = await getter()
        except Exception as e:
            log.debug("%s: price lookup failed: %s", b.name, e)
    if offer:
        return {**offer, "source": offer.get("source") or "skypilot_catalog", "configured_usd_per_hour": b.gpu_cost_per_hour}
    return {"usd_per_hour": b.gpu_cost_per_hour, "source": "configured"}


PRICES_REQUESTED = "gpu_prices:requested"  # set by the console's "Get latest prices" when the gateway has no SkyPilot
PRICES_RETRY_SECONDS = 300  # after prices couldn't be fetched, wait this long before trying again (unless asked)
_prices_failed: dict = {"at": None, "error": None}  # when the last fetch failed, and why (each reason is warned about once)


async def publish_gpu_prices(cfg: Config, store, state) -> None:
    """Keep SkyPilot's GPU prices in the store for the break-even report, whose gateway may not have SkyPilot: when
    they're older than gpu_sizing.CATALOG_MAX_AGE_HOURS, and right away (catalogs downloaded again) when asked."""
    from . import gpu_sizing

    asked = await state.get_json(PRICES_REQUESTED)
    catalog = await store.get_setting("gpu_catalog") or {}
    if not asked and gpu_sizing._age_hours(catalog.get("fetched_at")) <= gpu_sizing.CATALOG_MAX_AGE_HOURS:
        return
    failed = _prices_failed
    if not asked and failed["at"] is not None and time.monotonic() - failed["at"] < PRICES_RETRY_SECONDS:
        return  # a controller without SkyPilot (GPUs on Kubernetes alone) would otherwise try, and warn, every cycle
    try:
        out = await gpu_sizing.refresh(cfg, store, sizes=False, download=bool(asked))
        log.info("GPU prices published: %s offers", out.get("offers"))
        failed.update(at=None, error=None)
    except Exception as e:  # noqa: BLE001 (prices are a report's; the controller's own work goes on)
        (log.debug if str(e) == failed["error"] else log.warning)("GPU prices couldn't be fetched: %s", e)
        failed.update(at=time.monotonic(), error=str(e))
    if asked:
        await state.set_json(PRICES_REQUESTED, None)


async def wait_or_kick(state, interval: int, seen: str | None) -> str | None:
    """Sleep until the next cycle, waking early when an admin changes a GPU control. Redis being down only means no early
    wake-up: the cycle still comes."""
    deadline = time.monotonic() + interval
    while time.monotonic() < deadline:
        await asyncio.sleep(min(2, max(0.0, deadline - time.monotonic())))
        try:
            kick = await state.get_json("controller:kick")
        except Exception as e:  # noqa: BLE001 (tried again in 2 s)
            log.debug("can't read the controller's kick: %s", e)
            continue
        if kick and kick != seen:
            return kick
    return seen


LOCK_HELD = ("SELECT EXISTS (SELECT 1 FROM pg_locks WHERE locktype = 'advisory' AND pid = pg_backend_pid() AND granted "
             "AND objsubid = 1 AND ((classid::bigint << 32) | objid::bigint) = $1)")


async def keep_lock(store, conn):
    """Check, on its own connection, that this controller still holds the advisory lock. A dropped connection releases it,
    and a standby may then have taken it: two active controllers would launch and stop the same GPUs. When it can't be
    confirmed, the connection is let go, and the lock taken again on a new one if it's free. Returns (the connection,
    or None when no new one could be had; whether the lock is held)."""
    try:
        if conn is not None and await conn.fetchval(LOCK_HELD, LOCK_ID):
            return conn, True
    except Exception as e:  # noqa: BLE001 (the connection is gone: so is the lock)
        log.warning("can't confirm the controller lock: %s", e)
    if conn is not None:
        try:
            await store.pool.release(conn)
        except Exception as e:  # noqa: BLE001
            log.debug("releasing the lock's connection failed: %s", e)
    try:
        conn = await store.pool.acquire()
        return conn, bool(await conn.fetchval("SELECT pg_try_advisory_lock($1)", LOCK_ID))
    except Exception as e:  # noqa: BLE001 (Postgres is down: tried again next cycle)
        log.warning("can't take the controller lock again: %s", e)
        return None, False


async def run_controller(cfg: Config | None = None, once: bool = False, interval: int = 60, path: str | None = None) -> None:
    store = make_store(os.environ.get("MERITORDER_DATABASE_URL"))
    state = make_state(os.environ.get("MERITORDER_REDIS_URL"))
    if store.kind != "postgres":
        raise SystemExit("the controller needs Postgres: set MERITORDER_DATABASE_URL")
    await store.init()
    follower = ConfigFollower(store, path=path, cfg=cfg)
    cfg = await follower.current()
    from .kube import Sweeper
    from .skypilot import Fleet, load_identity

    # This install's id (in its SkyPilot cluster names) and the secret its replica keys come from, the same the gateways use.
    fleet = Fleet(await load_identity(store))
    deps = deployments(cfg, fleet)
    leases = [b.serve.lease_minutes for b in cfg.backends if b.serve and b.serve.provisioner == "skypilot" and b.enabled]
    if leases and interval * 3 > min(leases) * 60:
        log.warning("the controller renews SkyPilot leases every %ss, but a lease of %s min runs out after fewer than 3 missed "
                    "cycles: lower --interval or raise serve.lease_minutes", interval, min(leases))
    if not deps:
        log.warning("no self-hosted backends with a `serve:` section are enabled; nothing to control")
    lock_conn = await store.pool.acquire()
    kv: kv_events.Subscriber | None = None
    try:
        while not await lock_conn.fetchval("SELECT pg_try_advisory_lock($1)", LOCK_ID):
            log.info("another controller is active; standing by")
            if once:
                return
            await asyncio.sleep(interval)
        log.info("controller is active for: %s", ", ".join(d.backend.name for d in deps) or "(none)")
        try:
            kick = await state.get_json("controller:kick")
        except Exception as e:  # noqa: BLE001 (no early wake-ups until Redis is back)
            log.warning("can't read the controller's kick: %s", e)
            kick = None
        kv = kv_events.Subscriber(state)
        ksweep = Sweeper(fleet.identity.install_id)
        active = True
        while True:
            lock_conn, held = await keep_lock(store, lock_conn)
            if not held:
                if active:
                    log.error("this controller no longer holds the controller lock (its Postgres connection dropped, and "
                              "another controller may have taken it): it stops acting until it holds the lock again")
                active = False
                if once:
                    return
                await asyncio.sleep(interval)
                continue
            if not active:
                log.info("the controller lock is held again: acting")
                active = True
            try:
                latest = await follower.current()
                if latest is not cfg:
                    cfg, deps = latest, await carry_over(deps, deployments(latest, fleet), store, state)
                    log.info("configuration changed; controlling: %s", ", ".join(d.backend.name for d in deps) or "(none)")
                    for name in set(kv.deployments) - {d.backend.name for d in deps if d.backend.serve.kv_events}:
                        kv.sync(name, {})  # no longer asked for: stop following its replicas
                await tick(cfg, store, state, deps, kv=kv, fleet=fleet, ksweep=ksweep)
                await publish_gpu_prices(cfg, store, state)
            except Exception:
                log.exception("controller cycle failed")
            try:
                await state.set_json("controller:heartbeat", {"at": time.time(), "host": socket.gethostname(), "interval": interval,
                                                              "deployments": [d.backend.name for d in deps]})
            except Exception as e:  # noqa: BLE001 (Redis is down: the next cycle tries again)
                log.warning("couldn't write the controller's heartbeat: %s", e)
            if once:
                return
            kick = await wait_or_kick(state, interval, kick)
    finally:
        if kv is not None:
            await kv.close()
        if lock_conn is not None:
            await store.pool.release(lock_conn)
        await store.close()
        await state.close()
