"""Spend and GPU controls: what's running, what it costs, and the switches to stop it.

* GET  /v1/admin/controls                  budgets and safeguards with this month's spend against them (finance, admin)
* POST /v1/admin/gpu/{backend}/control     stop, pin, or return a GPU deployment to automatic (admin)
* POST /v1/admin/gpu/stop-all, /resume     emergency stop for the whole GPU fleet (admin)
* GET  /v1/admin/gpu/events                the GPU activity log: state changes, scaling, and who changed what

Controls live in Postgres (setting ``gpu_controls``) and are mirrored to Redis so gateways stop
routing to a stopped deployment within seconds; the capacity controller wakes up and acts on them.
Every change is written to the activity log and sent as an alert.
"""

from __future__ import annotations

import calendar
import time
import uuid
from datetime import UTC, datetime, timedelta
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from . import spend
from .alerts import notify
from .auth import AuthError, Principal
from .gpu import active_control
from .store import month_start


class GpuControl(BaseModel):
    mode: Literal["auto", "off", "on"]
    replicas: int | None = Field(default=None, ge=1, le=1000)  # mode=on: how many to keep running
    drain: bool = True  # mode=off: let warm sessions finish first (drain_minutes) instead of stopping now
    hours: float | None = Field(default=None, gt=0, le=24 * 90)  # return to automatic after this long
    reason: str | None = Field(default=None, max_length=500)


class StopAll(BaseModel):
    reason: str | None = Field(default=None, max_length=500)
    drain: bool = False


def month_fraction(now: datetime | None = None) -> float:
    now = now or datetime.now(UTC)
    days = calendar.monthrange(now.year, now.month)[1]
    return max((now - month_start(now)).total_seconds() / (days * 86400), 1e-6)


async def controller_status(state) -> dict:
    hb = await state.get_json("controller:heartbeat") or {}
    at = hb.get("at")
    interval = hb.get("interval") or 60
    alive = bool(at) and time.time() - at < max(3 * interval, 180)
    return {"alive": alive, "at": datetime.fromtimestamp(at, UTC).isoformat(timespec="seconds") if at else None,
            "host": hb.get("host"), "interval": interval}


async def fleet(gateway) -> dict:
    """Running replicas and burn rate across all self-hosted deployments, from the controller's published state."""
    replicas, burn, deployments = 0, 0.0, []
    for b in gateway.cfg.backends:
        if b.kind != "self_hosted" or b.serve is None:
            continue
        rec = await gateway.state.get_json(f"deploy:{b.name}") or {}
        if rec.get("state", "api_only") != "api_only":
            n = rec.get("replicas") or 0
            replicas += n
            burn += rec["burn_usd_per_hour"] if rec.get("burn_usd_per_hour") is not None else n * b.gpu_cost_per_hour
        deployments.append(b.name)
    return {"replicas": replicas, "usd_per_hour": round(burn, 2), "deployments": deployments}


def build_router(gateway) -> APIRouter:
    router = APIRouter(prefix="/v1/admin", tags=["controls"])

    async def principal(request: Request) -> Principal:
        try:
            return await gateway.authn.authenticate({k.lower(): v for k, v in request.headers.items()})
        except AuthError as e:
            raise HTTPException(e.status, e.message)

    async def admin(request: Request) -> Principal:
        p = await principal(request)
        if p.role != "admin":
            raise HTTPException(403, "only admins can change GPU controls")
        return p

    async def save(controls: dict, p: Principal, kind: str, backend: str | None, detail: dict, message: str) -> dict:
        await gateway.store.set_setting("gpu_controls", controls)
        await gateway.state.set_json("gpu:controls", controls)
        await gateway.state.set_json("controller:kick", uuid.uuid4().hex)
        await gateway.refresh_deployments()
        await gateway.store.gpu_event(kind, backend, p.user_id, detail)
        await notify(message, state=gateway.state, title=message.split(":")[0], level="warning", page="gpus")
        ctl = await controller_status(gateway.state)
        out = {"controls": controls, "controller": ctl}
        if not ctl["alive"]:
            out["warning"] = ("The capacity controller isn't running, so routing stopped but cloud GPUs were not changed. "
                              "Start it (`meritorder controller`). SkyPilot replicas stop by themselves once their lease runs out "
                              "(serve.lease_minutes, then serve.autostop_minutes); to stop them now, `sky down` their clusters "
                              "(`sky status` lists them, named mo-<install>-<deployment>-<n>).")
        return out

    @router.get("/controls")
    async def controls(request: Request):
        p = await principal(request)
        if p.role not in ("admin", "finance"):
            raise HTTPException(403, "finance and admins only")
        now = datetime.now(UTC)
        start = month_start(now)
        frac = month_fraction(now)
        g = gateway.cfg.safeguards
        # Every total here is spend.budget_spend's, the one admission counts: the ledger, what running batches reserved, and
        # for the organization GPU time no request paid for. Rounded to a millionth of a dollar, so sub-cent spend shows.
        org = await spend.budget_spend(gateway.store, gateway.cfg, "org", now=now)
        org_spent = org["spent"]

        def projected(s: dict) -> float:  # settled spend at this month's pace; reservations are already a whole amount
            return round((s["spent"] - s["reserved"]) / frac + s["reserved"], 2)

        gpu_by = await gateway.store.gpu_spend_since(start)
        gpu_spent = sum(gpu_by.values())
        by_team = {r["key"]: r for r in await gateway.store.summary(start, now + timedelta(days=1), "team")}
        teams = []
        for t in gateway.cfg.teams:
            ts = await spend.budget_spend(gateway.store, gateway.cfg, "team", t.name, now=now)
            teams.append({"name": t.name, "spent": round(ts["spent"], 6), "reserved": round(ts["reserved"], 6), "projected": projected(ts),
                          "budget_usd_month": t.budget_usd_month, "user_budget_usd_month": t.user_budget_usd_month,
                          "max_session_usd": t.max_session_usd, "rate_limits": t.rate_limits.model_dump(),
                          # What a failed attempt costs a person on the team, and whether routing counts it (expected_cost).
                          "human_cost_usd": t.human_cost_usd, "model_choice": gateway.router.model_choice(t)[0]})
        for name in by_team:  # spend by teams that aren't in the config (removed, or from old keys)
            if not any(t["name"] == name for t in teams):
                ts = await spend.budget_spend(gateway.store, gateway.cfg, "team", name, now=now)
                teams.append({"name": name, "spent": round(ts["spent"], 6), "reserved": round(ts["reserved"], 6), "projected": projected(ts),
                              "unconfigured": True})
        return {
            "month": {"start": start.isoformat(), "fraction": round(frac, 4)},
            "safeguards": g.model_dump(),
            "spend": {
                "org": {"spent": round(org_spent, 6), "projected": projected(org), "budget": g.org_budget_usd_month,
                        "api": round(org["api"], 6), "gpu": round(org["gpu"], 6), "gpu_idle": round(org["gpu_idle"], 6),
                        "reserved": round(org["reserved"], 6)},
                "gpu": {"spent": round(gpu_spent, 4), "projected": round(gpu_spent / frac, 2), "budget": g.gpu_budget_usd_month,
                        "by_backend": {k: round(v, 4) for k, v in gpu_by.items()}},
            },
            "fleet": await fleet(gateway),
            "gpu_controls": await gateway.store.get_setting("gpu_controls") or {},
            "controller": await controller_status(gateway.state),
            "teams": teams,
            "human_cost_usd": gateway.cfg.routing.human_cost_usd,
        }

    @router.post("/gpu/{backend}/control")
    async def control(backend: str, body: GpuControl, request: Request):
        p = await admin(request)
        b = gateway.cfg.backend(backend)
        if b is None or b.kind != "self_hosted" or b.serve is None:
            raise HTTPException(404, f"'{backend}' is not a self-hosted backend with a serve section")
        if body.mode == "on" and not body.replicas:
            raise HTTPException(422, "say how many replicas to keep running")
        controls = dict(await gateway.store.get_setting("gpu_controls") or {})
        deps = dict(controls.get("deployments") or {})
        if body.mode == "auto":
            deps.pop(backend, None)
            msg = f"GPU {backend} back to automatic: {p.user_id}" + (f" ({body.reason})" if body.reason else "")
        else:
            deps[backend] = {"mode": body.mode, "replicas": body.replicas if body.mode == "on" else None, "drain": body.drain,
                             "until": time.time() + body.hours * 3600 if body.hours else None, "reason": body.reason,
                             "by": p.user_id, "at": time.time()}
            what = (f"pinned to {body.replicas} replica(s)" if body.mode == "on"
                    else "stopping after warm sessions finish" if body.drain else "stopped now")
            msg = f"GPU {backend} {what}: by {p.user_id}" + (f" for {body.hours:g}h" if body.hours else "") + (f" ({body.reason})" if body.reason else "")
        controls["deployments"] = deps
        return await save(controls, p, "control", backend, body.model_dump(), msg)

    @router.post("/gpu/stop-all")
    async def stop_all(body: StopAll, request: Request):
        p = await admin(request)
        controls = dict(await gateway.store.get_setting("gpu_controls") or {})
        controls.update(paused=True, paused_by=p.user_id, paused_at=time.time(), reason=body.reason, drain=body.drain)
        return await save(controls, p, "stop_all", None, body.model_dump(),
                          f"All GPUs stopped: by {p.user_id}" + (f" ({body.reason})" if body.reason else "") + ". No GPUs will start until resumed.")

    @router.post("/gpu/resume")
    async def resume(request: Request):
        p = await admin(request)
        controls = dict(await gateway.store.get_setting("gpu_controls") or {})
        for k in ("paused", "paused_by", "paused_at", "reason", "drain"):
            controls.pop(k, None)
        return await save(controls, p, "resume", None, {}, f"GPU fleet resumed: by {p.user_id}. Deployments follow demand again.")

    @router.get("/gpu/events")
    async def events(request: Request, limit: int = Query(100, ge=1, le=1000), backend: str | None = None):
        p = await principal(request)
        if not p.at_least("team_lead"):
            raise HTTPException(403, "team leads, finance, and admins only")
        rows = await gateway.store.gpu_events(limit, backend)
        return {"events": [{**r, "ts": r["ts"].isoformat() if isinstance(r["ts"], datetime) else r["ts"]} for r in rows]}

    return router


def describe_control(controls: dict, backend: str, now: float | None = None) -> dict | None:
    """The control in force for a deployment, for display (the fleet-wide stop wins)."""
    if controls.get("paused"):
        return {"mode": "off", "fleet": True, "by": controls.get("paused_by"), "at": controls.get("paused_at"),
                "reason": controls.get("reason"), "drain": controls.get("drain", False)}
    return active_control(controls, backend, now or time.time())
