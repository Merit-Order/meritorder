"""Live data for the control panel (the Dispatch page and friends).

Admins and finance see the whole organization; team leads see their own team.
Users don't get these endpoints (they have /v1/reports/me).
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException, Query, Request

from . import spend
from .auth import AuthError, Principal
from .backends import describe
from .config import batch_api, effort_format
from .store import SAVINGS_CAUSES, month_start


def _speeds(speeds, backend: str, now: float) -> dict[str, dict]:
    out = {}
    for (name, model), sp in sorted(speeds.routes.items()):
        if name == backend and sp.samples:
            tps = sp.tokens_per_second()
            out[model] = {"first_token_ms": round((sp.first_token or 0) * 1000), "tokens_per_second": round(tps, 1) if tps else None,
                          "samples": sp.samples, "seconds_ago": round(now - sp.last)}
    return out


def _outage(o, now: float) -> dict:
    why = describe(o.status) if o.status else "repeated errors"
    return {"why": f"{why}: {o.message}" if o.message else why, "status": o.status, "seconds_left": max(0, int(o.until - now)),
            "capacity": o.capacity}


def build_router(gateway) -> APIRouter:
    router = APIRouter(prefix="/v1/console", tags=["console"])

    async def principal(request: Request) -> Principal:
        try:
            p = await gateway.authn.authenticate_console(request)
        except AuthError as e:
            raise HTTPException(e.status, e.message)
        if not p.at_least("team_lead"):
            raise HTTPException(403, "the control panel is for team leads, finance, and admins")
        return p

    def team_scope(p: Principal) -> str | None:
        return None if p.at_least("finance") else p.team

    @router.get("/overview")
    async def overview(request: Request):
        p = await principal(request)
        team = team_scope(p)
        now = datetime.now(UTC)
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        tomorrow = today + timedelta(days=1)
        store = gateway.store

        # Spend with GPU time in it (spend.py): for the organization the GPUs' accrued time, idle and starting included;
        # for a team its requests' shares of it.
        t_today, t_month = (await spend.total(store, gateway.cfg, today, tomorrow, team=team),
                            await spend.total(store, gateway.cfg, month_start(now), tomorrow, team=team))
        teams = [t for t in gateway.cfg.teams if team is None or t.name == team]
        # The organization's own budget when there is one: its spend has GPU time no team's budget counts.
        org_cap = gateway.cfg.safeguards.org_budget_usd_month if team is None else None
        budget = org_cap if org_cap is not None else sum(t.budget_usd_month or 0 for t in teams) or None
        elapsed_days = (now - month_start(now)).total_seconds() / 86400
        days_in_month = ((month_start(now) + timedelta(days=32)).replace(day=1) - month_start(now)).days
        month_spend = t_month["actual_cost"]
        health = await gateway.state.backend_health(list(gateway.backends), minutes=5)
        rpm = sum(h["requests"] for h in health.values()) / 5

        def pack(t):
            prompt = t["input_tokens"] + t["cache_read_tokens"] + t["cache_write_tokens"]
            return {"spend": t["actual_cost"], "api": t["api_cost"], "gpu": t["gpu_cost"], "gpu_idle": t["gpu_idle_cost"],
                    "baseline": t["baseline_cost"], "saved": t["savings_total"],
                    "requests": t["requests"], "cache_hit_rate": t["cache_read_tokens"] / prompt if prompt else 0,
                    "savings_by_cause": {c: t.get(f"savings_{c}", 0) for c in SAVINGS_CAUSES}}

        return {
            "scope": team or "organization",
            "today": pack(t_today),
            "month": pack(t_month),
            "budget": budget,
            # Set aside by running batches this month: not spent yet, but budgets count it (spend.budget_spend).
            "reserved": round((await spend.budget_spend(store, gateway.cfg, "team" if team else "org", team))["reserved"], 6),
            "forecast": month_spend / elapsed_days * days_in_month if elapsed_days > 0.25 else None,
            "requests_per_minute": round(rpm, 1),
            "gateways": await gateway.state.live_gateways(),
            "config_version": gateway.config_version,
        }

    @router.get("/merit")
    async def merit(request: Request, minutes: int = Query(15, ge=1, le=240)):
        """The dispatch stack: models in order of what a token actually cost, widths = tokens served."""
        p = await principal(request)
        now = datetime.now(UTC)
        rows = await gateway.store.summary(now - timedelta(minutes=minutes), now + timedelta(seconds=1), "model", team=team_scope(p))
        gpu_models = {m.name for m in gateway.cfg.models for r in m.routes
                      if r.backend in gateway.backends and gateway.backends[r.backend].kind == "self_hosted"
                      and gateway.backends[r.backend].available}
        tiers = []
        for r in rows:
            tokens = r["input_tokens"] + r["cache_read_tokens"] + r["cache_write_tokens"] + r["output_tokens"]
            if tokens <= 0:
                continue
            tiers.append({"model": r["key"], "tokens": tokens, "tokens_per_second": tokens / (minutes * 60),
                          "cost": r["actual_cost"], "price_per_m": r["actual_cost"] / tokens * 1e6,
                          "requests": r["requests"], "gpu": r["key"] in gpu_models})
        tiers.sort(key=lambda t: t["price_per_m"])
        total = sum(t["tokens"] for t in tiers)
        meaningful = [t for t in tiers if total and t["tokens"] / total >= 0.01]
        return {"minutes": minutes, "demand_tokens_per_second": total / (minutes * 60), "tiers": tiers,
                "clearing_price_per_m": meaningful[-1]["price_per_m"] if meaningful else None,
                "blended_price_per_m": sum(t["cost"] for t in tiers) / total * 1e6 if total else None}

    @router.get("/feed")
    async def feed(request: Request, limit: int = Query(60, ge=1, le=500)):
        p = await principal(request)
        if p.role == "finance":  # per-request rows (who, what, its error): finance sees aggregates only, as /v1/reports/requests
            raise HTTPException(403, "finance sees spend in aggregate; the request feed is for admins and team leads")
        team = team_scope(p)
        events = await gateway.state.recent_events(500 if team else limit)
        if team:
            events = [e for e in events if e.get("team") == team][:limit]
        return {"events": events}

    @router.get("/backends")
    async def backends(request: Request, minutes: int = Query(5, ge=1, le=30)):
        await principal(request)
        names = [b.name for b in gateway.cfg.backends]
        health = await gateway.state.backend_health(names, minutes=minutes)
        now = time.time()
        out = []
        for b in gateway.cfg.backends:
            live = gateway.backends.get(b.name)
            h = health.get(b.name, {})
            n = h.get("requests", 0)
            down = live.skipped(now) if live else None
            out.append({
                "name": b.name, "kind": b.kind, "format": b.format, "api": b.api, "host": urlparse(b.base_url).hostname if b.base_url else None,
                "enabled": b.enabled, "live": bool(live and live.available), "reason": b.disabled_reason,
                "circuit_open": bool(live and not live.healthy(now)), "draining": bool(live and live.draining),
                # Why it's being skipped and for how long; and models skipped on it (busy, or not deployed there).
                "skipped": _outage(down, now) if down else None,
                "skipped_models": {m: _outage(o, now) for m, o in (live.cooling_models(now).items() if live else [])},
                "effort_format": effort_format(b),
                # Its batch API (batches/providers.py), or why it has none: batches go only where there's one.
                "batch": dict(zip(("api", "why"), batch_api(b))),
                "models": sorted({m.name for m in gateway.cfg.models for r in m.routes if r.backend == b.name}),
                "requests": int(n), "errors": int(h.get("errors", 0)),
                "error_rate": h.get("errors", 0) / n if n else 0.0,
                "avg_latency_ms": h.get("latency_ms", 0) / n if n else None,
                "avg_ttft_ms": h.get("ttft_ms", 0) / h["ttft_n"] if h.get("ttft_n") else None,
                # What routing by speed uses: this gateway process's moving averages per upstream model.
                "speeds": _speeds(gateway.router.speeds, b.name, now),
            })
        return {"minutes": minutes, "backends": out}

    @router.get("/teams")
    async def teams(request: Request):
        """Each team's limits from config next to its spend this month."""
        p = await principal(request)
        team = team_scope(p)
        start = month_start()
        rows = {r["key"]: r for r in await gateway.store.summary(start, datetime.now(UTC) + timedelta(days=1), "team", team=team)}
        # What each team keeps in the Responses API's stored mode: counts and sizes, never content.
        stored = {r["team"]: r for r in await gateway.stored.sizes(team)}
        out = []
        names = [t.name for t in gateway.cfg.teams] + [k for k in rows if k not in {t.name for t in gateway.cfg.teams}]
        names += [k for k in stored if k not in names]
        for name in names:
            if team is not None and name != team:
                continue
            t = gateway.cfg.team(name)
            r = rows.get(name) or {}
            out.append({"name": name, "configured": any(x.name == name for x in gateway.cfg.teams),
                        "budget_usd_month": t.budget_usd_month, "user_budget_usd_month": t.user_budget_usd_month,
                        "max_session_usd": t.max_session_usd, "rate_limits": t.rate_limits.model_dump(),
                        "baseline_model": t.baseline_model or gateway.cfg.baseline_model,
                        "allowed_models": t.allowed_models, "allowed_backends": t.allowed_backends,
                        "compression": t.compression.enabled, "compression_model": t.compression.model.enabled,
                        "response_cache": t.response_cache.enabled,
                        "stored_responses": stored_view(t, stored.get(name)),
                        "judged_labels": {"enabled": t.judged_labels.enabled, "sample_rate": t.judged_labels.sample_rate,
                                          "max_usd_month": t.judged_labels.max_usd_month},
                        "spend": r.get("actual_cost", 0.0), "saved": r.get("savings_total", 0.0), "requests": r.get("requests", 0)})
        return {"teams": out, "stored_key": stored_key(p.role == "admin")}

    def stored_view(t, size: dict | None) -> dict:
        s = t.stored_responses
        size = size or {}
        return {"enabled": s.enabled, "retention_days": s.retention_days, "readers": s.readers,
                "responses": size.get("responses", 0), "conversations": size.get("conversations", 0), "items": size.get("items", 0),
                "bytes": size.get("response_bytes", 0) + size.get("conversation_bytes", 0), "people": size.get("people", 0),
                "running": size.get("running", 0), "old_key": size.get("old_key", 0),
                "oldest": size["oldest"].isoformat() if size.get("oldest") else None}

    def stored_key(admin: bool) -> dict:
        """Whether the gateway can seal stored responses, and with which key (its public id, never the key). Where the key
        is read from (a variable's name, a file's path) is for admins: it says where a secret sits on the gateway."""
        rs = gateway.cfg.response_store
        ring = gateway.stored.keyring()
        return {"configured": bool(rs.key()), "env": rs.key_env if admin else None, "file": rs.key_file if admin else None, "id": ring.current, "old_keys": len(ring.keys) - 1 if ring.usable else 0,
                "memory": gateway.store.kind == "memory"}

    @router.get("/attention")
    async def attention(request: Request):
        p = await principal(request)
        team = team_scope(p)
        items = []
        health = await gateway.state.backend_health(list(gateway.backends), minutes=5)
        now = time.time()
        for name, b in gateway.backends.items():
            h = health.get(name, {})
            n, errs = h.get("requests", 0), h.get("errors", 0)
            down = b.skipped(now)
            if down is not None:
                items.append({"level": "critical", "title": f"{name} is failing",
                              "detail": f"{_outage(down, now)['why']}. Skipped for {max(1, int(down.until - now))}s more; traffic goes to other backends.",
                              "page": "backends"})
            elif b.cooling_models(now):
                busy = b.cooling_models(now)
                items.append({"level": "warning", "title": f"{name}: {', '.join(sorted(busy))} skipped",
                              "detail": "; ".join(f"{m}: {_outage(o, now)['why']}" for m, o in sorted(busy.items()))
                                        + ". Their requests go to other backends meanwhile.", "page": "backends"})
            elif n >= 5 and errs / n >= 0.2:
                items.append({"level": "warning", "title": f"{name}: {errs / n:.0%} errors", "detail": f"{int(errs)} of {int(n)} requests failed in the last 5 minutes.", "page": "backends"})
        start = month_start()
        by_team = {t.name: (await spend.budget_spend(gateway.store, gateway.cfg, "team", t.name))["spent"]  # as admission counts it
                   for t in gateway.cfg.teams if t.budget_usd_month and (team is None or t.name == team)}
        for t in gateway.cfg.teams:
            if t.budget_usd_month and (team is None or t.name == team):
                used = by_team.get(t.name, 0.0) / t.budget_usd_month
                if used >= 0.8:
                    items.append({"level": "critical" if used >= 1 else "warning", "title": f"Team {t.name} at {used:.0%} of budget",
                                  "detail": f"${by_team.get(t.name, 0):,.2f} of ${t.budget_usd_month:,.0f} this month.", "page": "teams"})
        controls = await gateway.store.get_setting("gpu_controls") or {}
        if controls.get("paused"):
            items.append({"level": "warning", "title": "All GPUs are stopped",
                          "detail": f"By {controls.get('paused_by') or 'an admin'}" + (f": {controls['reason']}" if controls.get("reason") else "")
                                    + ". Traffic goes to APIs until someone resumes the fleet.", "page": "gpus"})
        running = False
        for b in gateway.cfg.backends:
            if b.kind == "self_hosted" and b.serve is not None and b.enabled:
                rec = await gateway.state.get_json(f"deploy:{b.name}") or {}
                running = running or rec.get("state", "api_only") != "api_only"
                if "failed" in (rec.get("reason") or ""):
                    items.append({"level": "critical" if "stop failed" in rec["reason"] else "warning",
                                  "title": f"GPU deployment {b.name}", "detail": rec["reason"], "page": "gpus"})
        if running:
            from .controls import controller_status

            if not (await controller_status(gateway.state))["alive"]:
                items.append({"level": "critical", "title": "Capacity controller is not running",
                              "detail": "GPUs are up but nothing is managing them: they won't scale down or stop on their own.", "page": "gpus"})
        if team is None:
            g = gateway.cfg.safeguards
            for label, budget, spent in (("Organization", g.org_budget_usd_month, None), ("GPU", g.gpu_budget_usd_month, None)):
                if not budget:
                    continue
                spent = ((await spend.budget_spend(gateway.store, gateway.cfg, "org"))["spent"] if label == "Organization"
                         else sum((await gateway.store.gpu_spend_since(start)).values()))
                used = spent / budget
                if used >= 0.8:
                    items.append({"level": "critical" if used >= 1 else "warning", "title": f"{label} spend at {used:.0%} of budget",
                                  "detail": f"${spent:,.2f} of ${budget:,.2f} this month."
                                            + (" Requests are being refused." if used >= 1 and label == "Organization" else "")
                                            + (" GPUs are draining and none will start." if used >= 1 and label == "GPU" else ""),
                                  "page": "controls"})
        cutoff = datetime.now(UTC) - timedelta(hours=24)
        for a in await gateway.state.recent_alerts(20):
            try:
                ts = datetime.fromisoformat(a["ts"])
            except (KeyError, ValueError):
                continue
            if ts >= cutoff and (team is None or a.get("team") in (None, team)):
                items.append({"level": a.get("level", "info"), "title": a.get("title", "Alert"), "detail": a.get("text", ""), "page": a.get("page"), "ts": a["ts"]})
        order = {"critical": 0, "warning": 1, "info": 2}
        items.sort(key=lambda i: order.get(i["level"], 3))
        return {"items": items}

    return router
