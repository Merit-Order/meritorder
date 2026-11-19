"""Reports API (docs/08-reporting.md). Metadata only; never prompt content.

Who sees what:
* user       -> their own usage
* team_lead  -> their team
* finance    -> everything, aggregated (no per-request detail)
* admin      -> everything

Every report says which store answered it ("postgres", "clickhouse", or "clickhouse+postgres"): in the
x-meritorder-answered-by header, and in the JSON as `answered_by`.
"""

from __future__ import annotations

import base64
import csv
import io
import json
import math
import re
import uuid
from datetime import UTC, date, datetime, timedelta
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .accounts import sign_in_status
from . import sky_clouds, spend
from .auth import AuthError, Principal
from .classifier import LEVEL_MEANINGS, label
from .config import LevelRules
from .store import (
    LEDGER_FIELDS,
    SORT_RANGE,
    SUM_FIELDS,
    RequestFilter,
    answered_by,
    collect_answers,
    month_start,
    savings_of,
    valid_status,
    validate_group,
)
from .spend import SPLIT_FIELDS
from .success import report as success_report

SORTS = Literal["time", "cost", "savings", "latency"]

COUNT_FIELDS = ["requests", "errors", "upgraded"]
ANSWERED_HEADER = b"x-meritorder-answered-by"


class AnsweredBy:
    """ASGI middleware: under /v1/reports, collect which stores answered the request's ledger queries (store.note_answer)
    and send it as the x-meritorder-answered-by header. Plain ASGI, so the endpoint runs in this context."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not scope["path"].startswith("/v1/reports"):
            return await self.app(scope, receive, send)
        seen = collect_answers()

        async def send_with_header(message):
            if message["type"] == "http.response.start" and seen:
                message = {**message, "headers": [*message.get("headers", []), (ANSWERED_HEADER, answered_by(seen).encode())]}
            await send(message)

        await self.app(scope, receive, send_with_header)


def _parse_day(value: str | None, default: datetime) -> datetime:
    if not value:
        return default
    try:
        return datetime.combine(date.fromisoformat(value), datetime.min.time(), tzinfo=UTC)
    except ValueError:
        raise HTTPException(400, f"dates must be YYYY-MM-DD, got {value!r}")


def _parse_time(value: str | None, name: str) -> datetime | None:
    """A day (YYYY-MM-DD, midnight UTC) or an ISO 8601 time; one without a zone is UTC."""
    if not value:
        return None
    try:
        if "T" in value or " " in value:
            t = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        else:
            t = datetime.combine(date.fromisoformat(value), datetime.min.time())
    except ValueError:
        raise HTTPException(400, f"{name} must be YYYY-MM-DD or an ISO 8601 time, got {value!r}")
    return t.replace(tzinfo=UTC) if t.tzinfo is None else t.astimezone(UTC)


_TAG_FILTER = re.compile(r"^([A-Za-z0-9_.-]{1,64})(?:=(.{1,200}))?$")


def _text(value: str | None) -> str | None:
    return (value or "").strip() or None


def request_filter(
    start: str | None = Query(None, description="YYYY-MM-DD or an ISO 8601 time, inclusive (default: no lower bound)"),
    end: str | None = Query(None, description="YYYY-MM-DD or an ISO 8601 time, exclusive (default: no upper bound)"),
    user: str | None = None,
    team: str | None = None,
    key: str | None = Query(None, description="an API key's id (key_...)"),
    session: str | None = Query(None, description="a session id"),
    requested_model: str | None = Query(None, description="the model the client asked for: auto, a name or an alias"),
    served_model: str | None = Query(None, description="the model that served it"),
    backend: str | None = None,
    format: str | None = Query(None, description="the client's API format (anthropic, openai, ...)"),
    status: str | None = Query(None, description="success, error (400 or more), a code (429), or a class (5xx)"),
    task: str | None = Query(None, description="the level's task type (coding, general)"),
    level: int | None = Query(None, ge=1, le=5),
    reason: str | None = Query(None, description="why it got its level and model: contains this, ignoring case"),
    tag: list[str] = Query([], description="name=value, or a name alone for any value; repeat it for more (all must match)"),
    min_cost: float | None = None,
    max_cost: float | None = None,
    min_savings: float | None = None,
    max_savings: float | None = None,
    savings_cause: Literal["routing", "caching", "self_hosting", "batch", "compression", "compression_model",
                           "response_cache"] | None = Query(None, description="rows where this cause saved (or cost) anything"),
    min_latency: int | None = Query(None, ge=0, description="milliseconds"),
    max_latency: int | None = Query(None, ge=0, description="milliseconds"),
    stream: bool | None = None,
    batch: bool | None = Query(None, description="through a provider's batch API (true) or live (false)"),
    cache_hit: bool | None = Query(None, description="read the prompt cache, or served from the response cache"),
    failover: bool | None = Query(None, description="another route was tried first and failed"),
) -> RequestFilter:
    """The Requests view's filters (docs/08-reporting.md). `user` and `team` are clamped to the caller's role after this."""
    tags: dict[str, str | None] = {}
    for t in tag:
        m = _TAG_FILTER.match(t.strip())
        if not m:
            raise HTTPException(400, f"tag filters are name=value or name, got {t!r}")
        tags[m.group(1)] = m.group(2)
    status = _text(status)
    if status is not None and not valid_status(status):
        raise HTTPException(400, f"status must be success, error, a code (429) or a class (5xx), got {status!r}")
    f = RequestFilter(start=_parse_time(start, "start"), end=_parse_time(end, "end"), user_id=_text(user), team=_text(team),
                      key_id=_text(key), session_id=_text(session), requested_model=_text(requested_model),
                      served_model=_text(served_model), backend=_text(backend), client_format=_text(format), status=status,
                      task=_text(task), level=level, reason=_text(reason), tags=tags, min_cost=min_cost, max_cost=max_cost,
                      min_savings=min_savings, max_savings=max_savings, savings_cause=savings_cause, min_latency=min_latency,
                      max_latency=max_latency, stream=stream, batch=batch, cache_hit=cache_hit, failover=failover)
    if f.start and f.end and f.end <= f.start:
        raise HTTPException(400, "end must be after start")
    return f


def _check_sort(f: RequestFilter, sort: str) -> None:
    """Time is the ledger's key, so a page by time is cheap on any range. Nothing indexes cost, savings or latency:
    sorting by one reads every row in the range, so the range is bounded."""
    if sort != "time" and (f.start is None or (f.end or datetime.now(UTC)) - f.start > SORT_RANGE):
        raise HTTPException(400, f"sorting by {sort} needs a time range of {SORT_RANGE.days} days or less "
                                 "(sorting by time works on any range)")


def _cursor(key: tuple, sort: str, order: str, forward: bool) -> str:
    """An opaque cursor: a row's (sort value, ts, id), which way to read from it, and the sort it belongs to."""
    v, ts, rid = key
    doc = {"s": sort, "o": order, "d": "n" if forward else "p", "v": None if sort == "time" else v,
           "t": ts.isoformat() if isinstance(ts, datetime) else ts, "i": str(rid)}
    return base64.urlsafe_b64encode(json.dumps(doc, separators=(",", ":")).encode()).decode().rstrip("=")


def _row_key(row: dict) -> tuple:
    return row["_sort"], row["ts"], row["id"]


def _read_cursor(value: str, sort: str, order: str) -> tuple[tuple, bool]:
    """(sort value, ts, id) and whether the page is after it (next) or before it (previous)."""
    try:
        doc = json.loads(base64.urlsafe_b64decode(value + "=" * (-len(value) % 4)))
        ts = datetime.fromisoformat(doc["t"])
        v = ts if sort == "time" else doc["v"]
        if (doc["s"] != sort or doc["o"] != order or doc["d"] not in ("n", "p") or ts.tzinfo is None
                or isinstance(v, bool) or not isinstance(v, (int, float, datetime))):
            raise ValueError
        key = (v, ts, uuid.UUID(doc["i"]))
    except (ValueError, KeyError, TypeError, AttributeError):
        raise HTTPException(400, "that cursor isn't one this view gave out for this sort and order: start from the first page")
    return key, doc["d"] == "n"


def _scope(p: Principal, user: str | None, team: str | None) -> tuple[str | None, str | None]:
    """Clamp the requested filters to what this principal may see."""
    if p.at_least("finance"):
        return user, team
    if p.role == "team_lead":
        return user, p.team
    return p.user_id, None


def _totals(rows: list[dict]) -> dict:
    t = {f: 0 for f in COUNT_FIELDS + SUM_FIELDS + ["savings_total", *SPLIT_FIELDS]}
    for r in rows:
        for f in t:
            t[f] += r.get(f) or 0
    return {k: round(v, 6) if isinstance(v, float) else v for k, v in t.items()}


_FORMULA_START = ("=", "+", "-", "@", "\t", "\r")


def csv_cell(value):
    """A text cell a spreadsheet would read as a formula (it starts with = + - @, a tab or a carriage return) gets a
    leading apostrophe, so a client-chosen value (a session id, a tag, a model name) can't run as one when an admin
    opens the CSV. Numbers are left as they are."""
    if isinstance(value, str) and value.startswith(_FORMULA_START):
        return "'" + value
    return value


class _SafeWriter:
    """csv.writer whose rows go through csv_cell."""

    def __init__(self, buf):
        self._w = csv.writer(buf)

    def writerow(self, row) -> None:
        self._w.writerow([csv_cell(v) for v in row])


def csv_writer(buf) -> _SafeWriter:
    return _SafeWriter(buf)


def build_router(gateway) -> APIRouter:
    router = APIRouter(prefix="/v1/reports", tags=["reports"])

    async def principal(request: Request) -> Principal:
        try:
            return await gateway.authn.authenticate_console(request)
        except AuthError as e:
            raise HTTPException(e.status, e.message)

    async def summary_rows(p, start, end, group_by, user, team):
        try:
            validate_group(group_by)
        except ValueError as e:
            raise HTTPException(400, str(e))
        user, team = _scope(p, user, team)
        rows = await spend.summary(gateway.store, gateway.cfg, start, end, group_by, user_id=user, team=team)
        return rows, user, team

    @router.get("/summary")
    async def summary(
        request: Request,
        start: str | None = Query(None, description="YYYY-MM-DD, inclusive (default: first of this month)"),
        end: str | None = Query(None, description="YYYY-MM-DD, exclusive (default: tomorrow)"),
        group_by: str = Query("model", description="user, team, model, backend, day, hour, month, none, or tag:<name>"),
        user: str | None = None,
        team: str | None = None,
    ):
        p = await principal(request)
        s = _parse_day(start, month_start())
        e = _parse_day(end, datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1))
        rows, user, team = await summary_rows(p, s, e, group_by, user, team)
        return {"start": s.isoformat(), "end": e.isoformat(), "group_by": group_by, "user": user, "team": team,
                "rows": rows, "totals": _totals(rows), "answered_by": answered_by()}

    @router.get("/summary.csv")
    async def summary_csv(request: Request, start: str | None = None, end: str | None = None, group_by: str = "model",
                          user: str | None = None, team: str | None = None):
        p = await principal(request)
        s = _parse_day(start, month_start())
        e = _parse_day(end, datetime.now(UTC) + timedelta(days=1))
        rows, _, _ = await summary_rows(p, s, e, group_by, user, team)
        buf = io.StringIO()
        fields = ["key"] + COUNT_FIELDS + SUM_FIELDS + ["savings_total", *SPLIT_FIELDS]
        w = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows({k: csv_cell(v) for k, v in r.items()} for r in rows)
        return Response(buf.getvalue(), media_type="text/csv",
                        headers={"content-disposition": f'attachment; filename="meritorder-{group_by.replace(":", "-")}.csv"'})

    @router.get("/people")
    async def people(request: Request):
        """What to call each person the caller may see in reports (their name, else their email address), by id: an SSO
        person's id is the identity provider's GUID, which the console shouldn't show. Finance and admins: everyone; a team
        lead: their team; a user: themselves."""
        p = await principal(request)
        if p.at_least("finance"):
            rows = await gateway.store.list_users()
        elif p.role == "team_lead":
            rows = await gateway.store.list_users(p.team)
        else:
            rows = [r for r in [await gateway.store.get_user(p.user_id)] if r]
        return {"people": {r["id"]: {"name": r.get("display_name"), "email": r.get("email")}
                           for r in rows if r.get("display_name") or r.get("email")}}

    @router.get("/me")
    async def me(request: Request):
        p = await principal(request)
        team_cfg = gateway.cfg.team(p.team)
        start = month_start()
        rows = await spend.summary(gateway.store, gateway.cfg, start, datetime.now(UTC) + timedelta(days=1), "model", user_id=p.user_id)
        totals = _totals(rows)
        record = await gateway.store.get_user(p.user_id) or {}
        return {
            "user": p.user_id,
            # What to call them: an SSO person's id is the identity provider's (a GUID for Entra), not something to show.
            "name": record.get("display_name"),
            "email": record.get("email"),
            "team": p.team,
            "role": p.role,
            "month": start.strftime("%Y-%m"),
            "spend_usd": round(await gateway.spend("user", p.user_id), 6),
            "budget_usd": p.monthly_limit(team_cfg.user_budget_usd_month),
            "team_spend_usd": round(await gateway.spend("team", p.team), 6) if p.at_least("team_lead") else None,
            "team_budget_usd": team_cfg.budget_usd_month,
            # The spend's split: APIs, and the person's requests' shares of GPU time (decision 64).
            "api_usd": totals["api_cost"],
            "gpu_usd": totals["gpu_cost"],
            "baseline_usd": totals["baseline_cost"],
            "savings_usd": totals["savings_total"],
            "requests": totals["requests"],
            "by_model": rows,
            "answered_by": answered_by(),
            "sign_in": await sign_in_status(gateway, p),
        }

    @router.get("/export.jsonl")
    async def export(request: Request, start: str | None = None, end: str | None = None):
        """Per-request metadata as JSON lines (no prompt or response content)."""
        p = await principal(request)
        if p.role == "finance":
            raise HTTPException(403, "finance role sees aggregates only; use /v1/reports/summary.csv (by team, tag, model, or day)")
        if not p.at_least("team_lead"):
            raise HTTPException(403, "team leads and admins only")
        s = _parse_day(start, month_start())
        e = _parse_day(end, datetime.now(UTC) + timedelta(days=1))
        team = None if p.role == "admin" else p.team
        rows = gateway.store.export_rows(s, e, team)
        first = await anext(rows, None)  # chooses the store (and reaches it) before the headers say which one answered

        async def lines():
            if first is None:
                return
            yield json.dumps(first, default=str, separators=(",", ":")) + "\n"
            async for row in rows:
                yield json.dumps(row, default=str, separators=(",", ":")) + "\n"

        name = f"meritorder-ledger-{s:%Y%m%d}-{e:%Y%m%d}.jsonl"
        return StreamingResponse(lines(), media_type="application/x-ndjson",
                                 headers={"content-disposition": f'attachment; filename="{name}"'})

    @router.get("/gpu")
    async def gpu(request: Request):
        """State of each self-hosted deployment, as published by the capacity controller."""
        from .controls import controller_status, describe_control, fleet

        p = await principal(request)
        if not p.at_least("team_lead"):
            raise HTTPException(403, "team leads, finance, and admins only")
        now = datetime.now(UTC)
        controls = await gateway.store.get_setting("gpu_controls") or {}
        month_cost = await gateway.store.gpu_spend_since(month_start(now))
        out = []

        def kube_spec(serve) -> dict:
            """Where a Kubernetes deployment runs: which API, namespace and node type (and that type's configured price)."""
            from .config import KubernetesServe

            k = serve.kubernetes or KubernetesServe()
            node = gateway.cfg.kubernetes.node_types.get(k.node_type) if k.node_type else None
            return {"api": k.api, "namespace": k.namespace or gateway.cfg.kubernetes.namespace, "node_type": k.node_type,
                    "node_usd_per_hour": node.usd_per_hour if node else None, "node_gpus": node.gpus if node else None,
                    "inference_pool": k.inference_pool}

        def stranded(backend: str) -> list[str]:
            """Chat models this deployment serves that have no API route: requests naming them fail while it's off."""
            names = []
            for m in gateway.cfg.models:
                if m.kind != "chat" or not any(r.backend == backend for r in m.routes):
                    continue
                if not any(r.backend in gateway.backends and gateway.backends[r.backend].kind != "self_hosted" for r in m.routes):
                    names.append(m.name)
            return names

        from .affinity import placements

        for b in gateway.cfg.backends:
            if b.kind != "self_hosted":
                continue
            rec = await gateway.state.get_json(f"deploy:{b.name}") or {}
            serve = b.serve
            # Where sessions sit on its replicas and how warm their caches probably are: every session for admins, a
            # team lead's own team's, none for finance (the per-replica counts only).
            rows, per_replica = await placements(gateway.state, b.name, rec, b.cache_ttl_seconds, now.timestamp())
            rows = rows if p.role == "admin" else [r for r in rows if p.role == "team_lead" and r.get("team") == p.team]
            out.append({"backend": b.name, "enabled": b.enabled, "managed": serve is not None,
                        "placements": rows, "replica_sessions": per_replica,
                        "affinity": {"enabled": serve.replica_affinity, "kv_events": serve.kv_events} if serve else None,
                        "model": serve.model if serve else None, "provisioner": serve.provisioner if serve else None,
                        "state": rec.get("state", "api_only" if b.enabled else "disabled"), "gpu_usd_per_hour": b.gpu_cost_per_hour,
                        "month_cost": round(month_cost.get(b.name, 0.0), 4),
                        "only_route_for": stranded(b.name),
                        "control": describe_control(controls, b.name) if serve else None,
                        "spec": {"accelerators": serve.accelerators, "use_spot": serve.use_spot, "clouds": serve.clouds, "regions": serve.regions,
                                 "min_replicas": serve.min_replicas, "max_replicas": serve.max_replicas,
                                 "location": serve.location, "drain_minutes": serve.drain_minutes,
                                 # SkyPilot: how long a replica keeps running without the controller, then sits idle
                                 "lease_minutes": serve.lease_minutes if serve.provisioner == "skypilot" else None,
                                 "autostop_minutes": serve.autostop_minutes if serve.provisioner == "skypilot" else None,
                                 "cold_start_minutes": serve.cold_start_minutes, "hold_minutes": serve.hold_minutes,
                                 "load_format": serve.load_format, "weights_cache": serve.weights_cache,
                                 "weights_cache_infra": serve.weights_cache_infra, "weights_cache_gb": serve.weights_cache_gb,
                                 "weights_cache_usd_per_month": sky_clouds.volume_usd_per_month(
                                     serve.weights_cache_infra, serve.weights_cache_gb, serve.weights_cache_usd_per_gb_month)
                                 if serve.weights_cache == "volume" else None,
                                 "kubernetes": kube_spec(serve) if serve.provisioner == "kubernetes" else None} if serve else None,
                        **{k: v for k, v in rec.items() if k != "state"}})
        g = gateway.cfg.safeguards
        from .gpu import VOLUMES_SETTING

        volumes = [{"name": n, **v} for n, v in sorted((await gateway.store.get_setting(VOLUMES_SETTING) or {}).items())]
        return {"deployments": out, "paused": describe_control(controls, "") if controls.get("paused") else None, "volumes": volumes,
                "controller": await controller_status(gateway.state), "fleet": await fleet(gateway),
                "gpu_month": {"spent": round(sum(month_cost.values()), 4), "budget": g.gpu_budget_usd_month},
                "limits": {"max_gpu_usd_per_hour": g.max_gpu_usd_per_hour, "max_gpu_replicas": g.max_gpu_replicas},
                "daily": await gateway.store.gpu_daily(now - timedelta(days=30)), "answered_by": answered_by()}

    @router.get("/gpu/candidates")
    async def gpu_candidates(request: Request):
        """Chat models with open weights (an hf_id): the ones that could be self-hosted."""
        p = await principal(request)
        team = gateway.cfg.team(p.team)
        return {"models": [{"id": m.name, "hf_id": m.hf_id, "quality": m.quality} for m in gateway.cfg.models
                           if m.kind == "chat" and m.hf_id and team.allows_model(m.name)]}

    @router.get("/gpu/whatif")
    async def gpu_whatif(request: Request, model: str, gpu_usd_per_hour: float = Query(..., gt=0),
                         capacity_tokens_per_second: float = Query(2000, gt=0), days: int = Query(30, ge=1, le=90)):
        """Replay recent traffic: what would self-hosting `model` have saved?"""
        from .gpu import simulate

        p = await principal(request)
        if not p.at_least("team_lead"):
            raise HTTPException(403, "team leads, finance, and admins only")
        m = gateway.cfg.model(model)
        if m is None or m.kind != "chat":
            raise HTTPException(404, f"chat model '{model}' is not configured")
        rows = await gateway.store.demand_by_hour(datetime.now(UTC) - timedelta(days=days))
        result = simulate(m, gateway.cfg, rows, gpu_usd_per_hour, capacity_tokens_per_second)
        return {**result, "days": days, "gpu_usd_per_hour": gpu_usd_per_hour, "capacity_tokens_per_second": capacity_tokens_per_second,
                "self_hostable": bool(m.hf_id), "answered_by": answered_by(),
                "note": "Upper bound: ignores cold starts and hysteresis. Demand is traffic this model's quality could serve, at API prices."}

    @router.get("/gpu/breakeven")
    async def gpu_breakeven(request: Request, days: int = Query(7, ge=1, le=90), utilization: float = Query(0.7, gt=0, le=1),
                            efficiency: float = Query(0.1, gt=0, le=1), spot: bool = True,
                            weights: Literal["published", "fp8", "4bit"] = "published", clouds: str | None = None):
        """How much traffic a GPU replica needs to pay for itself, by quality level and by open-weight model, beside the
        traffic of the last `days`. Each model's replica is sized to it and priced from SkyPilot's catalog. When the
        stored prices are over 6 hours old, a refresh starts in the background and `prices.refreshing` says so: the
        stored prices are answered at once, and the console asks again until it's done. `clouds` (comma-separated)
        narrows where. `weights`:
        served as published, or quantized (`fp8`: a 16-bit checkpoint in FP8; `4bit`: everything above 4 bits in 4)."""
        from . import gpu_sizing
        from .config import ServeConfig
        from .gpu import breakeven

        p = await principal(request)
        if not p.at_least("team_lead"):
            raise HTTPException(403, "team leads, finance, and admins only")
        now = datetime.now(UTC)
        catalog, sizes = await gpu_sizing.current(gateway.cfg, gateway.store)
        offered = sorted({o["cloud"] for o in catalog.get("offers") or []})
        chosen = {c.strip().lower() for c in clouds.split(",") if c.strip()} if clouds else set(offered)
        prices = {}
        for b in gateway.cfg.backends:
            if b.kind == "self_hosted":
                rec = await gateway.state.get_json(f"deploy:{b.name}") or {}
                if (rec.get("price") or {}).get("usd_per_hour"):
                    prices[b.name] = float(rec["price"]["usd_per_hour"])
        plans = gpu_sizing.replica_plans(gateway.cfg, catalog, sizes, chosen, spot, weights, replica_prices=prices)
        rows = await gateway.store.demand_by_hour(now - timedelta(days=days))
        # Averages over the time there's been traffic in the window, so a gateway that started yesterday isn't diluted.
        first = min((r["hour"] for r in rows), default=now)
        hours = max(1.0, min(days * 24.0, (now - first).total_seconds() / 3600))
        live = {name for name, b in gateway.backends.items() if b.available}
        margin = ServeConfig.model_fields["margin"].default  # what the controller asks of demand by default
        result = breakeven(gateway.cfg, live, rows, hours, plans, utilization, margin, efficiency)
        result["assumptions"].update({"spot": spot, "weights": weights, "clouds": sorted(chosen), "fit": gpu_sizing.FIT})
        return {**result, "days": days, "answered_by": answered_by(),
                "prices": {"fetched_at": catalog.get("fetched_at"), "catalog_updated": catalog.get("catalog_updated"),
                           "skypilot": catalog.get("skypilot"), "clouds": offered, "problem": catalog.get("problem"),
                           "refreshing": bool(catalog.get("refreshing")), "offers": len(catalog.get("offers") or [])},
                "sizes": {"fetched_at": sizes.get("fetched_at")}}

    @router.post("/gpu/breakeven/refresh")
    async def gpu_breakeven_refresh(request: Request):
        """Get the latest GPU prices now: SkyPilot downloads its catalogs again, and model sizes are fetched again. A
        refresh already running is joined rather than run twice. Without SkyPilot here, the GPU controller is asked to
        (202), and the report shows them once it has."""
        from . import gpu_sizing
        from .controller import PRICES_REQUESTED
        from .gpu import sky_available

        p = await principal(request)
        if not p.at_least("team_lead"):
            raise HTTPException(403, "team leads, finance, and admins only")
        if not sky_available():
            await gateway.state.set_json(PRICES_REQUESTED, {"by": p.user_id, "at": datetime.now(UTC).isoformat(timespec="seconds")})
            await gateway.state.set_json("controller:kick", uuid.uuid4().hex)  # wake it now rather than at its next cycle
            heartbeat = await gateway.state.get_json("controller:heartbeat")
            return JSONResponse({"requested": True, "controller": bool(heartbeat)}, status_code=202)
        try:
            return await gpu_sizing.refresh_once(gateway.cfg, gateway.store, download=True)
        except gpu_sizing.SkyPilotMissing as e:
            raise HTTPException(503, str(e))
        except Exception as e:  # noqa: BLE001 (the console shows why)
            raise HTTPException(502, f"couldn't get prices: {e}"[:300])

    @router.get("/levels")
    async def levels(request: Request):
        """How `auto` gives a request its level: the rules in effect (routing.levels) and their defaults, the caps on
        Claude Code's side jobs, and for each level the models good enough for it that the caller's team may use,
        cheapest first. Any signed-in person may look: it's what decides where their requests go."""
        p = await principal(request)
        cfg = gateway.cfg
        team = cfg.team(p.team)

        def price(m) -> float:
            live = [r.cost_price() for r in m.routes if r.priced and (b := gateway.backends.get(r.backend)) and b.available]
            return min((x.input + x.output for x in live), default=math.inf)

        usable = [(m, price(m)) for m in cfg.models if m.kind == "chat" and team.allows_model(m.name)]
        usable = sorted(((m, c) for m, c in usable if c != math.inf), key=lambda mc: (mc[1], mc[0].name))
        rules = cfg.routing.levels
        words = ("coding_words", "simple_words", "hard_words", "hardest_words", "analysis_words")
        choice, minimum = gateway.router.model_choice(team)

        def in_reach(task: str, n: int) -> list[dict]:
            """With cost per success or expected total cost, the models below the level that may take its work too: their
            chance there is at least the minimum (router._in_reach)."""
            if choice not in ("cost_per_success", "expected_cost") or task not in cfg.threshold_chance:
                return []
            return [{"name": m.name, "score": round(m.q(task), 1), "price": round(c, 4), "chance": chance}
                    for m, c in usable if 0 < m.q(task) < n and (chance := m.chances.get(task, {}).get(n)) is not None
                    and chance >= minimum]

        return {
            "auto_model": cfg.routing.auto_model,
            "rules": rules.model_dump(),
            "defaults": LevelRules().model_dump(),
            "labels": {w: [label(x) for x in getattr(rules, w)] for w in words},
            "request_class_levels": cfg.routing.request_class_levels,
            "model_choice": choice,
            "min_success_chance": minimum,
            "human_cost_usd": gateway.router.human_cost(team),
            "max_escalations": cfg.routing.cascade.max_escalations,
            "levels": [{"level": n, "means": LEVEL_MEANINGS[n],
                        "models": {task: [{"name": m.name, "score": round(m.q(task), 1),
                                           "price": round(c, 4)} for m, c in usable if m.q(task) >= n]
                                   for task in ("coding", "general")},
                        "in_reach": {task: in_reach(task, n) for task in ("coding", "general")}}
                       for n in range(1, 6)],
        }

    @router.get("/levels/outcomes")
    async def level_outcomes(request: Request, days: int = Query(30, ge=1, le=90)):
        """Your own traffic as the routing benchmark (decision 59): per task type and level, traffic and spend share,
        success rates by exploration arm (served, one level down, one level up, a canary), whether the level looks too
        high or too low, recommendations, and the catalog and score changes on the trend. Admins see everyone, team leads
        their team (recommendations are the organization's, so they're for admins)."""
        from . import benchmark

        p = await principal(request)
        if not p.at_least("team_lead") or p.role == "finance":
            raise HTTPException(403, "admins and team leads only")
        since = datetime.now(UTC) - timedelta(days=days)
        scope = None if p.at_least("admin") else p.team
        rows = await gateway.store.route_features(since, team=scope, text=False)
        daily = await gateway.store.floor_daily(since, scope)
        doc = await gateway.store.get_setting(benchmark.SETTING) or {}
        events = [e for e in doc.get("events") or [] if e.get("at", "") >= since.isoformat()[:19]]
        rep = benchmark.outcomes_report(rows, daily, gateway.cfg, events, await gateway.store.get_setting(benchmark.DISMISSED))
        rep["canaries"] = list(gateway.cfg.canaries)
        if not p.at_least("admin"):
            rep["recommendations"] = []
            for row in rep["levels"]:
                row["recommendation"] = None
        return {"days": days, "since": since.isoformat(), "team": scope, **rep, "answered_by": answered_by()}

    async def bills_report(request: Request, start: str | None, end: str | None, by: str) -> dict:
        from .billing import report

        p = await principal(request)
        if not p.at_least("finance"):
            raise HTTPException(403, "finance and admins only")
        s = _parse_day(start, month_start()).date()
        e = _parse_day(end, datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)).date()
        if e <= s:
            raise HTTPException(400, "end must be after start")
        return {**await report(gateway.cfg, gateway.store, s, e, by), "answered_by": answered_by()}

    @router.get("/bills")
    async def bills(request: Request, start: str | None = Query(None, description="YYYY-MM-DD, inclusive (default: first of this month)"),
                    end: str | None = Query(None, description="YYYY-MM-DD, exclusive (default: tomorrow)"),
                    by: Literal["day", "month"] = "day"):
        """What the ledger recorded against what each provider billed, by day (or month), provider and model; the bill lines that
        matched no model, and why; GPU time against the cloud bills; and spend no bill covers (docs/08-reporting.md)."""
        return await bills_report(request, start, end, by)

    @router.get("/bills.csv")
    async def bills_csv(request: Request, start: str | None = None, end: str | None = None, by: Literal["day", "month"] = "day"):
        rep = await bills_report(request, start, end, by)
        buf = io.StringIO()
        w = csv_writer(buf)
        w.writerow(["period", "source", "model", "status", "recorded", "billed", "difference", "pct", "requests"])
        for r in rep["rows"]:
            w.writerow([r["period"], r["source"], r["model"], r["status"], r["recorded"], r["billed"], r["difference"],
                        "" if r["pct"] is None else r["pct"], r["requests"]])
        for u in rep["unmatched"]:
            w.writerow([f"{u['first']}..{u['last']}", u["source"], u["line"], f"{u['status']}: {u['why']}", "", u["cost"], "", "", ""])
        for g in rep["gpu"]:
            w.writerow([g["period"], "gpu", g["deployment"], "partial" if g["partial"] else "gpu", g["recorded"], g["billed"],
                        g["difference"], "" if g["pct"] is None else g["pct"], ""])
        return Response(buf.getvalue(), media_type="text/csv",
                        headers={"content-disposition": f'attachment; filename="meritorder-bills-{rep["start"]}.csv"'})

    @router.post("/bills/refresh")
    async def bills_refresh(request: Request):
        """Fetch every bill source now, from this gateway (with its credentials), instead of waiting for the worker."""
        from .billing import refresh_once

        p = await principal(request)
        if not p.at_least("admin"):
            raise HTTPException(403, "admins only")
        if not any(s.enabled for s in gateway.cfg.billing.sources):
            raise HTTPException(400, "no bill sources are configured (billing.sources)")
        return {"sources": await refresh_once(gateway.cfg, gateway.store, gateway.state)}

    @router.get("/router")
    async def learned_router(request: Request, days: int = Query(7, ge=1, le=90), limit: int = Query(100, ge=1, le=1000)):
        """The learned router's shadow report: where it and the rules gave different levels, what each side's pick was
        estimated to cost for the task, and the outcome signals (seen on the side that acted, expected on both). Admins
        see everyone; team leads their team."""
        from . import learned_jobs

        p = await principal(request)
        if not p.at_least("team_lead") or p.role == "finance":
            raise HTTPException(403, "admins and team leads only")
        now = datetime.now(UTC)
        since = now - timedelta(days=days)
        scope = None if p.at_least("admin") else p.team
        rows = await gateway.store.route_features(since, team=scope, text=False)
        status = await learned_jobs.status(gateway.cfg, gateway.store, gateway.learned, gateway.state)
        if not p.at_least("admin"):
            status = {k: v for k, v in status.items() if k not in ("versions", "last_training")}
        rep = learned_jobs.report(rows, limit)
        rep["judged"].update(await judged_spend(since, now, scope, rows))
        rep["success"] = success_report(rows)
        return {"days": days, "since": since.isoformat(), "status": status, **rep, "answered_by": answered_by()}

    async def judged_spend(since: datetime, now: datetime, team: str | None, rows: list[dict]) -> dict:
        """What judged labels cost in the period, and each team's month so far against its cap (when the cap is spent, no
        new sample starts until next month). From Postgres, by the internal clients' user ids, as budgets read spend: like
        the rest of this report, never from ClickHouse."""
        from .judged import JUDGED_TAG, internal_user

        # Teams that turned judged labels on, and any whose requests were sampled in the period (a team without settings
        # of its own uses the default team's, with a cap of its own).
        names = sorted(n for n in {t.name for t in gateway.cfg.teams if t.judged_labels.enabled}
                       | {r["team"] for r in rows if r.get("judged") and r.get("team")} if team is None or n == team)
        cost = await gateway.store.spend_by_tag([internal_user(n) for n in names], JUDGED_TAG, since) if names else {}
        teams = []
        for name in names:
            s = gateway.cfg.team(name).judged_labels
            spent = await gateway.store.spend_since("user_id", internal_user(name), month_start(now))
            teams.append({"team": name, "enabled": s.enabled, "sample_rate": s.sample_rate, "cap": s.max_usd_month,
                          "spent_month": round(spent, 6), "capped": bool(s.max_usd_month) and spent >= s.max_usd_month,
                          "stronger": s.stronger if s.stronger != "auto" else f"auto:{s.stronger_level}", "judges": s.judges})
        replay, judge = cost.get("replay", 0.0), cost.get("judge", 0.0)
        return {"cost": {"replay": round(replay, 6), "judge": round(judge, 6), "total": round(replay + judge, 6)}, "teams": teams}

    @router.get("/batches")
    async def batches(request: Request, start: str | None = Query(None, description="YYYY-MM-DD, inclusive (default: first of this month)"),
                      end: str | None = Query(None, description="YYYY-MM-DD, exclusive (default: tomorrow)"),
                      limit: int = Query(200, ge=1, le=5000)):
        """Batches made in the period (newest first): their state, request counts, where their requests went, what was
        estimated and reserved against what the results cost, and what they saved. Scoped like the other reports."""
        from .batches.service import counts_of, openai_status

        p = await principal(request)
        s = _parse_day(start, month_start())
        e = _parse_day(end, datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1))
        user, team = _scope(p, None, None)
        rows = await gateway.store.list_batches(user_id=user, team=team, start=s, end=e, limit=limit)
        gpus = {bc.name for bc in gateway.cfg.backends if bc.kind == "self_hosted"}
        out = []
        for b in rows:
            status = openai_status(b)
            providers = b.get("providers") or {}
            out.append({"id": b["id"], "created_at": b["created_at"].isoformat(), "user": b["user_id"], "team": b["team"],
                        "format": b["client_format"], "endpoint": b.get("endpoint"), "status": status, "requests": b["request_count"],
                        "counts": counts_of(b), "providers": providers, "on_gpus": sum(v for k, v in providers.items() if k in gpus),
                        "savings_self_hosting": round(b.get("savings_self_hosting") or 0, 6),
                        "estimated_cost": round(b["estimated_cost"], 6), "reserved_cost": round(b["reserved_cost"], 6),
                        "actual_cost": round(b["actual_cost"], 6), "baseline_cost": round(b["baseline_cost"], 6),
                        "savings_batch": round(b["savings_batch"], 6), "savings_total": round(b["savings_total"], 6),
                        "ended_at": b["ended_at"].isoformat() if b.get("ended_at") else None,
                        "expires_at": b["expires_at"].isoformat() if b.get("expires_at") else None,
                        "cancelled_by": b.get("cancelled_by"), "purged": bool(b.get("purged_at")),
                        "errors": [x.get("message") for x in ((b.get("errors") or {}).get("data") or [])][:5],
                        "can_cancel": b["status"] not in ("ended", "finalizing") and not b.get("cancel_requested_at")
                        and (p.role == "admin" or b["user_id"] == p.user_id)})
        totals = {k: round(sum(r[k] for r in out), 6) for k in ("estimated_cost", "reserved_cost", "actual_cost", "savings_batch",
                                                                "savings_self_hosting", "savings_total")}
        totals.update(batches=len(out), requests=sum(r["requests"] for r in out), on_gpus=sum(r["on_gpus"] for r in out),
                      running=sum(1 for r in out if r["status"] in ("in_progress", "cancelling", "finalizing")))
        doc = {"start": s.isoformat(), "end": e.isoformat(), "rows": out, "totals": totals,
               "retention_days": gateway.cfg.batches.retention_days, "gpu_backends": sorted(gpus)}
        if p.at_least("admin"):  # where content is held and the batch key's rotation (batches/content.py)
            g = gateway.cfg.batches.gpu
            doc["storage"] = {**await (await gateway.batches.content()).status(),
                              "gpu": {"enabled": g.enabled, "handoff_hours": g.handoff_hours, "priority": g.priority}}
        return doc

    async def requests_scope(request: Request, f: RequestFilter, sort: str) -> RequestFilter:
        """The filter clamped to what the caller may see: their own requests, their team's, or everyone's. Every other
        filter is AND-ed with that, so none can widen it."""
        p = await principal(request)
        if p.role == "finance":
            raise HTTPException(403, "finance role sees aggregates only; use /v1/reports/summary")
        f.user_id, f.team = _scope(p, f.user_id, f.team)
        _check_sort(f, sort)
        return f

    @router.get("/requests")
    async def requests(request: Request, f: RequestFilter = Depends(request_filter),
                       limit: int = Query(50, ge=1, le=1000, description="rows a page (the console offers 25, 50, 100 and 200)"),
                       cursor: str | None = Query(None, description="`next` or `prev` from the page before"),
                       sort: SORTS = "time", order: Literal["desc", "asc"] = "desc"):
        """The ledger a page at a time, newest first unless sorted otherwise, with the Requests view's filters
        (docs/08-reporting.md). Keyset paging on (sort value, ts, id): `next` and `prev` are opaque cursors, and rows
        written meanwhile never shift a page. `total` is exact up to 10,000 rows and the planner's estimate past that
        (`total_exact` false)."""
        f = await requests_scope(request, f, sort)
        desc = order == "desc"
        key, forward = _read_cursor(cursor, sort, order) if cursor else (None, True)
        store = gateway.store
        rows = await store.requests_page(f, sort, desc, key, forward, limit + 1)
        more = len(rows) > limit
        rows = rows[:limit] if forward else rows[-limit:]
        if not forward and not more:  # back at the start: the first page, whole
            key, forward = None, True
            rows = await store.requests_page(f, sort, desc, None, True, limit + 1)
            more = len(rows) > limit
            rows = rows[:limit]
        if key is None:
            prev = None
        elif not rows:  # past the end (rows were deleted, or an old link): back from where it was
            prev = _cursor(key, sort, order, False)
        else:
            prev = _cursor(_row_key(rows[0]), sort, order, False)
        nxt = _cursor(_row_key(rows[-1]), sort, order, True) if rows and (more or not forward) else None
        total, exact = await store.requests_count(f)
        for r in rows:
            r.pop("_sort", None)
        return JSONResponse({"rows": rows, "next": nxt, "prev": prev, "limit": limit, "sort": sort, "order": order,
                             "total": total, "total_exact": exact, "user": f.user_id, "team": f.team,
                             "start": f.start.isoformat() if f.start else None, "end": f.end.isoformat() if f.end else None,
                             "sort_range_days": SORT_RANGE.days, "answered_by": answered_by()})

    @router.get("/requests.csv")
    async def requests_csv(request: Request, f: RequestFilter = Depends(request_filter), sort: SORTS = "time",
                           order: Literal["desc", "asc"] = "desc"):
        """Every row the Requests view's filters match, in its order, as CSV: streamed, so memory stays flat however many
        rows match, and scoped as the view is."""
        f = await requests_scope(request, f, sort)
        rows = gateway.store.requests_rows(f, sort, order == "desc")
        first = await anext(rows, None)  # reaches the store before the headers say which one answered
        fields = [*LEDGER_FIELDS, "savings_total"]

        def line(row: dict) -> str:
            out = {k: row.get(k) for k in LEDGER_FIELDS}
            out["savings_total"] = round(savings_of(row), 6)
            out["tags"] = json.dumps(out["tags"] or {}, separators=(",", ":"), sort_keys=True)
            buf = io.StringIO()
            csv.DictWriter(buf, fieldnames=fields).writerow({k: csv_cell(v.isoformat() if isinstance(v, datetime) else v)
                                                             for k, v in out.items()})
            return buf.getvalue()

        async def lines():
            yield ",".join(fields) + "\r\n"
            if first is None:
                return
            yield line(first)
            async for row in rows:
                yield line(row)

        stamp = (f.start or datetime.now(UTC)).strftime("%Y%m%d")
        return StreamingResponse(lines(), media_type="text/csv",
                                 headers={"content-disposition": f'attachment; filename="meritorder-requests-{stamp}.csv"'})

    return router
