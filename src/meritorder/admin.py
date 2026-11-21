"""Admin API: API keys and a read-only view of the effective configuration.

* admin      -> everything
* team_lead  -> keys for their own team; can only issue `user` keys
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Literal

import asyncio
import copy
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from .auth import ROLE_RANK, AuthError, Principal, lower_role
from .configstore import ConfigError, load_active, parse_and_validate, parse_document, summarize, to_yaml
from . import learned_jobs, spend
from .evals import jobs as eval_jobs
from .evals.runner import BUILTIN as BUILTIN_SUITES
from .evals.runner import DEFAULT_SUITE
from .evals.runner import MIN_HARD
from .store import ANY_VERSION, ConfigConflict


class NewKey(BaseModel):
    user: str = Field(min_length=1, max_length=200)
    team: str | None = None
    role: Literal["user", "team_lead", "finance", "admin"] = "user"
    name: str | None = Field(default=None, max_length=200)
    expires_days: int | None = Field(default=90, ge=1, le=3650)
    tags: dict[str, str] = {}


class UserUpdate(BaseModel):
    team: str | None = Field(default=None, max_length=200)
    role: Literal["user", "team_lead", "finance", "admin"] | None = None
    disabled: bool | None = None
    follow_idp: bool | None = None  # true: the sign-in mappings (Entra or Google groups, email rules) decide team and role again
    # This person's monthly limit in USD (0 stops their requests). null clears it, so their team's per-person default applies.
    budget_usd_month: float | None = Field(default=None, ge=0, le=10_000_000)


class ConfigText(BaseModel):
    """A configuration as YAML text or as a structured document (what the console's forms send)."""

    yaml: str | None = Field(default=None, max_length=2_000_000)
    document: dict | None = None
    comment: str | None = Field(default=None, max_length=500)
    # The version the editor opened (GET /config/source's ``version``; null when it came from the file). A save whose base
    # isn't the latest version is refused with 409, so two admins can't overwrite each other. Left out: no check.
    base_version: int | None = None


class Restore(BaseModel):
    base_version: int | None = None  # as ConfigText.base_version


def base_of(body: BaseModel | None) -> object:
    """The base_version a request sent, or ANY_VERSION when it left the field out (scripts written before it existed)."""
    if body is None or "base_version" not in body.model_fields_set:
        return ANY_VERSION
    return body.base_version  # type: ignore[attr-defined]


def conflict(e: ConfigConflict) -> HTTPException:
    """409 for a save based on a version that's no longer the latest."""
    return HTTPException(409, f"someone else saved the configuration (version {e.current}) since you opened "
                              f"{'version ' + str(e.base) if e.base is not None else 'it'}; reload to see their change, "
                              "then make yours again")


def resolved_price(price) -> dict:
    """A route's prices as they're billed: the cache prices it doesn't list are shown at their defaults (Price.read,
    write, write_1h), not as blanks."""
    return {**price.model_dump(), "cache_read": price.read(), "cache_write": price.write(), "cache_write_1h": price.write_1h()}


def _public(k: dict) -> dict:
    out = {key: (v.isoformat() if isinstance(v, datetime) else v) for key, v in k.items() if key != "key_hash"}
    now = datetime.now(UTC)
    out["status"] = "revoked" if k.get("revoked_at") else ("expired" if k.get("expires_at") and k["expires_at"] < now else "active")
    return out


class EvalRunRequest(BaseModel):
    models: list[str] = Field(min_length=1, max_length=50)
    suite: str = Field(DEFAULT_SUITE, max_length=500)  # built-in names or paths, comma-separated
    repeat: int = Field(1, ge=1, le=10)
    only: list[str] | None = Field(None, max_length=100)  # task id patterns, e.g. "c5-*"
    judges: list[str] | None = Field(None, max_length=4)
    concurrency: int = Field(4, ge=1, le=16)
    executor: Literal["worker", "cli"] = "worker"
    # Unset: each model runs each level at the effort `auto` uses for it, on its first live route (recorded on the run).
    effort: dict[str, dict[str, str]] | None = None
    backends: dict[str, str] | None = None
    # The most the run may spend: no new task starts once it has. Required to start one, so nobody runs an eval
    # without having seen what it costs (the estimate gives a default, 1.5x itself).
    max_cost_usd: float | None = Field(None, gt=0, le=100_000)


class EvalRunReport(BaseModel):
    status: Literal["running", "done", "failed", "cancelled"] | None = None
    progress: dict | None = None
    cost: float | None = Field(None, ge=0)
    scores: dict | None = None
    results: list[dict] | None = None
    error: str | None = Field(None, max_length=2000)


class RouterUse(BaseModel):
    version: int | None  # the learned router model gateways use; null: none, the rules decide


class JudgedEstimate(BaseModel):
    """What judged labels would cost a team a month: its judged_labels settings as they'd be (any left out keep the team's
    current ones), priced on its recorded task starts of the last ``days``."""

    team: str = Field(min_length=1, max_length=200)
    settings: dict = {}
    days: int = Field(14, ge=1, le=90)


class LevelApply(BaseModel):
    task: Literal["coding", "general"]
    level: int = Field(ge=1, le=5)
    to: int = Field(ge=1, le=5)


class LevelDismiss(LevelApply):
    """A recommendation to set aside, as the report showed it (its target and direction), or ``undo`` to bring it back."""
    direction: Literal["lower", "raise"]
    undo: bool = False


class EvalApply(BaseModel):
    models: list[str] | None = None  # only these models' scores (default: all in the run)
    # quality: routing uses it only with quality.source: config. quality_override: wins over published scores too.
    target: Literal["quality", "quality_override"] = "quality"
    dry_run: bool = False  # only say what would change


def build_router(gateway) -> APIRouter:
    router = APIRouter(prefix="/v1/admin", tags=["admin"])

    async def principal(request: Request, minimum: str = "team_lead") -> Principal:
        try:
            p = await gateway.authn.authenticate_console(request)
        except AuthError as e:
            raise HTTPException(e.status, e.message)
        if not p.at_least(minimum) or (minimum == "team_lead" and p.role == "finance"):
            raise HTTPException(403, "not allowed for your role")
        return p

    def _require_db(what: str, why: str) -> None:
        """409 without Postgres: ``what`` needs it, ``why`` says what would be lost without it."""
        if gateway.store.kind != "postgres":
            raise HTTPException(409, f"{what} needs Postgres (set MERITORDER_DATABASE_URL); {why}")

    @router.get("/keys")
    async def list_keys(request: Request, user: str | None = None):
        p = await principal(request)
        team = None if p.role == "admin" else p.team
        keys = await gateway.store.list_keys(user_id=user, team=team)
        records = {u: await gateway.store.get_user(u) for u in {k["user_id"] for k in keys}}
        out = []
        for k in keys:
            # A key acts as the lower of its role and its person's (auth.lower_role): say so where the person outranks it.
            rec = records.get(k["user_id"]) or {}
            out.append({**_public(k), "user_role": rec.get("role"), "acts_as": lower_role(rec.get("role") or k["role"], k["role"])})
        return {"keys": out}

    @router.post("/keys", status_code=201)
    async def create_key(request: Request, body: NewKey):
        p = await principal(request)
        if "eval_run" in body.tags:  # set only on the worker's keys for eval runs, which may choose a backend
            raise HTTPException(422, "the eval_run tag is reserved for eval runs' own keys")
        _require_db("key management", "in-memory keys don't survive restarts")
        team = body.team or p.team
        if p.role != "admin":
            if team != p.team:
                raise HTTPException(403, "team leads can only create keys for their own team")
            if ROLE_RANK[body.role] > ROLE_RANK["user"]:
                raise HTTPException(403, "team leads can only create user keys")
            if body.expires_days is None:
                raise HTTPException(403, "only an admin can create a key that never expires; set expires_days (up to 3650)")
        expires = datetime.now(UTC) + timedelta(days=body.expires_days) if body.expires_days else None
        if p.role != "admin" and body.user != p.user_id:
            # A key acts as its user's record (team and role), so a lead may only mint keys for people on
            # their own team with the user role; otherwise a lead could mint a working key for an admin.
            current = await gateway.store.get_user(body.user)
            if current and (current["team"] != p.team or ROLE_RANK.get(current["role"], 0) > ROLE_RANK["user"]):
                raise HTTPException(403, f"{body.user} is on team '{current['team']}' with role '{current['role']}'; "
                                         "only an admin can create keys for them")
        await gateway.store.upsert_user(body.user, team, body.role, source="ensure")
        existing = await gateway.store.get_user(body.user)
        role = body.role
        if existing and (existing["team"] != team or existing["role"] != body.role):
            # The record decides the team; the key acts as the lower of its own role and the record's (auth.lower_role),
            # so report what the key will actually act as.
            team, role = existing["team"], lower_role(existing["role"], body.role)
            role_note = f" (existing user: team {team}, role {existing['role']}; this key acts as {role})"
        else:
            role_note = ""
        key_id, secret = await gateway.store.create_key(body.user, team, role, body.name, body.tags, expires)
        return {"id": key_id, "api_key": secret, "user": body.user, "team": team, "role": role,
                "expires_at": expires.isoformat() if expires else None,
                "note": "Shown once. Merit Order stores only a hash of it." + role_note}

    @router.delete("/keys/{key_id}")
    async def revoke_key(request: Request, key_id: str):
        p = await principal(request)
        _require_db("key management", "in-memory keys don't survive restarts")
        key = await gateway.store.get_key(key_id)
        if key is None:
            raise HTTPException(404, "no such key")
        if p.role != "admin" and key["user_id"] != p.user_id:
            # The minting rule: a lead manages the keys of people on their own team with the user role (and their own),
            # so a lead can't switch off an admin's or another lead's key.
            owner = await gateway.store.get_user(key["user_id"])
            team, role = (owner["team"], owner["role"]) if owner else (key["team"], key["role"])
            if key["team"] != p.team or team != p.team:
                raise HTTPException(403, "you can only revoke keys for your own team")
            if ROLE_RANK.get(role, 0) > ROLE_RANK["user"] or ROLE_RANK.get(key["role"], 0) > ROLE_RANK["user"]:
                raise HTTPException(403, f"{key['user_id']} has the role '{role}'; only an admin can revoke their keys")
        revoked = await gateway.store.revoke_key(key_id)
        gateway.authn.forget(key_id)
        return {"id": key_id, "revoked": revoked}

    # ------------------------------------------------------------------ people

    @router.get("/users")
    async def list_users(request: Request):
        p = await principal(request)
        team = None if p.role == "admin" else p.team
        rows = await gateway.store.list_users(team=team)
        start = datetime.now(UTC).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        spend = {r["key"]: r for r in await gateway.store.summary(start, datetime.now(UTC) + timedelta(days=1), "user", team=team)}
        groups = gateway.authn.groups
        readings = await groups.records() if groups else {}
        out = []
        for u in rows:
            s = spend.get(u["id"]) or {}
            out.append({"id": u["id"], "email": u.get("email"), "display_name": u.get("display_name"), "team": u["team"], "role": u["role"],
                        "managed": u.get("managed", "idp"), "disabled": bool(u.get("disabled_at")),
                        "budget_usd_month": u.get("budget_usd_month"), "default_budget_usd_month": _default_limit(u["id"], u["team"]),
                        "active_keys": int(u.get("active_keys") or 0),
                        "has_password": bool(u.get("has_password")), "two_factor": bool(u.get("two_factor")),
                        "last_seen": u["last_seen"].isoformat() if u.get("last_seen") else None,
                        "spend_month": s.get("actual_cost", 0.0), "requests_month": s.get("requests", 0),
                        "google": _google_view(readings.get(u["id"]), every=p.role == "admin") if groups else None})
        import os as _os

        oidc = gateway.cfg.auth.oidc
        return {"users": out, "teams": [t.name for t in gateway.cfg.teams], "oidc": bool(oidc), "idp": oidc.provider if oidc else None,
                "scim": bool(_os.environ.get("MERITORDER_SCIM_TOKEN")), "google_groups": bool(groups),
                "google_groups_refresh_seconds": groups.cfg.refresh_seconds if groups else None}

    def _google_view(rec: dict | None, every: bool) -> dict | None:
        """A person's last Google groups reading for the People page. Admins see every group; a team lead sees only the
        groups the mappings use, since group names can say more about someone than their team does."""
        if not rec:
            return None
        oidc = gateway.cfg.auth.oidc
        mapped = {g.lower() for g in [*oidc.team_from_groups, *oidc.role_from_groups]}
        listed = [{**g, "mapped": g["email"] in mapped} for g in rec.get("groups") or []]
        shown = listed if every else [g for g in listed if g["mapped"]]

        def iso(ts):
            return datetime.fromtimestamp(float(ts), UTC).isoformat() if ts else None

        return {"groups": shown, "hidden": len(listed) - len(shown), "using": rec.get("using"), "error": rec.get("error"),
                "fetched_at": iso(rec.get("fetched_at")), "checked_at": iso(rec.get("checked_at")),
                "failing_since": iso(rec.get("failing_since")), "team": rec.get("team"), "role": rec.get("role"),
                "team_from": rec.get("team_from"), "role_from": rec.get("role_from")}

    def _default_limit(user_id: str, team: str) -> float | None:
        """What applies when a person has no limit of their own: their static key's limit in the config file, if they
        have one, else their team's per-person default."""
        static = next((k.budget_usd_month for k in gateway.cfg.auth.static_keys if k.user == user_id and k.budget_usd_month is not None), None)
        return static if static is not None else gateway.cfg.team(team).user_budget_usd_month

    @router.patch("/users/{user_id}")
    async def update_user(request: Request, user_id: str, body: UserUpdate):
        p = await principal(request, "admin")
        _require_db("changing a person's team, role or access", "in-memory changes don't survive restarts")
        if user_id == p.user_id and (body.disabled or (body.role and body.role != "admin")):
            raise HTTPException(400, "you can't disable yourself or remove your own admin role")
        if body.team is not None and body.team not in {t.name for t in gateway.cfg.teams}:
            raise HTTPException(400, f"team '{body.team}' isn't defined; add it under Config → Teams first")
        limit = {"budget_usd_month": body.budget_usd_month} if "budget_usd_month" in body.model_fields_set else {}
        user = await gateway.store.update_user(user_id, team=body.team, role=body.role, disabled=body.disabled,
                                               managed="idp" if body.follow_idp else None, **limit)
        if user is None:
            raise HTTPException(404, "no such user")
        if body.team is not None or body.role is not None or body.disabled:
            await gateway.store.revoke_user_invites(user_id)  # an open password link was made for who they were
        if body.follow_idp and gateway.authn.groups and await gateway.authn.groups.refresh(user_id):
            user = await gateway.store.get_user(user_id) or user  # Google groups read now: the mappings apply at once
        gateway.authn.forget_user(user_id)
        # Removing someone deletes what they stored in the Responses API's stored mode (their ledger rows stay).
        deleted = await gateway.stored.forget_user(user_id) if body.disabled else 0
        return {"id": user_id, "team": user["team"], "role": user["role"], "managed": user.get("managed"),
                "disabled": bool(user.get("disabled_at")), "budget_usd_month": user.get("budget_usd_month"),
                "default_budget_usd_month": _default_limit(user_id, user["team"]), "stored_responses_deleted": deleted}

    # ------------------------------------------------------------------ evals

    @router.get("/evals")
    async def get_evals(request: Request):
        await principal(request)
        return {"models": await eval_jobs.latest_view(gateway.store)}

    def models_view() -> dict[str, dict]:
        """Each model with its routes, whether they're live, and their prices (what estimates and the CLI use)."""
        formats = {b.name: b.format for b in gateway.cfg.backends}
        return {m.name: {"name": m.name, "kind": m.kind, "quality": m.quality, "quality_sources": m.quality_sources,
                         "chances": m.chances, "context_window": m.context_window, "hf_id": m.hf_id,
                         "max_output_tokens": m.max_output_tokens, "effort_levels": m.effort_levels,
                         "routes": [{"backend": r.backend, "model": r.model, "format": formats.get(r.backend),
                                     "live": r.backend in gateway.backends and gateway.backends[r.backend].available,
                                     "price": resolved_price(r.price) if r.price else None} for r in m.routes]}
                for m in gateway.cfg.models}

    def _run_public(row: dict, results: bool = False) -> dict:
        out = {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in row.items() if results or k != "results"}
        return out

    async def _plan(body: EvalRunRequest) -> tuple[dict, dict, list[str]]:
        view = models_view()
        unknown = [m for m in body.models if m not in view]
        if unknown:
            raise HTTPException(404, f"no such model: {', '.join(unknown)}")
        dead = [m for m in body.models if not any(r["live"] for r in view[m]["routes"])]
        if dead:
            raise HTTPException(409, f"no live route for {', '.join(dead)}; see the Backends page")
        try:
            suite = await asyncio.to_thread(eval_jobs.plan, body.model_dump())
        except Exception as e:  # noqa: BLE001 (a bad suite name or path is the caller's mistake)
            raise HTTPException(422, f"couldn't load the suite: {e}")
        if not suite["tasks"]:
            raise HTTPException(422, "no tasks match that filter")
        return view, suite, eval_jobs.pick_judges(view, body.judges)

    @router.get("/evals/suites")
    async def eval_suites(request: Request):
        await principal(request, "admin")
        view = models_view()
        return {"suites": await asyncio.to_thread(eval_jobs.suites), "runner": await gateway.store.get_setting("eval_runner"),
                "judges": eval_jobs.pick_judges(view), "default_judges": list(eval_jobs.DEFAULT_JUDGES)}

    @router.get("/evals/tasks")
    async def eval_tasks(request: Request, suite: str = "starter"):
        """A built-in suite's tasks, for the console's Tasks tab."""
        await principal(request, "admin")
        if suite not in BUILTIN_SUITES:
            raise HTTPException(404, f"no built-in suite '{suite}'")
        return {"suite": suite, "tasks": await asyncio.to_thread(eval_jobs.task_list, suite)}

    @router.get("/evals/tasks/{task_id}")
    async def eval_task(request: Request, task_id: str, suite: str = "starter"):
        """One task in full, with each model's latest results on it."""
        await principal(request, "admin")
        if suite not in BUILTIN_SUITES:
            raise HTTPException(404, f"no built-in suite '{suite}'")
        detail = await asyncio.to_thread(eval_jobs.task_detail, suite, task_id)
        if detail is None:
            raise HTTPException(404, f"no task '{task_id}' in {suite}")
        results = []
        for model, (_, pairs) in (await eval_jobs.latest_results(gateway.store)).items():
            # The results that count for this task: from the newest run that graded it.
            results += [{**r, "run_id": rid} for rid, r in pairs or [] if r.get("task_id") == task_id]
        finished = {}
        for rid in {r["run_id"] for r in results}:
            row = await gateway.store.get_eval_run(rid)
            finished[rid] = (row or {}).get("finished_at")
        results = [{**r, "ran_at": finished[r["run_id"]].isoformat(timespec="seconds") if isinstance(finished[r["run_id"]], datetime)
                    else finished[r["run_id"]]} for r in results]
        return {**detail, "results": sorted(results, key=lambda r: r["model"])}

    async def _budget_left() -> float | None:
        """What's left of the organization's monthly budget, or None without one."""
        cap = gateway.cfg.safeguards.org_budget_usd_month
        if cap is None:
            return None
        return round(cap - (await spend.budget_spend(gateway.store, gateway.cfg, "org"))["spent"], 2)

    @router.post("/evals/estimate")
    async def eval_estimate(request: Request, body: EvalRunRequest):
        await principal(request, "admin")
        view, suite, judges = await _plan(body)
        est = eval_jobs.estimate(view, suite, body.models, body.repeat, judges)
        cap = body.max_cost_usd or eval_jobs.default_cap(est["usd"])
        left = await _budget_left()
        return {**est, "judges": judges, "runner": await gateway.store.get_setting("eval_runner"),
                "default_max_cost_usd": eval_jobs.default_cap(est["usd"]), "budget_left_usd": left,
                "confirm_above_usd": eval_jobs.CONFIRM_ABOVE_USD, "warning": eval_jobs.cost_warning(est, cap, left)}

    @router.post("/evals/runs", status_code=201)
    async def eval_start(request: Request, body: EvalRunRequest):
        """Queue a run for the worker, or (executor=cli) record one that `meritorder eval` runs itself."""
        p = await principal(request, "admin")
        view, suite, judges = await _plan(body)
        est = eval_jobs.estimate(view, suite, body.models, body.repeat, judges)
        if body.max_cost_usd is None:
            raise HTTPException(422, f"this run is estimated at about ${est['usd']:,.2f} ({eval_jobs.ROUGH}); set max_cost_usd, "
                                     f"the most it may spend (for example ${eval_jobs.default_cap(est['usd']):,.0f}), to start it")
        req = {**body.model_dump(exclude={"executor"}), "judges": judges, "estimate_usd": est["usd"], "tasks": est["tasks"],
               "effort": body.effort or eval_jobs.efforts(view, body.models, gateway.cfg.routing.effort_by_floor),
               "backends": body.backends or eval_jobs.routes(view, sorted(set(body.models) | set(judges)))}
        req["formats"] = eval_jobs.formats(view, req["backends"])
        bad = {m: b for m, b in req["backends"].items() if not any(r["backend"] == b and r["live"] for r in view.get(m, {}).get("routes", []))}
        if bad:
            raise HTTPException(409, "no live route " + ", ".join(f"{m} on {b}" for m, b in bad.items()))
        cli = body.executor == "cli"
        row = await gateway.store.create_eval_run({"id": eval_jobs.new_run_id(), "created_by": p.user_id, "team": p.team,
                                                   "status": "running" if cli else "queued", "executor": body.executor, "request": req,
                                                   "started_at": datetime.now(UTC) if cli else None})
        return _run_public(row)

    @router.get("/evals/runs")
    async def eval_runs(request: Request):
        await principal(request)
        return {"runs": [_run_public(r) for r in await gateway.store.list_eval_runs(50)],
                "runner": await gateway.store.get_setting("eval_runner")}

    @router.get("/evals/runs/{run_id}")
    async def eval_run(request: Request, run_id: str):
        await principal(request)
        row = await gateway.store.get_eval_run(run_id)
        if not row:
            raise HTTPException(404, "no such eval run")
        out = _run_public(row, results=True)
        if row["status"] == "done" and row.get("results"):
            # On today's scale, scored together with every other model's latest result: what applying it would write.
            view = await eval_jobs.latest_view(gateway.store, row)
            out["scores"] = {m: view[m] for m in row["request"].get("models") or [] if m in view}
        return out

    @router.patch("/evals/runs/{run_id}")
    async def eval_report(request: Request, run_id: str, body: EvalRunReport):
        """Progress and results from `meritorder eval` for a run it's running itself. Returns the run's status, so
        the CLI sees a cancel made in the console."""
        await principal(request, "admin")
        row = await gateway.store.get_eval_run(run_id)
        if not row:
            raise HTTPException(404, "no such eval run")
        if row["executor"] != "cli" or row["status"] in eval_jobs.FINISHED:
            raise HTTPException(409, f"this run is {row['status']} and isn't reported by the CLI")
        fields = body.model_dump(exclude_none=True)
        if body.status in eval_jobs.FINISHED:
            fields["finished_at"] = datetime.now(UTC)
        row = await gateway.store.update_eval_run(run_id, **fields)
        if body.status == "done" and body.scores:
            latest = eval_jobs.merge_latest(await gateway.store.get_setting("evals"), run_id, row["request"].get("suite") or "", body.scores)
            await gateway.store.set_setting("evals", latest)
        return {"id": run_id, "status": row["status"]}

    @router.post("/evals/runs/{run_id}/cancel")
    async def eval_cancel(request: Request, run_id: str):
        await principal(request, "admin")
        row = await gateway.store.get_eval_run(run_id)
        if not row:
            raise HTTPException(404, "no such eval run")
        if row["status"] in eval_jobs.FINISHED:
            raise HTTPException(409, f"this run is already {row['status']}")
        if row["status"] == "queued":
            row = await gateway.store.update_eval_run(run_id, status="cancelled", finished_at=datetime.now(UTC))
        else:
            row = await gateway.store.update_eval_run(run_id, status="cancelling")  # finishes the tasks under way, then stops
        return _run_public(row)

    @router.post("/evals/runs/{run_id}/apply")
    async def eval_apply(request: Request, run_id: str, body: EvalApply):
        """Write a finished run's measured quality scores into the configuration, as a new version: as each model's
        `quality` (used with quality.source: config) or `quality_override` (wins over published scores too). They're
        scored together with every other model's latest result, so they're on the same scale."""
        p = await principal(request, "admin")
        if not body.dry_run:
            _require_db("applying scores", "they're saved as a new config version, and versions are kept there")
        row = await gateway.store.get_eval_run(run_id)
        if not row:
            raise HTTPException(404, "no such eval run")
        if row["status"] != "done" or not (row.get("results") or row.get("scores")):
            raise HTTPException(409, "only a finished run's scores can be applied")
        view = await eval_jobs.latest_view(gateway.store, row)
        in_use = {m.name: m.quality for m in gateway.cfg.models}
        raw, _, base = await _current_raw()
        doc = copy.deepcopy(raw)
        changed, changes, tasks = [], [], set()
        for model in row["request"].get("models") or []:
            if model not in view or (body.models and model not in body.models):
                continue
            for task, v in (view[model].get("tasks") or {}).items():
                if v.get("score") is None:
                    continue
                tasks.add(task)
                now = (((doc.get("models") or {}).get(model) or {}).get(body.target) or {}).get(task)
                if now != v["score"]:
                    changes.append({"model": model, "task": task, "from": now, "to": v["score"], "low": v["low"], "high": v["high"],
                                    "in_use": in_use.get(model, {}).get(task)})
                    changed.append(f"{model} {task} {now if now is not None else '–'}→{v['score']}")
                    doc.setdefault("models", {}).setdefault(model, {}).setdefault(body.target, {})[task] = v["score"]
        if body.dry_run or not changed:
            return {"version": None, "changed": changed, "changes": changes, "target": body.target, "source": gateway.cfg.quality.source}
        if body.target == "quality":
            # What a measured 5 means (the best model's pass rate on hard tasks), so routing by cost per success can turn
            # these scores into chances of success (config.apply_chances).
            for task in sorted(tasks):
                rates = [t["hard"]["passed"] / t["hard"]["runs"] for s in view.values()
                         if (t := (s.get("tasks") or {}).get(task)) and (t.get("hard") or {}).get("runs", 0) >= MIN_HARD]
                if rates and max(rates) > 0:
                    doc.setdefault("quality", {}).setdefault("measured_best_rate", {})[task] = round(max(rates), 4)
        prices = await gateway.store.get_setting("prices")
        try:
            parse_and_validate(None, gateway.file_cfg.config_dir, prices, document=doc)
        except ConfigError as e:
            raise HTTPException(422, str(e))
        try:
            version = await gateway.store.save_config(doc, p.user_id, f"{body.target} from eval run {run_id}: {'; '.join(changed)}"[:500],
                                                      base_version=base)
        except ConfigConflict as e:
            raise conflict(e)
        await load_active(gateway, publish=True)
        await gateway.store.update_eval_run(run_id, applied_version=version)
        return {"version": version, "changed": changed, "changes": changes, "target": body.target, "source": gateway.cfg.quality.source}

    @router.get("/router")
    async def router_status(request: Request):
        """The learned router: its settings, the model this gateway uses and whether it's stale, its versions, and the
        last training."""
        await principal(request, "admin")
        return await learned_jobs.status(gateway.cfg, gateway.store, gateway.learned, gateway.state)

    @router.post("/router/train", status_code=202)
    async def router_train(request: Request):
        """Ask the worker to train a model now (training never runs in a gateway: it would slow requests down)."""
        p = await principal(request, "admin")
        await gateway.state.set_json(learned_jobs.TRAIN_REQUESTED, {"by": p.user_id, "at": datetime.now(UTC).isoformat(timespec="seconds")})
        return {"requested": True}

    @router.post("/router/use")
    async def router_use(request: Request, body: RouterUse):
        """Choose the model version gateways use, e.g. to go back to an earlier one; null: none, so the rules decide."""
        await principal(request, "admin")
        if not await gateway.store.activate_router_model(body.version):
            raise HTTPException(404, f"there's no learned router model v{body.version}")
        await load_active(gateway, publish=True)
        return await learned_jobs.status(gateway.cfg, gateway.store, gateway.learned, gateway.state)

    @router.post("/judged-labels/estimate")
    async def judged_estimate(request: Request, body: JudgedEstimate):
        """The expected monthly cost of a team's judged labels before they're turned on, with what to say about it (it can
        run up to three times higher, and sampled prompts go to the stronger model and the judge)."""
        from .config import JudgedLabelsConfig
        from .judged import estimate

        await principal(request, "admin")
        team = gateway.cfg.team(body.team)
        try:
            s = JudgedLabelsConfig(**{**team.judged_labels.model_dump(), **body.settings, "enabled": False})
        except ValueError as e:
            raise HTTPException(422, f"judged_labels: {e}")
        now = datetime.now(UTC)
        # A team without settings of its own uses the default team's (``team``), but its traffic is its own.
        rows = await gateway.store.judged_traffic(body.team, now - timedelta(days=body.days))
        return {"team": body.team, "learned_mode": gateway.cfg.routing.learned.mode,
                **estimate(gateway.cfg, gateway.backends, team, s, rows, body.days, await _budget_left())}

    @router.put("/evals")
    async def put_evals(request: Request):
        """Store a `meritorder eval` run (latest result per model is kept)."""
        await principal(request, "admin")
        body = await request.json()
        current = await gateway.store.get_setting("evals") or {"models": {}}
        for model, result in (body.get("models") or {}).items():
            current["models"][model] = {**result, "suite": body.get("suite"), "ran_at": body.get("ran_at")}
        await gateway.store.set_setting("evals", current)
        return current

    @router.get("/config")
    async def config(request: Request):
        await principal(request, "admin")
        cfg = gateway.cfg
        backends = [{"name": b.name, "kind": b.kind, "format": b.format, "enabled": b.enabled,
                     "live": b.name in gateway.backends and gateway.backends[b.name].available,
                     "base_url": b.base_url or None} for b in cfg.backends]
        models = list(models_view().values())
        teams = [t.model_dump() for t in cfg.teams]
        from .config import load_catalog

        catalog = await gateway.store.get_setting("prices") or load_catalog(Path(cfg.config_dir) / cfg.prices_file)
        q = catalog.get("quality") or {}
        for m in models:
            m["quality_detail"] = (q.get("measures") or {}).get(m["name"], {})
            m["tokens"] = gateway.tokens.view(m["name"])  # how its prompt counts are calibrated (tokens.py)
        quality = {"source": cfg.quality.source, "scale": cfg.quality.scale, "reference": cfg.quality.reference,
                   "anchor": dict(cfg.quality.anchor) if cfg.quality.scale == "absolute" else None, "evidence": cfg.quality.evidence,
                   "refresh_hours": cfg.quality.refresh_hours, "updated": q.get("updated"), "status": q.get("status"),
                   "attribution": q.get("attribution"), "missing": q.get("missing", []),
                   # A new scale or anchor is worked out at once from the stored chances (config.live_section); a new source or
                   # reference waits for the worker.
                   "stale": bool(q) and (q.get("source"), q.get("reference")) != (cfg.quality.source, cfg.quality.reference)}
        return {"baseline_model": cfg.baseline_model, "backends": backends, "models": models, "teams": teams,
                "oidc": bool(cfg.auth.oidc), "effort_by_floor": cfg.routing.effort_by_floor,
                "prices": {**(await gateway.store.get_setting("prices_status") or {}), "refresh_minutes": cfg.prices_refresh_minutes},
                "quality": quality}

    @router.get("/compressor")
    async def compressor(request: Request):
        """Whether model-based compression's service answers, with which model, and which teams use it."""
        await principal(request, "admin")
        teams = [t.name for t in gateway.cfg.teams if t.compression.model.enabled]
        if gateway.compressor is None:
            return {"configured": False, "teams": teams}
        return {"configured": True, "teams": teams, **await gateway.compressor.status()}

    @router.get("/reporting")
    async def reporting(request: Request):
        """Where the ledger and reports live: Postgres, and with `reporting.clickhouse` a copy in ClickHouse that large
        reports read; how each is doing, and how far the copy has got (Config → Reporting)."""
        await principal(request, "admin")
        store = gateway.store
        try:
            pg_ok, pg_error = bool(await store.ping()), None
        except Exception as e:  # noqa: BLE001 (shown on the page)
            pg_ok, pg_error = False, str(e)[:300]
        rep = getattr(store, "clickhouse", None)
        configured = gateway.cfg.reporting.clickhouse is not None
        note = None
        if configured and rep is None:
            note = "reporting.clickhouse is set, but the ledger is in memory: ClickHouse needs Postgres (MERITORDER_DATABASE_URL)"
        return {"store": store.kind, "postgres": {"ok": pg_ok, "error": pg_error}, "configured": configured, "note": note,
                "clickhouse": await rep.describe() if rep is not None else None}

    # ------------------------------------------------------------------ configuration

    async def _current_raw() -> tuple[dict, str, int | None]:
        row = await gateway.store.active_config()
        if row:
            return row["body"], "database", row["id"]
        if gateway.file_raw is None:
            raise HTTPException(409, "the running configuration didn't come from a file or the database")
        return gateway.file_raw, "file", None

    @router.get("/config/source")
    async def config_source(request: Request):
        await principal(request, "admin")
        raw, source, version = await _current_raw()
        return {"source": source, "version": version, "yaml": to_yaml(raw), "document": raw, "summary": summarize(gateway.cfg),
                "storage": gateway.store.kind}

    @router.post("/config/validate")
    async def config_validate(request: Request, body: ConfigText):
        await principal(request, "admin")
        prices = await gateway.store.get_setting("prices")
        try:
            doc = parse_document(body.yaml, body.document)
        except ConfigError as e:
            return {"ok": False, "error": str(e)}
        rendered = {"document": doc, "yaml": to_yaml(doc)}
        try:
            _, cfg = parse_and_validate(None, gateway.file_cfg.config_dir, prices, document=doc)
        except ConfigError as e:
            return {"ok": False, "error": str(e), **rendered}
        return {"ok": True, "summary": summarize(cfg), **rendered}

    @router.put("/config")
    async def config_save(request: Request, body: ConfigText):
        p = await principal(request, "admin")
        _require_db("saving the config", "config versions are kept there; without it the gateway runs on its YAML file")
        prices = await gateway.store.get_setting("prices")
        try:
            raw, cfg = parse_and_validate(body.yaml, gateway.file_cfg.config_dir, prices, document=body.document)
        except ConfigError as e:
            raise HTTPException(422, str(e))
        try:
            version = await gateway.store.save_config(raw, p.user_id, body.comment, base_version=base_of(body))
        except ConfigConflict as e:
            raise conflict(e)
        await load_active(gateway, publish=True)
        return {"version": version, "summary": summarize(cfg)}

    @router.post("/levels/apply")
    async def levels_apply(request: Request, body: LevelApply):
        """Route the work at one task type and level at another (``routing.level_map``), as the outcome report recommends
        (decision 59): saved as a new configuration version. ``to`` equal to ``level`` takes a move back."""
        from . import benchmark

        p = await principal(request, "admin")
        _require_db("applying a level recommendation", "it's saved as a new config version, and versions are kept there")
        raw, _, base = await _current_raw()
        prices = await gateway.store.get_setting("prices")
        try:
            saved = await benchmark.apply_and_save(gateway.store, raw, gateway.file_cfg.config_dir, prices,
                                                   {"task": body.task, "level": body.level, "to": body.to}, p.user_id, base)
        except ConfigError as e:
            raise HTTPException(422, str(e))
        except ConfigConflict as e:
            raise conflict(e)
        if saved["version"]:
            await load_active(gateway, publish=True)
        return {"version": saved["version"], "changed": saved["changed"], "level_map": dict(gateway.cfg.routing.level_map)}

    @router.post("/levels/dismiss")
    async def levels_dismiss(request: Request, body: LevelDismiss):
        """Set a level recommendation aside (Reports → Levels, Dismiss). It stays dismissed, and the worker doesn't alert
        on it or apply it, until its evidence changes: it points another way, or routing.benchmark.min_outcomes more
        task starts are explored on the level it compares with (benchmark.dismissal)."""
        from . import benchmark

        p = await principal(request, "admin")
        doc = dict(await gateway.store.get_setting(benchmark.DISMISSED) or {})
        key = f"{body.task}:{body.level}"
        if body.undo:
            doc.pop(key, None)
        else:
            doc[key] = {"direction": body.direction, "to": body.to, "at": datetime.now(UTC).isoformat(timespec="seconds"),
                        "by": p.user_id}
        await gateway.store.set_setting(benchmark.DISMISSED, doc)
        return {"dismissed": doc}

    @router.get("/config/versions")
    async def config_versions(request: Request):
        await principal(request, "admin")
        rows = await gateway.store.config_versions()
        return {"active": gateway.config_version,
                "versions": [{**r, "created_at": r["created_at"].isoformat()} for r in rows]}

    @router.get("/config/versions/{vid}")
    async def config_version(request: Request, vid: int):
        await principal(request, "admin")
        row = await gateway.store.get_config(vid)
        if not row:
            raise HTTPException(404, "no such version")
        return {"id": row["id"], "author": row["author"], "comment": row["comment"],
                "created_at": row["created_at"].isoformat(), "yaml": to_yaml(row["body"])}

    @router.post("/config/versions/{vid}/restore")
    async def config_restore(request: Request, vid: int, body: Restore | None = None):
        p = await principal(request, "admin")
        _require_db("restoring a config version", "config versions are kept there")
        row = await gateway.store.get_config(vid)
        if not row:
            raise HTTPException(404, "no such version")
        prices = await gateway.store.get_setting("prices")
        try:
            # The rules a save follows (secret sources, ${NAME} references) apply to a restore too: the version may
            # predate them.
            parse_and_validate(None, gateway.file_cfg.config_dir, prices, document=row["body"])
        except ConfigError as e:
            raise HTTPException(422, f"version {vid} no longer validates: {e}")
        try:
            version = await gateway.store.save_config(row["body"], p.user_id, f"restored version {vid}", base_version=base_of(body))
        except ConfigConflict as e:
            raise conflict(e)
        await load_active(gateway, publish=True)
        return {"version": version, "restored_from": vid}

    @router.post("/quality/refresh")
    async def quality_refresh(request: Request):
        await principal(request, "admin")
        _require_db("refreshing quality scores", "the fetched scores are kept there for every gateway to load")
        from .quality_data import refresh_quality, refresh_quality_once

        result = await refresh_quality_once(gateway.cfg, gateway.store, gateway.state, refresh_quality)
        await load_active(gateway, publish=True)
        return result

    @router.post("/prices/refresh")
    async def prices_refresh(request: Request):
        p = await principal(request, "admin")
        _require_db("refreshing prices", "the fetched prices are kept there for every gateway to load")

        from .catalog import refresh_prices
        from .live_prices import refresh_once

        status = await refresh_once(gateway.cfg, gateway.store, gateway.state, refresh_prices)
        await load_active(gateway, publish=True)
        return {"priced_routes": status["priced_routes"], "missing": status["missing"], "updated": status["checked_at"], "by": p.user_id,
                "changed": status["changed"], "sources": status["sources"], "disagreements": status["disagreements"]}

    return router
