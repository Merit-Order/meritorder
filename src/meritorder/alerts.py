"""Alerts and weekly digests sent to a Slack/Teams/Discord-style incoming webhook.

* Budget thresholds: 80% and 100% of a team's or a person's monthly budget (once per month each), counted as admission
  counts it (spend.budget_spend: the ledger, running batches' reservations, and for the organization GPU time). A person's
  budget is their own limit if one is set (People page, or their static key), else their team's per-person default.
  A team's judged labels (judged.py) get the same two alerts against their own monthly cap.
* Spend spikes: a user's last hour is 5x their usual hourly spend (usually an agent in a loop).
* Weekly digest per team: spend, baseline, and savings by cause (once per ISO week); with GPUs, one for the organization
  too, with the GPU time no team's requests paid for (spend.py).
* Routes with no known price (critical, once a day each): requests for them are refused until they have one.
* Provider outages: a backend (or one model on it) being skipped after failures, and its recovery. At most
  one outage alert per backend and model every 30 minutes, and a recovery only after an outage alert.
GPU state changes are sent by the capacity controller.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime, timedelta

import httpx

from . import spend
from .config import Config
from .judged import internal_user, is_internal
from .state import claim
from .store import SAVINGS_CAUSES, month_start

log = logging.getLogger("meritorder.alerts")

THRESHOLDS = (0.8, 1.0)
SPIKE_FACTOR = 5.0
SPIKE_MIN_USD = 5.0


async def notify(text: str, url: str | None = None, state=None, title: str | None = None, level: str = "warning",
                 page: str | None = None, team: str | None = None) -> bool:
    """Log, show in the console's "needs attention" list, and send to the webhook if one is set."""
    url = url or os.environ.get("MERITORDER_ALERT_WEBHOOK_URL")
    log.warning("ALERT: %s", text)
    if state is not None:
        await state.push_alert({"ts": datetime.now(UTC).isoformat(), "level": level, "title": title or text[:80],
                                "text": text, "page": page, "team": team})
    if not url:
        return False
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.post(url, json={"text": text, "content": text})  # Slack/Teams use text, Discord content
        return r.status_code < 300
    except httpx.HTTPError as e:
        log.error("webhook failed: %s", e)
        return False


OUTAGE_ALERT_EVERY = 1800


async def outage(state, backend: str, model: str | None, why: str, seconds: float, alternatives: list[str]) -> bool:
    """A backend, or a model on it, is being skipped. Says where traffic goes meanwhile."""
    what = f"{backend} ({model})" if model else backend
    if not await _once(state, f"alert:outage:{backend}:{model or '*'}", OUTAGE_ALERT_EVERY):
        return False
    await state.set_json(f"outage:open:{backend}:{model or '*'}", True, 86400)
    where = f"Requests go to {', '.join(alternatives)} meanwhile." if alternatives else "No other live backend serves it."
    await notify(f"{what} is failing: {why}. Skipping it for {seconds:.0f}s, then trying again. {where}", state=state,
                 title=f"{what} is failing", level="warning" if model else "critical", page="backends")
    return True


async def recovered(state, backend: str, model: str | None) -> bool:
    key = f"outage:open:{backend}:{model or '*'}"
    if not await state.get_json(key):
        return False
    await state.set_json(key, False, 60)
    what = f"{backend} ({model})" if model else backend
    await notify(f"{what} is working again.", state=state, title=f"{what} recovered", level="info", page="backends")
    return True


async def _once(state, key: str, ttl: int) -> bool:
    """True the first time a key is seen (dedupes alerts), for one caller only: two workers checking at once send one."""
    return await claim(state, key, ttl)


UNPRICED_ALERT_EVERY = 86400


async def check_unpriced(cfg: Config, state) -> list[str]:
    """Routes with no known price, on backends that charge per token: requests naming their model are refused (an unknown
    price is never served as free) and ``auto`` passes them by. Critical, once a day per route, until a price is set or the
    price list has one."""
    from .config import UNPRICED, needs_price

    sent = []
    for m in cfg.models:
        for r in m.routes:
            b = cfg.backend(r.backend)
            if b is None or b.disabled_reason or not needs_price(r, b.kind):
                continue
            if await _once(state, f"alert:unpriced:{m.name}@{r.backend}:{r.model}", UNPRICED_ALERT_EVERY):
                text = (f"{m.name} on {r.backend} ({r.model}) has {UNPRICED}. Requests for it are refused, and auto routing "
                        "passes it by, until it has one.")
                await notify(text, state=state, title=f"No price for {m.name} on {r.backend}", level="critical", page="models")
                sent.append(f"{m.name}@{r.backend}")
    return sent


async def personal_limits(cfg: Config, store) -> dict[str, tuple[float, str]]:
    """People with a monthly limit of their own: user id -> (limit, team). A limit on the person's record wins over
    one on their static key in the config file, as it does when requests are admitted."""
    out = {k.user: (k.budget_usd_month, k.team) for k in cfg.auth.static_keys if k.budget_usd_month is not None}
    for u in await store.list_users():
        if u.get("budget_usd_month") is not None:
            out[u["id"]] = (u["budget_usd_month"], u["team"])
    return out


async def check_budgets(cfg: Config, store, state, now: datetime | None = None) -> list[str]:
    now = now or datetime.now(UTC)
    start = month_start(now)
    month = now.strftime("%Y-%m")
    sent: list[str] = []

    async def person(user: str, team: str, spent: float, limit: float) -> None:
        for t in THRESHOLDS:
            if spent >= limit * t and await _once(state, f"alert:budget:{month}:user:{user}:{t}", 40 * 86400):
                msg = f"{user} ({team}) has used {spent / limit:.0%} of their ${limit:,.2f} monthly budget."
                await notify(msg, state=state, title=f"{user} at {t:.0%} of personal budget", page="people", team=team)
                sent.append(msg)

    # Team and personal budgets as admission counts them (spend.budget_spend): the ledger and what running batches have
    # reserved, and a person's spend across every team they've been on this month.
    async def counted(kind: str, ident: str) -> float:
        return (await spend.budget_spend(store, cfg, kind, ident, now=now))["spent"]

    own = await personal_limits(cfg, store)
    for team in cfg.teams:
        if team.budget_usd_month:
            spent = await counted("team", team.name)
            for t in THRESHOLDS:
                if spent >= team.budget_usd_month * t and await _once(state, f"alert:budget:{month}:team:{team.name}:{t}", 40 * 86400):
                    msg = f"Team '{team.name}' has used {spent / team.budget_usd_month:.0%} of its ${team.budget_usd_month:,.0f} monthly LLM budget (${spent:,.2f})."
                    await notify(msg, state=state, title=f"Team {team.name} at {t:.0%} of budget", page="teams", team=team.name,
                                 level="critical" if t >= 1 else "warning")
                    sent.append(msg)
        if team.user_budget_usd_month:
            for r in await store.summary(start, now + timedelta(days=1), "user", team=team.name):
                if r["key"] not in own and not is_internal(r["key"]):  # judged labels have their own cap, below
                    await person(r["key"], team.name, await counted("user", r["key"]), team.user_budget_usd_month)
    # Judged labels' monthly caps: teams that turned them on, and teams without settings of their own that use the default
    # team's (each has a cap of its own). Their internal client's spend is the cap's.
    by_user = {r["key"]: r["actual_cost"] for r in await store.summary(start, now + timedelta(days=1), "user")}
    judged = {t.name for t in cfg.teams if t.judged_labels.enabled} | {u.split(":", 1)[1] for u in by_user if is_internal(u)}
    for name in sorted(judged):
        cap = cfg.team(name).judged_labels.max_usd_month
        if not cap:
            continue
        spent = await store.spend_since("user_id", internal_user(name), start)
        for t in THRESHOLDS:
            if spent >= cap * t and await _once(state, f"alert:budget:{month}:judged:{name}:{t}", 40 * 86400):
                msg = (f"Team '{name}' has spent {spent / cap:.0%} of its ${cap:,.2f} monthly cap on judged labels (${spent:,.2f}). "
                       + ("No new sample starts until next month, or until the cap is raised (Config → Routing → Judged labels)."
                          if t >= 1 else "At 100%, sampling stops for the rest of the month."))
                await notify(msg, state=state, title=f"Judged labels for {name} at {t:.0%} of their cap", page="reports", team=name)
                sent.append(msg)
    if any(limit for limit, _ in own.values()):
        for user, (limit, team) in own.items():
            if limit:  # a limit of 0 stops their requests outright; there's nothing to warn about
                await person(user, team, await counted("user", user), limit)
    g = cfg.safeguards
    for label, budget, consequence in (("Organization", g.org_budget_usd_month, "requests are refused"),
                                       ("GPU", g.gpu_budget_usd_month, "GPUs drain and none start")):
        if not budget:
            continue
        spent = ((await spend.budget_spend(store, cfg, "org", now=now))["spent"] if label == "Organization"
                 else sum((await store.gpu_spend_since(start)).values()))
        for t in g.alert_at:
            if spent >= budget * t and await _once(state, f"alert:budget:{month}:{label}:{t}", 40 * 86400):
                msg = (f"{label} spend is at {spent / budget:.0%} of its ${budget:,.2f} monthly budget (${spent:,.2f})."
                       + (f" At 100%, {consequence}." if t < 1 else f" {consequence[0].upper()}{consequence[1:]} until the budget is raised or the month ends."))
                await notify(msg, state=state, title=f"{label} spend at {t:.0%} of budget", page="controls",
                             level="critical" if t >= 1 else "warning")
                sent.append(msg)
    return sent


async def check_spikes(store, state, now: datetime | None = None) -> list[str]:
    now = now or datetime.now(UTC)
    last_hour = {r["key"]: r["actual_cost"] for r in await store.summary(now - timedelta(hours=1), now, "user")}
    week = {r["key"]: r["actual_cost"] for r in await store.summary(now - timedelta(days=7, hours=1), now - timedelta(hours=1), "user")}
    sent: list[str] = []
    for user, spent in last_hour.items():
        if is_internal(user):
            continue  # judged labels' replays: bounded by their own cap, and no agent of anyone's
        usual = week.get(user, 0.0) / (7 * 24)
        if spent >= SPIKE_MIN_USD and spent >= SPIKE_FACTOR * max(usual, SPIKE_MIN_USD / SPIKE_FACTOR):
            if await _once(state, f"alert:spike:{user}:{now:%Y%m%d%H}", 2 * 3600):
                msg = (f"Spend spike: {user} spent ${spent:,.2f} in the last hour vs a usual ${usual:,.2f}/hour. "
                       f"An agent may be stuck in a loop.")
                await notify(msg, state=state, title=f"Spend spike: {user}", level="critical", page="reports")
                sent.append(msg)
    return sent


async def weekly_digest(cfg: Config, store, state, now: datetime | None = None, force: bool = False) -> list[str]:
    now = now or datetime.now(UTC)
    year, week, _ = now.isocalendar()
    if not force and not await _once(state, f"digest:{year}-W{week:02d}", 8 * 86400):
        return []
    start = now - timedelta(days=7)
    rows = {r["key"]: r for r in await spend.summary(store, cfg, start, now, "team")}
    idle = rows.pop(spend.IDLE_KEY, None)  # GPU time no request paid for: the organization's, in its own digest below
    sent = []
    for team in sorted(rows):
        r = rows[team]
        msg = (f"Weekly LLM spend for '{team}': ${r['actual_cost']:,.2f}{_split(r)} across {r['requests']} requests. "
               f"Without Merit Order: ${r['baseline_cost']:,.2f}. {_verdict(r)}")
        await notify(msg, state=state, title=f"Weekly digest: {team}", level="info", page="reports", team=team)
        sent.append(msg)
    gpu = sum(r["gpu_cost"] for r in rows.values()) + (idle["gpu_cost"] if idle else 0)
    if gpu:  # with GPUs, the organization's own total: the teams' and the GPU time none of them paid for
        org = await spend.total(store, cfg, start, now)
        unpaid = f", ${org['gpu_idle_cost']:,.2f} of it GPU time no request paid for (idle, starting, draining)" if idle else ""
        msg = (f"Weekly LLM spend for the organization: ${org['actual_cost']:,.2f} (APIs ${org['api_cost']:,.2f}, "
               f"GPUs ${org['gpu_cost']:,.2f}{unpaid}) across {org['requests']} requests. "
               f"Without Merit Order: ${org['baseline_cost']:,.2f}. {_verdict(org)}")
        await notify(msg, state=state, title="Weekly digest: organization", level="info", page="reports")
        sent.append(msg)
    return sent


def _split(r: dict) -> str:
    return f" (APIs ${r['api_cost']:,.2f}, its share of GPU time ${r['gpu_cost']:,.2f})" if r.get("gpu_cost") else ""


def _verdict(r: dict) -> str:
    causes = ", ".join(f"{c} ${r[f'savings_{c}']:,.2f}" for c in SAVINGS_CAUSES if r.get(f"savings_{c}"))
    saved = round(r["savings_total"], 2)
    if saved > 0:
        return f"Saved ${saved:,.2f}" + (f" ({causes})." if causes else ".")
    if saved < 0:
        why = ("requested models pricier than the baseline, prompt-cache writes not yet read back, or GPUs that cost more than the "
               "API calls they replaced")
        return f"Cost ${-saved:,.2f} more than the baseline ({causes}): {why}." if causes else f"Cost ${-saved:,.2f} more than the baseline."
    return "No savings this week."
