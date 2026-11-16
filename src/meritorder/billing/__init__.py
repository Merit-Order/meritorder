"""Checking the ledger against providers' own cost data (docs/08-reporting.md, "Checking costs against the bills").

The worker fetches every configured source every ``billing.refresh_hours``: the last ``lookback_days`` each time, since
providers restate recent days. A fetch replaces that range's stored lines, so a restated day replaces the old one. An
export source (exports.py) replaces only the days of the refreshes whose files changed, and keeps what it read (its
manifests and their ETags) in its status, so the next fetch skips files it has already read.
Days younger than the source's settle time (or, for Anthropic, past its data_refreshed_at) are partial: shown, never
alerted on. A settled day whose bill and ledger differ by more than the configured threshold raises one alert per source
and day; a source that can't be fetched raises one every 6 hours.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime, timedelta

import httpx

from .. import alerts
from ..backends.auth import make_auth
from ..config import Config
from .reconcile import over, reconcile_all, report, settled
from .sources import BillError, Fetched, covered_backends, fetch, label_value, source_auth

__all__ = ["BillError", "BillRefresher", "Fetched", "check_gaps", "covered_backends", "label_value", "refresh_once", "report"]

log = logging.getLogger("meritorder.billing")

STATUS_KEY = "billing_status"


async def _fetch_one(src, cfg: Config, start: date, end: date, client: httpx.AsyncClient, auth_factory, state: dict | None) -> Fetched:
    spec = source_auth(src)
    if spec.type == "api_key" and not spec.key():
        raise BillError(f"{spec.describe()} is not set")
    try:
        auth = auth_factory(spec)
    except ImportError as e:
        extra = "gcp" if src.type == "gcp" else "azure"
        raise BillError(f"credentials need the {extra} extra (uv sync --extra {extra}; the Docker image has it): {e}") from e
    except Exception as e:  # noqa: BLE001 (no credentials found, a bad profile, ...: the status says which)
        raise BillError(f"no credentials: {type(e).__name__}: {e}"[:300]) from e
    if hasattr(auth, "available") and not auth.available():
        raise BillError("no AWS credentials found (profile, environment, or role)")
    return await fetch(src, cfg, start, end, client, auth, state)


async def refresh_once(cfg: Config, store, state, today: date | None = None, client: httpx.AsyncClient | None = None,
                       auth_factory=make_auth) -> dict:
    """Fetch every enabled source, store its lines, record how it went (``billing_status``), and check for gaps."""
    today = today or datetime.now(UTC).date()
    b = cfg.billing
    start, end = today - timedelta(days=b.lookback_days), today + timedelta(days=1)
    status = dict(await store.get_setting(STATUS_KEY) or {})
    own = client is None
    client = client or httpx.AsyncClient(timeout=180)
    try:
        for src in b.sources:
            if not src.enabled:
                continue
            old = status.get(src.name) or {}
            now = datetime.now(UTC).isoformat(timespec="seconds")
            try:
                got = await _fetch_one(src, cfg, start, end, client, auth_factory, old.get("export"))
                if got.ranges is None:  # an API: the whole range, restated
                    n, first = await store.replace_bill_lines(src.name, start, end, got.lines), start
                else:  # an export: the days of the refreshes whose files changed
                    n, first = 0, got.covered_from
                    for lo, hi in got.ranges:
                        n += await store.replace_bill_lines(src.name, lo, hi, [ln for ln in got.lines if lo <= ln["day"] < hi])
            except Exception as e:  # noqa: BLE001 (one source failing must not stop the others; the old lines stay)
                why = str(e) if isinstance(e, BillError) else f"{type(e).__name__}: {e}"
                status[src.name] = {**old, "ok": False, "error": why[:300], "failed_at": now}
                log.warning("bill fetch failed for %s: %s", src.name, why)
                if await alerts._once(state, f"alert:bill-source:{src.name}", 6 * 3600):
                    await alerts.notify(f"Couldn't fetch the {src.name} bill ({why[:200]}). Reports → Bills shows the last one fetched.",
                                        state=state, title=f"Bill unavailable: {src.name}", level="warning", page="reports")
                continue
            seen = [d for d in (date.fromisoformat(old["covered_from"]) if old.get("covered_from") else None, first) if d]
            covered = min(seen) if seen else None  # an export that has never found a file covers nothing yet
            status[src.name] = {"ok": True, "error": None, "fetched_at": now, "start": start.isoformat(), "end": end.isoformat(),
                                "covered_from": covered.isoformat() if covered else None, "lines": n, "notes": got.notes,
                                "complete_through": got.complete_through.isoformat() if got.complete_through else None,
                                **({"export": got.state} if got.state is not None else {})}
    finally:
        if own:
            await client.aclose()
    await store.set_setting(STATUS_KEY, status)
    await check_gaps(cfg, store, state, today)
    return status


def _usd(v: float) -> str:
    return f"{'+' if v > 0 else '-' if v < 0 else ''}${abs(v):,.2f}"


async def check_gaps(cfg: Config, store, state, today: date | None = None) -> list[str]:
    """One alert per source for settled days whose bill and ledger differ past the threshold, each day alerted once."""
    today = today or datetime.now(UTC).date()
    b = cfg.billing
    start = today - timedelta(days=b.lookback_days)
    sent = []
    for res in await reconcile_all(cfg, store, start, today + timedelta(days=1), today):
        new = []
        for day in sorted(res.days, reverse=True):
            d = res.days[day]
            gap = d.billed - d.recorded
            if settled(day, res.status, res.through) and over(gap, d.recorded, b):
                if await alerts._once(state, f"alert:bill:{res.src.name}:{day.isoformat()}", 60 * 86400):
                    new.append((day, d, gap))
        if not new:
            continue
        day, d, gap = new[0]
        pct = f", {gap / d.recorded:+.1%}" if d.recorded else ""
        worst = max(((m, r["billed"] - r["recorded"]) for (dd, m), r in res.rows.items() if dd == day), key=lambda x: abs(x[1]),
                    default=None)
        msg = (f"The {res.src.name} bill for {day.isoformat()} is ${d.billed:,.2f}; the ledger recorded ${d.recorded:,.2f} "
               f"({_usd(gap)}{pct}).")
        if worst and worst[1]:
            msg += f" Biggest difference: {worst[0]} {_usd(worst[1])}."
        if d.unmatched:
            msg += f" ${d.unmatched:,.2f} of the bill matched no model."
        if len(new) > 1:
            more = ", ".join(f"{dd.isoformat()} {_usd(g)}" for dd, _, g in new[1:4])
            msg += f" Also {len(new) - 1} more day{'s' if len(new) > 2 else ''} ({more}{', ...' if len(new) > 4 else ''})."
        msg += " Reports → Bills shows every line."
        await alerts.notify(msg, state=state, title=f"Bill differs from the ledger: {res.src.name}", level="warning", page="reports")
        sent.append(msg)
    return sent


class BillRefresher:
    """Calls ``refresh_once`` when ``billing.refresh_hours`` have passed since any worker last did (shared through state)."""

    async def maybe(self, cfg: Config, store, state, **kw) -> dict | None:
        if not any(s.enabled for s in cfg.billing.sources):
            return None
        last = await state.get_json("billing:last_run")
        now = datetime.now(UTC).timestamp()
        if last and now - float(last) < cfg.billing.refresh_hours * 3600:
            return None
        await state.set_json("billing:last_run", now, int(cfg.billing.refresh_hours * 3600) + 60)
        try:
            return await refresh_once(cfg, store, state, **kw)
        except Exception:  # noqa: BLE001 (a bad fetch must not stop the worker)
            log.exception("bill refresh failed")
            return None
