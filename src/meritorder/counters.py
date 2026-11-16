"""This month's spend counters (Redis) checked against the ledger (Postgres), which stays the source of truth.

Admission reads the counters, not the ledger: one round trip per request. A counter that's missing is seeded from the
ledger (``Gateway.spend``); one that exists was trusted for the rest of the month. A Redis that comes back from an old
snapshot (an RDB file, a failover to a replica behind) has counters that exist but are behind, and budgets would stop
binding until the month ends. So the worker recomputes every counter of this month from the ledger each cycle
(``reconcile``), from the sum seeding uses (``spend.budget_spend``'s ledger + reserved), and corrects the ones that
disagree.

A request adds to its counters a moment before its ledger row is written, and a batch's reservation moves between the two
in steps too, so a counter and the ledger can disagree for a moment with nothing wrong. A difference is corrected only
when it's the same on two cycles in a row, and by adding the difference (never a SET), so what requests add meanwhile is
kept.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from . import spend
from .state import claim

log = logging.getLogger("meritorder.counters")

KINDS = ("org", "user", "team")
SEEN = "spend-reconcile:seen"  # the differences found last cycle, shared by the workers
LOCK = "lock:spend-reconcile"


def month_key(now: datetime | None = None) -> str:
    return (now or datetime.now(UTC)).strftime("%Y-%m")


async def counted(store, cfg, kind: str, ident: str, now: datetime | None = None) -> float:
    """What a counter should hold: spend.budget_spend's ledger + reserved, the sum admission seeds it with (the GPU time no
    request paid for is added at admission, never stored in the counter)."""
    s = await spend.budget_spend(store, cfg, kind, None if kind == "org" else ident, now)
    return s["ledger"] + s["reserved"]


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-6 + 1e-9 * max(abs(a), abs(b))


async def reconcile(store, state, cfg, now: datetime | None = None, lease_seconds: int = 45) -> dict[str, float]:
    """Correct this month's counters that disagree with the ledger by the same amount two cycles running. Returns what
    was added to each corrected counter. One worker at a time (a lease), so a correction is never made twice."""
    if not await claim(state, LOCK, lease_seconds):
        return {}
    month = month_key(now)
    seen = await state.get_json(SEEN) or {}
    diffs: dict[str, float] = {}
    fixed: dict[str, float] = {}
    for key, value in (await state.spend_counters(f"{month}:")).items():
        _, kind, ident = key.split(":", 2)
        if kind not in KINDS:
            continue
        diff = await counted(store, cfg, kind, ident, now) - value
        if _close(diff, 0.0):
            continue
        last = seen.get(key)
        if last is not None and _close(diff, float(last)):
            await state.adjust_spend([key], diff)
            fixed[key] = diff
            log.warning("spend counter %s was $%.6f off the ledger (Redis restored from an old snapshot?); corrected", key, diff)
        else:
            diffs[key] = diff
    await state.set_json(SEEN, diffs, 600)
    return fixed
