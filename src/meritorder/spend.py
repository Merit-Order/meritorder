"""What was spent, with GPU time counted the way the GPUs bill it (docs/08-reporting.md, decision 64).

A request served on a self-hosted GPU carries a share of the GPU's price in the ledger (pricing.chargeback: its tokens at
the deployment's capacity, at the hourly price). That share is what its team and its person are charged, and what their
budgets count. The GPUs themselves bill for replica time: the capacity controller accrues it every cycle while replicas
exist (store.add_gpu_usage), starting, idle and draining included, and the weights cache volumes with it. For the whole
organization the accrued time takes the place of the requests' shares:

    total    = ledger + gpu_idle
    gpu_idle = GPU time accrued - the shares of the requests served on those GPUs

``gpu_idle`` ("GPU idle, unallocated") is GPU time no request paid for: starting up, waiting for traffic, draining, and
capacity the requests left unused. It belongs to no team, since the controller decides when GPUs run from everyone's
demand, so it shows at organization level only. It is negative when the requests' shares came to more than the GPU
cost (capacity_tokens_per_second set below what the GPU really serves).

Only backends with GPU time accrued in the window are set against it. A self-hosted backend no controller runs (your
own vLLM with a configured gpu_cost_per_hour) has none, and its requests' shares stay its cost.

Savings follow from the same sum: the self-hosting cause loses ``gpu_idle``, so for the organization it is the API cost
the GPU-served requests avoided minus what the GPUs cost, and the causes still add up to baseline - total. Either can be
negative, and is shown that way.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from .store import SUM_FIELDS, month_start, sort_summary

IDLE_KEY = "GPU idle, unallocated"  # the summary row that carries gpu_idle when rows are teams, people, models or tags
TIME_GROUPS = ("none", "hour", "day", "month")
SPLIT_FIELDS = ("api_cost", "gpu_cost", "gpu_idle_cost")  # what every summary row adds to the ledger's sums


def gpu_backends(cfg, accrued: set[str] | frozenset[str] = frozenset()) -> list[str]:
    """Backends whose requests are GPU spend: the configured self-hosted ones, and any with GPU time accrued (one since
    removed from the configuration, a volume's deployment)."""
    return sorted({b.name for b in cfg.backends if b.kind == "self_hosted"} | set(accrued))


@dataclass
class GpuTime:
    """GPU time accrued in a window, and the shares the ledger charged requests on GPUs, both by hour and backend."""

    hourly: list[dict]  # store.gpu_hourly
    shares: list[dict]  # store.gpu_ledger, for every GPU backend

    @property
    def accrued_backends(self) -> set[str]:
        return {h["backend"] for h in self.hourly}

    @property
    def accrued(self) -> float:
        return sum(h["cost"] for h in self.hourly)

    def idle(self, key=lambda hour, backend: "total") -> dict:
        """gpu_idle by ``key(hour, backend)``: accrued time less the shares of requests on backends with accrued time."""
        out: dict = defaultdict(float)
        for h in self.hourly:
            out[key(h["hour"], h["backend"])] += h["cost"]
        names = self.accrued_backends
        for s in self.shares:
            if s["backend"] in names:
                out[key(s["hour"], s["backend"])] -= s["actual_cost"]
        return dict(out)


async def gpu_time(store, cfg, start: datetime, end: datetime, raw: bool = False) -> GpuTime:
    """``raw``: the requests' shares from the ledger itself, never the hourly rollup (for budgets)."""
    hourly = await store.gpu_hourly(start, end)
    names = gpu_backends(cfg, {h["backend"] for h in hourly})
    return GpuTime(hourly, await store.gpu_ledger(start, end, names, raw=raw) if names else [])


def _bucket(group_by: str):
    """Where an hour of a backend's GPU time lands among summary rows grouped by ``group_by``."""
    if group_by == "none":
        return lambda hour, backend: "total"
    if group_by == "hour":
        return lambda hour, backend: hour.astimezone(UTC).isoformat()
    if group_by == "day":
        return lambda hour, backend: hour.astimezone(UTC).date().isoformat()
    if group_by == "month":
        return lambda hour, backend: hour.astimezone(UTC).strftime("%Y-%m")
    if group_by == "backend":
        return lambda hour, backend: backend
    return lambda hour, backend: IDLE_KEY  # teams, people, models, tags: idle time is none of theirs


def _norm(key, group_by: str):
    """A summary row's key as _bucket writes it: stores write time keys differently ("2026-11-15" in memory, the day's
    start as a timestamp from Postgres and ClickHouse)."""
    if group_by not in TIME_GROUPS[1:] or key is None:
        return key
    s = key if isinstance(key, str) else key.isoformat()
    if group_by == "day":
        return s[:10]
    if group_by == "month":
        return s[:7]
    return datetime.fromisoformat(s).astimezone(UTC).isoformat()


def _key_like(k: str, group_by: str, full: bool) -> str:
    """A new row's key in the style of the store's own (``full``: as a timestamp)."""
    if not full or group_by not in ("day", "month"):
        return k
    return f"{k}T00:00:00+00:00" if group_by == "day" else f"{k}-01T00:00:00+00:00"


def _empty(key) -> dict:
    return {"key": key, "requests": 0, "errors": 0, "upgraded": 0, **dict.fromkeys(SUM_FIELDS, 0.0), "gpu_cost": 0.0, "savings_total": 0.0,
            **{f: 0 for f in ("input_tokens", "cache_read_tokens", "cache_write_tokens", "output_tokens")}}


def add_idle(row: dict, amount: float) -> None:
    """Count GPU time no request paid for in a row: it's spend, on GPUs, and it comes off self-hosting's savings."""
    row["actual_cost"] += amount
    row["gpu_cost"] = row.get("gpu_cost", 0.0) + amount
    row["gpu_idle_cost"] = row.get("gpu_idle_cost", 0.0) + amount
    row["savings_self_hosting"] -= amount
    row["savings_total"] -= amount


def finish(row: dict) -> dict:
    row.setdefault("gpu_cost", 0.0)
    row.setdefault("gpu_idle_cost", 0.0)
    row["api_cost"] = row["actual_cost"] - row["gpu_cost"]
    for f in ("actual_cost", "gpu_cost", "gpu_idle_cost", "api_cost", "savings_self_hosting", "savings_total"):
        row[f] = round(row[f], 6)
    return row


async def summary(store, cfg, start: datetime, end: datetime, group_by: str, user_id: str | None = None,
                  team: str | None = None) -> list[dict]:
    """store.summary with GPU spend in it. Every row says what of its cost was GPUs (``gpu_cost``) and what APIs
    (``api_cost``). For the whole organization (no person or team), GPU time no request paid for is added too
    (``gpu_idle_cost``): to its hour, day or month, to its backend, or else as a row of its own, IDLE_KEY. A person's or
    a team's rows carry their requests' shares only."""
    gt = await gpu_time(store, cfg, start, end)
    rows = await store.summary(start, end, group_by, user_id=user_id, team=team, gpu_backends=gpu_backends(cfg, gt.accrued_backends))
    if user_id is None and team is None:
        idle = {k: v for k, v in gt.idle(_bucket(group_by)).items() if abs(v) >= 1e-9}
        if idle:
            index = {_norm(r["key"], group_by): r for r in rows}
            full = any("T" in str(r["key"]) for r in rows) if rows else getattr(store, "kind", "memory") != "memory"
            for k, v in idle.items():
                if k not in index:
                    index[k] = _empty(_key_like(k, group_by, full))
                    rows.append(index[k])
                add_idle(index[k], v)
            rows = sort_summary(rows)
    return [finish(r) for r in rows]


async def total(store, cfg, start: datetime, end: datetime, user_id: str | None = None, team: str | None = None) -> dict:
    """One summary row for the window (``group_by`` none), empty when nothing was spent."""
    rows = await summary(store, cfg, start, end, "none", user_id=user_id, team=team)
    return rows[0] if rows else finish(_empty("total"))


async def gpu_idle(store, cfg, start: datetime, end: datetime | None = None) -> float:
    """GPU time no request paid for in the window: what the organization's budget counts on top of the ledger."""
    gt = await gpu_time(store, cfg, start, end or datetime.now(UTC) + timedelta(days=1), raw=True)
    return gt.idle().get("total", 0.0)


BUDGET_FIELDS = {"org": None, "team": "team", "user": "user_id"}


async def budget_spend(store, cfg, kind: str = "org", ident: str | None = None, now: datetime | None = None) -> dict:
    """This month's spend against a budget, the one total every surface shows for it: admission (Gateway.spend seeds its
    counters from it), Controls, the attention panel and the budget alerts. ``kind``: org, team (``ident`` its name) or
    user (``ident`` the person, across every team they were on this month).

    ``spent`` = ``ledger`` (every row this month, read from Postgres, as budgets read spend) + ``reserved`` (what running
    batches have set aside, batches/service.py) + ``gpu_idle`` for the organization (GPU time no request paid for; a
    team's or a person's GPU requests are in their ledger at their shares). For the organization also the split into
    ``api`` and ``gpu`` (both settled spend, reservations not included) and each GPU backend's cost (accrued time, or
    for a backend without any, its requests' shares)."""
    if kind not in BUDGET_FIELDS:
        raise ValueError(f"kind must be one of {', '.join(BUDGET_FIELDS)}")
    start = month_start(now)
    field = BUDGET_FIELDS[kind]
    ledger = await store.spend_all_since(start) if field is None else await store.spend_since(field, ident, start)
    reserved = await store.batch_reserved(start.strftime("%Y-%m"), field, ident)
    out = {"spent": ledger + reserved, "ledger": ledger, "reserved": reserved, "gpu_idle": 0.0}
    if kind != "org":
        return out
    gt = await gpu_time(store, cfg, start, (now or datetime.now(UTC)) + timedelta(days=1), raw=True)
    idle = gt.idle().get("total", 0.0)
    shares = sum(s["actual_cost"] for s in gt.shares)
    by_backend: dict[str, float] = defaultdict(float)
    for h in gt.hourly:
        by_backend[h["backend"]] += h["cost"]
    for s in gt.shares:
        if s["backend"] not in gt.accrued_backends:
            by_backend[s["backend"]] += s["actual_cost"]
    return {**out, "spent": ledger + reserved + idle, "api": ledger - shares, "gpu": shares + idle, "gpu_idle": idle,
            "gpu_accrued": gt.accrued, "gpu_by_backend": dict(by_backend)}
