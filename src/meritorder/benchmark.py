"""Your own traffic as the routing benchmark (decision 59, docs/03-routing.md#the-outcome-report).

Four parts:

* ``outcomes_report``: per task type and level, the traffic and spend share, and the success rate of each exploration arm
  (served at the level, explored one level down, explored one level up, a canary), weighted by the inverse of each arm's
  chance (a Hájek estimate), so teams exploring at different shares, and canaries, pool without bias. From it, a "too
  high" estimate (how often one level down does as well), a "too low" one (how much more often one level up succeeds),
  the sample sizes and 90% intervals, and a recommendation once the evidence is enough.
* ``Watcher``: the worker's job. It notices when a model becomes the cheapest at a level or newly qualifies for it, makes
  it a canary (exploring teams try it on a share of the level's task starts before it takes the rest) until it has enough
  outcomes, records catalog and score changes as events for the report's trend, and alerts on new cheapest models and
  new recommendations. With ``routing.benchmark.auto_apply`` it applies a recommendation the learned router agrees with.
* ``apply``: a recommendation written into ``routing.level_map`` as a new configuration version.
* ``dismissal``: an admin can set a recommendation aside. It stays dismissed until its evidence changes: it points another
  way, or ``routing.benchmark.min_outcomes`` more task starts were explored on the level it compares with.

A task start's success is 1 minus the combined weight of its outcome signals (learned_jobs.signals: a raised floor, sent
again, stopped, asked again elsewhere, a correction, tests failing at the end, ...). Most task starts have no signal, so
rates are high and differences small; what matters is the comparison between arms of the same stratum, which the random
draw makes fair.
"""

from __future__ import annotations

import copy
import logging
import math
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any

from .learned_jobs import abandoned, failed, signals
from .store import ANY_VERSION, ConfigConflict

log = logging.getLogger("meritorder.benchmark")

SETTING = "routing_benchmark"  # the watcher's snapshot, canaries, events and the recommendations it alerted on
# Dismissed recommendations, {"coding:3": {"direction", "to", "at", "by"}}. A setting of its own, so the watcher's
# read-modify-write of SETTING can't lose one.
DISMISSED = "routing_benchmark_dismissed"
Z90 = 1.645
ARMS = ("served", "down", "up", "canary")
MAX_EVENTS = 200
WATCH_SECONDS = 900  # how often the watcher looks (the price refresh's default period)
REPORT_SECONDS = 3600  # how often it works the recommendations out again


def _rate(rows: list[tuple[float, float]]) -> dict:
    """Hájek estimate of a success rate from (weight, success) pairs: Σ w s / Σ w, with its standard error, and the
    effective sample size (Σ w)² / Σ w²."""
    sw = sum(w for w, _ in rows)
    if not rows or sw <= 0:
        return {"n": 0, "n_eff": 0.0, "rate": None, "se": None}
    rate = sum(w * s for w, s in rows) / sw
    var = sum((w * (s - rate)) ** 2 for w, s in rows) / sw ** 2
    n_eff = sw ** 2 / sum(w * w for w, _ in rows)
    return {"n": len(rows), "n_eff": round(n_eff, 1), "rate": round(rate, 4), "se": round(math.sqrt(var), 5)}


def _verdict(served: dict, other: dict, as_well: float, minimum: int, lower: bool) -> dict:
    """Compare an explored arm with the served one: the ratio of their rates, and whether the evidence says the other
    level does as well (``lower``: the level below) or clearly better (the level above)."""
    out: dict[str, Any] = {"ratio": None, "enough": False, "clear": False}
    if served["rate"] is None or other["rate"] is None or not served["rate"]:
        return out
    out["ratio"] = round(other["rate"] / served["rate"], 4)
    out["enough"] = min(served["n"], other["n"]) >= minimum
    if lower:  # the level below does as well: other - as_well x served is above 0, at the low end of its 90% interval
        diff, se = other["rate"] - as_well * served["rate"], math.hypot(other["se"], as_well * served["se"])
        out["low"] = round(diff - Z90 * se, 5)
        out["clear"] = out["enough"] and out["low"] > 0
    else:  # the level above does clearly better: served - as_well x other is below 0, at the high end of its interval
        diff, se = served["rate"] - as_well * other["rate"], math.hypot(served["se"], as_well * other["se"])
        out["high"] = round(diff + Z90 * se, 5)
        out["clear"] = out["enough"] and out["high"] < 0
    return out


def _after(stamps: list[datetime], at: str | None) -> int:
    """How many of ``stamps`` (task starts' times) fall after the ISO time ``at``."""
    try:
        t = datetime.fromisoformat(at or "")
    except ValueError:
        return len(stamps)
    t = t if t.tzinfo else t.replace(tzinfo=UTC)
    return sum(1 for s in stamps if (s if s.tzinfo else s.replace(tzinfo=UTC)) > t)


def dismissal(rec: dict, dismissed: dict | None, explored: list[datetime], min_outcomes: int) -> dict | None:
    """The dismissal that still holds for ``rec``, or None. One holds while the recommendation points the same way (same
    direction and target level) and fewer than ``min_outcomes`` task starts were explored on the compared level since
    (``explored``: their times). That's the evidence a recommendation needs in the first place, so new evidence of that
    size brings it back."""
    d = (dismissed or {}).get(f"{rec['task']}:{rec['level']}")
    if not d or d.get("direction") != rec["direction"] or d.get("to") != rec["to"]:
        return None
    new = _after(explored, d.get("at"))
    if new >= min_outcomes:
        return None
    return {"at": d.get("at"), "by": d.get("by"), "new_outcomes": new}


def outcomes_report(rows: list[dict], daily: list[dict], cfg: Any, events: list[dict] | None = None,
                    dismissed: dict | None = None) -> dict:
    """The two-sided report: ``rows`` are route_features rows (text optional), ``daily`` the ledger's requests and spend
    by day and floor (store.floor_daily), ``dismissed`` the DISMISSED setting. A dismissed recommendation stays on its
    level's row (``dismissed`` set) and leaves the ``recommendations`` list, so nothing alerts on it or applies it."""
    b = cfg.routing.benchmark
    gave_up = abandoned(rows) if rows and "text_features" in rows[0] else set()
    strata: dict[tuple[str, int], dict[str, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    learned_dir: dict[tuple[str, int], list[int]] = defaultdict(list)
    coding: dict[tuple[str, int], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    explored: dict[tuple[str, int], dict[str, list[datetime]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        arm, frm, p = r.get("arm"), r.get("arm_from"), r.get("propensity")
        if arm not in ARMS or not frm or not p or (r.get("status") or 0) >= 400 and r.get("status") != 499:
            continue
        key = (r["task"], int(frm))
        s = 1.0 - failed(signals(r, gave_up))
        strata[key][arm].append((1.0 / float(p), s))
        if arm in ("down", "up") and isinstance(r.get("ts"), datetime):
            explored[key][arm].append(r["ts"])
        lv = (r.get("learned") or {}).get("level")
        if arm == "served" and lv:
            learned_dir[key].append(int(lv) - int(frm))
        c = r.get("coding") or {}
        for k in ("tests", "tests_failed", "builds_failed", "turns"):
            coding[key][k] += int(c.get(k) or 0)
        coding[key]["corrected"] += 1 if c.get("correction") else 0
    # Traffic and spend share by floor, from the ledger.
    total_n = sum(d["requests"] for d in daily) or 1
    total_c = sum(d["cost"] for d in daily) or 0.0
    by_floor: dict[str, list[float]] = defaultdict(lambda: [0, 0.0])
    for d in daily:
        by_floor[d["floor"]][0] += d["requests"]
        by_floor[d["floor"]][1] += d["cost"]
    levels = []
    recommendations = []
    keys = sorted(set(strata) | {(f.split(":")[0], int(f.split(":")[1])) for f in by_floor if ":" in f and f.split(":")[1].isdigit()})
    for task, level in keys:
        arms = {a: _rate(strata[(task, level)].get(a, [])) for a in ARMS}
        down = _verdict(arms["served"], arms["down"], b.as_well, b.min_outcomes, True)
        up = _verdict(arms["served"], arms["up"], b.as_well, b.min_outcomes, False)
        n, c = by_floor.get(f"{task}:{level}", [0, 0.0])
        lv = learned_dir.get((task, level), [])
        row = {"task": task, "level": level, "requests": int(n), "traffic_share": round(n / total_n, 4),
               "spend": round(c, 4), "spend_share": round(c / total_c, 4) if total_c else None, "arms": arms,
               "too_high": down, "too_low": up, "learned": {"n": len(lv), "lower": sum(1 for x in lv if x < 0), "higher": sum(1 for x in lv if x > 0)},
               "coding": dict(coding.get((task, level), {})), "min_outcomes": b.min_outcomes}
        rec = None
        if down["clear"] and level > 1:
            rec = {"task": task, "level": level, "to": level - 1, "direction": "lower", "ratio": down["ratio"],
                   "n": [arms["served"]["n"], arms["down"]["n"]],
                   "text": f"{task} level {level}: level {level - 1} does as well in {down['ratio']:.0%} "
                           f"(n={arms['down']['n']:,} explored, {arms['served']['n']:,} served); consider lowering"}
        elif up["clear"] and level < 5:
            rec = {"task": task, "level": level, "to": level + 1, "direction": "raise", "ratio": up["ratio"],
                   "n": [arms["served"]["n"], arms["up"]["n"]],
                   "text": f"{task} level {level}: level {level + 1} succeeds {up['ratio']:.0%} as often "
                           f"(n={arms['up']['n']:,} explored, {arms['served']['n']:,} served); consider raising"}
        if rec:
            mapped = cfg.routing.level_map.get(f"{task}:{level}")
            rec["applied"] = mapped == rec["to"]
            arm = "down" if rec["direction"] == "lower" else "up"
            rec["dismissed"] = dismissal(rec, dismissed, explored[(task, level)][arm], b.min_outcomes)
            if not rec["dismissed"]:
                recommendations.append(rec)
        row["recommendation"] = rec
        levels.append(row)
    trend: dict[str, dict] = {}
    for d in daily:
        t = trend.setdefault(d["day"], {"day": d["day"], "requests": 0, "cost": 0.0, "by_floor": {}})
        t["requests"] += d["requests"]
        t["cost"] = round(t["cost"] + d["cost"], 6)
        t["by_floor"][d["floor"]] = round(t["by_floor"].get(d["floor"], 0.0) + d["cost"], 6)
    explored = sum(d["requests"] for d in daily if d.get("arm"))
    return {"levels": levels, "recommendations": recommendations, "trend": [trend[k] for k in sorted(trend)],
            "events": events or [], "explored_requests": explored, "requests": sum(d["requests"] for d in daily),
            "spend": round(total_c, 4), "level_map": dict(cfg.routing.level_map),
            "settings": {"min_outcomes": b.min_outcomes, "as_well": b.as_well, "auto_apply": b.auto_apply,
                         "canary_outcomes": b.canary_outcomes, "canary_days": b.canary_days},
            "exploring": sorted(t.name for t in cfg.teams if t.exploration.enabled),
            "recording": cfg.routing.learned.mode != "off"}


def apply(raw: dict, task: str, level: int, to: int) -> tuple[dict, list[str]]:
    """The configuration with ``task:level``'s work routed at ``to`` (routing.level_map), and what changed. Levels already
    mapped to ``level`` move with it, so a second recommendation on a moved level isn't lost; mapping a level to itself
    removes the entry."""
    doc = copy.deepcopy(raw)
    routing = doc.setdefault("routing", {})
    current = dict(routing.get("level_map") or {})
    changed = []
    for key in [k for k, v in current.items() if v == level and k.startswith(f"{task}:")] + [f"{task}:{level}"]:
        before = current.get(key, int(key.split(":")[1]))
        if before == to:
            continue
        if int(key.split(":")[1]) == to:
            current.pop(key, None)
        else:
            current[key] = to
        changed.append(f"{key} {before}→{to}")
    if current:
        routing["level_map"] = current
    else:
        routing.pop("level_map", None)
    return doc, changed


# ------------------------------------------------------------------ the worker's job


def snapshot(cfg: Any) -> dict:
    """Per task type and level: the cheapest model that qualifies (by list price of its cheapest priced route, input plus
    output per million tokens, mocks left out) and every model that qualifies; and each model's scores."""
    mocks = {bc.name for bc in cfg.backends if bc.kind == "mock"}

    def price(m) -> float:
        live = [r.cost_price() for r in m.routes if r.priced and r.backend not in mocks]
        return min((p.input + p.output for p in live), default=math.inf)

    chat = [(m, price(m)) for m in cfg.models if m.kind == "chat"]
    chat = [(m, p) for m, p in chat if p != math.inf]
    out: dict = {"cheapest": {}, "eligible": {}, "scores": {m.name: {t: round(m.q(t), 1) for t in ("coding", "general")} for m, _ in chat}}
    for task in ("coding", "general"):
        for level in range(1, 6):
            ok = sorted(((p, m.name) for m, p in chat if m.q(task) >= level))
            if ok:
                out["cheapest"][f"{task}:{level}"] = ok[0][1]
                out["eligible"][f"{task}:{level}"] = sorted(n for _, n in ok)
    return out


def changes(old: dict, new: dict, now: str) -> tuple[list[dict], list[dict]]:
    """Events (for the trend) and new canaries from one snapshot to the next."""
    events: list[dict] = []
    canaries: list[dict] = []
    added = sorted(set(new["scores"]) - set(old.get("scores") or {}))
    removed = sorted(set(old.get("scores") or {}) - set(new["scores"]))
    if added:
        events.append({"at": now, "kind": "catalog", "text": f"models added: {', '.join(added)}"})
    if removed:
        events.append({"at": now, "kind": "catalog", "text": f"models removed: {', '.join(removed)}"})
    moved = [f"{m} {t} {old['scores'][m][t]}→{s}" for m, per in new["scores"].items() for t, s in per.items()
             if m in (old.get("scores") or {}) and old["scores"][m].get(t) is not None and abs(old["scores"][m][t] - s) >= 0.1]
    if moved:
        events.append({"at": now, "kind": "scores", "text": "scores changed: " + "; ".join(moved[:10]) + (f"; and {len(moved) - 10} more" if len(moved) > 10 else "")})
    for key, model in new["cheapest"].items():
        task, level = key.split(":")
        before = (old.get("cheapest") or {}).get(key)
        if before and model != before:
            events.append({"at": now, "kind": "cheapest", "key": key, "model": model, "was": before,
                           "text": f"{model} is now the cheapest at {key} (was {before})"})
            canaries.append({"task": task, "level": int(level), "model": model, "since": now, "reason": f"newly cheapest (was {before})"})
        for name in sorted(set(new["eligible"].get(key, [])) - set((old.get("eligible") or {}).get(key, []))):
            if name == model and before and model != before:
                continue  # already a canary as the new cheapest
            if old.get("eligible") is not None and key in old.get("eligible", {}):
                events.append({"at": now, "kind": "eligible", "key": key, "model": name, "text": f"{name} now qualifies for {key}"})
                canaries.append({"task": task, "level": int(level), "model": name, "since": now, "reason": "newly qualifies"})
    return events, canaries


async def _learned_agrees(store, cfg: Any, rec: dict, row: dict) -> tuple[bool, str]:
    """Whether the learned router backs a recommendation: its model in use passed the held-out check, isn't stale, and
    on the level's served task starts it leans the same way more often than the other."""
    models = await store.router_models(5)
    active = next((m for m in models if m.get("active")), None)
    if active is None or not (active.get("metrics") or {}).get("better"):
        return False, "the learned router has no model in use that beat the rules"
    age = (datetime.now(UTC) - active["created_at"]).total_seconds() / 86400
    max_age = cfg.routing.learned.max_age_days
    if max_age and age > max_age:
        return False, f"the learned router's model is stale ({age:.0f} days old)"
    lean = row["learned"]
    ok = lean["lower"] > lean["higher"] if rec["direction"] == "lower" else lean["higher"] > lean["lower"]
    return ok, ("the learned router agrees" if ok else "the learned router doesn't lean that way on this level's requests")


async def apply_and_save(store, raw: dict, cfg_dir: str, prices: dict | None, rec: dict, author: str,
                         base_version: object = ANY_VERSION) -> dict:
    """Write a recommendation into a new configuration version (checked like any save). ``base_version``: the version
    ``raw`` is (None: the file). If another has been saved since, store.ConfigConflict, and nothing is saved."""
    from .configstore import parse_and_validate

    doc, changed = apply(raw, rec["task"], rec["level"], rec["to"])
    if not changed:
        return {"version": None, "changed": []}
    parse_and_validate(None, cfg_dir, prices, document=doc)
    version = await store.save_config(doc, author, f"routing.level_map from the outcome report: {'; '.join(changed)}"[:500],
                                      base_version=base_version)
    return {"version": version, "changed": changed, "document": doc}


class Watcher:
    """The worker's job (every WATCH_SECONDS): catalog events, canaries, and recommendation alerts."""

    def __init__(self) -> None:
        self.last = 0.0
        self.reported = 0.0

    async def maybe(self, cfg: Any, store, state, now: float | None = None, raw: dict | None = None) -> dict | None:
        now = now if now is not None else datetime.now(UTC).timestamp()
        if self.last and now - self.last < WATCH_SECONDS:
            return None
        self.last = now
        try:
            return await self.run(cfg, store, state, now, raw)
        except Exception:  # noqa: BLE001 (a bad run must not stop the worker)
            log.exception("routing benchmark watcher failed")
            return None

    async def run(self, cfg: Any, store, state, now: float, raw: dict | None = None) -> dict:
        from . import alerts
        from .configstore import current_token

        stamp = datetime.fromtimestamp(now, UTC).isoformat(timespec="seconds")
        doc = await store.get_setting(SETTING) or {}
        new = snapshot(cfg)
        events, fresh = changes(doc["snapshot"], new, stamp) if doc.get("snapshot") else ([], [])
        canaries = [c for c in doc.get("canaries") or [] if not any(f["task"] == c["task"] and f["level"] == c["level"]
                                                                     and f["model"] == c["model"] for f in fresh)] + fresh
        # A canary is done once it has enough recorded outcomes at its level, or has been one long enough.
        b = cfg.routing.benchmark
        kept = []
        if canaries:
            since = min(datetime.fromisoformat(c["since"]) for c in canaries)
            rows = await store.route_features(since, text=False)
            for c in canaries:
                start = datetime.fromisoformat(c["since"])
                n = sum(1 for r in rows if r["ts"] >= start and r.get("served_model") == c["model"] and r["task"] == c["task"]
                        and r.get("level") == c["level"])
                c["outcomes"] = n
                age = (now - start.timestamp()) / 86400
                if n >= b.canary_outcomes or age > b.canary_days or c["model"] not in new["eligible"].get(f"{c['task']}:{c['level']}", []):
                    events.append({"at": stamp, "kind": "canary", "text": f"canary {c['model']} at {c['task']}:{c['level']} done "
                                                                          f"({n} outcomes in {age:.1f} days)"})
                else:
                    kept.append(c)
        for e in events:
            if e["kind"] == "cheapest":
                await alerts.notify(f"{e['text']}. Teams that explore try it first on a share of the level's task starts "
                                    f"(Reports → Levels).", state=state, title=f"New cheapest model at {e['key']}", level="info",
                                    page="reports")
        out = {"events": len(events), "canaries": kept, "recommendations": []}
        if now - self.reported >= REPORT_SECONDS:
            self.reported = now
            since = datetime.fromtimestamp(now, UTC) - timedelta(days=30)
            rep = outcomes_report(await store.route_features(since, text=False), await store.floor_daily(since), cfg,
                                  dismissed=await store.get_setting(DISMISSED))
            alerted = set(doc.get("alerted") or [])
            for row in rep["levels"]:
                rec = row["recommendation"]
                if not rec or rec["applied"] or rec["dismissed"]:
                    continue
                key = f"{rec['task']}:{rec['level']}:{rec['direction']}"
                out["recommendations"].append(rec)
                if b.auto_apply and raw is not None:
                    agrees, why = await _learned_agrees(store, cfg, rec, row)
                    if agrees:
                        prices = await store.get_setting("prices")
                        # The latest version, read now: the cycle's copy may be older than an admin's save (or this pass's
                        # previous recommendation), and a change written onto it would silently undo theirs.
                        latest = await store.active_config()
                        base_raw, base = (latest["body"], latest["id"]) if latest else (raw, None)
                        try:
                            saved = await apply_and_save(store, base_raw, cfg.config_dir, prices, rec, "outcome-report", base)
                        except ConfigConflict as e:
                            log.info("not applying %s now: %s; the next pass tries again", key, e)
                            continue
                        if saved["version"]:
                            await state.set_config_token(await current_token(store, prices))
                            events.append({"at": stamp, "kind": "applied", "text": f"applied: {'; '.join(saved['changed'])}"})
                            await alerts.notify(f"Applied automatically ({why}): {rec['text']}", state=state,
                                                title="Level moved by the outcome report", level="info", page="reports")
                            alerted.add(key)
                            continue
                if key not in alerted:
                    alerted.add(key)
                    await alerts.notify(rec["text"], state=state, title=f"Level recommendation: {rec['task']} level {rec['level']}",
                                        level="info", page="reports")
            doc["alerted"] = sorted(alerted)
        doc.update(snapshot=new, canaries=kept, events=(doc.get("events") or []) + events)
        doc["events"] = doc["events"][-MAX_EVENTS:]
        await store.set_setting(SETTING, doc)
        prices = await store.get_setting("prices")
        if prices is not None and ((prices.get("exploration") or {}).get("canaries") or []) != kept:
            prices["exploration"] = {"canaries": kept, "updated": stamp}
            await store.set_setting("prices", prices)
            await state.set_config_token(await current_token(store, prices))
        return out
