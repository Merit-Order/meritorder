"""Matching bill lines to the ledger, and the report: what the ledger recorded against what the provider billed, by
provider, model and day (or month).

Every line of a source's bill ends up in one of three places, and the report says which and why:

* **matched**: the line names a model routed through the source's backends. It's compared with that model's ledger
  cost on the same UTC day. Batch-tier lines match too, and are also shown against the ledger's batch rows
  (``service_tier = 'batch'``), so a batch price that's off can be told from a live one.
* **not matched**: in scope but tied to no routed model: an unknown model name, another model the account uses, web
  search with no model on it. It counts in the source's billed total, so a gap it causes shows.
* **left out**: not comparable with the ledger at all: credits, refunds, taxes and other adjustments; lines ignored in
  the config; lines in a currency other than US dollars.

Model names differ between the ledger and bills ("anthropic.claude-sonnet-5-5", "Claude Sonnet 5.5", "gpt-6.1-terra-Inp-glbl").
Both sides are reduced to words (lowercase letters and digits; dates and provider prefixes dropped), and a line matches a
model when the model's words appear in it as a run, not followed by a version number or a variant word (so "Claude
Sonnet 5" doesn't take "Claude Sonnet 5.5", nor "Gemini 3.8 Flash" "Gemini 3.8 Flash-Lite"). The longest match wins.
``billing.sources.<name>.models`` maps lines the rule misses.
"""

from __future__ import annotations

import fnmatch
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta

from ..config import BillingConfig, BillSource, Config, uses_reported_cost
from .sources import SETTLE_DAYS, bill_of, covered_backends, gpu_deployments, vertex_projects

# What to do about costs in another currency, where the source can do something about it.
USD_HINT = {"azure": " (set cost_column: CostUSD)",
            "azure_export": " (this export has no US-dollar cost column: EA accounts' actual-cost exports don't; FOCUS has x_BilledCostInUsd)"}

TOKEN_KINDS = ("input", "cache_read", "cache_write", "output")
# Words that make a different model when they follow a model's name: "flash" -> "flash lite", "gpt 6 luna" -> "... mini".
VARIANTS = {"lite", "mini", "nano", "pro", "max", "plus", "turbo", "flash", "coder", "thinking", "vision", "instruct", "large", "small",
            "medium", "ultra", "air", "next", "haiku", "sonnet", "opus", "fable", "audio", "realtime", "search", "image", "embedding",
            "codex", "chat", "reasoning", "distill"}
_DATE = re.compile(r"(?<![0-9])(?:19|20)[0-9]{2}-?(?:0[1-9]|1[0-2])-?(?:0[1-9]|[12][0-9]|3[01])(?![0-9])")
_PROVIDER = re.compile(r"^(?:(?:us|eu|apac|global|jp|au|ca|us-gov)\.)?(?:anthropic|amazon|meta|mistral|cohere|ai21|deepseek|openai|"
                       r"qwen|writer|google|moonshot|moonshotai|minimax|zai|xai)\.")


def words(text: str) -> list[str]:
    """A line's text as words: lowercase letters and digits, dates left out."""
    return re.findall(r"[a-z0-9]+", _DATE.sub(" ", text.lower()))


def id_words(model_id: str) -> list[str]:
    """A model id as words, without a provider prefix ("google/", "us.anthropic."), a Bedrock version (":0") or a date."""
    s = model_id.strip().lower().rsplit("/", 1)[-1]
    s = _PROVIDER.sub("", s)
    s = re.sub(r"-v[0-9]+:[0-9]+$", "", s)
    return words(s.replace("@", " "))


def contains(line: list[str], key: tuple[str, ...]) -> bool:
    """The key's words appear in the line as a run, not continued by a version number or a variant word."""
    n = len(key)
    for i in range(len(line) - n + 1):
        if tuple(line[i:i + n]) == key:
            after = line[i + n] if i + n < len(line) else None
            if after is None or not ((after.isdigit() and len(after) <= 2) or after in VARIANTS):
                return True
    return False


class Matcher:
    """Which routed model, if any, a source's bill line is for."""

    def __init__(self, src: BillSource, cfg: Config, backends: list[str], upstream: dict[str, str] | None = None):
        keys: dict[tuple[str, ...], str] = {}
        for m in cfg.models:
            for r in m.routes:
                if r.backend in backends:
                    for k in (r.model, r.price_key, m.benchmark_name):
                        if k:
                            keys.setdefault(tuple(id_words(k)), m.name)
        for up, model in (upstream or {}).items():  # ids the ledger saw, including routes no longer configured
            keys.setdefault(tuple(id_words(up)), model)
        self.keys = sorted(((k, v) for k, v in keys.items() if k), key=lambda kv: -len(kv[0]))
        self.src, self.cfg, self.backends = src, cfg, backends

    def classify(self, ln: dict) -> tuple[str, str]:
        """("model", name), ("unmatched", why) or ("excluded", why)."""
        text, kind, detail = ln["line"], ln["kind"], ln.get("detail") or {}
        low = text.lower()
        if kind == "adjustment":
            return "excluded", "credit, refund, tax or other adjustment: not compared"
        if any(fnmatch.fnmatchcase(low, p.lower()) for p in self.src.ignore):
            return "excluded", f"ignored (billing.sources.{self.src.name}.ignore)"
        currency = (detail.get("currency") or "USD").upper()
        if currency != "USD":
            return "excluded", f"billed in {currency}; costs are compared in US dollars" + USD_HINT.get(self.src.type, "")
        for pattern, name in self.src.models.items():
            m = self.cfg.model(name)  # build_config checks the names; this is for a config built in code
            if m is not None and fnmatch.fnmatchcase(low, pattern.lower()):
                return "model", m.name
        if ln.get("model"):
            w = id_words(ln["model"])
            for k, name in self.keys:
                if tuple(w) == k:
                    return "model", name
            for k, name in self.keys:
                if contains(w, k):
                    return "model", name
            return "unmatched", f"{ln['model']} isn't routed through {', '.join(self.backends) or 'this source'}"
        w = words(text)
        for k, name in self.keys:
            if contains(w, k):
                return "model", name
        if kind == "web_search":
            return "unmatched", "web search not tied to a model"
        return "unmatched", "no routed model's name in this line"


def scoped(src: BillSource, cfg: Config, backends: list[str]) -> bool:
    """Whether the source reads a part of the account (a workspace, project, linked account, resource group) rather than
    all of it. Reading all of it is fine when Merit Order is its only user; otherwise other usage shows as gaps."""
    if src.type == "anthropic":
        return bool(src.workspace_ids or src.api_key_ids)
    if src.type == "openai":
        return bool(src.project_ids or src.api_key_ids)
    if src.type in ("aws", "aws_cur"):
        return bool(src.accounts)
    if src.type == "gcp":
        return bool(src.project_ids or vertex_projects(cfg, backends))
    return "/resourcegroups/" in (src.scope or "").lower() or bool(src.resource_types)


def complete_through(src: BillSource, status: dict, today: date) -> date | None:
    """The last settled day: what the provider says is complete, and at least settle_days old. None: never fetched."""
    if not status.get("fetched_at") or not status.get("covered_from"):
        return None
    settle = src.settle_days if src.settle_days is not None else SETTLE_DAYS[src.type]
    through = today - timedelta(days=settle)
    told = status.get("complete_through")
    return min(through, date.fromisoformat(told)) if told else through


def settled(day: date, status: dict, through: date | None) -> bool:
    return through is not None and date.fromisoformat(status["covered_from"]) <= day <= through


def over(gap: float, recorded: float, b: BillingConfig) -> bool:
    """A gap worth an alert: at least alert_min_usd, and at least alert_pct of what the ledger recorded."""
    return abs(gap) >= b.alert_min_usd and (recorded <= 0 or abs(gap) / recorded * 100 >= b.alert_pct)


def _pct(diff: float, recorded: float) -> float | None:
    return round(diff / recorded * 100, 2) if recorded else None


@dataclass
class Day:
    recorded: float = 0.0
    matched: float = 0.0
    unmatched: float = 0.0
    excluded: float = 0.0
    adjustments: float = 0.0

    @property
    def billed(self) -> float:
        return self.matched + self.unmatched


@dataclass
class SourceResult:
    src: BillSource
    backends: list[str]
    status: dict
    through: date | None
    rows: dict[tuple[date, str], dict] = field(default_factory=dict)  # (day, model) -> recorded and billed
    days: dict[date, Day] = field(default_factory=lambda: defaultdict(Day))
    unmatched: list[dict] = field(default_factory=list)  # one per line not matched or left out, with why


def _row() -> dict:
    return {"recorded": 0.0, "billed": 0.0, "requests": 0, "recorded_tokens": dict.fromkeys(TOKEN_KINDS, 0), "billed_tokens": {},
            "billed_by_kind": {}, "lines": 0, "recorded_batch": 0.0, "billed_batch": 0.0}


async def reconcile_source(src: BillSource, cfg: Config, store, start: date, end: date, status: dict, today: date) -> SourceResult:
    backends = covered_backends(src, cfg)
    t0, t1 = (datetime.combine(d, datetime.min.time(), tzinfo=UTC) for d in (start, end))
    ledger = await store.ledger_costs(t0, t1, backends) if backends else []
    res = SourceResult(src, backends, status, complete_through(src, status, today))
    for r in ledger:
        row = res.rows.setdefault((r["day"], r["served_model"] or "(unknown)"), _row())
        row["recorded"] += r["actual_cost"]
        row["requests"] += r["requests"]
        if r.get("service_tier") == "batch":
            row["recorded_batch"] += r["actual_cost"]
        for k in TOKEN_KINDS:
            row["recorded_tokens"][k] += r[f"{k}_tokens"]
        res.days[r["day"]].recorded += r["actual_cost"]
    upstream = {r["upstream_model"]: r["served_model"] for r in ledger if r["upstream_model"] and r["served_model"]}
    matcher = Matcher(src, cfg, backends, upstream)
    for ln in await store.bill_lines(start, end, src.name):
        if ln["kind"] == "gpu":
            continue
        cls, what = matcher.classify(ln)
        d = res.days[ln["day"]]
        if cls == "model":
            row = res.rows.setdefault((ln["day"], what), _row())
            row["billed"] += ln["cost"]
            row["lines"] += 1
            if (ln.get("detail") or {}).get("tier") == "batch":
                row["billed_batch"] += ln["cost"]
            row["billed_by_kind"][ln["kind"]] = row["billed_by_kind"].get(ln["kind"], 0.0) + ln["cost"]
            if ln.get("tokens") is not None and ln["kind"] in TOKEN_KINDS:
                row["billed_tokens"][ln["kind"]] = row["billed_tokens"].get(ln["kind"], 0) + ln["tokens"]
            d.matched += ln["cost"]
            continue
        if cls == "unmatched":
            d.unmatched += ln["cost"]
        elif ln["kind"] == "adjustment":
            d.adjustments += ln["cost"]
        else:
            d.excluded += ln["cost"]
        res.unmatched.append({"day": ln["day"], "line": ln["line"], "cost": ln["cost"], "status": cls, "why": what})
    return res


def _period(d: date, by: str) -> str:
    return d.isoformat() if by == "day" else d.strftime("%Y-%m")


def _money(v: float) -> float:
    return round(v, 6)


async def _gpu(cfg: Config, store, results: list[SourceResult], start: date, end: date, by: str) -> list[dict]:
    """GPU time the controller recorded (SkyPilot catalog prices) against what the clouds billed for instances tagged with
    each deployment. A deployment can move between clouds, so its bill is the sum over every source."""
    deployments = gpu_deployments(cfg)
    gpu_sources = [r for r in results if r.src.gpu_tag and bill_of(r.src.type) in ("aws", "gcp", "azure")]
    if not deployments or not gpu_sources:
        return []
    rows: dict[tuple[str, str], dict] = {}

    def row(day: date, backend: str) -> dict:
        r = rows.setdefault((_period(day, by), backend), {"period": _period(day, by), "deployment": backend, "recorded": 0.0, "billed": 0.0,
                                                          "billed_by_source": {}, "partial": False})
        settled_somewhere = any(settled(day, s.status, s.through) for s in gpu_sources)
        r["partial"] = r["partial"] or not settled_somewhere
        return r

    t0 = datetime.combine(start, datetime.min.time(), tzinfo=UTC)
    for g in await store.gpu_daily(t0):
        day = date.fromisoformat(g["day"])
        if day < end and g["backend"] in deployments.values():
            row(day, g["backend"])["recorded"] += g["cost"]
    for s in gpu_sources:
        for ln in await store.bill_lines(start, end, s.src.name):
            if ln["kind"] != "gpu":
                continue
            backend = deployments.get(ln["deployment"] or "")
            if backend is None:
                s.unmatched.append({"day": ln["day"], "line": ln["line"], "cost": ln["cost"], "status": "unmatched",
                                    "why": f"GPU tag {ln['deployment']!r} isn't a SkyPilot deployment in the config"})
                continue
            r = row(ln["day"], backend)
            r["billed"] += ln["cost"]
            r["billed_by_source"][s.src.name] = _money(r["billed_by_source"].get(s.src.name, 0.0) + ln["cost"])
    out = []
    for r in sorted(rows.values(), key=lambda x: (x["period"], x["deployment"])):
        diff = r["billed"] - r["recorded"]
        out.append({**r, "recorded": _money(r["recorded"]), "billed": _money(r["billed"]), "difference": _money(diff),
                    "pct": _pct(diff, r["recorded"]), "over": not r["partial"] and over(diff, r["recorded"], cfg.billing)})
    return out


async def _uncovered(cfg: Config, store, results: list[SourceResult], start: date, end: date) -> list[dict]:
    """Ledger spend on backends no bill source covers, and why."""
    covered = {b for r in results for b in r.backends}
    t0, t1 = (datetime.combine(d, datetime.min.time(), tzinfo=UTC) for d in (start, end))
    out = []
    for r in await store.summary(t0, t1, "backend"):
        name = r["key"]
        if not r["actual_cost"] or name in covered:
            continue
        b = cfg.backend(name)
        if b is None:
            why = "not in the config any more"
        elif b.kind == "self_hosted":
            why = "self-hosted: GPU time is compared with the cloud bill under GPUs"
        elif b.kind == "mock":
            why = "mock backend"
        elif uses_reported_cost(b):
            why = "cost is what the provider billed, returned with each response"
        else:
            why = "no bill source covers it (billing.sources)"
        out.append({"backend": name, "recorded": _money(r["actual_cost"]), "why": why})
    return sorted(out, key=lambda x: -x["recorded"])


async def reconcile_all(cfg: Config, store, start: date, end: date, today: date | None = None) -> list[SourceResult]:
    today = today or datetime.now(UTC).date()
    status = await store.get_setting("billing_status") or {}
    return [await reconcile_source(s, cfg, store, start, end, status.get(s.name) or {}, today)
            for s in cfg.billing.sources if s.enabled]


async def report(cfg: Config, store, start: date, end: date, by: str = "day", today: date | None = None) -> dict:
    """Recorded against billed for [start, end), by day or month, provider and model, with every line that didn't match."""
    results = await reconcile_all(cfg, store, start, end, today)
    b = cfg.billing
    gpu = await _gpu(cfg, store, results, start, end, by)  # first: GPU lines with an unknown tag join the unmatched
    rows, sources, unmatched = [], [], []
    for res in results:
        merged: dict[tuple[str, str], dict] = {}
        for (day, model), r in res.rows.items():
            m = merged.setdefault((_period(day, by), model), {**_row(), "partial": False})
            for k in ("recorded", "billed", "requests", "lines", "recorded_batch", "billed_batch"):
                m[k] += r[k]
            for k in ("recorded_tokens", "billed_tokens", "billed_by_kind"):
                for kk, v in r[k].items():
                    m[k][kk] = m[k].get(kk, 0) + v
            m["partial"] = m["partial"] or not settled(day, res.status, res.through)
        for (period, model), m in sorted(merged.items()):
            diff = m["billed"] - m["recorded"]
            status = ("partial" if m["partial"] else "matched" if not (m["billed"] or m["recorded"])
                      else "not_billed" if m["billed"] == 0 else "not_recorded" if m["recorded"] == 0 else "matched")
            rows.append({"period": period, "source": res.src.name, "type": res.src.type, "model": model, **m,
                         "recorded": _money(m["recorded"]), "billed": _money(m["billed"]), "difference": _money(diff),
                         "recorded_batch": _money(m["recorded_batch"]), "billed_batch": _money(m["billed_batch"]),
                         "pct": _pct(diff, m["recorded"]), "status": status,
                         "billed_by_kind": {k: _money(v) for k, v in m["billed_by_kind"].items()},
                         "over": status != "partial" and over(diff, m["recorded"], b)})
        groups: dict[tuple, dict] = {}
        for u in res.unmatched:
            g = groups.setdefault((u["line"], u["status"], u["why"]), {"source": res.src.name, "line": u["line"], "status": u["status"],
                                                                      "why": u["why"], "cost": 0.0, "days": set()})
            g["cost"] += u["cost"]
            g["days"].add(u["day"])
        for g in sorted(groups.values(), key=lambda x: -abs(x["cost"])):
            days = sorted(g.pop("days"))
            unmatched.append({**g, "cost": _money(g["cost"]), "days": len(days), "first": days[0].isoformat(), "last": days[-1].isoformat()})
        total, settled_total = Day(), Day()
        partial_days = 0
        for day, d in res.days.items():
            ok = settled(day, res.status, res.through)
            partial_days += 0 if ok else 1
            for t in (total, settled_total) if ok else (total,):
                for k in ("recorded", "matched", "unmatched", "excluded", "adjustments"):
                    setattr(t, k, getattr(t, k) + getattr(d, k))

        def totals(t: Day) -> dict:
            diff = t.billed - t.recorded
            return {"recorded": _money(t.recorded), "billed": _money(t.billed), "billed_matched": _money(t.matched),
                    "billed_unmatched": _money(t.unmatched), "left_out": _money(t.excluded), "adjustments": _money(t.adjustments),
                    "difference": _money(diff), "pct": _pct(diff, t.recorded)}

        st = res.status
        sources.append({"name": res.src.name, "type": res.src.type, "backends": res.backends, "ok": st.get("ok"), "error": st.get("error"),
                        "fetched_at": st.get("fetched_at"), "failed_at": st.get("failed_at"), "covered_from": st.get("covered_from"),
                        "complete_through": res.through.isoformat() if res.through else None, "notes": st.get("notes") or [],
                        "filtered": scoped(res.src, cfg, res.backends),
                        "partial_days": partial_days, "totals": totals(total), "settled": totals(settled_total)})
    overall = Day()
    for s in sources:
        overall.recorded += s["settled"]["recorded"]
        overall.matched += s["settled"]["billed_matched"]
        overall.unmatched += s["settled"]["billed_unmatched"]
        overall.excluded += s["settled"]["left_out"]
        overall.adjustments += s["settled"]["adjustments"]
    diff = overall.billed - overall.recorded
    return {"start": start.isoformat(), "end": end.isoformat(), "by": by,
            "thresholds": {"alert_pct": b.alert_pct, "alert_min_usd": b.alert_min_usd},
            "settled": {"recorded": _money(overall.recorded), "billed": _money(overall.billed), "billed_unmatched": _money(overall.unmatched),
                        "left_out": _money(overall.excluded), "adjustments": _money(overall.adjustments),
                        "difference": _money(diff), "pct": _pct(diff, overall.recorded)},
            "sources": sources, "rows": rows, "unmatched": unmatched,
            "gpu": gpu,
            "uncovered": await _uncovered(cfg, store, results, start, end)}
