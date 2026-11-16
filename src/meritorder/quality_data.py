"""Quality scores from published benchmarks, so `auto` routes on how good models are, not on numbers typed in.

Sources (``quality.source``):

- ``epoch``: Epoch AI's benchmarking data (https://epoch.ai/benchmarks, CC BY 4.0). Epoch places every model on one
  ability scale, the Epoch Capabilities Index (ECI), fitted across about 40 benchmarks, and every benchmark on the
  same scale by difficulty (EDI) and slope, so that a model's chance of solving a benchmark's tasks is
  ``sigmoid(slope x (ability - difficulty))``. That's what makes models comparable even when they've been tested on
  different benchmarks.
- ``artificial_analysis``: the Artificial Analysis Intelligence Index (https://artificialanalysis.ai, a free key in
  ``ARTIFICIAL_ANALYSIS_API_KEY``), used for every task type: AA no longer publishes a coding index.
- ``config``: the scores written in the configuration (or applied from this gateway's own eval runs).

With Epoch, for each task type:

1. A model's ability is its ECI, or, where it has results on the task type's own benchmarks (``quality.evidence``,
   e.g. FrontierSWE and FrontierCode for coding), the ability those results imply (each inverted through its
   benchmark's difficulty and slope, then averaged): its coding strength, not just its general one. A model with
   neither gets the ability its results on older benchmarks of the kind imply (``quality.fallback``, e.g. Terminal
   Bench, GSO and MirrorCode for coding). Those come after the ECI because, checked against the models that have
   FrontierSWE or FrontierCode results, none of them predicts those results better than the ECI does.
2. That ability becomes the chance of solving a task as hard as the reference benchmark (``quality.reference``,
   FrontierSWE by default for both coding and general work).
3. The scale (``quality.scale``) turns chances into 1-5 scores with one decimal: ``best`` is 5 x (the model's chance /
   the best model's), so the best is 5 and adding a weak model changes nobody else's score; ``range`` puts the
   weakest configured model at 1 and the best at 5.

The reference sets how spread out the scale is: against a harder task, weaker models fall further behind.
"""

from __future__ import annotations

import csv
import io
import math
import re
import zipfile
from datetime import UTC, datetime

import httpx

from .config import Config, ModelConfig

EPOCH_ZIP = "https://epoch.ai/data/benchmark_data.zip"
AA_MODELS = "https://artificialanalysis.ai/api/v2/language/models/free"
# Epoch's file for each benchmark that can be used as evidence (its name as in Epoch's difficulty table).
EPOCH_FILES = {
    "FrontierSWE": "frontierswe_external.csv",
    "FrontierCode": "frontiercode_external.csv",
    "SciCode": "scicode_external.csv",
    "Remote Labor Index": "rli_external.csv",
    "FrontierMath-2025-02-28-Private": "frontiermath_tiers_1_3_v2.csv",
    "SimpleQA Verified": "simpleqa_verified.csv",
    "ARC-AGI-2": "arc_agi_2_external.csv",
    "GPQA diamond": "gpqa_diamond.csv",
    "Terminal Bench": "terminalbench_external.csv",
    "GSO-Bench": "gso_external.csv",
    "MirrorCode": "mirrorcode.csv",
}
# The column holding the result Epoch fits difficulties on, first match wins. (WeirdML v3 isn't usable from its file:
# the index is fitted on a rescaled score that the file doesn't carry.)
SCORE_COLUMNS = ("Score", "Main score", "mean_score", "Pass@1 score", "Accuracy", "Accuracy mean", "Score OPT@1")
_EFFORT = re.compile(r"_(max|xhigh|high|medium|low|minimal|none|unknown)$")
ATTRIBUTION = {"epoch": "Epoch AI, Capabilities & benchmarking (epoch.ai/benchmarks), CC BY 4.0",
               "artificial_analysis": "Artificial Analysis (artificialanalysis.ai)"}


def norm(name: str) -> str:
    """'GPT-6.1 Sol' and 'gpt-6.1-sol' alike: lowercase, every run of other characters one hyphen."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def model_keys(m: ModelConfig) -> list[str]:
    """The names a model might have in benchmark data, most specific first (the first that matches is used): its
    benchmark_name, route model ids (without provider prefixes such as anthropic/ or anthropic.), its Hugging Face
    repository name (Qwen3-Coder-480B-A35B-Instruct, how Epoch names some open models), and aliases that aren't patterns."""
    names = [m.benchmark_name] if m.benchmark_name else []
    for r in m.routes:
        ident = r.price_key or r.model
        names.append(ident.rsplit("/", 1)[-1])
        if ident.startswith("anthropic."):  # Bedrock ids
            names.append(ident.split(".", 1)[1])
    if m.hf_id:
        names.append(m.hf_id.rsplit("/", 1)[-1])
    names += [a.rsplit("/", 1)[-1] for a in m.aliases if "*" not in a and "?" not in a]
    return list(dict.fromkeys(norm(n) for n in names if n))


def _logit(p: float) -> float:
    p = min(max(p, 0.01), 0.99)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


# ------------------------------------------------------------------ fetching

def parse_epoch(zip_bytes: bytes) -> dict:
    """From Epoch's benchmark_data.zip: {"eci": {model: ability}, "difficulty": {benchmark: (edi, slope)},
    "results": {benchmark: {model: [(effort variant, score)]}}}, models by normalized name."""
    out: dict = {"eci": {}, "difficulty": {}, "results": {}}
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        files = {n.rsplit("/", 1)[-1]: n for n in z.namelist()}

        def rows(fname):
            if fname not in files:
                return []
            return list(csv.DictReader(io.TextIOWrapper(z.open(files[fname]), encoding="utf-8", errors="replace")))
        for r in rows("eci_scores.csv"):
            try:
                out["eci"][norm(r.get("Display name") or r.get("Model") or "")] = float(r["eci"])
            except (KeyError, ValueError):
                continue
        for r in rows("edi_scores.csv"):
            try:
                out["difficulty"][r["benchmark_name"]] = (float(r["edi"]), float(r["estimated_slope_scaled"]))
            except (KeyError, ValueError):
                continue
        for bench, fname in EPOCH_FILES.items():
            table = rows(fname)
            if not table or "Model version" not in table[0]:
                continue
            col = next((c for c in SCORE_COLUMNS if c in table[0]), None)
            if col is None:
                continue
            for r in table:
                version = (r.get("Model version") or "").strip().lower()
                try:
                    score = float(r.get(col) or "")
                except ValueError:
                    continue
                m = _EFFORT.search(version)
                out["results"].setdefault(bench, {}).setdefault(norm(_EFFORT.sub("", version)), []).append((m.group(1) if m else "", score))
    return out


def fetch_epoch() -> dict:
    resp = httpx.get(EPOCH_ZIP, timeout=120, follow_redirects=True)
    resp.raise_for_status()
    return parse_epoch(resp.content)


def fetch_artificial_analysis(key: str) -> dict[str, dict]:
    """{normalized slug or name: {"intelligence": index}} from the free-tier models list (every page)."""
    out: dict[str, dict] = {}
    page = 1
    while True:
        resp = httpx.get(AA_MODELS, params={"page": page}, headers={"x-api-key": key}, timeout=60)
        resp.raise_for_status()
        body = resp.json()
        for m in body.get("data") or []:
            index = (m.get("evaluations") or {}).get("artificial_analysis_intelligence_index")
            if index is None:
                continue
            for name in (m.get("slug"), m.get("name")):
                if name:
                    out[norm(name)] = {"intelligence": float(index), "name": m.get("name")}
        if not (body.get("pagination") or {}).get("has_more"):
            return out
        page += 1


# ------------------------------------------------------------------ measuring

def _pick(variants: list[tuple[str, float]], effort: str | None) -> tuple[str, float]:
    """A model's result on a benchmark: at the effort it's routed with for the hardest work if published, else its best."""
    same = [v for v in variants if effort and v[0] == effort]
    return max(same or variants, key=lambda v: v[1])


def _evidence(data: dict, benches: list[str], keys: list[str], effort: str | None) -> dict[str, dict]:
    """{benchmark: {"score", "variant", "ability"}} for each of these benchmarks the model has a result on."""
    out = {}
    for bench in benches:
        table = data["results"].get(bench, {})
        found = next((table[k] for k in keys if k in table), None)
        if found and bench in data["difficulty"]:
            variant, score = _pick(found, effort)
            edi, slope = data["difficulty"][bench]
            out[bench] = {"score": round(score, 4), "variant": variant, "ability": round(edi + _logit(score) / slope, 2)}
    return out


def epoch_measures(cfg: Config, data: dict, effort: str | None) -> dict[str, dict[str, dict]]:
    """{model: {task: {"chance", "ability", "basis", "evidence": {benchmark: {"score", "variant", "ability"}}}}}."""
    out: dict[str, dict[str, dict]] = {}
    for m in cfg.models:
        if m.kind != "chat":
            continue
        keys = model_keys(m)
        eci = next((data["eci"][k] for k in keys if k in data["eci"]), None)
        for task, ref in cfg.quality.reference.items():
            if ref not in data["difficulty"]:
                continue
            evidence = _evidence(data, cfg.quality.evidence.get(task, []), keys, effort)
            fallback = {} if evidence or eci is not None else _evidence(data, cfg.quality.fallback.get(task, []), keys, effort)
            if evidence or fallback:
                found = evidence or fallback
                ability = sum(e["ability"] for e in found.values()) / len(found)
                basis = "results on " + ", ".join(found) + ("" if evidence else " (no index or newer results)")
                evidence = found
            elif eci is not None:
                ability, basis = eci, "Epoch Capabilities Index"
            else:
                continue
            edi, slope = data["difficulty"][ref]
            out.setdefault(m.name, {})[task] = {"chance": round(_sigmoid(slope * (ability - edi)), 4), "ability": round(ability, 2),
                                                "eci": eci, "basis": basis, "evidence": evidence}
    return out


def aa_measures(cfg: Config, data: dict) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for m in cfg.models:
        if m.kind != "chat":
            continue
        hit = next((data[k] for k in model_keys(m) if k in data), None)
        if hit:
            for task in cfg.quality.reference:
                out.setdefault(m.name, {})[task] = {"chance": hit["intelligence"] / 100, "basis": "Artificial Analysis Intelligence Index",
                                                    "evidence": {}}
    return out


def scores(measures: dict[str, dict[str, dict]], scale: str, anchor: dict[str, float] | None = None) -> dict[str, dict[str, float]]:
    """1-5 scores (one decimal) per model and task type from each model's chance on the reference task. ``anchor``: with
    scale absolute, the chance that scores 5 per task type (a task type without one uses the best model's)."""
    out: dict[str, dict[str, float]] = {}
    for task in {t for per in measures.values() for t in per}:
        values = {model: per[task]["chance"] for model, per in measures.items() if task in per}
        top, low = max(values.values()), min(values.values())
        five = (anchor or {}).get(task, top) if scale == "absolute" else top
        for model, v in values.items():
            s = (1 + 4 * (v - low) / (top - low) if top > low else 5.0) if scale == "range" else (5 * v / five if five else 1.0)
            # 1 is the floor (any model may take level-1 work); 5 the ceiling (a model above the anchor meets every level)
            out.setdefault(model, {})[task] = round(min(max(1.0, s), 5.0), 1)
    return out


def refresh_quality(cfg: Config, epoch: dict | None = None, aa: dict | None = None, aa_key: str | None = None) -> dict:
    """The live quality section: scores for every configured chat model the selected source covers, how each was
    worked out, and which models it doesn't cover (`auto` won't pick them for that task type unless they have a
    quality_override)."""
    src = cfg.quality.source
    now = datetime.now(UTC).isoformat(timespec="seconds")
    status: dict = {"ok": True, "fetched_at": now}
    measures: dict = {}
    try:
        if src == "epoch":
            measures = epoch_measures(cfg, epoch if epoch is not None else fetch_epoch(), cfg.routing.effort_by_floor.get(5))
        elif src == "artificial_analysis":
            if aa is None and not aa_key:
                raise ValueError("set ARTIFICIAL_ANALYSIS_API_KEY (a free key from artificialanalysis.ai/data-api)")
            measures = aa_measures(cfg, aa if aa is not None else fetch_artificial_analysis(aa_key or ""))
    except Exception as e:  # noqa: BLE001 (a data source that misbehaves must not stop prices refreshing; it's reported)
        status = {"ok": False, "fetched_at": now, "error": f"{type(e).__name__}: {e}"[:300]}
    chat = [m.name for m in cfg.models if m.kind == "chat" and not m.name.startswith("mock-")]
    q = cfg.quality
    return {"updated": now, "source": src, "scale": q.scale, "anchor": dict(q.anchor) if q.scale == "absolute" else None,
            "status": status, "attribution": ATTRIBUTION.get(src),
            "reference": dict(q.reference), "scores": scores(measures, q.scale, q.anchor) if src != "config" else {},
            "measures": measures, "missing": sorted(n for n in chat if n not in measures) if src != "config" else []}


# ------------------------------------------------------------------ keeping them live (the worker)

def _signature(cfg: Config) -> tuple:
    """What the scores depend on in the configuration: when it changes, they're worked out again right away."""
    q = cfg.quality
    return (q.source, q.scale, tuple(sorted(q.anchor.items())), tuple(sorted(q.reference.items())), tuple(sorted((k, tuple(v)) for k, v in q.evidence.items())),
            tuple(sorted((k, tuple(v)) for k, v in q.fallback.items())),
            tuple(sorted((m.name, m.benchmark_name or "", tuple(sorted(model_keys(m)))) for m in cfg.models if m.kind == "chat")))


def score_changes(old: dict, new: dict) -> list[str]:
    """Scores that moved by at least 0.1: "claude-opus coding 4.2 -> 4.4"."""
    out = []
    for model in sorted(set(old) | set(new)):
        for task in sorted(set(old.get(model, {})) | set(new.get(model, {}))):
            a, b = old.get(model, {}).get(task), new.get(model, {}).get(task)
            if a is None or b is None or abs(a - b) >= 0.1:
                out.append(f"{model} {task} {a if a is not None else '–'} -> {b if b is not None else '–'}")
    return out


async def refresh_quality_once(cfg: Config, store, state, fetch=refresh_quality) -> dict:
    """Work the scores out again, store them with the live catalog (gateways reload), and say what moved. A source
    that can't be reached keeps the last scores and raises an alert."""
    import asyncio
    import os

    from . import alerts
    from .configstore import current_token

    q = await asyncio.to_thread(fetch, cfg, None, None, os.environ.get("ARTIFICIAL_ANALYSIS_API_KEY"))
    doc = await store.get_setting("prices") or {}
    if "prices" not in doc:
        return {**q["status"], "stored": False, "why": "no live prices stored yet; the price refresh creates them first"}
    old = doc.get("quality") or {}
    if not q["status"]["ok"] and old.get("scores") and old.get("source") == q["source"]:
        q = {**old, "status": q["status"]}  # keep the last good scores
        if await alerts._once(state, f"alert:quality-source:{q['source']}", 6 * 3600):
            await alerts.notify(f"Couldn't refresh quality scores from {q['source']} ({q['status'].get('error')}); routing keeps the last ones.",
                                state=state, title="Quality scores not refreshed", level="warning", page="models")
    same_basis = all(old.get(k) == q.get(k) for k in ("source", "scale", "anchor", "reference"))
    moved = score_changes(old.get("scores") or {}, q.get("scores") or {}) if old and same_basis else []
    if not old or moved or not same_basis or old.get("measures") != q.get("measures") or old.get("missing") != q.get("missing"):
        doc["quality"] = q
        await store.set_setting("prices", doc)
        await state.set_config_token(await current_token(store, doc))
        if moved:
            shown = "; ".join(moved[:10]) + (f"; and {len(moved) - 10} more" if len(moved) > 10 else "")
            await alerts.notify(f"Quality scores changed ({len(moved)}): {shown}", state=state, title="Quality scores changed",
                                level="info", page="models")
    return {**q["status"], "stored": True, "changed": moved, "missing": q.get("missing", []), "scored": len(q.get("scores") or {})}


class QualityRefresher:
    """Calls ``refresh_quality_once`` every ``quality.refresh_hours``, and right away when the quality settings change."""

    def __init__(self) -> None:
        self.last = 0.0
        self.signature: tuple | None = None

    async def maybe(self, cfg: Config, store, state, now: float, fetch=refresh_quality) -> dict | None:
        sig = _signature(cfg)
        if self.last and sig == self.signature and now - self.last < cfg.quality.refresh_hours * 3600:
            return None
        self.last, self.signature = now, sig
        try:
            return await refresh_quality_once(cfg, store, state, fetch)
        except Exception:  # noqa: BLE001 (a bad refresh must not stop the worker; scores stay as they were)
            import logging

            logging.getLogger("meritorder.quality").exception("quality refresh failed")
            return None
