"""Expected total cost to finish a task, and the chance of success it rests on (decision 62, docs/03-routing.md#expected-total-cost).

With ``routing.model_choice: expected_cost`` an `auto` task goes to the model with the lowest expected cost to finish it:

    E(m) = try(m) + (1 - P(m)) x (H x stakes + escalation(m))

``try(m)`` is the route's cost for the rest of the session as the router prices it (cache rebuild included). ``H`` is the
person's cost of a failed attempt at the top of the scale (``human_cost_usd``, per team), and ``stakes`` the share of it
this request puts at risk (``stakes``, decision 67: a third for each level below 5, less under 8,000 tokens, all of it
after a correction). ``escalation(m)`` is what finishing costs after m
failed: the best model to move up to, from a cold cache, with its own chance of failing, as long as the task may still
escalate (``routing.cascade.max_escalations``). After the last allowed attempt a failure costs only H.

``P(m)``, the chance that m's answer needs no redo, combines, strongest last:

1. the benchmark prior: the model's chance at the task's difficulty on Epoch's scale (``config.apply_chances``, a fixed
   bar since decision 58), read between levels in log-odds;
2. the difficulty: the level the rules gave (classifier.py), or, with the learned router routing, its difficulty and how
   unsure it is (``learned.Prediction``), which widens the prior;
3. the fitted layer (``SuccessModel``, opt-in with ``routing.success.mode``): a calibration of the prior to what this
   organization's outcomes count as success, each model's own ability on its work, and offsets per team, repository and
   project, plus the triage features of the task's opening prompt (hashed word triples and the paths it mentions). Every
   part is shrunk towards the prior by a normal prior, so a model, team or repository with few outcomes keeps the prior.

Inside a task, ``posterior`` moves the task model's chance by what its tool results show (outcomes.py): failing tests and
builds after edits, fix loops, error results, a correction, and the scope growing. ``escalate`` compares continuing with
moving up, counting the cache rebuild there.

Nothing here holds content: repositories and projects are keyed hashes of 8 hex characters, words and paths hashed buckets.
"""

from __future__ import annotations

import array
import base64
import hashlib
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .learned import TASKS, sigmoid
from .outcomes import path_kind

SETTING = "success_model"  # the fitted layer's document (large; read when its version changes)
VERSION_SETTING = "success_model_version"  # its version and whether it's in use (small; read on every configuration poll)
FORMAT = 1
TRIAGE_BITS = 14
TRIAGE_BUCKETS = 1 << TRIAGE_BITS
OPENING = 1200  # characters of the opening prompt the triage features read
# Normal priors on the fitted layer, as precisions (in log-odds): a part with fewer outcomes than about this many keeps
# close to the prior. The calibration's slope is held near 1 and its intercept near 0 (the benchmark prior as it is).
PRIOR_GROUP = 20.0
PRIOR_TEXT = 8.0
PRIOR_CAL = 30.0
SWEEPS = 12
HOLDOUT = 0.2
MIN_EXAMPLES = 200
# In-task evidence, in log-odds of the task's model succeeding. Priors set from the outcome labels' weights (a signal with
# weight w at a task's end means it failed with chance about w); Reports -> Learned router shows how well they hold.
EVIDENCE = {"tests_failing": -0.6, "build_broken": -0.4, "fails_in_row": -0.3, "fix_loop": -0.8, "errors_in_row": -0.15,
            "correction": -1.4, "tests_green": 0.5, "files": -0.08, "core_edits": -0.05}
MAX_IN_ROW = 6
SCOPE_FILES = 5  # files touched beyond this many count against the task's model
_WORD = re.compile(rb"[a-z0-9_]+|[\x80-\xff]+")
_PATH = re.compile(r"(?<![\w/.-])((?:[\w.-]+/)+[\w.-]+|[\w-]+\.(?:py|ts|tsx|js|jsx|go|rs|java|kt|rb|php|c|h|cc|cpp|cs|swift|"
                   r"md|yaml|yml|json|toml|sql|sh|tf|proto|css|html))(?![\w/])")
_CWD = (re.compile(r"(?:Primary working directory|Working directory|working directory is|cwd):\s*`?([^\s`<]+)", re.I),
        re.compile(r"<cwd>\s*([^<\s]+)\s*</cwd>"))



# ------------------------------------------------------------------ context: repository, project, opening prompt


def _keyed(value: str, seed: bytes) -> str:
    return hashlib.blake2b(value.strip().encode("utf-8", "replace"), digest_size=4, key=seed or b"meritorder").hexdigest()


def repo_hash(body: dict, seed: bytes) -> str | None:
    """The repository the client works in, as a keyed hash of 8 hex characters: Claude Code names its working directory in
    the system prompt, Codex CLI in an <environment_context> message. Never the path itself."""
    sysp = body.get("system")
    texts = [sysp if isinstance(sysp, str) else " ".join(b.get("text") or "" for b in sysp or [] if isinstance(b, dict))]
    for m in (body.get("messages") or [])[:4]:
        c = m.get("content") if isinstance(m, dict) else None
        texts.append(c if isinstance(c, str) else " ".join(b.get("text") or "" for b in c or [] if isinstance(b, dict)))
    for text in texts:
        head = (text or "")[:20000]
        for rx in _CWD:
            m = rx.search(head)
            if m:
                return _keyed("repo:" + m.group(1).rstrip("/."), seed)
    return None


def project_hash(tags: dict[str, str], seed: bytes) -> str | None:
    """The client's ``project`` tag (X-MeritOrder-Tags), as a keyed hash."""
    p = (tags or {}).get("project")
    return _keyed("project:" + p, seed) if p else None


def triage_buckets(text: str, seed: bytes) -> list[int]:
    """The opening prompt's triage features as sorted buckets: its word triples (the learned router already has the words
    and pairs) and the paths it mentions (each one's kind, extension and directories). Keyed hashes, like learned.py's."""
    text = (text or "")[:OPENING]
    toks = _WORD.findall(text.lower().encode("utf-8", "replace"))
    keys = {b"3:" + b" ".join(toks[i:i + 3]) for i in range(len(toks) - 2)}
    for p in _PATH.findall(text)[:20]:
        parts = p.lower().split("/")
        keys.add(f"path-kind:{path_kind(p)}".encode())
        if "." in parts[-1]:
            keys.add(f"path-ext:{parts[-1].rsplit('.', 1)[1]}".encode())
        keys.update(f"path-dir:{d}".encode() for d in parts[:-1][:4])
    if not keys:
        return []
    mask = TRIAGE_BUCKETS - 1
    return sorted({int.from_bytes(hashlib.blake2b(k, digest_size=4, key=seed or b"meritorder").digest(), "little") & mask for k in keys})


# ------------------------------------------------------------------ the benchmark prior


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def prior_logit(chances: dict[int, float] | None, difficulty: float) -> float | None:
    """A model's chance at this difficulty (in level units) in log-odds, read between its chances at whole levels."""
    if not chances:
        return None
    d = min(max(difficulty, 1.0), 5.0)
    lo = min(int(math.floor(d)), 4)
    a, b = chances.get(lo), chances.get(lo + 1)
    if a is None or b is None:
        near = chances.get(int(round(d)))
        return None if near is None else _logit(near)
    t = d - lo
    return (1 - t) * _logit(a) + t * _logit(b)


@dataclass
class Difficulty:
    """How hard the task is, in level units, and how sure that is."""

    value: float
    spread: float = 0.0
    source: str = "the rules"

    def __str__(self) -> str:
        return f"{self.value:.1f}" + (f" ± {self.spread:.1f}" if self.spread >= 0.05 else "") + f" ({self.source})"


@dataclass
class Chance:
    """One model's chance of success on a task, and what it's made of (log-odds parts, for the routing reason)."""

    model: str
    p: float
    prior: float
    parts: list[tuple[str, float]] = field(default_factory=list)
    fitted: float | None = None  # the fitted layer's chance when it predicted (used, or recorded in shadow mode)
    estimated: bool = True  # False: no chance estimate, taken as a model exactly at the level

    def why(self) -> str:
        moved = ", ".join(f"{n} {v:+.1f}" for n, v in self.parts if abs(v) >= 0.05)
        return f"{self.p:.0%}" + (f" (benchmark {self.prior:.0%}; {moved})" if moved else " (benchmark)" if self.estimated else " (no estimate)")


# ------------------------------------------------------------------ the fitted layer


def _floats(values) -> str:
    return base64.b64encode(array.array("f", values).tobytes()).decode()


def _unfloats(text: str, n: int) -> array.array:
    a = array.array("f")
    a.frombytes(base64.b64decode(text))
    if len(a) != n:
        raise ValueError(f"expected {n} values, found {len(a)}")
    return a


@dataclass
class Example:
    """One task start's outcome, as the fitted layer learns from it."""

    task: str
    model: str
    prior: float  # the benchmark prior's log-odds for the served model at the task's difficulty
    y: float  # how sure we are it succeeded (1 - the outcome signals' combined weight)
    team: str = ""
    repo: str = ""
    project: str = ""
    text: list[int] = field(default_factory=list)
    ts: float = 0.0

    def groups(self) -> list[str]:
        g = [f"model:{self.model}"]
        if self.team:
            g.append(f"team:{self.team}")
        if self.repo:
            g.append(f"repo:{self.repo}")
        if self.project:
            g.append(f"project:{self.project}")
        return g


class SuccessModel:
    """The fitted layer: per task type, a calibration of the benchmark prior (intercept and slope), offsets by model, team,
    repository and project, and weights on the opening prompt's triage buckets."""

    def __init__(self, doc: dict, version: int = 0):
        if doc.get("format") != FORMAT:
            raise ValueError(f"unknown success model format {doc.get('format')!r}")
        self.doc, self.version = doc, version
        self.fv: str = doc.get("feature_version", "")
        self.trained_at = datetime.fromisoformat(doc["trained_at"])
        self.examples = int(doc.get("examples") or 0)
        self.better = bool(doc.get("better"))
        self.cal = {t: (float(v[0]), float(v[1])) for t, v in doc["calibration"].items()}
        self.groups = {t: {k: float(w) for k, w in g.items()} for t, g in doc["groups"].items()}
        self.text = {t: _unfloats(w, TRIAGE_BUCKETS) for t, w in doc["text"].items()}

    def logit(self, ex: Example) -> tuple[float, list[tuple[str, float]]]:
        a, b = self.cal.get(ex.task, (0.0, 1.0))
        parts = [("calibration", a + (b - 1) * ex.prior)]
        g = self.groups.get(ex.task, {})
        for key in ex.groups():
            w = g.get(key)
            if w is not None:
                parts.append((key.split(":", 1)[0] if not key.startswith("model:") else "model ability", w))
        tw = self.text.get(ex.task)
        if tw is not None and ex.text:
            parts.append(("opening prompt", sum(map(tw.__getitem__, ex.text)) / math.sqrt(len(ex.text))))
        return ex.prior + sum(v for _, v in parts), parts


def _fit(examples: list[Example], sweeps: int = SWEEPS) -> dict:
    """Coordinate-wise Newton steps on the penalized log-likelihood (soft labels): the calibration per task type, then each
    group offset, then each triage bucket, ``sweeps`` times. Pure Python, about 1 s per 10,000 task starts on a laptop."""
    cal = {t: [0.0, 1.0] for t in TASKS}
    groups: dict[str, dict[str, float]] = {t: {} for t in TASKS}
    text = {t: array.array("f", [0.0]) * TRIAGE_BUCKETS for t in TASKS}
    z = [e.prior for e in examples]
    by_group: dict[tuple[str, str], list[int]] = defaultdict(list)
    by_bucket: dict[tuple[str, int], list[int]] = defaultdict(list)
    norm = [1.0 / math.sqrt(len(e.text)) if e.text else 0.0 for e in examples]
    for i, e in enumerate(examples):
        for g in e.groups():
            by_group[(e.task, g)].append(i)
        for j in e.text:
            by_bucket[(e.task, j)].append(i)
    by_bucket = {k: v for k, v in by_bucket.items() if len(v) >= 3}  # a bucket in one or two requests says nothing
    by_task: dict[str, list[int]] = defaultdict(list)
    for i, e in enumerate(examples):
        by_task[e.task].append(i)
    for _ in range(sweeps):
        for task, idx in by_task.items():
            a, b = cal[task]
            for _step in range(2):  # Newton on (a, b), with priors at (0, 1)
                ga, gb, haa, hab, hbb = -PRIOR_CAL * a, -PRIOR_CAL * (b - 1), PRIOR_CAL, 0.0, PRIOR_CAL
                for i in idx:
                    p = sigmoid(z[i])
                    r, w, x = examples[i].y - p, p * (1 - p), examples[i].prior
                    ga += r
                    gb += r * x
                    haa += w
                    hab += w * x
                    hbb += w * x * x
                det = haa * hbb - hab * hab
                if det <= 1e-12:
                    break
                da, db = (hbb * ga - hab * gb) / det, (haa * gb - hab * ga) / det
                for i in idx:
                    z[i] += da + db * examples[i].prior
                a, b = a + da, b + db
            cal[task] = [a, b]
        for (task, g), idx in by_group.items():
            w0 = groups[task].get(g, 0.0)
            grad, hess = -PRIOR_GROUP * w0, PRIOR_GROUP
            for i in idx:
                p = sigmoid(z[i])
                grad += examples[i].y - p
                hess += p * (1 - p)
            step = grad / hess
            groups[task][g] = w0 + step
            for i in idx:
                z[i] += step
        for (task, j), idx in by_bucket.items():
            w0 = text[task][j]
            grad, hess = -PRIOR_TEXT * w0, PRIOR_TEXT
            for i in idx:
                p = sigmoid(z[i])
                grad += norm[i] * (examples[i].y - p)
                hess += norm[i] * norm[i] * p * (1 - p)
            step = grad / hess
            text[task][j] = w0 + step
            for i in idx:
                z[i] += step * norm[i]
    return {"calibration": {t: [round(a, 5), round(b, 5)] for t, (a, b) in cal.items()},
            "groups": {t: {k: round(v, 5) for k, v in sorted(g.items()) if abs(v) >= 1e-4} for t, g in groups.items()},
            "text": {t: _floats(w) for t, w in text.items()}}


def brier(examples: list[Example], probs: list[float]) -> float:
    return sum((p - e.y) ** 2 for e, p in zip(examples, probs)) / max(len(examples), 1)


def train(examples: list[Example], fv: str, now: datetime | None = None, min_examples: int = MIN_EXAMPLES) -> dict:
    """Fit on the oldest four fifths, compare with the benchmark prior alone on the newest fifth (Brier score, paired), and
    fit again on everything. Used (``better``) only if it beat the prior by more than twice the standard error."""
    now = now or datetime.now(UTC)
    examples = sorted(examples, key=lambda e: e.ts)
    base = {"examples": len(examples), "mean_success": round(sum(e.y for e in examples) / len(examples), 4) if examples else None}
    if len(examples) < min_examples:
        return {"doc": None, "better": False, "metrics": base,
                "why": f"{len(examples)} task starts with an outcome to learn from; the success model needs at least {min_examples}"}
    cut = int(len(examples) * (1 - HOLDOUT))
    head, tail = examples[:cut], examples[cut:]
    probe = SuccessModel({"format": FORMAT, "trained_at": now.isoformat(), **_fit(head)})
    fitted = [sigmoid(probe.logit(e)[0]) for e in tail]
    prior = [sigmoid(e.prior) for e in tail]
    diffs = [(p - e.y) ** 2 - (q - e.y) ** 2 for e, p, q in zip(tail, fitted, prior)]
    mean = sum(diffs) / len(diffs)
    se = math.sqrt(sum((d - mean) ** 2 for d in diffs) / max(len(diffs) - 1, 1) / len(diffs))
    better = mean < -2 * se
    metrics = {**base, "holdout": {"examples": len(tail), "brier": round(brier(tail, fitted), 5), "prior_brier": round(brier(tail, prior), 5),
                                   "difference": round(mean, 5), "se": round(se, 5)}, "better": better}
    doc = {"format": FORMAT, "feature_version": fv, "trained_at": now.isoformat(), "examples": len(examples), "better": better,
           "metrics": metrics, **_fit(examples)}
    why = None if better else (f"it predicted the newest {len(tail):,} task starts no better than the benchmark prior, beyond chance "
                               f"(Brier {metrics['holdout']['brier']:.4f} against {metrics['holdout']['prior_brier']:.4f}), so it isn't used")
    return {"doc": doc, "better": better, "metrics": metrics, "why": why}


# ------------------------------------------------------------------ the predictor


@dataclass
class TaskContext:
    """What the predictor knows about a task besides its difficulty: hashes and buckets, never content."""

    team: str = ""
    repo: str | None = None
    project: str | None = None
    triage: list[int] = field(default_factory=list)

    def record(self) -> dict:
        return {"repo": self.repo, "project": self.project}


class Predictor:
    """P(success | model, task): the benchmark prior at the task's difficulty, widened by how unsure the difficulty is, and
    the fitted layer where it's on. Lives across configuration reloads, like the learned router."""

    def __init__(self, model: SuccessModel | None = None, error: str | None = None):
        self.model = model
        self.error = error

    def chance(self, m: Any, task: str, d: Difficulty, ctx: TaskContext | None, mode: str, just: float | None) -> Chance:
        """``just``: the chance of a model exactly at the level (cfg.threshold_chance), for a model without estimates."""
        z0 = prior_logit((m.chances or {}).get(task), d.value)
        if z0 is None:
            p = just if just is not None and m.q(task) >= round(d.value) else (just or 0.5) * 0.5
            return Chance(m.name, p, p, estimated=False)
        parts: list[tuple[str, float]] = []
        z = z0
        if d.spread > 0:
            # The chance averaged over the difficulty's spread: the probit approximation, with the prior's slope per level.
            hi, lo = prior_logit(m.chances[task], d.value + 0.5), prior_logit(m.chances[task], d.value - 0.5)
            k = abs((lo or z0) - (hi or z0))
            z = z0 / math.sqrt(1 + math.pi * (k * d.spread) ** 2 / 8)
            if abs(z - z0) >= 0.05:
                parts.append(("unsure difficulty", z - z0))
        prior = sigmoid(z0)
        out = Chance(m.name, sigmoid(z), prior, parts)
        fm = self.model
        if fm is not None and mode != "off" and ctx is not None:
            fz, fparts = fm.logit(Example(task, m.name, z, 1.0, ctx.team, ctx.repo or "", ctx.project or "", ctx.triage))
            out.fitted = sigmoid(fz)
            if mode == "route" and fm.better:
                out.p, out.parts = out.fitted, parts + fparts
        return out


# ------------------------------------------------------------------ the objective


@dataclass
class Option:
    """A candidate as the objective sees it: its try cost from here (the router's estimate), the cost from a cold cache
    (what moving to it later would cost), and its chance."""

    key: Any
    model: str
    cost: float
    cold: float
    p: float


@dataclass
class Value:
    """An option's expected total cost to finish the task (``total``), the part of it that's tokens (``tokens``), and
    where it escalates after a failure (None: nowhere, or no escalation left)."""

    total: float
    tokens: float
    next: Option | None = None


# What a failure puts at stake (decision 67): the person's cost H is what a failed attempt at the top of the scale costs (a
# level-5 task, a full agent request). Each level below 5 puts a third as much at stake, and a request smaller than
# STAKES_TOKENS proportionally less. After a correction (the person saying the last answer was wrong) it counts as level 5.
STAKES_LEVEL = 1 / 3
STAKES_TOKENS = 8_000


@dataclass
class Stakes:
    """The share of the person's cost a failure of this request puts at stake, and what it's made of (for the reason)."""

    share: float
    level: int  # the level it's counted at: the request's, or 5 after a correction
    tokens: int
    corrected: bool = False

    def why(self) -> str:
        """'1/9 at level 3, 2,000 of 8,000 tokens', 'level 5 after a correction', or 'the full cost'."""
        parts = []
        if self.corrected:
            parts.append("level 5 after a correction")
        elif self.level < 5:
            parts.append(f"1/{round(STAKES_LEVEL ** (self.level - 5)):g} at level {self.level}")
        if self.tokens < STAKES_TOKENS:
            parts.append(f"{self.tokens:,} of {STAKES_TOKENS:,} tokens")
        return ", ".join(parts) or "the full cost"


def stakes(level: int, tokens: int, corrected: bool = False) -> Stakes:
    """What a failed attempt puts at stake, as a share of the person's cost: ``STAKES_LEVEL ** (5 - level)`` times the
    request's size, up to STAKES_TOKENS; level 5 after a correction.

    Why: a re-asked question costs a person seconds, a failed agent task minutes of review and a redo; SWE-bench Verified's
    annotated times to fix grow about fourfold from one bucket to the next (under 15 minutes, up to an hour, up to four).
    And the benchmark prior is least informative where the stakes are lowest: its chances are FrontierSWE's, measured at
    level 5 and read down to level 1 at the same gaps between models, which overstates how often a cheap model fails easy
    work. A correction is evidence that the work matters more than its wording said."""
    lv = 5 if corrected else min(max(int(level), 1), 5)
    size = max(int(tokens), 0)
    return Stakes(STAKES_LEVEL ** (5 - lv) * min(1.0, size / STAKES_TOKENS), lv, size, corrected)


# How much tasks the rules put at one level differ in difficulty for every model alike, in log-odds: why a task one model
# failed is harder for the next attempt too (on its model or another). Failures are conditioned on over this spread.
SPREAD = 1.0
_GRID = [-3.0 + 0.5 * i for i in range(13)]
_W = [math.exp(-x * x / 2) for x in _GRID]


def _curve(p: float) -> list[float]:
    """An option's chance of success at each point of the spread, centred so that its mean is about ``p``."""
    zc = _logit(p) * math.sqrt(1 + math.pi * SPREAD ** 2 / 8)
    return [sigmoid(zc - SPREAD * x) for x in _GRID]


def plan(options: list[Option], human: float, escalations: int) -> dict[Any, Value]:
    """Each option's expected total cost to finish the task, and where it goes after a failure:

        V_k(m) = try(m) + (1 - P(m)) x (H + min over next e of V_{k-1}(e | m failed))

    with V_0(m) = try(m) + (1 - P(m)) x H: after the last attempt allowed, a failure costs the person and nothing more.
    The next attempt is the same model again (the person resends, or the agent carries on) or a stronger one (a different
    model with a higher chance), from a cold cache; its chance is conditioned on the failures before it (``SPREAD``)."""
    by_model: dict[str, Option] = {}
    for o in sorted(options, key=lambda o: o.cold):
        by_model.setdefault(o.model, o)  # each model's cheapest route from a cold cache: where an escalation goes
    cold = list(by_model.values())
    curves = {id(o): _curve(o.p) for o in [*options, *cold]}
    memo: dict[tuple, tuple[Value, Option] | None] = {}

    def value(o: Option, k: int, start: float, weights: list[float] | None, path: tuple) -> Value:
        c = curves[id(o)]
        p = o.p if weights is None else sum(w * x for w, x in zip(weights, c)) / max(sum(weights), 1e-12)
        if k <= 0:
            return Value(start + (1 - p) * human, start)
        key = (o.model, k, path)
        if key not in memo:
            after = [w * (1 - x) for w, x in zip(weights or _W, c)]
            nxt = [(value(e, k - 1, o.cost if e.model == o.model else e.cold, after, (*path, o.model)), e) for e in cold
                   if e.model == o.model or e.p > o.p]
            memo[key] = min(nxt, key=lambda x: x[0].total) if nxt else None
        best = memo[key]
        if best is None:
            return Value(start + (1 - p) * human, start)
        v, e = best
        return Value(start + (1 - p) * (human + v.total), start + (1 - p) * v.tokens, e)

    return {o.key: value(o, escalations, o.cost, None, ()) for o in options}


# ------------------------------------------------------------------ inside a task


def posterior(p0: float, state: dict | None) -> tuple[float, list[tuple[str, float]]]:
    """The task model's chance of success now, from its chance at the task's start and what its tool results have shown
    (outcomes.py's state), in log-odds steps."""
    s = state or {}
    parts: list[tuple[str, float]] = []
    if s.get("last_test") == "fail":
        parts.append(("tests failing after the last edit", EVIDENCE["tests_failing"]))
    elif s.get("last_test") == "pass":
        parts.append(("tests passing after the last edit", EVIDENCE["tests_green"]))
    if s.get("last_build") == "fail" and s.get("last_test") != "pass":
        parts.append(("build broken", EVIDENCE["build_broken"]))
    row = min(int(s.get("fails_in_row") or 0), MAX_IN_ROW)
    if row > 1:
        parts.append((f"{row} failed runs in a row", EVIDENCE["fails_in_row"] * (row - 1)))
    if (s.get("fix_loop") or 0) >= 3 and s.get("last_test") != "pass":
        parts.append(("a fix loop", EVIDENCE["fix_loop"]))
    errs = min(int(s.get("errors_in_row") or 0), MAX_IN_ROW)
    if errs > 1:
        parts.append((f"{errs} error results in a row", EVIDENCE["errors_in_row"] * (errs - 1)))
    if s.get("correction"):
        parts.append((f"corrected ({s['correction']})", EVIDENCE["correction"]))
    files = int(s.get("files") or 0)
    if files > SCOPE_FILES:
        parts.append((f"{files} files touched", EVIDENCE["files"] * min(files - SCOPE_FILES, 8)))
    core = int((s.get("kinds") or {}).get("core") or 0)
    if core > SCOPE_FILES:
        parts.append((f"{core} core files edited", EVIDENCE["core_edits"] * min(core - SCOPE_FILES, 8)))
    return sigmoid(_logit(p0) + sum(v for _, v in parts)), parts


def failures_in_row(state: dict | None) -> int:
    s = state or {}
    return max(int(s.get("fails_in_row") or 0), int(s.get("errors_in_row") or 0))


def continues(state: dict | None) -> bool:
    """A new human message that carries on the same task rather than starting another: it says the last answer was wrong,
    or the code was left failing (the last run after the last edit failed)."""
    s = state or {}
    return bool(s.get("correction")) or (bool(s.get("edits")) and (s.get("last_test") == "fail" or s.get("last_build") == "fail"))


def track(task: dict | None, outcome: dict | None, model: str, chance: float | None, predicted: dict | None,
          escalated: bool, **start: Any) -> tuple[dict | None, dict | None]:
    """The task's state and its tool results' state after a request was routed, as the gateway keeps them
    (Gateway.update_session) and the routing simulator does: a new task records its model's chance at its start (``p0``,
    the plan's when it chose this model), its expected cost and ``start`` (the spend counters it's counted from); an
    escalation counts the new model's chance from here and clears what the old model's evidence ran up (failures and
    errors in a row, the correction)."""
    if task is None:
        p0 = predicted["p"] if predicted and predicted.get("model") == model else chance
        return {"p0": p0, "expected": predicted.get("expected") if predicted else None, "escalations": 0, **start}, outcome
    if escalated:
        task = {**task, "p0": chance}
        if outcome:
            outcome = {**outcome, "fails_in_row": 0, "errors_in_row": 0, "correction": None}
    return task, outcome


def usd(x: float) -> str:
    if x <= 0:
        return "$0"
    return f"${x:,.{max(2, 1 - math.floor(math.log10(x)))}f}"


# ------------------------------------------------------------------ the report


def _rows_examples(rows: list[dict], cfg: Any, gave_up: set[str]) -> list[tuple[dict, Example]]:
    """Task starts with an outcome, as the fitted layer learns from them (learned_jobs.signals gives the outcome)."""
    from .learned_jobs import failed, signals

    out = []
    for r in rows:
        if r.get("task") not in TASKS or not r.get("served_model") or (r.get("status") or 0) >= 400:
            continue
        m = cfg.model(r["served_model"])
        level = r.get("level") or r.get("rules_level")
        z = prior_logit((m.chances or {}).get(r["task"]) if m else None, float(level or 3))
        if z is None:
            continue
        ctx = r.get("context") or {}
        out.append((r, Example(r["task"], r["served_model"], z, 1.0 - failed(signals(r, gave_up)), r.get("team") or "",
                               ctx.get("repo") or "", ctx.get("project") or "", list(r.get("triage_features") or []),
                               r["ts"].timestamp())))
    return out


def examples(rows: list[dict], cfg: Any) -> list[Example]:
    from .learned_jobs import abandoned

    return [e for _, e in _rows_examples(rows, cfg, abandoned(rows))]


def reliability(pairs: list[tuple[float, float]], bins: int = 10) -> list[dict]:
    """Predicted against seen, in equal-width bins of the prediction."""
    out = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        inside = [(p, y) for p, y in pairs if lo <= p < hi or (b == bins - 1 and p == 1.0)]
        if inside:
            out.append({"from": lo, "to": hi, "n": len(inside), "predicted": round(sum(p for p, _ in inside) / len(inside), 4),
                        "seen": round(sum(y for _, y in inside) / len(inside), 4)})
    return out


def report(rows: list[dict]) -> dict:
    """Reports -> Learned router, the expected total cost section: predicted against seen success (the chance recorded for
    the model that served each task start, a reliability table and Brier score), the runner-up's chance, escalations (how
    often, and whether the task ended without a failure signal after), and expected against actual cost per finished task."""
    from .learned_jobs import abandoned, failed, signals

    gave_up = abandoned(rows) if rows and "text_features" in rows[0] else set()  # read without the words: none
    pairs: list[tuple[float, float]] = []
    fitted: list[tuple[float, float]] = []
    by_model: dict[str, list[tuple[float, float]]] = defaultdict(list)
    costs: list[tuple[float, float]] = []
    esc = {"tasks": 0, "escalated": 0, "helped": 0, "by_reason": defaultdict(int), "extra_cost": 0.0}
    for r in rows:
        pred = r.get("predicted") if isinstance(r.get("predicted"), dict) else None
        if not pred or (r.get("status") or 0) >= 400:
            continue
        y = 1.0 - failed(signals(r, gave_up))
        p = pred.get("p")
        if p is not None:
            pairs.append((float(p), y))
            by_model[r.get("served_model") or "?"].append((float(p), y))
        if pred.get("fitted") is not None:
            fitted.append((float(pred["fitted"]), y))
        c = r.get("coding") if isinstance(r.get("coding"), dict) else {}
        esc["tasks"] += 1
        if c.get("escalations"):
            esc["escalated"] += 1
            esc["helped"] += int(y >= 0.5)
            esc["extra_cost"] += float(c.get("escalation_cost") or 0.0)
            for why in c.get("escalation_reasons") or []:
                esc["by_reason"][why] += 1
        if pred.get("expected") is not None and c.get("cost") is not None and c.get("ended"):
            costs.append((float(pred["expected"]), float(c["cost"])))

    def score(ps: list[tuple[float, float]]) -> dict:
        if not ps:
            return {"n": 0, "brier": None, "predicted": None, "seen": None}
        return {"n": len(ps), "brier": round(sum((p - y) ** 2 for p, y in ps) / len(ps), 5),
                "predicted": round(sum(p for p, _ in ps) / len(ps), 4), "seen": round(sum(y for _, y in ps) / len(ps), 4)}

    return {
        "predicted": {**score(pairs), "bins": reliability(pairs)},
        "fitted": {**score(fitted), "bins": reliability(fitted)},
        "by_model": [{"model": m, **score(ps)} for m, ps in sorted(by_model.items(), key=lambda kv: -len(kv[1]))],
        "escalations": {"tasks": esc["tasks"], "escalated": esc["escalated"],
                        "rate": round(esc["escalated"] / esc["tasks"], 4) if esc["tasks"] else None,
                        "helped": esc["helped"], "helped_rate": round(esc["helped"] / esc["escalated"], 4) if esc["escalated"] else None,
                        "extra_cost": round(esc["extra_cost"], 6), "by_reason": dict(esc["by_reason"])},
        "cost": {"tasks": len(costs), "expected": round(sum(e for e, _ in costs) / len(costs), 6) if costs else None,
                 "actual": round(sum(a for _, a in costs) / len(costs), 6) if costs else None},
    }
