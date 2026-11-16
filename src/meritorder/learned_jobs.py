"""Training the learned router, and the shadow report (decision 44, docs/03-routing.md#the-learned-router).

Training runs as a command (`meritorder router train`) or a worker job (``routing.learned.retrain_hours``), never on
the request path. It reads the route_features rows the gateways recorded (one per task start: hashed words, named
tags, the rules' level, the model that served it and that model's score), turns what followed each answer into
labels, fits learned.Model, checks it against the rules alone on the most recent fifth of the requests, and writes it
to the store as a new version. A model that doesn't predict those held-out outcomes better than the rules is kept but
not used. Gateways load the active version on their next configuration reload.

The labels are weak, and the weights below say how weak (docs/03-routing.md#what-it-learns-from):

* the client raised X-Route-Floor (or sent X-Route-Hint: smart) on the next request: it said the level was too low;
* the client sent the same request again after an answer: often a regenerated answer, sometimes a client's own retry;
* the person stopped the answer (the stream was cancelled): often a wrong direction, sometimes they'd read enough;
* the person left the session and asked something with mostly the same words in a new one within 15 minutes;
* a model refused: a policy decision more than a sign the work was too hard for it.

Errors and capacity failovers say nothing about the work and aren't labels. Epoch's scores (or eval results applied as
scores) are the models' side of the model: the served model's score is what a request's difficulty is compared with.

Judged labels (judged.py, decision 52, opt-in per team) are a sample of requests replayed on a stronger model and graded
by a judge: whether the answer the request got was good enough. Unlike the signals, they say so either way. They join the
outcome labels as observations of their own, with a weight from how well they agree with the outcome signals on the
requests that have both (``judged_agreement``), and a tag of their own in place of the client's (a judge looks at every
answer, so how often it notices a failure isn't the client's). The check against the rules is on outcome labels only:
judged labels have to help predict what people did.
"""

from __future__ import annotations

import array
import asyncio
import base64
import logging
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from .learned import (FEATURE_VERSION, KEY_ENV, KEY_SETTING, TASKS, TEXT_BUCKETS, Features, LearnedRouter, Model, detection,
                      floats, rules_hash, sigmoid)
from .state import claim, release

log = logging.getLogger("meritorder.learned")

# How much each outcome signal says the answer wasn't good enough (1: certainly). Combined as 1 - prod(1 - w).
SIGNALS = {"floor_raised": 1.0, "retried": 0.6, "interrupted": 0.4, "abandoned": 0.3,
           # Coding outcomes from the tool results that followed (outcomes.py, decision 59). A person saying it was wrong is
           # nearly as clear as a raised floor; tests failing at the task's end often mean the agent gave up, sometimes that
           # the person took over; a broken build and a fix loop are weaker (agents often get there in the end).
           "corrected": 0.7, "tests_failing": 0.5, "build_broken": 0.4, "fix_loop": 0.3}
REFUSAL = 0.2  # a refusing model's own observation: it failed, a little
ABANDON_MINUTES = 15
ABANDON_OVERLAP = 0.5  # share of hashed words the new session's first request has in common with the last one
PRIOR = 4.0  # precision of each weight's prior: a feature nobody has seen moves nothing, with a spread of 0.5 levels
MEMBERS = 6  # models in the ensemble, each trained on a resampled copy of the requests
SLOPE_PRIOR = 10.0  # precision of the slope's prior around what Epoch's ability scale implies
SLOPE_RANGE = (0.25, 4.0)
MIN_EXAMPLES = 200
TRAIN_REQUESTED = "learned_router:train"  # set by the console's "Train now", done by the worker
LAST_TRAINING = "learned_router_training"  # what the last training did, for the console
HOLDOUT = 0.2
EPOCHS = 30
MIN_EPOCHS = 5
RATE = 0.3
# Judged labels' weight before there's evidence of how well they agree with the outcome signals: the weakest signal's.
# The evidence counts as much as this prior once JUDGED_PSEUDO of the judged requests have an outcome signal.
JUDGED_PRIOR = SIGNALS["abandoned"]
JUDGED_PSEUDO = 10.0
JUDGED_TAG = "judged:pair"  # a detection tag (learned.DETECTION): moves how often a failure is seen, never the difficulty


@dataclass
class Obs:
    text: list[int]
    tags: list[str]
    task: str
    rules: int
    score: float
    y: float  # how sure we are it failed (0: no signal)
    w: float = 1.0
    ts: float = 0.0
    kind: str = "outcome"  # outcome (what followed the answer, or a refusal), or judged (judged.py)

    def __post_init__(self) -> None:
        self.norm = 1.0 / math.sqrt(len(self.text)) if self.text else 0.0
        self.work = [t for t in self.tags if not detection(t)]  # tags about the work: part of the difficulty
        self.notice = [t for t in self.tags if detection(t)]  # tags about the client: how often a failure is noticed


def _jaccard(a: list[int], b: list[int]) -> float:
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb)


def abandoned(rows: list[dict]) -> set[str]:
    """Requests after which the person seems to have given up: the last one of its conversation, followed within
    ABANDON_MINUTES by the same person's first request in a new session with mostly the same (hashed) words. Subagents
    and Claude Code's side requests are left out: their conversations end by design."""
    def ignored(r: dict) -> bool:
        return "subagent" in r["tags"] or "class:auxiliary" in r["tags"] or "class:compaction" in r["tags"]

    last: dict[str, dict] = {}
    for r in rows:
        last[r["thread"]] = r
    ends = {id(r) for r in last.values()}
    out: set[str] = set()
    by_user: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if not ignored(r):
            by_user[r["user_id"]].append(r)
    for seq in by_user.values():
        for i, r in enumerate(seq):
            if id(r) not in ends or (r.get("status") or 0) >= 400:
                continue
            for nxt in seq[i + 1:]:
                if (nxt["ts"] - r["ts"]).total_seconds() > ABANDON_MINUTES * 60:
                    break
                if nxt["session_id"] != r["session_id"]:
                    if _jaccard(r["text_features"], nxt["text_features"]) >= ABANDON_OVERLAP:
                        out.add(str(r["request_id"]))
                    break
    return out


def signals(row: dict, gave_up: set[str]) -> dict[str, bool]:
    """The outcome signals seen for a request, on the model that answered it."""
    status = row.get("status") or 0
    c = row.get("coding") if isinstance(row.get("coding"), dict) else {}
    green = c.get("last_test") == "pass"  # tests passing after the last edit outweigh a broken build or a loop on the way
    return {"floor_raised": row.get("floor_raised") is not None, "retried": bool(row.get("retried")) and status < 400,
            "interrupted": status == 499, "abandoned": str(row["request_id"]) in gave_up,
            "corrected": bool(c.get("correction")), "tests_failing": c.get("last_test") == "fail",
            "build_broken": c.get("last_build") == "fail" and not green, "fix_loop": (c.get("fix_loop") or 0) >= 3 and not green}


def failed(sig: dict[str, bool]) -> float:
    miss = 1.0
    for name, seen in sig.items():
        if seen:
            miss *= 1.0 - SIGNALS[name]
    return 1.0 - miss


def observations(rows: list[dict], gave_up: set[str] | None = None) -> list[Obs]:
    """What each request says: the model that answered it (unless it failed for reasons that aren't about the work) and
    every model that refused it. A model without a score (unmeasured, a mock) says nothing about difficulty. Judged labels
    are separate (judged_observations)."""
    gave_up = abandoned(rows) if gave_up is None else gave_up
    out: list[Obs] = []
    for r in rows:
        if r["task"] not in TASKS:
            continue
        ts = r["ts"].timestamp()
        for score in r.get("refused_scores") or []:
            if score and score > 0:
                out.append(Obs(r["text_features"], r["tags"], r["task"], r["rules_level"], float(score), 1.0, REFUSAL, ts))
        status = r.get("status") or 0
        if not r.get("served_score") or (status >= 400 and status != 499):
            continue
        out.append(Obs(r["text_features"], r["tags"], r["task"], r["rules_level"], float(r["served_score"]),
                       failed(signals(r, gave_up)), 1.0, ts))
    return out


def _judged(r: dict) -> dict | None:
    """The request's judged label, when a judge gave one on an answer whose model has a score."""
    j = r.get("judged")
    if not isinstance(j, dict) or j.get("outcome") != "judged" or not isinstance(j.get("good_enough"), bool):
        return None
    return j if r.get("served_score") and r["task"] in TASKS and (r.get("status") or 0) == 200 else None


def judged_observations(rows: list[dict], weight: float) -> list[Obs]:
    """Each judged label as an observation of the model that answered: failed (1) when the judge said its answer wasn't
    good enough given the stronger one's, else succeeded (0), at ``weight``. The request's client tags give way to
    JUDGED_TAG: a judge looks at every answer it's given, whoever the client was."""
    if weight <= 0:
        return []
    out = []
    for r in rows:
        j = _judged(r)
        if j is None:
            continue
        tags = [t for t in r["tags"] if not detection(t)] + [JUDGED_TAG]
        out.append(Obs(r["text_features"], tags, r["task"], r["rules_level"], float(r["served_score"]), 0.0 if j["good_enough"] else 1.0,
                       weight, r["ts"].timestamp(), "judged"))
    return out


def judged_agreement(rows: list[dict], gave_up: set[str]) -> dict:
    """How well judged labels agree with the outcome signals, on the requests that have a judged label (every one of them
    can have a signal too): of the answers with a failure signal, the share the judge also failed ("caught"); of the rest,
    the share it failed anyway ("false alarm"). Their difference (Youden's J) doesn't depend on how rare signals are; it's
    0 when the judge's verdicts are unrelated to what people did, and 1 when they match. "No signal" isn't a success, so a
    judge that catches failures nobody noticed looks worse than it is: J errs low.

    The judged labels' weight is J (never below 0), shrunk towards JUDGED_PRIOR while few judged answers have a signal:
    (with signal x J + JUDGED_PSEUDO x prior) / (with signal + JUDGED_PSEUDO). Signals count by their own weights."""
    n, sig, caught, quiet, alarms, failed_j = 0, 0.0, 0.0, 0.0, 0.0, 0
    for r in rows:
        j = _judged(r)
        if j is None:
            continue
        y = failed(signals(r, gave_up))
        bad = 0.0 if j["good_enough"] else 1.0
        n += 1
        failed_j += int(bad)
        sig += y
        caught += y * bad
        quiet += 1 - y
        alarms += (1 - y) * bad
    recall = caught / sig if sig else None
    alarm = alarms / quiet if quiet else None
    agree = recall - alarm if recall is not None and alarm is not None else None
    weight = (sig * max(agree or 0.0, 0.0) + JUDGED_PSEUDO * JUDGED_PRIOR) / (sig + JUDGED_PSEUDO)
    return {"labels": n, "not_good_enough": failed_j, "with_signal": round(sig, 3), "caught": None if recall is None else round(recall, 4),
            "false_alarm": None if alarm is None else round(alarm, 4), "agreement": None if agree is None else round(agree, 4),
            "weight": round(weight, 4) if n else None}


def epoch_slope(cfg: Any, task: str) -> float:
    """What Epoch's ability scale says about the slope: how fast a failure signal gets likelier per level a model is short
    of the work. The chance of success falls by about 0.6 in log-odds per level (config.apply_chances); the log-odds of
    failing rise by that times the chance itself."""
    slopes = []
    for m in cfg.models:
        ch = m.chances.get(task) or {}
        for lvl in range(1, 5):
            if lvl in ch and lvl + 1 in ch:
                slopes.append(_logit(ch[lvl]) - _logit(ch[lvl + 1]))
    s = sum(slopes) / len(slopes) if slopes else 0.6
    return max(SLOPE_RANGE[0], min(SLOPE_RANGE[1], s * cfg.threshold_chance.get(task, 0.7)))


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


@dataclass
class Fit:
    text_w: array.array
    tags_w: dict[str, float]
    slope: dict[str, float]
    intercept: dict[str, float]
    shift: dict[str, float] = field(default_factory=lambda: dict.fromkeys(TASKS, 0.0))
    epochs: int = 0

    def z(self, o: Obs) -> tuple[float, float]:
        d = (o.rules + self.shift[o.task] + sum(map(self.text_w.__getitem__, o.text)) * o.norm
             + sum(self.tags_w.get(t, 0.0) for t in o.work))
        return self.slope[o.task] * (d - o.score) + self.intercept[o.task] + sum(self.tags_w.get(t, 0.0) for t in o.notice), d

    def anchor(self, obs: list[Obs]) -> None:
        """Make the corrections average to zero over the training requests of each task type, moving the intercept to
        match so no predicted outcome changes. The labels can't tell a request's absolute difficulty from how often failure
        signals are noticed at all (both move the same odds), so the rules are taken to be right on average: what's
        learned is which requests are harder or easier than the others the rules gave the same level."""
        sums: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
        for o in obs:
            _, d = self.z(o)
            sums[o.task][0] += o.w * (d - o.rules)
            sums[o.task][1] += o.w
        for task, (total, weight) in sums.items():
            delta = -total / weight if weight else 0.0
            self.shift[task] += delta
            self.intercept[task] -= self.slope[task] * delta

    def logits(self, obs: list[Obs]) -> list[float]:
        return [self.z(o)[0] for o in obs]


def _xent(o: Obs, p: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return -o.w * (o.y * math.log(p) + (1 - o.y) * math.log(1 - p))


def compare(learned: list[float], rules: list[float], obs: list[Obs]) -> dict:
    """The two models' held-out cross-entropy, and whether the learned one is better by more than chance: its mean
    improvement per request, paired request by request, is more than twice its standard error."""
    weight = sum(o.w for o in obs) or 1.0
    diffs = [r - x for r, x in zip(rules, learned)]
    n = len(diffs)
    mean = sum(diffs) / n if n else 0.0
    se = math.sqrt(sum((x - mean) ** 2 for x in diffs) / (n - 1) / n) if n > 1 else math.inf
    return {"loss": round(sum(learned) / weight, 5), "rules_loss": round(sum(rules) / weight, 5),
            "improvement": round(mean, 6), "error": round(se, 6), "better": n > 1 and mean > 2 * se}


def fit(obs: list[Obs], slope0: dict[str, float], features: bool = True, epochs: int = EPOCHS, seed: int = 0) -> Fit:
    """Maximum a posteriori fit in pure Python: each epoch, one AdaGrad pass over the observations for the feature weights,
    then Newton's method for each task type's slope and intercept. Each weight has a normal prior at 0 (precision PRIOR),
    applied lazily where the weight is used so that over an epoch it counts once; the slope has a prior at what Epoch's
    scale implies; the intercept starts at the base rate. ``features=False`` fits the rules alone (only the slope and
    intercept), the baseline a model has to beat."""
    n_text: Counter = Counter()
    n_tags: Counter = Counter()
    by_task: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for o in obs:
        if features:
            n_text.update(o.text)
            n_tags.update(o.tags)
        by_task[o.task][0] += o.w * o.y
        by_task[o.task][1] += o.w
    intercept = {t: _logit(max(by_task[t][0], 0.5) / max(by_task[t][1], 1.0)) for t in TASKS}
    c0 = dict(intercept)
    slope = {t: slope0.get(t, 0.42) for t in TASKS}
    f = Fit(array.array("d", bytes(8 * TEXT_BUCKETS)), {}, slope, intercept)
    g_text = array.array("d", bytes(8 * TEXT_BUCKETS))
    g_tags: dict[str, float] = defaultdict(float)
    rng = random.Random(seed)
    order = list(range(len(obs)))
    bt, btag = f.text_w, f.tags_w
    prev = math.inf
    if not features:
        for _ in range(3):
            _newton(f, obs, slope0, c0)
        return f
    for epoch in range(epochs):
        rng.shuffle(order)
        total = 0.0
        for i in order:
            o = obs[i]
            norm = o.norm
            d = o.rules + f.shift[o.task] + sum(map(bt.__getitem__, o.text)) * norm + sum(btag.get(t, 0.0) for t in o.work)
            a, c = slope[o.task], intercept[o.task]
            p = sigmoid(a * (d - o.score) + c + sum(btag.get(t, 0.0) for t in o.notice))
            pc = min(max(p, 1e-9), 1 - 1e-9)
            total -= o.w * (o.y * math.log(pc) + (1 - o.y) * math.log(1 - pc))
            g = o.w * (p - o.y)
            if features:
                gt = g * a * norm
                for j in o.text:
                    grad = gt + PRIOR * bt[j] / n_text[j]
                    g_text[j] += grad * grad
                    bt[j] -= RATE * grad / (math.sqrt(g_text[j]) + 1e-8)
                ga = g * a
                for t in o.tags:
                    w = btag.get(t, 0.0)
                    grad = (g if detection(t) else ga) + PRIOR * w / n_tags[t]
                    g_tags[t] += grad * grad
                    btag[t] = w - RATE * grad / (math.sqrt(g_tags[t]) + 1e-8)
        _newton(f, obs, slope0, c0)
        f.anchor(obs)
        f.epochs = epoch + 1
        if prev - total < 1e-4 * max(abs(prev), 1.0) and epoch >= MIN_EPOCHS:
            break
        prev = total
    return f


def _newton(f: Fit, obs: list[Obs], slope0: dict[str, float], c0: dict[str, float], steps: int = 5) -> None:
    """The slope and intercept of each task type, by Newton's method on all observations at once with the weights held:
    two numbers per task type, used by every request, which a per-request step would only learn slowly."""
    gaps: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
    for o in obs:
        _, d = f.z(o)
        gaps[o.task].append((d - o.score, o.y, o.w, sum(f.tags_w.get(t, 0.0) for t in o.notice)))
    for task, xs in gaps.items():
        a, c = f.slope[task], f.intercept[task]
        a0 = slope0.get(task, 0.42)
        for _ in range(steps):
            ga, gc = SLOPE_PRIOR * (a - a0), 0.01 * (c - c0[task])
            haa, hac, hcc = SLOPE_PRIOR, 0.0, 0.01
            for u, y, w, extra in xs:
                p = sigmoid(a * u + c + extra)
                r, h = w * (p - y), w * p * (1 - p)
                ga += r * u
                gc += r
                haa += h * u * u
                hac += h * u
                hcc += h
            det = haa * hcc - hac * hac
            if det <= 1e-12:
                break
            a = min(SLOPE_RANGE[1], max(SLOPE_RANGE[0], a - (hcc * ga - hac * gc) / det))
            c -= (haa * gc - hac * ga) / det
        f.slope[task], f.intercept[task] = a, c


def ensemble(obs: list[Obs], slope0: dict[str, float], seed: int = 0) -> list[Fit]:
    """MEMBERS fits, each on a different half of the observations, drawn without replacement. Where they disagree the
    evidence is thin: the spread of half-sample estimates around their mean is about the sampling spread of an estimate
    from all of them (half-sampling), at half the cost of fitting on resamples of the full set."""
    fits = []
    for k in range(MEMBERS):
        rng = random.Random(seed * 1000 + k)
        fits.append(fit(rng.sample(obs, k=max(1, len(obs) // 2)), slope0, seed=k))
    return fits


def ensemble_logits(fits: list[Fit], obs: list[Obs]) -> list[float]:
    """Each observation's log-odds of a failure signal under the members' mean difficulty, slope and intercept."""
    out = []
    n = len(fits)
    for o in obs:
        d = sum(f.z(o)[1] for f in fits) / n
        a = sum(f.slope[o.task] for f in fits) / n
        c = sum(f.intercept[o.task] + sum(f.tags_w.get(t, 0.0) for t in o.notice) for f in fits) / n
        out.append(a * (d - o.score) + c)
    return out


def held_out_losses(z: list[float], obs: list[Obs]) -> list[float]:
    """Each held-out observation's weighted cross-entropy, after moving each task type's log-odds by the one amount that
    fits these observations best. Both models get it, so neither wins by having guessed how often failures would be noticed
    in the newest requests: that rate drifts, and the levels don't depend on it (they're anchored on the rules)."""
    by_task: dict[str, list[int]] = defaultdict(list)
    for i, o in enumerate(obs):
        by_task[o.task].append(i)
    shift = [0.0] * len(obs)
    for idx in by_task.values():
        delta = 0.0
        for _ in range(20):
            g = h = 0.0
            for i in idx:
                p = sigmoid(z[i] + delta)
                g += obs[i].w * (p - obs[i].y)
                h += obs[i].w * p * (1 - p)
            if h <= 1e-12:
                break
            step = g / h
            delta -= step
            if abs(step) < 1e-9:
                break
        for i in idx:
            shift[i] = delta
    return [_xent(o, sigmoid(x + d)) for o, x, d in zip(obs, z, shift)]


def model_doc(fits: list[Fit], obs: list[Obs], cfg: Any, fv: str, now: datetime, metrics: dict) -> dict:
    unseen = array.array("B", [1]) * TEXT_BUCKETS
    tags: set[str] = set()
    for o in obs:
        for j in o.text:
            unseen[j] = 0
        tags.update(o.tags)
    return {"format": 1, "feature_version": fv, "trained_at": now.isoformat(), "rules_hash": rules_hash(cfg.routing.levels),
            "prior": PRIOR, "examples": len(obs), "metrics": metrics,
            "members": [{"text_w": floats(f.text_w), "tags_w": {t: round(w, 5) for t, w in sorted(f.tags_w.items())},
                         "slope": {t: round(v, 5) for t, v in f.slope.items()},
                         "intercept": {t: round(v, 5) for t, v in f.intercept.items()},
                         "shift": {t: round(v, 5) for t, v in f.shift.items()}} for f in fits],
            "unseen": base64.b64encode(unseen.tobytes()).decode(), "tags_seen": sorted(tags)}


def train(rows: list[dict], cfg: Any, fv: str, now: datetime | None = None, min_examples: int = MIN_EXAMPLES) -> dict:
    """Fit a model on these rows and check it on the newest HOLDOUT share of them against the rules alone. Returns
    ``doc`` (None when there isn't enough to train on), ``better`` (it predicts the held-out outcomes better than the
    rules: only then is it used) and ``metrics``. CPU-bound: callers run it in a thread."""
    now = now or datetime.now(UTC)
    gave_up = abandoned(rows)
    obs = sorted(observations(rows, gave_up), key=lambda o: o.ts)
    with_signal = sum(1 for o in obs if o.y > 0)
    agree = judged_agreement(rows, gave_up)
    base = {"requests": len(rows), "observations": len(obs), "with_signal": with_signal,
            "signals": dict(Counter(k for r in rows for k, v in signals(r, gave_up).items() if v)),
            "refusals": sum(len(r.get("refused_scores") or []) for r in rows), "judged": agree}
    if len(obs) < min_examples:
        return {"doc": None, "better": False, "metrics": base,
                "why": f"{len(obs)} requests with an outcome to learn from; training needs at least {min_examples}"}
    if with_signal == 0:
        return {"doc": None, "better": False, "metrics": base,
                "why": "no request has an outcome signal yet (no retries, raised floors, stopped answers or refusals)"}
    slope0 = {t: epoch_slope(cfg, t) for t in TASKS}
    cut = int(len(obs) * (1 - HOLDOUT))
    head, tail = obs[:cut], obs[cut:]
    # Judged labels train the learned model, at a weight from the older requests only; the rules alone have no use for them
    # (they'd only need an intercept of their own), and both are compared on the newest requests' outcome labels.
    cut_ts = tail[0].ts if tail else math.inf
    older = [r for r in rows if r["ts"].timestamp() < cut_ts]
    judged_head = judged_observations(older, judged_agreement(older, gave_up)["weight"] or 0.0)
    trial = ensemble(sorted(head + judged_head, key=lambda o: o.ts) if judged_head else head, slope0)
    rules_only = fit(head, slope0, features=False)
    held = compare(held_out_losses(ensemble_logits(trial, tail), tail), held_out_losses(rules_only.logits(tail), tail), tail)
    probe = Model(model_doc(trial, head, cfg, fv, now, {}), 0)
    moved: Counter = Counter()
    for o in tail:
        p = probe.predict(Features(o.task, o.rules, o.text, o.tags))
        if p.level != o.rules and p.confidence >= cfg.routing.learned.min_confidence:
            moved["up" if p.level > o.rules else "down"] += 1
    judged = judged_observations(rows, agree["weight"] or 0.0)
    everything = sorted(obs + judged, key=lambda o: o.ts) if judged else obs
    final = ensemble(everything, slope0, seed=1)
    n = len(final)
    metrics = {**base, "members": n, "slope_prior": {t: round(v, 4) for t, v in slope0.items()},
               "slope": {t: round(sum(f.slope[t] for f in final) / n, 4) for t in TASKS}, "epochs": max(f.epochs for f in final),
               "holdout": {"requests": len(tail), "with_signal": sum(1 for o in tail if o.y > 0), **held,
                           "would_raise": moved["up"], "would_lower": moved["down"]}}
    better = held["better"]
    metrics["better"] = better
    return {"doc": model_doc(final, everything, cfg, fv, now, metrics), "better": better, "metrics": metrics,
            "why": None if better else (f"it predicted the newest {len(tail):,} requests' outcomes no better than the rules alone, "
                                        f"beyond chance ({held['loss']:.4f} against {held['rules_loss']:.4f}), so it isn't used")}


async def feature_key(store) -> str | None:
    return os.environ.get(KEY_ENV) or await store.get_setting(KEY_SETTING)


async def train_and_save(cfg: Any, store, state, author: str, days: int | None = None, min_examples: int = MIN_EXAMPLES,
                         dry_run: bool = False, now: datetime | None = None) -> dict:
    """Train on the last ``days`` (routing.learned.train_days) of recorded requests and store the model as a new version,
    used at once if it beat the rules on the held-out requests. Gateways pick it up on their next configuration reload."""
    from .configstore import current_token

    now = now or datetime.now(UTC)
    days = days or cfg.routing.learned.train_days
    out: dict = {"version": None, "active": False, "metrics": {"requests": 0}, "days": days, "by": author, "at": now.isoformat(),
                 "dry_run": dry_run}
    key = await feature_key(store)
    if not key:
        out["why"] = "no request has been recorded yet: set routing.learned.mode to shadow and let traffic come in"
    else:
        fv = LearnedRouter(key=key).fv
        rows = await store.route_features(now - timedelta(days=days), fv=fv, limit=cfg.routing.learned.train_requests)
        result = await asyncio.to_thread(train, rows, cfg, fv, now, min_examples)
        out.update(metrics=result["metrics"], why=result["why"])
        if cfg.routing.success.mode != "off":
            out["success"] = await train_success(cfg, store, state, rows, fv, now, min_examples, dry_run)
        if result["doc"] is not None and not dry_run:
            version = await store.save_router_model(result["doc"], result["metrics"], author, active=result["better"])
            out.update(version=version, active=result["better"])
            if result["better"]:
                await state.set_config_token(await current_token(store, await store.get_setting("prices")))
            log.info("learned router: trained v%s on %s requests (%s)", version, result["metrics"]["requests"],
                     "in use" if result["better"] else result["why"])
    if not dry_run:
        await store.set_setting(LAST_TRAINING, out)  # the console shows it, whichever process trained
    return out


async def train_success(cfg: Any, store, state, rows: list[dict], fv: str, now: datetime, min_examples: int = MIN_EXAMPLES,
                        dry_run: bool = False) -> dict:
    """Fit the success predictor's layer (success.py, decision 62) on the same rows, and store it as a new version (in use in
    route mode only if it beat the benchmark prior on the newest fifth). Gateways pick it up on their next reload."""
    from . import success
    from .configstore import current_token

    exs = success.examples(rows, cfg)
    result = await asyncio.to_thread(success.train, exs, fv, now, min_examples)
    out = {"version": None, "better": result["better"], "metrics": result["metrics"], "why": result["why"]}
    if result["doc"] is not None and not dry_run:
        version = int(((await store.get_setting(success.VERSION_SETTING)) or {}).get("version") or 0) + 1
        await store.set_setting(success.SETTING, result["doc"])
        await store.set_setting(success.VERSION_SETTING, {"version": version, "better": result["better"], "at": now.isoformat(),
                                                          "metrics": result["metrics"]})
        out["version"] = version
        await state.set_config_token(await current_token(store, await store.get_setting("prices")))
        log.info("success predictor: trained v%s on %s task starts (%s)", version, len(exs),
                 "better than the benchmark prior" if result["better"] else result["why"])
    return out


TRAIN_LOCK = "lock:router-train"
TRAIN_LOCK_SECONDS = 3 * 3600  # longer than any training; released when it ends


class RouterTrainer:
    """The worker's job: delete request features older than keep_days, train when an admin asked for it on the console
    (TRAIN_REQUESTED), and retrain every retrain_hours when that's set."""

    def __init__(self) -> None:
        self.pruned_at = 0.0

    async def maybe(self, cfg: Any, store, state, now: datetime | None = None) -> dict | None:
        s = cfg.routing.learned
        now = now or datetime.now(UTC)
        if now.timestamp() - self.pruned_at >= 3600:  # whatever the mode: rows recorded before it was turned off go too
            n = await store.prune_route_features(now - timedelta(days=s.keep_days))
            self.pruned_at = now.timestamp()
            if n:
                log.info("learned router: deleted %d request features older than %d days", n, s.keep_days)
        asked = await state.get_json(TRAIN_REQUESTED)
        if s.mode == "off" and not asked:
            return None
        if not asked:
            if s.retrain_hours is None:
                return None
            latest = await store.router_models(1)
            if latest and (now - latest[0]["created_at"]).total_seconds() < s.retrain_hours * 3600:
                return None
        # One worker trains at a time: two that both saw the request (or a stale model) would each train and save a version.
        if not await claim(state, TRAIN_LOCK, TRAIN_LOCK_SECONDS):
            return None
        try:
            if asked:
                if not await state.get_json(TRAIN_REQUESTED):
                    return None  # another worker took the request and has trained already
                await state.set_json(TRAIN_REQUESTED, None)
                return await train_and_save(cfg, store, state, f"console:{asked.get('by') or 'admin'}", now=now)
            latest = await store.router_models(1)  # again, now that no one else can be saving one
            if latest and (now - latest[0]["created_at"]).total_seconds() < s.retrain_hours * 3600:
                return None
            return await train_and_save(cfg, store, state, "worker", now=now)
        finally:
            await release(state, TRAIN_LOCK)


# ------------------------------------------------------------------ the shadow report


def _usd(x: float | None) -> float | None:
    return None if x is None else round(x, 6)


async def status(cfg: Any, store, runtime: LearnedRouter | None, state=None, now: datetime | None = None) -> dict:
    """The learned router's settings, the model in use (and whether it's stale), its versions, and the last training."""
    now = now or datetime.now(UTC)
    s = cfg.routing.learned
    versions = await store.router_models(20)
    last = await store.get_setting(LAST_TRAINING)
    asked = await state.get_json(TRAIN_REQUESTED) if state is not None else None
    active = next((v for v in versions if v["active"]), None)
    if active is None and (in_use := await store.router_model_version()) is not None:
        active = await store.router_model(in_use)  # older than the newest 20 (retrains that didn't beat the rules)
    model = runtime.model if runtime else None
    stale = model.stale(cfg.routing.levels, runtime.fv, now, s.max_age_days) if model and runtime else None
    from .success import VERSION_SETTING

    fitted = await store.get_setting(VERSION_SETTING)
    return {"mode": s.mode, "min_confidence": s.min_confidence, "max_age_days": s.max_age_days, "retrain_hours": s.retrain_hours,
            "train_days": s.train_days, "train_requests": s.train_requests, "keep_days": s.keep_days, "feature_version": FEATURE_VERSION,
            "active": active["version"] if active else None,
            "loaded": model.version if model else None, "load_error": runtime.error if runtime else None,
            "trained_at": model.trained_at.isoformat() if model else None,
            "age_days": round((now - model.trained_at).total_seconds() / 86400, 1) if model else None,
            "stale": stale, "examples": model.examples if model else None,
            "metrics": active["metrics"] if active else None, "last_training": last, "train_requested": asked,
            "success": {"mode": cfg.routing.success.mode, "version": (fitted or {}).get("version"), "better": (fitted or {}).get("better"),
                        "trained_at": (fitted or {}).get("at"), "metrics": (fitted or {}).get("metrics")},
            "versions": [{**{k: v[k] for k in ("version", "created_by", "active", "metrics")},
                          "created_at": v["created_at"].isoformat() if hasattr(v["created_at"], "isoformat") else v["created_at"]}
                         for v in versions]}


def judged_report(rows: list[dict], gave_up: set[str]) -> dict:
    """Judged labels in these rows: how many samples were taken and what came of each (judged, pending, dropped and why,
    skipped, not taken because the cap was spent), the verdicts by task type and rules level, how far they agree with the
    outcome signals, and two checks on the judge: how often it preferred the answer shown first (about half, if the order
    doesn't sway it) and the stronger one. Costs come from the ledger (reports.py adds them)."""
    outcomes: Counter = Counter()
    why: Counter = Counter()
    levels: dict[tuple, list[int]] = {}
    first = stronger = preferred = 0
    recorded = 0.0
    for r in rows:
        j = r.get("judged")
        if not isinstance(j, dict):
            continue
        outcomes[j.get("outcome") or "unknown"] += 1
        if j.get("why"):
            why[j["why"]] += 1
        recorded += float(j.get("cost") or 0)
        if j.get("outcome") != "judged":
            continue
        g = levels.setdefault((r["task"], r["rules_level"]), [0, 0])
        g[0] += 1
        g[1] += not j.get("good_enough")
        if j.get("better") in ("cheaper", "stronger"):
            preferred += 1
            first += j["better"] == j.get("first")
            stronger += j["better"] == "stronger"
    return {"samples": sum(outcomes.values()), "outcomes": dict(outcomes), "why": dict(why.most_common(10)),
            "recorded_cost": round(recorded, 6), **{k: v for k, v in judged_agreement(rows, gave_up).items()},
            "levels": [{"task": t, "level": lvl, "labels": n, "not_good_enough": bad} for (t, lvl), (n, bad) in sorted(levels.items())],
            "preferred": preferred, "preferred_first": round(first / preferred, 4) if preferred else None,
            "preferred_stronger": round(stronger / preferred, 4) if preferred else None}


def report(rows: list[dict], limit: int = 100) -> dict:
    """Where the learned router and the rules disagreed, what each side's pick was estimated to cost for the task, and
    the outcome signals: seen, on the side that acted; expected (the model's own chance of a failure signal), on both.
    Rows read without their hashed words count no "asked again elsewhere" (it needs them); training still does."""
    gave_up = abandoned(rows) if rows and "text_features" in rows[0] else set()
    outcomes = Counter()
    groups: dict[tuple, dict] = {}
    by_direction: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # requests, with a failure signal
    shown = []
    totals = {"rules": 0.0, "learned": 0.0, "priced": 0}
    for r in rows:
        call = r.get("learned")
        if not call:
            continue
        outcomes[call["outcome"]] += 1
        if call.get("level") is None:
            continue
        rules, learned = call.get("capped") or (r["rules_level"], call["level"])
        sig = signals(r, gave_up)
        status = r.get("status") or 0
        seen = {**sig, "refused": bool(r.get("refused_scores")), "error": status >= 400 and status != 499}
        bad = any(v for k, v in seen.items() if k != "error")
        direction = "agreed" if rules == learned else "higher" if learned > rules else "lower"
        by_direction[direction][0] += 1
        by_direction[direction][1] += bad
        if direction == "agreed":
            continue
        used = bool(call.get("used"))
        rules_model, learned_model = (call.get("alt_model"), r.get("served_model")) if used else (r.get("served_model"), call.get("alt_model"))
        rules_cost, learned_cost = (call.get("alt_cost"), call.get("cost")) if used else (call.get("cost"), call.get("alt_cost"))
        fail = call.get("fail") or [None, None]
        g = groups.setdefault((r["task"], rules, learned), {"task": r["task"], "rules": rules, "learned": learned, "requests": 0, "used": 0,
                                                            "rules_cost": 0.0, "learned_cost": 0.0, "priced": 0, "signals": Counter(),
                                                            "with_signal": 0, "fail_rules": [], "fail_learned": [], "confidence": []})
        g["requests"] += 1
        g["used"] += used
        g["with_signal"] += bad
        g["signals"].update(k for k, v in seen.items() if v)
        g["confidence"].append(call["confidence"])
        if fail[0] is not None:
            g["fail_rules"].append(fail[0])
            g["fail_learned"].append(fail[1])
        if rules_cost is not None and learned_cost is not None:
            g["priced"] += 1
            g["rules_cost"] += rules_cost
            g["learned_cost"] += learned_cost
            totals["rules"] += rules_cost
            totals["learned"] += learned_cost
            totals["priced"] += 1
        shown.append({"ts": r["ts"].isoformat(), "team": r.get("team"), "user": r.get("user_id"), "task": r["task"], "version": call["version"],
                      "rules": rules, "learned": learned, "confidence": call["confidence"], "used": used, "outcome": call["outcome"],
                      "rules_model": rules_model, "learned_model": learned_model, "rules_cost": _usd(rules_cost),
                      "learned_cost": _usd(learned_cost), "acted": "learned" if used else "rules",
                      "signals": sorted(k for k, v in seen.items() if v), "fail": fail})

    def mean(xs: list[float]) -> float | None:
        return round(sum(xs) / len(xs), 4) if xs else None

    out_groups = []
    for g in sorted(groups.values(), key=lambda g: -g["requests"]):
        out_groups.append({**{k: g[k] for k in ("task", "rules", "learned", "requests", "used", "priced", "with_signal")},
                           "rules_cost": _usd(g["rules_cost"]), "learned_cost": _usd(g["learned_cost"]),
                           "signals": dict(g["signals"]), "confidence": mean(g["confidence"]),
                           "expected_rules": mean(g["fail_rules"]), "expected_learned": mean(g["fail_learned"])})
    compared = sum(v[0] for v in by_direction.values())
    return {"judged": judged_report(rows, gave_up), "requests": sum(outcomes.values()), "outcomes": dict(outcomes), "compared": compared,
            "agreed": by_direction["agreed"][0], "disagreed": compared - by_direction["agreed"][0],
            "cost": {"rules": _usd(totals["rules"]), "learned": _usd(totals["learned"]),
                     "difference": _usd(totals["learned"] - totals["rules"]), "priced": totals["priced"]},
            "signal_rates": {k: {"requests": v[0], "with_signal": v[1], "rate": round(v[1] / v[0], 4) if v[0] else None}
                             for k, v in by_direction.items()},
            "groups": out_groups, "rows": list(reversed(shown))[:limit]}
