"""Synthetic traffic for the learned router's tests: requests whose true level is known, served by models of known
scores, with outcome signals drawn from a known item response curve. Nothing here calls a model."""

from __future__ import annotations

import math
import random
import uuid
from datetime import UTC, datetime, timedelta

from meritorder.classifier import classify
from meritorder.learned import LearnedRouter, features

KEY = "test-feature-key"
LR = LearnedRouter(key=KEY)
SCORES = {"cheap": 1.5, "mid": 3.0, "strong": 4.0, "best": 5.0}
SERVICES = ["billing", "auth", "search", "checkout", "ledger", "inventory", "mailer", "gateway"]
FILES = ["parser.py", "app.py", "models.py", "views.ts", "main.go", "lib.rs"]
NOUNS = ["date", "total", "header", "offset", "count", "name"]
COUNTRIES = ["France", "Peru", "Japan", "Kenya", "Chile", "Norway"]
HEADERS = {"user-agent": "claude-cli/2.1.0 (external, cli)"}

# The known rule: the rules give every one of these coding:3 (one tool), but deploying to Kubernetes really needs
# level 4 and changing a default value only level 2. On average the rules are right, as the learned router assumes.
KINDS = {"kube": 4, "plain": 3, "rename": 2, "short": 1}


def request(kind: str, rng: random.Random) -> dict:
    tool = [{"name": "bash", "description": "run a command", "input_schema": {"type": "object"}}]
    if kind == "kube":
        text = f"deploy the {rng.choice(SERVICES)} service to kubernetes with helm and roll it out safely"
    elif kind == "plain":
        text = f"fix the bug in {rng.choice(FILES)} where the {rng.choice(NOUNS)} is wrong"
    elif kind == "rename":
        text = f"set the default {rng.choice(NOUNS)} to {rng.choice(NOUNS)}_value in {rng.choice(FILES)}"
    else:
        return {"messages": [{"role": "user", "content": f"what is the capital of {rng.choice(COUNTRIES)}?"}]}
    return {"messages": [{"role": "user", "content": text}], "tools": tool}


def failure_chance(true_level: int, score: float) -> float:
    """How often a model scoring ``score`` fails work of this level: a steep item response curve."""
    return 1 / (1 + math.exp(-(2.5 * (true_level - score) - 1.5)))


def rows(n: int, seed: int = 1, start: datetime | None = None, detect: float = 0.6, noise: float = 0.02,
         fv: str | None = None, judged: float = 0.0, judge_noise: float = 0.0) -> list[dict]:
    """``n`` route_features rows as the gateway records them, each request served by a random model (as named-model
    traffic and cost per success spread them), with a retry seen for ``detect`` of the failures and ``noise`` of the
    rest. ``judged``: the share that also has a judged label (judged.py), which says whether the answer really failed,
    except that ``judge_noise`` of them are a coin toss (1: a judge that only guesses)."""
    rng = random.Random(seed)
    jr = random.Random(seed + 7)  # its own draws, so rows without judged labels come out as they always have
    start = start or datetime.now(UTC) - timedelta(days=2)
    out = []
    kinds = list(KINDS)
    for i in range(n):
        kind = kinds[i % len(kinds)]
        body = request(kind, rng)
        rules = classify(body, {}, 50)
        f = features(rules, body, HEADERS, 50, "anthropic", LR.seed)
        model = rng.choice(list(SCORES))
        failed = rng.random() < failure_chance(KINDS[kind], SCORES[model])
        seen = rng.random() < (detect if failed else noise)
        out.append({"request_id": uuid.UUID(int=rng.getrandbits(128)), "ts": start + timedelta(seconds=10 * i), "team": "eng",
                    "user_id": f"u{i}", "session_id": f"s{i}", "thread": f"t{i}", "fv": fv or LR.fv, "task": f.task,
                    "rules_level": f.rules, "text_features": f.text, "tags": f.tags, "auto": True, "served_model": model,
                    "served_score": SCORES[model], "refused_scores": [], "status": 200, "retried": seen, "floor_raised": None,
                    "learned": None, "kind": kind})
        if judged and jr.random() < judged:
            good = (not failed) if jr.random() >= judge_noise else jr.random() < 0.5
            out[-1]["judged"] = {"outcome": "judged", "good_enough": good, "stronger_good_enough": True, "first": "cheaper",
                                 "better": "same" if good else "stronger", "served": model, "stronger": "best", "judge": "gpt-sol",
                                 "cost": 0.01}
    return out


def probe(kind: str, seed: int = 99):
    """A fresh request of this kind: its rules floor and features."""
    body = request(kind, random.Random(seed))
    rules = classify(body, {}, 50)
    return body, rules, features(rules, body, HEADERS, 50, "anthropic", LR.seed)
