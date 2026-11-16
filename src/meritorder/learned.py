"""The learned router (decision 44, docs/03-routing.md#the-learned-router): a request's level, learned from this
organization's own outcomes. Opt-in; the rules (classifier.py) stay the default and the fallback.

The rules give every request a level. The learned model corrects it: it estimates how hard the request really is,
in level units, as the rules' level plus what the request's features add or take away,

    difficulty = rules level + sum of the weights of the request's features

and predicts the level as that difficulty rounded to 1-5. It's trained (learned_jobs.py) on outcomes, as an item
response model: a request of difficulty d served by a model scoring q shows a failure signal (the same request sent
again, a raised X-Route-Floor, a stopped answer, ...) with probability sigmoid(slope x (d - q) + intercept + client),
the logistic link Epoch's ability scale uses between models and tasks. The client's own tags only move how often a
failure gets noticed, never the difficulty. The corrections average to zero over the training requests: the labels
can't say how hard work is in absolute terms, only which requests are harder than others the rules put at the same
level.

The model is an ensemble of six fits (learned_jobs.MEMBERS), each on a different half of the requests. Its prediction is their mean, and
its confidence the chance that the level is right given how far they spread (plus, for words and tags none of them saw,
the spread a weight has before any evidence). Below ``routing.learned.min_confidence`` the rules decide.

Features are computed at request time and never hold the text: the words and word pairs of the text the rules read
(the first and last WINDOW characters of it), hashed with a key into TEXT_BUCKETS buckets, and named facts about the
request (its rules, tools, size, client, Claude Code's request class). Inference is pure Python: about 0.4 ms at the
p99 for a long agent prompt on a laptop, most of it splitting the words.
"""

from __future__ import annotations

import array
import base64
import hashlib
import json
import logging
import math
import re
import secrets
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .classifier import Floor, _recent_user_text

log = logging.getLogger("meritorder.learned")

# Change whenever features() changes meaning: models and stored rows of another version aren't used. 4: the words leave
# out the harness's own text (Claude Code's system reminders, Codex's AGENTS.md), and the rules' level for coding comes
# from the task's wording, not the tool count (decision 61: `rule:agent` is now the tool count itself).
FEATURE_VERSION = 4
# (3: the words skip user messages made only of tool results, classifier._recent_user_text)
# (2: words hashed with keyed BLAKE2b; 1 used CRC32 seeded with the key, which is linear and hid nothing)
TEXT_BITS = 16
TEXT_BUCKETS = 1 << TEXT_BITS
_MASK = TEXT_BUCKETS - 1
WINDOW = 2000  # characters read from each end of the text the rules read (the rules' own hits are features too)
KEY_SETTING = "learned_router_key"  # the hashing key, when MERITORDER_FEATURE_KEY doesn't set one
KEY_ENV = "MERITORDER_FEATURE_KEY"
TASKS = ("coding", "general")
# Words (ASCII letters, digits, _), runs of non-ASCII bytes, and runs of two or more symbols (```, =>, ::, ->).
_WORD = re.compile(rb"[a-z0-9_]+|[\x80-\xff]+|[^\sa-z0-9_\x80-\xff]{2,}")
_NAME = re.compile(r"[^a-z0-9._-]")
# Tags about the client, not the work: they change how often a failure gets noticed (a chat app has a regenerate
# button, an agent rarely resends), so the model puts them in the intercept, never in the difficulty. A judged label
# (judged.py) carries "judged:pair" instead of its request's client tags: a judge looks at every answer it's given.
DETECTION = ("client:", "format:", "judged:")


def detection(tag: str) -> bool:
    return tag.startswith(DETECTION)


def key_seed(key: str) -> tuple[bytes, str]:
    """The hashing key a key gives (128 bits, for keyed BLAKE2b), and a short fingerprint of it (stored with features and
    models; no part of the hashing key)."""
    digest = hashlib.sha256(key.encode()).digest()
    return digest[:16], digest[16:22].hex()


def new_key() -> str:
    return secrets.token_urlsafe(32)


def rules_hash(levels: Any) -> str:
    """What a model was trained against: its weights correct these rules, so other rules make it stale."""
    dumped = levels.model_dump() if hasattr(levels, "model_dump") else levels
    return hashlib.sha256(json.dumps(dumped, sort_keys=True).encode()).hexdigest()[:16]


# ------------------------------------------------------------------ features


@dataclass
class Features:
    """What the learned router knows about a request. ``text``: the hashed buckets of its words and word pairs, sorted;
    ``tags``: named facts about it (never its text)."""

    task: str
    rules: int  # the rules' level, before Claude Code's caps
    text: list[int] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


def _band(n: int, edges: tuple[int, ...], labels: tuple[str, ...]) -> str:
    for edge, lab in zip(edges, labels):
        if n <= edge:
            return lab
    return labels[-1]


def _client(headers: dict[str, str]) -> str:
    """The client's product from its User-Agent (claude-cli, openai, anthropic, pi, ...): a name, not content."""
    ua = (headers.get("user-agent") or "").strip().lower()
    name = re.split(r"[/\s(]", ua, maxsplit=1)[0] if ua else ""
    return _NAME.sub("", name)[:24] or "unknown"


def _rule_tags(reason: str) -> list[str]:
    """Which rules set the level, from the classifier's reason (classifier.classify writes these phrases)."""
    found = [("hardest wording", "rule:hardest"), ("; hard wording", "rule:hard"), ("simple task wording", "rule:simple"),
             ("corrects the last answer", "rule:correction"), ("analysis wording", "rule:analysis"), ("short question", "rule:short"),
             ("code-like text", "rule:code"), ("X-Route-Hint", "rule:hint")]
    tags = [tag for phrase, tag in found if phrase in reason]
    if "coding:" in reason and " tool" in reason:
        tags.append("rule:tools")
    return tags


def text_buckets(text: str, seed: bytes) -> list[int]:
    """The keyed hashes of the text's words and word pairs, as sorted unique buckets. Only the first and last WINDOW
    characters are read: that keeps a long pasted prompt within the latency budget."""
    if len(text) > 2 * WINDOW:
        text = text[:WINDOW] + "\n" + text[-WINDOW:]
    toks = _WORD.findall(text.lower().encode("utf-8", "replace"))
    if not toks:
        return []
    # A keyed hash, so that without the key a bucket says nothing about the word that made it. (CRC32 seeded with the key
    # didn't: it's linear, so every word of one length moved by the same XOR whatever the key.)
    h = {t: int.from_bytes(hashlib.blake2b(t, digest_size=4, key=seed).digest(), "little") for t in set(toks)}
    out = {v & _MASK for v in h.values()}
    out.update(((h[a] * 0x01000193) ^ h[b]) & _MASK for a, b in set(zip(toks, toks[1:])))
    return sorted(out)


def features(rules: Floor, body: dict, headers: dict[str, str], tokens: int, client_format: str, seed: bytes,
             agent_tools: int = 5) -> Features:
    """Everything the learned router uses, computed while the request is in memory. Nothing here can be read back as
    the request's text: words are hashed with a key, and the tags are counts and names the gateway already logs.
    ``rule:agent``: at least ``agent_tools`` tools (routing.levels), an agent at work, whether or not that set the level."""
    tools = body.get("tools") or []
    msgs = body.get("messages") or []
    tags = [
        f"rules:{rules.task}:{rules.level}",
        "tools:" + _band(len(tools), (0, 2, 4, 9, 19, 49), ("0", "1-2", "3-4", "5-9", "10-19", "20-49", "50+")),
        "size:" + _band(tokens, (399, 1999, 7999, 31999, 127999, 511999), ("<400", "<2k", "<8k", "<32k", "<128k", "<512k", "512k+")),
        "turns:" + _band(len(msgs), (1, 3, 9, 29), ("1", "2-3", "4-9", "10-29", "30+")),
        f"format:{client_format or 'openai'}",
        f"client:{_client(headers)}",
        f"class:{_NAME.sub('', (headers.get('x-claude-code-request-class') or 'none').lower())[:16] or 'none'}",
        *_rule_tags(rules.reason),
    ]
    if len(tools) >= agent_tools:
        tags.append("rule:agent")
    if headers.get("x-claude-code-agent-id"):
        tags.append("subagent")
    effort = (body.get("output_config") or {}).get("effort") or body.get("reasoning_effort") or (body.get("reasoning") or {}).get("effort")
    if effort in ("minimal", "low", "medium", "high", "xhigh", "max"):
        tags.append(f"effort:{effort}")
    if body.get("thinking"):
        tags.append("thinking")
    if body.get("response_format"):
        tags.append("structured")
    return Features(rules.task, rules.level, text_buckets(_recent_user_text(body), seed), tags)


# ------------------------------------------------------------------ the model


def floats(values) -> str:
    return base64.b64encode(array.array("f", values).tobytes()).decode()


def unfloats(text: str, n: int) -> array.array:
    a = array.array("f")
    a.frombytes(base64.b64decode(text))
    if len(a) != n:
        raise ValueError(f"expected {n} values, found {len(a)}")
    return a


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


@dataclass
class Prediction:
    version: int
    task: str
    level: int
    rules: int
    confidence: float  # the chance the level is right, given the evidence on the request's features
    difficulty: float  # in level units: the rules' level plus the features' weights
    spread: float  # its standard deviation
    parts: list[tuple[str, float]] = field(default_factory=list)  # the largest contributions, for the routing reason
    noticed: float = 0.0  # how much likelier a failure is to be noticed from this client (log-odds), for fail_chance

    @property
    def floor(self) -> str:
        return f"{self.task}:{self.level}"

    def why(self) -> str:
        """'wording +0.7, tools:10-19 +0.2': what moved it from the rules' level."""
        return ", ".join(f"{name} {v:+.1f}" for name, v in self.parts) or "no feature moves it"


@dataclass
class Member:
    """One model of the ensemble, trained on a resampled copy of the requests."""

    text_w: array.array
    tags_w: dict[str, float]
    slope: dict[str, float]
    intercept: dict[str, float]
    shift: dict[str, float]  # per task type, so the corrections average to zero over the training requests


class Model:
    """A trained learned-router model (one version in the store's router_models), ready to predict.

    It's a small ensemble: each member was trained on a resampled copy of the requests (learned_jobs.train). The
    prediction is their mean; how far they spread is how sure it is, plus, for words and tags no training request had,
    the spread a weight has before any evidence (1 / ``prior``)."""

    def __init__(self, doc: dict, version: int, created_at: datetime | None = None):
        if doc.get("format") != 1:
            raise ValueError(f"unknown model format {doc.get('format')!r}")
        self.doc = doc
        self.version = version
        self.created_at = created_at
        self.fv: str = doc["feature_version"]
        self.trained_at = datetime.fromisoformat(doc["trained_at"])
        self.rules_hash: str = doc["rules_hash"]
        self._levels: Any = None  # the level rules last checked against, and their hash
        self._levels_hash = ""
        self.prior = float(doc["prior"])
        self.members = [Member(unfloats(m["text_w"], TEXT_BUCKETS), {k: float(v) for k, v in m["tags_w"].items()},
                               {k: float(v) for k, v in m["slope"].items()}, {k: float(v) for k, v in m["intercept"].items()},
                               {k: float(v) for k, v in m["shift"].items()})
                        for m in doc["members"]]
        if not self.members:
            raise ValueError("the model has no members")
        unseen = array.array("B")
        unseen.frombytes(base64.b64decode(doc["unseen"]))
        if len(unseen) != TEXT_BUCKETS:
            raise ValueError("the model's list of unseen words has the wrong size")
        self.unseen = unseen  # 1 for a bucket no training request had
        self.tags_seen = set(doc["tags_seen"])
        n = len(self.members)
        self.slope = {t: sum(m.slope.get(t, 0.5) for m in self.members) / n for t in TASKS}
        self.intercept = {t: sum(m.intercept.get(t, -3.0) for m in self.members) / n for t in TASKS}
        self.examples = int(doc.get("examples") or 0)

    def stale(self, levels: Any, fv: str, now: datetime, max_age_days: float | None) -> str | None:
        """Why this model shouldn't be used now, or None."""
        if fv != self.fv:
            return "its features were hashed differently (another feature version or key)"
        if levels is not self._levels:  # hashed once per configuration, not per request (a reload brings new rules)
            self._levels, self._levels_hash = levels, rules_hash(levels)
        if self.rules_hash != self._levels_hash:
            return "the level rules changed since it was trained"
        age = (now - self.trained_at).total_seconds() / 86400
        if max_age_days and age > max_age_days:
            return f"trained {age:.0f} days ago, more than the {max_age_days:g} allowed"
        return None

    def difficulty(self, f: Features) -> tuple[float, float, list[tuple[str, float]]]:
        """The request's difficulty in level units (the members' mean), its variance, and each feature group's mean
        contribution. The client's tags aren't part of it: they say how often a failure is noticed (``noticed``)."""
        n = len(self.members)
        norm = 1.0 / math.sqrt(len(f.text)) if f.text else 0.0  # the text vector has unit length
        words = [sum(map(m.text_w.__getitem__, f.text)) * norm for m in self.members] if f.text else [0.0] * n
        work = [t for t in f.tags if not detection(t)]
        tags = [[m.tags_w.get(t, 0.0) for t in work] for m in self.members]
        ds = [f.rules + m.shift.get(f.task, 0.0) + w + sum(t) for m, w, t in zip(self.members, words, tags)]
        d = sum(ds) / n
        spread = sum((x - d) ** 2 for x in ds) / (n - 1) if n > 1 else 0.0
        # Weights no training request informed have only the prior's spread.
        unseen = (sum(map(self.unseen.__getitem__, f.text)) * norm * norm if f.text else 0.0) + sum(t not in self.tags_seen for t in work)
        parts = [("wording", sum(words) / n)] if f.text else []
        parts += [(t, sum(row[i] for row in tags) / n) for i, t in enumerate(work)]
        return d, spread + unseen / self.prior, parts

    def noticed(self, f: Features) -> float:
        return sum(m.tags_w.get(t, 0.0) for m in self.members for t in f.tags if detection(t)) / len(self.members)

    def predict(self, f: Features) -> Prediction:
        d, var, parts = self.difficulty(f)
        if not math.isfinite(d) or not math.isfinite(var):
            raise ValueError("the model gave a difficulty that isn't a number")
        level = max(1, min(5, math.floor(d + 0.5)))
        s = math.sqrt(max(var, 1e-12))
        lo = -math.inf if level == 1 else level - 0.5
        hi = math.inf if level == 5 else level + 0.5
        confidence = (1.0 if hi == math.inf else _phi((hi - d) / s)) - (0.0 if lo == -math.inf else _phi((lo - d) / s))
        top = sorted((p for p in parts if abs(p[1]) >= 0.05), key=lambda p: -abs(p[1]))[:3]
        return Prediction(self.version, f.task, level, f.rules, confidence, d, s, top, self.noticed(f))

    def fail_chance(self, task: str, difficulty: float, score: float, noticed: float = 0.0) -> float:
        """The chance of a failure signal when work of this difficulty goes to a model with this score."""
        return sigmoid(self.slope.get(task, 0.5) * (difficulty - score) + self.intercept.get(task, -3.0) + noticed)


# ------------------------------------------------------------------ at request time


@dataclass
class LearnedCall:
    """What the learned router said about one request, and what came of it (Decision.learned)."""

    mode: str  # shadow or route
    rules: Floor  # the rules' level
    features: Features
    fv: str
    prediction: Prediction | None = None
    used: bool = False
    # used, agrees, shadow (disagrees, not used in shadow mode), unsure, stale, missing, failed
    outcome: str = "missing"
    note: str = ""  # for the routing reason
    # The pick at the other level (the learned one when the rules acted, the rules' when it did), and the estimated cost
    # of the task from here on each side, as the router priced it.
    alt_model: str | None = None
    alt_cost: float | None = None
    cost: float | None = None
    capped: tuple[int, int] | None = None  # the rules' and the learned level after Claude Code's caps, when either was capped
    # The model's own chance of a failure signal on the rules' pick and on its own (Model.fail_chance), where they differ.
    fail: tuple[float, float] | None = None

    @property
    def version(self) -> int | None:
        return self.prediction.version if self.prediction else None

    def learned_floor(self) -> Floor | None:
        p = self.prediction
        if p is None:
            return None
        why = f"learned router v{p.version}: {p.confidence:.0%} sure ({p.why()}); the rules said {self.rules} ({self.rules.reason})"
        return Floor(p.task, p.level, why)

    def record(self) -> dict:
        """The ``learned`` column of its route_features row."""
        p = self.prediction
        return {"mode": self.mode, "outcome": self.outcome, "used": self.used, "version": self.version,
                "level": p.level if p else None, "confidence": round(p.confidence, 4) if p else None,
                "difficulty": round(p.difficulty, 3) if p else None, "spread": round(p.spread, 3) if p else None,
                "alt_model": self.alt_model, "alt_cost": self.alt_cost, "cost": self.cost,
                "capped": list(self.capped) if self.capped else None,
                "fail": [round(x, 4) for x in self.fail] if self.fail else None}


class LearnedRouter:
    """The model gateways use (or none), with the hashing key. Lives across configuration reloads, like measured speeds."""

    def __init__(self, model: Model | None = None, key: str | None = None, error: str | None = None):
        self.model = model
        self.error = error  # why the stored model couldn't be loaded
        self.key_loaded = key is not None
        self.seed, fingerprint = key_seed(key) if key else (b"", "none")
        self.fv = f"f{FEATURE_VERSION}-{fingerprint}"

    def consult(self, rules: Floor, body: dict, headers: dict[str, str], tokens: int, client_format: str, settings: Any,
                levels: Any, now: datetime | None = None) -> LearnedCall:
        """The learned router's level for a request the rules gave ``rules``, and whether it's used. Never raises: a
        model that fails leaves the rules in charge, and says so."""
        mode = settings.mode
        call = LearnedCall(mode, rules, features(rules, body, headers, tokens, client_format, self.seed, getattr(levels, "agent_tools", 5)),
                           self.fv)
        shadow = " (shadow)" if mode == "shadow" else ""
        m = self.model
        if not self.key_loaded:
            call.outcome, call.note = "failed", "learned router: its feature key isn't loaded yet, so the rules decide"
            return call
        if m is None:
            call.note = (f"learned router: {self.error}, so the rules decide" if self.error
                         else f"learned router{shadow}: no model trained yet, so the rules decide")
            call.outcome = "failed" if self.error else "missing"
            return call
        stale = m.stale(levels, self.fv, now or datetime.now(UTC), settings.max_age_days)
        if stale:
            call.outcome, call.note = "stale", f"learned router v{m.version} is stale ({stale}), so the rules decide"
            return call
        try:
            p = m.predict(call.features)
        except Exception as e:  # noqa: BLE001 (any failure leaves the rules in charge)
            log.warning("learned router v%s failed: %s", m.version, e)
            call.outcome, call.note = "failed", f"learned router v{m.version} failed ({type(e).__name__}), so the rules decide"
            return call
        call.prediction = p
        sure = f"{p.confidence:.0%} sure"
        if p.level == rules.level:
            call.outcome, call.note = "agrees", f"learned router v{m.version}{shadow} agrees ({p.floor}, {sure})"
        elif p.confidence < settings.min_confidence:
            call.outcome = "unsure"
            call.note = (f"learned router v{m.version}{shadow}: {p.floor}, {sure} ({p.why()}), under the {settings.min_confidence:.0%} "
                         "it needs, so the rules decide")
        elif mode == "shadow":
            call.outcome, call.note = "shadow", f"learned router v{m.version} (shadow): {p.floor}, {sure} ({p.why()}); not used in shadow mode"
        else:
            call.outcome, call.used = "used", True
        return call
