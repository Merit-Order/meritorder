"""Offline routing simulator and scorecard (docs/03-routing.md#how-we-measure-routing).

Routing is measured here before any real traffic, and without a single model call: a population of tasks with a known
difficulty is run through each strategy's choices, every turn priced as the gateway prices it, and each model's success
drawn from how hard the task is against how able the model is. The strategies are the gateway's own code: requests are
built and handed to ``Router.decide`` (and through it the classifier and the learned router), exactly as proxy.py does,
and the session is kept as ``Gateway.update_session`` keeps it. Only the outcome is simulated.

Four parts:

* **The population** (``population``): coding-agent sessions (easy: tests, renames, docs; medium: features and bugs;
  hard: multi-file changes, concurrency, migrations; very hard: architecture), chat, analysis and short questions. Each
  task has an opening prompt from varied templates (some worded misleadingly: a "quick fix" that's hard work, an easy job
  that mentions concurrency), its turns, tokens, tools and pauses, and a difficulty on Epoch's ability scale.
* **Execution** (``Simulator``): for the model a strategy chose, success is ``sigmoid(ITEM_SLOPE x (ability - difficulty))``
  with the model's published ability (prices.json, Epoch), and the same random draw for every model on the same task, so
  strategies are compared on the same luck. The agent's tool calls and results are generated to match the outcome (tests
  failing, fix loops, a correction in the person's next prompt) and read by outcomes.py, as the gateway reads them. Every
  turn is priced with pricing.cost on the route's price, with the cache as the provider keeps it (5 minutes or an hour,
  lost after a pause or on a move to another model, rebuilt after compaction).
* **The scorecard** (``scorecard``): per strategy and task class, cost, success, cost per finished task, the person's
  time on failed attempts, escalations, how often the choice was higher or lower than needed against an oracle (the
  cheapest model that would have succeeded), and savings against always using Sonnet or Opus.
* **Replay** (``replay``): real task starts from an exported ledger, explored at random (decision 59), score other level
  policies with inverse-propensity weights, without spending anything.

Every number the simulation assumes, and where it comes from, is in docs/03-routing.md#how-we-measure-routing.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import importlib
import io
import json
import math
import random
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, get_args

from . import outcomes
from .backends.base import Backend
from .cache_shaping import FIVE_MINUTES, HOUR, needs_cache_marker
from .config import Config, ModelChoice, Route, build_config, load_catalog, load_raw_config
from .learned_jobs import failed, signals
from .pricing import Usage
from .pricing import cost as priced
from .router import Router, RoutingError
from .sessions import Session
from .success import track

# ------------------------------------------------------------------ calibration (docs/03-routing.md#how-we-measure-routing)

# Epoch's ability scale: a model of ability t solves a task of difficulty d with chance sigmoid(slope x (t - d)). A
# benchmark's slope is shallower than one task's, because its tasks spread in difficulty; a task's own is steeper.
ITEM_SLOPE = 0.25
# Epoch's difficulty (EDI) and slope for FrontierSWE (benchmark_data.zip, edi_scores.csv, fetched 15 November 2026), the
# reference the router's scores are measured against. The hard coding class is FrontierSWE-like: its mean and spread
# reproduce this curve for every model (``check_calibration``).
FRONTIER_SWE = (165.48, 0.1507)
# SWE-bench Verified on the same scale: a logistic fit of Epoch's own runs of 31 models (swe_bench_verified.csv) against
# their capabilities index (eci_scores.csv). The flat slope is the benchmark's ceiling: some tasks fail for every model.
SWE_VERIFIED = (130.9, 0.050)
# GDPval (real knowledge work: reports, analyses, plans) from edi_scores.csv: the analysis class is GDPval-like.
GDPVAL = (154.79, 0.1019)
# The share of tasks whose first prompt doesn't say enough: the first attempt fails on any model and the correction that
# follows says what was meant. With it the easy and medium mix reaches SWE-bench Verified's ceiling (about 90%).
AMBIGUOUS = 0.08
# Spread of a model's strength on one task around its ability (log-odds): models aren't ranked the same on every task.
TASK_NOISE = 0.5
HUMAN_COST = 15.0  # dollars of a person's time per failed attempt: noticing, re-prompting, reviewing again (10 minutes at $90/h)
MAX_ATTEMPTS = 3  # a person tries this many times before giving the task up (or doing it themselves)
PAUSE_SHARE = 0.10  # the share of later turns after a 5-60 minute pause (pauses.PRIOR, the router's own prior)
LONG_PAUSE = 0.02  # ...and after more than an hour
COMPACT_AT = 160_000  # Claude Code compacts a conversation about here (on a 200K window); the summary is COMPACT_TO
COMPACT_TO = 30_000
START = datetime(2026, 11, 16, 9, 0, tzinfo=UTC).timestamp()


@dataclass(frozen=True)
class TaskClass:
    """A kind of task: its type for routing (coding or general), its difficulty on Epoch's scale (mean and spread), and
    its shape: turns, the prompt it starts with, what each turn adds, how long each reply is (tokens, ranges)."""

    name: str
    task: str
    mu: float
    sigma: float
    turns: tuple[int, int]
    start: tuple[int, int]
    growth: tuple[int, int]
    output: tuple[int, int]
    agent: bool
    categories: tuple[str, ...]


CLASSES: dict[str, TaskClass] = {c.name: c for c in (
    TaskClass("coding-easy", "coding", 124.0, 8.0, (4, 14), (16_000, 24_000), (400, 2_500), (150, 700), True, ("tests", "rename", "docs")),
    TaskClass("coding-medium", "coding", 136.0, 9.0, (12, 40), (18_000, 30_000), (600, 3_000), (200, 900), True, ("feature", "bug")),
    TaskClass("coding-hard", "coding", FRONTIER_SWE[0], 8.45, (30, 90), (20_000, 40_000), (800, 3_500), (250, 1_200), True,
              ("multi-file", "concurrency", "migration")),
    TaskClass("coding-very-hard", "coding", 173.0, 8.0, (60, 160), (24_000, 45_000), (900, 4_000), (300, 1_500), True, ("architecture",)),
    TaskClass("chat-short", "general", 100.0, 10.0, (1, 1), (40, 250), (0, 0), (80, 400), False, ("question",)),
    TaskClass("chat", "general", 120.0, 10.0, (1, 4), (600, 4_000), (100, 800), (200, 900), False, ("writing", "explain")),
    TaskClass("analysis", "general", GDPVAL[0], 14.3, (1, 4), (2_000, 20_000), (200, 1_500), (500, 2_000), False, ("analysis",)),
)}
CODING = ("coding-easy", "coding-medium", "coding-hard", "coding-very-hard")
GENERAL = ("chat-short", "chat", "analysis")
# SWE-bench Verified's annotated time to fix (OpenAI, August 2024): under 15 minutes 196 tasks, 15-60 minutes 259, 1-4
# hours 42, over 4 hours 3.
SWE_MIX = {"coding-easy": 196 / 500, "coding-medium": 259 / 500, "coding-hard": 42 / 500, "coding-very-hard": 3 / 500}
# What people ask chat assistants (Chatterji et al., "How People Use ChatGPT", NBER w34255, 2025): seeking information,
# practical guidance and writing make most of it; analysis-heavy work is the smaller part.
CHAT_MIX = {"chat-short": 0.30, "chat": 0.50, "analysis": 0.20}
CODING_SHARE = 0.70  # a gateway in front of coding agents: most task starts (and nearly all the spend) are coding


def mix(coding: dict[str, float] | None = None, chat: dict[str, float] | None = None, coding_share: float = CODING_SHARE) -> dict[str, float]:
    """A population's share of each class: ``coding_share`` of coding tasks in the ``coding`` proportions, the rest chat."""
    coding, chat = coding or SWE_MIX, chat or CHAT_MIX
    cs, ts = sum(coding.values()) or 1.0, sum(chat.values()) or 1.0
    return {**{k: coding_share * v / cs for k, v in coding.items()}, **{k: (1 - coding_share) * v / ts for k, v in chat.items()}}


MIXES = {
    "standard": mix(),
    "easier": mix({"coding-easy": 0.55, "coding-medium": 0.40, "coding-hard": 0.04, "coding-very-hard": 0.01}),
    "harder": mix({"coding-easy": 0.20, "coding-medium": 0.45, "coding-hard": 0.25, "coding-very-hard": 0.10}),
    "coding-only": mix(coding_share=1.0),
    "chat-only": mix(coding_share=0.0),
}

# ------------------------------------------------------------------ prompts

_FILES = ["app/api.py", "src/billing/invoice.ts", "server/handlers.go", "lib/cache.rs", "core/scheduler.py", "web/src/Cart.tsx",
          "pkg/auth/session.go", "services/report.py", "src/parser.ts", "internal/queue/worker.go", "models/user.py", "src/db.rs"]
_THINGS = ["the invoice export", "session refresh", "the retry logic", "the CSV importer", "rate limiting", "the search index",
           "the payment webhook", "user settings", "the job queue", "image uploads", "the audit log", "feature flags"]
_NAMES = [("get_user", "fetch_user"), ("TotalPrice", "OrderTotal"), ("cfg", "settings"), ("doWork", "process_batch"),
          ("old_client", "api_client"), ("parse_row", "decode_row")]
_SYMPTOMS = ["returns stale data after an update", "times out under load", "drops the last row", "double-charges on retry",
             "shows the wrong timezone", "logs people out at random", "loses updates when two requests overlap"]
_STACKS = [("Flask", "FastAPI"), ("Postgres 14", "Postgres 17"), ("REST", "gRPC"), ("Redux", "Zustand"), ("Celery", "a Postgres queue"),
           ("setup.py", "pyproject.toml")]

TEMPLATES: dict[str, list[str]] = {
    "tests": ["add tests for {fn} in {file}", "write tests covering the empty-input cases of {thing}",
              "the coverage on {file} is low, add a few tests"],
    "rename": ["rename {old} to {new} everywhere", "move {old} out of {file} into its own module and update the imports",
               "rename the {thing} settings to match the new names"],
    "docs": ["update the README section on {thing}", "add docstrings to the public functions in {file}",
             "fix the typos in the docs for {thing}"],
    "feature": ["add a --dry-run option to the {thing} command", "add pagination to the endpoint behind {thing}",
                "let people export {thing} as JSON as well as CSV", "add a setting to turn off {thing}"],
    "bug": ["fix the bug where {thing} {symptom}", "{thing} {symptom}; find out why and fix it", "people report that {thing} {symptom}"],
    "multi-file": ["split {file} into separate modules for parsing and validation and update every caller",
                   "replace the global config object with dependency injection across the services",
                   "move {thing} behind an interface so we can swap the storage backend; update all the call sites"],
    "concurrency": ["fix the race condition in {thing} when two workers pick the same job", "{thing} deadlocks under load, fix it",
                    "make {thing} safe for concurrent requests"],
    "migration": ["migrate {thing} from {a} to {b}", "migrate the schema for {thing} without downtime", "upgrade us from {a} to {b}"],
    "architecture": ["design the architecture for multi-region {thing}, then implement the first step",
                     "design a plugin system for {thing} and move the built-in handlers onto it",
                     "re-architect {thing} as an event-sourced service, keeping the API the same"],
    "question": ["what's the difference between a list and a tuple?", "how many ounces in a liter?", "who wrote Middlemarch?",
                 "what does HTTP 409 mean?", "is it 'affect' or 'effect' here: the change will ___ everyone"],
    "writing": ["rewrite this announcement so it's shorter and friendlier:", "turn these notes into an email to the team:",
                "proofread this paragraph and fix the grammar:"],
    "explain": ["explain how this contract clause works in plain words:", "summarize this article in five bullet points:"],
    "analysis": ["compare these three vendor proposals and recommend one, with the trade-offs:",
                 "analyze last quarter's churn numbers below and tell me what drives them:",
                 "evaluate this pricing plan against our costs and suggest a strategy:"],
}
# Worded so the rules read them wrong: hard work sold as small, easy work that names a hard topic, analysis without the
# words that mark it, and small asks that happen to use them.
MISLEADING: dict[str, list[str]] = {
    "hard": ["quick fix: {thing} {symptom}", "small thing: {thing} {symptom}, should be a one-liner",
             "can you tidy up {thing}? it {symptom} sometimes", "minor cleanup in {thing} before the release"],
    "easy": ["add tests for the concurrency helpers in {file}", "rename the performance counters in {file} to snake_case",
             "update the security section of the README for {thing}", "document the migration steps for {thing} in the docs"],
    "analysis": ["what should we do about {thing}, given the numbers below?", "here are our options for {thing}; which one?"],
    "chat": ["compare these two subject lines and pick one:", "plan a short team lunch agenda for Friday:"],
}
MISLEADING_SHARE = {"coding-easy": 0.15, "coding-medium": 0.10, "coding-hard": 0.30, "coding-very-hard": 0.30,
                    "chat-short": 0.0, "chat": 0.10, "analysis": 0.25}
_CORRECTIONS = ["that's still not working, {thing} {symptom}", "that's wrong: {thing} still {symptom}", "you broke the tests, revert that",
                "that's not what I asked for", "hmm, can you look at {thing} again?", "no, the output is still off"]

# Tools by client. Claude Code sends about 16; a lighter agent fewer (outcomes.py knows both sets of names).
CLAUDE_CODE_TOOLS = ["Task", "Bash", "Glob", "Grep", "LS", "Read", "Edit", "MultiEdit", "Write", "NotebookEdit", "WebFetch", "TodoWrite",
                     "WebSearch", "BashOutput", "KillShell", "ExitPlanMode"]
SMALL_AGENT_TOOLS = ["shell", "apply_patch", "update_plan", "view_image"]
CLIENTS = (("claude-code", 0.75), ("small-agent", 0.15), ("chat-with-code", 0.10))  # coding tasks' clients
# The API each client calls (the learned router's ``client`` feature): Claude Code Anthropic's Messages, the small agent
# (Codex CLI's tools) the Responses API, people in a chat app OpenAI's Chat Completions.
CLIENT_API = {"claude-code": "anthropic", "small-agent": "responses", "chat-with-code": "openai", "chat": "openai"}
# The reply length each client asks for (max_tokens; 0: none, the gateway gives the reply what's left of the window): Claude
# Code's default for its main loop, a chat app's usual cap. The router keeps that much room in each model's window.
CLIENT_MAX_TOKENS = {"claude-code": 32_000, "small-agent": 0, "chat-with-code": 8_192, "chat": 4_096}
SNIPPETS = ["```python\ndef total(rows):\n    return sum(r.price for r in rows)\n```",
            "```ts\nexport const load = async (id: string) => fetch(`/api/${id}`)\n```",
            "Traceback (most recent call last):\n  File \"app.py\", line 12, in <module>\nKeyError: 'id'"]
TEST_COMMANDS = ["pytest -q", "npm test", "go test ./...", "cargo test"]


def _fill(template: str, rng: random.Random) -> str:
    old, new = rng.choice(_NAMES)
    a, b = rng.choice(_STACKS)
    return template.format(file=rng.choice(_FILES), thing=rng.choice(_THINGS), fn=old, old=old, new=new, symptom=rng.choice(_SYMPTOMS),
                           a=a, b=b)


def opening(cls: TaskClass, category: str, misleading: bool, rng: random.Random) -> str:
    """The person's first prompt for a task of this class (misleading: worded so the rules read it wrong)."""
    if misleading:
        key = ("hard" if cls.name in ("coding-hard", "coding-very-hard") else "easy" if cls.task == "coding"
               else "analysis" if cls.name == "analysis" else "chat")
        return _fill(rng.choice(MISLEADING[key]), rng)
    return _fill(rng.choice(TEMPLATES[category]), rng)


# ------------------------------------------------------------------ the population


@dataclass
class Task:
    """One task: what the person asks, how hard it really is, and the shape of the session that does it."""

    id: str
    cls: str
    task: str  # coding or general: which ability decides it
    category: str
    difficulty: float  # on Epoch's ability scale
    prompt: str
    misleading: bool
    client: str  # claude-code, small-agent, chat-with-code, chat
    tools: list[str]
    turns: int  # turns a successful attempt takes
    start_tokens: int
    growth: tuple[int, int]
    output: tuple[int, int]
    ambiguous: bool  # the first prompt doesn't say enough: the first attempt fails on any model
    start: float  # when it starts (simulated time)
    seed: str

    def rng(self, *what: Any) -> random.Random:
        """A random stream for one purpose on this task, the same for every strategy (common random numbers)."""
        return random.Random(":".join([self.seed, *map(str, what)]))


def population(n: int, seed: int = 1, shares: dict[str, float] | None = None) -> list[Task]:
    """``n`` tasks drawn from ``shares`` (a share per class, MIXES["standard"] by default), reproducible from ``seed``. The
    pauses between turns are drawn when a task runs (Simulator.pause_share), so one population serves every pause share."""
    shares = shares or MIXES["standard"]
    rng = random.Random(f"population:{seed}")
    names = [k for k, v in shares.items() if v > 0]
    weights = [shares[k] for k in names]
    out = []
    for i in range(n):
        cls = CLASSES[rng.choices(names, weights)[0]]
        category = rng.choice(cls.categories)
        misleading = rng.random() < MISLEADING_SHARE.get(cls.name, 0.0)
        if cls.task == "coding":
            client = rng.choices([c for c, _ in CLIENTS], [w for _, w in CLIENTS])[0]
        else:
            client = "chat"
        tools = CLAUDE_CODE_TOOLS if client == "claude-code" else SMALL_AGENT_TOOLS if client == "small-agent" else []
        lo, hi = cls.turns
        turns = rng.randint(lo, hi) if cls.agent and client != "chat-with-code" else rng.randint(1, max(1, min(hi, 6)))
        start = rng.randint(*cls.start) if client != "chat-with-code" else rng.randint(1_500, 6_000)
        text = opening(cls, category, misleading, rng)
        if client == "chat-with-code" and rng.random() < 0.9:
            text += "\n\n" + rng.choice(SNIPPETS)  # people paste the code they mean
        out.append(Task(id=f"t{i:05d}", cls=cls.name, task=cls.task, category=category,
                        difficulty=rng.gauss(cls.mu, cls.sigma), prompt=text, misleading=misleading,
                        client=client, tools=list(tools), turns=turns, start_tokens=start, growth=cls.growth, output=cls.output,
                        ambiguous=rng.random() < AMBIGUOUS and cls.name != "chat-short", start=START + i * 900.0,
                        seed=f"{seed}:{i}"))
    return out


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def chance(ability: float, difficulty: float, noise: float = 0.0) -> float:
    """A model's chance on one task: Epoch's logistic link at one task's slope, plus that model's luck on the task."""
    return sigmoid(ITEM_SLOPE * (ability - difficulty) + noise)


def check_calibration(abilities: dict[str, float], n: int = 4000, seed: int = 0) -> dict[str, dict[str, float]]:
    """For each ability: the simulated success rate of the FrontierSWE-like class and of SWE-bench Verified's mix (the
    first attempt, ambiguity included), next to what Epoch's curves say. Used by the tests and printed in the report."""
    rng = random.Random(f"calibration:{seed}")
    draws = {c: [rng.gauss(CLASSES[c].mu, CLASSES[c].sigma) for _ in range(n)] for c in CODING}
    out = {}
    for name, a in abilities.items():
        rate = {c: sum(chance(a, d) for d in ds) / n for c, ds in draws.items()}
        swe = sum(SWE_MIX[c] * rate[c] for c in CODING) * (1 - AMBIGUOUS)
        out[name] = {"ability": a, "hard": round(rate["coding-hard"], 4),
                     "frontier_swe": round(sigmoid(FRONTIER_SWE[1] * (a - FRONTIER_SWE[0])), 4),
                     "swe_mix": round(swe, 4), "swe_verified": round(sigmoid(SWE_VERIFIED[1] * (a - SWE_VERIFIED[0])), 4)}
    return out


# ------------------------------------------------------------------ the world: the shipped configuration, offline

SIM_BACKENDS = ("anthropic", "openai", "gemini", "xai", "openrouter")  # one per provider's own API, and OpenRouter for the rest
SIM_TEAM = "sim"


def sim_raw(raw: dict | None = None, backends: tuple[str, ...] = SIM_BACKENDS) -> dict:
    """The shipped configuration with these API backends on (no credentials: nothing is ever sent) and every other off."""
    raw = copy.deepcopy(raw if raw is not None else load_raw_config())
    for name, b in (raw.get("backends") or {}).items():
        b["enabled"] = name in backends
        if name in backends:
            b["auth"] = {"type": "none"}
    teams = raw.get("teams") if isinstance(raw.get("teams"), dict) else {}
    raw["teams"] = {**teams, SIM_TEAM: dict((teams or {}).get("default") or {})}
    return raw


def merge(base: dict, extra: dict) -> dict:
    out = dict(base)
    for k, v in extra.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


class World:
    """A configuration the strategies run against, built from a raw document: the real Config, its backends (base Backend
    objects, which can't send anything), and each model's true ability from the quality data."""

    def __init__(self, raw: dict, catalog: dict, config_dir: str = "config", quality_error: float = 0.0, seed: int = 0):
        self.raw, self.catalog = raw, catalog
        self.cfg: Config = build_config(raw, config_dir, prices_data=catalog, quiet=True)
        self.backends = {b.name: Backend(b) for b in self.cfg.backends if b.enabled and b.kind in ("api",)}
        measures = (catalog.get("quality") or {}).get("measures") or {}
        rng = random.Random(f"quality-error:{seed}")
        self.ability: dict[str, dict[str, float]] = {}
        for m in self.cfg.models:
            per = measures.get(m.name) or {}
            if m.kind != "chat" or not any(r.backend in self.backends and r.priced for r in m.routes):
                continue
            err = rng.gauss(0.0, quality_error) if quality_error else 0.0
            ab = {t: float(v["ability"]) + err for t, v in per.items() if isinstance(v, dict) and v.get("ability") is not None}
            if ab:
                self.ability[m.name] = ab

    def router(self) -> Router:
        """A fresh router: speeds, token calibration and pause counts start empty, as on a new gateway."""
        return Router(self.cfg, self.backends)

    def able(self, model: str, task: str) -> float | None:
        per = self.ability.get(model) or {}
        return per.get(task, per.get("general"))


def default_catalog() -> dict:
    return load_catalog(Path(__file__).resolve().parents[2] / "config" / "prices.json") or load_catalog(Path("config/prices.json"))


# ------------------------------------------------------------------ strategies


@dataclass
class Pick:
    """What a strategy chose for a turn: the model and route, the router's estimate for the session from here, the level."""

    model: str
    backend: str
    route: Route
    estimate: float
    floor: str | None = None
    reason: str = ""
    explore: dict | None = None
    floor_reason: str = ""


@dataclass
class Turn:
    """What a strategy sees when it chooses: the request as the gateway would see it, the session, the task's evidence."""

    task: Task
    attempt: int
    index: int  # the turn within the attempt
    body: dict
    headers: dict[str, str]
    session: Session
    prompt_tokens: int
    boundary: bool
    now: float
    state: dict | None  # outcomes.py's state for the task so far
    picks: list[Pick]  # this attempt's picks so far


class Strategy:
    """A way of choosing models, run through the real router. Subclass to plug one in: ``configure`` changes the raw
    configuration (routing.*, the team), ``route`` makes the turn's choice (the router's, by default), and ``escalate``
    may move the rest of an attempt to another level when the task's evidence says it's going badly."""

    def __init__(self, name: str, label: str = "", requested: str = "auto", routing: dict | None = None, team: dict | None = None,
                 headers: dict[str, str] | None = None, explore: bool = False):
        self.name, self.label = name, label or name
        self.requested = requested
        self.routing, self.team, self.extra_headers, self.explore = routing or {}, team or {}, headers or {}, explore

    def configure(self, raw: dict) -> dict:
        raw = merge(raw, {"routing": self.routing}) if self.routing else raw
        if self.team:
            raw = {**raw, "teams": {**raw["teams"], SIM_TEAM: merge(raw["teams"][SIM_TEAM], self.team)}}
        return raw

    def route(self, sim: "Simulator", t: Turn, requested: str | None = None, headers: dict[str, str] | None = None,
              boundary: bool | None = None) -> Pick:
        d = sim.router.decide(requested or self.requested, sim.world.cfg.team(SIM_TEAM), t.body, {**t.headers, **self.extra_headers,
                              **(headers or {})}, t.session if t.session.turns else None, t.prompt_tokens,
                              t.boundary if boundary is None else boundary, {}, t.now, client=CLIENT_API[t.task.client],
                              explore=self.explore)
        c = d.candidates[0]
        return Pick(c.model.name, c.backend.name, c.route, c.cost, str(d.floor) if d.floor else None, d.reason, d.explore,
                    d.floor.reason if d.floor else "")

    def escalate(self, sim: "Simulator", t: Turn) -> Pick | None:
        """A pick for the rest of the attempt when the evidence says it's failing, or None (the default: never)."""
        return None


class EscalateOnEvidence(Strategy):
    """An example of an in-task cascade: route as the threshold does, and once the task's tool results show it's going
    wrong (a fix loop: ``outcomes.LOOP`` edits of one file, each right after a failed run), move the rest of the attempt
    one level up, once. The move pays to rebuild the cache on the new model. It's here to exercise the harness's
    escalation accounting, not as a recommendation."""

    def escalate(self, sim: "Simulator", t: Turn) -> Pick | None:
        s, picks = t.state or {}, t.picks
        if not picks or any(p.reason.startswith("escalated") for p in picks):
            return None
        if s.get("fix_loop", 0) < outcomes.LOOP:
            return None
        level = int((picks[0].floor or f"{t.task.task}:3").split(":")[1])
        if level >= 5:
            return None
        pick = self.route(sim, t, requested="auto", headers={"x-route-floor": f"{t.task.task}:{level + 1}"}, boundary=True)
        pick.reason = f"escalated to {t.task.task}:{level + 1} on evidence ({pick.reason})"
        return pick


class RouterCascade(Strategy):
    """expected_cost with the router's own cascade (decision 62, ``Router._escalate``), kept as the gateway keeps it
    (proxy.py's ``_handle``, ``follow_up`` and ``update_session``): the session's tool results' state is folded from each
    request (outcomes.update), and at a human message it records whether the prompt says the last answer was wrong
    (outcomes.correction); the message carries the task on only where ``Router.carries_on`` says so; the router may
    escalate after failures in a row (``cascade="now"``) or at that message (``"human"``); and the task's chance at its
    start, and the reset after an escalation, are ``success.track``'s."""

    def route(self, sim: "Simulator", t: Turn, requested: str | None = None, headers: dict[str, str] | None = None,
              boundary: bool | None = None) -> Pick:
        s, router = t.session, sim.router
        hdrs = {**t.headers, **self.extra_headers, **(headers or {})}
        team = sim.world.cfg.team(SIM_TEAM)
        boundary = t.boundary if boundary is None else boundary
        new = s.turns == 0
        if new:
            s.task = s.outcome = None
        elif boundary and t.index == 0:  # a human message (compaction is a boundary too, but not one the person wrote)
            s.outcome = {**(s.outcome or outcomes.new_state()), "correction": outcomes.correction(t.body)}
        elif not boundary:
            s.outcome, _ = outcomes.update(s.outcome, t.body, 1, t.prompt_tokens)
        continuing = (boundary and t.index == 0 and not new and bool(s.model) and s.task is not None
                      and router.carries_on(s, t.body, hdrs, t.prompt_tokens))
        if boundary and not continuing:
            s.task = s.outcome = None
        cascade = "human" if continuing else "now" if not boundary else None
        d = router.decide(requested or self.requested, team, t.body, hdrs, None if new else s, t.prompt_tokens, boundary and not continuing,
                          {}, t.now, client=CLIENT_API[t.task.client], explore=self.explore, cascade=cascade)
        c = d.candidates[0]
        if d.escalation is not None and s.task is not None:
            s.task = {**s.task, "escalations": d.escalation["n"]}
        if (s.task is None and d.auto) or (d.escalation is not None and s.task is not None):
            s.task, s.outcome = track(s.task, s.outcome, c.model.name, c.chance, d.predicted, d.escalation is not None)
        return Pick(c.model.name, c.backend.name, c.route, c.cost, str(d.floor) if d.floor else None, d.reason, d.explore,
                    d.floor.reason if d.floor else "")


def cheapest_model(world: World, task: str = "coding") -> str:
    """The model with the cheapest route for a typical agent turn (30,000 tokens, mostly cached), among those with a known
    ability for the task type: what "always the cheapest" sends everything to."""
    r = world.router()
    best: tuple[float, str] | None = None
    for m in world.cfg.models:
        if world.able(m.name, task) is None:
            continue
        for route in m.routes:
            b = world.backends.get(route.backend)
            if b is None or not route.priced:
                continue
            c = r.estimate(route.cost_price().at(30_000), b, 30_000, None, False, 3, route.model)
            if best is None or c < best[0]:
                best = (c, m.name)
    assert best is not None, "no priced model with a known ability"
    return best[1]


def builtin_strategies(world: World) -> list[Strategy]:
    """Today's strategies: named models, the cheapest model, every `model_choice` the configuration accepts (so a new one
    on main is picked up without changing this file), and the example cascade. The oracle is the simulator's own."""
    out = [Strategy("always-opus", "Always Claude Opus", requested="claude-opus"),
           Strategy("always-sonnet", "Always Claude Sonnet", requested="claude-sonnet"),
           Strategy("always-cheapest", f"Always the cheapest ({cheapest_model(world)})", requested=cheapest_model(world))]
    for choice in get_args(ModelChoice):
        out.append(Strategy(f"auto-{choice.replace('_', '-')}", f"auto, model_choice {choice}", team={"model_choice": choice}))
    out.append(EscalateOnEvidence("auto-threshold-escalate", "auto, threshold, escalating on evidence (example)",
                                  team={"model_choice": "threshold"}))
    out.append(RouterCascade("auto-expected-cost-cascade", "auto, expected_cost, with its cascade (one escalation)",
                             team={"model_choice": "expected_cost"}, routing={"cascade": {"max_escalations": 1}}))
    return out


def load_plugin(spec: str) -> list[Strategy]:
    """Strategies from ``module:attribute``: a Strategy, a list of them, or a callable returning either. How a strategy that
    isn't on main yet (routing v2, a new classifier) is scored before it lands."""
    module, _, attr = spec.partition(":")
    obj = getattr(importlib.import_module(module), attr or "STRATEGIES")
    obj = obj() if callable(obj) and not isinstance(obj, Strategy) else obj
    items = obj if isinstance(obj, (list, tuple)) else [obj]
    if not all(isinstance(s, Strategy) for s in items):
        raise TypeError(f"{spec} must give a Strategy or a list of them")
    return list(items)


def entry_point_strategies() -> list[Strategy]:
    """Strategies other packages register under the ``meritorder.routing_strategies`` entry point group."""
    from importlib.metadata import entry_points

    out: list[Strategy] = []
    for ep in entry_points(group="meritorder.routing_strategies"):
        out += load_plugin(ep.value)
    return out


# ------------------------------------------------------------------ execution


@dataclass
class Attempt:
    """One try at a task: one human prompt and the agent's turns after it, until it ends."""

    success: bool
    models: list[str]
    backends: list[str]
    floor: str | None
    cost: float
    estimate: float  # the router's estimate at the start, for the session from there
    turns: int
    rebuilds: int  # turns that wrote most of the prompt again (a cold cache: a pause, a move, a compaction)
    escalations: list[list[str]]  # [from, to] models moved to within the attempt
    usage: dict[str, int]
    coding: dict  # outcomes.summary of the task's tool results
    correction: str | None = None  # the rule the person's next prompt matched (outcomes.correction), after a failure
    explore: dict | None = None
    reason: str = ""
    # Each task start within the attempt, as the gateway records one (the first prompt, and each compaction after it): when,
    # at which level, the exploration arm it drew, its model and what it cost until the next.
    segments: list[dict] = field(default_factory=list)


@dataclass
class Result:
    """A task as one strategy ran it."""

    task: str
    cls: str
    strategy: str
    attempts: list[Attempt]
    finished: bool
    ledger: list[dict] = field(default_factory=list)  # per-turn rows, when the run records them (export_rows)

    @property
    def cost(self) -> float:
        return sum(a.cost for a in self.attempts)

    @property
    def failures(self) -> int:
        return sum(1 for a in self.attempts if not a.success)


class Simulator:
    """Runs a population through strategies. One world (a configuration) per distinct ``Strategy.configure`` result."""

    def __init__(self, raw: dict | None = None, catalog: dict | None = None, pause_share: float = PAUSE_SHARE,
                 human_cost: float = HUMAN_COST, max_attempts: int = MAX_ATTEMPTS, quality_error: float = 0.0, seed: int = 0,
                 config_dir: str | None = None):
        self.raw = sim_raw(raw)
        self.catalog = catalog if catalog is not None else default_catalog()
        self.pause_share, self.human_cost, self.max_attempts = pause_share, human_cost, max_attempts
        self.quality_error, self.seed = quality_error, seed
        self.config_dir = config_dir or str(Path(__file__).resolve().parents[2] / "config")
        self._worlds: dict[str, World] = {}
        self.world = self.world_for(Strategy("base"))
        self.router = self.world.router()

    def world_for(self, s: Strategy) -> World:
        raw = s.configure(self.raw)
        key = hashlib.sha256(json.dumps(raw, sort_keys=True, default=str).encode()).hexdigest()
        if key not in self._worlds:
            self._worlds[key] = World(raw, self.catalog, self.config_dir, self.quality_error, self.seed)
        return self._worlds[key]

    # ---- one task

    def _gap(self, task: Task, attempt: int, i: int, output: int, rng: random.Random) -> float:
        """Seconds before a turn: the reply being written and the tool running, and sometimes a person stepping away."""
        gap = output / 60.0 + rng.uniform(2.0, 40.0)
        if task.client in ("chat", "chat-with-code"):
            gap += rng.uniform(20.0, 240.0)  # a person reads and types every turn
        u = rng.random()
        if u < self.pause_share:
            gap += rng.uniform(300.0, 3_600.0)
        elif u < self.pause_share + LONG_PAUSE:
            gap += rng.uniform(3_600.0, 7_200.0)
        return gap

    def _fate(self, task: Task, attempt: int, model: str) -> bool:
        """Whether this model succeeds on this attempt: the same draw for every model and strategy (stronger models
        succeed whenever weaker ones do, give or take their luck on the task)."""
        if attempt == 0 and task.ambiguous:
            return False
        ability = self.world.able(model, task.task)
        if ability is None:
            return False
        u = task.rng("attempt", attempt).random()
        noise = task.rng("noise", model).gauss(0.0, TASK_NOISE)
        return u < chance(ability, task.difficulty, noise)

    def _usage(self, cache: dict, backend: Backend, route: Route, prompt: int, output: int, now: float, later: bool) -> tuple[Usage, int]:
        """The turn's usage as the provider bills it, and the tokens it read from cache. Explicit caching (Claude) writes
        what's new at the 5-minute price, or the hour's on a backend set to an hour once the session has come back (decision
        53); automatic caching (everyone else) bills what's new at the input price. A cache lasts its lifetime from its
        last use; another model, or another backend, has its own."""
        explicit = backend.format == "anthropic" or needs_cache_marker(backend.cfg, route.model)
        ttl = (HOUR if later and backend.cfg.cache_ttl_seconds >= HOUR else FIVE_MINUTES) if explicit else backend.cfg.cache_ttl_seconds
        key = (backend.name, route.model)
        held, until = cache.get(key, (0, 0.0))
        hit = held if until > now and held <= prompt else 0
        new = prompt - hit
        if explicit:
            u = Usage(input_tokens=0, cache_read_tokens=hit, cache_write_tokens=new, output_tokens=output,
                      cache_write_1h_tokens=new if ttl >= HOUR else 0)
        else:
            u = Usage(input_tokens=new, cache_read_tokens=hit, output_tokens=output)
        cache[key] = (prompt, now + ttl)
        return u, hit

    def _messages(self, task: Task, attempt: int, i: int, fate: bool, n: int, last_file: list[str], rng: random.Random) -> list[dict]:
        """The agent's last tool call and its result, consistent with how the attempt ends: tests that pass at the end
        (success, or a wrong fix that passes them), or fail more and more, edits to one file after failed runs (a fix loop)."""
        if not task.tools:
            return [{"role": "assistant", "content": "Here's the change."}]
        shell = "Bash" if task.client == "claude-code" else "shell"
        edit = "Edit" if task.client == "claude-code" else "apply_patch"
        cid = f"toolu_{task.id}_{attempt}_{i}"
        silent = task.rng("silent", attempt).random() < 0.4  # a failure that passes the tests: the person finds it later
        phase = i / max(n - 1, 1)
        if phase < 0.2:
            call = {"name": "Read" if task.client == "claude-code" else shell, "input": {"file_path": rng.choice(_FILES)}
                    if task.client == "claude-code" else {"command": f"cat {rng.choice(_FILES)}"}}
            result, error = "     1\timport os\n     2\t...", False
        elif i % 2 == 0 or i == n - 1:
            cmd = rng.choice(TEST_COMMANDS)
            call = {"name": shell, "input": {"command": cmd}}
            last = i == n - 1
            if fate or silent:
                ok = last or rng.random() < (0.35 + 0.6 * phase)
            else:
                ok = not last and rng.random() < max(0.05, 0.5 - phase)
            result = f"{rng.randint(5, 80)} passed in {rng.uniform(0.5, 9):.2f}s" if ok else \
                f"FAILED tests/test_core.py::test_case - AssertionError\n{rng.randint(1, 4)} failed, {rng.randint(3, 60)} passed"
            error = not ok
        else:
            if not last_file or (fate and rng.random() < 0.5):
                last_file[:] = [rng.choice(_FILES)]
            path = last_file[0]
            call = {"name": edit, "input": {"file_path": path, "old_string": "a", "new_string": "b"} if edit == "Edit"
                    else {"input": f"*** Begin Patch\n*** Update File: {path}\n@@\n-a\n+b\n*** End Patch"}}
            result, error = "The file has been updated.", False
        return [{"role": "assistant", "content": [{"type": "tool_use", "id": cid, **call}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": cid, "content": result, "is_error": error}]}]

    @staticmethod
    def _body(task: Task, msgs: list[dict], tools: list[dict]) -> dict:
        """A request as the task's client sends it: its messages, its tools, and the reply length it asks for."""
        body: dict[str, Any] = {"messages": msgs}
        if tools:
            body["tools"] = tools
        if CLIENT_MAX_TOKENS.get(task.client):
            body["max_tokens"] = CLIENT_MAX_TOKENS[task.client]
        return body

    def run_task(self, s: Strategy, task: Task, record: bool = False) -> Result:
        world = self.world
        session = Session(id=f"sim-{task.id}-{s.name}")
        headers = {"user-agent": "claude-cli/2.1.30 (external, cli)" if task.client == "claude-code" else
                   "pi/0.9" if task.client == "small-agent" else "openai-python/2.8.0", "x-session-id": session.id}
        tools = [{"name": n, "description": f"the {n} tool", "input_schema": {"type": "object"}} for n in task.tools]
        cache: dict = {}
        now, prompt, state = task.start, task.start_tokens, None
        attempts: list[Attempt] = []
        ledger: list[dict] = []
        first: dict = {"role": "user", "content": task.prompt}
        history: list[dict] = [first]
        for k in range(self.max_attempts):
            rng = task.rng("shape", k)  # outputs, growth, pauses: the same for every strategy
            evidence = task.rng("evidence", k)
            picks: list[Pick] = []
            if k:
                last = attempts[-1]
                gap = rng.uniform(600.0, 7_200.0) if last.coding.get("last_test") == "pass" or not task.tools else rng.uniform(60.0, 900.0)
                text = _fill(task.rng("correction", k).choice(_CORRECTIONS), task.rng("fill", k))
                history = [first, {"role": "assistant", "content": "Done."}, {"role": "user", "content": text}]
                body = self._body(task, history, tools)
                last.correction = outcomes.correction(body)
                now += gap
                self.router.pauses.observe(SIM_TEAM, gap, now)
            fate, n = None, task.turns
            usage = {"input": 0, "cache_read": 0, "cache_write": 0, "output": 0}
            cost = estimate = 0.0
            rebuilds, escalations, turns = 0, [], 0
            last_file: list[str] = []
            segments: list[dict] = []
            state = None
            i = 0
            while i < n:
                boundary = i == 0
                msgs = history if i == 0 else [history[0], *self._messages(task, k, i - 1, bool(fate), n, last_file, evidence)]
                body = self._body(task, msgs, tools)
                if i and task.tools:
                    state, _ = outcomes.update(state, body, 1, prompt)
                if prompt > COMPACT_AT:  # the client compacts: history rewritten, every cache gone, and a new task start
                    prompt, session.warm_until, boundary = COMPACT_TO, {}, True
                t = Turn(task, k, i, body, headers, session, prompt, boundary, now, state, picks)
                pick = s.escalate(self, t) if i else None
                if pick is not None:
                    escalations.append([picks[-1].model, pick.model])
                    fate = self._fate(task, k, pick.model)
                    n = max(i + 3, i + (n - i) // 2) if fate else n
                else:
                    try:
                        pick = s.route(self, t)
                    except RoutingError:
                        break
                    if picks and pick.model != picks[-1].model:
                        escalations.append([picks[-1].model, pick.model])  # the router moved it (outgrown, unavailable)
                        fate = self._fate(task, k, pick.model)
                if fate is None:
                    fate = self._fate(task, k, pick.model)
                    if not fate and task.tools:
                        n = int(n * (1.5 if task.rng("silent", k).random() >= 0.4 else 0.8)) or 1
                    estimate = pick.estimate
                picks.append(pick)
                if boundary:
                    segments.append({"ts": now, "turn": i, "floor": pick.floor, "explore": pick.explore, "model": pick.model, "cost": 0.0})
                out = rng.randint(*task.output)
                backend = world.backends[pick.backend]
                u, hit = self._usage(cache, backend, pick.route, prompt, out, now, session.turns > 0)
                c = priced(pick.route.cost_price(), u)
                if i and prompt > 4_000 and hit < prompt / 2:
                    rebuilds += 1
                cost += c
                segments[-1]["cost"] += c
                turns += 1
                for key, v in (("input", u.input_tokens), ("cache_read", u.cache_read_tokens), ("cache_write", u.cache_write_tokens),
                               ("output", u.output_tokens)):
                    usage[key] += v
                if record:
                    ledger.append({"ts": now, "attempt": k, "segment": len(segments) - 1, "model": pick.model, "backend": pick.backend,
                                   "floor": pick.floor, "cost": c, "usage": asdict(u)})
                self._update_session(session, pick, backend, prompt, len(msgs), now)
                prompt += out + (rng.randint(*task.growth) if task.growth[1] else 0)
                gap = self._gap(task, k, i, out, rng)
                now += gap
                if i + 1 < n:
                    self.router.pauses.observe(SIM_TEAM, gap, now)
                i += 1
            if not picks:
                break
            coding = outcomes.summary(state) if state else {}
            attempts.append(Attempt(success=bool(fate) and turns >= n, models=list(dict.fromkeys(p.model for p in picks)),
                                    backends=list(dict.fromkeys(p.backend for p in picks)), floor=picks[0].floor, cost=cost,
                                    estimate=estimate, turns=turns, rebuilds=rebuilds, escalations=escalations, usage=usage,
                                    coding=coding, explore=picks[0].explore, reason=picks[0].reason, segments=segments))
            if attempts[-1].success:
                break
        return Result(task.id, task.cls, s.name, attempts, bool(attempts) and attempts[-1].success, ledger)

    @staticmethod
    def _update_session(s: Session, pick: Pick, backend: Backend, prompt: int, msgs: int, now: float) -> None:
        """As Gateway.update_session keeps it: the model, backend, level, turns, size, and how long the cache lasts there."""
        s.model, s.backend = pick.model, pick.backend
        if pick.floor:
            s.floor, s.floor_reason = pick.floor, pick.floor_reason
        s.turns += 1
        s.last_seen = now
        s.msg_count = msgs
        s.prefix_tokens = prompt
        explicit = backend.format == "anthropic" or needs_cache_marker(backend.cfg, pick.route.model)
        ttl = (HOUR if s.turns > 1 and backend.cfg.cache_ttl_seconds >= HOUR else FIVE_MINUTES) if explicit else backend.cfg.cache_ttl_seconds
        s.warm_until[backend.name] = now + ttl
        s.warm_until = {k: v for k, v in s.warm_until.items() if v > now}

    # ---- a whole run

    def run(self, s: Strategy, tasks: list[Task], record: bool = False) -> list[Result]:
        """Every task through one strategy, with a fresh router (nothing measured yet) on the strategy's configuration."""
        self.world = self.world_for(s)
        self.router = self.world.router()
        self.router.rand = random.Random(f"explore:{self.seed}:{s.name}").random
        return [self.run_task(s, t, record) for t in tasks]

    def candidates(self, task: str) -> list[str]:
        """The models the oracle tries: every model with a priced route here and a known ability for the task type."""
        return [m for m, per in self.world.ability.items() if task in per or "general" in per]

    def oracle(self, tasks: list[Task]) -> tuple[list[Result], dict[str, dict[str, tuple[bool, float]]]]:
        """For each task, every candidate model named for the whole task; the oracle takes the one with the lowest cost
        including the person's time. Also returns each model's first attempt per task (did it succeed, what it cost), for
        the "too high" and "too low" rates."""
        self.world = self.world_for(Strategy("base"))
        firsts: dict[str, dict[str, tuple[bool, float]]] = defaultdict(dict)
        best: dict[str, Result] = {}
        for model in sorted({m for t in tasks for m in self.candidates(t.task)}):
            s = Strategy(f"oracle:{model}", requested=model)
            self.router = self.world.router()
            for t in tasks:
                if model not in self.candidates(t.task):
                    continue
                r = self.run_task(s, t)
                if not r.attempts:
                    continue
                firsts[t.id][model] = (r.attempts[0].success, r.attempts[0].cost)
                total = r.cost + self.human_cost * r.failures
                if t.id not in best or total < best[t.id].cost + self.human_cost * best[t.id].failures - 1e-12:
                    best[t.id] = r
        out = []
        for t in tasks:
            r = best[t.id]
            out.append(Result(r.task, r.cls, "oracle", r.attempts, r.finished))
        return out, dict(firsts)


# ------------------------------------------------------------------ the scorecard


def _ratio(a: float, b: float) -> float | None:
    return round(a / b, 6) if b else None


def summarize(results: list[Result], human_cost: float, firsts: dict | None = None, ability: Callable[[str, str], float | None] | None = None,
              task_of: dict[str, str] | None = None) -> dict:
    """The scorecard's numbers for one strategy over some tasks."""
    n = len(results)
    cost = sum(r.cost for r in results)
    done = sum(1 for r in results if r.finished)
    fails = sum(r.failures for r in results)
    human = human_cost * fails
    first_ok = sum(1 for r in results if r.attempts and r.attempts[0].success)
    esc_tasks = helped = esc = 0
    high = low = judged = 0
    est = actual = 0.0
    levels: dict[str, int] = defaultdict(int)
    models: dict[str, float] = defaultdict(float)
    rebuilds = turns = 0
    for r in results:
        any_esc = False
        prev_best: float | None = None
        for a in r.attempts:
            task = (task_of or {}).get(r.task, "coding")
            abil = [ability(m, task) or 0.0 for m in a.models] if ability else []
            moved_up = bool(a.escalations) and (not ability or any((ability(b, task) or 0) > (ability(f, task) or 0) for f, b in a.escalations))
            retry_up = prev_best is not None and abil and max(abil) > prev_best
            if moved_up or retry_up:
                esc += 1
                any_esc = True
                helped += 1 if a.success else 0
            if abil:
                prev_best = max(abil + ([prev_best] if prev_best is not None else []))
            rebuilds += a.rebuilds
            turns += a.turns
            for m in a.models:
                models[m] += a.cost / len(a.models)
        esc_tasks += 1 if any_esc else 0
        if r.attempts:
            a0 = r.attempts[0]
            levels[a0.floor or "named"] += 1
            est += a0.estimate
            actual += a0.cost
            f = (firsts or {}).get(r.task)
            if f:
                judged += 1
                cheaper = any(ok and c < a0.cost * 0.9 for m, (ok, c) in f.items())
                if a0.success and cheaper:
                    high += 1
                if not a0.success and any(ok for ok, _ in f.values()):
                    low += 1
    return {
        "tasks": n, "cost": round(cost, 4), "success_rate": _ratio(done, n), "first_try_success": _ratio(first_ok, n),
        "finished": done, "failed_attempts": fails, "cost_per_task": _ratio(cost, n), "cost_per_finished": _ratio(cost, done),
        "human_cost": round(human, 2), "total_with_human": round(cost + human, 4),
        "total_per_finished_with_human": _ratio(cost + human, done),
        "escalation_rate": _ratio(esc_tasks, n), "escalations": esc, "escalations_helped": _ratio(helped, esc),
        "too_high": _ratio(high, judged) if firsts else None, "too_low": _ratio(low, judged) if firsts else None,
        "estimate_ratio": _ratio(est, actual), "rebuilds_per_100_turns": round(100 * rebuilds / turns, 2) if turns else None,
        "levels": dict(sorted(levels.items())), "spend_by_model": {m: round(v, 4) for m, v in sorted(models.items(), key=lambda kv: -kv[1])[:6]},
    }


def scorecard(runs: dict[str, list[Result]], tasks: list[Task], human_cost: float, firsts: dict | None = None,
              ability: Callable[[str, str], float | None] | None = None, labels: dict[str, str] | None = None) -> dict:
    """Per strategy, overall and per task class, with savings against always-Sonnet and always-Opus on the same tasks."""
    task_of = {t.id: t.task for t in tasks}
    classes = [c for c in CLASSES if any(t.cls == c for t in tasks)]
    out: dict[str, Any] = {"strategies": {}, "order": list(runs), "classes": classes, "tasks": len(tasks), "human_cost": human_cost}
    for name, results in runs.items():
        overall = summarize(results, human_cost, firsts, ability, task_of)
        per = {c: summarize([r for r in results if r.cls == c], human_cost, firsts, ability, task_of) for c in classes}
        out["strategies"][name] = {"label": (labels or {}).get(name, name), "overall": overall, "by_class": per}
    for name, row in out["strategies"].items():
        for ref in ("always-sonnet", "always-opus"):
            if ref not in out["strategies"]:
                continue
            for scope in ["overall", *classes]:
                mine = row["overall"] if scope == "overall" else row["by_class"][scope]
                base = out["strategies"][ref]["overall"] if scope == "overall" else out["strategies"][ref]["by_class"][scope]
                key = ref.split("-")[1]
                mine[f"savings_vs_{key}"] = round(1 - mine["cost"] / base["cost"], 4) if base["cost"] else None
                mine[f"savings_vs_{key}_with_human"] = (round(1 - mine["total_with_human"] / base["total_with_human"], 4)
                                                        if base["total_with_human"] else None)
    return out


def intervals(runs: dict[str, list[Result]], human_cost: float, reps: int = 400, seed: int = 0, ref: str = "always-sonnet") -> dict:
    """90% intervals from a paired bootstrap over tasks (every strategy resampled on the same tasks): success, cost per
    finished task, model spend alone and in total counting failed attempts, and the saving against ``ref``. A few long, hard tasks carry much of the
    spend, so the totals move more between populations than the success rates do."""
    names = list(runs)
    n = len(next(iter(runs.values()), []))
    if not n:
        return {}
    cols = {s: [(r.cost, r.failures, r.finished) for r in runs[s]] for s in names}
    rng = random.Random(f"bootstrap:{seed}")
    draws: dict[str, dict[str, list[float]]] = {s: defaultdict(list) for s in names}
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]
        totals = {}
        for s in names:
            c = sum(cols[s][i][0] for i in idx)
            f = sum(cols[s][i][1] for i in idx)
            d = sum(cols[s][i][2] for i in idx) or 1
            totals[s] = c
            draws[s]["success_rate"].append(d / n)
            draws[s]["cost_per_finished"].append(c / d)
            draws[s]["total_per_finished_with_human"].append((c + human_cost * f) / d)
        for s in names:
            if ref in totals and totals[ref]:
                draws[s]["savings_vs_" + ref.split("-")[-1]].append(1 - totals[s] / totals[ref])

    def band(xs: list[float]) -> list[float]:
        xs = sorted(xs)
        return [round(xs[int(0.05 * (len(xs) - 1))], 6), round(xs[int(0.95 * (len(xs) - 1))], 6)]

    return {s: {k: band(v) for k, v in draws[s].items()} for s in names}


def human_sensitivity(runs: dict[str, list[Result]], costs: tuple[float, ...] = (0.0, 5.0, 15.0, 50.0)) -> dict:
    """Cost per finished task including the person's time, at other prices of that time (no new simulation needed)."""
    out = {}
    for name, results in runs.items():
        done = sum(1 for r in results if r.finished) or 1
        cost, fails = sum(r.cost for r in results), sum(r.failures for r in results)
        out[name] = {str(h): round((cost + h * fails) / done, 4) for h in costs}
    return out


# ------------------------------------------------------------------ running it


@dataclass
class Settings:
    tasks: int = 1000
    seed: int = 1
    mix: str = "standard"
    human_cost: float = HUMAN_COST
    pause_share: float = PAUSE_SHARE
    max_attempts: int = MAX_ATTEMPTS
    quality_error: float = 0.0
    oracle: bool = True
    sensitivity: bool = True
    sensitivity_tasks: int = 400
    strategies: list[str] | None = None  # names to run (default: all)


def shares_of(spec: str) -> dict[str, float]:
    if spec in MIXES:
        return MIXES[spec]
    shares = json.loads(spec)
    unknown = set(shares) - set(CLASSES)
    if unknown:
        raise ValueError(f"unknown task classes {sorted(unknown)}; use {', '.join(CLASSES)}")
    return {k: float(v) for k, v in shares.items()}


def evaluate(settings: Settings, plugins: list[Strategy] | None = None, raw: dict | None = None, catalog: dict | None = None,
             progress: Callable[[str], None] | None = None) -> dict:
    """The whole evaluation: the standard run with every strategy (and the oracle), then the sensitivity runs."""
    say = progress or (lambda _m: None)
    sim = Simulator(raw, catalog, settings.pause_share, settings.human_cost, settings.max_attempts, settings.quality_error, settings.seed)
    strategies = builtin_strategies(sim.world) + list(plugins or [])
    if settings.strategies:
        strategies = [s for s in strategies if s.name in settings.strategies]
    tasks = population(settings.tasks, settings.seed, shares_of(settings.mix))
    runs: dict[str, list[Result]] = {}
    for s in strategies:
        say(f"running {s.name} on {len(tasks)} tasks")
        runs[s.name] = sim.run(s, tasks)
    firsts = None
    if settings.oracle:
        say("running the oracle (every model on every task)")
        runs["oracle"], firsts = sim.oracle(tasks)
    sim.world = sim.world_for(Strategy("base"))
    labels = {s.name: s.label for s in strategies} | {"oracle": "Oracle: the cheapest model that succeeds (lower bound)"}
    card = scorecard(runs, tasks, settings.human_cost, firsts, sim.world.able, labels)
    card["settings"] = asdict(settings)
    card["human_sensitivity"] = human_sensitivity(runs)
    card["intervals"] = intervals(runs, settings.human_cost, seed=settings.seed)
    card["calibration"] = check_calibration({m: per["coding"] for m, per in sim.world.ability.items() if "coding" in per})
    card["population"] = {"classes": {c: sum(1 for t in tasks if t.cls == c) for c in CLASSES},
                          "misleading": sum(1 for t in tasks if t.misleading), "ambiguous": sum(1 for t in tasks if t.ambiguous)}
    card["config"] = {"backends": sorted(sim.world.backends), "prices_updated": sim.catalog.get("updated"),
                      "quality_updated": (sim.catalog.get("quality") or {}).get("updated")}
    if settings.sensitivity:
        sens: dict[str, dict] = {}
        main = [s for s in strategies if s.name in ("always-sonnet", "auto-threshold", "auto-cost-per-success", "auto-expected-cost")] or strategies[:3]
        small = settings.sensitivity_tasks
        variants = [("mix", m, {"mix": m}) for m in ("easier", "standard", "harder")]
        variants += [("pause_share", str(p), {"pause_share": p}) for p in (0.0, 0.1, 0.3)]
        variants += [("quality_error", str(q), {"quality_error": q}) for q in (0.0, 3.0)]
        for axis, value, change in variants:
            say(f"sensitivity: {axis} = {value}")
            ps = change.get("pause_share", settings.pause_share)
            sub = Simulator(raw, catalog, ps, settings.human_cost, settings.max_attempts, change.get("quality_error", settings.quality_error),
                            settings.seed)
            ts = population(small, settings.seed + 1, shares_of(change.get("mix", settings.mix)))
            rr = {s.name: sub.run(s, ts) for s in main}
            sub.world = sub.world_for(Strategy("base"))
            sc = scorecard(rr, ts, settings.human_cost, None, sub.world.able)
            sens.setdefault(axis, {})[value] = {n: {k: v["overall"][k] for k in ("cost_per_finished", "success_rate",
                                                                                   "total_per_finished_with_human")}
                                                for n, v in sc["strategies"].items()}
        card["sensitivity"] = sens
    return card


# ------------------------------------------------------------------ reports

# Per finished task: model spend, time lost to failed attempts (failed attempts x the price per failed attempt) and the
# total counting failed attempts. "time_lost_per_finished" is derived (total - model spend), not stored in the card.
_COLUMNS = [("success_rate", "Success", "pct"), ("cost_per_finished", "Model spend /finished", "usd"),
            ("time_lost_per_finished", "Time lost /finished", "usd"), ("total_per_finished_with_human", "Total /finished, counting failed attempts", "usd"),
            ("cost", "Model spend, all", "usd"), ("human_cost", "Time lost to failed attempts, all", "usd0"),
            ("escalation_rate", "Escalated", "pct"), ("escalations_helped", "Escalation helped", "pct"),
            ("too_high", "Too high", "pct"), ("too_low", "Too low", "pct"), ("savings_vs_sonnet", "Saves vs Sonnet (model spend)", "pct"),
            ("savings_vs_opus", "Saves vs Opus (model spend)", "pct"), ("savings_vs_sonnet_with_human", "Saves vs Sonnet in total", "pct")]
EXPLAIN = ("A failed attempt costs someone time: noticing it, asking again, waiting, cleaning up; here that time has a price per "
           "failed attempt, an assumption you can change (`--human-cost`). Totals count model spend plus failed attempts times that "
           "price: they're what the work costs, not time saved.")


def _cell(o: dict, key: str, kind: str) -> str:
    if key == "time_lost_per_finished":
        t, c = o.get("total_per_finished_with_human"), o.get("cost_per_finished")
        return fmt(None if t is None or c is None else t - c, kind)
    return fmt(o.get(key), kind)


def fmt(v: Any, kind: str) -> str:
    if v is None:
        return "–"
    if kind == "pct":
        return f"{v * 100:.1f}%"
    if kind == "usd0":
        return f"${v:,.0f}"
    if v and abs(v) < 0.01:
        return f"${v:.2g}"
    return f"${v:,.4f}" if abs(v) < 1 else f"${v:,.2f}"


def markdown(card: dict) -> str:
    s = card["settings"]
    lines = ["# Routing scorecard", "",
             f"{card['tasks']} simulated tasks (mix `{s['mix']}`, seed {s['seed']}), a price of ${card['human_cost']:g} per failed attempt, "
             f"{s['pause_share']:.0%} of later turns after a 5-60 minute pause, up to {s['max_attempts']} attempts per task. "
             f"Prices and quality scores from prices.json ({card['config'].get('prices_updated') or 'unknown'}). "
             "No model was called: outcomes are drawn from each model's published ability against each task's difficulty "
             "(docs/03-routing.md#how-we-measure-routing).", "", EXPLAIN, "",
             "## All tasks", "", "| Strategy | " + " | ".join(h for _, h, _ in _COLUMNS) + " |", "|---" * (len(_COLUMNS) + 1) + "|"]
    order = [n for n in card.get("order") or card["strategies"] if n in card["strategies"]]
    for name in order:
        row, o = card["strategies"][name], card["strategies"][name]["overall"]
        lines.append(f"| {row['label']} | " + " | ".join(_cell(o, k, kind) for k, _, kind in _COLUMNS) + " |")
    lines += ["", "## By task class", ""]
    short = [("success_rate", "Success", "pct"), ("cost_per_finished", "Model spend /finished", "usd"),
             ("time_lost_per_finished", "Time lost /finished", "usd"), ("total_per_finished_with_human", "Total /finished, counting failed attempts", "usd"), ("too_high", "Too high", "pct"), ("too_low", "Too low", "pct")]
    for c in card["classes"]:
        n = next(iter(card["strategies"].values()))["by_class"][c]["tasks"]
        lines += [f"### {c} ({n} tasks)", "", "| Strategy | " + " | ".join(h for _, h, _ in short) + " | Levels |", "|---" * (len(short) + 2) + "|"]
        for name in order:
            row = card["strategies"][name]
            o = row["by_class"][c]
            levels = ", ".join(f"{k} {v}" for k, v in o["levels"].items())
            lines.append(f"| {row['label']} | " + " | ".join(_cell(o, k, kind) for k, _, kind in short) + f" | {levels} |")
        lines.append("")
    iv = card.get("intervals") or {}
    if iv:
        lines += ["## How sure", "", "90% intervals from a paired bootstrap over the tasks (400 resamples). A few long, hard tasks carry much "
                  "of the spend, so totals move more than success rates.", "",
                  "| Strategy | Success | Model spend /finished | Total /finished, counting failed attempts | Saves vs Sonnet (model spend) |", "|---|---|---|---|---|"]
        for name, row in ((n, iv[n]) for n in order if n in iv):
            cells = []
            for k, kind in (("success_rate", "pct"), ("cost_per_finished", "usd"), ("total_per_finished_with_human", "usd"),
                            ("savings_vs_sonnet", "pct")):
                lo_hi = row.get(k)
                cells.append(f"{fmt(lo_hi[0], kind)} to {fmt(lo_hi[1], kind)}" if lo_hi else "–")
            lines.append(f"| {card['strategies'][name]['label']} | " + " | ".join(cells) + " |")
        lines.append("")
    hs = card.get("human_sensitivity") or {}
    if hs:
        costs = sorted(next(iter(hs.values())), key=float)
        lines += ["## Sensitivity: the price per failed attempt", "",
                  "Total per finished task, counting failed attempts, at each price per failed attempt ($0 is model spend alone).",
                  "", "| Strategy | " + " | ".join(f"${float(c):g}" for c in costs) + " |", "|---" * (len(costs) + 1) + "|"]
        for name, row in ((n, hs[n]) for n in order if n in hs):
            lines.append(f"| {card['strategies'][name]['label']} | " + " | ".join(fmt(row[c], "usd") for c in costs) + " |")
        lines.append("")
    for axis, values in (card.get("sensitivity") or {}).items():
        first = next(iter(values.values()))
        names = [n for n in order if n in first] + [n for n in first if n not in order]
        lines += [f"## Sensitivity: {axis.replace('_', ' ')}", "", f"{card['settings']['sensitivity_tasks']} tasks per value. "
                  "Each cell: success, model spend per finished task, and the total per finished task counting failed attempts.", "",
                  "| Strategy | " + " | ".join(f"{axis} {v}" for v in values) + " |", "|---" * (len(values) + 1) + "|"]
        for name in names:
            cells = [f"{fmt(values[v][name]['success_rate'], 'pct')}, {fmt(values[v][name]['cost_per_finished'], 'usd')}, "
                     f"{fmt(values[v][name]['total_per_finished_with_human'], 'usd')}" for v in values]
            lines.append(f"| {card['strategies'].get(name, {}).get('label', name)} | " + " | ".join(cells) + " |")
        lines.append("")
    cal = card.get("calibration") or {}
    if cal:
        lines += ["## Calibration", "", "Simulated first-attempt success against Epoch's curves, by model (coding ability).", "",
                  "| Model | Ability | Hard class | FrontierSWE (Epoch) | SWE-bench mix | SWE-bench Verified (fit) |", "|---|---|---|---|---|---|"]
        for m, c in sorted(cal.items(), key=lambda kv: -kv[1]["ability"]):
            lines.append(f"| {m} | {c['ability']:.1f} | {c['hard']:.0%} | {c['frontier_swe']:.0%} | {c['swe_mix']:.0%} | {c['swe_verified']:.0%} |")
        lines.append("")
    return "\n".join(lines)


HTML_STYLE = """
:root{--bg:#F7F7F9;--surface:#FFFFFF;--fg:#0E0F14;--fg2:#3B3E4A;--muted:#62677A;--line:#E7E7EC;--head:#F1F1F5;--accent:#4338CA}
@media (prefers-color-scheme:dark){:root{--bg:#0B0C0F;--surface:#14151B;--fg:#EDEEF3;--fg2:#C3C6D2;--muted:#9A9EAE;--line:#23252E;
--head:#111217;--accent:#A9A6FF}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 "Geist Variable",Geist,system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1200px;margin:0 auto;padding:24px 16px}
h1{font-size:20px;font-weight:600;margin:0 0 4px} h2{font-size:16px;font-weight:600;margin:28px 0 8px;color:var(--accent)}
h3{font-size:13px;font-weight:600;margin:20px 0 6px;color:var(--fg2)} p{color:var(--muted);max-width:90ch;font-size:13px}
.wrap{overflow-x:auto;background:var(--surface);border:1px solid var(--line);border-radius:10px}
table{border-collapse:collapse;width:100%;font:12.5px/1.4 "Geist Mono Variable","Geist Mono",ui-monospace,SFMono-Regular,Menlo,monospace}
th,td{padding:6px 12px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left;font-family:"Geist Variable",Geist,system-ui,sans-serif}
th{background:var(--head);font-weight:500;font-size:12px;color:var(--muted)} tr:last-child td{border-bottom:0}
"""


def html(card: dict) -> str:
    """The Markdown report as a static page in the console's "Clean technical" style (ui/src/theme.ts: Geist, hairline
    panels, one indigo accent), light and dark."""
    import html as h

    body: list[str] = []
    table: list[list[str]] = []

    def flush() -> None:
        if table:
            head, rows = table[0], table[2:]
            body.append("<div class=wrap><table><thead><tr>" + "".join(f"<th>{h.escape(c)}</th>" for c in head) + "</tr></thead><tbody>"
                        + "".join("<tr>" + "".join(f"<td>{h.escape(c)}</td>" for c in r) + "</tr>" for r in rows) + "</tbody></table></div>")
            table.clear()

    for line in markdown(card).splitlines():
        if line.startswith("|"):
            table.append([c.strip() for c in line.strip("|").split("|")])
            continue
        flush()
        if line.startswith("### "):
            body.append(f"<h3>{h.escape(line[4:])}</h3>")
        elif line.startswith("## "):
            body.append(f"<h2>{h.escape(line[3:])}</h2>")
        elif line.startswith("# "):
            body.append(f"<h1>{h.escape(line[2:])}</h1>")
        elif line.strip():
            body.append(f"<p>{h.escape(line)}</p>")
    flush()
    return ("<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>Routing scorecard</title><style>{HTML_STYLE}</style></head><body><main>" + "".join(body) + "</main></body></html>")


# ------------------------------------------------------------------ the regression check (CI)

CHECK_TASKS = 1000  # enough that every class but the rarest (very hard coding, about 0.4%) has 50 or more
CHECK_SEED = 7
CHECK_STRATEGY = "default"  # the configuration's own routing.model_choice
COST_TOLERANCE = 0.05  # the total per finished task, counting failed attempts, may rise this much (5%)
SUCCESS_TOLERANCE = 0.02  # success may drop this much (2 points)
# Per task class, the default's model spend per finished task may be at most this much above the threshold's on the same
# tasks (decision 67: the cheapest model that meets the bar, unless the default's objective buys something with it).
CLASS_TOLERANCE = 0.10
CLASS_MIN_TASKS = 20  # a class with fewer tasks in the check is too noisy to compare (very hard coding: about 4)


def _per_class(results: list[Result], human_cost: float) -> dict[str, dict]:
    out = {}
    for c in CLASSES:
        rs = [r for r in results if r.cls == c]
        if rs:
            o = summarize(rs, human_cost)
            out[c] = {"tasks": o["tasks"], "success_rate": o["success_rate"], "cost_per_finished": o["cost_per_finished"]}
    return out


def check_run(raw: dict | None = None, catalog: dict | None = None, tasks: int = CHECK_TASKS, seed: int = CHECK_SEED) -> dict:
    """The default strategy (`auto` as the configuration routes it) on the standard population, and the threshold on the
    same tasks, per task class: what CI compares with the stored baseline and with each other."""
    sim = Simulator(raw, catalog, seed=seed)
    ts = population(tasks, seed)
    results = sim.run(Strategy(CHECK_STRATEGY, requested="auto"), ts)
    model_choice = sim.world.cfg.routing.model_choice
    o = summarize(results, sim.human_cost)
    threshold = sim.run(Strategy("threshold", team={"model_choice": "threshold"}), ts)
    return {"tasks": tasks, "seed": seed, "human_cost": sim.human_cost, "model_choice": model_choice,
            "success_rate": o["success_rate"], "cost_per_finished": o["cost_per_finished"],
            "total_per_finished_with_human": o["total_per_finished_with_human"], "levels": o["levels"],
            "by_class": _per_class(results, sim.human_cost), "threshold_by_class": _per_class(threshold, sim.human_cost)}


def class_problems(now: dict, tolerance: float = CLASS_TOLERANCE) -> list[str]:
    """Task classes where the default spent more per finished task than the threshold on the same tasks, by more than
    ``tolerance`` (empty: none)."""
    problems = []
    for c, mine in (now.get("by_class") or {}).items():
        ref = (now.get("threshold_by_class") or {}).get(c) or {}
        a, b = mine.get("cost_per_finished"), ref.get("cost_per_finished")
        if a is None or b is None or mine.get("tasks", 0) < CLASS_MIN_TASKS or a <= b * (1 + tolerance) + 1e-9:
            continue
        problems.append(f"{c}: model spend per finished task ${a:.4g} against the threshold's ${b:.4g} on the same tasks "
                        f"({(a / b - 1) if b else math.inf:+.1%}, more than the {tolerance:.0%} allowed)")
    return problems


def compare(now: dict, baseline: dict, cost_tolerance: float = COST_TOLERANCE, success_tolerance: float = SUCCESS_TOLERANCE) -> list[str]:
    """What got worse than the baseline by more than the tolerances, and the task classes where the default spends more than
    the threshold (``class_problems``); empty: nothing."""
    problems = class_problems(now)
    b, n = baseline["total_per_finished_with_human"], now["total_per_finished_with_human"]
    if b and n > b * (1 + cost_tolerance):
        problems.append(f"total per finished task, counting failed attempts, rose from ${b:.4f} to ${n:.4f} "
                        f"({(n / b - 1):+.1%}, more than the {cost_tolerance:.0%} allowed)")
    if now["success_rate"] < baseline["success_rate"] - success_tolerance:
        problems.append(f"success fell from {baseline['success_rate']:.1%} to {now['success_rate']:.1%} "
                        f"(more than the {success_tolerance * 100:.0f} points allowed)")
    return problems


# ------------------------------------------------------------------ replay: real task starts, weighted by propensity

EXPLORED = ("served", "down", "up")


def _level_map_policy(mapping: dict[str, int]) -> Callable[[str, int], int]:
    return lambda task, frm: int(mapping.get(f"{task}:{frm}", frm))


POLICIES: dict[str, Callable[[str, int], int]] = {
    "served": lambda task, frm: frm,
    "one-down": lambda task, frm: max(1, frm - 1),
    "one-up": lambda task, frm: min(5, frm + 1),
}


def policy(spec: str) -> tuple[str, Callable[[str, int], int]]:
    """A named policy (served, one-down, one-up) or a level map as JSON: {"coding:4": 3}."""
    if spec in POLICIES:
        return spec, POLICIES[spec]
    mapping = json.loads(spec)
    return f"level_map {json.dumps(mapping, sort_keys=True)}", _level_map_policy(mapping)


def read_rows(path: str | Path) -> list[dict]:
    """Rows from a JSON lines, CSV or Parquet file (the export's formats). JSON in CSV cells (tags, coding) is decoded."""
    path = Path(path)
    if path.suffix == ".parquet":
        from .export import need_pyarrow

        need_pyarrow()
        import pyarrow.parquet as pq

        rows = pq.read_table(str(path)).to_pylist()
        for r in rows:
            if isinstance(r.get("tags"), list):
                r["tags"] = {k: v for k, v in r["tags"]}
        return rows
    text = path.read_text()
    if path.suffix == ".csv":
        rows = list(csv.DictReader(io.StringIO(text)))
        for r in rows:
            for k, v in list(r.items()):
                if isinstance(v, str) and v[:1] in "{[":
                    try:
                        r[k] = json.loads(v)
                    except ValueError:
                        pass
                elif v == "":
                    r[k] = None
        return rows
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _ts(v: Any) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, datetime):
        return v.timestamp()
    return datetime.fromisoformat(str(v).replace("Z", "+00:00")).timestamp()


def join_costs(starts: list[dict], ledger: list[dict]) -> None:
    """Each task start's cost: the ledger's ``actual_cost`` for its session from its start until the session's next task
    start (set as ``cost`` on rows that have none)."""
    by_session: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for r in ledger:
        by_session[str(r.get("session_id"))].append((_ts(r["ts"]), float(r.get("actual_cost") or 0.0)))
    starts_by: dict[str, list[dict]] = defaultdict(list)
    for s in starts:
        starts_by[str(s.get("session_id"))].append(s)
    for sid, rows in starts_by.items():
        rows.sort(key=lambda r: _ts(r["ts"]))
        spend = sorted(by_session.get(sid, []))
        for i, r in enumerate(rows):
            if r.get("cost") is not None:
                continue
            lo, hi = _ts(r["ts"]), _ts(rows[i + 1]["ts"]) if i + 1 < len(rows) else math.inf
            r["cost"] = sum(c for t, c in spend if lo <= t < hi)


def success_of(row: dict) -> float:
    """A task start's success: its own ``success`` when the export has one, else 1 minus the weight of its outcome signals
    (learned_jobs.signals, as the outcome report reads them)."""
    if row.get("success") is not None:
        return float(row["success"])
    return 1.0 - failed(signals({"request_id": row.get("request_id", ""), **row}, set()))


def replay(rows: list[dict], policies: dict[str, Callable[[str, int], int]], human_cost: float = HUMAN_COST) -> dict:
    """Score level policies on explored task starts (decision 59) without running them: each row whose routed level is
    the one the policy would have chosen counts with weight 1 / the chance of the arm it drew (a Hájek estimate), so the
    few task starts explored a level down stand for every task start the policy would send there. Canary rows, and rows
    with no arm or propensity (teams that don't explore), are left out."""
    usable = []
    for r in rows:
        arm, frm, p = r.get("arm"), r.get("arm_from"), r.get("propensity")
        if arm not in EXPLORED or not frm or not p or (int(r.get("status") or 200) >= 400 and int(r.get("status") or 0) != 499):
            continue
        level = int(r.get("level") or frm)
        usable.append((str(r.get("task") or "coding"), int(frm), level, 1.0 / float(p), success_of(r), float(r.get("cost") or 0.0)))
    out: dict[str, Any] = {"rows": len(rows), "usable": len(usable), "human_cost": human_cost, "policies": {}}
    for name, pol in policies.items():
        match = [(w, s, c) for task, frm, level, w, s, c in usable if level == pol(task, frm)]
        sw = sum(w for w, _, _ in match)
        if not match or sw <= 0:
            out["policies"][name] = {"n": 0, "n_eff": 0.0, "success_rate": None, "cost_per_task": None, "cost_per_finished": None,
                                     "total_per_finished_with_human": None, "se_success": None}
            continue
        rate = sum(w * s for w, s, _ in match) / sw
        cost = sum(w * c for w, _, c in match) / sw
        se = math.sqrt(sum((w * (s - rate)) ** 2 for w, s, _ in match)) / sw
        out["policies"][name] = {"n": len(match), "n_eff": round(sw ** 2 / sum(w * w for w, _, _ in match), 1),
                                 "success_rate": round(rate, 4), "se_success": round(se, 5), "cost_per_task": round(cost, 6),
                                 "cost_per_finished": round(cost / rate, 6) if rate else None,
                                 "total_per_finished_with_human": round((cost + human_cost * (1 - rate)) / rate, 6) if rate else None}
    return out


def export_rows(results: list[Result], tasks: list[Task], world: World, team: str = SIM_TEAM) -> tuple[list[dict], list[dict]]:
    """A simulated run as the gateway would have recorded it: task-start rows shaped like route_features (level, arm,
    propensity, served model, the coding outcomes and the correction that followed) and ledger rows (one per turn, as
    `meritorder export` writes them). One task start per human prompt and per compaction, as the gateway counts them; the
    outcomes go on an attempt's last one. For testing replay on data whose truth is known. Needs a run with ``record``."""
    by_id = {t.id: t for t in tasks}
    starts, ledger = [], []
    for r in results:
        t = by_id[r.task]
        sid = f"sim-{t.id}"
        for k, a in enumerate(r.attempts):
            for j, seg in enumerate(a.segments):
                x = seg["explore"] or {}
                last = j == len(a.segments) - 1
                coding = dict(a.coding) if last else {}
                if last and a.correction:
                    coding["correction"] = a.correction
                m = world.cfg.model(seg["model"])
                level = int(seg["floor"].split(":")[1]) if seg["floor"] else None
                starts.append({"request_id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{sid}/{k}/{j}")),
                               "ts": datetime.fromtimestamp(seg["ts"], UTC).isoformat(), "team": team, "session_id": sid, "task": t.task,
                               "rules_level": x.get("from") or level, "level": level, "arm": x.get("arm"), "arm_from": x.get("from"),
                               "propensity": x.get("p"), "served_model": seg["model"], "served_score": m.q(t.task) if m else None,
                               "status": 200, "retried": False, "floor_raised": None, "coding": coding or None,
                               "truth": {"success": a.success, "class": t.cls, "attempt": k, "segment": j}})
        for row in r.ledger:
            x = r.attempts[row["attempt"]].segments[row["segment"]]["explore"] or {}
            u = row["usage"]
            ledger.append({"id": str(uuid.uuid4()), "ts": datetime.fromtimestamp(row["ts"], UTC).isoformat(), "team": team,
                           "session_id": sid, "floor": row["floor"], "served_model": row["model"], "backend": row["backend"],
                           "input_tokens": u["input_tokens"], "cache_read_tokens": u["cache_read_tokens"],
                           "cache_write_tokens": u["cache_write_tokens"], "output_tokens": u["output_tokens"],
                           "actual_cost": round(row["cost"], 8), "status": 200,
                           "tags": {"explore": x.get("arm"), "explore_from": f"{t.task}:{x.get('from')}"} if x else {}})
    return starts, ledger


def explore_strategy(share: float = 0.5) -> Strategy:
    """The threshold with exploration on (the learned router recording, as exploration needs): synthetic exported rows."""
    return Strategy("explore", "auto, threshold, exploring", routing={"learned": {"mode": "shadow"}},
                    team={"model_choice": "threshold", "exploration": {"enabled": True, "share": share}}, explore=True)


# ------------------------------------------------------------------ the command line (scripts/routing_sim.py)

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "tests" / "fixtures" / "routing_eval_baseline.json"
FROZEN_PRICES = ROOT / "tests" / "fixtures" / "routing_eval_prices.json"  # the prices and scores the baseline was made with


def _write(out: Path, card: dict, as_html: bool) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / "scorecard.md", out / "scorecard.json"]
    paths[0].write_text(markdown(card) + "\n")
    paths[1].write_text(json.dumps(card, indent=1, default=str) + "\n")
    if as_html:
        paths.append(out / "scorecard.html")
        paths[2].write_text(html(card))
    return paths


def main(argv: list[str] | None = None) -> int:
    import argparse

    argv = list(argv if argv is not None else [])
    if not argv or argv[0] not in ("run", "check", "replay", "export", "-h", "--help"):
        argv = ["run", *argv]
    ap = argparse.ArgumentParser(prog="routing_sim", description="Offline routing simulator and scorecard: no model calls, no spend.")
    sub = ap.add_subparsers(dest="cmd")
    run = sub.add_parser("run", help="simulate the strategies on a population and write the scorecard (the default)")
    run.add_argument("--tasks", type=int, default=Settings.tasks)
    run.add_argument("--seed", type=int, default=Settings.seed)
    run.add_argument("--mix", default="standard", help=f"{', '.join(MIXES)}, or JSON shares per class")
    run.add_argument("--human-cost", type=float, default=HUMAN_COST, help="price per failed attempt, in dollars: the time it costs someone (an assumption)")
    run.add_argument("--pause-share", type=float, default=PAUSE_SHARE, help="share of later turns after a 5-60 minute pause")
    run.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS)
    run.add_argument("--quality-error", type=float, default=0.0,
                     help="spread (ability points) of each model's true ability around its published one")
    run.add_argument("--no-oracle", action="store_true")
    run.add_argument("--no-sensitivity", action="store_true")
    run.add_argument("--sensitivity-tasks", type=int, default=Settings.sensitivity_tasks)
    run.add_argument("--strategy", action="append", help="run only these strategies (repeat)")
    run.add_argument("--plugin", action="append", default=[], help="module:attribute giving more strategies (repeat)")
    run.add_argument("--prices", help="a prices.json to use instead of config/prices.json")
    run.add_argument("--out", default=str(ROOT / "docs" / "routing-eval"))
    run.add_argument("--html", action="store_true", help="also write a static HTML report")
    chk = sub.add_parser("check", help="compare the default strategy with the stored baseline (what CI runs)")
    chk.add_argument("--baseline", default=str(BASELINE))
    chk.add_argument("--update", action="store_true", help="write the current result as the new baseline")
    rep = sub.add_parser("replay", help="score level policies on explored task starts from an export, weighted by propensity")
    rep.add_argument("rows", help="task starts: JSON lines, CSV or Parquet (route_features columns, with cost or a ledger)")
    rep.add_argument("--ledger", help="the ledger export (meritorder export), for each task start's cost")
    rep.add_argument("--policy", action="append", help="served, one-down, one-up, or a level map as JSON (repeat)")
    rep.add_argument("--human-cost", type=float, default=HUMAN_COST)
    exp = sub.add_parser("export", help="write synthetic exported rows (task starts and ledger) from an exploring run")
    exp.add_argument("--tasks", type=int, default=400)
    exp.add_argument("--seed", type=int, default=3)
    exp.add_argument("--share", type=float, default=0.5, help="exploration share (at most 0.5)")
    exp.add_argument("--out", default=".")
    args = ap.parse_args(argv)

    if args.cmd == "run":
        catalog = load_catalog(Path(args.prices)) if args.prices else None
        plugins = [s for spec in args.plugin for s in load_plugin(spec)] + entry_point_strategies()
        settings = Settings(tasks=args.tasks, seed=args.seed, mix=args.mix, human_cost=args.human_cost, pause_share=args.pause_share,
                            max_attempts=args.max_attempts, quality_error=args.quality_error, oracle=not args.no_oracle,
                            sensitivity=not args.no_sensitivity, sensitivity_tasks=args.sensitivity_tasks, strategies=args.strategy)
        card = evaluate(settings, plugins, catalog=catalog, progress=lambda m: print(m, flush=True))
        for p in _write(Path(args.out), card, args.html):
            print(f"wrote {p}")
        return 0
    if args.cmd == "check":
        now = check_run(catalog=load_catalog(FROZEN_PRICES))
        path = Path(args.baseline)
        if args.update:
            path.write_text(json.dumps(now, indent=1, sort_keys=True) + "\n")
            print(f"baseline written to {path}: {json.dumps(now)}")
            return 0
        problems = compare(now, json.loads(path.read_text()))
        print(json.dumps(now))
        for p in problems:
            print(f"worse than the baseline: {p}")
        return 1 if problems else 0
    if args.cmd == "replay":
        rows = read_rows(args.rows)
        if args.ledger:
            join_costs(rows, read_rows(args.ledger))
        policies = dict(policy(p) for p in (args.policy or list(POLICIES)))
        print(json.dumps(replay(rows, policies, args.human_cost), indent=1))
        return 0
    sim = Simulator(seed=args.seed)  # export
    ts = population(args.tasks, args.seed)
    results = sim.run(explore_strategy(args.share), ts, record=True)
    starts, ledger = export_rows(results, ts, sim.world)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("task_starts.jsonl", starts), ("ledger.jsonl", ledger)):
        (out / name).write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))
        print(f"wrote {out / name} ({len(rows)} rows)")
    return 0

