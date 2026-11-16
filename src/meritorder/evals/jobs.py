"""Eval runs as jobs: started from the console (the worker runs them) or by `meritorder eval` (which runs them
itself), with progress, cancelling, a cost estimate up front, and results kept for the console's Evals page."""

from __future__ import annotations

import asyncio
import fnmatch
import logging
import math
import re
import time
import uuid
from dataclasses import asdict, fields
from datetime import UTC, datetime
from pathlib import Path

import httpx

from ..config import Price
from .runner import (BUILTIN, DEFAULT_SUITE, RunContext, TaskResult, estimate_judging, judges_for, load_suite, needs, run_suite,
                     scale, score, turn_tokens)

log = logging.getLogger("meritorder.evals")

STATUSES = ("queued", "running", "cancelling", "done", "failed", "cancelled")
FINISHED = ("done", "failed", "cancelled")
# One judge per family, at the same tier (Claude: Fable > Opus > Sonnet > Haiku; GPT: Astra > Sol > Terra > Luna; Gemini 4: Carbon
# > Argon). Each answer is graded by the two from other families (runner.judges_for), so none grades its own family.
DEFAULT_JUDGES = ("claude-opus", "gpt-sol", "gemini-carbon")
REASONING_HEADROOM = 2.0  # reasoning models write more than the visible answer

# Evals cost real money, and the estimate is rough (a codebase run on Claude Haiku first came in at three times what
# was estimated, before the estimate followed turns and counted long-context prices; today's estimate puts the tasks
# that run finished at $4.81 against $3.80 spent, docs/11-evals.md). So every run has a spending cap: no new task starts once
# it has spent that much. It defaults to CAP_FACTOR times the estimate, and starting a run costing more than
# CONFIRM_ABOVE_USD needs an explicit "I understand" (TYPE_YES_ABOVE_USD: the CLI wants "yes" typed out).
CAP_FACTOR = 1.5
CONFIRM_ABOVE_USD = 1.0
TYPE_YES_ABOVE_USD = 10.0
ROUGH = "a rough guide: real runs have cost up to three times as much"


def default_cap(estimate_usd: float) -> float:
    """The spending cap a run gets unless one is given: CAP_FACTOR times the estimate, rounded up to a dollar."""
    return float(max(1, math.ceil(estimate_usd * CAP_FACTOR)))


def capped_note(cap: float | None, spent: float, not_started: int) -> str:
    return f"stopped at its spending cap of ${cap or 0:,.2f} (spent ${spent:,.2f}); {not_started} task runs didn't start"


def cost_warning(est: dict, cap: float, budget_left: float | None = None) -> list[str]:
    """What to say before a run, line by line: the estimate by model, how rough it is, the cap, and the budget."""
    lines = [f"Estimated cost: about ${est['usd']:,.2f} ({ROUGH})."]
    by = sorted(est.get("by_model", {}).items(), key=lambda kv: -kv[1])
    if len(by) > 1:
        lines.append("By model: " + ", ".join(f"{m} ${v:,.2f}" for m, v in by) + (
            f"; judging ${est['judging_usd']:,.2f}" if est.get("judging_usd") else "") + ".")
    if est.get("unpriced"):
        lines.append(f"No known price for {', '.join(est['unpriced'])}: not in the estimate.")
    lines.append(f"Spending cap: ${cap:,.2f}. No new task starts once the run has spent that much; tasks already running finish.")
    if budget_left is not None and budget_left < max(est["usd"], cap):
        lines.append(f"The organization has ${max(budget_left, 0):,.2f} of its monthly budget left: the run stops when that runs out.")
    return lines


def new_run_id() -> str:
    return "ev_" + uuid.uuid4().hex[:12]


def suites() -> list[dict]:
    """The built-in suites, with how many tasks of each kind and level they hold."""
    out = []
    for name, path in BUILTIN.items():
        if not Path(path).exists():
            continue
        s = load_suite(name)
        tasks = s["tasks"]
        out.append({"name": name, "tasks": len(tasks),
                    "kinds": {k: sum(1 for t in tasks if t.get("kind", "answer") == k) for k in ("answer", "code", "agent", "judged")},
                    "levels": {str(lvl): sum(1 for t in tasks if int(t["level"]) == lvl) for lvl in range(1, 6)},
                    "task_types": sorted({t["task"] for t in tasks})})
    return out


FIXTURES = Path(__file__).parent / "envs" / "fixtures"
FILE_LIMIT = 200_000  # characters of one file shown on the Tasks tab


def _hidden_tests(source: str) -> int:
    return len(re.findall(r"^\s*def test_\w*\(", source, re.M))


def _env_data(t: dict) -> tuple[dict[str, str], dict[str, str]]:
    """An agent task's environment, as files: what the model can reach through the tools, and what only the grader
    sees (hidden tests, the expected answer). Found from the fixture its env_args name (project, scenario, dataset)."""
    from .envs import load_env

    env = load_env(t["env"])
    if hasattr(env, "fixture_files"):  # an environment too big to show whole says what's worth showing
        return env.fixture_files(**(t.get("env_args") or {}))
    visible: dict[str, str] = {}
    grader: dict[str, str] = {}
    names = {str(v) for v in (t.get("env_args") or {}).values() if isinstance(v, str)}

    def read(path: Path) -> str:
        text = path.read_text(encoding="utf-8", errors="replace")
        return text if len(text) <= FILE_LIMIT else text[:FILE_LIMIT] + "\n[shortened]"

    for name in sorted(names):
        repo = FIXTURES / "repos" / name
        if repo.is_dir():
            for part, into in (("files", visible), ("hidden", grader)):
                for f in sorted((repo / part).rglob("*")):
                    if f.is_file() and "__pycache__" not in f.parts:
                        into[f"{part}/{f.relative_to(repo / part)}"] = read(f)
            for f in sorted(repo.glob("*.yaml")):
                grader[f.name] = read(f)
            continue
        for f in sorted(FIXTURES.glob(f"*/{name}.*")):
            grader[f"{f.parent.name}/{f.name}"] = read(f)
    return visible, grader


def graded_by(t: dict) -> str:
    """How a task is graded, in a few words."""
    kind = t.get("kind", "answer")
    if kind == "code":
        return f"{_hidden_tests(t['code'].get('tests_source', ''))} hidden tests, run in a sandbox"
    if kind == "agent":
        from .envs import load_env

        env = load_env(t["env"])
        if hasattr(env, "graded_by"):
            return env.graded_by(**(t.get("env_args") or {}))
        _, grader = _env_data(t)
        tests = sum(_hidden_tests(v) for k, v in grader.items() if k.startswith("hidden/"))
        return f"{tests} hidden tests on the edited repository, in a sandbox" if tests else load_env(t["env"]).grading
    if kind == "judged":
        return f"2 judges, {len(t.get('rubric') or [])} rubric points"
    n = len(t.get("checks") or [])
    return f"{n} check{'' if n == 1 else 's'} on the reply"


def task_list(name: str) -> list[dict]:
    """Every task in a suite, briefly: what it is and how it's graded."""
    out = []
    for t in load_suite(name)["tasks"]:
        prompt = t.get("prompt", "")
        out.append({"id": t["id"], "task": t["task"], "level": int(t["level"]), "kind": t.get("kind", "answer"),
                    "summary": t.get("summary") or prompt[:200], "graded_by": graded_by(t), "prompt_words": len(prompt.split()),
                    "generated": bool(t.get("generate")), "max_turns": t.get("max_turns") if t.get("kind") == "agent" else None})
    return out


def task_detail(name: str, task_id: str) -> dict | None:
    """One task in full: the prompt, the tools, and everything it's graded with (checks, hidden tests, reference
    solution, environment data, rubric). Never sent to the models it evaluates."""
    t = next((x for x in load_suite(name)["tasks"] if x["id"] == task_id), None)
    if t is None:
        return None
    kind = t.get("kind", "answer")
    base = Path(t["_dir"]) if t.get("_dir") else None
    out: dict = {"id": t["id"], "suite": name, "task": t["task"], "level": int(t["level"]), "kind": kind, "summary": t.get("summary"),
                 "graded_by": graded_by(t), "prompt": t.get("prompt", ""), "effort": t.get("effort"), "max_tokens": t.get("max_tokens"),
                 "generated": {"by": t["generate"], "args": t.get("generate_args") or {}} if t.get("generate") else None,
                 "checks": t.get("checks") or [], "tools": [], "code": None, "agent": None, "judged": None}
    if t.get("tools"):
        out["tools"] = [x.get("function", x) for x in t["tools"]]
    if kind == "code":
        spec = t["code"]

        def src(key):
            return (base / t[key]).read_text(encoding="utf-8") if base and t.get(key) and (base / t[key]).is_file() else None
        out["code"] = {"tests": spec.get("tests_source", ""), "tests_count": _hidden_tests(spec.get("tests_source", "")),
                       "forbid": spec.get("forbid") or [], "timeout": spec.get("timeout", 30), "per_test_timeout": spec.get("per_test_timeout", 10),
                       "reference": src("reference"), "wrong": src("wrong")}
    elif kind == "agent":
        from .envs import load_env

        env = load_env(t["env"])
        visible, grader = _env_data(t)
        out["tools"] = [x.get("function", x) for x in env.tools]
        out["agent"] = {"env": t["env"], "about": (env.__doc__ or "").strip(), "grading": env.grading, "env_args": t.get("env_args") or {},
                        "max_turns": int(t.get("max_turns", 30)), "needs_sandbox": bool(getattr(env, "needs_sandbox", False)),
                        "visible": visible, "grader": grader}
    elif kind == "judged":
        out["judged"] = {"rubric": t.get("rubric") or [], "reference_answer": t.get("reference_answer")}
    return out


def select(suite: dict, only: list[str] | None) -> dict:
    """Keep tasks whose id matches any of the glob patterns (all of them when none are given)."""
    if not only:
        return suite
    return {**suite, "tasks": [t for t in suite["tasks"] if any(fnmatch.fnmatchcase(t["id"], p) for p in only)]}


def plan(request: dict) -> dict:
    return select(load_suite(request.get("suite") or DEFAULT_SUITE), request.get("only"))


def _price(model: dict) -> dict | None:
    """Cheapest live route's price, from the admin config view (`/v1/admin/config` shape)."""
    prices = [r["price"] for r in model.get("routes", []) if r.get("live") and r.get("price")]
    return min(prices, key=lambda p: p["input"]) if prices else None


def _at(price: dict, context: int) -> dict:
    """``price`` (a price as the admin config view shows it) for a request of ``context`` tokens: Price.at's tiers."""
    if not price.get("tiers"):
        return price
    return Price(**{k: v for k, v in price.items() if k in Price.model_fields}).at(context).model_dump()


def estimate(models: dict[str, dict], suite: dict, names: list[str], repeat: int, judges: list[str]) -> dict:
    """Rough cost of a run, before running it. ``models`` maps names to the admin config view of each model."""
    by_model: dict[str, float] = {}
    for name in names:
        p = _price(models.get(name, {}))
        if p is None:
            by_model[name] = 0.0
            continue
        total = 0.0
        for t in suite["tasks"]:
            for context, inp, cached, out in turn_tokens(t):
                q = _at(p, context)  # long contexts cost more with some providers (Claude Haiku 5x above 100k tokens)
                read = q.get("cache_read") if q.get("cache_read") is not None else q["input"] * 0.1
                total += (inp * q["input"] + cached * read + out * REASONING_HEADROOM * q["output"]) / 1e6
        by_model[name] = total * repeat
    judging = 0.0
    for name in names:
        for j in judges_for(name, judges):
            p = _price(models.get(j, {}))
            if p is None:
                continue
            for t in suite["tasks"]:
                inp, out = estimate_judging(t, 1)
                judging += (inp * p["input"] + out * REASONING_HEADROOM * p["output"]) / 1e6 * repeat
    kinds = [needs(t) for t in suite["tasks"]]
    return {"usd": round(sum(by_model.values()) + judging, 4), "by_model": {k: round(v, 4) for k, v in by_model.items()},
            "judging_usd": round(judging, 4), "tasks": len(suite["tasks"]), "runs": len(suite["tasks"]) * repeat * len(names),
            "needs_sandbox": kinds.count("sandbox"), "judged": kinds.count("judges"),
            "unpriced": [n for n in names if _price(models.get(n, {})) is None]}


def efforts(models: dict[str, dict], names: list[str], by_level: dict) -> dict[str, dict[str, str]]:
    """The reasoning effort each model runs each task level at: what `auto` sends for that level
    (routing.effort_by_floor), where the model takes that effort. Without it, models would run at their own
    defaults, which differ (Claude Opus 5.5 and Haiku 5.5 default to medium, Fable and Sonnet to high)."""
    out: dict[str, dict[str, str]] = {}
    for name in names:
        levels = set(models.get(name, {}).get("effort_levels") or [])
        out[name] = {str(lvl): e for lvl, e in sorted(by_level.items(), key=lambda kv: int(kv[0])) if e in levels}
    return out


def routes(models: dict[str, dict], names: list[str]) -> dict[str, str]:
    """The backend each model runs on: its first live route. Configs list a model's own provider first, so a run
    measures the provider's API rather than whichever reseller the router finds cheapest or fastest that minute."""
    out = {}
    for name in names:
        live = [r["backend"] for r in models.get(name, {}).get("routes", []) if r.get("live")]
        if live:
            out[name] = live[0]
    return out


def formats(models: dict[str, dict], backends: dict[str, str]) -> dict[str, str]:
    """The API each model's chosen backend speaks ("anthropic" or "openai"). Agent tasks use the model's own: Claude
    keeps its thinking between tool calls only over Anthropic's Messages API, which is how Claude Code talks to it."""
    out = {}
    for name, backend in backends.items():
        fmt = next((r.get("format") for r in models.get(name, {}).get("routes", []) if r["backend"] == backend), None)
        if fmt:
            out[name] = fmt
    return out


def pick_judges(models: dict[str, dict], asked: list[str] | None = None) -> list[str]:
    """The judges to use: the ones asked for, else the defaults; only models with a live route."""
    return [j for j in (asked or list(DEFAULT_JUDGES)) if _price(models.get(j, {})) is not None or any(r.get("live") for r in models.get(j, {}).get("routes", []))]


def result_dicts(results: list[TaskResult]) -> list[dict]:
    return [asdict(r) for r in results]


# Errors that say nothing about the model: a provider account that can't pay, a rate limit, an outage, a lost connection.
_NOT_THE_MODEL = re.compile(r"HTTP (402|429|5\d\d)\b|(Connect|Read|Write|Pool)(Error|Timeout)\b|RemoteProtocolError\b|NetworkError\b")


def results_from(dicts: list[dict]) -> list[TaskResult]:
    """Stored results back as TaskResults. Provider errors recorded before they were left out of scores are marked
    skipped, as they would be now."""
    names = {f.name for f in fields(TaskResult)}
    out = []
    for d in dicts:
        r = TaskResult(**{k: v for k, v in d.items() if k in names})
        if not r.skipped and _NOT_THE_MODEL.match(r.error or ""):
            r.skipped = True
        out.append(r)
    return out


def rescore(row: dict) -> dict:
    """A run's scores worked out again from its results, so runs from before a change to scoring show today's."""
    if not row.get("results"):
        return row.get("scores") or {}
    out = score(results_from(row["results"]))
    if not any("model_ms" in d for d in row["results"]):
        # Recorded before the work was timed and counted: "not recorded", not zero seconds and zero tokens.
        for s in out.values():
            s.pop("efficiency", None)
            for t in s["tasks"].values():
                t.pop("efficiency", None)
    return out


def _iso(v) -> str | None:
    return v.isoformat(timespec="seconds") if isinstance(v, datetime) else v


def _not_graded(d: dict) -> bool:
    return bool(d.get("skipped")) or bool(_NOT_THE_MODEL.match(d.get("error") or ""))


def combine(rows: list[dict], model: str) -> list[tuple[str, dict]]:
    """One model's results across its runs (``rows`` newest first), as (run id, result) pairs: each task's come from
    the newest run that graded it (all of that run's tries, with --repeat), so running skipped tasks again fills them
    in. A task no run graded keeps its newest skipped result, so it still shows as skipped."""
    graded: dict[str, list[tuple[str, dict]]] = {}
    skipped: dict[str, list[tuple[str, dict]]] = {}
    for row in rows:
        by_task: dict[str, list[dict]] = {}
        for d in row.get("results") or []:
            if d.get("model") == model:
                by_task.setdefault(d["task_id"], []).append(d)
        for tid, ds in by_task.items():
            done = [d for d in ds if not _not_graded(d)]
            (graded if done else skipped).setdefault(tid, [(row["id"], d) for d in (done or ds)])
    return [x for xs in graded.values() for x in xs] + [x for tid, xs in skipped.items() if tid not in graded for x in xs]


async def latest_results(store, run: dict | None = None) -> dict[str, tuple[dict, list[tuple[str, dict]] | None]]:
    """Every model's entry in the latest view with its results: its latest run's, filled in task by task from its
    earlier finished runs (``combine``). With ``run`` (a finished run's row), that run counts as its models' latest.
    Results are None for a model recorded without them (an old run, or `meritorder eval` before runs were kept)."""
    entries = {m: dict(e) for m, e in ((await store.get_setting("evals") or {}).get("models") or {}).items()}
    if run:
        for m in run["request"].get("models") or []:
            entries[m] = {"run_id": run["id"], "suite": run["request"].get("suite") or "", "ran_at": _iso(run.get("finished_at"))}
    finished = [h for h in await store.list_eval_runs(limit=None) if h["status"] == "done"]  # newest first, without results
    rows: dict[str, dict | None] = {}

    async def load(rid: str) -> dict | None:
        if rid not in rows:
            rows[rid] = run if run and rid == run["id"] else await store.get_eval_run(rid)
        return rows[rid]

    out: dict[str, tuple[dict, list[tuple[str, dict]] | None]] = {}
    for m, e in entries.items():
        latest = await load(e["run_id"]) if e.get("run_id") else None
        if e.get("run_id") and not latest:
            continue
        if not latest or not latest.get("results"):
            out[m] = (e, None)
            continue
        older = [h["id"] for h in finished if h["id"] != latest["id"] and h["created_at"] < latest["created_at"]
                 and m in ((h.get("request") or {}).get("models") or [])]
        history = [latest] + [r for r in [await load(i) for i in older] if r and r.get("results")]
        out[m] = (e, combine(history, m))
    return out


async def latest_view(store, run: dict | None = None) -> dict[str, dict]:
    """Every model's latest result (see ``latest_results``), worked out again from the stored results and scored
    together on one scale. With ``run`` (a finished run's row), that run's models count as latest: what applying it
    would write."""
    models = {}
    for m, (e, pairs) in (await latest_results(store, run)).items():
        if pairs is None:
            row = await store.get_eval_run(e["run_id"]) if e.get("run_id") else None
            stats = rescore(row).get(m) if row else e  # recorded without a run: as stored
        else:
            stats = score(results_from([d for _, d in pairs])).get(m)
            if stats and not all("model_ms" in d for _, d in pairs):
                # Some were recorded before the work was timed and counted: "not recorded", not zero.
                stats.pop("efficiency", None)
                for t in stats["tasks"].values():
                    t.pop("efficiency", None)
            if stats:
                stats["runs"] = list(dict.fromkeys(rid for rid, _ in pairs))
        if stats:
            models[m] = {**stats, "suite": e.get("suite"), "ran_at": e.get("ran_at"), "run_id": e.get("run_id")}
    return scale(models)


def merge_latest(current: dict | None, run_id: str, suite: str, scores: dict) -> dict:
    """The console's 'latest result per model' view, updated with a finished run."""
    current = current or {"models": {}}
    ran_at = datetime.now(UTC).isoformat(timespec="seconds")
    for model, result in scores.items():
        current.setdefault("models", {})[model] = {**result, "suite": suite, "ran_at": ran_at, "run_id": run_id}
    return current


class Progress:
    """Counts results as they arrive and saves them every few seconds; tells the runner to stop when cancelled."""

    def __init__(self, total: int, save, check_cancel, every: float = 3.0, cap: float | None = None):
        self.total, self.done, self.passed, self.cost = total, 0, 0, 0.0
        self.save, self.check_cancel, self.every, self.cap = save, check_cancel, every, cap
        self.last_save = 0.0
        self.last_check = 0.0
        self.stopping = False
        self.capped: str | None = None  # "spending cap" once the run has stopped starting tasks at its cap

    def snapshot(self) -> dict:
        out = {"total": self.total, "done": self.done, "passed": self.passed}
        if self.cap is not None:
            out["cap"] = self.cap
        if self.capped:
            out["stopped"] = "spending cap"
        return out

    async def __call__(self, r: TaskResult) -> None:
        self.done += 1
        self.passed += int(r.passed)
        self.cost += r.cost
        now = time.monotonic()
        if now - self.last_save >= self.every or self.done == self.total:
            self.last_save = now
            await self.save(self.snapshot(), self.cost)

    async def stop(self) -> bool:
        if not self.stopping and self.cap is not None and self.cost >= self.cap:
            self.stopping = True
            self.capped = "spending cap"
            return True
        now = time.monotonic()
        if not self.stopping and now - self.last_check >= self.every:
            self.last_check = now
            self.stopping = await self.check_cancel()
        return self.stopping


async def execute(store, run: dict, gateway_url: str, key: str, sandbox=None, concurrency: int = 4,
                  client: httpx.AsyncClient | None = None) -> dict:
    """Run a queued (or client-claimed) eval run to the end and save its results. Returns the final row."""
    req = run["request"]
    try:
        suite = plan(req)
    except Exception as e:  # noqa: BLE001 (a bad suite fails the run, not the worker)
        return await store.update_eval_run(run["id"], status="failed", error=f"couldn't load the suite: {e}", finished_at=datetime.now(UTC))
    models, repeat = req["models"], int(req.get("repeat", 1))
    ctx = RunContext(sandbox=sandbox, judges=list(req.get("judges") or []), effort=dict(req.get("effort") or {}),
                     backends=dict(req.get("backends") or {}), formats=dict(req.get("formats") or {}))

    async def save(progress: dict, cost: float) -> None:
        await store.update_eval_run(run["id"], progress=progress, cost=round(cost, 6))

    async def cancelled() -> bool:
        row = await store.get_eval_run(run["id"])
        return bool(row and row["status"] in ("cancelling", "cancelled"))

    tracker = Progress(len(suite["tasks"]) * repeat * len(models), save, cancelled, cap=req.get("max_cost_usd"))
    await save(tracker.snapshot(), 0.0)
    try:
        results = await run_suite(gateway_url, key, models, suite, concurrency=concurrency, repeat=repeat, progress=tracker,
                                  client=client, ctx=ctx, should_stop=tracker.stop)
    except Exception as e:  # noqa: BLE001
        log.exception("eval run %s failed", run["id"])
        return await store.update_eval_run(run["id"], status="failed", error=f"{type(e).__name__}: {e}"[:500], finished_at=datetime.now(UTC))
    scores = score(results)
    # Stopped at the cap, the run still counts: its results are as good as a smaller run's.
    status = "cancelled" if tracker.stopping and not tracker.capped else "done"
    spent = sum(r.cost for r in results)
    note = capped_note(tracker.cap, spent, tracker.total - len(results)) if tracker.capped else None
    row = await store.update_eval_run(run["id"], status=status, scores=scores, results=result_dicts(results),
                                      progress={**tracker.snapshot(), "done": len(results)}, cost=round(spent, 6),
                                      finished_at=datetime.now(UTC), error=note)
    if status == "done" and scores:
        await store.set_setting("evals", merge_latest(await store.get_setting("evals"), run["id"], suite["name"], scores))
    return row


async def interrupted(store) -> int:
    """Runs a stopped worker left half-done: mark them failed so they don't look stuck forever."""
    n = 0
    for r in await store.list_eval_runs(200):
        if r["executor"] == "worker" and r["status"] in ("running", "cancelling"):
            await store.update_eval_run(r["id"], status="failed", error="interrupted: the worker stopped while it was running",
                                        finished_at=datetime.now(UTC))
            n += 1
    return n


async def worker_loop(store, gateway_url: str, poll: float = 5.0, once: bool = False, sandbox_factory=None,
                      client: httpx.AsyncClient | None = None) -> None:
    """In the worker: run queued eval runs one at a time, each with a temporary key for the person who started it."""
    from datetime import timedelta

    from .sandbox import DockerSandbox

    sandbox = sandbox_factory() if sandbox_factory else DockerSandbox()
    why = await asyncio.to_thread(sandbox.available) if sandbox is not None else "no sandbox configured"
    if why:
        sandbox = None
    await store.set_setting("eval_runner", {"sandbox": why is None, "why": why, "gateway": gateway_url,
                                            "seen_at": datetime.now(UTC).isoformat(timespec="seconds")})
    if await interrupted(store):
        log.warning("marked interrupted eval runs as failed")
    while True:
        run = None
        try:
            run = await store.claim_eval_run()
            if run:
                key_id, secret = await store.create_key(run["created_by"], run["team"], "user", name=f"eval run {run['id']}",
                                                        tags={"eval_run": run["id"]}, expires_at=datetime.now(UTC) + timedelta(hours=12))
                try:
                    log.info("running eval %s: %s on %s", run["id"], run["request"].get("suite"), ", ".join(run["request"]["models"]))
                    await execute(store, run, gateway_url, secret, sandbox, concurrency=int(run["request"].get("concurrency", 4)), client=client)
                finally:
                    await store.revoke_key(key_id)
        except Exception:  # noqa: BLE001 (keep serving runs after a database hiccup)
            log.exception("eval runner cycle failed")
        if once:
            return
        if not run:
            await asyncio.sleep(poll)
