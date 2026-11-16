"""Routing by expected total cost to finish a task (routing.model_choice: expected_cost, decision 62): the objective with a
person's cost of a failure and the escalation after it, the success predictor's layers (the benchmark prior between levels,
an unsure difficulty, the fitted layer trained on outcomes and checked against the prior), the cascade inside a task and
its bounds, what the gateway records (ledger tags, the escalation charged against routing savings, the task's row), the
report, and a synthetic comparison with today's choices. Every model call is mocked."""

from __future__ import annotations

import copy
import math
import random
import time
from datetime import UTC, datetime, timedelta

import httpx
import pytest
import yaml
from conftest import ADMIN_KEY, TEST_CONFIG, TEST_KEY, auth
from pydantic import ValidationError
from test_cost_per_success import agent, build, decide

from meritorder import learned_jobs, outcomes, success
from meritorder.app import build_gateway, create_app
from meritorder.classifier import Floor
from meritorder.config import CascadeConfig, RoutingConfig, TeamConfig, load_config
from meritorder.configstore import load_active, token
from meritorder.learned import LearnedCall, Prediction
from meritorder.learned import Features as LFeatures
from meritorder.sessions import Session
from meritorder.state import MemoryState
from meritorder.store import MemoryStore
from meritorder.success import (Difficulty, Example, Option, Predictor, SuccessModel, TaskContext, continues, plan, posterior,
                                prior_logit, project_hash, repo_hash, triage_buckets)


def ec(human=1.0, **routing):
    return build(routing={"model_choice": "expected_cost", "human_cost_usd": human, "cascade": {"max_escalations": 1}, **routing})


# ------------------------------------------------------------------ settings


def test_settings_and_defaults():
    r = RoutingConfig()
    assert r.model_choice == "expected_cost" and r.human_cost_usd == 1.0  # the default since the simulator; a minute at $60 an hour
    assert r.cascade.max_escalations == 0 and r.cascade.failures_now == 3 and r.success.mode == "off"
    assert TeamConfig(name="t", model_choice="expected_cost", human_cost_usd=0).human_cost_usd == 0
    for bad in ({"human_cost_usd": -1},):
        with pytest.raises(ValidationError):
            RoutingConfig(**bad)
        with pytest.raises(ValidationError):
            TeamConfig(name="t", **bad)
    with pytest.raises(ValidationError):
        CascadeConfig(max_escalations=3)
    with pytest.raises(ValidationError):
        CascadeConfig(failures_now=0)


# ------------------------------------------------------------------ the prior and the objective


def test_the_prior_reads_between_levels_in_log_odds():
    ch = {1: 0.95, 2: 0.9, 3: 0.8, 4: 0.7, 5: 0.5}
    assert math.isclose(prior_logit(ch, 3), math.log(0.8 / 0.2))
    mid = 1 / (1 + math.exp(-prior_logit(ch, 3.5)))
    assert 0.7 < mid < 0.8
    assert prior_logit(ch, 0.2) == prior_logit(ch, 1) and prior_logit(ch, 9) == prior_logit(ch, 5)  # kept within 1-5
    assert prior_logit({}, 3) is None and prior_logit({3: 0.8}, 3.4) == pytest.approx(math.log(4))  # only whole levels known


def test_plan_counts_the_persons_time_and_the_attempt_after_a_failure():
    cheap, strong = Option("c", "cheap", 0.01, 0.012, 0.6), Option("s", "strong", 0.10, 0.12, 0.9)
    v = plan([cheap, strong], human=0.0, escalations=0)
    assert v["c"].total == pytest.approx(0.01) and v["s"].total == pytest.approx(0.10)  # no person's cost, no next attempt: the try
    v = plan([cheap, strong], human=1.0, escalations=0)
    assert v["c"].total == pytest.approx(0.01 + 0.4) and v["s"].total == pytest.approx(0.10 + 0.1)
    # With nothing for the person to lose, a failure is simply tried again where it is.
    v = plan([cheap, strong], human=0.0, escalations=1)
    assert v["c"].next.model == "cheap" and v["c"].total == pytest.approx(0.01 + 0.4 * 0.01)
    # With a dollar of the person's time, cheap's failure escalates to strong from a cold cache, whose chance is lower after a
    # failure than on its own (the same task is hard for it too), so it costs more than strong's plain 0.12 + 10% x $1.
    v = plan([cheap, strong], human=1.0, escalations=1)
    assert v["c"].next.model == "strong"
    after = (v["c"].total - 0.01) / 0.4 - 1.0
    assert 0.12 + 0.1 < after < 0.12 + 0.5
    assert v["c"].tokens == pytest.approx(0.01 + 0.4 * 0.12)
    assert v["s"].next.model == "strong"  # nothing stronger: strong again
    # Two routes of a model: an escalation goes to its cheapest cold route.
    other = Option("s2", "strong", 0.2, 0.05, 0.9)
    assert plan([cheap, strong, other], 1.0, 1)["c"].next.key == "s2"


def test_posterior_and_when_a_message_carries_on_the_task():
    p, parts = posterior(0.7, None)
    assert p == pytest.approx(0.7) and parts == []
    s = {**outcomes.new_state(), "edits": 2, "last_test": "fail", "fails_in_row": 3, "fix_loop": 3, "errors_in_row": 2,
         "correction": "wrong", "files": 9, "kinds": {"core": 8}}
    p, parts = posterior(0.7, s)
    assert p < 0.1 and {n for n, _ in parts} >= {"tests failing after the last edit", "3 failed runs in a row", "a fix loop",
                                                    "corrected (wrong)", "9 files touched", "8 core files edited",
                                                    "2 error results in a row"}
    assert posterior(0.7, {"last_test": "pass"})[0] > 0.7
    assert posterior(0.7, {"last_build": "fail"})[0] < 0.7
    assert continues({"correction": "undo"}) and continues({"edits": 1, "last_test": "fail"}) and continues({"edits": 1, "last_build": "fail"})
    assert not continues({"edits": 1, "last_test": "pass"}) and not continues(None) and not continues({"last_test": "fail"})
    assert success.failures_in_row({"fails_in_row": 2, "errors_in_row": 4}) == 4 and success.usd(0) == "$0"


# ------------------------------------------------------------------ the router


def test_expected_cost_picks_by_total_cost_and_says_why():
    router, cfg = ec(human=0.0)
    d = decide(router, cfg)
    # No person's cost: the cheapest try among the models in reach (weak's chance at coding:4 is under the 60% minimum).
    assert [c.model.name for c in d.candidates][:2] == ["mid", "strong"]
    assert "expected total cost: mid ~" in d.reason and "a failure costs $0 of a person's time, then mid again; strong ~" in d.reason
    p = d.predicted
    assert p["model"] == "mid" and p["runner_up"]["model"] == "strong" and p["human_cost"] == 0.0 and p["then"] == "mid"
    assert p["p"] == pytest.approx(cfg.model("mid").chances["coding"][4], abs=1e-3) and p["runner_up"]["p"] > p["p"]
    assert p["expected"] >= p["try"] and p["why"].startswith(f"{p['p']:.0%}")
    # A person's time worth more than the price difference: the stronger model.
    router, cfg = ec(human=5.0)
    d = decide(router, cfg)
    assert d.candidates[0].model.name == "strong" and d.predicted["runner_up"]["model"] == "mid"
    # A team's own cost of a failure.
    router, cfg = build(routing={"model_choice": "expected_cost"}, teams={"default": {}, "cheap": {"human_cost_usd": 0}})
    assert decide(router, cfg, team="cheap").candidates[0].model.name == "mid" and router.human_cost(cfg.team("cheap")) == 0
    assert router.human_cost(cfg.team("default")) == 1.0


def test_a_clients_floor_and_missing_chances_fall_back_as_cost_per_success_does():
    router, cfg = ec(human=0.0)
    d = decide(router, cfg, headers={"x-route-floor": "coding:5"})
    assert {c.model.name for c in d.candidates} == {"strong"} and "the threshold decides" in d.reason
    router, cfg = build(live=False, routing={"model_choice": "expected_cost"})
    d = decide(router, cfg)
    assert d.predicted is None and "expected total cost: no chances of success for coding" in d.reason


def test_a_learned_difficulty_and_its_spread_widen_the_prior():
    router, cfg = ec()
    floor = Floor("coding", 4, "rules")
    assert router.difficulty(floor, None).value == 4.0
    pred = Prediction(1, "coding", 4, 3, 0.7, 3.8, 0.9)
    call = LearnedCall("route", Floor("coding", 3, "rules"), LFeatures("coding", 3), "f", prediction=pred, used=True)
    cfg.routing.learned.mode = "shadow"
    assert router.difficulty(floor, call).spread == 0.0  # shadow mode changes nothing
    cfg.routing.learned.mode = "route"
    d = router.difficulty(floor, call)
    assert d.value == pytest.approx(3.8) and d.spread == 0.9 and "learned router v1" in str(d)
    sure = router.success.chance(cfg.model("mid"), "coding", Difficulty(4.0), None, "off", 0.7)
    unsure = router.success.chance(cfg.model("mid"), "coding", Difficulty(4.0, 1.5), None, "off", 0.7)
    assert abs(unsure.p - 0.5) < abs(sure.p - 0.5) and "unsure difficulty" in unsure.why()
    # Without estimates: a model at the level gets the chance of one exactly at it, and one below half of that.
    plain = cfg.model("strong").model_copy(update={"chances": {}})
    assert router.success.chance(plain, "coding", Difficulty(4.0), None, "off", 0.7).p == 0.7
    weak = cfg.model("weak").model_copy(update={"chances": {}})
    assert router.success.chance(weak, "coding", Difficulty(4.0), None, "off", 0.7).why().endswith("(no estimate)")


def session_on(model: str, outcome: dict, task: dict | None = None, floor="coding:4") -> Session:
    return Session(id="s", model=model, backend={"strong": "a", "mid": "b", "weak": "c"}[model], floor=floor, floor_reason="agent",
                   turns=3, prefix_tokens=5_000, warm_until={"b": time.time() + 300}, outcome=outcome,
                   task=task if task is not None else {"p0": 0.68, "escalations": 0})


FAILING = {**outcomes.new_state(), "edits": 3, "last_test": "fail", "fails_in_row": 3, "red": True}


def mid_task(router, cfg, session, cascade="now", up=True, model="auto", team="default"):
    return router.decide(model, cfg.team(team), agent(), {}, session, 5_000, False, {}, time.time(), cascade=cascade, explore_up=up)


def test_the_cascade_escalates_when_continuing_costs_more():
    router, cfg = ec(human=1.0)
    d = mid_task(router, cfg, session_on("mid", FAILING))
    assert d.escalation["from"] == "mid" and d.escalation["to"] == "strong" and d.escalation["n"] == 1
    assert d.escalation["stage"] == "now" and d.escalation["reason"] == "tests failing"
    assert d.escalation["p_now"] < d.escalation["p_before"] and d.escalation["escalate"] < d.escalation["continue"]
    assert d.candidates[0].model.name == "strong" and all(c.model.name != "mid" for c in d.candidates)
    assert d.reason.startswith("escalated after failures in a row (1 of 1): mid's chance now")
    assert "cache rebuild included" in d.reason
    # At a human message that carries on the task: a correction is enough, without failures in a row.
    d = mid_task(router, cfg, session_on("mid", {**outcomes.new_state(), "correction": "wrong"}), cascade="human")
    assert d.escalation["stage"] == "human" and d.escalation["reason"] == "a correction"
    assert d.reason.startswith("escalated at the next human message")


@pytest.mark.parametrize("case", ["few failures", "used up", "budget", "named", "strongest", "cheap person", "threshold team"])
def test_where_the_cascade_doesnt_escalate(case):
    router, cfg = ec(human=0.0001 if case == "cheap person" else 1.0, cascade={"max_escalations": 1})
    if case == "threshold team":
        router, cfg = build(routing={"model_choice": "expected_cost"}, teams={"default": {"model_choice": "threshold"}})
    state = {**FAILING, "fails_in_row": 1} if case == "few failures" else FAILING
    task = {"p0": 0.68, "escalations": 1} if case == "used up" else None
    session = session_on("strong" if case == "strongest" else "mid", state, task)
    d = mid_task(router, cfg, session, up=case != "budget", model="mid" if case == "named" else "auto")
    assert d.escalation is None
    assert d.candidates[0].model.name == session.model
    if case == "used up":
        assert "escalations used up (1 of 1)" in d.reason
    if case == "cheap person":
        assert "continuing ~" in d.reason and d.reason.endswith(": continuing")


def test_a_continuing_message_keeps_the_task_model_when_it_doesnt_escalate():
    router, cfg = ec(human=0.0001)
    d = mid_task(router, cfg, session_on("mid", {**outcomes.new_state(), "correction": "wrong"}), cascade="human")
    assert d.escalation is None and d.reason.startswith("continuing the task: keeping session model mid")


# ------------------------------------------------------------------ context and triage features


def test_repositories_projects_and_the_opening_prompt_are_hashed():
    seed = b"k" * 16
    cc = {"system": [{"type": "text", "text": "You are Claude Code.\nPrimary working directory: /home/ana/src/payments\n"}], "messages": []}
    codex = {"messages": [{"role": "user", "content": "<environment_context><cwd>/home/ana/src/payments</cwd></environment_context>"}]}
    h = repo_hash(cc, seed)
    assert h == repo_hash(codex, seed) and len(h) == 8 and "payments" not in h
    assert repo_hash(cc, b"j" * 16) != h and repo_hash({"messages": [{"role": "user", "content": "hi"}]}, seed) is None
    assert repo_hash({"system": "Working directory: /a/b", "messages": []}, seed) is not None
    assert project_hash({"project": "atlas"}, seed) and project_hash({}, seed) is None
    buckets = triage_buckets("Fix the flaky test in tests/test_api.py and src/app/main.py, then update docs/README.md", seed)
    assert buckets == sorted(set(buckets)) and all(0 <= b < success.TRIAGE_BUCKETS for b in buckets)
    assert triage_buckets("", seed) == [] and triage_buckets("hi", seed) == []
    assert [outcomes.path_kind(p) for p in ("tests/test_a.py", "docs/x.md", "pyproject.toml", "src/a.py", "web/a.test.ts")] == \
        ["test", "docs", "config", "core", "test"]


def test_outcomes_keep_runs_and_errors_in_a_row_and_the_scope():
    def turn(state, name, args, result, error=None):
        body = {"messages": [{"role": "assistant", "content": [{"type": "tool_use", "id": "x", "name": name, "input": args}]},
                             {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "x", "content": result,
                                                           **({"is_error": error} if error is not None else {})}]}]}
        return outcomes.update(state, body, 0, 100)[0]

    s = turn(None, "Edit", {"file_path": "src/a.py", "new_string": "x = 1\n"}, "ok")
    s = turn(s, "Write", {"file_path": "tests/test_a.py", "content": "def test(): pass\n"}, "ok")
    s = turn(s, "Bash", {"command": "pytest -q"}, "1 failed", True)
    s = turn(s, "Bash", {"command": "pytest -q"}, "2 failed", True)
    assert s["fails_in_row"] == 2 and s["errors_in_row"] == 2 and s["files"] == 2 and s["kinds"] == {"core": 1, "test": 1}
    assert s["edit_chars"] == len("x = 1\n") + len("def test(): pass\n")
    s = turn(s, "Bash", {"command": "pytest -q"}, "3 passed", False)
    assert s["fails_in_row"] == 0 and s["errors_in_row"] == 0
    patch = {"input": "*** Begin Patch\n*** Update File: src/b.py\n+line one\n+two\n*** End Patch"}
    assert turn(None, "apply_patch", patch, "Done")["edit_chars"] == len("line one") + len("two")
    multi = {"file_path": "src/c.py", "edits": [{"new_string": "abc"}, {"new_string": "de"}]}
    assert turn(None, "MultiEdit", multi, "ok")["edit_chars"] == 5


# ------------------------------------------------------------------ the fitted layer


def synthetic(n=3000, seed=7) -> list[Example]:
    """Task starts where the benchmark prior is too pessimistic (people see fewer failures than FrontierSWE's grading), and
    one model ('mid') and one repository are better than the rest, which the fitted layer should learn."""
    rnd = random.Random(seed)
    out = []
    for i in range(n):
        model = rnd.choice(["mid", "strong", "weak"])
        z = {"mid": 0.4, "strong": 1.2, "weak": -0.3}[model] + rnd.gauss(0, 0.3)
        repo = rnd.choice(["r1", "r2", "r3"])
        true = z + 1.0 + (0.8 if model == "mid" else 0.0) + (-1.0 if repo == "r3" else 0.0)
        y = 1.0 if rnd.random() < 1 / (1 + math.exp(-true)) else 0.0
        out.append(Example("coding", model, z, y, "default", repo, "", [i % 50, 50 + (i % 7)], float(i)))
    return out


def test_the_fitted_layer_calibrates_and_learns_abilities_and_is_checked_against_the_prior():
    res = success.train(synthetic(), "f3-x", datetime(2026, 11, 15, tzinfo=UTC))
    assert res["better"] and res["metrics"]["holdout"]["brier"] < res["metrics"]["holdout"]["prior_brier"]
    m = SuccessModel(res["doc"], 3)
    g = m.groups["coding"]
    assert g["model:mid"] > g["model:strong"] and g["repo:r3"] < g["repo:r1"]
    assert m.cal["coding"][0] > 0.3  # outcomes count more successes than the benchmark prior
    z, parts = m.logit(Example("coding", "mid", 0.4, 1.0, "default", "r3", "", [1]))
    assert {n for n, _ in parts} >= {"calibration", "model ability", "repo", "team", "opening prompt"}
    # Too few, or no better than the prior: not used.
    small = success.train(synthetic(100), "f")
    assert small["doc"] is None and "at least 200" in small["why"]
    noise = [Example("coding", "mid", 0.0, 1.0 if i % 2 else 0.0, ts=float(i)) for i in range(400)]
    flat = success.train(noise, "f")
    assert not flat["better"] and "no better than the benchmark prior" in flat["why"]
    with pytest.raises(ValueError):
        SuccessModel({"format": 9})


def test_the_predictor_uses_the_fitted_layer_only_in_route_mode_and_once_it_beat_the_prior():
    router, cfg = ec()
    res = success.train(synthetic(), "f")
    model = SuccessModel(res["doc"], 1)
    pred = Predictor(model)
    ctx = TaskContext("default", "r3", None, [])
    mid = cfg.model("mid")
    shadow = pred.chance(mid, "coding", Difficulty(4.0), ctx, "shadow", 0.7)
    assert shadow.fitted is not None and shadow.p == shadow.prior  # recorded, not used
    used = pred.chance(mid, "coding", Difficulty(4.0), ctx, "route", 0.7)
    assert used.p == used.fitted and "model ability" in used.why()
    model.better = False
    assert pred.chance(mid, "coding", Difficulty(4.0), ctx, "route", 0.7).p == shadow.prior
    assert pred.chance(mid, "coding", Difficulty(4.0), None, "route", 0.7).fitted is None


def test_the_report_compares_predicted_with_seen_success_escalations_and_costs():
    now = datetime.now(UTC)

    def row(i, p, ok, esc=0, cost=0.05):
        coding = {"ended": True, "cost": cost, "escalations": esc, "escalation_reasons": ["tests failing"] * esc,
                  "escalation_cost": 0.01 * esc, "last_test": "pass" if ok else "fail"}
        return {"request_id": f"r{i}", "ts": now, "thread": f"t{i}", "user_id": "u", "session_id": f"s{i}", "tags": [], "status": 200,
                "text_features": [], "served_model": "mid", "predicted": {"p": p, "fitted": p + 0.05, "expected": 0.04},
                "coding": coding, "retried": False}

    rows = [row(i, 0.8, i % 5 != 0) for i in range(20)] + [row(99, 0.4, True, esc=1), {**row(98, 0.5, True), "predicted": None}]
    rep = success.report(rows)
    assert rep["predicted"]["n"] == 21 and rep["predicted"]["brier"] is not None and rep["predicted"]["bins"]
    assert rep["fitted"]["n"] == 21 and rep["by_model"][0]["model"] == "mid"
    assert rep["escalations"] == {"tasks": 21, "escalated": 1, "rate": round(1 / 21, 4), "helped": 1, "helped_rate": 1.0,
                                  "extra_cost": 0.01, "by_reason": {"tests failing": 1}}
    assert rep["cost"]["tasks"] == 21 and rep["cost"]["expected"] == 0.04 and rep["cost"]["actual"] == 0.05
    assert success.report([])["predicted"]["n"] == 0 and success.reliability([(1.0, 1.0)])[0]["to"] == 1.0


# ------------------------------------------------------------------ against today's choices, on synthetic tasks


def simulate(human: float, n: int = 600) -> dict:
    """Synthetic agent tasks at coding:3-5, each served by threshold (today's default), cost_per_success and expected_cost
    (with its cascade), on the router's own prices and choices. Each task has a hidden difficulty of its own (a normal
    draw in log-odds, the same for every model), so a task one model failed is harder for the next one too; on average a
    model succeeds about as often as its chance at the level says. A failure costs the person ``human`` times what the
    task puts at stake (Router.stakes: its level and size, decision 67) and the task is tried once more: today on the same
    choice (the person resends), with the cascade on the model the plan escalates to. The same draws for every policy.
    Returns the mean cost per task: tokens, failures, and the total with the person's."""
    out = {}
    for choice in ("threshold", "cost_per_success", "expected_cost"):
        router, cfg = build(routing={"model_choice": choice, "human_cost_usd": human})
        rnd = random.Random(11)
        tokens = fails = person = 0.0
        for i in range(n):
            hidden, u1, u2 = rnd.gauss(0, 1), rnd.random(), rnd.random()
            tools = 5 if i % 3 else 2
            text = "fix the race condition in the scheduler" if i % 5 == 0 else "fix the failing test in app.py"
            body = {"messages": [{"role": "user", "content": text}], "tools": [{"name": f"t{j}"} for j in range(tools)]}
            size = 5_000 + 1_000 * (i % 7)
            d = router.decide("auto", cfg.team("default"), body, {}, None, size, True, {}, time.time())
            level = d.floor.level
            at_stake = human * router.stakes(d.floor, body, size).share

            def solves(c, u) -> bool:
                ch = c.model.chances["coding"][level]
                return u < 1 / (1 + math.exp(-(math.log(ch / (1 - ch)) - hidden)))

            first = d.candidates[0]
            tokens += first.cost
            if solves(first, u1):
                continue
            fails += 1
            person += at_stake
            nxt = (d.predicted or {}).get("then") if choice == "expected_cost" else None
            second = next((c for c in d.candidates if c.model.name == nxt), first)
            tokens += second.cold if second.cold is not None else second.cost
            if not solves(second, u2):
                fails += 1
                person += at_stake
        out[choice] = {"tokens": round(tokens / n, 5), "failures": round(fails / n, 4), "total": round((tokens + person) / n, 5)}
    return out


def test_expected_cost_costs_least_in_total_on_synthetic_tasks():
    for human in (0.25, 1.0, 5.0, 25.0):
        r = simulate(human)
        assert r["expected_cost"]["total"] <= min(r["threshold"]["total"], r["cost_per_success"]["total"]), r
    cheap = simulate(0.0)  # with no person's cost it never pays more in tokens than the threshold
    assert cheap["expected_cost"]["tokens"] <= cheap["threshold"]["tokens"]


# ------------------------------------------------------------------ the gateway


def gw_config(tmp_path, human=20.0, success_mode="off"):
    """A gateway routing by expected cost with one escalation. These requests are a few dozen tokens at coding:3, so a
    failure puts little of the person's cost at stake (Router.stakes): $20 makes it a few cents, as a real agent's turns."""
    raw = copy.deepcopy(TEST_CONFIG)
    raw["quality"] = {"source": "config", "measured_best_rate": {"coding": 0.7, "general": 0.7}}
    raw["routing"] = {"learned": {"mode": "shadow"}, "model_choice": "expected_cost", "human_cost_usd": human,
                      "success": {"mode": success_mode}, "cascade": {"max_escalations": 1}}
    path = tmp_path / "meritorder.yaml"
    path.write_text(yaml.safe_dump(raw))
    return load_config(path), raw


@pytest.fixture
async def ec_gw(tmp_path):
    cfg, raw = gw_config(tmp_path)
    gw = build_gateway(cfg, store=MemoryStore(), state=MemoryState())
    gw.file_raw = raw
    await load_active(gw)
    app = create_app(gateway=gw)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        yield gw, client
    await gw.drain()


TOOLS = [{"type": "function", "function": {"name": "bash", "parameters": {"type": "object"}}}]


def failing_turn(i: int) -> list[dict]:
    return [{"role": "assistant", "content": None, "tool_calls": [{"id": f"c{i}", "type": "function",
                                                                    "function": {"name": "bash", "arguments": '{"command": "pytest -q"}'}}]},
            {"role": "tool", "tool_call_id": f"c{i}", "content": "FAILED tests/test_x.py::test_a\n1 failed"}]


async def send(client, msgs, sid="e-1", gw=None):
    body = {"model": "auto", "messages": msgs, "tools": TOOLS}
    r = await client.post("/v1/chat/completions", headers={**auth(TEST_KEY), "x-session-id": sid}, json=body)
    assert r.status_code == 200, r.text
    if gw is not None:
        for _ in range(2):  # the ledger row is written after the answer, and the spend counters after it, in the background
            await gw.drain()
    return r


async def test_a_task_escalates_after_failures_and_the_ledger_says_so(ec_gw):
    gw, client = ec_gw
    msgs = [{"role": "system", "content": "Working directory: /srv/repo"},
            {"role": "user", "content": "write a python function that parses dates"}]
    r = await send(client, msgs, gw=gw)
    assert r.headers["x-meritorder-model"] == "small"
    first = (await gw.store.recent(1))[0]
    assert "expected total cost: small" in first["route_reason"]
    feats = (await gw.store.route_features(datetime.now(UTC) - timedelta(days=1)))[0]
    assert feats["predicted"]["model"] == "small" and feats["predicted"]["runner_up"]["model"] == "big"
    assert len(feats["context"]["repo"]) == 8 and feats["context"]["project"] is None
    for i in range(3):  # three failed test runs in a row
        msgs = msgs + failing_turn(i)
        r = await send(client, msgs, gw=gw)
    assert r.headers["x-meritorder-model"] == "big"
    await gw.drain()
    row = (await gw.store.recent(1))[0]
    assert row["tags"]["escalation"] == "1" and row["tags"]["escalated_from"] == "small"
    assert row["route_reason"].startswith("escalated after failures in a row (1 of 1)")
    # The failed attempt is charged against routing: what it would have cost on the baseline model.
    # The escalated model is the baseline here, so routing saved nothing on it: what's left is the charge, the baseline
    # cost of the three turns before.
    before = [x for x in await gw.store.recent(10) if x["id"] != row["id"]]
    assert row["savings_routing"] == pytest.approx(-sum(x["baseline_cost"] for x in before), rel=1e-6)
    # Its later turns keep the tags; a client can't set them.
    msgs = msgs + failing_turn(9)
    r = await client.post("/v1/chat/completions", headers={**auth(TEST_KEY), "x-session-id": "e-1",
                                                           "x-meritorder-tags": "escalation=7"},
                          json={"model": "auto", "messages": msgs, "tools": TOOLS})
    await gw.drain()
    row = (await gw.store.recent(1))[0]
    assert row["tags"]["escalation"] == "1" and r.headers["x-meritorder-model"] == "big"
    # A passing run, then a new task: the first one's row gets its cost and escalations.
    msgs = msgs + [{"role": "assistant", "content": None, "tool_calls": [{"id": "ok", "type": "function",
                                                                          "function": {"name": "bash", "arguments": '{"command": "pytest -q"}'}}]},
                   {"role": "tool", "tool_call_id": "ok", "content": "5 passed"}]
    await send(client, msgs, gw=gw)
    await send(client, msgs + [{"role": "assistant", "content": "done"}, {"role": "user", "content": "now add a docstring"}], gw=gw)
    await gw.drain()
    rows = await gw.store.route_features(datetime.now(UTC) - timedelta(days=1))
    coding = rows[0]["coding"]
    assert coding["ended"] and coding["escalations"] == 1 and coding["escalation_reasons"] == ["tests failing"]
    assert coding["cost"] > 0 and coding["escalation_cost"] > 0
    # The report.
    rep = await client.get("/v1/reports/router", headers=auth(ADMIN_KEY))
    s = rep.json()["success"]
    assert s["escalations"]["escalated"] == 1 and s["predicted"]["n"] >= 1 and s["cost"]["tasks"] == 1
    assert rep.json()["status"]["success"]["mode"] == "off"
    lv = (await client.get("/v1/reports/levels", headers=auth(TEST_KEY))).json()
    assert lv["model_choice"] == "expected_cost" and lv["human_cost_usd"] == 20.0 and lv["max_escalations"] == 1


async def test_a_correction_carries_on_the_task_and_escalates_at_the_human_message(ec_gw):
    gw, client = ec_gw
    msgs = [{"role": "user", "content": "write a python function that parses dates"}]
    assert (await send(client, msgs, "e-2", gw)).headers["x-meritorder-model"] == "small"
    msgs = msgs + [{"role": "assistant", "content": "here"}, {"role": "user", "content": "that's wrong, it ignores time zones"}]
    r = await send(client, msgs, "e-2", gw)
    assert r.headers["x-meritorder-model"] == "big"
    await gw.drain()
    row = (await gw.store.recent(1))[0]
    assert row["route_reason"].startswith("escalated at the next human message") and row["tags"]["escalation"] == "1"
    rows = await gw.store.route_features(datetime.now(UTC) - timedelta(days=1))
    assert len(rows) == 1 and rows[0]["coding"]["correction"] == "wrong"  # still one task: the correction is its outcome


async def test_the_fitted_layer_is_trained_saved_and_loaded(tmp_path):
    cfg, raw = gw_config(tmp_path, success_mode="shadow")
    gw = build_gateway(cfg, store=MemoryStore(), state=MemoryState())
    gw.file_raw = raw
    await load_active(gw)
    assert gw.router.success.model is None
    now = datetime.now(UTC)
    rows = []
    rnd = random.Random(3)
    for i in range(400):
        model = "small" if i % 2 else "big"
        ok = rnd.random() < (0.95 if model == "big" else 0.85)
        rows.append({"request_id": f"{i:08d}-0000-0000-0000-000000000000", "ts": now - timedelta(minutes=400 - i), "team": "default",
                     "user_id": "u", "session_id": f"s{i}", "thread": f"t{i}", "fv": gw.learned.fv, "task": "coding", "rules_level": 3,
                     "text_features": [], "tags": [], "auto": True, "served_model": model, "served_score": 3.0, "refused_scores": [],
                     "status": 200, "retried": False, "level": 3, "coding": {"last_test": "pass" if ok else "fail"},
                     "context": {"repo": "abcd1234"}, "triage_features": [i % 9]})
    out = await learned_jobs.train_success(gw.cfg, gw.store, gw.state, rows, gw.learned.fv, now)
    assert out["version"] == 1 and out["metrics"]["examples"] == 400
    await load_active(gw)
    assert gw.router.success.model is not None and gw.router.success.model.version == 1
    assert await load_active(gw) is False  # nothing changed
    status = await learned_jobs.status(gw.cfg, gw.store, gw.learned, gw.state)
    assert status["success"]["version"] == 1 and status["success"]["mode"] == "shadow"
    # Turned off: the benchmark prior alone again.
    gw.cfg.routing.success.mode = "off"
    await gw.load_success(1)
    assert gw.router.success.model is None
    # A document that can't be read leaves the prior in charge, and says why.
    gw.cfg.routing.success.mode = "route"
    await gw.store.set_setting(success.SETTING, {"format": 7})
    await gw.load_success(2)
    assert gw.router.success.model is None and "couldn't load success model v2" in gw.router.success.error
    await gw.store.set_setting(success.SETTING, None)
    await gw.load_success(3)
    assert "LookupError" in gw.router.success.error
    assert token(1, None, 2, 3).endswith(":r2:s3") and token(1, None) == "1::"
    await gw.drain()


async def test_escalating_is_never_why_a_request_is_refused(ec_gw):
    gw, client = ec_gw
    gw.cfg.safeguards.max_request_usd = 0.005  # big's reply could cost more; small's can't
    msgs = [{"role": "user", "content": "write a python function that parses dates"}]
    await send(client, msgs, "e-3", gw)
    for i in range(3):
        msgs = msgs + failing_turn(i)
        body = {"model": "auto", "messages": msgs, "tools": TOOLS, "max_tokens": 1000}
        r = await client.post("/v1/chat/completions", headers={**auth(TEST_KEY), "x-session-id": "e-3"}, json=body)
        assert r.status_code == 200 and r.headers["x-meritorder-model"] == "small"
    await gw.drain()
    row = (await gw.store.recent(1))[0]
    assert "escalation" not in row["tags"] and row["savings_routing"] >= 0


def test_the_simulator_drives_the_routers_own_cascade():
    """The routing simulator's strategy for expected_cost with its cascade (routing_eval.RouterCascade): the session keeps
    the task's chance and evidence, and some tasks escalate."""
    from meritorder import routing_eval as ev
    from meritorder.config import load_catalog

    sim = ev.Simulator(catalog=load_catalog(ev.FROZEN_PRICES), seed=1)
    s = next(x for x in ev.builtin_strategies(sim.world) if x.name == "auto-expected-cost-cascade")
    results = sim.run(s, ev.population(120, 5))
    assert any(a.escalations for r in results for a in r.attempts)
    assert ev.summarize(results, sim.human_cost)["success_rate"] > 0.9
