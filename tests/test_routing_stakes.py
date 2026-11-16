"""What a failure puts at stake (decision 67, amending 62): the person's cost scaled by the request's level and size, and
to level 5 after a correction; trivial work on the shipped configuration going to the cheapest model that meets its level;
the tie band and the margin for moving a session counted on money, not on a person's time; a new, harder prompt after a
failing run routed as a new task; a substituted model's floor; the escalation charged once; the simulator keeping the
cascade as the gateway does; and the CI check's guard per task class. Every model call is mocked."""

from __future__ import annotations

import copy
import time
from datetime import UTC, datetime, timedelta

import httpx
import pytest
import yaml
from conftest import TEST_CONFIG, TEST_KEY, auth
from test_cost_per_success import agent, build, decide

from meritorder import outcomes
from meritorder import routing_eval as ev
from meritorder.app import build_gateway, create_app
from meritorder.classifier import classify
from meritorder.config import load_catalog, load_config
from meritorder.configstore import load_active
from meritorder.sessions import Session
from meritorder.state import MemoryState
from meritorder.store import MemoryStore
from meritorder.success import STAKES_TOKENS, stakes, track

# ------------------------------------------------------------------ the stakes


def test_stakes_scale_with_the_level_and_the_size_and_a_correction_counts_as_level_5():
    full = stakes(5, 20_000)
    assert full.share == 1 and full.why() == "the full cost"
    assert stakes(4, STAKES_TOKENS).share == pytest.approx(1 / 3) and stakes(1, STAKES_TOKENS).share == pytest.approx(1 / 81)
    small = stakes(2, 2_000)
    assert small.share == pytest.approx(0.25 / 27) and small.why() == "1/27 at level 2, 2,000 of 8,000 tokens"
    fixed = stakes(2, 2_000, corrected=True)
    assert fixed.share == pytest.approx(0.25) and fixed.level == 5 and fixed.why() == "level 5 after a correction, 2,000 of 8,000 tokens"
    assert stakes(9, -5).share == 0 and stakes(0, STAKES_TOKENS).share == pytest.approx(1 / 81)  # kept within 1-5, sizes from 0


def test_the_person_at_stake_is_in_the_reason_and_the_record():
    router, cfg = build(routing={"model_choice": "expected_cost", "human_cost_usd": 3.0})
    d = decide(router, cfg)  # coding:4, 5,000 tokens: a third of the cost, times 5/8
    share = 1 / 3 * 5_000 / STAKES_TOKENS
    assert d.predicted["stakes"] == pytest.approx(share, abs=1e-6) and d.predicted["human_cost"] == pytest.approx(3.0 * share, abs=1e-6)
    assert "of a person's time (of $3.00: 1/3 at level 4, 5,000 of 8,000 tokens)" in d.reason
    big = decide(router, cfg, body=agent(), tokens=40_000, headers={"x-claude-code-request-class": ""})
    assert big.predicted["stakes"] == pytest.approx(1 / 3) and "5,000 of" not in big.reason
    # A correction: the person said the last answer was wrong, so the work matters more than its level said.
    body = {**agent(), "messages": [{"role": "user", "content": "that's wrong, the leak is still there"}]}
    d = decide(router, cfg, body=body, tokens=40_000)
    assert d.predicted["stakes"] == 1.0 and "(of $3.00" not in d.reason


# ------------------------------------------------------------------ trivial work on the shipped configuration

SHORT = {"messages": [{"role": "user", "content": "What's the capital of Portugal?"}]}
TOOLS = [{"name": f"t{i}", "input_schema": {"type": "object"}} for i in range(20)]
TESTS = {"messages": [{"role": "user", "content": "write unit tests for the date parser"}], "tools": TOOLS}
FLAG = {"messages": [{"role": "user", "content": "add a --dry-run flag to the deploy command"}], "tools": TOOLS}


@pytest.fixture(scope="module", params=["shipped", "frozen"])
def worlds(request):
    """The shipped configuration with its API backends on (nothing is ever sent), on config/prices.json and on the
    simulator's frozen prices: a router per model_choice."""
    sim = ev.Simulator(catalog=None if request.param == "shipped" else load_catalog(ev.FROZEN_PRICES))
    return {c: sim.world_for(ev.Strategy(c, team={"model_choice": c})) for c in ("threshold", "expected_cost")}


def first(world, body, tokens):
    return world.router().decide("auto", world.cfg.team(ev.SIM_TEAM), body, {}, None, tokens, True, {}, time.time(), client="openai")


@pytest.mark.parametrize("body,tokens,floor", [(SHORT, 20, "general:1"), (TESTS, 20_000, "coding:2"), (FLAG, 30_000, "coding:3")])
def test_trivial_work_goes_to_the_cheapest_model_that_meets_its_level(worlds, body, tokens, floor):
    """A new session: under decision 62 alone these went to gpt-terra at 26 to 40 times the cost."""
    thr, exp = first(worlds["threshold"], body, tokens), first(worlds["expected_cost"], body, tokens)
    assert str(exp.floor) == str(thr.floor) == floor
    assert exp.candidates[0].model.name == thr.candidates[0].model.name, exp.reason
    assert exp.candidates[0].cost <= thr.candidates[0].cost * 1.0001
    assert exp.predicted["stakes"] <= 1 / 9 and "expected total cost:" in exp.reason


# ------------------------------------------------------------------ the tie band and the margin for moving


def test_the_tie_band_is_a_share_of_money_not_of_a_persons_time():
    router, cfg = build(routing={"model_choice": "expected_cost"})
    d = decide(router, cfg)
    cheap = next(c for c in d.candidates if c.model.name == "mid")
    dear = next(c for c in d.candidates if c.model.name == "strong")
    # Expected totals 1% apart, nearly all of it the person's time: 2% of the cheaper's money is $0.00002, so it stays first.
    cheap.cost, cheap.tokens, cheap.expected = 0.001, 0.001, 0.500
    dear.cost, dear.tokens, dear.expected = 0.010, 0.010, 0.505
    assert router.rank([cheap, dear], "cost", False, time.time())[0][0] is cheap
    # Ranked on money alone (the threshold, cost per success), 1% apart is a tie, and the better model goes first.
    cheap.tokens = dear.tokens = None
    assert router.rank([cheap, dear], "cost", False, time.time())[0][0] is dear


def test_a_session_moves_when_the_saving_beats_the_margin_on_money():
    """A new task in a session warm on mid: strong is $0.022 cheaper in expected total, more than 10% of mid's money plus
    a cent ($0.013), less than 10% of its total with the person's time ($0.038), which used to keep it where it was."""
    router, cfg = build(routing={"model_choice": "expected_cost", "human_cost_usd": 3.6})
    now = time.time()
    s = Session(id="s1", turns=4, model="mid", backend="b", warm_until={"b": now + 300}, prefix_tokens=4_000)
    d = router.decide("auto", cfg.team("default"), agent(), {}, s, 5_000, True, {}, now)
    mid = next(c for c in d.candidates if c.model.name == "mid")
    assert d.candidates[0].model.name == "strong" and "staying on" not in d.reason
    assert mid.spend == mid.tokens < mid.ranked_cost


# ------------------------------------------------------------------ a substituted model's floor


def test_a_substituted_model_is_routed_at_its_score_rounded_up_as_a_clients_floor():
    router, cfg = build(routing={"model_choice": "expected_cost"}, teams={"default": {"substitute_unavailable": True}})
    router.backends["b"].available = False  # mid (coding 3.8) has nowhere to run
    body = {"messages": [{"role": "user", "content": "write unit tests for the date parser"}], "tools": [{"name": "t"}]}
    d = router.decide("mid", cfg.team("default"), body, {}, None, 5_000, True, {}, time.time())
    assert d.substituted_for == "mid" and d.floor.explicit and str(d.floor) == "coding:4", d.reason
    assert {c.model.name for c in d.candidates} == {"strong"} and "the threshold decides" in d.reason
    assert "mid's score for coding, 3.8" in d.reason


# ------------------------------------------------------------------ the gateway: a new task, and the escalation's charge

GW_TOOLS = [{"type": "function", "function": {"name": "bash", "parameters": {"type": "object"}}},
            {"type": "function", "function": {"name": "edit", "parameters": {"type": "object"}}}]


def gw_raw(extra_models: dict | None = None) -> dict:
    raw = copy.deepcopy(TEST_CONFIG)
    raw["models"].update(extra_models or {})
    raw["quality"] = {"source": "config", "measured_best_rate": {"coding": 0.7, "general": 0.7}}
    # The requests are a few dozen tokens, so little of the person's cost is at stake: $20 makes it cents (decision 67).
    raw["routing"] = {"learned": {"mode": "shadow"}, "model_choice": "expected_cost", "human_cost_usd": 20.0,
                      "cascade": {"max_escalations": 1}}
    return raw


async def gateway_for(tmp_path, raw):
    path = tmp_path / "meritorder.yaml"
    path.write_text(yaml.safe_dump(raw))
    gw = build_gateway(load_config(path), store=MemoryStore(), state=MemoryState())
    gw.file_raw = raw
    await load_active(gw)
    return gw


@pytest.fixture
async def ec_gw(tmp_path):
    gw = await gateway_for(tmp_path, gw_raw())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(gateway=gw)), base_url="http://test") as client:
        yield gw, client
    await gw.drain()


def edit_then_failing_test(i: int) -> list[dict]:
    return [{"role": "assistant", "content": None, "tool_calls": [{"id": f"e{i}", "type": "function", "function": {
                "name": "edit", "arguments": '{"file_path": "app.py", "new_string": "x"}'}}]},
            {"role": "tool", "tool_call_id": f"e{i}", "content": "ok"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": f"c{i}", "type": "function", "function": {
                "name": "bash", "arguments": '{"command": "pytest -q"}'}}]},
            {"role": "tool", "tool_call_id": f"c{i}", "content": "FAILED tests/test_x.py::test_a\n1 failed"}]


async def send(client, gw, msgs, sid="e-1", **extra):
    r = await client.post("/v1/chat/completions", headers={**auth(TEST_KEY), "x-session-id": sid},
                          json={"model": "auto", "messages": msgs, "tools": GW_TOOLS, **extra})
    assert r.status_code == 200, r.text
    for _ in range(2):  # the ledger row is written after the answer, and the spend counters after it
        await gw.drain()
    return r


async def test_a_new_harder_prompt_after_a_failing_run_is_a_new_task_at_its_own_level(ec_gw):
    gw, client = ec_gw
    msgs = [{"role": "user", "content": "write a python function that parses dates"}]
    r = await send(client, gw, msgs)
    assert r.headers["x-meritorder-floor"] == "coding:3" and r.headers["x-meritorder-model"] == "small"
    msgs = msgs + edit_then_failing_test(0)
    await send(client, gw, msgs)  # one failed run after an edit: the code is left failing
    harder = "forget that. design the architecture for a multi-region event-sourced billing service and implement the first step"
    assert classify({"messages": [{"role": "user", "content": harder}], "tools": GW_TOOLS}, {}).level == 5
    r = await send(client, gw, msgs + [{"role": "assistant", "content": "I'll stop here."}, {"role": "user", "content": harder}])
    row = (await gw.store.recent(1))[0]
    assert r.headers["x-meritorder-floor"] == "coding:5", row["route_reason"]
    assert row["route_reason"].startswith("classified coding:5") and "escalation" not in row["tags"]
    rows = await gw.store.route_features(datetime.now(UTC) - timedelta(days=1))
    assert len(rows) == 2 and rows[0]["coding"]["ended"]  # the first task ended, with its cost


async def test_a_prompt_no_harder_than_the_task_after_a_failing_run_carries_it_on(ec_gw):
    gw, client = ec_gw
    msgs = [{"role": "user", "content": "write a python function that parses dates"}]
    await send(client, gw, msgs, "e-2")
    msgs = msgs + edit_then_failing_test(0)
    await send(client, gw, msgs, "e-2")
    r = await send(client, gw, msgs + [{"role": "assistant", "content": "Still failing."}, {"role": "user", "content": "keep going"}], "e-2")
    row = (await gw.store.recent(1))[0]
    assert r.headers["x-meritorder-floor"] == "coding:3"
    assert "continuing the task" in row["route_reason"] or row["route_reason"].startswith("escalated at the next human message")
    assert len(await gw.store.route_features(datetime.now(UTC) - timedelta(days=1))) == 1  # still one task


async def test_an_escalation_is_charged_once_when_its_first_model_refuses(tmp_path, monkeypatch):
    """The escalated request's first model refuses and the next one answers: two rows, one charge (on the answer's)."""
    mid = {"quality": {"coding": 4, "general": 4},
           "routes": [{"backend": "mock-anthropic", "model": "mid-1", "price": {"input": 6, "output": 30, "cache_read": 0.6}}]}
    gw = await gateway_for(tmp_path, gw_raw({"mid": mid}))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(gateway=gw)), base_url="http://test") as client:
        msgs = [{"role": "user", "content": "write a python function that parses dates"}]
        assert (await send(client, gw, msgs, "e-3")).headers["x-meritorder-model"] == "small"
        real, refusals = gw.detect_refusal, []

        async def refuse_once(resp, fmt, stream):
            was, resp, usage = await real(resp, fmt, stream)
            if not refusals:
                refusals.append(True)
                return True, resp, usage
            return was, resp, usage

        for i in range(3):  # three failed runs in a row: the third request escalates, its first model refuses
            msgs = msgs + edit_then_failing_test(i)
            if i == 2:
                monkeypatch.setattr(gw, "detect_refusal", refuse_once)
            r = await send(client, gw, msgs, "e-3")
        await gw.drain()
    rows = [x for x in await gw.store.recent(10) if (x.get("tags") or {}).get("escalation")]
    assert refusals and len(rows) == 2 and r.headers["x-meritorder-model"] != "small", [x["route_reason"] for x in rows]
    before = [x for x in await gw.store.recent(10) if not (x.get("tags") or {}).get("escalation")]
    charge = sum(x["baseline_cost"] for x in before)
    charged = [x for x in rows if x["baseline_cost"] < 0]
    assert charge > 0 and len(charged) == 1 and charged[0]["error"] is None, rows
    await gw.drain()


# ------------------------------------------------------------------ the simulator keeps the cascade as the gateway does


def test_track_starts_a_task_and_resets_the_evidence_after_an_escalation():
    task, out = track(None, None, "small", 0.6, {"model": "small", "p": 0.55, "expected": 0.2}, False, spend0=1.0)
    assert task == {"p0": 0.55, "expected": 0.2, "escalations": 0, "spend0": 1.0} and out is None
    assert track(None, None, "big", 0.9, {"model": "small", "p": 0.55}, False)[0]["p0"] == 0.9  # the plan chose another model
    state = {**outcomes.new_state(), "fails_in_row": 3, "errors_in_row": 2, "correction": "wrong", "edits": 4}
    task2, out2 = track({**task, "escalations": 1}, state, "big", 0.9, None, True)
    assert task2["p0"] == 0.9 and task2["escalations"] == 1
    assert (out2["fails_in_row"], out2["errors_in_row"], out2["correction"], out2["edits"]) == (0, 0, None, 4)
    assert track(task2, out2, "big", 0.5, None, False) == (task2, out2)  # neither: nothing changes


def test_the_simulator_sends_each_clients_api_and_reply_length(monkeypatch):
    sim = ev.Simulator(catalog=load_catalog(ev.FROZEN_PRICES), seed=1)
    seen = []
    real = sim.world.router().__class__.decide

    def spy(self, requested, team, body, headers, *a, **k):
        seen.append((k.get("client"), body.get("max_tokens")))
        return real(self, requested, team, body, headers, *a, **k)

    monkeypatch.setattr(ev.Router, "decide", spy)
    tasks = ev.population(60, 2)
    sim.run(ev.Strategy("default"), tasks)
    assert {c for c, _ in seen} == {"anthropic", "responses", "openai"}
    assert (("anthropic", 32_000) in seen and ("responses", None) in seen and ("openai", 4_096) in seen)


def test_the_simulated_cascade_reads_corrections_and_new_tasks_as_the_gateway_does(monkeypatch):
    sim = ev.Simulator(catalog=load_catalog(ev.FROZEN_PRICES), seed=1)
    s = next(x for x in ev.builtin_strategies(sim.world) if x.name == "auto-expected-cost-cascade")
    stages, carried = [], []
    real_decide, real_carries = ev.Router.decide, ev.Router.carries_on

    def spy(self, *a, **k):
        stages.append(k.get("cascade"))
        return real_decide(self, *a, **k)

    def spy_carries(self, session, body, headers, tokens):
        out = real_carries(self, session, body, headers, tokens)
        carried.append((outcomes.correction(body), out))
        return out

    monkeypatch.setattr(ev.Router, "decide", spy)
    monkeypatch.setattr(ev.Router, "carries_on", spy_carries)
    results = sim.run(s, ev.population(150, 5))
    assert "now" in stages and None in stages
    # Every retry is a human message: it carries the task on when it says the last answer was wrong, and only then or
    # when the code was left failing; never forced.
    assert carried and all(out for corr, out in carried if corr)
    assert any(a.escalations for r in results for a in r.attempts)


# ------------------------------------------------------------------ the CI check, per task class


def test_the_check_guards_each_classs_spend_against_the_threshold():
    base = {"success_rate": 0.95, "total_per_finished_with_human": 2.0}
    now = {**base, "by_class": {"chat": {"tasks": 50, "cost_per_finished": 0.00295}, "analysis": {"tasks": 50, "cost_per_finished": 0.05},
                                "coding-very-hard": {"tasks": 4, "cost_per_finished": 90.0}},
           "threshold_by_class": {"chat": {"tasks": 50, "cost_per_finished": 0.0027}, "analysis": {"tasks": 50, "cost_per_finished": 0.04},
                                  "coding-very-hard": {"tasks": 4, "cost_per_finished": 50.0}}}
    # chat 9% over: within the 10%; very hard coding has too few tasks in the check to compare.
    problems = ev.compare(now, base)
    assert len(problems) == 1 and problems[0].startswith("analysis: model spend per finished task $0.05 against the threshold's $0.04")
    assert "+25.0%" in problems[0]
    assert ev.class_problems({}) == [] and ev.class_problems({"by_class": {"chat": {"tasks": 50, "cost_per_finished": 1.0}}}) == []
