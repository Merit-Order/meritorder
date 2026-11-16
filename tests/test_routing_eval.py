"""The offline routing simulator (routing_eval.py): a population calibrated to public benchmarks, execution through the
real router with the provider's cache rules, the scorecard, replay with inverse-propensity weights, and the regression
check CI runs against the stored baseline. Nothing here calls a model."""

import json
import math
import random

import pytest

from meritorder import outcomes
from meritorder import routing_eval as ev
from meritorder.config import load_catalog

FROZEN = load_catalog(ev.FROZEN_PRICES)


@pytest.fixture(scope="module")
def sim():
    return ev.Simulator(catalog=FROZEN, seed=3)


# ------------------------------------------------------------------ the regression check (CI)

def test_the_default_strategy_is_no_worse_than_the_baseline():
    """The default strategy on the standard population, with the frozen prices and scores, against the stored baseline.
    A change that's worse on purpose: `uv run python scripts/routing_sim.py check --update` (docs/11-evals.md)."""
    now = ev.check_run(catalog=FROZEN)
    baseline = json.loads(ev.BASELINE.read_text())
    assert (now["tasks"], now["seed"], now["human_cost"]) == (baseline["tasks"], baseline["seed"], baseline["human_cost"])
    assert ev.compare(now, baseline) == []


def test_compare_names_what_got_worse():
    base = {"success_rate": 0.95, "total_per_finished_with_human": 2.0}
    assert ev.compare({"success_rate": 0.95, "total_per_finished_with_human": 2.09}, base) == []
    worse = ev.compare({"success_rate": 0.92, "total_per_finished_with_human": 2.2}, base)
    assert len(worse) == 2 and "rose from $2.0000 to $2.2000" in worse[0] and "fell from 95.0% to 92.0%" in worse[1]
    assert ev.compare({"success_rate": 0.99, "total_per_finished_with_human": 1.0}, base) == []  # better is fine


# ------------------------------------------------------------------ the population and its calibration

def test_the_population_is_reproducible_and_follows_the_mix():
    a, b = ev.population(300, seed=5), ev.population(300, seed=5)
    assert [(t.cls, t.prompt, t.difficulty) for t in a] == [(t.cls, t.prompt, t.difficulty) for t in b]
    assert [t.prompt for t in ev.population(50, seed=6)] != [t.prompt for t in a[:50]]
    coding = [t for t in a if t.task == "coding"]
    assert 0.6 < len(coding) / len(a) < 0.8
    assert any(t.misleading for t in a) and any(t.ambiguous for t in a)
    assert {t.client for t in coding} == {"claude-code", "small-agent", "chat-with-code"}
    only = ev.population(40, seed=1, shares=ev.shares_of('{"coding-hard": 1}'))
    assert {t.cls for t in only} == {"coding-hard"} and all(t.turns >= 30 or t.client == "chat-with-code" for t in only)
    with pytest.raises(ValueError, match="unknown task classes"):
        ev.shares_of('{"coding-impossible": 1}')
    assert abs(sum(ev.MIXES["harder"].values()) - 1) < 1e-9 and ev.MIXES["chat-only"]["coding-easy"] == 0


def test_misleading_wording_fools_the_rules():
    rng = random.Random(1)
    quick = ev.opening(ev.CLASSES["coding-hard"], "concurrency", True, rng)
    honest = ev.opening(ev.CLASSES["coding-hard"], "concurrency", False, rng)
    from meritorder.classifier import classify

    tools = [{"name": n} for n in ev.CLAUDE_CODE_TOOLS]
    # Sold as small: one sign of hard work (a symptom) instead of two, so a level lower than the honest wording.
    assert classify({"messages": [{"role": "user", "content": quick}], "tools": tools}, {}, 20_000).level == 4
    assert classify({"messages": [{"role": "user", "content": honest}], "tools": tools}, {}, 20_000).level == 5
    easy = ev.opening(ev.CLASSES["coding-easy"], "tests", True, random.Random(2))
    assert classify({"messages": [{"role": "user", "content": easy}], "tools": tools}, {}, 20_000).level == 4  # not 2: "concurrency"


def test_the_hard_class_reproduces_epochs_frontierswe_curve_and_the_mix_swe_bench_verified():
    cal = ev.check_calibration({"weak": 150.0, "haiku-like": 163.0, "opus-like": 167.4, "best": 171.1}, n=6000)
    for c in cal.values():
        assert abs(c["hard"] - c["frontier_swe"]) < 0.03
        assert abs(c["swe_mix"] - c["swe_verified"]) < 0.07
    assert cal["best"]["hard"] > cal["opus-like"]["hard"] > cal["haiku-like"]["hard"] > cal["weak"]["hard"]


def test_chances_follow_ability_and_luck_is_shared():
    assert ev.chance(170, 160) > ev.chance(160, 160) == 0.5 > ev.chance(150, 160)
    assert ev.sigmoid(-800) == 0.0 and ev.sigmoid(800) == 1.0
    t = ev.population(1, seed=9)[0]
    assert t.rng("attempt", 0).random() == t.rng("attempt", 0).random() != t.rng("attempt", 1).random()


# ------------------------------------------------------------------ execution

def test_every_strategy_runs_through_the_real_router(sim):
    names = [s.name for s in ev.builtin_strategies(sim.world)]
    # Every model_choice the configuration accepts is a strategy: a new one on main is scored without changing the harness.
    assert {"always-opus", "always-sonnet", "always-cheapest", "auto-threshold", "auto-cost-per-success"} <= set(names)
    tasks = ev.population(40, seed=2)
    for s in ev.builtin_strategies(sim.world):
        results = sim.run(s, tasks)
        assert len(results) == 40 and all(r.attempts for r in results)
        if s.name == "always-sonnet":
            assert {m for r in results for a in r.attempts for m in a.models} == {"claude-sonnet"}
        if s.name == "auto-threshold":
            floors = {r.attempts[0].floor for r in results}
            assert {"coding:4", "general:1"} <= floors  # levels the classifier gave
            assert all("classified" in r.attempts[0].reason for r in results)


def test_success_is_the_same_draw_for_every_strategy(sim):
    tasks = ev.population(60, seed=4)
    opus = sim.run(ev.Strategy("o", requested="claude-opus"), tasks)
    again = sim.run(ev.Strategy("o2", requested="claude-opus"), tasks)
    assert [r.finished for r in opus] == [r.finished for r in again]
    assert [round(r.cost, 9) for r in opus] == [round(r.cost, 9) for r in again]
    ambiguous = [r for r, t in zip(opus, tasks) if t.ambiguous]
    assert ambiguous and all(not r.attempts[0].success for r in ambiguous)  # the first prompt didn't say enough


def _session_task(**kw) -> ev.Task:
    t = ev.population(1, seed=1, shares={"coding-medium": 1})[0]
    for k, v in {"client": "claude-code", "tools": list(ev.CLAUDE_CODE_TOOLS), "ambiguous": False, "difficulty": 0.0, **kw}.items():
        setattr(t, k, v)
    return t


def test_claude_caches_for_an_hour_from_the_second_turn_and_survives_pauses(sim):
    sim.world = sim.world_for(ev.Strategy("base"))
    b, route = sim.world.backends["anthropic"], next(r for r in sim.world.cfg.model("claude-sonnet").routes if r.backend == "anthropic")
    cache: dict = {}
    u1, hit1 = sim._usage(cache, b, route, 30_000, 500, 0.0, later=False)
    assert hit1 == 0 and u1.cache_write_tokens == 30_000 and u1.cache_write_1h_tokens == 0  # first turn: 5-minute write
    u2, hit2 = sim._usage(cache, b, route, 32_000, 500, 60.0, later=True)
    assert hit2 == 30_000 and u2.cache_write_tokens == 2_000 and u2.cache_write_1h_tokens == 2_000  # then the hour's
    u3, hit3 = sim._usage(cache, b, route, 33_000, 500, 60.0 + 1_800, later=True)  # half an hour later: still there
    assert hit3 == 32_000
    u4, hit4 = sim._usage(cache, b, route, 34_000, 500, 60.0 + 1_800 + 4_000, later=True)  # over an hour: rebuilt
    assert hit4 == 0 and u4.cache_write_tokens == 34_000


def test_automatic_caching_bills_new_tokens_as_input_and_lasts_five_minutes(sim):
    sim.world = sim.world_for(ev.Strategy("base"))
    b, route = sim.world.backends["openai"], next(r for r in sim.world.cfg.model("gpt-terra").routes if r.backend == "openai")
    cache: dict = {}
    u1, _ = sim._usage(cache, b, route, 30_000, 500, 0.0, later=False)
    assert u1.input_tokens == 30_000 and u1.cache_write_tokens == 0
    u2, hit = sim._usage(cache, b, route, 31_000, 500, 100.0, later=True)
    assert hit == 30_000 and u2.input_tokens == 1_000
    _, hit = sim._usage(cache, b, route, 32_000, 500, 100.0 + 600, later=True)  # a 10-minute pause: gone
    assert hit == 0
    other = next(r for r in sim.world.cfg.model("gpt-sol").routes if r.backend == "openai")
    _, hit = sim._usage(cache, b, other, 32_000, 500, 701.0, later=True)  # another model has its own cache
    assert hit == 0


def test_pauses_cost_rebuilds_where_the_cache_is_short():
    tasks = ev.population(30, seed=8, shares={"coding-medium": 1})
    costs = {}
    for share in (0.0, 0.5):
        s = ev.Simulator(catalog=FROZEN, pause_share=share, seed=1)
        terra = s.run(ev.Strategy("t", requested="gpt-terra"), tasks)
        sonnet = s.run(ev.Strategy("s", requested="claude-sonnet"), tasks)
        costs[share] = (sum(r.cost for r in terra), sum(r.cost for r in sonnet),
                        sum(a.rebuilds for r in terra for a in r.attempts), sum(a.rebuilds for r in sonnet for a in r.attempts))
    assert costs[0.5][2] > costs[0.0][2] * 3  # OpenAI's 5-minute cache: rebuilt after most pauses
    assert costs[0.5][0] / costs[0.0][0] > costs[0.5][1] / costs[0.0][1]  # Claude's hour absorbs them


def test_evidence_matches_the_outcome_and_outcomes_py_reads_it(sim):
    sim.world = sim.world_for(ev.Strategy("base"))
    good = _session_task(difficulty=-100.0, turns=20)
    bad = _session_task(difficulty=400.0, turns=20)
    s = ev.Strategy("sonnet", requested="claude-sonnet")
    ok = sim.run_task(s, good)
    assert ok.finished and len(ok.attempts) == 1 and ok.attempts[0].coding["last_test"] == "pass"
    worse = sim.run_task(s, bad)
    assert not worse.finished and len(worse.attempts) == ev.MAX_ATTEMPTS
    seen = [a.coding for a in worse.attempts]
    assert any(c["last_test"] == "fail" or c["fix_loop"] >= outcomes.LOOP for c in seen) or all(c["last_test"] == "pass" for c in seen)
    corrections = [a.correction for a in worse.attempts[:-1]]
    assert all(c is None or c in {name for name, _ in outcomes.CORRECTIONS} for c in corrections)


def test_a_failing_attempt_shows_a_fix_loop_or_failing_tests():
    sim = ev.Simulator(catalog=FROZEN, seed=1)
    loops = 0
    for i in range(12):
        t = _session_task(difficulty=400.0, turns=24)
        t.seed = f"loop:{i}"
        r = sim.run_task(ev.Strategy("s", requested="claude-sonnet"), t)
        for a in r.attempts:
            loops += a.coding["fix_loop"] >= outcomes.LOOP or a.coding["last_test"] == "fail"
    assert loops >= 10


def test_compaction_starts_a_new_task_with_a_cold_cache():
    sim = ev.Simulator(catalog=FROZEN, seed=1)
    t = _session_task(difficulty=-100.0, turns=60, start_tokens=150_000, growth=(3_000, 3_000))
    r = sim.run_task(ev.Strategy("auto"), t, record=True)
    assert r.finished and r.attempts[0].rebuilds >= 1
    assert any(row["usage"]["cache_write_tokens"] >= ev.COMPACT_TO or row["usage"]["input_tokens"] >= ev.COMPACT_TO
               for row in r.ledger[1:])


def test_the_example_cascade_escalates_on_a_fix_loop_and_pays_the_rebuild():
    sim = ev.Simulator(catalog=FROZEN, seed=1)
    s = ev.EscalateOnEvidence("esc", team={"model_choice": "threshold"})
    moved = []
    for i in range(10):
        t = _session_task(difficulty=166.0, turns=30)
        t.seed = f"esc:{i}"
        r = sim.run(s, [t])[0]
        moved += [a for a in r.attempts if a.escalations]
    assert moved and all(e[0] != e[1] for a in moved for e in a.escalations)
    assert all(a.rebuilds >= 1 for a in moved)  # the new model starts cold


def test_the_oracle_is_a_lower_bound(sim):
    tasks = ev.population(30, seed=11)
    oracle, firsts = sim.oracle(tasks)
    sonnet = sim.run(ev.Strategy("s", requested="claude-sonnet"), tasks)
    total = lambda rs: sum(r.cost + sim.human_cost * r.failures for r in rs)  # noqa: E731
    assert total(oracle) <= total(sonnet) + 1e-9
    assert set(firsts) == {t.id for t in tasks} and all(len(v) > 5 for v in firsts.values())
    assert ev.cheapest_model(sim.world) in {m for v in firsts.values() for m in v}


# ------------------------------------------------------------------ the scorecard and reports

def test_the_scorecard_has_every_measure(sim, tmp_path):
    card = ev.evaluate(ev.Settings(tasks=40, seed=2, sensitivity=True, sensitivity_tasks=20), catalog=FROZEN)
    o = card["strategies"]["auto-threshold"]["overall"]
    for k in ("cost", "success_rate", "cost_per_finished", "human_cost", "total_with_human", "escalation_rate", "escalations_helped",
              "too_high", "too_low", "savings_vs_sonnet", "savings_vs_opus", "savings_vs_sonnet_with_human", "estimate_ratio"):
        assert k in o
    assert card["strategies"]["always-sonnet"]["overall"]["savings_vs_sonnet"] == 0
    assert card["strategies"]["oracle"]["overall"]["too_high"] == 0 and card["strategies"]["oracle"]["overall"]["too_low"] == 0
    assert set(card["sensitivity"]) == {"mix", "pause_share", "quality_error"}
    assert set(card["human_sensitivity"]["always-opus"]) == {"0.0", "5.0", "15.0", "50.0"}
    lo, hi = card["intervals"]["auto-threshold"]["cost_per_finished"]
    assert lo <= o["cost_per_finished"] <= hi and card["intervals"]["always-sonnet"]["savings_vs_sonnet"] == [0, 0]
    assert ev.intervals({}, 1.0) == {}
    md = ev.markdown(card)
    assert "| Always Claude Sonnet |" in md and "## Calibration" in md and "## Sensitivity: mix" in md and "## How sure" in md
    page = ev.html(card)
    assert page.startswith("<!doctype html>") and "<table>" in page and "prefers-color-scheme:dark" in page
    paths = ev._write(tmp_path, card, True)
    assert [p.name for p in paths] == ["scorecard.md", "scorecard.json", "scorecard.html"]
    assert json.loads(paths[1].read_text())["tasks"] == 40


def test_escalation_accounting():
    a = ev.Attempt(False, ["weak"], ["x"], "coding:3", 1.0, 1.0, 5, 0, [], {}, {})
    b = ev.Attempt(True, ["strong"], ["x"], "coding:4", 2.0, 2.0, 5, 0, [], {}, {})
    c = ev.Attempt(True, ["weak", "strong"], ["x"], "coding:3", 2.0, 1.0, 5, 1, [["weak", "strong"]], {}, {})
    ability = {"weak": 150.0, "strong": 170.0}.get
    s = ev.summarize([ev.Result("1", "coding-hard", "x", [a, b], True), ev.Result("2", "coding-hard", "x", [c], True)], 10.0,
                     {"1": {"weak": (False, 1.0), "strong": (True, 2.0)}, "2": {"weak": (False, 1.0), "strong": (True, 0.5)}},
                     lambda m, t: ability(m))
    assert s["escalations"] == 2 and s["escalations_helped"] == 1.0 and s["escalation_rate"] == 1.0
    assert s["too_low"] == 0.5 and s["too_high"] == 0.5  # task 1 failed first though strong would have; task 2 cost more than needed
    assert s["human_cost"] == 10.0 and s["total_per_finished_with_human"] == pytest.approx((5.0 + 10.0) / 2)
    assert ev.summarize([], 1.0)["success_rate"] is None


def test_formats():
    assert [ev.fmt(v, k) for v, k in ((None, "pct"), (0.1234, "pct"), (12.0, "usd0"), (0.0042, "usd"), (0.5, "usd"), (12.5, "usd"))] == \
        ["–", "12.3%", "$12", "$0.0042", "$0.5000", "$12.50"]


# ------------------------------------------------------------------ plugins

PLUGGED = ev.Strategy("plugged", requested="claude-haiku")


def plugged_list():
    return [PLUGGED]


def test_strategies_plug_in_from_a_module():
    assert ev.load_plugin(f"{__name__}:PLUGGED") == [PLUGGED]
    assert ev.load_plugin(f"{__name__}:plugged_list") == [PLUGGED]
    with pytest.raises(TypeError):
        ev.load_plugin("json:dumps")
    assert ev.entry_point_strategies() == []
    card = ev.evaluate(ev.Settings(tasks=10, oracle=False, sensitivity=False, strategies=["plugged", "always-sonnet"]), [PLUGGED],
                       catalog=FROZEN)
    assert set(card["strategies"]) == {"plugged", "always-sonnet"}


def test_a_strategy_changes_the_configuration_it_runs_on(sim):
    s = ev.Strategy("lvl", routing={"level_map": {"coding:4": 3}}, team={"model_choice": "threshold"})
    w = sim.world_for(s)
    assert w.cfg.routing.level_map == {"coding:4": 3} and w is sim.world_for(s) and w is not sim.world_for(ev.Strategy("base"))
    t = _session_task(difficulty=-100.0)
    r = sim.run(s, [t])[0]
    assert r.attempts[0].floor == "coding:3"


# ------------------------------------------------------------------ replay

@pytest.fixture(scope="module")
def exported():
    sim = ev.Simulator(catalog=FROZEN, seed=5)
    tasks = ev.population(500, seed=5, shares=ev.MIXES["coding-only"])
    results = sim.run(ev.explore_strategy(0.5), tasks, record=True)
    starts, ledger = ev.export_rows(results, tasks, sim.world)
    return sim, tasks, starts, ledger


def test_exported_rows_look_like_the_gateways(exported):
    _, _, starts, ledger = exported
    arms = {r["arm"] for r in starts}
    assert {"served", "down", "up"} <= arms
    explored = [r for r in starts if r["arm"]]
    assert all(0 < r["propensity"] < 1 and r["arm_from"] for r in explored)
    assert all(r["level"] == r["arm_from"] + {"served": 0, "down": -1, "up": 1}[r["arm"]] for r in explored)
    assert {"session_id", "ts", "actual_cost", "served_model", "tags"} <= set(ledger[0])


def test_replay_estimates_a_policy_it_never_ran(exported):
    """Replay on explored rows says what one level down would do; the simulator, running it for real, agrees."""
    sim, tasks, starts, ledger = exported
    rows = [dict(r, cost=None) for r in starts]
    ev.join_costs(rows, ledger)
    assert sum(r["cost"] for r in rows) == pytest.approx(sum(x["actual_cost"] for x in ledger), rel=1e-6)
    truth = [dict(r, success=r["truth"]["success"]) for r in rows]
    out = ev.replay(truth, dict(ev.policy(p) for p in ("served", "one-down", "one-up", '{"coding:4": 3}')))
    served, down = out["policies"]["served"], out["policies"]["one-down"]
    assert served["n"] > down["n"] > 20 and down["n_eff"] > 10
    # Ground truth: everyone routed one level down (a level map), run on the same tasks.
    lower = ev.Strategy("down", routing={"level_map": {f"coding:{k}": k - 1 for k in range(2, 6)}}, team={"model_choice": "threshold"})
    real = sim.run(lower, tasks)
    segments = [(a.success, seg["cost"]) for r in real for a in r.attempts for seg in a.segments]  # task starts, as rows count them
    rate = sum(ok for ok, _ in segments) / len(segments)
    cost = sum(c for _, c in segments) / len(segments)
    assert abs(down["success_rate"] - rate) < 4 * down["se_success"] + 0.03
    assert 0.6 < down["cost_per_task"] / cost < 1.6
    assert out["policies"]['level_map {"coding:4": 3}']["n"] > 0
    # From the outcome signals instead of the truth: fewer failures are seen than happen, but the order is kept.
    seen = ev.replay(rows, {"served": ev.POLICIES["served"]})["policies"]["served"]
    assert seen["success_rate"] >= served["success_rate"] - 0.05


def test_replay_reads_every_export_format(exported, tmp_path):
    _, _, starts, ledger = exported
    jsonl = tmp_path / "starts.jsonl"
    jsonl.write_text("".join(json.dumps(r) + "\n" for r in starts[:50]))
    assert ev.read_rows(jsonl)[0]["arm"] == starts[0]["arm"]
    import csv as _csv

    path = tmp_path / "ledger.csv"
    with path.open("w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(ledger[0]))
        w.writeheader()
        for r in ledger[:30]:
            w.writerow({**r, "tags": json.dumps(r["tags"])})
    back = ev.read_rows(path)
    assert isinstance(back[0]["tags"], dict) and float(back[0]["actual_cost"]) == pytest.approx(ledger[0]["actual_cost"])
    pq = pytest.importorskip("pyarrow.parquet")
    import pyarrow as pa

    table = pa.Table.from_pylist([{"session_id": r["session_id"], "ts": r["ts"], "actual_cost": r["actual_cost"],
                                   "tags": list(r["tags"].items())} for r in ledger[:10]],
                                 schema=pa.schema([("session_id", pa.string()), ("ts", pa.string()), ("actual_cost", pa.float64()),
                                                   ("tags", pa.map_(pa.string(), pa.string()))]))
    pq.write_table(table, str(tmp_path / "l.parquet"))
    assert ev.read_rows(tmp_path / "l.parquet")[0]["session_id"] == ledger[0]["session_id"]


def test_replay_skips_rows_it_cant_weight():
    rows = [{"task": "coding", "arm": None, "arm_from": 4, "propensity": None, "level": 4, "cost": 1.0, "success": 1},
            {"task": "coding", "arm": "canary", "arm_from": 4, "propensity": 0.2, "level": 4, "cost": 1.0, "success": 1},
            {"task": "coding", "arm": "served", "arm_from": 4, "propensity": 0.9, "level": 4, "cost": 1.0, "status": 500},
            {"task": "coding", "arm": "served", "arm_from": 4, "propensity": 0.9, "level": 4, "cost": 2.0, "coding": {"last_test": "fail"}}]
    out = ev.replay(rows, {"served": ev.POLICIES["served"], "one-up": ev.POLICIES["one-up"]})
    assert out["usable"] == 1 and out["policies"]["one-up"]["n"] == 0
    s = out["policies"]["served"]
    assert s["success_rate"] == pytest.approx(0.5) and s["cost_per_finished"] == pytest.approx(4.0)  # tests failing: weight 0.5
    assert ev._ts(5) == 5.0 and ev._ts("2026-11-16T09:00:00Z") > 0


# ------------------------------------------------------------------ the command line

def test_the_command_line(tmp_path, capsys, monkeypatch):
    out = tmp_path / "card"
    assert ev.main(["--tasks", "12", "--no-oracle", "--no-sensitivity", "--strategy", "always-sonnet", "--out", str(out), "--html",
                    "--prices", str(ev.FROZEN_PRICES)]) == 0
    assert (out / "scorecard.md").exists() and (out / "scorecard.html").exists()
    base = tmp_path / "baseline.json"
    assert ev.main(["check", "--baseline", str(base), "--update"]) == 0
    assert ev.main(["check", "--baseline", str(base)]) == 0
    worse = json.loads(base.read_text())
    worse["success_rate"] = 1.2  # a baseline nothing can meet
    base.write_text(json.dumps(worse))
    assert ev.main(["check", "--baseline", str(base)]) == 1
    assert "worse than the baseline" in capsys.readouterr().out
    monkeypatch.setattr(ev, "default_catalog", lambda: FROZEN)
    rows = tmp_path / "rows"
    assert ev.main(["export", "--tasks", "30", "--out", str(rows)]) == 0
    capsys.readouterr()
    assert ev.main(["replay", str(rows / "task_starts.jsonl"), "--ledger", str(rows / "ledger.jsonl"), "--policy", "one-down"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert set(printed["policies"]) == {"one-down"} and printed["rows"] > 0
    assert not math.isnan(printed["human_cost"])
