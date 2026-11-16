"""Routing by expected cost per success (routing.model_choice: cost_per_success): each model's chance of solving work at
a level from its quality score, a cheaper and slightly worse model winning when it costs less per solved task, the
minimum chance that keeps a cheap model that usually fails out, the threshold wherever chances are missing, and the
cache, stickiness, priorities and team settings it combines with."""

import time

import pytest
from conftest import ADMIN_KEY, auth, last_row
from pydantic import ValidationError

from meritorder.backends.base import Backend
from meritorder.config import QualityConfig, RoutingConfig, TeamConfig, apply_chances, apply_quality, build_config
from meritorder.router import Router, _usd
from meritorder.sessions import Session

TOP = 0.7  # what a 5 solves on the reference work, as Epoch puts Claude Fable on FrontierSWE
BACKENDS = {**{n: {"format": "openai", "base_url": f"https://{n}.example/v1"} for n in ("a", "b", "c", "d", "e")},
            "mock": {"kind": "mock", "format": "openai"}}
SCORES = {"strong": 5.0, "mid": 3.8, "weak": 2.5}


def price(k: float) -> dict:
    """k times the strong model's price."""
    return {"input": 5 * k, "output": 25 * k}


def catalog(scores: dict[str, dict[str, float]], source="epoch", scale="best") -> dict:
    """The live quality section, as the worker stores it: the scores and the chances they were worked out from."""
    measures = {m: {t: {"chance": round(s / 5 * TOP, 4)} for t, s in per.items()} for m, per in scores.items()}
    return {"quality": {"source": source, "scale": scale, "scores": scores, "measures": measures}}


def build(prices: dict[str, float] | None = None, live=True, routing=None, teams=None, quality=None, extra=None, scores=None):
    """strong (5.0), mid (3.8) and weak (2.5) for coding, each on its own backend, priced relative to strong."""
    scores = {**SCORES, **(scores or {})}
    prices = {"strong": 1.0, "mid": 0.35, "weak": 0.02, "four": 0.6, "plain": 0.5, **(prices or {})}
    backend = {"strong": "a", "mid": "b", "weak": "c", "four": "d", "plain": "e"}
    models = {name: {"quality": {"coding": scores[name], "general": scores[name]},
                     "routes": [{"backend": backend[name], "model": name[0], "price": price(prices[name])}]} for name in scores}
    models.update(extra or {})
    raw = {"backends": BACKENDS, "models": models, "teams": teams or {"default": {}},
           "routing": {"model_choice": "cost_per_success", **(routing or {})}, "quality": quality or {"anchor": {"coding": TOP, "general": TOP}}}
    live_scores = {n: {"coding": s, "general": s} for n, s in scores.items()}
    cfg = build_config(raw, prices_data=catalog(live_scores) if live else {}, quiet=True)
    return Router(cfg, {b.name: Backend(b) for b in cfg.backends}), cfg


def agent(n_tools=5):
    """An agent's turn at hard work: coding:4."""
    return {"messages": [{"role": "user", "content": "find the memory leak in app.py"}], "tools": [{"name": f"t{i}"} for i in range(n_tools)]}


def decide(router, cfg, body=None, team="default", headers=None, session=None, tokens=5_000, now=None):
    return router.decide("auto", cfg.team(team), body or agent(), headers or {}, session, tokens, True, {}, now or time.time())


def models(d):
    return [c.model.name for c in d.candidates]


# ------------------------------------------------------------------ chances from scores

def test_a_score_becomes_a_chance_at_each_level():
    _, cfg = build(scores={"four": 4.0}, extra={"tuned": {"quality_override": {"coding": 4.0}, "routes": [{"backend": "e", "model": "t"}]}})
    strong, mid, four = (cfg.model(n).chances["coding"] for n in ("strong", "mid", "four"))
    assert cfg.threshold_chance == {"coding": TOP, "general": TOP}
    assert strong[5] == TOP and mid[5] == pytest.approx(3.8 / 5 * TOP, abs=1e-4)  # level 5 is the reference work itself
    assert four[4] == pytest.approx(TOP, abs=1e-4)  # exactly at a level: what the best does on the reference
    assert strong[4] == pytest.approx(0.8105, abs=1e-3) and mid[4] == pytest.approx(0.6758, abs=1e-3)
    assert [strong[lvl] for lvl in range(1, 6)] == sorted((strong[lvl] for lvl in range(1, 6)), reverse=True)  # easier work, more often
    # An override is a score on the same scale, so it gets the chance that score stands for; the source doesn't cover it.
    assert cfg.model("tuned").chances["coding"] == four


def test_where_chances_come_from_and_where_they_dont():
    # A mock model keeps its configured scores, which aren't on the source's scale: no chances.
    _, cfg = build(extra={"toy": {"quality": {"coding": 4}, "routes": [{"backend": "mock", "model": "toy"}]}})
    assert cfg.model("toy").chances == {} and cfg.model("strong").chances

    # Nothing fetched yet, or a source that isn't a chance (the Artificial Analysis index): none at all.
    _, never = build(live=False)
    assert never.threshold_chance == {} and never.model("strong").chances == {}
    apply_quality(never, {"quality": {"source": "epoch", "scale": "best", "scores": {"strong": {"coding": 1.0}},
                                      "measures": {"strong": {"coding": {"chance": 0.0}}}}})
    assert never.threshold_chance == {}  # nothing solves the reference work: no scale to read chances from
    _, aa = build(quality={"source": "artificial_analysis"})
    apply_chances(aa, catalog({"strong": {"coding": 5.0}}, source="artificial_analysis"))
    assert aa.threshold_chance == {} and aa.model("strong").chances == {}

    # Scores written in the config (applied from eval runs): chances once the best measured pass rate is known.
    _, measured = build(live=False, quality={"source": "config", "measured_best_rate": {"coding": 0.9}})
    assert measured.threshold_chance == {"coding": 0.9} and measured.model("strong").chances["coding"][5] == 0.9
    assert "general" not in measured.model("strong").chances  # no measured rate for general: the threshold decides there

    # On the worst-to-best scale a score of 1 is the weakest model's chance, and it's still at the level for level 1.
    _, spread = build(quality={"scale": "range"})
    apply_quality(spread, catalog({"strong": {"coding": 5.0}, "weak": {"coding": 1.0}}, scale="range"))
    assert spread.model("weak").chances["coding"][1] == pytest.approx(TOP, abs=1e-4)


def test_settings_are_validated():
    for bad in (0, 1, 1.5):
        with pytest.raises(ValidationError):
            RoutingConfig(min_success_chance=bad)
        with pytest.raises(ValidationError):
            TeamConfig(name="t", min_success_chance=bad)
    with pytest.raises(ValidationError):
        RoutingConfig(model_choice="cheapest")
    with pytest.raises(ValidationError):
        TeamConfig(name="t", model_choice="cheapest")
    with pytest.raises(ValidationError):
        QualityConfig(measured_best_rate={"coding": 1.5})
    assert RoutingConfig().model_choice == "expected_cost" and RoutingConfig().min_success_chance == 0.6  # the default since decision 62


def test_amounts_read_well():
    assert [_usd(x) for x in (0, 0.0043, 0.0697, 0.12, 1.5, 1234.5)] == ["$0", "$0.0043", "$0.070", "$0.12", "$1.50", "$1,234.50"]


# ------------------------------------------------------------------ choosing

def test_a_cheaper_slightly_worse_model_wins_when_it_costs_less_per_success():
    router, cfg = build()
    cfg.routing.model_choice = "threshold"
    d = decide(router, cfg)
    assert d.floor.level == 4 and models(d) == ["strong"]  # the threshold: only strong scores 4 or more

    cfg.routing.model_choice = "cost_per_success"
    d = decide(router, cfg)
    assert models(d) == ["mid", "strong"]  # weak is under the 60% minimum
    mid, strong = d.candidates
    assert mid.chance == pytest.approx(0.6758, abs=1e-3) and strong.chance == pytest.approx(0.8105, abs=1e-3)
    # A failed try on mid is paid for, then the work is done on strong, the threshold's pick, until it succeeds.
    assert strong.expected == pytest.approx(strong.cost / strong.chance)
    assert mid.expected == pytest.approx(mid.cost + (1 - mid.chance) * strong.cost / strong.chance)
    assert mid.expected < strong.expected and mid.cost < strong.cost
    assert ("cost per success: mid, expected $0.072 per success (68% at coding:4) vs $0.12 (81%) on strong, "
            "the cheapest at the level") in d.reason

    # Not cheap enough to make up for failing more often: the better model wins.
    router, cfg = build(prices={"mid": 0.9})
    d = decide(router, cfg)
    assert models(d) == ["strong", "mid"]
    assert "cost per success: strong, expected $0.12 per success (81% at coding:4) vs $0.12 (68%) on mid" in d.reason


def test_the_minimum_chance_keeps_a_cheap_model_that_usually_fails_out():
    router, cfg = build(teams={"default": {}, "bold": {"min_success_chance": 0.45}})
    weak = cfg.model("weak").chances["coding"][4]
    assert weak == pytest.approx(0.497, abs=1e-3)
    d = decide(router, cfg)
    assert "weak" not in models(d)  # the cheapest per success if allowed, but it fails half the time
    bold = decide(router, cfg, team="bold")  # a team that accepts lower odds
    assert models(bold)[0] == "weak" and "weak, expected" in bold.reason and "(50% at coding:4)" in bold.reason

    # No model reaches the minimum: the threshold decides.
    router, cfg = build(routing={"min_success_chance": 0.95})
    d = decide(router, cfg)
    assert models(d) == ["strong"] and "no model has at least a 95% chance at coding:4, so the threshold decides" in d.reason


def test_the_minimum_holds_for_models_at_the_level_too_and_the_threshold_steps_in_when_nothing_can_run():
    # four meets level 4 exactly (a 70% chance there); with a 75% minimum only strong (81%) is in reach.
    router, cfg = build(scores={"four": 4.0}, prices={"four": 0.6}, routing={"min_success_chance": 0.75})
    assert models(decide(router, cfg)) == ["strong"]
    del router.backends["a"]  # strong has nowhere to run
    d = decide(router, cfg)
    assert models(d) == ["four"] and "nothing in reach can run now, so the threshold decides" in d.reason


def test_without_chances_the_threshold_decides():
    router, cfg = build(live=False)  # nothing fetched yet: scores from the config, no chances
    d = decide(router, cfg)
    assert models(d) == ["strong"] and "cost per success: no chances of success for coding, so the threshold decides" in d.reason

    # One model without an estimate: below the level it's never picked, however cheap; at the level it counts as
    # just meeting it.
    router, cfg = build(scores={"plain": 3.0}, prices={"plain": 0.001})
    cfg.model("plain").chances = {}
    assert "plain" not in models(decide(router, cfg))
    cfg.model("plain").quality["coding"] = 4.5
    d = decide(router, cfg)
    assert models(d)[0] == "plain" and "plain, expected" in d.reason and "(70% at coding:4, no estimate)" in d.reason

    # A level the client set is a minimum it asked for.
    router, cfg = build()
    d = decide(router, cfg, headers={"x-route-floor": "coding:4"})
    assert models(d) == ["strong"] and "the client set the level (X-Route-Floor)" in d.reason


def test_nothing_at_the_level_the_most_likely_model_is_the_redo():
    router, cfg = build(teams={"default": {}, "no-strong": {"allowed_models": ["*", "!strong"]}})
    d = decide(router, cfg, team="no-strong")
    assert models(d) == ["mid"]
    mid = d.candidates[0]
    assert mid.expected == pytest.approx(mid.cost / mid.chance)  # it redoes its own failures
    assert d.reason.endswith("cost per success: mid, expected $0.050 per success (68% at coding:4)")


def test_a_warm_cache_still_matters():
    """The same new task: a session whose cache is on strong stays there (moving would rebuild 148,000 tokens of
    cache on mid first); once that cache has expired, mid is cheaper per success."""
    router, cfg = build()
    now = time.time()
    body = agent()  # coding:4: a long context doesn't change the level
    warm =Session(id="s", model="strong", backend="a", turns=3, prefix_tokens=148_000, warm_until={"a": now + 200})
    d = decide(router, cfg, body=body, session=warm, tokens=150_000, now=now)
    assert d.floor.level == 4 and models(d)[0] == "strong"
    assert d.candidates[0].warm and d.candidates[0].expected < d.candidates[1].expected

    cold = Session(id="s", model="strong", backend="a", turns=3, prefix_tokens=148_000)
    d = decide(router, cfg, body=body, session=cold, tokens=150_000, now=now)
    assert models(d)[0] == "mid" and "staying on" not in d.reason

    # Stickiness compares expected costs too: a move that saves less than 10% + $0.01 per success doesn't happen.
    router, cfg = build(prices={"mid": 0.7})
    assert models(decide(router, cfg, now=now))[0] == "mid"  # about 11% cheaper per success
    d = decide(router, cfg, session=Session(id="s", model="strong", backend="a", turns=3), now=now)
    assert models(d)[0] == "strong" and "staying on a" in d.reason


def test_priorities_compare_expected_costs():
    router, cfg = build(prices={"mid": 0.75}, routing={"speed_check_every": 0})
    now = time.time()
    for backend, model, seconds in (("a", "s", 2.0), ("b", "m", 9.0)):
        for _ in range(2):
            router.speeds.record(backend, model, seconds, seconds / 5, 600, now)
    d = decide(router, cfg, now=now)
    assert models(d) == ["mid", "strong"]  # 7% cheaper per success: more than cost's 2%
    d = decide(router, cfg, headers={"x-meritorder-priority": "balanced"}, now=now)
    assert models(d) == ["strong", "mid"]  # within balanced's 20%, and faster


async def test_the_console_shows_each_models_chances(client, gateway):
    gateway.cfg.model("big").chances = {"coding": {1: 0.97, 2: 0.93, 3: 0.88, 4: 0.81, 5: 0.7}}
    r = await client.get("/v1/admin/config", headers=auth(ADMIN_KEY))
    shown = {m["name"]: m for m in r.json()["models"]}
    assert shown["big"]["chances"] == {"coding": {"1": 0.97, "2": 0.93, "3": 0.88, "4": 0.81, "5": 0.7}}
    assert shown["small"]["chances"] == {}  # mocks keep their configured scores: no chances


def _chances_on_the_test_gateway(gateway):
    """The test gateway's models are mocks, which get no chances from a source: give them some by hand."""
    gateway.cfg.threshold_chance = {"coding": TOP, "general": TOP}
    gateway.cfg.model("small").chances = {"coding": {3: TOP, 4: 0.65, 5: 0.45}}
    gateway.cfg.model("big").chances = {"coding": {3: 0.88, 4: 0.81, 5: TOP}}
    gateway.cfg.routing.model_choice = "cost_per_success"


async def test_the_ledger_keeps_the_cost_per_success_reason(client, gateway):
    _chances_on_the_test_gateway(gateway)
    tools = [{"type": "function", "function": {"name": f"t{i}", "parameters": {"type": "object"}}} for i in range(5)]
    r = await client.post("/v1/chat/completions", headers=auth(),
                          json={"model": "auto", "tools": tools, "messages": [{"role": "user", "content": "find the memory leak"}]})
    assert r.status_code == 200, r.text
    assert r.headers["x-meritorder-model"] == "small" and r.headers["x-meritorder-floor"] == "coding:4"  # below the level
    reason = (await last_row(gateway))["route_reason"]
    assert reason.startswith("classified coding:4 (coding: 5 tools; hard wording (memory leak): +1)")
    assert "cost per success: small, expected $" in reason and "(65% at coding:4) vs $" in reason and "(81%) on big" in reason


async def test_the_levels_page_lists_models_in_reach(client, gateway):
    _chances_on_the_test_gateway(gateway)
    data = (await client.get("/v1/reports/levels", headers=auth())).json()
    assert data["model_choice"] == "cost_per_success" and data["min_success_chance"] == 0.6
    by_level = {lv["level"]: lv for lv in data["levels"]}
    assert by_level[4]["models"]["coding"] == [{"name": "big", "score": 5, "price": 18}]  # the threshold's, as before
    assert by_level[4]["in_reach"]["coding"] == [{"name": "small", "score": 3, "price": 1.2, "chance": 0.65}]
    assert by_level[5]["in_reach"]["coding"] == []  # 45%: under the minimum
    assert by_level[4]["in_reach"]["general"] == []  # no chances for general on these models

    gateway.cfg.routing.model_choice = "threshold"
    data = (await client.get("/v1/reports/levels", headers=auth())).json()
    assert data["model_choice"] == "threshold" and all(not lv["in_reach"]["coding"] for lv in data["levels"])
