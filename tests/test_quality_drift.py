"""The relative-scale drift (decision 58): on the best-anchored scale a new, stronger model lowers every other model's
score though none got worse, so the same work moves to dearer models with each release. The absolute scale fixes what a
5 stands for (``quality.anchor``), so unchanged work keeps going where it went."""

import time

import pytest

from meritorder import quality_data as qd
from meritorder.backends.base import Backend
from meritorder.config import build_config
from meritorder.router import Router

BACKENDS = {n: {"format": "openai", "base_url": f"https://{n}.example/v1"} for n in ("a", "b", "c", "d")}
# Each model's published chance of solving a FrontierSWE-hard task, and its price relative to the strongest of today.
TODAY = {"strong": (0.70, 1.0), "mid": (0.50, 0.35), "cheap": (0.45, 0.1)}
FRONTIER = {"frontier": (0.95, 3.0)}  # released tomorrow: better than anything, and dearer


def router(models: dict[str, tuple[float, float]], scale: str, model_choice: str = "threshold"):
    backend = dict(zip(models, "abcd"))
    measures = {n: {"coding": {"chance": c}, "general": {"chance": c}} for n, (c, _) in models.items()}
    q = {"scale": scale, "anchor": {"coding": 0.70, "general": 0.70}}
    catalog = {"quality": {"source": "epoch", "scale": scale, "anchor": q["anchor"] if scale == "absolute" else None,
                           "scores": qd.scores(measures, scale, q["anchor"]), "measures": measures}}
    raw = {"backends": BACKENDS, "quality": q, "routing": {"model_choice": model_choice},
           "models": {n: {"quality": {"coding": 1, "general": 1},
                          "routes": [{"backend": backend[n], "model": n, "price": {"input": 5 * k, "output": 25 * k}}]}
                      for n, (_, k) in models.items()}}
    cfg = build_config(raw, prices_data=catalog, quiet=True)
    return Router(cfg, {b.name: Backend(b) for b in cfg.backends}), cfg


def pick(scale: str, models: dict, model_choice: str = "threshold") -> tuple[str, dict[str, float]]:
    r, cfg = router(models, scale, model_choice)
    body = {"messages": [{"role": "user", "content": "add a --verbose flag"}], "tools": [{"name": "edit"}]}  # coding:3
    d = r.decide("auto", cfg.team("default"), body, {}, None, 5_000, True, {}, time.time())
    assert d.floor is not None and str(d.floor) == "coding:3"
    return d.candidates[0].model.name, {m.name: m.q("coding") for m in cfg.models}


def test_on_the_best_scale_a_stronger_model_moves_unchanged_work_to_dearer_models():
    before, scores = pick("best", TODAY)
    assert before == "cheap" and scores["cheap"] == pytest.approx(3.2)  # 5 x 0.45 / 0.70
    after, scores = pick("best", {**TODAY, **FRONTIER})
    # Nobody got worse, but every score fell: 5 x 0.45 / 0.95 = 2.4, 5 x 0.50 / 0.95 = 2.6, 5 x 0.70 / 0.95 = 3.7.
    assert scores["cheap"] == pytest.approx(2.4) and scores["mid"] == pytest.approx(2.6) and scores["strong"] == pytest.approx(3.7)
    assert after == "strong"  # the same coding:3 request now costs ten times as much


def test_on_the_absolute_scale_the_same_work_goes_where_it_went():
    before, scores = pick("absolute", TODAY)
    after, later = pick("absolute", {**TODAY, **FRONTIER})
    assert before == after == "cheap"
    assert {n: later[n] for n in TODAY} == scores  # a new model changes nobody else's score
    assert later["frontier"] == 5.0  # above the anchor: 5, so it meets every level


def test_cost_per_success_chances_dont_move_either():
    for scale, moved in (("best", True), ("absolute", False)):
        r1, c1 = router(TODAY, scale, "cost_per_success")
        r2, c2 = router({**TODAY, **FRONTIER}, scale, "cost_per_success")
        assert (c1.model("cheap").chances["coding"] != c2.model("cheap").chances["coding"]) is moved
        assert (c1.threshold_chance != c2.threshold_chance) is moved
    assert c2.threshold_chance == {"coding": 0.7, "general": 0.7}  # a model exactly at a level: the anchor, on its level's work


def test_today_the_absolute_scale_gives_the_best_anchored_scores():
    """The default anchors are the best model's chances when the scale was fixed: switching changed no score then."""
    assert qd.scores({n: {"coding": {"chance": c}} for n, (c, _) in TODAY.items()}, "absolute", {"coding": 0.70}) == \
        qd.scores({n: {"coding": {"chance": c}} for n, (c, _) in TODAY.items()}, "best")


def test_an_anchor_must_be_a_chance():
    with pytest.raises(ValueError, match="quality.anchor"):
        build_config({"backends": BACKENDS, "models": {}, "quality": {"anchor": {"coding": 1.5}}}, prices_data={}, quiet=True)
