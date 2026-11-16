"""The learned router (learned.py, learned_jobs.py, decision 44): features that hold no content, training that recovers a
known rule from synthetic traffic with mocked outcomes, shadow mode that never changes a routing decision, route mode
that uses the learned level only when it's sure, every fallback to the rules, the outcome signals the gateway records,
the shadow report and its API, the worker job and the CLI."""

from __future__ import annotations

import copy
import functools
import json
import math
import random
import time
import uuid
import zlib
from datetime import UTC, datetime, timedelta

import httpx
import pytest
import yaml
from conftest import ADMIN_KEY, LEAD_KEY, TEST_CONFIG, TEST_KEY, auth
from learned_synth import HEADERS, KEY, KINDS, LR, SCORES, probe, rows
from pydantic import ValidationError

from meritorder import cli, learned_jobs
from meritorder.app import build_gateway, create_app
from meritorder.batches.poller import BatchPoller
from meritorder.backends.base import Backend
from meritorder.classifier import Floor, classify
from meritorder.config import LearnedRouterConfig, LevelRules, build_config, load_config
from meritorder.configstore import load_active, token
from meritorder.learned import (KEY_ENV, KEY_SETTING, TEXT_BUCKETS, WINDOW, Features, LearnedRouter, Model, Prediction, features,
                                floats, key_seed, rules_hash, text_buckets)
from meritorder.learned_jobs import LAST_TRAINING, TRAIN_REQUESTED, Obs, RouterTrainer, abandoned, fit, observations, train
from meritorder.router import Router
from meritorder.state import MemoryState
from meritorder.store import MemoryStore


@pytest.fixture(autouse=True)
def no_feature_key_env(monkeypatch):
    monkeypatch.delenv(KEY_ENV, raising=False)


ROUTED = {**SCORES, "cheap": 2.0}  # the router's world: level-2 work has a model of its own


def plain_cfg(learned: dict | None = None, levels: dict | None = None, batch: tuple[str, ...] = ()):
    """Four coding models on their own priced API backends: cheap (2.0), mid (3.0), strong (4.0) and best (5.0). The
    backends named in ``batch`` have a batch API (OpenAI's)."""
    price = {"cheap": 0.1, "mid": 1.0, "strong": 3.0, "best": 10.0}
    backends = {n: {"format": "openai", "base_url": f"https://{n}.example/v1", **({"batch": {"api": "openai"}} if n in batch else {})}
                for n in ROUTED}
    models = {n: {"quality": {"coding": s, "general": s},
                  "routes": [{"backend": n, "model": n, "price": {"input": price[n], "output": 5 * price[n]}}]} for n, s in ROUTED.items()}
    raw = {"backends": backends, "models": models, "quality": {"source": "config"},
           "routing": {"learned": learned or {"mode": "shadow"}, **({"levels": levels} if levels else {})}}
    return build_config(raw, prices_data={}, quiet=True)


@functools.lru_cache(maxsize=1)
def trained() -> dict:
    """A model trained on 6,000 synthetic requests (learned_synth): the result of learned_jobs.train."""
    return train(rows(6000), plain_cfg(), LR.fv, datetime.now(UTC))


def router(cfg=None, model: bool = True, key: str = KEY, version: int = 1) -> Router:
    cfg = cfg or plain_cfg()
    learned = LearnedRouter(Model(trained()["doc"], version) if model else None, key)
    return Router(cfg, {b.name: Backend(b) for b in cfg.backends}, learned=learned)


def decide(r: Router, body: dict, headers: dict | None = None, model: str = "auto", session=None, batch: bool = False):
    return r.decide(model, r.cfg.team("default"), body, headers or {}, session, 50, True, {}, time.time(), client="anthropic",
                    batch=batch)


# ------------------------------------------------------------------ features hold no content


def test_features_are_hashed_buckets_and_named_tags_never_the_text():
    secret = "acquire globex before the earnings call"
    body = {"messages": [{"role": "user", "content": f"draft the plan to {secret}"}], "tools": [{"name": f"t{i}"} for i in range(7)],
            "reasoning_effort": "high", "response_format": {"type": "json_object"}, "thinking": {"type": "enabled"}}
    headers = {"user-agent": "claude-cli/2.1.0 (external, cli)", "x-claude-code-request-class": "main", "x-claude-code-agent-id": "a1"}
    rules = classify(body, headers, 5000)
    f = features(rules, body, headers, 5000, "anthropic", LR.seed)
    assert all(isinstance(b, int) and 0 <= b < TEXT_BUCKETS for b in f.text) and f.text == sorted(set(f.text))
    stored = json.dumps({"text": f.text, "tags": f.tags})
    for word in secret.split():
        assert word not in stored  # only buckets and names leave the request
    assert set(f.tags) >= {"rules:coding:3", "tools:5-9", "size:<8k", "turns:1", "format:anthropic", "client:claude-cli",
                           "class:main", "rule:agent", "rule:tools", "subagent", "effort:high", "structured", "thinking"}
    # Another key hashes into other buckets: without it, the buckets can't be checked against guessed words.
    other = features(rules, body, headers, 5000, "anthropic", key_seed("another key")[0])
    assert other.text != f.text and other.tags == f.tags
    assert LearnedRouter(key="another key").fv != LR.fv


def test_without_the_key_one_word_bucket_says_nothing_about_another():
    """A database reader without the key must not be able to check guessed words. With CRC32 seeded by the key, every
    word of one length moved by the same XOR, whatever the key (CRC is linear): knowing one common word's bucket (or
    trying all 65,536 shifts) gave every other word of that length, so the key kept out of the database didn't help."""
    shifts = {text_buckets(w, LR.seed)[0] ^ (zlib.crc32(w.encode()) & (TEXT_BUCKETS - 1)) for w in ("cat", "dog", "sun", "map", "fix")}
    assert len(shifts) > 1
    assert key_seed(KEY)[1] not in key_seed(KEY)[0].hex()  # the stored fingerprint isn't part of the hashing key


def test_only_the_ends_of_a_long_prompt_are_read():
    head, tail = "alpha " * (WINDOW // 6 + 2), "omega " * (WINDOW // 6 + 2)
    a = text_buckets(head + "middle words " * 2000 + tail, LR.seed)
    b = text_buckets(head + "other filler " * 2000 + tail, LR.seed)
    assert a == b and text_buckets("", LR.seed) == []
    # Words, runs of non-ASCII and runs of symbols are all tokens; case doesn't matter.
    assert text_buckets("Café => ```x```", LR.seed) == text_buckets("CAFÉ => ```X```", LR.seed)


def test_tags_name_the_rules_that_set_the_level():
    def tags(body, headers=None, size=50):
        rules = classify(body, headers or {}, size)
        return features(rules, body, headers or {}, size, "", LR.seed).tags

    assert "rule:short" in tags({"messages": [{"role": "user", "content": "hi"}]})
    assert "rule:analysis" in tags({"messages": [{"role": "user", "content": "compare these two plans in depth " * 30}]}, size=900)
    assert {"rule:code", "rule:hard"} <= set(tags({"messages": [{"role": "user", "content": "fix the race condition in app.py"}]}))
    assert "rule:hint" in tags({"messages": [{"role": "user", "content": "hi"}]}, {"x-route-hint": "smart"})
    assert "client:unknown" in tags({"messages": []}) and "format:openai" in tags({"messages": []})


# ------------------------------------------------------------------ training recovers a known rule


def test_training_recovers_the_known_rule():
    """The synthetic world: the rules give all three coding kinds coding:3, but Kubernetes deployments need 4 and default-value changes
    2 (learned_synth.KINDS). Requests are served by random models and a retry is seen for most failures."""
    result = trained()
    m = result["metrics"]
    assert result["better"] and m["holdout"]["loss"] < m["holdout"]["rules_loss"]
    assert m["requests"] == 6000 and m["with_signal"] > 500 and m["members"] == 6
    model = Model(result["doc"], 1)
    for seed in (99, 7, 13):
        kube = model.predict(probe("kube", seed)[2])
        plain = model.predict(probe("plain", seed)[2])
        rename = model.predict(probe("rename", seed)[2])
        short = model.predict(probe("short", seed)[2])
        assert (kube.level, plain.level, rename.level, short.level) == (4, 3, 2, 1), seed
        assert kube.rules == plain.rules == rename.rules == 3
        assert kube.confidence >= 0.8 and plain.confidence >= 0.8 and short.confidence >= 0.8
        assert kube.difficulty > plain.difficulty + 0.4 > rename.difficulty + 0.8
        assert kube.parts[0][0] == "wording" and kube.why().startswith("wording +")


def test_the_rules_stay_right_on_average_and_the_client_only_changes_how_often_failures_are_noticed():
    obs = observations(rows(2000, seed=3))
    f = fit(obs, {"coding": 0.42, "general": 0.42})
    for task in ("coding", "general"):
        mine = [o for o in obs if o.task == task]
        assert abs(sum(f.z(o)[1] - o.rules for o in mine) / len(mine)) < 1e-6  # anchored on the rules
    d0 = f.z(Obs([1, 2], ["rules:coding:3", "client:pi"], "coding", 3, 3.0, 0))[1]
    d1 = f.z(Obs([1, 2], ["rules:coding:3", "client:claude-cli"], "coding", 3, 3.0, 0))[1]
    assert d0 == d1  # a client tag never makes work harder or easier


def test_a_model_trained_on_nothing_reproduces_the_rules():
    body, rules, feats = probe("kube")
    obs = [Obs(feats.text, feats.tags, "coding", 3, 3.0, 0.0)] * 10
    f = fit(obs, {"coding": 0.42}, epochs=3)
    assert f.z(obs[0])[1] == pytest.approx(3.0)


def test_training_needs_enough_outcomes():
    cfg = plain_cfg()
    few = train(rows(50), cfg, LR.fv)
    assert few["doc"] is None and "training needs at least 200" in few["why"]
    quiet = train(rows(400, detect=0, noise=0), cfg, LR.fv)
    assert quiet["doc"] is None and "no request has an outcome signal yet" in quiet["why"]


def test_a_model_no_better_than_the_rules_is_kept_but_not_used():
    # Outcomes that have nothing to do with the request: nothing to learn.
    data = rows(1200, seed=5)
    rng = random.Random(5)
    for r in data:
        r["retried"] = rng.random() < 0.15
    result = train(data, plain_cfg(), LR.fv)
    assert result["doc"] is not None and not result["better"] and "no better than the rules alone" in result["why"]


def test_labels_from_every_signal_and_what_isnt_a_label():
    base = rows(8, seed=2)
    a, b, c, d, e, f, g, h = base
    a["floor_raised"] = 4
    b["retried"] = True
    c["status"] = 499  # the person stopped the answer
    d["refused_scores"] = [3.0]
    e["status"] = 502  # an error says nothing about the work
    f["served_score"] = None  # an unmeasured model neither
    g["retried"], g["status"] = True, 503  # a retry after an error is the client's own
    obs = {(o.ts, o.score, o.w): o for o in observations([a, b, c, d, e, f, g, h])}
    ys = sorted(round(o.y, 2) for o in obs.values())
    assert len(obs) == 6  # e, f, g dropped; d refused once and answered once
    assert ys.count(1.0) == 2  # a's raised floor, d's refusal (weight 0.2)
    assert 0.6 in ys and 0.4 in ys
    assert [o.w for o in obs.values() if o.w != 1.0] == [learned_jobs.REFUSAL]


def test_abandoned_sessions_are_inferred_from_overlapping_wording():
    t0 = datetime.now(UTC)

    def row(i, user, session, minutes, text, tags=()):
        return {"request_id": uuid.UUID(int=i), "ts": t0 + timedelta(minutes=minutes), "user_id": user, "session_id": session,
                "thread": session, "text_features": text, "tags": list(tags), "status": 200}

    same = list(range(40))
    data = [row(1, "ann", "s1", 0, same), row(2, "ann", "s2", 5, same[:35] + [99]),  # gave up and asked again: abandoned
            row(3, "bob", "s3", 0, same), row(4, "bob", "s4", 30, same),  # asked again much later: not
            row(5, "cy", "s5", 0, same), row(6, "cy", "s6", 2, list(range(100, 140))),  # something else: not
            row(7, "di", "s7", 0, same, ["subagent"]), row(8, "di", "s8", 1, same),  # a subagent's conversation ends by design
            row(9, "ed", "s9", 0, same), row(10, "ed", "s9", 1, same)]  # same session: it went on
    assert abandoned(sorted(data, key=lambda r: r["ts"])) == {str(uuid.UUID(int=1))}


# ------------------------------------------------------------------ the model in the router


def test_inference_is_well_under_a_millisecond_on_a_long_agent_prompt():
    model = Model(trained()["doc"], 1)
    code = "def handler(event, context):\n    return {'status': 200, 'body': json.dumps(event)}  # ok\n" * 300
    body = {"messages": [{"role": "user", "content": "start"}, {"role": "assistant", "content": "ok"}] * 100
            + [{"role": "user", "content": [{"type": "text", "text": code}, {"type": "text", "text": "now refactor it for the race condition"}]}],
            "tools": [{"name": f"tool{i}", "description": "d" * 200} for i in range(40)]}
    rules = classify(body, HEADERS, 300_000)
    times = []
    for _ in range(300):
        t0 = time.perf_counter()
        model.predict(features(rules, body, HEADERS, 300_000, "anthropic", LR.seed))
        times.append(time.perf_counter() - t0)
    times.sort()
    p50, p99 = times[150] * 1000, times[296] * 1000
    print(f"learned router on a long agent prompt: p50 {p50:.3f} ms, p99 {p99:.3f} ms")
    assert p99 < 5  # about 0.4 ms on a laptop; CI machines and coverage tracing are slower


def test_shadow_mode_never_changes_a_routing_decision():
    """The same synthetic traffic through the router without the learned router and with it in shadow mode: every
    decision picks the same candidates in the same order at the same level, and the reason only gains a note at its end."""
    off, shadow = router(plain_cfg({"mode": "off"})), router(plain_cfg({"mode": "shadow"}))
    seen = {"agrees": 0, "shadow": 0}
    for i, kind in enumerate(list(KINDS) * 15):
        body = probe(kind, seed=i)[0]
        headers = {"x-claude-code-request-class": "auxiliary"} if i % 7 == 0 else {}
        a, b = decide(off, body, headers), decide(shadow, body, headers)
        assert [(c.model.name, c.backend.name, c.cost) for c in a.candidates] == [(c.model.name, c.backend.name, c.cost) for c in b.candidates]
        assert a.floor.level == b.floor.level and a.floor.task == b.floor.task and a.learned is None
        assert b.reason.startswith(a.reason) and b.learned is not None and not b.learned.used
        seen[b.learned.outcome] = seen.get(b.learned.outcome, 0) + 1
    assert seen["shadow"] > 10 and seen["agrees"] > 10  # it disagreed often, and still changed nothing
    assert off.speeds.decisions == shadow.speeds.decisions  # pricing the other side counts no speed check


def test_shadow_mode_says_what_it_would_have_picked_and_what_that_costs():
    d = decide(router(), probe("kube")[0])
    assert d.candidates[0].model.name == "mid" and str(d.floor) == "coding:3"
    call = d.learned
    assert call.outcome == "shadow" and call.alt_model == "strong" and call.alt_cost > call.cost > 0
    assert call.fail[0] > call.fail[1]  # the model expects the rules' pick to fail more often
    assert "learned router v1 (shadow): coding:4" in d.reason and "% sure (wording +" in d.reason
    assert "not used in shadow mode; at coding:4 it would have picked strong (~$" in d.reason
    rec = call.record()
    assert rec["mode"] == "shadow" and rec["version"] == 1 and rec["level"] == 4 and rec["alt_model"] == "strong"


def test_route_mode_uses_the_learned_level_when_sure():
    r = router(plain_cfg({"mode": "route"}))
    d = decide(r, probe("kube")[0])
    assert str(d.floor) == "coding:4" and d.candidates[0].model.name == "strong" and d.learned.used
    assert d.reason.startswith("classified coding:4 (learned router v1: ") and "the rules said coding:3 (coding: 1 tool)" in d.reason
    assert "the rules' coding:3 would have picked mid" in d.reason
    lower = decide(r, probe("rename")[0])
    assert str(lower.floor) == "coding:2" and lower.candidates[0].model.name == "cheap"
    same = decide(r, probe("plain")[0])
    assert str(same.floor) == "coding:3" and same.learned.outcome == "agrees" and "learned router v1 agrees (coding:3" in same.reason


def test_the_rules_decide_when_the_model_is_unsure(monkeypatch):
    r = router(plain_cfg({"mode": "route", "min_confidence": 0.9}))
    monkeypatch.setattr(Model, "predict", lambda self, f: Prediction(1, "coding", 2, 3, 0.62, 2.4, 0.3, [("wording", -0.6)]))
    d = decide(r, probe("rename", seed=1)[0])
    assert str(d.floor) == "coding:3" and d.learned.outcome == "unsure" and not d.learned.used
    assert "learned router v1: coding:2, 62% sure (wording -0.6), under the 90% it needs, so the rules decide" in d.reason
    assert d.learned.alt_model == "cheap"  # what the learned level would have picked, for the report


def test_the_rules_decide_without_a_model_or_its_key():
    d = decide(router(plain_cfg({"mode": "route"}), model=False), probe("kube")[0])
    assert str(d.floor) == "coding:3" and d.learned.outcome == "missing"
    assert "learned router: no model trained yet, so the rules decide" in d.reason
    shadow = decide(router(model=False), probe("kube")[0])
    assert "learned router (shadow): no model trained yet" in shadow.reason
    cfg = plain_cfg({"mode": "route"})
    keyless = Router(cfg, {b.name: Backend(b) for b in cfg.backends}, learned=LearnedRouter())
    d = decide(keyless, probe("kube")[0])
    assert d.learned.outcome == "failed" and "its feature key isn't loaded yet" in d.reason
    broken = Router(cfg, {b.name: Backend(b) for b in cfg.backends}, learned=LearnedRouter(None, KEY, "couldn't load model v3 (KeyError)"))
    assert "learned router: couldn't load model v3 (KeyError), so the rules decide" in decide(broken, probe("kube")[0]).reason


def test_the_rules_decide_when_the_model_is_stale():
    doc = copy.deepcopy(trained()["doc"])
    doc["trained_at"] = (datetime.now(UTC) - timedelta(days=45)).isoformat()
    cfg = plain_cfg({"mode": "route"})
    old = Router(cfg, {b.name: Backend(b) for b in cfg.backends}, learned=LearnedRouter(Model(doc, 2), KEY))
    d = decide(old, probe("kube")[0])
    assert str(d.floor) == "coding:3" and d.learned.outcome == "stale"
    assert "learned router v2 is stale (trained 45 days ago, more than the 30 allowed), so the rules decide" in d.reason
    forever = plain_cfg({"mode": "route", "max_age_days": 0})
    assert decide(Router(forever, old.backends, learned=old.learned), probe("kube")[0]).learned.used
    # Other rules: the weights corrected the old ones.
    changed = plain_cfg({"mode": "route"}, levels={"coding": 2})
    d = decide(router(changed), probe("kube")[0])
    assert d.learned.outcome == "stale" and "the level rules changed since it was trained" in d.reason
    # Another key: the features are hashed differently.
    d = decide(router(plain_cfg({"mode": "route"}), key="rotated"), probe("kube")[0])
    assert "its features were hashed differently" in d.reason


def test_the_rules_decide_when_the_model_fails(monkeypatch):
    r = router(plain_cfg({"mode": "route"}))
    monkeypatch.setattr(Model, "predict", lambda self, f: (_ for _ in ()).throw(RuntimeError("boom")))
    d = decide(r, probe("kube")[0])
    assert str(d.floor) == "coding:3" and d.learned.outcome == "failed"
    assert "learned router v1 failed (RuntimeError), so the rules decide" in d.reason
    monkeypatch.undo()
    nan = copy.deepcopy(trained()["doc"])
    nan["members"][0]["shift"]["coding"] = float("nan")
    with pytest.raises(ValueError, match="isn't a number"):
        Model(nan, 1).predict(probe("kube")[2])


def test_the_client_level_and_claude_codes_caps_come_first():
    r = router(plain_cfg({"mode": "route"}))
    explicit = decide(r, probe("kube")[0], {"x-route-floor": "coding:3"})
    assert explicit.learned is None and str(explicit.floor) == "coding:3"
    capped = decide(r, probe("kube")[0], {"x-claude-code-request-class": "auxiliary"})
    assert capped.floor.level == 1 and capped.learned.capped == (1, 1)
    assert "capped at 1 either way for a Claude Code request of this class" in capped.reason


def test_a_named_model_is_recorded_for_training_and_never_rerouted():
    r = router(plain_cfg({"mode": "route"}))
    d = decide(r, probe("kube")[0], model="cheap")
    assert [c.model.name for c in d.candidates] == ["cheap"] and d.learned.outcome == "named"
    assert d.learned.prediction is None and d.learned.features.rules == 3 and "learned router" not in d.reason
    assert decide(router(plain_cfg({"mode": "off"})), probe("kube")[0], model="cheap").learned is None
    assert decide(r, probe("kube")[0], {"x-route-floor": "coding:2"}, model="cheap").learned is None


def test_a_batch_line_is_leveled_as_a_live_task_and_its_other_side_priced_at_batch_routes():
    """A batch line (batches/, decision 45) gets its level as a new live task does, so route mode applies to it, and the
    level that didn't act is priced among routes with a batch API too. strong has none here: live kube work at coding:4
    goes to strong, a batch line's to best."""
    batch = ("cheap", "mid", "best")
    shadow = decide(router(plain_cfg({"mode": "shadow"}, batch=batch)), probe("kube")[0], batch=True)
    assert str(shadow.floor) == "coding:3" and shadow.candidates[0].model.name == "mid" and shadow.learned.outcome == "shadow"
    assert shadow.learned.alt_model == "best" and "at coding:4 it would have picked best" in shadow.reason
    assert all(c.backend.batch_api for c in shadow.candidates)
    r = router(plain_cfg({"mode": "route"}, batch=batch))
    d = decide(r, probe("kube")[0], batch=True)
    assert str(d.floor) == "coding:4" and d.learned.used and [c.model.name for c in d.candidates] == ["best"]
    assert "the rules' coding:3 would have picked mid" in d.reason
    assert decide(r, probe("kube")[0]).candidates[0].model.name == "strong"  # live, the same request
    # A named model's features are only ever training data, and a batch line is never recorded: none are made.
    named = decide(r, probe("kube")[0], model="cheap", batch=True)
    assert [c.model.name for c in named.candidates] == ["cheap"] and named.learned is None


def test_model_documents_are_checked_when_loaded():
    doc = trained()["doc"]
    for broken, message in [({**doc, "format": 9}, "unknown model format"), ({**doc, "members": []}, "no members"),
                            ({**doc, "unseen": ""}, "wrong size")]:
        with pytest.raises(ValueError, match=message):
            Model(broken, 1)
    bad = copy.deepcopy(doc)
    bad["members"][0]["text_w"] = floats([0.0] * 10)
    with pytest.raises(ValueError, match="expected 65536 values, found 10"):
        Model(bad, 1)
    assert rules_hash(LevelRules()) == rules_hash(LevelRules().model_dump())


def test_settings_are_checked():
    assert LearnedRouterConfig().mode == "off"
    assert LearnedRouterConfig(max_age_days="", retrain_hours=" ").max_age_days is None
    for bad in ({"mode": "on"}, {"min_confidence": 1}, {"train_days": 0}, {"typo": 1}):
        with pytest.raises(ValidationError):
            LearnedRouterConfig(**bad)


# ------------------------------------------------------------------ in the gateway


def gateway_cfg(tmp_path, learned: dict):
    raw = copy.deepcopy(TEST_CONFIG)
    raw["routing"] = {"learned": learned}
    path = tmp_path / "meritorder.yaml"
    path.write_text(yaml.safe_dump(raw))
    return load_config(path), raw


@pytest.fixture
async def shadow_gw(tmp_path):
    cfg, raw = gateway_cfg(tmp_path, {"mode": "shadow"})
    gw = build_gateway(cfg, store=MemoryStore(), state=MemoryState())
    gw.file_raw = raw
    await load_active(gw)
    app = create_app(gateway=gw)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        yield gw, client
    await gw.drain()


async def chat(client, content, headers=None, model="auto", key=TEST_KEY, messages=None, **extra):
    body = {"model": model, "messages": messages or [{"role": "user", "content": content}], **extra}
    return await client.post("/v1/chat/completions", headers={**auth(key), **(headers or {})}, json=body)


async def features_rows(gw) -> list[dict]:
    await gw.drain()
    return await gw.store.route_features(datetime.now(UTC) - timedelta(days=1))


async def test_shadow_mode_records_features_and_what_followed(shadow_gw):
    gw, client = shadow_gw
    key = await gw.store.get_setting(KEY_SETTING)
    assert key and gw.learned.key_loaded and gw.learned.fv == LearnedRouter(key=key).fv  # made once, kept in the store
    sid = {"x-session-id": "learn-1"}
    msgs = [{"role": "user", "content": "write a python function that parses dates"}]
    r = await chat(client, "", sid, messages=msgs)
    assert r.status_code == 200
    first = (await features_rows(gw))[0]
    assert first["fv"] == gw.learned.fv and first["task"] == "coding" and first["rules_level"] == 3 and first["auto"]
    assert first["served_model"] == r.headers["x-meritorder-model"] and first["served_score"] == 3.0 and first["status"] == 200
    assert first["learned"]["outcome"] == "missing" and not first["learned"]["used"]
    assert "parses" not in json.dumps(first, default=str)  # no content
    # The same request again, every message unchanged: a regenerated answer.
    assert (await chat(client, "", sid, messages=msgs)).status_code == 200
    rows_ = await features_rows(gw)
    assert len(rows_) == 2 and [r["retried"] for r in rows_] == [True, False]
    # A higher floor on the next prompt: the last answer's level was too low.
    more = msgs + [{"role": "assistant", "content": "ok"}, {"role": "user", "content": "now handle time zones"}]
    assert (await chat(client, "", {**sid, "x-route-floor": "coding:5"}, messages=more)).status_code == 200
    rows_ = await features_rows(gw)
    assert rows_[1]["floor_raised"] == 5 and len(rows_) == 2  # an explicit floor isn't second-guessed or recorded
    # A smart hint counts as one level more.
    again = more + [{"role": "assistant", "content": "ok"}, {"role": "user", "content": "and leap seconds"}]
    assert (await chat(client, "", {"x-session-id": "learn-2"}, messages=msgs)).status_code == 200
    assert (await chat(client, "", {"x-session-id": "learn-2", "x-route-hint": "smart"}, messages=again)).status_code == 200
    rows_ = await features_rows(gw)
    assert rows_[2]["floor_raised"] == 4
    text = (await client.get("/metrics")).text
    assert 'meritorder_learned_router_total{mode="shadow",outcome="missing"}' in text


async def test_refusals_errors_and_named_models_are_recorded(shadow_gw):
    gw, client = shadow_gw
    r = await chat(client, "explain this exploit [refuse:small-1]")
    assert r.status_code == 200 and r.headers["x-meritorder-refused-by"] == "small"
    rows_ = await features_rows(gw)
    assert len(rows_) == 1 and rows_[0]["refused_scores"] == [3.0] and rows_[0]["served_model"] == "big"
    assert (await chat(client, "summarize this", model="small")).status_code == 200
    named = (await features_rows(gw))[-1]
    assert not named["auto"] and named["learned"] is None and named["served_model"] == "small"
    gw.backends["mock-openai"].record_failure(time.time(), status=401)  # small's only backend is down
    gw.backends["mock-anthropic"].record_failure(time.time(), status=401)
    r = await chat(client, "hello there", headers={"x-session-id": "down"})
    assert r.status_code == 503  # nothing to record: no backend was tried


async def test_eval_traffic_isnt_recorded(shadow_gw):
    gw, client = shadow_gw
    r = await chat(client, "write a python function that parses dates", {"x-meritorder-tags": "eval=starter-1"})
    assert r.status_code == 200 and await features_rows(gw) == []
    # Claude Code side requests are capped at level 1: whatever their level, there's nothing to learn.
    assert (await chat(client, "title this session", {"x-claude-code-request-class": "auxiliary"})).status_code == 200
    assert await features_rows(gw) == []
    assert (await chat(client, "write a python function that parses dates")).status_code == 200
    assert len(await features_rows(gw)) == 1


async def test_batch_lines_record_no_features(shadow_gw):
    """No outcome signal can follow a batch line (no retry, raised floor, stopped stream, or new session), so it would only
    ever look like a success: batch lines get no route_features row, whether auto or named, and leave the live ones alone."""
    gw, client = shadow_gw
    assert (await chat(client, "write a python function that parses dates", {"x-session-id": "live-1"})).status_code == 200
    reqs = [{"custom_id": f"r{i}", "params": {"model": model, "max_tokens": 50,
                                              "messages": [{"role": "user", "content": "write a python function that parses dates"}]}}
            for i, model in enumerate(["auto", "auto", "small"])]
    r = await client.post("/v1/messages/batches", headers=auth(), json={"requests": reqs})
    assert r.status_code == 200, r.text
    for j in gw.store._jobs.values():
        j["next_poll_at"] = datetime.now(UTC) - timedelta(seconds=1)
    assert await BatchPoller(gw).run_once() >= 1
    await gw.drain()
    batch_rows = [row for row in gw.store.ledger if row.get("service_tier") == "batch"]
    assert len(batch_rows) == 3 and all(row["status"] == 200 for row in batch_rows)
    rows_ = await features_rows(gw)
    assert len(rows_) == 1 and rows_[0]["session_id"] == "live-1" and not rows_[0]["retried"]


async def test_a_trained_model_reaches_every_gateway_on_reload(shadow_gw, monkeypatch):
    gw, client = shadow_gw
    key = await gw.store.get_setting(KEY_SETTING)
    monkeypatch.setenv(KEY_ENV, key)
    v = await gw.store.save_router_model({**copy.deepcopy(trained()["doc"]), "feature_version": gw.learned.fv}, {}, "test", True)
    assert await load_active(gw) is False  # no stored configuration: the YAML file stays, the model is loaded
    assert gw.learned.model.version == v and gw.router.learned is gw.learned
    assert gw.config_token == token(None, None, v)
    r = await chat(client, "write a python function that parses dates")
    row = (await features_rows(gw))[-1]
    assert row["learned"]["version"] == v and row["learned"]["outcome"] in ("agrees", "shadow", "unsure")
    # A stored model that can't be read: the rules decide, and say why.
    v2 = await gw.store.save_router_model({"format": 1}, {}, "test", True)
    await load_active(gw)
    assert gw.learned.model is None and gw.learned.error == f"couldn't load model v{v2} (KeyError)"
    await gw.store.activate_router_model(None)
    await load_active(gw)
    assert gw.learned.model is None and gw.learned.error is None
    # Off: nothing is loaded or recorded.
    gw.cfg.routing.learned.mode = "off"
    before = gw.learned
    await gw.load_learned(v)
    assert gw.learned is before


async def test_a_second_gateway_uses_the_same_key(tmp_path):
    cfg, raw = gateway_cfg(tmp_path, {"mode": "shadow"})
    store = MemoryStore()
    a, b = build_gateway(cfg, store=store, state=MemoryState()), build_gateway(cfg, store=store, state=MemoryState())
    for gw in (a, b):
        gw.file_raw = raw
        await load_active(gw)
    assert a.learned.fv == b.learned.fv


# ------------------------------------------------------------------ the shadow report


def report_rows() -> list[dict]:
    """Rows as the gateway records them in shadow mode: agreements, disagreements up and down, and a used one."""
    data = rows(12, seed=4)
    for i, r in enumerate(data):
        r["kind"] = None
        level = [3, 4, 2, 3][i % 4]
        used = i == 11
        outcome = "used" if used else ("agrees" if level == r["rules_level"] else "shadow")
        r["learned"] = {"mode": "route" if used else "shadow", "outcome": outcome, "used": used, "version": 1, "level": level,
                        "confidence": 0.9, "difficulty": level, "spread": 0.1, "alt_model": None if level == r["rules_level"] else "strong",
                        "alt_cost": None if level == r["rules_level"] else 0.03, "cost": None if level == r["rules_level"] else 0.01,
                        "capped": None, "fail": None if level == r["rules_level"] else [0.2, 0.1]}
    data[0]["learned"] = {"mode": "shadow", "outcome": "missing", "used": False, "version": None, "level": None}
    data[1]["retried"] = True
    return data


def test_the_report_compares_both_sides():
    rep = learned_jobs.report(report_rows())
    assert rep["requests"] == 12 and rep["outcomes"]["missing"] == 1 and rep["compared"] == 11
    assert rep["disagreed"] == rep["compared"] - rep["agreed"] > 0
    assert rep["cost"]["priced"] == rep["disagreed"] and rep["cost"]["difference"] == pytest.approx(rep["cost"]["learned"] - rep["cost"]["rules"])
    groups = {(g["task"], g["rules"], g["learned"]): g for g in rep["groups"]}
    up = groups[("coding", 3, 4)]
    assert up["requests"] >= 1 and up["expected_rules"] == 0.2 and up["expected_learned"] == 0.1
    assert sum(g["used"] for g in rep["groups"]) == 1
    row = rep["rows"][0]
    assert {"rules_model", "learned_model", "rules_cost", "learned_cost", "signals", "acted"} <= set(row)
    assert any(r["acted"] == "learned" for r in rep["rows"])
    assert set(rep["signal_rates"]) <= {"agreed", "higher", "lower"}


async def test_the_report_api_and_who_may_see_it(shadow_gw):
    gw, client = shadow_gw
    for r in report_rows():
        r["team"] = "default" if r["request_id"].int % 2 else "other"
        await gw.store.record_route_features(r)
    rep = (await client.get("/v1/reports/router", headers=auth(ADMIN_KEY))).json()
    assert rep["requests"] == 12 and rep["status"]["mode"] == "shadow" and "versions" in rep["status"]
    assert rep["answered_by"] == "memory"  # request features are never copied to ClickHouse (decision 47)
    lead = (await client.get("/v1/reports/router?days=3", headers=auth(LEAD_KEY))).json()
    assert 0 < lead["requests"] < 12 and "versions" not in lead["status"] and lead["days"] == 3
    assert (await client.get("/v1/reports/router", headers=auth(TEST_KEY))).status_code == 403


async def test_admins_train_and_choose_versions(shadow_gw):
    gw, client = shadow_gw
    h = auth(ADMIN_KEY)
    status = (await client.get("/v1/admin/router", headers=h)).json()
    assert status["mode"] == "shadow" and status["active"] is None and status["versions"] == [] and status["loaded"] is None
    assert (await client.post("/v1/admin/router/train", headers=h)).status_code == 202
    assert (await gw.state.get_json(TRAIN_REQUESTED))["by"] == "root"
    assert (await client.get("/v1/admin/router", headers=h)).json()["train_requested"]["by"] == "root"
    v = await gw.store.save_router_model({**copy.deepcopy(trained()["doc"]), "feature_version": gw.learned.fv}, {"better": True}, "t", False)
    r = await client.post("/v1/admin/router/use", headers=h, json={"version": v})
    assert r.status_code == 200 and r.json()["active"] == v and r.json()["loaded"] == v and r.json()["stale"] is None
    assert r.json()["versions"][0]["version"] == v
    assert (await client.post("/v1/admin/router/use", headers=h, json={"version": 99})).status_code == 404
    assert (await client.post("/v1/admin/router/use", headers=h, json={"version": None})).json()["loaded"] is None
    assert (await client.get("/v1/admin/router", headers=auth(LEAD_KEY))).status_code == 403


# ------------------------------------------------------------------ the training job


async def stored_rows(store, n=1500, seed=1, fv=None):
    for r in rows(n, seed=seed, fv=fv):
        r.pop("kind")
        await store.record_route_features(r)


async def test_train_and_save_stores_a_version_and_gateways_reload():
    store, state = MemoryStore(), MemoryState()
    cfg = plain_cfg()
    nothing = await learned_jobs.train_and_save(cfg, store, state, "test")
    assert nothing["version"] is None and "no request has been recorded yet" in nothing["why"]
    await store.set_setting(KEY_SETTING, KEY)
    await stored_rows(store)
    dry = await learned_jobs.train_and_save(cfg, store, state, "test", dry_run=True)
    assert dry["version"] is None and dry["metrics"]["requests"] == 1500 and await store.router_models() == []
    out = await learned_jobs.train_and_save(cfg, store, state, "test")
    assert out["version"] == 1 and out["active"] and await store.router_model_version() == 1
    assert await state.get_config_token() == token(None, None, 1)
    assert (await store.get_setting(LAST_TRAINING))["version"] == 1
    status = await learned_jobs.status(cfg, store, LearnedRouter(Model((await store.router_model(1))["doc"], 1), KEY), state)
    assert status["loaded"] == 1 and status["examples"] == 1500 and status["metrics"]["better"] and status["age_days"] == 0


async def test_the_worker_prunes_trains_when_asked_and_on_schedule():
    store, state = MemoryStore(), MemoryState()
    await store.set_setting(KEY_SETTING, KEY)
    await stored_rows(store, 1500)
    old = rows(3, seed=9, start=datetime.now(UTC) - timedelta(days=200))
    for r in old:
        r.pop("kind")
        await store.record_route_features(r)
    job = RouterTrainer()
    assert await job.maybe(plain_cfg({"mode": "off"}), store, state) is None
    assert await job.maybe(plain_cfg({"mode": "shadow"}), store, state) is None  # no schedule: prunes only
    assert len(await store.route_features(datetime(2000, 1, 1, tzinfo=UTC))) == 1500
    await state.set_json(TRAIN_REQUESTED, {"by": "root"})
    done = await job.maybe(plain_cfg({"mode": "off"}), store, state)  # asked from the console: trains even when off
    assert done["version"] == 1 and done["by"] == "console:root" and await state.get_json(TRAIN_REQUESTED) is None
    scheduled = plain_cfg({"mode": "shadow", "retrain_hours": 24})
    assert await job.maybe(scheduled, store, state) is None  # trained less than a day ago
    later = datetime.now(UTC) + timedelta(hours=25)
    assert (await job.maybe(scheduled, store, state, now=later))["version"] == 2


async def test_the_model_in_use_is_shown_however_many_versions_came_after_it():
    """Daily retraining that doesn't beat the rules saves a version a day and leaves the one in use active. The status
    (console, CLI) must still say which that is after 20 newer ones, not "none in use"."""
    store, state = MemoryStore(), MemoryState()
    v1 = await store.save_router_model({"a": 1}, {"better": True}, "worker", True)
    for _ in range(20):
        await store.save_router_model({"a": 2}, {"better": False}, "worker", False)
    s = await learned_jobs.status(plain_cfg(), store, None, state)
    assert s["active"] == v1 and s["metrics"] == {"better": True} and len(s["versions"]) == 20


async def test_old_request_features_are_deleted_with_the_learned_router_off_too():
    """keep_days is how long recorded features are kept, whatever the mode: turning the learned router off after a
    month in shadow mode must not keep those rows for good."""
    store, state = MemoryStore(), MemoryState()
    for r in rows(3, seed=9, start=datetime.now(UTC) - timedelta(days=200)):
        r.pop("kind")
        await store.record_route_features(r)
    assert await RouterTrainer().maybe(plain_cfg({"mode": "off"}), store, state) is None
    assert await store.route_features(datetime(2000, 1, 1, tzinfo=UTC)) == []


async def test_route_feature_storage(store):
    """Both stores: rows in, outcomes marked, read by time and feature version, pruned; models saved and chosen."""
    now = datetime.now(UTC)
    data = rows(3, seed=8, start=now - timedelta(hours=1))
    data[2]["fv"] = "f1-other"
    data[0]["ts"] = now - timedelta(days=100)
    for r in data:
        r.pop("kind")
        await store.record_route_features(r)
        await store.record_route_features(r)  # written twice: kept once
    assert await store.mark_route_outcome(str(data[1]["request_id"]), retried=True, floor_raised=4)
    assert await store.mark_route_outcome(str(data[1]["request_id"]), floor_raised=3)
    assert not await store.mark_route_outcome(str(uuid.uuid4()), retried=True)
    got = await store.route_features(now - timedelta(days=1), fv=LR.fv)
    assert len(got) == 1 and got[0]["retried"] and got[0]["floor_raised"] == 4 and list(got[0]["text_features"]) == data[1]["text_features"]
    assert len(await store.route_features(now - timedelta(days=200), until=now - timedelta(days=50))) == 1
    newest = await store.route_features(now - timedelta(days=200), limit=1, text=False)
    assert [r["request_id"] for r in newest] == [data[2]["request_id"]] and "text_features" not in newest[0]
    assert await store.route_features(now - timedelta(days=200), team="nobody") == []
    assert await store.prune_route_features(now - timedelta(days=90)) == 1
    assert await store.router_model_version() is None
    v1 = await store.save_router_model({"a": 1}, {"better": True}, "t", True)
    v2 = await store.save_router_model({"a": 2}, {"better": False}, "t", False)
    assert await store.router_model_version() == v1 and (await store.router_model(v2))["doc"] == {"a": 2}
    assert [m["version"] for m in await store.router_models()] == [v2, v1] and await store.router_model(99) is None
    assert await store.activate_router_model(v2) and await store.router_model_version() == v2
    assert not await store.activate_router_model(99)
    assert await store.activate_router_model(None) and await store.router_model_version() is None
    await store.add_setting("k", "first")
    await store.add_setting("k", "second")
    assert await store.get_setting("k") == "first"


# ------------------------------------------------------------------ the command line


@pytest.fixture
def pg_store(monkeypatch, tmp_path):
    store = MemoryStore()
    store.kind = "postgres"
    monkeypatch.setattr("meritorder.store.make_store", lambda url: store)
    monkeypatch.setenv("MERITORDER_ENV_FILE", str(tmp_path / "missing.env"))
    monkeypatch.delenv("MERITORDER_REDIS_URL", raising=False)
    path = tmp_path / "meritorder.yaml"
    raw = copy.deepcopy(TEST_CONFIG)
    raw["routing"] = {"learned": {"mode": "shadow"}}
    path.write_text(yaml.safe_dump(raw))
    return store, str(path)


def test_cli_trains_and_chooses_versions(pg_store, capsys):
    store, path = pg_store
    cli.main(["router", "train", "--config", path])
    assert "not saved: no request has been recorded yet" in capsys.readouterr().out
    import asyncio

    asyncio.run(store.set_setting(KEY_SETTING, KEY))
    asyncio.run(stored_rows(store, 1500))
    cli.main(["router", "train", "--config", path, "--dry-run", "--days", "7"])
    out = capsys.readouterr().out
    assert "1,500 requests from the last 7 days" in out and "held out" in out and "not saved" not in out.split("held out")[0]
    cli.main(["router", "train", "--config", path])
    out = capsys.readouterr().out
    assert "saved as version 1, in use: gateways load it within seconds" in out and "would raise" in out
    cli.main(["router", "use", "none", "--config", path])
    assert "no learned router model in use: the rules decide" in capsys.readouterr().out
    cli.main(["router", "use", "1", "--config", path])
    assert "gateways now use model v1" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="no learned router model v7"):
        cli.main(["router", "use", "7", "--config", path])
    cli.main(["router", "train", "--config", path, "--min-examples", "100000"])
    assert "training needs at least 100000" in capsys.readouterr().out


def test_cli_report(monkeypatch, capsys):
    rep = {"days": 7, "status": {"mode": "shadow", "loaded": 2, "age_days": 1.5, "examples": 1500, "stale": None},
           **learned_jobs.report(report_rows())}
    seen = {}

    def get(url, params=None, headers=None, timeout=None):
        seen.update(url=url, params=params)
        return httpx.Response(200, json=rep)

    monkeypatch.setattr(cli.httpx, "get", get)
    cli.main(["router", "report", "--gateway", "http://gw", "--key", "k", "--days", "7"])
    out = capsys.readouterr().out
    assert seen["url"] == "http://gw/v1/reports/router" and seen["params"]["days"] == 7
    assert "mode shadow; model v2, trained 1.5 days ago on 1,500 requests" in out and "agreeing with the rules" in out
    assert "the rules' picks $" in out and "coding" in out and "[used]" in out
    monkeypatch.setattr(cli.httpx, "get", lambda *a, **k: httpx.Response(403, text="admins and team leads only"))
    with pytest.raises(SystemExit, match="403"):
        cli.main(["router", "report", "--gateway", "http://gw", "--key", "k"])


def test_the_slope_starts_from_epochs_ability_scale():
    cfg = plain_cfg()
    assert learned_jobs.epoch_slope(cfg, "coding") == pytest.approx(0.42)  # no chances: 0.6 per level times 70%
    cfg.models[0].chances = {"coding": {1: 0.97, 2: 0.93, 3: 0.88, 4: 0.81, 5: 0.70}}
    cfg.threshold_chance = {"coding": 0.7}
    assert 0.4 < learned_jobs.epoch_slope(cfg, "coding") < 0.5


def test_when_nothing_can_run_at_the_other_level_it_says_so():
    r = router()
    for name in ("strong", "best"):
        r.backends[name].record_failure(time.time(), status=401)
    d = decide(r, probe("kube")[0])
    assert d.candidates[0].model.name == "mid" and d.learned.alt_model is None and "nothing could run at coding:4" in d.reason


def test_named_features_floor_is_a_floor():
    f = Floor("coding", 3, "coding: 1 tool")
    feats = features(f, {"messages": [], "tools": [{}]}, {}, 10, "openai", 0)
    assert isinstance(feats, Features) and "rule:tools" in feats.tags and math.isfinite(len(feats.text))
