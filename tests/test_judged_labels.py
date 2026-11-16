"""Judged labels for the learned router (judged.py, decision 52): off by default and per team; sampled at request time
and replayed in the background without the client waiting; nothing of the prompt or the answers stored; the replay and
the judge metered as the team's internal client, against its budget, the cap and the organization's; a judge from
another family grading the pair in random order; the verdict joining training at a weight from how well it agrees with
the outcome signals; and the estimate, the report, the alerts and the CLI. Every model call goes to a mock backend."""

from __future__ import annotations

import copy
import json
import time
from datetime import UTC, datetime, timedelta

import httpx
import pytest
import yaml
from conftest import ADMIN_KEY, BACKINGS, LEAD_KEY, TEST_CONFIG, TEST_KEY, auth, real_state, real_store
from learned_synth import LR, probe, rows

from meritorder import alerts, cli, judged, learned_jobs
from meritorder.app import build_gateway, create_app
from meritorder.backends.mock import MockBackend
from meritorder.config import JudgedLabelsConfig, Price, build_config, load_config
from meritorder.configstore import load_active
from meritorder.evals.jobs import DEFAULT_JUDGES
from meritorder.formats.translate import text_of
from meritorder.judged import (CONCURRENCY, JUDGE_INSTRUCTIONS, Sample, choose_judge, estimate, internal_user, is_internal,
                               parse_pair, reply_text, transcript)
from meritorder.learned import KEY_ENV, Model, detection
from meritorder.learned_jobs import JUDGED_PRIOR, JUDGED_TAG, judged_agreement, judged_observations, train
from meritorder.state import MemoryState
from meritorder.store import MemoryStore

ON = {"enabled": True, "sample_rate": 1.0, "max_usd_month": 5.0, "judges": ["judge"]}
CODING = "write a python function that parses dates"


@pytest.fixture(autouse=True)
def no_feature_key_env(monkeypatch):
    monkeypatch.delenv(KEY_ENV, raising=False)


def judged_raw(settings: dict | None = None, learned: str = "shadow", **extra) -> dict:
    """The tests' configuration with a judge model of its own (another family than big and small, coding 4) and the
    default team's judged labels on at a sample rate of 1."""
    raw = copy.deepcopy(TEST_CONFIG)
    raw["models"]["judge"] = {"quality": {"coding": 4, "general": 4},
                              "routes": [{"backend": "mock-openai", "model": "judge-1", "price": {"input": 1.0, "output": 4.0}}]}
    raw["routing"] = {"learned": {"mode": learned}}
    raw["teams"]["default"]["judged_labels"] = {**ON, **(settings or {})}
    raw.update(extra)
    return raw


class Rolls:
    """The sampler's dice, in turn: the sample-rate roll, then which answer the judge sees first (under 0.5: the stronger)."""

    def __init__(self, *values: float):
        self.values = list(values)

    def random(self) -> float:
        return self.values.pop(0) if self.values else 0.9


@pytest.fixture
def verdict(monkeypatch):
    """The mock judge's reply: whatever ``verdict["reply"]`` holds (a dict is sent as JSON). Other requests are answered as
    the mock always answers them."""
    v = {"reply": {"a": {"good_enough": False, "why": "it misses the time zone"}, "b": {"good_enough": True, "why": "complete"},
                   "better": "b"}, "seen": []}
    original = MockBackend._reply

    def reply(self, body):
        prompt = "\n".join(text_of(m.get("content")) for m in body.get("messages") or [])
        if "<answer_a>" in prompt:
            v["seen"].append(prompt)
            r = v["reply"]
            return (r if isinstance(r, str) else json.dumps(r)), None
        return original(self, body)

    monkeypatch.setattr(MockBackend, "_reply", reply)
    return v


async def make_gw(tmp_path, raw: dict, backing: str = "memory"):
    path = tmp_path / "meritorder.yaml"
    path.write_text(yaml.safe_dump(raw))
    store, state = (MemoryStore(), MemoryState()) if backing == "memory" else (await real_store(), await real_state())
    gw = build_gateway(load_config(path), store=store, state=state)
    gw.file_raw = raw
    await load_active(gw)  # loads the learned router's feature key
    return gw


async def client_for(gw):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(gateway=gw)), base_url="http://test")


@pytest.fixture(params=BACKINGS)
async def jgw(tmp_path, request, verdict):
    gw = await make_gw(tmp_path, judged_raw(), request.param)
    async with await client_for(gw) as client:
        yield gw, client
    await settled(gw)
    await gw.state.close()
    await gw.store.close()


async def settled(gw) -> None:
    """Every background task finished, including the replays and judging they started."""
    for _ in range(30):
        await gw.drain()
        if not gw._tasks:
            return


async def chat(client, content, headers=None, model="auto", key=TEST_KEY, **extra):
    body = {"model": model, "messages": [{"role": "user", "content": content}], **extra}
    return await client.post("/v1/chat/completions", headers={**auth(key), **(headers or {})}, json=body)


async def feature_rows(gw) -> list[dict]:
    await settled(gw)
    return await gw.store.route_features(datetime.now(UTC) - timedelta(days=1))


async def ledger_rows(gw) -> list[dict]:
    await settled(gw)
    return (await gw.store.recent(1000))[::-1]


# ------------------------------------------------------------------ settings


def test_off_by_default_and_a_cap_is_required():
    cfg = build_config(copy.deepcopy(TEST_CONFIG), prices_data={}, quiet=True)
    s = cfg.team("default").judged_labels
    assert not s.enabled and s.max_usd_month is None and s.sample_rate == 0.01
    assert s.skip_classes == ["auxiliary", "compaction"] and s.min_level == 2 and s.stronger == "auto" and s.stronger_level == 5
    assert tuple(s.judges) == DEFAULT_JUDGES  # the evals' tier-matched judges, one per family
    with pytest.raises(ValueError, match="need a monthly spending cap"):
        JudgedLabelsConfig(enabled=True)
    with pytest.raises(ValueError, match="at least one judge"):
        JudgedLabelsConfig(enabled=True, max_usd_month=5, judges=[])
    assert JudgedLabelsConfig(max_usd_month="  ").max_usd_month is None  # ${VAR:-} unset: no cap, so it can't be on
    with pytest.raises(ValueError):
        JudgedLabelsConfig(sample_rate=0)


@pytest.mark.parametrize("settings, error", [
    ({"judges": ["nobody"]}, "judges names models that aren't configured chat models: nobody"),
    ({"judges": ["embed-small"]}, "aren't configured chat models: embed-small"),
    ({"stronger": "nobody"}, "stronger is 'nobody', which isn't a configured chat model"),
])
def test_settings_must_name_configured_models(settings, error):
    with pytest.raises(ValueError, match=error):
        build_config(judged_raw(settings), prices_data={}, quiet=True)


def test_the_team_must_be_allowed_to_use_the_models():
    raw = judged_raw({"stronger": "big"})
    raw["teams"]["default"]["allowed_models"] = ["small", "judge"]
    with pytest.raises(ValueError, match="'big', which team 'default' isn't allowed to use"):
        build_config(raw, prices_data={}, quiet=True)
    raw = judged_raw()
    raw["teams"]["default"]["allowed_models"] = ["*", "!judge"]
    with pytest.raises(ValueError, match="isn't allowed to use any of judge"):
        build_config(raw, prices_data={}, quiet=True)
    assert build_config(judged_raw({"stronger": "big"}), prices_data={}, quiet=True).team("default").judged_labels.enabled


# ------------------------------------------------------------------ what the judge reads


def test_the_transcript_has_the_conversation_and_tools_in_either_format():
    anthropic = {"system": [{"type": "text", "text": "You are terse."}],
                 "tools": [{"name": "bash", "description": "run a command", "input_schema": {"type": "object"}}],
                 "messages": [{"role": "user", "content": [{"type": "text", "text": "list files"}, {"type": "image", "source": {}}]},
                              {"role": "assistant", "content": [{"type": "thinking", "thinking": "secret plan"},
                                                                {"type": "tool_use", "id": "t1", "name": "bash", "input": {"cmd": "ls"}}]},
                              {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "a.py", "is_error": True}]}]}
    t = transcript(anthropic, "anthropic")
    assert "[system]\nYou are terse." in t and "- bash: run a command\n  parameters: {\"type\":\"object\"}" in t
    assert '[tool call bash] {"cmd":"ls"}' in t and "[tool result (error)]\na.py" in t and "[image: not shown]" in t
    assert "secret plan" not in t  # thinking isn't graded
    openai = {"tools": [{"type": "function", "function": {"name": "grep", "parameters": {"type": "object"}}}],
              "messages": [{"role": "system", "content": "be brief"}, {"role": "user", "content": [{"type": "text", "text": "find x"}]},
                           {"role": "assistant", "content": None, "tool_calls": [{"function": {"name": "grep", "arguments": "{\"q\": \"x\"}"}}]},
                           {"role": "tool", "tool_call_id": "c1", "content": "x.py:1"}, "junk"]}
    t = transcript(openai, "openai")
    assert "[system]\nbe brief" in t and "- grep\n" in t and '[tool call grep] {"q": "x"}' in t and "[tool result]\nx.py:1" in t


def test_answers_are_read_as_text_and_tool_calls():
    msg = {"type": "message", "content": [{"type": "thinking", "thinking": "hmm"}, {"type": "text", "text": "Done."},
                                          {"type": "tool_use", "name": "edit", "input": {"path": "a.py"}}]}
    assert reply_text(msg, "anthropic") == 'Done.\n[tool call edit] {"path":"a.py"}'
    chat_ = {"choices": [{"message": {"content": "ok", "reasoning_content": "hidden",
                                      "tool_calls": [{"function": {"name": "run", "arguments": "{}"}}]}}]}
    assert reply_text(chat_, "openai") == "ok\n[tool call run] {}"
    assert reply_text(None, "openai") == "" and reply_text({"choices": []}, "openai") == ""


def test_the_verdict_is_read_strictly():
    assert parse_pair('Sure. {"a": {"good_enough": true}, "b": {"good_enough": false}, "better": "A"}') == \
        {"a": True, "b": False, "better": "a"}
    assert parse_pair('{"a": {"good_enough": true}, "b": {"good_enough": true}}')["better"] == "same"
    for bad in ("no json", "{not json}", "[1]", '{"a": {"good_enough": "yes"}, "b": {"good_enough": true}}', '{"a": true, "b": true}'):
        assert parse_pair(bad) is None
    assert "must not guess" in JUDGE_INSTRUCTIONS and "ignore" in JUDGE_INSTRUCTIONS  # blind, and robust to injected instructions


def test_a_replay_has_to_be_able_to_say_more_than_the_answer():
    cfg = build_config(judged_raw({"stronger": "big"}), prices_data={}, quiet=True)
    s = cfg.team("default").judged_labels
    small, big = cfg.model("small"), cfg.model("big")
    assert judged.is_stronger(cfg, s, small, "coding") is None
    assert judged.is_stronger(cfg, s, big, "coding") == "the answer came from big, which scores at least as well as big"
    assert judged.is_stronger(cfg, s, cfg.model("judge"), "coding") is None
    assert judged.is_stronger(cfg, s, None, "coding") == "the answer's model has no score to compare with"
    s.stronger = "gone"
    assert judged.is_stronger(cfg, s, small, "coding") == "the stronger model gone isn't configured"
    assert judged.stronger_models(cfg, cfg.team("default"), s, "coding") == []
    assert transcript({"messages": [{"role": "user", "content": ["plain", 5, {"type": "text", "text": "x"}]}, "junk"]}, "openai") == \
        "[user]\nplain\nx"


def test_the_judge_is_from_another_family_than_both_answers_where_it_can_be():
    raw = copy.deepcopy(TEST_CONFIG)
    for name in ("claude-opus", "gpt-sol", "gemini-carbon"):
        raw["models"][name] = {"quality": {"coding": 4}, "routes": [{"backend": "mock-openai", "model": name, "price": {"input": 1, "output": 2}}]}
    cfg = build_config(raw, prices_data={}, quiet=True)
    team = cfg.team("default")
    backends = {b.name: MockBackend(b) for b in cfg.backends}
    judges = list(DEFAULT_JUDGES)
    assert choose_judge(cfg, backends, team, judges, "claude-haiku", "gpt-terra") == "gemini-carbon"
    assert choose_judge(cfg, backends, team, judges, "claude-haiku", "claude-fable") == "gpt-sol"
    assert choose_judge(cfg, backends, team, ["claude-opus", "gpt-sol"], "gpt-luna", "claude-fable") == "claude-opus"  # not the cheaper one's
    assert choose_judge(cfg, backends, team, ["claude-opus"], "claude-haiku", "claude-fable") == "claude-opus"  # the only one there is
    assert choose_judge(cfg, backends, team, ["nobody"], "a", "b") is None
    team.allowed_backends = ["!mock-openai", "*"]  # no route the team may use
    assert choose_judge(cfg, backends, team, judges, "claude-haiku", "gpt-terra") is None


# ------------------------------------------------------------------ in the gateway


async def test_a_sampled_request_is_replayed_and_judged_in_the_background(jgw, verdict):
    gw, client = jgw
    gw.judged.rng = Rolls(0.0, 0.9)  # sampled; the answer it got is shown to the judge first
    r = await chat(client, CODING, {"x-session-id": "s1", "x-meritorder-tags": "judged_label=replay"})
    assert r.status_code == 200 and r.headers["x-meritorder-model"] == "small"
    feats = await feature_rows(gw)
    assert len(feats) == 1  # the replay and the judge are never training data
    j = feats[0]["judged"]
    assert j["outcome"] == "judged" and j["good_enough"] is False and j["stronger_good_enough"] is True
    assert (j["first"], j["better"], j["served"], j["stronger"], j["judge"]) == ("cheaper", "stronger", "small", "big", "judge")
    kept = json.dumps(feats[0], default=str)
    assert "parses" not in kept and "time zone" not in kept  # no prompt, no answer, no judge's reasons
    # The judge read the conversation and both answers, the cheaper first this time.
    seen = verdict["seen"][0]
    assert CODING in seen and seen.index("[small-1] You said") < seen.index("[big-1] You said")
    rows_ = await ledger_rows(gw)
    by_tag = {row["tags"].get("judged_label"): row for row in rows_}
    assert set(by_tag) == {None, "replay", "judge"}  # a client can't set the tag: the gateway does, on its own requests only
    replay, judge = by_tag["replay"], by_tag["judge"]
    assert replay["user_id"] == judge["user_id"] == "judged-labels:default" and replay["team"] == judge["team"] == "default"
    assert replay["served_model"] == "big" and replay["route_reason"].startswith("classified coding:5")  # auto at the stronger level
    assert judge["requested_model"] == "judge" and judge["served_model"] == "judge"
    assert replay["savings_routing"] == judge["savings_routing"] == 0  # the gateway's own work isn't savings
    assert j["cost"] == pytest.approx(replay["actual_cost"] + judge["actual_cost"], abs=1e-6) and j["cost"] > 0
    # Spent against the team's budget and the cap (the internal client's month).
    assert await gw.store.spend_since("user_id", "judged-labels:default", datetime.now(UTC) - timedelta(days=1)) == \
        pytest.approx(j["cost"], abs=1e-6)
    text = (await client.get("/metrics")).text
    assert 'meritorder_judged_labels_total{outcome="judged"} 1\n' in text


async def test_the_order_is_random_and_the_verdict_follows_the_answers(tmp_path, verdict):
    gw = await make_gw(tmp_path, judged_raw())
    async with await client_for(gw) as client:
        gw.judged.rng = Rolls(0.0, 0.1)  # the stronger answer first: A is the stronger, B the cheaper
        assert (await chat(client, CODING)).status_code == 200
        j = (await feature_rows(gw))[0]["judged"]
        assert (j["first"], j["good_enough"], j["stronger_good_enough"], j["better"]) == ("stronger", True, False, "cheaper")
        seen = verdict["seen"][0]
        assert seen.index("[big-1] You said") < seen.index("[small-1] You said")


async def test_a_streamed_answer_in_anthropics_format_is_judged_too(tmp_path, verdict):
    gw = await make_gw(tmp_path, judged_raw())
    gw.judged.rng = Rolls(0.0)
    async with await client_for(gw) as client:
        r = await client.post("/v1/messages", headers={**auth(), "anthropic-version": "2023-06-01"},
                              json={"model": "auto", "max_tokens": 300, "stream": True, "messages": [{"role": "user", "content": CODING}]})
        assert r.status_code == 200 and "message_stop" in r.text
        j = (await feature_rows(gw))[0]["judged"]
        assert j["outcome"] == "judged" and j["served"] == "small"
        replay = next(row for row in await ledger_rows(gw) if row["tags"].get("judged_label") == "replay")
        assert replay["client_format"] == "anthropic" and not replay["stream"]  # the client's own format, not streamed
        assert "[small-1] You said" in verdict["seen"][0]  # the streamed answer, assembled


async def test_what_isnt_sampled(tmp_path, verdict):
    gw = await make_gw(tmp_path, judged_raw())
    async with await client_for(gw) as client:
        assert (await chat(client, "what is 2+2?")).status_code == 200  # a short question: level 1
        assert (await chat(client, CODING, {"x-claude-code-request-class": "compaction"})).status_code == 200
        assert (await chat(client, CODING, {"x-claude-code-request-class": "auxiliary"})).status_code == 200  # not even recorded
        assert (await chat(client, CODING, {"x-meritorder-tags": "eval=starter-1"})).status_code == 200  # not recorded
        assert (await chat(client, CODING, model="big")).status_code == 200  # already at the stronger level
        image = [{"type": "text", "text": CODING}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}]
        assert (await client.post("/v1/chat/completions", headers=auth(),
                                  json={"model": "auto", "messages": [{"role": "user", "content": image}]})).status_code == 200
        gw.judged.rng = Rolls(0.995)  # over the rate
        gw.cfg.team("default").judged_labels.sample_rate = 0.5
        assert (await chat(client, CODING, {"x-session-id": "rate"})).status_code == 200
        feats = await feature_rows(gw)
        assert len(feats) == 5 and all(f["judged"] is None for f in feats)
        assert not any(row["tags"].get("judged_label") for row in await ledger_rows(gw)) and not verdict["seen"]


async def test_nothing_is_sampled_while_the_learned_router_is_off(tmp_path, verdict):
    gw = await make_gw(tmp_path, judged_raw(learned="off"))
    async with await client_for(gw) as client:
        assert (await chat(client, CODING)).status_code == 200
        assert await feature_rows(gw) == [] and not verdict["seen"]
    off = judged_raw()
    off["teams"]["default"]["judged_labels"]["enabled"] = False
    gw = await make_gw(tmp_path, off)
    async with await client_for(gw) as client:
        assert (await chat(client, CODING)).status_code == 200
        assert (await feature_rows(gw))[0]["judged"] is None and not verdict["seen"]


async def test_a_team_without_settings_samples_with_the_defaults_and_a_cap_of_its_own(tmp_path, verdict):
    gw = await make_gw(tmp_path, judged_raw())
    async with await client_for(gw) as client:
        assert (await chat(client, CODING, key=ADMIN_KEY)).status_code == 200  # root is in "ops", which isn't configured
        assert (await feature_rows(gw))[0]["judged"]["outcome"] == "judged"
        replay = next(row for row in await ledger_rows(gw) if row["tags"].get("judged_label") == "replay")
        assert replay["user_id"] == "judged-labels:ops" and replay["team"] == "ops"
        rep = (await client.get("/v1/reports/router", headers=auth(ADMIN_KEY))).json()["judged"]
        assert {t["team"] for t in rep["teams"]} == {"default", "ops"}


async def test_the_cap_stops_new_samples_and_says_so(tmp_path, verdict):
    gw = await make_gw(tmp_path, judged_raw({"max_usd_month": 0.000001}))
    async with await client_for(gw) as client:
        assert (await chat(client, CODING, {"x-session-id": "one"})).status_code == 200
        await settled(gw)
        # The first sample started under the cap, so it finished; the second finds the cap spent, and isn't taken.
        assert (await chat(client, CODING + " again", {"x-session-id": "two"})).status_code == 200
        outcomes = [f["judged"]["outcome"] for f in await feature_rows(gw)]
        assert outcomes == ["judged", "capped"]
        assert sum(1 for row in await ledger_rows(gw) if row["tags"].get("judged_label") == "replay") == 1
        rep = (await client.get("/v1/reports/router", headers=auth(ADMIN_KEY))).json()["judged"]
        assert rep["outcomes"] == {"judged": 1, "capped": 1} and rep["labels"] == 1
        team = rep["teams"][0]
        assert team["team"] == "default" and team["capped"] and team["spent_month"] > team["cap"]
        assert rep["cost"]["total"] == pytest.approx(rep["cost"]["replay"] + rep["cost"]["judge"]) and rep["cost"]["total"] > 0
        sent = await alerts.check_budgets(gw.cfg, gw.store, gw.state)
        assert any("monthly cap on judged labels" in m and "No new sample starts" in m for m in sent)
        assert not any("judged-labels:default" in m for m in sent)  # no per-person alert for the internal client


async def test_the_organizations_budget_stops_replays(tmp_path, verdict):
    gw = await make_gw(tmp_path, judged_raw(safeguards={"org_budget_usd_month": 0.0000001}))
    async with await client_for(gw) as client:
        assert (await chat(client, CODING)).status_code == 200  # admitted with nothing spent; it spends past the budget
        j = (await feature_rows(gw))[0]["judged"]
        assert j["outcome"] == "dropped" and j["why"] == "the replay was refused: the organization's budget is spent (HTTP 429)"
        assert not any(row["tags"].get("judged_label") for row in await ledger_rows(gw))


async def test_a_sample_that_cant_start_in_time_is_dropped(tmp_path, verdict):
    gw = await make_gw(tmp_path, judged_raw({"window_seconds": 0.01}))
    for _ in range(CONCURRENCY):
        await gw.judged.slots.acquire()  # this gateway is busy with other samples
    async with await client_for(gw) as client:
        assert (await chat(client, CODING)).status_code == 200
        j = (await feature_rows(gw))[0]["judged"]
        assert j == {"outcome": "dropped", "why": "couldn't start within 0.01 seconds of the answer"}
        assert len(gw.judged.held) == 0  # its prompt and answer are gone
    for _ in range(CONCURRENCY):
        gw.judged.slots.release()


async def test_a_gateway_holds_only_so_many_samples(tmp_path, verdict, monkeypatch):
    monkeypatch.setattr(judged, "MAX_HELD", 0)
    gw = await make_gw(tmp_path, judged_raw())
    async with await client_for(gw) as client:
        assert (await chat(client, CODING)).status_code == 200
        assert (await feature_rows(gw))[0]["judged"] == {"outcome": "dropped", "why": "this gateway already held 0 samples"}


async def test_failed_replays_and_unusable_verdicts_are_dropped_with_what_they_cost(tmp_path, verdict):
    gw = await make_gw(tmp_path, judged_raw())
    async with await client_for(gw) as client:
        verdict["reply"] = "A is better, I think."
        assert (await chat(client, CODING, {"x-session-id": "junk"})).status_code == 200
        j = (await feature_rows(gw))[0]["judged"]
        assert j["outcome"] == "dropped" and j["why"] == "the judge's reply wasn't a usable verdict" and j["cost"] > 0
        gw.backends["mock-anthropic"].record_failure(time.time(), status=401)  # big, the only coding:5 model, can't run
        assert (await chat(client, CODING + " now", {"x-session-id": "down"})).status_code == 200
        j = (await feature_rows(gw))[1]["judged"]
        assert j["outcome"] == "dropped" and j["why"].startswith("the replay failed (HTTP ")
        rep = learned_jobs.report(await feature_rows(gw))["judged"]
        assert rep["samples"] == 2 and rep["outcomes"] == {"dropped": 2} and len(rep["why"]) == 2


async def test_settling_a_sample(tmp_path):
    gw = await make_gw(tmp_path, judged_raw())
    small, big = gw.cfg.model("small"), gw.cfg.model("big")

    def sample(answer="ok"):
        s = Sample("default", "coding", 3, "openai", {"messages": []}, {}, answer=answer)
        gw.judged.held.add(s)
        return s

    mark, run = gw.judged.settle(sample(), "r1", small, 499, "client disconnected")
    assert mark == {"outcome": "dropped", "why": "the answer didn't finish (HTTP 499)"} and run is None
    assert gw.judged.settle(sample(""), "r2", small, 200, None)[0]["why"] == "the answer had no text or tool calls to judge"
    mark, run = gw.judged.settle(sample(), "r3", big, 200, None)  # a failover to a model at the stronger level
    assert mark == {"outcome": "skipped", "why": "the answer came from big, already at coding:5 or above"} and run is None
    marked = Sample("default", "coding", 3, "openai", None, {}, mark={"outcome": "capped"})  # never held: pick marked it
    assert gw.judged.settle(marked, "r4", small, 200, None) == ({"outcome": "capped"}, None)
    mark, run = gw.judged.settle(sample(), "r5", small, 200, None)
    assert mark["outcome"] == "pending" and run is not None
    run.close()
    assert len(gw.judged.held) == 1  # only the pending one keeps its prompt


async def test_what_the_judging_says_when_it_cant_finish(tmp_path, monkeypatch):
    """The replay's and judge's own results, through a stand-in for the gateway's pipeline."""
    gw = await make_gw(tmp_path, judged_raw())
    replies = []

    async def send(team, kind, body, headers, fmt):
        return replies.pop(0)

    monkeypatch.setattr(gw.judged, "_send", send)

    def sample():
        s = Sample("default", "coding", 3, "openai", {"messages": [{"role": "user", "content": "hi"}]}, {}, answer="ok")
        s.served, s.request_id = "small", "r"
        return s

    ok = {"choices": [{"message": {"content": "better"}}]}
    replies[:] = [(429, None, {"x-meritorder-limit": "budget"}, "team 'default' reached its monthly budget of $1.00")]
    assert (await gw.judged._judge(sample()))["why"] == "the replay was refused: the team's budget is spent (HTTP 429)"
    replies[:] = [(429, None, {"x-meritorder-limit": "budget"}, "this session has spent $9")]
    assert (await gw.judged._judge(sample()))["why"] == "the replay was refused: a spending limit is spent (HTTP 429)"
    replies[:] = [(200, ok, {"x-meritorder-model": "small", "x-meritorder-cost-usd": "0.5"}, "")]
    j = await gw.judged._judge(sample())
    assert j == {"outcome": "dropped", "why": "the replay went to small, the answer's own model", "cost": 0.5}
    replies[:] = [(200, {"choices": [{"message": {"content": ""}}]}, {"x-meritorder-model": "big"}, "")]
    assert (await gw.judged._judge(sample()))["why"] == "the replay had no text or tool calls to judge"
    replies[:] = [(200, ok, {"x-meritorder-model": "big", "x-meritorder-cost-usd": "0.5"}, ""), (503, None, {}, "down")]
    j = await gw.judged._judge(sample())
    assert j["why"] == "the judge failed (HTTP 503)" and j["judge"] == "judge" and j["cost"] == 0.5
    gw.cfg.team("default").judged_labels.judges = ["nobody"]
    replies[:] = [(200, ok, {"x-meritorder-model": "big"}, "")]
    assert (await gw.judged._judge(sample()))["why"] == "no judge the team can use can run now"
    # Another gateway's samples spent the cap since this one was taken: it doesn't start.
    await gw.state.add_spend([f"{datetime.now(UTC):%Y-%m}:user:judged-labels:default"], 10.0)
    assert (await gw.judged._judge(sample()))["outcome"] == "capped"
    gw.cfg.team("default").judged_labels.enabled = False
    assert (await gw.judged._judge(sample()))["why"] == "judged labels were turned off before it ran"

    async def broken(s):
        raise RuntimeError("boom")

    monkeypatch.setattr(gw.judged, "_judge", broken)
    s = sample()
    await gw.judged.run(s)
    assert s.body is None and s.answer is None  # freed, whatever happened


# ------------------------------------------------------------------ training


def test_judged_labels_are_observations_with_a_tag_of_their_own():
    data = rows(40, judged=1.0)
    obs = judged_observations(data, 0.5)
    assert len(obs) == 40 and all(o.kind == "judged" and o.w == 0.5 and o.y in (0.0, 1.0) for o in obs)
    assert all(JUDGED_TAG in o.tags and not any(t.startswith(("client:", "format:")) for t in o.tags) for o in obs)
    assert detection(JUDGED_TAG)  # it moves how often a failure is seen, never a request's difficulty
    assert judged_observations(data, 0.0) == []
    data[0]["judged"] = {"outcome": "dropped", "why": "x"}
    data[1]["status"] = 499
    data[2]["served_score"] = None
    assert len(judged_observations(data, 1.0)) == 37


def test_their_weight_comes_from_how_well_they_agree_with_the_outcome_signals():
    good = judged_agreement(rows(3000, judged=0.3), set())
    assert good["labels"] > 800 and good["caught"] > 0.85 and good["false_alarm"] < 0.2
    assert good["agreement"] == pytest.approx(good["caught"] - good["false_alarm"], abs=1e-3) and good["weight"] > 0.6
    guess = judged_agreement(rows(3000, judged=0.3, judge_noise=1.0), set())
    assert abs(guess["agreement"]) < 0.15 and guess["weight"] < 0.1  # a judge that only guesses hardly counts
    # Before any judged answer has a signal, the prior.
    quiet = judged_agreement(rows(200, judged=1.0, detect=0.0, noise=0.0), set())
    assert quiet["with_signal"] == 0 and quiet["agreement"] is None and quiet["weight"] == JUDGED_PRIOR
    assert judged_agreement(rows(10), set())["weight"] is None


def test_judged_labels_help_where_outcome_signals_are_rare():
    """With a retry seen for only one failure in ten, the outcome signals alone can't tell default-value changes from Kubernetes work;
    with judged labels on half the requests, it learns the known rule. The check against the rules stays on outcome labels."""
    cfg = build_config({"backends": {"b": {"format": "openai", "base_url": "https://b.example/v1"}},
                        "models": {"m": {"quality": {"coding": 3}, "routes": [{"backend": "b", "model": "m", "price": {"input": 1, "output": 1}}]}},
                        "quality": {"source": "config"}}, prices_data={}, quiet=True)

    def gap(result) -> float:
        m = Model(result["doc"], 1)
        return m.predict(probe("kube")[2]).difficulty - m.predict(probe("rename")[2]).difficulty

    without = train(rows(3000, detect=0.1), cfg, LR.fv, datetime.now(UTC))
    with_judged = train(rows(3000, detect=0.1, judged=0.5), cfg, LR.fv, datetime.now(UTC))
    assert without["metrics"]["judged"]["labels"] == 0 and with_judged["metrics"]["judged"]["labels"] > 1400
    assert 0 < with_judged["metrics"]["judged"]["weight"] <= 1
    assert gap(with_judged) > 1.0 > gap(without) + 0.5
    assert with_judged["metrics"]["holdout"]["requests"] == without["metrics"]["holdout"]["requests"]  # outcome labels only


# ------------------------------------------------------------------ the estimate


def traffic(n: int, prompt: int = 2000, output: int = 500, **kw) -> list[dict]:
    return [{"task": "coding", "rules_level": 3, "tags": ["class:main"], "served_model": "small", "status": 200,
             "prompt_tokens": prompt, "output_tokens": output, **kw} for _ in range(n)]


def test_the_estimate_prices_each_recorded_request_at_its_size():
    cfg = build_config(judged_raw(), prices_data={}, quiet=True)
    backends = {b.name: MockBackend(b) for b in cfg.backends}
    team, s = cfg.team("default"), cfg.team("default").judged_labels
    s.sample_rate = 0.1
    data = traffic(14) + traffic(3, status=499) + traffic(2, tags=["class:compaction"]) + traffic(2, rules_level=1)
    est = estimate(cfg, backends, team, s, data, days=14, budget_left=1000.0)
    # 14 eligible in 14 days: 30 a month, 3 samples. Each: a replay on big (2,000 written to its cache at $3.75, 1,000 out at
    # $15) and judging on judge (2,000 + 500 + 1,000 + 700 read at $1, 1,500 written at $4).
    replay, judging = (2000 * 3.75 + 1000 * 15) / 1e6, (4200 * 1 + 1500 * 4) / 1e6
    assert est["eligible"] == 14 and est["eligible_month"] == 30 and est["samples_month"] == 3
    assert est["usd_month"] == pytest.approx(3 * (replay + judging), abs=1e-4)
    assert est["per_sample_usd"] == pytest.approx(replay + judging, abs=1e-6)
    assert est["stronger"] == {"coding": ["big"]} and est["judges"] == ["judge"]
    lines = " ".join(est["warning"])
    assert "real costs have run up to three times as much" in lines and "sent to the stronger model (auto at level 5: big) and to a judge (judge)" in lines
    assert "Spending cap: $5.00 a month" in lines and "No new sample starts once it's spent" in lines
    # Long prompts cost more where the provider says so (tier-aware).
    cfg.model("big").routes[0].price = Price(input=3, output=15, cache_write=3.75, tiers=[{"above": 100_000, "input": 15, "output": 75}])
    long = estimate(cfg, backends, team, s, traffic(14, prompt=150_000), days=14)
    flat = estimate(build_config(judged_raw(), prices_data={}, quiet=True), backends, team, s, traffic(14, prompt=150_000), days=14)
    assert long["replay_usd_month"] > 3 * flat["replay_usd_month"]


def test_the_estimate_says_what_it_cant_price_and_what_the_cap_means():
    cfg = build_config(judged_raw(), prices_data={}, quiet=True)
    backends = {b.name: MockBackend(b) for b in cfg.backends}
    team, s = cfg.team("default"), cfg.team("default").judged_labels
    s.max_usd_month = None
    nothing = estimate(cfg, backends, team, s, [], days=14)
    assert nothing["usd_month"] == 0 and nothing["per_sample_usd"] is None
    assert any("No recorded task starts" in w for w in nothing["warning"]) and any("there's no default" in w for w in nothing["warning"])
    s.max_usd_month = 0.5
    s.stronger = "big"
    cfg.model("judge").routes[0].price = None  # the judge has no price
    est = estimate(cfg, backends, team, s, traffic(10, served_model="small"), days=7, budget_left=0.1)
    assert est["unpriced"] == ["judge"] and est["priced"] == 0
    lines = " ".join(est["warning"])
    assert "No live price for judge" in lines and "sent to the stronger model (big) and to a judge (judge)" in lines
    assert "$0.10 of its monthly budget left" in lines
    s.max_usd_month = 0.0001
    cfg.model("judge").routes[0].price = Price(input=1, output=4)
    assert any("below three times the estimate" in w for w in estimate(cfg, backends, team, s, traffic(10), days=7)["warning"])
    # Requests whose answer's model is as strong aren't counted; a stronger model without a price is named.
    assert estimate(cfg, backends, team, s, traffic(10, served_model="big"), days=7)["priced"] == 0
    cfg.model("big").routes[0].price = None
    est = estimate(cfg, backends, team, s, traffic(10), days=7)
    assert est["unpriced"] == ["big"] and est["priced"] == 0 and est["sent_to"] == {"stronger": ["big"], "judges": ["judge"]}


async def test_the_estimate_api_reads_the_teams_recorded_traffic(tmp_path, verdict):
    raw = judged_raw()
    raw["teams"]["default"]["judged_labels"]["enabled"] = False
    gw = await make_gw(tmp_path, raw)
    async with await client_for(gw) as client:
        for i in range(3):
            assert (await chat(client, f"{CODING} {i}", {"x-session-id": f"e{i}"})).status_code == 200
        assert (await chat(client, "what is 2+2?")).status_code == 200
        await settled(gw)
        body = {"team": "default", "settings": {"sample_rate": 0.5, "judges": ["judge"]}, "days": 7}
        est = (await client.post("/v1/admin/judged-labels/estimate", headers=auth(ADMIN_KEY), json=body)).json()
        assert est["team"] == "default" and est["learned_mode"] == "shadow" and est["requests"] == 4 and est["eligible"] == 3
        assert est["usd_month"] > 0 and est["confirm_above_usd"] == 1.0 and est["cap"] == 5.0
        assert (await client.post("/v1/admin/judged-labels/estimate", headers=auth(LEAD_KEY), json=body)).status_code == 403
        bad = {**body, "settings": {"sample_rate": 2}}
        assert (await client.post("/v1/admin/judged-labels/estimate", headers=auth(ADMIN_KEY), json=bad)).status_code == 422


async def test_judged_traffic_storage(store):
    """Both stores: a team's recorded task starts with their ledger token counts, and a verdict on a row."""
    now = datetime.now(UTC)
    data = rows(3, seed=5, start=now - timedelta(hours=1))
    data[2]["team"] = "other"
    for r in data:
        r.pop("kind")
        await store.record_route_features(r)
        await store.record({"id": r["request_id"], "ts": r["ts"], "team": r["team"], "user_id": r["user_id"], "status": 200,
                            "input_tokens": 100, "cache_read_tokens": 900, "cache_write_tokens": 0, "output_tokens": 50})
    got = await store.judged_traffic("eng", now - timedelta(days=1))
    assert len(got) == 2 and all(g["prompt_tokens"] == 1000 and g["output_tokens"] == 50 and g["task"] == "coding" for g in got)
    assert await store.mark_judged(str(data[0]["request_id"]), {"outcome": "judged", "good_enough": True})
    assert not await store.mark_judged("00000000-0000-0000-0000-000000000000", {"outcome": "judged"})
    feats = {str(r["request_id"]): r for r in await store.route_features(now - timedelta(days=1))}
    assert feats[str(data[0]["request_id"])]["judged"] == {"outcome": "judged", "good_enough": True}
    assert feats[str(data[1]["request_id"])]["judged"] is None
    # What the internal clients spent, by kind: from the ledger by user id (Postgres's index), never a tag breakdown.
    for user, kind, cost in (("judged-labels:eng", "replay", 0.25), ("judged-labels:eng", "judge", 0.05),
                             ("judged-labels:other", "replay", 1.0), ("alice", "replay", 9.0)):
        await store.record({"ts": now, "user_id": user, "team": user.split(":")[-1], "status": 200, "actual_cost": cost,
                            "tags": {"judged_label": kind}})
    await store.record({"ts": now, "user_id": "judged-labels:eng", "team": "eng", "status": 200, "actual_cost": 7.0, "tags": {}})
    got = await store.spend_by_tag(["judged-labels:eng"], "judged_label", now - timedelta(hours=1))
    assert got == pytest.approx({"replay": 0.25, "judge": 0.05})
    assert await store.spend_by_tag(["nobody"], "judged_label", now - timedelta(hours=1)) == {}


# ------------------------------------------------------------------ the report, the console and the CLI


def test_the_report_counts_verdicts_and_checks_the_judge():
    data = rows(400, judged=0.5)
    for i, r in enumerate(data[:20]):
        r["judged"] = {"outcome": "dropped", "why": "couldn't start within 60 seconds of the answer"} if i % 2 else {"outcome": "capped", "why": "spent"}
    rep = learned_jobs.report(data)["judged"]
    assert rep["samples"] == sum(1 for r in data if r.get("judged")) and rep["outcomes"]["dropped"] == 10 and rep["outcomes"]["capped"] == 10
    assert rep["why"]["couldn't start within 60 seconds of the answer"] == 10
    assert rep["labels"] == rep["outcomes"]["judged"] and sum(g["labels"] for g in rep["levels"]) == rep["labels"]
    assert rep["preferred"] > 0 and rep["preferred_first"] is not None and rep["preferred_stronger"] == 1.0
    assert rep["weight"] is not None and rep["recorded_cost"] > 0


async def test_who_sees_the_judged_labels_report(tmp_path, verdict):
    gw = await make_gw(tmp_path, judged_raw())
    async with await client_for(gw) as client:
        assert (await chat(client, CODING)).status_code == 200
        await settled(gw)
        lead = (await client.get("/v1/reports/router", headers=auth(LEAD_KEY))).json()["judged"]
        assert lead["labels"] == 1 and [t["team"] for t in lead["teams"]] == ["default"] and lead["cost"]["total"] > 0
        teams = (await client.get("/v1/console/teams", headers=auth(ADMIN_KEY))).json()["teams"]
        assert next(t for t in teams if t["name"] == "default")["judged_labels"] == {"enabled": True, "sample_rate": 1.0, "max_usd_month": 5.0}


def test_cli_report_shows_judged_labels(monkeypatch, capsys):
    data = rows(300, judged=0.5)
    rep = {"days": 7, "status": {"mode": "shadow", "loaded": None, "load_error": None, "stale": None}, **learned_jobs.report(data)}
    rep["judged"].update(cost={"replay": 1.5, "judge": 0.5, "total": 2.0},
                         teams=[{"team": "eng", "enabled": True, "sample_rate": 0.01, "cap": 2.0, "spent_month": 2.0, "capped": True}])
    monkeypatch.setattr(cli.httpx, "get", lambda *a, **k: httpx.Response(200, json=rep))
    cli.main(["router", "report", "--gateway", "http://gw", "--key", "k"])
    out = capsys.readouterr().out
    assert "judged labels:" in out and "cost $2.0000 (replays $1.5000, judging $0.5000)" in out
    assert "agreement with the outcome signals: caught" in out and "weight in training" in out
    assert "the judge preferred the answer shown first" in out and "[cap reached: no new samples until next month]" in out
    quiet = {**rep, "judged": {**learned_jobs.report(rows(50, judged=1.0, detect=0.0, noise=0.0))["judged"], "teams": []}}
    monkeypatch.setattr(cli.httpx, "get", lambda *a, **k: httpx.Response(200, json=quiet))
    cli.main(["router", "report", "--gateway", "http://gw", "--key", "k"])
    assert "no judged answer has an outcome signal yet" in capsys.readouterr().out


def test_cli_training_says_how_judged_labels_counted(capsys):
    cli._training({"days": 30, "metrics": {"requests": 10, "with_signal": 1, "judged": {"labels": 4, "not_good_enough": 1, "agreement": 0.5,
                                                                                         "weight": 0.4}}, "why": "too few"})
    assert "4 judged labels (1 not good enough), agreement with the outcome signals +0.50: weight 0.40" in capsys.readouterr().out
    cli._training({"days": 30, "metrics": {"requests": 10, "judged": {"labels": 4, "not_good_enough": 1, "agreement": None, "weight": 0.3}},
                   "why": "too few"})
    assert "no outcome signal on them yet: weight 0.30" in capsys.readouterr().out


def test_the_internal_client_is_known_by_its_name():
    assert internal_user("eng") == "judged-labels:eng" and is_internal("judged-labels:eng")
    assert not is_internal("alice") and not is_internal(None) and not is_internal("judged-labels")


async def test_the_internal_client_gets_no_spike_alerts(tmp_path):
    store, state = MemoryStore(), MemoryState()
    now = datetime.now(UTC)
    for user in ("judged-labels:eng", "alice"):
        await store.record({"ts": now - timedelta(minutes=5), "user_id": user, "team": "eng", "actual_cost": 50.0, "status": 200})
    sent = await alerts.check_spikes(store, state)
    assert len(sent) == 1 and "alice" in sent[0]
