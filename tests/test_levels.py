"""Levels come from the work (wording, tools, the client's floor or hint), never from how long the context is; a long
context only rules out the models whose context window can't hold it. The rules are configurable and visible."""

import time

import pytest
from conftest import LEAD_KEY, auth, last_row
from pydantic import ValidationError

from meritorder.backends.base import Backend, BackendError
from meritorder.classifier import classify, label, matched
from meritorder.config import BackendConfig, Config, LevelRules, ModelConfig, Price, Route, RoutingConfig, TeamConfig
from meritorder.router import Router, fits, reply_room
from meritorder.sessions import Session

TOOLS = [{"name": f"t{i}", "input_schema": {"type": "object"}} for i in range(8)]  # Claude Code sends dozens


def agent_turn(text: str = "optimize the CLI's startup", **kw) -> dict:
    return {"model": "auto", "tools": TOOLS, "messages": [{"role": "user", "content": text}], **kw}


# ------------------------------------------------------------------ the classifier


@pytest.mark.parametrize("size", [5_000, 90_000, 400_000, 2_000_000])
def test_a_long_coding_session_keeps_its_work_based_level(size):
    floor = classify(agent_turn(), {}, size)
    assert str(floor) == "coding:4" and floor.reason == "coding: 8 tools; hard wording (optimi[sz]e): +1"


@pytest.mark.parametrize("size", [1_000, 30_000, 500_000])
def test_a_long_general_request_is_general_work(size):
    assert str(classify({"messages": [{"role": "user", "content": "summarize this report"}]}, {}, size)) == "general:2"


def test_the_work_sets_the_level_and_the_reason_says_which_rule():
    def floor(text, tools=(), headers=None, size=None):
        f = classify({"messages": [{"role": "user", "content": text}], "tools": list(tools)}, headers or {}, size)
        return str(f), f.reason

    assert floor("what is the capital of Portugal?") == ("general:1", "short question: under 400 tokens, no tools")
    assert floor("what is the capital of Portugal?", size=401) == ("general:2", "general")
    assert floor("compare these two vendor proposals") == ("general:3", "analysis wording (compare)")
    assert floor("why does app.py crash?") == ("coding:3", "coding: code-like text (.py)")
    assert floor("go on", tools=[{"name": "t"}]) == ("coding:3", "coding: 1 tool")
    assert floor("fix the race condition", tools=TOOLS) == ("coding:4", "coding: 8 tools; hard wording (race condition): +1")
    assert floor("design a sharded architecture", tools=TOOLS) == ("coding:5", "coding: 8 tools; hardest wording (design (a|the)): level 5")
    assert floor("write tests for the parser", tools=TOOLS) == ("coding:2", "coding: 8 tools; simple task wording (write (unit )?tests): level 2")
    assert floor("compare these", headers={"x-route-hint": "smart"}) == ("general:4", "analysis wording (compare); X-Route-Hint: smart, +1")
    assert floor("hi", headers={"x-route-hint": "cheap"}) == ("general:1", "short question: under 400 tokens, no tools; X-Route-Hint: cheap, -1")
    explicit = classify({}, {"x-route-floor": "coding:5"})
    assert str(explicit) == "coding:5" and explicit.explicit and explicit.reason == "set by X-Route-Floor"


def test_wording_is_read_from_what_a_person_wrote_not_from_tool_results():
    """In an Anthropic agent loop the last user messages are often only tool results: they're passed over, so the wording
    of the prompts behind the task still counts mid-task. OpenAI's tool results are `tool` messages, never counted."""
    result = {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]}
    call = {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "bash", "input": {}}]}
    anthropic = {"tools": TOOLS[:2], "messages": [
        {"role": "user", "content": "fix the race condition in the scheduler"}, call, result,
        {"role": "user", "content": [{"type": "text", "text": "also keep the old flag working"}]}, call, result, call, result]}
    floor = classify(anthropic, {})
    assert str(floor) == "coding:4" and "hard wording (race condition)" in floor.reason
    openai = {"tools": [{"type": "function", "function": {"name": "bash"}}], "messages": [
        {"role": "user", "content": "fix the race condition in the scheduler"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "bash", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "t1", "content": "ok"}]}
    assert "hard wording (race condition)" in classify(openai, {}).reason
    # Two prompts back at most: older wording doesn't count, however many tool results lie between.
    older = {"tools": TOOLS[:2], "messages": [{"role": "user", "content": "fix the race condition"}, call, result,
                                               {"role": "user", "content": "rename a variable"}, call, result,
                                               {"role": "user", "content": "and another"}, call, result]}
    assert "hard wording" not in classify(older, {}).reason


def test_the_rules_are_configurable():
    rules = LevelRules(agent_by_tools=True, agent_tools=10, agent=5, hard_raise=0, coding_words=[r"\bkotlin\b"], short_tokens=0, general=3,
                       analysis_words=[r"\bweigh up\b"], analysis=4)
    body = lambda text, n=0: {"messages": [{"role": "user", "content": text}], "tools": TOOLS[:n]}  # noqa: E731
    assert str(classify(body("a race condition", 8), {}, rules=rules)) == "coding:3"  # 8 tools isn't an agent now; hard_raise off
    assert str(classify(body("go", 8) | {"tools": TOOLS + TOOLS}, {}, rules=rules)) == "coding:5"
    assert str(classify(body("kotlin coroutines"), {}, rules=rules)) == "coding:3"
    assert str(classify(body("why does app.py crash?"), {}, rules=rules)) == "general:3"  # .py isn't a coding word any more
    assert str(classify(body("hi"), {}, 10, rules=rules)) == "general:3"  # no short-question level
    assert str(classify(body("weigh up the options"), {}, rules=rules)) == "general:4"


def test_the_router_uses_the_configured_rules(gateway):
    gateway.cfg.routing.levels = LevelRules(hard_raise=0)
    d = gateway.router.decide("auto", gateway.cfg.team("default"), agent_turn(), {}, None, 5_000, True, {}, time.time())
    assert str(d.floor) == "coding:3" and d.reason.startswith("classified coding:3 (coding: 8 tools)")


@pytest.mark.parametrize("bad, says", [
    ({"coding_words": ["("]}, "isn't a valid regular expression"),
    ({"hard_words": ["   "]}, "would match every request"),
    ({"analysis_words": ["a*"]}, "matches empty text"),
    ({"analysis_words": ["(?i)plan"]}, "inline flags"),
    ({"agent": 6}, "less than or equal to 5"),
    ({"short": 0}, "greater than or equal to 1"),
    ({"agent_tools": 0}, "greater than or equal to 1"),
    ({"hard_raise": 5}, "less than or equal to 4"),
    ({"agent_tool": 8}, "Extra inputs are not permitted"),  # a misspelt rule isn't ignored
])
def test_bad_rules_are_refused(bad, says):
    with pytest.raises(ValidationError, match=says.replace("(", r"\(")):
        RoutingConfig(levels=bad)


def test_a_partial_rules_section_keeps_the_other_defaults():
    r = RoutingConfig.model_validate({"levels": {"agent_tools": 3}}).levels
    assert r.agent_tools == 3 and r.agent == 4 and r.coding_words == LevelRules().coding_words


def test_pattern_labels_and_matches_never_quote_the_prompt():
    assert label(r"\brefactor") == "refactor" and label(r"\.py\b") == ".py" and label(r"\bdesign (a|the)") == "design (a|the)"
    assert matched([r"\bclass \w+"], "class SecretProjectName:") == r"class \w+"  # the pattern, not the text it matched
    assert matched([r"\bbug\b", r"\brefactor"], "refactor it, then the bug") == "refactor"  # the first match in the text
    assert matched([], "anything") is None


# ------------------------------------------------------------------ the router: windows decide for long requests


def _router(models: list[ModelConfig]) -> tuple[Router, Config]:
    backends = [BackendConfig(name="api", format="anthropic", base_url="https://a")]
    cfg = Config(backends=backends, models=models, teams=[TeamConfig(name="default")])
    return Router(cfg, {b.name: Backend(b) for b in backends}), cfg


def _model(name: str, coding: float, window: int, price: float, max_output: int | None = None) -> ModelConfig:
    return ModelConfig(name=name, quality={"coding": coding, "general": coding}, context_window=window, max_output_tokens=max_output,
                       routes=[Route(backend="api", model=name, price=Price(input=price, output=price * 5))])


MODELS = [
    _model("fable", 5.0, 1_048_576, 8.0),  # the best, and dear
    _model("terra", 4.2, 922_000, 1.0),
    _model("open-coder", 4.0, 262_144, 0.3),  # the cheapest at level 4, with a small window
    _model("haiku", 3.0, 1_000_000, 0.5),  # not good enough for level 4
]


def test_a_long_coding_session_goes_to_the_cheapest_model_whose_window_fits():
    router, cfg = _router(MODELS)
    team = cfg.team("default")
    short = router.decide("auto", team, agent_turn(), {}, None, 100_000, True, {}, time.time())
    assert str(short.floor) == "coding:4" and short.candidates[0].model.name == "open-coder"

    long = router.decide("auto", team, agent_turn(), {}, None, 300_000, True, {}, time.time())
    assert str(long.floor) == "coding:4"  # the same level: length doesn't make the work harder
    assert [c.model.name for c in long.candidates] == ["terra", "fable"]  # not haiku (level), not open-coder (window)
    assert "ruled out open-coder (262,144): window too small for about 300,000 tokens" in long.reason


def test_the_reply_the_client_asked_for_needs_room_too():
    router, cfg = _router(MODELS)
    team = cfg.team("default")
    d = router.decide("auto", team, agent_turn(), {}, None, 240_000, True, {}, time.time())
    assert d.candidates[0].model.name == "open-coder"  # 240,000 is within 95% of 262,144
    d = router.decide("auto", team, agent_turn(max_tokens=32_000), {}, None, 240_000, True, {}, time.time())
    assert d.candidates[0].model.name == "terra"
    assert "ruled out open-coder (262,144): window too small for about 240,000 tokens plus a 32,000-token reply" in d.reason

    m = _model("m", 4, 100_000, 1.0, max_output=8_000)
    assert reply_room(m, {"max_tokens": 32_000}) == 8_000  # the gateway lowers it to what the model can write
    assert reply_room(m, {"max_completion_tokens": 5_000}) == 5_000 and reply_room(m, {}) == 0
    assert reply_room(m, {"max_tokens": "lots"}) == 0
    assert fits(m, 87_000, 8_000) and not fits(m, 87_001, 8_000)


def test_too_long_for_every_model_at_its_level_goes_to_the_largest_window():
    router, cfg = _router(MODELS)
    d = router.decide("auto", cfg.team("default"), agent_turn(), {}, None, 1_500_000, True, {}, time.time())
    assert str(d.floor) == "coding:4" and [c.model.name for c in d.candidates] == ["fable"]
    assert ("no model at coding:4 has room for about 1,500,000 tokens: sent to the largest window, fable (1,048,576), "
            "and the provider decides") in d.reason
    assert all(c.model.name != "haiku" for c in d.candidates)  # never below the level, whatever its window


def test_a_named_model_is_sent_anyway_and_the_reason_says_so():
    router, cfg = _router(MODELS)
    d = router.decide("open-coder", cfg.team("default"), agent_turn(), {}, None, 300_000, True, {}, time.time())
    assert [c.model.name for c in d.candidates] == ["open-coder"]
    assert "open-coder (262,144) may be too small for about 300,000 tokens: sent anyway, the provider decides" in d.reason


def test_a_mid_task_turn_says_why_its_task_has_its_level():
    router, cfg = _router(MODELS)
    s = Session(id="s", model="open-coder", backend="api", floor="coding:4", floor_reason="coding: 8 tools; hard wording (optimi[sz]e): +1", turns=3)
    body = agent_turn() | {"messages": [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "1", "name": "t0", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "1", "content": "ok"}]}]}
    d = router.decide("auto", cfg.team("default"), body, {}, s, 100_000, False, {}, time.time())
    assert d.reason.startswith("mid-task: keeping session model open-coder at coding:4 (coding: 8 tools; hard wording (optimi[sz]e): +1)")

    # It outgrows the model's window: the task keeps its level and moves to the cheapest model with room.
    d = router.decide("auto", cfg.team("default"), body, {}, s, 300_000, False, {}, time.time())
    assert d.candidates[0].model.name == "terra"
    assert ("outgrew open-coder's context window (300,000 of 262,144 tokens, 114%; estimate, not calibrated yet); "
            "re-routed at coding:4 (coding: 8 tools; hard wording") in d.reason


# ------------------------------------------------------------------ the gateway: end to end, the ledger, the console


async def test_too_long_for_every_model_ends_in_prompt_is_too_long(client, gateway, monkeypatch):
    """The fallback sends it to the largest window; when the provider refuses it, Claude Code sees the error it compacts on."""
    gateway.cfg.model("small").context_window = 1_000
    gateway.cfg.model("big").context_window = 2_000
    backend = gateway.backends["mock-anthropic"]

    async def send(body, stream, headers):
        raise BackendError(400, "This model's maximum context length is 2000 tokens. However, your messages resulted in 3100 tokens.")

    monkeypatch.setattr(backend, "send", send)
    r = await client.post("/v1/messages", headers=auth(), json={"model": "auto", "max_tokens": 50, "tools": TOOLS,
                                                                "messages": [{"role": "user", "content": "x" * 12_000}]})
    assert r.status_code == 400 and r.json()["error"]["message"].startswith("prompt is too long: This model's maximum context length")
    row = await last_row(gateway)
    assert "no model at coding:3 has room for about" in row["route_reason"] and "big (2,000)" in row["route_reason"]


async def test_the_ledger_and_the_feed_say_why(client, gateway):
    r = await client.post("/v1/chat/completions", headers=auth(), json={"model": "auto", "messages": [{"role": "user", "content": "what is 2+2?"}]})
    assert r.status_code == 200
    row = await last_row(gateway)
    assert row["floor"] == "general:1" and row["route_reason"].startswith("classified general:1 (short question: under 400 tokens, no tools)")
    rows = (await client.get("/v1/reports/requests", headers=auth())).json()["rows"]
    assert rows[0]["route_reason"] == row["route_reason"]
    feed = (await client.get("/v1/console/feed", headers=auth(LEAD_KEY))).json()["events"]
    assert feed[0]["reason"] == row["route_reason"]

    # A request that names a model says so.
    await client.post("/v1/chat/completions", headers=auth(), json={"model": "big", "messages": [{"role": "user", "content": "hi"}]})
    assert (await last_row(gateway))["route_reason"].startswith("harness requested big")


async def test_the_session_remembers_why_its_task_got_its_level(client, gateway):
    headers = {**auth(), "x-session-id": "why"}
    first = {"model": "auto", "tools": TOOLS[:2], "messages": [{"role": "user", "content": "fix the bug"}]}
    await client.post("/v1/messages", headers=headers, json={**first, "max_tokens": 50})
    turn = {**first, "max_tokens": 50, "messages": [
        {"role": "user", "content": "fix the bug"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "c1", "name": "t0", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "ok"}]}]}
    r = await client.post("/v1/messages", headers=headers, json=turn)
    assert r.status_code == 200, r.text
    assert (await last_row(gateway))["route_reason"].startswith("mid-task: keeping session model small at coding:3 (coding: 2 tools)")


async def test_anyone_signed_in_can_see_the_level_rules(client, gateway):
    gateway.cfg.routing.levels = LevelRules(agent_tools=6)
    gateway.cfg.teams[0].allowed_models = ["small"]
    data = (await client.get("/v1/reports/levels", headers=auth())).json()
    assert data["rules"]["agent_tools"] == 6 and data["defaults"]["agent_tools"] == 5
    assert "refactor" in data["labels"]["coding_words"] and "race condition" in data["labels"]["hard_words"]
    assert data["request_class_levels"] == {"auxiliary": 1, "compaction": 3} and data["auto_model"] == "auto"
    by_level = {lv["level"]: lv for lv in data["levels"]}
    assert by_level[3]["models"]["coding"] == [{"name": "small", "score": 3.0, "price": 1.2}]  # big isn't allowed for this team
    assert by_level[4]["models"]["coding"] == [] and by_level[5]["means"].startswith("the hardest")
    assert (await client.get("/v1/reports/levels")).status_code == 401


def test_the_shipped_gemini_routes_go_through_the_same_window_check():
    """Gemini's native-API routes (`gemini`, `vertex-gemini-native`) belong to the same models as its other routes, so
    they share each model's window from the catalog: a long session at coding:4 keeps them while the window holds it."""
    from pathlib import Path

    from meritorder.config import build_config, load_raw_config

    here = Path(__file__).resolve().parent.parent / "config"
    cfg = build_config(load_raw_config(here / "meritorder.yaml"), here, quiet=True)
    cfg.routing.model_choice = "threshold"  # only models at the level: with expected cost, Gemini Argon (below it) is in reach too
    carbon = cfg.model("gemini-carbon")
    assert carbon.context_window == 1_048_576 and {"gemini", "vertex-gemini-native"} <= {r.backend for r in carbon.routes}
    router = Router(cfg, {b.name: Backend(b) for b in cfg.backends})
    body = agent_turn(max_tokens=32_000)
    d = router.decide("auto", cfg.team("default"), body, {}, None, 960_000, True, {}, time.time())
    assert str(d.floor) == "coding:4" and "ruled out" in d.reason
    assert d.candidates and all(fits(c.model, 960_000, reply_room(c.model, body)) for c in d.candidates)
    assert {"gemini", "vertex-gemini-native"} <= {c.backend.name for c in d.candidates if c.model.name == "gemini-carbon"}

    # Too long for every real model: the largest real window, not the shipped mock whose window isn't known.
    d = router.decide("auto", cfg.team("default"), body, {}, None, 1_100_000, True, {}, time.time())
    assert {c.model.name for c in d.candidates} == {"gemini-carbon"}
    assert "no model at coding:4 has room for about 1,100,000 tokens plus a 32,000-token reply: sent to the largest window" in d.reason
