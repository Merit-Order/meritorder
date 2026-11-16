"""Context windows (a conversation that outgrows its model, a prompt a provider refuses as too long) and the
headers Claude Code sends (its session id, subagents and side requests in their own threads, levels by request
class)."""

from conftest import auth, ledger

from meritorder.backends.base import BackendError
from meritorder.router import fits
from meritorder.sessions import Session, session_key, thread_suffix

TOOLS = [{"type": "function", "function": {"name": f"t{i}", "parameters": {"type": "object"}}} for i in range(2)]
CLAUDE_TOOLS = [{"name": f"t{i}", "input_schema": {"type": "object"}} for i in range(6)]


def first_turn(text: str = "fix the bug") -> dict:
    return {"model": "auto", "tools": TOOLS, "messages": [{"role": "user", "content": text}]}


def tool_turn(result: str) -> dict:
    """A turn in the middle of a task: the last message is a tool result, not a new prompt."""
    return {"model": "auto", "tools": TOOLS, "messages": [
        {"role": "user", "content": "fix the bug"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "t0", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": result},
    ]}


def test_fits_leaves_room_and_unknown_windows_fit():
    from meritorder.config import ModelConfig

    m = ModelConfig(name="m", context_window=1000, routes=[])
    assert fits(m, 950) and not fits(m, 951)
    assert fits(ModelConfig(name="n", routes=[]), 10**9)


async def test_a_session_that_outgrows_its_model_mid_task_moves_to_one_with_room(client, gateway):
    gateway.cfg.model("small").context_window = 1000
    headers = {**auth(), "x-session-id": "grow"}
    r = await client.post("/v1/chat/completions", headers=headers, json=first_turn())
    assert r.headers["x-meritorder-model"] == "small" and r.headers["x-meritorder-floor"] == "coding:3"

    r = await client.post("/v1/chat/completions", headers=headers, json=tool_turn("ok"))
    assert r.headers["x-meritorder-model"] == "small"  # mid-task and it fits: stays

    r = await client.post("/v1/chat/completions", headers=headers, json=tool_turn("x" * 8000))
    assert r.status_code == 200, r.text
    assert r.headers["x-meritorder-model"] == "big"  # no longer fits small: moved, keeping the task's level
    assert r.headers["x-meritorder-floor"] == "coding:3"


def test_router_says_why_it_moved(gateway):
    gateway.cfg.model("small").context_window = 1000
    session = Session(id="s", model="small", backend="mock-openai", floor="coding:3", turns=2)
    d = gateway.router.decide(requested="auto", team=gateway.cfg.team("default"), body=tool_turn("x" * 8000), headers={},
                              session=session, prompt_tokens=2500, boundary=False, utilization={}, now=0.0)
    assert [c.model.name for c in d.candidates] == ["big"]
    assert ("outgrew small's context window (2,500 of 1,000 tokens, 250%; estimate, not calibrated yet); re-routed at coding:3"
            in d.reason)


async def test_claude_code_side_and_compaction_requests_get_lower_levels(client, gateway):
    body = {"model": "auto", "max_tokens": 50, "tools": CLAUDE_TOOLS, "messages": [{"role": "user", "content": "refactor this for performance"}]}
    r = await client.post("/v1/messages", headers=auth(), json=body)
    assert r.headers["x-meritorder-floor"] == "coding:4"  # hard wording

    r = await client.post("/v1/messages", headers={**auth(), "x-claude-code-request-class": "auxiliary"}, json=body)
    assert r.headers["x-meritorder-floor"] == "coding:1"
    r = await client.post("/v1/messages", headers={**auth(), "x-claude-code-request-class": "compaction"}, json=body)
    assert r.headers["x-meritorder-floor"] == "coding:3"
    r = await client.post("/v1/messages", headers={**auth(), "x-claude-code-request-class": "main"}, json=body)
    assert r.headers["x-meritorder-floor"] == "coding:4"  # not listed: not capped

    # An explicit floor is the client's decision.
    r = await client.post("/v1/messages", headers={**auth(), "x-claude-code-request-class": "auxiliary",
                                                    "x-route-floor": "coding:4"}, json=body)
    assert r.headers["x-meritorder-floor"] == "coding:4"

    # Configurable; a level outside 1-5 is refused.
    gateway.cfg.routing.request_class_levels = {"auxiliary": 2}
    r = await client.post("/v1/messages", headers={**auth(), "x-claude-code-request-class": "auxiliary"}, json=body)
    assert r.headers["x-meritorder-floor"] == "coding:2"
    from meritorder.config import RoutingConfig

    try:
        RoutingConfig(request_class_levels={"auxiliary": 0})
    except ValueError as e:
        assert "1-5" in str(e)
    else:
        raise AssertionError("a level of 0 was accepted")


def test_thread_suffix():
    assert thread_suffix({}) == ""
    assert thread_suffix({"x-claude-code-agent-id": "a1"}) == "/agent:a1"
    assert thread_suffix({"x-claude-code-request-class": "auxiliary"}) == "/aux"
    assert thread_suffix({"x-claude-code-agent-id": "a1", "x-claude-code-request-class": "auxiliary"}) == "/agent:a1/aux"
    assert thread_suffix({"x-claude-code-request-class": "subagent"}) == ""


async def test_subagents_and_side_requests_keep_their_own_routing_state(client, gateway):
    """They share Claude Code's session id but are different conversations: a subagent's turn mustn't reset the main
    conversation's model and cache state. Spend and the ledger still count the whole session."""
    main = {**auth(), "x-claude-code-session-id": "cc-1"}
    await client.post("/v1/chat/completions", headers=main, json=first_turn())
    before = await gateway.state.get_session(session_key("alice", "cc-1"))
    assert before and before["model"] == "small" and before["msg_count"] == 1

    agent = {**main, "x-claude-code-agent-id": "explore-7", "x-claude-code-request-class": "subagent"}
    await client.post("/v1/chat/completions", headers=agent,
                      json={"model": "auto", "messages": [{"role": "user", "content": "design a distributed lock " + "y" * 3000}]})
    side = {**main, "x-claude-code-request-class": "auxiliary"}
    await client.post("/v1/chat/completions", headers=side, json={"model": "auto", "messages": [{"role": "user", "content": "title?"}]})

    after = await gateway.state.get_session(session_key("alice", "cc-1"))
    assert after == before  # untouched by the other two conversations
    sub = await gateway.state.get_session(session_key("alice", "cc-1/agent:explore-7"))
    aux = await gateway.state.get_session(session_key("alice", "cc-1/aux"))
    assert sub and sub["turns"] == 1 and aux and aux["floor"] == "general:1"

    rows = await ledger(gateway)
    assert [r["session_id"] for r in rows] == ["cc-1"] * 3  # one session in reports and spend limits


async def test_too_long_for_one_model_goes_to_a_bigger_one(client, gateway, monkeypatch):
    gateway.cfg.model("small").context_window = 1000
    backend = gateway.backends["mock-openai"]
    real = backend.send

    async def send(body, stream, headers):
        if body.get("model") == "small-1":
            raise BackendError(400, "This model's maximum context length is 1000 tokens. However, your messages resulted in 1200 tokens.")
        return await real(body, stream, headers)

    monkeypatch.setattr(backend, "send", send)
    r = await client.post("/v1/chat/completions", headers=auth(), json=first_turn())
    assert r.status_code == 200 and r.headers["x-meritorder-model"] == "big"
    assert "small-1 prompt too long" in r.headers["x-meritorder-failed-over"]

    # A named model can't move. Claude Code compacts on "prompt is too long", so the error says it that way.
    r = await client.post("/v1/messages", headers=auth(), json={"model": "small", "max_tokens": 10,
                                                                "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["type"] == "invalid_request_error" and err["message"].startswith("prompt is too long: This model's maximum context length")


async def test_a_mid_task_session_refused_as_too_long_is_re_routed(client, gateway, monkeypatch):
    """The token estimate said it fits, the provider disagreed: the pinned session moves instead of failing."""
    gateway.cfg.model("small").context_window = 100_000
    headers = {**auth(), "x-session-id": "pinned"}
    await client.post("/v1/chat/completions", headers=headers, json=first_turn())
    backend = gateway.backends["mock-openai"]
    real = backend.send

    async def send(body, stream, headers):
        if body.get("model") == "small-1":
            raise BackendError(400, "prompt is too long: 100500 tokens > 100000 maximum")
        return await real(body, stream, headers)

    monkeypatch.setattr(backend, "send", send)
    r = await client.post("/v1/chat/completions", headers=headers, json=tool_turn("ok"))
    assert r.status_code == 200, r.text
    assert r.headers["x-meritorder-model"] == "big"


# ------------------------------------------------------------------ reasoning effort across formats, cache keys


def test_fit_effort_keeps_what_the_model_accepts():
    from meritorder.proxy import fit_effort

    levels = ["low", "medium", "high"]
    assert fit_effort("high", levels) == "high"
    assert fit_effort("max", levels) == "high" and fit_effort("xhigh", levels) == "high"  # the highest below
    assert fit_effort("minimal", levels) == "low"  # nothing below: the lowest
    assert fit_effort("high", []) is None  # the model's levels aren't known: not sent (some models reject it)
    assert fit_effort(None, levels) is None and fit_effort("turbo", levels) is None


async def test_an_anthropic_clients_effort_reaches_an_openai_model(client, gateway, monkeypatch):
    """Claude Code sets output_config.effort; Chat Completions has no such field, so it was lost in translation."""
    gateway.cfg.backend("mock-openai").effort_format = "openai"
    gateway.cfg.model("small").effort_levels = ["low", "medium", "high"]
    backend = gateway.backends["mock-openai"]
    real, sent = backend.send, []

    async def send(body, stream, headers):
        sent.append(body)
        return await real(body, stream, headers)

    monkeypatch.setattr(backend, "send", send)
    body = {"model": "small", "max_tokens": 50, "output_config": {"effort": "max"}, "messages": [{"role": "user", "content": "hi"}]}
    r = await client.post("/v1/messages", headers=auth(), json=body)
    assert r.status_code == 200, r.text
    assert sent[-1]["reasoning_effort"] == "high"  # max isn't offered: the nearest level it accepts


def test_long_thread_ids_hash_into_the_cache_key_instead_of_being_cut():
    from meritorder.cache_shaping import shape_openai
    from meritorder.config import BackendConfig

    b = BackendConfig(name="o", kind="api", format="openai", supports_prompt_cache_key=True)
    short, a, c = {}, {}, {}
    shape_openai(short, b, "sess-1/agent:x")
    assert short["prompt_cache_key"] == "sess-1/agent:x"
    base = "s" * 60
    shape_openai(a, b, base + "/agent:one")
    shape_openai(c, b, base + "/agent:two")
    assert len(a["prompt_cache_key"]) == 64 and a["prompt_cache_key"] != c["prompt_cache_key"]


def test_request_class_is_matched_in_any_case():
    assert thread_suffix({"x-claude-code-request-class": "Auxiliary"}) == "/aux"


def test_a_mid_task_session_whose_model_is_over_capacity_everywhere_moves(gateway):
    """Before: it kept retrying the busy model; an `auto` session should move to a model that can answer now."""
    team = gateway.cfg.team("default")
    session = Session(id="s", model="small", backend="mock-openai", floor="coding:3", turns=2)
    gateway.backends["mock-openai"].record_failure(0.0, "small-1", 429, 60.0)
    d = gateway.router.decide(requested="auto", team=team, body=tool_turn("ok"), headers={}, session=session,
                              prompt_tokens=100, boundary=False, utilization={}, now=1.0)
    assert [c.model.name for c in d.candidates] == ["big"] and not d.last_resort
    assert "over capacity everywhere; re-routed" in d.reason

    # Nothing else can take it: the busy route is still tried, as a last resort.
    gateway.backends["mock-anthropic"].record_failure(0.0, "big-1", 429, 60.0)
    d = gateway.router.decide(requested="auto", team=team, body=tool_turn("ok"), headers={}, session=session,
                              prompt_tokens=100, boundary=False, utilization={}, now=1.0)
    assert d.last_resort and "trying" in d.reason


async def test_autos_window_counts_only_models_auto_can_pick(client, gateway):
    """A model without a score is never picked by auto, so its bigger window mustn't be advertised for auto."""
    gateway.cfg.model("small").context_window = 262_144
    gateway.cfg.model("big").context_window = 1_050_000
    windows = lambda r: {m["id"]: m.get("context_window") for m in r.json()["data"]}  # noqa: E731
    assert windows(await client.get("/v1/models", headers=auth()))["auto"] == 1_050_000
    gateway.cfg.model("big").quality = {"coding": 0, "general": 0}  # unmeasured
    w = windows(await client.get("/v1/models", headers=auth()))
    assert w["auto"] == 262_144 and w["big"] == 1_050_000  # its own entry still says what it holds
