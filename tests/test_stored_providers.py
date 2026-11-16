"""Stored mode against fakes of two real providers' APIs, through the official OpenAI SDK: Anthropic's Messages API and
OpenAI's Responses API, which stores responses itself. A conversation chained with previous_response_id moves between
them, fails over mid-chain, and, on a backend with responses_store, continues OpenAI's own stored response."""

from __future__ import annotations

import copy
import json

import httpx
import openai
import pytest
from conftest import BACKINGS, TEST_CONFIG, TEST_KEY, ledger, real_state, real_store

from meritorder.app import build_gateway, create_app
from meritorder.config import build_config
from meritorder.state import MemoryState
from meritorder.store import MemoryStore

USAGE = {"input_tokens": 40, "input_tokens_details": {"cached_tokens": 0}, "output_tokens": 5, "total_tokens": 45}


class FakeAnthropic:
    """Anthropic's Messages API: answers with how many messages it was sent and the last one."""

    def __init__(self):
        self.bodies: list[dict] = []
        self.fail: list[int] = []  # statuses to answer with first

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        if self.fail:
            return httpx.Response(self.fail.pop(0), json={"type": "error", "error": {"type": "overloaded_error", "message": "busy"}})
        last = body["messages"][-1]["content"]
        last = last if isinstance(last, str) else "".join(p.get("text", "") for p in last if p.get("type") == "text")
        return httpx.Response(200, json={"id": f"msg_{len(self.bodies)}", "type": "message", "role": "assistant", "model": body["model"],
                                         "content": [{"type": "text", "text": f"claude saw {len(body['messages'])}: {last}"}],
                                         "stop_reason": "end_turn", "usage": {"input_tokens": 30, "output_tokens": 6}})


class FakeOpenAI:
    """OpenAI's Responses API, storing what it's asked to store and chaining from it, and deleting it."""

    def __init__(self):
        self.bodies: list[dict] = []
        self.stored: dict[str, list] = {}  # response id -> the context it holds (input then output)
        self.deleted: list[str] = []
        self.fail: list[int] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.method == "DELETE":
            rid = request.url.path.rsplit("/", 1)[-1]
            self.deleted.append(rid)
            if self.stored.pop(rid, None) is None:
                return httpx.Response(404, json={"error": {"message": f"Response with id '{rid}' not found."}})
            return httpx.Response(200, json={"id": rid, "object": "response.deleted", "deleted": True})
        body = json.loads(request.content)
        self.bodies.append(body)
        if self.fail:
            return httpx.Response(self.fail.pop(0), json={"error": {"message": "server busy", "type": "server_error"}})
        context = list(body["input"])
        prev = body.get("previous_response_id")
        if prev:
            if prev not in self.stored:
                return httpx.Response(400, json={"error": {"message": f"Previous response with id '{prev}' not found.",
                                                           "type": "invalid_request_error", "param": "previous_response_id"}})
            context = self.stored[prev] + context
        rid = f"resp_up{len(self.bodies)}"
        texts = [p.get("text", "") for i in context if i.get("type") == "message" for p in
                 (i["content"] if isinstance(i["content"], list) else [{"text": i["content"]}])]
        output = [{"type": "reasoning", "id": f"rs_{rid}", "summary": []},
                  {"type": "message", "id": f"msg_{rid}", "role": "assistant", "status": "completed",
                   "content": [{"type": "output_text", "text": f"gpt saw {len(texts)}: {texts[-1] if texts else ''}"}]}]
        if body.get("store"):
            self.stored[rid] = context + output
        return httpx.Response(200, json={"id": rid, "object": "response", "status": "completed", "model": body["model"], "output": output,
                                         "usage": USAGE})


@pytest.fixture(autouse=True)
def keys(monkeypatch):
    monkeypatch.setenv("MERITORDER_RESPONSES_KEY", "provider-test-key")
    monkeypatch.setenv("MO_TEST_ANT", "key-ant")
    monkeypatch.setenv("MO_TEST_OAI", "key-oai")


def raw_config(responses_store: bool) -> dict:
    raw = copy.deepcopy(TEST_CONFIG)
    raw["backends"] = {
        "anth": {"kind": "api", "format": "anthropic", "base_url": "https://api.anthropic.example",
                 "auth": {"type": "api_key", "header": "x-api-key", "value_env": "MO_TEST_ANT"}},
        "oai": {"kind": "api", "format": "openai", "base_url": "https://api.openai.example/v1", "api": "responses",
                "responses_store": responses_store, "max_tokens_field": "max_completion_tokens",
                "auth": {"type": "api_key", "value_env": "MO_TEST_OAI"}},
    }
    raw["models"] = {
        "claude-x": {"quality": {"coding": 4, "general": 4}, "routes": [{"backend": "anth", "model": "claude-x-1", "price": {"input": 3, "output": 15}}]},
        "gpt-x": {"quality": {"coding": 4, "general": 4}, "routes": [{"backend": "oai", "model": "gpt-x-1", "price": {"input": 1, "output": 8}}]},
        # One model on both: the OpenAI route is cheaper, the Anthropic one is where it fails over to.
        "both": {"quality": {"coding": 4, "general": 4}, "routes": [{"backend": "oai", "model": "gpt-x-1", "price": {"input": 1, "output": 8}},
                                                                     {"backend": "anth", "model": "claude-x-1", "price": {"input": 3, "output": 15}}]},
    }
    raw["baseline_model"] = "claude-x"
    raw["teams"]["default"]["stored_responses"] = {"enabled": True, "retention_days": 7}
    return raw


@pytest.fixture(params=[False, True], ids=["gateway-only", "responses_store"])
def responses_store(request):
    return request.param


@pytest.fixture(params=BACKINGS)
async def setup(request, responses_store):
    store, state = (MemoryStore(), MemoryState()) if request.param == "memory" else (await real_store(), await real_state())
    gw = build_gateway(build_config(raw_config(responses_store), quiet=True), store=store, state=state)
    ant, oai_ = FakeAnthropic(), FakeOpenAI()
    gw.backends["anth"].client = httpx.AsyncClient(transport=httpx.MockTransport(ant))
    gw.backends["oai"].client = httpx.AsyncClient(transport=httpx.MockTransport(oai_))
    app = create_app(gateway=gw)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http:
        client = openai.AsyncOpenAI(api_key=TEST_KEY, base_url="http://test/v1", http_client=http, max_retries=0)
        yield gw, client, ant, oai_
    await gw.drain()
    await state.close()
    await store.close()


def said(item) -> str:
    """The text of a message: a dict as a provider got it, a content-block list, or an SDK item."""
    if not isinstance(item, dict):
        item = item.model_dump()
    c = item.get("content")
    return c if isinstance(c, str) else "".join(p.get("text", "") for p in c or [] if isinstance(p, dict))


def claude_saw(fake: FakeAnthropic) -> list[str]:
    return [said(m) for m in fake.bodies[-1]["messages"]]


async def test_a_chain_moves_between_providers(setup, responses_store):
    gw, client, ant, oai_ = setup
    r1 = await client.responses.create(model="claude-x", input="My name is Ada.", instructions="Be brief.")
    assert claude_saw(ant) == ["My name is Ada."]
    r2 = await client.responses.create(model="gpt-x", input="What is my name?", previous_response_id=r1.id, instructions="Be brief.")
    up = oai_.bodies[-1]
    # OpenAI never saw turn 1, so it gets all of it, whatever responses_store says, and the instructions of this turn only.
    assert "previous_response_id" not in up and [said(i) for i in up["input"] if i["type"] == "message"][-3:] == [
        "My name is Ada.", r1.output_text, "What is my name?"]
    assert up["store"] is responses_store
    if responses_store:
        assert up["instructions"] == "Be brief." and all(i.get("role") != "developer" for i in up["input"])
    # Without responses_store the instructions go as a developer message, as for any stateless request.
    assert r2.output_text == f"gpt saw {3 if responses_store else 4}: What is my name?"

    r3 = await client.responses.create(model="claude-x", input="And again?", previous_response_id=r2.id)
    assert claude_saw(ant) == ["My name is Ada.", r1.output_text, "What is my name?", r2.output_text, "And again?"]
    assert "system" not in ant.bodies[-1]  # r1's and r2's instructions aren't carried over
    rows = [await gw.store.get_stored_response(r.id) for r in (r1, r2, r3)]
    assert [r["backend"] for r in rows] == ["anth", "oai", "anth"]
    assert [bool(r["upstream_id"]) for r in rows] == [False, responses_store, False]


async def test_failover_mid_chain_keeps_the_whole_conversation(setup):
    gw, client, ant, oai_ = setup
    r1 = await client.responses.create(model="both", input="Remember 42.")
    assert oai_.bodies and not ant.bodies  # the cheaper route
    oai_.fail = [503]
    raw = await client.responses.with_raw_response.create(model="both", input="What did I ask you to remember?", previous_response_id=r1.id)
    r2 = raw.parse()
    assert "oai/gpt-x-1 503" in raw.headers["x-meritorder-failed-over"]
    assert claude_saw(ant) == ["Remember 42.", r1.output_text, "What did I ask you to remember?"]
    assert r2.output_text.startswith("claude saw 3")
    row = await gw.store.get_stored_response(r2.id)
    assert row["backend"] == "anth" and row["upstream_id"] is None and row["previous_response_id"] == r1.id
    # One ledger row a turn, as for any request that failed over.
    assert [(r["backend"], r["status"]) for r in await ledger(gw)] == [("oai", 200), ("anth", 200)]

    # The next turn goes back to OpenAI once it answers again, and still gets everything, including what Claude said.
    gw.backends["oai"].record_success("gpt-x-1")
    r3 = await client.responses.create(model="both", input="Thanks.", previous_response_id=r2.id)
    assert [said(i) for i in oai_.bodies[-1]["input"] if i["type"] == "message"][-3:] == [
        "What did I ask you to remember?", r2.output_text, "Thanks."]
    assert r3.output_text == "gpt saw 5: Thanks."


async def test_openai_continues_its_own_stored_response(setup, responses_store):
    """With responses_store, a turn that lands on OpenAI after a turn there goes as OpenAI's previous_response_id and only
    the new input, so OpenAI keeps its reasoning items. Anything else gets the whole conversation from the gateway."""
    if not responses_store:
        pytest.skip("needs responses_store")
    gw, client, ant, oai_ = setup
    r1 = await client.responses.create(model="gpt-x", input="One.", instructions="Count.")
    up1 = oai_.bodies[-1]
    assert up1["store"] is True and up1["instructions"] == "Count." and "previous_response_id" not in up1
    r2 = await client.responses.create(model="gpt-x", input="Two.", previous_response_id=r1.id, instructions="Count.")
    up2 = oai_.bodies[-1]
    assert up2["previous_response_id"] == "resp_up1" and [said(i) for i in up2["input"]] == ["Two."]
    assert up2["instructions"] == "Count."  # instructions are never carried over, so they're sent each turn
    assert r2.output_text == "gpt saw 3: Two."  # OpenAI's context: One., its answer, Two.

    # The gateway still holds it all: input_items and a turn elsewhere see the whole conversation.
    assert [said(i) for i in (await client.responses.input_items.list(r2.id, order="asc")).data] == ["One.", r1.output_text, "Two."]
    await client.responses.create(model="claude-x", input="Three.", previous_response_id=r2.id)
    assert len(ant.bodies[-1]["messages"]) == 5

    # OpenAI lost the turn before (expired, or deleted there): the gateway sends everything instead.
    oai_.stored.clear()
    r4 = await client.responses.create(model="gpt-x", input="Four.", previous_response_id=r2.id)
    assert oai_.bodies[-2]["previous_response_id"] == "resp_up2" and "previous_response_id" not in oai_.bodies[-1]
    assert [said(i) for i in oai_.bodies[-1]["input"]] == ["One.", r1.output_text, "Two.", r2.output_text, "Four."]
    assert r4.output_text == "gpt saw 5: Four."
    assert len(await ledger(gw)) == 4  # one row for the turn, not one per try

    # Deleting at the gateway deletes OpenAI's copy too.
    up4 = (await gw.store.get_stored_response(r4.id))["upstream_id"]
    await client.responses.delete(r4.id)
    assert oai_.deleted == [up4] and up4 not in oai_.stored


async def test_streamed_turns_note_the_providers_id(setup, responses_store):
    if not responses_store:
        pytest.skip("needs responses_store")
    gw, client, ant, oai_ = setup
    events = [{"type": "response.created", "response": {"id": "resp_streamed", "model": "gpt-x-1", "created_at": 1}},
              {"type": "response.output_text.delta", "delta": "hi"},
              {"type": "response.completed", "response": {"id": "resp_streamed", "status": "completed", "usage": USAGE}}]
    sse = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()
    gw.backends["oai"].client = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse)))
    stream = await client.responses.create(model="gpt-x", input="stream it", stream=True)
    final = [e async for e in stream][-1].response
    row = await gw.store.get_stored_response(final.id)
    assert row["upstream_id"] == "resp_streamed" and row["backend"] == "oai"


async def test_a_stream_the_provider_fails_is_kept_as_failed(setup):
    gw, client, ant, oai_ = setup
    events = [{"type": "response.created", "response": {"id": "resp_broken", "model": "gpt-x-1", "created_at": 1}},
              {"type": "response.output_text.delta", "delta": "half an answ"},
              {"type": "response.failed", "response": {"id": "resp_broken", "status": "failed",
                                                       "error": {"code": "server_error", "message": "the model crashed"}}}]
    sse = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()
    gw.backends["oai"].client = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse)))
    stream = await client.responses.create(model="gpt-x", input="go", stream=True)
    seen = [e async for e in stream]
    assert seen[-1].type == "response.failed" and "the model crashed" in seen[-1].response.error.message
    row = await gw.store.get_stored_response(seen[-1].response.id)
    assert row["status"] == "failed"
    with pytest.raises(openai.BadRequestError, match="is failed"):
        await client.responses.create(model="gpt-x", input="more", previous_response_id=row["id"])
    assert (await ledger(gw))[-1]["status"] >= 400
