"""POST /v1/responses/compact and POST /v1/responses/input_tokens (decision 54), through the official OpenAI SDK, on the
mock backends: the gateway writes the summary with a model the router picks, and the compaction item it returns is read
wherever the next turn goes."""

from __future__ import annotations

import httpx
import openai
import pytest
from conftest import TEST_KEY, auth, ledger

from meritorder.app import create_app
from meritorder.config import StoredResponsesConfig
from meritorder.formats.responses import ResponsesError, compaction_item, compaction_summary, responses_to_chat


@pytest.fixture(autouse=True)
def responses_key(monkeypatch):
    monkeypatch.setenv("MERITORDER_RESPONSES_KEY", "test-responses-key")


@pytest.fixture
async def http(gateway):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(gateway=gateway)), base_url="http://test") as c:
        yield c


def oai(http) -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(api_key=TEST_KEY, base_url="http://test/v1", http_client=http, max_retries=0)


CONVERSATION = [
    {"role": "user", "content": "fix the failing test in tests/test_app.py"},
    {"type": "function_call", "call_id": "c1", "name": "shell", "arguments": '{"cmd": "pytest -q"}'},
    {"type": "function_call_output", "call_id": "c1", "output": "1 failed: test_app.py::test_login"},
    {"role": "assistant", "content": "The login test fails because the token expired."},
    {"role": "user", "content": "go on"},
]


def spy(gateway, backend: str) -> list[tuple[dict, dict]]:
    sent: list[tuple[dict, dict]] = []
    b = gateway.backends[backend]
    original = b.send

    async def send(body, stream, headers, **kw):
        sent.append((body, headers))
        return await original(body, stream, headers, **kw)

    b.send = send
    return sent


async def test_compact_writes_the_summary_and_the_next_turn_reads_it(gateway, http):
    sent = spy(gateway, "mock-openai")
    client = oai(http)
    out = await client.responses.compact(model="small", input=CONVERSATION, instructions="you are a coding agent")
    assert out.object == "response.compaction" and out.id.startswith("resp_")
    # As OpenAI has it: every user message, then one compaction item.
    assert [o.type for o in out.output] == ["message", "message", "compaction"]
    assert [o.role for o in out.output[:2]] == ["user", "user"]
    item = out.output[-1]
    summary = compaction_summary(item.encrypted_content)
    assert summary and item.id.startswith("cmp_")
    assert out.usage.input_tokens > 0 and out.usage.output_tokens > 0
    body, headers = sent[0]
    assert body["messages"][0] == {"role": "system", "content": "you are a coding agent"}
    assert "Compact the conversation" in body["messages"][-1]["content"]
    assert headers["x-claude-code-request-class"] == "compaction"
    row = (await ledger(gateway))[-1]
    assert row["tags"]["compaction"] == "responses" and row["status"] == 200

    # The compacted window goes back as the next turn's input, to any backend.
    window = [o.model_dump(exclude_none=True, warnings=False) for o in out.output] + [{"role": "user", "content": "next step?"}]
    nxt = await client.responses.create(model="big", input=window)
    assert nxt.status == "completed"
    chat = responses_to_chat({"model": "x", "input": window})
    assert chat["messages"][2] == {"role": "user", "content": f"Summary of the conversation so far (earlier turns were compacted):\n\n{summary}"}


async def test_compaction_level_is_capped(gateway, http):
    gateway.cfg.routing.request_class_levels = {"compaction": 1}
    r = await http.post("/v1/responses/compact", headers={**auth(), "x-route-hint": "smart"}, json={"model": "auto", "input": "hi"})
    assert r.status_code == 200, r.text
    row = (await ledger(gateway))[-1]
    assert "capped at 1" in row["route_reason"]


async def test_compact_errors(gateway, http):
    assert (await http.post("/v1/responses/compact", headers=auth(), json={"input": "x"})).status_code == 400
    assert (await http.post("/v1/responses/compact", headers=auth(), json={"model": "small"})).status_code == 400
    assert (await http.post("/v1/responses/compact", headers=auth(), content=b"{nope")).status_code == 400
    assert (await http.post("/v1/responses/compact", headers=auth(), json=["x"])).status_code == 400
    assert (await http.post("/v1/responses/compact", headers={"authorization": "Bearer mo-wrong"}, json={})).status_code == 401
    assert (await http.post("/v1/responses/input_tokens", headers={"authorization": "Bearer mo-wrong"}, json={})).status_code == 401
    r = await http.post("/v1/responses/compact", headers=auth(), json={"model": "small", "previous_response_id": "resp_1"})
    assert r.status_code == 400 and "stored responses" in r.json()["error"]["message"]
    foreign = {"type": "compaction", "encrypted_content": "gAAAAABpM0Yj-opaque="}
    r = await http.post("/v1/responses/compact", headers=auth(), json={"model": "small", "input": [foreign]})
    assert r.status_code == 400 and "wasn't made by this gateway" in r.json()["error"]["message"]
    r = await http.post("/v1/responses/compact", headers=auth(), json={"model": "no-such-model", "input": "x"})
    assert r.status_code >= 400


async def test_compact_from_a_stored_chain(gateway, http):
    gateway.cfg.team("default").stored_responses = StoredResponsesConfig(enabled=True, retention_days=30)
    client = oai(http)
    first = await client.responses.create(model="small", input="remember the number 42")
    out = await client.responses.compact(model="small", previous_response_id=first.id, input=[{"role": "user", "content": "and 7"}],
                                         prompt_cache_key="k1")
    assert [o.type for o in out.output] == ["message", "message", "compaction"]
    assert out.output[0].content[0].text == "remember the number 42"


def test_compaction_items_decode():
    item = compaction_item("cmp_1", "the plan: ünïcode kept")
    assert compaction_summary(item["encrypted_content"]) == "the plan: ünïcode kept"
    assert compaction_summary("mo1.!!!") is None and compaction_summary("mo1.WzFd") is None and compaction_summary(None) is None
    with pytest.raises(ResponsesError):
        responses_to_chat({"input": [{"type": "compaction_trigger"}]})


async def test_input_tokens(gateway, http):
    client = oai(http)
    counted = await client.responses.input_tokens.count(model="small", input=CONVERSATION, instructions="be brief")
    assert counted.object == "response.input_tokens" and counted.input_tokens > 10
    more = await client.responses.input_tokens.count(model="small", input=CONVERSATION * 2)
    assert more.input_tokens > counted.input_tokens
    r = await http.post("/v1/responses/input_tokens", headers=auth(), json={"input": "hello"})
    assert r.status_code == 200 and r.json()["input_tokens"] > 0  # no model: the plain estimate
    gateway.cfg.team("default").allowed_models = ["big"]
    r = await http.post("/v1/responses/input_tokens", headers=auth(), json={"model": "small", "input": "hello"})
    assert r.status_code == 403
    gateway.cfg.team("default").allowed_models = ["*"]
    r = await http.post("/v1/responses/input_tokens", headers=auth(), json={"model": "small", "previous_response_id": "resp_1"})
    assert r.status_code == 400
    r = await http.post("/v1/responses/input_tokens", headers=auth(), json={"model": "small", "input": [{"type": "item_reference", "id": "x"}]})
    assert r.status_code == 400
