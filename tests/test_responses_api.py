import json

from conftest import auth, last_row

from meritorder.formats.responses import responses_to_chat


def test_responses_input_to_chat():
    body = {
        "model": "x", "instructions": "be brief", "max_output_tokens": 100, "reasoning": {"effort": "high"},
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "weather?"}]},
            {"type": "reasoning", "encrypted_content": "abc", "summary": []},
            {"type": "function_call", "call_id": "c1", "name": "weather", "arguments": '{"city":"Paris"}'},
            {"type": "function_call", "call_id": "c2", "name": "time", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "sunny"},
            {"type": "function_call_output", "call_id": "c2", "output": "noon"},
        ],
        "tools": [{"type": "function", "name": "weather", "parameters": {"type": "object"}}],
    }
    out = responses_to_chat(body)
    assert out["messages"][0] == {"role": "system", "content": "be brief"}
    assert out["messages"][1] == {"role": "user", "content": "weather?"}
    assert [tc["id"] for tc in out["messages"][2]["tool_calls"]] == ["c1", "c2"]  # parallel calls stay in one turn
    assert [m["role"] for m in out["messages"][3:]] == ["tool", "tool"]
    assert out["max_tokens"] == 100 and out["reasoning_effort"] == "high"
    assert out["tools"][0]["function"]["name"] == "weather"


async def test_responses_endpoint_non_stream_and_stream(client, gateway):
    body = {"model": "big", "input": "please use a tool", "tools": [{"type": "function", "name": "search", "parameters": {"type": "object"}}]}
    r = await client.post("/v1/responses", headers=auth(), json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["object"] == "response" and data["status"] == "completed"
    types = [o["type"] for o in data["output"]]
    assert types == ["message", "function_call"]
    assert data["usage"]["input_tokens"] > 0

    r = await client.post("/v1/responses", headers=auth(), json={**body, "stream": True})
    events = [json.loads(line[6:]) for line in r.text.splitlines() if line.startswith("data: ")]
    kinds = [e["type"] for e in events]
    assert kinds[0] == "response.created" and kinds[-1] == "response.completed"
    assert "response.output_text.delta" in kinds and "response.function_call_arguments.delta" in kinds
    assert [e["sequence_number"] for e in events] == list(range(1, len(events) + 1))
    final = events[-1]["response"]
    assert final["output"][1]["name"] == "search" and json.loads(final["output"][1]["arguments"])
    assert final["usage"]["output_tokens"] > 0
    await gateway.drain()
    assert (await last_row(gateway))["client_format"] == "openai"


async def test_previous_response_id_rejected(client):
    r = await client.post("/v1/responses", headers=auth(), json={"model": "small", "input": "hi", "previous_response_id": "resp_1"})
    assert r.status_code == 400 and "store" in r.json()["error"]["message"]
