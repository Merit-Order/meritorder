"""/v1/messages/count_tokens sends the prompt upstream, so team policy and limits apply to it."""

from conftest import auth

from meritorder.backends.base import Backend
from meritorder.config import BackendConfig, ModelConfig, Route


class Counting(Backend):
    def __init__(self, name):
        super().__init__(BackendConfig(name=name, kind="api", format="anthropic", base_url="https://x"))
        self.calls = 0

    async def count_tokens(self, body, headers):
        self.calls += 1
        return {"input_tokens": 42}


async def test_count_tokens_respects_backend_and_model_policy(gateway, client):
    us = Counting("anthropic-us")
    gateway.backends["anthropic-us"] = us
    gateway.cfg.models.append(ModelConfig(name="claude-x", quality={"general": 4},
                                          routes=[Route(backend="anthropic-us", model="claude-x-1", price={"input": 3, "output": 15})]))
    body = {"model": "claude-x", "messages": [{"role": "user", "content": "hello there"}]}
    team = gateway.cfg.team("default")

    r = await client.post("/v1/messages/count_tokens", json=body, headers=auth())
    assert r.json() == {"input_tokens": 42} and us.calls == 1

    team.allowed_backends = ["bedrock-eu"]  # data residency: this team's prompts may only go to the EU backend
    r = await client.post("/v1/messages/count_tokens", json=body, headers=auth())
    assert r.status_code == 200 and us.calls == 1  # estimated locally; nothing sent to the US backend
    assert r.json()["input_tokens"] > 0

    team.allowed_models = ["small"]
    r = await client.post("/v1/messages/count_tokens", json=body, headers=auth())
    assert r.status_code == 403

    team.allowed_models, team.allowed_backends = ["*"], ["*"]
    team.rate_limits.user_rpm = 1
    assert (await client.post("/v1/messages/count_tokens", json=body, headers=auth())).status_code == 200
    assert (await client.post("/v1/messages/count_tokens", json=body, headers=auth())).status_code == 429
    assert (await client.post("/v1/messages/count_tokens", json=[1, 2], headers=auth())).status_code in (400, 429)


async def test_sessions_are_scoped_to_their_user(gateway, client):
    """Another user sending the same session id gets a separate session (and spend cap)."""
    from conftest import LEAD_KEY

    from meritorder.sessions import session_key

    gateway.cfg.team("default").max_session_usd = 0.000001
    body = {"model": "big", "messages": [{"role": "user", "content": "hi"}]}
    h = {"x-session-id": "shared-123"}
    assert (await client.post("/v1/chat/completions", json=body, headers={**auth(), **h})).status_code == 200
    await gateway.drain()
    assert (await client.post("/v1/chat/completions", json=body, headers={**auth(), **h})).status_code == 429  # alice's cap
    r = await client.post("/v1/chat/completions", json=body, headers={**auth(LEAD_KEY), **h})
    assert r.status_code == 200 and r.headers["x-meritorder-session"] == "shared-123"  # lee is unaffected
    assert await gateway.state.get_session(session_key("alice", "shared-123")) is not None
    assert await gateway.state.get_session("shared-123") is None


async def test_key_tags_win_over_header_tags(gateway, client):
    from conftest import last_row

    hdr = {"x-meritorder-tags": "cost_center=marketing,project=x"}
    for p in gateway.authn.static.values():
        p.tags = {"cost_center": "eng-42"}
    r = await client.post("/v1/chat/completions", json={"model": "small", "messages": [{"role": "user", "content": "hi"}]},
                          headers={**auth(), **hdr})
    assert r.status_code == 200
    row = await last_row(gateway)
    assert row["tags"] == {"cost_center": "eng-42", "project": "x"}
