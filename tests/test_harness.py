import json

from meritorder import harness


def test_write_pi_merges_and_backs_up(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "KEY_FILE", tmp_path / "api_key")
    agent = tmp_path / "agent"
    agent.mkdir()
    (agent / "models.json").write_text(json.dumps({"providers": {"ollama": {"baseUrl": "http://localhost:11434/v1"}}}))
    models = [{"id": "auto", "context_window": 1000000}, {"id": "claude-sonnet", "type": "chat"}, {"id": "text-embedding-3-small", "type": "embedding"}]
    path = harness.write_pi("http://gw:8080", models, agent)
    data = json.loads(path.read_text())
    assert "ollama" in data["providers"]  # existing providers kept
    mo = data["providers"]["meritorder"]
    assert mo["baseUrl"] == "http://gw:8080/v1" and mo["api"] == "openai-completions"
    assert mo["apiKey"].startswith("!cat ") and str(tmp_path / "api_key") in mo["apiKey"]
    assert mo["compat"]["sessionAffinityFormat"] == "openrouter"  # pi sends x-session-id
    assert [m["id"] for m in mo["models"]] == ["auto", "claude-sonnet"]
    assert mo["models"][0]["contextWindow"] == 1000000
    assert (agent / "models.json.bak").exists()


def test_write_claude_code(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "KEY_FILE", tmp_path / "api_key")
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"theme": "dark", "env": {"FOO": "1"}}))
    harness.write_claude_code("http://gw:8080/", settings)
    data = json.loads(settings.read_text())
    assert data["theme"] == "dark" and data["env"]["FOO"] == "1"
    assert data["env"]["ANTHROPIC_BASE_URL"] == "http://gw:8080"
    assert data["env"]["CLAUDE_CODE_GATEWAY_HINT_HEADERS"] == "1"  # else no x-claude-code-request-class
    assert data["apiKeyHelper"].startswith("cat ")


async def test_pi_session_header_is_used(client, gateway):
    from conftest import auth

    from meritorder.sessions import session_key

    headers = {**auth(), "x-session-id": "pi-session-123"}
    await client.post("/v1/chat/completions", headers=headers, json={"model": "small", "messages": [{"role": "user", "content": "hi"}]})
    assert await gateway.state.get_session(session_key("alice", "pi-session-123")) is not None
