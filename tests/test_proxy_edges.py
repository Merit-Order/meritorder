"""Gateway pipeline edge paths: bad bodies, token counting, embeddings failures, pins, budgets, compaction,
refusals in every format, and failures in side work (session, ledger, tracing) that must not fail the request."""

import json

import pytest
from conftest import ADMIN_KEY, LEAD_KEY, auth, last_row, ledger_row

from meritorder.backends.base import BackendError
from meritorder.backends.mock import MockBackend
from meritorder.config import BackendConfig, ModelConfig, Price, Route, Safeguards, ServeConfig
from meritorder.sessions import session_key

HI = [{"role": "user", "content": "hi"}]


async def chat(client, body, headers=None, path="/v1/chat/completions"):
    return await client.post(path, json=body, headers=headers or auth())


# ------------------------------------------------------------------ request validation


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/messages", "/v1/responses"])
async def test_bad_bodies_are_400s_in_the_clients_format(client, path):
    r = await client.post(path, content=b"{not json", headers={**auth(), "content-type": "application/json"})
    assert r.status_code == 400 and "valid JSON" in r.text
    r = await client.post(path, content=b"\xff\xfe", headers={**auth(), "content-type": "application/json"})
    assert r.status_code == 400
    r = await client.post(path, json=[1, 2], headers=auth())
    assert r.status_code == 400
    if path != "/v1/responses":
        r = await client.post(path, json={"model": "small", "messages": "hi"}, headers=auth())
        assert r.status_code == 400 and "'messages' list" in r.text
    r = await client.post(path, json={"model": "small", "messages": HI})
    assert r.status_code == 401
    shape = r.json()
    assert ("type" in shape and shape["type"] == "error") if path == "/v1/messages" else "error" in shape


async def test_rejections_are_counted_in_metrics(client, gateway):
    await client.post("/v1/chat/completions", json={"model": "small", "messages": HI})
    await client.post("/v1/chat/completions", json={"model": "nope", "messages": HI}, headers=auth())
    text = (await client.get("/metrics")).text
    assert 'meritorder_rejected_requests_total{reason="authentication_error",status="401"}' in text \
        or 'meritorder_rejected_requests_total{status="401",reason="authentication_error"}' in text


# ------------------------------------------------------------------ count_tokens


async def test_count_tokens_estimates_without_an_anthropic_api_backend(client):
    r = await client.post("/v1/messages/count_tokens", json={"model": "big", "messages": [{"role": "user", "content": "hello " * 100}]},
                          headers=auth())
    assert r.status_code == 200 and r.json()["input_tokens"] > 50
    assert (await client.post("/v1/messages/count_tokens", json={"messages": HI})).status_code == 401
    r = await client.post("/v1/messages/count_tokens", content=b"nope", headers={**auth(), "content-type": "application/json"})
    assert r.status_code == 400


async def test_count_tokens_asks_the_provider_and_falls_back_on_errors(client, gateway):
    api = MockBackend(BackendConfig(name="anthropic-api", kind="api", format="anthropic"))
    gateway.backends["anthropic-api"] = api
    gateway.cfg.model("big").routes.insert(0, Route(backend="anthropic-api", model="claude-real"))
    seen = []

    async def exact(body, headers):
        seen.append(body["model"])
        return {"input_tokens": 1234}

    api.count_tokens = exact
    r = await client.post("/v1/messages/count_tokens", json={"model": "big", "messages": HI}, headers=auth())
    assert r.json() == {"input_tokens": 1234} and seen == ["claude-real"]

    async def failing(body, headers):
        raise BackendError(500, "down", retryable=True)

    api.count_tokens = failing
    r = await client.post("/v1/messages/count_tokens", json={"model": "auto", "messages": HI}, headers=auth())
    assert r.status_code == 200 and isinstance(r.json()["input_tokens"], int)  # estimate, not an error


# ------------------------------------------------------------------ embeddings


async def test_embeddings_validation(client, gateway):
    post = lambda body, headers=None: client.post("/v1/embeddings", json=body, headers=headers if headers is not None else auth())  # noqa: E731
    assert (await post({"model": "embed-small", "input": "x"}, {})).status_code == 401
    r = await client.post("/v1/embeddings", content=b"{", headers={**auth(), "content-type": "application/json"})
    assert r.status_code == 400
    assert (await post({"model": "embed-small"})).status_code == 400
    assert (await post({"model": "nope", "input": "x"})).status_code == 404
    assert (await post({"model": "small", "input": "x"})).status_code == 404  # a chat model isn't an embedding model
    gateway.cfg.team("default").allowed_models = ["small"]
    r = await post({"model": "embed-small", "input": "x"})
    assert r.status_code == 403 and "not allowed" in r.text


async def test_embeddings_failover_and_errors(client, gateway):
    backup = MockBackend(BackendConfig(name="embed-backup", kind="mock", format="openai"))
    gateway.backends["embed-backup"] = backup
    gateway.cfg.model("embed-small").routes.append(Route(backend="embed-backup", model="e2", price={"input": 0.05, "output": 0}))
    primary = gateway.backends["mock-openai"]
    original = primary.embed

    async def down(body):
        raise BackendError(503, "busy", retryable=True)

    primary.embed = down
    r = await client.post("/v1/embeddings", json={"model": "embed-small", "input": ["a", "b"]}, headers=auth())
    assert r.status_code == 200 and r.headers["x-meritorder-backend"] == "embed-backup"
    row = await last_row(gateway)
    assert row["backend"] == "embed-backup" and row["upstream_model"] == "e2" and row["client_format"] == "embeddings"
    assert not primary.healthy(__import__("time").time(), "embed-small-1")  # busy: skipped for a while, not just this once
    primary.record_success("embed-small-1")

    async def bad_request(body):
        raise BackendError(400, "input too long")

    primary.embed = bad_request
    r = await client.post("/v1/embeddings", json={"model": "embed-small", "input": "x"}, headers=auth())
    assert r.status_code == 400 and "input too long" in r.text  # the request's fault: no failover

    primary.embed, backup.embed = down, down
    r = await client.post("/v1/embeddings", json={"model": "embed-small", "input": "x"}, headers=auth())
    assert r.status_code == 503 and "busy" in r.text
    r = await client.post("/v1/embeddings", json={"model": "embed-small", "input": "x"}, headers=auth())
    assert r.status_code == 503 and "busy" in r.text  # both over capacity: still tried rather than refused outright

    primary.embed = original
    for b in (primary, backup):
        for _ in range(3):
            b.record_failure(__import__("time").time())
    r = await client.post("/v1/embeddings", json={"model": "embed-small", "input": "x"}, headers=auth())
    assert r.status_code == 503 and "no healthy backend" in r.text


# ------------------------------------------------------------------ budgets, pins, per-request cap


async def test_team_budget_reached(client, gateway):
    team = gateway.cfg.team("default")
    team.user_budget_usd_month, team.budget_usd_month = None, 1.0
    await gateway.store.record(ledger_row(user_id="bob", team="default", actual_cost=1.5))
    r = await chat(client, {"model": "small", "messages": HI})
    assert r.status_code == 429 and "team 'default' reached its monthly budget" in r.text


async def test_admin_backend_pin(client, gateway):
    body = {"model": "small", "messages": HI}
    r = await chat(client, body, {**auth(LEAD_KEY), "x-meritorder-backend": "mock-openai"})
    assert r.status_code == 403 and "only admins" in r.text
    r = await chat(client, body, {**auth(ADMIN_KEY), "x-meritorder-backend": "mock-anthropic"})
    assert r.status_code == 400 and "does not serve this model" in r.text
    r = await chat(client, {"model": "auto", "messages": HI}, {**auth(ADMIN_KEY), "x-meritorder-backend": "mock-anthropic"})
    assert r.status_code == 200 and r.headers["x-meritorder-backend"] == "mock-anthropic"


async def test_a_pinned_gpu_that_isnt_running_says_so(client, gateway):
    """Pinning a GPU deployment that's still starting (or stopped) is a wait, not a wrong model: found on kind, where a
    request right as the deployment warmed was told the backend "does not serve this model" until gateways saw it warm."""
    serve = ServeConfig(model="pricey", provisioner="static", static_url="http://gpu.test/v1")
    gpu = MockBackend(BackendConfig(name="gpu", kind="self_hosted", format="openai", mock_latency_ms=0, serve=serve))
    gateway.backends["gpu"] = gpu
    gateway.cfg.models.append(ModelConfig(name="pricey", quality={"general": 4}, routes=[
        Route(backend="mock-openai", model="pricey-api", price=Price(input=1, output=2)), Route(backend="gpu", model="pricey-gpu")]))
    for state in ({}, {"state": "provisioning"}):
        gpu.deploy = state
        r = await chat(client, {"model": "pricey", "messages": HI}, {**auth(ADMIN_KEY), "x-meritorder-backend": "gpu"})
        assert r.status_code == 503 and "backend 'gpu' isn't serving now" in r.text and r.headers["retry-after"]
    r = await chat(client, {"model": "small", "messages": HI}, {**auth(ADMIN_KEY), "x-meritorder-backend": "gpu"})
    assert r.status_code == 400 and "does not serve this model" in r.text  # a model it never serves: still the request's fault


async def test_per_request_cap_treats_gpus_as_free(client, gateway):
    gpu = MockBackend(BackendConfig(name="gpu", kind="self_hosted", format="openai", mock_latency_ms=0))
    gateway.backends["gpu"] = gpu
    gateway.cfg.models.append(ModelConfig(name="pricey", quality={"general": 4}, routes=[
        Route(backend="mock-openai", model="pricey-api", price={"input": 1000, "output": 1000}),
        Route(backend="gpu", model="pricey-gpu", price={"input": 1000, "output": 1000})]))
    gateway.cfg.safeguards = Safeguards(max_request_usd=0.01)
    r = await chat(client, {"model": "pricey", "max_tokens": 100, "messages": HI})
    assert r.status_code == 200 and r.headers["x-meritorder-backend"] == "gpu"


# ------------------------------------------------------------------ sessions


async def test_compaction_resets_warm_caches_and_allows_a_move(client, gateway):
    headers = {**auth(), "x-session-id": "compact-1"}
    convo = [{"role": "user", "content": "start a long task " * 50}]
    for i in range(3):
        r = await chat(client, {"model": "small", "messages": convo}, headers)
        assert r.status_code == 200
        convo += [{"role": "assistant", "content": f"step {i}"}, {"role": "user", "content": f"continue {i}"}]
    await gateway.drain()
    before = await gateway.state.get_session(session_key("alice", "compact-1"))
    assert before["warm_until"]
    # The harness compacts: history is replaced by a summary, so the prefix no longer matches.
    r = await chat(client, {"model": "small", "messages": [{"role": "user", "content": "summary of earlier work"}]}, headers)
    assert r.status_code == 200
    await gateway.drain()
    after = await gateway.state.get_session(session_key("alice", "compact-1"))
    assert after["turns"] == before["turns"] + 1


async def test_side_work_failures_never_fail_the_request(client, gateway, monkeypatch, caplog):
    import meritorder.proxy as proxy

    def broken_span(row, carrier):
        raise RuntimeError("collector down")

    async def broken_session(ctx):
        raise RuntimeError("redis hiccup")

    async def broken_record(row):
        raise RuntimeError("postgres down")

    monkeypatch.setattr(proxy, "record_span", broken_span)
    monkeypatch.setattr(gateway, "update_session", broken_session)
    monkeypatch.setattr(gateway.store, "record", broken_record)
    for stream in (False, True):
        r = await chat(client, {"model": "small", "stream": stream, "messages": HI})
        assert r.status_code == 200
    await gateway.drain()
    text = caplog.text
    assert "could not record a trace span" in text and "failed to update session" in text and "failed to write ledger row" in text


# ------------------------------------------------------------------ refusals in each format


@pytest.fixture
def two_strong_models(gateway):
    """big (Anthropic format) is cheapest; big-openai is the fallback."""
    gateway.cfg.models.append(ModelConfig(name="big-openai", quality={"coding": 5, "general": 5}, routes=[
        Route(backend="mock-openai", model="big-o", price={"input": 30, "output": 150})]))
    return gateway


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/messages"])
async def test_anthropic_refusal_retries_on_the_next_model(client, two_strong_models, stream, path):
    body = {"model": "auto", "max_tokens": 50, "stream": stream,
            "messages": [{"role": "user", "content": "do the thing [refuse:big-1]"}]}
    r = await client.post(path, json=body, headers={**auth(), "x-route-floor": "general:5"})
    assert r.status_code == 200 and r.headers["x-meritorder-model"] == "big-openai"
    assert r.headers["x-meritorder-refused-by"] == "big"
    assert "refusal" not in r.text and "content_filter" not in r.text


async def test_anthropic_client_on_openai_backend_non_stream(client):
    r = await chat(client, {"model": "small", "max_tokens": 30, "messages": HI}, path="/v1/messages")
    d = r.json()
    assert r.status_code == 200 and d["type"] == "message" and d["content"][0]["type"] == "text"
    assert d["usage"]["input_tokens"] > 0 and r.headers["x-meritorder-backend"] == "mock-openai"


# ------------------------------------------------------------------ candidates and attempts


async def test_attempts_are_capped_and_broken_backends_not_retried(client, gateway):
    calls = []

    def broken(name):
        async def send(body, stream, headers):
            calls.append((name, body["model"]))
            raise BackendError(503, f"{name} busy", retryable=True)
        return send

    for name in ("mock-openai", "mock-anthropic"):
        gateway.backends[name].send = broken(name)
    # Three models on the same two backends. Over capacity (503) is about one model on a backend, so the others
    # there are still tried; no route is tried twice.
    gateway.cfg.models.append(ModelConfig(name="small-2", quality={"coding": 3, "general": 3},
                                          routes=[Route(backend="mock-openai", model="small-2", price={"input": 0.3, "output": 1.2})]))
    gateway.cfg.model("small").routes.append(Route(backend="mock-anthropic", model="small-on-anthropic", price={"input": 0.25, "output": 1}))
    gateway.cfg.routing.max_attempts = 5
    r = await chat(client, {"model": "auto", "messages": HI}, {**auth(), "x-route-floor": "general:3"})
    assert r.status_code == 503
    assert len(calls) == len(set(calls)) == 4 and {c[0] for c in calls} == {"mock-openai", "mock-anthropic"}
    row = await last_row(gateway)
    assert row["status"] == 503 and "busy" in row["error"]

    calls.clear()
    for name in ("mock-openai", "mock-anthropic"):
        gateway.backends[name].record_success()
    gateway.cfg.routing.max_attempts = 1
    r = await chat(client, {"model": "auto", "messages": HI}, {**auth(), "x-route-floor": "general:3"})
    assert r.status_code == 503 and len(calls) == 1


async def test_non_retryable_error_stops_at_once(client, gateway):
    calls = []

    async def bad(body, stream, headers):
        calls.append(1)
        raise BackendError(400, "messages: invalid role")

    gateway.backends["mock-openai"].send = bad
    gateway.cfg.model("small").routes.append(Route(backend="mock-anthropic", model="s2", price={"input": 0.3, "output": 1.2}))
    r = await chat(client, {"model": "small", "messages": HI})
    assert r.status_code == 400 and "invalid role" in r.json()["error"]["message"] and calls == [1]


async def test_openai_passthrough_stream_drops_usage_chunks_unless_asked(client):
    r = await chat(client, {"model": "small", "stream": True, "messages": HI})
    chunks = [json.loads(line[6:]) for line in r.text.splitlines() if line.startswith("data: {")]
    assert chunks and not any(c.get("usage") and not c.get("choices") for c in chunks)
    r = await chat(client, {"model": "small", "stream": True, "stream_options": {"include_usage": True}, "messages": HI})
    chunks = [json.loads(line[6:]) for line in r.text.splitlines() if line.startswith("data: {")]
    assert any(c.get("usage") for c in chunks)
