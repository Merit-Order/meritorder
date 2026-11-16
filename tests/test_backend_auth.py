"""How the gateway authenticates to providers, which backends get built, and the exact-match response cache."""

import sys
import time
import types

import pytest

from meritorder.backends import build_backends
from meritorder.backends.auth import ApiKeyAuth, AzureADAuth, BackendAuthProvider, GoogleAuth, SigV4Auth, make_auth
from meritorder.backends.http import AnthropicBackend, OpenAIBackend
from meritorder.backends.mock import MockBackend
from meritorder.config import BackendAuth, BackendConfig, Config
from meritorder.response_cache import ResponseCacheConfig, cache_key, replay_stream, wanted

# ------------------------------------------------------------------ api keys


async def test_api_key_default_bearer_and_custom_header(monkeypatch):
    monkeypatch.setenv("MO_TEST_KEY", "sk-test")
    a = make_auth(BackendAuth(type="api_key", value_env="MO_TEST_KEY"))
    assert isinstance(a, ApiKeyAuth) and a.available()
    assert await a.apply("POST", "u", {"x": "1"}, b"") == {"x": "1", "authorization": "Bearer sk-test"}
    b = make_auth(BackendAuth(type="api_key", header="X-Api-Key", value_env="MO_TEST_KEY"))
    assert await b.apply("POST", "u", {}, b"") == {"x-api-key": "sk-test"}  # no prefix outside authorization
    c = make_auth(BackendAuth(type="api_key", header="authorization", prefix="Token ", value="lit"))
    assert await c.apply("POST", "u", {}, b"") == {"authorization": "Token lit"}


def test_api_key_missing_or_empty_env_is_unavailable(monkeypatch):
    monkeypatch.delenv("MO_TEST_MISSING", raising=False)
    assert not make_auth(BackendAuth(type="api_key", value_env="MO_TEST_MISSING")).available()
    monkeypatch.setenv("MO_TEST_MISSING", "")
    assert not make_auth(BackendAuth(type="api_key", value_env="MO_TEST_MISSING")).available()
    assert not make_auth(BackendAuth(type="api_key")).available()


async def test_no_auth_passes_headers_through():
    a = make_auth(BackendAuth())
    assert type(a) is BackendAuthProvider and a.available()
    assert await a.apply("POST", "u", {"h": "v"}, b"") == {"h": "v"}


# ------------------------------------------------------------------ AWS


@pytest.fixture
def no_aws(monkeypatch, tmp_path):
    """No ambient AWS credentials from the developer's machine."""
    for var in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_PROFILE", "AWS_DEFAULT_PROFILE",
                "AWS_BEARER_TOKEN_BEDROCK", "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", "AWS_CONTAINER_CREDENTIALS_FULL_URI",
                "AWS_WEB_IDENTITY_TOKEN_FILE", "AWS_ROLE_ARN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "none"))
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "none"))
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")


async def test_bedrock_bearer_token_takes_priority(no_aws, monkeypatch):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "brk-token")
    a = make_auth(BackendAuth(type="aws_sigv4", region="us-west-2"))
    assert isinstance(a, ApiKeyAuth) and await a.apply("POST", "u", {}, b"") == {"x-api-key": "brk-token"}
    # Only for the Bedrock (Mantle) service: other services still sign.
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIDEXAMPLE")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    assert isinstance(make_auth(BackendAuth(type="aws_sigv4", service="bedrock")), SigV4Auth)


def test_sigv4_without_credentials_is_unavailable(no_aws):
    a = make_auth(BackendAuth(type="aws_sigv4"))
    assert isinstance(a, SigV4Auth) and not a.available()
    assert a.service == "bedrock-mantle" and a.region == "us-east-1"


async def test_sigv4_signs_the_request(no_aws, monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIDEXAMPLE")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    a = make_auth(BackendAuth(type="aws_sigv4", service="bedrock", region="eu-west-1"))
    assert a.available()
    out = await a.apply("POST", "https://bedrock-runtime.eu-west-1.amazonaws.com/model/x/invoke", {"content-type": "application/json"}, b"{}")
    assert out["Authorization"].startswith("AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/")
    assert "/eu-west-1/bedrock/aws4_request" in out["Authorization"] and "X-Amz-Date" in out


# ------------------------------------------------------------------ Azure and Google (optional extras, faked)


def fake_module(monkeypatch, name: str, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


async def test_azure_ad_token_is_cached_until_near_expiry(monkeypatch):
    calls = []

    class Credential:
        async def get_token(self, scope):
            calls.append(scope)
            return types.SimpleNamespace(token=f"tok{len(calls)}", expires_on=time.time() + (100 if len(calls) == 1 else 3600))

    fake_module(monkeypatch, "azure")
    fake_module(monkeypatch, "azure.identity")
    fake_module(monkeypatch, "azure.identity.aio", DefaultAzureCredential=Credential)
    a = make_auth(BackendAuth(type="azure_ad"))
    assert isinstance(a, AzureADAuth) and a.available()
    assert (await a.apply("POST", "u", {}, b""))["authorization"] == "Bearer tok1"
    # The first token expires in 100 s, inside the 120 s refresh window: fetch a new one.
    assert (await a.apply("POST", "u", {}, b""))["authorization"] == "Bearer tok2"
    assert (await a.apply("POST", "u", {}, b""))["authorization"] == "Bearer tok2"  # cached
    assert calls == ["https://cognitiveservices.azure.com/.default"] * 2
    b = make_auth(BackendAuth(type="azure_ad", scope="api://custom/.default"))
    await b.apply("POST", "u", {}, b"")
    assert calls[-1] == "api://custom/.default"


async def test_google_refreshes_invalid_credentials(monkeypatch):
    class Creds:
        valid, token, refreshed = False, None, 0

        def refresh(self, request):
            self.refreshed += 1
            self.valid, self.token = True, "ya29.test"

    creds = Creds()
    google = fake_module(monkeypatch, "google")
    google_auth = fake_module(monkeypatch, "google.auth", default=lambda scopes: (creds, "proj"))
    google.auth = google_auth
    fake_module(monkeypatch, "google.auth.transport")
    fake_module(monkeypatch, "google.auth.transport.requests", Request=lambda: object())
    a = make_auth(BackendAuth(type="google"))
    assert isinstance(a, GoogleAuth)
    assert (await a.apply("POST", "u", {}, b"")) == {"authorization": "Bearer ya29.test"}
    await a.apply("POST", "u", {}, b"")
    assert creds.refreshed == 1  # still valid: no second refresh


# ------------------------------------------------------------------ build_backends


def test_build_backends_picks_classes_and_disables_unusable_ones(no_aws, monkeypatch):
    monkeypatch.setenv("MO_TEST_KEY", "k")
    monkeypatch.setitem(sys.modules, "azure.identity.aio", None)  # the azure extra isn't installed

    def broken(scopes):
        raise RuntimeError("no ADC found")

    google = fake_module(monkeypatch, "google")
    google.auth = fake_module(monkeypatch, "google.auth", default=broken)
    cfg = Config(backends=[
        BackendConfig(name="mock", kind="mock", format="openai"),
        BackendConfig(name="anth", format="anthropic", base_url="https://a", auth=BackendAuth(type="api_key", value_env="MO_TEST_KEY")),
        BackendConfig(name="oai", format="openai", base_url="https://o/v1", auth=BackendAuth(type="api_key", value_env="MO_TEST_KEY")),
        BackendConfig(name="off", enabled=False, format="openai", base_url="https://x"),
        BackendConfig(name="bedrock", format="anthropic", base_url="https://b", auth=BackendAuth(type="aws_sigv4")),
        BackendConfig(name="azure", format="openai", base_url="https://z", auth=BackendAuth(type="azure_ad")),
        BackendConfig(name="vertex", format="anthropic", flavor="vertex", project="p", auth=BackendAuth(type="google")),
    ], models=[])
    out = build_backends(cfg)
    assert isinstance(out["mock"], MockBackend) and isinstance(out["anth"], AnthropicBackend) and isinstance(out["oai"], OpenAIBackend)
    assert set(out) == {"mock", "anth", "oai"}
    reasons = {b.name: b.disabled_reason for b in cfg.backends if not b.enabled}
    assert reasons["off"] is None  # disabled in config, not by us
    assert reasons["bedrock"] == "no credentials found"
    assert reasons["azure"].startswith("missing optional dependency")
    assert reasons["vertex"] == "credentials unavailable (RuntimeError: no ADC found)"


# ------------------------------------------------------------------ response cache


def test_cache_wanted_by_header_or_team_setting():
    off, on = ResponseCacheConfig(), ResponseCacheConfig(enabled=True)
    assert not wanted(off, {}) and wanted(on, {})
    assert wanted(off, {"x-meritorder-cache": "ON"}) and not wanted(on, {"x-meritorder-cache": "off"})
    assert wanted(on, {"x-meritorder-cache": "maybe"}) and not wanted(off, {"x-meritorder-cache": "maybe"})


def test_cache_key_ignores_transport_fields_and_respects_scope():
    team, user = ResponseCacheConfig(), ResponseCacheConfig(scope="user")
    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    k = cache_key("openai", body, "t", "alice", team)
    assert k.startswith("rc:") and len(k) == 3 + 64
    assert k == cache_key("openai", {**body, "stream": True, "stream_options": {}, "metadata": {"a": 1}, "user": "u"}, "t", "bob", team)
    assert k == cache_key("openai", dict(reversed(list(body.items()))), "t", "alice", team)  # key order doesn't matter
    assert k != cache_key("anthropic", body, "t", "alice", team)
    assert k != cache_key("openai", {**body, "temperature": 0}, "t", "alice", team)
    assert k != cache_key("openai", body, "other-team", "alice", team)
    assert cache_key("openai", body, "t", "alice", user) != cache_key("openai", body, "t", "bob", user)


def test_replay_stream_anthropic_text_and_tool_use():
    resp = {"id": "msg", "type": "message", "role": "assistant", "model": "m", "stop_reason": "tool_use",
            "content": [{"type": "text", "text": "Let me look."}, {"type": "thinking", "thinking": "skip"},
                        {"type": "tool_use", "id": "tu1", "name": "read", "input": {"path": "a.py"}}],
            "usage": {"input_tokens": 5, "output_tokens": 9}}
    evs = replay_stream("anthropic", resp)
    assert [e.event for e in evs] == ["message_start", "content_block_start", "content_block_delta", "content_block_stop",
                                      "content_block_start", "content_block_delta", "content_block_stop", "message_delta", "message_stop"]
    assert evs[0].json()["message"]["content"] == []
    assert evs[2].json()["delta"] == {"type": "text_delta", "text": "Let me look."}
    assert evs[4].json()["content_block"] == {"type": "tool_use", "id": "tu1", "name": "read", "input": {}} and evs[4].json()["index"] == 2
    assert evs[5].json()["delta"] == {"type": "input_json_delta", "partial_json": '{"path": "a.py"}'}
    assert evs[7].json()["delta"]["stop_reason"] == "tool_use" and evs[7].json()["usage"] == {"output_tokens": 9}


def test_replay_stream_openai_with_tool_calls():
    resp = {"id": "c", "created": 1, "model": "m", "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
        "role": "assistant", "content": None, "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]}}]}
    first, last, done = replay_stream("openai", resp)
    delta = first.json()["choices"][0]["delta"]
    assert delta["content"] == "" and delta["tool_calls"][0]["index"] == 0 and delta["tool_calls"][0]["id"] == "t1"
    assert first.json()["object"] == "chat.completion.chunk" and last.json()["choices"][0]["finish_reason"] == "tool_calls"
    assert done.data == "[DONE]"
    _, last, _ = replay_stream("openai", {"choices": [{"message": {"content": "x"}}]})
    assert last.json()["choices"][0]["finish_reason"] == "stop"
