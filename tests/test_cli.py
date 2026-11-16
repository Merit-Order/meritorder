"""The `meritorder` command: offline subcommands directly, HTTP ones against a live gateway on a free port."""

from __future__ import annotations

import argparse
import asyncio
import copy
import csv
import json
import os
import re
import sys
import threading
import time
from types import SimpleNamespace

import httpx
import pytest
import uvicorn
import yaml
from conftest import ADMIN_KEY, TEST_CONFIG, TEST_KEY

from meritorder import catalog, cli, harness
from meritorder.app import build_gateway, create_app
from meritorder.state import MemoryState
from meritorder.store import MemoryStore

ENV_VARS = ("MERITORDER_DATABASE_URL", "MERITORDER_REDIS_URL", "MERITORDER_URL", "MERITORDER_API_KEY", "MERITORDER_WORKERS",
            "MERITORDER_CONFIG", "MERITORDER_PORT", "HF_TOKEN")


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """No .env, no saved credentials, no database, no real home directory: commands see only what a test sets."""
    monkeypatch.setenv("MERITORDER_ENV_FILE", str(tmp_path / "missing.env"))
    for var in ENV_VARS:
        monkeypatch.setenv(var, "")  # recorded, so anything a command sets is undone afterwards
        monkeypatch.delenv(var)
    home = tmp_path / "cfg"
    monkeypatch.setattr(harness, "CONFIG_DIR", home)
    monkeypatch.setattr(harness, "KEY_FILE", home / "api_key")
    monkeypatch.setattr(harness, "CREDENTIALS", home / "credentials.json")
    monkeypatch.setattr(cli, "KEY_FILE", home / "api_key")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(tmp_path / "pi"))


def write_config(tmp_path, raw: dict | None = None):
    path = tmp_path / "meritorder.yaml"
    path.write_text(yaml.safe_dump(raw or TEST_CONFIG))
    return path


@pytest.fixture
def pg_store(monkeypatch):
    """An in-memory store standing in for Postgres (these commands refuse anything else)."""
    store = MemoryStore()
    store.kind = "postgres"
    monkeypatch.setattr("meritorder.store.make_store", lambda url: store)
    return store


# ------------------------------------------------------------------ offline commands


def test_load_dotenv_keeps_existing_variables(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# comment\n\nMO_A=1\nMO_B=\"quoted\"\nMO_C='single'\nnot a pair\nMO_SET=from-file\nMO_EMPTY=\n")
    monkeypatch.setattr(cli.os, "environ", {"MO_SET": "already"})
    cli.load_dotenv(env)
    assert cli.os.environ == {"MO_SET": "already", "MO_A": "1", "MO_B": "quoted", "MO_C": "single"}
    cli.load_dotenv(tmp_path / "nope")  # a missing file is fine


def test_load_dotenv_reads_trailing_comments_like_docker_compose(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("MO_CAP=50          # all spend\nMO_Q=\"a # b\"  # note\nMO_URL=https://x.example/#frag\nMO_ONLY=   # nothing\n"
                   "MO_OPEN=\"unclosed\n")
    monkeypatch.setattr(cli.os, "environ", {})
    cli.load_dotenv(env)
    assert cli.os.environ == {"MO_CAP": "50", "MO_Q": "a # b", "MO_URL": "https://x.example/#frag", "MO_OPEN": "unclosed"}


def test_check_shows_usable_backends_and_models(tmp_path, capsys):
    raw = copy.deepcopy(TEST_CONFIG)
    raw["backends"]["openrouter"] = {"format": "openai", "base_url": "https://openrouter.ai/api/v1",
                                     "auth": {"type": "api_key", "value_env": "MO_TEST_UNSET_KEY"}}
    raw["backends"]["vertex-gemini-native"] = {"format": "openai", "api": "gemini", "flavor": "vertex", "project": "proj-1", "enabled": False}
    raw["models"]["glm"] = {"quality": {"general": 3}, "routes": [{"backend": "openrouter", "model": "z-ai/glm-4.6"}]}
    cli.main(["check", "--config", str(write_config(tmp_path, raw))])
    out = capsys.readouterr().out
    assert re.search(r"ON\s+mock-anthropic\s+anthropic", out)
    assert re.search(r"off\s+openrouter\s+openai\s+api\s+https://openrouter.ai/api/v1", out)  # no key: disabled
    assert re.search(r"off\s+vertex-gemini-native\s+gemini\s+api\s+Vertex AI, proj-1, global", out)
    assert re.search(r"glm .*routes: \(none available\)", out) and re.search(r"small .*routes: mock-openai", out)


def test_serve_hands_options_to_uvicorn(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: calls.append((a, kw)))
    path = str(write_config(tmp_path))
    cli.main(["--log-level", "warning", "serve", "--host", "0.0.0.0", "--port", "9123", "--config", path])
    (args, kw), = calls
    assert args == ("meritorder.app:create_app",) and kw["factory"] and kw["proxy_headers"]
    assert (kw["host"], kw["port"], kw["workers"], kw["log_level"]) == ("0.0.0.0", 9123, 1, "warning")
    assert os.environ["MERITORDER_CONFIG"] == path

    with pytest.raises(SystemExit, match="needs Postgres and Redis"):
        cli.main(["serve", "--workers", "2"])  # processes share state only through Postgres and Redis
    monkeypatch.setenv("MERITORDER_DATABASE_URL", "postgresql://db/mo")
    monkeypatch.setenv("MERITORDER_REDIS_URL", "redis://cache")
    cli.main(["serve", "--workers", "4"])
    monkeypatch.setenv("MERITORDER_WORKERS", "3")
    cli.main(["serve"])
    assert [kw["workers"] for _, kw in calls] == [1, 4, 3]
    assert [kw["port"] for _, kw in calls] == [9123, 8080, 8080]


def test_serve_listens_on_the_port_in_env(tmp_path, monkeypatch):
    """.env.example's MERITORDER_PORT is the port `meritorder serve` takes too, not only Docker Compose's host port."""
    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: calls.append(kw))
    env = tmp_path / ".env"
    env.write_text("MERITORDER_PORT=18080\n")
    monkeypatch.setenv("MERITORDER_ENV_FILE", str(env))
    cli.main(["serve"])
    cli.main(["serve", "--port", "9000"])
    assert [kw["port"] for kw in calls] == [18080, 9000]


def test_database_commands_need_postgres():
    with pytest.raises(SystemExit, match="needs Postgres"):
        cli.main(["migrate"])
    with pytest.raises(SystemExit, match="needs Postgres"):
        cli.main(["keys", "list"])


def test_migrate_and_key_lifecycle(pg_store, capsys):
    cli.main(["migrate"])
    assert "database is up to date" in capsys.readouterr().out

    cli.main(["keys", "create", "--user", "bob", "--team", "eng", "--role", "team_lead", "--name", "laptop",
              "--tag", "project=checkout", "--tag", "env=dev", "--expires-days", "30"])
    out = capsys.readouterr().out
    key_id = re.search(r"key id:\s+(\S+)", out).group(1)
    assert re.search(r"api key: \S+", out) and "not shown again" in out
    [key] = pg_store.keys.values()
    assert key["tags"] == {"project": "checkout", "env": "dev"} and key["expires_at"] is not None
    assert pg_store.users["bob"]["team"] == "eng"  # the user is created if missing

    cli.main(["keys", "list", "--user", "bob"])
    assert re.search(rf"{key_id}\s+bob\s+eng\s+team_lead\s+active\s+laptop", capsys.readouterr().out)
    cli.main(["keys", "revoke", key_id])
    cli.main(["keys", "revoke", key_id])
    assert capsys.readouterr().out.splitlines() == ["revoked", "no active key with that id"]
    cli.main(["keys", "create", "--user", "carol"])
    capsys.readouterr()
    [carol] = [k for k in pg_store.keys.values() if k["user_id"] == "carol"]
    carol["expires_at"] = carol["created_at"].replace(year=2000)
    cli.main(["keys", "list"])
    out = capsys.readouterr().out
    assert re.search(r"bob .* revoked", out) and re.search(r"carol .* expired", out)


def test_worker_and_controller_commands_pass_options(tmp_path, monkeypatch):
    seen = {}

    async def fake_worker(**kw):
        seen["worker"] = kw

    async def fake_controller(**kw):
        seen["controller"] = kw

    monkeypatch.setattr("meritorder.worker.run_worker", fake_worker)
    monkeypatch.setattr("meritorder.controller.run_controller", fake_controller)
    path = str(write_config(tmp_path))
    cli.main(["worker", "--once", "--backfill", "--interval", "5", "--config", path, "--metrics-port", "9464"])
    cli.main(["controller", "--once", "--config", path])
    assert seen["worker"] == {"once": True, "interval": 5, "path": path, "backfill": True, "metrics_port": 9464}
    assert seen["controller"] == {"once": True, "interval": 60, "path": path}


def test_prices_refresh_writes_openrouter_prices(tmp_path, monkeypatch, capsys):
    raw = copy.deepcopy(TEST_CONFIG)
    raw["backends"]["openrouter"] = {"format": "openai", "base_url": "https://openrouter.ai/api/v1",
                                     "auth": {"type": "api_key", "value_env": "MO_TEST_UNSET_KEY"}}
    raw["models"]["glm"] = {"quality": {"general": 3}, "routes": [{"backend": "openrouter", "model": "z-ai/glm-4.6"}]}
    raw["models"]["ghost"] = {"quality": {"general": 3}, "routes": [{"backend": "openrouter", "model": "nobody/ghost"}]}
    models = {"data": [{"id": "z-ai/glm-4.6", "context_length": 200000, "supported_parameters": ["tools", "reasoning"],
                        "pricing": {"prompt": "0.0000006", "completion": "0.0000022", "input_cache_read": "0.00000011",
                                    "input_cache_write": "-1"}}]}
    fetched = []

    def fake_get(url, timeout):
        fetched.append(url)
        return httpx.Response(200, json=models if url == catalog.OPENROUTER_MODELS else {"data": []}, request=httpx.Request("GET", url))

    monkeypatch.setattr(catalog.httpx, "get", fake_get)
    from meritorder import quality_data
    monkeypatch.setattr(quality_data, "fetch_epoch", lambda: {"eci": {}, "difficulty": {}, "results": {}})
    cli.main(["prices", "refresh", "--config", str(write_config(tmp_path, raw))])
    out = capsys.readouterr().out
    assert "Wrote 1 route prices" in out and "ghost@openrouter (nobody/ghost)" in out
    assert fetched == [catalog.OPENROUTER_MODELS, catalog.OPENROUTER_EMBEDDINGS]
    data = json.loads((tmp_path / "prices.json").read_text())
    assert data["prices"] == {"openrouter:z-ai/glm-4.6": {"input": 0.6, "output": 2.2, "cache_read": 0.11, "source": "openrouter"}}
    assert data["context"] == {"glm": 200000} and data["reasoning"] == ["glm"]


def test_models_discover_filters_and_limits(tmp_path, monkeypatch, capsys):
    async def fake_discover(cfg, backends):
        assert set(backends) == {"mock-openai"}
        return {"mock-openai": ["a-small", "b-big", "c-small"], "broken": "HTTP 401: bad key"}

    monkeypatch.setattr(catalog, "discover", fake_discover)
    cli.main(["models", "discover", "--config", str(write_config(tmp_path)), "--backend", "mock-openai", "--grep", "SMALL", "--limit", "1"])
    assert capsys.readouterr().out.splitlines() == ["mock-openai: 2 models", "  a-small", "broken: HTTP 401: bad key"]


async def test_discover_asks_each_backend(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "openai.test":
            assert request.url.path == "/v1/models" and request.headers["authorization"] == "Bearer t"
            return httpx.Response(200, json={"data": [{"id": "gpt-x"}, {"id": "gpt-a"}, {"name": "no id"}]})
        if request.url.host == "gemini.test":  # the Gemini API's own listing
            assert request.url.path == "/v1beta/models" and request.url.params["pageSize"] == "1000"
            return httpx.Response(200, json={"models": [{"name": "models/gemini-4-carbon"}, {"name": "models/gemini-3.8-flash"}, {}]})
        assert request.url.path == "/v1/models" and request.headers["anthropic-version"] == "2023-06-01"
        return httpx.Response(401, text="bad key")

    real = httpx.AsyncClient
    monkeypatch.setattr(catalog.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))

    class Auth:
        def __init__(self, fail: bool = False):
            self.fail = fail

        async def apply(self, method, url, headers, body):
            if self.fail:
                raise RuntimeError("no credentials")
            return {**headers, "authorization": "Bearer t"}

    def b(kind, fmt, url, flavor="standard", fail=False, api="chat"):
        return SimpleNamespace(kind=kind, format=fmt, cfg=SimpleNamespace(base_url=url, flavor=flavor, api=api), auth=Auth(fail))

    out = await catalog.discover(None, {"oa": b("api", "openai", "https://openai.test/v1/"), "an": b("api", "anthropic", "https://anthropic.test"),
                                        "mock": b("mock", "openai", ""), "vx": b("api", "anthropic", "https://vx.test", flavor="vertex"),
                                        "bad": b("api", "openai", "https://bad.test", fail=True),
                                        "gm": b("api", "openai", "https://gemini.test/v1beta", api="gemini")})
    assert out == {"oa": ["gpt-a", "gpt-x"], "an": "HTTP 401: bad key", "bad": "RuntimeError: no credentials",
                   "gm": ["gemini-3.8-flash", "gemini-4-carbon"]}


def test_use_key_token_and_gateway_defaults(monkeypatch, capsys):
    with pytest.raises(SystemExit, match="no API key"):
        cli.main(["token"])
    assert cli._gateway(argparse.Namespace()) == "http://localhost:8080"
    cli.main(["use-key", "mo-saved", "--gateway", "http://gw.example:8080/"])
    assert harness.KEY_FILE.read_text().strip() == "mo-saved" and oct(harness.KEY_FILE.stat().st_mode)[-3:] == "600"
    assert cli._gateway(argparse.Namespace()) == "http://gw.example:8080"  # from the saved credentials
    cli.main(["token"])
    monkeypatch.setenv("MERITORDER_API_KEY", "mo-env")
    cli.main(["token"])
    cli.main(["token", "--key", "mo-flag"])
    assert capsys.readouterr().out.splitlines()[-3:] == ["mo-saved", "mo-env", "mo-flag"]


def test_login_device_flow_saves_the_key(monkeypatch, capsys):
    polls = iter([{"error": "authorization_pending"}, {"error": "slow_down"}, {"id_token": "idt"}])
    sleeps = []
    install_idp(monkeypatch, lambda: next(polls))
    monkeypatch.setattr(cli.time, "sleep", sleeps.append)
    cli.main(["login", "--gateway", "http://gw.test/"])
    out = capsys.readouterr().out
    assert "Open https://idp.test/activate and enter code ABCD" in out
    assert "Logged in as alice (team eng); key expires 2026-12-09." in out
    assert sleeps == [1, 1, 6]  # slow_down adds 5 seconds
    assert harness.KEY_FILE.read_text().strip() == "mo-new"
    creds = json.loads(harness.CREDENTIALS.read_text())
    assert creds["gateway"] == "http://gw.test" and creds["user"] == "alice" and "api_key" in creds


@pytest.mark.parametrize("case, message", [
    ("no-oidc", "no OIDC login configured"),
    ("denied", "login failed: user said no"),
    ("timeout", "login timed out"),
    ("exchange", "key exchange failed: 403"),
])
def test_login_failures(monkeypatch, case, message):
    install_idp(monkeypatch, lambda: {"error": "access_denied", "error_description": "user said no"} if case == "denied" else {"id_token": "idt"},
                oidc=case != "no-oidc", expires_in=0 if case == "timeout" else 60, exchange_status=403 if case == "exchange" else 200)
    monkeypatch.setattr(cli.time, "sleep", lambda s: None)
    with pytest.raises(SystemExit, match=message):
        cli.main(["login", "--gateway", "http://gw.test"])


def test_login_with_google_device_flow(monkeypatch, capsys):
    """Google's device flow differs from RFC 8628 in small ways: `verification_url`, no `message`, the (public) client secret
    on every poll, and 428/403 statuses while waiting."""
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        url, form = str(request.url), dict(httpx.QueryParams(request.content.decode()))
        seen.append(url)
        if url == "http://gw.test/auth/config":
            return httpx.Response(200, json={"oidc": {"issuer": "https://accounts.google.com", "client_id": "cid.apps.googleusercontent.com",
                                                      "scopes": ["openid", "email", "profile"], "client_secret": "GOCSPX-public"}})
        if url == "https://accounts.google.com/.well-known/openid-configuration":
            return httpx.Response(200, json={"issuer": "https://accounts.google.com", "token_endpoint": "https://oauth2.googleapis.com/token",
                                             "device_authorization_endpoint": "https://oauth2.googleapis.com/device/code"})
        if url == "https://oauth2.googleapis.com/device/code":
            assert form == {"client_id": "cid.apps.googleusercontent.com", "scope": "openid email profile"}  # no secret here
            return httpx.Response(200, json={"device_code": "4/4-GMMh", "user_code": "GQVQ-JKEC", "verification_url": "https://www.google.com/device",
                                             "expires_in": 1800, "interval": 5})
        if url == "https://oauth2.googleapis.com/token":
            assert form == {"grant_type": "urn:ietf:params:oauth:grant-type:device_code", "device_code": "4/4-GMMh",
                            "client_id": "cid.apps.googleusercontent.com", "client_secret": "GOCSPX-public"}
            polls = seen.count(url)
            if polls == 1:
                return httpx.Response(428, json={"error": "authorization_pending", "error_description": "Precondition Required"})
            if polls == 2:
                return httpx.Response(403, json={"error": "slow_down", "error_description": "Forbidden"})
            return httpx.Response(200, json={"access_token": "ya29.x", "id_token": "google-idt", "expires_in": 3599, "token_type": "Bearer"})
        if url == "http://gw.test/auth/exchange":
            assert json.loads(request.content) == {"id_token": "google-idt"}
            return httpx.Response(200, json={"api_key": "mo-g", "user": "alice@example.com", "team": "eng", "expires_at": "2026-12-09T00:00:00Z"})
        raise AssertionError(f"unexpected request {url}")

    real = httpx.Client
    monkeypatch.setattr(cli.httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    sleeps = []
    monkeypatch.setattr(cli.time, "sleep", sleeps.append)
    cli.main(["login", "--gateway", "http://gw.test"])
    out = capsys.readouterr().out
    assert "Open https://www.google.com/device and enter code GQVQ-JKEC" in out and "Logged in as alice@example.com (team eng)" in out
    assert sleeps == [5, 5, 10] and harness.KEY_FILE.read_text().strip() == "mo-g"


@pytest.mark.parametrize("discovery, device, message", [
    ({"token_endpoint": "https://idp.test/token"}, None, "doesn't offer device login"),
    ({"token_endpoint": "https://idp.test/token", "device_authorization_endpoint": "https://idp.test/device"},
     {"error": "invalid_scope", "error_description": "Some requested scopes were invalid."}, "login failed: Some requested scopes were invalid."),
])
def test_login_when_the_provider_cant_start_device_login(monkeypatch, discovery, device, message):
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == "http://gw.test/auth/config":
            return httpx.Response(200, json={"oidc": {"issuer": "https://idp.test", "client_id": "cli", "scopes": ["openid", "offline_access"]}})
        if url == "https://idp.test/.well-known/openid-configuration":
            return httpx.Response(200, json=discovery)
        if url == "https://idp.test/device":
            return httpx.Response(400, json=device)
        raise AssertionError(f"unexpected request {url}")

    real = httpx.Client
    monkeypatch.setattr(cli.httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    with pytest.raises(SystemExit, match=message):
        cli.main(["login", "--gateway", "http://gw.test"])


def install_idp(monkeypatch, poll, oidc=True, expires_in=60, exchange_status=200):
    """A fake gateway + identity provider for the OIDC device-code flow."""
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == "http://gw.test/auth/config":
            return httpx.Response(200, json={"oidc": {"issuer": "https://idp.test/", "client_id": "cli", "scopes": ["openid", "email"]} if oidc else None})
        if url == "https://idp.test/.well-known/openid-configuration":
            return httpx.Response(200, json={"device_authorization_endpoint": "https://idp.test/device", "token_endpoint": "https://idp.test/token"})
        if url == "https://idp.test/device":
            assert b"client_id=cli" in request.content and b"scope=openid+email" in request.content
            return httpx.Response(200, json={"device_code": "dc", "user_code": "ABCD", "verification_uri": "https://idp.test/activate",
                                             "interval": 1, "expires_in": expires_in})
        if url == "https://idp.test/token":
            assert b"device_code=dc" in request.content
            return httpx.Response(200, json=poll())
        if url == "http://gw.test/auth/exchange":
            assert json.loads(request.content) == {"id_token": "idt"}
            if exchange_status != 200:
                return httpx.Response(exchange_status, text="not in a team")
            return httpx.Response(200, json={"api_key": "mo-new", "user": "alice", "team": "eng", "expires_at": "2026-12-09T00:00:00Z"})
        raise AssertionError(f"unexpected request {url}")

    real = httpx.Client
    monkeypatch.setattr(cli.httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))


# ------------------------------------------------------------------ commands that talk to a gateway


class LiveGateway:
    """A real gateway (in-memory storage, mock backends) served by uvicorn on a free port, in a thread."""

    def __init__(self, cfg):
        self.gw = build_gateway(cfg, store=MemoryStore(), state=MemoryState())
        self.gw.file_raw = copy.deepcopy(TEST_CONFIG)
        self.server = uvicorn.Server(uvicorn.Config(create_app(gateway=self.gw), host="127.0.0.1", port=0, log_level="warning", lifespan="off"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self):
        self.thread.start()
        deadline = time.monotonic() + 10
        while not self.server.started:
            assert time.monotonic() < deadline, "gateway did not start"
            time.sleep(0.02)
        self.url = f"http://127.0.0.1:{self.server.servers[0].sockets[0].getsockname()[1]}"
        return self

    def __exit__(self, *exc):
        self.server.should_exit = True
        self.thread.join(10)

    def chat(self, n: int = 1, key: str = TEST_KEY, model: str = "small") -> None:
        before = len(self.gw.store.ledger)
        for _ in range(n):
            r = httpx.post(f"{self.url}/v1/chat/completions", headers={"authorization": f"Bearer {key}"},
                           json={"model": model, "messages": [{"role": "user", "content": "hi"}]}, timeout=30)
            assert r.status_code == 200, r.text
        deadline = time.monotonic() + 5
        while len(self.gw.store.ledger) < before + n:  # ledger rows are written just after the response
            assert time.monotonic() < deadline
            time.sleep(0.01)


@pytest.fixture
def live(cfg):
    with LiveGateway(cfg) as g:
        yield g


def test_usage_shows_spend_and_savings(live, capsys):
    live.chat(2)
    cli.main(["usage", "--gateway", live.url, "--key", TEST_KEY])
    out = capsys.readouterr().out
    assert out.startswith("alice (default)") and "of $10.00" in out and "across 2 requests" in out
    assert re.search(r"small\s+2\s", out)  # the by-model table
    with pytest.raises(SystemExit, match="^401"):
        cli.main(["usage", "--gateway", live.url, "--key", "mo-wrong"])


def test_eval_records_the_run_and_takes_filters_and_judges(live, tmp_path, capsys):
    suite = tmp_path / "suite.yaml"
    suite.write_text(yaml.safe_dump({"name": "mixed", "tasks": [
        {"id": "g1", "task": "general", "level": 1, "prompt": "Say OK", "checks": [{"contains": "OK"}]},
        {"id": "g2", "task": "general", "level": 2, "prompt": "Say hi", "checks": [{"contains": "hi"}]},
        {"id": "j3", "task": "general", "level": 3, "kind": "judged", "prompt": "Write a haiku", "rubric": ["Has three lines"]},
        {"id": "c3", "task": "coding", "level": 3, "kind": "code", "prompt": "Write f", "code": {"tests_source": "def test_x():\n    pass\n"}},
    ]}))
    with pytest.raises(SystemExit, match="no tasks match --only"):
        cli.main(["eval", "--suite", str(suite), "--gateway", live.url, "--key", ADMIN_KEY, "--models", "small", "--only", "zzz", "--yes"])
    cli.main(["eval", "--suite", str(suite), "--gateway", live.url, "--key", ADMIN_KEY, "--models", "small", "--only", "g*,j*,c*",
              "--judges", "big", "--sandbox", "none", "--yes"])
    out = capsys.readouterr().out
    assert "Judges for 1 open-ended tasks: big" in out and "Recorded as run ev_" in out
    assert "2 task runs were skipped" in out  # the code task (no sandbox), and the judged one (the mock judge gives no verdict)
    runs = asyncio.run(live.gw.store.list_eval_runs())
    assert runs[0]["executor"] == "cli" and runs[0]["status"] == "done" and runs[0]["progress"]["done"] == 4


def test_setup_pi_and_claude_code(live, tmp_path, capsys):
    cli.main(["setup", "pi", "--gateway", live.url, "--key", TEST_KEY])
    provider = json.loads((tmp_path / "pi" / "models.json").read_text())["providers"]["meritorder"]
    ids = [m["id"] for m in provider["models"]]
    assert provider["baseUrl"] == f"{live.url}/v1" and "auto" in ids and "small" in ids and "embed-small" not in ids
    assert harness.KEY_FILE.read_text().strip() == TEST_KEY  # saved on first setup
    cli.main(["setup", "claude-code", "--gateway", live.url, "--key", TEST_KEY])
    settings = json.loads((tmp_path / "home" / ".claude" / "settings.json").read_text())
    assert settings["env"]["ANTHROPIC_BASE_URL"] == live.url and str(harness.KEY_FILE) in settings["apiKeyHelper"]
    out = capsys.readouterr().out
    assert "Wrote the 'meritorder' provider" in out and "ANTHROPIC_BASE_URL" in out and "separate profile" in out
    with pytest.raises(SystemExit, match="could not list models"):
        cli.main(["setup", "pi", "--gateway", live.url, "--key", "mo-wrong"])


def test_setup_claude_code_starts_on_auto_with_the_gateways_window(live, tmp_path, monkeypatch, capsys):
    """Claude Code doesn't know `auto`: it would assume 200K tokens and leave it out of /model. Setup says both."""
    live.gw.cfg.model("small").context_window = 262_144
    live.gw.cfg.model("big").context_window = 1_000_000  # auto's window: the largest the team can use
    cli.main(["setup", "claude-code", "--gateway", live.url, "--key", TEST_KEY])
    settings = json.loads((tmp_path / "home" / ".claude" / "settings.json").read_text())
    env = settings["env"]
    assert settings["model"] == "auto" and env["ANTHROPIC_CUSTOM_MODEL_OPTION"] == "auto"
    assert env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "1000000" and env["CLAUDE_CODE_GATEWAY_HINT_HEADERS"] == "1"
    assert "1,000,000-token context window" in capsys.readouterr().out

    # A named non-Claude model gets its own window.
    cli.main(["setup", "claude-code", "--gateway", live.url, "--key", TEST_KEY, "--model", "small"])
    assert json.loads((tmp_path / "home" / ".claude" / "settings.json").read_text())["env"]["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "262144"

    # Its own Claude names: Claude Code knows their windows, so nothing extra is set (and earlier extras go).
    cli.main(["setup", "claude-code", "--gateway", live.url, "--key", TEST_KEY, "--model", "claude-sonnet-5-5"])
    settings = json.loads((tmp_path / "home" / ".claude" / "settings.json").read_text())
    assert settings["model"] == "claude-sonnet-5-5"
    assert not {"ANTHROPIC_CUSTOM_MODEL_OPTION", "CLAUDE_CODE_MAX_CONTEXT_TOKENS"} & set(settings["env"])

    # A separate profile, chosen by flag or by CLAUDE_CONFIG_DIR; `none` leaves the model alone.
    profile = tmp_path / "profile"
    cli.main(["setup", "claude-code", "--gateway", live.url, "--key", TEST_KEY, "--config-dir", str(profile), "--model", "none"])
    settings = json.loads((profile / "settings.json").read_text())
    assert "model" not in settings and settings["env"]["ANTHROPIC_BASE_URL"] == live.url
    assert f"CLAUDE_CONFIG_DIR={profile}" in capsys.readouterr().out
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "from-env"))
    cli.main(["setup", "claude-code", "--gateway", live.url, "--key", TEST_KEY])
    assert json.loads((tmp_path / "from-env" / "settings.json").read_text())["model"] == "auto"

    with pytest.raises(SystemExit, match="isn't a model this gateway serves"):
        cli.main(["setup", "claude-code", "--gateway", live.url, "--key", TEST_KEY, "--model", "nope"])


def test_setup_codex(live, tmp_path, monkeypatch, capsys):
    """Codex CLI gets a Merit Order provider over WebSocket; the rest of its config.toml stays as it was."""
    import tomllib

    live.gw.cfg.model("big").context_window = 1_000_000
    path = tmp_path / "home" / ".codex" / "config.toml"
    path.parent.mkdir(parents=True)
    path.write_text('model = "gpt-6.1-sol"\napproval_policy = "on-request"\n\n[model_providers.meritorder]\nname = "old"\n'
                    '[model_providers.meritorder.auth]\ncommand = "old"\n\n[profiles.fast]\nmodel = "x"\n')
    cli.main(["setup", "codex", "--gateway", live.url + "/", "--key", TEST_KEY])
    data = tomllib.loads(path.read_text())
    assert data["model_provider"] == "meritorder" and data["model"] == "auto" and data["model_context_window"] == 1_000_000
    assert data["approval_policy"] == "on-request" and data["profiles"] == {"fast": {"model": "x"}}
    provider = data["model_providers"]["meritorder"]
    assert provider == {"name": "Merit Order", "base_url": f"{live.url}/v1", "wire_api": "responses", "supports_websockets": True,
                        "auth": {"command": "cat", "args": [str(harness.KEY_FILE)]}}
    assert path.with_suffix(".toml.bak").read_text().startswith('model = "gpt-6.1-sol"')
    out = capsys.readouterr().out
    assert "over WebSocket" in out and "1,000,000-token context window" in out

    # Its own model names, in a separate folder chosen by flag or by CODEX_HOME.
    other = tmp_path / "codex-mo"
    cli.main(["setup", "codex", "--gateway", live.url, "--key", TEST_KEY, "--model", "none", "--config-dir", str(other)])
    data = tomllib.loads((other / "config.toml").read_text())
    assert "model" not in data and data["model_provider"] == "meritorder"
    assert f"CODEX_HOME={other} codex" in capsys.readouterr().out
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "from-env"))
    cli.main(["setup", "codex", "--gateway", live.url, "--key", TEST_KEY, "--model", "small"])
    assert tomllib.loads((tmp_path / "from-env" / "config.toml").read_text())["model"] == "small"
    with pytest.raises(SystemExit, match="isn't a model this gateway serves"):
        cli.main(["setup", "codex", "--gateway", live.url, "--key", TEST_KEY, "--model", "nope"])


def test_export_to_jsonl_csv_and_s3(live, tmp_path, monkeypatch, capsys):
    live.chat(3)
    jsonl = tmp_path / "rows.jsonl"
    cli.main(["export", "--out", str(jsonl), "--gateway", live.url, "--key", ADMIN_KEY])
    rows = [json.loads(line) for line in jsonl.read_text().splitlines()]
    assert len(rows) == 3 and {r["user_id"] for r in rows} == {"alice"} and "exported 3 requests" in capsys.readouterr().out

    as_csv = tmp_path / "rows.csv"
    cli.main(["export", "--format", "csv", "--out", str(as_csv), "--gateway", live.url, "--key", ADMIN_KEY])
    with open(as_csv) as f:
        table = list(csv.DictReader(f))
    assert len(table) == 3 and isinstance(json.loads(table[0]["tags"]), dict)

    puts = fake_s3(monkeypatch)
    monkeypatch.chdir(tmp_path)  # the staging file goes in the working directory
    cli.main(["export", "--out", "s3://bucket/ledger/2026-11.jsonl", "--gateway", live.url, "--key", ADMIN_KEY])
    [put] = puts
    assert (put["Bucket"], put["Key"], put["ContentType"]) == ("bucket", "ledger/2026-11.jsonl", "application/x-ndjson")
    assert put["Body"].count(b"\n") == 3 and not list(tmp_path.glob(".meritorder-export-*"))

    with pytest.raises(SystemExit, match="^403"):
        cli.main(["export", "--out", str(tmp_path / "denied.jsonl"), "--gateway", live.url, "--key", TEST_KEY])
    assert not (tmp_path / "denied.jsonl").exists()


def fake_s3(monkeypatch) -> list[dict]:
    """botocore's S3 client, recording each put_object with the body it read (it's given an open file, not bytes)."""
    pytest.importorskip("botocore.session")
    puts = []

    class FakeSession:
        def create_client(self, name, endpoint_url=None):
            assert name == "s3"
            return SimpleNamespace(put_object=lambda **kw: puts.append({**kw, "Body": kw["Body"].read()}))

    monkeypatch.setattr("botocore.session.Session", FakeSession)
    return puts


def test_export_to_parquet_with_typed_columns_and_row_groups(live, tmp_path, monkeypatch, capsys):
    pq = pytest.importorskip("pyarrow.parquet")
    import pyarrow as pa

    from meritorder import export
    from meritorder.store import LEDGER_FIELDS

    live.chat(3)
    asyncio.run(live.gw.store.record({"user_id": "bob", "team": "default", "tags": {"project": "checkout", "n": 7},
                                      "served_model": "small", "status": 200, "actual_cost": 1}))  # tag values that aren't text
    # 4 rows, packed 2 at a time: a full row group of 3 (from two batches), then the last row when the file closes.
    monkeypatch.setattr(export, "ROW_GROUP_ROWS", 3)
    monkeypatch.setattr(export, "_BATCH_ROWS", 2)
    out = tmp_path / "ledger.parquet"
    cli.main(["export", "--format", "parquet", "--out", str(out), "--gateway", live.url, "--key", ADMIN_KEY])
    assert "exported 4 requests" in capsys.readouterr().out
    f = pq.ParquetFile(out)
    assert f.metadata.num_rows == 4 and [f.metadata.row_group(i).num_rows for i in range(f.num_row_groups)] == [3, 1]
    schema = f.schema_arrow
    assert schema.names == LEDGER_FIELDS
    assert schema.field("ts").type == pa.timestamp("us", tz="UTC") and schema.field("tags").type == pa.map_(pa.string(), pa.string())
    assert (schema.field("input_tokens").type, schema.field("status").type, schema.field("actual_cost").type,
            schema.field("upgraded").type, schema.field("id").type) == (pa.int64(), pa.int32(), pa.float64(), pa.bool_(), pa.string())
    rows = f.read().to_pylist()
    assert {r["user_id"] for r in rows} == {"alice", "bob"} and all(r["ts"].tzinfo is not None for r in rows)
    assert rows[-1]["tags"] == [("project", "checkout"), ("n", "7")] and rows[-1]["actual_cost"] == 1.0
    assert rows[0]["input_tokens"] > 0 and isinstance(rows[0]["actual_cost"], float)

    # The same time ranges as the other formats; an empty range is still a valid file with every column.
    empty = tmp_path / "empty.parquet"
    cli.main(["export", "--format", "parquet", "--start", "2020-01-01", "--end", "2020-02-01", "--out", str(empty),
              "--gateway", live.url, "--key", ADMIN_KEY])
    assert pq.ParquetFile(empty).metadata.num_rows == 0 and pq.read_schema(empty).names == LEDGER_FIELDS

    # And the same S3 target.
    puts = fake_s3(monkeypatch)
    monkeypatch.chdir(tmp_path)
    cli.main(["export", "--format", "parquet", "--out", "s3://bucket/ledger/2026-11.parquet", "--gateway", live.url, "--key", ADMIN_KEY])
    [put] = puts
    assert (put["Key"], put["ContentType"]) == ("ledger/2026-11.parquet", "application/vnd.apache.parquet")
    assert put["Body"][:4] == put["Body"][-4:] == b"PAR1" and not list(tmp_path.glob(".meritorder-export-*"))


def test_parquet_export_without_pyarrow_says_how_to_get_it(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "pyarrow", None)  # import pyarrow now raises ImportError
    with pytest.raises(SystemExit, match=r"needs pyarrow, which the image doesn't include: .*uv sync --extra parquet"):
        cli.main(["export", "--format", "parquet", "--out", str(tmp_path / "x.parquet"), "--gateway", "http://gw.invalid", "--key", "k"])
    assert not (tmp_path / "x.parquet").exists()


def test_parquet_writer_notes_unknown_columns_and_cleans_up_a_failed_export(tmp_path, monkeypatch, capsys):
    pq = pytest.importorskip("pyarrow.parquet")
    from meritorder.export import LedgerWriter

    w = LedgerWriter("parquet", tmp_path / "a.parquet")
    w.write(json.dumps({"id": "x", "ts": "2026-11-12 10:00:00.123456+00:00", "new_column": 1, "tags": None}))
    w.close()
    w.close()  # twice is fine
    assert "doesn't know the ledger columns new_column" in capsys.readouterr().err
    [row] = pq.read_table(tmp_path / "a.parquet").to_pylist()
    assert row["ts"].microsecond == 123456 and row["tags"] == [] and row["input_tokens"] is None

    # The download breaks part way: the half-written file is removed.
    def stream(*a, **kw):
        class Resp:
            status_code = 200

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def iter_lines(self):
                yield json.dumps({"id": "1", "ts": "2026-11-12T10:00:00+00:00"})
                raise httpx.ReadError("connection reset")
        return Resp()

    monkeypatch.setattr(cli.httpx, "stream", stream)
    out = tmp_path / "broken.parquet"
    with pytest.raises(httpx.ReadError):
        cli.main(["export", "--format", "parquet", "--out", str(out), "--gateway", "http://gw.invalid", "--key", "k"])
    assert not out.exists()


def test_eval_runs_a_suite_and_saves_scores(live, tmp_path, monkeypatch, capsys):
    suite = tmp_path / "suite.yaml"
    suite.write_text(yaml.safe_dump({"name": "tiny", "tasks": [
        {"id": "g1", "task": "general", "level": 1, "prompt": "Say OK", "checks": [{"contains": "OK"}]},
        {"id": "c1", "task": "coding", "level": 1, "prompt": "Write a function", "checks": [{"regex": "def "}]},
    ]}))
    common = ["--suite", str(suite), "--gateway", live.url]
    with pytest.raises(SystemExit, match="needs an admin key"):
        cli.main(["eval", *common, "--key", TEST_KEY, "--yes"])
    with pytest.raises(SystemExit, match="no live route for: nope"):
        cli.main(["eval", *common, "--key", ADMIN_KEY, "--models", "nope", "--yes"])
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")
    with pytest.raises(SystemExit, match="cancelled"):
        cli.main(["eval", *common, "--key", ADMIN_KEY])
    # An expensive run needs "yes" typed out: "y" isn't enough. The warning says how rough the estimate is, and the cap.
    from meritorder.evals import jobs

    real = jobs.estimate
    monkeypatch.setattr(jobs, "estimate", lambda *a, **k: {**real(*a, **k), "usd": 250.0, "by_model": {"big": 200.0, "small": 50.0}})
    prompts = []
    monkeypatch.setattr("builtins.input", lambda prompt="": prompts.append(prompt) or "y")
    capsys.readouterr()
    with pytest.raises(SystemExit, match="cancelled"):
        cli.main(["eval", *common, "--key", ADMIN_KEY])
    out = capsys.readouterr().out
    assert "Estimated cost: about $250.00 (a rough guide: real runs have cost up to three times as much)." in out
    assert "By model: big $200.00, small $50.00." in out and "Spending cap: $375.00." in out
    assert prompts == ["This spends real money: about $250.00, at most $375.00. Type yes to run it: "]
    monkeypatch.setattr(jobs, "estimate", real)

    results = tmp_path / "results.json"
    cli.main(["eval", *common, "--key", ADMIN_KEY, "--yes", "--max-cost", "2", "--out", str(results), "--apply"])
    out = capsys.readouterr().out
    assert "Spending cap: $2.00." in out  # shown with --yes too
    run = asyncio.run(live.gw.store.list_eval_runs())[0]
    assert run["request"]["max_cost_usd"] == 2.0
    assert "Suite 'tiny': 2 tasks x 1 on 2 models: big, small" in out  # every live chat model; embeddings skipped
    assert "Results saved" in out and re.search(r"Applied: config version|Not applied: \d+", out)
    saved = json.loads(results.read_text())
    assert saved["suite"] == "tiny" and set(saved["models"]) == {"big", "small"}
    stored = asyncio.run(live.gw.store.get_setting("evals"))
    assert set(stored["models"]) == {"big", "small"}  # the console shows the latest run
