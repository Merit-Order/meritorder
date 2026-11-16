"""Shared fixtures.

Tests run against in-memory state and storage. Set MERITORDER_TEST_REDIS_URL and
MERITORDER_TEST_DATABASE_URL (CI does) and every test that uses the ``gateway``, ``store`` or
``state`` fixtures also runs against real Redis and Postgres. Both are wiped before each test, so
point them at a throwaway Redis database and Postgres database.
"""

from __future__ import annotations

import copy
import os

import httpx
import pytest
import yaml

from meritorder.app import build_gateway, create_app
from meritorder.config import Config, load_config
from meritorder.state import MemoryState, RedisState
from meritorder.store import MemoryStore, PostgresStore

REDIS_URL = os.environ.get("MERITORDER_TEST_REDIS_URL")
DATABASE_URL = os.environ.get("MERITORDER_TEST_DATABASE_URL")
BACKINGS = ["memory", pytest.param("real", marks=pytest.mark.skipif(not (REDIS_URL and DATABASE_URL),
                                                                    reason="set MERITORDER_TEST_REDIS_URL and MERITORDER_TEST_DATABASE_URL"))]

TEST_KEY = "mo-test-key"
LEAD_KEY = "mo-lead-key"
ADMIN_KEY = "mo-admin-key"

TEST_CONFIG = {
    "baseline_model": "big",
    "backends": {
        "mock-anthropic": {"kind": "mock", "format": "anthropic", "mock_latency_ms": 0},
        "mock-openai": {"kind": "mock", "format": "openai", "mock_latency_ms": 0},
    },
    "models": {
        "big": {
            "aliases": ["claude-big-*"],
            "quality": {"coding": 5, "general": 5},
            "routes": [{"backend": "mock-anthropic", "model": "big-1", "price": {"input": 3, "output": 15, "cache_read": 0.3, "cache_write": 3.75}}],
        },
        "embed-small": {
            "kind": "embedding",
            "routes": [{"backend": "mock-openai", "model": "embed-small-1", "price": {"input": 0.02, "output": 0}}],
        },
        "small": {
            "quality": {"coding": 3, "general": 3},
            "routes": [{"backend": "mock-openai", "model": "small-1", "price": {"input": 0.2, "output": 1.0, "cache_read": 0.02}}],
        },
    },
    "teams": {"default": {"budget_usd_month": 100, "user_budget_usd_month": 10}},
    "auth": {
        "static_keys": [
            {"key": TEST_KEY, "user": "alice", "team": "default", "role": "user"},
            {"key": LEAD_KEY, "user": "lee", "team": "default", "role": "team_lead"},
            {"key": ADMIN_KEY, "user": "root", "team": "ops", "role": "admin"},
        ]
    },
}


@pytest.fixture
def cfg(tmp_path) -> Config:
    path = tmp_path / "meritorder.yaml"
    path.write_text(yaml.safe_dump(TEST_CONFIG))
    return load_config(path)


async def real_store() -> PostgresStore:
    store = PostgresStore(DATABASE_URL)
    await store.init()
    await store.migrate()
    async with store.pool.acquire() as c:
        tables = [r["tablename"] for r in await c.fetch(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public' AND tablename <> 'schema_migrations' "
            "AND tablename NOT IN (SELECT inhrelid::regclass::text FROM pg_inherits)")]
        if tables:
            await c.execute(f"TRUNCATE {', '.join(tables)} RESTART IDENTITY CASCADE")
    return store


async def real_state() -> RedisState:
    state = RedisState(REDIS_URL)
    await state.r.flushdb()
    return state


@pytest.fixture(params=BACKINGS)
async def store(request):
    s = MemoryStore() if request.param == "memory" else await real_store()
    yield s
    await s.close()


@pytest.fixture(params=BACKINGS)
async def state(request):
    s = MemoryState() if request.param == "memory" else await real_state()
    yield s
    await s.close()


@pytest.fixture(params=BACKINGS)
async def gateway(cfg, request):
    if request.param == "memory":
        store, state = MemoryStore(), MemoryState()
    else:
        store, state = await real_store(), await real_state()
    gw = build_gateway(cfg, store=store, state=state)
    gw.file_raw = copy.deepcopy(TEST_CONFIG)
    yield gw
    await gw.drain()
    await state.close()
    await store.close()


def ledger_row(**kw) -> dict:
    """A complete ledger row as the gateway writes one, with overrides."""
    import uuid
    from datetime import UTC, datetime

    from meritorder.store import complete_row

    return complete_row({"id": uuid.uuid4(), "ts": datetime.now(UTC), "user_id": "alice", "team": "default",
                         "status": 200, "backend": "mock", "served_model": "big", **kw})


async def ledger(gateway) -> list[dict]:
    """Every ledger row, oldest first, on either store."""
    await gateway.drain()
    return (await gateway.store.recent(100_000))[::-1]


async def last_row(gateway) -> dict:
    """The newest ledger row, on either store."""
    await gateway.drain()
    return (await gateway.store.recent(1))[0]


@pytest.fixture
async def client(gateway):
    app = create_app(gateway=gateway)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def auth(key: str = TEST_KEY) -> dict:
    return {"authorization": f"Bearer {key}"}


@pytest.fixture(autouse=True)
def no_real_claude_settings(monkeypatch):
    """`setup claude-code` writes to $CLAUDE_CONFIG_DIR when it's set (as it is inside a Claude Code session):
    never let a test reach the real one. Nor Codex CLI's ($CODEX_HOME)."""
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    monkeypatch.delenv("CODEX_HOME", raising=False)


@pytest.fixture(autouse=True)
def no_real_providers(monkeypatch):
    """No test reaches a real provider: a request through httpx's network transports to anything but this machine (or a
    service a MERITORDER_TEST_* variable names, as CI's are) fails at once. Tests fake providers with httpx.MockTransport."""
    from urllib.parse import urlparse

    allowed = {"localhost", "127.0.0.1", "::1", "testserver", "test"}
    allowed |= {urlparse(v).hostname for k, v in os.environ.items() if k.startswith("MERITORDER_TEST_") and "://" in v} - {None}
    real_async, real_sync = httpx.AsyncHTTPTransport.handle_async_request, httpx.HTTPTransport.handle_request

    def check(request: httpx.Request) -> None:
        if request.url.host not in allowed:
            raise httpx.ConnectError(f"tests never reach real services ({request.url.host}): fake it with httpx.MockTransport",
                                     request=request)

    async def guarded_async(self, request):
        check(request)
        return await real_async(self, request)

    def guarded_sync(self, request):
        check(request)
        return real_sync(self, request)

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", guarded_async)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", guarded_sync)


@pytest.fixture(autouse=True)
def no_eval_waiting(monkeypatch):
    """Evals retry rate-limited requests for up to RETRY_FOR seconds; tests that want that turn it back on."""
    import meritorder.evals.runner as runner

    monkeypatch.setattr(runner, "RETRY_FOR", 0.0)
