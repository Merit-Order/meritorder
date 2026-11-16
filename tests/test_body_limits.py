"""Oversized request bodies are refused before a handler reads them."""

import httpx
import pytest
from conftest import auth

from meritorder.app import create_app


@pytest.fixture
async def small_client(gateway, monkeypatch):
    monkeypatch.setenv("MERITORDER_MAX_BODY_MB", "0.01")  # ~10 KB for the API routes
    app = create_app(gateway=gateway)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def test_api_body_limit(small_client):
    ok = {"model": "small", "messages": [{"role": "user", "content": "hi"}]}
    assert (await small_client.post("/v1/chat/completions", json=ok, headers=auth())).status_code == 200
    big = {"model": "small", "messages": [{"role": "user", "content": "x" * 20_000}]}
    r = await small_client.post("/v1/chat/completions", json=big, headers=auth())
    assert r.status_code == 413 and r.json()["error"]["type"] == "request_too_large"


async def test_streamed_body_without_length_is_counted(small_client):
    async def chunks():
        for _ in range(30):
            yield b"x" * 1000

    r = await small_client.post("/v1/chat/completions", content=chunks(), headers={**auth(), "content-type": "application/json"})
    assert r.status_code == 413


async def test_sign_in_exchange_is_small_and_type_checked(client):
    assert (await client.post("/auth/exchange", content=b"{" + b" " * 70_000 + b"}")).status_code == 413
    assert (await client.post("/auth/exchange", json=[1, 2])).status_code == 400
    assert (await client.post("/auth/exchange", content=b"not json")).status_code == 400


async def test_console_cannot_be_framed(client):
    r = await client.get("/ui/")
    if r.status_code == 200:  # only when the UI has been built
        assert "frame-ancestors 'none'" in r.headers["content-security-policy"] and r.headers["x-frame-options"] == "DENY"
    api = await client.get("/healthz")
    assert api.headers["x-content-type-options"] == "nosniff" and "content-security-policy" not in api.headers


async def test_console_says_how_to_build_it_when_it_isnt(gateway, monkeypatch, tmp_path):
    """`uv run meritorder serve` from a fresh clone has no console yet: /ui/ says how to get one, rather than a bare 404."""
    from meritorder import app as app_module

    monkeypatch.setattr(app_module, "UI_DIR", tmp_path / "missing")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(gateway=gateway)), base_url="http://test") as c:
        for path in ("/ui/", "/ui", "/ui/index.html"):
            r = await c.get(path)
            assert r.status_code == 404 and "npm run build" in r.text and r.headers["content-type"].startswith("text/html")
        assert (await c.get("/", follow_redirects=False)).headers["location"] == "/ui/"

    built = tmp_path / "built"
    built.mkdir()
    (built / "index.html").write_text('<div id="root"></div>')
    (built / "third-party-notices.txt").write_text("Name: react\nLicense: MIT\n")  # the build writes it (vite.config.ts)
    monkeypatch.setattr(app_module, "UI_DIR", built)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(gateway=gateway)), base_url="http://test") as c:
        r = await c.get("/ui/")
        assert r.status_code == 200 and '<div id="root">' in r.text
        notices = await c.get("/ui/third-party-notices.txt")  # linked from the console's footer, and from GET /about
        assert notices.status_code == 200 and "License: MIT" in notices.text
        assert (await c.get("/about")).json()["third_party_notices"] == "/ui/third-party-notices.txt"


def test_batch_uploads_get_the_providers_own_limits():
    from meritorder.guards import _limit

    assert _limit("/v1/files", 32) == 512 * 1024 * 1024 and _limit("/v1/messages/batches", 32) == 256 * 1024 * 1024
    assert _limit("/v1/messages", 32) == 32
