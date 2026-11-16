"""Robustness regressions: settings refused at load instead of failing every request, prices shown as they're billed, one
compression formula, a bounded WebSocket queue, and a stream's ledger row kept when closing it fails."""

from __future__ import annotations

import asyncio
import copy

import httpx
import pytest
from conftest import ADMIN_KEY, TEST_CONFIG, auth

from meritorder.app import build_gateway, create_app
from meritorder.config import Price, build_config
from meritorder.pricing import compression_savings
from meritorder.state import MemoryState
from meritorder.store import MemoryStore


def make(tmp_path, **overrides):
    raw = copy.deepcopy(TEST_CONFIG)
    raw.update(overrides)
    return build_config(raw, tmp_path)


@pytest.mark.parametrize("overrides, why", [
    ({"routing": {"max_attempts": 0}}, "max_attempts"),  # it would try no backend: a 500 on every request
    ({"routing": {"effort_by_floor": {3: "hgih"}}}, "unknown effort"),
    ({"routing": {"effort_by_floor": {7: "high"}}}, "1 to 5"),
    ({"teams": {"default": {"budget_usd_month": -5}}}, "budget_usd_month"),  # would refuse everyone
    ({"teams": {"default": {"user_budget_usd_month": -1}}}, "user_budget_usd_month"),
    ({"teams": {"default": {"max_session_usd": -1}}}, "max_session_usd"),
    ({"teams": {"default": {"rate_limits": {"user_rpm": -1}}}}, "user_rpm"),
    ({"teams": {"default": {"rate_limits": {"team_tpm": -1}}}}, "team_tpm"),
    ({"default_max_tokens": 0}, "default_max_tokens"),
])
def test_settings_that_would_break_every_request_are_refused_at_load(tmp_path, overrides, why):
    with pytest.raises(ValueError, match=why):
        make(tmp_path, **overrides)


def test_the_edges_of_those_settings_still_load(tmp_path):
    cfg = make(tmp_path, routing={"max_attempts": 1, "effort_by_floor": {1: "minimal", 5: "max"}}, default_max_tokens=1,
               teams={"default": {"budget_usd_month": 0, "rate_limits": {"user_rpm": 0}}})
    assert cfg.routing.max_attempts == 1 and cfg.team("default").budget_usd_month == 0


def test_compression_savings_counts_new_tokens_as_input_and_history_as_cache_reads():
    price = Price(input=3.0, output=15.0)
    assert compression_savings(price, 1_000_000, 0) == 3.0
    assert compression_savings(price, 0, 1_000_000) == pytest.approx(0.3)  # the default cache read: a tenth of input


async def test_the_model_table_shows_the_cache_prices_that_are_billed(client):
    models = (await client.get("/v1/admin/config", headers=auth(ADMIN_KEY))).json()["models"]
    price = next(m for m in models if m["name"] == "embed-small")["routes"][0]["price"]  # lists no cache prices
    assert price["cache_read"] == pytest.approx(0.002) and price["cache_write"] == 0.02 and price["cache_write_1h"] == 0.04
    small = next(m for m in models if m["name"] == "small")["routes"][0]["price"]
    assert small["cache_read"] == 0.02  # a listed one is kept


async def test_a_websocket_stream_queues_only_so_many_responses():
    from meritorder import responses_ws

    sent = []

    class Conn(responses_ws.ResponsesSocket):
        def __init__(self):
            self.queues, self.workers, self.jobs = {}, [], {}

        async def send(self, event):
            sent.append(event)

    conn = Conn()
    conn.lane = lambda lane, queue: asyncio.sleep(3600)  # a stream busy with its first response
    msg = '{"type": "response.create", "stream_id": "s1"}'
    for _ in range(responses_ws.MAX_QUEUED + 1):
        await conn.dispatch(msg)
    assert conn.queues["s1"].qsize() == responses_ws.MAX_QUEUED
    assert [e["error"]["code"] for e in sent] == ["websocket_queue_full"] and sent[0]["status"] == 429
    for w in conn.workers:
        w.cancel()


async def test_a_stream_whose_close_fails_still_writes_its_ledger_row(tmp_path, monkeypatch, caplog):
    from meritorder.backends.base import UpstreamResponse

    cfg = make(tmp_path)
    gw = build_gateway(cfg, store=MemoryStore(), state=MemoryState())
    app = create_app(gateway=gw)
    backend = gw.backends["mock-openai"]
    real_send = backend.send

    async def send(*a, **kw):
        resp = await real_send(*a, **kw)
        if resp.events is not None:
            async def broken_close():
                raise RuntimeError("connection reset while closing")
            resp = UpstreamResponse(resp.status, resp.headers, events=resp.events, _close=broken_close)
        return resp

    monkeypatch.setattr(backend, "send", send)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/v1/chat/completions", headers=auth(), json={"model": "small", "stream": True,
                                                                        "messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 200
    await gw.drain()
    assert gw.store.ledger and gw.store.ledger[-1]["status"] == 200
    assert "closing the stream from mock-openai failed" in caplog.text


async def test_a_stream_that_breaks_and_then_fails_to_close_keeps_its_estimated_row(tmp_path, monkeypatch, caplog):
    """Both at once: the upstream breaks mid-stream (502, usage estimated by broken_off_usage) and closing it fails too."""
    from meritorder.backends.base import UpstreamResponse

    cfg = make(tmp_path)
    gw = build_gateway(cfg, store=MemoryStore(), state=MemoryState())
    app = create_app(gateway=gw)
    backend = gw.backends["mock-openai"]
    real_send = backend.send

    async def send(*a, **kw):
        resp = await real_send(*a, **kw)
        if resp.events is None:
            return resp
        inner = resp.events

        async def breaking():  # breaks once the reply has started (failover has already committed to this route)
            n = 0
            async for ev in inner:
                yield ev
                n += 1
                if n == 3:
                    raise ConnectionError("upstream went away")

        async def broken_close():
            raise RuntimeError("connection reset while closing")

        return UpstreamResponse(resp.status, resp.headers, events=breaking(), _close=broken_close)

    monkeypatch.setattr(backend, "send", send)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        await c.post("/v1/chat/completions", headers=auth(), json={"model": "small", "stream": True,
                                                                   "messages": [{"role": "user", "content": "hello there"}]})
    await gw.drain()
    row = gw.store.ledger[-1]
    assert row["status"] == 502 and (row.get("tags") or {}).get("usage") == "estimated" and row["input_tokens"] > 0
    assert "closing the stream from mock-openai failed" in caplog.text
