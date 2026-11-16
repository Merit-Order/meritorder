"""A background response its gateway is still working on is never given up on as stale, however long it takes: the
gateway touches its row while it runs, and the worker fails only the ones no gateway has touched for the timeout."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from test_stored_responses import http, oai, responses_key, stored_mode  # noqa: F401 (fixtures)

pytestmark = pytest.mark.usefixtures("responses_key")


@pytest.mark.parametrize("store", [True, False])
async def test_a_background_response_running_longer_than_the_timeout_is_not_failed(http, gateway, store):  # noqa: F811
    stored_mode(gateway)
    timeout = 0.2
    gateway.cfg.response_store.background_timeout_seconds = timeout
    original = gateway.backends["mock-openai"].send

    async def slow(body, stream, headers, **kw):
        await asyncio.sleep(timeout * 4)  # a long reasoning turn: four timeouts
        return await original(body, stream, headers, **kw)

    gateway.backends["mock-openai"].send = slow
    client = oai(http)
    r = await client.responses.create(model="small", input="think for a long time", background=True, store=store)
    for _ in range(6):  # the worker's maintenance, run often while the response is still being made
        await asyncio.sleep(timeout / 2)
        assert await gateway.store.stale_background(datetime.now(UTC) - timedelta(seconds=timeout)) == []
        if not store:
            row = await gateway.store.get_stored_response(r.id)
            assert row["expires_at"] > datetime.now(UTC), "a store: false response expired while it was still running"
    await gateway.drain()
    assert (await client.responses.retrieve(r.id)).status == "completed"
