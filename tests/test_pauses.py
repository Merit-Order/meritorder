"""How often sessions pause 5-60 minutes (decision 60): counted per team, shrunk towards the organization's share and a
prior, decayed over time, and shared between gateways through the store."""

import pytest
from conftest import ADMIN_KEY, auth

from meritorder.pauses import HALF_LIFE, PRIOR, PSEUDO, STALE, Pauses


def test_the_share_starts_at_the_prior_and_follows_the_teams_own_turns():
    p = Pauses("g1")
    assert p.share("eng") == pytest.approx(PRIOR)
    for gap in [30] * 50 + [600] * 50 + [7200] * 10:  # half the turns after a 10-minute pause; two hours doesn't count
        p.observe("eng", gap, now=p.at)
    p.observe("eng", 0)  # not a later turn
    turns, paused = p.counts("eng")
    assert (turns, paused) == (110, 50)
    org = (50 + PSEUDO * PRIOR) / (110 + PSEUDO)
    assert p.share("eng") == pytest.approx((50 + PSEUDO * org) / (110 + PSEUDO))
    assert p.share("ops") == pytest.approx(org)  # a team without turns of its own: the organization's share
    assert p.describe("eng")["turns"] == 110


def test_counts_halve_every_week_and_are_shared_between_gateways():
    a, b = Pauses("a"), Pauses("b")
    for _ in range(100):
        a.observe("eng", 900, now=1_000.0)
    doc = a.merged({}, now=1_000.0)
    b.load(doc, now=1_000.0 + HALF_LIFE)  # another gateway, a week later
    assert b.counts("eng") == pytest.approx((50, 50))
    a.load(doc, now=1_000.0)
    assert a.others == {}  # its own entry isn't counted twice
    b.observe("eng", 10, now=2_000.0)
    both = b.merged(doc, now=2_000.0)
    assert set(both["gateways"]) == {"a", "b"}
    assert set(b.merged(both, now=2_000.0 + STALE + 1)["gateways"]) == {"b"}  # a gateway gone a month is dropped


async def test_the_gateway_counts_pauses_and_saves_them(gateway, client):
    if not hasattr(gateway.state, "_kv"):
        pytest.skip("reaches into the in-memory state to age the session")
    body = {"model": "auto", "messages": [{"role": "user", "content": "hi"}]}
    headers = {**auth(ADMIN_KEY), "x-session-id": "pause-1"}
    assert (await client.post("/v1/chat/completions", json=body, headers=headers)).status_code == 200
    key = next(k for k in gateway.state._kv if k.startswith("sess:") and "pause-1" in k)
    value, ttl = gateway.state._kv[key]
    value["last_seen"] -= 600  # the person came back ten minutes later
    gateway.state._kv[key] = (value, ttl)
    body["messages"] += [{"role": "assistant", "content": "hello"}, {"role": "user", "content": "and now?"}]
    assert (await client.post("/v1/chat/completions", json=body, headers=headers)).status_code == 200
    assert gateway.router.pauses.counts()[1] == 1
    await gateway.save_token_calibration()
    saved = await gateway.store.get_setting("session_pauses")
    assert saved["gateways"][gateway.router.pauses.id]["teams"]
    fresh = Pauses("other")
    fresh.load(saved)
    assert fresh.counts()[1] == pytest.approx(1, abs=0.01)
