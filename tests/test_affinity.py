"""Replica affinity on GPU deployments: sessions go back to the replica that holds their KV cache unless it's gone,
full, draining or too busy; eviction is noticed from vLLM's KV events or estimated from its KV-cache metrics."""

from __future__ import annotations

import asyncio
import json
import socket
import time

import httpx
import pytest
import yaml
from conftest import ADMIN_KEY, LEAD_KEY, auth
from test_http_backend import OPENAI_OK, Upstream, wire

from meritorder import controller, gpu, kv_events
from meritorder.affinity import Replica, Warmth, estimated_share, kv_session_id, mark, placements, replicas_of, share_on
from meritorder.backends.mock import MockBackend
from meritorder.config import BackendConfig, Config, ModelConfig, Route, ServeConfig
from meritorder.kv_events import DeploymentEvents, ReplicaIndex, Subscriber, decode_batch, decode_event
from meritorder.router import RoutingError
from meritorder.sessions import Session, session_key
from meritorder.state import MemoryState

BODY = {"messages": [{"role": "user", "content": "go on"}]}
PRICE = {"input": 1, "output": 2, "cache_read": 0.1}


def add_gpu(gateway, replicas: int = 2, api_route: bool = False, max_concurrency: int | None = None, **serve) -> MockBackend:
    """A two-replica GPU deployment (in-process mock) serving model 'gpu-only'."""
    serve.setdefault("static_urls", [f"http://r{i}.test/v1" for i in range(1, replicas + 1)])
    cfg = BackendConfig(name="gpu", kind="self_hosted", format="openai", mock_latency_ms=0, max_concurrency=max_concurrency,
                        serve=ServeConfig(model="gpu-only", provisioner="static", **serve))
    b = MockBackend(cfg)
    gateway.backends["gpu"] = b
    gateway.cfg.backends.append(cfg)
    routes = [Route(backend="gpu", model="m", price=PRICE)]
    if api_route:
        routes.append(Route(backend="mock-openai", model="m-api", price=PRICE))
    gateway.cfg.models.append(ModelConfig(name="gpu-only", quality={"coding": 5, "general": 5}, routes=routes))
    return b


def rep(i, util: float = 0.1, status: str = "READY", **kw) -> dict:
    return {"id": i, "status": status, "url": f"http://r{i}.test/v1", "endpoint": f"http://r{i}.test/v1", "utilization": util, **kw}


async def publish(gateway, *replicas: dict, state: str = "warm") -> None:
    await gateway.state.set_json("deploy:gpu", {"state": state, "endpoint": "http://lb.test/v1", "replicas": len(replicas),
                                                "replicas_info": list(replicas)})
    await gateway.refresh_deployments()


def placed(replica: str | None = "1", tokens: int = 200_000, turns: int = 5, at: float | None = None, **mark_) -> Session:
    """A session whose last turn ran on ``replica`` of 'gpu' just now."""
    at = time.time() if at is None else at
    s = Session(id="s", model="gpu-only", backend="gpu", turns=turns, last_seen=at, prefix_tokens=tokens - 2000)
    s.placement = {"gpu": {"replica": replica, "at": at, "tokens": tokens, **mark_}}
    s.warm_until = {"gpu": at + 300}
    return s


def decide(gateway, session=None, tokens=200_000, util=None, kv=None, now=None, queue_full=frozenset()):
    return gateway.router.decide("gpu-only", gateway.cfg.team("default"), BODY, {}, session, tokens, False, util or {},
                                 now or time.time(), queue_full, kv)


# ------------------------------------------------------------------ routing


async def test_a_session_goes_back_to_the_replica_that_served_its_last_turn(gateway):
    add_gpu(gateway)
    await publish(gateway, rep(1, 0.3), rep(2, 0.1))
    d = decide(gateway, placed("1"))
    top = d.candidates[0]
    assert top.replica.id == "1" and top.warm and top.warmth == Warmth(1.0, "ttl")
    assert "gpu: replica 1 (100% warm (ttl), 30% load)" in d.reason
    # A new session has no cache anywhere: the least loaded replica.
    assert decide(gateway).candidates[0].replica.id == "2"


async def test_saturation_drain_and_a_gone_replica_override_affinity(gateway):
    add_gpu(gateway)
    await publish(gateway, rep(1, 1.0), rep(2, 0.2))
    d = decide(gateway, placed("1"))
    assert d.candidates[0].replica.id == "2" and not d.candidates[0].warm
    assert "replica 1 is full; replica 2 (cold, 20% load)" in d.reason

    await publish(gateway, rep(1, 0.1, status="SHUTTING_DOWN"), rep(2, 0.2))
    d = decide(gateway, placed("1"))
    assert d.candidates[0].replica.id == "2" and "replica 1 is draining" in d.reason

    await publish(gateway, rep(2, 0.2), rep(3, 0.5))
    d = decide(gateway, placed("1"))
    assert d.candidates[0].replica.id == "2" and "replica 1 is gone" in d.reason


async def test_a_busy_warm_replica_loses_to_an_idle_cold_one_only_when_rebuilding_pays(gateway):
    add_gpu(gateway)
    await publish(gateway, rep(1, 0.9), rep(2, 0.0))
    # A long session: the turns it would spend on a busy replica cost more than rebuilding 200k tokens on an idle one.
    d = decide(gateway, placed("1", tokens=200_000))
    assert d.candidates[0].replica.id == "2"
    assert "replica 2 (cold, 0% load) over replica 1 (100% warm (ttl), 90% load)" in d.reason
    # A short one isn't worth moving: staying within the switch margin keeps its cache.
    d = decide(gateway, placed("1", tokens=10_000), tokens=10_000)
    assert d.candidates[0].replica.id == "1" and "staying on replica 1" in d.reason
    # Below the shadow price's knee every replica costs the same per turn, so the warm one wins.
    await publish(gateway, rep(1, 0.5), rep(2, 0.0))
    assert decide(gateway, placed("1")).candidates[0].replica.id == "1"


async def test_every_replica_full_queues_on_the_shortest_queue(gateway):
    add_gpu(gateway)
    await publish(gateway, rep(1, 1.0, waiting=9), rep(2, 1.0, waiting=2))
    d = decide(gateway, placed("1"))
    assert d.last_resort and d.candidates[0].queued and d.candidates[0].replica.id == "2"
    assert "every replica is full: queued on replica 2" in d.reason


async def test_live_inflight_counts_fill_one_replica(gateway):
    add_gpu(gateway, max_concurrency=2)
    await publish(gateway, rep(1, 0.1), rep(2, 0.1))
    held = [gateway.inflight.acquire("gpu#1", 600), gateway.inflight.acquire("gpu#1", 600)]
    util, _ = gateway.load({"util": {}, "inflight": {"other": {"gpu#2": 1}}})
    assert util["gpu#1"] == 1.0 and util["gpu#2"] == 0.5
    assert decide(gateway, placed("1"), util=util).candidates[0].replica.id == "2"
    for r in held:
        r()


async def test_a_draining_deployment_finishes_warm_sessions_on_their_replica_only(gateway):
    add_gpu(gateway, api_route=True)
    await publish(gateway, rep(1, 0.2), rep(2, 0.1), state="draining")
    d = decide(gateway, placed("1"))
    assert d.candidates[0].backend.name == "gpu" and d.candidates[0].replica.id == "1"
    cold = decide(gateway, placed("1", at=time.time() - 900))  # past the cache TTL, no metrics: cold
    assert [c.backend.name for c in cold.candidates] == ["mock-openai"]
    assert [c.backend.name for c in decide(gateway).candidates] == ["mock-openai"]  # no new sessions


async def test_kv_events_say_the_cache_was_evicted_and_the_rebuild_is_priced(gateway):
    add_gpu(gateway)
    await publish(gateway, rep(1, 0.3), rep(2, 0.3))
    s, now = placed("1"), time.time()
    warm = decide(gateway, s, now=now).candidates[0]
    gone = decide(gateway, s, kv={"gpu": {"replica": "1", "held": 0, "tokens": 200_000, "at": now}}, now=now).candidates[0]
    half = decide(gateway, s, kv={"gpu": {"replica": "1", "held": 100_000, "tokens": 200_000, "at": now}}, now=now).candidates[0]
    assert warm.warmth == Warmth(1.0, "ttl") and gone.warmth == Warmth(0.0, "events") and half.warmth == Warmth(0.5, "events")
    rebuild = 198_000 * (1 - 0.1) / 1e6 * 0.02  # the cold prefix at the input price, less cache reads, at the shadow price
    assert gone.cost - warm.cost == pytest.approx(rebuild) and half.cost - warm.cost == pytest.approx(rebuild / 2)
    # The events put its blocks on replica 2 (the load balancer sent it there): replica 1 is cold, 2 is warm.
    moved = decide(gateway, s, kv={"gpu": {"replica": "2", "held": 200_000, "tokens": 200_000, "at": now}}, now=now)
    assert moved.candidates[0].replica.id == "2" and moved.candidates[0].warmth.source == "events"
    # A note older than the last turn describes a cache that turn has since rebuilt: ignored.
    stale = decide(gateway, s, kv={"gpu": {"replica": "1", "held": 0, "tokens": 200_000, "at": now - 60}}, now=now)
    assert stale.candidates[0].warmth == Warmth(1.0, "ttl")


async def test_eviction_estimated_from_kv_metrics(gateway):
    add_gpu(gateway)
    now = time.time()
    kvm = {"kv_capacity_tokens": 1_000_000, "kv_usage": 0.5, "sampled_at": now, "started_at": 100.0}
    s = placed("1", tokens=200_000, at=now, mark=10_000_000, free=500_000, started=100.0)
    for written, share in ((10_100_000, 1.0), (10_600_000, 0.5), (10_800_000, 0.0)):
        await publish(gateway, rep(1, 0.3, kv_written_tokens=written, **kvm), rep(2, 0.3, kv_written_tokens=0, **kvm))
        c = decide(gateway, s, now=now).candidates[0]
        assert c.warmth == Warmth(share, "estimate") and c.replica.id == "1"
    # Old turns stay warm on an idle replica long after the TTL: vLLM evicts on pressure, not time.
    later = placed("1", tokens=200_000, at=now - 3600, mark=10_000_000, free=500_000, started=100.0)
    await publish(gateway, rep(1, 0.3, kv_written_tokens=10_000_000, **kvm), rep(2, 0.3, **kvm))
    assert decide(gateway, later, now=now).candidates[0].warmth == Warmth(1.0, "estimate")
    # A restarted server holds nothing.
    await publish(gateway, rep(1, 0.3, kv_written_tokens=10_000_000, **{**kvm, "started_at": 999.0}), rep(2, 0.3))
    assert decide(gateway, s, now=now).candidates[0].warmth.share == 0.0


def test_estimated_share_edges():
    r = Replica(id="1", url="u", kv_capacity=100, kv_usage=0.0, kv_written=50, kv_rate=10.0, sampled_at=0.0)
    assert estimated_share({"mark": 50, "free": 100, "tokens": 50}, r, 10.0) == 1.0  # 100 written: just the free space
    assert estimated_share({"mark": 50, "free": 100, "tokens": 50}, r, 15.0) == 0.0  # 150 written: all of it
    assert estimated_share({"mark": 60, "free": 100, "tokens": 50}, r, 0.0) == 0.0  # counters went back: restarted
    assert estimated_share({"tokens": 50}, r, 0.0) is None and estimated_share({"mark": 1, "free": 1}, Replica("2", "u"), 0) is None
    assert r.free_tokens() == 100 and Replica("x", None).free_tokens() is None and Replica("x", None).written(1) is None
    assert Replica("x", None).unavailable() == "not reachable on its own" and Replica("x", "u", status="FAILED").unavailable() == "failed"
    assert Warmth(0.4, "estimate").describe() == "40% warm (estimate)" and Warmth(0.0, "events").describe() == "cold (events)"


def test_a_mark_ahead_of_the_next_sample_isnt_a_restart():
    """A turn's mark guesses the replica's counter from its last sample, at the rate it was writing then. When writing slows
    (a burst ended) the next sample's counter is below that guess: nothing was evicted, and the server didn't restart."""
    busy = Replica("1", "u", kv_capacity=100_000, kv_usage=0.2, kv_written=1_000_000, kv_rate=5_000.0, sampled_at=0.0, started_at=50.0)
    p = mark(busy, 30.0, 20_000)  # 30 s after the sample: a guess of 150,000 tokens written since it
    quiet = Replica("1", "u", kv_capacity=100_000, kv_usage=0.0, kv_written=1_010_000, kv_rate=100.0, sampled_at=60.0, started_at=50.0)
    assert estimated_share(p, quiet, 65.0) == 1.0  # at most 10,500 written since the turn: well within the free space
    # A counter below the one the turn saw: the server started again (when its start time isn't published).
    reset = Replica("1", "u", kv_capacity=100_000, kv_usage=0.0, kv_written=2_000, kv_rate=10.0, sampled_at=60.0)
    assert estimated_share(p, reset, 65.0) == 0.0


async def test_behind_a_load_balancer_the_cache_is_found_one_time_in_n(gateway):
    add_gpu(gateway, replica_affinity=False)
    await publish(gateway, rep(1, 0.3), rep(2, 0.3), rep(3, 0.3), rep(4, 0.3))
    c = decide(gateway, placed(None)).candidates[0]
    assert c.replica is None and c.warmth == Warmth(0.25, "ttl")
    now = time.time()
    c = decide(gateway, placed(None, at=now), kv={"gpu": {"replica": "2", "held": 200_000, "tokens": 200_000, "at": now}}, now=now).candidates[0]
    assert c.warmth == Warmth(0.25, "events")
    assert share_on(None, "gpu", None, now) == Warmth(0.0, "cold")
    # Draining: a session cached on any replica finishes there; one cached nowhere goes elsewhere.
    await publish(gateway, rep(1, 0.3), rep(2, 0.3), rep(3, 0.3), rep(4, 0.3), state="draining")
    assert decide(gateway, placed(None)).candidates[0].backend.name == "gpu"
    with pytest.raises(RoutingError) as e:
        decide(gateway, placed(None, at=now - 900))
    assert "gpu: draining, only finishing existing sessions" in e.value.message


async def test_unreachable_replica_falls_back_to_the_deployment_endpoint(gateway, client):
    from meritorder.backends import BackendError

    b = add_gpu(gateway)
    await publish(gateway, rep(1, 0.1), rep(2, 0.5))
    real = b.send
    calls = []

    async def send(body, stream, headers, base_url=None, session_tag=None):
        calls.append(base_url)
        if base_url:
            raise BackendError(502, "gpu: ConnectError: refused", retryable=True, unreachable=True)
        return await real(body, stream, headers, base_url=base_url, session_tag=session_tag)

    b.send = send
    r = await client.post("/v1/chat/completions", json={"model": "gpu-only", **BODY}, headers=auth())
    assert r.status_code == 200 and calls == ["http://r1.test/v1", None]
    assert b.unreachable["1"] > time.time() and b.addressable(time.time())[0].id == "2"
    assert gateway.metrics.counters["meritorder_gpu_replica_unreachable_total"]


# ------------------------------------------------------------------ the gateway end to end


async def test_turns_of_a_session_land_on_one_replica_and_hit_its_cache(gateway, client):
    b = add_gpu(gateway)
    await publish(gateway, rep(1, 0.1), rep(2, 0.1))
    long = "the whole repository, " * 400
    msgs = [{"role": "user", "content": long}]
    headers = {**auth(), "x-session-id": "sess-1"}
    r1 = await client.post("/v1/chat/completions", json={"model": "gpu-only", "messages": msgs}, headers=headers)
    msgs += [{"role": "assistant", "content": r1.json()["choices"][0]["message"]["content"]}, {"role": "user", "content": "and then?"}]
    r2 = await client.post("/v1/chat/completions", json={"model": "gpu-only", "messages": msgs}, headers=headers)
    assert r1.status_code == r2.status_code == 200
    kv_id = kv_session_id(session_key("alice", "sess-1"))
    assert b.targets == [("http://r1.test/v1", kv_id)] * 2  # same replica both turns, tagged for vLLM's KV events
    assert r2.json()["usage"]["prompt_tokens_details"]["cached_tokens"] > 1000  # the mock caches per replica
    s = Session.from_dict(await gateway.state.get_session(session_key("alice", "sess-1")), "sess-1")
    assert s.placement["gpu"]["replica"] == "1" and s.placement["gpu"]["tokens"] > 1000
    await gateway.drain()
    reg = await gateway.state.hgetall_json("placements:gpu")
    assert reg[kv_id]["replica"] == "1" and reg[kv_id]["user"] == "alice" and reg[kv_id]["expected"] == 1.0
    # The warm-cache expectation is checked against what the server reported.
    assert gateway.metrics.counters["meritorder_gpu_cache_turns_total"][
        (("backend", "gpu"), ("expected", "warm"), ("result", "hit"))] == 1

    # Replica 1 fills up: the next turn goes to replica 2 (and that turn's cache starts there).
    await publish(gateway, rep(1, 1.0), rep(2, 0.1))
    msgs += [{"role": "assistant", "content": "ok"}, {"role": "user", "content": "more"}]
    await client.post("/v1/chat/completions", json={"model": "gpu-only", "messages": msgs}, headers=headers)
    assert b.targets[-1][0] == "http://r2.test/v1"
    # Its cache is on replica 2 now, so it stays there even with replica 1 free again...
    await publish(gateway, rep(1, 0.1), rep(2, 0.3))
    msgs += [{"role": "assistant", "content": "ok"}, {"role": "user", "content": "again"}]
    await client.post("/v1/chat/completions", json={"model": "gpu-only", "messages": msgs}, headers=headers)
    assert b.targets[-1][0] == "http://r2.test/v1"
    # ...until the conversation is compacted: no cache anywhere, so the least loaded replica.
    await client.post("/v1/chat/completions", json={"model": "gpu-only", "messages": [{"role": "user", "content": "summary"}]}, headers=headers)
    assert b.targets[-1][0] == "http://r1.test/v1"


async def test_gpu_report_shows_placements_by_role(gateway, client):
    add_gpu(gateway)
    await publish(gateway, rep(1, 0.1), rep(2, 0.1, status="SHUTTING_DOWN"))
    now = time.time()
    await gateway.state.hset_json("placements:gpu", "k1", {"replica": "1", "at": now, "tokens": 5000, "session": "sess-a", "user": "alice",
                                                            "team": "default", "expected": 1.0, "source": "ttl"})
    await gateway.state.hset_json("placements:gpu", "k2", {"replica": "2", "at": now - 10, "tokens": 900, "session": "sess-b", "user": "bob",
                                                            "team": "ops"})
    await gateway.state.hset_json("placements:gpu", "k3", {"replica": None, "at": now - 20, "tokens": 900, "team": "ops"})
    await gateway.state.hset_json("placements:gpu", "old", {"replica": "1", "at": now - 7 * 3600, "team": "ops"})
    await gateway.state.set_json("kv:k1", {"gpu": {"replica": "1", "held": 2500, "tokens": 5000, "at": now + 1}})
    r = await client.get("/v1/reports/gpu", headers=auth(ADMIN_KEY))
    [d] = [x for x in r.json()["deployments"] if x["backend"] == "gpu"]
    rows = {p["id"]: p for p in d["placements"]}
    assert set(rows) == {"k1", "k2", "k3"}
    assert (rows["k1"]["share"], rows["k1"]["source"]) == (0.5, "events")
    assert rows["k2"]["replica_status"] == "SHUTTING_DOWN" and rows["k3"]["replica"] is None
    assert d["replica_sessions"] == {"1": {"sessions": 1, "warm": 1}, "2": {"sessions": 1, "warm": 0}}
    assert d["affinity"] == {"enabled": True, "kv_events": False}
    lead = await client.get("/v1/reports/gpu", headers=auth(LEAD_KEY))
    [d] = [x for x in lead.json()["deployments"] if x["backend"] == "gpu"]
    assert [p["id"] for p in d["placements"]] == ["k1"]  # their own team's sessions only


async def test_placements_without_replicas(state):
    now = time.time()
    await state.hset_json("placements:gpu", "a", {"replica": "9", "at": now, "tokens": 10})
    rows, per = await placements(state, "gpu", {}, 300, now)
    assert rows[0]["source"] == "cold" and rows[0]["replica_status"] == "gone" and per == {}
    await state.hdel("placements:gpu", ["a"])
    await state.hdel("placements:gpu", [])
    assert await state.hgetall_json("placements:gpu") == {}


def test_mark_records_the_replica_counters():
    r = Replica("2", "u", kv_capacity=1000, kv_usage=0.25, kv_written=500, kv_rate=10, sampled_at=100, started_at=50)
    assert mark(r, 110, 300) == {"replica": "2", "at": 110, "tokens": 300, "mark": 600, "seen": 500, "free": 750, "started": 50}
    assert mark(None, 1, 2) == {"replica": None, "at": 1, "tokens": 2}
    assert [x.id for x in replicas_of({"replicas_info": [{"id": 1}, {"status": "READY"}]})] == ["1"]


# ------------------------------------------------------------------ KV events


def stored(hashes, parent=None, session="s1", medium="GPU", **kw):
    return {"type": "BlockStored", "block_hashes": hashes, "parent_block_hash": parent, "token_ids": list(range(16 * len(hashes))),
            "block_size": 16, "lora_id": None, "medium": medium, "lora_name": None, "session_id": session, **kw}


def test_events_decode_in_both_encodings():
    assert decode_event(stored([b"a"]))["session_id"] == "s1"
    old = decode_event(["BlockStored", [b"a"], None, [1] * 16, 16, None, "GPU", None])
    assert old["block_hashes"] == [b"a"] and old["medium"] == "GPU" and old.get("session_id") is None
    assert decode_event(["BlockRemoved", [b"a"], "GPU"]) == {"type": "BlockRemoved", "block_hashes": [b"a"], "medium": "GPU"}
    assert decode_event({"type": "AllBlocksCleared"}) == {"type": "AllBlocksCleared"}
    assert decode_event({"type": "Other"}) is None and decode_event(["Other"]) is None and decode_event(3) is None
    assert decode_batch([12.5, [stored([b"a"]), {"type": "nope"}]])[0] == 12.5
    assert len(decode_batch({"ts": 1, "events": [{"type": "AllBlocksCleared"}]})[1]) == 1
    assert decode_batch("junk") == (0.0, [])


def test_replica_index_tracks_each_sessions_chain():
    idx = ReplicaIndex("1")
    assert idx.apply(stored([1, 2, 3]), 1.0) == {"s1"}
    assert idx.apply(stored([4], parent=3), 2.0) == {"s1"}
    assert idx.note("s1") == {"replica": "1", "held": 64, "tokens": 64, "updated": 2.0}
    assert idx.apply({"type": "BlockRemoved", "block_hashes": [3], "medium": "GPU"}, 3.0) == {"s1"}
    assert idx.note("s1")["held"] == 32  # a prefix lookup stops at the first missing block
    assert idx.apply({"type": "BlockRemoved", "block_hashes": [99], "medium": "GPU"}, 3.0) == set()
    idx.apply(stored([3], session=None), 4.0)  # computed again by another request: held again
    assert idx.note("s1")["held"] == 64
    idx.apply(stored([5], parent=2), 5.0)  # the conversation was edited after block 2: the chain branches there
    assert idx.chains["s1"].hashes == [1, 2, 5] and 3 not in idx.owners and 3 not in idx.held
    idx.apply(stored([7, 8], parent=42, session="s2"), 6.0)  # starts after a block it shares with someone else
    assert idx.note("s2")["tokens"] == 32
    assert idx.apply(stored([9], medium="CPU"), 7.0) == set() and idx.apply(stored([9], group_idx=1), 7.0) == set()
    assert idx.apply({"type": "AllBlocksCleared"}, 8.0) == {"s1", "s2"} and idx.note("s1")["held"] == 0
    assert idx.note("nobody") is None
    small = ReplicaIndex("2", max_sessions=1)
    small.apply(stored([1]), 1.0)
    small.apply(stored([2], session="s2"), 2.0)
    assert list(small.chains) == ["s2"] and 1 not in small.owners
    assert small.apply(stored([], session="s3"), 3.0) == set()
    small.apply(stored([[1, 2]], session="s4", token_ids=[0] * 32, block_size=None), 4.0)  # old tuple hashes, size from tokens
    assert small.note("s4")["tokens"] == 32


async def test_notes_follow_the_latest_replica_and_lost_events_fall_back():
    state = MemoryState()
    clock = [100.0]
    dep = DeploymentEvents("gpu", state, clock=lambda: clock[0])
    dep.receive("1", 0, [1.0, [stored([1, 2])]])
    clock[0] = 101.0
    dep.receive("2", 7, [1.0, [stored([5], session="s1")]])  # the load balancer sent its next turn to replica 2
    assert await dep.flush() == 1
    assert (await state.get_json("kv:s1"))["gpu"] == {"replica": "2", "held": 16, "tokens": 16, "at": 101.0}
    dep.receive("2", 9, [1.0, []])  # 8 was lost: replica 2's index is dropped, its sessions fall back to replica 1
    await dep.flush()
    assert (await state.get_json("kv:s1"))["gpu"]["replica"] == "1" and dep.sessions("2") == 0 and dep.sessions("1") == 1
    dep.forget("1")
    await dep.flush()
    assert await state.get_json("kv:s1") == {}  # known nowhere: the gateways estimate instead


async def test_subscriber_follows_a_real_zmq_publisher():
    import msgpack
    import zmq
    import zmq.asyncio

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    ctx = zmq.asyncio.Context.instance()
    pub = ctx.socket(zmq.PUB)
    pub.bind(f"tcp://127.0.0.1:{port}")
    state = MemoryState()
    sub = Subscriber(state)
    sub.sync("gpu", {"1": f"tcp://127.0.0.1:{port}"})
    assert sub.following("gpu") == {"1"}
    kv_events.FLUSH_SECONDS, flush = 0.05, kv_events.FLUSH_SECONDS
    try:
        # vLLM's wire format: topic, 8-byte big-endian sequence, msgpack [ts, [events]] (map-encoded events).
        payload = msgpack.packb([time.time(), [stored([b"\x01" * 32, b"\x02" * 32], session="kv-sess")]], use_bin_type=True)
        for seq in range(200):  # a SUB socket misses what's sent before it has connected
            await pub.send_multipart([b"", seq.to_bytes(8, "big"), payload])
            await asyncio.sleep(0.02)
            if await state.get_json("kv:kv-sess"):
                break
        note = (await state.get_json("kv:kv-sess"))["gpu"]
        assert note["replica"] == "1" and note["held"] == note["tokens"] == 32
        sub.sync("gpu", {})  # the replica went away
        await asyncio.sleep(0)
        assert sub.following("gpu") == set()
    finally:
        kv_events.FLUSH_SECONDS = flush
        await sub.close()
        pub.close(linger=0)


async def test_subscriber_reconnects_a_reader_that_died(monkeypatch):
    closed = []

    class Broken:
        async def recv_multipart(self):
            await asyncio.sleep(0)
            return [b"", b"\x00" * 8]  # too few frames: skipped, then the socket fails

        def close(self):
            closed.append(1)

    class Dead(Broken):
        n = 0

        async def recv_multipart(self):
            Dead.n += 1
            if Dead.n > 1:
                raise RuntimeError("connection lost")
            return await super().recv_multipart()

    sub = Subscriber(MemoryState(), socket=lambda endpoint: Dead(), unpack=lambda b: b)
    sub.sync("gpu", {"1": "tcp://h:5557"})
    for _ in range(5):
        await asyncio.sleep(0)
    assert sub.following("gpu") == set() and closed
    sub.sync("gpu", {"1": "tcp://h:5557"})  # the next cycle connects again
    assert sub.following("gpu") == {"1"}
    await sub.close()
    monkeypatch.setattr(kv_events, "available", lambda: "missing")
    lazy = Subscriber(MemoryState())
    lazy.sync("gpu", {"1": "tcp://h:5557"})  # without pyzmq and msgpack: nothing to follow with
    assert lazy.following("gpu") == set() and lazy.socket is None


def test_kv_events_available_here():
    assert kv_events.available() is None  # the dev dependencies include pyzmq and msgpack


# ------------------------------------------------------------------ controller and provisioners


async def test_controller_samples_each_replica_and_its_kv_cache(monkeypatch):
    answers = {
        "http://r1.test/v1": {"utilization": 0.4, "waiting": 1, "running": 3, "kv_capacity_tokens": 1000, "kv_usage": 0.4,
                              "kv_written_tokens": 5000.0, "started_at": 10.0, "prefix_cache_hits": 80.0, "prefix_cache_queries": 100.0},
        "http://r2.test/v1": None,
    }

    async def fetch_load(url, headers=None):
        return answers.get(url)

    monkeypatch.setattr(controller, "fetch_load", fetch_load)
    before = [{"id": 1, "sampled_at": 90.0, "kv_written_tokens": 4000.0, "started_at": 10.0, "prefix_cache_hits": 40.0,
               "prefix_cache_queries": 50.0}]
    reps = [{"id": 1, "status": "READY", "endpoint": "r1.test"}, {"id": 2, "status": "READY", "endpoint": "http://r2.test/v1"},
            {"id": 3, "status": "PROVISIONING", "endpoint": None}]
    await controller.sample_replicas(reps, before, 100.0)
    r1, r2, r3 = reps
    assert r1["url"] == "http://r1.test/v1" and r1["utilization"] == 0.4 and r1["kv_written_per_second"] == 100.0
    assert r1["prefix_hit_rate"] == 0.8 and "utilization" not in r2 and r3["url"] is None
    serve = ServeConfig(model="m", kv_events=True)
    assert controller.kv_endpoints(serve, reps) == {"1": "tcp://r1.test:5557", "2": "tcp://r2.test:5557"}
    assert controller.cache_signal(serve, {"replicas_info": reps}, set())["source"] == "estimate"
    assert controller.cache_signal(serve, {"replicas_info": reps}, {"1"})["source"] == "events"
    off = controller.cache_signal(ServeConfig(model="m", provisioner="static", replica_affinity=False), {}, set())
    assert off == {"affinity": False, "source": "ttl", "kv_events": False, "kv_events_problem": None}


async def test_controller_tick_publishes_replicas_follows_kv_events_and_prunes(monkeypatch):
    from test_worker_controller import gpu_config

    from meritorder.gpu import Deployment, StaticProvisioner
    from meritorder.store import MemoryStore

    async def fetch_load(url, headers=None):
        return {"utilization": 0.2 if "r1" in url else 0.6, "waiting": 0, "running": 1}

    async def probe(url):
        return True

    monkeypatch.setattr(controller, "fetch_load", fetch_load)
    monkeypatch.setattr(gpu, "_probe", probe)
    cfg = gpu_config(min_replicas=1, kv_events=True, static_urls=["http://r1.test/v1", "http://r2.test/v1"])
    b = cfg.backends[0]
    state = MemoryState()
    await state.hset_json("placements:gpu", "old", {"at": 1.0})
    await state.hset_json("placements:gpu", "new", {"at": time.time()})

    class FakeKV:
        def __init__(self):
            self.synced = {}

        def sync(self, backend, endpoints):
            self.synced[backend] = endpoints

        def following(self, backend):
            return set(self.synced.get(backend, {}))

    kv = FakeKV()
    deps = [Deployment(backend=b, model=cfg.models[0], provisioner=StaticProvisioner(b, cfg))]
    [rec] = await controller.tick(cfg, MemoryStore(), state, deps, kv=kv)
    assert rec["state"] == "warm" and rec["utilization"] == pytest.approx(0.4)  # the replicas' mean, not the front's
    assert [r["kv_events"] for r in rec["replicas_info"]] == [True, True]
    assert kv.synced["gpu"] == {"1": "tcp://r1.test:5557", "2": "tcp://r2.test:5557"}
    assert rec["cache"]["source"] == "events" and rec["cache"]["affinity"]
    assert set(await state.hgetall_json("placements:gpu")) == {"new"}


def test_vllm_metrics_with_kv_cache_figures():
    text = "\n".join([
        'vllm:cache_config_info{block_size="16",cache_dtype="auto",num_gpu_blocks="1000",num_cpu_blocks="0"} 1.0',
        'vllm:kv_cache_usage_perc{model_name="m"} 0.25',
        'vllm:prompt_tokens_total{model_name="m"} 9000.0',
        'vllm:generation_tokens_total{model_name="m"} 1000.0',
        'vllm:prefix_cache_hits_total{model_name="m"} 6000.0',
        'vllm:prefix_cache_queries_total{model_name="m"} 8000.0',
        "process_start_time_seconds 1.7e9",
    ])
    load = gpu.vllm_load(text)
    assert load["kv_capacity_tokens"] == 16_000 and load["kv_usage"] == 0.25
    assert load["kv_written_tokens"] == 4000.0 and load["prefix_cache_hits"] == 6000.0 and load["started_at"] == 1.7e9
    old = gpu.vllm_load('vllm:cache_config_info{block_size="x",num_gpu_blocks="1"} 1\nvllm:gpu_prefix_cache_hits_total 5\n'
                        'vllm:gpu_prefix_cache_queries_total 9\n')
    assert "kv_capacity_tokens" not in old and old["prefix_cache_queries"] == 9.0 and "kv_written_tokens" not in old


def test_replica_url_forms():
    assert gpu.replica_url("3.4.5.6:8000") == "http://3.4.5.6:8000/v1"
    assert gpu.replica_url("https://h:8000/") == "https://h:8000/v1" and gpu.replica_url("http://h/v1") == "http://h/v1"
    assert gpu.replica_url("-") is None and gpu.replica_url(None) is None


async def test_static_server_with_one_url_per_replica(monkeypatch):
    up = {"http://r1.test/v1": False, "http://r2.test/v1": True, "http://lb.test/v1": False}

    async def probe(url):
        return up[url]

    monkeypatch.setattr(gpu, "_probe", probe)
    b = BackendConfig(name="gpu", kind="self_hosted", format="openai",
                      serve=ServeConfig(model="m", provisioner="static", static_urls=["r1.test", "http://r2.test/v1"]))
    st = await gpu.StaticProvisioner(b, Config(backends=[b], models=[])).status()
    assert st.ready and st.endpoint == "http://r2.test/v1" and st.replicas == 1
    assert [(r["id"], r["status"]) for r in st.replicas_info] == [(1, "NOT_READY"), (2, "READY")]
    b.serve.static_url = "http://lb.test/v1"
    st = await gpu.StaticProvisioner(b, Config(backends=[b], models=[])).status()
    assert st.endpoint == "http://r2.test/v1"  # the front doesn't answer: a replica does
    up["http://lb.test/v1"] = True
    assert (await gpu.StaticProvisioner(b, Config(backends=[b], models=[])).status()).endpoint == "http://lb.test/v1"


def test_skypilot_task_publishes_kv_events_when_asked():
    model = ModelConfig(name="ow", hf_id="o/m", routes=[Route(backend="sky", model="m")])
    b = BackendConfig(name="sky", kind="self_hosted", format="openai", serve=ServeConfig(model="ow", kv_events=True, kv_events_port=6000))
    from meritorder.skypilot import Fleet, Identity, SkyPilotProvisioner

    prov = SkyPilotProvisioner(b, Config(backends=[b], models=[model]), Fleet(Identity("ab12", "s")))
    task = yaml.safe_load(yaml.safe_dump(prov.task_config(prov.cluster(2))))
    assert task["resources"]["ports"] == [8000, 6000]  # the gate, and the KV events
    assert "--enable-prompt-tokens-details" in task["run"]
    cfg = json.loads(task["run"].split("--kv-events-config ")[1].split()[0].strip("'"))
    assert cfg == {"enable_kv_cache_events": True, "publisher": "zmq", "endpoint": "tcp://*:6000"}
    b.serve.kv_events = False
    plain = prov.task_config(prov.cluster(1))
    assert plain["resources"]["ports"] == [8000] and "kv-events" not in plain["run"]


async def test_http_backend_sends_to_the_replica_with_the_session_tag():
    up = Upstream(httpx.Response(200, json=OPENAI_OK))
    b = wire(BackendConfig(name="gpu", kind="self_hosted", format="openai", base_url="http://lb.test/v1"), up)
    await b.send({"model": "m", "messages": []}, False, {}, base_url="http://r2.test/v1", session_tag="abc")
    assert str(up.requests[0].url) == "http://r2.test/v1/chat/completions" and up.requests[0].headers["x-session-id"] == "abc"
    await b.send({"model": "m", "messages": []}, False, {})
    assert str(up.requests[1].url) == "http://lb.test/v1/chat/completions" and "x-session-id" not in up.requests[1].headers
    from meritorder.backends import BackendError

    for err in (httpx.ConnectError("refused"), httpx.ConnectTimeout("slow")):
        down = wire(BackendConfig(name="gpu", kind="self_hosted", format="openai", base_url="http://lb.test/v1"), Upstream(err))
        with pytest.raises(BackendError) as e:
            await down.send({"model": "m", "messages": []}, False, {}, base_url="http://r1.test/v1")
        assert e.value.unreachable
    ant = Upstream(httpx.Response(200, json={"id": "m", "type": "message", "role": "assistant", "content": [], "usage": {}}))
    a = wire(BackendConfig(name="gpu-a", kind="self_hosted", format="anthropic", base_url="http://lb.test"), ant)
    await a.send({"model": "m", "messages": [], "max_tokens": 10}, False, {}, base_url="http://r1.test", session_tag="t")
    assert str(ant.requests[0].url) == "http://r1.test/v1/messages" and ant.requests[0].headers["x-session-id"] == "t"
