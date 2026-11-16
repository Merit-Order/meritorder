"""Spend counters and the ledger: a counter behind the ledger (Redis restored from an old snapshot) is corrected by the
worker, a request that finishes while a counter is seeded isn't lost, a batch that can't be stored leaves no
reservation, a reservation counts in the month it settles, and the worker's jobs run once however many workers there are."""

from __future__ import annotations

import asyncio
import copy
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from conftest import TEST_KEY, auth, ledger_row

from meritorder import counters
from meritorder.app import build_gateway, create_app
from meritorder.auth import Principal
from meritorder.batches.service import month_key, spend_keys
from meritorder.config import build_config
from meritorder.proxy import _month_key
from meritorder.state import MemoryState, claim, release
from meritorder.store import MemoryStore


async def test_a_spend_counter_behind_the_ledger_is_corrected_by_the_worker(cfg):
    """docs/06: "Redis counters seeded from the Postgres ledger, which stays the source of truth". A Redis restored from a
    snapshot has a counter that exists but is behind: the worker finds it two cycles running and corrects it, and the
    budget binds again."""
    store, state = MemoryStore(), MemoryState()
    gw = build_gateway(cfg, store=store, state=state)
    team = cfg.team("default")  # user_budget_usd_month: 10
    p = Principal("alice", "default")
    month = _month_key()
    await store.record(ledger_row(user_id="alice", actual_cost=6.0))
    pre = await gw.admit(p, team, 10)
    assert pre["spend"][f"{month}:user:alice"] == 6.0
    snapshot = dict(state._spend)
    for _ in range(3):  # three more requests of $2 each: counter and ledger both say $12, over the $10 limit
        await store.record(ledger_row(user_id="alice", actual_cost=2.0))
        await state.record_request(None, None, gw.spend_keys(p), 2.0)
    with pytest.raises(Exception, match="monthly budget"):
        await gw.admit(p, team, 10)
    state._spend = snapshot  # Redis restarts from its snapshot: the counter exists again, at $6
    assert await counters.reconcile(store, state, cfg) == {}  # seen once: it could be a request between its counter and its row
    await release(state, counters.LOCK)
    fixed = await counters.reconcile(store, state, cfg)
    assert fixed[f"{month}:user:alice"] == pytest.approx(6.0) and fixed[f"{month}:team:default"] == pytest.approx(6.0)
    assert await state.get_spend(f"{month}:org:all") == pytest.approx(12.0)
    with pytest.raises(Exception, match="monthly budget"):
        await gw.admit(p, team, 10)


async def test_a_difference_seen_once_is_left_alone(store, state, cfg):
    """A request adds to its counters a moment before its ledger row is written: that difference is gone a cycle later,
    and correcting it would count the request twice."""
    month = month_key()
    await state.seed_spend(f"{month}:user:bob", 0.0)
    await state.record_request(None, None, [f"{month}:user:bob"], 1.5)  # counted; its row isn't written yet
    assert await counters.reconcile(store, state, cfg, lease_seconds=1) == {}
    await store.record(ledger_row(user_id="bob", actual_cost=1.5))  # now it is
    await release(state, counters.LOCK)
    assert await counters.reconcile(store, state, cfg, lease_seconds=1) == {}
    assert await state.get_spend(f"{month}:user:bob") == pytest.approx(1.5)


async def test_one_worker_reconciles_at_a_time():
    state = MemoryState()
    await state.seed_spend(f"{month_key()}:user:bob", 5.0)
    assert await claim(state, counters.LOCK, 60)
    assert await counters.reconcile(MemoryStore(), state, None) == {}  # another worker holds the lease: nothing is looked at
    assert await state.get_json(counters.SEEN) is None


async def test_a_request_that_finishes_while_its_counter_is_seeded_is_kept(store, state):
    """A request whose cost lands between the ledger read and the seed: added to the seed, not dropped."""
    key = f"{month_key()}:user:carol"
    await state.begin_seed(key)
    from_ledger = 4.0  # read before the request's row was written
    await state.adjust_spend([key, f"{month_key()}:team:none"], 0.5)  # the request finishes now
    assert await state.seed_spend(key, from_ledger) == pytest.approx(4.5)
    assert await state.seed_spend(key, 99.0) == pytest.approx(4.5)  # a second seeder reads the first one's value
    assert await state.get_spend(f"{month_key()}:team:none") is None  # a counter nobody is seeding stays missing
    await state.adjust_spend([key], 1.0)
    counted = await state.spend_counters(f"{month_key()}:")
    assert counted == {key: pytest.approx(5.5)}


async def test_claims_are_atomic_on_every_backing(state):
    got = await asyncio.gather(*[claim(state, "lock:x", 60) for _ in range(10)])
    assert got.count(True) == 1
    await release(state, "lock:x")
    await asyncio.sleep(1.1)
    assert await claim(state, "lock:x", 60)


async def test_a_batch_that_fails_to_be_stored_leaves_no_reservation(tmp_path, monkeypatch):
    """batches/service.py create(): the estimate is reserved in the spend counters before the batch row is written. A
    write that fails takes the reservation back, rather than leaving spend no batch will ever release."""
    from test_batch_service import CONFIG, lines

    cfg = build_config(copy.deepcopy(CONFIG), tmp_path)
    store, state = MemoryStore(), MemoryState()
    gw = build_gateway(cfg, store=store, state=state)
    month = _month_key()
    assert await gw.spend("user", "alice") == 0.0  # seeds the counters

    async def boom(*a, **kw):
        raise RuntimeError("postgres went away")

    monkeypatch.setattr(store, "create_batch", boom)
    transport = httpx.ASGITransport(app=create_app(gateway=gw), raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post("/v1/messages/batches", json={"requests": lines(("a", "small", "hi"), ("b", "small", "there"))}, headers=auth(TEST_KEY))
    assert r.status_code >= 500
    assert await store.batch_reserved(month, "user_id", "alice") == 0.0
    assert await state.get_spend(f"{month}:user:alice") == pytest.approx(0.0)


async def test_a_batch_reservation_counts_in_the_month_it_is_still_outstanding(store, cfg):
    """A batch made on October 31 that settles on November 1 is charged in November: November's budgets count its
    reservation until it does."""
    now = datetime.now(UTC)
    last = month_key(now.replace(day=1) - timedelta(days=1))
    await store.create_batch({"id": "batch_old", "created_at": now, "user_id": "alice", "team": "default", "client_format": "openai",
                              "status": "in_progress", "estimated_cost": 3.0, "reserved_cost": 3.0, "reserve_month": last,
                              "request_count": 1}, [], [])
    assert await store.batch_reserved(month_key(now), "user_id", "alice") == pytest.approx(3.0)
    assert await store.batch_reserved(last) == pytest.approx(3.0)
    assert await counters.counted(store, cfg, "team", "default") == pytest.approx(3.0)


async def test_a_batch_settled_in_the_next_month_moves_its_cost_there():
    from meritorder.batches.poller import BatchPoller, WorkerContext

    store, state = MemoryStore(), MemoryState()
    now = datetime.now(UTC)
    this, last = month_key(now), month_key(now.replace(day=1) - timedelta(days=1))
    for m in (this, last):
        for k in spend_keys("alice", "default", m):
            await state.seed_spend(k, 3.0)  # each month's counter holds the outstanding reservation
    b = {"id": "batch_old", "user_id": "alice", "team": "default", "reserve_month": last, "reserved_cost": 3.0}

    async def settle(batch_id, chunk):
        return 3.0, 2.5, []  # released, actual, the rows written

    store.settle_batch_lines = settle
    poller = BatchPoller(WorkerContext(store, state))
    poller.ctx.cfg = None
    import meritorder.batches.poller as poller_mod

    orig = poller_mod.ledger_row
    poller_mod.ledger_row = lambda *a, **kw: {}
    try:
        from meritorder.batches.providers import Outcome

        await poller.settle_lines(b, {"id": "j", "api": "anthropic"}, [({"idx": 0}, Outcome(key="r0", status="succeeded"))])
    finally:
        poller_mod.ledger_row = orig
    assert await state.get_spend(f"{this}:user:alice") == pytest.approx(2.5)  # the reservation out, what it cost in
    assert await state.get_spend(f"{last}:user:alice") == pytest.approx(0.0)


async def test_the_learned_router_trains_on_one_worker_at_a_time(cfg, monkeypatch):
    """Two workers that both see an admin's request to train would each train and save a version."""
    from meritorder import learned_jobs

    state, store = MemoryState(), MemoryStore()
    calls = []

    async def train(cfg, store, state, by, now=None):
        calls.append(by)
        await asyncio.sleep(0.05)
        return {"by": by}

    monkeypatch.setattr(learned_jobs, "train_and_save", train)
    await state.set_json(learned_jobs.TRAIN_REQUESTED, {"by": "root"})
    a, b = learned_jobs.RouterTrainer(), learned_jobs.RouterTrainer()
    results = await asyncio.gather(a.maybe(cfg, store, state), b.maybe(cfg, store, state))
    assert calls == ["console:root"] and results.count(None) == 1
    assert await state.get_json(learned_jobs.TRAIN_REQUESTED) is None
    assert await claim(state, learned_jobs.TRAIN_LOCK, 1)  # released when the training ended


async def test_two_workers_creating_the_same_partition_both_succeed(store):
    """The gateways (at start) and every worker create next months' ledger partitions: two at once mustn't fail."""
    await asyncio.gather(store.ensure_partitions(months_ahead=4), store.ensure_partitions(months_ahead=4))
