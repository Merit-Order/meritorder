"""Saturated GPUs: spill to other routes, queue as a last resort (bounded), and see load within seconds."""

import time

import pytest
from conftest import auth

from meritorder.backends.mock import MockBackend
from meritorder.config import BackendConfig, ModelConfig, Route
from meritorder.gpu import vllm_load
from meritorder.router import RoutingError
from meritorder.sessions import Session
from meritorder.state import MemoryState, _live_inflight

BODY = {"messages": [{"role": "user", "content": "hi"}]}


def add_gpu(gateway, max_concurrency=None, max_waiting=32, api_route=False):
    """A self-hosted (in-process) backend serving model 'gpu-only', optionally also on the mock API."""
    b = MockBackend(BackendConfig(name="gpu", kind="self_hosted", format="openai", mock_latency_ms=0,
                                  max_concurrency=max_concurrency, max_waiting=max_waiting))
    gateway.backends["gpu"] = b
    routes = [Route(backend="gpu", model="m", price={"input": 1, "output": 1})]
    if api_route:
        routes.append(Route(backend=gateway.cfg.models[0].routes[0].backend, model="m", price={"input": 1, "output": 1}))
    gateway.cfg.models.append(ModelConfig(name="gpu-only", quality={"coding": 5, "general": 5}, routes=routes))
    return b


def decide(gateway, model, util, queue_full=frozenset(), session=None, boundary=True):
    return gateway.router.decide(model, gateway.cfg.team("default"), BODY, {}, session, 10, boundary, util, time.time(), queue_full)


def test_saturated_gpu_spills_to_api_route_first(gateway):
    add_gpu(gateway, api_route=True)
    d = decide(gateway, "gpu-only", {"gpu": 1.0})
    assert [c.backend.name for c in d.candidates] == [gateway.cfg.models[0].routes[0].backend]
    assert not d.candidates[0].queued


def test_only_route_saturated_queues_until_the_queue_is_full(gateway):
    add_gpu(gateway)
    d = decide(gateway, "gpu-only", {"gpu": 1.0})
    assert [c.backend.name for c in d.candidates] == ["gpu"] and d.candidates[0].queued
    assert "every route is saturated, queued on gpu" in d.reason
    with pytest.raises(RoutingError) as e:
        decide(gateway, "gpu-only", {"gpu": 1.0}, queue_full={"gpu"})
    assert e.value.status == 503 and "saturated and its queue is full" in e.value.message


def test_auto_session_on_full_gpu_moves_to_a_model_with_room(gateway):
    add_gpu(gateway)
    s = Session(id="s")
    s.model, s.backend, s.turns, s.floor = "gpu-only", "gpu", 3, "coding:3"
    d = decide(gateway, "auto", {"gpu": 1.0}, session=s, boundary=False)
    assert d.candidates[0].model.name != "gpu-only" and not d.candidates[0].queued
    assert "saturated; re-routed" in d.reason


def test_gateway_load_counts_inflight_across_gateways(gateway):
    add_gpu(gateway, max_concurrency=2, max_waiting=2)
    releases = [gateway.inflight.acquire("gpu", 600) for _ in range(2)]
    pre = {"util": {"gpu": 0.1}, "inflight": {"other-gw": {"gpu": 1}, gateway.gateway_id: {"gpu": 99}}}
    util, full = gateway.load(pre)
    assert util["gpu"] == 1.5 and not full  # 2 local + 1 elsewhere over 2 slots; our own published count is ignored
    pre["inflight"]["other-gw"]["gpu"] = 2
    util, full = gateway.load(pre)
    assert full == {"gpu"}  # 4 in flight, 2 running, 2 waiting = max_waiting
    gateway.backends["gpu"].replicas = 2  # twice the slots
    assert gateway.load(pre) == ({"gpu": 1.0}, set())
    for r in releases:
        r()
        r()  # releasing twice is harmless
    assert gateway.inflight.count("gpu") == 0


def test_inflight_entries_expire():
    from meritorder.proxy import InFlight

    f = InFlight()
    f.acquire("gpu", -1)  # already past its deadline: a leaked release
    f.acquire("gpu", 60)
    assert f.count("gpu") == 1 and f.counts() == {"gpu": 1}


def test_without_max_concurrency_the_controller_waiting_count_bounds_the_queue(gateway):
    b = add_gpu(gateway, max_waiting=4)
    b.reported_waiting = 3
    assert gateway.load({"util": {"gpu": 1.0}})[1] == set()
    b.reported_waiting = 4
    assert gateway.load({"util": {"gpu": 1.0}})[1] == {"gpu"}


def test_without_a_recent_sample_the_gateways_in_flight_count_bounds_the_queue(gateway):
    """The controller's sample is over 5 minutes old (reported_waiting None): every request in flight there may be waiting,
    so the queue stays bounded instead of never counting as full."""
    b = add_gpu(gateway, max_waiting=4)
    b.reported_waiting = None
    releases = [gateway.inflight.acquire("gpu", 600) for _ in range(3)]
    assert gateway.load({"util": {"gpu": 1.0}})[1] == set()
    assert gateway.load({"util": {"gpu": 1.0}, "inflight": {"other-gw": {"gpu": 1}}})[1] == {"gpu"}
    for r in releases:
        r()


async def test_end_to_end_queue_header_then_503_and_slots_released(gateway, client):
    add_gpu(gateway, max_concurrency=1, max_waiting=1)
    body = {"model": "gpu-only", **BODY}
    hold = gateway.inflight.acquire("gpu", 600)  # one long request already running
    r = await client.post("/v1/chat/completions", json=body, headers=auth())
    assert r.status_code == 200 and r.headers["x-meritorder-queued"] == "1"
    r = await client.post("/v1/chat/completions", json={**body, "stream": True}, headers=auth())
    assert r.status_code == 200 and r.headers["x-meritorder-queued"] == "1"
    assert gateway.inflight.count("gpu") == 1  # both finished and released their slots
    hold2 = gateway.inflight.acquire("gpu", 600)  # now one running, one waiting: the queue is full
    r = await client.post("/v1/chat/completions", json=body, headers=auth())
    assert r.status_code == 503 and "queue is full" in r.json()["error"]["message"]
    hold(), hold2()
    r = await client.post("/v1/chat/completions", json=body, headers=auth())
    assert r.status_code == 200 and "x-meritorder-queued" not in r.headers


async def test_state_shares_inflight_and_forgets_dead_gateways():
    s = MemoryState()
    await s.publish_inflight("a", {"gpu": 3})
    await s.publish_inflight("b", {"gpu": 0})
    pre = await s.admit([], [], None)
    assert pre["inflight"] == {"a": {"gpu": 3}}
    assert _live_inflight({"a|gpu": "3", "old|gpu": "5"}, {"a": time.time(), "old": time.time() - 60}) == {"a": {"gpu": 3}}


def test_vllm_load_reports_waiting_and_running():
    text = ('vllm:kv_cache_usage_perc{model_name="m"} 0.5\nvllm:num_requests_waiting{model_name="m"} 12\n'
            'vllm:num_requests_running{model_name="m"} 40\n')
    assert vllm_load(text) == {"utilization": 1.0, "waiting": 12, "running": 40}
