"""/metrics: time-to-first-token histogram, in-flight and availability gauges, and GPU state for dashboards."""

import re

from conftest import auth

from meritorder.backends.mock import MockBackend
from meritorder.config import BackendConfig


def series(text: str, name: str) -> dict[str, float]:
    return {m.group(1): float(m.group(2)) for m in re.finditer(rf"^{name}(\{{[^}}]*\}})? ([0-9.e+-]+)$", text, re.M)}


async def test_metrics_expose_ttft_inflight_availability_and_gpu_state(gateway, client):
    r = await client.post("/v1/chat/completions", headers=auth(),
                          json={"model": "small", "stream": True, "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    gpu = MockBackend(BackendConfig(name="gpu", kind="self_hosted", format="openai", mock_latency_ms=0))
    gpu.deploy = {"state": "warm", "replicas": 2, "burn_usd_per_hour": 8.7, "utilization": 0.64, "waiting": 3}
    gateway.backends["gpu"] = gpu
    hold = gateway.inflight.acquire("gpu", 60)
    await gateway.drain()

    text = (await client.get("/metrics")).text
    assert "# TYPE meritorder_time_to_first_token_seconds histogram" in text
    assert series(text, "meritorder_time_to_first_token_seconds_count")['{backend="mock-openai",model="small"}'] == 1
    assert "# TYPE meritorder_inflight_requests gauge" in text
    assert series(text, "meritorder_inflight_requests")['{backend="gpu"}'] == 1
    assert series(text, "meritorder_backend_available")['{backend="mock-openai"}'] == 1
    assert series(text, "meritorder_gpu_replicas")['{backend="gpu"}'] == 2
    assert series(text, "meritorder_gpu_usd_per_hour")['{backend="gpu"}'] == 8.7
    assert series(text, "meritorder_gpu_utilization")['{backend="gpu"}'] == 0.64
    assert series(text, "meritorder_gpu_waiting_requests")['{backend="gpu"}'] == 3

    hold()
    gpu.deploy = {"state": "api_only", "replicas": 0}
    gpu.available = False
    text = (await client.get("/metrics")).text
    assert series(text, "meritorder_gpu_replicas")['{backend="gpu"}'] == 0
    assert series(text, "meritorder_backend_available")['{backend="gpu"}'] == 0
    assert series(text, "meritorder_inflight_requests")['{backend="gpu"}'] == 0


async def test_requests_refused_before_any_backend_are_counted(gateway, client):
    await client.post("/v1/chat/completions", json={"model": "nope", "messages": [{"role": "user", "content": "x"}]}, headers=auth())
    await client.post("/v1/chat/completions", json={"model": "small", "messages": []}, headers={"authorization": "Bearer mo-wrong"})
    await client.post("/v1/embeddings", json={"model": "nope", "input": "x"}, headers=auth())
    text = (await client.get("/metrics")).text
    got = series(text, "meritorder_rejected_requests_total")
    assert got['{reason="not_found_error",status="404"}'] == 2
    assert got['{reason="authentication_error",status="401"}'] == 1
