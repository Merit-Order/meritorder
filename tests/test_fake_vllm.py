"""The vLLM stand-in (scripts/fake_vllm.py) that the kind checks run where the Kubernetes provisioner expects vLLM: it takes
the exact `vllm serve` command the provisioner writes, and serves what the controller and gateways read."""

from __future__ import annotations

import importlib.util
import threading
from pathlib import Path

import httpx
import pytest

from meritorder import gpu
from meritorder.config import BackendConfig, Config, KubernetesServe, ModelConfig, Route, ServeConfig

spec = importlib.util.spec_from_file_location("fake_vllm", Path(__file__).parent.parent / "scripts" / "fake_vllm.py")
fake = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fake)


@pytest.fixture
def server():
    engine = fake.Engine("org/open-model", served="open-model")
    srv = fake.ThreadingHTTPServer(("127.0.0.1", 0), fake.handler(engine, "open-model"))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", engine
    srv.shutdown()
    srv.server_close()


def test_takes_the_command_the_provisioner_writes():
    serve = ServeConfig(provisioner="kubernetes", model="ow", accelerators="L4:2", vllm_args="--max-model-len 4096", kv_events=True,
                        kubernetes=KubernetesServe(cpu="4", memory="16Gi"))
    backend = BackendConfig(name="gpu", kind="self_hosted", format="openai", gpu_cost_per_hour=1.0, serve=serve)
    cfg = Config(backends=[backend], models=[ModelConfig(name="ow", hf_id="org/open-model", routes=[Route(backend="gpu", model="open-model")])])
    command = gpu.vllm_command(backend, cfg, 8000) + ["--max-model-len", "4096"]
    assert command[:2] == ["vllm", "serve"]
    args = fake.parse(command[1:])
    assert (args.model_tag, args.served_model_name, args.host, args.port) == ("org/open-model", ["open-model"], "0.0.0.0", 8000)
    assert "--enable-prefix-caching" in args.ignored and "--kv-events-config" in args.ignored
    assert fake.parse(["--port", "8095", "--model", "m"]).model == "m"  # the old command line still works
    assert fake.parse(["serve", "m", "--fake-exit", "3"]).fake_exit == 3
    with pytest.raises(SystemExit) as e:
        fake.main(["serve", "m", "--fake-exit", "3"])
    assert e.value.code == 3


def test_serves_models_chat_and_vllm_metrics(server):
    url, engine = server
    assert httpx.get(f"{url}/health").status_code == 200
    assert httpx.get(f"{url}/v1/models").json()["data"][0]["id"] == "open-model"
    first = [{"role": "system", "content": "a long system prompt " * 20}, {"role": "user", "content": "hi"}]
    r = httpx.post(f"{url}/v1/chat/completions", json={"model": "open-model", "messages": first, "priority": 5}).json()
    assert r["choices"][0]["message"]["content"] == "[open-model] You said: hi" and r["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
    more = first + [{"role": "assistant", "content": "hello"}, {"role": "user", "content": "more"}]
    with httpx.stream("POST", f"{url}/v1/chat/completions", json={"model": "open-model", "messages": more, "stream": True,
                                                                     "stream_options": {"include_usage": True}}) as s:
        lines = [x for x in s.iter_lines() if x]
    assert lines[-1] == "data: [DONE]" and '"cached_tokens": ' in lines[-2] and '"cached_tokens": 0' not in lines[-2]
    assert httpx.post(f"{url}/v1/completions", json={"model": "open-model", "prompt": "x"}).json()["choices"][0]["text"]
    assert httpx.post(f"{url}/v1/chat/completions", json={"model": "other", "messages": first}).status_code == 404
    load = gpu.vllm_load(httpx.get(f"{url}/metrics").text)
    assert load["kv_capacity_tokens"] == fake.BLOCK_SIZE * fake.GPU_BLOCKS and load["prefix_cache_hits"] > 0 and load["running"] == 0
    httpx.post(f"{url}/admin/utilization", params={"value": 0.95, "waiting": 3})
    assert gpu.vllm_load(httpx.get(f"{url}/metrics").text)["waiting"] == 3 and engine.priorities == [5, 0, 0]
    engine.ready_at = float("inf")  # still loading the model
    assert httpx.get(f"{url}/health").status_code == 503 and httpx.get(f"{url}/v1/models").status_code == 503
