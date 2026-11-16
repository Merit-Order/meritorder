from datetime import UTC, datetime

import yaml

from meritorder.config import BackendConfig, Config, ModelConfig, Route, ServeConfig
from meritorder.controller import tick
from meritorder.gpu import (
    API_ONLY, DRAINING, PROVISIONING, WARM, Demand, Deployment, Status, addressable, decide, desired_replicas, parse_vllm_metrics,
)
from meritorder.skypilot import Fleet, Identity, SkyPilotProvisioner
from meritorder.state import MemoryState
from meritorder.store import MemoryStore

SERVE = ServeConfig(model="ow", scale_up_after_minutes=10, hold_minutes=30, drain_minutes=15, max_replicas=4)


def test_desired_replicas_must_pay_for_themselves():
    # $50/h of API spend can pay for 2 replicas at $20/h (with 20% margin), throughput needs 3.
    d = desired_replicas(SERVE, 20.0, 1000, Demand(usd_per_hour=50, tokens_per_second=2000))
    assert d == 2
    assert desired_replicas(SERVE, 20.0, 1000, Demand(usd_per_hour=10, tokens_per_second=2000)) == 0
    warm = ServeConfig(model="ow", min_replicas=1)
    assert desired_replicas(warm, 20.0, 1000, Demand(0, 0)) == 1


def test_state_machine_full_cycle():
    t = 1_000_000.0
    rec, act = decide({}, SERVE, desired=2, status=Status(), now=t)
    assert rec["state"] == API_ONLY and act is None  # must be justified for 10 minutes first
    rec, act = decide(rec, SERVE, 2, Status(), t + 601)
    assert rec["state"] == PROVISIONING and act.kind == "up" and act.replicas == 2
    rec, act = decide(rec, SERVE, 2, Status(ready=True, endpoint="http://gpu/v1"), t + 1500)
    assert rec["state"] == WARM and rec["endpoint"] == "http://gpu/v1"
    rec, act = decide(rec, SERVE, 3, Status(ready=True, endpoint="http://gpu/v1"), t + 1600)
    assert act.kind == "scale" and act.replicas == 3
    # demand disappears, but GPUs are held for 30 minutes after warming
    rec, act = decide(rec, SERVE, 0, Status(ready=True, endpoint="http://gpu/v1"), t + 1700)
    assert rec["state"] == WARM
    rec, act = decide(rec, SERVE, 0, Status(ready=True, endpoint="http://gpu/v1"), t + 1500 + 1801)
    assert rec["state"] == DRAINING
    rec, act = decide(rec, SERVE, 0, Status(ready=True, endpoint="http://gpu/v1"), t + 1500 + 1801 + 901)
    assert rec["state"] == API_ONLY and act.kind == "down"


def test_provisioning_timeout_cools_down():
    t = 0.0
    rec = {"state": PROVISIONING, "since": t}
    rec, act = decide(rec, SERVE, 2, Status(), t + SERVE.provision_timeout_minutes * 60 + 1)
    assert rec["state"] == API_ONLY and act.kind == "down"
    rec, act = decide(rec, SERVE, 2, Status(), t + SERVE.provision_timeout_minutes * 60 + 700)
    assert act is None  # still cooling down


def test_addressable_demand_uses_quality_floor():
    cfg = Config(backends=[], models=[ModelConfig(name="ow", quality={"coding": 4}, routes=[Route(backend="x", model="y")])])
    rows = [
        {"floor": "coding:3", "served_model": "sonnet", "requested_model": "auto", "cost": 10.0, "tokens": 100},
        {"floor": "coding:5", "served_model": "opus", "requested_model": "auto", "cost": 50.0, "tokens": 500},
        {"floor": None, "served_model": "ow", "requested_model": "ow", "cost": 2.0, "tokens": 20},
    ]
    assert addressable(cfg.models[0], rows, cfg) == (12.0, 120)


def test_parse_vllm_metrics():
    text = """# HELP vllm:kv_cache_usage_perc KV-cache usage
vllm:kv_cache_usage_perc{model_name="m"} 0.42
vllm:num_requests_waiting{model_name="m"} 0.0
"""
    assert parse_vllm_metrics(text) == 0.42
    assert parse_vllm_metrics(text.replace("waiting{model_name=\"m\"} 0.0", "waiting{model_name=\"m\"} 4")) > 0.85


def test_skypilot_task():
    model = ModelConfig(name="ow", hf_id="Org/Model-7B", routes=[Route(backend="gpu", model="org-model")])
    backend = BackendConfig(name="gpu", kind="self_hosted", format="openai",
                            serve=ServeConfig(model="ow", accelerators="L4:2", clouds=["aws", "gcp"], use_spot=True, vllm_args="--max-model-len 32768"))
    cfg = Config(backends=[backend], models=[model])
    prov = SkyPilotProvisioner(backend, cfg, Fleet(Identity("ab12", "s3cret")))
    task = prov.task_config(prov.cluster(2))
    assert task["name"] == "mo-ab12-gpu-2" and yaml.safe_load(yaml.safe_dump(task)) == task
    assert task["resources"]["accelerators"] == "L4:2" and task["resources"]["use_spot"] is True
    assert task["resources"]["any_of"] == [{"infra": "aws", "ports": [8000]}, {"infra": "gcp", "ports": [8000]}]
    assert "vllm serve Org/Model-7B" in task["run"] and "--host 127.0.0.1 --port 8001" in task["run"]
    assert "--served-model-name org-model" in task["run"] and "--tensor-parallel-size 2" in task["run"]
    assert "--enable-prefix-caching" in task["run"] and task["run"].rstrip().endswith("--max-model-len 32768")


class FakeProvisioner:
    def __init__(self):
        self.calls = []
        self.ready = False

    async def status(self):
        return Status(ready=self.ready, endpoint="http://127.0.0.1:9/v1" if self.ready else None)

    async def up(self, n):
        self.calls.append(("up", n))

    async def scale(self, n):
        self.calls.append(("scale", n))

    async def down(self):
        self.calls.append(("down",))


async def test_controller_tick_scales_up_on_addressable_demand():
    model = ModelConfig(name="ow", quality={"coding": 4}, hf_id="o/m", routes=[Route(backend="gpu", model="m")])
    backend = BackendConfig(name="gpu", kind="self_hosted", format="openai", gpu_cost_per_hour=10.0,
                            serve=ServeConfig(model="ow", scale_up_after_minutes=0))
    cfg = Config(backends=[backend], models=[model])
    store, state = MemoryStore(), MemoryState()
    now = datetime.now(UTC)
    for _ in range(30):  # $30 of coding:3 traffic in the last hour, served by an API model
        await store.record({"ts": now, "floor": "coding:3", "served_model": "sonnet", "requested_model": "auto",
                            "served_api_cost": 1.0, "actual_cost": 1.0, "status": 200, "upgraded": False,
                            "input_tokens": 1000, "output_tokens": 100})
    prov = FakeProvisioner()
    deps = [Deployment(backend=backend, model=model, provisioner=prov)]
    [rec] = await tick(cfg, store, state, deps)
    assert rec["state"] == PROVISIONING and prov.calls == [("up", 1)]  # ~9 tok/s fits one replica
    prov.ready = True
    [rec] = await tick(cfg, store, state, deps)
    assert rec["state"] == WARM and rec["endpoint"].startswith("http://127.0.0.1")


def test_simulate_whatif():
    from datetime import UTC, datetime, timedelta

    from meritorder.gpu import simulate

    model = ModelConfig(name="ow", quality={"coding": 4}, routes=[Route(backend="x", model="y")])
    cfg = Config(backends=[], models=[model])
    t0 = datetime(2026, 11, 2, 9, tzinfo=UTC)
    rows = []
    for h in range(10):  # 8 busy hours at $50/h of coding:3 traffic, 2 quiet hours at $2/h
        cost = 50.0 if h < 8 else 2.0
        rows.append({"hour": t0 + timedelta(hours=h), "floor": "coding:3", "served_model": "sonnet", "requested_model": "auto",
                     "cost": cost, "tokens": 3_600_000 if h < 8 else 10_000})
        rows.append({"hour": t0 + timedelta(hours=h), "floor": "coding:5", "served_model": "opus", "requested_model": "auto",
                     "cost": 100.0, "tokens": 1})  # too hard for this model: never counted
    r = simulate(model, cfg, rows, gpu_cost_per_hour=16.0, capacity_tps=4000)
    assert r["hours_gpus_would_run"] == 8 and r["peak_replicas"] == 1
    assert r["addressable_api_cost"] == 404.0
    assert r["net_savings"] == 8 * (50 - 16)


def test_sky_python_found_next_to_sky_even_with_spaces_and_symlinks(tmp_path, monkeypatch):
    import os

    from meritorder.gpu import _shebang_python, sky_python

    monkeypatch.delenv("MERITORDER_SKY_PYTHON", raising=False)
    venv = tmp_path / "Example LLC" / "sky-venv" / "bin"
    venv.mkdir(parents=True)
    py = venv / "python"
    py.write_text("#!/bin/sh\n")
    py.chmod(0o755)
    sky = venv / "sky"
    sky.write_text(f"#!/bin/sh\n'''exec' \"{py}\" \"$0\" \"$@\"\n' '''\n")
    sky.chmod(0o755)
    links = tmp_path / "bin"
    links.mkdir()
    os.symlink(sky, links / "sky")  # like ~/.local/bin/sky -> the tool's venv
    monkeypatch.setenv("PATH", str(links))
    assert sky_python() == str(py)
    monkeypatch.setenv("MERITORDER_SKY_PYTHON", "/explicit/python")
    assert sky_python() == "/explicit/python"
    # Without a python next to it, the launcher itself says which interpreter it runs.
    assert _shebang_python(sky.read_text().splitlines()) == str(py)
    assert _shebang_python(["#!/usr/local/bin/python3.12"]) == "/usr/local/bin/python3.12"
    assert _shebang_python(["#!/bin/bash", "echo"]) is None and _shebang_python([]) is None


def test_breakeven_by_level_and_by_model():
    from datetime import timedelta

    from meritorder.gpu import breakeven, need, token_mix

    flat = lambda usd: {"input": usd, "output": usd, "cache_read": usd, "cache_write": usd}  # noqa: E731 (same price for every token)
    api = BackendConfig(name="api", format="openai")
    free = BackendConfig(name="free", format="openai")
    gpu = BackendConfig(name="gpu", kind="self_hosted", format="openai", gpu_cost_per_hour=10.0, capacity_tokens_per_second=1000,
                        serve=ServeConfig(model="open", provisioner="static", static_url="http://gpu.test/v1"))
    cfg = Config(backends=[api, free, gpu], models=[
        ModelConfig(name="frontier", quality={"coding": 5}, routes=[Route(backend="api", model="f", price=flat(10))]),
        ModelConfig(name="cheap", quality={"coding": 2}, routes=[Route(backend="api", model="c", price=flat(0.1)),
                                                                Route(backend="free", model="c:free", price=flat(0))]),
        ModelConfig(name="open", hf_id="org/open", quality={"coding": 4}, routes=[Route(backend="gpu", model="o", price=flat(1)),
                                                                                  Route(backend="api", model="o", price=flat(1))]),
        ModelConfig(name="open-small", hf_id="org/small", quality={"coding": 3}, routes=[Route(backend="api", model="s", price=flat(2))]),
        ModelConfig(name="huge", hf_id="org/huge", quality={"coding": 5}, routes=[Route(backend="api", model="h", price=flat(20))]),
        ModelConfig(name="unscored", hf_id="org/u", routes=[Route(backend="api", model="u", price=flat(0.5))]),
    ])
    plans = {
        # Its own deployment: $10/h, 1,000 tok/s measured.
        "open": {"deployment": "gpu", "replica": {"usd_per_hour": 10.0, "source": "configured"}, "size": None, "capacity_tps": 1000},
        # Sized: one GPU at $1/h with 100 TFLOPS; 1B active parameters, so 10% of peak is 5,000 tok/s.
        "open-small": {"deployment": None, "replica": {"usd_per_hour": 1.0, "tflops": 100.0, "accelerator": "L4", "count": 1, "cloud": "aws",
                                                       "spot": True, "source": "catalog"},
                       "size": {"active_params": 1e9, "weight_bytes": 2e9, "total_params": 1e9, "basis": "test", "bits": 4, "published_bits": 16},
                       "capacity_tps": None},
        "huge": {"error": "needs 9,999 GB of GPU memory"},
        "unscored": {"error": "its size hasn't been fetched yet"},
    }
    t0 = datetime(2026, 11, 2, 9, tzinfo=UTC)
    rows = [{"hour": t0, "floor": "coding:4", "served_model": "open", "requested_model": "auto", "cost": 0.6, "tokens": 600_000,
             "requests": 30, "input": 600_000, "cache_read": 0, "cache_write": 0, "output": 0},
            {"hour": t0 + timedelta(hours=1), "floor": "coding:4", "served_model": "open", "requested_model": "auto", "cost": 0.06,
             "tokens": 60_000, "requests": 3, "input": 60_000, "cache_read": 0, "cache_write": 0, "output": 0}]
    r = breakeven(cfg, {"api", "free", "gpu"}, rows, hours=10, plans=plans, utilization=0.5, margin=0.0, efficiency=0.1)
    assert r["mix"]["source"] == "traffic" and r["mix"]["input"] == 1.0 and r["mix"]["tokens_per_request"] == 20_000
    g = {x["floor"]: x for x in r["groups"]}
    # Level 3: the cheaper-per-token open model is the small one ($1/h for 9M tokens an hour when half full: $0.11/M), against
    # the cheapest API model good enough ($1/M): it pays from 1M tokens an hour.
    assert g["coding:3"]["gpu"]["model"] == "open-small" and g["coding:3"]["gpu"]["capacity_basis"] == "estimated"
    assert g["coding:3"]["gpu"]["capacity_tokens_per_second"] == 5000 and g["coding:3"]["gpu_usd_per_million"] == 0.1111
    assert g["coding:3"]["tokens_per_minute"] == 16_667 and g["coding:3"]["pays"] and g["coding:3"]["requests_per_hour"] == 50.0
    # Level 4: only its own deployment qualifies ($5.56/M at half load), against $1/M on APIs: never.
    assert g["coding:4"]["gpu"]["model"] == "open" and g["coding:4"]["gpu"]["deployment"] == "gpu" and not g["coding:4"]["pays"]
    assert g["coding:4"]["traffic"] == {"tokens_per_minute": 1100.0, "peak_tokens_per_minute": 10_000.0, "requests_per_hour": 3.3,
                                        "usd_per_hour": 0.066}
    # Level 2: $0.10/M on APIs (the free tier doesn't count); the small one needs 10M tokens an hour, more than it can serve.
    assert g["coding:2"]["cheapest"]["model"] == "cheap" and g["coding:2"]["tokens_per_minute"] == 166_667 and not g["coding:2"]["pays"]
    assert g["coding:2"]["open_models"] == ["huge", "open", "open-small"]
    # Level 5: the only open model that qualifies can't run on one machine.
    assert g["coding:5"]["gpu"] is None and g["coding:5"]["why"] == "none of the open models that qualify can be run: huge needs 9,999 GB of GPU memory"
    assert g["coding:5"]["tokens_per_minute"] is None and g["coding:5"]["api_usd_per_million"] == 10.0
    m = {x["model"]: x for x in r["models"]}
    # Its traffic cost $1/M; its $10/h deployment needs 10M tokens an hour and can serve 1.8M.
    assert m["open"]["api_usd_per_million"] == 1.0 and m["open"]["tokens_per_minute"] == 166_667 and not m["open"]["pays"]
    assert m["open"]["gpu"]["capacity_basis"] == "configured" and m["open"]["own_api_usd_per_million"] == 1.0
    assert m["open-small"]["basis"] == "no traffic yet: the cheapest API model at coding:3" and m["open-small"]["pays"]
    assert m["open-small"]["size"]["active_params"] == 1e9
    # Served in 4 bits, published in 16: its quality scores are for the published weights, and it says so.
    assert m["open-small"]["size"]["weights"] == "4-bit" and m["open-small"]["size"]["published_weights"] == "16-bit"
    assert "measure it with an eval" in m["open-small"]["size"]["quality_note"] and g["coding:3"]["gpu"]["quality_note"]
    assert m["open"]["size"] is None and g["coding:4"]["gpu"]["weights"] is None
    # The best coding level you can host: the open model at 4 runs on its deployment; the 5 doesn't fit anywhere.
    assert r["hostable"]["coding"] | {"gpu": None} == {"level": 4, "model": "open", "quality": 4.0, "gpu": None, "better_unrunnable": {
        "model": "huge", "quality": 5.0, "why": "needs 9,999 GB of GPU memory"}}
    assert m["huge"]["why"] == "needs 9,999 GB of GPU memory" and m["huge"]["tokens_per_minute"] is None
    assert "no quality score" in m["unscored"]["basis"] and m["unscored"]["gpu"] is None
    assert [x["model"] for x in r["models"]][0] == "open-small"  # ones that pay first
    assert breakeven(cfg, {"api"}, [], 1, plans)["mix"]["source"] == "assumed"
    # Two machines it could have: the cheaper one (100 tok/s) can't serve the 500,000 tokens an hour it would need at
    # level 3's $1/M, the faster one can, so that's the one; at level 2 neither pays, so the cheaper per token.
    slow = {"usd_per_hour": 0.5, "tflops": 2.0, "accelerator": "T4", "count": 1, "source": "catalog"}
    fast = {"usd_per_hour": 1.0, "tflops": 100.0, "accelerator": "L4", "count": 1, "source": "catalog"}
    two = {**plans, "open-small": {**plans["open-small"], "replica": slow, "options": [slow, fast]}}
    r2 = breakeven(cfg, {"api", "free", "gpu"}, rows, hours=10, plans=two, utilization=0.5, margin=0.0, efficiency=0.1)
    g2 = {x["floor"]: x for x in r2["groups"]}
    assert g2["coding:3"]["gpu"]["accelerator"] == "L4" and g2["coding:3"]["pays"] and g2["coding:3"]["tokens_per_minute"] == 16_667
    assert g2["coding:2"]["gpu"]["accelerator"] == "L4" and not g2["coding:2"]["pays"]
    # Neither pays at level 2: eight GPUs are 6% cheaper per token than one, so the one shows (the same verdict).
    big = {"usd_per_hour": 8.0, "tflops": 850.0, "accelerator": "L4", "count": 8, "source": "catalog"}
    near = {**plans, "open-small": {**plans["open-small"], "replica": fast, "options": [fast, big]}}
    g4 = {x["floor"]: x for x in breakeven(cfg, {"api", "free", "gpu"}, rows, 10, near, 0.5, 0.0, 0.1)["groups"]}
    assert g4["coding:2"]["gpu"]["count"] == 1 and not g4["coding:2"]["pays"]
    only_slow = {**plans, "open-small": {**plans["open-small"], "replica": slow, "options": [slow]}}
    g3 = {x["floor"]: x for x in breakeven(cfg, {"api"}, rows, 10, only_slow, 0.5, 0.0, 0.1)["groups"]}
    # On its one slow machine it can't pay ($2.78/M full), nor can the other open model ($5.56/M): the cheaper per token.
    assert (g3["coding:3"]["gpu"]["model"], g3["coding:3"]["gpu"]["accelerator"], g3["coding:3"]["pays"]) == ("open-small", "T4", False)
    assert token_mix(rows[1:], min_requests=20) is None and token_mix([]) is None
    assert need(16, 4000, 0.7, 0.2, None, None) | {"gpu_usd_per_million": 0} == {
        "capacity_tokens_per_minute": 168_000, "gpu_usd_per_million": 0, "api_usd_per_million": None, "tokens_per_minute": None,
        "requests_per_hour": None, "pays": False}


def test_catalog_lookups_never_use_cloud_credentials(monkeypatch):
    import os

    from meritorder.gpu import _catalog_env

    monkeypatch.setenv("AWS_PROFILE", "some-other-project")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAEXAMPLE")
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "token")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    env = _catalog_env()
    assert not {"AWS_PROFILE", "AWS_ACCESS_KEY_ID", "AWS_BEARER_TOKEN_BEDROCK"} & set(env) and env["AWS_REGION"] == "us-east-1"
    assert env["AWS_CONFIG_FILE"] == os.devnull and env["AWS_SHARED_CREDENTIALS_FILE"] == os.devnull
