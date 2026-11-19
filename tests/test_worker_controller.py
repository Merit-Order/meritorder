"""Background processes: the worker cycle, the capacity controller loop, GPU provisioners, alerts, and app startup."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import shlex
import sys
import time
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import httpx
import pytest
import yaml
from conftest import TEST_CONFIG, auth
from test_gpu import FakeProvisioner

from meritorder import alerts, configstore, controller, gpu, telemetry, worker
from meritorder import app as app_module
from meritorder.app import build_gateway, create_app
from meritorder.config import BackendConfig, Config, ModelConfig, OIDCConfig, Route, Safeguards, ServeConfig, TeamConfig
from meritorder.configstore import ConfigError, ConfigFollower, load_active, parse_and_validate, summarize, to_yaml
from meritorder.gpu import API_ONLY, WARM, Deployment, StaticProvisioner
from meritorder.skypilot import SkyPilotProvisioner
from meritorder.state import MemoryState
from meritorder.store import MemoryStore, month_start


REAL_ASYNC_CLIENT = httpx.AsyncClient
EMPTY = {"backends": [], "models": []}


class StopLoop(Exception):
    pass


class FakePool:
    """Stands in for the asyncpg pool the controller takes its advisory lock on."""

    def __init__(self, locked: bool = True):
        self.locked = locked
        self.held = locked  # what pg_locks says about this connection
        self.released = 0
        self.acquired = 0

    async def acquire(self):
        self.acquired += 1
        return self

    async def release(self, conn):
        self.released += 1

    async def fetchval(self, sql, *args):
        assert args == (controller.LOCK_ID,)
        if sql == controller.LOCK_HELD:
            if isinstance(self.held, Exception):
                raise self.held
            return self.held
        assert "pg_try_advisory_lock" in sql
        self.held = self.locked
        return self.locked


def pg_memory_store(locked: bool = True) -> MemoryStore:
    store = MemoryStore()
    store.kind = "postgres"  # the worker and controller refuse anything else
    store.pool = FakePool(locked)
    return store


def gpu_config(provisioner: str = "static", **serve) -> Config:
    model = ModelConfig(name="ow", quality={"coding": 4}, hf_id="o/m", routes=[Route(backend="gpu", model="m")])
    backend = BackendConfig(name="gpu", kind="self_hosted", format="openai", gpu_cost_per_hour=2.0,
                            serve=ServeConfig(model="ow", provisioner=provisioner, static_url="http://gpu.test/v1", **serve))
    return Config(backends=[backend], models=[model])


def mock_http(monkeypatch, module, handler):
    monkeypatch.setattr(module.httpx, "AsyncClient", lambda **kw: REAL_ASYNC_CLIENT(transport=httpx.MockTransport(handler), **kw))


async def ledger_row(store, cost: float, team: str = "eng", user: str = "bob", ts: datetime | None = None, **extra) -> None:
    await store.record({"ts": ts or datetime.now(UTC), "team": team, "user_id": user, "actual_cost": cost, "baseline_cost": cost * 2,
                        "served_api_cost": cost, "status": 200, "served_model": "small", "requested_model": "auto",
                        "savings_routing": cost, "input_tokens": 100, "output_tokens": 10, **extra})


# ------------------------------------------------------------------ worker


async def test_worker_needs_postgres(monkeypatch):
    monkeypatch.delenv("MERITORDER_DATABASE_URL", raising=False)
    with pytest.raises(SystemExit, match="needs Postgres"):
        await worker.run_worker(once=True, cfg=Config(**EMPTY))


async def test_worker_cycle_rolls_up_alerts_and_sends_the_monday_digest(monkeypatch):
    store, state = pg_memory_store(), MemoryState()
    monkeypatch.setattr(worker, "make_store", lambda url: store)
    monkeypatch.setattr(worker, "make_state", lambda url: state)
    rollups, digests = [], []

    async def rollup(full=False):
        rollups.append(full)
        return 3

    async def digest(cfg, store_, state_, now):
        digests.append(now)
        return []

    monkeypatch.setattr(store, "rollup", rollup)
    monkeypatch.setattr(worker, "weekly_digest", digest)

    class Monday(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 11, 9, 10, 0, tzinfo=UTC)  # a Monday, after 09:00

    monkeypatch.setattr(worker, "datetime", Monday)
    cfg = Config(**EMPTY, teams=[TeamConfig(name="eng", budget_usd_month=10, user_budget_usd_month=5)],
                 safeguards=Safeguards(org_budget_usd_month=10, gpu_budget_usd_month=1))
    for _ in range(2):
        await ledger_row(store, 6.0)
    await store.add_gpu_usage("gpu", datetime.now(UTC), 3600, 2.0)
    await worker.run_worker(once=True, cfg=cfg, backfill=True)

    assert rollups == [True] and digests == [Monday.now()]
    titles = {a["title"] for a in await state.recent_alerts()}
    assert {"Team eng at 80% of budget", "Team eng at 100% of budget", "bob at 100% of personal budget",
            "Organization spend at 100% of budget", "GPU spend at 100% of budget", "Spend spike: bob"} <= titles

    # A failing cycle is logged and the loop carries on (here: returns, since once=True).
    async def broken(full=False):
        raise RuntimeError("database went away")

    monkeypatch.setattr(store, "rollup", broken)
    await worker.run_worker(once=True, cfg=cfg)


# ------------------------------------------------------------------ alerts


async def test_weekly_digest_once_per_week_with_causes(monkeypatch):
    store, state = MemoryStore(), MemoryState()
    now = datetime(2026, 11, 9, 10, 0, tzinfo=UTC)
    await ledger_row(store, 4.0, ts=now - timedelta(days=1), savings_caching=1.5)
    [msg] = await alerts.weekly_digest(Config(**EMPTY), store, state, now)
    assert "Weekly LLM spend for 'eng': $4.00 across 1 requests" in msg and "routing $4.00" in msg and "caching $1.50" in msg
    assert await alerts.weekly_digest(Config(**EMPTY), store, state, now) == []  # already sent this week
    assert len(await alerts.weekly_digest(Config(**EMPTY), store, state, now, force=True)) == 1


async def test_notify_posts_to_the_webhook(monkeypatch):
    posted = []

    def handler(request):
        posted.append(json.loads(request.content))
        return httpx.Response(500 if b"fail" in request.content else 200)

    mock_http(monkeypatch, alerts, handler)
    state = MemoryState()
    assert await alerts.notify("GPU tier up", url="https://hooks.test/x", state=state, title="GPU", page="gpus") is True
    assert posted == [{"text": "GPU tier up", "content": "GPU tier up"}]  # Slack/Teams read text, Discord content
    assert await alerts.notify("fail please", url="https://hooks.test/x") is False
    monkeypatch.delenv("MERITORDER_ALERT_WEBHOOK_URL", raising=False)
    assert await alerts.notify("no webhook") is False
    [alert] = await state.recent_alerts()
    assert alert["title"] == "GPU" and alert["page"] == "gpus"

    def down(request):
        raise httpx.ConnectError("refused")

    mock_http(monkeypatch, alerts, down)
    assert await alerts.notify("unreachable", url="https://hooks.test/x") is False


# ------------------------------------------------------------------ controller


@pytest.fixture
def no_metrics(monkeypatch):
    loads = []

    async def fetch_load(endpoint, headers=None):
        loads.append(endpoint)
        return {"utilization": 0.5, "waiting": 3, "running": 7}

    monkeypatch.setattr(controller, "fetch_load", fetch_load)
    return loads


def install(monkeypatch, store, state):
    monkeypatch.setattr(controller, "make_store", lambda url: store)
    monkeypatch.setattr(controller, "make_state", lambda url: state)


async def test_controller_once_publishes_state_load_and_heartbeat(monkeypatch, no_metrics):
    store, state = pg_memory_store(), MemoryState()
    install(monkeypatch, store, state)

    async def probe(endpoint):
        return True

    monkeypatch.setattr(gpu, "_probe", probe)
    await controller.run_controller(cfg=gpu_config(min_replicas=1, location="rack 4"), once=True)
    rec = await state.get_json("deploy:gpu")
    assert rec["state"] == WARM and rec["endpoint"] == "http://gpu.test/v1"  # a static server goes warm in one cycle
    assert rec["replicas_info"][0]["infra"] == "rack 4" and rec["price"]["source"] == "configured"
    assert rec["waiting"] == 3 and (await state.utilization())["gpu"] == 0.5
    hb = await state.get_json("controller:heartbeat")
    assert hb["deployments"] == ["gpu"] and hb["interval"] == 60
    assert store.pool.released == 1
    assert ("state", "warm") in [(e["kind"], e["detail"].get("to")) for e in await store.gpu_events()]


async def test_controller_needs_postgres_and_stands_by_for_another(monkeypatch, no_metrics):
    monkeypatch.delenv("MERITORDER_DATABASE_URL", raising=False)
    with pytest.raises(SystemExit, match="needs Postgres"):
        await controller.run_controller(cfg=gpu_config(), once=True)
    store, state = pg_memory_store(locked=False), MemoryState()
    install(monkeypatch, store, state)
    await controller.run_controller(cfg=gpu_config(), once=True)
    assert await state.get_json("controller:heartbeat") is None and store.pool.released == 1


async def test_controller_follows_config_changes_and_survives_failed_cycles(monkeypatch, no_metrics, caplog):
    store, state = pg_memory_store(), MemoryState()
    install(monkeypatch, store, state)
    seen = []

    async def wait(state_, interval, kick):
        seen.append(kick)
        if len(seen) == 1:
            await store.save_config(copy.deepcopy(TEST_CONFIG), "root", "no GPUs")  # an admin saves a config without the GPU backend
            return "kick-1"
        if len(seen) == 2:
            async def broken(since):
                raise RuntimeError("ledger unavailable")

            monkeypatch.setattr(store, "demand_rows", broken)
            return kick
        raise StopLoop

    monkeypatch.setattr(controller, "wait_or_kick", wait)
    beats = []
    real_set = state.set_json

    async def set_json(key, value, ttl=0):
        if key == "controller:heartbeat":
            beats.append(value["deployments"])
        await real_set(key, value, ttl)

    monkeypatch.setattr(state, "set_json", set_json)
    with caplog.at_level(logging.INFO, "meritorder.controller"), pytest.raises(StopLoop):
        await controller.run_controller(cfg=gpu_config(), interval=1)
    assert beats == [["gpu"], [], []] and seen == [None, "kick-1", "kick-1"]
    assert "configuration changed; controlling: (none)" in caplog.text and "controller cycle failed" in caplog.text
    assert store.pool.released == 1  # the lock is released even when the loop ends by an exception

    caplog.clear()
    await controller.run_controller(cfg=Config(**EMPTY), once=True)
    assert "nothing to control" in caplog.text


async def test_wait_or_kick_wakes_early_for_admin_changes(monkeypatch):
    clock = [0.0]

    async def sleep(seconds):
        clock[0] += seconds

    monkeypatch.setattr(controller, "asyncio", SimpleNamespace(sleep=sleep))
    monkeypatch.setattr(controller, "time", SimpleNamespace(monotonic=lambda: clock[0], time=time.time))
    state = MemoryState()
    assert await controller.wait_or_kick(state, 10, "old") == "old" and clock[0] == 10  # slept through, in 2 s steps
    await state.set_json("controller:kick", "new")
    clock[0] = 0
    assert await controller.wait_or_kick(state, 60, "old") == "new" and clock[0] == 2
    assert await controller.wait_or_kick(state, 4, "new") == "new"  # a kick already acted on doesn't wake it


class FlakyProvisioner(FakeProvisioner):
    def __init__(self):
        super().__init__()
        self.ready = True
        self.fail: set[str] = set()

    async def up(self, n):
        await super().up(n)
        if "up" in self.fail:
            raise RuntimeError("no capacity in us-east-1")

    async def down(self):
        await super().down()
        if "down" in self.fail:
            raise RuntimeError("cloud API timed out")


async def test_tick_scales_retries_failed_stops_and_cools_down_failed_starts(no_metrics):
    cfg = gpu_config("skypilot")
    store, state = MemoryStore(), MemoryState()
    prov = FlakyProvisioner()
    deps = [Deployment(backend=cfg.backends[0], model=cfg.models[0], provisioner=prov)]
    t = time.time()

    async def control(**c):
        await store.set_setting("gpu_controls", {"deployments": {"gpu": {"by": "root", **c}}})

    await control(mode="on", replicas=1)
    [rec] = await controller.tick(cfg, store, state, deps, now=t)
    assert rec["state"] == WARM and prov.calls == [("up", 1)] and no_metrics  # pinned: started at once, utilization sampled
    await control(mode="on", replicas=2)
    [rec] = await controller.tick(cfg, store, state, deps, now=t + 60)
    assert prov.calls[-1] == ("scale", 2) and rec["replicas"] == 2
    assert any(e["kind"] == "scale" and e["detail"]["replicas"] == 2 for e in await store.gpu_events())

    prov.fail = {"down"}
    await control(mode="off", drain=False)
    [rec] = await controller.tick(cfg, store, state, deps, now=t + 120)
    assert rec["state"] == WARM and rec["reason"].startswith("stop failed, will retry: cloud API timed out")  # never shown off while billing
    assert any(e["kind"] == "error" and e["detail"]["action"] == "down" for e in await store.gpu_events())
    prov.fail = set()
    [rec] = await controller.tick(cfg, store, state, deps, now=t + 180)
    assert rec["state"] == API_ONLY and prov.calls[-1] == ("down",)

    prov.fail = {"up"}
    await control(mode="on", replicas=1)
    [rec] = await controller.tick(cfg, store, state, deps, now=t + 240)
    assert rec["state"] == API_ONLY and rec["reason"].startswith("up failed: no capacity") and rec["cooldown_until"] > t + 240


def test_deployments_skip_unknown_models_and_disabled_backends(caplog):
    cfg = gpu_config()
    cfg.backends += [
        BackendConfig(name="lost", kind="self_hosted", format="openai", serve=ServeConfig(model="missing")),
        BackendConfig(name="off", kind="self_hosted", format="openai", enabled=False, serve=ServeConfig(model="ow")),
        BackendConfig(name="sky", kind="self_hosted", format="openai", serve=ServeConfig(model="ow")),
        BackendConfig(name="api", format="openai"),
    ]
    deps = controller.deployments(cfg)
    assert [(d.backend.name, type(d.provisioner)) for d in deps] == [("gpu", StaticProvisioner), ("sky", SkyPilotProvisioner)]
    assert "serves unknown model 'missing'" in caplog.text


async def test_replica_price_prefers_the_catalog():
    cfg = gpu_config()

    class Broken:
        async def expected_price(self):
            raise RuntimeError("catalog offline")

    class Priced:
        async def expected_price(self):
            return {"usd_per_hour": 4.35, "cloud": "GCP"}

    def dep(p):
        return Deployment(backend=cfg.backends[0], model=cfg.models[0], provisioner=p)

    assert await controller.replica_price(dep(Broken())) == {"usd_per_hour": 2.0, "source": "configured"}
    assert await controller.replica_price(dep(object())) == {"usd_per_hour": 2.0, "source": "configured"}
    priced = await controller.replica_price(dep(Priced()))
    assert priced["source"] == "skypilot_catalog" and priced["configured_usd_per_hour"] == 2.0


# ------------------------------------------------------------------ provisioners


async def test_static_provisioner_probes_the_server(monkeypatch):
    cfg = gpu_config(location="on-prem, rack 4", accelerators="H100:8")
    answers = iter([httpx.Response(200, json={"data": []}), httpx.Response(503)])

    def handler(request):
        assert str(request.url) == "http://gpu.test/v1/models"
        try:
            return next(answers)
        except StopIteration:
            raise httpx.ConnectError("refused")

    mock_http(monkeypatch, gpu, handler)
    prov = StaticProvisioner(cfg.backends[0], cfg)
    st = await prov.status()
    assert st.ready and st.endpoint == "http://gpu.test/v1" and st.replicas == 1
    assert st.replicas_info == [{"id": 1, "status": "READY", "infra": "on-prem, rack 4", "resources": "H100:8", "spot": False,
                                 "endpoint": "http://gpu.test/v1"}]
    assert not (await prov.status()).ready and not (await prov.status()).ready  # 503, then unreachable
    assert await prov.expected_price() is None
    await prov.up(1), await prov.scale(2), await prov.down()  # your own server: nothing to start or stop
    nowhere = BackendConfig(name="x", kind="self_hosted", format="openai", serve=ServeConfig(model="ow", provisioner="static"))
    assert (await StaticProvisioner(nowhere, cfg).status()).failed


async def test_fetch_load_reads_vllm_metrics(monkeypatch):
    def handler(request):
        if request.url.host == "down.test":
            raise httpx.ConnectError("refused")
        if request.url.host == "old.test":
            return httpx.Response(404)
        assert request.url.path == "/metrics"
        return httpx.Response(200, text='vllm:kv_cache_usage_perc{m="x"} 0.4\nvllm:num_requests_running{m="x"} 9\n')

    mock_http(monkeypatch, gpu, handler)
    assert await gpu.fetch_load("http://gpu.test/v1") == {"utilization": 0.4, "waiting": 0, "running": 9}
    assert await gpu.fetch_utilization("http://gpu.test") == 0.4
    assert await gpu.fetch_load("http://old.test/v1") is None and await gpu.fetch_utilization("http://down.test/v1") is None


def test_make_provisioner_and_served_name():
    cfg = gpu_config("skypilot", served_model_name="qwen-coder")
    prov = gpu.make_provisioner(cfg.backends[0], cfg)
    assert isinstance(prov, SkyPilotProvisioner)
    assert gpu.served_name(cfg.backends[0], cfg) == "qwen-coder"
    with pytest.raises(ValueError, match="no serve config"):
        gpu.make_provisioner(BackendConfig(name="x", format="openai"), cfg)
    cfg.models[0].hf_id = None
    with pytest.raises(ValueError, match="needs an hf_id"):
        prov.task_config(prov.cluster(1))


async def test_expected_price_is_cached(monkeypatch):
    cfg = gpu_config("skypilot", accelerators="H100:4", use_spot=True, clouds=["gcp"])
    calls = []

    def placements(accelerators, clouds, regions, spot):
        calls.append((accelerators, clouds, regions, spot))
        return {"best": {"usd_per_hour": 4.35}, "pairs": [["gcp", "europe-north1"]]}

    monkeypatch.setattr(gpu, "catalog_placements", placements)
    prov = gpu.make_provisioner(cfg.backends[0], cfg)
    assert await prov.expected_price() == {"usd_per_hour": 4.35} == await prov.expected_price()
    assert calls == [("H100:4", ["gcp"], [], True)]  # looked up once per 6 hours


@pytest.fixture
def fake_sky(tmp_path, monkeypatch):
    """A `sky` command on PATH: a shell launcher like the one pip writes when the interpreter's path has spaces."""
    if __import__("importlib").util.find_spec("sky") is not None:
        pytest.skip("SkyPilot is installed here; these tests exercise the CLI path")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    script = bindir / "sky"
    script.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} -c "print(1)" "$@"\n')
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.delenv("MERITORDER_SKY_PYTHON", raising=False)
    return SimpleNamespace(script=script)


def test_sky_python_comes_from_the_cli_shebang(fake_sky, monkeypatch):
    assert gpu.sky_python() is None  # a shell launcher: not a Python entry point
    fake_sky.script.write_text("#!/opt/skypilot/bin/python3 -I\nimport sky\n")
    assert gpu.sky_python() == "/opt/skypilot/bin/python3"
    monkeypatch.setenv("MERITORDER_SKY_PYTHON", "/usr/local/sky/python")
    assert gpu.sky_python() == "/usr/local/sky/python"  # an explicit setting wins
    monkeypatch.delenv("MERITORDER_SKY_PYTHON")
    monkeypatch.setenv("PATH", "/nonexistent")
    assert gpu.sky_python() is None


FAKE_CATALOG = '''
class Offer:
    def __init__(self, cloud, region, instance_type, price, spot_price):
        self.cloud, self.region, self.instance_type, self.price, self.spot_price = cloud, region, instance_type, price, spot_price


def get_hourly_cost(instance, use_spot, region, zone, clouds=None):
    if instance == "boom":
        raise ValueError("unknown instance")
    return {"p5.48xlarge": 21.45 if use_spot else 98.32}[instance]


def list_accelerators(name_filter, quantity_filter, clouds, all_regions, case_sensitive):
    assert (name_filter, quantity_filter, all_regions) == ("H100", 4, True)
    offers = [Offer("AWS", "us-east-1", "p5.48xlarge", 98.3, 21.45), Offer("GCP", "europe-north1", "a3-highgpu-4g", 43.9, 4.35),
              Offer("GCP", "us-central1", "a3-highgpu-4g", 43.92, float("nan")), Offer("Azure", "eastus", "nd-h100", None, -1),
              Offer("Lambda", "us-east-1", "gpu_4x_h100_sxm5", 16.36, None)]
    return {"H100": [o for o in offers if not clouds or o.cloud.lower() in clouds]}
'''


@pytest.fixture
def sky_package(tmp_path, monkeypatch):
    """A tiny stand-in for the SkyPilot package's catalog (sky.catalog)."""
    if __import__("importlib").util.find_spec("sky") is not None:
        pytest.skip("SkyPilot is installed here")
    site = tmp_path / "site"
    pkg = site / "sky"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "catalog.py").write_text(FAKE_CATALOG)
    yield site
    for name in [m for m in sys.modules if m == "sky" or m.startswith("sky.")]:
        del sys.modules[name]


def check_catalog_answers():
    assert gpu.catalog_cheapest("H100:4", [], [], True) == {"usd_per_hour": 4.35, "cloud": "GCP", "region": "europe-north1",
                                                            "instance_type": "a3-highgpu-4g", "spot": True}
    assert gpu.catalog_cheapest("H100:4", ["AWS"], [], True)["usd_per_hour"] == 21.45
    assert gpu.catalog_cheapest("H100:4", [], ["us-east-1"], False)["instance_type"] == "gpu_4x_h100_sxm5"
    assert gpu.catalog_cheapest("H100:4", ["Azure"], [], True) is None  # no usable price
    # The launch goes where the price came from: us-east-1 pairs with AWS and Lambda, which both have one, never with GCP.
    found = gpu.catalog_placements("H100:4", ["aws", "gcp", "lambda"], ["us-east-1"], False)
    assert found["pairs"] == [["aws", "us-east-1"], ["lambda", "us-east-1"]] and found["best"]["cloud"] == "Lambda"
    assert gpu.catalog_placements("H100:4", ["aws", "gcp", "lambda"], ["us-east-1"], True)["pairs"] == [["aws", "us-east-1"]]  # no spot on Lambda
    assert gpu.catalog_price("p5.48xlarge", "AWS", "us-east-1", None, True) == 21.45
    assert gpu.catalog_price("boom", "AWS", "us-east-1", None, True) is None


def test_catalog_in_process_when_skypilot_is_importable(sky_package, monkeypatch):
    monkeypatch.syspath_prepend(str(sky_package))
    check_catalog_answers()


def test_catalog_in_skypilots_own_python(sky_package, monkeypatch):
    monkeypatch.setenv("MERITORDER_SKY_PYTHON", sys.executable)
    monkeypatch.setenv("PYTHONPATH", str(sky_package))  # only the child process can import it
    check_catalog_answers()
    monkeypatch.delenv("PYTHONPATH")
    assert gpu.catalog_price("p5.48xlarge", "AWS", "us-east-1", None, True) is None  # SkyPilot missing there too
    monkeypatch.setenv("MERITORDER_SKY_PYTHON", "/nonexistent/python")
    assert gpu.catalog_price("p5.48xlarge", "AWS", "us-east-1", None, True) is None


# ------------------------------------------------------------------ configuration store


def test_parse_and_validate_rejects_bad_documents():
    for text, message in (("a: [", "not valid YAML"), ("- just a list", "must be a mapping"),
                          ("include: [other.yaml]", "include"),
                          ("backends: {x: {format: openai, auth: {value: sk-live-123}}}", "literal secret"),
                          ("auth: {static_keys: [{key: mo-123, user: a}]}", "literal key"),
                          ("backends: {x: {format: carrier-pigeon}}", "format")):
        with pytest.raises(ConfigError, match=message):
            parse_and_validate(text, ".", None)
    raw, cfg = parse_and_validate(None, ".", None, document={"backends": {"x": {"format": "openai", "auth": {"value": "${MERITORDER_BACKEND_MO}"}, "base_url": ""}}})
    assert raw == {"backends": {"x": {"format": "openai", "auth": {"value": "${MERITORDER_BACKEND_MO}"}}}}  # blank fields pruned
    assert summarize(cfg)["backends_enabled"] == [] or "x" in summarize(cfg)["backends_disabled"]
    assert yaml.safe_load(to_yaml(raw)) == raw


async def test_config_follower_uses_stored_versions_and_keeps_the_last_good_one(tmp_path, caplog):
    path = tmp_path / "meritorder.yaml"
    path.write_text(yaml.safe_dump(TEST_CONFIG))
    store = MemoryStore()
    follower = ConfigFollower(store, path=path)
    first = await follower.current()
    assert first is follower.file_cfg and await follower.current() is first  # nothing stored: the file
    raw = copy.deepcopy(TEST_CONFIG)
    raw["teams"]["default"]["budget_usd_month"] = 5
    await store.save_config(raw, "root", "lower budget")
    assert (await follower.current()).team("default").budget_usd_month == 5
    good = follower.cfg
    await store.save_config({"backends": {"x": {"format": "carrier-pigeon"}}}, "root", "broken")
    assert await follower.current() is good and "doesn't load" in caplog.text
    await store.set_setting("prices", {"updated": "2026-11-09", "prices": {}})
    await follower.current()  # stored prices alone also trigger a reload attempt


async def test_load_active_applies_and_publishes(gateway):
    if gateway.store.kind != "memory":
        pytest.skip("covered in memory")
    assert await load_active(gateway) is False  # nothing stored: the file stays in effect
    raw = copy.deepcopy(TEST_CONFIG)
    raw["teams"]["default"]["budget_usd_month"] = 7
    vid = await gateway.store.save_config(raw, "root", None)
    assert await load_active(gateway, publish=True) is True and gateway.config_version == vid
    assert await gateway.state.get_config_token() == configstore.token(vid, None)
    assert await load_active(gateway) is False  # already applied


# ------------------------------------------------------------------ app startup and plain endpoints


async def test_lifespan_applies_stored_config_and_runs_the_follow_loops(cfg, monkeypatch):
    real_sleep = asyncio.sleep

    async def fast(seconds):
        await real_sleep(0.005)

    monkeypatch.setattr(app_module, "asyncio", SimpleNamespace(create_task=asyncio.create_task, sleep=fast))
    store, state = MemoryStore(), MemoryState()
    gw = build_gateway(cfg, store=store, state=state)
    gw.file_raw = copy.deepcopy(TEST_CONFIG)
    raw = copy.deepcopy(TEST_CONFIG)
    raw["teams"]["default"]["budget_usd_month"] = 5
    v1 = await store.save_config(raw, "root", "first")
    app = create_app(gateway=gw)
    async with app.router.lifespan_context(app):
        assert gw.config_version == v1  # applied at startup
        v2 = await store.save_config({**raw, "teams": {"default": {"budget_usd_month": 9}}}, "root", "second")
        await state.set_config_token(configstore.token(v2, None))  # what the gateway that saved it publishes
        for _ in range(200):
            if gw.config_version == v2 and gw.gateway_id in state._gateways:
                break
            await real_sleep(0.005)
        assert gw.config_version == v2 and gw.cfg.team("default").budget_usd_month == 9
        assert gw.gateway_id in state._gateways  # heartbeat with in-flight counts


async def test_lifespan_logs_and_survives_storage_errors(cfg, monkeypatch, caplog):
    real_sleep = asyncio.sleep

    async def fast(seconds):
        await real_sleep(0.005)

    monkeypatch.setattr(app_module, "asyncio", SimpleNamespace(create_task=asyncio.create_task, sleep=fast))

    class BrokenState(MemoryState):
        async def publish_inflight(self, gateway_id, counts):
            raise ConnectionError("redis down")

        async def get_config_token(self):
            raise ConnectionError("redis down")

    class BrokenStore(MemoryStore):
        async def active_config(self):
            raise ConnectionError("postgres down")

    gw = build_gateway(cfg, store=BrokenStore(), state=BrokenState())
    app = create_app(gateway=gw)
    async with app.router.lifespan_context(app):
        await real_sleep(0.05)
    for message in ("could not load the stored configuration", "could not publish in-flight counts",
                    "could not refresh configuration or deployments", "using in-memory storage"):
        assert message in caplog.text


async def test_create_app_from_the_config_file_and_plain_endpoints(tmp_path, monkeypatch):
    path = tmp_path / "meritorder.yaml"
    path.write_text(yaml.safe_dump(TEST_CONFIG))
    monkeypatch.setenv("MERITORDER_CONFIG", str(path))
    for var in ("MERITORDER_DATABASE_URL", "MERITORDER_REDIS_URL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("MERITORDER_METRICS_TOKEN", "scrape")
    app = create_app()
    gw = app.state.gateway
    assert gw.file_raw["baseline_model"] == "big" and gw.store.kind == "memory"
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        assert (await c.get("/healthz")).json()["status"] == "ok"
        assert (await c.get("/readyz")).json()["status"] == "ok"
        assert (await c.get("/metrics")).status_code == 401
        assert (await c.get("/metrics", headers={"authorization": "Bearer scrape"})).status_code == 200
        assert (await c.get("/auth/config")).json() == {"oidc": None, "local": {"enabled": True}, "sso": None}
        assert (await c.get("/v1/models", headers=auth("mo-wrong"))).status_code == 401
        assert (await c.post("/auth/exchange", json={"id_token": "x"})).status_code >= 400
        gw.cfg.auth.oidc = OIDCConfig(issuer="https://idp.test", client_id="cli")
        oidc = (await c.get("/auth/config")).json()["oidc"]
        assert oidc["issuer"] == "https://idp.test" and oidc["client_id"] == "cli" and oidc["scopes"]

        async def down():
            raise ConnectionError("postgres down")

        monkeypatch.setattr(gw.store, "ping", down)
        r = await c.get("/readyz")
        assert r.status_code == 503 and "postgres down" in r.json()["error"]


# ------------------------------------------------------------------ telemetry


def test_tracing_setup_needs_an_endpoint_and_the_exporter(monkeypatch, caplog):
    for var in ("OTEL_EXPORTER_OTLP_ENDPOINT", "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"):
        monkeypatch.delenv(var, raising=False)
    assert telemetry.setup_tracing() is False
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4318")
    try:
        import opentelemetry.exporter.otlp.proto.http.trace_exporter  # noqa: F401
    except ImportError:
        assert telemetry.setup_tracing() is False and "tracing isn't installed" in caplog.text
    else:
        providers = []
        monkeypatch.setattr(telemetry.trace, "set_tracer_provider", providers.append)
        assert telemetry.setup_tracing() is True and providers
    assert telemetry.provider_name("anthropic-direct") == "anthropic" and telemetry.provider_name("bedrock-us") == "aws.bedrock"
    assert telemetry.provider_name("my-vllm") == "my-vllm" and telemetry.provider_name(None) == "unknown"


def test_error_spans_carry_status_and_notes():
    from test_telemetry import EXPORTER

    EXPORTER.clear()
    base = {"latency_ms": 120, "requested_model": "auto", "served_model": "big", "backend": "anthropic", "client_format": "openai"}
    telemetry.record_span({**base, "status": 529, "error": "overloaded", "floor": "coding:4", "ttft_ms": 80})
    telemetry.record_span({**base, "status": 200, "error": "refused by the model; retried on another model"})
    telemetry.record_span({**base, "client_format": "embeddings", "status": 200})
    failed, noted, embed = [s for s in EXPORTER.get_finished_spans() if s.instrumentation_scope.name == "meritorder"]
    assert failed.attributes["error.type"] == "529" and failed.status.status_code.name == "ERROR"
    assert failed.attributes["meritorder.floor"] == "coding:4" and failed.attributes["meritorder.time_to_first_token_ms"] == 80
    assert noted.attributes["meritorder.note"].startswith("refused") and noted.status.status_code.name == "UNSET"
    assert embed.name == "embeddings big" and embed.attributes["gen_ai.operation.name"] == "embeddings"


def test_month_start_is_utc():
    assert month_start(datetime(2026, 11, 9, 10, tzinfo=UTC)) == datetime(2026, 11, 1, tzinfo=UTC)
