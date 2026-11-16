"""GPU controls (stop, pin, stop all), fleet safeguards, cost accrual, and spend safeguards."""

from datetime import UTC, datetime

from conftest import ADMIN_KEY, LEAD_KEY, TEST_KEY, auth
from test_gpu import FakeProvisioner

from meritorder.config import BackendConfig, Config, ModelConfig, Route, Safeguards, ServeConfig
from meritorder.controller import tick
from meritorder.gpu import API_ONLY, DRAINING, PROVISIONING, WARM, Deployment, Plan, Status, decide, plan_fleet
from meritorder.state import MemoryState
from meritorder.store import MemoryStore

SERVE = ServeConfig(model="ow", hold_minutes=30, drain_minutes=15, max_replicas=4)
READY = Status(ready=True, endpoint="http://gpu/v1")


def test_force_off_stops_or_drains():
    t = 0.0
    warm = {"state": WARM, "since": t, "warm_at": t, "replicas": 2, "endpoint": "http://gpu/v1", "started_at": t, "run_cost": 5.0}
    rec, act = decide(warm, SERVE, 0, READY, t + 10, force="off", drain=False, why="stopped by root")
    assert rec["state"] == API_ONLY and act.kind == "down" and rec["reason"] == "stopped by root"
    assert rec["last_run"]["cost"] == 5.0 and rec["started_at"] is None  # the run is summarized when it ends
    rec, act = decide(warm, SERVE, 0, READY, t + 10, force="off", drain=True)  # skips the 30-minute hold
    assert rec["state"] == DRAINING and act is None
    rec, act = decide(rec, SERVE, 3, READY, t + 60, force="off", drain=True)
    assert rec["state"] == DRAINING  # demand coming back doesn't cancel a stop
    rec, act = decide(rec, SERVE, 0, READY, t + 10 + 15 * 60, force="off", drain=True)
    assert rec["state"] == API_ONLY and act.kind == "down"
    rec, act = decide({"state": PROVISIONING, "since": t}, SERVE, 0, Status(), t + 5, force="off")
    assert rec["state"] == API_ONLY and act.kind == "down"  # provisioning is cancelled outright
    rec, act = decide({}, SERVE, 0, Status(), t, force="off")
    assert rec["state"] == API_ONLY and act is None


def test_force_on_skips_timers_and_cooldown():
    rec, act = decide({"state": API_ONLY, "since": 0, "cooldown_until": 10_000}, SERVE, 2, Status(), 100, force="on", why="pinned")
    assert rec["state"] == PROVISIONING and act.kind == "up" and act.replicas == 2 and rec["started_at"] == 100


def test_plan_fleet_controls_budget_and_caps():
    items = [{"name": "a", "desired": 2, "price": 10.0, "max_replicas": 4, "value": 100.0},
             {"name": "b", "desired": 2, "price": 10.0, "max_replicas": 4, "value": 30.0},
             {"name": "c", "desired": 0, "price": 10.0, "max_replicas": 4, "value": 0.0}]
    none = Safeguards()
    plans = plan_fleet(items, {"deployments": {"c": {"mode": "on", "replicas": 9, "by": "root"}}}, none, 0, now=1)
    assert plans["a"] == Plan(2) and plans["c"].force == "on" and plans["c"].desired == 4  # clamped to max_replicas
    plans = plan_fleet(items, {"deployments": {"a": {"mode": "off", "drain": False, "by": "root", "reason": "test"}}}, none, 0, now=1)
    assert plans["a"].force == "off" and not plans["a"].drain and "root" in plans["a"].note
    plans = plan_fleet(items, {"deployments": {"a": {"mode": "off", "until": 5}}}, none, 0, now=10)
    assert plans["a"] == Plan(2)  # expired controls are ignored
    plans = plan_fleet(items, {"paused": True, "paused_by": "root"}, none, 0, now=1)
    assert all(p.force == "off" for p in plans.values())
    plans = plan_fleet(items, {}, Safeguards(gpu_budget_usd_month=100), 100.0, now=1)
    assert all(p.force == "off" and p.drain for p in plans.values()) and "GPU budget" in plans["a"].note
    # A $30/h fleet cap: the deployment whose demand best covers its cost gets capacity first.
    plans = plan_fleet(items, {}, Safeguards(max_gpu_usd_per_hour=30), 0, now=1)
    assert plans["a"].desired == 2 and plans["b"].desired == 1 and "fleet limit" in plans["b"].note
    plans = plan_fleet(items, {"deployments": {"b": {"mode": "on", "replicas": 2}}}, Safeguards(max_gpu_replicas=2), 0, now=1)
    assert plans["b"].desired == 2 and plans["a"].force == "off"  # pins come first under a cap


async def test_tick_accrues_cost_logs_events_and_obeys_stop():
    model = ModelConfig(name="ow", quality={"coding": 4}, hf_id="o/m", routes=[Route(backend="gpu", model="m")])
    backend = BackendConfig(name="gpu", kind="self_hosted", format="openai", gpu_cost_per_hour=36.0,
                            serve=ServeConfig(model="ow", scale_up_after_minutes=0))
    cfg = Config(backends=[backend], models=[model])
    store, state = MemoryStore(), MemoryState()
    now = datetime.now(UTC)
    for _ in range(60):
        await store.record({"ts": now, "floor": "coding:3", "served_model": "sonnet", "requested_model": "auto",
                            "served_api_cost": 1.0, "actual_cost": 1.0, "status": 200, "upgraded": False,
                            "input_tokens": 1000, "output_tokens": 100})
    prov = FakeProvisioner()
    prov.ready = True
    deps = [Deployment(backend=backend, model=model, provisioner=prov)]
    t = now.timestamp()
    [rec] = await tick(cfg, store, state, deps, now=t)
    assert rec["state"] == WARM
    [rec] = await tick(cfg, store, state, deps, now=t + 100)  # 1 replica x $36/h x 100 s = $1
    assert abs(rec["run_cost"] - 1.0) < 1e-6
    assert abs(sum((await store.gpu_spend_since(now.replace(day=1, hour=0, minute=0, second=0, microsecond=0))).values()) - 1.0) < 1e-6
    await store.set_setting("gpu_controls", {"deployments": {"gpu": {"mode": "off", "drain": False, "by": "root"}}})
    [rec] = await tick(cfg, store, state, deps, now=t + 110)
    assert rec["state"] == API_ONLY and prov.calls[-1] == ("down",) and "root" in rec["reason"]
    assert rec["last_run"]["cost"] >= 1.0
    kinds = [(e["kind"], e["detail"].get("to")) for e in await store.gpu_events()]
    assert ("state", "api_only") in kinds and ("state", "warm") in kinds
    assert await state.get_json("gpu:controls")  # mirrored for gateways


async def test_admin_controls_api_and_routing(client, gateway):
    # Not a self-hosted backend
    r = await client.post("/v1/admin/gpu/mock-openai/control", json={"mode": "off"}, headers=auth(ADMIN_KEY))
    assert r.status_code == 404
    r = await client.post("/v1/admin/gpu/stop-all", json={"reason": "cost review"}, headers=auth(LEAD_KEY))
    assert r.status_code == 403
    r = await client.post("/v1/admin/gpu/stop-all", json={"reason": "cost review"}, headers=auth(ADMIN_KEY))
    assert r.status_code == 200 and r.json()["controls"]["paused"] and "warning" in r.json()  # no controller running
    ev = (await client.get("/v1/admin/gpu/events", headers=auth(ADMIN_KEY))).json()["events"]
    assert ev[0]["kind"] == "stop_all" and ev[0]["actor"] == "root"
    gpu = (await client.get("/v1/reports/gpu", headers=auth(ADMIN_KEY))).json()
    assert gpu["paused"]["by"] == "root" and gpu["controller"]["alive"] is False
    att = (await client.get("/v1/console/attention", headers=auth(ADMIN_KEY))).json()["items"]
    assert any(i["title"] == "All GPUs are stopped" for i in att)
    r = await client.post("/v1/admin/gpu/resume", headers=auth(ADMIN_KEY))
    assert "paused" not in r.json()["controls"]
    c = (await client.get("/v1/admin/controls", headers=auth(ADMIN_KEY))).json()
    assert c["teams"][0]["budget_usd_month"] == 100 and c["spend"]["org"]["budget"] is None
    assert (await client.get("/v1/admin/controls", headers=auth(TEST_KEY))).status_code == 403


async def test_stopped_gpu_leaves_routing_at_once(gateway):
    from meritorder.config import BackendConfig

    b = BackendConfig(name="gpu", kind="self_hosted", format="openai", serve=ServeConfig(model="small", provisioner="static", static_url="http://x/v1"))
    from meritorder.backends import build_backends

    gateway.backends.update(build_backends(Config(backends=[b], models=[])))
    await gateway.state.set_json("deploy:gpu", {"state": "warm", "endpoint": "http://x/v1"})
    await gateway.refresh_deployments()
    assert gateway.backends["gpu"].available and not gateway.backends["gpu"].draining
    await gateway.state.set_json("gpu:controls", {"deployments": {"gpu": {"mode": "off", "drain": True}}})
    await gateway.refresh_deployments()
    assert gateway.backends["gpu"].draining  # only warm sessions finish there
    await gateway.state.set_json("gpu:controls", {"paused": True})
    await gateway.refresh_deployments()
    assert not gateway.backends["gpu"].available


async def test_org_budget_and_per_request_cap(client, gateway):
    gateway.cfg.safeguards = Safeguards(max_request_usd=0.05)
    body = {"model": "big", "max_tokens": 10_000, "messages": [{"role": "user", "content": "hi"}]}
    r = await client.post("/v1/chat/completions", json=body, headers=auth())
    assert r.status_code == 400 and "per-request limit" in r.json()["error"]["message"]  # 10k x $15/M = $0.15
    r = await client.post("/v1/chat/completions", json={**body, "max_tokens": 1000}, headers=auth())
    assert r.status_code == 200
    gateway.cfg.safeguards = Safeguards(org_budget_usd_month=0.0)
    r = await client.post("/v1/chat/completions", json={**body, "max_tokens": 100}, headers=auth())
    assert r.status_code == 429 and "organization" in r.json()["error"]["message"]
    assert r.headers["x-meritorder-limit"] == "budget" and "retry-after" not in r.headers  # waiting won't clear it


def test_catalog_without_skypilot(monkeypatch):
    from meritorder import gpu
    from meritorder.gpu import catalog_cheapest, catalog_offers, catalog_placements, catalog_price

    try:
        import sky  # noqa: F401
    except ImportError:  # not importable here; and with no `sky` to run either, prices fall back to the configured value
        monkeypatch.setattr(gpu, "sky_python", lambda: None)
        assert not gpu.sky_available()
        assert catalog_price("p5.48xlarge", "AWS", "us-east-1", None, True) is None
        assert catalog_cheapest("H100:4", [], [], True) is None and catalog_placements("H100:4", [], ["us-east-1"], True) is None
        assert catalog_offers(["aws"]) is None


class PricedProvisioner(FakeProvisioner):
    async def expected_price(self):
        return {"usd_per_hour": 4.35, "cloud": "GCP", "region": "europe-north1", "instance_type": "a3-highgpu-4g", "spot": True}

    async def status(self):
        st = await super().status()
        if self.ready:
            st.replicas_info = [{"id": 1, "cloud": "GCP", "region": "europe-north1", "spot": True, "usd_per_hour": 5.0, "status": "READY"}]
        return st


async def test_catalog_prices_drive_sizing_and_cost():
    model = ModelConfig(name="ow", quality={"coding": 4}, hf_id="o/m", routes=[Route(backend="gpu", model="m")])
    # Configured at $40/h, so $10/h of demand wouldn't justify a replica; the catalog says $4.35/h.
    backend = BackendConfig(name="gpu", kind="self_hosted", format="openai", gpu_cost_per_hour=40.0,
                            serve=ServeConfig(model="ow", scale_up_after_minutes=0))
    cfg = Config(backends=[backend], models=[model])
    store, state = MemoryStore(), MemoryState()
    now = datetime.now(UTC)
    for _ in range(10):
        await store.record({"ts": now, "floor": "coding:3", "served_model": "sonnet", "requested_model": "auto",
                            "served_api_cost": 1.0, "actual_cost": 1.0, "status": 200, "upgraded": False,
                            "input_tokens": 1000, "output_tokens": 100})
    prov = PricedProvisioner()
    prov.ready = True
    deps = [Deployment(backend=backend, model=model, provisioner=prov)]
    t = now.timestamp()
    [rec] = await tick(cfg, store, state, deps, now=t)
    assert rec["state"] == WARM and rec["price"]["source"] == "skypilot_catalog" and rec["gpu_usd_per_hour"] == 4.35
    assert rec["burn_usd_per_hour"] == 5.0  # the replica's own catalog price wins over the expected one
    [rec] = await tick(cfg, store, state, deps, now=t + 360)
    assert abs(rec["run_cost"] - 0.5) < 1e-6  # $5/h for 6 minutes


async def test_stranded_auto_session_reroutes_and_named_model_explains(gateway, client):
    from meritorder.config import BackendConfig

    b = BackendConfig(name="gpu", kind="self_hosted", format="openai", serve=ServeConfig(model="gpu-only", provisioner="static", static_url="http://x/v1"))
    from meritorder.backends import build_backends

    gateway.backends.update(build_backends(Config(backends=[b], models=[])))
    gpu_only = ModelConfig(name="gpu-only", quality={"coding": 5, "general": 5},
                           routes=[Route(backend="gpu", model="m", price={"input": 0.1, "output": 0.1})])
    gateway.cfg.models.append(gpu_only)
    await gateway.state.set_json("gpu:controls", {"deployments": {"gpu": {"mode": "off", "drain": False, "by": "root", "reason": "cost review"}}})
    await gateway.refresh_deployments()
    r = await client.post("/v1/chat/completions", json={"model": "gpu-only", "messages": [{"role": "user", "content": "hi"}]}, headers=auth())
    msg = r.json()["error"]["message"]
    assert r.status_code == 503 and "stopped by root: cost review" in msg and "'auto'" in msg
    # An auto session that was on gpu-only keeps working on another model.
    from meritorder.sessions import Session

    s = Session(id="sess-1")
    s.model, s.backend, s.turns, s.floor = "gpu-only", "gpu", 3, "coding:3"
    decision = gateway.router.decide("auto", gateway.cfg.team("default"), {"messages": [{"role": "user", "content": "go on"}]}, {}, s, 10,
                                     False, {}, 0.0)
    assert decision.candidates and decision.candidates[0].model.name != "gpu-only" and "unavailable; re-routed" in decision.reason


async def test_opt_in_substitution_for_named_model(gateway, client):
    from meritorder.backends import build_backends
    from meritorder.config import BackendConfig

    b = BackendConfig(name="gpu", kind="self_hosted", format="openai", serve=ServeConfig(model="gpu-only", provisioner="static", static_url="http://x/v1"))
    gateway.backends.update(build_backends(Config(backends=[b], models=[])))
    gateway.cfg.models.append(ModelConfig(name="gpu-only", quality={"coding": 4, "general": 4},
                                          routes=[Route(backend="gpu", model="m", price={"input": 0.1, "output": 0.1})]))
    await gateway.refresh_deployments()  # never started: unavailable
    body = {"model": "gpu-only", "messages": [{"role": "user", "content": "hi"}]}
    assert (await client.post("/v1/chat/completions", json=body, headers=auth())).status_code == 503
    gateway.cfg.team("default").substitute_unavailable = True
    r = await client.post("/v1/chat/completions", json=body, headers=auth())
    # "small" is level 3, so only "big" (level 5) is good enough.
    assert r.status_code == 200 and r.headers["x-meritorder-substituted-for"] == "gpu-only" and r.headers["x-meritorder-model"] == "big"
    gateway.cfg.models[-1].quality = {"coding": 5, "general": 5}
    gateway.cfg.models = [m for m in gateway.cfg.models if m.name != "big"] + [gateway.cfg.models[-1]]
    r = await client.post("/v1/chat/completions", json=body, headers=auth())
    assert r.status_code == 503  # nothing at least as good: never a weaker model
