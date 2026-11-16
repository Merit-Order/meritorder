"""GPU controls API: per-deployment stop/pin/auto, the fleet summary, the spend page, and the activity log."""

import time
from datetime import UTC, datetime

import pytest
from conftest import ADMIN_KEY, LEAD_KEY, TEST_KEY, auth, ledger_row

from meritorder.backends import build_backends
from meritorder.config import BackendConfig, Config, ServeConfig
from meritorder.controls import describe_control, fleet, month_fraction


@pytest.fixture
def gpus(gateway):
    """Two managed deployments the gateway can route to."""
    cfgs = [BackendConfig(name=n, kind="self_hosted", format="openai", gpu_cost_per_hour=3.0,
                          serve=ServeConfig(model="small", provisioner="static", static_url=f"http://{n}.test/v1"))
            for n in ("gpu-a", "gpu-b")]
    gateway.cfg.backends.extend(cfgs)
    gateway.backends.update(build_backends(Config(backends=cfgs, models=[])))
    return gateway


async def post(client, path, body=None, key=ADMIN_KEY):
    return await client.post(path, json=body if body is not None else {}, headers=auth(key))


async def test_only_admins_change_controls(client, gpus):
    _, fin =await gpus.store.create_key("fin", "ops", "finance")
    for key in (TEST_KEY, LEAD_KEY, fin):
        assert (await post(client, "/v1/admin/gpu/gpu-a/control", {"mode": "off"}, key)).status_code == 403
        assert (await post(client, "/v1/admin/gpu/resume", None, key)).status_code == 403
    assert (await client.post("/v1/admin/gpu/stop-all", json={})).status_code == 401
    assert (await client.get("/v1/admin/controls", headers=auth(fin))).status_code == 200  # finance can look
    assert (await client.get("/v1/admin/controls", headers=auth(LEAD_KEY))).status_code == 403


async def test_stop_now_with_expiry_takes_the_backend_out_of_routing(client, gpus):
    await gpus.state.set_json("deploy:gpu-a", {"state": "warm", "endpoint": "http://gpu-a.test/v1", "replicas": 1})
    await gpus.refresh_deployments()
    assert gpus.backends["gpu-a"].available
    r = await post(client, "/v1/admin/gpu/gpu-a/control", {"mode": "off", "drain": False, "hours": 2, "reason": "cost review"})
    assert r.status_code == 200, r.text
    dep = r.json()["controls"]["deployments"]["gpu-a"]
    assert dep["mode"] == "off" and dep["drain"] is False and dep["by"] == "root" and dep["replicas"] is None
    assert 2 * 3600 - 60 < dep["until"] - time.time() <= 2 * 3600
    assert "warning" in r.json()  # no controller heartbeat: routing changed, cloud GPUs didn't
    assert not gpus.backends["gpu-a"].available and "stopped by root: cost review" in gpus.backends["gpu-a"].unavailable_reason
    assert await gpus.state.get_json("controller:kick")  # the controller is woken early
    alert = (await gpus.state.recent_alerts(1))[0]
    assert alert["title"] == "GPU gpu-a stopped now" and "for 2h" in alert["text"] and "(cost review)" in alert["text"]
    ev = (await client.get("/v1/admin/gpu/events?backend=gpu-a", headers=auth(LEAD_KEY))).json()["events"]
    assert ev[0]["kind"] == "control" and ev[0]["detail"]["mode"] == "off" and isinstance(ev[0]["ts"], str)


async def test_drain_pin_and_back_to_auto(client, gpus):
    await gpus.state.set_json("controller:heartbeat", {"at": time.time(), "host": "ctl-1", "interval": 60})
    r = await post(client, "/v1/admin/gpu/gpu-b/control", {"mode": "off"})
    assert r.json()["controls"]["deployments"]["gpu-b"]["drain"] is True and "warning" not in r.json()
    assert r.json()["controller"]["alive"] and r.json()["controller"]["host"] == "ctl-1"
    assert (await gpus.state.recent_alerts(1))[0]["title"] == "GPU gpu-b stopping after warm sessions finish"

    assert (await post(client, "/v1/admin/gpu/gpu-b/control", {"mode": "on"})).status_code == 422  # how many?
    assert (await post(client, "/v1/admin/gpu/gpu-b/control", {"mode": "on", "replicas": 0})).status_code == 422
    r = await post(client, "/v1/admin/gpu/gpu-b/control", {"mode": "on", "replicas": 2})
    assert r.json()["controls"]["deployments"]["gpu-b"] == {**r.json()["controls"]["deployments"]["gpu-b"], "mode": "on", "replicas": 2}
    assert describe_control(r.json()["controls"], "gpu-b")["mode"] == "on"

    r = await post(client, "/v1/admin/gpu/gpu-b/control", {"mode": "auto", "reason": "done"})
    assert "gpu-b" not in r.json()["controls"]["deployments"]
    assert (await gpus.state.recent_alerts(1))[0]["text"] == "GPU gpu-b back to automatic: root (done)"
    assert (await post(client, "/v1/admin/gpu/nope/control", {"mode": "off"})).status_code == 404
    assert (await post(client, "/v1/admin/gpu/gpu-b/control", {"mode": "sideways"})).status_code == 422
    kinds = [e["kind"] for e in (await client.get("/v1/admin/gpu/events?limit=10", headers=auth(ADMIN_KEY))).json()["events"]]
    assert kinds == ["control", "control", "control"]
    assert (await client.get("/v1/admin/gpu/events", headers=auth(TEST_KEY))).status_code == 403


async def test_stop_all_then_resume_keeps_per_deployment_controls(client, gpus):
    await post(client, "/v1/admin/gpu/gpu-a/control", {"mode": "on", "replicas": 1})
    r = await post(client, "/v1/admin/gpu/stop-all", {"reason": "incident", "drain": True})
    c = r.json()["controls"]
    assert c["paused"] and c["drain"] is True and c["paused_by"] == "root"
    d = describe_control(c, "gpu-a")
    assert d["fleet"] is True and d["mode"] == "off" and d["reason"] == "incident"  # the fleet-wide stop wins
    await gpus.state.set_json("deploy:gpu-a", {"state": "warm", "endpoint": "http://gpu-a.test/v1", "replicas": 1})
    await gpus.refresh_deployments()
    assert gpus.backends["gpu-a"].draining  # drain: warm sessions finish
    r = await post(client, "/v1/admin/gpu/resume")
    c = r.json()["controls"]
    assert not any(k in c for k in ("paused", "paused_by", "reason", "drain"))
    assert c["deployments"]["gpu-a"]["mode"] == "on"


async def test_fleet_summary(gpus):
    await gpus.state.set_json("deploy:gpu-a", {"state": "warm", "replicas": 2})  # no live price: configured $3/h each
    await gpus.state.set_json("deploy:gpu-b", {"state": "draining", "replicas": 1, "burn_usd_per_hour": 4.35})
    f = await fleet(gpus)
    assert f == {"replicas": 3, "usd_per_hour": 10.35, "deployments": ["gpu-a", "gpu-b"]}
    await gpus.state.set_json("deploy:gpu-a", {"state": "api_only", "replicas": 0})
    assert (await fleet(gpus))["replicas"] == 1


async def test_controls_page_spend_and_projection(client, gpus):
    gpus.cfg.safeguards.org_budget_usd_month = 500.0
    for team, cost in (("default", 20.0), ("retired", 5.0)):
        await gpus.store.record(ledger_row(team=team, actual_cost=cost))
    await gpus.store.rollup()
    await gpus.store.add_gpu_usage("gpu-a", datetime.now(UTC).replace(minute=0, second=0, microsecond=0), 3600, 3.0)
    c = (await client.get("/v1/admin/controls", headers=auth(ADMIN_KEY))).json()
    frac = c["month"]["fraction"]
    assert 0 < frac <= 1 and abs(frac - month_fraction()) < 0.01
    # The organization's spend: $25 of requests on APIs, and $3 of GPU time no request paid for (decision 64).
    assert c["spend"]["org"]["spent"] == 28.0 and c["spend"]["org"]["budget"] == 500.0
    assert c["spend"]["org"]["api"] == 25.0 and c["spend"]["org"]["gpu"] == 3.0 and c["spend"]["org"]["gpu_idle"] == 3.0
    assert abs(c["spend"]["org"]["projected"] - 28.0 / frac) < 1
    assert c["spend"]["gpu"]["by_backend"] == {"gpu-a": 3.0} and c["fleet"]["deployments"] == ["gpu-a", "gpu-b"]
    teams = {t["name"]: t for t in c["teams"]}
    assert teams["default"]["spent"] == 20.0 and teams["retired"]["unconfigured"] is True and teams["retired"]["spent"] == 5.0
    assert c["safeguards"]["org_budget_usd_month"] == 500.0 and c["controller"]["alive"] is False


def test_month_fraction_is_never_zero():
    assert month_fraction(datetime(2026, 11, 1, tzinfo=UTC)) > 0
    assert abs(month_fraction(datetime(2026, 2, 15, tzinfo=UTC)) - 14 / 28) < 1e-9
