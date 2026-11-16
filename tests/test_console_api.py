"""Console endpoints: scoping (team lead vs organization), the merit stack, backend health, teams, and attention items."""

import time
from datetime import UTC, datetime, timedelta

from conftest import ADMIN_KEY, LEAD_KEY, auth, ledger_row

from meritorder.backends.mock import MockBackend
from meritorder.config import BackendConfig, ModelConfig, Route, ServeConfig


async def get(client, path, key=ADMIN_KEY, **params):
    r = await client.get(path, params=params, headers=auth(key))
    assert r.status_code == 200, r.text
    return r.json()


async def seed_spend(gateway, team="default", cost=85.0, user="alice"):
    await gateway.store.record(ledger_row(user_id=user, team=team, actual_cost=cost, baseline_cost=cost * 2,
                                          savings_routing=cost, input_tokens=1000, cache_read_tokens=3000, output_tokens=200))
    await gateway.store.rollup()


def titles(att):
    return [i["title"] for i in att["items"]]


async def test_every_console_endpoint_needs_a_lead(client):
    for path in ("overview", "merit", "feed", "backends", "teams", "attention"):
        assert (await client.get(f"/v1/console/{path}", headers=auth())).status_code == 403, path
        assert (await client.get(f"/v1/console/{path}")).status_code == 401, path


async def test_overview_scope_budget_and_cache_hit_rate(client, gateway):
    await seed_spend(gateway, "default", 40.0)
    await seed_spend(gateway, "ops", 10.0, user="carol")
    org = await get(client, "/v1/console/overview")
    assert org["scope"] == "organization" and org["month"]["spend"] == 50.0 and org["today"]["requests"] == 2
    assert org["month"]["cache_hit_rate"] == 0.75  # 3000 cached of 4000 prompt tokens
    assert org["month"]["savings_by_cause"]["routing"] == 50.0
    lead = await get(client, "/v1/console/overview", LEAD_KEY)
    assert lead["scope"] == "default" and lead["month"]["spend"] == 40.0 and lead["budget"] == 100
    elapsed = (datetime.now(UTC) - datetime.now(UTC).replace(day=1, hour=0, minute=0, second=0, microsecond=0)).total_seconds()
    assert (lead["forecast"] is None) == (elapsed <= 0.25 * 86400)


async def test_overview_with_no_traffic(client, gateway):
    ov = await get(client, "/v1/console/overview")
    assert {k: v for k, v in ov["today"].items() if k != "savings_by_cause"} == {
        "spend": 0, "api": 0, "gpu": 0, "gpu_idle": 0, "baseline": 0, "saved": 0, "requests": 0, "cache_hit_rate": 0}
    assert ov["requests_per_minute"] == 0


async def test_merit_stack_orders_by_price_and_flags_gpus(client, gateway):
    gpu = MockBackend(BackendConfig(name="gpu", kind="self_hosted", format="openai", mock_latency_ms=0))
    gateway.backends["gpu"] = gpu
    gateway.cfg.models.append(ModelConfig(name="coder", quality={"coding": 4}, routes=[Route(backend="gpu", model="m")]))
    for model, cost, tokens in (("big", 3.0, 100_000), ("coder", 0.01, 400_000), ("small", 0.05, 100_000), ("idle", 1.0, 0)):
        await gateway.store.record(ledger_row(served_model=model, actual_cost=cost, input_tokens=tokens))
    m = await get(client, "/v1/console/merit", minutes=15)
    assert [t["model"] for t in m["tiers"]] == ["coder", "small", "big"]  # cheapest per token first; no-token rows dropped
    assert [t["gpu"] for t in m["tiers"]] == [True, False, False]
    assert m["clearing_price_per_m"] == m["tiers"][-1]["price_per_m"] == 30.0
    assert abs(m["blended_price_per_m"] - 3.06 / 600_000 * 1e6) < 1e-9
    assert (await client.get("/v1/console/merit?minutes=0", headers=auth(ADMIN_KEY))).status_code == 422
    empty = await get(client, "/v1/console/merit", LEAD_KEY, minutes=1)
    assert empty["tiers"] == [] or all(t["requests"] for t in empty["tiers"])


async def test_feed_limit_and_team_filter(client, gateway):
    for i in range(5):
        await gateway.state.push_event({"i": i, "team": "default" if i % 2 else "ops"})
    assert [e["i"] for e in (await get(client, "/v1/console/feed", limit=2))["events"]] == [4, 3]
    lead = (await get(client, "/v1/console/feed", LEAD_KEY, limit=10))["events"]
    assert [e["i"] for e in lead] == [3, 1]


async def test_backends_health_circuit_and_reasons(client, gateway):
    now = time.time()
    for _ in range(3):
        gateway.backends["mock-openai"].record_failure(now)
    gateway.backends["mock-anthropic"].draining = True
    await gateway.state.record_backend("mock-anthropic", True, 200, 50)
    await gateway.state.record_backend("mock-anthropic", False, 400, None)
    gateway.cfg.backends.append(BackendConfig(name="off", format="openai", base_url="https://off.example.com/v1",
                                              enabled=False, disabled_reason="OFF_KEY is not set"))
    b = {x["name"]: x for x in (await get(client, "/v1/console/backends", LEAD_KEY))["backends"]}
    assert b["mock-openai"]["circuit_open"] and not b["mock-anthropic"]["circuit_open"]
    a = b["mock-anthropic"]
    assert a["draining"] and a["requests"] == 2 and a["errors"] == 1 and a["error_rate"] == 0.5
    assert a["avg_latency_ms"] == 300 and a["avg_ttft_ms"] == 50 and a["models"] == ["big"]
    assert b["off"]["live"] is False and b["off"]["reason"] == "OFF_KEY is not set" and b["off"]["host"] == "off.example.com"
    assert b["off"]["avg_latency_ms"] is None and b["off"]["avg_ttft_ms"] is None


async def test_teams_spend_next_to_limits(client, gateway):
    await seed_spend(gateway, "default", 12.0)
    await seed_spend(gateway, "ghost", 3.0, user="gus")  # spend from a team that's no longer configured
    teams = {t["name"]: t for t in (await get(client, "/v1/console/teams"))["teams"]}
    assert teams["default"]["configured"] and teams["default"]["spend"] == 12.0 and teams["default"]["budget_usd_month"] == 100
    assert teams["ghost"]["configured"] is False and teams["ghost"]["spend"] == 3.0
    assert teams["default"]["baseline_model"] == "big" and "user_rpm" in teams["default"]["rate_limits"]
    lead = (await get(client, "/v1/console/teams", LEAD_KEY))["teams"]
    assert [t["name"] for t in lead] == ["default"]


async def test_attention_backend_problems(client, gateway):
    for _ in range(3):
        gateway.backends["mock-openai"].record_failure(time.time())
    for ok in (True, True, True, False, False):
        await gateway.state.record_backend("mock-anthropic", ok, 100, None)
    att = await get(client, "/v1/console/attention")
    assert titles(att)[0] == "mock-openai is failing" and att["items"][0]["level"] == "critical"
    assert "mock-anthropic: 40% errors" in titles(att)


async def test_attention_budgets(client, gateway):
    await seed_spend(gateway, "default", 85.0)
    att = await get(client, "/v1/console/attention", LEAD_KEY)
    [item] = [i for i in att["items"] if i["title"].startswith("Team default")]
    assert item["level"] == "warning" and "85%" in item["title"]
    await seed_spend(gateway, "default", 20.0)
    item = next(i for i in (await get(client, "/v1/console/attention"))["items"] if i["title"].startswith("Team default"))
    assert item["level"] == "critical"

    gateway.cfg.safeguards.org_budget_usd_month = 100.0
    gateway.cfg.safeguards.gpu_budget_usd_month = 10.0
    await gateway.store.add_gpu_usage("gpu", datetime.now(UTC).replace(minute=0, second=0, microsecond=0), 3600, 9.0)
    items = {i["title"]: i for i in (await get(client, "/v1/console/attention"))["items"]}
    # The organization's spend counts GPU time no request paid for: $105 of requests and $9 of idle GPU.
    assert items["Organization spend at 114% of budget"]["level"] == "critical"
    assert "Requests are being refused" in items["Organization spend at 114% of budget"]["detail"]
    assert items["GPU spend at 90% of budget"]["level"] == "warning"
    # Team leads don't see organization-wide budgets.
    assert not any("Organization" in t for t in titles(await get(client, "/v1/console/attention", LEAD_KEY)))


async def test_attention_gpu_items(client, gateway):
    await gateway.store.set_setting("gpu_controls", {"paused": True, "paused_by": "root", "reason": "audit"})
    gateway.cfg.backends.append(BackendConfig(name="gpu-a", kind="self_hosted", format="openai",
                                              serve=ServeConfig(model="small", provisioner="static", static_url="http://a/v1")))
    gateway.cfg.backends.append(BackendConfig(name="gpu-b", kind="self_hosted", format="openai",
                                              serve=ServeConfig(model="small", provisioner="static", static_url="http://b/v1")))
    await gateway.state.set_json("deploy:gpu-a", {"state": "warm", "reason": "stop failed: cloud API timeout"})
    await gateway.state.set_json("deploy:gpu-b", {"state": "api_only", "reason": "provision failed: no capacity"})
    att = await get(client, "/v1/console/attention")
    by = {i["title"]: i for i in att["items"]}
    assert by["All GPUs are stopped"]["detail"].startswith("By root: audit")
    gpu_items = [i for i in att["items"] if i["title"].startswith("GPU deployment")]
    assert [(i["title"], i["level"]) for i in gpu_items] == [("GPU deployment gpu-a", "critical"), ("GPU deployment gpu-b", "warning")]
    assert by["Capacity controller is not running"]["level"] == "critical"  # GPUs up, no heartbeat
    await gateway.state.set_json("controller:heartbeat", {"at": time.time(), "host": "c", "interval": 60})
    assert "Capacity controller is not running" not in titles(await get(client, "/v1/console/attention"))
    levels = [i["level"] for i in att["items"]]
    assert levels == sorted(levels, key={"critical": 0, "warning": 1, "info": 2}.get)


async def test_attention_recent_alerts_scoped_and_aged(client, gateway):
    now = datetime.now(UTC)
    await gateway.state.push_alert({"ts": (now - timedelta(hours=30)).isoformat(), "title": "old", "level": "warning"})
    await gateway.state.push_alert({"ts": now.isoformat(), "title": "ops only", "team": "ops", "level": "info"})
    await gateway.state.push_alert({"ts": now.isoformat(), "title": "everyone", "level": "warning", "page": "teams"})
    await gateway.state.push_alert({"title": "no timestamp"})
    await gateway.state.push_alert({"ts": "not a date", "title": "bad timestamp"})
    org = titles(await get(client, "/v1/console/attention"))
    assert "everyone" in org and "ops only" in org and "old" not in org and "no timestamp" not in org
    lead = titles(await get(client, "/v1/console/attention", LEAD_KEY))
    assert "everyone" in lead and "ops only" not in lead
