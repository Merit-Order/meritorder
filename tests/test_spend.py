"""GPU time in spend (spend.py, decision 64): totals, savings after GPU cost, team shares, the org budget, and every surface
that shows them. Each test runs on both stores (conftest's BACKINGS)."""

from datetime import UTC, datetime, timedelta

import pytest
from conftest import ADMIN_KEY, LEAD_KEY, auth, ledger_row

from meritorder import spend
from meritorder.alerts import check_budgets, weekly_digest
from meritorder.config import BackendConfig, ServeConfig
from meritorder.metrics import Metrics, cost_kind
from meritorder.store import SAVINGS_FIELDS, month_start

HOUR = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
TODAY = HOUR.replace(hour=0)


def gpu_backend(name: str = "gpu-small", serve: bool = True) -> BackendConfig:
    return BackendConfig(name=name, kind="self_hosted", format="openai", gpu_cost_per_hour=1.29, capacity_tokens_per_second=2000,
                         serve=ServeConfig(model="small", provisioner="static", static_url=f"http://{name}.test/v1") if serve else None)


def gpu_row(team: str = "default", user: str = "alice", backend: str = "gpu-small", ts: datetime | None = None, **kw) -> dict:
    """A request served on a GPU: its share of the GPU's time, the API cost it avoided, and the baseline."""
    costs = {"actual_cost": 0.00003, "served_api_cost": 0.0006, "baseline_cost": 0.00066, "savings_routing": 0.00006,
             "savings_self_hosting": 0.00057, **kw}
    return ledger_row(team=team, user_id=user, backend=backend, backend_kind="self_hosted", served_model="small",
                      input_tokens=1000, output_tokens=200, **({"ts": ts} if ts else {}), **costs)


def adds_up(r: dict) -> bool:
    """Savings by cause add up to baseline - spend, and spend to APIs + GPUs."""
    return (abs(sum(r[f] for f in SAVINGS_FIELDS) - r["savings_total"]) < 1e-6
            and abs(r["baseline_cost"] - r["actual_cost"] - r["savings_total"]) < 1e-6
            and abs(r["api_cost"] + r["gpu_cost"] - r["actual_cost"]) < 1e-6)


@pytest.fixture
def cfg_gpu(cfg):
    cfg.backends.append(gpu_backend())
    return cfg


async def live_scenario(store, ts: datetime | None = None) -> None:
    """The live run that showed the bug: a GPU up for most of a day ($1.6184 accrued, starting and idle included), three
    small requests served on it, and nothing else."""
    for _ in range(3):
        await store.record(gpu_row(ts=ts))
    await store.add_gpu_usage("gpu-small", ts or HOUR, 3 * 3600, 1.6184)


async def test_live_scenario_counts_the_gpu_and_savings_go_negative(store, cfg_gpu):
    await live_scenario(store)
    before = (await store.summary(TODAY, TODAY + timedelta(days=1), "none"))[0]
    assert round(before["actual_cost"], 6) == 0.00009 and before["savings_total"] > 0  # what Dispatch showed: GPUs look free
    t = await spend.total(store, cfg_gpu, TODAY, TODAY + timedelta(days=1))
    assert t["actual_cost"] == 1.6184 and t["gpu_cost"] == 1.6184 and t["api_cost"] == 0.0
    assert t["gpu_idle_cost"] == round(1.6184 - 0.00009, 6)
    # Saved = baseline - spend, after the GPUs: $0.00198 - $1.6184, below zero, and the causes still add up.
    assert t["savings_total"] == round(0.00198 - 1.6184, 6) and adds_up(t)
    assert t["savings_self_hosting"] == round(3 * 0.0006 - 1.6184, 6)  # API cost avoided less what the GPU cost
    assert t["savings_routing"] == round(3 * 0.00006, 6)


async def test_a_gpu_with_no_requests_is_still_spend(store, cfg_gpu):
    await store.add_gpu_usage("gpu-small", HOUR, 3600, 1.29)
    t = await spend.total(store, cfg_gpu, TODAY, TODAY + timedelta(days=1))
    assert t["requests"] == 0 and t["actual_cost"] == 1.29 and t["savings_total"] == -1.29 and t["savings_self_hosting"] == -1.29
    by_day = await spend.summary(store, cfg_gpu, TODAY - timedelta(days=1), TODAY + timedelta(days=1), "day")
    assert len(by_day) == 1 and str(by_day[0]["key"])[:10] == TODAY.date().isoformat() and by_day[0]["gpu_idle_cost"] == 1.29
    by_hour = await spend.summary(store, cfg_gpu, TODAY, TODAY + timedelta(days=1), "hour")
    assert [datetime.fromisoformat(str(r["key"])) for r in by_hour] == [HOUR]


async def test_teams_carry_their_requests_shares_and_idle_time_is_the_organizations(store, cfg_gpu):
    await store.record(gpu_row(team="default"))
    await store.record(gpu_row(team="default"))
    await store.record(gpu_row(team="eng", user="bob"))
    await store.record(ledger_row(team="eng", user_id="bob", actual_cost=0.5, served_api_cost=0.5, baseline_cost=0.6, savings_routing=0.1))
    await store.add_gpu_usage("gpu-small", HOUR, 3600, 1.0)
    end = TODAY + timedelta(days=1)
    rows = {r["key"]: r for r in await spend.summary(store, cfg_gpu, TODAY, end, "team")}
    assert set(rows) == {"default", "eng", spend.IDLE_KEY}
    assert rows["default"]["gpu_cost"] == 0.00006 and rows["default"]["api_cost"] == 0.0 and rows["default"]["gpu_idle_cost"] == 0
    assert rows["eng"]["gpu_cost"] == 0.00003 and rows["eng"]["api_cost"] == 0.5
    idle = rows[spend.IDLE_KEY]
    assert idle["requests"] == 0 and idle["actual_cost"] == round(1.0 - 0.00009, 6) and idle["savings_self_hosting"] == -idle["actual_cost"]
    org = await spend.total(store, cfg_gpu, TODAY, end)
    assert abs(sum(r["actual_cost"] for r in rows.values()) - org["actual_cost"]) < 1e-6  # teams + unallocated = the organization
    assert org["actual_cost"] == 1.5 and all(adds_up(r) for r in [*rows.values(), org])
    # A team's own view: its requests' shares, never the idle time.
    lead = await spend.total(store, cfg_gpu, TODAY, end, team="default")
    assert lead["actual_cost"] == 0.00006 and lead["gpu_idle_cost"] == 0
    person = await spend.summary(store, cfg_gpu, TODAY, end, "model", user_id="bob")
    assert spend.IDLE_KEY not in {r["key"] for r in person}
    # By backend, a GPU's row is what it cost to run; by user or tag the idle time is its own row again.
    backends = {r["key"]: r for r in await spend.summary(store, cfg_gpu, TODAY, end, "backend")}
    assert backends["gpu-small"]["actual_cost"] == 1.0 and backends["mock"]["gpu_cost"] == 0
    assert spend.IDLE_KEY in {r["key"] for r in await spend.summary(store, cfg_gpu, TODAY, end, "user")}


async def test_a_gpu_no_controller_runs_keeps_its_requests_shares(store, cfg):
    cfg.backends.append(gpu_backend("own-vllm", serve=False))  # your own vLLM: no GPU time accrued, its shares are its cost
    await store.record(gpu_row(backend="own-vllm"))
    t = await spend.total(store, cfg, TODAY, TODAY + timedelta(days=1))
    assert t["actual_cost"] == 0.00003 and t["gpu_cost"] == 0.00003 and t["gpu_idle_cost"] == 0 and adds_up(t)
    org = await spend.budget_spend(store, cfg, "org")
    assert org["gpu"] == 0.00003 and org["gpu_by_backend"] == {"own-vllm": 0.00003} and org["gpu_idle"] == 0


async def test_shares_above_the_gpu_cost_show_as_a_credit(store, cfg_gpu):
    """capacity_tokens_per_second set below what the GPU serves: the requests' shares come to more than the GPU cost,
    and the organization's total is still the GPU's cost."""
    await store.record(gpu_row(actual_cost=0.5, served_api_cost=0.6, baseline_cost=0.66, savings_routing=0.06, savings_self_hosting=0.1))
    await store.add_gpu_usage("gpu-small", HOUR, 3600, 0.2)
    t = await spend.total(store, cfg_gpu, TODAY, TODAY + timedelta(days=1))
    assert t["actual_cost"] == 0.2 and t["gpu_idle_cost"] == -0.3 and adds_up(t)


async def test_budget_spend_splits_apis_and_gpus(store, cfg_gpu):
    await live_scenario(store)
    await store.record(ledger_row(actual_cost=2.0, baseline_cost=2.0))
    org = await spend.budget_spend(store, cfg_gpu, "org")
    assert round(org["spent"], 6) == 3.6184 and round(org["api"], 6) == 2.0 and round(org["gpu"], 6) == 1.6184
    assert round(org["gpu_idle"], 6) == round(1.6184 - 0.00009, 6) and org["gpu_by_backend"] == {"gpu-small": 1.6184}
    assert round(await spend.gpu_idle(store, cfg_gpu, month_start()), 6) == round(org["gpu_idle"], 6)


async def test_long_ranges_read_the_rollup_and_agree(store, cfg_gpu):
    """Over more than a week Postgres reads the hourly rollup for both the ledger's sums and the GPU shares."""
    old = HOUR - timedelta(days=3)
    await live_scenario(store, ts=old)
    await store.rollup(full=True)
    t = await spend.total(store, cfg_gpu, TODAY - timedelta(days=20), TODAY + timedelta(days=1))
    assert t["actual_cost"] == 1.6184 and t["gpu_idle_cost"] == round(1.6184 - 0.00009, 6) and adds_up(t)
    months = await spend.summary(store, cfg_gpu, TODAY - timedelta(days=20), TODAY + timedelta(days=1), "month")
    assert abs(sum(r["actual_cost"] for r in months) - 1.6184) < 1e-6


# ---------------------------------------------------------------- the surfaces


@pytest.fixture
def gw_gpu(gateway):
    gateway.cfg.backends.append(gpu_backend())
    return gateway


async def test_dispatch_and_reports_show_the_gpu(client, gw_gpu):
    await live_scenario(gw_gpu.store)
    await gw_gpu.store.rollup()  # month-long reports read the hourly rollup on Postgres, as the worker keeps it
    ov = (await client.get("/v1/console/overview", headers=auth(ADMIN_KEY))).json()
    today = ov["today"]
    assert today["spend"] == 1.6184 and today["gpu"] == 1.6184 and today["api"] == 0 and today["saved"] < 0
    assert today["gpu_idle"] == round(1.6184 - 0.00009, 6) and today["savings_by_cause"]["self_hosting"] < 0
    lead = (await client.get("/v1/console/overview", headers=auth(LEAD_KEY))).json()
    assert lead["today"]["spend"] == 0.00009 and lead["today"]["gpu_idle"] == 0  # the team's own shares
    gw_gpu.cfg.safeguards.org_budget_usd_month = 50.0  # the organization's pace is against its own budget, GPU time and all
    assert (await client.get("/v1/console/overview", headers=auth(ADMIN_KEY))).json()["budget"] == 50.0
    s = (await client.get("/v1/reports/summary", params={"group_by": "team"}, headers=auth(ADMIN_KEY))).json()
    assert {r["key"] for r in s["rows"]} == {"default", spend.IDLE_KEY} and s["totals"]["actual_cost"] == 1.6184
    assert s["totals"]["gpu_idle_cost"] == round(1.6184 - 0.00009, 6)
    csv = (await client.get("/v1/reports/summary.csv", params={"group_by": "day"}, headers=auth(ADMIN_KEY))).text
    assert csv.splitlines()[0].endswith("savings_total,api_cost,gpu_cost,gpu_idle_cost") and "1.6184" in csv
    me = (await client.get("/v1/reports/me", headers=auth())).json()
    assert me["gpu_usd"] == 0.00009 and me["api_usd"] == 0


async def test_controls_and_alerts_count_the_gpu_in_the_org_budget(client, gw_gpu):
    gw_gpu.cfg.safeguards.org_budget_usd_month = 2.0
    await live_scenario(gw_gpu.store)
    c = (await client.get("/v1/admin/controls", headers=auth(ADMIN_KEY))).json()
    assert c["spend"]["org"]["spent"] == 1.6184 and c["spend"]["org"]["gpu"] == 1.6184 and c["spend"]["org"]["api"] == 0
    sent = await check_budgets(gw_gpu.cfg, gw_gpu.store, gw_gpu.state)
    assert any(m.startswith("Organization spend is at 81%") for m in sent)  # $1.62 of $2, nearly all of it GPU time


async def test_org_budget_refuses_once_gpu_time_uses_it_up(client, gw_gpu, monkeypatch):
    body = {"model": "small", "messages": [{"role": "user", "content": "hi"}]}
    gw_gpu.cfg.safeguards.org_budget_usd_month = 1.0
    assert (await client.post("/v1/chat/completions", json=body, headers=auth())).status_code == 200
    await gw_gpu.store.add_gpu_usage("gpu-small", HOUR, 3600, 1.29)  # an hour of an idle GPU
    r = await client.post("/v1/chat/completions", json=body, headers=auth())
    assert r.status_code == 200  # read again once a minute: the gateway still holds the amount it read
    gw_gpu._gpu_idle = None
    r = await client.post("/v1/chat/completions", json=body, headers=auth())
    assert r.status_code == 429 and "GPU time included" in r.json()["error"]["message"]
    assert r.headers["x-meritorder-limit"] == "budget"

    async def broken(*a, **kw):
        raise RuntimeError("database away")

    # A store that can't answer leaves the last amount read standing.
    monkeypatch.setattr(spend, "gpu_idle", broken)
    gw_gpu._gpu_idle = (gw_gpu._gpu_idle[0], 0.0, gw_gpu._gpu_idle[2])
    assert await gw_gpu.gpu_idle() == 1.29
    gw_gpu._gpu_idle = None
    assert await gw_gpu.gpu_idle() == 0.0


async def test_explore_up_headroom_counts_gpu_time(gw_gpu):
    from meritorder.proxy import _month_key, headroom

    gw_gpu.cfg.safeguards.org_budget_usd_month = 10.0
    p = await gw_gpu.authn.authenticate(auth())
    team = gw_gpu.cfg.team(p.team)
    pre = {"spend": {f"{_month_key()}:org:all": 5.0}}
    assert headroom(pre, p, team, gw_gpu.cfg)
    assert not headroom({**pre, "gpu_idle": 4.5}, p, team, gw_gpu.cfg)  # $9.50 of $10: past 90%


async def test_weekly_digest_has_the_organizations_gpu_time(gw_gpu):
    await live_scenario(gw_gpu.store, ts=datetime.now(UTC) - timedelta(hours=1))
    sent = await weekly_digest(gw_gpu.cfg, gw_gpu.store, gw_gpu.state, force=True)
    team, org = sent
    assert team.startswith("Weekly LLM spend for 'default': $0.00 (APIs $0.00, its share of GPU time $0.00)")
    assert org.startswith("Weekly LLM spend for the organization: $1.62 (APIs $0.00, GPUs $1.62, $1.62 of it GPU time no request")
    assert "Cost $1.62 more than the baseline" in org and "GPUs that cost more than the API calls they replaced" in org


async def test_cost_metric_says_what_kind_of_cost(cfg_gpu):
    cfg_gpu.backends.append(gpu_backend("own-vllm", serve=False))
    assert [cost_kind(cfg_gpu, b) for b in ("gpu-small", "own-vllm", "mock", "gone", None)] == ["gpu_share", "gpu", "api", "api", "api"]
    m = Metrics()
    m.record({"served_model": "small", "backend": "gpu-small", "actual_cost": 0.1, "status": 200}, "gpu_share")
    assert 'meritorder_cost_usd_total{backend="gpu-small",kind="gpu_share",model="small"} 0.1' in m.render()


async def test_controls_and_admission_count_the_same_org_spend_while_a_batch_runs(client, gateway):
    """A running batch's reservation counts against budgets when a request is admitted; Controls, the attention panel and
    the budget alerts count the same total (spend.budget_spend), and show the reservation on its own."""
    from meritorder.batches.service import month_key, spend_keys

    await client.post("/v1/chat/completions", headers=auth(), json={"model": "small", "messages": [{"role": "user", "content": "hi"}]})
    await gateway.drain()
    now = datetime.now(UTC)
    # As BatchService.create does: the row with its reservation, and the same amount added to the month's spend counters.
    await gateway.store.create_batch({"id": "batch_audit", "created_at": now, "user_id": "alice", "team": "default", "client_format": "openai",
                                      "status": "in_progress", "estimated_cost": 5.0, "reserved_cost": 5.0, "reserve_month": month_key(now),
                                      "request_count": 1}, [], [])
    await gateway.state.adjust_spend(spend_keys("alice", "default", month_key(now)), 5.0)
    admitted = await gateway.spend("org", "all")  # what admit() compares with org_budget_usd_month
    c = (await client.get("/v1/admin/controls", headers=auth(ADMIN_KEY))).json()
    shown = c["spend"]["org"]
    assert shown["spent"] == pytest.approx(admitted, abs=1e-6), f"Controls shows ${shown['spent']}, admission counts ${admitted}"
    assert shown["reserved"] == 5.0 and 0 < shown["spent"] - 5.0 < 0.01  # sub-cent spend shows, not rounded away
    team = next(t for t in c["teams"] if t["name"] == "default")
    assert team["spent"] == pytest.approx(await gateway.spend("team", "default"), abs=1e-6) and team["reserved"] == 5.0
    ov = (await client.get("/v1/console/overview", headers=auth(ADMIN_KEY))).json()
    assert ov["reserved"] == 5.0
    # A person's budget alert counts the reservation too, as admission does.
    gateway.cfg.team("default").user_budget_usd_month = 6.0
    await gateway.store.rollup()  # who spent this month, from the hourly rollup on Postgres
    sent = await check_budgets(gateway.cfg, gateway.store, gateway.state)
    assert any(m.startswith("alice (default) has used 8") for m in sent), sent


async def test_an_estimated_row_on_a_gpu_moves_cost_from_idle_to_its_team(store, cfg_gpu):
    """A broken-off stream on a GPU is recorded with estimated usage (proxy.Gateway.broken_off_usage, tagged usage:
    estimated) and its share: the team pays it, the organization's total stays the GPU's time."""
    await live_scenario(store)
    end = TODAY + timedelta(days=1)
    before = await spend.total(store, cfg_gpu, TODAY, end)
    await store.record(gpu_row(status=499, tags={"usage": "estimated"}))
    after = await spend.total(store, cfg_gpu, TODAY, end)
    assert after["actual_cost"] == before["actual_cost"] == 1.6184 and after["gpu_idle_cost"] == round(before["gpu_idle_cost"] - 0.00003, 6)
    team = await spend.total(store, cfg_gpu, TODAY, end, team="default")
    assert team["gpu_cost"] == 0.00012 and adds_up(after)
