"""Reports API: who sees what (user / team lead / finance / admin), every grouping, CSV and JSONL export, GPU reports."""

import csv
import io
import json
from datetime import UTC, datetime, timedelta

import pytest
from conftest import ADMIN_KEY, LEAD_KEY, TEST_KEY, auth, ledger_row

from meritorder.config import BackendConfig, ModelConfig, Route, ServeConfig

NOW = datetime.now(UTC)


async def seed(gateway):
    """alice and bob in team default, carol in ops; one tagged request; one error."""
    rows = [
        ledger_row(user_id="alice", team="default", served_model="small", backend="mock-openai", actual_cost=0.10,
                   baseline_cost=0.50, savings_routing=0.40, tags={"project": "apollo"}),
        ledger_row(user_id="alice", team="default", served_model="big", backend="mock-anthropic", actual_cost=1.00,
                   baseline_cost=1.00),
        ledger_row(user_id="bob", team="default", served_model="big", backend="mock-anthropic", actual_cost=2.00,
                   baseline_cost=2.00, status=500, error="upstream failed"),
        ledger_row(user_id="carol", team="ops", served_model="small", backend="mock-openai", actual_cost=4.00,
                   baseline_cost=8.00, savings_caching=4.00, upgraded=True),
    ]
    for r in rows:
        await gateway.store.record(r)
    await gateway.store.rollup()  # Postgres reads month-long ranges from the hourly rollup (the worker keeps it fresh)


async def finance_key(gateway) -> str:
    return (await gateway.store.create_key("fin", "ops", "finance", name="finance"))[1]


async def summary(client, key, **params):
    r = await client.get("/v1/reports/summary", params=params, headers=auth(key))
    assert r.status_code == 200, r.text
    return r.json()


async def test_role_scoping_on_summary(client, gateway):
    await seed(gateway)
    fin = await finance_key(gateway)

    # A user sees only themselves, whatever they ask for.
    for params in ({}, {"user": "bob"}, {"team": "ops"}):
        d = await summary(client, TEST_KEY, group_by="user", **params)
        assert [r["key"] for r in d["rows"]] == ["alice"] and d["user"] == "alice" and d["team"] is None

    # A team lead sees their team; asking for another team is clamped to their own.
    d = await summary(client, LEAD_KEY, group_by="user")
    assert {r["key"] for r in d["rows"]} == {"alice", "bob"} and d["team"] == "default"
    assert (await summary(client, LEAD_KEY, group_by="user", team="ops"))["team"] == "default"
    assert (await summary(client, LEAD_KEY, group_by="user", user="carol"))["rows"] == []  # carol isn't on their team
    assert [r["key"] for r in (await summary(client, LEAD_KEY, group_by="user", user="bob"))["rows"]] == ["bob"]

    # Finance and admins see everyone, and can filter.
    for key in (fin, ADMIN_KEY):
        d = await summary(client, key, group_by="team")
        assert {r["key"]: r["actual_cost"] for r in d["rows"]} == {"default": 3.1, "ops": 4.0}
        assert d["totals"]["requests"] == 4 and d["totals"]["errors"] == 1 and d["totals"]["upgraded"] == 1
        assert d["totals"]["actual_cost"] == pytest.approx(7.1) and d["totals"]["savings_total"] == pytest.approx(4.4)
        assert [r["key"] for r in (await summary(client, key, group_by="user", team="ops"))["rows"]] == ["carol"]


@pytest.mark.parametrize("group_by, expected", [
    ("model", {"small": 2, "big": 2}),
    ("backend", {"mock-openai": 2, "mock-anthropic": 2}),
    ("none", {"total": 4}),
    ("tag:project", {"apollo": 1, "(untagged)": 3}),
])
async def test_every_grouping(client, gateway, group_by, expected):
    await seed(gateway)
    d = await summary(client, ADMIN_KEY, group_by=group_by)
    assert {str(r["key"]): r["requests"] for r in d["rows"]} == expected


@pytest.mark.parametrize("group_by", ["day", "hour", "month"])
async def test_time_groupings(client, gateway, group_by):
    await seed(gateway)
    d = await summary(client, ADMIN_KEY, group_by=group_by)
    assert sum(r["requests"] for r in d["rows"]) == 4 and len(d["rows"]) >= 1


async def test_date_range_and_bad_input(client, gateway):
    await seed(gateway)
    today = datetime.now(UTC).date()  # the rows' day: a run that starts before midnight (UTC) gets here after it
    d = await summary(client, ADMIN_KEY, group_by="none", start=str(today), end=str(today + timedelta(days=1)))
    assert d["rows"][0]["requests"] == 4 and d["start"].startswith(str(today))
    past = await summary(client, ADMIN_KEY, group_by="none", start="2020-01-01", end="2020-02-01")
    assert past["rows"] == [] and past["totals"]["requests"] == 0
    for params in ({"start": "yesterday"}, {"end": "2026-13-40"}):
        r = await client.get("/v1/reports/summary", params=params, headers=auth(ADMIN_KEY))
        assert r.status_code == 400 and "YYYY-MM-DD" in r.text
    for bad in ("nonsense", "tag:", "tag:bad name", "user; DROP TABLE requests"):
        r = await client.get("/v1/reports/summary", params={"group_by": bad}, headers=auth(ADMIN_KEY))
        assert r.status_code == 400, bad
    assert (await client.get("/v1/reports/summary")).status_code == 401
    assert (await client.get("/v1/reports/summary", headers=auth("mo-not-a-key"))).status_code == 401


async def test_summary_csv(client, gateway):
    await seed(gateway)
    r = await client.get("/v1/reports/summary.csv", params={"group_by": "tag:project"}, headers=auth(ADMIN_KEY))
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv")
    assert 'filename="meritorder-tag-project.csv"' in r.headers["content-disposition"]
    rows = list(csv.DictReader(io.StringIO(r.text)))
    assert {row["key"]: row["requests"] for row in rows} == {"apollo": "1", "(untagged)": "3"}
    assert {"key", "requests", "actual_cost", "savings_total"} <= set(rows[0])
    # Same scoping as JSON: a user's CSV holds only their rows.
    mine = list(csv.DictReader(io.StringIO((await client.get("/v1/reports/summary.csv?group_by=user", headers=auth(TEST_KEY))).text)))
    assert [row["key"] for row in mine] == ["alice"]
    assert (await client.get("/v1/reports/summary.csv?group_by=bogus", headers=auth(ADMIN_KEY))).status_code == 400


async def test_me(client, gateway):
    await seed(gateway)
    me = (await client.get("/v1/reports/me", headers=auth(TEST_KEY))).json()
    assert me["user"] == "alice" and me["role"] == "user" and me["requests"] == 2
    assert me["budget_usd"] == 10 and me["team_spend_usd"] is None  # only leads see team spend
    assert {r["key"] for r in me["by_model"]} == {"small", "big"} and me["savings_usd"] == pytest.approx(0.4)
    lead = (await client.get("/v1/reports/me", headers=auth(LEAD_KEY))).json()
    assert lead["team_spend_usd"] == pytest.approx(3.1) and lead["team_budget_usd"] == 100


async def test_export_jsonl_scoping(client, gateway):
    await seed(gateway)
    assert (await client.get("/v1/reports/export.jsonl", headers=auth(TEST_KEY))).status_code == 403

    def users(r):
        assert r.status_code == 200 and r.headers["content-type"].startswith("application/x-ndjson")
        rows = [json.loads(line) for line in r.text.splitlines() if line]
        assert all("messages" not in row and "prompt" not in row for row in rows)  # metadata only
        return sorted(row["user_id"] for row in rows)

    assert users(await client.get("/v1/reports/export.jsonl", headers=auth(LEAD_KEY))) == ["alice", "alice", "bob"]
    r = await client.get("/v1/reports/export.jsonl", headers=auth(await finance_key(gateway)))
    assert r.status_code == 403 and "summary.csv" in r.text  # finance sees aggregates only
    r = await client.get("/v1/reports/export.jsonl", headers=auth(ADMIN_KEY))
    assert users(r) == ["alice", "alice", "bob", "carol"] and ".jsonl" in r.headers["content-disposition"]
    old = await client.get("/v1/reports/export.jsonl", params={"start": "2020-01-01", "end": "2020-01-02"}, headers=auth(ADMIN_KEY))
    assert users(old) == []


async def test_export_jsonl_from_either_store_makes_typed_parquet(client, gateway, tmp_path):
    """What `meritorder export --format parquet` writes, from the rows Postgres returns (UUIDs, timestamps, jsonb tags)."""
    pq = pytest.importorskip("pyarrow.parquet")
    from meritorder.export import LedgerWriter

    before = datetime.now(UTC)
    await seed(gateway)
    r = await client.get("/v1/reports/export.jsonl", headers=auth(ADMIN_KEY))
    w = LedgerWriter("parquet", tmp_path / "ledger.parquet")
    for line in r.text.splitlines():
        w.write(line)
    w.close()
    rows = sorted(pq.read_table(tmp_path / "ledger.parquet").to_pylist(), key=lambda x: (x["user_id"], x["actual_cost"]))
    assert [(x["user_id"], x["actual_cost"], x["status"], x["upgraded"]) for x in rows] == [
        ("alice", 0.1, 200, False), ("alice", 1.0, 200, False), ("bob", 2.0, 500, False), ("carol", 4.0, 200, True)]
    assert rows[0]["tags"] == [("project", "apollo")] and rows[2]["error"] == "upstream failed"
    assert before <= rows[0]["ts"] <= datetime.now(UTC) and len(rows[0]["id"]) == 36


async def test_requests_detail_scoping(client, gateway):
    await seed(gateway)
    rows = lambda r: sorted(x["user_id"] for x in r.json()["rows"])  # noqa: E731
    assert rows(await client.get("/v1/reports/requests?user=carol", headers=auth(TEST_KEY))) == ["alice", "alice"]
    assert rows(await client.get("/v1/reports/requests?team=ops", headers=auth(LEAD_KEY))) == ["alice", "alice", "bob"]
    assert rows(await client.get("/v1/reports/requests?team=ops", headers=auth(ADMIN_KEY))) == ["carol"]
    assert len((await client.get("/v1/reports/requests?limit=2", headers=auth(ADMIN_KEY))).json()["rows"]) == 2
    r = await client.get("/v1/reports/requests", headers=auth(await finance_key(gateway)))
    assert r.status_code == 403 and "aggregates only" in r.text
    assert (await client.get("/v1/reports/requests?limit=0", headers=auth(ADMIN_KEY))).status_code == 422


def add_gpu_backend(gateway, with_api_route: bool):
    from meritorder.backends.mock import MockBackend

    b = BackendConfig(name="gpu", kind="self_hosted", format="openai", gpu_cost_per_hour=4.0,
                      serve=ServeConfig(model="open-coder", provisioner="static", static_url="http://gpu.test/v1",
                                        accelerators="L4:1", max_replicas=2))
    gateway.cfg.backends.append(b)
    gateway.backends["gpu"] = MockBackend(BackendConfig(name="gpu", kind="self_hosted", format="openai"))
    routes = [Route(backend="gpu", model="m", price={"input": 0.1, "output": 0.2})]
    if with_api_route:
        routes.append(Route(backend="mock-openai", model="m", price={"input": 0.1, "output": 0.2}))
    gateway.cfg.models.append(ModelConfig(name="open-coder", hf_id="org/open-coder", quality={"coding": 4, "general": 3},
                                          routes=routes))


async def test_gpu_report(client, gateway):
    add_gpu_backend(gateway, with_api_route=False)
    await gateway.state.set_json("deploy:gpu", {"state": "warm", "replicas": 1, "endpoint": "http://gpu.test/v1",
                                                "burn_usd_per_hour": 4.0})
    hour = NOW.replace(minute=0, second=0, microsecond=0)
    await gateway.store.add_gpu_usage("gpu", hour, 3600, 4.0)
    assert (await client.get("/v1/reports/gpu", headers=auth(TEST_KEY))).status_code == 403
    d = (await client.get("/v1/reports/gpu", headers=auth(LEAD_KEY))).json()
    [dep] = d["deployments"]
    assert dep["backend"] == "gpu" and dep["state"] == "warm" and dep["managed"] and dep["model"] == "open-coder"
    assert dep["only_route_for"] == ["open-coder"]  # no API route: requests naming it fail while it's off
    assert dep["spec"]["accelerators"] == "L4:1" and dep["month_cost"] == 4.0 and dep["replicas"] == 1
    assert d["gpu_month"]["spent"] == 4.0 and d["paused"] is None and "controller" in d and "fleet" in d
    assert sum(x["cost"] for x in d["daily"]) == 4.0


async def test_gpu_report_shows_where_kubernetes_deployments_run(client, gateway):
    from meritorder.config import KubernetesServe, NodeType

    add_gpu_backend(gateway, with_api_route=True)
    serve = gateway.cfg.backend("gpu").serve
    serve.provisioner, serve.kubernetes = "kubernetes", KubernetesServe(node_type="g6.12xlarge")
    gateway.cfg.kubernetes.namespace = "gpus"
    gateway.cfg.kubernetes.node_types = {"g6.12xlarge": NodeType(usd_per_hour=4.6, gpus=4)}
    [dep] = (await client.get("/v1/reports/gpu", headers=auth(ADMIN_KEY))).json()["deployments"]
    assert dep["provisioner"] == "kubernetes" and dep["spec"]["kubernetes"] == {
        "api": "kserve", "namespace": "gpus", "node_type": "g6.12xlarge", "node_usd_per_hour": 4.6, "node_gpus": 4, "inference_pool": None}
    serve.kubernetes = None
    [dep] = (await client.get("/v1/reports/gpu", headers=auth(ADMIN_KEY))).json()["deployments"]
    assert dep["spec"]["kubernetes"]["node_type"] is None and dep["spec"]["kubernetes"]["node_usd_per_hour"] is None


async def test_gpu_report_with_api_route_and_paused_fleet(client, gateway):
    add_gpu_backend(gateway, with_api_route=True)
    await gateway.store.set_setting("gpu_controls", {"paused": True, "paused_by": "root", "reason": "audit"})
    d = (await client.get("/v1/reports/gpu", headers=auth(ADMIN_KEY))).json()
    assert d["deployments"][0]["only_route_for"] == [] and d["deployments"][0]["state"] == "api_only"
    assert d["paused"] is not None


async def test_gpu_candidates_and_whatif(client, gateway):
    add_gpu_backend(gateway, with_api_route=True)
    cands = (await client.get("/v1/reports/gpu/candidates", headers=auth(TEST_KEY))).json()["models"]
    assert [c["id"] for c in cands] == ["open-coder"] and cands[0]["hf_id"] == "org/open-coder"
    for _ in range(3):
        await gateway.store.record(ledger_row(served_model="big", floor="coding:3", served_api_cost=2.0, input_tokens=5000,
                                              output_tokens=500))
    q = {"model": "open-coder", "gpu_usd_per_hour": 4, "days": 7}
    assert (await client.get("/v1/reports/gpu/whatif", params=q, headers=auth(TEST_KEY))).status_code == 403
    r = await client.get("/v1/reports/gpu/whatif", params=q, headers=auth(LEAD_KEY))
    assert r.status_code == 200, r.text
    w = r.json()
    assert w["self_hostable"] is True and w["days"] == 7 and w["gpu_usd_per_hour"] == 4 and "Upper bound" in w["note"]
    r = await client.get("/v1/reports/gpu/whatif", params={**q, "model": "nope"}, headers=auth(LEAD_KEY))
    assert r.status_code == 404
    r = await client.get("/v1/reports/gpu/whatif", params={**q, "model": "embed-small"}, headers=auth(LEAD_KEY))
    assert r.status_code == 404  # embeddings aren't chat models
    r = await client.get("/v1/reports/gpu/whatif", params={**q, "gpu_usd_per_hour": 0}, headers=auth(LEAD_KEY))
    assert r.status_code == 422


async def test_gpu_breakeven(client, gateway, monkeypatch):
    from meritorder import gpu_sizing

    add_gpu_backend(gateway, with_api_route=True)
    gateway.cfg.backend("gpu").capacity_tokens_per_second = 500
    await gateway.state.set_json("deploy:gpu", {"state": "api_only", "price": {"usd_per_hour": 3.5, "source": "configured"}})
    now = datetime.now(UTC).isoformat(timespec="seconds")
    offers = [{"cloud": "aws", "accelerator": "L4", "count": 1, "on_demand": {"usd_per_hour": 0.8, "region": "us-east-1", "instance_type": "g6"},
               "spot": {"usd_per_hour": 0.3, "region": "us-west-2", "instance_type": "g6"}}]
    await gateway.store.set_setting("gpu_catalog", {"offers": offers, "fetched_at": now, "catalog_updated": {"aws": now}, "skypilot": "0.14.2",
                                                    "clouds": ["aws"]})
    await gateway.store.set_setting("model_sizes", {"fetched_at": now, "version": gpu_sizing.SIZES_VERSION, "models": {"org/open-coder": {
        "weight_bytes": 8e9, "bytes_by_bits": {"16": 8e9, "8": 0, "4": 0}, "params_by_bits": {"16": 4e9, "8": 0, "4": 0}, "bits": 16,
        "published_bits": 16, "total_params": 4e9, "active_params": 4e9, "basis": "test"}}})
    for _ in range(3):
        await gateway.store.record(ledger_row(served_model="big", floor="coding:3", served_api_cost=2.0, input_tokens=5000,
                                              cache_read_tokens=4000, output_tokens=1000))
    assert (await client.get("/v1/reports/gpu/breakeven", headers=auth(TEST_KEY))).status_code == 403
    r = await client.get("/v1/reports/gpu/breakeven", params={"days": 7, "efficiency": 0.2}, headers=auth(LEAD_KEY))
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["days"] == 7 and d["assumptions"] | {"clouds": None} == {"utilization": 0.7, "margin": 0.2, "efficiency": 0.2, "spot": True,
                                                                        "weights": "published", "fit": 0.7, "clouds": None}
    assert d["assumptions"]["clouds"] == ["aws"] and d["prices"]["skypilot"] == "0.14.2" and d["prices"]["offers"] == 1
    assert d["prices"]["refreshing"] is False and d["prices"]["problem"] is None  # fresh prices: nothing to fetch
    assert d["mix"] == {"input": 0.5, "cache_read": 0.4, "cache_write": 0.0, "output": 0.1, "tokens_per_request": 10_000, "source": "traffic"}
    coder = next(m for m in d["models"] if m["model"] == "open-coder")
    # A server it already runs (static): the controller's price and its measured capacity. Its traffic is worth $200/M.
    assert coder["gpu"]["deployment"] == "gpu" and coder["gpu"]["usd_per_hour"] == 3.5 and coder["gpu"]["source"] == "controller"
    assert coder["gpu"]["capacity_tokens_per_second"] == 500 and coder["api_usd_per_million"] == 200.0 and coder["pays"]
    assert coder["size"]["active_params"] == 4e9 and coder["size"]["weights"] == "16-bit" and coder["size"]["quality_note"] is None
    assert d["hostable"]["coding"]["model"] == "open-coder" and d["hostable"]["coding"]["level"] == int(coder["quality"]["coding"])
    g = next(x for x in d["groups"] if x["floor"] == "coding:3")
    assert g["gpu"]["model"] == "open-coder" and g["traffic"]["requests_per_hour"] > 0
    assert (await client.get("/v1/reports/gpu/breakeven", params={"utilization": 1.5}, headers=auth(LEAD_KEY))).status_code == 422
    narrowed = (await client.get("/v1/reports/gpu/breakeven", params={"clouds": "gcp", "spot": "false"}, headers=auth(LEAD_KEY))).json()
    assert narrowed["assumptions"]["clouds"] == ["gcp"] and narrowed["assumptions"]["spot"] is False
    four = (await client.get("/v1/reports/gpu/breakeven", params={"weights": "4bit"}, headers=auth(LEAD_KEY))).json()
    coder = next(m for m in four["models"] if m["model"] == "open-coder")  # 8 GB at 16 bits is 2.1 GB at 4.25
    assert four["assumptions"]["weights"] == "4bit" and coder["size"]["weight_bytes"] == 2_125_000_000 and coder["size"]["weights"] == "4-bit"
    assert "16-bit weights as published" in coder["size"]["quality_note"]
    assert (await client.get("/v1/reports/gpu/breakeven", params={"weights": "int3"}, headers=auth(LEAD_KEY))).status_code == 422

    # Get latest prices: SkyPilot downloads its catalogs again; without SkyPilot here, the GPU controller is asked to.
    from meritorder import gpu
    from meritorder.controller import PRICES_REQUESTED

    assert (await client.post("/v1/reports/gpu/breakeven/refresh", headers=auth(TEST_KEY))).status_code == 403
    monkeypatch.setattr(gpu, "sky_available", lambda: False)
    asked = await client.post("/v1/reports/gpu/breakeven/refresh", headers=auth(LEAD_KEY))
    assert asked.status_code == 202 and asked.json() == {"requested": True, "controller": False}
    assert (await gateway.state.get_json(PRICES_REQUESTED))["by"] == "lee" and await gateway.state.get_json("controller:kick")
    monkeypatch.setattr(gpu, "sky_available", lambda: True)

    async def missing(cfg, store, client=None, sizes=True, download=False, catalog=True):
        raise gpu_sizing.SkyPilotMissing("SkyPilot isn't installed where the gateway runs")

    monkeypatch.setattr(gpu_sizing, "refresh", missing)
    r = await client.post("/v1/reports/gpu/breakeven/refresh", headers=auth(LEAD_KEY))
    assert r.status_code == 503 and "SkyPilot isn't installed" in r.json()["detail"]

    async def fails(cfg, store, client=None, sizes=True, download=False, catalog=True):
        raise RuntimeError("SkyPilot's catalog lookup failed")

    monkeypatch.setattr(gpu_sizing, "refresh", fails)
    assert (await client.post("/v1/reports/gpu/breakeven/refresh", headers=auth(LEAD_KEY))).status_code == 502

    async def works(cfg, store, client=None, sizes=True, download=False, catalog=True):
        assert download is True
        return {"offers": 317}

    monkeypatch.setattr(gpu_sizing, "refresh", works)
    assert (await client.post("/v1/reports/gpu/breakeven/refresh", headers=auth(LEAD_KEY))).json() == {"offers": 317}