import copy

import yaml
from conftest import ADMIN_KEY, LEAD_KEY, TEST_CONFIG, auth


async def _traffic(client, gateway, n=3):
    for i in range(n):
        await client.post("/v1/chat/completions", headers=auth(), json={"model": "small", "messages": [{"role": "user", "content": f"hi {i}"}]})
        await client.post("/v1/messages", headers=auth(ADMIN_KEY), json={"model": "big", "max_tokens": 20, "messages": [{"role": "user", "content": "hello"}]})
    await gateway.drain()


async def test_console_requires_lead(client):
    assert (await client.get("/v1/console/overview", headers=auth())).status_code == 403


async def test_console_endpoints(client, gateway):
    await _traffic(client, gateway)
    ov = (await client.get("/v1/console/overview", headers=auth(ADMIN_KEY))).json()
    assert ov["scope"] == "organization" and ov["today"]["requests"] == 6 and ov["requests_per_minute"] > 0
    merit = (await client.get("/v1/console/merit", headers=auth(ADMIN_KEY))).json()
    prices = [t["price_per_m"] for t in merit["tiers"]]
    assert prices == sorted(prices) and [t["model"] for t in merit["tiers"]] == ["small", "big"]
    assert merit["clearing_price_per_m"] == prices[-1]
    feed = (await client.get("/v1/console/feed", headers=auth(ADMIN_KEY))).json()["events"]
    assert len(feed) == 6 and {"user", "model", "backend", "cost", "saved", "latency_ms"} <= set(feed[0])
    assert all(isinstance(e["latency_ms"], int) and isinstance(e["saved"], float | int) for e in feed)  # the feed's Saved and Latency columns
    # team leads only see their own team
    lead_feed = (await client.get("/v1/console/feed", headers=auth(LEAD_KEY))).json()["events"]
    assert lead_feed and all(e["team"] == "default" for e in lead_feed)
    backends = {b["name"]: b for b in (await client.get("/v1/console/backends", headers=auth(ADMIN_KEY))).json()["backends"]}
    assert backends["mock-openai"]["requests"] == 3 and backends["mock-openai"]["error_rate"] == 0
    att = (await client.get("/v1/console/attention", headers=auth(ADMIN_KEY))).json()
    assert isinstance(att["items"], list)


async def test_config_versions(client, gateway, monkeypatch):
    gateway.store.kind = "postgres"  # config versions need a durable store; MemoryStore behaves the same
    try:
        src = (await client.get("/v1/admin/config/source", headers=auth(ADMIN_KEY))).json()
        assert src["source"] == "file" and "backends" in yaml.safe_load(src["yaml"])
        assert (await client.get("/v1/admin/config/source", headers=auth(LEAD_KEY))).status_code == 403

        bad = (await client.post("/v1/admin/config/validate", headers=auth(ADMIN_KEY), json={"yaml": "backends: [unclosed"})).json()
        assert bad["ok"] is False and "YAML" in bad["error"]
        literal = (await client.post("/v1/admin/config/validate", headers=auth(ADMIN_KEY), json={"yaml": src["yaml"]})).json()
        assert literal["ok"] is False and "literal key" in literal["error"]  # the test config has literal keys

        monkeypatch.setenv("T_USER", "mo-test-key")
        monkeypatch.setenv("T_ADMIN", "mo-admin-key")
        doc = copy.deepcopy(TEST_CONFIG)
        doc["auth"]["static_keys"] = [{"key_env": "T_USER", "user": "alice", "team": "default"},
                                      {"key_env": "T_ADMIN", "user": "root", "team": "ops", "role": "admin"}]
        doc["teams"]["default"]["budget_usd_month"] = 1234
        ok = (await client.post("/v1/admin/config/validate", headers=auth(ADMIN_KEY), json={"yaml": yaml.safe_dump(doc)})).json()
        assert ok["ok"] and ok["summary"]["models"] == 2 and ok["summary"]["embedding_models"] == 1
        saved = (await client.put("/v1/admin/config", headers=auth(ADMIN_KEY), json={"yaml": yaml.safe_dump(doc), "comment": "raise budget"})).json()
        assert saved["version"] == 1 and gateway.cfg.team("default").budget_usd_month == 1234
        # the new config is live: requests still work with the env-referenced keys
        r = await client.post("/v1/chat/completions", headers=auth(), json={"model": "small", "messages": [{"role": "user", "content": "x"}]})
        assert r.status_code == 200
        doc["teams"]["default"]["budget_usd_month"] = 99
        await client.put("/v1/admin/config", headers=auth(ADMIN_KEY), json={"yaml": yaml.safe_dump(doc)})
        assert gateway.cfg.team("default").budget_usd_month == 99
        versions = (await client.get("/v1/admin/config/versions", headers=auth(ADMIN_KEY))).json()
        assert [v["id"] for v in versions["versions"]] == [2, 1] and versions["active"] == 2
        restored = (await client.post("/v1/admin/config/versions/1/restore", headers=auth(ADMIN_KEY))).json()
        assert restored["version"] == 3 and gateway.cfg.team("default").budget_usd_month == 1234
        assert await gateway.state.get_config_token() is not None  # other gateways will pick it up
    finally:
        gateway.store.kind = "memory"


async def test_config_document_roundtrip(client, gateway, monkeypatch):
    monkeypatch.setenv("T_USER", "mo-test-key")
    monkeypatch.setenv("T_ADMIN", "mo-admin-key")
    src = (await client.get("/v1/admin/config/source", headers=auth(ADMIN_KEY))).json()
    doc = src["document"]
    doc["auth"]["static_keys"] = [{"key_env": "T_USER", "user": "alice", "team": "default"},
                                  {"key_env": "T_ADMIN", "user": "root", "team": "ops", "role": "admin"}]
    # blank form fields come back as None or "": they mean "use the default"
    doc["teams"]["research"] = {"budget_usd_month": 250, "user_budget_usd_month": None, "baseline_model": "", "rate_limits": {"user_rpm": None}}
    r = (await client.post("/v1/admin/config/validate", headers=auth(ADMIN_KEY), json={"document": doc})).json()
    assert r["ok"], r.get("error")
    assert "research" in r["summary"]["teams"]
    assert r["document"]["teams"]["research"] == {"budget_usd_month": 250}
    assert "research:" in r["yaml"]
    bad = (await client.post("/v1/admin/config/validate", headers=auth(ADMIN_KEY),
                             json={"document": {**doc, "models": {"x": {"routes": [{"backend": "nope", "model": "m"}]}}}})).json()
    assert not bad["ok"] and "nope" in bad["error"] and "yaml" in bad  # rendered even when invalid


async def test_people(client, gateway):
    from meritorder.store import MemoryStore

    gateway.cfg.teams.append(gateway.cfg.team("default").model_copy(update={"name": "research"}))
    gateway.store.kind = "postgres"
    try:
        created = (await client.post("/v1/admin/keys", headers=auth(ADMIN_KEY), json={"user": "dana@example.com", "team": "default"})).json()
        key = created["api_key"]
        me = (await client.get("/v1/reports/me", headers=auth(key))).json()
        assert me["team"] == "default"
        users = (await client.get("/v1/admin/users", headers=auth(ADMIN_KEY))).json()
        dana = next(u for u in users["users"] if u["id"] == "dana@example.com")
        assert dana["active_keys"] == 1 and dana["managed"] == "idp" and "research" in users["teams"]

        # move her: her existing key follows the team at once, but a record never raises a key's role (it was issued as
        # user); a new key acts as team lead. The change sticks across Entra logins.
        r = await client.patch("/v1/admin/users/dana@example.com", headers=auth(ADMIN_KEY), json={"team": "research", "role": "team_lead"})
        assert r.json()["managed"] == "local"
        me = (await client.get("/v1/reports/me", headers=auth(key))).json()
        assert me["team"] == "research" and me["role"] == "user"
        lead = (await client.post("/v1/admin/keys", headers=auth(ADMIN_KEY), json={"user": "dana@example.com", "role": "team_lead"})).json()
        assert (await client.get("/v1/reports/me", headers=auth(lead["api_key"]))).json()["role"] == "team_lead"
        await gateway.store.upsert_user("dana@example.com", "default", "user", source="idp")  # a later Entra login
        assert (await gateway.store.get_user("dana@example.com"))["team"] == "research"
        await client.patch("/v1/admin/users/dana@example.com", headers=auth(ADMIN_KEY), json={"follow_idp": True})
        await gateway.store.upsert_user("dana@example.com", "default", "user", source="idp")
        assert (await gateway.store.get_user("dana@example.com"))["team"] == "default"

        # disable: every key stops working
        await client.patch("/v1/admin/users/dana@example.com", headers=auth(ADMIN_KEY), json={"disabled": True})
        r = await client.get("/v1/reports/me", headers=auth(key))
        assert r.status_code == 401 and "disabled" in r.json()["detail"]

        # guard rails
        assert (await client.patch("/v1/admin/users/dana@example.com", headers=auth(ADMIN_KEY), json={"team": "nope"})).status_code == 400
        assert (await client.patch("/v1/admin/users/dana@example.com", headers=auth(LEAD_KEY), json={"disabled": False})).status_code == 403
        lead_view = (await client.get("/v1/admin/users", headers=auth(LEAD_KEY))).json()["users"]
        assert all(u["team"] == "default" for u in lead_view)
    finally:
        gateway.store.kind = MemoryStore.kind
