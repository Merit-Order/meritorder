"""Admin API: who may mint and revoke keys, people management, evals, and config versions."""

import copy

import pytest
import yaml
from conftest import ADMIN_KEY, LEAD_KEY, TEST_CONFIG, auth, ledger_row


@pytest.fixture
def durable(gateway):
    """Key and config management need a durable store; the in-memory one behaves the same for these tests."""
    kind = gateway.store.kind
    gateway.store.kind = "postgres"
    yield gateway
    gateway.store.kind = kind


async def mint(client, body, key=LEAD_KEY):
    return await client.post("/v1/admin/keys", json=body, headers=auth(key))


async def test_lead_cannot_mint_a_key_that_acts_as_an_admin_or_another_team(client, durable):
    """Regression: a key acts as its user's record, so a lead minting a key for an existing admin got admin access."""
    await durable.store.upsert_user("boss", "ops", "admin")
    await durable.store.upsert_user("carol", "ops", "user")
    await durable.store.upsert_user("dana", "default", "team_lead")
    for user in ("boss", "carol", "dana"):
        r = await mint(client, {"user": user})
        assert r.status_code == 403 and "only an admin" in r.text, user
    assert not await durable.store.list_keys(user_id="boss")
    # Their own team's users, new people, and themselves are fine.
    await durable.store.upsert_user("erin", "default", "user")
    for user in ("erin", "newcomer@example.com", "lee"):
        assert (await mint(client, {"user": user})).status_code == 201, user
    # An admin can mint for anyone; the key takes the record's team and acts as the lower of the two roles, and the
    # response says so.
    r = await mint(client, {"user": "boss", "role": "user"}, ADMIN_KEY)
    assert r.status_code == 201 and r.json()["role"] == "user" and r.json()["team"] == "ops" and "acts as user" in r.json()["note"]
    p = await durable.authn.authenticate({"authorization": f"Bearer {r.json()['api_key']}"})
    assert (p.user_id, p.team, p.role) == ("boss", "ops", "user")
    r = await mint(client, {"user": "boss", "role": "admin"}, ADMIN_KEY)
    p = await durable.authn.authenticate({"authorization": f"Bearer {r.json()['api_key']}"})
    assert r.json()["role"] == "admin" and p.role == "admin"


async def test_key_lifecycle_and_scoping(client, durable):
    r = await mint(client, {"user": "zed", "name": "ci", "tags": {"project": "x"}, "expires_days": 7})
    assert r.status_code == 201 and r.json()["expires_at"] and "Shown once" in r.json()["note"]
    lead_key = r.json()["id"]
    r = await mint(client, {"user": "olga", "team": "ops"}, ADMIN_KEY)
    ops_key = r.json()["id"]
    lead_view = {k["id"] for k in (await client.get("/v1/admin/keys", headers=auth(LEAD_KEY))).json()["keys"]}
    assert lead_key in lead_view and ops_key not in lead_view
    admin_view = (await client.get("/v1/admin/keys?user=olga", headers=auth(ADMIN_KEY))).json()["keys"]
    assert [k["id"] for k in admin_view] == [ops_key] and admin_view[0]["status"] == "active"
    r = await client.delete(f"/v1/admin/keys/{ops_key}", headers=auth(LEAD_KEY))
    assert r.status_code == 403 and "your own team" in r.text
    assert (await client.delete("/v1/admin/keys/key_nope", headers=auth(ADMIN_KEY))).status_code == 404
    assert (await client.delete(f"/v1/admin/keys/{ops_key}", headers=auth(ADMIN_KEY))).json()["revoked"] is True
    statuses = {k["id"]: k["status"] for k in (await client.get("/v1/admin/keys", headers=auth(ADMIN_KEY))).json()["keys"]}
    assert statuses[ops_key] == "revoked"
    _, fin = await durable.store.create_key("fin", "ops", "finance")
    assert (await client.get("/v1/admin/keys", headers=auth(fin))).status_code == 403  # finance isn't a lead
    assert (await client.get("/v1/admin/keys")).status_code == 401


async def test_expired_key_status(client, durable):
    from datetime import UTC, datetime, timedelta

    key_id, _ = await durable.store.create_key("old", "default", "user", expires_at=datetime.now(UTC) - timedelta(days=1))
    keys = {k["id"]: k for k in (await client.get("/v1/admin/keys", headers=auth(ADMIN_KEY))).json()["keys"]}
    assert keys[key_id]["status"] == "expired" and "key_hash" not in keys[key_id]


async def test_key_and_config_management_need_a_durable_store(client, gateway):
    if gateway.store.kind != "memory":
        pytest.skip("only the in-memory store refuses")
    for method, path, body, what in (
            ("post", "/v1/admin/keys", {"user": "x"}, "key management"), ("delete", "/v1/admin/keys/k", None, "key management"),
            ("patch", "/v1/admin/users/x", {"team": "default"}, "changing a person's team"),
            ("put", "/v1/admin/config", {"yaml": "x: 1"}, "saving the config"),
            ("post", "/v1/admin/config/versions/1/restore", None, "restoring a config version"),
            ("post", "/v1/admin/prices/refresh", None, "refreshing prices"), ("post", "/v1/admin/quality/refresh", None, "refreshing quality")):
        kwargs = {"json": body} if body is not None else {}
        r = await client.request(method, path, headers=auth(ADMIN_KEY), **kwargs)
        assert r.status_code == 409 and r.json()["detail"].startswith(f"{what}") and "needs Postgres" in r.text, path


async def test_people(client, durable):
    await durable.store.upsert_user("alice", "default", "user", email="alice@example.com")
    await durable.store.upsert_user("carol", "ops", "user")
    await durable.store.record(ledger_row(user_id="alice", actual_cost=2.5))
    await durable.store.rollup()
    lead = (await client.get("/v1/admin/users", headers=auth(LEAD_KEY))).json()
    assert [u["id"] for u in lead["users"]] == ["alice"] and lead["users"][0]["spend_month"] == 2.5
    admin = (await client.get("/v1/admin/users", headers=auth(ADMIN_KEY))).json()
    assert {u["id"] for u in admin["users"]} == {"alice", "carol"} and admin["teams"] == ["default"]
    assert admin["oidc"] is False and admin["scim"] is False and admin["idp"] is None
    from meritorder.config import OIDCConfig

    for issuer, idp in [("https://accounts.google.com", "google"), ("https://login.microsoftonline.com/t/v2.0", "entra"),
                        ("https://idp.example.com", "oidc")]:
        durable.cfg.auth.oidc = OIDCConfig(issuer=issuer, client_id="c", hosted_domains=["example.com"])
        assert (await client.get("/v1/admin/users", headers=auth(ADMIN_KEY))).json()["idp"] == idp  # the People page's wording
    durable.cfg.auth.oidc = None

    patch = lambda uid, body, key=ADMIN_KEY: client.patch(f"/v1/admin/users/{uid}", json=body, headers=auth(key))  # noqa: E731
    assert (await patch("alice", {"team": "default"}, LEAD_KEY)).status_code == 403
    r = await patch("root", {"disabled": True})
    assert r.status_code == 400 and "yourself" in r.text
    assert (await patch("root", {"role": "user"})).status_code == 400
    r = await patch("alice", {"team": "nowhere"})
    assert r.status_code == 400 and "isn't defined" in r.text
    assert (await patch("ghost", {"role": "user"})).status_code == 404
    r = (await patch("alice", {"role": "team_lead"})).json()
    assert r["role"] == "team_lead" and r["managed"] == "local"
    r = (await patch("alice", {"follow_idp": True})).json()
    assert r["managed"] == "idp"
    r = (await patch("alice", {"role": "user", "follow_idp": True})).json()  # both in one save (Postgres used to fail)
    assert r["role"] == "user" and r["managed"] == "idp"
    r = (await patch("alice", {"disabled": True})).json()
    assert r["disabled"] is True


async def test_evals_storage(client, gateway):
    assert (await client.get("/v1/admin/evals", headers=auth(LEAD_KEY))).json() == {"models": {}}
    run = {"suite": "starter", "ran_at": "2026-11-10T00:00:00Z", "models": {"small": {"score": 0.7}, "big": {"score": 0.9}}}
    assert (await client.put("/v1/admin/evals", json=run, headers=auth(LEAD_KEY))).status_code == 403
    await client.put("/v1/admin/evals", json=run, headers=auth(ADMIN_KEY))
    later = {"suite": "starter", "ran_at": "2026-11-11T00:00:00Z", "models": {"small": {"score": 0.75}}}
    out = (await client.put("/v1/admin/evals", json=later, headers=auth(ADMIN_KEY))).json()
    assert out["models"]["small"] == {"score": 0.75, "suite": "starter", "ran_at": "2026-11-11T00:00:00Z"}
    assert out["models"]["big"]["score"] == 0.9  # the latest result per model is kept


async def test_config_overview(client, gateway):
    c = (await client.get("/v1/admin/config", headers=auth(ADMIN_KEY))).json()
    assert c["baseline_model"] == "big" and {b["name"] for b in c["backends"]} >= {"mock-anthropic", "mock-openai"}
    big = next(m for m in c["models"] if m["name"] == "big")
    assert big["routes"][0]["live"] is True and big["routes"][0]["price"]["input"] == 3
    assert (await client.get("/v1/admin/config", headers=auth(LEAD_KEY))).status_code == 403


async def test_config_source_without_a_file_or_version(client, gateway):
    gateway.file_raw = None
    r = await client.get("/v1/admin/config/source", headers=auth(ADMIN_KEY))
    assert r.status_code == 409


async def test_config_save_validate_and_restore_errors(client, durable, monkeypatch):
    monkeypatch.setenv("T_USER", "mo-test-key")
    monkeypatch.setenv("T_ADMIN", "mo-admin-key")
    doc = copy.deepcopy(TEST_CONFIG)
    doc["auth"]["static_keys"] = [{"key_env": "T_USER", "user": "alice", "team": "default"},
                                  {"key_env": "T_ADMIN", "user": "root", "team": "ops", "role": "admin"}]
    v = await client.post("/v1/admin/config/validate", json={"document": {"models": "not a mapping"}}, headers=auth(ADMIN_KEY))
    assert v.json()["ok"] is False
    r = await client.put("/v1/admin/config", json={"yaml": "models: [unclosed"}, headers=auth(ADMIN_KEY))
    assert r.status_code == 422
    r = await client.put("/v1/admin/config", json={"document": doc, "comment": "first"}, headers=auth(ADMIN_KEY))
    assert r.status_code == 200 and r.json()["version"] == 1
    src = (await client.get("/v1/admin/config/source", headers=auth(ADMIN_KEY))).json()
    assert src["source"] == "database" and src["version"] == 1 and yaml.safe_load(src["yaml"])["baseline_model"] == "big"
    one = (await client.get("/v1/admin/config/versions/1", headers=auth(ADMIN_KEY))).json()
    assert one["author"] == "root" and one["comment"] == "first"
    assert (await client.get("/v1/admin/config/versions/99", headers=auth(ADMIN_KEY))).status_code == 404
    assert (await client.post("/v1/admin/config/versions/99/restore", headers=auth(ADMIN_KEY))).status_code == 404
    # A stored version that no longer validates (e.g. it names a key variable that's gone) can't be restored.
    bad = copy.deepcopy(doc)
    bad["baseline_model"] = "a-model-that-was-removed"
    vid = await durable.store.save_config(bad, "root", "broken")
    r = await client.post(f"/v1/admin/config/versions/{vid}/restore", headers=auth(ADMIN_KEY))
    assert r.status_code == 422 and "no longer validates" in r.text


async def test_prices_refresh(client, durable, monkeypatch):
    import meritorder.catalog as catalog

    def fake_refresh(cfg, path, write):
        return None, 3, ["small"], {"updated": "2026-11-10", "prices": {"mock-openai:small-1": {"input": 1, "output": 2, "source": "openrouter"}},
                                    "sources": {"openrouter": {"ok": True}}, "disagreements": []}

    monkeypatch.setattr(catalog, "refresh_prices", fake_refresh)
    r = await client.post("/v1/admin/prices/refresh", headers=auth(ADMIN_KEY))
    body = r.json()
    assert r.status_code == 200 and body["priced_routes"] == 3 and body["missing"] == ["small"] and body["by"] == "root"
    assert body["sources"] == {"openrouter": {"ok": True}} and body["changed"] == [] and body["disagreements"] == []
    assert (await durable.store.get_setting("prices"))["updated"] == "2026-11-10"
    assert (await durable.store.get_setting("prices_status"))["priced_routes"] == 3
    shown = (await client.get("/v1/admin/config", headers=auth(ADMIN_KEY))).json()["prices"]
    assert shown["priced_routes"] == 3 and shown["refresh_minutes"] == 15
    assert (await client.post("/v1/admin/prices/refresh", headers=auth(LEAD_KEY))).status_code == 403
