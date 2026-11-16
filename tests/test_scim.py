from conftest import auth

from meritorder.config import OIDCConfig

SCIM = {"authorization": "Bearer scim-secret", "content-type": "application/scim+json"}


async def test_scim_disabled_without_token(client):
    r = await client.get("/scim/v2/Users", headers=SCIM)
    assert r.status_code == 404


async def test_scim_lifecycle(client, gateway, monkeypatch):
    monkeypatch.setenv("MERITORDER_SCIM_TOKEN", "scim-secret")
    research = gateway.cfg.team("default").model_copy(update={"name": "research"})
    gateway.cfg.teams.append(research)
    gateway.cfg.auth.oidc = OIDCConfig(issuer="https://login.example.com", client_id="x",
                                       team_from_groups={"grp-research": "research"}, role_from_groups={"grp-leads": "team_lead"})
    assert (await client.get("/scim/v2/Users", headers={"authorization": "Bearer wrong"})).status_code == 401

    # Entra checks whether the user exists, then creates them
    r = await client.get('/scim/v2/Users?filter=userName eq "pat@example.com"', headers=SCIM)
    assert r.json()["totalResults"] == 0
    r = await client.post("/scim/v2/Users", headers=SCIM, json={
        "schemas": ["urn:ietf:params:scim:schemas:core:2.0:User"], "externalId": "oid-pat", "userName": "pat@example.com",
        "displayName": "Pat Lee", "active": True, "emails": [{"value": "pat@example.com", "type": "work", "primary": True}]})
    assert r.status_code == 201 and r.json()["id"] == "oid-pat"
    assert (await client.post("/scim/v2/Users", headers=SCIM, json={"externalId": "oid-pat", "userName": "pat@example.com"})).status_code == 409
    assert (await gateway.store.get_user("oid-pat"))["team"] == "default"

    # groups: membership drives team and role
    g = (await client.post("/scim/v2/Groups", headers=SCIM, json={"displayName": "Research", "externalId": "grp-research", "members": []})).json()
    await client.patch(f"/scim/v2/Groups/{g['id']}", headers=SCIM, json={"Operations": [{"op": "Add", "path": "members", "value": [{"value": "oid-pat"}]}]})
    assert (await gateway.store.get_user("oid-pat"))["team"] == "research"
    leads = (await client.post("/scim/v2/Groups", headers=SCIM, json={"displayName": "Leads", "externalId": "grp-leads",
                                                                         "members": [{"value": "oid-pat"}]})).json()
    assert (await gateway.store.get_user("oid-pat"))["role"] == "team_lead"
    r = await client.get('/scim/v2/Groups?filter=displayName eq "Leads"&excludedAttributes=members', headers=SCIM)
    assert r.json()["Resources"][0]["id"] == leads["id"] and "members" not in r.json()["Resources"][0]
    await client.patch(f"/scim/v2/Groups/{g['id']}", headers=SCIM, json={"Operations": [{"op": "Remove", "path": 'members[value eq "oid-pat"]'}]})
    assert (await gateway.store.get_user("oid-pat"))["team"] == "default"
    await client.delete(f"/scim/v2/Groups/{leads['id']}", headers=SCIM)
    assert (await gateway.store.get_user("oid-pat"))["role"] == "user"

    # their key works until Entra deactivates them, then stops at once
    key = (await gateway.store.create_key("oid-pat", "default", "user"))[1]
    assert (await client.get("/v1/reports/me", headers=auth(key))).status_code == 200
    r = await client.patch("/scim/v2/Users/oid-pat", headers=SCIM, json={"Operations": [{"op": "Replace", "path": "active", "value": "False"}]})
    assert r.json()["active"] is False
    assert (await client.get("/v1/reports/me", headers=auth(key))).status_code == 401
    await client.patch("/scim/v2/Users/oid-pat", headers=SCIM, json={"Operations": [{"op": "replace", "value": {"active": True}}]})
    assert (await client.get("/v1/reports/me", headers=auth(key))).status_code == 200
    assert (await client.delete("/scim/v2/Users/oid-pat", headers=SCIM)).status_code == 204
    assert (await client.get("/v1/reports/me", headers=auth(key))).status_code == 401
    assert (await client.get("/scim/v2/ServiceProviderConfig", headers=SCIM)).json()["patch"]["supported"] is True
