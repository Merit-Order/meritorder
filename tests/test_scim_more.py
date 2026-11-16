"""SCIM 2.0: the requests Microsoft Entra actually sends, plus errors and edge cases."""

import pytest
from conftest import auth

from meritorder.config import OIDCConfig

TOKEN = "scim-secret"
SCIM = {"authorization": f"Bearer {TOKEN}", "content-type": "application/scim+json"}


@pytest.fixture
def scim(gateway, monkeypatch):
    monkeypatch.setenv("MERITORDER_SCIM_TOKEN", TOKEN)
    for name in ("research", "ops"):
        if name not in {t.name for t in gateway.cfg.teams}:
            gateway.cfg.teams.append(gateway.cfg.team("default").model_copy(update={"name": name}))
    gateway.cfg.auth.oidc = OIDCConfig(issuer="https://login.example.com", client_id="x", default_team="default",
                                       team_from_groups={"grp-research": "research"},
                                       role_from_groups={"grp-leads": "team_lead", "grp-admins": "admin"})
    return gateway


async def make_user(client, uid, email=None, **extra):
    body = {"externalId": uid, "userName": email or f"{uid}@example.com", "active": True, **extra}
    r = await client.post("/scim/v2/Users", headers=SCIM, json=body)
    assert r.status_code == 201, r.text
    return r.json()


async def make_group(client, name, external_id=None, members=()):
    r = await client.post("/scim/v2/Groups", headers=SCIM,
                          json={"displayName": name, "externalId": external_id, "members": [{"value": m} for m in members]})
    assert r.status_code == 201, r.text
    return r.json()


async def test_auth_failures_on_every_endpoint(client, scim):
    paths = [("get", "/scim/v2/Users"), ("post", "/scim/v2/Users"), ("get", "/scim/v2/Users/x"), ("put", "/scim/v2/Users/x"),
             ("patch", "/scim/v2/Users/x"), ("delete", "/scim/v2/Users/x"), ("get", "/scim/v2/Groups"),
             ("post", "/scim/v2/Groups"), ("get", "/scim/v2/Groups/x"), ("patch", "/scim/v2/Groups/x"),
             ("put", "/scim/v2/Groups/x"), ("delete", "/scim/v2/Groups/x"), ("get", "/scim/v2/ResourceTypes"),
             ("get", "/scim/v2/ServiceProviderConfig")]
    for method, path in paths:
        for headers in ({}, {"authorization": "Bearer wrong"}, {"authorization": TOKEN}):  # missing, wrong, no "Bearer "
            kwargs = {"json": {}} if method in ("post", "put", "patch") else {}
            r = await client.request(method, path, headers=headers, **kwargs)
            assert r.status_code == 401, (method, path, headers)
            assert r.json()["schemas"] == ["urn:ietf:params:scim:api:messages:2.0:Error"]
    # A gateway key (even an admin's) isn't a SCIM token.
    from conftest import ADMIN_KEY

    assert (await client.get("/scim/v2/Users", headers=auth(ADMIN_KEY))).status_code == 401


async def test_discovery(client, scim):
    rt = (await client.get("/scim/v2/ResourceTypes", headers=SCIM)).json()
    assert {r["id"] for r in rt["Resources"]} == {"User", "Group"}
    spc = (await client.get("/scim/v2/ServiceProviderConfig", headers=SCIM))
    assert spc.headers["content-type"].startswith("application/scim+json")
    assert spc.json()["bulk"]["supported"] is False and spc.json()["filter"]["supported"] is True


async def test_user_crud_and_filters(client, scim):
    u = await make_user(client, "oid-1", "Pat@Example.com", displayName="Pat",
                        emails=[{"value": "other@example.com"}, {"value": "pat@example.com", "primary": True}])
    assert u["userName"] == "pat@example.com" and u["displayName"] == "Pat" and u["active"] is True  # primary email wins
    assert u["meta"]["location"].endswith("/scim/v2/Users/oid-1")
    await make_user(client, "oid-2")

    lst = lambda q: client.get(f"/scim/v2/Users?filter={q}", headers=SCIM)  # noqa: E731
    for q in ('userName eq "PAT@example.com"', 'emails.value eq "pat@example.com"', 'externalId eq "oid-1"', 'id eq "oid-1"'):
        r = (await lst(q)).json()
        assert r["totalResults"] == 1 and r["Resources"][0]["id"] == "oid-1", q
    assert (await lst('userName eq "oid-2"')).json()["Resources"][0]["id"] == "oid-2"  # falls back to the id
    assert (await lst('externalId eq "nobody"')).json()["totalResults"] == 0
    for bad in ('displayName eq "Pat"', 'userName co "pat"', "garbage"):
        r = await lst(bad)
        assert r.status_code == 400 and r.json()["scimType"] == "invalidFilter", bad

    page = (await client.get("/scim/v2/Users?startIndex=2&count=1", headers=SCIM)).json()
    assert page["totalResults"] == 2 and page["itemsPerPage"] == 1 and page["startIndex"] == 2
    assert (await client.get("/scim/v2/Users?startIndex=0", headers=SCIM)).json()["startIndex"] == 1

    assert (await client.get("/scim/v2/Users/oid-1", headers=SCIM)).json()["displayName"] == "Pat"
    assert (await client.get("/scim/v2/Users/nobody", headers=SCIM)).status_code == 404

    r = await client.put("/scim/v2/Users/oid-1", headers=SCIM, json={"userName": "pat.new@example.com", "displayName": "Pat N",
                                                                     "active": True})
    assert r.json()["userName"] == "pat.new@example.com" and r.json()["displayName"] == "Pat N"
    assert (await client.put("/scim/v2/Users/nobody", headers=SCIM, json={})).status_code == 404

    r = await client.post("/scim/v2/Users", headers=SCIM, json={"displayName": "no id"})
    assert r.status_code == 400 and r.json()["scimType"] == "invalidValue"
    assert (await client.delete("/scim/v2/Users/nobody", headers=SCIM)).status_code == 404


async def test_patch_user_paths_entra_sends(client, scim):
    await make_user(client, "oid-1")
    ops = [{"op": "Replace", "path": "displayName", "value": "Renamed"},
           {"op": "Replace", "path": 'emails[type eq "work"].value', "value": "renamed@example.com"},
           {"op": "Add", "path": "name.givenName", "value": "ignored"},
           {"op": "Remove", "path": "displayName"}]  # removes are ignored: Entra replaces instead
    r = (await client.patch("/scim/v2/Users/oid-1", headers=SCIM, json={"Operations": ops})).json()
    assert r["displayName"] == "Renamed" and r["userName"] == "renamed@example.com"
    r = (await client.patch("/scim/v2/Users/oid-1", headers=SCIM,
                            json={"Operations": [{"op": "replace", "path": "userName", "value": "u@example.com"}]})).json()
    assert r["userName"] == "u@example.com"
    r = (await client.patch("/scim/v2/Users/oid-1", headers=SCIM,
                            json={"Operations": [{"op": "replace", "value": {"active": "false", "displayName": "Off"}}]})).json()
    assert r["active"] is False and r["displayName"] == "Off"
    assert (await client.patch("/scim/v2/Users/nobody", headers=SCIM, json={"Operations": []})).status_code == 404


async def test_deactivation_stops_keys_and_reactivation_restores(client, scim):
    await make_user(client, "oid-1")
    _, key = await scim.store.create_key("oid-1", "default", "user")
    assert (await client.get("/v1/reports/me", headers=auth(key))).status_code == 200  # also caches the key
    r = await client.patch("/scim/v2/Users/oid-1", headers=SCIM, json={"Operations": [{"op": "Replace", "path": "active", "value": False}]})
    assert r.json()["active"] is False
    assert (await client.get("/v1/reports/me", headers=auth(key))).status_code == 401  # the cache was dropped
    await client.put("/scim/v2/Users/oid-1", headers=SCIM, json={"userName": "oid-1@example.com", "active": True})
    assert (await client.get("/v1/reports/me", headers=auth(key))).status_code == 200
    # Spend history is kept: deleting disables rather than removes.
    assert (await client.delete("/scim/v2/Users/oid-1", headers=SCIM)).status_code == 204
    assert (await scim.store.get_user("oid-1"))["disabled_at"] is not None
    assert (await client.get("/scim/v2/Users/oid-1", headers=SCIM)).json()["active"] is False


async def test_group_membership_drives_team_and_role(client, scim):
    await make_user(client, "oid-1")
    await make_user(client, "oid-2")
    research = await make_group(client, "Research", "grp-research", ["oid-1"])
    assert (await scim.store.get_user("oid-1"))["team"] == "research"
    # Mapping can name a group by its display name or id, too.
    scim.cfg.auth.oidc.role_from_groups["Leads"] = "team_lead"
    leads = await make_group(client, "Leads", None, ["oid-1", "oid-2"])
    assert (await scim.store.get_user("oid-2"))["role"] == "team_lead"
    admins = await make_group(client, "Admins", "grp-admins", ["oid-1"])
    assert (await scim.store.get_user("oid-1"))["role"] == "admin"  # the highest role wins

    gid = research["id"]
    patch = lambda ops: client.patch(f"/scim/v2/Groups/{gid}", headers=SCIM, json={"Operations": ops})  # noqa: E731
    r = await patch([{"op": "Add", "path": "members", "value": [{"value": "oid-2"}, {"value": "oid-1"}]}])
    assert sorted(m["value"] for m in r.json()["members"]) == ["oid-1", "oid-2"]  # no duplicates
    assert (await scim.store.get_user("oid-2"))["team"] == "research"
    r = await patch([{"op": "Remove", "path": "members", "value": [{"value": "oid-1"}]}])
    assert [m["value"] for m in r.json()["members"]] == ["oid-2"]
    assert (await scim.store.get_user("oid-1"))["team"] == "default"
    r = await patch([{"op": "Replace", "path": "members", "value": [{"value": "oid-1"}]}])
    assert [m["value"] for m in r.json()["members"]] == ["oid-1"]
    assert (await scim.store.get_user("oid-2"))["team"] == "default" and (await scim.store.get_user("oid-1"))["team"] == "research"
    r = await patch([{"op": "Remove", "path": "members"}])  # no value: remove everyone
    assert r.json()["members"] == [] and (await scim.store.get_user("oid-1"))["team"] == "default"
    r = await patch([{"op": "Replace", "path": "displayName", "value": "Research 2"},
                     {"op": "Replace", "path": "externalId", "value": "grp-x"}])
    assert r.json()["displayName"] == "Research 2" and r.json()["externalId"] == "grp-x"
    r = await patch([{"op": "Replace", "value": {"displayName": "Research 3", "externalId": "grp-research"}}])
    assert r.json()["displayName"] == "Research 3" and r.json()["externalId"] == "grp-research"

    # PUT replaces membership and recomputes who changed.
    r = await client.put(f"/scim/v2/Groups/{gid}", headers=SCIM, json={"displayName": "Research", "members": [{"value": "oid-2"}]})
    assert [m["value"] for m in r.json()["members"]] == ["oid-2"] and (await scim.store.get_user("oid-2"))["team"] == "research"

    # Deleting a group drops its roles.
    assert (await client.delete(f"/scim/v2/Groups/{admins['id']}", headers=SCIM)).status_code == 204
    assert (await scim.store.get_user("oid-1"))["role"] == "team_lead"
    assert (await client.delete(f"/scim/v2/Groups/{leads['id']}", headers=SCIM)).status_code == 204
    assert (await scim.store.get_user("oid-1"))["role"] == "user"


async def test_admin_pin_wins_over_groups(client, scim):
    await make_user(client, "oid-1")
    await scim.store.update_user("oid-1", team="ops")  # an admin moved them: pinned
    await make_group(client, "Research", "grp-research", ["oid-1"])
    assert (await scim.store.get_user("oid-1"))["team"] == "ops"


async def test_members_provisioned_before_the_user_exists(client, scim):
    g = await make_group(client, "Research", "grp-research", ["oid-later"])  # Entra may push groups first
    assert await scim.store.get_user("oid-later") is None
    await make_user(client, "oid-later")
    assert (await scim.store.get_user("oid-later"))["team"] == "research"
    assert (await client.get(f"/scim/v2/Groups/{g['id']}", headers=SCIM)).json()["members"] == [{"value": "oid-later"}]


async def test_group_errors_and_filters(client, scim):
    g = await make_group(client, "Research", "grp-research")
    r = await client.post("/scim/v2/Groups", headers=SCIM, json={"members": []})
    assert r.status_code == 400 and r.json()["scimType"] == "invalidValue"
    for verb in ("get", "patch", "put", "delete"):
        kwargs = {"json": {"Operations": []}} if verb in ("patch", "put") else {}
        assert (await client.request(verb, "/scim/v2/Groups/nope", headers=SCIM, **kwargs)).status_code == 404, verb
    for q, n in (('externalId eq "grp-research"', 1), (f'id eq "{g["id"]}"', 1), ('displayName eq "Nope"', 0)):
        assert (await client.get(f"/scim/v2/Groups?filter={q}", headers=SCIM)).json()["totalResults"] == n, q
    for bad in ('members eq "x"', "nonsense"):
        r = await client.get(f"/scim/v2/Groups?filter={bad}", headers=SCIM)
        assert r.status_code == 400 and r.json()["scimType"] == "invalidFilter"
    full = (await client.get("/scim/v2/Groups", headers=SCIM)).json()["Resources"][0]
    assert "members" in full and full["meta"]["location"].endswith(f"/Groups/{g['id']}")


async def test_paging_parameters(client, scim):
    for i in range(3):
        await make_group(client, f"G{i}")
    r = await client.get("/scim/v2/Users?startIndex=abc", headers=SCIM)
    assert r.status_code == 400 and r.json()["scimType"] == "invalidValue"
    assert (await client.get("/scim/v2/Groups?count=x", headers=SCIM)).status_code == 400
    # A negative count means zero results (not "all but the last N").
    r = (await client.get("/scim/v2/Groups?count=-2", headers=SCIM)).json()
    assert r["totalResults"] == 3 and r["itemsPerPage"] == 0 and r["Resources"] == []
    r = (await client.get("/scim/v2/Groups?startIndex=3&count=10", headers=SCIM)).json()
    assert r["itemsPerPage"] == 1
