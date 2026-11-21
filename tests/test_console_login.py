"""Console sign-in (decision 69): local accounts, sessions in an HttpOnly cookie, invite and reset links, keys traded
for sessions, and the cross-site checks on requests that carry the cookie."""

import pytest
from conftest import ADMIN_KEY, LEAD_KEY, TEST_KEY, auth

from meritorder import accounts
from meritorder.auth import CSRF_HEADER, SESSION_COOKIE

CONSOLE = {CSRF_HEADER: "1"}
PASSWORD = "correct horse battery staple"


@pytest.fixture(autouse=True)
def fast_hashing(monkeypatch):
    monkeypatch.setattr(accounts, "SCRYPT_N", 2**10)  # the real cost is a quarter second a hash


@pytest.fixture
def durable(gateway):
    kind = gateway.store.kind
    gateway.store.kind = "postgres"
    yield gateway
    gateway.store.kind = kind


async def invite(client, email, key=ADMIN_KEY, **kw):
    r = await client.post("/v1/admin/invites", json={"email": email, **kw}, headers=auth(key))
    assert r.status_code == 201, r.text
    return r.json()


async def accept(client, token, password=PASSWORD, **kw):
    return await client.post("/auth/invite/accept", json={"token": token, "password": password, **kw}, headers=CONSOLE)


async def test_invite_set_password_sign_in_and_out(client, durable):
    inv = await invite(client, "Pat@Example.com", team="default", role="team_lead")
    assert inv["path"] == f"/ui/#invite={inv['token']}" and inv["kind"] == "invite" and inv["role"] == "team_lead"
    info = (await client.post("/auth/invite/info", json={"token": inv["token"]}, headers=CONSOLE)).json()
    assert info["email"] == "pat@example.com" and info["kind"] == "invite"
    assert (await accept(client, inv["token"], "short")).status_code == 422
    r = await accept(client, inv["token"], name="Pat")
    assert r.status_code == 200 and r.json()["role"] == "team_lead"
    cookie = r.headers["set-cookie"]
    assert "httponly" in cookie.lower() and "samesite=strict" in cookie.lower() and "secure" not in cookie.lower()  # plain http here
    # The cookie alone signs the console in.
    assert (await client.get("/auth/session")).json() == {"signed_in": True, "user": "pat@example.com", "team": "default",
                                                          "role": "team_lead", "method": "password"}
    me = await client.get("/v1/reports/me")
    assert me.status_code == 200 and me.json()["user"] == "pat@example.com" and me.json()["sign_in"]["method"] == "password"
    assert (me.json()["name"], me.json()["email"]) == ("Pat", "pat@example.com")  # what the console shows
    # A link works once.
    assert (await accept(client, inv["token"])).status_code == 410
    # Sign out ends the session on the server, not only in this browser.
    token = client.cookies.get(SESSION_COOKIE)
    assert (await client.post("/auth/logout", headers=CONSOLE)).status_code == 200
    client.cookies.set(SESSION_COOKIE, token)
    assert (await client.get("/v1/reports/me")).status_code == 401
    assert (await client.get("/auth/session")).json() == {"signed_in": False, "reason": "your console session has ended; sign in again"}
    client.cookies.clear()
    assert (await client.get("/auth/session")).json() == {"signed_in": False, "reason": None}
    # And the password signs in again.
    bad = await client.post("/auth/login", json={"email": "pat@example.com", "password": "nope nope nope"}, headers=CONSOLE)
    assert bad.status_code == 401 and "wrong email address or password" in bad.text
    unknown = await client.post("/auth/login", json={"email": "nobody@example.com", "password": PASSWORD}, headers=CONSOLE)
    assert unknown.status_code == 401 and unknown.text == bad.text  # an unknown address looks the same
    r = await client.post("/auth/login", json={"email": "PAT@example.com", "password": PASSWORD}, headers=CONSOLE)
    assert r.status_code == 200 and (await client.get("/v1/console/merit")).status_code == 200


async def test_session_follows_the_record_and_disabling_ends_it(client, durable):
    inv = await invite(client, "sam@example.com")
    await accept(client, inv["token"])
    assert (await client.get("/v1/reports/me")).json()["role"] == "user"
    await durable.store.update_user("sam@example.com", role="team_lead")
    durable.authn.forget_user("sam@example.com")
    assert (await client.get("/v1/reports/me")).json()["role"] == "team_lead"
    await durable.store.update_user("sam@example.com", disabled=True)
    durable.authn.forget_user("sam@example.com")
    assert (await client.get("/v1/reports/me")).status_code == 401
    client.cookies.clear()
    r = await client.post("/auth/login", json={"email": "sam@example.com", "password": PASSWORD}, headers=CONSOLE)
    assert r.status_code == 403


async def test_changes_with_the_cookie_need_the_console_header_and_this_origin(client, durable):
    inv = await invite(client, "ann@example.com", role="admin")
    await accept(client, inv["token"])
    body = {"email": "x@example.com"}
    assert (await client.post("/v1/admin/invites", json=body)).status_code == 403  # no console header: a cross-site form
    r = await client.post("/v1/admin/invites", json=body, headers={**CONSOLE, "origin": "https://evil.example"})
    assert r.status_code == 403 and "another site" in r.text
    r = await client.post("/v1/admin/invites", json=body, headers={**CONSOLE, "origin": "http://test"})
    assert r.status_code == 201
    # Reading needs neither; keys never need them.
    assert (await client.get("/v1/admin/invites")).status_code == 200
    client.cookies.clear()
    assert (await client.post("/v1/admin/invites", json={"email": "y@example.com"}, headers=auth(ADMIN_KEY))).status_code == 201
    # Sign-in itself is console-only too.
    assert (await client.post("/auth/login", json={"api_key": ADMIN_KEY})).status_code == 403


async def test_a_key_typed_into_the_console_becomes_a_session_tied_to_it(client, durable):
    r = await client.post("/v1/admin/keys", json={"user": "kim", "role": "team_lead"}, headers=auth(ADMIN_KEY))
    key, key_id = r.json()["api_key"], r.json()["id"]
    r = await client.post("/auth/login", json={"api_key": key}, headers=CONSOLE)
    assert r.status_code == 200 and r.json()["method"] == "key"
    me = (await client.get("/v1/reports/me")).json()
    assert me["user"] == "kim" and me["sign_in"]["method"] == "key"
    session = client.cookies.get(SESSION_COOKIE)
    client.cookies.clear()
    await client.delete(f"/v1/admin/keys/{key_id}", headers=auth(ADMIN_KEY))
    client.cookies.set(SESSION_COOKIE, session)
    assert (await client.get("/v1/reports/me")).status_code == 401  # revoking the key ends its sessions
    client.cookies.clear()
    bad = await client.post("/auth/login", json={"api_key": "mo-not-a-key"}, headers=CONSOLE)
    assert bad.status_code == 401


async def test_lead_invites_only_users_on_their_team(client, durable):
    r = await client.post("/v1/admin/invites", json={"email": "boss@example.com", "role": "admin"}, headers=auth(LEAD_KEY))
    assert r.status_code == 403
    r = await client.post("/v1/admin/invites", json={"email": "o@example.com", "team": "ops"}, headers=auth(LEAD_KEY))
    assert r.status_code == 403
    await durable.store.upsert_user("big@example.com", "default", "admin", email="big@example.com")
    r = await client.post("/v1/admin/users/big@example.com/reset", headers=auth(LEAD_KEY))
    assert r.status_code == 403
    inv = await invite(client, "new@example.com", key=LEAD_KEY)
    assert inv["team"] == "default" and inv["role"] == "user"
    assert (await client.post("/v1/admin/invites", json={"email": "z@example.com"}, headers=auth(TEST_KEY))).status_code == 403
    listed = (await client.get("/v1/admin/invites", headers=auth(LEAD_KEY))).json()["invites"]
    assert [i["user"] for i in listed] == ["new@example.com"] and "token" not in listed[0]
    assert (await client.delete(f"/v1/admin/invites/{inv['id']}", headers=auth(LEAD_KEY))).json()["withdrawn"]
    assert (await accept(client, inv["token"])).status_code == 404


async def test_reset_link_and_password_change_end_other_sessions(client, durable):
    inv = await invite(client, "rae@example.com")
    await accept(client, inv["token"])
    first = client.cookies.get(SESSION_COOKIE)
    client.cookies.clear()
    r = await client.post("/auth/login", json={"email": "rae@example.com", "password": PASSWORD}, headers=CONSOLE)
    assert r.status_code == 200
    r = await client.post("/auth/password", json={"current": PASSWORD, "new": "a whole new passphrase"}, headers=CONSOLE)
    assert r.status_code == 200 and r.json()["other_sessions_ended"] == 1
    assert (await client.get("/v1/reports/me")).status_code == 200  # this one stays
    second = client.cookies.get(SESSION_COOKIE)
    client.cookies.clear()
    client.cookies.set(SESSION_COOKIE, first)
    assert (await client.get("/v1/reports/me")).status_code == 401
    client.cookies.clear()
    # An admin's reset link: the old password and every session end when it's used.
    reset = (await client.post("/v1/admin/users/rae@example.com/reset", headers=auth(ADMIN_KEY))).json()
    assert reset["kind"] == "reset"
    assert (await accept(client, reset["token"], "the newest passphrase")).status_code == 200
    client.cookies.clear()
    client.cookies.set(SESSION_COOKIE, second)
    assert (await client.get("/v1/reports/me")).status_code == 401
    client.cookies.clear()
    old = await client.post("/auth/login", json={"email": "rae@example.com", "password": "a whole new passphrase"}, headers=CONSOLE)
    assert old.status_code == 401


async def test_failed_sign_ins_are_limited(client, durable):
    inv = await invite(client, "lim@example.com")
    await accept(client, inv["token"])
    client.cookies.clear()
    for _ in range(accounts.LOGIN_TRIES_PER_ACCOUNT):
        r = await client.post("/auth/login", json={"email": "lim@example.com", "password": "wrong wrong wrong"}, headers=CONSOLE)
        assert r.status_code == 401
    r = await client.post("/auth/login", json={"email": "lim@example.com", "password": PASSWORD}, headers=CONSOLE)
    assert r.status_code == 429


async def test_password_sign_in_can_be_turned_off(client, durable):
    durable.cfg.auth.local.enabled = False
    try:
        r = await client.post("/auth/login", json={"email": "a@example.com", "password": PASSWORD}, headers=CONSOLE)
        assert r.status_code == 403 and "SSO" in r.text
        assert (await client.post("/v1/admin/invites", json={"email": "a@example.com"}, headers=auth(ADMIN_KEY))).status_code == 403
        assert (await client.get("/auth/config")).json()["local"] == {"enabled": False}
        # Keys still open a console session.
        assert (await client.post("/auth/login", json={"api_key": ADMIN_KEY}, headers=CONSOLE)).status_code == 200
    finally:
        durable.cfg.auth.local.enabled = True


async def test_me_says_how_you_signed_in(client, durable):
    me = (await client.get("/v1/reports/me", headers=auth(ADMIN_KEY))).json()
    assert me["sign_in"] == {"method": "key", "has_password": False, "two_factor": False}
    inv = await invite(client, "real@example.com", role="admin")
    await accept(client, inv["token"])
    me = (await client.get("/v1/reports/me")).json()
    assert me["sign_in"] == {"method": "password", "has_password": True, "two_factor": False}


def test_hashes_verify_and_are_salted():
    a, b = accounts.hash_password_sync("pw-one-two-three"), accounts.hash_password_sync("pw-one-two-three")
    assert a != b and a.startswith("scrypt$")
    assert accounts.verify_password_sync("pw-one-two-three", a) and not accounts.verify_password_sync("pw-one-two-four", a)
    assert not accounts.verify_password_sync("x", "garbage")


async def test_people_names_are_scoped_like_reports(client, durable):
    """GET /v1/reports/people: what the console shows instead of an SSO person's GUID, only for people the viewer may see."""
    store = durable.store
    await store.upsert_user("0c1-guid", "default", "user", email="sam@example.com", display_name="Sam Lee")
    await store.upsert_user("0c2-guid", "ops", "user", email="ops@example.com")
    await store.upsert_user("lee", "default", "team_lead")  # no name or address: left out, the id is all there is
    admin = (await client.get("/v1/reports/people", headers=auth(ADMIN_KEY))).json()["people"]
    assert admin == {"0c1-guid": {"name": "Sam Lee", "email": "sam@example.com"}, "0c2-guid": {"name": None, "email": "ops@example.com"}}
    assert set((await client.get("/v1/reports/people", headers=auth(LEAD_KEY))).json()["people"]) == {"0c1-guid"}
    assert (await client.get("/v1/reports/people", headers=auth(TEST_KEY))).json()["people"] == {}
