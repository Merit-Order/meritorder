"""Console sign-in after its security review (decision 69): a team lead's links can't lead to admin, a reset ends what the
old password opened, attempts at once can't pass the limit, passwords only for accounts invited by email, link tokens
off the URL, no free text from the identity provider, and the __Host- cookie over HTTPS."""

import asyncio
from urllib.parse import unquote

import pytest
from conftest import ADMIN_KEY, LEAD_KEY, auth

from meritorder import accounts
from meritorder.auth import SECURE_SESSION_COOKIE, SESSION_COOKIE

CONSOLE = {"x-meritorder-console": "1"}
PW = "correct horse battery staple"


@pytest.fixture(autouse=True)
def fast_hashing(monkeypatch):
    monkeypatch.setattr(accounts, "SCRYPT_N", 2**10)


@pytest.fixture
def durable(gateway):
    kind = gateway.store.kind
    gateway.store.kind = "postgres"
    yield gateway
    gateway.store.kind = kind


async def link(client, email, key=ADMIN_KEY, **kw):
    r = await client.post("/v1/admin/invites", json={"email": email, **kw}, headers=auth(key))
    assert r.status_code == 201, r.text
    return r.json()["token"]


async def accept(client, token, password=PW):
    return await client.post("/auth/invite/accept", json={"token": token, "password": password}, headers=CONSOLE)


async def login(client, email, password=PW):
    return await client.post("/auth/login", json={"email": email, "password": password}, headers=CONSOLE)


async def test_a_leads_link_never_leads_to_admin(client, durable):
    store = durable.store
    # 1. An open link from a lead stops working once the person is promoted.
    token = await link(client, "amy@example.com", key=LEAD_KEY)
    await store.update_user("amy@example.com", role="admin")
    r = await accept(client, token)
    assert r.status_code == 403 and "team lead" in r.text
    # 2. A password a lead's link set acts at most as a user, even after a promotion.
    token = await link(client, "bo@example.com", key=LEAD_KEY)
    assert (await accept(client, token)).json()["role"] == "user"
    await store.update_user("bo@example.com", role="admin")
    durable.authn.forget_user("bo@example.com")
    assert (await client.get("/v1/reports/me")).json()["role"] == "user"  # the session it opened stays a user's
    client.cookies.clear()
    r = await login(client, "bo@example.com")
    assert r.status_code == 403 and "ask an admin" in r.text
    assert (await client.post("/auth/exchange", json={"email": "bo@example.com", "password": PW})).status_code == 403
    # An admin's new link lifts the cap.
    assert (await accept(client, await link(client, "bo@example.com"), "an admin chose this one")).json()["role"] == "admin"
    # 3. Changing a person's team or role withdraws their open link.
    token = await link(client, "cy@example.com", key=LEAD_KEY)
    assert (await client.patch("/v1/admin/users/cy@example.com", json={"role": "team_lead"}, headers=auth(ADMIN_KEY))).status_code == 200
    assert (await accept(client, token)).status_code == 404


async def test_a_reset_or_change_revokes_keys_the_old_password_got(client, durable):
    await accept(client, await link(client, "dee@example.com"))
    client.cookies.clear()
    stolen = (await client.post("/auth/exchange", json={"email": "dee@example.com", "password": PW})).json()["api_key"]
    await accept(client, (await client.post("/v1/admin/users/dee@example.com/reset", headers=auth(ADMIN_KEY))).json()["token"], "a new one now")
    assert (await client.get("/v1/reports/me", headers=auth(stolen))).status_code == 401
    client.cookies.clear()
    assert (await client.post("/auth/login", json={"api_key": stolen}, headers=CONSOLE)).status_code == 401
    # And changing it yourself.
    assert (await login(client, "dee@example.com", "a new one now")).status_code == 200
    again = (await client.post("/auth/exchange", json={"email": "dee@example.com", "password": "a new one now"})).json()["api_key"]
    r = await client.post("/auth/password", json={"current": "a new one now", "new": "and another one"}, headers=CONSOLE)
    assert r.status_code == 200
    assert (await client.get("/v1/reports/me", headers=auth(again))).status_code == 401


async def test_attempts_sent_at_once_cant_pass_the_limit(client, durable):
    await accept(client, await link(client, "eve@example.com"))
    client.cookies.clear()
    results = await asyncio.gather(*(login(client, "eve@example.com", "wrong wrong wrong") for _ in range(30)))
    codes = [r.status_code for r in results]
    assert codes.count(401) <= accounts.LOGIN_TRIES_PER_ACCOUNT and codes.count(429) >= 20
    assert (await login(client, "eve@example.com")).status_code == 429


async def test_a_right_password_doesnt_use_up_the_limit(client, durable):
    await accept(client, await link(client, "fay@example.com"))
    client.cookies.clear()
    for _ in range(accounts.LOGIN_TRIES_PER_ACCOUNT + 3):
        assert (await login(client, "fay@example.com")).status_code == 200


async def test_passwords_only_for_accounts_invited_by_email(client, durable):
    store = durable.store
    await store.upsert_user("oid-123", "default", "user", email="sso@example.com")  # signed in with SSO
    r = await client.post("/v1/admin/users/oid-123/reset", headers=auth(ADMIN_KEY))
    assert r.status_code == 409 and "SSO" in r.text
    r = await client.post("/v1/admin/invites", json={"email": "sso@example.com"}, headers=auth(ADMIN_KEY))
    assert r.status_code == 409
    # A password on such a record (set before this rule) still can't sign in.
    await store.set_password("oid-123", accounts.hash_password_sync(PW))
    assert (await login(client, "sso@example.com")).status_code == 401
    # A lead learns nothing about people outside their team.
    await store.upsert_user("boss@example.com", "ops", "admin", email="boss@example.com")
    r = await client.post("/v1/admin/invites", json={"email": "boss@example.com"}, headers=auth(LEAD_KEY))
    assert r.status_code == 403 and r.json()["detail"] == "only an admin can do this for them"


async def test_link_tokens_never_go_in_the_url(client, durable):
    token = await link(client, "gil@example.com")
    assert (await client.get(f"/auth/invite/{token}")).status_code in (404, 405)
    assert (await client.post("/auth/invite/info", json={"token": token})).status_code == 403  # console-only, like accept
    assert (await client.post("/auth/invite/info", json={"token": token}, headers=CONSOLE)).json()["email"] == "gil@example.com"


async def test_sso_errors_show_codes_not_the_providers_text(client, gateway):
    from test_console_sso import ISSUER, CLIENT  # noqa: F401  (a configured provider)
    from meritorder.config import OIDCConfig

    gateway.cfg.auth.oidc = OIDCConfig(issuer=ISSUER, client_id=CLIENT)
    r = await client.get("/auth/sso/callback", params={"error": "access_denied<b>", "error_description": "Call 555-0100 to restore"})
    message = unquote(r.headers["location"].split("=", 1)[1])
    assert message == "the identity provider refused the sign-in (access_deniedb)" and "555" not in message
    assert accounts.sso_code("") == "unknown_error"


async def test_over_https_only_the_host_prefixed_cookie_counts(client, durable):
    await link(client, "hal@example.com")
    https = {**CONSOLE, "x-forwarded-proto": "https"}
    r = await client.post("/auth/invite/accept", json={"token": await link(client, "hal@example.com"), "password": PW}, headers=https)
    cookies = r.headers.get_list("set-cookie")
    assert any(c.startswith(SECURE_SESSION_COOKIE + "=") and "Secure" in c for c in cookies)
    token = r.cookies.get(SECURE_SESSION_COOKIE) or next(c.split(";")[0].split("=", 1)[1] for c in cookies
                                                          if c.startswith(SECURE_SESSION_COOKIE + "="))
    client.cookies.clear()
    client.cookies.set(SESSION_COOKIE, token)  # the same token under the name any subdomain could set
    assert (await client.get("/auth/session", headers={"x-forwarded-proto": "https"})).json()["signed_in"] is False
    client.cookies.clear()
    client.cookies.set(SECURE_SESSION_COOKIE, token)
    assert (await client.get("/auth/session", headers={"x-forwarded-proto": "https"})).json()["signed_in"] is True
    r = await client.post("/auth/logout", headers=https)
    assert sum(1 for c in r.headers.get_list("set-cookie") if "Max-Age=0" in c or "max-age=0" in c.lower()) == 2


async def test_hashing_refuses_rather_than_queue_without_end(monkeypatch):
    monkeypatch.setattr(accounts, "_waiting", accounts.MAX_WAITING)
    with pytest.raises(accounts.HTTPException) as e:
        await accounts.verify_password("x", None)
    assert e.value.status_code == 503
