"""Two-factor sign-in for local accounts (decision 73): turning it on, codes at sign-in (console and CLI), recovery codes,
replays, turning it off, and an admin's reset."""

import time

import pytest
from conftest import ADMIN_KEY, LEAD_KEY, auth

from meritorder import accounts, twofactor

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


async def member(client, email="tf@example.com", role="user"):
    inv = (await client.post("/v1/admin/invites", json={"email": email, "role": role}, headers=auth(ADMIN_KEY))).json()
    r = await client.post("/auth/invite/accept", json={"token": inv["token"], "password": PW}, headers=CONSOLE)
    assert r.status_code == 200


def now_code(secret, offset=0):
    return twofactor.code_at(secret, int(time.time() // twofactor.STEP) + offset)


async def turn_on(client) -> tuple[str, list[str]]:
    setup = (await client.post("/auth/two-factor/setup", headers=CONSOLE)).json()
    assert setup["uri"].startswith("otpauth://totp/Merit%20Order%3Atf%40example.com?secret=")
    bad = await client.post("/auth/two-factor/enable", json={"code": "000000"}, headers=CONSOLE)
    assert bad.status_code == 422
    r = await client.post("/auth/two-factor/enable", json={"code": now_code(setup["secret"])}, headers=CONSOLE)
    assert r.status_code == 200 and len(r.json()["recovery_codes"]) == 10
    return setup["secret"], r.json()["recovery_codes"]


async def login(client, code=None, password=PW, email="tf@example.com"):
    return await client.post("/auth/login", json={"email": email, "password": password, **({"code": code} if code else {})},
                             headers=CONSOLE)


async def test_turn_on_then_codes_are_required(client, durable):
    await member(client)
    secret, recovery = await turn_on(client)
    assert (await client.get("/v1/reports/me")).json()["sign_in"]["two_factor"] is True  # this session stays
    client.cookies.clear()
    r = await login(client)
    assert r.status_code == 401 and r.headers["x-meritorder-two-factor"] == "required" and "authenticator" in r.text
    assert (await login(client, "123456")).status_code == 401
    code = now_code(secret, -1)  # a little clock drift is fine
    assert (await login(client, code)).status_code == 200
    client.cookies.clear()
    r = await login(client, code)  # the same code twice: refused
    assert r.status_code == 401 and "already used" in r.text
    # A recovery code works once, typed any way.
    assert (await login(client, recovery[0].upper().replace("-", " "))).status_code == 200
    client.cookies.clear()
    assert (await login(client, recovery[0])).status_code == 401
    # The CLI's password login asks the same.
    r = await client.post("/auth/exchange", json={"email": "tf@example.com", "password": PW})
    assert r.status_code == 401 and r.headers["x-meritorder-two-factor"] == "required"
    r = await client.post("/auth/exchange", json={"email": "tf@example.com", "password": PW, "code": recovery[1]})
    assert r.status_code == 200 and r.json()["api_key"].startswith("mo-")


async def test_a_reset_link_doesnt_remove_two_factor_but_an_admin_can(client, durable):
    await member(client)
    secret, _ = await turn_on(client)
    reset = (await client.post("/v1/admin/users/tf@example.com/reset", headers=auth(ADMIN_KEY))).json()
    client.cookies.clear()
    await client.post("/auth/invite/accept", json={"token": reset["token"], "password": "a brand new passphrase"}, headers=CONSOLE)
    client.cookies.clear()
    assert (await login(client, password="a brand new passphrase")).headers.get("x-meritorder-two-factor") == "required"
    # A lead can't turn it off for anyone; an admin can.
    assert (await client.post("/v1/admin/users/tf@example.com/two-factor/reset", headers=auth(LEAD_KEY))).status_code == 403
    assert (await client.post("/v1/admin/users/tf@example.com/two-factor/reset", headers=auth(ADMIN_KEY))).json()["reset"]
    assert (await login(client, password="a brand new passphrase")).status_code == 200
    people = (await client.get("/v1/admin/users", headers=auth(ADMIN_KEY))).json()["users"]
    row = next(u for u in people if u["id"] == "tf@example.com")
    assert row["has_password"] is True and row["two_factor"] is False


async def test_turning_it_off_needs_the_password(client, durable):
    await member(client)
    await turn_on(client)
    assert (await client.post("/auth/two-factor/setup", headers=CONSOLE)).status_code == 409
    assert (await client.post("/auth/two-factor/disable", json={"password": "wrong one entirely"}, headers=CONSOLE)).status_code == 401
    assert (await client.post("/auth/two-factor/disable", json={"password": PW}, headers=CONSOLE)).json() == {"enabled": False}
    client.cookies.clear()
    assert (await login(client)).status_code == 200


async def test_only_for_password_accounts(client, durable):
    await client.post("/auth/login", json={"api_key": ADMIN_KEY}, headers=CONSOLE)
    r = await client.post("/auth/two-factor/setup", headers=CONSOLE)
    assert r.status_code == 400 and "SSO" in r.text


def test_secrets_are_sealed_with_the_auth_key(monkeypatch):
    monkeypatch.setenv("MERITORDER_AUTH_KEY", "k" * 32)
    s = twofactor.new_secret()
    stored = twofactor.seal(s, "amy")
    assert stored.startswith("sealed:") and s not in stored and twofactor.unseal(stored, "amy") == s
    with pytest.raises(Exception):
        twofactor.unseal(stored, "bob")  # tied to its person
    monkeypatch.delenv("MERITORDER_AUTH_KEY")
    assert twofactor.seal(s, "amy") == "plain:" + s


def test_rfc_6238_vector():
    assert twofactor.code_at("GEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQ", 59 // 30) == "287082"
    assert twofactor.matching_step("GEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQ", "287082", now=59) == 1
    assert twofactor.matching_step("GEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQ", "28708", now=59) is None
