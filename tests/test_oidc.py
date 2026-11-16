"""Entra/OIDC tokens: verified against the issuer's keys, mapped to team and role, exchanged for keys.

A local RSA key plays the identity provider; discovery and JWKS fetches are stubbed, so no network.
"""

import base64
import json
import time

import jwt
import pytest
from conftest import auth
from cryptography.hazmat.primitives.asymmetric import rsa

from meritorder.auth import AuthError, Authenticator, OIDCVerifier
from meritorder.config import OIDCConfig

ISSUER = "https://login.example.com/tenant/v2.0"
CLIENT = "meritorder-client"


def _key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


IDP_KEY, OTHER_KEY = _key(), _key()


def jwk(private, kid):
    d = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(private.public_key()))
    return {**d, "kid": kid, "use": "sig", "alg": "RS256"}


def token(claims=None, key=IDP_KEY, kid="k1", alg="RS256", **over):
    now = int(time.time())
    body = {"iss": ISSUER, "aud": CLIENT, "iat": now, "nbf": now, "exp": now + 3600, "sub": "sub-1", "oid": "oid-alice",
            "preferred_username": "alice@example.com", "name": "Alice", "groups": ["g-eng"], **(claims or {})}
    for k, v in over.items():
        if v is None:
            body.pop(k, None)
        else:
            body[k] = v
    return jwt.encode(body, key, algorithm=alg, headers={"kid": kid})


@pytest.fixture
def oidc_cfg():
    return OIDCConfig(issuer=ISSUER, client_id=CLIENT, team_from_groups={"g-eng": "eng", "g-ops": "ops"},
                      role_from_groups={"g-leads": "team_lead", "g-admins": "admin"}, default_team="default", key_ttl_days=7)


@pytest.fixture
def idp(monkeypatch):
    """Serve discovery and JWKS (key k1 only) without network."""
    jwks = {"keys": [jwk(IDP_KEY, "k1")]}
    fetched = []

    async def discovery(self):
        return {"issuer": ISSUER, "jwks_uri": "https://login.example.com/keys"}

    def fetch_data(self):
        fetched.append(self.uri)
        return jwks

    monkeypatch.setattr(OIDCVerifier, "discovery", discovery)
    monkeypatch.setattr(jwt.PyJWKClient, "fetch_data", fetch_data)
    return fetched


async def test_valid_token_maps_groups_to_team_and_highest_role(idp, oidc_cfg):
    v = OIDCVerifier(oidc_cfg)
    claims = await v.verify(token({"groups": ["g-other", "g-eng", "g-leads", "g-admins"]}))
    p, email, name = v.principal(claims)
    assert (p.user_id, p.team, p.role, email, name) == ("oid-alice", "eng", "admin", "alice@example.com", "Alice")
    p, _, _ = v.principal({"sub": "s-2", "groups": []})  # no oid: falls back to sub; no groups: defaults
    assert (p.user_id, p.team, p.role) == ("s-2", "default", "user")


@pytest.mark.parametrize("case", ["expired", "wrong_audience", "wrong_issuer", "forged_signature", "unknown_kid",
                                  "not_yet_valid", "no_exp"])
async def test_bad_tokens_are_rejected(idp, oidc_cfg, case):
    v = OIDCVerifier(oidc_cfg)
    now = int(time.time())
    bad = {
        "expired": lambda: token(exp=now - 10),
        "wrong_audience": lambda: token(aud="someone-else"),
        "wrong_issuer": lambda: token(iss="https://evil.example.com"),
        "forged_signature": lambda: token(key=OTHER_KEY),  # claims kid k1 but signed by another key
        "unknown_kid": lambda: token(kid="k-missing"),
        "not_yet_valid": lambda: token(nbf=now + 600),
        "no_exp": lambda: token(exp=None),  # must not be valid forever
    }[case]()
    with pytest.raises(AuthError) as e:
        await v.verify(bad)
    assert e.value.status == 401 and "invalid identity token" in e.value.message


async def test_algorithm_confusion_is_rejected(idp, oidc_cfg):
    """'none' and HS256-signed-with-the-public-key tokens must never verify."""
    v = OIDCVerifier(oidc_cfg)
    unsigned = jwt.encode({"iss": ISSUER, "aud": CLIENT, "exp": int(time.time()) + 60, "sub": "x"}, None, algorithm="none",
                          headers={"kid": "k1"})
    with pytest.raises(AuthError):
        await v.verify(unsigned)
    from cryptography.hazmat.primitives import serialization

    pem = IDP_KEY.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    header = base64.urlsafe_b64encode(json.dumps({"alg": "HS256", "kid": "k1", "typ": "JWT"}).encode()).rstrip(b"=")
    payload = base64.urlsafe_b64encode(json.dumps({"iss": ISSUER, "aud": CLIENT, "exp": int(time.time()) + 60,
                                                   "sub": "x"}).encode()).rstrip(b"=")
    import hashlib
    import hmac

    sig = base64.urlsafe_b64encode(hmac.new(pem, header + b"." + payload, hashlib.sha256).digest()).rstrip(b"=")
    with pytest.raises(AuthError):
        await v.verify((header + b"." + payload + b"." + sig).decode())


async def test_bearer_token_through_the_gateway_is_cached_and_follows_the_user_record(idp, oidc_cfg, gateway, client):
    gateway.cfg.auth.oidc = oidc_cfg
    gateway.authn = Authenticator(gateway.cfg, gateway.store)
    tok = token({"groups": ["g-ops"]})
    body = {"model": "small", "messages": [{"role": "user", "content": "hi"}]}
    r = await client.post("/v1/chat/completions", json=body, headers={"authorization": f"Bearer {tok}"})
    assert r.status_code == 200
    p = await gateway.authn.authenticate({"authorization": f"Bearer {tok}"})
    assert (p.user_id, p.team) == ("oid-alice", "ops")
    assert len(idp) == 1  # JWKS fetched once, then cached

    # An admin moves Alice; after the per-gateway cache is dropped, her token maps to the new team.
    await gateway.store.upsert_user("oid-alice", "ops", "user")
    await gateway.store.update_user("oid-alice", team="eng")
    gateway.authn.forget_user("oid-alice")
    assert (await gateway.authn.authenticate({"authorization": f"Bearer {tok}"})).team == "eng"
    # Disabled: refused even with a valid token.
    await gateway.store.update_user("oid-alice", disabled=True)
    gateway.authn.forget_user("oid-alice")
    r = await client.post("/v1/chat/completions", json=body, headers={"authorization": f"Bearer {tok}"})
    assert r.status_code == 401 and "disabled" in r.text

    r = await client.post("/v1/chat/completions", json=body, headers={"authorization": f"Bearer {token(exp=int(time.time()) - 5)}"})
    assert r.status_code == 401


async def test_oidc_cache_never_outlives_the_token(idp, oidc_cfg, gateway):
    gateway.cfg.auth.oidc = oidc_cfg
    authn = Authenticator(gateway.cfg, gateway.store)
    tok = token(exp=int(time.time()) + 5)
    await authn.authenticate({"authorization": f"Bearer {tok}"})
    [(_, until)] = authn._cache.values()
    assert until <= time.time() + 5.5


async def test_exchange_trades_a_token_for_a_short_lived_key(idp, oidc_cfg, gateway, client):
    gateway.cfg.auth.oidc = oidc_cfg
    gateway.authn = Authenticator(gateway.cfg, gateway.store)
    r = await client.post("/auth/exchange", json={"id_token": token({"groups": ["g-eng", "g-leads"]})})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["user"] == "alice@example.com" and out["team"] == "eng"
    user = await gateway.store.get_user("oid-alice")
    assert (user["team"], user["role"], user["email"]) == ("eng", "team_lead", "alice@example.com")
    me = await client.get("/v1/reports/me", headers=auth(out["api_key"]))
    assert me.status_code == 200
    bad = await client.post("/auth/exchange", json={"id_token": token(key=OTHER_KEY)})
    assert bad.status_code == 401
    assert (await client.post("/auth/exchange", json={})).status_code == 401


async def test_exchange_without_oidc_configured(gateway, client):
    r = await client.post("/auth/exchange", json={"id_token": "a.b.c"})
    assert r.status_code == 400 and "not configured" in r.text


async def test_api_key_states(gateway):
    from datetime import UTC, datetime, timedelta

    authn = gateway.authn
    with pytest.raises(AuthError, match="missing API key"):
        await authn.authenticate({})
    with pytest.raises(AuthError, match="invalid API key"):
        await authn.authenticate({"x-api-key": "mo-nope"})
    key_id, secret = await gateway.store.create_key("bob", "default", "user", expires_at=datetime.now(UTC) - timedelta(days=1))
    with pytest.raises(AuthError, match="expired"):
        await authn.authenticate({"x-api-key": secret})
    key_id, secret = await gateway.store.create_key("bob", "default", "user")
    assert (await authn.authenticate({"authorization": f"Bearer {secret}"})).key_id == key_id
    await gateway.store.revoke_key(key_id)
    authn.forget(key_id)
    with pytest.raises(AuthError, match="revoked"):
        await authn.authenticate({"x-api-key": secret})


async def test_bootstrap_admin_key(monkeypatch, gateway):
    monkeypatch.setenv("MERITORDER_BOOTSTRAP_ADMIN_KEY", "mo-bootstrap-123")
    p = await Authenticator(gateway.cfg, gateway.store).authenticate({"x-api-key": "mo-bootstrap-123"})
    assert p.role == "admin" and p.key_id == "static:bootstrap"
