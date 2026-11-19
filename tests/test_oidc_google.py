"""Google sign-in: Google's discovery document and JWKS (mocked at the HTTP layer), and ID tokens shaped like Google's.

Google's tokens carry no groups, an `hd` claim for Workspace accounts, `email_verified`, and either form of the issuer.
Nothing here reaches Google: discovery goes through an httpx MockTransport, the JWKS through a fake urllib opener.
"""

import io
import json
import time

import jwt
import pytest
from conftest import auth
from cryptography.hazmat.primitives.asymmetric import rsa
from pydantic import ValidationError

from meritorder import auth as auth_mod
from meritorder.auth import AuthError, Authenticator, OIDCVerifier
from meritorder.config import OIDCConfig

CLIENT = "123456789012-abc123def456.apps.googleusercontent.com"
DISCOVERY_URL = "https://accounts.google.com/.well-known/openid-configuration"
JWKS_URL = "https://www.googleapis.com/oauth2/v3/certs"
# Google's discovery document as published (abridged only in response_types_supported).
DISCOVERY = {
    "issuer": "https://accounts.google.com",
    "authorization_endpoint": "https://accounts.google.com/o/oauth2/v2/auth",
    "device_authorization_endpoint": "https://oauth2.googleapis.com/device/code",
    "token_endpoint": "https://oauth2.googleapis.com/token",
    "userinfo_endpoint": "https://openidconnect.googleapis.com/v1/userinfo",
    "revocation_endpoint": "https://oauth2.googleapis.com/revoke",
    "jwks_uri": JWKS_URL,
    "response_types_supported": ["code", "token", "id_token", "code id_token", "none"],
    "subject_types_supported": ["public"],
    "id_token_signing_alg_values_supported": ["RS256"],
    "scopes_supported": ["openid", "email", "profile"],
    "token_endpoint_auth_methods_supported": ["client_secret_post", "client_secret_basic"],
    "claims_supported": ["aud", "email", "email_verified", "exp", "family_name", "given_name", "iat", "iss", "name", "picture", "sub"],
    "code_challenge_methods_supported": ["plain", "S256"],
    "grant_types_supported": ["authorization_code", "refresh_token", "urn:ietf:params:oauth:grant-type:device_code",
                              "urn:ietf:params:oauth:grant-type:jwt-bearer"],
}


def _key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


CURRENT, NEXT, STRANGER = _key(), _key(), _key()  # Google publishes two keys at a time while it rotates


def jwk(private, kid):
    return {**json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(private.public_key())), "kid": kid, "use": "sig", "alg": "RS256"}


def google_token(key=CURRENT, kid="g-current", **over):
    """An ID token like the one Google's device flow returns for openid, email and profile, for a Workspace account."""
    now = int(time.time())
    body = {"iss": "https://accounts.google.com", "azp": CLIENT, "aud": CLIENT, "sub": "110169484474386276334", "hd": "example.com",
            "email": "alice@example.com", "email_verified": True, "at_hash": "HK6E_P6Dh8Y93mRNtsDB1Q", "name": "Alice Example",
            "picture": "https://lh3.googleusercontent.com/a/x", "given_name": "Alice", "family_name": "Example", "iat": now, "exp": now + 3600}
    for k, v in over.items():
        if v is None:
            body.pop(k, None)
        else:
            body[k] = v
    return jwt.encode(body, key, algorithm="RS256", headers={"kid": kid, "typ": "JWT"})


@pytest.fixture
def google(monkeypatch):
    """Serve Google's discovery document and JWKS without network; records every URL fetched."""
    import httpx

    fetched = []

    def handler(request: httpx.Request) -> httpx.Response:
        fetched.append(str(request.url))
        assert str(request.url) == DISCOVERY_URL, f"unexpected request {request.url}"
        return httpx.Response(200, json=DISCOVERY)

    real = httpx.AsyncClient  # the gateway's test client brings its own transport; everything else gets the mock
    monkeypatch.setattr(auth_mod.httpx, "AsyncClient", lambda **kw: real(**{"transport": httpx.MockTransport(handler), **kw}))

    class Opener:
        def open(self, request, timeout=None):
            fetched.append(request.full_url)
            assert request.full_url == JWKS_URL
            return io.BytesIO(json.dumps({"keys": [jwk(CURRENT, "g-current"), jwk(NEXT, "g-next")]}).encode())

    monkeypatch.setattr(jwt.jwks_client.urllib.request, "build_opener", lambda *handlers: Opener())
    return fetched


def google_cfg(**over) -> OIDCConfig:
    return OIDCConfig(**{"issuer": "https://accounts.google.com", "client_id": CLIENT, "public_client_secret": "GOCSPX-public",
                         "hosted_domains": ["example.com"], "default_team": "default",
                         "team_from_emails": {"@example.com": "eng", "bob@example.com": "ops", "research.example.com": "research"},
                         "role_from_emails": {"alice@example.com": "admin", "@finance.example.com": "finance"}, **over})


async def test_good_google_token_maps_by_email_and_uses_sub(google):
    v = OIDCVerifier(google_cfg())
    claims = await v.verify(google_token())
    p, email, name = v.principal(claims)
    assert (p.user_id, p.team, p.role, email, name) == ("110169484474386276334", "eng", "admin", "alice@example.com", "Alice Example")
    assert google == [DISCOVERY_URL, JWKS_URL]
    # Google signs with either form of its issuer, and with whichever of its published keys is current.
    await v.verify(google_token(iss="accounts.google.com", key=NEXT, kid="g-next"))
    assert google == [DISCOVERY_URL, JWKS_URL]  # both read once


@pytest.mark.parametrize("over, team, role", [
    ({"email": "Bob@Example.com"}, "ops", "user"),  # an exact address beats its domain; case doesn't matter
    ({"email": "rae@research.example.com", "hd": "research.example.com"}, "research", "user"),  # a bare domain works too
    ({"email": "fin@finance.example.com", "hd": "finance.example.com"}, "default", "finance"),  # no team mapping: the default
])
async def test_email_and_domain_mappings(google, over, team, role):
    v = OIDCVerifier(google_cfg(hosted_domains=["example.com", "research.example.com", "finance.example.com"]))
    p, _, _ = v.principal(await v.verify(google_token(**over)))
    assert (p.team, p.role) == (team, role)


def test_email_mappings_need_a_verified_address_and_groups_still_count():
    """Entra tokens have no email_verified (and their addresses can be set by users), so email lists never apply to them:
    groups do. Where both apply, an exact address beats a group, and a group beats a domain."""
    v = OIDCVerifier(OIDCConfig(issuer="https://login.example.com/t/v2.0", client_id="c", team_from_groups={"g-ops": "ops"},
                                role_from_groups={"g-leads": "team_lead"}, team_from_emails={"@example.com": "eng", "carol@example.com": "fin"},
                                role_from_emails={"carol@example.com": "admin"}))
    p, email, _ = v.principal({"sub": "s1", "email": "carol@example.com", "preferred_username": "carol@example.com", "groups": ["g-leads"]})
    assert (p.team, p.role, email) == ("default", "team_lead", "carol@example.com")
    p, _, _ = v.principal({"sub": "s1", "email": "dan@example.com", "email_verified": True, "groups": ["g-ops"]})
    assert p.team == "ops"
    p, _, _ = v.principal({"sub": "s1", "email": "carol@example.com", "email_verified": "true", "groups": ["g-ops"]})
    assert (p.team, p.role) == ("fin", "admin")


@pytest.mark.parametrize("case, status, message", [
    ("wrong_domain", 403, "limited to accounts in example.com; this one is in other.com"),
    ("consumer_account", 403, "this one isn't in a Google Workspace"),  # a gmail.com account has no hd claim
    ("unverified_email", 403, "email address isn't verified"),
    ("unverified_as_string", 403, "email address isn't verified"),
    ("expired", 401, "invalid identity token: Signature has expired"),
    ("wrong_audience", 401, "invalid identity token: Audience doesn't match"),  # a token Google issued to another app
    ("wrong_issuer", 401, "invalid identity token: Invalid issuer"),
    ("signed_by_someone_else", 401, "invalid identity token"),
])
async def test_google_tokens_that_are_refused(google, case, status, message):
    v = OIDCVerifier(google_cfg())
    now = int(time.time())
    bad = {
        "wrong_domain": lambda: google_token(hd="other.com", email="eve@other.com"),
        "consumer_account": lambda: google_token(hd=None, email="eve@gmail.com"),
        "unverified_email": lambda: google_token(email_verified=False),
        "unverified_as_string": lambda: google_token(email_verified="false"),
        "expired": lambda: google_token(iat=now - 7200, exp=now - 3600),
        "wrong_audience": lambda: google_token(aud="999-other.apps.googleusercontent.com"),
        "wrong_issuer": lambda: google_token(iss="https://accounts.google.com.evil.example"),
        "signed_by_someone_else": lambda: google_token(key=STRANGER),
    }[case]()
    with pytest.raises(AuthError) as e:
        await v.verify(bad)
    assert e.value.status == status and message in e.value.message


@pytest.mark.parametrize("issuer", ["https://accounts.google.com:443", "https://Accounts.Google.com"])
async def test_google_spelled_another_way_still_needs_hosted_domains(google, issuer):
    """The config check knows Google by the issuer as written, but these spellings reach the same discovery document and the
    tokens are Google's. Without hosted_domains every Google account, gmail.com included, would get a key."""
    v = OIDCVerifier(OIDCConfig(issuer=issuer, client_id=CLIENT))  # passes the config check: no hosted_domains asked for
    with pytest.raises(AuthError) as e:
        await v.verify(google_token(hd=None, email="eve@gmail.com"))
    assert e.value.status == 403 and "hosted_domains" in e.value.message


def test_google_config_needs_hosted_domains_and_gets_its_scopes():
    with pytest.raises(ValidationError, match="Google sign-in needs hosted_domains"):
        OIDCConfig(issuer="https://accounts.google.com/", client_id=CLIENT)
    with pytest.raises(ValidationError, match="needs hosted_domains"):
        OIDCConfig(issuer="https://accounts.google.com", client_id=CLIENT, hosted_domains="")  # ${MERITORDER_OIDC_HOSTED_DOMAINS} unset
    c = OIDCConfig(issuer="https://accounts.google.com", client_id=CLIENT, hosted_domains="Example.com, @example.org,", public_client_secret=" ")
    assert c.is_google and c.hosted_domains == ["example.com", "example.org"] and c.public_client_secret is None
    assert c.scopes == ["openid", "email", "profile"]  # Google refuses offline_access
    c = OIDCConfig(issuer="https://accounts.google.com", client_id=CLIENT, hosted_domains=["example.com"], scopes=["openid", "email"])
    assert c.scopes == ["openid", "email"]  # set explicitly: kept
    entra = OIDCConfig(issuer="https://login.microsoftonline.com/t/v2.0", client_id="c", hosted_domains=None)
    assert not entra.is_google and "offline_access" in entra.scopes and entra.hosted_domains == []
    with pytest.raises(ValidationError, match="valid list"):
        OIDCConfig(issuer="https://accounts.google.com", client_id=CLIENT, hosted_domains=7)


def test_the_shipped_config_takes_google_from_the_environment(monkeypatch):
    """Four variables in .env set up Google sign-in, as two do for Entra; a missing domain list stops the gateway starting."""
    from pathlib import Path

    from meritorder.config import load_config

    shipped = Path(__file__).resolve().parents[1] / "config" / "meritorder.yaml"
    for var in ("MERITORDER_OIDC_CLIENT_SECRET", "MERITORDER_OIDC_HOSTED_DOMAINS"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("MERITORDER_OIDC_ISSUER", "https://login.microsoftonline.com/t/v2.0")
    monkeypatch.setenv("MERITORDER_OIDC_CLIENT_ID", "entra-app")
    entra = load_config(shipped).auth.oidc
    assert (entra.provider, entra.public_client_secret, entra.hosted_domains) == ("entra", None, [])
    monkeypatch.setenv("MERITORDER_OIDC_ISSUER", "https://accounts.google.com")
    monkeypatch.setenv("MERITORDER_OIDC_CLIENT_ID", CLIENT)
    with pytest.raises(ValidationError, match="Google sign-in needs hosted_domains"):
        load_config(shipped)
    monkeypatch.setenv("MERITORDER_OIDC_CLIENT_SECRET", "GOCSPX-public")
    monkeypatch.setenv("MERITORDER_OIDC_HOSTED_DOMAINS", "example.com,example.org")
    google = load_config(shipped).auth.oidc
    assert (google.public_client_secret, google.hosted_domains, google.scopes) == ("GOCSPX-public", ["example.com", "example.org"],
                                                                                   ["openid", "email", "profile"])
    from meritorder.config import load_raw_config
    from meritorder.configstore import secret_problems

    assert secret_problems(load_raw_config(shipped)) == []  # the console can save it as the first stored version


async def test_google_sign_in_through_the_gateway(google, gateway, client):
    gateway.cfg.auth.oidc = google_cfg()
    gateway.authn = Authenticator(gateway.cfg, gateway.store)
    r = await client.get("/auth/config")
    assert r.json() == {"oidc": {"issuer": "https://accounts.google.com", "client_id": CLIENT, "scopes": ["openid", "email", "profile"],
                                 "client_secret": "GOCSPX-public"}, "local": {"enabled": True},
                        "sso": None}  # Google's browser sign-in needs its Web application client

    r = await client.post("/auth/exchange", json={"id_token": google_token()})
    assert r.status_code == 200, r.text
    out = r.json()
    assert (out["user"], out["team"]) == ("alice@example.com", "eng")
    user = await gateway.store.get_user("110169484474386276334")
    assert (user["team"], user["role"], user["email"], user["display_name"]) == ("eng", "admin", "alice@example.com", "Alice Example")
    assert (await client.get("/v1/reports/me", headers=auth(out["api_key"]))).status_code == 200

    r = await client.post("/auth/exchange", json={"id_token": google_token(hd="other.com", email="eve@other.com", sub="2")})
    assert r.status_code == 403 and "limited to accounts in example.com" in r.json()["detail"]
    assert await gateway.store.get_user("2") is None  # refused before anyone is created
    # Sent directly as a bearer token: the same checks.
    body = {"model": "small", "messages": [{"role": "user", "content": "hi"}]}
    r = await client.post("/v1/chat/completions", json=body, headers={"authorization": f"Bearer {google_token(email_verified=False)}"})
    assert r.status_code == 403 and "isn't verified" in r.text
    r = await client.post("/v1/chat/completions", json=body, headers={"authorization": f"Bearer {google_token()}"})
    assert r.status_code == 200


async def test_auth_config_leaves_out_an_unset_client_secret(gateway, client):
    gateway.cfg.auth.oidc = OIDCConfig(issuer="https://login.example.com/t/v2.0", client_id="c")
    assert "client_secret" not in (await client.get("/auth/config")).json()["oidc"]
