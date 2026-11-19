"""Console SSO (decision 69): the authorization-code flow with PKCE against a fake identity provider, the checks on its
answer (state, the browser binding, nonce, audience), and `meritorder login` with a password on gateways without SSO."""

import base64
import hashlib
from urllib.parse import parse_qs, unquote, urlsplit

import pytest
from conftest import ADMIN_KEY, auth
from test_oidc import CLIENT, IDP_KEY, ISSUER, jwk, token

from meritorder import accounts
from meritorder.auth import SECURE_SESSION_COOKIE, Authenticator, OIDCVerifier
from meritorder.config import OIDCConfig

BROWSER = "meritorder-web"
CONSOLE = {"x-meritorder-console": "1"}


class FakeIdP:
    """Discovery, keys, and a token endpoint that answers with whatever ID token the test sets."""

    def __init__(self):
        self.id_token = None
        self.posted = []

    async def discovery(self, _verifier=None):
        return {"issuer": ISSUER, "jwks_uri": "https://login.example.com/keys", "authorization_endpoint": "https://login.example.com/authorize",
                "token_endpoint": "https://login.example.com/token"}


@pytest.fixture
def idp(monkeypatch):
    import jwt

    fake = FakeIdP()
    monkeypatch.setattr(OIDCVerifier, "discovery", lambda self: fake.discovery())
    monkeypatch.setattr(jwt.PyJWKClient, "fetch_data", lambda self: {"keys": [jwk(IDP_KEY, "k1")]})

    class Client:
        def __init__(self, **_):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        async def post(self, url, data=None, headers=None):
            fake.posted.append((url, data))

            class R:
                @staticmethod
                def json():
                    return {"id_token": fake.id_token} if fake.id_token else {"error": "invalid_grant", "error_description": "bad code"}
            return R()

    monkeypatch.setattr(accounts.httpx, "AsyncClient", Client)
    return fake


@pytest.fixture
def sso(gateway, idp, monkeypatch):
    monkeypatch.setenv("WEB_SECRET", "s3cret")
    gateway.cfg.auth.oidc = OIDCConfig(issuer=ISSUER, client_id=CLIENT, browser_client_id=BROWSER, browser_client_secret_env="WEB_SECRET",
                                       team_from_groups={"g-eng": "eng"}, role_from_groups={"g-admins": "admin"})
    gateway.authn = Authenticator(gateway.cfg, gateway.store)
    return gateway


async def start(client):
    r = await client.get("/auth/sso/start")
    assert r.status_code == 303, r.text
    loc = urlsplit(r.headers["location"])
    return r, {k: v[0] for k, v in parse_qs(loc.query).items()}


async def test_sso_signs_in_with_pkce_and_maps_groups(client, sso, idp):
    cfg = (await client.get("/auth/config")).json()
    assert cfg["sso"] == {"provider": "oidc", "start": "/auth/sso/start"}
    r, q = await start(client)
    assert r.headers["location"].startswith("https://login.example.com/authorize?")
    assert q["client_id"] == BROWSER and q["redirect_uri"] == "http://test/auth/sso/callback" and q["code_challenge_method"] == "S256"
    assert "offline_access" not in q["scope"] and "mo_sso" in r.headers["set-cookie"] and "samesite=lax" in r.headers["set-cookie"].lower()
    idp.id_token = token({"aud": BROWSER, "nonce": q["nonce"], "groups": ["g-eng", "g-admins"]})
    r = await client.get("/auth/sso/callback", params={"state": q["state"], "code": "the-code"})
    assert r.status_code == 303 and r.headers["location"] == "/ui/"
    url, form = idp.posted[-1]
    assert url == "https://login.example.com/token" and form["client_secret"] == "s3cret" and form["code"] == "the-code"
    # The verifier sent is the one the challenge was made from.
    assert base64.urlsafe_b64encode(hashlib.sha256(form["code_verifier"].encode()).digest()).rstrip(b"=").decode() == q["code_challenge"]
    me = (await client.get("/v1/reports/me")).json()
    assert (me["user"], me["team"], me["role"], me["sign_in"]["method"]) == ("oid-alice", "eng", "admin", "sso")
    # state works once.
    r = await client.get("/auth/sso/callback", params={"state": q["state"], "code": "the-code"})
    assert "already used" in unquote(r.headers["location"])


async def error_of(r) -> str:
    assert r.status_code == 303 and r.headers["location"].startswith("/ui/#sso-error=")
    return unquote(r.headers["location"].split("=", 1)[1])


async def test_sso_refuses_answers_not_meant_for_this_browser_or_sign_in(client, sso, idp):
    _, q = await start(client)
    idp.id_token = token({"aud": BROWSER, "nonce": "someone-elses"})
    assert "nonce" in await error_of(await client.get("/auth/sso/callback", params={"state": q["state"], "code": "c"}))
    _, q = await start(client)
    idp.id_token = token({"nonce": q["nonce"]})  # issued to the device-login client, not the browser's
    assert "audience" in (await error_of(await client.get("/auth/sso/callback", params={"state": q["state"], "code": "c"}))).lower()
    _, q = await start(client)
    client.cookies.delete("mo_sso", path="/auth/sso")  # another browser's link
    for c in list(client.cookies.jar):
        if c.name == "mo_sso":
            client.cookies.jar.clear(c.domain, c.path, c.name)
    assert "another browser" in await error_of(await client.get("/auth/sso/callback", params={"state": q["state"], "code": "c"}))
    assert "expired" in await error_of(await client.get("/auth/sso/callback", params={"state": "made-up", "code": "c"}))
    _, q = await start(client)
    assert "access_denied" in await error_of(await client.get("/auth/sso/callback", params={"state": q["state"], "error": "access_denied"}))
    _, q = await start(client)
    idp.id_token = None
    assert "invalid_grant" in await error_of(await client.get("/auth/sso/callback", params={"state": q["state"], "code": "c"}))
    assert (await client.get("/auth/session")).json()["signed_in"] is False


async def test_sso_refuses_a_disabled_person(client, sso, idp):
    await sso.store.upsert_user("oid-alice", "eng", "user")
    await sso.store.update_user("oid-alice", disabled=True)
    _, q = await start(client)
    idp.id_token = token({"aud": BROWSER, "nonce": q["nonce"]})
    assert "disabled" in await error_of(await client.get("/auth/sso/callback", params={"state": q["state"], "code": "c"}))


async def test_sso_needs_googles_web_client(client, gateway, idp):
    gateway.cfg.auth.oidc = OIDCConfig(issuer="https://accounts.google.com", client_id="tv-client", hosted_domains=["example.com"])
    assert (await client.get("/auth/config")).json()["sso"] is None
    assert "isn't set up" in await error_of(await client.get("/auth/sso/start"))
    assert "isn't set up" in await error_of(await client.get("/auth/sso/callback?state=x&code=y"))
    gateway.cfg.auth.oidc.browser = False
    gateway.cfg.auth.oidc = OIDCConfig(issuer=ISSUER, client_id=CLIENT, browser=False)
    assert (await client.get("/auth/config")).json()["sso"] is None


async def test_sso_start_reports_an_unreachable_provider(client, sso, monkeypatch):
    async def down(self):
        raise RuntimeError("connection refused")
    monkeypatch.setattr(OIDCVerifier, "discovery", down)
    assert "couldn't reach" in await error_of(await client.get("/auth/sso/start"))


async def test_cli_login_with_a_password_gets_a_key(client, gateway, monkeypatch):
    monkeypatch.setattr(accounts, "SCRYPT_N", 2**10)
    kind = gateway.store.kind
    gateway.store.kind = "postgres"
    try:
        inv = (await client.post("/v1/admin/invites", json={"email": "cli@example.com"}, headers=auth(ADMIN_KEY))).json()
        await client.post("/auth/invite/accept", json={"token": inv["token"], "password": "a long enough passphrase"}, headers=CONSOLE)
        client.cookies.clear()
        bad = await client.post("/auth/exchange", json={"email": "cli@example.com", "password": "wrong wrong wrong"})
        assert bad.status_code == 401
        r = await client.post("/auth/exchange", json={"email": "CLI@example.com", "password": "a long enough passphrase"})
        assert r.status_code == 200 and r.json()["user"] == "cli@example.com" and r.json()["api_key"].startswith("mo-")
        p = await gateway.authn.authenticate(auth(r.json()["api_key"]))
        assert p.user_id == "cli@example.com" and p.role == "user"
        gateway.cfg.auth.local.enabled = False
        assert (await client.post("/auth/exchange", json={"email": "cli@example.com", "password": "x"})).status_code == 403
    finally:
        gateway.store.kind = kind
        gateway.cfg.auth.local.enabled = True


async def test_the_session_cookie_is_secure_behind_https(client, sso, idp):
    _, q = await start(client)
    idp.id_token = token({"aud": BROWSER, "nonce": q["nonce"]})
    r = await client.get("/auth/sso/callback", params={"state": q["state"], "code": "c"}, headers={"x-forwarded-proto": "https"})
    cookie = [h for h in r.headers.get_list("set-cookie") if h.startswith(SECURE_SESSION_COOKIE + "=")][0]
    assert "secure" in cookie.lower() and "httponly" in cookie.lower()
