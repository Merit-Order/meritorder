"""Google Groups → teams and roles: a fake Google (token endpoint, metadata server, Cloud Identity Groups API) behind an
httpx MockTransport, and Google ID tokens from test_oidc_google. Nothing here reaches Google.

The fake answers as Google documents the Cloud Identity API v1: ``searchTransitiveGroups`` and ``searchDirectGroups`` with
a CEL query, pages of results with ``nextPageToken``, ``GroupRelation`` objects with ``groupKey.id`` and ``relationType``,
and errors as ``{"error": {"code", "message", "status", "details": [{"reason"}]}}``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from types import SimpleNamespace

import httpx
import jwt
import pytest
from conftest import ADMIN_KEY, LEAD_KEY, auth
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from pydantic import ValidationError
from test_oidc_google import CLIENT, google, google_token  # noqa: F401 (google is a fixture)

from meritorder import google_groups as gg
from meritorder.auth import Authenticator, assign
from meritorder.config import GoogleGroupsConfig, OIDCConfig
from meritorder.google_groups import SCOPE, GroupsClient, GroupsError
from meritorder.state import MemoryState

SA = "meritorder-groups@my-project.iam.gserviceaccount.com"
SA_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
OTHER_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
ALICE = "110169484474386276334"  # the Google ID token's sub


def pem(key) -> str:
    return key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()).decode()


def sa_json(key=SA_KEY, kid="key-1", **over) -> str:
    """A service-account key as the Cloud console downloads it."""
    return json.dumps({"type": "service_account", "project_id": "my-project", "private_key_id": kid, "private_key": pem(key),
                       "client_email": SA, "client_id": "1234567890", "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                       "token_uri": "https://oauth2.googleapis.com/token", **over})


class FakeGoogle:
    """Google's token endpoint, the metadata server and the Groups API over one directory of memberships."""

    def __init__(self, members: dict[str, list[str]], enterprise: bool = True, page: int = 2):
        self.members = members  # member address -> the groups it's directly in
        self.enterprise = enterprise  # Workspace Enterprise or Cloud Identity Premium: transitive search works
        self.page = page
        self.fail: str | None = None
        self.calls: list[tuple[str, str]] = []  # (method, member) per page fetched
        self.tokens: list[dict] = []  # claims of every assertion exchanged
        self.metadata: list[httpx.Request] = []
        self.issued = 0
        self.refuse_token_once = False
        self.transport = httpx.MockTransport(self.handle)

    def transitive(self, member: str) -> list[dict]:
        found: dict[str, set[str]] = {}
        queue = [(g, "DIRECT") for g in self.members.get(member, [])]
        while queue:
            g, how = queue.pop(0)
            if how in found.setdefault(g, set()):
                continue
            found[g].add(how)
            queue += [(p, "INDIRECT") for p in self.members.get(g, [])]
        rel = {frozenset({"DIRECT"}): "DIRECT", frozenset({"INDIRECT"}): "INDIRECT"}
        return [{"group": f"groups/{g}", "groupKey": {"id": g}, "displayName": g.split("@")[0].title(),
                 "relationType": rel.get(frozenset(h), "DIRECT_AND_INDIRECT"), "roles": [{"role": "MEMBER"}],
                 "labels": {gg.EVERY_GROUP: ""}} for g, h in found.items()]

    def direct(self, member: str) -> list[dict]:
        return [{"group": f"groups/{g}", "groupKey": {"id": g}, "displayName": g.split("@")[0].title(),
                 "membership": f"groups/{g}/memberships/m", "roles": [{"name": "MEMBER"}], "labels": {gg.EVERY_GROUP: ""}}
                for g in self.members.get(member, [])]

    @staticmethod
    def error(code: int, status: str, message: str, reason: str | None = None) -> httpx.Response:
        details = [{"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": reason, "domain": "googleapis.com"}] if reason else []
        return httpx.Response(code, json={"error": {"code": code, "message": message, "status": status, "details": details}})

    def handle(self, request: httpx.Request) -> httpx.Response:
        url = request.url
        if url.host == "oauth2.googleapis.com":
            form = dict(x.split("=", 1) for x in request.content.decode().split("&"))
            assert form["grant_type"] == "urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Ajwt-bearer"
            try:
                claims = jwt.decode(form["assertion"], SA_KEY.public_key(), algorithms=["RS256"], audience=str(url))
            except jwt.PyJWTError as e:
                return httpx.Response(400, json={"error": "invalid_grant", "error_description": f"Invalid JWT Signature ({type(e).__name__})."})
            self.tokens.append({**claims, "_kid": jwt.get_unverified_header(form["assertion"]).get("kid")})
            self.issued += 1
            return httpx.Response(200, json={"access_token": f"ya29.token-{self.issued}", "expires_in": 3599, "token_type": "Bearer"})
        if url.host == "metadata.google.internal":
            self.metadata.append(request)
            if self.fail == "no-metadata":
                raise httpx.ConnectError("Name or service not known")
            self.issued += 1
            return httpx.Response(200, json={"access_token": f"ya29.wi-{self.issued}", "expires_in": 3599, "token_type": "Bearer"})
        assert url.host == "cloudidentity.googleapis.com", f"unexpected request {url}"
        assert request.headers["authorization"].startswith("Bearer ya29.")
        method = url.path.rsplit(":", 1)[1]
        query = url.params["query"]
        m = re.match(r"member_key_id == '((?:[^'\\]|\\.)*)' && '([^']+)' in labels$", query)
        assert m, query
        member = re.sub(r"\\(.)", r"\1", m.group(1))
        self.calls.append((method, member))
        if self.fail == "outage":
            raise httpx.ConnectError("connection refused")
        if self.fail == "503":
            return self.error(503, "UNAVAILABLE", "The service is currently unavailable.")
        if self.fail == "scope":
            return self.error(403, "PERMISSION_DENIED", "Request had insufficient authentication scopes.", "ACCESS_TOKEN_SCOPE_INSUFFICIENT")
        if self.fail == "disabled":
            return self.error(403, "PERMISSION_DENIED", "Cloud Identity API has not been used in project 1234567890 before or it is "
                              "disabled.", "SERVICE_DISABLED")
        if self.fail == "role":
            return self.error(403, "PERMISSION_DENIED", "The caller does not have permission")
        if self.refuse_token_once:
            self.refuse_token_once = False
            return self.error(401, "UNAUTHENTICATED", "Request had invalid authentication credentials.")
        if method == "searchTransitiveGroups":
            if not self.enterprise:
                return self.error(403, "PERMISSION_DENIED", "Error(4003): Permission denied for the requested resource.")
            rows = self.transitive(member)
        else:
            assert method == "searchDirectGroups"
            rows = self.direct(member)
        start = int(url.params.get("pageToken") or 0)
        body: dict = {"memberships": rows[start:start + self.page]}
        if start + self.page < len(rows):
            body["nextPageToken"] = str(start + self.page)
        return httpx.Response(200, json=body)


DIRECTORY = {
    "alice@example.com": ["eng@example.com", "all@example.com"],
    "eng@example.com": ["staff@example.com"],              # nested: alice is in staff@ through eng@
    "staff@example.com": ["gateway-admins@example.com"],   # two levels down
    "bob@example.com": ["all@example.com"],
}


@pytest.fixture
def fake():
    return FakeGoogle({k: list(v) for k, v in DIRECTORY.items()})


def client_for(fake, monkeypatch, **cfg) -> GroupsClient:
    monkeypatch.setenv("GG_KEY", sa_json())
    c = GroupsClient(GoogleGroupsConfig(**{"credentials_env": "GG_KEY", **cfg}), fake.transport)
    return c


async def lookup(c: GroupsClient, address: str) -> list[dict]:
    async with c.http() as http:
        return await c.groups_of(address, http)


# ---------------------------------------------------------------------------- the client


async def test_a_key_signs_a_token_and_transitive_search_finds_nested_groups(fake, tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    keyfile = tmp_path / "groups.json"
    keyfile.write_text(sa_json())
    c = GroupsClient(GoogleGroupsConfig(credentials_file=str(keyfile)), fake.transport)
    got = await lookup(c, "Alice@Example.com".lower())
    assert got == [{"email": "all@example.com", "name": "All", "direct": True}, {"email": "eng@example.com", "name": "Eng", "direct": True},
                   {"email": "gateway-admins@example.com", "name": "Gateway-Admins", "direct": False},
                   {"email": "staff@example.com", "name": "Staff", "direct": False}]
    # Four groups, two a page: two calls, and the second carried the page token.
    assert fake.calls == [("searchTransitiveGroups", "alice@example.com")] * 2 and c.transitive is True
    # The assertion: the service account itself (no domain-wide delegation, so no sub), the read-only scope, Google's
    # token endpoint as the audience, and the key's id.
    (claims,) = fake.tokens
    assert claims["iss"] == SA and claims["scope"] == SCOPE and claims["aud"] == "https://oauth2.googleapis.com/token"
    assert "sub" not in claims and claims["_kid"] == "key-1" and claims["exp"] - claims["iat"] == 3600
    await lookup(c, "bob@example.com")
    assert len(fake.tokens) == 1  # the token is reused until shortly before it expires
    text = caplog.text
    assert "PRIVATE KEY" not in text and "ya29." not in text


async def test_a_key_from_the_environment_and_a_rotated_one(fake, monkeypatch):
    c = client_for(fake, monkeypatch)
    await lookup(c, "bob@example.com")
    monkeypatch.setenv("GG_KEY", sa_json(kid="key-2"))  # rotated: the new key gets its own token at once
    await lookup(c, "bob@example.com")
    assert [t["_kid"] for t in fake.tokens] == ["key-1", "key-2"] and c.credentials.account == SA


async def test_without_a_key_the_metadata_server_gives_the_identity(fake, monkeypatch):
    monkeypatch.delenv("GG_KEY", raising=False)
    c = GroupsClient(GoogleGroupsConfig(credentials_env="GG_KEY"), fake.transport)  # set but empty: workload identity
    assert [g["email"] for g in await lookup(c, "bob@example.com")] == ["all@example.com"]
    (req,) = fake.metadata
    assert req.headers["metadata-flavor"] == "Google" and req.url.params["scopes"] == SCOPE
    assert req.url.path == "/computeMetadata/v1/instance/service-accounts/default/token"
    fake.fail = "no-metadata"
    c = GroupsClient(GoogleGroupsConfig(), fake.transport)
    with pytest.raises(GroupsError, match="no service-account key .* and no metadata server"):
        await lookup(c, "bob@example.com")


@pytest.mark.parametrize("key, message", [
    ("{not json", "isn't JSON"),
    (json.dumps({"type": "authorized_user", "client_id": "x"}), "doesn't hold a service-account key"),
    (sa_json(private_key="-----BEGIN PRIVATE KEY-----\nnot a key\n-----END PRIVATE KEY-----\n"), "private_key in \\$GG_KEY can't be read"),
    (sa_json(key=OTHER_KEY), r"Google refused the service account's key \(400 invalid_grant: Invalid JWT Signature"),  # deleted key
])
async def test_keys_that_dont_work_say_why_without_showing_the_key(fake, monkeypatch, key, message):
    monkeypatch.setenv("GG_KEY", key)
    c = GroupsClient(GoogleGroupsConfig(credentials_env="GG_KEY"), fake.transport)
    for _ in range(2):  # every time: a key that can't be used never sends the gateway to another identity (the metadata server)
        with pytest.raises(GroupsError, match=message) as e:
            await lookup(c, "bob@example.com")
        assert "PRIVATE KEY" not in str(e.value) and e.value.outage is False


async def test_editions_without_transitive_search_follow_the_nesting_themselves(fake, monkeypatch, caplog):
    fake.enterprise = False  # Business editions: Google refuses transitive search with a 403
    fake.members["gateway-admins@example.com"] = ["staff@example.com"]  # a loop: must not spin
    c = client_for(fake, monkeypatch)
    with caplog.at_level(logging.INFO, "meritorder.google_groups"):
        got = await lookup(c, "alice@example.com")
    assert {g["email"]: g["direct"] for g in got} == {"all@example.com": True, "eng@example.com": True, "staff@example.com": False,
                                                       "gateway-admins@example.com": False}
    assert c.transitive is False and "direct search works" in caplog.text
    assert fake.calls[0] == ("searchTransitiveGroups", "alice@example.com")
    assert [m for m, _ in fake.calls[1:]] == ["searchDirectGroups"] * (len(fake.calls) - 1)
    fake.calls.clear()
    await lookup(c, "bob@example.com")  # transitive isn't tried again, and all@'s parents are already known
    assert fake.calls == [("searchDirectGroups", "bob@example.com")]


async def test_a_nested_group_that_cant_be_read_is_left_out(fake, monkeypatch, caplog):
    fake.enterprise = False
    c = client_for(fake, monkeypatch)
    real = fake.handle

    def handle(request):
        if "staff%40example.com" in str(request.url) or "staff@example.com" in str(request.url.params.get("query")):
            return FakeGoogle.error(403, "PERMISSION_DENIED", "The caller does not have permission")
        return real(request)

    c.transport = httpx.MockTransport(handle)
    assert {g["email"] for g in await lookup(c, "alice@example.com")} == {"all@example.com", "eng@example.com", "staff@example.com"}
    assert "couldn't read which groups contain staff@example.com" in caplog.text


@pytest.mark.parametrize("fail, message", [
    ("scope", "the token lacks the scope https://www.googleapis.com/auth/cloud-identity.groups.readonly"),
    ("disabled", "turn on the Cloud Identity API in the service account's project"),
    ("role", f"give {SA} the Groups Reader admin role \\(Admin console → Account → Admin roles\\)"),
])
async def test_a_403_says_what_to_fix(fake, monkeypatch, fail, message):
    fake.fail = fail
    c = client_for(fake, monkeypatch)
    with pytest.raises(GroupsError, match=message) as e:
        await lookup(c, "alice@example.com")
    assert e.value.status == 403 and not e.value.outage
    # Transitive search, then direct search to tell an edition limit from a permission problem: both refused.
    assert [m for m, _ in fake.calls] == ["searchTransitiveGroups", "searchDirectGroups"] and c.transitive is None


async def test_an_expired_token_is_replaced_once_and_outages_are_outages(fake, monkeypatch):
    c = client_for(fake, monkeypatch)
    fake.refuse_token_once = True
    assert len(await lookup(c, "bob@example.com")) == 1 and fake.issued == 2  # a 401: one new token, then it worked
    for fail in ("outage", "503"):
        fake.fail = fail
        with pytest.raises(GroupsError) as e:
            await lookup(c, "bob@example.com")
        assert e.value.outage, fail


async def test_addresses_are_quoted_for_the_query(fake, monkeypatch):
    fake.members["o'brien@example.com"] = ["eng@example.com"]
    c = client_for(fake, monkeypatch)
    assert [g["email"] for g in await lookup(c, "o'brien@example.com")] == ["eng@example.com", "gateway-admins@example.com",
                                                                              "staff@example.com"]
    assert gg._cel("a'b\\c") == "'a\\'b\\\\c'"


async def test_security_groups_only_asks_for_the_security_label(fake, monkeypatch):
    c = client_for(fake, monkeypatch, security_groups_only=True)
    seen = []
    real = fake.handle
    c.transport = httpx.MockTransport(lambda r: (seen.append(r.url.params.get("query")), real(r))[1])
    await lookup(c, "bob@example.com")
    assert any(q and "'cloudidentity.googleapis.com/groups.security' in labels" in q for q in seen)


# ---------------------------------------------------------------------------- mapping


def google_groups_cfg(**over) -> OIDCConfig:
    return OIDCConfig(**{"issuer": "https://accounts.google.com", "client_id": CLIENT, "public_client_secret": "GOCSPX-public",
                         "hosted_domains": ["example.com"],
                         "team_from_groups": {"research@example.com": "research", "Eng@Example.com": "eng"},
                         "role_from_groups": {"gateway-admins@example.com": "admin", "leads@example.com": "team_lead"},
                         "team_from_emails": {"@example.com": "default", "carol@example.com": "ops"},
                         "role_from_emails": {"carol@example.com": "finance"},
                         "google_groups": {"credentials_env": "GG_KEY", "refresh_seconds": 60, "keep_last_known_hours": 24}, **over})


def test_precedence_address_then_group_then_domain_then_default():
    cfg = google_groups_cfg()
    a = assign(cfg, "carol@example.com", ["eng@example.com", "gateway-admins@example.com"])
    assert (a.team, a.team_from, a.role, a.role_from) == ("ops", {"by": "address", "key": "carol@example.com"},
                                                          "admin", {"by": "group", "key": "gateway-admins@example.com"})
    # The first group in team_from_groups wins, whatever order Google lists them in; case doesn't matter.
    a = assign(cfg, "dan@example.com", ["ENG@example.com", "research@example.com"])
    assert (a.team, a.team_from, a.role, a.role_from) == ("research", {"by": "group", "key": "research@example.com"}, "user", None)
    a = assign(cfg, "dan@example.com", [])
    assert (a.team, a.team_from) == ("default", {"by": "domain", "key": "@example.com"})
    a = assign(cfg.model_copy(update={"team_from_emails": {}}), "dan@example.com", [])
    assert (a.team, a.team_from) == ("default", {"by": "default", "key": None})
    a = assign(cfg, "carol@example.com", ["leads@example.com"])  # the highest role wins: finance (address) over team_lead
    assert (a.role, a.role_from) == ("finance", {"by": "address", "key": "carol@example.com"})


def test_google_groups_config_is_checked():
    with pytest.raises(ValidationError, match="google_groups works only with Google sign-in"):
        OIDCConfig(issuer="https://login.microsoftonline.com/t/v2.0", client_id="c", google_groups={})
    OIDCConfig(issuer="https://login.microsoftonline.com/t/v2.0", client_id="c", google_groups={"enabled": "false"})  # off: fine
    with pytest.raises(ValidationError, match="keyed by group address .* not an address: 0e3b-guid"):
        google_groups_cfg(team_from_groups={"0e3b-guid": "eng"})
    g = google_groups_cfg(google_groups={"credentials_file": " ", "credentials_env": ""}).google_groups
    assert (g.credentials_file, g.credentials_env, g.refresh_seconds, g.keep_last_known_hours) == (None, None, 60, 24)
    with pytest.raises(ValidationError):
        GoogleGroupsConfig(refresh_seconds=0)


def test_the_shipped_config_reads_groups_from_the_environment(monkeypatch):
    from pathlib import Path

    from meritorder.config import load_config, load_raw_config
    from meritorder.configstore import secret_problems

    shipped = Path(__file__).resolve().parents[1] / "config" / "meritorder.yaml"
    monkeypatch.setenv("MERITORDER_OIDC_ISSUER", "https://accounts.google.com")
    monkeypatch.setenv("MERITORDER_OIDC_CLIENT_ID", CLIENT)
    monkeypatch.setenv("MERITORDER_OIDC_HOSTED_DOMAINS", "example.com")
    monkeypatch.delenv("MERITORDER_GOOGLE_GROUPS", raising=False)
    assert not load_config(shipped).auth.oidc.reads_google_groups
    monkeypatch.setenv("MERITORDER_GOOGLE_GROUPS", "true")
    oidc = load_config(shipped).auth.oidc
    assert oidc.reads_google_groups and oidc.google_groups.credentials_env == "MERITORDER_GOOGLE_GROUPS_CREDENTIALS"
    assert secret_problems(load_raw_config(shipped)) == []  # still saveable from the console


# ---------------------------------------------------------------------------- through the gateway


@pytest.fixture
def durable(gateway):
    kind = gateway.store.kind
    gateway.store.kind = "postgres"  # People page edits need a durable store; the in-memory one behaves the same here
    yield gateway
    gateway.store.kind = kind


@pytest.fixture
def signed_in(google, gateway, fake, monkeypatch):  # noqa: F811 (google is the fixture imported above)
    monkeypatch.setenv("GG_KEY", sa_json())
    monkeypatch.delenv("MERITORDER_ALERT_WEBHOOK_URL", raising=False)
    gateway.cfg.auth.oidc = google_groups_cfg()
    gateway.authn = Authenticator(gateway.cfg, gateway.store, gateway.state)
    gateway.authn.groups.client.transport = fake.transport
    return gateway


async def exchange(client, **over):
    r = await client.post("/auth/exchange", json={"id_token": google_token(**over)})
    assert r.status_code == 200, r.text
    return r.json()


async def age(gw, user_id: str, seconds: float, fetched: bool = False) -> None:
    """Pretend the last reading happened ``seconds`` earlier (and the person used the gateway since)."""
    rec = await gw.state.hget_json(gg.RECORDS, user_id)
    rec["checked_at"] -= seconds
    if fetched:
        rec["fetched_at"] -= seconds
    await gw.state.hset_json(gg.RECORDS, user_id, rec)
    await gw.authn.groups.seen(user_id)


async def test_sign_in_reads_groups_and_the_people_page_shows_them(signed_in, durable, client, fake):
    gw = signed_in
    out = await exchange(client)
    assert out["team"] == "eng"
    user = await gw.store.get_user(ALICE)
    assert (user["team"], user["role"]) == ("eng", "admin")  # eng@ directly, gateway-admins@ two levels down
    rec = await gw.state.hget_json(gg.RECORDS, ALICE)
    assert rec["using"] == "live" and rec["team_from"] == {"by": "group", "key": "eng@example.com"}
    assert rec["role_from"] == {"by": "group", "key": "gateway-admins@example.com"}

    people = (await client.get("/v1/admin/users", headers=auth(ADMIN_KEY))).json()
    assert people["google_groups"] is True and people["google_groups_refresh_seconds"] == 60
    alice = next(u for u in people["users"] if u["id"] == ALICE)["google"]
    assert [(g["email"], g["direct"], g["mapped"]) for g in alice["groups"]] == [
        ("all@example.com", True, False), ("eng@example.com", True, True), ("gateway-admins@example.com", False, True),
        ("staff@example.com", False, False)]
    assert alice["hidden"] == 0 and alice["error"] is None and alice["fetched_at"]

    # A team lead sees only the groups the mappings use. Put alice on the lead's team to check.
    await gw.store.update_user(ALICE, team="default")
    lead = (await client.get("/v1/admin/users", headers=auth(LEAD_KEY))).json()
    alice = next(u for u in lead["users"] if u["id"] == ALICE)["google"]
    assert [g["email"] for g in alice["groups"]] == ["eng@example.com", "gateway-admins@example.com"] and alice["hidden"] == 2

    # Follow sign-in: the groups are read at once and the mappings apply now, not at the next sign-in.
    r = await client.patch(f"/v1/admin/users/{ALICE}", json={"follow_idp": True}, headers=auth(ADMIN_KEY))
    assert (r.json()["team"], r.json()["managed"]) == ("eng", "idp")


async def test_a_removal_takes_effect_at_the_next_refresh_for_active_people(signed_in, client, fake):
    gw = signed_in
    key = (await exchange(client))["api_key"]
    await exchange(client, sub="2", email="bob@example.com", name="Bob")
    assert (await gw.store.get_user("2"))["team"] == "default"  # all@ isn't mapped: the domain
    me = await client.get("/v1/reports/me", headers=auth(key))
    assert me.status_code == 200

    fake.members["alice@example.com"] = ["all@example.com", "research@example.com"]  # moved out of eng@ (and so admins)
    fake.calls.clear()
    assert await gw.authn.groups.refresh_due() == []  # read moments ago: not due yet
    await age(gw, ALICE, 61)
    rec = await gw.state.hget_json(gg.RECORDS, "2")
    await gw.state.hset_json(gg.RECORDS, "2", {**rec, "checked_at": rec["checked_at"] - 61})  # bob hasn't used it since
    await gw.state.hset_json(gg.SEEN, "2", rec["checked_at"] - 100)
    assert await gw.authn.groups.refresh_due() == [ALICE]
    assert {m for _, m in fake.calls} == {"alice@example.com"}  # bob wasn't active: not asked about
    user = await gw.store.get_user(ALICE)
    assert (user["team"], user["role"]) == ("research", "user")
    rec = await gw.state.hget_json(gg.RECORDS, ALICE)
    assert rec["team_from"] == {"by": "group", "key": "research@example.com"} and rec["role_from"] is None
    # The key already issued follows at once on this gateway (its cache entry was dropped), and on others within a minute.
    assert (await gw.authn.authenticate(auth(key))).team == "research"

    # Pinned on the People page: the groups are still read, but the pin wins.
    await gw.store.update_user(ALICE, team="ops")
    fake.members["alice@example.com"] = ["eng@example.com"]
    await age(gw, ALICE, 61)
    assert await gw.authn.groups.refresh_due() == []
    assert (await gw.store.get_user(ALICE))["team"] == "ops"
    assert (await gw.state.hget_json(gg.RECORDS, ALICE))["team"] == "eng"  # what the mappings would give, for the page


async def test_when_google_is_unreachable_the_last_known_groups_stand_for_a_while(signed_in, client, fake):
    gw = signed_in
    await exchange(client)
    fake.fail = "outage"
    await age(gw, ALICE, 61)
    assert await gw.authn.groups.refresh_due() == []
    user = await gw.store.get_user(ALICE)
    assert (user["team"], user["role"]) == ("eng", "admin")  # last known groups
    rec = await gw.state.hget_json(gg.RECORDS, ALICE)
    assert rec["using"] == "last_known" and "couldn't reach the Cloud Identity API" in rec["error"] and rec["failing_since"]
    alerts = await gw.state.recent_alerts(10)
    assert [a["title"] for a in alerts] == ["Google Groups unavailable"] and alerts[0]["page"] == "people"
    assert "last known groups for up to 24 hours" in alerts[0]["text"]

    await age(gw, ALICE, 25 * 3600, fetched=True)  # still failing a day later: the email and domain rules alone
    assert await gw.authn.groups.refresh_due() == [ALICE]
    user = await gw.store.get_user(ALICE)
    assert (user["team"], user["role"]) == ("default", "user")
    rec = await gw.state.hget_json(gg.RECORDS, ALICE)
    assert rec["using"] == "none" and rec["team_from"] == {"by": "domain", "key": "@example.com"}
    assert "Google Groups expired: email rules apply" in [a["title"] for a in await gw.state.recent_alerts(10)]

    fake.fail = None  # back: the groups apply again, and the recovery is announced once
    await age(gw, ALICE, 61)
    assert await gw.authn.groups.refresh_due() == [ALICE]
    assert (await gw.store.get_user(ALICE))["team"] == "eng"
    titles = [a["title"] for a in await gw.state.recent_alerts(10)]
    assert titles[0] == "Google Groups readable again" and titles.count("Google Groups readable again") == 1


async def test_sign_in_while_google_refuses_falls_back_to_the_email_rules(signed_in, client, fake):
    gw = signed_in
    fake.fail = "scope"
    out = await exchange(client)
    assert out["team"] == "default"  # no groups known: the domain mapping
    rec = await gw.state.hget_json(gg.RECORDS, ALICE)
    assert rec["using"] == "none" and "lacks the scope" in rec["error"]
    (alert,) = await gw.state.recent_alerts(10)
    assert alert["level"] == "critical" and "lacks the scope" in alert["text"]  # a setup problem, not an outage

    fake.fail = None
    await exchange(client)  # signed in again: read live
    fake.fail = "outage"
    out = await exchange(client)  # Google down at the next sign-in: the groups read a moment ago stand
    assert out["team"] == "eng" and (await gw.state.hget_json(gg.RECORDS, ALICE))["using"] == "last_known"


async def test_a_google_token_sent_directly_uses_the_cached_groups(signed_in, client, fake):
    gw = signed_in
    await exchange(client)
    await gw.store.update_user(ALICE, team="ops")  # the record decides; check the mapping the token gets on its own
    principal, _, _ = gw.authn.oidc.principal(await gw.authn.oidc.verify(google_token()), await gw.authn.groups.known_groups(ALICE))
    assert principal.team == "eng"
    calls = len(fake.calls)
    body = {"model": "small", "messages": [{"role": "user", "content": "hi"}]}
    r = await client.post("/v1/chat/completions", json=body, headers={"authorization": f"Bearer {google_token()}"})
    assert r.status_code == 200 and len(fake.calls) == calls  # nothing asked of Google on a request


async def test_deleted_and_disabled_people_are_not_read_again(signed_in, client, fake):
    gw = signed_in
    await exchange(client)
    await gw.store.update_user(ALICE, disabled=True)
    await age(gw, ALICE, 61)
    fake.calls.clear()
    assert await gw.authn.groups.refresh_due() == [] and fake.calls == []
    if gw.store.kind == "memory":
        gw.store.users.pop(ALICE)
    else:
        async with gw.store.pool.acquire() as c:
            await c.execute("DELETE FROM users WHERE id=$1", ALICE)
    await age(gw, ALICE, 61)
    assert await gw.authn.groups.refresh_due() == []
    assert await gw.state.hget_json(gg.RECORDS, ALICE) is None  # forgotten


async def test_one_gateway_runs_each_round(signed_in, monkeypatch):
    gw = signed_in
    rounds = []

    async def refresh_due():
        rounds.append(time.time())
        return []

    monkeypatch.setattr(gg, "TICK_SECONDS", 0.01)
    monkeypatch.setattr(gw.authn.groups, "refresh_due", refresh_due)
    task = asyncio.create_task(gg.follow(gw))
    await asyncio.sleep(0.2)
    task.cancel()
    assert rounds  # claimed through the shared state, so a second gateway on the same state would skip the round

    gw.authn.groups = None  # turned off by a configuration change: the loop idles
    rounds.clear()
    task = asyncio.create_task(gg.follow(gw))
    await asyncio.sleep(0.05)
    task.cancel()
    assert rounds == []


class RoundTrips(MemoryState):
    """The shared state as Redis is to a gateway: every call waits for an answer, so two gateways' calls interleave."""

    async def get_json(self, key):
        await asyncio.sleep(0)
        return await super().get_json(key)

    async def set_json(self, key, value, ttl=0):
        await asyncio.sleep(0)
        return await super().set_json(key, value, ttl)

    async def set_json_once(self, items, ttl):
        await asyncio.sleep(0)
        return await super().set_json_once(items, ttl)


async def test_two_gateways_ticking_together_take_one_round(signed_in, monkeypatch):
    gw = signed_in
    rounds = []

    async def refresh_due():
        rounds.append(time.time())
        return []

    monkeypatch.setattr(gg, "TICK_SECONDS", 0.01)
    monkeypatch.setattr(gw.authn.groups, "refresh_due", refresh_due)
    shared = RoundTrips()
    tasks = [asyncio.create_task(gg.follow(SimpleNamespace(authn=gw.authn, state=shared))) for _ in range(2)]
    await asyncio.sleep(0.2)
    for task in tasks:
        task.cancel()
    assert len(rounds) == 1  # a round is claimed for a second: one gateway reads Google, the other skips it


def test_describe():
    assert gg.describe({"by": "group", "key": "eng@example.com"}) == "group eng@example.com"
    assert gg.describe({"by": "default", "key": None}) == gg.describe(None) == "default team"


async def test_people_who_signed_in_before_groups_were_on_are_read_at_their_next_use(signed_in, client, fake):
    gw = signed_in
    fake.members["dana@example.com"] = ["eng@example.com"]
    await gw.store.upsert_user("104857600000000000001", "default", "user", email="Dana@example.com")  # a Google sign-in, no reading
    await gw.store.upsert_user("dave@example.com", "default", "user", source="ensure")  # given a key on the Keys page: no address
    for uid in ("104857600000000000001", "dave@example.com"):
        await gw.authn.groups.seen(uid)
    assert await gw.authn.groups.refresh_due() == ["104857600000000000001"]
    assert (await gw.store.get_user("104857600000000000001"))["team"] == "eng"
    assert await gw.state.hget_json(gg.RECORDS, "dave@example.com") is None
    assert set(fake.calls) == {("searchTransitiveGroups", "dana@example.com")}  # two pages
    await gw.authn.groups.refresh_due()
    assert "dave@example.com" in gw.authn.groups._not_google  # not looked up in the store again


def _refuse(request):
    raise httpx.ConnectTimeout("slow")


async def test_token_sources_that_fail(monkeypatch):
    monkeypatch.setenv("GG_KEY", sa_json())
    c = GroupsClient(GoogleGroupsConfig(credentials_env="GG_KEY"), httpx.MockTransport(_refuse))
    with pytest.raises(GroupsError, match="couldn't reach Google's token endpoint") as e:
        await lookup(c, "bob@example.com")
    assert e.value.outage
    c = GroupsClient(GoogleGroupsConfig(), httpx.MockTransport(lambda r: httpx.Response(404, text="not found")))
    with pytest.raises(GroupsError, match="the metadata server gave no token \\(404\\)"):
        await lookup(c, "bob@example.com")


async def test_a_failure_for_one_person_doesnt_stop_the_round(signed_in, client, fake, monkeypatch, caplog):
    gw = signed_in
    await exchange(client)
    await age(gw, ALICE, 61)

    async def broken(user_id, client=None):
        raise RuntimeError("store down")

    monkeypatch.setattr(gw.authn.groups, "refresh", broken)
    assert await gw.authn.groups.refresh_due() == []
    assert f"couldn't refresh {ALICE}'s Google groups" in caplog.text

    async def failing():
        raise RuntimeError("redis down")

    monkeypatch.setattr(gg, "TICK_SECONDS", 0.01)
    monkeypatch.setattr(gw.authn.groups, "refresh_due", failing)
    task = asyncio.create_task(gg.follow(gw))
    await asyncio.sleep(0.05)
    task.cancel()
    assert "couldn't refresh Google groups" in caplog.text  # logged; the loop goes on


def test_check_says_what_google_groups_would_call_as(tmp_path, monkeypatch, capsys):
    import copy

    import yaml
    from conftest import TEST_CONFIG

    from meritorder import cli

    raw = copy.deepcopy(TEST_CONFIG)
    raw["auth"]["oidc"] = {"issuer": "https://accounts.google.com", "client_id": CLIENT, "hosted_domains": "example.com",
                           "team_from_groups": {"eng@example.com": "default"},
                           "google_groups": {"credentials_env": "GG_KEY", "credentials_file": str(tmp_path / "missing.json")}}
    path = tmp_path / "meritorder.yaml"
    path.write_text(yaml.safe_dump(raw))
    monkeypatch.setenv("GG_KEY", sa_json())
    cli.main(["check", "--config", str(path)])
    out = capsys.readouterr().out
    assert "Sign-in: google, https://accounts.google.com" in out
    assert f"Google Groups: on; key in {tmp_path / 'missing.json'} or $GG_KEY (service account {SA}); 1 group(s) mapped" in out
    assert "PRIVATE KEY" not in out  # nothing is sent to Google, and the key is never shown
    monkeypatch.delenv("GG_KEY")
    cli.main(["check", "--config", str(path)])
    assert "no key: the identity of the machine or pod" in capsys.readouterr().out
    monkeypatch.setenv("GG_KEY", "{oops")
    cli.main(["check", "--config", str(path)])
    assert "isn't JSON" in capsys.readouterr().out
