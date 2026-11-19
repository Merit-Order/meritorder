"""Client authentication: virtual API keys, plus OIDC tokens (Entra, Google, others) exchanged for keys.

Harnesses send a Merit Order key as ``Authorization: Bearer <key>`` or
``x-api-key: <key>`` (so Claude Code and OpenAI SDKs work unchanged). Services
can send an OIDC access/ID token directly as the bearer token.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from urllib.parse import urlsplit

import httpx

from .config import GOOGLE_ISSUERS, Config, OIDCConfig
from .store import hash_key

log = logging.getLogger("meritorder.auth")

ROLE_RANK = {"user": 0, "team_lead": 1, "finance": 2, "admin": 3}
SESSION_COOKIE = "mo_session"
CSRF_HEADER = "x-meritorder-console"  # the console sends it on every call; a cross-site form or link can't
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


def lower_role(a: str, b: str) -> str:
    """The lower of two roles by ROLE_RANK. A person's record may lower what a key acts as, never raise it: a key a lead
    minted as `user` stays a user key even if its person turns out to be an admin."""
    return a if ROLE_RANK.get(a, 0) <= ROLE_RANK.get(b, 0) else b


class AuthError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


@dataclass
class Principal:
    user_id: str
    team: str
    role: str = "user"
    key_id: str | None = None
    tags: dict = field(default_factory=dict)
    budget_usd_month: float | None = None  # the person's own monthly limit; None means their team's per-person default
    # The gateway's own requests (judged labels' replays and judging, judged.py): "replay" or "judge". Never from a key.
    internal: str | None = None
    session: str | None = None  # how the console session was opened ("password", "sso", "key"), for a request with its cookie

    def monthly_limit(self, team_default: float | None) -> float | None:
        return self.budget_usd_month if self.budget_usd_month is not None else team_default

    def at_least(self, role: str) -> bool:
        return ROLE_RANK.get(self.role, 0) >= ROLE_RANK[role]


def check_same_origin(headers: dict[str, str]) -> None:
    """For a request sent with the session cookie that changes something: the console's header, and an Origin (when the
    browser sends one) naming this gateway. SameSite=Strict on the cookie already keeps it off cross-site requests."""
    if not headers.get(CSRF_HEADER):
        raise AuthError(403, f"a change made with the console's session needs the {CSRF_HEADER} header")
    origin = headers.get("origin")
    if origin and origin != "null":
        host = headers.get("x-forwarded-host") or headers.get("host") or ""
        if urlsplit(origin).netloc.lower() != host.split(",")[0].strip().lower():
            raise AuthError(403, "this request came from another site")
    elif origin == "null":
        raise AuthError(403, "this request came from another site")


def bearer_token(headers: dict[str, str]) -> str | None:
    auth = headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip() or None
    return headers.get("x-api-key") or None


class OIDCVerifier:
    def __init__(self, cfg: OIDCConfig):
        self.cfg = cfg
        self._discovery: dict | None = None
        self._jwks = None

    async def discovery(self) -> dict:
        if self._discovery is None:
            url = self.cfg.issuer.rstrip("/") + "/.well-known/openid-configuration"
            async with httpx.AsyncClient(timeout=10) as c:
                resp = await c.get(url)
                resp.raise_for_status()
                self._discovery = resp.json()
        return self._discovery

    async def verify(self, token: str, audience: str | None = None) -> dict:
        """The token's claims, if it's genuine, current and for this gateway: ``audience`` (default: the configured
        audience or client_id), the issuer, and an account allowed in."""
        import jwt

        disc = await self.discovery()
        if self._jwks is None:
            self._jwks = jwt.PyJWKClient(disc["jwks_uri"], cache_keys=True)
        issuer = disc.get("issuer", self.cfg.issuer)
        google = issuer.rstrip("/") in GOOGLE_ISSUERS  # what the tokens are, however the configured issuer is spelled
        try:
            key = await asyncio.to_thread(self._jwks.get_signing_key_from_jwt, token)
            claims = jwt.decode(
                token,
                key.key,
                algorithms=["RS256", "RS384", "RS512", "ES256", "ES384"],
                audience=audience or self.cfg.audience or self.cfg.client_id,
                issuer=list(GOOGLE_ISSUERS) if google else issuer,
                options={"require": ["exp", "iss", "aud"]},  # a token without an expiry must not be valid forever
            )
        except jwt.PyJWTError as e:
            raise AuthError(401, f"invalid identity token: {e}")
        self._check_account(claims, google)
        return claims

    def _check_account(self, claims: dict, google: bool = False) -> None:
        """A genuine token can still be for someone who mustn't get in: an address the provider says isn't verified, or
        (Google) an account outside the organization's Workspace domains. Google's issuer is shared by every Google
        account, so the `hd` claim is what keeps strangers out; the domain in an email address proves nothing."""
        if "email_verified" in claims and not _verified(claims):
            raise AuthError(403, "this account's email address isn't verified; verify it with your identity provider and sign in again")
        if google and not self.cfg.hosted_domains:  # the config check missed it (an issuer spelled another way): refuse everyone
            raise AuthError(403, "Google sign-in needs auth.oidc.hosted_domains (your Google Workspace domains): without it any "
                                 "Google account could sign in")
        if self.cfg.hosted_domains:
            hd = str(claims.get("hd") or "").lower()
            if hd not in self.cfg.hosted_domains:
                where = f"is in {hd}" if hd else "isn't in a Google Workspace"
                raise AuthError(403, f"sign-in is limited to accounts in {', '.join(self.cfg.hosted_domains)}; this one {where}")

    def principal(self, claims: dict, groups: list[str] | None = None) -> tuple[Principal, str | None, str | None]:
        """Team and role (``assign``) from the token's groups, or from ``groups`` when the gateway read them itself
        (Google Groups), and for a verified address, the email mappings."""
        if groups is None and self.cfg.groups_claim not in claims and self.cfg.groups_claim in (claims.get("_claim_names") or {}):
            # Entra's group overage: someone in more than 200 groups gets a pointer to Microsoft Graph instead of the list.
            # The gateway doesn't call Graph, so only the email mappings and the defaults apply (docs/06).
            log.warning("%s's token lists no groups (Entra group overage: more than 200 groups); group mappings don't apply. "
                        "Filter the token's groups to those assigned to the app, or map this person by email", user_id_of(claims))
        found = assign(self.cfg, verified_address(claims), (claims.get(self.cfg.groups_claim) or []) if groups is None else groups)
        email = claims.get(self.cfg.email_claim) or claims.get("email")
        return Principal(user_id=user_id_of(claims), team=found.team, role=found.role), email, claims.get("name")


def user_id_of(claims: dict) -> str:
    return str(claims.get("oid") or claims.get("sub"))


def verified_address(claims: dict) -> str | None:
    """The token's email address, lowercased, if the token says it's verified (Google's do; Entra's never say)."""
    return str(claims["email"]).lower() if claims.get("email") and _verified(claims) else None


def _verified(claims: dict) -> bool:
    return claims.get("email_verified") in (True, "true")  # older Google tokens sent the string


def _by_email(mapping: dict[str, str]) -> dict[str, str]:
    """Keys lowercased; a bare domain ("example.com") means the same as "@example.com"."""
    return {(k if "@" in k else "@" + k).lower(): v for k, v in mapping.items()}


@dataclass
class Assignment:
    """A team and role from the sign-in mappings, and which mapping gave each, as ``{"by": "address" | "group" |
    "domain" | "default", "key": ...}``. ``role_from`` is None when no mapping gave a role (everyone is a user)."""

    team: str
    role: str
    team_from: dict
    role_from: dict | None = None


def assign(cfg: OIDCConfig, address: str | None, groups) -> Assignment:
    """Team: the mapping for the exact address, else the first group in ``team_from_groups`` the person is in, else the
    address's domain, else ``default_team``. Role: the highest any of those mappings gives. Groups match ignoring case
    (Entra's are IDs, Google's are addresses). Address and domain mappings apply only to a verified ``address``."""
    member = {str(g).lower() for g in groups}
    domain = "@" + address.rpartition("@")[2] if address else None
    teams, roles = _by_email(cfg.team_from_emails), _by_email(cfg.role_from_emails)
    by_group = next(((g.lower(), t) for g, t in cfg.team_from_groups.items() if g.lower() in member), None)
    if address and address in teams:
        team, team_from = teams[address], {"by": "address", "key": address}
    elif by_group:
        team, team_from = by_group[1], {"by": "group", "key": by_group[0]}
    elif domain and domain in teams:
        team, team_from = teams[domain], {"by": "domain", "key": domain}
    else:
        team, team_from = cfg.default_team, {"by": "default", "key": None}
    role, role_from = "user", None
    candidates = [*(("group", g.lower(), r) for g, r in cfg.role_from_groups.items() if g.lower() in member),
                  ("address", address, roles.get(address or "")), ("domain", domain, roles.get(domain or ""))]
    for by, key, r in candidates:
        if r and ROLE_RANK[r] > ROLE_RANK[role]:
            role, role_from = r, {"by": by, "key": key}
    return Assignment(team=team, role=role, team_from=team_from, role_from=role_from)


class Authenticator:
    def __init__(self, cfg: Config, store, state=None):
        self.cfg = cfg
        self.store = store
        self.static: dict[str, Principal] = {}
        for sk in cfg.auth.static_keys:
            key = sk.resolved()
            if key and len(key) < 24:
                log.warning("static key for %s is %d characters; use at least 24 random ones (openssl rand -hex 24)", sk.user, len(key))
            if key:
                self.static[hash_key(key)] = Principal(user_id=sk.user, team=sk.team, role=sk.role, key_id=f"static:{sk.user}",
                                                       budget_usd_month=sk.budget_usd_month)
        bootstrap = os.environ.get("MERITORDER_BOOTSTRAP_ADMIN_KEY")
        if bootstrap:
            self.static[hash_key(bootstrap)] = Principal(user_id="admin", team="default", role="admin", key_id="static:bootstrap")
        self.oidc = OIDCVerifier(cfg.auth.oidc) if cfg.auth.oidc else None
        self.groups = None  # Google Groups (google_groups.py), when sign-in is Google's and reading groups is on
        if cfg.auth.oidc and cfg.auth.oidc.reads_google_groups:
            from .google_groups import GoogleGroups
            from .state import MemoryState

            self.groups = GoogleGroups(cfg.auth.oidc, store, state if state is not None else MemoryState(), forget=self.forget_user)
        self._cache: dict[str, tuple[Principal, float]] = {}
        self._sessions: dict[str, tuple[Principal, float]] = {}  # console sessions, by the cookie's hash
        self._limits: dict[str, tuple[float | None, float]] = {}  # static-key person -> (limit from their record, expiry)

    async def authenticate(self, headers: dict[str, str]) -> Principal:
        token = bearer_token(headers)
        if not token:
            raise AuthError(401, "missing API key: send 'Authorization: Bearer <key>' or 'x-api-key: <key>'")
        h = hash_key(token)
        cached = self._cache.get(h)
        if h not in self.static and not (cached and cached[1] > time.time()) and self.oidc and token.count(".") == 2 \
                and not token.startswith("mo-"):
            claims = await self.oidc.verify(token)
            groups = await self.groups.known_groups(user_id_of(claims)) if self.groups else None  # cached: no call on this path
            principal, _, _ = self.oidc.principal(claims, groups)
            principal = await self._apply_user_record(principal)
            self._cache[h] = (principal, min(time.time() + 60, float(claims.get("exp", time.time() + 60))))
            return principal
        return await self.by_key_hash(h)

    async def by_key_hash(self, h: str) -> Principal:
        """The principal for a Merit Order key (static or issued), by its hash: also how a console session opened with a
        key is checked, so revoking the key ends the session."""
        if h in self.static:
            return await self._static_with_limit(self.static[h])
        cached = self._cache.get(h)
        if cached and cached[1] > time.time():
            return cached[0]
        row = await self.store.find_key(h)
        if not row:
            raise AuthError(401, "invalid API key")
        now = datetime.now(UTC)
        if row.get("revoked_at"):
            raise AuthError(401, "API key has been revoked")
        if row.get("expires_at") and row["expires_at"] < now:
            raise AuthError(401, "API key has expired; run 'meritorder login' for a new one")
        principal = Principal(user_id=row["user_id"], team=row["team"], role=row["role"], key_id=row["id"], tags=row.get("tags") or {})
        principal = await self._apply_user_record(principal, key_role=row["role"])
        self._cache[h] = (principal, time.time() + 60)
        return principal

    async def authenticate_console(self, request) -> Principal:
        """The console's APIs (reports, console, admin, controls) and /auth: a key in the headers as everywhere, or the
        browser's session cookie. A cookie is sent by the browser on its own, so a request that changes something must
        also carry the console's own header and, if it says where it came from, come from this gateway (no cross-site
        request can do either: the gateway allows no other origin)."""
        headers = {k.lower(): v for k, v in request.headers.items()}
        if bearer_token(headers):
            return await self.authenticate(headers)
        token = request.cookies.get(SESSION_COOKIE)
        if not token:
            raise AuthError(401, "sign in to the console, or send 'Authorization: Bearer <key>'")
        if request.method not in SAFE_METHODS:
            check_same_origin(headers)
        return await self.session_principal(token)

    async def session_principal(self, token: str) -> Principal:
        """A console session: its person's record as it is now (a minute's cache at most, like keys), or, for one opened
        with a key, that key as it is now."""
        h = hash_key(token)
        cached = self._sessions.get(h)
        if cached and cached[1] > time.time():
            return cached[0]
        row = await self.store.find_session(h)
        now = datetime.now(UTC)
        if not row or row.get("revoked_at") or row["expires_at"] <= now:
            raise AuthError(401, "your console session has ended; sign in again")
        if row.get("key_hash"):
            principal = await self.by_key_hash(row["key_hash"])
        else:
            user = await self.store.get_user(row["user_id"])
            if not user:
                raise AuthError(401, "your console session has ended; sign in again")
            if user.get("disabled_at"):
                raise AuthError(401, "this account is disabled; contact your Merit Order admin")
            principal = Principal(user_id=user["id"], team=user["team"], role=user["role"], budget_usd_month=user.get("budget_usd_month"))
        principal = replace(principal, session=row["method"])  # a copy: the key's own cache entry stays a key's
        self._sessions[h] = (principal, min(time.time() + 60, row["expires_at"].timestamp()))
        return principal

    def forget_session(self, token: str) -> None:
        self._sessions.pop(hash_key(token), None)

    async def _apply_user_record(self, p: Principal, key_role: str | None = None) -> Principal:
        """The user's record (from the sign-in mappings at login, or set by an admin) decides team and role, so moving
        or disabling someone takes effect within a minute, whatever their keys say. For a key (``key_role``, the role it
        was issued with) the record may only lower the role: the key acts as the lower of the two."""
        user = await self.store.get_user(p.user_id)
        if not user:
            return p
        if user.get("disabled_at"):
            raise AuthError(401, "this account is disabled; contact your Merit Order admin")
        if self.groups:
            try:
                await self.groups.seen(p.user_id)  # their Google groups are read again at the next round
            except Exception as e:  # noqa: BLE001 (a state hiccup mustn't refuse the request)
                log.warning("couldn't note %s as active for the Google groups refresh: %s", p.user_id, e)
        role = user["role"] if key_role is None else lower_role(user["role"], key_role)
        return Principal(user_id=p.user_id, team=user["team"], role=role, key_id=p.key_id, tags=p.tags,
                         budget_usd_month=user.get("budget_usd_month"))

    async def _static_with_limit(self, p: Principal) -> Principal:
        """A static key's team, role and limit come from the config file until the person has a record (People page,
        sign-in, SCIM). Then the record applies as it does to any key: disabling them refuses the key, their team is
        the record's, their role the lower of the record's and the file's, and a monthly limit there wins over the
        file's. The record is looked up at most once a minute per person."""
        cached = self._limits.get(p.user_id)
        if cached is None or cached[1] <= time.time():
            try:
                user = await self.store.get_user(p.user_id)
            except Exception as e:  # static keys keep working without the database (the config file applies)
                log.warning("couldn't read %s's record, using the config file's team, role and limit: %s", p.user_id, e)
                user = None
            cached = (user, time.time() + 60)
            self._limits[p.user_id] = cached
        user = cached[0]
        if not user:
            return p
        if user.get("disabled_at"):
            raise AuthError(401, "this account is disabled; contact your Merit Order admin")
        limit = user.get("budget_usd_month")
        return Principal(user_id=p.user_id, team=user.get("team") or p.team, role=lower_role(user.get("role") or p.role, p.role),
                         key_id=p.key_id, tags=p.tags, budget_usd_month=p.budget_usd_month if limit is None else limit)

    def forget_user(self, user_id: str) -> None:
        self._limits.pop(user_id, None)
        for cache in (self._cache, self._sessions):
            for h, (principal, _) in list(cache.items()):
                if principal.user_id == user_id:
                    cache.pop(h, None)

    def forget(self, key_id: str) -> None:
        """Drop a key from the lookup cache (after revocation) so it stops working at once on this gateway, with the
        console sessions opened with it."""
        for cache in (self._cache, self._sessions):
            for h, (principal, _) in list(cache.items()):
                if principal.key_id == key_id:
                    cache.pop(h, None)

    async def sign_in(self, claims: dict) -> tuple[Principal, str | None]:
        """Someone with a verified identity token signs in (the CLI's device login, or the console's SSO): their record
        is created or refreshed from the sign-in mappings, and they act as it says (an admin's pinned team and role win).
        Refused for a disabled account. Returns their principal as the record has it, and their email address."""
        groups = None
        address = verified_address(claims)
        if self.groups and address:  # Google's tokens carry no groups: ask Google (or, if it can't answer, the cache)
            groups = await self.groups.at_sign_in(user_id_of(claims), address)
        principal, email, name = self.oidc.principal(claims, groups)
        current = await self.store.get_user(principal.user_id)
        if current and current.get("disabled_at"):  # refuse before a key or session exists, not on its first use
            raise AuthError(403, "this account is disabled; contact your Merit Order admin")
        await self.store.upsert_user(principal.user_id, principal.team, principal.role, email, name)
        record = await self.store.get_user(principal.user_id) or {}
        self.forget_user(principal.user_id)  # the record may have changed: the next request reads it again
        return replace(principal, team=record.get("team") or principal.team, role=record.get("role") or principal.role), email

    async def exchange(self, id_token: str) -> dict:
        """Trade a verified OIDC token for a short-lived Merit Order key."""
        if not self.oidc:
            raise AuthError(400, "OIDC is not configured on this gateway")
        claims = await self.oidc.verify(id_token)
        principal, email = await self.sign_in(claims)
        # The key is issued as the record says, so it acts as the person does now; a later change to their record can
        # lower it, never raise it.
        team, role = principal.team, principal.role
        expires = datetime.now(UTC) + timedelta(days=self.cfg.auth.oidc.key_ttl_days)
        key_id, secret = await self.store.create_key(principal.user_id, team, role, name="oidc login", expires_at=expires)
        return {"api_key": secret, "key_id": key_id, "expires_at": expires.isoformat(), "user": email or principal.user_id, "team": team}
