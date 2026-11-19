"""Console sign-in (decision 69): local accounts, browser sessions and invite links.

People sign in to the console with an email address and password (local accounts, for teams without an identity
provider) or with their organization's SSO (OIDC), and get a session cookie. Keys are for harnesses, CI and services;
the console still accepts one, but exchanges it for a session instead of keeping it in the browser.

* GET  /auth/session                     whether this browser is signed in (200 either way)
* POST /auth/login                       email and password, or an API key: opens a session (HttpOnly cookie)
* POST /auth/logout                      ends this session
* POST /auth/password                    change your own password; your other sessions end
* GET  /auth/sso/start                   to the organization's sign-in page (OIDC authorization code with PKCE)
* GET  /auth/sso/callback                back from it: a session, then the console
* POST /auth/exchange (email, password)  a short-lived key for `meritorder login` on a gateway with local accounts
* GET  /auth/invite/{token}              who an invite or password-reset link is for
* POST /auth/invite/{token}              set a password with it, and sign in
* GET  /v1/admin/invites                 open invite and reset links (admin: all; team lead: their team's)
* POST /v1/admin/invites                 invite someone (a one-time link, shown once)
* DELETE /v1/admin/invites/{id}          withdraw one
* POST /v1/admin/users/{user_id}/reset   a password-reset link for someone

Passwords are stored as scrypt hashes (no extra dependency), session cookies and links only as SHA-256 hashes.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import logging
import re
import secrets
from datetime import UTC, datetime, timedelta

from urllib.parse import quote, urlencode

import httpx
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from .auth import ROLE_RANK, SESSION_COOKIE, AuthError, Principal, check_same_origin, user_id_of
from .config import Role
from .store import hash_key

log = logging.getLogger("meritorder.accounts")

# scrypt at OWASP's recommended cost (N=2^17, r=8, p=1: 128 MiB and about a quarter second per hash). A few at a time,
# so a burst of sign-ins can't take the gateway's memory.
SCRYPT_N, SCRYPT_R, SCRYPT_P = 2**17, 8, 1
MAX_PASSWORD = 1024
_hashing = asyncio.Semaphore(4)
EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
# Failed sign-ins: at most this many per address in 15 minutes, and per client address in a minute.
LOGIN_TRIES_PER_ACCOUNT, LOGIN_TRIES_PER_IP = 10, 30


def _scrypt(password: str, salt: bytes, n: int, r: int, p: int) -> bytes:
    return hashlib.scrypt(password.encode(), salt=salt, n=n, r=r, p=p, maxmem=256 * 1024 * 1024, dklen=32)


def hash_password_sync(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = _scrypt(password, salt, SCRYPT_N, SCRYPT_R, SCRYPT_P)
    b64 = lambda b: base64.b64encode(b).decode()  # noqa: E731
    return f"scrypt${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}${b64(salt)}${b64(digest)}"


def verify_password_sync(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt, digest = stored.split("$")
        if scheme != "scrypt":
            return False
        got = _scrypt(password, base64.b64decode(salt), int(n), int(r), int(p))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(got, base64.b64decode(digest))


_DUMMY: str | None = None  # checked against when the address has no password, so a wrong address takes as long


def _dummy() -> str:
    global _DUMMY
    if _DUMMY is None:
        _DUMMY = hash_password_sync(secrets.token_urlsafe(16))
    return _DUMMY


async def hash_password(password: str) -> str:
    async with _hashing:
        return await asyncio.to_thread(hash_password_sync, password)


async def verify_password(password: str, stored: str | None) -> bool:
    async with _hashing:
        ok = await asyncio.to_thread(verify_password_sync, password, stored or _dummy())
    return ok and stored is not None


def password_problem(password: str, email: str | None, minimum: int) -> str | None:
    """Why a new password isn't acceptable, or None. Length is what counts (NIST SP 800-63B); no composition rules."""
    if len(password) < minimum:
        return f"use at least {minimum} characters"
    if len(password) > MAX_PASSWORD:
        return f"use at most {MAX_PASSWORD} characters"
    if email and password.strip().lower() in {email.lower(), email.split("@")[0].lower()}:
        return "don't use your email address as your password"
    return None


def secure_request(request: Request) -> bool:
    """Whether the browser reached the gateway over HTTPS (directly or through a proxy that says so)."""
    proto = request.headers.get("x-forwarded-proto", "").split(",")[0].strip().lower()
    return request.url.scheme == "https" or proto == "https"


def set_session_cookie(request: Request, response: Response, token: str, hours: int) -> None:
    response.set_cookie(SESSION_COOKIE, token, max_age=hours * 3600, httponly=True, samesite="strict",
                        secure=secure_request(request), path="/")


def clear_session_cookie(request: Request, response: Response) -> None:
    response.delete_cookie(SESSION_COOKIE, path="/", httponly=True, samesite="strict", secure=secure_request(request))


SSO_COOKIE = "mo_sso"  # ties the identity provider's answer to the browser that asked (SameSite=Lax: it comes back by redirect)
SSO_SECONDS = 600


def sso_ready(gateway) -> bool:
    o = gateway.cfg.auth.oidc
    return bool(o and o.browser_ready)


def public_base(request: Request) -> str:
    """The gateway's address as the browser sees it, behind a proxy too."""
    proto = request.headers.get("x-forwarded-proto", "").split(",")[0].strip() or request.url.scheme
    host = request.headers.get("x-forwarded-host", "").split(",")[0].strip() or request.headers.get("host") or request.url.netloc
    return f"{proto}://{host}"


def _b64url(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


class Login(BaseModel):
    email: str | None = None
    password: str | None = Field(default=None, max_length=MAX_PASSWORD)
    api_key: str | None = None


class SetPassword(BaseModel):
    password: str = Field(max_length=MAX_PASSWORD)
    name: str | None = Field(default=None, max_length=200)


class ChangePassword(BaseModel):
    current: str = Field(max_length=MAX_PASSWORD)
    new: str = Field(max_length=MAX_PASSWORD)


class NewInvite(BaseModel):
    email: str = Field(max_length=320)
    team: str | None = None
    role: Role = "user"


def _iso(d: datetime | None) -> str | None:
    return d.isoformat() if d else None


async def open_session(gateway, request: Request, response: Response, user_id: str, method: str,
                       key_hash: str | None = None) -> str:
    hours = gateway.cfg.auth.session_hours
    token = await gateway.store.create_session(user_id, method, datetime.now(UTC) + timedelta(hours=hours), key_hash=key_hash,
                                               user_agent=(request.headers.get("user-agent") or "")[:300] or None)
    set_session_cookie(request, response, token, hours)
    return token


def build_router(gateway) -> APIRouter:
    router = APIRouter(tags=["sign-in"])

    def local_cfg():
        return gateway.cfg.auth.local

    async def principal(request: Request, minimum: str = "user") -> Principal:
        try:
            p = await gateway.authn.authenticate_console(request)
        except AuthError as e:
            raise HTTPException(e.status, e.message)
        if not p.at_least(minimum) or (minimum == "team_lead" and p.role == "finance"):
            raise HTTPException(403, "not allowed for your role")
        return p

    def same_origin(request: Request) -> None:
        """/auth's own calls come only from the console: the same header and origin check as session requests."""
        try:
            check_same_origin({k.lower(): v for k, v in request.headers.items()})
        except AuthError as e:
            raise HTTPException(e.status, e.message)

    def require_local() -> None:
        if not local_cfg().enabled:
            raise HTTPException(403, "password sign-in is turned off on this gateway (auth.local.enabled); sign in with SSO")

    def require_db() -> None:
        if gateway.store.kind != "postgres":
            raise HTTPException(409, "accounts need Postgres (set MERITORDER_DATABASE_URL); in memory they'd be gone at the next restart")

    async def too_many(request: Request, email: str) -> None:
        ip = request.client.host if request.client else "unknown"
        if await gateway.state.rate_hit(f"login-ip:{ip}", 60, 0) >= LOGIN_TRIES_PER_IP or \
                await gateway.state.rate_hit(f"login:{email}", 900, 0) >= LOGIN_TRIES_PER_ACCOUNT:
            raise HTTPException(429, "too many failed sign-ins; wait a few minutes and try again", headers={"retry-after": "300"})

    async def failed(request: Request, email: str) -> None:
        ip = request.client.host if request.client else "unknown"
        await gateway.state.rate_hit(f"login-ip:{ip}", 60)
        await gateway.state.rate_hit(f"login:{email}", 900)

    async def find_local(email: str) -> dict | None:
        return await gateway.store.get_user(email) or await gateway.store.find_user_by_email(email)

    # ------------------------------------------------------------------ sign in and out

    @router.get("/auth/session")
    async def session(request: Request):
        """Whether this browser is signed in, answered with 200 either way, so a signed-out console logs no error. With a
        cookie that no longer works, why (expired, ended, disabled)."""
        try:
            p = await gateway.authn.authenticate_console(request)
        except AuthError as e:
            return {"signed_in": False, "reason": e.message if request.cookies.get(SESSION_COOKIE) else None}
        return {"signed_in": True, "user": p.user_id, "team": p.team, "role": p.role, "method": p.session or "key"}

    @router.post("/auth/login")
    async def login(request: Request, response: Response, body: Login):
        same_origin(request)
        if body.api_key:
            # A key typed into the console: checked like any key, then traded for a session tied to it. The browser
            # never stores the key, and revoking the key ends the session.
            try:
                p = await gateway.authn.authenticate({"authorization": f"Bearer {body.api_key.strip()}"})
            except AuthError as e:
                raise HTTPException(401, e.message)
            if not p.key_id:
                raise HTTPException(400, "sign in with a Merit Order key (mo-…) here, not an identity token")
            await open_session(gateway, request, response, p.user_id, "key", key_hash=hash_key(body.api_key.strip()))
            return {"user": p.user_id, "team": p.team, "role": p.role, "method": "key"}
        require_local()
        email = (body.email or "").strip().lower()
        if not email or not body.password:
            raise HTTPException(400, "enter your email address and password")
        await too_many(request, email)
        user = await find_local(email)
        stored = await gateway.store.get_password(user["id"]) if user else None
        if not await verify_password(body.password, stored) or not user:
            await failed(request, email)
            raise HTTPException(401, "wrong email address or password")
        if user.get("disabled_at"):
            raise HTTPException(403, "this account is disabled; contact your Merit Order admin")
        await open_session(gateway, request, response, user["id"], "password")
        return {"user": user["id"], "team": user["team"], "role": user["role"], "method": "password"}

    @router.post("/auth/logout")
    async def logout(request: Request, response: Response):
        same_origin(request)
        token = request.cookies.get(SESSION_COOKIE)
        if token:
            await gateway.store.revoke_session(hash_key(token))
            gateway.authn.forget_session(token)
        clear_session_cookie(request, response)
        return {"signed_out": True}

    @router.post("/auth/password")
    async def change_password(request: Request, body: ChangePassword):
        require_local()
        p = await principal(request)
        stored = await gateway.store.get_password(p.user_id)
        if stored is None:
            raise HTTPException(400, "you don't have a password here; you sign in with SSO or a key")
        if not await verify_password(body.current, stored):
            raise HTTPException(401, "your current password is wrong")
        user = await gateway.store.get_user(p.user_id) or {}
        if why := password_problem(body.new, user.get("email") or p.user_id, local_cfg().min_password_length):
            raise HTTPException(422, why)
        await gateway.store.set_password(p.user_id, await hash_password(body.new))
        token = request.cookies.get(SESSION_COOKIE)
        ended = await gateway.store.revoke_user_sessions(p.user_id, keep=hash_key(token) if token else None)
        gateway.authn.forget_user(p.user_id)
        return {"changed": True, "other_sessions_ended": ended}

    # ------------------------------------------------------------------ SSO in the browser

    def sso_failed(message: str) -> RedirectResponse:
        log.info("console SSO sign-in refused: %s", message)
        resp = RedirectResponse(f"/ui/#sso-error={quote(message)}", status_code=303)
        resp.delete_cookie(SSO_COOKIE, path="/auth/sso")
        return resp

    @router.get("/auth/sso/start")
    async def sso_start(request: Request):
        """To the identity provider's sign-in page. state (single use, ten minutes) carries the PKCE verifier and nonce; a
        cookie ties the answer to this browser, so a link someone else started can't sign you in as them."""
        o = gateway.cfg.auth.oidc
        if not sso_ready(gateway):
            return sso_failed("single sign-on isn't set up on this gateway")
        try:
            disc = await gateway.authn.oidc.discovery()
        except Exception as e:  # noqa: BLE001 (the provider's discovery document: unreachable, or not JSON)
            return sso_failed(f"couldn't reach the identity provider: {e}")
        state, nonce, verifier, binding = (secrets.token_urlsafe(32) for _ in range(4))
        redirect_uri = o.redirect_uri or f"{public_base(request)}/auth/sso/callback"
        await gateway.state.set_json(f"sso:{hash_key(state)}", {"verifier": verifier, "nonce": nonce, "binding": hash_key(binding),
                                                                "redirect_uri": redirect_uri}, ttl=SSO_SECONDS)
        query = {"response_type": "code", "client_id": o.browser_client_id or o.client_id, "redirect_uri": redirect_uri,
                 "scope": " ".join(sc for sc in o.scopes if sc != "offline_access"), "state": state, "nonce": nonce,
                 "code_challenge": _b64url(hashlib.sha256(verifier.encode()).digest()), "code_challenge_method": "S256"}
        if o.is_google and len(o.hosted_domains) == 1:
            query["hd"] = o.hosted_domains[0]  # a hint for Google's account chooser; the token's hd is still checked
        resp = RedirectResponse(f"{disc['authorization_endpoint']}?{urlencode(query)}", status_code=303)
        resp.set_cookie(SSO_COOKIE, binding, max_age=SSO_SECONDS, httponly=True, samesite="lax", secure=secure_request(request),
                        path="/auth/sso")
        return resp

    @router.get("/auth/sso/callback")
    async def sso_callback(request: Request, state: str = "", code: str = "", error: str = "", error_description: str = ""):
        o = gateway.cfg.auth.oidc
        if not sso_ready(gateway):
            return sso_failed("single sign-on isn't set up on this gateway")
        if error:
            return sso_failed(error_description or error)
        key = f"sso:{hash_key(state)}"
        pending = await gateway.state.get_json(key) if state else None
        if not pending or pending.get("used"):
            return sso_failed("that sign-in expired or was already used; try again")
        await gateway.state.set_json(key, {"used": True}, ttl=SSO_SECONDS)
        binding = request.cookies.get(SSO_COOKIE)
        if not binding or not hmac.compare_digest(hash_key(binding), pending["binding"]):
            return sso_failed("that sign-in was started in another browser; start it again here")
        if not code:
            return sso_failed("the identity provider sent no authorization code")
        client_id = o.browser_client_id or o.client_id
        form = {"grant_type": "authorization_code", "code": code, "redirect_uri": pending["redirect_uri"], "client_id": client_id,
                "code_verifier": pending["verifier"]}
        if o.browser_secret:
            form["client_secret"] = o.browser_secret
        try:
            disc = await gateway.authn.oidc.discovery()
            async with httpx.AsyncClient(timeout=20) as c:
                tok = (await c.post(disc["token_endpoint"], data=form, headers={"accept": "application/json"})).json()
        except Exception as e:  # noqa: BLE001
            return sso_failed(f"couldn't reach the identity provider: {e}")
        if "id_token" not in tok:
            return sso_failed(f"the identity provider refused: {tok.get('error_description') or tok.get('error') or 'no ID token'}")
        try:
            claims = await gateway.authn.oidc.verify(tok["id_token"], audience=client_id)
            if not hmac.compare_digest(str(claims.get("nonce") or ""), pending["nonce"]):
                raise AuthError(401, "the identity token wasn't issued for this sign-in (nonce)")
            principal, _ = await gateway.authn.sign_in(claims)
        except AuthError as e:
            return sso_failed(e.message)
        resp = RedirectResponse("/ui/", status_code=303)
        resp.delete_cookie(SSO_COOKIE, path="/auth/sso")
        await open_session(gateway, request, resp, principal.user_id, "sso")
        log.info("%s signed in to the console with SSO", user_id_of(claims))
        return resp

    # ------------------------------------------------------------------ invite and reset links

    async def live_invite(token: str) -> tuple[str, dict]:
        h = hash_key(token)
        inv = await gateway.store.find_invite(h)
        if not inv or inv.get("revoked_at"):
            raise HTTPException(404, "this link isn't valid; ask your Merit Order admin for a new one")
        if inv.get("used_at"):
            raise HTTPException(410, "this link has already been used; sign in, or ask your admin for a new one")
        if inv["expires_at"] <= datetime.now(UTC):
            raise HTTPException(410, "this link has expired; ask your Merit Order admin for a new one")
        return h, inv

    @router.get("/auth/invite/{token}")
    async def invite_info(token: str):
        require_local()
        _, inv = await live_invite(token)
        user = await gateway.store.get_user(inv["user_id"]) or {}
        return {"kind": inv["kind"], "email": user.get("email") or inv["user_id"], "team": user.get("team"), "role": user.get("role"),
                "expires_at": _iso(inv["expires_at"]), "min_password_length": local_cfg().min_password_length}

    @router.post("/auth/invite/{token}")
    async def accept_invite(token: str, request: Request, response: Response, body: SetPassword):
        same_origin(request)
        require_local()
        h, inv = await live_invite(token)
        user = await gateway.store.get_user(inv["user_id"])
        if not user:
            raise HTTPException(404, "this link's account no longer exists; ask your Merit Order admin")
        if user.get("disabled_at"):
            raise HTTPException(403, "this account is disabled; contact your Merit Order admin")
        if why := password_problem(body.password, user.get("email") or user["id"], local_cfg().min_password_length):
            raise HTTPException(422, why)
        hashed = await hash_password(body.password)
        if not await gateway.store.use_invite(h):  # someone used it a moment ago
            raise HTTPException(410, "this link has already been used; sign in, or ask your admin for a new one")
        await gateway.store.set_password(user["id"], hashed)
        if body.name and body.name.strip():
            await gateway.store.upsert_user(user["id"], user["team"], user["role"], display_name=body.name.strip(), source="ensure")
        await gateway.store.revoke_user_sessions(user["id"])  # a reset ends sessions opened with the old password
        gateway.authn.forget_user(user["id"])
        await open_session(gateway, request, response, user["id"], "password")
        return {"user": user["id"], "team": user["team"], "role": user["role"], "method": "password"}

    def may_manage(p: Principal, team: str, role: str) -> None:
        """The same rule as for keys: a team lead invites and resets only people on their own team with the user role."""
        if p.role == "admin":
            return
        if team != p.team or ROLE_RANK.get(role, 0) > ROLE_RANK["user"]:
            raise HTTPException(403, f"that person is on team '{team}' with role '{role}'; only an admin can do this for them")

    def invite_out(inv: dict, token: str | None = None) -> dict:
        out = {"id": inv["id"], "user": inv["user_id"], "kind": inv["kind"], "team": inv.get("team"), "role": inv.get("role"),
               "created_by": inv.get("created_by"), "created_at": _iso(inv.get("created_at")), "expires_at": _iso(inv["expires_at"])}
        if token:
            out |= {"token": token, "path": f"/ui/#invite={token}",
                    "note": "Shown once. Send the link to them; it works once, until it expires."}
        return out

    @router.get("/v1/admin/invites")
    async def list_invites(request: Request):
        p = await principal(request, "team_lead")
        rows = await gateway.store.list_invites(None if p.role == "admin" else p.team)
        return {"invites": [invite_out(r) for r in rows]}

    @router.post("/v1/admin/invites", status_code=201)
    async def create_invite(request: Request, body: NewInvite):
        p = await principal(request, "team_lead")
        require_local()
        require_db()
        email = body.email.strip().lower()
        if not EMAIL.match(email):
            raise HTTPException(422, "that isn't an email address")
        team = body.team or p.team
        existing = await gateway.store.get_user(email)
        if existing:
            may_manage(p, existing["team"], existing["role"])
            team, role = existing["team"], existing["role"]
        else:
            may_manage(p, team, body.role)
            role = body.role
            await gateway.store.upsert_user(email, team, role, email=email, source="ensure")
            await gateway.store.update_user(email, team=team, role=role)  # set by a person here, so sign-in mappings don't move it
        await gateway.store.revoke_user_invites(email)  # one open link per person
        expires = datetime.now(UTC) + timedelta(hours=local_cfg().invite_hours)
        kind = "reset" if existing and await gateway.store.get_password(email) else "invite"
        iid, token = await gateway.store.create_invite(email, kind, expires, team=team, role=role, created_by=p.user_id)
        log.info("%s created a %s link for %s", p.user_id, kind, email)
        return invite_out({"id": iid, "user_id": email, "kind": kind, "team": team, "role": role, "created_by": p.user_id,
                           "created_at": datetime.now(UTC), "expires_at": expires}, token)

    @router.delete("/v1/admin/invites/{invite_id}")
    async def withdraw_invite(request: Request, invite_id: str):
        p = await principal(request, "team_lead")
        rows = {r["id"]: r for r in await gateway.store.list_invites(None if p.role == "admin" else p.team)}
        if invite_id not in rows:
            raise HTTPException(404, "no such open link")
        may_manage(p, rows[invite_id].get("team") or "", rows[invite_id].get("role") or "user")
        await gateway.store.revoke_invite(invite_id)
        return {"withdrawn": True}

    @router.post("/v1/admin/users/{user_id}/reset", status_code=201)
    async def reset_link(request: Request, user_id: str):
        p = await principal(request, "team_lead")
        require_local()
        require_db()
        user = await gateway.store.get_user(user_id)
        if not user:
            raise HTTPException(404, "no such person")
        may_manage(p, user["team"], user["role"])
        await gateway.store.revoke_user_invites(user_id)
        expires = datetime.now(UTC) + timedelta(hours=local_cfg().invite_hours)
        kind = "reset" if await gateway.store.get_password(user_id) else "invite"
        iid, token = await gateway.store.create_invite(user_id, kind, expires, team=user["team"], role=user["role"], created_by=p.user_id)
        log.info("%s created a %s link for %s", p.user_id, kind, user_id)
        return invite_out({"id": iid, "user_id": user_id, "kind": kind, "team": user["team"], "role": user["role"],
                           "created_by": p.user_id, "created_at": datetime.now(UTC), "expires_at": expires}, token)

    return router


async def sign_in_status(gateway, p: Principal) -> dict:
    """For /v1/reports/me: how this person signed in, and whether they have a password here."""
    return {"method": p.session or ("key" if p.key_id else None),
            "has_password": await gateway.store.get_password(p.user_id) is not None}


async def password_key(gateway, request: Request, email: str, password: str) -> dict:
    """`meritorder login` on a gateway with local accounts: the password signs in, and the person gets a short-lived key
    (``auth.oidc.key_ttl_days``, or 30 days) for their harnesses, as with SSO's device login."""
    if not gateway.cfg.auth.local.enabled:
        raise HTTPException(403, "password sign-in is turned off on this gateway (auth.local.enabled); use `meritorder login` with SSO")
    email = email.strip().lower()
    ip = request.client.host if request.client else "unknown"
    if await gateway.state.rate_hit(f"login-ip:{ip}", 60, 0) >= LOGIN_TRIES_PER_IP or \
            await gateway.state.rate_hit(f"login:{email}", 900, 0) >= LOGIN_TRIES_PER_ACCOUNT:
        raise HTTPException(429, "too many failed sign-ins; wait a few minutes and try again", headers={"retry-after": "300"})
    user = await gateway.store.get_user(email) or await gateway.store.find_user_by_email(email)
    stored = await gateway.store.get_password(user["id"]) if user else None
    if not await verify_password(password, stored) or not user:
        await gateway.state.rate_hit(f"login-ip:{ip}", 60)
        await gateway.state.rate_hit(f"login:{email}", 900)
        raise HTTPException(401, "wrong email address or password")
    if user.get("disabled_at"):
        raise HTTPException(403, "this account is disabled; contact your Merit Order admin")
    days = gateway.cfg.auth.oidc.key_ttl_days if gateway.cfg.auth.oidc else 30
    expires = datetime.now(UTC) + timedelta(days=days)
    key_id, secret = await gateway.store.create_key(user["id"], user["team"], user["role"], name="password login", expires_at=expires)
    return {"api_key": secret, "key_id": key_id, "expires_at": expires.isoformat(), "user": user.get("email") or user["id"], "team": user["team"]}
