"""Console sign-in (decision 69): local accounts, browser sessions and invite links.

People sign in to the console with an email address and password (local accounts, for teams without an identity
provider) or with their organization's SSO (OIDC), and get a session cookie. Keys are for harnesses, CI and services;
the console still accepts one, but exchanges it for a session instead of keeping it in the browser.

* GET  /auth/session                     whether this browser is signed in (200 either way)
* POST /auth/login                       email and password, or an API key: opens a session (HttpOnly cookie)
* POST /auth/logout                      ends this session
* POST /auth/password                    change your own password; your other sessions end
* POST /auth/two-factor/setup            start turning on two-factor: a new authenticator secret (QR link)
* POST /auth/two-factor/enable           confirm with a code: two-factor on, ten recovery codes shown once
* POST /auth/two-factor/disable          turn it off (your password)
* POST /v1/admin/users/{user_id}/two-factor/reset   an admin turns it off for someone who lost their device
* GET  /auth/sso/start                   to the organization's sign-in page (OIDC authorization code with PKCE)
* GET  /auth/sso/callback                back from it: a session, then the console
* POST /auth/exchange (email, password)  a short-lived key for `meritorder login` on a gateway with local accounts
* POST /auth/invite/info                 who an invite or password-reset link is for (the token in the body, never the URL)
* POST /auth/invite/accept               set a password with it, and sign in
* GET  /v1/admin/invites                 open invite and reset links (admin: all; team lead: their team's)
* POST /v1/admin/invites                 invite someone (a one-time link, shown once)
* DELETE /v1/admin/invites/{id}          withdraw one
* POST /v1/admin/users/{user_id}/reset   a password-reset link for someone

Passwords are stored as scrypt hashes (no extra dependency), session cookies and links only as SHA-256 hashes.

Passwords are only for accounts invited by email (their id is the address). People who sign in with SSO or have only a
key never get one, so leaving the identity provider or losing a key ends their access. A link a team lead makes sets a
password that acts at most as a user: promoting the person later doesn't promote whoever chose that password.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import logging
import re
import secrets
import weakref
from datetime import UTC, datetime, timedelta

from urllib.parse import quote, urlencode

import httpx
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from .auth import (ROLE_RANK, SECURE_SESSION_COOKIE, SESSION_COOKIE, AuthError, Principal, check_same_origin, lower_role,
                   secure_request, session_cookie_name, session_token, user_id_of)
from .config import Role
from . import twofactor
from .store import hash_key

log = logging.getLogger("meritorder.accounts")

# scrypt at OWASP's recommended cost (N=2^17, r=8, p=1: 128 MiB and about a quarter second per hash). A few at a time,
# so a burst of sign-ins can't take the gateway's memory.
SCRYPT_N, SCRYPT_R, SCRYPT_P = 2**17, 8, 1
MAX_PASSWORD = 1024
_hashing: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()  # four hashing slots per event loop
MAX_WAITING = 64  # sign-ins waiting for a hashing slot; past this the gateway answers 503 rather than queue without end
_waiting = 0
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


def verify_password_sync(password: str, stored: str | None) -> bool:
    if stored is None:
        verify_password_sync(password, _dummy())  # an unknown address takes as long as a wrong password
        return False
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


async def _hash_slot(fn, *args):
    """Run a hash in a thread, at most four at a time; refuse rather than queue without end."""
    global _waiting
    if _waiting >= MAX_WAITING:
        raise HTTPException(503, "too many sign-ins at once; try again in a moment", headers={"retry-after": "5"})
    _waiting += 1
    loop = asyncio.get_running_loop()
    slots = _hashing.get(loop) or _hashing.setdefault(loop, asyncio.Semaphore(4))
    try:
        async with slots:
            return await asyncio.to_thread(fn, *args)
    finally:
        _waiting -= 1


async def hash_password(password: str) -> str:
    return await _hash_slot(hash_password_sync, password)


async def verify_password(password: str, stored: str | None) -> bool:
    return await _hash_slot(verify_password_sync, password, stored)


def local_account(user_id: str) -> bool:
    """An account that may have a password: one invited by email, whose id is the address. SSO people (an IdP's id) and
    key-only people (a name from the config file) never get one."""
    return bool(EMAIL.match(user_id or ""))


def password_problem(password: str, email: str | None, minimum: int) -> str | None:
    """Why a new password isn't acceptable, or None. Length is what counts (NIST SP 800-63B); no composition rules."""
    if len(password) < minimum:
        return f"use at least {minimum} characters"
    if len(password) > MAX_PASSWORD:
        return f"use at most {MAX_PASSWORD} characters"
    if email and password.strip().lower() in {email.lower(), email.split("@")[0].lower()}:
        return "don't use your email address as your password"
    return None


def set_session_cookie(request: Request, response: Response, token: str, hours: int) -> None:
    response.set_cookie(session_cookie_name(request), token, max_age=hours * 3600, httponly=True, samesite="strict",
                        secure=secure_request(request), path="/")


def clear_session_cookie(request: Request, response: Response) -> None:
    for name in (SESSION_COOKIE, SECURE_SESSION_COOKIE):
        response.delete_cookie(name, path="/", httponly=True, samesite="strict", secure=name == SECURE_SESSION_COOKIE)


def sso_code(value: str) -> str:
    """An OAuth error code as it may be shown (``access_denied``): letters and underscores only. The provider's free text
    never reaches the console, so a crafted link can't put words of its choosing on the sign-in page."""
    return re.sub(r"[^a-z_]", "", (value or "").lower())[:40] or "unknown_error"


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
    code: str | None = Field(default=None, max_length=40)  # an authenticator or recovery code, with two-factor on


class TwoFactorCode(BaseModel):
    code: str = Field(max_length=40)


class PasswordOnly(BaseModel):
    password: str = Field(max_length=MAX_PASSWORD)


class InviteToken(BaseModel):
    token: str = Field(max_length=200)


class SetPassword(InviteToken):
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
                       key_hash: str | None = None, max_role: str | None = None) -> str:
    hours = gateway.cfg.auth.session_hours
    token = await gateway.store.create_session(user_id, method, datetime.now(UTC) + timedelta(hours=hours), key_hash=key_hash,
                                               user_agent=(request.headers.get("user-agent") or "")[:300] or None, max_role=max_role)
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

    # ------------------------------------------------------------------ sign in and out

    @router.get("/auth/session")
    async def session(request: Request):
        """Whether this browser is signed in, answered with 200 either way, so a signed-out console logs no error. With a
        cookie that no longer works, why (expired, ended, disabled)."""
        try:
            p = await gateway.authn.authenticate_console(request)
        except AuthError as e:
            return {"signed_in": False, "reason": e.message if session_token(request) else None}
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
        if not (body.email or "").strip() or not body.password:
            raise HTTPException(400, "enter your email address and password")
        user, cap = await password_sign_in(gateway, request, body.email or "", body.password, body.code)
        await open_session(gateway, request, response, user["id"], "password", max_role=cap)
        return {"user": user["id"], "team": user["team"], "role": user["role"], "method": "password"}

    @router.post("/auth/logout")
    async def logout(request: Request, response: Response):
        same_origin(request)
        token = session_token(request)
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
        await gateway.store.set_password(p.user_id, await hash_password(body.new))  # chosen by the person: no cap
        token = session_token(request)
        ended = await gateway.store.revoke_user_sessions(p.user_id, keep=hash_key(token) if token else None)
        await revoke_password_keys(gateway, p.user_id)
        gateway.authn.forget_user(p.user_id)
        return {"changed": True, "other_sessions_ended": ended}

    # ------------------------------------------------------------------ two-factor (decision 73)

    async def own_credential(request: Request) -> tuple[Principal, dict]:
        require_local()
        p = await principal(request)
        cred = await gateway.store.get_credential(p.user_id)
        if cred is None:
            raise HTTPException(400, "two-factor is for accounts with a password here; SSO accounts use your identity provider's")
        return p, cred

    @router.post("/auth/two-factor/setup")
    async def two_factor_setup(request: Request):
        same_origin(request)
        p, cred = await own_credential(request)
        if cred.get("totp_enabled_at"):
            raise HTTPException(409, "two-factor is already on; turn it off first to set up another device")
        secret = twofactor.new_secret()
        await gateway.store.set_totp(p.user_id, twofactor.seal(secret, p.user_id, gateway.store.kind == "memory"), enabled=False)
        user = await gateway.store.get_user(p.user_id) or {}
        return {"secret": secret, "uri": twofactor.uri(secret, user.get("email") or p.user_id)}

    @router.post("/auth/two-factor/enable")
    async def two_factor_enable(request: Request, body: TwoFactorCode):
        same_origin(request)
        p, cred = await own_credential(request)
        if cred.get("totp_enabled_at") or not cred.get("totp_secret"):
            raise HTTPException(409, "start with setup" if not cred.get("totp_secret") else "two-factor is already on")
        secret = twofactor.unseal(cred["totp_secret"], p.user_id, gateway.store.kind == "memory")
        if twofactor.matching_step(secret, body.code) is None:
            raise HTTPException(422, "that code isn't right; check your device's time and try the newest code")
        codes = twofactor.new_recovery_codes()
        await gateway.store.set_totp(p.user_id, cred["totp_secret"], enabled=True, recovery_codes=[twofactor.hash_recovery(c) for c in codes])
        token = session_token(request)
        await gateway.store.revoke_user_sessions(p.user_id, keep=hash_key(token) if token else None)  # opened without it
        gateway.authn.forget_user(p.user_id)
        log.info("%s turned on two-factor sign-in", p.user_id)
        return {"enabled": True, "recovery_codes": codes,
                "note": "Shown once. Keep them somewhere safe: each signs you in once if you lose your device."}

    @router.post("/auth/two-factor/disable")
    async def two_factor_disable(request: Request, body: PasswordOnly):
        same_origin(request)
        p, cred = await own_credential(request)
        if not await verify_password(body.password, cred["password_hash"]):
            raise HTTPException(401, "your password is wrong")
        await gateway.store.set_totp(p.user_id, None, enabled=False)
        log.info("%s turned off two-factor sign-in", p.user_id)
        return {"enabled": False}

    @router.post("/v1/admin/users/{user_id}/two-factor/reset")
    async def two_factor_reset(request: Request, user_id: str):
        p = await principal(request, "admin")
        if await gateway.store.get_credential(user_id) is None:
            raise HTTPException(404, "no one with a password here by that id")
        await gateway.store.set_totp(user_id, None, enabled=False)
        await gateway.store.revoke_user_sessions(user_id)
        gateway.authn.forget_user(user_id)
        log.info("%s turned off two-factor sign-in for %s", p.user_id, user_id)
        return {"reset": True, "note": "They sign in with their password alone now, and can turn two-factor on again."}

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
            log.warning("console SSO: the identity provider's discovery document: %s", e)
            return sso_failed("couldn't reach the identity provider")
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
            return sso_failed(f"the identity provider refused the sign-in ({sso_code(error)})")
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
            log.warning("console SSO: the identity provider's token endpoint: %s", e)
            return sso_failed("couldn't reach the identity provider")
        if "id_token" not in tok:
            log.warning("console SSO: the token endpoint answered without an ID token: %s %s", tok.get("error"), tok.get("error_description"))
            return sso_failed(f"the identity provider refused the sign-in ({sso_code(tok.get('error') or 'no_id_token')})")
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

    @router.post("/auth/invite/info")
    async def invite_info(request: Request, body: InviteToken):
        same_origin(request)
        require_local()
        _, inv = await live_invite(body.token)
        user = await gateway.store.get_user(inv["user_id"]) or {}
        return {"kind": inv["kind"], "email": user.get("email") or inv["user_id"], "team": user.get("team"), "role": user.get("role"),
                "expires_at": _iso(inv["expires_at"]), "min_password_length": local_cfg().min_password_length}

    @router.post("/auth/invite/accept")
    async def accept_invite(request: Request, response: Response, body: SetPassword):
        same_origin(request)
        require_local()
        h, inv = await live_invite(body.token)
        user = await gateway.store.get_user(inv["user_id"])
        if not user or not local_account(user["id"]):
            raise HTTPException(404, "this link's account no longer exists; ask your Merit Order admin")
        if user.get("disabled_at"):
            raise HTTPException(403, "this account is disabled; contact your Merit Order admin")
        # A team lead's link sets a password for someone on their team with the user role, and only while that's still so;
        # the password then acts at most as a user, whatever the person becomes later.
        cap = None
        if inv.get("creator_role") != "admin":
            if user["team"] != inv.get("team") or ROLE_RANK.get(user["role"], 0) > ROLE_RANK["user"]:
                raise HTTPException(403, "this link was made by a team lead, and this account's team or role has changed since; "
                                         "ask an admin for a new link")
            cap = "user"
        if why := password_problem(body.password, user.get("email") or user["id"], local_cfg().min_password_length):
            raise HTTPException(422, why)
        hashed = await hash_password(body.password)
        if not await gateway.store.use_invite(h):  # someone used it a moment ago
            raise HTTPException(410, "this link has already been used; sign in, or ask your admin for a new one")
        await gateway.store.set_password(user["id"], hashed, max_role=cap)
        if body.name and body.name.strip():
            await gateway.store.upsert_user(user["id"], user["team"], user["role"], display_name=body.name.strip(), source="ensure")
        # A reset ends what the old password opened: sessions, and keys from `meritorder login`.
        await gateway.store.revoke_user_sessions(user["id"])
        await revoke_password_keys(gateway, user["id"])
        gateway.authn.forget_user(user["id"])
        await open_session(gateway, request, response, user["id"], "password", max_role=cap)
        role = lower_role(user["role"], cap) if cap else user["role"]
        return {"user": user["id"], "team": user["team"], "role": role, "method": "password"}

    def may_manage(p: Principal, team: str, role: str) -> None:
        """The same rule as for keys: a team lead invites and resets only people on their own team with the user role. The
        refusal doesn't say the person's team or role, so a lead can't learn them for people outside their team."""
        if p.role == "admin":
            return
        if team != p.team or ROLE_RANK.get(role, 0) > ROLE_RANK["user"]:
            raise HTTPException(403, "only an admin can do this for them")

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
        other = await gateway.store.find_user_by_email(email)
        if other and other["id"] != email:  # someone who signs in with SSO (or has a key) already has this address
            if p.role != "admin" and other["team"] != p.team:
                raise HTTPException(403, "only an admin can do this for them")
            raise HTTPException(409, "someone with this address already signs in with SSO or a key; they don't need a password here")
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
        iid, token = await gateway.store.create_invite(email, kind, expires, team=team, role=role, created_by=p.user_id,
                                                       creator_role=p.role)
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
        if not local_account(user_id):
            raise HTTPException(409, "they sign in with SSO or a key; passwords are only for people invited by email address")
        await gateway.store.revoke_user_invites(user_id)
        expires = datetime.now(UTC) + timedelta(hours=local_cfg().invite_hours)
        kind = "reset" if await gateway.store.get_password(user_id) else "invite"
        iid, token = await gateway.store.create_invite(user_id, kind, expires, team=user["team"], role=user["role"], created_by=p.user_id,
                                                       creator_role=p.role)
        log.info("%s created a %s link for %s", p.user_id, kind, user_id)
        return invite_out({"id": iid, "user_id": user_id, "kind": kind, "team": user["team"], "role": user["role"],
                           "created_by": p.user_id, "created_at": datetime.now(UTC), "expires_at": expires}, token)

    return router


async def sign_in_status(gateway, p: Principal) -> dict:
    """For /v1/reports/me: how this person signed in, and whether they have a password here."""
    cred = await gateway.store.get_credential(p.user_id)
    return {"method": p.session or ("key" if p.key_id else None), "has_password": cred is not None,
            "two_factor": bool(cred and cred.get("totp_enabled_at"))}


async def revoke_password_keys(gateway, user_id: str) -> None:
    """The keys `meritorder login` got with a password: ended when the password is reset or changed, so whoever knew the
    old one keeps nothing."""
    for k in await gateway.store.list_keys(user_id=user_id):
        if k.get("name") == "password login" and not k.get("revoked_at") and await gateway.store.revoke_key(k["id"]):
            gateway.authn.forget(k["id"])


TWO_FACTOR_HEADER = "x-meritorder-two-factor"  # "required" on the 401 that asks for a code


async def second_factor_ok(gateway, user_id: str, cred: dict, code: str) -> bool:
    """An authenticator code (each 30-second step's once per person) or an unused recovery code."""
    if twofactor.looks_like_recovery(code):
        return await gateway.store.use_recovery_code(user_id, twofactor.hash_recovery(code))
    try:
        secret = twofactor.unseal(cred["totp_secret"], user_id, gateway.store.kind == "memory")
    except Exception as e:  # noqa: BLE001 (a key that changed: they need an admin's reset or a recovery code)
        log.warning("couldn't open %s's authenticator secret: %s", user_id, e)
        return False
    step = twofactor.matching_step(secret, code)
    if step is None:
        return False
    mine = secrets.token_hex(8)
    held = await gateway.state.set_json_once({f"totp-used:{user_id}:{step}": mine}, ttl=120)
    return held.get(f"totp-used:{user_id}:{step}") == mine  # a code seen once already is refused


async def password_sign_in(gateway, request: Request, email: str, password: str, code: str | None = None) -> tuple[dict, str | None]:
    """An email address and password, checked: the person's record and the password's role cap (or None). Each attempt is
    counted before the password is checked, so attempts sent at once can't all slip under the limit; a right password
    takes its own attempt back off."""
    if not gateway.cfg.auth.local.enabled:
        raise HTTPException(403, "password sign-in is turned off on this gateway (auth.local.enabled); sign in with SSO")
    email = email.strip().lower()
    ip = request.client.host if request.client else "unknown"
    by_account = await gateway.state.rate_hit(f"login:{email}", 900)
    by_ip = await gateway.state.rate_hit(f"login-ip:{ip}", 60)
    if by_account > LOGIN_TRIES_PER_ACCOUNT or by_ip > LOGIN_TRIES_PER_IP:
        raise HTTPException(429, "too many failed sign-ins; wait a few minutes and try again", headers={"retry-after": "300"})
    user = await gateway.store.get_user(email) if local_account(email) else None
    cred = await gateway.store.get_credential(user["id"]) if user else None
    if not await verify_password(password, cred["password_hash"] if cred else None) or not user:
        raise HTTPException(401, "wrong email address or password")
    await gateway.state.rate_hit(f"login:{email}", 900, -1)
    await gateway.state.rate_hit(f"login-ip:{ip}", 60, -1)
    if user.get("disabled_at"):
        raise HTTPException(403, "this account is disabled; contact your Merit Order admin")
    cap = cred.get("max_role")
    if cap and ROLE_RANK.get(user["role"], 0) > ROLE_RANK.get(cap, 0):
        raise HTTPException(403, "your role has changed since a team lead's link set this password; ask an admin for a new "
                                 "password link")
    if cred.get("totp_enabled_at"):
        if not (code or "").strip():
            raise HTTPException(401, "enter the code from your authenticator app", headers={TWO_FACTOR_HEADER: "required"})
        if not await second_factor_ok(gateway, user["id"], cred, code):
            await gateway.state.rate_hit(f"login:{email}", 900)  # a wrong code counts like a wrong password
            await gateway.state.rate_hit(f"login-ip:{ip}", 60)
            raise HTTPException(401, "that code isn't right (or was already used)", headers={TWO_FACTOR_HEADER: "required"})
    return user, cap


async def password_key(gateway, request: Request, email: str, password: str, code: str | None = None) -> dict:
    """`meritorder login` on a gateway with local accounts: the password signs in, and the person gets a short-lived key
    (``auth.oidc.key_ttl_days``, or 30 days) for their harnesses, as with SSO's device login."""
    user, cap = await password_sign_in(gateway, request, email, password, code)
    role = lower_role(user["role"], cap) if cap else user["role"]
    days = gateway.cfg.auth.oidc.key_ttl_days if gateway.cfg.auth.oidc else 30
    expires = datetime.now(UTC) + timedelta(days=days)
    key_id, secret = await gateway.store.create_key(user["id"], user["team"], role, name="password login", expires_at=expires)
    return {"api_key": secret, "key_id": key_id, "expires_at": expires.isoformat(), "user": user.get("email") or user["id"], "team": user["team"]}
