"""Google Groups mapped to teams and roles, read from the Cloud Identity Groups API (docs/06-auth-and-spend.md#google-groups).

Google's ID tokens carry no groups, so the gateway asks Google which groups a person is in: at sign-in, and again every
``refresh_seconds`` while they use the gateway, so a removal takes effect as an Entra group change does (the user record
changes, and every gateway's key cache follows within a minute). Followed: Cloud Identity API v1, November 2026.

* **Who calls.** A service account holding the Groups Reader admin role (Admin console → Account → Admin roles), with
  the read-only scope ``cloud-identity.groups.readonly``. No domain-wide delegation: the account acts as itself.
* **Which call.** ``searchTransitiveGroups`` lists every group a person is in, nested ones included, in one call (a page
  per 1,000 groups). Google offers it on Workspace Enterprise and Cloud Identity Premium only and answers 403 elsewhere;
  then, if ``searchDirectGroups`` works, the gateway uses it and follows the nesting itself, one call per group, each
  group's parents cached for a refresh period. A 403 from both is a permission problem, and the error says which.
* **Credentials.** A service-account key: a file (re-read when it changes, as a mounted Secret does) or the JSON in an
  environment variable, signed into a token with PyJWT. Without one, the metadata server's identity (GKE Workload
  Identity, Compute Engine, Cloud Run). Keys and tokens are never logged.
* **Cache.** Each person's last reading lives in the shared state (Redis), with the team and role it gave and which
  mapping gave them, for the People page. If Google can't be reached, the last known groups stand for
  ``keep_last_known_hours``; after that the person's team and role follow the email and domain rules alone. Failures
  are logged and raise an alert on the console (and the webhook), as do the fallback and the recovery.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from collections.abc import Callable

import httpx

from . import alerts
from .auth import Assignment, assign
from .config import GoogleGroupsConfig, OIDCConfig, read_secret_file

log = logging.getLogger("meritorder.google_groups")

SCOPE = "https://www.googleapis.com/auth/cloud-identity.groups.readonly"
SEARCH = "https://cloudidentity.googleapis.com/v1/groups/-/memberships:"
TOKEN_URI = "https://oauth2.googleapis.com/token"
JWT_BEARER = "urn:ietf:params:oauth:grant-type:jwt-bearer"
EVERY_GROUP = "cloudidentity.googleapis.com/groups.discussion_forum"  # Google puts this label on every Google group
SECURITY = "cloudidentity.googleapis.com/groups.security"
RECORDS = "google_groups"        # state hash: person -> their last reading and what it gave
SEEN = "google_groups_seen"      # state hash: person -> when they last used the gateway
OUTAGE = "google_groups:outage"  # set while an outage alert is open, so the recovery is announced once
ROUND = "google_groups:round"    # held by the gateway doing this refresh round
ALERT_EVERY = 3600
TICK_SECONDS = 10
CONCURRENCY = 10                 # lookups in flight at once during a refresh round


class GroupsError(Exception):
    """A lookup that didn't work. ``outage``: Google was unreachable or overloaded, as opposed to a setup problem."""

    def __init__(self, message: str, status: int | None = None, outage: bool = False):
        super().__init__(message)
        self.status = status
        self.outage = outage


def _cel(value: str) -> str:
    """A CEL string literal (an address may hold an apostrophe: o'brien@example.com)."""
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


class Credentials:
    """An access token for the Cloud Identity API: from a service-account key, else from the metadata server."""

    def __init__(self, cfg: GoogleGroupsConfig):
        self.cfg = cfg
        self.account: str | None = None  # the service account's address, when the key names it (for error messages)
        self._raw: str | None = None
        self._key: dict | None = None
        self._token: str | None = None
        self._expires = 0.0

    def where(self) -> str:
        places = [p for p in (self.cfg.credentials_file, f"${self.cfg.credentials_env}" if self.cfg.credentials_env else None) if p]
        return " or ".join(places) or "no key configured"

    def key(self) -> dict | None:
        """The service-account key, parsed; None when none is configured or the file and variable are empty."""
        raw = (read_secret_file(self.cfg.credentials_file) if self.cfg.credentials_file else None) or (
            (os.environ.get(self.cfg.credentials_env) or "").strip() if self.cfg.credentials_env else None) or None
        if raw != self._raw:  # a rotated key: parse it again and drop the token the old one got
            # Remembered only once it parses: a key that can't be used fails every time, rather than once and then leaving
            # no key, which would send the gateway to the metadata server's identity instead.
            self._raw, self._key, self._token = None, None, None
            if raw:
                try:
                    key = json.loads(raw)
                except ValueError:
                    raise GroupsError(f"the service-account key in {self.where()} isn't JSON") from None
                if not isinstance(key, dict) or key.get("type") != "service_account" or not key.get("private_key") or not key.get("client_email"):
                    raise GroupsError(f"{self.where()} doesn't hold a service-account key (a JSON key with type service_account)")
                self._key = key
                self.account = key["client_email"]
            self._raw = raw
        return self._key

    def forget(self) -> None:
        self._token = None

    async def token(self, client: httpx.AsyncClient) -> str:
        key = self.key()
        if self._token and time.time() < self._expires - 300:
            return self._token
        token, lifetime = await (self._from_key(client, key) if key else self._from_metadata(client))
        self._token, self._expires = token, time.time() + lifetime
        return token

    async def _from_key(self, client: httpx.AsyncClient, key: dict) -> tuple[str, float]:
        import jwt

        uri = key.get("token_uri") or TOKEN_URI
        now = int(time.time())
        try:
            assertion = jwt.encode({"iss": key["client_email"], "scope": SCOPE, "aud": uri, "iat": now, "exp": now + 3600},
                                   key["private_key"], algorithm="RS256",
                                   headers={"kid": key["private_key_id"]} if key.get("private_key_id") else None)
        except Exception:  # noqa: BLE001 (a damaged PEM raises several kinds; the key itself must never reach a message)
            raise GroupsError(f"the private_key in {self.where()} can't be read") from None
        try:
            r = await client.post(uri, data={"grant_type": JWT_BEARER, "assertion": assertion})
        except httpx.HTTPError as e:
            raise GroupsError(f"couldn't reach Google's token endpoint ({type(e).__name__})", outage=True) from None
        body = _json(r)
        if r.status_code != 200 or not body.get("access_token"):
            why = ": ".join(str(body[k]) for k in ("error", "error_description") if body.get(k)) or r.reason_phrase
            raise GroupsError(f"Google refused the service account's key ({r.status_code} {why})"[:300], r.status_code,
                              outage=r.status_code == 429 or r.status_code >= 500)
        return body["access_token"], float(body.get("expires_in") or 3600)

    async def _from_metadata(self, client: httpx.AsyncClient) -> tuple[str, float]:
        host = os.environ.get("GCE_METADATA_HOST") or "metadata.google.internal"
        url = f"http://{host}/computeMetadata/v1/instance/service-accounts/default/token"
        try:
            r = await client.get(url, params={"scopes": SCOPE}, headers={"Metadata-Flavor": "Google"})
        except httpx.HTTPError as e:
            raise GroupsError(f"no service-account key ({self.where()}) and no metadata server to get an identity from "
                              f"(GKE Workload Identity, Compute Engine, Cloud Run): {type(e).__name__}") from None
        body = _json(r)
        if r.status_code != 200 or not body.get("access_token"):
            raise GroupsError(f"the metadata server gave no token ({r.status_code}): is a service account attached to this "
                              f"workload?", r.status_code, outage=r.status_code >= 500)
        return body["access_token"], float(body.get("expires_in") or 3600)


def _json(r: httpx.Response) -> dict:
    try:
        body = r.json()
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}


class GroupsClient:
    """Which groups an address is in, nested ones included, from the Cloud Identity Groups API."""

    def __init__(self, cfg: GoogleGroupsConfig, transport: httpx.AsyncBaseTransport | None = None):
        self.cfg = cfg
        self.credentials = Credentials(cfg)
        self.transport = transport  # tests put a MockTransport here
        self.transitive: bool | None = None  # None: not known yet; False: Google refused it but direct search works
        self._parents: dict[str, tuple[float, list[dict]]] = {}

    def http(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=10, transport=self.transport)

    async def groups_of(self, address: str, client: httpx.AsyncClient) -> list[dict]:
        """``[{"email", "name", "direct"}]``, sorted by address."""
        refused: GroupsError | None = None
        if self.transitive is not False:
            try:
                found = await self._search(client, "searchTransitiveGroups", address)
            except GroupsError as e:
                if e.status != 403 or self.transitive:
                    raise
                refused = e
            else:
                self.transitive = True
                return sorted(({"email": g["email"], "name": g["name"], "direct": g["relation"] in ("DIRECT", "DIRECT_AND_INDIRECT")}
                               for g in found), key=lambda g: g["email"])
        direct = await self._search(client, "searchDirectGroups", address)  # a 403 here too: a permission problem
        if self.transitive is None:
            self.transitive = False
            log.info("Google refused transitive group search (%s), but direct search works: following nested groups one "
                     "call per group (Workspace editions below Enterprise don't offer transitive search)", refused)
        return await self._expand(client, direct)

    async def _expand(self, client: httpx.AsyncClient, direct: list[dict]) -> list[dict]:
        found = {g["email"]: {"email": g["email"], "name": g["name"], "direct": True} for g in direct}
        queue = list(found)
        while queue:
            for parent in await self._parents_of(client, queue.pop()):
                if parent["email"] not in found:
                    found[parent["email"]] = {"email": parent["email"], "name": parent["name"], "direct": False}
                    queue.append(parent["email"])
        return sorted(found.values(), key=lambda g: g["email"])

    async def _parents_of(self, client: httpx.AsyncClient, group: str) -> list[dict]:
        hit = self._parents.get(group)
        if hit and hit[0] > time.monotonic():
            return hit[1]
        try:
            parents = await self._search(client, "searchDirectGroups", group)
        except GroupsError as e:
            if e.status not in (403, 404):
                raise
            # A group the account can't see into (another organization's, or outside a Groups Reader role limited to
            # some groups): its parents are left out, which can only give less.
            log.warning("couldn't read which groups contain %s, so they're left out: %s", group, e)
            parents = []
        self._parents[group] = (time.monotonic() + self.cfg.refresh_seconds, parents)
        return parents

    async def _search(self, client: httpx.AsyncClient, method: str, member: str) -> list[dict]:
        label = SECURITY if self.cfg.security_groups_only else EVERY_GROUP
        params = {"query": f"member_key_id == {_cel(member)} && '{label}' in labels", "pageSize": "1000"}
        out: list[dict] = []
        pages: set[str] = set()
        while True:
            body = await self._get(client, SEARCH + method, params)
            for m in body.get("memberships") or []:
                address = str((m.get("groupKey") or {}).get("id") or "").lower()
                if address:
                    out.append({"email": address, "name": m.get("displayName") or address, "relation": m.get("relationType")})
            token = body.get("nextPageToken")
            if not token or token in pages:
                return out
            pages.add(token)
            params = {**params, "pageToken": token}

    async def _get(self, client: httpx.AsyncClient, url: str, params: dict) -> dict:
        for attempt in (1, 2):
            token = await self.credentials.token(client)
            try:
                r = await client.get(url, params=params, headers={"authorization": f"Bearer {token}"})
            except httpx.HTTPError as e:
                raise GroupsError(f"couldn't reach the Cloud Identity API ({type(e).__name__})", outage=True) from None
            if r.status_code == 401 and attempt == 1:  # the token was revoked or expired early: get a new one, once
                self.credentials.forget()
                continue
            if r.status_code == 200:
                return _json(r)
            raise self._error(r)
        raise AssertionError("unreachable")  # pragma: no cover

    def _error(self, r: httpx.Response) -> GroupsError:
        err = _json(r).get("error")
        err = err if isinstance(err, dict) else {}
        said = str(err.get("message") or r.reason_phrase)[:200]
        reasons = {d.get("reason") for d in err.get("details") or [] if isinstance(d, dict)}
        if r.status_code == 403:
            if "ACCESS_TOKEN_SCOPE_INSUFFICIENT" in reasons:
                fix = f"the token lacks the scope {SCOPE}"
            elif "SERVICE_DISABLED" in reasons:
                fix = "turn on the Cloud Identity API in the service account's project (APIs & Services → Library)"
            else:
                who = self.credentials.account or "the service account the gateway runs as"
                fix = f"give {who} the Groups Reader admin role (Admin console → Account → Admin roles)"
            return GroupsError(f"Google refused to list groups (403: {said}): {fix}", 403)
        return GroupsError(f"the Cloud Identity API answered {r.status_code}: {said}", r.status_code,
                           outage=r.status_code == 429 or r.status_code >= 500)


def known(rec: dict | None, keep_hours: float, now: float | None = None) -> list[str] | None:
    """The group addresses last read for a person, while they're recent enough to stand; None when there are none."""
    if not rec or rec.get("groups") is None or not rec.get("fetched_at"):
        return None
    if (now or time.time()) - float(rec["fetched_at"]) > keep_hours * 3600:
        return None
    return [g["email"] for g in rec["groups"]]


class GoogleGroups:
    """Each person's Google groups: read at sign-in, read again while they use the gateway, cached in the shared state."""

    def __init__(self, oidc: OIDCConfig, store, state, forget: Callable[[str], None] | None = None,
                 transport: httpx.AsyncBaseTransport | None = None):
        assert oidc.google_groups is not None
        self.oidc, self.cfg = oidc, oidc.google_groups
        self.store, self.state = store, state
        self.forget = forget  # the gateway's key cache: forget a person whose team or role changed
        self.client = GroupsClient(self.cfg, transport)
        self._not_google: set[str] = set()  # people seen who didn't sign in with Google (given a key on the Keys page, say)

    async def record(self, user_id: str) -> dict | None:
        return await self.state.hget_json(RECORDS, user_id)

    async def records(self) -> dict[str, dict]:
        return await self.state.hgetall_json(RECORDS)

    async def known_groups(self, user_id: str) -> list[str]:
        """What the cache says, without asking Google (for a Google ID token sent directly as a bearer token)."""
        return known(await self.record(user_id), self.cfg.keep_last_known_hours) or []

    async def seen(self, user_id: str) -> None:
        """The person used the gateway: their groups are read again at the next round that's due."""
        await self.state.hset_json(SEEN, user_id, time.time())

    async def at_sign_in(self, user_id: str, address: str) -> list[str]:
        """The person's groups, read now. If Google can't answer, the last known ones while they last, else none (the
        email and domain rules alone)."""
        async with self.client.http() as client:
            rec = await self._read(user_id, address, await self.record(user_id), client)
        groups = known(rec, self.cfg.keep_last_known_hours)
        await self._save(user_id, rec, assign(self.oidc, address, groups or []), groups)
        await self.seen(user_id)
        return groups or []

    async def refresh(self, user_id: str, client: httpx.AsyncClient | None = None) -> bool:
        """Read one person's groups again and apply them (an admin's pin still wins). True if their team or role changed."""
        rec = await self.record(user_id)
        if not rec or not rec.get("email"):
            return False
        user = await self.store.get_user(user_id)
        if user is None:  # deleted: nothing to keep for them
            await self.state.hdel(RECORDS, [user_id])
            return False
        if user.get("disabled_at"):
            return False
        if client is None:
            async with self.client.http() as client:
                rec = await self._read(user_id, rec["email"], rec, client)
        else:
            rec = await self._read(user_id, rec["email"], rec, client)
        groups = known(rec, self.cfg.keep_last_known_hours)
        found = assign(self.oidc, rec["email"], groups or [])
        await self._save(user_id, rec, found, groups)
        if user.get("managed", "idp") != "idp" or (user["team"], user["role"]) == (found.team, found.role):
            return False
        await self.store.upsert_user(user_id, found.team, found.role, source="idp")
        log.info("%s now on team %s as %s (was %s, %s): %s", rec["email"], found.team, found.role, user["team"], user["role"],
                 describe(found.team_from))
        if self.forget:
            self.forget(user_id)
        return True

    async def refresh_due(self) -> list[str]:
        """Everyone who used the gateway since their groups were last read, if that was ``refresh_seconds`` ago or more.
        Returns the people whose team or role changed."""
        now = time.time()
        seen, recs = await self.state.hgetall_json(SEEN), await self.records()
        await self._adopt(seen, recs)
        due = [u for u, r in recs.items() if float(seen.get(u) or 0) > float(r.get("checked_at") or 0)
               and now - float(r.get("checked_at") or 0) >= self.cfg.refresh_seconds]
        for u in due:  # claimed: another gateway's round skips them meanwhile
            await self.state.hset_json(RECORDS, u, {**recs[u], "checked_at": now})
        gate = asyncio.Semaphore(CONCURRENCY)

        async def one(user_id: str, client: httpx.AsyncClient) -> bool:
            async with gate:
                try:
                    return await self.refresh(user_id, client)
                except Exception:  # noqa: BLE001 (one person's failure mustn't stop the round)
                    log.exception("couldn't refresh %s's Google groups", user_id)
                    return False

        async with self.client.http() as client:
            changed = await asyncio.gather(*(one(u, client) for u in due))
        return [u for u, c in zip(due, changed) if c]

    async def _adopt(self, seen: dict, recs: dict) -> None:
        """People who signed in with Google before groups were read have no reading yet: read them at their first use
        rather than at their next sign-in. Their record shows it was Google: a Google account number (Entra's IDs are
        GUIDs, and people given a key on the Keys page have no address) and an address in a hosted domain."""
        for user_id in seen:
            if user_id in recs or user_id in self._not_google:
                continue
            user = await self.store.get_user(user_id)
            email = str((user or {}).get("email") or "").lower()
            if user and user_id.isdigit() and email.rpartition("@")[2] in self.oidc.hosted_domains:
                recs[user_id] = {"email": email, "checked_at": 0}
            else:
                self._not_google.add(user_id)

    async def _read(self, user_id: str, address: str, rec: dict | None, client: httpx.AsyncClient) -> dict:
        now = time.time()
        rec = {**(rec or {}), "email": address, "checked_at": now}
        try:
            groups = await self.client.groups_of(address, client)
        except GroupsError as e:
            log.warning("couldn't read %s's Google groups: %s", address, e)
            rec.update(error=str(e), failing_since=rec.get("failing_since") or now)
            await self._failed(e)
        else:
            rec.update(groups=groups, fetched_at=now, error=None, failing_since=None)
            await self._recovered()
        return rec

    async def _save(self, user_id: str, rec: dict, found: Assignment, groups: list[str] | None) -> None:
        using = "none" if groups is None else "live" if rec.get("error") is None else "last_known"
        if using == "none" and rec.get("using") in ("live", "last_known"):  # the last known groups just aged out
            await self._expired(rec)
        await self.state.hset_json(RECORDS, user_id, {**rec, "team": found.team, "role": found.role, "team_from": found.team_from,
                                                       "role_from": found.role_from, "using": using})

    async def _failed(self, e: GroupsError) -> None:
        if not await alerts._once(self.state, "alert:google-groups", ALERT_EVERY):
            return
        await self.state.set_json(OUTAGE, True, 7 * 86400)
        hours = f"{self.cfg.keep_last_known_hours:g}"
        await alerts.notify(f"Couldn't read Google Groups: {e}. Everyone keeps their last known groups for up to {hours} hours; "
                            "after that their team and role follow the email and domain rules alone.", state=self.state,
                            title="Google Groups unavailable", level="warning" if e.outage else "critical", page="people")

    async def _recovered(self) -> None:
        if not await self.state.get_json(OUTAGE):
            return
        await self.state.set_json(OUTAGE, False, 60)
        await self.state.set_json("alert:google-groups", False, 60)  # the next failure alerts at once
        await alerts.notify("Google Groups can be read again: teams and roles follow them.", state=self.state,
                            title="Google Groups readable again", level="info", page="people")

    async def _expired(self, rec: dict) -> None:
        log.warning("%s's Google groups were last read over %g hours ago: their team and role follow the email and domain rules",
                    rec["email"], self.cfg.keep_last_known_hours)
        if await alerts._once(self.state, "alert:google-groups-expired", ALERT_EVERY):
            await alerts.notify(f"Google Groups have been unreadable for over {self.cfg.keep_last_known_hours:g} hours: "
                                f"{rec['email']}'s team and role, and others' as their groups age out, now follow the email and "
                                "domain rules alone. The People page shows whose.", state=self.state,
                                title="Google Groups expired: email rules apply", level="critical", page="people")


def describe(source: dict | None) -> str:
    """Which mapping gave a team or role, in words: group eng@example.com, address bob@example.com, default team."""
    if not source or source.get("by") == "default":
        return "default team"
    return f"{source['by']} {source['key']}"


async def follow(gw) -> None:
    """Runs on every gateway; one of them does each round, claimed through the shared state."""
    while True:
        groups = gw.authn.groups
        tick = min(TICK_SECONDS, groups.cfg.refresh_seconds) if groups else TICK_SECONDS
        await asyncio.sleep(tick)
        groups = gw.authn.groups  # a configuration change may have replaced it
        if groups is None:
            continue
        try:
            # Claimed in one step (set if absent): a read then a write would let two gateways ticking together both take it.
            token = uuid.uuid4().hex
            if (await gw.state.set_json_once({ROUND: token}, max(1, round(tick)))).get(ROUND) == token:
                await groups.refresh_due()
        except Exception:  # noqa: BLE001 (Redis or the store hiccuped: the next round tries again)
            log.exception("couldn't refresh Google groups")
