"""SCIM 2.0 provisioning (RFC 7643/7644), the subset Microsoft Entra uses.

Entra (or Okta, etc.) pushes people and groups here as they change:
* People are keyed by their directory object ID (``externalId``), the same ID
  ``meritorder login`` uses, so provisioning and sign-in meet on one record.
* Group membership recomputes a person's team and role with the same mapping as
  sign-in (Config → Access); an admin's pin on the People page still wins.
* Deactivating or deleting someone in Entra disables them at once: every key
  stops working within a minute. Their spend history is kept; what they stored
  in the Responses API's stored mode is deleted.

Enabled by setting MERITORDER_SCIM_TOKEN; Entra sends it as a bearer token.
"""

from __future__ import annotations

import hmac
import os
import re
import uuid
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

from .auth import assign

USER_SCHEMA = "urn:ietf:params:scim:schemas:core:2.0:User"
GROUP_SCHEMA = "urn:ietf:params:scim:schemas:core:2.0:Group"
LIST_SCHEMA = "urn:ietf:params:scim:api:messages:2.0:ListResponse"
ERROR_SCHEMA = "urn:ietf:params:scim:api:messages:2.0:Error"
_FILTER = re.compile(r'^\s*(\w+(?:\.\w+)?)\s+eq\s+"([^"]*)"\s*$', re.IGNORECASE)
_MEMBER_PATH = re.compile(r'^members\[value eq "([^"]+)"\]$', re.IGNORECASE)


def scim_error(status: int, detail: str, scim_type: str | None = None) -> JSONResponse:
    body: dict[str, Any] = {"schemas": [ERROR_SCHEMA], "status": str(status), "detail": detail}
    if scim_type:
        body["scimType"] = scim_type
    return JSONResponse(body, status_code=status, media_type="application/scim+json")


def ok(body: Any, status: int = 200) -> JSONResponse:
    return JSONResponse(body, status_code=status, media_type="application/scim+json")


def truthy(v: Any) -> bool:
    return v.strip().lower() == "true" if isinstance(v, str) else bool(v)


def build_router(gateway) -> APIRouter:
    router = APIRouter(prefix="/scim/v2", tags=["scim"])

    def authorized(request: Request) -> JSONResponse | None:
        token = os.environ.get("MERITORDER_SCIM_TOKEN")
        if not token:
            return scim_error(404, "SCIM provisioning isn't enabled (set MERITORDER_SCIM_TOKEN)")
        sent = request.headers.get("authorization", "")
        if not hmac.compare_digest(sent.encode(), f"Bearer {token}".encode()):
            return scim_error(401, "invalid SCIM token")
        return None

    def base_url(request: Request) -> str:
        return str(request.base_url).rstrip("/") + "/scim/v2"

    # ------------------------------------------------------------------ mapping

    async def recompute(user_id: str) -> None:
        """Team and role from group membership, using the sign-in mapping (pins still win)."""
        user = await gateway.store.get_user(user_id)
        if user is None:
            return
        oidc = gateway.cfg.auth.oidc
        groups = [g for g in await gateway.store.scim_groups() if user_id in (g.get("members") or [])]
        names = {x for g in groups for x in (g["id"], g.get("external_id"), g["display_name"]) if x}
        team, role = "default", "user"
        if oidc:  # the sign-in mapping (auth.assign); Entra's addresses aren't verified, so only groups count
            found = assign(oidc, None, names)
            team, role = found.team, found.role
        await gateway.store.upsert_user(user_id, team, role, source="idp")
        gateway.authn.forget_user(user_id)

    async def user_resource(u: dict, request: Request) -> dict:
        email = u.get("email")
        return {
            "schemas": [USER_SCHEMA], "id": u["id"], "externalId": u["id"], "userName": email or u["id"],
            "displayName": u.get("display_name"), "active": not u.get("disabled_at"),
            "emails": [{"value": email, "type": "work", "primary": True}] if email else [],
            "meta": {"resourceType": "User", "location": f"{base_url(request)}/Users/{u['id']}"},
        }

    def group_resource(g: dict, request: Request, members: bool = True) -> dict:
        out = {"schemas": [GROUP_SCHEMA], "id": g["id"], "externalId": g.get("external_id"), "displayName": g["display_name"],
               "meta": {"resourceType": "Group", "location": f"{base_url(request)}/Groups/{g['id']}"}}
        if members:
            out["members"] = [{"value": m} for m in g.get("members") or []]
        return out

    def listing(resources: list, request: Request) -> JSONResponse:
        try:
            start = max(int(request.query_params.get("startIndex", 1)), 1)
            # RFC 7644 §3.4.2.4: a negative count means 0; we advertise maxResults 200.
            count = min(max(int(request.query_params.get("count", 100)), 0), 200)
        except ValueError:
            return scim_error(400, "startIndex and count must be integers", "invalidValue")
        page = resources[start - 1: start - 1 + count]
        return ok({"schemas": [LIST_SCHEMA], "totalResults": len(resources), "startIndex": start, "itemsPerPage": len(page), "Resources": page})

    async def apply_user_attrs(user_id: str, attrs: dict) -> None:
        email = attrs.get("userName")
        emails = attrs.get("emails") or []
        if isinstance(emails, list) and emails:
            primary = next((e for e in emails if e.get("primary")), emails[0])
            email = primary.get("value") or email
        user = await gateway.store.get_user(user_id)
        await gateway.store.upsert_user(user_id, user["team"] if user else "default", user["role"] if user else "user",
                                        email=email, display_name=attrs.get("displayName"), source="ensure")
        if "active" in attrs:
            await gateway.store.update_user(user_id, disabled=not truthy(attrs["active"]))
            gateway.authn.forget_user(user_id)
            if not truthy(attrs["active"]):
                await gateway.stored.forget_user(user_id)  # what they stored in the Responses API's stored mode goes too

    # ------------------------------------------------------------------ discovery

    @router.get("/ServiceProviderConfig")
    async def service_provider_config(request: Request):
        if (err := authorized(request)) is not None:
            return err
        return ok({"schemas": ["urn:ietf:params:scim:schemas:core:2.0:ServiceProviderConfig"],
                   "patch": {"supported": True}, "bulk": {"supported": False, "maxOperations": 0, "maxPayloadSize": 0},
                   "filter": {"supported": True, "maxResults": 200}, "changePassword": {"supported": False},
                   "sort": {"supported": False}, "etag": {"supported": False},
                   "authenticationSchemes": [{"type": "oauthbearertoken", "name": "Bearer token", "description": "MERITORDER_SCIM_TOKEN"}]})

    @router.get("/ResourceTypes")
    async def resource_types(request: Request):
        if (err := authorized(request)) is not None:
            return err
        return ok({"schemas": [LIST_SCHEMA], "totalResults": 2, "Resources": [
            {"id": "User", "name": "User", "endpoint": "/Users", "schema": USER_SCHEMA},
            {"id": "Group", "name": "Group", "endpoint": "/Groups", "schema": GROUP_SCHEMA}]})

    # ------------------------------------------------------------------ users

    @router.get("/Users")
    async def list_users(request: Request):
        if (err := authorized(request)) is not None:
            return err
        flt = request.query_params.get("filter")
        if flt:
            m = _FILTER.match(flt)
            if not m:
                return scim_error(400, f"unsupported filter: {flt}", "invalidFilter")
            attr, value = m.group(1).lower(), m.group(2)
            if attr in ("username", "emails.value"):
                u = await gateway.store.find_user_by_email(value) or await gateway.store.get_user(value)
            elif attr in ("externalid", "id"):
                u = await gateway.store.get_user(value)
            else:
                return scim_error(400, f"can't filter on {attr}", "invalidFilter")
            return listing([await user_resource(u, request)] if u else [], request)
        return listing([await user_resource(u, request) for u in await gateway.store.list_users()], request)

    @router.post("/Users")
    async def create_user(request: Request):
        if (err := authorized(request)) is not None:
            return err
        body = await request.json()
        user_id = body.get("externalId") or body.get("userName")
        if not user_id:
            return scim_error(400, "externalId or userName is required", "invalidValue")
        if await gateway.store.get_user(user_id):
            return scim_error(409, f"user {user_id} already exists", "uniqueness")
        await apply_user_attrs(user_id, body)
        await recompute(user_id)
        return ok(await user_resource(await gateway.store.get_user(user_id), request), 201)

    @router.get("/Users/{user_id}")
    async def get_user(request: Request, user_id: str):
        if (err := authorized(request)) is not None:
            return err
        u = await gateway.store.get_user(user_id)
        return ok(await user_resource(u, request)) if u else scim_error(404, "no such user")

    @router.put("/Users/{user_id}")
    async def replace_user(request: Request, user_id: str):
        if (err := authorized(request)) is not None:
            return err
        if not await gateway.store.get_user(user_id):
            return scim_error(404, "no such user")
        await apply_user_attrs(user_id, await request.json())
        return ok(await user_resource(await gateway.store.get_user(user_id), request))

    @router.patch("/Users/{user_id}")
    async def patch_user(request: Request, user_id: str):
        if (err := authorized(request)) is not None:
            return err
        if not await gateway.store.get_user(user_id):
            return scim_error(404, "no such user")
        attrs: dict[str, Any] = {}
        for op in (await request.json()).get("Operations") or []:
            if str(op.get("op", "")).lower() not in ("replace", "add"):
                continue
            path, value = op.get("path"), op.get("value")
            if path:
                key = {"active": "active", "displayname": "displayName", "username": "userName"}.get(path.lower())
                if key:
                    attrs[key] = value
                elif path.lower().startswith("emails"):
                    attrs["emails"] = [{"value": value, "primary": True}] if isinstance(value, str) else value
            elif isinstance(value, dict):
                attrs.update(value)
        await apply_user_attrs(user_id, attrs)
        return ok(await user_resource(await gateway.store.get_user(user_id), request))

    @router.delete("/Users/{user_id}")
    async def delete_user(request: Request, user_id: str):
        if (err := authorized(request)) is not None:
            return err
        if not await gateway.store.update_user(user_id, disabled=True):
            return scim_error(404, "no such user")
        gateway.authn.forget_user(user_id)
        await gateway.stored.forget_user(user_id)
        return Response(status_code=204)

    # ------------------------------------------------------------------ groups

    @router.get("/Groups")
    async def list_groups(request: Request):
        if (err := authorized(request)) is not None:
            return err
        members = "members" not in (request.query_params.get("excludedAttributes") or "")
        groups = await gateway.store.scim_groups()
        flt = request.query_params.get("filter")
        if flt:
            m = _FILTER.match(flt)
            if not m:
                return scim_error(400, f"unsupported filter: {flt}", "invalidFilter")
            attr, value = m.group(1).lower(), m.group(2)
            field = {"displayname": "display_name", "externalid": "external_id", "id": "id"}.get(attr)
            if not field:
                return scim_error(400, f"can't filter on {attr}", "invalidFilter")
            groups = [g for g in groups if (g.get(field) or "") == value]
        return listing([group_resource(g, request, members) for g in groups], request)

    @router.post("/Groups")
    async def create_group(request: Request):
        if (err := authorized(request)) is not None:
            return err
        body = await request.json()
        if not body.get("displayName"):
            return scim_error(400, "displayName is required", "invalidValue")
        group = {"id": uuid.uuid4().hex, "display_name": body["displayName"], "external_id": body.get("externalId"),
                 "members": [m["value"] for m in body.get("members") or [] if m.get("value")]}
        await gateway.store.save_scim_group(group)
        for uid in group["members"]:
            await recompute(uid)
        return ok(group_resource(group, request), 201)

    @router.get("/Groups/{gid}")
    async def get_group(request: Request, gid: str):
        if (err := authorized(request)) is not None:
            return err
        g = await gateway.store.scim_group(gid)
        return ok(group_resource(g, request)) if g else scim_error(404, "no such group")

    async def _set_members(g: dict, members: list[str], request: Request) -> JSONResponse:
        before = set(g.get("members") or [])
        g["members"] = list(dict.fromkeys(members))
        await gateway.store.save_scim_group(g)
        for uid in before ^ set(g["members"]):
            await recompute(uid)
        return ok(group_resource(g, request))

    @router.patch("/Groups/{gid}")
    async def patch_group(request: Request, gid: str):
        if (err := authorized(request)) is not None:
            return err
        g = await gateway.store.scim_group(gid)
        if not g:
            return scim_error(404, "no such group")
        members = list(g.get("members") or [])
        for op in (await request.json()).get("Operations") or []:
            kind, path, value = str(op.get("op", "")).lower(), op.get("path") or "", op.get("value")
            values = [v.get("value") for v in value] if isinstance(value, list) else []
            m = _MEMBER_PATH.match(path)
            if kind == "add" and path.lower() == "members":
                members += [v for v in values if v and v not in members]
            elif kind == "remove" and m:
                members = [x for x in members if x != m.group(1)]
            elif kind == "remove" and path.lower() == "members":
                members = [x for x in members if x not in values] if values else []
            elif kind == "replace" and path.lower() == "members":
                members = [v for v in values if v]
            elif kind == "replace" and path.lower() == "displayname":
                g["display_name"] = value
            elif kind == "replace" and path.lower() == "externalid":
                g["external_id"] = value
            elif kind == "replace" and not path and isinstance(value, dict):
                g["display_name"] = value.get("displayName", g["display_name"])
                g["external_id"] = value.get("externalId", g.get("external_id"))
        return await _set_members(g, members, request)

    @router.put("/Groups/{gid}")
    async def replace_group(request: Request, gid: str):
        if (err := authorized(request)) is not None:
            return err
        g = await gateway.store.scim_group(gid)
        if not g:
            return scim_error(404, "no such group")
        body = await request.json()
        g["display_name"] = body.get("displayName", g["display_name"])
        g["external_id"] = body.get("externalId", g.get("external_id"))
        return await _set_members(g, [m["value"] for m in body.get("members") or [] if m.get("value")], request)

    @router.delete("/Groups/{gid}")
    async def delete_group(request: Request, gid: str):
        if (err := authorized(request)) is not None:
            return err
        g = await gateway.store.scim_group(gid)
        if not g:
            return scim_error(404, "no such group")
        await gateway.store.delete_scim_group(gid)
        for uid in g.get("members") or []:
            await recompute(uid)
        return Response(status_code=204)

    return router
