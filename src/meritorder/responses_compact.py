"""``POST /v1/responses/compact`` and ``POST /v1/responses/input_tokens``, as OpenAI's API reference defines them
(openai-python 3.42, November 2026), on every backend (decision 54).

**Compaction is the gateway's own.** OpenAI's returns the conversation's user messages and one ``compaction`` item whose
``encrypted_content`` only OpenAI can read. Behind ``auto`` the next turn may go to any provider, which couldn't read it,
so the gateway writes the summary itself, with a model the router picks: the conversation goes through the pipeline as a
summarizing request, routed like Claude Code's compaction requests (capped at ``routing.request_class_levels.compaction``),
admitted against budgets and limits, and metered as one ledger row tagged ``compaction: responses``. The ``compaction``
item it returns holds the summary, encoded (formats.responses.compaction_item); a later turn that carries it is read as a
user message with the summary, wherever that turn goes.

Input tokens are the gateway's estimate for the named model, calibrated by providers' own counts (tokens.py), as
Anthropic's count_tokens is answered when no backend can count.
"""

from __future__ import annotations

import json
import time

from fastapi import Request
from fastapi.responses import JSONResponse, Response

from .auth import AuthError, Principal
from .formats.responses import ResponsesError, compaction_item, responses_to_chat, responses_usage
from .formats.translate import usage_from_openai
from .proxy import GatewayError
from .tokens import estimate

# The request class the summary is routed as: the cap Claude Code's compaction summaries get, and never a judged sample.
CLASS_HEADER = "x-claude-code-request-class"

SUMMARY_PROMPT = (
    "Compact the conversation above so the work can continue without it. Write a summary to be read in place of every "
    "earlier turn. Keep the user's goals and constraints, the decisions made and why, what was done (files changed, "
    "commands run and their results), errors and how they were resolved, the current state, and what remains to do. "
    "Keep exact names, paths, identifiers, numbers and commands. Leave out what has been superseded. Write only the summary."
)


class ResponsesTools:
    """The two endpoints, on the gateway (proxy.Gateway) they run through."""

    def __init__(self, gw):
        self.gw = gw

    async def _read(self, request: Request) -> tuple[Principal | None, dict, dict, Response | None]:
        headers = {k.lower(): v for k, v in request.headers.items()}
        try:
            principal = await self.gw.authn.authenticate(headers)
        except AuthError as e:
            return None, headers, {}, self.gw.reject("openai", e.status, e.message, "authentication_error")
        try:
            body = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None, headers, {}, self.gw.reject("openai", 400, "request body must be valid JSON", "invalid_request_error")
        if not isinstance(body, dict):
            return None, headers, {}, self.gw.reject("openai", 400, "request body must be a JSON object", "invalid_request_error")
        return principal, headers, body, None

    async def items(self, principal: Principal, body: dict) -> list[dict]:
        """The whole conversation the request names: what's stored before ``previous_response_id`` or in the
        ``conversation`` (stored mode only, as for a new turn), then its ``input``."""
        from .stored.service import normalize_input

        if body.get("previous_response_id") or body.get("conversation"):
            turn = await self.gw.stored.begin(principal, {"model": body.get("model"), "input": body.get("input") or [], "store": False,
                                                          "previous_response_id": body.get("previous_response_id"),
                                                          "conversation": body.get("conversation")})
            return turn.history + turn.input
        return normalize_input(body.get("input") or [])

    async def compact(self, request: Request) -> Response:
        from .stored.service import StoredError, new_id

        t0 = time.monotonic()
        principal, headers, body, refused = await self._read(request)
        if refused is not None:
            return refused
        assert principal is not None
        if not body.get("model"):
            return self.gw.reject("openai", 400, "model is required", "invalid_request_error")
        try:
            items = await self.items(principal, body)
            if not items:
                raise ResponsesError("there is nothing to compact: send the conversation in `input`, or `previous_response_id`")
            chat = responses_to_chat({"model": body["model"], "instructions": body.get("instructions"), "input": items})
        except StoredError as e:
            return self.gw.reject("openai", e.status, e.message, e.kind)
        except ResponsesError as e:
            return self.gw.reject("openai", 400, str(e), "invalid_request_error")
        chat["messages"].append({"role": "user", "content": SUMMARY_PROMPT})
        if body.get("prompt_cache_key"):
            chat["prompt_cache_key"] = body["prompt_cache_key"]
        resp = await self.gw.respond(principal, chat, {**headers, CLASS_HEADER: "compaction"}, "openai", t0, tags={"compaction": "responses"})
        if resp.status_code >= 400:
            return resp
        data = json.loads(bytes(resp.body))
        content = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        summary = content if isinstance(content, str) else "".join(p.get("text", "") for p in content if isinstance(p, dict))
        # As OpenAI returns it: every user message, then the compaction item that stands for everything else.
        kept = [i for i in items if i.get("type") == "message" and i.get("role") == "user"]
        out = {"id": new_id("resp"), "object": "response.compaction", "created_at": int(time.time()),
               "output": kept + [compaction_item(new_id("cmp"), summary.strip())],
               "usage": responses_usage(usage_from_openai(data.get("usage")))}
        return JSONResponse(out, headers={k: v for k, v in resp.headers.items() if k.startswith("x-meritorder-")})

    async def input_tokens(self, request: Request) -> Response:
        from .stored.service import StoredError

        principal, headers, body, refused = await self._read(request)
        if refused is not None:
            return refused
        assert principal is not None
        team = self.gw.cfg.team(principal.team)
        model = self.gw.cfg.model(str(body.get("model") or ""))
        try:
            if model is not None and not team.allows_model(model.name):
                raise GatewayError(403, f"model '{model.name}' is not allowed for team '{team.name}'", "permission_error")
            await self.gw.admit(principal, team, 0)  # budgets and request-rate limits; counting uses no tokens
            items = await self.items(principal, body)
            rest = {k: v for k, v in body.items() if k not in ("previous_response_id", "conversation")}
            est = estimate(responses_to_chat({**rest, "input": items}))
        except GatewayError as e:
            return self.gw.reject("openai", e.status, e.message, e.kind, e.headers)
        except StoredError as e:
            return self.gw.reject("openai", e.status, e.message, e.kind)
        except ResponsesError as e:
            return self.gw.reject("openai", 400, str(e), "invalid_request_error")
        tokens = self.gw.tokens.count(model.name, est) if model else est.tokens
        return JSONResponse({"object": "response.input_tokens", "input_tokens": tokens})
