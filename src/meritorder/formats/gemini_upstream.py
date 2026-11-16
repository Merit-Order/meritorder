"""Sending Chat Completions requests to Gemini's native API (``generateContent`` / ``streamGenerateContent``).

Gemini's OpenAI-compatible endpoints (Vertex AI's, OpenRouter's) work, but lose or flatten what the native API
reports: thought summaries, cached-token counts, why a reply was blocked, and the thought signatures Gemini 3 and
later require back on function calls. A backend with ``api: gemini`` speaks the native API while the rest of the
gateway keeps seeing Chat Completions, like ``api: responses`` (responses_upstream.py): requests are converted on
the way out, and replies and streams are converted back.

Thought signatures ride on each tool call as ``extra_content.google.thought_signature``, the convention of Google's
own OpenAI-compatible endpoint, so a client that sends its messages back as it got them keeps Gemini's reasoning
between tool calls. A tool call without one (a client that rebuilt its history, a translated Anthropic request,
or a call another model made) is sent with the bypass value Google documents for that case, since Gemini rejects
a function call in the current turn that has no signature.
"""

from __future__ import annotations

import json
import mimetypes
import time
import uuid
from typing import Any
from urllib.parse import urlparse

from .sse import SSEEvent, event
from .translate import text_of

# Google's documented value for a function call the API didn't generate (or whose signature was lost): it skips
# signature validation. Gemini answers a current-turn function call that has no signature at all with a 400.
SKIP_SIGNATURE = "skip_thought_signature_validator"

# Thinking levels, weakest first (Gemini's thinkingLevel), and each model's own (generateContent thinking guide,
# November 2026). The longest matching model-id prefix wins. Gemini 3 and 4 can't turn thinking off.
LEVELS = ("minimal", "low", "medium", "high", "xhigh")
_MODEL_LEVELS = {
    "gemini-4-hydrogen": ("low", "medium", "high", "xhigh"),
    "gemini-4": ("low", "medium", "high"),
    "gemini-3.8-flash": ("low", "medium", "high"),
    "gemini-3.6-flash": ("minimal", "low", "medium", "high"),
    "gemini-3.5-flash-lite": ("minimal", "low", "medium", "high"),
    "gemini-3.1-pro": ("low", "medium", "high"),
    "gemini-3.1-flash-lite-image": ("minimal", "high"),
    "gemini-3.1-flash-lite": ("minimal", "low", "medium", "high"),
    "gemini-3-flash": ("minimal", "low", "medium", "high"),
    "gemini-3-pro": ("low", "high"),
    "gemini-robotics-er-2": ("minimal", "low", "medium", "high"),
}
_OTHER_LEVELS = ("low", "medium", "high")  # a model not listed: the levels nearly every Gemini 3 and 4 model takes
# Google's guide says Gemini 4 needs thinkingLevel; a request with no effort gets the model's default level.
_DEFAULT_LEVEL = {"gemini-4-hydrogen": "high", "gemini-4": "medium"}
# Gemini 2.5 takes a token budget instead: (least, most, whether 0 turns thinking off).
_BUDGETS = {"gemini-2.5-pro": (128, 32768, False), "gemini-2.5-flash-lite": (512, 24576, True), "gemini-2.5-flash": (0, 24576, True)}
_EFFORT_BUDGET = {"minimal": 512, "low": 1024, "medium": 8192, "high": 24576}

# finishReason values that mean Gemini's filters stopped the reply: a refusal (content_filter), which `auto` retries
# on another model. Values that mean the reply came out broken are errors, so the request goes to the next source.
BLOCKED = {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "LANGUAGE", "IMAGE_SAFETY",
           "IMAGE_PROHIBITED_CONTENT", "IMAGE_RECITATION", "ESCALATION"}
FAILED = {"MALFORMED_FUNCTION_CALL": 502, "MALFORMED_RESPONSE": 502, "MISSING_THOUGHT_SIGNATURE": 400,
          "PUP_LIMITED_DISABLED": 403}  # the last: the Google account is limited or disabled, so another backend
_FINISH = {"STOP": "stop", "MAX_TOKENS": "length", "CONTINUATION": "length"}
# Google's error statuses, for an error that carries no HTTP code.
_STATUS = {"RESOURCE_EXHAUSTED": 429, "UNAVAILABLE": 503, "INTERNAL": 500, "DEADLINE_EXCEEDED": 504, "INVALID_ARGUMENT": 400,
           "FAILED_PRECONDITION": 400, "PERMISSION_DENIED": 403, "UNAUTHENTICATED": 401, "NOT_FOUND": 404}


def model_id(model: str) -> str:
    """The id in the URL. Vertex's OpenAI-compatible endpoint names models ``google/<id>`` and the Gemini API lists
    them as ``models/<id>``; either form works on a route, so switching a backend to ``api: gemini`` needs no route change."""
    return str(model or "").removeprefix("models/").removeprefix("google/")


def _match(table: dict, model: str):
    best = max((k for k in table if model.startswith(k)), key=len, default=None)
    return table[best] if best else None


def _fit(effort: str, levels: tuple[str, ...]) -> str:
    """The effort if the model takes it, else the strongest level it takes below it, else its weakest (as the gateway
    fits efforts to a model's ``effort_levels``). ``none`` is the weakest: Gemini 3 and 4 can't turn thinking off."""
    if effort in levels:
        return effort
    below = [lv for lv in levels if effort in LEVELS and LEVELS.index(lv) <= LEVELS.index(effort)]
    return below[-1] if below else levels[0]


def thinking_config(model: str, effort: str | None) -> dict | None:
    """generationConfig.thinkingConfig for the gateway's reasoning effort on this model. Thought summaries are always
    asked for on Gemini models: they cost nothing extra (billing counts the whole thinking either way), and the
    reply carries them as ``reasoning_content``, as the gateway does for Claude's thinking. An effort Gemini has no
    level for is left out, as the gateway leaves out one a model doesn't take."""
    m = model_id(model)
    effort = "xhigh" if effort == "max" else effort
    if effort not in LEVELS and effort != "none":
        effort = None
    budget = _match(_BUDGETS, m)
    if budget:
        least, most, can_stop = budget
        out: dict[str, Any] = {"includeThoughts": True}
        if effort == "none":
            out["thinkingBudget"] = 0 if can_stop else least
        elif effort:
            out["thinkingBudget"] = min(max(_EFFORT_BUDGET.get(effort, most), least), most)
        return out
    levels = _match(_MODEL_LEVELS, m)
    if levels is None and (not m.startswith("gemini-") or m.startswith(("gemini-1", "gemini-2.0"))):
        # Not a thinking Gemini (Gemma, say, or Gemini 2.0): only what was asked for, and "none" is what it does already
        return {"thinkingLevel": _fit(effort, _OTHER_LEVELS)} if effort and effort != "none" else None
    out = {"includeThoughts": True}
    level = _fit(effort, levels or _OTHER_LEVELS) if effort else _match(_DEFAULT_LEVEL, m)
    if level:
        out["thinkingLevel"] = level
    return out


# ---------------------------------------------------------------- requests


def _media(url: str) -> dict | None:
    if url.startswith("data:") and ";base64," in url:
        mime, data = url[5:].split(";base64,", 1)
        return {"inlineData": {"mimeType": mime or "application/octet-stream", "data": data}}
    if not url:
        return None
    # fileData needs a MIME type: from the file name, or a JPEG for an image URL that doesn't say
    return {"fileData": {"mimeType": mimetypes.guess_type(urlparse(url).path)[0] or "image/jpeg", "fileUri": url}}


def _user_parts(content: Any) -> list[dict]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"text": content}] if content else []
    parts: list[dict] = []
    for p in content:
        if isinstance(p, str):
            parts.append({"text": p})
            continue
        t = p.get("type")
        if t in ("text", "input_text") and p.get("text"):
            parts.append({"text": p["text"]})
        elif t == "image_url":
            img = p.get("image_url")
            media = _media((img or {}).get("url", "") if isinstance(img, dict) else str(img or ""))
            if media:
                parts.append(media)
        elif t == "input_audio":
            audio = p.get("input_audio") or {}
            if audio.get("data"):
                parts.append({"inlineData": {"mimeType": f"audio/{audio.get('format') or 'wav'}", "data": audio["data"]}})
        elif t == "file":
            f = p.get("file") or {}
            data = str(f.get("file_data") or "")
            if data.startswith("data:") and ";base64," in data:
                parts.append({"inlineData": _media(data)["inlineData"]})
            elif data:
                parts.append({"inlineData": {"mimeType": mimetypes.guess_type(f.get("filename") or "")[0] or "application/pdf", "data": data}})
    return parts


def _args(arguments: Any) -> dict:
    if isinstance(arguments, dict):
        return arguments
    try:
        value = json.loads(arguments or "{}")
    except (json.JSONDecodeError, TypeError):
        return {"arguments": arguments}
    return value if isinstance(value, dict) else {"value": value}


def _function_output(content: Any) -> dict:
    """functionResponse.response must be an object: a tool's JSON object as it is, anything else under ``output``."""
    text = text_of(content)
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {"output": text}
    return value if isinstance(value, dict) else {"output": value}


def _signature(call: dict, details: dict[str, str]) -> str | None:
    """A tool call's thought signature: Google's convention (``extra_content.google``), or OpenRouter's
    ``reasoning_details`` entry for that call, when the session was on Gemini through OpenRouter before."""
    google = ((call.get("extra_content") or {}).get("google") or {}) if isinstance(call.get("extra_content"), dict) else {}
    return google.get("thought_signature") or details.get(str(call.get("id") or ""))


def chat_to_gemini_request(body: dict) -> dict:
    """A GenerateContentRequest for a Chat Completions request (the model goes in the URL, not the body)."""
    system: list[dict] = []
    contents: list[dict] = []
    names: dict[str, str] = {}  # tool call id -> function name, for the functionResponse that answers it

    def push(role: str, parts: list[dict]) -> None:
        if not parts:
            return
        if contents and contents[-1]["role"] == role:
            contents[-1]["parts"].extend(parts)
        else:
            contents.append({"role": role, "parts": parts})

    for m in body.get("messages") or []:
        role = m.get("role")
        if role in ("system", "developer"):
            text = text_of(m.get("content"))
            if text:
                system.append({"text": text})
        elif role == "user":
            push("user", _user_parts(m.get("content")))
        elif role == "assistant":
            parts: list[dict] = []
            text = text_of(m.get("content"))
            if text:
                parts.append({"text": text})
            details = {str(d.get("id")): d["data"] for d in m.get("reasoning_details") or []
                       if isinstance(d, dict) and d.get("type") == "reasoning.encrypted" and d.get("id") and d.get("data")}
            calls = []
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                call: dict[str, Any] = {"name": fn.get("name", ""), "args": _args(fn.get("arguments"))}
                if tc.get("id"):
                    call["id"] = tc["id"]
                    names[tc["id"]] = call["name"]
                part: dict[str, Any] = {"functionCall": call}
                sig = _signature(tc, details)
                if sig:
                    part["thoughtSignature"] = sig
                calls.append(part)
            if calls and not any("thoughtSignature" in p for p in calls):
                calls[0]["thoughtSignature"] = SKIP_SIGNATURE  # only the first call of a step carries one
            push("model", parts + calls)
        elif role == "tool":
            call_id = m.get("tool_call_id") or ""
            response: dict[str, Any] = {"name": names.get(call_id) or m.get("name") or "tool", "response": _function_output(m.get("content"))}
            if call_id:
                response["id"] = call_id
            push("user", [{"functionResponse": response}])

    out: dict[str, Any] = {"contents": contents}
    if system:
        out["systemInstruction"] = {"parts": system}

    decls = []
    for t in body.get("tools") or []:
        if t.get("type") != "function":
            continue  # Gemini's built-in tools (Search, code execution) have no Chat Completions equivalent here
        fn = t.get("function") or {}
        decl: dict[str, Any] = {"name": fn.get("name", "")}
        if fn.get("description"):
            decl["description"] = fn["description"]
        if fn.get("parameters"):
            decl["parametersJsonSchema"] = fn["parameters"]  # JSON Schema as clients write it, not Gemini's OpenAPI subset
        decls.append(decl)
    if decls:
        out["tools"] = [{"functionDeclarations": decls}]
        choice = body.get("tool_choice")
        mode = {"auto": "AUTO", "none": "NONE", "required": "ANY"}.get(choice) if isinstance(choice, str) else None
        config: dict[str, Any] = {"mode": mode} if mode else {}
        if isinstance(choice, dict) and (choice.get("function") or {}).get("name"):
            config = {"mode": "ANY", "allowedFunctionNames": [choice["function"]["name"]]}
        if config:
            out["toolConfig"] = {"functionCallingConfig": config}

    gen: dict[str, Any] = {}
    limit = body.get("max_completion_tokens") or body.get("max_tokens")
    if limit:
        gen["maxOutputTokens"] = int(limit)
    for src, dst in (("temperature", "temperature"), ("top_p", "topP"), ("seed", "seed"),
                     ("presence_penalty", "presencePenalty"), ("frequency_penalty", "frequencyPenalty")):
        if body.get(src) is not None:
            gen[dst] = body[src]
    stop = body.get("stop")
    if stop:
        gen["stopSequences"] = [stop] if isinstance(stop, str) else list(stop)
    rf = body.get("response_format") or {}
    if rf.get("type") in ("json_schema", "json_object"):
        gen["responseMimeType"] = "application/json"
        schema = (rf.get("json_schema") or {}).get("schema")
        if schema:
            gen["responseJsonSchema"] = schema
    reasoning = body.get("reasoning") if isinstance(body.get("reasoning"), dict) else {}  # OpenRouter's form, from a client
    thinking = thinking_config(str(body.get("model") or ""), body.get("reasoning_effort") or reasoning.get("effort"))
    if thinking:
        gen["thinkingConfig"] = thinking
    if gen:
        out["generationConfig"] = gen
    return out


# ---------------------------------------------------------------- replies


def chat_usage(meta: dict | None) -> dict | None:
    """usageMetadata in Chat Completions terms. Thinking is billed as output, and so are prompt tokens that tool use
    added. promptTokenCount already includes the cached tokens."""
    if not meta:
        return None
    prompt = (meta.get("promptTokenCount") or 0) + (meta.get("toolUsePromptTokenCount") or 0)
    thoughts = meta.get("thoughtsTokenCount") or 0
    completion = (meta.get("candidatesTokenCount") or 0) + thoughts
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": meta.get("totalTokenCount") or prompt + completion,
            "prompt_tokens_details": {"cached_tokens": meta.get("cachedContentTokenCount") or 0},
            "completion_tokens_details": {"reasoning_tokens": thoughts}}


def _call_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"


def _tool_call(part: dict) -> dict:
    fc = part.get("functionCall") or {}
    call: dict[str, Any] = {"id": fc.get("id") or _call_id(), "type": "function",
                            "function": {"name": fc.get("name", ""), "arguments": json.dumps(fc.get("args") or {}, ensure_ascii=False)}}
    if part.get("thoughtSignature"):
        call["extra_content"] = {"google": {"thought_signature": part["thoughtSignature"]}}
    return call


def _why(reason: str, message: str | None, what: str = "this reply") -> str:
    return f"Gemini stopped {what} ({reason})" + (f": {message}" if message else "")


def _blocked_prompt(feedback: dict) -> str:
    return _why(str(feedback.get("blockReason")), feedback.get("blockReasonMessage"), "the prompt")


def _error(err: Any) -> dict:
    """A Google error object as a Chat Completions error, with a code the failover code reads as a status."""
    err = err if isinstance(err, dict) else {"message": str(err)}
    out: dict[str, Any] = {"message": err.get("message") or "upstream error", "type": "api_error"}
    code = err.get("code") if isinstance(err.get("code"), int) else _STATUS.get(str(err.get("status") or ""))
    if code:
        out["code"] = code
    return out


def _failure(reason: str, message: str | None) -> dict:
    return {"error": {"message": _why(reason, message), "type": "api_error", "code": FAILED[reason]}}


def gemini_to_chat_completion(resp: dict, model: str) -> dict:
    """A Chat Completions response for a GenerateContentResponse. A reply that came out broken (a malformed function
    call, say) becomes an error body, which the gateway treats as a failure of this source."""
    if resp.get("error"):
        return {"error": _error(resp["error"])}
    text, thinking, calls, refusal = "", "", [], None
    feedback = resp.get("promptFeedback") or {}
    candidate = (resp.get("candidates") or [{}])[0] or {}
    reason = candidate.get("finishReason") or ""
    if reason in FAILED:
        return _failure(reason, candidate.get("finishMessage"))
    for part in (candidate.get("content") or {}).get("parts") or []:
        if part.get("functionCall"):
            calls.append(_tool_call(part))
        elif isinstance(part.get("text"), str):
            if part.get("thought"):
                thinking += part["text"]
            else:
                text += part["text"]
    if feedback.get("blockReason"):
        refusal, finish = _blocked_prompt(feedback), "content_filter"
    elif reason in BLOCKED:
        refusal, finish = _why(reason, candidate.get("finishMessage")), "content_filter"
    else:
        finish = "tool_calls" if calls and reason in ("", "STOP") else _FINISH.get(reason, "stop")
    msg: dict[str, Any] = {"role": "assistant", "content": text if text or not (calls or refusal) else None}
    if thinking:
        msg["reasoning_content"] = thinking
    if refusal:
        msg["refusal"] = refusal
    if calls:
        msg["tool_calls"] = calls
    out: dict[str, Any] = {"id": f"chatcmpl-{resp.get('responseId') or uuid.uuid4().hex}", "object": "chat.completion",
                           "created": int(time.time()), "model": resp.get("modelVersion") or model,
                           "choices": [{"index": 0, "message": msg, "finish_reason": finish}]}
    usage = chat_usage(resp.get("usageMetadata"))
    if usage:
        out["usage"] = usage
    return out


class GeminiToChatStream:
    """Turns streamGenerateContent chunks (``alt=sse``) into Chat Completions chunks. Gemini has no closing event, so
    the finish reason and usage wait for the end of the stream (``finish_events``)."""

    def __init__(self, model: str):
        self.id = f"chatcmpl-{uuid.uuid4().hex}"
        self.model = model
        self.created = int(time.time())
        self.started = False
        self.calls = 0
        self.finish: str | None = None
        self.usage: dict = {}
        self.done = False

    def _chunk(self, delta: dict | None = None, finish: str | None = None, usage: dict | None = None) -> SSEEvent:
        d: dict[str, Any] = {"id": self.id, "object": "chat.completion.chunk", "created": self.created, "model": self.model,
                             "choices": [] if delta is None else [{"index": 0, "delta": delta, "finish_reason": finish}]}
        if usage is not None:
            d["usage"] = usage
        return event(d)

    def _end(self, error: dict) -> list[SSEEvent]:
        """The error, then the usage Gemini reported up to it: those tokens are billed, failed reply or not. The error
        comes first, so a stream that fails before any content still goes to the next source (failover.start_stream)."""
        self.done = True
        usage = chat_usage(self.usage)
        return [event(error), *([self._chunk(None, usage=usage)] if usage else []), SSEEvent(data="[DONE]")]

    def feed(self, ev: SSEEvent) -> list[SSEEvent]:
        if self.done:
            return []
        d = ev.json()
        if not d:
            return []
        if d.get("error"):  # an error after the stream started: quota, overload
            return self._end({"error": _error(d["error"])})
        out: list[SSEEvent] = []
        if not self.started:
            self.started = True
            if d.get("responseId"):
                self.id = f"chatcmpl-{d['responseId']}"
            self.model = d.get("modelVersion") or self.model
            out.append(self._chunk({"role": "assistant", "content": ""}))
        for k, v in (d.get("usageMetadata") or {}).items():  # counts are cumulative; keep the largest of each
            self.usage[k] = max(self.usage.get(k) or 0, v) if isinstance(v, int) else v
        feedback = d.get("promptFeedback") or {}
        if feedback.get("blockReason"):
            out.append(self._chunk({"refusal": _blocked_prompt(feedback)}))
            self.finish = "content_filter"
        candidate = (d.get("candidates") or [{}])[0] or {}
        reason = candidate.get("finishReason") or ""
        if reason in FAILED:
            return out + self._end(_failure(reason, candidate.get("finishMessage")))
        for part in (candidate.get("content") or {}).get("parts") or []:
            if part.get("functionCall"):
                call = _tool_call(part)
                out.append(self._chunk({"tool_calls": [{"index": self.calls, **call}]}))
                self.calls += 1
            elif part.get("text"):
                out.append(self._chunk({"reasoning_content" if part.get("thought") else "content": part["text"]}))
        if reason in BLOCKED:
            out.append(self._chunk({"refusal": _why(reason, candidate.get("finishMessage"))}))
            self.finish = "content_filter"
        elif reason and self.finish != "content_filter":
            self.finish = "tool_calls" if self.calls and reason == "STOP" else _FINISH.get(reason, "stop")
        return out

    def finish_events(self) -> list[SSEEvent]:
        if self.done:
            return []
        self.done = True
        out = [] if self.started else [self._chunk({"role": "assistant", "content": ""})]
        out.append(self._chunk({}, self.finish or ("tool_calls" if self.calls else "stop")))
        usage = chat_usage(self.usage)
        if usage:
            out.append(self._chunk(None, usage=usage))
        return out + [SSEEvent(data="[DONE]")]


async def to_chat_stream(events, model: str):
    conv = GeminiToChatStream(model)
    async for ev in events:
        for out in conv.feed(ev):
            yield out
    for out in conv.finish_events():
        yield out


def retry_delay(body: dict | None) -> float | None:
    """Seconds Google asks us to wait, from the RetryInfo in a 429's details (``"retryDelay": "17s"``): Google says it
    there rather than in a Retry-After header."""
    details = ((body or {}).get("error") or {}).get("details") if isinstance((body or {}).get("error"), dict) else None
    for d in details or []:
        delay = d.get("retryDelay") if isinstance(d, dict) else None
        if isinstance(delay, str) and delay.endswith("s"):
            try:
                return max(0.0, float(delay[:-1]))
            except ValueError:
                return None
    return None


# ---------------------------------------------------------------- embeddings


def embed_inputs(body: dict) -> list[str]:
    """An OpenAI embeddings request's inputs, one text each (a Gemini request gives one embedding, so each is sent alone)."""
    raw = body.get("input")
    items = raw if isinstance(raw, list) and raw and not isinstance(raw[0], int) else [raw]
    return [x if isinstance(x, str) else json.dumps(x) for x in items]


def embed_request(text: str, body: dict) -> dict:
    """One EmbedContentRequest (Gemini API v1beta) for one input; ``dimensions`` becomes its output dimensionality."""
    req: dict = {"content": {"parts": [{"text": text}]}}
    if body.get("model"):
        req["model"] = f"models/{model_id(str(body["model"]))}"
    if isinstance(body.get("dimensions"), int):
        req["outputDimensionality"] = body["dimensions"]
    return req


def _embed_tokens(resp: dict) -> int:
    """Tokens an embedding read: ``usageMetadata.promptTokenCount`` (the reference), else ``tokenCount`` (what batch output
    lines have carried). 0 when neither is there."""
    n = (resp.get("usageMetadata") or {}).get("promptTokenCount") or resp.get("tokenCount") or 0
    try:
        return int(n)
    except (TypeError, ValueError):
        return 0


def embeddings_reply(responses: list[dict], model: str, tokens: int | None = None) -> dict:
    """Gemini embeddings (EmbedContentResponses, or the embeddings of a batchEmbedContents reply) as OpenAI's embeddings
    reply. ``tokens``: the count when the replies give none per embedding."""
    data, used = [], 0
    for i, r in enumerate(responses):
        emb = r.get("embedding") if isinstance(r.get("embedding"), dict) else r
        data.append({"object": "embedding", "index": i, "embedding": list(emb.get("values") or [])})
        used += _embed_tokens(r)
    n = used or tokens or 0
    return {"object": "list", "data": data, "model": model, "usage": {"prompt_tokens": n, "total_tokens": n}}
