"""OpenAI-format backends with `api: gemini`: Chat Completions requests go to Gemini's native generateContent (the
Gemini API or Vertex AI) and come back as Chat Completions, thought summaries, signatures, blocks and usage included.
The payloads follow Google's API reference (November 2026); nothing here calls Google."""

import copy
import json
import time

import httpx
import pytest
from conftest import TEST_CONFIG
from pydantic import ValidationError
from test_http_backend import OPENAI_OK, Upstream, chunks, collect, post, stream_response

from meritorder.app import build_gateway
from meritorder.backends.auth import BackendAuthProvider
from meritorder.backends.base import BackendError
from meritorder.backends.http import GeminiBackend
from meritorder.catalog import price_source
from meritorder.config import BackendConfig, build_config, effort_format
from meritorder.formats.gemini_upstream import (SKIP_SIGNATURE, GeminiToChatStream, chat_to_gemini_request, chat_usage,
                                                gemini_to_chat_completion, model_id, retry_delay, thinking_config)
from meritorder.formats.sse import SSEEvent, event, iter_sse
from meritorder.state import MemoryState
from meritorder.store import MemoryStore

TOOL = {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
                                         "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                                                        "required": ["city"], "additionalProperties": False}}}
SIG = "CiQBjz1rX8hKpZ0q3Vb2mTq9yR1x7LbN0e4cWf6sHt2uJg=="
USAGE = {"promptTokenCount": 1520, "candidatesTokenCount": 18, "totalTokenCount": 1702, "cachedContentTokenCount": 1024,
         "thoughtsTokenCount": 164, "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 1520}],
         "cacheTokensDetails": [{"modality": "TEXT", "tokenCount": 1024}]}
CALL_REPLY = {
    "candidates": [{"content": {"role": "model", "parts": [
        {"text": "**Checking the weather**\n\nThe user wants Paris; I'll call get_weather.", "thought": True},
        {"functionCall": {"id": "fc_8x2k", "name": "get_weather", "args": {"city": "Paris"}}, "thoughtSignature": SIG}]},
        "finishReason": "STOP", "index": 0}],
    "usageMetadata": USAGE, "modelVersion": "gemini-4-carbon", "responseId": "kQ3zaN7xKqXXz7IPq8"}
TEXT_REPLY = {"candidates": [{"content": {"role": "model", "parts": [{"text": "Sunny, 21°C."}, {"text": "", "thoughtSignature": "c2ln"}]},
                              "finishReason": "STOP", "index": 0}],
              "usageMetadata": {"promptTokenCount": 40, "candidatesTokenCount": 6, "totalTokenCount": 46}, "modelVersion": "gemini-4-carbon"}
QUOTA = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
                   "message": "You exceeded your current quota, please check your plan and billing details.",
                   "details": [{"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [
                       {"quotaMetric": "generativelanguage.googleapis.com/generate_content_paid_tier_input_token_count",
                        "quotaId": "GenerateContentPaidTierInputTokensPerModelPerMinute",
                        "quotaDimensions": {"location": "global", "model": "gemini-4-carbon"}}]},
                       {"@type": "type.googleapis.com/google.rpc.Help", "links": [{"url": "https://ai.google.dev/gemini-api/docs/rate-limits"}]},
                       {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "23s"}]}}
OVERLOADED = {"error": {"code": 503, "message": "The model is overloaded. Please try again later.", "status": "UNAVAILABLE"}}


def sse(*chunks_: dict) -> bytes:
    return b"".join(f"data: {json.dumps(c)}\r\n\r\n".encode() for c in chunks_)


# ------------------------------------------------------------------ requests


def test_request_covers_roles_system_media_tools_results_and_generation_config():
    body = {"model": "gemini-4-carbon", "max_tokens": 900, "temperature": 0.4, "top_p": 0.9, "seed": 7, "stop": "END",
            "tools": [TOOL, {"type": "custom", "custom": {"name": "shell"}}], "tool_choice": "required", "reasoning_effort": "high",
            "messages": [
                {"role": "system", "content": "Be brief."},
                {"role": "developer", "content": [{"type": "text", "text": "Use metric units."}]},
                {"role": "user", "content": [
                    {"type": "text", "text": "What's the weather where this was taken?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
                    {"type": "image_url", "image_url": {"url": "https://example.com/photos/eiffel.webp"}},
                    {"type": "image_url", "image_url": {"url": "https://example.com/render?id=4"}},
                    {"type": "input_audio", "input_audio": {"data": "UklGRg==", "format": "mp3"}},
                    {"type": "file", "file": {"file_data": "data:application/pdf;base64,JVBERi0=", "filename": "trip.pdf"}},
                    {"type": "file", "file": {"file_data": "JVBERi0x", "filename": "notes.pdf"}}]},
                {"role": "assistant", "content": "Checking.", "tool_calls": [
                    {"id": "fc_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"},
                     "extra_content": {"google": {"thought_signature": SIG}}},
                    {"id": "fc_2", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":\"Lyon\"}"}}]},
                {"role": "tool", "tool_call_id": "fc_1", "content": "{\"temp_c\": 21, \"sky\": \"clear\"}"},
                {"role": "tool", "tool_call_id": "fc_2", "content": [{"type": "text", "text": "rain"}]},
                {"role": "user", "content": "And tomorrow?"},
            ]}
    out = chat_to_gemini_request(body)
    assert "model" not in out and "messages" not in out
    assert out["systemInstruction"] == {"parts": [{"text": "Be brief."}, {"text": "Use metric units."}]}
    user, model, results = out["contents"]
    assert user == {"role": "user", "parts": [
        {"text": "What's the weather where this was taken?"},
        {"inlineData": {"mimeType": "image/png", "data": "iVBORw0KGgo="}},
        {"fileData": {"mimeType": "image/webp", "fileUri": "https://example.com/photos/eiffel.webp"}},
        {"fileData": {"mimeType": "image/jpeg", "fileUri": "https://example.com/render?id=4"}},
        {"inlineData": {"mimeType": "audio/mp3", "data": "UklGRg=="}},
        {"inlineData": {"mimeType": "application/pdf", "data": "JVBERi0="}},
        {"inlineData": {"mimeType": "application/pdf", "data": "JVBERi0x"}}]}
    assert model == {"role": "model", "parts": [
        {"text": "Checking."},
        {"functionCall": {"name": "get_weather", "args": {"city": "Paris"}, "id": "fc_1"}, "thoughtSignature": SIG},
        {"functionCall": {"name": "get_weather", "args": {"city": "Lyon"}, "id": "fc_2"}}]}  # only the first of a step is signed
    # Both results and the next question go in one user turn, in order, each named after the call it answers.
    assert results == {"role": "user", "parts": [
        {"functionResponse": {"name": "get_weather", "response": {"temp_c": 21, "sky": "clear"}, "id": "fc_1"}},
        {"functionResponse": {"name": "get_weather", "response": {"output": "rain"}, "id": "fc_2"}},
        {"text": "And tomorrow?"}]}
    assert out["tools"] == [{"functionDeclarations": [{"name": "get_weather", "description": "Current weather for a city",
                                                       "parametersJsonSchema": TOOL["function"]["parameters"]}]}]
    assert out["toolConfig"] == {"functionCallingConfig": {"mode": "ANY"}}
    assert out["generationConfig"] == {"maxOutputTokens": 900, "temperature": 0.4, "topP": 0.9, "seed": 7, "stopSequences": ["END"],
                                       "thinkingConfig": {"includeThoughts": True, "thinkingLevel": "high"}}


def test_request_tool_choice_json_modes_and_signatures_from_elsewhere():
    named = chat_to_gemini_request({"messages": [{"role": "user", "content": "hi"}], "tools": [TOOL],
                                    "tool_choice": {"type": "function", "function": {"name": "get_weather"}}})
    assert named["toolConfig"] == {"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": ["get_weather"]}}
    assert chat_to_gemini_request({"messages": [], "tools": [TOOL], "tool_choice": "none"})["toolConfig"]["functionCallingConfig"] == {"mode": "NONE"}
    assert chat_to_gemini_request({"messages": [], "tools": [TOOL], "tool_choice": "auto"})["toolConfig"]["functionCallingConfig"] == {"mode": "AUTO"}
    assert "toolConfig" not in chat_to_gemini_request({"messages": [], "tools": [TOOL]})
    assert "tools" not in chat_to_gemini_request({"messages": [], "tools": [{"type": "web_search"}]})
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    js = chat_to_gemini_request({"messages": [], "max_completion_tokens": 50,
                                 "response_format": {"type": "json_schema", "json_schema": {"name": "a", "schema": schema, "strict": True}}})
    assert js["generationConfig"] == {"maxOutputTokens": 50, "responseMimeType": "application/json", "responseJsonSchema": schema}
    assert chat_to_gemini_request({"messages": [], "response_format": {"type": "json_object"}})["generationConfig"] == {"responseMimeType": "application/json"}
    assert chat_to_gemini_request({"messages": [{"role": "user", "content": ""}, {"role": "user", "content": None}]}) == {"contents": []}
    loose = chat_to_gemini_request({"messages": [
        {"role": "user", "content": ["plain", {"type": "image_url", "image_url": {"url": ""}}]},
        {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": {"already": "parsed"}}}]}]})["contents"]
    assert loose[0]["parts"] == [{"text": "plain"}] and loose[1]["parts"][0]["functionCall"]["args"] == {"already": "parsed"}

    # A session that was on Gemini through OpenRouter: its reasoning_details carry the signature for each call.
    history = [{"role": "user", "content": "go"},
               {"role": "assistant", "content": None, "tool_calls": [{"id": "tool_0", "type": "function", "function": {"name": "ls", "arguments": ""}}],
                "reasoning_details": [{"type": "reasoning.text", "text": "listing", "index": 0},
                                      {"type": "reasoning.encrypted", "data": "b3JzaWc=", "id": "tool_0", "format": "google-gemini-v1", "index": 1}]},
               {"role": "tool", "tool_call_id": "tool_0", "content": "a.py"},
               # Claude Code's history, translated: nothing to send back, so Google's documented bypass value.
               {"role": "assistant", "content": [{"type": "text", "text": "Reading it."}],
                "tool_calls": [{"id": "toolu_01", "type": "function", "function": {"name": "read", "arguments": "not json"}}]},
               {"role": "tool", "tool_call_id": "toolu_01", "name": "read", "content": "[1, 2]"},
               {"role": "tool", "tool_call_id": "unknown", "content": "orphan"}]
    out = chat_to_gemini_request({"messages": history})["contents"]
    assert out[1]["parts"] == [{"functionCall": {"name": "ls", "args": {}, "id": "tool_0"}, "thoughtSignature": "b3JzaWc="}]
    assert out[3]["parts"] == [{"text": "Reading it."}, {"functionCall": {"name": "read", "args": {"arguments": "not json"}, "id": "toolu_01"},
                                                        "thoughtSignature": SKIP_SIGNATURE}]
    assert out[4]["parts"] == [{"functionResponse": {"name": "read", "response": {"output": [1, 2]}, "id": "toolu_01"}},
                               {"functionResponse": {"name": "tool", "response": {"output": "orphan"}, "id": "unknown"}}]


@pytest.mark.parametrize("model,effort,want", [
    ("gemini-4-carbon", "low", {"includeThoughts": True, "thinkingLevel": "low"}),
    ("gemini-4-carbon", None, {"includeThoughts": True, "thinkingLevel": "medium"}),  # Google's guide: Gemini 4 needs a level
    ("models/gemini-4-argon", "max", {"includeThoughts": True, "thinkingLevel": "high"}),  # no xhigh on Argon
    ("gemini-4-carbon", "none", {"includeThoughts": True, "thinkingLevel": "low"}),  # Gemini 4 can't stop thinking
    ("gemini-4-carbon", "minimal", {"includeThoughts": True, "thinkingLevel": "low"}),
    ("gemini-4-hydrogen", "xhigh", {"includeThoughts": True, "thinkingLevel": "xhigh"}),
    ("gemini-4-hydrogen", None, {"includeThoughts": True, "thinkingLevel": "high"}),
    ("google/gemini-3.8-flash", "minimal", {"includeThoughts": True, "thinkingLevel": "low"}),
    ("gemini-3.5-flash-lite", "minimal", {"includeThoughts": True, "thinkingLevel": "minimal"}),
    ("gemini-3.5-flash-lite", None, {"includeThoughts": True}),
    ("gemini-3-pro-preview", "medium", {"includeThoughts": True, "thinkingLevel": "low"}),
    ("gemini-3.1-flash-lite-image", "medium", {"includeThoughts": True, "thinkingLevel": "minimal"}),
    ("gemini-5-neon", "xhigh", {"includeThoughts": True, "thinkingLevel": "high"}),  # not listed yet: the common levels
    ("gemini-4-carbon", "turbo", {"includeThoughts": True, "thinkingLevel": "medium"}),  # no such level: as if unset
    ("gemini-2.5-flash", "none", {"includeThoughts": True, "thinkingBudget": 0}),
    ("gemini-2.5-pro", "none", {"includeThoughts": True, "thinkingBudget": 128}),
    ("gemini-2.5-pro", "max", {"includeThoughts": True, "thinkingBudget": 32768}),
    ("gemini-2.5-flash-lite", "minimal", {"includeThoughts": True, "thinkingBudget": 512}),
    ("gemini-2.5-flash", "medium", {"includeThoughts": True, "thinkingBudget": 8192}),
    ("gemini-2.5-flash", None, {"includeThoughts": True}),
    ("gemma-4-31b-it", None, None),
    ("gemini-2.0-flash", None, None),  # doesn't think
    ("gemma-4-31b-it", "high", {"thinkingLevel": "high"}),
    ("gemma-4-31b-it", "none", None),  # it doesn't think unless asked: "none" mustn't turn thinking on
    ("gemini-2.0-flash", "none", None),
])
def test_thinking_follows_each_models_levels(model, effort, want):
    assert thinking_config(model, effort) == want


def test_effort_in_openrouter_form_and_model_ids():
    out = chat_to_gemini_request({"model": "gemini-4-carbon", "messages": [], "reasoning": {"effort": "low"}})
    assert out["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "low"
    odd = chat_to_gemini_request({"model": "gemini-4-carbon", "messages": [], "reasoning": True})
    assert odd["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "medium"
    assert model_id("google/gemini-4-carbon") == model_id("models/gemini-4-carbon") == "gemini-4-carbon"


# ------------------------------------------------------------------ replies


def test_reply_with_thoughts_a_signed_call_and_usage():
    out = gemini_to_chat_completion(CALL_REPLY, "gemini-4-carbon")
    choice = out["choices"][0]
    assert out["object"] == "chat.completion" and out["id"] == "chatcmpl-kQ3zaN7xKqXXz7IPq8" and out["model"] == "gemini-4-carbon"
    assert choice["finish_reason"] == "tool_calls" and choice["message"]["content"] is None
    assert choice["message"]["reasoning_content"].startswith("**Checking the weather**")
    assert choice["message"]["tool_calls"] == [{"id": "fc_8x2k", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"},
                                                "extra_content": {"google": {"thought_signature": SIG}}}]
    # Thinking is billed as output; promptTokenCount already includes the cached tokens.
    assert out["usage"] == {"prompt_tokens": 1520, "completion_tokens": 182, "total_tokens": 1702,
                            "prompt_tokens_details": {"cached_tokens": 1024}, "completion_tokens_details": {"reasoning_tokens": 164}}
    text = gemini_to_chat_completion(TEXT_REPLY, "x")["choices"][0]
    assert text == {"index": 0, "message": {"role": "assistant", "content": "Sunny, 21°C."}, "finish_reason": "stop"}
    cut = gemini_to_chat_completion({"candidates": [{"content": {"parts": [{"text": "Once upon"}]}, "finishReason": "MAX_TOKENS"}]}, "x")
    assert cut["choices"][0]["finish_reason"] == "length" and "usage" not in cut
    no_id = gemini_to_chat_completion({"candidates": [{"content": {"parts": [{"functionCall": {"name": "ls"}}]}, "finishReason": "STOP"}]}, "x")
    call = no_id["choices"][0]["message"]["tool_calls"][0]  # Gemini 2.5 sends no ids
    assert call["id"].startswith("call_") and call["function"]["arguments"] == "{}" and "extra_content" not in call
    assert chat_usage({"promptTokenCount": 10, "toolUsePromptTokenCount": 5, "candidatesTokenCount": 2})["prompt_tokens"] == 15
    assert chat_usage(None) is None


@pytest.mark.parametrize("reason", ["SAFETY", "RECITATION", "PROHIBITED_CONTENT", "SPII", "BLOCKLIST", "IMAGE_SAFETY"])
def test_safety_and_recitation_stops_are_refusals(reason):
    out = gemini_to_chat_completion({"candidates": [{"content": {"parts": [{"text": "Here is the full text of"}]}, "finishReason": reason,
                                                     "safetyRatings": [{"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "probability": "HIGH", "blocked": True}]}],
                                     "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 5, "totalTokenCount": 17}}, "x")
    choice = out["choices"][0]
    assert choice["finish_reason"] == "content_filter" and choice["message"]["refusal"] == f"Gemini stopped this reply ({reason})"
    assert choice["message"]["content"] == "Here is the full text of" and out["usage"]["prompt_tokens"] == 12


def test_a_blocked_prompt_is_a_refusal_and_broken_replies_are_errors():
    blocked = gemini_to_chat_completion({"promptFeedback": {"blockReason": "PROHIBITED_CONTENT", "blockReasonMessage": "Prompt was blocked."},
                                         "usageMetadata": {"promptTokenCount": 9, "totalTokenCount": 9}, "modelVersion": "gemini-4-carbon"}, "x")
    assert blocked["choices"][0]["finish_reason"] == "content_filter" and blocked["choices"][0]["message"]["content"] is None
    assert blocked["choices"][0]["message"]["refusal"] == "Gemini stopped the prompt (PROHIBITED_CONTENT): Prompt was blocked."
    malformed = gemini_to_chat_completion({"candidates": [{"finishReason": "MALFORMED_FUNCTION_CALL",
                                                           "finishMessage": "Malformed function call: print(default_api.ls("}]}, "x")
    assert malformed == {"error": {"message": "Gemini stopped this reply (MALFORMED_FUNCTION_CALL): Malformed function call: print(default_api.ls(",
                                   "type": "api_error", "code": 502}}
    assert gemini_to_chat_completion({"candidates": [{"finishReason": "MISSING_THOUGHT_SIGNATURE"}]}, "x")["error"]["code"] == 400
    assert gemini_to_chat_completion({"candidates": [{"finishReason": "PUP_LIMITED_DISABLED"}]}, "x")["error"]["code"] == 403
    busy = gemini_to_chat_completion({"error": {"message": "busy", "status": "UNAVAILABLE"}}, "x")  # no HTTP code: from the status
    assert busy == {"error": {"message": "busy", "type": "api_error", "code": 503}}
    assert retry_delay(QUOTA) == 23.0 and retry_delay(OVERLOADED) is None and retry_delay(None) is None
    assert retry_delay({"error": {"details": [{"retryDelay": "soon s"}]}}) is None


def feed_all(conv: GeminiToChatStream, *chunks_: dict) -> list[dict]:
    out = [e for c in chunks_ for e in conv.feed(event(c))] + conv.finish_events()
    assert out[-1].data == "[DONE]"
    return [e.json() for e in out[:-1]]


STREAM = [
    {"candidates": [{"content": {"role": "model", "parts": [{"text": "**Planning**\n\nLooking at the layout first.", "thought": True}]}, "index": 0}],
     "usageMetadata": {"promptTokenCount": 2100, "totalTokenCount": 2190, "thoughtsTokenCount": 90}, "modelVersion": "gemini-4-carbon", "responseId": "r-77"},
    {"candidates": [{"content": {"role": "model", "parts": [{"text": "I'll read the "}]}, "index": 0}],
     "usageMetadata": {"promptTokenCount": 2100, "candidatesTokenCount": 4, "totalTokenCount": 2284, "thoughtsTokenCount": 180}},
    {"candidates": [{"content": {"role": "model", "parts": [{"text": "file."}]}, "index": 0}]},
    {"candidates": [{"content": {"role": "model", "parts": [{"functionCall": {"id": "fc_1", "name": "Read", "args": {"path": "a.py"}},
                                                            "thoughtSignature": "c2lnLTE="}]}, "finishReason": "STOP", "index": 0}],
     "usageMetadata": {"promptTokenCount": 2100, "candidatesTokenCount": 30, "totalTokenCount": 2310, "cachedContentTokenCount": 2048,
                       "thoughtsTokenCount": 180}},
]


def test_stream_thoughts_text_a_signed_call_then_finish_and_usage():
    data = feed_all(GeminiToChatStream("gemini-4-carbon"), *STREAM, {"candidates": [{"content": {"parts": [{"text": "after"}]}}]})
    assert all(d["id"] == "chatcmpl-r-77" and d["object"] == "chat.completion.chunk" for d in data)
    deltas = [d["choices"][0]["delta"] for d in data if d["choices"]]
    assert deltas[0] == {"role": "assistant", "content": ""}
    assert deltas[1] == {"reasoning_content": "**Planning**\n\nLooking at the layout first."}
    assert "".join(x.get("content") or "" for x in deltas) == "I'll read the file.after"
    calls = [x["tool_calls"][0] for x in deltas if x.get("tool_calls")]
    assert calls == [{"index": 0, "id": "fc_1", "type": "function", "function": {"name": "Read", "arguments": "{\"path\": \"a.py\"}"},
                      "extra_content": {"google": {"thought_signature": "c2lnLTE="}}}]
    assert data[-2]["choices"][0]["finish_reason"] == "tool_calls" and data[-1]["choices"] == []
    assert data[-1]["usage"] == {"prompt_tokens": 2100, "completion_tokens": 210, "total_tokens": 2310,
                                 "prompt_tokens_details": {"cached_tokens": 2048}, "completion_tokens_details": {"reasoning_tokens": 180}}


def test_stream_blocks_errors_and_an_empty_stream():
    blocked = feed_all(GeminiToChatStream("m"), {"promptFeedback": {"blockReason": "SAFETY"}, "usageMetadata": {"promptTokenCount": 8}})
    assert blocked[1]["choices"][0]["delta"] == {"refusal": "Gemini stopped the prompt (SAFETY)"}
    assert blocked[2]["choices"][0]["finish_reason"] == "content_filter"
    recited = feed_all(GeminiToChatStream("m"), {"candidates": [{"content": {"parts": [{"text": "It was the best"}]}}]},
                       {"candidates": [{"finishReason": "RECITATION", "index": 0}]}, {"candidates": [{"finishReason": "STOP"}]})
    assert [d["choices"][0]["delta"] for d in recited[1:3]] == [{"content": "It was the best"}, {"refusal": "Gemini stopped this reply (RECITATION)"}]
    assert recited[3]["choices"][0]["finish_reason"] == "content_filter"
    cut = feed_all(GeminiToChatStream("m"), {"candidates": [{"content": {"parts": [{"text": "a"}]}, "finishReason": "MAX_TOKENS"}]})
    assert cut[-1]["choices"][0]["finish_reason"] == "length"  # no usage: no usage chunk
    conv = GeminiToChatStream("m")
    out = conv.feed(event({"candidates": [{"finishReason": "MALFORMED_FUNCTION_CALL"}]}))
    assert out[-2].json()["error"]["code"] == 502 and out[-1].data == "[DONE]" and conv.finish_events() == [] and conv.feed(event(STREAM[0])) == []
    late = GeminiToChatStream("m")
    late.feed(event(STREAM[1]))
    err = late.feed(event(QUOTA))
    assert err[0].json() == {"error": {"message": QUOTA["error"]["message"], "type": "api_error", "code": 429}} and err[2].data == "[DONE]"
    assert err[1].json()["usage"]["prompt_tokens"] == 2100  # what Gemini reported before it failed is still billed
    assert GeminiToChatStream("m").feed(SSEEvent(data="not json")) == []
    empty = feed_all(GeminiToChatStream("m"))
    assert empty[0]["choices"][0]["delta"]["role"] == "assistant" and empty[1]["choices"][0]["finish_reason"] == "stop"


# ------------------------------------------------------------------ the backend


def backend(upstream: Upstream, **cfg) -> GeminiBackend:
    class Bearer(BackendAuthProvider):
        async def apply(self, method, url, headers, body):
            return {**headers, "authorization": "Bearer ya29.token"}

    defaults = {"name": "gm", "kind": "api", "format": "openai", "api": "gemini", "base_url": "https://generativelanguage.example/v1beta"}
    b = GeminiBackend(BackendConfig(**{**defaults, **cfg}), Bearer())
    b.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    return b


async def test_urls_for_the_gemini_api_and_vertex():
    up = Upstream(httpx.Response(200, json=TEXT_REPLY))
    out = await backend(up).send({"model": "models/gemini-4-carbon", "max_tokens": 100, "messages": [{"role": "user", "content": "hi"}]}, False, {})
    assert str(up.requests[0].url) == "https://generativelanguage.example/v1beta/models/gemini-4-carbon:generateContent"
    assert out.body["choices"][0]["message"]["content"] == "Sunny, 21°C." and up.body()["generationConfig"]["maxOutputTokens"] == 100

    up = Upstream(lambda req: stream_response(sse(*STREAM)))
    vx = backend(up, flavor="vertex", project="proj-1", region="us-central1", base_url="")
    out = await vx.send({"model": "google/gemini-4-argon", "max_tokens": 100, "messages": [{"role": "user", "content": "hi"}]}, True, {})
    assert str(up.requests[0].url) == ("https://us-central1-aiplatform.googleapis.com/v1/projects/proj-1/locations/us-central1"
                                       "/publishers/google/models/gemini-4-argon:streamGenerateContent?alt=sse")
    assert up.requests[0].headers["authorization"] == "Bearer ya29.token"
    events = [e async for e in out.events]
    assert events[-1].data == "[DONE]" and any((e.json() or {}).get("usage") for e in events)
    assert vx._url("gemini-4-argon", False) == ("https://us-central1-aiplatform.googleapis.com/v1/projects/proj-1/locations/us-central1"
                                                "/publishers/google/models/gemini-4-argon:generateContent")
    glob = backend(up, flavor="vertex", project="p", base_url="")
    assert glob._url("gemini-4-carbon", False) == ("https://aiplatform.googleapis.com/v1/projects/p/locations/global"
                                                   "/publishers/google/models/gemini-4-carbon:generateContent")
    assert hasattr(vx, "embed")  # embeddings on the Gemini API (on Vertex AI it says it can't)


async def test_a_long_reply_streams_and_is_assembled_with_its_signature():
    up = Upstream(lambda req: stream_response(sse(*STREAM)))
    out = await backend(up).send({"model": "gemini-4-carbon", "max_tokens": 200_000, "messages": [{"role": "user", "content": "go"}]}, False, {})
    assert str(up.requests[0].url).endswith(":streamGenerateContent?alt=sse") and out.events is None
    msg = out.body["choices"][0]["message"]
    assert msg["content"] == "I'll read the file." and msg["reasoning_content"].startswith("**Planning**")
    assert msg["tool_calls"][0]["extra_content"] == {"google": {"thought_signature": "c2lnLTE="}}
    assert out.body["choices"][0]["finish_reason"] == "tool_calls" and out.body["usage"]["prompt_tokens_details"]["cached_tokens"] == 2048
    # A provider that answers a long request with plain JSON is read as the reply it is.
    up = Upstream(httpx.Response(200, json=TEXT_REPLY))
    out = await backend(up).send({"model": "gemini-4-carbon", "messages": [{"role": "user", "content": "go"}]}, False, {})
    assert out.body["choices"][0]["message"]["content"] == "Sunny, 21°C."


async def test_google_retry_info_becomes_retry_after():
    with pytest.raises(BackendError) as e:
        await backend(Upstream(httpx.Response(429, json=QUOTA))).send({"model": "gemini-4-carbon", "max_tokens": 5, "messages": []}, False, {})
    assert e.value.status == 429 and e.value.retryable and e.value.retry_after == 23.0 and "exceeded your current quota" in e.value.message
    with pytest.raises(BackendError) as e:
        await backend(Upstream(httpx.Response(503, json=OVERLOADED, headers={"retry-after": "4"}))).send(
            {"model": "gemini-4-carbon", "max_tokens": 5, "messages": []}, False, {})
    assert e.value.status == 503 and e.value.retry_after == 4.0  # a header, when there is one, wins


def test_config_validation_effort_and_prices():
    with pytest.raises(ValidationError, match="api: gemini needs format: openai"):
        BackendConfig(name="x", format="anthropic", api="gemini")
    gm = BackendConfig(name="gm", format="openai", api="gemini", base_url="https://generativelanguage.googleapis.com/v1beta")
    vx = BackendConfig(name="vx", format="openai", api="gemini", flavor="vertex", project="p")
    assert effort_format(gm) == effort_format(vx) == "openai"
    assert price_source(gm) == ("litellm", "gemini/") and price_source(vx) == ("litellm", "vertex_ai/")


# ------------------------------------------------------------------ through the gateway


def gateway(monkeypatch, gemini: Upstream, fallback: Upstream | None = None):
    """Model 'g' on a native Gemini backend 'gm' (cheaper, tried first) and an OpenAI-compatible route 'or'; model 'h'
    (pricier, as good) only on 'or', for `auto` to fall back to."""
    monkeypatch.setenv("MO_TEST_GEMINI", "key-gm")
    monkeypatch.setenv("MO_TEST_OR", "key-or")
    raw = copy.deepcopy(TEST_CONFIG)
    raw["backends"] = {
        "gm": {"kind": "api", "format": "openai", "api": "gemini", "base_url": "https://generativelanguage.example/v1beta",
               "auth": {"type": "api_key", "header": "x-goog-api-key", "value_env": "MO_TEST_GEMINI"}},
        "or": {"kind": "api", "format": "openai", "base_url": "https://or.example/v1", "auth": {"type": "api_key", "value_env": "MO_TEST_OR"}},
    }
    raw["models"] = {
        "g": {"quality": {"coding": 4, "general": 4}, "effort_levels": ["low", "medium", "high"], "max_output_tokens": 8000, "routes": [
            {"backend": "gm", "model": "gemini-4-carbon", "price": {"input": 2, "output": 10, "cache_read": 0.1}},
            {"backend": "or", "model": "google/gemini-4-carbon", "price": {"input": 2.2, "output": 11}}]},
        "h": {"quality": {"coding": 4, "general": 4}, "max_output_tokens": 8000,
              "routes": [{"backend": "or", "model": "other-1", "price": {"input": 5, "output": 25}}]},
    }
    raw["baseline_model"] = "h"
    gw = build_gateway(build_config(raw, quiet=True), store=MemoryStore(), state=MemoryState())
    gw.backends["gm"].client = httpx.AsyncClient(transport=httpx.MockTransport(gemini))
    gw.backends["or"].client = httpx.AsyncClient(transport=httpx.MockTransport(fallback or Upstream(httpx.Response(200, json=OPENAI_OK))))
    return gw


async def test_tools_round_trip_keeps_the_signature_and_records_cached_and_thinking_tokens(monkeypatch):
    up = Upstream(httpx.Response(200, json=CALL_REPLY), httpx.Response(200, json=TEXT_REPLY))
    gw = gateway(monkeypatch, up)
    assert isinstance(gw.backends["gm"], GeminiBackend)
    ask = [{"role": "user", "content": "Weather in Paris?"}]
    r = await post(gw, {"model": "g", "max_tokens": 500, "tools": [TOOL], "messages": ask})
    assert r.status_code == 200 and r.headers["x-meritorder-backend"] == "gm"
    reply = r.json()["choices"][0]["message"]
    assert reply["tool_calls"][0]["extra_content"]["google"]["thought_signature"] == SIG and reply["reasoning_content"]
    sent = up.requests[0]
    assert sent.headers["x-goog-api-key"] == "key-gm" and "authorization" not in sent.headers
    assert str(sent.url) == "https://generativelanguage.example/v1beta/models/gemini-4-carbon:generateContent"
    assert up.body(0)["tools"][0]["functionDeclarations"][0]["name"] == "get_weather" and up.body(0)["generationConfig"]["maxOutputTokens"] == 500

    # The client sends the assistant message back as it got it, with the tool's result.
    r = await post(gw, {"model": "g", "max_tokens": 500, "tools": [TOOL], "reasoning_effort": "low",
                        "messages": ask + [reply, {"role": "tool", "tool_call_id": "fc_8x2k", "content": "{\"temp_c\": 21}"}]})
    assert r.json()["choices"][0]["message"]["content"] == "Sunny, 21°C."
    turn = up.body(1)["contents"]
    assert turn[1] == {"role": "model", "parts": [{"functionCall": {"name": "get_weather", "args": {"city": "Paris"}, "id": "fc_8x2k"},
                                                   "thoughtSignature": SIG}]}
    assert turn[2] == {"role": "user", "parts": [{"functionResponse": {"name": "get_weather", "response": {"temp_c": 21}, "id": "fc_8x2k"}}]}
    assert up.body(1)["generationConfig"]["thinkingConfig"] == {"includeThoughts": True, "thinkingLevel": "low"}
    await gw.drain()
    first = (await gw.store.recent(2))[-1]
    assert first["input_tokens"] == 496 and first["cache_read_tokens"] == 1024 and first["output_tokens"] == 182
    assert first["actual_cost"] == pytest.approx((496 * 2 + 1024 * 0.1 + 182 * 10) / 1e6, abs=1e-6)  # stored to the microdollar


async def test_claude_code_stream_through_gemini(monkeypatch):
    up = Upstream(lambda req: stream_response(*(lambda b: [b[i:i + 97] for i in range(0, len(b), 97)])(sse(*STREAM))))
    gw = gateway(monkeypatch, up)
    tools = [{"name": "Read", "description": "Read a file", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}}]
    r = await post(gw, {"model": "g", "max_tokens": 4000, "stream": True, "tools": tools, "output_config": {"effort": "high"},
                        "messages": [{"role": "user", "content": "read a.py"},
                                     {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"path": "b.py"}}]},
                                     {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "x = 1"}]}]},
                   path="/v1/messages")
    assert r.status_code == 200
    sent = up.body()
    assert str(up.requests[0].url).endswith("/models/gemini-4-carbon:streamGenerateContent?alt=sse")
    assert sent["contents"][1]["parts"][0]["thoughtSignature"] == SKIP_SIGNATURE  # Claude Code can't carry Gemini's signature
    assert sent["contents"][2]["parts"][0]["functionResponse"]["name"] == "Read"
    assert sent["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "high"  # Claude Code's effort, carried over
    evs = await collect(iter_sse(chunks(r.content)))
    kinds = [e.event for e in evs]
    assert kinds[0] == "message_start" and kinds[-1] == "message_stop"
    text = "".join((e.json() or {}).get("delta", {}).get("text", "") for e in evs if e.event == "content_block_delta")
    starts = [e.json()["content_block"] for e in evs if e.event == "content_block_start"]
    assert text == "I'll read the file." and starts[-1]["type"] == "tool_use" and starts[-1]["id"] == "fc_1" and starts[-1]["name"] == "Read"
    assert any((e.json() or {}).get("delta", {}).get("stop_reason") == "tool_use" for e in evs)
    await gw.drain()
    row = (await gw.store.recent(1))[0]
    assert row["cache_read_tokens"] == 2048 and row["input_tokens"] == 52 and row["output_tokens"] == 210


@pytest.mark.parametrize("failure,status", [(httpx.Response(429, json=QUOTA), 429), (httpx.Response(503, json=OVERLOADED), 503)])
async def test_quota_and_overload_are_capacity_and_go_to_the_next_source(monkeypatch, failure, status):
    gw = gateway(monkeypatch, Upstream(failure))
    r = await post(gw, {"model": "g", "max_tokens": 50, "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200 and r.headers["x-meritorder-backend"] == "or"
    assert r.headers["x-meritorder-failed-over"] == f"gm/gemini-4-carbon {status}"
    outage = gw.backends["gm"].skipped(time.time(), "gemini-4-carbon")
    assert outage.capacity and outage.status == status and gw.backends["gm"].healthy(time.time())  # only that model, and busy
    if status == 429:
        assert 20 < outage.until - time.time() <= 23  # Google's retryDelay


async def test_out_of_quota_with_nowhere_else_passes_googles_wait_on(monkeypatch):
    gw = gateway(monkeypatch, Upstream(httpx.Response(429, json=QUOTA)))
    gw.cfg.model("g").routes = gw.cfg.model("g").routes[:1]  # only the Gemini API
    r = await post(gw, {"model": "g", "max_tokens": 50, "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 429 and r.headers["retry-after"] == "23" and "quota" in r.json()["error"]["message"]


@pytest.mark.parametrize("first", [QUOTA, {"candidates": [{"finishReason": "MALFORMED_FUNCTION_CALL", "index": 0}]}])
async def test_a_stream_that_fails_before_content_goes_to_the_next_source(monkeypatch, first):
    ok = (b'data: {"id":"c","choices":[{"index":0,"delta":{"content":"from or"}}]}\n\n'
          b'data: {"id":"c","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n')
    gw = gateway(monkeypatch, Upstream(lambda req: stream_response(sse(first))), Upstream(lambda req: stream_response(ok)))
    r = await post(gw, {"model": "g", "max_tokens": 50, "stream": True, "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200 and r.headers["x-meritorder-backend"] == "or" and "from or" in r.text


async def test_a_stream_that_fails_after_content_still_records_the_tokens_gemini_billed(monkeypatch):
    """A reply that breaks after it started (a malformed function call) reaches the client as an error, and its tokens
    are billed all the same: the ledger keeps what usageMetadata said up to then."""
    chunks_ = [{"candidates": [{"content": {"role": "model", "parts": [{"text": "Let me check "}]}, "index": 0}],
                "usageMetadata": {"promptTokenCount": 2100, "candidatesTokenCount": 4, "totalTokenCount": 2104}},
               {"candidates": [{"finishReason": "MALFORMED_FUNCTION_CALL", "index": 0}],
                "usageMetadata": {"promptTokenCount": 2100, "candidatesTokenCount": 40, "totalTokenCount": 2140}}]
    gw = gateway(monkeypatch, Upstream(lambda req: stream_response(sse(*chunks_))))
    r = await post(gw, {"model": "g", "max_tokens": 50, "stream": True, "messages": [{"role": "user", "content": "hi"}]})
    assert r.headers["x-meritorder-backend"] == "gm" and "MALFORMED_FUNCTION_CALL" in r.text and '"usage"' not in r.text
    await gw.drain()
    row = (await gw.store.recent(1))[0]
    assert row["input_tokens"] == 2100 and row["output_tokens"] == 40
    assert row["actual_cost"] == pytest.approx((2100 * 2 + 40 * 10) / 1e6, abs=1e-6)


async def test_blocks_are_refusals_auto_retries_and_a_named_model_returns(monkeypatch):
    blocked = {"promptFeedback": {"blockReason": "SAFETY"}, "usageMetadata": {"promptTokenCount": 9, "totalTokenCount": 9}}
    gw = gateway(monkeypatch, Upstream(lambda req: stream_response(sse(blocked)) if "alt=sse" in str(req.url) else httpx.Response(200, json=blocked)))
    body = {"model": "auto", "max_tokens": 50, "messages": [{"role": "user", "content": "how do locks work"}]}
    r = await post(gw, body, **{"x-route-floor": "general:3"})
    assert r.status_code == 200 and r.headers["x-meritorder-model"] == "h" and r.headers["x-meritorder-refused-by"] == "g"
    r = await post(gw, {**body, "stream": True}, **{"x-route-floor": "general:3"})
    assert r.headers["x-meritorder-model"] == "h"
    await gw.drain()
    assert (await gw.store.recent(2))[1]["input_tokens"] == 9  # the streamed refusal's prompt, billed, from its usage chunk
    named = await post(gw, {**body, "model": "g"})
    msg = named.json()["choices"][0]
    assert msg["finish_reason"] == "content_filter" and msg["message"]["refusal"] == "Gemini stopped the prompt (SAFETY)"
    claude = await post(gw, {**body, "model": "g"}, path="/v1/messages")
    assert claude.json()["stop_reason"] == "refusal"
    await gw.drain()
    assert any("refused" in (row.get("error") or "") and row["input_tokens"] == 9 for row in await gw.store.recent(10))
