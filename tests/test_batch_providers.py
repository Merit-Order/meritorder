"""Each provider's batch API against a fake of it (httpx.MockTransport), with payloads shaped like the providers' API
references of November 2026 (docs/02-request-flow.md#batches-work-that-can-wait says which parts still need a real account to confirm)."""

import json
import re
from urllib.parse import parse_qs, unquote, urlparse

import httpx
import pytest

from meritorder.backends.auth import BackendAuthProvider, make_auth
from meritorder.backends.base import BackendError
from meritorder.backends.http import AnthropicBackend, GeminiBackend, OpenAIBackend
from meritorder.batches.providers import (APIS, AnthropicBatches, AzureBatches, BedrockBatches, GeminiBatches, MistralBatches, MoonshotBatches,
                                          OpenAIBatches, OpenRouterBatches, Outcome, XaiBatches,
                                          VertexBatches, batch_api_for, split_bucket)
from meritorder.config import BackendConfig, BatchesConfig, batch_api, batch_discount

SETTINGS = BatchesConfig(retention_days=7)
ANTHROPIC_MSG = {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-x", "content": [{"type": "text", "text": "hi"}],
                 "stop_reason": "end_turn", "usage": {"input_tokens": 10, "output_tokens": 2}}
CHAT = {"id": "c1", "object": "chat.completion", "created": 1, "model": "m", "choices": [{"index": 0, "message": {"role": "assistant",
        "content": "hi"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}}
GEMINI_REPLY = {"candidates": [{"content": {"role": "model", "parts": [{"text": "hi"}]}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 2, "totalTokenCount": 12}, "modelVersion": "gemini-4-argon"}
JOB = {"id": "bj_1", "batch_id": "batch_1", "model": "m-1", "endpoint": "chat", "provider": {}}


class Fake:
    """A provider: routes (method, path regex) to handlers, and records every request."""

    def __init__(self):
        self.routes: list[tuple[str, re.Pattern, object]] = []
        self.requests: list[httpx.Request] = []

    def on(self, method: str, pattern: str, reply):
        self.routes.append((method, re.compile(pattern), reply))
        return self

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        target = request.url.raw_path.decode()  # as sent: an object name's slashes stay %2F
        for method, rx, reply in reversed(self.routes):
            if method == request.method and rx.search(target):
                return reply(request) if callable(reply) else reply
        return httpx.Response(404, json={"error": {"message": f"no route for {request.method} {target}"}})

    def sent(self, method: str, pattern: str) -> list[httpx.Request]:
        return [r for r in self.requests if r.method == method and re.search(pattern, r.url.raw_path.decode())]


def wire(cfg: dict, fake: Fake, auth=None):
    bc = BackendConfig.model_validate({"name": "b", **cfg})
    cls = AnthropicBackend if bc.format == "anthropic" else GeminiBackend if bc.api == "gemini" else OpenAIBackend
    b = cls(bc, auth or BackendAuthProvider())
    b.client = httpx.AsyncClient(transport=httpx.MockTransport(fake))
    return b, batch_api_for(b, SETTINGS)


def jsonl(*rows) -> bytes:
    return b"".join(json.dumps(r).encode() + b"\n" for r in rows)


def lines_of(content: bytes) -> list[dict]:
    return [json.loads(x) for x in content.decode().splitlines() if x.strip()]


async def every(api, job, want=None) -> dict[str, Outcome]:
    return {o.key: o async for o in api.results(job, want)}


# ---------------------------------------------------------------- which API a backend has


def test_batch_api_is_found_from_the_address_and_needs_a_bucket_where_the_provider_does():
    def api(**kw):
        return batch_api(BackendConfig.model_validate({"name": "x", **kw}))

    assert api(format="anthropic", base_url="https://api.anthropic.com") == ("anthropic", "")
    assert api(format="openai", base_url="https://api.openai.com/v1") == ("openai", "")
    assert api(format="openai", base_url="https://res.openai.azure.com/openai/v1") == ("azure", "")
    assert api(format="openai", api="gemini", base_url="https://generativelanguage.googleapis.com/v1beta") == ("gemini", "")
    assert api(kind="mock", format="openai") == ("mock", "")
    assert api(format="openai", base_url="https://openrouter.ai/api/v1") == ("openrouter", "")
    assert api(format="openai", base_url="https://api.mistral.ai/v1") == ("mistral", "")
    assert api(format="openai", base_url="https://api.moonshot.ai/v1") == ("moonshot", "")
    assert api(format="openai", base_url="https://api.x.ai/v1") == ("xai", "")
    for host in ("https://api.deepseek.com/v1", "https://api.z.ai/api/paas/v4", "https://api.minimax.io/v1"):  # none has one
        assert api(format="openai", base_url=host) == (None, "no batch API the gateway speaks at this address")
    assert "max_concurrency" in api(kind="self_hosted", format="openai", base_url="http://gpu")[1]
    assert api(kind="self_hosted", format="openai", base_url="http://gpu", max_concurrency=8) == ("gpu", "")
    assert "OpenAI-format" in api(kind="self_hosted", format="anthropic", base_url="http://gpu", max_concurrency=8)[1]
    assert batch_discount(BackendConfig.model_validate({"name": "x", "format": "openai", "base_url": "https://api.x.ai/v1"})) == 0.2
    assert batch_discount(BackendConfig.model_validate({"name": "x", "format": "openai", "base_url": "https://api.moonshot.ai/v1"})) == 0.4
    assert batch_discount(BackendConfig.model_validate({"name": "x", "format": "openai", "base_url": "https://api.mistral.ai/v1"})) == 0.5
    assert batch_discount(BackendConfig.model_validate({"name": "x", "format": "openai", "base_url": "https://api.x.ai/v1",
                                                        "batch": {"discount": 0.3}})) == 0.3 and batch_discount(None) == 0.5
    assert api(format="anthropic", base_url="https://api.anthropic.com", batch={"enabled": False}) == (None, "batch.enabled is false")
    bedrock = {"format": "anthropic", "base_url": "https://bedrock-mantle.us-east-1.api.aws/anthropic", "auth": {"type": "aws_sigv4"}}
    assert "batch.bucket" in api(**bedrock)[1]
    assert api(**bedrock, batch={"bucket": "s3://b/p", "role_arn": "arn:aws:iam::1:role/r"}) == ("bedrock", "")
    keyed = {**bedrock, "auth": {"type": "api_key", "value": "k"}}
    assert "AWS credentials" in api(**keyed, batch={"bucket": "s3://b", "role_arn": "r", "api": "bedrock"})[1]
    vertex = {"format": "anthropic", "flavor": "vertex", "project": "p", "auth": {"type": "google"}}
    assert "gs://" in api(**vertex)[1] and api(**vertex, batch={"bucket": "gs://b"}) == ("vertex", "")
    assert "native Gemini" in api(format="openai", base_url="https://x", batch={"api": "vertex", "bucket": "gs://b"})[1]
    assert api(format="openai", base_url="https://my.host/v1", batch={"api": "openai"}) == ("openai", "")
    with pytest.raises(ValueError, match="s3://bucket/prefix"):
        BackendConfig.model_validate({"name": "x", "format": "openai", "batch": {"bucket": "/tmp/x"}})
    assert split_bucket("s3://bucket/a/b/") == ("bucket", "a/b") and split_bucket("gs://bucket") == ("bucket", "")
    assert set(APIS) == {"anthropic", "openai", "azure", "moonshot", "mistral", "openrouter", "xai", "gemini", "bedrock", "vertex", "mock"}


# ---------------------------------------------------------------- Anthropic


async def test_anthropic_message_batches():
    results = jsonl({"custom_id": "r0", "result": {"type": "succeeded", "message": ANTHROPIC_MSG}},
                    {"custom_id": "r1", "result": {"type": "errored", "error": {"type": "error", "error": {"type": "invalid_request_error",
                                                                                                          "message": "bad"}}}},
                    {"custom_id": "r2", "result": {"type": "expired"}}, {"custom_id": "r3", "result": {"type": "canceled"}})
    fake = (Fake().on("POST", r"/v1/messages/batches$", httpx.Response(200, json={"id": "msgbatch_9", "processing_status": "in_progress"}))
            .on("GET", r"/v1/messages/batches/msgbatch_9$", httpx.Response(200, json={
                "id": "msgbatch_9", "processing_status": "ended", "results_url": "https://api.anthropic.com/v1/messages/batches/msgbatch_9/results",
                "request_counts": {"processing": 0, "succeeded": 1, "errored": 1, "canceled": 1, "expired": 1}}))
            .on("GET", r"/results$", httpx.Response(200, content=results))
            .on("POST", r"/cancel$", httpx.Response(200, json={"processing_status": "canceling"}))
            .on("DELETE", r"/msgbatch_9$", httpx.Response(200, json={"type": "message_batch_deleted"})))
    _, api = wire({"format": "anthropic", "base_url": "https://api.anthropic.com", "anthropic_version": "2026-10-21"}, fake)
    assert isinstance(api, AnthropicBatches) and not api.one_model and api.max_requests == 100_000
    sub = await api.submit(JOB, [("r0", {"model": "claude-x", "max_tokens": 5, "messages": [], "stream": True})])
    sent = json.loads(fake.requests[0].content)
    assert sub.provider_id == "msgbatch_9" and sent == {"requests": [{"custom_id": "r0", "params": {"model": "claude-x", "max_tokens": 5,
                                                                                                    "messages": []}}]}
    assert fake.requests[0].headers["anthropic-version"] == "2026-10-21"
    job = {**JOB, "provider_id": "msgbatch_9"}
    polled = await api.poll(job)
    assert polled.ended and polled.expired and polled.counts == {"processing": 0, "succeeded": 1, "errored": 1, "canceled": 1, "expired": 1}
    job["provider"] = polled.provider
    got = await every(api, job)
    assert got["r0"].status == "succeeded" and got["r0"].body["id"] == "msg_1"
    assert (got["r1"].status, got["r1"].code, got["r1"].kind, got["r1"].message) == ("errored", 400, "invalid_request_error", "bad")
    assert got["r2"].status == "expired" and got["r3"].status == "canceled"
    await api.cancel(job)
    await api.purge(job)
    assert [r.method for r in fake.requests[-2:]] == ["POST", "DELETE"]


async def test_provider_errors_say_what_failed_and_whether_to_retry():
    fake = Fake().on("POST", r"/v1/messages/batches$", httpx.Response(429, headers={"retry-after": "7"}, json={"error": {"message": "slow down"}}))
    _, api = wire({"format": "anthropic", "base_url": "https://api.anthropic.com"}, fake)
    with pytest.raises(BackendError) as e:
        await api.submit(JOB, [("r0", {"model": "m", "max_tokens": 1, "messages": []})])
    assert e.value.status == 429 and e.value.retryable and e.value.retry_after == 7 and "creating a message batch: slow down" in e.value.message

    def broken(request):
        raise httpx.ConnectError("refused", request=request)

    fake.on("GET", r"/v1/messages/batches/x$", broken).on("GET", r"/v1/messages/batches/y$", httpx.Response(200, content=b"not json"))
    with pytest.raises(BackendError, match="ConnectError"):
        await api.poll({**JOB, "provider_id": "x"})
    with pytest.raises(BackendError, match="isn't JSON"):
        await api.poll({**JOB, "provider_id": "y"})
    fake.on("GET", r"/results$", httpx.Response(500, content=b"down"))
    with pytest.raises(BackendError, match="down"):
        await every(api, {**JOB, "provider_id": "y"})

    class Exploding(BackendAuthProvider):
        async def apply(self, method, url, headers, body):
            raise RuntimeError("no credentials")

    _, api = wire({"format": "anthropic", "base_url": "https://api.anthropic.com"}, Fake(), Exploding())
    with pytest.raises(BackendError, match="RuntimeError: no credentials"):
        await api.cancel({**JOB, "provider_id": "x"})


# ---------------------------------------------------------------- OpenAI and Azure


def multipart_fields(request: httpx.Request) -> dict[str, bytes]:
    boundary = request.headers["content-type"].split("boundary=")[1]
    out = {}
    for part in request.content.split(f"--{boundary}".encode())[1:-1]:
        head, _, data = part.strip(b"\r\n").partition(b"\r\n\r\n")
        out[re.search(rb'name="([^"]+)"', head).group(1).decode()] = data
    return out


def openai_fake(status="completed", base="/v1") -> Fake:
    out = jsonl({"id": "batch_req_1", "custom_id": "r0", "response": {"status_code": 200, "request_id": "q", "body": CHAT}, "error": None})
    bad = {"error": {"message": "bad", "type": "invalid_request_error"}}
    err = jsonl({"id": "batch_req_2", "custom_id": "r1", "response": {"status_code": 400, "body": bad},
                 "error": None},
                {"id": "batch_req_3", "custom_id": "r2", "response": None, "error": {"code": "batch_expired", "message": "late"}},
                {"id": "batch_req_4", "custom_id": "r3", "response": None, "error": {"code": "batch_cancelled", "message": "stop"}},
                {"id": "batch_req_5", "custom_id": "r4", "response": None, "error": {"code": "server_error", "message": "boom"}})
    return (Fake().on("POST", rf"{base}/files$", httpx.Response(200, json={"id": "file-in", "object": "file"}))
            .on("POST", rf"{base}/batches$", httpx.Response(200, json={"id": "batch_9", "status": "validating"}))
            .on("GET", rf"{base}/batches/batch_9$", httpx.Response(200, json={
                "id": "batch_9", "status": status, "output_file_id": "file-out", "error_file_id": "file-err",
                "request_counts": {"total": 5, "completed": 1, "failed": 4},
                "errors": {"data": [{"code": "invalid_json_line", "message": "line 3 isn't JSON"}]} if status == "failed" else None}))
            .on("GET", r"/files/file-out/content$", httpx.Response(200, content=out))
            .on("GET", r"/files/file-err/content$", httpx.Response(200, content=err))
            .on("POST", r"/batches/batch_9/cancel$", httpx.Response(200, json={"status": "cancelling"}))
            .on("DELETE", r"/files/file-(in|out)$", httpx.Response(200, json={"deleted": True}))
            .on("DELETE", r"/files/file-err$", httpx.Response(404, json={"error": {"message": "gone"}})))


async def test_openai_batch_api_uploads_a_file_then_makes_the_batch():
    fake = openai_fake()
    _, api = wire({"format": "openai", "base_url": "https://api.openai.com/v1", "max_tokens_field": "max_completion_tokens"}, fake)
    assert isinstance(api, OpenAIBatches) and api.one_model and api.max_requests == 50_000
    sub = await api.submit(JOB, [("r0", {"model": "gpt-x", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 9, "stream": False})])
    assert sub.provider_id == "batch_9" and sub.provider == {"input_file_id": "file-in"}
    form = multipart_fields(fake.requests[0])
    assert form["purpose"] == b"batch" and form["expires_after[anchor]"] == b"created_at" and form["expires_after[seconds]"] == b"604800"
    assert lines_of(form["file"]) == [{"custom_id": "r0", "method": "POST", "url": "/v1/chat/completions",
                                       "body": {"model": "gpt-x", "messages": [{"role": "user", "content": "hi"}], "max_completion_tokens": 9}}]
    create = json.loads(fake.requests[1].content)
    assert create == {"input_file_id": "file-in", "endpoint": "/v1/chat/completions", "completion_window": "24h",
                      "metadata": {"meritorder_job": "bj_1", "meritorder_batch": "batch_1"},
                      "output_expires_after": {"anchor": "created_at", "seconds": 604800}}
    job = {**JOB, "provider_id": "batch_9", "provider": sub.provider}
    polled = await api.poll(job)
    assert polled.ended and polled.counts["succeeded"] == 1 and polled.counts["errored"] == 4
    job["provider"] = {**job["provider"], **polled.provider}
    got = await every(api, job)
    assert got["r0"].status == "succeeded" and got["r0"].body["choices"][0]["message"]["content"] == "hi"
    assert (got["r1"].code, got["r1"].kind, got["r1"].message) == (400, "invalid_request_error", "bad")
    assert got["r2"].status == "expired" and got["r3"].status == "canceled" and got["r4"].kind == "server_error"
    assert set(await every(api, job, "succeeded")) == {"r0"} and set(await every(api, job, "failed")) == {"r1", "r2", "r3", "r4"}
    await api.cancel(job)
    await api.purge(job)  # the error file is already gone (404): fine
    assert len(fake.sent("DELETE", r"/files/")) == 3


async def test_openai_failed_batch_and_a_create_that_fails_cleans_up_its_file():
    fake = openai_fake("failed")
    _, api = wire({"format": "openai", "base_url": "https://api.openai.com/v1"}, fake)
    polled = await api.poll({**JOB, "provider_id": "batch_9"})
    assert polled.failed == "line 3 isn't JSON"
    fake.on("POST", r"/v1/batches$", httpx.Response(400, json={"error": {"message": "model_not_found"}}))
    with pytest.raises(BackendError, match="model_not_found"):
        await api.submit(JOB, [("r0", {"model": "x", "messages": []})])
    assert fake.sent("DELETE", r"/files/file-in$")
    fake.on("DELETE", r"/files/file-in$", httpx.Response(500, json={"error": {"message": "down"}}))
    with pytest.raises(BackendError, match="model_not_found"):  # the cleanup failing doesn't hide why
        await api.submit(JOB, [("r0", {"model": "x", "messages": []})])
    with pytest.raises(BackendError, match="down"):
        await api.purge({**JOB, "provider": {"input_file_id": "file-in"}})


async def test_openai_responses_api_and_embeddings_lines():
    fake = openai_fake()
    _, api = wire({"format": "openai", "base_url": "https://api.openai.com/v1", "api": "responses"}, fake)
    await api.submit(JOB, [("r0", {"model": "gpt-x", "messages": [{"role": "user", "content": "hi"}]})])
    line = lines_of(multipart_fields(fake.requests[0])["file"])[0]
    assert line["url"] == "/v1/responses" and line["body"]["input"][0]["content"] == "hi"
    assert json.loads(fake.requests[1].content)["endpoint"] == "/v1/responses"
    reply = {"id": "resp_1", "output": [{"type": "message", "content": [{"type": "output_text", "text": "hey"}]}],
             "usage": {"input_tokens": 5, "output_tokens": 1}}
    fake.on("GET", r"/files/file-out/content$", httpx.Response(200, content=jsonl({"custom_id": "r0", "response": {"status_code": 200, "body": reply}})))
    got = await every(api, {**JOB, "provider": {"output_file_id": "file-out"}}, "succeeded")
    assert got["r0"].body["choices"][0]["message"]["content"] == "hey" and got["r0"].body["usage"]["prompt_tokens"] == 5
    await api.submit({**JOB, "endpoint": "embeddings"}, [("r0", {"model": "emb", "input": ["a"]})])
    line = lines_of(multipart_fields(fake.sent("POST", r"/files$")[-1])["file"])[0]
    assert line["url"] == "/v1/embeddings" and line["body"] == {"model": "emb", "input": ["a"]}


async def test_azure_global_batch_takes_its_own_endpoint_spelling_and_expiry_range():
    fake = openai_fake(base="/openai/v1")
    _, api = wire({"format": "openai", "base_url": "https://res.openai.azure.com/openai/v1"}, fake)
    assert isinstance(api, AzureBatches) and api.max_requests == 100_000
    await api.submit({**JOB, "model": "terra-batch"}, [("r0", {"model": "terra-batch", "messages": []})])
    form = multipart_fields(fake.requests[0])
    assert form["expires_after.anchor"] == b"created_at" and form["expires_after.seconds"] == b"1209600"  # Azure's least: 14 days
    assert lines_of(form["file"])[0]["url"] == "/v1/chat/completions"
    create = json.loads(fake.requests[1].content)
    assert create["endpoint"] == "/chat/completions" and create["output_expires_after"]["seconds"] == 1209600
    assert not api.embeddings  # Azure lists no embedding model for its batch deployments


async def test_moonshot_takes_openai_calls_with_a_window_in_days_and_no_sampling_fields():
    fake = openai_fake()
    _, api = wire({"format": "openai", "base_url": "https://api.moonshot.ai/v1"}, fake)
    assert isinstance(api, MoonshotBatches) and not api.embeddings and api.max_bytes == 100 * 1024 * 1024
    await api.submit(JOB, [("r0", {"model": "kimi-k3.1", "messages": [], "temperature": 0.2, "top_p": 1, "max_tokens": 9})])
    form = multipart_fields(fake.requests[0])
    assert set(form) == {"purpose", "file"}  # no expiry fields
    assert lines_of(form["file"])[0]["body"] == {"model": "kimi-k3.1", "messages": [], "max_tokens": 9}
    create = json.loads(fake.requests[1].content)
    assert create["completion_window"] == "1d" and create["endpoint"] == "/v1/chat/completions" and "output_expires_after" not in create


async def test_mistral_batch_jobs_for_chat_and_embeddings():
    out = jsonl({"id": "x", "custom_id": "r0", "response": {"status_code": 200, "body": {"object": "list", "model": "mistral-embed",
                 "usage": {"prompt_tokens": 28, "total_tokens": 28}, "data": [{"object": "embedding", "embedding": [0.1], "index": 0}]}}, "error": None})
    err = jsonl({"custom_id": "r1", "response": {"status_code": 422, "body": {"error": {"message": "bad input", "type": "invalid_request"}}}})
    fake = (Fake().on("POST", r"/v1/files$", httpx.Response(200, json={"id": "file-m", "purpose": "batch"}))
            .on("POST", r"/v1/batch/jobs$", httpx.Response(200, json={"id": "job-m", "status": "QUEUED"}))
            .on("GET", r"/v1/batch/jobs/job-m$", httpx.Response(200, json={
                "id": "job-m", "status": "SUCCESS", "total_requests": 2, "succeeded_requests": 1, "failed_requests": 1,
                "output_file": "file-o", "error_file": "file-e"}))
            .on("GET", r"/v1/files/file-o/content$", httpx.Response(200, content=out))
            .on("GET", r"/v1/files/file-e/content$", httpx.Response(200, content=err))
            .on("POST", r"/v1/batch/jobs/job-m/cancel$", httpx.Response(200, json={"status": "CANCELLATION_REQUESTED"}))
            .on("DELETE", r"/v1/files/", httpx.Response(200, json={"deleted": True}))
            .on("DELETE", r"/v1/batch/jobs/job-m$", httpx.Response(405, json={"message": "no"})))
    _, api = wire({"format": "openai", "base_url": "https://api.mistral.ai/v1"}, fake)
    assert isinstance(api, MistralBatches) and api.embeddings and api.max_requests == 1_000_000
    job = {**JOB, "model": "mistral-embed", "endpoint": "embeddings"}
    sub = await api.submit(job, [("r0", {"model": "mistral-embed", "input": ["a"]}), ("r1", {"model": "mistral-embed", "input": [""]})])
    form = multipart_fields(fake.requests[0])
    assert form["purpose"] == b"batch" and lines_of(form["file"])[0] == {"custom_id": "r0", "body": {"model": "mistral-embed", "input": ["a"]}}
    assert json.loads(fake.requests[1].content) == {"input_files": ["file-m"], "endpoint": "/v1/embeddings", "model": "mistral-embed",
                                                    "timeout_hours": 24, "metadata": {"meritorder_job": "bj_1", "meritorder_batch": "batch_1"}}
    job.update(provider_id=sub.provider_id, provider=sub.provider)
    polled = await api.poll(job)
    assert polled.ended and polled.counts == {"processing": 0, "succeeded": 1, "errored": 1, "canceled": 0, "expired": 0}
    job["provider"] = {**job["provider"], **polled.provider}
    got = await every(api, job)
    assert got["r0"].status == "succeeded" and got["r0"].body["usage"]["prompt_tokens"] == 28
    assert (got["r1"].code, got["r1"].message) == (422, "bad input")
    await api.cancel(job)
    await api.purge(job)  # files deleted; the job's own DELETE isn't offered (405): fine
    assert len(fake.sent("DELETE", r"/v1/files/")) == 3
    fake.on("GET", r"/v1/batch/jobs/job-t$", httpx.Response(200, json={"status": "TIMEOUT_EXCEEDED", "total_requests": 3, "succeeded_requests": 1}))
    assert (await api.poll({**job, "provider_id": "job-t"})).expired
    fake.on("GET", r"/v1/batch/jobs/job-f$", httpx.Response(200, json={"status": "FAILED", "errors": [{"message": "bad file", "count": 1}]}))
    assert (await api.poll({**job, "provider_id": "job-f"})).failed == "bad file"
    fake.on("GET", r"/v1/batch/jobs/job-r$", httpx.Response(200, json={"status": "RUNNING", "total_requests": 3, "succeeded_requests": 1}))
    assert (await api.poll({**job, "provider_id": "job-r"})).counts["processing"] == 2
    fake.on("DELETE", r"/v1/batch/jobs/job-m$", httpx.Response(500, json={"message": "down"}))
    with pytest.raises(BackendError):
        await api.purge(job)


async def test_openrouter_batches_inline_with_results_on_the_batch():
    done = {"id": "orb_1", "status": "completed", "request_counts": {"total": 3, "completed": 1, "failed": 2}, "results": [
        {"custom_id": "r0", "response": {"status_code": 200, "body": CHAT}, "error": None},
        {"custom_id": "r1", "response": None, "error": {"type": "invalid_request_error", "message": "too long"}},
        {"custom_id": "r2", "response": None, "error": {"type": "expired", "message": "not run"}}, "junk"]}
    fake = (Fake().on("POST", r"/api/v1/batches$", httpx.Response(202, json={"id": "orb_1", "status": "validating"}))
            .on("GET", r"/api/v1/batches/orb_1$", httpx.Response(200, json=done))
            .on("DELETE", r"/api/v1/batches/orb_1$", httpx.Response(200, json={"deletion": {"upstream": {"status": "deleted"}}})))
    _, api = wire({"format": "openai", "base_url": "https://openrouter.ai/api/v1"}, fake)
    assert isinstance(api, OpenRouterBatches) and api.one_model and api.embeddings
    sub = await api.submit({**JOB, "model": "qwen/qwen3"}, [("r0", {"model": "qwen/qwen3", "messages": [], "stream": True})])
    sent = json.loads(fake.requests[0].content)
    assert list(sent) == ["endpoint", "model", "completion_window", "requests"]  # requests last: OpenRouter reads the body as a stream
    assert sent["requests"] == [{"custom_id": "r0", "body": {"model": "qwen/qwen3", "messages": []}}] and sub.provider_id == "orb_1"
    job = {**JOB, "provider_id": "orb_1"}
    polled = await api.poll(job)
    assert polled.ended and polled.counts["succeeded"] == 1 and polled.counts["errored"] == 2
    got = await every(api, job)
    assert got["r0"].status == "succeeded" and (got["r1"].code, got["r1"].message) == (400, "too long") and got["r2"].status == "expired"
    assert await api.cancel(job) is None  # OpenRouter has no cancel
    await api.purge(job)
    fake.on("GET", r"/api/v1/batches/orb_2$", httpx.Response(200, json={"status": "failed", "error": {"message": "no batch variant"}}))
    assert (await api.poll({**job, "provider_id": "orb_2"})).failed == "no batch variant"
    assert await api.submit({**JOB, "endpoint": "embeddings"}, [("r0", {"model": "e", "input": "a"})])
    assert json.loads(fake.requests[-1].content)["endpoint"] == "/v1/embeddings"
    fake.on("DELETE", r"/api/v1/batches/orb_1$", httpx.Response(410, json={}))
    await api.purge(job)  # already gone: fine


async def test_xai_batches_add_requests_in_pieces_and_page_through_results():
    pages = {None: {"results": [{"batch_request_id": "r0", "batch_result": {"response": {"chat_get_completion": CHAT}}}], "pagination_token": "p2"},
             "p2": {"results": [{"batch_request_id": "r1", "batch_result": {"error": "rate limited upstream"}}], "pagination_token": None}}

    def results(request):
        return httpx.Response(200, json=pages[parse_qs(urlparse(str(request.url)).query).get("pagination_token", [None])[0]])

    fake = (Fake().on("POST", r"/v1/batches$", httpx.Response(200, json={"batch_id": "xb_1", "name": "bj_1"}))
            .on("POST", r"/v1/batches/xb_1/requests$", httpx.Response(200, json={}))
            .on("GET", r"/v1/batches/xb_1$", httpx.Response(200, json={"batch_id": "xb_1", "state": {
                "num_requests": 2, "num_pending": 0, "num_success": 1, "num_error": 1, "num_cancelled": 0}}))
            .on("GET", r"/v1/batches/xb_1/results", results)
            .on("POST", r"/v1/batches/xb_1:cancel$", httpx.Response(200, json={})))
    _, api = wire({"format": "openai", "base_url": "https://api.x.ai/v1"}, fake)
    assert isinstance(api, XaiBatches) and not api.one_model and not api.embeddings
    api.PIECE = 1
    sub = await api.submit(JOB, [("r0", {"model": "grok-4.7", "messages": [], "stream": False}), ("r1", {"model": "grok-4.7", "messages": []})])
    assert sub.provider_id == "xb_1" and len(fake.sent("POST", r"/requests$")) == 2
    piece = json.loads(fake.sent("POST", r"/requests$")[0].content)
    assert piece == {"batch_requests": [{"batch_request_id": "r0", "batch_request": {"chat_get_completion": {"model": "grok-4.7", "messages": []}}}]}
    job = {**JOB, "provider_id": "xb_1"}
    polled = await api.poll(job)
    assert polled.ended and polled.counts == {"processing": 0, "succeeded": 1, "errored": 1, "canceled": 0, "expired": 0}
    got = await every(api, job)
    assert got["r0"].status == "succeeded" and got["r1"].message == "rate limited upstream"
    await api.cancel(job)
    assert await api.purge(job) is None  # xAI has no delete
    fake.on("GET", r"/v1/batches/xb_2$", httpx.Response(200, json={"state": {"num_requests": 2, "num_pending": 2}, "cancel_time": "t"}))
    assert (await api.poll({**job, "provider_id": "xb_2"})).provider_status == "cancelling"
    fake.on("POST", r"/v1/batches/xb_1/requests$", httpx.Response(400, json={"error": "model not supported for batch"}))
    with pytest.raises(BackendError, match="not supported for batch"):  # the half-made batch is cancelled
        await api.submit(JOB, [("r0", {"model": "grok-2", "messages": []})])
    assert len(fake.sent("POST", r":cancel$")) == 2


# ---------------------------------------------------------------- Gemini API


async def test_gemini_embeddings_batch_one_request_per_input_joined_back():
    uploaded = {}

    def upload(request):
        uploaded["data"] = request.content
        return httpx.Response(200, json={"file": {"name": "files/in-e"}})

    out = jsonl({"key": "r0:1:2", "response": {"embedding": {"values": [0.3]}, "tokenCount": "4"}},
                {"key": "r0:0:2", "response": {"embedding": {"values": [0.1, 0.2]}, "usageMetadata": {"promptTokenCount": 3}}},
                {"key": "r1:0:1", "error": {"code": 400, "message": "empty", "status": "INVALID_ARGUMENT"}},
                {"key": "r2:0:2", "response": {"embedding": {"values": [1]}}})
    fake = (Fake().on("POST", r"^/upload/v1beta/files$", httpx.Response(200, headers={"x-goog-upload-url": "https://upload.example/u1"}, json={}))
            .on("POST", r"^/u1$", upload)
            .on("POST", r"/v1beta/models/gemini-embedding-001:asyncBatchEmbedContent$", httpx.Response(200, json={"name": "batches/e1"}))
            .on("GET", r"/v1beta/batches/e1$", httpx.Response(200, json={"name": "batches/e1", "done": True, "metadata": {
                "state": "BATCH_STATE_SUCCEEDED", "batchStats": {"requestCount": "4", "successfulRequestCount": "3"}},
                "response": {"output": {"responsesFile": "files/out-e"}}}))
            .on("GET", r"^/download/v1beta/files/out-e:download\?alt=media$", httpx.Response(200, content=out)))
    _, api = wire({"format": "openai", "api": "gemini", "base_url": "https://generativelanguage.googleapis.com/v1beta"}, fake)
    job = {**JOB, "model": "gemini-embedding-001", "endpoint": "embeddings"}
    sub = await api.submit(job, [("r0", {"model": "gemini-embedding-001", "input": ["a", "b"], "dimensions": 2}),
                                 ("r1", {"model": "gemini-embedding-001", "input": ""}), ("r2", {"model": "gemini-embedding-001", "input": [[1, 2], "x"]})])
    rows = lines_of(uploaded["data"])
    assert [r["key"] for r in rows] == ["r0:0:2", "r0:1:2", "r1:0:1", "r2:0:2", "r2:1:2"]
    assert rows[0]["request"] == {"content": {"parts": [{"text": "a"}]}, "model": "models/gemini-embedding-001", "outputDimensionality": 2}
    job.update(provider_id=sub.provider_id, provider=sub.provider)
    polled = await api.poll(job)  # the output on the operation's response
    job["provider"] = {**job["provider"], **polled.provider}
    got = await every(api, job)
    body = got["r0"].body
    assert [d["embedding"] for d in body["data"]] == [[0.1, 0.2], [0.3]] and body["usage"]["prompt_tokens"] == 7
    assert body["object"] == "list" and body["model"] == "gemini-embedding-001"
    assert (got["r1"].code, got["r1"].kind) == (400, "INVALID_ARGUMENT")
    assert got["r2"].status == "errored" and "1 of 2" in got["r2"].message  # an input with no result fails its request


async def test_gemini_backend_embeds_live_through_batch_embed_contents():
    from meritorder.formats.gemini_upstream import embed_inputs

    fake = Fake().on("POST", r"/models/gemini-embedding-001:batchEmbedContents$", lambda r: httpx.Response(200, json={
        "embeddings": [{"values": [0.5]} for _ in json.loads(r.content)["requests"]]}))
    b, _ = wire({"format": "openai", "api": "gemini", "base_url": "https://generativelanguage.googleapis.com/v1beta"}, fake)
    resp = await b.embed({"model": "gemini-embedding-001", "input": ["hello world", "again"]})
    assert len(resp.body["data"]) == 2 and resp.body["usage"]["prompt_tokens"] > 0  # no count in the reply: the estimate
    assert embed_inputs({"input": [1, 2, 3]}) == ["[1, 2, 3]"]  # one token array is one input
    v, _ = wire({"format": "openai", "api": "gemini", "flavor": "vertex", "project": "p"}, fake)
    with pytest.raises(BackendError, match="Vertex"):
        await v.embed({"model": "gemini-embedding-001", "input": "x"})


async def test_gemini_batch_mode_uploads_through_the_files_api():
    uploaded = {}

    def upload(request):
        uploaded["data"] = request.content
        return httpx.Response(200, json={"file": {"name": "files/in-1"}})

    out = jsonl({"key": "r0", "response": GEMINI_REPLY}, {"key": "r1", "error": {"code": 400, "message": "bad", "status": "INVALID_ARGUMENT"}},
                {"key": "r2", "response": {"candidates": [{"finishReason": "MALFORMED_FUNCTION_CALL"}]}})
    fake = (Fake().on("POST", r"^/upload/v1beta/files$", httpx.Response(200, headers={"x-goog-upload-url": "https://upload.example/u1"}, json={}))
            .on("POST", r"^/u1$", upload)
            .on("POST", r"/v1beta/models/gemini-4-argon:batchGenerateContent$", httpx.Response(200, json={"name": "batches/b1", "metadata": {}}))
            .on("GET", r"/v1beta/batches/b1$", httpx.Response(200, json={"name": "batches/b1", "done": True, "metadata": {
                "state": "BATCH_STATE_SUCCEEDED", "batchStats": {"requestCount": "3", "successfulRequestCount": "1", "failedRequestCount": "2"},
                "output": {"responsesFile": "files/out-1"}}}))
            .on("GET", r"^/download/v1beta/files/out-1:download\?alt=media$", httpx.Response(200, content=out))
            .on("POST", r"/batches/b1:cancel$", httpx.Response(200, json={}))
            .on("DELETE", r"/v1beta/(batches/b1|files/in-1)$", httpx.Response(200, json={}))
            .on("DELETE", r"/v1beta/files/out-1$", httpx.Response(404, json={})))
    _, api = wire({"format": "openai", "api": "gemini", "base_url": "https://generativelanguage.googleapis.com/v1beta"}, fake)
    assert isinstance(api, GeminiBatches)
    job = {**JOB, "model": "gemini-4-argon"}
    sub = await api.submit(job, [("r0", {"model": "gemini-4-argon", "messages": [{"role": "user", "content": "hi"}]})])
    start = fake.requests[0]
    assert start.headers["x-goog-upload-protocol"] == "resumable" and start.headers["x-goog-upload-command"] == "start"
    assert int(start.headers["x-goog-upload-header-content-length"]) == len(uploaded["data"])
    assert fake.requests[1].headers["x-goog-upload-command"] == "upload, finalize"
    line = lines_of(uploaded["data"])[0]
    assert line["key"] == "r0" and line["request"]["contents"] == [{"role": "user", "parts": [{"text": "hi"}]}]
    assert json.loads(fake.requests[2].content) == {"batch": {"display_name": "bj_1", "input_config": {"file_name": "files/in-1"}}}
    assert sub.provider_id == "batches/b1"
    job.update(provider_id="batches/b1", provider=sub.provider)
    polled = await api.poll(job)
    assert polled.ended and polled.provider_status == "succeeded" and polled.counts["errored"] == 2
    job["provider"] = {**job["provider"], **polled.provider}
    got = await every(api, job)
    assert got["r0"].status == "succeeded" and got["r0"].body["choices"][0]["message"]["content"] == "hi"
    assert (got["r1"].code, got["r1"].kind) == (400, "INVALID_ARGUMENT") and got["r2"].status == "errored"
    await api.cancel(job)
    await api.purge(job)
    fake.on("GET", r"/v1beta/batches/b2$", httpx.Response(200, json={"name": "batches/b2", "state": "JOB_STATE_FAILED",
                                                                      "error": {"message": "quota"}}))
    assert (await api.poll({**job, "provider_id": "batches/b2"})).failed == "quota"
    assert [o async for o in api.results({**job, "provider": {}})] == []
    fake.on("POST", r"^/upload/v1beta/files$", httpx.Response(200, json={}))
    with pytest.raises(BackendError, match="no upload URL"):
        await api.submit(job, [("r0", {"messages": []})])


# ---------------------------------------------------------------- Bedrock


def s3_list(keys: list[str], token: str | None = None) -> httpx.Response:
    items = "".join(f"<Contents><Key>{k}</Key></Contents>" for k in keys)
    more = f"<NextContinuationToken>{token}</NextContinuationToken>" if token else ""
    return httpx.Response(200, content=f'<?xml version="1.0"?><ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">{items}{more}'
                                       "</ListBucketResult>".encode())


BEDROCK = {"format": "anthropic", "base_url": "https://bedrock-mantle.us-west-2.api.aws/anthropic",
           "auth": {"type": "aws_sigv4", "region": "us-west-2"}, "batch": {"bucket": "s3://bkt/pre", "role_arn": "arn:aws:iam::1:role/batch"}}


async def test_bedrock_batch_inference_through_s3(monkeypatch, tmp_path):
    """Fake keys only: never the developer's own AWS configuration."""
    for var in ("AWS_PROFILE", "AWS_DEFAULT_PROFILE", "AWS_SESSION_TOKEN", "AWS_BEARER_TOKEN_BEDROCK", "AWS_ROLE_ARN",
                "AWS_WEB_IDENTITY_TOKEN_FILE", "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", "AWS_CONTAINER_CREDENTIALS_FULL_URI"):
        monkeypatch.delenv(var, raising=False)
    for k, v in {"AWS_ACCESS_KEY_ID": "AKIDEXAMPLE", "AWS_SECRET_ACCESS_KEY": "secret", "AWS_SHARED_CREDENTIALS_FILE": str(tmp_path / "none"),
                 "AWS_CONFIG_FILE": str(tmp_path / "none"), "AWS_EC2_METADATA_DISABLED": "true"}.items():
        monkeypatch.setenv(k, v)
    put = {}

    def store(request):
        put[request.url.path] = request.content
        return httpx.Response(200)

    out = jsonl({"recordId": "r0", "modelInput": {}, "modelOutput": ANTHROPIC_MSG},
                {"recordId": "r1", "modelInput": {}, "error": {"errorCode": 400, "errorMessage": "bad input"}},
                {"recordId": "r2", "modelInput": {}, "error": {"errorCode": "ThrottlingException", "errorMessage": "slow"}})
    folder = "pre/batch_1/bj_1"
    fake = (Fake().on("PUT", rf"^/{folder}/input.jsonl$", store)
            .on("POST", r"^/model-invocation-job$", httpx.Response(200, json={"jobArn": "arn:aws:bedrock:us-west-2:1:model-invocation-job/abc123"}))
            .on("GET", r"^/model-invocation-job/arn", httpx.Response(200, json={"status": "PartiallyCompleted", "totalRecordCount": 3,
                                                                              "successRecordCount": 1, "errorRecordCount": 2}))
            .on("GET", r"list-type=2.*continuation-token=t2", s3_list([f"{folder}/output/abc123/input.jsonl.out"]))
            .on("GET", r"list-type=2&prefix=pre%2Fbatch_1%2Fbj_1%2Foutput%2F$", s3_list([f"{folder}/output/abc123/manifest.json.out"], "t2"))
            .on("GET", r"list-type=2&prefix=pre%2Fbatch_1%2Fbj_1%2F$", s3_list([f"{folder}/input.jsonl", f"{folder}/output/abc123/input.jsonl.out"]))
            .on("GET", rf"^/{folder}/output/abc123/input.jsonl.out$", httpx.Response(200, content=out))
            .on("POST", r"/stop$", httpx.Response(200))
            .on("DELETE", rf"^/{folder}/input.jsonl$", httpx.Response(204))
            .on("DELETE", r"input.jsonl.out$", httpx.Response(404)))
    backend, api = wire(BEDROCK, fake, make_auth(BackendConfig.model_validate({"name": "b", **BEDROCK}).auth))
    assert isinstance(api, BedrockBatches) and api.min_requests == 100 and api.region() == "us-west-2"
    job = {**JOB, "model": "anthropic.claude-sonnet-5-5-v1:0"}
    sub = await api.submit(job, [("r0", {"model": "anthropic.claude-sonnet-5-5", "max_tokens": 5, "messages": [], "stream": False})])
    assert lines_of(put[f"/{folder}/input.jsonl"]) == [{"recordId": "r0", "modelInput": {"anthropic_version": "bedrock-2023-05-31",
                                                                                           "max_tokens": 5, "messages": []}}]
    s3_put = fake.requests[0]
    assert s3_put.url.host == "bkt.s3.us-west-2.amazonaws.com" and "x-amz-content-sha256" in s3_put.headers
    assert "/s3/aws4_request" in s3_put.headers["authorization"]
    create = fake.requests[1]
    assert create.url.host == "bedrock.us-west-2.amazonaws.com" and "/bedrock/aws4_request" in create.headers["authorization"]
    body = json.loads(create.content)
    assert body["modelId"] == "anthropic.claude-sonnet-5-5-v1:0" and body["roleArn"] == "arn:aws:iam::1:role/batch"
    assert body["inputDataConfig"]["s3InputDataConfig"]["s3Uri"] == f"s3://bkt/{folder}/input.jsonl" and body["jobName"] == "mo-bj-1"
    job.update(provider_id=sub.provider_id, provider=sub.provider)
    polled = await api.poll(job)
    assert polled.ended and polled.counts == {"processing": 0, "succeeded": 1, "errored": 2, "canceled": 0, "expired": 0}
    assert unquote(fake.requests[-1].url.path).endswith("model-invocation-job/arn:aws:bedrock:us-west-2:1:model-invocation-job/abc123")
    got = await every(api, job)
    assert got["r0"].body["id"] == "msg_1" and (got["r1"].code, got["r1"].message) == (400, "bad input") and got["r2"].code == 500
    await api.cancel(job)
    await api.purge(job)
    assert len(fake.sent("DELETE", r"")) == 2
    fake.on("GET", r"^/model-invocation-job/arn", httpx.Response(200, json={"status": "Failed", "message": "role can't read the bucket"}))
    assert (await api.poll(job)).failed == "role can't read the bucket"


# ---------------------------------------------------------------- Vertex AI


async def test_vertex_batch_prediction_for_claude_and_gemini():
    put = {}

    def store(request):
        put[parse_qs(urlparse(str(request.url)).query)["name"][0]] = request.content
        return httpx.Response(200, json={})

    claude_out = jsonl({"custom_id": "r0", "request": {}, "response": ANTHROPIC_MSG, "status": ""},
                       {"custom_id": "r1", "request": {}, "response": {}, "status": "Bad request: max_tokens"})
    gemini_out = jsonl({"status": "", "request": {"labels": {"meritorder_key": "r0"}}, "response": GEMINI_REPLY})
    out_dir = "gs://bkt/pre/batch_1/bj_1/output/prediction-model-1"
    fake = (Fake().on("POST", r"^/upload/storage/v1/b/bkt/o\?uploadType=media", store)
            .on("POST", r"/v1/projects/p/locations/us-east5/batchPredictionJobs$", httpx.Response(200, json={
                "name": "projects/p/locations/us-east5/batchPredictionJobs/77", "state": "JOB_STATE_PENDING"}))
            .on("GET", r"/v1/projects/p/locations/us-east5/batchPredictionJobs/77$", httpx.Response(200, json={
                "state": "JOB_STATE_SUCCEEDED", "completionStats": {"successfulCount": "1", "failedCount": "1"},
                "outputInfo": {"gcsOutputDirectory": out_dir}}))
            .on("GET", r"/storage/v1/b/bkt/o\?prefix=pre%2Fbatch_1%2Fbj_1%2Foutput%2Fprediction-model-1%2F$", httpx.Response(200, json={
                "items": [{"name": "pre/batch_1/bj_1/output/prediction-model-1/predictions.jsonl"}], "nextPageToken": "n2"}))
            .on("GET", r"pageToken=n2", httpx.Response(200, json={"items": [{"name": "pre/batch_1/bj_1/output/prediction-model-1/x.txt"}]}))
            .on("GET", r"/o/pre%2Fbatch_1%2Fbj_1%2Foutput%2Fprediction-model-1%2Fpredictions.jsonl\?alt=media$", httpx.Response(200, content=claude_out))
            .on("POST", r"/batchPredictionJobs/77:cancel$", httpx.Response(200, json={}))
            .on("GET", r"/storage/v1/b/bkt/o\?prefix=pre%2Fbatch_1%2Fbj_1%2F$",
                httpx.Response(200, json={"items": [{"name": "pre/batch_1/bj_1/input.jsonl"}]}))
            .on("DELETE", r"/storage/v1/b/bkt/o/", httpx.Response(204))
            .on("DELETE", r"/batchPredictionJobs/77$", httpx.Response(404, json={})))
    _, api = wire({"format": "anthropic", "flavor": "vertex", "project": "p", "region": "global",
                   "batch": {"bucket": "gs://bkt/pre", "region": "us-east5"}}, fake)
    assert isinstance(api, VertexBatches) and api.claude and api.region() == "us-east5"
    job = {**JOB, "model": "claude-sonnet-5-5"}
    sub = await api.submit(job, [("r0", {"model": "claude-sonnet-5-5", "max_tokens": 5, "messages": []})])
    assert lines_of(put["pre/batch_1/bj_1/input.jsonl"]) == [{"custom_id": "r0", "request": {"anthropic_version": "vertex-2023-10-16",
                                                                                             "max_tokens": 5, "messages": []}}]
    create = json.loads(fake.requests[1].content)
    assert create["model"] == "publishers/anthropic/models/claude-sonnet-5-5"
    assert create["outputConfig"]["gcsDestination"]["outputUriPrefix"] == "gs://bkt/pre/batch_1/bj_1/output"
    job.update(provider_id=sub.provider_id, provider=sub.provider)
    polled = await api.poll(job)
    assert polled.ended and polled.provider == {"output": out_dir}
    job["provider"] = {**job["provider"], **polled.provider}
    got = await every(api, job)
    assert got["r0"].body["id"] == "msg_1" and (got["r1"].status, got["r1"].message) == ("errored", "Bad request: max_tokens")
    await api.cancel(job)
    await api.purge(job)

    # Gemini on Vertex: the key rides along as a request label, which the output echoes.
    _, gapi = wire({"format": "openai", "api": "gemini", "flavor": "vertex", "project": "p", "region": "global", "batch": {"bucket": "gs://bkt/pre"}},
                   fake)
    assert not gapi.claude and gapi.region() == "global" and gapi._host() == "https://aiplatform.googleapis.com"
    assert gapi._line("r5", {"messages": [{"role": "user", "content": "hi"}]})["request"]["labels"] == {"meritorder_key": "r5"}
    fake.on("GET", r"/o/pre%2Fbatch_1%2Fbj_1%2Foutput%2Fprediction-model-1%2Fpredictions.jsonl\?alt=media$", httpx.Response(200, content=gemini_out))
    got = await every(gapi, {**job, "model": "gemini-4-argon"})
    assert got["r0"].body["choices"][0]["message"]["content"] == "hi"
    _, claude_global = wire({"format": "anthropic", "flavor": "vertex", "project": "p", "batch": {"bucket": "gs://bkt"}}, fake)
    assert claude_global.region() == "us-central1"  # Claude has no global batch prediction


async def test_mock_batch_api_answers_at_once_and_forgets_when_purged():
    from meritorder.backends.mock import MockBackend
    from meritorder.store import MemoryStore

    store = MemoryStore()
    b = MockBackend(BackendConfig.model_validate({"name": "m", "kind": "mock", "format": "openai", "mock_latency_ms": 0}))
    api = batch_api_for(b, SETTINGS, store)
    sub = await api.submit(JOB, [("r0", {"model": "x", "messages": [{"role": "user", "content": "hi"}]})])
    job = {**JOB, "provider_id": sub.provider_id}
    assert (await api.poll(job)).counts["succeeded"] == 1
    assert (await every(api, job))["r0"].body["object"] == "chat.completion"
    await api.cancel(job)
    await api.purge(job)
    assert (await api.poll(job)).failed == "the mock batch is gone"
