"""Fetching providers' cost data: each API's request format and its reply turned into bill lines. Every provider is mocked with
httpx.MockTransport and payloads shaped like the providers' documented ones; the real auth providers sign or authorize each
request (botocore SigV4 with fake keys, google-auth and azure-identity faked like in test_backend_auth)."""

from __future__ import annotations

import json
import sys
import time
import types
from datetime import date
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from meritorder.backends.auth import make_auth
from meritorder.billing import sources
from meritorder.billing.sources import BillError, fetch, kind_of, label_value, source_auth
from meritorder.config import BackendAuth, BackendConfig, BillSource, Config, ModelConfig, Route, ServeConfig

START, END = date(2026, 10, 19), date(2026, 10, 22)


def client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def query(request: httpx.Request) -> dict[str, list[str]]:
    return parse_qs(urlparse(str(request.url)).query)


class Bearer:
    """Stands in for an auth provider in tests that aren't about auth."""

    async def apply(self, method, url, headers, body):
        return {**headers, "authorization": "Bearer test-token"}


def gpu_cfg() -> Config:
    """A config with one SkyPilot deployment (so GPU queries run) and one static server (no cloud bill)."""
    return Config(backends=[
        BackendConfig(name="gpu-qwen", kind="self_hosted", format="openai", serve=ServeConfig(model="qwen")),
        BackendConfig(name="rack", kind="self_hosted", format="openai", serve=ServeConfig(model="qwen", provisioner="static")),
        BackendConfig(name="vertex", format="anthropic", flavor="vertex", project="llm-prod"),
        BackendConfig(name="vertex-gemini", format="openai",
                      base_url="https://aiplatform.googleapis.com/v1/projects/llm-gemini/locations/global/endpoints/openapi"),
    ], models=[ModelConfig(name="qwen", hf_id="Qwen/Qwen3", routes=[Route(backend="gpu-qwen", model="qwen3")])])


# ------------------------------------------------------------------ helpers


@pytest.mark.parametrize("text, kind", [
    ("Gemini 4 Carbon Input Text Tokens", "input"),
    ("gpt-6.1-terra-Inp-glbl", "input"),
    ("gpt-6.1-terra-Outp-glbl", "output"),
    ("USE1-MP:USE1_CacheReadInputTokenCount-Units", "cache_read"),
    ("USE1-MP:USE1_CacheWriteInputTokenCount-Units", "cache_write"),
    ("Gemini 3.8 Flash Cached Input Tokens", "cache_read"),
    ("Gemini 3.8 Flash Context Cache Storage per hour", "other"),
    ("Grounding with Google Search", "web_search"),
    ("Online Prediction Node Hours", "other"),
])
def test_kind_of_free_text(text, kind):
    assert kind_of(text) == kind


def test_label_value_follows_gcp_label_rules():
    assert label_value("GPU.Qwen Coder") == "gpu-qwen-coder"
    assert len(label_value("x" * 80)) == 63


def test_default_credentials_per_type(monkeypatch):
    assert source_auth(BillSource(type="anthropic")) == BackendAuth(type="api_key", header="x-api-key", value_env="ANTHROPIC_ADMIN_KEY")
    assert source_auth(BillSource(type="openai")).value_env == "OPENAI_ADMIN_KEY"
    aws = source_auth(BillSource(type="aws", auth=BackendAuth(type="aws_sigv4", profile="billing")))
    assert (aws.service, aws.region, aws.profile) == ("ce", "us-east-1", "billing")  # Cost Explorer, whatever the backend signs for
    assert source_auth(BillSource(type="azure")).scope == "https://management.azure.com/.default"
    assert source_auth(BillSource(type="azure", auth=BackendAuth(type="azure_ad"))).scope == "https://management.azure.com/.default"
    assert source_auth(BillSource(type="gcp")).type == "google"
    # An admin key from a mounted secret file still goes in x-api-key; an OAuth token can choose authorization.
    own = source_auth(BillSource(type="anthropic", auth=BackendAuth(type="api_key", value_file="/run/secrets/admin")))
    assert own.header == "x-api-key" and own.value_file == "/run/secrets/admin"
    oauth = source_auth(BillSource(type="anthropic", auth=BackendAuth(type="api_key", header="authorization", value_env="T")))
    assert oauth.header == "authorization"


# ------------------------------------------------------------------ Anthropic


def anthropic_cost_row(day: str, model: str | None, cost_type: str, token_type: str | None, cents: str, **group) -> dict:
    return {"starting_at": f"{day}T00:00:00Z", "ending_at": f"{day}T00:00:00Z",  # ending_at isn't used
            "group": {"model": model, "cost_type": cost_type, "token_type": token_type, "service_tier": "standard",
                      "context_window": "0-200k" if model else None, "product": "messages_api", "speed": "standard", **group},
            "cost": {"amount": cents, "currency": "USD", "list_amount": cents, "requests": None}}


async def test_anthropic_cost_and_usage(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_ADMIN_KEY", "sk-ant-admin01-test")
    seen = []
    cost_page1 = {"data": [
        anthropic_cost_row("2026-10-20", "claude-sonnet-5-5", "tokens", "uncached_input_tokens", "1500.000000"),
        anthropic_cost_row("2026-10-20", "claude-sonnet-5-5", "tokens", "output_tokens", "4500.000000"),
    ], "data_refreshed_at": "2026-10-22T04:02:11Z", "has_more": True, "next_page": "page_Mg=="}
    cost_page2 = {"data": [
        anthropic_cost_row("2026-10-20", "claude-sonnet-5-5", "tokens", "cache_read_input_tokens", "30.000000"),
        anthropic_cost_row("2026-10-20", "claude-sonnet-5-5", "tokens", "cache_creation.ephemeral_1h_input_tokens", "600.000000"),
        anthropic_cost_row("2026-10-20", "claude-sonnet-5-5", "tokens", "uncached_input_tokens", "250.000000",
                           service_tier="batch", product="message_batches_api"),
        anthropic_cost_row("2026-10-21", None, "web_search", None, "1000.000000"),
    ], "data_refreshed_at": "2026-10-22T04:02:11Z", "has_more": False, "next_page": None}
    usage = {"data": [{
        "starting_at": "2026-10-20T00:00:00Z", "ending_at": "2026-10-21T00:00:00Z",
        "group": {"model": "claude-sonnet-5-5", "service_tier": "standard", "context_window": "0-200k", "product": "messages_api",
                  "speed": "standard"},
        "usage": {"uncached_input_tokens": 5_000_000, "output_tokens": 3_000_000, "cache_read_input_tokens": 1_000_000,
                  "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 1_000_000},
                  "server_tool_use": {"web_search_requests": 0}, "requests": 1200, "total_tokens": 10_000_000},
    }], "data_refreshed_at": "2026-10-22T04:02:11Z", "has_more": False, "next_page": None}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        assert request.headers["x-api-key"] == "sk-ant-admin01-test"
        assert request.headers["anthropic-beta"] == "analytics-usage-cost-2026-08-24"
        assert request.headers["anthropic-version"] == "2026-10-21"
        q = query(request)
        assert q["time_bucket_width"] == ["1d"] and q["limit"] == ["1000"] and q["starting_at"] == ["2026-10-19T00:00:00Z"]
        assert q["workspace_ids[]"] == ["wrkspc_01merit"]
        if request.url.path == "/v1/organizations/analytics/cost":
            assert q["group_by[]"] == ["model", "service_tier", "context_window", "product", "speed", "cost_type", "token_type"]
            return httpx.Response(200, json=cost_page2 if q.get("page") == ["page_Mg=="] else cost_page1)
        assert request.url.path == "/v1/organizations/analytics/usage"
        assert q["group_by[]"] == ["model", "service_tier", "context_window", "product", "speed"]
        return httpx.Response(200, json=usage)

    src = BillSource(name="anthropic", type="anthropic", workspace_ids=["wrkspc_01merit"])
    async with client(handler) as c:
        got = await fetch(src, Config(backends=[], models=[]), START, END, c, make_auth(source_auth(src)))
    assert len(seen) == 3
    lines = {ln["line"]: ln for ln in got.lines}
    first = lines["claude-sonnet-5-5 · uncached_input_tokens · standard · 0-200k · messages_api"]
    assert first["cost"] == 15.0 and first["kind"] == "input" and first["tokens"] == 5_000_000 and first["model"] == "claude-sonnet-5-5"
    assert first["day"] == date(2026, 10, 20) and first["detail"]["list_cost"] == 15.0
    one_hour = lines["claude-sonnet-5-5 · cache_creation.ephemeral_1h_input_tokens · standard · 0-200k · messages_api"]
    assert one_hour["kind"] == "cache_write" and one_hour["tokens"] == 1_000_000 and one_hour["cost"] == 6.0
    assert lines["claude-sonnet-5-5 · output_tokens · standard · 0-200k · messages_api"]["tokens"] == 3_000_000
    batch = lines["claude-sonnet-5-5 · uncached_input_tokens · batch · 0-200k · message_batches_api"]
    assert batch["detail"]["tier"] == "batch" and batch["tokens"] is None
    search = lines["(no model) · web_search · standard · messages_api"]
    assert search["kind"] == "web_search" and search["model"] is None and search["cost"] == 10.0
    assert got.complete_through == date(2026, 10, 21)  # data refreshed at 04:02 on the 22nd: the 21st is complete


async def test_anthropic_long_ranges_are_split_and_an_expired_cursor_starts_over(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_ADMIN_KEY", "k")
    calls = []

    def handler(request):
        q = query(request)
        calls.append((request.url.path, q["starting_at"][0], q.get("page", [None])[0]))
        if q.get("page") == ["stale"]:
            return httpx.Response(410, json={"type": "error", "error": {"type": "gone", "message": "cursor expired"}})
        first = not any(c[2] == "stale" for c in calls)
        page = {"data": [], "data_refreshed_at": None, "has_more": first, "next_page": "stale" if first else None}
        return httpx.Response(200, json=page)

    src = BillSource(name="a", type="anthropic")
    async with client(handler) as c:
        got = await fetch(src, Config(backends=[], models=[]), date(2026, 8, 1), date(2026, 10, 15), c, make_auth(source_auth(src)))
    starts = sorted({c[1] for c in calls})
    assert starts == ["2026-08-01T00:00:00Z", "2026-09-01T00:00:00Z", "2026-10-02T00:00:00Z"]  # 31 days at most per request
    assert got.lines == [] and got.complete_through == date(2026, 7, 31)  # nothing exported yet: nothing complete


async def test_anthropic_errors_say_why_without_the_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_ADMIN_KEY", "sk-ant-admin01-secret")
    reply = {"type": "error", "error": {"type": "permission_error", "message": "This endpoint requires an Admin API key"}}
    src = BillSource(name="a", type="anthropic")
    async with client(lambda r: httpx.Response(403, json=reply)) as c:
        with pytest.raises(BillError, match="HTTP 403: This endpoint requires an Admin API key") as e:
            await fetch(src, Config(backends=[], models=[]), START, END, c, make_auth(source_auth(src)))
    assert "secret" not in str(e.value)


# ------------------------------------------------------------------ OpenAI


async def test_openai_costs_by_line_item(monkeypatch):
    monkeypatch.setenv("OPENAI_ADMIN_KEY", "sk-admin-test")
    page1 = {"object": "page", "has_more": True, "next_page": "page_AAAA", "data": [
        {"object": "bucket", "start_time": 1792454400, "end_time": 1792540800, "results": [  # 2026-10-20
            {"object": "organization.costs.result", "amount": {"value": 12.5, "currency": "usd"}, "line_item": "gpt-6.1-terra, input_tokens",
             "project_id": None, "quantity": 5_000_000, "quantity_unit": "tokens"},
            {"object": "organization.costs.result", "amount": {"value": 1.25, "currency": "usd"},
             "line_item": "gpt-6.1-terra, cached_input_tokens", "project_id": None, "quantity": 5_000, "quantity_unit": "1000_tokens"},
        ]}]}
    page2 = {"object": "page", "has_more": False, "next_page": None, "data": [
        {"object": "bucket", "start_time": 1792454400, "end_time": 1792540800, "results": [
            {"object": "organization.costs.result", "amount": {"value": 40.0, "currency": "usd"}, "line_item": "gpt-6.1-terra, output_tokens",
             "quantity": 2_000_000, "quantity_unit": "tokens"},
            {"object": "organization.costs.result", "amount": {"value": 3.0, "currency": "usd"},
             "line_item": "gpt-6.1-terra, input_tokens, batch", "quantity": 2_400_000, "quantity_unit": "tokens"},
            {"object": "organization.costs.result", "amount": {"value": 0.5, "currency": "usd"}, "line_item": "web search tool calls",
             "quantity": None, "quantity_unit": None},
        ]}]}

    def handler(request):
        assert request.headers["authorization"] == "Bearer sk-admin-test"
        assert request.url.path == "/v1/organization/costs"
        q = query(request)
        assert q["group_by"] == ["line_item"] and q["bucket_width"] == ["1d"] and q["project_ids"] == ["proj_merit"]
        assert q["start_time"] == ["1792368000"] and q["end_time"] == ["1792627200"]
        return httpx.Response(200, json=page2 if q.get("page") == ["page_AAAA"] else page1)

    src = BillSource(name="openai", type="openai", project_ids=["proj_merit"])
    async with client(handler) as c:
        got = await fetch(src, Config(backends=[], models=[]), START, END, c, make_auth(source_auth(src)))
    lines = {ln["line"]: ln for ln in got.lines}
    assert lines["gpt-6.1-terra, input_tokens"] | {"detail": None} == {
        "day": date(2026, 10, 20), "line": "gpt-6.1-terra, input_tokens", "model": "gpt-6.1-terra", "kind": "input", "deployment": None,
        "cost": 12.5, "tokens": 5_000_000, "detail": None}
    assert lines["gpt-6.1-terra, cached_input_tokens"]["kind"] == "cache_read"
    assert lines["gpt-6.1-terra, cached_input_tokens"]["tokens"] == 5_000_000  # 1000_tokens units
    assert lines["gpt-6.1-terra, output_tokens"]["kind"] == "output"
    assert lines["gpt-6.1-terra, input_tokens, batch"]["detail"]["tier"] == "batch"
    assert lines["web search tool calls"]["model"] is None and lines["web search tool calls"]["kind"] == "other"
    assert got.complete_through is None  # settle_days decides


# ------------------------------------------------------------------ AWS Cost Explorer


@pytest.fixture
def fake_aws(monkeypatch, tmp_path):
    """Fake keys only: never the developer's own AWS configuration."""
    for var in ("AWS_PROFILE", "AWS_DEFAULT_PROFILE", "AWS_SESSION_TOKEN", "AWS_BEARER_TOKEN_BEDROCK", "AWS_ROLE_ARN",
                "AWS_WEB_IDENTITY_TOKEN_FILE", "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", "AWS_CONTAINER_CREDENTIALS_FULL_URI"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "none"))
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "none"))
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIDEXAMPLE")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")


def ce_group(keys: list[str], cost: str, qty: str | None = None, unit: str = "Count") -> dict:
    metrics = {"UnblendedCost": {"Amount": cost, "Unit": "USD"}}
    if qty is not None:
        metrics["UsageQuantity"] = {"Amount": qty, "Unit": unit}
    return {"Keys": keys, "Metrics": metrics}


def ce_day(day: str, end: str, groups: list[dict], estimated: bool = True) -> dict:
    return {"TimePeriod": {"Start": day, "End": end}, "Total": {}, "Groups": groups, "Estimated": estimated}


async def test_aws_bedrock_by_model_credits_and_gpu_tags(fake_aws):
    bodies = []

    def handler(request):
        assert request.url.host == "ce.us-east-1.amazonaws.com" and request.method == "POST"
        assert request.headers["content-type"] == "application/x-amz-json-1.1"
        auth = request.headers["authorization"]
        assert auth.startswith("AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/") and "/us-east-1/ce/aws4_request" in auth
        target = request.headers["x-amz-target"]
        body = json.loads(request.content)
        bodies.append((target, body))
        assert target == "AWSInsightsIndexService.GetCostAndUsage"
        assert body["Granularity"] == "DAILY" and body["TimePeriod"] == {"Start": "2026-10-19", "End": "2026-10-22"}
        kinds = [(g["Type"], g["Key"]) for g in body["GroupBy"]]
        if kinds == [("PRODUCT_ATTRIBUTE", "model"), ("PRODUCT_ATTRIBUTE", "inferenceType")]:
            assert body["Filter"] == {"And": [{"Dimensions": {"Key": "RECORD_TYPE", "Values": ["Usage"]}},
                                              {"Dimensions": {"Key": "LINKED_ACCOUNT", "Values": ["123456789012"]}}]}
            if "NextPageToken" not in body:
                return httpx.Response(200, json={"GroupDefinitions": body["GroupBy"], "NextPageToken": "next", "ResultsByTime": [
                    ce_day("2026-10-20", "2026-10-21", [ce_group(["model$Claude Sonnet 5.5", "inferenceType$Input tokens"], "15.02", "5006000"),
                                                        ce_group(["model$Claude Sonnet 5.5", "inferenceType$Output tokens"], "45.00", "3000000",
                                                                 unit="Tokens")])]})
            return httpx.Response(200, json={"ResultsByTime": [
                ce_day("2026-10-20", "2026-10-21", [ce_group(["model$Claude Sonnet 5.5", "inferenceType$Cache read input tokens"], "0.30", "1")]),
                ce_day("2026-10-21", "2026-10-22", [ce_group(["model$Claude Sonnet 5.5", "inferenceType$Batch input tokens"], "2.00", "1")])]})
        if kinds == [("DIMENSION", "RECORD_TYPE"), ("PRODUCT_ATTRIBUTE", "model")]:
            assert body["Filter"]["And"][0] == {"Not": {"Dimensions": {"Key": "RECORD_TYPE", "Values": ["Usage"]}}}
            return httpx.Response(200, json={"ResultsByTime": [ce_day("2026-10-20", "2026-10-21", [
                ce_group(["Credit", "model$Claude Sonnet 5.5"], "-10.00"), ce_group(["Tax", "model$"], "0")])]})
        assert kinds == [("TAG", "meritorder-deployment"), ("DIMENSION", "SERVICE")]
        return httpx.Response(200, json={"ResultsByTime": [ce_day("2026-10-20", "2026-10-21", [
            ce_group(["meritorder-deployment$gpu-qwen", "Amazon Elastic Compute Cloud - Compute"], "96.00"),
            ce_group(["meritorder-deployment$", "Amazon Elastic Compute Cloud - Compute"], "500.00")])]})  # untagged: not ours

    src = BillSource(name="aws", type="aws", accounts=["123456789012"])
    async with client(handler) as c:
        got = await fetch(src, gpu_cfg(), START, END, c, make_auth(source_auth(src)))
    lines = {ln["line"]: ln for ln in got.lines}
    assert lines["Claude Sonnet 5.5 · Input tokens"]["cost"] == 15.02 and lines["Claude Sonnet 5.5 · Input tokens"]["model"] == "Claude Sonnet 5.5"
    assert lines["Claude Sonnet 5.5 · Input tokens"]["tokens"] is None  # "Count" isn't a token unit we can trust
    assert lines["Claude Sonnet 5.5 · Output tokens"]["tokens"] == 3_000_000 and lines["Claude Sonnet 5.5 · Output tokens"]["kind"] == "output"
    assert lines["Claude Sonnet 5.5 · Cache read input tokens"]["kind"] == "cache_read"
    assert lines["Claude Sonnet 5.5 · Batch input tokens"]["detail"]["tier"] == "batch"
    assert lines["Credit · Claude Sonnet 5.5"] | {"detail": None} == {
        "day": date(2026, 10, 20), "line": "Credit · Claude Sonnet 5.5", "model": "Claude Sonnet 5.5", "kind": "adjustment",
        "deployment": None, "cost": -10.0, "tokens": None, "detail": None}
    assert "Tax · (no model)" not in lines  # zero
    gpu = lines["gpu-qwen · Amazon Elastic Compute Cloud - Compute"]
    assert gpu["kind"] == "gpu" and gpu["deployment"] == "gpu-qwen" and gpu["cost"] == 96.0
    assert len(got.lines) == 6 and not got.notes
    assert len(bodies) == 4  # two pages, adjustments, GPU tags


async def test_aws_falls_back_to_service_and_usage_type(fake_aws):
    """Before September 2026 Cost Explorer has no model breakdown: match on service and usage type names instead."""

    def handler(request):
        target = request.headers["x-amz-target"].rsplit(".", 1)[1]
        body = json.loads(request.content)
        if target == "GetDimensionValues":
            assert body["Dimension"] == "SERVICE" and body["SearchString"] == "Bedrock" and body["Context"] == "COST_AND_USAGE"
            return httpx.Response(200, json={"DimensionValues": [{"Value": "Amazon Bedrock", "Attributes": {}},
                                                                 {"Value": "Claude Sonnet 5.5 (Amazon Bedrock Edition)", "Attributes": {}}],
                                             "ReturnSize": 2, "TotalSize": 2})
        if body["GroupBy"][0]["Type"] == "PRODUCT_ATTRIBUTE":
            why = "Product attribute data is available for time periods that start on or after September 1, 2026."
            return httpx.Response(400, json={"__type": "com.amazon.awsinsights#DataUnavailableException", "Message": why})
        assert [g["Key"] for g in body["GroupBy"]] == ["SERVICE", "USAGE_TYPE"]
        assert body["Filter"]["And"][0] == {"Dimensions": {"Key": "SERVICE", "Values": ["Amazon Bedrock",
                                                                                         "Claude Sonnet 5.5 (Amazon Bedrock Edition)"]}}
        return httpx.Response(200, json={"ResultsByTime": [ce_day("2026-08-20", "2026-08-21", [
            ce_group(["Claude Sonnet 5.5 (Amazon Bedrock Edition)", "USE1-MP:USE1_InputTokenCount-Units"], "3.00", "1000"),
            ce_group(["Amazon Bedrock", "USE1-NovaPro-input-tokens"], "0.40", "500")])]})

    src = BillSource(name="aws", type="aws", gpu_tag=None)
    async with client(handler) as c:
        got = await fetch(src, Config(backends=[], models=[]), date(2026, 8, 19), date(2026, 8, 22), c, make_auth(source_auth(src)))
    lines = {ln["line"]: ln for ln in got.lines}
    assert lines["Claude Sonnet 5.5 (Amazon Bedrock Edition) · USE1-MP:USE1_InputTokenCount-Units"]["kind"] == "input"
    assert lines["Amazon Bedrock · USE1-NovaPro-input-tokens"]["model"] is None
    assert "service and usage type" in got.notes[0]


async def test_aws_other_errors_are_raised(fake_aws):
    reply = {"__type": "AccessDeniedException", "Message": "User is not authorized to perform: ce:GetCostAndUsage"}
    src = BillSource(name="aws", type="aws", auth=BackendAuth(type="aws_sigv4", region="cn-northwest-1"))

    def handler(request):
        assert request.url.host == "ce.cn-northwest-1.amazonaws.com.cn"
        return httpx.Response(400, json=reply)

    async with client(handler) as c:
        with pytest.raises(BillError, match="AccessDeniedException: User is not authorized"):
            await fetch(src, Config(backends=[], models=[]), START, END, c, make_auth(source_auth(src)))


# ------------------------------------------------------------------ Google Cloud: BigQuery billing export


def fake_module(monkeypatch, name: str, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


@pytest.fixture
def fake_google(monkeypatch):
    class Creds:
        valid, token = False, None

        def refresh(self, request):
            self.valid, self.token = True, "ya29.billing"

    google = fake_module(monkeypatch, "google")
    google.auth = fake_module(monkeypatch, "google.auth", default=lambda scopes: (Creds(), "proj"))
    fake_module(monkeypatch, "google.auth.transport")
    fake_module(monkeypatch, "google.auth.transport.requests", Request=lambda: object())


def bq_rows(*rows) -> list[dict]:
    return [{"f": [{"v": v} for v in r]} for r in rows]


BQ_SCHEMA = {"fields": [{"name": n, "type": t} for n, t in [("day", "DATE"), ("service", "STRING"), ("sku", "STRING"), ("cost_type", "STRING"),
                                                         ("deployment", "STRING"), ("cost", "FLOAT"), ("credits", "FLOAT"),
                                                         ("quantity", "FLOAT"), ("unit", "STRING")]]}


async def test_gcp_billing_export_query_waits_and_pages(fake_google):
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path, query(request)))
        assert request.headers["authorization"] == "Bearer ya29.billing"
        job = {"projectId": "billing-proj", "jobId": "job_1", "location": "US"}
        if request.method == "POST":
            body = json.loads(request.content)
            assert request.url.path == "/bigquery/v2/projects/billing-proj/queries"
            assert body["useLegacySql"] is False and body["parameterMode"] == "NAMED" and body["location"] == "US"
            assert "FROM `billing-proj.billing.gcp_billing_export_v1_0123AB_4567CD_89EF01`" in body["query"]
            assert "_PARTITIONTIME >= @partition_start" in body["query"]
            params = {p["name"]: p for p in body["queryParameters"]}
            assert params["start"]["parameterValue"]["value"] == "2026-10-19 00:00:00+00"
            assert params["partition_start"]["parameterValue"]["value"] == "2026-10-18 00:00:00+00"  # late rows land in later partitions
            assert [v["value"] for v in params["projects"]["parameterValue"]["arrayValues"]] == ["llm-gemini", "llm-prod"]
            assert params["gpu_label"]["parameterValue"]["value"] == "meritorder-deployment"
            return httpx.Response(200, json={"kind": "bigquery#queryResponse", "jobReference": job, "jobComplete": False})
        q = query(request)
        assert request.url.path == "/bigquery/v2/projects/billing-proj/queries/job_1" and q["location"] == ["US"]
        if "pageToken" not in q:
            return httpx.Response(200, json={"kind": "bigquery#getQueryResultsResponse", "jobReference": job, "jobComplete": True,
                                             "schema": BQ_SCHEMA, "totalRows": "5", "pageToken": "p2", "rows": bq_rows(
                ["2026-10-20", "Vertex AI", "Gemini 3.8 Flash Input Text Tokens", "regular", None, "6.0", "-1.5", "20000000", "count"],
                ["2026-10-20", "Vertex AI", "Gemini 3.8 Flash-Lite Output Text Tokens", "regular", None, "0.4", "0", "100", "count"])})
        return httpx.Response(200, json={"jobReference": job, "jobComplete": True, "totalRows": "5", "rows": bq_rows(
            ["2026-10-20", "Claude Sonnet 5.5", "Claude Sonnet 5.5 Output Tokens", "regular", None, "45.0", "0", "3000000", "count"],
            ["2026-10-20", "Compute Engine", "Nvidia H100 80GB GPU running in Americas", "regular", "gpu-qwen", "80.0", "-8.0", "24", "hour"],
            ["2026-10-21", "Vertex AI", "Tax", "tax", None, "1.2", "0", None, None])})

    src = BillSource(name="gcp", type="gcp", table="billing-proj.billing.gcp_billing_export_v1_0123AB_4567CD_89EF01", location="US")
    cfg = gpu_cfg()
    async with client(handler) as c:
        got = await fetch(src, cfg, START, END, c, make_auth(source_auth(src)))
    assert [s[0] for s in seen] == ["POST", "GET", "GET"]
    lines = {ln["line"]: ln for ln in got.lines}
    flash = lines["Vertex AI · Gemini 3.8 Flash Input Text Tokens"]
    assert flash["cost"] == 6.0 and flash["kind"] == "input" and flash["detail"]["unit"] == "count"
    assert lines["Vertex AI · Gemini 3.8 Flash Input Text Tokens · credits"]["cost"] == -1.5
    assert lines["Vertex AI · Gemini 3.8 Flash Input Text Tokens · credits"]["kind"] == "adjustment"
    gpu = lines["gpu-qwen · Compute Engine · Nvidia H100 80GB GPU running in Americas"]
    assert gpu["kind"] == "gpu" and gpu["cost"] == 80.0 and gpu["detail"]["credits"] == -8.0
    assert lines["Vertex AI · Tax · tax"]["kind"] == "adjustment"


async def test_gcp_needs_a_table_name_it_can_quote():
    for table in (None, "dataset_only", "p.d.t`; DROP TABLE x; --"):
        with pytest.raises(BillError, match="project.dataset.table"):
            await fetch(BillSource(name="gcp", type="gcp", table=table), Config(backends=[], models=[]), START, END, None, Bearer())


async def test_gcp_view_without_partitions(fake_google):
    def handler(request):
        body = json.loads(request.content)
        assert "_PARTITIONTIME" not in body["query"] and "partition_start" not in {p["name"] for p in body["queryParameters"]}
        assert request.url.path == "/bigquery/v2/projects/other-proj/queries"
        return httpx.Response(200, json={"jobComplete": True, "schema": BQ_SCHEMA, "rows": [], "jobReference": {"jobId": "j"}})

    src = BillSource(name="gcp", type="gcp", table="p.d.v", partitioned=False, query_project="other-proj", project_ids=["x"])
    async with client(handler) as c:
        assert (await fetch(src, Config(backends=[], models=[]), START, END, c, make_auth(source_auth(src)))).lines == []


# ------------------------------------------------------------------ Azure Cost Management


@pytest.fixture
def fake_azure(monkeypatch):
    scopes = []

    class Credential:
        async def get_token(self, scope):
            scopes.append(scope)
            return types.SimpleNamespace(token="eyJ0eXAi.azure", expires_on=time.time() + 3600)

    fake_module(monkeypatch, "azure")
    fake_module(monkeypatch, "azure.identity")
    fake_module(monkeypatch, "azure.identity.aio", DefaultAzureCredential=Credential)
    return scopes


async def test_azure_query_pages_and_gpu_tags(fake_azure):
    scope = "/subscriptions/1b2c3d4e-0000-0000-0000-000000000000/resourceGroups/llm"
    url = f"https://management.azure.com{scope}/providers/Microsoft.CostManagement/query?api-version=2026-06-01"
    next_link = f"{url}&$skiptoken=AQAAAA%3D%3D"
    columns = [{"name": "Cost", "type": "Number"}, {"name": "UsageDate", "type": "Number"}, {"name": "ServiceName", "type": "String"},
               {"name": "Meter", "type": "String"}, {"name": "Currency", "type": "String"}]
    seen = []

    def handler(request):
        assert request.headers["authorization"] == "Bearer eyJ0eXAi.azure" and request.method == "POST"
        body = json.loads(request.content)
        seen.append(str(request.url))
        assert body["type"] == "ActualCost" and body["timeframe"] == "Custom"
        assert body["timePeriod"] == {"from": "2026-10-19T00:00:00Z", "to": "2026-10-21T23:59:59Z"}
        ds = body["dataset"]
        assert ds["granularity"] == "Daily" and ds["aggregation"] == {"totalCost": {"name": "Cost", "function": "Sum"}}
        if ds["grouping"][0]["type"] == "TagKey":
            assert ds["filter"] == {"tags": {"name": "meritorder-deployment", "operator": "In", "values": ["gpu-qwen"]}}
            return httpx.Response(200, json={"properties": {"nextLink": None, "columns": [
                {"name": "Cost", "type": "Number"}, {"name": "UsageDate", "type": "Number"}, {"name": "TagKey", "type": "String"},
                {"name": "TagValue", "type": "String"}, {"name": "ServiceName", "type": "String"}, {"name": "Currency", "type": "String"}],
                "rows": [[55.5, 20261020, "meritorder-deployment", "gpu-qwen", "Virtual Machines", "USD"],
                         [3.0, 20261020, "meritorder-deployment", None, "Storage", "USD"]]}})
        assert ds["grouping"] == [{"type": "Dimension", "name": "ServiceName"}, {"type": "Dimension", "name": "Meter"}]
        assert ds["filter"] == {"dimensions": {"name": "ResourceType", "operator": "In", "values": ["microsoft.cognitiveservices/accounts"]}}
        if str(request.url) == url:
            return httpx.Response(200, json={"id": "q", "name": "q", "type": "Microsoft.CostManagement/query", "properties": {
                "nextLink": next_link, "columns": columns,
                "rows": [[12.4, 20261020, "Foundry Models", "gpt-6.1-terra-Inp-glbl Tokens", "USD"],
                         [40.1, 20261020, "Foundry Models", "gpt-6.1-terra-Outp-glbl Tokens", "USD"]]}})
        return httpx.Response(200, json={"properties": {"nextLink": None, "columns": columns, "rows": [
            [8.0, 20261021, "Claude Sonnet 5.5 (Anthropic)", "Claude Sonnet 5.5 Input Tokens", "EUR"]]}})

    src = BillSource(name="azure", type="azure", scope=scope)
    async with client(handler) as c:
        got = await fetch(src, gpu_cfg(), START, END, c, make_auth(source_auth(src)))
    assert fake_azure == ["https://management.azure.com/.default"]
    assert seen[:2] == [url, next_link]
    lines = {ln["line"]: ln for ln in got.lines}
    assert lines["Foundry Models · gpt-6.1-terra-Inp-glbl Tokens"]["kind"] == "input"
    assert lines["Foundry Models · gpt-6.1-terra-Outp-glbl Tokens"]["cost"] == 40.1
    assert lines["Claude Sonnet 5.5 (Anthropic) · Claude Sonnet 5.5 Input Tokens"]["detail"]["currency"] == "EUR"
    assert lines["Claude Sonnet 5.5 (Anthropic) · Claude Sonnet 5.5 Input Tokens"]["day"] == date(2026, 10, 21)
    assert lines["gpu-qwen · Virtual Machines"]["kind"] == "gpu" and lines["gpu-qwen · Virtual Machines"]["cost"] == 55.5
    assert len(got.lines) == 4  # the untagged storage row isn't a GPU line


async def test_azure_needs_a_scope_and_a_cost_column():
    with pytest.raises(BillError, match="scope must be set"):
        await fetch(BillSource(name="az", type="azure"), Config(backends=[], models=[]), START, END, None, Bearer())

    def handler(request):
        return httpx.Response(200, json={"properties": {"columns": [{"name": "UsageDate", "type": "Number"}], "rows": []}})

    async with client(handler) as c:
        with pytest.raises(BillError, match="no cost or date column"):
            await fetch(BillSource(name="az", type="azure", scope="/subscriptions/x", gpu_tag=None, resource_types=[]),
                        Config(backends=[], models=[]), START, END, c, Bearer())


async def test_azure_usd_column_and_throttling():
    def handler(request):
        body = json.loads(request.content)
        assert "filter" not in body["dataset"]  # resource_types: [] reads the whole scope
        assert body["dataset"]["aggregation"]["totalCost"]["name"] == "CostUSD"
        return httpx.Response(200, json={"properties": {"columns": [{"name": "CostUSD", "type": "Number"}, {"name": "UsageDate", "type": "Number"},
                                                                    {"name": "ServiceName", "type": "String"}, {"name": "Meter", "type": "String"},
                                                                    {"name": "Currency", "type": "String"}],
                                                        "rows": [[9.0, "2026-10-20T00:00:00", "Foundry Models", "x", "EUR"]]}})

    src = BillSource(name="az", type="azure", scope="/subscriptions/x", cost_column="CostUSD", resource_types=[], gpu_tag=None)
    async with client(handler) as c:
        got = await fetch(src, Config(backends=[], models=[]), START, END, c, Bearer())
    assert got.lines[0]["detail"]["currency"] == "USD" and got.lines[0]["day"] == date(2026, 10, 20)
    throttled = {"error": {"code": "429", "message": "Too many requests. Please retry."}}
    async with client(lambda r: httpx.Response(429, json=throttled)) as c:
        with pytest.raises(BillError, match="HTTP 429: Too many requests"):
            await fetch(src, Config(backends=[], models=[]), START, END, c, Bearer())


async def test_a_dropped_connection_is_a_bill_error():
    def handler(request):
        raise httpx.ConnectError("connection refused")

    async with client(handler) as c:
        with pytest.raises(BillError, match="OpenAI costs: ConnectError"):
            await fetch(BillSource(name="o", type="openai"), Config(backends=[], models=[]), START, END, c, Bearer())


def test_task_yaml_labels_gpu_instances():
    from meritorder.skypilot import INSTALL_LABEL, Fleet, Identity, SkyPilotProvisioner

    cfg = gpu_cfg()
    prov = SkyPilotProvisioner(cfg.backends[0], cfg, Fleet(Identity("ab12", "s")))
    task = prov.task_config(prov.cluster(1))
    assert task["resources"]["labels"] == {sources.GPU_LABEL: "gpu-qwen", INSTALL_LABEL: "ab12"}


# ------------------------------------------------------------------ paging limits and odd replies


@pytest.mark.parametrize("src, reply", [
    (BillSource(name="a", type="anthropic"), {"data": [], "has_more": True, "next_page": "again"}),
    (BillSource(name="o", type="openai"), {"data": [], "has_more": True, "next_page": "again"}),
    (BillSource(name="aws", type="aws", gpu_tag=None), {"ResultsByTime": [], "NextPageToken": "again"}),
    (BillSource(name="az", type="azure", scope="/subscriptions/x", gpu_tag=None),
     {"properties": {"columns": [], "rows": [], "nextLink": "https://management.azure.com/next"}}),
    (BillSource(name="gcp", type="gcp", table="p.d.t"), {"jobComplete": True, "pageToken": "again", "jobReference": {"jobId": "j"}}),
    (BillSource(name="gcp", type="gcp", table="p.d.t"), {"jobComplete": False, "jobReference": {"jobId": "j"}}),
])
async def test_a_reply_that_never_ends_is_an_error(monkeypatch, src, reply):
    monkeypatch.setattr(sources, "MAX_PAGES", 3)
    async with client(lambda r: httpx.Response(200, json=reply)) as c:
        with pytest.raises(BillError, match="too many pages|didn't finish"):
            await fetch(src, Config(backends=[], models=[]), START, END, c, Bearer())


async def test_errors_that_arent_json():
    async with client(lambda r: httpx.Response(502, text="<html>Bad gateway</html>")) as c:
        with pytest.raises(BillError, match="HTTP 502: <html>Bad gateway</html>"):
            await fetch(BillSource(name="o", type="openai"), Config(backends=[], models=[]), START, END, c, Bearer())
        with pytest.raises(BillError, match="Cost Explorer GetCostAndUsage: 502: <html>"):
            await fetch(BillSource(name="aws", type="aws"), Config(backends=[], models=[]), START, END, c, Bearer())
    async with client(lambda r: httpx.Response(400, json=["bad", "request"])) as c:
        with pytest.raises(BillError, match=r"HTTP 400: \['bad', 'request'\]"):
            await fetch(BillSource(name="o", type="openai"), Config(backends=[], models=[]), START, END, c, Bearer())
    async with client(lambda r: httpx.Response(400, json={"message": "start_time is required"})) as c:
        with pytest.raises(BillError, match="HTTP 400: start_time is required"):
            await fetch(BillSource(name="o", type="openai"), Config(backends=[], models=[]), START, END, c, Bearer())


async def test_azure_no_content_and_anthropic_repeated_rows():
    async with client(lambda r: httpx.Response(204)) as c:  # Cost Management: nothing for this period
        with pytest.raises(BillError, match="no cost or date column"):
            await fetch(BillSource(name="az", type="azure", scope="/subscriptions/x", gpu_tag=None), Config(backends=[], models=[]),
                        START, END, c, Bearer())
    row = anthropic_cost_row("2026-10-20", "claude-sonnet-5-5", "tokens", "output_tokens", "100.000000")

    def handler(request):
        rows = [row, row] if request.url.path.endswith("/cost") else []
        return httpx.Response(200, json={"data": rows, "data_refreshed_at": None, "has_more": False, "next_page": None})

    async with client(handler) as c:
        got = await fetch(BillSource(name="a", type="anthropic"), Config(backends=[], models=[]), START, END, c, Bearer())
    assert [ln["cost"] for ln in got.lines] == [2.0]
