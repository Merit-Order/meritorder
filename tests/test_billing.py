"""Checking the ledger against bills: matching bill lines to models, the report (recorded vs billed by provider, model and
day), partial days, alerts, the worker job, the reports API, the CLI, and storage on both backings. Providers are mocked
(test_billing_sources covers each API's request and reply in detail)."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
import yaml
from conftest import ADMIN_KEY, LEAD_KEY, auth, ledger_row

from meritorder import cli
from meritorder.billing import BillRefresher, check_gaps, refresh_once, report
from meritorder.billing.reconcile import Matcher, complete_through, contains, id_words, words
from meritorder.billing.sources import covered_backends
from meritorder.config import BackendConfig, BillingConfig, BillSource, ServeConfig, build_config

TODAY = date(2026, 10, 25)
DAY = date(2026, 10, 20)  # settled for every source
LATE = date(2026, 10, 24)  # too young for AWS (2 days) and past Anthropic's data_refreshed_at

PRICE = {"input": 3, "output": 15, "cache_read": 0.3, "cache_write": 3.75}
RAW = {
    "backends": {
        "anthropic": {"format": "anthropic", "base_url": "https://api.anthropic.com", "auth": {"type": "api_key", "header": "x-api-key", "value": "k"}},
        "bedrock": {"format": "anthropic", "base_url": "https://bedrock-mantle.us-east-1.api.aws/anthropic", "auth": {"type": "aws_sigv4"}},
        "vertex": {"format": "anthropic", "flavor": "vertex", "project": "llm-prod", "region": "global", "auth": {"type": "google"}},
        "vertex-gemini": {"format": "openai", "auth": {"type": "google"},
                          "base_url": "https://aiplatform.googleapis.com/v1/projects/llm-prod/locations/global/endpoints/openapi"},
        "gemini": {"format": "openai", "api": "gemini", "base_url": "https://generativelanguage.googleapis.com/v1beta",
                   "auth": {"type": "api_key", "header": "x-goog-api-key", "value": "k"}},  # the Gemini API bills a Cloud project
        "azure-openai": {"format": "openai", "base_url": "https://example.openai.azure.com/openai/v1",
                         "auth": {"type": "api_key", "header": "api-key", "value": "k"}},
        "openai": {"format": "openai", "base_url": "https://api.openai.com/v1", "auth": {"type": "api_key", "value": "k"}},
        "openrouter": {"format": "openai", "base_url": "https://openrouter.ai/api/v1", "auth": {"type": "api_key", "value": "k"}},
        "xai": {"format": "openai", "base_url": "https://api.x.ai/v1", "auth": {"type": "api_key", "value": "k"}},
        "mock": {"kind": "mock", "format": "openai"},
        "gpu-qwen": {"kind": "self_hosted", "format": "openai", "serve": {"model": "qwen"}},
    },
    "models": {
        "claude-sonnet": {"routes": [{"backend": "anthropic", "model": "claude-sonnet-5-5", "price": PRICE},
                                     {"backend": "bedrock", "model": "anthropic.claude-sonnet-5-5", "price": PRICE},
                                     {"backend": "vertex", "model": "claude-sonnet-5-5@20260901", "price": PRICE}]},
        "claude-sonnet-old": {"routes": [{"backend": "anthropic", "model": "claude-sonnet-5", "price": PRICE}]},
        "gemini-flash": {"routes": [{"backend": "vertex-gemini", "model": "google/gemini-3.8-flash", "price_key": "gemini-3.8-flash",
                                     "price": {"input": 0.3, "output": 2.5}},
                                    {"backend": "gemini", "model": "gemini-3.8-flash", "price": {"input": 0.3, "output": 2.5}}]},
        "gpt-terra": {"routes": [{"backend": "openai", "model": "gpt-6.1-terra", "price": {"input": 2.5, "output": 20}},
                                 {"backend": "azure-openai", "model": "terra-prod", "price_key": "gpt-6.1-terra", "price": {"input": 2.5, "output": 20}},
                                 {"backend": "openrouter", "model": "openai/gpt-6.1-terra", "price": {"input": 2.5, "output": 20}}]},
        "qwen": {"hf_id": "Qwen/Qwen3", "routes": [{"backend": "gpu-qwen", "model": "qwen3"}]},
    },
    "billing": {
        "alert_pct": 5, "alert_min_usd": 1,
        "sources": {
            "anthropic": {"type": "anthropic"},
            "aws": {"type": "aws"},
            "gcp": {"type": "gcp", "table": "billing-proj.billing.gcp_billing_export_v1_0123AB", "gpu_tag": None},
            "azure": {"type": "azure", "scope": "/subscriptions/s/resourceGroups/llm", "gpu_tag": None},
            "openai": {"type": "openai", "models": {"terra legacy*": "gpt-terra"}, "ignore": ["*fine-tuning*"]},
        },
    },
}


@pytest.fixture
def bcfg(tmp_path):
    return build_config(RAW, tmp_path, quiet=True)


@pytest.fixture(autouse=True)
def no_webhook(monkeypatch):
    monkeypatch.delenv("MERITORDER_ALERT_WEBHOOK_URL", raising=False)


def at(day: date, hour: int = 12) -> datetime:
    return datetime(day.year, day.month, day.day, hour, tzinfo=UTC)


# ------------------------------------------------------------------ configuration


def test_billing_config_from_a_mapping_and_its_checks(bcfg, tmp_path):
    assert [s.name for s in bcfg.billing.sources] == ["anthropic", "aws", "gcp", "azure", "openai"]
    assert bcfg.billing.source("gcp").table.endswith("0123AB") and bcfg.billing.source("nope") is None
    assert bcfg.billing.refresh_hours == 6 and bcfg.billing.lookback_days == 40
    bad = {**RAW, "billing": {"sources": {"x": {"type": "anthropic", "backends": ["nowhere"]}}}}
    with pytest.raises(ValueError, match="billing.sources.x.backends names backends that aren't defined: nowhere"):
        build_config(bad, tmp_path, quiet=True)
    bad = {**RAW, "billing": {"sources": {"x": {"type": "anthropic", "models": {"*": "gpt-nine"}}}}}
    with pytest.raises(ValueError, match="maps bill lines to models that aren't configured: gpt-nine"):
        build_config(bad, tmp_path, quiet=True)


def test_each_source_covers_its_own_backends(bcfg):
    covered = {s.name: covered_backends(s, bcfg) for s in bcfg.billing.sources}
    assert covered == {"anthropic": ["anthropic"], "aws": ["bedrock"], "gcp": ["vertex", "vertex-gemini", "gemini"], "azure": ["azure-openai"],
                       "openai": ["openai"]}  # OpenRouter's cost is what it billed; GPUs are compared on their own
    assert covered_backends(BillSource(type="openai", backends=["openrouter"]), bcfg) == ["openrouter"]


# ------------------------------------------------------------------ matching


def test_words_and_ids():
    assert id_words("anthropic.claude-sonnet-5-5") == ["claude", "sonnet", "5", "5"]
    assert id_words("us.anthropic.claude-haiku-5-5-20260301-v1:0") == ["claude", "haiku", "5", "5"]
    assert id_words("google/gemini-3.8-flash") == ["gemini", "3", "8", "flash"]
    assert id_words("claude-sonnet-5-5@20260901") == ["claude", "sonnet", "5", "5"]
    assert id_words("deepseek-v3") == ["deepseek", "v3"]  # only Bedrock's ":0" versions are dropped
    assert words("gpt-6.1-terra-2026-05-01, input_tokens") == ["gpt", "6", "1", "terra", "input", "tokens"]


@pytest.mark.parametrize("line, key, hit", [
    ("claude sonnet 5 5 input tokens", "claude sonnet 5 5", True),
    ("claude sonnet 5 5 input tokens", "claude sonnet 5", False),  # a version continues: 5.5 isn't 5
    ("gemini 3 8 flash lite output", "gemini 3 8 flash", False),  # a variant isn't the model
    ("gemini 3 8 flash input", "gemini 3 8 flash", True),
    ("gpt 6 1 terra 1105 inp glbl", "gpt 6 1 terra", True),  # a 4-digit code isn't a version
    ("gpt 6 luna mini input", "gpt 6 luna", False),
    ("foundry models terra prod", "terra prod", True),
])
def test_contains(line, key, hit):
    assert contains(line.split(), tuple(key.split())) is hit


def bill(text: str, kind: str = "input", model: str | None = None, cost: float = 1.0, **detail) -> dict:
    return {"day": DAY, "line": text, "model": model, "kind": kind, "deployment": None, "cost": cost, "tokens": None, "detail": detail}


def test_classify_every_kind_of_line(bcfg):
    src = {s.name: s for s in bcfg.billing.sources}

    def classify(source: str, ln: dict, upstream=None):
        return Matcher(src[source], bcfg, covered_backends(src[source], bcfg), upstream).classify(ln)

    assert classify("anthropic", bill("x", model="claude-sonnet-5-5")) == ("model", "claude-sonnet")
    assert classify("anthropic", bill("x", model="claude-sonnet-5")) == ("model", "claude-sonnet-old")
    assert classify("anthropic", bill("x", model="claude-sonnet-5-5-20260929")) == ("model", "claude-sonnet")
    assert classify("anthropic", bill("x", model="claude-haiku-5-5")) == ("unmatched", "claude-haiku-5-5 isn't routed through anthropic")
    assert classify("anthropic", bill("x", model="claude-haiku-5-5"), {"claude-haiku-5-5": "claude-haiku"}) == ("model", "claude-haiku")
    assert classify("anthropic", bill("x", kind="web_search")) == ("unmatched", "web search not tied to a model")
    assert classify("anthropic", bill("x", model="claude-sonnet-5-5", tier="batch")) == ("model", "claude-sonnet")  # batches match too
    assert classify("aws", bill("Claude Sonnet 5.5 · Input tokens", model="Claude Sonnet 5.5")) == ("model", "claude-sonnet")
    assert classify("aws", bill("x", model="Claude Sonnet 5.5 (Amazon Bedrock Edition)")) == ("model", "claude-sonnet")
    assert classify("aws", bill("Claude Sonnet 5 · Input tokens", model="Claude Sonnet 5"))[0] == "unmatched"  # not on Bedrock here
    assert classify("aws", bill("Credit · Claude Sonnet 5.5", kind="adjustment", cost=-5)) == (
        "excluded", "credit, refund, tax or other adjustment: not compared")
    assert classify("gcp", bill("Vertex AI · Gemini 3.8 Flash Input Text Tokens")) == ("model", "gemini-flash")
    assert classify("gcp", bill("Vertex AI · Gemini 3.8 Flash-Lite Output Text Tokens")) == ("unmatched", "no routed model's name in this line")
    assert classify("gcp", bill("Claude Sonnet 5.5 · Claude Sonnet 5.5 Output Tokens")) == ("model", "claude-sonnet")
    assert classify("azure", bill("Foundry Models · gpt-6.1-terra-Inp-glbl Tokens")) == ("model", "gpt-terra")
    assert classify("azure", bill("Foundry Models · terra-prod Output", currency="EUR")) == (
        "excluded", "billed in EUR; costs are compared in US dollars (set cost_column: CostUSD)")
    assert classify("openai", bill("gpt-6.1-terra, input_tokens", model="gpt-6.1-terra")) == ("model", "gpt-terra")
    assert classify("openai", bill("Terra legacy plan, input")) == ("model", "gpt-terra")  # the source's own mapping
    assert classify("openai", bill("Fine-tuning training, gpt-6.1-terra")) == ("excluded", "ignored (billing.sources.openai.ignore)")


def test_complete_through_takes_the_earlier_of_settle_time_and_the_provider():
    src = BillSource(type="aws")
    assert complete_through(src, {}, TODAY) is None  # never fetched: nothing is settled
    fetched = {"fetched_at": "2026-10-25T06:00:00+00:00", "covered_from": "2026-09-15"}
    assert complete_through(src, fetched, TODAY) == date(2026, 10, 23)
    assert complete_through(src, {**fetched, "complete_through": "2026-10-21"}, TODAY) == date(2026, 10, 21)
    assert complete_through(BillSource(type="aws", settle_days=0), fetched, TODAY) == TODAY


# ------------------------------------------------------------------ the whole job, every provider mocked


def anthropic_reply(request: httpx.Request) -> httpx.Response:
    q = parse_qs(urlparse(str(request.url)).query)
    base = {"data_refreshed_at": "2026-10-24T04:00:00Z", "has_more": False, "next_page": None}
    g = {"model": "claude-sonnet-5-5", "service_tier": "standard", "context_window": "0-200k", "product": "messages_api", "speed": "standard"}
    if q["starting_at"] != ["2026-10-16T00:00:00Z"]:  # 2026-09-15 + 31 days
        return httpx.Response(200, json={**base, "data": []})
    if request.url.path.endswith("/usage"):
        return httpx.Response(200, json={**base, "data": [{"starting_at": "2026-10-20T00:00:00Z", "ending_at": "2026-10-21T00:00:00Z", "group": g,
                                                            "usage": {"uncached_input_tokens": 5_000_000, "output_tokens": 3_000_000,
                                                                      "cache_read_input_tokens": 1_000_000, "cache_creation": {}}}]})

    def row(cost_type, token_type, cents, **over):
        return {"starting_at": "2026-10-20T00:00:00Z", "ending_at": "2026-10-21T00:00:00Z",
                "group": {**g, "cost_type": cost_type, "token_type": token_type, **over}, "cost": {"amount": cents, "currency": "USD"}}

    return httpx.Response(200, json={**base, "data": [
        row("tokens", "uncached_input_tokens", "1500.000000"), row("tokens", "output_tokens", "4500.000000"),
        row("tokens", "cache_read_input_tokens", "30.000000"),
        row("tokens", "uncached_input_tokens", "250.000000", service_tier="batch", product="message_batches_api"),
        row("web_search", None, "50.000000", model=None, context_window=None)]})


def ce_reply(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    keys = [g["Key"] for g in body["GroupBy"]]

    def day(d, groups):
        return {"TimePeriod": {"Start": d, "End": d}, "Groups": groups, "Estimated": True, "Total": {}}

    def g(k, cost):
        return {"Keys": k, "Metrics": {"UnblendedCost": {"Amount": cost, "Unit": "USD"}}}

    if keys == ["model", "inferenceType"]:
        return httpx.Response(200, json={"ResultsByTime": [day("2026-10-20", [g(["model$Claude Sonnet 5.5", "inferenceType$Input tokens"], "15.02"),
                                                                              g(["model$Claude Sonnet 5.5", "inferenceType$Output tokens"], "45.00")])]})
    if keys == ["RECORD_TYPE", "model"]:
        return httpx.Response(200, json={"ResultsByTime": [day("2026-10-20", [g(["Credit", "model$Claude Sonnet 5.5"], "-10.00")])]})
    return httpx.Response(200, json={"ResultsByTime": [day("2026-10-20", [g(["meritorder-deployment$gpu-qwen", "Amazon EC2"], "96.00"),
                                                                          g(["meritorder-deployment$gpu-old", "Amazon EC2"], "4.00")])]})


def bq_reply(request: httpx.Request) -> httpx.Response:
    fields = ["day", "service", "sku", "cost_type", "deployment", "cost", "credits", "quantity", "unit"]
    rows = [["2026-10-20", "Vertex AI", "Gemini 3.8 Flash Input Text Tokens", "regular", None, "6.0", "0", None, None],
            ["2026-10-20", "Vertex AI", "Gemini 3.8 Flash-Lite Output Text Tokens", "regular", None, "0.4", "0", None, None],
            ["2026-10-20", "Gemini API", "Gemini 3.8 Flash Output Text Tokens", "regular", None, "0.5", "0", None, None],
            ["2026-10-20", "Claude Sonnet 5.5", "Claude Sonnet 5.5 Output Tokens", "regular", None, "45.0", "0", None, None]]
    return httpx.Response(200, json={"jobComplete": True, "jobReference": {"jobId": "j"}, "schema": {"fields": [{"name": f} for f in fields]},
                                     "rows": [{"f": [{"v": v} for v in r]} for r in rows]})


def azure_reply(request: httpx.Request) -> httpx.Response:
    cols = [{"name": n} for n in ("Cost", "UsageDate", "ServiceName", "Meter", "Currency")]
    return httpx.Response(200, json={"properties": {"nextLink": None, "columns": cols, "rows": [
        [12.4, 20261020, "Foundry Models", "gpt-6.1-terra-Inp-glbl Tokens", "USD"],
        [40.1, 20261020, "Foundry Models", "gpt-6.1-terra-Outp-glbl Tokens", "USD"],
        [8.0, 20261020, "Foundry Models", "terra-prod cached input", "EUR"]]}})


def openai_reply(request: httpx.Request) -> httpx.Response:
    if request.headers.get("authorization") == "Bearer broken":
        return httpx.Response(401, json={"error": {"message": "Incorrect API key provided", "type": "invalid_request_error"}})
    return httpx.Response(200, json={"object": "page", "has_more": False, "next_page": None, "data": [
        {"object": "bucket", "start_time": int(at(DAY, 0).timestamp()), "end_time": int(at(DAY, 0).timestamp()) + 86400, "results": [
            {"object": "organization.costs.result", "amount": {"value": 2.0, "currency": "usd"}, "line_item": "gpt-6.1-terra, input_tokens",
             "quantity": 800_000, "quantity_unit": "tokens"}]}]})


HOSTS = {"api.anthropic.com": anthropic_reply, "ce.us-east-1.amazonaws.com": ce_reply, "bigquery.googleapis.com": bq_reply,
         "management.azure.com": azure_reply, "api.openai.com": openai_reply}


def providers() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(lambda r: HOSTS[r.url.host](r)))


class FakeAuth:
    def __init__(self, token: str = "ok"):
        self.token = token

    async def apply(self, method, url, headers, body):
        return {**headers, "authorization": f"Bearer {self.token}"}


def fake_auths(broken: set[str] = frozenset()):
    """Credentials for every source without touching any real credential chain; the ``broken`` key types get a bad key."""
    return lambda spec: FakeAuth("broken" if spec.value_env in broken else "ok")


async def seed_ledger(store):
    rows = [
        ledger_row(ts=at(DAY), backend="anthropic", served_model="claude-sonnet", upstream_model="claude-sonnet-5-5", actual_cost=60.0,
                   input_tokens=5_000_000, output_tokens=3_000_000, cache_read_tokens=1_000_000),
        # A batch's request on the same day: the bill's batch-tier line is matched against it.
        ledger_row(ts=at(DAY, 2), backend="anthropic", served_model="claude-sonnet", upstream_model="claude-sonnet-5-5", actual_cost=2.5,
                   service_tier="batch", batch_id="msgbatch_1"),
        ledger_row(ts=at(DAY, 13), backend="bedrock", served_model="claude-sonnet", upstream_model="anthropic.claude-sonnet-5-5", actual_cost=50.0),
        ledger_row(ts=at(LATE), backend="bedrock", served_model="claude-sonnet", upstream_model="anthropic.claude-sonnet-5-5", actual_cost=5.0),
        ledger_row(ts=at(DAY), backend="vertex-gemini", served_model="gemini-flash", upstream_model="google/gemini-3.8-flash", actual_cost=6.1),
        ledger_row(ts=at(DAY), backend="gemini", served_model="gemini-flash", upstream_model="gemini-3.8-flash", actual_cost=0.5),
        ledger_row(ts=at(DAY), backend="azure-openai", served_model="gpt-terra", upstream_model="terra-prod", actual_cost=52.5),
        ledger_row(ts=at(DAY), backend="openai", served_model="gpt-terra", upstream_model="gpt-6.1-terra", actual_cost=2.0),
        ledger_row(ts=at(DAY), backend="openrouter", served_model="gpt-terra", upstream_model="openai/gpt-6.1-terra", actual_cost=7.0),
        ledger_row(ts=at(DAY), backend="xai", served_model="grok", upstream_model="grok-4.7", actual_cost=3.0),
        ledger_row(ts=at(DAY), backend="gpu-qwen", served_model="qwen", upstream_model="qwen3", actual_cost=2.5),
        ledger_row(ts=at(DAY), backend="mock", served_model="qwen", upstream_model="m", actual_cost=0.5),
        ledger_row(ts=at(DAY), backend="retired", served_model="qwen", upstream_model="m", actual_cost=0.25),
    ]
    for r in rows:
        await store.record(r)
    await store.add_gpu_usage("gpu-qwen", at(DAY), 3600 * 6, 90.0)
    await store.rollup()


async def test_refresh_stores_every_bill_reports_and_alerts_once(bcfg, store, state, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_ADMIN_KEY", "sk-ant-admin01-x")
    monkeypatch.setenv("OPENAI_ADMIN_KEY", "sk-admin-x")
    await seed_ledger(store)
    async with providers() as c:
        status = await refresh_once(bcfg, store, state, today=TODAY, client=c, auth_factory=fake_auths())
    assert {k: v["ok"] for k, v in status.items()} == dict.fromkeys(["anthropic", "aws", "gcp", "azure", "openai"], True)
    assert status["anthropic"]["complete_through"] == "2026-10-23" and status["anthropic"]["covered_from"] == "2026-09-15"
    assert status["aws"]["lines"] == 5  # two tokens, one credit, two GPU tags

    rep = await report(bcfg, store, date(2026, 10, 1), TODAY + timedelta(days=1), today=TODAY)
    rows = {(r["period"], r["source"], r["model"]): r for r in rep["rows"]}
    a = rows[("2026-10-20", "anthropic", "claude-sonnet")]
    # The batch-tier line (2.50) is matched to the model too, and shown beside the ledger's batch rows.
    assert (a["recorded"], a["billed"], a["difference"], a["pct"], a["status"], a["over"]) == (62.5, 62.8, 0.3, 0.48, "matched", False)
    assert (a["recorded_batch"], a["billed_batch"]) == (2.5, 2.5)
    assert a["billed_by_kind"] == {"input": 17.5, "output": 45.0, "cache_read": 0.3}
    assert a["billed_tokens"] == {"input": 5_000_000, "output": 3_000_000, "cache_read": 1_000_000}
    assert a["recorded_tokens"] == {"input": 5_000_000, "cache_read": 1_000_000, "cache_write": 0, "output": 3_000_000}
    b = rows[("2026-10-20", "aws", "claude-sonnet")]
    assert (b["billed"], b["difference"], b["pct"], b["over"]) == (60.02, 10.02, 20.04, True)
    assert rows[("2026-10-24", "aws", "claude-sonnet")]["status"] == "partial"  # not settled: shown, never alerted
    assert rows[("2026-10-20", "gcp", "claude-sonnet")]["status"] == "not_recorded"  # someone else's Claude on Vertex
    gemini = rows[("2026-10-20", "gcp", "gemini-flash")]  # Vertex AI and the Gemini API both bill the Cloud project
    assert (gemini["recorded"], gemini["billed"], gemini["difference"]) == (6.6, 6.5, pytest.approx(-0.1))
    assert rows[("2026-10-20", "azure", "gpt-terra")]["difference"] == 0 and rows[("2026-10-20", "openai", "gpt-terra")]["status"] == "matched"

    src = {s["name"]: s for s in rep["sources"]}
    assert src["anthropic"]["settled"] == {"recorded": 62.5, "billed": 63.3, "billed_matched": 62.8, "billed_unmatched": 0.5, "left_out": 0.0,
                                          "adjustments": 0.0, "difference": 0.8, "pct": 1.28}
    assert src["aws"]["settled"]["adjustments"] == -10.0 and src["aws"]["partial_days"] == 1 and src["aws"]["totals"]["recorded"] == 55.0
    assert src["azure"]["settled"]["left_out"] == 8.0 and src["azure"]["filtered"] is True and src["aws"]["filtered"] is False
    why = {(u["source"], u["line"]): (u["status"], u["why"]) for u in rep["unmatched"]}
    assert why[("anthropic", "(no model) · web_search · standard · messages_api")] == ("unmatched", "web search not tied to a model")
    assert ("anthropic", "claude-sonnet-5-5 · uncached_input_tokens · batch · 0-200k · message_batches_api") not in why
    assert why[("gcp", "Vertex AI · Gemini 3.8 Flash-Lite Output Text Tokens")][0] == "unmatched"
    assert why[("aws", "gpu-old · Amazon EC2")] == ("unmatched", "GPU tag 'gpu-old' isn't a SkyPilot deployment in the config")
    assert rep["gpu"] == [{"period": "2026-10-20", "deployment": "gpu-qwen", "recorded": 90.0, "billed": 96.0, "billed_by_source": {"aws": 96.0},
                           "partial": False, "difference": 6.0, "pct": 6.67, "over": True}]
    assert rep["uncovered"] == [
        {"backend": "openrouter", "recorded": 7.0, "why": "cost is what the provider billed, returned with each response"},
        {"backend": "xai", "recorded": 3.0, "why": "no bill source covers it (billing.sources)"},
        {"backend": "gpu-qwen", "recorded": 2.5, "why": "self-hosted: GPU time is compared with the cloud bill under GPUs"},
        {"backend": "mock", "recorded": 0.5, "why": "mock backend"},
        {"backend": "retired", "recorded": 0.25, "why": "not in the config any more"}]
    assert rep["settled"]["recorded"] == pytest.approx(173.6)

    alerts = {a["title"]: a["text"] for a in await state.recent_alerts()}
    assert set(alerts) == {"Bill differs from the ledger: aws", "Bill differs from the ledger: gcp"}
    assert alerts["Bill differs from the ledger: aws"].startswith(
        "The aws bill for 2026-10-20 is $60.02; the ledger recorded $50.00 (+$10.02, +20.0%). Biggest difference: claude-sonnet +$10.02.")
    assert "$0.40 of the bill matched no model" in alerts["Bill differs from the ledger: gcp"]
    # The next refresh finds the same gaps: no new alerts.
    async with providers() as c:
        await refresh_once(bcfg, store, state, today=TODAY, client=c, auth_factory=fake_auths())
    assert len(await state.recent_alerts()) == 2

    month = await report(bcfg, store, date(2026, 10, 1), TODAY + timedelta(days=1), by="month", today=TODAY)
    aws_month = next(r for r in month["rows"] if r["source"] == "aws")
    assert aws_month["period"] == "2026-10" and aws_month["recorded"] == 55.0 and aws_month["status"] == "partial"


async def test_a_failed_source_keeps_its_last_bill_and_alerts(bcfg, store, state, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_ADMIN_KEY", "k")
    monkeypatch.setenv("OPENAI_ADMIN_KEY", "k")
    off = BillSource(name="off", type="anthropic", enabled=False)
    bcfg.billing.sources = [s for s in bcfg.billing.sources if s.name == "openai"] + [off]
    async with providers() as c:
        await refresh_once(bcfg, store, state, today=TODAY, client=c, auth_factory=fake_auths())
        assert len(await store.bill_lines(DAY, TODAY, "openai")) == 1
        status = await refresh_once(bcfg, store, state, today=TODAY, client=c, auth_factory=fake_auths({"OPENAI_ADMIN_KEY"}))
    assert status["openai"]["ok"] is False and "HTTP 401: Incorrect API key provided" in status["openai"]["error"]
    assert "off" not in status  # disabled: never fetched
    assert status["openai"]["fetched_at"] and len(await store.bill_lines(DAY, TODAY, "openai")) == 1  # the last bill stays
    # Newest first: the failure, after the first fetch's gap (an empty ledger against a $2 bill).
    assert [a["title"] for a in await state.recent_alerts()] == ["Bill unavailable: openai", "Bill differs from the ledger: openai"]
    rep = await report(bcfg, store, DAY, TODAY, today=TODAY)
    assert rep["sources"][0]["ok"] is False and rep["sources"][0]["error"].startswith("OpenAI costs: HTTP 401")


async def test_missing_credentials_are_reported_not_fetched(bcfg, store, state, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_ADMIN_KEY", raising=False)
    bcfg.billing.sources = [s for s in bcfg.billing.sources if s.name == "anthropic"]
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: pytest.fail("no request without a key"))) as c:
        status = await refresh_once(bcfg, store, state, today=TODAY, client=c)
    assert status["anthropic"] == {"ok": False, "error": "ANTHROPIC_ADMIN_KEY is not set", "failed_at": status["anthropic"]["failed_at"]}
    rep = await report(bcfg, store, DAY, TODAY, today=TODAY)
    assert rep["sources"][0]["complete_through"] is None  # never fetched: nothing settled, nothing alerted


async def test_credentials_that_cant_load(bcfg, store, state, monkeypatch, tmp_path):
    import sys

    monkeypatch.setitem(sys.modules, "azure.identity.aio", None)  # the azure extra isn't installed
    for var in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_PROFILE", "AWS_DEFAULT_PROFILE", "AWS_ROLE_ARN",
                "AWS_WEB_IDENTITY_TOKEN_FILE", "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", "AWS_CONTAINER_CREDENTIALS_FULL_URI"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "none"))
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "none"))
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")

    def no_adc(spec):
        if spec.type == "google":
            raise RuntimeError("Your default credentials were not found")
        from meritorder.backends.auth import make_auth

        return make_auth(spec)

    bcfg.billing.sources = [s for s in bcfg.billing.sources if s.name in ("aws", "gcp", "azure")]
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: pytest.fail("no request without credentials"))) as c:
        status = await refresh_once(bcfg, store, state, today=TODAY, client=c, auth_factory=no_adc)
    assert status["aws"]["error"] == "no AWS credentials found (profile, environment, or role)"
    assert status["gcp"]["error"] == "no credentials: RuntimeError: Your default credentials were not found"
    assert status["azure"]["error"].startswith("credentials need the azure extra (uv sync --extra azure")


async def test_alerts_respect_the_thresholds(bcfg, store, state):
    """$0.80 on $60 is under 5%; $2 on $10 is over 5% and over $1; with no ledger at all, any gap over $1 counts."""
    bcfg.billing.sources = [BillSource(name="anthropic", type="anthropic")]
    await store.record(ledger_row(ts=at(DAY), backend="anthropic", served_model="claude-sonnet", upstream_model="claude-sonnet-5-5", actual_cost=10))
    await store.replace_bill_lines("anthropic", DAY - timedelta(days=5), TODAY, [
        bill("claude-sonnet-5-5 · output", model="claude-sonnet-5-5", cost=12.0),
        {**bill("claude-sonnet-5-5 · output", model="claude-sonnet-5-5", cost=1.5), "day": DAY - timedelta(days=1)},
        {**bill("claude-sonnet-5-5 · output", model="claude-sonnet-5-5", cost=0.5), "day": DAY - timedelta(days=2)}])
    await store.set_setting("billing_status", {"anthropic": {"ok": True, "fetched_at": "x", "covered_from": "2026-10-01"}})
    sent = await check_gaps(bcfg, store, state, TODAY)
    assert len(sent) == 1 and sent[0].startswith("The anthropic bill for 2026-10-20 is $12.00; the ledger recorded $10.00 (+$2.00, +20.0%).")
    assert "Also 1 more day (2026-10-19 +$1.50)." in sent[0]
    assert await check_gaps(bcfg, store, state, TODAY) == []  # each day once


async def test_the_worker_job_runs_once_per_refresh_period(bcfg, store, state, monkeypatch):
    calls = []

    async def fake_refresh(cfg, st, sta, **kw):
        calls.append(kw)
        return {}

    monkeypatch.setattr("meritorder.billing.refresh_once", fake_refresh)
    job = BillRefresher()
    assert await job.maybe(bcfg, store, state) == {} and await job.maybe(bcfg, store, state) is None
    assert len(calls) == 1
    await state.set_json("billing:last_run", (datetime.now(UTC) - timedelta(hours=7)).timestamp())
    await job.maybe(bcfg, store, state)
    assert len(calls) == 2
    bcfg.billing = BillingConfig()
    assert await job.maybe(bcfg, store, state) is None  # nothing configured, nothing fetched

    async def broken(*a, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr("meritorder.billing.refresh_once", broken)
    bcfg.billing = BillingConfig(sources=[BillSource(name="a", type="anthropic")])
    await state.set_json("billing:last_run", 0)
    assert await job.maybe(bcfg, store, state) is None  # logged, the worker carries on


# ------------------------------------------------------------------ storage


async def test_bill_lines_are_replaced_per_range_and_merged(store):
    lines = [bill("a", cost=1.0), bill("a", cost=2.0), {**bill("b", cost=4.0), "tokens": 10}, {**bill("b", cost=1.0), "tokens": 5},
             {**bill("c"), "day": DAY + timedelta(days=3)}]
    assert await store.replace_bill_lines("s1", DAY, DAY + timedelta(days=5), lines) == 3
    await store.replace_bill_lines("s2", DAY, DAY + timedelta(days=5), [bill("a", cost=9.0)])
    got = await store.bill_lines(DAY, DAY + timedelta(days=5), "s1")
    assert [(r["line"], r["cost"], r["tokens"]) for r in got] == [("a", 3.0, None), ("b", 5.0, 15), ("c", 1.0, None)]
    assert got[0]["detail"] == {} and got[0]["day"] == DAY and got[0]["source"] == "s1"
    # A later fetch of a shorter range replaces only that range.
    await store.replace_bill_lines("s1", DAY, DAY + timedelta(days=1), [bill("a", cost=7.0, tier="batch")])
    got = await store.bill_lines(DAY, DAY + timedelta(days=5))
    assert [(r["source"], r["line"], r["cost"]) for r in got] == [("s1", "a", 7.0), ("s2", "a", 9.0), ("s1", "c", 1.0)]
    assert got[0]["detail"] == {"tier": "batch"}


async def test_two_refreshes_of_one_source_at_once(store, monkeypatch):
    """The worker's refresh and an admin's "Refresh now" (or two workers) can store one source's bill at the same time.
    Both must work, and the bill is stored once."""
    await store.replace_bill_lines("s1", DAY, DAY + timedelta(days=5), [bill("a", cost=1.0), bill("b", cost=2.0)])
    if store.kind == "postgres":  # a refresh's delete and its insert a moment apart, as on a busy database
        import asyncpg

        insert = asyncpg.Connection.executemany

        async def slow(self, *args, **kw):
            await asyncio.sleep(0.2)
            return await insert(self, *args, **kw)

        monkeypatch.setattr(asyncpg.Connection, "executemany", slow)
    new = [bill("a", cost=3.0), bill("b", cost=4.0), bill("c", cost=5.0)]
    assert await asyncio.gather(*(store.replace_bill_lines("s1", DAY, DAY + timedelta(days=5), new) for _ in range(2))) == [3, 3]
    got = await store.bill_lines(DAY, DAY + timedelta(days=5), "s1")
    assert [(r["line"], r["cost"]) for r in got] == [("a", 3.0), ("b", 4.0), ("c", 5.0)]


async def test_ledger_costs_by_day_backend_and_model(store):
    for r in [ledger_row(ts=at(DAY, 1), backend="b1", served_model="m", upstream_model="m-1", actual_cost=1.0, input_tokens=10, output_tokens=2),
              ledger_row(ts=at(DAY, 23), backend="b1", served_model="m", upstream_model="m-1", actual_cost=2.0, cache_read_tokens=5),
              ledger_row(ts=at(DAY + timedelta(days=1), 0), backend="b1", served_model="m", upstream_model="m-1", actual_cost=4.0),
              ledger_row(ts=at(DAY, 2), backend="b2", served_model="m", upstream_model="m-2", actual_cost=8.0)]:
        await store.record(r)
    got = await store.ledger_costs(at(DAY, 0), at(DAY + timedelta(days=2), 0), ["b1"])
    got = sorted(got, key=lambda r: r["day"])
    assert got == [{"day": DAY, "backend": "b1", "served_model": "m", "upstream_model": "m-1", "service_tier": "", "requests": 2,
                    "input_tokens": 10,
                    "cache_read_tokens": 5, "cache_write_tokens": 0, "output_tokens": 2, "actual_cost": 3.0},
                   {"day": DAY + timedelta(days=1), "backend": "b1", "served_model": "m", "upstream_model": "m-1", "service_tier": "", "requests": 1,
                    "input_tokens": 0,
                    "cache_read_tokens": 0, "cache_write_tokens": 0, "output_tokens": 0, "actual_cost": 4.0}]


# ------------------------------------------------------------------ reports API and CLI


async def seed_api(gateway):
    gateway.cfg.billing = BillingConfig(sources=[BillSource(name="anthropic", type="anthropic", backends=["mock-anthropic"]),
                                                 BillSource(name="aws", type="aws", backends=[])])
    gateway.cfg.backends.append(BackendConfig(name="gpu-x", kind="self_hosted", format="openai", serve=ServeConfig(model="big")))
    await gateway.store.add_gpu_usage("gpu-x", at(DAY), 3600, 10.0)
    await gateway.store.replace_bill_lines("aws", DAY, DAY + timedelta(days=1), [
        {**bill("gpu-x · Amazon EC2", kind="gpu", cost=12.0), "deployment": "gpu-x"}])
    await gateway.store.record(ledger_row(ts=at(DAY), backend="mock-anthropic", served_model="big", upstream_model="big-1", actual_cost=4.0))
    await gateway.store.replace_bill_lines("anthropic", DAY, DAY + timedelta(days=1), [
        bill("big-1 · uncached_input_tokens", model="big-1", cost=4.5), bill("mystery", cost=1.0)])
    fetched = {"ok": True, "fetched_at": "2026-10-25T00:00:00+00:00", "covered_from": "2026-10-01", "complete_through": "2026-10-23"}
    await gateway.store.set_setting("billing_status", {"anthropic": fetched, "aws": fetched})
    await gateway.store.rollup()


async def test_bills_api_for_finance_and_admins(client, gateway):
    await seed_api(gateway)
    fin = (await gateway.store.create_key("fin", "ops", "finance", name="finance"))[1]
    params = {"start": "2026-10-01", "end": "2026-11-01"}
    for key in (fin, ADMIN_KEY):
        r = await client.get("/v1/reports/bills", params=params, headers=auth(key))
        assert r.status_code == 200, r.text
        rep = r.json()
        assert rep["rows"][0] | {"recorded_tokens": None, "billed_tokens": None} == {
            "period": "2026-10-20", "source": "anthropic", "type": "anthropic", "model": "big", "recorded": 4.0, "billed": 4.5,
            "difference": 0.5, "pct": 12.5, "requests": 1, "lines": 1, "partial": False, "status": "matched", "over": False,
            "billed_by_kind": {"input": 4.5}, "recorded_tokens": None, "billed_tokens": None, "recorded_batch": 0.0, "billed_batch": 0.0}
        assert rep["unmatched"] == [{"source": "anthropic", "line": "mystery", "status": "unmatched", "why": "no routed model's name in this line",
                                     "cost": 1.0, "days": 1, "first": "2026-10-20", "last": "2026-10-20"}]
        assert rep["settled"]["difference"] == 1.5 and rep["sources"][0]["complete_through"] is not None
        assert rep["gpu"][0] | {"billed_by_source": None} == {"period": "2026-10-20", "deployment": "gpu-x", "recorded": 10.0, "billed": 12.0,
                                                               "difference": 2.0, "pct": 20.0, "partial": False, "over": True,
                                                               "billed_by_source": None}
    for key in (LEAD_KEY,):
        assert (await client.get("/v1/reports/bills", headers=auth(key))).status_code == 403
    assert (await client.get("/v1/reports/bills", params={"start": "2026-10-05", "end": "2026-10-01"}, headers=auth(ADMIN_KEY))).status_code == 400
    assert (await client.get("/v1/reports/bills", params={"by": "week"}, headers=auth(ADMIN_KEY))).status_code == 422
    csv_text = (await client.get("/v1/reports/bills.csv", params=params, headers=auth(fin))).text
    assert csv_text.splitlines()[:3] == ["period,source,model,status,recorded,billed,difference,pct,requests",
                                         "2026-10-20,anthropic,big,matched,4.0,4.5,0.5,12.5,1",
                                         "2026-10-20..2026-10-20,anthropic,mystery,unmatched: no routed model's name in this line,,1.0,,,"]
    assert csv_text.splitlines()[-1] == "2026-10-20,gpu,gpu-x,gpu,10.0,12.0,2.0,20.0,"


async def test_bills_refresh_endpoint(client, gateway, monkeypatch):
    assert (await client.post("/v1/reports/bills/refresh", headers=auth(ADMIN_KEY))).status_code == 400  # nothing configured
    gateway.cfg.billing = BillingConfig(sources=[BillSource(name="anthropic", type="anthropic", backends=["mock-anthropic"])])
    assert (await client.post("/v1/reports/bills/refresh", headers=auth(LEAD_KEY))).status_code == 403
    monkeypatch.delenv("ANTHROPIC_ADMIN_KEY", raising=False)
    r = await client.post("/v1/reports/bills/refresh", headers=auth(ADMIN_KEY))
    assert r.status_code == 200 and r.json()["sources"]["anthropic"]["error"] == "ANTHROPIC_ADMIN_KEY is not set"


def test_usage_bills_in_the_cli(cfg, capsys):
    import asyncio

    from test_cli import LiveGateway

    with LiveGateway(cfg) as live:
        asyncio.run(seed_api(live.gw))
        cli.main(["usage", "--bills", "--start", "2026-10-01", "--end", "2026-11-01", "--gateway", live.url, "--key", ADMIN_KEY, "--limit", "0"])
        out = capsys.readouterr().out
        assert out.startswith("Ledger against bills, 2026-10-01 to 2026-11-01")
        assert "anthropic (anthropic: mock-anthropic), ok; complete through" in out
        assert "recorded $4.00  billed $5.50  difference +1.50 (+37.5%)  not matched $1.00" in out
        assert "2026-10-20 anthropic    big" in out and "+12.5%  matched" in out
        assert "... and 1 more (--limit)" in out
        live.gw.cfg.billing = BillingConfig()
        cli.main(["usage", "--bills", "--gateway", live.url, "--key", ADMIN_KEY])
        assert "no bill sources configured" in capsys.readouterr().out
        with pytest.raises(SystemExit, match="^403"):
            cli.main(["usage", "--bills", "--gateway", live.url, "--key", LEAD_KEY])


def test_config_file_example_parses(tmp_path):
    """The commented example in config/meritorder.yaml, uncommented, is a valid billing section."""
    text = (Path(__file__).parent.parent / "config" / "meritorder.yaml").read_text()
    block = text.split("# billing:", 1)[1].split("\n\n", 1)[0]
    example = "billing:" + "\n".join(line.removeprefix("# ").removeprefix("#") for line in block.splitlines())
    raw = {**RAW, **yaml.safe_load(example)}
    cfg = build_config(raw, tmp_path, quiet=True)
    assert {s.type for s in cfg.billing.sources} == {"anthropic", "openai", "aws", "gcp", "azure"}
    # The export alternatives, uncommented in place of the APIs they replace (both at once would count every day twice).
    swapped = "\n".join(line.replace("# aws-cur:", "aws-cur:").replace("# azure-export:", "azure-export:") for line in example.splitlines()
                        if not line.lstrip().startswith(("aws:", "azure:")))
    cfg = build_config({**RAW, **yaml.safe_load(swapped)}, tmp_path, quiet=True)
    assert {s.type for s in cfg.billing.sources} == {"anthropic", "openai", "aws_cur", "gcp", "azure_export"}


def test_usage_bills_prints_every_section(monkeypatch, capsys):
    rep = {"start": "2026-10-01", "end": "2026-11-01", "by": "day", "settled": {}, "thresholds": {"alert_pct": 5, "alert_min_usd": 1},
           "sources": [{"name": "aws", "type": "aws", "backends": [], "ok": False, "error": "AccessDeniedException: no", "complete_through": None,
                        "settled": {"recorded": 1.0, "billed": 0.0, "difference": -1.0, "pct": None, "billed_unmatched": 0.0, "adjustments": 0.0}},
                       {"name": "gcp", "type": "gcp", "backends": ["vertex"], "ok": None, "error": None, "complete_through": None,
                        "settled": {"recorded": 0.0, "billed": 0.0, "difference": 0.0, "pct": None, "billed_unmatched": 0.0, "adjustments": 0.0}},
                       {"name": "cur", "type": "aws_cur", "backends": ["bedrock"], "ok": True, "error": None, "complete_through": None,
                        "notes": ["No export files found under s3://b/cur for 2026-10: check the path."],
                        "settled": {"recorded": 0.0, "billed": 0.0, "difference": 0.0, "pct": None, "billed_unmatched": 0.0, "adjustments": 0.0}}],
           "rows": [{"period": "2026-10-20", "source": "aws", "model": "claude-sonnet", "recorded": 50.0, "billed": 60.0, "difference": 10.0,
                     "pct": 20.0, "status": "matched", "over": True}],
           "unmatched": [{"source": "aws", "cost": 2.0, "line": "Nova Pro · Input tokens", "status": "unmatched", "why": "not routed"}],
           "gpu": [{"period": "2026-10-20", "deployment": "gpu-qwen", "recorded": 90.0, "billed": 96.0, "difference": 6.0, "pct": 6.67,
                    "partial": True}],
           "uncovered": [{"backend": "xai", "recorded": 3.0, "why": "no bill source covers it (billing.sources)"}]}
    monkeypatch.setattr(cli.httpx, "get", lambda url, params, headers, timeout: httpx.Response(200, json=rep))
    cli.main(["usage", "--bills", "--gateway", "http://gw.test", "--key", ADMIN_KEY])
    out = capsys.readouterr().out
    assert "aws (aws: no backends), failed: AccessDeniedException: no; complete through –" in out
    assert "gcp (gcp: vertex), not fetched yet" in out
    assert "cur (aws_cur: bedrock), ok; complete through –" in out
    assert "    note: No export files found under s3://b/cur for 2026-10: check the path." in out
    assert "+20.0%  matched (over the alert threshold)" in out
    assert "Nova Pro · Input tokens" in out and "not matched: not routed" in out
    assert "gpu-qwen" in out and "+6.7%  partial" in out
    assert "not checked: xai $3.00 (no bill source covers it (billing.sources))" in out
