"""Eval grading edge cases, task runs that fail, suite loading; and tracing setup with an OTLP exporter."""

import json
import sys
import types

import httpx
import pytest

from meritorder import telemetry
from meritorder.evals.runner import grade, load_suite, run_task


def reply(text="", calls=()):
    msg = {"role": "assistant", "content": text}
    if calls:
        msg["tool_calls"] = [{"id": f"c{i}", "type": "function", "function": {"name": n, "arguments": a}} for i, (n, a) in enumerate(calls)]
    return {"choices": [{"message": msg}]}


@pytest.mark.parametrize("check, text, passed, why", [
    ({"number": 42}, "no digits here", False, "no number in the reply"),
    ({"number": 3.14, "tolerance": 0.01}, "pi is about 3.141", True, ""),
    ({"number": 10}, "first 3 then 10", True, ""),  # the last number is the answer
    ({"json": {"a": 1}}, "not json at all", False, "reply isn't valid JSON"),
    ({"json": {"a": 1}}, "[1, 2]", False, "reply isn't a JSON object"),
    ({"json": {"a": 1}}, '{"a": 1, "b": 2}', True, ""),
    ({"regex": r"^\d+ items$"}, "12 items", True, ""),
    ({"regex_behaves": {"match": ["a"]}}, "x" * 600, False, "regex too long"),
    ({"regex_behaves": {"match": ["a"]}}, "(unclosed", False, "not a valid regex"),
    ({"regex_behaves": {"match": ["ab"], "reject": ["abc"]}}, "/ab/", True, ""),
    ({"telepathy": True}, "anything", False, "unknown check telepathy"),
])
def test_grade_edge_cases(check, text, passed, why):
    ok, reason = grade(check, reply(text))
    assert ok is passed and why in reason


def test_tool_call_grading():
    want = {"tool_call": {"name": "get_weather", "arguments": {"city": "Oslo"}}}
    assert grade(want, reply(calls=[("other", "{}"), ("get_weather", '{"city": " oslo "}')]))[0]  # skips other tools
    ok, why = grade(want, reply(calls=[("get_weather", "{broken")]))
    assert not ok and "aren't valid JSON" in why
    ok, why = grade(want, reply(calls=[("get_weather", '{"city": "Bergen"}')]))
    assert not ok and "expected {'city': 'Oslo'}" in why
    assert grade({"tool_call": {"name": "noop"}}, reply(calls=[("noop", None)]))[0]  # no arguments needed
    assert grade({"contains": "x"}, {})[0] is False  # an empty reply never passes


def test_load_suite_validation_and_names(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("tasks:\n  - id: t1\n    task: coding\n    prompt: hi\n")
    with pytest.raises(ValueError, match="t1 needs id, task, level, prompt, and checks"):
        load_suite(bad)
    good = tmp_path / "my-suite.yaml"
    good.write_text("tasks:\n  - {id: t1, task: coding, level: 1, prompt: hi, checks: [{contains: hi}]}\n")
    assert load_suite(good)["name"] == "my-suite"
    named = tmp_path / "x.yaml"
    named.write_text("name: custom\ntasks: []\n")
    assert load_suite(named) == {"name": "custom", "tasks": []}


async def test_run_task_sends_tools_and_reports_failures():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append((body, dict(request.headers)))
        if "boom" in body["messages"][0]["content"]:
            return httpx.Response(429, text="slow down", headers={"x-meritorder-cost-usd": "0"})
        return httpx.Response(200, json=reply("42"), headers={"x-meritorder-cost-usd": "0.0012"})

    task = {"id": "t", "task": "coding", "level": 2, "prompt": "what is 6*7", "checks": [{"number": 42}],
            "tools": [{"type": "function", "function": {"name": "calc"}}], "max_tokens": 50}
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
        ok = await run_task(c, "http://gw", "mo-key", "small", task, "suite-1")
        failed = await run_task(c, "http://gw", "mo-key", "small", {**task, "prompt": "boom"}, "suite-1")
    assert ok.passed and ok.cost == 0.0012 and ok.error is None
    body, headers = seen[0]
    assert body["tools"] == task["tools"] and body["max_tokens"] == 50
    assert headers["x-meritorder-cache"] == "off" and headers["x-meritorder-tags"] == "eval=suite-1"
    assert headers["x-session-id"].startswith("eval-")
    assert not failed.passed and failed.error == "HTTP 429: slow down"

    def down(request):
        raise httpx.ConnectError("refused")

    async with httpx.AsyncClient(transport=httpx.MockTransport(down)) as c:
        res = await run_task(c, "http://gw", "k", "small", task, "s")
    assert res.skipped and res.error == "not graded: lost the connection to the gateway (after 1 try): ConnectError: refused"


# ------------------------------------------------------------------ tracing setup


def test_setup_tracing_with_an_otlp_exporter(monkeypatch, caplog):
    exporters = []

    class FakeExporter:
        def __init__(self):
            exporters.append(self)

        def export(self, spans):  # pragma: no cover - the batch processor isn't flushed here
            return None

        def shutdown(self):
            return None

    mod = types.ModuleType("opentelemetry.exporter.otlp.proto.http.trace_exporter")
    mod.OTLPSpanExporter = FakeExporter
    monkeypatch.setitem(sys.modules, "opentelemetry.exporter.otlp.proto.http.trace_exporter", mod)
    providers = []
    monkeypatch.setattr(telemetry.trace, "set_tracer_provider", providers.append)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://collector:4318/v1/traces")
    monkeypatch.setenv("OTEL_SERVICE_NAME", "mo-test")
    caplog.set_level("INFO")
    assert telemetry.setup_tracing() is True
    assert exporters and providers
    assert providers[0].resource.attributes["service.name"] == "mo-test"
    assert "tracing on" in caplog.text
    for p in providers:
        p.shutdown()


def test_record_span_is_a_no_op_without_opentelemetry(monkeypatch):
    monkeypatch.setattr(telemetry, "trace", None)
    telemetry.record_span({"served_model": "small", "latency_ms": 5})  # must not raise
    assert telemetry.setup_tracing() is False
