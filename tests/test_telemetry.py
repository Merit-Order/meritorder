from conftest import auth
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

EXPORTER = InMemorySpanExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(EXPORTER))
trace.set_tracer_provider(_provider)


async def test_spans_follow_genai_conventions_and_join_client_trace(client, gateway):
    EXPORTER.clear()
    parent_trace = "4bf92f3577b34da6a3ce929d0e0e4736"
    headers = {**auth(), "traceparent": f"00-{parent_trace}-00f067aa0ba902b7-01"}
    r = await client.post("/v1/chat/completions", headers=headers, json={"model": "small", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    await gateway.drain()
    spans = EXPORTER.get_finished_spans()
    [span] = [s for s in spans if s.instrumentation_scope.name == "meritorder"]
    assert span.name == "chat small"
    assert format(span.context.trace_id, "032x") == parent_trace  # joined the agent's trace
    a = span.attributes
    assert a["gen_ai.operation.name"] == "chat" and a["gen_ai.request.model"] == "small" and a["gen_ai.response.model"] == "small-1"
    assert a["gen_ai.usage.input_tokens"] > 0 and a["meritorder.backend"] == "mock-openai" and a["meritorder.cost_usd"] > 0
