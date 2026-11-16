"""OpenTelemetry traces: one span per LLM request, in the GenAI semantic conventions.

Off unless an OTLP endpoint is configured (OTEL_EXPORTER_OTLP_ENDPOINT or
OTEL_EXPORTER_OTLP_TRACES_ENDPOINT) and the `otel` extra is installed. If a client
sends a W3C `traceparent` header, the gateway's span joins that trace, so an
agent's own spans and its LLM calls line up in one view.
"""

from __future__ import annotations

import logging
import os
import time

log = logging.getLogger("meritorder.telemetry")

try:  # the API is a light dependency; without an SDK every call below is a no-op
    from opentelemetry import propagate, trace
    from opentelemetry.trace import SpanKind, Status, StatusCode
except ImportError:  # pragma: no cover
    trace = None

PROVIDERS = {"anthropic": "anthropic", "bedrock": "aws.bedrock", "vertex": "gcp.vertex_ai", "foundry": "az.ai.inference",
             "azure": "az.ai.openai", "openai": "openai", "openrouter": "openrouter", "deepseek": "deepseek", "mistral": "mistral_ai"}


def setup_tracing() -> bool:
    if trace is None or not (os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") or os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")):
        return False
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        log.warning("OTLP endpoint is set but tracing isn't installed: `uv sync --extra otel` (the image has it)")
        return False
    provider = TracerProvider(resource=Resource.create({"service.name": os.environ.get("OTEL_SERVICE_NAME", "meritorder-gateway")}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    log.info("OpenTelemetry tracing on")
    return True


def provider_name(backend: str | None) -> str:
    b = (backend or "").lower()
    return next((v for k, v in PROVIDERS.items() if k in b), b or "unknown")


def record_span(row: dict, carrier: dict | None = None) -> None:
    """Emit a finished span for one request (start = now - latency)."""
    if trace is None:
        return
    tracer = trace.get_tracer("meritorder")
    end_ns = time.time_ns()
    start_ns = end_ns - int((row.get("latency_ms") or 0) * 1_000_000)
    operation = "embeddings" if row.get("client_format") == "embeddings" else "chat"
    parent = propagate.extract(carrier) if carrier else None
    span = tracer.start_span(f"{operation} {row.get('served_model') or row.get('requested_model') or ''}".strip(),
                             context=parent, kind=SpanKind.CLIENT, start_time=start_ns)
    attrs = {
        "gen_ai.operation.name": operation,
        "gen_ai.system": provider_name(row.get("backend")),
        "gen_ai.request.model": row.get("requested_model") or "",
        "gen_ai.response.model": row.get("upstream_model") or row.get("served_model") or "",
        "gen_ai.usage.input_tokens": int((row.get("input_tokens") or 0) + (row.get("cache_read_tokens") or 0) + (row.get("cache_write_tokens") or 0)),
        "gen_ai.usage.output_tokens": int(row.get("output_tokens") or 0),
        "meritorder.model": row.get("served_model") or "",
        "meritorder.backend": row.get("backend") or "",
        "meritorder.team": row.get("team") or "",
        "meritorder.user": row.get("user_id") or "",
        "meritorder.session": row.get("session_id") or "",
        "meritorder.cache_read_tokens": int(row.get("cache_read_tokens") or 0),
        "meritorder.cost_usd": float(row.get("actual_cost") or 0),
        "meritorder.baseline_usd": float(row.get("baseline_cost") or 0),
        "meritorder.free_upgrade": bool(row.get("upgraded")),
    }
    if row.get("floor"):
        attrs["meritorder.floor"] = row["floor"]
    if row.get("ttft_ms") is not None:
        attrs["meritorder.time_to_first_token_ms"] = int(row["ttft_ms"])
    span.set_attributes(attrs)
    status = row.get("status") or 0
    if status >= 400:
        span.set_attribute("error.type", str(status))
        span.set_status(Status(StatusCode.ERROR, (row.get("error") or "")[:200]))
    elif row.get("error"):
        span.set_attribute("meritorder.note", row["error"][:200])
    span.end(end_time=end_ns)
