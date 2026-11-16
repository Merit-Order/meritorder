"""Generate the Grafana dashboard for Merit Order's /metrics (charts/meritorder/dashboards/meritorder.json).

    uv run python scripts/build_grafana_dashboard.py

Counters are per gateway process, so traffic panels sum across pods. GPU gauges come from the
controller's published state and every gateway reports the same value, so those use max(). Batch requests are
counted by whoever settles them (the worker, on its own metrics port), only in the meritorder_batch_* counters, so
adding them to the live ones never counts a request twice.
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "charts" / "meritorder" / "dashboards" / "meritorder.json"
DS = {"type": "prometheus", "uid": "${datasource}"}
_id = 0


def panel(title, exprs, x, y, w=8, h=8, kind="timeseries", unit="short", stack=False, desc="", decimals=None, thresholds=None):
    global _id
    _id += 1
    targets = [{"datasource": DS, "expr": e, "legendFormat": legend, "refId": chr(65 + i), "range": True}
               for i, (e, legend) in enumerate(exprs)]
    defaults: dict = {"unit": unit}
    if decimals is not None:
        defaults["decimals"] = decimals
    if thresholds:
        defaults["thresholds"] = {"mode": "absolute", "steps": thresholds}
        defaults["color"] = {"mode": "thresholds"}
    if kind == "timeseries":
        defaults["custom"] = {"drawStyle": "line", "lineWidth": 1, "fillOpacity": 18 if stack else 6, "showPoints": "never",
                              "stacking": {"mode": "normal" if stack else "none", "group": "A"}, "spanNulls": True}
    p = {"id": _id, "type": kind, "title": title, "description": desc, "datasource": DS, "targets": targets,
         "gridPos": {"x": x, "y": y, "w": w, "h": h}, "fieldConfig": {"defaults": defaults, "overrides": []}}
    if kind == "stat":
        p["options"] = {"reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False}, "colorMode": "value",
                        "graphMode": "area", "textMode": "auto", "justifyMode": "auto"}
    if kind == "timeseries":
        p["options"] = {"legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
                        "tooltip": {"mode": "multi", "sort": "desc"}}
    return p


def row(title, y):
    global _id
    _id += 1
    return {"id": _id, "type": "row", "title": title, "collapsed": False, "gridPos": {"x": 0, "y": y, "w": 24, "h": 1}, "panels": []}


R = "$__rate_interval"
# Spend with GPUs counted as they bill (decision 64): requests' cost, except the shares of GPUs the controller runs, plus
# those GPUs' burn, idle and starting time included. A share is already in the burn, so counting both would count it twice.
API_H = f'(sum(rate(meritorder_cost_usd_total{{kind!="gpu_share"}}[{R}])) or vector(0)) * 3600'
BURN = "(sum(max by (backend) (meritorder_gpu_usd_per_hour)) or vector(0))"
SHARE_H = f'(sum(rate(meritorder_cost_usd_total{{kind="gpu_share"}}[{R}])) or vector(0)) * 3600'
COST_H = f"{API_H} + {BURN}"
BASE_H = f"sum(rate(meritorder_baseline_usd_total[{R}])) * 3600"
SAVE_H = f"sum(rate(meritorder_savings_usd_total[{R}])) * 3600"
REQS = f"sum(rate(meritorder_requests_total[{R}]))"
# Backend errors (in the ledger) plus requests refused before any backend (auth, budget, rate limit, routing).
ERRS = (f'(sum(rate(meritorder_requests_total{{status!~"2.."}}[{R}])) or vector(0)) + '
        f"(sum(rate(meritorder_rejected_requests_total[{R}])) or vector(0))")
ALL = f"({REQS} or vector(0)) + (sum(rate(meritorder_rejected_requests_total[{R}])) or vector(0))"


def build() -> dict:
    panels = [
        row("Spend and savings", 0),
        panel("Spend", [(COST_H, "spend")], 0, 1, w=4, h=5, kind="stat", unit="currencyUSD", decimals=2,
              desc="What is being spent per hour right now: requests on APIs, plus what running GPU replicas cost, idle or busy."),
        panel("Would have cost", [(BASE_H, "baseline")], 4, 1, w=4, h=5, kind="stat", unit="currencyUSD", decimals=2,
              desc="The same requests at the baseline model's API price, per hour."),
        panel("Saving", [(f"({BASE_H} - ({COST_H})) / clamp_min({BASE_H}, 1e-9)", "saving")], 8, 1, w=4, h=5, kind="stat",
              unit="percentunit", decimals=0, desc="What the requests would have cost less what is being spent, GPUs included, as a "
                                                   "share of what they would have cost. Negative when GPUs cost more than they save."),
        panel("GPU burn", [("sum(max by (backend) (meritorder_gpu_usd_per_hour)) or vector(0)", "GPUs")], 12, 1, w=4, h=5, kind="stat",
              unit="currencyUSD", decimals=2, desc="What running GPU replicas cost per hour (SkyPilot catalog prices)."),
        panel("Requests", [(ALL, "req/s")], 16, 1, w=4, h=5, kind="stat", unit="reqps", decimals=1),
        panel("Errors", [(f"({ERRS}) / clamp_min({ALL}, 1e-9)", "errors")], 20, 1, w=4, h=5, kind="stat",
              unit="percentunit", decimals=1, desc="Failed requests, including ones refused before reaching a backend.",
              thresholds=[{"color": "green", "value": None}, {"color": "orange", "value": 0.02}, {"color": "red", "value": 0.05}]),
        panel("Spend by model ($/h)", [(f"sum by (model) (rate(meritorder_cost_usd_total[{R}])) * 3600", "{{model}}")],
              0, 6, w=12, unit="currencyUSD", stack=True),
        panel("Savings by cause ($/h)", [(f"sum by (cause) (rate(meritorder_savings_usd_total[{R}])) * 3600", "{{cause}}"),
                                         (f"{SHARE_H} - {BURN}", "GPU time no request paid for")],
              12, 6, w=12, unit="currencyUSD", stack=True,
              desc="routing: a cheaper model; caching: prompt-cache reads; self_hosting: GPUs instead of APIs, at each request's "
                   "share of the GPU; GPU time no request paid for: the rest of the GPUs' burn (idle, starting, draining), which "
                   "comes off self_hosting; "
                   "compression: tool results shortened by the rules; compression_model: by model-based compression (LLMLingua-2); "
                   "response_cache: exact repeats served from cache. Live traffic only: batches are in the Batches row "
                   "below (the worker's meritorder_batch_* counters)."),
        row("Traffic", 14),
        panel("Requests by model", [(f"sum by (model) (rate(meritorder_requests_total[{R}]))", "{{model}}")], 0, 15, unit="reqps", stack=True),
        panel("Requests by backend", [(f"sum by (backend) (rate(meritorder_requests_total[{R}]))", "{{backend}}")], 8, 15, unit="reqps",
              stack=True),
        panel("Errors by backend and refusals by reason", [
            (f'sum by (backend) (rate(meritorder_requests_total{{status!~"2.."}}[{R}]))', "{{backend}}"),
            (f"sum by (reason) (rate(meritorder_rejected_requests_total[{R}]))", "refused: {{reason}}")],
              16, 15, unit="reqps",
              desc="Backend failures by backend, and requests refused before a backend: rate_limit_error is budgets "
                   "and rate limits, overloaded_error is nowhere to run, authentication_error is bad keys."),
        panel("Tokens by kind", [(f"sum by (kind) (rate(meritorder_tokens_total[{R}]))", "{{kind}}")], 0, 23, unit="short", stack=True,
              desc="Input, cache reads, cache writes, and output tokens per second."),
        panel("Prompt-cache hit ratio", [(f'(sum(rate(meritorder_tokens_total{{kind="cache_read"}}[{R}])) or vector(0)) / '
                                          f'clamp_min(sum(rate(meritorder_tokens_total{{kind=~"input|cache_read|cache_write"}}[{R}])), 1e-9)',
                                          "cached share of input")], 8, 23, unit="percentunit",
              desc="Share of input tokens read from a prompt cache. Stable sessions keep this high."),
        panel("Backends available", [("min by (backend) (meritorder_backend_available)", "{{backend}}")], 16, 23, unit="bool_on_off",
              desc="Enabled, running, and circuit breaker closed."),
        row("Latency", 31),
        panel("Time to first token (p50 / p95)", [
            (f"histogram_quantile(0.5, sum by (le, model) (rate(meritorder_time_to_first_token_seconds_bucket[{R}])))", "p50 {{model}}"),
            (f"histogram_quantile(0.95, sum by (le, model) (rate(meritorder_time_to_first_token_seconds_bucket[{R}])))", "p95 {{model}}")],
              0, 32, w=12, unit="s"),
        panel("Request duration (p50 / p95)", [
            (f"histogram_quantile(0.5, sum by (le, model) (rate(meritorder_request_duration_seconds_bucket[{R}])))", "p50 {{model}}"),
            (f"histogram_quantile(0.95, sum by (le, model) (rate(meritorder_request_duration_seconds_bucket[{R}])))", "p95 {{model}}")],
              12, 32, w=12, unit="s"),
        row("GPUs", 40),
        panel("Replicas", [("max by (backend) (meritorder_gpu_replicas)", "{{backend}}")], 0, 41, w=6, unit="short", stack=True),
        panel("Burn ($/h)", [("max by (backend) (meritorder_gpu_usd_per_hour)", "{{backend}}")], 6, 41, w=6, unit="currencyUSD", stack=True),
        panel("Utilization", [("max by (backend) (meritorder_gpu_utilization)", "{{backend}}")], 12, 41, w=6, unit="percentunit",
              desc="vLLM KV-cache use, pushed up by waiting requests. At 100% new work spills to other routes."),
        panel("Waiting and in flight", [("max by (backend) (meritorder_gpu_waiting_requests)", "waiting in vLLM {{backend}}"),
                                        ('sum by (backend) (meritorder_inflight_requests{backend=~".+"})', "in flight {{backend}}")],
              18, 41, w=6, unit="short"),
        row("Batches", 49),
        panel("Batch spend ($/h)", [(f"sum by (model) (rate(meritorder_batch_cost_usd_total[{R}])) * 3600", "{{model}}")],
              0, 50, w=8, unit="currencyUSD", stack=True,
              desc="What settled batch requests cost, from the worker that settles them (MERITORDER_WORKER_METRICS_PORT). "
                   "Not in Spend above, which is live traffic: add the two for everything."),
        panel("Batch savings by cause ($/h)", [(f"sum by (cause) (rate(meritorder_batch_savings_usd_total[{R}])) * 3600", "{{cause}}")],
              8, 50, w=8, unit="currencyUSD", stack=True,
              desc="batch: the providers' batch discount; self_hosting: batch work run on GPUs below the batch price; routing: "
                   "a cheaper model than the team's default."),
        panel("Batch requests by where they ran", [
            (f"sum by (api, status) (rate(meritorder_batch_requests_total[{R}]))", "{{api}} {{status}}"),
            (f"sum by (why) (rate(meritorder_batch_handoffs_total[{R}]))", "handed from a GPU: {{why}}")],
              16, 50, w=8, unit="reqps", desc="Settled batch requests by batch API (gpu: run on a self-hosted GPU) and status, "
                                              "and GPU batch work moved to a provider's batch API (deadline, stopped, failed)."),
    ]
    return {
        "title": "Merit Order", "uid": "meritorder", "tags": ["meritorder", "llm"], "timezone": "browser",
        "schemaVersion": 39, "version": 1, "editable": True, "refresh": "30s", "time": {"from": "now-6h", "to": "now"},
        "description": "Spend, savings, traffic, latency, and GPUs from Merit Order's /metrics.",
        "templating": {"list": [{"name": "datasource", "label": "Prometheus", "type": "datasource", "query": "prometheus",
                                 "current": {}, "hide": 0, "refresh": 1}]},
        "annotations": {"list": []}, "links": [], "panels": panels,
    }


if __name__ == "__main__":
    OUT.write_text(json.dumps(build(), indent=1) + "\n")
    print(f"wrote {OUT}")
