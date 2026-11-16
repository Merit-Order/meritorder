"""Smoke-test every enabled provider through a running gateway (costs a few cents at most).

For each enabled backend, picks its cheapest chat model and checks OpenAI format (plain,
streaming, tool call) and Anthropic format (plain, streaming); and its cheapest embedding model, if any.

    uv run python scripts/smoke_providers.py --gateway http://localhost:8080 --key $MERITORDER_DEV_KEY
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import httpx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from meritorder.backends import build_backends  # noqa: E402
from meritorder.cli import load_dotenv  # noqa: E402
from meritorder.config import load_config  # noqa: E402

TOOL = {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}


def sse(text: str) -> list[dict]:
    out = []
    for line in text.splitlines():
        if line.startswith("data: ") and line[6:].strip() not in ("", "[DONE]"):
            try:
                out.append(json.loads(line[6:]))
            except json.JSONDecodeError:
                pass
    return out


def run(gateway: str, key: str, only: list[str] | None) -> int:
    cfg = load_config()
    backends = build_backends(cfg)
    targets = []
    for name, b in backends.items():
        if b.kind == "mock" or (only and name not in only):
            continue
        for kind in ("chat", "embedding"):
            routes = [(r.cost_price().input, m.name) for m in cfg.models if m.kind == kind
                      for r in m.routes if r.backend == name and r.priced]
            if routes:
                targets.append((name, min(routes)[1], kind))
    if not targets:
        print("No enabled provider backends. Add keys to .env and run `uv run meritorder check`.")
        return 1
    failures = 0
    c = httpx.Client(timeout=120)
    print(f"{'backend':16} {'model':22} {'test':18} result")
    for backend, model, kind in targets:
        h = {"authorization": f"Bearer {key}", "x-meritorder-backend": backend, "content-type": "application/json"}
        msg = [{"role": "user", "content": "Reply with exactly: pong"}]
        tests = {"embeddings": lambda: c.post(f"{gateway}/v1/embeddings", headers=h, json={"model": model, "input": ["pong"]})} \
            if kind == "embedding" else {
            "openai": lambda: c.post(f"{gateway}/v1/chat/completions", headers=h, json={"model": model, "max_tokens": 512, "messages": msg}),
            "openai stream": lambda: c.post(f"{gateway}/v1/chat/completions", headers=h,
                                            json={"model": model, "max_tokens": 512, "stream": True, "stream_options": {"include_usage": True}, "messages": msg}),
            "openai tool call": lambda: c.post(f"{gateway}/v1/chat/completions", headers=h, json={
                "model": model, "max_tokens": 512, "tools": [TOOL],
                "messages": [{"role": "user", "content": "What's the weather in Paris? Use the tool."}]}),
            "anthropic": lambda: c.post(f"{gateway}/v1/messages", headers=h, json={"model": model, "max_tokens": 512, "messages": msg}),
            "anthropic stream": lambda: c.post(f"{gateway}/v1/messages", headers=h, json={"model": model, "max_tokens": 512, "stream": True, "messages": msg}),
        }
        for test, call in tests.items():
            t0 = time.time()
            try:
                r = call()
                ok, detail = r.status_code == 200, f"HTTP {r.status_code}"
                if ok and test == "embeddings":
                    vec = (r.json().get("data") or [{}])[0].get("embedding") or []
                    ok, detail = bool(vec), f"{len(vec)} dims"
                elif ok and test == "openai":
                    text = r.json()["choices"][0]["message"].get("content") or ""
                    ok, detail = bool(text.strip()), repr(text[:30])
                elif ok and test == "openai stream":
                    chunks = sse(r.text)
                    text = "".join((ch.get("choices") or [{}])[0].get("delta", {}).get("content") or "" for ch in chunks if ch.get("choices"))
                    usage = next((ch["usage"] for ch in chunks if ch.get("usage")), None)
                    ok = bool(text) and bool(usage)
                    detail = f"{text[:30]!r} usage={'yes' if usage else 'MISSING'}"
                elif ok and test == "openai tool call":
                    calls = r.json()["choices"][0]["message"].get("tool_calls") or []
                    ok = bool(calls) and "city" in json.loads(calls[0]["function"]["arguments"] or "{}")
                    detail = calls[0]["function"]["arguments"][:40] if calls else "no tool call"
                elif ok and test == "anthropic":
                    text = "".join(b.get("text", "") for b in r.json()["content"] if b.get("type") == "text")
                    ok, detail = bool(text.strip()), repr(text[:30])
                elif ok and test == "anthropic stream":
                    events = sse(r.text)
                    text = "".join((e.get("delta") or {}).get("text") or "" for e in events if e.get("type") == "content_block_delta")
                    ok = bool(text.strip()) and any(e.get("type") == "message_stop" for e in events)
                    detail = f"{text[:30]!r} {len(events)} events"
                if not ok and r.status_code != 200:
                    detail += f" {r.text[:160]}"
                cost = r.headers.get("x-meritorder-cost-usd")
                detail += f" ({time.time() - t0:.1f}s{', $' + cost if cost else ''})"
            except Exception as e:
                ok, detail = False, f"{type(e).__name__}: {e}"
            failures += 0 if ok else 1
            print(f"{backend:16} {model:22} {test:18} {'PASS' if ok else 'FAIL'}  {detail}")
    return 1 if failures else 0


if __name__ == "__main__":
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--gateway", default=os.environ.get("MERITORDER_URL", "http://localhost:8080"))
    p.add_argument("--key", default=os.environ.get("MERITORDER_DEV_KEY"))
    p.add_argument("--backend", action="append")
    a = p.parse_args()
    sys.exit(run(a.gateway.rstrip("/"), a.key, a.backend))
