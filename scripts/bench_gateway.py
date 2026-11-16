"""Measure what Merit Order adds per call, against zero-latency mock backends.

Start a gateway whose config uses mock backends with `mock_latency_ms: 0` (and Postgres + Redis for
realistic numbers), then:

    BENCH_KEY=<a key> MERITORDER_BENCH_GATEWAY=http://127.0.0.1:8080 MERITORDER_BENCH_REDIS_PORT=6379 \
      uv run --with redis python scripts/bench_gateway.py

Subtract "HTTP floor" and "mock backend itself" from each row to get Merit Order's own time.
For throughput, use scripts/bench_load.py (one gateway per feature, its own load generator), or `ab` or `oha` against
/v1/chat/completions.
"""

import asyncio
import os
import statistics
import time
import uuid

import httpx
import redis

GW = os.environ.get("MERITORDER_BENCH_GATEWAY", "http://127.0.0.1:8080")
KEY = os.environ["BENCH_KEY"]
H = {"authorization": f"Bearer {KEY}", "content-type": "application/json"}
r = redis.Redis(port=int(os.environ.get("MERITORDER_BENCH_REDIS_PORT", "6379")))


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))]


def show(name, xs):
    print(f"{name:<52} p50 {statistics.median(xs):6.2f} ms   p95 {pct(xs, .95):6.2f}   p99 {pct(xs, .99):6.2f}   (n={len(xs)})")


def body(model, tokens, stream=False, tools=False):
    # ~4 characters per token: a coding-agent style history with big tool results
    msgs = [{"role": "system", "content": "You are a coding agent. " * 20}]
    chunk = "def handler(event):\n    return process(event['payload'])  # line\n" * 20
    while sum(len(m["content"]) for m in msgs) < tokens * 4:
        msgs.append({"role": "user", "content": "Fix the failing test in handler.py"})
        msgs.append({"role": "assistant", "content": "Reading the file."})
        msgs.append({"role": "user", "content": chunk})
    msgs.append({"role": "user", "content": "now refactor the retry logic"})
    b = {"model": model, "messages": msgs, "max_tokens": 200, "stream": stream}
    if tools:
        b["tools"] = [{"type": "function", "function": {"name": f"tool{i}", "parameters": {"type": "object"}}} for i in range(8)]
    return b


def redis_cmds():
    return int(r.info("stats")["total_commands_processed"])


async def seq(c, name, b, n, stream=False, session=True, path="/v1/chat/completions"):
    sid = uuid.uuid4().hex
    xs = []
    cmds0 = redis_cmds()
    for _ in range(n):
        h = {**H, "x-session-id": sid if session else uuid.uuid4().hex}
        t0 = time.perf_counter()
        if stream:
            async with c.stream("POST", f"{GW}{path}", json=b, headers=h) as resp:
                it = resp.aiter_bytes()
                await it.__anext__()
                xs.append((time.perf_counter() - t0) * 1000)  # time to first byte
                async for _ in it:
                    pass
                assert resp.status_code == 200, resp.status_code
        else:
            resp = await c.post(f"{GW}{path}", json=b, headers=h)
            xs.append((time.perf_counter() - t0) * 1000)
            assert resp.status_code == 200, resp.text[:200]
    show(name, xs)
    return (redis_cmds() - cmds0) / n


async def floor(c, n=300):
    xs = []
    for _ in range(n):
        t0 = time.perf_counter()
        await c.get(f"{GW}/healthz")
        xs.append((time.perf_counter() - t0) * 1000)
    show("HTTP floor: GET /healthz (client + uvicorn)", xs)


async def mock_only(b, n=50):
    """The mock backend's own cost for this body, run in-process (to subtract)."""
    from meritorder.backends.mock import MockBackend
    from meritorder.config import BackendConfig

    m = MockBackend(BackendConfig(name="m", kind="mock", format="openai", mock_latency_ms=0))
    xs = []
    for _ in range(n):
        t0 = time.perf_counter()
        await m.send({**b, "model": "small-1"}, False, {})
        xs.append((time.perf_counter() - t0) * 1000)
    return statistics.median(xs)


async def main():
    async with httpx.AsyncClient(timeout=60, limits=httpx.Limits(max_connections=100)) as c:
        for _ in range(30):  # warm up
            await c.post(f"{GW}/v1/chat/completions", json=body("small", 200), headers=H)
        await floor(c)
        for label, b, n in [("small model, 200-token prompt", body("small", 200), 300),
                            ("auto, 2k-token prompt", body("auto", 2000), 300),
                            ("auto, agent: 30k tokens + 8 tools", body("auto", 30000, tools=True), 100),
                            ("auto, agent: 120k tokens + 8 tools", body("auto", 120000, tools=True), 50)]:
            cmds = await seq(c, label, b, n)
            m = await mock_only(b)
            print(f"{'':<52} (mock backend itself: {m:.2f} ms; Redis commands/request: {cmds:.0f})")
        await seq(c, "streaming TTFB, auto, 2k-token prompt", body("auto", 2000, stream=True), 200, stream=True)
        await seq(c, "streaming TTFB, auto, 120k tokens + tools", body("auto", 120000, stream=True, tools=True), 50, stream=True)
        await seq(c, "anthropic format client → openai backend (translate)",
                  {"model": "small", "max_tokens": 200, "messages": [{"role": "user", "content": "hi " * 500}]}, 200, path="/v1/messages")


if __name__ == "__main__":
    asyncio.run(main())
