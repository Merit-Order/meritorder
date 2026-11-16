"""Throughput with each request-path feature on, against zero-latency mock backends (docs/07-deployment.md).

    uv run python scripts/bench_load.py all [--seconds 20] [--processes 1] [--out results.json]
    uv run python scripts/bench_load.py serve --feature compression-model --port 18100   # one gateway, to load by hand
    uv run python scripts/bench_load.py load --url http://127.0.0.1:18100 --body agent --seconds 20

``all`` starts a gateway per feature (or ``--processes`` of them, on ports next to each other, loaded together),
warms it up, loads it from as many client processes for ``--seconds``, and reports requests per second, latency and the
gateway's CPU time per request. Features:

* baseline: nothing opt-in on (the configuration bench_gateway.py uses: models small and big on two mocks at 0 ms)
* calibrated: token counts calibrated for both models (ratios learned from 50 replies each)
* levels: ``routing.levels`` with three times the default word lists
* cost-per-success: ``routing.model_choice: cost_per_success``
* learned-shadow, learned-route: the learned router with a model trained on synthetic traffic (tests/learned_synth.py)
* exploration: the learned router in shadow mode and the default team exploring (decision 59). With ``agent-grow``
  bodies (a task that grows an edit and a test run each request) coding outcomes read every new turn (outcomes.py)
* expected-cost: ``routing.model_choice: expected_cost`` with the cascade, the learned router in shadow mode and the
  success predictor's fitted layer in shadow mode (decision 62), scores with chances (``quality.source: config``)
* compression-rules: the rules on the default team
* compression-model: the rules and model-based compression, against a stub compressor that answers at once (so this
  measures the gateway, not the model); ``agent-new`` bodies bring a new tool result every request
* affinity, no-affinity: a GPU deployment (a mock, two replicas) serving a named model, with replica affinity on or off

The gateway uses in-memory state and storage unless MERITORDER_REDIS_URL and MERITORDER_DATABASE_URL are set (then
every process shares them, as in production). Nothing here calls a provider.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import multiprocessing as mp
import os
import re
import subprocess
import sys
import time

from fastapi import Request  # at module level: the stub's handler annotation is resolved here

ROOT =os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEY = "mo-bench-key"
PRICE_BIG = {"input": 3, "output": 15, "cache_read": 0.3, "cache_write": 3.75}
PRICE_SMALL = {"input": 0.2, "output": 1.0, "cache_read": 0.02}
CONFIG = {
    "baseline_model": "big",
    "backends": {
        "mock-anthropic": {"kind": "mock", "format": "anthropic", "mock_latency_ms": 0},
        "mock-openai": {"kind": "mock", "format": "openai", "mock_latency_ms": 0},
    },
    "models": {
        "big": {"quality": {"coding": 5, "general": 5}, "routes": [{"backend": "mock-anthropic", "model": "big-1", "price": PRICE_BIG}]},
        "small": {"quality": {"coding": 3, "general": 3}, "routes": [{"backend": "mock-openai", "model": "small-1", "price": PRICE_SMALL}]},
    },
    "teams": {"default": {}},
    "auth": {"static_keys": [{"key": KEY, "user": "bench", "team": "default", "role": "user"}]},
}
FEATURES = ["baseline", "calibrated", "levels", "cost-per-success", "learned-shadow", "learned-route", "exploration", "compression-rules",
            "compression-model", "no-affinity", "affinity", "expected-cost"]
# What each feature is measured with: compression only acts on tool results, affinity only on a GPU deployment.
DEFAULT_BODIES = {"baseline": ["chat", "agent"], "compression-rules": ["agent"], "compression-model": ["agent", "agent-new"],
                  "affinity": ["gpu"], "no-affinity": ["gpu"], "learned-shadow": ["chat", "agent-grow"], "exploration": ["chat", "agent-grow"],
                  "expected-cost": ["chat", "agent-grow"]}


# ------------------------------------------------------------------ request bodies


def _read(path: str) -> str:
    return open(os.path.join(ROOT, path), encoding="utf-8").read()


def chat_body() -> dict:
    """bench_gateway.py's "auto, 200-token prompt": a coding agent's short turn."""
    chunk = "def handler(event):\n    return process(event['payload'])  # line\n" * 20
    msgs = [{"role": "system", "content": "You are a coding agent. " * 20}, {"role": "user", "content": "Fix the failing test"},
            {"role": "assistant", "content": "Reading the file."}, {"role": "user", "content": chunk[:300]},
            {"role": "user", "content": "now refactor the retry logic"}]
    return {"model": "auto", "messages": msgs, "max_tokens": 200}


def agent_body(marker: str = "") -> dict:
    """An agent's turn with 12 tool results (about 100k characters, 28k tokens) and 8 tools: docs pages (prose), source
    files, and a test log with terminal colours and repeated lines (what the rules shorten). ``marker`` goes into the
    last result, so each request can bring a new one."""
    docs = _read("docs/09-decisions.md") + _read("docs/00-how-it-all-works.md")
    code = _read("src/meritorder/router.py")
    log = "".join(f"\x1b[32mPASSED\x1b[0m tests/test_router.py::test_case_{i % 7}  \n" for i in range(160))
    results = []
    for i in range(12):
        if i % 3 == 0:
            results.append(docs[i * 9000:i * 9000 + 9000])
        elif i % 3 == 1:
            results.append(code[i * 6000:i * 6000 + 9000])
        else:
            results.append(log)
    results[-1] = docs[60000:69000].replace("Merit Order", f"Merit Order {marker}", 1) if marker else docs[60000:69000]
    msgs = [{"role": "system", "content": "You are a coding agent. Use the tools. " * 10},
            {"role": "user", "content": "Read the routing docs and the router, run the tests, and explain the switch margin."}]
    for i, text in enumerate(results):
        call = {"id": f"call_{i}", "type": "function", "function": {"name": ["web_fetch", "bash", "bash"][i % 3], "arguments": "{}"}}
        msgs += [{"role": "assistant", "content": None, "tool_calls": [call]}, {"role": "tool", "tool_call_id": f"call_{i}", "content": text}]
    tools = [{"type": "function", "function": {"name": n, "parameters": {"type": "object"}}}
             for n in ("bash", "web_fetch", "read", "edit", "write", "grep", "glob", "todo")]
    return {"model": "auto", "messages": msgs, "tools": tools, "max_tokens": 1000}


PYTEST_LOG = "".join(f"tests/test_dates.py::test_case_{i} PASSED\n" for i in range(40)) + "===== 1 failed, 39 passed in 0.31s ====="


def grow_bodies(turns: int = 20) -> list[bytes]:
    """An agent's task growing turn by turn: the agent body, then ``j`` turns of an edit and a test run (pytest's output,
    a failure at the end) for request j of a session. What coding outcomes read (outcomes.py): a new turn's tool results."""
    base = agent_body()
    out = []
    for j in range(turns):
        msgs = list(base["messages"])
        for k in range(j):
            calls = [{"id": f"e{k}", "type": "function", "function": {"name": "edit", "arguments": json.dumps({"path": "app/dates.py"})}},
                     {"id": f"t{k}", "type": "function", "function": {"name": "bash", "arguments": json.dumps({"command": "pytest -q"})}}]
            msgs += [{"role": "assistant", "content": None, "tool_calls": calls}, {"role": "tool", "tool_call_id": f"e{k}", "content": "ok"},
                     {"role": "tool", "tool_call_id": f"t{k}", "content": PYTEST_LOG}]
        out.append(json.dumps({**base, "messages": msgs}).encode())
    return out


def gpu_body() -> dict:
    return {**chat_body(), "model": "gpu-only"}


def request_bytes(host: str, path: str, body: dict, session: str) -> bytes:
    data = json.dumps(body).encode()
    head = (f"POST {path} HTTP/1.1\r\nhost: {host}\r\nauthorization: Bearer {KEY}\r\ncontent-type: application/json\r\n"
            f"x-session-id: {session}\r\ncontent-length: {len(data)}\r\n\r\n")
    return head.encode() + data


# ------------------------------------------------------------------ the gateway, with one feature on


def config_for(feature: str, stub_url: str = "") -> dict:
    import copy

    raw = copy.deepcopy(CONFIG)
    team = raw["teams"]["default"]
    routing = raw.setdefault("routing", {})
    if feature == "levels":
        from meritorder.config import ANALYSIS_WORDS, CODING_WORDS, HARD_WORDS

        extra = [f"{w}s?x{i}" for i in range(2) for w in ("deploy", "migrate", "refactor", "benchmark")]
        routing["levels"] = {"coding_words": CODING_WORDS * 3 + extra, "hard_words": HARD_WORDS * 3, "analysis_words": ANALYSIS_WORDS * 3}
    elif feature == "cost-per-success":
        routing["model_choice"] = "cost_per_success"
    elif feature == "exploration":  # the learned router in shadow mode (it records outcomes), and the team exploring
        routing["learned"] = {"mode": "shadow", "max_age_days": 0}
        team["exploration"] = {"enabled": True}
    elif feature in ("learned-shadow", "learned-route"):
        routing["learned"] = {"mode": feature.split("-")[1], "max_age_days": 0}
    elif feature == "expected-cost":
        routing.update(learned={"mode": "shadow", "max_age_days": 0}, model_choice="expected_cost", success={"mode": "shadow"})
        raw["quality"] = {"source": "config", "measured_best_rate": {"coding": 0.7, "general": 0.77}}
    elif feature == "compression-rules":
        team["compression"] = {"enabled": True}
    elif feature == "compression-model":
        raw["compressor"] = {"url": stub_url, "model": "stub/llmlingua-2-xlm-roberta-large-stub"}
        team["compression"] = {"enabled": True, "model": {"enabled": True}}
    elif feature in ("affinity", "no-affinity"):
        raw["backends"]["gpu"] = {"kind": "self_hosted", "format": "openai", "mock_latency_ms": 0, "max_concurrency": 1000,
                                  "serve": {"model": "gpu-only", "provisioner": "static", "replica_affinity": feature == "affinity",
                                            "static_urls": ["http://r1.test/v1", "http://r2.test/v1"]}}
        raw["models"]["gpu-only"] = {"quality": {"coding": 5, "general": 5},
                                     "routes": [{"backend": "gpu", "model": "m", "price": {"input": 1, "output": 2, "cache_read": 0.1}}]}
    return raw


async def setup(gw, feature: str) -> None:
    """What a feature needs in the gateway's state before traffic: learned ratios, a trained model, live replicas."""
    if feature == "calibrated":
        from meritorder.tokens import Estimate

        for model in ("small", "big"):
            for i in range(50):
                gw.tokens.record(model, Estimate(text=4000 + 100 * i), int((4000 + 100 * i) * 1.12))
    elif feature.startswith("learned-") or feature in ("exploration", "expected-cost"):
        sys.path.insert(0, os.path.join(ROOT, "tests"))
        from learned_synth import KEY as FEATURE_KEY
        from test_learned_router import trained

        os.environ["MERITORDER_FEATURE_KEY"] = FEATURE_KEY
        version = await gw.store.save_router_model(trained()["doc"], {}, "bench", True)
        await gw.load_learned(version)
        if feature == "expected-cost":  # a fitted success layer trained on synthetic task starts (tests/test_expected_cost.py)
            from test_expected_cost import synthetic

            from meritorder import success

            await gw.store.set_setting(success.SETTING, success.train(synthetic(), gw.learned.fv)["doc"])
            await gw.load_success(1)
    elif feature in ("affinity", "no-affinity"):
        from meritorder.backends.mock import MockBackend

        gpu = next(b for b in gw.cfg.backends if b.name == "gpu")
        gw.backends["gpu"] = MockBackend(gpu)
        replicas = [{"id": str(i), "status": "READY", "url": f"http://r{i}.test/v1", "endpoint": f"http://r{i}.test/v1", "utilization": 0.1}
                    for i in (1, 2)]
        await gw.state.set_json("deploy:gpu", {"state": "warm", "endpoint": "http://lb.test/v1", "replicas": 2, "replicas_info": replicas})
        await gw.refresh_deployments()


def serve(feature: str, port: int, stub_url: str, profile: str | None) -> None:
    import cProfile

    import uvicorn

    from meritorder.app import build_gateway, create_app
    from meritorder.config import build_config

    cfg = build_config(config_for(feature, stub_url), ROOT, prices_data={}, quiet=True)
    gw = build_gateway(cfg)
    app = create_app(gateway=gw)
    started = app.router.lifespan_context

    @contextlib.asynccontextmanager
    async def lifespan(a):
        async with started(a) as value:  # the gateway's own start (store, state, config), then the feature's
            await setup(gw, feature)
            yield value

    app.router.lifespan_context = lifespan
    prof = cProfile.Profile() if profile else None
    if prof:
        prof.enable()
    try:
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
    finally:
        if prof:
            prof.disable()
            prof.dump_stats(profile)


# ------------------------------------------------------------------ a stub compressor: answers at once


def stub(port: int) -> None:
    """POST /v1/compress: drops every fifth word of five letters or more in each text, with no model."""
    import uvicorn
    from fastapi import FastAPI

    app = FastAPI()
    word = re.compile(r"\b[a-z]{5,}\b")

    @app.get("/healthz")
    async def healthz():
        return {"ok": True, "model": "stub/llmlingua-2-xlm-roberta-large-stub", "version": "stub", "device": "none"}

    @app.post("/v1/compress")
    async def compress(request: Request):
        body = await request.json()
        out = [{"drop": [[m.start(), m.end()] for i, m in enumerate(word.finditer(t)) if i % 5 == 0]} for t in body["texts"]]
        return {"model": body.get("model"), "version": "stub", "results": out}

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


# ------------------------------------------------------------------ the load generator


async def _client(host: str, port: int, body: str, n_conns: int, seconds: float, offset: int, path: str) -> dict:
    """``n_conns`` keep-alive connections sending one request after another for ``seconds``."""
    fixed = {"chat": chat_body, "agent": agent_body, "gpu": gpu_body}
    if body == "agent-grow":
        variants = grow_bodies()
        turn: dict[str, int] = {}

        def make(i: int, session: str) -> bytes:
            j = turn.get(session, 0)
            turn[session] = j + 1
            data = variants[j % len(variants)]
            sid = f"{session}-{j // len(variants)}"  # a new session once the task has grown its 20 turns
            return (f"POST {path} HTTP/1.1\r\nhost: {host}\r\nauthorization: Bearer {KEY}\r\ncontent-type: application/json\r\n"
                    f"x-session-id: {sid}\r\ncontent-length: {len(data)}\r\n\r\n").encode() + data
    elif body == "agent-new":
        template = json.dumps(agent_body(marker="\x00")).encode()
        head_part, tail_part = template.split(b"\\u0000")

        def make(i: int, session: str) -> bytes:
            data = head_part + f"run {offset}-{i}".encode() + tail_part
            return (f"POST {path} HTTP/1.1\r\nhost: {host}\r\nauthorization: Bearer {KEY}\r\ncontent-type: application/json\r\n"
                    f"x-session-id: {session}\r\ncontent-length: {len(data)}\r\n\r\n").encode() + data
    else:
        payload = fixed[body]()
        cache: dict[str, bytes] = {}

        def make(i: int, session: str) -> bytes:
            if session not in cache:
                cache[session] = request_bytes(host, path, payload, session)
            return cache[session]

    lat: list[float] = []
    errors = 0
    count = [0]
    stop_at = time.perf_counter() + seconds

    async def conn(c: int) -> None:
        nonlocal errors
        reader, writer = await asyncio.open_connection(host, port)
        session = f"bench-{offset}-{c}"
        try:
            while time.perf_counter() < stop_at:
                data = make(count[0], session)
                count[0] += 1
                t0 = time.perf_counter()
                writer.write(data)
                head = await reader.readuntil(b"\r\n\r\n")
                length = int(re.search(rb"(?i)content-length: *(\d+)", head).group(1))
                await reader.readexactly(length)
                lat.append(time.perf_counter() - t0)
                if not head.startswith(b"HTTP/1.1 200"):
                    errors += 1
        finally:
            writer.close()

    await asyncio.gather(*(conn(c) for c in range(n_conns)))
    return {"n": len(lat), "errors": errors, "lat": lat[:: max(1, len(lat) // 20000)]}


def _client_proc(args: tuple) -> dict:
    return asyncio.run(_client(*args))


def load(urls: list[str], body: str, seconds: float, connections: int, procs: int, path: str = "/v1/chat/completions") -> dict:
    """Load every URL at once, ``connections`` per URL, from ``procs`` client processes per URL."""
    jobs = []
    for u, url in enumerate(urls):
        host, port = url.split("//")[1].split(":")
        for p in range(procs):
            jobs.append((host, int(port), body, max(1, connections // procs), seconds, u * procs + p, path))
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        t0 = time.perf_counter()
        parts = pool.map(_client_proc, jobs)
        wall = time.perf_counter() - t0
    lat = sorted(x for p in parts for x in p["lat"])
    n = sum(p["n"] for p in parts)
    pct = lambda q: lat[min(len(lat) - 1, int(q * len(lat)))] * 1000 if lat else 0  # noqa: E731
    return {"requests": n, "errors": sum(p["errors"] for p in parts), "rps": n / seconds, "wall": wall,
            "p50_ms": pct(0.5), "p95_ms": pct(0.95), "p99_ms": pct(0.99)}


# ------------------------------------------------------------------ every feature in turn


def cpu_seconds(pid: int) -> float:
    """User + system CPU time of a process so far (ps prints [[dd-]hh:]mm:ss.cc)."""
    out = subprocess.run(["ps", "-o", "time=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    secs = 0.0
    for part in out.replace("-", ":").split(":"):
        secs = secs * 60 + float(part)
    return secs


def wait_up(url: str, timeout: float = 60) -> None:
    import httpx

    end = time.time() + timeout
    while time.time() < end:
        try:
            if httpx.get(f"{url}/healthz", timeout=1).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.2)
    raise SystemExit(f"{url} didn't come up")


def run_all(args) -> None:
    results = {}
    stub_proc = None
    stub_url = f"http://127.0.0.1:{args.port - 1}"
    features = args.features.split(",") if args.features else FEATURES
    try:
        if any(f == "compression-model" for f in features):
            stub_proc = subprocess.Popen([sys.executable, __file__, "stub", "--port", str(args.port - 1)])
            wait_up(stub_url)
        for feature in features:
            bodies = args.bodies.split(",") if args.bodies else DEFAULT_BODIES.get(feature, ["chat"])
            for body in bodies:
                ports = [args.port + i for i in range(args.processes)]
                procs = [subprocess.Popen([sys.executable, __file__, "serve", "--feature", feature, "--port", str(p), "--stub-url", stub_url]
                                          + (["--profile", f"{args.profile}-{feature}-{body}-{p}.prof"] if args.profile else []))
                         for p in ports]
                urls = [f"http://127.0.0.1:{p}" for p in ports]
                try:
                    for u in urls:
                        wait_up(u)
                    load(urls, body, args.warmup, args.connections, args.client_procs)
                    cpu0 = [cpu_seconds(p.pid) for p in procs]
                    res = load(urls, body, args.seconds, args.connections, args.client_procs)
                    cpu = sum(cpu_seconds(p.pid) - c for p, c in zip(procs, cpu0))
                    res["cpu_ms_per_request"] = 1000 * cpu / max(1, res["requests"])
                finally:
                    for p in procs:
                        p.send_signal(2)  # Ctrl-C: uvicorn shuts down, and --profile writes its file
                    for p in procs:
                        p.wait(30)
                results[f"{feature} / {body}"] = res
                print(f"{feature:<18} {body:<10} {res['rps']:>8,.0f} req/s  p50 {res['p50_ms']:6.1f} ms  p95 {res['p95_ms']:6.1f}  "
                      f"p99 {res['p99_ms']:6.1f}  CPU {res['cpu_ms_per_request']:.2f} ms/request  errors {res['errors']}", flush=True)
    finally:
        if stub_proc:
            stub_proc.terminate()
    if args.out:
        json.dump(results, open(args.out, "w"), indent=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("all")
    a.add_argument("--features", help=f"comma-separated, default all: {','.join(FEATURES)}")
    a.add_argument("--bodies", help="chat, agent, agent-new or gpu (default: the feature's own)")
    a.add_argument("--processes", type=int, default=1, help="gateway processes, each loaded at once")
    a.add_argument("--client-procs", type=int, default=1, help="load generator processes per gateway")
    a.add_argument("--connections", type=int, default=64, help="per gateway")
    a.add_argument("--seconds", type=float, default=20)
    a.add_argument("--warmup", type=float, default=3)
    a.add_argument("--port", type=int, default=18100)
    a.add_argument("--profile", help="write a cProfile file per gateway, with this prefix")
    a.add_argument("--out")
    s = sub.add_parser("serve")
    s.add_argument("--feature", default="baseline", choices=FEATURES)
    s.add_argument("--port", type=int, default=18100)
    s.add_argument("--stub-url", default="http://127.0.0.1:18099")
    s.add_argument("--profile")
    st = sub.add_parser("stub")
    st.add_argument("--port", type=int, default=18099)
    ld = sub.add_parser("load")
    ld.add_argument("--url", action="append", required=True)
    ld.add_argument("--body", default="chat")
    ld.add_argument("--seconds", type=float, default=20)
    ld.add_argument("--connections", type=int, default=64)
    ld.add_argument("--client-procs", type=int, default=1)
    args = ap.parse_args()
    if args.cmd == "serve":
        serve(args.feature, args.port, args.stub_url, args.profile)
    elif args.cmd == "stub":
        stub(args.port)
    elif args.cmd == "load":
        print(json.dumps(load(args.url, args.body, args.seconds, args.connections, args.client_procs), indent=1))
    else:
        run_all(args)


if __name__ == "__main__":
    main()
