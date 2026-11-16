"""A stand-in for a vLLM server, for exercising the self-hosted path without GPUs. Standard library only, so it also
runs as a tiny container image (scripts/kind/Dockerfile.fake-vllm) where the Kubernetes provisioner expects vLLM.

It takes vLLM's own command line (``vllm serve <model> --host --port --served-model-name ...``; flags it doesn't know
are ignored, as they only tune a real engine) and serves what Merit Order and llm-d's endpoint picker use:

- ``/health`` (503 while "loading", see ``--fake-load-seconds``), ``/v1/models``;
- ``/v1/chat/completions`` and ``/v1/completions``, streaming or not, with usage and cached prompt tokens (a prompt
  whose leading messages were seen before counts them as prefix-cache hits, as vLLM's prefix caching would);
- ``/metrics`` with vLLM's metric names: requests running and waiting, KV-cache usage, prompt, generation and
  prefix-cache token counters, the cache config (blocks and block size) and the process start time.

    uv run python scripts/fake_vllm.py --port 8095 --model Qwen/Qwen3-Coder-Next
    curl -X POST 'localhost:8095/admin/utilization?value=0.95&waiting=3'   # pretend to be busy
    curl localhost:8095/admin/state

``--delay-ms`` holds each request that long before answering, to exercise saturation and queueing. ``--fake-exit N``
exits with code N at start (a crash loop on Kubernetes). On SIGTERM it stops taking requests and finishes the ones in
flight, as vLLM does, so a scale-down's grace period can be checked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import signal
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

BLOCK_SIZE = 16
GPU_BLOCKS = 8192  # what the metrics say the KV cache holds: 131,072 tokens


def tokens(text: str) -> int:
    """About four characters a token, like most tokenizers on English and code."""
    return max(1, (len(text) + 3) // 4) if text else 0


def text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


class Engine:
    """What the fake engine counts: in-flight requests, tokens, and the prompt prefixes its cache holds."""

    def __init__(self, model: str, delay_ms: int = 0, load_seconds: float = 0.0, served: str | None = None):
        self.model, self.served = model, served or model
        self.delay_ms = delay_ms
        self.ready_at = time.time() + load_seconds
        self.started = time.time()
        self.lock = threading.Lock()
        self.running = 0
        self.util: float | None = None  # set by /admin/utilization; else from the cache's fill
        self.waiting = 0
        self.requests = 0
        self.priorities: list[int] = []
        self.prompt_tokens = self.generation_tokens = self.cache_queries = self.cache_hits = 0
        self.prefixes: dict[str, int] = {}  # hash of the first n messages -> their tokens

    def ready(self) -> bool:
        return time.time() >= self.ready_at

    def prompt(self, messages: list[dict]) -> tuple[int, int]:
        """The prompt's tokens and how many of them the cache already held (its longest known prefix of messages)."""
        h = hashlib.sha256()
        total = cached = 0
        seen = []
        for m in messages:
            h.update(json.dumps([m.get("role"), text_of(m.get("content"))]).encode())
            total += tokens(text_of(m.get("content"))) + 4
            seen.append((h.hexdigest(), total))
        with self.lock:
            for key, n in seen[:-1]:  # the last message is new by definition
                if key in self.prefixes:
                    cached = n
            for key, n in seen:
                self.prefixes[key] = n
            while len(self.prefixes) > GPU_BLOCKS:  # evict the oldest, like a full cache
                self.prefixes.pop(next(iter(self.prefixes)))
            self.prompt_tokens += total
            self.cache_queries += total
            self.cache_hits += cached
        return total, cached

    def kv_usage(self) -> float:
        if self.util is not None:
            return self.util
        held = sum(self.prefixes.values()) if self.prefixes else 0
        return min(1.0, held / (BLOCK_SIZE * GPU_BLOCKS))

    def metrics(self) -> str:
        label = f'engine="0",model_name="{self.served}"'
        with self.lock:
            lines = [
                "# HELP vllm:num_requests_running Number of requests in model execution batches.",
                "# TYPE vllm:num_requests_running gauge",
                f"vllm:num_requests_running{{{label}}} {float(self.running)}",
                "# TYPE vllm:num_requests_waiting gauge",
                f"vllm:num_requests_waiting{{{label}}} {float(self.waiting)}",
                "# TYPE vllm:kv_cache_usage_perc gauge",
                f"vllm:kv_cache_usage_perc{{{label}}} {self.kv_usage()}",
                "# TYPE vllm:prompt_tokens_total counter",
                f"vllm:prompt_tokens_total{{{label}}} {float(self.prompt_tokens)}",
                "# TYPE vllm:generation_tokens_total counter",
                f"vllm:generation_tokens_total{{{label}}} {float(self.generation_tokens)}",
                "# TYPE vllm:prefix_cache_queries_total counter",
                f"vllm:prefix_cache_queries_total{{{label}}} {float(self.cache_queries)}",
                "# TYPE vllm:prefix_cache_hits_total counter",
                f"vllm:prefix_cache_hits_total{{{label}}} {float(self.cache_hits)}",
                "# TYPE vllm:request_success_total counter",
                f'vllm:request_success_total{{{label},finished_reason="stop"}} {float(self.requests)}',
                "# TYPE vllm:cache_config_info gauge",
                f'vllm:cache_config_info{{block_size="{BLOCK_SIZE}",cache_dtype="auto",enable_prefix_caching="True",'
                f'engine="0",gpu_memory_utilization="0.9",num_gpu_blocks="{GPU_BLOCKS}"}} 1.0',
                "# TYPE process_start_time_seconds gauge",
                f"process_start_time_seconds {self.started}",
            ]
        return "\n".join(lines) + "\n"


def handler(engine: Engine, served: str):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # quiet: one line per request is noise in a pod's log
            pass

        def _send(self, status: int, body, content_type: str = "application/json") -> None:
            data = body.encode() if isinstance(body, str) else json.dumps(body).encode()
            self.send_response(status)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/health":
                return self._send(200, "") if engine.ready() else self._send(503, {"error": "loading the model"})
            if url.path == "/metrics":
                return self._send(200, engine.metrics(), "text/plain; version=0.0.4")
            if url.path == "/admin/state":
                return self._send(200, {"util": engine.util, "waiting": engine.waiting, "running": engine.running,
                                        "requests": engine.requests, "priorities": engine.priorities[-100:]})
            if not engine.ready():
                return self._send(503, {"error": "loading the model"})
            if url.path == "/v1/models":
                return self._send(200, {"object": "list", "data": [{"id": served, "object": "model", "created": int(engine.started),
                                                                    "owned_by": "vllm", "root": engine.model, "max_model_len": 131072}]})
            return self._send(404, {"error": f"no route {url.path}"})

        def do_POST(self):
            url = urlparse(self.path)
            length = int(self.headers.get("content-length") or 0)
            raw = self.rfile.read(length) if length else b""
            if url.path == "/admin/utilization":
                q = parse_qs(url.query)
                engine.util = float(q.get("value", ["0.1"])[0])
                engine.waiting = int(q.get("waiting", ["0"])[0])
                return self._send(200, {"util": engine.util, "waiting": engine.waiting})
            if url.path not in ("/v1/chat/completions", "/v1/completions"):
                return self._send(404, {"error": f"no route {url.path}"})
            if not engine.ready():
                return self._send(503, {"error": "loading the model"})
            try:
                body = json.loads(raw or b"{}")
            except ValueError:
                return self._send(400, {"error": {"message": "body is not JSON"}})
            if body.get("model") not in (served, engine.model):
                return self._send(404, {"error": {"message": f"The model `{body.get('model')}` does not exist.", "type": "NotFoundError"}})
            chat = url.path.endswith("chat/completions")
            messages = (body.get("messages") or []) if chat else [{"role": "user", "content": str(body.get("prompt") or "")}]
            with engine.lock:
                engine.running += 1
                engine.requests += 1
                engine.priorities.append(int(body.get("priority") or 0))
            try:
                if engine.delay_ms:
                    time.sleep(engine.delay_ms / 1000)
                prompt, cached = engine.prompt(messages)
                last = text_of((messages[-1] if messages else {}).get("content"))
                reply = f"[{served}] You said: {last[:200]}"
                out = tokens(reply)
                with engine.lock:
                    engine.generation_tokens += out
                usage = {"prompt_tokens": prompt, "completion_tokens": out, "total_tokens": prompt + out,
                         "prompt_tokens_details": {"cached_tokens": cached}}
                rid = f"{'chatcmpl' if chat else 'cmpl'}-{uuid.uuid4().hex}"
                if body.get("stream"):
                    return self._stream(rid, chat, reply, usage, bool((body.get("stream_options") or {}).get("include_usage")))
                choice = ({"index": 0, "message": {"role": "assistant", "content": reply}, "finish_reason": "stop"} if chat
                          else {"index": 0, "text": reply, "finish_reason": "stop"})
                return self._send(200, {"id": rid, "object": "chat.completion" if chat else "text_completion", "created": int(time.time()),
                                        "model": served, "choices": [choice], "usage": usage})
            finally:
                with engine.lock:
                    engine.running -= 1

        def _stream(self, rid: str, chat: bool, reply: str, usage: dict, include_usage: bool) -> None:
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("cache-control", "no-cache")
            self.send_header("connection", "close")
            self.end_headers()
            self.close_connection = True
            obj = "chat.completion.chunk" if chat else "text_completion"

            def event(choices, extra=None) -> None:
                chunk = {"id": rid, "object": obj, "created": int(time.time()), "model": served, "choices": choices, **(extra or {})}
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()

            words = reply.split(" ")
            for i, w in enumerate(words):
                piece = w if i == 0 else " " + w
                delta = {"role": "assistant", "content": piece} if chat and i == 0 else {"content": piece}
                event([{"index": 0, "delta": delta, "finish_reason": None}] if chat else [{"index": 0, "text": piece, "finish_reason": None}])
            event([{"index": 0, "delta": {}, "finish_reason": "stop"}] if chat else [{"index": 0, "text": "", "finish_reason": "stop"}])
            if include_usage:
                event([], {"usage": usage})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    return Handler


def parse(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="a stand-in for `vllm serve`")
    p.add_argument("command", nargs="?", help="serve (as in `vllm serve`), optional")
    p.add_argument("model_tag", nargs="?", help="the model, as `vllm serve <model>` takes it")
    p.add_argument("--model", default=None)
    p.add_argument("--served-model-name", nargs="+", default=None)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8095)
    p.add_argument("--delay-ms", type=int, default=0)
    p.add_argument("--fake-load-seconds", type=float, default=0.0, help="answer /health with 503 this long, like a model loading")
    p.add_argument("--fake-exit", type=int, default=None, help="exit with this code at start (a crash loop on Kubernetes)")
    args, rest = p.parse_known_args(argv)  # the rest are vLLM engine flags: accepted and ignored
    if args.command and args.command != "serve" and not args.model_tag:  # `fake_vllm.py MODEL`
        args.command, args.model_tag = "serve", args.command
    args.ignored = rest
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse(sys.argv[1:] if argv is None else argv)
    model = args.model_tag or args.model or "Qwen/Qwen3-Coder-Next"
    served = (args.served_model_name or [model])[0]
    if args.fake_exit is not None:
        print(f"fake vLLM: exiting with code {args.fake_exit} as asked (--fake-exit)", file=sys.stderr, flush=True)
        sys.exit(args.fake_exit)
    engine = Engine(model, args.delay_ms, args.fake_load_seconds, served)
    server = ThreadingHTTPServer((args.host, args.port), handler(engine, served))
    server.daemon_threads = True

    def stop(signum, _frame) -> None:
        # Like vLLM: stop taking requests, finish the ones in flight, then exit.
        print(f"fake vLLM: signal {signum}, finishing {engine.running} request(s) in flight", file=sys.stderr, flush=True)
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    print(f"fake vLLM serving {model} as {served} on {args.host}:{args.port}"
          + (f" (ignoring {' '.join(args.ignored)})" if args.ignored else ""), file=sys.stderr, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    while engine.running:
        time.sleep(0.05)
    server.server_close()
    print("fake vLLM: stopped", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
