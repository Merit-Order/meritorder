"""Fill a development database with synthetic ledger rows so the reporting UI has something to show.

Never run this against a real deployment: every row is invented.

    MERITORDER_DATABASE_URL=... uv run python scripts/seed_demo.py --days 30
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import uuid
from datetime import UTC, datetime, timedelta

from meritorder.config import Price
from meritorder.pricing import Usage, attribute
from meritorder.store import make_store

SONNET = Price(input=2, output=10, cache_read=0.2, cache_write=2.5)
MODELS = [  # (name, backend, price, share, floor)
    ("claude-sonnet", "bedrock", SONNET, 0.30, "coding:5"),
    ("claude-opus", "anthropic", Price(input=4, output=20, cache_read=0.2, cache_write=5), 0.07, "coding:5"),
    ("kimi-k3.1", "openrouter", Price(input=2, output=10, cache_read=0.2), 0.10, "coding:4"),
    ("qwen3-coder-next", "local-vllm", Price(input=0.12, output=0.8, cache_read=0.07), 0.25, "coding:3"),
    ("deepseek-v4.1-flash", "openrouter", Price(input=0.051, output=0.567, cache_read=0.018), 0.18, "general:2"),
    ("claude-haiku", "bedrock", Price(input=0.1, output=0.5, cache_read=0.01, cache_write=0.125), 0.10, "general:1"),
]
TEAMS = {"platform": ["ana@example.com", "raj@example.com", "kim@example.com"],
         "data-science": ["lee@example.com", "sam@example.com"],
         "support-bots": ["svc-helpdesk"]}
PROJECTS = ["checkout-rewrite", "search", "infra", "eval-suite"]


async def main(days: int, per_day: int) -> None:
    store = make_store(os.environ.get("MERITORDER_DATABASE_URL"))
    if store.kind != "postgres":
        raise SystemExit("set MERITORDER_DATABASE_URL to a development database")
    await store.init()
    await store.migrate()
    rng = random.Random(7)
    now = datetime.now(UTC)
    n = 0
    for d in range(days, -1, -1):
        day = now - timedelta(days=d)
        weekday = day.weekday() < 5
        for _ in range(int(per_day * (1.0 if weekday else 0.25) * rng.uniform(0.8, 1.2))):
            name, backend, price, _, floor = rng.choices(MODELS, weights=[m[3] for m in MODELS])[0]
            team = rng.choice(list(TEAMS))
            user = rng.choice(TEAMS[team])
            prompt = int(rng.lognormvariate(10.3, 0.8))
            cached = int(prompt * rng.uniform(0.6, 0.95))
            usage = Usage(input_tokens=int((prompt - cached) * 0.3), cache_read_tokens=cached,
                          cache_write_tokens=int((prompt - cached) * 0.7), output_tokens=int(rng.lognormvariate(6.2, 0.7)))
            self_hosted = backend == "local-vllm"
            costs = attribute(usage, price, SONNET, proxy_cache=rng.random() < 0.7,
                              self_hosted_cost=(usage.total_tokens / 2500) * 8 / 3600 if self_hosted else None)
            ts = day.replace(hour=rng.randint(8, 19), minute=rng.randint(0, 59), second=rng.randint(0, 59))
            if ts > now:
                ts = now - timedelta(minutes=rng.randint(1, 60))
            await store.record({
                "id": uuid.uuid4(), "ts": ts, "user_id": user, "team": team, "key_id": "demo", "session_id": f"demo-{rng.randint(1, 400)}",
                "tags": {"project": rng.choice(PROJECTS)}, "client_format": rng.choice(["openai", "anthropic"]),
                "requested_model": "auto", "floor": floor, "served_model": name, "upstream_model": name, "backend": backend,
                "backend_kind": "self_hosted" if self_hosted else "api", "status": 200 if rng.random() > 0.01 else 529, "error": None,
                "stream": True, "upgraded": self_hosted and floor == "general:2", "proxy_cache": True,
                "input_tokens": usage.input_tokens, "cache_read_tokens": usage.cache_read_tokens,
                "cache_write_tokens": usage.cache_write_tokens, "output_tokens": usage.output_tokens,
                "latency_ms": int(rng.lognormvariate(8.3, 0.6)), "ttft_ms": int(rng.lognormvariate(6.5, 0.5)),
                "actual_cost": costs.actual, "served_api_cost": costs.served_api, "baseline_cost": costs.baseline,
                "list_cost": costs.list_price, "savings_routing": costs.savings_routing, "savings_caching": costs.savings_caching,
                "savings_self_hosting": costs.savings_self_hosting, "savings_compression": 0.0, "savings_compression_model": 0.0, "savings_response_cache": 0.0,
            })
            n += 1
    rolled = await store.rollup(full=True)
    await store.close()
    print(f"seeded {n} synthetic requests over {days + 1} days ({rolled} recent hourly rollup rows)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--per-day", type=int, default=120)
    a = p.parse_args()
    asyncio.run(main(a.days, a.per_day))
