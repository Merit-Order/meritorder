"""Hot-path state: session metadata, spend counters, rate limits, GPU utilization.

Redis in real deployments; an in-memory version for development and tests.
Postgres (store.py) stays the source of truth for spend; these counters are a
fast cache of it, re-seeded from Postgres when missing.
"""

from __future__ import annotations

import json
import os
import secrets
import time
from collections import deque
from typing import Any

HEALTH_FIELDS = ("requests", "errors", "latency_ms", "ttft_ms", "ttft_n")


def _from_ledger(spend_key: str) -> bool:
    """A month's spend counter (organization, team, person), which the gateway seeds from the ledger when it's missing.
    After a request only one that exists is added to: one started from a single request's cost would read as the whole
    month's spend once Redis had lost it (a flush, a failover, eviction), and budgets would stop binding. A session's
    counter (``session:``) has no ledger to come from, so a request starts it, and so do a conversation's own (``task:``,
    ``task-base:``: what it cost and what it would have cost on the baseline model, for expected_cost's tasks)."""
    return not spend_key.startswith(("session:", "task:", "task-base:"))


def _live_inflight(fields: dict, gateways: dict, within: float = 20) -> dict[str, dict[str, int]]:
    """In-flight requests per gateway per backend, from gateways seen in the last ``within`` seconds."""
    cutoff = time.time() - within
    live = {g for g, t in gateways.items() if float(t) > cutoff}
    out: dict[str, dict[str, int]] = {}
    for field_, n in fields.items():
        gw, _, backend = field_.partition("|")
        if gw in live and int(n):
            out.setdefault(gw, {})[backend] = int(n)
    return out


async def claim(state, key: str, ttl: int) -> bool:
    """Take ``key`` for ``ttl`` seconds unless someone holds it: True for exactly one caller however many ask at once (an
    alert sent once, a worker job only one worker runs). Atomic: a set-if-absent, never a read then a write."""
    token = secrets.token_hex(8)
    return (await state.set_json_once({key: token}, ttl)).get(key) == token


async def release(state, key: str) -> None:
    """Let go of a ``claim`` early (the job is done): the key goes within a second."""
    await state.set_json(key, None, 1)


class MemoryState:
    """Shared state in this process's memory: for development, tests and a single gateway without Redis. It never evicts
    (expired keys, sessions' and tasks' spend counters stay until the process restarts), so its memory grows with traffic;
    production runs on Redis (MERITORDER_REDIS_URL), where keys expire (docs/07-deployment.md)."""

    def __init__(self) -> None:
        self._kv: dict[str, tuple[Any, float]] = {}
        self._spend: dict[str, float] = {}
        self._seeding: dict[str, float] = {}  # what was added to a counter while it was being seeded (begin_seed)
        self._util: dict[str, float] = {}
        self._events: deque = deque(maxlen=500)
        self._alerts: deque = deque(maxlen=200)
        self._health: dict[str, dict[str, float]] = {}
        self._gateways: dict[str, float] = {}
        self._inflight: dict[str, int] = {}
        self._hashes: dict[str, dict[str, Any]] = {}
        self._config_version: str | None = None

    def _get(self, key: str) -> Any:
        item = self._kv.get(key)
        if item is None:
            return None
        value, expires = item
        if expires and expires < time.time():
            self._kv.pop(key, None)
            return None
        return value

    async def ping(self) -> bool:
        return True

    async def close(self) -> None:
        return None

    async def get_session(self, sid: str) -> dict | None:
        return self._get(f"sess:{sid}")

    async def put_session(self, sid: str, data: dict, ttl: int) -> None:
        self._kv[f"sess:{sid}"] = (data, time.time() + ttl)

    async def get_spend(self, key: str) -> float | None:
        return self._spend.get(key)

    async def begin_seed(self, key: str) -> None:
        """A counter is about to be seeded from the ledger: from now until ``seed_spend``, what requests add to it is kept
        aside and added to the seed, so a request that finishes while the ledger is being read isn't lost."""
        if key not in self._spend:
            self._seeding.setdefault(key, 0.0)

    async def seed_spend(self, key: str, amount: float) -> float:
        """Start a missing counter at ``amount`` (plus what was added since ``begin_seed``); returns what it holds."""
        pending = self._seeding.pop(key, 0.0)
        return self._spend.setdefault(key, amount + pending)

    async def spend_counters(self, prefix: str) -> dict[str, float]:
        """Every spend counter whose key starts with ``prefix`` (a month's: ``2026-11:``), for checking against the ledger."""
        return {k: v for k, v in self._spend.items() if k.startswith(prefix)}

    async def add_spend(self, keys: list[str], amount: float) -> None:
        for k in keys:
            self._spend[k] = self._spend.get(k, 0.0) + amount

    async def adjust_spend(self, keys: list[str], amount: float) -> None:
        """Add to the counters that exist, and to those being seeded (``begin_seed``). A missing one is seeded from the
        ledger (and reservations) when next read, which already counts what this would add."""
        for k in keys:
            if k in self._spend:
                self._spend[k] += amount
            elif k in self._seeding:
                self._seeding[k] += amount

    async def admit(self, spend_keys: list[str], rates: list[tuple[str, int, int]], session_id: str | None,
                    thread_id: str | None = None, kv_id: str | None = None) -> dict:
        state_id = thread_id or session_id
        return {
            "spend": {k: self._spend.get(k) for k in spend_keys},
            "rates": {key: await self.rate_hit(key, window, amount) for key, window, amount in rates},
            "session": await self.get_session(state_id) if state_id else None,
            "session_spend": self._spend.get(f"session:{session_id}") if session_id else None,
            "util": dict(self._util),
            "inflight": _live_inflight(self._inflight, self._gateways),
            "kv": self._get(f"kv:{kv_id}") if kv_id else None,
        }

    async def publish_inflight(self, gateway_id: str, counts: dict[str, int]) -> None:
        self._gateways[gateway_id] = time.time()
        self._inflight.update({f"{gateway_id}|{b}": n for b, n in counts.items()})

    async def record_request(self, event: dict | None, health: tuple | None, spend_keys: list[str], amount: float,
                             extra: dict[str, float] | None = None) -> None:
        """``spend_keys``: the month's counters only grow where they exist (see ``_from_ledger``); a session's starts here.
        ``extra``: other counters that start here, each with its own amount (a conversation's baseline cost)."""
        if event is not None:
            await self.push_event(event)
        if health is not None:
            await self.record_backend(*health)
        if amount:
            await self.adjust_spend([k for k in spend_keys if _from_ledger(k)], amount)
            await self.add_spend([k for k in spend_keys if not _from_ledger(k)], amount)
        for k, v in (extra or {}).items():
            if v:
                await self.add_spend([k], v)

    async def rate_hit(self, key: str, window_seconds: int, amount: int = 1) -> int:
        bucket = f"rate:{key}:{int(time.time() // window_seconds)}"
        count = (self._get(bucket) or 0) + amount
        self._kv[bucket] = (count, time.time() + window_seconds)
        return count

    async def utilization(self) -> dict[str, float]:
        return dict(self._util)

    async def gpu_load(self) -> dict:
        """The GPUs' load outside a request: the controller's samples and every live process's in-flight counts (batch
        work on GPUs, batches/gpu.py)."""
        return {"util": dict(self._util), "inflight": _live_inflight(self._inflight, self._gateways)}

    async def set_utilization(self, backend: str, util: float) -> None:
        self._util[backend] = util

    async def set_json(self, key: str, value: Any, ttl: int = 0) -> None:
        self._kv[key] = (value, time.time() + ttl if ttl else 0)

    async def get_json(self, key: str) -> Any:
        return self._get(key)

    # --- write-once values (model-based compression's decisions, compressor.py)
    async def get_json_many(self, keys: list[str], ttl: int) -> list[Any]:
        """The values at ``keys`` (None where missing); each found one lives ``ttl`` seconds more."""
        out = []
        for k in keys:
            v = self._get(k)
            if v is not None:
                self._kv[k] = (v, time.time() + ttl)
            out.append(v)
        return out

    async def set_json_once(self, items: dict[str, Any], ttl: int) -> dict[str, Any]:
        """Store each value unless its key already has one; returns what each key holds now (the first writer wins)."""
        out = {}
        for k, v in items.items():
            if self._get(k) is None:
                self._kv[k] = (v, time.time() + ttl)
            out[k] = self._get(k)
        return out

    # --- small JSON hashes (where sessions sit on GPU replicas, for the console; people's Google groups)
    async def hset_json(self, key: str, field_: str, value: Any) -> None:
        self._hashes.setdefault(key, {})[field_] = value

    async def hgetall_json(self, key: str) -> dict[str, Any]:
        return dict(self._hashes.get(key, {}))

    async def hget_json(self, key: str, field_: str) -> Any:
        return self._hashes.get(key, {}).get(field_)

    async def hdel(self, key: str, fields: list[str]) -> None:
        for f in fields:
            self._hashes.get(key, {}).pop(f, None)

    # --- live telemetry for the console
    async def push_event(self, event: dict) -> None:
        self._events.appendleft(event)

    async def recent_events(self, limit: int = 100) -> list[dict]:
        return list(self._events)[:limit]

    async def push_alert(self, alert: dict) -> None:
        self._alerts.appendleft(alert)

    async def recent_alerts(self, limit: int = 50) -> list[dict]:
        return list(self._alerts)[:limit]

    async def record_backend(self, backend: str, ok: bool, latency_ms: int | None, ttft_ms: int | None) -> None:
        minute = int(time.time() // 60)
        h = self._health.setdefault(f"{backend}:{minute}", dict.fromkeys(HEALTH_FIELDS, 0.0))
        h["requests"] += 1
        h["errors"] += 0 if ok else 1
        h["latency_ms"] += latency_ms or 0
        if ttft_ms:
            h["ttft_ms"] += ttft_ms
            h["ttft_n"] += 1

    async def backend_health(self, backends: list[str], minutes: int = 5) -> dict[str, dict[str, float]]:
        now = int(time.time() // 60)
        out = {}
        for b in backends:
            agg = dict.fromkeys(HEALTH_FIELDS, 0.0)
            for m in range(now - minutes + 1, now + 1):
                for k, v in self._health.get(f"{b}:{m}", {}).items():
                    agg[k] += v
            out[b] = agg
        return out

    async def heartbeat(self, gateway_id: str) -> None:
        self._gateways[gateway_id] = time.time()

    async def live_gateways(self, within: int = 20) -> int:
        return sum(1 for t in self._gateways.values() if t > time.time() - within)

    async def get_config_token(self) -> str | None:
        return self._config_version

    async def set_config_token(self, token: str) -> None:
        self._config_version = token


class RedisState:
    def __init__(self, url: str) -> None:
        import redis.asyncio as redis

        # A blocking pool waits for a free connection instead of failing under bursts.
        pool = redis.BlockingConnectionPool.from_url(url, decode_responses=True, timeout=10,
                                                     max_connections=int(os.environ.get("MERITORDER_REDIS_MAX_CONNECTIONS", "200")))
        self.r = redis.Redis(connection_pool=pool)

    async def ping(self) -> bool:
        return bool(await self.r.ping())

    async def close(self) -> None:
        await self.r.aclose()

    async def get_session(self, sid: str) -> dict | None:
        raw = await self.r.get(f"mo:sess:{sid}")
        return json.loads(raw) if raw else None

    async def put_session(self, sid: str, data: dict, ttl: int) -> None:
        await self.r.set(f"mo:sess:{sid}", json.dumps(data), ex=ttl)

    async def get_spend(self, key: str) -> float | None:
        raw = await self.r.get(f"mo:spend:{key}")
        return float(raw) if raw is not None else None

    async def begin_seed(self, key: str) -> None:
        """See MemoryState.begin_seed. The aside lives a minute at most: a seeder that died leaves nothing behind."""
        await self.r.set(f"mo:spend-seed:{key}", 0, nx=True, ex=60)

    _SEED = ("if redis.call('exists', KEYS[1]) == 0 then redis.call('set', KEYS[1], ARGV[1], 'EX', ARGV[2]) "
             "local p = redis.call('get', KEYS[2]) if p then redis.call('incrbyfloat', KEYS[1], p) end end "
             "redis.call('del', KEYS[2]) return redis.call('get', KEYS[1])")

    async def seed_spend(self, key: str, amount: float) -> float:
        """See MemoryState.seed_spend; atomic, so the first seeder wins and every other reads its value."""
        return float(await self.r.eval(self._SEED, 2, f"mo:spend:{key}", f"mo:spend-seed:{key}", repr(float(amount)), 40 * 86400))

    async def spend_counters(self, prefix: str) -> dict[str, float]:
        """See MemoryState.spend_counters (SCAN, so Redis isn't blocked)."""
        keys = [k async for k in self.r.scan_iter(match=f"mo:spend:{prefix}*", count=1000)]
        values = await self.r.mget(keys) if keys else []
        return {k[len("mo:spend:"):]: float(v) for k, v in zip(keys, values) if v is not None}

    async def add_spend(self, keys: list[str], amount: float) -> None:
        pipe = self.r.pipeline()
        for k in keys:
            pipe.incrbyfloat(f"mo:spend:{k}", amount)
            pipe.expire(f"mo:spend:{k}", 40 * 86400)
        await pipe.execute()

    # KEYS: the counters, then each one's seeding aside (begin_seed), in the same order.
    _ADJUST = ("local n = #KEYS / 2 for i = 1, n do if redis.call('exists', KEYS[i]) == 1 then redis.call('incrbyfloat', KEYS[i], ARGV[1]) "
               "elseif redis.call('exists', KEYS[n + i]) == 1 then redis.call('incrbyfloat', KEYS[n + i], ARGV[1]) end end return 0")

    @staticmethod
    def _adjust_keys(keys: list[str]) -> list[str]:
        return [f"mo:spend:{k}" for k in keys] + [f"mo:spend-seed:{k}" for k in keys]

    async def adjust_spend(self, keys: list[str], amount: float) -> None:
        """Add to the counters that exist, or are being seeded (see MemoryState.adjust_spend), atomically."""
        if keys and amount:
            await self.r.eval(self._ADJUST, 2 * len(keys), *self._adjust_keys(keys), repr(float(amount)))

    async def admit(self, spend_keys: list[str], rates: list[tuple[str, int, int]], session_id: str | None,
                    thread_id: str | None = None, kv_id: str | None = None) -> dict:
        """Everything a request needs before routing, in one round trip: spend counters, rate-limit hits
        (counted now), the conversation's routing state (``thread_id``, or the session's), the session's spend,
        GPU utilization, and what the GPUs' KV events say about the conversation's cache (``kv_id``)."""
        state_id = thread_id or session_id
        pipe = self.r.pipeline(transaction=False)
        pipe.mget([f"mo:spend:{k}" for k in spend_keys] or ["mo:spend:-"])
        now = time.time()
        for key, window, amount in rates:
            bucket = f"mo:rate:{key}:{int(now // window)}"
            pipe.incrby(bucket, amount)
            pipe.expire(bucket, window * 2)
        pipe.get(f"mo:sess:{state_id}" if state_id else "mo:sess:-")
        pipe.get(f"mo:spend:session:{session_id}" if session_id else "mo:spend:-")
        pipe.hgetall("mo:util")
        pipe.hgetall("mo:inflight")
        pipe.hgetall("mo:gateways")
        pipe.get(f"mo:kv:{kv_id}" if kv_id else "mo:kv:-")
        res = await pipe.execute()
        spends = res[0]
        rate_counts = res[1:1 + 2 * len(rates):2]
        sess_raw, sess_spend, util, inflight, gateways, kv_raw = res[1 + 2 * len(rates):]
        return {
            "kv": json.loads(kv_raw) if kv_id and kv_raw else None,
            "inflight": _live_inflight(inflight, gateways),
            "spend": {k: (float(v) if v is not None else None) for k, v in zip(spend_keys, spends)},
            "rates": {key: int(c) for (key, _, _), c in zip(rates, rate_counts)},
            "session": json.loads(sess_raw) if state_id and sess_raw else None,
            "session_spend": float(sess_spend) if session_id and sess_spend is not None else None,
            "util": {k: float(v) for k, v in util.items()},
        }

    async def record_request(self, event: dict | None, health: tuple | None, spend_keys: list[str], amount: float,
                             extra: dict[str, float] | None = None) -> None:
        """After a request: feed event, backend health, and spend counters in one round trip."""
        pipe = self.r.pipeline(transaction=False)
        if event is not None:
            pipe.lpush("mo:events", json.dumps(event, default=str))
            pipe.ltrim("mo:events", 0, 499)
        if health is not None:
            backend, ok, latency_ms, ttft_ms = health
            key = f"mo:bh:{backend}:{int(time.time() // 60)}"
            pipe.hincrby(key, "requests", 1)
            if not ok:
                pipe.hincrby(key, "errors", 1)
            if latency_ms:
                pipe.hincrby(key, "latency_ms", int(latency_ms))
            if ttft_ms:
                pipe.hincrby(key, "ttft_ms", int(ttft_ms))
                pipe.hincrby(key, "ttft_n", 1)
            pipe.expire(key, 1800)
        if amount:
            month = [k for k in spend_keys if _from_ledger(k)]
            if month:
                pipe.eval(self._ADJUST, 2 * len(month), *self._adjust_keys(month), repr(float(amount)))
            for k in spend_keys:
                if not _from_ledger(k):
                    pipe.incrbyfloat(f"mo:spend:{k}", amount)
                    pipe.expire(f"mo:spend:{k}", 40 * 86400)
        for k, v in (extra or {}).items():
            if v:
                pipe.incrbyfloat(f"mo:spend:{k}", v)
                pipe.expire(f"mo:spend:{k}", 40 * 86400)
        await pipe.execute()

    async def publish_inflight(self, gateway_id: str, counts: dict[str, int]) -> None:
        """Heartbeat plus this gateway's in-flight requests per backend; forget gateways gone for 2 minutes."""
        now = time.time()
        pipe = self.r.pipeline(transaction=False)
        pipe.hset("mo:gateways", gateway_id, now)
        if counts:
            pipe.hset("mo:inflight", mapping={f"{gateway_id}|{b}": n for b, n in counts.items()})
        pipe.hgetall("mo:gateways")
        pipe.hkeys("mo:inflight")
        res = await pipe.execute()
        gateways, fields = res[-2], res[-1]
        dead = {g for g, t in gateways.items() if float(t) < now - 120}
        if dead:
            pipe = self.r.pipeline(transaction=False)
            pipe.hdel("mo:gateways", *dead)
            stale = [f for f in fields if f.split("|", 1)[0] in dead]
            if stale:
                pipe.hdel("mo:inflight", *stale)
            await pipe.execute()

    async def rate_hit(self, key: str, window_seconds: int, amount: int = 1) -> int:
        bucket = f"mo:rate:{key}:{int(time.time() // window_seconds)}"
        pipe = self.r.pipeline()
        pipe.incrby(bucket, amount)
        pipe.expire(bucket, window_seconds * 2)
        count, _ = await pipe.execute()
        return int(count)

    async def utilization(self) -> dict[str, float]:
        raw = await self.r.hgetall("mo:util")
        return {k: float(v) for k, v in raw.items()}

    async def gpu_load(self) -> dict:
        pipe = self.r.pipeline(transaction=False)
        pipe.hgetall("mo:util")
        pipe.hgetall("mo:inflight")
        pipe.hgetall("mo:gateways")
        util, inflight, gateways = await pipe.execute()
        return {"util": {k: float(v) for k, v in util.items()}, "inflight": _live_inflight(inflight, gateways)}

    async def set_utilization(self, backend: str, util: float) -> None:
        await self.r.hset("mo:util", backend, util)

    async def set_json(self, key: str, value: Any, ttl: int = 0) -> None:
        await self.r.set(f"mo:{key}", json.dumps(value), ex=ttl or None)

    async def get_json(self, key: str) -> Any:
        raw = await self.r.get(f"mo:{key}")
        return json.loads(raw) if raw else None

    async def get_json_many(self, keys: list[str], ttl: int) -> list[Any]:
        """The values at ``keys`` (None where missing), in one round trip; each found one lives ``ttl`` seconds more."""
        pipe = self.r.pipeline(transaction=False)
        for k in keys:
            pipe.getex(f"mo:{k}", ex=ttl)
        return [json.loads(raw) if raw else None for raw in await pipe.execute()]

    async def set_json_once(self, items: dict[str, Any], ttl: int) -> dict[str, Any]:
        """Store each value unless its key already has one, and return what each key holds now, in one transaction:
        whichever gateway writes a key first wins, and every other gets the winner's value."""
        pipe = self.r.pipeline(transaction=True)
        for k, v in items.items():
            pipe.set(f"mo:{k}", json.dumps(v), ex=ttl, nx=True)
            pipe.get(f"mo:{k}")
        res = await pipe.execute()
        return {k: json.loads(raw) if raw else None for k, raw in zip(items, res[1::2])}

    async def hset_json(self, key: str, field_: str, value: Any) -> None:
        await self.r.hset(f"mo:{key}", field_, json.dumps(value))

    async def hgetall_json(self, key: str) -> dict[str, Any]:
        return {k: json.loads(v) for k, v in (await self.r.hgetall(f"mo:{key}")).items()}

    async def hget_json(self, key: str, field_: str) -> Any:
        raw = await self.r.hget(f"mo:{key}", field_)
        return json.loads(raw) if raw else None

    async def hdel(self, key: str, fields: list[str]) -> None:
        if fields:
            await self.r.hdel(f"mo:{key}", *fields)

    # --- live telemetry for the console (shared by every gateway)
    async def push_event(self, event: dict) -> None:
        pipe = self.r.pipeline()
        pipe.lpush("mo:events", json.dumps(event, default=str))
        pipe.ltrim("mo:events", 0, 499)
        await pipe.execute()

    async def recent_events(self, limit: int = 100) -> list[dict]:
        return [json.loads(x) for x in await self.r.lrange("mo:events", 0, limit - 1)]

    async def push_alert(self, alert: dict) -> None:
        pipe = self.r.pipeline()
        pipe.lpush("mo:alerts", json.dumps(alert, default=str))
        pipe.ltrim("mo:alerts", 0, 199)
        await pipe.execute()

    async def recent_alerts(self, limit: int = 50) -> list[dict]:
        return [json.loads(x) for x in await self.r.lrange("mo:alerts", 0, limit - 1)]

    async def record_backend(self, backend: str, ok: bool, latency_ms: int | None, ttft_ms: int | None) -> None:
        key = f"mo:bh:{backend}:{int(time.time() // 60)}"
        pipe = self.r.pipeline()
        pipe.hincrby(key, "requests", 1)
        if not ok:
            pipe.hincrby(key, "errors", 1)
        if latency_ms:
            pipe.hincrby(key, "latency_ms", int(latency_ms))
        if ttft_ms:
            pipe.hincrby(key, "ttft_ms", int(ttft_ms))
            pipe.hincrby(key, "ttft_n", 1)
        pipe.expire(key, 1800)
        await pipe.execute()

    async def backend_health(self, backends: list[str], minutes: int = 5) -> dict[str, dict[str, float]]:
        now = int(time.time() // 60)
        pipe = self.r.pipeline()
        for b in backends:
            for m in range(now - minutes + 1, now + 1):
                pipe.hgetall(f"mo:bh:{b}:{m}")
        results = await pipe.execute()
        out, i = {}, 0
        for b in backends:
            agg = dict.fromkeys(HEALTH_FIELDS, 0.0)
            for _ in range(minutes):
                for k, v in (results[i] or {}).items():
                    agg[k] = agg.get(k, 0.0) + float(v)
                i += 1
            out[b] = agg
        return out

    async def heartbeat(self, gateway_id: str) -> None:
        await self.r.hset("mo:gateways", gateway_id, time.time())

    async def live_gateways(self, within: int = 20) -> int:
        raw = await self.r.hgetall("mo:gateways")
        return sum(1 for t in raw.values() if float(t) > time.time() - within)

    async def get_config_token(self) -> str | None:
        return await self.r.get("mo:config_token")

    async def set_config_token(self, token: str) -> None:
        await self.r.set("mo:config_token", token)


def make_state(redis_url: str | None):
    return RedisState(redis_url) if redis_url else MemoryState()
