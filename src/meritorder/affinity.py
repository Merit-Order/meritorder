"""Which replica of a GPU deployment holds a session's KV cache, and how much of it is probably still there.

Each replica of a self-hosted deployment has its own prefix cache. When the gateway can address replicas one by one
(a static server's ``static_urls``, SkyPilot's clusters, or Kubernetes pods), it remembers which replica served a session's
last turn (its *placement*) and the router weighs going back there against each replica's load (router.py).

How much of the session's prefix a replica still holds comes from the best source available, in this order:

1. **KV events** (``kv_events.py``): the controller follows vLLM's block stored / removed events, tagged with the
   session, and publishes a note per session: how many of its tokens the replica still holds.
2. **An estimate** from the replica's KV-cache metrics. vLLM evicts the least recently freed blocks first, and a
   finished request's blocks go to the back of that queue, last block first. So everything that was free when the
   turn ended is reused before the session's blocks are, and then its blocks go from the end of the conversation
   backwards. With F the free KV space when the turn ended, S the session's prefix, and W what the replica has
   written into its cache since::

       share still cached = clamp((F + S - W) / S, 0, 1)

   A restarted server (its counters went back, or its start time moved) holds nothing.
3. **The backend's cache TTL** (``cache_ttl_seconds``), as for an API, where neither is available.

Behind a load balancer that ignores sessions (no addressable replicas), a turn lands on the replica holding the
cache about one time in N, so the share is divided by the number of replicas.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

WARM_SHARE = 0.5  # at least this much of the prefix still cached: the session counts as warm on that replica
NOTE_SLACK = 5.0  # seconds: a KV-event note this much older than the turn may still describe it (clocks, batching)
UNREACHABLE_SECONDS = 60.0  # a replica the gateway couldn't connect to is skipped this long (the load balancer serves)


def kv_session_id(state_key: str) -> str:
    """The id a session is known by on the GPUs (vLLM's ``X-Session-ID``, KV-event notes, the console): a hash of
    its user-scoped routing key, so the servers never see who the user is."""
    return hashlib.sha256(state_key.encode()).hexdigest()[:32]


@dataclass
class Replica:
    """One replica of a deployment, as the controller last published it (``deploy:<backend>`` → replicas_info)."""

    id: str
    url: str | None
    status: str = "READY"
    utilization: float | None = None  # vLLM's load at the controller's last sample
    waiting: int | None = None
    kv_capacity: int | None = None  # tokens the KV cache holds
    kv_usage: float | None = None  # share of it held by running requests
    kv_written: float | None = None  # tokens written into the cache since the server started, at sampled_at
    kv_rate: float = 0.0  # tokens per second written, between the last two samples
    sampled_at: float = 0.0
    started_at: float | None = None
    events: bool = False  # the controller follows this replica's KV events

    @property
    def ready(self) -> bool:
        return self.status == "READY" and bool(self.url)

    def written(self, now: float) -> float | None:
        if self.kv_written is None:
            return None
        return self.kv_written + self.kv_rate * max(0.0, now - self.sampled_at)

    def free_tokens(self) -> float | None:
        if not self.kv_capacity:
            return None
        return (1.0 - min(max(self.kv_usage or 0.0, 0.0), 1.0)) * self.kv_capacity

    def unavailable(self) -> str:
        """Why a session can't go back to this replica, in words."""
        if self.status in ("SHUTTING_DOWN", "DRAINING"):
            return "draining"
        if not self.url:
            return "not reachable on its own"
        return self.status.replace("_", " ").lower() or "not ready"


def replicas_of(rec: dict | None) -> list[Replica]:
    """The replicas in a deployment record the controller published."""
    out = []
    for r in (rec or {}).get("replicas_info") or []:
        if r.get("id") is None:
            continue
        out.append(Replica(id=str(r["id"]), url=r.get("url"), status=str(r.get("status") or ""),
                           utilization=r.get("utilization"), waiting=r.get("waiting"), kv_capacity=r.get("kv_capacity_tokens"),
                           kv_usage=r.get("kv_usage"), kv_written=r.get("kv_written_tokens"), kv_rate=r.get("kv_written_per_second") or 0.0,
                           sampled_at=r.get("sampled_at") or 0.0, started_at=r.get("started_at"), events=bool(r.get("kv_events"))))
    return out


def mark(replica: Replica | None, now: float, tokens: int) -> dict:
    """What to remember about where a turn ran, for estimating later how much of it is still cached there."""
    p: dict = {"replica": replica.id if replica else None, "at": now, "tokens": tokens}
    if replica is not None:
        written, free = replica.written(now), replica.free_tokens()
        if written is not None and free is not None:
            # `mark` is a guess from the last sample's rate; `seen` the counter that sample read, which a restart goes below.
            p.update(mark=round(written), seen=round(replica.kv_written or 0), free=round(free))
        if replica.started_at:
            p["started"] = replica.started_at
    return p


@dataclass
class Warmth:
    share: float  # of the session's prefix probably still cached there, 0 to 1
    source: str  # "events", "estimate", "ttl" or "cold"

    @property
    def warm(self) -> bool:
        return self.share >= WARM_SHARE

    def describe(self) -> str:
        if self.source == "cold" or self.share <= 0:
            return "cold" + (f" ({self.source})" if self.source not in ("cold", "ttl") else "")
        return f"{self.share:.0%} warm ({'KV events' if self.source == 'events' else self.source})"


def estimated_share(p: dict, replica: Replica, now: float) -> float | None:
    """How much of a session's prefix a replica probably still holds, from its KV-cache metrics (see the module
    docstring). None when the replica's metrics or the turn's mark are missing."""
    written = replica.written(now)
    if written is None or p.get("mark") is None or p.get("free") is None:
        return None
    restarted = replica.started_at and p.get("started") and replica.started_at > p["started"] + 1
    # Counters that went back: a restart. Compared sample to sample, since `mark` was extrapolated: a replica that wrote
    # less than that guess since (a burst ended) is below it without having restarted, and nothing more was evicted.
    went_back = replica.kv_written < p["seen"] if p.get("seen") is not None else written < p["mark"]
    if restarted or went_back:
        return 0.0
    tokens = max(int(p.get("tokens") or 0), 1)
    return min(1.0, max(0.0, (p["free"] + tokens - (written - p["mark"])) / tokens))


def _fresh(note: dict | None, since: float) -> bool:
    return bool(note) and (note.get("at") or 0) >= since - NOTE_SLACK


async def placements(state, backend: str, rec: dict | None, ttl: float, now: float, limit: int = 50) -> tuple[list[dict], dict[str, dict]]:
    """Where sessions sit on a deployment's replicas, newest first, each with how much of its cache is probably still
    there now (and from what), and per replica how many sessions sit there and how many are warm. For the console."""
    from .sessions import Session  # noqa: PLC0415 (sessions is light, but keep this module free of it at import)

    placed = await state.hgetall_json(f"placements:{backend}")
    reps = {r.id: r for r in replicas_of(rec)}
    count = max(1, len([r for r in reps.values() if r.ready]))
    rows, per_replica = [], {rid: {"sessions": 0, "warm": 0} for rid in reps}
    for kv_id, p in sorted(placed.items(), key=lambda x: -(x[1] or {}).get("at", 0)):
        if not p or p.get("at", 0) < now - 6 * 3600:
            continue
        s = Session(id=p.get("session") or kv_id, last_seen=p["at"], placement={backend: p}, warm_until={backend: p["at"] + ttl})
        rep = reps.get(str(p.get("replica"))) if p.get("replica") is not None else None
        note = ((await state.get_json(f"kv:{kv_id}")) or {}).get(backend) if len(rows) < limit else None
        usable = rep.ready if rep is not None else p.get("replica") is None  # a replica that's gone or going holds nothing usable
        w = share_on(s, backend, rep, now, note, count) if usable else Warmth(0.0, "cold")
        if rep is not None:
            per_replica[rep.id]["sessions"] += 1
            per_replica[rep.id]["warm"] += int(w.warm)
        if len(rows) < limit:
            rows.append({"id": kv_id[:12], "session": p.get("session"), "thread": p.get("thread"), "user": p.get("user"),
                         "team": p.get("team"), "model": p.get("model"), "replica": p.get("replica"), "at": p["at"],
                         "tokens": p.get("tokens"), "expected": p.get("expected"), "share": round(w.share, 3), "source": w.source,
                         "replica_status": rep.status if rep else (None if p.get("replica") is None else "gone")})
    return rows, per_replica


def share_on(session, backend: str, replica: Replica | None, now: float, note: dict | None = None, replicas: int = 1) -> Warmth:
    """How much of ``session``'s prefix is probably cached on ``replica`` of ``backend`` (None: whichever replica the
    load balancer picks, out of ``replicas``). ``note`` is the controller's KV-event note for the session there."""
    if session is None:
        return Warmth(0.0, "cold")
    p = (session.placement or {}).get(backend) or {}
    since = p.get("at") or session.last_seen or 0.0
    events = _fresh(note, since) and bool(note.get("tokens"))
    if replica is None:
        if events:
            w = Warmth(min(1.0, note.get("held", 0) / note["tokens"]), "events")
        else:
            w = Warmth(1.0 if session.warm(backend, now) else 0.0, "ttl")
        return Warmth(w.share / max(1, replicas), w.source)
    if events and note.get("replica") is not None:
        held = note.get("held", 0) / note["tokens"] if str(note["replica"]) == replica.id else 0.0
        return Warmth(min(1.0, held), "events")
    if p.get("replica") != replica.id:
        return Warmth(0.0, "cold")
    est = estimated_share(p, replica, now)
    if est is not None:
        return Warmth(est, "estimate")
    return Warmth(1.0 if session.warm(backend, now) else 0.0, "ttl")
