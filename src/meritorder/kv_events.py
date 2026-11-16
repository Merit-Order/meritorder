"""Which sessions' KV-cache blocks each GPU replica still holds, from vLLM's KV events (docs/04-caching-and-sessions.md).

vLLM publishes a batch of events per engine step on a ZeroMQ PUB socket when started with
``--kv-events-config '{"enable_kv_cache_events": true, "publisher": "zmq", "endpoint": "tcp://*:5557"}'``
(the SkyPilot provisioner adds it when ``serve.kv_events`` is on). Each message is three frames: topic, an 8-byte
big-endian sequence number, and a msgpack payload: ``[ts, [event, ...], data_parallel_rank]``. Events are
``BlockStored`` (block hashes, the parent block's hash, token ids, block size, medium, and since vLLM 0.30 the
``session_id`` the request came with, which the gateway sets to ``affinity.kv_session_id`` through
``X-Session-ID``), ``BlockRemoved`` (block hashes, medium) and ``AllBlocksCleared``. vLLM encoded them as arrays up
to mid-2026 and as maps since; both are read.

The controller keeps, per replica, each session's chain of blocks in the order they were stored, and which of them
are still held. A store whose parent is unknown (or none) starts a new chain: a new conversation, or one whose start
is shared with another session. A store whose parent is inside the chain cuts the chain there first (the
conversation was edited from that point). What's held is the chain up to its first missing block, which is what a
prefix-cache lookup would find. Only blocks on the GPU count (``medium`` GPU, or unset).

When a session's held share changes, the controller publishes a note for it (``kv:<session id>``, per deployment:
replica, tokens held, tokens in the chain, when), which gateways read with the session (affinity.share_on). A gap in
the sequence numbers means events were lost: that replica's index is dropped, and its sessions fall back to the
estimate until they're seen again.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable

log = logging.getLogger("meritorder.kv_events")

MAX_SESSIONS = 20_000  # chains kept per replica; the least recently stored are forgotten first
NOTE_TTL = 6 * 3600  # as long as the gateway keeps the session itself
FLUSH_SECONDS = 1.0
GPU_MEDIA = (None, "GPU", "gpu")

# Field order of the array encoding (vLLM before mid-2026: msgspec array_like structs, tag first).
_STORED = ("block_hashes", "parent_block_hash", "token_ids", "block_size", "lora_id", "medium", "lora_name", "extra_keys",
           "group_idx", "kv_cache_spec_kind", "kv_cache_spec_sliding_window", "locality", "session_id")
_REMOVED = ("block_hashes", "medium", "group_idx", "locality")


def _key(h: Any) -> Any:
    """Block hashes arrive as bytes or ints; lists (old tuple hashes) aren't hashable."""
    return tuple(h) if isinstance(h, list) else h


def decode_event(raw: Any) -> dict | None:
    """One event as a dict with ``type``, from either encoding. None for anything unrecognised."""
    if isinstance(raw, dict):
        kind = raw.get("type")
        return {**raw, "type": kind} if kind in ("BlockStored", "BlockRemoved", "AllBlocksCleared") else None
    if isinstance(raw, (list, tuple)) and raw and isinstance(raw[0], str):
        names = {"BlockStored": _STORED, "BlockRemoved": _REMOVED, "AllBlocksCleared": ()}.get(raw[0])
        if names is None:
            return None
        return {"type": raw[0], **dict(zip(names, raw[1:]))}
    return None


def decode_batch(obj: Any) -> tuple[float, list[dict]]:
    """A KVEventBatch (already unpacked from msgpack) as (timestamp, events)."""
    if isinstance(obj, dict):  # not how vLLM sends it, but cheap to accept
        ts, events = obj.get("ts") or 0.0, obj.get("events") or []
    elif isinstance(obj, (list, tuple)) and len(obj) >= 2:
        ts, events = obj[0], obj[1] or []
    else:
        return 0.0, []
    return float(ts or 0.0), [e for e in (decode_event(x) for x in events) if e is not None]


@dataclass
class Chain:
    hashes: list = field(default_factory=list)
    block_size: int = 16
    updated: float = 0.0


class ReplicaIndex:
    """Sessions' block chains on one replica, and which of those blocks it still holds."""

    def __init__(self, replica: str, max_sessions: int = MAX_SESSIONS):
        self.replica = replica
        self.max_sessions = max_sessions
        self.chains: OrderedDict[str, Chain] = OrderedDict()
        self.owners: dict[Any, set[str]] = {}  # block hash -> sessions whose chain has it
        self.held: set = set()

    def _drop(self, sid: str) -> None:
        chain = self.chains.pop(sid, None)
        for h in chain.hashes if chain else []:
            who = self.owners.get(h)
            if who is not None:
                who.discard(sid)
                if not who:
                    self.owners.pop(h, None)
                    self.held.discard(h)

    def _cut(self, sid: str, chain: Chain, keep: int) -> None:
        for h in chain.hashes[keep:]:
            who = self.owners.get(h)
            if who is not None:
                who.discard(sid)
                if not who:
                    self.owners.pop(h, None)
                    self.held.discard(h)
        del chain.hashes[keep:]

    def apply(self, ev: dict, now: float) -> set[str]:
        """Apply one event. Returns the sessions whose held share may have changed."""
        kind = ev.get("type")
        if kind == "AllBlocksCleared":
            self.held.clear()
            return set(self.chains)
        if ev.get("medium") not in GPU_MEDIA or (ev.get("group_idx") or 0) != 0:
            return set()  # offloaded copies, and the extra KV groups of hybrid models, aren't tracked
        hashes = [_key(h) for h in ev.get("block_hashes") or []]
        if kind == "BlockRemoved":
            dirty: set[str] = set()
            for h in hashes:
                if h in self.held:
                    self.held.discard(h)
                    dirty |= self.owners.get(h, set())
            return dirty
        # BlockStored
        dirty = set()
        for h in hashes:  # blocks some chain has, stored again (by this session or another): held again
            if h in self.owners and h not in self.held:
                self.held.add(h)
                dirty |= self.owners[h]
        sid = ev.get("session_id")
        if not sid or not hashes:
            return dirty
        parent = _key(ev.get("parent_block_hash"))
        chain = self.chains.get(sid)
        at = chain.hashes.index(parent) if chain is not None and parent is not None and parent in chain.hashes else None
        if chain is None or at is None:
            self._drop(sid)
            chain = Chain()
            self.chains[sid] = chain
        else:
            self._cut(sid, chain, at + 1)
            self.chains.move_to_end(sid)
        size = ev.get("block_size") or (len(ev.get("token_ids") or []) // len(hashes) if ev.get("token_ids") else 0)
        if size:
            chain.block_size = int(size)
        for h in hashes:
            chain.hashes.append(h)
            self.owners.setdefault(h, set()).add(sid)
            self.held.add(h)
        chain.updated = now
        dirty.add(sid)
        while len(self.chains) > self.max_sessions:
            self._drop(next(iter(self.chains)))
        return dirty

    def note(self, sid: str) -> dict | None:
        """Tokens of the session's chain held (up to the first missing block) and in all, or None if unknown."""
        chain = self.chains.get(sid)
        if chain is None:
            return None
        held = 0
        for h in chain.hashes:
            if h not in self.held:
                break
            held += 1
        return {"replica": self.replica, "held": held * chain.block_size, "tokens": len(chain.hashes) * chain.block_size,
                "updated": chain.updated}


class DeploymentEvents:
    """KV events from every replica of one deployment, folded into notes per session."""

    def __init__(self, backend: str, state, unpack: Callable[[bytes], Any] | None = None, clock: Callable[[], float] = time.time):
        self.backend = backend
        self.state = state
        self.unpack = unpack
        self.clock = clock
        self.indexes: dict[str, ReplicaIndex] = {}
        self.last_seq: dict[str, int] = {}
        self.dirty: set[str] = set()

    def receive(self, replica: str, seq: int, payload: Any) -> None:
        """One message from a replica: its sequence number and payload (bytes, or already unpacked)."""
        idx = self.indexes.get(replica)
        last = self.last_seq.get(replica)
        if idx is None or (last is not None and seq != last + 1):
            if idx is not None:
                log.warning("%s replica %s: KV events %s to %s were lost; forgetting what it holds", self.backend, replica, last, seq)
                self.dirty |= set(idx.chains)
            idx = self.indexes[replica] = ReplicaIndex(replica)
        self.last_seq[replica] = seq
        obj = self.unpack(payload) if self.unpack is not None and isinstance(payload, (bytes, bytearray)) else payload
        _, events = decode_batch(obj)
        now = self.clock()
        for ev in events:
            self.dirty |= idx.apply(ev, now)

    def forget(self, replica: str) -> None:
        """The replica is gone: its sessions hold nothing there any more."""
        idx = self.indexes.pop(replica, None)
        self.last_seq.pop(replica, None)
        if idx is not None:
            self.dirty |= set(idx.chains)

    def latest(self, sid: str) -> dict | None:
        """The session's chain on the replica that stored it last (where its next turn would find it)."""
        notes = [n for n in (idx.note(sid) for idx in self.indexes.values()) if n]
        return max(notes, key=lambda n: n["updated"]) if notes else None

    async def flush(self) -> int:
        """Publish a note for every session that changed; returns how many."""
        dirty, self.dirty = self.dirty, set()
        now = self.clock()
        for sid in dirty:
            n = self.latest(sid)
            rec = await self.state.get_json(f"kv:{sid}") or {}
            if n:
                rec[self.backend] = {"replica": n["replica"], "held": n["held"], "tokens": n["tokens"], "at": now}
            else:  # no longer known here (events lost, replica gone): the gateways estimate instead
                rec.pop(self.backend, None)
            await self.state.set_json(f"kv:{sid}", rec, NOTE_TTL)
        return len(dirty)

    def sessions(self, replica: str) -> int:
        idx = self.indexes.get(replica)
        return len(idx.chains) if idx else 0


def _zmq_socket(endpoint: str):
    import zmq
    import zmq.asyncio

    ctx = zmq.asyncio.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.SUBSCRIBE, b"")
    sock.setsockopt(zmq.RCVHWM, 100_000)
    sock.connect(endpoint)
    return sock


def _msgpack_unpack():
    import msgpack

    return lambda payload: msgpack.unpackb(payload, raw=False, strict_map_key=False)


def available() -> str | None:
    """None when the KV-event subscriber can run here, else why not."""
    try:
        import msgpack  # noqa: F401
        import zmq  # noqa: F401
    except ImportError:
        return "install the kv-events extra (pyzmq, msgpack) where the controller runs"
    return None


class Subscriber:
    """Follows the KV events of every ready replica of the deployments that ask for it, in the controller."""

    def __init__(self, state, socket: Callable[[str], Any] | None = None, unpack: Callable[[bytes], Any] | None = None):
        self.state = state
        self.socket = socket
        self.unpack = unpack
        self.deployments: dict[str, DeploymentEvents] = {}
        self.tasks: dict[tuple[str, str], tuple[str, asyncio.Task]] = {}  # (backend, replica) -> (endpoint, reader)
        self.flusher: asyncio.Task | None = None

    def sync(self, backend: str, endpoints: dict[str, str]) -> None:
        """Follow these replicas of ``backend`` (replica id -> tcp:// endpoint), and stop following any others."""
        if self.socket is None:
            if available():
                return
            self.socket, self.unpack = _zmq_socket, _msgpack_unpack()
        dep = self.deployments.setdefault(backend, DeploymentEvents(backend, self.state, self.unpack))
        for (b, rid), (endpoint, task) in list(self.tasks.items()):
            if b == backend and endpoints.get(rid) != endpoint:
                task.cancel()
                del self.tasks[(b, rid)]
                dep.forget(rid)
        for rid, endpoint in endpoints.items():
            running = self.tasks.get((backend, rid))
            if running is None or running[1].done():  # new, or its reader stopped: (re)connect
                self.tasks[(backend, rid)] = (endpoint, asyncio.create_task(self._read(dep, rid, endpoint)))
        if self.flusher is None or self.flusher.done():
            self.flusher = asyncio.create_task(self._flush_forever())

    def following(self, backend: str) -> set[str]:
        return {rid for (b, rid), (_, task) in self.tasks.items() if b == backend and not task.done()}

    async def _read(self, dep: DeploymentEvents, rid: str, endpoint: str) -> None:
        sock = self.socket(endpoint)
        try:
            while True:
                frames = await sock.recv_multipart()
                if len(frames) < 3:
                    continue
                dep.receive(rid, int.from_bytes(frames[1], "big", signed=True), frames[2])
        except asyncio.CancelledError:
            raise
        except Exception as e:  # a reader that dies is restarted by the next sync
            log.warning("%s replica %s: KV events stopped (%s)", dep.backend, rid, e)
            dep.forget(rid)
        finally:
            close = getattr(sock, "close", None)
            if close is not None:
                close()

    async def _flush_forever(self) -> None:
        while True:
            await asyncio.sleep(FLUSH_SECONDS)
            for dep in list(self.deployments.values()):
                try:
                    await dep.flush()
                except Exception:  # noqa: BLE001 (Redis hiccup: the next flush publishes them)
                    log.exception("could not publish KV-event notes for %s", dep.backend)

    async def close(self) -> None:
        for _, task in self.tasks.values():
            task.cancel()
        if self.flusher is not None:
            self.flusher.cancel()
        self.tasks.clear()
