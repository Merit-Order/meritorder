"""Session identity, prefix hashing, and task-boundary detection.

The gateway keeps only metadata about a session (hashes, timestamps, which
backend holds its cache). Conversation content is never stored.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

_CLAUDE_CODE_SESSION = re.compile(r"session_([0-9a-fA-F-]{8,})")
SESSION_TTL = 6 * 3600  # a session's routing state (and its stickiness) lasts this long after its last turn


def _canon(value: Any) -> bytes:
    # Only hashed: surrogatepass takes half an emoji too, and gives the same bytes as UTF-8 for any other text
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8", "surrogatepass")


def _first_user_text(body: dict) -> str:
    for m in body.get("messages") or []:
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, str):
                return c[:2000]
            return _canon(c)[:2000].decode(errors="replace")
    return ""


def session_key(user_id: str, sid: str) -> str:
    """Where a session's routing state and spend live. Scoped to the user, so another caller sending the
    same session id can't read or change it (or use up its spend cap). Length-prefixed: no collisions."""
    return f"{len(user_id)}:{user_id}:{sid}"


# A client naming its session. Codex CLI sends session-id (session_id in older versions).
SESSION_HEADERS = ("x-session-id", "x-meritorder-session", "x-claude-code-session-id", "session-id", "session_id")


# What a session id from a header may hold. It's the client's text, and it lands in the ledger, the console and CSVs, so
# anything else (quotes, `=`, spaces, control characters) becomes "_". Real clients send UUIDs or similar.
_UNSAFE_SESSION = re.compile(r"[^A-Za-z0-9._:/@+-]")


def safe_id(value: str, limit: int = 128) -> str:
    return _UNSAFE_SESSION.sub("_", value[:limit])


def session_id(headers: dict[str, str], body: dict, fmt: str, user_id: str) -> tuple[str, str]:
    """Returns (session_id, source)."""
    for h in SESSION_HEADERS:
        if headers.get(h):
            return safe_id(headers[h]), "header"
    meta_user = str((body.get("metadata") or {}).get("user_id") or "")
    m = _CLAUDE_CODE_SESSION.search(meta_user)
    if m:
        return m.group(1), "metadata"
    system = body.get("system") if fmt == "anthropic" else [
        x.get("content") for x in body.get("messages") or [] if x.get("role") in ("system", "developer")
    ]
    digest = hashlib.sha256(_canon([user_id, system, _first_user_text(body)])).hexdigest()[:24]
    return f"fp-{digest}", "fingerprint"


def thread_suffix(headers: dict[str, str]) -> str:
    """Which conversation inside a session a request belongs to, for routing state.

    Claude Code sends its subagents' turns and its side requests (titles, summaries, permission checks) under the
    main session's id. Each is a separate conversation with its own history and cache, so each keeps its own model,
    level and cache state; otherwise every subagent turn would reset the main conversation's. Spend limits and the
    ledger still count the whole session.
    """
    suffix = ""
    agent = headers.get("x-claude-code-agent-id")
    if agent:
        suffix += f"/agent:{safe_id(agent, 64)}"
    elif headers.get("x-codex-parent-thread-id") and headers.get("thread-id"):
        suffix += f"/agent:{safe_id(headers['thread-id'], 64)}"  # a Codex CLI subagent: its own thread, under the parent's session
    if (headers.get("x-claude-code-request-class") or "").lower() == "auxiliary":
        suffix += "/aux"
    return suffix


def _messages(body: dict) -> list:
    return body.get("messages") or []


def prefix_hash(body: dict, upto: int) -> str:
    """Hash of everything that forms the cacheable prefix up to message ``upto``.

    One serialization call for the whole prefix: per-message calls cost milliseconds on long agent sessions.
    """
    return hashlib.sha256(_canon([body.get("tools"), body.get("system"), _messages(body)[:upto]])).hexdigest()


def is_task_boundary(body: dict) -> bool:
    """A new human prompt starts a task; tool results continue one."""
    msgs = _messages(body)
    if not msgs:
        return True
    last = msgs[-1]
    if last.get("role") == "tool":
        return False
    if last.get("role") != "user":
        return False
    content = last.get("content")
    if isinstance(content, list) and any(isinstance(b, dict) and b.get("type") == "tool_result" for b in content):
        return False
    return True


@dataclass
class Session:
    id: str
    model: str | None = None
    backend: str | None = None
    floor: str | None = None
    floor_reason: str | None = None  # why the task got its floor: shown for every request of the task, mid-task ones too
    turns: int = 0
    last_seen: float = 0.0
    msg_count: int = 0
    prefix_hash: str | None = None
    prefix_tokens: int = 0
    warm_until: dict[str, float] = field(default_factory=dict)
    # Per GPU deployment: the replica that served the last turn there, when, and its KV-cache counters then
    # (affinity.mark), so a later turn can go back to it and estimate how much of the cache is left.
    placement: dict[str, dict] = field(default_factory=dict)
    effort: str | None = None  # chosen once per session: changing it would invalidate the prompt cache
    thread: str | None = None  # where this conversation's routing state lives (see thread_suffix); None = the session
    # The learned router's record of this task (the route_features row of the request that started it), so what the
    # client does next (sends the same request again, raises X-Route-Floor) can be counted against it.
    learn_row: str | None = None
    # The task's exploration arm (decision 59: {"arm", "from", "level", "p"}), kept for its later turns' ledger tags.
    explore: dict | None = None
    # What the task's tool results have shown so far (outcomes.py): counts and flags, never content.
    outcome: dict | None = None
    # Expected total cost (decision 62): the task's model's chance at its start ("p0"), what was expected to finish it, the
    # thread's spend counters when it started ("spend0", "base0"), and its escalations so far ("escalations", "from",
    # "reasons", "charged").
    task: dict | None = None

    @property
    def state_id(self) -> str:
        return self.thread or self.id

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None, sid: str, thread: str | None = None) -> "Session":
        thread = thread if thread and thread != sid else None
        if not d:
            return cls(id=sid, thread=thread)
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        known["id"], known["thread"] = sid, thread
        return cls(**known)

    def compacted(self, body: dict) -> bool:
        """True when the harness rewrote history (so every cache for it is dead)."""
        if not self.prefix_hash or not self.msg_count:
            return False
        if len(_messages(body)) < self.msg_count:
            return True
        return prefix_hash(body, self.msg_count) != self.prefix_hash

    def warm(self, backend: str, now: float) -> bool:
        return self.warm_until.get(backend, 0.0) > now
