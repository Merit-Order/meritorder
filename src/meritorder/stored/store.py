"""Where stored responses live: three tables, as mixins of the two stores (store.py).

* ``stored_responses``: one row per response a client created with ``store: true`` in a team's stored mode. Its content
  (the response object as the client got it, and the input items sent with it) is one sealed value (sealing.py); the
  columns in the clear are what routing and the console need: owner, chain, conversation, status, where it ran, sizes.
* ``stored_conversations``: one row per conversation (``/v1/conversations``); its metadata is sealed.
* ``conversation_items``: the conversation's items in order, each sealed on its own, so they can be listed and deleted
  one at a time.

A chain is the responses linked by ``previous_response_id``; ``thread_id`` is its first response, and ``touched_at`` its
last turn, so a chain's retention counts from the last time anyone continued it. ``item_ids`` are the ids of the items in
a response (random, never content), so an ``item_reference`` can find the response that holds the item. ``expires_at`` is
set only on a background response the client asked not to store: it's kept just long enough to be polled. Deleting a row
deletes its content: no copy is kept, and nothing else holds it (the ledger never has content).
"""

from __future__ import annotations

import copy
from datetime import UTC, datetime
from typing import Any

STORED_SCHEMA = """
CREATE TABLE IF NOT EXISTS stored_responses (
    id text PRIMARY KEY,
    user_id text NOT NULL,
    team text NOT NULL,
    key_id text,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    touched_at timestamptz NOT NULL DEFAULT now(),
    thread_id text NOT NULL,
    previous_response_id text,
    conversation_id text,
    conversation_upto bigint,
    session_id text,
    status text NOT NULL,
    background boolean NOT NULL DEFAULT false,
    model text,
    backend text,
    upstream_model text,
    upstream_id text,
    request_id uuid,
    seal_key text NOT NULL,
    content bytea NOT NULL,
    bytes bigint NOT NULL DEFAULT 0,
    input_items int NOT NULL DEFAULT 0,
    output_items int NOT NULL DEFAULT 0,
    cancel_requested boolean NOT NULL DEFAULT false,
    gateway text,
    item_ids text[] NOT NULL DEFAULT '{}',
    expires_at timestamptz
);
CREATE INDEX IF NOT EXISTS stored_responses_user ON stored_responses (user_id);
CREATE INDEX IF NOT EXISTS stored_responses_items ON stored_responses USING gin (item_ids);
CREATE INDEX IF NOT EXISTS stored_responses_expires ON stored_responses (expires_at) WHERE expires_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS stored_responses_thread ON stored_responses (thread_id);
CREATE INDEX IF NOT EXISTS stored_responses_team_touched ON stored_responses (team, touched_at);
CREATE INDEX IF NOT EXISTS stored_responses_conversation ON stored_responses (conversation_id) WHERE conversation_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS stored_responses_running ON stored_responses (updated_at) WHERE status IN ('queued', 'in_progress');
CREATE TABLE IF NOT EXISTS stored_conversations (
    id text PRIMARY KEY,
    user_id text NOT NULL,
    team text NOT NULL,
    key_id text,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    touched_at timestamptz NOT NULL DEFAULT now(),
    session_id text,
    seal_key text NOT NULL,
    content bytea NOT NULL,
    bytes bigint NOT NULL DEFAULT 0,
    items int NOT NULL DEFAULT 0,
    item_bytes bigint NOT NULL DEFAULT 0,
    next_seq bigint NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS stored_conversations_user ON stored_conversations (user_id);
CREATE INDEX IF NOT EXISTS stored_conversations_team_touched ON stored_conversations (team, touched_at);
CREATE TABLE IF NOT EXISTS conversation_items (
    conversation_id text NOT NULL REFERENCES stored_conversations (id) ON DELETE CASCADE,
    seq bigint NOT NULL,
    item_id text NOT NULL,
    response_id text,
    created_at timestamptz NOT NULL DEFAULT now(),
    seal_key text NOT NULL,
    content bytea NOT NULL,
    bytes bigint NOT NULL DEFAULT 0,
    PRIMARY KEY (conversation_id, seq)
);
CREATE UNIQUE INDEX IF NOT EXISTS conversation_items_item ON conversation_items (conversation_id, item_id);
CREATE INDEX IF NOT EXISTS conversation_items_id ON conversation_items (item_id);
"""

RESPONSE_FIELDS = ["id", "user_id", "team", "key_id", "created_at", "updated_at", "touched_at", "thread_id", "previous_response_id",
                   "conversation_id", "conversation_upto", "session_id", "status", "background", "model", "backend", "upstream_model",
                   "upstream_id", "request_id", "seal_key", "content", "bytes", "input_items", "output_items", "cancel_requested",
                   "gateway", "item_ids", "expires_at"]
CONVERSATION_FIELDS = ["id", "user_id", "team", "key_id", "created_at", "updated_at", "touched_at", "session_id", "seal_key", "content",
                       "bytes", "items", "item_bytes", "next_seq"]
ITEM_FIELDS = ["conversation_id", "seq", "item_id", "response_id", "created_at", "seal_key", "content", "bytes"]
RUNNING = ("queued", "in_progress")
_DEFAULTS: dict[str, Any] = {"background": False, "bytes": 0, "input_items": 0, "output_items": 0, "cancel_requested": False, "items": 0,
                             "item_bytes": 0, "next_seq": 1, "item_ids": []}
_CHAIN_GUARD = 1_000_000  # links only point to older responses, so a chain can't loop; this stops a damaged table from looping


def _complete(row: dict, fields: list[str]) -> dict:
    now = datetime.now(UTC)
    out = {f: row.get(f, _DEFAULTS.get(f)) for f in fields}
    for f in ("created_at", "updated_at", "touched_at"):
        if f in out and out[f] is None:
            out[f] = now
    return out


def _empty_size() -> dict:
    return {"responses": 0, "response_bytes": 0, "conversations": 0, "items": 0, "conversation_bytes": 0, "people": 0, "oldest": None,
            "running": 0, "old_key": 0}


class MemoryStored:
    """The in-memory store's stored responses (development and tests)."""

    def __init__(self) -> None:
        self._responses: dict[str, dict] = {}
        self._conversations: dict[str, dict] = {}
        self._items: dict[str, dict[int, dict]] = {}

    async def save_stored_response(self, row: dict) -> dict:
        r = _complete(row, RESPONSE_FIELDS)
        self._responses[r["id"]] = r
        return copy.deepcopy(r)

    async def get_stored_response(self, rid: str) -> dict | None:
        r = self._responses.get(rid)
        return copy.deepcopy(r) if r else None

    async def update_stored_response(self, rid: str, unless_status: tuple[str, ...] = (), **fields) -> dict | None:
        """``unless_status``: leave the row as it is when its status is one of these (a cancelled background response
        isn't marked completed by the turn that was still running)."""
        r = self._responses.get(rid)
        if r is None or r["status"] in unless_status:
            return None
        r.update({k: v for k, v in fields.items() if k in RESPONSE_FIELDS and k != "id"}, updated_at=datetime.now(UTC))
        return copy.deepcopy(r)

    async def stored_chain(self, rid: str) -> list[dict]:
        """The response and the ones before it, newest first, as far back as they're stored."""
        out, seen = [], set()
        while rid and rid in self._responses and rid not in seen and len(out) < _CHAIN_GUARD:
            seen.add(rid)
            out.append(copy.deepcopy(self._responses[rid]))
            rid = self._responses[rid]["previous_response_id"]
        return out

    async def touch_stored_thread(self, thread_id: str, ts: datetime) -> None:
        for r in self._responses.values():
            if r["thread_id"] == thread_id and r["touched_at"] < ts:
                r["touched_at"] = ts

    async def delete_stored_response(self, rid: str) -> dict | None:
        return self._responses.pop(rid, None)

    async def stored_with_items(self, user_id: str, item_ids: list[str]) -> tuple[list[dict], list[dict]]:
        """The person's responses and conversation items that hold any of these item ids (for ``item_reference``)."""
        wanted = set(item_ids)
        responses = [copy.deepcopy(r) for r in self._responses.values() if r["user_id"] == user_id and wanted & set(r["item_ids"])]
        items = [{**copy.deepcopy(i), "conversation": copy.deepcopy(c)} for c in self._conversations.values() if c["user_id"] == user_id
                 for i in self._items.get(c["id"], {}).values() if i["item_id"] in wanted]
        return responses, items

    # --- conversations

    async def create_conversation(self, row: dict, items: list[dict]) -> tuple[dict, list[dict]]:
        c = _complete(row, CONVERSATION_FIELDS)
        self._conversations[c["id"]] = c
        self._items[c["id"]] = {}
        added = await self.add_conversation_items(c["id"], items) if items else []
        return copy.deepcopy(self._conversations[c["id"]]), added

    async def get_conversation(self, cid: str) -> dict | None:
        c = self._conversations.get(cid)
        return copy.deepcopy(c) if c else None

    async def update_conversation(self, cid: str, **fields) -> dict | None:
        c = self._conversations.get(cid)
        if c is None:
            return None
        c.update({k: v for k, v in fields.items() if k in CONVERSATION_FIELDS and k != "id"}, updated_at=datetime.now(UTC))
        return copy.deepcopy(c)

    async def delete_conversation(self, cid: str) -> dict | None:
        self._items.pop(cid, None)
        for r in self._responses.values():
            if r["conversation_id"] == cid:
                r["conversation_id"] = None  # OpenAI keeps a deleted conversation's responses; their items go with it
        return self._conversations.pop(cid, None)

    async def add_conversation_items(self, cid: str, items: list[dict]) -> list[dict]:
        """Append items (seq is given here, in order) and count them on the conversation, whose retention restarts."""
        c = self._conversations.get(cid)
        if c is None:
            return []
        now = datetime.now(UTC)
        out = []
        for item in items:
            row = _complete({**item, "conversation_id": cid, "seq": c["next_seq"], "created_at": now}, ITEM_FIELDS)
            self._items[cid][row["seq"]] = row
            c["next_seq"] += 1
            c["items"] += 1
            c["item_bytes"] += row["bytes"]
            out.append(copy.deepcopy(row))
        c["touched_at"] = c["updated_at"] = now
        return out

    async def conversation_items(self, cid: str, after: int | None = None, limit: int | None = None, desc: bool = False,
                                 upto: int | None = None) -> list[dict]:
        """Items in order (``desc``: newest first), after the item with seq ``after``, up to seq ``upto``."""
        rows = sorted(self._items.get(cid, {}).values(), key=lambda r: r["seq"], reverse=desc)
        rows = [r for r in rows if (upto is None or r["seq"] <= upto)
                and (after is None or (r["seq"] < after if desc else r["seq"] > after))]
        return copy.deepcopy(rows[:limit] if limit else rows)

    async def get_conversation_item(self, cid: str, item_id: str) -> dict | None:
        r = next((r for r in self._items.get(cid, {}).values() if r["item_id"] == item_id), None)
        return copy.deepcopy(r) if r else None

    async def delete_conversation_item(self, cid: str, item_id: str) -> bool:
        items = self._items.get(cid, {})
        seq = next((s for s, r in items.items() if r["item_id"] == item_id), None)
        if seq is None:
            return False
        row = items.pop(seq)
        c = self._conversations[cid]
        c["items"] -= 1
        c["item_bytes"] -= row["bytes"]
        c["updated_at"] = datetime.now(UTC)
        return True

    # --- removal, retention, sizes, rotation

    async def delete_stored_for_user(self, user_id: str) -> list[dict]:
        """Everything a person has stored. Returns the deleted responses (their provider copies are deleted next)."""
        gone = [self._responses.pop(rid) for rid in [k for k, r in self._responses.items() if r["user_id"] == user_id]]
        for cid in [k for k, c in self._conversations.items() if c["user_id"] == user_id]:
            await self.delete_conversation(cid)
        return gone

    async def purge_stored(self, cutoffs: dict[str, datetime], default: datetime) -> tuple[list[dict], int]:
        """Delete chains and conversations whose last turn is before their team's cutoff (``default`` for teams not
        listed). Returns the deleted responses and how many conversations went."""
        now = datetime.now(UTC)
        gone = [self._responses.pop(rid) for rid in [k for k, r in self._responses.items()
                                                     if r["touched_at"] < cutoffs.get(r["team"], default)
                                                     or (r["expires_at"] is not None and r["expires_at"] < now)]]
        convs = [k for k, c in self._conversations.items() if c["touched_at"] < cutoffs.get(c["team"], default)]
        for cid in convs:
            await self.delete_conversation(cid)
        return gone, len(convs)

    async def stored_sizes(self, current_key: str | None) -> list[dict]:
        """Per team: how many responses, conversations and items are stored, their size in bytes (sealed), how many
        people they belong to, the oldest, and how many values a key other than ``current_key`` sealed. Never content."""
        out: dict[str, dict] = {}
        people: dict[str, set] = {}
        for r in self._responses.values():
            s = out.setdefault(r["team"], _empty_size())
            s["responses"] += 1
            s["response_bytes"] += r["bytes"]
            s["running"] += r["status"] in RUNNING
            s["old_key"] += r["seal_key"] != current_key
            s["oldest"] = min(filter(None, (s["oldest"], r["created_at"])))
            people.setdefault(r["team"], set()).add(r["user_id"])
        for c in self._conversations.values():
            s = out.setdefault(c["team"], _empty_size())
            s["conversations"] += 1
            s["items"] += c["items"]
            s["conversation_bytes"] += c["bytes"] + c["item_bytes"]
            s["old_key"] += (c["seal_key"] != current_key) + sum(i["seal_key"] != current_key for i in self._items.get(c["id"], {}).values())
            s["oldest"] = min(filter(None, (s["oldest"], c["created_at"])))
            people.setdefault(c["team"], set()).add(c["user_id"])
        for t, s in out.items():
            s["people"] = len(people.get(t, ()))
        return [{"team": t, **s} for t, s in sorted(out.items())]

    async def stored_to_reseal(self, keys: list[str], limit: int) -> list[dict]:
        """Up to ``limit`` sealed values one of ``keys`` sealed (the old keys still on the ring: a value sealed with a key
        that's gone can't be opened, and left in the answer it would hold its place at the front of every pass):
        {kind, id, (conversation_id, seq), user_id, team, seal_key, content}."""
        out: list[dict] = []
        for r in self._responses.values():
            if r["seal_key"] in keys:
                out.append({"kind": "response", "id": r["id"], "user_id": r["user_id"], "team": r["team"], "seal_key": r["seal_key"],
                            "content": r["content"]})
        for c in self._conversations.values():
            if c["seal_key"] in keys:
                out.append({"kind": "conversation", "id": c["id"], "user_id": c["user_id"], "team": c["team"], "seal_key": c["seal_key"],
                            "content": c["content"]})
            for i in self._items.get(c["id"], {}).values():
                if i["seal_key"] in keys:
                    out.append({"kind": "item", "id": i["item_id"], "conversation_id": c["id"], "seq": i["seq"], "user_id": c["user_id"],
                                "team": c["team"], "seal_key": i["seal_key"], "content": i["content"]})
        return out[:limit]

    async def stored_seal_keys(self) -> set[str]:
        """The ids of every key that sealed something stored."""
        keys = {r["seal_key"] for r in self._responses.values()} | {c["seal_key"] for c in self._conversations.values()}
        keys |= {i["seal_key"] for items in self._items.values() for i in items.values()}
        return {k for k in keys if k}

    async def reseal_stored(self, value: dict, content: bytes, seal_key: str) -> bool:
        """Replace one value's content with the same content sealed again, if nothing changed it meanwhile."""
        if value["kind"] == "response":
            row = self._responses.get(value["id"])
        elif value["kind"] == "conversation":
            row = self._conversations.get(value["id"])
        else:
            row = self._items.get(value["conversation_id"], {}).get(value["seq"])
        if row is None or row["seal_key"] != value["seal_key"] or row["content"] != value["content"]:
            return False
        diff = len(content) - len(row["content"])
        row.update(content=content, seal_key=seal_key, bytes=len(content))
        if value["kind"] == "item":
            self._conversations[value["conversation_id"]]["item_bytes"] += diff
        return True

    async def stale_background(self, before: datetime) -> list[dict]:
        """Mark failed the background responses still running that no gateway has touched since ``before``."""
        out = []
        for r in self._responses.values():
            if r["background"] and r["status"] in RUNNING and r["updated_at"] < before:
                r.update(status="failed", updated_at=datetime.now(UTC))
                out.append(copy.deepcopy(r))
        return out


def _insert(table: str, fields: list[str]) -> str:
    return f"INSERT INTO {table} ({', '.join(fields)}) VALUES ({', '.join(f'${i + 1}' for i in range(len(fields)))})"


class PostgresStored:
    """The Postgres store's stored responses."""

    pool: Any

    async def save_stored_response(self, row: dict) -> dict:
        r = _complete(row, RESPONSE_FIELDS)
        sets = ", ".join(f"{f}=EXCLUDED.{f}" for f in RESPONSE_FIELDS if f not in ("id", "created_at"))
        async with self.pool.acquire() as c:
            out = await c.fetchrow(_insert("stored_responses", RESPONSE_FIELDS) + f" ON CONFLICT (id) DO UPDATE SET {sets} RETURNING *",
                                   *[r[f] for f in RESPONSE_FIELDS])
        return dict(out)

    async def get_stored_response(self, rid: str) -> dict | None:
        async with self.pool.acquire() as c:
            row = await c.fetchrow("SELECT * FROM stored_responses WHERE id=$1", rid)
        return dict(row) if row else None

    async def update_stored_response(self, rid: str, unless_status: tuple[str, ...] = (), **fields) -> dict | None:
        fields = {k: v for k, v in fields.items() if k in RESPONSE_FIELDS and k != "id"}
        sets = ", ".join([f"{k}=${i}" for i, k in enumerate(fields, 3)] + ["updated_at=now()"])
        async with self.pool.acquire() as c:
            row = await c.fetchrow(f"UPDATE stored_responses SET {sets} WHERE id=$1 AND NOT (status = ANY($2::text[])) RETURNING *",
                                   rid, list(unless_status), *fields.values())
        return dict(row) if row else None

    async def stored_chain(self, rid: str) -> list[dict]:
        sql = f"""WITH RECURSIVE chain AS (
                      SELECT r.*, 0 AS depth FROM stored_responses r WHERE r.id = $1
                      UNION ALL
                      SELECT r.*, c.depth + 1 FROM stored_responses r JOIN chain c ON r.id = c.previous_response_id
                      WHERE c.depth < {_CHAIN_GUARD})
                  SELECT * FROM chain ORDER BY depth"""
        async with self.pool.acquire() as c:
            rows = await c.fetch(sql, rid)
        return [{k: v for k, v in dict(r).items() if k != "depth"} for r in rows]

    async def touch_stored_thread(self, thread_id: str, ts: datetime) -> None:
        async with self.pool.acquire() as c:
            await c.execute("UPDATE stored_responses SET touched_at=$2 WHERE thread_id=$1 AND touched_at < $2", thread_id, ts)

    async def delete_stored_response(self, rid: str) -> dict | None:
        async with self.pool.acquire() as c:
            row = await c.fetchrow("DELETE FROM stored_responses WHERE id=$1 RETURNING *", rid)
        return dict(row) if row else None

    async def stored_with_items(self, user_id: str, item_ids: list[str]) -> tuple[list[dict], list[dict]]:
        async with self.pool.acquire() as c:
            responses = await c.fetch("SELECT * FROM stored_responses WHERE user_id=$1 AND item_ids && $2::text[]", user_id, item_ids)
            items = await c.fetch("""SELECT i.* FROM conversation_items i JOIN stored_conversations s ON s.id = i.conversation_id
                                     WHERE s.user_id=$1 AND i.item_id = ANY($2::text[])""", user_id, item_ids)
            convs = {r["id"]: dict(r) for r in await c.fetch("SELECT * FROM stored_conversations WHERE id = ANY($1::text[])",
                                                             list({i["conversation_id"] for i in items}))} if items else {}
        return [dict(r) for r in responses], [{**dict(i), "conversation": convs[i["conversation_id"]]} for i in items]

    # --- conversations

    async def create_conversation(self, row: dict, items: list[dict]) -> tuple[dict, list[dict]]:
        conv = _complete(row, CONVERSATION_FIELDS)
        async with self.pool.acquire() as c:
            async with c.transaction():
                await c.execute(_insert("stored_conversations", CONVERSATION_FIELDS), *[conv[f] for f in CONVERSATION_FIELDS])
                added = await self._add_items(c, conv["id"], items) if items else []
                out = await c.fetchrow("SELECT * FROM stored_conversations WHERE id=$1", conv["id"])
        return dict(out), added

    async def get_conversation(self, cid: str) -> dict | None:
        async with self.pool.acquire() as c:
            row = await c.fetchrow("SELECT * FROM stored_conversations WHERE id=$1", cid)
        return dict(row) if row else None

    async def update_conversation(self, cid: str, **fields) -> dict | None:
        fields = {k: v for k, v in fields.items() if k in CONVERSATION_FIELDS and k != "id"}
        sets = ", ".join([f"{k}=${i}" for i, k in enumerate(fields, 2)] + ["updated_at=now()"])
        async with self.pool.acquire() as c:
            row = await c.fetchrow(f"UPDATE stored_conversations SET {sets} WHERE id=$1 RETURNING *", cid, *fields.values())
        return dict(row) if row else None

    async def delete_conversation(self, cid: str) -> dict | None:
        async with self.pool.acquire() as c:
            async with c.transaction():
                await c.execute("UPDATE stored_responses SET conversation_id=NULL WHERE conversation_id=$1", cid)
                row = await c.fetchrow("DELETE FROM stored_conversations WHERE id=$1 RETURNING *", cid)
        return dict(row) if row else None

    async def _add_items(self, c, cid: str, items: list[dict]) -> list[dict]:
        n, size = len(items), sum(int(i.get("bytes") or 0) for i in items)
        end = await c.fetchval("UPDATE stored_conversations SET next_seq=next_seq + $2, items=items + $2, item_bytes=item_bytes + $3, "
                               "touched_at=now(), updated_at=now() WHERE id=$1 RETURNING next_seq", cid, n, size)
        if end is None:
            return []
        rows = [_complete({**item, "conversation_id": cid, "seq": end - n + i}, ITEM_FIELDS) for i, item in enumerate(items)]
        await c.executemany(_insert("conversation_items", ITEM_FIELDS), [[r[f] for f in ITEM_FIELDS] for r in rows])
        return rows

    async def add_conversation_items(self, cid: str, items: list[dict]) -> list[dict]:
        async with self.pool.acquire() as c:
            async with c.transaction():
                return await self._add_items(c, cid, items)

    async def conversation_items(self, cid: str, after: int | None = None, limit: int | None = None, desc: bool = False,
                                 upto: int | None = None) -> list[dict]:
        where, args = ["conversation_id=$1"], [cid]
        if upto is not None:
            args.append(upto)
            where.append(f"seq <= ${len(args)}")
        if after is not None:
            args.append(after)
            where.append(f"seq {'<' if desc else '>'} ${len(args)}")
        sql = f"SELECT * FROM conversation_items WHERE {' AND '.join(where)} ORDER BY seq {'DESC' if desc else 'ASC'}"
        if limit:
            args.append(limit)
            sql += f" LIMIT ${len(args)}"
        async with self.pool.acquire() as c:
            return [dict(r) for r in await c.fetch(sql, *args)]

    async def get_conversation_item(self, cid: str, item_id: str) -> dict | None:
        async with self.pool.acquire() as c:
            row = await c.fetchrow("SELECT * FROM conversation_items WHERE conversation_id=$1 AND item_id=$2", cid, item_id)
        return dict(row) if row else None

    async def delete_conversation_item(self, cid: str, item_id: str) -> bool:
        async with self.pool.acquire() as c:
            async with c.transaction():
                size = await c.fetchval("DELETE FROM conversation_items WHERE conversation_id=$1 AND item_id=$2 RETURNING bytes", cid, item_id)
                if size is None:
                    return False
                await c.execute("UPDATE stored_conversations SET items=items - 1, item_bytes=item_bytes - $2, updated_at=now() WHERE id=$1",
                                cid, size)
        return True

    # --- removal, retention, sizes, rotation

    async def delete_stored_for_user(self, user_id: str) -> list[dict]:
        async with self.pool.acquire() as c:
            async with c.transaction():
                gone = await c.fetch("DELETE FROM stored_responses WHERE user_id=$1 RETURNING *", user_id)
                await c.execute("DELETE FROM stored_conversations WHERE user_id=$1", user_id)
        return [dict(r) for r in gone]

    async def purge_stored(self, cutoffs: dict[str, datetime], default: datetime) -> tuple[list[dict], int]:
        cutoff = ("coalesce((SELECT t.cutoff FROM unnest($1::text[], $2::timestamptz[]) AS t(team, cutoff) WHERE t.team = {table}.team), $3)")
        teams, times = list(cutoffs), [cutoffs[t] for t in cutoffs]
        async with self.pool.acquire() as c:
            async with c.transaction():
                gone = await c.fetch(f"DELETE FROM stored_responses WHERE touched_at < {cutoff.format(table='stored_responses')} "
                                     "OR expires_at < now() RETURNING *", teams, times, default)
                convs = await c.fetch(f"DELETE FROM stored_conversations WHERE touched_at < {cutoff.format(table='stored_conversations')} "
                                      "RETURNING id", teams, times, default)
                if convs:
                    await c.execute("UPDATE stored_responses SET conversation_id=NULL WHERE conversation_id = ANY($1::text[])",
                                    [r["id"] for r in convs])
        return [dict(r) for r in gone], len(convs)

    async def stored_sizes(self, current_key: str | None) -> list[dict]:
        async with self.pool.acquire() as c:
            responses = await c.fetch("""SELECT team, count(*) AS responses, coalesce(sum(bytes), 0) AS response_bytes, min(created_at) AS oldest,
                                                count(*) FILTER (WHERE status IN ('queued', 'in_progress')) AS running,
                                                count(*) FILTER (WHERE seal_key IS DISTINCT FROM $1) AS old_key
                                         FROM stored_responses GROUP BY team""", current_key)
            convs = await c.fetch("""SELECT team, count(*) AS conversations, coalesce(sum(items), 0) AS items,
                                            coalesce(sum(bytes + item_bytes), 0) AS conversation_bytes, min(created_at) AS oldest,
                                            count(*) FILTER (WHERE seal_key IS DISTINCT FROM $1) AS old_key
                                     FROM stored_conversations GROUP BY team""", current_key)
            items_old = await c.fetch("""SELECT s.team, count(*) AS n FROM conversation_items i JOIN stored_conversations s ON s.id = i.conversation_id
                                         WHERE i.seal_key IS DISTINCT FROM $1 GROUP BY s.team""", current_key)
            people = await c.fetch("""SELECT team, count(DISTINCT user_id) AS n FROM (SELECT team, user_id FROM stored_responses
                                      UNION ALL SELECT team, user_id FROM stored_conversations) t GROUP BY team""")
        out: dict[str, dict] = {}
        for r in responses:
            out.setdefault(r["team"], _empty_size()).update(responses=r["responses"], response_bytes=int(r["response_bytes"]),
                                                            oldest=r["oldest"], running=r["running"], old_key=r["old_key"])
        for r in convs:
            s = out.setdefault(r["team"], _empty_size())
            s.update(conversations=r["conversations"], items=int(r["items"]), conversation_bytes=int(r["conversation_bytes"]))
            s["oldest"] = min(filter(None, (s["oldest"], r["oldest"])))
            s["old_key"] += r["old_key"]
        for r in items_old:
            out.setdefault(r["team"], _empty_size())["old_key"] += r["n"]
        for r in people:
            out.setdefault(r["team"], _empty_size())["people"] = r["n"]
        return [{"team": t, **s} for t, s in sorted(out.items())]

    async def stored_to_reseal(self, keys: list[str], limit: int) -> list[dict]:
        async with self.pool.acquire() as c:
            rows = [{"kind": "response", **dict(r)} for r in await c.fetch(
                "SELECT id, user_id, team, seal_key, content FROM stored_responses WHERE seal_key = ANY($1::text[]) LIMIT $2", keys, limit)]
            if len(rows) < limit:
                rows += [{"kind": "conversation", **dict(r)} for r in await c.fetch(
                    "SELECT id, user_id, team, seal_key, content FROM stored_conversations WHERE seal_key = ANY($1::text[]) LIMIT $2",
                    keys, limit - len(rows))]
            if len(rows) < limit:
                rows += [{"kind": "item", **dict(r)} for r in await c.fetch(
                    """SELECT i.item_id AS id, i.conversation_id, i.seq, s.user_id, s.team, i.seal_key, i.content FROM conversation_items i
                       JOIN stored_conversations s ON s.id = i.conversation_id WHERE i.seal_key = ANY($1::text[]) LIMIT $2""",
                    keys, limit - len(rows))]
        return rows

    async def stored_seal_keys(self) -> set[str]:
        async with self.pool.acquire() as c:
            rows = await c.fetch("""SELECT seal_key FROM stored_responses UNION SELECT seal_key FROM stored_conversations
                                    UNION SELECT seal_key FROM conversation_items""")
        return {r["seal_key"] for r in rows if r["seal_key"]}

    async def reseal_stored(self, value: dict, content: bytes, seal_key: str) -> bool:
        async with self.pool.acquire() as c:
            if value["kind"] == "response":
                res = await c.execute("UPDATE stored_responses SET content=$2, seal_key=$3, bytes=$4 WHERE id=$1 AND seal_key=$5 AND content=$6",
                                      value["id"], content, seal_key, len(content), value["seal_key"], value["content"])
            elif value["kind"] == "conversation":
                res = await c.execute("UPDATE stored_conversations SET content=$2, seal_key=$3, bytes=$4 WHERE id=$1 AND seal_key=$5 AND content=$6",
                                      value["id"], content, seal_key, len(content), value["seal_key"], value["content"])
            else:
                async with c.transaction():
                    old = await c.fetchval("SELECT bytes FROM conversation_items WHERE conversation_id=$1 AND seq=$2 AND seal_key=$3 "
                                           "AND content=$4 FOR UPDATE", value["conversation_id"], value["seq"], value["seal_key"], value["content"])
                    if old is None:
                        return False
                    await c.execute("UPDATE conversation_items SET content=$3, seal_key=$4, bytes=$5 WHERE conversation_id=$1 AND seq=$2",
                                    value["conversation_id"], value["seq"], content, seal_key, len(content))
                    await c.execute("UPDATE stored_conversations SET item_bytes=item_bytes + $2 WHERE id=$1", value["conversation_id"],
                                    len(content) - old)
                return True
        return res.endswith(" 1")

    async def stale_background(self, before: datetime) -> list[dict]:
        async with self.pool.acquire() as c:
            rows = await c.fetch("UPDATE stored_responses SET status='failed', updated_at=now() WHERE background "
                                 "AND status IN ('queued', 'in_progress') AND updated_at < $1 RETURNING *", before)
        return [dict(r) for r in rows]
