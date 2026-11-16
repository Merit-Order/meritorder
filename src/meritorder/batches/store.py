"""Where batches live: four tables of metadata and one of encrypted uploads, as mixins of the two stores (store.py).

* ``batches``: one row per client batch: who, which API, its state, request counts, the estimated cost reserved against
  budgets, and what its results cost and saved once settled.
* ``batch_jobs``: one row per provider job a batch was split into: the provider's own ids, its state, when to ask again,
  and a lease, so several workers (and restarts) share the polling without doing a job twice.
* ``batch_lines``: one row per request: the client's ``custom_id``, where it was routed and why, its estimated cost and
  how it ended. Never its content. Results are read from the providers when the client asks for them.
* ``batch_files``, ``batch_file_chunks``: OpenAI-style uploads. The content (AES-GCM, see content.Keyring) is held only until
  its batch is handed to the providers, or ``batches.retention_days`` if no batch uses it; output and error files are
  rows without content, joined from the providers' results on demand. With ``batches.storage.bucket`` the content is in
  the operator's bucket instead, and the row says where (``storage``) and which key sealed it (``sealed_with``).
* ``batch_blobs``: batch work for self-hosted GPUs (gpu.py), sealed: each request until it runs or is handed to a
  provider, each result until the retention period ends. Not used with a storage bucket, which holds them instead.
"""

from __future__ import annotations

import copy
from datetime import UTC, datetime, timedelta
from typing import Any

BATCH_SCHEMA = """
CREATE TABLE IF NOT EXISTS batches (
    id text PRIMARY KEY,
    created_at timestamptz NOT NULL DEFAULT now(),
    user_id text NOT NULL,
    team text NOT NULL,
    key_id text,
    tags jsonb NOT NULL DEFAULT '{}',
    client_format text NOT NULL,
    endpoint text,
    status text NOT NULL,
    outcome text,
    expires_at timestamptz,
    cancel_requested_at timestamptz,
    cancelled_by text,
    finalizing_at timestamptz,
    ended_at timestamptz,
    purged_at timestamptz,
    input_file_id text,
    output_file_id text,
    error_file_id text,
    metadata jsonb NOT NULL DEFAULT '{}',
    request_count int NOT NULL DEFAULT 0,
    counts jsonb NOT NULL DEFAULT '{}',
    providers jsonb NOT NULL DEFAULT '{}',
    estimated_cost double precision NOT NULL DEFAULT 0,
    reserved_cost double precision NOT NULL DEFAULT 0,
    reserve_month text,
    actual_cost double precision NOT NULL DEFAULT 0,
    baseline_cost double precision NOT NULL DEFAULT 0,
    savings_batch double precision NOT NULL DEFAULT 0,
    savings_total double precision NOT NULL DEFAULT 0,
    errors jsonb
);
CREATE INDEX IF NOT EXISTS batches_created ON batches (created_at DESC);
CREATE INDEX IF NOT EXISTS batches_user ON batches (user_id, created_at DESC);
CREATE TABLE IF NOT EXISTS batch_jobs (
    id text PRIMARY KEY,
    batch_id text NOT NULL REFERENCES batches (id) ON DELETE CASCADE,
    backend text NOT NULL,
    api text NOT NULL,
    model text,
    endpoint text,
    provider_id text,
    provider jsonb NOT NULL DEFAULT '{}',
    prices jsonb NOT NULL DEFAULT '{}',
    status text NOT NULL,
    provider_status text,
    counts jsonb NOT NULL DEFAULT '{}',
    request_count int NOT NULL DEFAULT 0,
    estimated_cost double precision NOT NULL DEFAULT 0,
    created_at timestamptz NOT NULL DEFAULT now(),
    ended_at timestamptz,
    next_poll_at timestamptz NOT NULL DEFAULT now(),
    polls int NOT NULL DEFAULT 0,
    unchanged int NOT NULL DEFAULT 0,
    failures int NOT NULL DEFAULT 0,
    cancel_sent boolean NOT NULL DEFAULT false,
    lease_owner text,
    lease_until timestamptz,
    error text,
    purged_at timestamptz
);
CREATE INDEX IF NOT EXISTS batch_jobs_batch ON batch_jobs (batch_id);
CREATE INDEX IF NOT EXISTS batch_jobs_due ON batch_jobs (next_poll_at) WHERE status IN ('submitting', 'running', 'ended');
CREATE TABLE IF NOT EXISTS batch_lines (
    batch_id text NOT NULL REFERENCES batches (id) ON DELETE CASCADE,
    idx int NOT NULL,
    custom_id text NOT NULL,
    job_id text,
    kind text NOT NULL,
    requested_model text,
    model text,
    upstream_model text,
    backend text,
    floor text,
    route_reason text,
    upgraded boolean NOT NULL DEFAULT false,
    estimated_cost double precision NOT NULL DEFAULT 0,
    status text NOT NULL,
    error text,
    error_code int,
    settled boolean NOT NULL DEFAULT false,
    PRIMARY KEY (batch_id, idx)
);
CREATE INDEX IF NOT EXISTS batch_lines_job ON batch_lines (job_id);
CREATE TABLE IF NOT EXISTS batch_files (
    id text PRIMARY KEY,
    created_at timestamptz NOT NULL DEFAULT now(),
    user_id text NOT NULL,
    team text NOT NULL,
    purpose text NOT NULL,
    filename text,
    bytes bigint NOT NULL DEFAULT 0,
    kind text NOT NULL,
    batch_id text,
    expires_at timestamptz,
    deleted_at timestamptz
);
CREATE INDEX IF NOT EXISTS batch_files_user ON batch_files (user_id, created_at DESC);
CREATE TABLE IF NOT EXISTS batch_file_chunks (
    file_id text NOT NULL REFERENCES batch_files (id) ON DELETE CASCADE,
    seq int NOT NULL,
    data bytea NOT NULL,
    PRIMARY KEY (file_id, seq)
);
"""
# Schema migration 13 (decision 49): uploads in the operator's bucket and the key that sealed them; batch work on GPUs
# (held sealed in batch_blobs; each line's tries there).
BATCH_SCHEMA_13 = """
ALTER TABLE batches ADD COLUMN IF NOT EXISTS savings_self_hosting double precision NOT NULL DEFAULT 0;
ALTER TABLE batch_files ADD COLUMN IF NOT EXISTS storage text;
ALTER TABLE batch_files ADD COLUMN IF NOT EXISTS sealed_with text;
ALTER TABLE batch_lines ADD COLUMN IF NOT EXISTS tries int NOT NULL DEFAULT 0;
CREATE TABLE IF NOT EXISTS batch_blobs (
    name text NOT NULL,
    seq int NOT NULL,
    data bytea NOT NULL,
    PRIMARY KEY (name, seq)
);
"""

BATCH_FIELDS = ["id", "created_at", "user_id", "team", "key_id", "tags", "client_format", "endpoint", "status", "outcome", "expires_at",
                "cancel_requested_at", "cancelled_by", "finalizing_at", "ended_at", "purged_at", "input_file_id", "output_file_id",
                "error_file_id", "metadata", "request_count", "counts", "providers", "estimated_cost", "reserved_cost", "reserve_month",
                "actual_cost", "baseline_cost", "savings_batch", "savings_total", "errors", "savings_self_hosting"]
JOB_FIELDS = ["id", "batch_id", "backend", "api", "model", "endpoint", "provider_id", "provider", "prices", "status", "provider_status", "counts",
              "request_count", "estimated_cost", "created_at", "ended_at", "next_poll_at", "polls", "unchanged", "failures", "cancel_sent",
              "lease_owner", "lease_until", "error", "purged_at"]
LINE_FIELDS = ["batch_id", "idx", "custom_id", "job_id", "kind", "requested_model", "model", "upstream_model", "backend", "floor",
               "route_reason", "upgraded", "estimated_cost", "status", "error", "error_code", "settled", "tries"]
FILE_FIELDS = ["id", "created_at", "user_id", "team", "purpose", "filename", "bytes", "kind", "batch_id", "expires_at", "deleted_at",
               "storage", "sealed_with"]
OPEN_JOBS = ("submitting", "running", "ended")  # a worker still has something to do for these
_DEFAULTS = {"tags": {}, "metadata": {}, "counts": {}, "providers": {}, "provider": {}, "prices": {}, "polls": 0, "unchanged": 0,
             "failures": 0, "cancel_sent": False, "upgraded": False, "settled": False, "request_count": 0, "bytes": 0, "tries": 0,
             **dict.fromkeys(("estimated_cost", "reserved_cost", "actual_cost", "baseline_cost", "savings_batch", "savings_total",
                              "savings_self_hosting"), 0.0)}


def _complete(row: dict, fields: list[str]) -> dict:
    now = datetime.now(UTC)
    out = {f: row.get(f, copy.deepcopy(_DEFAULTS.get(f))) for f in fields}
    for f in ("created_at", "next_poll_at"):
        if f in out and out[f] is None:
            out[f] = now
    return out


def _totals(rows: list[dict]) -> dict[str, float]:
    """What settled ledger rows add to their batch's totals."""
    return {"actual_cost": sum(r.get("actual_cost") or 0 for r in rows), "baseline_cost": sum(r.get("baseline_cost") or 0 for r in rows),
            "savings_batch": sum(r.get("savings_batch") or 0 for r in rows),
            "savings_self_hosting": sum(r.get("savings_self_hosting") or 0 for r in rows),
            "savings_total": sum(sum(r.get(k) or 0 for k in r if k.startswith("savings_")) for r in rows)}


class MemoryBatches:
    """The in-memory store's batches (development and tests)."""

    def __init__(self) -> None:
        self._batches: dict[str, dict] = {}
        self._jobs: dict[str, dict] = {}
        self._lines: dict[str, dict[int, dict]] = {}
        self._files: dict[str, dict] = {}
        self._chunks: dict[str, list[bytes]] = {}
        self._blobs: dict[str, list[bytes]] = {}

    async def create_batch(self, batch: dict, lines: list[dict], jobs: list[dict]) -> dict:
        b = _complete(batch, BATCH_FIELDS)
        self._batches[b["id"]] = b
        self._lines[b["id"]] = {ln["idx"]: _complete({**ln, "batch_id": b["id"]}, LINE_FIELDS) for ln in lines}
        await self.add_batch_jobs(jobs)
        return copy.deepcopy(b)

    async def get_batch(self, batch_id: str) -> dict | None:
        b = self._batches.get(batch_id)
        return copy.deepcopy(b) if b else None

    async def list_batches(self, user_id: str | None = None, team: str | None = None, limit: int | None = 50, after: str | None = None,
                           start: datetime | None = None, end: datetime | None = None, client_format: str | None = None) -> list[dict]:
        """Newest first. ``after``: the id of the last batch of the previous page (it and everything newer are left out). A
        cursor that names no batch (deleted since) gives an empty page, as in Postgres, not the first page again."""
        rows = sorted(self._batches.values(), key=lambda b: (b["created_at"], b["id"]), reverse=True)
        if after:
            cursor = self._batches.get(after)
            if cursor is None:
                return []
            rows = [b for b in rows if (b["created_at"], b["id"]) < (cursor["created_at"], cursor["id"])]
        rows = [b for b in rows if (user_id is None or b["user_id"] == user_id) and (team is None or b["team"] == team)
                and (start is None or b["created_at"] >= start) and (end is None or b["created_at"] < end)
                and (client_format is None or b["client_format"] == client_format)]
        return copy.deepcopy(rows[:limit] if limit else rows)

    async def update_batch(self, batch_id: str, **fields) -> dict | None:
        b = self._batches.get(batch_id)
        if b is None:
            return None
        b.update({k: v for k, v in fields.items() if k in BATCH_FIELDS})
        return copy.deepcopy(b)

    async def add_batch_jobs(self, jobs: list[dict]) -> None:
        for j in jobs:
            self._jobs[j["id"]] = _complete(j, JOB_FIELDS)

    async def batch_jobs(self, batch_id: str) -> list[dict]:
        return copy.deepcopy(sorted((j for j in self._jobs.values() if j["batch_id"] == batch_id), key=lambda j: (j["created_at"], j["id"])))

    async def get_batch_job(self, job_id: str) -> dict | None:
        j = self._jobs.get(job_id)
        return copy.deepcopy(j) if j else None

    async def update_batch_job(self, job_id: str, owner: str | None = None, release: bool = False, **fields) -> dict | None:
        """``owner``: change it only while this worker holds its lease. ``release``: give the lease back."""
        j = self._jobs.get(job_id)
        if j is None or (owner is not None and j.get("lease_owner") not in (owner, None)):
            return None
        j.update({k: v for k, v in fields.items() if k in JOB_FIELDS})
        if release:
            j["lease_owner"] = j["lease_until"] = None
        return copy.deepcopy(j)

    async def claim_batch_jobs(self, owner: str, limit: int = 20, lease_seconds: float = 900) -> list[dict]:
        now = datetime.now(UTC)
        due = sorted((j for j in self._jobs.values() if j["status"] in OPEN_JOBS and j["next_poll_at"] <= now
                      and (j["lease_until"] is None or j["lease_until"] < now)), key=lambda j: j["next_poll_at"])[:limit]
        for j in due:
            j["lease_owner"], j["lease_until"] = owner, now + timedelta(seconds=lease_seconds)
        return copy.deepcopy(due)

    async def batch_lines(self, batch_id: str, job_id: str | None = None) -> list[dict]:
        lines = self._lines.get(batch_id, {})
        return copy.deepcopy([ln for _, ln in sorted(lines.items()) if job_id is None or ln["job_id"] == job_id])

    async def update_batch_lines(self, batch_id: str, updates: list[dict]) -> None:
        lines = self._lines.get(batch_id, {})
        for u in updates:
            if u["idx"] in lines:
                lines[u["idx"]].update({k: v for k, v in u.items() if k in LINE_FIELDS and k not in ("batch_id", "idx")})

    async def settle_batch_lines(self, batch_id: str, settled: list[dict]) -> tuple[float, float, list[dict]]:
        """Mark lines settled with how they ended, write their ledger rows, and move their estimates off the batch's
        reservation, once per line however often it's called. Returns (estimates released, actual cost added, the ledger
        rows this call wrote: whoever settles a line counts it once in its metrics)."""
        from ..store import complete_row

        lines = self._lines.get(batch_id, {})
        fresh = [s for s in settled if s["idx"] in lines and not lines[s["idx"]]["settled"]]
        released = 0.0
        rows = []
        for s in fresh:
            ln = lines[s["idx"]]
            ln.update(settled=True, status=s["status"], error=s.get("error"))
            released += ln["estimated_cost"] or 0.0
            if s.get("row"):
                rows.append(complete_row(s["row"]))
        self.ledger.extend(rows)  # type: ignore[attr-defined]  (MemoryStore's)
        t = _totals(rows)
        b = self._batches.get(batch_id)
        if b is not None:
            b["reserved_cost"] = max(0.0, b["reserved_cost"] - released)
            for k, v in t.items():
                b[k] += v
        return released, t["actual_cost"], rows

    async def add_batch_reserved(self, batch_id: str, amount: float) -> None:
        """Change a batch's reservation by ``amount``, whatever settling has released from it meanwhile."""
        b = self._batches.get(batch_id)
        if b is not None:
            b["reserved_cost"] = max(0.0, b["reserved_cost"] + amount)

    async def batch_reserved(self, month: str, field: str | None = None, value: str | None = None) -> float:
        """What running batches have reserved, as it counts in ``month``: everyone's, or one person's or team's (``field``
        user_id or team). A reservation counts in every month it's outstanding: a batch made on October 31 that settles in
        November is charged in November, so November's budgets hold room for it until it does."""
        return sum(b["reserved_cost"] for b in self._batches.values()
                   if b.get("reserve_month") and b["reserve_month"] <= month and (field is None or b.get(field) == value))

    async def batches_to_purge(self, before: datetime) -> list[dict]:
        return copy.deepcopy([b for b in self._batches.values() if b["status"] == "ended" and b["ended_at"] and b["ended_at"] < before
                              and b["purged_at"] is None])

    # --- uploaded files

    async def create_batch_file(self, meta: dict, chunks: list[bytes]) -> dict:
        f = _complete(meta, FILE_FIELDS)
        self._files[f["id"]] = f
        self._chunks[f["id"]] = list(chunks)
        return dict(f)

    async def get_batch_file(self, file_id: str) -> dict | None:
        f = self._files.get(file_id)
        return dict(f) if f else None

    async def batch_file_chunks(self, file_id: str) -> list[bytes]:
        return list(self._chunks.get(file_id, []))

    async def drop_batch_file(self, file_id: str) -> None:
        """Delete the file's content and mark it deleted (its row stays, so the batch that used it can still name it)."""
        self._chunks.pop(file_id, None)
        if file_id in self._files:
            self._files[file_id]["deleted_at"] = self._files[file_id]["deleted_at"] or datetime.now(UTC)

    async def list_batch_files(self, user_id: str, purpose: str | None = None, limit: int = 10000) -> list[dict]:
        rows = sorted((f for f in self._files.values() if f["user_id"] == user_id and f["deleted_at"] is None
                       and (purpose is None or f["purpose"] == purpose)), key=lambda f: f["created_at"], reverse=True)
        return [dict(f) for f in rows[:limit]]

    async def stale_batch_files(self, before: datetime) -> list[dict]:
        return [dict(f) for f in self._files.values() if f["kind"] == "input" and f["deleted_at"] is None and f["created_at"] < before]

    async def held_batch_files(self) -> list[dict]:
        """Uploads whose content is still held (where, and which key sealed them)."""
        return [dict(f) for f in self._files.values() if f["kind"] == "input" and f["deleted_at"] is None]

    # --- held GPU work (content.py)

    async def put_batch_blob(self, name: str, chunks: list[bytes]) -> None:
        self._blobs[name] = list(chunks)

    async def get_batch_blob(self, name: str) -> list[bytes] | None:
        b = self._blobs.get(name)
        return list(b) if b is not None else None

    async def batch_blob_names(self, prefix: str) -> list[str]:
        return sorted(n for n in self._blobs if n.startswith(prefix))

    async def delete_batch_blobs(self, prefix: str) -> None:
        for n in [n for n in self._blobs if n.startswith(prefix)]:
            del self._blobs[n]


def _jsonable(row: dict) -> dict:
    return {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in row.items()}


class PostgresBatches:
    """The Postgres store's batches."""

    pool: Any

    @staticmethod
    def _insert(table: str, fields: list[str]) -> str:
        return f"INSERT INTO {table} ({', '.join(fields)}) VALUES ({', '.join(f'${i + 1}' for i in range(len(fields)))})"

    async def create_batch(self, batch: dict, lines: list[dict], jobs: list[dict]) -> dict:
        b = _complete(batch, BATCH_FIELDS)
        rows = [_complete({**ln, "batch_id": b["id"]}, LINE_FIELDS) for ln in lines]
        async with self.pool.acquire() as c:
            async with c.transaction():
                await c.execute(self._insert("batches", BATCH_FIELDS), *[b[f] for f in BATCH_FIELDS])
                if jobs:
                    await c.executemany(self._insert("batch_jobs", JOB_FIELDS), [[_complete(j, JOB_FIELDS)[f] for f in JOB_FIELDS] for j in jobs])
                if rows:
                    await c.copy_records_to_table("batch_lines", records=[tuple(r[f] for f in LINE_FIELDS) for r in rows], columns=LINE_FIELDS)
        return await self.get_batch(b["id"])

    async def get_batch(self, batch_id: str) -> dict | None:
        async with self.pool.acquire() as c:
            row = await c.fetchrow("SELECT * FROM batches WHERE id=$1", batch_id)
        return dict(row) if row else None

    async def list_batches(self, user_id=None, team=None, limit=50, after=None, start=None, end=None, client_format=None) -> list[dict]:
        where, args = [], []
        for col, v in (("user_id", user_id), ("team", team), ("client_format", client_format)):
            if v is not None:
                args.append(v)
                where.append(f"{col} = ${len(args)}")
        if start is not None:
            args.append(start)
            where.append(f"created_at >= ${len(args)}")
        if end is not None:
            args.append(end)
            where.append(f"created_at < ${len(args)}")
        if after:
            args.append(after)
            where.append(f"(created_at, id) < (SELECT created_at, id FROM batches WHERE id = ${len(args)})")
        args.append(limit)
        sql = (f"SELECT * FROM batches {'WHERE ' + ' AND '.join(where) if where else ''} "
               f"ORDER BY created_at DESC, id DESC LIMIT ${len(args)}")
        async with self.pool.acquire() as c:
            return [dict(r) for r in await c.fetch(sql, *args)]

    async def update_batch(self, batch_id: str, **fields) -> dict | None:
        fields = {k: v for k, v in fields.items() if k in BATCH_FIELDS and k != "id"}
        if not fields:
            return await self.get_batch(batch_id)
        sets = ", ".join(f"{k}=${i}" for i, k in enumerate(fields, 2))
        async with self.pool.acquire() as c:
            row = await c.fetchrow(f"UPDATE batches SET {sets} WHERE id=$1 RETURNING *", batch_id, *fields.values())
        return dict(row) if row else None

    async def add_batch_jobs(self, jobs: list[dict]) -> None:
        if not jobs:
            return
        async with self.pool.acquire() as c:
            await c.executemany(self._insert("batch_jobs", JOB_FIELDS), [[_complete(j, JOB_FIELDS)[f] for f in JOB_FIELDS] for j in jobs])

    async def batch_jobs(self, batch_id: str) -> list[dict]:
        async with self.pool.acquire() as c:
            return [dict(r) for r in await c.fetch("SELECT * FROM batch_jobs WHERE batch_id=$1 ORDER BY created_at, id", batch_id)]

    async def get_batch_job(self, job_id: str) -> dict | None:
        async with self.pool.acquire() as c:
            row = await c.fetchrow("SELECT * FROM batch_jobs WHERE id=$1", job_id)
        return dict(row) if row else None

    async def update_batch_job(self, job_id: str, owner: str | None = None, release: bool = False, **fields) -> dict | None:
        fields = {k: v for k, v in fields.items() if k in JOB_FIELDS and k != "id"}
        sets = [f"{k}=${i}" for i, k in enumerate(fields, 3)]
        if release:
            sets += ["lease_owner=NULL", "lease_until=NULL"]
        if not sets:
            return await self.get_batch_job(job_id)
        sql = f"UPDATE batch_jobs SET {', '.join(sets)} WHERE id=$1 AND ($2::text IS NULL OR lease_owner IS NULL OR lease_owner=$2) RETURNING *"
        async with self.pool.acquire() as c:
            row = await c.fetchrow(sql, job_id, owner, *fields.values())
        return dict(row) if row else None

    async def claim_batch_jobs(self, owner: str, limit: int = 20, lease_seconds: float = 900) -> list[dict]:
        sql = """UPDATE batch_jobs SET lease_owner=$1, lease_until=now() + make_interval(secs => $3) WHERE id IN (
                     SELECT id FROM batch_jobs WHERE status = ANY($4::text[]) AND next_poll_at <= now()
                     AND (lease_until IS NULL OR lease_until < now()) ORDER BY next_poll_at LIMIT $2 FOR UPDATE SKIP LOCKED)
                 RETURNING *"""
        async with self.pool.acquire() as c:
            return [dict(r) for r in await c.fetch(sql, owner, limit, float(lease_seconds), list(OPEN_JOBS))]

    async def batch_lines(self, batch_id: str, job_id: str | None = None) -> list[dict]:
        async with self.pool.acquire() as c:
            rows = await c.fetch("SELECT * FROM batch_lines WHERE batch_id=$1 AND ($2::text IS NULL OR job_id=$2) ORDER BY idx", batch_id, job_id)
        return [dict(r) for r in rows]

    async def update_batch_lines(self, batch_id: str, updates: list[dict]) -> None:
        groups: dict[tuple[str, ...], list[dict]] = {}
        for u in updates:
            keys = tuple(sorted(k for k in u if k in LINE_FIELDS and k not in ("batch_id", "idx")))
            if keys:
                groups.setdefault(keys, []).append(u)
        async with self.pool.acquire() as c:
            async with c.transaction():
                for keys, rows in groups.items():
                    sets = ", ".join(f"{k}=${i}" for i, k in enumerate(keys, 3))
                    await c.executemany(f"UPDATE batch_lines SET {sets} WHERE batch_id=$1 AND idx=$2",
                                        [[batch_id, r["idx"], *[r[k] for k in keys]] for r in rows])

    async def settle_batch_lines(self, batch_id: str, settled: list[dict]) -> tuple[float, float, list[dict]]:
        from ..store import LEDGER_FIELDS, complete_row

        by_idx = {s["idx"]: s for s in settled}
        async with self.pool.acquire() as c:
            async with c.transaction():
                fresh = await c.fetch("UPDATE batch_lines SET settled=true WHERE batch_id=$1 AND idx = ANY($2::int[]) AND NOT settled "
                                      "RETURNING idx, estimated_cost", batch_id, list(by_idx))
                if not fresh:
                    return 0.0, 0.0, []
                await c.executemany("UPDATE batch_lines SET status=$3, error=$4 WHERE batch_id=$1 AND idx=$2",
                                    [[batch_id, r["idx"], by_idx[r["idx"]]["status"], by_idx[r["idx"]].get("error")] for r in fresh])
                rows = [complete_row(by_idx[r["idx"]]["row"]) for r in fresh if by_idx[r["idx"]].get("row")]
                if rows:
                    await c.executemany(self._insert("requests", LEDGER_FIELDS), [[r[f] for f in LEDGER_FIELDS] for r in rows])
                released = sum(float(r["estimated_cost"] or 0) for r in fresh)
                t = _totals(rows)
                await c.execute("UPDATE batches SET reserved_cost=greatest(reserved_cost - $2, 0), actual_cost=actual_cost + $3, "
                                "baseline_cost=baseline_cost + $4, savings_batch=savings_batch + $5, savings_total=savings_total + $6, "
                                "savings_self_hosting=savings_self_hosting + $7 WHERE id=$1",
                                batch_id, released, t["actual_cost"], t["baseline_cost"], t["savings_batch"], t["savings_total"],
                                t["savings_self_hosting"])
        return released, t["actual_cost"], rows

    async def add_batch_reserved(self, batch_id: str, amount: float) -> None:
        async with self.pool.acquire() as c:
            await c.execute("UPDATE batches SET reserved_cost=greatest(reserved_cost + $2, 0) WHERE id=$1", batch_id, amount)

    async def batch_reserved(self, month: str, field: str | None = None, value: str | None = None) -> float:
        if field not in (None, "user_id", "team"):
            raise ValueError(field)
        cond = f" AND {field} = $2" if field else ""
        async with self.pool.acquire() as c:
            return float(await c.fetchval(f"SELECT coalesce(sum(reserved_cost), 0) FROM batches WHERE reserve_month <= $1 AND reserved_cost > 0{cond}",
                                          month, *([value] if field else [])))

    async def batches_to_purge(self, before: datetime) -> list[dict]:
        async with self.pool.acquire() as c:
            return [dict(r) for r in await c.fetch("SELECT * FROM batches WHERE status='ended' AND ended_at < $1 AND purged_at IS NULL", before)]

    async def create_batch_file(self, meta: dict, chunks: list[bytes]) -> dict:
        f = _complete(meta, FILE_FIELDS)
        async with self.pool.acquire() as c:
            async with c.transaction():
                await c.execute(self._insert("batch_files", FILE_FIELDS), *[f[k] for k in FILE_FIELDS])
                if chunks:
                    await c.executemany("INSERT INTO batch_file_chunks (file_id, seq, data) VALUES ($1, $2, $3)",
                                        [(f["id"], i, chunk) for i, chunk in enumerate(chunks)])
        return f

    async def get_batch_file(self, file_id: str) -> dict | None:
        async with self.pool.acquire() as c:
            row = await c.fetchrow("SELECT * FROM batch_files WHERE id=$1", file_id)
        return dict(row) if row else None

    async def batch_file_chunks(self, file_id: str) -> list[bytes]:
        async with self.pool.acquire() as c:
            return [bytes(r["data"]) for r in await c.fetch("SELECT data FROM batch_file_chunks WHERE file_id=$1 ORDER BY seq", file_id)]

    async def drop_batch_file(self, file_id: str) -> None:
        async with self.pool.acquire() as c:
            async with c.transaction():
                await c.execute("DELETE FROM batch_file_chunks WHERE file_id=$1", file_id)
                await c.execute("UPDATE batch_files SET deleted_at=coalesce(deleted_at, now()) WHERE id=$1", file_id)

    async def list_batch_files(self, user_id: str, purpose: str | None = None, limit: int = 10000) -> list[dict]:
        async with self.pool.acquire() as c:
            rows = await c.fetch("SELECT * FROM batch_files WHERE user_id=$1 AND deleted_at IS NULL AND ($2::text IS NULL OR purpose=$2) "
                                 "ORDER BY created_at DESC LIMIT $3", user_id, purpose, limit)
        return [dict(r) for r in rows]

    async def stale_batch_files(self, before: datetime) -> list[dict]:
        async with self.pool.acquire() as c:
            return [dict(r) for r in await c.fetch("SELECT * FROM batch_files WHERE kind='input' AND deleted_at IS NULL AND created_at < $1",
                                                   before)]

    async def held_batch_files(self) -> list[dict]:
        async with self.pool.acquire() as c:
            return [dict(r) for r in await c.fetch("SELECT * FROM batch_files WHERE kind='input' AND deleted_at IS NULL")]

    async def put_batch_blob(self, name: str, chunks: list[bytes]) -> None:
        async with self.pool.acquire() as c:
            async with c.transaction():
                await c.execute("DELETE FROM batch_blobs WHERE name=$1", name)
                await c.executemany("INSERT INTO batch_blobs (name, seq, data) VALUES ($1, $2, $3)", [(name, i, x) for i, x in enumerate(chunks)])

    async def get_batch_blob(self, name: str) -> list[bytes] | None:
        async with self.pool.acquire() as c:
            rows = await c.fetch("SELECT data FROM batch_blobs WHERE name=$1 ORDER BY seq", name)
        return [bytes(r["data"]) for r in rows] if rows else None

    async def batch_blob_names(self, prefix: str) -> list[str]:
        async with self.pool.acquire() as c:
            rows = await c.fetch("SELECT DISTINCT name FROM batch_blobs WHERE left(name, length($1)) = $1 ORDER BY name", prefix)
        return [r["name"] for r in rows]

    async def delete_batch_blobs(self, prefix: str) -> None:
        async with self.pool.acquire() as c:
            await c.execute("DELETE FROM batch_blobs WHERE left(name, length($1)) = $1", prefix)
