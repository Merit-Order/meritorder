"""What the gateway holds for batches, and how: sealed with AES-GCM under ``MERITORDER_BATCH_KEY``, in Postgres or, with
``batches.storage.bucket``, in the operator's own bucket (S3, Cloud Storage or Azure Blob Storage, buckets.py).

Two things are held, and only for as long as they're needed:

* An OpenAI-style upload (the JSONL), from ``POST /v1/files`` until its batch is handed on, or ``batches.retention_days``
  if no batch uses it. In Postgres its pieces are rows of ``batch_file_chunks``; in a bucket, one object.
* Batch work on a self-hosted GPU (gpu.py): each request until it runs there (or is handed to a provider's batch API), and
  its result until ``batches.retention_days`` after the batch ends. There's no provider to keep these, so the gateway does.
  In Postgres they're rows of ``batch_blobs``; in a bucket, objects under ``held/``.

Every piece names the key that sealed it (``Keyring``), so the key can be rotated while uploads wait: set the new key in
``MERITORDER_BATCH_KEY`` and the old one in ``MERITORDER_BATCH_KEY_PREVIOUS``. New content is sealed with the new key; what
the old one sealed stays readable for ``batches.key_grace_days`` after a gateway first saw the new key (by default the
retention period plus two days, the longest anything sealed before then is kept). Content from before key ids existed
(decision 45's) has no id and is tried with each key.
"""

from __future__ import annotations

import hashlib
import logging
import os
import secrets
import struct
from datetime import UTC, datetime, timedelta

import httpx

from ..backends.auth import BackendAuthProvider, make_auth
from ..config import BackendAuth, BatchesConfig, Config
from .buckets import BUCKETS, Bucket, _AwsSigner, bucket_kind

log = logging.getLogger("meritorder.batches")

MB = 1024 * 1024
CHUNK = 4 * MB  # content is sealed in pieces this big
MAGIC = b"MO\x02"  # a piece that names its key: MAGIC, the key's id (8 bytes), the nonce (12), the ciphertext
KEYS_SETTING = "batches:keys_seen"  # key id -> when a gateway first sealed with it (where a rotation's grace period starts)


class SealError(Exception):
    """Content that can't be opened with any key this gateway has (or may still use)."""


def key_bytes(secret: str) -> bytes:
    return hashlib.sha256(secret.encode()).digest()


def key_id(key: bytes) -> bytes:
    """A key's public name: 8 bytes of a hash of it, so a piece says which key sealed it without saying anything about it."""
    return hashlib.sha256(b"meritorder-batch-key-id:" + key).digest()[:8]


class Keyring:
    """The batch key, and while a rotation's grace period lasts, the previous one (for opening only). An in-memory store,
    which keeps nothing past the process anyway, gets a random key when none is set."""

    _process_key = secrets.token_bytes(32)

    def __init__(self, secret: str | None, memory: bool = False, previous: str | None = None, previous_until: datetime | None = None):
        self.key = key_bytes(secret) if secret else (self._process_key if memory else None)
        self.previous = key_bytes(previous) if previous and previous != secret else None
        self.previous_until = previous_until

    @property
    def id(self) -> str | None:
        return key_id(self.key).hex() if self.key else None

    @property
    def previous_id(self) -> str | None:
        return key_id(self.previous).hex() if self.previous else None

    def previous_usable(self, now: datetime | None = None) -> bool:
        return self.previous is not None and (self.previous_until is None or (now or datetime.now(UTC)) < self.previous_until)

    def seal(self, name: str, data: bytes) -> list[bytes]:
        """``data`` in pieces of 4 MB, each sealed for this name and its place (moving a piece elsewhere breaks it)."""
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        if self.key is None:
            raise SealError("no batch key is set (MERITORDER_BATCH_KEY)")
        kid, aes, out = key_id(self.key), AESGCM(self.key), []
        for seq, i in enumerate(range(0, max(len(data), 1), CHUNK)):
            nonce = os.urandom(12)
            out.append(MAGIC + kid + nonce + aes.encrypt(nonce, data[i:i + CHUNK], f"{name}:{seq}".encode()))
        return out

    def sealed_with(self, chunks: list[bytes]) -> str | None:
        """The id of the key that sealed this content (None for content from before key ids)."""
        first = chunks[0] if chunks else b""
        return first[3:11].hex() if first.startswith(MAGIC) and len(first) > 23 else None

    def open(self, name: str, chunks: list[bytes]) -> bytes:
        from cryptography.exceptions import InvalidTag
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        keys = [k for k in (self.key, self.previous if self.previous_usable() else None) if k]
        by_id = {key_id(k): k for k in keys}
        out = []
        for seq, c in enumerate(chunks):
            aad = f"{name}:{seq}".encode()
            tries: list[tuple[bytes, bytes]] = []
            if c.startswith(MAGIC) and len(c) > 23 and c[3:11] in by_id:
                tries.append((by_id[c[3:11]], c[11:]))
            tries += [(k, c) for k in keys]  # content from before key ids: nonce and ciphertext only
            for k, body in tries:
                try:
                    out.append(AESGCM(k).decrypt(body[:12], body[12:], aad))
                    break
                except (InvalidTag, ValueError):
                    continue
            else:
                raise SealError(self._why(c))
        return b"".join(out)

    def _why(self, chunk: bytes) -> str:
        kid = chunk[3:11] if chunk.startswith(MAGIC) and len(chunk) > 23 else None
        if kid is not None and self.previous is not None and kid == key_id(self.previous):
            until = self.previous_until.strftime("%Y-%m-%d %H:%M UTC") if self.previous_until else "?"
            return (f"it was sealed with the previous batch key, whose grace period ended {until} (batches.key_grace_days); "
                    "it can't be read any more")
        if kid is not None:
            return (f"it was sealed with a batch key this gateway doesn't have (key {kid.hex()}): was MERITORDER_BATCH_KEY changed "
                    "without setting the old one in MERITORDER_BATCH_KEY_PREVIOUS?")
        return "it can't be read with the gateway's batch key (was MERITORDER_BATCH_KEY changed?)"


async def load_keyring(settings: BatchesConfig, store) -> Keyring:
    """The keyring, with the previous key's grace period counted from when a gateway first used the current key (recorded
    in the store the first time, so every gateway and worker counts from the same moment)."""
    memory = store.kind == "memory"
    current, previous = settings.key(), settings.previous_key()
    ring = Keyring(current, memory, previous)
    if ring.previous is not None and ring.id:
        seen = dict(await store.get_setting(KEYS_SETTING) or {})
        if ring.id not in seen:
            seen[ring.id] = datetime.now(UTC).isoformat()
            await store.set_setting(KEYS_SETTING, seen)
        ring.previous_until = datetime.fromisoformat(seen[ring.id]) + timedelta(days=settings.grace_days())
    return ring


# ---------------------------------------------------------------- where it's kept


def _frame(chunks: list[bytes]) -> bytes:
    """Sealed pieces as one object: each one's length (4 bytes), then the piece."""
    return b"".join(struct.pack(">I", len(c)) + c for c in chunks)


def _unframe(data: bytes) -> list[bytes]:
    out, i = [], 0
    while i + 4 <= len(data):
        (n,) = struct.unpack(">I", data[i:i + 4])
        out.append(data[i + 4:i + 4 + n])
        i += 4 + n
    return out


def storage_auth(cfg: Config, uri: str) -> tuple[BackendAuthProvider, str | None]:
    """Credentials for the storage bucket: those of the backend ``batches.storage.backend`` names, else of the first
    enabled backend that signs in to that cloud, else that cloud's default credentials. And the bucket's region (S3)."""
    s, kind = cfg.batches.storage, bucket_kind(uri)
    want = {"s3": "aws_sigv4", "gs": "google", "azure": "azure_ad"}[kind or "s3"]
    named = cfg.backend(s.backend) if s.backend else None
    spec = named.auth if named is not None else next((b.auth for b in cfg.backends if b.enabled and b.auth.type == want), None)
    if spec is None or spec.type != want:
        spec = BackendAuth(type=want)  # type: ignore[arg-type]
    if kind == "azure":
        spec = spec.model_copy(update={"scope": "https://storage.azure.com/.default"})
    auth = make_auth(spec)
    region = s.region or spec.region
    if kind == "s3":
        auth = _AwsSigner(auth, "s3", region or "us-east-1")
    return auth, region


class Content:
    """Sealed content: in Postgres (``batch_blobs``; uploads in ``batch_file_chunks``) or in the operator's bucket."""

    def __init__(self, cfg: Config, store, ring: Keyring, bucket: Bucket | None = None):
        self.cfg, self.store, self.ring = cfg, store, ring
        self.bucket = bucket
        self._client: httpx.AsyncClient | None = None
        uri = cfg.batches.storage.bucket
        if bucket is None and uri:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0))
            auth, region = storage_auth(cfg, uri)
            self.bucket = BUCKETS[bucket_kind(uri) or "s3"](uri, self._client, auth, "batch storage", region)

    @property
    def where(self) -> str:
        """``database``, or the bucket's address."""
        return self.bucket.uri if self.bucket is not None else "database"

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()

    # --- uploaded files

    async def put_file(self, meta: dict, data: bytes) -> dict:
        chunks = self.ring.seal(meta["id"], data)
        meta = {**meta, "storage": self.where, "sealed_with": self.ring.id}
        if self.bucket is None:
            return await self.store.create_batch_file(meta, chunks)
        await self.bucket.put(self.bucket.name("files", meta["id"]), _frame(chunks))
        return await self.store.create_batch_file(meta, [])

    async def file(self, f: dict) -> bytes:
        where = f.get("storage") or "database"
        if where == "database":
            chunks = await self.store.batch_file_chunks(f["id"])
        else:
            data = await self._bucket_for(where).get(self._bucket_for(where).name("files", f["id"]))
            chunks = _unframe(data or b"")
        return self.ring.open(f["id"], chunks)

    async def drop_file(self, f: dict) -> None:
        where = f.get("storage") or "database"
        if where != "database":
            b = self._bucket_for(where)
            await b.delete(b.name("files", f["id"]))
        await self.store.drop_batch_file(f["id"])

    def _bucket_for(self, where: str) -> Bucket:
        if self.bucket is not None and self.bucket.uri == where:
            return self.bucket
        raise SealError(f"this file is held in {where}, which isn't batches.storage.bucket any more: set it back to read the file")

    # --- held batch work (GPU requests and results)

    async def put(self, name: str, data: bytes) -> None:
        chunks = self.ring.seal(name, data)
        if self.bucket is None:
            await self.store.put_batch_blob(name, chunks)
        else:
            await self.bucket.put(self.bucket.name("held", name), _frame(chunks))

    async def get(self, name: str) -> bytes | None:
        if self.bucket is None:
            chunks = await self.store.get_batch_blob(name)
        else:
            data = await self.bucket.get(self.bucket.name("held", name))
            chunks = None if data is None else _unframe(data)
        return None if chunks is None else self.ring.open(name, chunks)

    async def names(self, prefix: str) -> list[str]:
        if self.bucket is None:
            return await self.store.batch_blob_names(prefix)
        root = self.bucket.name("held") + "/"
        return sorted(n[len(root):] for n in await self.bucket.list(root + prefix))

    async def delete(self, prefix: str) -> None:
        """Delete everything held under ``prefix`` (a job's folder: its requests and results)."""
        if self.bucket is None:
            await self.store.delete_batch_blobs(prefix)
            return
        for n in await self.bucket.list(self.bucket.name("held", prefix)):
            await self.bucket.delete(n)

    async def status(self) -> dict:
        """Where content is held and the key's rotation, for admins (Reports → Batches, ``meritorder batch storage``)."""
        files = await self.store.held_batch_files()
        ring = self.ring
        return {"held_in": self.where, "key": ring.id, "previous_key": ring.previous_id,
                "previous_readable_until": ring.previous_until.isoformat() if ring.previous_until and ring.previous else None,
                "previous_usable": ring.previous_usable(),
                "files_held": len(files), "files_under_previous": sum(1 for f in files if ring.previous_id and f.get("sealed_with") == ring.previous_id),
                "files_elsewhere": sum(1 for f in files if (f.get("storage") or "database") != self.where)}


class ContentCache:
    """One Content per process, kept while ``batches.storage`` stays the same; its keyring is read again each time (a key
    file may have been rotated)."""

    def __init__(self):
        self.content: Content | None = None
        self.storage = None

    async def get(self, cfg: Config, store) -> Content:
        ring = await load_keyring(cfg.batches, store)
        if self.content is None or self.storage != cfg.batches.storage or self.content.store is not store:
            old, self.content, self.storage = self.content, Content(cfg, store, ring), cfg.batches.storage
            if old is not None:
                await old.close()
        else:
            self.content.ring, self.content.cfg = ring, cfg
        return self.content

    async def close(self) -> None:
        if self.content is not None:
            await self.content.close()
