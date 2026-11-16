"""What the gateway holds for batches (content.py): sealed under a key that can be rotated while uploads wait, in
Postgres or the operator's bucket (S3, Cloud Storage, Azure Blob Storage, buckets.py)."""

import copy
import io
import json
import re
import xml.sax.saxutils
from datetime import UTC, datetime, timedelta
from urllib.parse import parse_qs, unquote, urlparse

import httpx
import openai
import pytest
from conftest import ADMIN_KEY, TEST_KEY, auth

from meritorder.app import build_gateway, create_app
from meritorder.backends.auth import BackendAuthProvider
from meritorder.batches import content as content_mod
from meritorder.batches.buckets import AzureContainer, GCSBucket, S3Bucket, bucket_kind, split_bucket
from meritorder.batches.content import KEYS_SETTING, MAGIC, Content, Keyring, SealError, _frame, _unframe, load_keyring, storage_auth
from meritorder.config import BatchesConfig, BatchStorage, build_config
from meritorder.state import MemoryState
from meritorder.store import MemoryStore

CONFIG = {
    "backends": {"mock-openai": {"kind": "mock", "format": "openai", "mock_latency_ms": 0}},
    "models": {"small": {"quality": {"general": 3}, "routes": [{"backend": "mock-openai", "model": "small-1", "price": {"input": 0.2, "output": 1}}]}},
    "auth": {"static_keys": [{"key": TEST_KEY, "user": "alice", "team": "default", "role": "user"},
                             {"key": ADMIN_KEY, "user": "root", "team": "ops", "role": "admin"}]},
}


class Objects:
    """An object store behind any of the three REST APIs: keeps objects by name, lists them in pages of two."""

    def __init__(self, kind: str):
        self.kind, self.objects, self.requests = kind, {}, []

    def name(self, request: httpx.Request) -> str:
        path = unquote(request.url.raw_path.decode().split("?")[0])
        if self.kind == "s3":
            return path.lstrip("/")
        if self.kind == "azure":
            return path.split("/", 2)[2]  # /<container>/<name>
        return path.rsplit("/o/", 1)[1]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        q = parse_qs(urlparse(str(request.url)).query)
        if request.method == "GET" and ("list-type" in q or "comp" in q or (self.kind == "gs" and request.url.path.endswith("/o"))):
            prefix = q.get("prefix", [""])[0]
            names = sorted(n for n in self.objects if n.startswith(prefix))
            start = int((q.get("continuation-token") or q.get("marker") or q.get("pageToken") or ["0"])[0])
            page, more = names[start:start + 2], start + 2 < len(names)
            if self.kind == "gs":
                return httpx.Response(200, json={"items": [{"name": n} for n in page], **({"nextPageToken": str(start + 2)} if more else {})})
            if self.kind == "azure":
                blobs = "".join(f"<Blob><Name>{xml.sax.saxutils.escape(n)}</Name></Blob>" for n in page)
                nxt = f"<NextMarker>{start + 2}</NextMarker>" if more else "<NextMarker />"
                return httpx.Response(200, content=f'<?xml version="1.0"?><EnumerationResults><Blobs>{blobs}</Blobs>{nxt}</EnumerationResults>'.encode())
            keys = "".join(f"<Contents><Key>{n}</Key></Contents>" for n in page)
            token = f"<NextContinuationToken>{start + 2}</NextContinuationToken>" if more else ""
            return httpx.Response(200, content=f"<ListBucketResult>{keys}{token}</ListBucketResult>".encode())
        if request.method in ("PUT", "POST"):
            name = q["name"][0] if self.kind == "gs" else self.name(request)
            self.objects[name] = request.content
            return httpx.Response(201 if self.kind == "azure" else 200, json={})
        name = self.name(request)
        if request.method == "GET":
            return httpx.Response(200, content=self.objects[name]) if name in self.objects else httpx.Response(404, json={})
        if request.method == "DELETE":
            return httpx.Response(202 if self.kind == "azure" else 204) if self.objects.pop(name, None) is not None else httpx.Response(404)
        return httpx.Response(405)


def bucket(kind: str, uri: str):
    objs = Objects(kind)
    client = httpx.AsyncClient(transport=httpx.MockTransport(objs))
    cls = {"s3": S3Bucket, "gs": GCSBucket, "azure": AzureContainer}[kind]
    return cls(uri, client, BackendAuthProvider(), "batch storage", "eu-west-1"), objs


# ---------------------------------------------------------------- the key and its rotation


def test_a_rotated_key_stays_readable_for_its_grace_period_and_says_so_after():
    old = Keyring("old key")
    chunks = old.seal("file-1", b"waiting upload")
    assert chunks[0].startswith(MAGIC) and old.sealed_with(chunks) == old.id
    now = datetime.now(UTC)
    rotated = Keyring("new key", previous="old key", previous_until=now + timedelta(days=1))
    assert rotated.previous_id == old.id and rotated.id != old.id and rotated.previous_usable()
    assert rotated.open("file-1", chunks) == b"waiting upload"
    assert rotated.sealed_with(rotated.seal("f", b"x")) == rotated.id  # new content: the new key
    late = Keyring("new key", previous="old key", previous_until=now - timedelta(seconds=1))
    with pytest.raises(SealError, match="grace period ended"):
        late.open("file-1", chunks)
    with pytest.raises(SealError, match="MERITORDER_BATCH_KEY_PREVIOUS"):
        Keyring("new key").open("file-1", chunks)
    # Content from before key ids (decision 45: the nonce and ciphertext only) opens with either key.
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    legacy = [b"n" * 12 + AESGCM(content_mod.key_bytes("old key")).encrypt(b"n" * 12, b"from before", b"file-0:0")]
    assert rotated.open("file-0", legacy) == b"from before" and rotated.sealed_with(legacy) is None
    with pytest.raises(SealError, match="was MERITORDER_BATCH_KEY changed"):
        Keyring("other").open("file-0", legacy)
    assert Keyring("same", previous="same").previous is None  # the same key twice isn't a rotation


async def test_the_grace_period_counts_from_when_the_new_key_was_first_used(monkeypatch):
    store = MemoryStore()
    monkeypatch.setenv("MERITORDER_BATCH_KEY", "new key")
    settings = BatchesConfig(retention_days=3)
    ring = await load_keyring(settings, store)
    assert ring.previous is None and await store.get_setting(KEYS_SETTING) is None
    monkeypatch.setenv("MERITORDER_BATCH_KEY_PREVIOUS", "old key")
    ring = await load_keyring(settings, store)
    seen = await store.get_setting(KEYS_SETTING)
    first = datetime.fromisoformat(seen[ring.id])
    assert ring.previous_until == first + timedelta(days=5)  # retention (3) plus two days
    again = await load_keyring(BatchesConfig(retention_days=3, key_grace_days=1), store)
    assert again.previous_until == first + timedelta(days=1)  # every gateway counts from the same moment


async def test_rotation_through_the_api_while_an_upload_waits(tmp_path, monkeypatch):
    monkeypatch.setenv("MERITORDER_BATCH_KEY", "old key")
    gw = build_gateway(build_config(copy.deepcopy(CONFIG), tmp_path), store=MemoryStore(), state=MemoryState())
    c = httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(gateway=gw)), base_url="http://test")
    client = openai.AsyncOpenAI(api_key=TEST_KEY, base_url="http://test/v1", http_client=c, max_retries=0)
    row = {"custom_id": "a", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "small", "messages": [{"role": "user", "content": "x"}]}}
    f = await client.files.create(file=("in.jsonl", io.BytesIO(json.dumps(row).encode())), purpose="batch")
    monkeypatch.setenv("MERITORDER_BATCH_KEY", "new key")
    monkeypatch.setenv("MERITORDER_BATCH_KEY_PREVIOUS", "old key")
    status = (await c.get("/v1/reports/batches", headers=auth(ADMIN_KEY))).json()["storage"]
    assert status["held_in"] == "database" and status["files_held"] == 1 and status["files_under_previous"] == 1 and status["previous_usable"]
    assert "storage" not in (await c.get("/v1/reports/batches", headers=auth())).json()  # admins only
    b = await client.batches.create(input_file_id=f.id, endpoint="/v1/chat/completions", completion_window="24h")
    assert b.status == "in_progress"
    monkeypatch.delenv("MERITORDER_BATCH_KEY_PREVIOUS")
    f2 = await client.files.create(file=("in.jsonl", io.BytesIO(json.dumps(row).encode())), purpose="batch")
    monkeypatch.setenv("MERITORDER_BATCH_KEY", "newer key")  # rotated again, without the previous key set
    r = await c.post("/v1/batches", headers=auth(), json={"input_file_id": f2.id, "endpoint": "/v1/chat/completions"})
    assert r.status_code == 500 and "MERITORDER_BATCH_KEY_PREVIOUS" in r.text


# ---------------------------------------------------------------- buckets


def test_bucket_addresses():
    assert split_bucket("az://acct/cont/a/b") == ("acct/cont", "a/b")
    assert split_bucket("https://acct.blob.core.windows.net/cont/pre/") == ("acct/cont", "pre")
    assert bucket_kind("az://a/b") == bucket_kind("https://a.blob.core.windows.net/c") == "azure"
    assert bucket_kind("gs://x") == "gs" and bucket_kind("s3://x") == "s3" and bucket_kind("/tmp") is None
    assert BatchStorage(bucket="s3://b/p/").bucket == "s3://b/p" and BatchStorage(bucket="").bucket is None
    with pytest.raises(ValueError, match="batches.storage.bucket"):
        BatchStorage(bucket="ftp://nope")


@pytest.mark.parametrize("kind,uri", [("s3", "s3://bkt/mo"), ("gs", "gs://bkt/mo"), ("azure", "https://acct.blob.core.windows.net/cont/mo")])
async def test_each_bucket_holds_lists_and_deletes(kind, uri):
    b, objs = bucket(kind, uri)
    for i in range(3):
        await b.put(b.name("held", f"j/{i}"), f'{{"n": {i}}}\n'.encode(), "application/jsonl")
    assert sorted(await b.list(b.name("held", "j"))) == ["mo/held/j/0", "mo/held/j/1", "mo/held/j/2"]  # in pages
    assert await b.get("mo/held/j/1") == b'{"n": 1}\n' and await b.get("mo/nope") is None
    assert [d async for d in b.lines("mo/held/j/2")] == [{"n": 2}]
    await b.delete("mo/held/j/0")
    await b.delete("mo/held/j/0")  # already gone: fine
    assert len(objs.objects) == 2
    if kind == "azure":
        put = next(r for r in objs.requests if r.method == "PUT")
        assert put.headers["x-ms-blob-type"] == "BlockBlob" and put.headers["x-ms-version"] and put.headers["x-ms-date"]
        assert b.address("x") == "https://acct.blob.core.windows.net/cont/x"
    if kind == "s3":
        assert objs.requests[0].url.host == "bkt.s3.eu-west-1.amazonaws.com" and b.address("x") == "s3://bkt/x"


async def test_content_in_a_bucket_uploads_and_gpu_work(tmp_path, monkeypatch):
    store = MemoryStore()
    cfg = build_config({**copy.deepcopy(CONFIG), "batches": {"storage": {"bucket": "s3://bkt/mo"}}}, tmp_path)
    b, objs = bucket("s3", "s3://bkt/mo")
    content = Content(cfg, store, Keyring("k"), b)
    assert content.where == "s3://bkt/mo"
    f = await content.put_file({"id": "file-1", "user_id": "alice", "team": "default", "purpose": "batch", "kind": "input"}, b"hello" * 10)
    assert f["storage"] == "s3://bkt/mo" and f["sealed_with"] == content.ring.id and await store.batch_file_chunks("file-1") == []
    assert list(objs.objects) == ["mo/files/file-1"] and b"hello" not in objs.objects["mo/files/file-1"]
    assert await content.file(f) == b"hello" * 10
    await content.put("batch_1/bj_1/requests/000000", b"req")
    await content.put("batch_1/bj_1/results/1", b"res")
    assert await content.names("batch_1/bj_1/") == ["batch_1/bj_1/requests/000000", "batch_1/bj_1/results/1"]
    assert await content.get("batch_1/bj_1/results/1") == b"res" and await content.get("batch_1/nope") is None
    await content.delete("batch_1/")
    await content.drop_file(f)
    assert objs.objects == {} and (await store.get_batch_file("file-1"))["deleted_at"]
    # A file held where storage no longer points says what to do.
    moved = Content(cfg, store, Keyring("k"), bucket("gs", "gs://other")[0])
    with pytest.raises(SealError, match="isn't batches.storage.bucket any more"):
        await moved.file(f)
    assert _unframe(_frame([b"a", b"", b"ccc"])) == [b"a", b"", b"ccc"]
    st = await content.status()
    assert st["held_in"] == "s3://bkt/mo" and st["previous_key"] is None


def test_storage_signs_in_with_the_backends_credentials(tmp_path, monkeypatch):
    made = []
    monkeypatch.setattr(content_mod, "make_auth", lambda spec: made.append(spec) or BackendAuthProvider())
    monkeypatch.setattr(content_mod, "_AwsSigner", lambda auth, service, region: ("signer", service, region))
    raw = {**copy.deepcopy(CONFIG), "batches": {"storage": {"bucket": "s3://bkt", "backend": "aws"}}}
    raw["backends"]["aws"] = {"format": "anthropic", "base_url": "https://bedrock.example", "auth": {"type": "aws_sigv4", "region": "us-west-2",
                                                                                                    "profile": "prod"}}
    cfg = build_config(raw, tmp_path)
    auth_, region = storage_auth(cfg, "s3://bkt")
    assert made[-1].profile == "prod" and auth_ == ("signer", "s3", "us-west-2") and region == "us-west-2"
    auth_, _ = storage_auth(cfg, "https://acct.blob.core.windows.net/c")
    assert made[-1].type == "azure_ad" and made[-1].scope == "https://storage.azure.com/.default"
    storage_auth(cfg, "gs://bkt")
    assert made[-1].type == "google"


async def test_uploads_go_to_the_bucket_through_the_api(tmp_path, monkeypatch):
    monkeypatch.setenv("MERITORDER_BATCH_KEY", "k")
    raw = {**copy.deepcopy(CONFIG), "batches": {"storage": {"bucket": "gs://bkt/mo"}}}
    gw = build_gateway(build_config(raw, tmp_path), store=MemoryStore(), state=MemoryState())
    b, objs = bucket("gs", "gs://bkt/mo")
    monkeypatch.setattr(content_mod, "storage_auth", lambda cfg, uri: (BackendAuthProvider(), None))
    real_init = Content.__init__

    def init(self, cfg, store, ring, bucket_=None):
        real_init(self, cfg, store, ring, b)

    monkeypatch.setattr(Content, "__init__", init)
    c = httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(gateway=gw)), base_url="http://test")
    row = {"custom_id": "a", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "small", "messages": [{"role": "user", "content": "x"}]}}
    up = (await c.post("/v1/files", headers=auth(), files={"file": ("in.jsonl", json.dumps(row).encode())}, data={"purpose": "batch"})).json()
    assert list(objs.objects) == ["mo/files/" + up["id"]]
    assert (await c.get(f"/v1/files/{up['id']}/content", headers=auth())).json()["custom_id"] == "a"
    r = await c.post("/v1/batches", headers=auth(), json={"input_file_id": up["id"], "endpoint": "/v1/chat/completions"})
    assert r.status_code == 200 and objs.objects == {}  # handed on: deleted from the bucket
    assert re.match(r"gs://bkt/mo", (await c.get("/v1/reports/batches", headers=auth(ADMIN_KEY))).json()["storage"]["held_in"])
