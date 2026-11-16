"""Object stores the operator gives: Amazon S3, Google Cloud Storage and Azure Blob Storage, over their REST APIs with the
credentials the backends use (AWS SigV4, Google sign-in, Entra ID). No SDK beyond what signs the requests.

Used by the batch APIs that read their input from a bucket and write their results there (Bedrock, Vertex AI), and, with
``batches.storage.bucket``, for what the gateway holds for batches (content.py): uploads waiting for their batch, and the
requests and results of batch work on self-hosted GPUs.

Written from each service's REST reference as of November 2026: S3's ListObjectsV2, GetObject, PutObject and
DeleteObject (virtual-hosted addresses); the Cloud Storage JSON API v1 (media uploads, ``alt=media`` downloads); Azure
Blob Storage's Put Blob (BlockBlob), Get Blob, Delete Blob and List Blobs, with ``x-ms-version`` 2025-11-05.
"""

from __future__ import annotations

import asyncio
import email.utils
import json
import re
import xml.etree.ElementTree as ET
from collections.abc import AsyncIterator
from urllib.parse import quote

import httpx

from ..backends.auth import BackendAuthProvider
from ..backends.base import RETRYABLE_STATUS, BackendError
from ..backends.http import _error_message, retry_after

AZURE_VERSION = "2025-11-05"
AZURE_HOST = re.compile(r"^https://([a-z0-9]+)\.blob\.core\.windows\.net/([^/]+)/?(.*)$")


# ---------------------------------------------------------------- HTTP, for the batch APIs and the buckets alike


async def call(client: httpx.AsyncClient, auth: BackendAuthProvider, who: str, method: str, url: str, *, content: bytes = b"",
               headers: dict | None = None, what: str = "") -> httpx.Response:
    """One signed request. A failure is a BackendError saying who, what, and whether trying again may help."""
    try:
        hdrs = await auth.apply(method, url, dict(headers or {}), content)
        r = await client.request(method, url, content=content or None, headers=hdrs)
    except httpx.HTTPError as e:
        raise BackendError(502, f"{who}: {what or 'batch'}: {type(e).__name__}: {e}", retryable=True)
    except BackendError:
        raise
    except Exception as e:  # credentials and the like
        raise BackendError(502, f"{who}: {what or 'batch'}: {type(e).__name__}: {e}", retryable=True)
    if r.status_code >= 400:
        raw = r.content
        try:
            parsed = json.loads(raw)
        except ValueError:
            parsed = None
        raise BackendError(r.status_code, f"{who}: {what or 'batch'}: {_error_message(parsed, raw)}",
                           body=parsed if isinstance(parsed, dict) else None, retryable=r.status_code in RETRYABLE_STATUS,
                           retry_after=retry_after(r.headers))
    return r


async def json_lines(client: httpx.AsyncClient, auth: BackendAuthProvider, who: str, url: str, headers: dict | None = None,
                     what: str = "") -> AsyncIterator[dict]:
    """A JSON-lines download, read line by line: results pass through the gateway without being kept."""
    hdrs = await auth.apply("GET", url, dict(headers or {}), b"")
    try:
        async with client.stream("GET", url, headers=hdrs) as r:
            if r.status_code >= 400:
                raw = await r.aread()
                raise BackendError(r.status_code, f"{who}: {what or 'results'}: {_error_message(None, raw)}",
                                   retryable=r.status_code in RETRYABLE_STATUS)
            async for line in r.aiter_lines():
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if isinstance(d, dict):
                    yield d
    except httpx.HTTPError as e:
        raise BackendError(502, f"{who}: {what or 'results'}: {type(e).__name__}: {e}", retryable=True)


# ---------------------------------------------------------------- the stores


def split_bucket(uri: str) -> tuple[str, str]:
    """("bucket", "prefix") for s3://bucket/prefix or gs://bucket/prefix; for Azure, ("account/container", "prefix")."""
    m = AZURE_HOST.match(uri)
    if m:
        return f"{m.group(1)}/{m.group(2)}", m.group(3).strip("/")
    rest = uri.split("://", 1)[1]
    bucket, _, prefix = rest.partition("/")
    if uri.startswith("az://"):  # az://account/container/prefix
        container, _, prefix = prefix.partition("/")
        bucket = f"{bucket}/{container}"
    return bucket, prefix.strip("/")


class _AwsSigner(BackendAuthProvider):
    """SigV4 for another AWS service (Bedrock's control plane, S3) with a backend's credentials."""

    def __init__(self, source, service: str, region: str):
        self.credentials, self.service, self.region = source.credentials, service, region

    async def apply(self, method, url, headers, body):
        from botocore.auth import S3SigV4Auth
        from botocore.auth import SigV4Auth as BotoSigV4
        from botocore.awsrequest import AWSRequest

        frozen = await asyncio.to_thread(self.credentials.get_frozen_credentials)
        request = AWSRequest(method=method, url=url, data=body, headers=dict(headers))
        (S3SigV4Auth if self.service == "s3" else BotoSigV4)(frozen, self.service, self.region).add_auth(request)
        return dict(request.headers.items())


class Bucket:
    """One bucket (and a prefix in it). ``put``, ``get``, ``list``, ``lines`` and ``delete`` take object names in it."""

    kind = ""

    def __init__(self, uri: str, client: httpx.AsyncClient, auth: BackendAuthProvider, who: str = "bucket", region: str | None = None):
        self.uri, self.client, self.auth, self.who, self.region = uri.rstrip("/"), client, auth, who, region
        self.bucket, self.prefix = split_bucket(self.uri)

    def name(self, *parts: str) -> str:
        """An object name under the bucket's prefix."""
        return "/".join(p.strip("/") for p in (self.prefix, *parts) if p)

    def address(self, name: str) -> str:
        """How the provider names the object (s3://bucket/name, gs://bucket/name)."""
        return f"{self.uri.split('://', 1)[0]}://{self.bucket}/{name}"

    async def _call(self, method: str, url: str, what: str, **kw) -> httpx.Response:
        return await call(self.client, self.auth, self.who, method, url, what=what, **kw)

    async def put(self, name: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        raise NotImplementedError

    async def get(self, name: str) -> bytes | None:
        """The object's content, or None when there's no such object."""
        try:
            r = await self._call("GET", self._get_url(name), "reading an object")
        except BackendError as e:
            if e.status == 404:
                return None
            raise
        return r.content

    async def list(self, prefix: str) -> list[str]:
        raise NotImplementedError

    def lines(self, name: str) -> AsyncIterator[dict]:
        return json_lines(self.client, self.auth, self.who, self._get_url(name), what="reading batch results")

    async def delete(self, name: str) -> None:
        """Delete the object; one already gone is fine."""
        try:
            await self._call("DELETE", self._object_url(name), "deleting an object")
        except BackendError as e:
            if e.status != 404:
                raise

    def _object_url(self, name: str) -> str:
        raise NotImplementedError

    def _get_url(self, name: str) -> str:
        return self._object_url(name)


class S3Bucket(Bucket):
    kind = "s3"

    def _object_url(self, name: str = "") -> str:
        return f"https://{self.bucket}.s3.{self.region or 'us-east-1'}.amazonaws.com/{quote(name, safe='/~')}"

    async def put(self, name, data, content_type="application/octet-stream"):
        await self._call("PUT", self._object_url(name), "writing an object to S3", content=data, headers={"content-type": content_type})

    async def list(self, prefix):
        names, token = [], None
        while True:
            url = f"{self._object_url()}?list-type=2&prefix={quote(prefix, safe='')}" + (f"&continuation-token={quote(token, safe='')}" if token else "")
            r = await self._call("GET", url, "listing objects in S3")
            root = ET.fromstring(r.content)
            ns = {"s": root.tag.split("}")[0].strip("{")} if root.tag.startswith("{") else {}
            find = (lambda e, t: e.findall(f"s:{t}", ns)) if ns else (lambda e, t: e.findall(t))
            names += [k.text or "" for c in find(root, "Contents") for k in find(c, "Key")]
            nxt = find(root, "NextContinuationToken")
            token = nxt[0].text if nxt else None
            if not token:
                return names


class GCSBucket(Bucket):
    kind = "gs"
    storage = "https://storage.googleapis.com"

    def _object_url(self, name: str) -> str:
        return f"{self.storage}/storage/v1/b/{self.bucket}/o/{quote(name, safe='')}"

    def _get_url(self, name: str) -> str:
        return self._object_url(name) + "?alt=media"

    async def put(self, name, data, content_type="application/octet-stream"):
        await self._call("POST", f"{self.storage}/upload/storage/v1/b/{self.bucket}/o?uploadType=media&name={quote(name, safe='')}",
                         "writing an object to Cloud Storage", content=data, headers={"content-type": content_type})

    async def list(self, prefix):
        names, token = [], None
        while True:
            url = f"{self.storage}/storage/v1/b/{self.bucket}/o?prefix={quote(prefix, safe='')}"
            url += f"&pageToken={quote(token, safe='')}" if token else ""
            r = await self._call("GET", url, "listing objects in Cloud Storage")
            d = r.json() if r.content else {}
            names += [str(i.get("name")) for i in d.get("items") or [] if i.get("name")]
            token = d.get("nextPageToken")
            if not token:
                return names


class AzureContainer(Bucket):
    """An Azure Blob Storage container (``https://<account>.blob.core.windows.net/<container>/<prefix>``, or
    ``az://<account>/<container>/<prefix>``), with an Entra ID token for ``https://storage.azure.com/.default``."""

    kind = "azure"

    @property
    def _base(self) -> str:
        account, container = self.bucket.split("/", 1)
        return f"https://{account}.blob.core.windows.net/{container}"

    def address(self, name: str) -> str:
        return f"{self._base}/{name}"

    def _headers(self, extra: dict | None = None) -> dict:
        return {"x-ms-version": AZURE_VERSION, "x-ms-date": email.utils.formatdate(usegmt=True), **(extra or {})}

    async def _call(self, method: str, url: str, what: str, **kw) -> httpx.Response:
        kw["headers"] = self._headers(kw.get("headers"))
        return await call(self.client, self.auth, self.who, method, url, what=what, **kw)

    def _object_url(self, name: str) -> str:
        return f"{self._base}/{quote(name, safe='/~')}"

    def lines(self, name):
        return json_lines(self.client, self.auth, self.who, self._object_url(name), headers=self._headers(), what="reading batch results")

    async def put(self, name, data, content_type="application/octet-stream"):
        await self._call("PUT", self._object_url(name), "writing a blob to Azure Storage", content=data,
                         headers={"x-ms-blob-type": "BlockBlob", "content-type": content_type})

    async def list(self, prefix):
        names, marker = [], None
        while True:
            url = f"{self._base}?restype=container&comp=list&prefix={quote(prefix, safe='')}" + (f"&marker={quote(marker, safe='')}" if marker else "")
            r = await self._call("GET", url, "listing blobs in Azure Storage")
            root = ET.fromstring(r.content)
            names += [n.text for n in root.findall("./Blobs/Blob/Name") if n.text]
            nxt = root.find("NextMarker")
            marker = nxt.text if nxt is not None and nxt.text else None
            if not marker:
                return names


def bucket_kind(uri: str) -> str | None:
    if uri.startswith("s3://"):
        return "s3"
    if uri.startswith("gs://"):
        return "gs"
    if uri.startswith("az://") or AZURE_HOST.match(uri):
        return "azure"
    return None


BUCKETS: dict[str, type[Bucket]] = {"s3": S3Bucket, "gs": GCSBucket, "azure": AzureContainer}
