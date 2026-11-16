"""Bill sources read from export files: AWS Data Exports' CUR 2.0 in S3 and Azure Cost Management exports in Blob Storage.

The fixtures in tests/fixtures/bills are small files in the providers' documented column layouts (November 2026):

* cur2-2026-10.csv, cur2-2026-11.csv: CUR 2.0's COST_AND_USAGE_REPORT table with INCLUDE_RESOURCES and no split cost
  allocation data, 115 columns in the order of the table dictionary's groups (as AWS's Cloud Intelligence Dashboards
  query them), map columns as JSON objects, hourly line items. Bedrock rows use the product map's provider, model,
  inference_type and feature keys and Bedrock's usage types; a Marketplace model has its own product code and name.
* azure-focus-1.2-preview.csv: Microsoft's FOCUS 1.2-preview cost and usage details, in the schema page's column order.
* azure-actual-mca-2026-10-01.csv, azure-actual-ea-2026-10-01.csv: the actual-cost schema, dataset version 2026-10-01,
  for an MCA (67 columns, costInUsd) and an EA (59 columns, billing currency only) account.

Neither provider documents a manifest well enough to copy: AWS gives no example (the test manifests list ``dataFiles``,
and one lists them another way), Azure's follows the 2024-04-01 example in the exports tutorial. S3 and Blob Storage are
mocked with httpx.MockTransport, answering ListObjectsV2 and List Blobs as the REST APIs do; the real signers sign.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from urllib.parse import parse_qs, unquote
from xml.sax.saxutils import escape

import httpx
import pytest
from conftest import ledger_row
from test_billing_sources import fake_module, gpu_cfg

from meritorder.backends.auth import make_auth
from meritorder.billing import refresh_once, report
from meritorder.billing.exports import arm_type, cur_tag_keys, mapping, token_count, when, winners, Run
from meritorder.billing.reconcile import Matcher
from meritorder.billing.sources import BillError, double_counted, fetch, source_auth
from meritorder.config import BackendAuth, BillSource, Config, build_config

FIXTURES = Path(__file__).parent / "fixtures" / "bills"
START, END = date(2026, 10, 4), date(2026, 11, 14)  # a 40-day lookback on 2026-11-13
ACCOUNT, SUB = "123456789012", "1b2c3d4e-0000-0000-0000-000000000000"


def fixture(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def rows_of(name: str) -> list[dict]:
    return list(csv.DictReader(io.StringIO(fixture(name).decode())))


def to_csv(rows: list[dict], header: list[str]) -> bytes:
    buf = io.StringIO()
    w = csv.DictWriter(buf, header, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue().encode()


def by_line(got) -> dict[tuple[date, str], dict]:
    return {(ln["day"], ln["line"]): ln for ln in got.lines}


@pytest.fixture
def fake_aws(monkeypatch, tmp_path):
    """Fake keys only: never the developer's own AWS configuration."""
    for var in ("AWS_PROFILE", "AWS_DEFAULT_PROFILE", "AWS_SESSION_TOKEN", "AWS_BEARER_TOKEN_BEDROCK", "AWS_ROLE_ARN",
                "AWS_WEB_IDENTITY_TOKEN_FILE", "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", "AWS_CONTAINER_CREDENTIALS_FULL_URI"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "none"))
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "none"))
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIDEXAMPLE")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")


@pytest.fixture
def fake_azure(monkeypatch):
    import time
    import types

    scopes = []

    class Credential:
        async def get_token(self, scope):
            scopes.append(scope)
            return types.SimpleNamespace(token="eyJ0eXAi.storage", expires_on=time.time() + 3600)

    fake_module(monkeypatch, "azure")
    fake_module(monkeypatch, "azure.identity")
    fake_module(monkeypatch, "azure.identity.aio", DefaultAzureCredential=Credential)
    return scopes


# ------------------------------------------------------------------ the stores, as their REST APIs answer


class FakeS3:
    """One bucket: ListObjectsV2 (two keys a page, so continuation tokens are followed, and the "/" delimiter), GetObject."""

    def __init__(self, bucket: str = "example-billing", region: str = "us-east-1"):
        self.bucket, self.region = bucket, region
        self.objects: dict[str, tuple[bytes, str, str]] = {}
        self.gets: list[str] = []
        self.status: tuple[int, str, dict] | None = None

    def put(self, key: str, body: bytes, modified: str, etag: str | None = None) -> None:
        self.objects[key] = (body, etag or hashlib.md5(body).hexdigest(), modified)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        assert request.url.host == f"{self.bucket}.s3.{self.region}.amazonaws.com" and request.method == "GET"
        auth = request.headers["authorization"]
        assert auth.startswith("AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/") and f"/{self.region}/s3/aws4_request" in auth
        assert request.headers["x-amz-content-sha256"] == hashlib.sha256(b"").hexdigest()  # S3 wants the payload's hash
        if self.status:
            code, body, headers = self.status
            return httpx.Response(code, text=body, headers=headers)
        key = unquote(request.url.path.lstrip("/"))
        q = parse_qs(request.url.query.decode())
        if not key:
            assert q["list-type"] == ["2"]
            prefix, delim = q["prefix"][0], (q.get("delimiter") or [None])[0]
            keys = sorted(k for k in self.objects if k.startswith(prefix) and not (delim and delim in k[len(prefix):]))
            i = int((q.get("continuation-token") or ["0"])[0])
            page, more = keys[i:i + 2], i + 2 < len(keys)
            items = "".join(f"<Contents><Key>{escape(k)}</Key><LastModified>{self.objects[k][2]}</LastModified>"
                            f"<ETag>&quot;{self.objects[k][1]}&quot;</ETag><Size>{len(self.objects[k][0])}</Size>"
                            "<StorageClass>STANDARD</StorageClass></Contents>" for k in page)
            token = f"<NextContinuationToken>{i + 2}</NextContinuationToken>" if more else ""
            return httpx.Response(200, text=f'<?xml version="1.0" encoding="UTF-8"?>\n<ListBucketResult xmlns="http://s3.amazonaws.com/'
                                            f'doc/2006-03-01/"><Name>{self.bucket}</Name><Prefix>{escape(prefix)}</Prefix><KeyCount>'
                                            f"{len(page)}</KeyCount><MaxKeys>2</MaxKeys><IsTruncated>{str(more).lower()}</IsTruncated>"
                                            f"{items}{token}</ListBucketResult>")
        if key not in self.objects:
            return httpx.Response(404, text="<Error><Code>NoSuchKey</Code><Message>The specified key does not exist.</Message></Error>")
        self.gets.append(key)
        body = self.objects[key][0]
        # Served as stored: a .csv.gz with Content-Encoding: gzip must not be unzipped on the way.
        return httpx.Response(200, stream=httpx.ByteStream(body), headers={"etag": f'"{self.objects[key][1]}"',
                                                          **({"content-encoding": "gzip"} if key.endswith(".gz") else {})})


class FakeBlob:
    """One container: List Blobs (two a page, with NextMarker) and Get Blob, with a bearer token or a SAS."""

    def __init__(self, account: str = "examplebilling", container: str = "costs", sas: str | None = None):
        self.account, self.container, self.sas = account, container, sas
        self.blobs: dict[str, tuple[bytes, str, str]] = {}
        self.gets: list[str] = []
        self.status: tuple[int, str] | None = None

    def put(self, name: str, body: bytes, modified: datetime, etag: str | None = None) -> None:
        self.blobs[name] = (body, etag or "0x8DE" + hashlib.md5(body).hexdigest()[:12].upper(), modified.strftime("%a, %d %b %Y %H:%M:%S GMT"))

    def __call__(self, request: httpx.Request) -> httpx.Response:
        assert request.url.host == f"{self.account}.blob.core.windows.net" and request.method == "GET"
        assert request.headers["x-ms-version"] == "2026-04-06" and request.headers["x-ms-date"].endswith(" GMT")
        q = parse_qs(request.url.query.decode())
        if self.sas:
            assert q["sig"] == ["c2lnbmF0dXJl"] and q["sp"] == ["rl"] and "authorization" not in request.headers
        else:
            assert request.headers["authorization"] == "Bearer eyJ0eXAi.storage" and "sig" not in q
        if self.status:
            return httpx.Response(self.status[0], text=self.status[1])
        path = unquote(request.url.path.lstrip("/"))
        if path == self.container:
            assert q["restype"] == ["container"] and q["comp"] == ["list"]
            prefix = q["prefix"][0]
            names = sorted(n for n in self.blobs if n.startswith(prefix))
            i = int((q.get("marker") or ["0"])[0])
            page = names[i:i + 2]
            items = "".join(f"<Blob><Name>{escape(n)}</Name><Properties><Creation-Time>{self.blobs[n][2]}</Creation-Time>"
                            f"<Last-Modified>{self.blobs[n][2]}</Last-Modified><Etag>0x{self.blobs[n][1][2:]}</Etag>"
                            f"<Content-Length>{len(self.blobs[n][0])}</Content-Length><BlobType>BlockBlob</BlobType></Properties>"
                            "</Blob>" for n in page)
            marker = f"<NextMarker>{i + 2}</NextMarker>" if i + 2 < len(names) else "<NextMarker />"
            return httpx.Response(200, text=f'﻿<?xml version="1.0" encoding="utf-8"?><EnumerationResults ServiceEndpoint="https://'
                                            f'{self.account}.blob.core.windows.net/" ContainerName="{self.container}"><Prefix>'
                                            f"{escape(prefix)}</Prefix><MaxResults>2</MaxResults><Blobs>{items}</Blobs>{marker}"
                                            "</EnumerationResults>")
        name = path.split("/", 1)[1]
        if name not in self.blobs:
            return httpx.Response(404, text='﻿<?xml version="1.0" encoding="utf-8"?><Error><Code>BlobNotFound</Code>'
                                            "<Message>The specified blob does not exist.</Message></Error>")
        self.gets.append(name)
        return httpx.Response(200, stream=httpx.ByteStream(self.blobs[name][0]), headers={"etag": f'"{self.blobs[name][1]}"'})


def client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# ------------------------------------------------------------------ AWS: CUR 2.0 in S3

ROOT = "cur/meritorder-cur"


def cur_manifest(period: str, files: list[str]) -> bytes:
    """A Data Exports manifest. AWS documents what it holds (the columns, the files, the period) but shows no example; the
    parser takes ``dataFiles``, or any data file path in it."""
    return json.dumps({"exportArn": f"arn:aws:bcm-data-exports:us-east-1:{ACCOUNT}:export/meritorder-cur-1b2c3d4e",
                       "exportName": "meritorder-cur", "billingPeriod": period, "executionId": "0f9e8d7c-6b5a-4f3e-9d2c-1b0a9f8e7d6c",
                       "dataFiles": [f"s3://example-billing/{f}" for f in files],
                       "columns": [{"name": "bill_bill_type", "type": "string"}],
                       "additionalOutputFiles": {"createTableSql": f"s3://example-billing/{ROOT}/metadata/BILLING_PERIOD={period}/"
                                                                   "meritorder-cur-create-table.sql"}}).encode()


def cur_bucket() -> FakeS3:
    s3 = FakeS3()
    oct_data, nov_data = f"{ROOT}/data/BILLING_PERIOD=2026-10", f"{ROOT}/data/BILLING_PERIOD=2026-11"
    s3.put(f"{oct_data}/meritorder-cur-00001.csv.gz", gzip.compress(fixture("cur2-2026-10.csv")), "2026-11-12T08:01:00.000Z")
    s3.put(f"{oct_data}/meritorder-cur-00002.csv.gz", b"", "2026-11-12T08:01:00.000Z")  # an empty chunk left by a shrinking export
    s3.put(f"{ROOT}/metadata/BILLING_PERIOD=2026-10/meritorder-cur-Manifest.json",
           cur_manifest("2026-10", [f"{oct_data}/meritorder-cur-00001.csv.gz", f"{oct_data}/meritorder-cur-00002.csv.gz"]),
           "2026-11-12T08:02:00.000Z", etag="m10a")
    s3.put(f"{ROOT}/metadata/BILLING_PERIOD=2026-10/meritorder-cur-create-table.sql", b"CREATE TABLE ...", "2026-11-12T08:02:00.000Z")
    s3.put(f"{nov_data}/meritorder-cur-00001.csv.gz", gzip.compress(fixture("cur2-2026-11.csv")), "2026-11-13T04:10:00.000Z")
    s3.put(f"{ROOT}/metadata/BILLING_PERIOD=2026-11/meritorder-cur-Manifest.json",
           cur_manifest("2026-11", [f"{nov_data}/meritorder-cur-00001.csv.gz"]), "2026-11-13T04:11:00.000Z", etag="m11a")
    return s3


def cur_source(**kw) -> BillSource:
    return BillSource(name="aws-cur", type="aws_cur", export=f"s3://example-billing/{ROOT}", accounts=[ACCOUNT], **kw)


async def test_cur_reads_every_line_item_kind_and_each_refresh_once(fake_aws):
    s3 = cur_bucket()
    src = cur_source()
    async with client(s3) as c:
        got = await fetch(src, gpu_cfg(), START, END, c, make_auth(source_auth(src)), {})
    lines = by_line(got)
    d20, d21 = date(2026, 10, 20), date(2026, 10, 21)
    sonnet_in = lines[(d20, "Amazon Bedrock · USE1-Claude5.5Sonnet-input-tokens")]
    assert sonnet_in["cost"] == pytest.approx(15.018) and sonnet_in["tokens"] == 5_006_000  # two hourly rows, summed per day
    assert (sonnet_in["kind"], sonnet_in["model"], sonnet_in["detail"]["unit"]) == ("input", "Claude Sonnet 5.5", "Tokens")
    assert lines[(d20, "Amazon Bedrock · USE1-Claude5.5Sonnet-output-tokens")]["kind"] == "output"
    assert lines[(d20, "Amazon Bedrock · USE1-Claude5.5Sonnet-cache-read-input-token-count")]["kind"] == "cache_read"
    assert lines[(d20, "Amazon Bedrock · USE1-Claude5.5Sonnet-cache-write-input-token-count")]["kind"] == "cache_write"
    batch = lines[(d21, "Amazon Bedrock · USE1-Claude5.5Sonnet-input-tokens · batch")]  # product feature: Batch
    assert batch["detail"]["tier"] == "batch" and batch["cost"] == 2.0
    opus = lines[(d20, "Claude Opus 5.5 (Amazon Bedrock Edition) · USE1-MP:USE1_InputTokenCount-Units")]  # sold on the Marketplace
    assert (opus["model"], opus["tokens"], opus["kind"], opus["cost"]) == (None, None, "input", 5.0)
    assert lines[(d20, "Credit · Amazon Bedrock · Claude Sonnet 5.5")] | {"detail": None} == {
        "day": d20, "line": "Credit · Amazon Bedrock · Claude Sonnet 5.5", "model": "Claude Sonnet 5.5", "kind": "adjustment",
        "deployment": None, "cost": -10.0, "tokens": None, "detail": None}
    assert lines[(d21, "Refund · Amazon Bedrock · Claude Sonnet 5.5")]["cost"] == -2.0
    assert lines[(date(2026, 10, 1), "Tax · AmazonBedrock")]["cost"] == 1.2  # a month-long row: on its first day, left out
    gpu = lines[(d20, "gpu-qwen · Amazon Elastic Compute Cloud")]  # Savings Plan usage at the On-Demand rate, and the disk
    assert (gpu["kind"], gpu["deployment"], gpu["cost"]) == ("gpu", "gpu-qwen", pytest.approx(97.1))
    assert lines[(d20, "SavingsPlanNegation · gpu-qwen · Amazon Elastic Compute Cloud")]["kind"] == "adjustment"
    assert not any("m7i" in t or "Storage Service" in t for _, t in lines)  # other workloads aren't read
    assert sum(ln["cost"] for ln in got.lines if ln["day"] == d20 and "input-tokens" in ln["line"]) == pytest.approx(15.018)  # no 7.00
    assert lines[(date(2026, 11, 12), "Amazon Bedrock · USE1-Claude5.5Sonnet-input-tokens")]["cost"] == 3.0
    assert got.ranges == [(date(2026, 10, 1), date(2026, 11, 1)), (date(2026, 11, 1), date(2026, 12, 1))]
    assert got.covered_from == date(2026, 10, 1) and got.complete_through == date(2026, 11, 11)  # the last day with usage is partial
    assert got.state["read"][f"{ROOT}/metadata/BILLING_PERIOD=2026-10/meritorder-cur-Manifest.json"]["through"] == "2026-10-31"
    assert len(s3.gets) == 5 and not got.notes  # two manifests and three data files; never the Athena SQL

    # Nothing changed: the manifests are listed and their ETags compared; no file is read again.
    s3.gets.clear()
    async with client(s3) as c:
        again = await fetch(src, gpu_cfg(), START, END, c, make_auth(source_auth(src)), got.state)
    assert (again.lines, again.ranges, s3.gets, again.complete_through) == ([], [], [], date(2026, 11, 11))
    assert again.state == got.state

    # A new refresh of November rewrites its files and manifest: only November is read and replaced.
    nov = rows_of("cur2-2026-11.csv")
    nov.append({**nov[1], "line_item_usage_start_date": "2026-11-13T02:00:00Z", "line_item_usage_end_date": "2026-11-13T03:00:00Z"})
    s3.put(f"{ROOT}/data/BILLING_PERIOD=2026-11/meritorder-cur-00001.csv.gz", gzip.compress(to_csv(nov, list(nov[0]))), "2026-11-13T12:00:00.000Z")
    body = s3.objects[f"{ROOT}/metadata/BILLING_PERIOD=2026-11/meritorder-cur-Manifest.json"][0]
    s3.put(f"{ROOT}/metadata/BILLING_PERIOD=2026-11/meritorder-cur-Manifest.json", body, "2026-11-13T12:01:00.000Z", etag="m11b")
    s3.gets.clear()
    async with client(s3) as c:
        third = await fetch(src, gpu_cfg(), START, END, c, make_auth(source_auth(src)), again.state)
    assert third.ranges == [(date(2026, 11, 1), date(2026, 12, 1))] and len(s3.gets) == 2
    assert third.complete_through == date(2026, 11, 12) and len(third.lines) == 3


def cur_parquet(rows: list[dict]) -> bytes:
    """The rows as Data Exports writes Parquet: timestamps, doubles, and the map columns as maps."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    times = {"bill_billing_period_end_date", "bill_billing_period_start_date", "line_item_usage_end_date", "line_item_usage_start_date"}
    doubles = {"line_item_blended_cost", "line_item_unblended_cost", "line_item_net_unblended_cost", "line_item_usage_amount",
               "pricing_public_on_demand_cost", "discount_total_discount", "discount_bundled_discount"}
    maps = {"product", "resource_tags", "cost_category", "tags"}
    cols = {}
    for name in rows[0]:
        vals = [r[name] for r in rows]
        if name in times:
            cols[name] = pa.array([datetime.fromisoformat(v.replace("Z", "+00:00")) if v else None for v in vals], pa.timestamp("ms", "UTC"))
        elif name in doubles:
            cols[name] = pa.array([float(v) if v else None for v in vals], pa.float64())
        elif name in maps:
            cols[name] = pa.array([list(json.loads(v or "{}").items()) for v in vals], pa.map_(pa.string(), pa.string()))
        elif name == "discount":
            cols[name] = pa.array([[] for _ in vals], pa.map_(pa.string(), pa.float64()))
        else:
            cols[name] = pa.array([v or None for v in vals], pa.string())
    buf = io.BytesIO()
    pq.write_table(pa.table(cols), buf, compression="snappy")
    return buf.getvalue()


async def test_cur_parquet_in_create_new_mode(fake_aws):
    """"Create new data export file": each refresh in its own <timestamp>-<execution id> folder, and the period's manifest
    in metadata/BILLING_PERIOD=YYYY-MM/ for the newest. Tags only in the un-normalized ``tags`` column here."""
    pytest.importorskip("pyarrow")
    rows = [{**r, "resource_tags": "{}"} for r in rows_of("cur2-2026-10.csv")]
    s3 = FakeS3()
    run = f"{ROOT}/data/BILLING_PERIOD=2026-10/20261112T080100Z-0f9e8d7c-6b5a-4f3e-9d2c-1b0a9f8e7d6c"
    s3.put(f"{run}/meritorder-cur-00001.snappy.parquet", cur_parquet(rows), "2026-11-12T08:01:00.000Z")
    s3.put(f"{ROOT}/data/BILLING_PERIOD=2026-10/20261111T080100Z-1a1a/meritorder-cur-00001.snappy.parquet", b"old", "2026-11-11T08:01:00.000Z")
    newest = json.dumps({"files": [{"key": f"{run}/meritorder-cur-00001.snappy.parquet", "rows": len(rows)}]}).encode()
    s3.put(f"{ROOT}/metadata/BILLING_PERIOD=2026-10/Manifest.json", newest, "2026-11-12T08:02:00.000Z")
    s3.put(f"{ROOT}/metadata/BILLING_PERIOD=2026-10/20261112T080100Z-0f9e8d7c-6b5a-4f3e-9d2c-1b0a9f8e7d6c/Manifest.json", newest,
           "2026-11-12T08:02:00.000Z")
    s3.put(f"{ROOT}/metadata/BILLING_PERIOD=2026-10/20261111T080100Z-1a1a/Manifest.json", b"not read", "2026-11-11T08:02:00.000Z")
    src = cur_source()
    async with client(s3) as c:
        got = await fetch(src, gpu_cfg(), START, date(2026, 11, 1), c, make_auth(source_auth(src)), {})
    assert s3.gets == [f"{ROOT}/metadata/BILLING_PERIOD=2026-10/Manifest.json", f"{run}/meritorder-cur-00001.snappy.parquet"]
    lines = by_line(got)
    d20 = date(2026, 10, 20)
    assert lines[(d20, "Amazon Bedrock · USE1-Claude5.5Sonnet-input-tokens")]["cost"] == pytest.approx(15.018)
    assert lines[(d20, "gpu-qwen · Amazon Elastic Compute Cloud")]["cost"] == pytest.approx(97.1)
    assert got.ranges == [(date(2026, 10, 1), date(2026, 11, 1))] and got.complete_through == date(2026, 10, 31)


async def test_cur_net_cost_column_and_other_services(fake_aws):
    rows = rows_of("cur2-2026-10.csv")
    rows[0] = {**rows[0], "line_item_net_unblended_cost": "5.4"}  # a 10% private pricing discount on this row; others have none
    s3 = FakeS3()
    s3.put(f"{ROOT}/data/BILLING_PERIOD=2026-10/meritorder-cur-00001.csv", to_csv(rows, list(rows[0])), "2026-11-12T08:01:00.000Z")
    s3.put(f"{ROOT}/metadata/BILLING_PERIOD=2026-10/meritorder-cur-Manifest.json",
           cur_manifest("2026-10", ["data/BILLING_PERIOD=2026-10/meritorder-cur-00001.csv"]), "2026-11-12T08:02:00.000Z")
    src = cur_source(cost_column="line_item_net_unblended_cost", services="(?i)bedrock|amazons3", gpu_tag=None)
    async with client(s3) as c:
        got = await fetch(src, gpu_cfg(), START, date(2026, 11, 1), c, make_auth(source_auth(src)), {})
    lines = by_line(got)
    d20 = date(2026, 10, 20)
    assert lines[(d20, "Amazon Bedrock · USE1-Claude5.5Sonnet-input-tokens")]["cost"] == pytest.approx(14.418)  # 5.40 + 9.018
    assert lines[(d20, "Amazon Simple Storage Service · USE1-TimedStorage-ByteHrs")]["cost"] == 3.0
    assert not any(ln["kind"] == "gpu" for ln in got.lines)  # gpu_tag: null; the tagged instance isn't Bedrock or S3
    assert (d20, "Amazon Elastic Compute Cloud · USE1-BoxUsage:p5.48xlarge") not in lines


async def test_cur_errors_say_what_to_fix(fake_aws, monkeypatch):
    cfg = Config(backends=[], models=[])
    for export, why in [(None, "export must be set: s3://"), ("https://example-billing/cur", "must be s3://<bucket>"),
                        ("s3://example-billing", "must be s3://<bucket>")]:
        src = BillSource(name="c", type="aws_cur", export=export)
        with pytest.raises(BillError, match=why):
            await fetch(src, cfg, START, END, None, make_auth(source_auth(src)), {})
    src = cur_source()
    s3 = FakeS3()
    s3.status = (301, "<Error><Code>PermanentRedirect</Code><Message>The bucket you are attempting to access must be addressed using "
                      "the specified endpoint.</Message></Error>", {"x-amz-bucket-region": "eu-west-1"})
    async with client(s3) as c:
        with pytest.raises(BillError, match=r"HTTP 301: PermanentRedirect: .* \(the bucket is in eu-west-1: set auth: \{region: eu-west-1\}"):
            await fetch(src, cfg, START, END, c, make_auth(source_auth(src)), {})
    s3.status = (400, "<Error><Code>AuthorizationHeaderMalformed</Code><Message>The authorization header is malformed; the region "
                      "'us-east-1' is wrong; expecting 'ap-south-1'</Message><Region>ap-south-1</Region></Error>", {})
    async with client(s3) as c:
        with pytest.raises(BillError, match=r"AuthorizationHeaderMalformed: .* \(the bucket is in ap-south-1"):
            await fetch(src, cfg, START, END, c, make_auth(source_auth(src)), {})
    s3.status = (503, "Service Unavailable", {})
    async with client(s3) as c:
        with pytest.raises(BillError, match="HTTP 503: Service Unavailable$"):
            await fetch(src, cfg, START, END, c, make_auth(source_auth(src)), {})
    s3.status = (403, "<Error><Code>AccessDenied</Code><Message>Access Denied</Message></Error>", {})
    async with client(s3) as c:
        with pytest.raises(BillError, match="S3 listing s3://example-billing/cur/meritorder-cur/metadata/BILLING_PERIOD=2026-10/: "
                                            "HTTP 403: AccessDenied: Access Denied"):
            await fetch(src, cfg, START, END, c, make_auth(source_auth(src)), {})

    manifest = f"{ROOT}/metadata/BILLING_PERIOD=2026-10/meritorder-cur-Manifest.json"
    data = f"{ROOT}/data/BILLING_PERIOD=2026-10/x-00001.csv"
    s3 = FakeS3()
    s3.put(manifest, b"<html>", "2026-11-12T08:02:00.000Z")
    async with client(s3) as c:
        with pytest.raises(BillError, match="isn't JSON"):
            await fetch(src, cfg, START, END, c, make_auth(source_auth(src)), {})
    s3.put(manifest, cur_manifest("2026-10", [data]), "2026-11-12T08:02:00.000Z")
    async with client(s3) as c:
        with pytest.raises(BillError, match="HTTP 404: NoSuchKey"):  # the manifest lists a file that isn't there
            await fetch(src, cfg, START, END, c, make_auth(source_auth(src)), {})
    s3.put(data, b"identity_line_item_id,cost\nabc,1\n", "2026-11-12T08:01:00.000Z")
    async with client(s3) as c:
        with pytest.raises(BillError, match="no line_item_usage_start_date .* is it a CUR 2.0 export"):
            await fetch(src, cfg, START, END, c, make_auth(source_auth(src)), {})
    monthly = [{**r, "line_item_usage_start_date": "2026-10-01T00:00:00Z", "line_item_usage_end_date": "2026-11-01T00:00:00Z"}
               for r in rows_of("cur2-2026-10.csv")]
    s3.put(data, to_csv(monthly, list(monthly[0])), "2026-11-12T08:01:00.000Z")
    async with client(s3) as c:
        with pytest.raises(BillError, match="time granularity is monthly"):
            await fetch(src, cfg, START, END, c, make_auth(source_auth(src)), {})
    s3.put(data, b"PAR1 a Parquet file", "2026-11-12T08:01:00.000Z")
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", None)
    async with client(s3) as c:
        with pytest.raises(BillError, match="Parquet, which needs the parquet extra"):
            await fetch(src, cfg, START, END, c, make_auth(source_auth(src)), {})

    async def dropped(request):
        raise httpx.ConnectError("connection reset")

    for broken in ("list", "manifest", "data"):
        s3 = cur_bucket()

        def handler(request, s3=s3, broken=broken):
            key = unquote(request.url.path.lstrip("/"))
            if (broken == "list" and not key) or (broken == "manifest" and key.endswith(".json")) or (broken == "data" and key.endswith(".gz")):
                raise httpx.ConnectError("connection reset")
            return s3(request)

        async with client(handler) as c:
            with pytest.raises(BillError, match="S3 (listing|reading) .*ConnectError: connection reset"):
                await fetch(src, gpu_cfg(), START, END, c, make_auth(source_auth(src)), {})


async def test_cur_lists_every_page_and_finds_manifests_named_either_way(fake_aws, monkeypatch):
    from meritorder.billing import exports

    s3 = FakeS3(bucket="example.billing", region="eu-west-1")  # a dotted name: path-style URLs
    seen = []

    def handler(request):
        seen.append(str(request.url))
        assert request.url.host == "s3.eu-west-1.amazonaws.com" and request.url.path.startswith("/example.billing")
        request.url = request.url.copy_with(host="example.billing.s3.eu-west-1.amazonaws.com",
                                            path=request.url.path.removeprefix("/example.billing") or "/")
        return s3(request)

    for i in range(5):
        s3.put(f"{ROOT}/metadata/BILLING_PERIOD=2026-10/other-{i}.txt", b"x", "2026-11-12T08:02:00.000Z")
    s3.put(f"{ROOT}/metadata/BILLING_PERIOD=2026-10/Manifest.json", json.dumps({"dataFiles": []}).encode(), "2026-11-12T08:02:00.000Z")
    src = BillSource(name="c", type="aws_cur", export=f"s3://example.billing/{ROOT}", auth=BackendAuth(type="aws_sigv4", region="eu-west-1"))
    async with client(handler) as c:
        got = await fetch(src, Config(backends=[], models=[]), START, date(2026, 11, 1), c, make_auth(source_auth(src)), {})
    assert sum("continuation-token=" in u for u in seen) == 2 and got.lines == [] and got.ranges == [(date(2026, 10, 1), date(2026, 11, 1))]
    assert got.complete_through == date(2026, 9, 30)  # nothing with usage yet: nothing complete
    monkeypatch.setattr(exports, "MAX_PAGES", 2)
    async with client(handler) as c:
        with pytest.raises(BillError, match="S3: too many pages"):
            await fetch(src, Config(backends=[], models=[]), START, date(2026, 11, 1), c, make_auth(source_auth(src)), {})


async def test_an_export_with_no_files_yet(fake_aws):
    async with client(FakeS3()) as c:
        got = await fetch(cur_source(), Config(backends=[], models=[]), START, END, c, make_auth(source_auth(cur_source())), {})
    assert got.lines == [] and got.ranges == [] and got.covered_from is None and got.complete_through == date(2026, 9, 30)
    assert got.notes == ["No export files found under s3://example-billing/cur/meritorder-cur for 2026-10, 2026-11: check the path, and "
                         "that the export has run (its first run can take a day)."]


# ------------------------------------------------------------------ Azure: Cost Management exports in Blob Storage

EXPORT = "exports/focus-daily"


def azure_manifest(name: str, run: str, start: str, end: str, submitted: str, blobs: list[tuple[str, int]], kind: str = "FocusCost",
                   granularity: str | None = "Daily") -> bytes:
    """As in the exports tutorial's example (manifest version 2024-04-01), for a cost export."""
    return json.dumps({
        "manifestVersion": "2024-04-01", "byteCount": 8032, "blobCount": len(blobs), "dataRowCount": sum(n for _, n in blobs),
        "exportConfig": {"exportName": "focus-daily", "resourceId": f"/subscriptions/{SUB}/providers/Microsoft.CostManagement/exports/focus-daily",
                         "dataVersion": "1.2-preview", "apiVersion": "2026-06-01", "type": kind, "timeFrame": "MonthToDate",
                         "granularity": granularity},
        "deliveryConfig": {"partitionData": True, "dataOverwriteBehavior": "CreateNewReport", "fileFormat": "Csv",
                           "compressionMode": "gzip", "containerUri": f"/subscriptions/{SUB}/resourceGroups/billing/providers/"
                                                                     "Microsoft.Storage/storageAccounts/examplebilling",
                           "rootFolderPath": "exports"},
        "runInfo": {"executionType": "Scheduled", "submittedTime": submitted, "runId": run, "startDate": start, "endDate": end},
        "blobs": [{"blobName": f" {b}", "byteCount": 100, "dataRowCount": n} for b, n in blobs]}).encode()


def focus_rows() -> list[dict]:
    return rows_of("azure-focus-1.2-preview.csv")


def azure_container(sas: str | None = None) -> FakeBlob:
    blob = FakeBlob(sas=sas)
    rows = focus_rows()
    header = list(rows[0])
    # October's run of Oct 22 (Oct 1-21), superseded by the full re-export of October on Nov 2: never read.
    a = f"{EXPORT}/20261001-20261031/7d1e0a3c-0000-0000-0000-00000000000a"
    blob.put(f"{a}/part_0_0001.csv.gz", gzip.compress(to_csv(rows[:2], header)), datetime(2026, 10, 22, 6, tzinfo=UTC))
    blob.put(f"{a}/manifest.json", azure_manifest("a", "7d1e0a3c-a", "2026-10-01T00:00:00", "2026-10-21T00:00:00Z",
                                                  "2026-10-22T05:58:00.1234567Z", [(f"{a}/part_0_0001.csv.gz", 2)]),
             datetime(2026, 10, 22, 6, 1, tzinfo=UTC))
    b = f"{EXPORT}/20261001-20261031/7d1e0a3c-0000-0000-0000-00000000000b"
    blob.put(f"{b}/part_0_0001.csv.gz", gzip.compress(to_csv(rows, header)), datetime(2026, 11, 2, 5, tzinfo=UTC))
    blob.put(f"{b}/part_1_0001.csv.gz", gzip.compress(to_csv([], header)), datetime(2026, 11, 2, 5, tzinfo=UTC))  # headers only
    blob.put(f"{b}/manifest.json", azure_manifest("b", "7d1e0a3c-b", "2026-10-01T00:00:00", "2026-10-31T00:00:00Z",
                                                  "2026-11-02T04:12:00.7654321Z", [(f"{b}/part_0_0001.csv.gz", len(rows)),
                                                                                   (f"{b}/part_1_0001.csv.gz", 0)]),
             datetime(2026, 11, 2, 5, 1, tzinfo=UTC))
    c = f"{EXPORT}/20261101-20261130/7d1e0a3c-0000-0000-0000-00000000000c"
    nov = [{**rows[0], "ChargePeriodStart": "2026-11-11T00:00:00Z", "ChargePeriodEnd": "2026-11-12T00:00:00Z"},
           {**rows[1], "ChargePeriodStart": "2026-11-12T00:00:00Z", "ChargePeriodEnd": "2026-11-13T00:00:00Z"}]
    blob.put(f"{c}/part_0_0001.csv.gz", gzip.compress(to_csv(nov, header)), datetime(2026, 11, 13, 4, tzinfo=UTC))
    blob.put(f"{c}/manifest.json", azure_manifest("c", "7d1e0a3c-c", "2026-11-01T00:00:00", "2026-11-12T00:00:00Z",
                                                  "2026-11-13T03:40:00Z", [(f"{c}/part_0_0001.csv.gz", 2)]),
             datetime(2026, 11, 13, 4, 1, tzinfo=UTC))
    old = f"{EXPORT}/20260901-20260930/7d1e0a3c-0000-0000-0000-00000000000s"  # before the window: listed, never read
    blob.put(f"{old}/manifest.json", b"{}", datetime(2026, 9, 30, 6, tzinfo=UTC))
    late = f"{EXPORT}/20260901-20260930/7d1e0a3c-0000-0000-0000-00000000000t"  # September's last re-export: its manifest only
    blob.put(f"{late}/part_0_0001.csv.gz", gzip.compress(to_csv(rows, header)), datetime(2026, 10, 3, 4, tzinfo=UTC))
    blob.put(f"{late}/manifest.json", azure_manifest("t", "7d1e0a3c-t", "2026-09-01T00:00:00", "2026-09-30T00:00:00Z",
                                                     "2026-10-03T03:40:00Z", [(f"{late}/part_0_0001.csv.gz", len(rows))]),
             datetime(2026, 10, 3, 4, 1, tzinfo=UTC))
    return blob


def azure_source(**kw) -> BillSource:
    return BillSource(name="azure-export", type="azure_export", export=f"https://examplebilling.blob.core.windows.net/costs/{EXPORT}",
                      **({"scope": f"/subscriptions/{SUB}"} | kw))


async def test_focus_export_reads_each_days_newest_run(fake_azure):
    blob = azure_container()
    src = azure_source()
    async with client(blob) as c:
        got = await fetch(src, gpu_cfg(), START, END, c, make_auth(source_auth(src)), {})
    assert fake_azure == ["https://storage.azure.com/.default"]
    b, cc = (f"{EXPORT}/20261001-20261031/7d1e0a3c-0000-0000-0000-00000000000b", f"{EXPORT}/20261101-20261130/7d1e0a3c-0000-0000-0000-00000000000c")
    assert not any("00000000000a/part" in g or "00000000000s" in g or "00000000000t/part" in g for g in blob.gets)  # superseded, or before the window
    assert f"{b}/part_0_0001.csv.gz" in blob.gets and f"{cc}/part_0_0001.csv.gz" in blob.gets
    lines = by_line(got)
    d20 = date(2026, 10, 20)
    terra = lines[(d20, "Foundry Models · gpt-6.1-terra Inp Global 1M Tokens")]
    assert terra["cost"] == pytest.approx(15.4) and terra["tokens"] == 6_160_000  # two Foundry resources in the subscription
    assert terra["kind"] == "input" and terra["detail"]["currency"] == "USD"  # x_BilledCostInUsd
    assert lines[(d20, "Foundry Models · gpt-6.1-terra cchd Inp Global 1M Tokens")]["kind"] == "cache_read"
    assert lines[(date(2026, 10, 21), "Foundry Models · gpt-6.1-terra Batch Inp Global 1M Tokens")]["detail"]["tier"] == "batch"
    claude = lines[(d20, "SaaS · Claude in Microsoft Foundry · claude-sonnet-5-5 · input_tokens")]  # one meter; model and kind in tags
    assert (claude["model"], claude["kind"], claude["cost"]) == ("claude-sonnet-5-5", "input", 8.0)
    assert lines[(d20, "SaaS · Claude in Microsoft Foundry · claude-sonnet-5-5 · output_tokens")]["kind"] == "output"
    assert lines[(d20, "Purchase · Foundry Models · Provisioned Throughput Reservation")]["kind"] == "adjustment"
    assert lines[(d20, "gpu-qwen · Virtual Machines")] | {"detail": None} == {
        "day": d20, "line": "gpu-qwen · Virtual Machines", "model": None, "kind": "gpu", "deployment": "gpu-qwen", "cost": 55.5,
        "tokens": None, "detail": None}
    assert lines[(d20, "gpu-qwen · Storage")]["cost"] == 1.0
    assert not any("D8s" in t or "Azure Credit" in t for _, t in lines)  # not a Foundry resource, nor tagged
    assert got.ranges == [(date(2026, 10, 1), date(2026, 11, 1)), (date(2026, 11, 1), date(2026, 11, 13))]
    assert got.covered_from == date(2026, 10, 1) and got.complete_through == date(2026, 11, 11)

    blob.gets.clear()
    async with client(blob) as c:
        again = await fetch(src, gpu_cfg(), START, END, c, make_auth(source_auth(src)), got.state)
    assert (again.lines, again.ranges, blob.gets) == ([], [], [])  # every manifest's ETag is the same: nothing read

    # Today's run: it covers November so far, so it replaces yesterday's for every day.
    d = f"{EXPORT}/20261101-20261130/7d1e0a3c-0000-0000-0000-00000000000d"
    blob.put(f"{d}/part_0_0001.csv.gz", blob.blobs[f"{cc}/part_0_0001.csv.gz"][0], datetime(2026, 11, 14, 4, tzinfo=UTC))
    blob.put(f"{d}/manifest.json", azure_manifest("d", "7d1e0a3c-d", "2026-11-01T00:00:00", "2026-11-13T00:00:00Z",
                                                  "2026-11-14T03:40:00Z", [(f"{d}/part_0_0001.csv.gz", 2)]),
             datetime(2026, 11, 14, 4, 1, tzinfo=UTC))
    blob.gets.clear()
    async with client(blob) as c:
        third = await fetch(src, gpu_cfg(), START, END + timedelta(days=1), c, make_auth(source_auth(src)), again.state)
    assert blob.gets == [f"{d}/manifest.json", f"{d}/part_0_0001.csv.gz"]
    assert third.ranges == [(date(2026, 11, 1), date(2026, 11, 14))]
    assert set(third.state["read"]) == {f"{b}/manifest.json", f"{d}/manifest.json"}

    # A resource group as the scope: the other group's Foundry resource and the GPU group are out.
    src = azure_source(scope=f"/subscriptions/{SUB}/resourceGroups/llm")
    async with client(blob) as c:
        got = await fetch(src, gpu_cfg(), START, END, c, make_auth(source_auth(src)), {})
    lines = by_line(got)
    assert lines[(d20, "Foundry Models · gpt-6.1-terra Inp Global 1M Tokens")]["cost"] == 12.4
    assert not any(ln["kind"] == "gpu" for ln in got.lines)


async def test_actual_cost_export_with_a_sas(monkeypatch):
    """An MCA account's actual-cost export, read with a SAS token: costInUsd, MM/DD/YYYY dates, tags without braces."""
    monkeypatch.setenv("AZURE_EXPORT_SAS", "?sv=2026-04-06&ss=b&srt=co&sp=rl&se=2027-01-01T00:00:00Z&sig=c2lnbmF0dXJl")
    blob = FakeBlob(sas="yes")
    run = "exports/actual/20261001-20261031/0b5d"
    rows = rows_of("azure-actual-mca-2026-10-01.csv")
    blob.put(f"{run}/part_0_0001.csv", to_csv(rows, list(rows[0])), datetime(2026, 11, 2, 5, tzinfo=UTC))
    blob.put(f"{run}/_manifest.json", azure_manifest("actual", "0b5d", "2026-10-01T00:00:00", "2026-10-31T00:00:00Z",
                                                     "2026-11-02T04:12:00Z", [(f"costs/{run}/part_0_0001.csv", len(rows))], kind="ActualCost"),
             datetime(2026, 11, 2, 5, 1, tzinfo=UTC))
    src = BillSource(name="az", type="azure_export", export="https://examplebilling.blob.core.windows.net/costs/exports/actual",
                     auth=BackendAuth(type="api_key", value_env="AZURE_EXPORT_SAS"))
    assert source_auth(src).type == "api_key"
    async with client(blob) as c:
        got = await fetch(src, gpu_cfg(), START, END, c, make_auth(source_auth(src)), {})
    lines = by_line(got)
    d20 = date(2026, 10, 20)
    terra = lines[(d20, "Foundry Models · gpt-6.1-terra Inp Global 1M Tokens")]
    assert (terra["cost"], terra["tokens"], terra["detail"]["currency"]) == (12.4, 4_960_000, "USD")  # "1M" on a token meter
    claude = lines[(d20, "SaaS · Claude Sonnet 5.5 Input Tokens")]  # a per-model Marketplace plan, from before one meter
    assert (claude["kind"], claude["tokens"]) == ("input", 2_666_667)
    assert lines[(date(2026, 10, 21), "Refund · Foundry Models · gpt-6.1-terra Outp Global 1M Tokens")]["cost"] == -4.0
    assert lines[(d20, "gpu-qwen · Virtual Machines")]["cost"] == 55.5
    assert (d20, "Virtual Machines · D8s v5") not in lines
    assert got.complete_through == date(2026, 10, 19)


async def test_ea_export_in_parquet_bills_in_euros(fake_azure, tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    rows = rows_of("azure-actual-ea-2026-10-01.csv")
    cols = {}
    for name in rows[0]:
        vals = [r[name] for r in rows]
        if name == "Date":
            cols[name] = pa.array([datetime(2026, 10, 20) for _ in vals], pa.timestamp("us"))
        elif name in ("Quantity", "CostInBillingCurrency", "EffectivePrice", "PayGPrice"):
            cols[name] = pa.array([float(v) for v in vals], pa.float64())
        else:
            cols[name] = pa.array(vals, pa.string())
    buf = io.BytesIO()
    pq.write_table(pa.table(cols), buf, compression="snappy")
    blob = FakeBlob()
    run = "exports/ea/202610/0c6e"
    blob.put(f"{run}/part_0_0001.snappy.parquet", buf.getvalue(), datetime(2026, 11, 2, 5, tzinfo=UTC))
    blob.put(f"{run}/manifest.json", azure_manifest("ea", "0c6e", "2026-10-01T00:00:00", "2026-10-31T00:00:00Z", "2026-11-02T04:12:00Z",
                                                    [(f"{run}/part_0_0001.snappy.parquet", 2)], kind="ActualCost"),
             datetime(2026, 11, 2, 5, 1, tzinfo=UTC))
    src = BillSource(name="az", type="azure_export", export="https://examplebilling.blob.core.windows.net/costs/exports/ea")
    async with client(blob) as c:
        got = await fetch(src, gpu_cfg(), START, END, c, make_auth(source_auth(src)), {})
    ln = by_line(got)[(date(2026, 10, 20), "Foundry Models · gpt-6.1-terra Inp Global 1M Tokens")]
    assert ln["cost"] == 11.41 and ln["detail"]["currency"] == "EUR"
    matcher = Matcher(src, Config(backends=[], models=[]), [])
    assert matcher.classify(ln) == ("excluded", "billed in EUR; costs are compared in US dollars (this export has no US-dollar cost "
                                                "column: EA accounts' actual-cost exports don't; FOCUS has x_BilledCostInUsd)")


async def test_azure_export_errors_say_what_to_fix(fake_azure):
    cfg = Config(backends=[], models=[])
    for export, why in [(None, "export must be set: https://<account>"), ("s3://x/y/z", "must be https://<account>"),
                        ("https://examplebilling.blob.core.windows.net/costs", "must be https://<account>"),
                        ("https://examplebilling.blob.core.windows.net/costs/x?sv=1&sig=abc", "put a SAS token in auth")]:
        with pytest.raises(BillError, match=why):
            src = BillSource(name="a", type="azure_export", export=export)
            await fetch(src, cfg, START, END, None, make_auth(source_auth(src)), {})
    src = azure_source(resource_types=[])
    blob = FakeBlob()
    blob.status = (403, '﻿<?xml version="1.0" encoding="utf-8"?><Error><Code>AuthorizationPermissionMismatch</Code><Message>This '
                        "request is not authorized to perform this operation using this permission.</Message></Error>")
    async with client(blob) as c:
        with pytest.raises(BillError, match="Blob Storage listing costs/exports/focus-daily/: HTTP 403: AuthorizationPermissionMismatch"):
            await fetch(src, cfg, START, END, c, make_auth(source_auth(src)), {})

    def one_run(**kw) -> FakeBlob:
        blob = FakeBlob()
        run = f"{EXPORT}/20261001-20261031/r1"
        body = kw.pop("body", None)
        blob.put(f"{run}/part_0.csv", body if body is not None else to_csv(focus_rows(), list(focus_rows()[0])), datetime(2026, 11, 2, tzinfo=UTC))
        manifest = kw.pop("manifest", None) or azure_manifest("x", "r1", "2026-10-01T00:00:00", "2026-10-31T00:00:00Z", "2026-11-02T00:00:00Z",
                                                              [(f"{run}/part_0.csv", kw.pop("rows", 12))], **kw)
        blob.put(f"{run}/manifest.json", manifest, datetime(2026, 11, 2, 0, 1, tzinfo=UTC))
        return blob

    cases = [
        (one_run(manifest=b"{not json"), "isn't JSON"),
        (one_run(manifest=b'{"blobs": []}'), "has no runInfo.startDate and endDate"),
        (one_run(granularity="Monthly"), "granularity is monthly"),
        (one_run(rows=13), "lists 13 rows, but its files have 12: the export may have been rewritten while it was read"),
        (one_run(body=b"Name,Value\nx,1\n", rows=1), "neither FOCUS .* nor actual-cost"),
    ]
    monthly = [{**r, "ChargePeriodStart": "2026-10-01T00:00:00Z", "ChargePeriodEnd": "2026-11-01T00:00:00Z"} for r in focus_rows()]
    cases.append((one_run(body=to_csv(monthly, list(monthly[0])), granularity=None), "every charge spans a month"))
    for blob, why in cases:
        async with client(blob) as c:
            with pytest.raises(BillError, match=why):
                await fetch(src, cfg, START, END, c, make_auth(source_auth(src)), {})
    async with client(one_run()) as c:
        with pytest.raises(BillError, match=r"no EffectiveCostUSD column \(billing.sources.azure-export.cost_column\)"):
            await fetch(azure_source(cost_column="EffectiveCostUSD"), cfg, START, END, c, make_auth(source_auth(src)), {})
    async with client(one_run()) as c:
        got = await fetch(azure_source(cost_column="x_EffectiveCostInUsd", resource_types=[], gpu_tag=None), cfg, START, END, c,
                          make_auth(source_auth(src)), {})
    lines = by_line(got)
    d20 = date(2026, 10, 20)
    assert (d20, "Purchase · Foundry Models · Provisioned Throughput Reservation") not in lines  # amortized: $0 on the purchase
    assert lines[(d20, "Credit · Azure Credit · Azure Credit")]["cost"] == -5.0  # resource_types: [] reads everything in scope
    assert lines[(d20, "Virtual Machines · NC80adis H100 v5")]["kind"] == "other"  # gpu_tag: null
    async with client(one_run(kind="PriceSheet")) as c:
        got = await fetch(src, cfg, START, END, c, make_auth(source_auth(src)), {})
    assert got.lines == [] and got.notes[0].endswith("is a PriceSheet export, not cost details: left out")
    blob = one_run()
    blob.blobs.pop(f"{EXPORT}/20261001-20261031/r1/part_0.csv")
    async with client(blob) as c:
        with pytest.raises(BillError, match="HTTP 404: BlobNotFound: The specified blob does not exist"):
            await fetch(src, cfg, START, END, c, make_auth(source_auth(src)), {})
    for broken in ("comp=list", "manifest.json", "part_0"):
        blob = one_run()

        def handler(request, blob=blob, broken=broken):
            if broken in str(request.url):
                raise httpx.ReadTimeout("timed out")
            return blob(request)

        async with client(handler) as c:
            with pytest.raises(BillError, match="Blob Storage (listing|reading) .*ReadTimeout: timed out"):
                await fetch(src, cfg, START, END, c, make_auth(source_auth(src)), {})


async def test_blob_listing_pages_and_limit(fake_azure, monkeypatch):
    from meritorder.billing import exports

    blob = azure_container()
    monkeypatch.setattr(exports, "MAX_PAGES", 2)
    async with client(blob) as c:
        with pytest.raises(BillError, match="Blob Storage: too many pages"):
            await fetch(azure_source(), gpu_cfg(), START, END, c, make_auth(source_auth(azure_source())), {})


# ------------------------------------------------------------------ helpers


@pytest.mark.parametrize("value, expected", [
    ("2026-10-20T01:00:00Z", datetime(2026, 10, 20, 1, tzinfo=UTC)),
    ("2026-10-20T00:00Z", datetime(2026, 10, 20, tzinfo=UTC)),  # FOCUS 1.0
    ("2025-03-21T21:04:06.5234447Z", datetime(2025, 3, 21, 21, 4, 6, tzinfo=UTC)),
    ("10/20/2026", datetime(2026, 10, 20, tzinfo=UTC)),
    ("20261020", datetime(2026, 10, 20, tzinfo=UTC)),
    (date(2026, 10, 20), datetime(2026, 10, 20, tzinfo=UTC)),
    (datetime(2026, 10, 20, 5), datetime(2026, 10, 20, 5, tzinfo=UTC)),
    ("", None), (None, None), ("yesterday", None),
])
def test_when_reads_every_date_format(value, expected):
    assert when(value) == expected


def test_map_columns_in_every_form():
    assert mapping('{"user_meritorder_deployment":"gpu-qwen"}') == {"user_meritorder_deployment": "gpu-qwen"}
    assert mapping('"Meritorder-Deployment": "gpu-qwen","app": "web"') == {"meritorder-deployment": "gpu-qwen", "app": "web"}
    assert mapping("{user_environment=prod, user_team=llm}") == {"user_environment": "prod", "user_team": "llm"}
    assert mapping([("model", "Claude Sonnet 5.5"), ("feature", None)]) == {"model": "Claude Sonnet 5.5", "feature": ""}
    assert mapping({"A": 1}) == {"a": "1"} and mapping("") == {} and mapping(None) == {} and mapping("[1, 2]") == {}


def test_token_counts_and_tags():
    assert token_count(2_000_000, "Tokens") == 2_000_000
    assert token_count(4.96, "1M", "gpt-6.1-terra Inp Global 1M Tokens") == 4_960_000
    assert token_count(2666.667, "1K tokens") == 2_666_667
    assert token_count(1000, "Units") is None and token_count(0, "Tokens") is None and token_count(8, "1 Hour") is None
    assert cur_tag_keys("meritorder-deployment")[:2] == ("user_meritorder_deployment", "resourcetags/meritorder-deployment")
    assert cur_tag_keys("CostCenter")[0] == "user_cost_center"
    assert arm_type(f"/subscriptions/{SUB}/resourceGroups/llm/providers/Microsoft.CognitiveServices/accounts/x") == \
        "microsoft.cognitiveservices/accounts"
    assert arm_type("") == ""


def test_the_newest_refresh_wins_each_day():
    def run(name, start, end, submitted):
        return Run(name, "e", date(2026, 10, start), date(2026, 10, end), submitted, [])

    got = winners([run("a", 1, 22, "2026-10-22T06:00:00"), run("b", 1, 11, "2026-10-11T06:00:00"),
                   run("c", 15, 18, "2026-10-25T06:00:00")], date(2026, 10, 5))
    assert got == {"a": [(date(2026, 10, 5), date(2026, 10, 15)), (date(2026, 10, 18), date(2026, 10, 22))],
                   "c": [(date(2026, 10, 15), date(2026, 10, 18))]}
    assert winners([], date(2026, 10, 1)) == {}


# ------------------------------------------------------------------ one bill, two sources: picked in the config

RAW = {
    "backends": {
        "bedrock": {"format": "anthropic", "base_url": "https://bedrock-mantle.us-east-1.api.aws/anthropic", "auth": {"type": "aws_sigv4"}},
        "bedrock-west": {"format": "anthropic", "base_url": "https://bedrock-mantle.us-west-2.api.aws/anthropic", "auth": {"type": "aws_sigv4"}},
        "azure-openai": {"format": "openai", "base_url": "https://example.openai.azure.com/openai/v1",
                         "auth": {"type": "api_key", "header": "api-key", "value": "k"}},
        "gpu-qwen": {"kind": "self_hosted", "format": "openai", "serve": {"model": "qwen"}},
    },
    "models": {
        "claude-sonnet": {"routes": [{"backend": "bedrock", "model": "anthropic.claude-sonnet-5-5", "price": {"input": 3, "output": 15}},
                                     {"backend": "bedrock-west", "model": "anthropic.claude-sonnet-5-5", "price": {"input": 3, "output": 15}}]},
        "claude-opus": {"routes": [{"backend": "bedrock", "model": "anthropic.claude-opus-5-5", "price": {"input": 5, "output": 25}}]},
        "gpt-terra": {"routes": [{"backend": "azure-openai", "model": "gpt-6.1-terra", "price": {"input": 2.5, "output": 20}}]},
        "qwen": {"hf_id": "Qwen/Qwen3", "routes": [{"backend": "gpu-qwen", "model": "qwen3"}]},
    },
}


def billing(sources: dict) -> dict:
    return {**RAW, "billing": {"sources": sources}}


def test_one_bill_is_read_one_way(tmp_path):
    ce = {"type": "aws"}
    cur = {"type": "aws_cur", "export": "s3://example-billing/cur/meritorder-cur"}
    with pytest.raises(ValueError, match=r"billing.sources.ce \(aws\) and billing.sources.cur \(aws_cur\) both read the AWS bill for "
                                         r"backends bedrock, bedrock-west and GPU instances, so every day would count twice: turn one off"):
        build_config(billing({"ce": ce, "cur": cur}), tmp_path, quiet=True)
    for ok in ({"ce": {**ce, "enabled": False}, "cur": cur},
               {"ce": {**ce, "backends": ["bedrock-west"], "gpu_tag": None}, "cur": {**cur, "backends": ["bedrock"]}},
               {"ce": {**ce, "accounts": ["111111111111"]}, "cur": {**cur, "accounts": [ACCOUNT]}},
               {"ce": ce, "ce2": ce}):  # two of the same kind (say, two payer accounts) are the operator's to split
        build_config(billing(ok), tmp_path, quiet=True)
    split = {"ce": {**ce, "backends": ["bedrock-west"]}, "cur": {**cur, "backends": ["bedrock"]}}
    with pytest.raises(ValueError, match="for GPU instances, so every day"):
        build_config(billing(split), tmp_path, quiet=True)
    no_gpus = {**billing(split), "backends": {k: v for k, v in RAW["backends"].items() if k != "gpu-qwen"},
               "models": {k: v for k, v in RAW["models"].items() if k != "qwen"}}
    build_config(no_gpus, tmp_path, quiet=True)  # no SkyPilot deployments: GPU costs aren't compared

    api = {"type": "azure", "scope": f"/subscriptions/{SUB}/resourceGroups/llm"}
    export = {"type": "azure_export", "export": "https://examplebilling.blob.core.windows.net/costs/exports/focus-daily"}
    with pytest.raises(ValueError, match=r"\(azure\) and billing.sources.ex \(azure_export\) both read the AZURE bill for backends azure-openai"):
        build_config(billing({"api": api, "ex": export}), tmp_path, quiet=True)
    with pytest.raises(ValueError, match="both read the AZURE bill"):
        build_config(billing({"api": api, "ex": {**export, "scope": f"/subscriptions/{SUB}"}}), tmp_path, quiet=True)  # holds the group
    build_config(billing({"api": api, "ex": {**export, "scope": f"/subscriptions/{SUB}/resourceGroups/other"}}), tmp_path, quiet=True)
    cfg = build_config(billing({"cur": cur, "ex": export}), tmp_path, quiet=True)
    assert double_counted(cfg) == []
    assert [s.type for s in cfg.billing.sources] == ["aws_cur", "azure_export"]


# ------------------------------------------------------------------ the whole job: stored, compared, read once

TODAY = date(2026, 11, 13)


def at(day: date, hour: int = 12) -> datetime:
    return datetime(day.year, day.month, day.day, hour, tzinfo=UTC)


async def test_refresh_stores_export_lines_by_period_and_compares_them(store, state, tmp_path, fake_aws, monkeypatch):
    monkeypatch.delenv("MERITORDER_ALERT_WEBHOOK_URL", raising=False)
    cfg = build_config(billing({"aws-cur": {"type": "aws_cur", "export": f"s3://example-billing/{ROOT}", "accounts": [ACCOUNT]}}),
                       tmp_path, quiet=True)
    d20 = date(2026, 10, 20)
    for r in [ledger_row(ts=at(d20), backend="bedrock", served_model="claude-sonnet", upstream_model="anthropic.claude-sonnet-5-5",
                         actual_cost=60.0, input_tokens=5_006_000, output_tokens=3_000_000),
              ledger_row(ts=at(d20, 13), backend="bedrock", served_model="claude-opus", upstream_model="anthropic.claude-opus-5-5",
                         actual_cost=30.0)]:
        await store.record(r)
    await store.add_gpu_usage("gpu-qwen", at(d20), 3600 * 2, 96.0)
    await store.rollup()
    # A line stored before, for a day outside the periods read: it stays.
    await store.replace_bill_lines("aws-cur", date(2026, 9, 30), date(2026, 10, 1), [
        {"day": date(2026, 9, 30), "line": "Amazon Bedrock · earlier", "model": None, "kind": "other", "deployment": None, "cost": 1.0,
         "tokens": None, "detail": {}}])
    s3 = cur_bucket()
    async with client(s3) as c:
        status = await refresh_once(cfg, store, state, today=TODAY, client=c)
    st = status["aws-cur"]
    assert st["ok"] is True and st["covered_from"] == "2026-10-01" and st["complete_through"] == "2026-11-11"
    assert set(st["export"]["read"]) == {f"{ROOT}/metadata/BILLING_PERIOD=2026-10/meritorder-cur-Manifest.json",
                                         f"{ROOT}/metadata/BILLING_PERIOD=2026-11/meritorder-cur-Manifest.json"}
    stored = await store.bill_lines(date(2026, 9, 1), date(2026, 12, 1), "aws-cur")
    assert st["lines"] == len(stored) - 1 and stored[0]["line"] == "Amazon Bedrock · earlier"

    rep = await report(cfg, store, date(2026, 10, 1), TODAY + timedelta(days=1), today=TODAY)
    rows = {(r["period"], r["model"]): r for r in rep["rows"]}
    sonnet = rows[("2026-10-20", "claude-sonnet")]
    assert (sonnet["recorded"], sonnet["billed"], sonnet["status"]) == (60.0, pytest.approx(60.918), "matched")
    assert sonnet["billed_tokens"] == {"input": 5_006_000, "output": 3_000_000, "cache_read": 1_000_000, "cache_write": 160_000}
    opus = rows[("2026-10-20", "claude-opus")]  # the Marketplace product's name is matched by words
    assert (opus["recorded"], opus["billed"], opus["difference"]) == (30.0, 30.0, 0.0)
    src = rep["sources"][0]
    assert src["type"] == "aws_cur" and src["filtered"] is True and src["complete_through"] == "2026-11-11"
    assert src["settled"]["adjustments"] == pytest.approx(-10 - 2 + 1.2 - 60)
    assert rep["gpu"][0] | {"billed_by_source": None} == {"period": "2026-10-20", "deployment": "gpu-qwen", "recorded": 96.0,
                                                          "billed": 97.1, "difference": pytest.approx(1.1), "pct": 1.15,
                                                          "partial": False, "over": False, "billed_by_source": None}
    assert rows[("2026-11-12", "claude-sonnet")]["status"] == "partial"  # newer than the export's last full day

    s3.gets.clear()
    async with client(s3) as c:
        status = await refresh_once(cfg, store, state, today=TODAY, client=c)
    assert s3.gets == [] and status["aws-cur"]["lines"] == 0  # read once
    assert len(await store.bill_lines(date(2026, 9, 1), date(2026, 12, 1), "aws-cur")) == len(stored)  # and kept

    s3.status = (403, "<Error><Code>AccessDenied</Code><Message>Access Denied</Message></Error>", {})
    async with client(s3) as c:
        status = await refresh_once(cfg, store, state, today=TODAY, client=c)
    assert status["aws-cur"]["ok"] is False and status["aws-cur"]["export"]  # what it read stays, for when access is back
    s3.status = None
    s3.gets.clear()
    async with client(s3) as c:
        status = await refresh_once(cfg, store, state, today=TODAY, client=c)
    assert status["aws-cur"]["ok"] is True and s3.gets == []


async def test_an_export_that_has_never_delivered_settles_nothing(store, state, tmp_path, fake_aws):
    cfg = build_config(billing({"aws-cur": {"type": "aws_cur", "export": f"s3://example-billing/{ROOT}"}}), tmp_path, quiet=True)
    await store.record(ledger_row(ts=at(date(2026, 10, 20)), backend="bedrock", served_model="claude-sonnet",
                                  upstream_model="anthropic.claude-sonnet-5-5", actual_cost=60.0))
    async with client(FakeS3()) as c:
        status = await refresh_once(cfg, store, state, today=TODAY, client=c)
    assert status["aws-cur"]["covered_from"] is None and status["aws-cur"]["notes"][0].startswith("No export files found")
    rep = await report(cfg, store, date(2026, 10, 1), TODAY, today=TODAY)
    assert rep["sources"][0]["complete_through"] is None and all(r["status"] == "partial" for r in rep["rows"])
    assert await state.recent_alerts() == []  # no "not on the bill" alert for an export that isn't there yet


async def test_a_missing_sas_is_reported_not_fetched(store, state, tmp_path, monkeypatch):
    monkeypatch.delenv("AZURE_EXPORT_SAS", raising=False)
    cfg = build_config(billing({"az": {"type": "azure_export", "export": "https://examplebilling.blob.core.windows.net/costs/exports/x",
                                       "auth": {"type": "api_key", "value_env": "AZURE_EXPORT_SAS"}}}), tmp_path, quiet=True)
    async with client(lambda r: pytest.fail("no request without a SAS")) as c:
        status = await refresh_once(cfg, store, state, today=TODAY, client=c)
    assert status["az"]["error"] == "AZURE_EXPORT_SAS is not set"


async def test_odd_rows_files_and_replies(fake_aws, fake_azure):
    from meritorder.billing.exports import number

    assert number("n/a") == 0.0 and number(None) == 0.0
    assert source_auth(BillSource(type="azure_export", auth=BackendAuth(type="azure_ad"))).scope == "https://storage.azure.com/.default"
    # CUR: a row with no start, one for another period, a manifest naming another bucket's file and an empty gzip.
    rows = rows_of("cur2-2026-10.csv")
    odd = [{**rows[0], "line_item_usage_start_date": ""}, {**rows[0], "line_item_usage_start_date": "2026-09-30T23:00:00Z"}, rows[0]]
    s3 = FakeS3()
    data = f"{ROOT}/data/BILLING_PERIOD=2026-10"
    s3.put(f"{data}/x-00001.csv", to_csv(odd, list(rows[0])), "2026-11-12T08:01:00.000Z")
    s3.put(f"{data}/x-00002.csv.gz", gzip.compress(b""), "2026-11-12T08:01:00.000Z")
    s3.put(f"{ROOT}/metadata/BILLING_PERIOD=2026-10/x-Manifest.json", json.dumps({"dataFiles": [
        f"s3://example-billing/{data}/x-00001.csv", f"s3://example-billing/{data}/x-00002.csv.gz", "s3://someone-else/cur/x-00001.csv",
        f"s3://example-billing/{ROOT}/metadata/BILLING_PERIOD=2026-10/x-create-table.sql"]}).encode(), "2026-11-12T08:02:00.000Z")
    src = cur_source()
    async with client(s3) as c:
        got = await fetch(src, gpu_cfg(), START, date(2026, 11, 1), c, make_auth(source_auth(src)), {})
    assert [(ln["day"], ln["cost"]) for ln in got.lines] == [(date(2026, 10, 20), 6.0)]
    assert s3.gets == [f"{ROOT}/metadata/BILLING_PERIOD=2026-10/x-Manifest.json", f"{data}/x-00001.csv", f"{data}/x-00002.csv.gz"]

    def no_manifest(request):
        if request.url.path.endswith(".json"):
            return httpx.Response(403, text="<Error><Code>AccessDenied</Code><Message>Access Denied</Message></Error>")
        return s3(request)

    async with client(no_manifest) as c:
        with pytest.raises(BillError, match="S3 reading s3://example-billing/.*x-Manifest.json: HTTP 403: AccessDenied"):
            await fetch(src, gpu_cfg(), START, date(2026, 11, 1), c, make_auth(source_auth(src)), {})

    # Azure: a row with no date, one outside the run's days, a credit on a GPU instance, and a reply that isn't XML.
    focus = focus_rows()
    vm = next(r for r in focus if r["x_SkuMeterName"] == "NC80adis H100 v5")
    odd = [{**focus[0], "ChargePeriodStart": ""}, {**focus[0], "ChargePeriodStart": "2026-09-30T00:00:00Z"},
           {**vm, "ChargeCategory": "Credit", "BilledCost": "-5.5", "x_BilledCostInUsd": "-5.5"},
           {**vm, "ChargeCategory": "Adjustment", "x_BilledCostInUsd": "0"}]
    blob = FakeBlob()
    run = f"{EXPORT}/20261001-20261031/r9"
    blob.put(f"{run}/part_0.csv", to_csv(odd, list(focus[0])), datetime(2026, 11, 2, tzinfo=UTC))
    blob.put(f"{run}/manifest.json", azure_manifest("x", "r9", "2026-10-01T00:00:00", "2026-10-31T00:00:00Z", "2026-11-02T00:00:00Z",
                                                    [(f"{run}/part_0.csv", 4)]), datetime(2026, 11, 2, 0, 1, tzinfo=UTC))
    src = azure_source()
    async with client(blob) as c:
        got = await fetch(src, gpu_cfg(), START, END, c, make_auth(source_auth(src)), {})
    assert [(ln["line"], ln["kind"], ln["cost"]) for ln in got.lines] == [("Credit · gpu-qwen · Virtual Machines", "adjustment", -5.5)]

    def broken(request):
        if request.url.path.endswith("manifest.json"):
            return httpx.Response(500, text="upstream connect error")
        return blob(request)

    async with client(broken) as c:
        with pytest.raises(BillError, match="Blob Storage reading costs/.*manifest.json: HTTP 500: upstream connect error"):
            await fetch(src, gpu_cfg(), START, END, c, make_auth(source_auth(src)), {})
