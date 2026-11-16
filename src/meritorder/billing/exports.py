"""Bill sources read from the providers' export files (docs/08-reporting.md, "Checking costs against the bills"):

* ``aws_cur``: AWS Data Exports' CUR 2.0 table (COST_AND_USAGE_REPORT) in S3, gzipped CSV or Parquet.
* ``azure_export``: Azure Cost Management's scheduled exports in Blob Storage, FOCUS or actual (or amortized) cost,
  CSV (gzipped or not) or Parquet.

Both providers rewrite a whole billing period on every refresh and list the files that make it up in a manifest. So a
fetch lists the manifests for the months the lookback reaches into, works out which refresh is the newest for each day,
and reads a refresh's files only when its manifest is new or its ETag changed since the last fetch. The lines read
replace what was stored for that refresh's days (``Fetched.ranges``); days whose refresh didn't change keep their lines.
Each file streams to a temporary file, then is read row by row in a thread (Parquet needs the ``parquet`` extra), and
only rows in scope are kept, summed per UTC day and line. A day is complete up to the day before the last one the files
have usage for, so an export that stops being delivered never turns into gaps.

Versions followed (each provider's docs as of November 2026): AWS Data Exports for CUR 2.0 (export delivery and the
CUR 2.0 table dictionary; the docs give no example manifest, so the file list is read from ``dataFiles`` or any data
file path in it); Azure exports with manifest version 2024-04-01, FOCUS 1.0, 1.0r2 and 1.2-preview, the EA and MCA
actual-cost schemas 2023-12-01-preview and 2026-10-01; the Blob service REST API at x-ms-version 2026-04-06.
"""

from __future__ import annotations

import asyncio
import csv
import gzip
import io
import json
import re
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from email.utils import formatdate, parsedate_to_datetime
from typing import IO, Any
from urllib.parse import quote, urlparse

import httpx

from ..config import BillSource, Config
from .sources import MAX_PAGES, BillError, Fetched, kind_of, line, source_auth

BLOB_VERSION = "2026-04-06"  # the newest x-ms-version deployed to every region (Blob service docs, October 2026)
DATA_FILE = re.compile(r"\.(csv|csv\.gz|gz|parquet)$", re.IGNORECASE)
COST_EXPORTS = {"focuscost", "actualcost", "amortizedcost", "usage"}  # manifest exportConfig.type values with cost rows
# A usage row longer than this is a month's charge. When every row is, the export has monthly granularity, which can't be
# compared by day; a few in a daily export (a monthly storage charge) count on their first day, as the provider bills them.
LONGEST_ROW = timedelta(hours=25)

# CUR 2.0 line item types (line_item_line_item_type). Usage covered by a Savings Plan is billed at the On-Demand rate and
# offset by a SavingsPlanNegation row, so the usage counts and the negation is left out with the other adjustments.
CUR_USAGE = {"Usage", "DiscountedUsage", "SavingsPlanCoveredUsage"}
CUR_COLUMNS = {"line_item_usage_start_date", "line_item_usage_end_date", "line_item_line_item_type", "line_item_product_code",
               "line_item_usage_type", "line_item_usage_account_id", "line_item_unblended_cost", "line_item_net_unblended_cost",
               "line_item_currency_code", "line_item_usage_amount", "product", "product_servicecode", "product_product_family",
               "pricing_unit", "product_pricing_unit", "resource_tags", "tags"}
FOCUS_COLUMNS = {"chargeperiodstart", "chargeperiodend", "chargecategory", "billedcost", "x_billedcostinusd", "billingcurrency",
                 "servicename", "x_skumetercategory", "x_skumetername", "skumeter", "chargedescription", "resourceid", "x_resourcetype",
                 "subaccountid", "x_resourcegroupname", "tags", "consumedquantity", "consumedunit"}
ACTUAL_COLUMNS = {"date", "chargetype", "costinusd", "costinbillingcurrency", "cost", "pretaxcost", "billingcurrency",
                  "billingcurrencycode", "metercategory", "metername", "resourceid", "instanceid", "subscriptionid",
                  "resourcegroupname", "resourcegroup", "tags", "quantity", "unitofmeasure"}
USD_COLUMNS = {"x_billedcostinusd", "x_effectivecostinusd", "x_listcostinusd", "x_contractedcostinusd", "costinusd", "paygcostinusd"}


# ---------------------------------------------------------------- values as the files write them


def when(v: Any) -> datetime | None:
    """A time as CUR, FOCUS or the actual-cost schema writes it (ISO with or without seconds, MM/DD/YYYY, YYYYMMDD), or a
    Parquet timestamp. UTC."""
    if v is None or v == "":
        return None
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=UTC)
    if isinstance(v, date):
        return datetime(v.year, v.month, v.day, tzinfo=UTC)
    s = str(v).strip()
    m = re.match(r"^(\d{4})-?(\d{2})-?(\d{2})(?:[T ](\d{2}):(\d{2})(?::(\d{2}))?)?", s)
    if m:
        y, mo, d, h, mi, sec = (int(x) if x else 0 for x in m.groups())
        return datetime(y, mo, d, h, mi, sec, tzinfo=UTC)
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})", s)
    if m:
        return datetime(int(m.group(3)), int(m.group(1)), int(m.group(2)), tzinfo=UTC)
    return None


def number(v: Any) -> float:
    if v is None or v == "":
        return 0.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def mapping(v: Any) -> dict[str, str]:
    """A map column: a dict, Parquet's list of (key, value) pairs, a JSON object, Azure's bare ``"k": "v"`` pairs, or
    ``{k=v, k2=v2}``. Keys lowercased."""
    if not v:
        return {}
    if isinstance(v, dict):
        items = v.items()
    elif isinstance(v, list):
        items = [tuple(x) for x in v if isinstance(x, (list, tuple)) and len(x) == 2]
    else:
        s = str(v).strip()
        try:
            got = json.loads(s if s.startswith("{") else "{" + s + "}")
            items = got.items() if isinstance(got, dict) else []
        except ValueError:
            items = [tuple(p.split("=", 1)) for p in s.strip("{}").split(", ") if "=" in p]
    return {str(k).strip().lower(): "" if x is None else str(x).strip() for k, x in items}


def token_count(qty: float, unit: str | None, also: str = "") -> int | None:
    """Tokens, where the unit says the quantity is tokens ("Tokens", "1K tokens", Azure's "1M" on a "... 1M Tokens" meter)."""
    u = (unit or "").strip().lower()
    if not qty or "token" not in f"{u} {also.lower()}":
        return None
    m = re.match(r"^(\d+)?\s*([km])?\b", u)
    times = int(m.group(1) or 1) * {"k": 1000, "m": 1_000_000}.get(m.group(2) or "", 1) if m else 1
    return int(round(qty * times))


def cur_tag_keys(tag: str) -> tuple[str, ...]:
    """Where CUR 2.0 puts a user tag: ``resource_tags`` normalizes the key (an underscore before capitals, lowercase,
    anything else an underscore, prefixed ``user_``); the ``tags`` column keeps it, prefixed ``resourceTags/``."""
    norm = re.sub(r"_+", "_", re.sub(r"[^a-z0-9]", "_", re.sub(r"([A-Z])", r"_\1", f"user:{tag}").lower())).strip("_")
    return norm, f"resourcetags/{tag.lower()}", f"resourcetags/user:{tag.lower()}", f"user:{tag.lower()}"


def arm_type(resource_id: str) -> str:
    """microsoft.cognitiveservices/accounts from an Azure resource id."""
    m = re.search(r"/providers/([^/]+)/([^/]+)", resource_id, re.IGNORECASE)
    return f"{m.group(1)}/{m.group(2)}".lower() if m else ""


# ---------------------------------------------------------------- reading a file


def read_table(f: IO[bytes], wanted: set[str], start: Callable[[list[str]], Callable[[dict], None]]) -> int:
    """Every row of one export file, as a dict of the ``wanted`` columns that it has (names lowercased), passed to the
    handler that ``start`` makes from the file's columns. CSV (gzipped or not) or Parquet, by the file's first bytes.
    Returns the number of rows."""
    f.seek(0)
    head = f.read(4)
    f.seek(0)
    if head == b"PAR1":
        try:
            import pyarrow.parquet as pq
        except ImportError as e:
            raise BillError("the export is in Parquet, which needs the parquet extra (uv sync --extra parquet, or build the image "
                            "with --build-arg EXTRAS=\"... --extra parquet\"); or export gzipped CSV instead") from e
        pf = pq.ParquetFile(f)
        names = {n.lower(): n for n in pf.schema_arrow.names}
        handle = start(list(names))
        n = 0
        for batch in pf.iter_batches(batch_size=50_000, columns=[names[c] for c in names if c in wanted]):
            for row in batch.to_pylist():
                handle({k.lower(): v for k, v in row.items()})
                n += 1
        return n
    if not head:
        return 0  # Data Exports leaves empty chunks behind when an export shrinks
    raw: IO[bytes] = gzip.GzipFile(fileobj=f) if head[:2] == b"\x1f\x8b" else f
    text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
    csv.field_size_limit(max(csv.field_size_limit(), 1 << 26))  # Azure's additionalInfo can be long
    reader = csv.reader(text)
    header = [h.strip().lower() for h in next(reader, [])]
    if not header:
        return 0
    handle = start(header)
    cols = [(i, h) for i, h in enumerate(header) if h in wanted]
    n = 0
    for rec in reader:
        if rec:
            handle({h: rec[i] if i < len(rec) else "" for i, h in cols})
            n += 1
    return n


class Sum:
    """Bill lines summed per (day, line) as rows are read, for the days a refresh covers."""

    def __init__(self, ranges: list[tuple[date, date]]):
        self.lines: dict[tuple[date, str], dict] = {}
        self.ranges = ranges
        self.last: date | None = None  # the last day any usage row is for
        self.days = self.months = 0  # usage rows of a day or less, and longer ones

    def keep(self, day: date) -> bool:
        return any(a <= day < b for a, b in self.ranges)

    def used(self, day: date, start: datetime, end: datetime | None) -> None:
        if not self.keep(day):
            return
        if self.last is None or day > self.last:
            self.last = day
        if end is not None and end - start > LONGEST_ROW:
            self.months += 1
        else:
            self.days += 1

    def add(self, day: date, text: str, cost: float, kind: str, model: str | None = None, tokens: int | None = None,
            deployment: str | None = None, quantity: float | None = None, **detail: Any) -> None:
        cur = self.lines.get((day, text[:500]))
        if cur is None:
            self.lines[(day, text[:500])] = line(day, text, cost, kind, model=model, tokens=tokens, deployment=deployment,
                                                 quantity=quantity, **detail)
            return
        cur["cost"] += cost
        if tokens is not None:
            cur["tokens"] = (cur["tokens"] or 0) + tokens
        if quantity is not None:
            cur["detail"]["quantity"] = cur["detail"].get("quantity", 0) + quantity


# ---------------------------------------------------------------- storage: S3 and Blob Storage


@dataclass
class Obj:
    name: str
    etag: str
    modified: datetime | None


def _xml_kids(el: ET.Element, tag: str) -> list[ET.Element]:
    return [c for c in el if c.tag.rsplit("}", 1)[-1] == tag]


def _xml_text(el: ET.Element, tag: str) -> str | None:
    kids = _xml_kids(el, tag)
    return kids[0].text if kids else None


def _xml_why(raw: bytes) -> str:
    """S3's and Blob Storage's error reply: <Error><Code>...</Code><Message>...</Message></Error>."""
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return raw.decode("utf-8", "replace")[:200]
    code, msg = _xml_text(root, "Code"), _xml_text(root, "Message")
    return ": ".join(x.strip() for x in (code, msg) if x)[:300] or raw.decode("utf-8", "replace")[:200]


class Bucket:
    """Where an export's files are: list by prefix, read a small file whole, stream a large one to a file."""

    what = ""

    async def list(self, prefix: str, flat: bool = False) -> list[Obj]:
        raise NotImplementedError

    async def text(self, name: str) -> str:
        raise NotImplementedError

    async def download(self, name: str, out: IO[bytes]) -> None:
        raise NotImplementedError

    def _fail(self, r: httpx.Response, raw: bytes, doing: str) -> BillError:
        return BillError(f"{self.what} {doing}: HTTP {r.status_code}: {_xml_why(raw)}", code=str(r.status_code))

    async def _stream(self, client: httpx.AsyncClient, url: str, headers: dict, out: IO[bytes], doing: str) -> None:
        try:
            async with client.stream("GET", url, headers=headers) as r:
                if r.status_code >= 300:
                    raise self._fail(r, await r.aread(), doing)
                async for chunk in r.aiter_raw():  # as stored: a .csv.gz stays gzipped whatever Content-Encoding says
                    out.write(chunk)
        except httpx.HTTPError as e:
            raise BillError(f"{self.what} {doing}: {type(e).__name__}: {e}"[:300]) from e


class S3Bucket(Bucket):
    """A Data Export's S3 folder, through S3's REST API (ListObjectsV2, GetObject), signed with SigV4."""

    what = "S3"

    def __init__(self, export: str, region: str, client: httpx.AsyncClient, auth):
        u = urlparse(export or "")
        if u.scheme != "s3" or not u.netloc or not u.path.strip("/"):
            raise BillError(f"the export must be s3://<bucket>/<prefix>/<export name>, got {export!r}")
        self.bucket, self.root = u.netloc, u.path.strip("/")
        self.region, self.client, self.auth = region, client, auth
        host = f"s3.{region}.amazonaws.com{'.cn' if region.startswith('cn-') else ''}"
        # Virtual-hosted style, except for bucket names with dots, which TLS can't match against *.s3.<region>.amazonaws.com.
        self.base = f"https://{host}/{self.bucket}" if "." in self.bucket else f"https://{self.bucket}.{host}"

    def _fail(self, r: httpx.Response, raw: bytes, doing: str) -> BillError:
        err = super()._fail(r, raw, doing)
        actual = r.headers.get("x-amz-bucket-region")
        if not actual:  # a signature for the wrong region: <Error><Code>AuthorizationHeaderMalformed</Code><Region>...</Region>
            try:
                actual = _xml_text(ET.fromstring(raw), "Region")
            except ET.ParseError:
                actual = None
        if actual and actual != self.region:
            err = BillError(f"{err} (the bucket is in {actual}: set auth: {{region: {actual}}} on this source)", code=err.code)
        return err

    async def _signed(self, url: str) -> dict:
        return await self.auth.apply("GET", url, {}, b"")

    async def list(self, prefix: str, flat: bool = False) -> list[Obj]:
        out, token = [], None
        for _ in range(MAX_PAGES):
            q = f"list-type=2&prefix={quote(prefix, safe='')}" + ("&delimiter=%2F" if flat else "")
            url = f"{self.base}/?{q}" + (f"&continuation-token={quote(token, safe='')}" if token else "")
            try:
                r = await self.client.get(url, headers=await self._signed(url))
            except httpx.HTTPError as e:
                raise BillError(f"S3 listing s3://{self.bucket}/{prefix}: {type(e).__name__}: {e}"[:300]) from e
            if r.status_code >= 300:
                raise self._fail(r, r.content, f"listing s3://{self.bucket}/{prefix}")
            root = ET.fromstring(r.content)
            for c in _xml_kids(root, "Contents"):
                out.append(Obj(_xml_text(c, "Key") or "", (_xml_text(c, "ETag") or "").strip('"'), when(_xml_text(c, "LastModified"))))
            token = _xml_text(root, "NextContinuationToken") if _xml_text(root, "IsTruncated") == "true" else None
            if not token:
                return out
        raise BillError("S3: too many pages")

    def _url(self, name: str) -> str:
        return f"{self.base}/{quote(name, safe='/~')}"

    async def text(self, name: str) -> str:
        url = self._url(name)
        try:
            r = await self.client.get(url, headers=await self._signed(url))
        except httpx.HTTPError as e:
            raise BillError(f"S3 reading s3://{self.bucket}/{name}: {type(e).__name__}: {e}"[:300]) from e
        if r.status_code >= 300:
            raise self._fail(r, r.content, f"reading s3://{self.bucket}/{name}")
        return r.text

    async def download(self, name: str, out: IO[bytes]) -> None:
        url = self._url(name)
        await self._stream(self.client, url, await self._signed(url), out, f"reading s3://{self.bucket}/{name}")


class BlobFolder(Bucket):
    """An export's folder in Azure Blob Storage, through the Blob service REST API (List Blobs, Get Blob), with an Entra
    token (Storage Blob Data Reader) or a SAS token (read and list) appended to each URL."""

    what = "Blob Storage"

    def __init__(self, export: str, client: httpx.AsyncClient, auth, sas: str | None):
        u = urlparse(export or "")
        container, _, path = u.path.strip("/").partition("/")
        if u.scheme != "https" or not u.netloc or not container or not path:
            raise BillError("the export must be https://<account>.blob.core.windows.net/<container>/<directory>/<export name>, "
                            f"got {export!r}")
        if u.query:
            raise BillError("the export URL has a query string: put a SAS token in auth (type: api_key, value_env or value_file), "
                            "not in the URL")
        self.account, self.container, self.root = f"https://{u.netloc}", container, path.strip("/")
        self.client, self.auth, self.sas = client, auth, (sas or "").lstrip("?") or None

    async def _headers(self, url: str) -> dict:
        h = {"x-ms-version": BLOB_VERSION, "x-ms-date": formatdate(usegmt=True)}
        return h if self.sas else await self.auth.apply("GET", url, h, b"")

    def _with_sas(self, url: str) -> str:
        return url if not self.sas else f"{url}{'&' if '?' in url else '?'}{self.sas}"

    async def list(self, prefix: str, flat: bool = False) -> list[Obj]:
        out, marker = [], None
        for _ in range(MAX_PAGES):
            url = (f"{self.account}/{self.container}?restype=container&comp=list&prefix={quote(prefix, safe='')}"
                   + ("&delimiter=%2F" if flat else "") + (f"&marker={quote(marker, safe='')}" if marker else ""))
            try:
                r = await self.client.get(self._with_sas(url), headers=await self._headers(url))
            except httpx.HTTPError as e:
                raise BillError(f"Blob Storage listing {self.container}/{prefix}: {type(e).__name__}: {e}"[:300]) from e
            if r.status_code >= 300:
                raise self._fail(r, r.content, f"listing {self.container}/{prefix}")
            root = ET.fromstring(r.content)
            for blobs in _xml_kids(root, "Blobs"):
                for b in _xml_kids(blobs, "Blob"):
                    props = (_xml_kids(b, "Properties") or [ET.Element("Properties")])[0]
                    modified = _xml_text(props, "Last-Modified")
                    out.append(Obj(_xml_text(b, "Name") or "", (_xml_text(props, "Etag") or "").strip('"'),
                                   parsedate_to_datetime(modified) if modified else None))
            marker = _xml_text(root, "NextMarker")
            if not marker:
                return out
        raise BillError("Blob Storage: too many pages")

    def _url(self, name: str) -> str:
        return f"{self.account}/{self.container}/{quote(name, safe='/~')}"

    async def text(self, name: str) -> str:
        url = self._url(name)
        try:
            r = await self.client.get(self._with_sas(url), headers=await self._headers(url))
        except httpx.HTTPError as e:
            raise BillError(f"Blob Storage reading {self.container}/{name}: {type(e).__name__}: {e}"[:300]) from e
        if r.status_code >= 300:
            raise self._fail(r, r.content, f"reading {self.container}/{name}")
        return r.content.decode("utf-8-sig")

    async def download(self, name: str, out: IO[bytes]) -> None:
        url = self._url(name)
        await self._stream(self.client, self._with_sas(url), await self._headers(url), out, f"reading {self.container}/{name}")


# ---------------------------------------------------------------- refreshes, and which one each day comes from


@dataclass
class Run:
    """One refresh of an export: the files that make up [start, end) as of ``submitted``."""

    name: str  # the manifest
    etag: str
    start: date
    end: date
    submitted: str  # ISO, UTC: the newest refresh of a day wins
    files: list[str]
    rows: int | None = None  # the manifest's row count (Azure), to check that every row was read
    kind: str = ""  # Azure's export type (FocusCost, ActualCost, ...)
    monthly: bool = False

    def cached(self) -> dict:
        return {"etag": self.etag, "start": self.start.isoformat(), "end": self.end.isoformat(), "submitted": self.submitted,
                "files": self.files, "rows": self.rows, "kind": self.kind, "monthly": self.monthly}

    @classmethod
    def from_cache(cls, name: str, c: dict) -> Run:
        return cls(name, c["etag"], date.fromisoformat(c["start"]), date.fromisoformat(c["end"]), c["submitted"], list(c["files"]),
                   c.get("rows"), c.get("kind") or "", bool(c.get("monthly")))


def winners(runs: list[Run], lo: date) -> dict[str, list[tuple[date, date]]]:
    """For each day from ``lo`` on, the newest refresh that covers it; returned as each refresh's ranges of days."""
    out: dict[str, list[tuple[date, date]]] = {}
    hi = max((r.end for r in runs), default=lo)
    d, prev = lo, None
    while d < hi:
        cover = [r for r in runs if r.start <= d < r.end]
        best = max(cover, key=lambda r: (r.submitted, r.name)).name if cover else None
        if best is not None:
            ranges = out.setdefault(best, [])
            if prev == best and ranges and ranges[-1][1] == d:
                ranges[-1] = (ranges[-1][0], d + timedelta(days=1))
            else:
                ranges.append((d, d + timedelta(days=1)))
        prev, d = best, d + timedelta(days=1)
    return out


def _months(lo: date, hi: date) -> list[date]:
    out, d = [], lo.replace(day=1)
    while d < hi:
        out.append(d)
        d = (d + timedelta(days=32)).replace(day=1)
    return out


def _next_month(d: date) -> date:
    return (d.replace(day=1) + timedelta(days=32)).replace(day=1)


class Export:
    """One export source: its refreshes, and reading one."""

    wanted: set[str] = set()  # the columns its rows are read for (lowercase)

    def __init__(self, src: BillSource, cfg: Config, bucket: Bucket):
        self.src, self.cfg, self.bucket = src, cfg, bucket
        self.notes: list[str] = []
        self.seen: dict[str, dict] = {}  # every manifest listed, as Run.cached(), in or out of the window: read again only when it changes

    async def runs(self, lo: date, hi: date, known: dict[str, dict]) -> list[Run]:
        """The refreshes with days in [lo, hi); ``known`` holds manifests read before, by name, to reuse while their ETag holds."""
        raise NotImplementedError

    def handler(self, columns: list[str], acc: Sum, run: Run) -> Callable[[dict], None]:
        raise NotImplementedError

    async def read(self, run: Run, ranges: list[tuple[date, date]]) -> Sum:
        """Every file of one refresh, streamed to a temporary file and read in a thread, summed into bill lines."""
        acc, rows = Sum(ranges), 0
        for name in run.files:
            with tempfile.TemporaryFile() as f:
                await self.bucket.download(name, f)
                rows += await asyncio.to_thread(read_table, f, self.wanted | {self.src.cost_column.lower()},
                                                lambda cols: self.handler(cols, acc, run))
        if run.rows is not None and rows != run.rows:
            raise BillError(f"{run.name} lists {run.rows:,} rows, but its files have {rows:,}: the export may have been rewritten "
                            "while it was read; it's read again next time")
        if acc.months and not acc.days:
            raise BillError(f"{run.name}: every charge spans a month, so the export's time granularity is monthly. "
                            "Create one with daily (or, for AWS, hourly) granularity to compare by day")
        return acc


# ---------------------------------------------------------------- AWS Data Exports (CUR 2.0)


class AwsCur(Export):
    """Data Exports writes s3://<bucket>/<prefix>/<export>/data/BILLING_PERIOD=YYYY-MM/<export>-00001.csv.gz (or
    .snappy.parquet; in "create new" mode under a <timestamp>-<execution id> folder), and in metadata/BILLING_PERIOD=YYYY-MM/
    a manifest of the period's newest refresh, written once all of its files are there."""

    wanted = CUR_COLUMNS

    def __init__(self, src: BillSource, cfg: Config, bucket: S3Bucket):
        super().__init__(src, cfg, bucket)
        self.s3 = bucket
        self.services = re.compile(src.services if "services" in src.model_fields_set else "(?i)bedrock")
        self.cost = src.cost_column.lower() if "cost_column" in src.model_fields_set else "line_item_unblended_cost"
        self.tag_keys = cur_tag_keys(src.gpu_tag) if src.gpu_tag else ()

    def files(self, doc: Any) -> list[str]:
        """The data files a manifest lists, as keys: ``dataFiles``, or any data file path in it (not additionalOutputFiles)."""
        found: list[str] = []

        def walk(v: Any) -> None:
            if isinstance(v, dict):
                for k, x in v.items():
                    if k != "additionalOutputFiles":
                        walk(x)
            elif isinstance(v, list):
                for x in v:
                    walk(x)
            elif isinstance(v, str) and DATA_FILE.search(v.strip()):
                found.append(v.strip())

        walk(doc.get("dataFiles", doc) if isinstance(doc, dict) else doc)
        keys = []
        for f in found:
            if f.startswith("s3://"):
                bucket, _, f = f[5:].partition("/")
                if bucket != self.s3.bucket:
                    continue
            k = f.lstrip("/")
            if not k.startswith(self.s3.root + "/"):
                k = f"{self.s3.root}/{k}"
            if f"{self.s3.root}/data/" in k and k not in keys:
                keys.append(k)
        return keys

    async def runs(self, lo: date, hi: date, known: dict[str, dict]) -> list[Run]:
        out = []
        for month in _months(lo, hi):
            folder = f"{self.s3.root}/metadata/BILLING_PERIOD={month:%Y-%m}/"
            found = [o for o in await self.s3.list(folder, flat=True) if o.name.lower().endswith("manifest.json")]
            if not found:
                continue
            m = max(found, key=lambda o: (o.modified or datetime.min.replace(tzinfo=UTC), o.name))
            c = known.get(m.name)
            if c and c.get("etag") == m.etag:
                run = Run.from_cache(m.name, c)
            else:
                try:
                    doc = json.loads(await self.s3.text(m.name))
                except ValueError as e:
                    raise BillError(f"s3://{self.s3.bucket}/{m.name} isn't JSON: {e}") from e
                submitted = (m.modified or datetime.now(UTC)).astimezone(UTC).isoformat(timespec="seconds")
                run = Run(m.name, m.etag, month, _next_month(month), submitted, self.files(doc))
            self.seen[m.name] = run.cached()
            out.append(run)
        return out

    def _tag(self, r: dict) -> str | None:
        if not self.tag_keys:
            return None
        for col in ("resource_tags", "tags"):
            tags = mapping(r.get(col))
            for k in self.tag_keys:
                if tags.get(k):
                    return tags[k]
        return None

    def handler(self, columns: list[str], acc: Sum, run: Run) -> Callable[[dict], None]:
        if "line_item_usage_start_date" not in columns or "line_item_line_item_type" not in columns:
            raise BillError(f"{run.name}: the files have no line_item_usage_start_date or line_item_line_item_type column: is it a "
                            "CUR 2.0 export?")
        cost_col = self.cost if self.cost in columns else "line_item_unblended_cost"
        accounts = set(self.src.accounts)

        def handle(r: dict) -> None:
            start = when(r.get("line_item_usage_start_date"))
            if start is None:
                return
            typ = r.get("line_item_line_item_type") or ""
            usage, day = typ in CUR_USAGE, start.date()
            if usage:
                acc.used(day, start, when(r.get("line_item_usage_end_date")))
            if not acc.keep(day) or (accounts and r.get("line_item_usage_account_id") not in accounts):
                return
            raw = r.get(cost_col)
            cost = number(raw if raw not in (None, "") else r.get("line_item_unblended_cost"))
            product = mapping(r.get("product"))
            service = product.get("product_name") or r.get("product_servicecode") or r.get("line_item_product_code") or ""
            currency = (r.get("line_item_currency_code") or "USD").upper()
            deployment = self._tag(r)
            if deployment:
                if usage:
                    acc.add(day, f"{deployment} · {service}", cost, "gpu", deployment=deployment, currency=currency)
                elif cost:
                    acc.add(day, f"{typ} · {deployment} · {service}", cost, "adjustment", record_type=typ, currency=currency)
                return
            names = " ".join(str(x or "") for x in (r.get("line_item_product_code"), r.get("product_servicecode"),
                                                     r.get("product_product_family"), product.get("product_name")))
            if not self.services.search(names):
                return
            model = product.get("model") or None
            usage_type = r.get("line_item_usage_type") or ""
            if not usage:
                if cost:
                    acc.add(day, f"{typ} · {service}" + (f" · {model}" if model else ""), cost, "adjustment", model=model,
                            record_type=typ, currency=currency)
                return
            batch = (product.get("feature") or "").lower() == "batch" or "batch" in usage_type.lower()
            kind = kind_of(usage_type)
            if kind == "other":
                kind = kind_of(product.get("inference_type") or "")
            qty, unit = number(r.get("line_item_usage_amount")), r.get("pricing_unit") or r.get("product_pricing_unit") or None
            text = f"{service} · {usage_type}" + (" · batch" if batch and "batch" not in usage_type.lower() else "")
            acc.add(day, text, cost, kind, model=model, tokens=token_count(qty, unit), quantity=qty, unit=unit,
                    tier="batch" if batch else None, currency=currency)

        return handle


# ---------------------------------------------------------------- Azure Cost Management exports


class AzureExport(Export):
    """Exports write <container>/<directory>/<export>/<YYYYMMDD-YYYYMMDD>/<run id>/ with the run's files and a
    manifest.json (manifest version 2024-04-01) listing them. The layout varies between API versions, so the manifests
    are found by listing the folder, and the files by reading them. A month-to-date export rewrites the month on every
    daily run; with "overwrite" off, older runs stay, and the newest covering a day wins."""

    wanted = FOCUS_COLUMNS | ACTUAL_COLUMNS

    def __init__(self, src: BillSource, cfg: Config, bucket: BlobFolder):
        super().__init__(src, cfg, bucket)
        self.blob = bucket
        self.cost = src.cost_column.lower() if "cost_column" in src.model_fields_set else None
        self.scope = src.scope.rstrip("/").lower() + "/" if src.scope else None
        self.types = [t.lower() for t in src.resource_types]

    def _blob_name(self, name: str) -> str:
        n = name.strip().lstrip("/")
        if n.startswith(self.blob.container + "/") and not n.startswith(self.blob.root.split("/", 1)[0] + "/"):
            n = n[len(self.blob.container) + 1:]
        return n

    async def runs(self, lo: date, hi: date, known: dict[str, dict]) -> list[Run]:
        out = []
        since = datetime.combine(lo, datetime.min.time(), tzinfo=UTC)
        for o in await self.blob.list(self.blob.root + "/"):
            if not o.name.lower().endswith("manifest.json") or (o.modified is not None and o.modified < since):
                continue  # a manifest written before the window only covers days before it
            c = known.get(o.name)
            if c and c.get("etag") == o.etag:
                run = Run.from_cache(o.name, c)
            else:
                try:
                    doc = json.loads(await self.blob.text(o.name))
                except ValueError as e:
                    raise BillError(f"{self.blob.container}/{o.name} isn't JSON: {e}") from e
                info, conf = doc.get("runInfo") or {}, doc.get("exportConfig") or {}
                first, last = when(info.get("startDate")), when(info.get("endDate"))
                if first is None or last is None:
                    raise BillError(f"{self.blob.container}/{o.name} has no runInfo.startDate and endDate: is it an export manifest?")
                submitted = (when(info.get("submittedTime")) or o.modified or datetime.now(UTC)).astimezone(UTC)
                run = Run(o.name, o.etag, first.date(), last.date() + timedelta(days=1), submitted.isoformat(timespec="seconds"),
                          [self._blob_name(b["blobName"]) for b in doc.get("blobs") or [] if b.get("blobName")],
                          doc.get("dataRowCount"), str(conf.get("type") or ""), str(conf.get("granularity") or "").lower() == "monthly")
            self.seen[o.name] = run.cached()
            if run.kind and run.kind.lower() not in COST_EXPORTS:
                note = f"{o.name} is a {run.kind} export, not cost details: left out"
                if note not in self.notes:
                    self.notes.append(note)
                continue
            if run.end <= lo or run.start >= hi:
                continue
            if run.monthly:
                raise BillError(f"{o.name}: the export's granularity is monthly. Set it to daily to compare by day")
            out.append(run)
        return out

    def handler(self, columns: list[str], acc: Sum, run: Run) -> Callable[[dict], None]:
        focus = "chargeperiodstart" in columns
        if not focus and "date" not in columns:
            raise BillError(f"{run.name}: the files have neither FOCUS (ChargePeriodStart) nor actual-cost (Date) columns")
        if self.cost:
            if self.cost not in columns:
                raise BillError(f"{run.name}: no {self.src.cost_column} column (billing.sources.{self.src.name}.cost_column)")
            cost_col = self.cost
        elif focus:
            cost_col = "x_billedcostinusd" if "x_billedcostinusd" in columns else "billedcost"
        else:
            cost_col = next((c for c in ("costinusd", "costinbillingcurrency", "cost", "pretaxcost") if c in columns), "cost")
        in_usd = cost_col in USD_COLUMNS
        tag = (self.src.gpu_tag or "").lower()

        def handle(r: dict) -> None:
            if focus:
                start, end = when(r.get("chargeperiodstart")), when(r.get("chargeperiodend"))
                typ = r.get("chargecategory") or ""
                category = r.get("x_skumetercategory") or r.get("servicename") or ""
                meter = r.get("x_skumetername") or r.get("skumeter") or r.get("chargedescription") or ""
                sub, group = r.get("subaccountid") or "", r.get("x_resourcegroupname") or ""
                qty, unit = number(r.get("consumedquantity")), r.get("consumedunit")
                rtype = (r.get("x_resourcetype") or "").lower()
            else:
                start, end = when(r.get("date")), None
                typ = r.get("chargetype") or ""
                category, meter = r.get("metercategory") or "", r.get("metername") or ""
                sub, group = r.get("subscriptionid") or "", r.get("resourcegroupname") or r.get("resourcegroup") or ""
                qty, unit = number(r.get("quantity")), r.get("unitofmeasure")
                rtype = ""
            if start is None:
                return
            usage, day = typ.lower() == "usage", start.date()
            if usage:
                acc.used(day, start, end)
            if not acc.keep(day):
                return
            rid = (r.get("resourceid") or r.get("instanceid") or "").lower()
            sub = sub.lower()
            path = rid or (f"/subscriptions/{sub.rsplit('/', 1)[-1]}" + (f"/resourcegroups/{group.lower()}" if group else "") if sub else "")
            if self.scope and not (path.rstrip("/") + "/").startswith(self.scope):
                return
            cost = number(r.get(cost_col))
            currency = "USD" if in_usd else (r.get("billingcurrency") or r.get("billingcurrencycode") or "USD").upper()
            tags = mapping(r.get("tags"))
            if tag and tags.get(tag):
                value = tags[tag]
                if usage:
                    acc.add(day, f"{value} · {category}", cost, "gpu", deployment=value, currency=currency)
                elif cost:
                    acc.add(day, f"{typ} · {value} · {category}", cost, "adjustment", record_type=typ, currency=currency)
                return
            if self.types and not any(x.startswith(t) for x in (rtype, arm_type(rid)) if x for t in self.types):
                return
            # Claude in Foundry on one Marketplace meter carries the model and the kind of token as tags.
            model, kind_tag = tags.get("model") or None, tags.get("usage_type") or ""
            text = f"{category} · {meter}" + (f" · {model}" if model and model.lower() not in meter.lower() else "") + (
                f" · {kind_tag}" if kind_tag else "")
            if not usage:
                if cost:
                    acc.add(day, f"{typ} · {text}", cost, "adjustment", model=model, record_type=typ, currency=currency)
                return
            acc.add(day, text, cost, kind_of(kind_tag or meter), model=model, tokens=token_count(qty, unit, meter), quantity=qty,
                    unit=unit or None, tier="batch" if "batch" in text.lower() else None, currency=currency)

        return handle


# ---------------------------------------------------------------- the fetch


async def fetch_export(src: BillSource, cfg: Config, start: date, end: date, client: httpx.AsyncClient, auth,
                       state: dict) -> Fetched:
    """The months the lookback reaches into, from each day's newest refresh; files are read only for refreshes that are
    new or changed (by manifest and ETag) since ``state``, what the last fetch read."""
    if not src.export:
        where = "s3://<bucket>/<prefix>/<export name>" if src.type == "aws_cur" else \
            "https://<account>.blob.core.windows.net/<container>/<directory>/<export name>"
        raise BillError(f"billing.sources.{src.name}.export must be set: {where}")
    spec = source_auth(src)
    if src.type == "aws_cur":
        reader: Export = AwsCur(src, cfg, S3Bucket(src.export, spec.region or "us-east-1", client, auth))
    else:
        reader = AzureExport(src, cfg, BlobFolder(src.export, client, auth, spec.key() if spec.type == "api_key" else None))
    lo = start.replace(day=1)
    known = state.get("manifests") or {}
    runs = await reader.runs(lo, end, known)
    best = winners(runs, lo)
    before = state.get("read") or {}
    lines: list[dict] = []
    replaced: list[tuple[date, date]] = []
    read: dict[str, dict] = {}
    for run in runs:
        ranges = best.get(run.name)
        if not ranges:
            continue  # every day it covers has a newer refresh
        spans = [[a.isoformat(), b.isoformat()] for a, b in ranges]
        prev = before.get(run.name)
        if prev and prev.get("etag") == run.etag and prev.get("ranges") == spans:
            read[run.name] = prev
            continue
        acc = await reader.read(run, ranges)
        lines += list(acc.lines.values())
        replaced += ranges
        through = None
        if acc.last is not None:
            # The last day with usage is partial, unless the refresh came after its period ended.
            closed = acc.last + timedelta(days=1) >= run.end and run.end < date.fromisoformat(run.submitted[:10])
            through = acc.last if closed else acc.last - timedelta(days=1)
        read[run.name] = {"etag": run.etag, "ranges": spans, "through": through.isoformat() if through else None,
                          "files": len(run.files)}
    if not runs:
        months = ", ".join(f"{m:%Y-%m}" for m in _months(lo, end))
        reader.notes.append(f"No export files found under {src.export} for {months}: check the path, and that the export "
                            "has run (its first run can take a day).")
    done = [date.fromisoformat(r["through"]) for r in read.values() if r.get("through")]
    firsts = [date.fromisoformat(r["ranges"][0][0]) for r in read.values() if r.get("ranges")]
    return Fetched(lines, complete_through=max(done) if done else lo - timedelta(days=1), notes=reader.notes, ranges=replaced,
                   covered_from=min(firsts) if firsts else None,
                   state={"manifests": reader.seen, "read": read})
