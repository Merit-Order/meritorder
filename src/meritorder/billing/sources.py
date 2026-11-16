"""Fetching providers' own cost data (docs/08-reporting.md, "Checking costs against the bills").

Each fetcher returns bill lines: one cost per UTC day and line item, in US dollars, with the provider's model name
where the line carries one, the kind of charge (input, output, cache_read, cache_write, web_search, other, adjustment,
gpu), and token counts where the provider reports them. Sources:

* Anthropic: the Usage and Cost Analytics API (``/v1/organizations/analytics/cost`` and ``/usage``), Admin API key.
* OpenAI: the organization Costs API (``/v1/organization/costs``, by line item), Admin key.
* AWS: Cost Explorer (``GetCostAndUsage``), Bedrock by model and inference type; by service and usage type for
  periods without the model breakdown. GPU instances by their SkyPilot tag.
* Google Cloud: the Cloud Billing export in BigQuery (``jobs.query``), Vertex AI by service and SKU; GPUs by label.
* Azure: the Cost Management Query API, by service and meter; GPUs by tag.
* AWS and Azure exports (exports.py): Data Exports' CUR 2.0 files in S3, and Cost Management's scheduled exports in Blob
  Storage, read file by file. They read the same bills as Cost Explorer and the Query API, so a config uses one of each
  pair for a given account (``double_counted``).

Requests go through httpx with the backends' own auth providers (botocore SigV4, google-auth, azure-identity), so tests
mock every one of them with ``httpx.MockTransport``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any
from urllib.parse import urlparse

import httpx

from ..config import BackendAuth, BackendConfig, BillSource, Config, uses_reported_cost

SETTLE_DAYS = {"anthropic": 1, "openai": 1, "aws": 2, "aws_cur": 2, "gcp": 2, "azure": 3, "azure_export": 3}
# The bill a source type reads, where two types read the same one: from the provider's API, or from its export files.
BILL_OF = {"aws_cur": "aws", "azure_export": "azure"}
GPU_LABEL = "meritorder-deployment"  # SkyPilot puts it on every GPU instance it starts for a deployment (gpu.py)

ANTHROPIC_VERSION = "2026-10-21"
ANTHROPIC_BETA = "analytics-usage-cost-2026-08-24"
ANTHROPIC_GROUPS = ("model", "service_tier", "context_window", "product", "speed")  # the API's limit is 5, besides the two below
ANTHROPIC_TOKENS = {"uncached_input_tokens": "input", "output_tokens": "output", "cache_read_input_tokens": "cache_read",
                    "cache_creation.ephemeral_5m_input_tokens": "cache_write", "cache_creation.ephemeral_1h_input_tokens": "cache_write"}
AZURE_API_VERSION = "2026-06-01"
STORAGE_SCOPE = "https://storage.azure.com/.default"
MAX_PAGES = 200  # a runaway paginator must not run forever


class BillError(Exception):
    """A provider refused or failed a request. The message says which and why, never a credential."""

    def __init__(self, message: str, code: str | None = None):
        super().__init__(message)
        self.code = code


@dataclass
class Fetched:
    lines: list[dict]
    # The last day the provider says is complete (Anthropic's data_refreshed_at; for an export, the day before the last
    # one its files have data for); None: settle_days decides.
    complete_through: date | None = None
    notes: list[str] = field(default_factory=list)
    # The ranges [start, end) the lines replace. None: the whole range fetched (the APIs). An export replaces only the
    # billing periods whose files changed since it was last read, and leaves the others as stored.
    ranges: list[tuple[date, date]] | None = None
    covered_from: date | None = None  # an export: the first day of the earliest period it has read
    state: dict | None = None  # an export: which manifests it read, for the next fetch


def line(day: date, text: str, cost: float, kind: str, model: str | None = None, tokens: int | None = None,
         deployment: str | None = None, **detail: Any) -> dict:
    return {"day": day, "line": text[:500], "model": model, "kind": kind, "deployment": deployment, "cost": float(cost),
            "tokens": tokens, "detail": {k: v for k, v in detail.items() if v is not None}}


def kind_of(text: str) -> str:
    """The kind of charge a free-text line item describes: "Gemini 4 Carbon Input Text Tokens", "gpt-6.1-terra-Inp-glbl",
    "USE1-MP:USE1_CacheReadInputTokenCount-Units"."""
    t = re.sub(r"([a-z])([A-Z])", r"\1 \2", text).lower()
    w = set(re.findall(r"[a-z]+", t))
    if w & {"cache", "cached", "cchd", "caching"}:
        if w & {"write", "writes", "creation", "create"}:
            return "cache_write"
        if w & {"storage", "stored", "hour", "hours"}:
            return "other"
        return "cache_read"
    if w & {"output", "outp", "out", "completion", "completions"}:
        return "output"
    if w & {"input", "inp", "prompt"}:
        return "input"
    if "search" in w or "grounding" in w:
        return "web_search"
    return "other"


def label_value(name: str) -> str:
    """A backend name as a cloud label value: lowercase letters, digits, - and _, at most 63 (GCP's rule, the strictest)."""
    return re.sub(r"[^a-z0-9_-]", "-", name.lower())[:63]


def backend_type(b: BackendConfig) -> str | None:
    """Which bill a backend's requests land on."""
    host = (urlparse(b.base_url).hostname or "").lower()
    if b.flavor == "vertex" or host.endswith(("aiplatform.googleapis.com", "generativelanguage.googleapis.com")):  # the Gemini API bills a Cloud project
        return "gcp"
    if b.auth.type == "aws_sigv4" or host.endswith(".api.aws") or ("bedrock" in host and host.endswith(".amazonaws.com")):
        return "aws"
    if host == "api.anthropic.com":
        return "anthropic"
    if host == "api.openai.com":
        return "openai"
    if host.endswith(".azure.com"):
        return "azure"
    return None


def bill_of(source_type: str) -> str:
    """Which provider's bill a source type reads: aws_cur is AWS's, azure_export Azure's, the others their own."""
    return BILL_OF.get(source_type, source_type)


def covered_backends(src: BillSource, cfg: Config) -> list[str]:
    """The ledger backends a bill source covers: the ones it names, or every API backend of its type."""
    if src.backends is not None:
        return list(src.backends)
    return [b.name for b in cfg.backends if b.kind == "api" and not uses_reported_cost(b) and backend_type(b) == bill_of(src.type)]


def _apart(a: BillSource, b: BillSource) -> bool:
    """Whether two sources of one bill read different parts of it: disjoint linked accounts, or Azure scopes neither of
    which holds the other."""
    if a.accounts and b.accounts:
        return not set(a.accounts) & set(b.accounts)
    if a.scope and b.scope:
        x, y = (s.rstrip("/").lower() + "/" for s in (a.scope, b.scope))
        return not (x.startswith(y) or y.startswith(x))
    return False


def double_counted(cfg: Config) -> list[str]:
    """Enabled sources that read one bill two ways (Cost Explorer and the CUR export, or Azure's Query API and its
    export) for the same backends or GPU deployments. Each day would count twice, in the totals, the alerts and the GPU
    comparison, so the operator picks one (or splits them with ``backends`` and ``gpu_tag``)."""
    on = [s for s in cfg.billing.sources if s.enabled]
    gpus = bool(gpu_deployments(cfg))
    out = []
    for i, a in enumerate(on):
        for b in on[i + 1:]:
            if a.type == b.type or bill_of(a.type) != bill_of(b.type) or _apart(a, b):
                continue
            shared = sorted(set(covered_backends(a, cfg)) & set(covered_backends(b, cfg)))
            what = [f"backends {', '.join(shared)}"] if shared else []
            if gpus and a.gpu_tag and b.gpu_tag:
                what.append("GPU instances")
            if what:
                out.append(f"billing.sources.{a.name} ({a.type}) and billing.sources.{b.name} ({b.type}) both read the "
                           f"{bill_of(a.type).upper()} bill for {' and '.join(what)}, so every day would count twice: "
                           "turn one off (enabled: false), or split them with backends: and gpu_tag: null")
    return out


def gpu_deployments(cfg: Config) -> dict[str, str]:
    """Label value -> backend, for the GPU deployments SkyPilot runs (static servers have no cloud bill)."""
    return {label_value(b.name): b.name for b in cfg.backends
            if b.kind == "self_hosted" and b.serve is not None and b.serve.provisioner == "skypilot"}


def source_auth(src: BillSource) -> BackendAuth:
    """The credentials a source uses: its own ``auth`` if set, else the type's default."""
    defaults = {
        "anthropic": BackendAuth(type="api_key", header="x-api-key", value_env="ANTHROPIC_ADMIN_KEY"),
        "openai": BackendAuth(type="api_key", value_env="OPENAI_ADMIN_KEY"),
        "aws": BackendAuth(type="aws_sigv4", service="ce", region="us-east-1"),
        "aws_cur": BackendAuth(type="aws_sigv4", service="s3", region="us-east-1"),
        "gcp": BackendAuth(type="google"),
        "azure": BackendAuth(type="azure_ad", scope="https://management.azure.com/.default"),
        "azure_export": BackendAuth(type="azure_ad", scope=STORAGE_SCOPE),
    }
    a = src.auth
    if a is None or a.type == "none":
        return defaults[src.type]
    if src.type == "anthropic" and a.type == "api_key" and "header" not in a.model_fields_set:
        a = a.model_copy(update={"header": "x-api-key"})  # admin keys go in x-api-key; an OAuth token can set authorization
    if src.type == "aws":
        # Cost Explorer has one endpoint per partition, whatever region the backends use.
        region = "cn-northwest-1" if (a.region or "").startswith("cn-") else "us-east-1"
        a = a.model_copy(update={"type": "aws_sigv4", "service": "ce", "region": region})
    if src.type == "aws_cur":
        # S3, signed for the bucket's region (auth.region), with the same credential chain as Bedrock.
        a = a.model_copy(update={"type": "aws_sigv4", "service": "s3", "region": a.region or "us-east-1"})
    if src.type == "azure" and a.type == "azure_ad" and not a.scope:
        a = a.model_copy(update={"scope": "https://management.azure.com/.default"})
    if src.type == "azure_export" and a.type == "azure_ad" and not a.scope:
        a = a.model_copy(update={"scope": STORAGE_SCOPE})  # an api_key here is a SAS token, appended to each URL
    return a


def _chunks(start: date, end: date, days: int) -> list[tuple[date, date]]:
    out, lo = [], start
    while lo < end:
        hi = min(lo + timedelta(days=days), end)
        out.append((lo, hi))
        lo = hi
    return out


def _iso(d: date) -> str:
    return f"{d.isoformat()}T00:00:00Z"


def _why(r: httpx.Response) -> str:
    """The provider's own error message, short."""
    try:
        body = r.json()
    except ValueError:
        return r.text[:200]
    err = body.get("error") if isinstance(body, dict) else None
    if isinstance(err, dict):
        return str(err.get("message") or err.get("code") or err)[:200]
    if isinstance(body, dict):
        return str(body.get("message") or body.get("Message") or err or body)[:200]
    return str(body)[:200]


async def _send(client: httpx.AsyncClient, auth, what: str, method: str, url: str, params: list | None = None,
                body: dict | None = None, headers: dict | None = None) -> httpx.Response:
    full = str(httpx.URL(url, params=params)) if params else url
    data = json.dumps(body).encode() if body is not None else b""
    hdrs = {"accept": "application/json", **({"content-type": "application/json"} if body is not None else {}), **(headers or {})}
    hdrs = await auth.apply(method, full, hdrs, data)
    try:
        return await client.request(method, full, content=data or None, headers=hdrs)
    except httpx.HTTPError as e:
        raise BillError(f"{what}: {type(e).__name__}: {e}"[:300]) from e


def _check(r: httpx.Response, what: str) -> dict:
    if r.status_code >= 400:
        raise BillError(f"{what}: HTTP {r.status_code}: {_why(r)}", code=str(r.status_code))
    if r.status_code == 204 or not r.content:
        return {}
    return r.json()


# ---------------------------------------------------------------- Anthropic


async def _anthropic_rows(client, auth, url: str, params: list) -> tuple[list[dict], datetime | None]:
    """Every row of one analytics query, across pages. A cursor expires when the data refreshes (410): start over once."""
    headers = {"anthropic-version": ANTHROPIC_VERSION, "anthropic-beta": ANTHROPIC_BETA}
    for attempt in range(2):
        rows: list[dict] = []
        page, refreshed = None, None
        for _ in range(MAX_PAGES):
            r = await _send(client, auth, "Anthropic analytics", "GET", url, params + ([("page", page)] if page else []), headers=headers)
            if r.status_code == 410 and page and attempt == 0:
                break
            body = _check(r, "Anthropic analytics")
            rows += body.get("data") or []
            if body.get("data_refreshed_at"):
                refreshed = datetime.fromisoformat(body["data_refreshed_at"].replace("Z", "+00:00"))
            page = body.get("next_page") if body.get("has_more") else None
            if not page:
                return rows, refreshed
        else:
            raise BillError("Anthropic analytics: too many pages")
    raise BillError("Anthropic analytics: the data kept refreshing while it was read")  # pragma: no cover (needs two 410s in a row)


def _anthropic_text(model, kind_name, g: dict) -> str:
    parts = [model or "(no model)", kind_name, g.get("service_tier"), g.get("context_window"),
             "fast" if g.get("speed") == "fast" else None, g.get("product")]
    return " · ".join(str(p) for p in parts if p)


async def fetch_anthropic(src: BillSource, start: date, end: date, client: httpx.AsyncClient, auth) -> Fetched:
    """Cost by model, token type, tier, context window, speed and product per day; tokens from the usage endpoint."""
    base = (src.base_url or "https://api.anthropic.com").rstrip("/")
    filters = [("workspace_ids[]", w) for w in src.workspace_ids] + [("api_key_ids[]", k) for k in src.api_key_ids]
    groups = [("group_by[]", g) for g in ANTHROPIC_GROUPS]
    lines: dict[tuple, dict] = {}
    complete: list[date] = []
    # The API has data from 2026 on, and reads at most 31 days per request (lookback_days keeps within its 365).
    start = max(start, date(2026, 1, 1))
    for lo, hi in _chunks(start, end, 31):
        common = [("starting_at", _iso(lo)), ("ending_at", _iso(hi)), ("time_bucket_width", "1d"), ("limit", "1000"),
                  ("order_by", "time"), ("order", "asc"), *filters]
        cost, refreshed = await _anthropic_rows(client, auth, f"{base}/v1/organizations/analytics/cost",
                                                common + groups + [("group_by[]", "cost_type"), ("group_by[]", "token_type")])
        usage, _ = await _anthropic_rows(client, auth, f"{base}/v1/organizations/analytics/usage", common + groups)
        complete.append(refreshed.date() - timedelta(days=1) if refreshed else lo - timedelta(days=1))
        for row in cost:
            g, c = row.get("group") or {}, row.get("cost") or {}
            day = date.fromisoformat(row["starting_at"][:10])
            ctype, ttype = g.get("cost_type"), g.get("token_type")
            kind = ANTHROPIC_TOKENS.get(ttype, "other") if ctype == "tokens" else ("web_search" if ctype == "web_search" else "other")
            batch = g.get("service_tier") == "batch" or g.get("product") == "message_batches_api"
            text = _anthropic_text(g.get("model"), ttype or ctype or "cost", g)
            if (day, text) in lines:  # the same combination in another workspace or chunk
                lines[(day, text)]["cost"] += float(c.get("amount") or 0) / 100
                continue
            lines[(day, text)] = line(day, text, float(c.get("amount") or 0) / 100, kind, model=g.get("model"),
                                      tier="batch" if batch else g.get("service_tier"), product=g.get("product"),
                                      context_window=g.get("context_window"), speed=g.get("speed"), token_type=ttype,
                                      list_cost=float(c["list_amount"]) / 100 if c.get("list_amount") else None,
                                      currency=(c.get("currency") or "USD").upper())
        for row in usage:
            g, u = row.get("group") or {}, row.get("usage") or {}
            day = date.fromisoformat(row["starting_at"][:10])
            counts = {"uncached_input_tokens": u.get("uncached_input_tokens"), "output_tokens": u.get("output_tokens"),
                      "cache_read_input_tokens": u.get("cache_read_input_tokens"),
                      **{f"cache_creation.{k}": v for k, v in (u.get("cache_creation") or {}).items()}}
            for ttype, n in counts.items():
                hit = lines.get((day, _anthropic_text(g.get("model"), ttype, g)))
                if hit is not None and n is not None:
                    hit["tokens"] = (hit["tokens"] or 0) + int(n)
    return Fetched(list(lines.values()), min(complete) if complete else None)


# ---------------------------------------------------------------- OpenAI


async def fetch_openai(src: BillSource, start: date, end: date, client: httpx.AsyncClient, auth) -> Fetched:
    """Cost per day and line item ("gpt-6-astra, input_tokens"); the quantity is the token count."""
    base = (src.base_url or "https://api.openai.com").rstrip("/")
    ts = lambda d: str(int(datetime.combine(d, datetime.min.time(), tzinfo=UTC).timestamp()))  # noqa: E731
    params = [("start_time", ts(start)), ("end_time", ts(end)), ("bucket_width", "1d"), ("group_by", "line_item"), ("limit", "180"),
              *[("project_ids", p) for p in src.project_ids], *[("api_key_ids", k) for k in src.api_key_ids]]
    out, page = [], None
    for _ in range(MAX_PAGES):
        r = await _send(client, auth, "OpenAI costs", "GET", f"{base}/v1/organization/costs", params + ([("page", page)] if page else []))
        body = _check(r, "OpenAI costs")
        for bucket in body.get("data") or []:
            day = datetime.fromtimestamp(int(bucket["start_time"]), UTC).date()
            for res in bucket.get("results") or []:
                amount = res.get("amount") or {}
                item = res.get("line_item") or "(no line item)"
                model, _, what = item.partition(",")
                what = what.strip()
                w = set(re.findall(r"[a-z]+", what.lower()))
                kind = ("cache_read" if "cached" in w else "input" if "input" in w else "output" if "output" in w else "other") if what else "other"
                unit, qty = res.get("quantity_unit"), res.get("quantity")
                tokens = int(qty) if qty is not None and unit == "tokens" else int(qty * 1000) if qty is not None and unit == "1000_tokens" else None
                out.append(line(day, item, float(amount.get("value") or 0), kind, model=model.strip() if what else None, tokens=tokens,
                                tier="batch" if "batch" in item.lower() else None, currency=(amount.get("currency") or "usd").upper(),
                                project_id=res.get("project_id")))
        page = body.get("next_page") if body.get("has_more") else None
        if not page:
            return Fetched(out)
    raise BillError("OpenAI costs: too many pages")


# ---------------------------------------------------------------- AWS Cost Explorer


def _ce_url(region: str) -> str:
    return f"https://ce.{region}.amazonaws.com{'.cn' if region.startswith('cn-') else ''}/"


async def _ce(client, auth, region: str, op: str, body: dict) -> list[dict]:
    """Every page of one Cost Explorer call (each page costs $0.01). Returns the pages' bodies."""
    pages, token = [], None
    for _ in range(MAX_PAGES):
        req = {**body, **({"NextPageToken": token} if token else {})}
        r = await _send(client, auth, f"Cost Explorer {op}", "POST", _ce_url(region), body=req,
                        headers={"content-type": "application/x-amz-json-1.1", "x-amz-target": f"AWSInsightsIndexService.{op}"})
        if r.status_code >= 400:
            try:
                err = r.json()
            except ValueError:
                err = {}
            code = str(err.get("__type") or err.get("code") or r.status_code).rsplit("#", 1)[-1]
            raise BillError(f"Cost Explorer {op}: {code}: {err.get('message') or err.get('Message') or r.text[:200]}", code=code)
        page = r.json()
        pages.append(page)
        token = page.get("NextPageToken")
        if not token:
            return pages
    raise BillError(f"Cost Explorer {op}: too many pages")


def _ce_and(*exprs: dict | None) -> dict | None:
    exprs = tuple(e for e in exprs if e)
    return None if not exprs else exprs[0] if len(exprs) == 1 else {"And": list(exprs)}


def _ce_groups(pages: list[dict], types: list[str]):
    """(day, [group values], metrics, estimated) for every group. TAG and PRODUCT_ATTRIBUTE keys come as "key$value"."""
    for page in pages:
        for rbt in page.get("ResultsByTime") or []:
            day = date.fromisoformat(rbt["TimePeriod"]["Start"][:10])
            for g in rbt.get("Groups") or []:
                keys = [k.split("$", 1)[1] if t != "DIMENSION" and "$" in k else k for k, t in zip(g.get("Keys") or [], types)]
                yield day, keys, g.get("Metrics") or {}, bool(rbt.get("Estimated"))


def _amount(metrics: dict, name: str) -> tuple[float, str | None]:
    m = metrics.get(name) or {}
    return float(m.get("Amount") or 0), m.get("Unit")


async def fetch_aws(src: BillSource, cfg: Config, start: date, end: date, client: httpx.AsyncClient, auth) -> Fetched:
    """Bedrock by model and inference type (Cost Explorer's product attributes, from September 2026 on); credits and other
    record types by model; GPU instances by tag. Periods without the model breakdown fall back to service and usage type."""
    region = source_auth(src).region or "us-east-1"
    today = datetime.now(UTC).date()
    period = {"Start": start.isoformat(), "End": min(end, today + timedelta(days=1)).isoformat()}
    usage = {"Dimensions": {"Key": "RECORD_TYPE", "Values": ["Usage"]}}
    accounts = {"Dimensions": {"Key": "LINKED_ACCOUNT", "Values": list(src.accounts)}} if src.accounts else None
    out, notes = [], []

    def query(group_by: list[dict], filt: dict | None, metrics: list[str]) -> dict:
        return {"TimePeriod": period, "Granularity": "DAILY", "Metrics": metrics, "GroupBy": group_by, **({"Filter": filt} if filt else {})}

    by_model = [{"Type": "PRODUCT_ATTRIBUTE", "Key": "model"}, {"Type": "PRODUCT_ATTRIBUTE", "Key": "inferenceType"}]
    try:
        pages = await _ce(client, auth, region, "GetCostAndUsage", query(by_model, _ce_and(usage, accounts), ["UnblendedCost", "UsageQuantity"]))
        for day, (model, inference), metrics, estimated in _ce_groups(pages, ["PRODUCT_ATTRIBUTE", "PRODUCT_ATTRIBUTE"]):
            cost, _ = _amount(metrics, "UnblendedCost")
            qty, unit = _amount(metrics, "UsageQuantity")
            text = f"{model or '(no model)'} · {inference or '(no inference type)'}"
            tokens = int(qty) if unit and "token" in unit.lower() else None
            out.append(line(day, text, cost, kind_of(inference or ""), model=model or None, tokens=tokens,
                            tier="batch" if "batch" in text.lower() else None, quantity=qty, unit=unit, estimated=estimated or None))
        adjust = [{"Type": "DIMENSION", "Key": "RECORD_TYPE"}, {"Type": "PRODUCT_ATTRIBUTE", "Key": "model"}]
        pages = await _ce(client, auth, region, "GetCostAndUsage", query(adjust, _ce_and({"Not": usage}, accounts), ["UnblendedCost"]))
        for day, (record, model), metrics, _ in _ce_groups(pages, ["DIMENSION", "PRODUCT_ATTRIBUTE"]):
            cost, _ = _amount(metrics, "UnblendedCost")
            if cost:
                out.append(line(day, f"{record} · {model or '(no model)'}", cost, "adjustment", model=model or None, record_type=record))
    except BillError as e:
        if e.code != "DataUnavailableException":
            raise
        notes.append("Cost Explorer has no model breakdown for part of this period (it starts in September 2026), so lines are "
                     "matched on service and usage type names; credits weren't fetched.")
        found = await _ce(client, auth, region, "GetDimensionValues",
                          {"TimePeriod": period, "Dimension": "SERVICE", "SearchString": "Bedrock", "Context": "COST_AND_USAGE"})
        services = sorted({v["Value"] for p in found for v in p.get("DimensionValues") or []})
        if services:
            by_type = [{"Type": "DIMENSION", "Key": "SERVICE"}, {"Type": "DIMENSION", "Key": "USAGE_TYPE"}]
            filt = _ce_and({"Dimensions": {"Key": "SERVICE", "Values": services}}, usage, accounts)
            pages = await _ce(client, auth, region, "GetCostAndUsage", query(by_type, filt, ["UnblendedCost", "UsageQuantity"]))
            for day, (service, usage_type), metrics, estimated in _ce_groups(pages, ["DIMENSION", "DIMENSION"]):
                cost, _ = _amount(metrics, "UnblendedCost")
                qty, unit = _amount(metrics, "UsageQuantity")
                text = f"{service} · {usage_type}"
                out.append(line(day, text, cost, kind_of(usage_type), tier="batch" if "batch" in text.lower() else None,
                                quantity=qty, unit=unit, estimated=estimated or None))
    if src.gpu_tag and gpu_deployments(cfg):
        by_tag = [{"Type": "TAG", "Key": src.gpu_tag}, {"Type": "DIMENSION", "Key": "SERVICE"}]
        pages = await _ce(client, auth, region, "GetCostAndUsage", query(by_tag, _ce_and(usage, accounts), ["UnblendedCost"]))
        for day, (value, service), metrics, estimated in _ce_groups(pages, ["TAG", "DIMENSION"]):
            cost, _ = _amount(metrics, "UnblendedCost")
            if value and cost:
                out.append(line(day, f"{value} · {service}", cost, "gpu", deployment=value, estimated=estimated or None))
    return Fetched(out, notes=notes)


# ---------------------------------------------------------------- Google Cloud: billing export in BigQuery

_TABLE = re.compile(r"^[A-Za-z0-9_:-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_$-]+$")

GCP_SQL = """
WITH lines AS (
  SELECT DATE(usage_start_time) AS day, project.id AS project, service.description AS service, sku.description AS sku,
         cost_type, (SELECT l.value FROM UNNEST(labels) AS l WHERE l.key = @gpu_label LIMIT 1) AS deployment,
         cost / IFNULL(NULLIF(currency_conversion_rate, 0), 1) AS cost_usd,
         IFNULL((SELECT SUM(c.amount) FROM UNNEST(credits) AS c), 0) / IFNULL(NULLIF(currency_conversion_rate, 0), 1) AS credits_usd,
         usage.amount AS quantity, usage.unit AS unit
  FROM `{table}`
  WHERE usage_start_time >= @start AND usage_start_time < @end{partition}
)
SELECT day, service, sku, cost_type, deployment, SUM(cost_usd) AS cost, SUM(credits_usd) AS credits,
       SUM(quantity) AS quantity, ANY_VALUE(unit) AS unit
FROM lines
WHERE (REGEXP_CONTAINS(service, @services) AND (ARRAY_LENGTH(@projects) = 0 OR project IN UNNEST(@projects)))
   OR deployment IS NOT NULL
GROUP BY day, service, sku, cost_type, deployment
ORDER BY day
"""


def _bq_param(name: str, kind: str, value) -> dict:
    if kind == "ARRAY":
        return {"name": name, "parameterType": {"type": "ARRAY", "arrayType": {"type": "STRING"}},
                "parameterValue": {"arrayValues": [{"value": v} for v in value]}}
    return {"name": name, "parameterType": {"type": kind}, "parameterValue": {"value": value}}


def vertex_projects(cfg: Config, backends: list[str]) -> list[str]:
    """The GCP projects the Vertex backends call: `project:`, or the one in the base URL."""
    out = set()
    for b in cfg.backends:
        if b.name in backends:
            if b.project:
                out.add(b.project)
            m = re.search(r"/projects/([^/]+)/", b.base_url)
            if m:
                out.add(m.group(1))
    return sorted(out)


async def fetch_gcp(src: BillSource, cfg: Config, start: date, end: date, client: httpx.AsyncClient, auth) -> Fetched:
    """One query over the standard usage cost export: cost and credits per day, service and SKU, in US dollars."""
    if not src.table or not _TABLE.match(src.table):
        raise BillError(f"billing.sources.{src.name}.table must be the export table as project.dataset.table, got {src.table!r}")
    project = src.query_project or src.table.split(".", 1)[0]
    projects = list(src.project_ids) or vertex_projects(cfg, covered_backends(src, cfg))
    # Late rows land in later partitions, so the partition filter starts a day early and has no upper bound.
    partition = "\n    AND _PARTITIONTIME >= @partition_start" if src.partitioned else ""
    params = [_bq_param("start", "TIMESTAMP", f"{start.isoformat()} 00:00:00+00"), _bq_param("end", "TIMESTAMP", f"{end.isoformat()} 00:00:00+00"),
              _bq_param("services", "STRING", src.services), _bq_param("projects", "ARRAY", projects),
              _bq_param("gpu_label", "STRING", src.gpu_tag or "")]
    if src.partitioned:
        params.append(_bq_param("partition_start", "TIMESTAMP", f"{(start - timedelta(days=1)).isoformat()} 00:00:00+00"))
    body = {"query": GCP_SQL.format(table=src.table, partition=partition), "useLegacySql": False, "parameterMode": "NAMED",
            "queryParameters": params, "timeoutMs": 60000, **({"location": src.location} if src.location else {})}
    api = f"https://bigquery.googleapis.com/bigquery/v2/projects/{project}/queries"
    res = _check(await _send(client, auth, "BigQuery", "POST", api, body=body), "BigQuery")
    job = res.get("jobReference") or {}
    where = [("location", job.get("location") or src.location)] if job.get("location") or src.location else []
    for _ in range(MAX_PAGES):  # jobs.getQueryResults waits up to timeoutMs on the server for a job that's still running
        if res.get("jobComplete") is not False:
            break
        res = _check(await _send(client, auth, "BigQuery", "GET", f"{api}/{job.get('jobId')}", [("timeoutMs", "60000"), *where]), "BigQuery")
    else:
        raise BillError("BigQuery: the query didn't finish")
    schema, rows = res.get("schema") or {}, list(res.get("rows") or [])
    for _ in range(MAX_PAGES):
        if not res.get("pageToken"):
            break
        res = _check(await _send(client, auth, "BigQuery", "GET", f"{api}/{job.get('jobId')}",
                                 [("pageToken", res["pageToken"]), *where]), "BigQuery")
        rows += res.get("rows") or []
    else:
        raise BillError("BigQuery: too many pages")
    names = [f["name"] for f in schema.get("fields") or []]
    out = []
    for r in rows:
        v = dict(zip(names, (c.get("v") for c in r.get("f") or [])))
        day = date.fromisoformat(str(v["day"])[:10])
        service, sku, ctype = v.get("service") or "", v.get("sku") or "", (v.get("cost_type") or "regular").lower()
        cost, credits = float(v.get("cost") or 0), float(v.get("credits") or 0)
        qty = float(v["quantity"]) if v.get("quantity") is not None else None
        if v.get("deployment"):
            out.append(line(day, f"{v['deployment']} · {service} · {sku}", cost, "gpu", deployment=v["deployment"],
                            credits=credits or None))
            continue
        if ctype != "regular":
            out.append(line(day, f"{service} · {sku} · {ctype}", cost + credits, "adjustment", cost_type=ctype))
            continue
        text = f"{service} · {sku}"
        out.append(line(day, text, cost, kind_of(sku), tier="batch" if "batch" in sku.lower() else None, quantity=qty, unit=v.get("unit")))
        if credits:
            out.append(line(day, f"{text} · credits", credits, "adjustment"))
    return Fetched(out)


# ---------------------------------------------------------------- Azure Cost Management


async def _azure_rows(client, auth, url: str, body: dict) -> tuple[list[str], list[list]]:
    columns, rows = [], []
    for _ in range(MAX_PAGES):
        res = _check(await _send(client, auth, "Azure Cost Management", "POST", url, body=body), "Azure Cost Management")
        props = res.get("properties") or {}
        columns = [c["name"] for c in props.get("columns") or []] or columns
        rows += props.get("rows") or []
        url = props.get("nextLink")
        if not url:
            return columns, rows
    raise BillError("Azure Cost Management: too many pages")


def _azure_day(v) -> date:
    s = str(v)
    return date(int(s[:4]), int(s[4:6]), int(s[6:8])) if s.isdigit() else date.fromisoformat(s[:10])


async def fetch_azure(src: BillSource, cfg: Config, start: date, end: date, client: httpx.AsyncClient, auth) -> Fetched:
    """Actual cost per day, service and meter at the configured scope; GPU instances by tag."""
    if not src.scope or not src.scope.startswith("/"):
        raise BillError(f"billing.sources.{src.name}.scope must be set, e.g. /subscriptions/<id> or /subscriptions/<id>/resourceGroups/<name>")
    url = f"https://management.azure.com{src.scope.rstrip('/')}/providers/Microsoft.CostManagement/query?api-version={AZURE_API_VERSION}"
    period = {"from": f"{start.isoformat()}T00:00:00Z", "to": f"{(end - timedelta(days=1)).isoformat()}T23:59:59Z"}
    aggregation = {"totalCost": {"name": src.cost_column, "function": "Sum"}}

    def body(grouping: list[dict], filt: dict | None) -> dict:
        return {"type": "ActualCost", "timeframe": "Custom", "timePeriod": period,
                "dataset": {"granularity": "Daily", "aggregation": aggregation, "grouping": grouping, **({"filter": filt} if filt else {})}}

    def cols(names: list[str], *wanted: str) -> int | None:
        low = [n.lower() for n in names]
        return next((low.index(w.lower()) for w in wanted if w.lower() in low), None)

    usd = src.cost_column.upper().endswith("USD")
    filt = {"dimensions": {"name": "ResourceType", "operator": "In", "values": list(src.resource_types)}} if src.resource_types else None
    names, rows = await _azure_rows(client, auth, url, body([{"type": "Dimension", "name": "ServiceName"},
                                                            {"type": "Dimension", "name": "Meter"}], filt))
    ci, di = cols(names, src.cost_column, "totalCost", "Cost", "PreTaxCost"), cols(names, "UsageDate")
    si, mi, cur = cols(names, "ServiceName"), cols(names, "Meter"), cols(names, "Currency")
    if ci is None or di is None:
        raise BillError(f"Azure Cost Management: no cost or date column in the reply ({', '.join(names)})")
    out = []
    for r in rows:
        service = str(r[si]) if si is not None else ""
        meter = str(r[mi]) if mi is not None else ""
        text = f"{service} · {meter}"
        currency = "USD" if usd else str(r[cur]).upper() if cur is not None and r[cur] else "USD"
        out.append(line(_azure_day(r[di]), text, float(r[ci] or 0), kind_of(meter), tier="batch" if "batch" in text.lower() else None,
                        currency=currency))
    deployments = gpu_deployments(cfg)
    if src.gpu_tag and deployments:
        tag_filter = {"tags": {"name": src.gpu_tag, "operator": "In", "values": sorted(deployments)}}
        names, rows = await _azure_rows(client, auth, url, body([{"type": "TagKey", "name": src.gpu_tag},
                                                                {"type": "Dimension", "name": "ServiceName"}], tag_filter))
        ci, di = cols(names, src.cost_column, "totalCost", "Cost", "PreTaxCost"), cols(names, "UsageDate")
        vi, si, cur = cols(names, "TagValue", src.gpu_tag), cols(names, "ServiceName"), cols(names, "Currency")
        for r in rows:
            value = str(r[vi]) if vi is not None and r[vi] else ""
            if value and ci is not None and di is not None:
                service = str(r[si]) if si is not None else ""
                currency = "USD" if usd else str(r[cur]).upper() if cur is not None and r[cur] else "USD"
                out.append(line(_azure_day(r[di]), f"{value} · {service}", float(r[ci] or 0), "gpu", deployment=value, currency=currency))
    return Fetched(out)


async def fetch(src: BillSource, cfg: Config, start: date, end: date, client: httpx.AsyncClient, auth,
                state: dict | None = None) -> Fetched:
    """Bill lines for [start, end) from one source. ``state`` is what an export source read last time (``Fetched.state``)."""
    if src.type in ("aws_cur", "azure_export"):
        from .exports import fetch_export  # here: exports builds on this module

        return await fetch_export(src, cfg, start, end, client, auth, state or {})
    if src.type == "anthropic":
        return await fetch_anthropic(src, start, end, client, auth)
    if src.type == "openai":
        return await fetch_openai(src, start, end, client, auth)
    if src.type == "aws":
        return await fetch_aws(src, cfg, start, end, client, auth)
    if src.type == "gcp":
        return await fetch_gcp(src, cfg, start, end, client, auth)
    return await fetch_azure(src, cfg, start, end, client, auth)
