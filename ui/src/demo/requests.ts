/** The demo's Requests view (GET /v1/reports/requests and requests.csv): the gateway's filters, sorts and keyset paging
 * (store.RequestFilter, reports.py), over the recorded rows and the ones the demo's traffic adds. */

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Json = any;
type Answer = { status: number; body: Json; type?: string };

const SAVINGS = ["routing", "caching", "self_hosting", "batch", "compression", "compression_model", "response_cache"];
const SORT_DAYS = 7;
const savings = (r: Json) => SAVINGS.reduce((s, c) => s + (r[`savings_${c}`] ?? 0), 0);
const sortValue = (r: Json, sort: string): number =>
  sort === "cost" ? r.actual_cost ?? 0 : sort === "savings" ? savings(r) : sort === "latency" ? r.latency_ms ?? -1 : Date.parse(r.ts);
const bad = (detail: string): Answer => ({ status: 400, body: { detail } });

function time(v: string | null): number | null {
  if (!v) return null;
  const t = Date.parse(/T| /.test(v) ? v : `${v}T00:00:00Z`);
  return Number.isNaN(t) ? null : t;
}

function statusMatches(want: string, status: number): boolean {
  if (want === "success") return status < 400;
  if (want === "error") return status >= 400;
  if (want.endsWith("xx")) return Math.floor(status / 100) === Number(want[0]);
  return status === Number(want);
}

/** The filter as a predicate, or an error the gateway would give. */
function filter(q: URLSearchParams): ((r: Json) => boolean) | string {
  const start = time(q.get("start")), end = time(q.get("end"));
  const status = q.get("status");
  if (status && !/^(success|error|[1-5]xx|[1-5][0-9]{2})$/.test(status)) return `status must be success, error, a code (429) or a class (5xx), got '${status}'`;
  const eq: [string, string][] = [["user", "user_id"], ["team", "team"], ["key", "key_id"], ["session", "session_id"], ["requested_model", "requested_model"],
    ["served_model", "served_model"], ["backend", "backend"], ["format", "client_format"]];
  const tags = q.getAll("tag").map((t) => t.split("=") as [string, string?]);
  const num = (k: string) => (q.get(k) ? Number(q.get(k)) : null);
  const yes = (k: string) => (q.get(k) === "true" ? true : q.get(k) === "false" ? false : null);
  const [minC, maxC, minS, maxS, minL, maxL] = ["min_cost", "max_cost", "min_savings", "max_savings", "min_latency", "max_latency"].map(num);
  const reason = q.get("reason")?.toLowerCase();
  return (r: Json) => {
    const ts = Date.parse(r.ts);
    const [task, level] = (r.floor ?? "").split(":");
    const hit = (r.cache_read_tokens ?? 0) > 0 || r.backend_kind === "cache";
    return (start == null || ts >= start) && (end == null || ts < end)
      && eq.every(([k, col]) => !q.get(k) || r[col] === q.get(k))
      && (!status || statusMatches(status, r.status ?? 0))
      && (!q.get("task") || (!!r.floor && task === q.get("task"))) && (!q.get("level") || (!!r.floor && level === q.get("level")))
      && (!reason || (r.route_reason ?? "").toLowerCase().includes(reason))
      && tags.every(([k, v]) => (v === undefined ? k in (r.tags ?? {}) : r.tags?.[k] === v))
      && (minC == null || r.actual_cost >= minC) && (maxC == null || r.actual_cost <= maxC)
      && (minS == null || savings(r) >= minS) && (maxS == null || savings(r) <= maxS)
      && (!q.get("savings_cause") || (r[`savings_${q.get("savings_cause")}`] ?? 0) !== 0)
      && (minL == null || (r.latency_ms != null && r.latency_ms >= minL)) && (maxL == null || (r.latency_ms != null && r.latency_ms <= maxL))
      && (yes("stream") == null || !!r.stream === yes("stream")) && (yes("batch") == null || (r.service_tier === "batch") === yes("batch"))
      && (yes("cache_hit") == null || hit === yes("cache_hit")) && (yes("failover") == null || !!r.failed_over === yes("failover"));
  };
}

type Key = [number, number, string];
const keyOf = (r: Json, sort: string): Key => [sortValue(r, sort), Date.parse(r.ts), r.id];
const cmp = (a: Key, b: Key) => (a[0] - b[0]) || (a[1] - b[1]) || (a[2] < b[2] ? -1 : a[2] > b[2] ? 1 : 0);
const cursor = (k: Key, sort: string, order: string, next: boolean) => btoa(JSON.stringify({ k, s: sort, o: order, d: next ? "n" : "p" }));

function sorted(all: Json[], q: URLSearchParams): { rows: Json[]; sort: string; order: string } | string {
  const sort = q.get("sort") ?? "time", order = q.get("order") ?? "desc";
  const f = filter(q);
  if (typeof f === "string") return f;
  if (sort !== "time") {
    const start = time(q.get("start")), end = time(q.get("end")) ?? Date.now();
    if (start == null || end - start > SORT_DAYS * 864e5) return `sorting by ${sort} needs a time range of ${SORT_DAYS} days or less (sorting by time works on any range)`;
  }
  const seen = new Set<string>();
  const rows = all.filter((r) => !seen.has(r.id) && seen.add(r.id) && f(r));
  rows.sort((a, b) => (order === "desc" ? -1 : 1) * cmp(keyOf(a, sort), keyOf(b, sort)));
  return { rows, sort, order };
}

export function requestsPage(all: Json[], q: URLSearchParams): Answer {
  const s = sorted(all, q);
  if (typeof s === "string") return bad(s);
  const { rows, sort, order } = s;
  const limit = Math.min(1000, Math.max(1, Number(q.get("limit") ?? 50)));
  const desc = order === "desc";
  let from = 0, at: Key | null = null;
  const c = q.get("cursor");
  if (c) {
    let doc: Json;
    try { doc = JSON.parse(atob(c)); } catch { doc = null; }
    if (!doc || doc.s !== sort || doc.o !== order) return bad("that cursor isn't one this view gave out for this sort and order: start from the first page");
    at = doc.k;
    const after = (r: Json) => (desc ? cmp(keyOf(r, sort), at!) < 0 : cmp(keyOf(r, sort), at!) > 0);
    const i = rows.findIndex(after);
    const firstAfter = i < 0 ? rows.length : i;
    if (doc.d === "n") from = firstAfter;
    else {
      const before = rows.slice(0, firstAfter).filter((r) => cmp(keyOf(r, sort), at!) !== 0).length;
      from = before - limit;
      if (from <= 0) { from = 0; at = null; }  // back at the start: the first page, whole
    }
  }
  const page = rows.slice(from, from + limit);
  const prev = at == null && from === 0 ? null : page.length ? cursor(keyOf(page[0], sort), sort, order, false) : at ? cursor(at, sort, order, false) : null;
  const next = from + limit < rows.length && page.length ? cursor(keyOf(page[page.length - 1], sort), sort, order, true) : null;
  return { status: 200, body: { rows: page, next, prev, limit, sort, order, total: rows.length, total_exact: true,
    user: q.get("user"), team: q.get("team"), start: q.get("start"), end: q.get("end"), sort_range_days: SORT_DAYS, answered_by: "memory" } };
}

const COLUMNS = ["id", "ts", "user_id", "team", "key_id", "session_id", "tags", "client_format", "requested_model", "floor", "route_reason",
  "served_model", "upstream_model", "backend", "backend_kind", "status", "error", "stream", "upgraded", "proxy_cache", "input_tokens",
  "cache_read_tokens", "cache_write_tokens", "output_tokens", "latency_ms", "ttft_ms", "actual_cost", "served_api_cost", "baseline_cost",
  "list_cost", ...SAVINGS.map((c) => `savings_${c}`), "service_tier", "batch_id", "failed_over", "savings_total"];

export function requestsCsv(all: Json[], q: URLSearchParams): Answer {
  const s = sorted(all, q);
  if (typeof s === "string") return bad(s);
  const cell = (v: Json) => {
    const t = v == null ? "" : typeof v === "object" ? JSON.stringify(v) : String(v);
    return /[",\n]/.test(t) ? `"${t.replace(/"/g, '""')}"` : t;
  };
  const lines = s.rows.map((r) => COLUMNS.map((c) => cell(c === "savings_total" ? +savings(r).toFixed(6) : r[c])).join(","));
  return { status: 200, type: "text/csv", body: [COLUMNS.join(","), ...lines].join("\r\n") + "\r\n" };
}
