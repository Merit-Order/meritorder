import { Badge, Table, Text } from "@mantine/core";
import type { BreakevenGpu, BreakevenGroup, MeritTier } from "../api";
import { pct, usd } from "../api";

// Charts read only tokens (styles.css, theme.ts): every series muted (--mo-bar) except the one that matters, in the
// accent; the categorical --mo-s1..s4 only for savings by cause. Marks carry `data-tip`, which TipLayer shows.

/** The top of a value axis: the next of 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8 (× a power of ten) at or above `v`, so the top
 * gridline sits close to the data instead of up to twice above it. */
export function niceMax(v: number): number {
  if (v <= 0) return 1;
  const exp = Math.pow(10, Math.floor(Math.log10(v)));
  for (const m of [1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]) if (v <= m * exp + 1e-9) return m * exp;
  return 10 * exp;
}

/** Axis money: $0, $750, $1,250, $2.50: whole dollars from $100, cents below. */
const axisUsd = (v: number) => (v === 0 ? "$0" : v >= 100 ? `$${Math.round(v).toLocaleString()}` : usd(v));

/** Gridline fractions of an axis topped by niceMax: steps that land on round numbers (1, 2, 4 and 8 in quarters; 1.5, 3 and
 * 6 in thirds; 2.5 and 5 in fifths). */
export function ticks(max: number): number[] {
  const m = +(max / Math.pow(10, Math.floor(Math.log10(max)))).toFixed(2);
  const n = [1.5, 3, 6].includes(m) ? 3 : [2.5, 5].includes(m) ? 5 : 4;
  return Array.from({ length: n + 1 }, (_, i) => i / n);
}

/** The merit-order stack: models in order of what a token actually cost; widths are tokens served. API tiers are muted,
 * the self-hosted tier is the accent; the clearing price is a dashed line in the text color. */
export function MeritChart({ tiers, demand, clearing }: { tiers: MeritTier[]; demand: number; clearing: number | null }) {
  const W = 880, H = 300, L = 64, R = 12, T = 30, B = 40;
  const total = tiers.reduce((s, t) => s + t.tokens_per_second, 0);
  const maxPrice = niceMax(Math.max(...tiers.map((t) => t.price_per_m), 0.01));
  const xs = (v: number) => L + (total ? (v / total) * (W - L - R) : 0);
  const ys = (p: number) => T + (H - T - B) - (p / maxPrice) * (H - T - B);
  let acc = 0;
  if (!tiers.length) {
    return <Text c="dimmed" size="sm" py="xl" ta="center">No traffic in this window yet. The stack fills in as requests arrive.</Text>;
  }
  return (
    <div className="chart-scroll">
      <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Merit order: models sorted by cost per million tokens, widths are tokens served">
        {ticks(maxPrice).map((f) => f * maxPrice).map((t) => (
          <g key={t}>
            <line className="grid" x1={L} x2={W - R} y1={ys(t)} y2={ys(t)} />
            <text x={L - 8} y={ys(t) + 3.5} textAnchor="end">{usd(t)}</text>
          </g>
        ))}
        <text x={14} y={(H - B + T) / 2} transform={`rotate(-90 14 ${(H - B + T) / 2})`} textAnchor="middle">$ per million tokens</text>
        <text x={L} y={H - 12}>0</text>
        <text x={(L + W - R) / 2} y={H - 12} textAnchor="middle">tokens per second served</text>
        <text x={W - R} y={H - 12} textAnchor="end">{Math.round(total).toLocaleString()}</text>
        {tiers.map((t) => {
          const x0 = xs(acc);
          acc += t.tokens_per_second;
          const x1 = xs(acc);
          const h = Math.max(2.5, H - B - ys(t.price_per_m));
          const label = t.gpu ? `GPU ${t.model}` : t.model;
          const fits = Math.floor((x1 - x0 - 6) / 6.7);
          return (
            <g key={t.model}>
              <rect x={x0 + 1} y={H - B - h} width={Math.max(1, x1 - x0 - 2)} height={h} rx={2} fill={t.gpu ? "var(--mo-accent)" : "var(--mo-bar)"}
                data-tip={`${t.model}${t.gpu ? " (self-hosted)" : ""}\n${usd(t.price_per_m)}/M tokens after caching\n${Math.round(t.tokens_per_second).toLocaleString()} tok/s · ${t.requests.toLocaleString()} requests`} />
              {x1 - x0 > 40 && (
                <text x={(x0 + x1) / 2} y={H - B - h - 7} textAnchor="middle" className={t.gpu ? "t-acc" : "t-fg2"} style={{ fontSize: 11 }}>
                  {label.length > fits ? `${label.slice(0, Math.max(fits - 1, 2))}…` : label}
                </text>
              )}
            </g>
          );
        })}
        {clearing != null && (
          <g>
            <line x1={L} x2={W - R} y1={ys(clearing)} y2={ys(clearing)} stroke="var(--mantine-color-text)" strokeDasharray="4 4" strokeWidth={1} />
            <rect x={L + 6} y={ys(clearing) - 21} width={186} height={16} rx={3} fill="var(--mo-surface)" />
            <text x={L + 10} y={ys(clearing) - 9} className="t-fg" style={{ fontSize: 11 }}>{`clearing price ${usd(clearing)}/M`}</text>
          </g>
        )}
        <text x={W - R} y={16} textAnchor="end" className="t-fg" style={{ fontSize: 11 }}>{`demand ${Math.round(demand).toLocaleString()} tok/s`}</text>
      </svg>
    </div>
  );
}

/** The stack as a table, under the chart (it doubles as the chart's table view). */
export function MeritTable({ tiers }: { tiers: MeritTier[] }) {
  const total = tiers.reduce((s, t) => s + t.tokens_per_second, 0) || 1;
  if (!tiers.length) return null;
  return (
    <Table.ScrollContainer minWidth={560} mt="sm" style={{ border: "1px solid var(--mo-line)", borderRadius: 8 }}>
      <Table className="dense-first">
        <Table.Thead>
          <Table.Tr>
            <Table.Th>Model</Table.Th><Table.Th ta="right">$/M after caching</Table.Th><Table.Th ta="right">Tok/s</Table.Th>
            <Table.Th ta="right">Share</Table.Th><Table.Th ta="right">Requests</Table.Th>
          </Table.Tr>
        </Table.Thead>
        <Table.Tbody>
          {tiers.map((t) => (
            <Table.Tr key={t.model}>
              <Table.Td>
                <span className="swatch" style={{ background: t.gpu ? "var(--mo-accent)" : "var(--mo-bar)", width: 8, height: 8, marginRight: 8 }} />
                {t.model}{t.gpu && <Badge ml={6} size="sm" color="mo">GPU</Badge>}
              </Table.Td>
              <Table.Td className="num">{usd(t.price_per_m)}</Table.Td>
              <Table.Td className="num">{Math.round(t.tokens_per_second).toLocaleString()}</Table.Td>
              <Table.Td className="num">{pct(t.tokens_per_second / total)}</Table.Td>
              <Table.Td className="num">{t.requests.toLocaleString()}</Table.Td>
            </Table.Tr>
          ))}
        </Table.Tbody>
      </Table>
    </Table.ScrollContainer>
  );
}

const day = (label: string) => new Date(`${label.slice(0, 10)}T00:00:00`).toLocaleDateString(undefined, { month: "short", day: "numeric" });

/** Daily bars: cost without Merit Order behind (muted), actual spend in front (the accent). */
/** Daily spend against the cost without Merit Order. ``gpu``: the part of each day's spend that was GPUs (decision 64),
 * drawn in the accent's lighter tint on top of what APIs cost. */
export function SpendBars({ days }: { days: { label: string; actual: number; gpu?: number; baseline: number }[] }) {
  const W = 1380, H = 260, L = 64, B = 30, T = 14, R = 8;
  const max = niceMax(Math.max(...days.map((d) => Math.max(d.actual, d.baseline)), 0));
  const plotW = W - L - R, plotH = H - T - B;
  const step = days.length ? plotW / days.length : plotW;
  const bw = Math.max(3, Math.min(44, step * 0.5));
  const y = (v: number) => T + plotH - (v / max) * plotH;
  const every = Math.ceil(days.length / 16);
  return (
    <div className="chart-scroll">
      <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Daily spend compared with the cost without Merit Order">
        {ticks(max).map((f) => (
          <g key={f}><line className={f ? "grid" : "axis"} x1={L} x2={W - R} y1={y(f * max)} y2={y(f * max)} /><text x={L - 10} y={y(f * max) + 3.5} textAnchor="end">{axisUsd(f * max)}</text></g>
        ))}
        {days.map((d, i) => {
          const x = L + i * step + (step - bw) / 2;
          const last = i === days.length - 1;
          return (
            <g key={d.label} data-tip={`${day(d.label)}\nSpent ${usd(d.actual)}${d.gpu ? ` (APIs ${usd(d.actual - d.gpu)}, GPUs ${usd(d.gpu)})` : ""}`
              + `\nWithout Merit Order ${usd(d.baseline)}\nSaved ${usd(d.baseline - d.actual)}`}>
              <rect x={L + i * step} width={step} y={T} height={plotH} fill="transparent" className="hit" />
              <rect x={x} width={bw} y={y(d.baseline)} height={Math.max(0, T + plotH - y(d.baseline))} rx={2} fill="var(--mo-bar-2)" />
              <rect x={x + bw * 0.2} width={bw * 0.6} y={y(d.actual - Math.max(0, d.gpu ?? 0))} height={Math.max(0, T + plotH - y(d.actual - Math.max(0, d.gpu ?? 0)))}
                rx={1.5} fill="var(--mo-accent)" />
              {(d.gpu ?? 0) > 0 && <rect x={x + bw * 0.2} width={bw * 0.6} y={y(d.actual)} height={Math.max(0, y(d.actual - (d.gpu ?? 0)) - y(d.actual) - 1.5)}
                rx={1.5} fill="var(--mo-accent)" fillOpacity={0.45} className="gpu" />}
              {(i % every === 0 || last) && <text x={x + bw / 2} y={H - 10} textAnchor="middle" className={last ? "t-fg" : undefined}>{day(d.label)}</text>}
            </g>
          );
        })}
      </svg>
    </div>
  );
}

export type Series = { key: string; label: string; color: string };

/** Stacked daily bars with categorical series (savings by cause): the legend carries each series' total, since the
 * lighter colors fall under 3:1 on white. */
export function StackedBars({ days, series, format = usd }: { days: { label: string; values: Record<string, number> }[]; series: Series[]; format?: (v: number) => string }) {
  const W = 880, H = 260, L = 64, B = 30, T = 12, R = 8;
  // Negative values (self-hosting when the GPUs cost more than the API calls they replaced) stack down from zero.
  const up = Math.max(...days.map((d) => series.reduce((s, x) => s + Math.max(0, d.values[x.key] ?? 0), 0)), 0);
  const down = Math.max(...days.map((d) => series.reduce((s, x) => s - Math.min(0, d.values[x.key] ?? 0), 0)), 0);
  const max = up || !down ? niceMax(up) : 0;
  const min = down ? niceMax(down) : 0;
  const plotW = W - L - R, plotH = H - T - B;
  const step = days.length ? plotW / days.length : plotW;
  const bw = Math.max(3, Math.min(30, step * 0.6));
  const y = (v: number) => T + ((max - v) / (max + min)) * plotH;
  const every = Math.ceil(days.length / 8);
  const totals = Object.fromEntries(series.map((s) => [s.key, days.reduce((t, d) => t + (d.values[s.key] ?? 0), 0)]));
  return (
    <>
      <div className="legend" style={{ marginBottom: 8 }}>
        {series.map((s) => <span key={s.key}><i style={{ background: s.color }} />{s.label}<b>{format(totals[s.key])}</b></span>)}
      </div>
      <div className="chart-scroll">
        <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Daily totals by ${series.map((s) => s.label).join(", ")}`}>
          {[...(max ? [0.5, 1].map((f) => f * max) : []), 0, ...(min ? [-min] : [])].map((v) => (
            <g key={v}><line className={v ? "grid" : "axis"} x1={L} x2={W - R} y1={y(v)} y2={y(v)} /><text x={L - 10} y={y(v) + 3.5} textAnchor="end">{v < 0 ? `−${axisUsd(-v)}` : axisUsd(v)}</text></g>
          ))}
          {days.map((d, i) => {
            const x = L + i * step + (step - bw) / 2;
            let pos = 0, neg = 0;
            return (
              <g key={d.label} data-tip={`${day(d.label)}\n${series.map((s) => `${s.label} ${format(d.values[s.key] ?? 0)}`).join("\n")}`}>
                <rect x={L + i * step} width={step} y={T} height={plotH} fill="transparent" className="hit" />
                {series.map((s) => {
                  const v = d.values[s.key] ?? 0;
                  if (!v) return null;
                  if (v < 0) {
                    const y0 = y(neg), y1 = y(neg + v);
                    neg += v;
                    return <rect key={s.key} x={x} width={bw} y={y0 + (neg < v ? 2 : 0)} height={Math.max(0, y1 - y0 - (neg < v ? 2 : 0))} fill={s.color} rx={1} />;
                  }
                  const y0 = y(pos), y1 = y(pos + v);
                  pos += v;
                  return <rect key={s.key} x={x} width={bw} y={y1} height={Math.max(0, y0 - y1 - (pos > v ? 2 : 0))} fill={s.color} rx={1} />;
                })}
                {i % every === 0 && <text x={x + bw / 2} y={H - 10} textAnchor="middle">{day(d.label)}</text>}
              </g>
            );
          })}
        </svg>
      </div>
    </>
  );
}

/** Horizontal bars: muted, or the accent when the list is the one thing that matters on its panel. */
export function HBars({ items, accent, format = usd }: { items: { label: string; value: number; note?: string }[]; accent?: boolean; format?: (v: number) => string }) {
  const max = Math.max(...items.map((i) => Math.abs(i.value)), 0) || 1;
  if (!items.length) return <Text c="dimmed" size="sm">Nothing in this period yet.</Text>;
  return (
    <div className="hbars">
      {items.map((i) => (
        <div key={i.label} className={accent ? "hbar acc" : "hbar"} data-tip={`${i.label}: ${format(i.value)}${i.note ? ` · ${i.note}` : ""}`}>
          <span className="l">{i.label}</span>
          <span className="t"><i className={i.value < 0 ? "neg" : undefined} style={{ width: `${(Math.abs(i.value) / max) * 100}%` }} /></span>
          <span className="n">{format(i.value)}{i.note ? <small> {i.note}</small> : null}</span>
        </div>
      ))}
    </div>
  );
}

/** 1,100 · 32K · 1.1M: tokens and requests, compact. */
export const compact = (v: number | null | undefined) => (v == null ? "–" : v >= 1e6 ? `${+(v / 1e6).toFixed(v >= 1e7 ? 0 : 1)}M`
  : v >= 1e4 ? `${Math.round(v / 1e3)}K` : v >= 10 ? Math.round(v).toLocaleString() : v > 0 ? v.toFixed(1) : "0");

// Your traffic is the series that matters (the accent); what a replica needs is the reference, a ring in the text color.
const NEED = "var(--mantine-color-text)";
const TRAFFIC = "var(--mo-accent)";

/** "2× H100 · aws spot": a replica in a few words. */
export const replicaLabel = (g: BreakevenGpu | null | undefined) => (!g ? "–" : g.accelerator
  ? `${g.count ?? 1}× ${g.accelerator}${g.cloud ? ` · ${g.cloud}` : ""}${g.spot ? " spot" : g.spot === false ? " on-demand" : ""}`
  : g.deployment ? `${g.deployment} (configured)` : "configured");

/** Break-even by quality level, on one log axis of tokens per minute: what one GPU replica (of the open model and machine
 * that pay from the least traffic, or come closest) needs to pay for itself (a ring), the busiest hour of your traffic at
 * that level (the accent dot), and how much that replica can serve (the tick). Need past the tick means it can't pay at
 * any traffic: per token, the GPU costs more than the API. */
export function BreakevenChart({ groups }: { groups: BreakevenGroup[] }) {
  const W = 1380, L = 380, R = 130, T = 34, B = 40, ROW = 36;
  const H = T + groups.length * ROW + B;
  const values = groups.flatMap((g) => [g.tokens_per_minute ?? 0, g.traffic.peak_tokens_per_minute, g.capacity_tokens_per_minute ?? 0]);
  const hi = Math.pow(10, Math.ceil(Math.log10(Math.max(...values, 10) * 1.2)));
  const lo = 1;
  const x = (v: number) => L + (Math.log10(Math.max(v, lo) / lo) / Math.log10(hi / lo)) * (W - L - R);
  const ticks: number[] = [];
  for (let v = lo; v <= hi; v *= 10) ticks.push(v);
  if (!groups.length) return <Text c="dimmed" size="sm">No quality levels to show.</Text>;
  return (
    <div className="chart-scroll">
      <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" style={{ minWidth: 820 }}
        aria-label="Traffic needed for one GPU replica to pay for itself, by quality level, against your busiest hour and that replica's capacity">
        <circle cx={L + 5} cy={14} r={4.5} fill="none" stroke={NEED} strokeWidth={1.5} /><text x={L + 15} y={18}>needed for one replica to pay</text>
        <circle cx={L + 230} cy={14} r={5} fill={TRAFFIC} /><text x={L + 240} y={18}>your busiest hour</text>
        <line x1={L + 378} x2={L + 378} y1={7} y2={21} stroke="var(--mo-fg-2)" strokeWidth={2} /><text x={L + 386} y={18}>what that replica can serve</text>
        {ticks.map((t) => (
          <g key={t}>
            <line className="grid" x1={x(t)} x2={x(t)} y1={T - 4} y2={T + groups.length * ROW} />
            <text x={x(t)} y={H - B + 16} textAnchor="middle">{compact(t)}</text>
          </g>
        ))}
        <text x={(L + W - R) / 2} y={H - 6} textAnchor="middle">tokens per minute (log scale)</text>
        {groups.map((g, i) => {
          const y = T + i * ROW + ROW / 2;
          const need = g.tokens_per_minute;
          const cap = g.capacity_tokens_per_minute;
          const peak = g.traffic.peak_tokens_per_minute;
          const tip = `${g.task} level ${g.level}: the cheapest API model good enough is ${g.cheapest ? `${g.cheapest.model} on ${g.cheapest.backend}` : "none"}`
            + (g.api_usd_per_million != null ? ` at $${g.api_usd_per_million.toFixed(3)}/M tokens` : "")
            + (g.gpu ? `. Open model: ${g.gpu.model} on ${replicaLabel(g.gpu)}${g.gpu.weights ? ` (${g.gpu.weights} weights)` : ""} at $${g.gpu.usd_per_hour.toFixed(2)}/h, `
              + `about ${compact(g.gpu.capacity_tokens_per_second)} tok/s (${g.gpu.capacity_basis}), $${(g.gpu_usd_per_million ?? 0).toFixed(3)}/M when full` : `. ${g.why ?? ""}`)
            + (need != null ? `. One replica pays from ${compact(need)} tokens/min (about ${compact(g.requests_per_hour)} requests an hour)` : "")
            + `. Your traffic: ${compact(g.traffic.tokens_per_minute)}/min on average, ${compact(peak)}/min in the busiest hour.`;
          const pays = g.gpu && need != null && g.pays && peak >= need;
          return (
            <g key={g.floor} opacity={g.gpu ? 1 : 0.5} data-tip={tip}>
              <rect x={0} y={y - ROW / 2} width={W} height={ROW} fill="transparent" className="hit" />
              {i > 0 && <line className="grid" x1={0} x2={W} y1={y - ROW / 2} y2={y - ROW / 2} />}
              <text x={8} y={y - 3} className="t-fg" style={{ fontSize: 11.5 }}>{`${g.task} · level ${g.level}`}
                {g.cheapest ? ` vs ${g.cheapest.model} $${(g.api_usd_per_million ?? 0).toFixed(2)}/M` : ""}</text>
              <text x={8} y={y + 11}>{g.gpu ? `${g.gpu.model} · ${replicaLabel(g.gpu)} · $${g.gpu.usd_per_hour.toFixed(2)}/h` : "no open model can take it"}</text>
              {cap != null && <line x1={x(cap)} x2={x(cap)} y1={y - 9} y2={y + 9} stroke="var(--mo-fg-2)" strokeWidth={2} />}
              {need != null && <line x1={x(peak)} x2={x(need)} y1={y} y2={y} stroke="var(--mo-bar)" strokeWidth={1.5} />}
              {peak > 0
                ? <circle cx={x(peak)} cy={y} r={5} fill={TRAFFIC} stroke="var(--mo-surface)" strokeWidth={2} />
                : <circle cx={x(lo)} cy={y} r={4} fill="none" stroke={TRAFFIC} strokeWidth={1.5} />}
              {need != null && <circle cx={x(need)} cy={y} r={4.5} fill="var(--mo-surface)" stroke={NEED} strokeWidth={1.5} />}
              <text x={W - R + 12} y={y + 4} className={pays ? "t-acc" : "t-fg2"}>
                {!g.gpu ? "no open model" : need == null ? "–" : !g.pays ? "never pays" : peak >= need ? "pays now" : `from ${compact(need)}/min`}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
