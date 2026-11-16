import { Alert, Badge, Group, Paper, Skeleton, Stack, Text, Title } from "@mantine/core";
import { useEffect, useRef, type ReactNode } from "react";
import { usd } from "../api";
import { Icon, type IconName } from "./icons";

/** A panel: a header row (title, an optional dimmed hint, controls on the right), a hairline, the body, and an
 * optional grey footer for notes, "Answered by" and links. `flush` drops the body's padding, for tables. */
export function Panel({ title, hint, right, footer, flush, children, ...rest }:
  { title: ReactNode; hint?: ReactNode; right?: ReactNode; footer?: ReactNode; flush?: boolean; children?: ReactNode } & Record<string, unknown>) {
  return (
    <Paper p={0} className="panel" {...rest}>
      <div className="panel-h">
        <Text component="h2" className="panel-t" m={0}>{title}{hint && <span className="hint"> · {hint}</span>}</Text>
        {right}
      </div>
      {children != null && <div className={flush ? "panel-b flush" : "panel-b"}>{children}</div>}
      {footer && <div className="panel-f">{footer}</div>}
    </Paper>
  );
}

/** A KPI tile. Only the number that matters most on a page (what was saved) takes the accent. */
export function Kpi({ label, value, sub, accent }: { label: string; value: string; sub?: ReactNode; accent?: boolean }) {
  return (
    <Paper px={16} py={14} className="kpi">
      <div className="kpi-k">{label}</div>
      <div className={accent ? "kpi-v acc" : "kpi-v"}>{value}</div>
      {sub && <div className="kpi-s">{sub}</div>}
    </Paper>
  );
}

/** Spend split into APIs and GPUs (decision 64): nothing when there's no GPU spend. ``idle``: GPU time no request paid
 * for (starting, idle, draining), which the organization's totals carry; the tip says how much of the GPUs it is. */
export function SpendSplit({ api, gpu, idle }: { api: number; gpu: number; idle?: number }) {
  if (!gpu) return null;
  const tip = idle
    ? `GPUs: ${usd(gpu - idle)} as requests' shares of GPU time, and ${usd(idle)} of GPU time no request paid for (starting, idle, draining).`
    : "GPUs: the requests' shares of GPU time.";
  return <span className="split" data-tip={tip} tabIndex={0}>APIs {usd(api)} · GPUs {usd(gpu)}</span>;
}

export type StatusKind = "good" | "warn" | "bad" | "off" | "clock" | "info" | "drain" | "acc";
const STATUS_ICON: Record<StatusKind, IconName> = { good: "ok", warn: "warn", bad: "bad", off: "off", clock: "clock", info: "info", drain: "drain", acc: "arrow" };
const STATUS_CLASS: Record<StatusKind, string> = { good: "st-good", warn: "st-warn", bad: "st-bad", off: "st-off", clock: "st-off", info: "st-off", drain: "st-off", acc: "st-acc" };

/** A state in color, always with its icon and a word: green, amber and red mean status and nothing else. */
export function Status({ kind, children, icon, title, ...rest }: { kind: StatusKind; children?: ReactNode; icon?: IconName; title?: string } & Record<string, unknown>) {
  return <span className={`st ${STATUS_CLASS[kind]}`} title={title} {...rest}><Icon name={icon ?? STATUS_ICON[kind]} />{children}</span>;
}

const BADGE_COLOR: Partial<Record<StatusKind, string>> = { good: "green", warn: "yellow", bad: "red", acc: "mo" };

/** A status as a badge (an icon and a word on a soft fill): for table cells and headers. */
export function StatusBadge({ kind, children, size = "md", ...rest }: { kind: StatusKind; children: ReactNode; size?: string } & Record<string, unknown>) {
  return (
    <Badge color={BADGE_COLOR[kind] ?? "gray"} size={size} leftSection={<Icon name={STATUS_ICON[kind]} size={12} />} {...rest}>{children}</Badge>
  );
}

export function Loading({ loading, error, rows = 3 }: { loading: boolean; error: string | null; rows?: number }) {
  if (error) return <Alert color="red" title="Couldn't load this" icon={<Icon name="bad" size={16} />}>{error}</Alert>;
  if (loading) return <Stack gap={6}>{Array.from({ length: rows }, (_, i) => <Skeleton key={i} h={14} />)}</Stack>;
  return null;
}

export function PageTitle({ title, sub, children }: { title: string; sub?: ReactNode; children?: ReactNode }) {
  return (
    <Group justify="space-between" align="flex-end" mb="md" wrap="wrap" gap="sm">
      <div>
        <Title order={1} fz={22} lh={1.2} style={{ letterSpacing: "-.02em" }}>{title}</Title>
        {sub && <Text size="sm" c="dimmed" mt={3}>{sub}</Text>}
      </div>
      {children}
    </Group>
  );
}

const STORE_LABEL: Record<string, string> = { postgres: "Postgres", clickhouse: "ClickHouse", memory: "memory (no database)" };

/** Which store answered a report (the API's `answered_by`): Postgres, or ClickHouse for long ranges and tag breakdowns. */
export function AnsweredBy({ by }: { by?: string | null }) {
  if (!by) return null;
  const names = by.split("+").map((s) => STORE_LABEL[s] ?? s);
  return (
    <Text span size="xs" c="dimmed" title="Postgres holds every request. With ClickHouse set up, reports over more than 7 days and tag breakdowns read its copy (Config → Reporting).">
      Answered by {names.join(" and ")}
    </Text>
  );
}

export const STATE_LABEL: Record<string, string> = {
  api_only: "API only", provisioning: "Starting GPUs", warm: "Serving on GPUs", draining: "Draining", disabled: "Off",
};
const STATE_KIND: Record<string, StatusKind> = { warm: "good", provisioning: "clock", draining: "drain", api_only: "off", disabled: "off" };

/** A GPU deployment's state: green only while it serves; starting and draining are normal steps, so they stay neutral. */
export function StateBadge({ state }: { state: string }) {
  return <Status kind={STATE_KIND[state] ?? "off"}>{STATE_LABEL[state] ?? state}</Status>;
}

/** One floating tooltip for every element with a `data-tip` (chart marks, tiles, badges): lighter than a Mantine
 * Tooltip per SVG mark, and it follows the pointer. Mounted once, in main.tsx. */
export function TipLayer() {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const tip = ref.current!;
    let cur: Element | null = null;
    const place = (x: number, y: number) => {
      const w = tip.offsetWidth, h = tip.offsetHeight;
      let X = x + 14, Y = y + 16;
      if (X + w > innerWidth - 8) X = x - w - 14;
      if (Y + h > innerHeight - 8) Y = y - h - 14;
      tip.style.transform = `translate(${Math.max(8, X)}px, ${Math.max(8, Y)}px)`;
    };
    const show = (el: Element | null, x: number, y: number) => {
      cur?.classList.remove("is-hover");
      cur = el;
      if (!el) { tip.hidden = true; return; }
      el.classList.add("is-hover");
      tip.textContent = el.getAttribute("data-tip");
      tip.hidden = false;
      place(x, y);
    };
    const over = (e: MouseEvent) => show((e.target as Element).closest?.("[data-tip]") ?? null, e.clientX, e.clientY);
    const move = (e: MouseEvent) => { if (!tip.hidden) place(e.clientX, e.clientY); };
    const focus = (e: FocusEvent) => {
      const el = (e.target as Element).closest?.("[data-tip]");
      if (el) { const r = el.getBoundingClientRect(); show(el, r.left, r.bottom); }
    };
    const blur = () => show(null, 0, 0);
    document.addEventListener("mouseover", over);
    document.addEventListener("mousemove", move);
    document.addEventListener("focusin", focus);
    document.addEventListener("focusout", blur);
    return () => {
      document.removeEventListener("mouseover", over);
      document.removeEventListener("mousemove", move);
      document.removeEventListener("focusin", focus);
      document.removeEventListener("focusout", blur);
    };
  }, []);
  return <div ref={ref} className="tip" role="tooltip" hidden />;
}
