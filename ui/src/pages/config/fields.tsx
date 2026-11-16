import { ActionIcon, Group, NumberInput, Select, Stack, Text, TextInput, Title, Tooltip } from "@mantine/core";
import { Fragment, type ReactNode } from "react";
import type { Doc } from "../../api";

/** Help text with `backticks` shown as code. */
export function rich(text: string): ReactNode {
  return text.split(/(`[^`]+`)/).map((part, i) =>
    part.startsWith("`") && part.endsWith("`") && part.length > 2 ? <code key={i}>{part.slice(1, -1)}</code> : <Fragment key={i}>{part}</Fragment>);
}

/** A "?" that explains a setting. Opens on hover, keyboard focus, or tap, and screen readers announce it.
 * Clicking it never activates the field whose label it sits in (a switch doesn't toggle). */
export function Help({ text, about }: { text: string; about: string }) {
  return (
    <Tooltip label={rich(text)} multiline w={320} withArrow position="top-start" openDelay={120} events={{ hover: true, focus: true, touch: true }}>
      <ActionIcon component="span" role="button" tabIndex={0} size={16} radius="xl" variant="light" color="gray" aria-label={`About ${about}`}
        style={{ fontSize: 11, fontWeight: 700, cursor: "help", display: "inline-flex", verticalAlign: "text-bottom", marginLeft: 6 }}
        onClick={(e: React.MouseEvent) => { e.preventDefault(); e.stopPropagation(); }}
        onKeyDown={(e: React.KeyboardEvent) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); e.stopPropagation(); } }}>?</ActionIcon>
    </Tooltip>
  );
}

/** The label's last word and the "?" never wrap apart. */
function withHelp(label: string, icon: ReactNode): ReactNode {
  const cut = label.lastIndexOf(" ");
  return <>{label.slice(0, cut + 1)}<span style={{ whiteSpace: "nowrap" }}>{label.slice(cut + 1)}{icon}</span></>;
}

/** Label props for any Mantine input: the label with its "?", while the input's accessible name stays the plain label. */
export function lbl(label: string, help?: string): { label: ReactNode; "aria-label": string } {
  return {
    label: help ? withHelp(label, <Help text={help} about={label} />) : label,
    "aria-label": label,
  };
}

/** A section heading with its explanation underneath. */
export function SectionTitle({ title, intro, right }: { title: string; intro?: string; right?: ReactNode }) {
  return (
    <div>
      <Group justify="space-between" wrap="nowrap"><Title order={4} fz="sm">{title}</Title>{right}</Group>
      {intro && <Text size="xs" c="dimmed" mt={2} mb={4}>{rich(intro)}</Text>}
    </div>
  );
}

/** Immutable set at a path; undefined deletes the key. */
export function setIn(doc: Doc, path: (string | number)[], value: unknown): Doc {
  const root = structuredClone(doc);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let cur: any = root;
  for (let i = 0; i < path.length - 1; i++) {
    const k = path[i];
    if (cur[k] == null || typeof cur[k] !== "object") cur[k] = typeof path[i + 1] === "number" ? [] : {};
    cur = cur[k];
  }
  const last = path[path.length - 1];
  if (value === undefined) {
    if (Array.isArray(cur) && typeof last === "number") cur.splice(last, 1);
    else delete cur[last];
  } else cur[last] = value;
  return root;
}

export const isEnvRef = (v: unknown) => typeof v === "string" && v.includes("${");

/** A number field where blank means "use the default". Shows env references read-only-ish. */
export function Num({ label, value, onChange, description, min, max, step, suffix, prefix, w, placeholder = "default", help, error }: {
  label: string; value: unknown; onChange: (v: number | undefined) => void; description?: ReactNode; min?: number; max?: number; step?: number; suffix?: string;
  prefix?: string; w?: number | string; placeholder?: string; help?: string; error?: ReactNode;
}) {
  if (isEnvRef(value)) {
    return <TextInput {...lbl(label, help)} value={String(value)} description="from an environment variable" readOnly w={w} />;
  }
  return (
    <NumberInput {...lbl(label, help)} description={description} error={error} value={value == null ? "" : Number(value)} min={min} max={max} step={step} suffix={suffix} prefix={prefix} w={w}
      allowDecimal decimalScale={6} placeholder={placeholder} thousandSeparator=","
      onChange={(v) => onChange(v === "" || v == null ? undefined : Number(v))} />
  );
}

export function Txt({ label, value, onChange, description, placeholder, w, mono, help }: {
  label: string; value: unknown; onChange: (v: string | undefined) => void; description?: ReactNode; placeholder?: string; w?: number | string; mono?: boolean;
  help?: string;
}) {
  return (
    <TextInput {...lbl(label, help)} description={description} placeholder={placeholder} w={w} value={value == null ? "" : String(value)}
      styles={mono ? { input: { fontFamily: "var(--mantine-font-family-monospace)" } } : undefined}
      onChange={(e) => onChange(e.currentTarget.value === "" ? undefined : e.currentTarget.value)} />
  );
}

/** An environment variable *name* (never the secret itself). */
export function EnvName({ label, value, onChange, description, help }: {
  label: string; value: unknown; onChange: (v: string | undefined) => void; description?: ReactNode; help?: string;
}) {
  return (
    <TextInput {...lbl(label, help)} value={value == null ? "" : String(value)} placeholder="e.g. ANTHROPIC_API_KEY"
      description={description ?? "Name of the environment variable that holds the secret. The secret itself is never stored."}
      styles={{ input: { fontFamily: "var(--mantine-font-family-monospace)" } }}
      onChange={(e) => onChange(e.currentTarget.value.toUpperCase().replace(/[^A-Z0-9_]/g, "") || undefined)} />
  );
}

/** true / false / "from an environment variable" for flags like `enabled`. */
export function Flag({ label, value, onChange, description, help }: {
  label: string; value: unknown; onChange: (v: unknown) => void; description?: ReactNode; help?: string;
}) {
  const mode = isEnvRef(value) ? "env" : value === false ? "no" : "yes";
  return (
    <Stack gap={4}>
      <Select {...lbl(label, help)} description={description} allowDeselect={false} value={mode}
        data={[{ value: "yes", label: "Yes" }, { value: "no", label: "No" }, { value: "env", label: "From an environment variable…" }]}
        onChange={(m) => onChange(m === "yes" ? undefined : m === "no" ? false : "${MY_FLAG:-false}")} />
      {mode === "env" && <TextInput value={String(value)} onChange={(e) => onChange(e.currentTarget.value)} description="e.g. ${MERITORDER_ALLOW_FREE_MODELS:-false}"
        styles={{ input: { fontFamily: "var(--mantine-font-family-monospace)" } }} />}
    </Stack>
  );
}

/** Edit a string -> string mapping (e.g. Entra group ID -> team). */
export function KeyValue({ label, value, onChange, keyLabel, valueLabel, valueOptions, description, help }: {
  label: string; value: Record<string, string> | undefined; onChange: (v: Record<string, string> | undefined) => void;
  keyLabel: string; valueLabel: string; valueOptions?: string[]; description?: ReactNode; help?: string;
}) {
  const rows = Object.entries(value ?? {});
  const write = (next: [string, string][]) => onChange(next.length ? Object.fromEntries(next) : undefined);
  return (
    <Stack gap={6}>
      <Group justify="space-between">
        <div><Group gap={6} wrap="nowrap"><Text size="sm" fw={500}>{label}</Text>{help && <Help text={help} about={label} />}</Group>
          {description && <Text size="xs" c="dimmed">{description}</Text>}</div>
        <ActionIcon variant="light" aria-label={`Add ${keyLabel}`} onClick={() => write([...rows, ["", valueOptions?.[0] ?? ""]])}>+</ActionIcon>
      </Group>
      {rows.map(([k, v], i) => (
        <Group key={i} gap="xs" wrap="nowrap">
          <TextInput placeholder={keyLabel} value={k} style={{ flex: 2 }} styles={{ input: { fontFamily: "var(--mantine-font-family-monospace)" } }}
            onChange={(e) => write(rows.map((r, j) => (j === i ? [e.currentTarget.value, r[1]] : r)))} aria-label={keyLabel} />
          {valueOptions ? (
            <Select data={valueOptions} value={v} style={{ flex: 1 }} allowDeselect={false} aria-label={valueLabel}
              onChange={(nv) => write(rows.map((r, j) => (j === i ? [r[0], nv ?? ""] : r)))} />
          ) : (
            <TextInput placeholder={valueLabel} value={v} style={{ flex: 1 }} aria-label={valueLabel}
              onChange={(e) => write(rows.map((r, j) => (j === i ? [r[0], e.currentTarget.value] : r)))} />
          )}
          <ActionIcon variant="subtle" color="red" aria-label="Remove" onClick={() => write(rows.filter((_, j) => j !== i))}>×</ActionIcon>
        </Group>
      ))}
      {!rows.length && <Text size="xs" c="dimmed">None.</Text>}
    </Stack>
  );
}

export const ROLES = ["user", "team_lead", "finance", "admin"];

/** routing.model_choice and a team's model_choice (config.py ModelChoice). */
export const MODEL_CHOICES = [
  { value: "threshold", label: "Threshold: the cheapest model that meets the level" },
  { value: "cost_per_success", label: "Cost per success: the lowest expected cost per solved task" },
  { value: "expected_cost", label: "Expected total cost: the lowest cost to finish the task, a person's time included" },
];
