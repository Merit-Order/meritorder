import { Alert, Badge, Button, Group, NumberInput, Paper, SimpleGrid, Stack, Table, Text, TextInput, Tooltip } from "@mantine/core";
import { notifications } from "@mantine/notifications";
import { useState } from "react";
import type { Api, Controls as ControlsData, Me, Safeguards, SpendLine } from "../api";
import { pct, usd } from "../api";
import { Loading, PageTitle, Panel, SpendSplit, Status } from "../components/common";
import { Icon } from "../components/icons";
import { useData } from "../hooks";
import { ControllerBadge, StopAllButton } from "./Gpus";

type Edits = Record<string, Record<string, number | undefined>>;

/** A budget meter: what's spent (the accent; amber from 80% of the budget, red at it), the projection to the end of the
 * month hatched after it, and a tick at the budget. The scale reaches past whichever is larger, so the tick and an
 * overshooting projection both stay on it. */
export function Meter({ spent, projected, budget, unit = "budget", format = usd }:
  { spent: number; projected?: number; budget: number; unit?: string; format?: (v: number) => string }) {
  const scale = Math.max(budget, projected ?? 0, spent) * 1.05 || 1;
  const used = spent / budget;
  const w = (v: number) => `${Math.min(100, (v / scale) * 100)}%`;
  return (
    <>
      <div className="meter" role="img" aria-label={`${format(spent)} of ${format(budget)}${projected != null ? `, on pace for ${format(projected)}` : ""}`}>
        {projected != null && projected > spent && <i className="proj" style={{ width: w(projected) }} />}
        <i className={`used${used >= 1 ? " bad" : used >= 0.8 ? " warn" : ""}`} style={{ width: w(spent) }} />
        <span className="tick" style={{ left: `calc(${w(budget)} - 1px)` }} />
      </div>
      <div className="meter-axis" style={{ position: "relative" }}>
        <span>$0</span>
        <span style={{ position: "absolute", right: `${100 - Math.min(100, (budget / scale) * 100)}%` }}>{unit} {format(budget)}</span>
      </div>
    </>
  );
}

function SpendMeter({ line, label }: { line: SpendLine; label: string }) {
  if (!line.budget) return <Text size="xs" c="dimmed">No {label} budget. Projected {usd(line.projected)} this month.</Text>;
  const used = line.spent / line.budget;
  const over = line.projected > line.budget;
  return (
    <Stack gap={8}>
      <div>
        <Meter spent={line.spent} projected={line.projected} budget={line.budget} />
      </div>
      <Group gap="4px 14px" fz="xs" c="var(--mo-fg-2)">
        <span><span className="swatch" style={{ background: used >= 1 ? "var(--mo-bad)" : used >= 0.8 ? "var(--mo-warn)" : "var(--mo-accent)" }} />spent {pct(used)}</span>
        <span><span className="swatch proj" />projected {usd(line.projected)}</span>
        {over ? <Status kind="warn">on pace to go over</Status> : <Status kind="good">on pace</Status>}
      </Group>
      <Text size="xs" c="dimmed">{usd(line.spent)} of {usd(line.budget)} ({pct(used)}) · on pace for {usd(line.projected)}{over ? <Text span c="var(--mo-warn)" inherit> (over budget)</Text> : null}</Text>
    </Stack>
  );
}

const GUARDS: { key: keyof Safeguards; label: string; help: string; prefix?: string; int?: boolean }[] = [
  { key: "org_budget_usd_month", label: "Organization budget per month", prefix: "$", help: "All LLM spend: requests on APIs, and GPUs at what they cost to run, idle and starting time included. Once reached, every request is refused until the month ends or it's raised." },
  { key: "gpu_budget_usd_month", label: "GPU budget per month", prefix: "$", help: "GPU time alone, a cap of its own (it counts in the organization budget too). Once reached, running GPUs drain and none start (pins included)." },
  { key: "max_gpu_usd_per_hour", label: "Max GPU burn rate", prefix: "$", help: "Per hour, across all deployments. The best-covered deployments get capacity first." },
  { key: "max_gpu_replicas", label: "Max GPU replicas", int: true, help: "Across all deployments." },
  { key: "max_request_usd", label: "Max cost per request", prefix: "$", help: "A request that sets max_tokens is checked at its worst case (context × input price + max_tokens × output price): cheaper models are used instead, and if none fit it's refused. A request without max_tokens may write as much as the model can, less whatever wouldn't fit under this limit (the reply's x-meritorder-max-tokens header says how much it got)." },
];

export function Controls({ api, me }: { api: Api; me: Me }) {
  const res = useData(() => api.controls(), [api], 30_000);
  const admin = me.role === "admin";
  // Only the safeguards this admin edited: the rest show (and save) what the server has now, so a change another admin
  // saved meanwhile is never written back over with an old copy.
  const [guards, setGuards] = useState<Partial<Safeguards>>({});
  const [edits, setEdits] = useState<Edits>({});
  const [comment, setComment] = useState("");
  const [busy, setBusy] = useState(false);
  const d: ControlsData | null = res.data;
  const guardVal = (k: keyof Safeguards) => (k in guards ? guards[k] : d?.safeguards[k]);
  const guardChanges = d ? GUARDS.filter((g) => g.key in guards && (guards[g.key] ?? null) !== (d.safeguards[g.key] ?? null)) : [];
  const teamChanges = Object.entries(edits).flatMap(([team, f]) => Object.entries(f).map(([k, v]) => ({ team, k, v })));
  const dirty = guardChanges.length + teamChanges.length > 0;

  const save = async () => {
    setBusy(true);
    try {
      const changes: [(string | number)[], unknown][] = [
        ...guardChanges.map((g) => [["safeguards", g.key], guards[g.key] ?? undefined] as [(string | number)[], unknown]),
        ...teamChanges.map(({ team, k, v }) => [["teams", team, k], v] as [(string | number)[], unknown]),
      ];
      const what = [...guardChanges.map((g) => g.label.toLowerCase()), ...teamChanges.map((c) => `${c.team} ${c.k.replace(/_usd_month|_usd/, "").replace("_", " ")}`)];
      const r = await api.patchConfig(changes, comment || `Controls: ${what.join(", ")}`);
      notifications.show({ color: "green", title: `Saved as version ${r.version}`, message: "Every gateway picks it up within seconds." });
      setEdits({}); setComment(""); setGuards({});
      await res.reload();
    } catch (e) {
      notifications.show({ color: "red", title: "Not saved", message: e instanceof Error ? e.message : String(e) });
    } finally { setBusy(false); }
  };

  const teamVal = (team: string, k: string, current: number | null | undefined) => (k in (edits[team] ?? {}) ? edits[team][k] : current ?? undefined);
  const setTeam = (team: string, k: string, v: number | undefined) => setEdits((e) => ({ ...e, [team]: { ...(e[team] ?? {}), [k]: v } }));

  return (
    <>
      <PageTitle title="Controls" sub="Budgets and hard limits. Changes save as a new configuration version.">
        <Group gap="xs">
          {d && <ControllerBadge c={d.controller} />}
          {admin && d && d.fleet.deployments.length > 0 && !d.gpu_controls.paused && <StopAllButton api={api} onDone={() => void res.reload()} />}
        </Group>
      </PageTitle>
      <Loading loading={res.loading} error={res.error} />
      {d && (
        <Stack gap="md">
          {d.gpu_controls.paused && <Alert color="red" icon={<Icon name="bad" size={16} />} title="All GPUs are stopped">By {d.gpu_controls.paused_by}{d.gpu_controls.reason ? `: ${d.gpu_controls.reason}` : ""}. Resume on the GPUs page.</Alert>}
          <SimpleGrid cols={{ base: 1, md: 3 }}>
            <Panel title="Organization spend this month">
              <div className="kpi-v" style={{ marginBottom: 4 }}>{usd(d.spend.org.spent)}</div>
              {!!d.spend.org.gpu && <Text className="kpi-s" mb={6}><SpendSplit api={d.spend.org.api} gpu={d.spend.org.gpu} idle={d.spend.org.gpu_idle} /></Text>}
              {!!d.spend.org.reserved && <Text className="kpi-s" mb={6}>{usd(d.spend.org.reserved)} of it reserved by running batches</Text>}
              <SpendMeter line={d.spend.org} label="organization" />
            </Panel>
            <Panel title="GPU spend this month"><div className="kpi-v" style={{ marginBottom: 4 }}>{usd(d.spend.gpu.spent)}</div><SpendMeter line={d.spend.gpu} label="GPU" /></Panel>
            <Panel title="GPUs running now">
              <div className="kpi-v" style={{ marginBottom: 4 }}>{`${d.fleet.replicas} replica${d.fleet.replicas === 1 ? "" : "s"}`}</div>
              {d.safeguards.max_gpu_usd_per_hour ? (
                <Stack gap={8}>
                  <div><Meter spent={d.fleet.usd_per_hour} budget={d.safeguards.max_gpu_usd_per_hour} unit="limit" format={(v) => `${usd(v)}/h`} /></div>
                  <Text size="xs" c="dimmed">
                    {d.fleet.usd_per_hour / d.safeguards.max_gpu_usd_per_hour >= 0.8 && <Status kind="warn" style={{ marginRight: 6 }}>{pct(d.fleet.usd_per_hour / d.safeguards.max_gpu_usd_per_hour)} of the fleet limit</Status>}
                    {usd(d.fleet.usd_per_hour)}/h of {usd(d.safeguards.max_gpu_usd_per_hour)}/h · {d.fleet.deployments.length} deployment{d.fleet.deployments.length === 1 ? "" : "s"} configured
                  </Text>
                </Stack>
              ) : (
                <Text size="xs" c="dimmed">{usd(d.fleet.usd_per_hour)}/h · {d.fleet.deployments.length} deployment{d.fleet.deployments.length === 1 ? "" : "s"} configured · no burn-rate limit</Text>
              )}
            </Panel>
          </SimpleGrid>

          <Panel title="Safeguards" right={<Text size="xs" c="dimmed">Hard limits for everyone. Empty means no limit.</Text>}>
            <SimpleGrid cols={{ base: 1, sm: 2, lg: 3 }}>
              {GUARDS.map((g) => (
                <NumberInput key={g.key} label={g.label} description={g.help} prefix={g.prefix} min={0} allowDecimal={!g.int} decimalScale={2} thousandSeparator=","
                  placeholder="no limit" readOnly={!admin} value={guardVal(g.key) == null ? "" : Number(guardVal(g.key))}
                  onChange={(v) => setGuards((s) => ({ ...s, [g.key]: v === "" || v == null ? null : Number(v) }))} />
              ))}
            </SimpleGrid>
          </Panel>

          <Panel title="Team budgets" flush right={<Text size="xs" c="dimmed">Month to date, with the pace it's on</Text>}
            footer={<span>Reaching a team or personal budget refuses that team's or person's requests; a per-session cap stops a runaway agent. Alerts go out at {d.safeguards.alert_at.map((a) => pct(a)).join(" and ")}. Rate limits and model policy are on Config → Teams.</span>}>
            <Table.ScrollContainer minWidth={1040}>
              <Table>
                <Table.Thead><Table.Tr>
                  <Table.Th>Team</Table.Th><Table.Th w={260}>Spent</Table.Th><Table.Th ta="right">On pace for</Table.Th>
                  <Table.Th>Team budget / month</Table.Th><Table.Th>Per person / month</Table.Th><Table.Th>Per session</Table.Th>
                  <Table.Th><Tooltip multiline w={320} label="What a failed level-5 attempt costs a person on the team (noticing it, saying so, waiting for the redo): minutes of their time times their rate. Teams routing by expected total cost count it against each model's chance of failing, a third as much for each level below 5 and less for small requests; blank uses the organization's (Config → Routing).">
                    <span>A failure costs a person</span></Tooltip></Table.Th>
                </Table.Tr></Table.Thead>
                <Table.Tbody>
                  {d.teams.map((t) => {
                    const budget = teamVal(t.name, "budget_usd_month", t.budget_usd_month);
                    const used = budget ? t.spent / budget : null;
                    const was: Record<string, number | null | undefined> = { budget_usd_month: t.budget_usd_month, user_budget_usd_month: t.user_budget_usd_month,
                      max_session_usd: t.max_session_usd, human_cost_usd: t.human_cost_usd };
                    const edited = Object.keys(edits[t.name] ?? {}).some((k) => (edits[t.name][k] ?? null) !== (was[k] ?? null));
                    const field = (k: string, cur: number | null | undefined, placeholder = "none") => {
                      const changed = k in (edits[t.name] ?? {}) && (edits[t.name][k] ?? null) !== (cur ?? null);
                      return (
                        <>
                          <NumberInput size="xs" prefix="$" min={0} decimalScale={2} thousandSeparator="," placeholder={placeholder} w={130} readOnly={!admin || t.unconfigured}
                            styles={changed ? { input: { borderColor: "var(--mo-accent)", boxShadow: "0 0 0 3px var(--mo-accent-soft)" } } : undefined}
                            value={teamVal(t.name, k, cur) ?? ""} onChange={(v) => setTeam(t.name, k, v === "" || v == null ? undefined : Number(v))} />
                          {changed && <Text size="xs" c="dimmed" mt={2}>was {cur == null ? "none" : usd(cur)}</Text>}
                        </>
                      );
                    };
                    const over = budget != null && t.projected > budget;
                    return (
                      <Table.Tr key={t.name}>
                        <Table.Td>{t.name}{edited && <Badge ml={6} size="sm" color="mo">edited</Badge>}{t.unconfigured && <Badge ml={6} size="sm" variant="default">not in config</Badge>}</Table.Td>
                        <Table.Td>
                          <Text size="sm" className="mono">{usd(t.spent)}{used != null && <Text span c="dimmed" inherit> · {pct(used)}</Text>}</Text>
                          {used != null && <div className={`mini ${used >= 1 ? "bad" : used >= 0.8 ? "warn" : "acc"}`}><i style={{ width: `${Math.min(used, 1) * 100}%` }} /></div>}
                        </Table.Td>
                        <Table.Td ta="right" className="num">
                          {over ? <Status kind="warn" title="On pace to go over its budget">{usd(t.projected)}</Status> : <Text span size="sm" ff="monospace">{usd(t.projected)}</Text>}
                        </Table.Td>
                        <Table.Td>{field("budget_usd_month", t.budget_usd_month)}</Table.Td>
                        <Table.Td>{field("user_budget_usd_month", t.user_budget_usd_month)}</Table.Td>
                        <Table.Td>{field("max_session_usd", t.max_session_usd)}</Table.Td>
                        <Table.Td>
                          {t.unconfigured ? null : field("human_cost_usd", t.human_cost_usd, `org's ${usd(d.human_cost_usd ?? 1)}`)}
                          {!t.unconfigured && t.model_choice !== "expected_cost" && <Text size="xs" c="dimmed" mt={2}>not counted: routes by {t.model_choice === "cost_per_success" ? "cost per success" : "threshold"}</Text>}
                        </Table.Td>
                      </Table.Tr>
                    );
                  })}
                </Table.Tbody>
              </Table>
            </Table.ScrollContainer>
          </Panel>

          {admin && (
            <Paper px="md" py="sm" style={{ position: dirty ? "sticky" : "static", bottom: 12, zIndex: 5, boxShadow: dirty ? "0 8px 24px -12px rgba(0,0,0,.35)" : undefined,
              borderColor: dirty ? "var(--mo-accent-line)" : undefined }}>
              <Group justify="space-between" gap="xs" wrap="wrap">
                <Text size="sm" c={dirty ? undefined : "dimmed"}>{dirty
                  ? `${guardChanges.length + teamChanges.length} change${guardChanges.length + teamChanges.length === 1 ? "" : "s"}: ${[
                    ...guardChanges.map((g) => `${g.label.toLowerCase()} ${d.safeguards[g.key] == null ? "none" : String(d.safeguards[g.key])} to ${guardVal(g.key) == null ? "none" : String(guardVal(g.key))}`),
                    ...teamChanges.map((c) => `${c.team} ${c.k.replace(/_usd_month|_usd/, "").replace(/_/g, " ")} to ${c.v == null ? "none" : usd(c.v)}`)].join(", ")}`
                  : "No changes. Edit a limit above; saving makes a new configuration version."}</Text>
                <Group gap="xs">
                  {dirty && <TextInput placeholder="What changed and why (optional)" value={comment} onChange={(e) => setComment(e.currentTarget.value)} w={340} />}
                  {dirty && <Button variant="default" onClick={() => { setEdits({}); setGuards({}); }}>Discard</Button>}
                  <Button disabled={!dirty} loading={busy} onClick={save}>Save as new config version</Button>
                </Group>
              </Group>
            </Paper>
          )}
          {!admin && <Text size="xs" c="dimmed">Only admins can change limits.</Text>}
        </Stack>
      )}
    </>
  );
}
