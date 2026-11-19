import { Alert, Badge, Button, Grid, Group, Modal, NumberInput, Radio, SimpleGrid, Stack, Table, Text, TextInput, Tooltip } from "@mantine/core";
import { notifications } from "@mantine/notifications";
import { Fragment, useState } from "react";
import type { Api, ControlResult, Deployment, GpuControlState, GpuEvent, GpuReport, Me, ReplicaInfo, StartPhase } from "../api";
import { ago, num, pct, usd } from "../api";
import { HBars } from "../components/charts";
import { Kpi, Loading, PageTitle, Panel, StateBadge, Status, StatusBadge } from "../components/common";
import { Icon } from "../components/icons";
import { useData } from "../hooks";
import { WhatIfPanel } from "./Infra";
import { Person } from "../people";

const when = (unix: number | null | undefined) => (unix ? new Date(unix * 1000).toLocaleString() : "–");
const mins = (seconds: number | null | undefined) => (seconds == null ? "–" : seconds < 90 ? `${Math.round(seconds)}s` : `${(seconds / 60).toFixed(1)} min`);
const phases = (ps: StartPhase[] | null | undefined) => (ps ?? []).map((p) => `${p.phase} ${mins(p.seconds)}`).join(" · ");

function shown(r: ControlResult, title: string) {
  notifications.show({ color: r.warning ? "yellow" : "green", title, message: r.warning ?? "The controller is acting on it now.", autoClose: r.warning ? false : 4000 });
}

function failed(e: unknown) {
  notifications.show({ color: "red", title: "Not changed", message: e instanceof Error ? e.message : String(e) });
}

export function ControllerBadge({ c }: { c: GpuReport["controller"] | undefined }) {
  if (!c) return null;
  const last = c.at ? ago(Date.parse(c.at) / 1000) : null;
  return (
    <Tooltip label={c.alive ? `Last cycle ${last} ago on ${c.host}; every ${c.interval}s` : c.at ? `Last seen ${last} ago on ${c.host}` : "Never seen. Run `meritorder controller`."}>
      <StatusBadge kind={c.alive ? "good" : "bad"}>{c.alive ? "Controller running" : "Controller not running"}</StatusBadge>
    </Tooltip>
  );
}

export function ControlBadge({ c }: { c: GpuControlState | null | undefined }) {
  if (!c) return <Badge variant="default" leftSection={<Icon name="auto" size={12} />}>Automatic</Badge>;
  const until = c.until ? ` until ${new Date(c.until * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}` : "";
  const label = c.mode === "on" ? `Pinned: ${c.replicas}${until}` : c.fleet ? "Fleet stopped" : `Stopped${until}`;
  return (
    <Tooltip label={`${c.by ?? "an admin"}${c.at ? `, ${when(c.at)}` : ""}${c.reason ? `: ${c.reason}` : ""}`}>
      {c.mode === "on" ? <Badge color="mo" leftSection={<Icon name="pin" size={12} />}>{label}</Badge>
        : c.fleet ? <StatusBadge kind="bad">{label}</StatusBadge> : <Badge variant="default" leftSection={<Icon name="stop" size={12} />}>{label}</Badge>}
    </Tooltip>
  );
}

/** One modal for stop, pin, and stop-all: say why, and optionally for how long. */
function ControlModal({ api, target, onClose, onDone, all = [] }: {
  api: Api; target: { kind: "stop" | "pin" | "stop-all"; d?: Deployment } | null; onClose: () => void; onDone: () => void; all?: Deployment[];
}) {
  const [reason, setReason] = useState("");
  const [hours, setHours] = useState<number | undefined>(undefined);
  const [replicas, setReplicas] = useState<number>(1);
  const [how, setHow] = useState("drain");
  const [busy, setBusy] = useState(false);
  if (!target) return null;
  const d = target.d;
  const isStatic = d?.provisioner === "static";
  const title = target.kind === "stop-all" ? "Stop all GPUs" : target.kind === "pin" ? `Pin ${d?.backend}` : `Stop ${d?.backend}`;
  const submit = async () => {
    setBusy(true);
    try {
      if (target.kind === "stop-all") shown(await api.stopAllGpus(reason, how === "drain"), "All GPUs stopping");
      else if (target.kind === "pin") shown(await api.gpuControl(d!.backend, { mode: "on", replicas, hours, reason }), `${d!.backend} pinned to ${replicas}`);
      else shown(await api.gpuControl(d!.backend, { mode: "off", drain: how === "drain", hours, reason }), `${d!.backend} stopping`);
      onDone();
      onClose();
    } catch (e) { failed(e); } finally { setBusy(false); }
  };
  return (
    <Modal opened onClose={onClose} title={title} centered>
      <Stack gap="sm">
        {target.kind === "stop-all" && <Text size="sm">Every GPU deployment stops and none start, whatever the demand, until someone resumes the fleet. Traffic goes to APIs.</Text>}
        {target.kind === "pin" && <Text size="sm">Keeps this many replicas running regardless of demand. Safeguards (GPU budget, fleet limits) still apply.</Text>}
        {target.kind === "pin" && <NumberInput label="Replicas" value={replicas} min={1} max={d?.spec?.max_replicas ?? 1000} onChange={(v) => setReplicas(Number(v) || 1)}
          description={`${usd((d?.gpu_usd_per_hour ?? 0) * replicas)}/h while pinned`} />}
        {target.kind !== "pin" && (
          <Radio.Group label="How" value={how} onChange={setHow}>
            <Stack gap={6} mt={4}>
              <Radio value="drain" label={`Drain first: no new sessions; warm ones finish (up to ${d?.spec?.drain_minutes ?? 15} min)`} />
              <Radio value="now" label="Stop now: in-flight sessions move to APIs (their caches rebuild there)" />
            </Stack>
          </Radio.Group>
        )}
        {target.kind !== "pin" && (() => {
          const names = [...new Set((target.kind === "stop-all" ? all : d ? [d] : []).flatMap((x) => x.only_route_for ?? []))];
          return names.length > 0 && (
            <Alert color="yellow" icon={<Icon name="warn" size={16} />} title="Some models will have nowhere to run">
              {names.join(", ")} {names.length === 1 ? "has" : "have"} no API route. Requests that name {names.length === 1 ? "it" : "them"} will fail
              with an explanation until GPUs are back (or go to an equally good model, for teams that turned on substitution); <code>auto</code> traffic moves to other models. Add an API route (for example OpenRouter) in Config → Models to keep them available.
            </Alert>
          );
        })()}
        {isStatic && target.kind === "stop" && <Alert color="gray" variant="light">This is a server you run yourself: Merit Order stops sending it traffic, but can't power it off.</Alert>}
        {target.kind !== "stop-all" && <NumberInput label="For how long (hours)" value={hours ?? ""} min={0.25} step={1} placeholder="until changed back"
          onChange={(v) => setHours(v === "" ? undefined : Number(v))} description="Leave empty to keep it until someone returns it to automatic." />}
        <TextInput label="Reason" value={reason} onChange={(e) => setReason(e.currentTarget.value)} placeholder="Shown to other admins and in the activity log" />
        <Group justify="flex-end">
          <Button variant="default" onClick={onClose}>Cancel</Button>
          <Button color={target.kind === "pin" ? "mo" : "red"} loading={busy} onClick={submit}>{title}</Button>
        </Group>
      </Stack>
    </Modal>
  );
}

function Where({ d }: { d: Deployment }) {
  const reps = d.replicas_info ?? [];
  if (!reps.length) {
    const s = d.spec;
    if (!s) return <Text size="sm" c="dimmed">Manually managed.</Text>;
    const k = s.kubernetes;
    const where = s.location ?? (k
      ? `your cluster through ${k.api === "llm-d" ? `llm-d (InferencePool ${k.inference_pool ?? "?"})` : "KServe"}, namespace ${k.namespace ?? "of the controller"}`
        + (k.node_type ? `, on ${k.node_type} nodes` : "")
      : (s.clouds.length ? s.clouds.join(", ") : "cheapest cloud SkyPilot finds") + (s.regions.length ? ` (${s.regions.join(", ")})` : ""));
    if (d.state === "disabled") {
      return <Text size="sm" c="dimmed">Switched off in the configuration (Config → Backends), so it can't start. When on, it runs {s.accelerators}{s.use_spot ? " spot" : ""} on {where}.</Text>;
    }
    const lease = s.lease_minutes ? ` Each replica stops by itself ${s.lease_minutes} min after the controller last renewed its lease, and is deleted ${s.autostop_minutes} min later.` : "";
    return <Text size="sm" c="dimmed">{d.state === "api_only" ? "Not running." : "Waiting for replica details."} Runs {s.accelerators}{s.use_spot ? " spot" : ""} on {where}.{lease}</Text>;
  }
  const placed = d.replica_sessions ?? {};
  // SkyPilot replicas hold a lease the controller renews: say when it was renewed and when the replica stops by itself
  // without it. A cluster replica has no lease (the cluster's own controllers keep it).
  const leases = d.provisioner === "skypilot" || d.provisioner === "kubernetes";
  const leaseFor = (r: ReplicaInfo) => {
    if (d.provisioner === "kubernetes") return <><div>none</div><Text size="xs" c="dimmed">not used on a cluster</Text></>;
    if (r.lease_expires_in == null) return <Text size="xs" c="dimmed">no lease yet</Text>;
    const total = (d.spec?.lease_minutes ?? 0) * 60;
    const renewed = total ? Math.max(0, total - r.lease_expires_in) : null;
    return <>
      {renewed != null && <div>renewed {ago(Date.now() / 1000 - renewed)} ago</div>}
      <Text size="xs" c={r.lease_expires_in < 120 ? "var(--mo-warn)" : "dimmed"}>self-stop in {Math.max(0, Math.round(r.lease_expires_in / 60))} min</Text>
    </>;
  };
  const status = (r: ReplicaInfo) => {
    const word = r.status.replace(/_/g, " ").toLowerCase();
    if (r.status === "READY") return <Status kind="good">{word}</Status>;
    if (/FAIL|ERROR/.test(r.status)) return <Status kind="bad">{word}</Status>;
    return <Status kind="clock">{word}</Status>;
  };
  return (
    <Table.ScrollContainer minWidth={720} style={{ border: "1px solid var(--mo-line)", borderRadius: 8 }}>
      <Table className="nowrap-table">
        <Table.Thead><Table.Tr><Table.Th>#</Table.Th><Table.Th>Where</Table.Th><Table.Th>Resources</Table.Th><Table.Th ta="right">$/h</Table.Th><Table.Th ta="right">Up</Table.Th>
          <Table.Th ta="right">Load</Table.Th><Table.Th ta="right"><span data-tip="Sessions whose last turn ran here, and how many of them probably still have their prefix cached here (at least half of it)">Sessions · warm</span></Table.Th>
          {leases && <Table.Th>Lease</Table.Th>}<Table.Th>Status</Table.Th></Table.Tr></Table.Thead>
        <Table.Tbody>
          {reps.map((r: ReplicaInfo) => {
            const s = placed[String(r.id)];
            return (
              <Table.Tr key={String(r.id)}>
                <Table.Td c="dimmed">{r.id}</Table.Td>
                <Table.Td>
                  <Text size="sm" fw={600}>{r.cloud ?? r.infra ?? "–"}</Text>
                  {(r.region || r.zone) && <Text size="xs" c="dimmed">{[r.region, r.zone].filter(Boolean).join(" · ")}</Text>}
                </Table.Td>
                <Table.Td><Text size="sm" className="mono">{r.resources ?? "–"}</Text>{(r.instance_type || r.spot) && <Text size="xs" c="dimmed">{[r.instance_type, r.spot ? "spot" : null].filter(Boolean).join(" · ")}</Text>}</Table.Td>
                <Table.Td className="num">{r.usd_per_hour != null ? usd(r.usd_per_hour) : "–"}</Table.Td>
                <Table.Td className="num">{r.launched_at ? ago(r.launched_at) : r.launched ?? "–"}</Table.Td>
                <Table.Td className="num">
                  <span data-tip={[r.kv_usage != null ? `KV cache ${pct(r.kv_usage)} in use` : null,
                    r.kv_capacity_tokens ? `holds ${num(r.kv_capacity_tokens)} tokens` : null, r.prefix_hit_rate != null ? `${pct(r.prefix_hit_rate)} of prompt tokens read from the cache since the last sample` : null,
                    r.kv_events ? "KV events followed" : null].filter(Boolean).join(" · ") || "vLLM's load at the controller's last sample"}>
                    {r.utilization != null ? pct(r.utilization) : "none yet"}</span>
                  {r.waiting ? <Text size="xs" c="dimmed" ff="monospace">{r.waiting} waiting</Text> : null}
                </Table.Td>
                <Table.Td className="num">{s ? `${s.sessions} · ${s.warm}` : "none"}</Table.Td>
                {leases && <Table.Td fz="sm">{leaseFor(r)}</Table.Td>}
                <Table.Td>
                  <span data-tip={[r.message, r.cluster ? `SkyPilot cluster ${r.cluster}` : null, r.tunnel ? "reached through an SSH tunnel from the controller" : null]
                    .filter(Boolean).join(" · ") || undefined}>{status(r)}</span>
                  {r.tunnel && <Badge size="sm" variant="default" ml={4}>ssh tunnel</Badge>}
                  {r.startup && r.startup.phases.length > 0 && (
                    <Text size="xs" c="dimmed"><span data-tip={phases(r.startup.phases)} style={{ textDecoration: "underline dotted" }}>
                      {r.startup.total_seconds != null ? `started in ${mins(r.startup.total_seconds)}` : `${r.startup.phases[r.startup.phases.length - 1].phase} done`}
                    </span></Text>
                  )}
                </Table.Td>
              </Table.Tr>
            );
          })}
        </Table.Tbody>
      </Table>
    </Table.ScrollContainer>
  );
}

const SOURCE: Record<string, string> = {
  events: "from the replica's KV events",
  estimate: "estimated from the replica's KV-cache metrics: what it has written since this turn against what it had free",
  ttl: "estimated from the cache lifetime (cache_ttl_seconds): the replica reports no KV-cache figures",
  cold: "the replica is gone or isn't the one this session last ran on",
};

/** How the gateways keep sessions on the replica that holds their cache, in one line. */
function affinityLine(d: Deployment): { text: string; problem: string | null } {
  const c = d.cache;
  if (d.affinity && !d.affinity.enabled) return { text: "Off: the load balancer picks each request's replica", problem: null };
  if (!c) return { text: d.affinity?.kv_events ? "Per session, following KV events once running" : "Per session, once running", problem: null };
  if (!c.affinity) return { text: "Load balancer picks (replicas can't be reached one by one); warm caches estimated as 1 in N", problem: c.kv_events_problem };
  const how = c.source === "events" ? "KV events" : c.source === "estimate" ? "estimated from KV-cache metrics" : "estimated from the cache lifetime";
  return { text: `Per session · warm caches ${how}`, problem: c.kv_events_problem };
}

function Placements({ d }: { d: Deployment }) {
  const rows = d.placements ?? [];
  if (!rows.length) return null;
  return (
    <>
      <Text className="label" mt="md" mb={6}>Sessions on its replicas</Text>
      <Table fz="xs" style={{ border: "1px solid var(--mo-line)" }}>
        <Table.Thead><Table.Tr><Table.Th>Session</Table.Th><Table.Th>User</Table.Th><Table.Th>Replica</Table.Th><Table.Th ta="right">Last turn</Table.Th>
          <Table.Th ta="right">Context</Table.Th><Table.Th ta="right">Cached now</Table.Th></Table.Tr></Table.Thead>
        <Table.Tbody>
          {rows.slice(0, 12).map((p) => (
            <Table.Tr key={p.id}>
              <Table.Td><Text size="xs" className="mono">{p.session ?? p.id}</Text>{p.thread && <Text size="xs" c="dimmed">{p.thread}</Text>}</Table.Td>
              <Table.Td><Person id={p.user} /></Table.Td>
              <Table.Td>{p.replica ?? "load balancer"}{p.replica_status && p.replica_status !== "READY" && <Text span size="xs" c="dimmed"> ({p.replica_status.replace("_", " ").toLowerCase()})</Text>}</Table.Td>
              <Table.Td className="num">{ago(p.at)}</Table.Td>
              <Table.Td className="num">{p.tokens != null ? num(p.tokens) : "–"}</Table.Td>
              <Table.Td className="num">
                <Tooltip multiline w={280} label={`${SOURCE[p.source] ?? p.source}${p.expected != null ? `. Expected ${pct(p.expected)} when the turn was routed.` : ""}`}>
                  <Text span size="xs" c={p.share >= 0.5 ? "var(--mo-accent-text)" : p.share > 0 ? "var(--mo-fg-2)" : "dimmed"} style={{ textDecoration: "underline dotted" }}>
                    {pct(p.share)} · {p.source === "events" ? "events" : p.source === "cold" ? "cold" : "estimate"}
                  </Text>
                </Tooltip>
              </Table.Td>
            </Table.Tr>
          ))}
        </Table.Tbody>
      </Table>
      {rows.length > 12 && <Text size="xs" c="dimmed" mt={4}>{rows.length - 12} more not shown.</Text>}
    </>
  );
}

function DeploymentCard({ d, admin, act, onAuto }: { d: Deployment; admin: boolean; act: (kind: "stop" | "pin", d: Deployment) => void; onAuto: (d: Deployment) => void }) {
  const running = d.state !== "api_only" && d.state !== "disabled";
  const burn = running ? d.burn_usd_per_hour ?? (d.gpu_usd_per_hour ?? 0) * (d.replicas ?? 0) : 0;
  const p = d.price;
  const priceNote = p?.source === "skypilot_catalog"
    ? `SkyPilot catalog, cheapest ${p.spot ? "spot " : ""}offer: ${p.cloud} ${p.region} ${p.instance_type}` + (p.configured_usd_per_hour ? ` (config says ${usd(p.configured_usd_per_hour)})` : "")
    : p?.source === "node_type"
      ? `Configured price of a ${p.node_type} node: ${usd(p.node_usd_per_hour ?? 0)}/h${p.spot ? " spot" : ""} for ${p.node_gpus} GPUs, and a replica uses ${p.gpus}. `
        + "A cluster has no price catalog: set it in kubernetes.node_types."
      : "configured gpu_cost_per_hour" + (d.provisioner === "skypilot" ? " (install SkyPilot where the controller runs to use catalog prices)"
        : d.provisioner === "kubernetes" ? " (give its node type a price in kubernetes.node_types to use that)" : "");
  const how = d.provisioner === "kubernetes" ? `kubernetes, ${d.spec?.kubernetes?.api === "llm-d" ? "llm-d" : "KServe"}` : d.provisioner ?? "manual";
  const covered = burn ? (d.demand_usd_per_hour ?? 0) / burn : null;
  const row = (label: string, value: React.ReactNode) => <Fragment key={label}><div className="k">{label}</div><div className="v">{value}</div></Fragment>;
  const wide = running && (d.replicas_info ?? []).length > 0;
  const costs = (
    <div>
      <Text className="label" mb={6}>Cost and demand</Text>
      <div className="kv">
        {running && row("Running for", `${ago(d.started_at)}${d.warm_at ? ` · serving ${ago(d.warm_at)}` : " · not serving yet"}`)}
        {running && row("Burn rate", `${usd(burn)}/h for ${d.replicas ?? 0} replica${d.replicas === 1 ? "" : "s"}`)}
        {row("Price per replica", <span data-tip={priceNote} style={{ textDecoration: "underline dotted" }}>{usd(p?.usd_per_hour ?? d.gpu_usd_per_hour ?? 0)}/h · {p?.source === "skypilot_catalog" ? "catalog" : p?.source === "node_type" ? `${p.node_type} rate` : "configured"}</span>)}
        {running && row("This run so far", usd(d.run_cost ?? 0))}
        {row("This month", usd(d.month_cost ?? 0))}
        {row("Replicas", `${running ? d.replicas ?? 0 : 0} running · ${d.desired ?? 0} demand wants${d.planned != null && d.planned !== d.desired ? ` · ${d.planned} allowed` : ""}`)}
        {row("Demand at API prices", `${usd(d.demand_usd_per_hour ?? 0)}/h · ${num(d.demand_tokens_per_second ?? 0)} tok/s`)}
        {covered != null && row("Demand covers", <Status kind={covered >= 1 ? "good" : "warn"}>{pct(covered)} of GPU cost</Status>)}
        {running && d.utilization != null && row("Utilization", pct(d.utilization) + (d.waiting ? ` · ${d.waiting} waiting` : ""))}
        {d.managed && d.start_needs && row("A start needs", (() => {
          const n = d.start_needs, cs = d.cold_start;
          const tip = `The replica's price plus the margin, with a ${n.cold_start_minutes} min cold start (paid, not serving) paid back over the `
            + `${n.hold_minutes} min hold. ` + (n.basis === "measured" ? `Average of the last ${cs?.samples.length ?? 0} starts.` : "No start measured yet: serve.cold_start_minutes.");
          return <span data-tip={tip} style={{ textDecoration: "underline dotted" }}>{usd(n.usd_per_hour)}/h of demand · {n.basis === "static" ? "no cold start" : `${n.cold_start_minutes} min cold start, ${n.basis}`}</span>;
        })())}
        {d.cold_start?.last && row("Last start", <span data-tip={phases(d.cold_start.last.phases) || undefined}>{mins(d.cold_start.last.total_seconds)}, {ago(d.cold_start.last.at)} ago</span>)}
        {d.spec?.weights_cache === "volume" && row("Weights cache", `${d.spec.weights_cache_gb ?? "?"} GB volume on ${d.spec.weights_cache_infra ?? "?"}`
          + (d.spec.weights_cache_usd_per_month != null ? ` · about ${usd(d.spec.weights_cache_usd_per_month)}/month, running or not` : " · price not known"))}
        {!running && d.last_run && row("Last run", `${when(d.last_run.started_at)}, ${ago(d.last_run.started_at, d.last_run.ended_at)} · ${usd(d.last_run.cost)}`)}
        {d.managed && row("Cache affinity", (() => {
          const a = affinityLine(d);
          return <>{a.text}{a.problem && <div><Status kind="warn">KV events: {a.problem}</Status></div>}</>;
        })())}
      </div>
    </div>
  );
  const where = (
    <div style={{ minWidth: 0 }}>
      <Text className="label" mb={6}>Where</Text>
      <Where d={d} />
      {running && d.status_message && <Status kind="warn" style={{ marginTop: 10, whiteSpace: "normal" }}>{d.status_message}</Status>}
      {(d.note || d.reason) && (d.reason?.includes("failed")
        ? <Status kind="bad" style={{ marginTop: 10, whiteSpace: "normal" }}>{d.note ?? d.reason}</Status>
        : <Text size="xs" c="dimmed" mt="xs">{d.note ?? d.reason}</Text>)}
      <Placements d={d} />
    </div>
  );
  return (
    <Panel h="100%" title={<><span className="mono">{d.backend}</span><Text span display="block" size="xs" c="dimmed" fw={400}>{d.model ?? "–"} · {how}</Text></>}
      right={<Group gap={8}>{d.state !== "disabled" && <ControlBadge c={d.control} />}<StateBadge state={d.state} /></Group>}
      footer={admin && d.managed && d.state !== "disabled" && (
        <Group gap="xs">
          <Button size="compact-sm" color="red" variant="outline" leftSection={<Icon name="stop" size={12} />} disabled={d.control?.mode === "off"} onClick={() => act("stop", d)}>Stop</Button>
          <Button size="compact-sm" variant="default" leftSection={<Icon name="pin" size={12} />} onClick={() => act("pin", d)}>Pin replicas</Button>
          {d.control && !d.control.fleet && <Button size="compact-sm" variant="subtle" color="gray" leftSection={<Icon name="auto" size={12} />} onClick={() => onAuto(d)}>Back to automatic</Button>}
        </Group>
      )}>
      {wide ? (
        <Grid gap="lg">
          <Grid.Col span={{ base: 12, lg: 8 }}>{where}</Grid.Col>
          <Grid.Col span={{ base: 12, lg: 4 }}>{costs}</Grid.Col>
        </Grid>
      ) : <Stack gap="md">{where}{costs}</Stack>}
    </Panel>
  );
}

function describe(e: GpuEvent): string {
  const x = e.detail as Record<string, string | number | boolean | null>;
  switch (e.kind) {
    case "state": return `${String(x.from).replace("_", " ")} → ${String(x.to).replace("_", " ")}${x.reason ? `: ${x.reason}` : ""}`;
    case "scale": return `scaled to ${x.replicas}${x.reason && x.reason !== "demand" ? ` (${x.reason})` : ""}`;
    case "control":
      return x.mode === "auto" ? `returned to automatic${x.reason ? `: ${x.reason}` : ""}` : x.mode === "on" ? `pinned to ${x.replicas}${x.hours ? ` for ${x.hours}h` : ""}`
        : `${x.drain ? "stop after draining" : "stop now"}${x.hours ? ` for ${x.hours}h` : ""}` + (x.reason ? `: ${x.reason}` : "");
    case "stop_all": return `stopped all GPUs${x.reason ? `: ${x.reason}` : ""}`;
    case "resume": return "resumed the GPU fleet";
    case "error": return `${x.action} failed: ${x.error}`;
    case "status": return String(x.message);
    case "replica_ready":
      return `${x.cluster ?? "replica"} ready in ${mins(Number(x.total_seconds))}` + ((e.detail.phases as StartPhase[] | undefined)?.length ? `: ${phases(e.detail.phases as StartPhase[])}` : "");
    case "volume": return `made the weights cache volume ${x.volume} (${x.size_gb} GB on ${x.infra}${x.usd_per_month != null ? `, about ${usd(Number(x.usd_per_month))}/month` : ""})`;
    case "volume_deleted": return `deleted the weights cache volume ${x.volume}`;
    case "replica_failed":
      return `${x.cluster}: ${x.reason}` + (x.driver_version ? ` (driver ${x.driver_version}, CUDA ${x.cuda_version}, image CUDA ${x.image_cuda ?? "?"})` : "");
    default: return e.kind;
  }
}

export function Gpus({ api, me }: { api: Api; me: Me }) {
  const res = useData(() => api.gpu(), [api], 10_000);
  const events = useData(() => api.gpuEvents(50), [api], 15_000);
  const [target, setTarget] = useState<{ kind: "stop" | "pin" | "stop-all"; d?: Deployment } | null>(null);
  const admin = me.role === "admin";
  const r = res.data;
  const deps = r?.deployments ?? [];
  const reload = () => { void res.reload(); void events.reload(); };
  const auto = async (d: Deployment) => {
    try { shown(await api.gpuControl(d.backend, { mode: "auto" }), `${d.backend} back to automatic`); reload(); } catch (e) { failed(e); }
  };
  const resume = async () => { try { shown(await api.resumeGpus(), "GPU fleet resumed"); reload(); } catch (e) { failed(e); } };
  const byBackend = new Map<string, number>();
  for (const x of r?.daily ?? []) byBackend.set(x.backend, (byBackend.get(x.backend) ?? 0) + x.cost);
  const hours = (r?.daily ?? []).reduce((a, x) => a + x.replica_hours, 0);
  const gm = r?.gpu_month;
  return (
    <>
      <PageTitle title="GPUs" sub="Self-hosted deployments, the replicas they run and what they cost">
        <Group gap="xs">
          <ControllerBadge c={r?.controller} />
          {admin && !r?.paused && deps.some((d) => d.managed) && <Button size="sm" color="red" variant="outline" leftSection={<Icon name="stop" size={13} />} onClick={() => setTarget({ kind: "stop-all" })}>Stop all GPUs</Button>}
        </Group>
      </PageTitle>
      <Stack gap="md">
        <Loading loading={res.loading} error={res.error} />
        {r?.paused && (
          <Alert color="red" icon={<Icon name="stop" size={16} />} title="All GPUs are stopped">
            <Group justify="space-between" wrap="nowrap">
              <Text size="sm">By {r.paused.by ?? "an admin"} at {when(r.paused.at)}{r.paused.reason ? `: ${r.paused.reason}` : ""}. No GPU starts until the fleet is resumed; traffic goes to APIs.</Text>
              {admin && <Button size="xs" variant="white" color="red" onClick={resume}>Resume</Button>}
            </Group>
          </Alert>
        )}
        {r && !r.controller.alive && r.fleet.replicas > 0 && (
          <Alert color="red" icon={<Icon name="bad" size={16} />} title="Nothing is managing running GPUs">The capacity controller isn't running, so GPUs won't scale down or stop on their own. Start it with <code>meritorder controller</code>.</Alert>
        )}
        {r && (
          <SimpleGrid cols={{ base: 2, md: 4 }}>
            <Kpi label="Running now" value={`${r.fleet.replicas} replica${r.fleet.replicas === 1 ? "" : "s"}`} sub={`${usd(r.fleet.usd_per_hour)}/h burn rate`} />
            <Kpi label="GPU spend this month" value={usd(gm?.spent ?? 0)} sub={gm?.budget ? `of ${usd(gm.budget)} budget (${pct((gm.spent ?? 0) / gm.budget)})` : "no GPU budget set"}
            />
            <Kpi label="Last 30 days" value={usd([...byBackend.values()].reduce((a, b) => a + b, 0))} sub={`${hours < 10 ? hours.toFixed(1) : num(hours)} replica-hours`} />
            <Kpi label="Fleet limits" value={r.limits.max_gpu_usd_per_hour != null ? `${usd(r.limits.max_gpu_usd_per_hour)}/h` : "none"}
              sub={r.limits.max_gpu_replicas != null ? `max ${r.limits.max_gpu_replicas} replicas` : "set them on Controls"} />
          </SimpleGrid>
        )}
        {!res.loading && !deps.length && <Panel title="Self-hosted deployments"><Text size="sm" c="dimmed">No self-hosted backends are configured. Add one with a <code>serve:</code> section in Config.</Text></Panel>}
        {/* A deployment with replicas takes the full width (replicas left, cost and demand right); idle ones pair up. */}
        {deps.filter((d) => (d.replicas_info ?? []).length > 0 && d.state !== "api_only" && d.state !== "disabled").map((d) =>
          <DeploymentCard key={d.backend} d={d} admin={admin} act={(kind, dd) => setTarget({ kind, d: dd })} onAuto={auto} />)}
        <SimpleGrid cols={{ base: 1, lg: 2 }}>
          {deps.filter((d) => !((d.replicas_info ?? []).length > 0 && d.state !== "api_only" && d.state !== "disabled")).map((d) =>
            <DeploymentCard key={d.backend} d={d} admin={admin} act={(kind, dd) => setTarget({ kind, d: dd })} onAuto={auto} />)}
        </SimpleGrid>
        <SimpleGrid cols={{ base: 1, lg: 2 }}>
          <Panel title="GPU cost by deployment, last 30 days" footer="Replica time × each replica's price: SkyPilot's catalog for its cloud, region, and instance, or on Kubernetes the configured rate for its node type (the configured price when neither is known), from when provisioning starts until the replicas are down. Catalog prices are list prices: discounts, disks, and egress aren't included.">
            <HBars items={[...byBackend.entries()].sort((a, b) => b[1] - a[1]).map(([label, value]) => ({ label, value }))} />
            {(r?.volumes ?? []).length > 0 && (
              <Text size="xs" c="dimmed" mt="sm">
                Weights cache volumes, billed every hour they exist and counted here at their monthly price:{" "}
                {(r?.volumes ?? []).map((v) => `${v.name} (${v.size_gb ?? "?"} GB, ${v.usd_per_month != null ? `${usd(v.usd_per_month)}/month` : "price not known"})`).join(", ")}.
                {" "}Only <code>meritorder gpu volumes delete</code> removes one.
              </Text>
            )}
          </Panel>
          <Panel title="Activity">
            <Loading loading={events.loading} error={events.error} />
            {!events.loading && !(events.data?.events ?? []).length && <Text size="sm" c="dimmed">No GPU activity yet.</Text>}
            <Stack gap={8} mah={380} style={{ overflowY: "auto" }}>
              {(events.data?.events ?? []).map((e) => (
                <Group key={e.id} gap={10} wrap="nowrap" align="flex-start">
                  <Text size="xs" c="dimmed" className="mono" w={110} style={{ flexShrink: 0 }}>{new Date(e.ts).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}</Text>
                  {e.kind === "error" || e.kind === "replica_failed"
                    ? <Text span c="var(--mo-bad)" style={{ flexShrink: 0, display: "inline-flex", paddingTop: 1 }} aria-label="Error"><Icon name="bad" size={12} /></Text>
                    : <Text span c="dimmed" style={{ flexShrink: 0, width: 12, textAlign: "center" }} aria-hidden="true">·</Text>}
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <Text size="sm" c={e.kind === "error" || e.kind === "replica_failed" ? "var(--mo-bad)" : "var(--mo-fg-2)"}>
                      <Text span className="mono" fw={600} c="var(--mantine-color-text)" inherit>{e.backend ?? "fleet"}</Text> {describe(e)} <Text span c="dimmed" inherit>· {e.actor === "controller" ? "controller" : e.actor}</Text>
                    </Text>
                    {Array.isArray(e.detail.tail) && e.detail.tail.length > 0 && (
                      <details style={{ fontSize: "var(--mantine-font-size-xs)" }}>
                        <summary>Last {e.detail.tail.length} lines of output ({String(e.detail.source)})</summary>
                        <pre className="mono" style={{ whiteSpace: "pre-wrap", margin: 0 }}>{(e.detail.tail as string[]).join("\n")}</pre>
                      </details>
                    )}
                  </div>
                </Group>
              ))}
            </Stack>
          </Panel>
        </SimpleGrid>
        <WhatIfPanel api={api} />
      </Stack>
      <ControlModal key={target ? `${target.kind}-${target.d?.backend}` : "none"} api={api} target={target} onClose={() => setTarget(null)} onDone={reload} all={deps} />
    </>
  );
}

export function StopAllButton({ api, onDone }: { api: Api; onDone: () => void }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <Button size="sm" color="red" variant="outline" leftSection={<Icon name="stop" size={13} />} onClick={() => setOpen(true)}>Stop all GPUs</Button>
      {open && <ControlModal api={api} target={{ kind: "stop-all" }} onClose={() => setOpen(false)} onDone={onDone} />}
    </>
  );
}
