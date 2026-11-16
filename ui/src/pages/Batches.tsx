import { Badge, Button, Group, SimpleGrid, Stack, Table, Text } from "@mantine/core";
import { notifications } from "@mantine/notifications";
import { useState } from "react";
import type { Api, BatchRow, BatchStatus, BatchStorage, Me, Range } from "../api";
import { num, usd } from "../api";
import { Kpi, Loading, Panel, StatusBadge, type StatusKind } from "../components/common";
import { useData } from "../hooks";

const STATUS: Record<BatchStatus, { label: string; kind: StatusKind; help: string }> = {
  in_progress: { label: "running", kind: "clock", help: "At the providers, or waiting for room on a GPU; the worker checks on it until it ends" },
  cancelling: { label: "cancelling", kind: "clock", help: "Cancelled; waiting for the providers to stop. What already ran is kept and billed" },
  finalizing: { label: "settling", kind: "clock", help: "The providers are done; its results are being written into the ledger" },
  completed: { label: "completed", kind: "good", help: "Every request ran (some may have failed on their own)" },
  cancelled: { label: "cancelled", kind: "off", help: "Cancelled; requests that ran before it stopped are kept and billed" },
  expired: { label: "expired", kind: "warn", help: "A provider (or a GPU, for requests with no other batch route) didn't run every request within 24 hours; the rest expired, unbilled" },
  failed: { label: "failed", kind: "bad", help: "No provider accepted it" },
};

function Counts({ b }: { b: BatchRow }) {
  const c = b.counts;
  const parts = [
    c.processing ? `${num(c.processing)} waiting` : "",
    c.errored ? `${num(c.errored)} failed` : "",
    c.canceled ? `${num(c.canceled)} cancelled` : "",
    c.expired ? `${num(c.expired)} expired` : "",
  ].filter(Boolean);
  return <span title={b.errors.join("\n")}>{num(c.succeeded)} / {num(b.requests)}{parts.length ? <Text span size="xs" c="dimmed"> ({parts.join(", ")})</Text> : null}</span>;
}

/** Where batch content is held, sealed, and the batch key's rotation (admins). */
function Storage({ s }: { s: BatchStorage }) {
  const until = s.previous_readable_until ? new Date(s.previous_readable_until).toLocaleString() : "";
  return (
    <Panel title="Held for batches" right={<Text size="xs" c="dimmed">Uploads until their batch is handed on, and GPU batch work, sealed with MERITORDER_BATCH_KEY</Text>}>
      <SimpleGrid cols={{ base: 1, sm: 3 }}>
        <Stack gap={2}>
          <Text size="xs" c="dimmed">Held in</Text>
          <Text className="mono" size="sm">{s.held_in === "database" ? "Postgres" : s.held_in}</Text>
          <Text size="xs" c="dimmed">{num(s.files_held)} uploads waiting{s.files_elsewhere ? `, ${num(s.files_elsewhere)} in an earlier place (still read there)` : ""}</Text>
        </Stack>
        <Stack gap={2}>
          <Text size="xs" c="dimmed">Batch key</Text>
          <Text className="mono" size="sm">{s.key ?? "not set: uploads are refused and batch work skips the GPUs"}</Text>
          {s.previous_key && <Text size="xs" c={s.previous_usable ? "dimmed" : "var(--mo-bad)"}>
            previous key {s.previous_key} {s.previous_usable ? `readable until ${until}` : `no longer read (since ${until})`};
            {" "}{num(s.files_under_previous)} uploads still sealed with it</Text>}
        </Stack>
        <Stack gap={2}>
          <Text size="xs" c="dimmed">Batch work on GPUs</Text>
          <Text size="sm">{s.gpu.enabled ? "on" : "off (batches.gpu.enabled)"}</Text>
          {s.gpu.enabled && <Text size="xs" c="dimmed">In running deployments' headroom at vLLM priority {s.gpu.priority};
            what hasn't run after {s.gpu.handoff_hours} h goes to a provider's batch API</Text>}
        </Stack>
      </SimpleGrid>
    </Panel>
  );
}

/** Batches through the providers' batch APIs and on self-hosted GPUs (Reports → Batches), scoped like the other reports. */
export function Batches({ api, me, range }: { api: Api; me: Me; range: Range }) {
  const res = useData(() => api.batches(range), [api, range.start, range.end], 30_000);
  const [busy, setBusy] = useState<string | null>(null);
  const d = res.data;

  const cancel = async (b: BatchRow) => {
    setBusy(b.id);
    try {
      await api.cancelBatch(b);
      notifications.show({ color: "green", title: "Batch cancelled", message: `${b.id}: the providers stop what hasn't run yet.` });
      await res.reload();
    } catch (e) {
      notifications.show({ color: "red", title: "Couldn't cancel the batch", message: e instanceof Error ? e.message : String(e) });
    } finally {
      setBusy(null);
    }
  };

  if (res.loading || res.error || !d) return <Loading loading={res.loading} error={res.error} />;
  const t = d.totals;
  const gpus = new Set(d.gpu_backends);
  return (
    <Stack gap="md">
      <SimpleGrid cols={{ base: 1, xs: 2, md: 5 }}>
        <Kpi label="Batches" value={num(t.batches)} sub={`${num(t.requests)} requests, ${num(t.running)} still running`} />
        <Kpi label="Spent" value={usd(t.actual_cost)} sub={`estimated ${usd(t.estimated_cost)}; ${usd(t.reserved_cost)} still reserved`} />
        <Kpi label="Batch discount" value={usd(t.savings_batch)} sub="the same requests at live prices, less what the providers' batch prices take off" />
        <Kpi label="On your GPUs" value={num(t.on_gpus)}
          sub={`requests; ${usd(t.savings_self_hosting)} below the batch price, in GPUs' headroom`} />
        <Kpi label="Saved in all" value={usd(t.savings_total)} accent sub="batch discount, GPUs and routing, against the team's default model" />
      </SimpleGrid>
      {d.storage && <Storage s={d.storage} />}
      <Panel title="Batches" right={<Text size="xs" c="dimmed">Results stay with the providers (the gateway holds GPU results, sealed); it deletes them {d.retention_days} days after a batch ends</Text>}>
        <Table.ScrollContainer minWidth={1180}>
          <Table>
            <Table.Thead>
              <Table.Tr><Table.Th>Created</Table.Th><Table.Th>Batch</Table.Th>{me.role !== "user" && <Table.Th>Who</Table.Th>}<Table.Th>Status</Table.Th>
                <Table.Th ta="right">Done / requests</Table.Th><Table.Th>Where</Table.Th><Table.Th ta="right">Estimated</Table.Th>
                <Table.Th ta="right">Cost</Table.Th><Table.Th ta="right">Batch discount</Table.Th><Table.Th ta="right">Saved</Table.Th><Table.Th /></Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {d.rows.map((b) => (
                <Table.Tr key={b.id}>
                  <Table.Td className="mono" style={{ whiteSpace: "nowrap" }}>{new Date(b.created_at).toLocaleString()}</Table.Td>
                  <Table.Td><Text className="mono" size="xs">{b.id}</Text>
                    <Text size="xs" c="dimmed">{b.format === "anthropic" ? "Anthropic Message Batches" : `OpenAI ${b.endpoint ?? ""}`}</Text></Table.Td>
                  {me.role !== "user" && <Table.Td>{b.user} <Text span size="xs" c="dimmed">{b.team}</Text></Table.Td>}
                  <Table.Td><StatusBadge size="sm" kind={STATUS[b.status].kind} title={STATUS[b.status].help}>{STATUS[b.status].label}</StatusBadge>
                    {b.cancelled_by && <Text size="xs" c="dimmed">by {b.cancelled_by}</Text>}</Table.Td>
                  <Table.Td className="num"><Counts b={b} /></Table.Td>
                  <Table.Td><Group gap={4}>{Object.entries(b.providers).map(([k, v]) => gpus.has(k)
                    ? <Badge key={k} size="sm" color="mo" title="A self-hosted GPU, in its headroom at low priority">GPU {k} {num(v)}</Badge>
                    : <Badge key={k} size="sm" variant="default">{k} {num(v)}</Badge>)}</Group></Table.Td>
                  <Table.Td className="num" title={b.reserved_cost ? `${usd(b.reserved_cost)} still reserved against budgets` : ""}>{usd(b.estimated_cost)}</Table.Td>
                  <Table.Td className="num">{b.status === "in_progress" || b.status === "cancelling" ? "–" : usd(b.actual_cost)}</Table.Td>
                  <Table.Td className="num">{usd(b.savings_batch)}</Table.Td>
                  <Table.Td className="num" title={b.on_gpus ? `${usd(b.savings_self_hosting)} of it from GPUs` : ""}>{usd(b.savings_total)}</Table.Td>
                  <Table.Td>{b.can_cancel && (me.role === "admin" || b.user === me.user) &&
                    <Button size="compact-xs" variant="outline" color="red" loading={busy === b.id} onClick={() => cancel(b)}>Cancel</Button>}</Table.Td>
                </Table.Tr>
              ))}
            </Table.Tbody>
          </Table>
        </Table.ScrollContainer>
        {!d.rows.length && <Text c="dimmed" size="sm" mt="sm">No batches in this period. Send work that can wait to <code>/v1/batches</code> (OpenAI) or
          <code> /v1/messages/batches</code> (Anthropic), or with <code>meritorder batch submit</code>: about half the price of live requests, near free
          on a GPU that's running with room.</Text>}
      </Panel>
    </Stack>
  );
}
