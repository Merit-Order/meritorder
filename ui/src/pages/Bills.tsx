import { Alert, Badge, Button, Group, SegmentedControl, SimpleGrid, Stack, Table, Text } from "@mantine/core";
import { notifications } from "@mantine/notifications";
import { useState } from "react";
import type { Api, BillRow, Me, Range } from "../api";
import { compact, usd } from "../api";
import { AnsweredBy, Kpi, Loading, Panel, StatusBadge, type StatusKind } from "../components/common";
import { Icon } from "../components/icons";
import { useData } from "../hooks";

const signed = (v: number) => (v > 0 ? `+${usd(v)}` : usd(v));
const percent = (v: number | null) =>
  v == null ? "–" : Math.abs(v) >= 1000 ? (v > 0 ? ">+999%" : "<-999%") : `${v > 0 ? "+" : ""}${v.toFixed(1)}%`;
const sum = (o: Record<string, number>) => Object.values(o).reduce((a, b) => a + b, 0);

const STATUS: Record<BillRow["status"], { label: string; kind: StatusKind; help: string }> = {
  matched: { label: "matched", kind: "good", help: "On the bill and in the ledger" },
  partial: { label: "partial", kind: "clock", help: "The provider is still adding to this day's bill: never alerted on" },
  not_billed: { label: "not on the bill", kind: "warn", help: "Recorded, but no bill line matched this model (a name the matcher missed, or a bill not fetched)" },
  not_recorded: { label: "not in the ledger", kind: "warn", help: "Billed, but nothing went through Merit Order: other usage in the same account" },
};

function Tokens({ r }: { r: BillRow }) {
  const kinds = Object.keys(r.billed_tokens);
  if (!kinds.length) return <>–</>;
  const recorded = kinds.reduce((a, k) => a + (r.recorded_tokens[k] ?? 0), 0);
  const detail = kinds.map((k) => `${k}: ${compact(r.recorded_tokens[k] ?? 0)} recorded, ${compact(r.billed_tokens[k])} billed`).join("\n");
  return <span title={detail}>{compact(recorded)} / {compact(sum(r.billed_tokens))}</span>;
}

/** The ledger against providers' own cost data (Reports → Bills; finance and admins). */
export function Bills({ api, me, range }: { api: Api; me: Me; range: Range }) {
  const [by, setBy] = useState<"day" | "month">("day");
  const [busy, setBusy] = useState(false);
  const res = useData(() => api.bills(range, by), [api, range.start, range.end, by]);
  const d = res.data;

  const fetchNow = async () => {
    setBusy(true);
    try {
      const r = await api.refreshBills();
      const failed = Object.entries(r.sources).filter(([, s]) => !s.ok);
      notifications.show({ color: failed.length ? "yellow" : "green", title: failed.length ? `${failed.length} source(s) failed` : "Bills fetched",
        message: failed.length ? failed.map(([n, s]) => `${n}: ${s.error}`).join("; ") : `${Object.keys(r.sources).length} source(s) up to date.` });
      await res.reload();
    } catch (e) {
      notifications.show({ color: "red", title: "Couldn't fetch the bills", message: e instanceof Error ? e.message : String(e) });
    } finally {
      setBusy(false);
    }
  };

  if (res.loading || res.error || !d) return <Loading loading={res.loading} error={res.error} />;
  if (!d.sources.length) {
    return (
      <Panel title="Ledger against bills">
        <Text size="sm">No bill sources are configured. Add <code>billing.sources</code> to the configuration (Anthropic and OpenAI admin
          keys, AWS Cost Explorer or a CUR 2.0 export in S3, the Google Cloud billing export in BigQuery, Azure Cost Management or its
          exports in Blob Storage) and the worker fetches them every few hours. docs/08-reporting.md lists the keys and permissions each
          one needs.</Text>
      </Panel>
    );
  }
  const s = d.settled;
  return (
    <Stack gap="md">
      <SimpleGrid cols={{ base: 1, xs: 2, md: 4 }}>
        <Kpi label="Recorded" value={usd(s.recorded)} sub="the ledger, settled days" />
        <Kpi label="Billed" value={usd(s.billed)} sub="the providers' own cost data" />
        <Kpi label="Difference" value={signed(s.difference)}
          sub={`${percent(s.pct)} of recorded; alerts at ${d.thresholds.alert_pct}% and ${usd(d.thresholds.alert_min_usd)}`} />
        <Kpi label="Not matched to a model" value={usd(s.billed_unmatched)} sub={`credits ${usd(s.adjustments)}, left out ${usd(s.left_out)}`} />
      </SimpleGrid>
      <AnsweredBy by={d.answered_by} />

      <Panel title="Sources" right={me.role === "admin" && <Button size="xs" variant="default" loading={busy} onClick={fetchNow}>Fetch bills now</Button>}>
        <Table.ScrollContainer minWidth={900}>
          <Table verticalSpacing={6} fz="sm">
            <Table.Thead>
              <Table.Tr><Table.Th>Source</Table.Th><Table.Th>Covers</Table.Th><Table.Th>Fetched</Table.Th><Table.Th>Complete through</Table.Th>
                <Table.Th ta="right">Recorded</Table.Th><Table.Th ta="right">Billed</Table.Th><Table.Th ta="right">Difference</Table.Th><Table.Th>State</Table.Th></Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {d.sources.map((x) => (
                <Table.Tr key={x.name}>
                  <Table.Td><b>{x.name}</b> <Text span size="xs" c="dimmed">{x.type}</Text></Table.Td>
                  <Table.Td>{x.backends.join(", ") || "–"}</Table.Td>
                  <Table.Td className="mono">{x.fetched_at ? new Date(x.fetched_at).toLocaleString() : "never"}</Table.Td>
                  <Table.Td className="mono">{x.complete_through ?? "–"}{x.partial_days ? <Text span size="xs" c="dimmed"> ({x.partial_days} partial)</Text> : null}</Table.Td>
                  <Table.Td className="num">{usd(x.settled.recorded)}</Table.Td>
                  <Table.Td className="num">{usd(x.settled.billed)}</Table.Td>
                  <Table.Td className="num">{signed(x.settled.difference)} <Text span size="xs" c="dimmed">{percent(x.settled.pct)}</Text></Table.Td>
                  <Table.Td>
                    {x.ok === false ? <StatusBadge kind="bad" title={x.error ?? ""}>failed</StatusBadge> : x.ok ? <StatusBadge kind="good">ok</StatusBadge> : <StatusBadge kind="off">not fetched</StatusBadge>}
                    {!x.filtered && <Badge ml={6} variant="default" title="The whole account is compared: other usage in it shows as gaps. Set workspace_ids, project_ids, accounts or a resource group.">whole account</Badge>}
                  </Table.Td>
                </Table.Tr>
              ))}
            </Table.Tbody>
          </Table>
        </Table.ScrollContainer>
        {d.sources.filter((x) => x.ok === false || x.notes.length).map((x) => (
          <Alert key={x.name} mt="sm" color={x.ok === false ? "red" : "gray"} icon={<Icon name={x.ok === false ? "bad" : "info"} size={16} />} title={x.name}>
            {x.ok === false && <Text size="sm">{x.error}{x.fetched_at ? " The last bill fetched is still shown." : ""}</Text>}
            {x.notes.map((n) => <Text key={n} size="sm">{n}</Text>)}
          </Alert>
        ))}
      </Panel>

      <Panel title={`By ${by}, provider and model`} right={
        <Group gap="sm">
          <SegmentedControl size="xs" value={by} onChange={(v) => setBy(v as "day" | "month")} data={[{ value: "day", label: "Day" }, { value: "month", label: "Month" }]} />
          <Button size="xs" variant="default" onClick={() => api.downloadBills(range, by)}>Download CSV</Button>
        </Group>}>
        <Table.ScrollContainer minWidth={900}>
          <Table>
            <Table.Thead>
              <Table.Tr><Table.Th>{by === "day" ? "Day" : "Month"}</Table.Th><Table.Th>Provider</Table.Th><Table.Th>Model</Table.Th>
                <Table.Th ta="right">Recorded</Table.Th><Table.Th ta="right">Billed</Table.Th><Table.Th ta="right">Difference</Table.Th>
                <Table.Th ta="right">%</Table.Th><Table.Th ta="right" title="Recorded / billed, for the token kinds the bill reports">Tokens</Table.Th>
                <Table.Th>Status</Table.Th></Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {d.rows.map((r) => (
                <Table.Tr key={`${r.period}-${r.source}-${r.model}`}>
                  <Table.Td className="mono">{r.period}</Table.Td>
                  <Table.Td>{r.source}</Table.Td>
                  <Table.Td title={Object.entries(r.billed_by_kind).map(([k, v]) => `${k}: ${usd(v)}`).join("\n")}><span>{r.model}</span>
                    {(r.recorded_batch > 0 || r.billed_batch > 0) && <Badge ml={6} size="sm" variant="default"
                      title="Through the provider's batch API, included in the totals: recorded / billed">batch {usd(r.recorded_batch)} / {usd(r.billed_batch)}</Badge>}</Table.Td>
                  <Table.Td className="num">{usd(r.recorded)}</Table.Td>
                  <Table.Td className="num">{usd(r.billed)}</Table.Td>
                  <Table.Td className="num" c={r.over ? "var(--mo-bad)" : undefined}>{signed(r.difference)}</Table.Td>
                  <Table.Td className="num" c={r.over ? "var(--mo-bad)" : undefined}>{percent(r.pct)}</Table.Td>
                  <Table.Td className="num"><Tokens r={r} /></Table.Td>
                  <Table.Td><StatusBadge size="sm" kind={r.over ? "bad" : STATUS[r.status].kind} title={STATUS[r.status].help}>
                    {r.over ? "over threshold" : STATUS[r.status].label}</StatusBadge></Table.Td>
                </Table.Tr>
              ))}
            </Table.Tbody>
          </Table>
        </Table.ScrollContainer>
        {!d.rows.length && <Text c="dimmed" size="sm" mt="sm">Nothing recorded or billed in this period.</Text>}
      </Panel>

      {d.unmatched.length > 0 && (
        <Panel title="On the bills, not matched to a model" right={<Text size="xs" c="dimmed">Not matched counts in billed; left out doesn't</Text>}>
          <Table.ScrollContainer minWidth={800}>
            <Table verticalSpacing={5} fz="sm">
              <Table.Thead>
                <Table.Tr><Table.Th>Provider</Table.Th><Table.Th>Bill line</Table.Th><Table.Th ta="right">Billed</Table.Th><Table.Th ta="right">Days</Table.Th>
                  <Table.Th>Why</Table.Th></Table.Tr>
              </Table.Thead>
              <Table.Tbody>
                {d.unmatched.map((u) => (
                  <Table.Tr key={`${u.source}-${u.line}-${u.why}`}>
                    <Table.Td>{u.source}</Table.Td>
                    <Table.Td className="mono">{u.line}</Table.Td>
                    <Table.Td className="num">{usd(u.cost)}</Table.Td>
                    <Table.Td className="num" title={`${u.first} to ${u.last}`}>{u.days}</Table.Td>
                    <Table.Td>{u.status === "excluded" ? <Badge size="sm" mr={6} variant="default">left out</Badge> : <StatusBadge size="sm" mr={6} kind="warn">not matched</StatusBadge>}{u.why}</Table.Td>
                  </Table.Tr>
                ))}
              </Table.Tbody>
            </Table>
          </Table.ScrollContainer>
        </Panel>
      )}

      {d.gpu.length > 0 && (
        <Panel title="GPU time against the cloud bills" right={<Text size="xs" c="dimmed">Recorded at SkyPilot catalog prices; billed for instances tagged with the deployment</Text>}>
          <Table verticalSpacing={5} fz="sm">
            <Table.Thead>
              <Table.Tr><Table.Th>{by === "day" ? "Day" : "Month"}</Table.Th><Table.Th>Deployment</Table.Th><Table.Th ta="right">Recorded</Table.Th>
                <Table.Th ta="right">Billed</Table.Th><Table.Th ta="right">Difference</Table.Th><Table.Th ta="right">%</Table.Th><Table.Th>Clouds</Table.Th></Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {d.gpu.map((g) => (
                <Table.Tr key={`${g.period}-${g.deployment}`}>
                  <Table.Td className="mono">{g.period}{g.partial && <Badge ml={6} size="xs" color="gray">partial</Badge>}</Table.Td>
                  <Table.Td>{g.deployment}</Table.Td>
                  <Table.Td className="num">{usd(g.recorded)}</Table.Td>
                  <Table.Td className="num">{usd(g.billed)}</Table.Td>
                  <Table.Td className="num" c={g.over ? "var(--mo-bad)" : undefined}>{signed(g.difference)}</Table.Td>
                  <Table.Td className="num" c={g.over ? "var(--mo-bad)" : undefined}>{percent(g.pct)}</Table.Td>
                  <Table.Td>{Object.entries(g.billed_by_source).map(([k, v]) => `${k} ${usd(v)}`).join(", ") || "–"}</Table.Td>
                </Table.Tr>
              ))}
            </Table.Tbody>
          </Table>
        </Panel>
      )}

      {d.uncovered.length > 0 && (
        <Panel title="Not checked against a bill">
          <Table verticalSpacing={5} fz="sm">
            <Table.Tbody>
              {d.uncovered.map((u) => (
                <Table.Tr key={u.backend}>
                  <Table.Td>{u.backend}</Table.Td><Table.Td className="num">{usd(u.recorded)}</Table.Td><Table.Td c="dimmed">{u.why}</Table.Td>
                </Table.Tr>
              ))}
            </Table.Tbody>
          </Table>
        </Panel>
      )}
    </Stack>
  );
}
