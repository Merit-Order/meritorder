import { Alert, Button, Group, Paper, Select, SimpleGrid, Stack, Switch, Table, Text, Title } from "@mantine/core";
import type { Api, ClickHouseStatus, Doc } from "../../api";
import { num } from "../../api";
import { Loading, StatusBadge } from "../../components/common";
import { Icon } from "../../components/icons";
import { useData } from "../../hooks";
import { EnvName, Num, Txt, lbl, rich, setIn } from "./fields";
import { REPORTING as H } from "./help";

const READS = [
  { value: "large", label: "Large reports: over 7 days, and tag breakdowns" },
  { value: "all", label: "All reports it can answer" },
  { value: "none", label: "None: copy only" },
];
const NEW_CLICKHOUSE = { url: "${MERITORDER_CLICKHOUSE_URL:-}", database: "${MERITORDER_CLICKHOUSE_DATABASE:-meritorder}",
  user: "${MERITORDER_CLICKHOUSE_USER:-default}", auth: { value_env: "MERITORDER_CLICKHOUSE_PASSWORD" } };

function bytes(n: number): string {
  const units = ["bytes", "KB", "MB", "GB", "TB"];
  let i = 0;
  while (n >= 1000 && i < units.length - 1) { n /= 1000; i++; }
  return i ? `${n.toFixed(1)} ${units[i]}` : `${n} bytes`;
}

const when = (iso?: string | null) => (iso ? new Date(iso).toLocaleString() : "–");

/** What the running gateways see right now: Postgres, and ClickHouse with how far its copy has got. */
function Status({ api }: { api: Api }) {
  const res = useData(() => api.reporting(), [api], 30_000);
  const d = res.data;
  const ch: ClickHouseStatus | null = d?.clickhouse ?? null;
  return (
    <Paper>
      <Group justify="space-between" mb="xs">
        <Title order={4} fz="sm">Reports come from {ch ? "Postgres and ClickHouse" : d?.store === "memory" ? "memory (no database)" : "Postgres"}</Title>
        <Button size="xs" variant="default" onClick={() => void res.reload()}>Check again</Button>
      </Group>
      <Loading loading={res.loading && !d} error={res.error} />
      {d && (
        <Stack gap="xs">
          <Group gap="xs">
            <Text size="sm" w={110}>Postgres</Text>
            <StatusBadge kind={d.postgres.ok ? "good" : "bad"}>{d.postgres.ok ? "reachable" : "not reachable"}</StatusBadge>
            <Text size="xs" c="dimmed">{d.postgres.error ?? "the record: every request, budgets, keys and configuration"}</Text>
          </Group>
          {d.note && <Alert color="yellow" icon={<Icon name="warn" size={16} />}>{d.note}</Alert>}
          {!ch && !d.note && <Text size="sm" c="dimmed">ClickHouse isn't set up. Postgres handles millions of requests a month; for more, set it up below.</Text>}
          {ch && (
            <>
              <Group gap="xs">
                <Text size="sm" w={110}>ClickHouse</Text>
                <StatusBadge kind={ch.ok ? "good" : "bad"}>{ch.ok ? "reachable" : "not reachable"}</StatusBadge>
                <Text size="xs" c="dimmed" className="mono">{ch.url}, database {ch.database}, user {ch.user}{ch.password_set ? ", password set" : ", no password"}
                  {ch.version ? ` · version ${ch.version}, ${ch.latency_ms} ms` : ""}</Text>
              </Group>
              {ch.error && <Alert color="red" icon={<Icon name="bad" size={16} />} title="ClickHouse can't be reached">{ch.error}</Alert>}
              <Alert color={ch.answering ? "green" : "gray"} icon={<Icon name={ch.answering ? "ok" : "info"} size={16} />}>
                {ch.answering ? (ch.reads === "all" ? "Every report it can answer reads ClickHouse." : "Reports over more than 7 days and tag breakdowns read ClickHouse.")
                  : `Postgres answers every report: ${ch.answers}`}
              </Alert>
              <Table verticalSpacing={4} fz="sm" maw={720}>
                <Table.Tbody>
                  <Table.Tr><Table.Td w={200}>Holds every request from</Table.Td>
                    <Table.Td>{ch.copy.from?.startsWith("1970-01-01") ? "the ledger's first" : when(ch.copy.from)}</Table.Td></Table.Tr>
                  <Table.Tr><Table.Td>Complete up to</Table.Td><Table.Td>{when(ch.copy.through)}
                    {ch.copy.behind_seconds != null ? ` (${num(ch.copy.behind_seconds)} s behind)` : ""}</Table.Td></Table.Tr>
                  {ch.copy.error && <Table.Tr><Table.Td>Last copy failed</Table.Td><Table.Td c="var(--mo-bad)">{when(ch.copy.error_at)}: {ch.copy.error}</Table.Td></Table.Tr>}
                  {!!ch.copy.repaired?.length && <Table.Tr><Table.Td>Hours copied again</Table.Td><Table.Td>{ch.copy.repaired.map(when).join(", ")}</Table.Td></Table.Tr>}
                  {!!ch.copy.ahead?.length && <Table.Tr><Table.Td>Rows Postgres no longer has</Table.Td>
                    <Table.Td>{ch.copy.ahead.map(when).join(", ")}: ClickHouse keeps them</Table.Td></Table.Tr>}
                  {Object.entries(ch.tables).map(([name, t]) => (
                    <Table.Tr key={name}><Table.Td>{name === "requests" ? "Ledger rows" : name === "usage_hourly" ? "Hourly rollup rows" : name}</Table.Td>
                      <Table.Td>{num(t.rows)} ({bytes(t.bytes_on_disk)} on disk)</Table.Td></Table.Tr>
                  ))}
                </Table.Tbody>
              </Table>
              {!ch.copy.from && <Text size="xs" c="dimmed">The worker copies new requests. To copy what the ledger already holds, run <code>meritorder clickhouse backfill</code>.</Text>}
            </>
          )}
        </Stack>
      )}
    </Paper>
  );
}

export function ReportingEditor({ draft, setDraft, api }: { draft: Doc; setDraft: (d: Doc) => void; api: Api }) {
  const ch: Doc | undefined = draft.reporting?.clickhouse ?? undefined;
  const set = (path: (string | number)[], v: unknown) => setDraft(setIn(draft, ["reporting", "clickhouse", ...path], v));
  return (
    <Stack gap="md" maw={980}>
      <Text size="sm" c="dimmed">{rich(H.section)}</Text>
      <Status api={api} />
      <Paper>
        <Group justify="space-between" mb="xs">
          <Title order={4} fz="sm">ClickHouse copy of the ledger</Title>
          <Switch label="Set up" checked={!!ch} onChange={(e) => setDraft(e.currentTarget.checked
            ? setIn(draft, ["reporting", "clickhouse"], NEW_CLICKHOUSE)
            : setIn(draft, ["reporting"], undefined))} />
        </Group>
        {!ch && <Text size="sm" c="dimmed">Off: Postgres answers every report.</Text>}
        {ch && (
          <Stack gap="md">
            <SimpleGrid cols={{ base: 1, sm: 2 }}>
              <Txt label="URL" value={ch.url} onChange={(v) => set(["url"], v)} mono help={H.url} placeholder="${MERITORDER_CLICKHOUSE_URL:-}"
                description="http://clickhouse:8123, or https://<host>:8443 for ClickHouse Cloud" />
              <Txt label="Database" value={ch.database} onChange={(v) => set(["database"], v)} mono help={H.database} placeholder="meritorder" />
              <Txt label="User" value={ch.user} onChange={(v) => set(["user"], v)} mono help={H.user} placeholder="default" />
              <Select {...lbl("Reports that read it", H.reads)} data={READS} value={ch.reads ?? "large"} allowDeselect={false}
                onChange={(v) => set(["reads"], v === "large" ? undefined : v)} />
              <EnvName label="Password environment variable" value={ch.auth?.value_env} onChange={(v) => set(["auth", "value_env"], v)} help={H.password_env} />
              <Txt label="Password file" value={ch.auth?.value_file} onChange={(v) => set(["auth", "value_file"], v)} mono help={H.password_file}
                placeholder="/run/secrets/meritorder/MERITORDER_CLICKHOUSE_PASSWORD" />
              <Num label="Keep ledger rows (days)" value={ch.ttl_days} onChange={(v) => set(["ttl_days"], v)} min={1} placeholder="forever" help={H.ttl_days} />
              <Num label="Keep the hourly rollup (days)" value={ch.rollup_ttl_days} onChange={(v) => set(["rollup_ttl_days"], v)} min={1} placeholder="forever"
                help={H.rollup_ttl_days} />
              <Num label="Copy every (seconds)" value={ch.copy_seconds} onChange={(v) => set(["copy_seconds"], v)} min={1} placeholder="30" help={H.copy_seconds} />
              <Num label="Rows per insert" value={ch.batch_rows} onChange={(v) => set(["batch_rows"], v)} min={1} placeholder="50,000" help={H.batch_rows} />
              <Num label="Timeout (seconds)" value={ch.timeout_seconds} onChange={(v) => set(["timeout_seconds"], v)} min={1} placeholder="60" help={H.timeout_seconds} />
            </SimpleGrid>
          </Stack>
        )}
      </Paper>
    </Stack>
  );
}
