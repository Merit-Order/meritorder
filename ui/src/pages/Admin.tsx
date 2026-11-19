import { Alert, Badge, Button, Code, CopyButton, Group, NumberInput, Paper, Select,
  SimpleGrid, Stack, Table, Text, TextInput, Title } from "@mantine/core";
import { notifications } from "@mantine/notifications";
import { useState } from "react";
import type { Api, ApiKey, Me, NewKeyResult, TeamStatus } from "../api";
import { num, pct, storedLine, usd } from "../api";
import { HBars } from "../components/charts";
import { Kpi, Loading, PageTitle, Panel, SpendSplit, Status, StatusBadge } from "../components/common";
import { Icon } from "../components/icons";
import { Meter } from "./Controls";
import { useData } from "../hooks";

const errorText = (e: unknown) => (e instanceof Error ? e.message : String(e));

// ------------------------------------------------------------------ Teams

export function Teams({ api, me, onNavigate }: { api: Api; me: Me; onNavigate: (p: string) => void }) {
  const res = useData(() => api.teams(), [api], 30_000);
  return (
    <>
      <PageTitle title="Teams & budgets">{me.role === "admin" && <Button size="sm" variant="default" onClick={() => onNavigate("config")}>Edit limits in Config</Button>}</PageTitle>
      <Loading loading={res.loading} error={res.error} />
      <SimpleGrid cols={{ base: 1, md: 2, xl: 3 }}>
        {(res.data?.teams ?? []).map((t: TeamStatus) => {
          const used = t.budget_usd_month ? t.spend / t.budget_usd_month : null;
          return (
            <Paper key={t.name}>
              <Group justify="space-between" mb="xs">
                <Title order={3} fz={16}>{t.name}</Title>
                {!t.configured && <Badge variant="default" title="Uses the default team's settings">default settings</Badge>}
              </Group>
              <Group justify="space-between">
                <div className="kpi-v">{usd(t.spend)}</div>
                <Text size="sm" c="dimmed">{t.budget_usd_month ? `of ${usd(t.budget_usd_month)}` : "no budget"}</Text>
              </Group>
              {used != null && <div className={`mini ${used >= 1 ? "bad" : used >= 0.8 ? "warn" : "acc"}`} style={{ marginTop: 8 }}><i style={{ width: `${Math.min(used, 1) * 100}%` }} /></div>}
              {used != null && used >= 0.8 && <Status kind={used >= 1 ? "bad" : "warn"} style={{ marginTop: 6 }}>{pct(used)} of its budget{used >= 1 ? ": requests refused" : ""}</Status>}
              <Text size="xs" c="dimmed" mt={4}>{num(t.requests)} requests · saved <Text span c="var(--mo-accent-text)" inherit>{usd(t.saved)}</Text></Text>
              <Table fz="xs" mt="sm" verticalSpacing={2} withRowBorders={false}>
                <Table.Tbody>
                  <Table.Tr><Table.Td c="dimmed">Per person / month</Table.Td><Table.Td ta="right">{t.user_budget_usd_month ? usd(t.user_budget_usd_month) : "–"}</Table.Td></Table.Tr>
                  <Table.Tr><Table.Td c="dimmed">Per session</Table.Td><Table.Td ta="right">{t.max_session_usd ? usd(t.max_session_usd) : "–"}</Table.Td></Table.Tr>
                  <Table.Tr><Table.Td c="dimmed">Rate limits</Table.Td><Table.Td ta="right">{Object.entries(t.rate_limits).filter(([, v]) => v).map(([k, v]) => `${k.replace("_", " ")} ${num(v as number)}`).join(", ") || "–"}</Table.Td></Table.Tr>
                  <Table.Tr><Table.Td c="dimmed">Baseline model</Table.Td><Table.Td ta="right">{t.baseline_model ?? "–"}</Table.Td></Table.Tr>
                  <Table.Tr><Table.Td c="dimmed">Models</Table.Td><Table.Td ta="right">{t.allowed_models.join(", ")}</Table.Td></Table.Tr>
                  <Table.Tr><Table.Td c="dimmed">Backends</Table.Td><Table.Td ta="right">{t.allowed_backends.join(", ")}</Table.Td></Table.Tr>
                  <Table.Tr><Table.Td c="dimmed">Compression: rules · model</Table.Td><Table.Td ta="right">{t.compression ? "on" : "off"} · {t.compression_model ? "on" : "off"}</Table.Td></Table.Tr>
                  <Table.Tr><Table.Td c="dimmed">Response cache</Table.Td><Table.Td ta="right">{t.response_cache ? "on" : "off"}</Table.Td></Table.Tr>
                  <Table.Tr><Table.Td c="dimmed">Stored responses</Table.Td><Table.Td ta="right">
                    {t.stored_responses?.enabled ? `on, kept ${t.stored_responses.retention_days} days after the last turn` : "off"}</Table.Td></Table.Tr>
                  {(t.stored_responses?.responses > 0 || t.stored_responses?.conversations > 0) && (
                    <Table.Tr><Table.Td c="dimmed">Content held</Table.Td><Table.Td ta="right">{storedLine(t.stored_responses)}</Table.Td></Table.Tr>
                  )}
                  <Table.Tr><Table.Td c="dimmed">Judged labels</Table.Td><Table.Td ta="right">{t.judged_labels?.enabled
                    ? `on: ${pct(t.judged_labels.sample_rate)} sampled, cap ${t.judged_labels.max_usd_month != null ? usd(t.judged_labels.max_usd_month) : "–"} a month`
                    : "off"}</Table.Td></Table.Tr>
                </Table.Tbody>
              </Table>
            </Paper>
          );
        })}
      </SimpleGrid>
    </>
  );
}

// ------------------------------------------------------------------ Keys

export function Keys({ api, me }: { api: Api; me: Me }) {
  const [tick, setTick] = useState(0);
  const res = useData(() => api.keys(), [api, tick]);
  const [user, setUser] = useState("");
  const [team, setTeam] = useState("");
  const [role, setRole] = useState<string | null>("user");
  const [name, setName] = useState("");
  const [days, setDays] = useState<number | string>(90);
  const [created, setCreated] = useState<NewKeyResult | null>(null);
  const [busy, setBusy] = useState(false);
  const roles = me.role === "admin" ? ["user", "team_lead", "finance", "admin"] : ["user"];
  const issue = async () => {
    setBusy(true);
    try {
      const r = await api.createKey({ user: user.trim(), role: role ?? "user", name: name || undefined,
        team: me.role === "admin" && team ? team : undefined, expires_days: Number(days) || undefined });
      setCreated(r);
      setUser("");
      setName("");
      setTick((n) => n + 1);
    } catch (e) {
      notifications.show({ color: "red", title: "Couldn't issue the key", message: errorText(e) });
    } finally {
      setBusy(false);
    }
  };
  const revoke = async (k: ApiKey) => {
    try {
      await api.revokeKey(k.id);
      notifications.show({ color: "green", title: "Key revoked", message: `${k.id} (${k.user_id}) stops working within a minute on every gateway.` });
      setTick((n) => n + 1);
    } catch (e) {
      notifications.show({ color: "red", title: "Couldn't revoke", message: errorText(e) });
    }
  };
  return (
    <>
      <PageTitle title="API keys" />
      <Stack gap="md">
        <Panel title="Issue a key">
          <Group align="flex-end" gap="sm">
            <TextInput label="User" placeholder="name@company.com" value={user} onChange={(e) => setUser(e.currentTarget.value)} w={240} />
            {me.role === "admin" && <TextInput label="Team" placeholder={me.team} value={team} onChange={(e) => setTeam(e.currentTarget.value)} w={150} />}
            <Select label="Role" data={roles.map((r) => ({ value: r, label: r.replace("_", " ") }))} value={role} onChange={setRole} allowDeselect={false} w={140} />
            <TextInput label="Label" placeholder="laptop, CI…" value={name} onChange={(e) => setName(e.currentTarget.value)} w={150} />
            <NumberInput label="Expires in days" value={days} onChange={setDays} min={1} max={3650} w={130} />
            <Button onClick={issue} loading={busy} disabled={!user.trim()}>Issue key</Button>
          </Group>
          {created && (
            <Alert mt="md" color="green" icon={<Icon name="ok" size={16} />} title={`Key for ${created.user} (${created.team}, ${created.role})`}>
              <Text size="sm" mb="xs">Copy it now; it isn't shown again. They run <Code>meritorder use-key &lt;key&gt;</Code>, then <Code>meritorder setup pi</Code>, <Code>claude-code</Code> or <Code>codex</Code>.</Text>
              <Group gap="xs"><Code fz="sm">{created.api_key}</Code>
                <CopyButton value={created.api_key}>{({ copied, copy }) => <Button size="xs" variant="default" onClick={copy}>{copied ? "Copied" : "Copy"}</Button>}</CopyButton></Group>
            </Alert>
          )}
        </Panel>
        <Panel title={me.role === "admin" ? "All keys" : `Keys for ${me.team}`}>
          <Loading loading={res.loading} error={res.error} />
          <Table.ScrollContainer minWidth={860}>
            <Table>
              <Table.Thead><Table.Tr><Table.Th>Key</Table.Th><Table.Th>User</Table.Th><Table.Th>Team</Table.Th><Table.Th>Role</Table.Th><Table.Th>Label</Table.Th>
                <Table.Th>Status</Table.Th><Table.Th>Expires</Table.Th><Table.Th>Last used</Table.Th><Table.Th /></Table.Tr></Table.Thead>
              <Table.Tbody>
                {(res.data?.keys ?? []).slice().reverse().map((k) => (
                  <Table.Tr key={k.id}>
                    <Table.Td className="mono">{k.id}</Table.Td><Table.Td>{k.user_id}</Table.Td><Table.Td>{k.team}</Table.Td><Table.Td>{k.role.replace("_", " ")}
                      {k.user_role && k.user_role !== k.role && k.acts_as === k.role && (
                        <Text size="xs" c="dimmed">{k.user_id} is {k.user_role.replace("_", " ")}; this key acts as {k.role.replace("_", " ")}</Text>)}</Table.Td>
                    <Table.Td>{k.name ?? ""}</Table.Td>
                    <Table.Td>{k.status === "active" ? <StatusBadge kind="good" size="sm">{k.status}</StatusBadge> : <StatusBadge kind="off" size="sm">{k.status}</StatusBadge>}</Table.Td>
                    <Table.Td>{k.expires_at ? new Date(k.expires_at).toLocaleDateString() : "never"}</Table.Td>
                    <Table.Td>{k.last_used_at ? new Date(k.last_used_at).toLocaleString() : "–"}</Table.Td>
                    <Table.Td>{k.status === "active" && <Button size="compact-xs" color="red" variant="outline" onClick={() => revoke(k)}>Revoke</Button>}</Table.Td>
                  </Table.Tr>
                ))}
              </Table.Tbody>
            </Table>
          </Table.ScrollContainer>
        </Panel>
      </Stack>
    </>
  );
}

// ------------------------------------------------------------------ My usage & sign in

export function MyUsage({ me }: { me: Me }) {
  return (
    <>
      <PageTitle title="My usage" />
      <Stack gap="md">
        <SimpleGrid cols={{ base: 1, sm: 3 }}>
          <Kpi label={`Spent in ${me.month}`} value={usd(me.spend_usd)} sub={<>
            <div>{me.budget_usd != null ? `of ${usd(me.budget_usd)} budget` : "no personal budget"}</div>
            <SpendSplit api={me.api_usd ?? 0} gpu={me.gpu_usd ?? 0} />
          </>} />
          <Kpi label="Without Merit Order" value={usd(me.baseline_usd)} />
          <Kpi label="Saved" value={usd(me.savings_usd)} accent={me.savings_usd >= 0} sub={`${num(me.requests)} requests`} />
        </SimpleGrid>
        {me.budget_usd != null && (
          <Panel title="Monthly budget">
            <Meter spent={me.spend_usd} budget={me.budget_usd} />
            <Text size="xs" c="dimmed" mt={6}>{pct(me.spend_usd / me.budget_usd)} used. Requests are refused at 100%; ask an admin to raise it.</Text>
          </Panel>
        )}
        <Panel title="Models this month">
          <HBars items={me.by_model.map((r) => ({ label: String(r.key), value: r.actual_cost, note: `${num(r.requests)}` }))} />
        </Panel>
      </Stack>
    </>
  );
}

