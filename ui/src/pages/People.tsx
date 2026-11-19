import { Alert, Badge, Button, Code, CopyButton, Group, Modal, NumberInput, Select, Stack, Switch, Table, Text, TextInput, Tooltip } from "@mantine/core";
import { notifications } from "@mantine/notifications";
import { useEffect, useMemo, useState } from "react";
import type { Api, GoogleReading, Idp, Invite, MappedFrom, Me, Person, PersonChange } from "../api";
import { usd } from "../api";
import { Loading, PageTitle, Panel, Status, StatusBadge } from "../components/common";
import { useData } from "../hooks";
import { ROLES } from "./config/fields";

/** A person's own monthly limit. Blank means the default applies: their team's per-person limit (or their static key's). */
function MonthlyLimit({ u, admin, onSave }: { u: Person; admin: boolean; onSave: (v: number | null) => void }) {
  const fallback = u.default_budget_usd_month == null ? "no limit" : `default ${usd(u.default_budget_usd_month)}`;
  const [value, setValue] = useState<string | number>(u.budget_usd_month ?? "");
  useEffect(() => setValue(u.budget_usd_month ?? ""), [u.budget_usd_month]);
  if (!admin) {
    return u.budget_usd_month != null ? <Text size="sm" className="num">{usd(u.budget_usd_month)}</Text> : <Text size="sm" c="dimmed">{fallback}</Text>;
  }
  const commit = () => {
    const next = value === "" || value == null ? null : Number(value);
    if (next !== (u.budget_usd_month ?? null)) onSave(next);
  };
  return (
    <NumberInput size="xs" w={140} prefix="$" min={0} allowNegative={false} decimalScale={2} thousandSeparator="," placeholder={fallback}
      value={value} onChange={setValue} onBlur={commit} onKeyDown={(e) => { if (e.key === "Enter") e.currentTarget.blur(); }}
      aria-label={`Monthly limit for ${u.id}`} title="Blank: the default applies. $0 stops their requests." />
  );
}

const IDP_NAME: Record<Idp, string> = { google: "Google", entra: "Entra", oidc: "your identity provider" };
/** Where sign-in takes team and role from. Google tokens carry no groups: the gateway reads them itself when Google Groups
 * is on (Config → Access), and otherwise maps by email address and domain. */
const mappedBy = (idp: Idp, googleGroups: boolean) =>
  idp === "google" ? (googleGroups ? "Google groups, email addresses and domains" : "email addresses and domains")
    : idp === "entra" ? "Entra groups" : "groups";
const managedBadge = (idp: Idp, googleGroups: boolean) =>
  idp === "google" ? (googleGroups ? "Google sign-in" : "email mappings") : idp === "entra" ? "Entra groups" : "IdP groups";

const when = (iso: string | null) => (iso ? new Date(iso).toLocaleString() : "never");

/** Which mapping gave a team or role, in words. */
function source(from: MappedFrom | null): string {
  if (!from || from.by === "default") return "no mapping (the default)";
  return from.by === "group" ? `group ${from.key}` : from.by === "address" ? "their address" : `domain ${from.key}`;
}

/** A person's Google groups: the mapped ones as badges (the one that set their team filled), the rest on hover, which
 * mapping set their team and role, and whether Google answered last time. */
function GoogleGroupsCell({ g, pinned, admin, signedIn }: { g: GoogleReading | null | undefined; pinned: boolean; admin: boolean; signedIn: boolean }) {
  // People given a key on the Keys page have no address from Google, so there's nothing to read for them.
  if (!g) return <Text size="xs" c="dimmed">{signedIn ? "Not read yet: read when they next use the gateway" : "–"}</Text>;
  const mapped = g.groups.filter((x) => x.mapped);
  const others = g.groups.filter((x) => !x.mapped);
  const more = others.length + g.hidden;
  const how = (x: { direct: boolean }) => (x.direct ? "member" : "member through a nested group");
  return (
    <Stack gap={4}>
      <Group gap={4}>
        {mapped.map((x) => (
          <Badge key={x.email} size="sm" color="mo" variant={g.team_from.by === "group" && g.team_from.key === x.email ? "filled" : "light"}
            title={`${x.name}: ${how(x)}${g.team_from.key === x.email ? "; sets their team" : ""}`} style={{ textTransform: "none", maxWidth: "100%" }}
            rightSection={x.direct ? undefined : <Text span fz={10} c="dimmed">nested</Text>}>
            {x.email}
          </Badge>
        ))}
        {more > 0 && (
          <Tooltip multiline w={320} withArrow label={others.length
            ? others.map((x) => `${x.email}${x.direct ? "" : " (nested)"}`).join(", ") + (g.hidden ? `, and ${g.hidden} more` : "")
            : `${g.hidden} more that no mapping uses${admin ? "" : " (admins see them)"}`}>
            <Text size="xs" c="dimmed" style={{ cursor: "help" }}>{mapped.length ? `+${more} more` : `${more} group${more === 1 ? "" : "s"}, none mapped`}</Text>
          </Tooltip>
        )}
        {!g.groups.length && !g.hidden && g.using !== "none" && <Text size="xs" c="dimmed">In no groups</Text>}
      </Group>
      <Text size="xs" c="dimmed">
        {pinned ? `Pinned here; the mappings would give ${g.team}, from ${source(g.team_from)}` : `Team from ${source(g.team_from)}`}
        {g.role_from ? `; ${g.role.replace("_", " ")} from ${source(g.role_from)}` : ""}
      </Text>
      {g.using === "last_known" && <Status kind="warn" title={g.error ?? undefined} style={{ whiteSpace: "normal", fontSize: 12 }}>Google didn't answer at {when(g.checked_at)}: using the groups read at {when(g.fetched_at)}</Status>}
      {g.using === "none" && <Status kind="bad" style={{ whiteSpace: "normal", fontSize: 12 }}>No groups known: the email and domain rules apply</Status>}
      {g.using !== "live" && g.error && <Text size="xs" c="dimmed" lineClamp={3} title={g.error}>{g.error}</Text>}
    </Stack>
  );
}

const linkFor = (inv: Invite) => `${location.origin}${inv.path}`;

/** A link just made: shown once, to copy and send. */
function NewLink({ invite, onClose }: { invite: Invite | null; onClose: () => void }) {
  if (!invite) return null;
  const reset = invite.kind === "reset";
  return (
    <Modal opened onClose={onClose} title={reset ? `Password reset link for ${invite.user}` : `Invite link for ${invite.user}`} size="lg">
      <Stack gap="sm">
        <Text size="sm">Send it to them yourself (chat or email). It works once, until {new Date(invite.expires_at).toLocaleString()}, and isn't
          shown again. {reset ? "Using it sets a new password and signs them out everywhere." : `They choose a password and join ${invite.team} as ${invite.role?.replace("_", " ")}.`}</Text>
        <Group gap="xs" wrap="nowrap"><Code fz="sm" style={{ overflowWrap: "anywhere" }}>{linkFor(invite)}</Code>
          <CopyButton value={linkFor(invite)}>{({ copied, copy }) => <Button size="compact-sm" variant="default" onClick={copy}>{copied ? "Copied" : "Copy"}</Button>}</CopyButton></Group>
        <Group justify="flex-end"><Button onClick={onClose}>Done</Button></Group>
      </Stack>
    </Modal>
  );
}

/** Invite someone who'll sign in with a password (local accounts), and the links still open. */
function Invites({ api, me, teams, made, onMade }: { api: Api; me: Me; teams: string[]; made: number; onMade: (i: Invite) => void }) {
  const admin = me.role === "admin";
  const res = useData(() => api.invites(), [api, made]);
  const [email, setEmail] = useState("");
  const [team, setTeam] = useState<string | null>(me.team);
  const [role, setRole] = useState<string | null>("user");
  const [busy, setBusy] = useState(false);
  const create = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    try {
      onMade(await api.createInvite({ email: email.trim(), team: admin ? team ?? undefined : undefined, role: admin ? role ?? "user" : "user" }));
      setEmail("");
    } catch (err) {
      notifications.show({ color: "red", title: "No link made", message: err instanceof Error ? err.message : String(err) });
    } finally {
      setBusy(false);
    }
  };
  const withdraw = async (i: Invite) => {
    try {
      await api.withdrawInvite(i.id);
      onMade({ ...i, token: undefined });
    } catch (err) {
      notifications.show({ color: "red", title: "Not withdrawn", message: err instanceof Error ? err.message : String(err) });
    }
  };
  const open = res.data?.invites ?? [];
  return (
    <Panel title="Invite someone">
      <Text size="sm" c="dimmed" mb="sm">For people who sign in with an email address and a password Merit Order keeps. You get a one-time link to send them.
        {admin ? "" : " Team leads invite people to their own team with the user role."}</Text>
      <form onSubmit={create}>
        <Group gap="sm" align="flex-end">
          <TextInput size="xs" label="Email address" type="email" value={email} onChange={(e) => setEmail(e.currentTarget.value)} w={260} required />
          {admin && <Select size="xs" label="Team" data={[...new Set([...teams, me.team])]} value={team} onChange={setTeam} allowDeselect={false} w={150} />}
          {admin && <Select size="xs" label="Role" data={ROLES.map((r) => ({ value: r, label: r.replace("_", " ") }))} value={role} onChange={setRole} allowDeselect={false} w={130} />}
          <Button size="xs" type="submit" loading={busy} disabled={!email.trim()}>Make invite link</Button>
        </Group>
      </form>
      {open.length > 0 && (
        <Table mt="md">
          <Table.Thead><Table.Tr><Table.Th>Open link for</Table.Th><Table.Th>Kind</Table.Th><Table.Th>Team</Table.Th><Table.Th>Made by</Table.Th><Table.Th>Expires</Table.Th><Table.Th /></Table.Tr></Table.Thead>
          <Table.Tbody>
            {open.map((i) => (
              <Table.Tr key={i.id}>
                <Table.Td>{i.user}</Table.Td><Table.Td>{i.kind === "reset" ? "password reset" : "invite"}</Table.Td>
                <Table.Td>{i.team ?? "–"}</Table.Td><Table.Td>{i.created_by ?? "–"}</Table.Td>
                <Table.Td>{new Date(i.expires_at).toLocaleString()}</Table.Td>
                <Table.Td><Button size="compact-xs" variant="subtle" color="red" onClick={() => void withdraw(i)}>Withdraw</Button></Table.Td>
              </Table.Tr>
            ))}
          </Table.Tbody>
        </Table>
      )}
    </Panel>
  );
}

export function People({ api, me }: { api: Api; me: Me }) {
  const [tick, setTick] = useState(0);
  const res = useData(() => api.users(), [api, tick]);
  const [filter, setFilter] = useState("");
  const [team, setTeam] = useState<string | null>(null);
  const [made, setMade] = useState<Invite | null>(null);
  const [links, setLinks] = useState(0);  // bumped when a link is made or withdrawn
  const madeLink = (i: Invite) => { if (i.token) setMade(i); setLinks((n) => n + 1); };
  const resetLink = async (u: Person) => {
    try {
      madeLink(await api.resetLink(u.id));
    } catch (e) {
      notifications.show({ color: "red", title: "No link made", message: e instanceof Error ? e.message : String(e) });
    }
  };
  const admin = me.role === "admin";
  const idp: Idp = res.data?.idp ?? "oidc";
  const googleGroups = !!res.data?.google_groups;
  const every = res.data?.google_groups_refresh_seconds ?? 60;
  const rows = useMemo(() => (res.data?.users ?? []).filter((u) =>
    (!team || u.team === team) && `${u.id} ${u.email ?? ""} ${u.display_name ?? ""}`.toLowerCase().includes(filter.toLowerCase())), [res.data, filter, team]);
  const patch = async (u: Person, change: PersonChange, message: string) => {
    try {
      const out = await api.updateUser(u.id, change);
      const gone = out.stored_responses_deleted ?? 0;
      notifications.show({ color: "green", title: u.email ?? u.id,
        message: gone ? `${message} ${gone} stored response${gone === 1 ? "" : "s"} deleted.` : message });
      setTick((n) => n + 1);
    } catch (e) {
      notifications.show({ color: "red", title: "Not changed", message: e instanceof Error ? e.message : String(e) });
    }
  };
  return (
    <>
      <PageTitle title="People">
        <Group gap="sm">
          <TextInput size="xs" placeholder="Search people" value={filter} onChange={(e) => setFilter(e.currentTarget.value)} aria-label="Search people" />
          <Select size="xs" placeholder="All teams" data={res.data?.teams ?? []} value={team} onChange={setTeam} clearable w={160} aria-label="Team filter" />
        </Group>
      </PageTitle>
      <Alert color="gray" variant="light" mb="md">
        {res.data?.oidc
          ? `People appear here when they first sign in with ${IDP_NAME[idp]} (meritorder login) or when someone issues them a key. Their team and role follow the ${mappedBy(idp, googleGroups)} mapped in Config → Access unless you set them here, which pins them.`
          : "People appear here when you invite them (below) or issue them a key on the Keys page. Turn on single sign-on in Config → Access to create them automatically from Entra or Google."}
        {googleGroups && ` Their Google groups are read at sign-in and again every ${every} seconds while they use the gateway, and a change reaches every gateway within a minute after that. If Google can't be reached, their last known groups stand for a while (Config → Access), then the email and domain rules apply.`}
        {" "}Each person can have their own monthly limit; blank means their team's per-person default applies. Team and organization budgets still apply on top.
        {" "}Changes apply within a minute on every gateway, including keys already issued. Disabling someone, here or through SCIM, also deletes
        the responses and conversations they kept in stored mode (Config → Teams); re-enabling them doesn't bring those back.
        {res.data?.scim ? " Entra provisioning (SCIM) is on: people and group changes arrive as they happen, and anyone deactivated in Entra is disabled here at once."
          : idp === "google" ? " Google doesn't provision people over SCIM: disable leavers here. Their keys from sign-in also expire on their own (Config → Access)."
          : " Turn on Entra provisioning (SCIM) to disable people the moment they leave: see Config → Access."}
      </Alert>
      <Invites api={api} me={me} teams={res.data?.teams ?? []} made={links} onMade={madeLink} />
      <NewLink invite={made} onClose={() => setMade(null)} />
      <Panel title={`${rows.length} people`}>
        <Loading loading={res.loading} error={res.error} />
        <Table.ScrollContainer minWidth={googleGroups ? 1400 : 1120}>
          <Table>
            <Table.Thead><Table.Tr><Table.Th>Person</Table.Th><Table.Th>Team</Table.Th><Table.Th>Role</Table.Th><Table.Th>Managed by</Table.Th>
              {googleGroups && <Table.Th>Google groups</Table.Th>}
              <Table.Th ta="right">Keys</Table.Th><Table.Th ta="right">Spend this month</Table.Th><Table.Th>Monthly limit</Table.Th><Table.Th>Last seen</Table.Th><Table.Th>Access</Table.Th></Table.Tr></Table.Thead>
            <Table.Tbody>
              {rows.map((u) => (
                <Table.Tr key={u.id} style={{ opacity: u.disabled ? 0.55 : 1 }}>
                  <Table.Td><Text size="sm" fw={600}>{u.display_name ?? u.email ?? u.id}</Text>{(u.display_name || u.email) && <Text size="xs" c="dimmed">{u.email ?? u.id}</Text>}</Table.Td>
                  <Table.Td>
                    {admin ? <Select size="xs" data={[...new Set([...(res.data?.teams ?? []), u.team])]} value={u.team} allowDeselect={false} w={150} aria-label={`Team for ${u.id}`}
                      onChange={(v) => v && v !== u.team && patch(u, { team: v }, `Moved to ${v}.`)} /> : u.team}
                    {res.data && !res.data.teams.includes(u.team) && <Status kind="warn" title="Add it under Config → Teams to give it its own budget and policy" style={{ whiteSpace: "normal", fontSize: 12, display: "flex", marginTop: 4 }}>“{u.team}” isn't defined: default settings apply</Status>}
                  </Table.Td>
                  <Table.Td>{admin ? <Select size="xs" data={ROLES.map((r) => ({ value: r, label: r.replace("_", " ") }))} value={u.role} allowDeselect={false} w={130}
                    aria-label={`Role for ${u.id}`} onChange={(v) => v && v !== u.role && patch(u, { role: v }, `Role set to ${v.replace("_", " ")}.`)} /> : u.role.replace("_", " ")}</Table.Td>
                  <Table.Td>{u.managed === "local"
                    ? <Group gap={6}><Badge color="mo">set here</Badge>{admin && res.data?.oidc && <Button size="compact-xs" variant="subtle" onClick={() => patch(u, { follow_idp: true },
                        googleGroups && u.google ? `Team and role follow the ${mappedBy(idp, googleGroups)} again, from their groups read just now.`
                          : `Team and role will follow the ${mappedBy(idp, googleGroups)} at next sign-in.`)}>Follow sign-in</Button>}</Group>
                    : <Badge variant="default">{res.data?.oidc ? managedBadge(idp, googleGroups) : "key issuer"}</Badge>}</Table.Td>
                  {googleGroups && <Table.Td maw={360}><GoogleGroupsCell g={u.google} pinned={u.managed === "local"} admin={admin} signedIn={!!u.email} /></Table.Td>}
                  <Table.Td className="num">{u.active_keys}</Table.Td>
                  <Table.Td className="num">{usd(u.spend_month)}</Table.Td>
                  <Table.Td><MonthlyLimit u={u} admin={admin} onSave={(v) => patch(u, { budget_usd_month: v },
                    v == null ? "Own limit removed: the default applies." : v === 0 ? "Limit set to $0: their requests are refused." : `Monthly limit set to ${usd(v)}.`)} /></Table.Td>
                  <Table.Td>{u.last_seen ? new Date(u.last_seen).toLocaleString() : "–"}</Table.Td>
                  <Table.Td><Stack gap={4} align="flex-start">{admin ? <Switch checked={!u.disabled} label={u.disabled ? "disabled" : "active"} disabled={u.id === me.user}
                    onChange={(e) => patch(u, { disabled: !e.currentTarget.checked }, e.currentTarget.checked ? "Access restored."
                      : "Disabled: all their keys and console sessions stop working, and what they kept in stored responses is deleted.")} />
                    : <StatusBadge kind={u.disabled ? "off" : "good"}>{u.disabled ? "disabled" : "active"}</StatusBadge>}
                    {!u.disabled && u.id !== me.user && (admin || (u.team === me.team && u.role === "user")) &&
                      <Button size="compact-xs" variant="subtle" onClick={() => void resetLink(u)}
                        title="A one-time link that sets their password (a new one, if they have one)">Password link</Button>}</Stack></Table.Td>
                </Table.Tr>
              ))}
            </Table.Tbody>
          </Table>
        </Table.ScrollContainer>
        {!res.loading && !rows.length && <Text size="sm" c="dimmed" mt="sm">No people yet. Invite someone above, issue a key on the Keys page, or turn on single sign-on.</Text>}
      </Panel>
    </>
  );
}
