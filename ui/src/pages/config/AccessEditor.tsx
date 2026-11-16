import { ActionIcon, Alert, Button, Group, Paper, Select, SimpleGrid, Stack, Switch, Text, TextInput, Title } from "@mantine/core";
import type { Doc } from "../../api";
import { EnvName, Flag, KeyValue, Num, ROLES, Txt, lbl, setIn } from "./fields";
import { ACCESS as H } from "./help";

/** Google Groups (Cloud Identity API): whether the gateway reads them, as whom, and how often. */
function GoogleGroups({ gg, set, toggle }: { gg: Doc | undefined; set: (path: string[], v: unknown) => void; toggle: (on: boolean) => void }) {
  return (
    <Paper withBorder p="sm">
      <Group justify="space-between" mb="xs">
        <Title order={5} fz="sm">Google Groups (Google sign-in only)</Title>
        <Switch label="Read Google Groups" checked={!!gg} onChange={(e) => toggle(e.currentTarget.checked)} />
      </Group>
      <Text size="xs" c="dimmed" mb={gg ? "sm" : 0}>
        Google's tokens carry no groups, so the gateway asks Google's Cloud Identity API which groups each person is in, nested groups
        included: at sign-in, and again while they use the gateway, so a removal takes effect. Map groups by address in the group lists
        above. The gateway calls as a service account that has the Groups Reader admin role (Google Admin console → Account → Admin
        roles), with a read-only scope and no domain-wide delegation. Setup steps: docs/06-auth-and-spend.md.
      </Text>
      {gg && (
        <SimpleGrid cols={{ base: 1, sm: 2 }}>
          <Flag label="Enabled" value={gg.enabled} onChange={(v) => set(["enabled"], v)} help={H.gg_enabled} />
          <Switch mt={28} {...lbl("Security groups only", H.gg_security_groups_only)} checked={gg.security_groups_only === true}
            onChange={(e) => set(["security_groups_only"], e.currentTarget.checked || undefined)} />
          <EnvName label="Key in environment variable" value={gg.credentials_env} onChange={(v) => set(["credentials_env"], v)} help={H.gg_credentials_env}
            description="Holds the service account's JSON key. With no key at all, the gateway uses its workload identity." />
          <Txt label="Key file" value={gg.credentials_file} onChange={(v) => set(["credentials_file"], v)} mono help={H.gg_credentials_file}
            placeholder="/run/secrets/meritorder/MERITORDER_GOOGLE_GROUPS_CREDENTIALS" description="A mounted JSON key, re-read when it changes. Tried before the variable." />
          <Num label="Read groups again every (seconds)" value={gg.refresh_seconds} onChange={(v) => set(["refresh_seconds"], v)} min={1} placeholder="60"
            help={H.gg_refresh_seconds} />
          <Num label="Keep last known groups for (hours)" value={gg.keep_last_known_hours} onChange={(v) => set(["keep_last_known_hours"], v)} min={0.1}
            placeholder="24" help={H.gg_keep_last_known_hours} />
        </SimpleGrid>
      )}
    </Paper>
  );
}

export function AccessEditor({ draft, setDraft, teamNames }: { draft: Doc; setDraft: (d: Doc) => void; teamNames: string[] }) {
  const auth: Doc = draft.auth ?? {};
  const oidc: Doc | undefined = auth.oidc ?? undefined;
  const set = (path: (string | number)[], v: unknown) => setDraft(setIn(draft, path, v));
  const keys: Doc[] = auth.static_keys ?? [];
  return (
    <Stack gap="md" maw={980}>
      <Paper>
        <Group justify="space-between" mb="xs">
          <Title order={4} fz="sm">Single sign-on (Entra ID, Google, or any OIDC provider)</Title>
          <Switch label="Enabled" checked={!!oidc} onChange={(e) => set(["auth", "oidc"], e.currentTarget.checked
            ? { issuer: "${MERITORDER_OIDC_ISSUER}", client_id: "${MERITORDER_OIDC_CLIENT_ID}", public_client_secret: "${MERITORDER_OIDC_CLIENT_SECRET}",
                hosted_domains: "${MERITORDER_OIDC_HOSTED_DOMAINS}" } : undefined)} />
        </Group>
        <Text size="xs" c="dimmed" mb="sm">
          People run <code>meritorder login</code>, sign in with Entra or Google, and get a key. Their team and role come from the mappings below
          (Entra groups; for Google, Google groups if the gateway reads them, and email addresses and domains), re-read at each login, unless an
          admin pinned them on the People page.
        </Text>
        {oidc && (
          <Stack gap="md">
            <SimpleGrid cols={{ base: 1, sm: 2 }}>
              <Txt label="Issuer" value={oidc.issuer} onChange={(v) => set(["auth", "oidc", "issuer"], v)} mono help={H.issuer}
                description="Entra: https://login.microsoftonline.com/<tenant-id>/v2.0. Google: https://accounts.google.com (or ${MERITORDER_OIDC_ISSUER})" />
              <Txt label="Client ID" value={oidc.client_id} onChange={(v) => set(["auth", "oidc", "client_id"], v)} mono help={H.client_id}
                description="Entra: the app registration (public client, device-code flow allowed). Google: a TVs and Limited Input devices client." />
              <Txt label="Client secret (Google)" value={oidc.public_client_secret} onChange={(v) => set(["auth", "oidc", "public_client_secret"], v)} mono
                placeholder="${MERITORDER_OIDC_CLIENT_SECRET}" help={H.public_client_secret} description="Google's device clients send one; Entra's don't." />
              <Txt label="Google Workspace domains" value={Array.isArray(oidc.hosted_domains) ? oidc.hosted_domains.join(", ") : oidc.hosted_domains}
                onChange={(v) => set(["auth", "oidc", "hosted_domains"], v)} mono placeholder="${MERITORDER_OIDC_HOSTED_DOMAINS}" help={H.hosted_domains}
                description="Required with Google. Comma-separated, e.g. example.com, example.org" />
              <Select {...lbl("Team when no mapping matches", H.default_team)} data={teamNames} value={oidc.default_team ?? "default"}
                onChange={(v) => set(["auth", "oidc", "default_team"], v === "default" ? undefined : v)} allowDeselect={false} />
              <Num label="Keys from login expire after (days)" value={oidc.key_ttl_days} onChange={(v) => set(["auth", "oidc", "key_ttl_days"], v)} min={1} placeholder="30" help={H.key_ttl_days} />
              <Txt label="Groups claim" value={oidc.groups_claim} onChange={(v) => set(["auth", "oidc", "groups_claim"], v)} placeholder="groups" help={H.groups_claim} />
              <Txt label="Email claim" value={oidc.email_claim} onChange={(v) => set(["auth", "oidc", "email_claim"], v)} placeholder="preferred_username" help={H.email_claim} />
            </SimpleGrid>
            <KeyValue label="Groups → teams" keyLabel="Entra group object ID, or Google group address" valueLabel="Team" valueOptions={teamNames}
              description="The first group in this list the person is in wins. Add teams on the Teams tab." help={H.team_from_groups} value={oidc.team_from_groups}
              onChange={(v) => set(["auth", "oidc", "team_from_groups"], v)} />
            <KeyValue label="Groups → roles" keyLabel="Entra group object ID, or Google group address" valueLabel="Role" valueOptions={ROLES}
              description="The highest matching role wins; everyone else is a user." help={H.role_from_groups} value={oidc.role_from_groups}
              onChange={(v) => set(["auth", "oidc", "role_from_groups"], v)} />
            <KeyValue label="Email addresses or domains → teams (Google)" keyLabel="Address or @domain" valueLabel="Team" valueOptions={teamNames}
              description="Team: the exact address, then a group, then the domain, then the default team." help={H.team_from_emails}
              value={oidc.team_from_emails} onChange={(v) => set(["auth", "oidc", "team_from_emails"], v)} />
            <KeyValue label="Email addresses or domains → roles (Google)" keyLabel="Address or @domain" valueLabel="Role" valueOptions={ROLES}
              description="Only for addresses the token marks verified." help={H.role_from_emails} value={oidc.role_from_emails}
              onChange={(v) => set(["auth", "oidc", "role_from_emails"], v)} />
            <GoogleGroups gg={oidc.google_groups} set={(path, v) => set(["auth", "oidc", "google_groups", ...path], v)}
              toggle={(on) => set(["auth", "oidc", "google_groups"], on ? { credentials_env: "MERITORDER_GOOGLE_GROUPS_CREDENTIALS" } : undefined)} />
          </Stack>
        )}
      </Paper>
      <Paper>
        <Title order={4} fz="sm" mb="xs">Automatic provisioning (SCIM)</Title>
        <Text size="sm" mb="xs">Lets Entra push people and group changes as they happen, and disables anyone deactivated or removed in Entra immediately.</Text>
        <Text size="xs" c="dimmed">1. On the gateways, set <code>MERITORDER_SCIM_TOKEN</code> to a long random secret.</Text>
        <Text size="xs" c="dimmed">2. In Entra: Enterprise applications → your app → Provisioning → Automatic. Tenant URL: <code>{window.location.origin}/scim/v2</code>, Secret token: the same value.</Text>
        <Text size="xs" c="dimmed">3. Assign the groups to provision. Their membership maps to teams and roles with the group mappings above.</Text>
      </Paper>
      <Paper>
        <Group justify="space-between" mb="xs">
          <Title order={4} fz="sm">Static keys</Title>
          <Button size="xs" variant="default" onClick={() => set(["auth", "static_keys"], [...keys, { key_env: "", user: "", team: "default", role: "user" }])}>Add</Button>
        </Group>
        <Text size="xs" c="dimmed" mb="sm">For bootstrap and service accounts: the key lives in an environment variable on the gateways. Prefer keys issued on the Keys page, which are easier to rotate and revoke.</Text>
        {keys.length === 0 && <Text size="sm" c="dimmed">None.</Text>}
        <Stack gap="xs">
          {keys.map((k, i) => (
            <Group key={i} gap="xs" align="flex-end" wrap="nowrap">
              <div style={{ flex: 2 }}><EnvName label="Key environment variable" value={k.key_env} onChange={(v) => set(["auth", "static_keys", i, "key_env"], v)} description=" " help={H.key_env} /></div>
              <TextInput {...lbl("User", H.user)} value={k.user ?? ""} onChange={(e) => set(["auth", "static_keys", i, "user"], e.currentTarget.value)} style={{ flex: 2 }} />
              <Select {...lbl("Team", H.team)} data={teamNames} value={k.team ?? "default"} onChange={(v) => set(["auth", "static_keys", i, "team"], v)} style={{ flex: 1 }} allowDeselect={false} />
              <Select {...lbl("Role", H.role)} data={ROLES} value={k.role ?? "user"} onChange={(v) => set(["auth", "static_keys", i, "role"], v)} style={{ flex: 1 }} allowDeselect={false} />
              <div style={{ flex: 1 }}><Num label="Monthly limit" value={k.budget_usd_month} onChange={(v) => set(["auth", "static_keys", i, "budget_usd_month"], v)} min={0} prefix="$"
                placeholder="team's" help={H.budget_usd_month} /></div>
              <ActionIcon variant="subtle" color="red" mb={6} aria-label="Remove" onClick={() => set(["auth", "static_keys", i], undefined)}>×</ActionIcon>
            </Group>
          ))}
        </Stack>
        {keys.some((k) => k.role === "admin") && <Alert mt="sm" color="gray" variant="light">Keep at least one admin key, or you can lock yourself out of this page.</Alert>}
      </Paper>
    </Stack>
  );
}
