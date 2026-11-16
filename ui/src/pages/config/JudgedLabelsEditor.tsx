import { Icon } from "../../components/icons";

import { Alert, Badge, Button, Checkbox, Group, Modal, MultiSelect, NumberInput, Paper, Select, SimpleGrid, Stack, Table, Text } from "@mantine/core";
import { useEffect, useState } from "react";
import type { Api, Doc, JudgedEstimate } from "../../api";
import { pct, usd } from "../../api";
import { SectionTitle, lbl, setIn } from "./fields";
import { ROUTING as H, SECTION } from "./help";

/** The evals' judges, one per family at the same tier (config.py JudgedLabelsConfig.judges). */
const DEFAULT_JUDGES = ["claude-opus", "gpt-sol", "gemini-carbon"];
const LEVELS = [1, 2, 3, 4, 5].map((l) => ({ value: String(l), label: `level ${l}` }));
const errorText = (e: unknown) => (e instanceof Error ? e.message : String(e));

function stronger(s: Doc): string {
  return (s.stronger ?? "auto") === "auto" ? `auto at level ${s.stronger_level ?? 5}` : String(s.stronger);
}

/** Turning on (or changing) one team's judged labels: the settings, the expected monthly cost from its recorded traffic,
 * what it sends where, and an explicit "I understand" above $1 a month. Nothing changes until the configuration is saved. */
function JudgedDialog({ api, team, current, modelNames, learnedOff, onClose, onApply }: {
  api: Api; team: string | null; current: Doc; modelNames: string[]; learnedOff: boolean; onClose: () => void; onApply: (s: Doc) => void;
}) {
  const [rate, setRate] = useState<number>(0.01);
  const [strong, setStrong] = useState("auto");
  const [level, setLevel] = useState(5);
  const [judges, setJudges] = useState<string[]>([]);
  const [cap, setCap] = useState<number | null>(null);
  const [est, setEst] = useState<JudgedEstimate | null>(null);
  const [estError, setEstError] = useState<string | null>(null);
  const [agreed, setAgreed] = useState(false);
  useEffect(() => {
    if (!team) return;
    setRate(Number(current.sample_rate ?? 0.01));
    setStrong(String(current.stronger ?? "auto"));
    setLevel(Number(current.stronger_level ?? 5));
    setJudges((current.judges as string[] | undefined) ?? DEFAULT_JUDGES.filter((j) => modelNames.includes(j)));
    setCap(current.max_usd_month != null ? Number(current.max_usd_month) : null); // no default: whoever turns it on sets one
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [team]);
  const settings: Doc = { sample_rate: rate, stronger: strong, stronger_level: level, judges, ...(cap ? { max_usd_month: cap } : {}) };
  const key = JSON.stringify(settings);
  useEffect(() => {
    setEst(null);
    setEstError(null);
    setAgreed(false); // a new estimate needs a new "I understand"
    if (!team || !(rate > 0 && rate <= 1)) return;
    const t = setTimeout(async () => {
      try { setEst(await api.judgedEstimate(team, settings)); } catch (e) { setEstError(errorText(e)); }
    }, 300);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, team]);
  const confirm = !!est && est.usd_month >= est.confirm_above_usd;
  const ready = !!est && !!cap && cap > 0 && judges.length > 0 && rate > 0 && rate <= 1 && (!confirm || agreed);
  const sentTo = est ? [...est.sent_to.stronger, ...est.sent_to.judges] : [];
  return (
    <Modal opened={!!team} onClose={onClose} size="lg" title={team ? `Judged labels for ${team}` : ""}>
      <Stack gap="sm">
        <Alert color="red" icon={<Icon name="bad" size={16} />} title="Sampled prompts and answers are sent to two more models">
          <Text size="sm">A sample of this team's requests, prompt and answer, goes to the stronger model and to a judge
            {sentTo.length ? ` (${[...new Set(sentTo)].join(", ")})` : ""}: models its requests' own routing didn't choose. Only the
            judge's verdict is kept, next to the request's hashed features.</Text>
        </Alert>
        {learnedOff && <Alert color="yellow" icon={<Icon name="warn" size={16} />}>The learned router is off, so nothing is sampled until it's set to Shadow or Route
          (Learned router, above). The estimate reads its records, so it needs a few days of them.</Alert>}
        <SimpleGrid cols={{ base: 1, sm: 2 }}>
          <NumberInput {...lbl("Sample rate (0–1)", H.judged_rate)} value={rate} min={0.0001} max={1} step={0.005} decimalScale={4}
            description={`${pct(rate)} of eligible task starts`} onChange={(v) => setRate(Number(v) || 0)} />
          <NumberInput {...lbl("Monthly spending cap", H.judged_cap)} prefix="$" min={0.01} decimalScale={2} value={cap ?? ""} placeholder="required"
            description="Required: there's no default." thousandSeparator="," onChange={(v) => setCap(v === "" || v == null ? null : Number(v) || null)} />
          <Select {...lbl("Stronger model", H.judged_stronger)} allowDeselect={false} searchable value={strong}
            data={[{ value: "auto", label: "auto at a level" }, ...modelNames.map((m) => ({ value: m, label: m }))]} onChange={(v) => setStrong(v ?? "auto")} />
          {strong === "auto" && <Select {...lbl("…at", H.judged_level)} data={LEVELS} allowDeselect={false} value={String(level)}
            onChange={(v) => setLevel(Number(v ?? 5))} />}
        </SimpleGrid>
        <MultiSelect {...lbl("Judges", H.judged_judges)} data={modelNames} value={judges} onChange={setJudges} searchable
          description="Each sample is graded by the first the team may use whose family is neither answer's." error={judges.length ? undefined : "Pick at least one."} />
        {estError && <Alert color="red" icon={<Icon name="bad" size={16} />}>{estError}</Alert>}
        {est && (
          <Alert color={est.usd_month >= 50 ? "red" : confirm ? "yellow" : "mo"} icon={<Icon name={est.usd_month >= 50 || confirm ? "warn" : "info"} size={16} />}
            title={`This costs real money: about ${usd(est.usd_month)} a month`}>
            <Stack gap={2}>
              {est.warning.map((w) => <Text key={w} size="xs">{w}</Text>)}
              <Text size="xs" fw={600}>A rough guide: real costs have run up to three times the estimate. A stronger model can write far
                more than the answer it replays, and an agent's conversation is read in full by the replay and the judge.</Text>
            </Stack>
          </Alert>
        )}
        {confirm && (
          <Checkbox checked={agreed} onChange={(e) => setAgreed(e.currentTarget.checked)}
            label={`I understand this costs about ${usd(est!.usd_month)} a month, maybe several times that, stops at ${usd(cap ?? 0)} a month, `
              + "and sends sampled prompts and answers to these models."} />
        )}
        <Group justify="flex-end">
          <Button variant="default" onClick={onClose}>Cancel</Button>
          <Button disabled={!ready} onClick={() => onApply({ ...current, enabled: true, sample_rate: rate, stronger: strong, stronger_level: level, judges,
            max_usd_month: cap })}>{est ? `Turn on (about ${usd(est.usd_month)} a month)` : "Turn on"}</Button>
        </Group>
        <Text size="xs" c="dimmed">Nothing changes until you save the configuration.</Text>
      </Stack>
    </Modal>
  );
}

/** Config → Routing → Judged labels: which teams have them, with what, and the dialog that turns them on. */
export function JudgedLabelsEditor({ api, draft, setDraft, modelNames }: {
  api: Api; draft: Doc; setDraft: (d: Doc) => void; modelNames: string[];
}) {
  const teams: Doc = draft.teams ?? {};
  const names = Object.keys(teams).length ? Object.keys(teams) : ["default"];
  const [editing, setEditing] = useState<string | null>(null);
  const learnedOff = (draft.routing?.learned?.mode ?? "off") === "off";
  const settings = (team: string): Doc => teams[team]?.judged_labels ?? {};
  const write = (team: string, value: Doc | undefined) => setDraft(setIn(draft, ["teams", team, "judged_labels"], value));
  const turnOff = (team: string) => {
    const rest: Doc = { ...settings(team) };
    delete rest.enabled;
    write(team, Object.keys(rest).length ? rest : undefined);
  };
  return (
    <Paper>
      <SectionTitle title="Judged labels" intro={SECTION.judged} />
      <Text size="xs" fw={600} c="var(--mo-bad)" mt={4}>A team that turns this on sends a sample of its prompts and answers to the stronger model and a judge.</Text>
      {learnedOff && <Text size="xs" c="dimmed">The learned router is off, so no team's requests are sampled until it's set to Shadow or Route.</Text>}
      <Table.ScrollContainer minWidth={720} mt="xs">
        <Table verticalSpacing={5} fz="sm">
          <Table.Thead><Table.Tr><Table.Th>Team</Table.Th><Table.Th>Status</Table.Th><Table.Th>Stronger model</Table.Th>
            <Table.Th>Judges</Table.Th><Table.Th ta="right">Cap / month</Table.Th><Table.Th /></Table.Tr></Table.Thead>
          <Table.Tbody>
            {names.map((n) => {
              const s = settings(n);
              return (
                <Table.Tr key={n}>
                  <Table.Td fw={600}>{n}</Table.Td>
                  <Table.Td>{s.enabled ? <Badge color="mo">on, {pct(Number(s.sample_rate ?? 0.01))} sampled</Badge>
                    : <Badge color="gray" variant="light">off</Badge>}</Table.Td>
                  <Table.Td className="mono">{s.enabled ? stronger(s) : "–"}</Table.Td>
                  <Table.Td><Text size="xs" c="dimmed" lineClamp={1}>{s.enabled ? ((s.judges as string[] | undefined) ?? DEFAULT_JUDGES).join(", ") : "–"}</Text></Table.Td>
                  <Table.Td className="num">{s.enabled && s.max_usd_month != null ? usd(Number(s.max_usd_month)) : "–"}</Table.Td>
                  <Table.Td ta="right">
                    <Group gap={6} justify="flex-end" wrap="nowrap">
                      <Button size="compact-xs" variant={s.enabled ? "default" : "light"} onClick={() => setEditing(n)}
                        aria-label={`${s.enabled ? "Change" : "Turn on"} judged labels for ${n}`}>{s.enabled ? "Change…" : "Turn on…"}</Button>
                      {s.enabled && <Button size="compact-xs" variant="outline" color="red" onClick={() => turnOff(n)}
                        aria-label={`Turn off judged labels for ${n}`}>Turn off</Button>}
                    </Group>
                  </Table.Td>
                </Table.Tr>
              );
            })}
          </Table.Tbody>
        </Table>
      </Table.ScrollContainer>
      <Text size="xs" c="dimmed" mt={4}>Teams that aren't listed use the default team's settings, judged labels included, each with a cap of its own.
        What they cost and how far their verdicts agree with what people did: Reports → Learned router.</Text>
      <JudgedDialog api={api} team={editing} current={editing ? settings(editing) : {}} modelNames={modelNames} learnedOff={learnedOff}
        onClose={() => setEditing(null)} onApply={(s) => { if (editing) write(editing, s); setEditing(null); }} />
    </Paper>
  );
}
