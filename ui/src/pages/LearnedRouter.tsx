import { Badge, Button, Code, Group, SegmentedControl, SimpleGrid, Stack, Table, Text, Tooltip } from "@mantine/core";
import { notifications } from "@mantine/notifications";
import { useState } from "react";
import type { Api, Me, RouterJudged, RouterReport, RouterStatus } from "../api";
import { num, pct, usd } from "../api";
import { AnsweredBy, Kpi, Loading, Panel, StatusBadge } from "../components/common";
import { useData } from "../hooks";
import { ExpectedCost } from "./ExpectedCost";
import { Person } from "../people";

const SIGNALS: Record<string, string> = {
  floor_raised: "raised X-Route-Floor", retried: "sent again", interrupted: "stopped", abandoned: "asked again elsewhere",
  corrected: "corrected in the next prompt", tests_failing: "tests failing at the end", build_broken: "build broken at the end",
  fix_loop: "fix loop", refused: "refused", error: "error",
};
const OUTCOMES: Record<string, string> = {
  agrees: "agreed with the rules", shadow: "disagreed (shadow: not used)", used: "used", unsure: "unsure: the rules decided",
  stale: "stale: the rules decided", missing: "no model yet", failed: "failed: the rules decided",
  named: "named a model (recorded for training)",
};
const JUDGED_OUTCOMES: Record<string, string> = {
  judged: "judged", pending: "waiting for its replay or judge", dropped: "dropped", skipped: "skipped: the answer's model was as strong",
  capped: "not taken: the cap was spent",
};
const money = (v: number | null | undefined) => (v == null ? "–" : usd(v));
const share = (v: number | null | undefined) => (v == null ? "–" : pct(v));
const signed = (v: number) => (v > 0 ? `+${usd(v)}` : usd(v));

/** Judged labels: how many, what they cost, how far they agree with the outcome signals, each team's cap, and the judge's checks. */
function Judged({ j, weightInUse }: { j: RouterJudged; weightInUse: number | null | undefined }) {
  const notGood = j.labels ? j.not_good_enough / j.labels : null;
  return (
    <Panel title="Judged labels" right={<Text size="xs" c="dimmed">Sampled prompts and answers went to a stronger model and a judge; only the verdicts are kept.</Text>}>
      <SimpleGrid cols={{ base: 1, xs: 2, md: 4 }}>
        <Kpi label="Judged labels" value={num(j.labels)} sub={`${num(j.samples)} samples taken`} />
        <Kpi label="Not good enough" value={share(notGood)} sub={`${num(j.not_good_enough)} answers, next to the stronger one`} />
        <Kpi label="Cost in this period" value={usd(j.cost.total)} sub={`replays ${usd(j.cost.replay)} · judging ${usd(j.cost.judge)}`} />
        <Tooltip multiline w={340} label="Of the judged answers with a failure signal (sent again, a raised floor, asked again elsewhere), the share the judge also failed, less the share of the rest it failed: 0 means its verdicts are unrelated to what people did, 1 that they match. It sets the labels' weight in training, shrunk towards 0.3 while few judged answers have a signal.">
          <div><Kpi label="Agreement with outcome signals" value={j.agreement == null ? "–" : `${j.agreement > 0 ? "+" : ""}${j.agreement.toFixed(2)}`}
            sub={j.agreement == null ? (j.labels ? "no judged answer has a signal yet" : "no labels yet")
              : `caught ${share(j.caught)} · false alarms ${share(j.false_alarm)}`} /></div>
        </Tooltip>
      </SimpleGrid>
      <Text size="xs" c="dimmed" mt="xs">
        Weight in training: {j.weight == null ? "–" : j.weight.toFixed(2)} from these requests
        {weightInUse != null ? `; ${weightInUse.toFixed(2)} in the model in use` : ""}. {j.preferred_first != null
          ? `The judge preferred the answer shown first ${pct(j.preferred_first)} of the time (about half: the order doesn't sway it), and the stronger one ${pct(j.preferred_stronger ?? 0)}.`
          : ""}
      </Text>
      {Object.keys(j.outcomes).length > 0 && (
        <Group gap={6} mt="xs">{Object.entries(j.outcomes).map(([k, n]) => (
          <Badge key={k} color={k === "capped" ? "yellow" : k === "judged" ? "mo" : "gray"}>{JUDGED_OUTCOMES[k] ?? k}: {num(n)}</Badge>))}</Group>
      )}
      {j.teams.length > 0 && (
        <Table.ScrollContainer minWidth={760} mt="sm">
          <Table verticalSpacing={4} fz="sm">
            <Table.Thead><Table.Tr><Table.Th>Team</Table.Th><Table.Th>Sampled</Table.Th><Table.Th>Stronger model</Table.Th><Table.Th>Judges</Table.Th>
              <Table.Th ta="right">This month / cap</Table.Th></Table.Tr></Table.Thead>
            <Table.Tbody>
              {j.teams.map((t) => (
                <Table.Tr key={t.team}>
                  <Table.Td fw={600}>{t.team}</Table.Td>
                  <Table.Td>{t.enabled ? pct(t.sample_rate) : <Badge size="xs" color="gray">off</Badge>}</Table.Td>
                  <Table.Td className="mono">{t.stronger}</Table.Td>
                  <Table.Td><Text size="xs" c="dimmed">{t.judges.join(", ")}</Text></Table.Td>
                  <Table.Td className="num">{usd(t.spent_month)} / {money(t.cap)}
                    {t.capped && <StatusBadge size="sm" kind="warn" ml={6} title="No new sample starts until next month, or until the cap is raised">cap reached</StatusBadge>}</Table.Td>
                </Table.Tr>
              ))}
            </Table.Tbody>
          </Table>
        </Table.ScrollContainer>
      )}
      {j.levels.length > 0 && (
        <Text size="xs" c="dimmed" mt="xs">Not good enough, by the rules' level: {j.levels.map((g) =>
          `${g.task}:${g.level} ${num(g.not_good_enough)} of ${num(g.labels)}`).join(" · ")}</Text>
      )}
      {Object.keys(j.why).length > 0 && (
        <Text size="xs" c="dimmed" mt={4}>No label: {Object.entries(j.why).map(([w, n]) => `${w} (${num(n)})`).join(" · ")}</Text>
      )}
    </Panel>
  );
}

function ModelStatus({ s }: { s: RouterStatus }) {
  const mode = { off: "Off", shadow: "Shadow: compares, changes nothing", route: "Route: used when sure" }[s.mode];
  const h = s.metrics?.holdout;
  return (
    <Stack gap={6}>
      <Group gap="xs">
        {s.mode === "route" ? <StatusBadge kind="good">{mode}</StatusBadge> : <Badge color={s.mode === "shadow" ? "mo" : "gray"}>{mode}</Badge>}
        {s.loaded ? <Text size="sm">Model v{s.loaded}, trained {s.age_days} days ago on {num(s.examples ?? 0)} requests</Text>
          : <Text size="sm" c="dimmed">{s.load_error ?? "No model in use: the rules decide every level."}</Text>}
        {s.stale && <StatusBadge kind="warn" title={s.stale}>stale: the rules decide</StatusBadge>}
      </Group>
      {h && (
        <Text size="xs" c="dimmed">
          On the newest {num(h.requests)} requests it was trained without, it predicted outcomes with a cross-entropy of {h.loss.toFixed(4)},
          against {h.rules_loss.toFixed(4)} for the rules alone (lower is better{h.better ? ", by more than chance" : ", not by more than chance"}).
          At {pct(s.min_confidence)} confidence it would have raised {num(h.would_raise)} of them and lowered {num(h.would_lower)}.
          {s.metrics?.judged?.labels ? ` It also learned from ${num(s.metrics.judged.labels)} judged labels, at a weight of ${(s.metrics.judged.weight ?? 0).toFixed(2)}.` : ""}
        </Text>
      )}
      <Text size="xs" c="dimmed">
        Settings: Config → Routing → Learned router. Train with <Code>meritorder router train</Code>
        {s.retrain_hours ? `, or the worker does every ${s.retrain_hours} hours` : ""}. A model is used only if it beats the rules on the requests
        it was trained without, and the rules decide whenever it's missing, stale, unsure (under {pct(s.min_confidence)}) or fails.
      </Text>
    </Stack>
  );
}

function Versions({ api, s, onChange }: { api: Api; s: RouterStatus; onChange: () => void }) {
  const [busy, setBusy] = useState<number | "none" | "train" | null>(null);
  const act = async (what: number | "none" | "train") => {
    setBusy(what);
    try {
      if (what === "train") {
        await api.trainRouter();
        notifications.show({ color: "green", title: "Training requested", message: "The worker trains a model within a minute or so." });
      } else {
        await api.useRouterModel(what === "none" ? null : what);
        notifications.show({ color: "green", title: what === "none" ? "No model in use" : `Using model v${what}`, message: "Gateways follow within seconds." });
      }
      onChange();
    } catch (e) {
      notifications.show({ color: "red", title: "Couldn't do that", message: e instanceof Error ? e.message : String(e) });
    } finally {
      setBusy(null);
    }
  };
  const last = s.last_training;
  return (
    <Panel title="Model versions" right={
      <Group gap="xs">
        {s.active != null && <Button size="xs" variant="default" loading={busy === "none"} onClick={() => act("none")}>Use none</Button>}
        <Button size="xs" loading={busy === "train"} disabled={!!s.train_requested} onClick={() => act("train")}>
          {s.train_requested ? "Training requested" : "Train now"}</Button>
      </Group>}>
      {last && (
        <Text size="xs" c="dimmed" mb="xs">
          Last training {new Date(last.at).toLocaleString()} by {last.by}: {num(last.metrics.requests)} requests,
          {" "}{num(last.metrics.with_signal ?? 0)} with an outcome signal. {last.version ? `Saved as v${last.version}${last.active ? ", in use." : `, not in use: ${last.why}`}`
            : `Not saved: ${last.why}`}
        </Text>
      )}
      {s.versions?.length ? (
        <Table.ScrollContainer minWidth={760}>
          <Table verticalSpacing={5} fz="sm">
            <Table.Thead><Table.Tr><Table.Th>Version</Table.Th><Table.Th>Trained</Table.Th><Table.Th>By</Table.Th><Table.Th ta="right">Requests</Table.Th>
              <Table.Th ta="right">With a signal</Table.Th><Table.Th ta="right">Judged labels (weight)</Table.Th>
              <Table.Th ta="right">Held out: learned / rules</Table.Th><Table.Th /></Table.Tr></Table.Thead>
            <Table.Tbody>
              {s.versions.map((v) => (
                <Table.Tr key={v.version}>
                  <Table.Td className="mono">v{v.version} {v.active && <Badge size="sm" color="mo" ml={4}>in use</Badge>}</Table.Td>
                  <Table.Td>{new Date(v.created_at).toLocaleString()}</Table.Td>
                  <Table.Td>{v.created_by}</Table.Td>
                  <Table.Td className="num">{num(v.metrics?.requests ?? 0)}</Table.Td>
                  <Table.Td className="num">{num(v.metrics?.with_signal ?? 0)}</Table.Td>
                  <Table.Td className="num">{v.metrics?.judged?.labels ? `${num(v.metrics.judged.labels)} (${(v.metrics.judged.weight ?? 0).toFixed(2)})` : "–"}</Table.Td>
                  <Table.Td className="num">{v.metrics?.holdout ? `${v.metrics.holdout.loss.toFixed(4)} / ${v.metrics.holdout.rules_loss.toFixed(4)}` : "–"}
                    {v.metrics?.better === false && <Badge size="xs" color="gray" ml={6}>no better</Badge>}</Table.Td>
                  <Table.Td ta="right">{!v.active && <Button size="compact-xs" variant="default" loading={busy === v.version}
                    onClick={() => act(v.version)}>Use this one</Button>}</Table.Td>
                </Table.Tr>
              ))}
            </Table.Tbody>
          </Table>
        </Table.ScrollContainer>
      ) : <Text size="sm" c="dimmed">No model trained yet.</Text>}
    </Panel>
  );
}

function Groups({ r }: { r: RouterReport }) {
  return (
    <Panel title="Where it disagrees with the rules">
      <Text size="xs" c="dimmed" mb="sm">
        Costs are the router's estimate for the task from that request on, for the pick at each level. Signals were seen on the side that
        acted; the expected failure rates are the learned model's own estimate for each side's pick, so they can't confirm it.
      </Text>
      <Table.ScrollContainer minWidth={980}>
        <Table>
          <Table.Thead>
            <Table.Tr><Table.Th>Task</Table.Th><Table.Th>Rules → learned</Table.Th><Table.Th ta="right">Requests</Table.Th><Table.Th ta="right">Used</Table.Th>
              <Table.Th ta="right">Sure</Table.Th><Table.Th ta="right">Rules' picks</Table.Th><Table.Th ta="right">Learned picks</Table.Th>
              <Table.Th ta="right">Signals seen</Table.Th><Table.Th ta="right">Expected: rules / learned</Table.Th></Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {r.groups.map((g) => (
              <Table.Tr key={`${g.task}-${g.rules}-${g.learned}`}>
                <Table.Td>{g.task}</Table.Td>
                <Table.Td className="mono">{g.rules} → {g.learned} <Badge size="sm" variant="default" ml={4}>
                  {g.learned > g.rules ? "higher" : "lower"}</Badge></Table.Td>
                <Table.Td className="num">{num(g.requests)}</Table.Td>
                <Table.Td className="num">{num(g.used)}</Table.Td>
                <Table.Td className="num">{share(g.confidence)}</Table.Td>
                <Table.Td className="num">{money(g.rules_cost)}</Table.Td>
                <Table.Td className="num">{money(g.learned_cost)}</Table.Td>
                <Table.Td className="num" title={Object.entries(g.signals).map(([k, n]) => `${SIGNALS[k] ?? k}: ${n}`).join(", ")}>
                  {num(g.with_signal)}</Table.Td>
                <Table.Td className="num">{share(g.expected_rules)} / {share(g.expected_learned)}</Table.Td>
              </Table.Tr>
            ))}
          </Table.Tbody>
        </Table>
      </Table.ScrollContainer>
    </Panel>
  );
}

function Rows({ r }: { r: RouterReport }) {
  return (
    <Panel title="Recent disagreements" right={<Text size="xs" c="dimmed">Hashed features only: prompts are never stored.</Text>}>
      <Table.ScrollContainer minWidth={1100}>
        <Table>
          <Table.Thead>
            <Table.Tr><Table.Th>Time</Table.Th><Table.Th>User</Table.Th><Table.Th>Rules</Table.Th><Table.Th>Learned</Table.Th><Table.Th>Acted</Table.Th>
              <Table.Th ta="right">Rules' pick</Table.Th><Table.Th ta="right">Learned pick</Table.Th><Table.Th>Signals</Table.Th></Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {r.rows.map((x, i) => (
              <Table.Tr key={i}>
                <Table.Td className="mono" style={{ whiteSpace: "nowrap" }}>{new Date(x.ts).toLocaleString()}</Table.Td>
                <Table.Td><Person id={x.user} /></Table.Td>
                <Table.Td className="mono">{x.task}:{x.rules}</Table.Td>
                <Table.Td className="mono">{x.task}:{x.learned} <Text span size="xs" c="dimmed">({pct(x.confidence)}, v{x.version})</Text></Table.Td>
                <Table.Td><Badge size="sm" color={x.acted === "learned" ? "mo" : "gray"}>{x.acted === "learned" ? "learned" : OUTCOMES[x.outcome] ?? x.outcome}</Badge></Table.Td>
                <Table.Td className="num">{x.rules_model ?? "–"} {money(x.rules_cost)}</Table.Td>
                <Table.Td className="num">{x.learned_model ?? "–"} {money(x.learned_cost)}</Table.Td>
                <Table.Td>{x.signals.length ? x.signals.map((k) => <Badge key={k} size="sm" variant="default" mr={4}>{SIGNALS[k] ?? k}</Badge>) : "–"}</Table.Td>
              </Table.Tr>
            ))}
          </Table.Tbody>
        </Table>
      </Table.ScrollContainer>
      {!r.rows.length && <Text c="dimmed" size="sm" mt="sm">No disagreements in this period.</Text>}
    </Panel>
  );
}

/** Reports → Learned router: the shadow report (admins and team leads; versions and training for admins). */
export function LearnedRouter({ api, me }: { api: Api; me: Me }) {
  const [days, setDays] = useState("7");
  const res = useData(() => api.routerReport(Number(days)), [api, days]);
  const r = res.data;
  const rate = (k: string) => r?.signal_rates[k];
  return (
    <Stack gap="md">
      <Group justify="flex-end" gap="sm">
        <AnsweredBy by={r?.answered_by} />
        <SegmentedControl size="xs" value={days} onChange={setDays} aria-label="Days"
          data={[{ value: "1", label: "Last day" }, { value: "7", label: "7 days" }, { value: "30", label: "30 days" }]} />
      </Group>
      <Loading loading={res.loading} error={res.error} />
      {r && (
        <>
          <Panel title="Learned router"><ModelStatus s={r.status} /></Panel>
          {r.status.mode === "off" && !r.requests && (
            <Text size="sm" c="dimmed">The learned router is off. Set Config → Routing → Learned router → Mode to Shadow: gateways then record
              each request's hashed features and what followed its answer, and this page compares the learned router with the rules once a
              model is trained.</Text>
          )}
          <SimpleGrid cols={{ base: 1, xs: 2, md: 4 }}>
            <Kpi label="Compared" value={num(r.compared)} sub={`${num(r.requests)} requests seen`} />
            <Kpi label="Agreed with the rules" value={r.compared ? pct(r.agreed / r.compared) : "–"} sub={`${num(r.disagreed)} disagreements`} />
            <Kpi label="Cost difference" value={r.cost.priced ? signed(r.cost.difference) : "–"}
              sub={r.cost.priced ? `learned ${usd(r.cost.learned)} vs rules ${usd(r.cost.rules)}` : "nothing priced yet"} />
            <Tooltip label="Requests with a failure signal (sent again, a raised floor, stopped, asked again elsewhere, refused) on the side that acted, where the learned router would have gone higher, lower, or agreed." multiline w={320}>
              <div><Kpi label="Signals: higher / lower / agreed" value={[rate("higher"), rate("lower"), rate("agreed")].map((x) => share(x?.rate)).join(" / ")}
                sub="share of requests with a failure signal" /></div>
            </Tooltip>
          </SimpleGrid>
          {Object.keys(r.outcomes).length > 0 && (
            <Group gap={6}>{Object.entries(r.outcomes).map(([k, n]) => <Badge key={k} variant="light" color="gray">{OUTCOMES[k] ?? k}: {num(n)}</Badge>)}</Group>
          )}
          {r.groups.length > 0 && <Groups r={r} />}
          {r.judged && (r.judged.samples > 0 || r.judged.teams.length > 0)
            ? <Judged j={r.judged} weightInUse={r.status.metrics?.judged?.weight} />
            : <Text size="xs" c="dimmed">Judged labels are off for every team. They add a label that says whether an answer was good enough, at a
              price and by sending a sample of a team's prompts to a stronger model and a judge: Config → Routing → Judged labels.</Text>}
          {r.success && (r.success.predicted.n > 0 || r.success.escalations.escalated > 0 || r.status.success?.mode !== "off")
            ? <ExpectedCost s={r.success} status={r.status} />
            : <Text size="xs" c="dimmed">No team routes by expected total cost yet. With Config → Routing → Choose models by → Expected total cost,
              each task start records the chance it was given and what followed, and this page compares them.</Text>}
          {(r.compared > 0 || r.rows.length > 0) && <Rows r={r} />}
          {me.role === "admin" && <Versions api={api} s={r.status} onChange={() => void res.reload()} />}
        </>
      )}
    </Stack>
  );
}
