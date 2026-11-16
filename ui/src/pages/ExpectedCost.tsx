import { Badge, Group, SimpleGrid, Table, Text, Tooltip } from "@mantine/core";
import type { RouterStatus, SuccessBin, SuccessReport } from "../api";
import { num, pct, usd } from "../api";
import { Kpi, Panel } from "../components/common";

const share = (v: number | null | undefined) => (v == null ? "–" : pct(v));
const money = (v: number | null | undefined) => (v == null ? "–" : usd(v));
const MODES: Record<string, string> = { off: "off: the benchmark prior alone", shadow: "shadow: predicted and recorded, not used",
  route: "route: used once it beats the benchmark prior" };

/** A reliability table: in each band of predicted chance, how many task starts, their mean prediction, and the share that
 * ended without a failure signal. A calibrated predictor has the two close in every band. */
function Bins({ bins }: { bins: SuccessBin[] }) {
  return (
    <Table.ScrollContainer minWidth={520} mt="xs">
      <Table verticalSpacing={3} fz="sm">
        <Table.Thead><Table.Tr><Table.Th>Predicted</Table.Th><Table.Th ta="right">Task starts</Table.Th><Table.Th ta="right">Mean predicted</Table.Th>
          <Table.Th ta="right">Seen</Table.Th><Table.Th w={180}>Seen against predicted</Table.Th></Table.Tr></Table.Thead>
        <Table.Tbody>
          {bins.map((b) => (
            <Table.Tr key={b.from}>
              <Table.Td className="mono">{pct(b.from)}–{pct(b.to)}</Table.Td>
              <Table.Td className="num">{num(b.n)}</Table.Td>
              <Table.Td className="num">{pct(b.predicted)}</Table.Td>
              <Table.Td className="num">{pct(b.seen)}</Table.Td>
              <Table.Td>
                <div className="mini acc" title={`predicted ${pct(b.predicted)}, seen ${pct(b.seen)}`}><i style={{ width: `${b.seen * 100}%` }} /></div>
                <div className="mini" style={{ marginTop: 2 }}><i style={{ width: `${b.predicted * 100}%`, background: "var(--mo-fg-3)" }} /></div>
              </Table.Td>
            </Table.Tr>
          ))}
        </Table.Tbody>
      </Table>
    </Table.ScrollContainer>
  );
}

/** Reports → Learned router: routing by expected total cost (decision 62). How well the predicted chances match what
 * followed, how often tasks escalated and whether that helped, and expected against actual cost per finished task. */
export function ExpectedCost({ s, status }: { s: SuccessReport; status: RouterStatus }) {
  const fitted = status.success;
  const hold = fitted?.metrics?.holdout;
  const e = s.escalations;
  return (
    <Panel title="Expected total cost" right={<Text size="xs" c="dimmed">Teams routing by expected total cost: the chance each task start was given, against what followed.</Text>}>
      <SimpleGrid cols={{ base: 1, xs: 2, md: 4 }}>
        <Tooltip multiline w={340} label="The chance recorded for the model that served each task start, and the share of them that ended without a failure signal (sent again, a raised floor, a correction, tests failing at the end, ...). The Brier score is the mean squared difference: lower is better, 0.25 is a coin toss.">
          <div><Kpi label="Predicted / seen success" value={`${share(s.predicted.predicted)} / ${share(s.predicted.seen)}`}
            sub={s.predicted.n ? `${num(s.predicted.n)} task starts · Brier ${s.predicted.brier?.toFixed(3)}` : "no task start recorded yet"} /></div>
        </Tooltip>
        <Kpi label="Escalated" value={share(e.rate)} sub={`${num(e.escalated)} of ${num(e.tasks)} tasks`} />
        <Tooltip multiline w={320} label="Escalated tasks that ended without a failure signal: the stronger model finished what the first one couldn't.">
          <div><Kpi label="Escalations that helped" value={share(e.helped_rate)} sub={`${num(e.helped)} tasks · ${usd(e.extra_cost)} charged against routing savings`} /></div>
        </Tooltip>
        <Tooltip multiline w={340} label="Per finished task (one that ended with the next human prompt): the expected total cost at its start, the person's time included, against what its requests cost in tokens.">
          <div><Kpi label="Expected / actual cost per task" value={`${money(s.cost.expected)} / ${money(s.cost.actual)}`}
            sub={s.cost.tasks ? `${num(s.cost.tasks)} finished tasks; expected includes the person's time` : "no finished task yet"} /></div>
        </Tooltip>
      </SimpleGrid>
      <Group gap={6} mt="xs">
        <Badge variant="light" color={fitted?.mode === "route" ? "mo" : "gray"}>Fitted layer {MODES[fitted?.mode ?? "off"]}</Badge>
        {fitted?.version != null && <Badge variant="light" color={fitted.better ? "mo" : "yellow"}>v{fitted.version}: {fitted.better ? "beat the benchmark prior" : "no better than the prior"}</Badge>}
        {hold && <Text size="xs" c="dimmed">Held-out Brier {hold.brier.toFixed(3)} against the prior's {hold.prior_brier.toFixed(3)} on {num(hold.examples)} task starts.</Text>}
        {Object.entries(e.by_reason).map(([k, n]) => <Badge key={k} variant="default">escalated on {k}: {num(n)}</Badge>)}
      </Group>
      {s.predicted.bins.length > 0 && <Bins bins={s.predicted.bins} />}
      {s.fitted.n > 0 && (
        <Text size="xs" c="dimmed" mt="xs">The fitted layer's own chances on the same task starts: {share(s.fitted.predicted)} predicted against {share(s.fitted.seen)} seen,
          Brier {s.fitted.brier?.toFixed(3)}{fitted?.mode === "shadow" ? " (shadow mode: recorded, not used)" : ""}.</Text>
      )}
      {s.by_model.length > 0 && (
        <Text size="xs" c="dimmed" mt={4}>By model: {s.by_model.slice(0, 8).map((m) => `${m.model} ${share(m.predicted)} predicted, ${share(m.seen)} seen (${num(m.n)})`).join(" · ")}</Text>
      )}
    </Panel>
  );
}
