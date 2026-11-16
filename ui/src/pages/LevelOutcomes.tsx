import { Badge, Button, Code, Group, SegmentedControl, Stack, Table, Text } from "@mantine/core";
import { notifications } from "@mantine/notifications";
import { Fragment, useState } from "react";
import type { Api, ArmRate, LevelOutcomeRow, LevelOutcomes, LevelRecommendation } from "../api";
import { num, pct, usd } from "../api";
import { niceMax } from "../components/charts";
import { AnsweredBy, Loading, Panel, Status, StatusBadge } from "../components/common";
import { Icon } from "../components/icons";
import { useData } from "../hooks";

const ARM_NAMES: Record<string, string> = { served: "served at the level", down: "explored one level down", up: "explored one level up",
  canary: "a canary at the level" };
const LO = 0.5, HI = 1;  // the interval glyph's scale: success rates sit high, so 50% to 100% shows the differences

/** The 90% interval as a small glyph: the line is the interval, the dot the rate; dashed while there's too little. */
function Interval({ rate, half, served, low }: { rate: number; half: number; served: boolean; low: boolean }) {
  const W = 112;
  const x = (v: number) => 2 + ((Math.min(Math.max(v, LO), HI) - LO) / (HI - LO)) * (W - 4);
  const c = low ? "var(--mantine-color-dimmed)" : served ? "var(--mo-accent)" : "var(--mo-fg-2)";
  return (
    <svg width={W} height={12} viewBox={`0 0 ${W} 12`} aria-hidden="true" style={{ display: "block" }}>
      <line x1={2} x2={W - 2} y1={6} y2={6} stroke="var(--mo-line)" />
      <line x1={x(rate - half)} x2={x(rate + half)} y1={6} y2={6} stroke={c} strokeWidth={1.5} strokeDasharray={low ? "2 2" : undefined} />
      <line x1={x(rate - half)} x2={x(rate - half)} y1={3} y2={9} stroke={c} />
      <line x1={x(rate + half)} x2={x(rate + half)} y1={3} y2={9} stroke={c} />
      <circle cx={x(rate)} cy={6} r={3} fill={c} stroke="var(--mo-surface)" strokeWidth={1.5} />
    </svg>
  );
}

/** A success rate with its 90% interval and sample size; dimmed while there's too little to go on. */
function Rate({ arm, min, name }: { arm: ArmRate; min: number; name: string }) {
  if (!arm.n || arm.rate === null) return <Text size="sm" c="dimmed">none</Text>;
  const low = arm.n < min;
  const half = arm.se !== null ? 1.645 * arm.se : 0;
  const tip = `${ARM_NAMES[name]}: ${num(arm.n)} task starts (${num(arm.n_eff)} after weighting by each arm's chance). `
    + `90% interval ${pct(Math.max(0, arm.rate - half))} to ${pct(Math.min(1, arm.rate + half))}.${low ? ` Fewer than ${num(min)}: not enough to recommend anything.` : ""}`;
  return (
    <div data-tip={tip} tabIndex={0} style={{ display: "grid", gap: 3, width: "max-content" }}>
      <div className="mono">
        <Text span fw={500} size="sm" ff="monospace" c={low ? "dimmed" : undefined}>{pct(arm.rate)}</Text>
        {arm.se !== null && <Text span size="xs" ff="monospace" c="dimmed"> ± {(half * 100).toFixed(1)}</Text>}
      </div>
      <Interval rate={arm.rate} half={half} served={name === "served"} low={low} />
      <Text size="xs" ff="monospace" c="dimmed">n={num(arm.n)}</Text>
    </div>
  );
}

function Verdict({ row, side }: { row: LevelOutcomeRow; side: "too_high" | "too_low" }) {
  const v = row[side];
  if (v.ratio === null) return <Text size="sm" c="dimmed">none</Text>;
  const text = side === "too_high" ? `level ${row.level - 1} does ${pct(v.ratio)} as well` : `level ${row.level + 1} succeeds ${pct(v.ratio)} as often`;
  return (
    <Group gap={6} wrap="wrap" maw={240}>
      <Text size="sm" c={v.enough ? "var(--mo-fg-2)" : "dimmed"}>{text}</Text>
      {v.clear && (side === "too_high" ? <Badge size="sm" color="mo">too high</Badge> : <StatusBadge kind="warn" size="sm">too low</StatusBadge>)}
      {!v.enough && <Badge size="sm" variant="default">too few</Badge>}
    </Group>
  );
}

/** Daily spend (the accent line over a faint area) with the days the catalog or the scores changed marked and numbered,
 * matching the list under it. */
function Trend({ data }: { data: LevelOutcomes }) {
  const W = 1360, H = 190, L = 56, R = 10, T = 28, B = 24;
  const days = data.trend;
  if (!days.length) return <Text size="sm" c="dimmed">No requests with a level in this period.</Text>;
  const max = niceMax(Math.max(...days.map((d) => d.cost), 0.0001));
  const x = (i: number) => L + (days.length > 1 ? (i / (days.length - 1)) * (W - L - R) : (W - L - R) / 2);
  const y = (v: number) => T + (H - T - B) * (1 - v / max);
  const line = days.map((d, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(d.cost).toFixed(1)}`).join(" ");
  const area = `${line} L${x(days.length - 1).toFixed(1)},${y(0).toFixed(1)} L${x(0).toFixed(1)},${y(0).toFixed(1)} Z`;
  const index = new Map(days.map((d, i) => [d.day, i]));
  const marks = data.events.slice(-8).map((e, k) => ({ e, k })).filter(({ e }) => index.has(e.at.slice(0, 10)));
  const every = Math.max(1, Math.ceil(days.length / 6));
  return (
    <div className="chart-scroll">
      <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Spend per day, with catalog and score changes marked">
        {[0, 0.5, 1].map((f) => (
          <g key={f}><line className="grid" x1={L} x2={W - R} y1={y(f * max)} y2={y(f * max)} />
            <text x={L - 6} y={y(f * max) + 3.5} textAnchor="end">{usd(f * max)}</text></g>
        ))}
        {days.map((d, i) => (i % every === 0 || i === days.length - 1) && (
          <text key={d.day} x={x(i)} y={H - 6} textAnchor={i === 0 ? "start" : i === days.length - 1 ? "end" : "middle"}>
            {new Date(`${d.day}T00:00:00`).toLocaleDateString(undefined, { month: "short", day: "numeric" })}</text>
        ))}
        {marks.map(({ e, k }) => {
          const xi = x(index.get(e.at.slice(0, 10)) ?? 0);
          return (
            <g key={k} data-tip={`${e.at.slice(0, 16).replace("T", " ")}: ${e.text}`}>
              <line x1={xi} x2={xi} y1={T - 6} y2={y(0)} stroke="var(--mantine-color-dimmed)" strokeDasharray="3 3" />
              <rect x={xi - 8} y={T - 20} width={16} height={15} rx={4} fill="var(--mo-surface)" stroke="var(--mo-line-strong)" />
              <text x={xi} y={T - 9} textAnchor="middle" className="t-fg">{k + 1}</text>
            </g>
          );
        })}
        <path d={area} fill="var(--mo-accent)" opacity={0.08} />
        <path d={line} fill="none" stroke="var(--mo-accent)" strokeWidth={2} strokeLinejoin="round" />
        {days.map((d, i) => (
          <rect key={d.day} className="hit" x={x(i) - (W - L - R) / Math.max(days.length - 1, 1) / 2} width={(W - L - R) / Math.max(days.length - 1, 1)}
            y={T} height={H - T - B} fill="transparent" data-tip={`${d.day}: ${usd(d.cost)}, ${num(d.requests)} requests`} />
        ))}
      </svg>
    </div>
  );
}

/** A recommendation, on its own row under the level it's about: Apply and Dismiss for admins, or what became of it. */
function Recommendation({ rec, admin, min, onApply, onDismiss }: { rec: LevelRecommendation; admin: boolean; min: number;
  onApply: (r: LevelRecommendation) => void; onDismiss: (r: LevelRecommendation, undo: boolean) => void }) {
  if (rec.dismissed) {
    return (
      <div className="rec dismissed">
        <Icon name="off" />
        <span className="text">Dismissed{rec.dismissed.by ? ` by ${rec.dismissed.by}` : ""} on {rec.dismissed.at.slice(0, 10)}: {rec.text}.
          It comes back after {num(min)} more explored task starts ({num(rec.dismissed.new_outcomes)} so far) or if it points another way.</span>
        {admin && <Button size="compact-sm" variant="subtle" color="gray" onClick={() => onDismiss(rec, true)}>Undo</Button>}
      </div>
    );
  }
  return (
    <div className="rec">
      <Text span c="var(--mo-accent-text)" style={{ display: "inline-flex" }}><Icon name="arrow" size={12} /></Text>
      <span className="text">{rec.text}</span>
      {rec.applied ? <StatusBadge kind="good" size="sm">applied</StatusBadge> : admin && (
        <Group gap={6}>
          <Button size="compact-sm" onClick={() => onApply(rec)}>Apply</Button>
          <Button size="compact-sm" variant="subtle" color="gray" onClick={() => onDismiss(rec, false)}>Dismiss</Button>
        </Group>
      )}
    </div>
  );
}

/** Reports → Levels: whether each level is too high or too low for the work, from your own traffic. */
export function LevelOutcomesPanel({ api, admin }: { api: Api; admin: boolean }) {
  const [days, setDays] = useState("30");
  const res = useData(() => api.levelOutcomes(Number(days)), [api, days]);
  const [busy, setBusy] = useState(false);
  const d = res.data;
  if (res.error && /403|admins and team leads/.test(res.error)) return null;  // people without a team to lead see the rules only

  async function act(what: () => Promise<{ title: string; message: string } | null>, failed: string) {
    setBusy(true);
    try {
      const done = await what();
      if (done) notifications.show({ color: "green", ...done });
      await res.reload();
    } catch (e) {
      notifications.show({ color: "red", title: failed, message: e instanceof Error ? e.message : String(e) });
    } finally {
      setBusy(false);
    }
  }
  const apply = (rec: LevelRecommendation) => act(async () => {
    const out = await api.applyLevel(rec.task, rec.level, rec.to);
    return { title: out.version ? `Saved as version ${out.version}` : "Nothing to change", message: out.changed.join("; ") || "The level map already says so." };
  }, "Couldn't apply");
  const dismiss = (rec: LevelRecommendation, undo: boolean) => act(async () => {
    await api.dismissLevel(rec, undo);
    return null;
  }, undo ? "Couldn't bring it back" : "Couldn't dismiss");

  return (
    <Panel title="Is each level right for the work?" flush right={
      <SegmentedControl value={days} onChange={setDays} data={[{ value: "7", label: "7 days" }, { value: "30", label: "30 days" },
        { value: "90", label: "90 days" }]} />}
      footer={d && <><span>{new Date(d.since).toLocaleDateString(undefined, { month: "short", day: "numeric" })} to today</span><AnsweredBy by={d.answered_by} /></>}>
      {(res.loading || res.error) && <div style={{ padding: 16 }}><Loading loading={res.loading} error={res.error} /></div>}
      {d && (
        <>
          <Stack gap="sm" p="md">
            <Text size="sm" c="var(--mo-fg-2)" maw="96ch">
              Teams that explore send a few of their task starts one level down or up from the level they'd get, at random (Config → Teams →
              Exploration). Comparing how often those succeed with the ones served at the level says whether a level is too high (the level
              below does as well, for less) or too low (the level above clearly does better). Success is a task start without a failure
              signal: no correction in the next prompt, no tests failing at the end, not sent again, and the rest
              (docs/03-routing.md#the-outcome-report). Each arm is weighted by the inverse of its chance, so teams exploring at different
              shares pool fairly. Nothing is recommended below {num(d.settings.min_outcomes)} task starts on each side.
            </Text>
            {!d.recording && <Status kind="warn">The learned router is off, so no outcomes are recorded. Set Config → Routing →
              Learned router to shadow mode to start.</Status>}
            {d.recording && !d.exploring.length && <Status kind="warn">No team explores yet, so only the served level is measured.</Status>}
            <Group gap="8px 22px" fz="sm" c="var(--mo-fg-2)">
              <span><Text span c="dimmed" mr={6}>Exploring</Text>{d.exploring.length ? d.exploring.join(", ") : "no team"}</span>
              <span><Text span c="dimmed" mr={6}>Explored</Text><span className="mono">{num(d.explored_requests)} of {num(d.requests)}</span> requests</span>
              <span><Text span c="dimmed" mr={6}>Levels moved</Text>{Object.keys(d.level_map).length
                ? Object.entries(d.level_map).map(([k, v]) => <Code key={k} mr={4}>{k} → {v}</Code>) : "none"}</span>
              <span><Text span c="dimmed" mr={6}>Auto-apply</Text>{d.settings.auto_apply ? "on" : "off"}</span>
            </Group>
            <Text size="xs" c="dimmed">Success rate by arm, with its 90% interval on a 50% to 100% scale. A dashed interval has fewer than
              {" "}{num(d.settings.min_outcomes)} task starts.</Text>
          </Stack>
          <Table.ScrollContainer minWidth={1080}>
            <Table highlightOnHover={false} style={{ borderTop: "1px solid var(--mo-line)" }}>
              <Table.Thead>
                <Table.Tr>
                  <Table.Th>Level</Table.Th><Table.Th ta="right">Traffic</Table.Th><Table.Th ta="right">Spend</Table.Th>
                  <Table.Th>Served</Table.Th><Table.Th>One down</Table.Th><Table.Th>One up</Table.Th>
                  <Table.Th>Too high?</Table.Th><Table.Th>Too low?</Table.Th>
                </Table.Tr>
              </Table.Thead>
              <Table.Tbody>
                {d.levels.map((r) => {
                  const maxTraffic = Math.max(...d.levels.map((x) => x.traffic_share), 0.0001);
                  const maxSpend = Math.max(...d.levels.map((x) => x.spend_share ?? 0), 0.0001);
                  return (
                    <Fragment key={`${r.task}:${r.level}`}>
                      <Table.Tr style={{ verticalAlign: "top", borderBottomColor: r.recommendation ? "transparent" : undefined }}>
                        <Table.Td className="mono" py={9}>{r.task}:{r.level}</Table.Td>
                        <Table.Td className="num" py={9}>{pct(r.traffic_share)}<div className="mini"><i style={{ width: `${(r.traffic_share / maxTraffic) * 100}%` }} /></div></Table.Td>
                        <Table.Td className="num" py={9}>{r.spend_share === null ? "–" : pct(r.spend_share)}<div className="mini"><i style={{ width: `${((r.spend_share ?? 0) / maxSpend) * 100}%` }} /></div></Table.Td>
                        <Table.Td py={9}><Rate arm={r.arms.served} min={r.min_outcomes} name="served" /></Table.Td>
                        <Table.Td py={9}><Rate arm={r.arms.down} min={r.min_outcomes} name="down" /></Table.Td>
                        <Table.Td py={9}><Rate arm={r.arms.up} min={r.min_outcomes} name="up" /></Table.Td>
                        <Table.Td py={9}><Verdict row={r} side="too_high" /></Table.Td>
                        <Table.Td py={9}><Verdict row={r} side="too_low" /></Table.Td>
                      </Table.Tr>
                      {r.recommendation && (
                        <Table.Tr>
                          <Table.Td pt={0} />
                          <Table.Td colSpan={7} pt={0} pb={12}>
                            <Recommendation rec={r.recommendation} admin={admin && !busy} min={r.min_outcomes} onApply={apply} onDismiss={dismiss} />
                          </Table.Td>
                        </Table.Tr>
                      )}
                    </Fragment>
                  );
                })}
              </Table.Tbody>
            </Table>
          </Table.ScrollContainer>
          <Stack gap="sm" p="md" style={{ borderTop: "1px solid var(--mo-line)" }}>
            <Text size="xs" c="dimmed">Applying writes <Code>routing.level_map</Code> as a new configuration version: that level's work
              is then routed at the new one. {d.settings.auto_apply ? "Recommendations the learned router agrees with are applied automatically."
              : <>With <Code>routing.benchmark.auto_apply</Code>, the worker applies those the learned router agrees with.</>} A dismissed
              recommendation stays dismissed until its evidence changes: it points another way, or {num(d.settings.min_outcomes)} more task
              starts are explored on the level it compares with.</Text>
            {d.canaries.length > 0 && (
              <Text size="xs" c="dimmed">Canaries (tried on a share of their level's task starts before they take the rest): {d.canaries.map((c) =>
                `${c.model} at ${c.task}:${c.level} (${c.reason}${c.outcomes !== undefined ? `, ${c.outcomes} outcomes` : ""})`).join("; ")}</Text>
            )}
            <Text size="sm" fw={600}>Spend per day, with catalog and score changes</Text>
            <Trend data={d} />
            {d.events.length > 0 && (
              <Stack gap={6}>
                {d.events.slice(-8).map((e, i) => (
                  <Group key={i} gap={8} wrap="nowrap" align="baseline" fz="sm" c="var(--mo-fg-2)">
                    <Badge size="xs" variant="default" className="mono" w={18} px={0}>{i + 1}</Badge>
                    <Text span className="mono" size="xs" c="dimmed" w={84}>{e.at.slice(0, 10)}</Text>
                    <Text span size="sm">{e.text}</Text>
                  </Group>
                ))}
              </Stack>
            )}
          </Stack>
        </>
      )}
    </Panel>
  );
}
