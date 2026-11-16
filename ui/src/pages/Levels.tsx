import { Badge, Code, Group, Stack, Table, Text, Tooltip } from "@mantine/core";
import type { ReactNode } from "react";
import type { Api, LevelModel, LevelRules, LevelsView, Me } from "../api";
import { num, usd } from "../api";
import { Loading, Panel } from "../components/common";
import { useData } from "../hooks";
import { LevelOutcomesPanel } from "./LevelOutcomes";

const CLASS_NAMES: Record<string, string> = {
  auxiliary: "Claude Code side requests (session titles, summaries)",
  compaction: "Claude Code compaction summaries",
  subagent: "Claude Code subagent turns",
  main: "Claude Code conversation turns",
};

function Words({ words }: { words: string[] }) {
  return (
    <Group gap={4} mt={4}>
      {words.map((w) => <Code key={w} fz="xs">{w}</Code>)}
    </Group>
  );
}

/** "(default 5)" next to a value someone changed in routing.levels. */
function Changed({ rules, defaults, k }: { rules: LevelRules; defaults: LevelRules; k: keyof LevelRules }) {
  if (JSON.stringify(rules[k]) === JSON.stringify(defaults[k])) return null;
  const d = defaults[k];
  return <Text span size="xs" c="var(--mo-accent-text)"> (changed; default {Array.isArray(d) ? `${d.length} patterns` : String(d)})</Text>;
}

function Models({ list, reach = [] }: { list: LevelModel[]; reach?: LevelModel[] }) {
  if (!list.length && !reach.length) return <Text size="sm" c="dimmed">none: the best available model is used</Text>;
  const shown = list.slice(0, 3);
  const all = list.map((m) => `${m.name} (${m.score})`).join(", ");
  // Routing by cost per success: models below the level whose chance there is at least the minimum.
  const below = reach.map((m) => `${m.name} (${m.score}, ${Math.round((m.chance ?? 0) * 100)}%)`).join(", ");
  return (
    <Stack gap={2}>
      {list.length > 0 && (
        <Tooltip label={all} multiline w={360} withArrow>
          <Text size="sm">
            {shown.map((m) => `${m.name} (${m.score})`).join(", ")}
            {list.length > shown.length && <Text span c="dimmed"> and {list.length - shown.length} more</Text>}
          </Text>
        </Tooltip>
      )}
      {reach.length > 0 && (
        <Tooltip label={`Below the level, with a high enough chance of success at it: ${below}`} multiline w={360} withArrow>
          <Text size="xs" c="dimmed" lineClamp={1}>also, by cost per success: {below}</Text>
        </Tooltip>
      )}
    </Stack>
  );
}

/** The rules routing.levels sets, as a table people can read: which request gets which level. */
export function LevelRulesTable({ view }: { view: LevelsView }) {
  const r = view.rules;
  const d = view.defaults;
  const rows: [ReactNode, ReactNode][] = [];
  rows.push([<>Plain text: anything that isn't coding</>, <>general {r.general}<Changed rules={r} defaults={d} k="general" /></>]);
  if (r.short_tokens > 0) {
    rows.push([<>Plain text, and the whole request is under {num(r.short_tokens)} tokens: a question on its own
      <Changed rules={r} defaults={d} k="short_tokens" /></>, <>general {r.short}<Changed rules={r} defaults={d} k="short" /></>]);
  }
  rows.push([<>Plain text with analysis wording<Changed rules={r} defaults={d} k="analysis_words" /><Words words={view.labels.analysis_words} /></>,
    <>general {r.analysis}<Changed rules={r} defaults={d} k="analysis" /></>]);
  rows.push([<>Any tools, or text that looks like code<Changed rules={r} defaults={d} k="coding_words" /><Words words={view.labels.coding_words} /></>,
    <>coding {r.coding}<Changed rules={r} defaults={d} k="coding" /></>]);
  if (r.agent_by_tools) {
    rows.push([<>{r.agent_tools} or more tools: an agent at work, whatever the wording (simple wording doesn't lower it)
      <Changed rules={r} defaults={d} k="agent_by_tools" /><Changed rules={r} defaults={d} k="agent_tools" /></>,
      <>coding {r.agent}<Changed rules={r} defaults={d} k="agent" /></>]);
  }
  if (r.simple_words.length > 0) {
    rows.push([<>Coding with simple wording and nothing harder<Changed rules={r} defaults={d} k="simple_words" /><Words words={view.labels.simple_words} /></>,
      <>coding {r.simple}<Changed rules={r} defaults={d} k="simple" /></>]);
  }
  if (r.hard_raise > 0) {
    const most = r.hard_raise * r.hard_signals;
    rows.push([<>Coding with hard wording, for each different sign of hard work (up to {r.hard_signals})
      <Changed rules={r} defaults={d} k="hard_words" /><Changed rules={r} defaults={d} k="hard_signals" /><Words words={view.labels.hard_words} /></>,
      <>{r.hard_raise} level{r.hard_raise > 1 ? "s" : ""} higher a sign{most > r.hard_raise ? `, ${most} at most` : ""}, never past 5
        <Changed rules={r} defaults={d} k="hard_raise" /></>]);
  }
  if (r.hardest_words.length > 0) {
    rows.push([<>Coding with the hardest wording: system design, proofs, codebase-wide work<Changed rules={r} defaults={d} k="hardest_words" />
      <Words words={view.labels.hardest_words} /></>, <>coding {r.hardest}<Changed rules={r} defaults={d} k="hardest" /></>]);
  }
  if (r.correction_raise > 0) {
    rows.push([<>A new prompt that says the last answer was wrong ("that's wrong", "still not working", "revert", ...)</>,
      <>{r.correction_raise} level{r.correction_raise > 1 ? "s" : ""} higher, at most 5<Changed rules={r} defaults={d} k="correction_raise" /></>]);
  }
  rows.push([<>Header <Code>X-Route-Hint: cheap</Code> or <Code>smart</Code></>, <>one level lower or higher</>]);
  for (const [cls, cap] of Object.entries(view.request_class_levels)) {
    rows.push([<>{CLASS_NAMES[cls] ?? `Claude Code ${cls} requests`} (<Code>x-claude-code-request-class: {cls}</Code>)</>, <>at most {cap}</>]);
  }
  rows.push([<>Header <Code>X-Route-Floor: coding:4</Code></>, <>exactly that, never capped</>]);
  return (
    <Table verticalSpacing={6} fz="sm">
      <Table.Thead><Table.Tr><Table.Th>The request</Table.Th><Table.Th w={220}>Level</Table.Th></Table.Tr></Table.Thead>
      <Table.Tbody>
        {rows.map(([when, level], i) => <Table.Tr key={i}><Table.Td>{when}</Table.Td><Table.Td>{level}</Table.Td></Table.Tr>)}
      </Table.Tbody>
    </Table>
  );
}

export function Levels({ api, me }: { api: Api; me?: Me }) {
  const res = useData(() => api.levels(), [api]);
  const v = res.data;
  return (
    <Stack gap="md">
      <Loading loading={res.loading} error={res.error} />
      {v && (
        <Panel title="How requests get a level">
          <Text size="sm" c="dimmed" mb="sm">
            A request sent to <Code>{v.auto_model}</Code> gets a level for its kind of work, coding or general, from 1 to 5, and goes to
            the cheapest model scoring at least that. The level comes from the work: the request's wording, its tools, and the client's
            own floor or hint. Tools say the work is coding; the task's wording says how hard, since coding agents send dozens of tools on
            every request. Where several wording rows apply, the hardest wins; otherwise the row further down wins: analysis wording beats
            a short question, and <Code>X-Route-Floor</Code> beats everything. Wording is looked for in the last two messages from the
            person, not in tool results or what the coding tool adds (Claude Code's system reminders, Codex's AGENTS.md). A task keeps its
            level until the person's next prompt.
          </Text>
          <LevelRulesTable view={v} />
          <Text size="sm" mt="sm">
            <Badge size="sm" color="gray" mr={6}>Length</Badge>
            How long the conversation is never raises the level. It only rules out the models whose context window can't hold it
            plus the reply. A request too long for every model at its level goes to the largest window, and the provider decides.
          </Text>
          <Text size="xs" c="dimmed" mt="xs">Admins change these rules under Config → Routing → How requests get a level.</Text>
        </Panel>
      )}
      {v && (
        <Panel title="What each level means">
          <Text size="sm" c="dimmed" mb="sm">
            A model qualifies for a level when its score for that kind of work (1 to 5; a 5 solves FrontierSWE-hard tasks as often as the best model did in November 2026, a fixed bar) is at least the level. These are
            the models your team may use, cheapest first, with their scores.
            {v.model_choice === "cost_per_success" && ` Your team routes by cost per success: a model below a level can take its work too when
              its chance of solving it is at least ${Math.round((v.min_success_chance ?? 0.6) * 100)}% and it costs less per solved task.
              Those are listed under each level with their chance.`}
            {v.model_choice === "expected_cost" && ` Your team routes by expected total cost: each task goes to the model with the lowest
              expected cost to finish it, counting what a failed attempt costs a person on your team (${usd(v.human_cost_usd ?? 1)} at level 5, a
              third as much for each level below, less for small requests) and the move to a stronger model after it. A model below a level can take its work when its chance there is at least
              ${Math.round((v.min_success_chance ?? 0.6) * 100)}%; those are listed under each level with their chance. Inside a task, failing
              tests, fix loops or a correction can move it up${v.max_escalations ? `, at most ${v.max_escalations} time${v.max_escalations === 1 ? "" : "s"} per task` : " (off: no escalations)"}.`}
          </Text>
          <Table.ScrollContainer minWidth={760}>
            <Table verticalSpacing={6} fz="sm">
              <Table.Thead><Table.Tr><Table.Th w={60}>Level</Table.Th><Table.Th w={240}>Means</Table.Th><Table.Th>Coding</Table.Th><Table.Th>General</Table.Th></Table.Tr></Table.Thead>
              <Table.Tbody>
                {v.levels.map((l) => (
                  <Table.Tr key={l.level}>
                    <Table.Td className="mono">{l.level}</Table.Td>
                    <Table.Td>{l.means}</Table.Td>
                    <Table.Td><Models list={l.models.coding} reach={l.in_reach?.coding} /></Table.Td>
                    <Table.Td><Models list={l.models.general} reach={l.in_reach?.general} /></Table.Td>
                  </Table.Tr>
                ))}
              </Table.Tbody>
            </Table>
          </Table.ScrollContainer>
        </Panel>
      )}
      {me && me.role !== "user" && me.role !== "finance" && <LevelOutcomesPanel api={api} admin={me.role === "admin"} />}
    </Stack>
  );
}
