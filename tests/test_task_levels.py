"""Tools say a request is coding; the task's wording sets its level (decision 61). Coding agents send dozens of tools on
every request, so the tool count can't say whether the task is a typo or a redesign. These tests use request bodies
shaped as Claude Code and Codex CLI send them: their tool lists, Claude Code's system reminders (CLAUDE.md) and Codex's
AGENTS.md and environment context, none of which may set the level."""

import pytest
from conftest import auth

from meritorder.classifier import classify, matched_all
from meritorder.config import LevelRules
from meritorder.formats.responses import responses_to_chat
from meritorder.formats.translate import person_text
from meritorder.learned import features

# Claude Code's built-in tools (November 2026) and a few MCP servers': it sends all of them on every request.
CC_TOOL_NAMES = ["Task", "Bash", "Glob", "Grep", "LS", "ExitPlanMode", "Read", "Edit", "MultiEdit", "Write", "NotebookEdit",
                 "WebFetch", "TodoWrite", "WebSearch", "BashOutput", "KillShell", "SlashCommand", "Skill", "AskUserQuestion",
                 "ToolSearch", "Monitor", "Workflow", "SendMessage", "EnterWorktree", "ExitWorktree"]
CC_TOOLS = ([{"name": n, "description": f"The {n} tool.", "input_schema": {"type": "object", "properties": {}}} for n in CC_TOOL_NAMES]
            + [{"name": f"mcp__github__{n}", "description": n, "input_schema": {"type": "object"}}
               for n in ("create_issue", "get_pull_request", "list_commits", "search_code", "create_pull_request")]
            + [{"name": f"mcp__linear__{n}", "description": n, "input_schema": {"type": "object"}} for n in ("get_issue", "update_issue")])
# A CLAUDE.md full of words the rules look for: none of them is the task.
CLAUDE_MD = ("<system-reminder>\nAs you answer the user's questions, you can use the following context:\n# claudeMd\n"
             "Read docs/architecture.md first. Write tests for every change and run the linter. Security review: never log "
             "secrets. Performance matters on the hot path. Explain trade-offs in the PR. Migrations go in db/migrations.\n"
             "</system-reminder>\n")
TODO_REMINDER = "<system-reminder>\nThe TodoWrite tool hasn't been used recently. Design a plan if it helps.\n</system-reminder>"


def claude_code(prompt: str, *later: str) -> dict:
    """A Claude Code main-conversation request: system blocks, every tool, CLAUDE.md in a system reminder before the
    first prompt, and, for each later prompt, a tool loop before it."""
    msgs = [{"role": "user", "content": [{"type": "text", "text": CLAUDE_MD}, {"type": "text", "text": prompt}]}]
    for i, text in enumerate(later):
        msgs += [{"role": "assistant", "content": [{"type": "tool_use", "id": f"tu{i}", "name": "Bash", "input": {"command": "ls"}}]},
                 {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"tu{i}", "content": "app.py\ntests/"}]},
                 {"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
                 {"role": "user", "content": [{"type": "text", "text": text}, {"type": "text", "text": TODO_REMINDER}]}]
    return {"model": "auto", "max_tokens": 32000, "stream": False, "tools": CC_TOOLS,
            "system": [{"type": "text", "text": "You are Claude Code, Anthropic's official CLI for Claude."},
                       {"type": "text", "text": "You are an interactive CLI tool that helps users with software engineering tasks."}],
            "messages": msgs}


def mid_task(body: dict) -> dict:
    """The same task one step on: the last user message is only a tool result."""
    msgs = body["messages"] + [{"role": "assistant", "content": [{"type": "tool_use", "id": "t9", "name": "Read", "input": {}}]},
                               {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t9", "content": "def f(): ..."}]}]
    return {**body, "messages": msgs}


# Codex CLI's tools as codex-rs sends them to a custom provider, plus the MCP tools a typical setup adds.
CODEX_TOOLS = ([{"type": "function", "name": n, "strict": False, "description": n,
                 "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}}}
                for n in ("exec_command", "write_stdin", "update_plan", "view_image", "list_mcp_resources", "read_mcp_resource")]
               + [{"type": "custom", "name": "apply_patch", "description": "Edit files.", "format": {"type": "text"}}]
               + [{"type": "function", "name": f"mcp__github__{n}", "strict": False, "description": n, "parameters": {"type": "object"}}
                  for n in ("create_issue", "get_issue", "list_pull_requests", "search_code", "get_file_contents", "create_branch",
                            "push_files", "merge_pull_request", "list_commits", "add_comment")])
AGENTS_MD = ("# AGENTS.md instructions for /repo\n\n<INSTRUCTIONS>\nThis repo's architecture is in ARCHITECTURE.md. Optimize for "
             "readability. Fix the race condition checklist before merging. Write tests.\n</INSTRUCTIONS>")
ENV = "<environment_context>\n  <cwd>/repo</cwd>\n  <approval_policy>on-request</approval_policy>\n</environment_context>"


def codex(prompt: str) -> dict:
    def user(text):
        return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}
    return {"model": "auto", "stream": False, "store": False, "tools": CODEX_TOOLS, "tool_choice": "auto",
            "instructions": "You are Codex, based on GPT-6. You are running as a coding agent in the Codex CLI.",
            "input": [{"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "<permissions instructions>"}]},
                      user(AGENTS_MD), user(ENV), user(prompt)]}


CASES = [
    ("write unit tests for the date parser", 2, "simple task wording (write (unit )?tests): level 2"),
    ("Fix the typo in the README's install section", 2, "simple task wording (typo): level 2"),
    ("rename get_user to fetch_user in api/users.py", 2, "simple task wording (rename): level 2"),
    ("add docstrings to the public functions in utils.py", 2, "simple task wording (docstring): level 2"),
    ("explain what this module does", 2, "simple task wording (explain): level 2"),
    ("bump requests to 2.33", 2, "simple task wording (bump): level 2"),
    ("add a --dry-run flag to the deploy command", 3, None),
    ("the export endpoint returns 500 when the list is empty, fix it", 3, None),
    ("fix the race condition in the job scheduler", 4, "hard wording (race condition): +1"),
    ("move the session store from Redis to Postgres: write the migration", 4, "hard wording (migrat): +1"),
    ("the checkout page is slow, find the performance problem", 4, "hard wording (performance): +1"),
    ("write tests for the deadlock fix", 4, "hard wording (deadlock): +1"),  # harder wording wins over simple
    ("design a plugin system for the importers", 5, "hardest wording (design (a|the)): level 5"),
    ("propose an architecture for multi-region failover", 5, "hardest wording (architect): level 5"),
]


@pytest.mark.parametrize("prompt, level, why", CASES)
def test_claude_code_tasks_get_the_level_of_their_wording(prompt, level, why):
    floor = classify(claude_code(prompt), {"x-claude-code-request-class": "main"})
    assert str(floor) == f"coding:{level}"
    assert floor.reason.startswith(f"coding: {len(CC_TOOLS)} tools")
    assert floor.reason == f"coding: {len(CC_TOOLS)} tools" + (f"; {why}" if why else "")


@pytest.mark.parametrize("prompt, level, why", CASES)
def test_codex_tasks_get_the_level_of_their_wording(prompt, level, why):
    chat = responses_to_chat(codex(prompt))
    floor = classify(chat, {})
    assert str(floor) == f"coding:{level}"
    assert floor.reason == f"coding: {len(chat['tools'])} tools" + (f"; {why}" if why else "")


def test_the_harness_own_text_never_sets_the_level():
    """CLAUDE.md (in a system reminder), a todo reminder, AGENTS.md and the environment context are the same on every
    task: their words ("architecture", "security", "write tests") are left out, and a message of nothing else is passed
    over, so the window of two prompts reaches what the person wrote."""
    assert person_text([{"type": "text", "text": CLAUDE_MD}]).strip() == ""
    assert person_text(AGENTS_MD).strip() == "" and person_text(ENV).strip() == ""
    assert person_text("plain words") == "plain words"
    assert person_text([{"type": "text", "text": TODO_REMINDER}, {"type": "text", "text": "go on"}]) == "go on"
    assert str(classify(claude_code("add a --dry-run flag"), {})) == "coding:3"
    chat = responses_to_chat(codex("add a --dry-run flag"))
    assert str(classify(chat, {})) == "coding:3"
    # A real prompt still counts when a reminder shares its message.
    assert str(classify(claude_code("go", "now fix the race condition in it"), {})) == "coding:4"


def test_the_level_holds_mid_task_from_the_prompt_that_started_it():
    """Mid-task requests end in tool results: the wording is still the prompt's, so the classifier says the same thing at
    every step (the router keeps the task's model and level anyway until the next prompt)."""
    start = claude_code("write unit tests for the date parser")
    step = mid_task(start)
    assert classify(step, {}).reason == classify(start, {}).reason
    assert str(classify(mid_task(mid_task(step)), {})) == "coding:2"


def test_a_new_prompt_is_a_new_task_and_reads_the_last_two_prompts():
    # The second prompt is read with the first: "go on" after a design task is still the design task.
    assert str(classify(claude_code("design a plugin system for the importers", "go on"), {})) == "coding:5"
    # Three prompts on, the first is out of the window.
    assert str(classify(claude_code("design a plugin system", "add a flag", "write tests for the flag"), {})) == "coding:2"


def test_a_prompt_that_corrects_the_last_answer_goes_a_level_up():
    floor = classify(claude_code("write unit tests for the date parser", "that's wrong, the tests still fail"), {})
    assert str(floor) == "coding:3"
    assert floor.reason.endswith("simple task wording (write (unit )?tests): level 2; the new prompt corrects the last answer (wrong): +1")
    # Only the newest prompt: a correction further back doesn't keep raising later tasks.
    assert "corrects" not in classify(claude_code("that's wrong", "write unit tests"), {}).reason
    # Mid-task (a tool result last) there's no new prompt to read.
    assert "corrects" not in classify(mid_task(claude_code("that's wrong")), {}).reason
    off = LevelRules(correction_raise=0)
    assert "corrects" not in classify(claude_code("fix the typo", "that's wrong"), {}, rules=off).reason


def test_agent_by_tools_restores_the_rule_before_decision_61():
    old = LevelRules(agent_by_tools=True)
    n = len(CC_TOOLS)
    floor = classify(claude_code("write unit tests for the date parser"), {}, rules=old)
    assert str(floor) == "coding:4" and floor.reason == f"agentic coding: {n} tools (at least 5)"  # simple wording doesn't lower it
    floor = classify(claude_code("fix the race condition"), {}, rules=old)
    assert str(floor) == "coding:5" and floor.reason == f"agentic coding: {n} tools (at least 5); hard wording (race condition): +1"
    assert str(classify(claude_code("design a plugin system"), {}, rules=old)) == "coding:5"
    # Fewer tools than agent_tools: the wording decides, as by default.
    few = {"messages": [{"role": "user", "content": "write tests"}], "tools": CC_TOOLS[:2]}
    assert str(classify(few, {}, rules=old)) == "coding:2"


def test_each_sign_of_hard_work_counts_up_to_hard_signals():
    floor = classify(claude_code("the job queue deadlocks under load, fix it"), {})
    assert str(floor) == "coding:5" and floor.reason.endswith("hard wording (deadlock, under load): +2")
    one = LevelRules(hard_signals=1)
    floor = classify(claude_code("the job queue deadlocks under load, fix it"), {}, rules=one)
    assert str(floor) == "coding:4" and floor.reason.endswith("hard wording (deadlock): +1")
    # Effort words aren't a kind of task: "a quick fix" for a symptom of hard work is still hard work.
    assert str(classify(claude_code("quick fix: checkout double-charges on retry"), {})) == "coding:4"
    assert matched_all([r"\bdeadlock", r"\bunder load", r"\bkotlin"], "deadlocks under load") == ["deadlock", "under load"]
    assert matched_all([r"\bkotlin"], "nothing here") == [] and matched_all([], "x") == []


def test_the_wording_levels_are_configurable():
    rules = LevelRules(simple=1, simple_words=[r"\bkotlin\b"], hardest=4, hardest_words=[r"\bsharding\b"], hard_raise=2,
                       hard_words=[r"\bgpu\b"])
    assert str(classify(claude_code("port it to kotlin"), {}, rules=rules)) == "coding:1"
    assert str(classify(claude_code("write tests"), {}, rules=rules)) == "coding:3"  # not a simple word now
    assert str(classify(claude_code("add sharding"), {}, rules=rules)) == "coding:4"
    assert str(classify(claude_code("use the gpu"), {}, rules=rules)) == "coding:5"
    # A simple level above coding's, or a hardest level below it, doesn't move it the wrong way.
    odd = LevelRules(simple=4, hardest=2)
    assert str(classify(claude_code("write tests"), {}, rules=odd)) == "coding:3"
    assert str(classify(claude_code("design a plugin system"), {}, rules=odd)) == "coding:3"
    # Empty lists switch a rule off.
    assert str(classify(claude_code("write tests"), {}, rules=LevelRules(simple_words=[]))) == "coding:3"


def test_code_like_text_without_tools_uses_the_wording_too():
    def floor(text):
        return str(classify({"messages": [{"role": "user", "content": text}]}, {}))
    assert floor("add a docstring to def parse(s):") == "coding:2"
    assert floor("why does app.py crash?") == "coding:3"
    assert floor("design a cache for app.py") == "coding:5"


def test_claude_code_side_requests_and_compaction_stay_capped(gateway):
    """Titles and summaries go to level 1 and compaction summaries to 3 at most, as before. A cap only lowers: compacting
    a level-2 task stays at 2."""
    for cls, prompt, level in [("auxiliary", "design a plugin system", 1), ("compaction", "design a plugin system", 3),
                               ("compaction", "write unit tests", 2), ("subagent", "design a plugin system", 5),
                               ("main", "write unit tests", 2)]:
        floor, _ = gateway.router.level(claude_code(prompt), {"x-claude-code-request-class": cls}, 5_000)
        assert floor.level == level, (cls, prompt)


def test_the_learned_router_sees_the_task_rules_and_the_tool_count():
    rules = classify(claude_code("write unit tests"), {})
    tags = features(rules, claude_code("write unit tests"), {}, 5_000, "anthropic", b"k" * 16).tags
    assert {"rules:coding:2", "rule:simple", "rule:tools", "rule:agent", "tools:20-49"} <= set(tags)
    hard = classify(claude_code("go", "that's wrong, fix the race condition"), {})
    tags = features(hard, {}, {}, 5_000, "anthropic", b"k" * 16).tags
    assert {"rule:hard", "rule:correction"} <= set(tags) and "rule:agent" not in tags  # no tools in that body
    assert "rule:hardest" in features(classify(claude_code("design a plugin system"), {}), {}, {}, 1, "anthropic", b"k" * 16).tags
    few = features(rules, {"tools": CC_TOOLS}, {}, 1, "anthropic", b"k" * 16, agent_tools=100).tags
    assert "rule:agent" not in few


async def test_a_claude_code_session_keeps_its_task_level_until_the_next_prompt(client, gateway):
    headers = {**auth(), "x-claude-code-session-id": "cc-task", "x-claude-code-request-class": "main"}
    first = claude_code("write unit tests for the date parser")
    r = await client.post("/v1/messages", headers=headers, json=first)
    assert r.status_code == 200, r.text
    assert r.headers["x-meritorder-floor"] == "coding:2"
    # The agent loop: tool results only, the same task, the same level.
    step = mid_task(first)
    r = await client.post("/v1/messages", headers=headers, json=step)
    assert r.headers["x-meritorder-floor"] == "coding:2"
    # A new prompt is a task boundary: its wording sets the next level.
    nxt = {**step, "messages": step["messages"] + [{"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
                                                     {"role": "user", "content": "now design a plugin system for the importers"}]}
    r = await client.post("/v1/messages", headers=headers, json=nxt)
    assert r.headers["x-meritorder-floor"] == "coding:5"
    r = await client.post("/v1/messages", headers=headers, json=mid_task(nxt))
    assert r.headers["x-meritorder-floor"] == "coding:5"


async def test_a_codex_request_through_the_responses_api(client):
    r = await client.post("/v1/responses", headers=auth(), json=codex("write unit tests for the date parser"))
    assert r.status_code == 200, r.text
    assert r.headers["x-meritorder-floor"] == "coding:2"
    r = await client.post("/v1/responses", headers=auth(), json=codex("fix the race condition in the job scheduler"))
    assert r.headers["x-meritorder-floor"] == "coding:4"
