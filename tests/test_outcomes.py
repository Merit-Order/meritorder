"""Coding outcomes from the tool results the gateway already sees (decision 59): tests and builds after an edit, fix
loops, turns and tokens, and a correction in the next prompt; in Claude Code's, pi's (Chat Completions) and Codex CLI's
formats; only counts and flags stored; and cheap per request."""

import json
import time

import pytest
from test_learned_router import chat, features_rows, shadow_gw  # noqa: F401 (a fixture)

from meritorder import outcomes as oc
from meritorder.formats.responses import responses_to_chat
from meritorder.learned_jobs import failed, signals

PYTEST_FAIL = "....F\n=================== FAILURES ===================\n_ test_dates _\nE  assert 1 == 2\n========= 1 failed, 4 passed in 0.12s ========="
PYTEST_PASS = ".....\n============ 5 passed in 0.10s ============"


def claude_turn(calls: list[tuple[str, str, dict]], results: list[tuple[str, str, bool]]) -> list[dict]:
    """Claude Code: the answer's tool_use blocks, then a user message of tool_result blocks."""
    return [{"role": "assistant", "content": [{"type": "tool_use", "id": i, "name": n, "input": a} for i, n, a in calls]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": i, "content": [{"type": "text", "text": t}],
                                          "is_error": e} for i, t, e in results]}]


def chat_turn(calls: list[tuple[str, str, dict]], results: list[tuple[str, str]]) -> list[dict]:
    """pi and other Chat Completions agents: tool_calls, then tool messages."""
    return [{"role": "assistant", "content": None,
             "tool_calls": [{"id": i, "type": "function", "function": {"name": n, "arguments": json.dumps(a)}} for i, n, a in calls]},
            *({"role": "tool", "tool_call_id": i, "content": t} for i, t in results)]


def run(turns: list[list[dict]], first=None) -> dict:
    """A task's requests, one per turn, each carrying the history so far."""
    msgs = first or [{"role": "user", "content": "fix the failing date test"}]
    state = None
    for t in turns:
        since = len(msgs)
        msgs = msgs + t
        state, _ = oc.update(state, {"messages": msgs}, since, 1_000)
    return state


def test_claude_code_tests_after_an_edit_and_a_fix_loop():
    edit = ("e1", "Edit", {"file_path": "/repo/app/dates.py", "old_string": "a", "new_string": "b"})
    test = ("t1", "Bash", {"command": "python -m pytest tests/test_dates.py -q"})
    s = run([claude_turn([edit, test], [("e1", "ok", False), ("t1", PYTEST_FAIL, True)])] * 4)
    assert s["edits"] == 4 and s["tests"] == 4 and s["tests_failed"] == 4 and s["last_test"] == "fail"
    assert s["fix_loop"] == 3  # three edits to the same file, each right after a failed run
    assert s["turns"] == 4 and s["tokens"] == 4_000
    done = run([claude_turn([edit, test], [("e1", "ok", False), ("t1", PYTEST_FAIL, True)]),
                claude_turn([edit, test], [("e1", "ok", False), ("t1", PYTEST_PASS, False)])])
    assert done["last_test"] == "pass" and done["tests_failed"] == 1
    summary = oc.summary(s)
    assert "loops" not in summary and "dates" not in json.dumps(summary)  # no paths, hashed or not


def test_an_edit_after_the_last_run_means_nothing_is_known_about_the_code():
    edit = ("e", "Write", {"file_path": "a.py"})
    test = ("t", "Bash", {"command": "pytest"})
    s = run([claude_turn([test], [("t", PYTEST_PASS, False)]), claude_turn([edit], [("e", "ok", False)])])
    assert s["last_test"] is None and s["tests"] == 1


def test_pi_chat_completions_builds_and_exit_codes():
    edit = ("c1", "edit", {"path": "src/main.ts", "oldText": "x", "newText": "y"})
    build = ("c2", "bash", {"command": "npx tsc --noEmit"})
    s = run([chat_turn([edit, build], [("c1", "ok"), ("c2", "src/main.ts(3,7): error TS2322: Type 'string' is not assignable")])])
    assert s["builds"] == 1 and s["last_build"] == "fail"
    s = run([chat_turn([build], [("c2", "Command exited with code 0")])])
    assert s["last_build"] == "pass"
    jest = ("c3", "bash", {"command": "npm test"})
    s = run([chat_turn([jest], [("c3", "Tests:       2 failed, 10 passed, 12 total")])])
    assert s["last_test"] == "fail"


def test_codex_cli_responses_items_go_test_cargo_and_apply_patch():
    patch = "*** Begin Patch\n*** Update File: pkg/dates.go\n@@\n-a\n+b\n*** End Patch"
    items = [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "make the tests pass"}]},
             {"type": "custom_tool_call", "call_id": "p1", "name": "apply_patch", "input": patch},
             {"type": "custom_tool_call_output", "call_id": "p1", "output": "Success. Updated the following files:\nM pkg/dates.go"},
             {"type": "function_call", "call_id": "s1", "name": "shell", "arguments": json.dumps({"command": ["bash", "-lc", "go test ./..."]})},
             {"type": "function_call_output", "call_id": "s1",
              "output": json.dumps({"output": "--- FAIL: TestParse (0.00s)\nFAIL\tpkg\t0.01s", "metadata": {"exit_code": 1}})}]
    body = responses_to_chat({"model": "auto", "input": items})
    s, changed = oc.update(None, body, 1, 500)
    assert changed and s["edits"] == 1 and s["last_test"] == "fail"
    cargo = [{"type": "function_call", "call_id": "s2", "name": "exec_command", "arguments": json.dumps({"cmd": "cargo test"})},
             {"type": "function_call_output", "call_id": "s2", "output": "test result: ok. 12 passed; 0 failed\nExit code: 0"}]
    s, _ = oc.update(s, responses_to_chat({"model": "auto", "input": items + cargo}), 1 + 4, 500)
    assert s["last_test"] == "pass" and s["tests"] == 2


def test_other_commands_and_unknown_results_say_nothing():
    grep = ("g", "Bash", {"command": "grep -rn foo src"})
    test = ("t", "Bash", {"command": "pytest"})
    s, changed = oc.update(None, {"messages": [{"role": "user", "content": "x"},
                                               *claude_turn([grep, test], [("g", "", True), ("t", "collected 0 items", False)])]}, 1, 0)
    assert s["tests"] == 1 and s["last_test"] == "pass"  # pytest without a summary: the harness's error flag decides
    s, changed = oc.update(None, {"messages": [{"role": "user", "content": "x"}, *chat_turn([test], [("t", "running...")])]}, 1, 0)
    assert s["tests"] == 0 and not changed  # nothing to go on


@pytest.mark.parametrize(("text", "rule"), [("that's wrong, the date is UTC", "wrong"), ("undo that please", "undo"),
                                            ("revert the last change", "revert"), ("you broke the build", "broke"),
                                            ("it's still failing", "not working"), ("not what I asked for", "not asked"),
                                            ("great, now add time zones", None)])
def test_corrections_in_the_next_prompt_are_kept_as_a_rule_name(text, rule):
    assert oc.correction({"messages": [{"role": "user", "content": [{"type": "text", "text": text}]}]}) == rule


def test_the_signals_and_their_weights():
    base = {"request_id": "r", "status": 200}
    assert failed(signals({**base, "coding": {"correction": "wrong"}}, set())) == pytest.approx(0.7)
    assert failed(signals({**base, "coding": {"last_test": "fail"}}, set())) == pytest.approx(0.5)
    green = {"last_test": "pass", "last_build": "fail", "fix_loop": 4}
    assert failed(signals({**base, "coding": green}, set())) == 0  # passing tests at the end outweigh the way there
    assert failed(signals({**base, "coding": {"last_build": "fail", "fix_loop": 3}}, set())) == pytest.approx(1 - 0.6 * 0.7)


def test_reading_a_long_agent_turn_is_cheap():
    """docs/07's load numbers: a gateway's own work per request is a few milliseconds. This adds tens of microseconds."""
    edit = ("e", "Edit", {"file_path": "a.py", "old_string": "x" * 5_000, "new_string": "y" * 5_000})
    test = ("t", "Bash", {"command": "pytest -q"})
    history = [{"role": "user", "content": "x" * 200_000}] + claude_turn([edit], [("e", "ok", False)]) * 50
    turn = claude_turn([edit, test], [("e", "ok", False), ("t", ("log line\n" * 5_000) + PYTEST_FAIL, True)])
    body = {"messages": history + turn}
    since = len(history)
    n = 2_000
    t0 = time.perf_counter()
    for _ in range(n):
        oc.update(None, body, since, 1)
    per = (time.perf_counter() - t0) / n
    assert per < 0.001  # well under a millisecond (about 20 microseconds on a laptop)


async def test_the_gateway_records_coding_outcomes_on_the_tasks_row(shadow_gw):  # noqa: F811
    gw, client = shadow_gw
    sid = {"x-session-id": "code-1"}
    msgs = [{"role": "user", "content": "fix the failing date test in app.py"}]
    assert (await chat(client, "", sid, messages=msgs)).status_code == 200
    edit = ("c1", "edit", {"path": "app.py"})
    test = ("c2", "bash", {"command": "pytest -q"})
    msgs = msgs + chat_turn([edit, test], [("c1", "ok"), ("c2", PYTEST_FAIL)])
    assert (await chat(client, "", sid, messages=msgs)).status_code == 200
    rows = await features_rows(gw)
    assert len(rows) == 1 and rows[0]["coding"]["last_test"] == "fail" and rows[0]["coding"]["turns"] == 1
    msgs = msgs + [{"role": "assistant", "content": "done"}, {"role": "user", "content": "that's wrong, revert it"}]
    assert (await chat(client, "", sid, messages=msgs)).status_code == 200
    rows = await features_rows(gw)
    assert rows[0]["coding"]["correction"] == "wrong" and rows[0]["coding"]["tokens"] > 0
    assert "pytest" not in json.dumps(rows[0]["coding"]) and "app.py" not in json.dumps(rows[0]["coding"])
    assert "meritorder_outcome_seconds_bucket" in (await client.get("/metrics")).text
