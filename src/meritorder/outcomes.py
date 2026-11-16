"""Coding outcomes from traffic the gateway already sees (decision 59, docs/03-routing.md#coding-outcomes).

An agent's next request carries the results of the tools its last answer called: the test run, the build, the edit.
They say whether the answer worked, far more often than a person resending or raising a floor does. ``Tracker.update``
reads only what's new since the session's last turn (the answer's tool calls and their results), and keeps per task:

* edits, and the files edited (each path hashed to 8 hex characters, at most 20 kept);
* test runs (pytest, jest, vitest, go test, cargo test, npm test and the like, known by the command) and whether each
  passed or failed, from the runner's summary line or, where the harness shows it, the exit code; and whether the last
  one after the last edit passed;
* builds, compiles and type checks (tsc, mypy, cargo build, go build, make, ...) the same way;
* fix loops: an edit to a file right after a failed run, counted per file; three on one file is a loop;
* turns and prompt tokens to the end of the task;
* a correction in the person's next prompt ("that's wrong", "undo", "revert", ...), as the name of the rule that
  matched, never the text;
* for escalation inside a task (success.posterior, decision 62): failed runs and error results in a row, the files the
  task has touched, the size of its edits, and what kind of file each edit was to (test, docs, config or core, from the
  path's shape; only the kind is kept).

Nothing of a command, a path or an output is stored: only counts, flags and rule names, in the session's routing
state while the task runs and on the task's ``route_features`` row (the ``coding`` column) when something changes.

Formats: Anthropic messages (Claude Code: tool_use and tool_result blocks, ``is_error`` from a non-zero exit), Chat
Completions (pi and other OpenAI-format agents: tool_calls and tool messages), and the Responses API (Codex CLI: its
items arrive here as Chat Completions after formats.responses; function_call_output text with "Exit code: N", or JSON
with metadata.exit_code; apply_patch as a custom tool whose input names the files).
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from .formats.translate import person_text

TAIL = 4000  # characters of a tool result read: runners print their summary last
HEAD = 2000  # characters of a command read
MAX_FILES = 20
LOOP = 3  # edits to one file, each right after a failed run: a fix loop

EDIT_TOOLS = {"edit", "multiedit", "write", "notebookedit", "str_replace_editor", "str_replace_based_edit_tool", "apply_patch",
              "edit_file", "write_file", "create_file", "replace_in_file", "patch"}
SHELL_TOOLS = {"bash", "shell", "exec_command", "local_shell", "shell_command", "run_terminal_cmd", "run_command", "terminal",
               "container.exec", "execute_command"}
TEST_CMD = re.compile(r"\b(py\.?test|jest|vitest|mocha|go\s+test|cargo\s+(?:test|nextest)|(?:npm|pnpm|yarn|bun)\s+(?:run\s+)?test|"
                      r"unittest|rspec|phpunit|tox|nox|make\s+test|mvn\s+test|gradle\w*\s+test|dotnet\s+test|ctest)\b", re.I)
BUILD_CMD = re.compile(r"\b(tsc|mypy|pyright|cargo\s+(?:build|check|clippy)|go\s+(?:build|vet)|make|cmake|(?:npm|pnpm|yarn)\s+(?:run\s+)?"
                       r"(?:build|lint|typecheck|type-check)|gcc|g\+\+|clang|javac|mvn\s+(?:compile|package)|gradle\w*\s+build|"
                       r"dotnet\s+build|swift\s+build|ruff\s+check|eslint)\b", re.I)
# A runner's own verdict, read from the end of its output.
FAILED = re.compile(r"(\b[1-9]\d* failed\b|^FAILED\b|^FAIL\b|--- FAIL|test result: FAILED|\b[1-9]\d* (?:failing|errors?)\b|"
                    r"^ERROR\b|AssertionError|panicked at|BUILD FAILED|Traceback \(most recent call last\))", re.M)
PASSED = re.compile(r"(\b\d+ passed\b|^ok\s|^PASS\b|test result: ok\.|Tests:\s+\d+ passed|\b\d+ passing\b|^OK\b|BUILD SUCCESS)", re.M)
BUILD_ERROR = re.compile(r"(error TS\d+|error\[E\d+\]|^\S+:\d+:\d+: error|: error:|SyntaxError|IndentationError|cannot find symbol|"
                         r"undefined: |Found \d+ errors?|found \d+ errors?|\berror: could not compile|compilation failed)", re.M | re.I)
EXIT = re.compile(r"(?:exit(?:ed)?(?: with)? code|exit_code\"?)\s*[:=]?\s*(-?\d+)", re.I)
# The person's next prompt saying the last answer was wrong. Only the rule's name is kept.
CORRECTIONS = [
    ("wrong", re.compile(r"\b(that'?s|this is|it'?s|you'?re) (wrong|incorrect|not right)\b|\bwrong (file|approach|answer)\b", re.I)),
    ("undo", re.compile(r"\bundo\b|\bput it back\b|\bgo back to\b", re.I)),
    ("revert", re.compile(r"\brevert\b|\broll ?back\b", re.I)),
    ("broke", re.compile(r"\byou broke\b|\b(that|this|it) broke\b|\bbroke (it|the|everything)\b", re.I)),
    ("not working", re.compile(r"\b(still|it'?s|that'?s|this is) (not working|failing|broken)\b|\b(doesn'?t|didn'?t) work\b", re.I)),
    ("not asked", re.compile(r"\bnot what i (asked|wanted|meant)\b|\bi (didn'?t|did not) ask\b", re.I)),
]
PATCH_FILE = re.compile(r"^\*\*\* (?:Update|Add|Delete) File: (.+)$", re.M)
_TEST_PATH = re.compile(r"(^|/)(tests?|__tests__|spec)(/|$)|(^|/)test_[^/]*$|_test\.\w+$|\.(test|spec)\.\w+$", re.I)
_DOCS_PATH = re.compile(r"(^|/)docs?(/|$)|\.(md|rst|txt|adoc)$", re.I)
_CONFIG_PATH = re.compile(r"\.(ya?ml|json|toml|ini|cfg|conf|env|lock)$|(^|/)(Dockerfile|Makefile|\.github)(/|$)?", re.I)


def path_kind(path: str) -> str:
    """test, docs, config or core: what kind of file a path is (only the kind is kept)."""
    if _TEST_PATH.search(path):
        return "test"
    if _DOCS_PATH.search(path):
        return "docs"
    if _CONFIG_PATH.search(path):
        return "config"
    return "core"


def _hash(path: str) -> str:
    return hashlib.blake2b(path.strip().encode("utf-8", "replace"), digest_size=4).hexdigest()


def _text(content: Any) -> str:
    """A tool result's text, its end only (where runners print their summary)."""
    if isinstance(content, str):
        return content[-TAIL:]
    if isinstance(content, list):
        parts = [b.get("text") or "" if isinstance(b, dict) else str(b) for b in content]
        return "\n".join(p for p in parts if p)[-TAIL:]
    return "" if content is None else str(content)[-TAIL:]


def _args(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw[:1] == "{":
        try:
            v = json.loads(raw)
            return v if isinstance(v, dict) else {}
        except ValueError:
            return {}
    return {}


def _command(args: dict) -> str:
    c = args.get("command") or args.get("cmd") or args.get("script") or ""
    if isinstance(c, list):
        c = " ".join(str(x) for x in c)
    return str(c)[:HEAD]


def _files(name: str, args: dict) -> list[str]:
    for k in ("file_path", "path", "filename", "target_file", "notebook_path"):
        if isinstance(args.get(k), str):
            return [args[k]]
    patch = args.get("input") or args.get("patch") or ""
    return PATCH_FILE.findall(patch[:20000]) if isinstance(patch, str) else []


def calls_and_results(messages: list) -> tuple[dict[str, tuple[str, dict]], list[tuple[str, str, bool | None]]]:
    """The tool calls in these messages, by id, as (tool name, arguments), and the results, in order, as (call id, the
    end of its text, the harness's error flag where it sends one)."""
    calls: dict[str, tuple[str, dict]] = {}
    results: list[tuple[str, str, bool | None]] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role, content = m.get("role"), m.get("content")
        for tc in m.get("tool_calls") or []:  # Chat Completions (and the Responses API, translated)
            fn = tc.get("function") or {}
            calls[tc.get("id") or ""] = (str(fn.get("name") or "").lower(), _args(fn.get("arguments")))
        if role == "tool":
            results.append((m.get("tool_call_id") or "", _text(content), None))
        elif isinstance(content, list):  # Anthropic content blocks
            for b in content:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "tool_use":
                    calls[b.get("id") or ""] = (str(b.get("name") or "").lower(), b.get("input") if isinstance(b.get("input"), dict) else {})
                elif b.get("type") == "tool_result":
                    results.append((b.get("tool_use_id") or "", _text(b.get("content")), bool(b.get("is_error"))))
    return calls, results


def verdict(text: str, error: bool | None) -> bool | None:
    """Did the run pass (True), fail (False), or can't we tell (None)? The runner's summary first, then the exit code."""
    if FAILED.search(text):  # any failure counts, even next to passes ("1 failed, 4 passed")
        return False
    if PASSED.search(text):
        return True
    code = EXIT.findall(text)
    if code:
        return code[-1] == "0"
    if error is not None:
        return not error
    return None


def correction(body: dict) -> str | None:
    """The rule a person's newest prompt matched saying the last answer was wrong, if any (its first 1,000 characters,
    without what the harness added to it: translate.person_text)."""
    msgs = body.get("messages") or []
    last = msgs[-1] if msgs else None
    if not isinstance(last, dict) or last.get("role") != "user":
        return None
    text = person_text(last.get("content"))[:1000]
    return next((name for name, rule in CORRECTIONS if rule.search(text)), None)


def new_state() -> dict:
    return {"edits": 0, "tests": 0, "tests_failed": 0, "builds": 0, "builds_failed": 0, "last_test": None, "last_build": None,
            "red": False, "loops": {}, "fix_loop": 0, "turns": 0, "tokens": 0, "correction": None,
            "fails_in_row": 0, "errors_in_row": 0, "files": 0, "edit_chars": 0, "kinds": {}}


def _edit_size(args: dict) -> int:
    """Characters an edit writes: the new text of Edit and Write, a patch's added lines."""
    size = 0
    for k in ("new_string", "content", "new_str", "file_text", "code_edit"):
        if isinstance(args.get(k), str):
            size += len(args[k])
    for e in args.get("edits") or []:
        if isinstance(e, dict) and isinstance(e.get("new_string"), str):
            size += len(e["new_string"])
    patch = args.get("input") or args.get("patch")
    if isinstance(patch, str):
        size += sum(len(line) - 1 for line in patch[:20000].splitlines() if line.startswith("+"))
    return size


def update(state: dict | None, body: dict, since: int, prompt_tokens: int) -> tuple[dict, bool]:
    """Fold a task's next request into its state: what the last answer's tool calls did, read from the messages after
    the first ``since`` (the previous request's). Returns the state and whether a run's result changed it (worth
    writing to the task's row)."""
    s = dict(state or new_state())
    s["turns"] += 1
    s["tokens"] += max(int(prompt_tokens), 0)
    msgs = body.get("messages") or []
    calls, results = calls_and_results(msgs[since:] if 0 < since <= len(msgs) else msgs[-2:])
    changed = False
    loops = dict(s.get("loops") or {})
    kinds = dict(s.get("kinds") or {})
    for call_id, (name, args) in calls.items():
        if name in EDIT_TOOLS or name.endswith("apply_patch"):
            s["edits"] += 1
            s["last_test"] = s["last_build"] = None  # runs before this edit no longer say how the code stands
            s["edit_chars"] = s.get("edit_chars", 0) + _edit_size(args)
            for f in _files(name, args):
                h = _hash(f)
                kind = path_kind(f)
                kinds[kind] = kinds.get(kind, 0) + 1
                if h not in loops:
                    s["files"] = s.get("files", 0) + 1
                if s["red"]:
                    loops[h] = loops.get(h, 0) + 1
                elif h not in loops and len(loops) < MAX_FILES:
                    loops[h] = 0
    s["kinds"] = kinds
    for call_id, text, error in results:
        name, args = calls.get(call_id, ("", {}))
        if error is not None:
            s["errors_in_row"] = s.get("errors_in_row", 0) + 1 if error else 0
        if name not in SHELL_TOOLS and not name.endswith("exec_command") and not name.endswith("shell"):
            continue
        cmd = _command(args)
        kind = "test" if TEST_CMD.search(cmd) else "build" if BUILD_CMD.search(cmd) else None
        if kind is None:
            continue
        ok = verdict(text, error)
        if kind == "build" and ok is not False and BUILD_ERROR.search(text):
            ok = False
        if ok is None:
            continue
        s[f"{kind}s"] += 1
        s[f"{kind}s_failed"] += 0 if ok else 1
        s[f"last_{kind}"] = "pass" if ok else "fail"
        s["red"] = not ok
        s["fails_in_row"] = 0 if ok else s.get("fails_in_row", 0) + 1
        changed = True
    s["loops"] = dict(sorted(loops.items(), key=lambda kv: -kv[1])[:MAX_FILES])
    loop = max(s["loops"].values(), default=0)
    changed = changed or (loop >= LOOP and s["fix_loop"] < LOOP)
    s["fix_loop"] = loop
    return s, changed


def summary(state: dict) -> dict:
    """What goes on the task's route_features row: counts and flags, no file hashes."""
    return {k: v for k, v in state.items() if k not in ("loops", "red")}
