"""A real codebase: the model makes one change to this repository as it stood before the change was merged, and the
change's own tests grade it.

Each task is one commit from Merit Order's history (6,000 to 17,000 lines of Python, its tests, config, and docs). The
model gets the repository as it was before the commit and an issue describing what to build: the behaviour wanted
and the names the tests call, never how the commit did it. It works through file tools and runs pytest in the
sandbox. Grading runs the tests as they were after the commit (the model's own edits to tests/ don't count):

    fail_to_pass   the tests the commit added or changed: they fail before it and pass after it
    pass_to_pass   every other test that passed before and after: the change mustn't break them

Both lists come from running the tests in the same sandbox image, before and after (scripts/build_codebase_suite.py).
The tasks were built while the repository was private and newer than every model's training data. It's public since
release 0.0.1, so a model trained after that may have seen these changes: a strong result from a newer model may be
memory, not skill (docs/11-evals.md). Nothing here depends on the repository being unseen; the sandbox has no network,
so a model can't fetch the answer during a run.

fixtures/codebase/ holds the data: blobs.json (each file's text once, by hash: the snapshots share most files) and
one <task>.json per task (paths to hashes for the repository before, the tests after, and the commit's own change to
the code, kept as the reference solution).
"""

from __future__ import annotations

import base64
import functools
import json
import os
import re
import secrets
import threading
import zlib
from pathlib import Path

from . import Env
from . import patch as patching
from .repo import _clip, _err, _tool, norm_path

FIXTURE = Path(__file__).with_name("fixtures") / "codebase"
REQUIREMENTS = FIXTURE / "requirements.txt"  # the repository's locked dependencies, with hashes (uv export)
# A whole test run needs about 200 MB and 35 seconds on one CPU. Docker Desktop's VM is often small (4 CPUs and 8 GB,
# shared with everything else running), so at most PARALLEL codebase runs share it, however many tasks run at once:
# a model's turns mostly wait on the model, not on the tests.
SANDBOX = {"requirements": REQUIREMENTS, "memory": "1g", "cpus": "1", "pids": 256, "tmp": "256m", "tmp_exec": True}
PARALLEL = max(1, int(os.environ.get("MERITORDER_EVAL_SANDBOXES") or 2))
_RUNS = threading.BoundedSemaphore(PARALLEL)
READ_LINES = 400        # lines one read_file returns
MAX_RESULT = 16_000     # characters of one tool result (the runner cuts at 20,000)
MAX_WRITE = 300_000     # characters in one written file (the largest source file is about 70,000)
MAX_FILES = 2_000
MAX_MATCHES = 80
TEST_TIMEOUT = 600.0    # the whole suite takes about 35 seconds in the sandbox; a slow machine gets ten minutes
SHELL_TIMEOUT = 60.0
_TARGET = re.compile(r"^[A-Za-z0-9_./-]+(::[^\s]+)?$")

# The tests run with this repository's own pytest settings (pyproject.toml), with src/ importable and nothing
# written outside /tmp. Output is kept to its last part: that's where the failures and the summary are.
RUN_TESTS = 'python -m pytest -q -p no:cacheprovider -o pythonpath=src --tb=short --color=no "$@" > /tmp/out.txt 2>&1; ' \
            'code=$?; tail -c 14000 /tmp/out.txt; exit $code'
# A command in a throwaway, read-only copy of the repository (writes go nowhere but /tmp). Its output, cut to the end.
SHELL = '(eval "$1") > /tmp/out.txt 2>&1; code=$?; tail -c 14000 /tmp/out.txt; exit $code'

# Runs the grading tests inside the sandbox and prints one line: <nonce> and {"tests": {nodeid: outcome}, ...},
# compressed (a thousand test ids would pass the sandbox's output limit). pytest's own output goes to a file, so
# nothing the tests print can be mistaken for the result. A test file that can't be imported (it needs a name the
# change adds) loses only its own tests, not the run.
GRADER = '''\
import base64, json, os, sys, zlib
nonce = sys.argv[1]
sys.path[:0] = [os.path.join(os.getcwd(), "src"), os.getcwd()]
import pytest

class Collect:
    def __init__(self):
        self.tests, self.errors = {}, []
    def pytest_runtest_logreport(self, report):
        if report.when == "call" and hasattr(report, "wasxfail"):
            outcome = "xfailed" if report.skipped else "xpassed"
        elif report.failed:
            outcome = "failed"
        elif report.skipped:
            outcome = "skipped"
        elif report.when == "call":
            outcome = "passed"
        else:
            return
        if self.tests.get(report.nodeid) != "failed":
            self.tests[report.nodeid] = outcome
        if report.failed and len(self.errors) < 40:
            self.errors.append([report.nodeid, str(report.longrepr)[-600:]])
    def pytest_collectreport(self, report):
        if report.failed and len(self.errors) < 40:
            self.errors.append([report.nodeid or "collection", str(report.longrepr)[-600:]])

c = Collect()
log = open("/tmp/pytest.log", "w")
out, err = os.dup(1), os.dup(2)
os.dup2(log.fileno(), 1)
os.dup2(log.fileno(), 2)
try:
    code = int(pytest.main(["-q", "-p", "no:cacheprovider", "--tb=short", "--color=no", "--continue-on-collection-errors",
                            *(sys.argv[2:] or ["tests"])], plugins=[c]))
except BaseException as e:
    code = -1
    c.errors.append(["pytest", type(e).__name__ + ": " + str(e)])
finally:
    sys.stdout.flush()
    sys.stderr.flush()
    os.dup2(out, 1)
    os.dup2(err, 2)
result = json.dumps({"exit": code, "tests": c.tests, "errors": c.errors}).encode()
print(nonce + base64.b64encode(zlib.compress(result, 9)).decode())
'''


@functools.lru_cache(maxsize=1)
def _blobs() -> dict[str, str]:
    return json.loads((FIXTURE / "blobs.json").read_text(encoding="utf-8"))


def task_ids() -> list[str]:
    return sorted(p.stem for p in FIXTURE.glob("*.json") if p.name != "blobs.json")


@functools.lru_cache(maxsize=64)
def load_task(task: str) -> dict:
    """One task's data, with file texts filled in: files (before), tests (after), reference (the commit's change:
    path -> new text, or None for a deleted file), fail_to_pass, pass_to_pass, commit, base."""
    if not re.fullmatch(r"[a-z0-9_-]+", str(task)) or not (FIXTURE / f"{task}.json").is_file():
        raise ValueError(f"unknown codebase task {task!r}")
    m = json.loads((FIXTURE / f"{task}.json").read_text(encoding="utf-8"))
    blobs = _blobs()
    return {**m, "files": {p: blobs[h] for p, h in m["files"].items()}, "tests": {p: blobs[h] for p, h in m["tests"].items()},
            "reference": {p: (blobs[h] if h else None) for p, h in m["reference"].items()}}


def grading_files(files: dict[str, str], tests: dict[str, str]) -> dict[str, str]:
    """The repository to grade: the edited code, with tests/ exactly as the commit left it."""
    out = {p: v for p, v in files.items() if not p.startswith("tests/") and p != "_grade.py"}
    out.update(tests)
    out["_grade.py"] = GRADER
    return out


def run_grader(sandbox, files: dict[str, str], timeout: float = TEST_TIMEOUT, only: list[str] | None = None) -> dict:
    """Run the tests (every one, or ``only`` these ids) once in ``sandbox``:
    {"tests": {nodeid: outcome}, "errors": [[nodeid, text]], "error": str | None}."""
    nonce = secrets.token_hex(8)
    with _RUNS:
        r = sandbox.run(files, ["python", "-I", "-B", "_grade.py", nonce, *(only or [])], timeout)
    if r.timed_out:
        return {"tests": {}, "errors": [], "error": f"the tests timed out after {timeout:.0f}s"}
    line = next((ln for ln in reversed(r.stdout.splitlines()) if ln.startswith(nonce)), None)
    if line is None:
        return {"tests": {}, "errors": [], "error": f"the test run crashed (exit {r.exit_code}): {(r.stderr or r.stdout)[-500:]}"}
    try:
        return {**json.loads(zlib.decompress(base64.b64decode(line[len(nonce):]))), "error": None}
    except (ValueError, zlib.error) as e:
        return {"tests": {}, "errors": [], "error": f"the test run's result couldn't be read: {e}"}


class CodebaseEnv(Env):
    """Make a change to a real codebase (Merit Order itself, as it was before the change), with file tools and
    pytest. Graded by the change's own tests, and by every test that already passed."""

    grading = "The commit's own tests, and every test that passed before it, run on the edited repository in a sandbox"
    needs_sandbox = True
    sandbox_spec = SANDBOX

    tools = [
        _tool("list_files", "List the repository's files, as paths relative to the repository root.",
              {"directory": {"type": "string", "description": "Only list files under this directory (default: all files)"}}, []),
        _tool("read_file", f"Read a file, with line numbers. Returns up to {READ_LINES} lines at a time; give start_line to read on.",
              {"path": {"type": "string", "description": "Path relative to the repository root, e.g. src/meritorder/router.py"},
               "start_line": {"type": "integer", "description": "First line to return (default 1)"},
               "end_line": {"type": "integer", "description": f"Last line to return (default: start_line + {READ_LINES - 1})"}}, ["path"]),
        _tool("search", "Find the lines that contain a piece of text. Returns path:line: text for each match.",
              {"text": {"type": "string", "description": "Text to look for (case-insensitive), or a Python regular expression when regex is true"},
               "directory": {"type": "string", "description": "Only search files under this directory (default: everywhere)"},
               "regex": {"type": "boolean", "description": "Treat text as a regular expression (default false)"}}, ["text"]),
        _tool("edit_file", "Replace one exact piece of a file with new text. old_text must appear exactly once in the file "
                           "(unless replace_all is true), matching it character for character, without the line numbers read_file shows.",
              {"path": {"type": "string", "description": "Path relative to the repository root"},
               "old_text": {"type": "string", "description": "The exact text to replace"},
               "new_text": {"type": "string", "description": "What to put in its place"},
               "replace_all": {"type": "boolean", "description": "Replace every occurrence (default false)"}}, ["path", "old_text", "new_text"]),
        _tool("write_file", "Create a file or replace its whole content.",
              {"path": {"type": "string", "description": "Path relative to the repository root"},
               "content": {"type": "string", "description": "The complete new file content"}}, ["path", "content"]),
        _tool("delete_file", "Delete a file.", {"path": {"type": "string", "description": "Path relative to the repository root"}}, ["path"]),
        _tool("apply_patch", "Change files with a patch: '*** Begin Patch', then sections '*** Update File: <path>' (hunks that "
                             "start with '@@', optionally followed by a line to find first, then lines starting with ' ' for "
                             "context, '-' to remove and '+' to add), '*** Add File: <path>' (every line starting with '+') and "
                             "'*** Delete File: <path>', then '*** End Patch'. All of it applies, or none of it.",
              {"patch": {"type": "string", "description": "The patch text"}}, ["patch"]),
        _tool("shell", f"Run a shell command (sh) in a copy of the repository, for looking around or trying things: grep -rn, "
                       f"find, sed -n, python -c, and so on. The copy is read-only and thrown away afterwards, so change files "
                       f"with the editing tools. No network; {SHELL_TIMEOUT:.0f} s limit; returns the end of the output.",
              {"command": {"type": "string", "description": "The command, e.g. grep -rn 'def route' src/"}}, ["command"]),
        _tool("run_tests", "Run the repository's tests with pytest and return the end of the output. With no target it runs "
                           "everything in tests/ (about a minute); give test files or test ids to run fewer.",
              {"target": {"type": "string", "description": "Space-separated test files or ids, e.g. tests/test_router.py "
                                                           "or tests/test_router.py::test_cheapest_wins (default: all of tests/)"},
               "keyword": {"type": "string", "description": "Only tests whose names match this pytest -k expression"}}, []),
    ]

    def __init__(self, sandbox=None, task: str = "", **args):
        super().__init__(sandbox, task=task, **args)
        data = load_task(task)
        self.files: dict[str, str] = dict(data["files"])
        self.original = dict(data["files"])
        self.tests: dict[str, str] = data["tests"]
        self.fail_to_pass: list[str] = data["fail_to_pass"]
        self.pass_to_pass: list[str] = data["pass_to_pass"]

    @staticmethod
    def graded_by(task: str = "", **_: object) -> str:
        d = load_task(task)
        return (f"{len(d['fail_to_pass'])} new or changed tests and {len(d['pass_to_pass'])} existing ones, "
                "on the edited repository in a sandbox")

    @staticmethod
    def fixture_files(task: str = "", **_: object) -> tuple[dict[str, str], dict[str, str]]:
        """For the console's Tasks tab: what the commit changed, as the model gets it, and what grades it. The rest
        of the repository (a few hundred files) is listed, not shown."""
        d = load_task(task)
        before, tests_before = d["files"], {p: v for p, v in d["files"].items() if p.startswith("tests/")}
        visible = {"repository/FILES": "\n".join(sorted(before))}
        visible.update({f"repository/{p}": before[p] for p in d["reference"] if p in before})
        grader = {f"tests after the commit/{p}": v for p, v in d["tests"].items() if tests_before.get(p) != v}
        grader.update({f"reference solution/{p}": (v if v is not None else "(deleted)") for p, v in d["reference"].items()})
        grader["fail_to_pass"] = "\n".join(d["fail_to_pass"])
        grader["pass_to_pass"] = "\n".join(d["pass_to_pass"])
        grader["commit"] = f"{d['commit']}\n{d['subject']}\n\nbefore it: {d['base']}"
        return visible, grader

    def call(self, name: str, arguments: dict) -> str:
        return getattr(self, f"_t_{name}")(arguments)

    def changed(self) -> list[str]:
        paths = set(self.files) | set(self.original)
        return sorted(p for p in paths if self.files.get(p) != self.original.get(p))

    # ----------------------------------------------------------------- tools
    def _under(self, directory: object) -> tuple[list[str] | None, str | None]:
        if directory in (None, "", ".", "/"):
            return sorted(self.files), None
        d = norm_path(directory)
        if d is None:
            return None, f"not a directory in the repository: {directory!r}"
        paths = sorted(p for p in self.files if p.startswith(d.rstrip("/") + "/"))
        if not paths:
            return None, f"no files under {d}/ (use list_files with no directory to see everything)"
        return paths, None

    def _file(self, path: object) -> tuple[str | None, str | None]:
        p = norm_path(path)
        if p is None:
            return None, "path must be a file path relative to the repository root"
        if p not in self.files:
            if any(f.startswith(p + "/") for f in self.files):
                return None, f"{p} is a directory; use list_files"
            return None, f"no such file: {p} (use list_files to see the files)"
        return p, None

    def _t_list_files(self, args: dict) -> str:
        paths, error = self._under(args.get("directory"))
        return _err(error) if error else _clip("\n".join(paths), MAX_RESULT)

    def _t_read_file(self, args: dict) -> str:
        p, error = self._file(args.get("path"))
        if error:
            return _err(error)
        lines = self.files[p].splitlines()
        try:
            start = max(1, int(args.get("start_line") or 1))
            end = int(args.get("end_line") or start + READ_LINES - 1)
        except (TypeError, ValueError):
            return _err("start_line and end_line must be whole numbers")
        end = min(end, start + READ_LINES - 1, len(lines))
        if not lines:
            return f"{p} is empty"
        if start > len(lines):
            return _err(f"{p} has {len(lines)} lines")
        out, used = [], 0
        for n in range(start, end + 1):
            row = f"{n:>5}  {lines[n - 1]}"
            if used + len(row) > MAX_RESULT - 200:
                end = n - 1
                break
            out.append(row)
            used += len(row) + 1
        note = f"\n[lines {start}-{end} of {len(lines)}; read on with start_line={end + 1}]" if end < len(lines) or start > 1 else ""
        return "\n".join(out) + note

    def _t_search(self, args: dict) -> str:
        text = args.get("text")
        if not isinstance(text, str) or not text.strip():
            return _err("text must be a non-empty string")
        try:
            pattern = re.compile(text if args.get("regex") else re.escape(text), re.IGNORECASE)
        except re.error as e:
            return _err(f"not a valid regular expression: {e}")
        paths, error = self._under(args.get("directory"))
        if error:
            return _err(error)
        hits = []
        for path in paths:
            for i, line in enumerate(self.files[path].splitlines(), 1):
                if pattern.search(line):
                    hits.append(f"{path}:{i}: {line.strip()[:200]}")
        if not hits:
            return f"no matches for {text!r}"
        more = f"\n[{len(hits) - MAX_MATCHES} more matches; narrow the search]" if len(hits) > MAX_MATCHES else ""
        return _clip("\n".join(hits[:MAX_MATCHES]) + more, MAX_RESULT)

    def _writable(self, path: object) -> tuple[str | None, str | None]:
        p = norm_path(path)
        if p is None:
            return None, "path must be a file path relative to the repository root"
        if p == "_grade.py":
            return None, f"{p} is a reserved path"
        if any(f.startswith(p + "/") for f in self.files):
            return None, f"{p} is a directory"
        return p, None

    def _t_edit_file(self, args: dict) -> str:
        p, error = self._file(args.get("path"))
        if error:
            return _err(error)
        old, new = args.get("old_text"), args.get("new_text")
        if not isinstance(old, str) or not old or not isinstance(new, str):
            return _err("old_text (not empty) and new_text must be strings")
        n = self.files[p].count(old)
        if n == 0:
            return _err(f"old_text isn't in {p}: it must match exactly, including spaces and indentation (read_file shows the text)")
        if n > 1 and not args.get("replace_all"):
            return _err(f"old_text appears {n} times in {p}; include more of the surrounding text so it's unique, or set replace_all")
        text = self.files[p].replace(old, new) if args.get("replace_all") else self.files[p].replace(old, new, 1)
        if len(text) > MAX_WRITE:
            return _err(f"the file would be too long ({len(text)} characters; the limit is {MAX_WRITE})")
        self.files[p] = text
        return f"edited {p}: replaced {n if args.get('replace_all') else 1} occurrence{'s' if args.get('replace_all') and n > 1 else ''}"

    def _t_write_file(self, args: dict) -> str:
        p, error = self._writable(args.get("path"))
        content = args.get("content")
        if error:
            return _err(error)
        if not isinstance(content, str):
            return _err("content must be a string: the whole file")
        if len(content) > MAX_WRITE:
            return _err(f"content is too long ({len(content)} characters; the limit is {MAX_WRITE})")
        if p not in self.files and len(self.files) >= MAX_FILES:
            return _err("too many files in the repository")
        created = p not in self.files
        self.files[p] = content
        return f"{'created' if created else 'wrote'} {p} ({len(content.splitlines())} lines)"

    def _t_delete_file(self, args: dict) -> str:
        p, error = self._file(args.get("path"))
        if error:
            return _err(error)
        del self.files[p]
        return f"deleted {p}"

    def _t_apply_patch(self, args: dict) -> str:
        text = args.get("patch")
        if not isinstance(text, str) or not text.strip():
            return _err("patch must be the patch text")

        def ok(path: str) -> str | None:
            q, error = self._writable(path)
            if error:
                raise patching.PatchError(error)
            return q
        try:
            files, done = patching.apply(self.files, text, ok)
        except patching.PatchError as e:
            return _err(f"the patch didn't apply, so nothing changed: {e}")
        too = next((p for p, v in files.items() if self.files.get(p) != v and len(v) > MAX_WRITE), None)
        if too:
            return _err(f"{too} would be too long (the limit is {MAX_WRITE} characters); nothing changed")
        if len(files) > MAX_FILES:
            return _err("too many files in the repository; nothing changed")
        self.files = files
        return "; ".join(done)

    def _t_shell(self, args: dict) -> str:
        if self.sandbox is None:
            return _err("no sandbox is available to run commands")
        command = args.get("command")
        if not isinstance(command, str) or not command.strip():
            return _err("command must be a non-empty string")
        with _RUNS:
            r = self.sandbox.run({k: v for k, v in self.files.items() if k != "_grade.py"}, ["sh", "-c", SHELL, "sh", command], SHELL_TIMEOUT)
        if r.timed_out:
            return f"the command timed out after {SHELL_TIMEOUT:.0f}s"
        return _clip(f"exit code {r.exit_code}\n{r.stdout}{r.stderr}".rstrip() + "\n", MAX_RESULT, tail=True)

    def _t_run_tests(self, args: dict) -> str:
        if self.sandbox is None:
            return _err("no sandbox is available to run tests")
        targets = str(args.get("target") or "").split()
        bad = [t for t in targets if not _TARGET.match(t) or norm_path(t.split("::")[0]) is None]
        if bad:
            return _err(f"not a test file or id: {bad[0]!r} (e.g. tests/test_router.py or tests/test_router.py::test_name)")
        argv = ["sh", "-c", RUN_TESTS, "sh", *(targets or ["tests"])]
        if args.get("keyword"):
            argv += ["-k", str(args["keyword"])]
        with _RUNS:
            r = self.sandbox.run({k: v for k, v in self.files.items() if k != "_grade.py"}, argv, TEST_TIMEOUT)
        if r.timed_out:
            return f"the test run timed out after {TEST_TIMEOUT:.0f}s"
        return _clip(f"exit code {r.exit_code}\n{r.stdout}{r.stderr}".rstrip() + "\n", MAX_RESULT, tail=True)

    # ----------------------------------------------------------------- grading
    def grade(self, final_answer: str) -> tuple[bool, list[str]]:
        if self.sandbox is None:
            return False, ["no sandbox to run the tests"]
        result = run_grader(self.sandbox, grading_files(self.files, self.tests))
        if result.get("error"):
            return False, [result["error"]]
        got = result.get("tests") or {}
        broke = [t for t in self.fail_to_pass if got.get(t) != "passed"]
        regressed = [t for t in self.pass_to_pass if got.get(t) != "passed"]
        errors = dict((e[0], e[1]) for e in result.get("errors") or [])

        def why(t: str) -> str:
            last = (errors.get(t) or "").strip().splitlines()
            return f"{t} {got.get(t, 'not run')}" + (f": {last[-1][:200]}" if last else "")

        reasons = [f"needed: {why(t)}" for t in broke[:6]] + [f"broke: {why(t)}" for t in regressed[:6]]
        if len(broke) > 6 or len(regressed) > 6:
            reasons.append(f"{len(broke)} of {len(self.fail_to_pass)} new tests and {len(regressed)} of {len(self.pass_to_pass)} "
                           "existing tests didn't pass")
        if not got and result.get("errors"):
            reasons.append(f"no tests ran: {result['errors'][0][1][-300:]}")
        return not broke and not regressed, reasons
