"""Virtual repositories: the model explores, and for edit tasks fixes, a small Python project through file tools.

Each project is a directory under fixtures/repos/<name>/:

    files/      the repository the model sees (and edits)
    hidden/     unittest files the grader runs after the episode (edit tasks); the model never sees them
    meta.yaml   for question tasks, the key=value pairs the final answer must give

``RepoQAEnv`` answers a question about the code (list_files, read_file, search). ``RepoEnv`` adds write_file and
run_tests and is graded by running hidden/ against the edited repository in the sandbox, never in this process.
"""

from __future__ import annotations

import ast
import json
import posixpath
import re
import secrets
from pathlib import Path

import yaml

from . import Env, tool

REPOS = Path(__file__).with_name("fixtures") / "repos"
MAX_WRITE = 100_000     # characters in one written file
MAX_FILES = 200         # files in the repository after writes
MAX_RESULT = 12_000     # characters of tool output returned to the model
MAX_MATCHES = 60
TEST_TIMEOUT = 60.0
VISIBLE_TESTS = ["python", "-B", "-m", "unittest", "discover", "-s", "tests", "-v"]

# Runs the hidden tests inside the sandbox and prints one line: <nonce>{"total": n, "failed": [...], "error": ...}.
# Standard library only (it runs in a bare python image). The nonce marks the runner's own line.
HIDDEN_RUNNER = '''\
import io, json, os, sys, unittest
nonce = sys.argv[1]
sys.path.insert(0, os.getcwd())
out = {"total": 0, "failed": [], "error": None}
try:
    suite = unittest.TestLoader().discover("_hidden", top_level_dir="_hidden")
    result = unittest.TextTestRunner(stream=io.StringIO(), verbosity=0).run(suite)
    out["total"] = result.testsRun
    out["failed"] = [{"name": t.id(), "error": tb[-800:]} for t, tb in result.failures + result.errors]
    out["failed"] += [{"name": t.id(), "error": "unexpected success"} for t in result.unexpectedSuccesses]
except BaseException as e:
    out["error"] = (type(e).__name__ + ": " + str(e))[:1000]
print(nonce + json.dumps(out))
'''


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    """``tool()`` with an exact required list (``tool()`` treats an empty list as "all required")."""
    t = tool(name, description, properties)
    t["function"]["parameters"]["required"] = list(required)
    return t


def _read_tree(root: Path) -> dict[str, str]:
    if not root.is_dir():
        return {}
    return {p.relative_to(root).as_posix(): p.read_text(encoding="utf-8") for p in sorted(root.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts and not p.name.startswith(".")}


def load_project(name: str) -> tuple[dict[str, str], dict[str, str], dict]:
    """(visible files, hidden test files, meta) for fixtures/repos/<name>."""
    root = REPOS / str(name)
    if not re.fullmatch(r"[a-z0-9_-]+", str(name)) or not (root / "files").is_dir():
        raise ValueError(f"unknown project {name!r}")
    meta_path = root / "meta.yaml"
    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return _read_tree(root / "files"), _read_tree(root / "hidden"), meta or {}


def count_tests(sources: dict[str, str]) -> int:
    """test_* methods on classes in the given unittest files."""
    n = 0
    for text in sources.values():
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.ClassDef):
                n += sum(1 for f in node.body if isinstance(f, ast.FunctionDef) and f.name.startswith("test"))
    return n


def norm_path(path: object) -> str | None:
    """A repository-relative path, or None if it isn't one ('..', empty, not a string)."""
    if not isinstance(path, str):
        return None
    p = path.strip().replace("\\", "/").lstrip("/")
    p = posixpath.normpath(p) if p else ""
    if p in ("", ".") or p == ".." or p.startswith("../"):
        return None
    return p


def _clip(text: str, limit: int = MAX_RESULT, tail: bool = False) -> str:
    if len(text) <= limit:
        return text
    if tail:
        return "[... earlier output cut]\n" + text[-limit:]
    return text[:limit] + f"\n[... cut: {len(text) - limit} more characters]"


def _err(message: str) -> str:
    return json.dumps({"error": message})


class RepoQAEnv(Env):
    """Answer a question about a repository by reading it. Graded on key=value pairs in the final answer."""
    grading = 'The final answer must state the right key=value facts about the code'  # shown on the console's Tasks tab

    tools = [
        _tool("list_files", "List the repository's files, as paths relative to the repository root.",
              {"directory": {"type": "string", "description": "Only list files under this directory (default: all files)"}}, []),
        _tool("read_file", "Read one file and return its full text.",
              {"path": {"type": "string", "description": "Path relative to the repository root, e.g. pkg/module.py"}}, ["path"]),
        _tool("search", "Find the lines that contain a piece of text (plain text, not a regex; case-insensitive). "
                        "Returns path:line: text for each match.",
              {"text": {"type": "string", "description": "Text to look for"},
               "directory": {"type": "string", "description": "Only search files under this directory (default: everywhere)"}},
              ["text"]),
    ]

    def __init__(self, sandbox=None, project: str = "", **args):
        super().__init__(sandbox, project=project, **args)
        self.files, self.hidden, self.meta = load_project(project)

    def call(self, name: str, arguments: dict) -> str:
        return getattr(self, f"_t_{name}")(arguments)

    # ----------------------------------------------------------------- tools
    def _under(self, directory: object) -> tuple[list[str] | None, str | None]:
        if directory in (None, "", ".", "/"):
            return sorted(self.files), None
        d = norm_path(directory)
        if d is None:
            return None, f"not a directory in the repository: {directory!r}"
        paths = sorted(p for p in self.files if p.startswith(d + "/"))
        if not paths:
            return None, f"no files under {d}/ (use list_files with no directory to see everything)"
        return paths, None

    def _t_list_files(self, args: dict) -> str:
        paths, error = self._under(args.get("directory"))
        return _err(error) if error else "\n".join(paths)

    def _t_read_file(self, args: dict) -> str:
        p = norm_path(args.get("path"))
        if p is None:
            return _err("path must be a file path relative to the repository root")
        if p not in self.files:
            if any(f.startswith(p + "/") for f in self.files):
                return _err(f"{p} is a directory; use list_files")
            return _err(f"no such file: {p} (use list_files to see the files)")
        return _clip(self.files[p])

    def _t_search(self, args: dict) -> str:
        text = args.get("text")
        if not isinstance(text, str) or not text.strip():
            return _err("text must be a non-empty string")
        paths, error = self._under(args.get("directory"))
        if error:
            return _err(error)
        needle, hits = text.lower(), []
        for path in paths:
            for i, line in enumerate(self.files[path].splitlines(), 1):
                if needle in line.lower():
                    hits.append(f"{path}:{i}: {line.strip()[:200]}")
        if not hits:
            return f"no matches for {text!r}"
        more = f"\n[{len(hits) - MAX_MATCHES} more matches; narrow the search]" if len(hits) > MAX_MATCHES else ""
        return "\n".join(hits[:MAX_MATCHES]) + more

    # ----------------------------------------------------------------- grading
    def grade(self, final_answer: str) -> tuple[bool, list[str]]:
        reasons = []
        for key, want in (self.meta.get("answer") or {}).items():
            found = re.findall(rf"\b{re.escape(str(key))}\s*[=:]\s*([-+]?\d+(?:\.\d+)?)", final_answer or "", re.IGNORECASE)
            if not found:
                reasons.append(f"no {key}=<number> in the answer")
            elif abs(float(found[-1]) - float(want)) > 1e-9:
                reasons.append(f"{key}: expected {want}, got {found[-1]}")
        return not reasons, reasons


class RepoEnv(RepoQAEnv):
    """Fix (and extend) a repository. Graded by hidden tests run in the sandbox against the edited files."""

    grading = "Hidden tests, stricter than the repository's own, run on the edited repository in a sandbox"  # shown on the console's Tasks tab
    needs_sandbox = True  # grading runs the edited repository's hidden tests

    tools = [
        *RepoQAEnv.tools,
        _tool("write_file", "Create a file or replace its whole content.",
              {"path": {"type": "string", "description": "Path relative to the repository root"},
               "content": {"type": "string", "description": "The complete new file content"}}, ["path", "content"]),
        _tool("run_tests", "Run the repository's tests (python -m unittest discover -s tests -v) and return the output.", {}, []),
    ]

    def __init__(self, sandbox=None, project: str = "", **args):
        super().__init__(sandbox, project=project, **args)
        self.original = dict(self.files)
        self.expected_hidden = count_tests(self.hidden)

    def changed(self) -> list[str]:
        return sorted(p for p in self.files if self.original.get(p) != self.files[p])

    def _t_write_file(self, args: dict) -> str:
        p, content = norm_path(args.get("path")), args.get("content")
        if p is None:
            return _err("path must be a file path relative to the repository root")
        if not isinstance(content, str):
            return _err("content must be a string: the whole file")
        if len(content) > MAX_WRITE:
            return _err(f"content is too long ({len(content)} characters; the limit is {MAX_WRITE})")
        if any(f.startswith(p + "/") for f in self.files):
            return _err(f"{p} is a directory")
        if p not in self.files and len(self.files) >= MAX_FILES:
            return _err("too many files in the repository")
        if p.startswith("_hidden/") or p == "_grade.py":
            return _err(f"{p} is a reserved path")
        created = p not in self.files
        self.files[p] = content
        return f"{'created' if created else 'wrote'} {p} ({len(content.splitlines())} lines)"

    def _t_run_tests(self, args: dict) -> str:
        if self.sandbox is None:
            return _err("no sandbox is available to run tests")
        r = self.sandbox.run(dict(self.files), VISIBLE_TESTS, TEST_TIMEOUT)
        if r.timed_out:
            return f"the test run timed out after {TEST_TIMEOUT:.0f}s"
        return _clip(f"exit code {r.exit_code}\n{r.stdout}{r.stderr}".rstrip() + "\n", tail=True)

    def run_hidden(self) -> dict:
        """Run hidden/ against the current files in the sandbox: {"total", "failed", "error"}."""
        if self.sandbox is None:
            return {"total": 0, "failed": [], "error": "no sandbox to run the hidden tests"}
        nonce = secrets.token_hex(8)
        files = {k: v for k, v in self.files.items() if not k.startswith("_hidden/")}
        files.update({f"_hidden/{k}": v for k, v in self.hidden.items()})
        files["_grade.py"] = HIDDEN_RUNNER
        r = self.sandbox.run(files, ["python", "-I", "-B", "_grade.py", nonce], TEST_TIMEOUT)
        if r.timed_out:
            return {"total": 0, "failed": [], "error": f"the hidden tests timed out after {TEST_TIMEOUT:.0f}s"}
        line = next((ln for ln in reversed(r.stdout.splitlines()) if ln.startswith(nonce)), None)
        if line is None:
            return {"total": 0, "failed": [], "error": f"the hidden test run crashed (exit {r.exit_code}): {(r.stderr or r.stdout)[-500:]}"}
        return json.loads(line[len(nonce):])

    def grade(self, final_answer: str) -> tuple[bool, list[str]]:
        result = self.run_hidden()
        if result.get("error"):
            return False, [result["error"]]
        reasons = [f"hidden test {f['name']} failed: {(f['error'].strip().splitlines() or [''])[-1]}" for f in result.get("failed", [])]
        if not reasons and result.get("total") != self.expected_hidden:
            reasons.append(f"ran {result.get('total')} hidden tests, expected {self.expected_hidden}")
        return not reasons, reasons
