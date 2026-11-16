"""The codebase suite: real changes from this repository's history, graded by each change's own tests."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from meritorder.evals import jobs
from meritorder.evals.envs import codebase
from meritorder.evals.runner import estimate_tokens, load_suite
from meritorder.evals.sandbox import DockerSandbox, LocalSandbox, deps_image

SUITE = Path(codebase.__file__).parents[1] / "suites" / "codebase"

# A tiny repository in the suite's format: a package with a bug, the tests before, and the tests after.
APP = "def add(a, b):\n    return a - b\n\n\ndef double(x):\n    return x * 2\n"
FIXED = "def add(a, b):\n    return a + b\n\n\ndef double(x):\n    return x * 2\n"
TEST_OLD = "from app import double\n\n\ndef test_double():\n    assert double(2) == 4\n"
TEST_NEW = TEST_OLD + "\n\ndef test_add():\n    from app import add\n    assert add(2, 3) == 5\n"
PYPROJECT = "[tool.pytest.ini_options]\ntestpaths = [\"tests\"]\n"


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    files = {"src/app.py": APP, "tests/test_app.py": TEST_OLD, "pyproject.toml": PYPROJECT, "docs/notes.md": "# Notes\n" + "line\n" * 900}
    blobs = {}

    def ref(text):
        blobs[codebase_digest(text)] = text
        return codebase_digest(text)
    manifest = {"task": "tiny", "commit": "c0ffee", "base": "beef", "subject": "Fix add", "image": "x",
                "files": {p: ref(v) for p, v in files.items()}, "tests": {"tests/test_app.py": ref(TEST_NEW)},
                "reference": {"src/app.py": ref(FIXED)},
                "fail_to_pass": ["tests/test_app.py::test_add"], "pass_to_pass": ["tests/test_app.py::test_double"], "flaky": []}
    (tmp_path / "blobs.json").write_text(json.dumps(blobs))
    (tmp_path / "tiny.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(codebase, "FIXTURE", tmp_path)
    codebase._blobs.cache_clear()
    codebase.load_task.cache_clear()
    yield codebase.CodebaseEnv(sandbox=LocalSandbox(), task="tiny")
    codebase._blobs.cache_clear()
    codebase.load_task.cache_clear()


def codebase_digest(text):
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def test_reading_and_searching(tiny):
    assert tiny.call("list_files", {}) == "docs/notes.md\npyproject.toml\nsrc/app.py\ntests/test_app.py"
    assert tiny.call("list_files", {"directory": "src"}) == "src/app.py"
    assert "error" in tiny.call("list_files", {"directory": "../etc"})
    assert tiny.call("read_file", {"path": "src/app.py"}).splitlines()[1] == "    2      return a - b"
    page = tiny.call("read_file", {"path": "docs/notes.md"})
    assert page.endswith(f"[lines 1-{codebase.READ_LINES} of 901; read on with start_line={codebase.READ_LINES + 1}]")
    assert tiny.call("read_file", {"path": "docs/notes.md", "start_line": 900}).splitlines()[0] == "  900  line"
    assert "has 901 lines" in tiny.call("read_file", {"path": "docs/notes.md", "start_line": 5000})
    assert "is a directory" in tiny.call("read_file", {"path": "src"})
    assert "whole numbers" in tiny.call("read_file", {"path": "src/app.py", "start_line": "x"})
    assert tiny.call("search", {"text": "RETURN A"}) == "src/app.py:2: return a - b"
    assert tiny.call("search", {"text": r"def \w+\(x\)", "regex": True}) == "src/app.py:5: def double(x):"
    assert "not a valid regular expression" in tiny.call("search", {"text": "(", "regex": True})
    assert tiny.call("search", {"text": "nowhere"}) == "no matches for 'nowhere'"


def test_editing(tiny):
    assert "isn't in src/app.py" in tiny.call("edit_file", {"path": "src/app.py", "old_text": "a * b", "new_text": "x"})
    assert "appears 2 times" in tiny.call("edit_file", {"path": "src/app.py", "old_text": "return", "new_text": "yield"})
    assert tiny.call("edit_file", {"path": "src/app.py", "old_text": "a - b", "new_text": "a + b"}) == "edited src/app.py: replaced 1 occurrence"
    assert tiny.files["src/app.py"] == FIXED and tiny.changed() == ["src/app.py"]
    assert "replaced 2 occurrences" in tiny.call("edit_file", {"path": "src/app.py", "old_text": "return", "new_text": "return ",
                                                              "replace_all": True})
    assert tiny.call("write_file", {"path": "src/new.py", "content": "X = 1\n"}) == "created src/new.py (1 lines)"
    assert "reserved" in tiny.call("write_file", {"path": "_grade.py", "content": "print('passed')"})
    assert "is a directory" in tiny.call("write_file", {"path": "src", "content": ""})
    assert tiny.call("delete_file", {"path": "src/new.py"}) == "deleted src/new.py" and "src/new.py" not in tiny.files
    assert "no such file" in tiny.call("delete_file", {"path": "src/new.py"})


def test_running_tests_and_grading(tiny):
    out = tiny.call("run_tests", {})
    assert out.startswith("exit code 0") and "1 passed" in out  # the tests before the change
    assert "not a test file or id" in tiny.call("run_tests", {"target": "--rootdir=/"})
    assert "1 passed" in tiny.call("run_tests", {"target": "tests/test_app.py::test_double"})
    assert "1 deselected" in tiny.call("run_tests", {"keyword": "nothing_matches"})

    passed, reasons = tiny.grade("done")
    assert not passed and reasons[0].startswith("needed: tests/test_app.py::test_add failed")
    tiny.call("write_file", {"path": "tests/test_app.py", "content": "def test_add():\n    pass\n"})  # rewriting the tests doesn't help
    assert not tiny.grade("done")[0]
    tiny.call("edit_file", {"path": "src/app.py", "old_text": "a - b", "new_text": "a + b"})
    assert tiny.grade("done") == (True, [])
    tiny.call("edit_file", {"path": "src/app.py", "old_text": "x * 2", "new_text": "x * 3"})  # breaks what worked
    passed, reasons = tiny.grade("done")
    assert not passed and len(reasons) == 1 and reasons[0].startswith("broke: tests/test_app.py::test_double failed: ")


def test_grading_reads_only_its_own_line(tiny):
    """What the code under test prints can't pass for the grader's result."""
    tiny.call("write_file", {"path": "src/app.py", "content": FIXED.replace("return a + b", "print('{\"tests\": {}}')\n    return a + b")})
    assert tiny.grade("done") == (True, [])
    r = codebase.run_grader(LocalSandbox(), {"tests/test_x.py": "def test_x():\n    pass\n", "pyproject.toml": PYPROJECT,
                                             "_grade.py": "print('forged')"})
    assert r["error"] and "crashed" in r["error"]


def test_the_tasks_tab_shows_what_the_change_touched(tiny):
    visible, grader = codebase.CodebaseEnv.fixture_files(task="tiny")
    assert visible["repository/FILES"].splitlines() == ["docs/notes.md", "pyproject.toml", "src/app.py", "tests/test_app.py"]
    assert visible["repository/src/app.py"] == APP and "repository/docs/notes.md" not in visible
    assert grader["tests after the commit/tests/test_app.py"] == TEST_NEW and grader["reference solution/src/app.py"] == FIXED
    assert grader["fail_to_pass"] == "tests/test_app.py::test_add"
    assert codebase.CodebaseEnv.graded_by(task="tiny").startswith("1 new or changed tests and 1 existing ones")
    with pytest.raises(ValueError):
        codebase.load_task("../etc")


def test_sandbox_variants_name_their_packages(tmp_path):
    reqs = tmp_path / "requirements.txt"
    reqs.write_text("httpx==1.0 --hash=sha256:abc\n")
    box = DockerSandbox().variant(requirements=reqs, memory="2g", pids=512, tmp="512m")
    assert box.image == deps_image(reqs) and box.image.startswith("meritorder-eval-deps:") and (box.memory, box.pids, box.tmp) == ("2g", 512, "512m")
    assert box.cpus == "1" and box.base == "python:3.12-slim"
    reqs.write_text("httpx==1.1 --hash=sha256:def\n")
    assert deps_image(reqs) != box.image  # a changed lock file is a different image
    assert LocalSandbox().variant(requirements=reqs).available() is None


def test_a_missing_deps_image_is_built_once(tmp_path, monkeypatch):
    reqs = tmp_path / "requirements.txt"
    reqs.write_text("httpx==1.0 --hash=sha256:abc\n")
    calls = []

    def run(cmd, **kw):
        calls.append(cmd[1])
        if cmd[1] == "build":
            assert "FROM python:3.12-slim" in (Path(cmd[-1]) / "Dockerfile").read_text()
            assert (Path(cmd[-1]) / "requirements.txt").read_text() == reqs.read_text()
        return subprocess.CompletedProcess(cmd, 1 if cmd[1] == "image" else 0, b"", b"")
    monkeypatch.setattr("meritorder.evals.sandbox.shutil.which", lambda _: "/usr/bin/docker")
    monkeypatch.setattr("meritorder.evals.sandbox.subprocess.run", run)
    assert DockerSandbox().variant(requirements=reqs).available() is None
    assert calls == ["image", "image", "build"]  # never pulled: built from the requirements

    def fail(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 1, b"", b"no space left")
    monkeypatch.setattr("meritorder.evals.sandbox.subprocess.run", fail)
    assert "couldn't build" in DockerSandbox().variant(requirements=reqs).available()


# ------------------------------------------------------------------ the real suite

def test_every_task_has_its_data_and_an_issue():
    suite = load_suite("codebase")
    ids = {t["id"] for t in suite["tasks"]}
    assert ids and ids == set(codebase.task_ids())
    for t in suite["tasks"]:
        assert t["env"] == "meritorder.evals.envs.codebase:CodebaseEnv" and t["env_args"] == {"task": t["id"]}
        assert t["prompt"].startswith("You're working in Merit Order") and "## Issue" in t["prompt"]
        d = codebase.load_task(t["id"])
        assert d["fail_to_pass"] and not set(d["fail_to_pass"]) & set(d["pass_to_pass"])
        assert all(i.split("::")[0] in d["tests"] for i in d["fail_to_pass"] + d["pass_to_pass"]), t["id"]
        assert not any(p.startswith(("src/meritorder/evals/envs/fixtures/codebase/", "src/meritorder/evals/suites/codebase/")) for p in d["files"])
        assert t["commit"] and d["commit"].startswith(str(t["commit"]))
        fresh, cached, out = estimate_tokens(t)
        assert fresh > 50_000 and out > 20_000  # a run reads a lot of code


def test_the_suite_shows_on_the_evals_page():
    names = {s["name"]: s for s in jobs.suites()}
    assert names["codebase"]["kinds"]["agent"] == names["codebase"]["tasks"]
    row = next(r for r in jobs.task_list("codebase"))
    assert "new or changed tests" in row["graded_by"]
    detail = jobs.task_detail("codebase", row["id"])
    assert detail["agent"]["needs_sandbox"] and "repository/FILES" in detail["agent"]["visible"]
    assert any(k.startswith("reference solution/") for k in detail["agent"]["grader"])
    assert {x["name"] for x in detail["tools"]} == {"list_files", "read_file", "search", "edit_file", "write_file", "delete_file", "apply_patch", "shell", "run_tests"}


@pytest.mark.parametrize("task_id", codebase.task_ids())
def test_each_reference_solution_passes_its_new_tests(task_id):
    """The commit's own change passes the tests that fail before it, run here with this checkout's packages."""
    d = codebase.load_task(task_id)
    files = dict(d["files"])
    for p, v in d["reference"].items():
        if v is None:
            files.pop(p, None)
        else:
            files[p] = v
    # Only this Python and the system's own tools, as in the sandbox: nothing else installed here (such as SkyPilot) leaks in.
    box = LocalSandbox(path=f"{Path(sys.executable).parent}:/usr/bin:/bin")
    r = codebase.run_grader(box, codebase.grading_files(files, d["tests"]), timeout=300, only=d["fail_to_pass"])
    assert r["error"] is None and {t: r["tests"].get(t) for t in d["fail_to_pass"]} == {t: "passed" for t in d["fail_to_pass"]}, r["errors"][:3]


async def test_a_grader_that_breaks_skips_the_task_instead_of_ending_the_run(tiny, monkeypatch):
    """Docker stopped answering while grading: the task isn't graded, and the rest of the run carries on."""
    import httpx

    from meritorder.evals.runner import RunContext, run_task

    def stuck(*a, **k):
        raise subprocess.TimeoutExpired(["docker", "rm"], 60)
    monkeypatch.setattr(codebase.CodebaseEnv, "grade", lambda self, final: stuck())
    t = {"id": "tiny", "task": "coding", "level": 5, "kind": "agent", "prompt": "Fix it.", "env": "meritorder.evals.envs.codebase:CodebaseEnv",
         "env_args": {"task": "tiny"}}

    def reply(request):
        return httpx.Response(200, json={"choices": [{"index": 0, "message": {"role": "assistant", "content": "Done."}, "finish_reason": "stop"}]})
    async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as c:
        res = await run_task(c, "http://gw", "k", "m", t, "s", RunContext(sandbox=LocalSandbox()))
    assert res.skipped and res.error.startswith("not graded: grading failed: TimeoutExpired") and not res.passed


def test_a_stuck_docker_cleanup_doesnt_raise(monkeypatch):
    def run(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, kw.get("timeout", 1))
    monkeypatch.setattr("meritorder.evals.sandbox.subprocess.run", run)
    r = DockerSandbox().run({"a.py": "x"}, ["python", "a.py"], timeout=1)
    assert r.timed_out and r.exit_code == -1


def test_only_the_codebase_sandbox_lets_programs_run_from_tmp(monkeypatch):
    seen = []

    def run(cmd, **kw):
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, b"", b"")
    monkeypatch.setattr("meritorder.evals.sandbox.subprocess.run", run)
    DockerSandbox().run({"a.py": "x"}, ["python", "a.py"])
    DockerSandbox().variant(**{k: v for k, v in codebase.SANDBOX.items() if k != "requirements"}).run({"a.py": "x"}, ["python", "a.py"])
    plain, big = (c[c.index("--tmpfs") + 1] for c in seen)
    assert plain == "/tmp:rw,size=64m" and big == "/tmp:rw,exec,size=256m"
    assert seen[1][seen[1].index("--memory") + 1] == "1g" and seen[1][seen[1].index("--network") + 1] == "none"


def test_tools_refuse_what_they_cant_do(tiny, monkeypatch):
    assert "no files under" in tiny.call("list_files", {"directory": "nowhere"})
    assert "relative to the repository root" in tiny.call("read_file", {"path": "../x"})
    tiny.files["empty.txt"] = ""
    assert tiny.call("read_file", {"path": "empty.txt"}) == "empty.txt is empty"
    tiny.files["wide.txt"] = ("x" * 1000 + "\n") * 50  # 400 lines wouldn't fit: it stops at what does
    page = tiny.call("read_file", {"path": "wide.txt"})
    assert len(page) <= codebase.MAX_RESULT and "read on with start_line=" in page
    assert "non-empty string" in tiny.call("search", {"text": "  "})
    assert "not a directory" in tiny.call("search", {"text": "x", "directory": "../up"})
    assert "relative to the repository root" in tiny.call("write_file", {"path": "", "content": "x"})
    assert "no such file" in tiny.call("edit_file", {"path": "nope.py", "old_text": "a", "new_text": "b"})
    assert "must be strings" in tiny.call("edit_file", {"path": "src/app.py", "old_text": "", "new_text": "b"})
    monkeypatch.setattr(codebase, "MAX_WRITE", 50)
    assert "would be too long" in tiny.call("edit_file", {"path": "src/app.py", "old_text": "a - b", "new_text": "a + b" * 20})
    assert "too long" in tiny.call("write_file", {"path": "src/x.py", "content": "y" * 51})
    assert "must be a string" in tiny.call("write_file", {"path": "src/x.py", "content": 5})
    monkeypatch.setattr(codebase, "MAX_FILES", len(tiny.files))
    assert "too many files" in tiny.call("write_file", {"path": "src/x.py", "content": "y"})


def test_without_a_sandbox_or_with_a_slow_one(tiny):
    from meritorder.evals.sandbox import SandboxResult

    bare = codebase.CodebaseEnv(sandbox=None, task="tiny")
    assert "no sandbox" in bare.call("run_tests", {}) and bare.grade("done") == (False, ["no sandbox to run the tests"])

    class Slow:
        def run(self, files, argv, timeout=30.0):
            return SandboxResult(-1, "", "", timed_out=True)
    slow = codebase.CodebaseEnv(sandbox=Slow(), task="tiny")
    assert slow.call("run_tests", {}) == f"the test run timed out after {codebase.TEST_TIMEOUT:.0f}s"
    assert slow.grade("done") == (False, [f"the tests timed out after {codebase.TEST_TIMEOUT:.0f}s"])

    class Garbled:
        def run(self, files, argv, timeout=30.0):
            return SandboxResult(0, argv[-1] + "not-base64!!", "")  # the nonce, then something that isn't a result
    assert "couldn't be read" in codebase.run_grader(Garbled(), {})["error"]


def test_grading_lists_what_failed_and_says_when_nothing_ran(tiny, monkeypatch):
    many = [f"tests/test_app.py::test_{i}" for i in range(8)]
    tiny.fail_to_pass, tiny.pass_to_pass = many, many
    monkeypatch.setattr(codebase, "run_grader", lambda *a, **k: {"tests": {}, "errors": [["tests/test_app.py", "ImportError: boom"]], "error": None})
    passed, reasons = tiny.grade("done")
    assert not passed and len([r for r in reasons if r.startswith("needed: ")]) == 6 and len([r for r in reasons if r.startswith("broke: ")]) == 6
    assert "8 of 8 new tests and 8 of 8 existing tests didn't pass" in reasons[-2] and reasons[-1] == "no tests ran: ImportError: boom"


async def test_a_task_whose_sandbox_cant_be_built_is_skipped(tiny, monkeypatch):
    import httpx

    from meritorder.evals.runner import RunContext, run_task

    class NoImage(LocalSandbox):
        def variant(self, **_):
            return self

        def available(self):
            return "couldn't build meritorder-eval-deps:x: no space left"
    t = {"id": "tiny", "task": "coding", "level": 5, "kind": "agent", "prompt": "Fix it.", "env": "meritorder.evals.envs.codebase:CodebaseEnv",
         "env_args": {"task": "tiny"}}
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(500))) as c:
        res = await run_task(c, "http://gw", "k", "m", t, "s", RunContext(sandbox=NoImage()))
    assert res.skipped and res.error == "skipped: couldn't build meritorder-eval-deps:x: no space left" and res.turns == 0


def test_patches_and_the_shell(tiny):
    patch = "*** Begin Patch\n*** Update File: src/app.py\n@@ def add(a, b):\n-    return a - b\n+    return a + b\n*** End Patch"
    assert tiny.call("apply_patch", {"patch": patch}) == "updated src/app.py" and tiny.files["src/app.py"] == FIXED
    assert "didn't apply, so nothing changed" in tiny.call("apply_patch", {"patch": patch})  # already applied
    assert "reserved" in tiny.call("apply_patch", {"patch": "*** Begin Patch\n*** Add File: _grade.py\n+x\n*** End Patch"})
    assert "patch text" in tiny.call("apply_patch", {"patch": " "})
    assert tiny.grade("done") == (True, [])

    out = tiny.call("shell", {"command": "grep -rn 'def double' src && echo hi > /dev/null"})
    assert out.startswith("exit code 0") and "src/app.py:5:def double(x):" in out
    assert tiny.call("shell", {"command": "exit 3"}).startswith("exit code 3")
    assert "non-empty" in tiny.call("shell", {"command": ""})
    assert "no sandbox" in codebase.CodebaseEnv(sandbox=None, task="tiny").call("shell", {"command": "ls"})


def test_a_patch_that_would_overflow_changes_nothing(tiny, monkeypatch):
    monkeypatch.setattr(codebase, "MAX_WRITE", 40)
    big = "*** Begin Patch\n*** Add File: src/big.py\n" + "+x = 1\n" * 20 + "*** End Patch"
    assert "too long" in tiny.call("apply_patch", {"patch": big}) and "src/big.py" not in tiny.files
    monkeypatch.setattr(codebase, "MAX_WRITE", 300_000)
    monkeypatch.setattr(codebase, "MAX_FILES", len(tiny.files))
    assert "too many files" in tiny.call("apply_patch", {"patch": "*** Begin Patch\n*** Add File: src/n.py\n+x\n*** End Patch"})


def test_the_estimate_grows_the_context_and_charges_long_context_prices():
    from meritorder.evals.runner import turn_tokens
    from meritorder.tokens import text_tokens

    t = {"id": "x", "task": "coding", "level": 5, "kind": "agent", "prompt": "p" * 4000, "env": "meritorder.evals.envs.codebase:CodebaseEnv", "est_turns": 50,
         "est_tool_tokens": 5000, "est_output_tokens": 1000}
    rows = turn_tokens(t)
    prompt = round(text_tokens("p" * 4000)) + 50  # the gateway's own estimate of the prompt, plus the task's framing
    assert len(rows) == 50 and rows[0] == (prompt, prompt + 6000, 0, 1000) and rows[49][0] == prompt + 5000 * 49
    assert estimate_tokens(t) == (sum(r[1] for r in rows), sum(r[2] for r in rows), 50_000)
    flat = {"input": 1.0, "output": 5.0, "cache_read": 0.1}
    tiered = {**flat, "tiers": [{"above": 100_000, "input": 5.0, "output": 25.0, "cache_read": 0.5}]}
    view = {"flat": {"routes": [{"live": True, "price": flat}]}, "tiered": {"routes": [{"live": True, "price": tiered}]}}
    est = jobs.estimate(view, {"tasks": [t]}, ["flat", "tiered"], 1, [])["by_model"]
    # Turns past 100k tokens of context (about 30 of 50) cost five times as much.
    assert 3 * est["flat"] < est["tiered"] < 5 * est["flat"]
