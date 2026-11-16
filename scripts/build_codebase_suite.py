"""Build the codebase suite's data from this repository's git history.

    uv run python scripts/build_codebase_suite.py              # every task
    uv run python scripts/build_codebase_suite.py c30-eval-retries ...

Each task directory, src/meritorder/evals/suites/codebase/<id>/task.yaml, names a commit. For each one this takes:

- the repository at the commit's parent (what the model gets): src/, tests/, config/, the docs, pyproject.toml;
- tests/ at the commit (what grades it);
- the commit's change to everything else (the reference solution, shown on the console's Tasks tab and checked
  in CI).

It then runs the tests in the eval sandbox image: on the code before the commit with the tests after it, twice on
the code after it, and on the code and tests before it. Every test that passes after the commit (both times) has
to pass when grading. pass_to_pass are those that already passed before it: with the tests after it, or with the
tests before it in a file the commit didn't touch (a conftest.py fixture can need a name the change adds, which
fails every test with the new tests, though those tests didn't change). fail_to_pass are the rest: the tests the
commit added or changed. A test that passes once and fails once after the commit is flaky and left out of both.
A task with no fail_to_pass tests can't grade anything, so it stops the build.

It also refreshes fixtures/codebase/requirements.txt from uv.lock, so the sandbox has the repository's own pinned
packages. Needs git and Docker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from meritorder.evals.envs import codebase  # noqa: E402
from meritorder.evals.sandbox import DockerSandbox  # noqa: E402

SUITE = ROOT / "src" / "meritorder" / "evals" / "suites" / "codebase"
NEVER = ("src/meritorder/evals/envs/fixtures/codebase/", "src/meritorder/evals/suites/codebase/", "src/meritorder/ui_dist/")


def keep(path: str) -> bool:
    """The files a task's repository holds. Never this suite's own data (it holds the answers)."""
    if path.startswith(NEVER):
        return False
    return (path.startswith(("src/", "tests/", "config/")) or path in ("pyproject.toml", "README.md", "CHANGELOG.md")
            or re.fullmatch(r"docs/[^/]+\.md", path) is not None)


def git(*args: str) -> bytes:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, check=True).stdout


def tree(commit: str) -> dict[str, str]:
    paths = [p for p in git("ls-tree", "-r", "--name-only", "-z", commit).decode().split("\0") if p and keep(p)]
    out = {}
    for p in paths:
        try:
            out[p] = git("show", f"{commit}:{p}").decode("utf-8")
        except UnicodeDecodeError:
            continue  # binary: nothing the model could read anyway
    return out


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def refresh_requirements() -> None:
    reqs = subprocess.run(["uv", "export", "--frozen", "--all-extras", "--group", "dev", "--no-emit-project", "--no-header",
                           "--format", "requirements-txt"], cwd=ROOT, capture_output=True, check=True, text=True).stdout
    codebase.REQUIREMENTS.parent.mkdir(parents=True, exist_ok=True)
    codebase.REQUIREMENTS.write_text("# The repository's locked packages, for the codebase suite's sandbox. Made by\n"
                                     "# scripts/build_codebase_suite.py from uv.lock; don't edit.\n" + reqs, encoding="utf-8")


def build(task_dir: Path, box: DockerSandbox) -> tuple[dict, dict[str, str]]:
    spec = yaml.safe_load((task_dir / "task.yaml").read_text(encoding="utf-8"))
    commit = git("rev-parse", spec["commit"]).decode().strip()
    base = git("rev-parse", f"{commit}^").decode().strip()
    before, after = tree(base), tree(commit)
    tests = {p: v for p, v in after.items() if p.startswith("tests/")}
    code_before = {p: v for p, v in before.items() if not p.startswith("tests/")}
    code_after = {p: v for p, v in after.items() if not p.startswith("tests/")}
    reference = {p: code_after.get(p) for p in sorted(set(code_before) | set(code_after)) if code_before.get(p) != code_after.get(p)}

    tests_before = {p: v for p, v in before.items() if p.startswith("tests/")}
    with ThreadPoolExecutor(4) as pool:  # at most codebase.PARALLEL run at once (run_grader waits its turn)
        runs = list(pool.map(lambda job: codebase.run_grader(box, codebase.grading_files(*job)),
                             [(code_before, tests), (code_after, tests), (code_after, tests), (code_before, tests_before)]))
    for name, r in zip(("before (new tests)", "after", "after (again)", "before (old tests)"), runs):
        if r.get("error"):
            raise SystemExit(f"{task_dir.name}: the run {name} failed: {r['error']}")
    b, a1, a2, old = (r["tests"] for r in runs)
    flaky = sorted(t for t in set(a1) | set(a2) if a1.get(t) != a2.get(t))
    passed_after = {t for t in a1 if a1[t] == "passed" and a2.get(t) == "passed"}

    def already_passed(t: str) -> bool:
        path = t.split("::")[0]
        return b.get(t) == "passed" or (old.get(t) == "passed" and tests_before.get(path) == tests.get(path))
    f2p = sorted(t for t in passed_after if not already_passed(t))
    p2p = sorted(t for t in passed_after if already_passed(t))
    broken_by_commit = sorted(t for t in b if b[t] == "passed" and a1.get(t) not in ("passed", None) and t not in flaky)
    if not f2p:
        raise SystemExit(f"{task_dir.name}: no test fails before {commit[:8]} and passes after it, so nothing can grade it")
    blobs: dict[str, str] = {}

    def ref(text: str) -> str:
        h = digest(text)
        blobs[h] = text
        return h

    manifest = {
        "task": task_dir.name, "commit": commit, "base": base, "subject": git("log", "-1", "--format=%s", commit).decode().strip(),
        "image": box.image, "files": {p: ref(v) for p, v in sorted(before.items())}, "tests": {p: ref(v) for p, v in sorted(tests.items())},
        "reference": {p: (ref(v) if v is not None else None) for p, v in reference.items()},
        "fail_to_pass": f2p, "pass_to_pass": p2p, "flaky": flaky,
    }
    lines = sum(abs(len((v or "").splitlines()) - len(code_before.get(p, "").splitlines())) for p, v in reference.items())
    print(f"{task_dir.name}: {commit[:8]} {manifest['subject'][:60]}\n"
          f"    {len(f2p)} fail_to_pass, {len(p2p)} pass_to_pass, {len(flaky)} flaky; reference changes {len(reference)} files"
          f" (~{lines} lines net)" + (f"\n    passed before and not after: {broken_by_commit[:5]}" if broken_by_commit else ""))
    return manifest, blobs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tasks", nargs="*", help="task ids (default: every task directory)")
    ap.add_argument("--gc", action="store_true", help="build nothing: drop the data of tasks that no longer have a directory")
    args = ap.parse_args()
    if args.gc:
        for m in codebase.FIXTURE.glob("*.json"):
            if m.name != "blobs.json" and not (SUITE / m.stem / "task.yaml").exists():
                m.unlink()
                print(f"removed {m.name}")
        dirs = []
    if not args.gc:
        dirs = sorted(p.parent for p in SUITE.glob("*/task.yaml"))
    if args.tasks and not args.gc:
        dirs = [d for d in dirs if d.name in set(args.tasks)]
        missing = set(args.tasks) - {d.name for d in dirs}
        if missing:
            raise SystemExit(f"no task directory for {', '.join(sorted(missing))}")
    if dirs:
        refresh_requirements()
        box = DockerSandbox().variant(**codebase.SANDBOX)
        why = box.available()
        if why:
            raise SystemExit(why)
    blob_file = codebase.FIXTURE / "blobs.json"
    blobs = json.loads(blob_file.read_text(encoding="utf-8")) if blob_file.exists() else {}
    failed = []
    for d in dirs:
        try:
            manifest, new = build(d, box)
        except SystemExit as e:  # one task that can't be graded doesn't stop the others
            print(e, file=sys.stderr)
            failed.append(d.name)
            continue
        blobs.update(new)
        (codebase.FIXTURE / f"{d.name}.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    # Keep only what some task still uses, one blob per line so a new task shows as added lines.
    used: set[str] = set()
    for m in codebase.FIXTURE.glob("*.json"):
        if m.name != "blobs.json":
            data = json.loads(m.read_text(encoding="utf-8"))
            used |= set(data["files"].values()) | set(data["tests"].values()) | {h for h in data["reference"].values() if h}
    blob_file.write_text(json.dumps({h: blobs[h] for h in sorted(used)}, indent=0) + "\n", encoding="utf-8")
    print(f"{len(used)} distinct files in {blob_file.relative_to(ROOT)} ({blob_file.stat().st_size / 1e6:.1f} MB)")
    if failed:
        raise SystemExit(f"not built: {', '.join(failed)}")


if __name__ == "__main__":
    main()
