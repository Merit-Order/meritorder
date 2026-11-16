"""The frontier coding eval suite checks itself: every task's trusted reference passes all its hidden tests, its
deliberately buggy ``wrong.py`` fails at least one, and the task files are complete. New task directories are picked
up automatically. Reference solutions are trusted code, so they run in ``LocalSandbox``."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import yaml

import meritorder.evals
from meritorder.evals.pyrunner import forbidden_imports
from meritorder.evals.sandbox import LocalSandbox, run_tests

ROOT = Path(meritorder.evals.__file__).parent / "suites" / "frontier" / "coding"
TASK_DIRS = sorted(p for p in ROOT.iterdir() if (p / "task.yaml").is_file()) if ROOT.is_dir() else []
REQUIRED = {"id", "task", "level", "kind", "code", "reference", "wrong", "summary"}
MIN_TESTS = {1: 1, 2: 1, 3: 10, 4: 25, 5: 25}


def load(task_dir: Path) -> dict:
    return yaml.safe_load((task_dir / "task.yaml").read_text(encoding="utf-8"))


def prompt_of(task_dir: Path, spec: dict) -> str:
    return spec.get("prompt") or (task_dir / spec["prompt_file"]).read_text(encoding="utf-8")


def run(task_dir: Path, which: str) -> dict:
    try:
        spec = load(task_dir)
        solution = (task_dir / spec[which]).read_text(encoding="utf-8")
        tests = (task_dir / spec["code"]["tests"]).read_text(encoding="utf-8")
    except (OSError, KeyError, TypeError, yaml.YAMLError) as e:  # reported per task, not as a fixture error for all
        return {"total": 0, "passed": 0, "failed": [], "error": f"couldn't load the task: {type(e).__name__}: {e}"}
    return run_tests(LocalSandbox(), solution, tests, forbid=spec["code"].get("forbid"),
                     timeout=float(spec["code"].get("timeout", 30)))


@pytest.fixture(scope="module")
def outcomes() -> dict[tuple[str, str], dict]:
    """Every task's reference and wrong solution, run once and in parallel (each run is its own process)."""
    jobs = [(d, which) for d in TASK_DIRS for which in ("reference", "wrong")]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda job: run(*job), jobs))
    return {(d.name, which): r for (d, which), r in zip(jobs, results)}


def test_suite_has_tasks():
    assert TASK_DIRS, f"no task directories under {ROOT}"
    ids = [load(d)["id"] for d in TASK_DIRS]
    assert len(ids) == len(set(ids)), "task ids must be unique"


@pytest.mark.parametrize("task_dir", TASK_DIRS, ids=lambda p: p.name)
def test_task_files_are_complete(task_dir):
    spec = load(task_dir)
    assert REQUIRED <= set(spec), f"missing keys: {sorted(REQUIRED - set(spec))}"
    assert spec["id"] == task_dir.name
    assert spec["task"] == "coding" and spec["kind"] == "code"
    assert spec["level"] in MIN_TESTS and spec["id"].startswith(f"c{spec['level']}-")
    assert ("prompt" in spec) != ("prompt_file" in spec), "give exactly one of prompt and prompt_file"
    assert isinstance(spec["code"], dict) and "tests" in spec["code"]
    assert isinstance(spec["code"].get("forbid", []), list)
    for rel in [spec.get("prompt_file"), spec["code"]["tests"], spec["reference"], spec["wrong"]]:
        assert rel is None or (task_dir / rel).is_file(), f"{rel} doesn't exist"
    assert "```python" in prompt_of(task_dir, spec), "the prompt must ask for one ```python code block"


@pytest.mark.parametrize("task_dir", TASK_DIRS, ids=lambda p: p.name)
def test_reference_passes_every_test(task_dir, outcomes):
    r = outcomes[(task_dir.name, "reference")]
    assert r["error"] is None, r["error"]
    assert r["failed"] == [], r["failed"]
    assert r["passed"] == r["total"] >= MIN_TESTS[load(task_dir)["level"]], r


@pytest.mark.parametrize("task_dir", TASK_DIRS, ids=lambda p: p.name)
def test_wrong_solution_fails(task_dir, outcomes):
    r = outcomes[(task_dir.name, "wrong")]
    assert r["error"] is not None or r["failed"], "wrong.py passes every test: the tests miss its bug"


@pytest.mark.parametrize("task_dir", TASK_DIRS, ids=lambda p: p.name)
def test_reference_respects_forbid(task_dir):
    spec = load(task_dir)
    banned = set(spec["code"].get("forbid") or [])
    assert forbidden_imports((task_dir / spec["reference"]).read_text(encoding="utf-8"), banned) == []
