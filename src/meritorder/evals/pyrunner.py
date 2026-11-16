"""Runs a code task's hidden tests against a model's solution. Copied into the sandbox and run there as a script:

    python -I -B _run.py [--forbid re,regex]

The working directory holds ``solution.py`` (the model's code) and ``tests.py`` (the task's tests, which import
from ``solution``). Every top-level ``test_*`` function runs on its own, in the main thread; one failing doesn't stop
the others, and one stuck in a loop is stopped after ``--per-test`` seconds of CPU time (a CPU timer, so tests
that use the wall-clock timer themselves are unaffected).
Prints one JSON line: {"total": n, "passed": n, "failed": [{"name": ..., "error": ...}], "error": null | "..."}.
Standard library only: it runs in a bare python image with no network.
"""

import ast
import json
import signal
import sys
import traceback


class TestTimeout(Exception):
    pass


def _too_slow(signum, frame):
    raise TestTimeout("took too long (CPU time limit per test)")


def forbidden_imports(source: str, banned: set[str]) -> list[str]:
    found = []
    for node in ast.walk(ast.parse(source)):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        elif isinstance(node, ast.Call) and getattr(node.func, "id", None) == "__import__":
            names = [a.value for a in node.args[:1] if isinstance(a, ast.Constant) and isinstance(a.value, str)]
        found += [n for n in names if n.split(".")[0] in banned]
    return found


def main(argv: list[str]) -> dict:
    banned = set()
    if "--forbid" in argv:
        banned = {m for m in argv[argv.index("--forbid") + 1].split(",") if m}
    per_test = float(argv[argv.index("--per-test") + 1]) if "--per-test" in argv else 0.0
    try:
        source = open("solution.py", encoding="utf-8").read()
        bad = forbidden_imports(source, banned) if banned else []
    except SyntaxError as e:
        return {"total": 0, "passed": 0, "failed": [], "error": f"solution doesn't parse: {e}"}
    if bad:
        return {"total": 0, "passed": 0, "failed": [], "error": f"solution imports what the task forbids: {', '.join(sorted(set(bad)))}"}
    sys.path.insert(0, ".")
    for name in ("tests", "solution"):  # a fresh process in the sandbox; this matters only when run in-process
        sys.modules.pop(name, None)
    try:
        import tests  # noqa: PLC0415 (imports solution)
    except BaseException as e:  # noqa: BLE001 (a broken solution can raise anything, even SystemExit)
        return {"total": 0, "passed": 0, "failed": [], "error": f"loading the solution failed: {type(e).__name__}: {e}"[:2000]}
    names = sorted(n for n in dir(tests) if n.startswith("test_") and callable(getattr(tests, n)))
    failed = []
    timer = per_test > 0 and hasattr(signal, "ITIMER_VIRTUAL")
    if timer:
        signal.signal(signal.SIGVTALRM, _too_slow)
    for name in names:
        try:
            if timer:
                signal.setitimer(signal.ITIMER_VIRTUAL, per_test)
            getattr(tests, name)()
        except BaseException as e:  # noqa: BLE001
            tb = traceback.extract_tb(e.__traceback__)
            where = f" (line {tb[-1].lineno})" if tb else ""
            failed.append({"name": name, "error": f"{type(e).__name__}: {e}{where}"[:500]})
        finally:
            if timer:
                signal.setitimer(signal.ITIMER_VIRTUAL, 0)
    return {"total": len(names), "passed": len(names) - len(failed), "failed": failed, "error": None}


if __name__ == "__main__":
    print(json.dumps(main(sys.argv[1:])))
