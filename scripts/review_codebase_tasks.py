"""Check each codebase-suite issue against the tests that grade it, with a model from another family than the
one the issues were drafted with, so no model grades work in its own style (the default reviewer is GPT).

    uv run python scripts/review_codebase_tasks.py --gateway http://localhost:8080 --key $KEY [--model gpt-sol] [task ...]

For each task the reviewer gets the issue, the full source of every fail_to_pass test (and the test diff, for the
helpers they use), and the commit's own change. It returns, as JSON, the assertions a careful engineer couldn't
satisfy from the issue and the repository alone, and the places where the issue gives the solution away. Results
go to stdout and, with --out, a JSON file. It asks before spending: the cost estimate comes from the gateway's
prices.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import re
import subprocess
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from meritorder.evals.envs import codebase  # noqa: E402
from meritorder.evals.runner import load_task_dir  # noqa: E402

INSTRUCTIONS = """You review one task of a coding eval. A model under evaluation gets a repository as it was before a real
commit, plus the ISSUE below, and must implement the change. Then the commit's own tests (below) grade it: every
one must pass. You see what the evaluated model never sees: the tests and the commit's own change.

Judge two things, strictly:

1. Underspecified: for every assert in every test, could a strong engineer who reads only the issue and the
   repository (as it was before the commit) write code that passes it? List each assertion that depends on
   something the issue doesn't state and the existing code wouldn't suggest: an exact string, number, order,
   name, signature, default, or behaviour. Things the repository's existing conventions make clear, or
   documented public formats (such as a provider's published API) a strong engineer knows, are fine.
2. Gives away: list each place where the issue tells how to implement the change rather than what it must do:
   files or internal functions to edit, helpers to reuse, data structures, the reference's structure, or a
   restatement of the diff. Names the tests call directly are fine.

Then a verdict: "fair" (nothing material on either list), "fix" (fixable by editing the issue), or "drop" (the
tests check details no fair issue could state without dictating the implementation).

Reply with only a JSON object:
{"underspecified": [{"test": "...", "assertion": "...", "missing": "what the issue would need to say"}],
 "gives_away": [{"quote": "...", "why": "..."}],
 "verdict": "fair" | "fix" | "drop",
 "notes": "one or two sentences"}"""


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, check=True, text=True).stdout


def test_sources(tests: dict[str, str], ids: list[str]) -> str:
    """The source of each test function the ids name (parametrized ids once)."""
    out, seen = [], set()
    for tid in ids:
        path, _, name = tid.partition("::")
        name = re.sub(r"\[.*\]$", "", name).split("::")[-1]
        if (path, name) in seen or path not in tests:
            continue
        seen.add((path, name))
        src = tests[path]
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                decorators = "".join(f"@{ast.get_source_segment(src, d)}\n" for d in node.decorator_list)
                out.append(f"# {path}\n{decorators}{ast.get_source_segment(src, node)}")
                break
    return "\n\n".join(out)


def review_prompt(t: dict) -> str:
    d = codebase.load_task(t["id"])
    issue = t["prompt"].split("## Issue", 1)[-1]
    reference = git("diff", d["base"], d["commit"], "--", "src", "config")
    test_diff = git("diff", d["base"], d["commit"], "--", "tests")
    clip = 60_000
    return (f"<issue>\n## Issue{issue}\n</issue>\n\n<tests_that_grade_it>\n{test_sources(d['tests'], d['fail_to_pass'])}\n"
            f"</tests_that_grade_it>\n\n<test_diff_for_context>\n{test_diff[:clip]}\n</test_diff_for_context>\n\n"
            f"<the_commits_own_change>\n{reference[:clip]}\n</the_commits_own_change>")


async def review(client: httpx.AsyncClient, gateway: str, key: str, model: str, t: dict) -> dict:
    body = {"model": model, "reasoning_effort": "high", "messages": [
        {"role": "system", "content": INSTRUCTIONS}, {"role": "user", "content": review_prompt(t)}]}
    r = await client.post(f"{gateway}/v1/chat/completions", json=body, headers={"authorization": f"Bearer {key}"})
    cost = float(r.headers.get("x-meritorder-cost-usd") or 0)
    if r.status_code != 200:
        return {"task": t["id"], "error": f"HTTP {r.status_code}: {r.text[:300]}", "cost": cost}
    text = (r.json()["choices"][0]["message"].get("content") or "")
    m = re.search(r"\{.*\}", text, re.S)
    try:
        verdict = json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        verdict = None
    if not isinstance(verdict, dict):
        return {"task": t["id"], "error": "no JSON verdict", "raw": text[:2000], "cost": cost}
    return {"task": t["id"], **verdict, "cost": cost}


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tasks", nargs="*")
    ap.add_argument("--gateway", default="http://localhost:8080")
    ap.add_argument("--key", required=True)
    ap.add_argument("--model", default="gpt-sol")
    ap.add_argument("--out")
    ap.add_argument("--yes", action="store_true", help="don't ask before spending")
    args = ap.parse_args()
    tasks = []
    for d in sorted((ROOT / "src/meritorder/evals/suites/codebase").glob("*/task.yaml")):
        t = load_task_dir(d.parent)
        if args.tasks and d.parent.name not in args.tasks:
            continue
        if not t.get("prompt"):
            print(f"skipping {d.parent.name}: no issue yet")
            continue
        tasks.append(t)
    chars = sum(len(INSTRUCTIONS) + len(review_prompt(t)) for t in tasks)
    tokens_in, tokens_out = chars // 4, 8_000 * len(tasks)  # a reviewer reasons at high effort: allow ~8k output tokens a task
    usd = None
    try:
        view = httpx.get(f"{args.gateway.rstrip('/')}/v1/admin/config", headers={"authorization": f"Bearer {args.key}"}, timeout=30).json()
        prices = [r["price"] for m in view["models"] if m["name"] == args.model for r in m.get("routes", []) if r.get("live") and r.get("price")]
        if prices:
            p = min(prices, key=lambda x: x["input"])
            usd = (tokens_in * p["input"] + tokens_out * p["output"]) / 1e6
    except (httpx.HTTPError, KeyError, ValueError):
        pass
    print(f"{len(tasks)} tasks for {args.model}: about {tokens_in:,} input tokens"
          + (f", about ${usd:,.2f} (a rough guide: it can cost several times that)" if usd is not None else ", price unknown"))
    if not args.yes and input("This spends real money. Run it? [y/N] ").strip().lower() != "y":
        return
    async with httpx.AsyncClient(timeout=httpx.Timeout(None, connect=30)) as c:
        sem = asyncio.Semaphore(4)

        async def one(t):
            async with sem:
                res = await review(c, args.gateway.rstrip("/"), args.key, args.model, t)
                print(f"{res['task']:26} {res.get('verdict', 'ERROR'):5} under={len(res.get('underspecified') or [])} "
                      f"gives_away={len(res.get('gives_away') or [])} ${res.get('cost', 0):.3f}", flush=True)
                return res
        results = await asyncio.gather(*[one(t) for t in tasks])
    print(f"total ${sum(r.get('cost', 0) for r in results):.2f}")
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
