"""The frontier agent tasks check themselves: each environment is built from its task.yaml, a scripted oracle (the
right sequence of tool calls) must pass, and plausible careless policies must fail. Repository tasks run their tests
through ``LocalSandbox``; that's fine here because only the oracle's and these tests' own trusted code runs."""

from __future__ import annotations

import datetime as dt
import json
import re
import sqlite3
from decimal import Decimal
from pathlib import Path

import pytest
import yaml

import meritorder.evals
from meritorder.evals.envs import Env, load_env, repo, sqldb
from meritorder.evals.envs.incident import IncidentEnv
from meritorder.evals.envs.procurement import ProcurementEnv
from meritorder.evals.envs.repo import RepoEnv, RepoQAEnv, count_tests, load_project, norm_path
from meritorder.evals.envs.sqldb import SqlEnv, parse_amount
from meritorder.evals.sandbox import LocalSandbox, SandboxResult

ROOT = Path(meritorder.evals.__file__).parent / "suites" / "frontier" / "agent"
TASKS = ["a3-retry-trace", "a4-slot-merge", "a4-charger-order", "a5-oncall-handoff", "a5-march-revenue", "a5-checkout-incident"]
REQUIRED = {"id", "task", "level", "kind", "prompt", "env", "env_args", "max_turns", "summary"}


def spec(task_id: str) -> dict:
    return yaml.safe_load((ROOT / task_id / "task.yaml").read_text(encoding="utf-8"))


def build(task_id: str, sandbox=None) -> Env:
    s = spec(task_id)
    return load_env(s["env"])(sandbox=sandbox, **(s.get("env_args") or {}))


def J(text: str) -> dict:
    return json.loads(text)


def edit(env: Env, path: str, *replacements: tuple[str, str]) -> None:
    """Read a file through the tools, apply exact replacements, write it back."""
    text = env.handle("read_file", {"path": path})
    for old, new in replacements:
        assert old in text, (path, old)
        text = text.replace(old, new)
    assert env.handle("write_file", {"path": path, "content": text}).startswith("wrote")


# --------------------------------------------------------------------------------------------- the task files
def test_task_files():
    found = sorted(p.name for p in ROOT.iterdir() if (p / "task.yaml").is_file())
    assert found == sorted(TASKS)
    levels = []
    for task_id in TASKS:
        s = spec(task_id)
        assert set(s) >= REQUIRED, f"{task_id}: missing {sorted(REQUIRED - set(s))}"
        assert s["id"] == task_id and s["id"].startswith(f"a{s['level']}-") and s["kind"] == "agent"
        assert s["task"] in ("coding", "general") and s["prompt"].strip() and s["summary"].strip()
        env = build(task_id)
        names = [t["function"]["name"] for t in env.tools]
        assert len(names) == len(set(names)) and names
        for t in env.tools:
            params = t["function"]["parameters"]
            assert t["type"] == "function" and params["type"] == "object" and set(params["required"]) <= set(params["properties"])
        levels.append((s["level"], s["task"]))
    assert sorted(levels) == [(3, "coding"), (4, "coding"), (4, "general"), (5, "coding"), (5, "general"), (5, "general")]


@pytest.mark.parametrize("task_id", TASKS)
def test_handle_errors(task_id):
    env = build(task_id)
    assert "unknown tool" in J(env.handle("rm_rf", "{}"))["error"]
    assert "not valid JSON" in J(env.handle(env.tools[0]["function"]["name"], "{nope"))["error"]
    assert isinstance(env.handle(env.tools[0]["function"]["name"], "[1, 2]"), str)  # JSON, but not an object
    assert isinstance(env.handle(env.tools[0]["function"]["name"], {"self": 1, "name": 2}), str)
    for t in env.tools:  # every tool copes with junk arguments without raising
        out = env.handle(t["function"]["name"], {k: [1] for k in t["function"]["parameters"]["properties"]})
        assert isinstance(out, str)
    assert len(env.calls) == 4 + len(env.tools)


def test_env_loading_and_crashes():
    with pytest.raises(ValueError):
        load_env("meritorder.evals.envs.repo:REPOS")

    class Broken(RepoQAEnv):
        def call(self, name, arguments):
            raise RuntimeError("bug in the environment")

    out = J(Broken(project="reconcile").handle("list_files", None))
    assert out == {"error": "RuntimeError: bug in the environment"}


# --------------------------------------------------------------------------------------------- a3: repository question
def test_a3_oracle_and_careless():
    env = build("a3-retry-trace")
    files = env.handle("list_files", {}).splitlines()
    assert "reconcile/retry.py" in files and not any(f.startswith("_hidden") for f in files)
    hits = env.handle("search", {"text": "retry_policy("})
    assert "reconcile/jobs.py" in hits and "reconcile/config.py" in hits
    jobs = env.handle("read_file", {"path": "reconcile/jobs.py"})
    assert 'retry_policy("bankfeed")' in jobs
    config = env.handle("read_file", {"path": "./reconcile/config.py"})
    assert "Entries are\n        not combined" in config
    profiles = env.handle("read_file", {"path": "/reconcile/profiles.py"})
    assert '"bankfeed": {"max_retries": 4, "base_delay": 1.5}' in profiles
    retry = env.handle("read_file", {"path": "reconcile/retry.py"})
    factor = float(re.search(r"factor: float = ([\d.]+)", retry).group(1))
    max_delay = float(re.search(r"max_delay: float = ([\d.]+)", retry).group(1))
    waits = [min(1.5 * factor**n, max_delay) for n in range(4)]
    ok, why = env.grade(f"The production bankfeed entry wins.\nattempts={len(waits) + 1} final_wait={waits[-1]}")
    assert ok, why
    assert env.grade("attempts = 5, final_wait: 40.50 s")[0]

    docs = build("a3-retry-trace")  # believes docs/operations.md
    docs.handle("read_file", {"path": "docs/operations.md"})
    assert not docs.grade("attempts=4 final_wait=4")[0]
    inherits = build("a3-retry-trace")  # assumes the bankfeed entry inherits factor/max_delay from "default"
    ok, why = inherits.grade("attempts=5 final_wait=12")
    assert not ok and any("final_wait" in w for w in why)
    assert not inherits.grade("attempts=4 final_wait=40.5")[0]
    assert inherits.grade("attempts=5 final_wait=40.5")[1] == []
    assert any("no attempts" in w for w in inherits.grade("I don't know")[1])


def test_repo_tools():
    env = build("a3-retry-trace")
    assert env.handle("list_files", {"directory": "reconcile/legacy"}).splitlines() == [
        "reconcile/legacy/__init__.py", "reconcile/legacy/backoff.py", "reconcile/legacy/importer.py"]
    assert "no files under" in J(env.handle("list_files", {"directory": "nowhere"}))["error"]
    assert "not a directory" in J(env.handle("list_files", {"directory": "../etc"}))["error"]
    assert "is a directory" in J(env.handle("read_file", {"path": "reconcile"}))["error"]
    assert "no such file" in J(env.handle("read_file", {"path": "reconcile/nope.py"}))["error"]
    assert "relative to the repository root" in J(env.handle("read_file", {"path": "../../etc/passwd"}))["error"]
    assert "relative" in J(env.handle("read_file", {}))["error"]
    assert env.handle("search", {"text": "zzz-not-there"}).startswith("no matches")
    assert "non-empty" in J(env.handle("search", {"text": "  "}))["error"]
    assert "no files under" in J(env.handle("search", {"text": "x", "directory": "nowhere"}))["error"]
    assert "more matches" in env.handle("search", {"text": "e"})
    assert env.handle("search", {"text": "DOUBLES", "directory": "reconcile/legacy"}).startswith("reconcile/legacy/backoff.py:")
    assert "write_file" not in {t["function"]["name"] for t in env.tools}
    assert "unknown tool" in J(env.handle("write_file", {"path": "x", "content": "y"}))["error"]
    assert norm_path("a/./b/../c.py") == "a/c.py" and norm_path(3) is None and norm_path("  ") is None
    long = build("a3-retry-trace")
    long.files["big.txt"] = "x" * (repo.MAX_RESULT + 5)
    assert "more characters" in long.handle("read_file", {"path": "big.txt"})
    with pytest.raises(ValueError):
        RepoQAEnv(project="../repos")
    with pytest.raises(ValueError):
        load_project("missing")


# --------------------------------------------------------------------------------------------- a4: slotfinder fix
def oracle_slotfinder(env: RepoEnv) -> str:
    issue = env.handle("read_file", {"path": "ISSUE.md"})
    assert "busy_summary" in issue
    for path in ("slotfinder/availability.py", "slotfinder/summary.py", "slotfinder/calendar.py"):
        assert "merge" in env.handle("read_file", {"path": path})
    before = env.handle("run_tests", {})
    assert "FAILED (failures=2)" in before and "test_meeting_inside_all_hands" in before
    edit(env, "slotfinder/intervals.py",
         ("if result and iv.start < result[-1].end:", "if result and iv.start <= result[-1].end:"),
         ("Interval(result[-1].start, iv.end)", "Interval(result[-1].start, max(result[-1].end, iv.end))"))
    assert env.handle("run_tests", {}).rstrip().endswith("OK")
    return "merge() now merges touching intervals and keeps the later end of contained ones."


def test_a4_slotfinder():
    env = build("a4-slot-merge", LocalSandbox())
    assert env.expected_hidden == count_tests(env.hidden) == 20
    answer = oracle_slotfinder(env)
    assert env.changed() == ["slotfinder/intervals.py"]
    ok, why = env.grade(answer)
    assert ok, why

    # Careless: patches both symptoms where they show up; the visible tests pass, merge() stays broken.
    local = build("a4-slot-merge", LocalSandbox())
    edit(local, "slotfinder/availability.py",
         ("    for block in merge(clip(busy, day_start, day_end)):\n        if block.start > cursor:\n"
          "            free.append(Interval(cursor, block.start))\n        cursor = block.end\n",
          "    for block in sorted(clip(busy, day_start, day_end)):\n        if block.start > cursor:\n"
          "            free.append(Interval(cursor, block.start))\n        cursor = max(cursor, block.end)\n"))
    edit(local, "slotfinder/summary.py",
         ("    blocks = merge(busy)\n",
          "    blocks = []\n    for b in sorted(busy):\n        if blocks and b.start <= blocks[-1].end:\n"
          "            blocks[-1] = Interval(blocks[-1].start, max(blocks[-1].end, b.end))\n        else:\n"
          "            blocks.append(b)\n"),
         ("total(busy)", "sum(b.length for b in blocks)"))
    assert local.handle("run_tests", {}).rstrip().endswith("OK")
    ok, why = local.grade("fixed both reports")
    assert not ok and any("MergeTests" in w for w in why)

    # Careless: fixes only the comparison (touching intervals), doesn't rerun the tests.
    half = build("a4-slot-merge", LocalSandbox())
    edit(half, "slotfinder/intervals.py", ("iv.start < result[-1].end", "iv.start <= result[-1].end"))
    assert not half.grade("done")[0]

    # Careless: edits the tests instead of the code.
    cheat = build("a4-slot-merge", LocalSandbox())
    cheat.handle("write_file", {"path": "tests/test_slotfinder.py", "content": "import unittest\n"})
    assert not cheat.grade("tests pass now")[0]


def test_repo_env_write_and_sandbox_paths(monkeypatch):
    env = build("a4-slot-merge")
    assert "no sandbox" in J(env.handle("run_tests", {}))["error"]
    assert env.grade("x") == (False, ["no sandbox to run the hidden tests"])
    assert "relative" in J(env.handle("write_file", {"path": "../x.py", "content": ""}))["error"]
    assert "must be a string" in J(env.handle("write_file", {"path": "x.py", "content": 3}))["error"]
    assert "too long" in J(env.handle("write_file", {"path": "x.py", "content": "x" * (repo.MAX_WRITE + 1)}))["error"]
    assert "is a directory" in J(env.handle("write_file", {"path": "slotfinder", "content": ""}))["error"]
    assert "reserved" in J(env.handle("write_file", {"path": "_hidden/test_x.py", "content": ""}))["error"]
    assert env.handle("write_file", {"path": "notes/plan.md", "content": "a\nb\n"}) == "created notes/plan.md (2 lines)"
    assert "notes/plan.md" in env.handle("list_files", {"directory": "notes"})
    monkeypatch.setattr(repo, "MAX_FILES", len(env.files))
    assert "too many files" in J(env.handle("write_file", {"path": "more.py", "content": ""}))["error"]

    class Fake:
        def __init__(self, result):
            self.result, self.argv = result, None

        def run(self, files, argv, timeout=30.0):
            self.argv = argv
            return self.result(argv) if callable(self.result) else self.result

    env.sandbox = Fake(SandboxResult(-1, "", "", timed_out=True))
    assert "timed out" in env.handle("run_tests", {})
    assert "timed out" in env.grade("x")[1][0]
    env.sandbox = Fake(SandboxResult(1, "", "Traceback: boom"))
    assert "crashed" in env.grade("x")[1][0] and "boom" in env.grade("x")[1][0]
    env.sandbox = Fake(lambda argv: SandboxResult(0, argv[-1] + json.dumps({"total": 3, "failed": [], "error": None}), ""))
    assert env.grade("x") == (False, [f"ran 3 hidden tests, expected {env.expected_hidden}"])
    env.sandbox = Fake(lambda argv: SandboxResult(0, argv[-1] + json.dumps({"total": 0, "failed": [], "error": "no tests"}), ""))
    assert env.grade("x") == (False, ["no tests"])
    env.sandbox = Fake(SandboxResult(1, "x" * (repo.MAX_RESULT + 10), ""))
    assert env.handle("run_tests", {}).startswith("[... earlier output cut]")


# --------------------------------------------------------------------------------------------- a5: on-call fix + feature
def oracle_oncall(env: RepoEnv) -> str:
    for path in env.handle("list_files", {}).splitlines():
        if path.endswith((".py", ".md")):
            env.handle("read_file", {"path": path})
    before = env.handle("run_tests", {})
    assert "FAILED (failures=2, errors=1)" in before
    edit(env, "oncall/timeutil.py", ("ts - timedelta(days=ts.isoweekday() % 7)", "ts - timedelta(days=ts.weekday())"))
    after_first_fix = env.handle("run_tests", {})
    assert "FAIL: test_sunday_counts_in_its_own_week" in after_first_fix  # the report's compensating hack now breaks it
    edit(env, "oncall/report.py",
         ("from datetime import datetime, timedelta\n", "from datetime import datetime\n"),
         ("        # week_start() gives the Sunday before; step forward a day to get the Monday.\n"
          "        monday = week_start(hour) + timedelta(days=1)\n", "        monday = week_start(hour)\n"))
    edit(env, "oncall/rotation.py",
         ("length_days: int = 7):", "length_days: int = 7, handoff_hour: int = 0):"),
         ("        self.first_shift = week_start(start)\n",
          "        if isinstance(handoff_hour, bool) or not isinstance(handoff_hour, int) or not 0 <= handoff_hour <= 23:\n"
          "            raise ValueError(f\"handoff_hour must be an integer from 0 to 23, got {handoff_hour!r}\")\n"
          "        self.handoff_hour = handoff_hour\n"
          "        self.first_shift = week_start(start) + timedelta(hours=handoff_hour)\n"))
    edit(env, "oncall/config.py",
         ('length_days=_int_field(data, "length_days", 7, 1, 28))',
          'length_days=_int_field(data, "length_days", 7, 1, 28),\n'
          '                    handoff_hour=_int_field(data, "handoff_hour", 0, 0, 23))'))
    assert env.handle("run_tests", {}).rstrip().endswith("OK")
    return "Fixed week_start (it returned Sunday), removed report.py's +1 day workaround, added handoff_hour."


def test_a5_oncall():
    env = build("a5-oncall-handoff", LocalSandbox())
    assert len([f for f in env.files if f.endswith(".py")]) >= 8 and env.expected_hidden == 21
    answer = oracle_oncall(env)
    assert len(env.calls) >= 15
    ok, why = env.grade(answer)
    assert ok, why

    # Careless: takes the ISSUE's suggested patch (+1 day in Rotation), patches report.py, implements the feature.
    # Every visible test passes; week_start is still wrong, and so is a rotation that starts on a Sunday.
    suggested = build("a5-oncall-handoff", LocalSandbox())
    edit(suggested, "oncall/report.py", ("monday = week_start(hour) + timedelta(days=1)",
                                         "monday = (hour - timedelta(days=hour.weekday())).replace(hour=0)"))
    edit(suggested, "oncall/rotation.py",
         ("length_days: int = 7):", "length_days: int = 7, handoff_hour: int = 0):"),
         ("        self.first_shift = week_start(start)\n",
          "        if not isinstance(handoff_hour, int) or not 0 <= handoff_hour <= 23:\n"
          "            raise ValueError(\"handoff_hour\")\n"
          "        self.first_shift = week_start(start) + timedelta(days=1, hours=handoff_hour)\n"))
    edit(suggested, "oncall/config.py", ('length_days=_int_field(data, "length_days", 7, 1, 28))',
                                         'length_days=_int_field(data, "length_days", 7, 1, 28), '
                                         'handoff_hour=_int_field(data, "handoff_hour", 0, 0, 23))'))
    assert suggested.handle("run_tests", {}).rstrip().endswith("OK")
    ok, why = suggested.grade("done")
    assert not ok and any("WeekStartTests" in w for w in why) and any("on_sunday" in w for w in why)

    # Careless: fixes week_start and adds the feature but never reruns the tests (report.py's workaround remains),
    # and implements handoff_hour by shifting calendar weeks too.
    norerun = build("a5-oncall-handoff", LocalSandbox())
    edit(norerun, "oncall/timeutil.py", ("ts - timedelta(days=ts.isoweekday() % 7)", "ts - timedelta(days=ts.weekday())"))
    edit(norerun, "oncall/rotation.py",
         ("length_days: int = 7):", "length_days: int = 7, handoff_hour: int = 0):"),
         ("        self.first_shift = week_start(start)\n", "        self.first_shift = week_start(start) + timedelta(hours=handoff_hour)\n"))
    ok, why = norerun.grade("done")
    assert not ok and any("ReportTests" in w for w in why) and any("test_invalid" in w for w in why)


# --------------------------------------------------------------------------------------------- a4: procurement
def oracle_procurement(env: ProcurementEnv) -> str:
    results = J(env.handle("search_catalog", {"query": "USB-C charger 100W"}))["results"]
    candidates = []
    for r in results:
        offer = J(env.handle("get_offer", {"offer_id": r["offer_id"]}))
        if offer["condition"] == "new" and offer["specs"].get("output_watts", 0) >= 100:
            candidates.append(offer["offer_id"])
    address = next(a["id"] for a in J(env.handle("list_addresses", {}))["addresses"] if a["label"].endswith("Porto office"))
    quotes = []
    for oid in candidates:
        q = J(env.handle("get_quote", {"offer_id": oid, "quantity": 12, "address_id": address}))
        if "error" not in q and q["estimated_delivery"] <= "2026-11-20":
            quotes.append(q)
    best = min(quotes, key=lambda q: q["total_eur"])
    po = J(env.handle("get_purchase_order", {"po_number": "PO-4471"}))
    assert best["total_eur"] <= po["remaining_eur"]
    placed = J(env.handle("place_order", {"offer_id": best["offer_id"], "quantity": 12, "address_id": address,
                                          "po_number": "PO-4471", "cost_center": po["cost_center"]}))
    assert "504" in placed["error"]  # ambiguous: check before retrying
    orders = J(env.handle("list_orders", {"po_number": "PO-4471"}))["orders"]
    assert len(orders) == 1
    return f"Placed order {orders[0]['order_id']} for 12 x {best['offer_id']}: total EUR {orders[0]['total_eur']:.2f}."


def test_a4_procurement_oracle():
    env = build("a4-charger-order")
    answer = oracle_procurement(env)
    assert "ORD-88412" in answer and "384.00" in answer
    ok, why = env.grade(answer)
    assert ok, why
    assert env.grade("Order ORD-88412, EUR 384,00 in total")[0] and env.grade("ord-88412 for €384")[0]
    assert not env.grade("Order ORD-88412, total EUR 1384.00")[0]
    assert not env.grade("Order placed, EUR 384.00")[0]


def test_a4_procurement_careless():
    # Takes the cheapest headline price, gets rejected for the cost center, fixes that, then retries after the 504.
    cheap = build("a4-charger-order")
    top = J(cheap.handle("search_catalog", {"query": "usb-c charger"}))["results"][0]
    assert top["offer_id"] == "OF-733"
    order = {"offer_id": "OF-733", "quantity": 12, "address_id": "ADDR-OPO-1", "po_number": "PO-4471"}
    assert J(cheap.handle("place_order", order))["code"] == "COST_CENTER_REQUIRED"
    order["cost_center"] = J(cheap.handle("get_purchase_order", {"po_number": "po-4471"}))["cost_center"]
    assert "504" in J(cheap.handle("place_order", order))["error"]
    second = J(cheap.handle("place_order", order))
    ok, why = cheap.grade(f"Ordered: {second['order_id']}, EUR {second['total_eur']}")
    assert not ok and "2 were placed" in why[0] and any("OF-733" in w for w in why)

    # Right offer, but retries after the timeout without checking: a duplicate.
    dup = build("a4-charger-order")
    order = {"offer_id": "OF-311", "quantity": 12, "address_id": "ADDR-OPO-1", "po_number": "PO-4471", "cost_center": "CC-OPS-210"}
    dup.handle("place_order", order)
    again = J(dup.handle("place_order", order))
    assert again["order_id"] == "ORD-88413" and not dup.grade("Order ORD-88413, EUR 384.00")[0]

    # Compares headline price + shipping and skips the fees: Halcyon looks cheapest.
    fees = build("a4-charger-order")
    assert J(fees.handle("get_quote", {"offer_id": "OF-609", "quantity": 12, "address_id": "ADDR-OPO-1"}))["total_eur"] == 387.30
    fees.handle("place_order", {**order, "offer_id": "OF-609"})
    assert not fees.grade("Order ORD-88412, EUR 387.30")[0]

    # Doesn't check delivery dates: Northwind is cheaper but arrives late.
    late = build("a4-charger-order")
    q = J(late.handle("get_quote", {"offer_id": "OF-407", "quantity": 12, "address_id": "ADDR-OPO-1"}))
    assert (q["total_eur"], q["estimated_delivery"]) == (364.0, "2026-11-27")
    late.handle("place_order", {**order, "offer_id": "OF-407"})
    assert not late.grade("ORD-88412 EUR 364.00")[0]
    assert not build("a4-charger-order").grade("I placed nothing")[0]


def test_procurement_errors():
    env = build("a4-charger-order")
    assert "non-empty" in J(env.handle("search_catalog", {"query": " "}))["error"]
    assert "no searchable" in J(env.handle("search_catalog", {"query": "a b"}))["error"]
    assert J(env.handle("search_catalog", {"query": "espresso machine"}))["results"] == []
    assert J(env.handle("search_catalog", {"query": "laptop stand"}))["results"][0]["offer_id"] == "OF-861"
    assert "no offer" in J(env.handle("get_offer", {"offer_id": "OF-1"}))["error"]
    detail = J(env.handle("get_offer", {"offer_id": "of-311"}))
    assert "free on orders over" in detail["shipping"] and detail["fees"] == ["none"]
    assert J(env.handle("get_offer", {"offer_id": "OF-118"}))["fees"][0].startswith("Import handling")
    quote = {"offer_id": "OF-311", "quantity": 12, "address_id": "ADDR-OPO-1"}
    assert "integer" in J(env.handle("get_quote", {**quote, "quantity": "12"}))["error"]
    assert "integer" in J(env.handle("get_quote", {**quote, "quantity": True}))["error"]
    assert "between" in J(env.handle("get_quote", {**quote, "quantity": 0}))["error"]
    assert "packs of 10" in J(env.handle("get_quote", {**quote, "offer_id": "OF-512"}))["error"]
    assert J(env.handle("get_quote", {**quote, "offer_id": "OF-512", "quantity": 20}))["total_eur"] == 570.0
    assert J(env.handle("get_quote", {**quote, "quantity": 16}))["shipping_eur"] == 12.0  # 496 EUR
    assert J(env.handle("get_quote", {**quote, "quantity": 61}))["estimated_delivery"] == "2026-11-30"  # more than in stock
    assert J(env.handle("get_quote", {**quote, "quantity": 17}))["shipping_eur"] == 0.0  # 527 EUR: free shipping
    assert "unknown address" in J(env.handle("get_quote", {**quote, "address_id": "Porto office"}))["error"]
    assert "no offer" in J(env.handle("get_quote", {**quote, "offer_id": None}))["error"]
    assert "no purchase order" in J(env.handle("get_purchase_order", {"po_number": "PO-1"}))["error"]
    order = {**quote, "po_number": "PO-4471", "cost_center": "CC-OPS-210"}
    for bad, msg in (({"offer_id": "x"}, "no offer"), ({"quantity": -1}, "between"), ({"address_id": "x"}, "unknown address"),
                     ({"po_number": "PO-9"}, "no purchase order"), ({"cost_center": "CC-ENG-115"}, "cost_center")):
        assert msg in J(env.handle("place_order", {**order, **bad}))["error"]
    assert J(env.handle("place_order", {**order, "offer_id": "OF-118"}))["code"] == "OVER_BUDGET"
    assert env.orders == [] and J(env.handle("list_orders", {}))["orders"] == []
    env.handle("place_order", {**order, "po_number": "PO-4302", "cost_center": "CC-ENG-115"})
    assert J(env.handle("list_orders", {"po_number": "PO-4471"}))["orders"] == []
    assert len(J(env.handle("list_orders", {}))["orders"]) == 1
    ok, why = env.grade("ORD-88412 384.00")
    assert not ok and any("po_number" in w for w in why)
    with pytest.raises(ValueError):
        ProcurementEnv(scenario="../chargers")


# --------------------------------------------------------------------------------------------- a5: SQL analysis
ORACLE_SQL = """
WITH raw AS (
  SELECT provider_ref AS ref, order_id, datetime(event_at) AS at, amount_cents / 100.0 AS amount,
         COALESCE(currency, 'EUR') AS cur, lower(status) AS status FROM payments
  UNION ALL
  SELECT provider_ref, order_id, datetime(event_at), amount, currency, lower(status) FROM payments_v2
),
pay AS (
  SELECT ref, MIN(order_id) AS order_id, MIN(at) AS at, MIN(amount) AS amount, MIN(cur) AS cur, MIN(status) AS status
  FROM raw GROUP BY ref
),
real AS (
  SELECT pay.* FROM pay JOIN orders o ON o.id = pay.order_id LEFT JOIN customers c ON c.id = o.customer_id
  WHERE c.is_test IS NOT 1
),
revenue AS (
  SELECT SUM(amount * CASE WHEN cur = 'EUR' THEN 1.0 ELSE (SELECT eur_per_unit FROM fx_rates f
         WHERE f.currency = real.cur AND f.day <= date(real.at) ORDER BY f.day DESC LIMIT 1) END) AS v
  FROM real WHERE status = 'captured'
    AND strftime('%Y-%m', CASE WHEN at >= '2026-03-29 01:00:00' THEN datetime(at, '+1 hour') ELSE at END) = '2026-03'
),
refunded AS (
  SELECT SUM(COALESCE(r.amount, real.amount) * CASE WHEN real.cur = 'EUR' THEN 1.0 ELSE (SELECT eur_per_unit FROM fx_rates f
         WHERE f.currency = real.cur AND f.day <= date(r.refunded_at) ORDER BY f.day DESC LIMIT 1) END) AS v
  FROM refunds r JOIN real ON real.ref = r.provider_ref
  WHERE strftime('%Y-%m', CASE WHEN r.refunded_at >= '2026-03-29 01:00:00' THEN datetime(r.refunded_at, '+1 hour')
                          ELSE r.refunded_at END) = '2026-03'
)
SELECT ROUND((SELECT v FROM revenue) - (SELECT v FROM refunded), 2) AS net_eur
"""


def independent_answer() -> Decimal:
    """The expected figure computed in Python straight from the rows, following the prompt's definitions."""
    db = sqlite3.connect(":memory:")
    db.executescript((sqldb.DATASETS / "billing.sql").read_text(encoding="utf-8"))
    is_test = dict(db.execute("SELECT id, is_test FROM customers"))
    customer = dict(db.execute("SELECT id, customer_id FROM orders"))
    pays = {}
    for ref, oid, at, cents, cur, status in db.execute("SELECT provider_ref, order_id, event_at, amount_cents, currency, status FROM payments"):
        pays[ref] = (oid, dt.datetime.strptime(at, "%Y-%m-%d %H:%M:%S"), Decimal(cents) / 100, cur or "EUR", status.lower())
    for ref, oid, at, amount, cur, status in db.execute("SELECT provider_ref, order_id, event_at, amount, currency, status FROM payments_v2"):
        pays[ref] = (oid, dt.datetime.strptime(at, "%Y-%m-%dT%H:%M:%SZ"), Decimal(str(amount)), cur, status)
    fx = {(dt.date.fromisoformat(d), c): Decimal(str(r)) for d, c, r in db.execute("SELECT day, currency, eur_per_unit FROM fx_rates")}

    def rate(day: dt.date, cur: str) -> Decimal:
        while cur != "EUR" and (day, cur) not in fx:
            day -= dt.timedelta(days=1)
        return Decimal(1) if cur == "EUR" else fx[(day, cur)]

    def march(t: dt.datetime) -> bool:
        local = t + dt.timedelta(hours=1 if t >= dt.datetime(2026, 3, 29, 1) else 0)
        return (local.year, local.month) == (2026, 3)

    def real(oid: int) -> bool:
        return customer[oid] is None or is_test[customer[oid]] != 1

    total = sum((amt * rate(at.date(), cur) for oid, at, amt, cur, status in pays.values()
                 if status == "captured" and march(at) and real(oid)), Decimal(0))
    for ref, at, amount in db.execute("SELECT provider_ref, refunded_at, amount FROM refunds"):
        oid, _, full, cur, _ = pays[ref]
        when = dt.datetime.strptime(at, "%Y-%m-%d %H:%M:%S")
        if march(when) and real(oid):
            total -= (full if amount is None else Decimal(str(amount))) * rate(when.date(), cur)
    return total


def test_a5_sql_oracle():
    env = build("a5-march-revenue")
    assert float(independent_answer()) == pytest.approx(env.answer, abs=0.001)
    tables = {t["name"] for t in J(env.handle("list_tables", {}))["tables"]}
    assert tables == {"customers", "orders", "payments", "payments_v2", "refunds", "fx_rates"}
    for table in tables:
        assert "CREATE TABLE" in J(env.handle("describe_table", {"table": table}))["definition"]
    assert "NULL means EUR" in J(env.handle("describe_table", {"table": "payments"}))["definition"]
    statuses = J(env.handle("run_sql", {"query": "SELECT DISTINCT status FROM payments UNION SELECT DISTINCT status FROM payments_v2"}))
    assert {r[0] for r in statuses["rows"]} == {"CAPTURED", "FAILED", "VOIDED", "captured", "failed", "voided"}
    dups = J(env.handle("run_sql", {"query": "SELECT count(*) - count(DISTINCT provider_ref) FROM payments"}))["rows"][0][0]
    both = J(env.handle("run_sql", {"query": "SELECT count(*) FROM payments JOIN payments_v2 USING (provider_ref)"}))["rows"][0][0]
    assert dups > 10 and both > 4
    result = J(env.handle("run_sql", {"query": ORACLE_SQL}))
    value = result["rows"][0][0]
    ok, why = env.grade(f"Net revenue for March 2026 (Lisbon time) is EUR {value:,.2f}.\nANSWER: {value:.2f}")
    assert ok, why
    for text in ("ANSWER: 12,535.90", "ANSWER: €12.535,90", "answer = 12535.9 EUR", "ANSWER: 12 535,90."):
        assert env.grade(text)[0], text


CARELESS_SQL = {
    # Only the legacy table, UTC month, no dedup, no FX, no test-account filter.
    "legacy only": "SELECT SUM(amount_cents) / 100.0 FROM payments WHERE status = 'CAPTURED' AND event_at LIKE '2026-03%'",
    # Both tables but no dedup, text comparison of mixed timestamp formats, is_test = 0, FX ignored.
    "naive union": """
        SELECT SUM(a) FROM (
          SELECT p.amount_cents / 100.0 AS a, p.order_id FROM payments p WHERE p.status = 'CAPTURED'
            AND p.event_at >= '2026-03-01 00:00:00' AND p.event_at < '2026-03-31 23:00:00'
          UNION ALL
          SELECT v.amount, v.order_id FROM payments_v2 v WHERE v.status = 'captured'
            AND v.event_at >= '2026-03-01 00:00:00' AND v.event_at < '2026-03-31 23:00:00'
        ) x JOIN orders o ON o.id = x.order_id JOIN customers c ON c.id = o.customer_id WHERE c.is_test = 0""",
    # Everything right except the month boundary: UTC instead of Lisbon.
    "utc month": ORACLE_SQL.replace("'2026-03-29 01:00:00'", "'2099-01-01 00:00:00'"),
    # Everything right except NULL is_test (legacy accounts) treated as test accounts.
    "is_test = 0": ORACLE_SQL.replace("WHERE c.is_test IS NOT 1", "WHERE c.is_test = 0 OR o.customer_id IS NULL"),
    # Everything right except the FX rate must exist on the exact day.
    "exact-day fx": ORACLE_SQL.replace("f.day <= date(real.at) ORDER BY f.day DESC", "f.day = date(real.at)"),
    # Everything right except the refunds join duplicates webhook retries (no dedup before joining refunds).
    "refund join dup": ORACLE_SQL.replace(
        "FROM refunds r JOIN real ON",
        "FROM refunds r JOIN (SELECT raw.* FROM raw JOIN orders o ON o.id = raw.order_id LEFT JOIN customers c"
        " ON c.id = o.customer_id WHERE c.is_test IS NOT 1) real ON"),
}


@pytest.mark.parametrize("name", list(CARELESS_SQL))
def test_a5_sql_careless(name):
    env = build("a5-march-revenue")
    out = J(env.handle("run_sql", {"query": CARELESS_SQL[name]}))
    assert "error" not in out, out
    value = out["rows"][0][0]
    ok, why = env.grade(f"ANSWER: {value:.2f}")
    assert not ok and why[0].startswith("expected 12535.90"), (name, value)


def test_sql_safety(monkeypatch):
    env = build("a5-march-revenue")
    for query in ("INSERT INTO customers VALUES (999, 'x', 'PT', 0)", "DELETE FROM refunds", "DROP TABLE payments",
                  "CREATE TABLE t (a)", "UPDATE fx_rates SET eur_per_unit = 2", "PRAGMA query_only = OFF",
                  "ATTACH DATABASE '/tmp/x.db' AS x", "SELECT load_extension('x')"):
        assert "error" in J(env.handle("run_sql", {"query": query})), query
    assert J(env.handle("run_sql", {"query": "SELECT count(*) FROM refunds"}))["rows"] == [[12]]
    assert J(env.handle("run_sql", {"query": "PRAGMA table_info(refunds)"}))["row_count"] == 4
    assert J(env.handle("run_sql", {"query": "SELECT name FROM pragma_table_info('fx_rates')"}))["rows"][0] == ["day"]
    assert "one statement" in J(env.handle("run_sql", {"query": "SELECT 1; SELECT 2"}))["error"]
    assert "SQL error" in J(env.handle("run_sql", {"query": "SELEC 1"}))["error"]
    assert "non-empty" in J(env.handle("run_sql", {"query": ""}))["error"]
    assert "SQL error" in J(env.handle("run_sql", {"query": "SELECT zeroblob(5000000)"}))["error"]
    monkeypatch.setattr(sqldb, "MAX_STEPS", 200_000)
    runaway = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT count(*) FROM c"
    assert "aborted" in J(env.handle("run_sql", {"query": runaway}))["error"]
    assert J(env.handle("run_sql", {"query": "SELECT 1 AS ok"}))["rows"] == [[1]]  # the budget resets per query
    many = J(env.handle("run_sql", {"query": "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT x FROM c"}))
    assert many["truncated"] and many["row_count"] == sqldb.MAX_ROWS
    wide = J(env.handle("run_sql", {"query": "SELECT hex(zeroblob(200)), x'00ff', p.* FROM payments p"}))
    assert wide["truncated"] and wide["rows"][0][0].endswith("...") and wide["rows"][0][1] == "00ff"
    assert len(json.dumps(wide)) <= sqldb.MAX_RESULT
    assert "no table" in J(env.handle("describe_table", {"table": "sqlite_master"}))["error"]
    assert J(env.handle("describe_table", {"table": '"Payments"'}))["table"] == "payments"
    assert env.grade("about twelve thousand")[1] == ["no 'ANSWER: <number>' line in the final answer"]
    assert "couldn't read" in env.grade("ANSWER: 1.2.3.4")[1][0]
    assert parse_amount("1,234,567") == 1234567 and parse_amount("12,5") == 12.5 and parse_amount(".,") is None
    with pytest.raises(ValueError):
        SqlEnv(dataset="nope")


# --------------------------------------------------------------------------------------------- a5: incident
def oracle_incident(env: IncidentEnv) -> str:
    alerts = J(env.handle("get_alerts", {}))["alerts"]
    assert any(a["service"] == "checkout-api" for a in alerts)
    graph = {s["name"]: s for s in J(env.handle("list_services", {}))["services"]}
    assert "search-svc" not in graph["checkout-api"]["depends_on"]
    overview = J(env.handle("get_metrics", {"service": "checkout-api", "metric": "error_rate_pct", "start": "08:00",
                                            "end": "16:00", "step_minutes": 10}))["points"]
    assert [t for t, v in overview if v > 1] == ["13:40", "13:50", "14:00", "14:10"]
    points = J(env.handle("get_metrics", {"service": "checkout-api", "metric": "error_rate_pct", "start": "13:00", "end": "15:00"}))["points"]
    i = next(i for i, (_, v) in enumerate(points) if v > 1)
    start = points[i][0]
    end = next(t for t, v in points[i:] if v <= 1)
    errors = J(env.handle("search_logs", {"service": "checkout-api", "level": "ERROR", "start": start, "end": end}))
    upstream = {re.search(r"upstream=(\S+)", ln).group(1) for ln in errors["lines"]}
    assert upstream == {"pricing-svc"} and errors["total_matches"] > len(errors["lines"])
    pool = J(env.handle("search_logs", {"service": "pricing-svc", "query": "pool", "start": "08:00", "end": start}))
    assert any("DB_POOL_MAX not set" in ln for ln in pool["lines"])
    pool_max = J(env.handle("get_metrics", {"service": "pricing-svc", "metric": "db_pool_max", "start": "10:00", "end": "13:42"}))["points"]
    changed = next(t for (t, v), (_, prev) in zip(pool_max[1:], pool_max) if v != prev)
    deploys = J(env.handle("get_deploys", {"service": "pricing-svc", "end": changed}))["deploys"]
    culprit = J(env.handle("get_deploy", {"deploy_id": deploys[-1]["id"]}))
    key = next(k for k, (old, new) in culprit["changes"]["config"].items() if old == "40")
    for other in J(env.handle("get_deploys", {"start": "13:30", "end": start}))["deploys"]:  # rule out the nearby changes
        assert "DB_POOL" not in json.dumps(J(env.handle("get_deploy", {"deploy_id": other["id"]}))["changes"])
    inventory = J(env.handle("get_metrics", {"service": "inventory-svc", "metric": "error_rate_pct", "start": "13:30", "end": "14:30",
                                             "step_minutes": 5}))["points"]
    assert max(v for _, v in inventory) < 1
    return ("Root cause: the pricing-svc deploy removed DB_POOL_MAX, falling back to 8 connections per replica.\n"
            + json.dumps({"deploy_id": culprit["id"], "service": culprit["service"], "config_key": key,
                          "impact_start": start, "impact_end": end}))


def test_a5_incident_oracle():
    env = build("a5-checkout-incident")
    answer = oracle_incident(env)
    assert 10 <= len(env.calls) <= 30
    ok, why = env.grade(answer)
    assert ok, why
    assert json.loads(answer.splitlines()[-1])["impact_end"] == "14:17"
    alt = {"deploy_id": "DEP-8B2F", "service": "pricing-svc", "config_key": "db_pool_max (40 -> unset, default 8)",
           "impact_start": "2026-11-03T13:41Z", "impact_end": "2026-11-03 14:16:00 UTC"}
    assert env.grade("```json\n" + json.dumps(alt) + "\n```")[0]


@pytest.mark.parametrize("answer", [
    # Blames the deploy nearest the alert and dates the window from the alert.
    {"deploy_id": "dep-91aa", "service": "checkout-api", "config_key": "new-banner", "impact_start": "13:47", "impact_end": "14:22"},
    # Blames the inventory migration a minute before the errors (it did slow inventory down).
    {"deploy_id": "dep-4d10", "service": "inventory-svc", "config_key": "none", "impact_start": "13:42", "impact_end": "14:17"},
    # Right service, nearest pricing deploy before the errors.
    {"deploy_id": "dep-8c07", "service": "pricing-svc", "config_key": "DB_POOL_MAX", "impact_start": "13:42", "impact_end": "14:17"},
    # Right deploy, wrong key (the replica host change in the same deploy).
    {"deploy_id": "dep-8b2f", "service": "pricing-svc", "config_key": "PRICING_DB_READ_HOST", "impact_start": "13:42",
     "impact_end": "14:17"},
    {"deploy_id": "dep-8b2f", "service": "pricing-svc", "config_key": "PRICING_DB_POOL_MAX", "impact_start": "13:42",
     "impact_end": "14:17"},
    # Right cause, but stops the window when scaling halved the errors (still above 1%).
    {"deploy_id": "dep-8b2f", "service": "pricing-svc", "config_key": "DB_POOL_MAX", "impact_start": "13:42", "impact_end": "14:05"},
    # Names the hotfix.
    {"deploy_id": "dep-8d4a", "service": "pricing-svc", "config_key": "DB_POOL_MAX", "impact_start": "13:42", "impact_end": "14:17"},
])
def test_a5_incident_careless(answer):
    env = build("a5-checkout-incident")
    env.handle("get_alerts", {})
    env.handle("get_deploys", {"start": "13:00", "end": "13:47"})
    ok, why = env.grade("Postmortem summary.\n" + json.dumps(answer))
    assert not ok and why


def test_incident_tools():
    env = build("a5-checkout-incident")
    assert "unknown service" in J(env.handle("get_metrics", {"service": "db", "metric": "x", "start": "13:00", "end": "14:00"}))["error"]
    assert "has no metric" in J(env.handle("get_metrics", {"service": "pricing-svc", "metric": "cpu", "start": "13:00", "end": "14:00"}))["error"]
    m = {"service": "pricing-svc", "metric": "replicas"}
    assert "start is required" in J(env.handle("get_metrics", {**m, "end": "14:00"}))["error"]
    assert "more than" in J(env.handle("get_metrics", {**m, "start": "08:00", "end": "16:00"}))["error"]
    assert "positive integer" in J(env.handle("get_metrics", {**m, "start": "13:00", "end": "14:00", "step_minutes": 0}))["error"]
    assert "end must be after" in J(env.handle("get_metrics", {**m, "start": "14:00", "end": "13:00"}))["error"]
    assert "no data for 2026-11-02" in J(env.handle("get_metrics", {**m, "start": "2026-11-02T13:00", "end": "14:00"}))["error"]
    assert "can't read" in J(env.handle("get_metrics", {**m, "start": "1pm", "end": "14:00"}))["error"]
    assert "can't read" in J(env.handle("get_metrics", {**m, "start": "13:00", "end": "25:99"}))["error"]
    assert "can't read" in J(env.handle("get_deploys", {"end": "noon"}))["error"]
    pts = J(env.handle("get_metrics", {**m, "start": "2026-11-03T14:02:00Z", "end": "14:06"}))["points"]
    assert pts == [["14:02", 3], ["14:03", 3], ["14:04", 6], ["14:05", 6]]
    assert J(env.handle("get_metrics", {**m, "start": "07:00", "end": "08:02"}))["points"][0][0] == "08:00"
    assert "unknown service" in J(env.handle("get_deploys", {"service": "nope"}))["error"]
    assert "end must be after" in J(env.handle("get_deploys", {"start": "15:00", "end": "14:00"}))["error"]
    assert len(J(env.handle("get_deploys", {"service": "all"}))["deploys"]) == len(env.deploys)
    assert "no deploy" in J(env.handle("get_deploy", {"deploy_id": "dep-0000"}))["error"]
    logs = {"service": "search-svc", "start": "13:50", "end": "14:00"}
    assert "unknown service" in J(env.handle("search_logs", {**logs, "service": "web"}))["error"]
    assert "level must be" in J(env.handle("search_logs", {**logs, "level": "FATAL"}))["error"]
    assert "positive integer" in J(env.handle("search_logs", {**logs, "limit": "10"}))["error"]
    assert "end must be after" in J(env.handle("search_logs", {**logs, "end": "13:00"}))["error"]
    warn = J(env.handle("search_logs", {**logs, "level": "warning", "query": "CRAWLER", "limit": 500}))
    assert warn["total_matches"] == 50 and len(warn["lines"]) == 50 and "note" not in warn
    everything = J(env.handle("search_logs", {"service": "*", "query": "deploy dep-"}))
    assert everything["total_matches"] == 5 and everything["lines"][0].startswith("2026-11-03T12:52:")
    assert "showing the first 5" in J(env.handle("search_logs", {**logs, "limit": 5}))["note"]
    assert env.grade("no json here") == (False, ["no JSON object in the final answer"])
    ok, why = env.grade('{"deploy_id": "dep-8b2f"} then {not json} {"service": "pricing-svc", "impact_start": 13}')
    assert not ok and any("can't read" in w for w in why) and any("deploy_id" in w for w in why)
    with pytest.raises(ValueError):
        IncidentEnv(scenario="nope")
