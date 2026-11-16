"""The frontier general eval suite checks itself.

Every task file is complete, generators are deterministic, fast and the right size, and every exact answer is
re-derived here independently of the code that wrote it: filing logs and logic-grid clues are parsed back out of the
prompt text and replayed or brute-forced, counting answers are brute-forced, the robot plan is proven minimal by BFS,
and the capacity decision is recomputed from numbers parsed out of the sources. Constraint tasks get a reply that
passes and replies that fail, graded by a small checker that implements the documented constraint semantics.
"""

from __future__ import annotations

import difflib
import importlib
import itertools
import json
import math
import re
import time
from datetime import date
from fractions import Fraction
from functools import cache
from pathlib import Path

import pytest
import yaml

import meritorder.evals
from meritorder.evals.generators import puzzles, records

ROOT = Path(meritorder.evals.__file__).parent / "suites" / "frontier" / "general"
TASK_DIRS = sorted(p for p in ROOT.iterdir() if (p / "task.yaml").is_file()) if ROOT.is_dir() else []
NAMES = [p.name for p in TASK_DIRS]
REQUIRED = {"id", "task", "level", "kind", "summary"}
CHECKS = {"final", "final_number", "final_one_of", "constraints", "contains", "not_contains", "equals", "regex", "number",
          "json"}
CONSTRAINT_KEYS = {"words", "bullets", "paragraphs", "must_include", "must_not_include", "max_chars", "starts_with",
                   "ends_with", "all_lowercase", "json_keys"}


def spec_of(name: str) -> dict:
    return yaml.safe_load((ROOT / name / "task.yaml").read_text(encoding="utf-8"))


def generate(spec: dict) -> dict:
    module, fn = spec["generate"].split(":")
    return getattr(importlib.import_module(module), fn)(**spec.get("generate_args", {}))


@cache
def task(name: str) -> tuple[dict, str, list]:
    """(spec, prompt, checks) with generated tasks materialised."""
    spec = spec_of(name)
    if "generate" in spec:
        out = generate(spec)
        return spec, out["prompt"], out["checks"]
    prompt = spec.get("prompt") or (ROOT / name / spec["prompt_file"]).read_text(encoding="utf-8")
    return spec, prompt, spec.get("checks") or []


def by_generator(fn: str) -> list[str]:
    return [n for n in NAMES if spec_of(n).get("generate", "").endswith(fn)]


# ---------------------------------------------------------------- the documented grading semantics

def final_text(reply: str) -> str | None:
    """Text after the last 'Final answer:' (to the end of that line), without whitespace, **, backticks, final '.'."""
    parts = re.split(r"final answer\s*:", reply, flags=re.IGNORECASE)
    if len(parts) < 2:
        return None
    s = (parts[-1].strip().splitlines() or [""])[0]
    prev = None
    while s != prev:
        prev, s = s, s.strip().strip("*`").strip().removesuffix(".")
    return s


def as_number(s: str) -> float | None:
    s = s.replace(",", "").strip()
    try:
        if "/" in s:
            a, b = s.split("/")
            return float(Fraction(a.strip()) / Fraction(b.strip()))
        return float(s)
    except (ValueError, ZeroDivisionError):
        return None


def constraint_failures(text: str, rules: dict) -> list[str]:
    t = text.strip()
    bad = []
    assert set(rules) <= CONSTRAINT_KEYS, set(rules) - CONSTRAINT_KEYS
    if "words" in rules and not rules["words"][0] <= len(t.split()) <= rules["words"][1]:
        bad.append(f"words={len(t.split())}")
    if "bullets" in rules:
        n = sum(1 for line in t.splitlines() if line.startswith(("- ", "* ")))
        if n != rules["bullets"]:
            bad.append(f"bullets={n}")
    if "paragraphs" in rules:
        n = len([b for b in re.split(r"\n\s*\n", t) if b.strip()])
        if n != rules["paragraphs"]:
            bad.append(f"paragraphs={n}")
    bad += [f"missing {s!r}" for s in rules.get("must_include", []) if s.lower() not in t.lower()]
    bad += [f"contains {s!r}" for s in rules.get("must_not_include", []) if s.lower() in t.lower()]
    if "max_chars" in rules and len(t) > rules["max_chars"]:
        bad.append(f"chars={len(t)}")
    if "starts_with" in rules and not t.startswith(rules["starts_with"]):
        bad.append("start")
    if "ends_with" in rules and not t.endswith(rules["ends_with"]):
        bad.append("end")
    if rules.get("all_lowercase") and t != t.lower():
        bad.append("case")
    if "json_keys" in rules:
        try:
            obj = json.loads(t)
        except json.JSONDecodeError:
            obj = None
        if not isinstance(obj, dict) or set(obj) != set(rules["json_keys"]):
            bad.append("json_keys")
    return bad


def passes(check: dict, reply: str) -> bool:
    kind, expected = next(iter(check.items()))
    got = final_text(reply)
    if kind == "final":
        return got is not None and got.lower() == str(expected).lower()
    if kind == "final_one_of":
        return got is not None and got.lower() in [str(e).lower() for e in expected]
    if kind == "final_number":
        value = as_number(got or "")
        return value is not None and abs(value - float(expected)) <= check.get("tolerance", 1e-9)
    if kind == "constraints":
        return not constraint_failures(reply, expected)
    raise AssertionError(f"no local grader for {kind}")


def all_pass(checks: list, reply: str) -> bool:
    return all(passes(c, reply) for c in checks)


def test_grading_semantics():
    assert final_text("x\nFinal answer: 1\nmore\n**FINAL ANSWER:** `Carol`.") == "Carol"
    assert final_text("no answer") is None
    assert passes({"final": "carol"}, "Final answer: **Carol.**")
    assert passes({"final_number": 0.32098765432098764, "tolerance": 1e-9}, "Final answer: 26/81")
    assert not passes({"final_number": 0.32098765432098764, "tolerance": 1e-9}, "Final answer: 0.321")
    assert passes({"final_number": 2904}, "Final answer: 2,904")
    assert not passes({"final_number": 1}, "Final answer: one")
    assert passes({"final_one_of": ["a b", "a, b"]}, "Final answer: A, B")
    assert constraint_failures('{"a": 1, "b": 2}', {"json_keys": ["a", "b"]}) == []
    assert constraint_failures("[1]", {"json_keys": ["a"]}) == ["json_keys"]
    assert constraint_failures("nope", {"json_keys": ["a"]}) == ["json_keys"]


# ---------------------------------------------------------------- task files

def test_suite_layout():
    assert len(NAMES) == 14
    specs = {n: spec_of(n) for n in NAMES}
    levels = [s["level"] for s in specs.values()]
    assert {lvl: levels.count(lvl) for lvl in (3, 4, 5)} == {3: 4, 4: 5, 5: 5}
    assert sum(s["kind"] == "judged" for s in specs.values()) <= 4
    assert 1 <= sum(specs[n]["kind"] == "answer" and any("constraints" in c for c in task(n)[2]) for n in NAMES) <= 2
    multi_hop = {specs[n]["level"] for n in by_generator(":multi_hop")}
    assert multi_hop == {3, 4, 5}
    for name, s in specs.items():
        assert REQUIRED <= set(s), (name, REQUIRED - set(s))
        assert s["id"] == name and name.startswith(f"g{s['level']}-") and s["task"] == "general"
        assert s["kind"] in {"answer", "judged"} and "max_tokens" not in s  # models write as much as they can
        assert isinstance(s["summary"], str) and 20 < len(s["summary"]) <= 260
        assert sum(k in s for k in ("prompt", "prompt_file", "generate")) == 1, name
        if "prompt_file" in s:
            assert (ROOT / name / s["prompt_file"]).is_file()
        if "generate" in s:
            assert "checks" not in s and isinstance(s.get("generate_args"), dict)
        _, prompt, checks = task(name)
        assert checks and all(len(c) <= 2 and next(iter(c)) in CHECKS for c in checks), name
        if any(next(iter(c)).startswith("final") for c in checks):
            assert "Final answer:" in prompt, name
        if s["kind"] == "judged":
            assert isinstance(s["rubric"], list) and len(s["rubric"]) >= 6
            assert all(isinstance(r, str) and r.endswith(".") for r in s["rubric"])
            assert sum(bool(re.search(r"\d", r)) for r in s["rubric"]) >= len(s["rubric"]) // 2  # concrete, not vibes
            assert s["reference_answer"].strip()
            assert all_pass(checks, s["reference_answer"]), name


@pytest.mark.parametrize("name", [n for n in NAMES if "generate" in spec_of(n)])
def test_generators_are_deterministic_fast_and_sized(name):
    spec = spec_of(name)
    t0 = time.perf_counter()
    first = generate(spec)
    elapsed = time.perf_counter() - t0
    assert first == generate(spec)
    assert elapsed < 1.0, f"{name} took {elapsed:.2f}s"
    assert set(first) == {"prompt", "checks"} and first["prompt"].strip()
    args = spec["generate_args"]
    if "words" in args:
        assert abs(len(first["prompt"].split()) - args["words"]) <= 0.1 * args["words"]
    other = generate({**spec, "generate_args": {**args, "seed": args["seed"] + 1}})
    assert other["prompt"] != first["prompt"]


def test_records_rejects_unknown_hops():
    with pytest.raises(ValueError):
        records.build(1, 1000, 5)


# ---------------------------------------------------------------- filing logs (long-context multi-hop)

HEADER = re.compile(r"^(R-\d{4}) \| filed (\d{4}-\d{2}-\d{2}) \| ([A-Z ]+)$")
WHO = r"(E-\d{4}|[A-Z][a-z]+ [A-Z][a-z]+)"
PROJ = r"(P-\d{3}|[A-Z][a-z]+)"
BODY = {
    "STAFF ENTRY": re.compile(r"^([A-Z][a-z]+ [A-Z][a-z]+) joins as an? [a-z -]+\. Employee ID: (E-\d{4})\. "
                              r"Base office: ([A-Z][a-z]+)\."),
    "PROJECT CHARTER": re.compile(r"^Project ([A-Z][a-z]+) \((P-\d{3})\) is chartered to [^.]+\. Lead: " + WHO + r"\."),
    "LEAD CHANGE": re.compile(r"^Project ([A-Z][a-z]+): new lead " + WHO + r"\."),
    "TICKET OPENED": re.compile(r'^Ticket (T-\d{5}) "[^"]+" is opened under project ' + PROJ + r"\."),
    "TICKET TRANSFER": re.compile(r"^Ticket (T-\d{5}) is reassigned to project " + PROJ + r"\."),
    "OFFICE TRANSFER": re.compile(r"^" + WHO + r" relocates to the ([A-Z][a-z]+) office\."),
    "SITE HEAD": re.compile(r"^Site head of the ([A-Z][a-z]+) office: " + WHO + r"\."),
    "RESCISSION": re.compile(r"^Record (R-\d{4}) is rescinded and void"),
}
EFFECTIVE = re.compile(r" Effective (\d{4}-\d{2}-\d{2})\.$")


def parse_log(prompt: str, hops: int) -> dict:
    body = prompt.split("=== FILING LOG ===")[1].split("=== END OF LOG ===")[0].strip()
    recs, names, codenames = [], {}, {}
    for block in body.split("\n\n"):
        head, text = block.split("\n", 1)
        m = HEADER.match(head)
        assert m, head
        rec = {"rid": m[1], "filed": date.fromisoformat(m[2]), "kind": m[3], "text": text, "changes": [],
               "eff": None, "rescinds": None}
        recs.append(rec)
        if rec["kind"] not in BODY:
            assert not EFFECTIVE.search(text), "only changes carry effective dates"
            continue
        b = BODY[rec["kind"]].match(text)
        assert b, text
        e = EFFECTIVE.search(text)
        rec["eff"] = date.fromisoformat(e[1]) if e else None
        assert (e is not None) == (hops == 4 and rec["kind"] != "RESCISSION"), text
        if rec["kind"] == "STAFF ENTRY":
            assert b[2] not in names and b[1] not in names.values(), "names and IDs are unique"
            names[b[2]] = b[1]
            rec["changes"] = [(("office", b[2] if hops >= 3 else b[1]), b[3])]
        elif rec["kind"] == "PROJECT CHARTER":
            assert b[2] not in codenames and b[1] not in codenames.values()
            codenames[b[2]] = b[1]
            rec["changes"] = [(("lead", b[1]), b[3])]
        elif rec["kind"] == "LEAD CHANGE":
            rec["changes"] = [(("lead", b[1]), b[2])]
        elif rec["kind"] in ("TICKET OPENED", "TICKET TRANSFER"):
            rec["changes"] = [(("ticket", b[1]), b[2])]
        elif rec["kind"] == "OFFICE TRANSFER":
            rec["changes"] = [(("office", b[1]), b[2])]
        elif rec["kind"] == "SITE HEAD":
            rec["changes"] = [(("head", b[1]), b[2])]
        else:
            rec["rescinds"] = b[1]
    assert [r["rid"] for r in recs] == [f"R-{i:04d}" for i in range(1, len(recs) + 1)]
    assert [r["filed"] for r in recs] == sorted(r["filed"] for r in recs)
    by_rid = {r["rid"]: r for r in recs}
    for r in recs:
        if r["rescinds"]:
            target = by_rid[r["rescinds"]]
            assert target["changes"] and target["rid"] < r["rid"]
    return {"records": recs, "names": names, "codenames": codenames}


def replay(log: dict, as_of: date | None, rescissions: bool = True, dates: bool = True,
           filed_by: date | None = None) -> dict:
    void = {r["rescinds"] for r in log["records"] if r["rescinds"]} if rescissions else set()
    live = [r for r in log["records"] if r["changes"] and r["rid"] not in void]
    if filed_by:
        live = [r for r in live if r["filed"] <= filed_by]
    if dates and as_of:
        live = [r for r in live if r["eff"] <= as_of]
    state: dict = {}
    for r in sorted(live, key=lambda r: ((r["eff"] or r["filed"]) if dates else date.min, r["rid"])):
        for key, value in r["changes"]:
            state[key] = value
    return state


def chain(log: dict, state: dict, ticket: str, hops: int) -> tuple[str, list]:
    """The answer and the keys looked up on the way."""
    project = state[("ticket", ticket)]
    if hops == 2:
        return state[("lead", project)], [("ticket", ticket), ("lead", project)]
    code = log["codenames"][project]
    lead = state[("lead", code)]
    keys = [("ticket", ticket), ("lead", code)]
    if hops == 3:
        return log["names"][lead], keys
    office = state[("office", lead)]
    return log["names"][state[("head", office)]], keys + [("office", lead), ("head", office)]


@pytest.mark.parametrize("name", by_generator(":multi_hop"))
def test_filing_log_answer_is_unique_and_traps_are_live(name):
    spec, prompt, checks = task(name)
    hops = spec["generate_args"]["hops"]
    log = parse_log(prompt, hops)
    ticket = re.search(r"QUESTION: .*?ticket (T-\d{5})", prompt)[1]
    m = re.search(r"At the end of (\d{4}-\d{2}-\d{2})", prompt)
    as_of = date.fromisoformat(m[1]) if m else None
    assert (as_of is not None) == (hops == 4)
    opened = [r for r in log["records"] if r["kind"] == "TICKET OPENED"]
    tickets = [r["changes"][0][0][1] for r in opened]
    assert len(tickets) == len(set(tickets)) and ticket in tickets
    # every reference in a change resolves to something that exists
    people = set(log["names"]) if hops >= 3 else set(log["names"].values())
    projects = set(log["codenames"]) if hops >= 3 else set(log["codenames"].values())
    for r in log["records"]:
        for (kind, key), value in r["changes"]:
            if kind == "lead":
                assert key in log["codenames"].values() and value in people
            elif kind == "ticket":
                assert value in projects
            elif kind == "head":
                assert value in people
    state = replay(log, as_of)
    answer, keys = chain(log, state, ticket, hops)
    want = checks[0]["final_one_of"]
    assert want[0] == answer and passes(checks[0], f"...\nFinal answer: {answer}")
    eid = next(e for e, n in log["names"].items() if n == answer)
    assert passes(checks[0], f"Final answer: **{answer} ({eid})**")
    assert list(log["names"].values()).count(answer) == 1
    near = [n for n in log["names"].values() if n != answer and difflib.SequenceMatcher(None, n, answer).ratio() >= 0.8]
    assert near, "the answer has a near-duplicate name in the log"
    assert any(t != ticket and sorted(t) == sorted(ticket) for t in tickets), "the ticket has a near-duplicate"
    # the traps are live: each naive reading gives a different (wrong) answer, which the check rejects
    wrong = []
    if hops >= 3:
        wrong += [chain(log, replay(log, as_of, rescissions=False, dates=False), ticket, hops)[0],
                  chain(log, replay(log, as_of, rescissions=False), ticket, hops)[0]]
    if hops == 4:
        wrong += [chain(log, replay(log, as_of, dates=False), ticket, hops)[0],
                  chain(log, replay(log, as_of, filed_by=as_of), ticket, hops)[0]]
    twin = next(t for t in tickets if t != ticket and sorted(t) == sorted(ticket))
    wrong.append(chain(log, state, twin, hops)[0])  # read the look-alike ticket
    original = next(r for r in opened if r["changes"][0][0][1] == ticket)["changes"][0][1]
    first_project = original if hops == 2 else log["codenames"][original]
    current = keys[1][1]
    for code in (first_project, current) if hops <= 3 else ():  # the old project's lead now, the new one's first lead
        lead = state[("lead", code)] if code == first_project else next(
            r["changes"][0][1] for r in log["records"]
            if r["kind"] == "PROJECT CHARTER" and r["changes"][0][0] == ("lead", code))
        wrong.append(lead if hops == 2 else log["names"][lead])
    for w in wrong:
        assert w != answer and not passes(checks[0], f"Final answer: {w}"), w
    if hops == 4:  # no boundary cases and no ties on the chain
        for key in keys:
            effs = [r["eff"] for r in log["records"] if any(k == key for k, _ in r["changes"]) and r["eff"] <= as_of]
            assert as_of not in effs and len(effs) == len(set(effs)), key
        assert any(r["filed"] > as_of and r["eff"] <= as_of for r in log["records"] if r["changes"]), "retroactive"
    # the generator's own replay agrees, including on the naive readings
    d = records.build(spec["generate_args"]["seed"], spec["generate_args"]["words"], hops)
    assert d.prompt == prompt and d.answer == answer
    assert set(records.wrong_readings(d)) <= set(wrong)
    naive = replay(log, as_of, rescissions=False, dates=False)
    assert records.follow(records.replay(d.records, d.as_of, respect_rescissions=False, respect_dates=False),
                          ticket, hops) == chain(log, naive, ticket, hops)[0]


# ---------------------------------------------------------------- logic grids

GRID_TEXT = {
    "workshop": {
        "cats": ["{v}", "the {v} speaker", "the speaker from {v}", "the {v} drinker"],
        "positions": ["monday", "tuesday", "wednesday", "thursday", "friday"],
        "clues": {
            "same": "{X} is {Y}", "diff": "{X} is not {Y}", "either": "{X} is either {Y} or {Z}",
            "at": "{X} speaks on {P}", "not_at": "{X} does not speak on {P}",
            "end": "{X} speaks on either monday or friday", "next": "{X} speaks the day immediately before {Y}",
            "left": "{X} speaks on an earlier day than {Y}", "adj": "{X} and {Y} speak on consecutive days",
            "gap2": "{X} and {Y} speak exactly two days apart",
            "between": "{X} speaks on a day after one of {Y} and {Z} and before the other",
        },
    },
    "street": {
        "cats": ["{v}", "the {v}", "the {v} owner", "the driver of the {v} car", "the {v} player"],
        "positions": [f"house {i}" for i in range(1, 7)],
        "clues": {
            "same": "{X} is {Y}", "diff": "{X} is not {Y}", "either": "{X} is either {Y} or {Z}",
            "at": "{X} lives in {P}", "not_at": "{X} does not live in {P}",
            "end": r"{X} lives at one end of the street \(house 1 or house 6\)",
            "next": "{X} lives immediately west of {Y}", "left": "{X} lives somewhere west of {Y}",
            "adj": "{X} and {Y} are next-door neighbours", "gap2": "{X} and {Y} live exactly two houses apart",
            "between": "{X} lives somewhere between {Y} and {Z}",
        },
    },
}
MEANING = {
    "same": lambda p, x, y: p[x] == p[y], "diff": lambda p, x, y: p[x] != p[y],
    "either": lambda p, x, y, z: p[x] in (p[y], p[z]), "next": lambda p, x, y: p[x] + 1 == p[y],
    "left": lambda p, x, y: p[x] < p[y], "adj": lambda p, x, y: abs(p[x] - p[y]) == 1,
    "gap2": lambda p, x, y: abs(p[x] - p[y]) == 2, "between": lambda p, x, y, z: min(p[y], p[z]) < p[x] < max(p[y], p[z]),
}


def parse_grid(prompt: str, theme: str) -> tuple[list[list[str]], list[tuple]]:
    text = GRID_TEXT[theme]
    values = [line.split(": ", 1)[1].split(", ") for line in prompt.splitlines() if re.match(r"^- [A-Z][a-z ]+: ", line)]
    assert len(values) == len(text["cats"]) and len({len(v) for v in values}) == 1
    n = len(values[0])
    phrase = {fmt.format(v=v).lower(): (c, i) for c, fmt in enumerate(text["cats"]) for i, v in enumerate(values[c])}
    assert len(phrase) == n * len(values)
    alt = "|".join(re.escape(p) for p in sorted(phrase, key=len, reverse=True))
    pos_alt = "|".join(re.escape(p) for p in text["positions"])
    patterns = {kind: re.compile("^" + pat.replace("{X}", f"(?P<X>{alt})").replace("{Y}", f"(?P<Y>{alt})")
                                 .replace("{Z}", f"(?P<Z>{alt})").replace("{P}", f"(?P<P>{pos_alt})") + r"\.$")
                for kind, pat in text["clues"].items()}
    clues = []
    lines = [line for line in prompt.splitlines() if re.match(r"^\d+\. ", line)]
    for i, line in enumerate(lines, 1):
        number, sentence = line.split(". ", 1)
        assert int(number) == i
        hits = [(k, m) for k, p in patterns.items() if (m := p.match(sentence.lower()))]
        assert len(hits) == 1, f"clue is ambiguous or unparsed: {line}"
        kind, m = hits[0]
        refs = tuple(phrase[m[g]] for g in "XYZ" if g in m.groupdict() and m[g])
        if kind in ("same", "diff", "either"):
            assert len({c for c, _ in refs[:2]}) == 2
        if kind == "either":
            assert refs[1][0] == refs[2][0]
        clue = (kind, refs, text["positions"].index(m["P"]) if "P" in m.groupdict() else None)
        clues.append(clue)
    return values, clues


def brute_force(n: int, k: int, clues: list[tuple]) -> list[dict]:
    """All solutions, category by category over permutations, checking each clue once its categories are placed."""
    order = sorted(range(k), key=lambda c: -sum(any(r[0] == c for r in cl[1]) for cl in clues))
    rank = {c: i for i, c in enumerate(order)}
    due: dict[int, list] = {c: [] for c in order}
    for cl in clues:
        due[max((r[0] for r in cl[1]), key=rank.get)].append(cl)

    def ok(cl, p):
        kind, refs, place = cl
        if kind == "at":
            return p[refs[0]] == place
        if kind == "not_at":
            return p[refs[0]] != place
        if kind == "end":
            return p[refs[0]] in (0, n - 1)
        return MEANING[kind](p, *refs)

    out: list[dict] = []

    def place(i: int, p: dict) -> None:
        if i == k:
            out.append(dict(p))
            return
        c = order[i]
        for perm in itertools.permutations(range(n)):
            for v in range(n):
                p[(c, v)] = perm[v]
            if all(ok(cl, p) for cl in due[c]):
                place(i + 1, p)
        for v in range(n):
            del p[(c, v)]

    place(0, {})
    return out


@pytest.mark.parametrize("name", by_generator(":logic_grid"))
def test_logic_grid_has_exactly_one_solution(name):
    spec, prompt, checks = task(name)
    theme = spec["generate_args"]["theme"]
    values, clues = parse_grid(prompt, theme)
    n, k = len(values[0]), len(values)
    solutions = brute_force(n, k, clues)
    assert len(solutions) == 1
    (sol,) = solutions
    answer = "".join(values[0][v][0] for v in sorted(range(n), key=lambda v: sol[(0, v)]))
    assert checks[0]["final_one_of"][0] == answer and passes(checks[0], f"Final answer: `{answer}`")
    assert passes(checks[0], f"Final answer: {', '.join(answer)}")
    assert not passes(checks[0], f"Final answer: {answer[::-1]}")
    example = re.search(r"you would write `Final answer: ([A-Z]+)`", prompt)[1]
    assert example != answer and sorted(example) == sorted(answer)
    assert len({v[0] for v in values[0]}) == n, "initials identify the names"
    # the generator's structure agrees, and every clue is needed (irredundant)
    g = puzzles.grid(spec["generate_args"]["seed"], theme)
    assert g.answer == answer and len(g.clues) == len(clues)
    for i in range(len(g.clues)):
        assert len(puzzles.solve(n, k, g.clues[:i] + g.clues[i + 1:], limit=2)) == 2
    direct = sum(kind in ("same", "at") for kind, _, _ in clues)
    assert direct <= 2, "mostly relational clues"


def test_puzzle_holds_covers_partial_assignments():
    assert puzzles.holds(("at", (0, 0), 1), {}) is None
    assert puzzles.holds(("not_at", (0, 0), 1), {(0, 0): 2}) is True
    assert puzzles.holds(("left", (0, 0), (1, 0)), {(0, 0): 1}) is None
    assert puzzles.holds(("end", (0, 0), 4), {(0, 0): 4}) is True
    assert puzzles.holds(("between", (0, 0), (1, 0), (2, 0)), {(0, 0): 2, (1, 0): 3, (2, 0): 1}) is True
    contradiction = [("same", (0, 0), (1, 0)), ("diff", (0, 0), (1, 0))]
    assert puzzles.solve(3, 2, contradiction) == []
    assert puzzles.solve(3, 2, [("at", (0, 0), 0), ("at", (0, 1), 0)]) == []  # all-different conflict
    assert len(puzzles.solve(2, 1, [], limit=5)) == 2


# ---------------------------------------------------------------- exact counting answers

def test_shelf_arrangements_brute_force():
    _, _, checks = task("g3-shelf-arrangements")
    count = 0
    for reds in itertools.combinations(range(9), 4):
        rest = [i for i in range(9) if i not in reds]
        for blues in itertools.combinations(rest, 3):
            row = ["G"] * 9
            for i in reds:
                row[i] = "R"
            for i in blues:
                row[i] = "B"
            if row[0] != "G" and row[-1] != "G" and all(not (a == b != "G") for a, b in zip(row, row[1:], strict=False)):
                count += 1
    assert count == checks[0]["final_number"] == 66
    assert not passes(checks[0], "Final answer: 93")  # forgetting the end-slot rule


def test_dice_split_brute_force():
    _, _, checks = task("g4-dice-split")

    def splittable(dice):
        total = sum(dice)
        return any(2 * sum(d for i, d in enumerate(dice) if mask >> i & 1) == total for mask in range(1, 15))

    favourable = sum(splittable(d) for d in itertools.product(range(1, 7), repeat=4))
    p = Fraction(favourable, 6**4)
    assert p == Fraction(26, 81)
    assert abs(float(p) - checks[0]["final_number"]) < 1e-15
    assert passes(checks[0], "Final answer: 26/81") and passes(checks[0], "Final answer: 416/1296")
    assert not passes(checks[0], "Final answer: 25/81")
    assert not splittable((1, 1, 1, 4)) and splittable((1, 2, 3, 6)) and splittable((2, 2, 5, 5))


def test_bracelets_brute_force_and_burnside():
    _, _, checks = task("g5-bracelets")
    n, q = 10, 4
    full: list[tuple] = []

    def extend(seq: list[int]) -> None:  # every colouring of the loop with neighbours different
        if len(seq) == n:
            if seq[0] != seq[-1] and len(set(seq)) == q:
                full.append(tuple(seq))
            return
        for c in range(q):
            if c != seq[-1]:
                extend(seq + [c])

    for first in range(q):
        extend([first])
    assert len(full) == 54960
    seen, orbits = set(), 0
    for c in full:
        if c in seen:
            continue
        orbits += 1
        for r in range(n):
            rot = c[r:] + c[:r]
            seen.update((rot, rot[::-1]))
    assert orbits == checks[0]["final_number"] == 2904

    def proper_cycle(m, colours):  # chromatic polynomial of the cycle C_m
        return (colours - 1) ** m + (-1) ** m * (colours - 1)

    def onto(count):  # inclusion-exclusion over unused colours
        return sum((-1) ** j * math.comb(q, j) * count(q - j) for j in range(q + 1))

    rotations = onto(lambda c: proper_cycle(10, c)) + onto(lambda c: proper_cycle(5, c))  # identity, rotation by 5
    reflections = 5 * onto(lambda c: c * (c - 1) ** 5)  # axes through two beads: a proper path of 6 beads
    assert (rotations + reflections) // 20 == 2904 and (rotations + reflections) % 20 == 0
    assert not passes(checks[0], "Final answer: 2922")  # the classic slip: counting rotations by 2 instead of by 5


# ---------------------------------------------------------------- robot planning

DIRS = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}


def test_robot_plan_is_the_unique_bfs_minimum():
    _, prompt, checks = task("g4-robot-blocker")
    rows = [m[1].split() for line in prompt.splitlines() if (m := re.match(r"^\s*\d\s+((?:[.#ABX] ?){6})\s*$", line))]
    assert len(rows) == 6 and all(len(r) == 6 for r in rows)
    find = {ch: (r, c) for r, row in enumerate(rows) for c, ch in enumerate(row) if ch in "ABX"}
    walls = {(r, c) for r, row in enumerate(rows) for c, ch in enumerate(row) if ch == "#"}

    def slide(state, who, d):
        r, c = state[who]
        other = state[1 - who]
        while True:
            nr, nc = r + DIRS[d][0], c + DIRS[d][1]
            if not (0 <= nr < 6 and 0 <= nc < 6) or (nr, nc) in walls or (nr, nc) == other:
                return (r, c)
            r, c = nr, nc

    def shortest(start, robots):
        """All shortest move sequences that leave A on X."""
        frontier, seen, depth = {start: [[]]}, {start}, 0
        while frontier and depth < 30:
            done = [p for s, paths in frontier.items() if s[0] == find["X"] for p in paths]
            if done:
                return depth, done
            nxt: dict = {}
            for s, paths in frontier.items():
                for who in robots:
                    for d in "UDLR":
                        moved = slide(s, who, d)
                        if moved == s[who]:
                            continue
                        t = (moved, s[1]) if who == 0 else (s[0], moved)
                        if t in seen:
                            continue
                        nxt.setdefault(t, []).extend(p + ["AB"[who] + d] for p in paths)
            seen |= set(nxt)
            frontier, depth = nxt, depth + 1
        return None, []

    depth, plans = shortest((find["A"], find["B"]), (0, 1))
    assert depth == 7 and [" ".join(p) for p in plans] == ["AD BL BD AR AU BR AD"]
    accepted = {re.sub(r"[ ,]+", " ", a.upper()) for a in checks[0]["final_one_of"]}
    assert accepted == {" ".join(plans[0])}
    assert passes(checks[0], "Final answer: AD, BL, BD, AR, AU, BR, AD")
    assert not passes(checks[0], "Final answer: AD BL BD AR AU BR AD AU")
    assert shortest((find["A"], find["B"]), (0,)) == (None, []), "A cannot do it alone"


# ---------------------------------------------------------------- judged tasks: the facts behind the rubrics

def test_cooling_decision_recomputed_from_the_sources():
    spec, prompt, checks = task("g5-cooling-reconcile")
    sources = {int(m[1]): m[2] for m in re.finditer(r"## Source (\d+) — (.*?)(?=\n## Source |\Z)", prompt, re.S)}
    assert sorted(sources) == list(range(1, 11))

    def table(n):  # net sensible capacity by return-air temperature (22, 24, 27, 30 °C)
        row = re.search(r"\| Net sensible capacity \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|", sources[n])
        return dict(zip((22, 24, 27, 30), map(float, row.groups()), strict=True))

    rev_b, rev_d, cx60 = table(4), table(5), table(6)
    assert "supersedes Revisions A, B and C" in sources[5]
    units = {"CX-40": 0, "CX-60": 0}
    for m in re.finditer(r"^\| C \| C-(\d\d)(?: to C-(\d\d))? \| Boreal (CX-\d\d)", sources[9], re.M):
        units[m[3]] += int(m[2] or m[1]) - int(m[1]) + 1
    assert units == {"CX-40": 11, "CX-60": 1}
    existing = float(re.search(r"^\| C \| ([\d.]+) kW \|", sources[9], re.M)[1])
    overhead = int(re.search(r"allowance of (\d+)% of the IT load", sources[2])[1]) / 100
    racks, kw = map(int, re.search(r"(\d+) racks, each drawing (\d+) kW", sources[1]).groups())
    assert "Hall B and Hall C are N+2" in sources[2] and "Halls A, B and C operate at a return-air setpoint of 24" in sources[2]

    def max_racks(cx40=rev_d[24], cx60_kw=cx60[24], n40=11, n60=1, spare=2, allowance=overhead, load=existing):
        caps = sorted([cx40] * n40 + [cx60_kw] * n60, reverse=True)
        usable = sum(caps[spare:])
        return math.floor((usable / (1 + allowance) - load) / kw)

    assert max_racks() == 34 == checks[0]["final_number"] < racks
    assert (42 + 34 * 8) * 1.06 <= 340 < (42 + 35 * 8) * 1.06
    # each stale or out-of-scope source changes the answer, so the check discriminates
    assert max_racks(cx40=rev_b[24]) == 39  # superseded data sheet
    assert max_racks(cx40=rev_d[27], cx60_kw=cx60[27]) == 41  # wrong setpoint
    assert max_racks(allowance=0.05) == 35  # withdrawn 5% allowance from the wiki
    assert max_racks(load=0) == 40  # forgot the existing load
    assert max_racks(spare=1) == 38  # N+1
    assert max_racks(cx40=45.0) != 34  # the CX-40-HD figure
    assert max_racks(n40=10, n60=2) == 34  # the second CX-60 (phase 2) frees nothing under N+2
    assert "Final answer: 34" in spec["reference_answer"]


def test_incident_and_memo_facts_behind_the_rubrics():
    spec, prompt, _ = task("g3-incident-summary")
    assert "| 14:12:07 |" in prompt and "| 14:59 | Checkout success rate back" in prompt
    assert 59 - 12 == 47 and "47 minutes" in prompt
    assert "07:13 PDT" in prompt and "| 14:13 | First customer complaint" in prompt  # PDT is UTC-7
    for fact in ("18,412", "6,950", "$198,000", "31,000", "INC-2290"):
        assert fact in prompt and any(fact.strip("$") in r for r in spec["rubric"])
    assert "31,000" not in spec["reference_answer"].replace("early 31,000", "")
    assert "INC-2290" not in spec["reference_answer"]

    spec, prompt, _ = task("g4-pilot-memo-critique")
    for number in ("9.2", "7.8", "5.0", "3.4", "2 of the 61", "9% per year", "1,240", "223"):
        assert number in prompt
    assert round(9.2 / 7.8 - 1, 2) == 0.18 and round(0.18 * 1240) == 223  # the memo's own arithmetic is right
    pilot, comparison = 9.2 * 3.4, 7.8 * 5.0
    assert round(pilot, 1) == 31.3 and comparison == 39.0 and round(1 - pilot / comparison, 2) == 0.20
    quarterly = 2 / 61
    assert round(quarterly * 100, 1) == 3.3 and 0.12 < 1 - (1 - quarterly) ** 4 < 0.13 < 4 * quarterly < 0.135


# ---------------------------------------------------------------- constraint tasks

GOOD_UPDATE = """Status: delayed. Order 7731 for Halvorsen Marine will reach you later than planned. The container missed its \
connecting vessel in Rotterdam after a crane outage at the terminal, and it is now booked on the next sailing.

- New estimated delivery: Thursday 14 November.
- You will receive a 15% credit on the freight charge for this order.
- Track the shipment at any time with reference HM-7731-R.

We will confirm as soon as the container is loaded and again when it clears the terminal. If you need anything in the \
meantime, reply to this message and the team will pick it up the same day.
Harbor Desk"""

GOOD_NOTICE = """hi all, our co-op is throwing a big potluck in our courtyard this sunday, starting at 14:30. it is a \
way to chat, laugh and catch up with folks from all four floors, so do join us. kids can tag along, and so can dogs \
on a short lash.

to join, just bring a dish that six or so folks can try: a salad, a curry, a pasta, a tart, or anything you fancy. if \
it rains, our potluck will shift into our community room on our ground floor.

at around six, a handful of us must stay on to tidy up, so if you can assist, find kai by our main door. it would do \
us a world of good. with warm thanks, kai"""


@pytest.mark.parametrize(("name", "good", "bad"), [
    ("g3-client-update", GOOD_UPDATE, [
        GOOD_UPDATE.replace("Status: delayed. ", "Update: "),
        GOOD_UPDATE.replace("Harbor Desk", "Harbor Desk team"),
        GOOD_UPDATE.replace("same day.\n", "same day.\n\n"),  # sign-off as a fourth block
        GOOD_UPDATE.replace("- Track", "Track"),
        GOOD_UPDATE.replace("15% credit", "15 percent credit"),
        GOOD_UPDATE.replace("We will confirm", "We apologise for this. We will confirm"),
        GOOD_UPDATE.replace("the same day.", "the same day. " + "Thanks again for your patience. " * 6),
    ]),
    ("g5-lipogram-notice", GOOD_NOTICE, [
        GOOD_NOTICE.replace("our courtyard", "the courtyard"),
        GOOD_NOTICE.replace("hi all,", "Hi all,"),
        GOOD_NOTICE.replace("14:30", "half past two"),
        GOOD_NOTICE.replace("\n\nat around six", "\nat around six"),
        GOOD_NOTICE.replace(" kids can tag along, and so can dogs on a short lash.", "").replace(
            " a salad, a curry, a pasta, a tart, or anything you fancy.", "").replace(
            " it would do us a world of good.", ""),
        GOOD_NOTICE.replace("with warm thanks, kai", "with warm thanks, kai!"),
        GOOD_NOTICE.replace("to join, just", "- to join, just"),
    ]),
])
def test_constraint_task_accepts_a_good_reply_and_rejects_near_misses(name, good, bad):
    _, _, checks = task(name)
    (rules,) = [c["constraints"] for c in checks if "constraints" in c]
    assert constraint_failures(good, rules) == []
    assert all_pass(checks, good)
    for reply in bad:
        assert constraint_failures(reply, rules), reply
    # The eval runner's own grader agrees.
    from meritorder.evals.runner import grade

    def graded(text):
        return all(grade(c, {"choices": [{"message": {"content": text}}]})[0] for c in checks)
    assert graded(good)
    assert not any(graded(reply) for reply in bad)
