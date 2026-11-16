"""Logic-grid puzzles with a proven unique solution.

A random solution places one value of every category at each position (a day of the week, a house on a street).
Clues that are true of that solution are drawn in a weighted random order, biased towards relational and negative
clues, until a constraint solver finds exactly one solution; then every clue the solution doesn't need is dropped,
so the final set is irredundant. The answer is the names in position order, as a string of initials.

``grid`` returns the structure for tests; ``logic_grid`` is the task.yaml entry point.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from functools import cache

THEMES = {
    "workshop": {
        "intro": "Five speakers each give one talk at a workshop, one talk per day from Monday to Friday. Each "
                 "speaker has a different topic, comes from a different city and drinks a different tea.",
        "positions": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
        "categories": [
            ("Speakers", ["Anouk", "Bruno", "Chiara", "Dev", "Elsa"], "{v}"),
            ("Topics", ["robotics", "glaciology", "typography", "beekeeping", "cryptography"], "the {v} speaker"),
            ("Home cities", ["Oslo", "Quito", "Hanoi", "Accra", "Perth"], "the speaker from {v}"),
            ("Teas", ["rooibos", "sencha", "oolong", "chai", "mint"], "the {v} drinker"),
        ],
        "at": "{X} speaks on {P}.",
        "not_at": "{X} does not speak on {P}.",
        "end": "{X} speaks on either Monday or Friday.",
        "next": "{X} speaks the day immediately before {Y}.",
        "left": "{X} speaks on an earlier day than {Y}.",
        "adj": "{X} and {Y} speak on consecutive days.",
        "gap2": "{X} and {Y} speak exactly two days apart.",
        "between": "{X} speaks on a day after one of {Y} and {Z} and before the other.",
        "question": "Which speaker gives the talk on each day? End your reply with a line `Final answer: <initials>` "
                    "where <initials> is the first letters of the speakers' names from Monday to Friday as one "
                    "five-letter string (for example, if the order were Elsa, Dev, Chiara, Bruno, Anouk you would "
                    "write `Final answer: EDCBA`).",
    },
    "street": {
        "intro": "Six neighbours live in the six houses on Quay Street, numbered 1 to 6 from west to east. Each has a "
                 "different profession, keeps a different pet, drives a car of a different colour and plays a "
                 "different instrument.",
        "positions": ["house 1", "house 2", "house 3", "house 4", "house 5", "house 6"],
        "categories": [
            ("Names", ["Ines", "Oskar", "Priya", "Tomas", "Wanda", "Yusuf"], "{v}"),
            ("Professions", ["baker", "chemist", "dentist", "florist", "pilot", "tailor"], "the {v}"),
            ("Pets", ["cat", "dog", "ferret", "parrot", "rabbit", "tortoise"], "the {v} owner"),
            ("Car colours", ["black", "blue", "green", "red", "silver", "white"], "the driver of the {v} car"),
            ("Instruments", ["cello", "flute", "harp", "oboe", "trumpet", "violin"], "the {v} player"),
        ],
        "at": "{X} lives in {P}.",
        "not_at": "{X} does not live in {P}.",
        "end": "{X} lives at one end of the street (house 1 or house 6).",
        "next": "{X} lives immediately west of {Y}.",
        "left": "{X} lives somewhere west of {Y}.",
        "adj": "{X} and {Y} are next-door neighbours.",
        "gap2": "{X} and {Y} live exactly two houses apart.",
        "between": "{X} lives somewhere between {Y} and {Z}.",
        "question": "Who lives in each house? End your reply with a line `Final answer: <initials>` where <initials> "
                    "is the first letters of the neighbours' names for houses 1 to 6 in order, as one six-letter "
                    "string (for example, if the order were Yusuf, Wanda, Tomas, Priya, Oskar, Ines you would write "
                    "`Final answer: YWTPOI`).",
    },
}
COMMON = {
    "same": "{X} is {Y}.",
    "diff": "{X} is not {Y}.",
    "either": "{X} is either {Y} or {Z}.",
}
# How often each clue type is drawn; direct placements are rare so the puzzle needs chains of deduction.
WEIGHTS = {"same": 1.4, "diff": 1.6, "at": 0.25, "not_at": 0.8, "end": 0.5, "next": 1.4, "left": 1.6, "adj": 1.2,
           "gap2": 1.2, "either": 1.6, "between": 1.4}

Var = tuple[int, int]  # (category, value index)


@dataclass
class Grid:
    prompt: str
    answer: str
    solution: list[list[int]]  # solution[c][v] = position of value v of category c
    clues: list[tuple]
    theme: str


def holds(clue: tuple, pos) -> bool | None:
    """Whether a clue holds under a (partial) assignment ``pos`` (Var -> position); None if not yet decidable."""
    kind = clue[0]
    if kind in ("at", "not_at"):
        p = pos.get(clue[1])
        if p is None:
            return None
        return (p == clue[2]) == (kind == "at")
    vals = [pos.get(x) for x in clue[1:] if isinstance(x, tuple)]
    if any(v is None for v in vals):
        return None
    if kind == "end":
        return vals[0] in (0, clue[2])
    a, b = vals[0], vals[1]
    if kind == "same":
        return a == b
    if kind == "diff":
        return a != b
    if kind == "next":
        return a + 1 == b
    if kind == "left":
        return a < b
    if kind == "adj":
        return abs(a - b) == 1
    if kind == "gap2":
        return abs(a - b) == 2
    if kind == "either":
        return a == b or a == vals[2]
    return min(b, vals[2]) < a < max(b, vals[2])  # between


@cache
def _relation(kind: str, n: int) -> tuple:
    """Support tables for a clue kind on n positions. Binary kinds: (for_first, for_second), where for_first[mask]
    is the set of first-variable positions with a partner in ``mask``. Ternary kinds: the allowed position triples."""
    if kind in ("either", "between"):
        x, y, z = (0, 0), (1, 0), (2, 0)
        return tuple((a, b, c) for a in range(n) for b in range(n) for c in range(n)
                     if holds((kind, x, y, z), {x: a, y: b, z: c}))
    x, y = (0, 0), (1, 0)
    ok = [[bool(holds((kind, x, y), {x: a, y: b})) for b in range(n)] for a in range(n)]
    first, second = [0] * (1 << n), [0] * (1 << n)
    for mask in range(1, 1 << n):
        low = (mask & -mask).bit_length() - 1
        rest = mask & (mask - 1)
        first[mask] = first[rest] | sum(1 << a for a in range(n) if ok[a][low])
        second[mask] = second[rest] | sum(1 << b for b in range(n) if ok[low][b])
    return first, second


def solve(n: int, k: int, clues: list[tuple], limit: int = 2) -> list[dict]:
    """Up to ``limit`` solutions (Var -> position). Domains are bitmasks; each node propagates arc consistency over
    the clues (a worklist) and all-different with naked and hidden singles per category to a fixpoint, then branches
    on the smallest open domain."""
    index = {(c, v): c * n + v for c in range(k) for v in range(n)}
    full = (1 << n) - 1
    dom = [full] * (n * k)
    cons: list[tuple] = []  # (variable indices, tables)
    for cl in clues:
        kind = cl[0]
        if kind == "at":
            dom[index[cl[1]]] &= 1 << cl[2]
        elif kind == "not_at":
            dom[index[cl[1]]] &= full & ~(1 << cl[2])
        elif kind == "end":
            dom[index[cl[1]]] &= 1 | 1 << (n - 1)
        else:
            cons.append((tuple(index[x] for x in cl[1:]), _relation(kind, n)))
    watch: list[list[int]] = [[] for _ in dom]
    for ci, (vs, _) in enumerate(cons):
        for v in vs:
            watch[v].append(ci)
    groups = [range(c * n, c * n + n) for c in range(k)]

    def revise(ci: int, d: list[int]) -> list[int] | None:
        vs, table = cons[ci]
        if len(vs) == 2:
            a, b = vs
            na, nb = d[a] & table[0][d[b]], d[b] & table[1][d[a]]
            if not na or not nb:
                return None
            changed = [v for v, new in ((a, na), (b, nb)) if new != d[v]]
            d[a], d[b] = na, nb
            return changed
        a, b, c = vs
        da, db, dc = d[a], d[b], d[c]
        na = nb = nc = 0
        for x, y, z in table:
            if da >> x & 1 and db >> y & 1 and dc >> z & 1:
                na, nb, nc = na | 1 << x, nb | 1 << y, nc | 1 << z
        if not (na and nb and nc):
            return None
        changed = [v for v, new in ((a, na), (b, nb), (c, nc)) if new != d[v]]
        d[a], d[b], d[c] = na, nb, nc
        return changed

    def propagate(d: list[int], queue: list[int]) -> bool:
        queued = set(queue)
        while True:
            while queue:
                ci = queue.pop()
                queued.discard(ci)
                changed = revise(ci, d)
                if changed is None:
                    return False
                for v in changed:
                    for cj in watch[v]:
                        if cj not in queued:
                            queued.add(cj)
                            queue.append(cj)
            touched: list[int] = []
            for group in groups:
                for v in group:
                    if d[v] & (d[v] - 1) == 0:  # decided: nobody else in the category can be there
                        for w in group:
                            if w != v and d[w] & d[v]:
                                d[w] &= ~d[v]
                                if not d[w]:
                                    return False
                                touched.append(w)
                for x in range(n):
                    holders = [v for v in group if d[v] >> x & 1]
                    if not holders:
                        return False
                    if len(holders) == 1 and d[holders[0]] != 1 << x:
                        d[holders[0]] = 1 << x
                        touched.append(holders[0])
            if not touched:
                return True
            for v in touched:
                for cj in watch[v]:
                    if cj not in queued:
                        queued.add(cj)
                        queue.append(cj)

    found: list[dict] = []

    def search(d: list[int], queue: list[int]) -> None:
        if not propagate(d, queue):
            return
        open_ = [(bin(m).count("1"), i) for i, m in enumerate(d) if m & (m - 1)]
        if not open_:
            found.append({var: d[i].bit_length() - 1 for var, i in index.items()})
            return
        _, i = min(open_)
        for x in range(n):
            if d[i] >> x & 1:
                child = list(d)
                child[i] = 1 << x
                search(child, list(watch[i]))
                if len(found) >= limit:
                    return

    search(dom, list(range(len(cons))))
    return found


def _candidates(rng: random.Random, sol: list[list[int]], n: int, kinds: list[str]) -> list[tuple]:
    k = len(sol)
    at = {(c, v): sol[c][v] for c in range(k) for v in range(n)}
    variables = list(at)
    out: list[tuple] = []
    for x in variables:
        out.append(("at", x, at[x]))
        out.extend(("not_at", x, p) for p in range(n) if p != at[x])
        if at[x] in (0, n - 1):
            out.append(("end", x, n - 1))
    for i, x in enumerate(variables):
        for y in variables[i + 1:]:
            if x[0] != y[0]:
                out.append(("same" if at[x] == at[y] else "diff", x, y))
            if at[x] == at[y]:
                continue
            a, b = (x, y) if at[x] < at[y] else (y, x)
            out.append(("left", a, b))
            if at[b] - at[a] == 1:
                out.extend([("next", a, b), ("adj", x, y)])
            if at[b] - at[a] == 2:
                out.append(("gap2", x, y))
    for x in variables:
        for c in range(k):
            if c == x[0]:
                continue
            match = next(v for v in range(n) if sol[c][v] == at[x])
            other = rng.choice([v for v in range(n) if v != match])
            pair = sorted([(c, match), (c, other)])
            out.append(("either", x, pair[0], pair[1]))
    if "between" in kinds:
        for x in variables:
            for _ in range(3):
                y, z = rng.sample(variables, 2)
                if y[0] != z[0] and x not in (y, z) and min(at[y], at[z]) < at[x] < max(at[y], at[z]):
                    out.append(("between", x, y, z))
    return [cl for cl in out if cl[0] in kinds]


def _render(theme: dict, clue: tuple, n: int) -> str:
    cats, pos_names = theme["categories"], theme["positions"]

    def name(x: Var) -> str:
        _, values, fmt = cats[x[0]]
        return fmt.format(v=values[x[1]])

    kind = clue[0]
    template = COMMON.get(kind) or theme[kind]
    refs = [x for x in clue[1:] if isinstance(x, tuple)]
    fill = {"X": name(refs[0])}
    if len(refs) > 1:
        fill["Y"] = name(refs[1])
    if len(refs) > 2:
        fill["Z"] = name(refs[2])
    if kind in ("at", "not_at"):
        fill["P"] = pos_names[clue[2]]
    text = template.format(**fill)
    return text[0].upper() + text[1:]


def grid(seed: int, theme: str = "workshop") -> Grid:
    spec = THEMES[theme]
    rng = random.Random(seed)
    n, k = len(spec["positions"]), len(spec["categories"])
    kinds = list(WEIGHTS) if theme == "street" else [x for x in WEIGHTS if x != "between"]
    sol = [rng.sample(range(n), n) for _ in range(k)]
    pool = _candidates(rng, sol, n, kinds)
    pool.sort(key=lambda cl: rng.random() ** (1 / WEIGHTS[cl[0]]), reverse=True)  # weighted shuffle
    clues: list[tuple] = []
    for cl in pool:
        clues.append(cl)
        if len(clues) >= n and len(solve(n, k, clues, limit=2)) == 1:
            break
    for cl in list(reversed(clues)):  # drop every clue the unique solution doesn't need
        rest = [c for c in clues if c is not cl]
        if len(solve(n, k, rest, limit=2)) == 1:
            clues = rest
    rng.shuffle(clues)
    answer = "".join(spec["categories"][0][1][v][0] for v in sorted(range(n), key=lambda v: sol[0][v]))
    lines = [spec["intro"], ""]
    for label, values, _ in spec["categories"]:
        lines.append(f"- {label}: " + ", ".join(values))
    lines += ["", "Clues:"]
    lines += [f"{i}. {_render(spec, cl, n)}" for i, cl in enumerate(clues, 1)]
    lines += ["", "Every clue is true, and the clues determine exactly one arrangement. " + spec["question"]]
    return Grid(prompt="\n".join(lines), answer=answer, solution=sol, clues=clues, theme=theme)


def logic_grid(seed: int = 1, theme: str = "workshop") -> dict:
    """task.yaml entry point."""
    g = grid(seed, theme)
    return {"prompt": g.prompt, "checks": [{"final_one_of": [g.answer, " ".join(g.answer), ", ".join(g.answer),
                                                             ",".join(g.answer)]}]}
