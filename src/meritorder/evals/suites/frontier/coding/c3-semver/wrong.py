"""Plausible but buggy c3-semver: `$` accepts a trailing newline, and ^0.0.Z is treated like ^0.Y.Z."""

import re
from functools import cmp_to_key

_SEMVER = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?"
    r"(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
)


def parse(version):
    m = _SEMVER.match(version)
    if not m:
        raise ValueError(f"invalid version {version!r}")
    major, minor, patch, pre, build = m.groups()
    pre_ids = tuple(int(p) if p.isdigit() else p for p in pre.split(".")) if pre else ()
    build_ids = tuple(build.split(".")) if build else ()
    return int(major), int(minor), int(patch), pre_ids, build_ids


def _cmp(x, y):
    if x[:3] != y[:3]:
        return -1 if x[:3] < y[:3] else 1
    px, py = x[3], y[3]
    if not px and not py:
        return 0
    if not px:
        return 1
    if not py:
        return -1
    for a, b in zip(px, py):
        if a == b:
            continue
        if isinstance(a, int) and isinstance(b, int):
            return -1 if a < b else 1
        if isinstance(a, int):
            return -1
        if isinstance(b, int):
            return 1
        return -1 if a < b else 1
    return (len(px) > len(py)) - (len(px) < len(py))


def compare(a, b):
    return _cmp(parse(a), parse(b))


def sort_versions(versions):
    parsed = [(parse(v), v) for v in versions]
    return [v for _, v in sorted(parsed, key=cmp_to_key(lambda p, q: _cmp(p[0], q[0])))]


def _expand(token):
    m = re.match(r"(>=|<=|>|<|=|~|\^)?(.*)", token, re.S)
    op, body = m.group(1) or "=", m.group(2)
    v = parse(body)
    if op == "~":
        return [(">=", v), ("<", (v[0], v[1] + 1, 0, (), ()))]
    if op == "^":
        upper = (v[0] + 1, 0, 0, (), ()) if v[0] > 0 else (0, v[1] + 1, 0, (), ())
        return [(">=", v), ("<", upper)]
    return [(op, v)]


def satisfies(version, constraint):
    v = parse(version)
    ok = {">=": lambda c: c >= 0, "<=": lambda c: c <= 0, ">": lambda c: c > 0, "<": lambda c: c < 0, "=": lambda c: c == 0}
    result = False
    for alt in constraint.split("||"):
        tokens = alt.split()
        if not tokens:
            raise ValueError("empty alternative")
        comps = [c for t in tokens for c in _expand(t)]
        if all(ok[op](_cmp(v, bound)) for op, bound in comps):
            result = True
    return result
