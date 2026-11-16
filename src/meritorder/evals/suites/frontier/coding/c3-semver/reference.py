"""Reference solution for c3-semver."""

from functools import cmp_to_key

_DIGITS = frozenset("0123456789")
_IDENT = frozenset("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-")


def _number(s: str, what: str) -> int:
    if not s or not set(s) <= _DIGITS or (len(s) > 1 and s[0] == "0"):
        raise ValueError(f"invalid {what}: {s!r}")
    return int(s)


def _idents(s: str, what: str) -> list[str]:
    parts = s.split(".")
    for p in parts:
        if not p or not set(p) <= _IDENT:
            raise ValueError(f"invalid {what} identifier: {p!r}")
    return parts


def parse(version: str) -> tuple[int, int, int, tuple, tuple]:
    if not isinstance(version, str):
        raise ValueError("version must be a string")
    rest, plus, build_s = version.partition("+")
    if plus:
        build = tuple(_idents(build_s, "build"))
    else:
        build = ()
    core, dash, pre_s = rest.partition("-")
    pre: tuple = ()
    if dash:
        ids = []
        for p in _idents(pre_s, "prerelease"):
            ids.append(_number(p, "prerelease") if set(p) <= _DIGITS else p)
        pre = tuple(ids)
    nums = core.split(".")
    if len(nums) != 3:
        raise ValueError(f"invalid version core: {core!r}")
    major, minor, patch = (_number(n, "version number") for n in nums)
    return major, minor, patch, pre, build


def _cmp_parsed(x: tuple, y: tuple) -> int:
    if x[:3] != y[:3]:
        return -1 if x[:3] < y[:3] else 1
    px, py = x[3], y[3]
    if not px or not py:
        if px == py:
            return 0
        return -1 if px else 1
    for a, b in zip(px, py):
        if a == b:
            continue
        a_num, b_num = isinstance(a, int), isinstance(b, int)
        if a_num and b_num:
            return -1 if a < b else 1
        if a_num != b_num:
            return -1 if a_num else 1
        return -1 if a < b else 1
    if len(px) == len(py):
        return 0
    return -1 if len(px) < len(py) else 1


def compare(a: str, b: str) -> int:
    return _cmp_parsed(parse(a), parse(b))


def sort_versions(versions: list[str]) -> list[str]:
    keyed = [(parse(v), v) for v in versions]
    keyed.sort(key=cmp_to_key(lambda p, q: _cmp_parsed(p[0], q[0])))
    return [v for _, v in keyed]


def _comparator(token: str):
    for op in (">=", "<=", ">", "<", "=", "~", "^"):
        if token.startswith(op):
            body = token[len(op):]
            break
    else:
        op, body = "=", token
    v = parse(body)
    major, minor, patch = v[:3]
    if op == "~":
        upper = (major, minor + 1, 0, (), ())
        return [(">=", v), ("<", upper)]
    if op == "^":
        if major > 0:
            upper = (major + 1, 0, 0, (), ())
        elif minor > 0:
            upper = (0, minor + 1, 0, (), ())
        else:
            upper = (0, 0, patch + 1, (), ())
        return [(">=", v), ("<", upper)]
    return [(op, v)]


_OPS = {
    ">=": lambda c: c >= 0,
    "<=": lambda c: c <= 0,
    ">": lambda c: c > 0,
    "<": lambda c: c < 0,
    "=": lambda c: c == 0,
}


def satisfies(version: str, constraint: str) -> bool:
    v = parse(version)
    if not isinstance(constraint, str):
        raise ValueError("constraint must be a string")
    alternatives = []
    for alt in constraint.split("||"):
        tokens = alt.split()
        if not tokens:
            raise ValueError("empty alternative in constraint")
        checks = []
        for t in tokens:
            checks.extend(_comparator(t))
        alternatives.append(checks)
    return any(all(_OPS[op](_cmp_parsed(v, bound)) for op, bound in alt) for alt in alternatives)
