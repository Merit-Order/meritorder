"""Plausible but wrong c4-glob solution: the dot rule only stops wildcards from consuming a leading dot (so
`*.py` matches `.py`), and segment patterns are parsed lazily, so an invalid part the matcher never reaches is not
reported."""

GLOBSTAR = "globstar"


def _parse_bracket(seg, i):
    """seg[i] is just after '['. Returns (element, index after the closing ']')."""
    negated = False
    if i < len(seg) and seg[i] == "!":
        negated = True
        i += 1
    items = []
    first = True
    while True:
        if i >= len(seg):
            raise ValueError("unclosed bracket expression")
        c = seg[i]
        if c == "]" and not first:
            return ("class", negated, tuple(items)), i + 1
        first = False
        if c == "\\":
            if i + 1 >= len(seg):
                raise ValueError("trailing backslash")
            c = seg[i + 1]
            i += 2
        else:
            i += 1
        if i + 1 < len(seg) and seg[i] == "-" and seg[i + 1] != "]":
            i += 1
            d = seg[i]
            if d == "\\":
                if i + 1 >= len(seg):
                    raise ValueError("trailing backslash")
                d = seg[i + 1]
                i += 2
            else:
                i += 1
            if ord(c) > ord(d):
                raise ValueError("reversed range")
            items.append((ord(c), ord(d)))
        else:
            items.append((ord(c), ord(c)))


def _parse_segment(seg):
    if seg == "**":
        return GLOBSTAR
    elems, i = [], 0
    while i < len(seg):
        c = seg[i]
        if c == "\\":
            if i + 1 >= len(seg):
                raise ValueError("trailing backslash")
            elems.append(("lit", seg[i + 1]))
            i += 2
        elif c == "*":
            elems.append(("star",))
            i += 1
        elif c == "?":
            elems.append(("any",))
            i += 1
        elif c == "[":
            elem, i = _parse_bracket(seg, i + 1)
            elems.append(elem)
        else:
            elems.append(("lit", c))
            i += 1
    return tuple(elems)


class _Lazy(list):
    """Segment patterns, parsed the first time the matcher needs each one."""

    def __getitem__(self, i):
        raw = list.__getitem__(self, i)
        return raw if isinstance(raw, tuple) or raw == GLOBSTAR else _parse_segment(raw)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


def _compile(pattern):
    segs = pattern.split("/")
    if any(s == "" for s in segs):
        raise ValueError("empty segment in pattern")
    return _Lazy(s if s != "**" else GLOBSTAR for s in segs)


def _one(elem, ch):
    kind = elem[0]
    if kind == "lit":
        return elem[1] == ch
    if kind == "any":
        return True
    o = ord(ch)
    inside = any(lo <= o <= hi for lo, hi in elem[2])
    return inside != elem[1]


def _match_segment(elems, s):
    n = len(s)
    cur = [True] + [False] * n  # cur[j]: the elements so far match s[:j]
    for e in elems:
        nxt = [False] * (n + 1)
        if e[0] == "star":
            seen = False
            for j in range(n + 1):
                if j == 1 and s.startswith("."):
                    seen = False  # a star may not swallow the leading dot
                seen = seen or cur[j]
                nxt[j] = seen
        else:
            for j in range(n):
                if cur[j] and _one(e, s[j]) and not (j == 0 and s[0] == "." and e[0] != "lit"):
                    nxt[j + 1] = True
        cur = nxt
        if not any(cur):
            return False
    return cur[n]


def _match_compiled(compiled, path):
    parts = path.split("/")
    m = len(parts)
    cur = [True] + [False] * m  # cur[j]: the segment patterns so far match parts[:j]
    for p in compiled:
        nxt = [False] * (m + 1)
        if p == GLOBSTAR:
            for j in range(m + 1):
                nxt[j] = cur[j] or (j > 0 and nxt[j - 1] and not parts[j - 1].startswith("."))
        else:
            for j in range(m):
                if cur[j] and _match_segment(p, parts[j]):
                    nxt[j + 1] = True
        cur = nxt
        if not any(cur):
            return False
    return cur[m]


def match(pattern, path):
    return _match_compiled(_compile(pattern), path)


def ignored(rules, path):
    parsed = []
    for rule in rules:
        if rule == "" or rule.startswith("#"):
            continue
        negated = rule.startswith("!")
        if negated:
            rule = rule[1:]
        anchored = rule.startswith("/")
        if anchored:
            rule = rule[1:]
        compiled = _compile(rule)
        parsed.append((negated, anchored or "/" in rule, compiled))
    parts = path.split("/")
    ancestors = ["/".join(parts[:k]) for k in range(1, len(parts) + 1)]
    decision = False
    for negated, anchored, compiled in parsed:
        candidates = ancestors if anchored else parts
        if any(_match_compiled(compiled, c) for c in candidates):
            decision = not negated
    return decision
