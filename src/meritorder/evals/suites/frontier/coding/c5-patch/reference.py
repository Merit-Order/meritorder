"""Reference solution for c5-patch: apply a single-file unified diff with offset search and reverse mode."""

DIGITS = "0123456789"


class PatchError(Exception):
    def __init__(self, message: str, hunk: int = 0):
        super().__init__(message)
        self.hunk = hunk


def _lines(text: str) -> list[str]:
    if not text:
        return []
    parts = text.split("\n")
    out = [p + "\n" for p in parts[:-1]]
    if parts[-1]:
        out.append(parts[-1])
    return out


def _header(line: str) -> tuple[int, int, int, int] | None:
    pos = 0

    def lit(s: str) -> bool:
        nonlocal pos
        if line.startswith(s, pos):
            pos += len(s)
            return True
        return False

    def num() -> int | None:
        nonlocal pos
        start = pos
        while pos < len(line) and line[pos] in DIGITS:
            pos += 1
        return int(line[start:pos]) if pos > start else None

    def rng() -> tuple[int, int] | None:
        start = num()
        if start is None:
            return None
        count = 1
        if lit(","):
            count = num()
            if count is None:
                return None
        return start, count

    if not lit("@@ -"):
        return None
    old = rng()
    if old is None or not lit(" +"):
        return None
    new = rng()
    if new is None or not lit(" @@"):
        return None
    return old[0], old[1], new[0], new[1]


def _parse(diff: str) -> list[dict]:
    dl = [ln[:-1] if ln.endswith("\n") else ln for ln in _lines(diff)]
    n = len(dl)
    i = 0
    while i < n and not dl[i].startswith("@@"):
        i += 1
    if i == n:
        raise PatchError("the diff has no hunks", 0)
    hunks: list[dict] = []
    while True:
        k = len(hunks) + 1
        hdr = _header(dl[i])
        if hdr is None:
            raise PatchError(f"malformed hunk header: {dl[i]!r}", k)
        a, b, c, d = hdr
        if (b > 0 and a == 0) or (d > 0 and c == 0):
            raise PatchError("a non-empty range can't start at line 0", k)
        i += 1
        body: list[list] = []
        old = new = 0
        can_mark = False
        while True:
            if i < n and dl[i].startswith("\\"):
                if not can_mark:
                    raise PatchError("misplaced '\\' marker", k)
                body[-1][2] = False
                can_mark = False
                i += 1
                continue
            if old == b and new == d:
                break
            if i >= n:
                raise PatchError("the diff ends inside a hunk", k)
            line = dl[i]
            if line == "":
                kind, content = " ", ""
            elif line[0] in " -+":
                kind, content = line[0], line[1:]
            else:
                raise PatchError(f"unexpected line in hunk body: {line!r}", k)
            old += kind in " -"
            new += kind in " +"
            if old > b or new > d:
                raise PatchError("hunk body is longer than its header says", k)
            body.append([kind, content, True])
            can_mark = True
            i += 1
        hunks.append({"old": (a, b), "new": (c, d), "body": body})
        if i >= n:
            return hunks
        if dl[i].startswith("@@"):
            continue
        if all(ln == "" for ln in dl[i:]):
            return hunks
        raise PatchError(f"unexpected line after hunk: {dl[i]!r}", k)


def _nominal(start: int, count: int) -> int:
    return start - 1 if count > 0 else start


def apply_patch(original: str, diff: str, *, reverse: bool = False) -> tuple[str, list[int]]:
    hunks = _parse(diff)
    src_kinds, tgt_kinds, side = (" +", " -", "new") if reverse else (" -", " +", "old")
    plan = []
    for k, h in enumerate(hunks, 1):
        start, count = h[side]
        nominal = _nominal(start, count)
        if plan and nominal < plan[-1][0] + plan[-1][1]:
            raise PatchError("hunks are out of order or overlap", k)
        src = [c + ("\n" if nl else "") for kind, c, nl in h["body"] if kind in src_kinds]
        tgt = [c + ("\n" if nl else "") for kind, c, nl in h["body"] if kind in tgt_kinds]
        plan.append((nominal, count, src, tgt))

    lines = _lines(original)
    total = len(lines)
    out: list[str] = []
    offsets: list[int] = []
    lo = 0
    prev_offset = 0
    for k, (nominal, count, src, tgt) in enumerate(plan, 1):
        hi = total - count
        expected = nominal + prev_offset
        found = None
        if lo <= hi:
            for dist in range(max(0, expected - hi, lo - expected), max(hi - expected, expected - lo) + 1):
                for p in ((expected,) if dist == 0 else (expected + dist, expected - dist)):
                    if lo <= p <= hi and lines[p:p + count] == src:
                        found = p
                        break
                if found is not None:
                    break
        if found is None:
            raise PatchError("hunk doesn't apply", k)
        out.extend(lines[lo:found])
        out.extend(tgt)
        lo = found + count
        prev_offset = found - nominal
        offsets.append(prev_offset)
    out.extend(lines[lo:])
    return "".join(out), offsets
