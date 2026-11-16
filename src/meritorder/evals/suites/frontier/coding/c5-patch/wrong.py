"""Plausible but buggy: each hunk's search starts at its nominal position (the previous hunk's offset isn't carried
forward), and an empty diff line inside a body is rejected instead of being read as an empty context line."""

import re

_HEADER = re.compile(r"@@ -([0-9]+)(?:,([0-9]+))? \+([0-9]+)(?:,([0-9]+))? @@")


class PatchError(Exception):
    def __init__(self, message, hunk=0):
        super().__init__(message)
        self.hunk = hunk


def _split(text):
    if text == "":
        return []
    out = text.split("\n")
    lines = [s + "\n" for s in out[:-1]]
    if out[-1] != "":
        lines.append(out[-1])
    return lines


def _parse(diff):
    rows = [ln.rstrip("\n") for ln in _split(diff)]
    i = 0
    while i < len(rows) and not rows[i].startswith("@@"):
        i += 1
    if i == len(rows):
        raise PatchError("no hunks", 0)
    hunks = []
    while i < len(rows):
        num = len(hunks) + 1
        m = _HEADER.match(rows[i])
        if not m:
            raise PatchError("bad header", num)
        a, b, c, d = (int(g) if g is not None else 1 for g in m.groups())
        if (b and not a) or (d and not c):
            raise PatchError("bad range", num)
        i += 1
        body = []
        need_old, need_new = b, d
        markable = False
        while True:
            if i < len(rows) and rows[i].startswith("\\"):
                if not markable:
                    raise PatchError("bad marker", num)
                body[-1][2] = False
                markable = False
                i += 1
                continue
            if need_old == 0 and need_new == 0:
                break
            if i >= len(rows):
                raise PatchError("truncated hunk", num)
            row = rows[i]
            if not row or row[0] not in " +-":
                raise PatchError("bad body line", num)
            tag = row[0]
            if tag in " -":
                need_old -= 1
            if tag in " +":
                need_new -= 1
            if need_old < 0 or need_new < 0:
                raise PatchError("hunk too long", num)
            body.append([tag, row[1:], True])
            markable = True
            i += 1
        hunks.append(((a, b), (c, d), body))
        if i < len(rows) and not rows[i].startswith("@@"):
            if any(r != "" for r in rows[i:]):
                raise PatchError("junk after hunk", num)
            break
    return hunks


def apply_patch(original, diff, *, reverse=False):
    hunks = _parse(diff)
    prepared = []
    for num, (old, new, body) in enumerate(hunks, 1):
        start, count = new if reverse else old
        src_tags, dst_tags = (" +", " -") if reverse else (" -", " +")
        nominal = start - 1 if count else start
        if prepared and nominal < prepared[-1][0] + prepared[-1][1]:
            raise PatchError("misordered hunks", num)
        src = [t + ("\n" if nl else "") for tag, t, nl in body if tag in src_tags]
        dst = [t + ("\n" if nl else "") for tag, t, nl in body if tag in dst_tags]
        prepared.append((nominal, count, src, dst))

    lines = _split(original)
    result, offsets, pos = [], [], 0
    for num, (nominal, count, src, dst) in enumerate(prepared, 1):
        hi = len(lines) - count
        where = None
        for dist in range(0, len(lines) + abs(nominal) + 2):
            for p in (nominal + dist, nominal - dist) if dist else (nominal,):
                if pos <= p <= hi and lines[p:p + count] == src:
                    where = p
                    break
            if where is not None:
                break
        if where is None:
            raise PatchError("hunk failed", num)
        result += lines[pos:where] + dst
        pos = where + count
        offsets.append(where - nominal)
    result += lines[pos:]
    return "".join(result), offsets
