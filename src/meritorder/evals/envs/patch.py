"""Apply a patch in the format OpenAI's models are trained to write (the ``apply_patch`` tool of Codex):

    *** Begin Patch
    *** Update File: src/app.py
    @@ def add(a, b):
    -    return a - b
    +    return a + b
    *** Add File: src/new.py
    +X = 1
    *** Delete File: src/old.py
    *** End Patch

An update's hunks start with ``@@``, optionally followed by a line to find first (a ``def`` or ``class`` line, say);
then context lines (a leading space), removed lines (``-``) and added lines (``+``). Context and removed lines must
match the file in order, after the previous hunk; trailing whitespace, then all surrounding whitespace, are
forgiven if nothing matches exactly. ``*** Move to: <path>`` after an update renames the file, and
``*** End of File`` ends a hunk that must match at the end of the file.

Either the whole patch applies or none of it does.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class PatchError(ValueError):
    pass


@dataclass
class _Hunk:
    anchor: str
    lines: list[tuple[str, str]] = field(default_factory=list)  # (" " context, "-" removed or "+" added, text)
    at_end: bool = False

    @property
    def old(self) -> list[str]:
        return [t for op, t in self.lines if op != "+"]


def _find(lines: list[str], block: list[str], start: int, at_end: bool) -> int:
    """Where ``block`` starts in ``lines`` at or after ``start``: exact, then ignoring trailing, then all
    surrounding whitespace. -1 if nowhere."""
    if not block:
        return len(lines) if at_end else start
    for norm in (lambda s: s, str.rstrip, str.strip):
        want = [norm(b) for b in block]
        positions = range(len(lines) - len(block), start - 1, -1) if at_end else range(start, len(lines) - len(block) + 1)
        for i in positions:
            if [norm(x) for x in lines[i:i + len(block)]] == want:
                return i
    return -1


def _update(text: str, hunks: list[_Hunk], path: str) -> str:
    lines = text.split("\n")
    trailing = lines[-1] == ""
    if trailing:
        lines.pop()
    cursor = 0
    for n, h in enumerate(hunks, 1):
        if h.anchor.strip():
            hit = next((i for i in range(cursor, len(lines)) if lines[i].strip() == h.anchor.strip()), None)
            if hit is None:
                hit = next((i for i in range(cursor, len(lines)) if h.anchor.strip() in lines[i]), None)
            if hit is None:
                raise PatchError(f"{path}, hunk {n}: can't find the line after @@: {h.anchor.strip()!r}")
            cursor = hit  # the hunk is at or after its anchor (some patches repeat the anchor as their first context line)
        at = _find(lines, h.old, cursor, h.at_end)
        if at < 0:
            shown = "\n".join(h.old[:4])
            raise PatchError(f"{path}, hunk {n}: these lines aren't in the file (after the previous hunk):\n{shown}")
        new, j = [], at
        for op, t in h.lines:
            if op == " ":
                new.append(lines[j])  # context keeps the file's own text, whatever whitespace the patch had
            if op != "+":
                j += 1
            else:
                new.append(t)
        lines[at:at + len(h.old)] = new
        cursor = at + len(new)
    return "\n".join(lines) + ("\n" if trailing or not lines else "")


def parse(patch: str) -> list[tuple[str, str, object]]:
    """[(op, path, data)]: ("add", path, text), ("delete", path, None), ("update", path, (hunks, move_to))."""
    raw = patch.replace("\r\n", "\n").strip("\n").split("\n")
    if raw and raw[0].strip() == "*** Begin Patch":
        raw = raw[1:]
    if raw and raw[-1].strip() == "*** End Patch":
        raw = raw[:-1]
    elif any(r.strip() == "*** End Patch" for r in raw):
        raw = raw[:next(i for i, r in enumerate(raw) if r.strip() == "*** End Patch")]
    ops: list[tuple[str, str, object]] = []
    i = 0
    while i < len(raw):
        line = raw[i]
        if not line.strip():
            i += 1
            continue
        for op in ("Add File: ", "Delete File: ", "Update File: "):
            if line.startswith("*** " + op):
                path = line[len("*** " + op):].strip()
                break
        else:
            raise PatchError(f"expected '*** Add File: ', '*** Update File: ' or '*** Delete File: ', got: {line[:120]!r}")
        i += 1
        if op == "Delete File: ":
            ops.append(("delete", path, None))
            continue
        body: list[str] = []
        while i < len(raw) and not raw[i].startswith(("*** Add File: ", "*** Delete File: ", "*** Update File: ")):
            body.append(raw[i])
            i += 1
        if op == "Add File: ":
            bad = [b for b in body if b and not b.startswith("+")]
            if bad:
                raise PatchError(f"{path}: every line of an added file starts with '+', not: {bad[0][:120]!r}")
            ops.append(("add", path, "\n".join(b[1:] for b in body) + "\n"))
            continue
        move = None
        if body and body[0].startswith("*** Move to: "):
            move = body.pop(0)[len("*** Move to: "):].strip()
        hunks: list[_Hunk] = []
        for b in body:
            if b.startswith("@@"):
                hunks.append(_Hunk(anchor=b[2:].strip().removesuffix("@@").strip()))
                continue
            if b.strip() == "*** End of File":
                if hunks:
                    hunks[-1].at_end = True
                continue
            if not hunks:
                hunks.append(_Hunk(anchor=""))
            h = hunks[-1]
            if b.startswith(("+", "-")):
                h.lines.append((b[0], b[1:]))
            else:  # context: a leading space (models often drop it on an empty line)
                h.lines.append((" ", b[1:] if b.startswith(" ") else b))
        if not hunks and not move:
            raise PatchError(f"{path}: an update with no hunks")
        ops.append(("update", path, (hunks, move)))
    if not ops:
        raise PatchError("the patch changes nothing: it needs '*** Update File: ', '*** Add File: ' or '*** Delete File: ' sections")
    return ops


def apply(files: dict[str, str], patch: str, norm=lambda p: p) -> tuple[dict[str, str], list[str]]:
    """The files after the patch, and what it did. ``norm`` checks and normalizes each path (None: refused)."""
    out = dict(files)
    done: list[str] = []
    for op, path, data in parse(patch):
        p = norm(path)
        if p is None:
            raise PatchError(f"not a path in the repository: {path!r}")
        if op == "add":
            if p in out:
                raise PatchError(f"{p} already exists: use '*** Update File: '")
            out[p] = str(data)
            done.append(f"added {p}")
        elif op == "delete":
            if p not in out:
                raise PatchError(f"no such file to delete: {p}")
            del out[p]
            done.append(f"deleted {p}")
        else:
            hunks, move = data  # type: ignore[misc]
            if p not in out:
                raise PatchError(f"no such file to update: {p} (use '*** Add File: ' for a new one)")
            text = _update(out[p], hunks, p) if hunks else out[p]
            if move:
                m = norm(move)
                if m is None:
                    raise PatchError(f"not a path in the repository: {move!r}")
                del out[p]
                out[m] = text
                done.append(f"updated {p} and moved it to {m}")
            else:
                out[p] = text
                done.append(f"updated {p}")
    return out, done
