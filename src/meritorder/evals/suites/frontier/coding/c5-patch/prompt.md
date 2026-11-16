# Task: apply a unified diff

Write a Python module that applies a single-file unified diff to a text, the way `patch` does when the file has
drifted a little since the diff was made. Follow the rules below exactly; where they differ from what you know of
GNU or BSD `patch`, these rules win.

## Interface

```python
class PatchError(Exception):
    """Raised for a malformed diff or a hunk that can't be applied."""
    hunk: int   # the 1-based number of the hunk the error is about; 0 if the diff has no hunks at all

def apply_patch(original: str, diff: str, *, reverse: bool = False) -> tuple[str, list[int]]:
    """Returns (patched_text, offsets): offsets[k-1] is the offset at which hunk k was applied."""
```

`PatchError` must be an `Exception` subclass whose instances have an integer `hunk` attribute (construct it however
you like, e.g. `PatchError(message, hunk)`). Every error described below is a `PatchError`; nothing else is raised
for any `str` inputs.

## 1. Lines

A text is a sequence of lines. Only `"\n"` ends a line (`"\r"` is an ordinary character). Every line keeps its
terminating `"\n"`, except that the last line of a text may have none. The empty string has zero lines; `"a\n"` has
one line `"a\n"`; `"a\nb"` has two lines, `"a\n"` and `"b"`. Joining a text's lines gives the text back. Lines are
compared exactly, terminator included, so `"b"` and `"b\n"` are different lines.

The diff is split into lines the same way, and then each diff line's own terminating `"\n"` is dropped. (So a diff
that ends with `"\n"` has no extra empty last line, and one that doesn't end with `"\n"` is read the same way.)

## 2. Diff syntax

**Preamble.** Every line before the first diff line that starts with `@@` is ignored, whatever it contains
(`diff --git ...`, `index ...`, `--- a/file`, `+++ b/file`, notes, anything).

**Hunk header.** A line starting with `@@` begins a hunk. It must have exactly this form: `@@ -` then *A*, optionally
`,` followed by *B*, then ` +` then *C*, optionally `,` followed by *D*, then ` @@`, then the end of the line or any
further text at all (often a space and a function name). *A*, *B*, *C* and *D* are each one or more ASCII digits
`0`-`9`, read as decimal numbers. An omitted count (*B* or *D*) means 1. Single spaces exactly where shown; no other
spaces are allowed. *A*,*B* is the old range (*B* lines starting at line *A* of the old file, 1-based) and *C*,*D*
the new range. A non-empty range can't start at line 0: if *B* > 0 and *A* = 0, or *D* > 0 and *C* = 0, that's an
error. Any line starting with `@@` that doesn't have this form is an error for the hunk it would have started.

**Hunk body.** The lines after a header are the hunk's body, read one at a time until the hunk has exactly *B*
old-side lines and *D* new-side lines (immediately, if *B* = *D* = 0). Each body line is one of:

- `" "` + *text*: a context line, on both sides;
- `"-"` + *text*: a line on the old side only;
- `"+"` + *text*: a line on the new side only;
- an empty diff line (`""`): treated exactly like `" "` with empty text (editors often strip that trailing space).

The body line stands for the file line *text* + `"\n"` (but see the marker below). The prefix alone decides the kind
of line, so a body line `--- x` is an old-side line with text `-- x`, and `+++ y` a new-side line with text `++ y`.

While the hunk still needs lines, it is an error if: the diff ends; the next line is anything other than the four
forms above or a marker (for example a line starting with `@@` or with a letter); or a body line would give the old
side more than *B* lines or the new side more than *D* lines.

**No-newline marker.** A diff line that starts with a backslash (`\`, usually `\ No newline at end of file`, but any
text may follow it) says that the body line immediately before it has no `"\n"`: on both sides for a context line, on
the old side for a `-` line, on the new side for a `+` line. It doesn't count as a body line. A marker is allowed only
directly after a body line of the current hunk — including the body line that completes the hunk, so the marker may
be the line right after the body. A marker directly after a hunk header, or directly after another marker, is an
error. (Markers in the preamble are just ignored preamble lines.)

**After a hunk.** Once a hunk's body is complete (and after its marker, if one follows), the next diff line must start
a new hunk header, or the diff must end. Exception: if every remaining diff line is empty, they are ignored. Any other
line is an error for the hunk just completed.

Hunks are numbered 1, 2, 3, … in diff order. If the diff contains no line starting with `@@`, raise `PatchError` with
`hunk == 0`.

## 3. Direction

Normally (`reverse=False`) the **source** side of each hunk is its old side (context and `-` lines, range *A*,*B*)
and the **target** side is its new side (context and `+` lines, range *C*,*D*): the patch turns old into new. With
`reverse=True` the roles swap — the source is the new side (context and `+` lines, range *C*,*D*) and the target is
the old side (context and `-` lines) — so the patch is undone. Parsing and every check in section 2 are the same in
both directions. Below, *S*,*N* means the source range ((*A*,*B*) or (*C*,*D*)), and a hunk's source lines and target
lines are its lines of that side, in body order, each ending in `"\n"` unless a marker removed it.

## 4. Nominal positions and order

A hunk's **nominal position** is a 0-based index into the original text's lines: *S* - 1 if *N* > 0, and *S* if
*N* = 0 (an empty range at *S* means "just after line *S*"; *S* = 0 means "before the first line").

Hunks must be in order and must not overlap: for every hunk *k* ≥ 2, nominal(*k*) must be ≥ nominal(*k*-1) +
*N*(*k*-1); otherwise it's an error for hunk *k*.

The **entire diff is parsed and checked first** (sections 2 and 4, hunk by hunk in diff order, so the first problem
in diff order is the one reported), and only then are hunks located and applied. A malformed hunk 3 is therefore
reported even if hunk 1 wouldn't apply.

## 5. Locating each hunk

Let `lines` be the original text's lines and `L = len(lines)`. Hunks are located in order, each in the *original*
line numbering. For hunk *k*, with *N* source lines:

- `lo` = 0 for the first hunk; otherwise the position where hunk *k*-1 was applied plus *N*(*k*-1) (a hunk can never
  start inside, or before, the part of the original that the previous hunk consumed).
- `hi` = `L - N` (the last position at which *N* lines fit).
- `expected` = nominal(*k*) + the offset of hunk *k*-1 (0 for the first hunk). The previous hunk's offset is carried
  forward because a file that drifted usually drifted for every later hunk too.
- Try candidate positions in this order: `expected`, `expected+1`, `expected-1`, `expected+2`, `expected-2`, …
  (at equal distance, the later position first). Skip candidates outside `lo..hi`; keep going until every position
  in `lo..hi` has been tried. The first candidate `p` with `lines[p:p+N]` equal to the hunk's source lines (exact
  comparison, terminators included) is where the hunk applies. A hunk with *N* = 0 matches at any candidate.
- If no position in `lo..hi` matches (including when `lo > hi`), it's an error for hunk *k*.
- The hunk's **offset** is `p - nominal(k)`.

There is no fuzz (every source line must match exactly) and no special anchoring of hunks to the start or end of the
file: the search above is the only rule.

## 6. Result

The patched text is the original with, for each hunk, the `N` lines at its position replaced by its target lines;
everything else is kept. Return the patched text (lines joined with nothing in between) and the list of offsets, one
per hunk in order. A hunk's context lines come out the same either way, since they matched exactly.

## Examples

```python
diff = """--- a/f.txt
+++ b/f.txt
@@ -2,3 +2,3 @@
 b
-c
+C
 d
"""
apply_patch("a\nb\nc\nd\ne\n", diff)           # ("a\nb\nC\nd\ne\n", [0])
apply_patch("new\na\nb\nc\nd\ne\n", diff)      # ("new\na\nb\nC\nd\ne\n", [1])   applied one line later
apply_patch("a\nb\nC\nd\ne\n", diff, reverse=True)   # ("a\nb\nc\nd\ne\n", [0])

# a pure insertion (as from `diff -U0`): two lines after line 1
apply_patch("one\ntwo\n", "@@ -1,0 +2,2 @@\n+x\n+y\n")   # ("one\nx\ny\ntwo\n", [0])

# removing the final newline
apply_patch("a\nb\n", "@@ -2 +2 @@\n-b\n+b\n\\ No newline at end of file\n")   # ("a\nb", [0])

apply_patch("a\n", "@@ -1 +1 @@\n-z\n+y\n")    # raises PatchError with .hunk == 1 (doesn't apply)
apply_patch("a\n", "just some text\n")         # raises PatchError with .hunk == 0
```

Reply with exactly one ```python code block containing the complete module. Use only the Python 3.12 standard
library; the module must not print or do any work at import time.
