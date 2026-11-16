# Path glob matcher and ignore rules

Write a Python module with two functions:

```python
def match(pattern: str, path: str) -> bool: ...
def ignored(rules: list[str], path: str) -> bool: ...
```

You may not import `re`, `fnmatch`, `glob` or `pathlib`; implement the matching yourself.

## Paths

A path is a non-empty string made of one or more non-empty **segments** separated by single `/` characters, with no
leading or trailing `/`. Every path passed to your functions is valid in this sense (you don't need to check). A
segment may contain any character except `/`, including spaces, `*`, `[`, `\`, `.` and non-ASCII characters.
Characters are compared exactly (case-sensitive, by code point).

## Pattern syntax

1. The pattern is split at **every** `/` character into segment patterns. A `/` can never be escaped and never
   appears inside a bracket expression (so `[a/b]` is the two segment patterns `[a` and `b]`). The pattern is
   **invalid** if any segment pattern is empty: the empty pattern, a leading or trailing `/`, or `//`.
2. A segment pattern that is exactly `**` is a **globstar**. Any other segment pattern is a sequence of elements,
   read left to right:
   - `\` followed by any character: a literal element matching exactly that character (`\*` matches `*`, `\a`
     matches `a`, `\\` matches `\`). A `\` that is the last character of a segment pattern makes the pattern invalid.
   - `*`: matches any sequence of characters, including the empty one.
   - `?`: matches exactly one character.
   - `[`: starts a bracket expression (rule 3).
   - any other character: a literal element matching exactly that character.

   So `**` that is not a whole segment pattern (`a**`, `**.py`, `x**y`) is simply two `*` elements.
3. **Bracket expressions.** After the `[`, an optional `!` makes the expression negated (`^` has no special meaning
   anywhere). Then items are read until the closing `]`:
   - Read a character c. If c is `\`, it is replaced by the next character, taken literally (so `\]`, `\-`, `\\` are
     items); a `\` with nothing after it in the segment pattern makes the pattern invalid. An unescaped `]` closes the
     expression, **except** when it is the very first character after `[` (after `[!` if negated): there it is a
     literal item.
   - After reading c, if the next character is `-` and there is a character after that `-` which is not `]`, this is
     a range: read d the same way as c (it may be escaped) and the item is every character whose code point is from
     `ord(c)` to `ord(d)` inclusive. If `ord(c) > ord(d)` the pattern is invalid. Otherwise c alone is the item (so a
     `-` that is first, or last before the `]`, is literal: `[-a]`, `[a-]`; `[a-c-e]` is `a`..`c`, `-`, `e`; and
     a literal first `]` can start a range too: `[]-a]` is the range `]`..`a`). Any other character inside the
     brackets, including `[`, `!` (when not first), `*` and `?`, is an ordinary item.
   - If the segment pattern ends before the closing `]`, the pattern is invalid. Consequently `[]` and `[!]` are
     invalid (the `]` is a literal item and nothing closes the expression), while `[]]` matches `]` and `[!]]`
     matches any character except `]`.

   A bracket expression matches exactly one character that is in the set of items (negated: not in the set).

## Matching

`match(pattern, path)` returns True when the path's segments can be matched, in order, by the pattern's segment
patterns, where:

- a non-globstar segment pattern matches exactly one path segment: its elements must match the whole segment;
- a globstar matches **zero or more** consecutive whole path segments.

So `**` matches every path (subject to the dot rule below); `a/**` matches `a` itself and everything under `a`;
`**/b` matches `b` and `x/y/b`; `a/**/b` matches `a/b` and `a/x/y/b`; `*` never matches across a `/`.

**Dot rule.** A path segment that starts with `.` can only be matched by a segment pattern whose first element is
a literal `.` (written `.` or `\.`). In particular: `*`, `?` and bracket expressions (negated or not, even `[.]`)
never match the `.` at the start of a segment; a `*` that matches the empty string does not let a following literal
`.` match a segment's leading dot (`*.py` does not match `.py`); and a globstar never matches a segment that starts
with `.`. A `.` anywhere else in a segment is an ordinary character (`*` matches `a.b.c`).

**Validation.** The whole pattern is checked before any matching. An invalid pattern raises `ValueError`, whatever
the path is, even if the path could be rejected without looking at the invalid part (`match("a/[", "b/c")` raises).

**Speed.** Matching must never take exponential time. Patterns have at most 64 characters and paths at most 300;
every call, including adversarial ones such as `*a*a*a*a*a*a*a*a*a*a*b` against a long run of `a`s or many
globstars against a deep path, must return in well under one second.

## Ignore rules

`ignored(rules, path)` decides whether `path` is ignored by a list of gitignore-like rules:

1. A rule that is the empty string or starts with `#` is skipped. (Any other rule is used, so `!#x` is a negated rule
   whose pattern is `#x`. To match a name that starts with `#` or `!`, escape it: `\#x`, `\!x`.)
2. If the rule starts with `!`, it is a **negation** rule; remove that one `!`.
3. If what remains starts with `/`, remove that one `/`; the rule is **anchored**.
4. What remains is the rule's pattern, which must be valid (rules above). If the pattern contains a `/`, the rule is
   also anchored. A rule that is not anchored is **unanchored**.
5. An anchored rule matches the path if its pattern matches the path itself or any of its ancestors (for `a/b/c`:
   `a`, `a/b`, `a/b/c`). An unanchored rule matches the path if its pattern matches any single segment of the path
   (for `a/b/c`: `a`, `b` or `c`).
6. The **last** rule that matches decides: a negation rule means not ignored (False), any other rule means ignored
   (True). If no rule matches, the result is False. Unlike git, a negation rule can re-include a path even when one
   of its parent directories is ignored.

Every rule that isn't skipped is validated: if any rule's pattern is invalid (including an empty pattern, as in the
rules `!`, `/` and `!/`), raise `ValueError`, even if that rule could not change the result.

## Examples

```python
match("*.py", "main.py")              # True
match("*.py", "src/main.py")          # False (* stays inside one segment)
match("src/**/*.py", "src/main.py")   # True  (** matches zero segments)
match("**/test_*.py", "a/b/test_x.py")  # True
match("*.py", ".py")                  # False (dot rule)
match(".*", ".bashrc")                # True
match("**/x", ".git/x")               # False (a globstar never matches .git)
match("[!a-c]x", "dx")                # True
match("[]a]", "]")                    # True
match("a[", "a[")                     # raises ValueError (unclosed bracket expression)
ignored(["*.log", "!keep.log"], "logs/keep.log")   # False (last matching rule is a negation)
ignored(["/build"], "src/build")      # False (anchored: only build and build/... at the top)
ignored(["build"], "src/build/x.o")   # True  (unanchored: the segment build matches)
```

Reply with exactly one ```python code block containing the complete module. Use only the Python 3.12 standard
library; the module must not print anything or do any work at import time.
