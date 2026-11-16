# Task: a backtracking regular-expression engine

Write a Python module that implements regular-expression matching for the subset of Python's `re` syntax described
below, **without using the `re` module** (importing `re` — or `sre_compile`, `sre_parse`, `_sre`, `fnmatch` — makes
the submission fail). Everything the tests check is specified here.

For every valid pattern of this subset, your results must be identical to what Python 3.12's `re` module returns
when called with the `re.ASCII` flag and no other flags. The rules below spell that behaviour out; where they say
something is invalid, that is a rule of this task even if Python would accept it, and `sub`'s replacement string is
always taken literally (rule 9 below).

## API

```python
class Match:
    def group(self, g: int = 0) -> str | None: ...
    def span(self, g: int = 0) -> tuple[int, int]: ...
    def groups(self) -> tuple[str | None, ...]: ...

def search(pattern: str, text: str) -> Match | None: ...
def fullmatch(pattern: str, text: str) -> Match | None: ...
def findall(pattern: str, text: str) -> list: ...
def sub(pattern: str, repl: str, text: str, count: int = 0) -> str: ...
```

- Group 0 is the whole match; groups 1..N are the capturing groups, numbered by the position of their opening `(`
  from left to right (nesting doesn't matter: in `((a)(b))` group 1 is the outer one, 2 is `(a)`, 3 is `(b)`).
- `m.span(g)` is `(start, end)` (end exclusive) of what group `g` captured, or `(-1, -1)` if the group did not take
  part in the match. `m.group(g)` is the captured substring, or `None` if the group did not take part.
  `m.groups()` is `(m.group(1), ..., m.group(N))` (an empty tuple when there are no groups).
- `g` must be an `int` from 0 to N; anything else raises `IndexError`.
- An invalid pattern raises `ValueError` from every one of the four functions, before any matching and even when the
  text is empty.

## Syntax

A pattern is one or more **branches** separated by `|`; a branch is a sequence (possibly empty) of **atoms**, each
optionally followed by one **quantifier**.

**Atoms**

- An ordinary character matches itself (case-sensitive; any Unicode character can be ordinary). The special
  characters are `\ . [ ( ) | * + ? { ^ $`; `]` and `}` on their own are ordinary, and `{` is ordinary whenever it
  does not start a valid quantifier (see below).
- `.` matches any single character except a newline `"\n"`.
- `^` matches the empty string at position 0 only. `$` matches the empty string at the end of the text, and also
  just before a newline that is the last character of the text (so `a$` matches in `"a\n"`, but not in `"a\n\n"`).
  There is no multi-line mode: `^` and `$` never match anywhere else.
- `(...)` is a capturing group, `(?:...)` a non-capturing group. Both may contain alternation and may be empty.
  `(?` not immediately followed by `:` is invalid (so lookarounds, named groups and inline flags are invalid). A `(`
  without its `)` and a `)` without its `(` are invalid.
- Escapes (outside and inside character classes):
  - `\d` = ASCII digits `0-9`; `\w` = ASCII letters, digits and `_`, i.e. `[A-Za-z0-9_]`; `\s` = exactly the six
    characters space, `\t`, `\n`, `\r`, `\f`, `\v`. `\D`, `\W`, `\S` match any single character *not* in the
    corresponding set. These are ASCII-only: `é` is not a `\w`, `٣` is not a `\d`, `\x1c` is not a `\s`.
  - `\n`, `\t`, `\r`, `\f`, `\v` are the newline, tab, carriage-return, form-feed and vertical-tab characters.
  - A backslash followed by any character that is **not** an ASCII letter or digit matches that character
    literally (`\.`, `\*`, `\\`, `\-`, `\]`, `\{`, `\ `, `\_`, `\/`, ...).
  - A backslash followed by any other ASCII letter or digit (`\q`, `\b`, `\A`, `\1`, `\x`, ...) is invalid, and so is
    a backslash at the very end of the pattern.
- **Character class** `[...]` matches one character that is in the set; `[^...]` one character that is *not* in it.
  - After `[` (and after the `^` of `[^`), the very first character may be `]`, which is then an ordinary member.
    Otherwise `]` closes the class. A class with no closing `]` is invalid — so `[]` and `[^]` are invalid, while
    `[]a]` is the set {`]`, `a`}.
  - Members are read left to right. A member is a single character (ordinary, or a one-character escape such as
    `\]`, `\-`, `\n`) or a set escape (`\d \D \w \W \s \S`). Inside a class every character other than `\`, the
    closing `]` and the range `-` described next is ordinary (`.`, `*`, `(`, `|`, `$`, `[`, and `^` when not
    first, are just characters).
  - After reading a member, if the next character is `-`: when the character after that `-` is the closing `]`, the
    member and a literal `-` are added and the class ends (`[a-]` is {`a`, `-`}). Otherwise the `-` and the next
    member form a range from the first member to the second (by code point, inclusive). Both ends of a range must
    be single characters, not set escapes, and the start must not be greater than the end; otherwise the pattern is
    invalid (`[z-a]`, `[\d-z]`, `[a-\w]`). A range is finished once read: in `[a-c-e]` the second `-` is read as a
    new member, so the set is {`a`, `b`, `c`, `-`, `e`}. A `-` that is the first member is literal: `[-a]`, `[^-a]`.

**Quantifiers**, applied to the atom just before them:

| quantifier | repetitions |
|---|---|
| `*` | 0 or more |
| `+` | 1 or more |
| `?` | 0 or 1 |
| `{m}` | exactly m |
| `{m,}` | m or more |
| `{,n}` | 0 to n |
| `{m,n}` | m to n |
| `{,}` | 0 or more |

- `m` and `n` are one or more ASCII digits; no spaces are allowed. A `{` that is not followed by text forming exactly
  one of these shapes up to a `}` is an ordinary character, and then so are the characters after it (`a{`, `a{x}`,
  `a{1,2`, `a{ 2}`, `a{}` are all literal text). `{m,n}` with m > n is invalid.
- A quantifier followed by `?` is **lazy** (`*?`, `+?`, `??`, `{m,n}?`, ...); otherwise it is **greedy**.
- A quantifier with nothing to repeat is invalid: at the start of the pattern, right after `(`, `(?:` or `|`, or
  right after `^` or `$` (anchors can't be repeated). This includes `{2}` at the start (a valid quantifier shape),
  but not `{x}` (ordinary text).
- A quantifier may not directly follow another quantifier or its lazy `?`: `a**`, `a+*`, `a???`, `a*?*`, `a{2}{3}`
  and also `a*+`, `a++`, `a?+`, `a{2}+` (possessive quantifiers are **not** part of this subset) are invalid.

## Matching semantics

Matching is a backtracking search over choices, tried in **priority order**; the first complete match found is the
result.

1. **Alternation**: branches are tried left to right.
2. **Greedy** quantifiers prefer doing one more repetition (when allowed) over stopping; **lazy** quantifiers prefer
   stopping over doing one more. If the rest of the pattern then fails, the other choice is tried (backtracking).
3. **Repetitions**: for a quantifier with bounds m..n, the first m repetitions are *mandatory*; repetitions after
   those are *optional* and are only possible while fewer than n have been done. **Empty-repetition rule**: an
   optional repetition that matched the empty string ends the loop — after it, no further repetition is attempted
   and matching continues with what follows the quantified atom (that empty repetition, and anything it captured,
   still counts). Mandatory repetitions may match the empty string and never trigger this rule. Example: `(a|)*`
   on `"aa"` matches `"aa"` and group 1 is the empty string at `(2, 2)`; `(?:()|a){1,2}b` on `"ab"` matches `"ab"`
   with group 1 at `(0, 0)` (the mandatory first repetition captured it, the optional second one consumed `a`).
4. **Captures**: each time a group finishes matching, its span is recorded, overwriting any earlier value. Groups
   are *not* reset when an enclosing loop starts a new repetition: if a later repetition doesn't pass through a
   group, the group keeps the value from the earlier repetition (`(?:(a)|b)+` on `"ab"` matches `"ab"` with group
   1 = `"a"`). Captures made on a path that is later abandoned by backtracking are undone. A group that never
   captured on the successful path did not take part in the match.
5. **`search`** tries start positions 0, 1, ..., len(text) in that order and returns the highest-priority match at
   the first start position where one exists (so a match starting earlier always wins, whatever the priorities).
6. **`fullmatch`** only considers start position 0, and a path only counts as a match if it ends exactly at
   len(text): backtracking continues into lower-priority choices until one ends there (so `fullmatch("a|ab", "ab")`
   matches `"ab"`, and `fullmatch("a*?", "aaa")` matches `"aaa"`).
7. **`findall` and `sub`** find successive non-overlapping matches. The first search starts at position 0. After a
   match spanning `(s, e)`, the next search starts at `e`. If that previous match was empty (`s == e`), the next
   match may not be an empty match *at position `e`*: at `e` the engine looks for the highest-priority non-empty
   match starting there, and if there is none it continues with start positions `e + 1`, `e + 2`, ... where empty
   matches are allowed again. An empty match is allowed right after a non-empty one. Stop when no further match
   exists (searches never start beyond len(text)). Examples: `findall("a*", "baa")` is `["", "aa", ""]`;
   `findall("a??", "aa")` is `["", "a", "", "a", ""]`; `sub("x*", "-", "abxd")` is `"-a-b--d-"`.
8. `findall` returns a list with one item per match: the whole matched string if the pattern has no groups; the
   string captured by group 1 if it has exactly one group; otherwise a tuple of all groups' strings. A group that
   did not take part contributes `""` (never `None`).
9. `sub` replaces each match with `repl`, which is used **literally** (backslashes in `repl` are not special). With
   `count > 0`, only the first `count` matches are replaced; `count == 0` replaces all of them. Text between
   matches is copied unchanged.

## Limits

Tests use patterns of at most 60 characters and texts of at most 40 characters, and no test needs more than a few
thousand backtracking steps (no catastrophic patterns). A recursive backtracker may still need more than Python's
default 1000 frames; raising the limit with `sys.setrecursionlimit` is fine.

## Examples

```python
search(r"(\d+)-(\d+)", "tel 555-0199 x").span()       # (4, 12)
search(r"(\d+)-(\d+)", "tel 555-0199 x").groups()     # ("555", "0199")
search("a|ab", "ab").group()                          # "a"
search("(a|ab)(c|bcd)(d*)", "abcd").groups()          # ("a", "bcd", "")
search("<.+?>", "<a><b>").group()                     # "<a>"
search("(\\w)+", "abc").group(1)                      # "c"
search("(a)|b", "b").span(1)                          # (-1, -1)
search("x*", "abc").span()                            # (0, 0)
findall("(a)(x)?", "aa")                              # [("a", ""), ("a", "")]
sub("a", "X", "aaa", 2)                               # "XXa"
search("a{,2}", "aaa").span()                         # (0, 2)
search("a{x}", "a{x}").span()                         # (0, 4)
search("[^\\W\\d]+", "12ab_3").group()                # "ab_"
search("(", "")                                       # raises ValueError
```

Reply with exactly one ```python code block containing the complete module. Use only the Python 3.12 standard library
(and not `re`); the module must not print anything or do any work at import time beyond defining things.
