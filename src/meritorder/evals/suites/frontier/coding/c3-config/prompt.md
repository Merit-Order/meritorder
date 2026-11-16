Write a Python module that parses a small INI-like configuration language into nested dicts. It must define:

```python
class ConfigError(Exception): ...   # must have an int attribute `line`: the 1-based line number of the error

def parse_config(text: str) -> dict: ...
```

`parse_config` raises `ConfigError` for every error described below. Parsing goes line by line from the top and stops at the first invalid line, so `line` is the number of the first line that breaks a rule.

## Lines

Split `text` on `"\n"`; if a line then ends with `"\r"`, that one `"\r"` is removed. Lines are numbered from 1. In this spec, *whitespace* means spaces and tabs. Each line, with surrounding whitespace removed, is one of:

- **empty**: ignored.
- a **comment**: it starts with `#` or `;`. Ignored.
- a **section header**: it starts with `[`.
- otherwise a **key/value pair**.

## Section headers

`[NAME]`, where `NAME` may have whitespace around it inside the brackets and is one or more *parts* joined by single dots. Each part is one or more of the characters `A-Z a-z 0-9 _ -` (ASCII only), with no whitespace. After the closing `]` the line may contain only whitespace, optionally followed by a comment that starts with `#`. Anything else (`[]`, `[a..b]`, `[a b]`, `[a. b]`, `[a`, `[a]]`, `[a] x`) is an error.

A header `[a.b.c]` makes `result["a"]["b"]["c"]` the current table, creating any missing dicts along the way. Key/value pairs before the first header go into the top-level dict. The same dotted name may appear in a header only once (a second `[a.b]` is an error). A table created implicitly by a longer header may be declared later exactly once: `[a.b]` followed by `[a]` is fine.

## Key/value pairs

`KEY = VALUE`, split at the first `=`. No `=` on the line is an error. `KEY`, with surrounding whitespace removed, must be one or more of `A-Z a-z 0-9 _ -` (ASCII only, so no dots and no spaces). Within one table a name may be used only once: a key repeated in the same table, a header naming a table whose name is already used by a key (or whose path runs through a key's value), and a key whose name is already used by a table are all errors, reported on the line that causes the clash. Section names and keys are case-sensitive.

The value is the text after the `=`, with leading whitespace skipped. Its first character decides its type:

1. `"`: a **basic string**, ending at the next `"` that isn't escaped. Escapes: `\\`, `\"`, `\n` (newline), `\t` (tab), and `\uXXXX` with exactly four hex digits (either case) giving that code point, which must not be a surrogate (`D800`-`DFFF`). Any other backslash sequence, a bad `\u`, or no closing quote on the line is an error.
2. `'`: a **literal string**, ending at the next `'`. No escapes: backslashes are kept. No closing quote on the line is an error.
3. `[`: a **list** on one line. Elements are separated by commas, with optional whitespace around each element and comma. An element must be a basic string, a literal string, a number, `true` or `false` (no bare strings, no nested lists). A comma after the last element is allowed. `[]` and `[ ]` are empty lists. Commas and brackets inside quoted elements are ordinary characters. A missing `]`, an empty element (`[,]`, `[1,,2]`), or an invalid element is an error.
4. After a basic string, literal string or list, the rest of the line may contain only whitespace, optionally followed by a comment that starts with `#`; anything else is an error.
5. Anything else is **unquoted**. The unquoted text runs to the end of the line, except that a `#` starts a comment (ending the text) when it is the first character of the value or the character just before it is a space or tab. Trailing whitespace is then removed. The text is:
   - empty: the value is `""`;
   - exactly `true` or `false`: the `bool`;
   - if it starts with a digit, `+`, `-` or `.`: it must be a number (below), otherwise it's an error;
   - otherwise a **bare string**, kept exactly (inner spaces, quotes, `=`, `;`, backslashes and all).

## Numbers

Let *digits* mean one or more ASCII digits where single underscores may appear between two digits (`1_000`, not `1__0`, `_1` or `1_`).

- An **integer** is an optional `+` or `-`, then *digits* that are either exactly `0` or don't start with `0` (so no leading zeros: `012`, `00`, `0_1` and `-01` are invalid; `7`, `0`, `10` and `1_0` are fine). Its value is a Python `int`; `-0` is `0`.
- A **float** is an integer as above (same sign and leading-zero rules), followed by a fraction `.` *digits*, an exponent `e` or `E` with an optional `+` or `-` and then *digits*, or both (fraction first). Its value is the Python `float` of the text without underscores. So `1e3` is `1000.0`, while `.5`, `1.`, `1.e5`, `1e` and `01.5` are invalid.

## Example

```python
parse_config('''
# service settings
name = demo server   # trailing comment
port = 8_080
[db.primary]
hosts = ["a:1", 'b:2',]
ratio = 2.5e-1
url = postgres://u@h/db?x=1#frag
''') == {"name": "demo server", "port": 8080,
         "db": {"primary": {"hosts": ["a:1", "b:2"], "ratio": 0.25, "url": "postgres://u@h/db?x=1#frag"}}}
```

Reply with exactly one ```python code block containing the complete module. Use only the Python 3.12 standard library; the module must not print or do any work at import time.
