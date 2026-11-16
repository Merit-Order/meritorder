# Streaming JSON tokenizer

Write a Python module with a streaming JSON tokenizer. Text arrives in chunks that may be cut anywhere (inside a
string, an escape sequence, a number or a keyword); the tokenizer emits each token as soon as it is complete, checks
the full JSON grammar (RFC 8259) as it goes, and reports the earliest possible error position. You may not import
`json`.

```python
class JSONStreamError(ValueError):
    offset: int   # 0-based character offset in the whole stream
    line: int     # 1-based
    column: int   # 1-based

class Tokenizer:
    def __init__(self) -> None: ...
    def feed(self, chunk: str) -> list[tuple]: ...
    def close(self) -> list[tuple]: ...
```

`JSONStreamError` must be a subclass of `ValueError` and have the three integer attributes `offset`, `line`,
`column` (its constructor signature and message are up to you).

## Usage

Call `feed` any number of times with `str` chunks (possibly empty), then `close()` once. The concatenation of all
chunks is *the stream*; it must contain exactly one JSON value, optionally surrounded by whitespace. Each `feed` call
returns the list of tokens that became complete during that call, in stream order; `close()` returns the tokens that
complete at the end of the stream (at most one number). Feeding the same stream in any chunking must produce the
same tokens and the same errors.

## Tokens

Each token is a tuple `(kind, value, offset)` where `offset` is the 0-based offset of the token's first character in
the stream (for strings and keys, the opening quote):

| kind             | value                                  | for                                   |
|------------------|----------------------------------------|---------------------------------------|
| `"begin_object"` | `None`                                 | `{`                                   |
| `"end_object"`   | `None`                                 | `}`                                   |
| `"begin_array"`  | `None`                                 | `[`                                   |
| `"end_array"`    | `None`                                 | `]`                                   |
| `"key"`          | the decoded `str`                      | a string used as an object member name |
| `"string"`       | the decoded `str`                      | any other string                      |
| `"number"`       | `int` or `float` (see below)           | a number                              |
| `"bool"`         | `True` or `False`                      | `true`, `false`                       |
| `"null"`         | `None`                                 | `null`                                |

Colons, commas and whitespace produce no tokens. Duplicate keys are allowed (just emit them).

When tokens are emitted: `{ } [ ]` in the call that feeds that character; strings and keys in the call that feeds the
closing quote; `true`, `false` and `null` in the call that feeds their last letter; a number in the call that feeds
the first character after the number (a number can't be known to be complete before that), or in `close()` if the
stream ends right after it. So `feed("[12")` returns `[("begin_array", None, 0)]`, then `feed("3,")` returns
`[("number", 123, 1)]`.

## Grammar

- Whitespace is only space, tab (`\t`), line feed (`\n`) and carriage return (`\r`), allowed between any two tokens
  and before/after the value. Any other character outside a string (for example form feed, U+00A0 or U+FEFF) is an
  error.
- Values: object `{` [ key `:` value { `,` key `:` value } ] `}`; array `[` [ value { `,` value } ] `]`; string;
  number; `true`; `false`; `null` (lowercase only). Keys are strings. No trailing commas, no comments, no `NaN` or
  `Infinity`, no single quotes. Nesting depth is unlimited (at least 100000 levels must work).
- Numbers: `-`? ( `0` | `1`-`9` digits\* ) ( `.` digits+ )? ( (`e`|`E`) (`+`|`-`)? digits+ )?, digits being ASCII
  `0`-`9`. So `01`, `-01`, `+1`, `.5`, `1.`, `1.e5`, `1e`, `1e+` and `--1` are invalid. The value is `int(text)` if
  the text contains none of `.`, `e`, `E` (so `-0` gives the int `0`) and `float(text)` otherwise (so `1e2` gives
  `100.0` and `-0.0` gives negative zero). Integer literals have at most 4000 digits.
- Strings: delimited by `"`. Characters U+0000 to U+001F may not appear unescaped (U+007F and everything else may,
  including characters outside the BMP). Escapes: `\"` `\\` `\/` `\b` `\f` `\n` `\r` `\t` and `\uXXXX` with exactly
  four hex digits (either case). Any other character after a backslash is an error.
- Surrogates: a `\uXXXX` escape for a high surrogate (D800-DBFF) **immediately** followed by a `\uXXXX` escape for a
  low surrogate (DC00-DFFF) decodes to the single combined code point (`"\ud83d\ude00"` is one character, U+1F600).
  Every other surrogate escape is not an error: it decodes to that lone surrogate code point (so `"\ud83d"` is the
  one-character string `"\ud83d"`, `"\ude00\ud83d"` is two lone surrogates, `"\ud83d\ud83d\ude00"` is a lone
  `\ud83d` followed by U+1F600, and `"\ud83dx"` is `\ud83d` followed by `x`). Characters that appear unescaped in the
  stream are copied as-is and are never combined with an escape.

## Errors

Raise `JSONStreamError` at the **earliest** point where the stream can no longer be valid:

- From `feed`: at the first character `c` such that the stream up to and including `c` is not the beginning of any
  valid document. `offset` is the offset of `c`; the call that feeds `c` raises (never earlier, never later), and the
  tokens that call would have returned are discarded. Examples: `[1,]` fails at the `]` (offset 3); `[1 2]` at the
  `2`; `01` at the `1`; `{"a" 1}` at the `1`; `"\x"` at the `x`; `"\u12G4"` at the `G`; `tru"` at the `"`;
  `truex` at the `x`; `[1] [2]` at the second `[`; an unescaped line feed inside a string at that line feed.
- From `close()`: if the stream is a valid beginning of a document but not a complete one (empty stream, only
  whitespace, an unclosed container or string, `-`, `1.`, `tru`, ...), the error position is the end of the stream:
  `offset` equals the total number of characters fed.
- `line` is 1 plus the number of `\n` characters before `offset`; `column` is 1 plus the number of characters between
  the last `\n` before `offset` and `offset` (or between the start of the stream and `offset` if there is none).
  Only `\n` ends a line (`\r` is an ordinary character for counting). Offsets and columns count Python `str`
  characters (code points).
- Once an error has been raised, every later `feed` or `close` call raises `JSONStreamError` again with the same
  `offset`, `line` and `column`.
- After a successful `close()`, any further `feed` or `close` call raises `RuntimeError`.

## Example

```python
t = Tokenizer()
t.feed('{"a": [1, tr')     # [("begin_object", None, 0), ("key", "a", 1), ("begin_array", None, 6), ("number", 1, 7)]
t.feed('ue], "b": "x\\')   # chunk ends with one backslash:
                           # [("bool", True, 10), ("end_array", None, 14), ("key", "b", 17)]
t.feed('n"}')              # [("string", "x\n", 22), ("end_object", None, 27)]
t.close()                  # []

t = Tokenizer()
t.feed("[1,\n  2,,")        # raises JSONStreamError with offset 8, line 2, column 5
```

Reply with exactly one ```python code block containing the complete module. Use only the Python 3.12 standard
library; the module must not print or do any work at import time.
