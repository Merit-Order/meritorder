# Exact rational expression evaluator

Write a Python module that evaluates arithmetic expressions exactly, using `fractions.Fraction`, and reports errors
with precise positions. It must define:

```python
from fractions import Fraction

class ExprError(Exception): ...      # base class of both errors below
class ParseError(ExprError): ...     # the text is not a valid expression
class EvalError(ExprError): ...      # the text is valid but can't be evaluated

def evaluate(text: str, variables: dict[str, int | Fraction] | None = None) -> Fraction: ...
```

Both `ParseError` and `EvalError` instances must have an integer attribute `position` (0-based index into `text`,
defined below). The constructor signatures and messages are up to you. `evaluate` must never raise any other exception
type (no `ZeroDivisionError`, `KeyError`, `RecursionError`, ...) for any input described here.

## Tokens

The text is split into tokens from left to right. Between tokens there may be any amount of whitespace, which is
exactly the four characters space, tab (`\t`), newline (`\n`) and carriage return (`\r`); whitespace is otherwise
ignored. The tokens are:

- **Number**: the longest run starting at the current character that is either one or more ASCII digits `0`-`9`,
  optionally followed by `.` and zero or more ASCII digits (`12`, `007`, `1.25`, `5.`), or `.` followed by one or
  more ASCII digits (`.5`). Its value is the exact decimal value as a Fraction (`0.1` is exactly 1/10). There is no
  exponent notation and no sign inside a number.
- **Name**: an ASCII letter or `_`, followed by the longest run of ASCII letters, ASCII digits and `_` (`x`, `_a1`,
  `rate2`). Names are case-sensitive.
- **Operators and punctuation**: `+`, `-`, `*`, `/`, `//`, `%`, `^`, `(`, `)`, `,`. Two adjacent slashes (nothing
  between them) are always the single token `//`; `/ /` is two `/` tokens. `**` is NOT an operator (it is two `*`
  tokens).
- Any other character (anything not listed above, including non-ASCII digits like `²` or `٣`, non-ASCII letters, and
  other whitespace such as `\xa0` or `\f`, and a `.` that does not start a number) is an **invalid character**: a
  one-character token that can never appear in a valid expression.

So `2x` is the number `2` followed by the name `x` (there is no implicit multiplication, so it is a syntax error), and
`1..2` is the number `1.` followed by the number `.2`.

## Grammar

From lowest to highest precedence:

```
expression := term (("+" | "-") term)*                  left-associative
term       := unary (("*" | "/" | "//" | "%") unary)*   left-associative
unary      := ("-" | "+") unary | power
power      := primary ("^" unary)?                     right-associative via unary
primary    := Number | Name | Name "(" arguments ")" | "(" expression ")"
arguments  := expression ("," expression)*
```

The whole text must be exactly one `expression`. Consequences worth spelling out:

- `^` binds tighter than unary minus on its left: `-2^2` is `-(2^2)` = -4, but `(-2)^2` = 4.
- The exponent may itself carry a sign: `2^-1` = 1/2, `2^-3^2` = `2^(-(3^2))` = 1/512.
- `^` is right-associative: `2^3^2` = `2^9` = 512.
- Unary operators may repeat and may follow binary operators: `--3` = 3, `2--3` = 5, `2*-3` = -6.
- A unary minus applies to the whole power expression after it but not past a `*`, `/`, `//` or `%`:
  `-7 // 2` is `(-7) // 2` = -4.
- A Name immediately followed (whitespace allowed) by `(` is always a function call; otherwise a Name is a variable.
- Calls need at least one argument: `abs()` is a syntax error, and so is a trailing comma `max(1,)`.

## Syntax errors (`ParseError`)

Syntax is checked for the whole text before anything is evaluated: if the text is not a valid expression,
`ParseError` is raised even if evaluating some part of it would also fail (`1/0 +` raises `ParseError`).

`position` is defined as follows. Let the tokens be t1 ... tn. Find the longest prefix t1 ... tk that is the start of
the token sequence of at least one valid expression. If k < n, `position` is the index in `text` of the first
character of token t(k+1). If k = n (all the tokens are fine so far but the expression is incomplete, including when
there are no tokens at all), `position` is `len(text)`, even if the text ends with whitespace.

In other words, report the first token where a left-to-right parser gets stuck, or the end of the text. An invalid
character is only reported if no earlier token is already wrong. Examples (position after the arrow):

| text | position | why |
|---|---|---|
| `""`, `"   "` | 0, 3 | no tokens: end of text |
| `"1 +"` | 3 | incomplete |
| `"(1+2   "` | 7 | missing `)`: end of text |
| `"1+2)"` | 3 | unexpected `)` |
| `"1 + * 2"` | 4 | `*` can't start an operand |
| `"2**3"` | 2 | the second `*` |
| `"2x"` | 1 | a name can't follow a number |
| `"1 ) $"` | 2 | the `)` is already wrong; `$` is never reached |
| `"2 $ 3"` | 2 | invalid character |

## Evaluation

Every value is a `Fraction`, and `evaluate` returns a `Fraction` (also when the result is a whole number).

- `+`, `-`, `*`, `/`: exact rational arithmetic. Unary `-` negates, unary `+` does nothing.
- `a // b` is the floor of a/b (the largest integer <= a/b), returned as a Fraction: `7.5 // 2` = 3,
  `-7 // 2` = -4.
- `a % b` is `a - b * (a // b)`, so the result has the sign of `b`: `7 % -2` = -1, `-7 % 2` = 1,
  `7.5 % 2` = 3/2.
- `a ^ b` is a raised to the power b, exactly. The exponent must be a whole number (`4^(4/2)` = 16 is fine,
  `4^(1/2)` is an error) whose absolute value is at most 4096 (`2^4096` is fine, `2^4097` and `1^-5000` are errors).
  `0^0` = 1; 0 raised to a negative power is an error.
- Variables are looked up in `variables` (a missing or `None` dict means no variables); a value is an `int` or a
  `Fraction` and is used as `Fraction(value)`. A name that is not in the dict is an error.
- Functions: `abs(x)` (exactly one argument), `min(x1, ..., xn)` and `max(x1, ..., xn)` (one or more arguments).
  Calling any other name, including a name that is a variable, is an error ("unknown function"), and so is calling
  `abs` with more than one argument. The function names are not reserved: `abs + 1` with `variables={"abs": 3}`
  is 4.

**Evaluation order** is strictly left to right: for a binary operator the left operand is evaluated completely, then
the right operand, then the operator is applied. For a call, the function name and the number of arguments are checked
first, before any argument is evaluated; then the arguments are evaluated from left to right. The first error raised
in this order is the one reported.

## Evaluation errors (`EvalError`)

Raised only for syntactically valid text. `position` is:

- division by zero in `/`, `//` or `%`: the index of that operator token (for `//`, its first `/`);
- a bad exponent (not a whole number, absolute value over 4096, or 0 raised to a negative power): the index of that
  `^`;
- an unknown variable, an unknown function, or a wrong number of arguments to `abs`: the index of the name's first
  character.

Examples: `evaluate("1/(2-2)")` raises `EvalError` with position 1; `evaluate("1 + foo(1/0)")` raises `EvalError`
with position 4 (the unknown function is found before its argument is evaluated); `evaluate("x + y", {"x": 1})`
raises `EvalError` with position 4.

Input nesting (parentheses, calls and repeated unary operators combined) is at most 50 levels deep.

## More examples

```python
evaluate("1 + 2 * 3")            == Fraction(7)
evaluate("0.1 + 0.2")            == Fraction(3, 10)
evaluate("2 ^ 3 ^ 2")            == Fraction(512)
evaluate("-2 ^ -2")              == Fraction(-1, 4)
evaluate("(2/3) ^ -2")           == Fraction(9, 4)
evaluate("7 % -2")               == Fraction(-1)
evaluate("max(1, min(2, 3))^2")  == Fraction(4)
evaluate("x*y + z", {"x": 2, "y": Fraction(1, 3), "z": -1}) == Fraction(-1, 3)
```

Do not use `eval`, `exec`, `compile`, `ast` or `tokenize`; write the tokenizer and parser yourself.

Reply with exactly one ```python code block containing the complete module. Use only the Python 3.12 standard
library; the module must not print anything or do any work at import time.
