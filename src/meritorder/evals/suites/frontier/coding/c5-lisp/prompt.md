# Task: an interpreter for a small Scheme-like Lisp

Write a Python module that interprets the small Lisp dialect specified below. Everything the tests check is specified
here; follow it exactly, even where it differs from a real Scheme.

## API

```python
class LispError(Exception): ...

def run(source: str) -> str: ...
```

- `run` reads every top-level form in `source`, evaluates them in order in a **fresh global environment** (nothing
  carries over between calls), and returns the printed representation (see *Printing*) of the value of the **last**
  form. A source with no forms (only whitespace/comments, or empty) returns `"#<void>"`.
- **Every** error — syntax or runtime — must surface from `run` as a `LispError` (your own class, a subclass of
  `Exception`). No other exception type (`ZeroDivisionError`, `TypeError`, `IndexError`, `KeyError`, `RecursionError`,
  ...) may escape.

## Reader

- Whitespace is space, tab, newline and carriage return. `;` starts a comment that runs to the end of the line.
- `(` and `)` delimit lists; `'datum` is shorthand for `(quote datum)`. A `'` with no datum after it is an error.
- **Dotted lists**: inside a list, a `.` token must be preceded by at least one datum and followed by exactly one datum
  and then the closing `)`; any other use of `.` is a syntax error (`(1 . 2 3)`, `( . 2)`, `(1 .)`, a lone `.`).
  `(1 . (2 3))` reads as the list `(1 2 3)`.
- **Strings**: `"..."`, possibly containing raw newlines. The escapes `\"`, `\\`, `\n`, `\t` mean double quote,
  backslash, newline and tab; any other backslash escape, or a missing closing quote, is a syntax error.
- **Atoms**: any other maximal run of characters containing no whitespace and none of `(` `)` `'` `"` `;`. An atom made
  of an optional `+` or `-` followed by one or more ASCII digits is an integer (`42`, `-7`, `+5`). `#t` and `#f` are the
  booleans; any other atom starting with `#` is a syntax error. The atom `.` is only valid as described above.
  Everything else is a symbol (case-sensitive). There are no other number literals: `1.5`, `3/4`, `1+`, `-`, `a.b`
  and `->` are all symbols.
- A missing `)` or an extra `)` is a syntax error.

## Values

- **Numbers**: unbounded integers and exact rationals (Python's `fractions.Fraction` is fine). Rationals only arise from
  `/`. A result whose denominator is 1 is always the integer (`(/ 6 3)` is the integer `2`, and `(integer? ...)` of it
  is `#t`). The booleans are **not** numbers.
- **Booleans** `#t` and `#f`; **strings**; **symbols**; the **empty list** `()` (there is exactly one empty list
  value); **pairs** made by `cons` (a list is a chain of pairs ending in `()`); **procedures** (builtins and closures);
  and the **void** value, which is what `define`, `set!`, `(begin)`, an `if` without an else branch whose test is
  false, and a `cond` with no matching clause evaluate to.

## Printing

| value | printed as |
|---|---|
| integer | decimal, with `-` if negative: `42`, `-7`, `0` |
| rational | `numerator/denominator` in lowest terms, sign on the numerator: `3/2`, `-1/2` |
| booleans | `#t`, `#f` |
| string | in double quotes, with `\` written as `\\`, `"` as `\"`, newline as `\n`, tab as `\t`; other characters as is |
| symbol | its name |
| empty list | `()` |
| list | `(1 2 3)`: elements printed recursively, separated by single spaces |
| improper list | `(1 2 . 3)`, `(1 . 2)` |
| procedure (builtin or closure) | `#<procedure>` |
| void | `#<void>` |

Quoted data is not abbreviated when printed: `''a` prints as `(quote a)`.

## Evaluation

- Numbers, booleans and strings evaluate to themselves. A symbol evaluates to the value of the nearest enclosing
  binding of that name; an unbound symbol is an error. Evaluating the empty list `()` is an error.
- A non-empty list whose first element is one of the special-form keywords (`quote if define set! lambda begin let
  let* letrec cond and or`) is that special form. Tests never rebind these keywords.
- Any other non-empty list is a procedure call: evaluate the operator expression first, then the operand expressions
  from left to right, then apply. A call form must be a proper list. Applying a value that is not a procedure is an
  error.
- **Truthiness**: only `#f` is false. `0`, `()`, `""` are all true.
- Builtins are ordinary bindings in the global environment: a program may redefine them (`(define + -)`), which only
  affects that `run`.
- Environments are lexically scoped, mutable, and shared: a closure captures the environment in which its `lambda` was
  evaluated, and a `set!` on a captured variable is seen by every closure sharing that binding.

## Special forms

A **body** is one or more expressions, evaluated in order; its value is the value of the last one. A `define` inside a
body binds in that body's own new environment (the procedure call's, or the `let`'s), so it does not leak outside.

- `(quote datum)` → `datum`, unevaluated. Exactly one operand.
- `(if test then)` / `(if test then else)`: evaluate `test`; if it is not `#f` evaluate `then`, else `else` (or produce
  void when there is no else branch). Exactly 2 or 3 operands.
- `(define name expr)`: evaluate `expr` first, then bind `name` to it in the **current** environment (adding the binding,
  or replacing it if that environment already has one). `(define x x)` at top level is therefore an unbound-symbol
  error. `(define (name . params) body...)` is `(define name (lambda params body...))`, with `params` in any of the
  forms below (e.g. `(define (f a b) ...)`, `(define (f a . rest) ...)`, `(define (f . args) ...)`). Returns void.
- `(set! name expr)`: evaluate `expr` and store it in the nearest existing binding of `name`; if there is none, error.
  Returns void.
- `(lambda params body...)` → a closure. `params` is a list of symbols `(a b c)` (exactly that many arguments), a dotted
  list `(a b . rest)` (at least 2 arguments; `rest` is bound to a fresh list of the remaining ones, possibly `()`), or a
  single symbol `args` (any number of arguments, all bound as a list). Parameter names must be distinct symbols.
  Calling a closure with the wrong number of arguments is an error. The call evaluates the body in a new environment,
  holding the parameters, whose parent is the closure's captured environment.
- `(begin expr...)`: evaluate in order in the **current** environment (no new scope, so a top-level
  `(begin (define y 7))` defines a global); value of the last, or void for `(begin)`.
- `(let ((name expr) ...) body...)`: evaluate every `expr` in the current environment, then evaluate the body in a new
  environment binding the names (distinct) to those values. `(let () body...)` is allowed.
- **Named let** `(let loop ((name expr) ...) body...)`: evaluate the `expr`s in the current environment, then call a
  procedure with parameters `name ...` and the given body, where inside the body (only) `loop` is bound to that
  procedure itself. Equivalent to `((letrec ((loop (lambda (name ...) body...))) loop) expr ...)`.
- `(let* ((name expr) ...) body...)`: like nested `let`s, one per binding: each `expr` sees the bindings before it, and
  each binding is in its own new scope (so in `(let* ((x 1) (f (lambda () x)) (x 2)) (f))` the closure sees the first
  `x` and returns 1). The body runs in a new scope inside all of them.
- `(letrec ((name expr) ...) body...)`: create a new environment in which all the names are bound but **unassigned**;
  then evaluate the `expr`s from left to right in that environment, assigning each name as soon as its `expr` has been
  evaluated; then evaluate the body there. Reading a name that is still unassigned is an error
  (`(letrec ((a b) (b 1)) a)` fails, `(letrec ((a 1) (b (+ a 1))) b)` is 2).
- `(cond clause...)`: each clause is `(test expr...)` or `(else expr...)` (`else` only as the last clause). Evaluate the
  tests in order; for the first test that is not `#f`, evaluate that clause's expressions as a body and return the last
  value — or, if the clause has no expressions, return the test's value. An `else` clause always matches. No match →
  void.
- `(and expr...)`: evaluate left to right; return the first value that is `#f` (without evaluating the rest), otherwise
  the last value; `(and)` is `#t`. `(or expr...)`: return the first value that is not `#f` (without evaluating the
  rest), otherwise `#f`; `(or)` is `#f`.

A malformed special form is an error: the wrong number of operands where a number is fixed above, a missing body
(`(lambda (x))`, `(let ((x 1)))`), a non-symbol where a name is required (`(define 5 1)`, `(set! 1 2)`,
`(lambda (1) 1)`), duplicate parameter names, or a `let`/`let*`/`letrec` binding that is not a two-element list
`(name expr)` (`(let ((x)) x)`, `(let ((x 1 2)) x)`). Tests only contain malformed forms that are actually evaluated, so
you may detect them either when the form is evaluated or earlier.

## Proper tail calls

These are **tail positions**: both branches of `if`; the last expression of a `lambda` body, of a `begin`, and of the
bodies of `let`, named `let`, `let*` and `letrec`; the last expression of the chosen `cond` clause; the last operand of
`and` and of `or`. A procedure call in a tail position must not grow the Python call stack: a loop written as tail
calls (including mutual recursion between procedures, tail calls to variadic procedures, and tail calls to procedures
passed as arguments) must run in constant Python stack depth however many iterations it does. The tests check this
directly: they run the same loop with 10 and with 1,000 iterations while counting the Python call depth with
`sys.setprofile`/`threading.setprofile`, and require the maximum depth to be the same, give or take 50 frames
(raising the recursion limit is therefore no substitute). They also run loops of 100,000 iterations.

Calls that are not in tail position may use Python recursion. Tests contain non-tail recursion at most 1000 calls deep
(e.g. a naive recursive sum of 1..1000, or building a 1000-element list recursively); if your evaluator needs several
Python frames per Lisp call, raise the limit with `sys.setrecursionlimit` (for example to 20000).

## Builtins

`n` = number (integer or rational), `i` = integer, `x` = any value. A wrong number of arguments, or an argument of the
wrong type, is an error.

| name | arguments | result |
|---|---|---|
| `+` `*` | 0 or more `n` | sum / product (`(+)` is 0, `(*)` is 1) |
| `-` | 1 or more `n` | `(- a)` is the negation; otherwise `a` minus each of the rest in turn |
| `/` | 1 or more `n` | `(/ a)` is 1/a; otherwise `a` divided by each of the rest in turn, exactly. Dividing by 0 is an error |
| `quotient` | 2 `i` | integer division truncated toward zero: `(quotient -7 2)` is -3 |
| `remainder` | 2 `i` | `a - b*(quotient a b)`, sign of the dividend: `(remainder -7 2)` is -1, `(remainder 7 -2)` is 1 |
| `modulo` | 2 `i` | remainder with the sign of the divisor: `(modulo -7 2)` is 1, `(modulo 7 -2)` is -1 |
| `abs` | 1 `n` | absolute value |
| `=` `<` `>` `<=` `>=` | 2 or more `n` | `#t` if the comparison holds between each adjacent pair (`(< 1 2 3)`), else `#f` |
| `number?` `integer?` `boolean?` `symbol?` `string?` `procedure?` `null?` `pair?` `list?` | 1 `x` | type tests; `integer?` is `#t` only for integers; `null?` only for `()`; `pair?` for any pair; `list?` for proper lists including `()` |
| `not` | 1 `x` | `#t` if `x` is `#f`, else `#f` |
| `eq?` | 2 `x` | numbers: `#t` if numerically equal (whatever their size); booleans, symbols: same value; the empty list equals itself; strings, pairs and procedures: `#t` only if they are the very same object (e.g. the same variable's value); values of different types are never `eq?` |
| `equal?` | 2 `x` | like `eq?`, but strings are compared by content and pairs structurally (car and cdr recursively) |
| `cons` | 2 `x` | a new pair |
| `car` `cdr` | 1 pair | first / rest of a pair (`(car '())` is an error) |
| `list` | 0 or more `x` | a new list of the arguments |
| `length` | 1 proper list | number of elements |
| `append` | 0 or more proper lists | a new list of all their elements, in order (`(append)` is `()`) |
| `reverse` | 1 proper list | a new reversed list |
| `apply` | a procedure and a proper list | the procedure called with the list's elements as arguments |
| `map` | a procedure and a proper list | a new list of the procedure applied to each element, in order |
| `string-append` | 0 or more strings | concatenation |
| `string-length` | 1 string | number of characters |
| `number->string` | 1 `n` | the number's printed form as a string (`"-1/2"`) |
| `symbol->string` | 1 symbol | its name as a string |
| `string->symbol` | 1 string | the symbol with that name |

## Errors (all raise `LispError`)

Syntax errors as described under *Reader* and malformed special forms; unbound symbols; reading an unassigned `letrec`
variable; `set!` of an unbound name; evaluating `()`; calling a non-procedure; wrong argument counts for closures and
builtins; argument type errors in builtins (including booleans passed where numbers are required: `(+ 1 #t)` is an
error); division by zero in `/`, `quotient`, `remainder`, `modulo`.

## Examples

```python
run("(define (sq x) (* x x)) (sq 12)")                     # "144"
run("(/ 6 4)")                                              # "3/2"
run("(cons 1 (cons 2 3))")                                  # "(1 2 . 3)"
run("'(a \"b\" #t)")                                        # '(a "b" #t)'
run("(define x 1)")                                         # "#<void>"
run("((lambda (a . r) (list a r)) 1 2 3)")                  # "(1 (2 3))"
run("(let loop ((i 0)) (if (< i 100000) (loop (+ i 1)) i))")  # "100000"
run("(car '())")                                            # raises LispError
```

Reply with exactly one ```python code block containing the complete module. Use only the Python 3.12 standard library;
the module must not print anything or do any work at import time beyond defining things.
