Write a Python module that parses, orders and matches Semantic Versioning 2.0.0 version strings. It must define these four functions.

```python
def parse(version: str) -> tuple[int, int, int, tuple[int | str, ...], tuple[str, ...]]: ...
def compare(a: str, b: str) -> int: ...
def sort_versions(versions: list[str]) -> list[str]: ...
def satisfies(version: str, constraint: str) -> bool: ...
```

## Version syntax

A valid version is `MAJOR.MINOR.PATCH`, optionally followed by `-PRERELEASE`, optionally followed by `+BUILD`, and nothing else:

- `MAJOR`, `MINOR` and `PATCH` are non-empty runs of ASCII digits `0`-`9` with no leading zeros (`0` is fine, `01` is not). There is no size limit (`99999999999999999999.0.0` is valid).
- `PRERELEASE` is one or more identifiers separated by dots. Each identifier is non-empty and made only of ASCII letters, ASCII digits and hyphens (`[0-9A-Za-z-]`). An identifier made only of digits is *numeric* and must not have a leading zero (`0` is fine, `01` is not). Any other identifier is *alphanumeric* and may start with zeros (`0A`, `-`, `--x` and `001-a` are all fine).
- `BUILD` is one or more identifiers separated by dots, each non-empty and made only of `[0-9A-Za-z-]`. Leading zeros are allowed in build identifiers (`+001` is fine).
- The whole string must match. No surrounding or embedded whitespace, no trailing newline, no `v` prefix, no underscores, and only ASCII characters count as digits or letters (characters such as `١`, `²` or `é` are invalid everywhere).

Any string that breaks these rules is invalid, and every function raises `ValueError` when given an invalid version.

## `parse(version)`

Returns `(major, minor, patch, prerelease, build)`: three `int`s, then a tuple of prerelease identifiers in order (numeric identifiers converted to `int`, alphanumeric ones kept as `str`), then a tuple of build identifiers in order (always `str`, unconverted). Missing parts are empty tuples.

```python
parse("1.0.0-alpha.1+build.001") == (1, 0, 0, ("alpha", 1), ("build", "001"))
parse("2.10.3") == (2, 10, 3, (), ())
```

## Precedence: `compare(a, b)`

Returns `-1` if `a` has lower precedence than `b`, `0` if equal, `1` if higher (exactly these three ints). Precedence follows SemVer 2.0.0:

1. Compare `MAJOR`, `MINOR`, `PATCH` numerically, in that order.
2. If those are equal, a version with a prerelease has lower precedence than the same version without one (`1.0.0-rc.1 < 1.0.0`).
3. If both have prereleases, compare identifiers left to right: two numeric identifiers compare numerically; two alphanumeric identifiers compare character by character in ASCII order (so `B` < `a`, and `alpha` < `alpha-1` because a string that is a prefix of another sorts first); a numeric identifier always has lower precedence than an alphanumeric one. The first difference decides. If all identifiers of the shorter list are equal to the start of the longer one, the longer list has higher precedence.
4. Build metadata is ignored entirely (`1.0.0+a` and `1.0.0+b` compare equal).

## `sort_versions(versions)`

Returns a new list with the same strings sorted by ascending precedence. Strings with equal precedence keep their relative order from the input (a stable sort). Raises `ValueError` if any entry is invalid.

## `satisfies(version, constraint)`

Returns whether `version` meets `constraint`. Raises `ValueError` if `version` or `constraint` is invalid.

- A constraint is one or more *alternatives* separated by `||`; it is satisfied if any alternative is.
- An alternative is one or more *comparators* separated by whitespace (as `str.split()` with no arguments splits); it is satisfied if every comparator is. An alternative with no comparators (for example an empty or all-whitespace constraint, `1.0.0 ||`, or `|| 1.0.0`) is invalid.
- A comparator is an operator immediately followed by a valid version, with no whitespace between them (`>= 1.0.0` is invalid). The operators are:
  - `>=V`, `<=V`, `>V`, `<V`, `=V`: compare `version` with `V` by precedence. A bare `V` with no operator means `=V`.
  - `~V` means `>=V` and `<MAJOR.(MINOR+1).0`, where `MAJOR` and `MINOR` come from `V`.
  - `^V` means `>=V` and `<UPPER`, where `UPPER` is `(MAJOR+1).0.0` if `MAJOR > 0`; else `0.(MINOR+1).0` if `MINOR > 0`; else `0.0.(PATCH+1)`.
  - Anything else (`=>1.0.0`, `!1.0.0`, a lone `|`, a version that is invalid such as `1.2`) makes the constraint invalid.
- Matching is purely by precedence. Prerelease versions get no special treatment, unlike npm: `satisfies("2.0.0-rc.1", "^1.2.3")` is `True` because `2.0.0-rc.1` has lower precedence than `2.0.0`.

```python
satisfies("1.4.0", ">=1.2.0 <2.0.0")        # True
satisfies("0.3.9", "^0.2.0 || ~0.3.1")       # True
satisfies("1.3.0", "~1.2.3")                 # False
```

Reply with exactly one ```python code block containing the complete module. Use only the Python 3.12 standard library; the module must not print or do any work at import time.
