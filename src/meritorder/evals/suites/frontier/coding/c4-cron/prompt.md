# Cron next-run calculator

Write a Python module that computes when a cron expression next fires. It must provide exactly these two functions:

```python
from datetime import datetime

def next_run(expr: str, after: datetime) -> datetime: ...
def next_runs(expr: str, after: datetime, n: int) -> list[datetime]: ...
```

## Times

- `after` is a naive `datetime` (no time zone, no daylight saving: plain calendar arithmetic). If `after.tzinfo` is
  not `None`, raise `ValueError`. You may assume `1 <= after.year <= 9000`.
- `next_run` returns the **earliest** naive `datetime` `t` such that `t > after` (strictly later), `t.second == 0`,
  `t.microsecond == 0`, and `t` matches the expression. `after` itself may have non-zero seconds or microseconds:
  for `after = 10:00:30` the earliest candidate is `10:01:00`; for `after = 10:00:00` it is also `10:01:00`.
- `next_runs` returns the next `n` matching times in increasing order: the first is `next_run(expr, after)` and
  each following one is `next_run` of the previous result. `n == 0` returns `[]`; a negative `n` raises `ValueError`.
- Search horizon: only times `t < datetime(after.year + 51, 1, 1)` are considered. If no matching time exists in
  that range (for example the expression names February 30), raise `ValueError`.
- Performance: every call must finish within a fraction of a second, including when the answer is years away or
  the expression can never match. (Stepping minute by minute through years is too slow; stepping day by day is fine.)

## Expression syntax

After removing leading and trailing whitespace, an expression is either a macro or five fields separated by one or
more whitespace characters (exactly what `str.split()` with no arguments produces). Fewer or more than five fields is
a `ValueError`.

| # | field        | allowed numbers | names |
|---|--------------|-----------------|-------|
| 1 | minute       | 0-59            | none  |
| 2 | hour         | 0-23            | none  |
| 3 | day of month | 1-31            | none  |
| 4 | month        | 1-12            | `JAN` `FEB` `MAR` `APR` `MAY` `JUN` `JUL` `AUG` `SEP` `OCT` `NOV` `DEC` (1-12) |
| 5 | day of week  | 0-7             | `SUN` `MON` `TUE` `WED` `THU` `FRI` `SAT` (0-6) |

In the day-of-week field both 0 and 7 mean Sunday, 1 is Monday, ..., 6 is Saturday.

A field is a comma-separated list of one or more **items**. Every item must be non-empty. An item is one of:

- `*` - every value from the field's minimum to its maximum (for day of week that is 0-7).
- `V` - the single value `V`.
- `A-B` - every value from `A` through `B` inclusive. `A > B` is a `ValueError` (ranges never wrap around).
- `*/S`, `A-B/S` - the values of `*` or `A-B` starting at the first one and stepping by `S` (`10-50/20` is 10, 30, 50;
  `*/15` in the minute field is 0, 15, 30, 45).
- `A/S` - the same as `A-MAX/S` where `MAX` is the field's maximum (in the minute field `5/15` is 5, 20, 35, 50; in
  the day-of-week field `1/3` is 1, 4, 7, i.e. Monday, Thursday and Sunday).

Here `V`, `A` and `B` are each either a number (one or more ASCII digits `0-9`; leading zeros are allowed, so `05` is
5) or, only in the month and day-of-week fields, one of that field's three-letter names, matched case-insensitively
(`jan`, `Mon`, `fRi` are fine; `MONDAY` is not). Names and numbers can be mixed (`FRI-7`). `S` is a number and must
be at least 1; a step larger than the range is fine (`*/100` in the minute field is just 0). Every value must lie in
the field's allowed range. Anything else (a `-` sign, empty items such as in `1,,2` or `5,`, `?`, `L`, `W`, `#`,
unknown names, names in the wrong field, `**`, `*/`, `/5`, `5-`, a step of 0, ...) is a `ValueError`.

Macros (the whole expression, matched case-insensitively): `@yearly` and `@annually` = `0 0 1 1 *`, `@monthly` =
`0 0 1 * *`, `@weekly` = `0 0 * * 0`, `@daily` and `@midnight` = `0 0 * * *`, `@hourly` = `0 * * * *`. Any other
expression starting with `@` (for example `@reboot`) is a `ValueError`.

## Matching

A time matches when its minute is in the minute set, its hour is in the hour set, its month is in the month set and
its **day** matches. Whether the day matches follows the classic Vixie cron rule:

- Call the day-of-month field *starred* if its text begins with the character `*` (so `*` and `*/2` are starred,
  `1-31` is not). The same for the day-of-week field. Macros are expanded first.
- If **either** day field is starred, the day matches only when **both** sets match: its day of month is in the
  day-of-month set **and** its weekday is in the day-of-week set.
- If **neither** day field is starred, the day matches when **either** set matches: its day of month is in the
  day-of-month set **or** its weekday is in the day-of-week set.

For example `0 0 13 * 5` fires at midnight on every 13th of the month and also on every Friday, while
`0 0 */2 * 1` fires only on Mondays that fall on an odd day of the month (1, 3, ..., 31). `0 0 30 2 *` never fires
(raise `ValueError`), but `0 0 30 2 1` fires on every Monday in February.

## Examples

```python
next_run("*/15 * * * *", datetime(2024, 1, 1, 10, 7))       # datetime(2024, 1, 1, 10, 15)
next_run("0 9 * * MON-FRI", datetime(2024, 3, 8, 9, 0))      # datetime(2024, 3, 11, 9, 0)  (Fri -> Mon)
next_run("0 0 29 2 *", datetime(2024, 3, 1))                 # datetime(2028, 2, 29, 0, 0)
next_runs("@hourly", datetime(2024, 1, 1, 23, 30), 2)        # [datetime(2024, 1, 2, 0, 0), datetime(2024, 1, 2, 1, 0)]
```

Reply with exactly one ```python code block containing the complete module. Use only the Python 3.12 standard
library; the module must not print or do any work at import time.
