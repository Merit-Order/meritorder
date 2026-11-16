# Task: expand calendar recurrence rules

Write a Python module that expands a recurrence rule — a subset of the iCalendar `RRULE` of RFC 5545 — into the
list of dates and times it describes. The semantics are fully defined below and follow RFC 5545 closely, with a few
deliberate differences that are called out; where they differ from what you know of RFC 5545 or of
`dateutil.rrule`, these rules win.

## Interface

```python
from datetime import datetime

def expand(rule: str, dtstart: datetime, *, limit: int | None = None,
           between: tuple[datetime, datetime] | None = None) -> list[datetime]:
    ...
```

`dtstart`, the bounds in `between`, and every returned value are naive `datetime`s (no time zones anywhere). Every
error described below raises `ValueError`.

## 1. Rule syntax

- The rule may start with the prefix `RRULE:`. After it come one or more **parts** separated by `;`. Each part is
  `NAME=VALUE`: the name is everything before the part's first `=`, the value everything after it. A part without `=`
  or with an empty value is an error. Names and values are case-sensitive and use the uppercase spellings shown here;
  there is no whitespace anywhere. Parts may come in any order; a name given twice is an error.
- Supported names: `FREQ` (required), `INTERVAL`, `COUNT`, `UNTIL`, `BYMONTH`, `BYMONTHDAY`, `BYDAY`, `BYSETPOS`,
  `WKST`. Any other name — including RFC 5545 parts this task doesn't support, like `BYHOUR`, `BYWEEKNO` or
  `BYYEARDAY` — is an error.
- An *integer* is one or more ASCII digits. Only in `BYMONTHDAY`, `BYSETPOS` and `BYDAY` ordinals may it be
  preceded by one sign character, `+` or `-`. A *list* is one or more items separated by `,`, with no empty items;
  repeating a value in a list is allowed and means the same as giving it once.

| Part | Value | Default |
|---|---|---|
| `FREQ` | `DAILY`, `WEEKLY`, `MONTHLY` or `YEARLY` (anything else, e.g. `HOURLY`, is an error) | required |
| `INTERVAL` | integer ≥ 1 | 1 |
| `COUNT` | integer ≥ 1 | none |
| `UNTIL` | `YYYYMMDD` (a date) or `YYYYMMDDTHHMMSS` (a date and time); it must be a real calendar date and time, and no other form (e.g. a trailing `Z`, or dashes) is allowed | none |
| `BYMONTH` | list of integers 1..12 | see §3 |
| `BYMONTHDAY` | list of integers 1..31 or -31..-1 (-1 is the last day of the month, -2 the one before, …) | see §3 |
| `BYDAY` | list of weekday items: a code `MO` `TU` `WE` `TH` `FR` `SA` `SU`, optionally preceded by an ordinal, an integer 1..53 or -53..-1 (e.g. `MO`, `1MO`, `+2TU`, `-1FR`, `20MO`) | see §3 |
| `BYSETPOS` | list of integers 1..366 or -366..-1 | none |
| `WKST` | a weekday code | `MO` |

Further errors:

- `COUNT` and `UNTIL` in the same rule.
- `BYMONTHDAY` with `FREQ=WEEKLY`.
- A `BYDAY` item with an ordinal when `FREQ` is `DAILY` or `WEEKLY`.
- `BYSETPOS` in a rule that has none of `BYMONTH`, `BYMONTHDAY`, `BYDAY`.
- `limit` is not `None` and is less than 1.
- The rule has neither `COUNT` nor `UNTIL`, and `limit` and `between` are both `None` (the result would be infinite).

## 2. Periods

A recurrence is a sequence of **periods**. Period 0 contains `dtstart`'s date, and period *k* is *k* × `INTERVAL`
units after it:

- `DAILY`: a single day; period *k* is the day `dtstart.date()` + *k* × `INTERVAL` days.
- `WEEKLY`: seven consecutive days starting on the `WKST` weekday. Period 0 is the week (in that sense) containing
  `dtstart`'s date; period *k* starts 7 × *k* × `INTERVAL` days after period 0 starts. `WKST` has no effect on the
  other frequencies.
- `MONTHLY`: a calendar month; period 0 is `dtstart`'s month, period *k* is *k* × `INTERVAL` months later.
- `YEARLY`: a calendar year; period 0 is `dtstart`'s year, period *k* is *k* × `INTERVAL` years later.

## 3. The days a period produces

First, these **defaults** fill in parts the rule leaves out (they depend on `dtstart`):

- `WEEKLY` without `BYDAY`: `BYDAY` is `dtstart`'s weekday.
- `MONTHLY` with neither `BYDAY` nor `BYMONTHDAY`: `BYMONTHDAY` is `dtstart`'s day of the month.
- `YEARLY` with neither `BYDAY` nor `BYMONTHDAY`: `BYMONTHDAY` is `dtstart`'s day of the month and, if the rule has no
  `BYMONTH`, `BYMONTH` is `dtstart`'s month.
- No other defaults. (So `DAILY` uses only what the rule gives; `YEARLY;BYMONTHDAY=15` is the 15th of every month;
  `YEARLY;BYDAY=MO` is every Monday of the year.)

A period's **candidate days** are all its days that pass every one of these filters that is present (given or
defaulted):

- `BYMONTH`: the day's month is listed.
- `BYMONTHDAY`: the day is listed by its day of the month *d*, or by *d* - *D* - 1 where *D* is the number of days in
  its month (so -1 is the last day). A listed day that doesn't exist in a month — 31 in April, 30 or -30 in
  February — selects nothing in that month. Days are never moved or clamped to fit.
- `BYDAY`: the day matches at least one item (items are alternatives). A plain item such as `FR` matches every day
  with that weekday. An item with ordinal *n*, such as `2TU` or `-1FR`, matches a day with that weekday that is the
  *n*-th such weekday counting from the start of its **scope** (when *n* > 0) or the |*n*|-th counting back from the
  end of its scope (when *n* < 0); if the scope has no such weekday (say a 5th Monday), the item matches nothing in
  it. The scope is the day's month for `FREQ=MONTHLY`; for `FREQ=YEARLY` it is the day's month if the rule has
  `BYMONTH`, and the whole year otherwise (so `YEARLY;BYDAY=20MO` is the 20th Monday of each year).

**`BYSETPOS`**, if present, then keeps only some candidate days of each period: number the period's candidate days in
ascending order 1, 2, …, *n* and also -*n*, …, -1 (-1 being the latest); keep the days whose number is listed.
Positions beyond the set select nothing, and a day selected twice is kept once. `BYSETPOS` always sees the period's
whole candidate set, *before* the `dtstart` cut in §4 — even in period 0, whose earliest days may be before `dtstart`.
This holds for every frequency, `WEEKLY` included: period 0 is the full week containing `dtstart`.

## 4. Occurrences

The days kept in each period, taken period by period in chronological order, at `dtstart`'s time of day (hour,
minute, second and microsecond), are the **occurrences** — except that days earlier than `dtstart`'s date are
discarded (only period 0 can have any). `dtstart` itself is an occurrence only if the rule keeps its date: unlike RFC
5545, `dtstart` is *not* automatically the first occurrence.

Bounds, applied to that sequence:

- `UNTIL`: occurrences later than `UNTIL` don't exist (the recurrence ends there). With the `YYYYMMDD` form every
  occurrence on or before that date is allowed, at any time of day; with the `YYYYMMDDTHHMMSS` form, occurrences at or
  before that instant. An `UNTIL` before `dtstart` means there are no occurrences.
- `COUNT`: only the first `COUNT` occurrences exist, counted from `dtstart`.
- `between=(start, end)`: return only the occurrences with `start <= occurrence <= end`. It does not change which
  occurrences exist: `COUNT` still counts from `dtstart`, including occurrences before `start`. If `start > end` the
  result is empty.
- `limit`: return at most `limit` occurrences, the earliest of those that would otherwise be returned.

`expand` returns the occurrences that pass all the bounds as a list in ascending order, without duplicates. Dates
never go past 9999-12-31; a recurrence that reaches it simply ends there.

## Examples

```python
from datetime import datetime as dt

expand("FREQ=DAILY;COUNT=3", dt(2026, 3, 30, 9, 15))
# [dt(2026, 3, 30, 9, 15), dt(2026, 3, 31, 9, 15), dt(2026, 4, 1, 9, 15)]

expand("FREQ=MONTHLY;COUNT=4", dt(2026, 1, 31, 8))       # months without a 31st produce nothing
# [dt(2026, 1, 31, 8), dt(2026, 3, 31, 8), dt(2026, 5, 31, 8), dt(2026, 7, 31, 8)]

expand("FREQ=WEEKLY;INTERVAL=2;COUNT=4;BYDAY=TU,SU;WKST=MO", dt(1997, 8, 5, 9))
# [dt(1997, 8, 5, 9), dt(1997, 8, 10, 9), dt(1997, 8, 19, 9), dt(1997, 8, 24, 9)]
expand("FREQ=WEEKLY;INTERVAL=2;COUNT=4;BYDAY=TU,SU;WKST=SU", dt(1997, 8, 5, 9))
# [dt(1997, 8, 5, 9), dt(1997, 8, 17, 9), dt(1997, 8, 19, 9), dt(1997, 8, 31, 9)]

expand("FREQ=MONTHLY;BYDAY=-1FR;COUNT=2", dt(2026, 1, 1))   # last Friday of the month
# [dt(2026, 1, 30), dt(2026, 2, 27)]

expand("FREQ=MONTHLY;BYDAY=MO,TU,WE,TH,FR;BYSETPOS=1;COUNT=3", dt(2026, 1, 10, 9))
# [dt(2026, 2, 2, 9), dt(2026, 3, 2, 9), dt(2026, 4, 1, 9)]   January's first weekday (Jan 1) is before dtstart

expand("FREQ=YEARLY;BYDAY=FR;BYMONTHDAY=13", dt(2026, 1, 1), limit=3)   # every Friday the 13th
# [dt(2026, 2, 13), dt(2026, 3, 13), dt(2026, 11, 13)]

expand("FREQ=DAILY;COUNT=10", dt(2026, 1, 1), between=(dt(2026, 1, 8), dt(2026, 2, 1)))
# [dt(2026, 1, 8), dt(2026, 1, 9), dt(2026, 1, 10)]

expand("FREQ=DAILY", dt(2026, 1, 1))                  # raises ValueError (unbounded)
expand("FREQ=DAILY", dt(2026, 1, 1), limit=2)         # [dt(2026, 1, 1), dt(2026, 1, 2)]
```

Reply with exactly one ```python code block containing the complete module. Use only the Python 3.12 standard
library; the module must not print or do any work at import time.
