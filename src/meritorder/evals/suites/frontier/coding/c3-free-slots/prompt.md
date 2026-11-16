Write a Python module that finds free meeting slots in a calendar. It must define:

```python
def free_slots(busy: list[tuple[str, str]], window_start: str, window_end: str, *,
               workday: tuple[str, str] = ("09:00", "17:00"), min_minutes: int = 30,
               buffer: int = 0, weekdays_only: bool = True) -> list[tuple[str, str]]: ...
```

## Formats

- A **timestamp** is a string of exactly the form `YYYY-MM-DDTHH:MM`: 16 characters, all fields zero-padded ASCII digits, a literal `T`, a real calendar date (leap years count), hours `00`-`23`, minutes `00`-`59`. Nothing else is accepted: no seconds, no space instead of `T`, no time zone, no unpadded fields like `2024-1-05T09:00` or `2024-01-05T9:00`.
- A **time of day** in `workday` is exactly `HH:MM` (5 characters, zero-padded, hours `00`-`23`, minutes `00`-`59`), except that the second element (the end of the working day) may also be `24:00`, meaning midnight at the end of that day.

## Rules

1. `busy` is a list of `(start, end)` timestamp pairs. Each is the half-open interval `[start, end)`. Intervals may be in any order, may overlap or touch, and may extend outside the window or the working hours or across several days. An interval with `start == end` is ignored completely. An interval with `start > end` is an error.
2. `buffer` (minutes) widens every non-ignored busy interval on both sides: `[start, end)` becomes `[start - buffer, end + buffer)`.
3. The search window is the half-open interval `[window_start, window_end)`. If `window_end == window_start` the result is `[]`; if `window_end < window_start` it's an error.
4. For each calendar date from the date of `window_start` through the date of `window_end` inclusive, skipping Saturdays and Sundays when `weekdays_only` is true, the date's **working period** is `[date at workday[0], date at workday[1])`, intersected with the window.
5. The free time of a working period is that period minus the union of all the (buffered) busy intervals. Every maximal stretch of free time whose length is at least `min_minutes` minutes is a slot. Each date is handled separately: slots from different dates are never merged, even if one ends exactly where the next begins.
6. Return the slots sorted by start time, each as a `(start, end)` tuple of timestamps in the same `YYYY-MM-DDTHH:MM` format. A slot that ends at midnight is written with the next day's date and `T00:00`.

## Errors

Raise `ValueError` for: any malformed timestamp or time of day (in `busy`, the window or `workday`); a busy interval with `start > end`; `window_end < window_start`; `workday[0]` not strictly earlier than `workday[1]`; `min_minutes < 1`; `buffer < 0`.

## Example

2024-01-01 is a Monday.

```python
free_slots([("2024-01-01T10:00", "2024-01-01T11:00"), ("2024-01-01T10:30", "2024-01-01T12:15"),
            ("2024-01-01T16:40", "2024-01-02T09:20")],
           "2024-01-01T00:00", "2024-01-02T12:00", min_minutes=45)
== [("2024-01-01T09:00", "2024-01-01T10:00"), ("2024-01-01T12:15", "2024-01-01T16:40"),
    ("2024-01-02T09:20", "2024-01-02T12:00")]
```

Reply with exactly one ```python code block containing the complete module. Use only the Python 3.12 standard library; the module must not print or do any work at import time.
