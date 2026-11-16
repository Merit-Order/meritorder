# slotfinder

Finds free meeting slots in people's calendars. Times are minutes since midnight and every interval is
half-open, `[start, end)`: a meeting 09:00-10:00 and one 10:00-11:00 touch, with no gap between them.

```python
from slotfinder import Calendar

cal = Calendar()
cal.add("dana", "09:00-12:00", "all-hands")
cal.add("omar", "13:00-14:00", "planning")
cal.common_free(["dana", "omar"])   # [Interval(start=720, end=780), Interval(start=840, end=1020)]
```

Modules: `intervals` (the interval helpers everything else uses), `availability` (free slots in a working
day), `summary` (busy summaries for the weekly digest), `calendar`, `timefmt`.

Run the tests with `python -m unittest discover -s tests`.
