"""Plausible but buggy: days before dtstart are dropped *before* BYSETPOS picks positions, and a YEARLY ordinal such
as 2SU always counts within the whole year, even when BYMONTH is given."""

import calendar
from datetime import date, datetime, timedelta

DAYS = ["MO", "TU", "WE", "TH", "FR", "SA", "SU"]
KNOWN = {"FREQ", "INTERVAL", "COUNT", "UNTIL", "BYMONTH", "BYMONTHDAY", "BYDAY", "BYSETPOS", "WKST"}


def _num(text, signed=False):
    body = text[1:] if signed and text[:1] in "+-" else text
    if not body or not body.isascii() or not body.isdigit():
        raise ValueError(f"bad number {text!r}")
    return -int(body) if text.startswith("-") else int(body)


def _list(text):
    items = text.split(",")
    if "" in items:
        raise ValueError("empty item")
    return items


def _parse_until(text):
    try:
        if len(text) == 8 and text.isdigit():
            return datetime.strptime(text, "%Y%m%d").date()
        if len(text) == 15 and text[8] == "T" and (text[:8] + text[9:]).isdigit():
            return datetime.strptime(text, "%Y%m%dT%H%M%S")
    except ValueError:
        pass
    raise ValueError(f"bad UNTIL {text!r}")


class _Rule:
    def __init__(self, text):
        if text.startswith("RRULE:"):
            text = text[6:]
        parts = {}
        for chunk in text.split(";"):
            if "=" not in chunk:
                raise ValueError(f"bad part {chunk!r}")
            key, val = chunk.split("=", 1)
            if key not in KNOWN or key in parts or not val:
                raise ValueError(f"bad part {chunk!r}")
            parts[key] = val
        self.freq = parts.get("FREQ")
        if self.freq not in ("DAILY", "WEEKLY", "MONTHLY", "YEARLY"):
            raise ValueError("bad FREQ")
        self.interval = _num(parts.get("INTERVAL", "1"))
        self.count = _num(parts["COUNT"]) if "COUNT" in parts else None
        self.until = _parse_until(parts["UNTIL"]) if "UNTIL" in parts else None
        if self.interval < 1 or (self.count is not None and self.count < 1):
            raise ValueError("INTERVAL and COUNT must be positive")
        if self.count is not None and self.until is not None:
            raise ValueError("COUNT with UNTIL")
        self.months = [_num(m) for m in _list(parts["BYMONTH"])] if "BYMONTH" in parts else None
        if self.months and not all(1 <= m <= 12 for m in self.months):
            raise ValueError("bad BYMONTH")
        self.monthdays = [_num(d, True) for d in _list(parts["BYMONTHDAY"])] if "BYMONTHDAY" in parts else None
        if self.monthdays is not None:
            if self.freq == "WEEKLY" or not all(1 <= abs(d) <= 31 for d in self.monthdays):
                raise ValueError("bad BYMONTHDAY")
        self.weekdays = None
        if "BYDAY" in parts:
            self.weekdays = []
            for item in _list(parts["BYDAY"]):
                if item[-2:] not in DAYS:
                    raise ValueError("bad BYDAY")
                n = _num(item[:-2], True) if item[:-2] else 0
                if item[:-2] and (not 1 <= abs(n) <= 53 or self.freq in ("DAILY", "WEEKLY")):
                    raise ValueError("bad BYDAY ordinal")
                self.weekdays.append((n, DAYS.index(item[-2:])))
        self.setpos = [_num(p, True) for p in _list(parts["BYSETPOS"])] if "BYSETPOS" in parts else None
        if self.setpos is not None:
            if not ({"BYMONTH", "BYMONTHDAY", "BYDAY"} & set(parts)) or not all(1 <= abs(p) <= 366 for p in self.setpos):
                raise ValueError("bad BYSETPOS")
        wkst = parts.get("WKST", "MO")
        if wkst not in DAYS:
            raise ValueError("bad WKST")
        self.wkst = DAYS.index(wkst)

    def defaults(self, start):
        if self.freq == "WEEKLY" and self.weekdays is None:
            self.weekdays = [(0, start.weekday())]
        if self.freq in ("MONTHLY", "YEARLY") and self.weekdays is None and self.monthdays is None:
            self.monthdays = [start.day]
            if self.freq == "YEARLY" and self.months is None:
                self.months = [start.month]

    def periods(self, start):
        if self.freq == "DAILY":
            step, first = timedelta(days=self.interval), start
            while True:
                yield [first]
                first += step
        elif self.freq == "WEEKLY":
            first = start - timedelta(days=(start.weekday() - self.wkst) % 7)
            while True:
                yield [first + timedelta(days=i) for i in range(7)]
                first += timedelta(weeks=self.interval)
        elif self.freq == "MONTHLY":
            y, m = start.year, start.month
            while y <= 9999:
                yield [date(y, m, d) for d in range(1, calendar.monthrange(y, m)[1] + 1)]
                m += self.interval
                y, m = y + (m - 1) // 12, (m - 1) % 12 + 1
        else:
            y = start.year
            while y <= 9999:
                first = date(y, 1, 1)
                yield [first + timedelta(days=i) for i in range(366 if calendar.isleap(y) else 365)]
                y += self.interval

    def keep(self, d):
        if self.months is not None and d.month not in self.months:
            return False
        if self.monthdays is not None:
            length = calendar.monthrange(d.year, d.month)[1]
            if d.day not in self.monthdays and d.day - length - 1 not in self.monthdays:
                return False
        if self.weekdays is not None:
            if self.freq == "MONTHLY":
                index, length = d.day, calendar.monthrange(d.year, d.month)[1]
            else:
                index, length = d.timetuple().tm_yday, 366 if calendar.isleap(d.year) else 365
            ok = False
            for n, wd in self.weekdays:
                if wd != d.weekday():
                    continue
                if n == 0 or (n > 0 and (index - 1) // 7 + 1 == n) or (n < 0 and (length - index) // 7 + 1 == -n):
                    ok = True
            if not ok:
                return False
        return True


def expand(rule, dtstart, *, limit=None, between=None):
    r = _Rule(rule)
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if r.count is None and r.until is None and limit is None and between is None:
        raise ValueError("unbounded recurrence")
    start = dtstart.date()
    r.defaults(start)
    stop = None
    if r.until is not None:
        stop = r.until if isinstance(r.until, date) and not isinstance(r.until, datetime) else r.until.date()
    if between is not None:
        stop = min(stop, between[1].date()) if stop else between[1].date()
    found, produced = [], 0
    try:
        for period in r.periods(start):
            if stop is not None and period[0] > stop:
                break
            days = [d for d in period if d >= start and r.keep(d)]
            if r.setpos is not None:
                days = sorted({days[p - 1 if p > 0 else p] for p in r.setpos if (p > 0 and p <= len(days)) or (p < 0 and -p <= len(days))})
            for d in days:
                when = datetime.combine(d, dtstart.time())
                if r.until is not None:
                    if isinstance(r.until, datetime) and when > r.until:
                        return found
                    if not isinstance(r.until, datetime) and d > r.until:
                        return found
                produced += 1
                if between is not None and when > between[1]:
                    return found
                if between is None or when >= between[0]:
                    found.append(when)
                    if limit is not None and len(found) >= limit:
                        return found
                if r.count is not None and produced >= r.count:
                    return found
    except OverflowError:
        pass
    return found
