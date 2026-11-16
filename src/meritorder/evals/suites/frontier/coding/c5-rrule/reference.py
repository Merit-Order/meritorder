"""Reference solution for c5-rrule: expand an RFC 5545 RRULE subset (DAILY/WEEKLY/MONTHLY/YEARLY)."""

import calendar
from datetime import date, datetime, timedelta

WEEKDAYS = {"MO": 0, "TU": 1, "WE": 2, "TH": 3, "FR": 4, "SA": 5, "SU": 6}
FREQS = ("DAILY", "WEEKLY", "MONTHLY", "YEARLY")
PARTS = ("FREQ", "INTERVAL", "COUNT", "UNTIL", "BYMONTH", "BYMONTHDAY", "BYDAY", "BYSETPOS", "WKST")
DIGITS = "0123456789"


def _uint(s: str) -> int:
    if not s or any(c not in DIGITS for c in s):
        raise ValueError(f"not a number: {s!r}")
    return int(s)


def _sint(s: str) -> int:
    sign = 1
    if s[:1] in ("+", "-"):
        sign = -1 if s[0] == "-" else 1
        s = s[1:]
    return sign * _uint(s)


def _items(value: str) -> list[str]:
    items = value.split(",")
    if any(not it for it in items):
        raise ValueError("empty list item")
    return items


def _until(value: str):
    if len(value) == 8:
        return ("date", date(_uint(value[0:4]), _uint(value[4:6]), _uint(value[6:8])))
    if len(value) == 15 and value[8] == "T":
        return ("datetime", datetime(_uint(value[0:4]), _uint(value[4:6]), _uint(value[6:8]),
                                     _uint(value[9:11]), _uint(value[11:13]), _uint(value[13:15])))
    raise ValueError(f"bad UNTIL: {value!r}")


def _parse(rule: str) -> dict:
    if rule.startswith("RRULE:"):
        rule = rule[len("RRULE:"):]
    raw: dict[str, str] = {}
    for part in rule.split(";"):
        name, eq, value = part.partition("=")
        if not eq or not value:
            raise ValueError(f"bad part: {part!r}")
        if name not in PARTS:
            raise ValueError(f"unsupported part: {name!r}")
        if name in raw:
            raise ValueError(f"duplicate part: {name}")
        raw[name] = value
    if "FREQ" not in raw:
        raise ValueError("FREQ is required")
    freq = raw["FREQ"]
    if freq not in FREQS:
        raise ValueError(f"unsupported FREQ: {freq}")
    r = {"freq": freq, "interval": 1, "count": None, "until": None, "bymonth": None, "bymonthday": None,
         "byday": None, "bysetpos": None, "wkst": 0}
    if "INTERVAL" in raw:
        r["interval"] = _uint(raw["INTERVAL"])
        if r["interval"] < 1:
            raise ValueError("INTERVAL must be at least 1")
    if "COUNT" in raw:
        r["count"] = _uint(raw["COUNT"])
        if r["count"] < 1:
            raise ValueError("COUNT must be at least 1")
    if "UNTIL" in raw:
        r["until"] = _until(raw["UNTIL"])
    if r["count"] is not None and r["until"] is not None:
        raise ValueError("COUNT and UNTIL can't both be given")
    if "BYMONTH" in raw:
        r["bymonth"] = {_uint(v) for v in _items(raw["BYMONTH"])}
        if not all(1 <= m <= 12 for m in r["bymonth"]):
            raise ValueError("BYMONTH out of range")
    if "BYMONTHDAY" in raw:
        if freq == "WEEKLY":
            raise ValueError("BYMONTHDAY can't be used with FREQ=WEEKLY")
        r["bymonthday"] = {_sint(v) for v in _items(raw["BYMONTHDAY"])}
        if not all(1 <= abs(d) <= 31 for d in r["bymonthday"]):
            raise ValueError("BYMONTHDAY out of range")
    if "BYDAY" in raw:
        byday = set()
        for item in _items(raw["BYDAY"]):
            code, num = item[-2:], item[:-2]
            if code not in WEEKDAYS:
                raise ValueError(f"bad BYDAY item: {item!r}")
            if num:
                n = _sint(num)
                if not 1 <= abs(n) <= 53:
                    raise ValueError("BYDAY ordinal out of range")
                if freq in ("DAILY", "WEEKLY"):
                    raise ValueError("BYDAY ordinals need FREQ=MONTHLY or YEARLY")
                byday.add((n, WEEKDAYS[code]))
            else:
                byday.add((0, WEEKDAYS[code]))
        r["byday"] = byday
    if "BYSETPOS" in raw:
        if not any(k in raw for k in ("BYMONTH", "BYMONTHDAY", "BYDAY")):
            raise ValueError("BYSETPOS needs another BYxxx part")
        r["bysetpos"] = {_sint(v) for v in _items(raw["BYSETPOS"])}
        if not all(1 <= abs(p) <= 366 for p in r["bysetpos"]):
            raise ValueError("BYSETPOS out of range")
    if "WKST" in raw:
        if raw["WKST"] not in WEEKDAYS:
            raise ValueError("bad WKST")
        r["wkst"] = WEEKDAYS[raw["WKST"]]
    r["month_context"] = freq == "MONTHLY" or (freq == "YEARLY" and r["bymonth"] is not None)
    return r


def _apply_defaults(r: dict, start: date) -> None:
    if r["freq"] == "WEEKLY" and r["byday"] is None:
        r["byday"] = {(0, start.weekday())}
    if r["freq"] in ("MONTHLY", "YEARLY") and r["byday"] is None and r["bymonthday"] is None:
        r["bymonthday"] = {start.day}
        if r["freq"] == "YEARLY" and r["bymonth"] is None:
            r["bymonth"] = {start.month}


def _periods(r: dict, start: date):
    step = r["interval"]
    try:
        if r["freq"] == "DAILY":
            d = start
            while True:
                yield [d]
                d += timedelta(days=step)
        elif r["freq"] == "WEEKLY":
            d = start - timedelta(days=(start.weekday() - r["wkst"]) % 7)
            while True:
                yield [d + timedelta(days=i) for i in range(7)]
                d += timedelta(days=7 * step)
        elif r["freq"] == "MONTHLY":
            index = start.year * 12 + start.month - 1
            while index < 10000 * 12:
                y, m = divmod(index, 12)
                yield [date(y, m + 1, day) for day in range(1, calendar.monthrange(y, m + 1)[1] + 1)]
                index += step
        else:
            y = start.year
            while y <= 9999:
                months = sorted(r["bymonth"]) if r["bymonth"] else range(1, 13)
                yield [date(y, m, day) for m in months for day in range(1, calendar.monthrange(y, m)[1] + 1)]
                y += step
    except OverflowError:
        return


def _matches(r: dict, d: date) -> bool:
    if r["bymonth"] is not None and d.month not in r["bymonth"]:
        return False
    if r["bymonthday"] is not None:
        dim = calendar.monthrange(d.year, d.month)[1]
        if d.day not in r["bymonthday"] and d.day - dim - 1 not in r["bymonthday"]:
            return False
    if r["byday"] is not None:
        if r["month_context"]:
            pos, size = d.day, calendar.monthrange(d.year, d.month)[1]
        else:
            pos, size = d.timetuple().tm_yday, 366 if calendar.isleap(d.year) else 365
        nth, nth_from_end = (pos - 1) // 7 + 1, -((size - pos) // 7 + 1)
        if not any(wd == d.weekday() and n in (0, nth, nth_from_end) for n, wd in r["byday"]):
            return False
    return True


def _days(r: dict, start: date, horizon: date | None):
    for period in _periods(r, start):
        if horizon is not None and period[0] > horizon:
            return  # every later occurrence is past UNTIL or the window: stopping early changes nothing
        days = [d for d in period if _matches(r, d)]
        if r["bysetpos"] is not None:
            picked = set()
            for p in r["bysetpos"]:
                i = p - 1 if p > 0 else len(days) + p
                if 0 <= i < len(days):
                    picked.add(days[i])
            days = sorted(picked)
        for d in days:
            if d >= start:
                yield d


def expand(rule: str, dtstart: datetime, *, limit: int | None = None,
           between: tuple[datetime, datetime] | None = None) -> list[datetime]:
    r = _parse(rule)
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1")
    if r["count"] is None and r["until"] is None and limit is None and between is None:
        raise ValueError("the recurrence is unbounded")
    _apply_defaults(r, dtstart.date())
    out: list[datetime] = []
    emitted = 0
    until = r["until"]
    ends = [until[1] if until[0] == "date" else until[1].date()] if until else []
    ends += [between[1].date()] if between is not None else []
    for d in _days(r, dtstart.date(), min(ends) if ends else None):
        occ = datetime.combine(d, dtstart.time())
        if until is not None and (d > until[1] if until[0] == "date" else occ > until[1]):
            break
        emitted += 1
        if between is not None and occ > between[1]:
            break
        if between is None or occ >= between[0]:
            out.append(occ)
            if limit is not None and len(out) >= limit:
                break
        if r["count"] is not None and emitted >= r["count"]:
            break
    return out
