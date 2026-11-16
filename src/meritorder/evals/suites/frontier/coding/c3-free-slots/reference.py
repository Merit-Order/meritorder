"""Reference solution for c3-free-slots."""

from datetime import date, datetime, timedelta

_FMT = "%Y-%m-%dT%H:%M"


def _digits(s: str) -> bool:
    return bool(s) and all("0" <= c <= "9" for c in s)


def _timestamp(s) -> datetime:
    if not isinstance(s, str) or len(s) != 16 or s[4] != "-" or s[7] != "-" or s[10] != "T" or s[13] != ":":
        raise ValueError(f"bad timestamp {s!r}")
    parts = (s[0:4], s[5:7], s[8:10], s[11:13], s[14:16])
    if not all(_digits(p) for p in parts):
        raise ValueError(f"bad timestamp {s!r}")
    y, mo, d, h, mi = (int(p) for p in parts)
    if h > 23 or mi > 59:
        raise ValueError(f"bad timestamp {s!r}")
    try:
        return datetime(y, mo, d, h, mi)
    except ValueError:
        raise ValueError(f"bad timestamp {s!r}") from None


def _time_of_day(s, allow_24: bool) -> int:
    if not isinstance(s, str) or len(s) != 5 or s[2] != ":" or not _digits(s[:2]) or not _digits(s[3:]):
        raise ValueError(f"bad time {s!r}")
    h, m = int(s[:2]), int(s[3:])
    if allow_24 and (h, m) == (24, 0):
        return 24 * 60
    if h > 23 or m > 59:
        raise ValueError(f"bad time {s!r}")
    return h * 60 + m


def _fmt(t: datetime) -> str:
    return t.strftime(_FMT)


def free_slots(busy, window_start, window_end, *, workday=("09:00", "17:00"), min_minutes=30, buffer=0,
               weekdays_only=True):
    ws, we = _timestamp(window_start), _timestamp(window_end)
    day_start, day_end = _time_of_day(workday[0], False), _time_of_day(workday[1], True)
    intervals = []
    for start, end in busy:
        s, e = _timestamp(start), _timestamp(end)
        if s > e:
            raise ValueError("busy interval ends before it starts")
        if s < e:
            intervals.append((s, e))
    if we < ws:
        raise ValueError("window ends before it starts")
    if day_start >= day_end:
        raise ValueError("workday must start before it ends")
    if min_minutes < 1:
        raise ValueError("min_minutes must be at least 1")
    if buffer < 0:
        raise ValueError("buffer must not be negative")
    if we == ws:
        return []
    pad = timedelta(minutes=buffer)
    merged: list[list[datetime]] = []
    for s, e in sorted((s - pad, e + pad) for s, e in intervals):
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    slots = []
    d: date = ws.date()
    while d <= we.date():
        if not (weekdays_only and d.weekday() >= 5):
            midnight = datetime(d.year, d.month, d.day)
            lo = max(midnight + timedelta(minutes=day_start), ws)
            hi = min(midnight + timedelta(minutes=day_end), we)
            cursor = lo
            for s, e in merged:
                if e <= cursor or s >= hi:
                    continue
                if s > cursor:
                    slots.append((cursor, s))
                cursor = max(cursor, e)
                if cursor >= hi:
                    break
            if cursor < hi:
                slots.append((cursor, hi))
        d += timedelta(days=1)
    need = timedelta(minutes=min_minutes)
    return [(_fmt(s), _fmt(e)) for s, e in slots if e - s >= need]
