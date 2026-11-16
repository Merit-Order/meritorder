"""Plausible but buggy c3-free-slots: strptime accepts unpadded fields, and a slot of exactly min_minutes is dropped."""

from datetime import datetime, timedelta

FMT = "%Y-%m-%dT%H:%M"


def _parse(ts):
    return datetime.strptime(ts, FMT)


def _tod(s, end=False):
    if end and s == "24:00":
        return timedelta(days=1)
    t = datetime.strptime(s, "%H:%M")
    if len(s) != 5:
        raise ValueError(s)
    return timedelta(hours=t.hour, minutes=t.minute)


def free_slots(busy, window_start, window_end, *, workday=("09:00", "17:00"), min_minutes=30, buffer=0,
               weekdays_only=True):
    ws, we = _parse(window_start), _parse(window_end)
    if we < ws:
        raise ValueError("bad window")
    start_off, end_off = _tod(workday[0]), _tod(workday[1], end=True)
    if start_off >= end_off:
        raise ValueError("bad workday")
    if min_minutes < 1 or buffer < 0:
        raise ValueError("bad parameters")
    blocks = []
    for s, e in busy:
        s, e = _parse(s), _parse(e)
        if s > e:
            raise ValueError("bad interval")
        if s != e:
            blocks.append((s - timedelta(minutes=buffer), e + timedelta(minutes=buffer)))
    blocks.sort()
    result = []
    day = datetime(ws.year, ws.month, ws.day)
    while day.date() <= we.date():
        if not weekdays_only or day.weekday() < 5:
            lo, hi = max(day + start_off, ws), min(day + end_off, we)
            t = lo
            for s, e in blocks:
                if s >= hi:
                    break
                if e <= t:
                    continue
                if s > t and (s - t) > timedelta(minutes=min_minutes):
                    result.append((t, s))
                t = max(t, e)
            if t < hi and (hi - t) > timedelta(minutes=min_minutes):
                result.append((t, hi))
        day += timedelta(days=1)
    return [(a.strftime(FMT), b.strftime(FMT)) for a, b in result]
