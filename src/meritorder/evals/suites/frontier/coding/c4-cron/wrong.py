"""Cron next-run calculator."""

from datetime import date, datetime, timedelta

_MACROS = {
    "@yearly": "0 0 1 1 *", "@annually": "0 0 1 1 *", "@monthly": "0 0 1 * *", "@weekly": "0 0 * * 0",
    "@daily": "0 0 * * *", "@midnight": "0 0 * * *", "@hourly": "0 * * * *",
}
_MONTHS = {n: i for i, n in enumerate("JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split(), 1)}
_DAYS = {n: i for i, n in enumerate("SUN MON TUE WED THU FRI SAT".split())}
_FIELDS = [(0, 59, None), (0, 23, None), (1, 31, None), (1, 12, _MONTHS), (0, 7, _DAYS)]


def _number(text: str) -> int:
    if not text or not text.isascii() or not text.isdigit():
        raise ValueError(f"not a number: {text!r}")
    return int(text)


def _value(text: str, lo: int, hi: int, names: dict | None) -> int:
    if names is not None and text.upper() in names:
        v = names[text.upper()]
    else:
        v = _number(text)
    if not lo <= v <= hi:
        raise ValueError(f"{v} is outside {lo}-{hi}")
    return v


def _field(text: str, lo: int, hi: int, names: dict | None) -> set[int]:
    values: set[int] = set()
    for item in text.split(","):
        rng, slash, step_text = item.partition("/")
        step = _number(step_text) if slash else 1
        if step < 1:
            raise ValueError("step must be at least 1")
        if rng == "*":
            start, end = lo, hi
        elif "-" in rng:
            a, _, b = rng.partition("-")
            start, end = _value(a, lo, hi, names), _value(b, lo, hi, names)
            if start > end:
                raise ValueError(f"reversed range {rng!r}")
        else:
            start = _value(rng, lo, hi, names)
            end = (6 if hi == 7 else hi) if slash else start  # Sunday is already 0
        values.update(range(start, end + 1, step))
    return values


class _Spec:
    def __init__(self, expr: str):
        if not isinstance(expr, str):
            raise ValueError("expression must be a string")
        text = expr.strip()
        if text.startswith("@"):
            if text.lower() not in _MACROS:
                raise ValueError(f"unknown macro {text!r}")
            text = _MACROS[text.lower()]
        parts = text.split()
        if len(parts) != 5:
            raise ValueError("a cron expression has five fields")
        sets = [_field(p, lo, hi, names) for p, (lo, hi, names) in zip(parts, _FIELDS)]
        self.minutes, self.hours, self.doms, self.months, dows = sets
        if 7 in dows:
            dows = (dows - {7}) | {0}
        self.dows = dows
        self.either = parts[2] != "*" and parts[4] != "*"
        self.hour_list = sorted(self.hours)
        self.minute_list = sorted(self.minutes)

    def day_ok(self, d: date) -> bool:
        dom_ok = d.day in self.doms
        dow_ok = d.isoweekday() % 7 in self.dows
        return (dom_ok or dow_ok) if self.either else (dom_ok and dow_ok)

    def after(self, after: datetime) -> datetime:
        start = after.replace(second=0, microsecond=0) + timedelta(minutes=1)
        last_year = after.year + 50
        d = start.date()
        while d.year <= last_year:
            if d.month not in self.months:
                d = date(d.year + 1, 1, 1) if d.month == 12 else date(d.year, d.month + 1, 1)
                continue
            if self.day_ok(d):
                for h in self.hour_list:
                    for m in self.minute_list:
                        t = datetime(d.year, d.month, d.day, h, m)
                        if t >= start:
                            return t
            d += timedelta(days=1)
        raise ValueError("the expression doesn't match any time in the search horizon")


def _check_after(after: datetime) -> None:
    if not isinstance(after, datetime):
        raise ValueError("after must be a datetime")
    if after.tzinfo is not None:
        raise ValueError("after must be naive")


def next_run(expr: str, after: datetime) -> datetime:
    _check_after(after)
    return _Spec(expr).after(after)


def next_runs(expr: str, after: datetime, n: int) -> list[datetime]:
    _check_after(after)
    spec = _Spec(expr)
    if n < 0:
        raise ValueError("n must not be negative")
    out: list[datetime] = []
    t = after
    for _ in range(n):
        t = spec.after(t)
        out.append(t)
    return out
