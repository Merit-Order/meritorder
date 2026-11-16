"""Time helpers. Every timestamp in oncall is a naive datetime in UTC."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterator

HOUR = timedelta(hours=1)
DAY = timedelta(days=1)


def parse_ts(text: str) -> datetime:
    """Parse '2026-03-02T09:00' (a space instead of the T is fine too). Raises ValueError for anything else."""
    return datetime.strptime(text.strip().replace(" ", "T", 1), "%Y-%m-%dT%H:%M")


def week_start(ts: datetime) -> datetime:
    """The start of the ISO week containing ts: the Monday on or before ts, at 00:00."""
    monday = ts - timedelta(days=ts.isoweekday() % 7)
    return monday.replace(hour=0, minute=0, second=0, microsecond=0)


def hours_between(start: datetime, end: datetime) -> Iterator[datetime]:
    """The start of every whole hour in [start, end). start must be on the hour."""
    if start.minute or start.second or start.microsecond:
        raise ValueError(f"{start} isn't on the hour")
    t = start
    while t < end:
        yield t
        t += HOUR
