"""Parsing and formatting HH:MM times (minutes since midnight)."""

from __future__ import annotations

import re

from .intervals import Interval, make

_TIME = re.compile(r"^([01]\d|2[0-4]):([0-5]\d)$")


def parse(text: str) -> int:
    """'09:30' -> 570. '24:00' is allowed: it's the end of the day."""
    m = _TIME.match(text.strip())
    if not m or (m.group(1) == "24" and m.group(2) != "00"):
        raise ValueError(f"not a time: {text!r}")
    return int(m.group(1)) * 60 + int(m.group(2))


def fmt(minutes: int) -> str:
    """570 -> '09:30'."""
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def parse_range(text: str) -> Interval:
    """'09:00-10:30' -> Interval(540, 630)."""
    start, sep, end = text.partition("-")
    if not sep:
        raise ValueError(f"not a time range: {text!r}")
    return make(parse(start), parse(end))


def fmt_range(iv: Interval) -> str:
    """Interval(540, 630) -> '09:00-10:30'."""
    return f"{fmt(iv.start)}-{fmt(iv.end)}"
