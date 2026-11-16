"""Half-open time intervals, in minutes since midnight.

Interval(start, end) covers the minutes start <= t < end. Two intervals touch when one ends exactly where the next
starts; there is no gap between them.
"""

from __future__ import annotations

from typing import Iterable, NamedTuple


class Interval(NamedTuple):
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


def make(start: int, end: int) -> Interval:
    """An Interval, checking that it doesn't end before it starts."""
    if end < start:
        raise ValueError(f"interval ends before it starts: {start}-{end}")
    return Interval(start, end)


def merge(intervals: Iterable[Interval]) -> list[Interval]:
    """Merge overlapping and touching intervals.

    Returns a new sorted list of disjoint intervals, no two of them touching, that covers exactly the minutes the input
    covers. Empty intervals (start == end) cover nothing and are dropped. The input isn't modified.
    """
    result: list[Interval] = []
    for iv in sorted(intervals):
        if iv.start == iv.end:
            continue
        if result and iv.start < result[-1].end:
            result[-1] = Interval(result[-1].start, iv.end)
        else:
            result.append(iv)
    return result


def clip(intervals: Iterable[Interval], lo: int, hi: int) -> list[Interval]:
    """The parts of the intervals inside [lo, hi), leaving out any that end up empty."""
    out = []
    for iv in intervals:
        start, end = max(iv.start, lo), min(iv.end, hi)
        if start < end:
            out.append(Interval(start, end))
    return out


def total(intervals: Iterable[Interval]) -> int:
    """Minutes covered by the intervals, counting minutes covered more than once only once."""
    return sum(iv.length for iv in merge(intervals))
