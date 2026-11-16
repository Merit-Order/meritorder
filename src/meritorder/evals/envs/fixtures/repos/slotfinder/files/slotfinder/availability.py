"""Free time in a working day."""

from __future__ import annotations

from typing import Iterable

from .intervals import Interval, clip, merge

DAY_START = 9 * 60
DAY_END = 17 * 60


def free_slots(busy: Iterable[Interval], day_start: int = DAY_START, day_end: int = DAY_END,
               min_length: int = 30) -> list[Interval]:
    """The free time between day_start and day_end, given busy intervals.

    Busy intervals may overlap, touch, or reach outside the working day. Returns the gaps between busy time as sorted
    Intervals inside [day_start, day_end); gaps shorter than min_length minutes are left out.
    """
    free = []
    cursor = day_start
    for block in merge(clip(busy, day_start, day_end)):
        if block.start > cursor:
            free.append(Interval(cursor, block.start))
        cursor = block.end
    if cursor < day_end:
        free.append(Interval(cursor, day_end))
    return [slot for slot in free if slot.length >= min_length]


def first_free(busy: Iterable[Interval], length: int, day_start: int = DAY_START, day_end: int = DAY_END) -> Interval | None:
    """The earliest free slot of exactly ``length`` minutes, or None if there's no room."""
    for slot in free_slots(busy, day_start, day_end, min_length=length):
        return Interval(slot.start, slot.start + length)
    return None
