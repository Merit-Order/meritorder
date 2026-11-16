"""Busy summaries for the weekly digest."""

from __future__ import annotations

from typing import Iterable

from .intervals import Interval, merge, total


def busy_summary(busy: Iterable[Interval]) -> dict:
    """{"blocks": n, "busy_minutes": m, "longest_block": l} for one person's day.

    A block is a stretch of continuous busy time: meetings that overlap or follow each other with no gap form one block.
    busy_minutes counts each minute once; longest_block is the length of the longest block. All zero when nothing is
    busy.
    """
    busy = list(busy)
    blocks = merge(busy)
    return {"blocks": len(blocks), "busy_minutes": total(busy), "longest_block": max((b.length for b in blocks), default=0)}
