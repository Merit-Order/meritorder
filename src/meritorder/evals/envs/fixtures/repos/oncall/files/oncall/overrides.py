"""Overrides: someone covers part of another member's shift."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .rotation import Rotation


@dataclass(frozen=True)
class Override:
    """``member`` is on call for [start, end), whatever the rotation says."""

    start: datetime
    end: datetime
    member: str

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError("an override must end after it starts")

    def covers(self, ts: datetime) -> bool:
        return self.start <= ts < self.end


def on_call(rotation: Rotation, overrides: list[Override], ts: datetime) -> str:
    """Who is on call at ts: the last override in the list that covers ts, or else the rotation's scheduled member."""
    for override in reversed(overrides):
        if override.covers(ts):
            return override.member
    return rotation.scheduled(ts)
