"""Rotations: members take turns holding the pager."""

from __future__ import annotations

from datetime import datetime, timedelta

from .timeutil import week_start


class Rotation:
    """Members are on call for ``length_days`` days each, taking turns in the order given.

    Shifts line up with ISO weeks: the first shift starts at the beginning of the week that contains ``start`` (Monday
    00:00 UTC), and the next member takes over every ``length_days`` days after that.
    """

    def __init__(self, members: list[str], start: datetime, length_days: int = 7):
        if not members:
            raise ValueError("a rotation needs at least one member")
        if len(set(members)) != len(members):
            raise ValueError("rotation members must be unique")
        if length_days < 1:
            raise ValueError("length_days must be at least 1")
        self.members = list(members)
        self.length = timedelta(days=length_days)
        self.first_shift = week_start(start)

    def shift_index(self, ts: datetime) -> int:
        """How many handoffs have happened by ts (0 during the first shift). Raises ValueError before the first shift."""
        if ts < self.first_shift:
            raise ValueError(f"{ts} is before the rotation's first shift ({self.first_shift})")
        return (ts - self.first_shift) // self.length

    def shift_bounds(self, ts: datetime) -> tuple[datetime, datetime]:
        """(start, end) of the shift that contains ts."""
        start = self.first_shift + self.shift_index(ts) * self.length
        return start, start + self.length

    def scheduled(self, ts: datetime) -> str:
        """Who the rotation puts on call at ts (overrides aren't considered here; see overrides.on_call)."""
        return self.members[self.shift_index(ts) % len(self.members)]

    def next_member(self, member: str) -> str:
        """Who comes after ``member`` in the rotation."""
        return self.members[(self.members.index(member) + 1) % len(self.members)]
