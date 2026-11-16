"""The weekly on-call hours export for payroll."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta

from .overrides import Override, on_call
from .rotation import Rotation
from .timeutil import hours_between, week_start


def weekly_hours(rotation: Rotation, overrides: list[Override], start: datetime, end: datetime) -> dict[str, dict[str, int]]:
    """Hours on call per member per calendar week, counting each whole hour in [start, end).

    Returns {week: {member: hours}}, where week is the ISO date of the Monday that starts the calendar week (Monday 00:00
    to Monday 00:00 UTC) containing the hour, e.g. '2026-03-02'. Weeks follow the calendar, not the rotation's
    handoffs. Members without hours in a week, and weeks without hours, are left out.
    """
    out: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for hour in hours_between(start, end):
        # week_start() gives the Sunday before; step forward a day to get the Monday.
        monday = week_start(hour) + timedelta(days=1)
        out[monday.date().isoformat()][on_call(rotation, overrides, hour)] += 1
    return {week: dict(members) for week, members in out.items()}
