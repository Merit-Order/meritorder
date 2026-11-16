"""Who gets paged."""

from __future__ import annotations

from datetime import datetime

from .overrides import Override, on_call
from .rotation import Rotation


def escalation(rotation: Rotation, overrides: list[Override], ts: datetime) -> list[str]:
    """Who to page at ts, in order: whoever is on call (overrides included), then the backup.

    The backup is the member who has the next shift in the rotation; overrides don't change who that is. When the backup
    would be the person already paged first (a one-person rotation, or the next member covering this shift), the list
    has just one name.
    """
    primary = on_call(rotation, overrides, ts)
    backup = rotation.next_member(rotation.scheduled(ts))
    return [primary] if backup == primary else [primary, backup]
