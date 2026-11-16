"""People's calendars."""

from __future__ import annotations

from collections import defaultdict

from .availability import DAY_END, DAY_START, free_slots
from .intervals import Interval, merge
from .timefmt import parse_range


class Calendar:
    """Busy events per person, for one day."""

    def __init__(self) -> None:
        self._events: dict[str, list[tuple[Interval, str]]] = defaultdict(list)

    def add(self, person: str, when: str | Interval, title: str = "") -> Interval:
        """Add a busy event; ``when`` is 'HH:MM-HH:MM' or an Interval."""
        iv = parse_range(when) if isinstance(when, str) else when
        self._events[person].append((iv, title))
        return iv

    def events(self, person: str) -> list[tuple[Interval, str]]:
        """The person's events in the order they were added."""
        return list(self._events.get(person, []))

    def busy(self, person: str) -> list[Interval]:
        """The person's busy time: sorted, disjoint intervals with no two touching."""
        return merge(iv for iv, _ in self._events.get(person, []))

    def common_free(self, people: list[str], min_length: int = 30, day_start: int = DAY_START,
                    day_end: int = DAY_END) -> list[Interval]:
        """Slots of at least min_length minutes when everyone in ``people`` is free."""
        busy = [iv for person in people for iv in self.busy(person)]
        return free_slots(busy, day_start, day_end, min_length)
