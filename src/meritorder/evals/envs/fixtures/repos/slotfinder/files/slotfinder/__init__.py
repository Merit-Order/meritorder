"""slotfinder: find free meeting slots in people's calendars."""

from .availability import first_free, free_slots
from .calendar import Calendar
from .intervals import Interval, merge, total
from .summary import busy_summary

__all__ = ["Calendar", "Interval", "busy_summary", "first_free", "free_slots", "merge", "total"]
