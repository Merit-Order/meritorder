# Shifts change on Sunday instead of Monday; weekly report off for Sundays; add handoff_hour

## Bug

Since the move to the new scheduler, shifts change at Sunday 00:00 UTC instead of Monday 00:00 UTC.
Example: rotation ["ana", "ben", "chen"] starting 2026-03-02 (a Monday). On Sunday 2026-03-08 at 10:00
the pager goes to ben, but ana's week runs until Monday 2026-03-09 00:00.

The weekly hours export (`report.weekly_hours`) is wrong too: hours on a Sunday are counted in the
following week.

Suggested fix from #ops: "just add `+ timedelta(days=1)` in Rotation, like report.py already does".
Nobody has checked this; please find the real cause.

## Feature: handoff_hour

Teams want handoffs during working hours. Add an optional `handoff_hour`: an `int` from 0 to 23,
default 0.

- `Rotation(members, start, length_days=7, handoff_hour=0)`: every handoff happens at `handoff_hour:00`
  UTC instead of midnight. The first shift starts at `handoff_hour:00` on the Monday of the ISO week
  that contains `start`; asking about an earlier time raises ValueError, as it does today. Any other
  value than an int from 0 to 23 raises ValueError.
- `config.load_rotation` accepts an optional `"handoff_hour"` key with the same rules; invalid values
  raise ValueError naming the field, like the other fields.
- `report.weekly_hours` keeps counting hours per calendar week (Monday 00:00 UTC), whatever the handoff
  hour.
- `pager.escalation`: the backup is still whoever has the next shift.

Keep the public API backwards compatible.
