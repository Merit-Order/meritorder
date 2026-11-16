# oncall

On-call rotations for the platform teams: who holds the pager when, overrides when someone covers for a
colleague, who gets paged, and the weekly hours export for payroll.

All timestamps are naive `datetime`s in UTC. Weeks are ISO weeks: Monday 00:00 to the next Monday 00:00.

```python
from oncall.config import load_rotation
from oncall.overrides import Override, on_call
from oncall.timeutil import parse_ts

rot = load_rotation({"members": ["ana", "ben", "chen"], "start": "2026-03-02T00:00"})
on_call(rot, [Override(parse_ts("2026-03-04T08:00"), parse_ts("2026-03-04T20:00"), "chen")], parse_ts("2026-03-04T12:00"))
```

Modules: `timeutil`, `rotation`, `overrides`, `pager`, `report`, `config`.

Run the tests with `python -m unittest discover -s tests`.
