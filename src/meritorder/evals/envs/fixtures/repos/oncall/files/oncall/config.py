"""Rotations and overrides from config: the JSON the scheduling bot stores."""

from __future__ import annotations

from .overrides import Override
from .rotation import Rotation
from .timeutil import parse_ts


def _int_field(data: dict, name: str, default: int, lo: int, hi: int) -> int:
    value = data.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
        raise ValueError(f"{name} must be an integer from {lo} to {hi}, got {value!r}")
    return value


def _ts_field(data: dict, name: str):
    try:
        return parse_ts(data[name])
    except (KeyError, TypeError, ValueError, AttributeError):
        raise ValueError(f"{name} must be a timestamp like 2026-03-02T09:00") from None


def load_rotation(data: dict) -> Rotation:
    """A Rotation from {"members": [...], "start": "2026-03-02T00:00", "length_days": 7}.

    length_days is optional (default 7, from 1 to 28). Missing or invalid values raise ValueError naming the field.
    """
    members = data.get("members")
    if not isinstance(members, list) or not members or not all(isinstance(m, str) and m for m in members):
        raise ValueError("members must be a non-empty list of names")
    return Rotation(members, _ts_field(data, "start"), length_days=_int_field(data, "length_days", 7, 1, 28))


def load_overrides(items: list[dict]) -> list[Override]:
    """Overrides from [{"start": ..., "end": ..., "member": ...}, ...], in the order given."""
    out = []
    for item in items:
        member = item.get("member")
        if not isinstance(member, str) or not member:
            raise ValueError("member must be a name")
        out.append(Override(_ts_field(item, "start"), _ts_field(item, "end"), member))
    return out
