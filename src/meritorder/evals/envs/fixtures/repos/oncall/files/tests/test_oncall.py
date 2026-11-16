import unittest

from oncall.config import load_overrides, load_rotation
from oncall.overrides import Override, on_call
from oncall.pager import escalation
from oncall.report import weekly_hours
from oncall.rotation import Rotation
from oncall.timeutil import parse_ts as T

TEAM = ["ana", "ben", "chen"]


class RotationTests(unittest.TestCase):
    def test_first_week(self):
        rot = Rotation(TEAM, T("2026-03-02T00:00"))
        self.assertEqual(rot.scheduled(T("2026-03-03T12:00")), "ana")
        self.assertEqual(rot.scheduled(T("2026-03-11T12:00")), "ben")

    def test_hands_off_on_monday(self):  # ISSUE.md, bug
        rot = Rotation(TEAM, T("2026-03-02T00:00"))
        self.assertEqual(rot.scheduled(T("2026-03-08T10:00")), "ana")
        self.assertEqual(rot.scheduled(T("2026-03-09T00:00")), "ben")

    def test_handoff_hour(self):  # ISSUE.md, feature
        rot = Rotation(TEAM, T("2026-03-02T00:00"), handoff_hour=9)
        self.assertEqual(rot.scheduled(T("2026-03-09T08:00")), "ana")
        self.assertEqual(rot.scheduled(T("2026-03-09T09:00")), "ben")


class OverrideTests(unittest.TestCase):
    def test_override_wins(self):
        rot = Rotation(TEAM, T("2026-03-02T00:00"))
        cover = [Override(T("2026-03-04T08:00"), T("2026-03-04T20:00"), "chen")]
        self.assertEqual(on_call(rot, cover, T("2026-03-04T12:00")), "chen")
        self.assertEqual(on_call(rot, cover, T("2026-03-04T20:00")), "ana")


class ReportTests(unittest.TestCase):
    def test_tuesday(self):
        rot = Rotation(TEAM, T("2026-03-02T00:00"))
        self.assertEqual(weekly_hours(rot, [], T("2026-03-03T00:00"), T("2026-03-03T05:00")), {"2026-03-02": {"ana": 5}})

    def test_sunday_counts_in_its_own_week(self):  # ISSUE.md, bug
        rot = Rotation(TEAM, T("2026-03-02T00:00"))
        hours = weekly_hours(rot, [], T("2026-03-08T00:00"), T("2026-03-09T06:00"))
        self.assertEqual(hours, {"2026-03-02": {"ana": 24}, "2026-03-09": {"ben": 6}})


class PagerTests(unittest.TestCase):
    def test_escalation(self):
        rot = Rotation(TEAM, T("2026-03-02T00:00"))
        self.assertEqual(escalation(rot, [], T("2026-03-03T12:00")), ["ana", "ben"])


class ConfigTests(unittest.TestCase):
    def test_load_rotation(self):
        rot = load_rotation({"members": TEAM, "start": "2026-03-02T00:00", "length_days": 14})
        self.assertEqual(rot.scheduled(T("2026-03-12T00:00")), "ana")

    def test_bad_length(self):
        with self.assertRaisesRegex(ValueError, "length_days"):
            load_rotation({"members": TEAM, "start": "2026-03-02T00:00", "length_days": 0})

    def test_load_overrides(self):
        (o,) = load_overrides([{"start": "2026-03-04T08:00", "end": "2026-03-04T20:00", "member": "chen"}])
        self.assertEqual((o.member, o.end), ("chen", T("2026-03-04T20:00")))


if __name__ == "__main__":
    unittest.main()
