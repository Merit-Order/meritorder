"""Hidden tests for a5-oncall: week_start's contract, Monday handoffs, handoff_hour, and everything built on them."""

import unittest
from datetime import datetime, timedelta

from oncall.config import load_overrides, load_rotation
from oncall.overrides import Override, on_call
from oncall.pager import escalation
from oncall.report import weekly_hours
from oncall.rotation import Rotation
from oncall.timeutil import hours_between, parse_ts as T, week_start

TEAM = ["ana", "ben", "chen"]


class WeekStartTests(unittest.TestCase):
    def test_every_day_of_the_week(self):
        for day in range(7):  # 2026-03-02 is a Monday
            ts = T("2026-03-02T13:37") + timedelta(days=day)
            self.assertEqual(week_start(ts), datetime(2026, 3, 2), ts)

    def test_edges(self):
        self.assertEqual(week_start(T("2026-03-02T00:00")), datetime(2026, 3, 2))
        self.assertEqual(week_start(T("2026-03-08T23:59")), datetime(2026, 3, 2))
        self.assertEqual(week_start(T("2026-03-09T00:00")), datetime(2026, 3, 9))
        self.assertEqual(week_start(datetime(2026, 1, 1, 6, 30, 15, 99)), datetime(2025, 12, 29))

    def test_hours_between(self):
        self.assertEqual(len(list(hours_between(T("2026-03-08T22:00"), T("2026-03-09T02:00")))), 4)


class RotationTests(unittest.TestCase):
    def test_monday_midnight_handoffs(self):
        rot = Rotation(TEAM, T("2026-03-02T00:00"))
        self.assertEqual([rot.scheduled(T(t)) for t in ("2026-03-02T00:00", "2026-03-08T23:59", "2026-03-09T00:00",
                                                        "2026-03-15T23:00", "2026-03-16T00:00", "2026-03-23T00:00")],
                         ["ana", "ana", "ben", "ben", "chen", "ana"])

    def test_start_mid_week_and_on_sunday(self):
        for start in ("2026-03-04T12:00", "2026-03-08T18:00"):
            rot = Rotation(TEAM, T(start))
            self.assertEqual(rot.first_shift, datetime(2026, 3, 2), start)
            self.assertEqual(rot.scheduled(T("2026-03-02T00:00")), "ana")
            with self.assertRaises(ValueError):
                rot.scheduled(T("2026-03-01T23:59"))

    def test_positional_arguments_still_work(self):
        rot = Rotation(TEAM, T("2026-03-02T00:00"), 14)
        self.assertEqual(rot.scheduled(T("2026-03-15T23:00")), "ana")
        self.assertEqual(rot.scheduled(T("2026-03-16T00:00")), "ben")

    def test_shift_bounds(self):
        rot = Rotation(TEAM, T("2026-03-02T00:00"))
        self.assertEqual(rot.shift_bounds(T("2026-03-08T10:00")), (datetime(2026, 3, 2), datetime(2026, 3, 9)))


class HandoffHourTests(unittest.TestCase):
    def test_weekly(self):
        rot = Rotation(TEAM, T("2026-03-04T12:00"), handoff_hour=9)
        self.assertEqual(rot.shift_bounds(T("2026-03-02T09:00")), (datetime(2026, 3, 2, 9), datetime(2026, 3, 9, 9)))
        self.assertEqual(rot.scheduled(T("2026-03-02T09:00")), "ana")
        self.assertEqual(rot.scheduled(T("2026-03-09T08:59")), "ana")
        self.assertEqual(rot.scheduled(T("2026-03-09T09:00")), "ben")
        self.assertEqual(rot.scheduled(T("2026-03-16T09:00")), "chen")
        with self.assertRaises(ValueError):
            rot.scheduled(T("2026-03-02T08:59"))
        self.assertEqual(rot.shift_bounds(T("2026-03-10T00:00")), (datetime(2026, 3, 9, 9), datetime(2026, 3, 16, 9)))

    def test_daily(self):
        rot = Rotation(TEAM, T("2026-03-02T00:00"), length_days=1, handoff_hour=6)
        self.assertEqual(rot.scheduled(T("2026-03-03T05:59")), "ana")
        self.assertEqual(rot.scheduled(T("2026-03-03T06:00")), "ben")
        self.assertEqual(rot.scheduled(T("2026-03-05T06:00")), "ana")

    def test_default_is_midnight(self):
        self.assertEqual(Rotation(TEAM, T("2026-03-04T12:00"), handoff_hour=0).first_shift, datetime(2026, 3, 2))
        late = Rotation(TEAM, T("2026-03-04T12:00"), length_days=7, handoff_hour=23)
        self.assertEqual(late.shift_bounds(T("2026-03-03T00:00")), (datetime(2026, 3, 2, 23), datetime(2026, 3, 9, 23)))

    def test_invalid(self):
        for bad in (-1, 24, 9.5, "9"):
            with self.assertRaises(ValueError, msg=repr(bad)):
                Rotation(TEAM, T("2026-03-02T00:00"), handoff_hour=bad)


class ConfigTests(unittest.TestCase):
    def test_handoff_hour(self):
        rot = load_rotation({"members": TEAM, "start": "2026-03-02T00:00", "handoff_hour": 9})
        self.assertEqual(rot.scheduled(T("2026-03-09T08:00")), "ana")
        self.assertEqual(rot.scheduled(T("2026-03-09T09:00")), "ben")
        self.assertEqual(load_rotation({"members": TEAM, "start": "2026-03-04T12:00"}).first_shift, datetime(2026, 3, 2))

    def test_invalid_handoff_hour_names_the_field(self):
        for bad in (24, -3, "9", 9.0):
            with self.assertRaisesRegex(ValueError, "handoff_hour", msg=repr(bad)):
                load_rotation({"members": TEAM, "start": "2026-03-02T00:00", "handoff_hour": bad})

    def test_other_fields_unchanged(self):
        rot = load_rotation({"members": TEAM, "start": "2026-03-02 00:00", "length_days": 14})
        self.assertEqual(rot.scheduled(T("2026-03-16T00:00")), "ben")
        for data, field in (({"members": [], "start": "2026-03-02T00:00"}, "members"),
                            ({"members": TEAM, "start": "March 2"}, "start"),
                            ({"members": TEAM, "start": "2026-03-02T00:00", "length_days": 29}, "length_days")):
            with self.assertRaisesRegex(ValueError, field):
                load_rotation(data)
        with self.assertRaisesRegex(ValueError, "end"):
            load_overrides([{"start": "2026-03-04T08:00", "member": "chen"}])


class ReportTests(unittest.TestCase):
    def test_two_full_weeks(self):
        rot = Rotation(TEAM, T("2026-03-02T00:00"))
        self.assertEqual(weekly_hours(rot, [], T("2026-03-02T00:00"), T("2026-03-16T00:00")),
                         {"2026-03-02": {"ana": 168}, "2026-03-09": {"ben": 168}})

    def test_sunday_with_override(self):
        rot = Rotation(TEAM, T("2026-03-02T00:00"))
        cover = [Override(T("2026-03-08T10:00"), T("2026-03-08T14:00"), "chen")]
        self.assertEqual(weekly_hours(rot, cover, T("2026-03-08T00:00"), T("2026-03-09T02:00")),
                         {"2026-03-02": {"ana": 20, "chen": 4}, "2026-03-09": {"ben": 2}})

    def test_handoff_hour_keeps_calendar_weeks(self):
        rot = Rotation(TEAM, T("2026-03-02T00:00"), handoff_hour=9)
        self.assertEqual(weekly_hours(rot, [], T("2026-03-02T09:00"), T("2026-03-16T09:00")),
                         {"2026-03-02": {"ana": 159}, "2026-03-09": {"ana": 9, "ben": 159}, "2026-03-16": {"ben": 9}})


class PagerTests(unittest.TestCase):
    def test_sunday_and_monday(self):
        rot = Rotation(TEAM, T("2026-03-02T00:00"))
        self.assertEqual(escalation(rot, [], T("2026-03-08T10:00")), ["ana", "ben"])
        self.assertEqual(escalation(rot, [], T("2026-03-09T00:00")), ["ben", "chen"])

    def test_override_by_the_backup(self):
        rot = Rotation(TEAM, T("2026-03-02T00:00"))
        cover = [Override(T("2026-03-08T10:00"), T("2026-03-08T14:00"), "ben")]
        self.assertEqual(escalation(rot, cover, T("2026-03-08T11:00")), ["ben"])
        self.assertEqual(on_call(rot, cover, T("2026-03-08T14:00")), "ana")

    def test_with_handoff_hour(self):
        rot = Rotation(TEAM, T("2026-03-02T00:00"), handoff_hour=9)
        self.assertEqual(escalation(rot, [], T("2026-03-09T08:00")), ["ana", "ben"])
        self.assertEqual(escalation(rot, [], T("2026-03-09T09:00")), ["ben", "chen"])

    def test_one_person(self):
        self.assertEqual(escalation(Rotation(["solo"], T("2026-03-02T00:00")), [], T("2026-03-05T00:00")), ["solo"])
