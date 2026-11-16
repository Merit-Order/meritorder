import unittest

from slotfinder import Calendar, busy_summary, first_free, free_slots
from slotfinder.timefmt import fmt_range, parse, parse_range


def ranges(intervals):
    return [fmt_range(iv) for iv in intervals]


def busy(*texts):
    return [parse_range(t) for t in texts]


class TimeFormatTests(unittest.TestCase):
    def test_parse(self):
        self.assertEqual((parse("09:30"), parse("24:00")), (570, 1440))
        with self.assertRaises(ValueError):
            parse("9.30")

    def test_range(self):
        self.assertEqual(fmt_range(parse_range("09:00-10:30")), "09:00-10:30")


class FreeSlotTests(unittest.TestCase):
    def test_one_meeting(self):
        self.assertEqual(ranges(free_slots(busy("10:00-11:00"))), ["09:00-10:00", "11:00-17:00"])

    def test_short_gaps_are_dropped(self):
        self.assertEqual(ranges(free_slots(busy("09:00-10:00", "10:20-16:00"))), ["16:00-17:00"])

    def test_meeting_inside_all_hands(self):  # ISSUE.md, report 1
        self.assertEqual(ranges(free_slots(busy("09:00-12:00", "10:00-11:00"))), ["12:00-17:00"])

    def test_first_free(self):
        self.assertEqual(fmt_range(first_free(busy("09:00-09:45"), 60)), "09:45-10:45")


class SummaryTests(unittest.TestCase):
    def test_nothing_busy(self):
        self.assertEqual(busy_summary([]), {"blocks": 0, "busy_minutes": 0, "longest_block": 0})

    def test_back_to_back_meetings_are_one_block(self):  # ISSUE.md, report 2
        summary = busy_summary(busy("09:00-10:00", "10:00-10:30", "10:30-11:00"))
        self.assertEqual(summary, {"blocks": 1, "busy_minutes": 120, "longest_block": 120})


class CalendarTests(unittest.TestCase):
    def test_common_free(self):
        cal = Calendar()
        cal.add("dana", "09:00-12:00", "all-hands")
        cal.add("omar", "13:00-14:00", "planning")
        self.assertEqual(ranges(cal.common_free(["dana", "omar"])), ["12:00-13:00", "14:00-17:00"])


if __name__ == "__main__":
    unittest.main()
