"""Hidden tests for a4-slotfinder: documented behaviour of the shared interval helpers and everything built on them."""

import unittest

from slotfinder import Calendar, Interval, busy_summary, first_free, free_slots, merge, total
from slotfinder.intervals import clip
from slotfinder.timefmt import fmt_range, parse, parse_range


def iv(text):
    return parse_range(text)


def ranges(intervals):
    return [fmt_range(i) for i in intervals]


class MergeTests(unittest.TestCase):
    def test_touching(self):
        self.assertEqual(ranges(merge([iv("09:00-10:00"), iv("10:00-10:30")])), ["09:00-10:30"])

    def test_contained(self):
        self.assertEqual(ranges(merge([iv("09:00-12:00"), iv("10:00-11:00")])), ["09:00-12:00"])

    def test_unsorted_chains(self):
        got = merge([Interval(700, 800), Interval(540, 600), Interval(590, 710), Interval(900, 960), Interval(960, 1000),
                     Interval(905, 915)])
        self.assertEqual(got, [Interval(540, 800), Interval(900, 1000)])

    def test_several_nested(self):
        got = merge([Interval(540, 1000), Interval(560, 600), Interval(600, 620), Interval(700, 720), Interval(1000, 1010)])
        self.assertEqual(got, [Interval(540, 1010)])

    def test_empty_and_disjoint(self):
        self.assertEqual(merge([]), [])
        self.assertEqual(merge([Interval(600, 600), Interval(540, 560)]), [Interval(540, 560)])
        self.assertEqual(merge([Interval(600, 660), Interval(540, 570)]), [Interval(540, 570), Interval(600, 660)])

    def test_result_types_and_input_untouched(self):
        data = [Interval(600, 660), Interval(540, 620)]
        got = merge(data)
        self.assertEqual(data, [Interval(600, 660), Interval(540, 620)])
        self.assertTrue(all(isinstance(i, Interval) for i in got))
        self.assertEqual(got, [Interval(540, 660)])

    def test_accepts_any_iterable(self):
        self.assertEqual(merge(i for i in [Interval(60, 120), Interval(0, 60)]), [Interval(0, 120)])


class TotalAndClipTests(unittest.TestCase):
    def test_total_counts_each_minute_once(self):
        self.assertEqual(total([iv("09:00-12:00"), iv("10:00-11:00")]), 180)
        self.assertEqual(total([iv("09:00-10:00"), iv("10:00-10:30")]), 90)
        self.assertEqual(total([iv("09:00-10:00"), iv("09:30-11:00"), iv("10:15-10:20")]), 120)

    def test_clip(self):
        self.assertEqual(clip([Interval(480, 560), Interval(1000, 1100), Interval(400, 500)], 540, 1020),
                         [Interval(540, 560), Interval(1000, 1020)])


class FreeSlotTests(unittest.TestCase):
    def test_nested_meetings(self):
        self.assertEqual(ranges(free_slots([iv("09:00-12:00"), iv("10:00-11:00")])), ["12:00-17:00"])
        self.assertEqual(ranges(free_slots([iv("09:30-16:00"), iv("10:00-11:00"), iv("13:00-14:00")])),
                         ["09:00-09:30", "16:00-17:00"])

    def test_back_to_back(self):
        self.assertEqual(ranges(free_slots([iv("09:00-10:00"), iv("10:00-11:00"), iv("11:00-11:30")])), ["11:30-17:00"])

    def test_outside_the_day_and_whole_day(self):
        self.assertEqual(ranges(free_slots([iv("08:00-09:20")])), ["09:20-17:00"])
        self.assertEqual(free_slots([iv("08:00-18:00"), iv("10:00-11:00")]), [])

    def test_min_length_boundary(self):
        self.assertEqual(ranges(free_slots([iv("09:00-10:00"), iv("10:30-17:00")])), ["10:00-10:30"])
        self.assertEqual(ranges(free_slots([iv("09:00-10:00"), iv("10:29-17:00")])), [])
        self.assertEqual(ranges(free_slots([], day_start=parse("10:00"), day_end=parse("11:00"), min_length=60)),
                         ["10:00-11:00"])

    def test_first_free_after_nested(self):
        self.assertEqual(fmt_range(first_free([iv("09:00-12:00"), iv("10:00-11:00")], 60)), "12:00-13:00")
        self.assertIsNone(first_free([iv("09:00-16:30"), iv("12:00-13:00")], 60))


class SummaryTests(unittest.TestCase):
    def test_mixed(self):
        s = busy_summary([iv("09:00-12:00"), iv("10:00-11:00"), iv("12:00-12:30"), iv("13:20-13:50")])
        self.assertEqual(s, {"blocks": 2, "busy_minutes": 240, "longest_block": 210})

    def test_back_to_back(self):
        s = busy_summary([iv("09:00-10:00"), iv("10:00-10:30"), iv("10:30-11:00")])
        self.assertEqual(s, {"blocks": 1, "busy_minutes": 120, "longest_block": 120})

    def test_generator_input(self):
        s = busy_summary(i for i in [iv("09:00-10:00"), iv("09:30-10:30")])
        self.assertEqual(s, {"blocks": 1, "busy_minutes": 90, "longest_block": 90})


class CalendarTests(unittest.TestCase):
    def test_busy_is_merged(self):
        cal = Calendar()
        for when in ("09:00-12:00", "10:00-11:00", "12:00-12:30", "14:00-15:00"):
            cal.add("dana", when)
        self.assertEqual(ranges(cal.busy("dana")), ["09:00-12:30", "14:00-15:00"])
        self.assertEqual(len(cal.events("dana")), 4)

    def test_common_free_across_people(self):
        cal = Calendar()
        cal.add("dana", "09:00-12:00", "all-hands")
        cal.add("dana", "10:00-11:00", "1:1")
        cal.add("omar", "12:00-13:00", "lunch talk")
        cal.add("omar", "13:00-13:30", "planning")
        cal.add("lee", "15:00-16:00")
        self.assertEqual(ranges(cal.common_free(["dana", "omar", "lee"])), ["13:30-15:00", "16:00-17:00"])
        self.assertEqual(ranges(cal.common_free(["nobody"])), ["09:00-17:00"])


class TimeFormatTests(unittest.TestCase):
    def test_unchanged(self):
        self.assertEqual(parse("24:00"), 1440)
        for bad in ("24:30", "7:00", "12:60"):
            with self.assertRaises(ValueError):
                parse(bad)
        with self.assertRaises(ValueError):
            parse_range("10:00-09:00")
