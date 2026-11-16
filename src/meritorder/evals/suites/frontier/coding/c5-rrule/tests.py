from datetime import datetime as dt

from solution import expand


def _days(rule, start, **kw):
    """Occurrences as 'YYYY-MM-DD' strings, after checking they all carry dtstart's time of day."""
    out = expand(rule, start, **kw)
    assert isinstance(out, list)
    assert all(type(d) is dt and d.time() == start.time() for d in out), out
    return [d.strftime("%Y-%m-%d") for d in out]


def _bad(rule, start=dt(2026, 1, 1, 9), **kw):
    try:
        expand(rule, start, **kw)
    except ValueError:
        return True
    return False


# ------------------------------------------------------------------ the prompt's examples

def test_daily_count():
    assert expand("FREQ=DAILY;COUNT=3", dt(2026, 3, 30, 9, 15)) == [
        dt(2026, 3, 30, 9, 15), dt(2026, 3, 31, 9, 15), dt(2026, 4, 1, 9, 15)]


def test_monthly_31st_skips_short_months():
    assert expand("FREQ=MONTHLY;COUNT=4", dt(2026, 1, 31, 8)) == [
        dt(2026, 1, 31, 8), dt(2026, 3, 31, 8), dt(2026, 5, 31, 8), dt(2026, 7, 31, 8)]


def test_weekly_interval_wkst_rfc_examples():
    start = dt(1997, 8, 5, 9)
    assert _days("FREQ=WEEKLY;INTERVAL=2;COUNT=4;BYDAY=TU,SU;WKST=MO", start) == [
        "1997-08-05", "1997-08-10", "1997-08-19", "1997-08-24"]
    assert _days("FREQ=WEEKLY;INTERVAL=2;COUNT=4;BYDAY=TU,SU;WKST=SU", start) == [
        "1997-08-05", "1997-08-17", "1997-08-19", "1997-08-31"]


def test_unbounded_needs_limit_or_between():
    assert _bad("FREQ=DAILY", dt(2026, 1, 1))
    assert expand("FREQ=DAILY", dt(2026, 1, 1), limit=2) == [dt(2026, 1, 1), dt(2026, 1, 2)]


# ------------------------------------------------------------------ time of day, dtstart, defaults

def test_time_of_day_comes_from_dtstart():
    start = dt(2026, 2, 27, 23, 59, 58, 123456)
    assert expand("FREQ=DAILY;COUNT=3", start) == [
        start, dt(2026, 2, 28, 23, 59, 58, 123456), dt(2026, 3, 1, 23, 59, 58, 123456)]


def test_dtstart_counts_only_if_it_matches():
    assert _days("FREQ=MONTHLY;BYMONTHDAY=15;COUNT=2", dt(2026, 1, 1, 9)) == ["2026-01-15", "2026-02-15"]


def test_weekly_defaults_to_dtstart_weekday():
    assert _days("FREQ=WEEKLY;COUNT=3", dt(2026, 11, 11, 14)) == ["2026-11-11", "2026-11-18", "2026-11-25"]


def test_yearly_default_skips_non_leap_years():
    assert _days("FREQ=YEARLY;COUNT=3", dt(2024, 2, 29, 10)) == ["2024-02-29", "2028-02-29", "2032-02-29"]


def test_yearly_bymonth_defaults_day_of_month():
    assert _days("FREQ=YEARLY;BYMONTH=6,7;COUNT=3", dt(2026, 3, 10, 9)) == ["2026-06-10", "2026-07-10", "2027-06-10"]


def test_yearly_bymonthday_alone_means_every_month():
    assert _days("FREQ=YEARLY;BYMONTHDAY=15;COUNT=4", dt(2026, 1, 20, 9)) == [
        "2026-02-15", "2026-03-15", "2026-04-15", "2026-05-15"]


def test_yearly_byday_alone_means_every_such_weekday():
    assert _days("FREQ=YEARLY;BYDAY=TH;COUNT=3", dt(2026, 12, 20, 9)) == ["2026-12-24", "2026-12-31", "2027-01-07"]


# ------------------------------------------------------------------ BYMONTHDAY

def test_negative_monthday_is_last_day():
    assert _days("FREQ=MONTHLY;BYMONTHDAY=-1;COUNT=4", dt(2027, 12, 15, 7, 30)) == [
        "2027-12-31", "2028-01-31", "2028-02-29", "2028-03-31"]


def test_negative_monthday_that_does_not_exist_is_skipped():
    assert _days("FREQ=MONTHLY;BYMONTHDAY=-30;COUNT=3", dt(2026, 1, 1, 12)) == ["2026-01-02", "2026-03-02", "2026-04-01"]


def test_monthly_interval_skips_missing_days():
    assert _days("FREQ=MONTHLY;INTERVAL=3;BYMONTHDAY=31;COUNT=3", dt(2026, 1, 31, 9)) == [
        "2026-01-31", "2026-07-31", "2026-10-31"]


def test_repeated_monthdays_produce_one_occurrence():
    assert _days("FREQ=MONTHLY;BYMONTHDAY=1,-31,1;COUNT=3", dt(2026, 1, 1, 9)) == ["2026-01-01", "2026-02-01", "2026-03-01"]


# ------------------------------------------------------------------ BYDAY

def test_monthly_last_friday():
    assert expand("FREQ=MONTHLY;BYDAY=-1FR;COUNT=2", dt(2026, 1, 1)) == [dt(2026, 1, 30), dt(2026, 2, 27)]


def test_monthly_second_tuesday_with_plus_sign():
    assert _days("FREQ=MONTHLY;BYDAY=+2TU;COUNT=3", dt(2026, 1, 1, 18)) == ["2026-01-13", "2026-02-10", "2026-03-10"]


def test_monthly_fifth_monday_only_when_it_exists():
    assert _days("FREQ=MONTHLY;BYDAY=5MO;COUNT=4", dt(2026, 1, 1, 8)) == [
        "2026-03-30", "2026-06-29", "2026-08-31", "2026-11-30"]


def test_yearly_ordinal_counts_within_the_year():
    assert _days("FREQ=YEARLY;BYDAY=20MO;COUNT=3", dt(2026, 1, 1, 9)) == ["2026-05-18", "2027-05-17", "2028-05-15"]
    assert _days("FREQ=YEARLY;BYDAY=-1SU;COUNT=3", dt(2026, 1, 1, 9)) == ["2026-12-27", "2027-12-26", "2028-12-31"]


def test_yearly_53rd_weekday_only_in_some_years():
    assert _days("FREQ=YEARLY;BYDAY=53FR;COUNT=2", dt(2026, 1, 1, 9)) == ["2027-12-31", "2032-12-31"]


def test_yearly_ordinal_with_bymonth_counts_within_the_month():
    assert _days("FREQ=YEARLY;BYMONTH=3,11;BYDAY=2SU;COUNT=4", dt(2026, 1, 1, 2)) == [
        "2026-03-08", "2026-11-08", "2027-03-14", "2027-11-14"]
    assert _days("FREQ=YEARLY;BYMONTH=5;BYDAY=-1MO;COUNT=3", dt(2026, 1, 1, 9)) == ["2026-05-25", "2027-05-31", "2028-05-29"]


def test_yearly_bymonth_with_plain_byday():
    assert _days("FREQ=YEARLY;BYMONTH=2;BYDAY=MO;COUNT=5", dt(2026, 1, 1, 9)) == [
        "2026-02-02", "2026-02-09", "2026-02-16", "2026-02-23", "2027-02-01"]


def test_byday_with_bymonthday_must_match_both():
    assert _days("FREQ=MONTHLY;BYDAY=FR;BYMONTHDAY=13;COUNT=4", dt(2026, 1, 1)) == [
        "2026-02-13", "2026-03-13", "2026-11-13", "2027-08-13"]
    assert expand("FREQ=YEARLY;BYDAY=FR;BYMONTHDAY=13", dt(2026, 1, 1), limit=3) == [
        dt(2026, 2, 13), dt(2026, 3, 13), dt(2026, 11, 13)]


def test_byday_items_are_alternatives():
    assert _days("FREQ=MONTHLY;BYDAY=1MO,FR;COUNT=6", dt(2026, 6, 1, 9)) == [
        "2026-06-01", "2026-06-05", "2026-06-12", "2026-06-19", "2026-06-26", "2026-07-03"]


def test_weekly_wkst_with_sunday_dtstart():
    start = dt(2026, 1, 4, 9)  # a Sunday
    assert _days("FREQ=WEEKLY;INTERVAL=2;BYDAY=MO,SU;COUNT=4;WKST=MO", start) == [
        "2026-01-04", "2026-01-12", "2026-01-18", "2026-01-26"]
    assert _days("FREQ=WEEKLY;INTERVAL=2;BYDAY=MO,SU;COUNT=4;WKST=SU", start) == [
        "2026-01-04", "2026-01-05", "2026-01-18", "2026-01-19"]


# ------------------------------------------------------------------ BYMONTH as a limit, INTERVAL

def test_daily_with_month_and_monthday_filters():
    assert _days("FREQ=DAILY;BYMONTH=2;BYMONTHDAY=-1;COUNT=3", dt(2026, 1, 1, 6)) == ["2026-02-28", "2027-02-28", "2028-02-29"]


def test_weekly_limited_by_bymonth():
    assert _days("FREQ=WEEKLY;BYMONTH=1;COUNT=6", dt(2026, 1, 20, 9)) == [
        "2026-01-20", "2026-01-27", "2027-01-05", "2027-01-12", "2027-01-19", "2027-01-26"]


def test_monthly_interval_limited_by_bymonth():
    assert _days("FREQ=MONTHLY;INTERVAL=2;BYMONTH=1,2,3,4;BYMONTHDAY=5;COUNT=4", dt(2026, 1, 5, 9)) == [
        "2026-01-05", "2026-03-05", "2027-01-05", "2027-03-05"]


def test_yearly_interval():
    assert _days("FREQ=YEARLY;INTERVAL=4;BYMONTH=2;BYMONTHDAY=29;COUNT=2", dt(2020, 1, 1)) == ["2020-02-29", "2024-02-29"]


# ------------------------------------------------------------------ BYSETPOS

def test_bysetpos_last_weekday_of_month():
    assert _days("FREQ=MONTHLY;BYDAY=MO,TU,WE,TH,FR;BYSETPOS=-1;COUNT=3", dt(2026, 1, 1, 17)) == [
        "2026-01-30", "2026-02-27", "2026-03-31"]


def test_bysetpos_sees_days_before_dtstart():
    assert expand("FREQ=MONTHLY;BYDAY=MO,TU,WE,TH,FR;BYSETPOS=1;COUNT=3", dt(2026, 1, 10, 9)) == [
        dt(2026, 2, 2, 9), dt(2026, 3, 2, 9), dt(2026, 4, 1, 9)]


def test_bysetpos_multiple_positions_and_repeats():
    assert _days("FREQ=MONTHLY;BYMONTHDAY=20,1,10;BYSETPOS=-1,2,2;COUNT=4", dt(2026, 1, 15, 9)) == [
        "2026-01-20", "2026-02-10", "2026-02-20", "2026-03-10"]


def test_bysetpos_weekly_uses_the_whole_first_week():
    assert _days("FREQ=WEEKLY;BYDAY=TU,TH,FR;BYSETPOS=3;COUNT=3", dt(2026, 10, 21, 9)) == [
        "2026-10-23", "2026-10-30", "2026-11-06"]


def test_bysetpos_yearly_and_daily():
    assert _days("FREQ=YEARLY;BYDAY=MO;BYSETPOS=-1;COUNT=2", dt(2026, 1, 1, 9)) == ["2026-12-28", "2027-12-27"]
    assert _days("FREQ=DAILY;BYDAY=MO;BYSETPOS=-1;COUNT=2", dt(2026, 1, 1)) == ["2026-01-05", "2026-01-12"]


def test_bysetpos_with_date_until():
    assert _days("FREQ=YEARLY;BYMONTH=1,7;BYDAY=SA,SU;BYSETPOS=1,-1;UNTIL=20270101", dt(2026, 1, 1, 9)) == [
        "2026-01-03", "2026-07-26"]


# ------------------------------------------------------------------ UNTIL, between, limit

def test_until_datetime_cuts_by_instant():
    start = dt(2026, 1, 1, 9)
    assert _days("FREQ=DAILY;UNTIL=20260105T080000", start) == ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]
    assert _days("FREQ=DAILY;UNTIL=20260103T090000", start) == ["2026-01-01", "2026-01-02", "2026-01-03"]


def test_until_date_includes_the_whole_day():
    assert _days("FREQ=DAILY;UNTIL=20260105", dt(2026, 1, 1, 23, 30)) == [
        "2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"]


def test_until_before_dtstart():
    assert expand("FREQ=DAILY;UNTIL=20251231T235959", dt(2026, 1, 1)) == []


def test_between_does_not_reset_count():
    assert expand("FREQ=DAILY;COUNT=10", dt(2026, 1, 1), between=(dt(2026, 1, 8), dt(2026, 2, 1))) == [
        dt(2026, 1, 8), dt(2026, 1, 9), dt(2026, 1, 10)]
    assert expand("FREQ=DAILY;COUNT=3", dt(2026, 1, 1), between=(dt(2026, 1, 8), dt(2026, 2, 1))) == []


def test_between_on_an_unbounded_rule():
    assert _days("FREQ=WEEKLY;BYDAY=MO,FR", dt(2026, 1, 1, 9), between=(dt(2026, 3, 1), dt(2026, 3, 15, 23, 59))) == [
        "2026-03-02", "2026-03-06", "2026-03-09", "2026-03-13"]


def test_between_is_inclusive():
    start = dt(2026, 1, 1, 9)
    assert _days("FREQ=DAILY;INTERVAL=3", start, between=(dt(2026, 1, 4, 9), dt(2026, 1, 10, 9))) == [
        "2026-01-04", "2026-01-07", "2026-01-10"]
    assert _days("FREQ=DAILY;INTERVAL=3", start, between=(dt(2026, 1, 4, 9, 0, 1), dt(2026, 1, 10, 8, 59))) == ["2026-01-07"]
    assert expand("FREQ=DAILY;INTERVAL=3", start, between=(dt(2026, 2, 1), dt(2026, 1, 1))) == []


def test_limit_applies_after_between():
    assert _days("FREQ=DAILY;INTERVAL=3", dt(2026, 1, 1, 9), between=(dt(2026, 1, 4, 9), dt(2026, 2, 10, 9)), limit=2) == [
        "2026-01-04", "2026-01-07"]
    assert _days("FREQ=MONTHLY;COUNT=12", dt(2026, 1, 31, 9), limit=2) == ["2026-01-31", "2026-03-31"]


def test_prefix_and_part_order():
    assert _days("RRULE:COUNT=2;BYMONTHDAY=-1;FREQ=MONTHLY", dt(2026, 4, 1, 9)) == ["2026-04-30", "2026-05-31"]


# ------------------------------------------------------------------ errors

def test_rejects_bad_syntax():
    for rule in ["", "COUNT=3", "FREQ=HOURLY;COUNT=3", "FREQ=DAILY;COUNT=3;COUNT=4", "FREQ=DAILY;COUNT=",
                 "FREQ=DAILY;COUNT=2;FOO", "FREQ=DAILY;COUNT=2;BYHOUR=9", "FREQ=DAILY;COUNT=2;BYYEARDAY=1",
                 "FREQ=DAILY;COUNT=x", "FREQ=DAILY;COUNT=-2", "FREQ=MONTHLY;COUNT=2;BYMONTHDAY=1,,2",
                 "FREQ=MONTHLY;COUNT=2;BYDAY=XX", "FREQ=MONTHLY;COUNT=2;BYDAY=MO1", "FREQ=DAILY;COUNT=2;WKST=XX",
                 "FREQ=MONTHLY;COUNT=2;BYMONTH=+3"]:
        assert _bad(rule), rule


def test_rejects_out_of_range_values():
    for rule in ["FREQ=DAILY;COUNT=0", "FREQ=DAILY;INTERVAL=0;COUNT=1", "FREQ=MONTHLY;BYMONTH=13;COUNT=1",
                 "FREQ=MONTHLY;BYMONTH=0;COUNT=1", "FREQ=MONTHLY;BYMONTHDAY=0;COUNT=1",
                 "FREQ=MONTHLY;BYMONTHDAY=32;COUNT=1", "FREQ=MONTHLY;BYMONTHDAY=-32;COUNT=1",
                 "FREQ=MONTHLY;BYDAY=0MO;COUNT=1", "FREQ=YEARLY;BYDAY=54MO;COUNT=1", "FREQ=YEARLY;BYDAY=-54MO;COUNT=1",
                 "FREQ=MONTHLY;BYDAY=MO;BYSETPOS=0;COUNT=1", "FREQ=MONTHLY;BYDAY=MO;BYSETPOS=367;COUNT=1"]:
        assert _bad(rule), rule


def test_rejects_bad_until():
    for until in ["20260230", "20260101T250000", "20260101T090000Z", "2026-01-01", "202601011", "20260101T0900"]:
        assert _bad(f"FREQ=DAILY;UNTIL={until}"), until
    assert _bad("FREQ=DAILY;COUNT=2;UNTIL=20270101")


def test_rejects_invalid_combinations():
    assert _bad("FREQ=WEEKLY;BYMONTHDAY=1;COUNT=2")
    assert _bad("FREQ=WEEKLY;BYDAY=1MO;COUNT=2")
    assert _bad("FREQ=DAILY;BYDAY=-1FR;COUNT=2")
    assert _bad("FREQ=MONTHLY;BYSETPOS=1;COUNT=2")
    assert _bad("FREQ=WEEKLY;BYSETPOS=1;WKST=SU;COUNT=2")
    assert _bad("FREQ=DAILY;COUNT=2", limit=0)
    assert _days("FREQ=MONTHLY;BYMONTH=2;BYSETPOS=-1;COUNT=2", dt(2026, 1, 5, 9)) == ["2026-02-05", "2027-02-05"]
