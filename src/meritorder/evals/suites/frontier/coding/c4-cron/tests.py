from datetime import datetime as D
from datetime import timezone

from solution import next_run, next_runs


def _raises(expr, after=D(2024, 1, 1)):
    try:
        next_run(expr, after)
    except ValueError:
        return True
    return False


def test_every_minute():
    assert next_run("* * * * *", D(2024, 1, 1, 10, 0)) == D(2024, 1, 1, 10, 1)
    assert next_run("* * * * *", D(2023, 12, 31, 23, 59)) == D(2024, 1, 1, 0, 0)


def test_seconds_and_microseconds_round_up():
    assert next_run("* * * * *", D(2024, 1, 1, 10, 0, 30)) == D(2024, 1, 1, 10, 1)
    assert next_run("* * * * *", D(2024, 1, 1, 9, 59, 59, 999999)) == D(2024, 1, 1, 10, 0)
    assert next_run("30 10 * * *", D(2024, 1, 1, 10, 30, 0, 1)) == D(2024, 1, 2, 10, 30)
    assert next_run("30 10 * * *", D(2024, 1, 1, 10, 29, 59)) == D(2024, 1, 1, 10, 30)


def test_strictly_after():
    assert next_run("30 10 * * *", D(2024, 1, 1, 10, 30)) == D(2024, 1, 2, 10, 30)
    assert next_run("0 0 * * *", D(2024, 1, 1)) == D(2024, 1, 2)


def test_star_step():
    assert next_run("*/15 * * * *", D(2024, 1, 1, 10, 7)) == D(2024, 1, 1, 10, 15)
    assert next_run("*/15 * * * *", D(2024, 1, 1, 10, 45)) == D(2024, 1, 1, 11, 0)
    assert next_run("*/15 * * * *", D(2024, 1, 1, 10, 59, 30)) == D(2024, 1, 1, 11, 0)


def test_range_step():
    got = next_runs("10-50/20 * * * *", D(2024, 1, 1, 10, 0), 4)
    assert got == [D(2024, 1, 1, 10, 10), D(2024, 1, 1, 10, 30), D(2024, 1, 1, 10, 50), D(2024, 1, 1, 11, 10)]


def test_start_step_runs_to_field_maximum():
    got = next_runs("5/15 * * * *", D(2024, 1, 1, 10, 0), 5)
    assert [t.minute for t in got] == [5, 20, 35, 50, 5]
    assert got[-1] == D(2024, 1, 1, 11, 5)
    assert next_run("0 20/2 * * *", D(2024, 1, 1, 22, 0)) == D(2024, 1, 2, 20, 0)


def test_lists():
    got = next_runs("0,30 9,17 * * *", D(2024, 1, 1, 8, 0), 5)
    assert got == [D(2024, 1, 1, 9, 0), D(2024, 1, 1, 9, 30), D(2024, 1, 1, 17, 0), D(2024, 1, 1, 17, 30), D(2024, 1, 2, 9, 0)]


def test_lists_of_ranges_and_steps():
    got = next_runs("0 1-2,22-23/1,12 * * *", D(2024, 1, 1, 0, 0), 6)
    assert [t.hour for t in got] == [1, 2, 12, 22, 23, 1]


def test_rollover_to_next_year():
    assert next_run("0 0 * * *", D(2023, 12, 31, 23, 59)) == D(2024, 1, 1, 0, 0)
    assert next_run("59 23 31 12 *", D(2023, 12, 31, 23, 59)) == D(2024, 12, 31, 23, 59)


def test_month_names_case_insensitive():
    got = next_runs("0 12 1 jan,Jul *", D(2024, 1, 1, 12, 0), 3)
    assert got == [D(2024, 7, 1, 12, 0), D(2025, 1, 1, 12, 0), D(2025, 7, 1, 12, 0)]


def test_month_name_range_with_step():
    got = next_runs("0 0 1 JAN-DEC/3 *", D(2024, 1, 1), 4)
    assert got == [D(2024, 4, 1), D(2024, 7, 1), D(2024, 10, 1), D(2025, 1, 1)]


def test_weekday_name_range():
    assert next_run("0 9 * * MON-FRI", D(2024, 3, 8, 9, 0)) == D(2024, 3, 11, 9, 0)
    assert next_run("0 9 * * MON-FRI", D(2024, 3, 8, 8, 59)) == D(2024, 3, 8, 9, 0)


def test_sunday_is_0_and_7():
    want = [D(2024, 1, 7), D(2024, 1, 14), D(2024, 1, 21)]
    assert next_runs("0 0 * * 0", D(2024, 1, 1), 3) == want
    assert next_runs("0 0 * * 7", D(2024, 1, 1), 3) == want
    assert next_runs("0 0 * * sun", D(2024, 1, 1), 3) == want


def test_weekday_range_up_to_7():
    want = [D(2024, 1, 5), D(2024, 1, 6), D(2024, 1, 7), D(2024, 1, 12)]
    assert next_runs("0 0 * * 5-7", D(2024, 1, 1), 4) == want
    assert next_runs("0 0 * * FRI-7", D(2024, 1, 1), 4) == want


def test_weekday_start_step_includes_7():
    # 1/3 is 1, 4, 7: Monday, Thursday, Sunday.
    got = next_runs("0 0 * * 1/3", D(2024, 1, 1), 4)
    assert got == [D(2024, 1, 4), D(2024, 1, 7), D(2024, 1, 8), D(2024, 1, 11)]


def test_vixie_or_when_both_day_fields_restricted():
    got = next_runs("0 0 13 * 5", D(2024, 1, 1), 5)
    assert got == [D(2024, 1, 5), D(2024, 1, 12), D(2024, 1, 13), D(2024, 1, 19), D(2024, 1, 26)]


def test_and_when_day_of_month_is_a_star_step():
    got = next_runs("0 0 */2 * 1", D(2024, 1, 1), 4)
    assert got == [D(2024, 1, 15), D(2024, 1, 29), D(2024, 2, 5), D(2024, 2, 19)]


def test_and_when_day_of_week_is_a_star_step():
    # Day 1 of the month AND a Sunday/Tuesday/Thursday/Saturday.
    assert next_runs("0 0 1 * */2", D(2024, 1, 1), 2) == [D(2024, 2, 1), D(2024, 6, 1)]


def test_full_range_is_not_starred():
    # 1-31 doesn't begin with '*', so the OR rule applies and every day matches.
    assert next_run("0 0 1-31 * 1", D(2024, 1, 1)) == D(2024, 1, 2)
    assert next_run("0 0 * * 1", D(2024, 1, 1)) == D(2024, 1, 8)


def test_or_rule_within_month_restriction():
    assert next_runs("0 0 15 3 MON", D(2024, 1, 1), 3) == [D(2024, 3, 4), D(2024, 3, 11), D(2024, 3, 15)]


def test_day_31_skips_short_months():
    got = next_runs("0 0 31 * *", D(2024, 1, 31), 4)
    assert got == [D(2024, 3, 31), D(2024, 5, 31), D(2024, 7, 31), D(2024, 8, 31)]


def test_leap_day_and_century_rule():
    assert next_run("0 0 29 2 *", D(2024, 3, 1)) == D(2028, 2, 29)
    assert next_run("0 0 29 2 *", D(2096, 3, 1)) == D(2104, 2, 29)
    assert next_runs("0 0 29 2 *", D(2024, 1, 1), 3) == [D(2024, 2, 29), D(2028, 2, 29), D(2032, 2, 29)]


def test_end_of_february():
    assert next_run("45 23 * * *", D(2024, 2, 28, 23, 50)) == D(2024, 2, 29, 23, 45)
    assert next_run("45 23 * * *", D(2023, 2, 28, 23, 50)) == D(2023, 3, 1, 23, 45)


def test_never_matching_raises():
    for expr in ["0 0 30 2 *", "0 0 31 4,6,9,11 *", "0 0 30-31 2 *", "0 0 31 2 */1", "* * 31 FEB,apr *"]:
        assert _raises(expr), expr


def test_or_rule_makes_february_30_possible():
    assert next_runs("0 0 30 2 1", D(2024, 1, 1), 3) == [D(2024, 2, 5), D(2024, 2, 12), D(2024, 2, 19)]


def test_wrong_number_of_fields():
    for expr in ["* * * *", "* * * * * *", "", "   ", "0 0 1 1 * 2024"]:
        assert _raises(expr), repr(expr)


def test_values_out_of_range():
    for expr in ["60 * * * *", "* 24 * * *", "* * 0 * *", "* * 32 * *", "* * * 0 *", "* * * 13 *",
                 "* * * * 8", "0-60 * * * *", "*/5 0-24 * * *", "1,2,99 * * * *"]:
        assert _raises(expr), expr


def test_reversed_ranges_and_zero_steps():
    for expr in ["10-5 * * * *", "*/0 * * * *", "1-5/0 * * * *", "5/0 * * * *", "* * * * SAT-SUN", "* * * DEC-JAN *"]:
        assert _raises(expr), expr


def test_bad_names():
    for expr in ["* * * * FUNDAY", "* * * * MONDAY", "MON * * * *", "* JAN * * *", "* * JAN * *",
                 "* * * SUN *", "* * * * JAN", "* * * JANUARY *", "*/MON * * * *"]:
        assert _raises(expr), expr


def test_malformed_items():
    for expr in ["1,,2 * * * *", "5, * * * *", ",5 * * * *", "-5 * * * *", "5- * * * *", "/5 * * * *",
                 "*/ * * * *", "** * * * *", "* * ? * *", "* * L * *", "* * * * 5#2", "* * 15W * *",
                 "*/2/3 * * * *", "+5 * * * *", "1.5 * * * *", "1_0 * * * *", "1-2-3 * * * *", "*-5 * * * *"]:
        assert _raises(expr), expr


def test_only_ascii_digits():
    assert _raises("\u0663 * * * *")  # ARABIC-INDIC DIGIT THREE
    assert _raises("0 \uff11 * * *")  # FULLWIDTH DIGIT ONE


def test_leading_zeros_allowed():
    assert next_run("05 08 * * *", D(2024, 1, 1)) == D(2024, 1, 1, 8, 5)
    assert next_run("0 0 01 01 *", D(2024, 1, 1)) == D(2025, 1, 1)


def test_whitespace_between_fields():
    assert next_run("  */30\t*  *\t\t* *  ", D(2024, 1, 1, 0, 10)) == D(2024, 1, 1, 0, 30)


def test_names_case_insensitive_and_starred_day_of_month():
    assert next_runs("0 0 * jAn sUn", D(2024, 1, 1), 2) == [D(2024, 1, 7), D(2024, 1, 14)]
    assert next_run("0 0 * jAn sUn", D(2024, 1, 28)) == D(2025, 1, 5)


def test_macros():
    assert next_runs("@hourly", D(2024, 1, 1, 23, 30), 2) == [D(2024, 1, 2, 0, 0), D(2024, 1, 2, 1, 0)]
    assert next_run("@daily", D(2024, 1, 1)) == D(2024, 1, 2)
    assert next_run("@midnight", D(2024, 1, 1, 12, 0)) == D(2024, 1, 2)
    assert next_run("@weekly", D(2024, 1, 1)) == D(2024, 1, 7)
    assert next_run("@monthly", D(2024, 1, 1)) == D(2024, 2, 1)
    assert next_run("@yearly", D(2024, 1, 1)) == D(2025, 1, 1)
    assert next_run("@annually", D(2024, 6, 1)) == D(2025, 1, 1)


def test_macros_case_insensitive_and_trimmed():
    assert next_run("@DAILY", D(2024, 1, 1)) == D(2024, 1, 2)
    assert next_run("  @Hourly\t", D(2024, 1, 1, 0, 0)) == D(2024, 1, 1, 1, 0)


def test_bad_macros():
    for expr in ["@reboot", "@every 5m", "@", "@daily *", "@ daily"]:
        assert _raises(expr), expr


def test_next_runs_counts():
    assert next_runs("* * * * *", D(2024, 1, 1), 0) == []
    try:
        next_runs("* * * * *", D(2024, 1, 1), -1)
    except ValueError:
        pass
    else:
        raise AssertionError("negative n must raise ValueError")
    runs = next_runs("7 */5 * * 2-4", D(2024, 1, 1), 6)
    t = D(2024, 1, 1)
    for r in runs:
        t = next_run("7 */5 * * 2-4", t)
        assert r == t


def test_step_larger_than_range():
    assert next_run("*/100 * * * *", D(2024, 1, 1, 10, 0)) == D(2024, 1, 1, 11, 0)
    assert next_run("0 */25 * * *", D(2024, 1, 1, 10, 0)) == D(2024, 1, 2, 0, 0)


def test_aware_datetime_rejected():
    for call in [lambda: next_run("* * * * *", D(2024, 1, 1, tzinfo=timezone.utc)),
                 lambda: next_runs("* * * * *", D(2024, 1, 1, tzinfo=timezone.utc), 1)]:
        try:
            call()
        except ValueError:
            continue
        raise AssertionError("aware datetimes must raise ValueError")


def test_many_runs_far_apart_are_fast():
    got = next_runs("0 0 29 2 1", D(2024, 1, 1), 3)  # OR rule: Mondays in February, and Feb 29
    assert got == [D(2024, 2, 5), D(2024, 2, 12), D(2024, 2, 19)]
    leap_sundays = next_runs("0 0 29 2 */7", D(2000, 1, 1), 2)  # AND: Feb 29 that is a Sunday
    assert leap_sundays == [D(2004, 2, 29), D(2032, 2, 29)]
