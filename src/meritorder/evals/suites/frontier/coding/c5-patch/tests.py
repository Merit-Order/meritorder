import difflib
import random

from solution import PatchError, apply_patch


def _err(original, diff, reverse=False):
    """The hunk number of the PatchError apply_patch raises, or 'no error'."""
    try:
        apply_patch(original, diff, reverse=reverse)
    except PatchError as e:
        assert isinstance(e, Exception)
        return e.hunk
    return "no error"


def _split_lf(text):
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + ([parts[-1]] if parts[-1] else [])


def _make_diff(a, b, n):
    """A unified diff of a -> b with n lines of context, with no-newline markers."""
    al, bl = _split_lf(a), _split_lf(b)
    out = ["--- a/file\n", "+++ b/file\n"]
    for group in difflib.SequenceMatcher(None, al, bl, autojunk=False).get_grouped_opcodes(n):
        i1, i2, j1, j2 = group[0][1], group[-1][2], group[0][3], group[-1][4]

        def rng(s, e):
            return f"{s + 1 if e > s else s},{e - s}"

        out.append(f"@@ -{rng(i1, i2)} +{rng(j1, j2)} @@\n")

        def emit(prefix, line):
            out.append(prefix + line.rstrip("\n") + "\n")
            if not line.endswith("\n"):
                out.append("\\ No newline at end of file\n")

        for tag, x1, x2, y1, y2 in group:
            if tag == "equal":
                for ln in al[x1:x2]:
                    emit(" ", ln)
            else:
                for ln in al[x1:x2]:
                    emit("-", ln)
                for ln in bl[y1:y2]:
                    emit("+", ln)
    return "".join(out)


EXAMPLE = """--- a/f.txt
+++ b/f.txt
@@ -2,3 +2,3 @@
 b
-c
+C
 d
"""


def test_example_basic():
    assert apply_patch("a\nb\nc\nd\ne\n", EXAMPLE) == ("a\nb\nC\nd\ne\n", [0])


def test_example_offset():
    assert apply_patch("new\na\nb\nc\nd\ne\n", EXAMPLE) == ("new\na\nb\nC\nd\ne\n", [1])


def test_example_reverse():
    assert apply_patch("a\nb\nC\nd\ne\n", EXAMPLE, reverse=True) == ("a\nb\nc\nd\ne\n", [0])


def test_pure_insertion_after_line():
    assert apply_patch("one\ntwo\n", "@@ -1,0 +2,2 @@\n+x\n+y\n") == ("one\nx\ny\ntwo\n", [0])


def test_insertion_at_start():
    assert apply_patch("a\nb\n", "@@ -0,0 +1,2 @@\n+x\n+y\n") == ("x\ny\na\nb\n", [0])


def test_create_from_empty_diff_without_final_newline():
    assert apply_patch("", "--- /dev/null\n+++ b/new\n@@ -0,0 +1,2 @@\n+x\n+y") == ("x\ny\n", [0])


def test_delete_everything():
    assert apply_patch("a\nb\n", "@@ -1,2 +0,0 @@\n-a\n-b\n") == ("", [0])


def test_omitted_counts_mean_one():
    assert apply_patch("a\nb\nc\n", "@@ -2 +2,2 @@\n-b\n+B\n+B2\n") == ("a\nB\nB2\nc\n", [0])


def test_header_trailing_text_allowed():
    diff = "@@ -1,2 +1,2 @@ def main():\n-a\n+A\n b\n"
    assert apply_patch("a\nb\nc\n", diff) == ("A\nb\nc\n", [0])


def test_hunks_use_original_numbering():
    original = "".join(f"L{i}\n" for i in range(1, 11))
    diff = "@@ -1,2 +1,4 @@\n L1\n+n1\n+n2\n L2\n@@ -8,3 +10,2 @@\n L8\n-L9\n L10\n"
    want = "L1\nn1\nn2\n" + "".join(f"L{i}\n" for i in range(2, 9)) + "L10\n"
    assert apply_patch(original, diff) == (want, [0, 0])


def test_drift_forward_all_hunks():
    original = "".join(f"L{i}\n" for i in range(1, 11))
    diff = "@@ -1,2 +1,4 @@\n L1\n+n1\n+n2\n L2\n@@ -8,3 +10,2 @@\n L8\n-L9\n L10\n"
    drifted = "p1\np2\np3\n" + original
    want = "p1\np2\np3\nL1\nn1\nn2\n" + "".join(f"L{i}\n" for i in range(2, 9)) + "L10\n"
    assert apply_patch(drifted, diff) == (want, [3, 3])


def test_drift_backward():
    original = "".join(f"L{i}\n" for i in range(3, 11))  # L1 and L2 are gone
    diff = "@@ -4,3 +4,3 @@\n L4\n-L5\n+X\n L6\n@@ -8,3 +8,3 @@\n L8\n-L9\n+Y\n L10\n"
    want = "L3\nL4\nX\nL6\nL7\nL8\nY\nL10\n"
    assert apply_patch(original, diff) == (want, [-2, -2])


def test_offset_is_carried_forward():
    # The file grew by 3 lines above hunk 1, and a copy of hunk 2's context now sits at hunk 2's nominal position.
    diff = ("@@ -2,3 +2,3 @@\n A\n-B\n+B2\n C\n"
            "@@ -9,3 +9,3 @@\n }\n-x\n+y\n }\n")
    original = "head\nn1\nn2\nn3\nA\nB\nC\ng1\n}\nx\n}\n}\nx\n}\ntail\n"
    want = "head\nn1\nn2\nn3\nA\nB2\nC\ng1\n}\nx\n}\n}\ny\n}\ntail\n"
    assert apply_patch(original, diff) == (want, [3, 3])


def test_equal_distance_prefers_later_position():
    assert apply_patch("m\nq\nm\n", "@@ -2 +2 @@\n-m\n+M\n") == ("m\nq\nM\n", [1])


def test_search_continues_past_nearer_mismatches():
    original = "x\nx\nx\nx\nx\nk\nx\n"
    assert apply_patch(original, "@@ -1 +1 @@\n-k\n+K\n") == ("x\nx\nx\nx\nx\nK\nx\n", [5])


def test_never_overlaps_previous_hunk():
    diff = "@@ -1,2 +1,2 @@\n a\n-k\n+K\n@@ -3,2 +3,2 @@\n-k\n+K\n x\n"
    assert apply_patch("a\nk\nx\nq\nk\nx\n", diff) == ("a\nK\nx\nq\nK\nx\n", [0, 2])


def test_three_hunks_mixed_offsets():
    old = [f"l{i}\n" for i in range(1, 21)]
    diff = ("@@ -2,3 +2,3 @@\n l2\n-l3\n+L3\n l4\n"
            "@@ -9,3 +9,3 @@\n l9\n-l10\n+L10\n l11\n"
            "@@ -16,3 +16,3 @@\n l16\n-l17\n+L17\n l18\n")
    cur = ["x1\n", "x2\n"] + old[:12] + old[13:]  # 2 lines added at the top, l13 deleted
    want = ["x1\n", "x2\n"] + old[:12] + old[13:]
    for k, v in {"l3\n": "L3\n", "l10\n": "L10\n", "l17\n": "L17\n"}.items():
        want[want.index(k)] = v
    assert apply_patch("".join(cur), diff) == ("".join(want), [2, 2, 1])


def test_hunk_that_does_not_apply():
    diff = "@@ -1 +1 @@\n-a\n+A\n@@ -3 +3 @@\n-zzz\n+Z\n@@ -5 +5 @@\n-e\n+E\n"
    assert _err("a\nb\nc\nd\ne\n", diff) == 2
    assert _err("a\n", "@@ -1 +1 @@\n-z\n+y\n") == 1


def test_last_line_without_newline_is_kept():
    assert apply_patch("a\nb\nc", "@@ -1,2 +1,2 @@\n-a\n+A\n b\n") == ("A\nb\nc", [0])


def test_marker_on_new_side_removes_newline():
    diff = "@@ -2 +2 @@\n-b\n+b\n\\ No newline at end of file\n"
    assert apply_patch("a\nb\n", diff) == ("a\nb", [0])


def test_marker_on_old_side_adds_newline():
    diff = "@@ -2 +2 @@\n-b\n\\ No newline at end of file\n+b\n"
    assert apply_patch("a\nb", diff) == ("a\nb\n", [0])


def test_marker_on_context_line():
    diff = "@@ -1,2 +1,2 @@\n-a\n+A\n b\n\\ No newline at end of file\n"
    assert apply_patch("a\nb", diff) == ("A\nb", [0])


def test_marker_text_is_free():
    diff = "@@ -1,2 +1,2 @@\n-a\n+A\n b\n\\ Kein Zeilenumbruch am Dateiende."
    assert apply_patch("a\nb", diff) == ("A\nb", [0])


def test_newline_mismatch_does_not_match():
    assert _err("a\nb", "@@ -1,2 +1,2 @@\n-a\n+A\n b\n") == 1
    assert _err("a\nb\n", "@@ -1,2 +1,2 @@\n-a\n+A\n b\n\\ No newline at end of file\n") == 1


def test_markers_in_the_middle_of_a_body():
    diff = "@@ -1,2 +1,3 @@\n x\n-y\n\\ No newline at end of file\n+y\n+z\n\\ No newline at end of file\n"
    assert apply_patch("x\ny", diff) == ("x\ny\nz", [0])


def test_misplaced_markers():
    assert _err("a\n", "@@ -1 +1 @@\n\\ No newline at end of file\n-a\n+b\n") == 1
    assert _err("a", "@@ -1 +1 @@\n-a\n\\ No newline\n\\ No newline\n+b\n") == 1
    assert _err("a\nb\n", "@@ -1 +1 @@\n-a\n+A\n@@ -2 +2 @@\n-b\n+B\n\\ x\n\\ y\n") == 2


def test_marker_in_preamble_is_ignored():
    assert apply_patch("a\n", "\\ just a note\n--- a\n+++ b\n@@ -1 +1 @@\n-a\n+b\n") == ("b\n", [0])


def test_empty_diff_line_is_empty_context():
    diff = "@@ -1,3 +1,3 @@\n-a\n+A\n\n b\n"
    assert apply_patch("a\n\nb\n", diff) == ("A\n\nb\n", [0])


def test_body_lines_that_look_like_headers():
    diff = "--- a/f\n+++ b/f\n@@ -1,2 +1,2 @@\n--- old\n+++ new\n mid\n"
    assert apply_patch("-- old\nmid\n", diff) == ("++ new\nmid\n", [0])


def test_git_preamble_ignored():
    diff = ("diff --git a/f b/f\nindex 83db48f..bf269f4 100644\nold mode 100644\n--- a/f\n+++ b/f\n"
            "@@ -1,3 +1,3 @@\n one\n-two\n+TWO\n three\n")
    assert apply_patch("one\ntwo\nthree\n", diff) == ("one\nTWO\nthree\n", [0])


def test_no_hunks():
    assert _err("a\n", "") == 0
    assert _err("a\n", "--- a/f\n+++ b/f\n -a\n") == 0


def test_malformed_headers():
    body = "\n-a\n+b\n"
    for header in ["@@ -1 +1@@", "@@ -1 +1", "@@  -1 +1 @@", "@@ -1, +1 @@", "@@ -x +1 @@", "@@ -1 -1 @@",
                   "@@@ -1 +1 @@@", "@@ -1,1 +1,1 @ @", "@@-1 +1 @@", "@@ -1  +1 @@", "@@ +1 -1 @@", "@@"]:
        assert _err("a\n", header + body) == 1, header


def test_malformed_second_header():
    diff = "@@ -1 +1 @@\n-a\n+A\n@@ -3,1 +3,1@@\n-c\n+C\n"
    assert _err("a\nb\nc\n", diff) == 2


def test_nonempty_range_at_line_zero():
    assert _err("a\n", "@@ -0,1 +1 @@\n-a\n+b\n") == 1
    assert _err("a\n", "@@ -1 +0,1 @@\n-a\n+b\n") == 1


def test_body_ends_early():
    assert _err("a\nb\nc\n", "@@ -1,3 +1,3 @@\n a\n-b\n+B\n") == 1
    assert _err("a\nb\n", "@@ -1,2 +1,2 @@\n-a\n+A\n@@ -2 +2 @@\n-b\n+B\n") == 1


def test_body_longer_than_header():
    assert _err("a\nb\n", "@@ -1,2 +1,1 @@\n+x\n+y\n-a\n-b\n") == 1
    assert _err("a\nb\n", "@@ -1 +1 @@\n a\n-b\n+B\n") == 1


def test_unexpected_line_in_body():
    assert _err("a\nb\n", "@@ -1,2 +1,2 @@\n a\nb\n") == 1
    assert _err("a\nb\n", "@@ -1,2 +1,2 @@\n a\n@@ -2 +2 @@\n-b\n+B\n") == 1


def test_lines_after_a_hunk():
    two = "@@ -1 +1 @@\n-a\n+A\n@@ -3 +3 @@\n-c\n+C\n"
    assert apply_patch("a\nb\nc\n", two + "\n\n\n") == ("A\nb\nC\n", [0, 0])
    assert _err("a\nb\nc\n", two + "trailing junk\n") == 2
    assert _err("a\nb\nc\n", two + "\n\n-x\n") == 2
    assert _err("a\nb\nc\n", "@@ -1 +1 @@\n-a\n+A\n\n@@ -3 +3 @@\n-c\n+C\n") == 1


def test_hunks_out_of_order_or_overlapping():
    assert _err("a\nb\nc\nd\ne\n", "@@ -5 +5 @@\n-e\n+E\n@@ -2 +2 @@\n-b\n+B\n") == 2
    assert _err("a\nb\nc\nd\ne\n", "@@ -1,3 +1,3 @@\n a\n-b\n+B\n c\n@@ -3,2 +3,2 @@\n c\n-d\n+D\n") == 2
    adjacent = "@@ -1,2 +1,2 @@\n a\n-b\n+B\n@@ -3,1 +3,1 @@\n-c\n+C\n"
    assert apply_patch("a\nb\nc\nd\n", adjacent) == ("a\nB\nC\nd\n", [0, 0])


def test_whole_diff_is_checked_before_applying():
    assert _err("q\nq\nq\n", "@@ -1 +1 @@\n-a\n+A\n@@ -2 +2 @@\n-b\n") == 2
    assert _err("q\n", "@@ -1 +1 @@\n-a\n+A\n@@ -2 +2 @@\n-b\n+B\n@@ -1 +1 @@\n-c\n+C\n") == 3


def test_reverse_with_drift_uses_new_ranges():
    diff = "@@ -1,2 +1,4 @@\n L1\n+n1\n+n2\n L2\n@@ -8,3 +10,2 @@\n L8\n-L9\n L10\n"
    new = "L1\nn1\nn2\n" + "".join(f"L{i}\n" for i in range(2, 9)) + "L10\n"
    old = "".join(f"L{i}\n" for i in range(1, 11))
    assert apply_patch("top\n" + new, diff, reverse=True) == ("top\n" + old, [1, 1])


def test_reverse_marker():
    diff = "@@ -2 +2 @@\n-b\n+b\n\\ No newline at end of file\n"
    assert apply_patch("a\nb", diff, reverse=True) == ("a\nb\n", [0])
    assert _err("a\nb\n", diff, reverse=True) == 1


def test_reverse_order_check_uses_source_side():
    diff = "@@ -1 +1,3 @@\n-a\n+x\n+y\n+z\n@@ -2 +2 @@\n-b\n+B\n"
    assert apply_patch("a\nb\n", diff) == ("x\ny\nz\nB\n", [0, 0])
    assert _err("x\ny\nz\nB\n", diff, reverse=True) == 2


def test_reverse_pure_insertion_becomes_deletion():
    assert apply_patch("one\nx\ny\ntwo\n", "@@ -1,0 +2,2 @@\n+x\n+y\n", reverse=True) == ("one\ntwo\n", [0])


def test_carriage_return_is_ordinary():
    assert apply_patch("a\r\nb\r\n", "@@ -1 +1 @@\n-a\r\n+A\r\n") == ("A\r\nb\r\n", [0])
    assert _err("a\r\nb\r\n", "@@ -1 +1 @@\n-a\n+A\n") == 1


def test_empty_source_past_the_end_lands_at_nearest_position():
    assert apply_patch("a\nb\n", "@@ -5,0 +6 @@\n+z\n") == ("a\nb\nz\n", [-3])


def test_roundtrip_generated_diffs():
    rnd = random.Random(20261110)
    for _ in range(250):
        def text():
            s = "".join(rnd.choice("abcde") + "\n" for _ in range(rnd.randint(0, 12)))
            return s[:-1] if s and rnd.random() < 0.3 else s
        a, b = text(), text()
        diff = _make_diff(a, b, rnd.randint(0, 3))
        if "@@" not in diff:
            continue
        got, offsets = apply_patch(a, diff)
        assert got == b, (a, b, diff)
        assert offsets == [0] * len(offsets)
        back, _ = apply_patch(b, diff, reverse=True)
        assert back == a, (a, b, diff)


def test_generated_diffs_on_a_shifted_file():
    rnd = random.Random(7)
    for _ in range(150):
        base = [f"line {i}\n" for i in range(rnd.randint(1, 25))]
        changed = list(base)
        for _ in range(rnd.randint(1, 4)):
            p = rnd.randrange(len(changed) + 1)
            if rnd.random() < 0.4 and p < len(changed):
                del changed[p]
            else:
                changed.insert(p, f"new {rnd.randint(0, 999)}\n")
        diff = _make_diff("".join(base), "".join(changed), rnd.randint(1, 3))
        if "@@" not in diff:
            continue
        shift = rnd.randint(0, 6)
        prefix = "".join(f"extra {i}\n" for i in range(shift))
        got, offsets = apply_patch(prefix + "".join(base), diff)
        assert got == prefix + "".join(changed)
        assert offsets == [shift] * len(offsets)
