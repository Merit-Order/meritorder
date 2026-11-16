import signal

from solution import ignored, match


class _TooSlow(BaseException):
    pass


def _fast(fn, *args, seconds=1.0):
    """Run fn(*args), failing if it takes longer than `seconds` (so a slow solution fails only this test)."""
    def on_alarm(signum, frame):
        raise _TooSlow(f"took more than {seconds}s")
    old = signal.signal(signal.SIGALRM, on_alarm)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return fn(*args)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def _raises(fn, *args):
    try:
        result = fn(*args)
    except ValueError:
        return True
    raise AssertionError(f"{fn.__name__}{args!r} returned {result!r} instead of raising ValueError")


def _all(cases):
    for pattern, path, want in cases:
        got = match(pattern, path)
        assert got is want, (pattern, path, got)


def test_literal_paths():
    _all([("a/b.txt", "a/b.txt", True), ("a/b.txt", "a/b.txT", False), ("a", "a/b", False), ("a/b", "a", False),
          ("my file", "my file", True)])


def test_star_within_segment():
    _all([("*.py", "x.py", True), ("*.py", "dir/x.py", False), ("a*", "a", True), ("a*b*c", "aXbYc", True),
          ("a*b*c", "acb", False), ("*", "a.b.c", True), ("a*", "ab/c", False), ("*/*", "a/b", True)])


def test_question_mark():
    _all([("?.txt", "a.txt", True), ("?.txt", "ab.txt", False), ("a?c", "abc", True), ("??", "é!", True),
          ("?", "ab", False)])


def test_globstar_zero_or_more():
    _all([("**/x.py", "x.py", True), ("**/x.py", "a/b/x.py", True), ("a/**/b", "a/b", True),
          ("a/**/b", "a/x/y/b", True), ("a/**/b", "a/x/y/c", False), ("a/**", "a", True), ("a/**", "a/b/c", True),
          ("a/**", "b", False), ("**", "x/y/z", True), ("src/**/*.py", "src/main.py", True)])


def test_globstar_only_as_whole_segment():
    _all([("a/**b", "a/xb", True), ("a/**b", "a/x/b", False), ("**.py", "x.py", True), ("**.py", "x/y.py", False),
          ("x**y", "xay", True), ("x**y", "xa/y", False)])


def test_multiple_globstars():
    _all([("**/a/**/b/**", "a/b", True), ("**/a/**/b/**", "x/a/y/b/z", True), ("**/a/**/b/**", "b/a", False),
          ("**/**", "q", True), ("a/**/**/b", "a/b", True), ("a/**/b/**/b", "a/b/b/b", True),
          ("a/**/b/**/b", "a/b", False)])


def test_dot_rule_wildcards():
    _all([("*", ".bashrc", False), ("?bashrc", ".bashrc", False), ("[.]bashrc", ".bashrc", False),
          ("[!a]bashrc", ".bashrc", False), ("*.py", ".py", False), (".*", ".bashrc", True),
          ("\\.bashrc", ".bashrc", True), (".bashrc", ".bashrc", True)])


def test_dot_rule_only_at_segment_start():
    _all([("a*", "a.b", True), ("a[.]b", "a.b", True), ("*.*", "a.b", True), ("a?b", "a.b", True),
          ("x/*.txt", "x/notes.txt", True)])


def test_dot_rule_globstar_and_inner_segments():
    _all([("**/x", ".git/x", False), ("**/x", "a/.git/x", False), (".git/**", ".git/a/b", True),
          ("a/**", "a/.b", False), ("a/*/c", "a/.b/c", False), ("**", ".x", False), ("a/.*/c", "a/.b/c", True),
          ("**/.env", "x/.env", True)])


def test_bracket_basics():
    _all([("[abc]", "b", True), ("[abc]", "d", False), ("[a-c]x", "bx", True), ("[a-c]x", "dx", False),
          ("[!a-c]x", "dx", True), ("[!a-c]x", "bx", False), ("[a-cx-z]", "y", True), ("[ab]", "ab", False)])


def test_bracket_close_bracket_literal():
    _all([("[]a]", "]", True), ("[]a]", "a", True), ("[]a]", "b", False), ("[!]]", "]", False),
          ("[!]]", "x", True), ("[!]a]", "a", False), ("[[]", "[", True)])


def test_bracket_dash_rules():
    _all([("[a-]", "-", True), ("[-a]", "-", True), ("[a-c-e]", "-", True), ("[a-c-e]", "d", False),
          ("[a-c-e]", "e", True), ("[!-a]", "-", False), ("[!-a]", "b", True)])


def test_bracket_caret_is_literal():
    _all([("[^a]", "^", True), ("[^a]", "a", True), ("[^a]", "b", False)])


def test_bracket_escapes():
    _all([("[\\]]", "]", True), ("[\\\\]", "\\", True), ("[\\-a]", "-", True), ("[a\\-z]", "b", False),
          ("[a\\-z]", "z", True), ("[\\!a]", "!", True), ("[\\!a]", "b", False), ("[a-\\z]", "m", True)])


def test_bracket_code_point_ranges():
    _all([("[α-ω]", "β", True), ("[α-ω]", "a", False), ("[]-a]", "_", True),
          ("[]-a]", "a", True), ("[]-a]", "b", False), ("[!]-a]", "^", False)])


def test_escapes_outside_brackets():
    _all([("\\*", "*", True), ("\\*", "a", False), ("\\?", "?", True), ("\\?", "x", False), ("a\\[b", "a[b", True),
          ("\\a", "a", True), ("\\\\x", "\\x", True), ("*", "a*b", True), ("x\\*y", "x*y", True)])


def test_case_and_unicode():
    _all([("A", "a", False), ("é", "é", True), ("?", "日", True), ("e", "é", False)])


def test_invalid_unclosed_bracket():
    for pattern in ["a[", "[abc", "[]", "[!]", "[!", "[a-"]:
        assert _raises(match, pattern, "a"), pattern


def test_invalid_backslash_and_range():
    assert _raises(match, "a\\", "a")
    assert _raises(match, "[a\\", "a")
    assert _raises(match, "[z-a]", "a")
    assert _raises(match, "a\\/b", "a/b")
    assert _raises(match, "[a/b]", "a/b")


def test_invalid_empty_segments():
    for pattern, path in [("", "a"), ("/a", "a"), ("a/", "a"), ("a//b", "a/b"), ("/", "a")]:
        assert _raises(match, pattern, path), pattern


def test_validation_before_matching():
    assert _raises(match, "a/[", "b/c")
    assert _raises(match, "x/[z-a]", "y")
    assert _raises(match, "nope/**/x\\", "a")


def test_speed_many_stars():
    assert _fast(match, "*a" * 12 + "*b", "a" * 280) is False
    assert _fast(match, "*a*b*a*b*a*b*a*b*", "ab" * 140) is True
    assert _fast(match, "*?*?*?*?*?*?*?*?*?*?*!", "x" * 299) is False


def test_speed_many_globstars():
    deep = "/".join(["a"] * 140)
    assert _fast(match, "**/" * 15 + "z", deep) is False
    assert _fast(match, "**/a/" * 10 + "b", "/".join(["a"] * 100) + "/c") is False
    assert _fast(match, "**/" * 15 + "a", deep) is True


def test_speed_mixed():
    path = "/".join(["aaaaaaaaaaaaaaaaaaa"] * 15)
    assert _fast(match, "**/*a*a*a*a*a*a*/**/*a*b/**", path) is False
    assert _fast(ignored, ["**/*a*a*a*a*a*a*a*b", "*a*a*a*a*a*a*a*a*a*b"], path) is False


def test_ignored_unanchored():
    assert ignored(["*.log"], "x.log") is True
    assert ignored(["*.log"], "dir/x.log") is True
    assert ignored(["*.log"], "x.txt") is False
    assert ignored(["build"], "src/build/x.o") is True
    assert ignored(["node_modules"], "a/node_modules/b/c.js") is True


def test_ignored_last_matching_rule_wins():
    assert ignored(["*.log", "!keep.log"], "keep.log") is False
    assert ignored(["*.log", "!keep.log"], "logs/keep.log") is False
    assert ignored(["*.log", "!keep.log"], "b.log") is True
    assert ignored(["!keep.log", "*.log"], "keep.log") is True
    assert ignored(["*.log", "!keep.log", "logs"], "logs/keep.log") is True


def test_ignored_anchored_by_leading_slash():
    assert ignored(["/build"], "build") is True
    assert ignored(["/build"], "build/x.o") is True
    assert ignored(["/build"], "src/build") is False


def test_ignored_anchored_by_inner_slash():
    assert ignored(["docs/*.md"], "docs/a.md") is True
    assert ignored(["docs/*.md"], "x/docs/a.md") is False
    assert ignored(["docs/*.md"], "docs/sub/a.md") is False
    assert ignored(["docs/*.md"], "docs/x.md/inner") is True


def test_ignored_reinclude_inside_ignored_directory():
    assert ignored(["build", "!build/keep.txt"], "build/keep.txt") is False
    assert ignored(["build", "!build/keep.txt"], "build/other") is True
    assert ignored(["*.o", "!/src/main.o"], "src/main.o") is False
    assert ignored(["*.o", "!/src/main.o"], "lib/src/main.o") is True


def test_ignored_comments_and_escapes():
    assert ignored(["", "# comment", "*.tmp"], "a.tmp") is True
    assert ignored(["#x"], "#x") is False
    assert ignored(["\\#x"], "#x") is True
    assert ignored(["\\!x"], "!x") is True
    assert ignored(["!x"], "!x") is False
    assert ignored(["*", "!#x"], "a/#x") is False


def test_ignored_dot_rule_applies():
    assert ignored(["*"], ".env") is False
    assert ignored(["*"], ".git/config") is True
    assert ignored(["/**"], ".git/config") is False
    assert ignored(["/.git"], ".git/config") is True


def test_ignored_no_matching_rules():
    assert ignored([], "a") is False
    assert ignored(["# only a comment"], "a") is False
    assert ignored(["b", "/a/b"], "a") is False


def test_ignored_globstar_rules():
    assert ignored(["**/tmp"], "a/tmp/x") is True
    assert ignored(["**/tmp"], "tmp") is True
    assert ignored(["a/**/z"], "a/b/z/c") is True
    assert ignored(["a/**/z"], "b/a/z") is False


def test_ignored_invalid_rules():
    for rules in [["!"], ["/"], ["!/"], ["a/"], ["[x"], ["//a"], ["*.log", "[bad"], ["*", "!a/[z-a]"]]:
        assert _raises(ignored, rules, "x.log"), rules
