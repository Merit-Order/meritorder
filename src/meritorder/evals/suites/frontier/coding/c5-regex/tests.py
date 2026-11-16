"""Hidden tests for c5-regex. Expected values were computed with Python's re (re.ASCII)."""

from solution import findall, fullmatch, search, sub


def S(p, t):
    m = search(p, t)
    return None if m is None else (m.span(), tuple(m.span(i) for i in range(1, len(m.groups()) + 1)))


def F(p, t):
    m = fullmatch(p, t)
    return None if m is None else (m.span(), tuple(m.span(i) for i in range(1, len(m.groups()) + 1)))


def G(p, t):
    m = search(p, t)
    return None if m is None else (m.group(), m.groups())


def FG(p, t):
    m = fullmatch(p, t)
    return None if m is None else (m.group(), m.groups())


def invalid(p):
    """Every entry point must reject the pattern with ValueError, even on empty text."""
    for call in (lambda: search(p, ""), lambda: fullmatch(p, ""), lambda: findall(p, "abc"), lambda: sub(p, "x", "")):
        try:
            call()
        except ValueError:
            continue
        return False
    return True


def test_literals_and_leftmost():
    assert S('abc', 'xxabcx') == ((2, 5), ())
    assert S('abc', 'ababab') is None
    assert S('A', 'a') is None
    assert S('é+', 'xééa') == ((1, 3), ())
    assert S('', 'abc') == ((0, 0), ())
    assert S('b+|a', 'cab') == ((1, 2), ())
    assert S('c|ab', 'xabc') == ((1, 3), ())
    assert G('x*', 'abc') == ('', ())


def test_dot_excludes_newline():
    assert S('a.c', 'a\nc abc') == ((4, 7), ())
    assert S('.+', '\nab\nc') == ((1, 3), ())
    assert F('.*', 'ab\n') is None
    assert F('.*\n.', 'ab\nc') == ((0, 4), ())
    assert S('...', 'a\nbcd') == ((2, 5), ())


def test_set_escapes_are_ascii():
    assert G('\\w+', 'é_x9é') == ('_x9', ())
    assert S('\\s', 'a\x1cb\x0bc') == ((3, 4), ())
    assert S('\\d', '٣5') == ((1, 2), ())
    assert G('\\d+\\s\\w+', 'id: 42 foo_bar!') == ('42 foo_bar', ())
    assert S('\\W', 'ab_9é') == ((4, 5), ())
    assert S('\\S+', ' \t\n x-y ') == ((4, 7), ())
    assert S('\\D+', '12ab3') == ((2, 4), ())
    assert G('\\s+', 'a \t\r\x0c\x0b\nb') == (' \t\r\x0c\x0b\n', ())


def test_literal_escapes():
    assert G('\\(\\d\\)\\.\\*', 'x(3).*y') == ('(3).*', ())
    assert S('\\\\', 'a\\b') == ((1, 2), ())
    assert G('\\-\\/\\ \\_', 'x-/ _y') == ('-/ _', ())
    assert S('\\{2\\}', 'a{2}') == ((1, 4), ())
    assert S('a\\nb\\tc', 'a\nb\tc') == ((0, 5), ())
    assert S('\\$\\^\\|\\?\\+', '$^|?+') == ((0, 5), ())
    assert S('\\]\\}', ']}') == ((0, 2), ())


def test_classes_basic():
    assert G('[a-c]+', 'xxabcbd') == ('abcb', ())
    assert G('[^a-c]+', 'abxyzc') == ('xyz', ())
    assert S('[.]', 'abc') is None
    assert S('[.]', 'a.b') == ((1, 2), ())
    assert G('[*+?]+', 'a+*?b') == ('+*?', ())
    assert G('[$^|(]+', 'a|($^b') == ('|($^', ())
    assert S('[a^]', '^') == ((0, 1), ())
    assert G('[^\\W\\d]+', '12ab_3') == ('ab_', ())
    assert G('[\\d\\s]+', 'ab1 2\tc') == ('1 2\t', ())
    assert G('[^\\s]+', '  ab ') == ('ab', ())


def test_class_brackets_and_dashes():
    assert S('[]a]', 'x]') == ((1, 2), ())
    assert S('[^]a]', ']ab') == ((2, 3), ())
    assert S('[-b]', 'a-') == ((1, 2), ())
    assert S('[b-]', 'a-') == ((1, 2), ())
    assert S('[^-a]', '-ab') == ((2, 3), ())
    assert G('[a-c-e]+', 'x-ebd') == ('-eb', ())
    assert S('[\\]]', 'a]') == ((1, 2), ())
    assert G('[a\\-z]+', 'b-az') == ('-az', ())
    assert S('[[]', 'a[') == ((1, 2), ())
    assert G('[--/]+', 'a-./0') == ('-./', ())
    assert S('a]', 'a]') == ((0, 2), ())
    assert S('a}', 'a}') == ((0, 2), ())


def test_anchors():
    assert S('^a', 'ba') is None
    assert S('^a', 'ab') == ((0, 1), ())
    assert S('a$', 'a\n') == ((0, 1), ())
    assert S('a$', 'a\n\n') is None
    assert S('a$', 'ab') is None
    assert S('$', 'ab\n') == ((2, 2), ())
    assert S('x*$', 'ab\n') == ((2, 2), ())
    assert S('a^b', 'ab') is None
    assert S('a|^b', 'cb') is None
    assert S('^$', '') == ((0, 0), ())
    assert S('^$', '\n') == ((0, 0), ())
    assert S('(^a|b)+', 'ab') == ((0, 2), ((1, 2),))


def test_alternation_priority():
    assert S('a|ab', 'ab') == ((0, 1), ())
    assert F('a|ab', 'ab') == ((0, 2), ())
    assert G('(a|ab)(c|bcd)(d*)', 'abcd') == ('abcd', ('a', 'bcd', ''))
    assert FG('(a|ab)(c|bcd)(d*)', 'abcd') == ('abcd', ('a', 'bcd', ''))
    assert S('|a', 'a') == ((0, 0), ())
    assert S('a|', 'b') == ((0, 0), ())
    assert G('(x|xy|xyz)z', 'xyz') == ('xyz', ('xy',))
    assert G('(?:b|ab|abc)c', 'abcc') == ('abc', ())


def test_greedy_and_lazy():
    assert G('a+?', 'aaa') == ('a', ())
    assert G('a*?b', 'aaab') == ('aaab', ())
    assert G('<.+>', '<a><b>') == ('<a><b>', ())
    assert G('<.+?>', '<a><b>') == ('<a>', ())
    assert G('a??b', 'ab') == ('ab', ())
    assert G('(a??)(a*)', 'aaa') == ('aaa', ('', 'aaa'))
    assert G('(a*?)(a*)', 'aaa') == ('aaa', ('', 'aaa'))
    assert G('a{2,3}?', 'aaaa') == ('aa', ())
    assert G('a{2,}?', 'aaaa') == ('aa', ())
    assert FG('(a+?)(a*?)', 'aaa') == ('aaa', ('a', 'aa'))
    assert G('(\\w+?)(\\d*)$', 'ab12') == ('ab12', ('ab', '12'))
    assert G('(.*?),(.*)', 'a,b,c') == ('a,b,c', ('a', 'b,c'))


def test_counted_quantifiers():
    assert G('a{3}', 'aaaa') == ('aaa', ())
    assert G('a{2,}', 'aaaaa') == ('aaaaa', ())
    assert G('a{,2}', 'aaa') == ('aa', ())
    assert G('x{2,3}', 'xxxx') == ('xxx', ())
    assert S('ba{0}c', 'bc') == ((0, 2), ())
    assert S('(a){0}b', 'ab') == ((1, 2), ((-1, -1),))
    assert S('a{0,0}?b', 'ab') == ((1, 2), ())
    assert G('(?:ab){2}', 'abababa') == ('abab', ())
    assert G('a{,}', 'aaa') == ('aaa', ())
    assert G('(ab|a){2,3}?c', 'aababc') == ('aababc', ('ab',))


def test_braces_that_are_not_quantifiers():
    assert S('a{', 'xa{') == ((1, 3), ())
    assert S('a{x}', 'a{x}') == ((0, 4), ())
    assert S('a{1,2', 'aa{1,2') == ((1, 6), ())
    assert S('a{ 2}', 'a{ 2}') == ((0, 5), ())
    assert S('a{}', 'a{}') == ((0, 3), ())
    assert S('{x', '{x') == ((0, 2), ())
    assert S('x{,y}', 'x{,y}') == ((0, 5), ())
    assert S('a{2}{x', 'aa{x') == ((0, 4), ())


def test_group_numbering_and_spans():
    assert S('((a)(b(c)))', 'xabc') == ((1, 4), ((1, 4), (1, 2), (2, 4), (3, 4)))
    assert S('(?:(a)(?:b))(c)', 'abc') == ((0, 3), ((0, 1), (2, 3)))
    assert S('(a)|b', 'b') == ((0, 1), ((-1, -1),))
    assert S('(a)|(b)', 'b') == ((0, 1), ((-1, -1), (0, 1)))
    assert S('(a)?b', 'b') == ((0, 1), ((-1, -1),))
    assert S('()', 'x') == ((0, 0), ((0, 0),))
    assert S('(?:)', '') == ((0, 0), ())
    assert S('(a(?:b)?)+', 'aab') == ((0, 3), ((1, 3),))


def test_captures_keep_last_repetition():
    assert S('(\\w)+', 'abc') == ((0, 3), ((2, 3),))
    assert S('(?:(a)|b)+', 'ab') == ((0, 2), ((0, 1),))
    assert S('(?:(a)|(b))+', 'ab') == ((0, 2), ((0, 1), (1, 2)))
    assert S('(?:(a)|(b))+', 'ba') == ((0, 2), ((1, 2), (0, 1)))
    assert S('(?:(a)b|(a)c)+', 'abac') == ((0, 4), ((0, 1), (2, 3)))
    assert S('((a)|b)+', 'ab') == ((0, 2), ((1, 2), (0, 1)))
    assert S('(?:(x)|y)*z', 'xyyz') == ((0, 4), ((0, 1),))


def test_backtracking_undoes_captures():
    assert S('(?:(a)|(ab))+?c', 'ababc') == ((0, 5), ((-1, -1), (2, 4)))
    assert S('(a)?(a)?a', 'aa') == ((0, 2), ((0, 1), (-1, -1)))
    assert S('(a)*ab', 'aaab') == ((0, 4), ((1, 2),))
    assert S('(?:(a)x|a)b', 'ab') == ((0, 2), ((-1, -1),))
    assert S('((a)|(a)b)c', 'abc') == ((0, 3), ((0, 2), (-1, -1), (0, 1)))
    assert S('(?:(a)|a)*b', 'aab') == ((0, 3), ((1, 2),))


def test_empty_repetitions():
    assert S('(a*)*', 'aa') == ((0, 2), ((2, 2),))
    assert S('(a*)*', 'b') == ((0, 0), ((0, 0),))
    assert S('(a*)+', 'b') == ((0, 0), ((0, 0),))
    assert S('(a|)*', 'aa') == ((0, 2), ((2, 2),))
    assert S('(?:()|a){1,2}b', 'ab') == ((0, 2), ((0, 0),))
    assert S('(a?){3}b', 'ab') == ((0, 2), ((1, 1),))
    assert S('(|a)+', 'aa') == ((0, 0), ((0, 0),))
    assert F('(|a)+', 'aa') == ((0, 2), ((2, 2),))
    assert S('(a*)*b', 'aab') == ((0, 3), ((2, 2),))
    assert S('(a*?)*?b', 'aab') == ((0, 3), ((1, 2),))


def test_empty_repetitions_lazy_and_bounded():
    assert S('(a|)+?b', 'aab') == ((0, 3), ((1, 2),))
    assert S('(?:a|()){2,}c', 'ac') == ((0, 2), ((1, 1),))
    assert S('(?:()|a){2,3}', 'aaa') == ((0, 0), ((0, 0),))
    assert F('(?:()|a){2,3}', 'aaa') == ((0, 3), ((-1, -1),))
    assert S('(a?)+?$', 'aa') == ((0, 2), ((1, 2),))
    assert S('((a)|())*', 'aa') == ((0, 2), ((2, 2), (1, 2), (2, 2)))
    assert F('(b|a|)*?', 'ab') == ((0, 2), ((1, 2),))


def test_fullmatch_backtracks():
    assert F('a*?', 'aaa') == ((0, 3), ())
    assert F('(a|ab)*', 'abab') == ((0, 4), ((2, 4),))
    assert F('a$', 'a\n') is None
    assert F('', '') == ((0, 0), ())
    assert F('x*', '') == ((0, 0), ())
    assert F('(a)|b', 'b') == ((0, 1), ((-1, -1),))
    assert F('(a+)(a+)', 'aaaa') == ((0, 4), ((0, 3), (3, 4)))
    assert F('a+?b?', 'aab') == ((0, 3), ())
    assert F('(.)*', 'xyz') == ((0, 3), ((2, 3),))
    assert F('ab', 'abc') is None
    assert F('bc', 'abc') is None


def test_findall_shapes():
    assert findall('a.', 'abacad') == ['ab', 'ac', 'ad']
    assert findall('(a)(x)?', 'aa') == [('a', ''), ('a', '')]
    assert findall('(a)|b', 'ab') == ['a', '']
    assert findall('(\\d+)', 'a1b22c') == ['1', '22']
    assert findall('(\\w)(\\d)', 'a1b2c') == [('a', '1'), ('b', '2')]
    assert findall('x', 'abc') == []
    assert findall('(a)|(b)|c', 'abc') == [('a', ''), ('', 'b'), ('', '')]


def test_findall_empty_matches():
    assert findall('a*', 'baa') == ['', 'aa', '']
    assert findall('a??', 'aa') == ['', 'a', '', 'a', '']
    assert findall('x*', 'ab') == ['', '', '']
    assert findall('x*', '') == ['']
    assert findall('b*', 'ab') == ['', 'b', '']
    assert findall('$', 'a\n') == ['', '']
    assert findall('|a', 'aa') == ['', 'a', '', 'a', '']
    assert findall('a|', 'aba') == ['a', '', 'a', '']
    assert findall('(a*)', 'aab') == ['aa', '', '']
    assert findall('a*?', 'aa') == ['', 'a', '', 'a', '']


def test_sub_rules():
    assert sub('x*', '-', 'abxd') == '-a-b--d-'
    assert sub('a', 'X', 'aaa', 2) == 'XXa'
    assert sub('a', 'X', 'aaa') == 'XXX'
    assert sub('q', 'X', 'abc') == 'abc'
    assert sub('', '-', 'ab') == '-a-b-'
    assert sub('a|', '-', 'ab') == '--b-'
    assert sub('a*', '<>', 'baac') == '<>b<><>c<>'
    assert sub('(b)', '[\\1]', 'abc') == 'a[\\1]c'
    assert sub('\\s+', ' ', 'a  b\t\nc') == 'a b c'
    assert sub('x*', '-', 'abc', 2) == '-a-bc'
    assert sub('|a', '-', 'ab') == '---b-'
    assert sub('a*?', '<>', 'aa') == '<><><><><>'


def test_realistic_patterns():
    assert G('([\\w.]+)@(\\w+)\\.(com|org)', 'mail me: john.doe@example.org!') == ('john.doe@example.org', ('john.doe', 'example', 'org'))
    assert G('(\\d{4})-(\\d\\d)-(\\d\\d)', 'on 2024-02-29.') == ('2024-02-29', ('2024', '02', '29'))
    assert G('^\\s*(\\w+)\\s*=\\s*(.*?)\\s*$', '  key =  some value \n') == ('  key =  some value \n', ('key', 'some value'))
    assert G('(?:(\\d+)\\.)?(\\d+)\\.(\\d+)', 'v1.2') == ('1.2', (None, '1', '2'))
    assert G('([^,]*),([^,]*)', ',x,') == (',x', ('', 'x'))


def test_invalid_groups():
    assert invalid('(')
    assert invalid('a)')
    assert invalid(')')
    assert invalid('(a|b')
    assert invalid('((a)')
    assert invalid('(?q)')
    assert invalid('(?')
    assert invalid('a(b))')


def test_invalid_repeats():
    assert invalid('*a')
    assert invalid('a|*b')
    assert invalid('(*)')
    assert invalid('(?:+)')
    assert invalid('^*')
    assert invalid('$+')
    assert invalid('a**')
    assert invalid('a+*')
    assert invalid('a???')
    assert invalid('a*?*')
    assert invalid('a{2}{3}')
    assert invalid('{2}')
    assert invalid('x|{1,2}')
    assert invalid('a{3,2}')
    assert invalid('(|*)')


def test_invalid_classes_and_escapes():
    assert invalid('[a')
    assert invalid('[]')
    assert invalid('[^]')
    assert invalid('[z-a]')
    assert invalid('[\\d-z]')
    assert invalid('[a-\\w]')
    assert invalid('\\')
    assert invalid('a\\')
    assert invalid('\\q')
    assert invalid('[\\q]')
    assert invalid('\\1')
    assert invalid('[a-')
    assert invalid('[^')


NOT_IN_SUBSET = ['a*+', 'a++', 'a?+', 'a{2}+', '(?=a)', '(?P<n>a)', '\\b', '[\\b]', '\\A']


def test_match_object_api():
    m = search("(a)(x)?(b)", "zab")
    assert m.group() == "ab" and m.group(0) == "ab" and m.span() == (1, 3)
    assert m.group(1) == "a" and m.group(2) is None and m.group(3) == "b"
    assert m.span(2) == (-1, -1) and m.span(3) == (2, 3)
    assert m.groups() == ("a", None, "b")
    assert search("abc", "abc").groups() == ()


def test_bad_group_index_raises_index_error():
    m = search("(a)", "a")
    for g in (2, -1, 5):
        for call in (lambda: m.group(g), lambda: m.span(g)):
            try:
                call()
            except IndexError:
                continue
            raise AssertionError(f"group {g} should raise IndexError")


def test_no_match_returns_none():
    assert search("b", "aaa") is None
    assert fullmatch("a", "aa") is None
    assert findall("b", "aaa") == []


def test_python_extensions_are_invalid_here():
    for p in NOT_IN_SUBSET:
        assert invalid(p), p


def test_sub_repl_is_literal():
    assert sub("b", "\\1\\n", "abc") == "a\\1\\nc"
    assert sub("(a)", "\\g<1>", "xa") == "x\\g<1>"


def test_errors_raised_before_matching():
    assert invalid("(a")
    assert invalid("[z-a]x")
    assert invalid("ok|(")
