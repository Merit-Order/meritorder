from solution import compare, parse, satisfies, sort_versions


def _raises(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return True
    return False


def test_parse_plain():
    assert parse("2.10.3") == (2, 10, 3, (), ())
    assert parse("0.0.0") == (0, 0, 0, (), ())
    assert parse("99999999999999999999.0.1") == (99999999999999999999, 0, 1, (), ())


def test_parse_prerelease_and_build():
    assert parse("1.0.0-alpha.1+build.001") == (1, 0, 0, ("alpha", 1), ("build", "001"))
    assert parse("1.0.0+0.build.1-rc.10000aaa-kk-0.1") == (1, 0, 0, (), ("0", "build", "1-rc", "10000aaa-kk-0", "1"))
    assert parse("1.2.3-0") == (1, 2, 3, (0,), ())
    pre = parse("1.2.3-0A.is.legal")[3]
    assert pre == ("0A", "is", "legal") and isinstance(pre[0], str)


def test_parse_hyphen_identifiers():
    assert parse("1.0.0--") == (1, 0, 0, ("-",), ())
    assert parse("1.0.0-x-y-z.--") == (1, 0, 0, ("x-y-z", "--"), ())
    assert parse("1.2.3----RC-SNAPSHOT.12.9.1--.12+788") == (1, 2, 3, ("---RC-SNAPSHOT", 12, 9, "1--", 12), ("788",))
    assert parse("1.0.0-001-a")[3] == ("001-a",)


def test_invalid_structure():
    for bad in ["", "1", "1.2", "1.2.3.4", "1.2.3.DEV", "alpha", "1.2-SNAPSHOT", "-1.0.3-gamma+b7718",
                "+justmeta", "1..3", "1.2.", ".1.2"]:
        assert _raises(parse, bad), bad


def test_invalid_leading_zeros():
    for bad in ["01.1.1", "1.01.1", "1.1.01", "00.0.0", "1.2.3-0123", "1.2.3-alpha.01", "1.2.3-00"]:
        assert _raises(parse, bad), bad


def test_invalid_identifiers():
    for bad in ["1.2.3-", "1.2.3+", "1.2.3-+", "1.2.3-alpha..1", "1.2.3+a..b", "1.1.2+.123", "9.8.7+meta+meta",
                "9.8.7-whatever+meta+meta", "1.2.3-alpha_beta", "1.2.3-alpha.", "1.2.3+build!"]:
        assert _raises(parse, bad), bad


def test_invalid_characters_and_whitespace():
    for bad in ["v1.2.3", " 1.2.3", "1.2.3 ", "1.2.3\n", "1. 2.3", "1_0.0.0", "1.0.0-\u00e9", "1.2.3-\u03b1",
                "\u0661.2.3", "1.2.\u00b2", "1.2.3-\u0661", "+1.2.3", "1.2.3+\u00e9"]:
        assert _raises(parse, bad), repr(bad)


def test_compare_spec_chain():
    chain = ["1.0.0-alpha", "1.0.0-alpha.1", "1.0.0-alpha.beta", "1.0.0-beta", "1.0.0-beta.2", "1.0.0-beta.11",
             "1.0.0-rc.1", "1.0.0", "1.0.1", "1.1.0", "1.10.0", "2.0.0", "10.0.0"]
    for i, a in enumerate(chain):
        for j, b in enumerate(chain):
            want = (i > j) - (i < j)
            assert compare(a, b) == want, (a, b)


def test_compare_identifier_rules():
    assert compare("1.0.0-10", "1.0.0-9") == 1
    assert compare("1.0.0-9", "1.0.0-10a") == -1
    assert compare("1.0.0-999", "1.0.0-a") == -1
    assert compare("1.0.0-B", "1.0.0-a") == -1
    assert compare("1.0.0-alpha", "1.0.0-alpha-1") == -1
    assert compare("1.0.0-a.b.c", "1.0.0-a.b") == 1
    assert compare("1.0.0-rc.1", "0.9.9") == 1
    assert compare("1.0.0-18446744073709551616", "1.0.0-18446744073709551615") == 1
    assert compare("1.0.0-0A", "1.0.0-1") == 1


def test_compare_ignores_build():
    assert compare("1.0.0+a", "1.0.0+b") == 0
    assert compare("1.0.0-rc.1+x.7", "1.0.0-rc.1") == 0
    assert compare("1.0.0+zzz", "1.0.0-zzz") == 1
    assert compare("1.0.0", "1.0.0") == 0


def test_compare_rejects_invalid():
    assert _raises(compare, "1.0.0", "1.0")
    assert _raises(compare, "1.0.0-01", "1.0.0")


def test_sort_versions_stable():
    vs = ["1.0.0+b", "1.0.0-rc.1", "1.0.0+a", "0.9.0", "1.0.0", "1.0.0-rc.1+meta", "1.0.0-beta.11", "1.0.0-beta.2"]
    assert sort_versions(vs) == ["0.9.0", "1.0.0-beta.2", "1.0.0-beta.11", "1.0.0-rc.1", "1.0.0-rc.1+meta",
                                 "1.0.0+b", "1.0.0+a", "1.0.0"]
    assert vs[0] == "1.0.0+b"  # the input list is not modified
    assert sort_versions([]) == []
    assert _raises(sort_versions, ["1.0.0", "bad"])


def test_satisfies_basic_operators():
    assert satisfies("1.4.0", ">=1.2.0 <2.0.0")
    assert not satisfies("2.0.0", ">=1.2.0 <2.0.0")
    assert satisfies("1.2.0", ">=1.2.0")
    assert not satisfies("1.2.0", ">1.2.0")
    assert satisfies("1.2.0", "<=1.2.0") and not satisfies("1.2.1", "<=1.2.0")
    assert satisfies("1.2.0+build", "=1.2.0") and satisfies("1.2.0", "1.2.0+other")
    assert not satisfies("1.2.0-rc.1", "1.2.0")


def test_satisfies_caret():
    assert satisfies("1.9.9", "^1.2.3") and not satisfies("2.0.0", "^1.2.3") and not satisfies("1.2.2", "^1.2.3")
    assert satisfies("0.2.9", "^0.2.3") and not satisfies("0.3.0", "^0.2.3")
    assert satisfies("0.0.3", "^0.0.3") and not satisfies("0.0.4", "^0.0.3") and not satisfies("0.0.2", "^0.0.3")
    assert satisfies("0.0.4-alpha", "^0.0.3")
    assert satisfies("1.2.3-beta.10", "^1.2.3-beta.2") and not satisfies("1.2.3-beta.1", "^1.2.3-beta.2")


def test_satisfies_tilde():
    assert satisfies("1.2.9", "~1.2.3") and not satisfies("1.3.0", "~1.2.3") and not satisfies("1.2.2", "~1.2.3")
    assert satisfies("0.0.9", "~0.0.1") and not satisfies("0.1.0", "~0.0.1")
    assert satisfies("1.3.0-alpha", "~1.2.3")


def test_satisfies_prereleases_by_precedence_only():
    assert satisfies("2.0.0-rc.1", "^1.2.3")
    assert not satisfies("1.2.3-beta", "^1.2.3")
    assert satisfies("1.0.0-rc.1", ">=1.0.0-beta <1.0.0")


def test_satisfies_alternatives():
    c = " >=1.0.0 <1.5.0 ||  >=2.0.0\t<2.1.0 "
    assert satisfies("1.4.9", c) and satisfies("2.0.5", c)
    assert not satisfies("1.5.0", c) and not satisfies("2.1.0", c) and not satisfies("0.9.0", c)
    assert satisfies("0.3.9", "^0.2.0 || ~0.3.1")
    assert satisfies("3.0.0", "1.0.0||3.0.0")


def test_satisfies_invalid_constraints():
    for bad in ["", "   ", ">= 1.2.3", "^1.2", "1.0.0 ||", "|| 1.0.0", "1.0.0 || || 2.0.0", "=>1.0.0", "!1.0.0",
                "1.0.0 | 2.0.0", ">=v1.0.0", "~>1.0.0", "<1.0.0-01"]:
        assert _raises(satisfies, "1.0.0", bad), repr(bad)
    assert _raises(satisfies, "1.0", ">=1.0.0")
