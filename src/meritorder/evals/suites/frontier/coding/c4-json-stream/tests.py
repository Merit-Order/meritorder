import math

from solution import JSONStreamError, Tokenizer

BO, EO, BA, EA = "begin_object", "end_object", "begin_array", "end_array"


def _whole(doc):
    t = Tokenizer()
    return t.feed(doc) + t.close()


def _chunked(chunks):
    t = Tokenizer()
    out = []
    for c in chunks:
        out += t.feed(c)
    return out + t.close()


def _error(chunks):
    """(offset, line, column) of the error raised while feeding ``chunks`` and closing; None if there is none."""
    t = Tokenizer()
    try:
        for c in chunks:
            t.feed(c)
        t.close()
    except JSONStreamError as e:
        return (e.offset, e.line, e.column)
    return None


def _offset(doc):
    e = _error([doc])
    assert e is not None, f"no error for {doc!r}"
    return e[0]


def _every_split_same_tokens(doc):
    want = _whole(doc)
    for i in range(len(doc) + 1):
        assert _chunked([doc[:i], doc[i:]]) == want, (doc, i)
    assert _chunked(list(doc)) == want, doc
    return want


def test_scalar_documents():
    assert _whole("42") == [("number", 42, 0)]
    assert _whole("  -3.5e2 ") == [("number", -350.0, 2)]
    assert _whole('"hi"') == [("string", "hi", 0)]
    assert _whole("true") == [("bool", True, 0)]
    assert _whole(" false") == [("bool", False, 1)]
    assert _whole("null\n") == [("null", None, 0)]


def test_nested_tokens_and_offsets():
    doc = '{"a": [1, true, null], "b": {"c": "d"}}'
    assert _whole(doc) == [
        (BO, None, 0), ("key", "a", 1), (BA, None, 6), ("number", 1, 7), ("bool", True, 10), ("null", None, 16),
        (EA, None, 20), ("key", "b", 23), (BO, None, 28), ("key", "c", 29), ("string", "d", 34), (EO, None, 37),
        (EO, None, 38),
    ]


def test_keys_versus_strings():
    assert _whole('{"k": "v", "list": ["k", {"x": "y"}]}') == [
        (BO, None, 0), ("key", "k", 1), ("string", "v", 6), ("key", "list", 11), (BA, None, 19), ("string", "k", 20),
        (BO, None, 25), ("key", "x", 26), ("string", "y", 31), (EO, None, 34), (EA, None, 35), (EO, None, 36),
    ]


def test_empty_containers_and_duplicate_keys():
    assert _whole('{"a":{},"a":[],"b":[[]],"":""}') == [
        (BO, None, 0), ("key", "a", 1), (BO, None, 5), (EO, None, 6), ("key", "a", 8), (BA, None, 12), (EA, None, 13),
        ("key", "b", 15), (BA, None, 19), (BA, None, 20), (EA, None, 21), (EA, None, 22), ("key", "", 24),
        ("string", "", 27), (EO, None, 29),
    ]


def test_number_values_and_types():
    cases = {"0": 0, "-0": 0, "10": 10, "123456789012345678901234567890": 123456789012345678901234567890,
             "1.0": 1.0, "1e2": 100.0, "1E+2": 100.0, "1e-2": 0.01, "0.5": 0.5, "-12.25e1": -122.5, "0e0": 0.0}
    for text, want in cases.items():
        [(kind, value, off)] = _whole(text)
        assert kind == "number" and off == 0 and value == want, text
        assert isinstance(value, type(want)) and not isinstance(value, bool), text
    [(_, neg_zero, _)] = _whole("-0.0")
    assert isinstance(neg_zero, float) and neg_zero == 0 and math.copysign(1, neg_zero) == -1


def test_invalid_number_positions():
    cases = {"01": 1, "-01": 2, "+1": 0, ".5": 0, "[1.]": 3, "[1.e5]": 3, "[1e]": 3, "[1e+]": 4, "[--1]": 2,
             "[1-2]": 2, "[0x1]": 2, "[1.5.2]": 4, "[1e5e]": 4, "[-]": 2, "[00]": 2, "[1 2]": 3, "[-a]": 2,
             "[1,01]": 4, "[1E-+2]": 4}
    for doc, want in cases.items():
        assert _offset(doc) == want, doc


def test_truncated_numbers_fail_at_end_of_stream():
    for doc in ["-", "1.", "1e", "1e-", "-0.", " 7E+"]:
        assert _error([doc]) == (len(doc), 1, len(doc) + 1), doc


def test_number_waits_for_the_next_character():
    t = Tokenizer()
    assert t.feed("[12") == [(BA, None, 0)]
    assert t.feed("3") == []
    assert t.feed("4,") == [("number", 1234, 1)]
    assert t.feed("5") == []
    assert t.feed("]") == [("number", 5, 6), (EA, None, 7)]
    assert t.close() == []


def test_top_level_number_completes_on_close_or_whitespace():
    t = Tokenizer()
    assert t.feed("42") == []
    assert t.close() == [("number", 42, 0)]
    t = Tokenizer()
    assert t.feed("7") == []
    assert t.feed(" ") == [("number", 7, 0)]
    assert t.close() == []


def test_literals_complete_on_their_last_letter():
    t = Tokenizer()
    assert t.feed("[tr") == [(BA, None, 0)]
    assert t.feed("u") == []
    assert t.feed("e") == [("bool", True, 1)]
    assert t.feed(",nul") == []
    assert t.feed("l") == [("null", None, 6)]
    assert t.feed("]") == [(EA, None, 10)]
    assert t.close() == []


def test_strings_complete_on_the_closing_quote():
    t = Tokenizer()
    assert t.feed('{"ke') == [(BO, None, 0)]
    assert t.feed('y"') == [("key", "key", 1)]
    assert t.feed(': "va') == []
    assert t.feed('l"') == [("string", "val", 8)]
    assert t.feed("}") == [(EO, None, 13)]


def test_bad_literals():
    cases = {"trux": 3, "nulL": 3, "True": 0, "[truex]": 5, "[true1]": 5, "NaN": 0, "Infinity": 0,
             "-Infinity": 1, "falsey": 5, "[1true]": 2, 'tru"': 3, "nil": 1, "'a'": 0}
    for doc, want in cases.items():
        assert _offset(doc) == want, doc
    assert _error(["tru"]) == (3, 1, 4)
    assert _error(["[fals"]) == (5, 1, 6)


def test_simple_escapes():
    assert _whole(r'"a\"b\\c\/d\b\f\n\r\t"') == [("string", 'a"b\\c/d\b\f\n\r\t', 0)]
    assert _whole(r'"\u00e9\u00C9\u0041\u20ac"') == [("string", "\u00e9\u00c9A\u20ac", 0)]


def test_surrogate_pairs_combine():
    assert _whole(r'"\ud83d\ude00"') == [("string", "\U0001F600", 0)]
    assert _whole(r'["\uD83D\uDE00x"]') == [(BA, None, 0), ("string", "\U0001F600x", 1), (EA, None, 16)]


def test_lone_surrogates_are_kept():
    assert _whole(r'"\ud83d"') == [("string", "\ud83d", 0)]
    assert _whole(r'"\ude00\ud83d"') == [("string", "\ude00\ud83d", 0)]
    assert _whole(r'"\ud83dx"') == [("string", "\ud83dx", 0)]
    assert _whole(r'"\ud83d\u0041"') == [("string", "\ud83dA", 0)]
    assert _whole(r'"\ud83d\ud83d\ude00"') == [("string", "\ud83d\U0001F600", 0)]
    assert _whole(r'"\ud83d\n"') == [("string", "\ud83d\n", 0)]


def test_unescaped_surrogate_is_not_combined_with_an_escape():
    [(kind, value, _)] = _whole('"\\ud83d' + "\ude00" + '"')
    assert kind == "string" and value == "\ud83d\ude00" and len(value) == 2


def test_bad_escapes():
    cases = {r'"\x"': 2, r'"\u12G4"': 5, r'"\u12"': 5, r'"\U0041"': 2, '"\\\'"': 2, r'"\ud83d\uzzzz"': 9,
             r'["\a"]': 3}
    for doc, want in cases.items():
        assert _offset(doc) == want, doc
    assert _error(['"\\']) == (2, 1, 3)
    assert _error(['"\\u00']) == (5, 1, 6)


def test_control_characters_in_strings():
    assert _error(['"a\tb"']) == (2, 1, 3)
    assert _error(['"a\nb"']) == (2, 1, 3)
    assert _error(['"\x00"']) == (1, 1, 2)
    assert _error(['["ok", "\x1f"]']) == (8, 1, 9)
    assert _whole('"\x7f \x20"') == [("string", "\x7f  ", 0)]


def test_non_ascii_text_and_offsets():
    assert _whole('["h\u00e9llo \U0001F600 \u2028", 1]') == [
        (BA, None, 0), ("string", "h\u00e9llo \U0001F600 \u2028", 1), ("number", 1, 14), (EA, None, 15)]
    assert _error(['["\U0001F600", x]']) == (6, 1, 7)


def test_structure_errors():
    cases = {"[1,]": 3, '{"a":1,}': 7, '{"a" 1}': 5, '{"a":}': 5, '{"a"}': 4, "{1:2}": 1, "{,}": 1, "[,1]": 1,
             "[}": 1, "{]": 1, "]": 0, "}": 0, ":": 0, ",": 0, '{"a",': 4, '{"a":1 "b":2}': 7, '{"a":1:2}': 6,
             "[1]]": 3, '{"a":1}}': 7, '["a":1]': 4, '[1,,2]': 3, '{"a"::1}': 5, "[[]": None, '{"a":[}': 6}
    for doc, want in cases.items():
        if want is None:
            assert _error([doc]) == (len(doc), 1, len(doc) + 1), doc
        else:
            assert _offset(doc) == want, doc


def test_only_one_top_level_value():
    cases = {"[1] [2]": 4, '"a" "b"': 4, "1 2": 2, "{} x": 3, "null,": 4, "[]]": 2, '""1': 2}
    for doc, want in cases.items():
        assert _offset(doc) == want, doc
    assert _whole("[1] \n\t\r ") == [(BA, None, 0), ("number", 1, 1), (EA, None, 2)]


def test_whitespace_set():
    assert _offset("[1,\x0c2]") == 3
    assert _offset("\ufeff[]") == 0
    assert _offset("[\u00a01]") == 1
    assert _offset("[1\u2003]") == 2
    assert _whole("\r\n[ 1 ,\t2 ]\r\n") == [(BA, None, 2), ("number", 1, 4), ("number", 2, 8), (EA, None, 10)]


def test_incomplete_documents_fail_at_end_of_stream():
    assert _error([""]) == (0, 1, 1)
    assert _error([]) == (0, 1, 1)
    assert _error(["   "]) == (3, 1, 4)
    for doc in ["[", "[1", '{"a"', '{"a":', '{"a":1', '"abc', '{"a":1,', "[1,", '{"', '["\\u12']:
        assert _error([doc]) == (len(doc), 1, len(doc) + 1), doc
    assert _error(["[\n"]) == (2, 2, 1)
    assert _error(['{\n  "a": [\n']) == (11, 3, 1)


def test_line_and_column():
    doc = '{\n  "a": [1,\n    2,,\n  ]\n}'
    off = doc.index(",,") + 1
    assert _error([doc]) == (off, 3, 7)
    assert _error(["[1,\n  2,,"]) == (8, 2, 5)


def test_carriage_return_does_not_end_a_line():
    assert _error(["[\r\n1,\r\n]"]) == (7, 3, 1)
    assert _error(["[\r1,\r]"]) == (5, 1, 6)


def test_every_split_point_gives_the_same_tokens():
    docs = ['{"a": [1, -2.5e+3, true, false, null], "b\\u00e9": {"c": "d\\n\\"q\\""}, "e": [[], {}]}',
            r'["\ud83d\ude00", "\u20AC\\", 0, -0, 10, 1.5E-7, "", "\ud83d"]',
            "  \r\n 123456 \t",
            '{"key": "va\\u00e9l\\ud834\\udd1e"}']
    for doc in docs:
        assert _every_split_same_tokens(doc)


def test_three_way_splits_match():
    doc = '{"x": [10, 200, 3000], "y": "a\\u0042c", "z": [true, null, -0.5]}'
    want = _whole(doc)
    for i in range(0, len(doc) + 1, 3):
        for j in range(i, len(doc) + 1, 4):
            assert _chunked([doc[:i], doc[i:j], doc[j:]]) == want, (i, j)


def test_empty_chunks_are_harmless():
    t = Tokenizer()
    assert t.feed("") == []
    out = []
    for ch in '[1, "a"]':
        out += t.feed(ch) + t.feed("")
    assert out + t.close() == [(BA, None, 0), ("number", 1, 1), ("string", "a", 4), (EA, None, 7)]


def test_error_is_raised_by_the_chunk_containing_it():
    for doc in ["[1,]", '{"a":1 "b":2}', '["\\u12G4"]', "[0, 01]", '{"a": tru "b"}', "[1] [2]"]:
        k = _offset(doc)
        assert k < len(doc)
        for i in range(len(doc) + 1):
            t = Tokenizer()
            if i <= k:
                t.feed(doc[:i])  # must not raise yet
                try:
                    t.feed(doc[i:])
                except JSONStreamError as e:
                    assert e.offset == k, (doc, i)
                else:
                    raise AssertionError(f"no error feeding {doc!r} split at {i}")
            else:
                try:
                    t.feed(doc[:i])
                except JSONStreamError as e:
                    assert e.offset == k, (doc, i)
                else:
                    raise AssertionError(f"no error feeding {doc[:i]!r}")


def test_tokens_before_the_error_when_fed_one_char_at_a_time():
    t = Tokenizer()
    out = []
    doc = "[1, 2, }"
    for i, ch in enumerate(doc):
        try:
            out += t.feed(ch)
        except JSONStreamError as e:
            assert (i, e.offset, e.line, e.column) == (7, 7, 1, 8)
            break
    else:
        raise AssertionError("no error")
    assert out == [(BA, None, 0), ("number", 1, 1), ("number", 2, 4)]


def test_errors_are_sticky():
    t = Tokenizer()
    try:
        t.feed('{"a":\n x')
    except JSONStreamError as e:
        first = (e.offset, e.line, e.column)
    else:
        raise AssertionError("no error")
    assert first == (7, 2, 2)
    for call in [lambda: t.feed("1}"), t.close, lambda: t.feed("")]:
        try:
            call()
        except JSONStreamError as e:
            assert (e.offset, e.line, e.column) == first
        else:
            raise AssertionError("a broken tokenizer must keep raising")


def test_use_after_close():
    t = Tokenizer()
    t.feed("[]")
    assert t.close() == []
    for call in [lambda: t.feed("1"), lambda: t.feed(""), t.close]:
        try:
            call()
        except RuntimeError:
            continue
        raise AssertionError("using a closed tokenizer must raise RuntimeError")


def test_error_class():
    assert issubclass(JSONStreamError, ValueError)
    try:
        Tokenizer().feed("x")
    except ValueError as e:
        assert isinstance(e, JSONStreamError) and (e.offset, e.line, e.column) == (0, 1, 1)
    else:
        raise AssertionError("no error")


def test_deep_nesting():
    n = 100000
    toks = _chunked(["[" * n, "]" * n])
    assert len(toks) == 2 * n
    assert toks[0] == (BA, None, 0) and toks[n - 1] == (BA, None, n - 1) and toks[n] == (EA, None, n)
    assert toks[-1] == (EA, None, 2 * n - 1)
    m = 20000
    doc = '{"a":' * m + "1" + "}" * m
    toks = _whole(doc)
    assert len(toks) == 3 * m + 1 and toks[2 * m] == ("number", 1, 5 * m)
    assert _error(["[" * n + "]" * (n - 1)]) == (2 * n - 1, 1, 2 * n)


def test_many_small_chunks_on_a_larger_document():
    items = ", ".join(f'{{"id": {i}, "name": "n\\u00e9{i}", "ok": {"true" if i % 2 else "false"}, "v": {i}.5}}'
                      for i in range(400))
    doc = "[" + items + "]"
    want = _whole(doc)
    assert len(want) == 2 + 400 * 10
    assert want[1:11] == [(BO, None, 1), ("key", "id", 2), ("number", 0, 8), ("key", "name", 11),
                          ("string", "n\u00e90", 19), ("key", "ok", 31), ("bool", False, 37), ("key", "v", 44),
                          ("number", 0.5, 49), (EO, None, 52)]
    for size in (1, 7, 64):
        assert _chunked([doc[i:i + size] for i in range(0, len(doc), size)]) == want
