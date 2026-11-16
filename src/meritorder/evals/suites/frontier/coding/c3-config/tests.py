from solution import ConfigError, parse_config


def _error_line(text):
    try:
        parse_config(text)
    except ConfigError as e:
        return e.line
    return None


def _value(raw):
    return parse_config("k =" + raw)["k"]


def _bad_value(raw):
    return _error_line("ok = 1\nk =" + raw) == 2


def test_example_from_spec():
    text = ('\n# service settings\nname = demo server   # trailing comment\nport = 8_080\n[db.primary]\n'
            'hosts = ["a:1", \'b:2\',]\nratio = 2.5e-1\nurl = postgres://u@h/db?x=1#frag\n')
    assert parse_config(text) == {"name": "demo server", "port": 8080,
                                  "db": {"primary": {"hosts": ["a:1", "b:2"], "ratio": 0.25,
                                                     "url": "postgres://u@h/db?x=1#frag"}}}


def test_empty_and_comment_only():
    assert parse_config("") == {}
    assert parse_config("  \n\t\n# c\n   ; also a comment\n") == {}


def test_sections_nest_and_merge_implicitly():
    text = "top = 1\n[a.b]\nx = 1\n[ a ]\ny = 2\n[a.c.d]\nz = 3\n[A]\nw = 4\n"
    assert parse_config(text) == {"top": 1, "a": {"b": {"x": 1}, "y": 2, "c": {"d": {"z": 3}}}, "A": {"w": 4}}
    assert parse_config("[x-1._y]   # header comment\nk = v") == {"x-1": {"_y": {"k": "v"}}}
    assert parse_config("[s]#c\n") == {"s": {}}


def test_unquoted_values_and_comments():
    assert _value(" demo   server  ") == "demo   server"
    assert _value(" a#b") == "a#b"
    assert _value(" a #b") == "a"
    assert _value(" a\t#b") == "a"
    assert _value(" a ; b") == "a ; b"
    assert _value(" say \"hi\" = 'x'") == "say \"hi\" = 'x'"
    assert _value(" C:\\dir\\n") == "C:\\dir\\n"
    assert _value(" _1") == "_1"
    assert _value(" inf") == "inf"


def test_empty_values():
    assert _value("") == ""
    assert _value("   ") == ""
    assert _value(" # just a comment") == ""
    assert _value("#c") == ""


def test_booleans():
    assert _value(" true") is True and _value(" false # no") is False
    assert _value(" True") == "True" and _value(" FALSE") == "FALSE"
    assert _value(" truex") == "truex" and _value(" true#x") == "true#x"


def test_basic_strings():
    assert _value(' "a\\"b\\\\c\\n\\t" ') == 'a"b\\c\n\t'
    assert _value(' "\\u00e9\\u00E9\\u0041" # c') == "\u00e9\u00e9A"
    assert _value(' "# not a comment"') == "# not a comment"
    assert _value(' ""') == ""
    assert _value(' "x"#c') == "x"
    for bad in [' "abc', ' "a\\x"', ' "\\u12"', ' "\\u12G4"', ' "\\uD800"', ' "a" b', ' "a\\"', ' "a""b"']:
        assert _bad_value(bad), bad


def test_literal_strings():
    assert _value(" 'C:\\new\\table'") == "C:\\new\\table"
    assert _value(" '\"quoted\" # x'  # real comment") == '"quoted" # x'
    for bad in [" 'abc", " 'a' 'b'", " 'it''s'"]:
        assert _bad_value(bad), bad


def test_integers():
    assert _value(" 0") == 0 and _value(" -0") == 0 and _value(" +17") == 17
    assert _value(" 1_000_000") == 1000000 and _value(" -12_3") == -123
    v = _value(" 42 # answer")
    assert v == 42 and isinstance(v, int) and not isinstance(v, bool)
    for bad in [" 012", " 00", " -01", " 0_1", " 1__0", " 1_", " +", " -", " 1x", " 1 2", " 12#c", " +inf", " 1_000_"]:
        assert _bad_value(bad), bad


def test_floats():
    for raw, want in [(" 3.14", 3.14), (" -0.5", -0.5), (" 1e3", 1000.0), (" 2.5E-2", 0.025), (" +1_0.0_1", 10.01),
                      (" 0.0", 0.0), (" 6e+2", 600.0), (" 1_0e1_0", 1e11)]:
        v = _value(raw)
        assert v == want and isinstance(v, float), raw
    for bad in [" .5", " 1.", " 1.e5", " 1e", " 01.5", " 1.5.2", " 1e5.0", " 1._5", " 1e_5", " 1.5e+"]:
        assert _bad_value(bad), bad


def test_lists():
    assert _value(' [1, 2.5, "a,b", \'c]\', true, -3]') == [1, 2.5, "a,b", "c]", True, -3]
    assert _value(" []") == [] and _value(" [ ]") == []
    assert _value(" [ 1 ,2,\t3 , ] # tail") == [1, 2, 3]
    assert _value(' ["x"]#c') == ["x"]
    for bad in [" [1,,2]", " [,]", " [1 2]", " [a]", " [[1]]", " [1, 2", " [1] x", ' ["a" "b"]', " [012]",
                " [1 # c]", ' ["abc]']:
        assert _bad_value(bad), bad


def test_key_rules():
    assert parse_config(" \tmy-key_2 \t=\tv") == {"my-key_2": "v"}
    assert parse_config("a==b") == {"a": "=b"}
    assert parse_config("Key = 1\nkey = 2") == {"Key": 1, "key": 2}
    for text in ["no equals here", " = 1", "a.b = 1", "a b = 1", "\u00e9 = 1", "k\u00e9y = 2"]:
        assert _error_line("x = 1\n\n" + text) == 3, text


def test_bad_headers():
    for header in ["[]", "[ ]", "[a..b]", "[a b]", "[a. b]", "[a", "[a]]", "[a] x", "[.a]", "[a.]", "[\u00e9]",
                   "[a] ; c"]:
        assert _error_line("x = 1\n" + header + "\ny = 2") == 2, header


def test_duplicates_and_clashes():
    assert _error_line("a = 1\nb = 2\na = 3") == 3
    assert _error_line("[s]\na = 1\n[t]\n[s]\nb = 2") == 4
    assert _error_line("[a.b]\n[a]\n[a.b]") == 3
    assert _error_line("[a]\nb = 1\n[a.b]") == 3
    assert _error_line("[a.b]\nx = 1\n[a]\nb = 2") == 4
    assert _error_line("a = 1\n[a]") == 2
    assert _error_line("a = [1]\n[a.b]") == 2
    assert _error_line("[a]\n[a.b]\n[a.b.c]\n[a.b]") == 4


def test_crlf_and_line_numbers():
    text = "a = 1\r\n\r\n[s]\r\nb = 'x'\r\nc = \"y\"\r\n"
    assert parse_config(text) == {"a": 1, "s": {"b": "x", "c": "y"}}
    assert _error_line("a = 1\r\n# c\r\n\r\nb = 'x\r\n") == 4
    assert _value(" bare\r") == "bare"


def test_first_error_wins():
    assert _error_line("[ok]\nk = 1\nbad line\nk = 2\n[") == 3
