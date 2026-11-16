from solution import tokenize


def W(*texts):
    return [("word", t) for t in texts]


def w(t):
    return ("word", t)


def o(t):
    return ("op", t)


def _raises(line):
    try:
        tokenize(line)
    except ValueError:
        return True
    return False


def test_examples_from_spec():
    assert tokenize("grep -n 'a b' file.txt|wc -l > out # count") == [
        w("grep"), w("-n"), w("a b"), w("file.txt"), o("|"), w("wc"), w("-l"), o(">"), w("out")]
    assert tokenize('echo "x\\"y" \\; a#b') == W("echo", 'x"y', ";", "a#b")


def test_whitespace_runs_and_empty_input():
    assert tokenize("") == []
    assert tokenize(" \t\n ") == []
    assert tokenize("  ls\t\t-la \n  /tmp  ") == W("ls", "-la", "/tmp")
    assert tokenize("a\nb") == W("a", "b")
    assert tokenize("a\rb\u00a0c") == W("a\rb\u00a0c")


def test_single_quotes_are_literal():
    assert tokenize("'a \\b \"c\" #d |e\n'") == W('a \\b "c" #d |e\n')
    assert tokenize("echo 'it''s'") == W("echo", "its")


def test_double_quote_escapes():
    assert tokenize('"a\\"b\\\\c\\$d\\`e"') == W('a"b\\c$d`e')
    assert tokenize('"\\n\\x\\\'"') == W("\\n\\x\\'")
    assert tokenize('"a | b; c # d"') == W("a | b; c # d")
    assert tokenize('"it\'s"') == W("it's")


def test_backslash_outside_quotes():
    assert tokenize("a\\ b") == W("a b")
    assert tokenize("\\'x\\\"") == W("'x\"")
    assert tokenize("\\\\") == W("\\")
    assert tokenize("\\#x \\|y \\>\\>") == W("#x", "|y", ">>")
    assert tokenize("a\\\tb") == W("a\tb")


def test_adjacent_parts_join():
    assert tokenize("a\"b c\"'d'e") == W("ab cde")
    assert tokenize("pre'fix'\"$HOME\"post") == W("prefix$HOMEpost")


def test_empty_quoted_words():
    assert tokenize("''") == W("")
    assert tokenize('""') == W("")
    assert tokenize("a '' b") == W("a", "", "b")
    assert tokenize("a\"\"") == W("a")
    assert tokenize("'' \"\" ''") == W("", "", "")
    assert tokenize("x ''|y") == [w("x"), w(""), o("|"), w("y")]


def test_line_continuations():
    assert tokenize("ab\\\ncd") == W("abcd")
    assert tokenize('"ab\\\ncd"') == W("abcd")
    assert tokenize("'ab\\\ncd'") == W("ab\\\ncd")
    assert tokenize("a \\\n b") == W("a", "b")
    assert tokenize("\\\n") == []


def test_comments():
    assert tokenize("echo hi # comment ; | here") == W("echo", "hi")
    assert tokenize("#only") == []
    assert tokenize("a # c\nb") == W("a", "b")
    assert tokenize("echo a#b x#") == W("echo", "a#b", "x#")
    assert tokenize("echo '#x' \"#y\" \\#z") == W("echo", "#x", "#y", "#z")
    assert tokenize("a;#c\nb") == [w("a"), o(";"), w("b")]
    assert tokenize("a #'unterminated\nb") == W("a", "b")


def test_operators_split_words():
    assert tokenize("a|b") == [w("a"), o("|"), w("b")]
    assert tokenize("a||b&&c;d&e") == [w("a"), o("||"), w("b"), o("&&"), w("c"), o(";"), w("d"), o("&"), w("e")]
    assert tokenize("cat<in>>out>x") == [w("cat"), o("<"), w("in"), o(">>"), w("out"), o(">"), w("x")]
    assert tokenize("2>f") == [w("2"), o(">"), w("f")]
    assert tokenize(";;") == [o(";"), o(";")]


def test_operators_longest_match():
    assert tokenize("a|||b") == [w("a"), o("||"), o("|"), w("b")]
    assert tokenize(">>>") == [o(">>"), o(">")]
    assert tokenize("&&&") == [o("&&"), o("&")]
    assert tokenize("<<") == [o("<"), o("<")]
    assert tokenize("| |") == [o("|"), o("|")]


def test_quoted_and_escaped_operators_are_words():
    assert tokenize("'|' \"&&\" \\; '>'x") == W("|", "&&", ";", ">x")


def test_errors():
    for bad in ["'abc", '"abc', "abc\\", 'a "b\\"', "'", '"', "ok 'fine' \"no", '"abc\\']:
        assert _raises(bad), repr(bad)


def test_not_errors():
    assert tokenize("'\\'") == W("\\")
    assert tokenize('"\\\\"') == W("\\")
