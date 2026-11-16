"""Reference solution for c3-config."""

_NAME_CHARS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-")
_DIGITS = frozenset("0123456789")
_HEX = frozenset("0123456789abcdefABCDEF")
_WS = " \t"


class ConfigError(Exception):
    def __init__(self, line: int, message: str = "invalid configuration"):
        super().__init__(f"line {line}: {message}")
        self.line = line


class _Bad(Exception):
    pass


def _is_name(s: str) -> bool:
    return bool(s) and set(s) <= _NAME_CHARS


def _digits_end(s: str, i: int) -> int:
    """End index of a *digits* run starting at s[i], or -1 if there is none."""
    if i >= len(s) or s[i] not in _DIGITS:
        return -1
    i += 1
    while i < len(s):
        if s[i] in _DIGITS:
            i += 1
        elif s[i] == "_" and i + 1 < len(s) and s[i + 1] in _DIGITS:
            i += 2
        else:
            break
    return i


def _number(s: str):
    i = 0
    if s[:1] in ("+", "-"):
        i = 1
    if i >= len(s) or s[i] not in _DIGITS:
        raise _Bad("bad number")
    if s[i] == "0":
        j = i + 1
    else:
        j = _digits_end(s, i)
    is_float = False
    if j < len(s) and s[j] == ".":
        k = _digits_end(s, j + 1)
        if k < 0:
            raise _Bad("bad fraction")
        j, is_float = k, True
    if j < len(s) and s[j] in "eE":
        k = j + 1
        if k < len(s) and s[k] in "+-":
            k += 1
        k = _digits_end(s, k)
        if k < 0:
            raise _Bad("bad exponent")
        j, is_float = k, True
    if j != len(s):
        raise _Bad("bad number")
    clean = s.replace("_", "")
    return float(clean) if is_float else int(clean)


def _basic_string(s: str, i: int):
    """s[i] is the opening quote. Returns (value, index after the closing quote)."""
    out = []
    i += 1
    while True:
        if i >= len(s):
            raise _Bad("unterminated string")
        c = s[i]
        if c == '"':
            return "".join(out), i + 1
        if c == "\\":
            e = s[i + 1:i + 2]
            if e in ('"', "\\"):
                out.append(e)
                i += 2
            elif e == "n":
                out.append("\n")
                i += 2
            elif e == "t":
                out.append("\t")
                i += 2
            elif e == "u":
                h = s[i + 2:i + 6]
                if len(h) != 4 or not set(h) <= _HEX:
                    raise _Bad("bad \\u escape")
                cp = int(h, 16)
                if 0xD800 <= cp <= 0xDFFF:
                    raise _Bad("surrogate escape")
                out.append(chr(cp))
                i += 6
            else:
                raise _Bad("bad escape")
        else:
            out.append(c)
            i += 1


def _literal_string(s: str, i: int):
    j = s.find("'", i + 1)
    if j < 0:
        raise _Bad("unterminated string")
    return s[i + 1:j], j + 1


def _rest_ok(s: str, i: int) -> None:
    rest = s[i:].lstrip(_WS)
    if rest and rest[0] != "#":
        raise _Bad("unexpected text after value")


def _scalar(text: str):
    if text == "true":
        return True
    if text == "false":
        return False
    if text[:1] in _DIGITS or text[:1] in ("+", "-", "."):
        return _number(text)
    raise _Bad("bad list element")


def _list(s: str, i: int):
    items = []
    i += 1
    while True:
        while i < len(s) and s[i] in _WS:
            i += 1
        if i >= len(s):
            raise _Bad("unterminated list")
        c = s[i]
        if c == "]":
            return items, i + 1
        if c == '"':
            v, i = _basic_string(s, i)
        elif c == "'":
            v, i = _literal_string(s, i)
        else:
            j = i
            while j < len(s) and s[j] not in ",]":
                j += 1
            text = s[i:j].rstrip(_WS)
            if not text:
                raise _Bad("empty list element")
            v, i = _scalar(text), j
        items.append(v)
        while i < len(s) and s[i] in _WS:
            i += 1
        if i >= len(s):
            raise _Bad("unterminated list")
        if s[i] == ",":
            i += 1
        elif s[i] != "]":
            raise _Bad("expected , or ]")


def _value(raw: str):
    i = 0
    while i < len(raw) and raw[i] in _WS:
        i += 1
    s = raw[i:]
    if s[:1] == '"':
        v, j = _basic_string(s, 0)
        _rest_ok(s, j)
        return v
    if s[:1] == "'":
        v, j = _literal_string(s, 0)
        _rest_ok(s, j)
        return v
    if s[:1] == "[":
        v, j = _list(s, 0)
        _rest_ok(s, j)
        return v
    end = len(s)
    for k, c in enumerate(s):
        if c == "#" and (k == 0 or s[k - 1] in _WS):
            end = k
            break
    text = s[:end].rstrip(_WS)
    if not text:
        return ""
    if text in ("true", "false"):
        return text == "true"
    if text[0] in _DIGITS or text[0] in "+-.":
        return _number(text)
    return text


def _header(line: str) -> list[str]:
    close = line.find("]")
    if close < 0:
        raise _Bad("unterminated header")
    _rest_ok(line, close + 1)
    name = line[1:close].strip(_WS)
    parts = name.split(".")
    if not all(_is_name(p) for p in parts):
        raise _Bad("bad section name")
    return parts


def parse_config(text: str) -> dict:
    root: dict = {}
    current = root
    declared: set[tuple[str, ...]] = set()
    tables: set[int] = {id(root)}  # dicts that are tables (as opposed to values)
    for number, line in enumerate(text.split("\n"), start=1):
        if line.endswith("\r"):
            line = line[:-1]
        stripped = line.strip(_WS)
        try:
            if not stripped or stripped[0] in "#;":
                continue
            if stripped[0] == "[":
                path = tuple(_header(stripped))
                if path in declared:
                    raise _Bad("duplicate section")
                node = root
                for part in path:
                    if part not in node:
                        node[part] = {}
                        tables.add(id(node[part]))
                    elif not (isinstance(node[part], dict) and id(node[part]) in tables):
                        raise _Bad("section clashes with a key")
                    node = node[part]
                declared.add(path)
                current = node
                continue
            key, eq, raw = line.partition("=")
            if not eq:
                raise _Bad("missing =")
            key = key.strip(_WS)
            if not _is_name(key):
                raise _Bad("bad key")
            if key in current:
                raise _Bad("duplicate key")
            current[key] = _value(raw)
        except _Bad as e:
            raise ConfigError(number, str(e)) from None
    return root
