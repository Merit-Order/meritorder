"""Plausible but buggy c3-config: any `#` in an unquoted value starts a comment, and integers may have leading zeros."""

import re

NAME = re.compile(r"[A-Za-z0-9_-]+\Z")
INT = re.compile(r"[+-]?\d+(?:_\d+)*\Z")
FLOAT = re.compile(r"[+-]?(?:0|[1-9]\d*(?:_\d+)*)(?:\.\d+(?:_\d+)*)?(?:[eE][+-]?\d+(?:_\d+)*)?\Z")


class ConfigError(Exception):
    def __init__(self, line, message="invalid"):
        super().__init__(f"line {line}: {message}")
        self.line = line


class Bad(Exception):
    pass


def check_tail(rest):
    rest = rest.strip(" \t")
    if rest and not rest.startswith("#"):
        raise Bad("junk after value")


def read_basic(s, i):
    out = ""
    i += 1
    escapes = {'"': '"', "\\": "\\", "n": "\n", "t": "\t"}
    while i < len(s):
        c = s[i]
        if c == '"':
            return out, i + 1
        if c == "\\":
            nxt = s[i + 1:i + 2]
            if nxt in escapes:
                out += escapes[nxt]
                i += 2
                continue
            if nxt == "u":
                hexpart = s[i + 2:i + 6]
                if not re.fullmatch(r"[0-9a-fA-F]{4}", hexpart):
                    raise Bad("bad unicode escape")
                cp = int(hexpart, 16)
                if 0xD800 <= cp <= 0xDFFF:
                    raise Bad("surrogate")
                out += chr(cp)
                i += 6
                continue
            raise Bad("bad escape")
        out += c
        i += 1
    raise Bad("unterminated string")


def read_literal(s, i):
    end = s.find("'", i + 1)
    if end == -1:
        raise Bad("unterminated string")
    return s[i + 1:end], end + 1


def to_number(text):
    if INT.match(text):
        return int(text.replace("_", ""))
    if FLOAT.match(text) and any(c in text for c in ".eE"):
        return float(text.replace("_", ""))
    raise Bad("bad number")


def scalar(text):
    if text in ("true", "false"):
        return text == "true"
    if text and (text[0].isdigit() or text[0] in "+-."):
        return to_number(text)
    raise Bad("bad element")


def read_list(s, i):
    items = []
    i += 1
    while True:
        while i < len(s) and s[i] in " \t":
            i += 1
        if i >= len(s):
            raise Bad("unterminated list")
        if s[i] == "]":
            return items, i + 1
        if s[i] == '"':
            v, i = read_basic(s, i)
        elif s[i] == "'":
            v, i = read_literal(s, i)
        else:
            j = i
            while j < len(s) and s[j] not in ",]":
                j += 1
            text = s[i:j].strip(" \t")
            if not text:
                raise Bad("empty element")
            v, i = scalar(text), j
        items.append(v)
        while i < len(s) and s[i] in " \t":
            i += 1
        if i < len(s) and s[i] == ",":
            i += 1
        elif i >= len(s) or s[i] != "]":
            raise Bad("bad list")


def parse_value(raw):
    s = raw.lstrip(" \t")
    if s.startswith('"'):
        v, j = read_basic(s, 0)
        check_tail(s[j:])
        return v
    if s.startswith("'"):
        v, j = read_literal(s, 0)
        check_tail(s[j:])
        return v
    if s.startswith("["):
        v, j = read_list(s, 0)
        check_tail(s[j:])
        return v
    text = s.split("#", 1)[0].rstrip(" \t")
    if text == "":
        return ""
    if text in ("true", "false"):
        return text == "true"
    if text[0].isdigit() or text[0] in "+-.":
        return to_number(text)
    return text


def parse_config(text):
    result = {}
    table = result
    seen_headers = set()
    for lineno, line in enumerate(text.split("\n"), 1):
        if line.endswith("\r"):
            line = line[:-1]
        s = line.strip(" \t")
        try:
            if not s or s[0] in "#;":
                continue
            if s[0] == "[":
                end = s.find("]")
                if end == -1:
                    raise Bad("bad header")
                check_tail(s[end + 1:])
                parts = s[1:end].strip(" \t").split(".")
                if not all(NAME.match(p) for p in parts):
                    raise Bad("bad header")
                if tuple(parts) in seen_headers:
                    raise Bad("duplicate header")
                seen_headers.add(tuple(parts))
                node = result
                for p in parts:
                    node = node.setdefault(p, {})
                    if not isinstance(node, dict):
                        raise Bad("clash")
                table = node
                continue
            if "=" not in line:
                raise Bad("missing =")
            key, raw = line.split("=", 1)
            key = key.strip(" \t")
            if not NAME.match(key):
                raise Bad("bad key")
            if key in table:
                raise Bad("duplicate key")
            table[key] = parse_value(raw)
        except Bad as e:
            raise ConfigError(lineno, str(e)) from None
    return result
