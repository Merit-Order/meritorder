"""Reference solution for c3-shell-tokens."""

_WS = " \t\n"
_OPS = ("||", "&&", ">>", "|", "&", ";", ">", "<")
_OP_CHARS = frozenset("|&;><")
_DQ_ESCAPES = frozenset('"\\$`')


def tokenize(line: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    buf: list[str] = []
    in_word = False  # a word has started (possibly empty, from quotes)
    at_start = True  # a new token would start here (for comments)
    i, n = 0, len(line)

    def flush():
        nonlocal in_word
        if in_word:
            tokens.append(("word", "".join(buf)))
            buf.clear()
            in_word = False

    while i < n:
        c = line[i]
        if c in _WS:
            flush()
            at_start = True
            i += 1
        elif c == "\\":
            if i + 1 >= n:
                raise ValueError("backslash at end of input")
            if line[i + 1] == "\n":
                i += 2
                continue
            buf.append(line[i + 1])
            in_word = True
            at_start = False
            i += 2
        elif c == "#" and at_start:
            j = line.find("\n", i)
            i = n if j < 0 else j
        elif c in _OP_CHARS:
            flush()
            for op in _OPS:
                if line.startswith(op, i):
                    tokens.append(("op", op))
                    i += len(op)
                    break
            at_start = True
        elif c == "'":
            j = line.find("'", i + 1)
            if j < 0:
                raise ValueError("unterminated single quote")
            buf.append(line[i + 1:j])
            in_word = True
            at_start = False
            i = j + 1
        elif c == '"':
            i += 1
            while True:
                if i >= n:
                    raise ValueError("unterminated double quote")
                d = line[i]
                if d == '"':
                    i += 1
                    break
                if d == "\\" and i + 1 < n:
                    e = line[i + 1]
                    if e == "\n":
                        i += 2
                        continue
                    if e in _DQ_ESCAPES:
                        buf.append(e)
                        i += 2
                        continue
                    buf.append("\\" + e)
                    i += 2
                    continue
                buf.append(d)
                i += 1
            in_word = True
            at_start = False
        else:
            buf.append(c)
            in_word = True
            at_start = False
            i += 1
    flush()
    return tokens
