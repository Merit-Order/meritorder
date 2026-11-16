"""Plausible but buggy c3-shell-tokens: empty quoted words are dropped, and backslash-newline inside double quotes
is kept instead of removed."""

OPERATORS = ["||", "&&", ">>", "|", "&", ";", ">", "<"]


def tokenize(line):
    tokens = []
    current = ""
    quote = None
    i = 0
    word_start = True

    def finish():
        nonlocal current
        if current:
            tokens.append(("word", current))
        current = ""

    while i < len(line):
        ch = line[i]
        if quote == "'":
            if ch == "'":
                quote = None
            else:
                current += ch
            i += 1
            continue
        if quote == '"':
            if ch == '"':
                quote = None
                i += 1
            elif ch == "\\" and i + 1 < len(line) and line[i + 1] in '"\\$`':
                current += line[i + 1]
                i += 2
            else:
                current += ch
                i += 1
            continue
        if ch in " \t\n":
            finish()
            word_start = True
            i += 1
            continue
        if ch == "\\":
            if i + 1 == len(line):
                raise ValueError("trailing backslash")
            if line[i + 1] != "\n":
                current += line[i + 1]
                word_start = False
            i += 2
            continue
        if ch == "#" and word_start:
            while i < len(line) and line[i] != "\n":
                i += 1
            continue
        op = next((op for op in OPERATORS if line.startswith(op, i)), None)
        if op:
            finish()
            tokens.append(("op", op))
            i += len(op)
            word_start = True
            continue
        if ch in "'\"":
            quote = ch
        else:
            current += ch
        word_start = False
        i += 1
    if quote:
        raise ValueError("unterminated quote")
    finish()
    return tokens
