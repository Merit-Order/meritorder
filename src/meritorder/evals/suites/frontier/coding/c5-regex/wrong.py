"""Buggy solution for c5-regex: an empty mandatory repetition also ends the loop, and after an empty match
findall/sub simply move one character on instead of retrying a non-empty match at the same position."""

import sys

_DIGIT = frozenset("0123456789")
_WORD = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
_SPACE = frozenset(" \t\n\r\f\v")
_CLASS_ESCAPES = {"d": (_DIGIT, False), "D": (_DIGIT, True), "w": (_WORD, False), "W": (_WORD, True),
                  "s": (_SPACE, False), "S": (_SPACE, True)}
_CHAR_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "f": "\f", "v": "\v"}


class _Parser:
    def __init__(self, pattern):
        self.p = pattern
        self.i = 0
        self.ngroups = 0

    def error(self, msg):
        raise ValueError(f"{msg} at position {self.i}")

    def peek(self, k=0):
        j = self.i + k
        return self.p[j] if j < len(self.p) else None

    def parse(self):
        node = self.alternation()
        if self.i < len(self.p):  # only a stray ')' can stop the top level early
            self.error("unbalanced parenthesis")
        return node

    def alternation(self):
        branches = [self.sequence()]
        while self.peek() == "|":
            self.i += 1
            branches.append(self.sequence())
        return branches[0] if len(branches) == 1 else ("alt", branches)

    def escape(self, in_class):
        """At a backslash. Returns ('lit', ch) or ('set', chars, negated)."""
        self.i += 1
        c = self.peek()
        if c is None:
            self.error("bad escape (end of pattern)")
        self.i += 1
        if c in _CLASS_ESCAPES:
            chars, neg = _CLASS_ESCAPES[c]
            return ("set", chars, neg)
        if c in _CHAR_ESCAPES:
            return ("lit", _CHAR_ESCAPES[c])
        if c.isascii() and c.isalnum():
            self.i -= 2
            self.error(f"bad escape \\{c}")
        return ("lit", c)

    def quantifier(self):
        """If a quantifier starts here, consume it and return (min, max|None); otherwise None."""
        c = self.peek()
        if c == "*":
            self.i += 1
            return (0, None)
        if c == "+":
            self.i += 1
            return (1, None)
        if c == "?":
            self.i += 1
            return (0, 1)
        if c == "{":
            j = self.i + 1
            k = j
            while k < len(self.p) and self.p[k].isdigit() and self.p[k].isascii():
                k += 1
            lo_s = self.p[j:k]
            if k < len(self.p) and self.p[k] == "}":
                if not lo_s:
                    return None
                self.i = k + 1
                return (int(lo_s), int(lo_s))
            if k < len(self.p) and self.p[k] == ",":
                m = k + 1
                while m < len(self.p) and self.p[m].isdigit() and self.p[m].isascii():
                    m += 1
                hi_s = self.p[k + 1:m]
                if m < len(self.p) and self.p[m] == "}":
                    lo = int(lo_s) if lo_s else 0
                    hi = int(hi_s) if hi_s else None
                    if hi is not None and lo > hi:
                        self.error("min repeat greater than max repeat")
                    self.i = m + 1
                    return (lo, hi)
            return None
        return None

    def sequence(self):
        items = []
        while True:
            c = self.peek()
            if c is None or c in "|)":
                break
            start = self.i
            if self.quantifier() is not None:
                self.i = start
                self.error("nothing to repeat")
            atom, repeatable = self.atom()
            q = self.quantifier()
            if q is not None:
                if not repeatable:
                    self.i = start + 1
                    self.error("nothing to repeat")
                greedy = True
                if self.peek() == "?":
                    self.i += 1
                    greedy = False
                if self.quantifier() is not None:
                    self.error("multiple repeat")
                atom = ("rep", atom, q[0], q[1], greedy)
            items.append(atom)
        return items[0] if len(items) == 1 else ("seq", items)

    def atom(self):
        c = self.peek()
        if c == "(":
            self.i += 1
            idx = None
            if self.peek() == "?":
                if self.peek(1) != ":":
                    self.error("unsupported group extension")
                self.i += 2
            else:
                idx = self.ngroups
                self.ngroups += 1
            body = self.alternation()
            if self.peek() != ")":
                self.error("missing ), unterminated subpattern")
            self.i += 1
            return (("group", idx, body) if idx is not None else body), True
        if c == "[":
            return self.char_class(), True
        if c == ".":
            self.i += 1
            return ("any",), True
        if c == "^":
            self.i += 1
            return ("bol",), False
        if c == "$":
            self.i += 1
            return ("eol",), False
        if c == "\\":
            e = self.escape(False)
            if e[0] == "lit":
                return ("set", frozenset(e[1]), False), True
            return e, True
        self.i += 1
        return ("set", frozenset(c), False), True

    def char_class(self):
        start = self.i
        self.i += 1
        negated = False
        if self.peek() == "^":
            negated = True
            self.i += 1
        first = self.i
        singles = set()
        ranges = []
        sets = []
        while True:
            c = self.peek()
            if c is None:
                self.i = start
                self.error("unterminated character set")
            if c == "]" and self.i != first:
                self.i += 1
                break
            item = self.escape(True) if c == "\\" else ("lit", c)
            if c != "\\":
                self.i += 1
            if self.peek() == "-":
                self.i += 1
                nxt = self.peek()
                if nxt is None:
                    self.i = start
                    self.error("unterminated character set")
                if nxt == "]":
                    self._add(item, singles, sets)
                    singles.add("-")
                    self.i += 1
                    break
                end = self.escape(True) if nxt == "\\" else ("lit", nxt)
                if nxt != "\\":
                    self.i += 1
                if item[0] != "lit" or end[0] != "lit" or end[1] < item[1]:
                    self.error("bad character range")
                ranges.append((item[1], end[1]))
            else:
                self._add(item, singles, sets)
        return ("class", frozenset(singles), tuple(ranges), tuple(sets), negated)

    @staticmethod
    def _add(item, singles, sets):
        if item[0] == "lit":
            singles.add(item[1])
        else:
            sets.append((item[1], item[2]))


def _in_class(node, ch):
    _, singles, ranges, sets, negated = node
    hit = ch in singles or any(lo <= ch <= hi for lo, hi in ranges) or any((ch in s) != neg for s, neg in sets)
    return hit != negated


class _Program:
    def __init__(self, pattern):
        parser = _Parser(pattern)
        self.root = parser.parse()
        self.ngroups = parser.ngroups

    def run(self, text, start, accept):
        """Highest-priority match starting at ``start`` whose end satisfies ``accept(end)``: (end, caps) or None."""
        n = len(text)

        def m(node, i, caps, k):
            t = node[0]
            if t == "set":
                if i < n and (text[i] in node[1]) != node[2]:
                    return k(i + 1, caps)
                return None
            if t == "class":
                if i < n and _in_class(node, text[i]):
                    return k(i + 1, caps)
                return None
            if t == "any":
                if i < n and text[i] != "\n":
                    return k(i + 1, caps)
                return None
            if t == "bol":
                return k(i, caps) if i == 0 else None
            if t == "eol":
                return k(i, caps) if i == n or (i == n - 1 and text[i] == "\n") else None
            if t == "seq":
                items = node[1]
                last = len(items)

                def go(j, i, caps):
                    if j == last:
                        return k(i, caps)
                    return m(items[j], i, caps, lambda i2, c2: go(j + 1, i2, c2))
                return go(0, i, caps)
            if t == "alt":
                for branch in node[1]:
                    r = m(branch, i, caps, k)
                    if r is not None:
                        return r
                return None
            if t == "group":
                idx = node[1]
                return m(node[2], i, caps, lambda j, c: k(j, c[:idx] + ((i, j),) + c[idx + 1:]))
            if t == "rep":
                _, body, lo, hi, greedy = node

                def rep(count, i, caps, last_opt):
                    if count < lo:
                        return m(body, i, caps, lambda j, c: rep(count + 1, j, c, i))
                    more = (hi is None or count < hi) and i != last_opt
                    if greedy:
                        if more:
                            r = m(body, i, caps, lambda j, c: rep(count + 1, j, c, i))
                            if r is not None:
                                return r
                        return k(i, caps)
                    r = k(i, caps)
                    if r is not None or not more:
                        return r
                    return m(body, i, caps, lambda j, c: rep(count + 1, j, c, i))
                return rep(0, i, caps, None)
            raise AssertionError(t)

        caps0 = (None,) * self.ngroups
        old = sys.getrecursionlimit()
        sys.setrecursionlimit(max(old, 20000))
        try:
            return m(self.root, start, caps0, lambda i, c: (i, c) if accept(i) else None)
        finally:
            sys.setrecursionlimit(old)


class Match:
    def __init__(self, text, start, end, caps):
        self._text = text
        self._spans = ((start, end),) + caps

    def _span(self, g):
        if not isinstance(g, int) or isinstance(g, bool) or not 0 <= g < len(self._spans):
            raise IndexError("no such group")
        return self._spans[g]

    def span(self, g=0):
        s = self._span(g)
        return (-1, -1) if s is None else s

    def group(self, g=0):
        s = self._span(g)
        return None if s is None else self._text[s[0]:s[1]]

    def groups(self):
        return tuple(self.group(g) for g in range(1, len(self._spans)))

    def __repr__(self):
        return f"<Match span={self.span()} match={self.group()!r}>"


def _search_from(prog, text, pos, must_advance):
    for start in range(pos, len(text) + 1):
        if must_advance and start == pos:
            r = prog.run(text, start, lambda e, s=start: e != s)
        else:
            r = prog.run(text, start, lambda e: True)
        if r is not None:
            return Match(text, start, r[0], r[1])
    return None


def _iter(prog, text):
    pos, must_advance = 0, False
    while pos <= len(text):
        m = _search_from(prog, text, pos, must_advance)
        if m is None:
            return
        yield m
        s, e = m.span()
        pos = e + 1 if s == e else e


def search(pattern, text):
    return _search_from(_Program(pattern), text, 0, False)


def fullmatch(pattern, text):
    prog = _Program(pattern)
    r = prog.run(text, 0, lambda e: e == len(text))
    return None if r is None else Match(text, 0, r[0], r[1])


def findall(pattern, text):
    prog = _Program(pattern)
    out = []
    for m in _iter(prog, text):
        if prog.ngroups == 0:
            out.append(m.group())
        elif prog.ngroups == 1:
            out.append(m.group(1) or "")
        else:
            out.append(tuple(g or "" for g in m.groups()))
    return out


def sub(pattern, repl, text, count=0):
    prog = _Program(pattern)
    out, last, done = [], 0, 0
    for m in _iter(prog, text):
        if count and done == count:
            break
        s, e = m.span()
        out.append(text[last:s])
        out.append(repl)
        last, done = e, done + 1
    out.append(text[last:])
    return "".join(out)
