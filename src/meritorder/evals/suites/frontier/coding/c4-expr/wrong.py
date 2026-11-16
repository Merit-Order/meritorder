"""Plausible but wrong c4-expr solution: unary minus binds tighter than ^, and end-of-text errors point
after the last token instead of at len(text)."""

from fractions import Fraction


class ExprError(Exception):
    def __init__(self, message, position):
        super().__init__(f"{message} at position {position}")
        self.position = position


class ParseError(ExprError):
    pass


class EvalError(ExprError):
    pass


WHITESPACE = " \t\n\r"
DIGITS = "0123456789"
LETTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_"
SINGLE = set("+-*%^(),")
FUNCTIONS = {"abs", "min", "max"}
MAX_EXPONENT = 4096


def _tokenize(text):
    """All tokens as (kind, value, position). Invalid characters become 'bad' tokens; the parser decides
    whether it ever reaches them."""
    tokens, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if c in WHITESPACE:
            i += 1
        elif c in DIGITS or (c == "." and i + 1 < n and text[i + 1] in DIGITS):
            j = i
            while j < n and text[j] in DIGITS:
                j += 1
            if j < n and text[j] == ".":
                j += 1
                while j < n and text[j] in DIGITS:
                    j += 1
            tokens.append(("num", Fraction(text[i:j] if not text[i:j].endswith(".") else text[i:j - 1]), i))
            i = j
        elif c in LETTERS:
            j = i + 1
            while j < n and (text[j] in LETTERS or text[j] in DIGITS):
                j += 1
            tokens.append(("name", text[i:j], i))
            i = j
        elif c == "/":
            if i + 1 < n and text[i + 1] == "/":
                tokens.append(("op", "//", i))
                i += 2
            else:
                tokens.append(("op", "/", i))
                i += 1
        elif c in SINGLE:
            tokens.append(("op", c, i))
            i += 1
        else:
            tokens.append(("bad", c, i))
            i += 1
    return tokens


class _Parser:
    def __init__(self, text):
        self.text = text
        self.tokens = _tokenize(text)
        self.i = 0

    def peek(self):
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def fail(self):
        tok = self.peek()
        raise ParseError("unexpected " + ("end of input" if tok is None else repr(tok[1])),
                         len(self.text.rstrip()) if tok is None else tok[2])

    def at_op(self, *ops):
        tok = self.peek()
        return tok is not None and tok[0] == "op" and tok[1] in ops

    def take(self):
        tok = self.peek()
        self.i += 1
        return tok

    def expect(self, op):
        if not self.at_op(op):
            self.fail()
        return self.take()

    def parse(self):
        node = self.expression()
        if self.peek() is not None:
            self.fail()
        return node

    def expression(self):
        node = self.term()
        while self.at_op("+", "-"):
            op = self.take()
            node = ("bin", op[1], node, self.term(), op[2])
        return node

    def term(self):
        node = self.unary()
        while self.at_op("*", "/", "//", "%"):
            op = self.take()
            node = ("bin", op[1], node, self.unary(), op[2])
        return node

    def unary(self):
        return self.power()

    def power(self):
        base = self.signed()
        if self.at_op("^"):
            op = self.take()
            return ("bin", "^", base, self.power(), op[2])
        return base

    def signed(self):
        if self.at_op("-", "+"):
            op = self.take()
            return ("neg" if op[1] == "-" else "pos", self.signed())
        return self.primary()

    def primary(self):
        tok = self.peek()
        if tok is None:
            self.fail()
        kind, value, pos = tok
        if kind == "num":
            self.take()
            return ("num", value)
        if kind == "name":
            self.take()
            if self.at_op("("):
                self.take()
                args = [self.expression()]
                while self.at_op(","):
                    self.take()
                    args.append(self.expression())
                self.expect(")")
                return ("call", value, args, pos)
            return ("var", value, pos)
        if kind == "op" and value == "(":
            self.take()
            node = self.expression()
            self.expect(")")
            return node
        self.fail()


def _eval(node, variables):
    kind = node[0]
    if kind == "num":
        return node[1]
    if kind == "var":
        if node[1] not in variables:
            raise EvalError(f"unknown variable {node[1]!r}", node[2])
        return Fraction(variables[node[1]])
    if kind == "neg":
        return -_eval(node[1], variables)
    if kind == "pos":
        return _eval(node[1], variables)
    if kind == "call":
        _, name, args, pos = node
        if name not in FUNCTIONS:
            raise EvalError(f"unknown function {name!r}", pos)
        if name == "abs" and len(args) != 1:
            raise EvalError("abs takes exactly one argument", pos)
        values = [_eval(a, variables) for a in args]
        if name == "abs":
            return abs(values[0])
        return min(values) if name == "min" else max(values)
    _, op, left, right, pos = node
    a = _eval(left, variables)
    b = _eval(right, variables)
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op in ("/", "//", "%"):
        if b == 0:
            raise EvalError("division by zero", pos)
        if op == "/":
            return a / b
        if op == "//":
            return Fraction(a // b)
        return a - b * (a // b)
    # op == "^"
    if b.denominator != 1:
        raise EvalError("exponent is not a whole number", pos)
    if abs(b) > MAX_EXPONENT:
        raise EvalError("exponent too large", pos)
    if a == 0 and b < 0:
        raise EvalError("zero to a negative power", pos)
    return Fraction(a ** int(b))


def evaluate(text, variables=None):
    tree = _Parser(text).parse()
    return _eval(tree, variables or {})
