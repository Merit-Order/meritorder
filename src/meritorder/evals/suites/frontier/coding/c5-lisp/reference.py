"""Reference solution for c5-lisp: a small Scheme-like interpreter with closures and proper tail calls."""

import sys
from fractions import Fraction


class LispError(Exception):
    pass


class Symbol(str):
    __slots__ = ()


_SYMBOLS = {}


def sym(name):
    s = _SYMBOLS.get(name)
    if s is None:
        s = _SYMBOLS[name] = Symbol(name)
    return s


class LStr:
    __slots__ = ("s",)

    def __init__(self, s):
        self.s = s


class Pair:
    __slots__ = ("car", "cdr", "_items")

    def __init__(self, car, cdr):
        self.car, self.cdr, self._items = car, cdr, None


class _Nil:
    __slots__ = ()


class _Void:
    __slots__ = ()


NIL = _Nil()
VOID = _Void()
UNASSIGNED = object()


class Closure:
    __slots__ = ("params", "rest", "body", "env")

    def __init__(self, params, rest, body, env):
        self.params, self.rest, self.body, self.env = params, rest, body, env


class Builtin:
    __slots__ = ("name", "fn", "lo", "hi")

    def __init__(self, name, fn, lo, hi):
        self.name, self.fn, self.lo, self.hi = name, fn, lo, hi


class Env:
    __slots__ = ("vars", "parent")

    def __init__(self, vars, parent):
        self.vars, self.parent = vars, parent

    def lookup(self, s):
        e = self
        while e is not None:
            v = e.vars.get(s, e)
            if v is not e:
                if v is UNASSIGNED:
                    raise LispError(f"variable used before its letrec init: {s}")
                return v
            e = e.parent
        raise LispError(f"unbound symbol: {s}")

    def frame_of(self, s):
        e = self
        while e is not None:
            if s in e.vars:
                return e
            e = e.parent
        raise LispError(f"set! of an unbound symbol: {s}")


# ---------------------------------------------------------------- reader

_DELIMS = set("()'\";") | set(" \t\n\r")
_ESC = {'"': '"', "\\": "\\", "n": "\n", "t": "\t"}
_DOT = object()
_LP, _RP, _QT = object(), object(), object()
_PUNCT = {"(": _LP, ")": _RP, "'": _QT}


def tokenize(src):
    toks, i, n = [], 0, len(src)
    while i < n:
        c = src[i]
        if c in " \t\n\r":
            i += 1
        elif c == ";":
            while i < n and src[i] != "\n":
                i += 1
        elif c in "()'":
            toks.append(_PUNCT[c])
            i += 1
        elif c == '"':
            i += 1
            buf = []
            while True:
                if i >= n:
                    raise LispError("unterminated string")
                c = src[i]
                if c == '"':
                    i += 1
                    break
                if c == "\\":
                    if i + 1 >= n:
                        raise LispError("unterminated string")
                    e = src[i + 1]
                    if e not in _ESC:
                        raise LispError(f"bad string escape \\{e}")
                    buf.append(_ESC[e])
                    i += 2
                else:
                    buf.append(c)
                    i += 1
            toks.append(LStr("".join(buf)))
        else:
            j = i
            while j < n and src[j] not in _DELIMS:
                j += 1
            toks.append(_atom(src[i:j]))
            i = j
    return toks


def _atom(t):
    body = t[1:] if t[0] in "+-" else t
    if body and all("0" <= ch <= "9" for ch in body):
        return int(t)
    if t == "#t":
        return True
    if t == "#f":
        return False
    if t.startswith("#"):
        raise LispError(f"bad token {t}")
    if t == ".":
        return _DOT
    return sym(t)


def read_all(src):
    toks = tokenize(src)
    pos = 0
    out = []

    def read():
        nonlocal pos
        if pos >= len(toks):
            raise LispError("unexpected end of input")
        t = toks[pos]
        pos += 1
        if t is _LP:
            items, tail = [], NIL
            while True:
                if pos >= len(toks):
                    raise LispError("missing )")
                t = toks[pos]
                if t is _RP:
                    pos += 1
                    break
                if t is _DOT:
                    if not items:
                        raise LispError("bad dotted list")
                    pos += 1
                    tail = read()
                    if pos >= len(toks) or toks[pos] is not _RP:
                        raise LispError("bad dotted list")
                    pos += 1
                    break
                items.append(read())
            for item in reversed(items):
                tail = Pair(item, tail)
            return tail
        if t is _RP:
            raise LispError("unexpected )")
        if t is _QT:
            return Pair(sym("quote"), Pair(read(), NIL))
        if t is _DOT:
            raise LispError("unexpected .")
        return t

    while pos < len(toks):
        out.append(read())
    return out


# ---------------------------------------------------------------- values

def is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def is_num(x):
    return is_int(x) or type(x) is Fraction


def norm(x):
    if type(x) is Fraction and x.denominator == 1:
        return x.numerator
    return x


def items_of(x, what="list"):
    """Elements of a proper list as a tuple (cached on source pairs, which never change)."""
    if x is NIL:
        return ()
    if type(x) is not Pair:
        raise LispError(f"not a proper {what}")
    if x._items is not None:
        return x._items
    out, p = [], x
    while type(p) is Pair:
        out.append(p.car)
        p = p.cdr
    if p is not NIL:
        raise LispError(f"not a proper {what}")
    x._items = tuple(out)
    return x._items


def is_list(x):
    while type(x) is Pair:
        x = x.cdr
    return x is NIL


def make_list(xs, tail=NIL):
    for x in reversed(xs):
        tail = Pair(x, tail)
    return tail


def to_str(x):
    if x is True:
        return "#t"
    if x is False:
        return "#f"
    if is_int(x):
        return str(x)
    if type(x) is Fraction:
        return f"{x.numerator}/{x.denominator}"
    if type(x) is Symbol:
        return str(x)
    if type(x) is LStr:
        s = x.s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\t", "\\t")
        return f'"{s}"'
    if x is NIL:
        return "()"
    if x is VOID:
        return "#<void>"
    if type(x) is Pair:
        parts = []
        while type(x) is Pair:
            parts.append(to_str(x.car))
            x = x.cdr
        if x is not NIL:
            parts.append(".")
            parts.append(to_str(x))
        return "(" + " ".join(parts) + ")"
    if isinstance(x, (Closure, Builtin)):
        return "#<procedure>"
    raise AssertionError(repr(x))


def eqp(a, b):
    if is_num(a) and is_num(b):
        return a == b
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if type(a) is Symbol and type(b) is Symbol:
        return a == b
    return a is b


def equalp(a, b):
    while type(a) is Pair and type(b) is Pair:
        if not equalp(a.car, b.car):
            return False
        a, b = a.cdr, b.cdr
    if type(a) is LStr and type(b) is LStr:
        return a.s == b.s
    return eqp(a, b)


# ---------------------------------------------------------------- builtins

def _nums(name, args):
    for a in args:
        if not is_num(a):
            raise LispError(f"{name}: not a number: {to_str(a)}")
    return args


def _ints(name, args):
    for a in args:
        if not is_int(a):
            raise LispError(f"{name}: not an integer: {to_str(a)}")
    return args


def b_add(args):
    t = 0
    for a in _nums("+", args):
        t += a
    return norm(t)


def b_mul(args):
    t = 1
    for a in _nums("*", args):
        t *= a
    return norm(t)


def b_sub(args):
    _nums("-", args)
    if len(args) == 1:
        return -args[0]
    t = args[0]
    for a in args[1:]:
        t -= a
    return norm(t)


def b_div(args):
    _nums("/", args)
    if len(args) == 1:
        args = [1, args[0]]
    t = Fraction(args[0])
    for a in args[1:]:
        if a == 0:
            raise LispError("division by zero")
        t /= a
    return norm(t)


def _quot(a, b):
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b > 0) else -q


def b_quotient(args):
    a, b = _ints("quotient", args)
    if b == 0:
        raise LispError("division by zero")
    return _quot(a, b)


def b_remainder(args):
    a, b = _ints("remainder", args)
    if b == 0:
        raise LispError("division by zero")
    return a - b * _quot(a, b)


def b_modulo(args):
    a, b = _ints("modulo", args)
    if b == 0:
        raise LispError("division by zero")
    return a % b


def _compare(name, op):
    def f(args):
        _nums(name, args)
        return all(op(args[i], args[i + 1]) for i in range(len(args) - 1))
    return f


def b_car(args):
    if type(args[0]) is not Pair:
        raise LispError("car: not a pair")
    return args[0].car


def b_cdr(args):
    if type(args[0]) is not Pair:
        raise LispError("cdr: not a pair")
    return args[0].cdr


def b_length(args):
    return len(items_of(args[0]))


def b_append(args):
    out = []
    for a in args:
        out.extend(items_of(a))
    return make_list(out)


def b_reverse(args):
    return make_list(list(reversed(items_of(args[0]))))


def _need_proc(name, f):
    if not isinstance(f, (Closure, Builtin)):
        raise LispError(f"{name}: not a procedure")


def b_apply(args):
    _need_proc("apply", args[0])
    return call(args[0], list(items_of(args[1])))


def b_map(args):
    _need_proc("map", args[0])
    return make_list([call(args[0], [x]) for x in items_of(args[1])])


def _strs(name, args):
    for a in args:
        if type(a) is not LStr:
            raise LispError(f"{name}: not a string")
    return args


def b_number_to_string(args):
    _nums("number->string", args)
    return LStr(to_str(args[0]))


def b_symbol_to_string(args):
    if type(args[0]) is not Symbol:
        raise LispError("symbol->string: not a symbol")
    return LStr(str(args[0]))


def b_abs(args):
    _nums("abs", args)
    return abs(args[0])


BUILTINS = {
    "+": (b_add, 0, None), "*": (b_mul, 0, None), "-": (b_sub, 1, None), "/": (b_div, 1, None),
    "quotient": (b_quotient, 2, 2), "remainder": (b_remainder, 2, 2), "modulo": (b_modulo, 2, 2),
    "abs": (b_abs, 1, 1),
    "=": (_compare("=", lambda a, b: a == b), 2, None), "<": (_compare("<", lambda a, b: a < b), 2, None),
    ">": (_compare(">", lambda a, b: a > b), 2, None), "<=": (_compare("<=", lambda a, b: a <= b), 2, None),
    ">=": (_compare(">=", lambda a, b: a >= b), 2, None),
    "number?": (lambda a: is_num(a[0]), 1, 1),
    "integer?": (lambda a: is_int(a[0]), 1, 1),
    "boolean?": (lambda a: isinstance(a[0], bool), 1, 1),
    "symbol?": (lambda a: type(a[0]) is Symbol, 1, 1),
    "string?": (lambda a: type(a[0]) is LStr, 1, 1),
    "procedure?": (lambda a: isinstance(a[0], (Closure, Builtin)), 1, 1),
    "null?": (lambda a: a[0] is NIL, 1, 1),
    "pair?": (lambda a: type(a[0]) is Pair, 1, 1),
    "list?": (lambda a: is_list(a[0]), 1, 1),
    "not": (lambda a: a[0] is False, 1, 1),
    "eq?": (lambda a: eqp(a[0], a[1]), 2, 2),
    "equal?": (lambda a: equalp(a[0], a[1]), 2, 2),
    "cons": (lambda a: Pair(a[0], a[1]), 2, 2),
    "car": (b_car, 1, 1), "cdr": (b_cdr, 1, 1),
    "list": (lambda a: make_list(a), 0, None),
    "length": (b_length, 1, 1), "append": (b_append, 0, None), "reverse": (b_reverse, 1, 1),
    "apply": (b_apply, 2, 2), "map": (b_map, 2, 2),
    "string-append": (lambda a: LStr("".join(s.s for s in _strs("string-append", a))), 0, None),
    "string-length": (lambda a: len(_strs("string-length", a)[0].s), 1, 1),
    "number->string": (b_number_to_string, 1, 1),
    "symbol->string": (b_symbol_to_string, 1, 1),
    "string->symbol": (lambda a: sym(_strs("string->symbol", a)[0].s), 1, 1),
}


def call_builtin(f, args):
    n = len(args)
    if n < f.lo or (f.hi is not None and n > f.hi):
        raise LispError(f"{f.name}: wrong number of arguments ({n})")
    return f.fn(args)


def bind(f, args):
    n, k = len(args), len(f.params)
    if n < k or (f.rest is None and n > k):
        raise LispError(f"wrong number of arguments: expected {k}{'+' if f.rest else ''}, got {n}")
    vars = dict(zip(f.params, args))
    if f.rest is not None:
        vars[f.rest] = make_list(args[k:])
    return Env(vars, f.env)


def call(f, args):
    """Call a procedure from Python (used by apply and map)."""
    if type(f) is Builtin:
        return call_builtin(f, args)
    env = bind(f, args)
    return eval_body(f.body, env)


def eval_body(body, env):
    for x in body[:-1]:
        ev(x, env)
    return ev(body[-1], env)


# ---------------------------------------------------------------- evaluator

S_QUOTE, S_IF, S_DEFINE, S_SET, S_LAMBDA, S_BEGIN = map(sym, ["quote", "if", "define", "set!", "lambda", "begin"])
S_LET, S_LETSTAR, S_LETREC, S_COND, S_AND, S_OR, S_ELSE = map(sym, ["let", "let*", "letrec", "cond", "and", "or", "else"])
SPECIAL = {S_QUOTE, S_IF, S_DEFINE, S_SET, S_LAMBDA, S_BEGIN, S_LET, S_LETSTAR, S_LETREC, S_COND, S_AND, S_OR}


def parse_params(spec):
    params, p = [], spec
    while type(p) is Pair:
        if type(p.car) is not Symbol:
            raise LispError("parameter is not a symbol")
        params.append(p.car)
        p = p.cdr
    rest = None
    if p is not NIL:
        if type(p) is not Symbol:
            raise LispError("bad parameter list")
        rest = p
    names = params + ([rest] if rest is not None else [])
    if len(set(names)) != len(names):
        raise LispError("duplicate parameter")
    return tuple(params), rest


def make_lambda(spec, body, env):
    if not body:
        raise LispError("lambda needs a body")
    params, rest = parse_params(spec)
    return Closure(params, rest, tuple(body), env)


def parse_bindings(b):
    out = []
    for pair in items_of(b, "binding list"):
        parts = items_of(pair, "binding")
        if len(parts) != 2 or type(parts[0]) is not Symbol:
            raise LispError("bad binding")
        out.append(parts)
    return out


def ev(x, env):
    while True:
        t = type(x)
        if t is Symbol:
            return env.lookup(x)
        if t is not Pair:
            if x is NIL:
                raise LispError("cannot evaluate ()")
            return x
        form = items_of(x, "form")
        op = form[0]
        if type(op) is Symbol and op in SPECIAL:
            n = len(form)
            if op is S_QUOTE:
                if n != 2:
                    raise LispError("bad quote")
                return form[1]
            if op is S_IF:
                if n not in (3, 4):
                    raise LispError("bad if")
                if ev(form[1], env) is not False:
                    x = form[2]
                elif n == 4:
                    x = form[3]
                else:
                    return VOID
                continue
            if op is S_DEFINE:
                if n < 3:
                    raise LispError("bad define")
                target = form[1]
                if type(target) is Symbol:
                    if n != 3:
                        raise LispError("bad define")
                    env.vars[target] = ev(form[2], env)
                elif type(target) is Pair and type(target.car) is Symbol:
                    env.vars[target.car] = make_lambda(target.cdr, form[2:], env)
                else:
                    raise LispError("bad define")
                return VOID
            if op is S_SET:
                if n != 3 or type(form[1]) is not Symbol:
                    raise LispError("bad set!")
                frame = env.frame_of(form[1])
                frame.vars[form[1]] = ev(form[2], env)
                return VOID
            if op is S_LAMBDA:
                if n < 3:
                    raise LispError("bad lambda")
                return make_lambda(form[1], form[2:], env)
            if op is S_BEGIN:
                if n == 1:
                    return VOID
                for e in form[1:-1]:
                    ev(e, env)
                x = form[-1]
                continue
            if op is S_LET:
                if n >= 2 and type(form[1]) is Symbol:
                    if n < 4:
                        raise LispError("bad named let")
                    name, bindings, body = form[1], parse_bindings(form[2]), form[3:]
                    args = [ev(e, env) for _, e in bindings]
                    loop_env = Env({}, env)
                    proc = make_lambda(make_list([s for s, _ in bindings]), body, loop_env)
                    loop_env.vars[name] = proc
                    env = bind(proc, args)
                else:
                    if n < 3:
                        raise LispError("bad let")
                    bindings, body = parse_bindings(form[1]), form[2:]
                    names = [s for s, _ in bindings]
                    if len(set(names)) != len(names):
                        raise LispError("duplicate let variable")
                    env = Env({s: ev(e, env) for s, e in bindings}, env)
                for e in body[:-1]:
                    ev(e, env)
                x = body[-1]
                continue
            if op is S_LETSTAR:
                if n < 3:
                    raise LispError("bad let*")
                for s, e in parse_bindings(form[1]):
                    env = Env({s: ev(e, env)}, env)
                env = Env({}, env)
                for e in form[2:-1]:
                    ev(e, env)
                x = form[-1]
                continue
            if op is S_LETREC:
                if n < 3:
                    raise LispError("bad letrec")
                bindings = parse_bindings(form[1])
                env = Env({s: UNASSIGNED for s, _ in bindings}, env)
                for s, e in bindings:
                    env.vars[s] = ev(e, env)
                for e in form[2:-1]:
                    ev(e, env)
                x = form[-1]
                continue
            if op is S_COND:
                chosen = None
                for clause in form[1:]:
                    parts = items_of(clause, "cond clause")
                    if not parts:
                        raise LispError("empty cond clause")
                    if parts[0] is S_ELSE:
                        if len(parts) < 2:
                            raise LispError("empty else clause")
                        chosen = parts[1:]
                        break
                    v = ev(parts[0], env)
                    if v is not False:
                        if len(parts) == 1:
                            return v
                        chosen = parts[1:]
                        break
                if chosen is None:
                    return VOID
                for e in chosen[:-1]:
                    ev(e, env)
                x = chosen[-1]
                continue
            if op is S_AND or op is S_OR:
                if n == 1:
                    return op is S_AND
                stop = False
                for e in form[1:-1]:
                    v = ev(e, env)
                    if (v is False) if op is S_AND else (v is not False):
                        stop = True
                        break
                if stop:
                    return v
                x = form[-1]
                continue
        f = ev(op, env)
        args = [ev(a, env) for a in form[1:]]
        tf = type(f)
        if tf is Builtin:
            return call_builtin(f, args)
        if tf is not Closure:
            raise LispError(f"not a procedure: {to_str(f)}")
        env = bind(f, args)
        body = f.body
        for e in body[:-1]:
            ev(e, env)
        x = body[-1]


def global_env():
    return Env({sym(name): Builtin(name, fn, lo, hi) for name, (fn, lo, hi) in BUILTINS.items()}, None)


def run(source):
    forms = read_all(source)
    env = global_env()
    old = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old, 20000))
    try:
        value = VOID
        for form in forms:
            value = ev(form, env)
        return to_str(value)
    except RecursionError:
        raise LispError("recursion too deep") from None
    finally:
        sys.setrecursionlimit(old)
