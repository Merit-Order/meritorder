"""Reference solution: a streaming JSON tokenizer that validates structure and reports the earliest error."""

_WS = " \t\n\r"
_DIGITS = "0123456789"
_HEX = "0123456789abcdefABCDEF"
_ESCAPES = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}
_LITERALS = {"t": ("true", "bool", True), "f": ("false", "bool", False), "n": ("null", "null", None)}

# What the parser expects next (outside of a token).
_VALUE = 0          # a value: at the top level, after ':' and after ',' in an array
_VALUE_OR_END = 1   # right after '['
_KEY_OR_END = 2     # right after '{'
_KEY = 3            # after ',' in an object
_COLON = 4          # after a key
_COMMA_OR_END = 5   # after a value inside a container
_DONE = 6           # after the top-level value

# Number states; the ones in _NUM_FINAL may end the number.
_N_MINUS, _N_ZERO, _N_INT, _N_DOT, _N_FRAC, _N_E, _N_ESIGN, _N_EXP = range(8)
_NUM_FINAL = {_N_ZERO, _N_INT, _N_FRAC, _N_EXP}


class JSONStreamError(ValueError):
    def __init__(self, message: str, offset: int, line: int, column: int):
        super().__init__(f"{message} at line {line} column {column} (offset {offset})")
        self.message, self.offset, self.line, self.column = message, offset, line, column


class Tokenizer:
    def __init__(self) -> None:
        self._pos, self._line, self._col = 0, 1, 1
        self._stack: list[str] = []
        self._expect = _VALUE
        self._lex: str | None = None   # None, "string", "number" or "literal"
        self._start = 0                # offset where the current token began
        self._error: JSONStreamError | None = None
        self._closed = False
        self._out: list[tuple] = []
        # string state
        self._parts: list[str] = []
        self._is_key = False
        self._esc: str | None = None   # None, "\\" (after a backslash) or "u" (reading hex digits)
        self._hex = ""
        self._high: int | None = None  # a high surrogate escape waiting for its low half
        # number / literal state
        self._num: list[str] = []
        self._nstate = _N_MINUS
        self._lit = ""
        self._lit_i = 0
        self._lit_token: tuple = ()

    # ---------------------------------------------------------------- public API
    def feed(self, chunk: str) -> list[tuple]:
        self._check_usable()
        self._out = []
        for ch in chunk:
            self._step(ch)
            self._pos += 1
            if ch == "\n":
                self._line += 1
                self._col = 1
            else:
                self._col += 1
        return self._out

    def close(self) -> list[tuple]:
        self._check_usable()
        self._out = []
        if self._lex == "number" and self._nstate in _NUM_FINAL:
            self._end_number()
        if self._lex is not None or self._expect != _DONE:
            self._fail("unexpected end of input")
        self._closed = True
        return self._out

    # ---------------------------------------------------------------- internals
    def _check_usable(self) -> None:
        if self._error is not None:
            e = self._error
            raise JSONStreamError(e.message, e.offset, e.line, e.column)
        if self._closed:
            raise RuntimeError("the tokenizer is closed")

    def _fail(self, message: str) -> None:
        self._error = JSONStreamError(message, self._pos, self._line, self._col)
        raise JSONStreamError(message, self._pos, self._line, self._col)

    def _emit(self, kind: str, value, offset: int) -> None:
        self._out.append((kind, value, offset))

    def _after_value(self) -> None:
        self._expect = _COMMA_OR_END if self._stack else _DONE

    def _step(self, ch: str) -> None:
        if self._lex == "string":
            self._string_char(ch)
            return
        if self._lex == "literal":
            self._literal_char(ch)
            return
        if self._lex == "number" and self._number_char(ch):
            return
        self._structural(ch)

    def _structural(self, ch: str) -> None:
        if ch in _WS:
            return
        e = self._expect
        if e == _DONE:
            self._fail("extra data after the document")
        if ch == '"':
            if e in (_VALUE, _VALUE_OR_END):
                self._begin_string(False)
            elif e in (_KEY, _KEY_OR_END):
                self._begin_string(True)
            else:
                self._fail("unexpected string")
            return
        if ch == "}":
            if e in (_KEY_OR_END, _COMMA_OR_END) and self._stack[-1] == "{":
                self._stack.pop()
                self._emit("end_object", None, self._pos)
                self._after_value()
            else:
                self._fail("unexpected '}'")
            return
        if ch == "]":
            if e in (_VALUE_OR_END, _COMMA_OR_END) and self._stack[-1] == "[":
                self._stack.pop()
                self._emit("end_array", None, self._pos)
                self._after_value()
            else:
                self._fail("unexpected ']'")
            return
        if ch == ":":
            if e != _COLON:
                self._fail("unexpected ':'")
            self._expect = _VALUE
            return
        if ch == ",":
            if e != _COMMA_OR_END:
                self._fail("unexpected ','")
            self._expect = _KEY if self._stack[-1] == "{" else _VALUE
            return
        if e not in (_VALUE, _VALUE_OR_END):
            self._fail("expected a key, ':' or ','")
        if ch == "{":
            self._stack.append("{")
            self._emit("begin_object", None, self._pos)
            self._expect = _KEY_OR_END
        elif ch == "[":
            self._stack.append("[")
            self._emit("begin_array", None, self._pos)
            self._expect = _VALUE_OR_END
        elif ch in _LITERALS:
            self._lit, kind, value = _LITERALS[ch]
            self._lit_token = (kind, value, self._pos)
            self._lit_i = 1
            self._lex = "literal"
        elif ch == "-" or ch in _DIGITS:
            self._lex = "number"
            self._start = self._pos
            self._num = [ch]
            self._nstate = _N_MINUS if ch == "-" else _N_ZERO if ch == "0" else _N_INT
        else:
            self._fail("unexpected character")

    # literals ---------------------------------------------------------------
    def _literal_char(self, ch: str) -> None:
        if ch != self._lit[self._lit_i]:
            self._fail("bad literal")
        self._lit_i += 1
        if self._lit_i == len(self._lit):
            self._lex = None
            self._out.append(self._lit_token)
            self._after_value()

    # numbers ----------------------------------------------------------------
    def _number_char(self, ch: str) -> bool:
        """Consume ``ch`` as part of the number (True), or end the number and leave ``ch`` to the caller (False)."""
        s = self._nstate
        digit = ch in _DIGITS
        if s == _N_MINUS:
            if not digit:
                self._fail("expected a digit")
            nxt = _N_ZERO if ch == "0" else _N_INT
        elif s == _N_ZERO:
            if digit:
                self._fail("leading zero")
            nxt = _N_DOT if ch == "." else _N_E if ch in "eE" else None
        elif s == _N_INT:
            nxt = _N_INT if digit else _N_DOT if ch == "." else _N_E if ch in "eE" else None
        elif s == _N_DOT:
            if not digit:
                self._fail("expected a digit after '.'")
            nxt = _N_FRAC
        elif s == _N_FRAC:
            nxt = _N_FRAC if digit else _N_E if ch in "eE" else None
        elif s == _N_E:
            if not (digit or ch in "+-"):
                self._fail("expected an exponent")
            nxt = _N_EXP if digit else _N_ESIGN
        elif s == _N_ESIGN:
            if not digit:
                self._fail("expected an exponent digit")
            nxt = _N_EXP
        else:
            nxt = _N_EXP if digit else None
        if nxt is None:
            self._end_number()
            return False
        self._nstate = nxt
        self._num.append(ch)
        return True

    def _end_number(self) -> None:
        text = "".join(self._num)
        value = float(text) if any(c in text for c in ".eE") else int(text)
        self._lex = None
        self._emit("number", value, self._start)
        self._after_value()

    # strings ----------------------------------------------------------------
    def _begin_string(self, is_key: bool) -> None:
        self._lex = "string"
        self._is_key = is_key
        self._start = self._pos
        self._parts = []
        self._esc = None
        self._high = None

    def _flush_high(self) -> None:
        if self._high is not None:
            self._parts.append(chr(self._high))
            self._high = None

    def _string_char(self, ch: str) -> None:
        if self._esc is None:
            if ch == '"':
                self._flush_high()
                self._lex = None
                self._emit("key" if self._is_key else "string", "".join(self._parts), self._start)
                if self._is_key:
                    self._expect = _COLON
                else:
                    self._after_value()
            elif ch == "\\":
                self._esc = "\\"
            elif ch < " ":
                self._fail("control character in string")
            else:
                self._flush_high()
                self._parts.append(ch)
        elif self._esc == "\\":
            if ch in _ESCAPES:
                self._flush_high()
                self._parts.append(_ESCAPES[ch])
                self._esc = None
            elif ch == "u":
                self._esc = "u"
                self._hex = ""
            else:
                self._fail("bad escape")
        else:
            if ch not in _HEX:
                self._fail("bad \\u escape")
            self._hex += ch
            if len(self._hex) == 4:
                self._esc = None
                code = int(self._hex, 16)
                if 0xDC00 <= code <= 0xDFFF and self._high is not None:
                    self._parts.append(chr(0x10000 + ((self._high - 0xD800) << 10) + (code - 0xDC00)))
                    self._high = None
                else:
                    self._flush_high()
                    if 0xD800 <= code <= 0xDBFF:
                        self._high = code
                    else:
                        self._parts.append(chr(code))
