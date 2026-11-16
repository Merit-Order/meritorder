from fractions import Fraction

from solution import EvalError, ExprError, ParseError, evaluate

F = Fraction


def _error(text, variables=None):
    try:
        value = evaluate(text, variables)
    except ExprError as e:
        return e
    raise AssertionError(f"{text!r} returned {value!r} instead of raising")


def _parse_pos(text):
    e = _error(text)
    assert isinstance(e, ParseError), (text, type(e).__name__)
    return e.position


def _eval_pos(text, variables=None):
    e = _error(text, variables)
    assert isinstance(e, EvalError) and not isinstance(e, ParseError), (text, type(e).__name__)
    return e.position


def test_basic_precedence_and_type():
    r = evaluate("1 + 2 * 3")
    assert r == 7 and isinstance(r, Fraction)
    assert evaluate("(1 + 2) * 3") == 9
    assert evaluate("2 * 3 % 4") == 2


def test_exact_decimals():
    assert evaluate("0.1 + 0.2") == F(3, 10)
    assert evaluate(".5 + 5.") == F(11, 2)
    assert evaluate("007 * 1.250") == F(35, 4)


def test_exact_division():
    assert evaluate("1/3 + 1/6") == F(1, 2)
    assert evaluate("1/3*3") == 1


def test_left_associativity():
    assert evaluate("10 - 4 - 3") == 3
    assert evaluate("64 / 4 / 2") == 8
    assert evaluate("100 // 7 // 2") == 7
    assert evaluate("2 - 3 + 4") == 3


def test_power_right_associative():
    assert evaluate("2^3^2") == 512
    assert evaluate("(2^3)^2") == 64


def test_unary_minus_looser_than_power():
    assert evaluate("-2^2") == -4
    assert evaluate("(-2)^2") == 4
    assert evaluate("-2^-2") == F(-1, 4)
    assert evaluate("3 - -2^2") == 7


def test_signed_exponents():
    assert evaluate("2^-1") == F(1, 2)
    assert evaluate("2^-3^2") == F(1, 512)
    assert evaluate("2^--1") == 2
    assert evaluate("2^+3") == 8


def test_repeated_unary():
    assert evaluate("--3") == 3
    assert evaluate("2--3") == 5
    assert evaluate("2*-3") == -6
    assert evaluate("-+-1") == 1
    assert evaluate("-" * 50 + "1") == 1


def test_floor_division_and_modulo_signs():
    assert evaluate("-7 // 2") == -4
    assert evaluate("-(7 // 2)") == -3
    assert evaluate("7 // -2") == -4
    assert evaluate("7 % -2") == -1
    assert evaluate("-7 % 2") == 1
    assert evaluate("-7 % -2") == -1


def test_floor_division_and_modulo_on_fractions():
    r = evaluate("7.5 // 2")
    assert r == 3 and isinstance(r, Fraction)
    assert evaluate("7.5 % 2") == F(3, 2)
    assert evaluate("(1/3) % (1/4)") == F(1, 12)
    assert evaluate("-(1/3) // (1/4)") == -2


def test_term_operators_share_precedence():
    assert evaluate("2 + 7 // 2 * 2") == 8
    assert evaluate("12 / 4 * 3") == 9
    assert evaluate("13 % 5 * 2") == 6


def test_powers_of_fractions():
    assert evaluate("(2/3)^-2") == F(9, 4)
    assert evaluate("(-1/2)^3") == F(-1, 8)
    assert evaluate("0^0") == 1
    assert evaluate("4^(4/2)") == 16


def test_exponent_limit():
    r = evaluate("2^4096")
    assert r == 2 ** 4096 and isinstance(r, Fraction)
    assert evaluate("2^-4096") == F(1, 2 ** 4096)
    assert _eval_pos("2^4097") == 1
    assert _eval_pos("1 ^ -5000") == 2


def test_bad_exponents():
    assert _eval_pos("4^(1/2)") == 1
    assert _eval_pos("0^-1") == 1
    assert _eval_pos("2 + 0 ^ (0-3)") == 6


def test_variables():
    v = {"x": 2, "y": F(1, 3), "z": -1, "_a1": 10}
    r = evaluate("x*y + z", v)
    assert r == F(-1, 3) and isinstance(r, Fraction)
    assert evaluate("_a1 ^ x", v) == 100
    assert evaluate("abs + 1", {"abs": 3}) == 4


def test_unknown_variable():
    assert _eval_pos("x + y", {"x": 1}) == 4
    assert _eval_pos("X", {"x": 1}) == 0
    assert _eval_pos("1 + x") == 4
    assert _eval_pos("1 + x", None) == 4


def test_functions():
    assert evaluate("abs(-3/4)") == F(3, 4)
    assert evaluate("min(3, 1/2, 2)") == F(1, 2)
    assert evaluate("max(-1, -2)") == -1
    assert evaluate("max(1, min(2, 3))^2") == 4
    assert evaluate("abs (-2)") == 2
    assert evaluate("-abs(2)^2") == -4
    assert isinstance(evaluate("max(1, 2)"), Fraction)


def test_function_errors():
    assert _eval_pos("sqrt(4)") == 0
    assert _eval_pos("2 * ABS(1)") == 4
    assert _eval_pos("abs(1, 2)") == 0
    assert _eval_pos("x(2)", {"x": 1}) == 0


def test_call_checked_before_arguments():
    assert _eval_pos("1 + foo(1/0)") == 4
    assert _eval_pos("abs(1/0, 2)") == 0
    assert _eval_pos("max(1, 2/0, y)") == 8


def test_division_by_zero():
    assert _eval_pos("1/(2-2)") == 1
    assert _eval_pos("5 % 0") == 2
    assert _eval_pos("5 // 0.0") == 2


def test_left_to_right_evaluation_order():
    assert _eval_pos("1/0 + 2/0") == 1
    assert _eval_pos("(1 % 0) ^ (1/2)") == 3
    assert _eval_pos("q / 0", {}) == 0
    assert _eval_pos("2 ^ (1/2) + 1/0") == 2


def test_syntax_checked_before_evaluation():
    assert _parse_pos("1/0 +") == 5
    assert _parse_pos("foo(1/0") == 7
    assert _parse_pos("x y") == 2


def test_whitespace():
    assert evaluate("\t1\n+\r2 ") == 3
    assert evaluate("  max ( 1 ,\n 2 )  ") == 2


def test_parse_error_end_of_text():
    assert _parse_pos("") == 0
    assert _parse_pos("   ") == 3
    assert _parse_pos("1 +") == 3
    assert _parse_pos("(1+2") == 4
    assert _parse_pos("(1+2   ") == 7
    assert _parse_pos("2 ^ ") == 4
    assert _parse_pos("min(1, ") == 7


def test_parse_error_unexpected_token():
    assert _parse_pos("1+2)") == 3
    assert _parse_pos("1 + * 2") == 4
    assert _parse_pos("2 3") == 2
    assert _parse_pos("()") == 1
    assert _parse_pos("1,2") == 1
    assert _parse_pos("(1 2)") == 3


def test_parse_error_operators_not_in_language():
    assert _parse_pos("2**3") == 2
    assert _parse_pos("6 / / 2") == 4
    assert _parse_pos("2///3") == 3
    assert evaluate("6//4") == 1


def test_parse_error_numbers_and_names():
    assert _parse_pos("2x") == 1
    assert _parse_pos("1..2") == 2
    assert _parse_pos("1.2.3") == 3
    assert _parse_pos(".") == 0
    assert _parse_pos("1 + .") == 4
    assert _parse_pos("abs 2") == 4


def test_parse_error_calls():
    assert _parse_pos("abs()") == 4
    assert _parse_pos("max(1,)") == 6
    assert _parse_pos("max(1 2)") == 6


def test_invalid_characters():
    assert _parse_pos("2 $ 3") == 2
    assert _parse_pos("1 ) $") == 2
    assert _parse_pos("2²") == 1
    assert _parse_pos("٣") == 0
    assert _parse_pos("1\xa0+ 2") == 1
    assert _parse_pos("1 +\f2") == 3
    assert _parse_pos("é + 1") == 0


def test_error_hierarchy():
    p = _error("1 +")
    e = _error("1/0")
    assert isinstance(p, ParseError) and isinstance(p, ExprError) and not isinstance(p, EvalError)
    assert isinstance(e, EvalError) and isinstance(e, ExprError) and not isinstance(e, ParseError)
    assert isinstance(p.position, int) and isinstance(e.position, int)


def test_deep_nesting():
    assert evaluate("(" * 50 + "1" + ")" * 50) == 1
    assert evaluate("abs(" * 25 + "-1" + ")" * 25) == 1
    assert _parse_pos("(" * 50 + "1" + ")" * 49) == 100
