"""Hidden tests for c5-lisp."""

import sys
import threading

from solution import LispError, run


def raises(src):
    try:
        run(src)
    except LispError:
        return True
    return False


def peak_depth(src):
    """run(src) and the deepest Python call stack reached while it ran (counted with a profiler, in every thread)."""
    depth, peak = {}, [0]

    def prof(frame, event, arg):
        if event == "call":
            t = threading.get_ident()
            depth[t] = depth.get(t, 0) + 1
            peak[0] = max(peak[0], depth[t])
        elif event == "return":
            t = threading.get_ident()
            depth[t] = depth.get(t, 0) - 1

    sys.setprofile(prof)
    threading.setprofile(prof)
    try:
        out = run(src)
    finally:
        sys.setprofile(None)
        threading.setprofile(None)
    return out, peak[0]


def constant_depth(template, small=10, big=1000):
    """Outputs for N=small and N=big, and whether the Python stack depth stayed the same (within a small slack)."""
    out1, d1 = peak_depth(template.replace("N", str(small)))
    out2, d2 = peak_depth(template.replace("N", str(big)))
    return out1, out2, d2 - d1 <= 50


# ---------------------------------------------------------------- proper tail calls

def test_tail_call_in_if():
    small, big, ok = constant_depth("(define (loop i acc) (if (= i 0) acc (loop (- i 1) (+ acc 2)))) (loop N 0)")
    assert (small, big) == ("20", "2000")
    assert ok


def test_tail_call_in_cond_clauses():
    src = """(define c 0)
             (define (f n) (cond ((< n 0) 'neg)
                                 ((= n 0) (list 'done c))
                                 ((odd? n) (set! c (+ c 1)) (f (- n 1)))
                                 (else (f (- n 1)))))
             (define (odd? n) (= (remainder n 2) 1))
             (f N)"""
    small, big, ok = constant_depth(src)
    assert (small, big) == ("(done 5)", "(done 500)")
    assert ok


def test_tail_call_as_last_operand_of_and():
    small, big, ok = constant_depth("(define (f n) (if (= n 0) 'end (and #t 1 (f (- n 1))))) (f N)")
    assert (small, big) == ("end", "end")
    assert ok


def test_tail_call_as_last_operand_of_or():
    small, big, ok = constant_depth("(define (g n) (or (= n 0) #f (g (- n 1)))) (g N)")
    assert (small, big) == ("#t", "#t")
    assert ok


def test_tail_call_in_begin():
    small, big, ok = constant_depth("(define k 0) (define (f n) (if (= n 0) k (begin (set! k (+ k 1)) (f (- n 1))))) (f N)")
    assert (small, big) == ("10", "1000")
    assert ok


def test_tail_call_at_end_of_multi_expression_body():
    src = "(define (f n acc) (define step 3) (set! acc (+ acc step)) (if (= n 0) acc (f (- n 1) acc))) (f N 0)"
    small, big, ok = constant_depth(src)
    assert (small, big) == ("33", "3003")
    assert ok


def test_tail_call_in_let_let_star_and_letrec_bodies():
    for form in ["(let ((m (- n 1)))", "(let* ((k n) (m (- k 1)))", "(letrec ((m (- n 1)))"]:
        src = f"(define (f n) {form} (if (< m 0) 'done (f m)))) (f N)"
        small, big, ok = constant_depth(src)
        assert (small, big) == ("done", "done"), form
        assert ok, form


def test_tail_call_in_named_let():
    src = "(let loop ((i 0) (acc 0)) (if (= i N) acc (loop (+ i 1) (+ acc 1))))"
    small, big, ok = constant_depth(src)
    assert (small, big) == ("10", "1000")
    assert ok


def test_mutual_tail_recursion():
    src = """(define (my-even? n) (if (= n 0) #t (my-odd? (- n 1))))
             (define (my-odd? n) (if (= n 0) #f (my-even? (- n 1))))
             (list (my-even? N) (my-odd? N))"""
    small, big, ok = constant_depth(src)
    assert (small, big) == ("(#t #f)", "(#t #f)")
    assert ok


def test_tail_calls_to_variadic_and_passed_procedures():
    src = """(define (f n . extra) (if (= n 0) (length extra) (f (- n 1) 1 2)))
             (define (count-down n k) (if (= n 0) (k 'done) (count-down (- n 1) k)))
             (list (f N) (count-down N (lambda (x) (list x x))))"""
    small, big, ok = constant_depth(src)
    assert (small, big) == ("(2 (done done))", "(2 (done done))")
    assert ok


def test_loops_of_100000_iterations():
    assert run("(define (loop i acc) (if (= i 0) acc (loop (- i 1) (+ acc 1)))) (loop 100000 0)") == "100000"
    assert run("(let loop ((i 0)) (if (< i 100000) (loop (+ i 1)) i))") == "100000"
    assert run("""(define (e? n) (if (= n 0) #t (o? (- n 1))))
                  (define (o? n) (if (= n 0) #f (e? (- n 1))))
                  (e? 100001)""") == "#f"


def test_non_tail_recursion_1000_deep():
    assert run("(define (sum n) (if (= n 0) 0 (+ n (sum (- n 1))))) (sum 1000)") == "500500"
    assert run("(define (build n) (if (= n 0) '() (cons n (build (- n 1))))) (length (build 1000))") == "1000"


# ---------------------------------------------------------------- numbers

def test_integer_arithmetic():
    assert run("(+ 1 2 3)") == "6"
    assert run("(+)") == "0"
    assert run("(*)") == "1"
    assert run("(- 5)") == "-5"
    assert run("(- 10 1 2)") == "7"
    assert run("(* 2 -3)") == "-6"
    assert run("(+ +5 -3)") == "2"
    assert run("(* 99999999999 99999999999)") == "9999999999800000000001"
    assert run("(abs -7)") == "7"


def test_exact_rationals():
    assert run("(/ 1 3)") == "1/3"
    assert run("(/ 6 3)") == "2"
    assert run("(/ 6 4)") == "3/2"
    assert run("(/ 1 -2)") == "-1/2"
    assert run("(/ 2)") == "1/2"
    assert run("(/ 12 3 2)") == "2"
    assert run("(/ 1 2 3)") == "1/6"
    assert run("(+ (/ 1 2) (/ 1 3))") == "5/6"
    assert run("(* (/ 1 2) 2)") == "1"
    assert run("(- (/ 1 2))") == "-1/2"
    assert run("(list (integer? (* (/ 1 2) 4)) (integer? (/ 1 2)) (number? (/ 1 2)))") == "(#t #f #t)"
    assert run("(abs (/ -1 2))") == "1/2"


def test_quotient_remainder_modulo_signs():
    cases = {"(quotient 7 2)": "3", "(quotient -7 2)": "-3", "(quotient 7 -2)": "-3", "(quotient -7 -2)": "3",
             "(remainder 7 2)": "1", "(remainder -7 2)": "-1", "(remainder 7 -2)": "1", "(remainder -7 -2)": "-1",
             "(modulo 7 2)": "1", "(modulo -7 2)": "1", "(modulo 7 -2)": "-1", "(modulo -7 -2)": "-1",
             "(remainder 6 3)": "0", "(modulo -6 3)": "0"}
    for src, want in cases.items():
        assert run(src) == want, src


def test_comparisons_chain():
    assert run("(< 1 2 3)") == "#t"
    assert run("(< 1 3 2)") == "#f"
    assert run("(= 1 1 1)") == "#t"
    assert run("(>= 3 3 2)") == "#t"
    assert run("(<= 1 1 0)") == "#f"
    assert run("(> 3 2)") == "#t"
    assert run("(= (/ 1 2) (/ 2 4))") == "#t"
    assert run("(> (/ 1 3) (/ 1 4))") == "#t"


def test_booleans_are_not_numbers():
    assert run("(number? #t)") == "#f"
    assert run("(boolean? #f)") == "#t"
    assert run("(boolean? 0)") == "#f"
    assert run("(equal? 1 #t)") == "#f"
    assert run("(eq? 0 #f)") == "#f"
    assert raises("(+ 1 #t)")
    assert raises("(= 0 #f)")
    assert raises("(< #f 1)")


# ---------------------------------------------------------------- printing and reading

def test_printing_values():
    assert run("'(1 \"a\\\"b\" #t sym (2 . 3) ())") == '(1 "a\\"b" #t sym (2 . 3) ())'
    assert run("(cons 1 (cons 2 3))") == "(1 2 . 3)"
    assert run("'(1 . (2 . (3 . ())))") == "(1 2 3)"
    assert run("''a") == "(quote a)"
    assert run("'(a 'b)") == "(a (quote b))"
    assert run('"tab\\there\\nnew"') == '"tab\\there\\nnew"'
    assert run('"back\\\\slash"') == '"back\\\\slash"'
    assert run("car") == "#<procedure>"
    assert run("(lambda (x) x)") == "#<procedure>"
    assert run("(list (/ -6 4) 0 -0 'Hello)") == "(-3/2 0 0 Hello)"
    assert run("(list (if #f #f) (begin))") == "(#<void> #<void>)"


def test_program_value_is_last_form():
    assert run("") == "#<void>"
    assert run("   ; only a comment\n") == "#<void>"
    assert run("(define z 1)") == "#<void>"
    assert run("1 2 3") == "3"
    assert run("(define z 1) (set! z 2)") == "#<void>"


def test_reader_comments_whitespace_and_atoms():
    src = "; leading comment\r\n(define (sq x) ; trailing\n\t(* x x))\r\n\n(sq\t\t12) ; done"
    assert run(src) == "144"
    assert run("'(a.b 1+ -> 3/4 1.5 - +)") == "(a.b 1+ -> 3/4 1.5 - +)"
    assert run("(list (symbol? '3/4) (symbol? '1.5) (symbol? '-) (number? '-5))") == "(#t #t #t #t)"
    assert run('(string-length "line1\nline2")') == "11"
    assert run("'(a(b)c)") == "(a (b) c)"
    assert run("'x") == "x"


def test_strings():
    assert run('(string-append "ab" "c" "")') == '"abc"'
    assert run('(string-append)') == '""'
    assert run('(string-length "a\\nb")') == "3"
    assert run("(number->string (/ -1 2))") == '"-1/2"'
    assert run("(symbol->string 'abc)") == '"abc"'
    assert run('(string->symbol "hello")') == "hello"
    assert run("(list (string? \"a\") (string? 'a) (symbol? 'a) (symbol? \"a\"))") == "(#t #f #t #f)"


def test_quote():
    assert run("(quote (a b))") == "(a b)"
    assert run("'()") == "()"
    assert run("(car '(quote x))") == "quote"
    assert run("(list '3 '\"s\" '#t)") == '(3 "s" #t)'
    assert run("(symbol? (car ''a))") == "#t"


# ---------------------------------------------------------------- semantics

def test_truthiness_only_false_is_false():
    assert run("(if 0 'yes 'no)") == "yes"
    assert run("(if '() 'yes 'no)") == "yes"
    assert run('(if "" \'yes \'no)') == "yes"
    assert run("(if #f 1 2)") == "2"
    assert run("(not 0)") == "#f"
    assert run("(not #f)") == "#t"
    assert run("(if #f #f)") == "#<void>"


def test_and_or_values_and_short_circuit():
    assert run("(and)") == "#t"
    assert run("(or)") == "#f"
    assert run("(and 1 2 3)") == "3"
    assert run("(and 1 #f 3)") == "#f"
    assert run("(or #f 2 3)") == "2"
    assert run("(or #f #f)") == "#f"
    assert run("(or #f '())") == "()"
    assert run("(define x 0) (and #f (set! x 1)) (or 1 (set! x 2)) x") == "0"
    assert run("(or 1 (car '()))") == "1"


def test_define_and_scopes():
    assert run("(define x 1) (define x 2) x") == "2"
    assert raises("(define (f) (define inner 5) inner) (f) inner")
    assert run("(define x 1) (define (g x) (set! x 10) x) (list (g 2) x)") == "(10 1)"
    assert run("(define (f) (define a 1) (define (h) (+ a 1)) (h)) (f)") == "2"
    assert run("(begin (define y 7)) y") == "7"
    assert run("(define (f . args) args) (f 1 2)") == "(1 2)"


def test_closures_share_mutable_environments():
    src = """(define (make-counter)
               (let ((n 0)) (lambda () (set! n (+ n 1)) n)))
             (define c1 (make-counter))
             (define c2 (make-counter))
             (c1) (c1) (c2)
             (list (c1) (c2))"""
    assert run(src) == "(3 2)"
    src = """(define (make-box v)
               (list (lambda () v) (lambda (new) (set! v new))))
             (define b (make-box 1))
             ((car (cdr b)) 42)
             ((car b))"""
    assert run(src) == "42"


def test_lexical_not_dynamic_scope():
    assert run("(define x 'global) (define (show) x) (define (call-it x) (show)) (call-it 'local)") == "global"
    assert run("(define (adder n) (lambda (x) (+ x n))) (define n 100) ((adder 5) 1)") == "6"


def test_variadic_lambdas():
    assert run("((lambda args args) 1 2 3)") == "(1 2 3)"
    assert run("((lambda args args))") == "()"
    assert run("((lambda (a . r) (list a r)) 1)") == "(1 ())"
    assert run("((lambda (a b . r) r) 1 2 3 4)") == "(3 4)"
    assert run("(define (f . xs) (length xs)) (f)") == "0"
    assert raises("((lambda (a . r) a))")
    assert raises("((lambda (a b . r) a) 1)")


def test_let_family():
    assert run("(let ((x 1) (y 2)) (+ x y))") == "3"
    assert run("(define x 1) (let ((x 2) (y x)) y)") == "1"
    assert run("(let* ((x 1) (y (+ x 1))) (* x y))") == "2"
    assert run("(let* ((x 1) (f (lambda () x)) (x 2)) (list x (f)))") == "(2 1)"
    assert run("(let () 5)") == "5"
    assert run("(let ((x 1)) (define y 2) (+ x y))") == "3"
    src = """(letrec ((ev? (lambda (n) (if (= n 0) #t (od? (- n 1)))))
                      (od? (lambda (n) (if (= n 0) #f (ev? (- n 1))))))
               (list (ev? 10) (od? 7)))"""
    assert run(src) == "(#t #t)"
    assert run("(letrec ((a 1) (b (+ a 1))) b)") == "2"
    assert run("(let fact ((n 5) (acc 1)) (if (= n 0) acc (fact (- n 1) (* acc n))))") == "120"
    assert run("(define loop 'outer) (let loop ((x loop)) x)") == "outer"


def test_cond():
    assert run("(cond (#f 1) ((+ 1 1)) (else 3))") == "2"
    assert run("(cond (#f 1))") == "#<void>"
    assert run("(cond ((= 1 1) 'a 'b))") == "b"
    assert run("(cond (else 'e))") == "e"
    assert run("(define x 0) (cond (#f (set! x 1)) (#t (set! x 2)) (else (set! x 3))) x") == "2"


def test_set_bang():
    assert run("(define x 1) (set! x 5)") == "#<void>"
    assert run("(define x 1) (define (bump) (set! x (+ x 1))) (bump) (bump) x") == "3"
    assert run("(define x 1) (let ((x 10)) (set! x 20)) x") == "1"
    assert raises("(set! nope 1)")
    assert raises("(define (f) (set! local-only 1)) (f)")


def test_eq_and_equal():
    cases = {"(eq? 'a 'a)": "#t", "(eq? 'a 'b)": "#f", "(eq? '() '())": "#t", "(eq? (list) '())": "#t",
             "(eq? (list 1) (list 1))": "#f", "(let ((p (list 1))) (eq? p p))": "#t",
             "(eq? (* 1000000 1000000) (* 1000000 1000000))": "#t", "(eq? (/ 1 2) (/ 2 4))": "#t",
             "(eq? 1 #t)": "#f", "(eq? #t #t)": "#t", "(let ((s \"ab\")) (eq? s s))": "#t", "(eq? car car)": "#t",
             "(equal? (list 1 (list 2 \"x\")) '(1 (2 \"x\")))": "#t", "(equal? \"ab\" \"ab\")": "#t",
             "(equal? \"ab\" \"aB\")": "#f", "(equal? (cons 1 2) (cons 1 2))": "#t", "(equal? 2 (/ 4 2))": "#t",
             "(equal? '(1 2) '(1 2 3))": "#f", "(equal? 'a \"a\")": "#f"}
    for src, want in cases.items():
        assert run(src) == want, src


def test_list_builtins():
    cases = {"(list)": "()", "(cons 1 '())": "(1)", "(car '(1 2))": "1", "(cdr '(1 2))": "(2)",
             "(length '())": "0", "(append)": "()", "(append '(1) '() '(2 3))": "(1 2 3)",
             "(reverse '(1 2 3))": "(3 2 1)", "(null? '())": "#t", "(null? '(1))": "#f", "(pair? '())": "#f",
             "(pair? (cons 1 2))": "#t", "(list? (cons 1 2))": "#f", "(list? '())": "#t", "(list? '(1 2))": "#t",
             "(apply + '(1 2 3))": "6", "(apply list '())": "()", "(map (lambda (x) (* x x)) '(1 2 3))": "(1 4 9)",
             "(map car '((a 1) (b 2)))": "(a b)", "(procedure? car)": "#t", "(procedure? 'car)": "#f",
             "(procedure? (lambda () 1))": "#t"}
    for src, want in cases.items():
        assert run(src) == want, src


def test_left_to_right_evaluation():
    src = """(define trail '())
             (define (note x) (set! trail (cons x trail)) x)
             ((begin (note 'op) list) (note 1) (note 2) (note 3))
             (reverse trail)"""
    assert run(src) == "(op 1 2 3)"


def test_bigger_programs():
    src = """(define (fact n) (if (= n 0) 1 (* n (fact (- n 1)))))
             (define (fib n) (if (< n 2) n (+ (fib (- n 1)) (fib (- n 2)))))
             (define (filter p xs) (cond ((null? xs) '()) ((p (car xs)) (cons (car xs) (filter p (cdr xs))))
                                         (else (filter p (cdr xs)))))
             (define (fold f acc xs) (if (null? xs) acc (fold f (f acc (car xs)) (cdr xs))))
             (define (compose . fs) (if (null? fs) (lambda (x) x)
                                        (lambda (x) ((car fs) ((apply compose (cdr fs)) x)))))
             (list (fact 25) (fib 15) (filter (lambda (x) (= (modulo x 3) 0)) '(1 3 4 6 9 10))
                   (fold + 0 '(1 2 3 4)) ((compose (lambda (x) (* x 2)) (lambda (x) (+ x 1))) 5)
                   (fold (lambda (a x) (+ a (/ 1 x))) 0 '(1 2 3)))"""
    assert run(src) == "(15511210043330985984000000 610 (3 6 9) 10 12 11/6)"


def test_each_run_has_a_fresh_environment():
    assert run("(define leftover 5) leftover") == "5"
    assert raises("leftover")
    assert run("(define + -) (+ 5 3)") == "2"
    assert run("(+ 5 3)") == "8"


# ---------------------------------------------------------------- errors

def test_arity_errors():
    for src in ["((lambda (x) x))", "((lambda (x) x) 1 2)", "(car)", "(cons 1)", "(not 1 2)", "(-)", "(/)", "(< 1)",
                "(quotient 1 2 3)", "(define (f a b) a) (f 1)", "(length)", "(eq? 1)"]:
        assert raises(src), src


def test_type_errors():
    for src in ["(+ 1 \"a\")", "(car '())", "(cdr 5)", "(length 5)", "(length (cons 1 2))", "(string-length 'a)",
                "(quotient (/ 1 2) 1)", "(< 1 'a)", "(apply + 1)", "(map car 5)", "(symbol->string \"a\")",
                "(append '(1) 2)", "(- \"5\")", "(string-append \"a\" 'b)", "(apply 5 '())", "(abs 'x)"]:
        assert raises(src), src


def test_runtime_errors():
    for src in ["(foo)", "undefined-var", "(1 2)", "(\"f\")", "('a)", "(/ 1 0)", "(quotient 1 0)", "(modulo 1 0)",
                "(remainder 5 0)", "(/ 5 (- 2 2))", "()", "(letrec ((a b) (b 1)) a)", "(define x x)",
                "((if #f car 5) '(1))", "(+ 1 (car 2))"]:
        assert raises(src), src


def test_syntax_errors():
    for src in ["(+ 1 2", "(+ 1 2))", ")", '"abc', '"bad \\q escape"', "(1 . 2 3)", "'( . 2)", "'(1 .)", "'", "#x",
                ".", "(quote 1 2)", "(if)", "(if 1 2 3 4)", "(lambda (x))", "(lambda (1) 1)", "(let ((x)) x)",
                "(let ((x 1 2)) x)", "(define)", "(set! 1 2)", "(lambda (x x) x)", "(define 5 1)", "(let ((x 1)))"]:
        assert raises(src), src
