//! Integration tests for ES iterator-protocol array destructuring and the
//! related object-pattern fixes.
//!
//! Array destructuring must drive `value[Symbol.iterator]().next()` (not
//! index-based reads) so generators / custom iterables destructure correctly,
//! side effects order, and the iterator is closed on early/abrupt completion.
//! Uses the full parse → interpret pipeline (the same path test262 exercises).

use just_engine::parser::JsParser;
use just_engine::runner::ds::value::{JsNumberType, JsValue};
use just_engine::runner::eval::statement::execute_statement;
use just_engine::runner::plugin::registry::BuiltInRegistry;
use just_engine::runner::plugin::types::EvalContext;

/// Parse and execute JS, returning the completion value of the last statement.
fn run_js(code: &str) -> Result<JsValue, String> {
    let ast = JsParser::parse_to_ast_from_str(code).map_err(|e| format!("Parse error: {:?}", e))?;
    let mut ctx = EvalContext::new();
    ctx.install_core_builtins(BuiltInRegistry::with_core());
    let mut last = JsValue::Undefined;
    for stmt in &ast.body {
        let completion =
            execute_statement(stmt, &mut ctx).map_err(|e| format!("Runtime error: {:?}", e))?;
        if let Some(val) = completion.value {
            last = val;
        }
    }
    Ok(last)
}

fn assert_int(code: &str, expected: i64) {
    match run_js(code) {
        Ok(JsValue::Number(JsNumberType::Integer(n))) => {
            assert_eq!(n, expected, "code `{}` -> {} (expected {})", code, n, expected)
        }
        other => panic!("code `{}` -> {:?} (expected Integer {})", code, other, expected),
    }
}

fn assert_bool(code: &str, expected: bool) {
    match run_js(code) {
        Ok(JsValue::Boolean(b)) => {
            assert_eq!(b, expected, "code `{}` -> {} (expected {})", code, b, expected)
        }
        other => panic!("code `{}` -> {:?} (expected Boolean {})", code, other, expected),
    }
}

fn assert_str(code: &str, expected: &str) {
    match run_js(code) {
        Ok(JsValue::String(s)) => {
            assert_eq!(s, expected, "code `{}` -> {} (expected {})", code, s, expected)
        }
        other => panic!("code `{}` -> {:?} (expected String {})", code, other, expected),
    }
}

// ---- Well-known Symbol.iterator ---------------------------------------------

#[test]
fn symbol_iterator_is_a_symbol() {
    assert_str("typeof Symbol.iterator", "symbol");
}

// ---- Generator / iterator destructuring -------------------------------------

#[test]
fn destructures_a_generator() {
    assert_int("function* g(){ yield 1; yield 2; } var [a, b] = g(); a + b", 3);
}

#[test]
fn iterator_element_order_is_lazy() {
    // Each element pulls exactly one step, in order.
    assert_int(
        "var log = 0;
         function* g(){ log = 1; yield 10; log = 2; yield 20; }
         var [a, b] = g();
         a + b + log",
        32,
    );
}

#[test]
fn rest_drains_the_iterator() {
    assert_int(
        "function* g(){ yield 1; yield 2; yield 3; }
         var [head, ...tail] = g();
         head * 100 + tail.length * 10 + tail[1]",
        123,
    );
}

#[test]
fn rest_from_iterator_is_a_real_array() {
    assert_bool(
        "function* g(){ yield 1; } var [...xs] = g(); Array.isArray(xs)",
        true,
    );
}

#[test]
fn elision_consumes_a_step() {
    assert_int("function* g(){ yield 1; yield 2; yield 3; } var [, x, ] = g(); x", 2);
}

#[test]
fn default_applies_when_iterator_exhausted() {
    assert_int("function* g(){ yield 1; } var [a, b = 99] = g(); a + b", 100);
}

#[test]
fn nested_pattern_over_iterator() {
    assert_int(
        "function* g(){ yield [1, 2]; yield 3; }
         var [[a, b], c] = g();
         a + b + c",
        6,
    );
}

// ---- Custom iterables + iterator close --------------------------------------

#[test]
fn custom_symbol_iterator_is_used() {
    assert_int(
        "var obj = {};
         obj[Symbol.iterator] = function () {
           var i = 0;
           return { next: function () { i++; return { value: i, done: i > 2 }; } };
         };
         var [a, b] = obj;
         a * 10 + b",
        12,
    );
}

#[test]
fn iterator_is_closed_on_early_completion() {
    // A non-exhausted iterator gets its `return()` called exactly once.
    assert_int(
        "var closed = 0;
         var obj = {};
         obj[Symbol.iterator] = function () {
           return {
             next: function () { return { value: 1, done: false }; },
             return: function () { closed++; return {}; }
           };
         };
         var [x] = obj;
         closed",
        1,
    );
}

#[test]
fn iterator_not_closed_when_exhausted() {
    // Fully consuming the iterator (rest) must NOT call `return()`.
    assert_int(
        "var closed = 0;
         var obj = {};
         obj[Symbol.iterator] = function () {
           var i = 0;
           return {
             next: function () { i++; return { value: i, done: i > 2 }; },
             return: function () { closed++; return {}; }
           };
         };
         var [...xs] = obj;
         closed * 10 + xs.length",
        2,
    );
}

// ---- String iteration -------------------------------------------------------

#[test]
fn destructures_a_string() {
    assert_str("var [a, b] = 'hi'; a + b", "hi");
}

// ---- Plain-array fast path still works --------------------------------------

#[test]
fn plain_array_fast_path() {
    assert_int("var [a, b, c] = [7, 8, 9]; a + b + c", 24);
}

#[test]
fn plain_array_rest_is_array() {
    assert_bool("var [a, ...r] = [1, 2, 3]; Array.isArray(r) && r.length === 2", true);
}

// ---- Assignment (non-declaration) form --------------------------------------

#[test]
fn assignment_form_drives_iterator() {
    assert_int(
        "var a, b; function* g(){ yield 4; yield 5; } [a, b] = g(); a + b",
        9,
    );
}

// ---- Object-pattern fixes ---------------------------------------------------

#[test]
fn object_rest_skips_non_enumerable() {
    assert_bool(
        "var o = { a: 1, b: 2 };
         Object.defineProperty(o, 'hidden', { value: 3, enumerable: false });
         var { ...rest } = o;
         rest.hidden === undefined && rest.a === 1 && rest.b === 2",
        true,
    );
}

#[test]
fn object_rest_invokes_getter_once() {
    assert_int(
        "var count = 0;
         var src = { get v() { count++; return 7; } };
         var { ...x } = src;
         count * 10 + x.v",
        17,
    );
}

// ---- Anonymous function-name inference through defaults ----------------------

#[test]
fn array_default_infers_function_name() {
    assert_str("var [f = function () {}] = []; f.name", "f");
}

#[test]
fn array_default_infers_parenthesized_function_name() {
    assert_str("var [cover = (function () {})] = []; cover.name", "cover");
}

#[test]
fn array_default_infers_anonymous_class_name() {
    assert_str("var [cls = class {}] = []; cls.name", "cls");
}

#[test]
fn named_class_default_keeps_its_name() {
    assert_bool("var [xc = class X {}] = []; xc.name === 'X'", true);
}
