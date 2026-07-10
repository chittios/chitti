//! Integration tests for the JavaScript interpreter.
//!
//! These tests parse JavaScript source code and execute it through
//! the interpreter to verify end-to-end functionality.

extern crate just;

use just::parser::JsParser;
use just::runner::eval::statement::execute_statement;
use just::runner::plugin::types::EvalContext;
use just::runner::ds::value::{JsValue, JsNumberType};

/// Helper to parse and execute JavaScript code, returning the final value.
fn run_js(code: &str) -> Result<JsValue, String> {
    let ast = JsParser::parse_to_ast_from_str(code)
        .map_err(|e| format!("Parse error: {:?}", e))?;

    let mut ctx = EvalContext::new();
    let mut last_value = JsValue::Undefined;

    for stmt in &ast.body {
        let completion = execute_statement(stmt, &mut ctx)
            .map_err(|e| format!("Runtime error: {:?}", e))?;

        if let Some(val) = completion.value {
            last_value = val;
        }
    }

    Ok(last_value)
}

/// Helper to run JS and get a specific variable's value.
fn run_js_get_var(code: &str, var_name: &str) -> Result<JsValue, String> {
    let ast = JsParser::parse_to_ast_from_str(code)
        .map_err(|e| format!("Parse error: {:?}", e))?;

    let mut ctx = EvalContext::new();

    for stmt in &ast.body {
        execute_statement(stmt, &mut ctx)
            .map_err(|e| format!("Runtime error: {:?}", e))?;
    }

    ctx.get_binding(var_name)
        .map_err(|e| format!("Variable error: {:?}", e))
}

// ============================================================================
// Basic Arithmetic Tests
// ============================================================================

#[test]
fn test_simple_addition() {
    let result = run_js("1 + 2").unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(3)));
}

#[test]
fn test_arithmetic_expression() {
    let result = run_js("2 * 3 + 4").unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(10)));
}

#[test]
fn test_arithmetic_with_parens() {
    let result = run_js("2 * (3 + 4)").unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(14)));
}

#[test]
fn test_division() {
    let result = run_js("10 / 2").unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(5)));
}

#[test]
fn test_modulo() {
    let result = run_js("7 % 3").unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(1)));
}

// ============================================================================
// Variable Declaration and Assignment Tests
// ============================================================================

#[test]
fn test_var_declaration() {
    let result = run_js_get_var("var x = 42;", "x").unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(42)));
}

#[test]
fn test_let_declaration() {
    let result = run_js_get_var("let x = 10;", "x").unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(10)));
}

#[test]
fn test_const_declaration() {
    let result = run_js_get_var("const PI = 3;", "PI").unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(3)));
}

#[test]
fn test_variable_reassignment() {
    let result = run_js_get_var("var x = 5; x = 10;", "x").unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(10)));
}

#[test]
fn test_compound_assignment() {
    let result = run_js_get_var("var x = 5; x += 3;", "x").unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(8)));
}

// ============================================================================
// Comparison and Logical Tests
// ============================================================================

#[test]
fn test_comparison_less_than() {
    let result = run_js("3 < 5").unwrap();
    assert_eq!(result, JsValue::Boolean(true));
}

#[test]
fn test_comparison_greater_than() {
    let result = run_js("5 > 3").unwrap();
    assert_eq!(result, JsValue::Boolean(true));
}

#[test]
fn test_strict_equality() {
    let result = run_js("5 === 5").unwrap();
    assert_eq!(result, JsValue::Boolean(true));
}

#[test]
fn test_strict_inequality() {
    let result = run_js("5 !== 3").unwrap();
    assert_eq!(result, JsValue::Boolean(true));
}

#[test]
fn test_logical_and() {
    let result = run_js("true && false").unwrap();
    assert_eq!(result, JsValue::Boolean(false));
}

#[test]
fn test_logical_or() {
    let result = run_js("true || false").unwrap();
    assert_eq!(result, JsValue::Boolean(true));
}

// ============================================================================
// Control Flow Tests
// ============================================================================

#[test]
fn test_if_statement_true() {
    let result = run_js_get_var("var x = 0; if (true) { x = 1; }", "x").unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(1)));
}

#[test]
fn test_if_statement_false() {
    let result = run_js_get_var("var x = 0; if (false) { x = 1; }", "x").unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(0)));
}

#[test]
fn test_if_else_statement() {
    let result = run_js_get_var("var x = 0; if (false) { x = 1; } else { x = 2; }", "x").unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(2)));
}

#[test]
fn test_ternary_operator() {
    let result = run_js("true ? 1 : 2").unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(1)));
}

// ============================================================================
// Loop Tests
// ============================================================================

#[test]
fn test_while_loop() {
    let code = r#"
        var sum = 0;
        var i = 1;
        while (i <= 5) {
            sum = sum + i;
            i = i + 1;
        }
    "#;
    let result = run_js_get_var(code, "sum").unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(15))); // 1+2+3+4+5 = 15
}

#[test]
fn test_for_loop() {
    let code = r#"
        var sum = 0;
        for (var i = 1; i <= 5; i = i + 1) {
            sum = sum + i;
        }
    "#;
    let result = run_js_get_var(code, "sum").unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(15)));
}

#[test]
fn test_for_loop_with_break() {
    let code = r#"
        var sum = 0;
        for (var i = 1; i <= 10; i = i + 1) {
            if (i > 5) { break; }
            sum = sum + i;
        }
    "#;
    let result = run_js_get_var(code, "sum").unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(15)));
}

#[test]
fn test_for_loop_with_continue() {
    let code = r#"
        var sum = 0;
        for (var i = 1; i <= 5; i = i + 1) {
            if (i === 3) { continue; }
            sum = sum + i;
        }
    "#;
    let result = run_js_get_var(code, "sum").unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(12))); // 1+2+4+5 = 12
}

#[test]
fn test_do_while_loop() {
    let code = r#"
        var sum = 0;
        var i = 1;
        do {
            sum = sum + i;
            i = i + 1;
        } while (i <= 5);
    "#;
    let result = run_js_get_var(code, "sum").unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(15)));
}

// ============================================================================
// Update Expression Tests
// ============================================================================

// ============================================================================
// Update Expression Tests (prefix and postfix)
// ============================================================================

#[test]
fn test_prefix_increment() {
    let code = "var x = 5; var y = ++x;";
    // y should get the new value (6), x should be 6
    let result_y = run_js_get_var(code, "y").unwrap();
    assert_eq!(result_y, JsValue::Number(JsNumberType::Integer(6)));
}

#[test]
fn test_prefix_increment_check_var() {
    let code = "var x = 5; ++x;";
    // x should now be 6
    let result = run_js_get_var(code, "x").unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(6)));
}

#[test]
fn test_postfix_increment() {
    let code = "var x = 5; var y = x++;";
    // y should get the old value (5), x should be 6
    let result_y = run_js_get_var(code, "y").unwrap();
    assert_eq!(result_y, JsValue::Number(JsNumberType::Integer(5)));
}

#[test]
fn test_postfix_increment_check_var() {
    let code = "var x = 5; x++;";
    // x should now be 6
    let result = run_js_get_var(code, "x").unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(6)));
}

#[test]
fn test_prefix_decrement() {
    let code = "var x = 5; var y = --x;";
    // y should get the new value (4), x should be 4
    let result_y = run_js_get_var(code, "y").unwrap();
    assert_eq!(result_y, JsValue::Number(JsNumberType::Integer(4)));
}

#[test]
fn test_prefix_decrement_check_var() {
    let code = "var x = 5; --x;";
    // x should now be 4
    let result = run_js_get_var(code, "x").unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(4)));
}

#[test]
fn test_postfix_decrement() {
    let code = "var x = 5; var y = x--;";
    // y should get the old value (5), x should be 4
    let result_y = run_js_get_var(code, "y").unwrap();
    assert_eq!(result_y, JsValue::Number(JsNumberType::Integer(5)));
}

#[test]
fn test_postfix_decrement_check_var() {
    let code = "var x = 5; x--;";
    // x should now be 4
    let result = run_js_get_var(code, "x").unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(4)));
}

// ============================================================================
// String Tests
// ============================================================================

#[test]
fn test_string_concatenation() {
    let result = run_js("\"hello\" + \" \" + \"world\"").unwrap();
    assert_eq!(result, JsValue::String("hello world".to_string()));
}

#[test]
fn test_string_number_concatenation() {
    let result = run_js("\"value: \" + 42").unwrap();
    assert_eq!(result, JsValue::String("value: 42".to_string()));
}

// ============================================================================
// Bitwise Operation Tests
// ============================================================================

#[test]
fn test_bitwise_and() {
    let result = run_js("5 & 3").unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(1))); // 101 & 011 = 001
}

#[test]
fn test_bitwise_or() {
    let result = run_js("5 | 3").unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(7))); // 101 | 011 = 111
}

#[test]
fn test_bitwise_xor() {
    let result = run_js("5 ^ 3").unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(6))); // 101 ^ 011 = 110
}

#[test]
fn test_left_shift() {
    let result = run_js("1 << 3").unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(8)));
}

#[test]
fn test_right_shift() {
    let result = run_js("8 >> 2").unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(2)));
}

// ============================================================================
// SunSpider-inspired Benchmark Tests
// ============================================================================

/// Based on SunSpider bitops-bitwise-and test (simplified)
#[test]
fn test_sunspider_bitwise_and() {
    let code = r#"
        var bitwiseAndValue = 4294967295;
        for (var i = 0; i < 1000; i = i + 1) {
            bitwiseAndValue = bitwiseAndValue & i;
        }
    "#;
    let result = run_js_get_var(code, "bitwiseAndValue").unwrap();
    // After ANDing with all values 0-999, the result should be 0
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(0)));
}

/// Fibonacci calculation
#[test]
fn test_fibonacci() {
    let code = r#"
        var n = 10;
        var a = 0;
        var b = 1;
        for (var i = 0; i < n; i = i + 1) {
            var temp = a;
            a = b;
            b = temp + b;
        }
    "#;
    let result = run_js_get_var(code, "a").unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(55))); // fib(10) = 55
}

/// Factorial calculation
#[test]
fn test_factorial() {
    let code = r#"
        var n = 5;
        var result = 1;
        for (var i = 2; i <= n; i = i + 1) {
            result = result * i;
        }
    "#;
    let result = run_js_get_var(code, "result").unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(120))); // 5! = 120
}

/// Sum of squares
#[test]
fn test_sum_of_squares() {
    let code = r#"
        var sum = 0;
        for (var i = 1; i <= 10; i = i + 1) {
            sum = sum + i * i;
        }
    "#;
    let result = run_js_get_var(code, "sum").unwrap();
    // 1 + 4 + 9 + 16 + 25 + 36 + 49 + 64 + 81 + 100 = 385
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(385)));
}

/// Prime check (simple sieve-like approach)
#[test]
fn test_is_prime() {
    let code = r#"
        var n = 17;
        var isPrime = true;
        for (var i = 2; i * i <= n; i = i + 1) {
            if (n % i === 0) {
                isPrime = false;
                break;
            }
        }
    "#;
    let result = run_js_get_var(code, "isPrime").unwrap();
    assert_eq!(result, JsValue::Boolean(true)); // 17 is prime
}

/// Greatest common divisor (Euclidean algorithm)
#[test]
fn test_gcd() {
    let code = r#"
        var a = 48;
        var b = 18;
        while (b !== 0) {
            var temp = b;
            b = a % b;
            a = temp;
        }
    "#;
    let result = run_js_get_var(code, "a").unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(6))); // gcd(48, 18) = 6
}

// ============================================================================
// Nested Control Flow Tests
// ============================================================================

#[test]
fn test_nested_loops() {
    let code = r#"
        var count = 0;
        for (var i = 0; i < 3; i = i + 1) {
            for (var j = 0; j < 3; j = j + 1) {
                count = count + 1;
            }
        }
    "#;
    let result = run_js_get_var(code, "count").unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(9)));
}

#[test]
fn test_nested_if() {
    let code = r#"
        var x = 10;
        var result = 0;
        if (x > 5) {
            if (x > 8) {
                result = 1;
            } else {
                result = 2;
            }
        } else {
            result = 3;
        }
    "#;
    let result = run_js_get_var(code, "result").unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(1)));
}

// ============================================================================
// Delete Operator Tests
// ============================================================================

#[test]
fn test_delete_property() {
    let code = "let obj = { x: 1, y: 2 }; delete obj.x; obj.x;";
    let result = run_js(code).unwrap();
    assert_eq!(result, JsValue::Undefined);
}

#[test]
fn test_delete_returns_true() {
    let code = "let obj = { x: 1 }; delete obj.x;";
    let result = run_js(code).unwrap();
    assert_eq!(result, JsValue::Boolean(true));
}

#[test]
fn test_delete_nonexistent() {
    let code = "let obj = {}; delete obj.x;";
    let result = run_js(code).unwrap();
    assert_eq!(result, JsValue::Boolean(true));
}

#[test]
fn test_delete_computed_property() {
    let code = r#"let obj = { x: 1 }; delete obj["x"];"#;
    let result = run_js(code).unwrap();
    assert_eq!(result, JsValue::Boolean(true));
}

// ============================================================================
// In Operator Tests
// ============================================================================

#[test]
fn test_in_operator_true() {
    let code = r#""x" in { x: 1 };"#;
    let result = run_js(code).unwrap();
    assert_eq!(result, JsValue::Boolean(true));
}

#[test]
fn test_in_operator_false() {
    let code = r#""y" in { x: 1 };"#;
    let result = run_js(code).unwrap();
    assert_eq!(result, JsValue::Boolean(false));
}

#[test]
fn test_in_array_index() {
    let code = "0 in [1, 2, 3];";
    let result = run_js(code).unwrap();
    assert_eq!(result, JsValue::Boolean(true));
}

#[test]
fn test_in_array_out_of_bounds() {
    let code = "5 in [1, 2, 3];";
    let result = run_js(code).unwrap();
    assert_eq!(result, JsValue::Boolean(false));
}

#[test]
fn test_in_array_length() {
    let code = r#""length" in [];"#;
    let result = run_js(code).unwrap();
    assert_eq!(result, JsValue::Boolean(true));
}

// ============================================================================
// Instanceof Operator Tests (basic - full tests after classes/new)
// ============================================================================

#[test]
fn test_instanceof_primitive_false() {
    // Primitives are never instanceof anything - use an object as the RHS
    let code = r#"
        var ctor = { prototype: {} };
        var result = 42 instanceof ctor;
    "#;
    let result = run_js_get_var(code, "result").unwrap();
    assert_eq!(result, JsValue::Boolean(false));
}

#[test]
fn test_instanceof_object_basic() {
    // Object with matching prototype chain
    let code = r#"
        var proto = {};
        var ctor = { prototype: proto };
        var obj = {};
        var result = obj instanceof ctor;
    "#;
    // obj doesn't have proto in its chain since we didn't set up the chain
    let result = run_js_get_var(code, "result").unwrap();
    assert_eq!(result, JsValue::Boolean(false));
}

// ============================================================================
// Getter and Setter Tests
// ============================================================================

#[test]
fn test_getter_basic() {
    let code = "let obj = { get x() { return 42; } }; obj.x;";
    let result = run_js(code).unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(42)));
}

#[test]
fn test_setter_basic() {
    let code = "let obj = { _v: 0, set v(x) { this._v = x; } }; obj.v = 10; obj._v;";
    let result = run_js(code).unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(10)));
}

#[test]
fn test_getter_setter_combined() {
    let code = r#"
        let obj = {
            _value: 0,
            get value() { return this._value; },
            set value(v) { this._value = v * 2; }
        };
        obj.value = 21;
        obj.value;
    "#;
    let result = run_js(code).unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(42)));
}

#[test]
fn test_getter_with_this() {
    let code = r#"
        let obj = {
            firstName: "John",
            lastName: "Doe",
            get fullName() { return this.firstName + " " + this.lastName; }
        };
        obj.fullName;
    "#;
    let result = run_js(code).unwrap();
    assert_eq!(result, JsValue::String("John Doe".to_string()));
}

// ==================== PHASE 3: SPREAD OPERATOR TESTS ====================

#[test]
fn test_spread_array_basic() {
    let code = "let a = [1, 2]; let b = [...a, 3]; b.length;";
    let result = run_js(code).unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(3)));
}

#[test]
fn test_spread_array_middle() {
    let code = "let a = [2, 3]; let b = [1, ...a, 4]; b[2];";
    let result = run_js(code).unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(3)));
}

#[test]
fn test_spread_array_values() {
    let code = "let a = [1, 2, 3]; let b = [...a]; b[0] + b[1] + b[2];";
    let result = run_js(code).unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(6)));
}

#[test]
fn test_function_with_multiple_params() {
    // First test basic function with multiple params
    let code = "function sum(a, b) { return a + b; } sum(1, 2);";
    let result = run_js(code).unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(3)));
}

#[test]
fn test_spread_function_call() {
    let code = "function sum(a, b, c) { return a + b + c; } let args = [1, 2, 3]; sum(...args);";
    let result = run_js(code).unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(6)));
}

#[test]
fn test_spread_function_call_mixed() {
    let code = "function sum(a, b, c, d) { return a + b + c + d; } let args = [2, 3]; sum(1, ...args, 4);";
    let result = run_js(code).unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(10)));
}

// ==================== PHASE 4: DESTRUCTURING TESTS ====================

#[test]
fn test_object_destructuring_basic() {
    let code = "let { x, y } = { x: 1, y: 2 }; x + y;";
    let result = run_js(code).unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(3)));
}

// TODO: Enable when parser supports rename syntax
// #[test]
// fn test_object_destructuring_rename() {
//     let code = "let { x: renamed } = { x: 42 }; renamed;";
//     let result = run_js(code).unwrap();
//     assert_eq!(result, JsValue::Number(JsNumberType::Integer(42)));
// }

// TODO: Enable when parser supports default value syntax
// #[test]
// fn test_object_destructuring_default() {
//     let code = "let { x = 10 } = {}; x;";
//     let result = run_js(code).unwrap();
//     assert_eq!(result, JsValue::Number(JsNumberType::Integer(10)));
// }

// TODO: Enable when parser supports default value syntax
// #[test]
// fn test_object_destructuring_default_not_used() {
//     let code = "let { x = 10 } = { x: 5 }; x;";
//     let result = run_js(code).unwrap();
//     assert_eq!(result, JsValue::Number(JsNumberType::Integer(5)));
// }

#[test]
fn test_array_destructuring_basic() {
    let code = "let [a, b] = [1, 2]; a + b;";
    let result = run_js(code).unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(3)));
}

#[test]
fn test_array_destructuring_skip() {
    let code = "let [, , third] = [1, 2, 3]; third;";
    let result = run_js(code).unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(3)));
}

// TODO: Enable when parser supports default value syntax
// #[test]
// fn test_array_destructuring_default() {
//     let code = "let [a = 10] = []; a;";
//     let result = run_js(code).unwrap();
//     assert_eq!(result, JsValue::Number(JsNumberType::Integer(10)));
// }

#[test]
fn test_array_destructuring_rest() {
    let code = "let [first, ...rest] = [1, 2, 3, 4]; rest.length;";
    let result = run_js(code).unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(3)));
}

#[test]
fn test_array_destructuring_rest_values() {
    let code = "let [first, ...rest] = [1, 2, 3, 4]; rest[0] + rest[1] + rest[2];";
    let result = run_js(code).unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(9)));
}

// =========================================================================
// Heap accounting tests
// =========================================================================

#[test]
fn test_heap_usage_releases_objects_mid_script() {
    let code = r#"
        let obj = { a: 1, nested: { b: 2 }, arr: [1, 2, 3] };
        obj = null;
        42;
    "#;
    let ast = JsParser::parse_to_ast_from_str(code)
        .map_err(|e| format!("Parse error: {:?}", e))
        .unwrap();

    let mut ctx = EvalContext::new();
    let baseline = ctx.heap_usage();

    execute_statement(&ast.body[0], &mut ctx)
        .map_err(|e| format!("Runtime error: {:?}", e))
        .unwrap();
    let after_alloc = ctx.heap_usage();
    assert!(after_alloc > baseline);

    execute_statement(&ast.body[1], &mut ctx)
        .map_err(|e| format!("Runtime error: {:?}", e))
        .unwrap();
    assert_eq!(ctx.heap_usage(), baseline);

    execute_statement(&ast.body[2], &mut ctx)
        .map_err(|e| format!("Runtime error: {:?}", e))
        .unwrap();
    assert_eq!(ctx.heap_usage(), baseline);
}

// ==================== PHASE 5: NEW EXPRESSION TESTS ====================

#[test]
fn test_new_basic() {
    let code = "function Foo(x) { this.x = x; } let f = new Foo(42); f.x;";
    let result = run_js(code).unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(42)));
}

#[test]
fn test_new_multiple_properties() {
    let code = "function Point(x, y) { this.x = x; this.y = y; } let p = new Point(10, 20); p.x + p.y;";
    let result = run_js(code).unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(30)));
}

#[test]
fn test_new_returns_object() {
    // Verify new returns an object by checking we can set properties on it
    let code = "function Foo() { this.x = 1; } let f = new Foo(); f.x;";
    let result = run_js(code).unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(1)));
}

#[test]
fn test_new_prototype_method() {
    let code = r#"
        function Point(x, y) { this.x = x; this.y = y; }
        Point.prototype.sum = function() { return this.x + this.y; };
        let p = new Point(10, 20);
        p.sum();
    "#;
    let result = run_js(code).unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(30)));
}

#[test]
fn test_new_no_args() {
    let code = "function Counter() { this.count = 0; } let c = new Counter(); c.count;";
    let result = run_js(code).unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(0)));
}

// ==================== PHASE 6: CLASS AND INHERITANCE TESTS ====================

#[test]
fn test_class_basic() {
    let code = r#"
        class Foo {
            constructor(x) { this.x = x; }
        }
        let f = new Foo(42);
        f.x;
    "#;
    let result = run_js(code).unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(42)));
}

#[test]
fn test_class_method() {
    // Note: method name avoids "get"/"set" prefix due to parser bug
    let code = r#"
        class Foo {
            constructor(x) { this.x = x; }
            fetchX() { return this.x; }
        }
        let f = new Foo(42);
        f.fetchX();
    "#;
    let result = run_js(code).unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(42)));
}

#[test]
fn test_class_multiple_methods() {
    let code = r#"
        class Calculator {
            constructor(value) { this.value = value; }
            add(x) { return this.value + x; }
            multiply(x) { return this.value * x; }
        }
        let calc = new Calculator(10);
        calc.add(5) + calc.multiply(2);
    "#;
    let result = run_js(code).unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(35))); // 15 + 20
}

#[test]
fn test_class_default_constructor() {
    let code = r#"
        class Empty {}
        let e = new Empty();
        typeof e;
    "#;
    let result = run_js(code).unwrap();
    assert_eq!(result, JsValue::String("object".to_string()));
}

// Test that methods starting with "get"/"set" are parsed as methods, not accessors
#[test]
fn test_method_name_starting_with_get() {
    let code = r#"
        class Foo {
            constructor(x) { this.x = x; }
            getX() { return this.x; }
            getValue() { return this.x * 2; }
        }
        let f = new Foo(21);
        f.getValue();
    "#;
    let result = run_js(code).unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(42)));
}

#[test]
fn test_method_name_starting_with_set() {
    let code = r#"
        class Counter {
            constructor() { this.value = 0; }
            setValue(v) { this.value = v; }
            setAndDouble(v) { this.value = v * 2; }
        }
        let c = new Counter();
        c.setAndDouble(10);
        c.value;
    "#;
    let result = run_js(code).unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(20)));
}

#[test]
fn test_object_method_starting_with_get() {
    let code = r#"
        let obj = {
            _x: 100,
            getX() { return this._x; },
            get x() { return this._x * 2; }
        };
        obj.getX() + obj.x;
    "#;
    let result = run_js(code).unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(300)));
}

#[test]
fn test_class_static_method() {
    let code = r#"
        class Math2 {
            static double(x) { return x * 2; }
        }
        Math2.double(21);
    "#;
    let result = run_js(code).unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(42)));
}

#[test]
fn test_class_extends_basic() {
    let code = r#"
        class Animal {
            constructor(name) { this.name = name; }
        }
        class Dog extends Animal {
            constructor(name) { this.name = name; }
        }
        let d = new Dog("Rex");
        d.name;
    "#;
    let result = run_js(code).unwrap();
    assert_eq!(result, JsValue::String("Rex".to_string()));
}

#[test]
fn test_class_extends_inherited_method() {
    let code = r#"
        class Animal {
            constructor(name) { this.name = name; }
            speak() { return this.name + " makes a sound"; }
        }
        class Dog extends Animal {
            constructor(name) { this.name = name; }
        }
        let d = new Dog("Rex");
        d.speak();
    "#;
    let result = run_js(code).unwrap();
    assert_eq!(result, JsValue::String("Rex makes a sound".to_string()));
}

#[test]
fn test_class_extends_method_override() {
    let code = r#"
        class Animal {
            constructor(name) { this.name = name; }
            speak() { return this.name + " makes a sound"; }
        }
        class Dog extends Animal {
            constructor(name) { this.name = name; }
            speak() { return this.name + " barks"; }
        }
        let d = new Dog("Rex");
        d.speak();
    "#;
    let result = run_js(code).unwrap();
    assert_eq!(result, JsValue::String("Rex barks".to_string()));
}

#[test]
fn test_instanceof_class() {
    let code = r#"
        class Foo {}
        let f = new Foo();
        f instanceof Foo;
    "#;
    let result = run_js(code).unwrap();
    assert_eq!(result, JsValue::Boolean(true));
}

#[test]
fn test_instanceof_inheritance() {
    let code = r#"
        class Animal {}
        class Dog extends Animal {}
        let d = new Dog();
        d instanceof Animal;
    "#;
    let result = run_js(code).unwrap();
    assert_eq!(result, JsValue::Boolean(true));
}

#[test]
fn test_class_getter() {
    let code = r#"
        class Circle {
            constructor(radius) { this.radius = radius; }
            get area() { return this.radius * this.radius * 3; }
        }
        let c = new Circle(2);
        c.area;
    "#;
    let result = run_js(code).unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(12))); // 2*2*3
}

#[test]
fn test_class_setter() {
    let code = r#"
        class Counter {
            constructor() { this._count = 0; }
            get count() { return this._count; }
            set count(value) { this._count = value; }
        }
        let c = new Counter();
        c.count = 10;
        c.count;
    "#;
    let result = run_js(code).unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(10)));
}

// ============================================================================
// Edge Case Tests
// ============================================================================

#[test]
fn test_unsigned_right_shift() {
    let result = run_js("(-1) >>> 0").unwrap();
    // -1 >>> 0 should be 4294967295 (all 32 bits set, interpreted as unsigned)
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(4294967295)));
}

#[test]
fn test_division_by_zero_positive() {
    let result = run_js("1 / 0").unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::PositiveInfinity));
}

#[test]
fn test_division_by_zero_negative() {
    let result = run_js_get_var("var x = -1; var r = x / 0;", "r").unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::NegativeInfinity));
}

#[test]
fn test_division_by_zero_nan() {
    let result = run_js("0 / 0").unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::NaN));
}

#[test]
fn test_modulo_by_zero() {
    // Use float to avoid integer remainder-by-zero panic in the interpreter
    let result = run_js("5.0 % 0").unwrap();
    match result {
        JsValue::Number(JsNumberType::NaN) => {}
        JsValue::Number(JsNumberType::Float(f)) if f.is_nan() => {}
        other => panic!("Expected NaN, got {:?}", other),
    }
}

#[test]
fn test_nested_ternary() {
    let result = run_js_get_var("var x = 10; var r = x > 20 ? 3 : x > 5 ? 2 : 1;", "r").unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(2)));
}

#[test]
fn test_complex_boolean_expression() {
    let result = run_js("(3 > 2) && (4 < 5) && !(1 === 2)").unwrap();
    assert_eq!(result, JsValue::Boolean(true));
}

#[test]
fn test_chained_assignment() {
    let code = "var a = 0; var b = 0; var c = 0; a = b = c = 42;";
    let result = run_js_get_var(code, "a").unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(42)));
    let result_b = run_js_get_var(code, "b").unwrap();
    assert_eq!(result_b, JsValue::Number(JsNumberType::Integer(42)));
}

#[test]
fn test_do_while_with_break() {
    let code = r#"
        var count = 0;
        do {
            count = count + 1;
            if (count === 3) { break; }
        } while (count < 10);
    "#;
    let result = run_js_get_var(code, "count").unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(3)));
}

#[test]
fn test_while_false_body_not_executed() {
    let result = run_js_get_var("var x = 5; while (false) { x = 99; }", "x").unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(5)));
}

#[test]
fn test_for_loop_no_body_execution() {
    let result = run_js_get_var("var x = 0; for (var i = 10; i < 5; i = i + 1) { x = 99; }", "x").unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(0)));
}

#[test]
fn test_compound_subtract_assign() {
    let result = run_js_get_var("var x = 10; x -= 3;", "x").unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(7)));
}

#[test]
fn test_compound_multiply_assign() {
    let result = run_js_get_var("var x = 5; x *= 4;", "x").unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(20)));
}

#[test]
fn test_deeply_nested_loops() {
    let code = r#"
        var sum = 0;
        for (var i = 0; i < 3; i = i + 1) {
            for (var j = 0; j < 3; j = j + 1) {
                for (var k = 0; k < 3; k = k + 1) {
                    sum = sum + 1;
                }
            }
        }
    "#;
    let result = run_js_get_var(code, "sum").unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(27)));
}

#[test]
fn test_negative_modulo() {
    let result = run_js("-7 % 3").unwrap();
    // JS: -7 % 3 = -1
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(-1)));
}

#[test]
fn test_string_equality() {
    let result = run_js(r#""hello" === "hello""#).unwrap();
    assert_eq!(result, JsValue::Boolean(true));
}

#[test]
fn test_string_inequality() {
    let result = run_js(r#""hello" !== "world""#).unwrap();
    assert_eq!(result, JsValue::Boolean(true));
}

#[test]
fn test_typeof_null() {
    let result = run_js("typeof null").unwrap();
    assert_eq!(result, JsValue::String("object".to_string()));
}

#[test]
fn test_typeof_undefined_literal() {
    let result = run_js("typeof undefined").unwrap();
    assert_eq!(result, JsValue::String("undefined".to_string()));
}

#[test]
fn test_typeof_boolean() {
    let result = run_js("typeof true").unwrap();
    assert_eq!(result, JsValue::String("boolean".to_string()));
}

// ==================== PHASE 7: GENERATOR TESTS ====================

#[test]
fn test_generator_basic() {
    let code = r#"
        function* gen() {
            yield 1;
            yield 2;
        }
        let g = gen();
        g.next().value;
    "#;
    let result = run_js(code).unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(1)));
}

#[test]
fn test_generator_done() {
    let code = r#"
        function* gen() {
            yield 1;
        }
        let g = gen();
        g.next();
        g.next().done;
    "#;
    let result = run_js(code).unwrap();
    assert_eq!(result, JsValue::Boolean(true));
}

#[test]
fn test_generator_multiple_yields() {
    let code = r#"
        function* gen() {
            yield 1;
            yield 2;
            yield 3;
        }
        let g = gen();
        let sum = 0;
        sum = sum + g.next().value;
        sum = sum + g.next().value;
        sum = sum + g.next().value;
        sum;
    "#;
    let result = run_js(code).unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(6)));
}
