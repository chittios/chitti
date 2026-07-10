extern crate just;

use just::parser::JsParser;
use just::runner::jit;
use just::runner::ds::value::{JsNumberType, JsValue};
use just::runner::plugin::registry::BuiltInRegistry;

/// Helper: run code through the Cranelift JIT and return a variable's value.
/// Skips on non-x86_64.
fn jit_var(code: &str, var_name: &str) -> Option<JsValue> {
    if !cfg!(target_arch = "x86_64") {
        eprintln!("Skipping reg_jit test on non-x86_64 host");
        return None;
    }
    let ast = JsParser::parse_to_ast_from_str(code).unwrap();
    let (result, mut ctx) = jit::compile_and_run_reg_jit_with_ctx(&ast);
    assert!(result.is_ok(), "JIT execution failed: {:?}", result.err());
    Some(ctx.get_binding(var_name).unwrap_or(JsValue::Undefined))
}

fn assert_jit_int(code: &str, var_name: &str, expected: i64) {
    if let Some(val) = jit_var(code, var_name) {
        match val {
            JsValue::Number(JsNumberType::Integer(n)) => assert_eq!(n, expected, "code: {}", code),
            JsValue::Number(JsNumberType::Float(f)) => assert_eq!(f, expected as f64, "code: {}", code),
            other => panic!("{} was {:?}, expected {}", var_name, other, expected),
        }
    }
}

fn assert_jit_float(code: &str, var_name: &str, expected: f64) {
    if let Some(val) = jit_var(code, var_name) {
        match val {
            JsValue::Number(JsNumberType::Float(f)) => {
                assert!((f - expected).abs() < 0.001, "{} was {}, expected {}", var_name, f, expected);
            }
            JsValue::Number(JsNumberType::Integer(n)) => {
                assert!((n as f64 - expected).abs() < 0.001, "{} was {}, expected {}", var_name, n, expected);
            }
            other => panic!("{} was {:?}, expected {}", var_name, other, expected),
        }
    }
}

/// Helper: run code through JIT-or-VM fallback and return a variable's value.
fn jit_or_vm_var(code: &str, var_name: &str) -> JsValue {
    let ast = JsParser::parse_to_ast_from_str(code).unwrap();
    let chunk = jit::compile_reg(&ast);
    let ctx = just::runner::plugin::types::EvalContext::new();
    let registry = BuiltInRegistry::with_core();
    let (_result, mut ctx_out) = jit::execute_reg_jit_or_vm(&chunk, ctx, &registry).unwrap();
    ctx_out.get_binding(var_name).unwrap_or(JsValue::Undefined)
}

// ============================================================================
// Arithmetic
// ============================================================================

#[test]
fn test_reg_jit_simple_add() {
    assert_jit_int("var x = 3 + 4;", "x", 7);
}

#[test]
fn test_reg_jit_subtraction() {
    assert_jit_int("var x = 10 - 3;", "x", 7);
}

#[test]
fn test_reg_jit_multiplication() {
    assert_jit_int("var x = 6 * 7;", "x", 42);
}

#[test]
fn test_reg_jit_division() {
    assert_jit_float("var x = 10 / 4;", "x", 2.5);
}

#[test]
fn test_reg_jit_modulo() {
    assert_jit_int("var x = 17 % 5;", "x", 2);
}

#[test]
fn test_reg_jit_negate() {
    assert_jit_int("var x = -42;", "x", -42);
}

#[test]
fn test_reg_jit_unary_plus() {
    assert_jit_int("var x = +7;", "x", 7);
}

#[test]
fn test_reg_jit_complex_arithmetic() {
    assert_jit_int("var x = 2 + 3 * 4 - 1;", "x", 13);
}

// ============================================================================
// Comparison
// ============================================================================

#[test]
fn test_reg_jit_less_than_true() {
    assert_jit_int("var x = 0; if (3 < 5) { x = 1; }", "x", 1);
}

#[test]
fn test_reg_jit_less_than_false() {
    assert_jit_int("var x = 0; if (5 < 3) { x = 1; }", "x", 0);
}

#[test]
fn test_reg_jit_greater_than() {
    assert_jit_int("var x = 0; if (10 > 5) { x = 1; }", "x", 1);
}

#[test]
fn test_reg_jit_less_equal() {
    assert_jit_int("var x = 0; if (5 <= 5) { x = 1; }", "x", 1);
}

#[test]
fn test_reg_jit_greater_equal() {
    assert_jit_int("var x = 0; if (5 >= 6) { x = 1; }", "x", 0);
}

#[test]
fn test_reg_jit_strict_equal() {
    assert_jit_int("var x = 0; if (7 === 7) { x = 1; }", "x", 1);
}

#[test]
fn test_reg_jit_strict_not_equal() {
    assert_jit_int("var x = 0; if (7 !== 8) { x = 1; }", "x", 1);
}

// ============================================================================
// Bitwise
// ============================================================================

#[test]
fn test_reg_jit_bitwise_and() {
    assert_jit_int("var x = 5 & 3;", "x", 1);
}

#[test]
fn test_reg_jit_bitwise_or() {
    assert_jit_int("var x = 5 | 3;", "x", 7);
}

#[test]
fn test_reg_jit_bitwise_xor() {
    assert_jit_int("var x = 5 ^ 3;", "x", 6);
}

#[test]
fn test_reg_jit_bitwise_not() {
    assert_jit_int("var x = ~5;", "x", -6);
}

#[test]
fn test_reg_jit_shift_left() {
    assert_jit_int("var x = 1 << 4;", "x", 16);
}

#[test]
fn test_reg_jit_shift_right() {
    assert_jit_int("var x = 16 >> 2;", "x", 4);
}

#[test]
fn test_reg_jit_unsigned_shift_right() {
    assert_jit_int("var x = 16 >>> 2;", "x", 4);
}

// ============================================================================
// Logical / Not
// ============================================================================

#[test]
fn test_reg_jit_not_truthy() {
    assert_jit_int("var x = 0; if (!0) { x = 1; }", "x", 1);
}

#[test]
fn test_reg_jit_not_falsy() {
    assert_jit_int("var x = 0; if (!1) { x = 1; }", "x", 0);
}

// ============================================================================
// Variables
// ============================================================================

#[test]
fn test_reg_jit_var_declaration() {
    assert_jit_int("var x = 42;", "x", 42);
}

#[test]
fn test_reg_jit_var_reassignment() {
    assert_jit_int("var x = 1; x = 99;", "x", 99);
}

#[test]
fn test_reg_jit_multiple_vars() {
    assert_jit_int("var a = 10; var b = 20; var c = a + b;", "c", 30);
}

// ============================================================================
// Control flow
// ============================================================================

#[test]
fn test_reg_jit_if_true() {
    assert_jit_int("var x = 0; if (1) { x = 42; }", "x", 42);
}

#[test]
fn test_reg_jit_if_false() {
    assert_jit_int("var x = 0; if (0) { x = 42; }", "x", 0);
}

#[test]
fn test_reg_jit_if_else() {
    assert_jit_int("var x = 0; if (0) { x = 1; } else { x = 2; }", "x", 2);
}

#[test]
fn test_reg_jit_for_loop_sum() {
    let code = "var sum = 0; for (var i = 0; i < 10; i = i + 1) { sum = sum + i; }";
    assert_jit_int(code, "sum", 45);
}

#[test]
fn test_reg_jit_while_loop() {
    let code = "var i = 0; var sum = 0; while (i < 5) { sum = sum + i; i = i + 1; }";
    assert_jit_int(code, "sum", 10);
}

#[test]
fn test_reg_jit_nested_for() {
    let code = r#"
var count = 0;
for (var i = 0; i < 3; i = i + 1) {
    for (var j = 0; j < 4; j = j + 1) {
        count = count + 1;
    }
}
"#;
    assert_jit_int(code, "count", 12);
}

#[test]
fn test_reg_jit_fibonacci() {
    let code = r#"
var a = 0;
var b = 1;
for (var i = 0; i < 10; i = i + 1) {
    var temp = a;
    a = b;
    b = temp + b;
}
"#;
    assert_jit_int(code, "a", 55);
}

#[test]
fn test_reg_jit_factorial() {
    let code = r#"
var n = 5;
var result = 1;
for (var i = 2; i <= n; i = i + 1) {
    result = result * i;
}
"#;
    assert_jit_int(code, "result", 120);
}

#[test]
fn test_reg_jit_gcd() {
    let code = r#"
var a = 48;
var b = 18;
while (b !== 0) {
    var temp = b;
    b = a % b;
    a = temp;
}
"#;
    assert_jit_int(code, "a", 6);
}

#[test]
fn test_reg_jit_sum_of_squares() {
    let code = r#"
var sum = 0;
for (var i = 1; i <= 10; i = i + 1) {
    sum = sum + i * i;
}
"#;
    assert_jit_int(code, "sum", 385);
}

#[test]
fn test_reg_jit_prime_check() {
    let code = r#"
var n = 17;
var isPrime = 1;
for (var i = 2; i * i <= n; i = i + 1) {
    if (n % i === 0) {
        isPrime = 0;
    }
}
"#;
    assert_jit_int(code, "isPrime", 1);
}

#[test]
fn test_reg_jit_bitwise_loop() {
    let code = r#"
var val = 255;
for (var i = 0; i < 8; i = i + 1) {
    val = val & (val - 1);
}
"#;
    assert_jit_int(code, "val", 0);
}

// ============================================================================
// JIT bail → VM fallback
// ============================================================================

#[test]
fn test_reg_jit_fallback_to_vm_on_typeof() {
    // typeof triggers JIT bail; execute_reg_jit_or_vm should fall back to VM
    let val = jit_or_vm_var("var x = 42; var t = typeof x;", "t");
    assert_eq!(val, JsValue::String("number".to_string()));
}

#[test]
fn test_reg_jit_fallback_to_vm_on_typeof_number() {
    // typeof triggers JIT bail; VM handles the whole chunk
    let val = jit_or_vm_var("var x = 10; var t = typeof x;", "x");
    assert_eq!(val, JsValue::Number(JsNumberType::Integer(10)));
}
