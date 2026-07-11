//! Integration tests for BigInt (ES2020) — literal parsing, arithmetic,
//! coercion rules, comparison/equality, bitwise ops, and the `BigInt()` global.
//!
//! Uses the full parse → interpret pipeline (the same path test262 exercises).

use just_engine::parser::JsParser;
use just_engine::runner::ds::value::JsValue;
use just_engine::runner::eval::statement::execute_statement;
use just_engine::runner::plugin::registry::BuiltInRegistry;
use just_engine::runner::plugin::types::EvalContext;

/// Parse and execute JS, returning the completion value of the last statement.
fn run_js(code: &str) -> Result<JsValue, String> {
    let ast = JsParser::parse_to_ast_from_str(code).map_err(|e| format!("Parse error: {:?}", e))?;
    let mut ctx = EvalContext::new();
    // Install the core built-ins (String, BigInt, Number, …) so global
    // function calls resolve — the same setup the test262 harness uses.
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

/// Assert the code evaluates to `JsValue::BigInt(expected_decimal)`.
fn assert_bigint(code: &str, expected_decimal: &str) {
    match run_js(code) {
        Ok(JsValue::BigInt(b)) => assert_eq!(
            b.to_string(),
            expected_decimal,
            "code `{}` -> {} (expected {})",
            code,
            b,
            expected_decimal
        ),
        other => panic!("code `{}` did not produce a BigInt: {:?}", code, other),
    }
}

// ---------------------------------------------------------------------------
// Literal parsing (radices, separators, and rejected forms)
// ---------------------------------------------------------------------------

#[test]
fn literal_decimal() {
    assert_bigint("10n", "10");
    assert_bigint("0n", "0");
    assert_bigint("123456789012345678901234567890n", "123456789012345678901234567890");
}

#[test]
fn literal_radices() {
    assert_bigint("0xFFn", "255");
    assert_bigint("0b101n", "5");
    assert_bigint("0o17n", "15");
    assert_bigint("0XABn", "171");
}

#[test]
fn literal_numeric_separators() {
    assert_bigint("1_000n", "1000");
    assert_bigint("0xFF_FFn", "65535");
    assert_bigint("1_2_3n", "123");
}

#[test]
fn literal_rejected_forms_fail_to_parse() {
    // Leading-zero decimals, fractions, and exponents are NOT valid BigInts.
    for bad in ["00n", "08n", "09n", "1.0n", "1e2n", "0.5n"] {
        assert!(
            run_js(bad).is_err(),
            "`{}` should fail to parse as a BigInt literal",
            bad
        );
    }
}

// ---------------------------------------------------------------------------
// Arithmetic
// ---------------------------------------------------------------------------

#[test]
fn arithmetic_basic() {
    assert_bigint("1n + 2n", "3");
    assert_bigint("10n - 3n", "7");
    assert_bigint("10n * 10n", "100");
    assert_bigint("20n / 3n", "6"); // truncating division
    assert_bigint("7n % 3n", "1");
    assert_bigint("2n ** 10n", "1024");
    assert_bigint("-5n", "-5");
}

#[test]
fn arithmetic_bigint_string_concat() {
    // BigInt + String is string concatenation.
    match run_js("1n + \"a\"").unwrap() {
        JsValue::String(s) => assert_eq!(s, "1a"),
        other => panic!("expected string, got {:?}", other),
    }
}

#[test]
fn arithmetic_mixed_bigint_number_throws() {
    for code in ["1n + 1", "1 + 1n", "2n * 3", "10n - 1"] {
        assert!(run_js(code).is_err(), "`{}` should be a TypeError", code);
    }
}

#[test]
fn arithmetic_div_mod_by_zero_throws() {
    assert!(run_js("1n / 0n").is_err());
    assert!(run_js("1n % 0n").is_err());
}

#[test]
fn arithmetic_negative_exponent_throws() {
    assert!(run_js("2n ** -1n").is_err());
}

#[test]
fn update_operators() {
    assert_bigint("let x = 5n; x++; x", "6");
    assert_bigint("let y = 5n; y--; y", "4");
    assert_bigint("let z = 5n; ++z", "6");
}

// ---------------------------------------------------------------------------
// Coercion
// ---------------------------------------------------------------------------

#[test]
fn typeof_bigint() {
    match run_js("typeof 1n").unwrap() {
        JsValue::String(s) => assert_eq!(s, "bigint"),
        other => panic!("expected string, got {:?}", other),
    }
}

#[test]
fn to_boolean_bigint() {
    assert_eq!(run_js("!!0n").unwrap(), JsValue::Boolean(false));
    assert_eq!(run_js("!!1n").unwrap(), JsValue::Boolean(true));
    assert_eq!(run_js("!!-5n").unwrap(), JsValue::Boolean(true));
}

#[test]
fn to_string_bigint() {
    match run_js("String(255n)").unwrap() {
        JsValue::String(s) => assert_eq!(s, "255"),
        other => panic!("expected string, got {:?}", other),
    }
}

#[test]
fn unary_plus_bigint_throws() {
    assert!(run_js("+1n").is_err());
}

#[test]
fn to_number_bigint_throws() {
    // Number(1n) routes through ToNumber, which rejects BigInt.
    assert!(run_js("Number(1n)").is_err());
}

// ---------------------------------------------------------------------------
// Comparison and equality
// ---------------------------------------------------------------------------

#[test]
fn comparison_cross_type() {
    assert_eq!(run_js("2n < 3").unwrap(), JsValue::Boolean(true));
    assert_eq!(run_js("3n > 2").unwrap(), JsValue::Boolean(true));
    assert_eq!(run_js("2n <= 2").unwrap(), JsValue::Boolean(true));
    assert_eq!(run_js("2n < 2n").unwrap(), JsValue::Boolean(false));
}

#[test]
fn equality() {
    assert_eq!(run_js("2n == 2").unwrap(), JsValue::Boolean(true));
    assert_eq!(run_js("2n == \"2\"").unwrap(), JsValue::Boolean(true));
    assert_eq!(run_js("2n === 2").unwrap(), JsValue::Boolean(false));
    assert_eq!(run_js("2n === 2n").unwrap(), JsValue::Boolean(true));
    assert_eq!(run_js("2n != 3").unwrap(), JsValue::Boolean(true));
}

// ---------------------------------------------------------------------------
// Bitwise
// ---------------------------------------------------------------------------

#[test]
fn bitwise() {
    assert_bigint("6n & 3n", "2");
    assert_bigint("6n | 1n", "7");
    assert_bigint("6n ^ 3n", "5");
    assert_bigint("~0n", "-1");
    assert_bigint("1n << 4n", "16");
    assert_bigint("256n >> 2n", "64");
}

#[test]
fn unsigned_right_shift_bigint_throws() {
    assert!(run_js("8n >>> 2n").is_err());
}

// ---------------------------------------------------------------------------
// BigInt() global
// ---------------------------------------------------------------------------

#[test]
fn bigint_global_conversions() {
    assert_bigint("BigInt(255)", "255");
    assert_bigint("BigInt(\"0xff\")", "255");
    assert_bigint("BigInt(\"10\")", "10");
    assert_bigint("BigInt(true)", "1");
    assert_bigint("BigInt(false)", "0");
    assert_bigint("BigInt(5.0)", "5");
    assert_bigint("BigInt(\"\")", "0");
}

#[test]
fn bigint_global_errors() {
    assert!(run_js("BigInt(1.5)").is_err()); // RangeError: not an integer
    assert!(run_js("BigInt(\"xyz\")").is_err()); // SyntaxError
    assert!(run_js("BigInt(undefined)").is_err()); // TypeError
    assert!(run_js("BigInt(null)").is_err()); // TypeError
}

#[test]
fn new_bigint_throws() {
    // BigInt is not a constructor.
    assert!(run_js("new BigInt(1)").is_err());
}
