//! Tests for standard library built-in functions.
//!
//! These tests verify the functionality of built-in objects
//! like Math, String, Number, and JSON.

extern crate just;

use just::runner::plugin::registry::BuiltInRegistry;
use just::runner::plugin::types::EvalContext;
use just::runner::ds::value::{JsValue, JsNumberType};

// ============================================================================
// Math tests
// ============================================================================

mod math_tests {
    use super::*;

    fn call_math_method(registry: &BuiltInRegistry, method: &str, args: Vec<JsValue>) -> JsValue {
        let mut ctx = EvalContext::new();
        registry
            .get_method("Math", method)
            .expect(&format!("Math.{} should exist", method))
            .call(&mut ctx, JsValue::Undefined, args)
            .expect(&format!("Math.{} should succeed", method))
    }

    #[test]
    fn test_math_abs_positive() {
        let registry = BuiltInRegistry::with_core();
        let result = call_math_method(&registry, "abs", vec![JsValue::Number(JsNumberType::Integer(5))]);
        assert_eq!(result, JsValue::Number(JsNumberType::Integer(5)));
    }

    #[test]
    fn test_math_abs_negative() {
        let registry = BuiltInRegistry::with_core();
        let result = call_math_method(&registry, "abs", vec![JsValue::Number(JsNumberType::Integer(-5))]);
        assert_eq!(result, JsValue::Number(JsNumberType::Integer(5)));
    }

    #[test]
    fn test_math_floor() {
        let registry = BuiltInRegistry::with_core();
        let result = call_math_method(&registry, "floor", vec![JsValue::Number(JsNumberType::Float(3.7))]);
        assert_eq!(result, JsValue::Number(JsNumberType::Integer(3)));
    }

    #[test]
    fn test_math_ceil() {
        let registry = BuiltInRegistry::with_core();
        let result = call_math_method(&registry, "ceil", vec![JsValue::Number(JsNumberType::Float(3.2))]);
        assert_eq!(result, JsValue::Number(JsNumberType::Integer(4)));
    }

    #[test]
    fn test_math_round_down() {
        let registry = BuiltInRegistry::with_core();
        let result = call_math_method(&registry, "round", vec![JsValue::Number(JsNumberType::Float(3.4))]);
        assert_eq!(result, JsValue::Number(JsNumberType::Integer(3)));
    }

    #[test]
    fn test_math_round_up() {
        let registry = BuiltInRegistry::with_core();
        let result = call_math_method(&registry, "round", vec![JsValue::Number(JsNumberType::Float(3.6))]);
        assert_eq!(result, JsValue::Number(JsNumberType::Integer(4)));
    }

    #[test]
    fn test_math_min() {
        let registry = BuiltInRegistry::with_core();
        let result = call_math_method(&registry, "min", vec![
            JsValue::Number(JsNumberType::Integer(5)),
            JsValue::Number(JsNumberType::Integer(3)),
            JsValue::Number(JsNumberType::Integer(8)),
        ]);
        assert_eq!(result, JsValue::Number(JsNumberType::Integer(3)));
    }

    #[test]
    fn test_math_max() {
        let registry = BuiltInRegistry::with_core();
        let result = call_math_method(&registry, "max", vec![
            JsValue::Number(JsNumberType::Integer(5)),
            JsValue::Number(JsNumberType::Integer(3)),
            JsValue::Number(JsNumberType::Integer(8)),
        ]);
        assert_eq!(result, JsValue::Number(JsNumberType::Integer(8)));
    }

    #[test]
    fn test_math_sqrt() {
        let registry = BuiltInRegistry::with_core();
        let result = call_math_method(&registry, "sqrt", vec![JsValue::Number(JsNumberType::Integer(16))]);
        assert_eq!(result, JsValue::Number(JsNumberType::Integer(4)));
    }

    #[test]
    fn test_math_pow() {
        let registry = BuiltInRegistry::with_core();
        let result = call_math_method(&registry, "pow", vec![
            JsValue::Number(JsNumberType::Integer(2)),
            JsValue::Number(JsNumberType::Integer(10)),
        ]);
        assert_eq!(result, JsValue::Number(JsNumberType::Integer(1024)));
    }

    #[test]
    fn test_math_trunc_positive() {
        let registry = BuiltInRegistry::with_core();
        let result = call_math_method(&registry, "trunc", vec![JsValue::Number(JsNumberType::Float(3.9))]);
        assert_eq!(result, JsValue::Number(JsNumberType::Integer(3)));
    }

    #[test]
    fn test_math_trunc_negative() {
        let registry = BuiltInRegistry::with_core();
        let result = call_math_method(&registry, "trunc", vec![JsValue::Number(JsNumberType::Float(-3.9))]);
        assert_eq!(result, JsValue::Number(JsNumberType::Integer(-3)));
    }

    #[test]
    fn test_math_sign_positive() {
        let registry = BuiltInRegistry::with_core();
        let result = call_math_method(&registry, "sign", vec![JsValue::Number(JsNumberType::Integer(5))]);
        assert_eq!(result, JsValue::Number(JsNumberType::Integer(1)));
    }

    #[test]
    fn test_math_sign_negative() {
        let registry = BuiltInRegistry::with_core();
        let result = call_math_method(&registry, "sign", vec![JsValue::Number(JsNumberType::Integer(-5))]);
        assert_eq!(result, JsValue::Number(JsNumberType::Integer(-1)));
    }

    #[test]
    fn test_math_sign_zero() {
        let registry = BuiltInRegistry::with_core();
        let result = call_math_method(&registry, "sign", vec![JsValue::Number(JsNumberType::Integer(0))]);
        assert_eq!(result, JsValue::Number(JsNumberType::Integer(0)));
    }

    #[test]
    fn test_math_random() {
        let registry = BuiltInRegistry::with_core();
        let result = call_math_method(&registry, "random", vec![]);
        match result {
            JsValue::Number(JsNumberType::Float(f)) => {
                assert!(f >= 0.0 && f < 1.0, "Math.random should return value in [0, 1)");
            }
            _ => panic!("Math.random should return a float"),
        }
    }
}

// ============================================================================
// String tests
// ============================================================================

mod string_tests {
    use super::*;

    fn call_string_method(registry: &BuiltInRegistry, method: &str, this: JsValue, args: Vec<JsValue>) -> JsValue {
        let mut ctx = EvalContext::new();
        registry
            .get_method("String", method)
            .expect(&format!("String.{} should exist", method))
            .call(&mut ctx, this, args)
            .expect(&format!("String.{} should succeed", method))
    }

    #[test]
    fn test_string_char_at() {
        let registry = BuiltInRegistry::with_core();
        let result = call_string_method(
            &registry,
            "charAt",
            JsValue::String("hello".to_string()),
            vec![JsValue::Number(JsNumberType::Integer(1))],
        );
        assert_eq!(result, JsValue::String("e".to_string()));
    }

    #[test]
    fn test_string_char_at_out_of_bounds() {
        let registry = BuiltInRegistry::with_core();
        let result = call_string_method(
            &registry,
            "charAt",
            JsValue::String("hello".to_string()),
            vec![JsValue::Number(JsNumberType::Integer(10))],
        );
        assert_eq!(result, JsValue::String(String::new()));
    }

    #[test]
    fn test_string_char_code_at() {
        let registry = BuiltInRegistry::with_core();
        let result = call_string_method(
            &registry,
            "charCodeAt",
            JsValue::String("A".to_string()),
            vec![JsValue::Number(JsNumberType::Integer(0))],
        );
        assert_eq!(result, JsValue::Number(JsNumberType::Integer(65)));
    }

    #[test]
    fn test_string_substring() {
        let registry = BuiltInRegistry::with_core();
        let result = call_string_method(
            &registry,
            "substring",
            JsValue::String("hello world".to_string()),
            vec![
                JsValue::Number(JsNumberType::Integer(0)),
                JsValue::Number(JsNumberType::Integer(5)),
            ],
        );
        assert_eq!(result, JsValue::String("hello".to_string()));
    }

    #[test]
    fn test_string_slice() {
        let registry = BuiltInRegistry::with_core();
        let result = call_string_method(
            &registry,
            "slice",
            JsValue::String("hello world".to_string()),
            vec![JsValue::Number(JsNumberType::Integer(6))],
        );
        assert_eq!(result, JsValue::String("world".to_string()));
    }

    #[test]
    fn test_string_slice_negative() {
        let registry = BuiltInRegistry::with_core();
        let result = call_string_method(
            &registry,
            "slice",
            JsValue::String("hello world".to_string()),
            vec![JsValue::Number(JsNumberType::Integer(-5))],
        );
        assert_eq!(result, JsValue::String("world".to_string()));
    }

    #[test]
    fn test_string_index_of() {
        let registry = BuiltInRegistry::with_core();
        let result = call_string_method(
            &registry,
            "indexOf",
            JsValue::String("hello world".to_string()),
            vec![JsValue::String("world".to_string())],
        );
        assert_eq!(result, JsValue::Number(JsNumberType::Integer(6)));
    }

    #[test]
    fn test_string_index_of_not_found() {
        let registry = BuiltInRegistry::with_core();
        let result = call_string_method(
            &registry,
            "indexOf",
            JsValue::String("hello world".to_string()),
            vec![JsValue::String("xyz".to_string())],
        );
        assert_eq!(result, JsValue::Number(JsNumberType::Integer(-1)));
    }

    #[test]
    fn test_string_includes() {
        let registry = BuiltInRegistry::with_core();
        let result = call_string_method(
            &registry,
            "includes",
            JsValue::String("hello world".to_string()),
            vec![JsValue::String("world".to_string())],
        );
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn test_string_includes_false() {
        let registry = BuiltInRegistry::with_core();
        let result = call_string_method(
            &registry,
            "includes",
            JsValue::String("hello world".to_string()),
            vec![JsValue::String("xyz".to_string())],
        );
        assert_eq!(result, JsValue::Boolean(false));
    }

    #[test]
    fn test_string_starts_with() {
        let registry = BuiltInRegistry::with_core();
        let result = call_string_method(
            &registry,
            "startsWith",
            JsValue::String("hello world".to_string()),
            vec![JsValue::String("hello".to_string())],
        );
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn test_string_ends_with() {
        let registry = BuiltInRegistry::with_core();
        let result = call_string_method(
            &registry,
            "endsWith",
            JsValue::String("hello world".to_string()),
            vec![JsValue::String("world".to_string())],
        );
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn test_string_trim() {
        let registry = BuiltInRegistry::with_core();
        let result = call_string_method(
            &registry,
            "trim",
            JsValue::String("  hello  ".to_string()),
            vec![],
        );
        assert_eq!(result, JsValue::String("hello".to_string()));
    }

    #[test]
    fn test_string_trim_start() {
        let registry = BuiltInRegistry::with_core();
        let result = call_string_method(
            &registry,
            "trimStart",
            JsValue::String("  hello  ".to_string()),
            vec![],
        );
        assert_eq!(result, JsValue::String("hello  ".to_string()));
    }

    #[test]
    fn test_string_trim_end() {
        let registry = BuiltInRegistry::with_core();
        let result = call_string_method(
            &registry,
            "trimEnd",
            JsValue::String("  hello  ".to_string()),
            vec![],
        );
        assert_eq!(result, JsValue::String("  hello".to_string()));
    }

    #[test]
    fn test_string_to_upper_case() {
        let registry = BuiltInRegistry::with_core();
        let result = call_string_method(
            &registry,
            "toUpperCase",
            JsValue::String("hello".to_string()),
            vec![],
        );
        assert_eq!(result, JsValue::String("HELLO".to_string()));
    }

    #[test]
    fn test_string_to_lower_case() {
        let registry = BuiltInRegistry::with_core();
        let result = call_string_method(
            &registry,
            "toLowerCase",
            JsValue::String("HELLO".to_string()),
            vec![],
        );
        assert_eq!(result, JsValue::String("hello".to_string()));
    }

    #[test]
    fn test_string_repeat() {
        let registry = BuiltInRegistry::with_core();
        let result = call_string_method(
            &registry,
            "repeat",
            JsValue::String("ab".to_string()),
            vec![JsValue::Number(JsNumberType::Integer(3))],
        );
        assert_eq!(result, JsValue::String("ababab".to_string()));
    }

    #[test]
    fn test_string_pad_start() {
        let registry = BuiltInRegistry::with_core();
        let result = call_string_method(
            &registry,
            "padStart",
            JsValue::String("5".to_string()),
            vec![
                JsValue::Number(JsNumberType::Integer(3)),
                JsValue::String("0".to_string()),
            ],
        );
        assert_eq!(result, JsValue::String("005".to_string()));
    }

    #[test]
    fn test_string_pad_end() {
        let registry = BuiltInRegistry::with_core();
        let result = call_string_method(
            &registry,
            "padEnd",
            JsValue::String("5".to_string()),
            vec![
                JsValue::Number(JsNumberType::Integer(3)),
                JsValue::String("0".to_string()),
            ],
        );
        assert_eq!(result, JsValue::String("500".to_string()));
    }

    #[test]
    fn test_string_replace() {
        let registry = BuiltInRegistry::with_core();
        let result = call_string_method(
            &registry,
            "replace",
            JsValue::String("hello world".to_string()),
            vec![
                JsValue::String("world".to_string()),
                JsValue::String("rust".to_string()),
            ],
        );
        assert_eq!(result, JsValue::String("hello rust".to_string()));
    }

    #[test]
    fn test_string_concat() {
        let registry = BuiltInRegistry::with_core();
        let result = call_string_method(
            &registry,
            "concat",
            JsValue::String("hello".to_string()),
            vec![
                JsValue::String(" ".to_string()),
                JsValue::String("world".to_string()),
            ],
        );
        assert_eq!(result, JsValue::String("hello world".to_string()));
    }
}

// ============================================================================
// Number tests
// ============================================================================

mod number_tests {
    use super::*;

    fn call_number_method(registry: &BuiltInRegistry, method: &str, this: JsValue, args: Vec<JsValue>) -> JsValue {
        let mut ctx = EvalContext::new();
        registry
            .get_method("Number", method)
            .expect(&format!("Number.{} should exist", method))
            .call(&mut ctx, this, args)
            .expect(&format!("Number.{} should succeed", method))
    }

    #[test]
    fn test_number_is_nan_true() {
        let registry = BuiltInRegistry::with_core();
        let result = call_number_method(
            &registry,
            "isNaN",
            JsValue::Undefined,
            vec![JsValue::Number(JsNumberType::NaN)],
        );
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn test_number_is_nan_false() {
        let registry = BuiltInRegistry::with_core();
        let result = call_number_method(
            &registry,
            "isNaN",
            JsValue::Undefined,
            vec![JsValue::Number(JsNumberType::Integer(42))],
        );
        assert_eq!(result, JsValue::Boolean(false));
    }

    #[test]
    fn test_number_is_finite_true() {
        let registry = BuiltInRegistry::with_core();
        let result = call_number_method(
            &registry,
            "isFinite",
            JsValue::Undefined,
            vec![JsValue::Number(JsNumberType::Integer(42))],
        );
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn test_number_is_finite_infinity() {
        let registry = BuiltInRegistry::with_core();
        let result = call_number_method(
            &registry,
            "isFinite",
            JsValue::Undefined,
            vec![JsValue::Number(JsNumberType::PositiveInfinity)],
        );
        assert_eq!(result, JsValue::Boolean(false));
    }

    #[test]
    fn test_number_is_integer_true() {
        let registry = BuiltInRegistry::with_core();
        let result = call_number_method(
            &registry,
            "isInteger",
            JsValue::Undefined,
            vec![JsValue::Number(JsNumberType::Integer(42))],
        );
        assert_eq!(result, JsValue::Boolean(true));
    }

    #[test]
    fn test_number_is_integer_float() {
        let registry = BuiltInRegistry::with_core();
        let result = call_number_method(
            &registry,
            "isInteger",
            JsValue::Undefined,
            vec![JsValue::Number(JsNumberType::Float(3.14))],
        );
        assert_eq!(result, JsValue::Boolean(false));
    }

    #[test]
    fn test_number_parse_int() {
        let registry = BuiltInRegistry::with_core();
        let result = call_number_method(
            &registry,
            "parseInt",
            JsValue::Undefined,
            vec![JsValue::String("42".to_string())],
        );
        assert_eq!(result, JsValue::Number(JsNumberType::Integer(42)));
    }

    #[test]
    fn test_number_parse_int_radix() {
        let registry = BuiltInRegistry::with_core();
        let result = call_number_method(
            &registry,
            "parseInt",
            JsValue::Undefined,
            vec![
                JsValue::String("ff".to_string()),
                JsValue::Number(JsNumberType::Integer(16)),
            ],
        );
        assert_eq!(result, JsValue::Number(JsNumberType::Integer(255)));
    }

    #[test]
    fn test_number_parse_float() {
        let registry = BuiltInRegistry::with_core();
        let result = call_number_method(
            &registry,
            "parseFloat",
            JsValue::Undefined,
            vec![JsValue::String("3.14".to_string())],
        );
        match result {
            JsValue::Number(JsNumberType::Float(f)) => assert!((f - 3.14).abs() < 0.001),
            _ => panic!("Expected float"),
        }
    }

    #[test]
    fn test_number_to_fixed() {
        let registry = BuiltInRegistry::with_core();
        let result = call_number_method(
            &registry,
            "toFixed",
            JsValue::Number(JsNumberType::Float(3.14159)),
            vec![JsValue::Number(JsNumberType::Integer(2))],
        );
        assert_eq!(result, JsValue::String("3.14".to_string()));
    }

    #[test]
    fn test_number_to_string_base_10() {
        let registry = BuiltInRegistry::with_core();
        let result = call_number_method(
            &registry,
            "toString",
            JsValue::Number(JsNumberType::Integer(42)),
            vec![],
        );
        assert_eq!(result, JsValue::String("42".to_string()));
    }

    #[test]
    fn test_number_to_string_base_2() {
        let registry = BuiltInRegistry::with_core();
        let result = call_number_method(
            &registry,
            "toString",
            JsValue::Number(JsNumberType::Integer(10)),
            vec![JsValue::Number(JsNumberType::Integer(2))],
        );
        assert_eq!(result, JsValue::String("1010".to_string()));
    }

    #[test]
    fn test_number_to_string_base_16() {
        let registry = BuiltInRegistry::with_core();
        let result = call_number_method(
            &registry,
            "toString",
            JsValue::Number(JsNumberType::Integer(255)),
            vec![JsValue::Number(JsNumberType::Integer(16))],
        );
        assert_eq!(result, JsValue::String("ff".to_string()));
    }
}

// ============================================================================
// JSON tests
// ============================================================================

mod json_tests {
    use super::*;

    fn call_json_method(registry: &BuiltInRegistry, method: &str, args: Vec<JsValue>) -> Result<JsValue, String> {
        let mut ctx = EvalContext::new();
        registry
            .get_method("JSON", method)
            .expect(&format!("JSON.{} should exist", method))
            .call(&mut ctx, JsValue::Undefined, args)
            .map_err(|e| e.to_string())
    }

    #[test]
    fn test_json_stringify_number() {
        let registry = BuiltInRegistry::with_core();
        let result = call_json_method(
            &registry,
            "stringify",
            vec![JsValue::Number(JsNumberType::Integer(42))],
        )
        .unwrap();
        assert_eq!(result, JsValue::String("42".to_string()));
    }

    #[test]
    fn test_json_stringify_string() {
        let registry = BuiltInRegistry::with_core();
        let result = call_json_method(
            &registry,
            "stringify",
            vec![JsValue::String("hello".to_string())],
        )
        .unwrap();
        assert_eq!(result, JsValue::String("\"hello\"".to_string()));
    }

    #[test]
    fn test_json_stringify_boolean() {
        let registry = BuiltInRegistry::with_core();
        let result = call_json_method(&registry, "stringify", vec![JsValue::Boolean(true)]).unwrap();
        assert_eq!(result, JsValue::String("true".to_string()));
    }

    #[test]
    fn test_json_stringify_null() {
        let registry = BuiltInRegistry::with_core();
        let result = call_json_method(&registry, "stringify", vec![JsValue::Null]).unwrap();
        assert_eq!(result, JsValue::String("null".to_string()));
    }

    #[test]
    fn test_json_stringify_nan() {
        let registry = BuiltInRegistry::with_core();
        let result = call_json_method(
            &registry,
            "stringify",
            vec![JsValue::Number(JsNumberType::NaN)],
        )
        .unwrap();
        assert_eq!(result, JsValue::String("null".to_string()));
    }

    #[test]
    fn test_json_stringify_infinity() {
        let registry = BuiltInRegistry::with_core();
        let result = call_json_method(
            &registry,
            "stringify",
            vec![JsValue::Number(JsNumberType::PositiveInfinity)],
        )
        .unwrap();
        assert_eq!(result, JsValue::String("null".to_string()));
    }

    #[test]
    fn test_json_stringify_undefined() {
        let registry = BuiltInRegistry::with_core();
        let result = call_json_method(&registry, "stringify", vec![JsValue::Undefined]).unwrap();
        assert_eq!(result, JsValue::Undefined);
    }

    #[test]
    fn test_json_stringify_escape_quotes() {
        let registry = BuiltInRegistry::with_core();
        let result = call_json_method(
            &registry,
            "stringify",
            vec![JsValue::String("hello \"world\"".to_string())],
        )
        .unwrap();
        assert_eq!(result, JsValue::String("\"hello \\\"world\\\"\"".to_string()));
    }

    #[test]
    fn test_json_stringify_escape_newline() {
        let registry = BuiltInRegistry::with_core();
        let result = call_json_method(
            &registry,
            "stringify",
            vec![JsValue::String("line1\nline2".to_string())],
        )
        .unwrap();
        assert_eq!(result, JsValue::String("\"line1\\nline2\"".to_string()));
    }
}

// ============================================================================
// Console tests
// ============================================================================

mod console_tests {
    use super::*;

    #[test]
    fn test_console_log_exists() {
        let registry = BuiltInRegistry::with_core();
        assert!(registry.get_method("console", "log").is_some());
    }

    #[test]
    fn test_console_error_exists() {
        let registry = BuiltInRegistry::with_core();
        assert!(registry.get_method("console", "error").is_some());
    }

    #[test]
    fn test_console_warn_exists() {
        let registry = BuiltInRegistry::with_core();
        assert!(registry.get_method("console", "warn").is_some());
    }

    #[test]
    fn test_console_info_exists() {
        let registry = BuiltInRegistry::with_core();
        assert!(registry.get_method("console", "info").is_some());
    }

    #[test]
    fn test_console_log_returns_undefined() {
        let registry = BuiltInRegistry::with_core();
        let mut ctx = EvalContext::new();
        let result = registry
            .get_method("console", "log")
            .unwrap()
            .call(&mut ctx, JsValue::Undefined, vec![JsValue::String("test".to_string())])
            .unwrap();
        assert_eq!(result, JsValue::Undefined);
    }
}

// ============================================================================
// Object tests
// ============================================================================

mod object_tests {
    use super::*;

    fn call_object_method(registry: &BuiltInRegistry, method: &str, this: JsValue, args: Vec<JsValue>) -> JsValue {
        let mut ctx = EvalContext::new();
        registry
            .get_method("Object", method)
            .expect(&format!("Object.{} should exist", method))
            .call(&mut ctx, this, args)
            .expect(&format!("Object.{} should succeed", method))
    }

    #[test]
    fn test_object_to_string_undefined() {
        let registry = BuiltInRegistry::with_core();
        let result = call_object_method(&registry, "toString", JsValue::Undefined, vec![]);
        assert_eq!(result, JsValue::String("[object Undefined]".to_string()));
    }

    #[test]
    fn test_object_to_string_null() {
        let registry = BuiltInRegistry::with_core();
        let result = call_object_method(&registry, "toString", JsValue::Null, vec![]);
        assert_eq!(result, JsValue::String("[object Null]".to_string()));
    }

    #[test]
    fn test_object_to_string_boolean() {
        let registry = BuiltInRegistry::with_core();
        let result = call_object_method(&registry, "toString", JsValue::Boolean(true), vec![]);
        assert_eq!(result, JsValue::String("[object Boolean]".to_string()));
    }

    #[test]
    fn test_object_to_string_number() {
        let registry = BuiltInRegistry::with_core();
        let result = call_object_method(
            &registry,
            "toString",
            JsValue::Number(JsNumberType::Integer(42)),
            vec![],
        );
        assert_eq!(result, JsValue::String("[object Number]".to_string()));
    }

    #[test]
    fn test_object_to_string_string() {
        let registry = BuiltInRegistry::with_core();
        let result = call_object_method(
            &registry,
            "toString",
            JsValue::String("hello".to_string()),
            vec![],
        );
        assert_eq!(result, JsValue::String("[object String]".to_string()));
    }

    #[test]
    fn test_object_value_of() {
        let registry = BuiltInRegistry::with_core();
        let result = call_object_method(
            &registry,
            "valueOf",
            JsValue::Number(JsNumberType::Integer(42)),
            vec![],
        );
        assert_eq!(result, JsValue::Number(JsNumberType::Integer(42)));
    }
}

// ============================================================================
// Registry tests
// ============================================================================

mod registry_tests {
    use super::*;

    #[test]
    fn test_registry_has_core_objects() {
        let registry = BuiltInRegistry::with_core();
        assert!(registry.has_object("console"));
        assert!(registry.has_object("Object"));
        assert!(registry.has_object("Array"));
        assert!(registry.has_object("String"));
        assert!(registry.has_object("Number"));
        assert!(registry.has_object("Math"));
        assert!(registry.has_object("JSON"));
        assert!(registry.has_object("Error"));
    }

    #[test]
    fn test_registry_has_error_types() {
        let registry = BuiltInRegistry::with_core();
        assert!(registry.has_object("TypeError"));
        assert!(registry.has_object("ReferenceError"));
        assert!(registry.has_object("SyntaxError"));
        assert!(registry.has_object("RangeError"));
    }

    #[test]
    fn test_registry_get_object() {
        let registry = BuiltInRegistry::with_core();
        let math = registry.get_object("Math");
        assert!(math.is_some());
        assert_eq!(math.unwrap().name, "Math");
    }

    #[test]
    fn test_registry_has_method() {
        let registry = BuiltInRegistry::with_core();
        assert!(registry.has_method("Math", "abs"));
        assert!(registry.has_method("Math", "floor"));
        assert!(registry.has_method("String", "charAt"));
        assert!(registry.has_method("console", "log"));
    }

    #[test]
    fn test_registry_object_names() {
        let registry = BuiltInRegistry::with_core();
        let names = registry.object_names();
        assert!(names.len() >= 8); // At least 8 core objects
    }
}
