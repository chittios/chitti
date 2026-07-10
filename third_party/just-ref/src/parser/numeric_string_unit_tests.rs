// ChittiOS-alloc-prelude
#[allow(unused_imports)]
use num_traits::Float;
use alloc::borrow::ToOwned;
use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use crate::parser::ast::{ExtendedNumberLiteralType, JsErrorType, NumberLiteralType};
use crate::parser::JsParser;

fn assert_parse(input: &str, expected_output: ExtendedNumberLiteralType) {
    match JsParser::parse_numeric_string(&input.to_string(), false) {
        Ok(actual_output) => {
            assert!(
                actual_output.exactly_eq(&expected_output),
                "For the input: \"{}\", the expected out was: \"{:?}\", but got: \"{:?}\" ",
                input,
                expected_output,
                actual_output
            )
        }
        Err(err) => {
            assert!(false, "Test resulted into unexpected error: {:?}", err)
        }
    }
}

fn assert_error_on_empty_parse(input: &str) {
    assert!(
        if let Err(e) = JsParser::parse_numeric_string(&input.to_string(), true) {
            if let JsErrorType::ParserGeneralError = e.kind {
                true
            } else {
                false
            }
        } else {
            false
        },
        "Was expecting error for input \"{}\" but did not get it.",
        input
    )
}

#[test]
fn test_decimal_integer_parse1() {
    assert_parse(
        "1234",
        ExtendedNumberLiteralType::Std(NumberLiteralType::IntegerLiteral(1234)),
    )
}

#[test]
fn test_decimal_integer_parse2() {
    assert_parse(
        "01234",
        ExtendedNumberLiteralType::Std(NumberLiteralType::IntegerLiteral(1234)),
    )
}

#[test]
fn test_decimal_integer_parse3() {
    assert_parse(
        "    1234",
        ExtendedNumberLiteralType::Std(NumberLiteralType::IntegerLiteral(1234)),
    )
}

#[test]
fn test_decimal_integer_parse4() {
    assert_parse(
        "    1234   5",
        ExtendedNumberLiteralType::Std(NumberLiteralType::IntegerLiteral(1234)),
    )
}

#[test]
fn test_decimal_integer_parse5() {
    assert_parse(
        "    1234 abcd",
        ExtendedNumberLiteralType::Std(NumberLiteralType::IntegerLiteral(1234)),
    )
}

#[test]
fn test_decimal_integer_parse6() {
    assert_parse(
        "  abc  1234",
        ExtendedNumberLiteralType::Std(NumberLiteralType::IntegerLiteral(0)),
    )
}

#[test]
fn test_decimal_integer_parse7() {
    assert_error_on_empty_parse("  abc  1234")
}

#[test]
fn test_empty1() {
    assert_parse(
        "  ",
        ExtendedNumberLiteralType::Std(NumberLiteralType::IntegerLiteral(0)),
    )
}

#[test]
fn test_empty2() {
    assert_parse(
        "",
        ExtendedNumberLiteralType::Std(NumberLiteralType::IntegerLiteral(0)),
    )
}

#[test]
fn test_empty3() {
    assert_error_on_empty_parse("  ")
}

#[test]
fn test_decimal_fraction_parse1() {
    assert_parse(
        "1234.5",
        ExtendedNumberLiteralType::Std(NumberLiteralType::FloatLiteral(1234.5)),
    )
}

#[test]
fn test_decimal_fraction_parse2() {
    assert_parse(
        "-1234.5",
        ExtendedNumberLiteralType::Std(NumberLiteralType::FloatLiteral(-1234.5)),
    )
}

#[test]
fn test_decimal_fraction_parse3() {
    assert_parse(
        ".5",
        ExtendedNumberLiteralType::Std(NumberLiteralType::FloatLiteral(0.5)),
    )
}

#[test]
fn test_decimal_fraction_parse4() {
    assert_parse(
        "-.5",
        ExtendedNumberLiteralType::Std(NumberLiteralType::FloatLiteral(-0.5)),
    )
}

#[test]
fn test_exponent_parse1() {
    assert_parse(
        "12e2",
        ExtendedNumberLiteralType::Std(NumberLiteralType::IntegerLiteral(1200)),
    )
}

#[test]
fn test_exponent_parse2() {
    assert_parse(
        "1.2e2",
        ExtendedNumberLiteralType::Std(NumberLiteralType::FloatLiteral(120_f64)),
    )
}

#[test]
fn test_exponent_parse3() {
    assert_parse(
        "1.2e-2",
        ExtendedNumberLiteralType::Std(NumberLiteralType::FloatLiteral(0.012_f64)),
    )
}

#[test]
fn test_infinity1() {
    assert_parse("Infinity", ExtendedNumberLiteralType::Infinity)
}

#[test]
fn test_infinity2() {
    assert_parse("-Infinity", ExtendedNumberLiteralType::NegativeInfinity)
}

#[test]
fn test_negative_decimal_integer_parse1() {
    assert_parse(
        "-1234",
        ExtendedNumberLiteralType::Std(NumberLiteralType::IntegerLiteral(-1234)),
    )
}

#[test]
fn test_negative_decimal_integer_parse2() {
    assert_parse(
        "-01234",
        ExtendedNumberLiteralType::Std(NumberLiteralType::IntegerLiteral(-1234)),
    )
}

#[test]
fn test_negative_decimal_integer_parse3() {
    assert_parse(
        "    -1234",
        ExtendedNumberLiteralType::Std(NumberLiteralType::IntegerLiteral(-1234)),
    )
}

#[test]
fn test_negative_decimal_integer_parse4() {
    assert_parse(
        "    -1234   5",
        ExtendedNumberLiteralType::Std(NumberLiteralType::IntegerLiteral(-1234)),
    )
}
