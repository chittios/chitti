//! Tests for the expression evaluation engine.
//!
//! These tests verify the basic expression evaluation functionality
//! including literals, operators, and type conversions.

extern crate just;

use just::runner::eval::expression::{evaluate_expression, to_boolean};
use just::runner::plugin::types::EvalContext;
use just::runner::ds::value::{JsValue, JsNumberType};
use just::parser::ast::{
    ExpressionType, LiteralData, LiteralType, NumberLiteralType, Meta,
    BinaryOperator, UnaryOperator, LogicalOperator, UpdateOperator,
};
use std::rc::Rc;

/// Helper to create a simple meta for tests.
fn test_meta() -> Meta {
    Meta {
        start_index: 0,
        end_index: 0,
        script: Rc::new(String::new()),
    }
}

/// Helper to create a number literal expression.
fn num_expr(n: i64) -> ExpressionType {
    ExpressionType::Literal(LiteralData {
        meta: test_meta(),
        value: LiteralType::NumberLiteral(NumberLiteralType::IntegerLiteral(n)),
    })
}

/// Helper to create a float literal expression.
fn float_expr(f: f64) -> ExpressionType {
    ExpressionType::Literal(LiteralData {
        meta: test_meta(),
        value: LiteralType::NumberLiteral(NumberLiteralType::FloatLiteral(f)),
    })
}

/// Helper to create a string literal expression.
fn str_expr(s: &str) -> ExpressionType {
    ExpressionType::Literal(LiteralData {
        meta: test_meta(),
        value: LiteralType::StringLiteral(s.to_string()),
    })
}

/// Helper to create a boolean literal expression.
fn bool_expr(b: bool) -> ExpressionType {
    ExpressionType::Literal(LiteralData {
        meta: test_meta(),
        value: LiteralType::BooleanLiteral(b),
    })
}

/// Helper to create a null literal expression.
fn null_expr() -> ExpressionType {
    ExpressionType::Literal(LiteralData {
        meta: test_meta(),
        value: LiteralType::NullLiteral,
    })
}

/// Helper to create a binary expression.
fn binary_expr(op: BinaryOperator, left: ExpressionType, right: ExpressionType) -> ExpressionType {
    ExpressionType::BinaryExpression {
        meta: test_meta(),
        operator: op,
        left: Box::new(left),
        right: Box::new(right),
    }
}

/// Helper to create a unary expression.
fn unary_expr(op: UnaryOperator, arg: ExpressionType) -> ExpressionType {
    ExpressionType::UnaryExpression {
        meta: test_meta(),
        operator: op,
        argument: Box::new(arg),
    }
}

/// Helper to create a logical expression.
fn logical_expr(op: LogicalOperator, left: ExpressionType, right: ExpressionType) -> ExpressionType {
    ExpressionType::LogicalExpression {
        meta: test_meta(),
        operator: op,
        left: Box::new(left),
        right: Box::new(right),
    }
}

/// Helper to create a conditional expression.
fn cond_expr(test: ExpressionType, consequent: ExpressionType, alternate: ExpressionType) -> ExpressionType {
    ExpressionType::ConditionalExpression {
        meta: test_meta(),
        test: Box::new(test),
        consequent: Box::new(consequent),
        alternate: Box::new(alternate),
    }
}

// ============================================================================
// Literal tests
// ============================================================================

#[test]
fn test_number_literal() {
    let mut ctx = EvalContext::new();
    let expr = num_expr(42);
    let result = evaluate_expression(&expr, &mut ctx).unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(42)));
}

#[test]
fn test_float_literal() {
    let mut ctx = EvalContext::new();
    let expr = float_expr(3.14);
    let result = evaluate_expression(&expr, &mut ctx).unwrap();
    match result {
        JsValue::Number(JsNumberType::Float(f)) => assert!((f - 3.14).abs() < 0.001),
        _ => panic!("Expected float"),
    }
}

#[test]
fn test_string_literal() {
    let mut ctx = EvalContext::new();
    let expr = str_expr("hello");
    let result = evaluate_expression(&expr, &mut ctx).unwrap();
    assert_eq!(result, JsValue::String("hello".to_string()));
}

#[test]
fn test_boolean_literal_true() {
    let mut ctx = EvalContext::new();
    let expr = bool_expr(true);
    let result = evaluate_expression(&expr, &mut ctx).unwrap();
    assert_eq!(result, JsValue::Boolean(true));
}

#[test]
fn test_boolean_literal_false() {
    let mut ctx = EvalContext::new();
    let expr = bool_expr(false);
    let result = evaluate_expression(&expr, &mut ctx).unwrap();
    assert_eq!(result, JsValue::Boolean(false));
}

#[test]
fn test_null_literal() {
    let mut ctx = EvalContext::new();
    let expr = null_expr();
    let result = evaluate_expression(&expr, &mut ctx).unwrap();
    assert_eq!(result, JsValue::Null);
}

// ============================================================================
// Binary arithmetic tests
// ============================================================================

#[test]
fn test_addition_numbers() {
    let mut ctx = EvalContext::new();
    let expr = binary_expr(BinaryOperator::Add, num_expr(1), num_expr(2));
    let result = evaluate_expression(&expr, &mut ctx).unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(3)));
}

#[test]
fn test_addition_strings() {
    let mut ctx = EvalContext::new();
    let expr = binary_expr(BinaryOperator::Add, str_expr("hello"), str_expr(" world"));
    let result = evaluate_expression(&expr, &mut ctx).unwrap();
    assert_eq!(result, JsValue::String("hello world".to_string()));
}

#[test]
fn test_addition_mixed() {
    let mut ctx = EvalContext::new();
    let expr = binary_expr(BinaryOperator::Add, num_expr(1), str_expr("2"));
    let result = evaluate_expression(&expr, &mut ctx).unwrap();
    assert_eq!(result, JsValue::String("12".to_string()));
}

#[test]
fn test_subtraction() {
    let mut ctx = EvalContext::new();
    let expr = binary_expr(BinaryOperator::Subtract, num_expr(5), num_expr(3));
    let result = evaluate_expression(&expr, &mut ctx).unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(2)));
}

#[test]
fn test_multiplication() {
    let mut ctx = EvalContext::new();
    let expr = binary_expr(BinaryOperator::Multiply, num_expr(4), num_expr(3));
    let result = evaluate_expression(&expr, &mut ctx).unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(12)));
}

#[test]
fn test_division() {
    let mut ctx = EvalContext::new();
    let expr = binary_expr(BinaryOperator::Divide, num_expr(10), num_expr(2));
    let result = evaluate_expression(&expr, &mut ctx).unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(5)));
}

#[test]
fn test_modulo() {
    let mut ctx = EvalContext::new();
    let expr = binary_expr(BinaryOperator::Modulo, num_expr(7), num_expr(3));
    let result = evaluate_expression(&expr, &mut ctx).unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(1)));
}

// ============================================================================
// Comparison tests
// ============================================================================

#[test]
fn test_less_than_true() {
    let mut ctx = EvalContext::new();
    let expr = binary_expr(BinaryOperator::LessThan, num_expr(1), num_expr(2));
    let result = evaluate_expression(&expr, &mut ctx).unwrap();
    assert_eq!(result, JsValue::Boolean(true));
}

#[test]
fn test_less_than_false() {
    let mut ctx = EvalContext::new();
    let expr = binary_expr(BinaryOperator::LessThan, num_expr(2), num_expr(1));
    let result = evaluate_expression(&expr, &mut ctx).unwrap();
    assert_eq!(result, JsValue::Boolean(false));
}

#[test]
fn test_greater_than() {
    let mut ctx = EvalContext::new();
    let expr = binary_expr(BinaryOperator::GreaterThan, num_expr(5), num_expr(3));
    let result = evaluate_expression(&expr, &mut ctx).unwrap();
    assert_eq!(result, JsValue::Boolean(true));
}

#[test]
fn test_less_equal() {
    let mut ctx = EvalContext::new();
    let expr = binary_expr(BinaryOperator::LessThanEqual, num_expr(3), num_expr(3));
    let result = evaluate_expression(&expr, &mut ctx).unwrap();
    assert_eq!(result, JsValue::Boolean(true));
}

#[test]
fn test_greater_equal() {
    let mut ctx = EvalContext::new();
    let expr = binary_expr(BinaryOperator::GreaterThanEqual, num_expr(3), num_expr(5));
    let result = evaluate_expression(&expr, &mut ctx).unwrap();
    assert_eq!(result, JsValue::Boolean(false));
}

// ============================================================================
// Equality tests
// ============================================================================

#[test]
fn test_strict_equality_numbers() {
    let mut ctx = EvalContext::new();
    let expr = binary_expr(BinaryOperator::StrictlyEqual, num_expr(5), num_expr(5));
    let result = evaluate_expression(&expr, &mut ctx).unwrap();
    assert_eq!(result, JsValue::Boolean(true));
}

#[test]
fn test_strict_equality_different_types() {
    let mut ctx = EvalContext::new();
    let expr = binary_expr(BinaryOperator::StrictlyEqual, num_expr(5), str_expr("5"));
    let result = evaluate_expression(&expr, &mut ctx).unwrap();
    assert_eq!(result, JsValue::Boolean(false));
}

#[test]
fn test_strict_inequality() {
    let mut ctx = EvalContext::new();
    let expr = binary_expr(BinaryOperator::StrictlyUnequal, num_expr(5), num_expr(3));
    let result = evaluate_expression(&expr, &mut ctx).unwrap();
    assert_eq!(result, JsValue::Boolean(true));
}

#[test]
fn test_loose_equality_null_undefined() {
    let mut ctx = EvalContext::new();
    // null == null should be true
    let null_expr1 = null_expr();
    let null_expr2 = null_expr();
    let expr = binary_expr(BinaryOperator::LooselyEqual, null_expr1, null_expr2);
    let result = evaluate_expression(&expr, &mut ctx).unwrap();
    assert_eq!(result, JsValue::Boolean(true));
}

// ============================================================================
// Unary operator tests
// ============================================================================

#[test]
fn test_unary_minus() {
    let mut ctx = EvalContext::new();
    let expr = unary_expr(UnaryOperator::Minus, num_expr(5));
    let result = evaluate_expression(&expr, &mut ctx).unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(-5)));
}

#[test]
fn test_unary_plus() {
    let mut ctx = EvalContext::new();
    let expr = unary_expr(UnaryOperator::Plus, str_expr("42"));
    let result = evaluate_expression(&expr, &mut ctx).unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(42)));
}

#[test]
fn test_logical_not_true() {
    let mut ctx = EvalContext::new();
    let expr = unary_expr(UnaryOperator::LogicalNot, bool_expr(true));
    let result = evaluate_expression(&expr, &mut ctx).unwrap();
    assert_eq!(result, JsValue::Boolean(false));
}

#[test]
fn test_logical_not_false() {
    let mut ctx = EvalContext::new();
    let expr = unary_expr(UnaryOperator::LogicalNot, bool_expr(false));
    let result = evaluate_expression(&expr, &mut ctx).unwrap();
    assert_eq!(result, JsValue::Boolean(true));
}

#[test]
fn test_typeof_number() {
    let mut ctx = EvalContext::new();
    let expr = unary_expr(UnaryOperator::TypeOf, num_expr(42));
    let result = evaluate_expression(&expr, &mut ctx).unwrap();
    assert_eq!(result, JsValue::String("number".to_string()));
}

#[test]
fn test_typeof_string() {
    let mut ctx = EvalContext::new();
    let expr = unary_expr(UnaryOperator::TypeOf, str_expr("hello"));
    let result = evaluate_expression(&expr, &mut ctx).unwrap();
    assert_eq!(result, JsValue::String("string".to_string()));
}

#[test]
fn test_typeof_boolean() {
    let mut ctx = EvalContext::new();
    let expr = unary_expr(UnaryOperator::TypeOf, bool_expr(true));
    let result = evaluate_expression(&expr, &mut ctx).unwrap();
    assert_eq!(result, JsValue::String("boolean".to_string()));
}

#[test]
fn test_void() {
    let mut ctx = EvalContext::new();
    let expr = unary_expr(UnaryOperator::Void, num_expr(42));
    let result = evaluate_expression(&expr, &mut ctx).unwrap();
    assert_eq!(result, JsValue::Undefined);
}

#[test]
fn test_bitwise_not() {
    let mut ctx = EvalContext::new();
    let expr = unary_expr(UnaryOperator::BitwiseNot, num_expr(5));
    let result = evaluate_expression(&expr, &mut ctx).unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(-6)));
}

// ============================================================================
// Logical operator tests
// ============================================================================

#[test]
fn test_logical_and_both_true() {
    let mut ctx = EvalContext::new();
    let expr = logical_expr(LogicalOperator::And, bool_expr(true), num_expr(42));
    let result = evaluate_expression(&expr, &mut ctx).unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(42)));
}

#[test]
fn test_logical_and_first_false() {
    let mut ctx = EvalContext::new();
    let expr = logical_expr(LogicalOperator::And, bool_expr(false), num_expr(42));
    let result = evaluate_expression(&expr, &mut ctx).unwrap();
    assert_eq!(result, JsValue::Boolean(false));
}

#[test]
fn test_logical_or_first_true() {
    let mut ctx = EvalContext::new();
    let expr = logical_expr(LogicalOperator::Or, num_expr(1), num_expr(2));
    let result = evaluate_expression(&expr, &mut ctx).unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(1)));
}

#[test]
fn test_logical_or_first_false() {
    let mut ctx = EvalContext::new();
    let expr = logical_expr(LogicalOperator::Or, bool_expr(false), num_expr(42));
    let result = evaluate_expression(&expr, &mut ctx).unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(42)));
}

// ============================================================================
// Conditional (ternary) operator tests
// ============================================================================

#[test]
fn test_ternary_true() {
    let mut ctx = EvalContext::new();
    let expr = cond_expr(bool_expr(true), num_expr(1), num_expr(2));
    let result = evaluate_expression(&expr, &mut ctx).unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(1)));
}

#[test]
fn test_ternary_false() {
    let mut ctx = EvalContext::new();
    let expr = cond_expr(bool_expr(false), num_expr(1), num_expr(2));
    let result = evaluate_expression(&expr, &mut ctx).unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(2)));
}

// ============================================================================
// Bitwise operator tests
// ============================================================================

#[test]
fn test_bitwise_and() {
    let mut ctx = EvalContext::new();
    let expr = binary_expr(BinaryOperator::BitwiseAnd, num_expr(5), num_expr(3));
    let result = evaluate_expression(&expr, &mut ctx).unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(1))); // 101 & 011 = 001
}

#[test]
fn test_bitwise_or() {
    let mut ctx = EvalContext::new();
    let expr = binary_expr(BinaryOperator::BitwiseOr, num_expr(5), num_expr(3));
    let result = evaluate_expression(&expr, &mut ctx).unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(7))); // 101 | 011 = 111
}

#[test]
fn test_bitwise_xor() {
    let mut ctx = EvalContext::new();
    let expr = binary_expr(BinaryOperator::BitwiseXor, num_expr(5), num_expr(3));
    let result = evaluate_expression(&expr, &mut ctx).unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(6))); // 101 ^ 011 = 110
}

#[test]
fn test_left_shift() {
    let mut ctx = EvalContext::new();
    let expr = binary_expr(BinaryOperator::BitwiseLeftShift, num_expr(1), num_expr(4));
    let result = evaluate_expression(&expr, &mut ctx).unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(16)));
}

#[test]
fn test_right_shift() {
    let mut ctx = EvalContext::new();
    let expr = binary_expr(BinaryOperator::BitwiseRightShift, num_expr(16), num_expr(2));
    let result = evaluate_expression(&expr, &mut ctx).unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(4)));
}

// ============================================================================
// Type conversion tests (to_boolean)
// ============================================================================

#[test]
fn test_to_boolean_true() {
    assert!(to_boolean(&JsValue::Boolean(true)));
}

#[test]
fn test_to_boolean_false() {
    assert!(!to_boolean(&JsValue::Boolean(false)));
}

#[test]
fn test_to_boolean_zero() {
    assert!(!to_boolean(&JsValue::Number(JsNumberType::Integer(0))));
}

#[test]
fn test_to_boolean_nonzero() {
    assert!(to_boolean(&JsValue::Number(JsNumberType::Integer(42))));
}

#[test]
fn test_to_boolean_empty_string() {
    assert!(!to_boolean(&JsValue::String(String::new())));
}

#[test]
fn test_to_boolean_nonempty_string() {
    assert!(to_boolean(&JsValue::String("hello".to_string())));
}

#[test]
fn test_to_boolean_null() {
    assert!(!to_boolean(&JsValue::Null));
}

#[test]
fn test_to_boolean_undefined() {
    assert!(!to_boolean(&JsValue::Undefined));
}

#[test]
fn test_to_boolean_nan() {
    assert!(!to_boolean(&JsValue::Number(JsNumberType::NaN)));
}

// ============================================================================
// Variable declaration and lookup tests
// ============================================================================

use just::parser::ast::{
    ExpressionPatternType, IdentifierData, PatternType, PatternOrExpression,
    AssignmentOperator, VariableDeclarationData, VariableDeclarationKind,
    VariableDeclaratorData, StatementType, DeclarationType,
};
use just::runner::eval::statement::execute_statement;

/// Helper to create an identifier expression.
fn id_expr(name: &str) -> ExpressionType {
    ExpressionType::ExpressionWhichCanBePattern(ExpressionPatternType::Identifier(
        IdentifierData {
            name: name.to_string(),
            meta: test_meta(),
            is_binding_identifier: true,
        },
    ))
}

/// Helper to create an identifier pattern.
fn id_pattern(name: &str) -> PatternType {
    PatternType::PatternWhichCanBeExpression(ExpressionPatternType::Identifier(
        IdentifierData {
            name: name.to_string(),
            meta: test_meta(),
            is_binding_identifier: true,
        },
    ))
}

/// Helper to create an update expression.
fn update_expr(op: UpdateOperator, arg: ExpressionType, prefix: bool) -> ExpressionType {
    ExpressionType::UpdateExpression {
        meta: test_meta(),
        operator: op,
        argument: Box::new(arg),
        prefix,
    }
}

/// Helper to create a variable declaration statement.
fn var_decl_stmt(
    kind: VariableDeclarationKind,
    name: &str,
    init: Option<ExpressionType>,
) -> StatementType {
    StatementType::DeclarationStatement(DeclarationType::VariableDeclaration(
        VariableDeclarationData {
            meta: test_meta(),
            kind,
            declarations: vec![VariableDeclaratorData {
                meta: test_meta(),
                id: Box::new(id_pattern(name)),
                init: init.map(Box::new),
            }],
        },
    ))
}

/// Helper to create an assignment expression.
fn assign_expr(name: &str, value: ExpressionType) -> ExpressionType {
    ExpressionType::AssignmentExpression {
        meta: test_meta(),
        operator: AssignmentOperator::Equals,
        left: PatternOrExpression::Expression(Box::new(id_expr(name))),
        right: Box::new(value),
    }
}

#[test]
fn test_var_declaration_and_lookup() {
    let mut ctx = EvalContext::new();

    // var x = 42;
    let stmt = var_decl_stmt(VariableDeclarationKind::Var, "x", Some(num_expr(42)));
    execute_statement(&stmt, &mut ctx).unwrap();

    // x should be 42
    let lookup = id_expr("x");
    let result = evaluate_expression(&lookup, &mut ctx).unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(42)));
}

#[test]
fn test_let_declaration_and_lookup() {
    let mut ctx = EvalContext::new();

    // let y = "hello";
    let stmt = var_decl_stmt(VariableDeclarationKind::Let, "y", Some(str_expr("hello")));
    execute_statement(&stmt, &mut ctx).unwrap();

    // y should be "hello"
    let lookup = id_expr("y");
    let result = evaluate_expression(&lookup, &mut ctx).unwrap();
    assert_eq!(result, JsValue::String("hello".to_string()));
}

#[test]
fn test_const_declaration_and_lookup() {
    let mut ctx = EvalContext::new();

    // const z = true;
    let stmt = var_decl_stmt(VariableDeclarationKind::Const, "z", Some(bool_expr(true)));
    execute_statement(&stmt, &mut ctx).unwrap();

    // z should be true
    let lookup = id_expr("z");
    let result = evaluate_expression(&lookup, &mut ctx).unwrap();
    assert_eq!(result, JsValue::Boolean(true));
}

#[test]
fn test_var_declaration_no_init() {
    let mut ctx = EvalContext::new();

    // var x;
    let stmt = var_decl_stmt(VariableDeclarationKind::Var, "x", None);
    execute_statement(&stmt, &mut ctx).unwrap();

    // x should be undefined
    let lookup = id_expr("x");
    let result = evaluate_expression(&lookup, &mut ctx).unwrap();
    assert_eq!(result, JsValue::Undefined);
}

#[test]
fn test_assignment_expression() {
    let mut ctx = EvalContext::new();

    // var x = 10;
    let stmt = var_decl_stmt(VariableDeclarationKind::Var, "x", Some(num_expr(10)));
    execute_statement(&stmt, &mut ctx).unwrap();

    // x = 20
    let assign = assign_expr("x", num_expr(20));
    let result = evaluate_expression(&assign, &mut ctx).unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(20)));

    // x should now be 20
    let lookup = id_expr("x");
    let result = evaluate_expression(&lookup, &mut ctx).unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(20)));
}

#[test]
fn test_compound_assignment_add() {
    let mut ctx = EvalContext::new();

    // var x = 10;
    let stmt = var_decl_stmt(VariableDeclarationKind::Var, "x", Some(num_expr(10)));
    execute_statement(&stmt, &mut ctx).unwrap();

    // x += 5
    let assign = ExpressionType::AssignmentExpression {
        meta: test_meta(),
        operator: AssignmentOperator::AddEquals,
        left: PatternOrExpression::Expression(Box::new(id_expr("x"))),
        right: Box::new(num_expr(5)),
    };
    let result = evaluate_expression(&assign, &mut ctx).unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(15)));
}

#[test]
fn test_undefined_variable_error() {
    let mut ctx = EvalContext::new();

    // Try to access undefined variable
    let lookup = id_expr("undefined_var");
    let result = evaluate_expression(&lookup, &mut ctx);
    assert!(result.is_err());
}

// ============================================================================
// Member expression tests
// ============================================================================

use just::parser::ast::{MemberExpressionType, ExpressionOrSuper};

/// Helper to create a simple member expression (dot notation).
fn member_expr(obj: ExpressionType, prop_name: &str) -> ExpressionType {
    ExpressionType::MemberExpression(MemberExpressionType::SimpleMemberExpression {
        meta: test_meta(),
        object: ExpressionOrSuper::Expression(Box::new(obj)),
        property: IdentifierData {
            name: prop_name.to_string(),
            meta: test_meta(),
            is_binding_identifier: false,
        },
    })
}

/// Helper to create a computed member expression (bracket notation).
fn computed_member_expr(obj: ExpressionType, prop: ExpressionType) -> ExpressionType {
    ExpressionType::MemberExpression(MemberExpressionType::ComputedMemberExpression {
        meta: test_meta(),
        object: ExpressionOrSuper::Expression(Box::new(obj)),
        property: Box::new(prop),
    })
}

#[test]
fn test_string_length_property() {
    let mut ctx = EvalContext::new();

    // "hello".length
    let expr = member_expr(str_expr("hello"), "length");
    let result = evaluate_expression(&expr, &mut ctx).unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(5)));
}

#[test]
fn test_string_index_access() {
    let mut ctx = EvalContext::new();

    // "hello"[1]
    let expr = computed_member_expr(str_expr("hello"), num_expr(1));
    let result = evaluate_expression(&expr, &mut ctx).unwrap();
    assert_eq!(result, JsValue::String("e".to_string()));
}

#[test]
fn test_string_index_out_of_bounds() {
    let mut ctx = EvalContext::new();

    // "hello"[10]
    let expr = computed_member_expr(str_expr("hello"), num_expr(10));
    let result = evaluate_expression(&expr, &mut ctx).unwrap();
    assert_eq!(result, JsValue::Undefined);
}

#[test]
fn test_member_access_on_undefined() {
    let mut ctx = EvalContext::new();

    // Store undefined in a variable and try to access property
    let stmt = var_decl_stmt(VariableDeclarationKind::Var, "x", None);
    execute_statement(&stmt, &mut ctx).unwrap();

    // x.foo should throw TypeError
    let expr = member_expr(id_expr("x"), "foo");
    let result = evaluate_expression(&expr, &mut ctx);
    assert!(result.is_err());
}

#[test]
fn test_member_access_on_null() {
    let mut ctx = EvalContext::new();

    // null.foo should throw TypeError
    let expr = member_expr(null_expr(), "foo");
    let result = evaluate_expression(&expr, &mut ctx);
    assert!(result.is_err());
}

// ============================================================================
// Array and object literal tests
// ============================================================================

use just::parser::ast::{PropertyData, PropertyKind};

/// Helper to create an array expression.
fn array_expr(elements: Vec<ExpressionType>) -> ExpressionType {
    use just::parser::ast::ExpressionOrSpreadElement;
    ExpressionType::ArrayExpression {
        meta: test_meta(),
        elements: elements
            .into_iter()
            .map(|e| Some(ExpressionOrSpreadElement::Expression(Box::new(e))))
            .collect(),
    }
}

/// Helper to create an object expression with simple properties.
fn object_expr(properties: Vec<(&str, ExpressionType)>) -> ExpressionType {
    ExpressionType::ObjectExpression {
        meta: test_meta(),
        properties: properties
            .into_iter()
            .map(|(name, value)| PropertyData {
                meta: test_meta(),
                key: Box::new(id_expr(name)),
                value: Box::new(value),
                kind: PropertyKind::Init,
                method: false,
                shorthand: false,
                computed: false,
            })
            .collect(),
    }
}

#[test]
fn test_empty_array() {
    let mut ctx = EvalContext::new();

    // []
    let expr = ExpressionType::ArrayExpression {
        meta: test_meta(),
        elements: vec![],
    };
    let result = evaluate_expression(&expr, &mut ctx).unwrap();

    // Should be an object
    assert!(matches!(result, JsValue::Object(_)));

    // Access length
    let length_expr = member_expr(
        ExpressionType::ArrayExpression {
            meta: test_meta(),
            elements: vec![],
        },
        "length",
    );
    let length = evaluate_expression(&length_expr, &mut ctx).unwrap();
    assert_eq!(length, JsValue::Number(JsNumberType::Integer(0)));
}

#[test]
fn test_array_with_elements() {
    let mut ctx = EvalContext::new();

    // [1, 2, 3]
    let arr = array_expr(vec![num_expr(1), num_expr(2), num_expr(3)]);

    // Store it
    let stmt = var_decl_stmt(VariableDeclarationKind::Var, "arr", Some(arr));
    execute_statement(&stmt, &mut ctx).unwrap();

    // Access arr.length
    let length_expr = member_expr(id_expr("arr"), "length");
    let length = evaluate_expression(&length_expr, &mut ctx).unwrap();
    assert_eq!(length, JsValue::Number(JsNumberType::Integer(3)));

    // Access arr[0]
    let elem0 = computed_member_expr(id_expr("arr"), num_expr(0));
    let val = evaluate_expression(&elem0, &mut ctx).unwrap();
    assert_eq!(val, JsValue::Number(JsNumberType::Integer(1)));

    // Access arr[2]
    let elem2 = computed_member_expr(id_expr("arr"), num_expr(2));
    let val = evaluate_expression(&elem2, &mut ctx).unwrap();
    assert_eq!(val, JsValue::Number(JsNumberType::Integer(3)));
}

#[test]
fn test_empty_object() {
    let mut ctx = EvalContext::new();

    // {}
    let expr = ExpressionType::ObjectExpression {
        meta: test_meta(),
        properties: vec![],
    };
    let result = evaluate_expression(&expr, &mut ctx).unwrap();

    // Should be an object
    assert!(matches!(result, JsValue::Object(_)));
}

#[test]
fn test_object_with_properties() {
    let mut ctx = EvalContext::new();

    // { x: 1, y: 2 }
    let obj = object_expr(vec![("x", num_expr(1)), ("y", num_expr(2))]);

    // Store it
    let stmt = var_decl_stmt(VariableDeclarationKind::Var, "obj", Some(obj));
    execute_statement(&stmt, &mut ctx).unwrap();

    // Access obj.x
    let x_expr = member_expr(id_expr("obj"), "x");
    let x = evaluate_expression(&x_expr, &mut ctx).unwrap();
    assert_eq!(x, JsValue::Number(JsNumberType::Integer(1)));

    // Access obj.y
    let y_expr = member_expr(id_expr("obj"), "y");
    let y = evaluate_expression(&y_expr, &mut ctx).unwrap();
    assert_eq!(y, JsValue::Number(JsNumberType::Integer(2)));

    // Access obj["x"] (computed)
    let x_computed = computed_member_expr(id_expr("obj"), str_expr("x"));
    let x2 = evaluate_expression(&x_computed, &mut ctx).unwrap();
    assert_eq!(x2, JsValue::Number(JsNumberType::Integer(1)));
}

#[test]
fn test_object_undefined_property() {
    let mut ctx = EvalContext::new();

    // { x: 1 }
    let obj = object_expr(vec![("x", num_expr(1))]);

    // Store it
    let stmt = var_decl_stmt(VariableDeclarationKind::Var, "obj", Some(obj));
    execute_statement(&stmt, &mut ctx).unwrap();

    // Access obj.z (undefined)
    let z_expr = member_expr(id_expr("obj"), "z");
    let z = evaluate_expression(&z_expr, &mut ctx).unwrap();
    assert_eq!(z, JsValue::Undefined);
}

// ============================================================================
// Update expression tests (++, --)
// ============================================================================

#[test]
fn test_prefix_increment() {
    let mut ctx = EvalContext::new();

    // var x = 5
    let stmt = var_decl_stmt(VariableDeclarationKind::Var, "x", Some(num_expr(5)));
    execute_statement(&stmt, &mut ctx).unwrap();

    // ++x should return 6
    let inc_expr = update_expr(UpdateOperator::PlusPlus, id_expr("x"), true);
    let result = evaluate_expression(&inc_expr, &mut ctx).unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(6)));

    // x should now be 6
    let x_val = evaluate_expression(&id_expr("x"), &mut ctx).unwrap();
    assert_eq!(x_val, JsValue::Number(JsNumberType::Integer(6)));
}

#[test]
fn test_postfix_increment() {
    let mut ctx = EvalContext::new();

    // var x = 5
    let stmt = var_decl_stmt(VariableDeclarationKind::Var, "x", Some(num_expr(5)));
    execute_statement(&stmt, &mut ctx).unwrap();

    // x++ should return 5 (old value)
    let inc_expr = update_expr(UpdateOperator::PlusPlus, id_expr("x"), false);
    let result = evaluate_expression(&inc_expr, &mut ctx).unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(5)));

    // x should now be 6
    let x_val = evaluate_expression(&id_expr("x"), &mut ctx).unwrap();
    assert_eq!(x_val, JsValue::Number(JsNumberType::Integer(6)));
}

#[test]
fn test_prefix_decrement() {
    let mut ctx = EvalContext::new();

    // var x = 5
    let stmt = var_decl_stmt(VariableDeclarationKind::Var, "x", Some(num_expr(5)));
    execute_statement(&stmt, &mut ctx).unwrap();

    // --x should return 4
    let dec_expr = update_expr(UpdateOperator::MinusMinus, id_expr("x"), true);
    let result = evaluate_expression(&dec_expr, &mut ctx).unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(4)));

    // x should now be 4
    let x_val = evaluate_expression(&id_expr("x"), &mut ctx).unwrap();
    assert_eq!(x_val, JsValue::Number(JsNumberType::Integer(4)));
}

#[test]
fn test_postfix_decrement() {
    let mut ctx = EvalContext::new();

    // var x = 5
    let stmt = var_decl_stmt(VariableDeclarationKind::Var, "x", Some(num_expr(5)));
    execute_statement(&stmt, &mut ctx).unwrap();

    // x-- should return 5 (old value)
    let dec_expr = update_expr(UpdateOperator::MinusMinus, id_expr("x"), false);
    let result = evaluate_expression(&dec_expr, &mut ctx).unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(5)));

    // x should now be 4
    let x_val = evaluate_expression(&id_expr("x"), &mut ctx).unwrap();
    assert_eq!(x_val, JsValue::Number(JsNumberType::Integer(4)));
}

#[test]
fn test_update_in_loop() {
    let mut ctx = EvalContext::new();

    // var i = 0
    let var_stmt = var_decl_stmt(VariableDeclarationKind::Var, "i", Some(num_expr(0)));
    execute_statement(&var_stmt, &mut ctx).unwrap();

    // Simulate: while (i < 3) i++
    // Just test three increments
    for _ in 0..3 {
        let inc_expr = update_expr(UpdateOperator::PlusPlus, id_expr("i"), false);
        evaluate_expression(&inc_expr, &mut ctx).unwrap();
    }

    // i should be 3
    let i_val = evaluate_expression(&id_expr("i"), &mut ctx).unwrap();
    assert_eq!(i_val, JsValue::Number(JsNumberType::Integer(3)));
}
