extern crate just;

use just::parser::JsParser;
use just::runner::jit;
use just::runner::plugin::registry::BuiltInRegistry;
use just::runner::ds::value::{JsValue, JsNumberType};

fn jit_run_get_var(code: &str, var_name: &str) -> JsValue {
    let ast = JsParser::parse_to_ast_from_str(code).unwrap();
    let (_, mut ctx) = jit::compile_and_run_with_ctx(&ast);
    ctx.get_binding(var_name).unwrap_or(JsValue::Undefined)
}

fn jit_run_get_int(code: &str, var_name: &str) -> i64 {
    match jit_run_get_var(code, var_name) {
        JsValue::Number(JsNumberType::Integer(n)) => n,
        other => panic!("{} was {:?}, expected integer", var_name, other),
    }
}

fn jit_disasm(code: &str) -> String {
    let ast = JsParser::parse_to_ast_from_str(code).unwrap();
    let chunk = jit::compile(&ast);
    chunk.disassemble("test")
}

#[test]
fn test_jit_simple_var() {
    assert_eq!(jit_run_get_int("var x = 42;", "x"), 42);
}

#[test]
fn test_jit_assignment() {
    assert_eq!(jit_run_get_int("var a = 5; var b = 10; a = b;", "a"), 10);
}

#[test]
fn test_jit_arithmetic() {
    assert_eq!(jit_run_get_int("var x = 3 + 4 * 2;", "x"), 11);
}

#[test]
fn test_jit_simple_for() {
    let code = "var sum = 0; for (var i = 0; i < 5; i = i + 1) { sum = sum + i; }";
    eprintln!("{}", jit_disasm(code));
    assert_eq!(jit_run_get_int(code, "sum"), 10);
}

#[test]
fn test_jit_nested_for() {
    let code = r#"
var count = 0;
for (var i = 0; i < 3; i = i + 1) {
    for (var j = 0; j < 4; j = j + 1) {
        count = count + 1;
    }
}
"#;
    eprintln!("{}", jit_disasm(code));
    assert_eq!(jit_run_get_int(code, "count"), 12);
}

#[test]
fn test_jit_fibonacci() {
    // Minimal fibonacci with n=3 and debug trace
    let code = r#"
var a = 0;
var b = 1;
for (var i = 0; i < 3; i = i + 1) {
    var temp = a;
    a = b;
    b = temp + b;
}
"#;
    assert_eq!(jit_run_get_int(code, "a"), 2);
}

#[test]
fn test_jit_while_loop() {
    let code = "var i = 0; var sum = 0; while (i < 5) { sum = sum + i; i = i + 1; }";
    assert_eq!(jit_run_get_int(code, "sum"), 10);
}

#[test]
fn test_jit_if_else() {
    assert_eq!(jit_run_get_int("var x = 0; if (true) { x = 1; } else { x = 2; }", "x"), 1);
    assert_eq!(jit_run_get_int("var x = 0; if (false) { x = 1; } else { x = 2; }", "x"), 2);
}

#[test]
fn test_jit_comparison() {
    assert_eq!(jit_run_get_int("var x = 0; if (3 < 5) { x = 1; }", "x"), 1);
    assert_eq!(jit_run_get_int("var x = 0; if (5 < 3) { x = 1; }", "x"), 0);
}

#[test]
fn test_jit_break() {
    let code = r#"
var count = 0;
for (var i = 0; i < 10; i = i + 1) {
    if (i === 5) { break; }
    count = count + 1;
}
"#;
    assert_eq!(jit_run_get_int(code, "count"), 5);
}

// ============================================================================
// Bitwise operations
// ============================================================================

#[test]
fn test_jit_bitwise_and() {
    assert_eq!(jit_run_get_int("var x = 5 & 3;", "x"), 1);
}

#[test]
fn test_jit_bitwise_or() {
    assert_eq!(jit_run_get_int("var x = 5 | 3;", "x"), 7);
}

#[test]
fn test_jit_bitwise_xor() {
    assert_eq!(jit_run_get_int("var x = 5 ^ 3;", "x"), 6);
}

#[test]
fn test_jit_bitwise_not() {
    assert_eq!(jit_run_get_int("var x = ~5;", "x"), -6);
}

#[test]
fn test_jit_shift_left() {
    assert_eq!(jit_run_get_int("var x = 1 << 4;", "x"), 16);
}

#[test]
fn test_jit_shift_right() {
    assert_eq!(jit_run_get_int("var x = 16 >> 2;", "x"), 4);
}

#[test]
fn test_jit_unsigned_shift_right() {
    assert_eq!(jit_run_get_int("var x = 16 >>> 2;", "x"), 4);
}

// ============================================================================
// Modulo and unary
// ============================================================================

#[test]
fn test_jit_modulo() {
    assert_eq!(jit_run_get_int("var x = 17 % 5;", "x"), 2);
}

#[test]
fn test_jit_negate() {
    assert_eq!(jit_run_get_int("var x = -42;", "x"), -42);
}

#[test]
fn test_jit_unary_plus() {
    assert_eq!(jit_run_get_int("var x = +7;", "x"), 7);
}

#[test]
fn test_jit_not_truthy() {
    assert_eq!(jit_run_get_int("var x = 0; if (!false) { x = 1; }", "x"), 1);
}

#[test]
fn test_jit_not_falsy() {
    assert_eq!(jit_run_get_int("var x = 0; if (!true) { x = 1; }", "x"), 0);
}

// ============================================================================
// Let / const declarations
// ============================================================================

#[test]
fn test_jit_let_declaration() {
    assert_eq!(jit_run_get_int("let x = 99;", "x"), 99);
}

#[test]
fn test_jit_const_declaration() {
    assert_eq!(jit_run_get_int("const x = 77;", "x"), 77);
}

#[test]
fn test_jit_let_reassignment() {
    assert_eq!(jit_run_get_int("let x = 1; x = 50;", "x"), 50);
}

// ============================================================================
// Complex control flow
// ============================================================================

#[test]
fn test_jit_nested_if() {
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
    assert_eq!(jit_run_get_int(code, "result"), 1);
}

#[test]
fn test_jit_continue() {
    let code = r#"
var sum = 0;
for (var i = 0; i < 10; i = i + 1) {
    if (i % 2 === 0) { continue; }
    sum = sum + i;
}
"#;
    assert_eq!(jit_run_get_int(code, "sum"), 25); // 1+3+5+7+9
}

#[test]
fn test_jit_factorial() {
    let code = r#"
var n = 6;
var result = 1;
for (var i = 2; i <= n; i = i + 1) {
    result = result * i;
}
"#;
    assert_eq!(jit_run_get_int(code, "result"), 720);
}

#[test]
fn test_jit_gcd() {
    let code = r#"
var a = 48;
var b = 18;
while (b !== 0) {
    var temp = b;
    b = a % b;
    a = temp;
}
"#;
    assert_eq!(jit_run_get_int(code, "a"), 6);
}

#[test]
fn test_jit_sum_of_squares() {
    let code = r#"
var sum = 0;
for (var i = 1; i <= 10; i = i + 1) {
    sum = sum + i * i;
}
"#;
    assert_eq!(jit_run_get_int(code, "sum"), 385);
}

#[test]
fn test_jit_bitwise_loop() {
    let code = r#"
var val = 255;
for (var i = 0; i < 8; i = i + 1) {
    val = val & (val - 1);
}
"#;
    assert_eq!(jit_run_get_int(code, "val"), 0);
}

// ============================================================================
// Register VM path
// ============================================================================

fn reg_vm_get_int(code: &str, var_name: &str) -> i64 {
    let ast = JsParser::parse_to_ast_from_str(code).unwrap();
    let registry = BuiltInRegistry::with_core();
    let (_, mut ctx) = jit::compile_and_run_reg_with_ctx(&ast, &registry);
    match ctx.get_binding(var_name).unwrap_or(JsValue::Undefined) {
        JsValue::Number(JsNumberType::Integer(n)) => n,
        JsValue::Number(JsNumberType::Float(f)) => f as i64,
        other => panic!("{} was {:?}, expected integer", var_name, other),
    }
}

#[test]
fn test_reg_vm_simple_add() {
    assert_eq!(reg_vm_get_int("var x = 3 + 4;", "x"), 7);
}

#[test]
fn test_reg_vm_for_loop() {
    let code = "var sum = 0; for (var i = 0; i < 5; i = i + 1) { sum = sum + i; }";
    assert_eq!(reg_vm_get_int(code, "sum"), 10);
}

#[test]
fn test_reg_vm_bitwise_and() {
    assert_eq!(reg_vm_get_int("var x = 5 & 3;", "x"), 1);
}

#[test]
fn test_reg_vm_modulo() {
    assert_eq!(reg_vm_get_int("var x = 17 % 5;", "x"), 2);
}

#[test]
fn test_reg_vm_fibonacci() {
    let code = r#"
var a = 0;
var b = 1;
for (var i = 0; i < 10; i = i + 1) {
    var temp = a;
    a = b;
    b = temp + b;
}
"#;
    assert_eq!(reg_vm_get_int(code, "a"), 55);
}

#[test]
fn test_reg_vm_nested_loops() {
    let code = r#"
var count = 0;
for (var i = 0; i < 3; i = i + 1) {
    for (var j = 0; j < 4; j = j + 1) {
        count = count + 1;
    }
}
"#;
    assert_eq!(reg_vm_get_int(code, "count"), 12);
}

#[test]
fn test_reg_vm_if_else() {
    assert_eq!(reg_vm_get_int("var x = 0; if (true) { x = 1; } else { x = 2; }", "x"), 1);
    assert_eq!(reg_vm_get_int("var x = 0; if (false) { x = 1; } else { x = 2; }", "x"), 2);
}

// ── Stack VM function call tests ────────────────────────────

#[test]
fn test_jit_function_call_simple() {
    let code = r#"
function add(a, b) { return a + b; }
var result = add(3, 4);
"#;
    assert_eq!(jit_run_get_int(code, "result"), 7);
}

#[test]
fn test_jit_function_call_no_args() {
    let code = r#"
function five() { return 5; }
var x = five();
"#;
    assert_eq!(jit_run_get_int(code, "x"), 5);
}

#[test]
fn test_jit_function_call_recursive_factorial() {
    let code = r#"
function factorial(n) {
    if (n <= 1) { return 1; }
    return n * factorial(n - 1);
}
var result = factorial(6);
"#;
    assert_eq!(jit_run_get_int(code, "result"), 720);
}

#[test]
fn test_jit_function_call_closure() {
    let code = r#"
function makeAdder(x) {
    function adder(y) { return x + y; }
    return adder;
}
var add5 = makeAdder(5);
var result = add5(3);
"#;
    assert_eq!(jit_run_get_int(code, "result"), 8);
}

#[test]
fn test_jit_function_expression() {
    let code = r#"
var double = function(n) { return n * 2; };
var result = double(7);
"#;
    assert_eq!(jit_run_get_int(code, "result"), 14);
}

#[test]
fn test_jit_function_call_multiple() {
    let code = r#"
function square(n) { return n * n; }
var a = square(3);
var b = square(4);
var result = a + b;
"#;
    assert_eq!(jit_run_get_int(code, "result"), 25);
}

// ── Stack VM CallMethod tests ───────────────────────────────

#[test]
fn test_jit_call_method_math_abs() {
    let code = "var x = Math.abs(-7);";
    assert_eq!(jit_run_get_int(code, "x"), 7);
}

#[test]
fn test_jit_call_method_math_max() {
    let code = "var x = Math.max(3, 9, 1);";
    assert_eq!(jit_run_get_int(code, "x"), 9);
}

#[test]
fn test_jit_call_method_math_floor() {
    let code = "var x = Math.floor(4.9);";
    assert_eq!(jit_run_get_int(code, "x"), 4);
}

#[test]
fn test_jit_call_method_string_index_of() {
    let code = r#"var x = "hello world".indexOf("world");"#;
    assert_eq!(jit_run_get_int(code, "x"), 6);
}

// ── Super-global dynamic resolution tests ───────────────────

use just::runner::plugin::resolver::PluginResolver;
use just::runner::plugin::types::EvalContext;
use just::runner::ds::error::JErrorType;

/// A test plugin resolver that provides a single object "MyLib"
/// with a method "double" that doubles a number.
struct TestPluginResolver;

impl PluginResolver for TestPluginResolver {
    fn has_binding(&self, name: &str) -> bool {
        name == "MyLib"
    }

    fn resolve(&self, name: &str, _ctx: &mut EvalContext) -> Result<JsValue, JErrorType> {
        if name == "MyLib" {
            Ok(JsValue::Number(JsNumberType::Integer(42)))
        } else {
            Err(JErrorType::ReferenceError(format!("{} is not defined", name)))
        }
    }

    fn call_method(
        &self,
        object_name: &str,
        method_name: &str,
        _ctx: &mut EvalContext,
        _this: JsValue,
        args: Vec<JsValue>,
    ) -> Option<Result<JsValue, JErrorType>> {
        if object_name == "MyLib" && method_name == "double" {
            let n = match args.first() {
                Some(JsValue::Number(JsNumberType::Integer(n))) => *n,
                _ => 0,
            };
            Some(Ok(JsValue::Number(JsNumberType::Integer(n * 2))))
        } else {
            None
        }
    }

    fn name(&self) -> &str {
        "test_plugin"
    }
}

#[test]
fn test_super_global_custom_resolver_variable() {
    let ast = JsParser::parse_to_ast_from_str("var x = MyLib;").unwrap();
    let chunk = jit::compile(&ast);
    let mut ctx = EvalContext::new();
    ctx.add_resolver(Box::new(TestPluginResolver));
    let mut vm = jit::vm::Vm::new(&chunk, ctx);
    let _ = vm.run();
    let mut ctx = vm.into_ctx();
    // MyLib resolves to 42 via the custom resolver
    assert_eq!(
        ctx.get_binding("x").unwrap(),
        JsValue::Number(JsNumberType::Integer(42))
    );
}

#[test]
fn test_super_global_local_shadows_builtin() {
    // A local variable named "Math" should shadow the super-global Math
    let code = "var Math = 99; var x = Math;";
    assert_eq!(jit_run_get_int(code, "x"), 99);
}

#[test]
fn test_super_global_core_builtins_via_scope_chain() {
    // Math.abs should resolve through the super-global scope chain
    let code = "var x = Math.abs(-42);";
    assert_eq!(jit_run_get_int(code, "x"), 42);
}

#[test]
fn test_super_global_unresolved_name_errors() {
    let ast = JsParser::parse_to_ast_from_str("var x = NoSuchThing;").unwrap();
    let (result, _) = jit::compile_and_run_with_ctx(&ast);
    assert!(result.is_err(), "Accessing undefined super-global should error");
}

