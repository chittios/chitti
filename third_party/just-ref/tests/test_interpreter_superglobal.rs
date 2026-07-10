extern crate just;

use just::parser::JsParser;
use just::runner::plugin::resolver::PluginResolver;
use just::runner::plugin::types::EvalContext;
use just::runner::plugin::registry::BuiltInRegistry;
use just::runner::ds::error::JErrorType;
use just::runner::ds::value::{JsValue, JsNumberType};
use just::runner::eval::statement::execute_statement;
use std::cell::RefCell;
use std::rc::Rc;

/// Helper to run code with the interpreter
fn run_interpreter(code: &str, ctx: &mut EvalContext) -> Result<JsValue, JErrorType> {
    let ast = JsParser::parse_to_ast_from_str(code)
        .map_err(|e| JErrorType::SyntaxError(format!("Parse error: {:?}", e)))?;
    let mut last_value = JsValue::Undefined;
    
    for stmt in &ast.body {
        let completion = execute_statement(stmt, ctx)?;
        if let Some(val) = completion.value {
            last_value = val;
        }
    }
    
    Ok(last_value)
}

/// Helper to get a variable from context
fn get_var(ctx: &mut EvalContext, name: &str) -> JsValue {
    ctx.get_binding(name).unwrap_or(JsValue::Undefined)
}

// ── Built-in method resolution tests ─────────────────────────────────

#[test]
fn test_interpreter_math_abs() {
    let mut ctx = EvalContext::new();
    ctx.install_core_builtins(BuiltInRegistry::with_core());
    
    let result = run_interpreter("Math.abs(-42)", &mut ctx).unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(42)));
}

#[test]
fn test_interpreter_math_max() {
    let mut ctx = EvalContext::new();
    ctx.install_core_builtins(BuiltInRegistry::with_core());
    
    let result = run_interpreter("Math.max(5, 10, 3)", &mut ctx).unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(10)));
}

#[test]
fn test_interpreter_math_floor() {
    let mut ctx = EvalContext::new();
    ctx.install_core_builtins(BuiltInRegistry::with_core());
    
    let result = run_interpreter("Math.floor(4.9)", &mut ctx).unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(4)));
}

#[test]
fn test_interpreter_console_log() {
    let mut ctx = EvalContext::new();
    ctx.install_core_builtins(BuiltInRegistry::with_core());
    
    // console.log returns undefined
    let result = run_interpreter("console.log('test message')", &mut ctx).unwrap();
    assert_eq!(result, JsValue::Undefined);
}

// Note: String primitive methods require additional infrastructure in the interpreter
// that's beyond super-global resolution. The VM handles these via type-based dispatch.

// ── Constructor tests ────────────────────────────────────────────────

#[test]
fn test_interpreter_string_constructor() {
    let mut ctx = EvalContext::new();
    ctx.install_core_builtins(BuiltInRegistry::with_core());
    
    let code = "var s = new String('hello world'); s";
    let result = run_interpreter(code, &mut ctx).unwrap();
    assert_eq!(result, JsValue::String("hello world".to_string()));
}

#[test]
fn test_interpreter_number_constructor() {
    let mut ctx = EvalContext::new();
    ctx.install_core_builtins(BuiltInRegistry::with_core());
    
    let code = "var n = new Number(42); n";
    let result = run_interpreter(code, &mut ctx).unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(42)));
}

#[test]
fn test_interpreter_number_constructor_float() {
    let mut ctx = EvalContext::new();
    ctx.install_core_builtins(BuiltInRegistry::with_core());
    
    let code = "var n = new Number(3.14); n";
    let result = run_interpreter(code, &mut ctx).unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Float(3.14)));
}

#[test]
fn test_interpreter_string_constructor_empty() {
    let mut ctx = EvalContext::new();
    ctx.install_core_builtins(BuiltInRegistry::with_core());
    
    let code = "var s = new String(); s";
    let result = run_interpreter(code, &mut ctx).unwrap();
    assert_eq!(result, JsValue::String("".to_string()));
}

#[test]
fn test_interpreter_multiple_constructors() {
    let mut ctx = EvalContext::new();
    ctx.install_core_builtins(BuiltInRegistry::with_core());
    
    let code = r#"
        var s = new String("test");
        var n = new Number(100);
        var result = s;
        result
    "#;
    let result = run_interpreter(code, &mut ctx).unwrap();
    assert_eq!(result, JsValue::String("test".to_string()));
}

#[test]
fn test_interpreter_builtin_in_expression() {
    let mut ctx = EvalContext::new();
    ctx.install_core_builtins(BuiltInRegistry::with_core());
    
    let code = "var x = Math.abs(-10) + Math.abs(-5);";
    run_interpreter(code, &mut ctx).unwrap();
    
    let x = get_var(&mut ctx, "x");
    assert_eq!(x, JsValue::Number(JsNumberType::Integer(15)));
}

#[test]
fn test_interpreter_local_shadows_builtin() {
    let mut ctx = EvalContext::new();
    ctx.install_core_builtins(BuiltInRegistry::with_core());
    
    let code = r#"
        var Math = 99;
        var x = Math;
        x
    "#;
    
    let result = run_interpreter(code, &mut ctx).unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(99)));
}

// ── Custom plugin tests ──────────────────────────────────────────────

/// A test plugin that provides a "TestLib" object with methods
struct TestLibPlugin {
    console_output: Rc<RefCell<Vec<String>>>,
}

impl TestLibPlugin {
    fn new() -> Self {
        Self {
            console_output: Rc::new(RefCell::new(Vec::new())),
        }
    }
    
    fn get_output(&self) -> Vec<String> {
        self.console_output.borrow().clone()
    }
}

impl PluginResolver for TestLibPlugin {
    fn has_binding(&self, name: &str) -> bool {
        name == "TestLib" || name == "CustomObject"
    }
    
    fn resolve(&self, name: &str, _ctx: &mut EvalContext) -> Result<JsValue, JErrorType> {
        if name == "TestLib" || name == "CustomObject" {
            // Return a sentinel value to indicate the object exists
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
        if object_name != "TestLib" {
            return None;
        }
        
        match method_name {
            "double" => {
                let n = match args.first() {
                    Some(JsValue::Number(JsNumberType::Integer(n))) => *n,
                    Some(JsValue::Number(JsNumberType::Float(f))) => *f as i64,
                    _ => 0,
                };
                Some(Ok(JsValue::Number(JsNumberType::Integer(n * 2))))
            }
            "greet" => {
                let name = match args.first() {
                    Some(JsValue::String(s)) => s.clone(),
                    _ => "World".to_string(),
                };
                let message = format!("Hello, {}!", name);
                
                // Store the message
                self.console_output.borrow_mut().push(message.clone());
                
                // Also print to stdout for visibility
                println!("{}", message);
                
                Some(Ok(JsValue::String(message)))
            }
            "add" => {
                let a = match args.get(0) {
                    Some(JsValue::Number(JsNumberType::Integer(n))) => *n,
                    _ => 0,
                };
                let b = match args.get(1) {
                    Some(JsValue::Number(JsNumberType::Integer(n))) => *n,
                    _ => 0,
                };
                Some(Ok(JsValue::Number(JsNumberType::Integer(a + b))))
            }
            _ => None,
        }
    }
    
    fn call_constructor(
        &self,
        object_name: &str,
        _ctx: &mut EvalContext,
        args: Vec<JsValue>,
    ) -> Option<Result<JsValue, JErrorType>> {
        if object_name == "CustomObject" {
            // Create a custom object with a value property
            let value = match args.first() {
                Some(JsValue::Number(JsNumberType::Integer(n))) => *n,
                Some(JsValue::Number(JsNumberType::Float(f))) => *f as i64,
                _ => 0,
            };
            
            // For simplicity, return a number representing the constructed object
            // In a real implementation, this would return a proper object
            Some(Ok(JsValue::Number(JsNumberType::Integer(value * 10))))
        } else {
            None
        }
    }
    
    fn name(&self) -> &str {
        "test_lib_plugin"
    }
}

#[test]
fn test_custom_plugin_variable_access() {
    let mut ctx = EvalContext::new();
    ctx.add_resolver(Box::new(TestLibPlugin::new()));
    
    let code = "var x = TestLib;";
    run_interpreter(code, &mut ctx).unwrap();
    
    let x = get_var(&mut ctx, "x");
    assert_eq!(x, JsValue::Number(JsNumberType::Integer(42)));
}

#[test]
fn test_custom_plugin_method_double() {
    let mut ctx = EvalContext::new();
    ctx.add_resolver(Box::new(TestLibPlugin::new()));
    
    let code = "var result = TestLib.double(21);";
    run_interpreter(code, &mut ctx).unwrap();
    
    let result = get_var(&mut ctx, "result");
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(42)));
}

#[test]
fn test_custom_plugin_method_greet_prints_to_console() {
    let plugin = TestLibPlugin::new();
    let output_ref = plugin.console_output.clone();
    
    let mut ctx = EvalContext::new();
    ctx.add_resolver(Box::new(plugin));
    
    let code = r#"
        var greeting = TestLib.greet("Alice");
        greeting
    "#;
    
    let result = run_interpreter(code, &mut ctx).unwrap();
    
    // Check the return value
    assert_eq!(result, JsValue::String("Hello, Alice!".to_string()));
    
    // Check that the message was captured
    let output = output_ref.borrow();
    assert_eq!(output.len(), 1);
    assert_eq!(output[0], "Hello, Alice!");
}

#[test]
fn test_custom_plugin_method_add() {
    let mut ctx = EvalContext::new();
    ctx.add_resolver(Box::new(TestLibPlugin::new()));
    
    let code = "var sum = TestLib.add(10, 32);";
    run_interpreter(code, &mut ctx).unwrap();
    
    let sum = get_var(&mut ctx, "sum");
    assert_eq!(sum, JsValue::Number(JsNumberType::Integer(42)));
}

#[test]
fn test_custom_plugin_in_expression() {
    let mut ctx = EvalContext::new();
    ctx.add_resolver(Box::new(TestLibPlugin::new()));
    
    let code = "var result = TestLib.double(5) + TestLib.add(3, 7);";
    run_interpreter(code, &mut ctx).unwrap();
    
    let result = get_var(&mut ctx, "result");
    // double(5) = 10, add(3, 7) = 10, total = 20
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(20)));
}

#[test]
fn test_custom_plugin_with_core_builtins() {
    let mut ctx = EvalContext::new();
    ctx.install_core_builtins(BuiltInRegistry::with_core());
    ctx.add_resolver(Box::new(TestLibPlugin::new()));
    
    let code = r#"
        var a = Math.abs(-5);
        var b = TestLib.double(10);
        var result = a + b;
        result
    "#;
    
    let result = run_interpreter(code, &mut ctx).unwrap();
    // abs(-5) = 5, double(10) = 20, total = 25
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(25)));
}

#[test]
fn test_custom_plugin_multiple_calls() {
    let plugin = TestLibPlugin::new();
    let output_ref = plugin.console_output.clone();
    
    let mut ctx = EvalContext::new();
    ctx.add_resolver(Box::new(plugin));
    
    let code = r#"
        TestLib.greet("Alice");
        TestLib.greet("Bob");
        TestLib.greet("Charlie");
    "#;
    
    run_interpreter(code, &mut ctx).unwrap();
    
    let output = output_ref.borrow();
    assert_eq!(output.len(), 3);
    assert_eq!(output[0], "Hello, Alice!");
    assert_eq!(output[1], "Hello, Bob!");
    assert_eq!(output[2], "Hello, Charlie!");
}

#[test]
fn test_custom_plugin_undefined_method() {
    let mut ctx = EvalContext::new();
    ctx.add_resolver(Box::new(TestLibPlugin::new()));
    
    // This should fall through to normal property lookup and return undefined
    // since TestLib doesn't have a "nonexistent" method
    let code = "var x = TestLib.nonexistent;";
    let result = run_interpreter(code, &mut ctx);
    
    // The call will fail because TestLib.nonexistent is undefined
    // and we try to call it, which should error
    assert!(result.is_ok()); // Getting the property succeeds
}

#[test]
fn test_custom_plugin_constructor() {
    let mut ctx = EvalContext::new();
    ctx.add_resolver(Box::new(TestLibPlugin::new()));
    
    let code = "var obj = new CustomObject(5);";
    run_interpreter(code, &mut ctx).unwrap();
    
    let obj = get_var(&mut ctx, "obj");
    // CustomObject constructor multiplies by 10
    assert_eq!(obj, JsValue::Number(JsNumberType::Integer(50)));
}

#[test]
fn test_custom_plugin_constructor_with_builtin() {
    let mut ctx = EvalContext::new();
    ctx.install_core_builtins(BuiltInRegistry::with_core());
    ctx.add_resolver(Box::new(TestLibPlugin::new()));
    
    let code = r#"
        var s = new String("test");
        var custom = new CustomObject(7);
        custom
    "#;
    
    let result = run_interpreter(code, &mut ctx).unwrap();
    assert_eq!(result, JsValue::Number(JsNumberType::Integer(70)));
}
