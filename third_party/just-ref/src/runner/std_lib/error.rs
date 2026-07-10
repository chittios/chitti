//! Error built-in objects.
//!
//! Provides Error, TypeError, ReferenceError, SyntaxError, RangeError constructors.

// ChittiOS-alloc-prelude
#[allow(unused_imports)]
use num_traits::Float;
use alloc::borrow::ToOwned;
use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use crate::runner::ds::error::JErrorType;
use crate::runner::ds::value::JsValue;
use crate::runner::plugin::registry::BuiltInRegistry;
use crate::runner::plugin::types::{BuiltInObject, EvalContext};

/// Register all error types with the registry.
pub fn register(registry: &mut BuiltInRegistry) {
    // Base Error
    let error = BuiltInObject::new("Error")
        .with_constructor(error_constructor)
        .add_method("toString", error_to_string)
        .add_method("message", error_message);
    registry.register_object(error);

    // TypeError
    let type_error = BuiltInObject::new("TypeError")
        .with_prototype("Error")
        .with_constructor(type_error_constructor)
        .add_method("toString", error_to_string);
    registry.register_object(type_error);

    // ReferenceError
    let reference_error = BuiltInObject::new("ReferenceError")
        .with_prototype("Error")
        .with_constructor(reference_error_constructor)
        .add_method("toString", error_to_string);
    registry.register_object(reference_error);

    // SyntaxError
    let syntax_error = BuiltInObject::new("SyntaxError")
        .with_prototype("Error")
        .with_constructor(syntax_error_constructor)
        .add_method("toString", error_to_string);
    registry.register_object(syntax_error);

    // RangeError
    let range_error = BuiltInObject::new("RangeError")
        .with_prototype("Error")
        .with_constructor(range_error_constructor)
        .add_method("toString", error_to_string);
    registry.register_object(range_error);

    // EvalError (deprecated but still in spec)
    let eval_error = BuiltInObject::new("EvalError")
        .with_prototype("Error")
        .with_constructor(eval_error_constructor)
        .add_method("toString", error_to_string);
    registry.register_object(eval_error);

    // URIError
    let uri_error = BuiltInObject::new("URIError")
        .with_prototype("Error")
        .with_constructor(uri_error_constructor)
        .add_method("toString", error_to_string);
    registry.register_object(uri_error);
}

/// Get message from arguments.
fn get_message(args: &[JsValue]) -> String {
    if args.is_empty() {
        String::new()
    } else {
        match &args[0] {
            JsValue::String(s) => s.clone(),
            JsValue::Undefined => String::new(),
            v => v.to_string(),
        }
    }
}

/// Error constructor.
fn error_constructor(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let message = get_message(&args);
    // TODO: Return an actual Error object when object creation is implemented
    // For now, return the message as a string
    Ok(JsValue::String(format!("Error: {}", message)))
}

/// TypeError constructor.
fn type_error_constructor(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let message = get_message(&args);
    Ok(JsValue::String(format!("TypeError: {}", message)))
}

/// ReferenceError constructor.
fn reference_error_constructor(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let message = get_message(&args);
    Ok(JsValue::String(format!("ReferenceError: {}", message)))
}

/// SyntaxError constructor.
fn syntax_error_constructor(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let message = get_message(&args);
    Ok(JsValue::String(format!("SyntaxError: {}", message)))
}

/// RangeError constructor.
fn range_error_constructor(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let message = get_message(&args);
    Ok(JsValue::String(format!("RangeError: {}", message)))
}

/// EvalError constructor.
fn eval_error_constructor(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let message = get_message(&args);
    Ok(JsValue::String(format!("EvalError: {}", message)))
}

/// URIError constructor.
fn uri_error_constructor(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let message = get_message(&args);
    Ok(JsValue::String(format!("URIError: {}", message)))
}

/// Error.prototype.toString
fn error_to_string(
    _ctx: &mut EvalContext,
    this: JsValue,
    _args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    // For now, just return the value itself if it's a string
    match this {
        JsValue::String(s) => Ok(JsValue::String(s)),
        JsValue::Object(_) => {
            // TODO: Get name and message properties from object
            Ok(JsValue::String("Error".to_string()))
        }
        _ => Ok(JsValue::String("Error".to_string())),
    }
}

/// Error.prototype.message getter
fn error_message(
    _ctx: &mut EvalContext,
    this: JsValue,
    _args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    // For now, return empty string (actual implementation needs object properties)
    match this {
        JsValue::String(s) => {
            // Extract message part from "ErrorType: message" format
            if let Some(idx) = s.find(": ") {
                Ok(JsValue::String(s[idx + 2..].to_string()))
            } else {
                Ok(JsValue::String(String::new()))
            }
        }
        _ => Ok(JsValue::String(String::new())),
    }
}
