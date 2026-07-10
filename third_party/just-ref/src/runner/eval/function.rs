//! Function call execution.
//!
//! This module provides function call execution logic for the JavaScript interpreter.

// ChittiOS-alloc-prelude
#[allow(unused_imports)]
use num_traits::Float;
use alloc::borrow::ToOwned;
use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use crate::parser::ast::FunctionData;
use crate::runner::ds::error::JErrorType;
use crate::runner::ds::value::JsValue;
use crate::runner::plugin::types::EvalContext;
use crate::runner::plugin::registry::BuiltInRegistry;

use super::types::ValueResult;

/// Call a function with the given arguments.
pub fn call_function(
    _func: &FunctionData,
    _this_value: JsValue,
    _args: Vec<JsValue>,
    _ctx: &mut EvalContext,
) -> ValueResult {
    // TODO: Implement proper function calls
    // 1. Create new execution context
    // 2. Bind parameters to arguments
    // 3. Execute function body
    // 4. Return result

    Err(JErrorType::TypeError("Function calls not yet fully implemented".to_string()))
}

/// Call a built-in function from the registry.
pub fn call_builtin(
    registry: &BuiltInRegistry,
    object: &str,
    method: &str,
    this_value: JsValue,
    args: Vec<JsValue>,
    ctx: &mut EvalContext,
) -> ValueResult {
    if let Some(builtin_fn) = registry.get_method(object, method) {
        builtin_fn.call(ctx, this_value, args)
    } else {
        Err(JErrorType::TypeError(format!(
            "{}.{} is not a function",
            object, method
        )))
    }
}

/// Check if a value is callable.
pub fn is_callable(value: &JsValue) -> bool {
    match value {
        JsValue::Object(_obj) => {
            // TODO: Check if object has [[Call]] internal method
            false
        }
        _ => false,
    }
}

/// Check if a value is a constructor.
pub fn is_constructor(value: &JsValue) -> bool {
    match value {
        JsValue::Object(_obj) => {
            // TODO: Check if object has [[Construct]] internal method
            false
        }
        _ => false,
    }
}
