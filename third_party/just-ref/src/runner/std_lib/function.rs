//! Function built-in (prototype methods used by every callable).

// ChittiOS-alloc-prelude
#[allow(unused_imports)]
use num_traits::Float;
use alloc::borrow::ToOwned;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use crate::runner::ds::error::JErrorType;
use crate::runner::ds::value::JsValue;
use crate::runner::eval::expression::get_own_prop_value;
use crate::runner::plugin::registry::BuiltInRegistry;
use crate::runner::plugin::types::{BuiltInObject, EvalContext};

/// Register Function with the methods every callable is expected to have.
pub fn register(registry: &mut BuiltInRegistry) {
    let function = BuiltInObject::new("Function")
        .with_constructor(function_constructor_stub)
        .add_method("toString", function_to_string)
        .add_method("valueOf", function_value_of);
    registry.register_object(function);
}

fn function_constructor_stub(
    _ctx: &mut EvalContext,
    _this: JsValue,
    _args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    // Real `Function(...)` body construction is handled by the globals resolver.
    Err(JErrorType::TypeError(
        "Function constructor via registry not used".to_string(),
    ))
}

/// Function.prototype.toString — a non-source-preserving native rendering.
fn function_to_string(
    _ctx: &mut EvalContext,
    this: JsValue,
    _args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let name = match get_own_prop_value(&this, "name") {
        Some(JsValue::String(s)) => s,
        _ => String::new(),
    };
    Ok(JsValue::String(alloc::format!(
        "function {}() {{ [native code] }}",
        name
    )))
}

/// Function.prototype.valueOf — identity.
fn function_value_of(
    _ctx: &mut EvalContext,
    this: JsValue,
    _args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    Ok(this)
}
