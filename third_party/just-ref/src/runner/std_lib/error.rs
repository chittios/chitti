//! Error built-in objects.
//!
//! Provides Error, TypeError, ReferenceError, SyntaxError, RangeError constructors.
//! Each constructs a real object with `name` + `message` (and a `__builtin_name__`
//! tag so `Object.prototype.toString` reports the right class).

// ChittiOS-alloc-prelude
#[allow(unused_imports)]
use num_traits::Float;
use alloc::borrow::ToOwned;
use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use crate::runner::ds::error::JErrorType;
use crate::runner::ds::value::JsValue;
use crate::runner::eval::expression::{make_object, set_own_prop};
use crate::runner::plugin::registry::BuiltInRegistry;
use crate::runner::plugin::types::{BuiltInObject, EvalContext};

/// Register all error types with the registry.
pub fn register(registry: &mut BuiltInRegistry) {
    // Base Error
    let error = BuiltInObject::new("Error")
        .with_constructor(error_constructor)
        .add_method("toString", error_to_string);
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

/// Build a real Error-like object: `{ name, message }` + tag.
fn make_error(name: &str, message: String) -> JsValue {
    let obj = make_object(alloc::vec![
        ("name".to_string(), JsValue::String(name.to_string())),
        ("message".to_string(), JsValue::String(message)),
    ]);
    // So `Object.prototype.toString.call(e)` → `[object Error]`.
    set_own_prop(
        &obj,
        "__builtin_name__",
        JsValue::String(name.to_string()),
        false,
    );
    obj
}

/// Shared body for all Error* constructors.
fn construct_error(
    name: &str,
    ctx: &mut EvalContext,
    this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    use crate::runner::eval::expression::get_property_with_ctx;
    let message = get_message(&args);
    // Prefer to stamp fields onto the `new`-created `this` when it's already an
    // object (prototype already wired by `create_new_object_for_constructor`).
    let result = if let JsValue::Object(_) = &this {
        set_own_prop(&this, "name", JsValue::String(name.to_string()), true);
        set_own_prop(&this, "message", JsValue::String(message), true);
        set_own_prop(
            &this,
            "__builtin_name__",
            JsValue::String(name.to_string()),
            false,
        );
        this
    } else {
        // `Error("msg")` without `new` also creates an Error object (ES5+).
        make_error(name, message)
    };
    // Ensure `[[Prototype]]` is `Name.prototype` when the super-global
    // short-circuit path called us with `this === undefined` (no pre-wired
    // instance). Synthetic carriers for built-in constructors come from the
    // property path, not own data props.
    if let JsValue::Object(o) = &result {
        let needs_proto = o.borrow().as_js_object().get_prototype_of().is_none();
        if needs_proto {
            let sg = ctx.super_global.clone();
            let ctor = {
                let env = sg.borrow();
                env.resolve_binding(name, ctx).ok()
            }; // drop env borrow before further ctx use
            if let Some(ctor) = ctor {
                if let Ok(JsValue::Object(proto)) = get_property_with_ctx(&ctor, "prototype", ctx) {
                    o.borrow_mut()
                        .as_js_object_mut()
                        .get_object_base_mut()
                        .prototype = Some(proto);
                }
            }
        }
    }
    Ok(result)
}

fn error_constructor(
    ctx: &mut EvalContext,
    this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    construct_error("Error", ctx, this, args)
}

fn type_error_constructor(
    ctx: &mut EvalContext,
    this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    construct_error("TypeError", ctx, this, args)
}

fn reference_error_constructor(
    ctx: &mut EvalContext,
    this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    construct_error("ReferenceError", ctx, this, args)
}

fn syntax_error_constructor(
    ctx: &mut EvalContext,
    this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    construct_error("SyntaxError", ctx, this, args)
}

fn range_error_constructor(
    ctx: &mut EvalContext,
    this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    construct_error("RangeError", ctx, this, args)
}

fn eval_error_constructor(
    ctx: &mut EvalContext,
    this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    construct_error("EvalError", ctx, this, args)
}

fn uri_error_constructor(
    ctx: &mut EvalContext,
    this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    construct_error("URIError", ctx, this, args)
}

/// Error.prototype.toString → `"Name: message"` or just `Name`.
fn error_to_string(
    _ctx: &mut EvalContext,
    this: JsValue,
    _args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    use crate::runner::eval::expression::get_own_prop_value;
    let name = match get_own_prop_value(&this, "name") {
        Some(JsValue::String(s)) => s,
        _ => String::from("Error"),
    };
    let message = match get_own_prop_value(&this, "message") {
        Some(JsValue::String(s)) => s,
        Some(v) => v.to_string(),
        None => String::new(),
    };
    if message.is_empty() {
        Ok(JsValue::String(name))
    } else {
        Ok(JsValue::String(format!("{name}: {message}")))
    }
}
