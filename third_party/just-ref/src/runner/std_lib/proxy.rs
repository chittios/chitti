//! ChittiOS: `Proxy` and `Reflect` built-ins.
//!
//! `new Proxy(target, handler)` returns an object tagged with `__proxy_target__`
//! + `__proxy_handler__`; the interpreter's `get_property_with_receiver` /
//! `set_property_with_ctx` detect those and invoke the handler's `get`/`set`
//! traps (falling back to the target). `Reflect.*` are the matching static
//! operations over an object's own properties.

#[allow(unused_imports)]
use num_traits::Float;
use alloc::borrow::ToOwned;
use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use alloc::vec;
use crate::runner::ds::error::JErrorType;
use crate::runner::ds::value::JsValue;
use crate::runner::plugin::registry::BuiltInRegistry;
use crate::runner::plugin::types::{BuiltInObject, EvalContext};
use crate::runner::eval::expression as ex;

pub fn register(registry: &mut BuiltInRegistry) {
    let proxy = BuiltInObject::new("Proxy").with_constructor(proxy_constructor);
    registry.register_object(proxy);

    let reflect = BuiltInObject::new("Reflect")
        .with_no_prototype()
        .add_method("get", reflect_get)
        .add_method("set", reflect_set)
        .add_method("has", reflect_has)
        .add_method("deleteProperty", reflect_delete)
        .add_method("ownKeys", reflect_own_keys)
        .add_method("apply", reflect_apply);
    registry.register_object(reflect);
}

fn proxy_constructor(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let target = args.get(0).cloned().unwrap_or(JsValue::Undefined);
    let handler = args.get(1).cloned().unwrap_or(JsValue::Undefined);
    if !matches!(target, JsValue::Object(_)) || !matches!(handler, JsValue::Object(_)) {
        return Err(JErrorType::TypeError(
            "Cannot create proxy with a non-object as target or handler".to_string(),
        ));
    }
    let p = ex::make_object(vec![]);
    ex::set_own_prop(&p, "__proxy_target__", target, false);
    ex::set_own_prop(&p, "__proxy_handler__", handler, false);
    Ok(p)
}

fn key_arg(args: &[JsValue], i: usize) -> String {
    match args.get(i) {
        Some(JsValue::String(s)) => s.clone(),
        Some(other) => ex::value_to_property_key(other),
        None => "undefined".to_string(),
    }
}

fn reflect_get(
    ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let target = args.get(0).cloned().unwrap_or(JsValue::Undefined);
    ex::get_property(&target, &key_arg(&args, 1), ctx)
}

fn reflect_set(
    ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let target = args.get(0).cloned().unwrap_or(JsValue::Undefined);
    let value = args.get(2).cloned().unwrap_or(JsValue::Undefined);
    ex::set_property(&target, &key_arg(&args, 1), value, ctx)?;
    Ok(JsValue::Boolean(true))
}

fn reflect_has(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let target = args.get(0).cloned().unwrap_or(JsValue::Undefined);
    Ok(JsValue::Boolean(ex::has_own_prop(&target, &key_arg(&args, 1))))
}

fn reflect_delete(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let target = args.get(0).cloned().unwrap_or(JsValue::Undefined);
    Ok(JsValue::Boolean(ex::delete_own_prop(&target, &key_arg(&args, 1))))
}

fn reflect_own_keys(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let target = args.get(0).cloned().unwrap_or(JsValue::Undefined);
    let keys = ex::own_string_keys(&target)
        .into_iter()
        .map(JsValue::String)
        .collect();
    Ok(ex::make_array(keys))
}

fn reflect_apply(
    ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let f = args.get(0).cloned().unwrap_or(JsValue::Undefined);
    let this_arg = args.get(1).cloned().unwrap_or(JsValue::Undefined);
    let arg_list = args.get(2).cloned().unwrap_or(JsValue::Undefined);
    let call_args = ex::array_elements(&arg_list);
    ex::call_value(&f, this_arg, call_args, ctx)
}
