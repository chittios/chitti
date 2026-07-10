//! Object built-in.
//!
//! Provides Object constructor and prototype methods.

// ChittiOS-alloc-prelude
#[allow(unused_imports)]
use num_traits::Float;
use alloc::borrow::ToOwned;
use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use crate::runner::ds::error::JErrorType;
use crate::runner::ds::value::JsValue;
use crate::runner::ds::object::JsObject;
use crate::runner::ds::object_property::{PropertyDescriptor, PropertyDescriptorData, PropertyKey};
use crate::runner::plugin::registry::BuiltInRegistry;
use crate::runner::plugin::types::{BuiltInObject, EvalContext};
use crate::runner::eval::expression::{make_array, make_object};

/// Enumerable own (key, value) pairs of an object, ordered numeric-ascending
/// then lexicographic (correct for arrays; deterministic for plain objects,
/// since the underlying store is an unordered hash map).
fn enum_own_pairs(v: &JsValue) -> Vec<(String, JsValue)> {
    let mut pairs: Vec<(String, JsValue)> = Vec::new();
    if let JsValue::Object(o) = v {
        let b = o.borrow();
        for (k, d) in b.as_js_object().get_object_base().properties.iter() {
            let PropertyKey::Str(name) = k else { continue };
            let (enumerable, value) = match d {
                PropertyDescriptor::Data(dd) => (dd.enumerable, dd.value.clone()),
                PropertyDescriptor::Accessor(a) => (a.enumerable, JsValue::Undefined),
            };
            if enumerable {
                pairs.push((name.clone(), value));
            }
        }
    }
    pairs.sort_by(|a, b| match (a.0.parse::<i64>(), b.0.parse::<i64>()) {
        (Ok(x), Ok(y)) => x.cmp(&y),
        (Ok(_), Err(_)) => core::cmp::Ordering::Less,
        (Err(_), Ok(_)) => core::cmp::Ordering::Greater,
        (Err(_), Err(_)) => a.0.cmp(&b.0),
    });
    pairs
}

/// Register the Object built-in with the registry.
pub fn register(registry: &mut BuiltInRegistry) {
    let object = BuiltInObject::new("Object")
        .with_no_prototype()
        .with_constructor(object_constructor)
        .add_method("toString", object_to_string)
        .add_method("valueOf", object_value_of)
        .add_method("hasOwnProperty", object_has_own_property)
        .add_method("keys", object_keys)
        .add_method("values", object_values)
        .add_method("entries", object_entries)
        .add_method("assign", object_assign)
        .add_method("defineProperty", object_define_property)
        .add_method("defineProperties", object_define_properties)
        .add_method("getOwnPropertyDescriptor", object_get_own_property_descriptor)
        .add_method("getOwnPropertyNames", object_get_own_property_names)
        .add_method("freeze", object_identity_arg0)
        .add_method("seal", object_identity_arg0)
        .add_method("preventExtensions", object_identity_arg0)
        .add_method("isFrozen", object_false_arg0)
        .add_method("isSealed", object_false_arg0)
        .add_method("isExtensible", object_true_arg0)
        .add_method("create", object_create)
        .add_method("getPrototypeOf", object_get_prototype_of);

    registry.register_object(object);
}

/// Object constructor.
fn object_constructor(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    match args.first() {
        None | Some(JsValue::Null) | Some(JsValue::Undefined) => Ok(make_object(Vec::new())),
        Some(v) => Ok(v.clone()),
    }
}

/// Object.prototype.toString
fn object_to_string(
    _ctx: &mut EvalContext,
    this: JsValue,
    _args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let tag = match &this {
        JsValue::Undefined => "Undefined",
        JsValue::Null => "Null",
        JsValue::Boolean(_) => "Boolean",
        JsValue::Number(_) => "Number",
        JsValue::String(_) => "String",
        JsValue::Symbol(_) => "Symbol",
        JsValue::Object(_) => "Object",
    };
    Ok(JsValue::String(format!("[object {}]", tag)))
}

/// Object.prototype.valueOf
fn object_value_of(
    _ctx: &mut EvalContext,
    this: JsValue,
    _args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    // Returns the primitive value of the object (for primitives, returns itself)
    Ok(this)
}

/// Object.prototype.hasOwnProperty
fn object_has_own_property(
    _ctx: &mut EvalContext,
    this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let key = match args.first() {
        Some(JsValue::String(s)) => s.clone(),
        Some(other) => crate::runner::eval::expression::value_to_property_key(other),
        None => return Ok(JsValue::Boolean(false)),
    };
    if let JsValue::Object(o) = &this {
        let b = o.borrow();
        let has = b
            .as_js_object()
            .get_object_base()
            .properties
            .contains_key(&PropertyKey::Str(key));
        return Ok(JsValue::Boolean(has));
    }
    Ok(JsValue::Boolean(false))
}

/// Object.keys(obj) — enumerable own property names.
fn object_keys(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let target = args.into_iter().next().unwrap_or(JsValue::Undefined);
    let keys = enum_own_pairs(&target)
        .into_iter()
        .map(|(k, _)| JsValue::String(k))
        .collect();
    Ok(make_array(keys))
}

/// Object.values(obj) — enumerable own property values.
fn object_values(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let target = args.into_iter().next().unwrap_or(JsValue::Undefined);
    let vals = enum_own_pairs(&target).into_iter().map(|(_, v)| v).collect();
    Ok(make_array(vals))
}

/// Object.entries(obj) — enumerable own `[key, value]` pairs.
fn object_entries(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let target = args.into_iter().next().unwrap_or(JsValue::Undefined);
    let entries = enum_own_pairs(&target)
        .into_iter()
        .map(|(k, v)| make_array(alloc::vec![JsValue::String(k), v]))
        .collect();
    Ok(make_array(entries))
}

/// Object.defineProperty(obj, key, descriptor) — define a data/accessor prop.
fn object_define_property(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let target = args.get(0).cloned().unwrap_or(JsValue::Undefined);
    let key = match args.get(1) {
        Some(JsValue::String(s)) => s.clone(),
        Some(other) => crate::runner::eval::expression::value_to_property_key(other),
        None => return Ok(target),
    };
    let desc = args.get(2).cloned().unwrap_or(JsValue::Undefined);
    let value = crate::runner::eval::expression::get_own_prop_value(&desc, "value")
        .unwrap_or(JsValue::Undefined);
    let enumerable = matches!(
        crate::runner::eval::expression::get_own_prop_value(&desc, "enumerable"),
        Some(JsValue::Boolean(true))
    );
    crate::runner::eval::expression::set_own_prop(&target, &key, value, enumerable);
    Ok(target)
}

/// Object.defineProperties(obj, descriptors).
fn object_define_properties(
    ctx: &mut EvalContext,
    this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let target = args.get(0).cloned().unwrap_or(JsValue::Undefined);
    let descs = args.get(1).cloned().unwrap_or(JsValue::Undefined);
    for (k, d) in enum_own_pairs(&descs) {
        object_define_property(ctx, this.clone(), alloc::vec![target.clone(), JsValue::String(k), d])?;
    }
    Ok(target)
}

/// Object.getOwnPropertyDescriptor(obj, key) — `{value, writable, enumerable,
/// configurable}` or undefined.
fn object_get_own_property_descriptor(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let target = args.get(0).cloned().unwrap_or(JsValue::Undefined);
    let key = match args.get(1) {
        Some(JsValue::String(s)) => s.clone(),
        Some(other) => crate::runner::eval::expression::value_to_property_key(other),
        None => return Ok(JsValue::Undefined),
    };
    match crate::runner::eval::expression::get_own_prop_descriptor(&target, &key) {
        Some((v, writable, enumerable, configurable)) => Ok(make_object(alloc::vec![
            ("value".to_string(), v),
            ("writable".to_string(), JsValue::Boolean(writable)),
            ("enumerable".to_string(), JsValue::Boolean(enumerable)),
            ("configurable".to_string(), JsValue::Boolean(configurable)),
        ])),
        None => Ok(JsValue::Undefined),
    }
}

/// Object.getOwnPropertyNames(obj) — all own string keys (enumerable or not).
fn object_get_own_property_names(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let target = args.into_iter().next().unwrap_or(JsValue::Undefined);
    let keys = crate::runner::eval::expression::own_string_keys(&target)
        .into_iter()
        .map(JsValue::String)
        .collect();
    Ok(make_array(keys))
}

/// Object.create(proto[, props]) — a new object (proto chain best-effort).
fn object_create(
    ctx: &mut EvalContext,
    this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let obj = make_object(Vec::new());
    if let Some(descs) = args.get(1) {
        if matches!(descs, JsValue::Object(_)) {
            object_define_properties(ctx, this, alloc::vec![obj.clone(), descs.clone()])?;
        }
    }
    Ok(obj)
}

fn object_get_prototype_of(
    _ctx: &mut EvalContext,
    _this: JsValue,
    _args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    Ok(JsValue::Null)
}

fn object_identity_arg0(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    Ok(args.into_iter().next().unwrap_or(JsValue::Undefined))
}
fn object_false_arg0(
    _ctx: &mut EvalContext,
    _this: JsValue,
    _args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    Ok(JsValue::Boolean(false))
}
fn object_true_arg0(
    _ctx: &mut EvalContext,
    _this: JsValue,
    _args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    Ok(JsValue::Boolean(true))
}

/// Object.assign(target, ...sources) — copy enumerable own props into target.
fn object_assign(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let mut it = args.into_iter();
    let target = it.next().unwrap_or(JsValue::Undefined);
    if let JsValue::Object(o) = &target {
        for source in it {
            for (k, v) in enum_own_pairs(&source) {
                let mut b = o.borrow_mut();
                b.as_js_object_mut().get_object_base_mut().properties.insert(
                    PropertyKey::Str(k),
                    PropertyDescriptor::Data(PropertyDescriptorData {
                        value: v,
                        writable: true,
                        enumerable: true,
                        configurable: true,
                    }),
                );
            }
        }
    }
    Ok(target)
}
