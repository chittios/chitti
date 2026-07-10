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
        .add_method("assign", object_assign);

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
