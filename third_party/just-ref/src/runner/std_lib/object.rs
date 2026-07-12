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
use crate::runner::ds::object_property::{
    PropertyDescriptor, PropertyDescriptorAccessor, PropertyDescriptorData, PropertyKey,
};
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
///
/// `Object()` / `Object(null|undefined)` make a fresh object; an object argument
/// is returned unchanged; a **primitive** argument is boxed into a wrapper object
/// carrying its value under `__primitive_value__` (non-enumerable). The wrapper's
/// `ToPrimitive` (see `expression::to_primitive`) unwraps it, so `Object(2n) + 1n`
/// evaluates to `3n`.
fn object_constructor(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    match args.into_iter().next() {
        None | Some(JsValue::Null) | Some(JsValue::Undefined) => Ok(make_object(Vec::new())),
        Some(v @ JsValue::Object(_)) => Ok(v),
        Some(prim) => {
            let obj = make_object(Vec::new());
            crate::runner::eval::expression::set_own_prop(&obj, "__primitive_value__", prim, false);
            Ok(obj)
        }
    }
}

/// Object.prototype.toString
fn object_to_string(
    _ctx: &mut EvalContext,
    this: JsValue,
    _args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    // The [[Class]]-style tag. For objects, distinguish the common exotic
    // kinds the `Object.prototype.toString.call(x)` type-detection idiom relies
    // on (Array/Function/RegExp/Error/…) — bliss's `type()`, jQuery's `$.type`
    // and lodash all key off these.
    let tag = match &this {
        JsValue::Undefined => "Undefined",
        JsValue::Null => "Null",
        JsValue::Boolean(_) => "Boolean",
        JsValue::Number(_) => "Number",
        JsValue::String(_) => "String",
        JsValue::BigInt(_) => "BigInt",
        JsValue::Symbol(_) => "Symbol",
        JsValue::Object(_) => {
            use crate::runner::eval::expression::{get_own_prop_value, value_is_callable};
            // Arrays carry `__array__`; RegExp/Error/etc. carry a
            // `__builtin_name__`; a callable (with none of those) is a Function.
            if get_own_prop_value(&this, "__array__").is_some() {
                "Array"
            } else if let Some(JsValue::String(name)) =
                get_own_prop_value(&this, "__builtin_name__")
            {
                // "RegExp" → "[object RegExp]"; any "*Error" → "[object Error]".
                if name.ends_with("Error") {
                    "Error"
                } else if name == "RegExp" {
                    "RegExp"
                } else if value_is_callable(&this) {
                    "Function"
                } else {
                    "Object"
                }
            } else if value_is_callable(&this) {
                "Function"
            } else {
                "Object"
            }
        }
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
    use crate::runner::eval::expression as ex;
    let enumerable = matches!(
        ex::get_own_prop_value(&desc, "enumerable"),
        Some(JsValue::Boolean(true))
    );

    // Accessor descriptor: the descriptor object mentions `get` and/or `set`.
    // (A present-but-`undefined` `get`/`set` still makes it an accessor.)
    if ex::has_own_prop(&desc, "get") || ex::has_own_prop(&desc, "set") {
        let configurable = matches!(
            ex::get_own_prop_value(&desc, "configurable"),
            Some(JsValue::Boolean(true))
        );
        let to_fn = |v: Option<JsValue>| -> Option<crate::runner::ds::object::JsObjectType> {
            match v {
                Some(JsValue::Object(o)) => Some(o),
                _ => None,
            }
        };
        let get = to_fn(ex::get_own_prop_value(&desc, "get"));
        let set = to_fn(ex::get_own_prop_value(&desc, "set"));
        if let JsValue::Object(o) = &target {
            o.borrow_mut()
                .as_js_object_mut()
                .get_object_base_mut()
                .properties
                .insert(
                    PropertyKey::Str(key),
                    PropertyDescriptor::Accessor(PropertyDescriptorAccessor {
                        get,
                        set,
                        enumerable,
                        configurable,
                    }),
                );
        }
        return Ok(target);
    }

    // Data descriptor (unchanged: writable/configurable default to true here).
    let value = ex::get_own_prop_value(&desc, "value").unwrap_or(JsValue::Undefined);
    ex::set_own_prop(&target, &key, value, enumerable);
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
    if let JsValue::Object(o) = &target {
        let b = o.borrow();
        if let Some(desc) = b
            .as_js_object()
            .get_object_base()
            .properties
            .get(&PropertyKey::Str(key.clone()))
        {
            return Ok(match desc {
                PropertyDescriptor::Data(d) => make_object(alloc::vec![
                    ("value".to_string(), d.value.clone()),
                    ("writable".to_string(), JsValue::Boolean(d.writable)),
                    ("enumerable".to_string(), JsValue::Boolean(d.enumerable)),
                    ("configurable".to_string(), JsValue::Boolean(d.configurable)),
                ]),
                PropertyDescriptor::Accessor(a) => {
                    let getset = |f: &Option<crate::runner::ds::object::JsObjectType>| match f {
                        Some(fun) => JsValue::Object(fun.clone()),
                        None => JsValue::Undefined,
                    };
                    make_object(alloc::vec![
                        ("get".to_string(), getset(&a.get)),
                        ("set".to_string(), getset(&a.set)),
                        ("enumerable".to_string(), JsValue::Boolean(a.enumerable)),
                        ("configurable".to_string(), JsValue::Boolean(a.configurable)),
                    ])
                }
            });
        }
    }
    Ok(JsValue::Undefined)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runner::eval::expression::{get_own_prop_value, has_own_prop, make_object};
    use crate::runner::plugin::types::EvalContext;

    /// `Object(primitive)` boxes into a wrapper carrying `__primitive_value__`.
    #[test]
    fn object_constructor_boxes_bigint() {
        let mut ctx = EvalContext::new();
        let arg = JsValue::BigInt(num_bigint::BigInt::from(2));
        let r = object_constructor(&mut ctx, JsValue::Undefined, alloc::vec![arg]).unwrap();
        assert!(matches!(r, JsValue::Object(_)));
        match get_own_prop_value(&r, "__primitive_value__") {
            Some(JsValue::BigInt(b)) => assert_eq!(b, num_bigint::BigInt::from(2)),
            other => panic!("expected boxed 2n, got {:?}", other),
        }
    }

    /// An object argument is returned unchanged (not re-boxed).
    #[test]
    fn object_constructor_object_arg_unchanged() {
        let mut ctx = EvalContext::new();
        let obj = make_object(alloc::vec![("a".to_string(), JsValue::Boolean(true))]);
        let r =
            object_constructor(&mut ctx, JsValue::Undefined, alloc::vec![obj]).unwrap();
        assert!(!has_own_prop(&r, "__primitive_value__"));
        assert_eq!(get_own_prop_value(&r, "a"), Some(JsValue::Boolean(true)));
    }

    /// `Object()` / `Object(undefined)` makes a fresh plain object.
    #[test]
    fn object_constructor_empty_makes_object() {
        let mut ctx = EvalContext::new();
        let r = object_constructor(&mut ctx, JsValue::Undefined, alloc::vec![]).unwrap();
        assert!(matches!(r, JsValue::Object(_)));
        assert!(!has_own_prop(&r, "__primitive_value__"));
    }

    /// `defineProperty` with a `get` builds an accessor; `getOwnPropertyDescriptor`
    /// round-trips it (`get` present, no `value`).
    #[test]
    fn define_property_accessor_roundtrips() {
        let mut ctx = EvalContext::new();
        let target = make_object(alloc::vec![]);
        let getter = make_object(alloc::vec![]); // stand-in for a function object
        let desc = make_object(alloc::vec![("get".to_string(), getter)]);
        object_define_property(
            &mut ctx,
            JsValue::Undefined,
            alloc::vec![target.clone(), JsValue::String("x".to_string()), desc],
        )
        .unwrap();
        let out = object_get_own_property_descriptor(
            &mut ctx,
            JsValue::Undefined,
            alloc::vec![target, JsValue::String("x".to_string())],
        )
        .unwrap();
        assert!(has_own_prop(&out, "get"));
        assert!(has_own_prop(&out, "set"));
        assert!(!has_own_prop(&out, "value"));
    }

    /// A data-descriptor `defineProperty` still round-trips as a data descriptor.
    #[test]
    fn define_property_data_roundtrips() {
        let mut ctx = EvalContext::new();
        let target = make_object(alloc::vec![]);
        let desc = make_object(alloc::vec![
            ("value".to_string(), JsValue::Number(crate::runner::ds::value::JsNumberType::Integer(7))),
            ("enumerable".to_string(), JsValue::Boolean(true)),
        ]);
        object_define_property(
            &mut ctx,
            JsValue::Undefined,
            alloc::vec![target.clone(), JsValue::String("y".to_string()), desc],
        )
        .unwrap();
        let out = object_get_own_property_descriptor(
            &mut ctx,
            JsValue::Undefined,
            alloc::vec![target, JsValue::String("y".to_string())],
        )
        .unwrap();
        assert!(has_own_prop(&out, "value"));
        assert!(!has_own_prop(&out, "get"));
    }
}
