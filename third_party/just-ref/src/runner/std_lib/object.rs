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
        .add_method("isPrototypeOf", object_is_prototype_of)
        .add_method("propertyIsEnumerable", object_property_is_enumerable)
        .add_method("keys", object_keys)
        .add_method("values", object_values)
        .add_method("entries", object_entries)
        .add_method("assign", object_assign)
        .add_method("defineProperty", object_define_property)
        .add_method("defineProperties", object_define_properties)
        .add_method("getOwnPropertyDescriptor", object_get_own_property_descriptor)
        .add_method("getOwnPropertyDescriptors", object_get_own_property_descriptors)
        .add_method("getOwnPropertyNames", object_get_own_property_names)
        .add_method("getOwnPropertySymbols", object_get_own_property_symbols)
        .add_method("hasOwn", object_has_own)
        .add_method("freeze", object_identity_arg0)
        .add_method("seal", object_identity_arg0)
        .add_method("preventExtensions", object_identity_arg0)
        .add_method("isFrozen", object_false_arg0)
        .add_method("isSealed", object_false_arg0)
        .add_method("isExtensible", object_true_arg0)
        .add_method("create", object_create)
        .add_method("getPrototypeOf", object_get_prototype_of)
        .add_method("setPrototypeOf", object_set_prototype_of)
        .add_method("is", object_is)
        .add_method("fromEntries", object_from_entries);

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

/// Object.prototype.isPrototypeOf(V) — true if `this` appears on V's chain.
fn object_is_prototype_of(
    _ctx: &mut EvalContext,
    this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let JsValue::Object(this_obj) = &this else {
        return Ok(JsValue::Boolean(false));
    };
    let Some(JsValue::Object(mut cur)) = args.into_iter().next() else {
        return Ok(JsValue::Boolean(false));
    };
    loop {
        let next = cur.borrow().as_js_object().get_prototype_of();
        match next {
            None => return Ok(JsValue::Boolean(false)),
            Some(p) => {
                if alloc::rc::Rc::ptr_eq(&p, this_obj) {
                    return Ok(JsValue::Boolean(true));
                }
                cur = p;
            }
        }
    }
}

/// Object.prototype.propertyIsEnumerable(key).
fn object_property_is_enumerable(
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
        if let Some(desc) = b
            .as_js_object()
            .get_object_base()
            .properties
            .get(&PropertyKey::Str(key))
        {
            let enumerable = match desc {
                PropertyDescriptor::Data(d) => d.enumerable,
                PropertyDescriptor::Accessor(a) => a.enumerable,
            };
            return Ok(JsValue::Boolean(enumerable));
        }
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

/// Object.getOwnPropertyDescriptors(obj) — `{ [key]: descriptor, … }`.
fn object_get_own_property_descriptors(
    ctx: &mut EvalContext,
    this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let target = args.into_iter().next().unwrap_or(JsValue::Undefined);
    let out = make_object(Vec::new());
    for key in crate::runner::eval::expression::own_string_keys(&target) {
        let desc = object_get_own_property_descriptor(
            ctx,
            this.clone(),
            alloc::vec![target.clone(), JsValue::String(key.clone())],
        )?;
        if !matches!(desc, JsValue::Undefined) {
            crate::runner::eval::expression::set_own_prop(&out, &key, desc, true);
        }
    }
    Ok(out)
}

/// Object.getOwnPropertySymbols(obj) — symbol keys. We have no real Symbols as
/// own keys yet; return an empty array (correct for plain objects).
fn object_get_own_property_symbols(
    _ctx: &mut EvalContext,
    _this: JsValue,
    _args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    Ok(make_array(Vec::new()))
}

/// Object.hasOwn(obj, key) — ES2022 static form of `hasOwnProperty`.
fn object_has_own(
    ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let target = args.get(0).cloned().unwrap_or(JsValue::Undefined);
    let key = args.get(1).cloned().unwrap_or(JsValue::Undefined);
    object_has_own_property(ctx, target, alloc::vec![key])
}

/// Object.create(proto[, props]) — new object with `[[Prototype]] = proto`.
/// `proto` must be an Object or `null` (TypeError otherwise).
fn object_create(
    ctx: &mut EvalContext,
    this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    // Start with a null-prototype shell; the match below installs `proto`.
    // (Do not default to Object.prototype — `Object.create(null)` must stay
    // chain-free, and an explicit proto replaces any default.)
    let obj = make_object(Vec::new());
    match args.get(0) {
        None | Some(JsValue::Undefined) => {
            // Spec: ToObject is not applied; non-object throws. Undefined is
            // not a valid prototype.
            return Err(JErrorType::TypeError(
                "Object prototype may only be an Object or null".to_string(),
            ));
        }
        Some(JsValue::Null) => {
            // Explicit null prototype — leave ObjectBase.prototype as None.
        }
        Some(JsValue::Object(p)) => {
            if let JsValue::Object(o) = &obj {
                o.borrow_mut()
                    .as_js_object_mut()
                    .get_object_base_mut()
                    .prototype = Some(p.clone());
            }
        }
        Some(_) => {
            return Err(JErrorType::TypeError(
                "Object prototype may only be an Object or null".to_string(),
            ));
        }
    }
    if let Some(descs) = args.get(1) {
        if matches!(descs, JsValue::Object(_)) {
            object_define_properties(ctx, this, alloc::vec![obj.clone(), descs.clone()])?;
        }
    }
    Ok(obj)
}

/// Object.getPrototypeOf(obj) — `[[Prototype]]` or `null`.
fn object_get_prototype_of(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let target = args.into_iter().next().unwrap_or(JsValue::Undefined);
    match target {
        JsValue::Object(o) => match o.borrow().as_js_object().get_prototype_of() {
            Some(p) => Ok(JsValue::Object(p)),
            None => Ok(JsValue::Null),
        },
        JsValue::Null | JsValue::Undefined => Err(JErrorType::TypeError(
            "Cannot convert undefined or null to object".to_string(),
        )),
        // Primitives: real engines return the corresponding prototype object.
        // We don't materialise those wrappers; report null (no chain).
        _ => Ok(JsValue::Null),
    }
}

/// Object.setPrototypeOf(obj, proto) — install `[[Prototype]]`.
fn object_set_prototype_of(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let target = args.get(0).cloned().unwrap_or(JsValue::Undefined);
    let proto = args.get(1).cloned().unwrap_or(JsValue::Undefined);
    let JsValue::Object(o) = &target else {
        // Spec: ToObject first; for primitives the return is the boxed form —
        // we just return the input and ignore the write (best-effort).
        return Ok(target);
    };
    let proto_slot = match &proto {
        JsValue::Null => None,
        JsValue::Object(p) => Some(p.clone()),
        _ => {
            return Err(JErrorType::TypeError(
                "Object prototype may only be an Object or null".to_string(),
            ));
        }
    };
    let ok = o
        .borrow_mut()
        .as_js_object_mut()
        .set_prototype_of(proto_slot);
    if !ok {
        return Err(JErrorType::TypeError(
            "Cyclic __proto__ value".to_string(),
        ));
    }
    Ok(target)
}

/// Object.is(a, b) — SameValue (distinguishes ±0 and NaN correctly).
fn object_is(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    use crate::runner::ds::operations::test_and_comparison::same_value;
    let a = args.get(0).cloned().unwrap_or(JsValue::Undefined);
    let b = args.get(1).cloned().unwrap_or(JsValue::Undefined);
    Ok(JsValue::Boolean(same_value(&a, &b)))
}

/// Object.fromEntries(iterable) — build an object from `[key, value]` pairs.
/// Accepts a real array of pairs (the common call shape).
fn object_from_entries(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    use crate::runner::eval::expression::{array_elements, set_own_prop, value_to_property_key};
    let iterable = args.into_iter().next().unwrap_or(JsValue::Undefined);
    let obj = make_object(Vec::new());
    for entry in array_elements(&iterable) {
        let pair = array_elements(&entry);
        let key = pair
            .get(0)
            .map(value_to_property_key)
            .unwrap_or_else(|| String::from("undefined"));
        let val = pair.get(1).cloned().unwrap_or(JsValue::Undefined);
        set_own_prop(&obj, &key, val, true);
    }
    Ok(obj)
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

// ---------------------------------------------------------------------------
// Unit tests (host `cargo test -p just-engine`)
// ---------------------------------------------------------------------------
#[cfg(test)]
mod object_static_tests {
    use crate::parser::JsParser;
    use crate::runner::ds::value::JsValue;
    use crate::runner::eval::statement::{execute_statement, hoist_var_declarations};
    use crate::runner::plugin::registry::BuiltInRegistry;
    use crate::runner::plugin::types::EvalContext;

    fn run(code: &str) -> String {
        let ast = JsParser::parse_to_ast_from_str(code)
            .unwrap_or_else(|e| panic!("parse {:?}: {:?}", code, e));
        let mut ctx = EvalContext::new();
        ctx.install_core_builtins(BuiltInRegistry::with_core());
        hoist_var_declarations(&ast.body, &mut ctx);
        let mut last = JsValue::Undefined;
        for stmt in &ast.body {
            let c = execute_statement(stmt, &mut ctx)
                .unwrap_or_else(|e| panic!("runtime {:?}: {:?}", code, e));
            if let Some(v) = c.value {
                last = v;
            }
        }
        match last {
            JsValue::String(s) => s,
            JsValue::Boolean(b) => b.to_string(),
            JsValue::Number(n) => format!("{n}"),
            JsValue::Null => "null".into(),
            JsValue::Undefined => "undefined".into(),
            other => other.to_string(),
        }
    }

    #[test]
    fn object_create_sets_prototype_chain() {
        assert_eq!(
            run("var p={x:7}; var o=Object.create(p); o.x"),
            "7"
        );
        assert_eq!(
            run("var o=Object.create(null); Object.getPrototypeOf(o)"),
            "null"
        );
    }

    #[test]
    fn object_set_get_prototype_of() {
        assert_eq!(
            run(
                "var a={}; var b={y:1}; Object.setPrototypeOf(a,b); a.y"
            ),
            "1"
        );
        assert_eq!(
            run(
                "var p={}; var o=Object.create(p); Object.getPrototypeOf(o)===p"
            ),
            "true"
        );
    }

    #[test]
    fn object_statics_are_first_class_on_constructor() {
        // Extract then call — must not be undefined.
        assert_eq!(
            run("var c=Object.create; var o=c({z:3}); o.z"),
            "3"
        );
        assert_eq!(
            run("typeof Object.create"),
            "function"
        );
        assert_eq!(
            run("typeof Object.setPrototypeOf"),
            "function"
        );
        assert_eq!(
            run("typeof Object.keys"),
            "function"
        );
        // Statics stay OFF instances.
        assert_eq!(run("typeof ({}).create"), "undefined");
        assert_eq!(run("typeof ({}).keys"), "undefined");
    }

    #[test]
    fn object_keys_and_is_and_from_entries() {
        // keys returns an array; join for a stable string.
        assert_eq!(
            run("Object.keys({a:1,b:2}).sort().join(',')"),
            "a,b"
        );
        assert_eq!(run("Object.is(NaN, NaN)"), "true");
        assert_eq!(run("Object.is(1, 1)"), "true");
        assert_eq!(run("Object.is(1, 2)"), "false");
        assert_eq!(
            run("Object.fromEntries([['k',9]]).k"),
            "9"
        );
    }

    #[test]
    fn object_in_operator_sees_statics() {
        assert_eq!(run("'create' in Object"), "true");
        assert_eq!(run("'setPrototypeOf' in Object"), "true");
        assert_eq!(run("'keys' in Object"), "true");
        assert_eq!(run("'hasOwn' in Object"), "true");
        assert_eq!(run("'getOwnPropertyDescriptors' in Object"), "true");
    }

    #[test]
    fn object_has_own_and_descriptors() {
        assert_eq!(run("Object.hasOwn({a:1}, 'a')"), "true");
        assert_eq!(run("Object.hasOwn({a:1}, 'b')"), "false");
        assert_eq!(
            run("Object.getOwnPropertyDescriptors({x:7}).x.value"),
            "7"
        );
    }

    #[test]
    fn error_subclass_proto_chain() {
        assert_eq!(
            run("Object.getPrototypeOf(TypeError.prototype) === Error.prototype"),
            "true"
        );
        assert_eq!(
            run("new TypeError('x') instanceof Error"),
            "true"
        );
        assert_eq!(
            run("new TypeError('x') instanceof TypeError"),
            "true"
        );
        // After Object statics
        assert_eq!(
            run(
                "Object.create({}); Object.keys({a:1}); \
                 Object.getPrototypeOf(TypeError.prototype) === Error.prototype"
            ),
            "true"
        );
        // After class extends Error — must not corrupt the TypeError chain.
        assert_eq!(
            run(
                "class E extends Error { constructor(m) { super(m); this.name = 'E'; } } \
                 var e = new E('hi'); \
                 (e instanceof Error) && (e.message === 'hi') && \
                 (new TypeError('x') instanceof Error)"
            ),
            "true"
        );
    }
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
