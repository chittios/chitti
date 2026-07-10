//! ChittiOS: `Map` and `Set` built-ins.
//!
//! Instances are Ordinary objects tagged `__builtin_name__ = "Map"`/`"Set"` (so
//! the interpreter's instance-method dispatch routes `m.set(...)` etc. here)
//! plus parallel `__keys__`/`__vals__` arrays and a non-enumerable `size`
//! property. Key comparison is SameValue-ish (`js_same`).

#[allow(unused_imports)]
use num_traits::Float;
use alloc::borrow::ToOwned;
use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use alloc::vec;
use crate::runner::ds::error::JErrorType;
use crate::runner::ds::value::{JsValue, JsNumberType};
use crate::runner::plugin::registry::BuiltInRegistry;
use crate::runner::plugin::types::{BuiltInObject, EvalContext};
use crate::runner::eval::expression::{
    array_elements, array_set_elements, call_value, get_own_prop_value, make_array, make_object,
    set_own_prop, value_is_callable,
};

pub fn register(registry: &mut BuiltInRegistry) {
    let map = BuiltInObject::new("Map")
        .with_constructor(map_constructor)
        .add_method("set", map_set)
        .add_method("get", map_get)
        .add_method("has", map_has)
        .add_method("delete", map_delete)
        .add_method("clear", coll_clear)
        .add_method("forEach", map_for_each)
        .add_method("keys", map_keys)
        .add_method("values", map_values);
    registry.register_object(map);

    let set = BuiltInObject::new("Set")
        .with_constructor(set_constructor)
        .add_method("add", set_add)
        .add_method("has", set_has)
        .add_method("delete", set_delete)
        .add_method("clear", coll_clear)
        .add_method("forEach", set_for_each)
        .add_method("values", set_values);
    registry.register_object(set);
}

fn js_same(a: &JsValue, b: &JsValue) -> bool {
    match (a, b) {
        (JsValue::Undefined, JsValue::Undefined) => true,
        (JsValue::Null, JsValue::Null) => true,
        (JsValue::Boolean(x), JsValue::Boolean(y)) => x == y,
        (JsValue::String(x), JsValue::String(y)) => x == y,
        (JsValue::Number(x), JsValue::Number(y)) => nf(x) == nf(y),
        (JsValue::Object(x), JsValue::Object(y)) => alloc::rc::Rc::ptr_eq(x, y),
        _ => false,
    }
}
fn nf(n: &JsNumberType) -> f64 {
    match n {
        JsNumberType::Integer(i) => *i as f64,
        JsNumberType::Float(f) => *f,
        _ => f64::NAN,
    }
}

fn new_collection(name: &str) -> JsValue {
    let obj = make_object(vec![]);
    set_own_prop(&obj, "__builtin_name__", JsValue::String(name.to_string()), false);
    set_own_prop(&obj, "__keys__", make_array(vec![]), false);
    set_own_prop(&obj, "__vals__", make_array(vec![]), false);
    set_own_prop(&obj, "size", JsValue::Number(JsNumberType::Integer(0)), false);
    obj
}

fn keys_of(this: &JsValue) -> JsValue {
    get_own_prop_value(this, "__keys__").unwrap_or(JsValue::Undefined)
}
fn vals_of(this: &JsValue) -> JsValue {
    get_own_prop_value(this, "__vals__").unwrap_or(JsValue::Undefined)
}
fn set_size(this: &JsValue, n: usize) {
    set_own_prop(this, "size", JsValue::Number(JsNumberType::Integer(n as i64)), false);
}
fn find_key(keys: &[JsValue], k: &JsValue) -> Option<usize> {
    keys.iter().position(|e| js_same(e, k))
}

// ---- Map --------------------------------------------------------------------

fn map_constructor(
    ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let m = new_collection("Map");
    // Optional iterable of [k, v] pairs.
    if let Some(init) = args.first() {
        for entry in array_elements(init) {
            let pair = array_elements(&entry);
            let k = pair.get(0).cloned().unwrap_or(JsValue::Undefined);
            let v = pair.get(1).cloned().unwrap_or(JsValue::Undefined);
            map_set(ctx, m.clone(), vec![k, v])?;
        }
    }
    Ok(m)
}

fn map_set(
    _ctx: &mut EvalContext,
    this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let k = args.get(0).cloned().unwrap_or(JsValue::Undefined);
    let v = args.get(1).cloned().unwrap_or(JsValue::Undefined);
    let keys_arr = keys_of(&this);
    let vals_arr = vals_of(&this);
    let mut keys = array_elements(&keys_arr);
    let mut vals = array_elements(&vals_arr);
    if let Some(i) = find_key(&keys, &k) {
        vals[i] = v;
    } else {
        keys.push(k);
        vals.push(v);
    }
    array_set_elements(&keys_arr, &keys);
    array_set_elements(&vals_arr, &vals);
    set_size(&this, keys.len());
    Ok(this)
}

fn map_get(
    _ctx: &mut EvalContext,
    this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let k = args.get(0).cloned().unwrap_or(JsValue::Undefined);
    let keys = array_elements(&keys_of(&this));
    match find_key(&keys, &k) {
        Some(i) => Ok(array_elements(&vals_of(&this)).get(i).cloned().unwrap_or(JsValue::Undefined)),
        None => Ok(JsValue::Undefined),
    }
}

fn map_has(
    _ctx: &mut EvalContext,
    this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let k = args.get(0).cloned().unwrap_or(JsValue::Undefined);
    Ok(JsValue::Boolean(find_key(&array_elements(&keys_of(&this)), &k).is_some()))
}

fn map_delete(
    _ctx: &mut EvalContext,
    this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let k = args.get(0).cloned().unwrap_or(JsValue::Undefined);
    let keys_arr = keys_of(&this);
    let vals_arr = vals_of(&this);
    let mut keys = array_elements(&keys_arr);
    let mut vals = array_elements(&vals_arr);
    if let Some(i) = find_key(&keys, &k) {
        keys.remove(i);
        vals.remove(i);
        array_set_elements(&keys_arr, &keys);
        array_set_elements(&vals_arr, &vals);
        set_size(&this, keys.len());
        Ok(JsValue::Boolean(true))
    } else {
        Ok(JsValue::Boolean(false))
    }
}

fn map_for_each(
    ctx: &mut EvalContext,
    this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let cb = args.get(0).cloned().unwrap_or(JsValue::Undefined);
    if !value_is_callable(&cb) {
        return Err(JErrorType::TypeError("callback is not a function".to_string()));
    }
    let keys = array_elements(&keys_of(&this));
    let vals = array_elements(&vals_of(&this));
    for (k, v) in keys.into_iter().zip(vals.into_iter()) {
        call_value(&cb, JsValue::Undefined, vec![v, k, this.clone()], ctx)?;
    }
    Ok(JsValue::Undefined)
}

fn map_keys(
    _ctx: &mut EvalContext,
    this: JsValue,
    _args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    Ok(make_array(array_elements(&keys_of(&this))))
}

fn map_values(
    _ctx: &mut EvalContext,
    this: JsValue,
    _args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    Ok(make_array(array_elements(&vals_of(&this))))
}

// ---- Set --------------------------------------------------------------------

fn set_constructor(
    ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let s = new_collection("Set");
    if let Some(init) = args.first() {
        for v in array_elements(init) {
            set_add(ctx, s.clone(), vec![v])?;
        }
    }
    Ok(s)
}

fn set_add(
    _ctx: &mut EvalContext,
    this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let v = args.get(0).cloned().unwrap_or(JsValue::Undefined);
    let keys_arr = keys_of(&this);
    let mut keys = array_elements(&keys_arr);
    if find_key(&keys, &v).is_none() {
        keys.push(v);
        array_set_elements(&keys_arr, &keys);
        set_size(&this, keys.len());
    }
    Ok(this)
}

fn set_has(
    _ctx: &mut EvalContext,
    this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let v = args.get(0).cloned().unwrap_or(JsValue::Undefined);
    Ok(JsValue::Boolean(find_key(&array_elements(&keys_of(&this)), &v).is_some()))
}

fn set_delete(
    _ctx: &mut EvalContext,
    this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let v = args.get(0).cloned().unwrap_or(JsValue::Undefined);
    let keys_arr = keys_of(&this);
    let mut keys = array_elements(&keys_arr);
    if let Some(i) = find_key(&keys, &v) {
        keys.remove(i);
        array_set_elements(&keys_arr, &keys);
        set_size(&this, keys.len());
        Ok(JsValue::Boolean(true))
    } else {
        Ok(JsValue::Boolean(false))
    }
}

fn set_for_each(
    ctx: &mut EvalContext,
    this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let cb = args.get(0).cloned().unwrap_or(JsValue::Undefined);
    if !value_is_callable(&cb) {
        return Err(JErrorType::TypeError("callback is not a function".to_string()));
    }
    for v in array_elements(&keys_of(&this)) {
        call_value(&cb, JsValue::Undefined, vec![v.clone(), v, this.clone()], ctx)?;
    }
    Ok(JsValue::Undefined)
}

fn set_values(
    _ctx: &mut EvalContext,
    this: JsValue,
    _args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    Ok(make_array(array_elements(&keys_of(&this))))
}

// ---- shared -----------------------------------------------------------------

fn coll_clear(
    _ctx: &mut EvalContext,
    this: JsValue,
    _args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    array_set_elements(&keys_of(&this), &[]);
    array_set_elements(&vals_of(&this), &[]);
    set_size(&this, 0);
    Ok(JsValue::Undefined)
}
