//! Array built-in.
//!
//! Provides the Array constructor and prototype methods. Arrays are represented
//! as Ordinary objects with indexed string keys + a `length` property + an
//! `__array__` marker (see `eval::expression`). The interpreter routes
//! `arr.method(...)` here with the array instance as `this`.

// ChittiOS-alloc-prelude
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
    array_elements, array_set_elements, call_value, is_array as value_is_array, make_array,
    value_is_callable,
};

/// Register the Array built-in with the registry.
pub fn register(registry: &mut BuiltInRegistry) {
    let array = BuiltInObject::new("Array")
        .with_constructor(array_constructor)
        .add_method("push", array_push)
        .add_method("pop", array_pop)
        .add_method("shift", array_shift)
        .add_method("unshift", array_unshift)
        .add_method("slice", array_slice)
        .add_method("splice", array_splice)
        .add_method("indexOf", array_index_of)
        .add_method("includes", array_includes)
        .add_method("forEach", array_for_each)
        .add_method("map", array_map)
        .add_method("filter", array_filter)
        .add_method("reduce", array_reduce)
        .add_method("find", array_find)
        .add_method("every", array_every)
        .add_method("some", array_some)
        .add_method("join", array_join)
        .add_method("concat", array_concat)
        .add_method("reverse", array_reverse)
        .add_method("sort", array_sort)
        .add_method("isArray", is_array);

    registry.register_object(array);
}

// ---- shared helpers ---------------------------------------------------------

fn truthy(v: &JsValue) -> bool {
    match v {
        JsValue::Undefined | JsValue::Null => false,
        JsValue::Boolean(b) => *b,
        JsValue::Number(JsNumberType::Integer(0)) => false,
        JsValue::Number(JsNumberType::Float(f)) => *f != 0.0 && !f.is_nan(),
        JsValue::Number(JsNumberType::NaN) => false,
        JsValue::String(s) => !s.is_empty(),
        _ => true,
    }
}

fn js_eq(a: &JsValue, b: &JsValue) -> bool {
    match (a, b) {
        (JsValue::Undefined, JsValue::Undefined) => true,
        (JsValue::Null, JsValue::Null) => true,
        (JsValue::Boolean(x), JsValue::Boolean(y)) => x == y,
        (JsValue::String(x), JsValue::String(y)) => x == y,
        (JsValue::Number(x), JsValue::Number(y)) => num_f64(x) == num_f64(y),
        (JsValue::Object(x), JsValue::Object(y)) => alloc::rc::Rc::ptr_eq(x, y),
        _ => false,
    }
}

fn num_f64(n: &JsNumberType) -> f64 {
    match n {
        JsNumberType::Integer(i) => *i as f64,
        JsNumberType::Float(f) => *f,
        JsNumberType::NaN => f64::NAN,
        JsNumberType::PositiveInfinity => f64::INFINITY,
        JsNumberType::NegativeInfinity => f64::NEG_INFINITY,
    }
}

/// Coerce a value to its display string (Array.join / default sort key).
fn to_str(v: &JsValue) -> String {
    match v {
        JsValue::Undefined | JsValue::Null => String::new(),
        JsValue::Boolean(b) => b.to_string(),
        JsValue::String(s) => s.clone(),
        JsValue::Number(n) => match n {
            JsNumberType::Integer(i) => i.to_string(),
            JsNumberType::Float(f) => f.to_string(),
            JsNumberType::NaN => "NaN".to_string(),
            JsNumberType::PositiveInfinity => "Infinity".to_string(),
            JsNumberType::NegativeInfinity => "-Infinity".to_string(),
        },
        JsValue::Object(_) if value_is_array(v) => {
            let elems = array_elements(v);
            let parts: Vec<String> = elems.iter().map(to_str).collect();
            parts.join(",")
        }
        JsValue::Object(_) => "[object Object]".to_string(),
        JsValue::Symbol(_) => "Symbol()".to_string(),
    }
}

fn arg(args: &[JsValue], i: usize) -> JsValue {
    args.get(i).cloned().unwrap_or(JsValue::Undefined)
}

fn to_index(v: &JsValue, len: usize) -> i64 {
    let n = match v {
        JsValue::Number(JsNumberType::Integer(i)) => *i,
        JsValue::Number(JsNumberType::Float(f)) => *f as i64,
        JsValue::Undefined => 0,
        _ => 0,
    };
    if n < 0 {
        (len as i64 + n).max(0)
    } else {
        n.min(len as i64)
    }
}

// ---- constructor + methods --------------------------------------------------

/// `Array(...)` / `new Array(...)`.
fn array_constructor(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    if args.len() == 1 {
        if let JsValue::Number(JsNumberType::Integer(n)) = &args[0] {
            let len = (*n).max(0) as usize;
            return Ok(make_array(vec![JsValue::Undefined; len]));
        }
    }
    Ok(make_array(args))
}

fn array_push(
    _ctx: &mut EvalContext,
    this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let mut elems = array_elements(&this);
    elems.extend(args);
    let len = elems.len();
    array_set_elements(&this, &elems);
    Ok(JsValue::Number(JsNumberType::Integer(len as i64)))
}

fn array_pop(
    _ctx: &mut EvalContext,
    this: JsValue,
    _args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let mut elems = array_elements(&this);
    let popped = elems.pop().unwrap_or(JsValue::Undefined);
    array_set_elements(&this, &elems);
    Ok(popped)
}

fn array_shift(
    _ctx: &mut EvalContext,
    this: JsValue,
    _args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let mut elems = array_elements(&this);
    if elems.is_empty() {
        return Ok(JsValue::Undefined);
    }
    let first = elems.remove(0);
    array_set_elements(&this, &elems);
    Ok(first)
}

fn array_unshift(
    _ctx: &mut EvalContext,
    this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let mut elems = array_elements(&this);
    for (i, v) in args.into_iter().enumerate() {
        elems.insert(i, v);
    }
    let len = elems.len();
    array_set_elements(&this, &elems);
    Ok(JsValue::Number(JsNumberType::Integer(len as i64)))
}

fn array_slice(
    _ctx: &mut EvalContext,
    this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let elems = array_elements(&this);
    let len = elems.len();
    let start = to_index(&arg(&args, 0), len) as usize;
    let end = if args.len() > 1 {
        to_index(&args[1], len) as usize
    } else {
        len
    };
    let slice = if start < end { elems[start..end].to_vec() } else { Vec::new() };
    Ok(make_array(slice))
}

fn array_splice(
    _ctx: &mut EvalContext,
    this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let mut elems = array_elements(&this);
    let len = elems.len();
    let start = to_index(&arg(&args, 0), len) as usize;
    let delete = if args.len() > 1 {
        match &args[1] {
            JsValue::Number(JsNumberType::Integer(n)) => (*n).max(0) as usize,
            JsValue::Number(JsNumberType::Float(f)) => (*f).max(0.0) as usize,
            _ => 0,
        }
    } else {
        len.saturating_sub(start)
    };
    let end = (start + delete).min(len);
    let removed: Vec<JsValue> = if start < end { elems.drain(start..end).collect() } else { Vec::new() };
    for (i, v) in args.into_iter().skip(2).enumerate() {
        elems.insert(start + i, v);
    }
    array_set_elements(&this, &elems);
    Ok(make_array(removed))
}

fn array_index_of(
    _ctx: &mut EvalContext,
    this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let needle = arg(&args, 0);
    let elems = array_elements(&this);
    for (i, e) in elems.iter().enumerate() {
        if js_eq(e, &needle) {
            return Ok(JsValue::Number(JsNumberType::Integer(i as i64)));
        }
    }
    Ok(JsValue::Number(JsNumberType::Integer(-1)))
}

fn array_includes(
    _ctx: &mut EvalContext,
    this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let needle = arg(&args, 0);
    let found = array_elements(&this).iter().any(|e| js_eq(e, &needle));
    Ok(JsValue::Boolean(found))
}

fn require_callback(v: &JsValue) -> Result<JsValue, JErrorType> {
    if value_is_callable(v) {
        Ok(v.clone())
    } else {
        Err(JErrorType::TypeError("callback is not a function".to_string()))
    }
}

fn array_for_each(
    ctx: &mut EvalContext,
    this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let cb = require_callback(&arg(&args, 0))?;
    let elems = array_elements(&this);
    for (i, e) in elems.into_iter().enumerate() {
        call_value(&cb, JsValue::Undefined, vec![e, JsValue::Number(JsNumberType::Integer(i as i64)), this.clone()], ctx)?;
    }
    Ok(JsValue::Undefined)
}

fn array_map(
    ctx: &mut EvalContext,
    this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let cb = require_callback(&arg(&args, 0))?;
    let elems = array_elements(&this);
    let mut out = Vec::with_capacity(elems.len());
    for (i, e) in elems.into_iter().enumerate() {
        out.push(call_value(&cb, JsValue::Undefined, vec![e, JsValue::Number(JsNumberType::Integer(i as i64)), this.clone()], ctx)?);
    }
    Ok(make_array(out))
}

fn array_filter(
    ctx: &mut EvalContext,
    this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let cb = require_callback(&arg(&args, 0))?;
    let elems = array_elements(&this);
    let mut out = Vec::new();
    for (i, e) in elems.into_iter().enumerate() {
        let keep = call_value(&cb, JsValue::Undefined, vec![e.clone(), JsValue::Number(JsNumberType::Integer(i as i64)), this.clone()], ctx)?;
        if truthy(&keep) {
            out.push(e);
        }
    }
    Ok(make_array(out))
}

fn array_reduce(
    ctx: &mut EvalContext,
    this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let cb = require_callback(&arg(&args, 0))?;
    let elems = array_elements(&this);
    let mut iter = elems.into_iter().enumerate();
    let mut acc = if args.len() > 1 {
        args[1].clone()
    } else {
        match iter.next() {
            Some((_, v)) => v,
            None => return Err(JErrorType::TypeError("Reduce of empty array with no initial value".to_string())),
        }
    };
    for (i, e) in iter {
        acc = call_value(&cb, JsValue::Undefined, vec![acc, e, JsValue::Number(JsNumberType::Integer(i as i64)), this.clone()], ctx)?;
    }
    Ok(acc)
}

fn array_find(
    ctx: &mut EvalContext,
    this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let cb = require_callback(&arg(&args, 0))?;
    let elems = array_elements(&this);
    for (i, e) in elems.into_iter().enumerate() {
        let hit = call_value(&cb, JsValue::Undefined, vec![e.clone(), JsValue::Number(JsNumberType::Integer(i as i64)), this.clone()], ctx)?;
        if truthy(&hit) {
            return Ok(e);
        }
    }
    Ok(JsValue::Undefined)
}

fn array_every(
    ctx: &mut EvalContext,
    this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let cb = require_callback(&arg(&args, 0))?;
    let elems = array_elements(&this);
    for (i, e) in elems.into_iter().enumerate() {
        let r = call_value(&cb, JsValue::Undefined, vec![e, JsValue::Number(JsNumberType::Integer(i as i64)), this.clone()], ctx)?;
        if !truthy(&r) {
            return Ok(JsValue::Boolean(false));
        }
    }
    Ok(JsValue::Boolean(true))
}

fn array_some(
    ctx: &mut EvalContext,
    this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let cb = require_callback(&arg(&args, 0))?;
    let elems = array_elements(&this);
    for (i, e) in elems.into_iter().enumerate() {
        let r = call_value(&cb, JsValue::Undefined, vec![e, JsValue::Number(JsNumberType::Integer(i as i64)), this.clone()], ctx)?;
        if truthy(&r) {
            return Ok(JsValue::Boolean(true));
        }
    }
    Ok(JsValue::Boolean(false))
}

fn array_join(
    _ctx: &mut EvalContext,
    this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let sep = match arg(&args, 0) {
        JsValue::Undefined => ",".to_string(),
        other => to_str(&other),
    };
    let parts: Vec<String> = array_elements(&this)
        .iter()
        .map(|e| match e {
            JsValue::Undefined | JsValue::Null => String::new(),
            _ => to_str(e),
        })
        .collect();
    Ok(JsValue::String(parts.join(&sep)))
}

fn array_concat(
    _ctx: &mut EvalContext,
    this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let mut out = array_elements(&this);
    for a in args {
        if value_is_array(&a) {
            out.extend(array_elements(&a));
        } else {
            out.push(a);
        }
    }
    Ok(make_array(out))
}

fn array_reverse(
    _ctx: &mut EvalContext,
    this: JsValue,
    _args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let mut elems = array_elements(&this);
    elems.reverse();
    array_set_elements(&this, &elems);
    Ok(this)
}

fn array_sort(
    ctx: &mut EvalContext,
    this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let mut elems = array_elements(&this);
    let cmp = arg(&args, 0);
    if value_is_callable(&cmp) {
        // Insertion sort so we can propagate comparator errors + keep it simple.
        let n = elems.len();
        for i in 1..n {
            let mut j = i;
            while j > 0 {
                let r = call_value(&cmp, JsValue::Undefined, vec![elems[j - 1].clone(), elems[j].clone()], ctx)?;
                let gt = match r {
                    JsValue::Number(ref nn) => num_f64(nn) > 0.0,
                    _ => false,
                };
                if gt {
                    elems.swap(j - 1, j);
                    j -= 1;
                } else {
                    break;
                }
            }
        }
    } else {
        elems.sort_by(|a, b| to_str(a).cmp(&to_str(b)));
    }
    array_set_elements(&this, &elems);
    Ok(this)
}

/// `Array.isArray(x)` (static).
fn is_array(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    Ok(JsValue::Boolean(value_is_array(&arg(&args, 0))))
}
