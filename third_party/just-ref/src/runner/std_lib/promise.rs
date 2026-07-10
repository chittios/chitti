//! ChittiOS: a pragmatic `Promise` built-in.
//!
//! `just` has no event loop, so promises settle **synchronously**: a `.then`
//! reaction on an already-settled promise runs immediately, and `resolve`/
//! `reject` run queued reactions inline. Value flow through `.then` chains,
//! `.catch`, `Promise.resolve/reject/all` is correct; strict microtask ordering
//! relative to surrounding synchronous code is not modelled (that needs the
//! kernel's `events::EVENT_LOOP` microtask lane — future work, and the
//! prerequisite for real `async`/`await`).
//!
//! A promise is an object tagged `__builtin_name__ = "Promise"` with
//! `__state__` ("pending"|"fulfilled"|"rejected"), `__value__`, and a
//! `__reactions__` array of `[onFulfilled, onRejected, resultPromise]` triples.
//! `resolve`/`reject` passed to an executor are callable settler objects
//! (`__promise_op__` + `__promise_target__`), dispatched in `call_function_object`.

#[allow(unused_imports)]
use num_traits::Float;
use alloc::borrow::ToOwned;
use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use alloc::vec;
use alloc::format;
use crate::runner::ds::error::JErrorType;
use crate::runner::ds::value::JsValue;
use crate::runner::plugin::registry::BuiltInRegistry;
use crate::runner::plugin::types::{BuiltInObject, EvalContext};
use crate::runner::eval::expression::{
    array_elements, array_set_elements, call_value, get_own_prop_value, make_array, make_object,
    set_own_prop, value_is_callable,
};

pub fn register(registry: &mut BuiltInRegistry) {
    let promise = BuiltInObject::new("Promise")
        .with_constructor(promise_constructor)
        .add_method("then", promise_then)
        .add_method("catch", promise_catch)
        .add_method("finally", promise_finally)
        .add_method("resolve", static_resolve) // Promise.resolve(v)
        .add_method("reject", static_reject) // Promise.reject(e)
        .add_method("all", static_all); // Promise.all([...])
    registry.register_object(promise);
}

pub fn is_promise(v: &JsValue) -> bool {
    matches!(get_own_prop_value(v, "__builtin_name__"), Some(JsValue::String(ref s)) if s == "Promise")
}

/// Wrap a value in an already-fulfilled Promise (used by `async function`
/// return). If it's already a promise, return it unchanged.
pub fn resolve_value(v: JsValue) -> JsValue {
    if is_promise(&v) {
        return v;
    }
    let p = make_promise();
    set_own_prop(&p, "__state__", JsValue::String("fulfilled".to_string()), false);
    set_own_prop(&p, "__value__", v, false);
    p
}

/// `await v`: synchronously extract a settled promise's value (or return a
/// non-promise unchanged). A rejection becomes a thrown error.
pub fn await_value(v: JsValue) -> Result<JsValue, JErrorType> {
    if !is_promise(&v) {
        return Ok(v);
    }
    match state_of(&v).as_str() {
        "fulfilled" => Ok(get_own_prop_value(&v, "__value__").unwrap_or(JsValue::Undefined)),
        "rejected" => Err(JErrorType::TypeError(format!(
            "Uncaught (in promise) {:?}",
            get_own_prop_value(&v, "__value__").unwrap_or(JsValue::Undefined)
        ))),
        // Pending never settles without an event loop.
        _ => Ok(JsValue::Undefined),
    }
}

fn make_promise() -> JsValue {
    let p = make_object(vec![]);
    set_own_prop(&p, "__builtin_name__", JsValue::String("Promise".to_string()), false);
    set_own_prop(&p, "__state__", JsValue::String("pending".to_string()), false);
    set_own_prop(&p, "__value__", JsValue::Undefined, false);
    set_own_prop(&p, "__reactions__", make_array(vec![]), false);
    p
}

fn state_of(p: &JsValue) -> String {
    match get_own_prop_value(p, "__state__") {
        Some(JsValue::String(s)) => s,
        _ => "pending".to_string(),
    }
}

fn make_settler(promise: &JsValue, op: &str) -> JsValue {
    let s = make_object(vec![]);
    set_own_prop(&s, "__simple_function__", JsValue::Boolean(true), false);
    set_own_prop(&s, "__promise_op__", JsValue::String(op.to_string()), false);
    set_own_prop(&s, "__promise_target__", promise.clone(), false);
    s
}

/// Convert a thrown JErrorType into a JS rejection value.
fn err_to_value(e: &JErrorType) -> JsValue {
    JsValue::String(format!("{:?}", e))
}

/// Settle `promise` (fulfil or reject) with `value`, running queued reactions.
/// If `value` is itself a promise on fulfilment, adopt its eventual state.
pub fn settle(promise: &JsValue, is_reject: bool, value: JsValue, ctx: &mut EvalContext) {
    if state_of(promise) != "pending" {
        return; // already settled
    }
    // Resolving with a thenable → adopt its state instead of fulfilling with it.
    if !is_reject && is_promise(&value) {
        let res = make_settler(promise, "resolve");
        let rej = make_settler(promise, "reject");
        then_internal(&value, &res, &rej, ctx);
        return;
    }
    set_own_prop(
        promise,
        "__state__",
        JsValue::String(if is_reject { "rejected" } else { "fulfilled" }.to_string()),
        false,
    );
    set_own_prop(promise, "__value__", value.clone(), false);

    // Drain queued reactions.
    let reactions = get_own_prop_value(promise, "__reactions__").unwrap_or_else(|| make_array(vec![]));
    let list = array_elements(&reactions);
    array_set_elements(&reactions, &[]);
    for r in list {
        run_reaction(&r, is_reject, value.clone(), ctx);
    }
}

/// A reaction is `[onFulfilled, onRejected, resultPromise]`.
fn run_reaction(reaction: &JsValue, rejected: bool, value: JsValue, ctx: &mut EvalContext) {
    let parts = array_elements(reaction);
    let on_f = parts.get(0).cloned().unwrap_or(JsValue::Undefined);
    let on_r = parts.get(1).cloned().unwrap_or(JsValue::Undefined);
    let result = parts.get(2).cloned().unwrap_or(JsValue::Undefined);

    let handler = if rejected { on_r } else { on_f };
    if value_is_callable(&handler) {
        match call_value(&handler, JsValue::Undefined, vec![value], ctx) {
            Ok(r) => settle(&result, false, r, ctx),
            Err(e) => settle(&result, true, err_to_value(&e), ctx),
        }
    } else {
        // No handler → pass the settlement straight through.
        settle(&result, rejected, value, ctx);
    }
}

/// `p.then(onF, onR)` core: returns a new result promise.
fn then_internal(p: &JsValue, on_f: &JsValue, on_r: &JsValue, ctx: &mut EvalContext) -> JsValue {
    let result = make_promise();
    let reaction = make_array(vec![on_f.clone(), on_r.clone(), result.clone()]);
    match state_of(p).as_str() {
        "pending" => {
            let reactions = get_own_prop_value(p, "__reactions__").unwrap_or_else(|| make_array(vec![]));
            let mut list = array_elements(&reactions);
            list.push(reaction);
            array_set_elements(&reactions, &list);
        }
        st => {
            let rejected = st == "rejected";
            let value = get_own_prop_value(p, "__value__").unwrap_or(JsValue::Undefined);
            run_reaction(&reaction, rejected, value, ctx);
        }
    }
    result
}

// ---- constructor + methods --------------------------------------------------

fn promise_constructor(
    ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let p = make_promise();
    if let Some(executor) = args.first() {
        if value_is_callable(executor) {
            let resolve = make_settler(&p, "resolve");
            let reject = make_settler(&p, "reject");
            if let Err(e) = call_value(executor, JsValue::Undefined, vec![resolve, reject], ctx) {
                settle(&p, true, err_to_value(&e), ctx);
            }
        }
    }
    Ok(p)
}

fn promise_then(
    ctx: &mut EvalContext,
    this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let on_f = args.get(0).cloned().unwrap_or(JsValue::Undefined);
    let on_r = args.get(1).cloned().unwrap_or(JsValue::Undefined);
    Ok(then_internal(&this, &on_f, &on_r, ctx))
}

fn promise_catch(
    ctx: &mut EvalContext,
    this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let on_r = args.get(0).cloned().unwrap_or(JsValue::Undefined);
    Ok(then_internal(&this, &JsValue::Undefined, &on_r, ctx))
}

fn promise_finally(
    ctx: &mut EvalContext,
    this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    // Run the callback (if any) and pass the settlement through unchanged.
    let cb = args.get(0).cloned().unwrap_or(JsValue::Undefined);
    if value_is_callable(&cb) {
        let _ = call_value(&cb, JsValue::Undefined, vec![], ctx);
    }
    Ok(then_internal(&this, &JsValue::Undefined, &JsValue::Undefined, ctx))
}

/// `Promise.resolve(v)` (static; `this` is the Promise sentinel).
fn static_resolve(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let v = args.into_iter().next().unwrap_or(JsValue::Undefined);
    if is_promise(&v) {
        return Ok(v);
    }
    let p = make_promise();
    set_own_prop(&p, "__state__", JsValue::String("fulfilled".to_string()), false);
    set_own_prop(&p, "__value__", v, false);
    Ok(p)
}

fn static_reject(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let v = args.into_iter().next().unwrap_or(JsValue::Undefined);
    let p = make_promise();
    set_own_prop(&p, "__state__", JsValue::String("rejected".to_string()), false);
    set_own_prop(&p, "__value__", v, false);
    Ok(p)
}

/// `Promise.all([...])` — synchronous: fulfil with the array of results, or
/// reject with the first rejection.
fn static_all(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let arr = args.into_iter().next().unwrap_or(JsValue::Undefined);
    let items = array_elements(&arr);
    let mut results = Vec::with_capacity(items.len());
    for it in items {
        if is_promise(&it) {
            match state_of(&it).as_str() {
                "fulfilled" => results.push(get_own_prop_value(&it, "__value__").unwrap_or(JsValue::Undefined)),
                "rejected" => {
                    let p = make_promise();
                    set_own_prop(&p, "__state__", JsValue::String("rejected".to_string()), false);
                    set_own_prop(&p, "__value__", get_own_prop_value(&it, "__value__").unwrap_or(JsValue::Undefined), false);
                    return Ok(p);
                }
                _ => results.push(JsValue::Undefined), // pending never settles (no loop)
            }
        } else {
            results.push(it);
        }
    }
    let p = make_promise();
    set_own_prop(&p, "__state__", JsValue::String("fulfilled".to_string()), false);
    set_own_prop(&p, "__value__", make_array(results), false);
    Ok(p)
}
