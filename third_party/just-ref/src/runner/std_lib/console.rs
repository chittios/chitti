//! Console built-in object.
//!
//! Provides console.log, console.error, console.warn, and console.info methods.

// ChittiOS-alloc-prelude
#[allow(unused_imports)]
use num_traits::Float;
use alloc::borrow::ToOwned;
use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use crate::runner::ds::error::JErrorType;
use crate::runner::ds::value::{JsValue, JsNumberType};
use crate::runner::plugin::registry::BuiltInRegistry;
use crate::runner::plugin::types::{BuiltInObject, EvalContext};
use spin::Mutex;

/// ChittiOS: console output is buffered in a global sink so the embedding host
/// (the kernel browser `dom.log`) can drain it. On the std build we also mirror
/// to stdout/stderr for the webcompat harness.
static CONSOLE_LOG: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// Drain everything written to the console since the last drain.
pub fn drain_console_log() -> Vec<String> {
    core::mem::take(&mut *CONSOLE_LOG.lock())
}

fn emit(line: String, is_err: bool) {
    #[cfg(feature = "std")]
    {
        if is_err { std::eprintln!("{}", line); } else { std::println!("{}", line); }
    }
    let _ = is_err;
    CONSOLE_LOG.lock().push(line);
}

/// Register the console object with the registry.
pub fn register(registry: &mut BuiltInRegistry) {
    let console = BuiltInObject::new("console")
        .with_no_prototype()
        .add_method("log", console_log)
        .add_method("error", console_error)
        .add_method("warn", console_warn)
        .add_method("info", console_info);

    registry.register_object(console);
}

/// Format a JsValue for console output.
fn format_value(value: &JsValue) -> String {
    match value {
        JsValue::Undefined => "undefined".to_string(),
        JsValue::Null => "null".to_string(),
        JsValue::Boolean(b) => b.to_string(),
        JsValue::Number(n) => match n {
            JsNumberType::Integer(i) => i.to_string(),
            JsNumberType::Float(f) => {
                if f.fract() == 0.0 && f.abs() < 1e15 {
                    format!("{:.0}", f)
                } else {
                    f.to_string()
                }
            }
            JsNumberType::NaN => "NaN".to_string(),
            JsNumberType::PositiveInfinity => "Infinity".to_string(),
            JsNumberType::NegativeInfinity => "-Infinity".to_string(),
        },
        JsValue::String(s) => s.clone(),
        JsValue::BigInt(b) => format!("{}n", b),
        JsValue::Symbol(s) => s.to_string(),
        // A bare "[object Object]" is what a page's own error reporting turns
        // into, so a thrown error logged by a framework said nothing at all.
        // Errors print name + message; arrays and plain objects print one level
        // of contents, bounded.
        JsValue::Object(_) => format_object(value, 0),
    }
}

/// Longest object rendering console output will produce, per value.
const MAX_ENTRIES: usize = 8;

/// Render an object one level deep (two for the top-level call). Bounded in
/// both breadth and depth — a console formatter must never be the thing that
/// hangs on a cyclic structure.
fn format_object(value: &JsValue, depth: usize) -> String {
    use crate::runner::eval::expression::{get_own_prop_value, is_array, own_enumerable_string_keys};

    // Error-shaped: `name` + `message` is how every page reports a failure.
    if let (Some(JsValue::String(name)), Some(msg)) = (
        get_own_prop_value(value, "name"),
        get_own_prop_value(value, "message"),
    ) {
        let m = format_value(&msg);
        return if m.is_empty() { name } else { alloc::format!("{name}: {m}") };
    }
    if depth >= 2 {
        return "[object Object]".to_string();
    }
    if is_array(value) {
        let len = match get_own_prop_value(value, "length") {
            Some(JsValue::Number(JsNumberType::Integer(n))) if n >= 0 => n as usize,
            _ => 0,
        };
        let shown = len.min(MAX_ENTRIES);
        let mut parts: Vec<String> = Vec::new();
        for i in 0..shown {
            let v = get_own_prop_value(value, &i.to_string()).unwrap_or(JsValue::Undefined);
            parts.push(nested(&v, depth));
        }
        if len > shown {
            parts.push(alloc::format!("… {} more", len - shown));
        }
        return alloc::format!("[{}]", parts.join(", "));
    }
    let keys = own_enumerable_string_keys(value);
    if keys.is_empty() {
        return "{}".to_string();
    }
    let mut parts: Vec<String> = Vec::new();
    for k in keys.iter().take(MAX_ENTRIES) {
        let v = get_own_prop_value(value, k).unwrap_or(JsValue::Undefined);
        parts.push(alloc::format!("{k}: {}", nested(&v, depth)));
    }
    if keys.len() > MAX_ENTRIES {
        parts.push(alloc::format!("… {} more", keys.len() - MAX_ENTRIES));
    }
    alloc::format!("{{ {} }}", parts.join(", "))
}

/// A value nested inside an object: strings are quoted (so `{ a: "1" }` and
/// `{ a: 1 }` are distinguishable), objects recurse one more level.
fn nested(v: &JsValue, depth: usize) -> String {
    match v {
        JsValue::String(s) => alloc::format!("\"{s}\""),
        JsValue::Object(_) => format_object(v, depth + 1),
        other => format_value(other),
    }
}

/// Format all arguments for console output.
fn format_args(args: &[JsValue]) -> String {
    args.iter()
        .map(format_value)
        .collect::<Vec<_>>()
        .join(" ")
}

/// console.log - Log to stdout.
fn console_log(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    emit(format_args(&args), false);
    Ok(JsValue::Undefined)
}

/// console.error - Log to stderr.
fn console_error(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    emit(format_args(&args), true);
    Ok(JsValue::Undefined)
}

/// console.warn - Log warning to stderr.
fn console_warn(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    emit(format!("Warning: {}", format_args(&args)), true);
    Ok(JsValue::Undefined)
}

/// console.info - Log info to stdout (same as log).
fn console_info(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    emit(format_args(&args), false);
    Ok(JsValue::Undefined)
}
