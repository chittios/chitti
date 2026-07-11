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
        JsValue::Object(_) => "[object Object]".to_string(),
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
