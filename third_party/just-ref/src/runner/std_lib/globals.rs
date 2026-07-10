//! ChittiOS: global values + functions that live directly in scope (not on an
//! object) — `undefined`, `NaN`, `Infinity`, `globalThis`, and the callable
//! globals `isNaN`, `isFinite`, `parseInt`, `parseFloat`, `Boolean`, `Symbol`.
//! Installed by `EvalContext::install_core_builtins` alongside the core object
//! registry, so they work in the kernel browser and the webcompat harness.

#[allow(unused_imports)]
use num_traits::Float;
use alloc::borrow::ToOwned;
use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use alloc::vec;
use crate::runner::ds::error::JErrorType;
use crate::runner::ds::symbol::SymbolData;
use crate::runner::ds::value::{JsNumberType, JsValue};
use crate::runner::eval::expression::{make_object, set_own_prop};
use crate::runner::plugin::resolver::PluginResolver;
use crate::runner::plugin::types::EvalContext;

pub struct GlobalsResolver;

fn is_global_fn(name: &str) -> bool {
    matches!(name, "isNaN" | "isFinite" | "parseInt" | "parseFloat" | "Boolean" | "Symbol")
}

/// Coerce a value to an f64 (JS `Number(x)` semantics, loosely).
fn to_f64(v: &JsValue) -> f64 {
    match v {
        JsValue::Number(JsNumberType::Integer(i)) => *i as f64,
        JsValue::Number(JsNumberType::Float(f)) => *f,
        JsValue::Number(JsNumberType::NaN) => f64::NAN,
        JsValue::Number(JsNumberType::PositiveInfinity) => f64::INFINITY,
        JsValue::Number(JsNumberType::NegativeInfinity) => f64::NEG_INFINITY,
        JsValue::Boolean(true) => 1.0,
        JsValue::Boolean(false) => 0.0,
        JsValue::Null => 0.0,
        JsValue::Undefined => f64::NAN,
        JsValue::String(s) => {
            let t = s.trim();
            if t.is_empty() {
                0.0
            } else {
                t.parse().unwrap_or(f64::NAN)
            }
        }
        _ => f64::NAN,
    }
}

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

fn from_f64(f: f64) -> JsValue {
    if f.is_nan() {
        JsValue::Number(JsNumberType::NaN)
    } else if f == f64::INFINITY {
        JsValue::Number(JsNumberType::PositiveInfinity)
    } else if f == f64::NEG_INFINITY {
        JsValue::Number(JsNumberType::NegativeInfinity)
    } else if f.fract() == 0.0 && f.abs() < 9.007e15 {
        JsValue::Number(JsNumberType::Integer(f as i64))
    } else {
        JsValue::Number(JsNumberType::Float(f))
    }
}

fn parse_int(v: &JsValue, radix: &JsValue) -> JsValue {
    let s = match v {
        JsValue::String(s) => s.trim().to_string(),
        other => {
            // Number → its integer part.
            let f = to_f64(other);
            if f.is_nan() {
                return JsValue::Number(JsNumberType::NaN);
            }
            return from_f64(f.trunc());
        }
    };
    let mut chars = s.chars().peekable();
    let mut neg = false;
    match chars.peek() {
        Some('+') => { chars.next(); }
        Some('-') => { neg = true; chars.next(); }
        _ => {}
    }
    let mut radix = match radix {
        JsValue::Number(JsNumberType::Integer(n)) if *n != 0 => *n as u32,
        JsValue::Number(JsNumberType::Float(f)) if *f != 0.0 => *f as u32,
        _ => 10,
    };
    let rest: String = chars.collect();
    let mut rest = rest.as_str();
    if (radix == 16 || radix == 10) && (rest.starts_with("0x") || rest.starts_with("0X")) {
        radix = 16;
        rest = &rest[2..];
    }
    let mut acc: i64 = 0;
    let mut any = false;
    for c in rest.chars() {
        match c.to_digit(radix) {
            Some(d) => {
                acc = acc.saturating_mul(radix as i64).saturating_add(d as i64);
                any = true;
            }
            None => break,
        }
    }
    if !any {
        return JsValue::Number(JsNumberType::NaN);
    }
    JsValue::Number(JsNumberType::Integer(if neg { -acc } else { acc }))
}

fn parse_float(v: &JsValue) -> JsValue {
    let s = match v {
        JsValue::String(s) => s.trim().to_string(),
        other => return from_f64(to_f64(other)),
    };
    // Longest leading parseable float prefix.
    let bytes = s.as_bytes();
    let mut end = 0;
    let mut seen_dot = false;
    let mut seen_e = false;
    for (i, &b) in bytes.iter().enumerate() {
        let c = b as char;
        if c.is_ascii_digit() {
            end = i + 1;
        } else if (c == '+' || c == '-') && (i == 0 || bytes[i - 1] == b'e' || bytes[i - 1] == b'E') {
            // sign at start or after exponent
        } else if c == '.' && !seen_dot && !seen_e {
            seen_dot = true;
        } else if (c == 'e' || c == 'E') && !seen_e && end > 0 {
            seen_e = true;
        } else {
            break;
        }
    }
    if end == 0 {
        return JsValue::Number(JsNumberType::NaN);
    }
    match s[..s.len().min(if seen_e { s.len() } else { end.max(1) })].parse::<f64>() {
        Ok(f) => from_f64(f),
        Err(_) => match s[..end].parse::<f64>() {
            Ok(f) => from_f64(f),
            Err(_) => JsValue::Number(JsNumberType::NaN),
        },
    }
}

impl PluginResolver for GlobalsResolver {
    fn has_binding(&self, name: &str) -> bool {
        is_global_fn(name) || matches!(name, "undefined" | "NaN" | "Infinity" | "globalThis")
    }

    fn resolve(&self, name: &str, _ctx: &mut EvalContext) -> Result<JsValue, JErrorType> {
        Ok(match name {
            "undefined" => JsValue::Undefined,
            "NaN" => JsValue::Number(JsNumberType::NaN),
            "Infinity" => JsValue::Number(JsNumberType::PositiveInfinity),
            "globalThis" => make_object(vec![]),
            n if is_global_fn(n) => {
                let f = make_object(vec![]);
                set_own_prop(&f, "__builtin_name__", JsValue::String(n.to_string()), false);
                f
            }
            _ => JsValue::Undefined,
        })
    }

    fn call_method(
        &self,
        _object_name: &str,
        _method_name: &str,
        _ctx: &mut EvalContext,
        _this: JsValue,
        _args: Vec<JsValue>,
    ) -> Option<Result<JsValue, JErrorType>> {
        None
    }

    fn call_constructor(
        &self,
        object_name: &str,
        _ctx: &mut EvalContext,
        args: Vec<JsValue>,
    ) -> Option<Result<JsValue, JErrorType>> {
        if !is_global_fn(object_name) {
            return None;
        }
        let a0 = args.get(0).cloned().unwrap_or(JsValue::Undefined);
        let res = match object_name {
            "isNaN" => JsValue::Boolean(to_f64(&a0).is_nan()),
            "isFinite" => {
                let f = to_f64(&a0);
                JsValue::Boolean(!f.is_nan() && f.is_finite())
            }
            "parseInt" => parse_int(&a0, args.get(1).unwrap_or(&JsValue::Undefined)),
            "parseFloat" => parse_float(&a0),
            "Boolean" => JsValue::Boolean(truthy(&a0)),
            "Symbol" => {
                let desc = match &a0 {
                    JsValue::Undefined => String::new(),
                    JsValue::String(s) => s.clone(),
                    other => other.to_string(),
                };
                JsValue::Symbol(SymbolData::new(desc))
            }
            _ => JsValue::Undefined,
        };
        Some(Ok(res))
    }

    fn name(&self) -> &str {
        "chitti_globals"
    }
}
