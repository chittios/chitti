//! Number built-in.
//!
//! Provides Number constructor and methods.

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

/// Register the Number built-in with the registry.
pub fn register(registry: &mut BuiltInRegistry) {
    let number = BuiltInObject::new("Number")
        .with_constructor(number_constructor)
        .add_property("MAX_VALUE", JsValue::Number(JsNumberType::Float(f64::MAX)))
        .add_property("MIN_VALUE", JsValue::Number(JsNumberType::Float(f64::MIN_POSITIVE)))
        .add_property("POSITIVE_INFINITY", JsValue::Number(JsNumberType::PositiveInfinity))
        .add_property("NEGATIVE_INFINITY", JsValue::Number(JsNumberType::NegativeInfinity))
        .add_property("NaN", JsValue::Number(JsNumberType::NaN))
        .add_property("MAX_SAFE_INTEGER", JsValue::Number(JsNumberType::Integer(9007199254740991)))
        .add_property("MIN_SAFE_INTEGER", JsValue::Number(JsNumberType::Integer(-9007199254740991)))
        .add_property("EPSILON", JsValue::Number(JsNumberType::Float(f64::EPSILON)))
        .add_method("isNaN", number_is_nan)
        .add_method("isFinite", number_is_finite)
        .add_method("isInteger", number_is_integer)
        .add_method("isSafeInteger", number_is_safe_integer)
        .add_method("parseFloat", number_parse_float)
        .add_method("parseInt", number_parse_int)
        .add_method("toString", number_to_string)
        .add_method("toFixed", number_to_fixed)
        .add_method("toExponential", number_to_exponential)
        .add_method("toPrecision", number_to_precision);

    registry.register_object(number);
}

/// Number constructor.
fn number_constructor(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    if args.is_empty() {
        return Ok(JsValue::Number(JsNumberType::Integer(0)));
    }

    to_number(&args[0])
}

/// Number.isNaN - Check if value is NaN (strict).
fn number_is_nan(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    if args.is_empty() {
        return Ok(JsValue::Boolean(false));
    }

    let result = match &args[0] {
        JsValue::Number(JsNumberType::NaN) => true,
        JsValue::Number(JsNumberType::Float(f)) if f.is_nan() => true,
        _ => false,
    };

    Ok(JsValue::Boolean(result))
}

/// Number.isFinite - Check if value is finite (strict).
fn number_is_finite(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    if args.is_empty() {
        return Ok(JsValue::Boolean(false));
    }

    let result = match &args[0] {
        JsValue::Number(JsNumberType::Integer(_)) => true,
        JsValue::Number(JsNumberType::Float(f)) => f.is_finite(),
        JsValue::Number(JsNumberType::NaN) => false,
        JsValue::Number(JsNumberType::PositiveInfinity) => false,
        JsValue::Number(JsNumberType::NegativeInfinity) => false,
        _ => false,
    };

    Ok(JsValue::Boolean(result))
}

/// Number.isInteger - Check if value is an integer.
fn number_is_integer(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    if args.is_empty() {
        return Ok(JsValue::Boolean(false));
    }

    let result = match &args[0] {
        JsValue::Number(JsNumberType::Integer(_)) => true,
        JsValue::Number(JsNumberType::Float(f)) => f.fract() == 0.0 && f.is_finite(),
        _ => false,
    };

    Ok(JsValue::Boolean(result))
}

/// Number.isSafeInteger - Check if value is a safe integer.
fn number_is_safe_integer(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    if args.is_empty() {
        return Ok(JsValue::Boolean(false));
    }

    const MAX_SAFE: i64 = 9007199254740991;
    const MIN_SAFE: i64 = -9007199254740991;

    let result = match &args[0] {
        JsValue::Number(JsNumberType::Integer(i)) => *i >= MIN_SAFE && *i <= MAX_SAFE,
        JsValue::Number(JsNumberType::Float(f)) => {
            f.fract() == 0.0 && *f >= MIN_SAFE as f64 && *f <= MAX_SAFE as f64
        }
        _ => false,
    };

    Ok(JsValue::Boolean(result))
}

/// Number.parseFloat - Parse string to float (global parseFloat).
fn number_parse_float(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    if args.is_empty() {
        return Ok(JsValue::Number(JsNumberType::NaN));
    }

    let s = match &args[0] {
        JsValue::String(s) => s.trim(),
        _ => return Ok(JsValue::Number(JsNumberType::NaN)),
    };

    if s.is_empty() {
        return Ok(JsValue::Number(JsNumberType::NaN));
    }

    // Parse as much as possible as a number
    match s.parse::<f64>() {
        Ok(f) => {
            if f.fract() == 0.0 && f.abs() < i64::MAX as f64 {
                Ok(JsValue::Number(JsNumberType::Integer(f as i64)))
            } else {
                Ok(JsValue::Number(JsNumberType::Float(f)))
            }
        }
        Err(_) => Ok(JsValue::Number(JsNumberType::NaN)),
    }
}

/// Number.parseInt - Parse string to integer (global parseInt).
fn number_parse_int(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    if args.is_empty() {
        return Ok(JsValue::Number(JsNumberType::NaN));
    }

    let s = match &args[0] {
        JsValue::String(s) => s.trim(),
        JsValue::Number(JsNumberType::Integer(i)) => return Ok(JsValue::Number(JsNumberType::Integer(*i))),
        JsValue::Number(JsNumberType::Float(f)) => return Ok(JsValue::Number(JsNumberType::Integer(*f as i64))),
        _ => return Ok(JsValue::Number(JsNumberType::NaN)),
    };

    if s.is_empty() {
        return Ok(JsValue::Number(JsNumberType::NaN));
    }

    let radix = if args.len() > 1 {
        match &args[1] {
            JsValue::Number(JsNumberType::Integer(r)) => *r as u32,
            JsValue::Number(JsNumberType::Float(f)) => *f as u32,
            _ => 10,
        }
    } else {
        10
    };

    if radix < 2 || radix > 36 {
        return Ok(JsValue::Number(JsNumberType::NaN));
    }

    match i64::from_str_radix(s, radix) {
        Ok(i) => Ok(JsValue::Number(JsNumberType::Integer(i))),
        Err(_) => Ok(JsValue::Number(JsNumberType::NaN)),
    }
}

/// Number.prototype.toString
fn number_to_string(
    _ctx: &mut EvalContext,
    this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let radix = if args.is_empty() {
        10
    } else {
        match &args[0] {
            JsValue::Number(JsNumberType::Integer(r)) => *r as u32,
            JsValue::Number(JsNumberType::Float(f)) => *f as u32,
            _ => 10,
        }
    };

    if radix < 2 || radix > 36 {
        return Err(JErrorType::RangeError("radix must be between 2 and 36".to_string()));
    }

    let result = match &this {
        JsValue::Number(JsNumberType::Integer(i)) => {
            if radix == 10 {
                i.to_string()
            } else {
                format_radix(*i as i128, radix)
            }
        }
        JsValue::Number(JsNumberType::Float(f)) => {
            if radix == 10 {
                f.to_string()
            } else {
                // For non-base-10, convert to integer first
                format_radix(*f as i128, radix)
            }
        }
        JsValue::Number(JsNumberType::NaN) => "NaN".to_string(),
        JsValue::Number(JsNumberType::PositiveInfinity) => "Infinity".to_string(),
        JsValue::Number(JsNumberType::NegativeInfinity) => "-Infinity".to_string(),
        _ => "NaN".to_string(),
    };

    Ok(JsValue::String(result))
}

/// Number.prototype.toFixed
fn number_to_fixed(
    _ctx: &mut EvalContext,
    this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let digits = if args.is_empty() {
        0
    } else {
        match &args[0] {
            JsValue::Number(JsNumberType::Integer(i)) => *i as usize,
            JsValue::Number(JsNumberType::Float(f)) => *f as usize,
            _ => 0,
        }
    };

    if digits > 100 {
        return Err(JErrorType::RangeError("toFixed() digits argument must be between 0 and 100".to_string()));
    }

    let result = match &this {
        JsValue::Number(JsNumberType::Integer(i)) => format!("{:.1$}", *i as f64, digits),
        JsValue::Number(JsNumberType::Float(f)) => format!("{:.1$}", f, digits),
        JsValue::Number(JsNumberType::NaN) => "NaN".to_string(),
        JsValue::Number(JsNumberType::PositiveInfinity) => "Infinity".to_string(),
        JsValue::Number(JsNumberType::NegativeInfinity) => "-Infinity".to_string(),
        _ => "NaN".to_string(),
    };

    Ok(JsValue::String(result))
}

/// Number.prototype.toExponential
fn number_to_exponential(
    _ctx: &mut EvalContext,
    this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let digits = if args.is_empty() {
        None
    } else {
        Some(match &args[0] {
            JsValue::Number(JsNumberType::Integer(i)) => *i as usize,
            JsValue::Number(JsNumberType::Float(f)) => *f as usize,
            JsValue::Undefined => return number_to_exponential(_ctx, this, vec![]),
            _ => 0,
        })
    };

    if let Some(d) = digits {
        if d > 100 {
            return Err(JErrorType::RangeError("toExponential() argument must be between 0 and 100".to_string()));
        }
    }

    let result = match &this {
        JsValue::Number(JsNumberType::Integer(i)) => {
            let f = *i as f64;
            match digits {
                Some(d) => format!("{:.1$e}", f, d),
                None => format!("{:e}", f),
            }
        }
        JsValue::Number(JsNumberType::Float(f)) => {
            match digits {
                Some(d) => format!("{:.1$e}", f, d),
                None => format!("{:e}", f),
            }
        }
        JsValue::Number(JsNumberType::NaN) => "NaN".to_string(),
        JsValue::Number(JsNumberType::PositiveInfinity) => "Infinity".to_string(),
        JsValue::Number(JsNumberType::NegativeInfinity) => "-Infinity".to_string(),
        _ => "NaN".to_string(),
    };

    Ok(JsValue::String(result))
}

/// Number.prototype.toPrecision
fn number_to_precision(
    _ctx: &mut EvalContext,
    this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    if args.is_empty() {
        return number_to_string(_ctx, this, vec![]);
    }

    let precision = match &args[0] {
        JsValue::Undefined => return number_to_string(_ctx, this, vec![]),
        JsValue::Number(JsNumberType::Integer(i)) => *i as usize,
        JsValue::Number(JsNumberType::Float(f)) => *f as usize,
        _ => return number_to_string(_ctx, this, vec![]),
    };

    if precision == 0 || precision > 100 {
        return Err(JErrorType::RangeError("toPrecision() argument must be between 1 and 100".to_string()));
    }

    let result = match &this {
        JsValue::Number(JsNumberType::Integer(i)) => format!("{:.1$}", *i as f64, precision - 1),
        JsValue::Number(JsNumberType::Float(f)) => format!("{:.1$}", f, precision - 1),
        JsValue::Number(JsNumberType::NaN) => "NaN".to_string(),
        JsValue::Number(JsNumberType::PositiveInfinity) => "Infinity".to_string(),
        JsValue::Number(JsNumberType::NegativeInfinity) => "-Infinity".to_string(),
        _ => "NaN".to_string(),
    };

    Ok(JsValue::String(result))
}

/// Convert JsValue to number.
fn to_number(value: &JsValue) -> Result<JsValue, JErrorType> {
    Ok(match value {
        JsValue::Undefined => JsValue::Number(JsNumberType::NaN),
        JsValue::Null => JsValue::Number(JsNumberType::Integer(0)),
        JsValue::Boolean(true) => JsValue::Number(JsNumberType::Integer(1)),
        JsValue::Boolean(false) => JsValue::Number(JsNumberType::Integer(0)),
        JsValue::Number(n) => JsValue::Number(n.clone()),
        JsValue::String(s) => {
            let s = s.trim();
            if s.is_empty() {
                JsValue::Number(JsNumberType::Integer(0))
            } else if let Ok(i) = s.parse::<i64>() {
                JsValue::Number(JsNumberType::Integer(i))
            } else if let Ok(f) = s.parse::<f64>() {
                JsValue::Number(JsNumberType::Float(f))
            } else {
                JsValue::Number(JsNumberType::NaN)
            }
        }
        JsValue::Symbol(_) => {
            return Err(JErrorType::TypeError("Cannot convert Symbol to number".to_string()))
        }
        JsValue::Object(_) => JsValue::Number(JsNumberType::NaN),
    })
}

/// Format a number in a given radix.
fn format_radix(mut n: i128, radix: u32) -> String {
    const DIGITS: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyz";

    if n == 0 {
        return "0".to_string();
    }

    let negative = n < 0;
    if negative {
        n = -n;
    }

    let mut result = Vec::new();
    while n > 0 {
        result.push(DIGITS[(n % radix as i128) as usize]);
        n /= radix as i128;
    }

    if negative {
        result.push(b'-');
    }

    result.reverse();
    String::from_utf8(result).unwrap()
}
