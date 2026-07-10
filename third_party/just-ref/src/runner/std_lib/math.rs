//! Math built-in object.
//!
//! Provides mathematical constants and functions.

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

/// Register the Math object with the registry.
pub fn register(registry: &mut BuiltInRegistry) {
    let math = BuiltInObject::new("Math")
        .with_no_prototype()
        // Constants
        .add_property("E", JsValue::Number(JsNumberType::Float(core::f64::consts::E)))
        .add_property("LN10", JsValue::Number(JsNumberType::Float(core::f64::consts::LN_10)))
        .add_property("LN2", JsValue::Number(JsNumberType::Float(core::f64::consts::LN_2)))
        .add_property("LOG10E", JsValue::Number(JsNumberType::Float(core::f64::consts::LOG10_E)))
        .add_property("LOG2E", JsValue::Number(JsNumberType::Float(core::f64::consts::LOG2_E)))
        .add_property("PI", JsValue::Number(JsNumberType::Float(core::f64::consts::PI)))
        .add_property("SQRT1_2", JsValue::Number(JsNumberType::Float(core::f64::consts::FRAC_1_SQRT_2)))
        .add_property("SQRT2", JsValue::Number(JsNumberType::Float(core::f64::consts::SQRT_2)))
        // Methods
        .add_method("abs", math_abs)
        .add_method("floor", math_floor)
        .add_method("ceil", math_ceil)
        .add_method("round", math_round)
        .add_method("trunc", math_trunc)
        .add_method("sign", math_sign)
        .add_method("min", math_min)
        .add_method("max", math_max)
        .add_method("sqrt", math_sqrt)
        .add_method("cbrt", math_cbrt)
        .add_method("pow", math_pow)
        .add_method("exp", math_exp)
        .add_method("expm1", math_expm1)
        .add_method("log", math_log)
        .add_method("log10", math_log10)
        .add_method("log2", math_log2)
        .add_method("log1p", math_log1p)
        .add_method("sin", math_sin)
        .add_method("cos", math_cos)
        .add_method("tan", math_tan)
        .add_method("asin", math_asin)
        .add_method("acos", math_acos)
        .add_method("atan", math_atan)
        .add_method("atan2", math_atan2)
        .add_method("sinh", math_sinh)
        .add_method("cosh", math_cosh)
        .add_method("tanh", math_tanh)
        .add_method("asinh", math_asinh)
        .add_method("acosh", math_acosh)
        .add_method("atanh", math_atanh)
        .add_method("hypot", math_hypot)
        .add_method("random", math_random)
        .add_method("clz32", math_clz32)
        .add_method("imul", math_imul)
        .add_method("fround", math_fround);

    registry.register_object(math);
}

/// Convert JsValue to f64 for math operations.
fn to_f64(value: &JsValue) -> f64 {
    match value {
        JsValue::Number(JsNumberType::Integer(i)) => *i as f64,
        JsValue::Number(JsNumberType::Float(f)) => *f,
        JsValue::Number(JsNumberType::NaN) => f64::NAN,
        JsValue::Number(JsNumberType::PositiveInfinity) => f64::INFINITY,
        JsValue::Number(JsNumberType::NegativeInfinity) => f64::NEG_INFINITY,
        JsValue::Boolean(true) => 1.0,
        JsValue::Boolean(false) => 0.0,
        JsValue::Null => 0.0,
        JsValue::Undefined => f64::NAN,
        JsValue::String(s) => s.trim().parse().unwrap_or(f64::NAN),
        _ => f64::NAN,
    }
}

/// Convert f64 result to JsValue.
fn from_f64(f: f64) -> JsValue {
    if f.is_nan() {
        JsValue::Number(JsNumberType::NaN)
    } else if f == f64::INFINITY {
        JsValue::Number(JsNumberType::PositiveInfinity)
    } else if f == f64::NEG_INFINITY {
        JsValue::Number(JsNumberType::NegativeInfinity)
    } else if f.fract() == 0.0 && f.abs() < i64::MAX as f64 {
        JsValue::Number(JsNumberType::Integer(f as i64))
    } else {
        JsValue::Number(JsNumberType::Float(f))
    }
}

/// Math.abs
fn math_abs(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    if args.is_empty() {
        return Ok(JsValue::Number(JsNumberType::NaN));
    }
    Ok(from_f64(to_f64(&args[0]).abs()))
}

/// Math.floor
fn math_floor(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    if args.is_empty() {
        return Ok(JsValue::Number(JsNumberType::NaN));
    }
    Ok(from_f64(to_f64(&args[0]).floor()))
}

/// Math.ceil
fn math_ceil(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    if args.is_empty() {
        return Ok(JsValue::Number(JsNumberType::NaN));
    }
    Ok(from_f64(to_f64(&args[0]).ceil()))
}

/// Math.round
fn math_round(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    if args.is_empty() {
        return Ok(JsValue::Number(JsNumberType::NaN));
    }
    Ok(from_f64(to_f64(&args[0]).round()))
}

/// Math.trunc
fn math_trunc(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    if args.is_empty() {
        return Ok(JsValue::Number(JsNumberType::NaN));
    }
    Ok(from_f64(to_f64(&args[0]).trunc()))
}

/// Math.sign
fn math_sign(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    if args.is_empty() {
        return Ok(JsValue::Number(JsNumberType::NaN));
    }
    let x = to_f64(&args[0]);
    if x.is_nan() {
        Ok(JsValue::Number(JsNumberType::NaN))
    } else if x == 0.0 {
        Ok(JsValue::Number(JsNumberType::Integer(0)))
    } else if x > 0.0 {
        Ok(JsValue::Number(JsNumberType::Integer(1)))
    } else {
        Ok(JsValue::Number(JsNumberType::Integer(-1)))
    }
}

/// Math.min
fn math_min(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    if args.is_empty() {
        return Ok(JsValue::Number(JsNumberType::PositiveInfinity));
    }

    let mut result = f64::INFINITY;
    for arg in &args {
        let x = to_f64(arg);
        if x.is_nan() {
            return Ok(JsValue::Number(JsNumberType::NaN));
        }
        if x < result {
            result = x;
        }
    }
    Ok(from_f64(result))
}

/// Math.max
fn math_max(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    if args.is_empty() {
        return Ok(JsValue::Number(JsNumberType::NegativeInfinity));
    }

    let mut result = f64::NEG_INFINITY;
    for arg in &args {
        let x = to_f64(arg);
        if x.is_nan() {
            return Ok(JsValue::Number(JsNumberType::NaN));
        }
        if x > result {
            result = x;
        }
    }
    Ok(from_f64(result))
}

/// Math.sqrt
fn math_sqrt(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    if args.is_empty() {
        return Ok(JsValue::Number(JsNumberType::NaN));
    }
    Ok(from_f64(to_f64(&args[0]).sqrt()))
}

/// Math.cbrt
fn math_cbrt(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    if args.is_empty() {
        return Ok(JsValue::Number(JsNumberType::NaN));
    }
    Ok(from_f64(to_f64(&args[0]).cbrt()))
}

/// Math.pow
fn math_pow(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    if args.len() < 2 {
        return Ok(JsValue::Number(JsNumberType::NaN));
    }
    let base = to_f64(&args[0]);
    let exp = to_f64(&args[1]);
    Ok(from_f64(base.powf(exp)))
}

/// Math.exp
fn math_exp(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    if args.is_empty() {
        return Ok(JsValue::Number(JsNumberType::NaN));
    }
    Ok(from_f64(to_f64(&args[0]).exp()))
}

/// Math.expm1
fn math_expm1(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    if args.is_empty() {
        return Ok(JsValue::Number(JsNumberType::NaN));
    }
    Ok(from_f64(to_f64(&args[0]).exp_m1()))
}

/// Math.log (natural logarithm)
fn math_log(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    if args.is_empty() {
        return Ok(JsValue::Number(JsNumberType::NaN));
    }
    Ok(from_f64(to_f64(&args[0]).ln()))
}

/// Math.log10
fn math_log10(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    if args.is_empty() {
        return Ok(JsValue::Number(JsNumberType::NaN));
    }
    Ok(from_f64(to_f64(&args[0]).log10()))
}

/// Math.log2
fn math_log2(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    if args.is_empty() {
        return Ok(JsValue::Number(JsNumberType::NaN));
    }
    Ok(from_f64(to_f64(&args[0]).log2()))
}

/// Math.log1p
fn math_log1p(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    if args.is_empty() {
        return Ok(JsValue::Number(JsNumberType::NaN));
    }
    Ok(from_f64(to_f64(&args[0]).ln_1p()))
}

/// Math.sin
fn math_sin(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    if args.is_empty() {
        return Ok(JsValue::Number(JsNumberType::NaN));
    }
    Ok(from_f64(to_f64(&args[0]).sin()))
}

/// Math.cos
fn math_cos(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    if args.is_empty() {
        return Ok(JsValue::Number(JsNumberType::NaN));
    }
    Ok(from_f64(to_f64(&args[0]).cos()))
}

/// Math.tan
fn math_tan(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    if args.is_empty() {
        return Ok(JsValue::Number(JsNumberType::NaN));
    }
    Ok(from_f64(to_f64(&args[0]).tan()))
}

/// Math.asin
fn math_asin(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    if args.is_empty() {
        return Ok(JsValue::Number(JsNumberType::NaN));
    }
    Ok(from_f64(to_f64(&args[0]).asin()))
}

/// Math.acos
fn math_acos(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    if args.is_empty() {
        return Ok(JsValue::Number(JsNumberType::NaN));
    }
    Ok(from_f64(to_f64(&args[0]).acos()))
}

/// Math.atan
fn math_atan(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    if args.is_empty() {
        return Ok(JsValue::Number(JsNumberType::NaN));
    }
    Ok(from_f64(to_f64(&args[0]).atan()))
}

/// Math.atan2
fn math_atan2(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    if args.len() < 2 {
        return Ok(JsValue::Number(JsNumberType::NaN));
    }
    let y = to_f64(&args[0]);
    let x = to_f64(&args[1]);
    Ok(from_f64(y.atan2(x)))
}

/// Math.sinh
fn math_sinh(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    if args.is_empty() {
        return Ok(JsValue::Number(JsNumberType::NaN));
    }
    Ok(from_f64(to_f64(&args[0]).sinh()))
}

/// Math.cosh
fn math_cosh(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    if args.is_empty() {
        return Ok(JsValue::Number(JsNumberType::NaN));
    }
    Ok(from_f64(to_f64(&args[0]).cosh()))
}

/// Math.tanh
fn math_tanh(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    if args.is_empty() {
        return Ok(JsValue::Number(JsNumberType::NaN));
    }
    Ok(from_f64(to_f64(&args[0]).tanh()))
}

/// Math.asinh
fn math_asinh(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    if args.is_empty() {
        return Ok(JsValue::Number(JsNumberType::NaN));
    }
    Ok(from_f64(to_f64(&args[0]).asinh()))
}

/// Math.acosh
fn math_acosh(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    if args.is_empty() {
        return Ok(JsValue::Number(JsNumberType::NaN));
    }
    Ok(from_f64(to_f64(&args[0]).acosh()))
}

/// Math.atanh
fn math_atanh(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    if args.is_empty() {
        return Ok(JsValue::Number(JsNumberType::NaN));
    }
    Ok(from_f64(to_f64(&args[0]).atanh()))
}

/// Math.hypot
fn math_hypot(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    if args.is_empty() {
        return Ok(JsValue::Number(JsNumberType::Integer(0)));
    }

    let mut sum_sq = 0.0f64;
    for arg in &args {
        let x = to_f64(arg);
        if x.is_infinite() {
            return Ok(JsValue::Number(JsNumberType::PositiveInfinity));
        }
        sum_sq += x * x;
    }

    Ok(from_f64(sum_sq.sqrt()))
}

/// Math.random
fn math_random(
    _ctx: &mut EvalContext,
    _this: JsValue,
    _args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    // ChittiOS: no wall clock available under no_std here, so seed a
    // process-global LCG from an atomic counter advanced each call. Behaves the
    // same on the host build. A caller wanting entropy can reseed from the
    // kernel RNG later.
    use core::sync::atomic::{AtomicU64, Ordering};
    static STATE: AtomicU64 = AtomicU64::new(0x9E3779B97F4A7C15);
    let seed = STATE.fetch_add(0x2545F4914F6CDD1D, Ordering::Relaxed);
    let mixed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    // Top 53 bits → a double in [0, 1).
    let random = (mixed >> 11) as f64 / ((1u64 << 53) as f64);

    Ok(JsValue::Number(JsNumberType::Float(random)))
}

/// Math.clz32 - Count leading zeros in 32-bit integer.
fn math_clz32(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    if args.is_empty() {
        return Ok(JsValue::Number(JsNumberType::Integer(32)));
    }
    let x = to_f64(&args[0]) as u32;
    Ok(JsValue::Number(JsNumberType::Integer(x.leading_zeros() as i64)))
}

/// Math.imul - 32-bit integer multiplication.
fn math_imul(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    if args.len() < 2 {
        return Ok(JsValue::Number(JsNumberType::Integer(0)));
    }
    let a = to_f64(&args[0]) as i32;
    let b = to_f64(&args[1]) as i32;
    Ok(JsValue::Number(JsNumberType::Integer(a.wrapping_mul(b) as i64)))
}

/// Math.fround - Round to nearest 32-bit float.
fn math_fround(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    if args.is_empty() {
        return Ok(JsValue::Number(JsNumberType::NaN));
    }
    let x = to_f64(&args[0]);
    Ok(from_f64((x as f32) as f64))
}
