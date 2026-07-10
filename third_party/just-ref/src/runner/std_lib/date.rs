//! ChittiOS: `Date` built-in (UTC).
//!
//! Instances are objects tagged `__builtin_name__ = "Date"` with a `__time__`
//! millisecond timestamp; getters compute civil date/time from it. The "current
//! time" comes from a settable global (`set_now_ms`) the kernel updates from its
//! clock; on the host `std` build it seeds from the system clock. Calendar math
//! is Howard Hinnant's `civil_from_days` algorithm (public domain).

#[allow(unused_imports)]
use num_traits::Float;
use alloc::borrow::ToOwned;
use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use alloc::vec;
use alloc::format;
use core::sync::atomic::{AtomicU64, Ordering};
use crate::runner::ds::error::JErrorType;
use crate::runner::ds::value::{JsValue, JsNumberType};
use crate::runner::plugin::registry::BuiltInRegistry;
use crate::runner::plugin::types::{BuiltInObject, EvalContext};
use crate::runner::eval::expression::{get_own_prop_value, make_object, set_own_prop};

/// Millisecond wall clock. The kernel sets this from its RTC/counter; 0 until then.
static NOW_MS: AtomicU64 = AtomicU64::new(0);

/// Set the current wall-clock time (ms since the Unix epoch). Kernel-facing.
pub fn set_now_ms(ms: u64) {
    NOW_MS.store(ms, Ordering::Relaxed);
}

fn now_ms() -> i64 {
    #[cfg(feature = "std")]
    {
        use std::time::{SystemTime, UNIX_EPOCH};
        if let Ok(d) = SystemTime::now().duration_since(UNIX_EPOCH) {
            return d.as_millis() as i64;
        }
    }
    NOW_MS.load(Ordering::Relaxed) as i64
}

pub fn register(registry: &mut BuiltInRegistry) {
    let date = BuiltInObject::new("Date")
        .with_constructor(date_constructor)
        .add_method("now", date_now) // static (also callable as Date.now())
        .add_method("getTime", |_, t, _| Ok(num(time_of(&t))))
        .add_method("valueOf", |_, t, _| Ok(num(time_of(&t))))
        .add_method("getFullYear", |_, t, _| Ok(num(parts(time_of(&t)).0)))
        .add_method("getMonth", |_, t, _| Ok(num(parts(time_of(&t)).1 - 1)))
        .add_method("getDate", |_, t, _| Ok(num(parts(time_of(&t)).2)))
        .add_method("getHours", |_, t, _| Ok(num(parts(time_of(&t)).3)))
        .add_method("getMinutes", |_, t, _| Ok(num(parts(time_of(&t)).4)))
        .add_method("getSeconds", |_, t, _| Ok(num(parts(time_of(&t)).5)))
        .add_method("getDay", |_, t, _| Ok(num(weekday(time_of(&t)))))
        .add_method("toISOString", |_, t, _| Ok(JsValue::String(iso(time_of(&t)))))
        .add_method("toString", |_, t, _| Ok(JsValue::String(iso(time_of(&t)))));
    registry.register_object(date);
}

fn num(n: i64) -> JsValue {
    JsValue::Number(JsNumberType::Integer(n))
}

fn to_i64(v: &JsValue) -> Option<i64> {
    match v {
        JsValue::Number(JsNumberType::Integer(i)) => Some(*i),
        JsValue::Number(JsNumberType::Float(f)) => Some(*f as i64),
        JsValue::String(s) => s.trim().parse::<i64>().ok(),
        _ => None,
    }
}

/// `new Date()` → now; `new Date(ms)` → that timestamp.
fn date_constructor(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let t = match args.first() {
        Some(v) => to_i64(v).unwrap_or(0),
        None => now_ms(),
    };
    let obj = make_object(vec![]);
    set_own_prop(&obj, "__builtin_name__", JsValue::String("Date".to_string()), false);
    set_own_prop(&obj, "__time__", num(t), false);
    Ok(obj)
}

fn date_now(
    _ctx: &mut EvalContext,
    _this: JsValue,
    _args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    Ok(num(now_ms()))
}

fn time_of(this: &JsValue) -> i64 {
    get_own_prop_value(this, "__time__")
        .and_then(|v| to_i64(&v))
        .unwrap_or(0)
}

fn div_floor(a: i64, b: i64) -> i64 {
    let q = a / b;
    if (a % b != 0) && ((a < 0) != (b < 0)) { q - 1 } else { q }
}
fn rem_floor(a: i64, b: i64) -> i64 {
    a - div_floor(a, b) * b
}

/// (year, month[1..12], day[1..31], hour, minute, second) UTC from epoch ms.
fn parts(ms: i64) -> (i64, i64, i64, i64, i64, i64) {
    let secs = div_floor(ms, 1000);
    let days = div_floor(secs, 86400);
    let tod = rem_floor(secs, 86400);
    let (y, mo, d) = civil_from_days(days);
    (y, mo as i64, d as i64, tod / 3600, (tod % 3600) / 60, tod % 60)
}

/// 0 = Sunday. Jan 1 1970 was a Thursday (=4).
fn weekday(ms: i64) -> i64 {
    let days = div_floor(div_floor(ms, 1000), 86400);
    rem_floor(days + 4, 7)
}

/// Howard Hinnant's civil-from-days (days since 1970-01-01) → (y, m, d).
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as i64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as i64; // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m as u32, d)
}

fn iso(ms: i64) -> String {
    let (y, mo, d, h, mi, s) = parts(ms);
    let millis = rem_floor(ms, 1000);
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:03}Z",
        y, mo, d, h, mi, s, millis
    )
}
