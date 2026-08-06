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
    matches!(
        name,
        "isNaN" | "isFinite" | "parseInt" | "parseFloat" | "Boolean" | "Symbol" | "BigInt"
            | "eval"
            | "Function"
            | "encodeURIComponent" | "decodeURIComponent" | "encodeURI" | "decodeURI"
    )
}

/// Percent-encode `s` (UTF-8), leaving the unreserved set — plus any byte in
/// `extra` — unescaped. Backs `encodeURI`/`encodeURIComponent`.
fn uri_encode(s: &str, extra: &[u8]) -> String {
    let mut out = String::new();
    for &b in s.as_bytes() {
        let unreserved = b.is_ascii_alphanumeric()
            || matches!(b, b'-' | b'_' | b'.' | b'!' | b'~' | b'*' | b'\'' | b'(' | b')');
        if unreserved || extra.contains(&b) {
            out.push(b as char);
        } else {
            out.push('%');
            out.push(char::from_digit((b >> 4) as u32, 16).unwrap().to_ascii_uppercase());
            out.push(char::from_digit((b & 0xF) as u32, 16).unwrap().to_ascii_uppercase());
        }
    }
    out
}

/// Decode `%XX` escapes in `s` (UTF-8). Backs `decodeURI`/`decodeURIComponent`.
fn uri_decode(s: &str) -> Result<String, JErrorType> {
    let bytes = s.as_bytes();
    let mut out: alloc::vec::Vec<u8> = alloc::vec::Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            if i + 2 >= bytes.len() {
                return Err(JErrorType::TypeError("URI malformed".to_string()));
            }
            let hi = (bytes[i + 1] as char).to_digit(16);
            let lo = (bytes[i + 2] as char).to_digit(16);
            match (hi, lo) {
                (Some(h), Some(l)) => {
                    out.push((h * 16 + l) as u8);
                    i += 3;
                }
                _ => return Err(JErrorType::TypeError("URI malformed".to_string())),
            }
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    Ok(String::from_utf8_lossy(&out).into_owned())
}

/// Stringify a value for `Function` param/body assembly and `eval` non-string
/// pass-through.
fn as_source_str(v: &JsValue) -> String {
    match v {
        JsValue::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Parse `src` and execute it in the CURRENT context (direct-eval-like scope:
/// declarations land in the caller's environment, lookups see it). Returns the
/// completion value of the last statement — an expression statement yields its
/// value, so `eval("1+1")` is `2` and `eval("(function(){})")` is the function.
fn eval_source(src: &str, ctx: &mut EvalContext) -> Result<JsValue, JErrorType> {
    use crate::parser::JsParser;
    use crate::runner::eval::statement::execute_statement;
    // Direct eval inherits the caller's strictness — an octal literal or legacy
    // escape in `eval("… 01 …")` is a SyntaxError when the surrounding code is
    // strict (Annex-B), so thread `ctx.strict` into the parser.
    let ast = JsParser::parse_to_ast_from_str_strict(src, ctx.strict)
        .map_err(|e| JErrorType::SyntaxError(alloc::format!("{:?}", e)))?;
    let mut last = JsValue::Undefined;
    for stmt in &ast.body {
        let c = execute_statement(stmt, ctx)?;
        if let Some(v) = c.value.clone() {
            last = v;
        }
    }
    Ok(last)
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

/// `BigInt(value)` — the ToBigInt abstract operation.
///
/// Number → must be an integer value (else RangeError); String → parse
/// (empty ⇒ `0n`, invalid ⇒ SyntaxError); Boolean → `0n`/`1n`; BigInt → itself;
/// undefined/null/Symbol/Object → TypeError.
fn to_bigint(v: &JsValue) -> Result<JsValue, JErrorType> {
    use num_bigint::BigInt;
    Ok(match v {
        JsValue::BigInt(b) => JsValue::BigInt(b.clone()),
        JsValue::Boolean(b) => JsValue::BigInt(BigInt::from(if *b { 1 } else { 0 })),
        JsValue::Number(n) => {
            let f = match n {
                JsNumberType::Integer(i) => {
                    return Ok(JsValue::BigInt(BigInt::from(*i)));
                }
                JsNumberType::Float(f) => *f,
                _ => f64::NAN,
            };
            if !f.is_finite() || f.fract() != 0.0 {
                return Err(JErrorType::RangeError(
                    "The number is not a safe integer".to_string(),
                ));
            }
            match <BigInt as num_traits::FromPrimitive>::from_f64(f) {
                Some(b) => JsValue::BigInt(b),
                None => {
                    return Err(JErrorType::RangeError(
                        "The number is not a safe integer".to_string(),
                    ))
                }
            }
        }
        JsValue::String(s) => {
            let t = s.trim();
            if t.is_empty() {
                return Ok(JsValue::BigInt(BigInt::from(0)));
            }
            let (neg, rest) = match t.strip_prefix('-') {
                Some(r) => (true, r),
                None => (false, t.strip_prefix('+').unwrap_or(t)),
            };
            let (radix, digits): (u32, &str) = if let Some(r) =
                rest.strip_prefix("0x").or_else(|| rest.strip_prefix("0X"))
            {
                (16, r)
            } else if let Some(r) = rest.strip_prefix("0o").or_else(|| rest.strip_prefix("0O")) {
                (8, r)
            } else if let Some(r) = rest.strip_prefix("0b").or_else(|| rest.strip_prefix("0B")) {
                (2, r)
            } else {
                (10, rest)
            };
            match BigInt::parse_bytes(digits.as_bytes(), radix) {
                Some(b) => JsValue::BigInt(if neg { -b } else { b }),
                None => {
                    return Err(JErrorType::SyntaxError(
                        "Cannot convert string to a BigInt".to_string(),
                    ))
                }
            }
        }
        JsValue::Undefined | JsValue::Null => {
            return Err(JErrorType::TypeError(
                "Cannot convert undefined or null to a BigInt".to_string(),
            ))
        }
        JsValue::Symbol(_) => {
            return Err(JErrorType::TypeError(
                "Cannot convert a Symbol to a BigInt".to_string(),
            ))
        }
        JsValue::Object(_) => {
            return Err(JErrorType::TypeError(
                "Cannot convert an object to a BigInt".to_string(),
            ))
        }
    })
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
                // Real engines report these as typeof "function".
                set_own_prop(&f, "__host_fn__", JsValue::Boolean(true), false);
                f
            }
            _ => JsValue::Undefined,
        })
    }

    fn call_method(
        &self,
        object_name: &str,
        method_name: &str,
        _ctx: &mut EvalContext,
        _this: JsValue,
        args: Vec<JsValue>,
    ) -> Option<Result<JsValue, JErrorType>> {
        if object_name != "Symbol" {
            return None;
        }
        match method_name {
            // Global symbol registry. Same key → same symbol (description identity).
            // Prefix keeps these distinct from `Symbol("key")` under description Eq.
            "for" => {
                let key = match args.get(0) {
                    Some(JsValue::String(s)) => s.clone(),
                    Some(other) => other.to_string(),
                    None => String::from("undefined"),
                };
                Some(Ok(JsValue::Symbol(SymbolData::new(alloc::format!(
                    "Symbol.for({key})"
                )))))
            }
            "keyFor" => match args.get(0) {
                Some(JsValue::Symbol(s)) => {
                    let d = s.description();
                    if let Some(key) = d
                        .strip_prefix("Symbol.for(")
                        .and_then(|r| r.strip_suffix(')'))
                    {
                        Some(Ok(JsValue::String(key.to_string())))
                    } else {
                        Some(Ok(JsValue::Undefined))
                    }
                }
                _ => Some(Ok(JsValue::Undefined)),
            },
            _ => None,
        }
    }

    fn has_method(&self, object_name: &str, method_name: &str) -> bool {
        object_name == "Symbol" && matches!(method_name, "for" | "keyFor")
    }

    fn call_constructor(
        &self,
        object_name: &str,
        ctx: &mut EvalContext,
        this: JsValue,
        args: Vec<JsValue>,
    ) -> Option<Result<JsValue, JErrorType>> {
        if !is_global_fn(object_name) {
            return None;
        }
        let a0 = args.get(0).cloned().unwrap_or(JsValue::Undefined);
        // `eval` / `Function` need the parser + interpreter, and return early.
        match object_name {
            "eval" => {
                return Some(match &a0 {
                    // eval of a non-string returns it unchanged (per spec).
                    JsValue::String(s) => eval_source(s, ctx),
                    other => Ok(other.clone()),
                });
            }
            "Function" => {
                let (params, body) = if args.is_empty() {
                    (String::new(), String::new())
                } else {
                    let body = as_source_str(args.last().unwrap());
                    let params = args[..args.len() - 1]
                        .iter()
                        .map(as_source_str)
                        .collect::<Vec<_>>()
                        .join(",");
                    (params, body)
                };
                let src = alloc::format!("(function anonymous({}){{{}}})", params, body);
                return Some(eval_source(&src, ctx));
            }
            _ => {}
        }
        let res = match object_name {
            "isNaN" => JsValue::Boolean(to_f64(&a0).is_nan()),
            "isFinite" => {
                let f = to_f64(&a0);
                JsValue::Boolean(!f.is_nan() && f.is_finite())
            }
            "parseInt" => parse_int(&a0, args.get(1).unwrap_or(&JsValue::Undefined)),
            "parseFloat" => parse_float(&a0),
            "encodeURIComponent" => JsValue::String(uri_encode(&as_source_str(&a0), b"")),
            // encodeURI also preserves the reserved + mark characters.
            "encodeURI" => {
                JsValue::String(uri_encode(&as_source_str(&a0), b";,/?:@&=+$#"))
            }
            "decodeURIComponent" | "decodeURI" => {
                match uri_decode(&as_source_str(&a0)) {
                    Ok(s) => JsValue::String(s),
                    Err(e) => return Some(Err(e)),
                }
            }
            "Boolean" => {
                let b = JsValue::Boolean(truthy(&a0));
                // `new Boolean(x)` boxes; bare `Boolean(x)` coerces.
                // Detect `new` via [[Prototype]] === Boolean.prototype (same
                // rule as Number/String — don't treat globalThis as a target).
                if crate::runner::eval::expression::is_new_this_for("Boolean", &this, ctx) {
                    crate::runner::eval::expression::set_own_prop(
                        &this,
                        "__primitive_value__",
                        b,
                        false,
                    );
                    this
                } else {
                    b
                }
            }
            "BigInt" => {
                return Some(to_bigint(&a0));
            }
            "Symbol" => {
                // Every `Symbol(desc)` call is a distinct value, even when
                // descriptions match — contrast `Symbol.for`, which is keyed.
                let desc = match &a0 {
                    JsValue::Undefined => String::new(),
                    JsValue::String(s) => s.clone(),
                    other => other.to_string(),
                };
                use core::sync::atomic::{AtomicU64, Ordering};
                static NEXT: AtomicU64 = AtomicU64::new(1);
                let n = NEXT.fetch_add(1, Ordering::Relaxed);
                JsValue::Symbol(SymbolData::new(alloc::format!("Symbol({desc})#{n}")))
            }
            _ => JsValue::Undefined,
        };
        Some(Ok(res))
    }

    fn name(&self) -> &str {
        "chitti_globals"
    }
}
