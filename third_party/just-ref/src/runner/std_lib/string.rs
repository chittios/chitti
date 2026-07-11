//! String built-in.
//!
//! Provides String constructor and prototype methods.

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

/// Register the String built-in with the registry.
pub fn register(registry: &mut BuiltInRegistry) {
    let string = BuiltInObject::new("String")
        .with_constructor(string_constructor)
        .add_method("charAt", string_char_at)
        .add_method("charCodeAt", string_char_code_at)
        .add_method("substring", string_substring)
        .add_method("slice", string_slice)
        .add_method("indexOf", string_index_of)
        .add_method("lastIndexOf", string_last_index_of)
        .add_method("includes", string_includes)
        .add_method("startsWith", string_starts_with)
        .add_method("endsWith", string_ends_with)
        .add_method("split", string_split)
        .add_method("trim", string_trim)
        .add_method("trimStart", string_trim_start)
        .add_method("trimEnd", string_trim_end)
        .add_method("toUpperCase", string_to_upper_case)
        .add_method("toLowerCase", string_to_lower_case)
        .add_method("repeat", string_repeat)
        .add_method("padStart", string_pad_start)
        .add_method("padEnd", string_pad_end)
        .add_method("replace", string_replace)
        .add_method("match", string_match)
        .add_method("search", string_search)
        .add_method("concat", string_concat)
        .add_method("fromCharCode", string_from_char_code);

    registry.register_object(string);
}

/// Get a string from a JsValue.
fn to_string(value: &JsValue) -> String {
    match value {
        JsValue::String(s) => s.clone(),
        JsValue::Undefined => "undefined".to_string(),
        JsValue::Null => "null".to_string(),
        JsValue::Boolean(b) => b.to_string(),
        JsValue::Number(n) => n.to_string(),
        JsValue::BigInt(b) => b.to_string(),
        JsValue::Symbol(s) => s.to_string(),
        JsValue::Object(_) => "[object Object]".to_string(),
    }
}

/// String constructor.
fn string_constructor(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    if args.is_empty() {
        Ok(JsValue::String(String::new()))
    } else {
        Ok(JsValue::String(to_string(&args[0])))
    }
}

/// String.prototype.charAt
fn string_char_at(
    _ctx: &mut EvalContext,
    this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let s = to_string(&this);
    let index = if args.is_empty() {
        0
    } else {
        to_integer(&args[0])
    };

    if index < 0 || index as usize >= s.chars().count() {
        return Ok(JsValue::String(String::new()));
    }

    Ok(JsValue::String(
        s.chars().nth(index as usize).map(|c| c.to_string()).unwrap_or_default()
    ))
}

/// String.prototype.charCodeAt
fn string_char_code_at(
    _ctx: &mut EvalContext,
    this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let s = to_string(&this);
    let index = if args.is_empty() {
        0
    } else {
        to_integer(&args[0])
    };

    if index < 0 || index as usize >= s.chars().count() {
        return Ok(JsValue::Number(JsNumberType::NaN));
    }

    let code = s.chars().nth(index as usize).map(|c| c as i64).unwrap_or(0);
    Ok(JsValue::Number(JsNumberType::Integer(code)))
}

/// String.prototype.substring
fn string_substring(
    _ctx: &mut EvalContext,
    this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let s = to_string(&this);
    let len = s.chars().count() as i64;

    let mut start = if args.is_empty() {
        0
    } else {
        to_integer(&args[0]).max(0).min(len)
    };

    let mut end = if args.len() < 2 {
        len
    } else {
        to_integer(&args[1]).max(0).min(len)
    };

    // Swap if start > end
    if start > end {
        core::mem::swap(&mut start, &mut end);
    }

    let result: String = s.chars()
        .skip(start as usize)
        .take((end - start) as usize)
        .collect();

    Ok(JsValue::String(result))
}

/// String.prototype.slice
fn string_slice(
    _ctx: &mut EvalContext,
    this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let s = to_string(&this);
    let len = s.chars().count() as i64;

    let start = if args.is_empty() {
        0
    } else {
        let idx = to_integer(&args[0]);
        if idx < 0 {
            (len + idx).max(0)
        } else {
            idx.min(len)
        }
    };

    let end = if args.len() < 2 {
        len
    } else {
        let idx = to_integer(&args[1]);
        if idx < 0 {
            (len + idx).max(0)
        } else {
            idx.min(len)
        }
    };

    if start >= end {
        return Ok(JsValue::String(String::new()));
    }

    let result: String = s.chars()
        .skip(start as usize)
        .take((end - start) as usize)
        .collect();

    Ok(JsValue::String(result))
}

/// String.prototype.indexOf
fn string_index_of(
    _ctx: &mut EvalContext,
    this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let s = to_string(&this);

    if args.is_empty() {
        return Ok(JsValue::Number(JsNumberType::Integer(-1)));
    }

    let search = to_string(&args[0]);
    let position = if args.len() > 1 {
        to_integer(&args[1]).max(0) as usize
    } else {
        0
    };

    if position >= s.len() {
        return Ok(JsValue::Number(JsNumberType::Integer(-1)));
    }

    match s[position..].find(&search) {
        Some(idx) => Ok(JsValue::Number(JsNumberType::Integer((position + idx) as i64))),
        None => Ok(JsValue::Number(JsNumberType::Integer(-1))),
    }
}

/// String.prototype.lastIndexOf
fn string_last_index_of(
    _ctx: &mut EvalContext,
    this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let s = to_string(&this);

    if args.is_empty() {
        return Ok(JsValue::Number(JsNumberType::Integer(-1)));
    }

    let search = to_string(&args[0]);

    match s.rfind(&search) {
        Some(idx) => Ok(JsValue::Number(JsNumberType::Integer(idx as i64))),
        None => Ok(JsValue::Number(JsNumberType::Integer(-1))),
    }
}

/// String.prototype.includes
fn string_includes(
    _ctx: &mut EvalContext,
    this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let s = to_string(&this);

    if args.is_empty() {
        return Ok(JsValue::Boolean(false));
    }

    let search = to_string(&args[0]);
    let position = if args.len() > 1 {
        to_integer(&args[1]).max(0) as usize
    } else {
        0
    };

    if position >= s.len() {
        return Ok(JsValue::Boolean(false));
    }

    Ok(JsValue::Boolean(s[position..].contains(&search)))
}

/// String.prototype.startsWith
fn string_starts_with(
    _ctx: &mut EvalContext,
    this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let s = to_string(&this);

    if args.is_empty() {
        return Ok(JsValue::Boolean(false));
    }

    let search = to_string(&args[0]);
    let position = if args.len() > 1 {
        to_integer(&args[1]).max(0) as usize
    } else {
        0
    };

    if position >= s.len() {
        return Ok(JsValue::Boolean(search.is_empty()));
    }

    Ok(JsValue::Boolean(s[position..].starts_with(&search)))
}

/// String.prototype.endsWith
fn string_ends_with(
    _ctx: &mut EvalContext,
    this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let s = to_string(&this);

    if args.is_empty() {
        return Ok(JsValue::Boolean(false));
    }

    let search = to_string(&args[0]);
    let end_position = if args.len() > 1 {
        to_integer(&args[1]).max(0).min(s.len() as i64) as usize
    } else {
        s.len()
    };

    Ok(JsValue::Boolean(s[..end_position].ends_with(&search)))
}

/// String.prototype.split
fn string_split(
    _ctx: &mut EvalContext,
    this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let s = to_string(&this);
    use crate::runner::eval::expression::make_array;
    if args.is_empty() || matches!(args[0], JsValue::Undefined) {
        return Ok(make_array(alloc::vec![JsValue::String(s)]));
    }
    if crate::runner::std_lib::regexp::is_regexp(&args[0]) {
        return Ok(make_array(crate::runner::std_lib::regexp::string_split_regexp(&s, &args[0])));
    }
    let separator = to_string(&args[0]);
    let parts: Vec<JsValue> = if separator.is_empty() {
        s.chars().map(|c| JsValue::String(c.to_string())).collect()
    } else {
        s.split(separator.as_str())
            .map(|p| JsValue::String(p.to_string()))
            .collect()
    };
    Ok(make_array(parts))
}

/// String.prototype.trim
fn string_trim(
    _ctx: &mut EvalContext,
    this: JsValue,
    _args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let s = to_string(&this);
    Ok(JsValue::String(s.trim().to_string()))
}

/// String.prototype.trimStart
fn string_trim_start(
    _ctx: &mut EvalContext,
    this: JsValue,
    _args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let s = to_string(&this);
    Ok(JsValue::String(s.trim_start().to_string()))
}

/// String.prototype.trimEnd
fn string_trim_end(
    _ctx: &mut EvalContext,
    this: JsValue,
    _args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let s = to_string(&this);
    Ok(JsValue::String(s.trim_end().to_string()))
}

/// String.prototype.toUpperCase
fn string_to_upper_case(
    _ctx: &mut EvalContext,
    this: JsValue,
    _args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let s = to_string(&this);
    Ok(JsValue::String(s.to_uppercase()))
}

/// String.prototype.toLowerCase
fn string_to_lower_case(
    _ctx: &mut EvalContext,
    this: JsValue,
    _args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let s = to_string(&this);
    Ok(JsValue::String(s.to_lowercase()))
}

/// String.prototype.repeat
fn string_repeat(
    _ctx: &mut EvalContext,
    this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let s = to_string(&this);
    let count = if args.is_empty() {
        0
    } else {
        let n = to_integer(&args[0]);
        if n < 0 {
            return Err(JErrorType::RangeError("Invalid count value".to_string()));
        }
        n as usize
    };

    Ok(JsValue::String(s.repeat(count)))
}

/// String.prototype.padStart
fn string_pad_start(
    _ctx: &mut EvalContext,
    this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let s = to_string(&this);
    let target_len = if args.is_empty() {
        return Ok(JsValue::String(s));
    } else {
        to_integer(&args[0]).max(0) as usize
    };

    if s.len() >= target_len {
        return Ok(JsValue::String(s));
    }

    let pad_string = if args.len() > 1 {
        to_string(&args[1])
    } else {
        " ".to_string()
    };

    if pad_string.is_empty() {
        return Ok(JsValue::String(s));
    }

    let pad_len = target_len - s.len();
    let mut result = String::with_capacity(target_len);

    let full_pads = pad_len / pad_string.len();
    let remaining = pad_len % pad_string.len();

    for _ in 0..full_pads {
        result.push_str(&pad_string);
    }
    result.push_str(&pad_string[..remaining]);
    result.push_str(&s);

    Ok(JsValue::String(result))
}

/// String.prototype.padEnd
fn string_pad_end(
    _ctx: &mut EvalContext,
    this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let s = to_string(&this);
    let target_len = if args.is_empty() {
        return Ok(JsValue::String(s));
    } else {
        to_integer(&args[0]).max(0) as usize
    };

    if s.len() >= target_len {
        return Ok(JsValue::String(s));
    }

    let pad_string = if args.len() > 1 {
        to_string(&args[1])
    } else {
        " ".to_string()
    };

    if pad_string.is_empty() {
        return Ok(JsValue::String(s));
    }

    let pad_len = target_len - s.len();
    let mut result = s;

    let full_pads = pad_len / pad_string.len();
    let remaining = pad_len % pad_string.len();

    for _ in 0..full_pads {
        result.push_str(&pad_string);
    }
    result.push_str(&pad_string[..remaining]);

    Ok(JsValue::String(result))
}

/// String.prototype.replace
fn string_replace(
    _ctx: &mut EvalContext,
    this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let s = to_string(&this);

    if args.len() < 2 {
        return Ok(JsValue::String(s));
    }

    // RegExp replacement (honours the `g` flag + `$1`/`$&`).
    if crate::runner::std_lib::regexp::is_regexp(&args[0]) {
        let replacement = to_string(&args[1]);
        return Ok(JsValue::String(
            crate::runner::std_lib::regexp::string_replace_regexp(&s, &args[0], &replacement),
        ));
    }

    let search = to_string(&args[0]);
    let replacement = to_string(&args[1]);

    // Plain-string replacement (first occurrence only).
    Ok(JsValue::String(s.replacen(&search, &replacement, 1)))
}

/// String.prototype.match(re)
fn string_match(
    _ctx: &mut EvalContext,
    this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let s = to_string(&this);
    let re = match args.first() {
        Some(v) if crate::runner::std_lib::regexp::is_regexp(v) => v.clone(),
        Some(other) => crate::runner::std_lib::regexp::make_regexp(&to_string(other), ""),
        None => crate::runner::std_lib::regexp::make_regexp("", ""),
    };
    Ok(crate::runner::std_lib::regexp::string_match(&s, &re))
}

/// String.prototype.search(re)
fn string_search(
    _ctx: &mut EvalContext,
    this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let s = to_string(&this);
    let re = match args.first() {
        Some(v) if crate::runner::std_lib::regexp::is_regexp(v) => v.clone(),
        Some(other) => crate::runner::std_lib::regexp::make_regexp(&to_string(other), ""),
        None => crate::runner::std_lib::regexp::make_regexp("", ""),
    };
    Ok(JsValue::Number(JsNumberType::Integer(
        crate::runner::std_lib::regexp::string_search(&s, &re),
    )))
}

/// String.prototype.concat
fn string_concat(
    _ctx: &mut EvalContext,
    this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let mut result = to_string(&this);

    for arg in args {
        result.push_str(&to_string(&arg));
    }

    Ok(JsValue::String(result))
}

/// String.fromCharCode - Create string from character codes.
fn string_from_char_code(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let mut result = String::new();

    for arg in args {
        let code = to_integer(&arg);
        if let Some(c) = char::from_u32((code & 0xFFFF) as u32) {
            result.push(c);
        }
    }

    Ok(JsValue::String(result))
}

/// Convert JsValue to integer.
fn to_integer(value: &JsValue) -> i64 {
    match value {
        JsValue::Number(JsNumberType::Integer(i)) => *i,
        JsValue::Number(JsNumberType::Float(f)) => *f as i64,
        JsValue::Number(JsNumberType::NaN) => 0,
        JsValue::Number(JsNumberType::PositiveInfinity) => i64::MAX,
        JsValue::Number(JsNumberType::NegativeInfinity) => i64::MIN,
        JsValue::String(s) => s.trim().parse().unwrap_or(0),
        JsValue::Boolean(true) => 1,
        JsValue::Boolean(false) => 0,
        _ => 0,
    }
}
