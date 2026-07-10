//! JSON built-in object.
//!
//! Provides JSON.parse and JSON.stringify methods.

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

/// Register the JSON object with the registry.
pub fn register(registry: &mut BuiltInRegistry) {
    let json = BuiltInObject::new("JSON")
        .with_no_prototype()
        .add_method("parse", json_parse)
        .add_method("stringify", json_stringify);

    registry.register_object(json);
}

/// JSON.parse - Parse JSON string to JavaScript value.
fn json_parse(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    if args.is_empty() {
        return Err(JErrorType::SyntaxError("Unexpected end of JSON input".to_string()));
    }

    let s = match &args[0] {
        JsValue::String(s) => s.trim(),
        _ => {
            return Err(JErrorType::SyntaxError(
                "JSON.parse requires a string argument".to_string(),
            ))
        }
    };

    if s.is_empty() {
        return Err(JErrorType::SyntaxError("Unexpected end of JSON input".to_string()));
    }

    // Simple JSON parser
    let mut chars = s.chars().peekable();
    parse_value(&mut chars)
}

/// ChittiOS: parse a JSON string into a `JsValue` (used by `Response.json()`).
pub fn parse_str(s: &str) -> Result<JsValue, JErrorType> {
    let s = s.trim();
    if s.is_empty() {
        return Err(JErrorType::SyntaxError("Unexpected end of JSON input".to_string()));
    }
    let mut chars = s.chars().peekable();
    parse_value(&mut chars)
}

/// Parse a JSON value.
fn parse_value(chars: &mut core::iter::Peekable<core::str::Chars>) -> Result<JsValue, JErrorType> {
    skip_whitespace(chars);

    match chars.peek() {
        Some('"') => parse_string(chars),
        Some('0'..='9') | Some('-') => parse_number(chars),
        Some('t') | Some('f') => parse_boolean(chars),
        Some('n') => parse_null(chars),
        Some('[') => parse_array(chars),
        Some('{') => parse_object(chars),
        Some(c) => Err(JErrorType::SyntaxError(format!("Unexpected character: {}", c))),
        None => Err(JErrorType::SyntaxError("Unexpected end of JSON input".to_string())),
    }
}

/// Skip whitespace characters.
fn skip_whitespace(chars: &mut core::iter::Peekable<core::str::Chars>) {
    while let Some(&c) = chars.peek() {
        if c.is_whitespace() {
            chars.next();
        } else {
            break;
        }
    }
}

/// Parse a JSON string.
fn parse_string(chars: &mut core::iter::Peekable<core::str::Chars>) -> Result<JsValue, JErrorType> {
    chars.next(); // consume opening quote

    let mut result = String::new();

    loop {
        match chars.next() {
            Some('"') => return Ok(JsValue::String(result)),
            Some('\\') => {
                match chars.next() {
                    Some('"') => result.push('"'),
                    Some('\\') => result.push('\\'),
                    Some('/') => result.push('/'),
                    Some('b') => result.push('\u{0008}'),
                    Some('f') => result.push('\u{000C}'),
                    Some('n') => result.push('\n'),
                    Some('r') => result.push('\r'),
                    Some('t') => result.push('\t'),
                    Some('u') => {
                        let mut hex = String::with_capacity(4);
                        for _ in 0..4 {
                            match chars.next() {
                                Some(c) if c.is_ascii_hexdigit() => hex.push(c),
                                _ => return Err(JErrorType::SyntaxError("Invalid unicode escape".to_string())),
                            }
                        }
                        let code = u32::from_str_radix(&hex, 16)
                            .map_err(|_| JErrorType::SyntaxError("Invalid unicode escape".to_string()))?;
                        if let Some(c) = char::from_u32(code) {
                            result.push(c);
                        }
                    }
                    _ => return Err(JErrorType::SyntaxError("Invalid escape sequence".to_string())),
                }
            }
            Some(c) if c.is_control() => {
                return Err(JErrorType::SyntaxError("Invalid character in string".to_string()))
            }
            Some(c) => result.push(c),
            None => return Err(JErrorType::SyntaxError("Unterminated string".to_string())),
        }
    }
}

/// Parse a JSON number.
fn parse_number(chars: &mut core::iter::Peekable<core::str::Chars>) -> Result<JsValue, JErrorType> {
    let mut num_str = String::new();

    // Optional negative sign
    if chars.peek() == Some(&'-') {
        num_str.push(chars.next().unwrap());
    }

    // Integer part
    match chars.peek() {
        Some('0') => {
            num_str.push(chars.next().unwrap());
        }
        Some('1'..='9') => {
            while let Some(&c) = chars.peek() {
                if c.is_ascii_digit() {
                    num_str.push(chars.next().unwrap());
                } else {
                    break;
                }
            }
        }
        _ => return Err(JErrorType::SyntaxError("Invalid number".to_string())),
    }

    // Fractional part
    if chars.peek() == Some(&'.') {
        num_str.push(chars.next().unwrap());
        let mut has_digits = false;
        while let Some(&c) = chars.peek() {
            if c.is_ascii_digit() {
                num_str.push(chars.next().unwrap());
                has_digits = true;
            } else {
                break;
            }
        }
        if !has_digits {
            return Err(JErrorType::SyntaxError("Invalid number".to_string()));
        }
    }

    // Exponent part
    if let Some(&c) = chars.peek() {
        if c == 'e' || c == 'E' {
            num_str.push(chars.next().unwrap());

            // Optional sign
            if let Some(&c) = chars.peek() {
                if c == '+' || c == '-' {
                    num_str.push(chars.next().unwrap());
                }
            }

            // Exponent digits
            let mut has_digits = false;
            while let Some(&c) = chars.peek() {
                if c.is_ascii_digit() {
                    num_str.push(chars.next().unwrap());
                    has_digits = true;
                } else {
                    break;
                }
            }
            if !has_digits {
                return Err(JErrorType::SyntaxError("Invalid number".to_string()));
            }
        }
    }

    // Parse the number string
    if num_str.contains('.') || num_str.contains('e') || num_str.contains('E') {
        num_str
            .parse::<f64>()
            .map(|f| JsValue::Number(JsNumberType::Float(f)))
            .map_err(|_| JErrorType::SyntaxError("Invalid number".to_string()))
    } else {
        num_str
            .parse::<i64>()
            .map(|i| JsValue::Number(JsNumberType::Integer(i)))
            .map_err(|_| JErrorType::SyntaxError("Invalid number".to_string()))
    }
}

/// Parse a JSON boolean.
fn parse_boolean(chars: &mut core::iter::Peekable<core::str::Chars>) -> Result<JsValue, JErrorType> {
    if chars.peek() == Some(&'t') {
        if chars.next() == Some('t')
            && chars.next() == Some('r')
            && chars.next() == Some('u')
            && chars.next() == Some('e')
        {
            return Ok(JsValue::Boolean(true));
        }
    } else if chars.peek() == Some(&'f') {
        if chars.next() == Some('f')
            && chars.next() == Some('a')
            && chars.next() == Some('l')
            && chars.next() == Some('s')
            && chars.next() == Some('e')
        {
            return Ok(JsValue::Boolean(false));
        }
    }
    Err(JErrorType::SyntaxError("Invalid literal".to_string()))
}

/// Parse JSON null.
fn parse_null(chars: &mut core::iter::Peekable<core::str::Chars>) -> Result<JsValue, JErrorType> {
    if chars.next() == Some('n')
        && chars.next() == Some('u')
        && chars.next() == Some('l')
        && chars.next() == Some('l')
    {
        Ok(JsValue::Null)
    } else {
        Err(JErrorType::SyntaxError("Invalid literal".to_string()))
    }
}

/// Parse a JSON array.
fn parse_array(chars: &mut core::iter::Peekable<core::str::Chars>) -> Result<JsValue, JErrorType> {
    chars.next(); // consume '['
    skip_whitespace(chars);

    if chars.peek() == Some(&']') {
        chars.next();
        return Ok(crate::runner::eval::expression::make_array(Vec::new()));
    }

    let mut elements = Vec::new();

    loop {
        let value = parse_value(chars)?;
        elements.push(value);

        skip_whitespace(chars);

        match chars.peek() {
            Some(',') => {
                chars.next();
                skip_whitespace(chars);
            }
            Some(']') => {
                chars.next();
                return Ok(crate::runner::eval::expression::make_array(elements));
            }
            _ => return Err(JErrorType::SyntaxError("Expected ',' or ']'".to_string())),
        }
    }
}

/// Parse a JSON object.
fn parse_object(chars: &mut core::iter::Peekable<core::str::Chars>) -> Result<JsValue, JErrorType> {
    chars.next(); // consume '{'
    skip_whitespace(chars);

    if chars.peek() == Some(&'}') {
        chars.next();
        return Ok(crate::runner::eval::expression::make_object(Vec::new()));
    }

    let mut properties = Vec::new();

    loop {
        skip_whitespace(chars);

        // Parse key (must be a string)
        let key = match chars.peek() {
            Some('"') => {
                if let JsValue::String(s) = parse_string(chars)? {
                    s
                } else {
                    return Err(JErrorType::SyntaxError("Expected string key".to_string()));
                }
            }
            _ => return Err(JErrorType::SyntaxError("Expected string key".to_string())),
        };

        skip_whitespace(chars);

        // Expect ':'
        if chars.next() != Some(':') {
            return Err(JErrorType::SyntaxError("Expected ':'".to_string()));
        }

        skip_whitespace(chars);

        // Parse value
        let value = parse_value(chars)?;
        properties.push((key, value));

        skip_whitespace(chars);

        match chars.peek() {
            Some(',') => {
                chars.next();
            }
            Some('}') => {
                chars.next();
                return Ok(crate::runner::eval::expression::make_object(properties));
            }
            _ => return Err(JErrorType::SyntaxError("Expected ',' or '}'".to_string())),
        }
    }
}

/// JSON.stringify - Convert JavaScript value to JSON string.
fn json_stringify(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    if args.is_empty() {
        return Ok(JsValue::Undefined);
    }

    // The optional `space` argument: a number (clamped to 10 spaces) or a string
    // (first 10 chars) → pretty-print indentation.
    let indent = match args.get(2) {
        Some(JsValue::Number(JsNumberType::Integer(n))) => {
            " ".repeat((*n).clamp(0, 10) as usize)
        }
        Some(JsValue::Number(JsNumberType::Float(f))) => {
            " ".repeat((*f as i64).clamp(0, 10) as usize)
        }
        Some(JsValue::String(s)) => s.chars().take(10).collect(),
        _ => String::new(),
    };

    match stringify_inner(&args[0], &indent, "") {
        Some(s) => Ok(JsValue::String(s)),
        None => Ok(JsValue::Undefined),
    }
}

/// Stringify a value to JSON, honoring the `indent` unit and `cur` (the current
/// accumulated indentation). Returns `None` for values JSON omits (undefined,
/// functions, symbols) so a containing object can skip the property.
fn stringify_inner(value: &JsValue, indent: &str, cur: &str) -> Option<String> {
    use crate::runner::eval::expression::{
        array_elements, get_own_prop_value, is_array, own_string_keys, value_is_callable,
    };
    match value {
        JsValue::Null => Some("null".to_string()),
        JsValue::Boolean(true) => Some("true".to_string()),
        JsValue::Boolean(false) => Some("false".to_string()),
        JsValue::Number(JsNumberType::Integer(i)) => Some(i.to_string()),
        JsValue::Number(JsNumberType::Float(f)) => {
            Some(if f.is_finite() { f.to_string() } else { "null".to_string() })
        }
        JsValue::Number(_) => Some("null".to_string()), // NaN / ±Infinity
        JsValue::String(s) => Some(stringify_string(s)),
        JsValue::Undefined | JsValue::Symbol(_) => None,
        JsValue::Object(_) => {
            // Functions serialize to nothing (omitted), like undefined.
            if value_is_callable(value) {
                return None;
            }
            let nl = if indent.is_empty() { "" } else { "\n" };
            let inner_indent = alloc::format!("{cur}{indent}");
            let sep = if indent.is_empty() { "," } else { ",\n" };
            let colon = if indent.is_empty() { ":" } else { ": " };
            if is_array(value) {
                let elems = array_elements(value);
                if elems.is_empty() {
                    return Some("[]".to_string());
                }
                let mut parts: Vec<String> = Vec::with_capacity(elems.len());
                for e in &elems {
                    // Omitted values become `null` inside an array.
                    let s = stringify_inner(e, indent, &inner_indent)
                        .unwrap_or_else(|| "null".to_string());
                    parts.push(alloc::format!("{inner_indent}{s}"));
                }
                Some(alloc::format!(
                    "[{nl}{}{nl}{cur}]",
                    parts.join(sep)
                ))
            } else {
                let keys = own_string_keys(value);
                let mut parts: Vec<String> = Vec::new();
                for k in &keys {
                    let Some(v) = get_own_prop_value(value, k) else { continue };
                    if let Some(vs) = stringify_inner(&v, indent, &inner_indent) {
                        parts.push(alloc::format!(
                            "{inner_indent}{}{colon}{vs}",
                            stringify_string(k)
                        ));
                    }
                }
                if parts.is_empty() {
                    return Some("{}".to_string());
                }
                Some(alloc::format!("{{{nl}{}{nl}{cur}}}", parts.join(sep)))
            }
        }
    }
}

/// Stringify a string with proper escaping.
fn stringify_string(s: &str) -> String {
    let mut result = String::with_capacity(s.len() + 2);
    result.push('"');

    for c in s.chars() {
        match c {
            '"' => result.push_str("\\\""),
            '\\' => result.push_str("\\\\"),
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            '\u{0008}' => result.push_str("\\b"),
            '\u{000C}' => result.push_str("\\f"),
            c if c.is_control() => {
                result.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => result.push(c),
        }
    }

    result.push('"');
    result
}
