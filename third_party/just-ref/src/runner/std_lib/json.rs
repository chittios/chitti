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

    let value = &args[0];
    stringify_value(value)
}

/// Stringify a JavaScript value to JSON.
fn stringify_value(value: &JsValue) -> Result<JsValue, JErrorType> {
    let result = match value {
        JsValue::Null => "null".to_string(),
        JsValue::Boolean(true) => "true".to_string(),
        JsValue::Boolean(false) => "false".to_string(),
        JsValue::Number(JsNumberType::Integer(i)) => i.to_string(),
        JsValue::Number(JsNumberType::Float(f)) => {
            if f.is_finite() {
                f.to_string()
            } else {
                "null".to_string()
            }
        }
        JsValue::Number(JsNumberType::NaN) => "null".to_string(),
        JsValue::Number(JsNumberType::PositiveInfinity) => "null".to_string(),
        JsValue::Number(JsNumberType::NegativeInfinity) => "null".to_string(),
        JsValue::String(s) => stringify_string(s),
        JsValue::Undefined => return Ok(JsValue::Undefined),
        JsValue::Symbol(_) => return Ok(JsValue::Undefined),
        JsValue::Object(_) => {
            // TODO: Implement object/array stringification when those are available
            return Err(JErrorType::TypeError(
                "JSON.stringify for objects not yet implemented".to_string(),
            ));
        }
    };

    Ok(JsValue::String(result))
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
