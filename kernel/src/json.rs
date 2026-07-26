//! A minimal `no_std` JSON value type with a recursive-descent parser and a
//! pretty-printer, used for the on-disk config files (`/configs/core/*.json`).
//! Not a full spec implementation, but round-trips the config schema (objects,
//! arrays, strings, numbers, bools, null) and tolerates human edits.
//!
//! Objects preserve insertion order (a `Vec` of pairs) so serialized configs
//! keep a stable, readable field order.

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::Write;

#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    /// Look up a key in an object.
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(pairs) => pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Num(n) => Some(*n),
            _ => None,
        }
    }
    pub fn as_i64(&self) -> Option<i64> {
        self.as_f64().map(|n| n as i64)
    }
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Json::Bool(b) => Some(*b),
            _ => None,
        }
    }
    /// An object's `(key, value)` pairs, in document order.
    pub fn as_object(&self) -> Option<&[(String, Json)]> {
        match self {
            Json::Obj(pairs) => Some(pairs),
            _ => None,
        }
    }
    pub fn as_array(&self) -> Option<&[Json]> {
        match self {
            Json::Arr(a) => Some(a),
            _ => None,
        }
    }

    /// Parse a JSON document. `None` on malformed input.
    pub fn parse(s: &str) -> Option<Json> {
        let mut p = Parser { b: s.as_bytes(), i: 0 };
        p.ws();
        let v = p.value()?;
        p.ws();
        if p.i == p.b.len() {
            Some(v)
        } else {
            None
        }
    }

    /// Pretty-print with 2-space indentation.
    pub fn to_pretty(&self) -> String {
        let mut out = String::new();
        self.write_pretty(&mut out, 0);
        out
    }

    /// Compact single-line serialization (no whitespace) — for JSONL, where
    /// each record is one line.
    pub fn to_compact(&self) -> String {
        let mut out = String::new();
        self.write_compact(&mut out);
        out
    }

    fn write_compact(&self, out: &mut String) {
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Json::Num(n) => {
                let as_int = *n as i64;
                if as_int as f64 == *n {
                    let _ = write!(out, "{}", as_int);
                } else {
                    let _ = write!(out, "{}", n);
                }
            }
            Json::Str(s) => write_json_str(out, s),
            Json::Arr(a) => {
                out.push('[');
                for (i, v) in a.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    v.write_compact(out);
                }
                out.push(']');
            }
            Json::Obj(pairs) => {
                out.push('{');
                for (i, (k, v)) in pairs.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    write_json_str(out, k);
                    out.push(':');
                    v.write_compact(out);
                }
                out.push('}');
            }
        }
    }

    fn write_pretty(&self, out: &mut String, depth: usize) {
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Json::Num(n) => {
                // Integer-valued numbers print without a decimal point. Avoid
                // f64::fract/abs (std-only): round-trip through i64 instead.
                let as_int = *n as i64;
                if as_int as f64 == *n {
                    let _ = write!(out, "{}", as_int);
                } else {
                    let _ = write!(out, "{}", n);
                }
            }
            Json::Str(s) => write_json_str(out, s),
            Json::Arr(a) => {
                if a.is_empty() {
                    out.push_str("[]");
                    return;
                }
                out.push_str("[\n");
                for (i, v) in a.iter().enumerate() {
                    indent(out, depth + 1);
                    v.write_pretty(out, depth + 1);
                    out.push_str(if i + 1 < a.len() { ",\n" } else { "\n" });
                }
                indent(out, depth);
                out.push(']');
            }
            Json::Obj(pairs) => {
                if pairs.is_empty() {
                    out.push_str("{}");
                    return;
                }
                out.push_str("{\n");
                for (i, (k, v)) in pairs.iter().enumerate() {
                    indent(out, depth + 1);
                    write_json_str(out, k);
                    out.push_str(": ");
                    v.write_pretty(out, depth + 1);
                    out.push_str(if i + 1 < pairs.len() { ",\n" } else { "\n" });
                }
                indent(out, depth);
                out.push('}');
            }
        }
    }
}

fn indent(out: &mut String, depth: usize) {
    for _ in 0..depth * 2 {
        out.push(' ');
    }
}

fn write_json_str(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => out.push_str("\\r"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    /// Read four hex digits as a u32 (JSON `\u` body).
    fn hex4(&mut self) -> Option<u32> {
        let mut cp: u32 = 0;
        for _ in 0..4 {
            let h = self.peek()?;
            self.i += 1;
            cp = cp * 16 + (h as char).to_digit(16)?;
        }
        Some(cp)
    }

    fn ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\r' | b'\n') {
            self.i += 1;
        }
    }
    fn peek(&self) -> Option<u8> {
        self.b.get(self.i).copied()
    }

    fn value(&mut self) -> Option<Json> {
        self.ws();
        match self.peek()? {
            b'{' => self.object(),
            b'[' => self.array(),
            b'"' => self.string().map(Json::Str),
            b't' | b'f' => self.boolean(),
            b'n' => self.null(),
            _ => self.number(),
        }
    }

    fn object(&mut self) -> Option<Json> {
        self.i += 1; // '{'
        let mut pairs = Vec::new();
        self.ws();
        if self.peek() == Some(b'}') {
            self.i += 1;
            return Some(Json::Obj(pairs));
        }
        loop {
            self.ws();
            let key = self.string()?;
            self.ws();
            if self.peek() != Some(b':') {
                return None;
            }
            self.i += 1;
            let val = self.value()?;
            pairs.push((key, val));
            self.ws();
            match self.peek()? {
                b',' => self.i += 1,
                b'}' => {
                    self.i += 1;
                    return Some(Json::Obj(pairs));
                }
                _ => return None,
            }
        }
    }

    fn array(&mut self) -> Option<Json> {
        self.i += 1; // '['
        let mut items = Vec::new();
        self.ws();
        if self.peek() == Some(b']') {
            self.i += 1;
            return Some(Json::Arr(items));
        }
        loop {
            let v = self.value()?;
            items.push(v);
            self.ws();
            match self.peek()? {
                b',' => self.i += 1,
                b']' => {
                    self.i += 1;
                    return Some(Json::Arr(items));
                }
                _ => return None,
            }
        }
    }

    fn string(&mut self) -> Option<String> {
        if self.peek() != Some(b'"') {
            return None;
        }
        self.i += 1;
        let mut s = String::new();
        loop {
            let c = self.peek()?;
            self.i += 1;
            match c {
                b'"' => return Some(s),
                b'\\' => {
                    let e = self.peek()?;
                    self.i += 1;
                    match e {
                        b'"' => s.push('"'),
                        b'\\' => s.push('\\'),
                        b'/' => s.push('/'),
                        b'n' => s.push('\n'),
                        b't' => s.push('\t'),
                        b'r' => s.push('\r'),
                        b'b' => s.push('\u{0008}'),
                        b'f' => s.push('\u{000C}'),
                        b'u' => {
                            // JSON `\uXXXX` is UTF-16. Emoji and other non-BMP
                            // scalars arrive as a high+low surrogate pair
                            // (`\ud83d\ude0a` → 😊). Without pair assembly,
                            // each half is not a valid scalar and became U+FFFD
                            // (or garbage if callers mis-handled the stream).
                            let hi = self.hex4()?;
                            if (0xD800..=0xDBFF).contains(&hi) {
                                // Expect `\u` low surrogate next.
                                if self.peek() == Some(b'\\') {
                                    self.i += 1;
                                    if self.peek() == Some(b'u') {
                                        self.i += 1;
                                        let lo = self.hex4()?;
                                        if (0xDC00..=0xDFFF).contains(&lo) {
                                            let uni = 0x10000
                                                + ((hi - 0xD800) << 10)
                                                + (lo - 0xDC00);
                                            s.push(
                                                char::from_u32(uni).unwrap_or('\u{FFFD}'),
                                            );
                                        } else {
                                            s.push('\u{FFFD}');
                                        }
                                    } else {
                                        s.push('\u{FFFD}');
                                    }
                                } else {
                                    s.push('\u{FFFD}');
                                }
                            } else if (0xDC00..=0xDFFF).contains(&hi) {
                                s.push('\u{FFFD}'); // lone low surrogate
                            } else {
                                s.push(char::from_u32(hi).unwrap_or('\u{FFFD}'));
                            }
                        }
                        _ => return None,
                    }
                }
                // Multi-byte UTF-8 (raw emoji / CJK inside the JSON string).
                _ => {
                    if c < 0x80 {
                        s.push(c as char);
                    } else {
                        let len = if c >= 0xF0 {
                            4
                        } else if c >= 0xE0 {
                            3
                        } else if c >= 0xC0 {
                            2
                        } else {
                            // Lone continuation byte — skip as replacement.
                            s.push('\u{FFFD}');
                            continue;
                        };
                        let start = self.i - 1;
                        let end = start + len;
                        if end <= self.b.len() {
                            if let Ok(st) = core::str::from_utf8(&self.b[start..end]) {
                                if let Some(ch) = st.chars().next() {
                                    s.push(ch);
                                    self.i = start + ch.len_utf8();
                                    continue;
                                }
                            }
                        }
                        // Truncated / invalid sequence: replace and resync.
                        s.push('\u{FFFD}');
                    }
                }
            }
        }
    }

    fn boolean(&mut self) -> Option<Json> {
        if self.b[self.i..].starts_with(b"true") {
            self.i += 4;
            Some(Json::Bool(true))
        } else if self.b[self.i..].starts_with(b"false") {
            self.i += 5;
            Some(Json::Bool(false))
        } else {
            None
        }
    }

    fn null(&mut self) -> Option<Json> {
        if self.b[self.i..].starts_with(b"null") {
            self.i += 4;
            Some(Json::Null)
        } else {
            None
        }
    }

    fn number(&mut self) -> Option<Json> {
        let start = self.i;
        while self.i < self.b.len() && matches!(self.b[self.i], b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E') {
            self.i += 1;
        }
        let s = core::str::from_utf8(&self.b[start..self.i]).ok()?;
        s.parse::<f64>().ok().map(Json::Num)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn raw_utf8_emoji_in_string() {
        let j = Json::parse(r#"{"c":"hi 😊"}"#).expect("parse");
        assert_eq!(j.get("c").and_then(|v| v.as_str()), Some("hi 😊"));
    }

    #[test_case]
    fn escaped_surrogate_pair_emoji() {
        // OpenAI-style ensure_ascii: 😊 = U+1F60A = \ud83d\ude0a
        let j = Json::parse(r#"{"c":"hi \ud83d\ude0a"}"#).expect("parse");
        let s = j.get("c").and_then(|v| v.as_str()).expect("str");
        assert!(s.contains('😊'), "got {s:?}");
        assert!(!s.contains('\u{FFFD}'), "must not be replacement char");
    }

    #[test_case]
    fn content_field_like_chat_completion() {
        let body = r#"{"choices":[{"message":{"content":"Here \ud83c\udf89"}}]}"#;
        let j = Json::parse(body).expect("parse");
        let s = j
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|a| a.first())
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|s| s.as_str())
            .expect("content");
        assert!(s.contains('🎉'), "got {s:?}");
    }
}
