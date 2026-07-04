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
                            let mut cp: u32 = 0;
                            for _ in 0..4 {
                                let h = self.peek()?;
                                self.i += 1;
                                cp = cp * 16 + (h as char).to_digit(16)?;
                            }
                            s.push(char::from_u32(cp).unwrap_or('\u{FFFD}'));
                        }
                        _ => return None,
                    }
                }
                // Multi-byte UTF-8: copy continuation bytes through as-is.
                _ => {
                    // Rebuild the char from the original bytes for correctness.
                    if c < 0x80 {
                        s.push(c as char);
                    } else {
                        // Determine sequence length and copy.
                        let len = if c >= 0xF0 {
                            4
                        } else if c >= 0xE0 {
                            3
                        } else {
                            2
                        };
                        let start = self.i - 1;
                        let end = (start + len).min(self.b.len());
                        if let Ok(st) = core::str::from_utf8(&self.b[start..end]) {
                            if let Some(ch) = st.chars().next() {
                                s.push(ch);
                                self.i = start + ch.len_utf8();
                            }
                        }
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
