//! ChittiOS: a compact backtracking `RegExp` engine (no external deps).
//!
//! Supports the common ECMAScript subset: literals, `.`, char classes `[...]`
//! (ranges, `^` negation), `\d \D \w \W \s \S \b \B`, anchors `^ $`, greedy
//! quantifiers `* + ? {n} {n,} {n,m}` (+ lazy `*?`/`+?`/`??`), alternation `|`,
//! capturing `( )` / named `(?<n> )` / non-capturing `(?: )` groups,
//! backreferences (`\1`, `\k<name>`), the escapes `\0 \xHH \uHHHH \u{…}`, and
//! the flags `i` (ignore-case), `g` (global), `m` (multiline), `s` (dotAll),
//! `y` (sticky) and `u`/`v` (unicode; code-point aware). Not supported:
//! lookahead/behind execution, unicode property escapes.
//!
//! A parse-time [`validate`] pass rejects the ECMAScript early-error
//! (`SyntaxError`) cases — bad/duplicate flags, quantifiers with no atom,
//! reversed/`u`-invalid class ranges, `u`-mode escape strictness, etc. — so an
//! invalid literal becomes a `SyntaxError` instead of being silently accepted.
//!
//! `RegExp` values are objects tagged `__builtin_name__ = "RegExp"` carrying
//! `source`/`flags` (+ a mutable `lastIndex`). Instance methods `test`/`exec`
//! and `String.prototype.{match,replace,search,split}` route here.

#[allow(unused_imports)]
use num_traits::Float;
use alloc::borrow::ToOwned;
use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use alloc::vec;
use core::sync::atomic::Ordering;
use crate::runner::ds::error::JErrorType;
use crate::runner::ds::object::JsObject;
use crate::runner::ds::object_property::{PropertyDescriptor, PropertyDescriptorData, PropertyKey};
use crate::runner::ds::value::{JsValue, JsNumberType};
use crate::runner::plugin::registry::BuiltInRegistry;
use crate::runner::plugin::types::{BuiltInObject, EvalContext};
use crate::runner::eval::expression::{get_own_prop_value, make_array, make_object, set_own_prop};

// ---------------------------------------------------------------------------
// Flags
// ---------------------------------------------------------------------------

/// Parsed regex flags, consulted by the matcher.
#[derive(Clone, Copy, Default)]
struct Flags {
    icase: bool,
    multiline: bool,
    dotall: bool,
    sticky: bool,
    global: bool,
    unicode: bool,
}

impl Flags {
    fn parse(s: &str) -> Flags {
        Flags {
            icase: s.contains('i'),
            multiline: s.contains('m'),
            dotall: s.contains('s'),
            sticky: s.contains('y'),
            global: s.contains('g'),
            unicode: s.contains('u') || s.contains('v'),
        }
    }
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

#[derive(Clone)]
enum Node {
    Char(char),
    Any,
    Class { items: Vec<(char, char)>, negate: bool, specials: Vec<char> },
    Start,
    End,
    WordB(bool), // \b (true) / \B (false)
    Group { body: Box<Node>, cap: Option<usize> },
    Backref(usize),
    Seq(Vec<Node>),
    Alt(Vec<Node>),
    Repeat { node: Box<Node>, min: usize, max: Option<usize>, greedy: bool },
}

struct Parser {
    chars: Vec<char>,
    pos: usize,
    ncaps: usize,
    unicode: bool,
    /// capture-index (1-based) → name, in source order (used for `\k<name>`).
    names: Vec<Option<String>>,
}

impl Parser {
    fn parse(pattern: &str, unicode: bool) -> Option<(Node, usize, Vec<(String, usize)>)> {
        let chars: Vec<char> = pattern.chars().collect();
        let group_names = scan_captures(&chars);
        let mut p = Parser { chars, pos: 0, ncaps: 0, unicode, names: group_names };
        let n = p.alt()?;
        if p.pos != p.chars.len() {
            return None;
        }
        // Build name → capture-index map.
        let mut map = Vec::new();
        for (i, nm) in p.names.iter().enumerate() {
            if let Some(name) = nm {
                map.push((name.clone(), i + 1));
            }
        }
        Some((n, p.ncaps, map))
    }
    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }
    fn bump(&mut self) -> Option<char> {
        let c = self.chars.get(self.pos).copied();
        if c.is_some() {
            self.pos += 1;
        }
        c
    }
    fn alt(&mut self) -> Option<Node> {
        let mut branches = vec![self.seq()?];
        while self.peek() == Some('|') {
            self.bump();
            branches.push(self.seq()?);
        }
        if branches.len() == 1 {
            Some(branches.pop().unwrap())
        } else {
            Some(Node::Alt(branches))
        }
    }
    fn seq(&mut self) -> Option<Node> {
        let mut nodes = Vec::new();
        while let Some(c) = self.peek() {
            if c == '|' || c == ')' {
                break;
            }
            nodes.push(self.quant()?);
        }
        Some(Node::Seq(nodes))
    }
    fn quant(&mut self) -> Option<Node> {
        let atom = self.atom()?;
        let (min, max) = match self.peek() {
            Some('*') => { self.bump(); (0, None) }
            Some('+') => { self.bump(); (1, None) }
            Some('?') => { self.bump(); (0, Some(1)) }
            Some('{') => {
                if let Some((mn, mx)) = self.brace() { (mn, mx) } else { return Some(atom); }
            }
            _ => return Some(atom),
        };
        let greedy = if self.peek() == Some('?') { self.bump(); false } else { true };
        Some(Node::Repeat { node: Box::new(atom), min, max, greedy })
    }
    fn brace(&mut self) -> Option<(usize, Option<usize>)> {
        let save = self.pos;
        self.bump(); // {
        let mut mn = String::new();
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() { mn.push(c); self.bump(); } else { break; }
        }
        if mn.is_empty() { self.pos = save; return None; }
        let min: usize = mn.parse().ok()?;
        let max = match self.peek() {
            Some('}') => { self.bump(); Some(min) }
            Some(',') => {
                self.bump();
                let mut mx = String::new();
                while let Some(c) = self.peek() {
                    if c.is_ascii_digit() { mx.push(c); self.bump(); } else { break; }
                }
                if self.peek() != Some('}') { self.pos = save; return None; }
                self.bump();
                if mx.is_empty() { None } else { Some(mx.parse().ok()?) }
            }
            _ => { self.pos = save; return None; }
        };
        Some((min, max))
    }
    fn atom(&mut self) -> Option<Node> {
        match self.peek()? {
            '(' => {
                self.bump();
                let cap = if self.peek() == Some('?') {
                    // (?:…) non-capturing, (?=…)/(?!…) lookahead, (?<=…)/(?<!…)
                    // lookbehind, or (?<name>…) named capture.
                    self.bump(); // ?
                    match self.peek() {
                        Some(':') => { self.bump(); None }
                        Some('<') if !matches!(self.chars.get(self.pos + 1), Some('=') | Some('!')) => {
                            // Named capture (?<name>…). The name was recorded by
                            // `scan_captures`; just skip past `<name>` here.
                            self.bump(); // <
                            while let Some(c) = self.peek() {
                                if c == '>' { break; }
                                self.bump();
                            }
                            if self.peek() == Some('>') { self.bump(); }
                            self.ncaps += 1;
                            Some(self.ncaps)
                        }
                        // Lookahead/lookbehind: parsed leniently as non-capturing
                        // (execution unsupported). Skip the assertion marker chars
                        // so the body parses as its inner disjunction.
                        Some('=') | Some('!') => { self.bump(); None }
                        Some('<') => {
                            self.bump(); // <
                            self.bump(); // = or !
                            None
                        }
                        _ => None,
                    }
                } else {
                    self.ncaps += 1;
                    Some(self.ncaps)
                };
                let body = self.alt()?;
                if self.peek() != Some(')') { return None; }
                self.bump();
                Some(Node::Group { body: Box::new(body), cap })
            }
            '[' => self.class(),
            '.' => { self.bump(); Some(Node::Any) }
            '^' => { self.bump(); Some(Node::Start) }
            '$' => { self.bump(); Some(Node::End) }
            '\\' => { self.bump(); self.escape() }
            _ => Some(Node::Char(self.bump()?)),
        }
    }
    fn class(&mut self) -> Option<Node> {
        self.bump(); // [
        let negate = if self.peek() == Some('^') { self.bump(); true } else { false };
        let mut items: Vec<(char, char)> = Vec::new();
        let mut specials: Vec<char> = Vec::new();
        while let Some(c) = self.peek() {
            if c == ']' { self.bump(); return Some(Node::Class { items, negate, specials }); }
            let lo = if c == '\\' {
                self.bump();
                let e = self.peek()?;
                match e {
                    'd' | 'D' | 'w' | 'W' | 's' | 'S' => { self.bump(); specials.push(e); continue; }
                    _ => match self.class_escape_char()? {
                        Some(ch) => ch,
                        None => continue,
                    },
                }
            } else {
                self.bump()?
            };
            if self.peek() == Some('-') && self.chars.get(self.pos + 1) != Some(&']') {
                self.bump(); // -
                // The upper bound may itself be an escape.
                let hi = if self.peek() == Some('\\') {
                    self.bump();
                    match self.peek()? {
                        'd' | 'D' | 'w' | 'W' | 's' | 'S' => {
                            // `x-\d`: `-` is a literal in Annex B (lenient); push
                            // the low bound, the dash, then let the class escape
                            // be handled on the next loop turn.
                            self.bump();
                            items.push((lo, lo));
                            items.push(('-', '-'));
                            specials.push(self.chars[self.pos - 1]);
                            continue;
                        }
                        _ => match self.class_escape_char()? {
                            Some(ch) => ch,
                            None => { items.push((lo, lo)); items.push(('-', '-')); continue; }
                        },
                    }
                } else {
                    self.bump()?
                };
                items.push((lo, hi));
            } else {
                items.push((lo, lo));
            }
        }
        None
    }
    /// Decode a `\`-escape inside a character class (the `\` already consumed,
    /// cursor on the escape char). Returns `Some(ch)` for a single code point,
    /// `None` if it produced nothing usable as a range bound.
    fn class_escape_char(&mut self) -> Option<Option<char>> {
        let c = self.bump()?;
        Some(match c {
            'n' => Some('\n'),
            't' => Some('\t'),
            'r' => Some('\r'),
            'f' => Some('\u{000C}'),
            'v' => Some('\u{000B}'),
            'b' => Some('\u{0008}'), // backspace inside a class
            '0' if !self.peek().map_or(false, |d| d.is_ascii_digit()) => Some('\0'),
            'x' => self.hex_escape(),
            'u' => self.unicode_escape(),
            other => Some(other),
        })
    }
    /// Decode `\xHH` (the `\x` already consumed). Lenient: fewer than two hex
    /// digits falls back to the literal `x`.
    fn hex_escape(&mut self) -> Option<char> {
        let save = self.pos;
        let mut v = 0u32;
        let mut n = 0;
        for _ in 0..2 {
            if let Some(d) = self.peek().and_then(|c| c.to_digit(16)) {
                v = v * 16 + d;
                self.bump();
                n += 1;
            } else {
                break;
            }
        }
        if n < 2 {
            self.pos = save;
            return Some('x');
        }
        core::char::from_u32(v).or(Some('\u{FFFD}'))
    }
    /// Decode `\uHHHH` or `\u{…}` (the `\u` already consumed). Combines a
    /// surrogate pair `\uD800-DBFF \uDC00-DFFF` into its astral code point.
    fn unicode_escape(&mut self) -> Option<char> {
        let save = self.pos;
        if self.peek() == Some('{') {
            self.bump(); // {
            let mut v = 0u32;
            let mut any = false;
            while let Some(d) = self.peek().and_then(|c| c.to_digit(16)) {
                v = v.saturating_mul(16).saturating_add(d);
                any = true;
                self.bump();
            }
            if !any || self.peek() != Some('}') || v > 0x10FFFF {
                self.pos = save;
                return Some('u');
            }
            self.bump(); // }
            return core::char::from_u32(v).or(Some('\u{FFFD}'));
        }
        let mut v = 0u32;
        for _ in 0..4 {
            if let Some(d) = self.peek().and_then(|c| c.to_digit(16)) {
                v = v * 16 + d;
                self.bump();
            } else {
                self.pos = save;
                return Some('u');
            }
        }
        // High surrogate: try to combine with a following `\uXXXX` low surrogate.
        if (0xD800..=0xDBFF).contains(&v) {
            let sp = self.pos;
            if self.peek() == Some('\\') && self.chars.get(self.pos + 1) == Some(&'u') {
                self.pos += 2;
                let mut lo = 0u32;
                let mut ok = true;
                for _ in 0..4 {
                    if let Some(d) = self.peek().and_then(|c| c.to_digit(16)) {
                        lo = lo * 16 + d;
                        self.bump();
                    } else {
                        ok = false;
                        break;
                    }
                }
                if ok && (0xDC00..=0xDFFF).contains(&lo) {
                    let cp = 0x10000 + ((v - 0xD800) << 10) + (lo - 0xDC00);
                    return core::char::from_u32(cp).or(Some('\u{FFFD}'));
                }
                self.pos = sp;
            }
        }
        core::char::from_u32(v).or(Some('\u{FFFD}'))
    }
    fn escape(&mut self) -> Option<Node> {
        let c = self.peek()?;
        Some(match c {
            'd' | 'D' | 'w' | 'W' | 's' | 'S' => { self.bump(); Node::Class { items: vec![], negate: false, specials: vec![c] } }
            'b' => { self.bump(); Node::WordB(true) }
            'B' => { self.bump(); Node::WordB(false) }
            'n' => { self.bump(); Node::Char('\n') }
            't' => { self.bump(); Node::Char('\t') }
            'r' => { self.bump(); Node::Char('\r') }
            'f' => { self.bump(); Node::Char('\u{000C}') }
            'v' => { self.bump(); Node::Char('\u{000B}') }
            'x' => { self.bump(); Node::Char(self.hex_escape().unwrap_or('x')) }
            'u' => { self.bump(); Node::Char(self.unicode_escape().unwrap_or('u')) }
            'k' if self.chars.get(self.pos + 1) == Some(&'<') => {
                // Named backreference \k<name>.
                self.bump(); // k
                self.bump(); // <
                let mut name = String::new();
                while let Some(ch) = self.peek() {
                    if ch == '>' { break; }
                    name.push(ch);
                    self.bump();
                }
                if self.peek() == Some('>') { self.bump(); }
                match self.names.iter().position(|n| n.as_deref() == Some(name.as_str())) {
                    Some(i) => Node::Backref(i + 1),
                    None => Node::Backref(usize::MAX), // never matches a real group
                }
            }
            '0' if !self.chars.get(self.pos + 1).map_or(false, |d| d.is_ascii_digit()) => {
                self.bump(); // 0
                Node::Char('\0')
            }
            '1'..='9' => {
                // Decimal backreference.
                let mut num: usize = 0;
                while let Some(d) = self.peek().and_then(|c| c.to_digit(10)) {
                    num = num.saturating_mul(10).saturating_add(d as usize);
                    self.bump();
                }
                Node::Backref(num)
            }
            other => { self.bump(); Node::Char(other) }
        })
    }
}

/// For each capturing group in source order, its name (`Some`) or `None`.
/// Non-capturing groups and lookaround assertions are skipped.
fn scan_captures(chars: &[char]) -> Vec<Option<String>> {
    let mut out = Vec::new();
    let mut i = 0;
    let mut in_class = false;
    while i < chars.len() {
        let c = chars[i];
        if c == '\\' {
            i += 2;
            continue;
        }
        if in_class {
            if c == ']' {
                in_class = false;
            }
            i += 1;
            continue;
        }
        if c == '[' {
            in_class = true;
            i += 1;
            continue;
        }
        if c == '(' {
            if chars.get(i + 1) == Some(&'?') {
                if chars.get(i + 2) == Some(&'<')
                    && !matches!(chars.get(i + 3), Some(&'=') | Some(&'!'))
                {
                    let mut j = i + 3;
                    let mut name = String::new();
                    while let Some(&ch) = chars.get(j) {
                        if ch == '>' {
                            break;
                        }
                        name.push(ch);
                        j += 1;
                    }
                    out.push(Some(name));
                }
                // else: (?: / (?= / (?! / (?<= / (?<! — not a capture.
            } else {
                out.push(None);
            }
        }
        i += 1;
    }
    out
}

fn is_word(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// ECMAScript `WhiteSpace` + `LineTerminator` (the `\s` set). Notably excludes
/// U+180E (Mongolian vowel separator), which is not whitespace since Unicode 6.3.
fn is_js_whitespace(c: char) -> bool {
    matches!(c,
        '\t' | '\u{000B}' | '\u{000C}' | ' ' | '\u{00A0}' | '\u{FEFF}'
        | '\n' | '\r' | '\u{2028}' | '\u{2029}'
        | '\u{1680}' | '\u{2000}'..='\u{200A}' | '\u{202F}' | '\u{205F}' | '\u{3000}')
}

fn is_line_terminator(c: char) -> bool {
    matches!(c, '\n' | '\r' | '\u{2028}' | '\u{2029}')
}

fn special_matches(sp: char, c: char) -> bool {
    match sp {
        'd' => c.is_ascii_digit(),
        'D' => !c.is_ascii_digit(),
        'w' => is_word(c),
        'W' => !is_word(c),
        's' => is_js_whitespace(c),
        'S' => !is_js_whitespace(c),
        _ => false,
    }
}

struct M<'a> {
    text: &'a [char],
    flags: Flags,
    caps: Vec<Option<(usize, usize)>>,
}

impl<'a> M<'a> {
    fn ceq(&self, a: char, b: char) -> bool {
        if self.flags.icase {
            if a.eq_ignore_ascii_case(&b) {
                return true;
            }
            // Full Unicode simple case fold only applies under the `u`/`v` flag
            // (e.g. KELVIN SIGN U+212A ↔ 'k'); a non-unicode `i` match uses the
            // ASCII-only canonicalization above.
            self.flags.unicode && a.to_lowercase().eq(b.to_lowercase())
        } else {
            a == b
        }
    }
    /// Match `node` then continue matching via the continuation `k`. Returns the
    /// end index on success.
    fn m(&mut self, node: &Node, pos: usize, k: &dyn Fn(&mut M, usize) -> Option<usize>) -> Option<usize> {
        // The backtracking matcher itself: one pattern can explore
        // exponentially many paths from a single start position, so the tick
        // belongs here too, not only in the scan that calls it.
        //
        // It cannot return an error (the signature is `Option`), so it raises a
        // flag and unwinds; the public entry points turn that into the real
        // interrupt. Returning a bare `None` would be worse than the hang it
        // fixes: the regex would quietly report "no match" for a match that
        // exists.
        if INTERRUPTED.load(Ordering::Relaxed) {
            return None;
        }
        if crate::runner::host::host_tick() {
            INTERRUPTED.store(true, Ordering::Relaxed);
            return None;
        }
        match node {
            Node::Char(c) => {
                if pos < self.text.len() && self.ceq(self.text[pos], *c) {
                    k(self, pos + 1)
                } else {
                    None
                }
            }
            Node::Any => {
                if pos < self.text.len() && (self.flags.dotall || !is_line_terminator(self.text[pos])) {
                    k(self, pos + 1)
                } else {
                    None
                }
            }
            Node::Class { items, negate, specials } => {
                if pos >= self.text.len() {
                    return None;
                }
                let c = self.text[pos];
                let mut hit = items.iter().any(|(lo, hi)| {
                    if self.flags.icase {
                        let cl = c.to_ascii_lowercase();
                        (cl >= lo.to_ascii_lowercase() && cl <= hi.to_ascii_lowercase())
                            || (c >= *lo && c <= *hi)
                    } else {
                        c >= *lo && c <= *hi
                    }
                });
                if !hit {
                    hit = specials.iter().any(|sp| special_matches(*sp, c));
                }
                if hit != *negate {
                    k(self, pos + 1)
                } else {
                    None
                }
            }
            Node::Start => {
                if pos == 0 || (self.flags.multiline && pos > 0 && is_line_terminator(self.text[pos - 1])) {
                    k(self, pos)
                } else {
                    None
                }
            }
            Node::End => {
                if pos == self.text.len() || (self.flags.multiline && is_line_terminator(self.text[pos])) {
                    k(self, pos)
                } else {
                    None
                }
            }
            Node::WordB(want) => {
                let before = pos > 0 && is_word(self.text[pos - 1]);
                let after = pos < self.text.len() && is_word(self.text[pos]);
                let boundary = before != after;
                if boundary == *want { k(self, pos) } else { None }
            }
            Node::Backref(n) => {
                let span = if *n >= 1 && *n <= self.caps.len() { self.caps[*n - 1] } else { None };
                match span {
                    // An unmatched (or undefined) group backreference matches the
                    // empty string (ECMAScript semantics).
                    None => k(self, pos),
                    Some((s, e)) => {
                        let len = e - s;
                        if pos + len > self.text.len() {
                            return None;
                        }
                        for j in 0..len {
                            if !self.ceq(self.text[pos + j], self.text[s + j]) {
                                return None;
                            }
                        }
                        k(self, pos + len)
                    }
                }
            }
            Node::Group { body, cap } => {
                let cap = *cap;
                let start = pos;
                let inner_k = move |m: &mut M, end: usize| -> Option<usize> {
                    if let Some(ci) = cap {
                        if ci <= m.caps.len() && ci >= 1 {
                            m.caps[ci - 1] = Some((start, end));
                        }
                    }
                    k(m, end)
                };
                self.m(body, pos, &inner_k)
            }
            Node::Seq(nodes) => self.seq(nodes, 0, pos, k),
            Node::Alt(branches) => {
                for b in branches {
                    if let Some(e) = self.m(b, pos, k) {
                        return Some(e);
                    }
                }
                None
            }
            Node::Repeat { node, min, max, greedy } => {
                self.repeat(node, *min, *max, *greedy, 0, pos, k)
            }
        }
    }
    fn seq(&mut self, nodes: &[Node], i: usize, pos: usize, k: &dyn Fn(&mut M, usize) -> Option<usize>) -> Option<usize> {
        if i >= nodes.len() {
            return k(self, pos);
        }
        let rest_k = move |m: &mut M, p: usize| -> Option<usize> { m.seq(nodes, i + 1, p, k) };
        self.m(&nodes[i], pos, &rest_k)
    }
    fn repeat(&mut self, node: &Node, min: usize, max: Option<usize>, greedy: bool, count: usize, pos: usize, k: &dyn Fn(&mut M, usize) -> Option<usize>) -> Option<usize> {
        let can_more = max.map_or(true, |mx| count < mx);
        let try_more = |m: &mut M| -> Option<usize> {
            if !can_more {
                return None;
            }
            let more_k = move |m2: &mut M, p: usize| -> Option<usize> {
                if p == pos {
                    // zero-width match guard: stop to avoid infinite loop
                    return None;
                }
                m2.repeat(node, min, max, greedy, count + 1, p, k)
            };
            m.m(node, pos, &more_k)
        };
        let try_stop = |m: &mut M| -> Option<usize> {
            if count >= min {
                k(m, pos)
            } else {
                None
            }
        };
        if greedy {
            try_more(self).or_else(|| try_stop(self))
        } else {
            try_stop(self).or_else(|| try_more(self))
        }
    }
}

/// Result of a match: (start, end, capture spans, name→capture-index).
struct MatchResult {
    start: usize,
    end: usize,
    caps: Vec<Option<(usize, usize)>>,
    names: Vec<(String, usize)>,
}

/// Search `text` for `pattern` starting at `from`. Returns the leftmost match
/// (or, when `flags.sticky`, only a match anchored exactly at `from`).
fn search(pattern: &str, flags: Flags, text: &[char], from: usize) -> Option<MatchResult> {
    let (node, ncaps, names) = Parser::parse(pattern, flags.unicode)?;
    // A leading `^` anchors the search to index 0 — but only when `m` is off;
    // under `m`, `^` also matches at every line start, so the scan must proceed.
    let anchored = !flags.multiline
        && matches!(&node, Node::Seq(v) if v.first().map_or(false, |n| matches!(n, Node::Start)));
    let mut start = from;
    loop {
        // A regex scan is native code driven by page input, and a
        // catastrophically-backtracking pattern has no other bound. Without a
        // tick here the interpreter's hook never runs, so the kernel stops
        // answering Ctrl+C and the script budget for as long as the match takes.
        if INTERRUPTED.load(Ordering::Relaxed) {
            return None;
        }
        if crate::runner::host::host_tick() {
            INTERRUPTED.store(true, Ordering::Relaxed);
            return None;
        }
        if start > text.len() {
            return None;
        }
        let mut m = M { text, flags, caps: vec![None; ncaps] };
        let end_k = |_m: &mut M, p: usize| -> Option<usize> { Some(p) };
        if let Some(end) = m.m(&node, start, &end_k) {
            return Some(MatchResult { start, end, caps: m.caps, names });
        }
        if anchored || flags.sticky {
            return None;
        }
        start += 1;
    }
}

// ---------------------------------------------------------------------------
// Parse-time validation (ECMAScript early errors → SyntaxError)
// ---------------------------------------------------------------------------

fn is_syntax_char(c: char) -> bool {
    matches!(c, '^' | '$' | '\\' | '.' | '*' | '+' | '?' | '(' | ')' | '[' | ']' | '{' | '}' | '|')
}

/// Validate regex `flags`: only `d g i m s u v y`, no duplicates, `u`/`v`
/// mutually exclusive.
fn validate_flags(flags: &str) -> Result<(), String> {
    let mut seen = [false; 128];
    for c in flags.chars() {
        if !matches!(c, 'd' | 'g' | 'i' | 'm' | 's' | 'u' | 'v' | 'y') {
            return Err(alloc::format!("Invalid regular expression flag '{}'", c));
        }
        let idx = c as usize;
        if seen[idx] {
            return Err(alloc::format!("Duplicate regular expression flag '{}'", c));
        }
        seen[idx] = true;
    }
    if flags.contains('u') && flags.contains('v') {
        return Err("Invalid regular expression flags: 'u' and 'v' are mutually exclusive".to_string());
    }
    Ok(())
}

/// Validate a regex `pattern` + `flags` at parse time, returning `Err(msg)` for
/// the ECMAScript early-error (`SyntaxError`) cases and `Ok(())` otherwise.
pub fn validate(pattern: &str, flags: &str) -> Result<(), String> {
    validate_flags(flags)?;
    let unicode = flags.contains('u') || flags.contains('v');
    let chars: Vec<char> = pattern.chars().collect();
    let ncap = scan_captures(&chars).len();
    let mut v = ReValidator { chars: &chars, pos: 0, u: unicode, ncap };
    v.disjunction()?;
    if v.pos != v.chars.len() {
        return Err("Unmatched ')' in regular expression".to_string());
    }
    Ok(())
}

#[derive(Clone, Copy, PartialEq)]
enum AtomKind {
    Normal,
    Lookahead,
    Lookbehind,
    WordBoundary,
}

struct ReValidator<'a> {
    chars: &'a [char],
    pos: usize,
    u: bool,
    ncap: usize,
}

impl<'a> ReValidator<'a> {
    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }
    fn at(&self, o: usize) -> Option<char> {
        self.chars.get(self.pos + o).copied()
    }
    fn bump(&mut self) -> Option<char> {
        let c = self.peek();
        if c.is_some() {
            self.pos += 1;
        }
        c
    }
    fn disjunction(&mut self) -> Result<(), String> {
        self.alternative()?;
        while self.peek() == Some('|') {
            self.bump();
            self.alternative()?;
        }
        Ok(())
    }
    fn alternative(&mut self) -> Result<(), String> {
        loop {
            match self.peek() {
                None | Some('|') | Some(')') => break,
                _ => self.term()?,
            }
        }
        Ok(())
    }
    /// True if a valid quantifier prefix `{n}` / `{n,}` / `{n,m}` starts at
    /// `chars[i]` (which must be `{`).
    fn is_quantifier_prefix_at(&self, i: usize) -> bool {
        let mut j = i + 1;
        let mut saw = false;
        while self.chars.get(j).map_or(false, |c| c.is_ascii_digit()) {
            j += 1;
            saw = true;
        }
        if !saw {
            return false;
        }
        match self.chars.get(j) {
            Some('}') => true,
            Some(',') => {
                j += 1;
                while self.chars.get(j).map_or(false, |c| c.is_ascii_digit()) {
                    j += 1;
                }
                self.chars.get(j) == Some(&'}')
            }
            _ => false,
        }
    }
    fn term(&mut self) -> Result<(), String> {
        match self.peek() {
            Some('*') | Some('+') | Some('?') => {
                return Err("Nothing to repeat".to_string());
            }
            Some('{') => {
                if self.is_quantifier_prefix_at(self.pos) {
                    return Err("Nothing to repeat".to_string());
                }
                // A `{` that is not a quantifier: a literal in Annex B, a
                // SyntaxError under `u`.
                if self.u {
                    return Err("Lone quantifier bracket".to_string());
                }
                self.bump();
                return Ok(());
            }
            Some('}') | Some(']') if self.u => {
                // Lone `}`/`]` is a SyntaxError under `u`.
                return Err("Lone bracket in unicode mode".to_string());
            }
            _ => {}
        }
        let kind = self.assertion_or_atom()?;
        let quantified = self.quantifier()?;
        if quantified {
            match kind {
                AtomKind::Lookbehind => {
                    return Err("Lookbehind assertion cannot be quantified".to_string());
                }
                AtomKind::Lookahead if self.u => {
                    return Err("Lookahead assertion cannot be quantified in unicode mode".to_string());
                }
                AtomKind::WordBoundary if self.u => {
                    return Err("Assertion cannot be quantified in unicode mode".to_string());
                }
                _ => {}
            }
        }
        Ok(())
    }
    /// Consume an optional quantifier (`* + ?` or a valid `{…}` prefix, plus a
    /// lazy `?`). Returns whether one was present.
    fn quantifier(&mut self) -> Result<bool, String> {
        let consumed = match self.peek() {
            Some('*') | Some('+') | Some('?') => {
                self.bump();
                true
            }
            Some('{') if self.is_quantifier_prefix_at(self.pos) => {
                self.bump(); // {
                while self.peek().map_or(false, |c| c.is_ascii_digit()) {
                    self.bump();
                }
                if self.peek() == Some(',') {
                    self.bump();
                    while self.peek().map_or(false, |c| c.is_ascii_digit()) {
                        self.bump();
                    }
                }
                self.bump(); // }
                true
            }
            _ => false,
        };
        if consumed && self.peek() == Some('?') {
            self.bump();
        }
        Ok(consumed)
    }
    fn assertion_or_atom(&mut self) -> Result<AtomKind, String> {
        match self.peek() {
            Some('^') | Some('$') => {
                self.bump();
                Ok(AtomKind::Normal)
            }
            _ => self.atom(),
        }
    }
    fn atom(&mut self) -> Result<AtomKind, String> {
        match self.peek() {
            Some('.') => {
                self.bump();
                Ok(AtomKind::Normal)
            }
            Some('\\') => {
                self.bump();
                self.atom_escape()
            }
            Some('[') => {
                self.char_class()?;
                Ok(AtomKind::Normal)
            }
            Some('(') => self.group(),
            Some(_) => {
                self.bump();
                Ok(AtomKind::Normal)
            }
            None => Err("Unexpected end of pattern".to_string()),
        }
    }
    fn group(&mut self) -> Result<AtomKind, String> {
        self.bump(); // (
        let mut kind = AtomKind::Normal;
        if self.peek() == Some('?') {
            self.bump(); // ?
            match self.peek() {
                Some(':') => {
                    self.bump();
                }
                Some('=') | Some('!') => {
                    self.bump();
                    kind = AtomKind::Lookahead;
                }
                Some('<') if matches!(self.at(1), Some('=') | Some('!')) => {
                    self.bump(); // <
                    self.bump(); // = or !
                    kind = AtomKind::Lookbehind;
                }
                Some('<') => {
                    // (?<name>…) — name validity is checked elsewhere; consume it.
                    self.bump(); // <
                    while let Some(c) = self.peek() {
                        if c == '>' {
                            break;
                        }
                        self.bump();
                    }
                    if self.peek() != Some('>') {
                        return Err("Invalid named group".to_string());
                    }
                    self.bump(); // >
                }
                _ => {
                    // Modifiers group `(?ims-ims:…)` — validated elsewhere;
                    // consume leniently up to `:`.
                    while let Some(c) = self.peek() {
                        if c == ':' {
                            self.bump();
                            break;
                        }
                        if c == ')' {
                            break;
                        }
                        self.bump();
                    }
                }
            }
        }
        self.disjunction()?;
        if self.peek() != Some(')') {
            return Err("Unterminated group".to_string());
        }
        self.bump(); // )
        Ok(kind)
    }
    fn atom_escape(&mut self) -> Result<AtomKind, String> {
        let c = match self.bump() {
            Some(c) => c,
            None => return Err("Trailing backslash in regular expression".to_string()),
        };
        match c {
            'b' | 'B' => Ok(AtomKind::WordBoundary),
            'd' | 'D' | 'w' | 'W' | 's' | 'S' => Ok(AtomKind::Normal),
            'f' | 'n' | 'r' | 't' | 'v' => Ok(AtomKind::Normal),
            'c' => {
                match self.peek() {
                    Some(l) if l.is_ascii_alphabetic() => {
                        self.bump();
                        Ok(AtomKind::Normal)
                    }
                    _ => {
                        if self.u {
                            Err("Invalid control escape in unicode mode".to_string())
                        } else {
                            Ok(AtomKind::Normal)
                        }
                    }
                }
            }
            'x' => {
                if self.u {
                    for _ in 0..2 {
                        if !self.peek().map_or(false, |d| d.is_ascii_hexdigit()) {
                            return Err("Invalid hexadecimal escape in unicode mode".to_string());
                        }
                        self.bump();
                    }
                } else {
                    for _ in 0..2 {
                        if self.peek().map_or(false, |d| d.is_ascii_hexdigit()) {
                            self.bump();
                        }
                    }
                }
                Ok(AtomKind::Normal)
            }
            'u' => {
                self.validate_unicode_escape()?;
                Ok(AtomKind::Normal)
            }
            'p' | 'P' if self.u => {
                if self.peek() == Some('{') {
                    while let Some(ch) = self.bump() {
                        if ch == '}' {
                            break;
                        }
                    }
                    Ok(AtomKind::Normal)
                } else {
                    Err("Invalid property escape".to_string())
                }
            }
            'k' => {
                if self.peek() == Some('<') {
                    self.bump();
                    while let Some(ch) = self.peek() {
                        if ch == '>' {
                            break;
                        }
                        self.bump();
                    }
                    if self.peek() == Some('>') {
                        self.bump();
                    }
                    Ok(AtomKind::Normal)
                } else if self.u {
                    Err("Invalid \\k escape in unicode mode".to_string())
                } else {
                    Ok(AtomKind::Normal)
                }
            }
            '0' => {
                if self.peek().map_or(false, |d| d.is_ascii_digit()) {
                    if self.u {
                        return Err("Invalid legacy octal escape in unicode mode".to_string());
                    }
                    while self.peek().map_or(false, |d| ('0'..='7').contains(&d)) {
                        self.bump();
                    }
                }
                Ok(AtomKind::Normal)
            }
            '1'..='9' => {
                let mut num = c.to_digit(10).unwrap() as usize;
                while let Some(d) = self.peek().and_then(|c| c.to_digit(10)) {
                    num = num * 10 + d as usize;
                    self.bump();
                }
                if self.u && num > self.ncap {
                    return Err("Invalid backreference in unicode mode".to_string());
                }
                Ok(AtomKind::Normal)
            }
            other => {
                if self.u && !is_syntax_char(other) && other != '/' {
                    return Err(alloc::format!("Invalid identity escape '\\{}' in unicode mode", other));
                }
                Ok(AtomKind::Normal)
            }
        }
    }
    /// Validate `\uHHHH` / `\u{…}` (leading `\u` already consumed).
    fn validate_unicode_escape(&mut self) -> Result<(), String> {
        if self.peek() == Some('{') {
            self.bump(); // {
            let mut v: u32 = 0;
            let mut any = false;
            loop {
                match self.peek() {
                    Some('}') => break,
                    Some(d) if d.is_ascii_hexdigit() => {
                        v = v.saturating_mul(16).saturating_add(d.to_digit(16).unwrap());
                        any = true;
                        self.bump();
                    }
                    _ => {
                        if self.u {
                            return Err("Invalid unicode code point escape".to_string());
                        }
                        return Ok(());
                    }
                }
            }
            if self.peek() != Some('}') {
                if self.u {
                    return Err("Unterminated unicode code point escape".to_string());
                }
                return Ok(());
            }
            self.bump(); // }
            if self.u && !any {
                return Err("Empty unicode code point escape".to_string());
            }
            if v > 0x10FFFF {
                return Err("Unicode code point out of range".to_string());
            }
            return Ok(());
        }
        let mut cnt = 0;
        for _ in 0..4 {
            if self.peek().map_or(false, |d| d.is_ascii_hexdigit()) {
                self.bump();
                cnt += 1;
            } else {
                break;
            }
        }
        if cnt < 4 && self.u {
            return Err("Invalid unicode escape in unicode mode".to_string());
        }
        Ok(())
    }
    fn char_class(&mut self) -> Result<(), String> {
        self.bump(); // [
        if self.peek() == Some('^') {
            self.bump();
        }
        loop {
            match self.peek() {
                None => return Err("Unterminated character class".to_string()),
                Some(']') => {
                    self.bump();
                    return Ok(());
                }
                _ => {}
            }
            let (lo_val, lo_class) = self.class_atom()?;
            if self.peek() == Some('-') && self.at(1) != Some(']') && self.at(1).is_some() {
                self.bump(); // -
                let (hi_val, hi_class) = self.class_atom()?;
                if self.u && (lo_class || hi_class) {
                    return Err("Invalid character class range in unicode mode".to_string());
                }
                if !lo_class && !hi_class {
                    if let (Some(a), Some(b)) = (lo_val, hi_val) {
                        if (a as u32) > (b as u32) {
                            return Err("Range out of order in character class".to_string());
                        }
                    }
                }
            }
        }
    }
    /// Parse one class atom; returns `(value, is_class_escape)` where a class
    /// escape (`\d \w \s` …) cannot form a range bound.
    fn class_atom(&mut self) -> Result<(Option<char>, bool), String> {
        let c = self.peek().unwrap();
        if c != '\\' {
            self.bump();
            return Ok((Some(c), false));
        }
        self.bump(); // backslash
        let e = match self.bump() {
            Some(e) => e,
            None => return Err("Trailing backslash in character class".to_string()),
        };
        match e {
            'd' | 'D' | 'w' | 'W' | 's' | 'S' => Ok((None, true)),
            'b' => Ok((Some('\u{0008}'), false)),
            'f' => Ok((Some('\u{000C}'), false)),
            'n' => Ok((Some('\n'), false)),
            'r' => Ok((Some('\r'), false)),
            't' => Ok((Some('\t'), false)),
            'v' => Ok((Some('\u{000B}'), false)),
            'x' => {
                let mut v = 0u32;
                let mut n = 0;
                for _ in 0..2 {
                    if let Some(d) = self.peek().and_then(|c| c.to_digit(16)) {
                        v = v * 16 + d;
                        self.bump();
                        n += 1;
                    } else {
                        break;
                    }
                }
                if n < 2 && self.u {
                    return Err("Invalid hexadecimal escape in unicode mode".to_string());
                }
                Ok((core::char::from_u32(v), false))
            }
            'u' => {
                self.validate_unicode_escape()?;
                Ok((None, false))
            }
            'c' => match self.peek() {
                Some(l) if l.is_ascii_alphabetic() => {
                    self.bump();
                    Ok((None, false))
                }
                _ => {
                    if self.u {
                        Err("Invalid control escape in unicode mode".to_string())
                    } else {
                        Ok((Some('c'), false))
                    }
                }
            },
            'p' | 'P' if self.u => {
                if self.peek() == Some('{') {
                    while let Some(ch) = self.bump() {
                        if ch == '}' {
                            break;
                        }
                    }
                    Ok((None, true))
                } else {
                    Err("Invalid property escape".to_string())
                }
            }
            '0' => {
                if self.peek().map_or(false, |d| d.is_ascii_digit()) {
                    if self.u {
                        return Err("Invalid legacy octal escape in unicode mode".to_string());
                    }
                    while self.peek().map_or(false, |d| ('0'..='7').contains(&d)) {
                        self.bump();
                    }
                    return Ok((None, false));
                }
                Ok((Some('\0'), false))
            }
            '1'..='9' => {
                if self.u {
                    return Err("Invalid class escape in unicode mode".to_string());
                }
                Ok((Some(e), false))
            }
            other => {
                if self.u && !is_syntax_char(other) && other != '/' && other != '-' {
                    return Err(alloc::format!("Invalid identity escape '\\{}' in character class", other));
                }
                Ok((Some(other), false))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// RegExp built-in + string integration
// ---------------------------------------------------------------------------

/// Insert a data property with explicit attributes.
fn define_data_prop(v: &JsValue, name: &str, value: JsValue, writable: bool, enumerable: bool, configurable: bool) {
    if let JsValue::Object(o) = v {
        let mut b = o.borrow_mut();
        b.as_js_object_mut().get_object_base_mut().properties.insert(
            PropertyKey::Str(name.to_string()),
            PropertyDescriptor::Data(PropertyDescriptorData {
                value,
                writable,
                enumerable,
                configurable,
            }),
        );
    }
}

/// Build a RegExp value.
pub fn make_regexp(source: &str, flags: &str) -> JsValue {
    let obj = make_object(vec![]);
    set_own_prop(&obj, "__builtin_name__", JsValue::String("RegExp".to_string()), false);
    set_own_prop(&obj, "source", JsValue::String(source.to_string()), true);
    set_own_prop(&obj, "flags", JsValue::String(flags.to_string()), true);
    set_own_prop(&obj, "global", JsValue::Boolean(flags.contains('g')), true);
    set_own_prop(&obj, "ignoreCase", JsValue::Boolean(flags.contains('i')), true);
    set_own_prop(&obj, "multiline", JsValue::Boolean(flags.contains('m')), true);
    set_own_prop(&obj, "dotAll", JsValue::Boolean(flags.contains('s')), true);
    set_own_prop(&obj, "sticky", JsValue::Boolean(flags.contains('y')), true);
    set_own_prop(&obj, "unicode", JsValue::Boolean(flags.contains('u')), true);
    // `lastIndex` is writable but non-enumerable and non-configurable.
    define_data_prop(&obj, "lastIndex", JsValue::Number(JsNumberType::Integer(0)), true, false, false);
    obj
}

fn re_source(v: &JsValue) -> Option<(String, String)> {
    let src = match get_own_prop_value(v, "source")? {
        JsValue::String(s) => s,
        _ => return None,
    };
    let flags = match get_own_prop_value(v, "flags") {
        Some(JsValue::String(s)) => s,
        _ => String::new(),
    };
    Some((src, flags))
}

pub fn register(registry: &mut BuiltInRegistry) {
    let re = BuiltInObject::new("RegExp")
        .with_constructor(regexp_constructor)
        .add_method("test", regexp_test)
        .add_method("exec", regexp_exec)
        .add_method("toString", regexp_to_string);
    registry.register_object(re);
}

fn regexp_constructor(
    _ctx: &mut EvalContext,
    _this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let src = match args.first() {
        Some(JsValue::String(s)) => s.clone(),
        Some(v) if get_own_prop_value(v, "source").is_some() => {
            match get_own_prop_value(v, "source") {
                Some(JsValue::String(s)) => s,
                _ => String::new(),
            }
        }
        _ => String::new(),
    };
    let flags = match args.get(1) {
        Some(JsValue::String(s)) => s.clone(),
        _ => String::new(),
    };
    if let Err(msg) = validate(&src, &flags) {
        return Err(JErrorType::SyntaxError(msg));
    }
    Ok(make_regexp(&src, &flags))
}

fn regexp_to_string(
    _ctx: &mut EvalContext,
    this: JsValue,
    _args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let (src, flags) = re_source(&this).unwrap_or_default();
    Ok(JsValue::String(alloc::format!("/{}/{}", src, flags)))
}

fn str_arg(args: &[JsValue]) -> String {
    match args.first() {
        Some(JsValue::String(s)) => s.clone(),
        Some(other) => other.to_string(),
        None => "undefined".to_string(),
    }
}

/// Read `lastIndex` as a non-negative integer.
fn read_last_index(this: &JsValue) -> usize {
    match get_own_prop_value(this, "lastIndex") {
        Some(JsValue::Number(JsNumberType::Integer(n))) => if n < 0 { 0 } else { n as usize },
        Some(JsValue::Number(JsNumberType::Float(f))) => if f < 0.0 { 0 } else { f as usize },
        _ => 0,
    }
}

fn write_last_index(this: &JsValue, n: usize) {
    define_data_prop(this, "lastIndex", JsValue::Number(JsNumberType::Integer(n as i64)), true, false, false);
}

/// Shared `exec`/`test` engine: honours `g`/`y` `lastIndex`, sticky anchoring,
/// and updates `lastIndex` per spec. Returns the match (if any).
fn run_match(this: &JsValue, text: &[char]) -> Option<MatchResult> {
    let (src, flagstr) = re_source(this)?;
    let flags = Flags::parse(&flagstr);
    let uses_index = flags.global || flags.sticky;
    let start = if uses_index { read_last_index(this) } else { 0 };
    if start > text.len() {
        if uses_index {
            write_last_index(this, 0);
        }
        return None;
    }
    let res = search(&src, flags, text, start);
    if uses_index {
        match &res {
            Some(mr) => write_last_index(this, mr.end),
            None => write_last_index(this, 0),
        }
    }
    res
}

fn regexp_test(
    _ctx: &mut EvalContext,
    this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    if re_source(&this).is_none() {
        return Err(JErrorType::TypeError("not a RegExp".to_string()));
    }
    let text: Vec<char> = str_arg(&args).chars().collect();
    Ok(JsValue::Boolean(run_match(&this, &text).is_some()))
}

fn regexp_exec(
    _ctx: &mut EvalContext,
    this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    if re_source(&this).is_none() {
        return Err(JErrorType::TypeError("not a RegExp".to_string()));
    }
    let text: Vec<char> = str_arg(&args).chars().collect();
    match run_match(&this, &text) {
        Some(mr) => Ok(exec_result(&text, &mr)),
        None => Ok(JsValue::Null),
    }
}

fn span_str(text: &[char], s: usize, e: usize) -> String {
    text[s..e].iter().collect()
}

/// Build the `exec` result array: [fullMatch, group1, …] with `.index` and,
/// for named patterns, a `.groups` object.
fn exec_result(text: &[char], mr: &MatchResult) -> JsValue {
    let mut items = vec![JsValue::String(span_str(text, mr.start, mr.end))];
    for c in &mr.caps {
        match c {
            Some((s, e)) => items.push(JsValue::String(span_str(text, *s, *e))),
            None => items.push(JsValue::Undefined),
        }
    }
    let arr = make_array(items);
    set_own_prop(&arr, "index", JsValue::Number(JsNumberType::Integer(mr.start as i64)), true);
    if mr.names.is_empty() {
        set_own_prop(&arr, "groups", JsValue::Undefined, true);
    } else {
        let groups = make_object(vec![]);
        for (name, ci) in &mr.names {
            let val = match mr.caps.get(*ci - 1).copied().flatten() {
                Some((s, e)) => JsValue::String(span_str(text, s, e)),
                None => JsValue::Undefined,
            };
            set_own_prop(&groups, name, val, true);
        }
        set_own_prop(&arr, "groups", groups, true);
    }
    arr
}

/// `str.match(re)` — array of the match (+ groups), or all matches with `g`.
pub fn string_match(text: &str, re: &JsValue) -> JsValue {
    let Some((src, flagstr)) = re_source(re) else { return JsValue::Null };
    let chars: Vec<char> = text.chars().collect();
    let flags = Flags::parse(&flagstr);
    if flags.global {
        let mut out = Vec::new();
        let mut from = 0;
        while let Some(mr) = search(&src, flags, &chars, from) {
            out.push(JsValue::String(span_str(&chars, mr.start, mr.end)));
            from = if mr.end > mr.start { mr.end } else { mr.end + 1 };
        }
        if out.is_empty() { JsValue::Null } else { make_array(out) }
    } else {
        match search(&src, flags, &chars, 0) {
            Some(mr) => exec_result(&chars, &mr),
            None => JsValue::Null,
        }
    }
}

/// `str.replace(re, replacement)` — supports `$1..$9`, `$&`. `g` flag → all.
pub fn string_replace_regexp(text: &str, re: &JsValue, replacement: &str) -> String {
    let Some((src, flagstr)) = re_source(re) else { return text.to_string() };
    let chars: Vec<char> = text.chars().collect();
    let flags = Flags::parse(&flagstr);
    let global = flags.global;
    let mut out = String::new();
    let mut from = 0usize;
    loop {
        if INTERRUPTED.load(Ordering::Relaxed) || crate::runner::host::host_tick() {
            INTERRUPTED.store(true, Ordering::Relaxed);
            break;
        }
        match search(&src, flags, &chars, from) {
            Some(mr) => {
                out.extend(chars[from..mr.start].iter());
                out.push_str(&expand_replacement(replacement, &chars, &mr));
                let next = if mr.end > mr.start { mr.end } else { mr.end + 1 };
                if mr.end == mr.start && mr.end < chars.len() {
                    out.push(chars[mr.end]);
                }
                from = next;
                if !global || from > chars.len() {
                    break;
                }
            }
            None => break,
        }
    }
    if from <= chars.len() {
        out.extend(chars[from.min(chars.len())..].iter());
    }
    out
}

fn expand_replacement(repl: &str, text: &[char], mr: &MatchResult) -> String {
    let rc: Vec<char> = repl.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < rc.len() {
        if rc[i] == '$' && i + 1 < rc.len() {
            let n = rc[i + 1];
            if n == '&' {
                out.push_str(&span_str(text, mr.start, mr.end));
                i += 2;
                continue;
            } else if n.is_ascii_digit() {
                let gi = n as usize - '0' as usize;
                if gi >= 1 && gi <= mr.caps.len() {
                    if let Some((s, e)) = mr.caps[gi - 1] {
                        out.push_str(&span_str(text, s, e));
                    }
                    i += 2;
                    continue;
                }
            } else if n == '$' {
                out.push('$');
                i += 2;
                continue;
            }
        }
        out.push(rc[i]);
        i += 1;
    }
    out
}

/// `str.split(re)` — split on regex-separator matches.
pub fn string_split_regexp(text: &str, re: &JsValue) -> Vec<JsValue> {
    let Some((src, flagstr)) = re_source(re) else { return vec![JsValue::String(text.to_string())] };
    let chars: Vec<char> = text.chars().collect();
    let flags = Flags::parse(&flagstr);
    let mut out = Vec::new();
    let mut last = 0usize;
    let mut from = 0usize;
    while from <= chars.len() {
        match search(&src, flags, &chars, from) {
            Some(mr) if mr.end > mr.start => {
                out.push(JsValue::String(span_str(&chars, last, mr.start)));
                last = mr.end;
                from = mr.end;
            }
            _ => break,
        }
    }
    out.push(JsValue::String(span_str(&chars, last, chars.len())));
    out
}

/// Raised when the host asked to stop *inside* the matcher, which cannot
/// return an error. Public entry points call [`take_interrupt`] and turn it
/// into the interpreter's uncatchable interrupt.
static INTERRUPTED: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

/// True (once) if the matcher was interrupted; clears the flag.
pub fn take_interrupt() -> bool {
    INTERRUPTED.swap(false, Ordering::Relaxed)
}

/// `str.search(re)` — index of first match or -1.
pub fn string_search(text: &str, re: &JsValue) -> i64 {
    let Some((src, flagstr)) = re_source(re) else { return -1 };
    let chars: Vec<char> = text.chars().collect();
    let flags = Flags::parse(&flagstr);
    match search(&src, flags, &chars, 0) {
        Some(mr) => mr.start as i64,
        None => -1,
    }
}

/// Is `v` a RegExp value?
pub fn is_regexp(v: &JsValue) -> bool {
    matches!(get_own_prop_value(v, "__builtin_name__"), Some(JsValue::String(ref s)) if s == "RegExp")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn v(pattern: &str, flags: &str) -> bool {
        validate(pattern, flags).is_ok()
    }

    fn m(pattern: &str, flags: &str, text: &str) -> bool {
        let f = Flags::parse(flags);
        let chars: Vec<char> = text.chars().collect();
        search(pattern, f, &chars, 0).is_some()
    }

    #[test]
    fn validate_flags_ok_and_errors() {
        assert!(v(".", "gimsuy"));
        assert!(!v(".", "G")); // unknown
        assert!(!v(".", "gg")); // duplicate
        assert!(!v(".", "uv")); // mutually exclusive
        assert!(v(".", "gi"));
    }

    #[test]
    fn validate_quantifier_no_atom() {
        assert!(!v("{2}", ""));
        assert!(!v("{2,}", ""));
        assert!(!v("{2,3}", ""));
        assert!(!v("*", ""));
        assert!(!v("+", ""));
        assert!(v("a{2}", ""));
        assert!(v("a*", ""));
        // A lone `{` is a literal in Annex B but an error under `u`.
        assert!(v("{", ""));
        assert!(!v("{", "u"));
    }

    #[test]
    fn validate_class_ranges() {
        assert!(!v("[b-a]", ""));
        assert!(v("[a-b]", ""));
        assert!(v("[a-a]", ""));
        assert!(!v("[\\d-a]", "u"));
        assert!(!v("[a-\\d]", "u"));
        assert!(!v("[--\\d]", "u"));
        // Lenient in Annex B (non-u).
        assert!(v("[\\d-a]", ""));
    }

    #[test]
    fn validate_u_mode_escapes() {
        assert!(!v("\\M", "u")); // invalid identity escape
        assert!(v("\\M", "")); // fine in Annex B
        assert!(!v("\\1", "u")); // out-of-bounds backref
        assert!(v("(a)\\1", "u")); // valid backref
        assert!(!v("\\8", "u")); // oob decimal escape
        assert!(!v("\\u{110000}", "u")); // out of range
        assert!(!v("\\u{1,}", "u")); // non-hex
        assert!(v("\\u{1F600}", "u"));
        assert!(!v("\\c0", "u")); // invalid control escape
        assert!(!v("(?<a>\\a)", "u")); // invalid identity escape in capture
    }

    #[test]
    fn validate_quantified_assertions() {
        assert!(!v(".(?<=.)?", "")); // quantified lookbehind (any mode)
        assert!(!v(".(?<!.){2,3}", ""));
        assert!(v(".(?=.)?", "")); // quantified lookahead ok in Annex B
        assert!(!v(".(?=.)?", "u")); // but not under u
    }

    #[test]
    fn validate_groups_balanced() {
        assert!(v("(a)", ""));
        assert!(!v("(a", ""));
        assert!(!v("a)", ""));
        assert!(v("(?:ab)", ""));
        assert!(v("(?<name>ab)", ""));
    }

    #[test]
    fn match_flags_m_s_y() {
        // multiline: ^ matches after \n
        assert!(m("^b", "m", "a\nb"));
        assert!(!m("^b", "", "a\nb"));
        // dotall: . matches \n
        assert!(m("a.b", "s", "a\nb"));
        assert!(!m("a.b", "", "a\nb"));
        // sticky anchors at lastIndex 0 here
        assert!(m("a", "y", "abc"));
        assert!(!m("b", "y", "abc")); // must match at 0
    }

    #[test]
    fn match_escapes_decoded() {
        assert!(m("\\x41", "", "A"));
        assert!(m("\\u0041", "", "A"));
        assert!(m("\\u{41}", "u", "A"));
        assert!(m("\\0", "", "\0"));
    }

    #[test]
    fn match_backreferences() {
        assert!(m("(a)\\1", "", "aa"));
        assert!(!m("(a)\\1", "", "ab"));
        // forward reference: \1 before the group matches empty, then group matches
        assert!(m("\\1(a)", "", "a"));
        assert!(m("\\k<a>(?<a>x)", "", "x"));
    }

    #[test]
    fn mongolian_vowel_separator_not_whitespace() {
        assert!(!m("\\s", "", "\u{180E}"));
        assert!(m("\\s", "", " "));
    }
}
