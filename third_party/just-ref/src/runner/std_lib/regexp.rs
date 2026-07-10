//! ChittiOS: a compact backtracking `RegExp` engine (no external deps).
//!
//! Supports the common ECMAScript subset: literals, `.`, char classes `[...]`
//! (ranges, `^` negation), `\d \D \w \W \s \S \b \B`, anchors `^ $`, greedy
//! quantifiers `* + ? {n} {n,} {n,m}` (+ lazy `*?`/`+?`/`??`), alternation `|`,
//! capturing `( )` and non-capturing `(?: )` groups, and the `i` (ignore-case)
//! and `g` (global) flags. Not supported: lookahead/behind, backreferences,
//! named groups, unicode property escapes.
//!
//! `RegExp` values are objects tagged `__builtin_name__ = "RegExp"` carrying
//! `source`/`flags` (+ a mutable `lastIndex` for `g`). Instance methods
//! `test`/`exec` and `String.prototype.{match,replace,search,split}` route here.

#[allow(unused_imports)]
use num_traits::Float;
use alloc::borrow::ToOwned;
use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use alloc::vec;
use crate::runner::ds::error::JErrorType;
use crate::runner::ds::value::{JsValue, JsNumberType};
use crate::runner::plugin::registry::BuiltInRegistry;
use crate::runner::plugin::types::{BuiltInObject, EvalContext};
use crate::runner::eval::expression::{get_own_prop_value, make_array, make_object, set_own_prop};

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
    Seq(Vec<Node>),
    Alt(Vec<Node>),
    Repeat { node: Box<Node>, min: usize, max: Option<usize>, greedy: bool },
}

struct Parser {
    chars: Vec<char>,
    pos: usize,
    ncaps: usize,
}

impl Parser {
    fn parse(pattern: &str) -> Option<(Node, usize)> {
        let mut p = Parser { chars: pattern.chars().collect(), pos: 0, ncaps: 0 };
        let n = p.alt()?;
        if p.pos != p.chars.len() {
            return None;
        }
        Some((n, p.ncaps))
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
                    // (?:...) non-capturing
                    self.bump();
                    if self.peek() == Some(':') { self.bump(); }
                    None
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
                let e = self.bump()?;
                match e {
                    'd' | 'D' | 'w' | 'W' | 's' | 'S' => { specials.push(e); continue; }
                    'n' => '\n', 't' => '\t', 'r' => '\r', other => other,
                }
            } else {
                self.bump()?
            };
            if self.peek() == Some('-') && self.chars.get(self.pos + 1) != Some(&']') {
                self.bump(); // -
                let hi = self.bump()?;
                items.push((lo, hi));
            } else {
                items.push((lo, lo));
            }
        }
        None
    }
    fn escape(&mut self) -> Option<Node> {
        let c = self.bump()?;
        Some(match c {
            'd' | 'D' | 'w' | 'W' | 's' | 'S' => Node::Class { items: vec![], negate: false, specials: vec![c] },
            'b' => Node::WordB(true),
            'B' => Node::WordB(false),
            'n' => Node::Char('\n'),
            't' => Node::Char('\t'),
            'r' => Node::Char('\r'),
            other => Node::Char(other),
        })
    }
}

fn is_word(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

fn special_matches(sp: char, c: char) -> bool {
    match sp {
        'd' => c.is_ascii_digit(),
        'D' => !c.is_ascii_digit(),
        'w' => is_word(c),
        'W' => !is_word(c),
        's' => c.is_whitespace(),
        'S' => !c.is_whitespace(),
        _ => false,
    }
}

struct M<'a> {
    text: &'a [char],
    icase: bool,
    caps: Vec<Option<(usize, usize)>>,
}

impl<'a> M<'a> {
    fn ceq(&self, a: char, b: char) -> bool {
        if self.icase {
            a.eq_ignore_ascii_case(&b)
        } else {
            a == b
        }
    }
    /// Match `node` then continue matching `rest` (a slice of Seq nodes). Returns
    /// the end index on success. `k` is a continuation: match the tail from a pos.
    fn m(&mut self, node: &Node, pos: usize, k: &dyn Fn(&mut M, usize) -> Option<usize>) -> Option<usize> {
        match node {
            Node::Char(c) => {
                if pos < self.text.len() && self.ceq(self.text[pos], *c) {
                    k(self, pos + 1)
                } else {
                    None
                }
            }
            Node::Any => {
                if pos < self.text.len() && self.text[pos] != '\n' {
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
                    if self.icase {
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
                if pos == 0 { k(self, pos) } else { None }
            }
            Node::End => {
                if pos == self.text.len() { k(self, pos) } else { None }
            }
            Node::WordB(want) => {
                let before = pos > 0 && is_word(self.text[pos - 1]);
                let after = pos < self.text.len() && is_word(self.text[pos]);
                let boundary = before != after;
                if boundary == *want { k(self, pos) } else { None }
            }
            Node::Group { body, cap } => {
                let cap = *cap;
                let start = pos;
                // Continuation records the capture span, then runs the outer k.
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

/// Result of a match: (start, end, capture spans).
struct MatchResult {
    start: usize,
    end: usize,
    caps: Vec<Option<(usize, usize)>>,
}

/// Search `text` for `pattern` starting at `from`. Returns the leftmost match.
fn search(pattern: &str, icase: bool, text: &[char], from: usize) -> Option<MatchResult> {
    let (node, ncaps) = Parser::parse(pattern)?;
    let anchored = matches!(&node, Node::Seq(v) if v.first().map_or(false, |n| matches!(n, Node::Start)));
    let mut start = from;
    loop {
        if start > text.len() {
            return None;
        }
        let mut m = M { text, icase, caps: vec![None; ncaps] };
        let end_k = |_m: &mut M, p: usize| -> Option<usize> { Some(p) };
        if let Some(end) = m.m(&node, start, &end_k) {
            return Some(MatchResult { start, end, caps: m.caps });
        }
        if anchored {
            return None;
        }
        start += 1;
    }
}

// ---------------------------------------------------------------------------
// RegExp built-in + string integration
// ---------------------------------------------------------------------------

/// Build a RegExp value.
pub fn make_regexp(source: &str, flags: &str) -> JsValue {
    let obj = make_object(vec![]);
    set_own_prop(&obj, "__builtin_name__", JsValue::String("RegExp".to_string()), false);
    set_own_prop(&obj, "source", JsValue::String(source.to_string()), true);
    set_own_prop(&obj, "flags", JsValue::String(flags.to_string()), true);
    set_own_prop(&obj, "global", JsValue::Boolean(flags.contains('g')), true);
    set_own_prop(&obj, "ignoreCase", JsValue::Boolean(flags.contains('i')), true);
    set_own_prop(&obj, "lastIndex", JsValue::Number(JsNumberType::Integer(0)), true);
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

fn regexp_test(
    _ctx: &mut EvalContext,
    this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let (src, flags) = re_source(&this).ok_or_else(|| JErrorType::TypeError("not a RegExp".to_string()))?;
    let text: Vec<char> = str_arg(&args).chars().collect();
    Ok(JsValue::Boolean(search(&src, flags.contains('i'), &text, 0).is_some()))
}

fn regexp_exec(
    _ctx: &mut EvalContext,
    this: JsValue,
    args: Vec<JsValue>,
) -> Result<JsValue, JErrorType> {
    let (src, flags) = re_source(&this).ok_or_else(|| JErrorType::TypeError("not a RegExp".to_string()))?;
    let text: Vec<char> = str_arg(&args).chars().collect();
    let start = if flags.contains('g') {
        match get_own_prop_value(&this, "lastIndex") {
            Some(JsValue::Number(JsNumberType::Integer(n))) => n.max(0) as usize,
            _ => 0,
        }
    } else {
        0
    };
    match search(&src, flags.contains('i'), &text, start) {
        Some(mr) => {
            if flags.contains('g') {
                set_own_prop(&this, "lastIndex", JsValue::Number(JsNumberType::Integer(mr.end as i64)), true);
            }
            Ok(exec_result(&text, &mr))
        }
        None => {
            if flags.contains('g') {
                set_own_prop(&this, "lastIndex", JsValue::Number(JsNumberType::Integer(0)), true);
            }
            Ok(JsValue::Null)
        }
    }
}

fn span_str(text: &[char], s: usize, e: usize) -> String {
    text[s..e].iter().collect()
}

/// Build the `exec` result array: [fullMatch, group1, …] with `.index`.
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
    arr
}

/// `str.match(re)` — array of the match (+ groups), or all matches with `g`.
pub fn string_match(text: &str, re: &JsValue) -> JsValue {
    let Some((src, flags)) = re_source(re) else { return JsValue::Null };
    let chars: Vec<char> = text.chars().collect();
    let icase = flags.contains('i');
    if flags.contains('g') {
        let mut out = Vec::new();
        let mut from = 0;
        while let Some(mr) = search(&src, icase, &chars, from) {
            out.push(JsValue::String(span_str(&chars, mr.start, mr.end)));
            from = if mr.end > mr.start { mr.end } else { mr.end + 1 };
        }
        if out.is_empty() { JsValue::Null } else { make_array(out) }
    } else {
        match search(&src, icase, &chars, 0) {
            Some(mr) => exec_result(&chars, &mr),
            None => JsValue::Null,
        }
    }
}

/// `str.replace(re, replacement)` — supports `$1..$9`, `$&`. `g` flag → all.
pub fn string_replace_regexp(text: &str, re: &JsValue, replacement: &str) -> String {
    let Some((src, flags)) = re_source(re) else { return text.to_string() };
    let chars: Vec<char> = text.chars().collect();
    let icase = flags.contains('i');
    let global = flags.contains('g');
    let mut out = String::new();
    let mut from = 0usize;
    loop {
        match search(&src, icase, &chars, from) {
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
    let Some((src, flags)) = re_source(re) else { return vec![JsValue::String(text.to_string())] };
    let chars: Vec<char> = text.chars().collect();
    let icase = flags.contains('i');
    let mut out = Vec::new();
    let mut last = 0usize;
    let mut from = 0usize;
    while from <= chars.len() {
        match search(&src, icase, &chars, from) {
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

/// `str.search(re)` — index of first match or -1.
pub fn string_search(text: &str, re: &JsValue) -> i64 {
    let Some((src, flags)) = re_source(re) else { return -1 };
    let chars: Vec<char> = text.chars().collect();
    match search(&src, flags.contains('i'), &chars, 0) {
        Some(mr) => mr.start as i64,
        None => -1,
    }
}

/// Is `v` a RegExp value?
pub fn is_regexp(v: &JsValue) -> bool {
    matches!(get_own_prop_value(v, "__builtin_name__"), Some(JsValue::String(ref s)) if s == "RegExp")
}
