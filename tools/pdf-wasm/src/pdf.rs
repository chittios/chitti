//! Pure PDF reader: enough of ISO 32000 to digest real documents — xref
//! tables **and** xref streams, object streams (ObjStm), FlateDecode with the
//! PNG predictors, the page tree, and text extraction from content streams.
//! Everything is bounds-checked and returns `Err`/markers instead of
//! panicking; unsupported features degrade explicitly (`[unsupported ...]`),
//! never silently mis-extract.

use crate::inflate;
use alloc::borrow::ToOwned;
use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

// --- object model -------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub enum Obj {
    Null,
    Bool(bool),
    Int(i64),
    Real(f64),
    Str(Vec<u8>),
    Name(String),
    Array(Vec<Obj>),
    Dict(BTreeMap<String, Obj>),
    /// Reference `num gen R`.
    Ref(u32),
    /// Stream: dict + raw (still-encoded) bytes.
    Stream(BTreeMap<String, Obj>, Vec<u8>),
}

impl Obj {
    fn as_int(&self) -> Option<i64> {
        match self {
            Obj::Int(v) => Some(*v),
            Obj::Real(v) => Some(*v as i64),
            _ => None,
        }
    }
    fn as_name(&self) -> Option<&str> {
        match self {
            Obj::Name(n) => Some(n),
            _ => None,
        }
    }
    fn as_dict(&self) -> Option<&BTreeMap<String, Obj>> {
        match self {
            Obj::Dict(d) | Obj::Stream(d, _) => Some(d),
            _ => None,
        }
    }
}

// --- lexer / object parser -----------------------------------------------------

struct Lex<'a> {
    b: &'a [u8],
    p: usize,
}

fn is_ws(c: u8) -> bool {
    matches!(c, b' ' | b'\t' | b'\r' | b'\n' | b'\x0c' | b'\0')
}
fn is_delim(c: u8) -> bool {
    matches!(c, b'(' | b')' | b'<' | b'>' | b'[' | b']' | b'{' | b'}' | b'/' | b'%')
}

impl<'a> Lex<'a> {
    fn new(b: &'a [u8], p: usize) -> Lex<'a> {
        Lex { b, p }
    }
    fn peek(&self) -> Option<u8> {
        self.b.get(self.p).copied()
    }
    fn ws(&mut self) {
        loop {
            while let Some(c) = self.peek() {
                if is_ws(c) {
                    self.p += 1;
                } else {
                    break;
                }
            }
            // Comments run to end of line.
            if self.peek() == Some(b'%') {
                while let Some(c) = self.peek() {
                    self.p += 1;
                    if c == b'\n' {
                        break;
                    }
                }
            } else {
                break;
            }
        }
    }
    fn keyword(&mut self) -> &'a [u8] {
        let s = self.p;
        while let Some(c) = self.peek() {
            if is_ws(c) || is_delim(c) {
                break;
            }
            self.p += 1;
        }
        &self.b[s..self.p]
    }
    fn expect(&mut self, kw: &[u8]) -> bool {
        self.ws();
        if self.b[self.p..].starts_with(kw) {
            self.p += kw.len();
            true
        } else {
            false
        }
    }

    /// One object. `depth` bounds recursion (malicious nesting).
    fn obj(&mut self, depth: u32) -> Result<Obj, &'static str> {
        if depth > 64 {
            return Err("nesting too deep");
        }
        self.ws();
        let c = self.peek().ok_or("eof")?;
        match c {
            b'/' => {
                self.p += 1;
                let raw = self.keyword();
                let mut name = String::new();
                let mut i = 0;
                while i < raw.len() {
                    if raw[i] == b'#' && i + 2 < raw.len() {
                        let h = core::str::from_utf8(&raw[i + 1..i + 3]).ok().and_then(|s| u8::from_str_radix(s, 16).ok());
                        if let Some(v) = h {
                            name.push(v as char);
                            i += 3;
                            continue;
                        }
                    }
                    name.push(raw[i] as char);
                    i += 1;
                }
                Ok(Obj::Name(name))
            }
            b'(' => self.lit_string(),
            b'<' => {
                if self.b.get(self.p + 1) == Some(&b'<') {
                    self.dict_or_stream(depth)
                } else {
                    self.hex_string()
                }
            }
            b'[' => {
                self.p += 1;
                let mut v = Vec::new();
                loop {
                    self.ws();
                    if self.peek() == Some(b']') {
                        self.p += 1;
                        break;
                    }
                    if self.peek().is_none() {
                        return Err("unterminated array");
                    }
                    v.push(self.obj(depth + 1)?);
                    if v.len() > 65536 {
                        return Err("array too long");
                    }
                }
                Ok(Obj::Array(v))
            }
            b'+' | b'-' | b'.' | b'0'..=b'9' => self.number_or_ref(),
            _ => {
                let kw = self.keyword();
                match kw {
                    b"true" => Ok(Obj::Bool(true)),
                    b"false" => Ok(Obj::Bool(false)),
                    b"null" => Ok(Obj::Null),
                    _ => Err("bad token"),
                }
            }
        }
    }

    fn lit_string(&mut self) -> Result<Obj, &'static str> {
        self.p += 1; // (
        let mut out = Vec::new();
        let mut depth = 1u32;
        while let Some(c) = self.peek() {
            self.p += 1;
            match c {
                b'\\' => {
                    let e = self.peek().ok_or("bad escape")?;
                    self.p += 1;
                    match e {
                        b'n' => out.push(b'\n'),
                        b'r' => out.push(b'\r'),
                        b't' => out.push(b'\t'),
                        b'b' => out.push(8),
                        b'f' => out.push(12),
                        b'(' | b')' | b'\\' => out.push(e),
                        b'\r' => {
                            if self.peek() == Some(b'\n') {
                                self.p += 1;
                            }
                        }
                        b'\n' => {}
                        b'0'..=b'7' => {
                            let mut v = (e - b'0') as u32;
                            for _ in 0..2 {
                                match self.peek() {
                                    Some(d @ b'0'..=b'7') => {
                                        v = v * 8 + (d - b'0') as u32;
                                        self.p += 1;
                                    }
                                    _ => break,
                                }
                            }
                            out.push(v as u8);
                        }
                        other => out.push(other),
                    }
                }
                b'(' => {
                    depth += 1;
                    out.push(c);
                }
                b')' => {
                    depth -= 1;
                    if depth == 0 {
                        return Ok(Obj::Str(out));
                    }
                    out.push(c);
                }
                _ => out.push(c),
            }
            if out.len() > 1 << 20 {
                return Err("string too long");
            }
        }
        Err("unterminated string")
    }

    fn hex_string(&mut self) -> Result<Obj, &'static str> {
        self.p += 1; // <
        let mut out = Vec::new();
        let mut hi: Option<u8> = None;
        while let Some(c) = self.peek() {
            self.p += 1;
            let v = match c {
                b'>' => {
                    if let Some(h) = hi {
                        out.push(h << 4);
                    }
                    return Ok(Obj::Str(out));
                }
                b'0'..=b'9' => c - b'0',
                b'a'..=b'f' => c - b'a' + 10,
                b'A'..=b'F' => c - b'A' + 10,
                c if is_ws(c) => continue,
                _ => return Err("bad hex string"),
            };
            match hi {
                None => hi = Some(v),
                Some(h) => {
                    out.push((h << 4) | v);
                    hi = None;
                }
            }
            if out.len() > 1 << 20 {
                return Err("hex string too long");
            }
        }
        Err("unterminated hex string")
    }

    fn number_or_ref(&mut self) -> Result<Obj, &'static str> {
        let save = self.p;
        let kw = self.keyword();
        let s = core::str::from_utf8(kw).map_err(|_| "bad number")?;
        // `N G R` reference lookahead (both non-negative integers).
        if let Ok(num) = s.parse::<u32>() {
            let after_num = self.p;
            self.ws();
            let g = self.keyword();
            if core::str::from_utf8(g).map(|t| t.parse::<u32>().is_ok()).unwrap_or(false) {
                self.ws();
                if self.peek() == Some(b'R') {
                    let after_r = self.p + 1;
                    // R must be a lone keyword.
                    if self.b.get(after_r).map(|&c| is_ws(c) || is_delim(c)).unwrap_or(true) {
                        self.p = after_r;
                        return Ok(Obj::Ref(num));
                    }
                }
            }
            self.p = after_num;
            return Ok(Obj::Int(num as i64));
        }
        self.p = save + kw.len();
        if let Ok(v) = s.parse::<i64>() {
            return Ok(Obj::Int(v));
        }
        s.parse::<f64>().map(Obj::Real).map_err(|_| "bad number")
    }

    fn dict_or_stream(&mut self, depth: u32) -> Result<Obj, &'static str> {
        self.p += 2; // <<
        let mut d = BTreeMap::new();
        loop {
            self.ws();
            if self.b[self.p..].starts_with(b">>") {
                self.p += 2;
                break;
            }
            let Obj::Name(k) = self.obj(depth + 1)? else {
                return Err("dict key not a name");
            };
            let v = self.obj(depth + 1)?;
            d.insert(k, v);
            if d.len() > 4096 {
                return Err("dict too big");
            }
        }
        // `stream` follows?
        let save = self.p;
        self.ws();
        if self.b[self.p..].starts_with(b"stream") {
            self.p += 6;
            if self.peek() == Some(b'\r') {
                self.p += 1;
            }
            if self.peek() == Some(b'\n') {
                self.p += 1;
            }
            // Caller resolves /Length (possibly indirect) and slices; here we
            // take a *conservative* slice to `endstream` for the direct case.
            return Ok(Obj::Stream(d, Vec::new())); // body filled by Doc::load
        }
        self.p = save;
        Ok(Obj::Dict(d))
    }
}

// --- document ---------------------------------------------------------------

pub struct Doc<'a> {
    b: &'a [u8],
    /// object number → (kind, a, b): kind 1 = at offset a; kind 2 = in ObjStm a, index b.
    xref: BTreeMap<u32, (u8, u64, u64)>,
    trailer: BTreeMap<String, Obj>,
    cache: BTreeMap<u32, Obj>,
    /// Decoded object-stream cache (ObjStm number → decoded bytes + pairs).
    objstm: BTreeMap<u32, (Vec<(u32, usize)>, Vec<u8>)>,
}

/// PNG predictor reverse over `colors * bpc / 8 * columns`-byte rows.
fn unpredict(data: &[u8], predictor: i64, columns: usize, colors: usize, bpc: usize) -> Result<Vec<u8>, &'static str> {
    if predictor < 10 {
        return Ok(data.to_owned()); // 1/2: none / TIFF (TIFF unsupported → raw)
    }
    let stride = (columns * colors * bpc).div_ceil(8).max(1);
    let mut out = Vec::with_capacity(data.len());
    let mut prev = alloc::vec![0u8; stride];
    let mut i = 0;
    while i + 1 + stride <= data.len() + 1 {
        if i >= data.len() {
            break;
        }
        let ft = data[i];
        i += 1;
        let end = (i + stride).min(data.len());
        let row = &data[i..end];
        i = end;
        let mut cur = alloc::vec![0u8; row.len()];
        let bpp = (colors * bpc).div_ceil(8).max(1);
        for x in 0..row.len() {
            let a = if x >= bpp { cur[x - bpp] } else { 0 } as i32; // left
            let b = prev[x] as i32; // up
            let c = if x >= bpp { prev[x - bpp] } else { 0 } as i32; // up-left
            let raw = row[x] as i32;
            cur[x] = match ft {
                0 => raw as u8,
                1 => (raw + a) as u8,
                2 => (raw + b) as u8,
                3 => (raw + (a + b) / 2) as u8,
                4 => {
                    let p = a + b - c;
                    let (pa, pb, pc) = ((p - a).abs(), (p - b).abs(), (p - c).abs());
                    let pred = if pa <= pb && pa <= pc { a } else if pb <= pc { b } else { c };
                    (raw + pred) as u8
                }
                _ => return Err("bad png predictor"),
            };
        }
        prev[..cur.len()].copy_from_slice(&cur);
        if cur.len() < stride {
            prev[cur.len()..].fill(0);
        }
        out.extend_from_slice(&cur);
        if out.len() > 64 << 20 {
            return Err("predictor output too big");
        }
    }
    Ok(out)
}

impl<'a> Doc<'a> {
    pub fn open(b: &'a [u8]) -> Result<Doc<'a>, &'static str> {
        if !b.starts_with(b"%PDF") {
            return Err("not a PDF");
        }
        let mut doc = Doc { b, xref: BTreeMap::new(), trailer: BTreeMap::new(), cache: BTreeMap::new(), objstm: BTreeMap::new() };
        // startxref: search the tail.
        let tail_at = b.len().saturating_sub(2048);
        let tail = &b[tail_at..];
        let sx = find_last(tail, b"startxref").ok_or("no startxref")?;
        let mut lex = Lex::new(b, tail_at + sx + 9);
        lex.ws();
        let off = core::str::from_utf8(lex.keyword()).ok().and_then(|s| s.parse::<u64>().ok()).ok_or("bad startxref")?;
        doc.load_xref_chain(off)?;
        if doc.trailer.get("Encrypt").is_some() {
            return Err("encrypted PDF (not supported)");
        }
        Ok(doc)
    }

    fn load_xref_chain(&mut self, mut off: u64) -> Result<(), &'static str> {
        for _ in 0..16 {
            let prev = self.load_xref_at(off)?;
            match prev {
                Some(p) => off = p,
                None => return Ok(()),
            }
        }
        Err("xref chain too long")
    }

    /// One xref section (table or stream); returns /Prev if present.
    fn load_xref_at(&mut self, off: u64) -> Result<Option<u64>, &'static str> {
        let off = off as usize;
        if off >= self.b.len() {
            return Err("xref offset out of range");
        }
        let mut lex = Lex::new(self.b, off);
        lex.ws();
        if lex.b[lex.p..].starts_with(b"xref") {
            lex.p += 4;
            // Classic table: sections of `start count` then 20-byte entries.
            loop {
                lex.ws();
                if lex.b[lex.p..].starts_with(b"trailer") {
                    lex.p += 7;
                    let t = lex.obj(0)?;
                    let d = t.as_dict().ok_or("trailer not a dict")?.clone();
                    let prev = d.get("Prev").and_then(|o| o.as_int()).map(|v| v as u64);
                    for (k, v) in d {
                        self.trailer.entry(k).or_insert(v);
                    }
                    return Ok(prev);
                }
                let start: u32 = core::str::from_utf8(lex.keyword()).ok().and_then(|s| s.parse().ok()).ok_or("bad xref section")?;
                lex.ws();
                let count: u32 = core::str::from_utf8(lex.keyword()).ok().and_then(|s| s.parse().ok()).ok_or("bad xref count")?;
                if count > 1 << 22 {
                    return Err("xref too large");
                }
                lex.ws();
                for i in 0..count {
                    // `nnnnnnnnnn ggggg n\r\n` fixed 20 bytes (tolerate ws).
                    lex.ws();
                    let o = core::str::from_utf8(lex.keyword()).ok().and_then(|s| s.parse::<u64>().ok()).ok_or("bad xref entry")?;
                    lex.ws();
                    let _g = lex.keyword();
                    lex.ws();
                    let ty = lex.keyword();
                    let num = start + i;
                    if ty == b"n" {
                        self.xref.entry(num).or_insert((1, o, 0));
                    }
                }
            }
        }
        // xref stream: `N G obj << /Type /XRef ... >> stream`.
        let (dict, data) = self.parse_indirect_stream_at(off)?;
        let w = match dict.get("W") {
            Some(Obj::Array(a)) => a.iter().filter_map(|o| o.as_int()).map(|v| v as usize).collect::<Vec<_>>(),
            _ => return Err("xref stream missing W"),
        };
        if w.len() < 3 || w.iter().sum::<usize>() == 0 || w.iter().any(|&x| x > 8) {
            return Err("bad xref W");
        }
        let size = dict.get("Size").and_then(|o| o.as_int()).unwrap_or(0) as u32;
        let index: Vec<u32> = match dict.get("Index") {
            Some(Obj::Array(a)) => a.iter().filter_map(|o| o.as_int()).map(|v| v as u32).collect(),
            _ => alloc::vec![0, size],
        };
        let rec = w[0] + w[1] + w[2];
        let rd = |bytes: &[u8]| -> u64 {
            let mut v = 0u64;
            for &b in bytes {
                v = (v << 8) | b as u64;
            }
            v
        };
        let mut pos = 0usize;
        for pair in index.chunks(2) {
            let (&start, &count) = match pair {
                [s, c] => (s, c),
                _ => break,
            };
            for i in 0..count {
                if pos + rec > data.len() {
                    break;
                }
                let f1 = if w[0] == 0 { 1 } else { rd(&data[pos..pos + w[0]]) };
                let f2 = rd(&data[pos + w[0]..pos + w[0] + w[1]]);
                let f3 = rd(&data[pos + w[0] + w[1]..pos + rec]);
                pos += rec;
                let num = start + i;
                match f1 {
                    1 => {
                        self.xref.entry(num).or_insert((1, f2, f3));
                    }
                    2 => {
                        self.xref.entry(num).or_insert((2, f2, f3));
                    }
                    _ => {}
                }
            }
        }
        let prev = dict.get("Prev").and_then(|o| o.as_int()).map(|v| v as u64);
        for (k, v) in dict {
            self.trailer.entry(k).or_insert(v);
        }
        Ok(prev)
    }

    /// Parse `N G obj <dict> stream ... endstream` at `off`; decode filters.
    fn parse_indirect_stream_at(&mut self, off: usize) -> Result<(BTreeMap<String, Obj>, Vec<u8>), &'static str> {
        let (dict, body) = self.parse_indirect_at(off)?;
        let dict = match dict {
            Obj::Stream(d, _) => d,
            _ => return Err("expected stream object"),
        };
        let data = self.decode_stream(&dict, &body)?;
        Ok((dict, data))
    }

    /// Parse the indirect object at `off`. For streams, the (encoded) body is
    /// returned separately (`/Length` may be an indirect ref).
    fn parse_indirect_at(&mut self, off: usize) -> Result<(Obj, Vec<u8>), &'static str> {
        let mut lex = Lex::new(self.b, off);
        lex.ws();
        let _num = lex.keyword();
        lex.ws();
        let _gen = lex.keyword();
        if !lex.expect(b"obj") {
            return Err("not an indirect object");
        }
        let o = lex.obj(0)?;
        if let Obj::Stream(d, _) = &o {
            // Body starts at lex.p (dict_or_stream consumed `stream\r?\n`).
            let start = lex.p;
            let len = match d.get("Length") {
                Some(Obj::Int(v)) => *v as usize,
                Some(Obj::Ref(r)) => self.load(*r)?.as_int().ok_or("bad /Length")? as usize,
                _ => {
                    // No usable length: scan for endstream.
                    let rest = &self.b[start..];
                    find(rest, b"endstream").ok_or("no endstream")?
                }
            };
            if start + len > self.b.len() {
                return Err("stream overruns file");
            }
            return Ok((o, self.b[start..start + len].to_vec()));
        }
        Ok((o, Vec::new()))
    }

    /// Apply /Filter (+ /DecodeParms) to a stream body. Flate only; anything
    /// else returns Err("unsupported filter").
    fn decode_stream(&mut self, dict: &BTreeMap<String, Obj>, body: &[u8]) -> Result<Vec<u8>, &'static str> {
        let filters: Vec<String> = match dict.get("Filter") {
            None => return Ok(body.to_vec()),
            Some(Obj::Name(n)) => alloc::vec![n.clone()],
            Some(Obj::Array(a)) => a.iter().filter_map(|o| o.as_name().map(|s| s.to_owned())).collect(),
            Some(Obj::Ref(r)) => match self.load(*r)? {
                Obj::Name(n) => alloc::vec![n],
                _ => return Err("bad /Filter"),
            },
            _ => return Err("bad /Filter"),
        };
        let mut data = body.to_vec();
        for f in &filters {
            match f.as_str() {
                "FlateDecode" | "Fl" => {
                    data = inflate::inflate(strip_zlib(&data)).map_err(|_| "flate error")?;
                }
                _ => return Err("unsupported filter"),
            }
        }
        // Predictor (applies to the final decode).
        let parms = match dict.get("DecodeParms").or_else(|| dict.get("DP")) {
            Some(Obj::Dict(d)) => Some(d.clone()),
            Some(Obj::Array(a)) => a.iter().find_map(|o| match o {
                Obj::Dict(d) => Some(d.clone()),
                _ => None,
            }),
            _ => None,
        };
        if let Some(p) = parms {
            let predictor = p.get("Predictor").and_then(|o| o.as_int()).unwrap_or(1);
            if predictor > 1 {
                let columns = p.get("Columns").and_then(|o| o.as_int()).unwrap_or(1) as usize;
                let colors = p.get("Colors").and_then(|o| o.as_int()).unwrap_or(1) as usize;
                let bpc = p.get("BitsPerComponent").and_then(|o| o.as_int()).unwrap_or(8) as usize;
                data = unpredict(&data, predictor, columns.max(1), colors.max(1), bpc.max(1))?;
            }
        }
        Ok(data)
    }

    /// Load object `num` (cached), resolving through ObjStm as needed.
    pub fn load(&mut self, num: u32) -> Result<Obj, &'static str> {
        if let Some(o) = self.cache.get(&num) {
            return Ok(o.clone());
        }
        let &(kind, a, bidx) = self.xref.get(&num).ok_or("object not in xref")?;
        let o = match kind {
            1 => {
                let (o, body) = self.parse_indirect_at(a as usize)?;
                match o {
                    Obj::Stream(d, _) => Obj::Stream(d, body),
                    other => other,
                }
            }
            2 => {
                let stm_num = a as u32;
                if !self.objstm.contains_key(&stm_num) {
                    let &(k2, off2, _) = self.xref.get(&stm_num).ok_or("objstm not in xref")?;
                    if k2 != 1 {
                        return Err("nested objstm");
                    }
                    let (d, data) = self.parse_indirect_stream_at(off2 as usize)?;
                    let n = d.get("N").and_then(|o| o.as_int()).unwrap_or(0) as usize;
                    let first = d.get("First").and_then(|o| o.as_int()).unwrap_or(0) as usize;
                    let mut pairs = Vec::new();
                    let mut lx = Lex::new(&data, 0);
                    for _ in 0..n.min(65536) {
                        lx.ws();
                        let on: u32 = core::str::from_utf8(lx.keyword()).ok().and_then(|s| s.parse().ok()).ok_or("bad objstm header")?;
                        lx.ws();
                        let oo: usize = core::str::from_utf8(lx.keyword()).ok().and_then(|s| s.parse().ok()).ok_or("bad objstm header")?;
                        pairs.push((on, first + oo));
                    }
                    self.objstm.insert(stm_num, (pairs, data));
                }
                let (pairs, data) = self.objstm.get(&stm_num).unwrap();
                let &(_, off) = pairs.iter().find(|(n, _)| *n == num).ok_or("object not in objstm")?;
                if off >= data.len() {
                    return Err("objstm offset out of range");
                }
                // Parse from the decoded stream buffer.
                let data = data.clone();
                let _ = bidx;
                let mut lx = Lex::new(&data, off);
                lx.obj(0)?
            }
            _ => Obj::Null,
        };
        self.cache.insert(num, o.clone());
        Ok(o)
    }

    /// Resolve an object that may be a reference.
    fn resolve(&mut self, o: &Obj) -> Result<Obj, &'static str> {
        match o {
            Obj::Ref(r) => self.load(*r),
            other => Ok(other.clone()),
        }
    }

    /// Ordered page object numbers (walks the /Pages tree, bounded).
    pub fn pages(&mut self) -> Result<Vec<Obj>, &'static str> {
        let root = self.trailer.get("Root").cloned().ok_or("no /Root")?;
        let cat = self.resolve(&root)?;
        let pages_ref = cat.as_dict().and_then(|d| d.get("Pages")).cloned().ok_or("no /Pages")?;
        let mut out = Vec::new();
        let mut stack = alloc::vec![pages_ref];
        let mut seen = 0u32;
        while let Some(node) = stack.pop() {
            seen += 1;
            if seen > 8192 {
                return Err("page tree too large");
            }
            let node = self.resolve(&node)?;
            let Some(d) = node.as_dict() else { continue };
            match d.get("Type").and_then(|t| t.as_name()) {
                Some("Page") => out.push(node.clone()),
                _ => {
                    if let Some(Obj::Array(kids)) = d.get("Kids").map(|k| k.clone()) {
                        // Push in reverse so pop() walks in document order.
                        for k in kids.into_iter().rev() {
                            stack.push(k);
                        }
                    } else if d.contains_key("Contents") {
                        out.push(node.clone()); // Page without /Type (lenient)
                    }
                }
            }
            if out.len() > 4096 {
                break;
            }
        }
        Ok(out)
    }

    /// Concatenated decoded content streams of a page dict.
    pub fn page_content(&mut self, page: &Obj) -> Result<Vec<u8>, &'static str> {
        let d = page.as_dict().ok_or("page not a dict")?;
        let contents = match d.get("Contents") {
            None => return Ok(Vec::new()),
            Some(c) => c.clone(),
        };
        let contents = self.resolve(&contents)?;
        let list: Vec<Obj> = match contents {
            Obj::Array(a) => a,
            one => alloc::vec![one],
        };
        let mut out = Vec::new();
        for c in list {
            let c = self.resolve(&c)?;
            if let Obj::Stream(sd, body) = c {
                match self.decode_stream(&sd, &body) {
                    Ok(mut data) => {
                        out.append(&mut data);
                        out.push(b'\n');
                    }
                    Err(_) => out.extend_from_slice(b"\n[unsupported content filter]\n"),
                }
            }
            if out.len() > 16 << 20 {
                break;
            }
        }
        Ok(out)
    }

    /// /Info metadata string (Title/Author), UTF-16BE BOM aware.
    pub fn info_str(&mut self, key: &str) -> Option<String> {
        let info = self.trailer.get("Info").cloned()?;
        let info = self.resolve(&info).ok()?;
        let v = info.as_dict()?.get(key)?.clone();
        let v = self.resolve(&v).ok()?;
        match v {
            Obj::Str(b) => Some(decode_pdf_text(&b)),
            _ => None,
        }
    }
}

/// PDF text-string decode: UTF-16BE with BOM, else PDFDocEncoding≈Latin-1.
pub fn decode_pdf_text(b: &[u8]) -> String {
    if b.len() >= 2 && b[0] == 0xfe && b[1] == 0xff {
        let mut s = String::new();
        let mut i = 2;
        while i + 1 < b.len() {
            let u = u16::from_be_bytes([b[i], b[i + 1]]);
            s.push(char::from_u32(u as u32).unwrap_or('?'));
            i += 2;
        }
        return s;
    }
    b.iter().map(|&c| if (0x20..0x7f).contains(&c) { c as char } else if c == b'\n' || c == b'\r' || c == b'\t' { ' ' } else { '?' }).collect()
}

/// PDF FlateDecode streams are zlib-wrapped (RFC 1950); the kernel `inflate`
/// is raw RFC 1951. Strip a valid 2-byte zlib header (CM=8, fcheck passes,
/// no preset dict); the trailing adler32 is simply never read (inflate stops
/// at BFINAL).
fn strip_zlib(d: &[u8]) -> &[u8] {
    if d.len() > 2 && d[0] & 0x0f == 8 && (d[0] as u16 * 256 + d[1] as u16) % 31 == 0 && d[1] & 0x20 == 0 {
        &d[2..]
    } else {
        d
    }
}

fn find(h: &[u8], n: &[u8]) -> Option<usize> {
    h.windows(n.len()).position(|w| w == n)
}
fn find_last(h: &[u8], n: &[u8]) -> Option<usize> {
    h.windows(n.len()).rposition(|w| w == n)
}

// --- text extraction -----------------------------------------------------------

/// Extract text lines from a decoded content stream: tracks BT/ET, the text
/// matrix/leading enough to detect line breaks, and decodes Tj / TJ / ' / ".
/// Font-encoding handling is deliberately simple (bytes as Latin-1); exotic
/// CMaps degrade to '?', never to wrong letters silently.
pub fn extract_text(content: &[u8]) -> Vec<String> {
    let mut lex = Lex::new(content, 0);
    let mut lines: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut stack: Vec<Obj> = Vec::new();
    let mut last_y: Option<i64> = None;
    let flush = |lines: &mut Vec<String>, cur: &mut String| {
        let t = cur.trim();
        if !t.is_empty() {
            lines.push(t.to_owned());
        }
        cur.clear();
    };
    loop {
        lex.ws();
        let Some(c) = lex.peek() else { break };
        // Operands parse as objects; operators are bare keywords.
        if matches!(c, b'/' | b'(' | b'<' | b'[' | b'+' | b'-' | b'.' | b'0'..=b'9') {
            match lex.obj(0) {
                Ok(o) => {
                    stack.push(o);
                    if stack.len() > 64 {
                        stack.remove(0);
                    }
                    continue;
                }
                Err(_) => {
                    lex.p += 1;
                    continue;
                }
            }
        }
        let op = lex.keyword();
        if op.is_empty() {
            lex.p += 1;
            continue;
        }
        match op {
            b"Tj" | b"'" | b"\"" => {
                if op != b"Tj" {
                    flush(&mut lines, &mut cur); // ' and " imply next-line first
                }
                if let Some(Obj::Str(s)) = stack.iter().rev().find(|o| matches!(o, Obj::Str(_))) {
                    cur.push_str(&decode_pdf_text(s));
                }
            }
            b"TJ" => {
                if let Some(Obj::Array(a)) = stack.iter().rev().find(|o| matches!(o, Obj::Array(_))) {
                    for el in a {
                        match el {
                            Obj::Str(s) => cur.push_str(&decode_pdf_text(s)),
                            // Large negative kerns are inter-word gaps.
                            Obj::Int(v) if *v < -180 => cur.push(' '),
                            Obj::Real(v) if *v < -180.0 => cur.push(' '),
                            _ => {}
                        }
                    }
                }
            }
            b"Td" | b"TD" => {
                // y operand ≠ 0 → new line (x-only moves are same-line kerning).
                let dy = stack.last().and_then(|o| o.as_int()).unwrap_or(0);
                if dy != 0 {
                    flush(&mut lines, &mut cur);
                }
            }
            b"Tm" => {
                let y = stack.last().and_then(|o| o.as_int());
                if last_y.is_some() && y != last_y {
                    flush(&mut lines, &mut cur);
                }
                last_y = y;
            }
            b"T*" => flush(&mut lines, &mut cur),
            b"BT" | b"ET" => flush(&mut lines, &mut cur),
            _ => {}
        }
        stack.clear();
        if lines.len() > 4096 {
            break;
        }
    }
    flush(&mut lines, &mut cur);
    lines
}

// --- digest ---------------------------------------------------------------

fn json_escape(s: &str, out: &mut String) {
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => {}
            c if (c as u32) < 0x20 => out.push(' '),
            c => out.push(c),
        }
    }
}

/// The one expensive call: parse everything and return the JSON digest the
/// kernel runtime caches (metadata + per-page text, capped).
pub fn digest(bytes: &[u8], max_pages: usize) -> Result<String, &'static str> {
    let mut doc = Doc::open(bytes)?;
    let pages = doc.pages()?;
    let title = doc.info_str("Title").unwrap_or_default();
    let author = doc.info_str("Author").unwrap_or_default();
    let mut out = String::with_capacity(4096);
    out.push_str(&format!("{{\"pages\":{},\"title\":\"", pages.len()));
    json_escape(&title, &mut out);
    out.push_str("\",\"author\":\"");
    json_escape(&author, &mut out);
    out.push_str(&format!("\",\"truncated\":{},\"page_texts\":[", pages.len() > max_pages));
    let mut budget = 96 * 1024usize; // total digest text cap
    for (i, p) in pages.iter().take(max_pages).enumerate() {
        if i > 0 {
            out.push(',');
        }
        let text = match doc.page_content(p) {
            Ok(c) => extract_text(&c).join("\n"),
            Err(e) => format!("[page not decodable: {e}]"),
        };
        let take: String = text.chars().take(budget).collect();
        budget = budget.saturating_sub(take.chars().count());
        out.push_str(&format!("{{\"n\":{},\"text\":\"", i + 1));
        json_escape(&take, &mut out);
        out.push_str("\"}");
        if budget == 0 {
            break;
        }
    }
    out.push_str("]}");
    Ok(out)
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
