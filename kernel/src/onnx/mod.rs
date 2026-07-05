//! **Minimal zero-copy ONNX reader** — parses an ONNX model (a protobuf
//! `ModelProto`) directly from a byte slice, `no_std`, with tensor `raw_data`
//! left in place as borrowed slices. This is the loading layer the voice
//! models (silero-vad, parakeet STT, KittenTTS) sit on; the op executor grows
//! alongside each model's op set.
//!
//! Deliberately lenient: unknown fields are skipped (forward-compatible, the
//! protobuf way), and only the messages/fields the executor needs are decoded:
//! graph nodes (op/name/inputs/outputs/attributes — including *sub-graph*
//! attributes, which silero's `If` uses), initializers, and graph I/O names.

use alloc::string::String;
use alloc::vec::Vec;

pub mod exec;

/// A parsed model: IR metadata + the top-level graph.
pub struct Model<'a> {
    pub ir_version: i64,
    pub graph: Graph<'a>,
}

/// A computation graph.
#[derive(Default)]
pub struct Graph<'a> {
    pub name: &'a str,
    pub nodes: Vec<Node<'a>>,
    pub initializers: Vec<Tensor<'a>>,
    pub inputs: Vec<&'a str>,
    pub outputs: Vec<&'a str>,
}

/// One operator node.
pub struct Node<'a> {
    pub op: &'a str,
    pub name: &'a str,
    pub inputs: Vec<&'a str>,
    pub outputs: Vec<&'a str>,
    pub attrs: Vec<Attr<'a>>,
}

/// One node attribute (only the payloads we consume).
pub struct Attr<'a> {
    pub name: &'a str,
    pub i: i64,
    pub f: f32,
    pub s: &'a [u8],
    pub ints: Vec<i64>,
    pub floats: Vec<f32>,
    pub tensor: Option<Tensor<'a>>,
    pub graph: Option<Graph<'a>>,
}

/// A (possibly zero-copy) tensor. `raw` borrows the model bytes when the
/// tensor uses `raw_data`; otherwise the typed arrays are populated.
pub struct Tensor<'a> {
    pub name: &'a str,
    pub dims: Vec<i64>,
    pub dtype: i32, // 1=f32, 3=i8, 6=i32, 7=i64 (ONNX TensorProto.DataType)
    pub raw: &'a [u8],
    pub floats: Vec<f32>,
    pub ints: Vec<i64>,
}

impl Tensor<'_> {
    /// Element count from dims (empty dims = scalar = 1).
    pub fn len(&self) -> usize {
        self.dims.iter().product::<i64>().max(1) as usize
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// The tensor's f32 data: `raw_data` reinterpreted, or `float_data`.
    pub fn f32_at(&self, i: usize) -> f32 {
        if !self.raw.is_empty() {
            let o = i * 4;
            f32::from_le_bytes([self.raw[o], self.raw[o + 1], self.raw[o + 2], self.raw[o + 3]])
        } else {
            self.floats[i]
        }
    }
}

// --- protobuf wire reading -------------------------------------------------

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }
    fn eof(&self) -> bool {
        self.pos >= self.buf.len()
    }
    fn varint(&mut self) -> Option<u64> {
        let mut v = 0u64;
        let mut shift = 0;
        loop {
            let b = *self.buf.get(self.pos)?;
            self.pos += 1;
            v |= ((b & 0x7f) as u64) << shift;
            if b & 0x80 == 0 {
                return Some(v);
            }
            shift += 7;
            if shift >= 64 {
                return None;
            }
        }
    }
    fn bytes(&mut self) -> Option<&'a [u8]> {
        let n = self.varint()? as usize;
        let s = self.buf.get(self.pos..self.pos + n)?;
        self.pos += n;
        Some(s)
    }
    fn fixed32(&mut self) -> Option<u32> {
        let s = self.buf.get(self.pos..self.pos + 4)?;
        self.pos += 4;
        Some(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    }
    fn fixed64(&mut self) -> Option<u64> {
        let s = self.buf.get(self.pos..self.pos + 8)?;
        self.pos += 8;
        Some(u64::from_le_bytes([s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]]))
    }
    /// Skip one field of the given wire type.
    fn skip(&mut self, wire: u32) -> Option<()> {
        match wire {
            0 => {
                self.varint()?;
            }
            1 => {
                self.fixed64()?;
            }
            2 => {
                self.bytes()?;
            }
            5 => {
                self.fixed32()?;
            }
            _ => return None,
        }
        Some(())
    }
}

fn as_str(b: &[u8]) -> &str {
    core::str::from_utf8(b).unwrap_or("")
}

/// Zig-zag is NOT used by ONNX ints (plain two's-complement varints).
fn as_i64(v: u64) -> i64 {
    v as i64
}

// --- message parsers ---------------------------------------------------------

/// Parse a full `ModelProto`.
pub fn parse(bytes: &[u8]) -> Option<Model<'_>> {
    let mut r = Reader::new(bytes);
    let mut ir_version = 0i64;
    let mut graph = Graph::default();
    while !r.eof() {
        let key = r.varint()?;
        let (field, wire) = ((key >> 3) as u32, (key & 7) as u32);
        match (field, wire) {
            (1, 0) => ir_version = as_i64(r.varint()?),
            (7, 2) => graph = parse_graph(r.bytes()?)?,
            _ => r.skip(wire)?,
        }
    }
    Some(Model { ir_version, graph })
}

fn parse_graph(bytes: &[u8]) -> Option<Graph<'_>> {
    let mut r = Reader::new(bytes);
    let mut g = Graph::default();
    while !r.eof() {
        let key = r.varint()?;
        let (field, wire) = ((key >> 3) as u32, (key & 7) as u32);
        match (field, wire) {
            (1, 2) => g.nodes.push(parse_node(r.bytes()?)?),
            (2, 2) => g.name = as_str(r.bytes()?),
            (5, 2) => g.initializers.push(parse_tensor(r.bytes()?)?),
            (11, 2) => g.inputs.push(value_info_name(r.bytes()?)?),
            (12, 2) => g.outputs.push(value_info_name(r.bytes()?)?),
            _ => r.skip(wire)?,
        }
    }
    Some(g)
}

fn value_info_name(bytes: &[u8]) -> Option<&str> {
    let mut r = Reader::new(bytes);
    let mut name = "";
    while !r.eof() {
        let key = r.varint()?;
        let (field, wire) = ((key >> 3) as u32, (key & 7) as u32);
        match (field, wire) {
            (1, 2) => name = as_str(r.bytes()?),
            _ => r.skip(wire)?,
        }
    }
    Some(name)
}

fn parse_node(bytes: &[u8]) -> Option<Node<'_>> {
    let mut r = Reader::new(bytes);
    let mut n = Node { op: "", name: "", inputs: Vec::new(), outputs: Vec::new(), attrs: Vec::new() };
    while !r.eof() {
        let key = r.varint()?;
        let (field, wire) = ((key >> 3) as u32, (key & 7) as u32);
        match (field, wire) {
            (1, 2) => n.inputs.push(as_str(r.bytes()?)),
            (2, 2) => n.outputs.push(as_str(r.bytes()?)),
            (3, 2) => n.name = as_str(r.bytes()?),
            (4, 2) => n.op = as_str(r.bytes()?),
            (5, 2) => n.attrs.push(parse_attr(r.bytes()?)?),
            _ => r.skip(wire)?,
        }
    }
    Some(n)
}

fn parse_attr(bytes: &[u8]) -> Option<Attr<'_>> {
    let mut r = Reader::new(bytes);
    let mut a = Attr { name: "", i: 0, f: 0.0, s: &[], ints: Vec::new(), floats: Vec::new(), tensor: None, graph: None };
    while !r.eof() {
        let key = r.varint()?;
        let (field, wire) = ((key >> 3) as u32, (key & 7) as u32);
        match (field, wire) {
            (1, 2) => a.name = as_str(r.bytes()?),
            (2, 5) => a.f = f32::from_bits(r.fixed32()?),
            (3, 0) => a.i = as_i64(r.varint()?),
            (4, 2) => a.s = r.bytes()?,
            (5, 2) => a.tensor = Some(parse_tensor(r.bytes()?)?),
            (6, 2) => a.graph = Some(parse_graph(r.bytes()?)?),
            // floats: packed (wire 2) or repeated (wire 5).
            (7, 2) => {
                let s = r.bytes()?;
                for c in s.chunks_exact(4) {
                    a.floats.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
                }
            }
            (7, 5) => a.floats.push(f32::from_bits(r.fixed32()?)),
            // ints: packed (wire 2) or repeated (wire 0).
            (8, 2) => {
                let mut rr = Reader::new(r.bytes()?);
                while !rr.eof() {
                    a.ints.push(as_i64(rr.varint()?));
                }
            }
            (8, 0) => a.ints.push(as_i64(r.varint()?)),
            _ => r.skip(wire)?,
        }
    }
    Some(a)
}

fn parse_tensor(bytes: &[u8]) -> Option<Tensor<'_>> {
    let mut r = Reader::new(bytes);
    let mut t = Tensor { name: "", dims: Vec::new(), dtype: 0, raw: &[], floats: Vec::new(), ints: Vec::new() };
    while !r.eof() {
        let key = r.varint()?;
        let (field, wire) = ((key >> 3) as u32, (key & 7) as u32);
        match (field, wire) {
            (1, 0) => t.dims.push(as_i64(r.varint()?)),
            (1, 2) => {
                let mut rr = Reader::new(r.bytes()?);
                while !rr.eof() {
                    t.dims.push(as_i64(rr.varint()?));
                }
            }
            (2, 0) => t.dtype = r.varint()? as i32,
            (4, 2) => {
                let s = r.bytes()?;
                for c in s.chunks_exact(4) {
                    t.floats.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
                }
            }
            (4, 5) => t.floats.push(f32::from_bits(r.fixed32()?)),
            // int32_data (field 5): used for small int8/uint8/int32 tensors —
            // notably every quantized-weight zero_point in the voice models —
            // when the exporter doesn't pack them into raw_data.
            (5, 2) => {
                let mut rr = Reader::new(r.bytes()?);
                while !rr.eof() {
                    t.ints.push(as_i64(rr.varint()?));
                }
            }
            (5, 0) => t.ints.push(as_i64(r.varint()?)),
            (7, 2) => {
                let mut rr = Reader::new(r.bytes()?);
                while !rr.eof() {
                    t.ints.push(as_i64(rr.varint()?));
                }
            }
            (7, 0) => t.ints.push(as_i64(r.varint()?)),
            (8, 2) => t.name = as_str(r.bytes()?),
            (9, 2) => t.raw = r.bytes()?,
            _ => r.skip(wire)?,
        }
    }
    Some(t)
}

/// One-line summary of a model: graph name, node count, op histogram (top 8),
/// initializer count — the `/voice models` diagnostic.
pub fn summary(m: &Model<'_>) -> String {
    use alloc::format;
    let mut ops: Vec<(&str, usize)> = Vec::new();
    fn count<'a>(g: &Graph<'a>, ops: &mut Vec<(&'a str, usize)>) {
        for n in &g.nodes {
            match ops.iter_mut().find(|(o, _)| *o == n.op) {
                Some((_, c)) => *c += 1,
                None => ops.push((n.op, 1)),
            }
            for a in &n.attrs {
                if let Some(sub) = &a.graph {
                    count(sub, ops);
                }
            }
        }
    }
    count(&m.graph, &mut ops);
    ops.sort_by(|a, b| b.1.cmp(&a.1));
    let total: usize = ops.iter().map(|(_, c)| c).sum();
    let mut s = format!("'{}': {} nodes, {} initializers; ops:", m.graph.name, total, m.graph.initializers.len());
    for (op, c) in ops.iter().take(8) {
        s.push_str(&format!(" {}x{}", op, c));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The real silero-vad v5 model (630 KB), embedded so the parser is proven
    /// against production ONNX, not a toy fixture.
    static SILERO: &[u8] = include_bytes!("../../../assets/voice/silero_vad.onnx");

    #[test_case]
    fn parses_silero_vad_graph() {
        let m = parse(SILERO).expect("silero_vad.onnx must parse");
        assert!(m.ir_version > 0);
        // The v5 graph has real nodes and named I/O.
        let mut n_ops = m.graph.nodes.len();
        for n in &m.graph.nodes {
            for a in &n.attrs {
                if let Some(g) = &a.graph {
                    n_ops += g.nodes.len();
                }
            }
        }
        assert!(n_ops > 10, "expected a real graph, got {n_ops} nodes");
        assert!(!m.graph.inputs.is_empty() && !m.graph.outputs.is_empty());
        crate::serial_println!("onnx: {}", summary(&m));
    }
}
