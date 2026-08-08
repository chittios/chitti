//! A minimal PDF **writer** — indirect objects, the cross-reference table, and
//! FlateDecode content streams.
//!
//! Pure byte assembly. It is native rather than wasm on purpose: the *reader*
//! is sandboxed because it parses attacker-controlled files, while a writer
//! consumes only text this OS generated, so the argument for confining it does
//! not apply — and native gets `image::deflate::zlib_compress` and the font
//! metrics directly.
//!
//! Four details are load-bearing, and each produces a file that some readers
//! open and others reject, which is the worst failure mode available:
//!
//! * **An xref entry is exactly 20 bytes** — `nnnnnnnnnn ggggg n\r\n`, ten
//!   digits, five digits, a one-letter type and a two-byte terminator. A
//!   19-byte entry (a bare `\n`) leaves every later offset in the table
//!   unreachable, and a reader that scans instead of seeking will still open
//!   it, so it appears to work.
//! * **Offsets are absolute from the start of the file**, including the
//!   `%PDF-1.7` header, so the table can only be built after the body.
//! * **`/Size` is the highest object number plus one**, counting the free
//!   object 0 — not the number of objects written.
//! * **Object 0 is the head of the free list**, `0000000000 65535 f`, and must
//!   be present even though nothing refers to it.

use alloc::string::String;
use alloc::vec::Vec;

/// A PDF being built. Objects are appended and numbered from 1.
pub struct Pdf {
    /// Serialised body, starting with the header.
    buf: Vec<u8>,
    /// Byte offset of each object, indexed by number - 1.
    offsets: Vec<usize>,
}

/// An object number, so a reference cannot be confused with a length or a size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ObjId(pub u32);

impl ObjId {
    /// The `N 0 R` reference form.
    pub fn r(&self) -> String {
        alloc::format!("{} 0 R", self.0)
    }
}

impl Default for Pdf {
    fn default() -> Self {
        Self::new()
    }
}

impl Pdf {
    pub fn new() -> Self {
        // The binary comment on line 2 tells any tool that transfers this file
        // that it is not text; without it a naive transfer may translate line
        // endings and corrupt every stream.
        let mut buf = Vec::new();
        buf.extend_from_slice(b"%PDF-1.7\n");
        buf.extend_from_slice(b"%\xE2\xE3\xCF\xD3\n");
        Self {
            buf,
            offsets: Vec::new(),
        }
    }

    /// The number the next [`Pdf::object`] will get — needed when an object has
    /// to reference one that does not exist yet (a page's parent, say).
    pub fn next_id(&self) -> ObjId {
        ObjId(self.offsets.len() as u32 + 1)
    }

    /// Append an object whose body is `body` (everything between `obj` and
    /// `endobj`).
    pub fn object(&mut self, body: &[u8]) -> ObjId {
        let id = self.next_id();
        self.offsets.push(self.buf.len());
        self.buf.extend_from_slice(alloc::format!("{} 0 obj\n", id.0).as_bytes());
        self.buf.extend_from_slice(body);
        self.buf.extend_from_slice(b"\nendobj\n");
        id
    }

    pub fn dict(&mut self, body: &str) -> ObjId {
        self.object(alloc::format!("<< {body} >>").as_bytes())
    }

    /// A stream object, **Flate-compressed**.
    ///
    /// `/Length` must be the *compressed* length — the count of bytes between
    /// `stream` and `endstream` — not the original size.
    pub fn stream(&mut self, extra_dict: &str, data: &[u8]) -> ObjId {
        let z = crate::image::deflate::zlib_compress(data);
        let mut body = Vec::new();
        body.extend_from_slice(
            alloc::format!(
                "<< /Length {} /Filter /FlateDecode{}{} >>\nstream\n",
                z.len(),
                if extra_dict.is_empty() { "" } else { " " },
                extra_dict
            )
            .as_bytes(),
        );
        body.extend_from_slice(&z);
        body.extend_from_slice(b"\nendstream");
        self.object(&body)
    }

    /// An uncompressed stream, for data that is already compressed (a JPEG) or
    /// that a human may want to read in the file.
    pub fn raw_stream(&mut self, extra_dict: &str, data: &[u8]) -> ObjId {
        let mut body = Vec::new();
        body.extend_from_slice(
            alloc::format!(
                "<< /Length {}{}{} >>\nstream\n",
                data.len(),
                if extra_dict.is_empty() { "" } else { " " },
                extra_dict
            )
            .as_bytes(),
        );
        body.extend_from_slice(data);
        body.extend_from_slice(b"\nendstream");
        self.object(&body)
    }

    /// Finish: write the xref table and trailer, and return the file.
    ///
    /// `root` is the document catalogue.
    pub fn finish(mut self, root: ObjId, info: Option<ObjId>) -> Vec<u8> {
        let startxref = self.buf.len();
        let size = self.offsets.len() as u32 + 1;
        self.buf
            .extend_from_slice(alloc::format!("xref\n0 {size}\n").as_bytes());
        // Object 0: the head of the free list. Present even though unreferenced.
        self.buf.extend_from_slice(b"0000000000 65535 f\r\n");
        for off in &self.offsets {
            // Exactly 20 bytes: 10 + 1 + 5 + 1 + 1 + 2.
            self.buf
                .extend_from_slice(alloc::format!("{off:010} 00000 n\r\n").as_bytes());
        }
        let info_entry = match info {
            Some(i) => alloc::format!(" /Info {}", i.r()),
            None => String::new(),
        };
        self.buf.extend_from_slice(
            alloc::format!(
                "trailer\n<< /Size {size} /Root {}{info_entry} >>\nstartxref\n{startxref}\n%%EOF\n",
                root.r()
            )
            .as_bytes(),
        );
        self.buf
    }
}

/// Escape a string for a PDF **literal string** `( … )`.
///
/// `\`, `(` and `)` must be escaped or the string ends early and everything
/// after it is parsed as operators — which usually still renders *something*,
/// just not the document. Bytes outside printable ASCII are written as octal
/// escapes so the file stays 7-bit and cannot be damaged by a text-mode
/// transfer.
pub fn escape_literal(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for b in s.bytes() {
        match b {
            b'\\' => out.push_str("\\\\"),
            b'(' => out.push_str("\\("),
            b')' => out.push_str("\\)"),
            b'\n' => out.push_str("\\n"),
            b'\r' => out.push_str("\\r"),
            b'\t' => out.push_str("\\t"),
            0x20..=0x7e => out.push(b as char),
            other => out.push_str(&alloc::format!("\\{other:03o}")),
        }
    }
    out
}

/// Escape a PDF **name** (`/Foo`), where `#` hex-escapes anything awkward.
pub fn escape_name(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'+' => out.push(b as char),
            other => out.push_str(&alloc::format!("#{other:02X}")),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The structural invariants a reader depends on.
    #[test_case]
    fn the_xref_table_is_well_formed() {
        let mut pdf = Pdf::new();
        let a = pdf.dict("/Type /Catalog");
        let b = pdf.stream("", b"hello");
        let out = pdf.finish(a, None);
        let text = alloc::string::String::from_utf8_lossy(&out).into_owned();

        assert!(text.starts_with("%PDF-1.7\n"), "header");
        assert!(text.trim_end().ends_with("%%EOF"), "trailer");
        assert_eq!(b.0, 2, "objects number from 1");

        // `/Size` counts object 0.
        assert!(text.contains("/Size 3"), "{text}");

        let x = text.find("xref\n").expect("an xref table");
        let table = &text[x + 5..];
        assert!(table.starts_with("0 3\n"), "subsection header: {}", &table[..16]);
        let entries = &table["0 3\n".len()..];
        // **Twenty bytes each**, free entry first.
        assert_eq!(&entries[..20], "0000000000 65535 f\r\n");
        for i in 0..2 {
            let e = &entries[20 + i * 20..40 + i * 20];
            assert_eq!(e.len(), 20, "entry {i} must be exactly 20 bytes");
            assert!(e.ends_with(" n\r\n"), "entry {i}: {e:?}");
        }

        // `startxref` points at the literal `xref` keyword.
        let sx: usize = text
            .rsplit("startxref\n")
            .next()
            .unwrap()
            .lines()
            .next()
            .unwrap()
            .parse()
            .unwrap();
        assert_eq!(&out[sx..sx + 4], b"xref");
    }

    /// Every offset in the table lands on that object's `N 0 obj`.
    #[test_case]
    fn xref_offsets_point_at_their_objects() {
        let mut pdf = Pdf::new();
        for i in 0..6 {
            pdf.dict(&alloc::format!("/N {i}"));
        }
        let root = pdf.dict("/Type /Catalog");
        let out = pdf.finish(root, None);
        let text = alloc::string::String::from_utf8_lossy(&out).into_owned();
        let entries = &text[text.find("xref\n").unwrap() + 5..];
        let entries = &entries[entries.find('\n').unwrap() + 1..];
        for n in 1..=7u32 {
            let e = &entries[(n as usize) * 20..(n as usize) * 20 + 20];
            let off: usize = e[..10].parse().expect("ten digits");
            let here = alloc::string::String::from_utf8_lossy(&out[off..off + 12]).into_owned();
            assert!(
                here.starts_with(&alloc::format!("{n} 0 obj")),
                "object {n}: offset {off} points at {here:?}"
            );
        }
    }

    /// A stream's `/Length` is the **compressed** size, and the bytes really
    /// inflate back — checked with our own inflater, a different implementation
    /// from the compressor.
    ///
    /// Offsets are computed on the **raw bytes**, never on a `from_utf8_lossy`
    /// view: a compressed stream is not valid UTF-8, so every invalid sequence
    /// becomes a 3-byte replacement character and silently shifts every later
    /// index. That produced a plausible offset pointing into the middle of the
    /// stream and an inflater complaining the data was "not deflate".
    #[test_case]
    fn stream_length_is_the_compressed_length() {
        fn find(h: &[u8], n: &[u8], from: usize) -> Option<usize> {
            (from..=h.len().saturating_sub(n.len())).find(|&i| &h[i..i + n.len()] == n)
        }
        let payload = b"BT /F1 12 Tf 72 720 Td (hello world) Tj ET\n".repeat(20);
        let mut pdf = Pdf::new();
        let _ = pdf.stream("", &payload);
        let root = pdf.dict("/Type /Catalog");
        let out = pdf.finish(root, None);

        let len_at = find(&out, b"/Length ", 0).expect("a stream length") + 8;
        let digits: alloc::string::String = out[len_at..]
            .iter()
            .take_while(|b| b.is_ascii_digit())
            .map(|b| *b as char)
            .collect();
        let len: usize = digits.parse().unwrap();
        assert!(len < payload.len(), "the payload must actually compress");

        let s = find(&out, b"stream\n", len_at).unwrap() + 7;
        let (back, _) = crate::image::inflate::zlib_decompress_len(&out[s..s + len])
            .expect("our own inflater must read it");
        assert_eq!(back, payload, "the stream must round-trip");
    }

    /// Literal-string escaping. An unescaped `)` ends the string early and every
    /// byte after it is read as an operator.
    #[test_case]
    fn literal_strings_are_escaped() {
        assert_eq!(escape_literal("plain"), "plain");
        assert_eq!(escape_literal("a(b)c"), "a\\(b\\)c");
        assert_eq!(escape_literal("back\\slash"), "back\\\\slash");
        assert_eq!(escape_literal("line\nbreak"), "line\\nbreak");
        // Non-ASCII becomes octal, so the file stays 7-bit.
        assert_eq!(escape_literal("é"), "\\303\\251");
        // The result contains no bare delimiter.
        for s in ["()", "\\", "((()))", "mixed (a\\b) c"] {
            let e = escape_literal(s);
            let mut prev_escape = false;
            for c in e.chars() {
                if !prev_escape {
                    assert!(c != '(' && c != ')', "unescaped delimiter in {e:?}");
                }
                prev_escape = c == '\\' && !prev_escape;
            }
        }
    }

    /// Names hex-escape anything that is not a plain character.
    #[test_case]
    fn names_are_escaped() {
        assert_eq!(escape_name("Helvetica"), "Helvetica");
        assert_eq!(escape_name("F1"), "F1");
        assert_eq!(escape_name("a b"), "a#20b");
        assert_eq!(escape_name("a/b"), "a#2Fb");
    }

    /// `next_id` predicts the next number, which is how a page references a
    /// parent that has not been written yet.
    #[test_case]
    fn next_id_predicts_the_following_object() {
        let mut pdf = Pdf::new();
        let predicted = pdf.next_id();
        let actual = pdf.dict("/Type /Test");
        assert_eq!(predicted, actual);
        assert_eq!(predicted.r(), "1 0 R");
        assert_eq!(pdf.next_id().0, 2);
    }
}
