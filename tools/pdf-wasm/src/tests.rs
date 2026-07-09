//! Host-side parser tests (`cargo test` in tools/pdf-wasm). Fixtures are
//! built programmatically so every byte offset in the xref is correct by
//! construction: a classic-xref two-page document, a FlateDecode variant
//! (zlib header + DEFLATE stored blocks — valid Flate without a compressor),
//! and an xref-stream + ObjStm variant (the modern layout).

use super::*;
use alloc::vec::Vec;
use alloc::{format, vec};

/// Assemble a PDF from numbered bodies: returns bytes with a classic xref.
fn build_classic(bodies: &[&[u8]]) -> Vec<u8> {
    let mut out: Vec<u8> = b"%PDF-1.4\n".to_vec();
    let mut offs = vec![0usize; bodies.len() + 1];
    for (i, b) in bodies.iter().enumerate() {
        offs[i + 1] = out.len();
        out.extend_from_slice(format!("{} 0 obj\n", i + 1).as_bytes());
        out.extend_from_slice(b);
        out.extend_from_slice(b"\nendobj\n");
    }
    let xref_at = out.len();
    out.extend_from_slice(format!("xref\n0 {}\n", bodies.len() + 1).as_bytes());
    out.extend_from_slice(b"0000000000 65535 f \n");
    for &o in &offs[1..] {
        out.extend_from_slice(format!("{o:010} 00000 n \n").as_bytes());
    }
    out.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R /Info {} 0 R >>\nstartxref\n{}\n%%EOF\n",
            bodies.len() + 1,
            bodies.len(),
            xref_at
        )
        .as_bytes(),
    );
    out
}

/// Wrap raw bytes as a valid zlib/DEFLATE stream using stored blocks.
fn zlib_stored(raw: &[u8]) -> Vec<u8> {
    let mut out = vec![0x78, 0x01]; // CM=8, fcheck ok (0x7801 % 31 == 0)
    for (i, chunk) in raw.chunks(65535).enumerate() {
        let last = (i + 1) * 65535 >= raw.len();
        out.push(if last { 1 } else { 0 }); // BFINAL + BTYPE=00
        out.extend_from_slice(&(chunk.len() as u16).to_le_bytes());
        out.extend_from_slice(&(!(chunk.len() as u16)).to_le_bytes());
        out.extend_from_slice(chunk);
    }
    out.extend_from_slice(&[0, 0, 0, 0]); // adler32 (unchecked)
    out
}

const CONTENT_P1: &[u8] = b"BT /F1 12 Tf 72 700 Td (Hello PDF world) Tj 0 -20 Td (Second line) Tj ET";
const CONTENT_P2: &[u8] = b"BT 72 700 Td [(Frag) -250 (mented)] TJ ET";

fn two_page_pdf(flate: bool) -> Vec<u8> {
    let (c1, c2, filt) = if flate {
        (zlib_stored(CONTENT_P1), zlib_stored(CONTENT_P2), " /Filter /FlateDecode")
    } else {
        (CONTENT_P1.to_vec(), CONTENT_P2.to_vec(), "")
    };
    let mut s5 = format!("<< /Length {}{} >>\nstream\n", c1.len(), filt).into_bytes();
    s5.extend_from_slice(&c1);
    s5.extend_from_slice(b"\nendstream");
    let mut s6 = format!("<< /Length {}{} >>\nstream\n", c2.len(), filt).into_bytes();
    s6.extend_from_slice(&c2);
    s6.extend_from_slice(b"\nendstream");
    build_classic(&[
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>",
        b"<< /Type /Page /Parent 2 0 R /Contents 5 0 R /MediaBox [0 0 612 792] >>",
        b"<< /Type /Page /Parent 2 0 R /Contents 6 0 R >>",
        &s5,
        &s6,
        b"<< /Title (Test Doc) /Author (Chitti) >>",
    ])
}

#[test]
fn classic_xref_two_pages_text() {
    let pdf = two_page_pdf(false);
    let d = digest(&pdf, 20).expect("digest");
    assert!(d.contains("\"pages\":2"), "{d}");
    assert!(d.contains("Hello PDF world"), "{d}");
    assert!(d.contains("Second line"), "{d}");
    assert!(d.contains("Frag mented"), "{d}");
    assert!(d.contains("\"title\":\"Test Doc\""), "{d}");
    assert!(d.contains("\"author\":\"Chitti\""), "{d}");
}

#[test]
fn flate_streams_decode() {
    let pdf = two_page_pdf(true);
    let d = digest(&pdf, 20).expect("digest");
    assert!(d.contains("Hello PDF world"), "{d}");
    assert!(d.contains("Frag mented"), "{d}");
}

#[test]
fn xref_stream_and_objstm() {
    // Layout: 1=Catalog + 2=Pages + 3=Page packed in ObjStm 4; content 5;
    // xref stream 6. Offsets computed as we build.
    let mut out: Vec<u8> = b"%PDF-1.5\n".to_vec();

    // ObjStm payload: header pairs then the three objects.
    let o1 = b"<< /Type /Catalog /Pages 2 0 R >>";
    let o2 = b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>";
    let o3 = b"<< /Type /Page /Parent 2 0 R /Contents 5 0 R >>";
    let (p1, p2, p3) = (0usize, o1.len() + 1, o1.len() + 1 + o2.len() + 1);
    let hdr = format!("1 {p1} 2 {p2} 3 {p3} ");
    let mut payload = hdr.clone().into_bytes();
    payload.extend_from_slice(o1);
    payload.push(b' ');
    payload.extend_from_slice(o2);
    payload.push(b' ');
    payload.extend_from_slice(o3);
    // Object offsets in the payload are relative to /First.
    let first = hdr.len();

    let objstm_at = out.len();
    out.extend_from_slice(format!("4 0 obj\n<< /Type /ObjStm /N 3 /First {first} /Length {} >>\nstream\n", payload.len()).as_bytes());
    out.extend_from_slice(&payload);
    out.extend_from_slice(b"\nendstream\nendobj\n");

    let content_at = out.len();
    out.extend_from_slice(format!("5 0 obj\n<< /Length {} >>\nstream\n", CONTENT_P1.len()).as_bytes());
    out.extend_from_slice(CONTENT_P1);
    out.extend_from_slice(b"\nendstream\nendobj\n");

    // xref stream: W [1 4 2]; entries for objects 0..=6.
    let xref_at = out.len();
    let mut entries: Vec<u8> = Vec::new();
    let mut push = |t: u8, a: u32, b: u16, e: &mut Vec<u8>| {
        e.push(t);
        e.extend_from_slice(&a.to_be_bytes());
        e.extend_from_slice(&b.to_be_bytes());
    };
    push(0, 0, 0, &mut entries); // 0: free
    push(2, 4, 0, &mut entries); // 1: in objstm 4, idx 0
    push(2, 4, 1, &mut entries); // 2
    push(2, 4, 2, &mut entries); // 3
    push(1, objstm_at as u32, 0, &mut entries); // 4
    push(1, content_at as u32, 0, &mut entries); // 5
    push(1, xref_at as u32, 0, &mut entries); // 6
    out.extend_from_slice(
        format!(
            "6 0 obj\n<< /Type /XRef /Size 7 /W [1 4 2] /Root 1 0 R /Length {} >>\nstream\n",
            entries.len()
        )
        .as_bytes(),
    );
    out.extend_from_slice(&entries);
    out.extend_from_slice(b"\nendstream\nendobj\n");
    out.extend_from_slice(format!("startxref\n{xref_at}\n%%EOF\n").as_bytes());

    let d = digest(&out, 20).expect("digest");
    assert!(d.contains("\"pages\":1"), "{d}");
    assert!(d.contains("Hello PDF world"), "{d}");
}

#[test]
fn rejects_garbage_and_encrypted() {
    assert!(digest(b"not a pdf at all", 5).is_err());
    // Encrypted: trailer carries /Encrypt.
    let mut pdf = two_page_pdf(false);
    let t = find(&pdf, b"/Root").unwrap();
    // Splice an /Encrypt entry into the trailer dict.
    pdf.splice(t..t, b"/Encrypt 9 0 R ".iter().copied());
    // startxref offset unchanged (trailer text after xref table), so parse works.
    let e = digest(&pdf, 5);
    assert!(e.is_err(), "{e:?}");
}

#[test]
fn b64_roundtrip() {
    assert_eq!(crate::b64_decode("aGVsbG8=").unwrap(), b"hello");
    assert_eq!(crate::b64_decode("aGVsbG8h").unwrap(), b"hello!");
    assert!(crate::b64_decode("a!b").is_none());
}

#[test]
fn png_predictor_up() {
    // Two 3-byte rows, filter type 2 (Up): row2 = raw + row1.
    let data = [2u8, 10, 20, 30, 2, 1, 2, 3];
    let out = unpredict(&data, 12, 3, 1, 8).unwrap();
    assert_eq!(out, [10, 20, 30, 11, 22, 33]);
}
