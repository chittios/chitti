//! **PDF creation** — text, markdown and HTML in, a real PDF out.
//!
//! The reader side of PDF lives in `tools/pdf-wasm` (parsing) and
//! `tools/pdfrender-wasm` (rasterising), sandboxed because they consume files
//! from anywhere. This is the opposite direction: it consumes only text this OS
//! produced, so it is ordinary native code — and being native is what lets it
//! reuse `image::deflate::zlib_compress` for FlateDecode streams.
//!
//! * [`write`] — objects, xref, streams. The file format.
//! * [`layout`] — base-14 metrics, wrapping, pagination. The geometry.
//! * [`md`] / [`html`] — source formats to [`layout::Block`].

pub mod html;
pub mod layout;
pub mod md;
pub mod write;

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use layout::{Block, Font, Page};
use write::{escape_literal, escape_name, ObjId, Pdf};

/// Build a PDF from already-parsed blocks.
///
/// `title` goes in the document information dictionary, which is what a reader
/// shows in its window chrome and what `pdf_digest` reports.
pub fn build(blocks: &[Block], title: &str, page: Page) -> Vec<u8> {
    let pages = layout::flow(blocks, page);
    let mut pdf = Pdf::new();

    // Fonts first: every page's resource dictionary refers to them, and they do
    // not refer to anything, so they can be written before the tree exists.
    let mut font_ids: Vec<(Font, ObjId)> = Vec::new();
    for f in Font::all() {
        let id = pdf.dict(&alloc::format!(
            "/Type /Font /Subtype /Type1 /BaseFont /{} /Encoding /WinAnsiEncoding",
            escape_name(f.base_name())
        ));
        font_ids.push((f, id));
    }
    let font_res = font_ids
        .iter()
        .map(|(f, id)| alloc::format!("/{} {}", f.res(), id.r()))
        .collect::<Vec<_>>()
        .join(" ");

    // The page tree's `/Kids` needs every page object and each page's `/Parent`
    // needs the tree — a cycle. It is broken by **computing** the tree's object
    // number rather than reserving it: object numbers are assigned in order, and
    // the loop below writes exactly two objects per page (a content stream and
    // the page), so the tree lands immediately after them. `debug_assert` at the
    // end checks that arithmetic held; if it ever stops holding, every
    // `/Parent` dangles and readers differ in how loudly they say so.
    let pages_id = ObjId(pdf.next_id().0 + 2 * pages.len() as u32);
    let mut kids: Vec<ObjId> = Vec::new();

    for lines in &pages {
        let mut content = String::new();
        content.push_str("BT\n");
        let mut cur: Option<(Font, f32)> = None;
        for l in lines {
            if cur != Some((l.font, l.size)) {
                content.push_str(&alloc::format!("/{} {} Tf\n", l.font.res(), l.size));
                cur = Some((l.font, l.size));
            }
            // `Td` is *relative* to the previous line's origin, so absolute
            // placement uses `1 0 0 1 x y Tm` — a text matrix — instead. Using
            // `Td` with absolute numbers walks the text down and off the page,
            // one line's offset at a time.
            content.push_str(&alloc::format!(
                "1 0 0 1 {:.2} {:.2} Tm ({}) Tj\n",
                l.x,
                l.y,
                escape_literal(&l.text)
            ));
        }
        content.push_str("ET\n");
        let stream = pdf.stream("", content.as_bytes());
        let page_id = pdf.dict(&alloc::format!(
            "/Type /Page /Parent {} /MediaBox [0 0 {:.0} {:.0}] \
             /Resources << /Font << {} >> >> /Contents {}",
            pages_id.r(),
            page.width,
            page.height,
            font_res,
            stream.r()
        ));
        kids.push(page_id);
    }

    let kid_refs = kids.iter().map(|k| k.r()).collect::<Vec<_>>().join(" ");
    let written_pages = pdf.dict(&alloc::format!(
        "/Type /Pages /Count {} /Kids [{kid_refs}]",
        kids.len()
    ));
    let info = pdf.dict(&alloc::format!(
        "/Title ({}) /Producer (ChittiOS) /Creator (ChittiOS)",
        escape_literal(title)
    ));
    let root = pdf.dict(&alloc::format!("/Type /Catalog /Pages {}", written_pages.r()));
    // The reservation must have held: pages point at `pages_id`, so if the tree
    // landed on a different number every `/Parent` dangles. Readers vary in how
    // loudly they complain, so it is checked here rather than discovered later.
    debug_assert_eq!(
        written_pages, pages_id,
        "the page tree did not land on its reserved object number"
    );
    pdf.finish(root, Some(info))
}

/// Markdown in, PDF out.
pub fn from_markdown(source: &str, title: &str) -> Vec<u8> {
    build(&md::parse(source), title, Page::default())
}

/// HTML in, PDF out.
pub fn from_html(source: &str, title: &str) -> Vec<u8> {
    let blocks = html::parse(source);
    let t = if title.is_empty() {
        html::title(source).unwrap_or_default()
    } else {
        title.to_string()
    };
    build(&blocks, &t, Page::default())
}

/// Plain text in, PDF out — every line a paragraph, blank lines as spacing.
pub fn from_text(source: &str, title: &str) -> Vec<u8> {
    let mut blocks = Vec::new();
    for para in source.split("\n\n") {
        if para.trim().is_empty() {
            continue;
        }
        blocks.push(Block::Paragraph(para.replace('\n', " ").trim().to_string()));
    }
    build(&blocks, title, Page::default())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A built document has the structure a reader walks: catalogue → pages →
    /// page → contents, with every `/Parent` resolving.
    #[test_case]
    fn a_built_document_has_a_complete_page_tree() {
        let out = from_markdown("# Title\n\nSome body text.\n", "T");
        let text = alloc::string::String::from_utf8_lossy(&out).into_owned();
        assert!(text.contains("/Type /Catalog"));
        assert!(text.contains("/Type /Pages"));
        assert!(text.contains("/Type /Page "), "a page object");
        assert!(text.contains("/MediaBox [0 0 612 792]"));
        assert!(text.contains("/BaseFont /Helvetica"));
        assert!(text.contains("/Title (T)"));

        // Every `N 0 R` reference names an object that exists.
        let size: u32 = text
            .rsplit("/Size ")
            .next()
            .unwrap()
            .split(|c: char| !c.is_ascii_digit())
            .next()
            .unwrap()
            .parse()
            .unwrap();
        let mut refs = 0;
        for part in text.split(" 0 R") {
            if let Some(n) = part.rsplit(|c: char| !c.is_ascii_digit()).next() {
                if let Ok(n) = n.parse::<u32>() {
                    assert!(n >= 1 && n < size, "reference {n} is outside 1..{size}");
                    refs += 1;
                }
            }
        }
        assert!(refs >= 4, "expected several references, saw {refs}");
    }

    /// **Absolute placement uses `Tm`, not `Td`.** `Td` is relative to the
    /// previous line, so absolute numbers with it walk the text off the page.
    #[test_case]
    fn text_is_placed_with_an_absolute_matrix() {
        let out = from_markdown("one\n\ntwo\n\nthree\n", "T");
        let s = find_content_stream(&out);
        assert!(s.contains(" Tm "), "absolute placement: {s}");
        assert!(!s.contains(" Td "), "Td is relative and must not be used: {s}");
        assert!(s.starts_with("BT\n") && s.trim_end().ends_with("ET"));
    }

    /// The content stream really contains the words, escaped.
    #[test_case]
    fn the_content_stream_holds_the_text() {
        let out = from_markdown("a paren \\(x\\) and a backslash\n", "T");
        let s = find_content_stream(&out);
        assert!(s.contains("paren"), "{s}");
        assert!(s.contains("\\("), "delimiters must be escaped: {s}");
    }

    /// A long document produces several `/Type /Page` objects and a matching
    /// `/Count`.
    #[test_case]
    fn a_long_document_has_many_pages() {
        let src: alloc::string::String = (0..200)
            .map(|i| alloc::format!("Paragraph {i} with enough words to take a line.\n\n"))
            .collect();
        let out = from_markdown(&src, "Long");
        let text = alloc::string::String::from_utf8_lossy(&out).into_owned();
        let n = text.matches("/Type /Page ").count();
        assert!(n > 1, "expected several pages, got {n}");
        assert!(text.contains(&alloc::format!("/Count {n}")), "/Count must match {n}");
    }

    /// Inflate the first content stream back to text, using our own inflater.
    ///
    /// Searches the **raw bytes**. A `from_utf8_lossy` view cannot be used to
    /// compute offsets into the file: compressed streams are not valid UTF-8, so
    /// every invalid sequence becomes a 3-byte replacement character and each
    /// one shifts all later indices — the offsets come out plausible and wrong.
    fn find_content_stream(out: &[u8]) -> alloc::string::String {
        fn find(h: &[u8], n: &[u8], from: usize) -> Option<usize> {
            (from..=h.len().saturating_sub(n.len())).find(|&i| &h[i..i + n.len()] == n)
        }
        let mut at = 0usize;
        while let Some(len_at) = find(out, b"/Length ", at) {
            let digits: alloc::string::String = out[len_at + 8..]
                .iter()
                .take_while(|b| b.is_ascii_digit())
                .map(|b| *b as char)
                .collect();
            let Ok(len) = digits.parse::<usize>() else {
                at = len_at + 8;
                continue;
            };
            let Some(s) = find(out, b"stream\n", len_at) else { break };
            let s = s + 7;
            if s + len <= out.len() {
                if let Ok((data, _)) = crate::image::inflate::zlib_decompress_len(&out[s..s + len]) {
                    if data.starts_with(b"BT") {
                        return alloc::string::String::from_utf8_lossy(&data).into_owned();
                    }
                }
            }
            at = len_at + 8;
        }
        panic!("no content stream found");
    }
}
