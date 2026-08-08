//! HTML → [`Block`]s.
//!
//! Pure, and deliberately a **flow extractor** rather than a rendering engine.
//! The browser already lays HTML out with CSS and flex; printing a page is a
//! different job — what a PDF wants from a document is its *reading order*, and
//! trying to reproduce a screen layout on paper is how you get a PDF with a
//! navigation sidebar down the first page and the article starting on page four.
//!
//! Three things carry all the risk, and each silently produces a wrong document
//! rather than an error:
//!
//! * **`<script>` and `<style>` content is not text.** Missing them puts
//!   JavaScript in the middle of the prose, and it looks like the page really
//!   contained it.
//! * **Whitespace collapses, except inside `<pre>`.** HTML source is indented;
//!   emitting it verbatim gives a document of ragged half-lines.
//! * **Entities must be decoded.** `&amp;` printed literally is the single most
//!   obvious sign that an HTML-to-anything converter is naive.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use super::layout::Block;

/// Extract the document's `<title>`.
pub fn title(src: &str) -> Option<String> {
    let lower = src.to_ascii_lowercase();
    let open = lower.find("<title")?;
    let gt = lower[open..].find('>')? + open + 1;
    let close = lower[gt..].find("</title>")? + gt;
    let t = decode_entities(&collapse(&src[gt..close]));
    (!t.trim().is_empty()).then(|| t.trim().to_string())
}

/// Parse HTML into printable blocks.
pub fn parse(src: &str) -> Vec<Block> {
    let mut out: Vec<Block> = Vec::new();
    let mut text = String::new();
    let bytes: Vec<char> = src.chars().collect();
    let mut i = 0usize;
    // The tag that opened the current block, so `</p>` and `</h2>` know what to
    // emit. A stack rather than a scalar: `<li><p>x</p></li>` is ordinary.
    let mut stack: Vec<String> = Vec::new();
    let mut list_depth: u8 = 0;

    let flush = |text: &mut String, stack: &[String], list_depth: u8, out: &mut Vec<Block>| {
        let t = decode_entities(&collapse(text));
        text.clear();
        if t.trim().is_empty() {
            return;
        }
        let t = t.trim().to_string();
        let ctx = stack.iter().rev().find(|c| is_block_tag(c));
        match ctx.map(|s| s.as_str()) {
            Some("h1") => out.push(Block::Heading { level: 1, text: t }),
            Some("h2") => out.push(Block::Heading { level: 2, text: t }),
            Some("h3") | Some("h4") | Some("h5") | Some("h6") => {
                out.push(Block::Heading { level: 3, text: t })
            }
            Some("li") => out.push(Block::Bullet {
                depth: list_depth.saturating_sub(1).min(3),
                text: t,
            }),
            Some("blockquote") => out.push(Block::Bullet { depth: 0, text: t }),
            _ => out.push(Block::Paragraph(t)),
        }
    };

    while i < bytes.len() {
        if bytes[i] != '<' {
            text.push(bytes[i]);
            i += 1;
            continue;
        }
        // A comment runs to `-->`, and its content is not markup.
        if bytes[i..].starts_with(&['<', '!', '-', '-']) {
            i = find_from(&bytes, i, "-->").map(|p| p + 3).unwrap_or(bytes.len());
            continue;
        }
        let Some(end) = bytes[i..].iter().position(|c| *c == '>').map(|p| p + i) else {
            // An unterminated tag: the rest is not markup, so keep it as text
            // rather than discarding the tail of the document.
            text.extend(&bytes[i..]);
            break;
        };
        let raw: String = bytes[i + 1..end].iter().collect();
        i = end + 1;
        let closing = raw.starts_with('/');
        let name = tag_name(&raw);

        // **Script and style hold code, not prose.** Skip to the matching close.
        if !closing && (name == "script" || name == "style") {
            let close = alloc::format!("</{name}");
            i = find_from(&bytes, i, &close)
                .and_then(|p| bytes[p..].iter().position(|c| *c == '>').map(|q| p + q + 1))
                .unwrap_or(bytes.len());
            continue;
        }
        // `<pre>` is verbatim, and is the one place whitespace survives.
        if !closing && name == "pre" {
            flush(&mut text, &stack, list_depth, &mut out);
            let close = "</pre";
            let stop = find_from(&bytes, i, close).unwrap_or(bytes.len());
            let inner: String = bytes[i..stop].iter().collect();
            // Tags inside `<pre>` (a `<code>` wrapper is universal) are removed,
            // but the line structure is not.
            let cleaned = decode_entities(&strip_tags(&inner));
            out.push(Block::Code(
                cleaned.lines().map(|l| l.trim_end().to_string()).collect(),
            ));
            i = bytes[stop..]
                .iter()
                .position(|c| *c == '>')
                .map(|q| stop + q + 1)
                .unwrap_or(bytes.len());
            continue;
        }

        match name.as_str() {
            "br" => text.push(' '),
            "hr" => {
                flush(&mut text, &stack, list_depth, &mut out);
                out.push(Block::Rule);
            }
            "ul" | "ol" => {
                flush(&mut text, &stack, list_depth, &mut out);
                if closing {
                    list_depth = list_depth.saturating_sub(1);
                } else {
                    list_depth = list_depth.saturating_add(1);
                }
            }
            n if is_block_tag(n) => {
                // A block boundary ends whatever text was accumulating, in both
                // directions: `<p>a<p>b` has no closing tag and must still be
                // two paragraphs.
                flush(&mut text, &stack, list_depth, &mut out);
                if closing {
                    if let Some(p) = stack.iter().rposition(|s| s == n) {
                        stack.truncate(p);
                    }
                } else if !raw.trim_end().ends_with('/') {
                    stack.push(n.to_string());
                }
            }
            // Everything else is inline: it contributes no break.
            _ => {}
        }
    }
    flush(&mut text, &stack, list_depth, &mut out);
    out
}

fn find_from(b: &[char], from: usize, needle: &str) -> Option<usize> {
    let n: Vec<char> = needle.chars().collect();
    let lower: Vec<char> = n.iter().map(|c| c.to_ascii_lowercase()).collect();
    (from..b.len().saturating_sub(n.len() - 1)).find(|&i| {
        b[i..i + n.len()]
            .iter()
            .map(|c| c.to_ascii_lowercase())
            .eq(lower.iter().copied())
    })
}

fn tag_name(raw: &str) -> String {
    raw.trim_start_matches('/')
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric())
        .collect::<String>()
        .to_ascii_lowercase()
}

fn is_block_tag(n: &str) -> bool {
    matches!(
        n,
        "p" | "div"
            | "h1"
            | "h2"
            | "h3"
            | "h4"
            | "h5"
            | "h6"
            | "li"
            | "blockquote"
            | "section"
            | "article"
            | "header"
            | "footer"
            | "main"
            | "aside"
            | "nav"
            | "tr"
            | "table"
            | "figcaption"
    )
}

fn strip_tags(s: &str) -> String {
    let mut out = String::new();
    let mut depth = 0usize;
    for c in s.chars() {
        match c {
            '<' => depth += 1,
            '>' => depth = depth.saturating_sub(1),
            c if depth == 0 => out.push(c),
            _ => {}
        }
    }
    out
}

/// Collapse runs of whitespace to a single space — the HTML rule everywhere
/// except `<pre>`.
pub fn collapse(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut space = false;
    for c in s.chars() {
        if c.is_whitespace() {
            if !space && !out.is_empty() {
                out.push(' ');
            }
            space = true;
        } else {
            out.push(c);
            space = false;
        }
    }
    out
}

/// Decode the entities that actually appear in documents, plus numeric ones.
pub fn decode_entities(s: &str) -> String {
    if !s.contains('&') {
        return s.to_string(); // the overwhelmingly common case
    }
    let mut out = String::with_capacity(s.len());
    let b: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < b.len() {
        if b[i] != '&' {
            out.push(b[i]);
            i += 1;
            continue;
        }
        let Some(semi) = b[i..].iter().take(12).position(|c| *c == ';').map(|p| p + i) else {
            out.push('&');
            i += 1;
            continue;
        };
        let name: String = b[i + 1..semi].iter().collect();
        let ch = match name.as_str() {
            "amp" => Some('&'),
            "lt" => Some('<'),
            "gt" => Some('>'),
            "quot" => Some('"'),
            "apos" | "#39" => Some('\''),
            "nbsp" => Some(' '),
            "mdash" => Some('\u{2014}'),
            "ndash" => Some('\u{2013}'),
            "hellip" => Some('\u{2026}'),
            "copy" => Some('\u{a9}'),
            n => n
                .strip_prefix('#')
                .and_then(|d| match d.strip_prefix('x').or_else(|| d.strip_prefix('X')) {
                    Some(h) => u32::from_str_radix(h, 16).ok(),
                    None => d.parse::<u32>().ok(),
                })
                .and_then(char::from_u32),
        };
        match ch {
            Some(c) => {
                out.push(c);
                i = semi + 1;
            }
            // An unknown entity is left exactly as written rather than dropped:
            // it is more likely a literal `&` than markup we failed to parse.
            None => {
                out.push('&');
                i += 1;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn headings_paragraphs_and_lists() {
        let b = parse(
            "<h1>Title</h1><p>Some prose here.</p><h2>Sub</h2><ul><li>one</li><li>two</li></ul>",
        );
        assert_eq!(b[0], Block::Heading { level: 1, text: "Title".into() });
        assert_eq!(b[1], Block::Paragraph("Some prose here.".into()));
        assert_eq!(b[2], Block::Heading { level: 2, text: "Sub".into() });
        assert_eq!(b[3], Block::Bullet { depth: 0, text: "one".into() });
        assert_eq!(b[4], Block::Bullet { depth: 0, text: "two".into() });
    }

    /// **Script and style are code, not prose.** Including them puts JavaScript
    /// in the middle of the document and it looks like the page said it.
    #[test_case]
    fn script_and_style_are_not_text() {
        let b = parse(
            "<style>body{color:red}</style><p>real text</p>\
             <script>var x = 1; document.write('hi');</script><p>more</p>",
        );
        let all: String = b
            .iter()
            .map(|x| match x {
                Block::Paragraph(t) | Block::Heading { text: t, .. } | Block::Bullet { text: t, .. } => t.clone(),
                _ => String::new(),
            })
            .collect::<Vec<_>>()
            .join(" ");
        assert!(all.contains("real text") && all.contains("more"), "{all}");
        assert!(!all.contains("color:red"), "style leaked: {all}");
        assert!(!all.contains("document.write"), "script leaked: {all}");
    }

    /// Whitespace collapses outside `<pre>`, and survives inside it.
    #[test_case]
    fn whitespace_collapses_except_in_pre() {
        let b = parse("<p>a\n   lot   of\n\n  space</p>");
        assert_eq!(b[0], Block::Paragraph("a lot of space".into()));

        let b = parse("<pre><code>fn main() {\n    println!(\"x\");\n}</code></pre>");
        assert_eq!(
            b[0],
            Block::Code(alloc::vec![
                "fn main() {".to_string(),
                "    println!(\"x\");".to_string(),
                "}".to_string()
            ]),
            "indentation must survive inside pre"
        );
    }

    /// Entities decode — a literal `&amp;` is the giveaway of a naive converter.
    #[test_case]
    fn entities_decode() {
        assert_eq!(decode_entities("a &amp; b"), "a & b");
        assert_eq!(decode_entities("&lt;tag&gt;"), "<tag>");
        assert_eq!(decode_entities("&#65;&#x42;"), "AB");
        assert_eq!(decode_entities("caf&#233;"), "café");
        // An unknown entity stays as written rather than vanishing.
        assert_eq!(decode_entities("a &nosuch; b"), "a &nosuch; b");
        // A bare ampersand is not markup.
        assert_eq!(decode_entities("Tom & Jerry"), "Tom & Jerry");
        // No ampersand at all takes the fast path unchanged.
        assert_eq!(decode_entities("plain text"), "plain text");
    }

    /// Inline tags do not break a paragraph; block tags do, with or without a
    /// closing tag.
    #[test_case]
    fn block_boundaries_split_and_inline_ones_do_not() {
        let b = parse("<p>one <b>bold</b> and <a href='#'>a link</a> inline</p>");
        assert_eq!(b.len(), 1, "inline tags must not split: {b:?}");
        assert_eq!(b[0], Block::Paragraph("one bold and a link inline".into()));

        // Implied end tags: `<p>a<p>b` is two paragraphs.
        let b = parse("<p>first<p>second");
        assert_eq!(b.len(), 2, "{b:?}");
        assert_eq!(b[0], Block::Paragraph("first".into()));
        assert_eq!(b[1], Block::Paragraph("second".into()));
    }

    /// Comments are skipped, including markup inside them.
    #[test_case]
    fn comments_are_skipped() {
        let b = parse("<p>before</p><!-- <h1>not a heading</h1> --><p>after</p>");
        assert_eq!(b.len(), 2, "{b:?}");
        assert!(!b.iter().any(|x| matches!(x, Block::Heading { .. })));
    }

    /// A whole page round-trips into something with the right shape and a title.
    #[test_case]
    fn a_whole_page_extracts() {
        let page = "<!DOCTYPE html><html><head><title>H.264 &amp; You</title>\
                    <style>p{margin:0}</style></head><body>\
                    <nav><a href='/'>home</a></nav>\
                    <h1>H.264</h1><p>A video codec.</p>\
                    <ul><li>CAVLC</li><li>CABAC</li></ul>\
                    <hr><pre>NAL unit</pre></body></html>";
        assert_eq!(title(page).as_deref(), Some("H.264 & You"));
        let b = parse(page);
        assert!(b.iter().any(|x| matches!(x, Block::Heading { level: 1, text } if text == "H.264")));
        assert!(b.iter().any(|x| matches!(x, Block::Bullet { text, .. } if text == "CABAC")));
        assert!(b.iter().any(|x| matches!(x, Block::Rule)));
        assert!(b.iter().any(|x| matches!(x, Block::Code(l) if l == &alloc::vec!["NAL unit".to_string()])));
        assert!(!b.iter().any(|x| matches!(x, Block::Paragraph(t) if t.contains("margin"))));
    }

    /// Malformed input does not panic and does not lose the tail.
    #[test_case]
    fn malformed_html_is_survivable() {
        assert!(parse("").is_empty());
        let b = parse("<p>text with an <unterminated tag");
        assert!(!b.is_empty(), "the tail must not vanish");
        let b = parse("<<<>>>");
        let _ = b; // must simply not panic
    }
}
