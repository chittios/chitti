//! Markdown → [`Block`]s.
//!
//! Pure. A deliberately small subset — headings, paragraphs, bullets, ordered
//! items, fenced and indented code, rules, block quotes — because this feeds a
//! *printer*, not a browser: anything it cannot lay out is better shown as the
//! text it was than dropped.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use super::layout::Block;

/// Parse markdown into blocks.
pub fn parse(src: &str) -> Vec<Block> {
    let mut out: Vec<Block> = Vec::new();
    let mut para: Vec<String> = Vec::new();
    let mut code: Vec<String> = Vec::new();
    let mut in_fence = false;

    let flush_para = |para: &mut Vec<String>, out: &mut Vec<Block>| {
        if !para.is_empty() {
            out.push(Block::Paragraph(para.join(" ")));
            para.clear();
        }
    };

    for raw in src.lines() {
        let line = raw.trim_end();
        let t = line.trim_start();

        // A fence toggles verbatim mode. **The closing fence must be recognised
        // even when the content looks like markdown**, or a document containing
        // a `#` in a code block loses the rest of its structure.
        if t.starts_with("```") || t.starts_with("~~~") {
            if in_fence {
                out.push(Block::Code(core::mem::take(&mut code)));
                in_fence = false;
            } else {
                flush_para(&mut para, &mut out);
                in_fence = true;
            }
            continue;
        }
        if in_fence {
            code.push(line.to_string());
            continue;
        }

        if t.is_empty() {
            flush_para(&mut para, &mut out);
            continue;
        }
        // A rule: three or more of - * _ and nothing else.
        if t.len() >= 3
            && (t.chars().all(|c| c == '-') || t.chars().all(|c| c == '*') || t.chars().all(|c| c == '_'))
        {
            flush_para(&mut para, &mut out);
            out.push(Block::Rule);
            continue;
        }
        if let Some(rest) = t.strip_prefix('#') {
            let level = 1 + rest.chars().take_while(|c| *c == '#').count() as u8;
            let text = rest.trim_start_matches('#').trim();
            // `#hashtag` is not a heading — ATX requires a space.
            if rest.starts_with(' ') || rest.starts_with('#') {
                flush_para(&mut para, &mut out);
                out.push(Block::Heading {
                    level: level.min(6),
                    text: inline(text),
                });
                continue;
            }
        }
        // Indented code: four spaces or a tab, outside a paragraph **and not a
        // list item**. A nested bullet is indented too — `    - deeper` is a
        // depth-2 item, not code — and checking the indent first turned every
        // list nested past one level into a code block.
        let is_list_item = bullet_text(t).is_some() || ordered_text(t).is_some();
        if (line.starts_with("    ") || line.starts_with('\t')) && para.is_empty() && !is_list_item {
            let stripped = line.strip_prefix("    ").or_else(|| line.strip_prefix('\t')).unwrap_or(line);
            code.push(stripped.to_string());
            continue;
        }
        if !code.is_empty() {
            out.push(Block::Code(core::mem::take(&mut code)));
        }
        if let Some(text) = bullet_text(t) {
            flush_para(&mut para, &mut out);
            let depth = ((line.len() - t.len()) / 2).min(3) as u8;
            out.push(Block::Bullet {
                depth,
                text: inline(&text),
            });
            continue;
        }
        if let Some((n, text)) = ordered_text(t) {
            flush_para(&mut para, &mut out);
            let depth = ((line.len() - t.len()) / 2).min(3) as u8;
            out.push(Block::Bullet {
                depth,
                text: alloc::format!("{n}. {}", inline(&text)),
            });
            continue;
        }
        if let Some(q) = t.strip_prefix("> ") {
            flush_para(&mut para, &mut out);
            out.push(Block::Bullet {
                depth: 0,
                text: inline(q),
            });
            continue;
        }
        para.push(inline(t));
    }
    if in_fence || !code.is_empty() {
        out.push(Block::Code(code));
    }
    flush_para(&mut para, &mut out);
    out
}

fn bullet_text(t: &str) -> Option<String> {
    for m in ["- ", "* ", "+ "] {
        if let Some(r) = t.strip_prefix(m) {
            return Some(r.to_string());
        }
    }
    None
}

fn ordered_text(t: &str) -> Option<(u32, String)> {
    let digits: String = t.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        return None;
    }
    let rest = &t[digits.len()..];
    let rest = rest.strip_prefix(". ").or_else(|| rest.strip_prefix(") "))?;
    Some((digits.parse().ok()?, rest.to_string()))
}

/// Strip inline markup to plain text.
///
/// The base-14 fonts give us no italic/bold *within* a line without splitting it
/// into runs, which the layout does not do — so emphasis markers are removed
/// rather than shown. Leaving them in would print literal asterisks, which reads
/// as a bug; dropping the text would lose content.
pub fn inline(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let b: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            // `**bold**` / `__bold__`
            '*' | '_' if i + 1 < b.len() && b[i + 1] == b[i] => i += 2,
            '*' | '_' => i += 1,
            // `` `code` `` keeps its content, loses the ticks.
            '`' => i += 1,
            // `[text](url)` keeps the text and appends the target, since a
            // printed page cannot be clicked and a bare label loses the link.
            '[' => {
                let close = b[i..].iter().position(|c| *c == ']').map(|p| p + i);
                match close {
                    Some(c) if c + 1 < b.len() && b[c + 1] == '(' => {
                        let end = b[c..].iter().position(|x| *x == ')').map(|p| p + c);
                        let label: String = b[i + 1..c].iter().collect();
                        match end {
                            Some(e) => {
                                let url: String = b[c + 2..e].iter().collect();
                                out.push_str(&label);
                                if !url.is_empty() && url != label {
                                    out.push_str(&alloc::format!(" ({url})"));
                                }
                                i = e + 1;
                            }
                            None => {
                                out.push_str(&label);
                                i = c + 1;
                            }
                        }
                    }
                    _ => {
                        out.push('[');
                        i += 1;
                    }
                }
            }
            c => {
                out.push(c);
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
    fn headings_paragraphs_and_bullets() {
        let b = parse("# Top\n\nsome text\nmore text\n\n## Sub\n\n- one\n- two\n");
        assert_eq!(b[0], Block::Heading { level: 1, text: "Top".into() });
        // Consecutive lines join into one paragraph, as markdown says.
        assert_eq!(b[1], Block::Paragraph("some text more text".into()));
        assert_eq!(b[2], Block::Heading { level: 2, text: "Sub".into() });
        assert_eq!(b[3], Block::Bullet { depth: 0, text: "one".into() });
        assert_eq!(b[4], Block::Bullet { depth: 0, text: "two".into() });
    }

    /// **A fence protects its content.** A `#` inside a code block is not a
    /// heading, and missing that loses the rest of the document's structure.
    #[test_case]
    fn fenced_code_is_verbatim() {
        let b = parse("text\n\n```rust\n# not a heading\n- not a bullet\n```\n\nafter\n");
        let code = b.iter().find_map(|x| match x {
            Block::Code(l) => Some(l.clone()),
            _ => None,
        });
        assert_eq!(
            code,
            Some(alloc::vec!["# not a heading".to_string(), "- not a bullet".to_string()])
        );
        assert!(b.iter().any(|x| matches!(x, Block::Paragraph(p) if p == "after")));
        assert!(
            !b.iter().any(|x| matches!(x, Block::Heading { .. })),
            "nothing inside the fence may become a heading"
        );
    }

    /// An unterminated fence still yields its content rather than dropping it.
    #[test_case]
    fn an_unclosed_fence_keeps_its_content() {
        let b = parse("```\nline one\nline two\n");
        assert_eq!(
            b.last(),
            Some(&Block::Code(alloc::vec!["line one".into(), "line two".into()]))
        );
    }

    #[test_case]
    fn rules_and_ordered_lists() {
        let b = parse("---\n\n1. first\n2. second\n");
        assert_eq!(b[0], Block::Rule);
        assert_eq!(b[1], Block::Bullet { depth: 0, text: "1. first".into() });
        assert_eq!(b[2], Block::Bullet { depth: 0, text: "2. second".into() });
        // `***` is a rule; `*text*` is emphasis, not one.
        assert_eq!(parse("***\n")[0], Block::Rule);
        assert!(matches!(parse("*hi*\n")[0], Block::Paragraph(_)));
    }

    /// `#hashtag` is not a heading — ATX requires the space.
    #[test_case]
    fn a_hash_without_a_space_is_not_a_heading() {
        assert!(matches!(parse("#hashtag here\n")[0], Block::Paragraph(_)));
        assert!(matches!(parse("# real heading\n")[0], Block::Heading { .. }));
    }

    /// Inline markup is stripped, and a link keeps both its label and target —
    /// a printed page cannot be clicked.
    #[test_case]
    fn inline_markup_is_flattened() {
        assert_eq!(inline("**bold** and *em* and `code`"), "bold and em and code");
        assert_eq!(inline("see [the docs](https://x.dev)"), "see the docs (https://x.dev)");
        // A label equal to its target is not repeated.
        assert_eq!(inline("[https://x.dev](https://x.dev)"), "https://x.dev");
        // An unclosed bracket is left as typed rather than eating the line.
        assert_eq!(inline("an [unclosed thing"), "an [unclosed thing");
    }

    /// Nested bullets indent.
    #[test_case]
    fn nested_bullets_carry_depth() {
        let b = parse("- top\n  - nested\n    - deeper\n");
        assert_eq!(b[0], Block::Bullet { depth: 0, text: "top".into() });
        assert_eq!(b[1], Block::Bullet { depth: 1, text: "nested".into() });
        assert_eq!(b[2], Block::Bullet { depth: 2, text: "deeper".into() });
    }

    /// Empty input is an empty document, not a panic.
    #[test_case]
    fn empty_input_is_empty() {
        assert!(parse("").is_empty());
        assert!(parse("\n\n\n").is_empty());
    }
}
