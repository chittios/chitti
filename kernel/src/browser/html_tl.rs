//! `tl` as an **alternative** HTML tree builder — `/html ours|tl`.
//!
//! [`crate::browser::html`] stays the default and is not removed. This module
//! runs the vendored `tl` crate (`third_party/tl`, MIT, `no_std`-converted)
//! behind the same [`Document`] contract, so every consumer downstream —
//! layout, the JS DOM bridge, `/browse` — is unchanged and the two can be
//! swapped at runtime and compared on the same page.
//!
//! **What is actually being A/B'd.** Only the tree builder. The front end is
//! shared verbatim: `extract_assets_rich` (pulls `<style>`/`<script>`/
//! `<noscript>` out and cleans the source) and `preprocess` both run before
//! `tl` sees a byte, and attributes are applied through
//! `html::set_attribute`. That is deliberate — if each parser extracted its
//! own stylesheets and decided for itself what `width=` means, the comparison
//! would be measuring the divergence rather than the parser, and a difference
//! in rendered output could not be attributed to either.
//!
//! **Neither parser is spec-compliant, and they are non-compliant differently.**
//! `tl` is a flat-arena scanner with no insertion modes: like ours it nests
//! `<p>one<p>two` instead of making them siblings, so switching does not buy
//! implied end tags. What it does buy is speed and a genuinely independent
//! second implementation to cross-check against.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use super::html::{
    self, Document, Node, NodeKind, MAX_HTML_BYTES, MAX_NODES,
};

/// How deep the tree walk will descend.
///
/// **This is a stack bound, not a taste judgement.** `tl` nests an unclosed
/// tag rather than closing it, so `<div>` repeated 8,000 times produces a tree
/// 8,000 deep — and this adapter walks it *recursively*, where
/// `html::parse` walks with an explicit `Vec` stack and only pays heap. A
/// synthetic page of unclosed divs **double-faulted the kernel** before this
/// existed (`the_node_cap_holds_under_tl_too` reproduced it), which is a
/// hostile-page crash, not a rendering bug.
///
/// Dropping the tree is recursive too (a `Node` owns `Vec<Node>`), so the
/// depth has to be bounded at build time — refusing to descend further is the
/// only point at which the stack is still intact.
///
/// 256 is far past what real markup nests (tens), and matches the AST depth
/// guard the JS parser uses for the same reason.
const MAX_DEPTH: usize = 256;

/// Parse `html` into a [`Document`] using `tl` for the tree.
///
/// Mirrors [`html::parse`]'s contract exactly, including the byte and node
/// caps — a parser swap must not also change how much of a hostile page is
/// consumed, or a page that was refused under one engine renders under the
/// other and the switch has quietly become a policy change.
pub fn parse(source: &str) -> Document {
    let slice = if source.len() > MAX_HTML_BYTES {
        &source[..MAX_HTML_BYTES]
    } else {
        source
    };
    let (cleaned, stylesheets, script_tags, styles_ordered) = html::extract_assets_rich(slice);
    let scripts: Vec<String> = script_tags
        .iter()
        .filter(|t| t.src.is_none())
        .map(|t| t.body.clone())
        .collect();
    let prep = html::preprocess(&cleaned);

    let mut root = Node {
        kind: NodeKind::Document,
        children: Vec::new(),
        elem_idx: None,
    };
    let mut node_count = 1usize;
    let mut title = String::new();

    // A parse failure is not a panic and not an empty page: `tl` only fails
    // when the input exceeds `u32`, which `MAX_HTML_BYTES` already prevents.
    // Falling back to an empty document (rather than to *our* parser) keeps
    // the switch honest — a silent fallback would report `tl`'s timings for
    // our parser's work.
    if let Ok(dom) = tl::parse(&prep, tl::ParserOptions::default()) {
        let parser = dom.parser();
        for handle in dom.children() {
            build(handle, parser, &mut root, &mut node_count, &mut title, 0);
        }
    }

    // The same post-parse fixups the tokenizer path applies — they are
    // properties of a Document, not of a scanner. See `finalize_document`.
    html::finalize_document(&mut root, &mut title, &mut node_count);

    Document {
        title,
        root,
        node_count,
        stylesheets,
        scripts,
        script_tags,
        styles_ordered,
    }
}

/// Convert one `tl` node (and its subtree) into our tree, appending to `out`.
fn build(
    handle: &tl::NodeHandle,
    parser: &tl::Parser<'_>,
    out: &mut Node,
    node_count: &mut usize,
    title: &mut String,
    depth: usize,
) {
    if *node_count >= MAX_NODES || depth >= MAX_DEPTH {
        return;
    }
    let Some(node) = handle.get(parser) else {
        return;
    };

    match node {
        tl::Node::Raw(bytes) => {
            let text = bytes.as_utf8_str();
            if !text.trim().is_empty() {
                *node_count += 1;
                out.children.push(Node::text(html::decode_entities(&text)));
            }
        }
        // Comments and doctypes carry nothing layout or JS can use, and our
        // parser drops them too — keeping them would make the node counts
        // incomparable for no gain.
        tl::Node::Comment(_) => {}
        tl::Node::Tag(tag) => {
            let name = tag.name().as_utf8_str().to_ascii_lowercase();
            // Already lifted out by `extract_assets_rich`; if any survive the
            // clean they are dropped here exactly as `html::parse` drops them.
            if matches!(
                name.as_str(),
                "script" | "style" | "noscript" | "template"
            ) {
                return;
            }

            let mut el = Node::element(&name);
            *node_count += 1;

            for (key, value) in tag.attributes().iter() {
                let key = key.to_ascii_lowercase();
                // A valueless attribute (`disabled`, `hidden`) is the empty
                // string, matching what the tokenizer path produces for
                // `Token::Attribute { value: None }`.
                let val = value
                    .map(|v| html::decode_entities(&v))
                    .unwrap_or_default();
                html::set_attribute(&mut el, key, val);
            }

            for child in tag.children().top().iter() {
                build(child, parser, &mut el, node_count, title, depth + 1);
            }

            // `<title>` is the document title, not rendered content — our
            // parser captures it the same way, off the text it contains.
            if name == "title" && title.is_empty() {
                let mut buf = String::new();
                collect_text(&el, &mut buf);
                *title = buf.trim().to_string();
            }

            out.children.push(el);
        }
    }
}

/// Concatenate the text of a subtree (used only for `<title>`).
fn collect_text(n: &Node, out: &mut String) {
    match &n.kind {
        NodeKind::Text(t) => out.push_str(t),
        _ => {
            for c in &n.children {
                collect_text(c, out);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    /// Walk a tree into a comparable shape: `(depth, tag-or-#text)`.
    fn shape(n: &Node, depth: usize, out: &mut Vec<(usize, String)>) {
        match &n.kind {
            NodeKind::Element { tag, .. } => out.push((depth, tag.clone())),
            NodeKind::Text(t) => {
                let t = t.trim();
                if !t.is_empty() {
                    out.push((depth, alloc::format!("#text:{t}")));
                }
            }
            NodeKind::Document => {}
        }
        for c in &n.children {
            shape(c, depth + 1, out);
        }
    }

    fn shape_of(doc: &Document) -> Vec<(usize, String)> {
        let mut v = Vec::new();
        shape(&doc.root, 0, &mut v);
        v
    }

    /// The two engines must build the **same tree** for the markup our own
    /// pages contain.
    ///
    /// This is the assertion that makes `/html ours|tl` a switch rather than a
    /// second browser: if the trees differ, the faster engine is faster at a
    /// different job and every timing comparison is meaningless. It is also
    /// the cross-check that a single implementation cannot give itself — two
    /// independently written parsers agreeing is real evidence, where a parser
    /// agreeing with its own round trip is not.
    #[test_case]
    fn both_engines_build_the_same_tree_for_our_pages() {
        for src in [
            "<html><body><div id=\"a\" class=\"x y\"><p>hello</p></div></body></html>",
            "<div><span>one</span><span>two</span></div>",
            "<ul><li>a</li><li>b</li><li>c</li></ul>",
            "<a href=\"/x\" target=\"_blank\">link</a>",
            "<img src=\"a.png\" alt=\"pic\" width=\"10\" height=\"20\">",
            "<input type=\"text\" name=\"q\" placeholder=\"search\" disabled>",
            "<table><tr><td colspan=\"2\">x</td><td>y</td></tr></table>",
            "<div data-state=\"open\" aria-label=\"menu\">m</div>",
            "<br><hr><div>after voids</div>",
            "<p>a &amp; b &lt; c</p>",
            "<html><head><title>Hi</title></head><body>b</body></html>",
        ] {
            let ours = html::parse(src);
            let theirs = parse(src);
            assert_eq!(
                shape_of(&ours),
                shape_of(&theirs),
                "tree differs for {src:?}"
            );
            assert_eq!(ours.title, theirs.title, "title differs for {src:?}");
        }
    }

    /// Attributes land in the same typed fields under both engines.
    ///
    /// They go through one `html::set_attribute`, so this pins that the *tl*
    /// adapter feeds it the same lowercased key and entity-decoded value the
    /// tokenizer path does — the two places a second parser can quietly
    /// diverge without changing the tree's shape at all.
    #[test_case]
    fn attributes_are_applied_identically_by_both_engines() {
        let src = "<a href=\"/a?x=1&amp;y=2\" id=\"L\" class=\"btn\" \
                   data-state=\"OPEN\" rel=\"noopener\">go</a>";
        let ours = html::parse(src);
        let theirs = parse(src);

        fn first_link(n: &Node) -> Option<&Node> {
            if let NodeKind::Element { tag, .. } = &n.kind {
                if tag == "a" {
                    return Some(n);
                }
            }
            n.children.iter().find_map(first_link)
        }

        let a = first_link(&ours.root).expect("ours has an <a>");
        let b = first_link(&theirs.root).expect("tl has an <a>");
        let (NodeKind::Element { href: h1, id: i1, class: c1, rel: r1, extra_attrs: e1, .. },
             NodeKind::Element { href: h2, id: i2, class: c2, rel: r2, extra_attrs: e2, .. }) =
            (&a.kind, &b.kind)
        else {
            panic!("both must be elements")
        };
        // `&amp;` must already be decoded — a URL that keeps the entity is a
        // different request, and it would be invisible in a tree-shape diff.
        assert_eq!(h1.as_deref(), Some("/a?x=1&y=2"), "our href");
        assert_eq!(h1, h2, "href");
        assert_eq!(i1, i2, "id");
        assert_eq!(c1, c2, "class");
        assert_eq!(r1, r2, "rel");
        assert_eq!(e1, e2, "data-* bag");
    }

    /// The node cap holds, and deep nesting does not blow the kernel stack.
    ///
    /// A parser swap must not change how much of a hostile page is consumed —
    /// otherwise a document refused under one engine renders under the other
    /// and the switch has quietly become a policy change.
    ///
    /// The stack half is the one that bit: `tl` nests an unclosed tag, so this
    /// input is a tree thousands deep, and the adapter walks it recursively
    /// where `html::parse` uses an explicit `Vec`. Before `MAX_DEPTH` this
    /// test **double-faulted the kernel** rather than failing.
    #[test_case]
    fn the_node_cap_holds_under_tl_too() {
        let mut src = String::from("<html><body>");
        for i in 0..(MAX_NODES + 500) {
            src.push_str("<div>");
            let _ = i;
        }
        src.push_str("</body></html>");
        let theirs = parse(&src);
        assert!(
            theirs.node_count <= MAX_NODES + 1,
            "tl produced {} nodes, cap is {MAX_NODES}",
            theirs.node_count
        );
        // And the tree it handed back is walkable without recursing forever.
        fn depth_of(n: &Node) -> usize {
            1 + n.children.iter().map(depth_of).max().unwrap_or(0)
        }
        assert!(
            depth_of(&theirs.root) <= MAX_DEPTH + 2,
            "tree is {} deep, bound is {MAX_DEPTH}",
            depth_of(&theirs.root)
        );
    }

    /// Styles and scripts are lifted out before either engine runs.
    ///
    /// They share `extract_assets_rich`, so this is what pins that the shared
    /// front end really is shared — if `tl` ever saw a `<style>` body as text,
    /// the CSS would render as visible page content.
    #[test_case]
    fn tl_never_sees_style_or_script_bodies() {
        let src = "<html><head><style>.a{color:red}</style></head>\
                   <body><script>var x=1;</script><p>text</p></body></html>";
        let doc = parse(src);
        assert!(doc.stylesheets.contains("color:red"), "style extracted");
        assert_eq!(doc.scripts.len(), 1, "one inline script");
        let mut v = Vec::new();
        shape(&doc.root, 0, &mut v);
        for (_, label) in &v {
            assert!(!label.contains("color:red"), "CSS leaked into the tree");
            assert!(!label.contains("var x"), "JS leaked into the tree");
        }
    }
}
