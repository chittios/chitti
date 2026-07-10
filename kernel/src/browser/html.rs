//! HTML → DOM for the browser agent.
//!
//! Tokenization uses the **`htmlparser`** crate (`no_std`). We extract
//! `<style>` / `<script>` bodies (for the CSS/JS engines), strip them from the
//! tree, normalize void tags, then build our own DOM with id/class/style.

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use htmlparser::{ElementEnd, Token, Tokenizer};

// Void tags: prefer the MDN catalog (`elements::is_void`) so iframe/embed stay in sync.

/// Max HTML bytes accepted for parse (heap guard).
pub const MAX_HTML_BYTES: usize = 1 << 20; // 1 MiB
pub const MAX_NODES: usize = 8_000;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NodeKind {
    Document,
    Element {
        tag: String,
        href: Option<String>,
        alt: Option<String>,
        /// `src` for `<img>` (and similar).
        src: Option<String>,
        id: Option<String>,
        class: Option<String>,
        style_attr: Option<String>,
        /// Form-associated: `name=` on input/button/textarea/select.
        name: Option<String>,
        /// `value=` (input/button) or textarea default body (filled later).
        value: Option<String>,
        /// `type=` on `<input>` / `<button>` (text, password, submit, …).
        input_type: Option<String>,
        /// `action=` on `<form>`.
        action: Option<String>,
        /// `method=` on `<form>` (get/post).
        method: Option<String>,
        /// `placeholder=` on text inputs.
        placeholder: Option<String>,
        /// `srcdoc=` for `<iframe>` (inline document).
        srcdoc: Option<String>,
        /// `target=` for links/forms; `name=` of browsing context for iframe.
        target: Option<String>,
        /// `sandbox=` for iframe.
        sandbox: Option<String>,
        /// `width=` / `height=` presentational hints (px).
        width_attr: Option<i32>,
        height_attr: Option<i32>,
    },
    Text(String),
}

#[derive(Clone, Debug)]
pub struct Node {
    pub kind: NodeKind,
    pub children: Vec<Node>,
}

impl Node {
    pub fn element(tag: &str) -> Self {
        Node {
            kind: NodeKind::Element {
                tag: tag.to_ascii_lowercase(),
                href: None,
                alt: None,
                src: None,
                id: None,
                class: None,
                style_attr: None,
                name: None,
                value: None,
                input_type: None,
                action: None,
                method: None,
                placeholder: None,
                srcdoc: None,
                target: None,
                sandbox: None,
                width_attr: None,
                height_attr: None,
            },
            children: Vec::new(),
        }
    }

    pub fn text(s: impl Into<String>) -> Self {
        Node {
            kind: NodeKind::Text(s.into()),
            children: Vec::new(),
        }
    }

    pub fn tag_name(&self) -> Option<&str> {
        match &self.kind {
            NodeKind::Element { tag, .. } => Some(tag.as_str()),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Document {
    pub title: String,
    pub root: Node,
    pub node_count: usize,
    /// Concatenated CSS from `<style>` blocks.
    pub stylesheets: String,
    /// Each `<script>` body (no src= external yet).
    pub scripts: Vec<String>,
}

/// Extract `<style>` / `<script>` / `<noscript>` contents; return cleaned HTML.
pub fn extract_assets(html: &str) -> (String, String, Vec<String>) {
    let mut styles = String::new();
    let mut scripts = Vec::new();
    let mut s = html.to_string();
    s = take_blocks(&s, "style", &mut |body| {
        styles.push_str(body);
        styles.push('\n');
    });
    s = take_blocks(&s, "script", &mut |body| {
        scripts.push(body.to_string());
    });
    s = take_blocks(&s, "noscript", &mut |_| {});
    (s, styles, scripts)
}

fn take_blocks(html: &str, tag: &str, on_body: &mut dyn FnMut(&str)) -> String {
    let lower = html.to_ascii_lowercase();
    let open = format!("<{tag}");
    let close = format!("</{tag}>");
    let mut out = String::with_capacity(html.len());
    let mut rest = html;
    let mut rest_l = lower.as_str();
    loop {
        if let Some(i) = rest_l.find(&open) {
            out.push_str(&rest[..i]);
            let after = &rest[i..];
            let after_l = &rest_l[i..];
            // Find end of open tag `>`
            let gt = after.find('>').unwrap_or(0);
            let after_open = &after[gt + 1..];
            let after_open_l = &after_l[gt + 1..];
            if let Some(j) = after_open_l.find(&close) {
                on_body(&after_open[..j]);
                let next = gt + 1 + j + close.len();
                rest = &after[next..];
                rest_l = &after_l[next..];
            } else {
                break;
            }
        } else {
            out.push_str(rest);
            break;
        }
    }
    out
}

/// Parse HTML into a [`Document`] (styles/scripts extracted, not in the tree).
pub fn parse(html: &str) -> Document {
    let slice = if html.len() > MAX_HTML_BYTES {
        &html[..MAX_HTML_BYTES]
    } else {
        html
    };
    let (cleaned, stylesheets, scripts) = extract_assets(slice);
    let prep = preprocess(&cleaned);
    let mut root = Node {
        kind: NodeKind::Document,
        children: Vec::new(),
    };
    let mut stack: Vec<*mut Node> = alloc::vec![&mut root as *mut Node];
    let mut node_count = 1usize;
    let mut title = String::new();
    let mut in_title = false;
    let mut pending_tag: Option<String> = None;

    for item in Tokenizer::from(prep.as_str()) {
        let Ok(tok) = item else {
            continue;
        };
        if node_count >= MAX_NODES {
            break;
        }
        match tok {
            Token::ElementStart { local, .. } => {
                let tag = local.as_str().to_ascii_lowercase();
                // style/script already extracted; ignore if any remain.
                if matches!(tag.as_str(), "script" | "style" | "noscript" | "template") {
                    pending_tag = None;
                    continue;
                }
                push_element(&mut stack, Node::element(&tag), &mut node_count);
                if tag == "title" {
                    in_title = true;
                }
                pending_tag = Some(tag);
            }
            Token::Attribute { local, value, .. } => {
                let key = local.as_str().to_ascii_lowercase();
                let val = value.map(|v| decode_entities(v.as_str())).unwrap_or_default();
                if let Some(top) = stack.last().copied() {
                    // SAFETY: stack pointers into root for this function.
                    let n = unsafe { &mut *top };
                    if let NodeKind::Element {
                        href,
                        alt,
                        src,
                        id,
                        class,
                        style_attr,
                        name,
                        value,
                        input_type,
                        action,
                        method,
                        placeholder,
                        srcdoc,
                        target,
                        sandbox,
                        width_attr,
                        height_attr,
                        tag,
                        ..
                    } = &mut n.kind
                    {
                        match key.as_str() {
                            "href" if tag == "a" || tag == "area" || tag == "link" || tag == "base" => {
                                *href = Some(val)
                            }
                            "alt" if tag == "img" || tag == "area" => *alt = Some(val),
                            "src"
                                if matches!(
                                    tag.as_str(),
                                    "img"
                                        | "iframe"
                                        | "script"
                                        | "video"
                                        | "audio"
                                        | "source"
                                        | "embed"
                                        | "frame"
                                        | "track"
                                ) =>
                            {
                                if tag == "img" && alt.is_none() {
                                    *alt = Some(format!("[{val}]"));
                                }
                                *src = Some(val);
                            }
                            "id" => *id = Some(val),
                            "class" => *class = Some(val),
                            "style" => *style_attr = Some(val),
                            "name" => *name = Some(val),
                            "value" => *value = Some(val),
                            "type" => *input_type = Some(val.to_ascii_lowercase()),
                            "action" => *action = Some(val),
                            "method" => *method = Some(val.to_ascii_lowercase()),
                            "placeholder" => *placeholder = Some(val),
                            "srcdoc" => *srcdoc = Some(val),
                            "target" => *target = Some(val),
                            "sandbox" => *sandbox = Some(val),
                            "width" => *width_attr = val.parse().ok(),
                            "height" => *height_attr = val.parse().ok(),
                            _ => {}
                        }
                    }
                }
            }
            Token::ElementEnd { end, .. } => match end {
                ElementEnd::Open => {
                    if let Some(tag) = pending_tag.take() {
                        if is_void(&tag) {
                            pop_if_tag(&mut stack, &tag);
                            if tag == "title" {
                                in_title = false;
                            }
                        }
                    }
                }
                ElementEnd::Empty => {
                    if let Some(tag) = pending_tag.take() {
                        pop_if_tag(&mut stack, &tag);
                        if tag == "title" {
                            in_title = false;
                        }
                    }
                }
                ElementEnd::Close(_prefix, local) => {
                    let tag = local.as_str().to_ascii_lowercase();
                    pending_tag = None;
                    if tag == "title" {
                        in_title = false;
                    }
                    pop_if_tag(&mut stack, &tag);
                }
            },
            Token::Text { text } | Token::Cdata { text, .. } => {
                let t = decode_entities(text.as_str());
                let trimmed = if in_title {
                    t.trim().to_string()
                } else {
                    collapse_ws(&t)
                };
                if trimmed.is_empty() {
                    continue;
                }
                if in_title {
                    if !title.is_empty() {
                        title.push(' ');
                    }
                    title.push_str(&trimmed);
                }
                push_text(&mut stack, Node::text(trimmed), &mut node_count);
            }
            Token::Comment { .. }
            | Token::Declaration { .. }
            | Token::EmptyDtd { .. }
            | Token::DtdStart { .. }
            | Token::DtdEnd { .. }
            | Token::EntityDeclaration { .. }
            | Token::ProcessingInstruction { .. }
            | Token::ConditionalCommentStart { .. }
            | Token::ConditionalCommentEnd { .. } => {
                pending_tag = None;
            }
        }
    }

    if title.is_empty() {
        title = first_heading_text(&root).unwrap_or_else(|| String::from("Untitled"));
    }
    if root.children.is_empty() {
        root.children.push(Node::element("body"));
        node_count += 1;
    }

    Document {
        title,
        root,
        node_count,
        stylesheets,
        scripts,
    }
}

fn push_element(stack: &mut Vec<*mut Node>, child: Node, count: &mut usize) {
    if stack.is_empty() {
        return;
    }
    let parent = stack[stack.len() - 1];
    // SAFETY: parent is live under root for parse.
    let parent = unsafe { &mut *parent };
    parent.children.push(child);
    *count += 1;
    let last = parent.children.last_mut().unwrap() as *mut Node;
    stack.push(last);
}

fn push_text(stack: &mut Vec<*mut Node>, child: Node, count: &mut usize) {
    if stack.is_empty() {
        return;
    }
    let parent = stack[stack.len() - 1];
    // SAFETY: same as push_element.
    let parent = unsafe { &mut *parent };
    parent.children.push(child);
    *count += 1;
}

fn pop_if_tag(stack: &mut Vec<*mut Node>, tag: &str) {
    while stack.len() > 1 {
        let top = stack[stack.len() - 1];
        // SAFETY: stack entries are live tree nodes.
        let name = unsafe {
            match &(*top).kind {
                NodeKind::Element { tag: t, .. } => t.as_str(),
                _ => "",
            }
        };
        stack.pop();
        if name.eq_ignore_ascii_case(tag) {
            break;
        }
    }
}

fn is_void(tag: &str) -> bool {
    super::elements::is_void(tag)
}

fn preprocess(html: &str) -> String {
    let mut out = String::with_capacity(html.len() + 64);
    let bytes = html.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b'<' {
            out.push(bytes[i] as char);
            i += 1;
            continue;
        }
        let start = i;
        i += 1;
        if i < bytes.len() && (bytes[i] == b'!' || bytes[i] == b'?') {
            while i < bytes.len() && bytes[i] != b'>' {
                i += 1;
            }
            if i < bytes.len() {
                i += 1;
            }
            if let Ok(chunk) = core::str::from_utf8(&bytes[start..i]) {
                out.push_str(chunk);
            }
            continue;
        }
        let is_close = i < bytes.len() && bytes[i] == b'/';
        if is_close {
            i += 1;
        }
        let name_start = i;
        while i < bytes.len() && bytes[i].is_ascii_alphanumeric() {
            i += 1;
        }
        let name = core::str::from_utf8(&bytes[name_start..i])
            .unwrap_or("")
            .to_ascii_lowercase();
        let mut quote = 0u8;
        while i < bytes.len() {
            let c = bytes[i];
            if quote != 0 {
                if c == quote {
                    quote = 0;
                }
                i += 1;
                continue;
            }
            if c == b'"' || c == b'\'' {
                quote = c;
                i += 1;
                continue;
            }
            if c == b'>' {
                break;
            }
            i += 1;
        }
        let end = i;
        let self_close = end > start + 1 && bytes[end.saturating_sub(1)] == b'/';
        if end < bytes.len() {
            if !is_close && is_void(&name) && !self_close {
                out.push_str(core::str::from_utf8(&bytes[start..end]).unwrap_or(""));
                if !out.ends_with('/') {
                    out.push('/');
                }
                out.push('>');
            } else {
                out.push_str(core::str::from_utf8(&bytes[start..=end]).unwrap_or(""));
            }
            i = end + 1;
        } else {
            out.push_str(core::str::from_utf8(&bytes[start..]).unwrap_or(""));
            break;
        }
    }
    if !out.to_ascii_lowercase().contains("<html") {
        format!("<html><body>{out}</body></html>")
    } else {
        out
    }
}

fn collapse_ws(s: &str) -> String {
    let mut out = String::new();
    let mut sp = false;
    for c in s.chars() {
        if c.is_whitespace() {
            if !sp && !out.is_empty() {
                out.push(' ');
                sp = true;
            }
        } else {
            out.push(c);
            sp = false;
        }
    }
    out.trim().to_string()
}

pub fn decode_entities(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '&' {
            out.push(c);
            continue;
        }
        let mut ent = String::new();
        while let Some(&n) = chars.peek() {
            if n == ';' || ent.len() > 12 {
                break;
            }
            ent.push(n);
            chars.next();
        }
        if chars.peek() == Some(&';') {
            chars.next();
        }
        match ent.as_str() {
            "amp" => out.push('&'),
            "lt" => out.push('<'),
            "gt" => out.push('>'),
            "quot" => out.push('"'),
            "apos" => out.push('\''),
            "nbsp" => out.push(' '),
            _ if ent.starts_with('#') => {
                let hex = ent.as_bytes().get(1) == Some(&b'x') || ent.as_bytes().get(1) == Some(&b'X');
                let num = ent[1..].trim_start_matches(['x', 'X']);
                let code = if hex {
                    u32::from_str_radix(num, 16).ok()
                } else {
                    num.parse().ok()
                };
                if let Some(ch) = code.and_then(char::from_u32) {
                    out.push(ch);
                } else {
                    out.push('&');
                    out.push_str(&ent);
                }
            }
            _ => {
                out.push('&');
                out.push_str(&ent);
            }
        }
    }
    out
}

fn first_heading_text(n: &Node) -> Option<String> {
    if let NodeKind::Element { tag, .. } = &n.kind {
        if matches!(tag.as_str(), "h1" | "h2" | "h3" | "h4" | "h5" | "h6") {
            let t = collect_text(n);
            if !t.is_empty() {
                return Some(t);
            }
        }
    }
    for c in &n.children {
        if let Some(t) = first_heading_text(c) {
            return Some(t);
        }
    }
    None
}

pub fn collect_text(n: &Node) -> String {
    let mut out = String::new();
    collect_text_into(n, &mut out);
    collapse_ws(&out)
}

fn collect_text_into(n: &Node, out: &mut String) {
    match &n.kind {
        NodeKind::Text(t) => {
            if !out.is_empty() && !out.ends_with(' ') {
                out.push(' ');
            }
            out.push_str(t);
        }
        NodeKind::Element { tag, alt, .. } => {
            if matches!(tag.as_str(), "script" | "style" | "noscript") {
                return;
            }
            if tag == "br" {
                out.push(' ');
                return;
            }
            if tag == "img" {
                if let Some(a) = alt {
                    if !out.is_empty() {
                        out.push(' ');
                    }
                    out.push_str(a);
                }
                return;
            }
            for c in &n.children {
                collect_text_into(c, out);
            }
            if matches!(
                tag.as_str(),
                "p" | "div" | "li" | "h1" | "h2" | "h3" | "h4" | "h5" | "h6" | "blockquote"
            ) {
                out.push(' ');
            }
        }
        NodeKind::Document => {
            for c in &n.children {
                collect_text_into(c, out);
            }
        }
    }
}

pub fn collect_links(n: &Node, out: &mut Vec<(String, String)>) {
    if let NodeKind::Element { tag, href, .. } = &n.kind {
        if tag == "a" {
            if let Some(h) = href {
                let text = collect_text(n);
                out.push((
                    h.clone(),
                    if text.is_empty() {
                        h.clone()
                    } else {
                        text
                    },
                ));
            }
        }
    }
    for c in &n.children {
        collect_links(c, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn parses_title_and_paragraph() {
        let doc = parse(
            r#"<!DOCTYPE html><html><head><title>Hello &amp; Hi</title>
            <style>p{color:red}</style></head>
            <body><p>World</p><script>evil()</script><p>Next</p></body></html>"#,
        );
        assert_eq!(doc.title, "Hello & Hi");
        assert!(doc.stylesheets.contains("color"), "{}", doc.stylesheets);
        assert_eq!(doc.scripts.len(), 1);
        let text = collect_text(&doc.root);
        assert!(text.contains("World"), "got {text}");
        assert!(!text.contains("evil"), "script stripped: {text}");
    }

    #[test_case]
    fn parses_id_class_style() {
        let doc = parse(r#"<html><body><p id="a" class="x y" style="color:blue">Hi</p></body></html>"#);
        fn find_p(n: &Node) -> Option<&Node> {
            if n.tag_name() == Some("p") {
                return Some(n);
            }
            for c in &n.children {
                if let Some(p) = find_p(c) {
                    return Some(p);
                }
            }
            None
        }
        let p = find_p(&doc.root).expect("p");
        match &p.kind {
            NodeKind::Element {
                id, class, style_attr, ..
            } => {
                assert_eq!(id.as_deref(), Some("a"));
                assert_eq!(class.as_deref(), Some("x y"));
                assert_eq!(style_attr.as_deref(), Some("color:blue"));
            }
            _ => panic!(),
        }
    }

    #[test_case]
    fn decode_entities_basic() {
        assert_eq!(decode_entities("a&amp;b&lt;c&gt;"), "a&b<c>");
        assert_eq!(decode_entities("&#65;"), "A");
    }
}
