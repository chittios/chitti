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
        /// `colspan=` on `<td>` / `<th>` (default 1). Used by table layout so
        /// Hacker News subtext rows (`<td colspan=2></td><td class=subtext>`)
        /// align under the title column instead of under the rank.
        colspan_attr: Option<u32>,
        /// `rowspan=` on `<td>` / `<th>` (default 1).
        rowspan_attr: Option<u32>,
        /// HTML presentational `bgcolor=` (legacy; still used by HN, Google…).
        /// Stored raw (`#f6f6ef` / `orange`); layout parses via `css::parse_color`.
        bgcolor_attr: Option<String>,
        /// Presentational `width=` as a percentage (e.g. `width="85%"` on HN's
        /// `#hnmain` table). Pixel widths go through `width_attr`.
        width_pct: Option<u8>,
        /// Presentational `align=left|center|right` (HN rank column, etc.).
        align_attr: Option<String>,
        /// `rel=` on `<link>` / `<a>` (stylesheet, icon, …).
        rel: Option<String>,
        /// `srcset=` on `<img>` / `<source>` (responsive candidates).
        srcset: Option<String>,
        /// Event-handler attributes: any `on*` name (lowercased) with its
        /// value, e.g. `("onclick", "doThing()")`.
        on_attrs: Vec<(String, String)>,
        /// Catch-all for attributes not mapped to typed fields (`data-*`,
        /// ARIA, custom). Preserved so `getAttribute` / JS can read them.
        extra_attrs: Vec<(String, String)>,
    },
    Text(String),
}

#[derive(Clone, Debug)]
pub struct Node {
    pub kind: NodeKind,
    pub children: Vec<Node>,
    /// Slot for a stable element index assigned by later passes (JS DOM
    /// binding); the parser always leaves it `None`.
    pub elem_idx: Option<usize>,
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
                colspan_attr: None,
                rowspan_attr: None,
                bgcolor_attr: None,
                width_pct: None,
                align_attr: None,
                rel: None,
                srcset: None,
                on_attrs: Vec::new(),
                extra_attrs: Vec::new(),
            },
            children: Vec::new(),
            elem_idx: None,
        }
    }

    pub fn text(s: impl Into<String>) -> Self {
        Node {
            kind: NodeKind::Text(s.into()),
            children: Vec::new(),
            elem_idx: None,
        }
    }

    pub fn tag_name(&self) -> Option<&str> {
        match &self.kind {
            NodeKind::Element { tag, .. } => Some(tag.as_str()),
            _ => None,
        }
    }
}

/// A `<script>` element captured with its open-tag attributes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScriptTag {
    /// `src=` (external script) — `None` for inline bodies.
    pub src: Option<String>,
    /// Inline body text (empty for most `src=` scripts).
    pub body: String,
    /// Bare `async` attribute present.
    pub async_: bool,
    /// Bare `defer` attribute present.
    pub defer: bool,
    /// `type="module"`.
    pub module: bool,
}

/// One stylesheet source in document order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StyleSrc {
    /// A `<style>` block body.
    Inline(String),
    /// A `<link rel="stylesheet" href=…>` URL (as written, unresolved).
    External(String),
}

#[derive(Clone, Debug)]
pub struct Document {
    pub title: String,
    pub root: Node,
    pub node_count: usize,
    /// Concatenated CSS from `<style>` blocks.
    pub stylesheets: String,
    /// Each inline `<script>` body (no src=) — legacy shape.
    pub scripts: Vec<String>,
    /// Every `<script>` tag (inline + external) with attributes, in order.
    pub script_tags: Vec<ScriptTag>,
    /// Stylesheet sources — `<style>` and `<link rel=stylesheet>` — in
    /// exact document order (by byte offset in the source HTML).
    pub styles_ordered: Vec<StyleSrc>,
}

/// Extract `<style>` / `<script>` / `<noscript>` contents; return cleaned HTML.
/// Legacy shape (concatenated styles, all script bodies); rich callers use
/// [`extract_assets_rich`].
pub fn extract_assets(html: &str) -> (String, String, Vec<String>) {
    let (cleaned, styles, tags, _ordered) = extract_assets_rich(html);
    let scripts = tags
        .into_iter()
        .filter(|t| t.src.is_none())
        .map(|t| t.body)
        .collect();
    (cleaned, styles, scripts)
}

/// Rich extraction: cleaned HTML, concatenated inline CSS, all `<script>`
/// tags with parsed attributes, and every stylesheet source (inline
/// `<style>` + `<link rel=stylesheet>`) in exact document order.
pub fn extract_assets_rich(html: &str) -> (String, String, Vec<ScriptTag>, Vec<StyleSrc>) {
    let mut styles = String::new();
    let mut ordered: Vec<(usize, StyleSrc)> = Vec::new();
    let mut tags: Vec<ScriptTag> = Vec::new();
    // `<noscript>` goes FIRST, before any asset is harvested.
    //
    // Scripting is enabled here, so per the HTML parsing spec a `<noscript>`
    // element's content is **raw text** — the markup inside it does not exist,
    // and neither do its `<style>`, `<script>` or `<link>`. Harvesting styles
    // from the original source picked them up anyway, and a site that uses the
    // standard "blank the page for non-JS visitors" trick then blanked it for
    // us: google.com/search ships
    // `<noscript><style>table,div,span,p{display:none}</style>…</noscript>`,
    // and that one rule hid every element on the page. The result looked like a
    // broken renderer — a page that loads, reports its title, and draws almost
    // nothing.
    //
    // Every later pass runs on this same stripped text, so the recorded style
    // offsets and the `<link>` scan still line up with each other.
    let doc = take_blocks(html, "noscript", &mut |_| {});
    let s = take_blocks_attrs(&doc, "style", &mut |off, _attrs, body| {
        styles.push_str(body);
        styles.push('\n');
        ordered.push((off, StyleSrc::Inline(body.to_string())));
    });
    let cleaned = take_blocks_attrs(&s, "script", &mut |_off, attrs, body| {
        tags.push(parse_script_tag(attrs, body));
    });
    for (off, href) in scan_link_stylesheets(&doc) {
        ordered.push((off, StyleSrc::External(href)));
    }
    ordered.sort_by_key(|(off, _)| *off); // stable: offsets are unique anyway
    let styles_ordered = ordered.into_iter().map(|(_, s)| s).collect();
    (cleaned, styles, tags, styles_ordered)
}

fn take_blocks(html: &str, tag: &str, on_body: &mut dyn FnMut(&str)) -> String {
    take_blocks_attrs(html, tag, &mut |_off, _attrs, body| on_body(body))
}

/// Like [`take_blocks`], but hands the callback the byte offset of the open
/// tag in `html` and the raw open-tag attribute text alongside the body.
fn take_blocks_attrs(
    html: &str,
    tag: &str,
    on_block: &mut dyn FnMut(usize, &str, &str),
) -> String {
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
            let attrs = if gt > open.len() { &after[open.len()..gt] } else { "" };
            let after_open = &after[gt + 1..];
            let after_open_l = &after_l[gt + 1..];
            if let Some(j) = after_open_l.find(&close) {
                let off = html.len() - rest.len() + i;
                on_block(off, attrs, &after_open[..j]);
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

/// Parse open-tag attribute text into (lowercased name, value) pairs.
/// Bare attributes (`async`) get an empty value; values may be `"…"`,
/// `'…'`, or bare-until-whitespace.
fn parse_tag_attrs(s: &str) -> Vec<(String, String)> {
    let bytes = s.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        while i < bytes.len() && (bytes[i].is_ascii_whitespace() || bytes[i] == b'/') {
            i += 1;
        }
        let ns = i;
        while i < bytes.len()
            && !bytes[i].is_ascii_whitespace()
            && bytes[i] != b'='
            && bytes[i] != b'/'
        {
            i += 1;
        }
        if i == ns {
            // Stray `=` or end — skip a byte to guarantee progress.
            i += 1;
            continue;
        }
        let name = s[ns..i].to_ascii_lowercase();
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        let mut value = String::new();
        if i < bytes.len() && bytes[i] == b'=' {
            i += 1;
            while i < bytes.len() && bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            if i < bytes.len() && (bytes[i] == b'"' || bytes[i] == b'\'') {
                let q = bytes[i];
                i += 1;
                let vs = i;
                while i < bytes.len() && bytes[i] != q {
                    i += 1;
                }
                value = s[vs..i].to_string();
                if i < bytes.len() {
                    i += 1;
                }
            } else {
                let vs = i;
                while i < bytes.len() && !bytes[i].is_ascii_whitespace() {
                    i += 1;
                }
                value = s[vs..i].to_string();
            }
        }
        out.push((name, value));
    }
    out
}

/// Build a [`ScriptTag`] from open-tag attribute text + body.
fn parse_script_tag(attrs: &str, body: &str) -> ScriptTag {
    let mut src = None;
    let mut async_ = false;
    let mut defer = false;
    let mut module = false;
    for (k, v) in parse_tag_attrs(attrs) {
        match k.as_str() {
            "src" if !v.is_empty() => src = Some(v),
            "async" => async_ = true,
            "defer" => defer = true,
            "type" => module = v.trim().eq_ignore_ascii_case("module"),
            _ => {}
        }
    }
    ScriptTag {
        src,
        body: body.to_string(),
        async_,
        defer,
        module,
    }
}

/// Scan `html` for `<link rel~="stylesheet" href=…>` tags; return
/// `(byte offset, href)` per hit. The `rel` match is a case-insensitive
/// whitespace-separated word match.
/// The document's `<meta http-equiv="refresh" content="N; url=…">`, as
/// `(delay_seconds, url_as_written)`.
///
/// This is the only redirect a page can express in **markup**, and plenty of
/// the web still depends on it — google.com/search hands a non-JS browser a
/// page whose entire body is hidden (`display:none` plus a `<noscript>` block)
/// and a meta refresh to the real destination. Without it the browser sits on a
/// blank page that says "please click here if you are not redirected", which
/// reads as a broken renderer rather than an unimplemented redirect.
///
/// A refresh with **no url** is a self-reload timer, not a navigation, and is
/// deliberately not reported: honouring it would put the page in a loop.
pub fn meta_refresh(html: &str) -> Option<(u32, String)> {
    let lower = html.to_ascii_lowercase();
    let mut pos = 0;
    while let Some(i) = lower[pos..].find("<meta") {
        let off = pos + i;
        let after = &html[off + 5..];
        let boundary_ok = after
            .chars()
            .next()
            .map(|c| c.is_ascii_whitespace() || c == '>' || c == '/')
            .unwrap_or(false);
        let gt = after.find('>').map(|g| off + 5 + g).unwrap_or(html.len());
        if boundary_ok {
            let pairs = parse_tag_attrs(&html[off + 5..gt]);
            let is_refresh = pairs
                .iter()
                .any(|(k, v)| k == "http-equiv" && v.trim().eq_ignore_ascii_case("refresh"));
            if is_refresh {
                if let Some((_, content)) = pairs.iter().find(|(k, _)| k == "content") {
                    if let Some(hit) = parse_refresh_content(content) {
                        return Some(hit);
                    }
                }
            }
        }
        pos = if gt > off + 5 { gt } else { off + 5 };
    }
    None
}

/// `"0; url=/next"` → `(0, "/next")`. The delay may be absent or fractional,
/// the separator may be `;` or whitespace, and `url=` is case-insensitive and
/// optionally quoted — all forms seen in the wild.
fn parse_refresh_content(content: &str) -> Option<(u32, String)> {
    let (head, rest) = match content.split_once(&[';', ','][..]) {
        Some((a, b)) => (a, b),
        // `content="5"` is a self-reload; `content="url=/x"` has no delay.
        None => (content, ""),
    };
    let delay = head
        .trim()
        .split('.')
        .next()
        .unwrap_or("")
        .trim()
        .parse::<u32>()
        .unwrap_or(0);
    let rest = rest.trim();
    let url = match rest.get(..4) {
        Some(p) if p.eq_ignore_ascii_case("url=") => &rest[4..],
        // A bare URL after the delay (`content="0; /next"`) is not standard but
        // is accepted by browsers.
        _ if !rest.is_empty() => rest,
        _ => return None,
    };
    let url = url.trim().trim_matches(['"', '\''].as_ref()).trim();
    if url.is_empty() {
        return None;
    }
    Some((delay, url.to_string()))
}

fn scan_link_stylesheets(html: &str) -> Vec<(usize, String)> {
    let lower = html.to_ascii_lowercase();
    let mut out = Vec::new();
    let mut pos = 0;
    while let Some(i) = lower[pos..].find("<link") {
        let off = pos + i;
        let after = &html[off + 5..];
        // Tag-name boundary: `<linkage>` must not match.
        let boundary_ok = after
            .chars()
            .next()
            .map(|c| c.is_ascii_whitespace() || c == '>' || c == '/')
            .unwrap_or(false);
        let gt = after
            .find('>')
            .map(|g| off + 5 + g)
            .unwrap_or(html.len());
        if boundary_ok {
            let pairs = parse_tag_attrs(&html[off + 5..gt]);
            let rel_ok = pairs.iter().any(|(k, v)| {
                k == "rel"
                    && v.split_ascii_whitespace()
                        .any(|w| w.eq_ignore_ascii_case("stylesheet"))
            });
            if rel_ok {
                if let Some((_, href)) = pairs.iter().find(|(k, v)| k == "href" && !v.is_empty()) {
                    out.push((off, href.clone()));
                }
            }
        }
        pos = if gt > off + 5 { gt } else { off + 5 };
    }
    out
}

/// Pick a candidate URL from an `srcset` attribute value for viewport width
/// `vw` (px). Width descriptors (`480w`): smallest width ≥ `vw`, else the
/// largest available. Density descriptors (`2x`, none = `1x`): prefer `1x`,
/// else the first candidate.
pub fn pick_srcset_candidate(srcset: &str, vw: i32) -> Option<String> {
    struct Cand {
        url: String,
        w: Option<i32>,
        x: f32,
    }
    let mut cands: Vec<Cand> = Vec::new();
    for part in srcset.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let mut it = part.split_ascii_whitespace();
        let Some(url) = it.next() else { continue };
        let desc = it.next().unwrap_or("");
        let mut w = None;
        let mut x = 1.0f32; // no descriptor = 1x per the HTML spec
        if let Some(num) = desc.strip_suffix(['w', 'W']) {
            w = num.parse::<i32>().ok();
        } else if let Some(num) = desc.strip_suffix(['x', 'X']) {
            x = num.parse::<f32>().unwrap_or(1.0);
        }
        cands.push(Cand {
            url: url.to_string(),
            w,
            x,
        });
    }
    if cands.is_empty() {
        return None;
    }
    if cands.iter().any(|c| c.w.is_some()) {
        // Smallest width descriptor that still covers the viewport.
        let mut best: Option<&Cand> = None;
        for c in cands.iter().filter(|c| c.w.is_some()) {
            let cw = c.w.unwrap();
            match best {
                Some(b) if b.w.unwrap() >= vw => {
                    if cw >= vw && cw < b.w.unwrap() {
                        best = Some(c);
                    }
                }
                Some(b) => {
                    if cw >= vw || cw > b.w.unwrap() {
                        best = Some(c);
                    }
                }
                None => best = Some(c),
            }
        }
        return best.map(|c| c.url.clone());
    }
    // Density-only list: prefer an exact 1x, else the first candidate.
    if let Some(c) = cands.iter().find(|c| c.x == 1.0) {
        return Some(c.url.clone());
    }
    cands.first().map(|c| c.url.clone())
}

/// Apply one parsed attribute to an element node.
///
/// Factored out of the tokenizer loop so the **alternative** `tl` tree builder
/// (`browser::html_tl`) sets attributes through the exact same code. Two
/// parsers that each decide for themselves what `width=` or `on*` means would
/// diverge silently — and the A/B between them would then be measuring the
/// divergence rather than the parser.
///
/// `key` must already be lowercased and `val` already entity-decoded.
pub(crate) fn set_attribute(node: &mut Node, key: String, val: String) {
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
        colspan_attr,
        rowspan_attr,
        bgcolor_attr,
        width_pct,
        align_attr,
        rel,
        srcset,
        on_attrs,
        extra_attrs,
        tag,
    } = &mut node.kind
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
            "width" => {
                // `width="85%"` → percentage; bare number → px.
                let t = val.trim();
                if let Some(p) = t.strip_suffix('%') {
                    *width_pct = p.parse().ok().filter(|&n: &u8| n > 0 && n <= 100);
                    *width_attr = None;
                } else {
                    *width_attr = t.parse().ok();
                    *width_pct = None;
                }
            }
            "height" => *height_attr = val.parse().ok(),
            "colspan" => {
                *colspan_attr = val.parse().ok().filter(|&n| n >= 1);
            }
            "rowspan" => {
                *rowspan_attr = val.parse().ok().filter(|&n| n >= 1);
            }
            "bgcolor" => *bgcolor_attr = Some(val),
            "align" => *align_attr = Some(val.to_ascii_lowercase()),
            "rel" => *rel = Some(val),
            "srcset" => *srcset = Some(val),
            k if k.starts_with("on") => {
                on_attrs.push((k.to_string(), val))
            }
            _ => {
                extra_attrs.push((key, val));
            }
        }
    }
}

/// Parse HTML into a [`Document`] (styles/scripts extracted, not in the tree).
pub fn parse(html: &str) -> Document {
    let slice = if html.len() > MAX_HTML_BYTES {
        &html[..MAX_HTML_BYTES]
    } else {
        html
    };
    let (cleaned, stylesheets, script_tags, styles_ordered) = extract_assets_rich(slice);
    let scripts: Vec<String> = script_tags
        .iter()
        .filter(|t| t.src.is_none())
        .map(|t| t.body.clone())
        .collect();
    let prep = preprocess(&cleaned);
    let mut root = Node {
        kind: NodeKind::Document,
        children: Vec::new(),
        elem_idx: None,
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
                    set_attribute(n, key, val);
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

    finalize_document(&mut root, &mut title, &mut node_count);

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

/// The post-parse fixups that belong to a [`Document`], not to a tokenizer.
///
/// Shared with the alternative `tl` tree builder (`browser::html_tl`) because
/// they are decisions about what a *document* is, not about how markup is
/// scanned: a page with no `<title>` is named after its first heading (and
/// "Untitled" failing that), and a page whose body produced nothing still gets
/// a `<body>` so layout and the JS DOM have a mount point.
///
/// Leaving these in one parser is exactly the kind of divergence that makes an
/// engine A/B meaningless — it was caught here by
/// `both_engines_build_the_same_tree_for_our_pages`, which compared a title of
/// "Untitled" against "".
pub(crate) fn finalize_document(root: &mut Node, title: &mut String, node_count: &mut usize) {
    if title.is_empty() {
        *title = first_heading_text(root).unwrap_or_else(|| String::from("Untitled"));
    }
    if root.children.is_empty() {
        root.children.push(Node::element("body"));
        *node_count += 1;
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

pub(crate) fn preprocess(html: &str) -> String {
    let mut out = String::with_capacity(html.len() + 64);
    let bytes = html.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b'<' {
            // Copy the whole text run as UTF-8 — pushing bytes individually as
            // `char` is Latin-1 decoding and mangles multi-byte sequences (e.g.
            // the Indic language names in google.com's "Google offered in:").
            let start = i;
            while i < bytes.len() && bytes[i] != b'<' {
                i += 1;
            }
            match core::str::from_utf8(&bytes[start..i]) {
                Ok(s) => out.push_str(s),
                Err(_) => out.push_str(&alloc::string::String::from_utf8_lossy(&bytes[start..i])),
            }
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

/// Map a named HTML entity (without `&`/`;`) to its character. Covers the
/// common symbol/punctuation set and the Latin-1 supplement letters — the
/// entities real pages actually emit. Not the full HTML5 named-character table
/// (~2200 names); an unlisted name is decoded verbatim by the caller.
fn named_entity(name: &str) -> Option<char> {
    Some(match name {
        // Symbols & punctuation
        "copy" => '\u{00A9}', "reg" => '\u{00AE}', "trade" => '\u{2122}',
        "hellip" => '\u{2026}', "mdash" => '\u{2014}', "ndash" => '\u{2013}',
        "lsquo" => '\u{2018}', "rsquo" => '\u{2019}', "ldquo" => '\u{201C}',
        "rdquo" => '\u{201D}', "sbquo" => '\u{201A}', "bdquo" => '\u{201E}',
        "laquo" => '\u{00AB}', "raquo" => '\u{00BB}', "lsaquo" => '\u{2039}',
        "rsaquo" => '\u{203A}', "bull" => '\u{2022}', "middot" => '\u{00B7}',
        "dagger" => '\u{2020}', "Dagger" => '\u{2021}', "prime" => '\u{2032}',
        "Prime" => '\u{2033}', "permil" => '\u{2030}', "para" => '\u{00B6}',
        "sect" => '\u{00A7}', "deg" => '\u{00B0}', "plusmn" => '\u{00B1}',
        "times" => '\u{00D7}', "divide" => '\u{00F7}', "minus" => '\u{2212}',
        "frac12" => '\u{00BD}', "frac14" => '\u{00BC}', "frac34" => '\u{00BE}',
        "sup1" => '\u{00B9}', "sup2" => '\u{00B2}', "sup3" => '\u{00B3}',
        "micro" => '\u{00B5}', "cent" => '\u{00A2}', "pound" => '\u{00A3}',
        "yen" => '\u{00A5}', "euro" => '\u{20AC}', "curren" => '\u{00A4}',
        "iexcl" => '\u{00A1}', "iquest" => '\u{00BF}', "brvbar" => '\u{00A6}',
        "not" => '\u{00AC}', "shy" => '\u{00AD}', "macr" => '\u{00AF}',
        "acute" => '\u{00B4}', "cedil" => '\u{00B8}', "uml" => '\u{00A8}',
        "ordf" => '\u{00AA}', "ordm" => '\u{00BA}', "szlig" => '\u{00DF}',
        // Arrows & math
        "larr" => '\u{2190}', "uarr" => '\u{2191}', "rarr" => '\u{2192}',
        "darr" => '\u{2193}', "harr" => '\u{2194}', "infin" => '\u{221E}',
        "ne" => '\u{2260}', "le" => '\u{2264}', "ge" => '\u{2265}',
        "asymp" => '\u{2248}', "sum" => '\u{2211}', "radic" => '\u{221A}',
        "hearts" => '\u{2665}', "diams" => '\u{2666}', "clubs" => '\u{2663}',
        "spades" => '\u{2660}', "star" => '\u{2606}', "check" => '\u{2713}',
        "cross" => '\u{2717}', "ensp" => '\u{2002}', "emsp" => '\u{2003}',
        "thinsp" => '\u{2009}', "zwnj" => '\u{200C}', "zwj" => '\u{200D}',
        // Latin-1 accented letters (upper)
        "Agrave" => '\u{00C0}', "Aacute" => '\u{00C1}', "Acirc" => '\u{00C2}',
        "Atilde" => '\u{00C3}', "Auml" => '\u{00C4}', "Aring" => '\u{00C5}',
        "AElig" => '\u{00C6}', "Ccedil" => '\u{00C7}', "Egrave" => '\u{00C8}',
        "Eacute" => '\u{00C9}', "Ecirc" => '\u{00CA}', "Euml" => '\u{00CB}',
        "Igrave" => '\u{00CC}', "Iacute" => '\u{00CD}', "Icirc" => '\u{00CE}',
        "Iuml" => '\u{00CF}', "Ntilde" => '\u{00D1}', "Ograve" => '\u{00D2}',
        "Oacute" => '\u{00D3}', "Ocirc" => '\u{00D4}', "Otilde" => '\u{00D5}',
        "Ouml" => '\u{00D6}', "Oslash" => '\u{00D8}', "Ugrave" => '\u{00D9}',
        "Uacute" => '\u{00DA}', "Ucirc" => '\u{00DB}', "Uuml" => '\u{00DC}',
        "Yacute" => '\u{00DD}',
        // Latin-1 accented letters (lower)
        "agrave" => '\u{00E0}', "aacute" => '\u{00E1}', "acirc" => '\u{00E2}',
        "atilde" => '\u{00E3}', "auml" => '\u{00E4}', "aring" => '\u{00E5}',
        "aelig" => '\u{00E6}', "ccedil" => '\u{00E7}', "egrave" => '\u{00E8}',
        "eacute" => '\u{00E9}', "ecirc" => '\u{00EA}', "euml" => '\u{00EB}',
        "igrave" => '\u{00EC}', "iacute" => '\u{00ED}', "icirc" => '\u{00EE}',
        "iuml" => '\u{00EF}', "ntilde" => '\u{00F1}', "ograve" => '\u{00F2}',
        "oacute" => '\u{00F3}', "ocirc" => '\u{00F4}', "otilde" => '\u{00F5}',
        "ouml" => '\u{00F6}', "oslash" => '\u{00F8}', "ugrave" => '\u{00F9}',
        "uacute" => '\u{00FA}', "ucirc" => '\u{00FB}', "uuml" => '\u{00FC}',
        "yacute" => '\u{00FD}', "yuml" => '\u{00FF}',
        _ => return None,
    })
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
            "nbsp" => out.push('\u{00A0}'),
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
            // Named entities beyond the basic five: the common symbols/
            // punctuation and the Latin-1 letter set real pages use (`&copy;`,
            // `&mdash;`, `&eacute;`, …). An unknown name is left verbatim
            // (`&` + name), matching the pre-existing fallthrough.
            _ => match named_entity(&ent) {
                Some(ch) => out.push(ch),
                None => {
                    out.push('&');
                    out.push_str(&ent);
                }
            },
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
    fn meta_refresh_forms_and_non_redirects() {
        // The only redirect a page can express in markup. google.com/search
        // serves exactly this to a browser it will not give results to.
        assert_eq!(
            meta_refresh(r#"<meta content="0;url=/next" http-equiv="refresh">"#),
            Some((0, String::from("/next")))
        );
        // Attribute order, spacing, quoting and case all vary in the wild.
        assert_eq!(
            meta_refresh(r#"<META HTTP-EQUIV="Refresh" CONTENT="5; URL='/a b'">"#),
            Some((5, String::from("/a b")))
        );
        assert_eq!(
            meta_refresh("<meta http-equiv=refresh content=2;url=http://e.example/x>"),
            Some((2, String::from("http://e.example/x")))
        );
        // A fractional delay truncates rather than failing the whole tag.
        assert_eq!(
            meta_refresh(r#"<meta http-equiv="refresh" content="1.5; url=/z">"#),
            Some((1, String::from("/z")))
        );
        // No url = a self-reload timer, not a navigation. Following it loops.
        assert_eq!(meta_refresh(r#"<meta http-equiv="refresh" content="30">"#), None);
        // Not a refresh, and not a <meta> at all.
        assert_eq!(
            meta_refresh(r#"<meta http-equiv="content-type" content="0;url=/x">"#),
            None
        );
        assert_eq!(meta_refresh("<metadata content=\"0;url=/x\">"), None);
    }

    #[test_case]
    fn a_style_inside_noscript_does_not_apply() {
        // Scripting is enabled, so `<noscript>` content is raw text: its
        // markup — including any `<style>` — does not exist. Harvesting styles
        // from the raw source applied it anyway, and the standard "blank the
        // page for non-JS visitors" rule then blanked the page for us.
        // google.com/search ships `table,div,span,p{display:none}` that way.
        let (cleaned, styles, _tags, ordered) = extract_assets_rich(
            r#"<html><body>
                 <noscript><style>p{display:none}</style><p>fallback</p></noscript>
                 <style>p{color:#123456}</style>
                 <p>real content</p>
               </body></html>"#,
        );
        assert!(
            !styles.contains("display:none"),
            "the noscript style must not reach the page: {styles:?}"
        );
        assert!(styles.contains("#123456"), "the real style still does: {styles:?}");
        assert_eq!(ordered.len(), 1, "only the page's own sheet is recorded");
        assert!(
            !cleaned.contains("fallback"),
            "noscript content is not rendered either"
        );
        assert!(cleaned.contains("real content"));
    }

    #[test_case]
    fn a_link_stylesheet_inside_noscript_does_not_apply() {
        // Same rule, external form.
        let (_cleaned, _styles, _tags, ordered) = extract_assets_rich(
            r#"<html><head>
                 <noscript><link rel="stylesheet" href="hide.css"></noscript>
                 <link rel="stylesheet" href="real.css">
               </head><body><p>x</p></body></html>"#,
        );
        let hrefs: alloc::vec::Vec<String> = ordered
            .iter()
            .filter_map(|s| match s {
                StyleSrc::External(h) => Some(h.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(hrefs, alloc::vec![String::from("real.css")], "got {hrefs:?}");
    }

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

    #[test_case]
    fn preprocess_preserves_utf8_text() {
        // Multi-byte UTF-8 (Devanagari) in text must survive — the old
        // byte-as-char path turned "हिन्दी" into Latin-1 mojibake (google.com's
        // "Google offered in:" language list).
        let out = preprocess("<p>हिन्दी 中文</p>");
        assert!(out.contains("हिन्दी"), "devanagari preserved: {:?}", out);
        assert!(out.contains("中文"), "cjk preserved");
    }

    #[test_case]
    fn decode_entities_named() {
        // Common symbols real pages emit (google.com's footer uses `&copy;`).
        assert_eq!(decode_entities("&copy; 2026"), "\u{00A9} 2026");
        assert_eq!(decode_entities("a&mdash;b"), "a\u{2014}b");
        assert_eq!(decode_entities("caf&eacute;"), "caf\u{00E9}");
        assert_eq!(decode_entities("&laquo;x&raquo;"), "\u{00AB}x\u{00BB}");
        assert_eq!(decode_entities("&rarr;&hearts;&euro;"), "\u{2192}\u{2665}\u{20AC}");
        // NBSP decodes to U+00A0 (not a plain space).
        assert_eq!(decode_entities("a&nbsp;b"), "a\u{00A0}b");
        // An unknown name is left verbatim (google's stray `&z;`).
        assert_eq!(decode_entities("&z;"), "&z");
        assert_eq!(decode_entities("&notareal;"), "&notareal");
    }

    #[test_case]
    fn script_tags_attr_mix() {
        let doc = parse(concat!(
            r#"<html><head>"#,
            r#"<script>inline1()</script>"#,
            r#"<script src="a.js" async></script>"#,
            r#"<script src='b.js' defer></script>"#,
            r#"<script type="module">mod()</script>"#,
            r#"</head><body></body></html>"#,
        ));
        assert_eq!(doc.script_tags.len(), 4);
        let t = &doc.script_tags[0];
        assert_eq!(t.src, None);
        assert_eq!(t.body, "inline1()");
        assert!(!t.async_ && !t.defer && !t.module);
        let t = &doc.script_tags[1];
        assert_eq!(t.src.as_deref(), Some("a.js"));
        assert!(t.async_ && !t.defer);
        let t = &doc.script_tags[2];
        assert_eq!(t.src.as_deref(), Some("b.js"));
        assert!(t.defer && !t.async_);
        let t = &doc.script_tags[3];
        assert_eq!(t.src, None);
        assert!(t.module);
        assert_eq!(t.body, "mod()");
        // Legacy view: inline bodies only.
        assert_eq!(doc.scripts, alloc::vec!["inline1()".to_string(), "mod()".to_string()]);
    }

    #[test_case]
    fn styles_ordered_interleaving() {
        let doc = parse(concat!(
            r#"<html><head>"#,
            r#"<style>p{color:red}</style>"#,
            r#"<LINK REL="Stylesheet icon" href="x.css">"#,
            r#"<style>q{color:blue}</style>"#,
            r#"</head><body></body></html>"#,
        ));
        assert_eq!(doc.styles_ordered.len(), 3);
        assert_eq!(
            doc.styles_ordered[0],
            StyleSrc::Inline("p{color:red}".to_string())
        );
        assert_eq!(doc.styles_ordered[1], StyleSrc::External("x.css".to_string()));
        assert_eq!(
            doc.styles_ordered[2],
            StyleSrc::Inline("q{color:blue}".to_string())
        );
        // Legacy concatenation unchanged.
        assert!(doc.stylesheets.contains("color:red"));
        assert!(doc.stylesheets.contains("color:blue"));
        // Non-stylesheet links don't register.
        let d2 = parse(r#"<html><head><link rel="icon" href="f.ico"></head><body></body></html>"#);
        assert!(d2.styles_ordered.is_empty());
    }

    #[test_case]
    fn colspan_attr_parsed_on_td() {
        let doc = parse(r#"<html><body><table><tr><td colspan="2"></td><td>x</td></tr></table></body></html>"#);
        fn find_td_colspan(n: &Node) -> Option<u32> {
            if let NodeKind::Element {
                tag,
                colspan_attr,
                ..
            } = &n.kind
            {
                if tag == "td" {
                    if let Some(c) = colspan_attr {
                        return Some(*c);
                    }
                }
            }
            for c in &n.children {
                if let Some(v) = find_td_colspan(c) {
                    return Some(v);
                }
            }
            None
        }
        assert_eq!(find_td_colspan(&doc.root), Some(2));
    }

    #[test_case]
    fn on_attrs_and_rel_srcset_captured() {
        let doc = parse(concat!(
            r#"<html><body>"#,
            r#"<button onclick="doThing()" onmouseover='hover()'>Go</button>"#,
            r#"<img srcset="a.png 480w, b.png 800w" src="a.png">"#,
            r#"<link rel="stylesheet" href="s.css">"#,
            r#"</body></html>"#,
        ));
        fn find<'a>(n: &'a Node, want: &str) -> Option<&'a Node> {
            if n.tag_name() == Some(want) {
                return Some(n);
            }
            for c in &n.children {
                if let Some(m) = find(c, want) {
                    return Some(m);
                }
            }
            None
        }
        let b = find(&doc.root, "button").expect("button");
        match &b.kind {
            NodeKind::Element { on_attrs, .. } => {
                assert_eq!(
                    on_attrs.as_slice(),
                    &[
                        ("onclick".to_string(), "doThing()".to_string()),
                        ("onmouseover".to_string(), "hover()".to_string()),
                    ]
                );
            }
            _ => panic!(),
        }
        let img = find(&doc.root, "img").expect("img");
        match &img.kind {
            NodeKind::Element { srcset, .. } => {
                assert_eq!(srcset.as_deref(), Some("a.png 480w, b.png 800w"));
            }
            _ => panic!(),
        }
        let link = find(&doc.root, "link").expect("link");
        match &link.kind {
            NodeKind::Element { rel, href, .. } => {
                assert_eq!(rel.as_deref(), Some("stylesheet"));
                assert_eq!(href.as_deref(), Some("s.css"));
            }
            _ => panic!(),
        }
    }

    #[test_case]
    fn data_attrs_and_href_preserved() {
        let doc = parse(
            r#"<html><body><div id="box" data-k="v" class="a"></div><a id="link" href="index.html">x</a></body></html>"#,
        );
        fn find<'a>(n: &'a Node, want_id: &str) -> Option<&'a Node> {
            if let NodeKind::Element { id: Some(i), .. } = &n.kind {
                if i == want_id {
                    return Some(n);
                }
            }
            for c in &n.children {
                if let Some(m) = find(c, want_id) {
                    return Some(m);
                }
            }
            None
        }
        let boxn = find(&doc.root, "box").expect("box");
        match &boxn.kind {
            NodeKind::Element { extra_attrs, .. } => {
                assert!(
                    extra_attrs.iter().any(|(k, v)| k == "data-k" && v == "v"),
                    "{extra_attrs:?}"
                );
            }
            _ => panic!(),
        }
        let a = find(&doc.root, "link").expect("a");
        match &a.kind {
            NodeKind::Element { href, .. } => {
                assert_eq!(href.as_deref(), Some("index.html"));
            }
            _ => panic!(),
        }
    }

    #[test_case]
    fn srcset_picker_cases() {
        // Width descriptors: smallest >= vw.
        assert_eq!(
            pick_srcset_candidate("a.png 480w, b.png 800w, c.png 1200w", 600),
            Some("b.png".to_string())
        );
        // Nothing covers vw: largest available.
        assert_eq!(
            pick_srcset_candidate("a.png 480w, b.png 800w", 2000),
            Some("b.png".to_string())
        );
        // Exact match counts.
        assert_eq!(
            pick_srcset_candidate("a.png 480w, b.png 800w", 480),
            Some("a.png".to_string())
        );
        // Density: prefer 1x (bare candidate counts as 1x).
        assert_eq!(
            pick_srcset_candidate("hi.png 2x, lo.png 1x", 999),
            Some("lo.png".to_string())
        );
        assert_eq!(
            pick_srcset_candidate("base.png, hi.png 2x", 0),
            Some("base.png".to_string())
        );
        // No 1x: first candidate.
        assert_eq!(
            pick_srcset_candidate("hi.png 2x, hi3.png 3x", 0),
            Some("hi.png".to_string())
        );
        assert_eq!(pick_srcset_candidate("  ", 100), None);
    }

    #[test_case]
    fn elem_idx_defaults_none() {
        assert_eq!(Node::element("div").elem_idx, None);
        assert_eq!(Node::text("hi").elem_idx, None);
        let doc = parse("<html><body><p>Hi</p></body></html>");
        fn all_none(n: &Node) -> bool {
            n.elem_idx.is_none() && n.children.iter().all(all_none)
        }
        assert!(all_none(&doc.root));
    }
}
