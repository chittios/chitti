//! Block + inline layout for the browser agent. Pure: DOM + CSS → boxes.
//!
//! Element display defaults come from [`super::elements`] (MDN catalog /
//! Ladybird `HTML*Element` → layout node mapping). Containers with
//! `display:flex` / `display:grid` run a formatting context that lays each
//! child out as a fragment, then repositions boxes via [`super::flex`]
//! (Ladybird `FlexFormattingContext` / `GridFormattingContext` spirit).

use super::css::{self, Align, ComputedStyle, DisplayMode, FlexDirection, Stylesheet};
use super::elements::{self, DisplayKind};
use super::flex;
use super::html::{Node, NodeKind};
use alloc::string::{String, ToString};
use alloc::vec::Vec;

pub const DEFAULT_W: i32 = 640;
pub const DEFAULT_H: i32 = 400;

/// Fallback monospace cell if TTF is unavailable.
pub const CELL_W: i32 = 8;
pub const CELL_H_BASE: i32 = 16;

#[derive(Clone, Debug)]
pub struct TextRun {
    pub text: String,
    pub x: i32,
    pub y: i32,
    pub color: u32,
    pub link_href: Option<String>,
    pub font_size: i32,
    pub bold: bool,
}

#[derive(Clone, Debug)]
pub struct LinkBox {
    pub href: String,
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
}

#[derive(Clone, Debug)]
pub struct RectBox {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
    pub color: u32,
}

/// A laid-out image (decoded later or placeholder).
#[derive(Clone, Debug)]
pub struct ImageBox {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
    pub src: String,
    pub alt: String,
    /// Decoded `0x00RRGGBB` pixels (row-major); `src_w`×`src_h` before scale.
    pub pixels: Option<alloc::vec::Vec<u32>>,
    pub src_w: usize,
    pub src_h: usize,
}

/// Form control kind (Ladybird `HTMLInputElement` type subset).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ControlKind {
    Text,
    Password,
    Submit,
    Button,
    Checkbox,
    Hidden,
    TextArea,
}

impl ControlKind {
    pub fn from_input_type(t: Option<&str>, tag: &str) -> Self {
        if tag == "textarea" {
            return ControlKind::TextArea;
        }
        if tag == "button" {
            return ControlKind::Button;
        }
        match t.unwrap_or("text").to_ascii_lowercase().as_str() {
            "password" => ControlKind::Password,
            "submit" => ControlKind::Submit,
            "button" | "reset" => ControlKind::Button,
            "checkbox" | "radio" => ControlKind::Checkbox,
            "hidden" => ControlKind::Hidden,
            _ => ControlKind::Text,
        }
    }

    pub fn is_text_entry(self) -> bool {
        matches!(self, ControlKind::Text | ControlKind::Password | ControlKind::TextArea)
    }

    pub fn is_submit(self) -> bool {
        matches!(self, ControlKind::Submit)
    }
}

/// What kind of embedded box this is (iframe / video / canvas / …).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmbedKind {
    Iframe,
    Video,
    Audio,
    Canvas,
    Other,
}

/// Nested browsing context / media / canvas box.
/// Ladybird: `HTMLIFrameElement` / `HTMLVideoElement` / `HTMLCanvasElement`.
#[derive(Clone, Debug)]
pub struct FrameBox {
    pub index: usize,
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
    pub kind: EmbedKind,
    /// Absolute or relative URL of the nested document / media.
    pub src: String,
    /// Inline document HTML (`srcdoc`).
    pub srcdoc: String,
    pub name: String,
    pub sandbox: String,
    /// Host-filled RGB of nested paint / video frame / canvas buffer.
    pub pixels: Option<alloc::vec::Vec<u32>>,
    pub src_w: usize,
    pub src_h: usize,
    /// Canvas 2D state when `kind == Canvas` (host/JS mutates via index).
    pub canvas_id: Option<usize>,
}

/// Laid-out form control (input / button / textarea).
#[derive(Clone, Debug)]
pub struct FormControl {
    pub index: usize,
    pub kind: ControlKind,
    pub name: String,
    pub value: String,
    pub placeholder: String,
    pub form_action: String,
    pub form_method: String,
    /// Form group id (same for controls in one `<form>`; 0 = orphan).
    pub form_id: u32,
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
    pub focused: bool,
    pub checked: bool,
}

/// Hit-test result in content coordinates (y includes scroll offset).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Hit {
    Link(String),
    Control(usize),
    /// Embedded frame (iframe / video / canvas / …) by layout index.
    Embed(usize),
    /// Empty page area (still over document).
    Page,
}

#[derive(Clone, Debug)]
pub struct Layout {
    pub width: i32,
    pub height: i32,
    pub content_height: i32,
    pub runs: Vec<TextRun>,
    pub links: Vec<LinkBox>,
    pub rects: Vec<RectBox>,
    pub images: alloc::vec::Vec<ImageBox>,
    pub controls: Vec<FormControl>,
    pub frames: Vec<FrameBox>,
    pub bg: u32,
}

struct Cursor {
    x: i32,
    y: i32,
    max_w: i32,
    margin_x: i32,
    line_h: i32,
    content_bottom: i32,
}

/// Active form context while walking (Ladybird form owner association).
struct FormCtx {
    id: u32,
    action: String,
    method: String,
}

/// Layout `root` with stylesheet into a page of size `vw`×`vh`.
pub fn layout_document(root: &Node, sheet: &Stylesheet, vw: i32, vh: i32) -> Layout {
    let mut runs = Vec::new();
    let mut links = Vec::new();
    let mut rects = Vec::new();
    let mut images = Vec::new();
    let mut controls = Vec::new();
    let mut frames = Vec::new();
    let margin = 8i32;
    let mut cur = Cursor {
        x: margin,
        y: margin,
        max_w: vw - margin * 2,
        margin_x: margin,
        line_h: CELL_H_BASE,
        content_bottom: margin,
    };
    let root_style = ComputedStyle::default();
    // Page background: stylesheet rules for html/body first (LibWeb StyleComputer
    // similarly resolves canvas background from the root / body).
    let mut page_bg = 0xf5f0e8u32;
    {
        let mut probe = ComputedStyle::default();
        css::apply_decls(&mut probe, &sheet.matching_decls("html", None, None));
        css::apply_decls(&mut probe, &sheet.matching_decls("body", None, None));
        if let Some(bg) = probe.background {
            page_bg = bg;
        } else if let Some(bg) = body_background(root, sheet) {
            page_bg = bg;
        }
    }
    let mut next_form_id = 1u32;
    walk(
        root,
        sheet,
        &root_style,
        &mut cur,
        &mut runs,
        &mut links,
        &mut rects,
        &mut images,
        &mut controls,
        &mut frames,
        None,
        None,
        &mut next_form_id,
        &mut page_bg,
        vw,
        false, // in_head
    );
    cur.content_bottom = cur.content_bottom.max(cur.y + cur.line_h);
    Layout {
        width: vw,
        height: vh,
        content_height: cur.content_bottom + margin,
        runs,
        links,
        rects,
        images,
        controls,
        frames,
        bg: page_bg,
    }
}

fn body_background(root: &Node, sheet: &Stylesheet) -> Option<u32> {
    fn find(n: &Node, sheet: &Stylesheet, parent: &ComputedStyle) -> Option<u32> {
        if let NodeKind::Element {
            tag,
            id,
            class,
            style_attr,
            ..
        } = &n.kind
        {
            let st = css::compute(
                sheet,
                tag,
                id.as_deref(),
                class.as_deref(),
                style_attr.as_deref(),
                parent,
            );
            if tag == "body" || tag == "html" {
                if let Some(bg) = st.background {
                    return Some(bg);
                }
            }
            for c in &n.children {
                if let Some(bg) = find(c, sheet, &st) {
                    return Some(bg);
                }
            }
        } else {
            for c in &n.children {
                if let Some(bg) = find(c, sheet, parent) {
                    return Some(bg);
                }
            }
        }
        None
    }
    find(root, sheet, &ComputedStyle::default())
}

/// Back-compat: layout with empty stylesheet.
pub fn layout_document_plain(root: &Node, vw: i32, vh: i32) -> Layout {
    layout_document(root, &Stylesheet::default(), vw, vh)
}

/// Reader-mode layout: title + plain paragraphs on warm paper (no CSS).
/// Used when the DOM/CSS path produces almost no visible text (JS-heavy pages).
pub fn layout_reader(title: &str, plain: &str, vw: i32, vh: i32) -> Layout {
    let mut runs = Vec::new();
    let mut cur = Cursor {
        x: 8,
        y: 8,
        max_w: vw - 16,
        margin_x: 8,
        line_h: CELL_H_BASE + 4,
        content_bottom: 8,
    };
    let title_st = ComputedStyle {
        color: 0x1a1816,
        font_size: 20,
        bold: true,
        ..ComputedStyle::default()
    };
    if !title.is_empty() {
        emit_text(title, &mut cur, &mut runs, &mut Vec::new(), None, &title_st);
        new_line(&mut cur);
        cur.y += 6;
    }
    let body_st = ComputedStyle {
        color: 0x2a2a2a,
        font_size: 14,
        ..ComputedStyle::default()
    };
    // Soft-wrap plain text by whitespace.
    for para in plain.split(|c: char| c == '\n' || c == '\r') {
        let para = para.trim();
        if para.is_empty() {
            continue;
        }
        emit_text(para, &mut cur, &mut runs, &mut Vec::new(), None, &body_st);
        new_line(&mut cur);
        cur.y += 4;
        cur.content_bottom = cur.content_bottom.max(cur.y);
    }
    if runs.is_empty() {
        emit_text(
            "(no visible text — page may be script-only)",
            &mut cur,
            &mut runs,
            &mut Vec::new(),
            None,
            &body_st,
        );
    }
    Layout {
        width: vw,
        height: vh,
        content_height: cur.content_bottom.max(cur.y + cur.line_h) + 8,
        runs,
        links: Vec::new(),
        rects: Vec::new(),
        images: Vec::new(),
        controls: Vec::new(),
        frames: Vec::new(),
        bg: 0xf5f0e8,
    }
}

fn walk(
    n: &Node,
    sheet: &Stylesheet,
    parent_st: &ComputedStyle,
    cur: &mut Cursor,
    runs: &mut Vec<TextRun>,
    links: &mut Vec<LinkBox>,
    rects: &mut Vec<RectBox>,
    images: &mut Vec<ImageBox>,
    controls: &mut Vec<FormControl>,
    frames: &mut Vec<FrameBox>,
    link: Option<&str>,
    form: Option<&FormCtx>,
    next_form_id: &mut u32,
    page_bg: &mut u32,
    vw: i32,
    in_head: bool,
) {
    match &n.kind {
        NodeKind::Document => {
            for c in &n.children {
                walk(
                    c, sheet, parent_st, cur, runs, links, rects, images, controls, frames, link,
                    form, next_form_id, page_bg, vw, in_head,
                );
            }
        }
        NodeKind::Text(t) => {
            if in_head {
                return;
            }
            emit_text(t, cur, runs, links, link, parent_st);
        }
        NodeKind::Element {
            tag,
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
            target: _,
            sandbox,
            width_attr,
            height_attr,
        } => {
            let tag = tag.as_str();
            let dkind = elements::classify(tag);
            if matches!(dkind, DisplayKind::None) || matches!(tag, "script" | "style" | "noscript" | "meta" | "link" | "title" | "template" | "head") {
                return;
            }
            let in_head = in_head || tag == "head";
            if in_head {
                return;
            }
            let st = css::compute(
                sheet,
                tag,
                id.as_deref(),
                class.as_deref(),
                style_attr.as_deref(),
                parent_st,
            );
            if st.display_none || st.display == DisplayMode::None {
                return;
            }
            if tag == "body" {
                if let Some(bg) = st.background {
                    *page_bg = bg;
                }
            }
            let px = st.font_size.max(8) as f32;
            let line_h = (crate::font_ttf::line_height(px) + 0.5) as i32;

            // Flex / Grid formatting context — children become independently
            // measured fragments, then translated into place (not block-stacked).
            if matches!(st.display, DisplayMode::Flex | DisplayMode::Grid)
                && n.children
                    .iter()
                    .any(|c| matches!(c.kind, NodeKind::Element { .. }))
                && !matches!(
                    tag,
                    "img" | "input" | "br" | "hr" | "iframe" | "canvas" | "video" | "audio"
                )
            {
                layout_flex_grid_container(
                    n,
                    sheet,
                    &st,
                    cur,
                    runs,
                    links,
                    rects,
                    images,
                    controls,
                    frames,
                    link,
                    form,
                    next_form_id,
                    page_bg,
                    vw,
                    in_head,
                    line_h,
                );
                return;
            }

            match tag {
                "br" | "wbr" => {
                    cur.line_h = line_h;
                    new_line(cur);
                }
                "hr" => {
                    new_line(cur);
                    cur.y += st.margin_top.max(4);
                    rects.push(RectBox {
                        x: cur.margin_x,
                        y: cur.y,
                        w: cur.max_w,
                        h: 1,
                        color: st.color,
                    });
                    cur.y += 4 + st.margin_bottom;
                    cur.content_bottom = cur.content_bottom.max(cur.y);
                }
                "img" => {
                    if cur.x > cur.margin_x {
                        new_line(cur);
                    }
                    cur.y += st.margin_top.max(0);
                    let iw = width_attr
                        .or(st.width)
                        .unwrap_or(120)
                        .clamp(16, cur.max_w);
                    let ih = height_attr.unwrap_or((iw * 3 / 4).clamp(16, 240));
                    let src_s = src.clone().unwrap_or_default();
                    let alt_s = alt.clone().unwrap_or_else(|| String::from("[img]"));
                    images.push(ImageBox {
                        x: cur.margin_x + st.margin_left,
                        y: cur.y,
                        w: iw,
                        h: ih,
                        src: src_s,
                        alt: alt_s,
                        pixels: None,
                        src_w: 0,
                        src_h: 0,
                    });
                    cur.y += ih + st.margin_bottom.max(0);
                    cur.x = cur.margin_x;
                    cur.content_bottom = cur.content_bottom.max(cur.y);
                }
                "iframe" | "frame" | "embed" | "object" | "video" | "audio" | "canvas"
                | "fencedframe" => {
                    // Embedded browsing context / media / canvas (Ladybird NavigableContainer).
                    if cur.x > cur.margin_x {
                        new_line(cur);
                    }
                    cur.y += st.margin_top.max(4);
                    let kind = match tag {
                        "video" => EmbedKind::Video,
                        "audio" => EmbedKind::Audio,
                        "canvas" => EmbedKind::Canvas,
                        "iframe" | "frame" | "fencedframe" => EmbedKind::Iframe,
                        _ => EmbedKind::Other,
                    };
                    let default_w = match kind {
                        EmbedKind::Canvas => 300,
                        EmbedKind::Video => cur.max_w.min(480),
                        EmbedKind::Iframe => cur.max_w.min(560),
                        _ => 320,
                    };
                    let default_h = match kind {
                        EmbedKind::Canvas => 150,
                        EmbedKind::Video => 270,
                        EmbedKind::Iframe => 240,
                        EmbedKind::Audio => 48,
                        _ => 180,
                    };
                    let fw = width_attr
                        .or(st.width)
                        .unwrap_or(default_w)
                        .clamp(40, cur.max_w);
                    let fh = height_attr
                        .or(st.height)
                        .unwrap_or(default_h)
                        .clamp(24, 600);
                    let idx = frames.len();
                    let mut pixels = None;
                    let mut src_w = 0;
                    let mut src_h = 0;
                    let mut canvas_id = None;
                    if kind == EmbedKind::Canvas {
                        // Allocate blank canvas buffer immediately.
                        let mut c2d = super::canvas::Canvas2d::new(fw, fh);
                        // Light grey surface so empty canvas is visible.
                        c2d.fill_style = 0xf0f0f0;
                        c2d.fill_rect(0, 0, fw, fh);
                        src_w = c2d.w;
                        src_h = c2d.h;
                        pixels = Some(c2d.pixels);
                        canvas_id = Some(idx);
                    }
                    // `<video>` / `<audio>` may use nested `<source src=…>`.
                    let mut media_src = src.clone().unwrap_or_default();
                    if media_src.is_empty()
                        && matches!(kind, EmbedKind::Video | EmbedKind::Audio)
                    {
                        for ch in &n.children {
                            if let NodeKind::Element {
                                tag: ct,
                                src: Some(s),
                                ..
                            } = &ch.kind
                            {
                                if ct == "source" && !s.is_empty() {
                                    media_src = s.clone();
                                    break;
                                }
                            }
                        }
                    }
                    frames.push(FrameBox {
                        index: idx,
                        x: cur.margin_x + st.margin_left,
                        y: cur.y,
                        w: fw,
                        h: fh,
                        kind,
                        src: media_src,
                        srcdoc: srcdoc.clone().unwrap_or_default(),
                        name: name.clone().unwrap_or_default(),
                        sandbox: sandbox.clone().unwrap_or_default(),
                        pixels,
                        src_w,
                        src_h,
                        canvas_id,
                    });
                    cur.y += fh + st.margin_bottom.max(4);
                    cur.x = cur.margin_x;
                    cur.content_bottom = cur.content_bottom.max(cur.y);
                }
                "form" => {
                    block_before(cur, st.margin_top.max(6));
                    let ctx = FormCtx {
                        id: *next_form_id,
                        action: action.clone().unwrap_or_default(),
                        method: method
                            .clone()
                            .unwrap_or_else(|| String::from("get"))
                            .to_ascii_lowercase(),
                    };
                    *next_form_id = next_form_id.saturating_add(1);
                    for c in &n.children {
                        walk(
                            c, sheet, &st, cur, runs, links, rects, images, controls, frames, link,
                            Some(&ctx), next_form_id, page_bg, vw, in_head,
                        );
                    }
                    block_after(cur, st.margin_bottom.max(6));
                }
                "input" | "button" | "textarea" | "select" => {
                    push_control(
                        cur,
                        controls,
                        tag,
                        name.as_deref(),
                        value.as_deref(),
                        input_type.as_deref(),
                        placeholder.as_deref(),
                        form,
                        n,
                        line_h,
                    );
                }
                "table" | "thead" | "tbody" | "tfoot" => {
                    // Simplified table: block container (Ladybird table layout is far larger).
                    block_before(cur, st.margin_top.max(4));
                    for c in &n.children {
                        walk(
                            c, sheet, &st, cur, runs, links, rects, images, controls, frames, link,
                            form, next_form_id, page_bg, vw, in_head,
                        );
                    }
                    block_after(cur, st.margin_bottom.max(4));
                }
                "tr" => {
                    block_before(cur, 0);
                    for c in &n.children {
                        walk(
                            c, sheet, &st, cur, runs, links, rects, images, controls, frames, link,
                            form, next_form_id, page_bg, vw, in_head,
                        );
                    }
                    if cur.x > cur.margin_x {
                        new_line(cur);
                    }
                }
                "td" | "th" => {
                    // Cell as padded inline-block-ish block child.
                    let old_max = cur.max_w;
                    cur.max_w = (old_max / 2).max(40);
                    for c in &n.children {
                        walk(
                            c, sheet, &st, cur, runs, links, rects, images, controls, frames, link,
                            form, next_form_id, page_bg, vw, in_head,
                        );
                    }
                    cur.max_w = old_max;
                    emit_text("  ", cur, runs, links, None, &st);
                }
                "h1" | "h2" | "h3" | "h4" | "h5" | "h6" | "p" | "div" | "blockquote"
                | "section" | "article" | "li" | "header" | "footer" | "main" | "nav"
                | "aside" | "figure" | "figcaption" | "address" | "search" | "details"
                | "summary" | "dialog" | "fieldset" | "legend" | "center" | "dl" | "dt"
                | "dd" | "pre" | "menu" | "hgroup" => {
                    block_before(cur, st.margin_top);
                    cur.line_h = line_h;
                    let block_y0 = cur.y;
                    let block_x0 = cur.margin_x + st.margin_left;
                    if tag == "li" {
                        emit_text("• ", cur, runs, links, link, &st);
                    }
                    cur.x = (cur.margin_x + st.margin_left + st.padding_top.min(0)).max(cur.margin_x);
                    cur.y += st.padding_top;
                    let content_start_x = cur.x;
                    let old_max = cur.max_w;
                    if let Some(w) = st.width {
                        cur.max_w = w.min(old_max);
                    } else {
                        cur.max_w = old_max - st.margin_left - st.margin_right;
                    }
                    for c in &n.children {
                        walk(
                            c, sheet, &st, cur, runs, links, rects, images, controls, frames, link,
                            form, next_form_id, page_bg, vw, in_head,
                        );
                    }
                    if cur.x > content_start_x {
                        new_line(cur);
                    }
                    cur.y += st.padding_bottom;
                    if let Some(bg) = st.background {
                        let h = (cur.y - block_y0).max(line_h);
                        rects.push(RectBox {
                            x: block_x0,
                            y: block_y0,
                            w: cur.max_w.max(1),
                            h,
                            color: bg,
                        });
                    }
                    let _ = (st.text_align, Align::Left, content_start_x, vw);
                    cur.max_w = old_max;
                    block_after(cur, st.margin_bottom);
                }
                "a" => {
                    let href_s = href.as_deref();
                    let mut st_a = st;
                    if st_a.color == parent_st.color {
                        st_a.color = 0x1a73e8;
                    }
                    for c in &n.children {
                        walk(
                            c, sheet, &st_a, cur, runs, links, rects, images, controls, frames,
                            href_s.or(link), form, next_form_id, page_bg, vw, in_head,
                        );
                    }
                }
                "span" | "strong" | "b" | "em" | "i" | "code" | "label" | "small" | "u" | "s"
                | "sub" | "sup" | "mark" | "abbr" | "cite" | "q" | "time" | "var" | "kbd"
                | "samp" | "dfn" | "bdi" | "bdo" | "data" | "output" | "ins" | "del" | "font"
                | "tt" | "big" | "strike" | "ruby" | "rt" | "rb" | "slot" => {
                    let mut st_i = st;
                    if matches!(tag, "strong" | "b") {
                        st_i.bold = true;
                    }
                    cur.line_h = line_h.max(cur.line_h);
                    for c in &n.children {
                        walk(
                            c, sheet, &st_i, cur, runs, links, rects, images, controls, frames,
                            link, form, next_form_id, page_bg, vw, in_head,
                        );
                    }
                }
                "ul" | "ol" | "body" | "html" => {
                    cur.line_h = line_h;
                    for c in &n.children {
                        walk(
                            c, sheet, &st, cur, runs, links, rects, images, controls, frames, link,
                            form, next_form_id, page_bg, vw, in_head,
                        );
                    }
                }
                _ => {
                    // Unknown or remaining catalog tags: block vs inline from DisplayKind.
                    match dkind {
                        DisplayKind::Inline | DisplayKind::InlineBlock => {
                            cur.line_h = line_h.max(cur.line_h);
                            for c in &n.children {
                                walk(
                                    c, sheet, &st, cur, runs, links, rects, images, controls,
                                    frames, link, form, next_form_id, page_bg, vw, in_head,
                                );
                            }
                        }
                        DisplayKind::ListItem => {
                            block_before(cur, st.margin_top);
                            emit_text("• ", cur, runs, links, link, &st);
                            for c in &n.children {
                                walk(
                                    c, sheet, &st, cur, runs, links, rects, images, controls,
                                    frames, link, form, next_form_id, page_bg, vw, in_head,
                                );
                            }
                            block_after(cur, st.margin_bottom);
                        }
                        _ => {
                            block_before(cur, st.margin_top);
                            for c in &n.children {
                                walk(
                                    c, sheet, &st, cur, runs, links, rects, images, controls,
                                    frames, link, form, next_form_id, page_bg, vw, in_head,
                                );
                            }
                            if cur.x > cur.margin_x {
                                new_line(cur);
                            }
                            block_after(cur, st.margin_bottom);
                        }
                    }
                }
            }
        }
    }
}

/// Indices marking the start of a fragment in each geometry buffer.
#[derive(Clone, Copy)]
struct FragMark {
    runs: usize,
    links: usize,
    rects: usize,
    images: usize,
    controls: usize,
    frames: usize,
}

fn mark_frag(
    runs: &[TextRun],
    links: &[LinkBox],
    rects: &[RectBox],
    images: &[ImageBox],
    controls: &[FormControl],
    frames: &[FrameBox],
) -> FragMark {
    FragMark {
        runs: runs.len(),
        links: links.len(),
        rects: rects.len(),
        images: images.len(),
        controls: controls.len(),
        frames: frames.len(),
    }
}

/// Axis-aligned bounds of geometry added after `start` (content coords).
fn frag_bbox(
    start: FragMark,
    runs: &[TextRun],
    links: &[LinkBox],
    rects: &[RectBox],
    images: &[ImageBox],
    controls: &[FormControl],
    frames: &[FrameBox],
) -> (i32, i32, i32, i32) {
    let mut x0 = i32::MAX;
    let mut y0 = i32::MAX;
    let mut x1 = i32::MIN;
    let mut y1 = i32::MIN;
    let mut any = false;
    let mut expand = |x: i32, y: i32, w: i32, h: i32| {
        any = true;
        x0 = x0.min(x);
        y0 = y0.min(y);
        x1 = x1.max(x + w);
        y1 = y1.max(y + h);
    };
    for r in &runs[start.runs..] {
        let w = (crate::font_ttf::measure(&r.text, r.font_size.max(8) as f32) + 0.5) as i32;
        let h = (crate::font_ttf::line_height(r.font_size.max(8) as f32) + 0.5) as i32;
        expand(r.x, r.y, w.max(1), h.max(1));
    }
    for l in &links[start.links..] {
        expand(l.x, l.y, l.w.max(1), l.h.max(1));
    }
    for r in &rects[start.rects..] {
        expand(r.x, r.y, r.w.max(1), r.h.max(1));
    }
    for im in &images[start.images..] {
        expand(im.x, im.y, im.w.max(1), im.h.max(1));
    }
    for c in &controls[start.controls..] {
        if c.w > 0 && c.h > 0 {
            expand(c.x, c.y, c.w, c.h);
        }
    }
    for f in &frames[start.frames..] {
        expand(f.x, f.y, f.w.max(1), f.h.max(1));
    }
    if !any {
        return (0, 0, 1, 1);
    }
    (x0, y0, x1, y1)
}

fn translate_frag(
    start: FragMark,
    dx: i32,
    dy: i32,
    runs: &mut [TextRun],
    links: &mut [LinkBox],
    rects: &mut [RectBox],
    images: &mut [ImageBox],
    controls: &mut [FormControl],
    frames: &mut [FrameBox],
) {
    if dx == 0 && dy == 0 {
        return;
    }
    for r in &mut runs[start.runs..] {
        r.x += dx;
        r.y += dy;
    }
    for l in &mut links[start.links..] {
        l.x += dx;
        l.y += dy;
    }
    for r in &mut rects[start.rects..] {
        r.x += dx;
        r.y += dy;
    }
    for im in &mut images[start.images..] {
        im.x += dx;
        im.y += dy;
    }
    for c in &mut controls[start.controls..] {
        c.x += dx;
        c.y += dy;
    }
    for f in &mut frames[start.frames..] {
        f.x += dx;
        f.y += dy;
    }
}

/// Layout children of a flex/grid container: measure each element child as an
/// independent fragment, then place with [`flex`] geometry.
fn layout_flex_grid_container(
    n: &Node,
    sheet: &Stylesheet,
    st: &ComputedStyle,
    cur: &mut Cursor,
    runs: &mut Vec<TextRun>,
    links: &mut Vec<LinkBox>,
    rects: &mut Vec<RectBox>,
    images: &mut Vec<ImageBox>,
    controls: &mut Vec<FormControl>,
    frames: &mut Vec<FrameBox>,
    link: Option<&str>,
    form: Option<&FormCtx>,
    next_form_id: &mut u32,
    page_bg: &mut u32,
    vw: i32,
    in_head: bool,
    line_h: i32,
) {
    block_before(cur, st.margin_top.max(0));
    cur.y += st.padding_top;
    let box_x = cur.margin_x + st.margin_left;
    let box_y = cur.y;
    let container_w = st
        .width
        .or(st.max_width)
        .unwrap_or_else(|| (cur.max_w - st.margin_left - st.margin_right).max(1))
        .max(1);

    let children: Vec<&Node> = n
        .children
        .iter()
        .filter(|c| matches!(c.kind, NodeKind::Element { .. }))
        .collect();
    let n_items = children.len().max(1);

    // Equal-share item width for row flex / grid (overridden by child width).
    let gap = if st.display == DisplayMode::Grid {
        st.grid_gap
    } else {
        st.flex_gap
    };
    let default_item_w = match st.display {
        DisplayMode::Grid => {
            let cols = st.grid_columns.max(1);
            flex::grid_col_width(container_w, cols, gap)
        }
        DisplayMode::Flex if st.flex_direction == FlexDirection::Row => {
            let gaps = gap * (n_items as i32 - 1).max(0);
            ((container_w - gaps) / n_items as i32).max(24)
        }
        _ => container_w,
    };

    // Background rect first so it paints under child geometry (paint walks rects in order).
    let bg_idx = st.background.map(|bg| {
        let i = rects.len();
        rects.push(RectBox {
            x: box_x,
            y: box_y,
            w: container_w,
            h: 1,
            color: bg,
        });
        i
    });

    struct Item {
        mark: FragMark,
        x0: i32,
        y0: i32,
        w: i32,
        h: i32,
        grow: u32,
    }
    let mut items: Vec<Item> = Vec::with_capacity(children.len());

    for child in &children {
        // Child style for preferred width / flex-grow.
        let (ctag, cid, cclass, cstyle) = match &child.kind {
            NodeKind::Element {
                tag,
                id,
                class,
                style_attr,
                ..
            } => (
                tag.as_str(),
                id.as_deref(),
                class.as_deref(),
                style_attr.as_deref(),
            ),
            _ => continue,
        };
        let cst = css::compute(sheet, ctag, cid, cclass, cstyle, st);
        let item_w = cst
            .width
            .or(cst.max_width)
            .unwrap_or(default_item_w)
            .clamp(16, container_w);

        let mark = mark_frag(runs, links, rects, images, controls, frames);
        // Isolated cursor at origin so fragment coords start near (0,0).
        // Auto-size: use natural max_w; for row flex wrap prefer content width.
        let measure_w = if st.display == DisplayMode::Flex
            && st.flex_direction == FlexDirection::Row
            && st.flex_wrap != flex::FlexWrap::NoWrap
        {
            // Prefer natural width: roomy measure, then clamp to container.
            container_w
        } else {
            item_w
        };
        let mut child_cur = Cursor {
            x: 0,
            y: 0,
            max_w: measure_w,
            margin_x: 0,
            line_h,
            content_bottom: 0,
        };
        walk(
            child,
            sheet,
            st,
            &mut child_cur,
            runs,
            links,
            rects,
            images,
            controls,
            frames,
            link,
            form,
            next_form_id,
            page_bg,
            vw,
            in_head,
        );
        let (x0, y0, x1, y1) =
            frag_bbox(mark, runs, links, rects, images, controls, frames);
        let mut w = (x1 - x0).max(1);
        let mut h = (y1 - y0).max(1);
        if let Some(eh) = cst.height {
            h = h.max(eh);
        } else {
            h = h.max(child_cur.content_bottom.max(1));
        }
        if let Some(ew) = cst.width {
            w = ew.clamp(1, container_w);
        } else if st.display == DisplayMode::Flex
            && st.flex_direction == FlexDirection::Row
            && st.flex_wrap != flex::FlexWrap::NoWrap
        {
            // Natural content width for wrapping.
            w = w.min(container_w).max(16);
        }
        if w <= 1 && h <= 1 {
            w = item_w.min(container_w).max(24);
            h = line_h.max(16);
        }
        let grow = if cst.flex_grow > 0 {
            cst.flex_grow
        } else {
            st.flex_grow
        };
        items.push(Item {
            mark,
            x0,
            y0,
            w,
            h,
            grow,
        });
    }

    let mut content_h = line_h;
    let mut content_w = container_w;

    match st.display {
        DisplayMode::Flex => {
            let widths: Vec<i32> = items.iter().map(|it| it.w).collect();
            let heights: Vec<i32> = items.iter().map(|it| it.h).collect();
            let grows: Vec<u32> = items.iter().map(|it| it.grow).collect();
            let container_h = st.height.unwrap_or(10_000).max(1);
            let placed = flex::flex_place(
                &widths,
                &heights,
                &grows,
                container_w,
                container_h,
                gap,
                st.flex_direction,
                st.justify_content,
                st.align_items,
                st.flex_wrap,
                st.max_height.or(st.height),
            );
            let mut max_x = container_w;
            let mut max_y = line_h;
            for p in &placed {
                if let Some(it) = items.get(p.index) {
                    let dx = box_x + p.x - it.x0;
                    let dy = box_y + p.y - it.y0;
                    translate_frag(it.mark, dx, dy, runs, links, rects, images, controls, frames);
                    max_x = max_x.max(p.x + p.w);
                    max_y = max_y.max(p.y + p.h);
                }
            }
            content_h = max_y;
            content_w = max_x.max(container_w);
        }
        DisplayMode::Grid => {
            let cols = st.grid_columns.max(1);
            let cell_w = flex::grid_col_width(container_w, cols, gap);
            let cell_h = items
                .iter()
                .map(|it| it.h)
                .max()
                .unwrap_or(line_h)
                .max(line_h);
            if st.grid_dense {
                let sc: Vec<u8> = items.iter().map(|_| 1u8).collect();
                let sr: Vec<u8> = items.iter().map(|_| 1u8).collect();
                let placed = flex::grid_dense_place(&sc, &sr, cols, cell_w, cell_h, gap);
                let mut max_y = line_h;
                for p in &placed {
                    if let Some(it) = items.get(p.index) {
                        let ix =
                            flex::flex_cross_offset(it.w.min(cell_w), cell_w, st.align_items);
                        let iy =
                            flex::flex_cross_offset(it.h.min(cell_h), cell_h, st.align_items);
                        let dx = box_x + p.x + ix - it.x0;
                        let dy = box_y + p.y + iy - it.y0;
                        translate_frag(
                            it.mark, dx, dy, runs, links, rects, images, controls, frames,
                        );
                        max_y = max_y.max(p.y + p.h);
                    }
                }
                content_h = max_y;
            } else {
                for (i, it) in items.iter().enumerate() {
                    let (gx, gy) = flex::grid_cell(i, cols, cell_w, cell_h, gap);
                    let ix = flex::flex_cross_offset(it.w.min(cell_w), cell_w, st.align_items);
                    let iy = flex::flex_cross_offset(it.h.min(cell_h), cell_h, st.align_items);
                    let dx = box_x + gx + ix - it.x0;
                    let dy = box_y + gy + iy - it.y0;
                    translate_frag(it.mark, dx, dy, runs, links, rects, images, controls, frames);
                }
                let rows = (items.len() + cols as usize - 1) / cols as usize;
                content_h = if rows == 0 {
                    line_h
                } else {
                    rows as i32 * cell_h + gap * (rows as i32 - 1).max(0)
                };
            }
            content_w = container_w;
        }
        _ => {}
    }

    if let Some(h) = st.height {
        content_h = content_h.max(h);
    }
    if let Some(i) = bg_idx {
        rects[i].w = content_w.max(container_w);
        rects[i].h = content_h.max(1);
    }

    cur.y = box_y + content_h + st.padding_bottom;
    cur.x = cur.margin_x;
    cur.content_bottom = cur.content_bottom.max(cur.y);
    block_after(cur, st.margin_bottom);
}

fn push_control(
    cur: &mut Cursor,
    controls: &mut Vec<FormControl>,
    tag: &str,
    name: Option<&str>,
    value: Option<&str>,
    input_type: Option<&str>,
    placeholder: Option<&str>,
    form: Option<&FormCtx>,
    node: &Node,
    line_h: i32,
) {
    let kind = ControlKind::from_input_type(input_type, tag);
    if kind == ControlKind::Hidden {
        // Still register for form submit, zero geometry.
        let idx = controls.len();
        controls.push(FormControl {
            index: idx,
            kind,
            name: name.unwrap_or("").to_string(),
            value: value.unwrap_or("").to_string(),
            placeholder: String::new(),
            form_action: form.map(|f| f.action.clone()).unwrap_or_default(),
            form_method: form
                .map(|f| f.method.clone())
                .unwrap_or_else(|| String::from("get")),
            form_id: form.map(|f| f.id).unwrap_or(0),
            x: 0,
            y: 0,
            w: 0,
            h: 0,
            focused: false,
            checked: false,
        });
        return;
    }
    if cur.x > cur.margin_x + 4 {
        new_line(cur);
    }
    cur.y += 4;
    let (w, h) = match kind {
        ControlKind::Text | ControlKind::Password => (cur.max_w.min(280).max(120), line_h + 10),
        ControlKind::TextArea => (cur.max_w.min(320).max(160), line_h * 4 + 12),
        ControlKind::Submit | ControlKind::Button => {
            let label = value
                .filter(|s| !s.is_empty())
                .or(Some(if kind == ControlKind::Submit {
                    "Submit"
                } else {
                    "Button"
                }))
                .unwrap();
            let tw = (crate::font_ttf::measure(label, 13.0) as i32) + 24;
            (tw.clamp(64, cur.max_w), line_h + 12)
        }
        ControlKind::Checkbox => (18, 18),
        ControlKind::Hidden => (0, 0),
    };
    let mut val = value.unwrap_or("").to_string();
    if tag == "textarea" && val.is_empty() {
        val = super::html::collect_text(node);
    }
    if tag == "button" && val.is_empty() {
        val = super::html::collect_text(node);
        if val.is_empty() {
            val = String::from("Button");
        }
    }
    if kind == ControlKind::Submit && val.is_empty() {
        val = String::from("Submit");
    }
    let idx = controls.len();
    controls.push(FormControl {
        index: idx,
        kind,
        name: name.unwrap_or("").to_string(),
        value: val,
        placeholder: placeholder.unwrap_or("").to_string(),
        form_action: form.map(|f| f.action.clone()).unwrap_or_default(),
        form_method: form
            .map(|f| f.method.clone())
            .unwrap_or_else(|| String::from("get")),
        form_id: form.map(|f| f.id).unwrap_or(0),
        x: cur.margin_x,
        y: cur.y,
        w,
        h,
        focused: false,
        checked: false,
    });
    cur.y += h + 6;
    cur.x = cur.margin_x;
    cur.content_bottom = cur.content_bottom.max(cur.y);
}

fn block_before(cur: &mut Cursor, gap: i32) {
    if cur.x > cur.margin_x {
        new_line(cur);
    }
    cur.y += gap.max(0);
    cur.content_bottom = cur.content_bottom.max(cur.y);
}

fn block_after(cur: &mut Cursor, gap: i32) {
    cur.y += gap.max(0);
    cur.content_bottom = cur.content_bottom.max(cur.y);
}

fn new_line(cur: &mut Cursor) {
    cur.x = cur.margin_x;
    cur.y += cur.line_h;
    cur.content_bottom = cur.content_bottom.max(cur.y);
}

fn emit_text(
    text: &str,
    cur: &mut Cursor,
    runs: &mut Vec<TextRun>,
    links: &mut Vec<LinkBox>,
    link: Option<&str>,
    st: &ComputedStyle,
) {
    let px = st.font_size.max(8) as f32;
    let line_h = (crate::font_ttf::line_height(px) + 0.5) as i32;
    cur.line_h = cur.line_h.max(line_h.max(10));
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if cur.x <= cur.margin_x {
            while i < chars.len() && chars[i] == ' ' {
                i += 1;
            }
        }
        if i >= chars.len() {
            break;
        }
        let start = i;
        if chars[i] == ' ' {
            i += 1;
        } else {
            while i < chars.len() && chars[i] != ' ' {
                i += 1;
            }
        }
        let word: String = chars[start..i].iter().collect();
        // Proportional advance from the active TTF face.
        let w = (crate::font_ttf::measure(&word, px) + 0.5) as i32;
        let w = w.max(1);
        if cur.x > cur.margin_x && cur.x + w > cur.margin_x + cur.max_w {
            new_line(cur);
            if word == " " {
                continue;
            }
        }
        let run_x = cur.x;
        let run_y = cur.y;
        runs.push(TextRun {
            text: word.clone(),
            x: run_x,
            y: run_y,
            color: st.color,
            link_href: link.map(|s| s.to_string()),
            font_size: st.font_size,
            bold: st.bold,
        });
        if let Some(href) = link {
            links.push(LinkBox {
                href: href.to_string(),
                x: run_x,
                y: run_y,
                w,
                h: cur.line_h,
            });
        }
        cur.x += w;
        cur.content_bottom = cur.content_bottom.max(cur.y + cur.line_h);
    }
}

/// Link-only hit test (back-compat). Prefer [`hit_test_ex`].
pub fn hit_test(layout: &Layout, x: i32, y: i32) -> Option<String> {
    match hit_test_ex(layout, x, y) {
        Hit::Link(h) => Some(h),
        _ => None,
    }
}

/// Full hit test: controls first (topmost), then links, then frames.
pub fn hit_test_ex(layout: &Layout, x: i32, y: i32) -> Hit {
    for c in layout.controls.iter().rev() {
        if c.kind == ControlKind::Hidden || c.w <= 0 || c.h <= 0 {
            continue;
        }
        if x >= c.x && x < c.x + c.w && y >= c.y && y < c.y + c.h {
            return Hit::Control(c.index);
        }
    }
    for lb in layout.links.iter().rev() {
        if x >= lb.x && x < lb.x + lb.w && y >= lb.y && y < lb.y + lb.h {
            return Hit::Link(lb.href.clone());
        }
    }
    for f in layout.frames.iter().rev() {
        if x >= f.x && x < f.x + f.w && y >= f.y && y < f.y + f.h {
            return Hit::Embed(f.index);
        }
    }
    Hit::Page
}

/// True if point is over an iframe/embed box.
pub fn frame_at(layout: &Layout, x: i32, y: i32) -> Option<usize> {
    for f in layout.frames.iter().rev() {
        if x >= f.x && x < f.x + f.w && y >= f.y && y < f.y + f.h {
            return Some(f.index);
        }
    }
    None
}

/// Cursor affordance over content point (Ladybird pointer CSS).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CursorKind {
    Default,
    Pointer,
    Text,
}

pub fn cursor_at(layout: &Layout, x: i32, y: i32) -> CursorKind {
    match hit_test_ex(layout, x, y) {
        Hit::Link(_) => CursorKind::Pointer,
        Hit::Embed(i) => {
            if let Some(f) = layout.frames.get(i) {
                if f.kind == EmbedKind::Video || f.kind == EmbedKind::Iframe {
                    CursorKind::Pointer
                } else {
                    CursorKind::Default
                }
            } else {
                CursorKind::Default
            }
        }
        Hit::Control(i) => {
            if let Some(c) = layout.controls.get(i) {
                if c.kind.is_text_entry() {
                    CursorKind::Text
                } else {
                    CursorKind::Pointer
                }
            } else {
                CursorKind::Default
            }
        }
        Hit::Page => CursorKind::Default,
    }
}

/// Apply focus flag for paint chrome.
pub fn set_focus(layout: &mut Layout, focused: Option<usize>) {
    for c in &mut layout.controls {
        c.focused = focused == Some(c.index);
    }
}

/// Collect named fields for a form group (for submit).
pub fn form_fields(layout: &Layout, form_id: u32) -> alloc::vec::Vec<super::form::FormField> {
    let mut out = alloc::vec::Vec::new();
    for c in &layout.controls {
        if c.form_id != form_id || c.name.is_empty() {
            continue;
        }
        if c.kind == ControlKind::Submit || c.kind == ControlKind::Button {
            continue;
        }
        if c.kind == ControlKind::Checkbox && !c.checked {
            continue;
        }
        out.push(super::form::FormField {
            name: c.name.clone(),
            value: if c.kind == ControlKind::Checkbox {
                if c.value.is_empty() {
                    String::from("on")
                } else {
                    c.value.clone()
                }
            } else {
                c.value.clone()
            },
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::browser::{css, html};

    #[test_case]
    fn layout_has_text_and_link_boxes() {
        let doc = html::parse(
            r#"<html><body><h1>Title</h1><p>Hello <a href="/x">link</a> world</p></body></html>"#,
        );
        let sheet = Stylesheet::parse(&doc.stylesheets);
        let lay = layout_document(&doc.root, &sheet, DEFAULT_W, DEFAULT_H);
        assert!(!lay.runs.is_empty(), "expected text runs");
        assert!(!lay.links.is_empty(), "expected link boxes");
        let href = hit_test(&lay, lay.links[0].x + 1, lay.links[0].y + 1);
        assert_eq!(href.as_deref(), Some("/x"));
        assert_eq!(
            cursor_at(&lay, lay.links[0].x + 1, lay.links[0].y + 1),
            CursorKind::Pointer
        );
    }

    #[test_case]
    fn layout_form_controls_and_fields() {
        let doc = html::parse(
            r#"<html><body><form action="/s" method="get">
            <input type="text" name="q" value="hi" placeholder="Search">
            <input type="submit" value="Go">
            </form></body></html>"#,
        );
        let sheet = Stylesheet::parse(&doc.stylesheets);
        let lay = layout_document(&doc.root, &sheet, 400, 300);
        assert!(
            lay.controls.iter().any(|c| c.kind == ControlKind::Text && c.name == "q"),
            "controls={:?}",
            lay.controls
        );
        assert!(lay.controls.iter().any(|c| c.kind == ControlKind::Submit));
        let text = lay.controls.iter().find(|c| c.kind == ControlKind::Text).unwrap();
        assert_eq!(cursor_at(&lay, text.x + 2, text.y + 2), CursorKind::Text);
        let fields = form_fields(&lay, text.form_id);
        assert_eq!(fields.len(), 1);
        assert_eq!(fields[0].name, "q");
        assert_eq!(fields[0].value, "hi");
    }

    #[test_case]
    fn css_color_applies_to_runs() {
        let doc = html::parse(
            r#"<html><head><style>p{color:#ff0000}</style></head>
            <body><p>Red</p></body></html>"#,
        );
        let sheet = Stylesheet::parse(&doc.stylesheets);
        let lay = layout_document(&doc.root, &sheet, 320, 200);
        assert!(
            lay.runs.iter().any(|r| r.color == 0xff0000),
            "runs: {:?}",
            lay.runs.iter().map(|r| r.color).collect::<Vec<_>>()
        );
        let _ = css::parse_color("#fff");
    }

    #[test_case]
    fn layout_iframe_and_unknown_custom() {
        let doc = html::parse(
            r#"<html><body>
            <iframe src="/nested.html" width="200" height="100" name="f1"></iframe>
            <my-widget>Hello custom</my-widget>
            </body></html>"#,
        );
        let sheet = Stylesheet::parse(&doc.stylesheets);
        let lay = layout_document(&doc.root, &sheet, 400, 400);
        assert_eq!(lay.frames.len(), 1);
        assert_eq!(lay.frames[0].src, "/nested.html");
        assert_eq!(lay.frames[0].w, 200);
        assert_eq!(lay.frames[0].h, 100);
        assert!(
            lay.runs.iter().any(|r| r.text.contains("Hello") || r.text.contains("custom")),
            "custom element text should layout; runs={:?}",
            lay.runs
        );
    }

    #[test_case]
    fn flex_row_places_children_side_by_side() {
        // Three items in a flex row must not all share the same x (block stack).
        let doc = html::parse(
            r#"<html><body>
            <div id="row" style="display:flex; gap:10px; justify-content:flex-start">
              <span id="a">AAA</span>
              <span id="b">BBB</span>
              <span id="c">CCC</span>
            </div>
            </body></html>"#,
        );
        let sheet = Stylesheet::parse(&doc.stylesheets);
        let lay = layout_document(&doc.root, &sheet, 400, 200);
        let xs: Vec<i32> = lay
            .runs
            .iter()
            .filter(|r| r.text.contains('A') || r.text.contains('B') || r.text.contains('C'))
            .map(|r| r.x)
            .collect();
        assert!(
            xs.len() >= 2,
            "expected multiple flex item runs, runs={:?}",
            lay.runs
        );
        let min_x = *xs.iter().min().unwrap();
        let max_x = *xs.iter().max().unwrap();
        assert!(
            max_x > min_x + 10,
            "flex items should spread horizontally: xs={xs:?}"
        );
    }

    #[test_case]
    fn grid_places_items_in_columns() {
        let doc = html::parse(
            r#"<html><body>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px">
              <div>One</div><div>Two</div><div>Three</div><div>Four</div>
            </div>
            </body></html>"#,
        );
        let sheet = Stylesheet::parse(&doc.stylesheets);
        let lay = layout_document(&doc.root, &sheet, 400, 300);
        let ones: Vec<_> = lay.runs.iter().filter(|r| r.text.contains("One")).collect();
        let twos: Vec<_> = lay.runs.iter().filter(|r| r.text.contains("Two")).collect();
        let threes: Vec<_> = lay.runs.iter().filter(|r| r.text.contains("Three")).collect();
        assert!(!ones.is_empty() && !twos.is_empty() && !threes.is_empty());
        // Row 0: One left of Two
        assert!(
            ones[0].x < twos[0].x,
            "One.x={} Two.x={}",
            ones[0].x,
            twos[0].x
        );
        // Row 1: Three below One (greater y)
        assert!(
            threes[0].y > ones[0].y,
            "Three.y={} One.y={}",
            threes[0].y,
            ones[0].y
        );
    }

    #[test_case]
    fn flex_column_stacks_vertically() {
        let doc = html::parse(
            r#"<html><body>
            <div style="display:flex; flex-direction:column; gap:4px">
              <span>Top</span><span>Bot</span>
            </div>
            </body></html>"#,
        );
        let sheet = Stylesheet::parse(&doc.stylesheets);
        let lay = layout_document(&doc.root, &sheet, 300, 200);
        let top = lay.runs.iter().find(|r| r.text.contains("Top")).unwrap();
        let bot = lay.runs.iter().find(|r| r.text.contains("Bot")).unwrap();
        assert!(bot.y > top.y, "column flex: Bot.y={} Top.y={}", bot.y, top.y);
    }

    #[test_case]
    fn flex_wrap_second_line() {
        let doc = html::parse(
            r#"<html><body>
            <div style="display:flex; flex-wrap:wrap; width:80px; gap:0">
              <span style="width:50px">AA</span>
              <span style="width:50px">BB</span>
            </div>
            </body></html>"#,
        );
        let sheet = Stylesheet::parse(&doc.stylesheets);
        let lay = layout_document(&doc.root, &sheet, 200, 200);
        let a = lay.runs.iter().find(|r| r.text.contains('A'));
        let b = lay.runs.iter().find(|r| r.text.contains('B'));
        if let (Some(a), Some(b)) = (a, b) {
            assert!(
                b.y > a.y || b.x > a.x,
                "wrap should move second item: A=({},{}) B=({},{})",
                a.x,
                a.y,
                b.x,
                b.y
            );
        }
    }
}
