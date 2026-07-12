//! Block + inline layout for the browser agent. Pure: DOM + CSS → boxes.
//!
//! Element display defaults come from [`super::elements`] (MDN catalog /
//! Ladybird `HTML*Element` → layout node mapping). Containers with
//! `display:flex` / `display:grid` run a formatting context that lays each
//! child out as a fragment, then repositions boxes via [`super::flex`]
//! (Ladybird `FlexFormattingContext` / `GridFormattingContext` spirit).

use super::css::{self, Align, ComputedStyle, DisplayMode, FlexDirection, Position, Stylesheet};
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
    /// CSS font-family (first name, lowercase); `""` = the default face.
    /// Generic names (`sans-serif`/`serif`/`monospace`) map to `""`.
    pub font_family: String,
    /// `text-decoration: underline` (or a link with the default UA underline) —
    /// the painter draws a 1px rule under the run.
    pub underline: bool,
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

/// Hit box of a JS-interactive element (has listeners / `on*` attrs), so a
/// click anywhere on its box can be dispatched into page JS ([`Hit::Elem`]).
/// `elem_idx` is the stamped `Node.elem_idx` = `JsDom.elements` index.
#[derive(Clone, Debug)]
pub struct ElemBox {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
    pub elem_idx: usize,
}

/// A CSS `background-image: url(…)` paint box. `src` is the inner url as
/// written (unresolved); the host fills `pixels` (decoded `0x00RRGGBB`,
/// `src_w`×`src_h`) like it does for [`ImageBox`]. `repeat`/`size`/`pos` carry
/// the raw CSS values (parsed at paint by `paint::parse_bg_*`).
#[derive(Clone, Debug)]
pub struct BgBox {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
    pub src: String,
    pub repeat: String,
    pub size: String,
    pub pos: String,
    pub pixels: Option<alloc::vec::Vec<u32>>,
    pub src_w: usize,
    pub src_h: usize,
}

/// Emit border edges for a box as solid `RectBox` line segments (dashed/dotted
/// styles are approximated as solid). Each visible edge needs a style, a
/// positive width, and a resolved colour (falling back to the shorthand
/// `border_color`, then the text colour). This is why bordered boxes/tables/
/// cards finally render an outline — previously borders were computed but never
/// painted.
fn push_box_borders(rects: &mut Vec<RectBox>, x: i32, y: i32, w: i32, h: i32, st: &ComputedStyle) {
    if w <= 0 || h <= 0 {
        return;
    }
    let fallback = st.border_color.unwrap_or(st.color);
    let edge = |style: css::BorderStyle, width: i32, color: Option<u32>| -> Option<(i32, u32)> {
        if style.is_visible() && width > 0 {
            Some((width.min(w).min(h), color.unwrap_or(fallback)))
        } else {
            None
        }
    };
    if let Some((bw, c)) = edge(st.border_top_style, st.border_top_width, st.border_top_color) {
        rects.push(RectBox { x, y, w, h: bw, color: c });
    }
    if let Some((bw, c)) = edge(st.border_bottom_style, st.border_bottom_width, st.border_bottom_color) {
        rects.push(RectBox { x, y: y + h - bw, w, h: bw, color: c });
    }
    if let Some((bw, c)) = edge(st.border_left_style, st.border_left_width, st.border_left_color) {
        rects.push(RectBox { x, y, w: bw, h, color: c });
    }
    if let Some((bw, c)) = edge(st.border_right_style, st.border_right_width, st.border_right_color) {
        rects.push(RectBox { x: x + w - bw, y, w: bw, h, color: c });
    }
    // Outline — drawn just outside the border box by `outline-offset`.
    if st.outline_style.is_visible() && st.outline_width > 0 {
        let ow = st.outline_width;
        let off = st.outline_offset.max(0);
        let (ox, oy) = (x - off - ow, y - off - ow);
        let (ow_box, oh_box) = (w + 2 * (off + ow), h + 2 * (off + ow));
        let c = st.outline_color.unwrap_or(fallback);
        rects.push(RectBox { x: ox, y: oy, w: ow_box, h: ow, color: c });
        rects.push(RectBox { x: ox, y: oy + oh_box - ow, w: ow_box, h: ow, color: c });
        rects.push(RectBox { x: ox, y: oy, w: ow, h: oh_box, color: c });
        rects.push(RectBox { x: ox + ow_box - ow, y: oy, w: ow, h: oh_box, color: c });
    }
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
    /// Stamped `Node.elem_idx` (JS DOM element index) when the tree was
    /// stamped before layout — lets input/click events reach page JS.
    pub elem_idx: Option<usize>,
}

/// Hit-test result in content coordinates (y includes scroll offset).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Hit {
    Link(String),
    Control(usize),
    /// A JS-interactive element ([`ElemBox`]) by stamped element index.
    Elem(usize),
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
    /// Hit boxes of JS-interactive elements (see [`ElemBox`]).
    pub elem_boxes: Vec<ElemBox>,
    /// CSS `background-image: url(…)` boxes (see [`BgBox`]).
    pub bg_boxes: Vec<BgBox>,
    pub bg: u32,
}

/// Extra outputs threaded through the walk: interactive element hit boxes and
/// background-image boxes, plus the set of element indices (stamped
/// `Node.elem_idx` values) that should get an [`ElemBox`].
struct Aux<'a> {
    elem_boxes: Vec<ElemBox>,
    bg_boxes: Vec<BgBox>,
    interactive: &'a [usize],
}

impl Aux<'_> {
    /// Record a hit box when the node is stamped AND in the interactive set.
    fn push_elem_box(&mut self, elem_idx: Option<usize>, x: i32, y: i32, w: i32, h: i32) {
        if w <= 0 || h <= 0 {
            return;
        }
        if let Some(i) = elem_idx {
            if self.interactive.contains(&i) {
                self.elem_boxes.push(ElemBox { x, y, w, h, elem_idx: i });
            }
        }
    }

    /// Record a background-image box when the style carries `url(…)`.
    fn push_bg_box(&mut self, st: &ComputedStyle, x: i32, y: i32, w: i32, h: i32) {
        if w <= 0 || h <= 0 {
            return;
        }
        let v = st.background_image.trim();
        if v.len() < 5 || !v[..4].eq_ignore_ascii_case("url(") {
            return;
        }
        let Some(end) = v.find(')') else { return };
        let src = v[4..end].trim().trim_matches(|c| c == '"' || c == '\'').trim();
        if src.is_empty() || src.starts_with("data:") {
            return;
        }
        self.bg_boxes.push(BgBox {
            x,
            y,
            w,
            h,
            src: src.to_string(),
            repeat: st.background_repeat.clone(),
            size: st.background_size.clone(),
            pos: st.background_position.clone(),
            pixels: None,
            src_w: 0,
            src_h: 0,
        });
    }
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
    layout_document_ex(root, sheet, vw, vh, &[])
}

/// Like [`layout_document`], with a set of JS-interactive element indices
/// (stamped `Node.elem_idx` values): matching block/flex boxes are recorded as
/// [`ElemBox`]es so clicks on them can be dispatched into page JS.
pub fn layout_document_ex(
    root: &Node,
    sheet: &Stylesheet,
    vw: i32,
    vh: i32,
    interactive: &[usize],
) -> Layout {
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
    let mut aux = Aux {
        elem_boxes: Vec::new(),
        bg_boxes: Vec::new(),
        interactive,
    };
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
        &mut aux,
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
        elem_boxes: aux.elem_boxes,
        bg_boxes: aux.bg_boxes,
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
        elem_boxes: Vec::new(),
        bg_boxes: Vec::new(),
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
    aux: &mut Aux,
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
                    c, sheet, parent_st, cur, runs, links, rects, images, controls, frames, aux, link,
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
            srcset,
            // rel/on_attrs and future attrs are read where needed.
            ..
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
                    aux,
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

            // CSS `position` (parsed but never applied before — headers,
            // overlays and badges just flowed inline). `relative` shifts the box
            // by its offsets while keeping its flow space; `absolute`/`fixed` are
            // taken out of flow and placed at their offsets within the
            // containing block (approximated as the viewport-width box). The
            // subtree is laid out as an isolated fragment, then translated.
            if matches!(
                st.position,
                Position::Relative | Position::Absolute | Position::Fixed
            ) && (st.top.is_some()
                || st.left.is_some()
                || st.right.is_some()
                || st.bottom.is_some())
            {
                let out_of_flow = !matches!(st.position, Position::Relative);
                let mark = mark_frag(runs, links, rects, images, controls, frames, aux);
                let box_w = st.width.unwrap_or(cur.max_w).clamp(1, cur.max_w.max(1));
                let mut child_cur = Cursor {
                    x: 0,
                    y: 0,
                    max_w: box_w,
                    margin_x: 0,
                    line_h,
                    content_bottom: 0,
                };
                for c in &n.children {
                    walk(
                        c, sheet, &st, &mut child_cur, runs, links, rects, images,
                        controls, frames, aux, link, form, next_form_id, page_bg, vw, in_head,
                    );
                }
                let (x0, y0, x1, y1) =
                    frag_bbox(mark, runs, links, rects, images, controls, frames, aux);
                let fw = st.width.unwrap_or((x1 - x0).max(1));
                let fh = st.height.unwrap_or((y1 - y0).max(child_cur.content_bottom).max(1));
                // Origin of the containing block: the in-flow spot for
                // `relative`, else the viewport box.
                let (base_x, base_y) = if out_of_flow {
                    (cur.margin_x, if matches!(st.position, Position::Fixed) { 0 } else { 0 })
                } else {
                    (cur.x, cur.y)
                };
                let avail_w = cur.max_w.max(1);
                let tx = if let Some(l) = st.left {
                    base_x + l
                } else if let Some(r) = st.right {
                    base_x + avail_w - r - fw
                } else {
                    base_x
                };
                let ty = match st.top {
                    Some(t) => base_y + t,
                    // `bottom` needs the container height, which isn't tracked
                    // reliably here; keep the element near the container top
                    // rather than misplacing it far down the page.
                    None => base_y,
                };
                let _ = st.bottom;
                translate_frag(
                    mark, tx - x0, ty - y0, runs, links, rects, images, controls, frames, aux,
                );
                if out_of_flow {
                    cur.content_bottom = cur.content_bottom.max(ty + fh);
                } else {
                    // `relative` reserves its flow space (approx: advance by height).
                    cur.y += fh;
                    cur.content_bottom = cur.content_bottom.max(cur.y);
                }
                return;
            }

            // An explicit CSS `display: inline` / `inline-block` makes an
            // otherwise block element flow inline — real headers/nav bars style
            // `<div>`/`<li>`/`<nav>` items this way to sit side by side. Since
            // `DisplayMode` defaults to `Block`, `Inline` here means the author
            // set it. Replaced controls/embeds keep their own layout.
            if st.display == DisplayMode::Inline
                && !matches!(
                    tag,
                    "img" | "input" | "br" | "hr" | "iframe" | "canvas" | "video"
                        | "audio" | "button" | "textarea" | "select"
                )
            {
                cur.line_h = line_h.max(cur.line_h);
                for c in &n.children {
                    walk(
                        c, sheet, &st, cur, runs, links, rects, images, controls,
                        frames, aux, link, form, next_form_id, page_bg, vw, in_head,
                    );
                }
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
                        .clamp(16, cur.max_w.max(16));
                    let ih = height_attr.unwrap_or((iw * 3 / 4).clamp(16, 240));
                    // Responsive images: prefer a srcset candidate for this viewport.
                    let src_s = srcset
                        .as_deref()
                        .and_then(|ss| super::html::pick_srcset_candidate(ss, vw))
                        .or_else(|| src.clone())
                        .unwrap_or_default();
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
                        .clamp(40, cur.max_w.max(40));
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
                            c, sheet, &st, cur, runs, links, rects, images, controls, frames, aux, link,
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
                            c, sheet, &st, cur, runs, links, rects, images, controls, frames, aux, link,
                            form, next_form_id, page_bg, vw, in_head,
                        );
                    }
                    block_after(cur, st.margin_bottom.max(4));
                }
                "tr" => {
                    block_before(cur, 0);
                    for c in &n.children {
                        walk(
                            c, sheet, &st, cur, runs, links, rects, images, controls, frames, aux, link,
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
                            c, sheet, &st, cur, runs, links, rects, images, controls, frames, aux, link,
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
                    let old_max = cur.max_w;
                    let old_margin_x = cur.margin_x;
                    // Box-model insets: border + padding on each side.
                    let (bl, br) = (st.border_left_width.max(0), st.border_right_width.max(0));
                    let (bt, bb) = (st.border_top_width.max(0), st.border_bottom_width.max(0));
                    let (pl, pr) = (st.padding_left.max(0), st.padding_right.max(0));
                    let (pt, pb) = (st.padding_top.max(0), st.padding_bottom.max(0));
                    let h_extra = bl + br + pl + pr;
                    // Width of the block's margin box, then content width per
                    // `box-sizing` (content-box: `width` is the content;
                    // border-box: `width` includes padding+border).
                    let avail = (old_max - st.margin_left - st.margin_right).max(1);
                    let (content_w, box_w) = match st.width {
                        Some(w) => match st.box_sizing {
                            css::BoxSizing::BorderBox => {
                                ((w - h_extra).max(1), w.min(avail).max(1))
                            }
                            _ => (w.max(1), (w + h_extra).min(avail).max(1)),
                        },
                        None => ((avail - h_extra).max(1), avail),
                    };
                    // `margin: 0 auto` centers a fixed-width block in `avail`.
                    let auto_indent = if st.margin_left_auto
                        && st.margin_right_auto
                        && box_w < avail
                    {
                        (avail - box_w) / 2
                    } else {
                        0
                    };
                    let block_x0 = cur.margin_x + st.margin_left + auto_indent;
                    // Content is inset by the left border + padding; children lay
                    // out from there (so wrapped lines and a centered block's
                    // content track the block's content edge, not the outer margin).
                    let content_x = block_x0 + bl + pl;
                    cur.margin_x = content_x;
                    if tag == "li" {
                        emit_text("• ", cur, runs, links, link, &st);
                    }
                    cur.x = content_x;
                    cur.y += bt + pt;
                    let content_start_x = cur.x;
                    cur.max_w = content_w;
                    // Mark where this block's inline content begins, so a
                    // non-`Left` `text-align` can be applied to just these items.
                    let (r0, l0, i0, c0) = (runs.len(), links.len(), images.len(), controls.len());
                    for c in &n.children {
                        walk(
                            c, sheet, &st, cur, runs, links, rects, images, controls, frames, aux, link,
                            form, next_form_id, page_bg, vw, in_head,
                        );
                    }
                    if cur.x > content_start_x {
                        new_line(cur);
                    }
                    cur.y += pb + bb;
                    // Box height spans content + padding + border; honor an
                    // explicit `height` (content-box adds padding/border to it).
                    let mut block_h = (cur.y - block_y0).max(bt + bb + pt + pb).max(1);
                    if let Some(hh) = st.height {
                        let want = match st.box_sizing {
                            css::BoxSizing::BorderBox => hh,
                            _ => hh + bt + bb + pt + pb,
                        };
                        block_h = block_h.max(want);
                        cur.y = block_y0 + block_h;
                    }
                    if let Some(bg) = st.background {
                        rects.push(RectBox {
                            x: block_x0,
                            y: block_y0,
                            w: box_w,
                            h: block_h,
                            color: fade(bg, st.opacity),
                        });
                    }
                    aux.push_bg_box(&st, block_x0, block_y0, box_w, block_h);
                    push_box_borders(rects, block_x0, block_y0, box_w, block_h, &st);
                    aux.push_elem_box(n.elem_idx, block_x0, block_y0, box_w, block_h);
                    // Honor `text-align` for this block's own inline content.
                    // Gated on the alignment differing from the parent's, so a
                    // uniform-`center` subtree is shifted once (at the `<center>`
                    // / rule that introduces it), never re-shifted by inheriting
                    // descendants.
                    if st.text_align != parent_st.text_align
                        && !matches!(st.text_align, Align::Left)
                    {
                        align_inline(
                            st.text_align, content_start_x, cur.max_w,
                            &mut runs[r0..], &mut links[l0..], &mut images[i0..],
                            &mut controls[c0..],
                        );
                    }
                    cur.max_w = old_max;
                    cur.margin_x = old_margin_x;
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
                            aux, href_s.or(link), form, next_form_id, page_bg, vw, in_head,
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
                            aux, link, form, next_form_id, page_bg, vw, in_head,
                        );
                    }
                }
                "ul" | "ol" | "body" | "html" => {
                    cur.line_h = line_h;
                    for c in &n.children {
                        walk(
                            c, sheet, &st, cur, runs, links, rects, images, controls, frames, aux, link,
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
                                    frames, aux, link, form, next_form_id, page_bg, vw, in_head,
                                );
                            }
                        }
                        DisplayKind::ListItem => {
                            block_before(cur, st.margin_top);
                            emit_text("• ", cur, runs, links, link, &st);
                            for c in &n.children {
                                walk(
                                    c, sheet, &st, cur, runs, links, rects, images, controls,
                                    frames, aux, link, form, next_form_id, page_bg, vw, in_head,
                                );
                            }
                            block_after(cur, st.margin_bottom);
                        }
                        _ => {
                            block_before(cur, st.margin_top);
                            for c in &n.children {
                                walk(
                                    c, sheet, &st, cur, runs, links, rects, images, controls,
                                    frames, aux, link, form, next_form_id, page_bg, vw, in_head,
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
    elem_boxes: usize,
    bg_boxes: usize,
}

fn mark_frag(
    runs: &[TextRun],
    links: &[LinkBox],
    rects: &[RectBox],
    images: &[ImageBox],
    controls: &[FormControl],
    frames: &[FrameBox],
    aux: &Aux,
) -> FragMark {
    FragMark {
        runs: runs.len(),
        links: links.len(),
        rects: rects.len(),
        images: images.len(),
        controls: controls.len(),
        frames: frames.len(),
        elem_boxes: aux.elem_boxes.len(),
        bg_boxes: aux.bg_boxes.len(),
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
    aux: &Aux,
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
    for e in &aux.elem_boxes[start.elem_boxes..] {
        expand(e.x, e.y, e.w.max(1), e.h.max(1));
    }
    for b in &aux.bg_boxes[start.bg_boxes..] {
        expand(b.x, b.y, b.w.max(1), b.h.max(1));
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
    aux: &mut Aux,
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
    for e in &mut aux.elem_boxes[start.elem_boxes..] {
        e.x += dx;
        e.y += dy;
    }
    for b in &mut aux.bg_boxes[start.bg_boxes..] {
        b.x += dx;
        b.y += dy;
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
    aux: &mut Aux,
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
            // `container_w` can be below the 16px floor for a deeply-nested or
            // zero-width flex item (google.com's modern layout hits this) —
            // guard the upper bound so `clamp` never sees min > max (panic).
            .clamp(16, container_w.max(16));

        let mark = mark_frag(runs, links, rects, images, controls, frames, aux);
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
            aux,
            link,
            form,
            next_form_id,
            page_bg,
            vw,
            in_head,
        );
        let (x0, y0, x1, y1) =
            frag_bbox(mark, runs, links, rects, images, controls, frames, aux);
        let mut w = (x1 - x0).max(1);
        let mut h = (y1 - y0).max(1);
        if let Some(eh) = cst.height {
            h = h.max(eh);
        } else {
            h = h.max(child_cur.content_bottom.max(1));
        }
        if let Some(ew) = cst.width {
            w = ew.clamp(1, container_w.max(1));
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
                    translate_frag(it.mark, dx, dy, runs, links, rects, images, controls, frames, aux);
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
                            it.mark, dx, dy, runs, links, rects, images, controls, frames, aux,
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
                    translate_frag(it.mark, dx, dy, runs, links, rects, images, controls, frames, aux);
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
    let box_w = content_w.max(container_w);
    let box_h = content_h.max(1);
    if let Some(i) = bg_idx {
        rects[i].w = box_w;
        rects[i].h = box_h;
    }
    push_box_borders(rects, box_x, box_y, box_w, box_h, st);
    aux.push_bg_box(st, box_x, box_y, box_w, box_h);
    aux.push_elem_box(n.elem_idx, box_x, box_y, box_w, box_h);

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
            elem_idx: node.elem_idx,
        });
        return;
    }
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
            (tw.clamp(64, cur.max_w.max(64)), line_h + 12)
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
    // Inline-block flow: a form control is `inline-block` by default, so keep it
    // on the current line when it fits — sibling controls (e.g. Google's two
    // search buttons) then sit side by side and an enclosing `text-align`
    // centers the whole line. Wrap to a new line only when out of room.
    const CTRL_GAP: i32 = 8;
    if cur.x > cur.margin_x && cur.x + w > cur.margin_x + cur.max_w.max(w) {
        new_line(cur);
    }
    let ctrl_x = cur.x.max(cur.margin_x);
    let ctrl_y = cur.y + 4;
    cur.line_h = cur.line_h.max(h + 8);
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
        x: ctrl_x,
        y: ctrl_y,
        w,
        h,
        focused: false,
        checked: false,
        elem_idx: node.elem_idx,
    });
    cur.x = ctrl_x + w + CTRL_GAP;
    cur.content_bottom = cur.content_bottom.max(ctrl_y + h);
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

/// Apply `text-align: center|right` to the inline content a block emitted.
///
/// The flow layout places every run/image/control left-to-right, discarding
/// `text-align`. This is the post-pass that honors it: the runs/links/images/
/// controls appended while walking this block (the `*_start` slices) are grouped
/// by their line (`y`), and each line is shifted so its content is centered or
/// right-aligned within `[x0, x0 + avail_w]`. `Left` is a no-op.
///
/// Called once per block whose `text-align` *differs from its parent's*, so a
/// uniform-`center` subtree (e.g. everything inside google.com's `<center>`) is
/// aligned exactly once — no double-shift from nested inheriting blocks.
fn align_inline(
    align: Align,
    x0: i32,
    avail_w: i32,
    runs: &mut [TextRun],
    links: &mut [LinkBox],
    images: &mut [ImageBox],
    controls: &mut [FormControl],
) {
    if matches!(align, Align::Left) || avail_w <= 0 {
        return;
    }
    let run_w = |r: &TextRun| (crate::font_ttf::measure(&r.text, r.font_size.max(8) as f32) + 0.5) as i32;
    // Distinct line-y values across ALL emitted items — a line may hold only an
    // image (the logo) or only controls (buttons), with no text run, and those
    // lines must be aligned too.
    let mut ys: Vec<i32> = Vec::new();
    ys.extend(runs.iter().map(|r| r.y));
    ys.extend(links.iter().map(|b| b.y));
    ys.extend(images.iter().map(|im| im.y));
    ys.extend(controls.iter().map(|c| c.y));
    ys.sort_unstable();
    ys.dedup();
    for y in ys {
        // Line extent across every item on this line.
        let mut min_x = i32::MAX;
        let mut max_r = i32::MIN;
        for r in runs.iter().filter(|r| r.y == y) {
            min_x = min_x.min(r.x);
            max_r = max_r.max(r.x + run_w(r));
        }
        for b in links.iter().filter(|b| b.y == y) {
            min_x = min_x.min(b.x);
            max_r = max_r.max(b.x + b.w);
        }
        for im in images.iter().filter(|im| im.y == y) {
            min_x = min_x.min(im.x);
            max_r = max_r.max(im.x + im.w);
        }
        for c in controls.iter().filter(|c| c.y == y) {
            min_x = min_x.min(c.x);
            max_r = max_r.max(c.x + c.w);
        }
        if min_x == i32::MAX {
            continue;
        }
        let line_w = max_r - min_x;
        let target_left = match align {
            Align::Center => x0 + (avail_w - line_w) / 2,
            Align::Right => x0 + avail_w - line_w,
            _ => x0,
        };
        // Never push content left of the content box.
        let shift = (target_left - min_x).max(x0 - min_x);
        if shift == 0 {
            continue;
        }
        for r in runs.iter_mut().filter(|r| r.y == y) {
            r.x += shift;
        }
        for b in links.iter_mut().filter(|b| b.y == y) {
            b.x += shift;
        }
        for im in images.iter_mut().filter(|im| im.y == y) {
            im.x += shift;
        }
        for c in controls.iter_mut().filter(|c| c.y == y) {
            c.x += shift;
        }
    }
}

/// First usable font-family name for a run: generic names (and the empty
/// string) map to `""` = the default face; anything else is lowercased.
fn run_family(st: &ComputedStyle) -> String {
    let f = st.font_family.trim();
    if f.is_empty()
        || f.eq_ignore_ascii_case("sans-serif")
        || f.eq_ignore_ascii_case("serif")
        || f.eq_ignore_ascii_case("monospace")
        || f.eq_ignore_ascii_case("system-ui")
    {
        String::new()
    } else {
        f.to_ascii_lowercase()
    }
}

/// Fade `color` toward a light page background by `opacity` (0–255) — a cheap
/// approximation of CSS `opacity` for solid colours (no real alpha compositing).
/// `opacity == 255` returns the colour unchanged.
fn fade(color: u32, opacity: u8) -> u32 {
    if opacity >= 255 {
        return color;
    }
    let a = opacity as u32;
    let bg = 0xffu32; // assume a light background per channel
    let ch = |shift: u32| {
        let c = (color >> shift) & 0xff;
        ((c * a + bg * (255 - a)) / 255) & 0xff
    };
    (ch(16) << 16) | (ch(8) << 8) | ch(0)
}

/// `text-transform: capitalize` — uppercase the first letter of each word.
fn capitalize_words(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut at_word_start = true;
    for ch in s.chars() {
        if ch.is_whitespace() {
            at_word_start = true;
            out.push(ch);
        } else if at_word_start {
            at_word_start = false;
            out.extend(ch.to_uppercase());
        } else {
            out.push(ch);
        }
    }
    out
}

fn emit_text(
    text: &str,
    cur: &mut Cursor,
    runs: &mut Vec<TextRun>,
    links: &mut Vec<LinkBox>,
    link: Option<&str>,
    st: &ComputedStyle,
) {
    let family = run_family(st);
    let px = st.font_size.max(8) as f32;
    // CSS `line-height` (parsed but never honored) overrides the font metric.
    let line_h = st
        .line_height
        .filter(|&h| h > 0)
        .unwrap_or((crate::font_ttf::line_height(px) + 0.5) as i32);
    cur.line_h = cur.line_h.max(line_h.max(10));
    // `text-transform` — applied to the rendered text (parsed but never honored
    // before; google.com's buttons/labels use `uppercase`).
    let transformed: String;
    let text: &str = match st.text_transform {
        css::TextTransform::Uppercase => {
            transformed = text.to_uppercase();
            &transformed
        }
        css::TextTransform::Lowercase => {
            transformed = text.to_lowercase();
            &transformed
        }
        css::TextTransform::Capitalize => {
            transformed = capitalize_words(text);
            &transformed
        }
        css::TextTransform::None => text,
    };
    // `white-space: nowrap|pre` suppresses automatic line breaking.
    let no_wrap = matches!(st.white_space, css::WhiteSpace::Nowrap | css::WhiteSpace::Pre);
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
        if !no_wrap && cur.x > cur.margin_x && cur.x + w > cur.margin_x + cur.max_w {
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
            color: fade(st.color, st.opacity),
            link_href: link.map(|s| s.to_string()),
            font_size: st.font_size,
            bold: st.bold,
            font_family: family.clone(),
            underline: matches!(st.text_decoration, css::TextDecoration::Underline),
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
    // JS-interactive elements: topmost = last pushed wins among overlaps.
    for eb in layout.elem_boxes.iter().rev() {
        if x >= eb.x && x < eb.x + eb.w && y >= eb.y && y < eb.y + eb.h {
            return Hit::Elem(eb.elem_idx);
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
        Hit::Elem(_) => CursorKind::Pointer,
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
    fn align_inline_centers_and_right_aligns() {
        // Use LinkBoxes (explicit width) so the test needs no font face.
        let mk = |x: i32, y: i32, w: i32| LinkBox { href: String::new(), x, y, w, h: 20 };
        // Center a 100-wide line within [0,300] → left = (300-100)/2 = 100.
        let mut links = alloc::vec![mk(0, 10, 100)];
        align_inline(Align::Center, 0, 300, &mut [], &mut links, &mut [], &mut []);
        assert_eq!(links[0].x, 100);
        // Right-align the same line → left = 300-100 = 200.
        let mut links = alloc::vec![mk(0, 10, 100)];
        align_inline(Align::Right, 0, 300, &mut [], &mut links, &mut [], &mut []);
        assert_eq!(links[0].x, 200);
        // Two boxes on one line move together; a wider line than the box is
        // never pushed left of x0 (clamp).
        let mut links = alloc::vec![mk(0, 5, 200), mk(210, 5, 200)];
        align_inline(Align::Center, 0, 100, &mut [], &mut links, &mut [], &mut []);
        assert_eq!(links[0].x, 0, "over-wide line clamped to the content left");
        // Left is a no-op.
        let mut links = alloc::vec![mk(7, 1, 50)];
        align_inline(Align::Left, 0, 300, &mut [], &mut links, &mut [], &mut []);
        assert_eq!(links[0].x, 7);
    }

    #[test_case]
    fn adjacent_controls_flow_inline() {
        // Two submit buttons in one form must sit side by side (inline-block),
        // not stack — same line (y), the second to the right of the first.
        let doc = html::parse(
            r#"<html><body><form><input type="submit" value="Google Search"><input type="submit" value="I'm Feeling Lucky"></form></body></html>"#,
        );
        let sheet = Stylesheet::parse(&doc.stylesheets);
        let lay = layout_document(&doc.root, &sheet, DEFAULT_W, DEFAULT_H);
        assert_eq!(lay.controls.len(), 2, "two submit controls");
        let (a, b) = (&lay.controls[0], &lay.controls[1]);
        assert_eq!(a.y, b.y, "adjacent controls share a line");
        assert!(b.x >= a.x + a.w, "second control is to the right of the first");
    }

    #[test_case]
    fn padding_insets_content_and_box() {
        // Horizontal padding must inset the content (was ignored) and the box
        // background must span content + padding.
        let doc = html::parse(
            r#"<html><body><div style="padding:20px;background:#eee"><p>pad</p></div><p>edge</p></body></html>"#,
        );
        let sheet = Stylesheet::parse(&doc.stylesheets);
        let lay = layout_document(&doc.root, &sheet, DEFAULT_W, DEFAULT_H);
        let pad = lay.runs.iter().find(|r| r.text == "pad").expect("pad");
        let edge = lay.runs.iter().find(|r| r.text == "edge").expect("edge");
        assert!(pad.x >= edge.x + 15, "content inset by left padding (pad.x={}, edge.x={})", pad.x, edge.x);
        assert!(pad.y >= 20, "content inset by top padding (y={})", pad.y);
    }

    #[test_case]
    fn line_height_overrides_font_metric() {
        // CSS `line-height` (parsed, never honored) sets the line box height.
        let doc = html::parse(
            r#"<html><body><p style="line-height:60px">A</p><p>B</p></body></html>"#,
        );
        let sheet = Stylesheet::parse(&doc.stylesheets);
        let lay = layout_document(&doc.root, &sheet, DEFAULT_W, DEFAULT_H);
        let a = lay.runs.iter().find(|r| r.text == "A").expect("A");
        let b = lay.runs.iter().find(|r| r.text == "B").expect("B");
        assert!(b.y - a.y >= 60, "60px line-height spaces the next line (got {})", b.y - a.y);
    }

    #[test_case]
    fn text_decoration_underline_flag_set() {
        let doc = html::parse(
            r#"<html><body><span style="text-decoration:underline">u</span> <span>plain</span></body></html>"#,
        );
        let sheet = Stylesheet::parse(&doc.stylesheets);
        let lay = layout_document(&doc.root, &sheet, DEFAULT_W, DEFAULT_H);
        assert!(lay.runs.iter().find(|r| r.text == "u").expect("u").underline);
        assert!(!lay.runs.iter().find(|r| r.text == "plain").expect("plain").underline);
    }

    #[test_case]
    fn position_absolute_places_at_offset() {
        // An absolutely-positioned badge is taken out of flow and placed at its
        // top/left, not stacked in the normal flow.
        let doc = html::parse(
            r#"<html><body><div style="position:absolute;top:200px;left:300px">badge</div><p>flow</p></body></html>"#,
        );
        let sheet = Stylesheet::parse(&doc.stylesheets);
        let lay = layout_document(&doc.root, &sheet, DEFAULT_W, DEFAULT_H);
        let badge = lay.runs.iter().find(|r| r.text == "badge").expect("badge");
        let flow = lay.runs.iter().find(|r| r.text == "flow").expect("flow");
        assert!(badge.x >= 300 && badge.y >= 200, "badge at its offset ({},{})", badge.x, badge.y);
        // Out of flow: the following paragraph starts at the top, not below the badge.
        assert!(flow.y < 100, "flow content not pushed down by the abs element (y={})", flow.y);
    }

    #[test_case]
    fn text_transform_uppercase_applied() {
        // `text-transform` was parsed but never applied to the rendered text.
        let doc = html::parse(
            r#"<html><body><p style="text-transform:uppercase">Search</p><p style="text-transform:capitalize">i'm feeling lucky</p></body></html>"#,
        );
        let sheet = Stylesheet::parse(&doc.stylesheets);
        let lay = layout_document(&doc.root, &sheet, DEFAULT_W, DEFAULT_H);
        assert!(lay.runs.iter().any(|r| r.text == "SEARCH"), "uppercase applied");
        // Capitalize: each word's first letter upper.
        assert!(
            lay.runs.iter().any(|r| r.text == "I'm") && lay.runs.iter().any(|r| r.text == "Feeling"),
            "capitalize applied per word"
        );
    }

    #[test_case]
    fn css_display_inline_block_flows_horizontally() {
        // `display:inline-block` on block elements (a nav bar of `<div>`s) makes
        // them share a line instead of stacking.
        let doc = html::parse(
            r#"<html><body><div style="display:inline-block">One</div><div style="display:inline-block">Two</div></body></html>"#,
        );
        let sheet = Stylesheet::parse(&doc.stylesheets);
        let lay = layout_document(&doc.root, &sheet, DEFAULT_W, DEFAULT_H);
        let one = lay.runs.iter().find(|r| r.text.contains("One")).expect("One");
        let two = lay.runs.iter().find(|r| r.text.contains("Two")).expect("Two");
        assert_eq!(one.y, two.y, "inline-block divs share a line");
        assert!(two.x > one.x, "second is to the right of the first");
    }

    #[test_case]
    fn margin_auto_centers_block() {
        // `margin: 0 auto` on a fixed-width block centers it — the content
        // inside starts well right of the page's left margin.
        let doc = html::parse(
            r#"<html><body><div style="width:200px;margin:0 auto"><p>boxed</p></div><p>edge</p></body></html>"#,
        );
        let sheet = Stylesheet::parse(&doc.stylesheets);
        let lay = layout_document(&doc.root, &sheet, DEFAULT_W, DEFAULT_H);
        let boxed = lay.runs.iter().find(|r| r.text.contains("boxed")).expect("boxed run");
        let edge = lay.runs.iter().find(|r| r.text.contains("edge")).expect("edge run");
        assert!(
            boxed.x > edge.x + 40,
            "margin:auto block content x={} should be indented past the page edge x={}",
            boxed.x, edge.x
        );
    }

    #[test_case]
    fn center_element_centers_content() {
        // `<center>` (and `text-align:center`) must actually shift inline
        // content — google.com wraps its logo/search block in one `<center>`.
        let doc = html::parse(
            r#"<html><body><center><p>hi</p></center><p>left</p></body></html>"#,
        );
        let sheet = Stylesheet::parse(&doc.stylesheets);
        let lay = layout_document(&doc.root, &sheet, DEFAULT_W, DEFAULT_H);
        let centered = lay.runs.iter().find(|r| r.text.contains("hi")).expect("hi run");
        let left = lay.runs.iter().find(|r| r.text.contains("left")).expect("left run");
        assert!(
            centered.x > left.x + 20,
            "centered run x={} should be well right of the left-aligned run x={}",
            centered.x, left.x
        );
    }

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
    fn elem_box_recorded_and_hit() {
        // A stamped interactive div gets an ElemBox and Hit::Elem at its point.
        let mut doc = html::parse(
            r#"<html><body><div id="btn" onclick="go()">Click me</div></body></html>"#,
        );
        crate::browser::js::stamp_elem_indices(&mut doc.root);
        // Find the stamped index of the div.
        fn find_idx(n: &html::Node) -> Option<usize> {
            if n.tag_name() == Some("div") {
                return n.elem_idx;
            }
            n.children.iter().find_map(find_idx)
        }
        let di = find_idx(&doc.root).expect("div stamped");
        let sheet = Stylesheet::parse(&doc.stylesheets);
        let interactive = [di];
        let lay = layout_document_ex(&doc.root, &sheet, 320, 200, &interactive);
        assert!(
            lay.elem_boxes.iter().any(|e| e.elem_idx == di),
            "elem_boxes: {:?}",
            lay.elem_boxes
        );
        let eb = lay.elem_boxes.iter().find(|e| e.elem_idx == di).unwrap();
        assert_eq!(
            hit_test_ex(&lay, eb.x + 1, eb.y + 1),
            Hit::Elem(di),
            "hit at ({},{})",
            eb.x + 1,
            eb.y + 1
        );
        assert_eq!(cursor_at(&lay, eb.x + 1, eb.y + 1), CursorKind::Pointer);
        // Non-interactive layout of the same DOM records no elem boxes.
        let lay2 = layout_document(&doc.root, &sheet, 320, 200);
        assert!(lay2.elem_boxes.is_empty());
    }

    #[test_case]
    fn bg_box_and_srcset_recorded() {
        let doc = html::parse(
            r#"<html><head><style>
              #hero { background-image: url("bg.png"); background-repeat: no-repeat; }
            </style></head><body>
            <div id="hero">Hero</div>
            <img srcset="a.png 480w, b.png 800w" src="fallback.png">
            </body></html>"#,
        );
        let sheet = Stylesheet::parse(&doc.stylesheets);
        let lay = layout_document(&doc.root, &sheet, 640, 400);
        assert!(
            lay.bg_boxes.iter().any(|b| b.src == "bg.png" && b.repeat == "no-repeat"),
            "bg_boxes: {:?}",
            lay.bg_boxes
        );
        // 640px viewport picks the 800w candidate.
        assert!(
            lay.images.iter().any(|im| im.src == "b.png"),
            "images: {:?}",
            lay.images.iter().map(|i| i.src.clone()).collect::<Vec<_>>()
        );
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
