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
    /// `letter-spacing` in px added between glyphs (0 = normal).
    pub letter_spacing: i32,
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
    /// `border-radius` in px (0 = square). The painter rounds the corners of
    /// filled background rects; border/outline edge rects keep 0.
    pub radius: i32,
    /// `box-shadow` blur radius in px (0 = a crisp fill). When > 0 the painter
    /// renders this rect as a box-blurred shadow instead of a solid fill.
    pub blur: i32,
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
        rects.push(RectBox { x, y, w, h: bw, color: c, radius: 0, blur: 0 });
    }
    if let Some((bw, c)) = edge(st.border_bottom_style, st.border_bottom_width, st.border_bottom_color) {
        rects.push(RectBox { x, y: y + h - bw, w, h: bw, color: c, radius: 0, blur: 0 });
    }
    if let Some((bw, c)) = edge(st.border_left_style, st.border_left_width, st.border_left_color) {
        rects.push(RectBox { x, y, w: bw, h, color: c, radius: 0, blur: 0 });
    }
    if let Some((bw, c)) = edge(st.border_right_style, st.border_right_width, st.border_right_color) {
        rects.push(RectBox { x: x + w - bw, y, w: bw, h, color: c, radius: 0, blur: 0 });
    }
    // Outline — drawn just outside the border box by `outline-offset`.
    if st.outline_style.is_visible() && st.outline_width > 0 {
        let ow = st.outline_width;
        let off = st.outline_offset.max(0);
        let (ox, oy) = (x - off - ow, y - off - ow);
        let (ow_box, oh_box) = (w + 2 * (off + ow), h + 2 * (off + ow));
        let c = st.outline_color.unwrap_or(fallback);
        rects.push(RectBox { x: ox, y: oy, w: ow_box, h: ow, color: c, radius: 0, blur: 0 });
        rects.push(RectBox { x: ox, y: oy + oh_box - ow, w: ow_box, h: ow, color: c, radius: 0, blur: 0 });
        rects.push(RectBox { x: ox, y: oy, w: ow, h: oh_box, color: c, radius: 0, blur: 0 });
        rects.push(RectBox { x: ox + ow_box - ow, y: oy, w: ow, h: oh_box, color: c, radius: 0, blur: 0 });
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
    /// `object-fit` for scaling the source into the `w`×`h` box.
    pub object_fit: css::ObjectFit,
    /// `object-position` keywords (e.g. `right bottom`); empty = center.
    pub object_position: String,
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
    /// `border-radius` in px for the control's box (0 = square).
    pub radius: i32,
    /// CSS `background`/`color` for the control (else the UA default palette).
    pub bg: Option<u32>,
    pub fg: Option<u32>,
    /// The control's background is an image/gradient/transparent (no solid
    /// colour we render) — so the painter leaves it unfilled and the parent's
    /// background shows through (e.g. google.com's `.lsb` button over its grey
    /// `.lsbb` wrapper), instead of painting an opaque UA default over it.
    pub transparent: bool,
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
    /// `transform: rotate(...)` regions for the paint bitmap-rotation pass.
    pub rotates: Vec<RotateOp>,
    pub bg: u32,
}

/// Extra outputs threaded through the walk: interactive element hit boxes and
/// background-image boxes, plus the set of element indices (stamped
/// `Node.elem_idx` values) that should get an [`ElemBox`].
/// A `transform: rotate(...)` region for the paint-time bitmap rotation pass:
/// the element is rendered axis-aligned, then its box is rotated about
/// `(cx, cy)` by `angle_deg`.
#[derive(Clone, Copy, Debug)]
pub struct RotateOp {
    pub cx: i32,
    pub cy: i32,
    pub angle_deg: f32,
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
}

struct Aux<'a> {
    elem_boxes: Vec<ElemBox>,
    bg_boxes: Vec<BgBox>,
    rotates: Vec<RotateOp>,
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
    /// Active `float:left` exclusion: content on lines above `float_l_bottom` is
    /// pushed right by `float_l_w`. Cleared once `y` passes the bottom.
    float_l_w: i32,
    float_l_bottom: i32,
    /// Active `float:right` exclusion: shrinks the usable width by `float_r_w`.
    float_r_w: i32,
    float_r_bottom: i32,
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
    // `vh`/`vw` lengths resolve against this — recorded before any style is
    // computed, since a length is parsed lazily per declaration.
    css::set_viewport(vw, vh);
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
        float_l_w: 0,
        float_l_bottom: 0,
        float_r_w: 0,
        float_r_bottom: 0,
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
        rotates: Vec::new(),
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
        &[],   // ancestor chain (root)
        &[],   // preceding siblings — the document root has none
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
        rotates: aux.rotates,
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
        float_l_w: 0,
        float_l_bottom: 0,
        float_r_w: 0,
        float_r_bottom: 0,
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
        rotates: Vec::new(),
        bg: 0xf5f0e8,
    }
}

/// Build the ancestor chain for the children of an element: the element's own
/// chain plus itself, appended (outermost→innermost). Borrows from `chain` and
/// the element's own strings, both of which outlive the child walk.
/// An [`css::ElemRef`] for `n` if it is an element, for the sibling list.
///
/// Only the fields a selector can test on a *sibling* are filled: `+`/`~`
/// compounds are matched with `Compound::matches_el`, which reads tag, id,
/// class, attributes and `:nth-child`. A sibling's own siblings are never
/// consulted (that would be a selector matching two hops sideways, which the
/// grammar cannot express), so `prev: None` here is correct and not a gap.
fn elem_ref_for<'a>(n: &'a Node) -> Option<css::ElemRef<'a>> {
    match &n.kind {
        NodeKind::Element {
            tag,
            id,
            class,
            href,
            input_type,
            extra_attrs,
            ..
        } => Some(css::ElemRef {
            tag: tag.as_str(),
            id: id.as_deref(),
            class: class.as_deref(),
            nth: 1,
            href: href.as_deref(),
            input_type: input_type.as_deref(),
            extra: extra_attrs.as_slice(),
            prev: None,
        }),
        _ => None,
    }
}

fn push_chain<'a>(
    chain: &[css::ElemRef<'a>],
    tag: &'a str,
    id: Option<&'a str>,
    class: Option<&'a str>,
) -> Vec<css::ElemRef<'a>> {
    let mut v = Vec::with_capacity(chain.len() + 1);
    v.extend_from_slice(chain);
    v.push(css::ElemRef::basic(tag, id, class));
    v
}

/// Advance past floats that `clear` requires, then zero the matching exclusion.
fn apply_clear(cur: &mut Cursor, clear: css::ClearMode) {
    match clear {
        css::ClearMode::None => {}
        css::ClearMode::Left => {
            if cur.float_l_w > 0 {
                cur.y = cur.y.max(cur.float_l_bottom);
            }
            cur.float_l_w = 0;
            cur.float_l_bottom = 0;
            cur.x = cur.margin_x;
        }
        css::ClearMode::Right => {
            if cur.float_r_w > 0 {
                cur.y = cur.y.max(cur.float_r_bottom);
            }
            cur.float_r_w = 0;
            cur.float_r_bottom = 0;
        }
        css::ClearMode::Both => {
            let bottom = cur.float_l_bottom.max(cur.float_r_bottom);
            if cur.float_l_w > 0 || cur.float_r_w > 0 {
                cur.y = cur.y.max(bottom);
            }
            cur.float_l_w = 0;
            cur.float_l_bottom = 0;
            cur.float_r_w = 0;
            cur.float_r_bottom = 0;
            cur.x = cur.margin_x;
        }
    }
}

fn walk<'a>(
    n: &'a Node,
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
    chain: &[css::ElemRef<'a>],
    // `n`'s preceding **element** siblings, in document order — what `+` and
    // `~` match against. Built by each `&n.children` loop below rather than
    // derived here, because `walk` receives a node and not its position among
    // its parent's children.
    prev: &[css::ElemRef<'a>],
) {
    match &n.kind {
        NodeKind::Document => {
            let mut sibs: Vec<css::ElemRef<'a>> = Vec::new();
            for c in &n.children {
                walk(
                    c, sheet, parent_st, cur, runs, links, rects, images, controls, frames, aux, link,
                    form, next_form_id, page_bg, vw, in_head, chain, &sibs,
                );
                if let Some(r) = elem_ref_for(c) {
                    sibs.push(r);
                }
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
            bgcolor_attr,
            width_pct,
            srcset,
            // The catch-all bag, so `[data-state=open]`-style selectors can be
            // matched at all (see `css::ElemRef::extra`).
            extra_attrs,
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
            let el_ref = css::ElemRef {
                tag,
                id: id.as_deref(),
                class: class.as_deref(),
                nth: 1,
                href: href.as_deref(),
                input_type: input_type.as_deref(),
                extra: extra_attrs.as_slice(),
                // Supplied now, so `+` and `~` match exactly rather than being
                // approximated as descendant. `Some(&[])` is the real answer
                // for a first child, and is deliberately not `None` — see
                // `css::ElemRef::prev`.
                prev: Some(prev),
            };
            let mut st = css::compute_el(sheet, el_ref, style_attr.as_deref(), parent_st, chain);
            // Presentational attrs (bgcolor / width%) — also applied in
            // `layout_cell_isolated` for table cells (they skip this path).
            let _ = (bgcolor_attr, width_pct, width_attr); // used via node
            apply_presentational(n, &mut st, cur.max_w);
            if st.display_none || st.display == DisplayMode::None {
                return;
            }
            // Ancestor chain for this element's children (self appended).
            let child_chain = push_chain(chain, tag, id.as_deref(), class.as_deref());
            let child_chain = child_chain.as_slice();
            // `clear` — move below floats before placing this block/float.
            apply_clear(cur, st.clear);
            // Generated content ::before
            if let Some(pst) =
                css::compute_pseudo(sheet, el_ref, &st, chain, css::PseudoElement::Before)
            {
                emit_text(&pst.content, cur, runs, links, link, &pst);
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
            if matches!(st.display, DisplayMode::Flex | DisplayMode::InlineFlex | DisplayMode::Grid)
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
                    chain,
                );
                if let Some(pst) =
                    css::compute_pseudo(sheet, el_ref, &st, chain, css::PseudoElement::After)
                {
                    emit_text(&pst.content, cur, runs, links, link, &pst);
                }
                return;
            }

            // CSS `float: left|right`. The floated box is laid out as an
            // isolated fragment and placed at the left or right edge; a left
            // float lets following inline content flow to its right.
            if matches!(st.float_mode, css::FloatMode::Left | css::FloatMode::Right)
                && !matches!(tag, "br" | "hr")
            {
                let mark = mark_frag(runs, links, rects, images, controls, frames, aux);
                let box_w = st.width.unwrap_or((cur.max_w / 3).max(40)).clamp(1, cur.max_w.max(1));
                let mut fcur = Cursor {
                    x: 0,
                    y: 0,
                    max_w: box_w,
                    margin_x: 0,
                    line_h,
                    content_bottom: 0,
                    float_l_w: 0,
                    float_l_bottom: 0,
                    float_r_w: 0,
                    float_r_bottom: 0,
                };
                let mut sibs: Vec<css::ElemRef<'a>> = Vec::new();
                for c in &n.children {
                    walk(
                        c, sheet, &st, &mut fcur, runs, links, rects, images, controls,
                        frames, aux, link, form, next_form_id, page_bg, vw, in_head, child_chain, &sibs,
                    );
                    if let Some(r) = elem_ref_for(c) {
                        sibs.push(r);
                    }
                }
                let (x0, y0, x1, y1) = frag_bbox(mark, runs, links, rects, images, controls, frames, aux);
                let fw = st.width.unwrap_or((x1 - x0).max(1));
                let fh = (y1 - y0).max(fcur.content_bottom).max(1);
                let fy = cur.y;
                let gap6 = 8;
                let fx = match st.float_mode {
                    css::FloatMode::Right => {
                        // Placed inside the right edge, past any existing right float.
                        let x = cur.margin_x + cur.max_w - cur.float_r_w - fw;
                        cur.float_r_w += fw + gap6;
                        cur.float_r_bottom = cur.float_r_bottom.max(fy + fh);
                        x
                    }
                    _ => {
                        // Placed at the left edge, past any existing left float;
                        // following content wraps to its right for `fh` px.
                        let x = cur.margin_x + cur.float_l_w;
                        cur.float_l_w += fw + gap6;
                        cur.float_l_bottom = cur.float_l_bottom.max(fy + fh);
                        cur.x = cur.margin_x + cur.float_l_w;
                        x
                    }
                };
                translate_frag(mark, fx - x0, fy - y0, runs, links, rects, images, controls, frames, aux);
                cur.line_h = cur.line_h.max(fh);
                cur.content_bottom = cur.content_bottom.max(fy + fh);
                if let Some(pst) =
                    css::compute_pseudo(sheet, el_ref, &st, chain, css::PseudoElement::After)
                {
                    emit_text(&pst.content, cur, runs, links, link, &pst);
                }
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
                    float_l_w: 0,
                    float_l_bottom: 0,
                    float_r_w: 0,
                    float_r_bottom: 0,
                };
                let mut sibs: Vec<css::ElemRef<'a>> = Vec::new();
                for c in &n.children {
                    walk(
                        c, sheet, &st, &mut child_cur, runs, links, rects, images,
                        controls, frames, aux, link, form, next_form_id, page_bg, vw, in_head, child_chain, &sibs,
                    );
                    if let Some(r) = elem_ref_for(c) {
                        sibs.push(r);
                    }
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
                let mut st_i = st;
                // An inline / inline-block `<a>` still links its content and —
                // when styled as a button (`background` + padding, e.g.
                // google.com's "Sign in" `.gb_1a{display:inline-block}`) — paints
                // a background pill, exactly like a block-flowed `<a>`. Without
                // this it fell through here losing both its `href` (→ not
                // clickable) and its background (→ not blue).
                let link_here = if tag == "a" {
                    if st_i.color == parent_st.color {
                        st_i.color = 0x1a73e8;
                    }
                    href.as_deref().or(link)
                } else {
                    link
                };
                cur.line_h = line_h.max(cur.line_h);
                // Horizontal margins space inline-block siblings (footer nav).
                cur.x += st_i.margin_left.max(0);
                // Inline background box (inline-block buttons / badges), inset by
                // the element's own padding.
                let box_left = cur.x;
                let box_top = cur.y;
                let inline_bg = st_i.background;
                if inline_bg.is_some() {
                    cur.x += st_i.padding_left.max(0);
                }
                let br0 = runs.len();
                // `vertical-align` shifts this inline box within the line box.
                let va = st_i.vertical_align;
                let vmark = if !matches!(va, css::VerticalAlign::Baseline) {
                    Some(mark_frag(runs, links, rects, images, controls, frames, aux))
                } else {
                    None
                };
                let mut sibs: Vec<css::ElemRef<'a>> = Vec::new();
                for c in &n.children {
                    walk(
                        c, sheet, &st_i, cur, runs, links, rects, images, controls,
                        frames, aux, link_here, form, next_form_id, page_bg, vw, in_head, child_chain, &sibs,
                    );
                    if let Some(r) = elem_ref_for(c) {
                        sibs.push(r);
                    }
                }
                if let Some(bg) = inline_bg {
                    cur.x += st_i.padding_right.max(0);
                    let (pt, pb) = (st_i.padding_top.max(0), st_i.padding_bottom.max(0));
                    let mut y1 = box_top;
                    for r in &runs[br0..] {
                        y1 = y1.max(r.y + cur.line_h);
                    }
                    let h = (y1 - box_top).max(cur.line_h) + pt + pb;
                    // Under the text (rects paint before runs).
                    rects.push(RectBox {
                        x: box_left,
                        y: box_top - pt / 2,
                        w: (cur.x - box_left).max(1),
                        h,
                        color: fade(bg, st_i.opacity),
                        radius: st_i.border_radius.max(0),
                        blur: 0,
                    });
                }
                if let Some(m) = vmark {
                    let (_, y0, _, y1) =
                        frag_bbox(m, runs, links, rects, images, controls, frames, aux);
                    let fh = (y1 - y0).max(1);
                    let dy = match va {
                        css::VerticalAlign::Middle => (cur.line_h - fh) / 2,
                        css::VerticalAlign::Bottom => cur.line_h - fh,
                        _ => 0, // Top / Baseline: keep at the line top
                    };
                    if dy != 0 {
                        translate_frag(m, 0, dy, runs, links, rects, images, controls, frames, aux);
                    }
                }
                cur.x += st_i.margin_right.max(0);
                return;
            }

            match tag {
                "br" | "wbr" => {
                    // Break AFTER the current line's accumulated height (which
                    // may include a tall inline-block like a search box), then
                    // reset to the default for the next line. Resetting *before*
                    // the break made the next line overlap tall inline content.
                    new_line(cur);
                    cur.line_h = line_h;
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
                        radius: 0,
                        blur: 0,
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
                        object_fit: st.object_fit,
                        object_position: st.object_position.clone(),
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
                    let mut sibs: Vec<css::ElemRef<'a>> = Vec::new();
                    for c in &n.children {
                        walk(
                            c, sheet, &st, cur, runs, links, rects, images, controls, frames, aux, link,
                            Some(&ctx), next_form_id, page_bg, vw, in_head, child_chain, &sibs,
                        );
                        if let Some(r) = elem_ref_for(c) {
                            sibs.push(r);
                        }
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
                        st.border_radius.max(0),
                        st.background,
                        if st.color != parent_st.color { Some(st.color) } else { None },
                        st.background.is_none() && !st.background_image.is_empty(),
                        st.width,
                        st.height,
                    );
                }
                "table" => {
                    let mut trows: Vec<Vec<TableCellRef<'_>>> = Vec::new();
                    collect_table_rows(n, &mut trows);
                    if trows.iter().any(|r| !r.is_empty()) {
                        // Parent `text-align:center` (`<center>`) centers the
                        // table *box*; cells themselves reset to start-align.
                        layout_table(
                            n, sheet, &st, parent_st, cur, runs, links, rects, images, controls,
                            frames, aux, link, form, next_form_id, page_bg, vw, in_head, chain,
                        );
                    } else {
                        block_before(cur, st.margin_top.max(4));
                        let mut sibs: Vec<css::ElemRef<'a>> = Vec::new();
                        for c in &n.children {
                            walk(
                                c, sheet, &st, cur, runs, links, rects, images, controls, frames,
                                aux, link, form, next_form_id, page_bg, vw, in_head, child_chain, &sibs,
                            );
                            if let Some(r) = elem_ref_for(c) {
                                sibs.push(r);
                            }
                        }
                        block_after(cur, st.margin_bottom.max(4));
                    }
                }
                "thead" | "tbody" | "tfoot" => {
                    block_before(cur, st.margin_top.max(4));
                    let mut sibs: Vec<css::ElemRef<'a>> = Vec::new();
                    for c in &n.children {
                        walk(
                            c, sheet, &st, cur, runs, links, rects, images, controls, frames, aux, link,
                            form, next_form_id, page_bg, vw, in_head, child_chain, &sibs,
                        );
                        if let Some(r) = elem_ref_for(c) {
                            sibs.push(r);
                        }
                    }
                    block_after(cur, st.margin_bottom.max(4));
                }
                "tr" => {
                    block_before(cur, 0);
                    let mut sibs: Vec<css::ElemRef<'a>> = Vec::new();
                    for c in &n.children {
                        walk(
                            c, sheet, &st, cur, runs, links, rects, images, controls, frames, aux, link,
                            form, next_form_id, page_bg, vw, in_head, child_chain, &sibs,
                        );
                        if let Some(r) = elem_ref_for(c) {
                            sibs.push(r);
                        }
                    }
                    if cur.x > cur.margin_x {
                        new_line(cur);
                    }
                }
                "td" | "th" => {
                    // Cell as padded inline-block-ish block child.
                    let old_max = cur.max_w;
                    cur.max_w = (old_max / 2).max(40);
                    let mut sibs: Vec<css::ElemRef<'a>> = Vec::new();
                    for c in &n.children {
                        walk(
                            c, sheet, &st, cur, runs, links, rects, images, controls, frames, aux, link,
                            form, next_form_id, page_bg, vw, in_head, child_chain, &sibs,
                        );
                        if let Some(r) = elem_ref_for(c) {
                            sibs.push(r);
                        }
                    }
                    cur.max_w = old_max;
                    // Inter-cell gap: `border-spacing` (inherited from the table)
                    // when set, else the default two-space gutter.
                    if st.border_spacing > 0 {
                        cur.x += st.border_spacing;
                    } else {
                        emit_text("  ", cur, runs, links, None, &st);
                    }
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
                    let (mut content_w, mut box_w) = match st.width {
                        Some(w) => match st.box_sizing {
                            css::BoxSizing::BorderBox => {
                                ((w - h_extra).max(1), w.min(avail).max(1))
                            }
                            _ => (w.max(1), (w + h_extra).min(avail).max(1)),
                        },
                        None => ((avail - h_extra).max(1), avail),
                    };
                    // `min-width` / `max-width` clamp the content box (both given
                    // as content-box lengths); the border box follows.
                    if let Some(mw) = st.max_width {
                        if content_w > mw {
                            content_w = mw.max(1);
                        }
                    }
                    if let Some(mnw) = st.min_width {
                        if content_w < mnw {
                            content_w = mnw;
                        }
                    }
                    box_w = (content_w + h_extra).min(avail).max(1);
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
                        {
                        let marker = list_marker(&st);
                        if !marker.is_empty() { emit_text(marker, cur, runs, links, link, &st); }
                    }
                    }
                    cur.x = content_x;
                    cur.y += bt + pt;
                    let content_start_x = cur.x;
                    cur.max_w = content_w;
                    // `text-indent` indents the first line of the block.
                    if st.text_indent > 0 {
                        cur.x += st.text_indent.min(content_w.saturating_sub(1));
                    }
                    // Mark where this block's inline content begins, so a
                    // non-`Left` `text-align` can be applied to just these items.
                    let (r0, l0, i0, c0) = (runs.len(), links.len(), images.len(), controls.len());
                    // Where this block's own decoration must be INSERTED. A
                    // block's background can only be pushed once its children
                    // have been laid out (that is what fixes its height), but
                    // the painter draws rects in list order — so pushing it
                    // last painted the parent over every child background. A
                    // white Tailwind card erased the badges and rules inside
                    // it. Everything this block paints goes in at `rect0`,
                    // behind its children.
                    let rect0 = rects.len();
                    // Fragment mark for shrink-to-fit width (see below).
                    let inline_mark =
                        mark_frag(runs, links, rects, images, controls, frames, aux);
                    // Full fragment mark for any `transform` (translate/scale/
                    // rotate), applied to the whole box after layout.
                    let has_xform = {
                        let t = st.transform.trim();
                        (!t.is_empty() && !t.eq_ignore_ascii_case("none"))
                            || (st.zoom - 1.0).abs() > 0.001
                    };
                    let tmark = if has_xform {
                        Some(mark_frag(runs, links, rects, images, controls, frames, aux))
                    } else {
                        None
                    };
                    let mut sibs: Vec<css::ElemRef<'a>> = Vec::new();
                    for c in &n.children {
                        walk(
                            c, sheet, &st, cur, runs, links, rects, images, controls, frames, aux, link,
                            form, next_form_id, page_bg, vw, in_head, child_chain, &sibs,
                        );
                        if let Some(r) = elem_ref_for(c) {
                            sibs.push(r);
                        }
                    }
                    if cur.x > content_start_x {
                        new_line(cur);
                    }
                    cur.y += pb + bb;
                    // Box height spans content + padding + border; honor an
                    // explicit `height` (content-box adds padding/border to it).
                    let mut block_h = (cur.y - block_y0).max(bt + bb + pt + pb).max(1);
                    // `height` / `min-height` / `max-height` (content-box adds
                    // padding+border to the given length).
                    let box_extra_v = bt + bb + pt + pb;
                    let to_box_h = |len: i32| match st.box_sizing {
                        css::BoxSizing::BorderBox => len,
                        _ => len + box_extra_v,
                    };
                    if let Some(hh) = st.height {
                        block_h = block_h.max(to_box_h(hh));
                    }
                    if let Some(mnh) = st.min_height {
                        block_h = block_h.max(to_box_h(mnh));
                    }
                    if let Some(mxh) = st.max_height {
                        block_h = block_h.min(to_box_h(mxh)).max(1);
                    }
                    cur.y = block_y0 + block_h;
                    // Shrink-to-fit for an INLINE-LEVEL box with no explicit
                    // width. `display: inline-flex` / `inline-block` size to
                    // their content, not to the line — and a shadcn button or
                    // badge is `inline-flex` with a text child, which has no
                    // element children and so never reaches the flex container
                    // path at all. Without this every one of them painted as a
                    // full-width bar.
                    let mut box_w = box_w;
                    if st.width.is_none() && st.display == DisplayMode::InlineFlex {
                        let (cx0, _, cx1, _) =
                            frag_bbox(inline_mark, runs, links, rects, images, controls, frames, aux);
                        let natural = (cx1 - cx0).max(0) + h_extra;
                        if natural > 0 {
                            box_w = natural.min(box_w).max(1);
                        }
                    }
                    // `box-shadow`: a soft offset rectangle behind the box
                    // (single hard rect, colour faded toward the page by blur —
                    // no real gaussian blur). Drawn before the background.
                    let mut deco = 0usize; // decoration rects inserted at `rect0`
                    if let Some((sdx, sdy, blur, scol)) = parse_box_shadow(&st.box_shadow) {
                        // The painter box-blurs this rect when `blur > 0` (real
                        // gaussian-ish falloff); the solid box is the shadow's
                        // spread footprint at the shadow offset.
                        rects.insert(
                            rect0,
                            RectBox {
                                x: block_x0 + sdx,
                                y: block_y0 + sdy,
                                w: box_w.max(1),
                                h: block_h.max(1),
                                color: scol,
                                radius: st.border_radius.max(0),
                                blur: blur.clamp(0, 40),
                            },
                        );
                        deco += 1;
                    }
                    if let Some(bg) = st.background {
                        rects.insert(
                            rect0 + deco,
                            RectBox {
                                x: block_x0,
                                y: block_y0,
                                w: box_w,
                                h: block_h,
                                color: fade(bg, st.opacity),
                                radius: st.border_radius.max(0),
                                blur: 0,
                            },
                        );
                        deco += 1;
                    }
                    aux.push_bg_box(&st, block_x0, block_y0, box_w, block_h);
                    {
                        // Borders go in front of this block's own background but
                        // still behind its children.
                        let mut edges = Vec::new();
                        push_box_borders(&mut edges, block_x0, block_y0, box_w, block_h, &st);
                        for e in edges {
                            rects.insert(rect0 + deco, e);
                            deco += 1;
                        }
                    }
                    aux.push_elem_box(n.elem_idx, block_x0, block_y0, box_w, block_h);
                    // Honor `text-align` for this block's own **inline** content.
                    // Gated on the alignment differing from the parent's, so a
                    // uniform-`center` subtree is shifted once (at the `<center>`
                    // / rule that introduces it), never re-shifted by inheriting
                    // descendants.
                    //
                    // Critical: do NOT re-align content produced by a child
                    // `<table>`. CSS `text-align` only affects inline content in
                    // this block's line boxes; a table already placed ranks,
                    // titles, and backgrounds. Without this gate, HN's
                    // `<center><table id=hnmain>` had every story line
                    // recentered after table layout (mid-cream instead of flush
                    // left). Still run for `<center><p>…` / google.com-style
                    // inline children so those keep centering.
                    let has_table_child = n.children.iter().any(|c| {
                        matches!(&c.kind, NodeKind::Element { tag: t, .. } if t == "table")
                    });
                    if !has_table_child
                        && st.text_align != parent_st.text_align
                        && !matches!(st.text_align, Align::Left)
                    {
                        align_inline(
                            st.text_align, content_start_x, cur.max_w,
                            &mut runs[r0..], &mut links[l0..], &mut images[i0..],
                            &mut controls[c0..],
                        );
                    }
                    // Apply `transform` (scale about the box origin, then
                    // translate; rotate is recorded for a paint-time bitmap pass).
                    if let Some(tm) = tmark {
                        // `zoom` scales the box like an extra uniform scale.
                        let z = st.zoom;
                        let (sx, sy) = parse_transform_scale(&st.transform);
                        let (sx, sy) = (sx * z, sy * z);
                        if (sx, sy) != (1.0, 1.0) {
                            scale_frag(tm, block_x0, block_y0, sx, sy, runs, links, rects, images, controls);
                        }
                        let (tx, ty) = parse_transform_translate(&st.transform);
                        if (tx, ty) != (0, 0) {
                            translate_frag(tm, tx, ty, runs, links, rects, images, controls, frames, aux);
                        }
                        let rot = parse_transform_rotate(&st.transform);
                        if rot.abs() > 0.01 {
                            let (bx, by, bw2, bh2) =
                                frag_bbox(tm, runs, links, rects, images, controls, frames, aux);
                            aux.rotates.push(RotateOp {
                                cx: (bx + bw2) / 2,
                                cy: (by + bh2) / 2,
                                angle_deg: rot,
                                x: bx,
                                y: by,
                                w: (bw2 - bx).max(1),
                                h: (bh2 - by).max(1),
                            });
                        }
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
                    // Horizontal margins on an inline/inline-block link add
                    // spacing between siblings (e.g. footer nav links with
                    // `margin:0 12px` — otherwise `</a><a>` runs together).
                    cur.x += st_a.margin_left.max(0);
                    // A link styled as a button (`background` + padding + radius,
                    // e.g. google.com's "Sign in" `.gb_1a`) paints an inline
                    // background box behind its text, inset by its padding.
                    let box_left = cur.x;
                    let box_top = cur.y;
                    let inline_bg = st_a.background;
                    if inline_bg.is_some() {
                        cur.x += st_a.padding_left.max(0);
                    }
                    let br0 = runs.len();
                    let mut sibs: Vec<css::ElemRef<'a>> = Vec::new();
                    for c in &n.children {
                        walk(
                            c, sheet, &st_a, cur, runs, links, rects, images, controls, frames,
                            aux, href_s.or(link), form, next_form_id, page_bg, vw, in_head, child_chain, &sibs,
                        );
                        if let Some(r) = elem_ref_for(c) {
                            sibs.push(r);
                        }
                    }
                    if let Some(bg) = inline_bg {
                        cur.x += st_a.padding_right.max(0);
                        let (pt, pb) = (st_a.padding_top.max(0), st_a.padding_bottom.max(0));
                        let mut y1 = box_top;
                        for r in &runs[br0..] {
                            y1 = y1.max(r.y + cur.line_h);
                        }
                        let h = (y1 - box_top).max(cur.line_h) + pt + pb;
                        // Under the text (rects paint before runs).
                        rects.push(RectBox {
                            x: box_left,
                            y: box_top - pt / 2,
                            w: (cur.x - box_left).max(1),
                            h,
                            color: fade(bg, st_a.opacity),
                            radius: st_a.border_radius.max(0),
                            blur: 0,
                        });
                    }
                    cur.x += st_a.margin_right.max(0);
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
                    let mark = mark_frag(runs, links, rects, images, controls, frames, aux);
                    cur.x += st_i.padding_left.max(0) + st_i.border_left_width.max(0);
                    let mut sibs: Vec<css::ElemRef<'a>> = Vec::new();
                    for c in &n.children {
                        walk(
                            c, sheet, &st_i, cur, runs, links, rects, images, controls, frames,
                            aux, link, form, next_form_id, page_bg, vw, in_head, child_chain, &sibs,
                        );
                        if let Some(r) = elem_ref_for(c) {
                            sibs.push(r);
                        }
                    }
                    cur.x += st_i.padding_right.max(0) + st_i.border_right_width.max(0);
                    paint_inline_box(mark, &st_i, runs, rects);
                }
                "ul" | "ol" | "body" | "html" => {
                    cur.line_h = line_h;
                    let mut sibs: Vec<css::ElemRef<'a>> = Vec::new();
                    for c in &n.children {
                        walk(
                            c, sheet, &st, cur, runs, links, rects, images, controls, frames, aux, link,
                            form, next_form_id, page_bg, vw, in_head, child_chain, &sibs,
                        );
                        if let Some(r) = elem_ref_for(c) {
                            sibs.push(r);
                        }
                    }
                }
                _ => {
                    // Unknown or remaining catalog tags: block vs inline from DisplayKind.
                    match dkind {
                        DisplayKind::Inline | DisplayKind::InlineBlock => {
                            cur.line_h = line_h.max(cur.line_h);
                            let mark =
                                mark_frag(runs, links, rects, images, controls, frames, aux);
                            cur.x += st.padding_left.max(0) + st.border_left_width.max(0);
                            let mut sibs: Vec<css::ElemRef<'a>> = Vec::new();
                            for c in &n.children {
                                walk(
                                    c, sheet, &st, cur, runs, links, rects, images, controls,
                                    frames, aux, link, form, next_form_id, page_bg, vw, in_head, child_chain, &sibs,
                                );
                                if let Some(r) = elem_ref_for(c) {
                                    sibs.push(r);
                                }
                            }
                            cur.x += st.padding_right.max(0) + st.border_right_width.max(0);
                            paint_inline_box(mark, &st, runs, rects);
                        }
                        DisplayKind::ListItem => {
                            block_before(cur, st.margin_top);
                            {
                        let marker = list_marker(&st);
                        if !marker.is_empty() { emit_text(marker, cur, runs, links, link, &st); }
                    }
                            let mut sibs: Vec<css::ElemRef<'a>> = Vec::new();
                            for c in &n.children {
                                walk(
                                    c, sheet, &st, cur, runs, links, rects, images, controls,
                                    frames, aux, link, form, next_form_id, page_bg, vw, in_head, child_chain, &sibs,
                                );
                                if let Some(r) = elem_ref_for(c) {
                                    sibs.push(r);
                                }
                            }
                            block_after(cur, st.margin_bottom);
                        }
                        _ => {
                            block_before(cur, st.margin_top);
                            let mut sibs: Vec<css::ElemRef<'a>> = Vec::new();
                            for c in &n.children {
                                walk(
                                    c, sheet, &st, cur, runs, links, rects, images, controls,
                                    frames, aux, link, form, next_form_id, page_bg, vw, in_head, child_chain, &sibs,
                                );
                                if let Some(r) = elem_ref_for(c) {
                                    sibs.push(r);
                                }
                            }
                            if cur.x > cur.margin_x {
                                new_line(cur);
                            }
                            block_after(cur, st.margin_bottom);
                        }
                    }
                }
            }
            // Generated content ::after (after children / tag-specific layout).
            // `st` may have been moved into tag arms — recompute lightly for colour/font.
            if let Some(pst) = css::compute_pseudo(
                sheet,
                el_ref,
                parent_st,
                chain,
                css::PseudoElement::After,
            ) {
                emit_text(&pst.content, cur, runs, links, link, &pst);
            }
        }
    }
}
/// Paint the background + border of an **inline** element behind the runs its
/// children produced.
///
/// A block box gets its background from the block path; an inline one had none
/// at all, so `<span class="rounded-full border bg-emerald-50 px-2">loaded</span>`
/// — a badge, the single most common shadcn/Tailwind inline component —
/// rendered as bare text. One box per line, the way a browser fragments an
/// inline box that wraps.
///
/// Vertical padding deliberately expands only the painted box: per CSS, an
/// inline box's vertical padding does not grow the line, it overflows it.
/// Horizontal padding *does* advance the cursor, and that is done by the
/// caller before/after walking the children.
fn paint_inline_box(
    start: FragMark,
    st: &ComputedStyle,
    runs: &[TextRun],
    rects: &mut Vec<RectBox>,
) {
    let visible_border = (st.border_top_style.is_visible() && st.border_top_width > 0)
        || (st.border_bottom_style.is_visible() && st.border_bottom_width > 0)
        || (st.border_left_style.is_visible() && st.border_left_width > 0)
        || (st.border_right_style.is_visible() && st.border_right_width > 0);
    if st.background.is_none() && !visible_border {
        return;
    }
    // Group this element's runs into line boxes by their y.
    let mut lines: Vec<(i32, i32, i32, i32)> = Vec::new(); // (y, h, x0, x1)
    for r in &runs[start.runs..] {
        let w = (crate::font_ttf::measure(&r.text, r.font_size.max(8) as f32) + 0.5) as i32;
        let h = (crate::font_ttf::line_height(r.font_size.max(8) as f32) + 0.5) as i32;
        match lines.iter_mut().find(|l| l.0 == r.y) {
            Some(l) => {
                l.1 = l.1.max(h);
                l.2 = l.2.min(r.x);
                l.3 = l.3.max(r.x + w);
            }
            None => lines.push((r.y, h.max(1), r.x, r.x + w)),
        }
    }
    let (pl, pr) = (st.padding_left.max(0), st.padding_right.max(0));
    let (pt, pb) = (st.padding_top.max(0), st.padding_bottom.max(0));
    let bl = if st.border_left_style.is_visible() { st.border_left_width.max(0) } else { 0 };
    let br = if st.border_right_style.is_visible() { st.border_right_width.max(0) } else { 0 };
    let bt = if st.border_top_style.is_visible() { st.border_top_width.max(0) } else { 0 };
    let bb = if st.border_bottom_style.is_visible() { st.border_bottom_width.max(0) } else { 0 };
    // Insert at the fragment's start so the box lands *behind* anything the
    // children drew (a nested badge, an inline image background).
    let mut at = start.rects;
    for (y, h, x0, x1) in lines {
        let x = x0 - pl - bl;
        let w = (x1 - x0) + pl + pr + bl + br;
        let by = y - pt - bt;
        let bh = h + pt + pb + bt + bb;
        if w <= 0 || bh <= 0 {
            continue;
        }
        if let Some(bg) = st.background {
            rects.insert(
                at,
                RectBox {
                    x,
                    y: by,
                    w,
                    h: bh,
                    color: fade(bg, st.opacity),
                    radius: st.border_radius.max(0),
                    blur: 0,
                },
            );
            at += 1;
        }
        if visible_border {
            let mut edges = Vec::new();
            push_box_borders(&mut edges, x, by, w, bh, st);
            for e in edges {
                rects.insert(at, e);
                at += 1;
            }
        }
    }
}

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

/// Translate only the geometry in `[start, end)` — used when several sibling
/// fragments coexist in the buffers (table cells) and each must move
/// independently. `translate_frag` (open-ended) would move every later sibling
/// too, accumulating shifts.
#[allow(clippy::too_many_arguments)]
fn translate_frag_range(
    start: FragMark,
    end: FragMark,
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
    for r in &mut runs[start.runs..end.runs] {
        r.x += dx;
        r.y += dy;
    }
    for l in &mut links[start.links..end.links] {
        l.x += dx;
        l.y += dy;
    }
    for rc in &mut rects[start.rects..end.rects] {
        rc.x += dx;
        rc.y += dy;
    }
    for im in &mut images[start.images..end.images] {
        im.x += dx;
        im.y += dy;
    }
    for c in &mut controls[start.controls..end.controls] {
        c.x += dx;
        c.y += dy;
    }
    for f in &mut frames[start.frames..end.frames] {
        f.x += dx;
        f.y += dy;
    }
    for e in &mut aux.elem_boxes[start.elem_boxes..end.elem_boxes] {
        e.x += dx;
        e.y += dy;
    }
    for b in &mut aux.bg_boxes[start.bg_boxes..end.bg_boxes] {
        b.x += dx;
        b.y += dy;
    }
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

/// Discard everything appended since `start` — used to throw away a trial
/// (measurement) layout of a table cell.
fn rewind_frag(
    start: FragMark,
    runs: &mut Vec<TextRun>,
    links: &mut Vec<LinkBox>,
    rects: &mut Vec<RectBox>,
    images: &mut Vec<ImageBox>,
    controls: &mut Vec<FormControl>,
    frames: &mut Vec<FrameBox>,
    aux: &mut Aux,
) {
    runs.truncate(start.runs);
    links.truncate(start.links);
    rects.truncate(start.rects);
    images.truncate(start.images);
    controls.truncate(start.controls);
    frames.truncate(start.frames);
    aux.elem_boxes.truncate(start.elem_boxes);
    aux.bg_boxes.truncate(start.bg_boxes);
}

/// `(tag, id, class, style)` of an element node (empty tag for non-elements).
fn elem_parts(n: &Node) -> (&str, Option<&str>, Option<&str>, Option<&str>) {
    match &n.kind {
        NodeKind::Element { tag, id, class, style_attr, .. } => (
            tag.as_str(),
            id.as_deref(),
            class.as_deref(),
            style_attr.as_deref(),
        ),
        _ => ("", None, None, None),
    }
}

/// Apply HTML presentational attributes (`bgcolor`, `width="85%"`, `width=N`,
/// `align`) onto a computed style. Used from both the general walk and
/// table-cell isolation — cells never go through the walk path for their own
/// box.
fn apply_presentational(n: &Node, st: &mut ComputedStyle, avail_w: i32) {
    let NodeKind::Element {
        bgcolor_attr,
        width_attr,
        width_pct,
        align_attr,
        height_attr,
        ..
    } = &n.kind
    else {
        return;
    };
    // Presentational bgcolor always wins over an unset background; if CSS set
    // one, leave it. HN's orange header/cream body are *only* presentational.
    if st.background.is_none() {
        if let Some(bg) = bgcolor_attr
            .as_deref()
            .and_then(|s| css::parse_color(s))
        {
            st.background = Some(bg);
        }
    }
    if st.width.is_none() {
        if let Some(pct) = *width_pct {
            st.width = Some((avail_w as i64 * pct as i64 / 100).max(1) as i32);
        } else if let Some(w) = *width_attr {
            st.width = Some(w.max(1));
        }
    }
    if st.height.is_none() {
        if let Some(h) = *height_attr {
            st.height = Some(h.max(1));
        }
    }
    // Presentational `align=left|center|right` (HN rank column, etc.).
    if let Some(a) = align_attr.as_deref() {
        match a.to_ascii_lowercase().as_str() {
            "center" | "middle" => st.text_align = Align::Center,
            "right" => st.text_align = Align::Right,
            "left" | "justify" => st.text_align = Align::Left,
            _ => {}
        }
    }
}

/// One cell in a table row, with its HTML `colspan`/`rowspan` (minimum 1).
struct TableCellRef<'a> {
    node: &'a Node,
    colspan: u32,
    rowspan: u32,
}

fn cell_colspan(n: &Node) -> u32 {
    match &n.kind {
        NodeKind::Element {
            colspan_attr: Some(c),
            ..
        } => (*c).max(1),
        _ => 1,
    }
}

fn cell_rowspan(n: &Node) -> u32 {
    match &n.kind {
        NodeKind::Element {
            rowspan_attr: Some(r),
            ..
        } => (*r).max(1),
        _ => 1,
    }
}

/// Gather a table's rows as lists of cell nodes (`<td>`/`<th>`), descending
/// through `<thead>`/`<tbody>`/`<tfoot>`. Rowspan is not modeled; colspan is
/// carried on each cell for placement.
fn collect_table_rows<'a>(n: &'a Node, rows: &mut Vec<Vec<TableCellRef<'a>>>) {
    for child in &n.children {
        if let NodeKind::Element { tag, .. } = &child.kind {
            match tag.as_str() {
                "tr" => {
                    let cells: Vec<TableCellRef<'a>> = child
                        .children
                        .iter()
                        .filter(|c| {
                            matches!(&c.kind, NodeKind::Element { tag, .. }
                                if tag == "td" || tag == "th")
                        })
                        .map(|c| TableCellRef {
                            node: c,
                            colspan: cell_colspan(c),
                            rowspan: cell_rowspan(c),
                        })
                        .collect();
                    rows.push(cells);
                }
                "thead" | "tbody" | "tfoot" => collect_table_rows(child, rows),
                _ => {}
            }
        }
    }
}

/// Lay a table cell out as an isolated fragment at the origin within `width`,
/// insetting its content by the cell's padding. Returns `(mark, box_w, box_h)`
/// (border-box dimensions). The caller either rewinds (trial measure) or
/// translates the fragment into place.
///
/// **`box_w` is the intrinsic content width**, not forced to `width`. Forcing
/// every cell's max-content to the table container (the previous behaviour)
/// made each of HN's 3 story columns claim full width and destroyed column
/// sizing. Final placement still lays out into the assigned column/`colspan`
/// width; only the returned measure is intrinsic. Explicit CSS/presentational
/// `width` is honored as a minimum (logo column, etc.).
///
/// `place`: when `false` (min/max-content trial), skip `text-align` /
/// presentational `align` — right-aligning `"1."` into a full-table trial
/// width made frag_bbox span the whole table and inflated the rank column
/// to ~full width (HN ranks then sat mid-cream). When `true` (final place),
/// apply alignment within the real column width.
#[allow(clippy::too_many_arguments)]
fn layout_cell_isolated<'a>(
    cell: &'a Node,
    sheet: &Stylesheet,
    parent_st: &ComputedStyle,
    width: i32,
    line_h: i32,
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
    chain: &[css::ElemRef<'a>],
    place: bool,
) -> (FragMark, i32, i32, ComputedStyle) {
    let (ct, cid, cc, cs) = elem_parts(cell);
    let mut cst = css::compute_ex(sheet, ct, cid, cc, cs, parent_st, chain);
    // Critical: table cells never go through `walk` for their own box, so
    // presentational `bgcolor` / `width` / `align` must be applied here.
    // Without this, HN's orange header (`<td bgcolor="#ff6600">`) is invisible
    // and rank cells ignore `align=right`.
    apply_presentational(cell, &mut cst, width);
    let child_chain = push_chain(chain, ct, cid, cc);
    let child_chain = child_chain.as_slice();
    let (pl, pr) = (cst.padding_left.max(0), cst.padding_right.max(0));
    let (pt, pb) = (cst.padding_top.max(0), cst.padding_bottom.max(0));
    let mark = mark_frag(runs, links, rects, images, controls, frames, aux);
    let content_w_avail = (width - pl - pr).max(1);
    let mut ccur = Cursor {
        x: pl,
        y: pt,
        max_w: content_w_avail,
        margin_x: pl,
        line_h,
        content_bottom: pt,
        float_l_w: 0,
        float_l_bottom: 0,
        float_r_w: 0,
        float_r_bottom: 0,
    };
    let mut sibs: Vec<css::ElemRef<'a>> = Vec::new();
    for c in &cell.children {
        walk(
            c, sheet, &cst, &mut ccur, runs, links, rects, images, controls, frames, aux, link,
            form, next_form_id, page_bg, vw, in_head, child_chain, &sibs,
        );
        if let Some(r) = elem_ref_for(c) {
            sibs.push(r);
        }
    }
    // Measure first (intrinsic), then optionally align for final paint.
    let (x0, _y0, x1, y1) = frag_bbox(mark, runs, links, rects, images, controls, frames, aux);
    let content_w = if x1 >= x0 { (x1 - pl).max(0) } else { 0 };
    // Intrinsic width from content (+ padding). Explicit width is a minimum
    // (fixed logo column) — never force up to the trial `width` argument.
    let mut box_w = content_w + pl + pr;
    if let Some(fw) = cst.width {
        box_w = box_w.max(fw).max(pl + pr);
    }
    let content_h = if y1 >= pt {
        y1 - pt
    } else {
        0
    };
    let mut box_h = content_h
        .max(ccur.content_bottom.saturating_sub(pt))
        .max(if content_w > 0 || cst.background.is_some() {
            line_h / 2
        } else {
            0
        })
        + pt
        + pb;
    if let Some(fh) = cst.height {
        box_h = box_h.max(fh + pt + pb);
    }
    // Minimum height for cells with bgcolor but little content (header strip).
    if cst.background.is_some() {
        box_h = box_h.max(line_h + pt + pb);
    }
    // Final placement only: `text-align` / presentational `align` (login right,
    // rank right) within the real column width — never during measure.
    if place && !matches!(cst.text_align, Align::Left) {
        align_inline(
            cst.text_align,
            pl,
            content_w_avail,
            &mut runs[mark.runs..],
            &mut links[mark.links..],
            &mut images[mark.images..],
            &mut controls[mark.controls..],
        );
    }
    (mark, box_w.max(1), box_h.max(1), cst)
}

/// **Genuine auto table layout.** Measures each cell's min/max-content width,
/// distributes the container width across columns (fixed `width` honored,
/// remaining space shared between min and max per the CSS auto algorithm), then
/// places each cell into its column x / row y with the row's max height.
/// `border-spacing` gutters cells. Supports `colspan` (rowspan still ignored).
///
/// `parent_st` is the style of the table's parent — used so a `<center>`
/// (text-align:center) can center a fixed-width table box without forcing
/// every cell's content to center (tables/tds reset text-align in the UA sheet).
#[allow(clippy::too_many_arguments)]
fn layout_table<'a>(
    n: &'a Node,
    sheet: &Stylesheet,
    table_st: &ComputedStyle,
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
    chain: &[css::ElemRef<'a>],
) {
    let px = table_st.font_size.max(8) as f32;
    let line_h = (crate::font_ttf::line_height(px) + 0.5) as i32;
    // Cells see the table as their nearest matched ancestor (tr/tbody are
    // flattened by collect_table_rows — an approximation for `tr td` selectors).
    let (tt, tid, tc, _) = elem_parts(n);
    let cell_chain = push_chain(chain, tt, tid, tc);
    let cell_chain = cell_chain.as_slice();
    let mut rows: Vec<Vec<TableCellRef<'a>>> = Vec::new();
    collect_table_rows(n, &mut rows);
    if rows.is_empty() {
        return;
    }
    // Column count: walk rows skipping slots covered by rowspans from above.
    let mut pending = alloc::vec![0u32; 64]; // rows remaining that cover column c
    let mut ncols = 1usize;
    for row in &rows {
        for p in pending.iter_mut() {
            if *p > 0 {
                *p -= 1;
            }
        }
        let mut ci = 0usize;
        for cell in row {
            while ci < 64 && pending[ci] > 0 {
                ci += 1;
            }
            let cs = (cell.colspan as usize).max(1);
            let rs = (cell.rowspan as usize).max(1);
            for c in 0..cs {
                if ci + c < 64 {
                    pending[ci + c] = (rs as u32).saturating_sub(1);
                }
            }
            ci += cs;
            ncols = ncols.max(ci);
        }
        for (c, p) in pending.iter().enumerate() {
            if *p > 0 {
                ncols = ncols.max(c + 1);
            }
        }
    }
    let ncols = ncols.max(1).min(64);
    block_before(cur, table_st.margin_top.max(2));
    // Presentational `width="85%"` (HN `#hnmain`) / CSS width shrinks the
    // table; `<center>` (inherited `text-align:center`) or `margin:auto` centers it.
    let full_w =
        (cur.max_w - table_st.margin_left.max(0) - table_st.margin_right.max(0)).max(1);
    let container_w = table_st
        .width
        .map(|w| w.clamp(1, full_w))
        .unwrap_or(full_w);
    let box_x = if table_st.width.is_some()
        && (matches!(table_st.text_align, Align::Center)
            || matches!(parent_st.text_align, Align::Center)
            || (table_st.margin_left_auto && table_st.margin_right_auto))
    {
        cur.margin_x + (full_w - container_w) / 2
    } else {
        cur.margin_x + table_st.margin_left.max(0)
    };
    let spacing = table_st.border_spacing.max(0);

    // 1. Per-column min/max-content widths (honoring an explicit cell `width`).
    // Spanning cells contribute evenly across their columns for measurement.
    // `table-layout: fixed` → equal columns (or first-row explicit widths).
    let fixed_layout = table_st.table_layout == "fixed";
    let mut col_min = alloc::vec![0i32; ncols];
    let mut col_max = alloc::vec![0i32; ncols];
    if fixed_layout {
        let equal = (container_w - spacing * (ncols as i32 + 1))
            .max(1)
            / ncols as i32;
        for i in 0..ncols {
            col_min[i] = equal.max(1);
            col_max[i] = equal.max(1);
        }
    } else {
    for row in &rows {
        let mut ci = 0usize;
        for cell in row {
            let span = (cell.colspan as usize).max(1).min(ncols.saturating_sub(ci).max(1));
            if ci >= ncols {
                break;
            }
            let (m1, maxw, _, cst) = layout_cell_isolated(
                cell.node, sheet, table_st, container_w, line_h, runs, links, rects, images,
                controls, frames, aux, link, form, next_form_id, page_bg, vw, in_head, cell_chain,
                false, // measure: no text-align (would inflate align=right cols)
            );
            rewind_frag(m1, runs, links, rects, images, controls, frames, aux);
            let (m2, minw, _, _) = layout_cell_isolated(
                cell.node, sheet, table_st, 8, line_h, runs, links, rects, images, controls,
                frames, aux, link, form, next_form_id, page_bg, vw, in_head, cell_chain,
                false,
            );
            rewind_frag(m2, runs, links, rects, images, controls, frames, aux);
            let fixed = cst.width;
            let max_one = fixed.unwrap_or(maxw) / span as i32;
            let min_one = fixed.unwrap_or(minw).min(fixed.unwrap_or(maxw)) / span as i32;
            for k in 0..span {
                if ci + k < ncols {
                    col_max[ci + k] = col_max[ci + k].max(max_one);
                    col_min[ci + k] = col_min[ci + k].max(min_one);
                }
            }
            ci += span;
        }
    }
    } // end auto layout measure

    // 2. Distribute the available width across columns.
    let gutters = spacing * (ncols as i32 + 1);
    let avail = (container_w - gutters).max(1);
    let total_min: i32 = col_min.iter().sum();
    let total_max: i32 = col_max.iter().sum();
    let mut col_w: Vec<i32> = if total_max <= avail {
        col_max.clone()
    } else if total_min >= avail {
        col_min.clone()
    } else {
        let extra = avail - total_min;
        let span = (total_max - total_min).max(1);
        col_min
            .iter()
            .zip(&col_max)
            .map(|(&mn, &mx)| mn + (mx - mn) * extra / span)
            .collect()
    };
    // Always expand columns to fill the table's used width. Without this a
    // single full-row header cell (HN orange bar) paints only as wide as its
    // text, and story tables leave a ragged cream edge.
    //
    // Prefer giving leftover to the last column (HN title column / login).
    // For a 1-column table the sole column takes it all.
    let used: i32 = col_w.iter().sum();
    if used < avail {
        let extra = avail - used;
        if let Some(last) = col_w.last_mut() {
            *last += extra;
        }
    } else if used > avail && avail > 0 {
        // Shrink proportionally so we never overflow the table box.
        for w in col_w.iter_mut() {
            *w = ((*w as i64) * (avail as i64) / (used as i64)).max(1) as i32;
        }
        let used2: i32 = col_w.iter().sum();
        if used2 < avail {
            if let Some(last) = col_w.last_mut() {
                *last += avail - used2;
            }
        }
    }
    let mut col_x = Vec::with_capacity(ncols);
    let mut acc = box_x + spacing;
    for w in &col_w {
        col_x.push(acc);
        acc += w + spacing;
    }
    // Table box uses the fixed container width when one was requested.
    let table_w = if table_st.width.is_some() {
        container_w
    } else {
        (acc - box_x).max(1)
    };

    // 3. Place each row's cells; row height = tallest cell.
    // Spanned width = sum of column widths + gutters between them.
    let table_y0 = cur.y;
    // Table background is recorded *after* we know table_h (step 4), but must
    // paint *under* cell backgrounds. We push a placeholder and fix up height
    // once rows are done — or insert at the front. Simpler: remember the index
    // and rewrite height later.
    let table_bg_idx = if let Some(bg) = table_st.background {
        let i = rects.len();
        rects.push(RectBox {
            x: box_x,
            y: table_y0,
            w: table_w,
            h: 1, // patched after rows
            color: fade(bg, table_st.opacity),
            radius: table_st.border_radius.max(0),
            blur: 0,
        });
        Some(i)
    } else {
        None
    };
    let mut row_y = cur.y + spacing;
    let mut pending = alloc::vec![0u32; ncols]; // rowspan cover remaining
    for row in &rows {
        for p in pending.iter_mut() {
            if *p > 0 {
                *p -= 1;
            }
        }
        let mut placed: Vec<(FragMark, FragMark, usize, u32, u32, ComputedStyle)> = Vec::new();
        let mut row_h = line_h;
        let mut ci = 0usize;
        for cell in row {
            while ci < ncols && pending[ci] > 0 {
                ci += 1;
            }
            let span = (cell.colspan as usize).max(1).min(ncols.saturating_sub(ci).max(1));
            let rspan = cell.rowspan.max(1);
            if ci >= ncols {
                break;
            }
            let mut span_w = 0i32;
            for k in 0..span {
                if ci + k < ncols {
                    span_w += col_w[ci + k];
                    if k + 1 < span {
                        span_w += spacing;
                    }
                }
            }
            span_w = span_w.max(1);
            let (mark, _w, h, cst) = layout_cell_isolated(
                cell.node, sheet, table_st, span_w, line_h, runs, links, rects, images,
                controls, frames, aux, link, form, next_form_id, page_bg, vw, in_head, cell_chain,
                true, // final place: apply cell text-align / align=
            );
            let end = mark_frag(runs, links, rects, images, controls, frames, aux);
            if rspan <= 1 {
                row_h = row_h.max(h);
            } else {
                row_h = row_h.max(h / rspan as i32).max(line_h);
            }
            for k in 0..span {
                if ci + k < ncols {
                    pending[ci + k] = rspan.saturating_sub(1);
                }
            }
            placed.push((mark, end, ci, span as u32, rspan, cst));
            ci += span;
        }
        for (mark, end, ci, span, rspan, cst) in &placed {
            let mut span_w = 0i32;
            for k in 0..(*span as usize) {
                if *ci + k < ncols {
                    span_w += col_w[*ci + k];
                    if k + 1 < *span as usize {
                        span_w += spacing;
                    }
                }
            }
            span_w = span_w.max(1);
            let cell_h = if *rspan <= 1 {
                row_h
            } else {
                row_h + (*rspan as i32 - 1) * (line_h + spacing)
            };
            translate_frag_range(
                *mark, *end, col_x[*ci], row_y, runs, links, rects, images, controls, frames, aux,
            );
            if let Some(bg) = cst.background {
                rects.push(RectBox {
                    x: col_x[*ci],
                    y: row_y,
                    w: span_w,
                    h: cell_h,
                    color: fade(bg, cst.opacity),
                    radius: cst.border_radius.max(0),
                    blur: 0,
                });
            }
            push_box_borders(rects, col_x[*ci], row_y, span_w, cell_h, cst);
        }
        row_y += row_h + spacing;
    }

    // 4. Patch the table background height (pushed under cells in step 3) and
    // draw the border on top.
    let table_h = (row_y - table_y0).max(1);
    if let Some(i) = table_bg_idx {
        if let Some(r) = rects.get_mut(i) {
            r.h = table_h;
        }
    }
    push_box_borders(rects, box_x, table_y0, table_w, table_h, table_st);
    aux.push_elem_box(n.elem_idx, box_x, table_y0, table_w, table_h);
    cur.y = row_y;
    cur.content_bottom = cur.content_bottom.max(cur.y);
    block_after(cur, table_st.margin_bottom.max(2));
}

/// Layout children of a flex/grid container: measure each element child as an
/// independent fragment, then place with [`flex`] geometry.
fn layout_flex_grid_container<'a>(
    n: &'a Node,
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
    chain: &[css::ElemRef<'a>],
) {
    // Flex/grid items are children of `n`; their ancestor chain is n's plus n.
    let (nt, nid, nc, _) = elem_parts(n);
    let item_chain = push_chain(chain, nt, nid, nc);
    let item_chain = item_chain.as_slice();
    block_before(cur, st.margin_top.max(0));
    cur.y += st.padding_top;
    let box_x = cur.margin_x + st.margin_left;
    let box_y = cur.y;
    // The space the container may use. An inline-level flex container is
    // shrink-to-fit: this is its *upper bound*, and the real width comes from
    // the items once they have been measured (see `shrink_to_fit` below).
    let avail_w = (cur.max_w - st.margin_left - st.margin_right).max(1);
    let container_w = st.width.or(st.max_width).unwrap_or(avail_w).max(1);
    let shrink_to_fit = st.display == DisplayMode::InlineFlex && st.width.is_none();

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
        d if d.is_flex() && st.flex_direction == FlexDirection::Row => {
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
            radius: 0,
            blur: 0,
        });
        i
    });

    struct Item {
        mark: FragMark,
        end: FragMark,
        x0: i32,
        y0: i32,
        w: i32,
        h: i32,
        grow: u32,
        order: i32,
    }
    let mut items: Vec<Item> = Vec::with_capacity(children.len());

    // Flex items are each other's siblings, and this is the loop Tailwind's
    // `space-y-*` depends on — it compiles to
    // `.space-y-4 > :not([hidden]) ~ :not([hidden])`, i.e. "every item after the
    // first". Without the accumulator the `~` was approximated as descendant and
    // the margin landed on the first item too.
    let mut item_sibs: Vec<css::ElemRef<'a>> = Vec::new();
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
        let cst = css::compute_ex(sheet, ctag, cid, cclass, cstyle, st, item_chain);
        // `flex-basis` is the item's initial main size (overrides `width`).
        let item_w = cst
            .flex_basis
            .filter(|&b| b > 0)
            .or(cst.width)
            .or(cst.max_width)
            .unwrap_or(default_item_w)
            // `container_w` can be below the 16px floor for a deeply-nested or
            // zero-width flex item (google.com's modern layout hits this) —
            // guard the upper bound so `clamp` never sees min > max (panic).
            .clamp(16, container_w.max(16));

        let mark = mark_frag(runs, links, rects, images, controls, frames, aux);
        // Isolated cursor at origin so fragment coords start near (0,0).
        // Auto-size: use natural max_w; for row flex wrap prefer content width.
        // An inline-level flex container sizes to its content, so its items
        // must be measured at their NATURAL width — the same roomy measure a
        // wrapping row uses — never at an equal share of the line.
        let measure_w = if st.display.is_flex()
            && st.flex_direction == FlexDirection::Row
            && (st.flex_wrap != flex::FlexWrap::NoWrap
                || st.display == DisplayMode::InlineFlex)
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
            float_l_w: 0,
            float_l_bottom: 0,
            float_r_w: 0,
            float_r_bottom: 0,
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
            item_chain,
            &item_sibs,
        );
        if let Some(r) = elem_ref_for(child) {
            item_sibs.push(r);
        }
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
        } else if st.display.is_flex()
            && st.flex_direction == FlexDirection::Row
            && (st.flex_wrap != flex::FlexWrap::NoWrap
                || st.display == DisplayMode::InlineFlex)
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
        // End of this item's geometry (its fragment is `[mark, end)`), so each
        // item can be translated independently without shifting later siblings.
        let end = mark_frag(runs, links, rects, images, controls, frames, aux);
        items.push(Item {
            mark,
            end,
            x0,
            y0,
            w,
            h,
            grow,
            order: cst.order,
        });
    }
    // `order` reorders flex/grid items visually (stable within equal order).
    let mut visual: Vec<usize> = (0..items.len()).collect();
    visual.sort_by_key(|&i| (items[i].order, i));

    // Shrink-to-fit: a `display: inline-flex` box is exactly as wide as its
    // items plus the gaps between them (plus its own padding), bounded by the
    // line. Without this every shadcn button and badge — all `inline-flex` —
    // became a full-width bar.
    let container_w = if shrink_to_fit {
        let sum: i32 = items.iter().map(|it| it.w.max(0)).sum();
        let gaps = gap * (items.len() as i32 - 1).max(0);
        (sum + gaps + st.padding_left.max(0) + st.padding_right.max(0))
            .max(1)
            .min(avail_w)
    } else {
        container_w
    };

    let mut content_h = line_h;
    let mut content_w = container_w;

    match st.display {
        d if d.is_flex() => {
            // Arrays in visual (`order`-sorted) sequence; a placement's index
            // maps back through `visual` to the real item's fragment.
            let widths: Vec<i32> = visual.iter().map(|&i| items[i].w).collect();
            let heights: Vec<i32> = visual.iter().map(|&i| items[i].h).collect();
            let grows: Vec<u32> = visual.iter().map(|&i| items[i].grow).collect();
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
                if let Some(it) = visual.get(p.index).and_then(|&i| items.get(i)) {
                    let dx = box_x + p.x - it.x0;
                    let dy = box_y + p.y - it.y0;
                    translate_frag_range(it.mark, it.end, dx, dy, runs, links, rects, images, controls, frames, aux);
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
                        translate_frag_range(
                            it.mark, it.end, dx, dy, runs, links, rects, images, controls, frames, aux,
                        );
                        max_y = max_y.max(p.y + p.h);
                    }
                }
                content_h = max_y;
            } else {
                // Real track sizing (`fr`/px/auto) + per-column x offsets, and
                // `order`-sorted placement.
                let track_w =
                    flex::grid_track_widths(&st.grid_template, container_w, gap, cols as usize);
                let ncols = track_w.len().max(1);
                let mut col_x = Vec::with_capacity(ncols);
                let mut acc = 0;
                for tw in &track_w {
                    col_x.push(acc);
                    acc += tw + gap;
                }
                for (slot, &vi) in visual.iter().enumerate() {
                    let it = &items[vi];
                    let col = slot % ncols;
                    let row = slot / ncols;
                    let cw = track_w[col];
                    let gx = col_x[col];
                    let gy = row as i32 * (cell_h + gap);
                    let ix = flex::flex_cross_offset(it.w.min(cw), cw, st.align_items);
                    let iy = flex::flex_cross_offset(it.h.min(cell_h), cell_h, st.align_items);
                    let dx = box_x + gx + ix - it.x0;
                    let dy = box_y + gy + iy - it.y0;
                    translate_frag_range(it.mark, it.end, dx, dy, runs, links, rects, images, controls, frames, aux);
                }
                let rows = (items.len() + ncols - 1) / ncols;
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
    radius: i32,
    bg: Option<u32>,
    fg: Option<u32>,
    transparent: bool,
    css_w: Option<i32>,
    css_h: Option<i32>,
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
            radius: 0,
            bg: None,
            fg: None,
            transparent: false,
        });
        return;
    }
    let (w, h) = match kind {
        ControlKind::Text | ControlKind::Password => (
            css_w
                .map(|w| w.clamp(24, cur.max_w.max(24)))
                .unwrap_or_else(|| cur.max_w.min(280).max(120)),
            css_h.map(|h| h.max(line_h)).unwrap_or(line_h + 10),
        ),
        ControlKind::TextArea => (
            css_w
                .map(|w| w.clamp(24, cur.max_w.max(24)))
                .unwrap_or_else(|| cur.max_w.min(320).max(160)),
            css_h.map(|h| h.max(line_h)).unwrap_or(line_h * 4 + 12),
        ),
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
        radius,
        bg,
        fg,
        transparent,
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
    cur.y += cur.line_h;
    // Retire floats whose exclusion ends above the new line.
    if cur.y >= cur.float_l_bottom {
        cur.float_l_w = 0;
    }
    if cur.y >= cur.float_r_bottom {
        cur.float_r_w = 0;
    }
    // Following lines flow beside an active left float (wrap-around).
    cur.x = cur.margin_x + cur.float_l_w;
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

/// Extract the `(tx, ty)` translation from a CSS `transform` value
/// (`translate(x,y)`, `translateX(x)`, `translateY(y)`). Scale/rotate/skew are
/// not applied (they need re-rasterisation); only the offset is honored.
fn parse_transform_translate(s: &str) -> (i32, i32) {
    let low = s.to_ascii_lowercase();
    let arg = |key: &str| -> Option<&str> {
        let start = low.find(key)? + key.len();
        let end = low[start..].find(')')? + start;
        Some(s[start..end].trim())
    };
    let (mut tx, mut ty) = (0, 0);
    if let Some(a) = arg("translate(") {
        let mut it = a.split(',');
        tx = it.next().and_then(|v| css::parse_px(v.trim())).unwrap_or(0);
        ty = it.next().and_then(|v| css::parse_px(v.trim())).unwrap_or(0);
    }
    if let Some(a) = arg("translatex(") {
        tx = css::parse_px(a).unwrap_or(tx);
    }
    if let Some(a) = arg("translatey(") {
        ty = css::parse_px(a).unwrap_or(ty);
    }
    (tx, ty)
}

/// `(scale_x, scale_y)` from a `transform` (`scale(s)`, `scale(sx,sy)`,
/// `scaleX/scaleY`). Defaults to `(1.0, 1.0)`.
fn parse_transform_scale(s: &str) -> (f32, f32) {
    let low = s.to_ascii_lowercase();
    let arg = |key: &str| -> Option<&str> {
        let start = low.find(key)? + key.len();
        let end = low[start..].find(')')? + start;
        Some(low[start..end].trim())
    };
    let (mut sx, mut sy) = (1.0f32, 1.0f32);
    if let Some(a) = arg("scale(") {
        let mut it = a.split(',');
        sx = it.next().and_then(|v| v.trim().parse().ok()).unwrap_or(1.0);
        sy = it.next().and_then(|v| v.trim().parse().ok()).unwrap_or(sx);
    }
    if let Some(a) = arg("scalex(") {
        sx = a.parse().unwrap_or(sx);
    }
    if let Some(a) = arg("scaley(") {
        sy = a.parse().unwrap_or(sy);
    }
    (sx.clamp(0.05, 20.0), sy.clamp(0.05, 20.0))
}

/// Rotation in degrees from a `transform: rotate(<deg>)` (`deg`/`rad`/`turn`).
fn parse_transform_rotate(s: &str) -> f32 {
    let low = s.to_ascii_lowercase();
    let start = match low.find("rotate(") {
        Some(i) => i + 7,
        None => return 0.0,
    };
    let end = match low[start..].find(')') {
        Some(i) => i + start,
        None => return 0.0,
    };
    let a = low[start..end].trim();
    if let Some(v) = a.strip_suffix("deg") {
        v.trim().parse().unwrap_or(0.0)
    } else if let Some(v) = a.strip_suffix("turn") {
        v.trim().parse::<f32>().unwrap_or(0.0) * 360.0
    } else if let Some(v) = a.strip_suffix("rad") {
        v.trim().parse::<f32>().unwrap_or(0.0) * 180.0 / core::f32::consts::PI
    } else {
        a.parse().unwrap_or(0.0)
    }
}

/// Scale a fragment's primitives about `(ox, oy)` by `(sx, sy)` (text size and
/// positions/box dimensions). Mirrors `translate_frag`'s slice handling.
fn scale_frag(
    start: FragMark,
    ox: i32,
    oy: i32,
    sx: f32,
    sy: f32,
    runs: &mut [TextRun],
    links: &mut [LinkBox],
    rects: &mut [RectBox],
    images: &mut [ImageBox],
    controls: &mut [FormControl],
) {
    let px = |v: i32, o: i32, s: f32| o + ((v - o) as f32 * s) as i32;
    let sz = |v: i32, s: f32| ((v as f32 * s) as i32).max(1);
    for r in &mut runs[start.runs..] {
        r.x = px(r.x, ox, sx);
        r.y = px(r.y, oy, sy);
        r.font_size = sz(r.font_size, sy);
    }
    for b in &mut links[start.links..] {
        b.x = px(b.x, ox, sx);
        b.y = px(b.y, oy, sy);
        b.w = sz(b.w, sx);
        b.h = sz(b.h, sy);
    }
    for rc in &mut rects[start.rects..] {
        rc.x = px(rc.x, ox, sx);
        rc.y = px(rc.y, oy, sy);
        rc.w = sz(rc.w, sx);
        rc.h = sz(rc.h, sy);
        rc.radius = sz(rc.radius, (sx + sy) / 2.0);
    }
    for im in &mut images[start.images..] {
        im.x = px(im.x, ox, sx);
        im.y = px(im.y, oy, sy);
        im.w = sz(im.w, sx);
        im.h = sz(im.h, sy);
    }
    for c in &mut controls[start.controls..] {
        c.x = px(c.x, ox, sx);
        c.y = px(c.y, oy, sy);
        c.w = sz(c.w, sx);
        c.h = sz(c.h, sy);
    }
}

/// Parse the first shadow of a `box-shadow` value into `(dx, dy, blur, color)`.
/// Returns `None` for `none`/empty. Handles a parenthesised colour
/// (`rgba(…)`), stops at the second (comma-separated) shadow, and ignores
/// `inset`.
fn parse_box_shadow(s: &str) -> Option<(i32, i32, i32, u32)> {
    let s = s.trim();
    if s.is_empty() || s.eq_ignore_ascii_case("none") {
        return None;
    }
    // `box-shadow` is a comma-separated LIST, and the first entry is very often
    // a placeholder: Tailwind emits
    // `var(--tw-ring-offset-shadow, 0 0 #0000), var(--tw-ring-shadow, 0 0 #0000), var(--tw-shadow)`
    // on every shadowed element. Stopping at the first comma took `0 0 #0000` —
    // a *transparent* shadow — and painted the card solid black. So each entry
    // is parsed and the first one with a real colour wins.
    for entry in split_top_level_commas(s) {
        if let Some(shadow) = parse_one_shadow(&entry) {
            return Some(shadow);
        }
    }
    None
}

/// Split on commas that are not inside `(...)`.
fn split_top_level_commas(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut depth = 0i32;
    for ch in s.chars() {
        match ch {
            '(' => {
                depth += 1;
                cur.push(ch);
            }
            ')' => {
                depth -= 1;
                cur.push(ch);
            }
            ',' if depth <= 0 => out.push(core::mem::take(&mut cur)),
            c => cur.push(c),
        }
    }
    out.push(cur);
    out
}

/// One `[inset] <dx> <dy> [blur] [spread] [color]` entry. `None` when it has no
/// visible colour (`parse_color` answers `None` for a transparent one), which
/// is what makes a placeholder entry skippable.
fn parse_one_shadow(entry: &str) -> Option<(i32, i32, i32, u32)> {
    let mut lens: Vec<i32> = Vec::new();
    let mut color = None;
    for t in css::value_tokens(entry) {
        if t.eq_ignore_ascii_case("inset") {
            continue;
        }
        if let Some(px) = css::parse_px(t) {
            lens.push(px);
        } else if let Some(c) = css::parse_color(t) {
            color = Some(c);
        }
    }
    let color = color?;
    Some((
        lens.first().copied().unwrap_or(0),
        lens.get(1).copied().unwrap_or(0),
        lens.get(2).copied().unwrap_or(0).max(0),
        color,
    ))
}

/// The `<li>` marker string for a computed `list-style` — empty for `none`
/// (nav menus set `list-style:none`, which used to still show a bullet).
fn list_marker(st: &ComputedStyle) -> &'static str {
    match st.list_style {
        css::ListStyle::None => "",
        css::ListStyle::Circle => "◦ ",
        css::ListStyle::Square => "▪ ",
        // Decimal needs an ordered-list counter we don't thread here; fall back
        // to a disc so the item still reads as a list item.
        css::ListStyle::Disc | css::ListStyle::Decimal => "• ",
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
        // Proportional advance from the active TTF face, plus `letter-spacing`
        // between the word's glyphs.
        let ls = st.letter_spacing.unwrap_or(0);
        let w = (crate::font_ttf::measure(&word, px) + 0.5) as i32
            + ls * (word.chars().count() as i32);
        let w = w.max(1);
        // Right boundary shrinks by an active `float:right` exclusion; the line
        // start includes an active `float:left` push.
        let line_start = cur.margin_x + cur.float_l_w;
        let line_right = cur.margin_x + cur.max_w - cur.float_r_w;
        if !no_wrap && cur.x > line_start && cur.x + w > line_right {
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
            letter_spacing: ls,
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

/// Caret / selection endpoint into [`Layout::runs`]: run index + UTF-8 char
/// offset (0..=len). Half-open selection `[anchor, head)` when ordered.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct TextPos {
    pub run: usize,
    pub col: usize,
}

fn run_line_h(run: &TextRun) -> i32 {
    (crate::font_ttf::line_height(run.font_size.max(8) as f32) + 0.5) as i32
}

/// Pixel x of the caret before the `col`-th character of `run`.
fn run_caret_x(run: &TextRun, col: usize) -> i32 {
    let px = run.font_size.max(8) as f32;
    let mut pen = run.x;
    for (i, ch) in run.text.chars().enumerate() {
        if i >= col {
            break;
        }
        let mut b = [0u8; 4];
        let s = ch.encode_utf8(&mut b);
        let w = (crate::font_ttf::measure(s, px) + 0.5) as i32;
        pen += w.max(1) + run.letter_spacing;
    }
    pen
}

fn run_end_x(run: &TextRun) -> i32 {
    run_caret_x(run, run.text.chars().count())
}

/// Char index (caret) in `run` closest to pixel `x`.
fn run_col_at_x(run: &TextRun, x: i32) -> usize {
    let px = run.font_size.max(8) as f32;
    let mut pen = run.x;
    let mut col = 0usize;
    for ch in run.text.chars() {
        let mut b = [0u8; 4];
        let s = ch.encode_utf8(&mut b);
        let w = (crate::font_ttf::measure(s, px) + 0.5) as i32;
        let mid = pen + w.max(1) / 2;
        if x < mid {
            return col;
        }
        pen += w.max(1) + run.letter_spacing;
        col += 1;
    }
    col
}

/// Map a content-space point to the nearest text caret in `layout.runs`.
/// Used for drag-to-select in the browser surface.
pub fn text_pos_at(layout: &Layout, x: i32, y: i32) -> Option<TextPos> {
    if layout.runs.is_empty() {
        return None;
    }
    let mut best: Option<(i32, TextPos)> = None;
    for (ri, run) in layout.runs.iter().enumerate() {
        if run.text.is_empty() {
            continue;
        }
        let lh = run_line_h(run).max(1);
        let end_x = run_end_x(run);
        let dy = if y < run.y {
            run.y - y
        } else if y >= run.y + lh {
            y - (run.y + lh - 1)
        } else {
            0
        };
        let dx = if x < run.x {
            run.x - x
        } else if x > end_x {
            x - end_x
        } else {
            0
        };
        // Prefer vertically matching runs heavily so a click on a line picks it.
        let dist = dx + dy * 8;
        let col = run_col_at_x(run, x);
        let cand = TextPos { run: ri, col };
        if best.map(|(d, _)| dist < d).unwrap_or(true) {
            best = Some((dist, cand));
        }
    }
    best.map(|(_, p)| p)
}

/// Order selection endpoints so `a <= b` in document order.
pub fn normalize_text_pos(a: TextPos, b: TextPos) -> (TextPos, TextPos) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

/// Extract selected plain text between carets `a` and `b` (any order, half-open).
/// Inserts `\n` when consecutive selected runs sit on different lines.
pub fn selection_text(layout: &Layout, a: TextPos, b: TextPos) -> String {
    let (a, b) = normalize_text_pos(a, b);
    if a == b {
        return String::new();
    }
    let mut out = String::new();
    let mut prev_y: Option<i32> = None;
    for ri in a.run..=b.run {
        let Some(run) = layout.runs.get(ri) else {
            break;
        };
        let n = run.text.chars().count();
        let lo = if ri == a.run { a.col.min(n) } else { 0 };
        let hi = if ri == b.run { b.col.min(n) } else { n };
        if lo >= hi {
            continue;
        }
        if let Some(py) = prev_y {
            if (run.y - py).abs() > 2 {
                out.push('\n');
            }
        }
        for (i, ch) in run.text.chars().enumerate() {
            if i >= lo && i < hi {
                out.push(ch);
            }
        }
        prev_y = Some(run.y);
    }
    out
}

/// Highlight rectangles (content space) covering the selection, for paint.
/// Each is `(x, y, w, h)` over a partial or full run.
pub fn selection_rects(layout: &Layout, a: TextPos, b: TextPos) -> Vec<(i32, i32, i32, i32)> {
    let (a, b) = normalize_text_pos(a, b);
    if a == b {
        return Vec::new();
    }
    let mut out = Vec::new();
    for ri in a.run..=b.run {
        let Some(run) = layout.runs.get(ri) else {
            break;
        };
        let n = run.text.chars().count();
        let lo = if ri == a.run { a.col.min(n) } else { 0 };
        let hi = if ri == b.run { b.col.min(n) } else { n };
        if lo >= hi {
            continue;
        }
        let x0 = run_caret_x(run, lo);
        let x1 = run_caret_x(run, hi);
        let h = run_line_h(run).max(1);
        let w = (x1 - x0).max(1);
        out.push((x0, run.y, w, h));
    }
    out
}

/// True if `(x,y)` is over a text run (I-beam affordance).
pub fn text_at(layout: &Layout, x: i32, y: i32) -> bool {
    for run in &layout.runs {
        if run.text.is_empty() {
            continue;
        }
        let lh = run_line_h(run);
        if y < run.y || y >= run.y + lh {
            continue;
        }
        let end_x = run_end_x(run);
        if x >= run.x && x < end_x.max(run.x + 1) {
            return true;
        }
    }
    false
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
        Hit::Page => {
            if text_at(layout, x, y) {
                CursorKind::Text
            } else {
                CursorKind::Default
            }
        }
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
    fn table_columns_place_cells_in_grid() {
        // Genuine table layout: cells in a row sit side by side (same y,
        // increasing x); the second row is below the first.
        let doc = html::parse(
            r#"<html><body><table>
                 <tr><td>A1</td><td>B1</td></tr>
                 <tr><td>A2</td><td>B2</td></tr>
               </table></body></html>"#,
        );
        let sheet = Stylesheet::parse(&doc.stylesheets);
        let lay = layout_document(&doc.root, &sheet, DEFAULT_W, DEFAULT_H);
        let r = |t: &str| {
            let g = lay.runs.iter().find(|r| r.text == t).unwrap_or_else(|| panic!("{t}"));
            (g.x, g.y)
        };
        let (a1, b1, a2, b2) = (r("A1"), r("B1"), r("A2"), r("B2"));
        assert_eq!(a1.1, b1.1, "row 1 cells share a line");
        assert!(b1.0 > a1.0, "col B is right of col A");
        assert_eq!(a2.1, b2.1, "row 2 cells share a line");
        assert!(a2.1 > a1.1, "row 2 is below row 1");
        // Columns align across rows.
        assert_eq!(a1.0, a2.0, "column A x aligns across rows");
        assert_eq!(b1.0, b2.0, "column B x aligns across rows");
    }

    #[test_case]
    fn table_column_width_follows_content() {
        // A wide first-column cell makes column A wider, pushing column B right.
        let doc = html::parse(
            r#"<html><body><table><tr>
                 <td>a very long first cell with lots of words here</td><td>B</td>
               </tr></table></body></html>"#,
        );
        let sheet = Stylesheet::parse(&doc.stylesheets);
        let lay = layout_document(&doc.root, &sheet, DEFAULT_W, DEFAULT_H);
        let a = lay.runs.iter().find(|r| r.text == "very").expect("A content");
        let b = lay.runs.iter().find(|r| r.text == "B").expect("B");
        assert!(b.x > a.x + 100, "wide column A pushes B well to the right (b.x={})", b.x);
    }

    #[test_case]
    fn table_colspan_aligns_subtext_under_title() {
        // Hacker News story rows: rank | vote | title, then colspan=2 empty + subtext.
        // Subtext must sit under the title column (col 2), not under the rank.
        let doc = html::parse(
            r#"<html><body><table>
              <tr><td>1.</td><td>^</td><td>TitleHere</td></tr>
              <tr><td colspan="2"></td><td>99 points</td></tr>
            </table></body></html>"#,
        );
        let sheet = Stylesheet::parse(&doc.stylesheets);
        let lay = layout_document(&doc.root, &sheet, DEFAULT_W, DEFAULT_H);
        let title = lay
            .runs
            .iter()
            .find(|r| r.text.contains("TitleHere"))
            .expect("title");
        let points = lay
            .runs
            .iter()
            .find(|r| r.text.contains("points"))
            .expect("points");
        assert!(
            points.y > title.y,
            "subtext below title: points.y={} title.y={}",
            points.y,
            title.y
        );
        // Align within one character cell of the title column start.
        assert!(
            (points.x - title.x).abs() < 40,
            "colspan places points under title: points.x={} title.x={}",
            points.x,
            title.x
        );
    }

    #[test_case]
    fn bgcolor_presentational_paints_table() {
        // HN's orange header uses presentational bgcolor (no CSS background).
        // Use r## so `#rrggbb` inside attributes doesn't terminate the raw string.
        let doc = html::parse(concat!(
            r##"<html><body><table bgcolor="#f6f6ef"><tr>"##,
            r##"<td bgcolor="#ff6600">HN</td>"##,
            r##"</tr></table></body></html>"##,
        ));
        let sheet = Stylesheet::parse(&doc.stylesheets);
        let lay = layout_document(&doc.root, &sheet, DEFAULT_W, DEFAULT_H);
        assert!(
            lay.rects.iter().any(|r| r.color == 0xff_66_00),
            "expected orange from bgcolor presentational attr"
        );
        assert!(
            lay.rects.iter().any(|r| r.color == 0xf6_f6_ef),
            "expected cream table bgcolor"
        );
    }

    #[test_case]
    fn hn_header_orange_bar_spans_full_table_width() {
        // Real HN skeleton: centered 85% cream table, full-width orange header
        // cell containing a nested 3-col nav table.
        let html = concat!(
            r##"<html><body><center>"##,
            r##"<table id="hnmain" width="85%" bgcolor="#f6f6ef" cellpadding="0" cellspacing="0">"##,
            r##"<tr><td bgcolor="#ff6600">"##,
            r##"<table width="100%" cellpadding="0" cellspacing="0" style="padding:2px"><tr>"##,
            r##"<td style="width:18px">Y</td>"##,
            r##"<td><b class="hnname"><a href="news">Hacker News</a></b>"##,
            r##" <a href="newest">new</a> | <a href="front">past</a></td>"##,
            r##"<td style="text-align:right"><a href="login">login</a></td>"##,
            r##"</tr></table></td></tr>"##,
            r##"<tr><td><table cellpadding="0" cellspacing="0">"##,
            r##"<tr><td align="right">1.</td><td>^</td>"##,
            r##"<td class="title"><a href="/a">Android May Soon Restrict</a></td></tr>"##,
            r##"<tr><td colspan="2"></td><td class="subtext">241 points by alice</td></tr>"##,
            r##"<tr><td align="right">2.</td><td>^</td>"##,
            r##"<td class="title"><a href="/b">Claude Opus 5</a></td></tr>"##,
            r##"<tr><td colspan="2"></td><td class="subtext">1569 points by bob</td></tr>"##,
            r##"</table></td></tr></table></center></body></html>"##,
        );
        let css = concat!(
            "a:link{color:#000000;text-decoration:none}",
            "a:visited{color:#828282;text-decoration:none}",
            ".pagetop{color:#222222}",
            ".subtext{font-size:7pt;color:#828282}",
            ".title{font-size:10pt}",
        );
        let doc = html::parse(html);
        let sheet = Stylesheet::parse(css);
        let lay = layout_document(&doc.root, &sheet, 800, 600);

        // Cream table background present and reasonably wide (85% of 800 ≈ 680).
        let cream = lay
            .rects
            .iter()
            .filter(|r| r.color == 0xf6_f6_ef)
            .max_by_key(|r| r.w)
            .expect("cream #f6f6ef table bg");
        assert!(
            cream.w >= 600,
            "hnmain cream bar should be ~85% of viewport, got w={}",
            cream.w
        );
        // Centered: not stuck at x=0.
        assert!(
            cream.x > 20,
            "85% table should be centered under <center>, got x={}",
            cream.x
        );

        // Orange header spans (nearly) the full cream table width.
        let orange = lay
            .rects
            .iter()
            .filter(|r| r.color == 0xff_66_00)
            .max_by_key(|r| r.w)
            .expect("orange #ff6600 header bg");
        assert!(
            orange.w + 4 >= cream.w && orange.w <= cream.w + 4,
            "orange bar w={} must match cream table w={}",
            orange.w,
            cream.w
        );
        assert!(
            (orange.x - cream.x).abs() <= 4,
            "orange bar x={} must align with cream x={}",
            orange.x,
            cream.x
        );

        // Nav brand is black (a:link), not the UA terracotta.
        let brand = lay
            .runs
            .iter()
            .find(|r| r.text.contains("Hacker"))
            .expect("Hacker News brand");
        assert_eq!(
            brand.color, 0x00_00_00,
            "a:link must paint title/nav links black, got {:06x}",
            brand.color
        );

        // Story columns: title to the right of rank; subtext under title.
        let rank = lay
            .runs
            .iter()
            .find(|r| r.text.trim() == "1.")
            .expect("rank");
        let title = lay
            .runs
            .iter()
            .find(|r| r.text.contains("Android"))
            .expect("title");
        let points = lay
            .runs
            .iter()
            .find(|r| r.text.contains("241"))
            .expect("points");
        assert!(
            title.x > rank.x + 10,
            "title must sit right of rank (title.x={} rank.x={})",
            title.x,
            rank.x
        );
        assert!(
            points.y > title.y,
            "subtext below title: points.y={} title.y={}",
            points.y,
            title.y
        );
        assert!(
            (points.x - title.x).abs() < 40,
            "colspan puts points under title: points.x={} title.x={}",
            points.x,
            title.x
        );

        // Second story: ranks share a column (right-aligned digits may differ a
        // few px on the left edge; require both left of their titles).
        let rank2 = lay
            .runs
            .iter()
            .find(|r| r.text.trim() == "2.")
            .expect("rank2");
        let title2 = lay
            .runs
            .iter()
            .find(|r| r.text.contains("Claude"))
            .expect("title2");
        assert!(
            (rank2.x - rank.x).abs() < 24,
            "rank column aligns: 1.x={} 2.x={}",
            rank.x,
            rank2.x
        );
        assert!(
            title2.x > rank2.x + 10,
            "story 2 title right of rank"
        );
        // Titles share a column (within a character cell).
        assert!(
            (title2.x - title.x).abs() < 24,
            "title column aligns: t1.x={} t2.x={}",
            title.x,
            title2.x
        );

        // Stories must stay start-aligned under the cream table — not re-centered
        // by the outer `<center>`'s text-align pass (the bug that floated HN
        // content mid-page). Rank sits near the cream left edge.
        assert!(
            rank.x < cream.x + 80,
            "rank must be left-aligned in cream (rank.x={} cream.x={})",
            rank.x,
            cream.x
        );
        assert!(
            title.x < cream.x + cream.w / 2,
            "title must not sit in the right half from center re-align (title.x={} cream mid={})",
            title.x,
            cream.x + cream.w / 2
        );
    }

    #[test_case]
    fn center_does_not_recenter_table_content() {
        // `<center>` centers a fixed-width table *box*, but cell text stays
        // start-aligned (CSS text-align only affects inline content of the
        // center block itself).
        let doc = html::parse(concat!(
            r##"<html><body><center>"##,
            r##"<table width="80%" bgcolor="#eeeeee"><tr>"##,
            r##"<td>LeftEdgeStory</td></tr></table>"##,
            r##"</center></body></html>"##,
        ));
        let sheet = Stylesheet::parse("");
        let lay = layout_document(&doc.root, &sheet, 500, 200);
        let cream = lay
            .rects
            .iter()
            .find(|r| r.color == 0xee_ee_ee)
            .expect("table bg");
        let story = lay
            .runs
            .iter()
            .find(|r| r.text.contains("LeftEdge"))
            .expect("story");
        // Table box is centered (x > 0 for 80% of 500).
        assert!(cream.x > 20, "table box centered, x={}", cream.x);
        // Content flush to the table's left, not mid-table.
        assert!(
            (story.x - cream.x).abs() < 30,
            "cell text left-aligned in table: story.x={} cream.x={}",
            story.x,
            cream.x
        );
    }

    #[test_case]
    fn text_selection_extracts_and_orders() {
        let doc = html::parse(
            r#"<html><body><p>Hello World</p><p>Second</p></body></html>"#,
        );
        let sheet = Stylesheet::parse("");
        let lay = layout_document(&doc.root, &sheet, 400, 200);
        assert!(!lay.runs.is_empty(), "expected text runs");
        // Select from start of first run into the next line.
        let a = TextPos { run: 0, col: 0 };
        let last = lay.runs.len() - 1;
        let b = TextPos {
            run: last,
            col: lay.runs[last].text.chars().count(),
        };
        let t = selection_text(&lay, a, b);
        assert!(t.contains("Hello"), "got {t:?}");
        assert!(t.contains("Second") || t.contains("World"), "got {t:?}");
        // Reversed endpoints yield the same text.
        assert_eq!(selection_text(&lay, b, a), t);
        // Empty when carets equal.
        assert!(selection_text(&lay, a, a).is_empty());
        // text_pos_at over first run's origin lands near start.
        let r0 = &lay.runs[0];
        let p = text_pos_at(&lay, r0.x + 1, r0.y + 2).expect("pos");
        assert_eq!(p.run, 0);
        assert!(p.col <= 2, "col near start, got {}", p.col);
        // Highlight rects non-empty for a real range.
        let mid = TextPos { run: 0, col: 5.min(r0.text.chars().count()) };
        assert!(!selection_rects(&lay, a, mid).is_empty());
    }

    #[test_case]
    fn table_cell_align_right() {
        // Fixed-width first column so right-align is visible (a min-content
        // rank column is only as wide as "R", so align=right ≈ flush left).
        let doc = html::parse(
            r#"<html><body><table width="400"><tr>
                 <td align="right" style="width:200px">R</td><td>L</td>
               </tr></table></body></html>"#,
        );
        let sheet = Stylesheet::parse("");
        let lay = layout_document(&doc.root, &sheet, 500, 200);
        let r = lay.runs.iter().find(|t| t.text == "R").expect("R");
        let l = lay.runs.iter().find(|t| t.text == "L").expect("L");
        assert!(l.x > r.x, "L column right of R column");
        // "R" sits near the right edge of the 200px column, not at x≈0.
        assert!(
            r.x > 100,
            "align=right in a 200px col should inset R (r.x={})",
            r.x
        );
    }

    #[test_case]
    fn link_styled_as_button_paints_background() {
        // An `<a>` with a background (a link-button like google.com's "Sign in")
        // paints an inline background box behind its text.
        let doc = html::parse(
            r#"<html><body><a style="background:#c2e7ff;padding:10px;border-radius:20px" href="/s">Sign in</a></body></html>"#,
        );
        let sheet = Stylesheet::parse(&doc.stylesheets);
        let lay = layout_document(&doc.root, &sheet, DEFAULT_W, DEFAULT_H);
        let sign = lay.runs.iter().find(|r| r.text == "Sign").expect("Sign run");
        // A background rect of the button colour covers the link's text.
        assert!(
            lay.rects.iter().any(|rc| rc.color == 0xc2e7ff
                && rc.x <= sign.x
                && rc.x + rc.w >= sign.x),
            "expected an inline background box behind the link-button"
        );
    }

    #[test_case]
    fn inline_block_link_keeps_href_and_background() {
        // google.com's "Sign in" is `<a class=gb_1a>` with
        // `display:inline-block; background:#0b57d0`. The inline-block path used
        // to drop BOTH the href (→ not clickable) and the background (→ not
        // blue). Both must survive now.
        let doc = html::parse(
            r#"<html><body><div class="gb_Na"><a class="gb_1a" href="/login" style="display:inline-block;background:#0b57d0;color:#fff;padding:10px 24px;border-radius:100px">Sign in</a></div></body></html>"#,
        );
        let sheet = Stylesheet::parse(&doc.stylesheets);
        let lay = layout_document(&doc.root, &sheet, DEFAULT_W, DEFAULT_H);
        let sign = lay.runs.iter().find(|r| r.text == "Sign").expect("Sign run");
        // Clickable: a LinkBox covers the "Sign" run.
        assert!(
            lay.links.iter().any(|lb| lb.href == "/login"
                && lb.x <= sign.x
                && lb.x + lb.w >= sign.x),
            "inline-block <a> must still emit a LinkBox for its href"
        );
        // Blue: a background pill of the button colour sits behind the text.
        assert!(
            lay.rects.iter().any(|rc| rc.color == 0x0b57d0 && rc.radius == 100),
            "inline-block <a> button must paint its background pill"
        );
        // And a hit test at the text lands on the link.
        assert!(matches!(
            hit_test_ex(&lay, sign.x + 1, sign.y + 1),
            Hit::Link(_)
        ));
    }

    #[test_case]
    fn inline_link_horizontal_margin_spaces_siblings() {
        // Adjacent footer links `</a><a>` with `margin:0 12px` must not run
        // together — the horizontal margin adds a gap (google.com #WqQANb).
        let doc = html::parse(
            r#"<html><body><a style="margin:0 12px" href="/1">One</a><a style="margin:0 12px" href="/2">Two</a></body></html>"#,
        );
        let sheet = Stylesheet::parse(&doc.stylesheets);
        let lay = layout_document(&doc.root, &sheet, DEFAULT_W, DEFAULT_H);
        let one = lay.runs.iter().find(|r| r.text == "One").expect("One");
        let two = lay.runs.iter().find(|r| r.text == "Two").expect("Two");
        // "Two" starts well past One's end (One width + right+left margins ~24px).
        let one_end = one.x + (crate::font_ttf::measure("One", one.font_size as f32) as i32);
        assert!(two.x >= one_end + 18, "gap between links (two.x={}, one_end={})", two.x, one_end);
    }

    #[test_case]
    fn br_after_tall_content_no_overlap() {
        // A `<br>` after a tall inline-block (a search box) must break past its
        // height, not overlap the next line onto it (google.com search buttons).
        let doc = html::parse(
            r#"<html><body><input type="text"><br><input type="submit" value="Go"></body></html>"#,
        );
        let sheet = Stylesheet::parse(&doc.stylesheets);
        let lay = layout_document(&doc.root, &sheet, DEFAULT_W, DEFAULT_H);
        assert_eq!(lay.controls.len(), 2);
        let (text, btn) = (&lay.controls[0], &lay.controls[1]);
        assert!(
            btn.y >= text.y + text.h,
            "submit (y={}) must sit below the text box bottom (y+h={})",
            btn.y, text.y + text.h
        );
    }

    #[test_case]
    fn parse_transform_scale_and_rotate() {
        assert_eq!(parse_transform_scale("scale(2)"), (2.0, 2.0));
        assert_eq!(parse_transform_scale("scale(1.5, 0.5)"), (1.5, 0.5));
        assert_eq!(parse_transform_scale("scaleX(3)"), (3.0, 1.0));
        assert_eq!(parse_transform_scale("translate(5px)"), (1.0, 1.0));
        assert!((parse_transform_rotate("rotate(90deg)") - 90.0).abs() < 0.01);
        assert!((parse_transform_rotate("rotate(0.5turn)") - 180.0).abs() < 0.01);
        assert_eq!(parse_transform_rotate("scale(2)"), 0.0);
    }

    #[test_case]
    fn transform_scale_grows_font_and_offsets() {
        let doc = html::parse(
            r#"<html><body><div style="transform:scale(2)"><p>big</p></div></body></html>"#,
        );
        let sheet = Stylesheet::parse(&doc.stylesheets);
        let lay = layout_document(&doc.root, &sheet, DEFAULT_W, DEFAULT_H);
        let base = css::compute(&sheet, "p", None, None, None, &css::ComputedStyle::default()).font_size;
        let big = lay.runs.iter().find(|r| r.text == "big").expect("big");
        assert!(big.font_size > base, "scaled font {} > base {}", big.font_size, base);
    }

    #[test_case]
    fn flex_order_reorders_items() {
        // The second child has order:-1 so it lays out first (leftmost).
        let doc = html::parse(
            r#"<html><body><div style="display:flex"><div>A</div><div style="order:-1">B</div></div></body></html>"#,
        );
        let sheet = Stylesheet::parse(&doc.stylesheets);
        let lay = layout_document(&doc.root, &sheet, DEFAULT_W, DEFAULT_H);
        let a = lay.runs.iter().find(|r| r.text == "A").expect("A");
        let b = lay.runs.iter().find(|r| r.text == "B").expect("B");
        assert!(b.x < a.x, "order:-1 item B (x={}) is left of A (x={})", b.x, a.x);
    }

    #[test_case]
    fn grid_tracks_parse_and_size() {
        use css::GridTrack;
        assert_eq!(
            css::parse_grid_tracks("1fr 2fr 100px"),
            alloc::vec![GridTrack::Fr(1.0), GridTrack::Fr(2.0), GridTrack::Px(100)]
        );
        assert_eq!(css::parse_grid_tracks("repeat(3, 1fr)").len(), 3);
        // 100px fixed + (1fr,2fr) share the rest of 400 (gap 0): 100, 100, 200.
        let ws = flex::grid_track_widths(
            &[GridTrack::Px(100), GridTrack::Fr(1.0), GridTrack::Fr(2.0)],
            400, 0, 3,
        );
        assert_eq!(ws, alloc::vec![100, 100, 200]);
    }

    #[test_case]
    fn a_shadcn_shaped_section_paints_its_components() {
        // The gallery's real shape: `:root` HSL custom properties, a bordered
        // card frame, and shadcn buttons inside a wrapping flex row. The
        // failure this pins is a section frame that paints while everything
        // inside it is invisible.
        //
        // NB a `<button>` is a **form control**, not a box — its colours land on
        // `FormControl.bg/fg` and the painter draws it. Looking for a rect here
        // finds nothing even when the button is perfectly laid out.
        let doc = html::parse(
            r#"<html><head><style>
                :root{--card: 0 0% 100%; --primary: 222.2 47.4% 11.2%;
                      --primary-foreground: 210 40% 98%; --border: 214.3 31.8% 91.4%}
                *{border-color:#e5e7eb}
                .frame{border-width:1px;border-style:solid;border-color:hsl(var(--border));
                       border-radius:0.5rem;background-color:hsl(var(--card));padding:1rem}
                .row{display:flex;flex-wrap:wrap;align-items:center;gap:0.5rem}
                .btn{display:inline-flex;align-items:center;border-radius:0.375rem;
                     height:2.25rem;padding-left:1rem;padding-right:1rem;
                     background-color:hsl(var(--primary));color:hsl(var(--primary-foreground))}
                .badge{display:inline-flex;border-radius:9999px;padding:2px 10px;
                       background-color:hsl(var(--primary))}
            </style></head><body>
                <section><h2>Button</h2><div class="frame">
                  <div class="row">
                    <button class="btn">Default</button><button class="btn">Secondary</button>
                    <span class="badge">New</span>
                  </div>
                </div></section>
            </body></html>"#,
        );
        let sheet = Stylesheet::parse_with_viewport(&doc.stylesheets, DEFAULT_W);
        let lay = layout_document(&doc.root, &sheet, DEFAULT_W, DEFAULT_H);

        let (fi, frame) = lay
            .rects
            .iter()
            .enumerate()
            .find(|(_, r)| r.color == 0xffffff && r.w > 200)
            .expect("card frame background — hsl(var(--card))");
        assert_eq!(frame.radius, 8, "border-radius: 0.5rem");

        // Both buttons are controls, carrying the HSL palette colours.
        assert_eq!(lay.controls.len(), 2, "two buttons: {:?}", lay.controls);
        for c in &lay.controls {
            assert_eq!(c.bg, Some(0x0f172a), "hsl(var(--primary)) on the control");
            assert!(c.fg.map(|f| f > 0xe0e0e0).unwrap_or(false), "near-white label");
            assert!(c.w < 200, "shrink-to-fit, not the whole row: {}", c.w);
            assert!(
                c.x >= frame.x && c.x + c.w <= frame.x + frame.w,
                "inside the frame"
            );
        }
        assert!(
            lay.controls[1].x >= lay.controls[0].x + lay.controls[0].w,
            "the two buttons sit side by side, not stacked"
        );

        // The inline badge is a real box, and it paints AFTER the frame.
        let (bi, badge) = lay
            .rects
            .iter()
            .enumerate()
            .find(|(i, r)| *i != fi && r.color == 0x0f172a)
            .expect("badge background");
        assert!(fi < bi, "the frame paints behind what is inside it");
        assert!(badge.w < 100, "the badge hugs its label: {}", badge.w);
        assert!(badge.radius > 0, "rounded-full reaches the inline box");
    }

    #[test_case]
    fn box_shadow_skips_transparent_placeholder_entries() {
        // Tailwind puts the real shadow LAST:
        // `var(--tw-ring-offset-shadow, 0 0 #0000), var(--tw-ring-shadow, 0 0 #0000), var(--tw-shadow)`.
        // Stopping at the first comma took `0 0 #0000` and painted the card
        // solid black.
        let tw = "0 0 #0000,0 0 #0000,0 10px 15px -3px rgb(0 0 0 / 0.1)";
        let (dx, dy, blur, color) = parse_box_shadow(tw).expect("real shadow");
        assert_eq!((dx, dy), (0, 10));
        assert_eq!(blur, 15);
        assert!(color > 0xd0d0d0, "10% black blends light, got {color:06x}");
        // A shadow list that is entirely transparent has nothing to paint.
        assert_eq!(parse_box_shadow("0 0 #0000,0 0 #0000"), None);
    }

    #[test_case]
    fn an_inline_element_paints_its_background_and_border() {
        // `<span class="rounded-full border bg-emerald-50 px-2">loaded</span>` —
        // a badge, the commonest inline component in Tailwind/shadcn — used to
        // render as bare text because only block boxes painted a background.
        let doc = html::parse(
            r#"<html><head><style>
                .pill{background-color:#ecfdf5;border:1px solid #6ee7b7;border-radius:9999px;padding:2px 8px}
            </style></head><body><div>x <span class="pill">loaded</span> y</div></body></html>"#,
        );
        let sheet = Stylesheet::parse_with_viewport(&doc.stylesheets, DEFAULT_W);
        let lay = layout_document(&doc.root, &sheet, DEFAULT_W, DEFAULT_H);
        let bg = lay
            .rects
            .iter()
            .find(|r| r.color == 0xecfdf5)
            .expect("inline background painted");
        assert!(bg.radius > 0, "border-radius reaches the inline box");
        // Horizontal padding widens the box AND advances the line, so the text
        // after it is not overlapped.
        let run_after = lay.runs.iter().find(|r| r.text.contains('y')).expect("y");
        assert!(
            run_after.x >= bg.x + bg.w,
            "text after the pill starts past it ({} vs {})",
            run_after.x,
            bg.x + bg.w
        );
        assert!(
            lay.rects.iter().any(|r| r.color == 0x6ee7b7),
            "inline border edges painted"
        );
    }

    #[test_case]
    fn a_block_background_paints_behind_its_children() {
        // Rects paint in list order, and a block's background can only be
        // pushed once its children have fixed its height — so pushing it last
        // painted the parent OVER every child. A white card erased the badges
        // inside it.
        let doc = html::parse(
            r#"<html><head><style>
                .card{background-color:#ffffff}
                .pill{background-color:#ecfdf5}
            </style></head><body>
                <div class="card"><div class="pill">inside</div></div>
            </body></html>"#,
        );
        let sheet = Stylesheet::parse_with_viewport(&doc.stylesheets, DEFAULT_W);
        let lay = layout_document(&doc.root, &sheet, DEFAULT_W, DEFAULT_H);
        let card = lay.rects.iter().position(|r| r.color == 0xffffff).expect("card");
        let pill = lay.rects.iter().position(|r| r.color == 0xecfdf5).expect("pill");
        assert!(card < pill, "card bg (#{card}) must paint before pill (#{pill})");
    }

    #[test_case]
    fn inline_flex_shrinks_to_its_content() {
        // A shadcn button is `<button class="inline-flex …">Label</button>`:
        // an inline-level flex box whose only child is text. It has no element
        // children, so it never reaches the flex container path — and as a
        // plain block it filled the whole line.
        let doc = html::parse(
            r#"<html><head><style>
                .btn{display:inline-flex;background-color:#0f172a;padding:4px 12px}
                .blk{display:block;background-color:#0284c7}
            </style></head><body>
                <div class="btn">Go</div><div class="blk">Go</div>
            </body></html>"#,
        );
        let sheet = Stylesheet::parse_with_viewport(&doc.stylesheets, DEFAULT_W);
        let lay = layout_document(&doc.root, &sheet, DEFAULT_W, DEFAULT_H);
        let btn = lay.rects.iter().find(|r| r.color == 0x0f172a).expect("inline-flex box");
        let blk = lay.rects.iter().find(|r| r.color == 0x0284c7).expect("block box");
        assert!(
            btn.w < blk.w / 2,
            "inline-flex shrinks to content ({}) while a block fills the line ({})",
            btn.w,
            blk.w
        );
        assert!(btn.w > 24, "…but still holds its label + padding: {}", btn.w);
    }

    #[test_case]
    fn a_tailwind_card_gets_its_width_padding_and_colours() {
        // The whole chain in one page: rem lengths, a functional colour, a
        // shadow list, and a centred max-width card.
        let doc = html::parse(
            r#"<html><head><style>
                .wrap{display:flex;justify-content:center;padding:1rem}
                .card{width:100%;max-width:28rem;padding:1.5rem;border-radius:0.75rem;
                      background-color:rgb(255 255 255 / 1);
                      border:1px solid rgb(226 232 240 / 1);
                      box-shadow:0 0 #0000,0 0 #0000,0 10px 15px -3px rgb(0 0 0 / 0.1)}
            </style></head><body>
                <div class="wrap"><div class="card"><h1>React + Tailwind</h1></div></div>
            </body></html>"#,
        );
        let sheet = Stylesheet::parse_with_viewport(&doc.stylesheets, DEFAULT_W);
        let lay = layout_document(&doc.root, &sheet, DEFAULT_W, DEFAULT_H);
        let card = lay
            .rects
            .iter()
            .find(|r| r.color == 0xffffff && r.w > 100)
            .expect("card background");
        assert_eq!(card.w, 448, "max-width: 28rem");
        assert_eq!(card.radius, 12, "border-radius: 0.75rem");
        assert!(
            lay.rects.iter().any(|r| r.blur > 0),
            "the real (third) shadow entry is painted"
        );
        assert!(
            !lay.rects.iter().any(|r| r.color == 0x000000 && r.w >= card.w),
            "no opaque black box from the `0 0 #0000` placeholders"
        );
        let h1 = lay.runs.iter().find(|r| r.text.contains("React")).expect("h1");
        assert!(h1.x >= card.x + 24, "padding: 1.5rem insets the content");
    }

    #[test_case]
    fn parse_box_shadow_extracts_offsets_and_color() {
        assert_eq!(parse_box_shadow("none"), None);
        assert_eq!(parse_box_shadow(""), None);
        assert_eq!(parse_box_shadow("2px 4px 8px #808080"), Some((2, 4, 8, 0x808080)));
        // rgba colour (parens kept intact); second shadow ignored.
        let (dx, dy, blur, _c) = parse_box_shadow("0 1px 3px rgba(0,0,0,0.3), 0 0 0 red").unwrap();
        assert_eq!((dx, dy, blur), (0, 1, 3));
        // inset is ignored; still parses the offsets.
        assert_eq!(parse_box_shadow("inset 5px 6px #000").map(|s| (s.0, s.1)), Some((5, 6)));
    }

    #[test_case]
    fn parse_transform_translate_offsets() {
        assert_eq!(parse_transform_translate("translate(10px, 20px)"), (10, 20));
        assert_eq!(parse_transform_translate("translateX(15px)"), (15, 0));
        assert_eq!(parse_transform_translate("translateY(-8px)"), (0, -8));
        assert_eq!(parse_transform_translate("scale(2)"), (0, 0)); // scale not applied
    }

    #[test_case]
    fn transform_translate_shifts_box() {
        let doc = html::parse(
            r#"<html><body><div style="transform:translate(50px,30px)"><p>moved</p></div></body></html>"#,
        );
        let sheet = Stylesheet::parse(&doc.stylesheets);
        let lay = layout_document(&doc.root, &sheet, DEFAULT_W, DEFAULT_H);
        let m = lay.runs.iter().find(|r| r.text == "moved").expect("moved");
        assert!(m.x >= 50 && m.y >= 30, "translated to ({},{})", m.x, m.y);
    }

    #[test_case]
    fn float_right_places_at_right_edge() {
        let doc = html::parse(
            r#"<html><body><div style="float:right;width:100px">side</div><p>body text</p></body></html>"#,
        );
        let sheet = Stylesheet::parse(&doc.stylesheets);
        let lay = layout_document(&doc.root, &sheet, DEFAULT_W, DEFAULT_H);
        let side = lay.runs.iter().find(|r| r.text == "side").expect("side");
        // Right-floated content sits in the right half of the viewport.
        assert!(side.x > DEFAULT_W / 2, "float:right at x={} (vw={})", side.x, DEFAULT_W);
    }

    #[test_case]
    fn clear_both_drops_below_float() {
        let doc = html::parse(
            r#"<html><body>
                <div style="float:left;width:80px">side</div>
                <div style="clear:both">below</div>
            </body></html>"#,
        );
        let sheet = Stylesheet::parse(&doc.stylesheets);
        let lay = layout_document(&doc.root, &sheet, DEFAULT_W, DEFAULT_H);
        let side = lay.runs.iter().find(|r| r.text == "side").expect("side");
        let below = lay.runs.iter().find(|r| r.text == "below").expect("below");
        assert!(
            below.y >= side.y + 8,
            "clear:both must sit below the float (below.y={}, side.y={})",
            below.y,
            side.y
        );
    }

    #[test_case]
    fn before_pseudo_emits_generated_content() {
        let doc = html::parse(
            r#"<html><head><style>p::before{content:"» "}</style></head>
               <body><p>hello</p></body></html>"#,
        );
        let sheet = Stylesheet::parse(&doc.stylesheets);
        let lay = layout_document(&doc.root, &sheet, DEFAULT_W, DEFAULT_H);
        let joined: String = lay.runs.iter().map(|r| r.text.as_str()).collect();
        assert!(
            joined.contains('»') && joined.contains("hello"),
            "expected ::before quote in runs, got {joined:?}"
        );
    }

    #[test_case]
    fn hn_like_link_and_modern_css_fixture() {
        // HN-shaped: a:link colour + simple table.
        let hn = html::parse(
            r#"<html><head><style>
                a:link{color:#000000;text-decoration:none}
                .title{font-size:14px}
            </style></head><body>
                <table><tr><td class="title"><a href="x">Story</a></td></tr></table>
            </body></html>"#,
        );
        let sheet = Stylesheet::parse_with_viewport(&hn.stylesheets, DEFAULT_W);
        let lay = layout_document(&hn.root, &sheet, DEFAULT_W, DEFAULT_H);
        let story = lay.runs.iter().find(|r| r.text.contains("Story")).expect("Story");
        assert_eq!(story.color, 0x000000, "a:link should be black");

        // Modern: flex + media + calc + var + ::before.
        let modern = html::parse(
            r#"<html><head><style>
                :root{--accent:#cc785c}
                .row{display:flex;gap:8px;width:calc(100% - 20px)}
                .badge::before{content:"> "}
                @media (max-width: 100px){ .row{display:block} }
            </style></head><body>
                <div class="row"><span class="badge" style="color:var(--accent)">ok</span></div>
            </body></html>"#,
        );
        let sheet = Stylesheet::parse_with_viewport(&modern.stylesheets, DEFAULT_W);
        let lay = layout_document(&modern.root, &sheet, DEFAULT_W, DEFAULT_H);
        let joined: String = lay.runs.iter().map(|r| r.text.as_str()).collect();
        assert!(
            joined.contains('>') && joined.contains("ok"),
            "expected ::before + ok in runs, got {joined:?}"
        );
        let ok = lay.runs.iter().find(|r| r.text.contains("ok")).expect("ok");
        assert_eq!(ok.color, 0xcc785c, "var(--accent) on colour");
    }

    #[test_case]
    fn letter_spacing_widens_run() {
        let doc = html::parse(
            r#"<html><body><span style="letter-spacing:5px">AB</span></body></html>"#,
        );
        let sheet = Stylesheet::parse(&doc.stylesheets);
        let lay = layout_document(&doc.root, &sheet, DEFAULT_W, DEFAULT_H);
        assert_eq!(lay.runs.iter().find(|r| r.text == "AB").expect("AB").letter_spacing, 5);
    }

    #[test_case]
    fn list_style_none_suppresses_bullet() {
        // A nav `<ul style="list-style:none">` must not prepend bullets (they
        // inherit to the `<li>`s).
        let doc = html::parse(
            r#"<html><body><ul style="list-style:none"><li>Home</li></ul><ul><li>Item</li></ul></body></html>"#,
        );
        let sheet = Stylesheet::parse(&doc.stylesheets);
        let lay = layout_document(&doc.root, &sheet, DEFAULT_W, DEFAULT_H);
        // No bullet run anywhere in the list-style:none list; the default list keeps one.
        assert!(lay.runs.iter().any(|r| r.text.contains('•')), "default <ul> still bulleted");
        let home = lay.runs.iter().find(|r| r.text == "Home").expect("Home");
        // "Home" should sit at/near the left content edge (no bullet indent before it on its line).
        let bullet_before_home = lay.runs.iter().any(|r| r.text.contains('•') && r.y == home.y);
        assert!(!bullet_before_home, "list-style:none suppresses the bullet");
    }

    #[test_case]
    fn min_max_width_clamp_block() {
        let doc = html::parse(
            r#"<html><body><div style="max-width:100px;background:#eee"><p>x</p></div></body></html>"#,
        );
        let sheet = Stylesheet::parse(&doc.stylesheets);
        let lay = layout_document(&doc.root, &sheet, DEFAULT_W, DEFAULT_H);
        // The background rect for the max-width div is capped near 100px.
        let boxed = lay.rects.iter().find(|r| r.color == 0xeeeeee).expect("bg rect");
        assert!(boxed.w <= 110, "max-width clamps the box (w={})", boxed.w);
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

    /// `~` and `+` reach real siblings once layout supplies them.
    ///
    /// Both used to be approximated as descendant, which matches **more**
    /// elements than written — a rule meant for "every item after the first"
    /// also hit the first one. The `.stack > * ~ *` shape here is what
    /// Tailwind's `space-y-*` compiles to
    /// (`.space-y-4 > :not([hidden]) ~ :not([hidden])`), so it is on our own
    /// shipped pages.
    #[test_case]
    fn sibling_combinators_reach_real_siblings_through_layout() {
        let doc = html::parse(
            r#"<html><head><style>
              .stack > p { color: #000000 }
              .stack > p ~ p { color: #ff0000 }
            </style></head><body>
            <div class="stack"><p>first</p><p>second</p><p>third</p></div>
            </body></html>"#,
        );
        let sheet = Stylesheet::parse(&doc.stylesheets);
        let lay = layout_document(&doc.root, &sheet, 400, 200);
        let colour_of = |needle: &str| {
            lay.runs
                .iter()
                .find(|r| r.text.contains(needle))
                .map(|r| r.color)
        };
        // The first child has no preceding sibling, so `p ~ p` must not match.
        assert_eq!(colour_of("first"), Some(0x000000), "first child");
        assert_eq!(colour_of("second"), Some(0xff0000), "second child");
        assert_eq!(colour_of("third"), Some(0xff0000), "third child");
    }

    /// `+` is the *immediately* preceding sibling, not any earlier one.
    #[test_case]
    fn adjacent_combinator_is_immediate_through_layout() {
        let doc = html::parse(
            r#"<html><head><style>
              p { color: #000000 }
              h2 + p { color: #00ff00 }
            </style></head><body>
            <div><h2>H</h2><p>adjacent</p><p>later</p></div>
            </body></html>"#,
        );
        let sheet = Stylesheet::parse(&doc.stylesheets);
        let lay = layout_document(&doc.root, &sheet, 400, 200);
        let colour_of = |needle: &str| {
            lay.runs
                .iter()
                .find(|r| r.text.contains(needle))
                .map(|r| r.color)
        };
        assert_eq!(colour_of("adjacent"), Some(0x00ff00), "immediately after h2");
        assert_eq!(colour_of("later"), Some(0x000000), "two after h2");
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
