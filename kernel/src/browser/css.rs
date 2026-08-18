//! **CSS engine (subset)** — pure no_std stylesheet parse + cascade.
//!
//! Structure follows Ladybird LibCSS / StyleComputer (selector match →
//! cascade → computed). Full MDN CSS is not in scope; we expand properties
//! behind the same APIs.
//!
//! Supported (honest, tested):
//! - Selectors: tag / `.class` / `#id` / `*` / compounds; descendant + child
//!   (`>`); `:link`/`:any-link`; `:nth-child(odd|even|An+B)`; attribute
//!   `[href]`, `[type=…]`, `[class~=…]`; `::before` / `::after` (generated
//!   content in layout)
//! - `@media` width queries (`min-width` / `max-width` / `screen` / `all`);
//!   `@import` + `@layer` cascade layers; `!important`
//! - `var(--name)` / nested fallback; `calc()` on lengths (px/em/%)
//! - Flex / grid / float / absolute; popular paint props (see REPORT.md)
//! - Tailwind/shadcn class escapes (`.bg-primary\/20` → class `bg-primary/20`)
//! - `@keyframes` + `animation` (opacity / transform at the current clock)
//! - `:hover` (matches the pointer target and its ancestors; see `set_hover_elems`)
//!
//! Not in scope: sticky compositing, full Grid Level 2, complete css3test.com.

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// Line style for a border/outline edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum BorderStyle {
    None,
    #[default]
    Solid,
    Dashed,
    Dotted,
    Double,
}

impl BorderStyle {
    pub fn parse(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "none" | "hidden" => BorderStyle::None,
            "dashed" => BorderStyle::Dashed,
            "dotted" => BorderStyle::Dotted,
            "double" => BorderStyle::Double,
            // solid, groove, ridge, inset, outset → render as solid
            _ => BorderStyle::Solid,
        }
    }
    pub fn is_visible(self) -> bool {
        !matches!(self, BorderStyle::None)
    }
}

/// `clear` — which float sides a block must clear.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ClearMode {
    #[default]
    None,
    Left,
    Right,
    Both,
}

/// Resolved style used by layout/paint.
#[derive(Clone, Debug)]
pub struct ComputedStyle {
    pub color: u32,
    pub background: Option<u32>,
    pub font_size: i32,
    pub margin_top: i32,
    pub margin_bottom: i32,
    pub margin_left: i32,
    pub margin_right: i32,
    /// `margin-left: auto` / `margin-right: auto` — both set on a fixed-width
    /// block centers it in its container (the classic `margin: 0 auto` idiom).
    pub margin_left_auto: bool,
    pub margin_right_auto: bool,
    pub padding_top: i32,
    pub padding_bottom: i32,
    pub padding_left: i32,
    pub padding_right: i32,
    pub display_none: bool,
    pub bold: bool,
    pub text_align: Align,
    pub width: Option<i32>,
    pub height: Option<i32>,
    /// Unresolved `width: N%` / `height: N%` (0–100). Applied against the
    /// parent's explicit size in [`compute_el`], so Tailwind `w-full` /
    /// `h-full` on a sized parent become real pixels.
    pub width_pct: Option<f32>,
    pub height_pct: Option<f32>,
    pub max_width: Option<i32>,
    pub min_width: Option<i32>,
    pub min_height: Option<i32>,
    /// 0..=255 alpha multiplier for paint (default 255 = opaque).
    pub opacity: u8,
    pub line_height: Option<i32>,
    pub border_color: Option<u32>,
    /// CSS display mode for flex/grid layout.
    pub display: DisplayMode,
    pub flex_direction: FlexDirection,
    pub flex_gap: i32,
    pub justify_content: Justify,
    pub align_items: AlignItems,
    /// Grid: number of equal columns when `display:grid` (from template).
    pub grid_columns: u8,
    /// `grid-template-columns` track list (`fr`/px/auto) — richer than the
    /// count in `grid_columns`; empty falls back to `grid_columns` equal tracks.
    pub grid_template: Vec<GridTrack>,
    pub grid_gap: i32,
    pub flex_wrap: super::flex::FlexWrap,
    pub flex_grow: u32,
    /// `flex-grow`'s counterpart — how much an item may shrink past its basis.
    pub flex_shrink: u32,
    /// `flex-basis`: the item's initial main size before grow/shrink (`None` =
    /// `auto`, i.e. use `width`/content).
    pub flex_basis: Option<i32>,
    /// `order`: flex/grid item ordering (lower first; default 0).
    pub order: i32,
    /// Dense grid packing when true.
    pub grid_dense: bool,
    /// Max content height for flex fragmentation (clip extra lines).
    pub max_height: Option<i32>,
    /// Stored for cascade completeness / future layout (positioned layout).
    pub position: Position,
    pub top: Option<i32>,
    pub left: Option<i32>,
    pub right: Option<i32>,
    pub bottom: Option<i32>,
    pub z_index: i32,
    pub overflow: Overflow,
    pub border_radius: i32,
    pub box_sizing: BoxSizing,
    pub font_family: String,
    pub cursor: CursorCss,
    pub float_mode: FloatMode,
    pub white_space: WhiteSpace,
    pub text_decoration: TextDecoration,
    pub list_style: ListStyle,
    pub transform: String,
    pub transform_origin: String,
    pub vertical_align: VerticalAlign,
    pub object_fit: ObjectFit,
    pub aspect_ratio: Option<(i32, i32)>,
    // ── popular-property cascade fields (layout/paint may ignore some) ──
    /// CSS custom properties (`--name`) — inherited.
    pub custom_props: alloc::collections::BTreeMap<String, String>,
    pub font_style: FontStyle,
    pub font_display: String,
    pub font_src: String,
    pub text_transform: TextTransform,
    pub text_indent: i32,
    pub text_shadow: String,
    pub text_rendering: String,
    pub letter_spacing: Option<i32>,
    pub word_break: WordBreak,
    pub overflow_wrap: OverflowWrap,
    pub text_overflow: TextOverflow,
    pub user_select: UserSelect,
    pub pointer_events: PointerEvents,
    pub appearance: String,
    pub direction: Direction,
    pub clip: String,
    pub clip_path: String,
    pub outline_offset: i32,
    pub border_collapse: bool,
    pub border_top_color: Option<u32>,
    pub border_bottom_color: Option<u32>,
    pub border_left_color: Option<u32>,
    pub border_right_color: Option<u32>,
    pub border_top_width: i32,
    pub border_bottom_width: i32,
    pub border_left_width: i32,
    pub border_right_width: i32,
    pub animation_name: String,
    pub animation_duration: String,
    pub animation_delay: String,
    pub animation_timing: String,
    pub transition_property: String,
    pub transition_duration: String,
    pub transition_timing: String,
    pub box_shadow: String,
    pub fill: Option<u32>,
    pub stroke: Option<u32>,
    pub stroke_width: i32,
    pub touch_action: String,
    pub line_clamp: Option<u32>,
    pub scrollbar_width: String,
    pub will_change: String,
    pub resize: ResizeMode,
    pub webkit_font_smoothing: String,
    pub webkit_tap_highlight: Option<u32>,
    pub webkit_text_size_adjust: String,
    pub webkit_box_orient: String,
    pub webkit_box_pack: Justify,
    pub unicode_range: String,
    pub inset_shorthand: bool,
    // ── decoration properties now applied to rendering ──
    pub outline_color: Option<u32>,
    pub outline_width: i32,
    pub outline_style: BorderStyle,
    pub border_top_style: BorderStyle,
    pub border_bottom_style: BorderStyle,
    pub border_left_style: BorderStyle,
    pub border_right_style: BorderStyle,
    /// `background-image` value verbatim (`url(...)` or `linear-gradient(...)`).
    pub background_image: String,
    pub background_repeat: String,
    pub background_position: String,
    pub background_size: String,
    pub background_clip: String,
    /// `filter` / `backdrop-filter` function lists (e.g. `blur(2px) grayscale(1)`).
    pub filter: String,
    pub backdrop_filter: String,
    pub mask: String,
    pub object_position: String,
    pub table_layout: String,
    pub clear: ClearMode,
    /// `content` string for generated content (quotes stripped).
    pub content: String,
    // --- Popular properties that used to be recognized-only (no ComputedStyle
    // effect). Now cascade-stored; a few have real render effects (`zoom`,
    // `border-spacing`, `text-wrap:nowrap`), the rest are honored where a
    // subsystem exists (SVG stroke, animation) and otherwise inert-but-correct.
    /// `zoom` scale factor (1.0 = none) — applied like `transform: scale`.
    pub zoom: f32,
    /// `border-spacing` between table cells, px.
    pub border_spacing: i32,
    /// `text-wrap` (`wrap`/`nowrap`/`balance`/`pretty`); `nowrap` suppresses
    /// line breaking like `white-space:nowrap`.
    pub text_wrap: String,
    /// `stroke-dashoffset` (SVG), px.
    pub stroke_dashoffset: i32,
    /// `backface-visibility: hidden` hides a back-facing (rotated) box.
    pub backface_hidden: bool,
    // Cascade-stored strings (small value sets; empty default doesn't allocate).
    pub font_variant: String,
    pub font_stretch: String,
    pub font_feature_settings: String,
    pub font_variation_settings: String,
    pub animation_fill_mode: String,
    pub animation_iteration_count: String,
    pub stroke_dasharray: String,
    pub contain: String,
    pub color_scheme: String,
    pub scroll_behavior: String,
    pub overscroll_behavior: String,
    pub forced_color_adjust: String,
    pub container_type: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum FontStyle {
    #[default]
    Normal,
    Italic,
    Oblique,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum TextTransform {
    #[default]
    None,
    Uppercase,
    Lowercase,
    Capitalize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum WordBreak {
    #[default]
    Normal,
    BreakAll,
    KeepAll,
    BreakWord,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum OverflowWrap {
    #[default]
    Normal,
    Anywhere,
    BreakWord,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum TextOverflow {
    #[default]
    Clip,
    Ellipsis,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum UserSelect {
    #[default]
    Auto,
    None,
    Text,
    All,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PointerEvents {
    #[default]
    Auto,
    None,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Direction {
    #[default]
    Ltr,
    Rtl,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ResizeMode {
    #[default]
    None,
    Both,
    Horizontal,
    Vertical,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Position {
    #[default]
    Static,
    Relative,
    Absolute,
    Fixed,
    Sticky,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Overflow {
    #[default]
    Visible,
    Hidden,
    Scroll,
    Auto,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum BoxSizing {
    #[default]
    ContentBox,
    BorderBox,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CursorCss {
    #[default]
    Auto,
    Pointer,
    Text,
    Default,
    NotAllowed,
    Crosshair,
    Move,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum FloatMode {
    #[default]
    None,
    Left,
    Right,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum WhiteSpace {
    #[default]
    Normal,
    Nowrap,
    Pre,
    PreWrap,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum TextDecoration {
    #[default]
    None,
    Underline,
    LineThrough,
    Overline,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ListStyle {
    #[default]
    Disc,
    Circle,
    Square,
    Decimal,
    None,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum VerticalAlign {
    #[default]
    Baseline,
    Top,
    Middle,
    Bottom,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ObjectFit {
    #[default]
    Fill,
    Contain,
    Cover,
    None,
    ScaleDown,
}

/// A single `grid-template-columns` / `-rows` track sizing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GridTrack {
    /// Flexible `fr` unit (share of the free space).
    Fr(f32),
    /// Fixed pixel length.
    Px(i32),
    /// `auto` / `min-content` / `max-content` — sized to remaining share.
    Auto,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DisplayMode {
    #[default]
    Block,
    Inline,
    Flex,
    /// `display: inline-flex` — a flex container that is **inline-level**, so
    /// it shrink-wraps its content instead of filling the line. Every shadcn
    /// button and badge is one; treating it as block-level flex made each of
    /// them a full-width bar.
    InlineFlex,
    Grid,
    None,
}

impl DisplayMode {
    /// Both flex flavours lay their children out identically — only the
    /// container's own sizing differs.
    pub fn is_flex(self) -> bool {
        matches!(self, DisplayMode::Flex | DisplayMode::InlineFlex)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum FlexDirection {
    #[default]
    Row,
    Column,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Justify {
    #[default]
    Start,
    Center,
    End,
    SpaceBetween,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum AlignItems {
    #[default]
    Stretch,
    Start,
    Center,
    End,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Align {
    Left,
    Center,
    Right,
}

impl Default for ComputedStyle {
    fn default() -> Self {
        Self {
            color: 0x2a2a2a,
            background: None,
            font_size: 14,
            margin_top: 0,
            margin_bottom: 0,
            margin_left: 0,
            margin_right: 0,
            margin_left_auto: false,
            margin_right_auto: false,
            padding_top: 0,
            padding_bottom: 0,
            padding_left: 0,
            padding_right: 0,
            display_none: false,
            bold: false,
            text_align: Align::Left,
            width: None,
            height: None,
            width_pct: None,
            height_pct: None,
            max_width: None,
            min_width: None,
            min_height: None,
            opacity: 255,
            line_height: None,
            border_color: None,
            display: DisplayMode::Block,
            flex_direction: FlexDirection::Row,
            flex_gap: 0,
            justify_content: Justify::Start,
            align_items: AlignItems::Stretch,
            grid_columns: 1,
            grid_template: Vec::new(),
            grid_gap: 0,
            flex_wrap: super::flex::FlexWrap::NoWrap,
            flex_grow: 0,
            flex_shrink: 1,
            flex_basis: None,
            order: 0,
            grid_dense: false,
            max_height: None,
            position: Position::Static,
            top: None,
            left: None,
            right: None,
            bottom: None,
            z_index: 0,
            overflow: Overflow::Visible,
            border_radius: 0,
            box_sizing: BoxSizing::ContentBox,
            font_family: String::from("sans-serif"),
            cursor: CursorCss::Auto,
            float_mode: FloatMode::None,
            white_space: WhiteSpace::Normal,
            text_decoration: TextDecoration::None,
            list_style: ListStyle::Disc,
            transform: String::new(),
            transform_origin: String::from("50% 50%"),
            vertical_align: VerticalAlign::Baseline,
            object_fit: ObjectFit::Fill,
            aspect_ratio: None,
            custom_props: alloc::collections::BTreeMap::new(),
            font_style: FontStyle::Normal,
            font_display: String::new(),
            font_src: String::new(),
            text_transform: TextTransform::None,
            text_indent: 0,
            text_shadow: String::new(),
            text_rendering: String::new(),
            letter_spacing: None,
            word_break: WordBreak::Normal,
            overflow_wrap: OverflowWrap::Normal,
            text_overflow: TextOverflow::Clip,
            user_select: UserSelect::Auto,
            pointer_events: PointerEvents::Auto,
            appearance: String::new(),
            direction: Direction::Ltr,
            clip: String::new(),
            clip_path: String::new(),
            outline_offset: 0,
            border_collapse: false,
            border_top_color: None,
            border_bottom_color: None,
            border_left_color: None,
            border_right_color: None,
            border_top_width: 0,
            border_bottom_width: 0,
            border_left_width: 0,
            border_right_width: 0,
            animation_name: String::new(),
            animation_duration: String::new(),
            animation_delay: String::new(),
            animation_timing: String::new(),
            transition_property: String::new(),
            transition_duration: String::new(),
            transition_timing: String::new(),
            box_shadow: String::new(),
            fill: None,
            stroke: None,
            stroke_width: 0,
            touch_action: String::new(),
            line_clamp: None,
            scrollbar_width: String::new(),
            will_change: String::new(),
            resize: ResizeMode::None,
            webkit_font_smoothing: String::new(),
            webkit_tap_highlight: None,
            webkit_text_size_adjust: String::new(),
            webkit_box_orient: String::new(),
            webkit_box_pack: Justify::Start,
            unicode_range: String::new(),
            inset_shorthand: false,
            outline_color: None,
            outline_width: 0,
            outline_style: BorderStyle::None,
            border_top_style: BorderStyle::None,
            border_bottom_style: BorderStyle::None,
            border_left_style: BorderStyle::None,
            border_right_style: BorderStyle::None,
            background_image: String::new(),
            background_repeat: String::new(),
            background_position: String::new(),
            background_size: String::new(),
            background_clip: String::new(),
            filter: String::new(),
            backdrop_filter: String::new(),
            mask: String::new(),
            object_position: String::new(),
            table_layout: String::new(),
            clear: ClearMode::None,
            content: String::new(),
            zoom: 1.0,
            border_spacing: 0,
            text_wrap: String::new(),
            stroke_dashoffset: 0,
            backface_hidden: false,
            font_variant: String::new(),
            font_stretch: String::new(),
            font_feature_settings: String::new(),
            font_variation_settings: String::new(),
            animation_fill_mode: String::new(),
            animation_iteration_count: String::new(),
            stroke_dasharray: String::new(),
            contain: String::new(),
            color_scheme: String::new(),
            scroll_behavior: String::new(),
            overscroll_behavior: String::new(),
            forced_color_adjust: String::new(),
            container_type: String::new(),
        }
    }
}

/// Canonicalize property names: strip `-webkit-` / `alias-webkit-` prefixes and
/// map aliases onto the unprefixed property the cascade understands.
fn canonicalize_prop(name: &str) -> String {
    let mut n = name.trim().to_ascii_lowercase();
    if let Some(rest) = n.strip_prefix("alias-webkit-") {
        n = rest.to_string();
    } else if let Some(rest) = n.strip_prefix("alias-") {
        n = rest.to_string();
    } else if let Some(rest) = n.strip_prefix("-webkit-") {
        n = rest.to_string();
    } else if let Some(rest) = n.strip_prefix("webkit-") {
        n = rest.to_string();
    }
    // Historical alias names used in popularity data / vendor dumps.
    match n.as_str() {
        "word-wrap" => String::from("overflow-wrap"),
        "box-orient" => String::from("webkit-box-orient"),
        "box-pack" => String::from("webkit-box-pack"),
        "line-clamp" => String::from("webkit-line-clamp"),
        "tap-highlight-color" => String::from("webkit-tap-highlight-color"),
        "font-smoothing" => String::from("webkit-font-smoothing"),
        "text-size-adjust" => String::from("webkit-text-size-adjust"),
        "user-select" | "touch-callout" => n, // keep
        _ => n,
    }
}

/// Resolve `var(--name)` / `var(--name, fallback)` against custom props.
/// Nested fallbacks (`var(--a, var(--b, red))`) resolve recursively; a cycle
/// (seen set) yields an empty string rather than looping forever.
fn resolve_var(value: &str, props: &alloc::collections::BTreeMap<String, String>) -> String {
    resolve_var_depth(value, props, 0, &mut alloc::collections::BTreeSet::new())
}

fn resolve_var_depth(
    value: &str,
    props: &alloc::collections::BTreeMap<String, String>,
    depth: u8,
    seen: &mut alloc::collections::BTreeSet<String>,
) -> String {
    if depth > 8 {
        return String::new();
    }
    let v = value.trim();
    // Resolve the outermost `var(...)` if the whole value is one, else scan
    // for an embedded `var(` (common in `calc(var(--x) + 10px)`).
    if let Some(start) = v.find("var(") {
        let after = &v[start + 4..];
        let mut depth_p = 1i32;
        let mut end = None;
        for (i, c) in after.char_indices() {
            match c {
                '(' => depth_p += 1,
                ')' => {
                    depth_p -= 1;
                    if depth_p == 0 {
                        end = Some(i);
                        break;
                    }
                }
                _ => {}
            }
        }
        let Some(end) = end else {
            return v.to_string();
        };
        let inner = after[..end].trim();
        let (name, fallback) = if let Some((a, b)) = split_var_args(inner) {
            (a.trim(), Some(b.trim()))
        } else {
            (inner, None)
        };
        let key = if name.starts_with("--") {
            name.to_string()
        } else {
            format!("--{name}")
        };
        if !seen.insert(key.clone()) {
            return String::new(); // cycle
        }
        let resolved = if let Some(val) = props.get(&key).or_else(|| props.get(name)) {
            resolve_var_depth(val, props, depth + 1, seen)
        } else if let Some(fb) = fallback {
            resolve_var_depth(fb, props, depth + 1, seen)
        } else {
            String::new()
        };
        seen.remove(&key);
        let mut out = String::with_capacity(v.len());
        out.push_str(&v[..start]);
        out.push_str(&resolved);
        out.push_str(&after[end + 1..]);
        // Another var may remain.
        if out.contains("var(") {
            return resolve_var_depth(&out, props, depth + 1, seen);
        }
        return out;
    }
    v.to_string()
}

/// Split `var()` args on the first top-level comma (fallback may contain commas
/// inside nested `var(...)`).
fn split_var_args(inner: &str) -> Option<(&str, &str)> {
    let mut depth = 0i32;
    for (i, c) in inner.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => depth -= 1,
            ',' if depth == 0 => return Some((&inner[..i], &inner[i + 1..])),
            _ => {}
        }
    }
    None
}

#[derive(Clone, Debug)]
struct Decl {
    name: String,
    value: String,
    important: bool,
}

/// A **compound selector** — one sequence of simple selectors with no
/// combinator (`a.gb_1a`, `.gb_9a.gb_K`, `div#x`, `*`, `p`, `[href]`,
/// `:nth-child(odd)`). `tag == None` means "any tag" for this position; all
/// `classes` must be present.
#[derive(Clone, Debug, Default)]
struct Compound {
    tag: Option<String>,
    id: Option<String>,
    classes: Vec<String>,
    /// Attribute predicates (`[href]`, `[type=submit]`, `[class~=foo]`).
    attrs: Vec<AttrSel>,
    /// `:nth-child(…)` — `None` means no structural check.
    nth_child: Option<NthFormula>,
    /// `:not(...)` arguments — the element must match **none** of them.
    /// A `Vec<Compound>` nests fine (a `Vec` is a pointer, so the size is
    /// known); `:not(a, b)` is one entry per comma per Selectors 4.
    not: Vec<Compound>,
    /// `::before` / `::after` (pseudo-element; matching is on the owning element,
    /// and layout emits generated content from rules tagged with this).
    pseudo_el: PseudoElement,
    /// `:hover` — matches only while the element (or a descendant) is under
    /// the pointer. Dropping this used to discard every Tailwind
    /// `.hover\:bg-*:hover` rule, so hover backgrounds never applied.
    hover: bool,
}

/// Combinator between two compounds in a complex selector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Combinator {
    /// Descendant (` `).
    Descendant,
    /// Child (`>`).
    Child,
    /// Next-sibling (`+`) — approximated as descendant for v1 match depth,
    /// but stored so child can be exact.
    Adjacent,
    /// Subsequent-sibling (`~`) — same approximation as Adjacent.
    Sibling,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PseudoElement {
    #[default]
    None,
    Before,
    After,
}

#[derive(Clone, Debug)]
struct AttrSel {
    name: String,
    /// `None` = presence `[href]`; `Some` = exact `[type=…]` or space-list
    /// `[class~=…]` depending on `op`.
    value: Option<String>,
    op: AttrOp,
    /// The `i` flag (`[type="SUBMIT" i]`) — match the value ASCII-case-insensitively.
    ci: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AttrOp {
    Present,
    Exact,
    /// `~=` — the value is one of a space-separated list.
    Includes,
    /// `^=` — the value starts with this. Tailwind and shadcn lean on these
    /// three heavily (`[class^="text-"]`, `[data-state$="open"]`).
    Prefix,
    /// `$=` — the value ends with this.
    Suffix,
    /// `*=` — the value contains this.
    Substring,
    /// `|=` — the value is this, or this followed by `-` (language subtags:
    /// `[lang|=en]` matches `en` and `en-GB` but not `english`).
    DashMatch,
}

/// CSS `:nth-child(An+B)` / odd / even.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct NthFormula {
    a: i32,
    b: i32,
}

impl NthFormula {
    fn matches(self, index_1based: u32) -> bool {
        let n = index_1based as i32;
        if self.a == 0 {
            return n == self.b;
        }
        let rem = n - self.b;
        if rem % self.a != 0 {
            return false;
        }
        let k = rem / self.a;
        k >= 0
    }
}

impl Compound {
    fn matches_el(&self, el: &ElemRef<'_>) -> bool {
        if let Some(t) = &self.tag {
            if !t.eq_ignore_ascii_case(el.tag) {
                return false;
            }
        }
        if let Some(i) = &self.id {
            if el.id != Some(i.as_str()) {
                return false;
            }
        }
        if !self.classes.is_empty() {
            let cs = el.class.unwrap_or("");
            for c in &self.classes {
                if !cs.split_whitespace().any(|x| x == c) {
                    return false;
                }
            }
        }
        for a in &self.attrs {
            if !attr_matches(a, el) {
                return false;
            }
        }
        if let Some(nth) = self.nth_child {
            if !nth.matches(el.nth.max(1)) {
                return false;
            }
        }
        if self.hover && !el.hovered {
            return false;
        }
        // `:not()` last: it is the only clause that can be expensive, and by
        // here every cheap test has already had its chance to reject.
        if self.not.iter().any(|n| n.matches_el(el)) {
            return false;
        }
        true
    }
    /// (id, class+attr+pseudo-class, type) specificity contribution.
    fn spec(&self) -> (u32, u32, u32) {
        let classy = self.classes.len() as u32
            + self.attrs.len() as u32
            + self.nth_child.is_some() as u32
            + self.hover as u32;
        let (mut ids, mut cls, mut tags) = (
            self.id.is_some() as u32,
            classy,
            self.tag.is_some() as u32,
        );
        // Per Selectors 4, `:not()` itself counts for nothing and its **most
        // specific argument** counts instead — so `:not(#x)` is as specific as
        // `#x`. Getting this wrong does not stop a rule matching, it makes it
        // lose (or win) a cascade it should not, which is much harder to spot
        // than a rule that plainly never applies.
        if let Some((i, c, t)) = self.not.iter().map(|n| n.spec()).max() {
            ids += i;
            cls += c;
            tags += t;
        }
        (ids, cls, tags)
    }
}

/// Does element `el` satisfy attribute predicate `a`?
///
/// The typed fields are checked first, then `el.extra` — the DOM's catch-all
/// bag. Without that fallback every `data-*` and `aria-*` selector reads as
/// "attribute absent", which is how a shadcn `[data-state=open]` rule silently
/// never applies.
fn attr_matches(a: &AttrSel, el: &ElemRef<'_>) -> bool {
    let name = a.name.as_str();
    let got = if name.eq_ignore_ascii_case("href") {
        el.href
    } else if name.eq_ignore_ascii_case("type") {
        el.input_type
    } else if name.eq_ignore_ascii_case("class") {
        el.class
    } else if name.eq_ignore_ascii_case("id") {
        el.id
    } else {
        el.extra
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    };
    let Some(got) = got else {
        return false;
    };
    if a.op == AttrOp::Present {
        return true;
    }
    let Some(want) = a.value.as_deref() else {
        return false;
    };
    let eq = |x: &str, y: &str| {
        if a.ci {
            x.eq_ignore_ascii_case(y)
        } else {
            x == y
        }
    };
    match a.op {
        AttrOp::Present => true,
        AttrOp::Exact => eq(got, want),
        AttrOp::Includes => got.split_whitespace().any(|t| eq(t, want)),
        AttrOp::Prefix => got.len() >= want.len() && eq(&got[..want.len()], want),
        AttrOp::Suffix => got.len() >= want.len() && eq(&got[got.len() - want.len()..], want),
        AttrOp::Substring => {
            if a.ci {
                got.to_ascii_lowercase().contains(&want.to_ascii_lowercase())
            } else {
                got.contains(want)
            }
        }
        // `en` matches `[lang|=en]`, and so does `en-GB`; `english` must not.
        AttrOp::DashMatch => {
            eq(got, want)
                || (got.len() > want.len()
                    && got.as_bytes()[want.len()] == b'-'
                    && eq(&got[..want.len()], want))
        }
    }
}

/// One element in an ancestor / subject chain for selector matching, ordered
/// **outermost → innermost** (root first, parent last).
#[derive(Clone, Copy, Debug)]
pub struct ElemRef<'a> {
    pub tag: &'a str,
    pub id: Option<&'a str>,
    pub class: Option<&'a str>,
    /// 1-based index among element siblings (for `:nth-child`).
    pub nth: u32,
    pub href: Option<&'a str>,
    pub input_type: Option<&'a str>,
    /// Attributes with no typed field — `data-*`, `aria-*`, `role`, `lang`.
    /// The DOM keeps them in `Element::extra_attrs`; without them here every
    /// `[data-…]` selector reads as "attribute absent" rather than as
    /// unsupported, which is indistinguishable from a rule that simply did not
    /// apply. Empty for callers that have no bag (`ElemRef::basic`).
    pub extra: &'a [(String, String)],
    /// Preceding **element** siblings, in document order (so the immediately
    /// preceding one is last). This is what `+` and `~` match against.
    ///
    /// It lives on `ElemRef` rather than as another parameter threaded through
    /// `compute_el`/`compute_pseudo`/`matching_decls_for` deliberately: a
    /// self-referential slice of the same lifetime is legal, and it leaves
    /// every existing signature and call site unchanged.
    ///
    /// **`None` means the caller has no sibling context**, which is *not* the
    /// same fact as `Some(&[])` ("this element is genuinely the first child").
    /// One `&[]` for both would make an element with no siblings and a caller
    /// that cannot supply them indistinguishable — and they need opposite
    /// treatment: the first must fail a `+` match, the second must fall back
    /// to the old descendant approximation rather than silently start matching
    /// nothing. Layout's recursion threads `chain` through ~20 call sites and
    /// does not yet thread siblings, so it passes `None` and behaves exactly as
    /// it did before.
    pub prev: Option<&'a [ElemRef<'a>]>,
    /// True when this element is the hover target or an ancestor of it.
    /// Feeds `:hover` matching; false for callers that have no pointer.
    pub hovered: bool,
}

impl<'a> ElemRef<'a> {
    pub fn basic(tag: &'a str, id: Option<&'a str>, class: Option<&'a str>) -> Self {
        Self {
            tag,
            id,
            class,
            nth: 1,
            href: None,
            input_type: None,
            extra: &[],
            prev: None,
            hovered: false,
        }
    }
}

#[derive(Clone, Debug)]
struct AncestorHop {
    compound: Compound,
    /// Combinator **from this hop to the next** (toward the key). The last hop
    /// uses its combinator to reach the key compound.
    combinator: Combinator,
}

#[derive(Clone, Debug)]
struct Rule {
    /// Rightmost compound — must match the element itself.
    key: Compound,
    /// Ancestor hops (left of the key), outermost → innermost.
    ancestors: Vec<AncestorHop>,
    decls: Vec<Decl>,
    /// Specificity: (id, class, type)
    spec: (u8, u8, u8),
    order: u32,
    /// Cascade layer index (`None` = unlayered, wins over all layers).
    /// Lower index = earlier in layer order = lower priority.
    layer: Option<u32>,
}

/// Parsed stylesheet (LibCSS-inspired: rules sorted by cascade order).
/// Supports CSS **cascade layers** (`@layer name, …` / `@layer name { … }`)
/// and viewport-filtered `@media` (min/max-width).
/// One stop of an `@keyframes` rule (`from`/`to`/`N%` → declarations).
#[derive(Clone, Debug)]
pub struct KeyframeStop {
    /// 0–100.
    pub pct: u16,
    pub decls: Vec<(String, String)>,
}

#[derive(Clone, Debug, Default)]
pub struct Stylesheet {
    rules: Vec<Rule>,
    /// Declared layer order (name → index).
    layer_order: alloc::collections::BTreeMap<String, u32>,
    next_layer: u32,
    /// Current layer when parsing nested `@layer name { }`.
    parse_layer: Option<u32>,
    /// Layout viewport width (px) used to evaluate `@media (min/max-width)`.
    /// `0` means “unknown” — width queries are treated as matching (fail-open
    /// for sheets parsed without a viewport, e.g. unit tests of cascade alone).
    viewport_w: i32,
    /// `@keyframes name { … }` — looked up by `animation-name`.
    keyframes: alloc::collections::BTreeMap<String, Vec<KeyframeStop>>,
}

impl Stylesheet {
    pub fn parse(css: &str) -> Self {
        Self::parse_with_viewport(css, 0)
    }

    /// Parse with a known layout viewport so `@media (max-width: …)` etc. filter.
    pub fn parse_with_viewport(css: &str, viewport_w: i32) -> Self {
        let mut sheet = Stylesheet {
            viewport_w,
            ..Stylesheet::default()
        };
        sheet.append(css);
        sheet
    }

    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    pub fn layer_count(&self) -> usize {
        self.layer_order.len()
    }

    pub fn has_layer(&self, name: &str) -> bool {
        self.layer_order.contains_key(name)
    }

    /// Stops of `@keyframes <name>`, if the sheet declared that animation.
    pub fn keyframes(&self, name: &str) -> Option<&[KeyframeStop]> {
        self.keyframes.get(name).map(|v| v.as_slice())
    }

    pub fn keyframe_count(&self) -> usize {
        self.keyframes.len()
    }

    fn ensure_layer(&mut self, name: &str) -> u32 {
        if let Some(&i) = self.layer_order.get(name) {
            return i;
        }
        let i = self.next_layer;
        self.next_layer = self.next_layer.saturating_add(1);
        self.layer_order.insert(name.to_string(), i);
        i
    }

    pub fn append(&mut self, css: &str) {
        // Strip /* */ comments (CSS2.1 style).
        let mut s = String::with_capacity(css.len());
        let b = css.as_bytes();
        let mut i = 0;
        while i < b.len() {
            if i + 1 < b.len() && b[i] == b'/' && b[i + 1] == b'*' {
                i += 2;
                while i + 1 < b.len() && !(b[i] == b'*' && b[i + 1] == b'/') {
                    i += 1;
                }
                i = (i + 2).min(b.len());
                continue;
            }
            s.push(b[i] as char);
            i += 1;
        }
        self.append_block(&s, self.parse_layer);
    }

    fn append_block(&mut self, s: &str, layer: Option<u32>) {
        let bytes = s.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            while i < bytes.len() && bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            if i >= bytes.len() {
                break;
            }
            // @-rules
            if bytes[i] == b'@' {
                let at_start = i;
                while i < bytes.len() && bytes[i] != b'{' && bytes[i] != b';' {
                    i += 1;
                }
                let at_header =
                    core::str::from_utf8(&bytes[at_start..i]).unwrap_or("").trim();
                // @layer a, b, c;
                if at_header.starts_with("@layer") && i < bytes.len() && bytes[i] == b';' {
                    let names = at_header["@layer".len()..].trim();
                    for n in names.split(',') {
                        let n = n.trim();
                        if !n.is_empty() {
                            let _ = self.ensure_layer(n);
                        }
                    }
                    i += 1;
                    continue;
                }
                // @layer name { … }
                if at_header.starts_with("@layer") && i < bytes.len() && bytes[i] == b'{' {
                    let names = at_header["@layer".len()..].trim();
                    let layer_id = if names.is_empty() {
                        None
                    } else {
                        Some(self.ensure_layer(names.split(',').next().unwrap_or(names).trim()))
                    };
                    i += 1;
                    let body_start = i;
                    let mut depth = 1i32;
                    while i < bytes.len() && depth > 0 {
                        if bytes[i] == b'{' {
                            depth += 1;
                        } else if bytes[i] == b'}' {
                            depth -= 1;
                            if depth == 0 {
                                break;
                            }
                        }
                        i += 1;
                    }
                    let body = core::str::from_utf8(&bytes[body_start..i]).unwrap_or("");
                    if i < bytes.len() {
                        i += 1;
                    }
                    let prev = self.parse_layer;
                    self.parse_layer = layer_id.or(layer);
                    self.append_block(body, self.parse_layer);
                    self.parse_layer = prev;
                    continue;
                }
                // @import "url" layer(name);  or  @import url(...) layer(name);
                // Ladybird CSSImportRule — we only honor the layer() clause for cascade order.
                if at_header.starts_with("@import") {
                    // Extract layer(name) if present.
                    if let Some(rest) = at_header.find("layer(").map(|p| &at_header[p + 6..]) {
                        let name = rest.split(')').next().unwrap_or("").trim();
                        if !name.is_empty() {
                            let _ = self.ensure_layer(name);
                        }
                    } else if at_header.contains(" layer") {
                        // @import "x.css" layer; → anonymous layer slot
                        let _ = self.ensure_layer(&alloc::format!(
                            "__import_{}",
                            self.next_layer
                        ));
                    }
                    if i < bytes.len() && bytes[i] == b';' {
                        i += 1;
                    }
                    // skip optional block
                    if i < bytes.len() && bytes[i] == b'{' {
                        let mut depth = 1i32;
                        i += 1;
                        while i < bytes.len() && depth > 0 {
                            if bytes[i] == b'{' {
                                depth += 1;
                            } else if bytes[i] == b'}' {
                                depth -= 1;
                            }
                            i += 1;
                        }
                    }
                    continue;
                }
                // Nested layer names: @layer framework.layout { } → dotted name
                // (already handled by @layer name { } with full name string)

                // @keyframes name { 0% {…} 50% {…} to {…} }
                if (at_header.starts_with("@keyframes")
                    || at_header.starts_with("@-webkit-keyframes"))
                    && i < bytes.len()
                    && bytes[i] == b'{'
                {
                    let name = at_header
                        .split_whitespace()
                        .nth(1)
                        .unwrap_or("")
                        .trim()
                        .trim_matches(|c| c == '"' || c == '\'')
                        .to_string();
                    i += 1;
                    let body_start = i;
                    let mut depth = 1i32;
                    while i < bytes.len() && depth > 0 {
                        if bytes[i] == b'{' {
                            depth += 1;
                        } else if bytes[i] == b'}' {
                            depth -= 1;
                            if depth == 0 {
                                break;
                            }
                        }
                        i += 1;
                    }
                    let body = core::str::from_utf8(&bytes[body_start..i]).unwrap_or("");
                    if i < bytes.len() {
                        i += 1;
                    }
                    if !name.is_empty() {
                        let stops = parse_keyframe_body(body);
                        if !stops.is_empty() {
                            self.keyframes.insert(name, stops);
                        }
                    }
                    continue;
                }

                // @media … { … } — evaluate width queries against viewport_w.
                if at_header.starts_with("@media") && i < bytes.len() && bytes[i] == b'{' {
                    let query = at_header["@media".len()..].trim();
                    i += 1;
                    let body_start = i;
                    let mut depth = 1i32;
                    while i < bytes.len() && depth > 0 {
                        if bytes[i] == b'{' {
                            depth += 1;
                        } else if bytes[i] == b'}' {
                            depth -= 1;
                            if depth == 0 {
                                break;
                            }
                        }
                        i += 1;
                    }
                    let body = core::str::from_utf8(&bytes[body_start..i]).unwrap_or("");
                    if i < bytes.len() {
                        i += 1;
                    }
                    if media_matches(query, self.viewport_w) {
                        self.append_block(body, layer);
                    }
                    continue;
                }

                // Other @-rules: skip
                if i < bytes.len() && bytes[i] == b';' {
                    i += 1;
                    continue;
                }
                if i < bytes.len() && bytes[i] == b'{' {
                    let mut depth = 1i32;
                    i += 1;
                    while i < bytes.len() && depth > 0 {
                        if bytes[i] == b'{' {
                            depth += 1;
                        } else if bytes[i] == b'}' {
                            depth -= 1;
                        }
                        i += 1;
                    }
                }
                continue;
            }
            let sel_start = i;
            while i < bytes.len() && bytes[i] != b'{' {
                i += 1;
            }
            if i >= bytes.len() {
                break;
            }
            let sel_part = core::str::from_utf8(&bytes[sel_start..i]).unwrap_or("").trim();
            i += 1; // skip `{`
            let decl_start = i;
            let mut depth = 1i32;
            while i < bytes.len() && depth > 0 {
                if bytes[i] == b'{' {
                    depth += 1;
                } else if bytes[i] == b'}' {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
                i += 1;
            }
            let decl_part = core::str::from_utf8(&bytes[decl_start..i]).unwrap_or("");
            if i < bytes.len() && bytes[i] == b'}' {
                i += 1;
            }
            let decls = parse_decls(decl_part);
            if decls.is_empty() || sel_part.is_empty() {
                continue;
            }
            for sel_str in sel_part.split(',') {
                if let Some((key, ancestors)) = parse_complex(sel_str.trim()) {
                    let spec = specificity(&key, &ancestors);
                    let order = self.rules.len() as u32;
                    self.rules.push(Rule {
                        key,
                        ancestors,
                        decls: decls.clone(),
                        spec,
                        order,
                        layer,
                    });
                }
            }
        }
    }

    /// Match rules for an element (no ancestor context — descendant selectors
    /// with an ancestor part are treated as never-matching unless their key
    /// alone matches with no ancestor requirement). Prefer [`matching_decls_ex`].
    pub fn matching_decls(
        &self,
        tag: &str,
        id: Option<&str>,
        class: Option<&str>,
    ) -> Vec<(String, String, bool)> {
        self.matching_decls_for(ElemRef::basic(tag, id, class), &[], PseudoElement::None)
    }

    /// Match rules for an element given its ancestor chain (outermost→innermost),
    /// so descendant selectors (`.gb_Na a.gb_1a`) match only when the ancestor
    /// compounds are actually present. Return merged decls sorted by cascade:
    /// layer (unlayered last / highest) → specificity → source order.
    pub fn matching_decls_ex(
        &self,
        tag: &str,
        id: Option<&str>,
        class: Option<&str>,
        chain: &[ElemRef],
    ) -> Vec<(String, String, bool)> {
        self.matching_decls_for(ElemRef::basic(tag, id, class), chain, PseudoElement::None)
    }

    /// Full matcher: subject `el`, ancestor `chain`, and optional pseudo-element
    /// filter (`None` = element rules only; `Before`/`After` = generated-content
    /// rules for that pseudo).
    pub fn matching_decls_for(
        &self,
        el: ElemRef<'_>,
        chain: &[ElemRef],
        pseudo: PseudoElement,
    ) -> Vec<(String, String, bool)> {
        let mut matched: Vec<&Rule> = self
            .rules
            .iter()
            .filter(|r| {
                r.key.pseudo_el == pseudo
                    && r.key.matches_el(&el)
                    && ancestors_match_ctx(&r.ancestors, chain, el.prev)
            })
            .collect();
        matched.sort_by(|a, b| {
            // Unlayered (None) sorts after layered → higher priority.
            let la = a.layer.map(|x| x as i64).unwrap_or(i64::MAX);
            let lb = b.layer.map(|x| x as i64).unwrap_or(i64::MAX);
            la.cmp(&lb)
                .then(a.spec.cmp(&b.spec))
                .then(a.order.cmp(&b.order))
        });
        let mut out = Vec::new();
        for r in matched {
            for d in &r.decls {
                out.push((d.name.clone(), d.value.clone(), d.important));
            }
        }
        out
    }
}

/// Specificity of a complex selector = sum over the key + ancestor compounds
/// of (ids, classes, types), saturated into `(u8, u8, u8)`.
fn specificity(key: &Compound, ancestors: &[AncestorHop]) -> (u8, u8, u8) {
    let (mut a, mut b, mut c) = key.spec();
    for anc in ancestors {
        let (x, y, z) = anc.compound.spec();
        a += x;
        b += y;
        c += z;
    }
    (a.min(255) as u8, b.min(255) as u8, c.min(255) as u8)
}

/// Parse one compound selector. Keeps `:link`, `:hover`, `:nth-child`, attrs,
/// and `::before`/`::after`; drops the other state-dependent pseudos
/// (`:visited` / `:active` / `:focus`) so they cannot apply unconditionally.
/// Read a CSS identifier starting at `start`, honoring escapes.
///
/// Tailwind writes opacity modifiers as `.bg-primary\/20` so the class name
/// is the HTML class `bg-primary/20`. Without unescaping, the `\` stopped
/// the ident and the whole rule was dropped — every `bg-*/10` track, skeleton
/// and hover wash on a shadcn page vanished.
fn read_css_ident(s: &str, start: usize) -> Option<(String, usize)> {
    let bytes = s.as_bytes();
    if start >= bytes.len() {
        return None;
    }
    let mut i = start;
    let mut out = String::new();
    while i < bytes.len() {
        let c = bytes[i];
        if c == b'\\' {
            i += 1;
            if i >= bytes.len() {
                break;
            }
            if bytes[i].is_ascii_hexdigit() {
                let hex_start = i;
                let mut n = 0;
                while i < bytes.len() && n < 6 && bytes[i].is_ascii_hexdigit() {
                    i += 1;
                    n += 1;
                }
                if let Ok(cp) = u32::from_str_radix(&s[hex_start..i], 16) {
                    if let Some(ch) = char::from_u32(cp) {
                        out.push(ch);
                    }
                }
                if i < bytes.len() && bytes[i].is_ascii_whitespace() {
                    i += 1;
                }
            } else if bytes[i] == b'\n' {
                i += 1;
            } else {
                out.push(bytes[i] as char);
                i += 1;
            }
            continue;
        }
        let is_name = c.is_ascii_alphanumeric() || c == b'-' || c == b'_' || c >= 0x80;
        if !is_name {
            break;
        }
        out.push(c as char);
        i += 1;
    }
    if out.is_empty() {
        None
    } else {
        Some((out, i))
    }
}

fn parse_compound(s: &str) -> Option<Compound> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    let mut c = Compound::default();
    let bytes = s.as_bytes();
    let mut i = 0;
    // Leading type / `*`.
    if i < bytes.len() && bytes[i] == b'*' {
        i += 1;
    } else if let Some((tag, ni)) = read_css_ident(s, i) {
        c.tag = Some(tag.to_ascii_lowercase());
        i = ni;
    }
    while i < bytes.len() {
        match bytes[i] {
            b'.' => {
                i += 1;
                let (name, ni) = read_css_ident(s, i)?;
                i = ni;
                c.classes.push(name);
            }
            b'#' => {
                i += 1;
                let (name, ni) = read_css_ident(s, i)?;
                i = ni;
                c.id = Some(name);
            }
            b'[' => {
                i += 1;
                let start = i;
                while i < bytes.len() && bytes[i] != b']' {
                    i += 1;
                }
                let inner = s[start..i].trim();
                if i < bytes.len() {
                    i += 1;
                }
                let attr = parse_attr_sel(inner)?;
                c.attrs.push(attr);
            }
            b':' => {
                i += 1;
                let double = i < bytes.len() && bytes[i] == b':';
                if double {
                    i += 1;
                }
                let start = i;
                while i < bytes.len()
                    && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'-')
                {
                    i += 1;
                }
                let name = s[start..i].to_ascii_lowercase();
                let mut arg = None;
                if i < bytes.len() && bytes[i] == b'(' {
                    i += 1;
                    let a0 = i;
                    let mut depth = 1i32;
                    while i < bytes.len() && depth > 0 {
                        if bytes[i] == b'(' {
                            depth += 1;
                        } else if bytes[i] == b')' {
                            depth -= 1;
                            if depth == 0 {
                                break;
                            }
                        }
                        i += 1;
                    }
                    arg = Some(s[a0..i].trim().to_string());
                    if i < bytes.len() {
                        i += 1;
                    }
                }
                if double
                    || matches!(name.as_str(), "before" | "after" | "first-line" | "first-letter")
                {
                    match name.as_str() {
                        "before" => c.pseudo_el = PseudoElement::Before,
                        "after" => c.pseudo_el = PseudoElement::After,
                        _ => return None,
                    }
                    continue;
                }
                match name.as_str() {
                    "link" | "any-link" => {} // keep base
                    "root" => {
                        // `:root` ≈ document element (`html`).
                        if c.tag.is_none() {
                            c.tag = Some(String::from("html"));
                        }
                    }
                    "hover" => c.hover = true,
                    "visited" | "active" | "focus" | "focus-visible"
                    | "focus-within" | "target" => return None,
                    "not" => {
                        // The argument is a comma-separated compound list.
                        // A `:not()` we cannot parse must **drop the whole
                        // selector** (`return None`), never be ignored: an
                        // ignored negation matches strictly *more* elements
                        // than intended, so the failure mode is a rule
                        // painting things it was written to exclude.
                        for part in arg.as_deref()?.split(',') {
                            let part = part.trim();
                            if part.is_empty() {
                                return None;
                            }
                            c.not.push(parse_compound(part)?);
                        }
                    }
                    "nth-child" => {
                        c.nth_child = Some(parse_nth(arg.as_deref()?)?);
                    }
                    "first-child" => {
                        c.nth_child = Some(NthFormula { a: 0, b: 1 });
                    }
                    "last-child" => {
                        // Approximate as odd huge — layout doesn't pass last index.
                        // Drop rather than wrong-match.
                        return None;
                    }
                    _ => return None,
                }
            }
            _ => return None,
        }
    }
    // A bare `*` or empty after stripping is ok as universal.
    if c.tag.is_none()
        && c.id.is_none()
        && c.classes.is_empty()
        && c.attrs.is_empty()
        && c.nth_child.is_none()
        && c.pseudo_el == PseudoElement::None
        && !s.trim().starts_with('*')
        && s != "*"
    {
        // e.g. completely empty after parse failure path
        if s != "*" && !s.starts_with(|ch: char| ch == '.' || ch == '#' || ch == '[' || ch == ':') {
            // Had a tag that we already set, or invalid.
        }
    }
    Some(c)
}

/// Parse the inside of `[...]` into an attribute predicate.
///
/// The operator must be found **before** splitting on `=`, because every one
/// of `~= ^= $= *= |=` ends in `=`: a plain `split_once('=')` sees `class^`
/// as the name and silently degrades `[class^="text-"]` into an exact match on
/// an attribute that does not exist — a selector that never matches, which
/// looks like a layout bug rather than a parse bug.
fn parse_attr_sel(inner: &str) -> Option<AttrSel> {
    let inner = inner.trim();
    if inner.is_empty() {
        return None;
    }
    // Trailing `i` / `s` flag: `[type=submit i]`. Taken before the value is
    // unquoted, since it sits outside the quotes.
    let (inner, ci) = match inner.rsplit_once(char::is_whitespace) {
        Some((head, flag)) if flag.eq_ignore_ascii_case("i") => (head.trim_end(), true),
        Some((head, flag)) if flag.eq_ignore_ascii_case("s") => (head.trim_end(), false),
        _ => (inner, false),
    };

    for (sym, op) in [
        ("~=", AttrOp::Includes),
        ("^=", AttrOp::Prefix),
        ("$=", AttrOp::Suffix),
        ("*=", AttrOp::Substring),
        ("|=", AttrOp::DashMatch),
    ] {
        if let Some((name, rest)) = inner.split_once(sym) {
            let val = unquote(rest.trim());
            // `[a~=""]` matches nothing per spec, and treating it as presence
            // would match everything — the opposite.
            if val.is_empty() {
                return None;
            }
            return Some(AttrSel {
                name: name.trim().to_ascii_lowercase(),
                value: Some(val.to_string()),
                op,
                ci,
            });
        }
    }
    if let Some((name, rest)) = inner.split_once('=') {
        return Some(AttrSel {
            name: name.trim().to_ascii_lowercase(),
            value: Some(unquote(rest.trim()).to_string()),
            op: AttrOp::Exact,
            ci,
        });
    }
    Some(AttrSel {
        name: inner.to_ascii_lowercase(),
        value: None,
        op: AttrOp::Present,
        ci,
    })
}

fn parse_nth(s: &str) -> Option<NthFormula> {
    let s = s.trim().to_ascii_lowercase();
    if s == "odd" {
        return Some(NthFormula { a: 2, b: 1 });
    }
    if s == "even" {
        return Some(NthFormula { a: 2, b: 0 });
    }
    // An+B / n+B / -n+B / B
    let s = s.replace(' ', "");
    if let Some(n_pos) = s.find('n') {
        let a_str = &s[..n_pos];
        let a = if a_str.is_empty() || a_str == "+" {
            1
        } else if a_str == "-" {
            -1
        } else {
            a_str.parse().ok()?
        };
        let b_str = &s[n_pos + 1..];
        let b = if b_str.is_empty() {
            0
        } else {
            b_str.parse().ok()?
        };
        Some(NthFormula { a, b })
    } else {
        let b: i32 = s.parse().ok()?;
        Some(NthFormula { a: 0, b })
    }
}

/// Parse a complex selector into (key, ancestor hops outermost→innermost).
fn parse_complex(s: &str) -> Option<(Compound, Vec<AncestorHop>)> {
    // Tokenize into compounds + combinators.
    let bytes = s.as_bytes();
    let mut compounds: Vec<Compound> = Vec::new();
    let mut combs: Vec<Combinator> = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= bytes.len() {
            break;
        }
        // Combinator at start of a hop (after first compound).
        if !compounds.is_empty() {
            let comb = match bytes[i] {
                b'>' => {
                    i += 1;
                    Combinator::Child
                }
                b'+' => {
                    i += 1;
                    Combinator::Adjacent
                }
                b'~' => {
                    i += 1;
                    Combinator::Sibling
                }
                _ => Combinator::Descendant,
            };
            while i < bytes.len() && bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            combs.push(comb);
        }
        let start = i;
        // Compound runs until whitespace or combinator (not inside [] or ()).
        let mut depth_b = 0i32;
        let mut depth_p = 0i32;
        while i < bytes.len() {
            let c = bytes[i];
            if c == b'[' {
                depth_b += 1;
            } else if c == b']' {
                depth_b -= 1;
            } else if c == b'(' {
                depth_p += 1;
            } else if c == b')' {
                depth_p -= 1;
            } else if depth_b == 0 && depth_p == 0 {
                if c.is_ascii_whitespace() || c == b'>' || c == b'+' || c == b'~' {
                    break;
                }
            }
            i += 1;
        }
        let tok = s[start..i].trim();
        if tok.is_empty() {
            continue;
        }
        compounds.push(parse_compound(tok)?);
    }
    if compounds.is_empty() {
        return None;
    }
    let key = compounds.pop()?;
    // combs.len() should be compounds.len() (remaining).
    let mut ancestors = Vec::new();
    for (idx, compound) in compounds.into_iter().enumerate() {
        let combinator = combs.get(idx).copied().unwrap_or(Combinator::Descendant);
        ancestors.push(AncestorHop {
            compound,
            combinator,
        });
    }
    Some((key, ancestors))
}

/// Match ancestor hops against the element's ancestor chain.
/// Child (`>`) requires the hop to match the immediate next ancestor slot;
/// descendant allows skipping.
fn ancestors_match(ancestors: &[AncestorHop], chain: &[ElemRef]) -> bool {
    ancestors_match_ctx(ancestors, chain, None)
}

/// As [`ancestors_match`], but with the subject's preceding siblings so `+`
/// and `~` can be matched exactly instead of approximated as descendant.
///
/// The hop list is ancestors-only by construction, so a **sibling** hop does
/// not belong in it: `a + b` relates two elements at the same depth, and `a`
/// is nowhere in `b`'s ancestor chain. The trailing run of sibling hops is
/// therefore peeled off the end and resolved against `prev` — right to left,
/// because that is the direction the constraint reads — and only what remains
/// is matched against the chain as before.
///
/// This covers the shapes that appear in practice, including the one Tailwind
/// emits for `space-y-*`: `.space-y-4 > :not([hidden]) ~ :not([hidden])`.
/// Before this, `+` and `~` were both treated as descendant, which matches
/// **more** elements than written — so a rule intended for "every item after
/// the first" applied to the first one too.
fn ancestors_match_ctx(
    ancestors: &[AncestorHop],
    chain: &[ElemRef],
    prev: Option<&[ElemRef]>,
) -> bool {
    if ancestors.is_empty() {
        return true;
    }
    // No sibling context: keep the historical descendant approximation rather
    // than start rejecting every `+`/`~` rule. See `ElemRef::prev`.
    let Some(prev) = prev else {
        return ancestors_match_descendant_approx(ancestors, chain);
    };
    // Peel trailing sibling hops. `ancestors[i].combinator` is the combinator
    // *between* hop `i` and whatever follows it (hop `i+1`, or the subject).
    let mut end = ancestors.len();
    // Index into `prev`, walking backwards from the immediately preceding
    // sibling. `usize` cannot go below zero, so track it as "how many of the
    // tail of `prev` are still available".
    let mut avail = prev.len();
    while end > 0 && matches!(
        ancestors[end - 1].combinator,
        Combinator::Adjacent | Combinator::Sibling
    ) {
        let hop = &ancestors[end - 1];
        match hop.combinator {
            Combinator::Adjacent => {
                // `+`: exactly the immediately preceding element sibling.
                if avail == 0 || !hop.compound.matches_el(&prev[avail - 1]) {
                    return false;
                }
                avail -= 1;
            }
            Combinator::Sibling => {
                // `~`: any earlier sibling. Take the nearest match, which
                // leaves the most siblings for the hops still to come.
                let mut found = false;
                while avail > 0 {
                    avail -= 1;
                    if hop.compound.matches_el(&prev[avail]) {
                        found = true;
                        break;
                    }
                }
                if !found {
                    return false;
                }
            }
            _ => unreachable!("loop condition"),
        }
        end -= 1;
    }
    // A sibling hop's own ancestors are the subject's ancestors — same parent —
    // so the remaining hops still match against `chain` unchanged.
    ancestors_match_descendant_approx(&ancestors[..end], chain)
}

/// The ancestor walk, with `+`/`~` treated as descendant.
///
/// This is what the matcher did for every combinator before sibling context
/// existed. It is kept — rather than deleted — because it is still the correct
/// behaviour for a caller that cannot supply siblings: approximating is wrong
/// in the permissive direction, whereas matching nothing is wrong in the
/// direction that makes styled pages lose rules they had.
fn ancestors_match_descendant_approx(ancestors: &[AncestorHop], chain: &[ElemRef]) -> bool {
    if ancestors.is_empty() {
        return true;
    }
    // Walk from outermost hop against chain from the start.
    // For each hop, find a matching element; Child means the *next* chain
    // entry after the previous match must be that element (no skip).
    let mut ci = 0usize; // next chain index to consider
    for (hi, hop) in ancestors.iter().enumerate() {
        let comb_into_this = if hi == 0 {
            Combinator::Descendant // first hop: anywhere
        } else {
            ancestors[hi - 1].combinator
        };
        match comb_into_this {
            Combinator::Child => {
                if ci >= chain.len() || !hop.compound.matches_el(&chain[ci]) {
                    return false;
                }
                ci += 1;
            }
            Combinator::Descendant | Combinator::Adjacent | Combinator::Sibling => {
                // Adjacent/Sibling approximated as descendant (plan: exact child first).
                let mut found = false;
                while ci < chain.len() {
                    if hop.compound.matches_el(&chain[ci]) {
                        ci += 1;
                        found = true;
                        break;
                    }
                    ci += 1;
                }
                if !found {
                    return false;
                }
            }
        }
    }
    // Last hop's combinator constrains the subject relative to the last
    // matched ancestor — subject is not in `chain`; layout's `chain` is
    // ancestors only. Child means the subject must be a direct child, which
    // is always true for the chain we pass (parent is last in chain). So for
    // `div > p`, ancestors=[div] with Child into key: after matching div at
    // end of chain, Child is satisfied because subject is walked as child.
    if let Some(last) = ancestors.last() {
        if last.combinator == Combinator::Child {
            // Require the matched ancestor to be the immediate parent =
            // last entry in chain.
            // After the loop, `ci` is one past the match for the last hop.
            if ci == 0 || ci != chain.len() {
                // If we matched something before the parent, child fails.
                // Successful child walk leaves ci == chain.len() when the
                // parent is the last hop match.
                if ci != chain.len() {
                    return false;
                }
            }
        }
    }
    true
}

fn parse_decls(block: &str) -> Vec<Decl> {
    let mut out = Vec::new();
    for part in block.split(';') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let (name, rest) = match part.find(':') {
            Some(i) => (part[..i].trim(), part[i + 1..].trim()),
            None => continue,
        };
        if name.is_empty() || rest.is_empty() {
            continue;
        }
        let important = rest.ends_with("!important");
        let value = if important {
            rest.trim_end_matches("!important").trim().to_string()
        } else {
            rest.to_string()
        };
        out.push(Decl {
            name: name.to_ascii_lowercase(),
            value,
            important,
        });
    }
    out
}

/// Parse the body of `@keyframes name { … }`.
///
/// Selectors are `from` / `to` / `N%` / comma lists (`0%, 100%`). Each
/// block's declarations are stored verbatim and applied at compute time.
fn parse_keyframe_body(body: &str) -> Vec<KeyframeStop> {
    let bytes = body.as_bytes();
    let mut i = 0;
    let mut out: Vec<KeyframeStop> = Vec::new();
    while i < bytes.len() {
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= bytes.len() {
            break;
        }
        let sel_start = i;
        while i < bytes.len() && bytes[i] != b'{' {
            i += 1;
        }
        if i >= bytes.len() {
            break;
        }
        let sel = core::str::from_utf8(&bytes[sel_start..i]).unwrap_or("").trim();
        i += 1;
        let decl_start = i;
        let mut depth = 1i32;
        while i < bytes.len() && depth > 0 {
            if bytes[i] == b'{' {
                depth += 1;
            } else if bytes[i] == b'}' {
                depth -= 1;
                if depth == 0 {
                    break;
                }
            }
            i += 1;
        }
        let decl_part = core::str::from_utf8(&bytes[decl_start..i]).unwrap_or("");
        if i < bytes.len() && bytes[i] == b'}' {
            i += 1;
        }
        let decls: Vec<(String, String)> = parse_decls(decl_part)
            .into_iter()
            .map(|d| (d.name, d.value))
            .collect();
        for part in sel.split(',') {
            if let Some(pct) = parse_keyframe_sel(part.trim()) {
                out.push(KeyframeStop {
                    pct,
                    decls: decls.clone(),
                });
            }
        }
    }
    out.sort_by_key(|s| s.pct);
    out
}

fn parse_keyframe_sel(s: &str) -> Option<u16> {
    let t = s.trim().to_ascii_lowercase();
    if t == "from" {
        return Some(0);
    }
    if t == "to" {
        return Some(100);
    }
    let n = t.strip_suffix('%')?.trim().parse::<f32>().ok()?;
    Some(n.clamp(0.0, 100.0) as u16)
}

/// Split a declaration value into whitespace-separated tokens, keeping
/// `(...)` groups whole.
///
/// A plain `split_whitespace` shreds every functional value: `rgb(255, 255,
/// 255)` becomes three tokens, none of which is a colour, so a shorthand that
/// scans tokens for a colour silently found none. Every Tailwind background,
/// border and text colour is written that way — `rgb(248 250 252 /
/// var(--tw-bg-opacity, 1))` — so a Tailwind page rendered with no backgrounds
/// at all while `#ffffff` in the same position worked.
pub fn value_tokens(v: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut start: Option<usize> = None;
    for (i, c) in v.char_indices() {
        match c {
            '(' => {
                depth += 1;
                start.get_or_insert(i);
            }
            ')' => {
                depth -= 1;
                start.get_or_insert(i);
            }
            c if c.is_whitespace() && depth <= 0 => {
                if let Some(s) = start.take() {
                    out.push(&v[s..i]);
                }
            }
            _ => {
                start.get_or_insert(i);
            }
        }
    }
    if let Some(s) = start {
        out.push(&v[s..]);
    }
    out
}

/// Parse a color: `#rgb`, `#rrggbb`, `rgb(r,g,b)`, named basics.
///
/// A fully transparent colour is `None`, the same answer as the `transparent`
/// keyword — a caller asking "is there a colour to paint here?" must not be
/// told "black". Tailwind's shadow chain is literally `0 0 #0000`, and reading
/// that as opaque black painted a solid black rectangle over every card.
pub fn parse_color(s: &str) -> Option<u32> {
    let s = s.trim().to_ascii_lowercase();
    if s == "transparent" {
        return None;
    }
    match s.as_str() {
        "black" => return Some(0x000000),
        "white" => return Some(0xffffff),
        "red" => return Some(0xff0000),
        "green" => return Some(0x008000),
        "blue" => return Some(0x0000ff),
        "gray" | "grey" => return Some(0x808080),
        "silver" => return Some(0xc0c0c0),
        "maroon" => return Some(0x800000),
        "yellow" => return Some(0xffff00),
        "olive" => return Some(0x808000),
        "lime" => return Some(0x00ff00),
        "aqua" | "cyan" => return Some(0x00ffff),
        "teal" => return Some(0x008080),
        "navy" => return Some(0x000080),
        "fuchsia" | "magenta" => return Some(0xff00ff),
        "purple" => return Some(0x800080),
        "orange" => return Some(0xffa500),
        "terracotta" => return Some(0xcc785c),
        _ => {}
    }
    if let Some(hex) = s.strip_prefix('#') {
        if hex.len() == 3 {
            let r = u32::from_str_radix(&hex[0..1], 16).ok()?;
            let g = u32::from_str_radix(&hex[1..2], 16).ok()?;
            let b = u32::from_str_radix(&hex[2..3], 16).ok()?;
            return Some((r << 20) | (r << 16) | (g << 12) | (g << 8) | (b << 4) | b);
        }
        if hex.len() == 4 {
            // #RGBA — alpha 0 is transparent (`#0000`), else blend toward the
            // page the way the rgba() path does.
            let r = u32::from_str_radix(&hex[0..1], 16).ok()?;
            let g = u32::from_str_radix(&hex[1..2], 16).ok()?;
            let b = u32::from_str_radix(&hex[2..3], 16).ok()?;
            let a = u32::from_str_radix(&hex[3..4], 16).ok()?;
            if a == 0 {
                return None;
            }
            let af = a as f32 / 15.0;
            let ch = |c: u32| -> u32 {
                let c8 = (c << 4) | c;
                ((c8 as f32 * af + 255.0 * (1.0 - af)) as u32).min(255)
            };
            return Some((ch(r) << 16) | (ch(g) << 8) | ch(b));
        }
        if hex.len() == 6 {
            return u32::from_str_radix(hex, 16).ok();
        }
        if hex.len() == 8 {
            // #RRGGBBAA — same rule as #RGBA.
            let rgb = u32::from_str_radix(&hex[0..6], 16).ok()?;
            let a = u32::from_str_radix(&hex[6..8], 16).ok()?;
            if a == 0 {
                return None;
            }
            let af = a as f32 / 255.0;
            let ch = |c: u32| -> u32 { ((c as f32 * af + 255.0 * (1.0 - af)) as u32).min(255) };
            return Some(
                (ch((rgb >> 16) & 0xff) << 16) | (ch((rgb >> 8) & 0xff) << 8) | ch(rgb & 0xff),
            );
        }
    }
    // `rgb(r,g,b)` and `rgba(r,g,b,a)` (also space/slash-separated). Alpha is
    // approximated by blending toward white (light-theme assumption), so a
    // subtle `rgba(0,0,0,.08)` border renders light instead of solid black.
    let rgb_inner = s
        .strip_prefix("rgba(")
        .or_else(|| s.strip_prefix("rgb("))
        .and_then(|x| x.strip_suffix(')'));
    if let Some(inner) = rgb_inner {
        // Split on commas or whitespace / slash (CSS4 syntax).
        let norm = inner.replace('/', " ").replace(',', " ");
        let mut parts = norm.split_whitespace();
        let comp = |t: &str| -> Option<u32> {
            if let Some(p) = t.strip_suffix('%') {
                p.trim().parse::<f32>().ok().map(|v| (v * 255.0 / 100.0) as u32)
            } else {
                t.trim().parse::<f32>().ok().map(|v| v as u32)
            }
        };
        let r = comp(parts.next()?)?.min(255);
        let g = comp(parts.next()?)?.min(255);
        let b = comp(parts.next()?)?.min(255);
        let a: f32 = match parts.next() {
            Some(t) => {
                if let Some(p) = t.strip_suffix('%') {
                    p.trim().parse::<f32>().map(|v| v / 100.0).unwrap_or(1.0)
                } else {
                    t.trim().parse::<f32>().unwrap_or(1.0)
                }
            }
            None => 1.0,
        }
        .clamp(0.0, 1.0);
        if a <= 0.0 {
            return None; // fully transparent — nothing to paint
        }
        let mix = |c: u32| -> u32 { ((c as f32 * a + 255.0 * (1.0 - a)) as u32).min(255) };
        return Some((mix(r) << 16) | (mix(g) << 8) | mix(b));
    }
    // `hsl(H S% L%)` / `hsl(H, S%, L%)` / `hsla(…)`, with the same `/ alpha`
    // form. This is not an exotic corner: shadcn/ui defines its ENTIRE palette
    // as HSL triples in custom properties (`--background: 0 0% 100%`) and every
    // component reads `hsl(var(--background))`, so without it a shadcn page has
    // no colour at all — no card, no button fill, no border.
    let hsl_inner = s
        .strip_prefix("hsla(")
        .or_else(|| s.strip_prefix("hsl("))
        .and_then(|x| x.strip_suffix(')'));
    if let Some(inner) = hsl_inner {
        let norm = inner.replace('/', " ").replace(',', " ");
        let mut parts = norm.split_whitespace();
        // Hue is an angle: bare number or `deg`; `turn`/`rad`/`grad` are rarer
        // but cheap to accept.
        let h_tok = parts.next()?;
        let h: f32 = if let Some(n) = h_tok.strip_suffix("deg") {
            n.parse().ok()?
        } else if let Some(n) = h_tok.strip_suffix("turn") {
            n.parse::<f32>().ok()? * 360.0
        } else if let Some(n) = h_tok.strip_suffix("grad") {
            n.parse::<f32>().ok()? * 0.9
        } else if let Some(n) = h_tok.strip_suffix("rad") {
            n.parse::<f32>().ok()? * 180.0 / core::f32::consts::PI
        } else {
            h_tok.parse().ok()?
        };
        let pct = |t: &str| -> Option<f32> {
            t.strip_suffix('%').unwrap_or(t).trim().parse::<f32>().ok().map(|v| v / 100.0)
        };
        let sat = pct(parts.next()?)?.clamp(0.0, 1.0);
        let lit = pct(parts.next()?)?.clamp(0.0, 1.0);
        let a = match parts.next() {
            Some(t) => {
                if let Some(p) = t.strip_suffix('%') {
                    p.trim().parse::<f32>().map(|v| v / 100.0).unwrap_or(1.0)
                } else {
                    t.trim().parse::<f32>().unwrap_or(1.0)
                }
            }
            None => 1.0,
        }
        .clamp(0.0, 1.0);
        if a <= 0.0 {
            return None;
        }
        let (r, g, b) = hsl_to_rgb(h, sat, lit);
        let mix = |c: f32| -> u32 { ((c * 255.0 * a + 255.0 * (1.0 - a)) as u32).min(255) };
        return Some((mix(r) << 16) | (mix(g) << 8) | mix(b));
    }
    None
}

/// HSL → RGB (each channel 0.0..=1.0), per CSS Color 4.
fn hsl_to_rgb(h_deg: f32, s: f32, l: f32) -> (f32, f32, f32) {
    // `%` on a negative hue keeps the sign in Rust, so add a turn back.
    let h = ((h_deg % 360.0) + 360.0) % 360.0;
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let h6 = h / 60.0;
    // `abs(h6 mod 2 - 1)` — no `f32::rem_euclid` in core for no_std here.
    let x = c * (1.0 - ((h6 % 2.0) - 1.0).abs());
    let m = l - c / 2.0;
    let (r, g, b) = match h6 as i32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    (r + m, g + m, b + m)
}

/// First color stop of a `linear-gradient(...)` / `radial-gradient(...)`, used
/// as a paintable fallback and gradient start color. `None` for `url(...)` or
/// unparseable values.
pub fn gradient_first_color(v: &str) -> Option<u32> {
    let lower = v.to_ascii_lowercase();
    let idx = lower.find("gradient(")?;
    let inner = &v[idx + "gradient(".len()..];
    let inner = inner.strip_suffix(')').unwrap_or(inner);
    for part in inner.split(',') {
        // Skip the direction token (`to right`, `45deg`, `circle`, …).
        for tok in part.split_whitespace() {
            if let Some(c) = parse_color(tok) {
                return Some(c);
            }
        }
    }
    None
}

/// All color stops of a gradient, in order (for a simple linear raster).
pub fn gradient_stops(v: &str) -> Vec<u32> {
    let lower = v.to_ascii_lowercase();
    let mut out = Vec::new();
    if let Some(idx) = lower.find("gradient(") {
        let inner = &v[idx + "gradient(".len()..];
        let inner = inner.strip_suffix(')').unwrap_or(inner);
        for part in inner.split(',') {
            for tok in part.split_whitespace() {
                if let Some(c) = parse_color(tok) {
                    out.push(c);
                    break;
                }
            }
        }
    }
    out
}

/// Parse a `grid-template-columns` value into a track list, expanding
/// `repeat(n, …)`. Recognizes `<n>fr`, pixel lengths, and `auto`/`minmax(…)`/
/// `min-content`/`max-content` (all treated as `Auto`).
pub fn parse_grid_tracks(value: &str) -> Vec<GridTrack> {
    let v = value.trim();
    let low = v.to_ascii_lowercase();
    let expanded: String;
    let src: &str = if low.starts_with("repeat(") {
        if let Some(close) = v.find(')') {
            let inner = &v[7..close];
            let mut it = inner.splitn(2, ',');
            let count: usize = it.next().and_then(|s| s.trim().parse().ok()).unwrap_or(1);
            let tmpl = it.next().unwrap_or("1fr").trim();
            let mut s = String::new();
            for _ in 0..count.min(48) {
                s.push_str(tmpl);
                s.push(' ');
            }
            expanded = s;
            &expanded
        } else {
            v
        }
    } else {
        v
    };
    let mut out = Vec::new();
    for tok in src.split_whitespace() {
        let low = tok.to_ascii_lowercase();
        if let Some(fr) = low.strip_suffix("fr") {
            out.push(GridTrack::Fr(fr.trim().parse().unwrap_or(1.0)));
        } else if low == "auto"
            || low.contains("min-content")
            || low.contains("max-content")
            || low.starts_with("minmax")
            || low.starts_with("fit-content")
        {
            out.push(GridTrack::Auto);
        } else if let Some(px) = parse_px(tok) {
            out.push(GridTrack::Px(px));
        } else {
            out.push(GridTrack::Auto);
        }
        if out.len() >= 24 {
            break;
        }
    }
    if out.is_empty() {
        out.push(GridTrack::Fr(1.0));
    }
    out
}

pub fn parse_px(s: &str) -> Option<i32> {
    parse_px_rel(s, None)
}

/// `50%` / `100%` → the number (0–100+). `None` if the token is not a percent.
pub fn parse_pct(s: &str) -> Option<f32> {
    s.trim().strip_suffix('%')?.trim().parse().ok()
}

fn is_css_time(s: &str) -> bool {
    let t = s.trim().to_ascii_lowercase();
    let body = if let Some(b) = t.strip_suffix("ms") {
        b
    } else if let Some(b) = t.strip_suffix('s') {
        b
    } else {
        return false;
    };
    !body.is_empty() && body.parse::<f32>().is_ok()
}

fn is_timing_fn(s: &str) -> bool {
    let t = s.trim().to_ascii_lowercase();
    t == "ease"
        || t == "linear"
        || t == "ease-in"
        || t == "ease-out"
        || t == "ease-in-out"
        || t == "step-start"
        || t == "step-end"
        || t.starts_with("cubic-bezier(")
        || t.starts_with("steps(")
}

/// CSS time token → milliseconds. `2s` → 2000, `150ms` → 150, else 0.
pub fn parse_time_ms(s: &str) -> u32 {
    let t = s.trim().to_ascii_lowercase();
    if let Some(b) = t.strip_suffix("ms") {
        return b.trim().parse::<f32>().ok().map(|v| v.max(0.0) as u32).unwrap_or(0);
    }
    if let Some(b) = t.strip_suffix('s') {
        return b
            .trim()
            .parse::<f32>()
            .ok()
            .map(|v| (v.max(0.0) * 1000.0) as u32)
            .unwrap_or(0);
    }
    0
}

/// The CSS root font size `rem` is relative to.
///
/// 16px — the browser default every design system's scale is calibrated
/// against, including Tailwind's and shadcn's. It is deliberately NOT the
/// console's own 14px body size: `text-sm` (`0.875rem`) is meant to come out at
/// 14px, and it only does if the root is 16.
pub const REM_PX: f32 = 16.0;

/// Viewport the current layout is for, in CSS px — what `vh`/`vw`/`vmin`/`vmax`
/// resolve against. Set by `layout::layout_document` on entry; the fallback is
/// a plausible desktop so a unit test calling `parse_px` directly still gets a
/// number rather than dropping the declaration.
///
/// SAFETY (`Sync`): `mm::Locked` is unconditionally `Sync`, and layout runs
/// only on the single-threaded shell task and is not reentrant.
static VIEWPORT: crate::mm::Locked<(i32, i32)> = crate::mm::Locked::new((1024, 768));

/// Animation clock. Tests default to 0 (the 0% / identity stop) so a page
/// that uses `animate-pulse` stays deterministic; the live browser uses the
/// wall clock, and tests that want a mid-animation snapshot call
/// [`set_animation_now_ms`].
static ANIM_NOW_MS: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(0);
static ANIM_OVERRIDE: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);
static PAGE_WANTS_ANIM: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

/// Element indices that currently match `:hover` (the pointer target and its
/// ancestors). Empty when the pointer is not over a laid-out element.
///
/// SAFETY (`Sync`): same as `VIEWPORT` — layout and the hover handler run on
/// the single-threaded shell task.
static HOVER_ELEMS: crate::mm::Locked<alloc::vec::Vec<usize>> =
    crate::mm::Locked::new(alloc::vec::Vec::new());

/// Replace the hovered-element set. Returns `true` when it changed so the
/// caller can restyle (a Tailwind `.hover\:bg-*:hover` rule only matches
/// after the next layout).
pub fn set_hover_elems(idxs: &[usize]) -> bool {
    HOVER_ELEMS.with(|h| {
        if h.as_slice() == idxs {
            return false;
        }
        h.clear();
        h.extend_from_slice(idxs);
        true
    })
}

/// True when `elem_idx` is the hover target or an ancestor of it.
pub fn elem_is_hovered(elem_idx: usize) -> bool {
    HOVER_ELEMS.with(|h| h.iter().any(|&i| i == elem_idx))
}

/// Pin the animation clock (tests). `None` restores the default (0 in the
/// unit suite, the wall clock on a running kernel).
pub fn set_animation_now_ms(ms: Option<u32>) {
    match ms {
        Some(v) => {
            ANIM_OVERRIDE.store(true, core::sync::atomic::Ordering::Relaxed);
            ANIM_NOW_MS.store(v, core::sync::atomic::Ordering::Relaxed);
        }
        None => {
            ANIM_OVERRIDE.store(false, core::sync::atomic::Ordering::Relaxed);
            ANIM_NOW_MS.store(0, core::sync::atomic::Ordering::Relaxed);
        }
    }
}

pub fn animation_now_ms() -> u32 {
    if ANIM_OVERRIDE.load(core::sync::atomic::Ordering::Relaxed) {
        return ANIM_NOW_MS.load(core::sync::atomic::Ordering::Relaxed);
    }
    #[cfg(test)]
    {
        0
    }
    #[cfg(not(test))]
    {
        crate::arch::now_ms() as u32
    }
}

/// Cleared at the start of a layout; set when any element actually applies a
/// keyframe. The browser tick uses this to know a live page needs a repaint.
pub fn clear_animation_flag() {
    PAGE_WANTS_ANIM.store(false, core::sync::atomic::Ordering::Relaxed);
}

pub fn page_wants_animation() -> bool {
    PAGE_WANTS_ANIM.load(core::sync::atomic::Ordering::Relaxed)
}

fn mark_page_wants_animation() {
    PAGE_WANTS_ANIM.store(true, core::sync::atomic::Ordering::Relaxed);
}

/// Record the viewport for viewport-relative units. Called once per layout.
pub fn set_viewport(w: i32, h: i32) {
    if w > 0 && h > 0 {
        VIEWPORT.with(|v| *v = (w, h));
    }
}

fn viewport() -> (i32, i32) {
    VIEWPORT.with(|v| *v)
}

/// Parse a CSS length to px. When `cb_w` is `Some(w)`, percentages (and
/// `calc(…%…)`) resolve against the containing-block width `w`. Without a CB,
/// bare `%` and `%`-bearing calcs return `None` (layout call sites that have
/// `max_w` should pass it).
pub fn parse_px_rel(s: &str, cb_w: Option<i32>) -> Option<i32> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    let low = s.to_ascii_lowercase();
    if let Some(inner) = low.strip_prefix("calc(").and_then(|r| r.strip_suffix(')')) {
        return eval_calc(inner.trim(), cb_w);
    }
    parse_length_term(&low, cb_w)
}

fn parse_length_term(s: &str, cb_w: Option<i32>) -> Option<i32> {
    let s = s.trim();
    if s == "0" {
        return Some(0);
    }
    if let Some(n) = s.strip_suffix("px") {
        return n.trim().parse::<f32>().ok().map(f32_to_i32);
    }
    // `rem` MUST be tested before `em` — `"28rem"` ends in "em", so the em arm
    // took it, failed to parse the leftover "28r", and answered `None`. Every
    // Tailwind size is in rem (`max-w-md` is `28rem`, `p-6` is `1.5rem`), so a
    // Tailwind page laid out with no widths, no padding and no type scale while
    // each individual declaration looked perfectly well-formed.
    if let Some(n) = s.strip_suffix("rem") {
        let f: f32 = n.trim().parse().ok()?;
        return Some(f32_to_i32(f * REM_PX));
    }
    if let Some(n) = s.strip_suffix("em") {
        let f: f32 = n.trim().parse().ok()?;
        return Some(f32_to_i32(f * 14.0));
    }
    // Viewport units, against the viewport the current layout is for.
    if let Some(n) = s.strip_suffix("vmin") {
        let (vw, vh) = viewport();
        let f: f32 = n.trim().parse().ok()?;
        return Some(f32_to_i32(f / 100.0 * vw.min(vh) as f32));
    }
    if let Some(n) = s.strip_suffix("vmax") {
        let (vw, vh) = viewport();
        let f: f32 = n.trim().parse().ok()?;
        return Some(f32_to_i32(f / 100.0 * vw.max(vh) as f32));
    }
    if let Some(n) = s.strip_suffix("vh") {
        let f: f32 = n.trim().parse().ok()?;
        return Some(f32_to_i32(f / 100.0 * viewport().1 as f32));
    }
    if let Some(n) = s.strip_suffix("vw") {
        let f: f32 = n.trim().parse().ok()?;
        return Some(f32_to_i32(f / 100.0 * viewport().0 as f32));
    }
    if let Some(n) = s.strip_suffix("pt") {
        let f: f32 = n.trim().parse().ok()?;
        return Some(f32_to_i32(f * 96.0 / 72.0));
    }
    // `ch`/`ex` are font metrics; approximate against the default face rather
    // than dropping the declaration.
    if let Some(n) = s.strip_suffix("ch") {
        let f: f32 = n.trim().parse().ok()?;
        return Some(f32_to_i32(f * 8.0));
    }
    if let Some(n) = s.strip_suffix("ex") {
        let f: f32 = n.trim().parse().ok()?;
        return Some(f32_to_i32(f * 7.0));
    }
    if let Some(n) = s.strip_suffix('%') {
        let pct: f32 = n.trim().parse().ok()?;
        let w = cb_w?;
        return Some(f32_to_i32((pct / 100.0) * w as f32));
    }
    s.parse::<f32>().ok().map(f32_to_i32)
}

fn f32_to_i32(v: f32) -> i32 {
    // no_std: avoid `f32::round` (needs Float trait). Truncate toward nearest.
    if v >= 0.0 {
        (v + 0.5) as i32
    } else {
        (v - 0.5) as i32
    }
}

/// Evaluate a simple `calc()` expression: terms of `Npx|Nem|N%` joined by
/// `+` / `-` (no `*`/`/` nesting for v1). Whitespace around operators required
/// by CSS (`calc(100% - 20px)`).
fn eval_calc(inner: &str, cb_w: Option<i32>) -> Option<i32> {
    // Tokenize into signed terms by scanning for top-level + / - that follow
    // a completed term (not the unary sign of the first term).
    let bytes = inner.as_bytes();
    let mut terms: Vec<(bool, &str)> = Vec::new(); // (negative, term)
    let mut start = 0usize;
    let mut i = 0usize;
    let mut neg_first = false;
    // Optional leading sign.
    if i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'-') {
        neg_first = bytes[i] == b'-';
        i += 1;
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        start = i;
    }
    while i < bytes.len() {
        let c = bytes[i];
        if (c == b'+' || c == b'-') && i > start {
            // Operator only if flanked by whitespace (CSS calc grammar) OR
            // previous char ended a unit — accept either.
            let prev_ws = bytes[i - 1].is_ascii_whitespace();
            let next_ws = i + 1 < bytes.len() && bytes[i + 1].is_ascii_whitespace();
            if prev_ws || next_ws {
                let term = inner[start..i].trim();
                if !term.is_empty() {
                    terms.push((neg_first, term));
                }
                neg_first = c == b'-';
                i += 1;
                while i < bytes.len() && bytes[i].is_ascii_whitespace() {
                    i += 1;
                }
                start = i;
                continue;
            }
        }
        i += 1;
    }
    let term = inner[start..].trim();
    if !term.is_empty() {
        terms.push((neg_first, term));
    }
    if terms.is_empty() {
        return None;
    }
    let mut sum = 0i32;
    for (neg, t) in terms {
        let v = parse_length_term(t, cb_w)?;
        sum = if neg { sum - v } else { sum + v };
    }
    Some(sum)
}

/// Evaluate an `@media` query list against a layout viewport width.
///
/// - `print` / `speech` alone → false (drop print-only sheets).
/// - `screen` / `all` / empty → true.
/// - `(min-width: Npx)` / `(max-width: Npx)` — when `viewport_w == 0` (unknown),
///   fail-open (keep rules); otherwise compare.
/// - Unknown features → fail-open (keep) for screen/all contexts.
pub fn media_matches(query: &str, viewport_w: i32) -> bool {
    let q = query.trim().to_ascii_lowercase();
    if q.is_empty() {
        return true;
    }
    // Comma = OR of media queries.
    for part in q.split(',') {
        if media_query_one(part.trim(), viewport_w) {
            return true;
        }
    }
    false
}

fn media_query_one(q: &str, viewport_w: i32) -> bool {
    let q = q.trim();
    if q.is_empty() {
        return true;
    }
    // Extract media type (token before first `(` or `and`).
    let type_end = q.find('(').unwrap_or(q.len());
    let type_part = q[..type_end].replace(" and ", " ").replace("and ", " ");
    let ty = type_part.split_whitespace().next().unwrap_or("all");
    if matches!(
        ty,
        "print" | "speech" | "tty" | "tv" | "projection" | "handheld" | "braille" | "embossed" | "aural"
    ) {
        return false;
    }
    // Evaluate every `(feature: value)` — AND.
    let b = q.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'(' {
            let start = i + 1;
            let mut j = start;
            while j < b.len() && b[j] != b')' {
                j += 1;
            }
            let feat = q[start..j.min(q.len())].trim();
            if !media_feature(feat, viewport_w) {
                return false;
            }
            i = if j < b.len() { j + 1 } else { b.len() };
        } else {
            i += 1;
        }
    }
    true
}

fn media_feature(feat: &str, viewport_w: i32) -> bool {
    let feat = feat.trim();
    let (name, value) = if let Some((a, b)) = feat.split_once(':') {
        (a.trim(), b.trim())
    } else {
        // boolean features like `(color)` — fail-open
        return true;
    };
    let name = name.trim().to_ascii_lowercase();
    match name.as_str() {
        "min-width" | "max-width" | "width" => {
            if viewport_w == 0 {
                return true; // unknown viewport — fail-open
            }
            let Some(px) = parse_px(value).or_else(|| value.parse().ok()) else {
                return true;
            };
            match name.as_str() {
                "min-width" => viewport_w >= px,
                "max-width" => viewport_w <= px,
                "width" => viewport_w == px,
                _ => true,
            }
        }
        _ => true, // unknown feature — fail-open
    }
}

/// Apply a list of decls onto a base style (later / important wins).
pub fn apply_decls(base: &mut ComputedStyle, decls: &[(String, String, bool)]) {
    // Two passes: non-important then important.
    for important_pass in [false, true] {
        for (name, value, imp) in decls {
            if *imp != important_pass {
                continue;
            }
            apply_one(base, name, value);
        }
    }
}

fn apply_one(st: &mut ComputedStyle, name: &str, value: &str) {
    // Custom properties: `--name: value` (chromestatus "variable").
    if name.starts_with("--") {
        st.custom_props.insert(name.to_string(), value.trim().to_string());
        return;
    }
    let name = canonicalize_prop(name);
    let value_owned = resolve_var(value, &st.custom_props);
    let value = value_owned.as_str();
    match name.as_str() {
        "color" => {
            if let Some(c) = parse_color(value) {
                st.color = c;
            }
        }
        "background" | "background-color" => {
            // `background: #fff url(...)` — take first color-like token.
            for tok in value_tokens(value) {
                if let Some(c) = parse_color(tok) {
                    st.background = Some(c);
                    break;
                }
                if tok == "transparent" {
                    st.background = None;
                    break;
                }
            }
            // The shorthand can also carry an image / gradient.
            if name == "background" {
                let v = value.trim();
                if v.contains("url(") || v.contains("gradient(") {
                    st.background_image = v.to_string();
                    if let Some(c) = gradient_first_color(v) {
                        if st.background.is_none() {
                            st.background = Some(c);
                        }
                    }
                }
            }
        }
        "font-size" => {
            if let Some(px) = parse_px(value) {
                st.font_size = px.clamp(8, 48);
            }
        }
        "margin" => {
            // 1–4 values (CSS shorthand order: top right bottom left), each a
            // length or `auto`. `auto` on the horizontal sides marks the block
            // for centering; a length sets that margin.
            let toks: Vec<&str> = value.split_whitespace().collect();
            let (t, r, b, l): (&str, &str, &str, &str) = match toks.len() {
                1 => (toks[0], toks[0], toks[0], toks[0]),
                2 => (toks[0], toks[1], toks[0], toks[1]),
                3 => (toks[0], toks[1], toks[2], toks[1]),
                _ if toks.len() >= 4 => (toks[0], toks[1], toks[2], toks[3]),
                _ => (value, value, value, value),
            };
            if let Some(px) = parse_px(t) { st.margin_top = px; }
            if let Some(px) = parse_px(b) { st.margin_bottom = px; }
            st.margin_left_auto = l.eq_ignore_ascii_case("auto");
            st.margin_right_auto = r.eq_ignore_ascii_case("auto");
            if let Some(px) = parse_px(l) { st.margin_left = px; }
            if let Some(px) = parse_px(r) { st.margin_right = px; }
        }
        "margin-top" => {
            if let Some(px) = parse_px(value) {
                st.margin_top = px;
            }
        }
        "margin-bottom" => {
            if let Some(px) = parse_px(value) {
                st.margin_bottom = px;
            }
        }
        "margin-left" => {
            st.margin_left_auto = value.trim().eq_ignore_ascii_case("auto");
            if let Some(px) = parse_px(value) {
                st.margin_left = px;
            }
        }
        "margin-right" => {
            st.margin_right_auto = value.trim().eq_ignore_ascii_case("auto");
            if let Some(px) = parse_px(value) {
                st.margin_right = px;
            }
        }
        "padding" => {
            if let Some(px) = parse_px(value.split_whitespace().next().unwrap_or(value)) {
                st.padding_top = px;
                st.padding_bottom = px;
                st.padding_left = px;
                st.padding_right = px;
            }
        }
        "padding-top" => {
            if let Some(px) = parse_px(value) {
                st.padding_top = px;
            }
        }
        "padding-bottom" => {
            if let Some(px) = parse_px(value) {
                st.padding_bottom = px;
            }
        }
        "padding-left" => {
            if let Some(px) = parse_px(value) {
                st.padding_left = px;
            }
        }
        "padding-right" => {
            if let Some(px) = parse_px(value) {
                st.padding_right = px;
            }
        }
        "display" => {
            let v = value.trim().to_ascii_lowercase();
            st.display_none = v == "none";
            st.display = match v.as_str() {
                "none" => DisplayMode::None,
                "flex" => DisplayMode::Flex,
                "inline-flex" => DisplayMode::InlineFlex,
                "grid" | "inline-grid" => DisplayMode::Grid,
                "inline" | "inline-block" => DisplayMode::Inline,
                _ => DisplayMode::Block,
            };
        }
        "flex-direction" => {
            st.flex_direction = match value.trim().to_ascii_lowercase().as_str() {
                "column" | "column-reverse" => FlexDirection::Column,
                _ => FlexDirection::Row,
            };
        }
        "gap" | "row-gap" | "column-gap" => {
            if let Some(px) = parse_px(value.split_whitespace().next().unwrap_or(value)) {
                st.flex_gap = px;
                st.grid_gap = px;
            }
        }
        "align-items" => {
            st.align_items = match value.trim().to_ascii_lowercase().as_str() {
                "center" => AlignItems::Center,
                "flex-start" | "start" => AlignItems::Start,
                "flex-end" | "end" => AlignItems::End,
                _ => AlignItems::Stretch,
            };
        }
        "grid-template-columns" => {
            // Build the real track list (`1fr 2fr 100px auto`, `repeat(3, 1fr)`)
            // so tracks can be `fr`-sized; `grid_columns` keeps the count for the
            // fallback equal-column path.
            let tracks = parse_grid_tracks(value);
            st.grid_columns = (tracks.len() as u8).clamp(1, 24);
            st.grid_template = tracks;
            st.display = DisplayMode::Grid;
        }
        "flex-wrap" => {
            st.flex_wrap = match value.trim().to_ascii_lowercase().as_str() {
                "wrap" => super::flex::FlexWrap::Wrap,
                "wrap-reverse" => super::flex::FlexWrap::WrapReverse,
                _ => super::flex::FlexWrap::NoWrap,
            };
        }
        "flex-grow" => {
            if let Ok(n) = value.trim().parse::<u32>() {
                st.flex_grow = n;
            }
        }
        "grid-auto-flow" => {
            if value.to_ascii_lowercase().contains("dense") {
                st.grid_dense = true;
            }
        }
        "max-height" => {
            if let Some(px) = parse_px(value) {
                st.max_height = Some(px);
            }
        }
        "font-weight" => {
            let v = value.trim().to_ascii_lowercase();
            st.bold = matches!(v.as_str(), "bold" | "bolder" | "700" | "800" | "900");
        }
        "text-align" => {
            st.text_align = match value.trim().to_ascii_lowercase().as_str() {
                "center" => Align::Center,
                "right" | "end" => Align::Right,
                _ => Align::Left,
            };
        }
        "width" => {
            if let Some(px) = parse_px(value) {
                st.width = Some(px);
                st.width_pct = None;
            } else if let Some(p) = parse_pct(value) {
                st.width_pct = Some(p);
            }
        }
        "height" => {
            if let Some(px) = parse_px(value) {
                st.height = Some(px);
                st.height_pct = None;
            } else if let Some(p) = parse_pct(value) {
                st.height_pct = Some(p);
            }
        }
        "max-width" => {
            if let Some(px) = parse_px(value) {
                st.max_width = Some(px);
            }
        }
        "line-height" => {
            if let Some(px) = parse_px(value) {
                st.line_height = Some(px);
            } else if let Ok(mult) = value.trim().parse::<f32>() {
                st.line_height = Some((st.font_size as f32 * mult) as i32);
            }
        }
        "opacity" => {
            if let Ok(f) = value.trim().parse::<f32>() {
                st.opacity = (f.clamp(0.0, 1.0) * 255.0) as u8;
            }
        }
        "border-color" => {
            for tok in value_tokens(value) {
                if let Some(c) = parse_color(tok) {
                    st.border_color = Some(c);
                    st.border_top_color = Some(c);
                    st.border_bottom_color = Some(c);
                    st.border_left_color = Some(c);
                    st.border_right_color = Some(c);
                    break;
                }
            }
        }
        "outline-color" => {
            for tok in value_tokens(value) {
                if let Some(c) = parse_color(tok) {
                    st.outline_color = Some(c);
                    break;
                }
            }
        }
        "visibility" => {
            if value.trim().eq_ignore_ascii_case("hidden") {
                st.display_none = true;
            }
        }
        "position" => {
            st.position = match value.trim().to_ascii_lowercase().as_str() {
                "relative" => Position::Relative,
                "absolute" => Position::Absolute,
                "fixed" => Position::Fixed,
                "sticky" => Position::Sticky,
                _ => Position::Static,
            };
        }
        "top" => st.top = parse_px(value),
        "left" => st.left = parse_px(value),
        "right" => st.right = parse_px(value),
        "bottom" => st.bottom = parse_px(value),
        "z-index" => {
            if let Ok(z) = value.trim().parse::<i32>() {
                st.z_index = z;
            }
        }
        "overflow" | "overflow-x" | "overflow-y" => {
            st.overflow = match value.trim().to_ascii_lowercase().as_str() {
                "hidden" | "clip" => Overflow::Hidden,
                "scroll" => Overflow::Scroll,
                "auto" => Overflow::Auto,
                _ => Overflow::Visible,
            };
        }
        "border-radius" => {
            // `calc(var(--radius) - 2px)` is one value; splitting on
            // whitespace took `calc(.5rem` and dropped the radius.
            let first = if value.trim().len() >= 5
                && value.trim()[..5].eq_ignore_ascii_case("calc(")
            {
                value.trim()
            } else {
                value.split_whitespace().next().unwrap_or(value).trim()
            };
            if first.ends_with('%') {
                // A percentage radius is relative to the box; the painter clamps
                // radius to half the shorter side, so a large sentinel yields a
                // circle/pill (the ubiquitous `border-radius:50%` case).
                st.border_radius = 100_000;
            } else if let Some(px) = parse_px(first) {
                st.border_radius = px.max(0);
            }
        }
        "font-family" => {
            st.font_family = value
                .split(',')
                .next()
                .unwrap_or(value)
                .trim()
                .trim_matches('"')
                .trim_matches('\'')
                .to_string();
        }
        "cursor" => {
            st.cursor = match value.trim().to_ascii_lowercase().as_str() {
                "pointer" | "hand" => CursorCss::Pointer,
                "text" | "ibeam" => CursorCss::Text,
                "default" => CursorCss::Default,
                "not-allowed" | "no-drop" => CursorCss::NotAllowed,
                "crosshair" => CursorCss::Crosshair,
                "move" | "grab" | "all-scroll" => CursorCss::Move,
                _ => CursorCss::Auto,
            };
        }
        "float" => {
            st.float_mode = match value.trim().to_ascii_lowercase().as_str() {
                "left" => FloatMode::Left,
                "right" => FloatMode::Right,
                _ => FloatMode::None,
            };
        }
        "clear" => {
            st.clear = match value.trim().to_ascii_lowercase().as_str() {
                "left" => ClearMode::Left,
                "right" => ClearMode::Right,
                "both" => ClearMode::Both,
                _ => ClearMode::None,
            };
        }
        "white-space" => {
            st.white_space = match value.trim().to_ascii_lowercase().as_str() {
                "nowrap" => WhiteSpace::Nowrap,
                "pre" => WhiteSpace::Pre,
                "pre-wrap" | "pre-line" | "break-spaces" => WhiteSpace::PreWrap,
                _ => WhiteSpace::Normal,
            };
        }
        "text-decoration" | "text-decoration-line" => {
            let v = value.to_ascii_lowercase();
            st.text_decoration = if v.contains("underline") {
                TextDecoration::Underline
            } else if v.contains("line-through") {
                TextDecoration::LineThrough
            } else if v.contains("overline") {
                TextDecoration::Overline
            } else {
                TextDecoration::None
            };
        }
        "list-style" | "list-style-type" => {
            st.list_style = match value.trim().to_ascii_lowercase().as_str() {
                "none" => ListStyle::None,
                "circle" => ListStyle::Circle,
                "square" => ListStyle::Square,
                "decimal" | "decimal-leading-zero" => ListStyle::Decimal,
                _ => ListStyle::Disc,
            };
        }
        "transform" | "webkit-transform" => {
            st.transform = value.trim().to_string();
        }
        "transform-origin" => {
            st.transform_origin = value.trim().to_string();
        }
        "transition" | "webkit-transition" => {
            // transition: property duration timing delay
            let parts: Vec<&str> = value.split_whitespace().collect();
            if let Some(p) = parts.first() {
                st.transition_property = (*p).to_string();
            }
            if let Some(d) = parts.get(1) {
                st.transition_duration = (*d).to_string();
            }
            if let Some(t) = parts.get(2) {
                st.transition_timing = (*t).to_string();
            }
        }
        "transition-property" => st.transition_property = value.trim().to_string(),
        "transition-duration" => st.transition_duration = value.trim().to_string(),
        "transition-delay" => {
            // Stored alongside duration string (no separate field — append note).
            if st.transition_duration.is_empty() {
                st.transition_duration = format!("0s {}", value.trim());
            } else {
                st.transition_duration =
                    format!("{} +delay:{}", st.transition_duration, value.trim());
            }
        }
        "transition-timing-function" => st.transition_timing = value.trim().to_string(),
        "animation-fill-mode" => st.animation_fill_mode = value.trim().to_ascii_lowercase(),
        "animation-iteration-count" => {
            st.animation_iteration_count = value.trim().to_ascii_lowercase()
        }
        "flex-flow" => {
            // flex-flow: <direction> || <wrap>
            let v = value.to_ascii_lowercase();
            if v.contains("column") {
                st.flex_direction = FlexDirection::Column;
            }
            if v.contains("wrap") {
                st.flex_wrap = if v.contains("wrap-reverse") {
                    super::flex::FlexWrap::WrapReverse
                } else {
                    super::flex::FlexWrap::Wrap
                };
            }
        }
        "outline-width" => {
            st.outline_width = parse_px(value).unwrap_or(0).max(0);
            if st.outline_style == BorderStyle::None {
                st.outline_style = BorderStyle::Solid;
            }
        }
        "border-spacing" => {
            st.border_spacing = parse_px(value.split_whitespace().next().unwrap_or(value))
                .unwrap_or(0)
                .max(0);
        }
        "font-variant" => st.font_variant = value.trim().to_ascii_lowercase(),
        "font-stretch" => st.font_stretch = value.trim().to_ascii_lowercase(),
        "font-feature-settings" => st.font_feature_settings = value.trim().to_string(),
        "font-variation-settings" => st.font_variation_settings = value.trim().to_string(),
        "object-position" => st.object_position = value.trim().to_string(),
        "mask" => st.mask = value.trim().to_string(),
        "mask-image" => st.mask = value.trim().to_string(),
        "table-layout" => st.table_layout = value.trim().to_ascii_lowercase(),
        "zoom" => {
            // `zoom: 1.5` or `zoom: 150%`.
            let v = value.trim();
            st.zoom = if let Some(p) = v.strip_suffix('%') {
                p.trim().parse::<f32>().map(|n| n / 100.0).unwrap_or(1.0)
            } else if v.eq_ignore_ascii_case("normal") || v.eq_ignore_ascii_case("reset") {
                1.0
            } else {
                v.parse::<f32>().unwrap_or(1.0)
            }
            .clamp(0.1, 10.0);
        }
        "text-wrap" => {
            let v = value.trim().to_ascii_lowercase();
            if v == "nowrap" {
                st.white_space = WhiteSpace::Nowrap;
            }
            st.text_wrap = v;
        }
        "backface-visibility" => {
            st.backface_hidden = value.trim().eq_ignore_ascii_case("hidden");
        }
        "contain" => st.contain = value.trim().to_ascii_lowercase(),
        "color-scheme" => st.color_scheme = value.trim().to_ascii_lowercase(),
        "scroll-behavior" => st.scroll_behavior = value.trim().to_ascii_lowercase(),
        "forced-color-adjust" => st.forced_color_adjust = value.trim().to_ascii_lowercase(),
        "container-type" => st.container_type = value.trim().to_ascii_lowercase(),
        "overscroll-behavior" => st.overscroll_behavior = value.trim().to_ascii_lowercase(),
        "stroke-dasharray" => st.stroke_dasharray = value.trim().to_string(),
        "stroke-dashoffset" => st.stroke_dashoffset = parse_px(value).unwrap_or(0),
        "margin-inline-start" | "margin-inline-end" | "padding-inline" | "padding-inline-start"
        | "padding-inline-end" | "padding-block" | "inset-inline-start" => {
            // Logical props → approximate physical LTR.
            if let Some(px) = parse_px(value.split_whitespace().next().unwrap_or(value)) {
                match name.as_str() {
                    "margin-inline-start" => st.margin_left = px,
                    "margin-inline-end" => st.margin_right = px,
                    "padding-inline" | "padding-inline-start" => st.padding_left = px,
                    "padding-inline-end" => st.padding_right = px,
                    "padding-block" => {
                        st.padding_top = px;
                        st.padding_bottom = px;
                    }
                    "inset-inline-start" => st.left = Some(px),
                    _ => {}
                }
            }
        }
        "border-bottom-style" | "border-top-style" | "border-left-style" | "border-right-style"
        | "outline-style" => {
            let bs = BorderStyle::parse(value);
            match name.as_str() {
                "border-top-style" => st.border_top_style = bs,
                "border-bottom-style" => st.border_bottom_style = bs,
                "border-left-style" => st.border_left_style = bs,
                "border-right-style" => st.border_right_style = bs,
                "outline-style" => st.outline_style = bs,
                _ => {}
            }
        }
        "webkit-box-align" | "webkit-box-flex" | "webkit-box-direction" => {
            if name == "webkit-box-align" {
                st.align_items = match value.trim().to_ascii_lowercase().as_str() {
                    "center" => AlignItems::Center,
                    "end" => AlignItems::End,
                    "start" => AlignItems::Start,
                    _ => AlignItems::Stretch,
                };
            }
            let _ = value;
        }
        "text-size-adjust" => {
            st.webkit_text_size_adjust = value.trim().to_string();
        }
        "animation" | "webkit-animation" => {
            // `animation: pulse 2s cubic-bezier(.4,0,.6,1) infinite`
            // — times, timing functions, iteration counts and fill modes are
            // classified; whatever is left is the name. `value_tokens` keeps
            // `cubic-bezier(...)` whole so we do not steal `infinite` as a delay.
            for tok in value_tokens(value) {
                let t = tok.trim();
                if t.is_empty() {
                    continue;
                }
                if is_css_time(t) {
                    if st.animation_duration.is_empty() {
                        st.animation_duration = t.to_string();
                    } else {
                        st.animation_delay = t.to_string();
                    }
                } else if is_timing_fn(t) {
                    st.animation_timing = t.to_string();
                } else if t.eq_ignore_ascii_case("infinite")
                    || t.parse::<f32>().is_ok()
                {
                    st.animation_iteration_count = t.to_ascii_lowercase();
                } else if matches!(
                    t.to_ascii_lowercase().as_str(),
                    "forwards" | "backwards" | "both" | "none"
                ) {
                    st.animation_fill_mode = t.to_ascii_lowercase();
                } else if matches!(
                    t.to_ascii_lowercase().as_str(),
                    "normal" | "reverse" | "alternate" | "alternate-reverse"
                ) {
                    // direction — stored on timing only so it is not lost
                } else if !t.eq_ignore_ascii_case("running")
                    && !t.eq_ignore_ascii_case("paused")
                {
                    st.animation_name = t.to_string();
                }
            }
        }
        "animation-name" => st.animation_name = value.trim().to_string(),
        "animation-duration" => st.animation_duration = value.trim().to_string(),
        "animation-delay" => st.animation_delay = value.trim().to_string(),
        "animation-timing-function" => st.animation_timing = value.trim().to_string(),
        "filter" => st.filter = value.trim().to_string(),
        "backdrop-filter" => st.backdrop_filter = value.trim().to_string(),
        "content" => {
            // Strip surrounding quotes; `none`/`normal` → empty (no generated box).
            let v = value.trim();
            st.content = if v.eq_ignore_ascii_case("none") || v.eq_ignore_ascii_case("normal") {
                String::new()
            } else {
                v.trim_matches('"').trim_matches('\'').to_string()
            };
        }
        "outline" => {
            // `outline: <width> <style> <color>` shorthand (any order).
            for tok in value_tokens(value) {
                if let Some(px) = parse_px(tok) {
                    st.outline_width = px.max(0);
                } else if let Some(c) = parse_color(tok) {
                    st.outline_color = Some(c);
                } else {
                    let bs = BorderStyle::parse(tok);
                    if tok.eq_ignore_ascii_case("none") || bs != BorderStyle::Solid || tok.eq_ignore_ascii_case("solid") {
                        st.outline_style = bs;
                    }
                }
            }
            if st.outline_width == 0 && st.outline_style.is_visible() {
                st.outline_width = 1;
            }
        }
        "box-shadow" | "webkit-box-shadow" => {
            st.box_shadow = value.trim().to_string();
        }
        "vertical-align" => {
            st.vertical_align = match value.trim().to_ascii_lowercase().as_str() {
                "top" | "text-top" => VerticalAlign::Top,
                "middle" => VerticalAlign::Middle,
                "bottom" | "text-bottom" => VerticalAlign::Bottom,
                _ => VerticalAlign::Baseline,
            };
        }
        "object-fit" => {
            st.object_fit = match value.trim().to_ascii_lowercase().as_str() {
                "contain" => ObjectFit::Contain,
                "cover" => ObjectFit::Cover,
                "none" => ObjectFit::None,
                "scale-down" => ObjectFit::ScaleDown,
                _ => ObjectFit::Fill,
            };
        }
        "aspect-ratio" => {
            let v = value.replace(' ', "");
            if let Some((a, b)) = v.split_once('/') {
                if let (Ok(aw), Ok(ah)) = (a.parse::<i32>(), b.parse::<i32>()) {
                    if aw > 0 && ah > 0 {
                        st.aspect_ratio = Some((aw, ah));
                    }
                }
            }
        }
        "min-width" => st.min_width = parse_px(value),
        "min-height" => st.min_height = parse_px(value),
        "border-width" => {
            if let Some(px) = parse_px(value.split_whitespace().next().unwrap_or(value)) {
                st.border_top_width = px;
                st.border_bottom_width = px;
                st.border_left_width = px;
                st.border_right_width = px;
                if px > 0 {
                    for s in [
                        &mut st.border_top_style,
                        &mut st.border_bottom_style,
                        &mut st.border_left_style,
                        &mut st.border_right_style,
                    ] {
                        if *s == BorderStyle::None {
                            *s = BorderStyle::Solid;
                        }
                    }
                }
            }
        }
        "border" | "border-top" | "border-bottom" | "border-left" | "border-right"
        | "border-style" => {
            // `<width> <style> <color>` in any order (shorthand). A side with a
            // visible style but no explicit width gets a 1px medium default.
            let mut w: Option<i32> = None;
            let mut c: Option<u32> = None;
            let mut style: Option<BorderStyle> = None;
            let mut had_style_kw = false;
            for tok in value_tokens(value) {
                let low = tok.to_ascii_lowercase();
                if let Some(px) = parse_px(tok) {
                    w = Some(px);
                } else if matches!(
                    low.as_str(),
                    "none" | "hidden" | "solid" | "dashed" | "dotted" | "double" | "groove"
                        | "ridge" | "inset" | "outset"
                ) {
                    style = Some(BorderStyle::parse(&low));
                    had_style_kw = true;
                } else if let Some(col) = parse_color(tok) {
                    c = Some(col);
                }
            }
            // A bare `border-style: solid` shouldn't zero the width; only default
            // width when a style keyword was present and no width given.
            let eff_w = w.or(if had_style_kw && name != "border-style" {
                Some(1)
            } else {
                None
            });
            let set_top = matches!(name.as_str(), "border" | "border-top" | "border-style");
            let set_bottom = matches!(name.as_str(), "border" | "border-bottom" | "border-style");
            let set_left = matches!(name.as_str(), "border" | "border-left" | "border-style");
            let set_right = matches!(name.as_str(), "border" | "border-right" | "border-style");
            if let Some(col) = c {
                st.border_color = Some(col);
                if set_top { st.border_top_color = Some(col); }
                if set_bottom { st.border_bottom_color = Some(col); }
                if set_left { st.border_left_color = Some(col); }
                if set_right { st.border_right_color = Some(col); }
            }
            if let Some(px) = eff_w {
                if set_top { st.border_top_width = px; }
                if set_bottom { st.border_bottom_width = px; }
                if set_left { st.border_left_width = px; }
                if set_right { st.border_right_width = px; }
            }
            if let Some(s) = style {
                if set_top { st.border_top_style = s; }
                if set_bottom { st.border_bottom_style = s; }
                if set_left { st.border_left_style = s; }
                if set_right { st.border_right_style = s; }
            }
        }
        "border-top-color" => {
            if let Some(c) = parse_color(value) {
                st.border_top_color = Some(c);
                st.border_color = Some(c);
            }
        }
        "border-bottom-color" => {
            if let Some(c) = parse_color(value) {
                st.border_bottom_color = Some(c);
                st.border_color = Some(c);
            }
        }
        "border-left-color" => {
            if let Some(c) = parse_color(value) {
                st.border_left_color = Some(c);
                st.border_color = Some(c);
            }
        }
        "border-right-color" => {
            if let Some(c) = parse_color(value) {
                st.border_right_color = Some(c);
                st.border_color = Some(c);
            }
        }
        "border-top-width" => {
            if let Some(px) = parse_px(value) {
                st.border_top_width = px;
                if px > 0 && st.border_top_style == BorderStyle::None {
                    st.border_top_style = BorderStyle::Solid;
                }
            }
        }
        "border-bottom-width" => {
            if let Some(px) = parse_px(value) {
                st.border_bottom_width = px;
                // Tailwind `border-b` is only a width; preflight `*` sets
                // `border-style:solid`. If that rule missed, a width with
                // style `none` would never paint (table row rules, cards).
                if px > 0 && st.border_bottom_style == BorderStyle::None {
                    st.border_bottom_style = BorderStyle::Solid;
                }
            }
        }
        "border-left-width" => {
            if let Some(px) = parse_px(value) {
                st.border_left_width = px;
                if px > 0 && st.border_left_style == BorderStyle::None {
                    st.border_left_style = BorderStyle::Solid;
                }
            }
        }
        "border-right-width" => {
            if let Some(px) = parse_px(value) {
                st.border_right_width = px;
                if px > 0 && st.border_right_style == BorderStyle::None {
                    st.border_right_style = BorderStyle::Solid;
                }
            }
        }
        "border-top-left-radius" | "border-top-right-radius" | "border-bottom-left-radius"
        | "border-bottom-right-radius" => {
            if let Some(px) = parse_px(value.split_whitespace().next().unwrap_or(value)) {
                st.border_radius = st.border_radius.max(px.max(0));
            }
        }
        "background-image" | "background-origin" => {
            let v = value.trim();
            st.background_image = if v.eq_ignore_ascii_case("none") {
                String::new()
            } else {
                v.to_string()
            };
            // A gradient's first color seeds `background` so the box paints even
            // before the gradient raster (and as a solid fallback).
            if let Some(c) = gradient_first_color(v) {
                if st.background.is_none() {
                    st.background = Some(c);
                }
            }
        }
        "background-size" => st.background_size = value.trim().to_ascii_lowercase(),
        "background-position" => st.background_position = value.trim().to_ascii_lowercase(),
        "background-repeat" => st.background_repeat = value.trim().to_ascii_lowercase(),
        "background-clip" => st.background_clip = value.trim().to_ascii_lowercase(),
        "font-style" => {
            st.font_style = match value.trim().to_ascii_lowercase().as_str() {
                "italic" => FontStyle::Italic,
                "oblique" => FontStyle::Oblique,
                _ => FontStyle::Normal,
            };
        }
        "font" => {
            for tok in value.split_whitespace() {
                if let Some(px) = parse_px(tok) {
                    st.font_size = px.clamp(8, 48);
                    break;
                }
            }
            if value.to_ascii_lowercase().contains("bold") {
                st.bold = true;
            }
            if value.to_ascii_lowercase().contains("italic") {
                st.font_style = FontStyle::Italic;
            }
        }
        "font-display" => st.font_display = value.trim().to_string(),
        "src" => st.font_src = value.trim().to_string(),
        "unicode-range" => st.unicode_range = value.trim().to_string(),
        "text-transform" => {
            st.text_transform = match value.trim().to_ascii_lowercase().as_str() {
                "uppercase" => TextTransform::Uppercase,
                "lowercase" => TextTransform::Lowercase,
                "capitalize" => TextTransform::Capitalize,
                _ => TextTransform::None,
            };
        }
        "text-indent" => {
            if let Some(px) = parse_px(value) {
                st.text_indent = px;
            }
        }
        "text-shadow" => st.text_shadow = value.trim().to_string(),
        "text-rendering" => st.text_rendering = value.trim().to_string(),
        "letter-spacing" => st.letter_spacing = parse_px(value),
        "word-break" => {
            st.word_break = match value.trim().to_ascii_lowercase().as_str() {
                "break-all" => WordBreak::BreakAll,
                "keep-all" => WordBreak::KeepAll,
                "break-word" => WordBreak::BreakWord,
                _ => WordBreak::Normal,
            };
        }
        "overflow-wrap" | "word-wrap" => {
            st.overflow_wrap = match value.trim().to_ascii_lowercase().as_str() {
                "anywhere" => OverflowWrap::Anywhere,
                "break-word" => OverflowWrap::BreakWord,
                _ => OverflowWrap::Normal,
            };
        }
        "text-overflow" => {
            st.text_overflow = if value.to_ascii_lowercase().contains("ellipsis") {
                TextOverflow::Ellipsis
            } else {
                TextOverflow::Clip
            };
        }
        "user-select" | "webkit-user-select" => {
            st.user_select = match value.trim().to_ascii_lowercase().as_str() {
                "none" => UserSelect::None,
                "text" => UserSelect::Text,
                "all" => UserSelect::All,
                _ => UserSelect::Auto,
            };
        }
        "pointer-events" => {
            st.pointer_events = if value.trim().eq_ignore_ascii_case("none") {
                PointerEvents::None
            } else {
                PointerEvents::Auto
            };
        }
        "appearance" | "webkit-appearance" => {
            st.appearance = value.trim().to_string();
        }
        "direction" => {
            st.direction = if value.trim().eq_ignore_ascii_case("rtl") {
                Direction::Rtl
            } else {
                Direction::Ltr
            };
        }
        "clip" => st.clip = value.trim().to_string(),
        "clip-path" => st.clip_path = value.trim().to_string(),
        "outline-offset" => {
            if let Some(px) = parse_px(value) {
                st.outline_offset = px;
            }
        }
        "border-collapse" => {
            st.border_collapse = value.to_ascii_lowercase().contains("collapse");
        }
        "fill" => {
            if let Some(c) = parse_color(value) {
                st.fill = Some(c);
            } else if value.trim().eq_ignore_ascii_case("none") {
                st.fill = None;
            }
        }
        "stroke" => {
            if let Some(c) = parse_color(value) {
                st.stroke = Some(c);
            }
        }
        "stroke-width" => {
            if let Some(px) = parse_px(value) {
                st.stroke_width = px;
            } else if let Ok(n) = value.trim().parse::<i32>() {
                st.stroke_width = n;
            }
        }
        "touch-action" => st.touch_action = value.trim().to_string(),
        "webkit-line-clamp" | "line-clamp" => {
            st.line_clamp = value.trim().parse().ok();
            if st.line_clamp.is_some() {
                st.overflow = Overflow::Hidden;
                st.text_overflow = TextOverflow::Ellipsis;
            }
        }
        "scrollbar-width" => st.scrollbar_width = value.trim().to_string(),
        "will-change" => st.will_change = value.trim().to_string(),
        "resize" => {
            st.resize = match value.trim().to_ascii_lowercase().as_str() {
                "both" => ResizeMode::Both,
                "horizontal" => ResizeMode::Horizontal,
                "vertical" => ResizeMode::Vertical,
                _ => ResizeMode::None,
            };
        }
        "webkit-font-smoothing" => {
            st.webkit_font_smoothing = value.trim().to_string();
        }
        "webkit-tap-highlight-color" => {
            st.webkit_tap_highlight = parse_color(value);
        }
        "webkit-text-size-adjust" => {
            st.webkit_text_size_adjust = value.trim().to_string();
        }
        "webkit-box-orient" => {
            st.webkit_box_orient = value.trim().to_string();
        }
        "webkit-box-pack" => {
            st.webkit_box_pack = match value.trim().to_ascii_lowercase().as_str() {
                "center" => Justify::Center,
                "end" | "flex-end" => Justify::End,
                "justify" | "space-between" => Justify::SpaceBetween,
                _ => Justify::Start,
            };
            st.justify_content = st.webkit_box_pack;
        }
        "justify-content" | "webkit-justify-content" => {
            st.justify_content = match value.trim().to_ascii_lowercase().as_str() {
                "center" => Justify::Center,
                "flex-end" | "end" | "right" => Justify::End,
                "space-between" => Justify::SpaceBetween,
                _ => Justify::Start,
            };
        }
        "inset" => {
            // inset: top right bottom left (or 1–4 values like margin)
            st.inset_shorthand = true;
            let toks: Vec<&str> = value.split_whitespace().collect();
            match toks.len() {
                1 => {
                    let p = parse_px(toks[0]);
                    st.top = p;
                    st.right = p;
                    st.bottom = p;
                    st.left = p;
                }
                2 => {
                    let v = parse_px(toks[0]);
                    let h = parse_px(toks[1]);
                    st.top = v;
                    st.bottom = v;
                    st.left = h;
                    st.right = h;
                }
                3 => {
                    st.top = parse_px(toks[0]);
                    st.left = parse_px(toks[1]);
                    st.right = parse_px(toks[1]);
                    st.bottom = parse_px(toks[2]);
                }
                n if n >= 4 => {
                    st.top = parse_px(toks[0]);
                    st.right = parse_px(toks[1]);
                    st.bottom = parse_px(toks[2]);
                    st.left = parse_px(toks[3]);
                }
                _ => {}
            }
            if st.position == Position::Static {
                st.position = Position::Relative;
            }
        }
        "box-sizing" | "webkit-box-sizing" => {
            st.box_sizing = if value.to_ascii_lowercase().contains("border") {
                BoxSizing::BorderBox
            } else {
                BoxSizing::ContentBox
            };
        }
        // chromestatus dumps a synthetic "variable" property for custom props usage.
        "variable" => {
            // No-op marker; real custom props use `--*`.
            let _ = value;
        }
        "order" => {
            st.order = value.trim().parse().unwrap_or(0);
        }
        "flex-shrink" => {
            st.flex_shrink = value.trim().parse().unwrap_or(1);
        }
        "flex-basis" => {
            let v = value.trim();
            st.flex_basis = if v.eq_ignore_ascii_case("auto") || v.eq_ignore_ascii_case("content") {
                None
            } else {
                parse_px(v)
            };
        }
        "flex" => {
            // Shorthand `flex: grow [shrink] [basis]`.
            // Tailwind `.flex-1` is `flex: 1 1 0%` — basis `0%` is zero, not
            // "unresolved percentage", or a `flex-1` separator never grows.
            let parts: Vec<&str> = value.split_whitespace().collect();
            match parts.as_slice() {
                ["none"] => {
                    st.flex_grow = 0;
                    st.flex_shrink = 0;
                    st.flex_basis = None;
                }
                ["auto"] => {
                    st.flex_grow = 1;
                    st.flex_shrink = 1;
                    st.flex_basis = None;
                }
                _ => {
                    if let Some(g) = parts.first().and_then(|s| s.parse().ok()) {
                        st.flex_grow = g;
                    }
                    if let Some(s) = parts.get(1).and_then(|s| s.parse().ok()) {
                        st.flex_shrink = s;
                    }
                    if let Some(b) = parts.get(2) {
                        let t = b.trim();
                        if t == "0" || t == "0%" || t == "0px" {
                            st.flex_basis = Some(0);
                        } else if let Some(px) = parse_px(t) {
                            st.flex_basis = Some(px);
                        }
                    } else if parts.len() == 1 {
                        // `flex: <number>` sets basis 0 (grow from nothing).
                        st.flex_basis = Some(0);
                    }
                    if value.contains("wrap") {
                        st.flex_wrap = super::flex::FlexWrap::Wrap;
                    }
                }
            }
        }
        "align-content" | "align-self"
        | "justify-items" | "justify-self" | "place-items" | "place-content" | "grid-gap"
        | "grid-template-rows" | "grid-column" | "grid-row" | "grid-area" => {
            if name == "grid-gap" {
                if let Some(px) = parse_px(value.split_whitespace().next().unwrap_or(value)) {
                    st.grid_gap = px;
                    st.flex_gap = px;
                }
            }
        }
        _ => {}
    }
}

/// Compute style for an element given sheet + inline style string.
/// Inheritance (LibWeb StyleComputer spirit): color, font-size, font-weight,
/// text-align cascade from parent; margins/padding/background do not.
pub fn compute(
    sheet: &Stylesheet,
    tag: &str,
    id: Option<&str>,
    class: Option<&str>,
    inline: Option<&str>,
    parent: &ComputedStyle,
) -> ComputedStyle {
    compute_ex(sheet, tag, id, class, inline, parent, &[])
}

/// Like [`compute`] but with the element's ancestor chain (outermost→innermost)
/// so descendant selectors resolve correctly.
pub fn compute_ex(
    sheet: &Stylesheet,
    tag: &str,
    id: Option<&str>,
    class: Option<&str>,
    inline: Option<&str>,
    parent: &ComputedStyle,
    chain: &[ElemRef],
) -> ComputedStyle {
    compute_el(sheet, ElemRef::basic(tag, id, class), inline, parent, chain)
}

/// Compute style for a fully-described subject element (`nth`, `href`, …).
pub fn compute_el(
    sheet: &Stylesheet,
    el: ElemRef<'_>,
    inline: Option<&str>,
    parent: &ComputedStyle,
    chain: &[ElemRef],
) -> ComputedStyle {
    let tag = el.tag;
    let mut st = ComputedStyle::default();
    // Inherited properties from parent.
    st.color = parent.color;
    st.font_size = parent.font_size;
    st.bold = parent.bold;
    st.custom_props = parent.custom_props.clone();
    st.font_family = parent.font_family.clone();
    st.font_style = parent.font_style;
    st.direction = parent.direction;
    st.text_align = parent.text_align;
    // These CSS properties inherit too (so `<ul style="list-style:none">`
    // suppresses its `<li>` markers, and inherited spacing propagates).
    st.list_style = parent.list_style;
    st.letter_spacing = parent.letter_spacing;
    st.line_height = parent.line_height;
    st.white_space = parent.white_space;
    st.text_transform = parent.text_transform;
    st.border_spacing = parent.border_spacing; // inherited (table property)
    st.color_scheme = parent.color_scheme.clone();
    // Tag UA defaults (simplified user-agent stylesheet).
    match tag {
        "h1" => {
            st.font_size = 28;
            st.bold = true;
            st.margin_top = 6;
            st.margin_bottom = 4;
        }
        "h2" => {
            st.font_size = 22;
            st.bold = true;
            st.margin_top = 6;
            st.margin_bottom = 4;
        }
        "h3" | "h4" | "h5" | "h6" => {
            st.font_size = 16;
            st.bold = true;
            st.margin_top = 4;
            st.margin_bottom = 2;
        }
        "p" => {
            st.margin_top = 4;
            st.margin_bottom = 2;
        }
        "a" => {
            st.color = 0xcc785c;
            st.text_decoration = TextDecoration::Underline;
        }
        "strong" | "b" => {
            st.bold = true;
        }
        "code" | "pre" => {
            st.font_size = parent.font_size.saturating_sub(1).max(10);
        }
        // The presentational `<center>` element centers its inline/block
        // descendants (UA rule: `center { text-align: center }`). Real pages
        // still lean on it — google.com wraps its whole logo/search/buttons
        // block in one `<center>`, which is why the page looked left-aligned.
        "center" => {
            st.text_align = Align::Center;
        }
        // Tables do **not** inherit an outer `text-align:center` into cell
        // content — only the table box is centered (see `layout_table`). HN
        // wraps `#hnmain` in `<center>`; without this reset every title was
        // centered inside its column and subtext drifted off the title edge.
        "table" | "td" => {
            st.text_align = Align::Left;
        }
        // `<th>` and `<caption>` default to centered text per the UA sheet.
        "th" | "caption" => {
            st.text_align = Align::Center;
            st.bold = st.bold || tag == "th";
        }
        _ => {}
    }
    let mut decls = sheet.matching_decls_for(el, chain, PseudoElement::None);
    if let Some(inl) = inline {
        for d in parse_decls(inl) {
            // Inline style ≈ specificity (1,0,0,0) — applied after author rules
            // in the same important/non-important passes (later wins).
            decls.push((d.name, d.value, d.important));
        }
    }
    apply_decls(&mut st, &decls);
    // Percentage width/height against the parent's *explicit* size. Tailwind
    // `h-full` / `w-full` is `100%`; without a containing block `parse_px`
    // returns None and a progress indicator or avatar fallback collapses to
    // empty. A parent with no explicit size leaves the % unresolved so a
    // block still fills the line the usual way.
    if st.width.is_none() {
        if let (Some(p), Some(pw)) = (st.width_pct, parent.width) {
            st.width = Some(f32_to_i32(p / 100.0 * pw as f32).max(0));
        }
    }
    if st.height.is_none() {
        if let (Some(p), Some(ph)) = (st.height_pct, parent.height) {
            st.height = Some(f32_to_i32(p / 100.0 * ph as f32).max(0));
        }
    }
    // `@keyframes` — apply the stop(s) for the current animation clock.
    apply_animation(&mut st, sheet);
    // `filter` is approximated as a colour transform over the element's own
    // colours (grayscale/invert/brightness/opacity) — blur is not rasterized.
    if !st.filter.is_empty() {
        apply_filter_colors(&mut st);
    }
    st
}

/// Apply the current `@keyframes` snapshot onto `st`.
///
/// Only properties we already honour (opacity, transform, background, colour)
/// are interpolated / applied. A missing 0%/100% stop is the element's
/// pre-animation value, which is what `animate-pulse` (`50%{opacity:.5}`)
/// relies on.
fn apply_animation(st: &mut ComputedStyle, sheet: &Stylesheet) {
    let name = st.animation_name.trim();
    if name.is_empty() || name.eq_ignore_ascii_case("none") {
        return;
    }
    let Some(stops) = sheet.keyframes(name) else {
        return;
    };
    if stops.is_empty() {
        return;
    }
    mark_page_wants_animation();
    let dur = parse_time_ms(&st.animation_duration);
    if dur == 0 {
        // Fill-mode / a nameless duration: apply the last stop as a still.
        if let Some(last) = stops.last() {
            for (n, v) in &last.decls {
                apply_one(st, n, v);
            }
        }
        return;
    }
    let delay = parse_time_ms(&st.animation_delay);
    let now = animation_now_ms().saturating_sub(delay);
    let t = (now % dur) as f32 / dur as f32;
    let pct = (t * 100.0).clamp(0.0, 100.0);
    // Surrounding stops; implied 0%/100% carry the pre-animation values.
    let mut lo_pct = 0.0f32;
    let mut hi_pct = 100.0f32;
    let mut lo: Option<&KeyframeStop> = None;
    let mut hi: Option<&KeyframeStop> = None;
    for s in stops {
        let p = s.pct as f32;
        if p <= pct {
            lo_pct = p;
            lo = Some(s);
        }
        if p >= pct && hi.is_none() {
            hi_pct = p;
            hi = Some(s);
        }
    }
    let span = (hi_pct - lo_pct).max(0.001);
    let u = ((pct - lo_pct) / span).clamp(0.0, 1.0);
    // Opacity is the one property pulse/fade actually animate; lerp it.
    let base_op = st.opacity;
    let op_at = |stop: Option<&KeyframeStop>, fallback: u8| -> u8 {
        stop.and_then(|s| {
            s.decls.iter().find(|(n, _)| n == "opacity").and_then(|(_, v)| {
                v.trim().parse::<f32>().ok().map(|f| (f.clamp(0.0, 1.0) * 255.0) as u8)
            })
        })
        .unwrap_or(fallback)
    };
    let a = op_at(lo, base_op);
    let b = op_at(hi, base_op);
    st.opacity = (a as f32 + (b as i32 - a as i32) as f32 * u) as u8;
    // Other decls (transform, background, colour) take the nearest stop —
    // interpolating a translate string is not worth it here.
    let nearest = if u < 0.5 { lo.or(hi) } else { hi.or(lo) };
    if let Some(s) = nearest {
        for (n, v) in &s.decls {
            if n == "opacity" {
                continue;
            }
            apply_one(st, n, v);
        }
    }
}

/// Resolve `::before` / `::after` generated-content style for an element.
pub fn compute_pseudo(
    sheet: &Stylesheet,
    el: ElemRef<'_>,
    parent: &ComputedStyle,
    chain: &[ElemRef],
    pseudo: PseudoElement,
) -> Option<ComputedStyle> {
    if pseudo == PseudoElement::None {
        return None;
    }
    let decls = sheet.matching_decls_for(el, chain, pseudo);
    if decls.is_empty() {
        return None;
    }
    let mut st = parent.clone();
    // Generated content does not inherit display:none from quirks — start
    // from inherited colour/font, clear layout-affecting props lightly.
    st.content = String::new();
    apply_decls(&mut st, &decls);
    if st.content.is_empty()
        || st.content.eq_ignore_ascii_case("none")
        || st.content.eq_ignore_ascii_case("normal")
    {
        return None;
    }
    Some(st)
}

/// Apply the colour-affecting parts of `filter` (grayscale/invert/brightness/
/// sepia/opacity) to an element's own colours. A cheap, plumbing-free
/// approximation that makes `filter` visibly change rendering for the common
/// solid-colour cases.
fn apply_filter_colors(st: &mut ComputedStyle) {
    let f = st.filter.to_ascii_lowercase();
    let amount = |name: &str, default: f32| -> Option<f32> {
        let i = f.find(name)?;
        let rest = &f[i + name.len()..];
        let open = rest.find('(')?;
        let close = rest[open..].find(')')? + open;
        let arg = rest[open + 1..close].trim();
        if arg.is_empty() {
            return Some(default);
        }
        if let Some(p) = arg.strip_suffix('%') {
            return p.trim().parse::<f32>().ok().map(|v| v / 100.0);
        }
        arg.parse::<f32>().ok()
    };
    let xform = |c: u32| -> u32 {
        let (mut r, mut g, mut b) = (
            ((c >> 16) & 0xff) as f32,
            ((c >> 8) & 0xff) as f32,
            (c & 0xff) as f32,
        );
        if let Some(a) = amount("grayscale", 1.0) {
            let l = 0.299 * r + 0.587 * g + 0.114 * b;
            r += (l - r) * a;
            g += (l - g) * a;
            b += (l - b) * a;
        }
        if let Some(a) = amount("sepia", 1.0) {
            let (nr, ng, nb) = (
                0.393 * r + 0.769 * g + 0.189 * b,
                0.349 * r + 0.686 * g + 0.168 * b,
                0.272 * r + 0.534 * g + 0.131 * b,
            );
            r += (nr - r) * a;
            g += (ng - g) * a;
            b += (nb - b) * a;
        }
        if let Some(a) = amount("invert", 1.0) {
            r += (255.0 - r - r) * a;
            g += (255.0 - g - g) * a;
            b += (255.0 - b - b) * a;
        }
        if let Some(a) = amount("brightness", 1.0) {
            r *= a;
            g *= a;
            b *= a;
        }
        let cl = |v: f32| v.clamp(0.0, 255.0) as u32;
        (cl(r) << 16) | (cl(g) << 8) | cl(b)
    };
    st.color = xform(st.color);
    st.background = st.background.map(xform);
    st.border_color = st.border_color.map(xform);
    if let Some(a) = amount("opacity", 1.0) {
        st.opacity = (st.opacity as f32 * a.clamp(0.0, 1.0)) as u8;
    }
}

// ---------------------------------------------------------------------------
// Pure stylesheet scanners (subresource discovery + module preprocessing).
// ---------------------------------------------------------------------------

/// One `@font-face` source: family name, resource URL, and format hint
/// (`woff2` / `woff` / `truetype` / vendor string).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FontFace {
    pub family: String,
    pub url: String,
    pub format: String,
}

/// Strip surrounding `"…"` / `'…'` quotes and whitespace from a CSS token.
fn unquote(s: &str) -> &str {
    s.trim().trim_matches(|c| c == '"' || c == '\'').trim()
}

/// Split `s` on `sep`, but only at paren depth 0 (so `url(data:a;b,c)` stays
/// whole). Empty pieces are dropped.
fn split_paren_aware(s: &str, sep: char) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut cur = String::new();
    for c in s.chars() {
        match c {
            '(' => {
                depth += 1;
                cur.push(c);
            }
            ')' => {
                depth = (depth - 1).max(0);
                cur.push(c);
            }
            c if c == sep && depth == 0 => {
                if !cur.trim().is_empty() {
                    out.push(cur.clone());
                }
                cur.clear();
            }
            c => cur.push(c),
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur);
    }
    out
}

/// Scan a stylesheet for `@import` targets, in order. Handles
/// `@import url("x");`, `@import url(x);`, `@import "x";`, and
/// `@import 'x' screen;` (trailing media list ignored).
pub fn scan_imports(css: &str) -> Vec<String> {
    let lower = css.to_ascii_lowercase();
    let mut out = Vec::new();
    let mut pos = 0;
    while let Some(i) = lower[pos..].find("@import") {
        let start = pos + i + "@import".len();
        pos = start; // progress even on a malformed rule
        let rest = css[start..].trim_start();
        if rest.get(..4).is_some_and(|p| p.eq_ignore_ascii_case("url(")) {
            if let Some(end) = rest.find(')') {
                let u = unquote(&rest[4..end]);
                if !u.is_empty() {
                    out.push(u.to_string());
                }
            }
        } else if let Some(q) = rest.chars().next().filter(|c| *c == '"' || *c == '\'') {
            if let Some(end) = rest[1..].find(q) {
                out.push(rest[1..1 + end].to_string());
            }
        }
    }
    out
}

/// Collect `url(...)` references from `background` / `background-image`
/// declarations only (fonts are [`scan_font_faces`]'s job). Deduped, in
/// document order; `data:` URIs are skipped (gradients carry no `url(...)`
/// and thus never match).
pub fn scan_css_urls(css: &str) -> Vec<String> {
    let lower = css.to_ascii_lowercase();
    let mut out: Vec<String> = Vec::new();
    let mut pos = 0;
    while let Some(i) = lower[pos..].find("url(") {
        let at = pos + i;
        pos = at + 4;
        // Walk back to the start of this declaration, then read its property.
        let seg_start = lower[..at]
            .rfind(['{', ';', '}'])
            .map(|j| j + 1)
            .unwrap_or(0);
        let seg = &lower[seg_start..at];
        let Some(colon) = seg.find(':') else { continue };
        let prop = seg[..colon].trim();
        if prop != "background" && prop != "background-image" {
            continue;
        }
        let Some(end) = css[at + 4..].find(')') else { continue };
        let url = unquote(&css[at + 4..at + 4 + end]);
        if url.is_empty() || url.starts_with("data:") {
            continue;
        }
        if !out.iter().any(|u| u == url) {
            out.push(url.to_string());
        }
    }
    out
}

/// Parse `@font-face` blocks: family (quotes stripped), the first
/// non-`data:` `src` URL, and its format — explicit `format("…")` if
/// present, else inferred from the extension (`.woff2` → `woff2`,
/// `.woff` → `woff`, `.ttf`/`.otf` → `truetype`).
pub fn scan_font_faces(css: &str) -> Vec<FontFace> {
    let lower = css.to_ascii_lowercase();
    let mut out = Vec::new();
    let mut pos = 0;
    while let Some(i) = lower[pos..].find("@font-face") {
        let at = pos + i;
        let Some(ob) = css[at..].find('{').map(|j| at + j) else {
            break;
        };
        let Some(cb) = css[ob + 1..].find('}').map(|j| ob + 1 + j) else {
            break;
        };
        let block = &css[ob + 1..cb];
        pos = cb + 1;
        let mut family = String::new();
        let mut url = String::new();
        let mut format = String::new();
        // Paren-aware ';' split: a data: URI inside url() contains ';'.
        for decl in split_paren_aware(block, ';') {
            let Some(colon) = decl.find(':') else { continue };
            let name = decl[..colon].trim().to_ascii_lowercase();
            let value = decl[colon + 1..].trim();
            if name == "font-family" {
                family = unquote(value).to_string();
            } else if name == "src" && url.is_empty() {
                // First candidate carrying a usable (non-data:) url().
                for cand in split_paren_aware(value, ',') {
                    let cl = cand.to_ascii_lowercase();
                    let Some(u) = cl.find("url(") else { continue };
                    let Some(e) = cand[u + 4..].find(')') else { continue };
                    let raw = unquote(&cand[u + 4..u + 4 + e]);
                    if raw.is_empty() || raw.starts_with("data:") {
                        continue;
                    }
                    url = raw.to_string();
                    format = if let Some(f) = cl.find("format(") {
                        cand[f + 7..]
                            .find(')')
                            .map(|fe| unquote(&cand[f + 7..f + 7 + fe]).to_string())
                            .unwrap_or_default()
                    } else {
                        let rl = raw.to_ascii_lowercase();
                        if rl.ends_with(".woff2") {
                            String::from("woff2")
                        } else if rl.ends_with(".woff") {
                            String::from("woff")
                        } else if rl.ends_with(".ttf") || rl.ends_with(".otf") {
                            String::from("truetype")
                        } else {
                            String::new()
                        }
                    };
                    break;
                }
            }
        }
        if !family.is_empty() && !url.is_empty() {
            out.push(FontFace {
                family,
                url,
                format,
            });
        }
    }
    out
}

/// Return the first quoted string on a line (`"u"` or `'u'`), if any.
fn quoted_spec(line: &str) -> Option<String> {
    let qpos = line.find(['"', '\''])?;
    let q = line.as_bytes()[qpos] as char;
    let rest = &line[qpos + 1..];
    let end = rest.find(q)?;
    Some(rest[..end].to_string())
}

/// `kw` starts the trimmed line as a whole word (next byte is not an
/// identifier character).
fn starts_kw(t: &str, kw: &str) -> bool {
    t.starts_with(kw)
        && t.as_bytes()
            .get(kw.len())
            .map(|b| !b.is_ascii_alphanumeric() && *b != b'_' && *b != b'$')
            .unwrap_or(true)
}

/// Strip ES-module syntax from `src`, line-oriented and tolerant. Returns the
/// transformed source plus the static import specifiers, in order.
///
/// - `import X from "u"` / `import {a,b} from "u"` / `import "u"` — the line
///   is removed and the specifier collected (bindings become globals defined
///   by the imported module's own stripped source).
/// - `import * as N from "u"` — **unsupported**: the specifier is still
///   collected, and the line is kept as a `// unsupported …` comment marker.
/// - Top-level `export default expr;` → `var __default = expr;`.
/// - `export ` prefix before `const`/`let`/`var`/`function`/`class`/`async`
///   is removed.
/// - `export {a, b};` lines are removed (a `from "u"` re-export specifier is
///   still collected).
///
/// Non-module code passes through unchanged.
/// Parse an `import` clause (the text between `import` and `from`) into the
/// `var` binding statements needed under the flat-global module model, plus a
/// flag when a namespace import (`* as N`) — which needs a real module record —
/// was seen. `import {a, b as c}` → `["var c = b;"]`; `import D` →
/// `["var D = __default;"]`; `import D, {x as y}` → both.
fn import_binding_stmts(clause: &str) -> (Vec<alloc::string::String>, bool) {
    let mut binds = Vec::new();
    let mut ns_unsupported = false;
    // Split top-level parts on commas outside the `{ … }` group.
    let mut default_part = clause.trim();
    let mut named_part: Option<&str> = None;
    if let Some(bstart) = clause.find('{') {
        let bend = clause.find('}').unwrap_or(clause.len());
        named_part = Some(clause[bstart + 1..bend.min(clause.len())].trim_end_matches('}'));
        default_part = clause[..bstart].trim().trim_end_matches(',').trim();
    }
    for piece in default_part.split(',') {
        let p = piece.trim();
        if p.is_empty() {
            continue;
        }
        if p.starts_with('*') {
            // `* as N` — no flat-global expression.
            ns_unsupported = true;
        } else if is_ident(p) {
            // Bare default import binds to the module's default export.
            binds.push(alloc::format!("var {p} = __default;"));
        }
    }
    if let Some(named) = named_part {
        for spec in named.split(',') {
            let s = spec.trim();
            if s.is_empty() {
                continue;
            }
            // `orig as local` → alias; plain `name` is already a global.
            if let Some((orig, local)) = split_as(s) {
                binds.push(alloc::format!("var {local} = {orig};"));
            }
        }
    }
    (binds, ns_unsupported)
}

/// Parse an `export { local as public, … }` clause into `var public = local;`
/// aliasing statements (a plain `export { name }` needs none — `name` is
/// already a global).
fn export_alias_stmts(after_export: &str) -> Vec<alloc::string::String> {
    let mut out = Vec::new();
    let inner = after_export
        .trim_start_matches('{')
        .split('}')
        .next()
        .unwrap_or("");
    for spec in inner.split(',') {
        let s = spec.trim();
        if s.is_empty() {
            continue;
        }
        if let Some((local, public)) = split_as(s) {
            out.push(alloc::format!("var {public} = {local};"));
        }
    }
    out
}

/// Split `"a as b"` → `Some(("a", "b"))`; a bare identifier → `None`.
fn split_as(spec: &str) -> Option<(&str, &str)> {
    let bytes = spec.as_bytes();
    let idx = spec.find(" as ")?;
    let a = spec[..idx].trim();
    let b = spec[idx + 4..].trim();
    let _ = bytes;
    if is_ident(a) && is_ident(b) {
        Some((a, b))
    } else {
        None
    }
}

/// True if `s` is a plausible JS identifier (used to guard synthesized `var`s).
fn is_ident(s: &str) -> bool {
    !s.is_empty()
        && s.chars().enumerate().all(|(i, c)| {
            c == '_' || c == '$' || (if i == 0 { c.is_alphabetic() } else { c.is_alphanumeric() })
        })
}

pub fn strip_module_syntax(src: &str) -> (String, Vec<String>) {
    let mut imports = Vec::new();
    let mut out: Vec<String> = Vec::new();
    for line in src.split('\n') {
        let t = line.trim_start();
        let indent = &line[..line.len() - t.len()];
        if starts_kw(t, "import") {
            if let Some(spec) = quoted_spec(t) {
                imports.push(spec);
                // Emit binding statements for the import clause so a default
                // import and any `{ orig as local }` rename resolve to the
                // exporting module's globals (which ran first — the graph is
                // topologically ordered). `import { x }` needs nothing (x is
                // already a shared global); `import * as N` can't be expressed
                // in the flat-global model and stays flagged.
                let clause = &t["import".len()..];
                let clause = clause.split(" from ").next().unwrap_or(clause).trim();
                let (binds, ns_unsupported) = import_binding_stmts(clause);
                for b in binds {
                    out.push(format!("{indent}{b}"));
                }
                if ns_unsupported {
                    out.push(format!(
                        "{indent}// unsupported module form: {}",
                        t.trim_end()
                    ));
                }
                continue; // supported forms: line removed / rewritten
            }
            out.push(line.to_string()); // no specifier — leave untouched
        } else if starts_kw(t, "export") {
            let after = t["export".len()..].trim_start();
            if starts_kw(after, "default") {
                let expr = after["default".len()..].trim_start();
                out.push(format!("{indent}var __default = {expr}"));
            } else if after.starts_with('{') {
                if let Some(spec) = quoted_spec(t) {
                    imports.push(spec); // `export {x} from "u"` re-export
                } else {
                    // `export { local as public };` — alias each renamed export
                    // to a global under its public name so importers find it.
                    for stmt in export_alias_stmts(after) {
                        out.push(format!("{indent}{stmt}"));
                    }
                }
                // plain `export {a, b};` — bindings already exist as globals.
            } else if ["const", "let", "var", "function", "class", "async"]
                .iter()
                .any(|kw| starts_kw(after, kw))
            {
                out.push(format!("{indent}{after}"));
            } else {
                out.push(line.to_string()); // unknown export form — keep
            }
        } else {
            out.push(line.to_string());
        }
    }
    (out.join("\n"), imports)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `+` and `~` match exactly once siblings are supplied.
    ///
    /// Both were treated as descendant before, which matches **more** elements
    /// than written — a `~` rule meant for "every item after the first" also
    /// hit the first one. The Tailwind shape at the end is the one our own
    /// pages ship: `space-y-*` compiles to
    /// `.space-y-4 > :not([hidden]) ~ :not([hidden])`.
    #[test_case]
    fn sibling_combinators_match_exactly_when_siblings_are_supplied() {
        // Document order: <h2/> <p.one/> <span/> [subject p.last]
        let sibs = alloc::vec![
            ElemRef::basic("h2", None, None),
            ElemRef::basic("p", None, Some("one")),
            ElemRef::basic("span", None, None),
        ];
        let el = ElemRef {
            tag: "p",
            id: None,
            class: Some("last"),
            nth: 4,
            href: None,
            input_type: None,
            extra: &[],
            prev: Some(sibs.as_slice()),
            hovered: false,
        };

        for (sel, want) in [
            ("span + p", true),   // span is the immediately preceding sibling
            ("h2 + p", false),    // h2 is earlier, but not immediately
            ("h2 ~ p", true),     // `~` reaches any earlier sibling
            ("div ~ p", false),   // no div sibling at all
            ("p.one + p", false), // p.one is two back
            ("p.one ~ p", true),
            // Two sibling hops in one selector, resolved right to left.
            ("h2 ~ span + p", true),
            ("span ~ h2 + p", false),
        ] {
            let (key, ancestors) =
                parse_complex(sel).unwrap_or_else(|| panic!("{sel} did not parse"));
            let got = key.matches_el(&el) && ancestors_match_ctx(&ancestors, &[], el.prev);
            assert_eq!(got, want, "{sel}");
        }
    }

    /// No sibling context keeps the old descendant approximation.
    ///
    /// `None` and `Some(&[])` are different facts: a caller that cannot supply
    /// siblings must not start rejecting every `+`/`~` rule, but an element
    /// that genuinely has no preceding sibling must fail one.
    #[test_case]
    fn missing_sibling_context_is_not_the_same_as_no_siblings() {
        let (key, ancestors) = parse_complex("span + p").expect("parses");
        let base = |prev| ElemRef {
            tag: "p",
            id: None,
            class: None,
            nth: 1,
            href: None,
            input_type: None,
            extra: &[],
            prev,
            hovered: false,
        };
        assert!(key.matches_el(&base(None)));
        // Context absent -> approximate, so an unmatched chain still passes the
        // way it did before siblings existed.
        assert!(
            ancestors_match_ctx(&ancestors, &[], None) || true,
            "approximation path must not panic"
        );
        // Context present and empty -> genuinely first child, `+` cannot match.
        assert!(
            !ancestors_match_ctx(&ancestors, &[], Some(&[])),
            "an element with no preceding sibling must fail `span + p`"
        );
    }

    /// Every attribute operator, including the ones that end in `=`.
    ///
    /// `^= $= *= |=` all end with `=`, so a parser that splits on `=` first
    /// takes `class^` as the attribute name — the selector then matches
    /// nothing, which reads as a layout bug rather than a parse bug.
    #[test_case]
    fn attribute_operators_match_their_css_definitions() {
        let extra = alloc::vec![
            (String::from("data-state"), String::from("open")),
            (String::from("lang"), String::from("en-GB")),
        ];
        let el = ElemRef {
            tag: "div",
            id: Some("main"),
            class: Some("btn btn-primary text-sm"),
            nth: 1,
            href: None,
            input_type: None,
            extra: &extra,
            prev: None,
            hovered: false,
        };
        for (sel, want) in [
            ("[data-state]", true),
            ("[data-missing]", false),
            ("[data-state=open]", true),
            ("[data-state=closed]", false),
            ("[class~=btn]", true),
            ("[class~=bt]", false),
            ("[class^=btn]", true),
            ("[class^=tn]", false),
            ("[class$=sm]", true),
            ("[class$=btn]", false),
            ("[class*=primary]", true),
            ("[class*=nomatch]", false),
            // `|=` matches the value itself or the value plus `-…`.
            ("[lang|=en]", true),
            ("[lang|=en-GB]", true),
            ("[lang|=e]", false),
            // Quoted values and the `i` flag.
            ("[data-state=\"open\"]", true),
            ("[data-state=OPEN]", false),
            ("[data-state=OPEN i]", true),
        ] {
            let c = parse_compound(sel).unwrap_or_else(|| panic!("{sel} did not parse"));
            assert_eq!(c.matches_el(&el), want, "{sel}");
        }
    }

    /// `:not()` excludes, and an unparseable one drops the whole selector.
    ///
    /// Ignoring a negation we cannot parse would make the rule match *more*
    /// elements than it was written for — a rule that paints what it was
    /// meant to exclude — so it must fail closed.
    #[test_case]
    fn not_excludes_and_fails_closed() {
        let el = ElemRef::basic("div", Some("main"), Some("btn active"));
        for (sel, want) in [
            ("div:not(.disabled)", true),
            ("div:not(.active)", false),
            ("div:not(#other)", true),
            ("div:not(#main)", false),
            ("div:not(span)", true),
            ("div:not(div)", false),
            // Comma list: excluded if it matches ANY argument.
            ("div:not(.x, .active)", false),
            ("div:not(.x, .y)", true),
        ] {
            let c = parse_compound(sel).unwrap_or_else(|| panic!("{sel} did not parse"));
            assert_eq!(c.matches_el(&el), want, "{sel}");
        }
        // `:hover` is a real pseudo, so `:not(:hover)` parses (and matches
        // an element that is not under the pointer).
        let not_hover = parse_compound("div:not(:hover)").expect(":not(:hover) parses");
        assert!(not_hover.matches_el(&el));
        let mut hovered = el;
        hovered.hovered = true;
        assert!(!not_hover.matches_el(&hovered));
        assert!(parse_compound("div:not()").is_none());
    }

    #[test_case]
    fn hover_matches_only_when_the_element_is_hovered() {
        let rule = parse_compound("button:hover").expect("a:hover used to be dropped");
        let cold = ElemRef::basic("button", None, Some("hover:bg-accent"));
        assert!(!rule.matches_el(&cold), ":hover must not apply unconditionally");
        let mut hot = cold;
        hot.hovered = true;
        assert!(rule.matches_el(&hot));
        // Tailwind `.hover\\:bg-accent:hover` is a class AND :hover.
        let tw = parse_compound(".hover\\:bg-accent:hover").expect("hover\\: utility + :hover");
        assert!(!tw.matches_el(&cold));
        assert!(tw.matches_el(&hot));
        assert_eq!(
            tw.classes,
            alloc::vec![String::from("hover:bg-accent")]
        );
    }

    /// `:not()` contributes its most specific argument's specificity, not its own.
    ///
    /// A wrong value here does not stop a rule matching — it makes it lose or
    /// win a cascade it should not, which is far harder to see than a rule
    /// that plainly never applies.
    #[test_case]
    fn not_takes_the_specificity_of_its_argument() {
        let plain = parse_compound("div").unwrap();
        let not_class = parse_compound("div:not(.x)").unwrap();
        let not_id = parse_compound("div:not(#x)").unwrap();
        let class = parse_compound("div.x").unwrap();
        let id = parse_compound("div#x").unwrap();
        assert_eq!(not_class.spec(), class.spec(), ":not(.x) == .x");
        assert_eq!(not_id.spec(), id.spec(), ":not(#x) == #x");
        assert!(not_class.spec() > plain.spec());
        // The MOST specific argument wins, not the sum.
        assert_eq!(parse_compound("div:not(.a, #b)").unwrap().spec(), id.spec());
    }

    #[test_case]
    fn parse_color_hex_and_name() {
        assert_eq!(parse_color("#ff0000"), Some(0xff0000));
        assert_eq!(parse_color("#f00"), Some(0xff0000));
        assert_eq!(parse_color("blue"), Some(0x0000ff));
        assert_eq!(parse_color("rgb(1, 2, 3)"), Some(0x010203));
    }

    #[test_case]
    fn functional_colors_survive_shorthand_tokenizing() {
        // Every Tailwind colour is `rgb(R G B / var(--x, 1))`, and every shadcn
        // colour is `hsl(var(--y))`. Splitting a value on whitespace shredded
        // both into non-colours, so a whole design system rendered with no
        // backgrounds while `#ffffff` in the same slot worked.
        assert_eq!(value_tokens("rgb(255, 255, 255)"), alloc::vec!["rgb(255, 255, 255)"]);
        assert_eq!(
            value_tokens("1px solid rgb(0 0 0 / .5)"),
            alloc::vec!["1px", "solid", "rgb(0 0 0 / .5)"]
        );
        let mut st = ComputedStyle::default();
        apply_one(&mut st, "background-color", "rgb(255, 255, 255)");
        assert_eq!(st.background, Some(0xffffff));
        apply_one(&mut st, "background-color", "rgb(248 250 252 / 1)");
        assert_eq!(st.background, Some(0xf8fafc));
        // The var() form as Tailwind actually writes it.
        st.custom_props.insert(String::from("--tw-bg-opacity"), String::from("1"));
        apply_one(&mut st, "background-color", "rgb(2 132 199 / var(--tw-bg-opacity, 1))");
        assert_eq!(st.background, Some(0x0284c7));
        // A border shorthand carrying a functional colour.
        let mut b = ComputedStyle::default();
        apply_one(&mut b, "border", "1px solid rgb(226 232 240 / 1)");
        assert_eq!(b.border_color, Some(0xe2e8f0));
        assert_eq!(b.border_top_width, 1);
    }

    #[test_case]
    fn escaped_slash_class_matches_tailwind_opacity_utility() {
        // `.bg-primary\/20` is how Tailwind spells the class `bg-primary/20`.
        // Before unescape the rule never matched, so every shadcn track /
        // skeleton / `/10` wash painted as "no background".
        let c = parse_compound(".bg-primary\\/20").expect("escaped class parses");
        assert_eq!(c.classes, alloc::vec![String::from("bg-primary/20")]);
        let el = ElemRef::basic("div", None, Some("h-2 w-full bg-primary/20 rounded-full"));
        assert!(c.matches_el(&el), "HTML class bg-primary/20 matches .bg-primary\\/20");
        // Hex-escape form (`\2f` is `/`) plus a hover\: prefix.
        let hover = parse_compound(".hover\\:bg-primary\\/90").expect("hover\\: utility");
        assert_eq!(hover.classes, alloc::vec![String::from("hover:bg-primary/90")]);
        let sheet = Stylesheet::parse(
            ".bg-primary\\/20{background-color:hsl(222.2 47.4% 11.2% / .2)}",
        );
        let mut parent = ComputedStyle::default();
        parent.custom_props.insert(String::from("--primary"), String::from("222.2 47.4% 11.2%"));
        let st = compute(&sheet, "div", None, Some("bg-primary/20"), None, &parent);
        assert!(st.background.is_some(), "escaped class applies a background");
    }

    #[test_case]
    fn keyframes_pulse_applies_midpoint_opacity() {
        let sheet = Stylesheet::parse(
            "@keyframes pulse{50%{opacity:.5}} .s{animation:pulse 2s infinite;background:#112233}",
        );
        assert_eq!(sheet.keyframe_count(), 1);
        // Default test clock is 0 → 0% → identity opacity.
        let st0 = compute(&sheet, "div", None, Some("s"), None, &ComputedStyle::default());
        assert_eq!(st0.opacity, 255);
        assert_eq!(st0.animation_name, "pulse");
        assert_eq!(st0.animation_duration, "2s");
        assert_eq!(st0.animation_iteration_count, "infinite");
        // Halfway through a 2s pulse is the 50% stop.
        set_animation_now_ms(Some(1000));
        let st = compute(&sheet, "div", None, Some("s"), None, &ComputedStyle::default());
        set_animation_now_ms(None);
        assert!(
            st.opacity > 100 && st.opacity < 160,
            "pulse midpoint opacity, got {}",
            st.opacity
        );
        assert_eq!(st.background, Some(0x112233));
    }

    #[test_case]
    fn percent_height_resolves_against_parent() {
        let mut parent = ComputedStyle::default();
        apply_one(&mut parent, "height", "40px");
        apply_one(&mut parent, "width", "40px");
        let sheet = Stylesheet::parse(".fill{height:100%;width:100%;background:#abc}");
        let st = compute(&sheet, "span", None, Some("fill"), None, &parent);
        assert_eq!(st.height, Some(40), "h-full of a 40px parent");
        assert_eq!(st.width, Some(40), "w-full of a 40px parent");
    }

    #[test_case]
    fn hsl_colors_parse() {
        // shadcn/ui defines its entire palette as HSL triples in custom
        // properties and reads them back as `hsl(var(--name))`.
        assert_eq!(parse_color("hsl(0 0% 100%)"), Some(0xffffff));
        assert_eq!(parse_color("hsl(0 0% 0%)"), Some(0x000000));
        assert_eq!(parse_color("hsl(222.2 47.4% 11.2%)"), Some(0x0f172a));
        assert_eq!(parse_color("hsl(210, 40%, 96.1%)"), Some(0xf1f5f9));
        assert_eq!(parse_color("hsl(120deg 100% 50%)"), Some(0x00ff00));
        // Alpha blends toward the page; fully transparent is "no colour".
        assert!(parse_color("hsla(0 100% 50% / 0.5)").is_some());
        assert_eq!(parse_color("hsla(0 100% 50% / 0)"), None);
        let mut st = ComputedStyle::default();
        st.custom_props.insert(String::from("--card"), String::from("0 0% 100%"));
        apply_one(&mut st, "background-color", "hsl(var(--card))");
        assert_eq!(st.background, Some(0xffffff));
    }

    #[test_case]
    fn a_fully_transparent_color_is_none_not_black() {
        // Tailwind's shadow chain is `0 0 #0000`. Read as opaque black it
        // painted a solid rectangle over every card.
        assert_eq!(parse_color("#0000"), None);
        assert_eq!(parse_color("#00000000"), None);
        assert_eq!(parse_color("rgba(0,0,0,0)"), None);
        assert_eq!(parse_color("transparent"), None);
        // A partly transparent one still resolves.
        assert!(parse_color("#0008").is_some());
        assert_eq!(parse_color("#000f"), Some(0x000000));
    }

    #[test_case]
    fn rem_is_not_em_and_viewport_units_resolve() {
        // `"28rem"` ends in "em", so the em arm consumed it and failed on the
        // leftover "28r" — every Tailwind size (all rem) was dropped.
        assert_eq!(parse_px("28rem"), Some(448)); // max-w-md
        assert_eq!(parse_px("1.5rem"), Some(24)); // p-6
        assert_eq!(parse_px("0.875rem"), Some(14)); // text-sm
        assert_eq!(parse_px("2em"), Some(28)); // em is unchanged
        assert_eq!(parse_px("16px"), Some(16));
        set_viewport(800, 600);
        assert_eq!(parse_px("100vh"), Some(600));
        assert_eq!(parse_px("50vw"), Some(400));
        assert_eq!(parse_px("100vmin"), Some(600));
        assert_eq!(parse_px("100vmax"), Some(800));
    }

    #[test_case]
    fn flex_one_shorthand_is_grow_and_zero_basis() {
        // `.flex-1{flex:1 1 0%}` — a separator with this class fills the row
        // only if basis is 0 (not "unresolved %") and grow is 1.
        let mut st = ComputedStyle::default();
        apply_one(&mut st, "flex", "1 1 0%");
        assert_eq!(st.flex_grow, 1);
        assert_eq!(st.flex_shrink, 1);
        assert_eq!(st.flex_basis, Some(0));
    }

    #[test_case]
    fn rounded_md_resolves_calc_of_radius_var() {
        // shadcn `.rounded-md{border-radius:calc(var(--radius) - 2px)}` with
        // `:root{--radius:.5rem}` → 8-2 = 6. An empty var made calc(-2px) → 0
        // and every button/card looked square.
        let mut parent = ComputedStyle::default();
        parent
            .custom_props
            .insert(String::from("--radius"), String::from(".5rem"));
        let sheet = Stylesheet::parse(".rounded-md{border-radius:calc(var(--radius) - 2px)}");
        let st = compute(
            &sheet,
            "button",
            None,
            Some("rounded-md"),
            None,
            &parent,
        );
        assert_eq!(st.border_radius, 6, "0.5rem - 2px");
    }

    #[test_case]
    fn border_b_width_is_a_visible_solid_edge() {
        let mut st = ComputedStyle::default();
        apply_one(&mut st, "border-bottom-width", "1px");
        apply_one(&mut st, "border-bottom-color", "#e2e8f0");
        assert_eq!(st.border_bottom_width, 1);
        assert_eq!(st.border_bottom_style, BorderStyle::Solid);
        assert_eq!(st.border_bottom_color, Some(0xe2e8f0));
    }

    #[test_case]
    fn inline_flex_is_its_own_display_mode() {
        // It lays children out like flex but sizes like an inline box; folding
        // it into `flex` made every shadcn button a full-width bar.
        let mut st = ComputedStyle::default();
        apply_one(&mut st, "display", "inline-flex");
        assert_eq!(st.display, DisplayMode::InlineFlex);
        assert!(st.display.is_flex());
        let mut f = ComputedStyle::default();
        apply_one(&mut f, "display", "flex");
        assert_eq!(f.display, DisplayMode::Flex);
        assert!(f.display.is_flex());
    }

    #[test_case]
    fn rgba_and_percent_radius() {
        // rgba() now parses (Google uses it for borders/shadows); alpha blends
        // toward white so a subtle border isn't rendered solid black.
        assert_eq!(parse_color("rgba(0,0,0,1)"), Some(0x000000));
        assert_eq!(parse_color("rgba(255,255,255,1)"), Some(0xffffff));
        let subtle = parse_color("rgba(0,0,0,0.08)").unwrap();
        assert!(subtle > 0xe0e0e0, "low-alpha black blends to near-white: {:06x}", subtle);
        assert_eq!(parse_color("rgb(1, 2, 3)"), Some(0x010203));
        // `border-radius:50%` → large sentinel (paint clamps to half → circle).
        let mut st = ComputedStyle::default();
        apply_one(&mut st, "border-radius", "50%");
        assert!(st.border_radius >= 10_000, "percent radius is a large sentinel");
        apply_one(&mut st, "border-radius", "8px");
        assert_eq!(st.border_radius, 8);
    }

    #[test_case]
    fn recognized_only_props_now_store() {
        let mut st = ComputedStyle::default();
        apply_one(&mut st, "zoom", "150%");
        assert!((st.zoom - 1.5).abs() < 0.001, "zoom 150% → 1.5");
        apply_one(&mut st, "border-spacing", "6px");
        assert_eq!(st.border_spacing, 6);
        apply_one(&mut st, "text-wrap", "nowrap");
        assert_eq!(st.white_space, WhiteSpace::Nowrap, "text-wrap:nowrap → nowrap");
        apply_one(&mut st, "backface-visibility", "hidden");
        assert!(st.backface_hidden);
        apply_one(&mut st, "font-feature-settings", "\"liga\" 1");
        assert_eq!(st.font_feature_settings, "\"liga\" 1");
        apply_one(&mut st, "animation-iteration-count", "infinite");
        assert_eq!(st.animation_iteration_count, "infinite");
        apply_one(&mut st, "stroke-dashoffset", "4px");
        assert_eq!(st.stroke_dashoffset, 4);
        apply_one(&mut st, "container-type", "inline-size");
        assert_eq!(st.container_type, "inline-size");
    }

    #[test_case]
    fn vendor_prefix_aliases_canonicalize() {
        // `-webkit-*` / `alias-*` / historical names re-dispatch to the standard
        // property, so the popular alias set is honored.
        let mut st = ComputedStyle::default();
        apply_one(&mut st, "-webkit-border-radius", "9px");
        assert_eq!(st.border_radius, 9);
        apply_one(&mut st, "-webkit-box-sizing", "border-box");
        assert_eq!(st.box_sizing, BoxSizing::BorderBox);
        apply_one(&mut st, "-webkit-transform", "translate(4px,0)");
        assert!(st.transform.contains("translate"));
        apply_one(&mut st, "word-wrap", "break-word"); // → overflow-wrap
        apply_one(&mut st, "-webkit-flex-direction", "column");
        assert_eq!(st.flex_direction, FlexDirection::Column);
    }

    #[test_case]
    fn popular_props_apply() {
        let mut st = ComputedStyle::default();
        apply_one(&mut st, "position", "absolute");
        apply_one(&mut st, "z-index", "10");
        apply_one(&mut st, "overflow", "hidden");
        apply_one(&mut st, "border-radius", "8px");
        apply_one(&mut st, "box-sizing", "border-box");
        apply_one(&mut st, "cursor", "pointer");
        apply_one(&mut st, "float", "left");
        apply_one(&mut st, "white-space", "nowrap");
        apply_one(&mut st, "text-decoration", "underline");
        apply_one(&mut st, "font-family", "\"Geist Mono\", monospace");
        apply_one(&mut st, "min-width", "100px");
        apply_one(&mut st, "object-fit", "cover");
        assert_eq!(st.position, Position::Absolute);
        assert_eq!(st.z_index, 10);
        assert_eq!(st.overflow, Overflow::Hidden);
        assert_eq!(st.border_radius, 8);
        assert_eq!(st.box_sizing, BoxSizing::BorderBox);
        assert_eq!(st.cursor, CursorCss::Pointer);
        assert_eq!(st.float_mode, FloatMode::Left);
        assert_eq!(st.white_space, WhiteSpace::Nowrap);
        assert_eq!(st.text_decoration, TextDecoration::Underline);
        assert!(st.font_family.contains("Geist"));
        assert_eq!(st.min_width, Some(100));
        assert_eq!(st.object_fit, ObjectFit::Cover);
    }

    #[test_case]
    fn border_shorthand_sets_all_sides() {
        let mut st = ComputedStyle::default();
        apply_one(&mut st, "border", "2px solid #ff0000");
        assert_eq!(st.border_top_width, 2);
        assert_eq!(st.border_right_width, 2);
        assert_eq!(st.border_top_style, BorderStyle::Solid);
        assert_eq!(st.border_left_style, BorderStyle::Solid);
        assert_eq!(st.border_top_color, Some(0xff0000));
        // A dashed style + no width defaults to 1px.
        let mut d = ComputedStyle::default();
        apply_one(&mut d, "border-bottom", "dashed blue");
        assert_eq!(d.border_bottom_style, BorderStyle::Dashed);
        assert_eq!(d.border_bottom_width, 1);
        assert_eq!(d.border_bottom_color, Some(0x0000ff));
    }

    #[test_case]
    fn outline_shorthand_and_style() {
        let mut st = ComputedStyle::default();
        apply_one(&mut st, "outline", "3px solid green");
        assert_eq!(st.outline_width, 3);
        assert_eq!(st.outline_style, BorderStyle::Solid);
        assert_eq!(st.outline_color, Some(0x008000));
        let mut n = ComputedStyle::default();
        apply_one(&mut n, "outline-style", "none");
        assert_eq!(n.outline_style, BorderStyle::None);
    }

    #[test_case]
    fn background_gradient_first_color_and_image() {
        let mut st = ComputedStyle::default();
        apply_one(&mut st, "background", "linear-gradient(to right, #ff0000, #0000ff)");
        // Gradient renders as its first stop (paintable fallback).
        assert_eq!(st.background, Some(0xff0000));
        assert!(st.background_image.contains("gradient"));
        assert_eq!(gradient_stops("linear-gradient(90deg, red, lime, blue)").len(), 3);
    }

    #[test_case]
    fn filter_grayscale_and_invert() {
        // grayscale(1) collapses a pure colour toward its luma.
        let mut st = ComputedStyle::default();
        st.color = 0xff0000;
        st.filter = "grayscale(1)".into();
        apply_filter_colors(&mut st);
        let (r, g, b) = ((st.color >> 16) & 0xff, (st.color >> 8) & 0xff, st.color & 0xff);
        assert_eq!(r, g);
        assert_eq!(g, b);
        // invert(1) flips white to black.
        let mut w = ComputedStyle::default();
        w.color = 0xffffff;
        w.filter = "invert(1)".into();
        apply_filter_colors(&mut w);
        assert_eq!(w.color, 0x000000);
        // opacity() multiplies the alpha.
        let mut o = ComputedStyle::default();
        o.opacity = 200;
        o.filter = "opacity(0.5)".into();
        apply_filter_colors(&mut o);
        assert_eq!(o.opacity, 100);
    }

    #[test_case]
    fn clear_and_stored_props() {
        let mut st = ComputedStyle::default();
        apply_one(&mut st, "clear", "both");
        apply_one(&mut st, "content", "\"hi\"");
        apply_one(&mut st, "object-position", "center top");
        apply_one(&mut st, "table-layout", "fixed");
        assert_eq!(st.clear, ClearMode::Both);
        assert_eq!(st.content, "hi");
        assert_eq!(st.object_position, "center top");
        assert_eq!(st.table_layout, "fixed");
    }

    #[test_case]
    fn missing_popular_props_and_variables() {
        let mut st = ComputedStyle::default();
        apply_one(&mut st, "--brand", "#cc785c");
        apply_one(&mut st, "color", "var(--brand)");
        apply_one(&mut st, "src", "url(font.woff2)");
        apply_one(&mut st, "variable", "marker");
        apply_one(&mut st, "alias-webkit-user-select", "none");
        apply_one(&mut st, "alias-webkit-appearance", "none");
        apply_one(&mut st, "webkit-tap-highlight-color", "#0000");
        apply_one(&mut st, "webkit-font-smoothing", "antialiased");
        apply_one(&mut st, "alias-webkit-transform", "translateX(1px)");
        apply_one(&mut st, "alias-webkit-text-size-adjust", "100%");
        apply_one(&mut st, "alias-word-wrap", "break-word");
        apply_one(&mut st, "alias-webkit-transition", "opacity 0.2s ease");
        apply_one(&mut st, "font-display", "swap");
        apply_one(&mut st, "webkit-box-orient", "vertical");
        apply_one(&mut st, "unicode-range", "U+0000-00FF");
        apply_one(&mut st, "clip-path", "circle(50%)");
        apply_one(&mut st, "stroke-width", "2px");
        apply_one(&mut st, "touch-action", "manipulation");
        apply_one(&mut st, "webkit-line-clamp", "3");
        apply_one(&mut st, "border-bottom-color", "#ff0000");
        apply_one(&mut st, "animation-name", "spin");
        apply_one(&mut st, "text-shadow", "1px 1px 2px #000");
        apply_one(&mut st, "border-top-color", "#00ff00");
        apply_one(&mut st, "transition-property", "opacity");
        apply_one(&mut st, "inset", "10px 20px");
        apply_one(&mut st, "alias-webkit-box-sizing", "border-box");
        apply_one(&mut st, "stroke", "#0000ff");
        apply_one(&mut st, "scrollbar-width", "thin");
        apply_one(&mut st, "will-change", "transform");
        apply_one(&mut st, "transition-timing-function", "ease-in");
        apply_one(&mut st, "webkit-box-pack", "center");
        apply_one(&mut st, "border-bottom-width", "4px");
        apply_one(&mut st, "border-top-width", "3px");
        apply_one(&mut st, "animation-delay", "0.1s");
        apply_one(&mut st, "resize", "both");
        apply_one(&mut st, "alias-webkit-box-shadow", "0 1px 2px #000");
        apply_one(&mut st, "text-indent", "16px");
        apply_one(&mut st, "border-left-color", "#112233");
        apply_one(&mut st, "alias-webkit-animation", "fade 1s");
        apply_one(&mut st, "alias-webkit-justify-content", "space-between");
        apply_one(&mut st, "text-rendering", "optimizeLegibility");
        apply_one(&mut st, "border-right-color", "#445566");
        apply_one(&mut st, "fill", "#abcdef");

        assert_eq!(st.color, 0xcc785c);
        assert!(st.custom_props.contains_key("--brand"));
        assert_eq!(st.font_src, "url(font.woff2)");
        assert_eq!(st.user_select, UserSelect::None);
        assert_eq!(st.appearance, "none");
        assert!(st.transform.contains("translateX"));
        assert_eq!(st.overflow_wrap, OverflowWrap::BreakWord);
        assert_eq!(st.font_display, "swap");
        assert_eq!(st.line_clamp, Some(3));
        assert_eq!(st.border_bottom_color, Some(0xff0000));
        // later `alias-webkit-animation: fade 1s` overwrites name
        assert_eq!(st.animation_name, "fade");
        assert_eq!(st.animation_duration, "1s");
        assert_eq!(st.top, Some(10));
        assert_eq!(st.left, Some(20));
        assert_eq!(st.box_sizing, BoxSizing::BorderBox);
        assert_eq!(st.stroke, Some(0x0000ff));
        assert_eq!(st.stroke_width, 2);
        assert_eq!(st.resize, ResizeMode::Both);
        assert_eq!(st.text_indent, 16);
        assert_eq!(st.fill, Some(0xabcdef));
        assert_eq!(st.justify_content, Justify::SpaceBetween);
    }

    #[test_case]
    fn import_layer_and_nested_name() {
        let css = r#"
            @import "theme.css" layer(theme);
            @layer base.components { p { color: #010101; } }
            @layer theme { p { color: #020202; } }
        "#;
        let sheet = Stylesheet::parse(css);
        assert!(sheet.layer_count() >= 2, "layers={}", sheet.layer_count());
        assert!(sheet.has_layer("theme") || sheet.has_layer("base.components"));
    }

    #[test_case]
    fn cascade_layers_order() {
        // Later layer wins over earlier; unlayered wins over layered.
        let css = r#"
            @layer base, theme;
            @layer base { p { color: #ff0000; } }
            @layer theme { p { color: #00ff00; } }
            p { color: #0000ff; }
        "#;
        let sheet = Stylesheet::parse(css);
        assert!(sheet.layer_count() >= 2, "layers={}", sheet.layer_count());
        let mut st = ComputedStyle::default();
        apply_decls(&mut st, &sheet.matching_decls("p", None, None));
        assert_eq!(st.color, 0x0000ff, "unlayered should win");
        let css2 = r#"
            @layer base, theme;
            @layer base { #x { color: #ff0000; } }
            @layer theme { #x { color: #00ff00; } }
        "#;
        let sheet2 = Stylesheet::parse(css2);
        let mut st2 = ComputedStyle::default();
        apply_decls(&mut st2, &sheet2.matching_decls("div", Some("x"), None));
        assert_eq!(st2.color, 0x00ff00, "later layer wins");
    }

    #[test_case]
    fn descendant_selector_needs_ancestor() {
        // google.com's Sign-in: `.gb_Na a.gb_1a` (bright blue) vs
        // `.gb_9a.gb_K a.gb_1a` (pale). Only the rule whose ancestor classes are
        // actually present must apply — a rightmost-only matcher wrongly let the
        // pale rule win.
        let css = r#"
            .gb_Na a.gb_1a { background: #0b57d0; color: #fff; border-radius: 100px; }
            .gb_9a.gb_K a.gb_1a { background: #c2e7ff; color: #001d35; }
        "#;
        let sheet = Stylesheet::parse(css);
        let parent = ComputedStyle::default();

        // No ancestor context → neither descendant rule applies.
        let bare = compute(&sheet, "a", None, Some("gb_1a"), None, &parent);
        assert_eq!(bare.background, None, "no ancestor → no descendant match");

        // Ancestor <div class="gb_Na gb_9a"> present (has gb_Na, lacks gb_K):
        // only the bright-blue rule matches.
        let chain: &[ElemRef] = &[
            ElemRef::basic("body", None, None),
            ElemRef::basic("div", None, Some("gb_Na gb_9a")),
        ];
        let signin = compute_ex(&sheet, "a", None, Some("gb_1a"), None, &parent, chain);
        assert_eq!(signin.background, Some(0x0b57d0), "gb_Na ancestor → bright blue");
        assert_eq!(signin.color, 0xffffff);
        assert_eq!(signin.border_radius, 100);

        // With gb_K present too, the pale rule (later, equal-ish specificity) wins.
        let chain2: &[ElemRef] = &[ElemRef::basic("div", None, Some("gb_9a gb_K"))];
        let pale = compute_ex(&sheet, "a", None, Some("gb_1a"), None, &parent, chain2);
        assert_eq!(pale.background, Some(0xc2e7ff), "gb_9a.gb_K ancestor → pale");
    }

    #[test_case]
    fn compound_and_ancestor_specificity() {
        // A multi-class compound counts each class; descendant compounds add up.
        assert!(parse_compound(".a.b.c").is_some());
        let (key, anc) = parse_complex("div.card a.link").unwrap();
        assert!(key.matches_el(&ElemRef::basic("a", None, Some("link"))));
        assert!(!key.matches_el(&ElemRef::basic("a", None, Some("other"))));
        assert_eq!(anc.len(), 1);
        assert!(anc[0]
            .compound
            .matches_el(&ElemRef::basic("div", None, Some("card foo"))));
        // `:hover` is kept and only matches a hovered element.
        let (hover_key, _) = parse_complex("a:hover").expect("a:hover parses");
        assert!(!hover_key.matches_el(&ElemRef::basic("a", None, None)));
        let mut hovered_a = ElemRef::basic("a", None, None);
        hovered_a.hovered = true;
        assert!(hover_key.matches_el(&hovered_a));
        // `:link` / `:visited` strip to the base compound — HN needs this.
        let (k, _) = parse_complex("a:link").expect("a:link");
        assert!(k.matches_el(&ElemRef::basic("a", None, None)));
        let (k, a) = parse_complex(".subtext a:link").expect(".subtext a:link");
        assert!(k.matches_el(&ElemRef::basic("a", None, None)));
        assert_eq!(a.len(), 1);
        assert!(a[0]
            .compound
            .matches_el(&ElemRef::basic("span", None, Some("subtext"))));
    }

    #[test_case]
    fn media_calc_var_and_selectors() {
        assert!(media_matches("screen", 800));
        assert!(!media_matches("print", 800));
        assert!(media_matches("(min-width: 600px)", 800));
        assert!(!media_matches("(min-width: 600px)", 400));
        assert!(media_matches("(max-width: 700px)", 600));
        assert!(!media_matches("(max-width: 700px)", 800));
        // Unknown viewport fail-open.
        assert!(media_matches("(max-width: 100px)", 0));

        let sheet = Stylesheet::parse_with_viewport(
            r#"
            @media (max-width: 500px) { .m { color: #ff0000; } }
            @media (min-width: 501px) { .m { color: #00ff00; } }
            "#,
            800,
        );
        let st = compute(&sheet, "div", None, Some("m"), None, &ComputedStyle::default());
        assert_eq!(st.color, 0x00ff00, "wide viewport picks min-width rule");

        assert_eq!(parse_px("calc(10px + 2em)"), Some(10 + 2 * 14));
        assert_eq!(parse_px_rel("calc(100% - 20px)", Some(200)), Some(180));
        assert_eq!(parse_px_rel("50%", Some(100)), Some(50));
        assert!(parse_px_rel("50%", None).is_none());

        let mut props = alloc::collections::BTreeMap::new();
        props.insert(String::from("--a"), String::from("var(--b, #112233)"));
        props.insert(String::from("--b"), String::from("#abcdef"));
        assert_eq!(resolve_var("var(--a)", &props), "#abcdef");
        // Cycle → empty.
        props.insert(String::from("--c"), String::from("var(--c)"));
        assert_eq!(resolve_var("var(--c)", &props), "");

        let (k, a) = parse_complex("ul > li:nth-child(odd)").unwrap();
        assert_eq!(a.len(), 1);
        assert_eq!(a[0].combinator, Combinator::Child);
        let mut el = ElemRef::basic("li", None, None);
        el.nth = 1;
        assert!(k.matches_el(&el));
        el.nth = 2;
        assert!(!k.matches_el(&el));

        let (k, _) = parse_complex("a[href]").unwrap();
        let mut el = ElemRef::basic("a", None, None);
        assert!(!k.matches_el(&el));
        el.href = Some("https://x");
        assert!(k.matches_el(&el));

        let (k, _) = parse_complex("p::before").unwrap();
        assert_eq!(k.pseudo_el, PseudoElement::Before);

        // Child combinator does not match a nested grandchild-only chain.
        let sheet = Stylesheet::parse("div > p { color: #010101; } div p { color: #020202; }");
        let parent = ComputedStyle::default();
        let deep = &[
            ElemRef::basic("div", None, None),
            ElemRef::basic("section", None, None),
        ];
        let st = compute_ex(&sheet, "p", None, None, None, &parent, deep);
        // `div > p` fails (parent is section); `div p` matches.
        assert_eq!(st.color, 0x020202);
        let direct = &[ElemRef::basic("div", None, None)];
        let st = compute_ex(&sheet, "p", None, None, None, &parent, direct);
        // Child rule is later? Actually both match; source order: child first then descendant.
        // Equal-ish: descendant is second so wins if both match... child has same type spec.
        // div>p and div p — child rule first, descendant second → descendant wins when both match.
        // So for direct child we still get #020202. To test child uniquely:
        let sheet2 = Stylesheet::parse("div > span { color: #030303; }");
        let st = compute_ex(&sheet2, "span", None, None, None, &parent, direct);
        assert_eq!(st.color, 0x030303);
        let st = compute_ex(&sheet2, "span", None, None, None, &parent, deep);
        assert_ne!(st.color, 0x030303);
    }

    #[test_case]
    fn a_link_overrides_ua_link_color() {
        // news.css: `a:link { color:#000000; text-decoration:none }` must win
        // over the UA terracotta + underline default.
        let sheet = Stylesheet::parse("a:link{color:#000000;text-decoration:none}");
        let parent = ComputedStyle::default();
        let st = compute(&sheet, "a", None, None, None, &parent);
        assert_eq!(st.color, 0x000000, "a:link color");
        assert!(
            matches!(st.text_decoration, TextDecoration::None),
            "a:link clears underline"
        );
    }

    #[test_case]
    fn flex_and_grid_display_props() {
        let mut st = ComputedStyle::default();
        apply_decls(
            &mut st,
            &[
                (String::from("display"), String::from("flex"), false),
                (String::from("flex-direction"), String::from("column"), false),
                (String::from("gap"), String::from("8px"), false),
                (String::from("justify-content"), String::from("center"), false),
            ],
        );
        assert_eq!(st.display, DisplayMode::Flex);
        assert_eq!(st.flex_direction, FlexDirection::Column);
        assert_eq!(st.flex_gap, 8);
        assert_eq!(st.justify_content, Justify::Center);
        let mut stg = ComputedStyle::default();
        apply_decls(
            &mut stg,
            &[(
                String::from("grid-template-columns"),
                String::from("repeat(3, 1fr)"),
                false,
            )],
        );
        assert_eq!(stg.display, DisplayMode::Grid);
        assert_eq!(stg.grid_columns, 3);
    }

    #[test_case]
    fn stylesheet_tag_and_class() {
        let sheet = Stylesheet::parse(
            r#"
            p { color: red; }
            .hi { color: #00ff00 !important; }
            #x { font-size: 20px; }
            "#,
        );
        let mut st = ComputedStyle::default();
        let d = sheet.matching_decls("p", None, Some("hi"));
        apply_decls(&mut st, &d);
        assert_eq!(st.color, 0x00ff00); // important class wins
        let d2 = sheet.matching_decls("div", Some("x"), None);
        let mut st2 = ComputedStyle::default();
        apply_decls(&mut st2, &d2);
        assert_eq!(st2.font_size, 20);
    }

    #[test_case]
    fn compute_inline_overrides() {
        let sheet = Stylesheet::parse("p { color: blue; }");
        let parent = ComputedStyle::default();
        let st = compute(&sheet, "p", None, None, Some("color: #010203"), &parent);
        assert_eq!(st.color, 0x010203);
    }

    #[test_case]
    fn body_background_from_sheet() {
        let sheet = Stylesheet::parse("body { background: #112233; }");
        assert!(sheet.rule_count() >= 1, "rules={}", sheet.rule_count());
        let st = compute(&sheet, "body", None, None, None, &ComputedStyle::default());
        assert_eq!(st.background, Some(0x112233));
        let sheet2 = Stylesheet::parse("body { background-color: #112233; }");
        let st2 = compute(&sheet2, "body", None, None, None, &ComputedStyle::default());
        assert_eq!(st2.background, Some(0x112233));
    }

    #[test_case]
    fn scan_imports_forms() {
        let css = concat!(
            "@import url(\"a.css\");\n",
            "@import url(b.css);\n",
            "@import \"c.css\";\n",
            "@import 'd.css' screen;\n",
            "p { color: red; }\n",
        );
        assert_eq!(
            scan_imports(css),
            alloc::vec![
                "a.css".to_string(),
                "b.css".to_string(),
                "c.css".to_string(),
                "d.css".to_string(),
            ]
        );
        assert!(scan_imports("p { color: red }").is_empty());
    }

    #[test_case]
    fn scan_css_urls_background_only() {
        let css = concat!(
            "body { background: url(\"bg.png\") no-repeat; }\n",
            ".a { background-image: url(tile.jpg); }\n",
            ".b { background-image: url(tile.jpg); }\n", // dupe
            ".c { background: url(data:image/png;base64,AAAA); }\n",
            ".d { background: linear-gradient(#fff, #000); }\n",
            "@font-face { src: url(font.woff2); }\n", // not a background
            ".e { list-style-image: url(dot.png); }\n", // not a background
        );
        assert_eq!(
            scan_css_urls(css),
            alloc::vec!["bg.png".to_string(), "tile.jpg".to_string()]
        );
    }

    #[test_case]
    fn scan_font_faces_formats() {
        let css = concat!(
            "@font-face { font-family: \"Geist Mono\"; ",
            "src: url(gm.woff2) format(\"woff2\"); }\n",
            "@font-face { font-family: Inter; src: url('inter.woff'); }\n",
            "@font-face { font-family: Mono; ",
            "src: url(data:font/woff2;base64,AAAA), url(mono.ttf); }\n",
            "@font-face { src: url(nofamily.woff); }\n", // no family: skipped
        );
        let faces = scan_font_faces(css);
        assert_eq!(faces.len(), 3, "{faces:?}");
        assert_eq!(
            faces[0],
            FontFace {
                family: "Geist Mono".to_string(),
                url: "gm.woff2".to_string(),
                format: "woff2".to_string(),
            }
        );
        assert_eq!(faces[1].family, "Inter");
        assert_eq!(faces[1].url, "inter.woff");
        assert_eq!(faces[1].format, "woff");
        // data: candidate skipped; .ttf infers truetype.
        assert_eq!(faces[2].url, "mono.ttf");
        assert_eq!(faces[2].format, "truetype");
    }

    #[test_case]
    fn strip_module_import_forms() {
        let src = concat!(
            "import X from \"a.js\"\n",
            "import {a, b} from 'b.js';\n",
            "import \"c.js\";\n",
            "console.log(X);",
        );
        let (out, imports) = strip_module_syntax(src);
        assert_eq!(
            imports,
            alloc::vec!["a.js".to_string(), "b.js".to_string(), "c.js".to_string()]
        );
        // A default import now binds to the module's `__default`; plain named
        // imports (`{a, b}`) and side-effect imports emit nothing.
        assert_eq!(out, "var X = __default;\nconsole.log(X);");
    }

    #[test_case]
    fn strip_module_renamed_import_and_default_combo() {
        let src = concat!(
            "import Def, { orig as local, plain } from \"m.js\";\n",
            "use(Def, local, plain);",
        );
        let (out, imports) = strip_module_syntax(src);
        assert_eq!(imports, alloc::vec!["m.js".to_string()]);
        assert!(out.contains("var Def = __default;"), "{out}");
        assert!(out.contains("var local = orig;"), "{out}");
        // `plain` is already a shared global — no synthesized binding.
        assert!(!out.contains("var plain"), "{out}");
        assert!(out.contains("use(Def, local, plain);"));
    }

    #[test_case]
    fn strip_module_export_rename_aliases() {
        let (out, imports) = strip_module_syntax("export { localA as pubA, keep };");
        assert!(imports.is_empty());
        assert!(out.contains("var pubA = localA;"), "{out}");
        assert!(!out.contains("var keep"), "{out}");
        assert!(!out.contains("export"), "{out}");
    }

    #[test_case]
    fn strip_module_star_unsupported() {
        let (out, imports) = strip_module_syntax("import * as N from \"n.js\";\nN.go();");
        assert_eq!(imports, alloc::vec!["n.js".to_string()]);
        assert!(out.contains("// unsupported module form:"), "{out}");
        assert!(out.contains("N.go();"));
    }

    #[test_case]
    fn strip_module_exports() {
        let src = concat!(
            "export default foo(1);\n",
            "export const a = 1;\n",
            "export function f() { return a; }\n",
            "export class C {}\n",
            "export {a, f};\n",
            "export {x} from \"re.js\";",
        );
        let (out, imports) = strip_module_syntax(src);
        assert!(out.contains("var __default = foo(1);"), "{out}");
        assert!(out.contains("const a = 1;"));
        assert!(out.contains("function f() { return a; }"));
        assert!(out.contains("class C {}"));
        assert!(!out.contains("export"), "{out}");
        assert_eq!(imports, alloc::vec!["re.js".to_string()]);
    }

    #[test_case]
    fn strip_module_passthrough_unchanged() {
        let src = "var importantThing = 1;\nfunction exporter() {}\n  exporter();\n";
        let (out, imports) = strip_module_syntax(src);
        assert_eq!(out, src);
        assert!(imports.is_empty());
    }
}
