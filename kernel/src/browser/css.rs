//! **CSS engine (subset)** — pure no_std stylesheet parse + cascade.
//!
//! Structure follows Ladybird LibCSS / StyleComputer (selector match →
//! cascade → computed). Full
//! [MDN CSS](https://developer.mozilla.org/en-US/docs/Web/CSS) is not in
//! scope; we expand properties behind the same APIs.
//!
//! Supported:
//! - Rules: `tag`, `.class`, `#id`, `*` (simple; optional `tag.class` / `tag#id`)
//! - Declarations: `color`, `background`/`background-color`, `font-size`,
//!   `margin`/`margin-*`, `padding`/`padding-*`, `display`, `font-weight`,
//!   `text-align`, `width`/`height`/`max-width`, `opacity`, `line-height`,
//!   `border`/`border-color` (stored as background-ish chrome for boxes)
//! - Inline `style="…"` (highest priority after `!important` author rules)
//!
//! Not supported: @media, full flex/grid, calc(), variables, pseudo-elements,
//! combinators beyond a single simple selector, full cascade layers.

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
    pub padding_top: i32,
    pub padding_bottom: i32,
    pub padding_left: i32,
    pub padding_right: i32,
    pub display_none: bool,
    pub bold: bool,
    pub text_align: Align,
    pub width: Option<i32>,
    pub height: Option<i32>,
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
    pub grid_gap: i32,
    pub flex_wrap: super::flex::FlexWrap,
    pub flex_grow: u32,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DisplayMode {
    #[default]
    Block,
    Inline,
    Flex,
    Grid,
    None,
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
            padding_top: 0,
            padding_bottom: 0,
            padding_left: 0,
            padding_right: 0,
            display_none: false,
            bold: false,
            text_align: Align::Left,
            width: None,
            height: None,
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
            grid_gap: 0,
            flex_wrap: super::flex::FlexWrap::NoWrap,
            flex_grow: 0,
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

/// Resolve simple `var(--name)` / `var(--name, fallback)` against custom props.
fn resolve_var(value: &str, props: &alloc::collections::BTreeMap<String, String>) -> String {
    let v = value.trim();
    if let Some(rest) = v.strip_prefix("var(") {
        let inner = rest.trim_end_matches(')').trim();
        let (name, fallback) = if let Some((a, b)) = inner.split_once(',') {
            (a.trim(), Some(b.trim()))
        } else {
            (inner, None)
        };
        let key = name.trim_start_matches('-'); // allow --foo or foo
        let key = if name.starts_with("--") {
            name.to_string()
        } else {
            format!("--{key}")
        };
        if let Some(val) = props.get(&key).or_else(|| props.get(name)) {
            return val.clone();
        }
        if let Some(fb) = fallback {
            return fb.to_string();
        }
    }
    value.to_string()
}

#[derive(Clone, Debug)]
struct Decl {
    name: String,
    value: String,
    important: bool,
}

#[derive(Clone, Debug)]
enum Sel {
    Universal,
    Tag(String),
    Class(String),
    Id(String),
    TagClass(String, String),
    TagId(String, String),
}

#[derive(Clone, Debug)]
struct Rule {
    sel: Sel,
    decls: Vec<Decl>,
    /// Specificity: (id, class, type)
    spec: (u8, u8, u8),
    order: u32,
    /// Cascade layer index (`None` = unlayered, wins over all layers).
    /// Lower index = earlier in layer order = lower priority.
    layer: Option<u32>,
}

/// Parsed stylesheet (LibCSS-inspired: rules sorted by cascade order).
/// Supports CSS **cascade layers** (`@layer name, …` / `@layer name { … }`).
#[derive(Clone, Debug, Default)]
pub struct Stylesheet {
    rules: Vec<Rule>,
    /// Declared layer order (name → index).
    layer_order: alloc::collections::BTreeMap<String, u32>,
    next_layer: u32,
    /// Current layer when parsing nested `@layer name { }`.
    parse_layer: Option<u32>,
}

impl Stylesheet {
    pub fn parse(css: &str) -> Self {
        let mut sheet = Stylesheet::default();
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
                if let Some(sel) = parse_selector(sel_str.trim()) {
                    let spec = specificity(&sel);
                    let order = self.rules.len() as u32;
                    self.rules.push(Rule {
                        sel,
                        decls: decls.clone(),
                        spec,
                        order,
                        layer,
                    });
                }
            }
        }
    }

    /// Match rules for an element; return merged decls sorted by cascade.
    /// Order: layer (unlayered last / highest) → specificity → source order.
    pub fn matching_decls(
        &self,
        tag: &str,
        id: Option<&str>,
        class: Option<&str>,
    ) -> Vec<(String, String, bool)> {
        let mut matched: Vec<&Rule> = self
            .rules
            .iter()
            .filter(|r| matches_sel(&r.sel, tag, id, class))
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

fn specificity(sel: &Sel) -> (u8, u8, u8) {
    match sel {
        Sel::Universal => (0, 0, 0),
        Sel::Tag(_) => (0, 0, 1),
        Sel::Class(_) => (0, 1, 0),
        Sel::Id(_) => (1, 0, 0),
        Sel::TagClass(_, _) => (0, 1, 1),
        Sel::TagId(_, _) => (1, 0, 1),
    }
}

fn parse_selector(s: &str) -> Option<Sel> {
    let s = s.trim();
    if s.is_empty() || s.contains(' ') || s.contains('>') || s.contains('+') || s.contains('~') {
        // Combinators unsupported in v1.
        if s.contains(' ') || s.contains('>') {
            // Take the last simple part (descendant: use rightmost).
            let last = s.split(|c: char| c == ' ' || c == '>').last()?.trim();
            return parse_selector(last);
        }
    }
    if s == "*" {
        return Some(Sel::Universal);
    }
    if let Some(rest) = s.strip_prefix('#') {
        if !rest.is_empty() {
            return Some(Sel::Id(rest.to_string()));
        }
    }
    if let Some(rest) = s.strip_prefix('.') {
        if !rest.is_empty() {
            return Some(Sel::Class(rest.to_string()));
        }
    }
    // tag, tag.class, tag#id
    if let Some(i) = s.find('.') {
        let (t, c) = s.split_at(i);
        let c = &c[1..];
        if !t.is_empty() && !c.is_empty() {
            return Some(Sel::TagClass(t.to_ascii_lowercase(), c.to_string()));
        }
    }
    if let Some(i) = s.find('#') {
        let (t, id) = s.split_at(i);
        let id = &id[1..];
        if !t.is_empty() && !id.is_empty() {
            return Some(Sel::TagId(t.to_ascii_lowercase(), id.to_string()));
        }
    }
    if s.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_') {
        return Some(Sel::Tag(s.to_ascii_lowercase()));
    }
    None
}

fn matches_sel(sel: &Sel, tag: &str, id: Option<&str>, class: Option<&str>) -> bool {
    let classes = class.unwrap_or("");
    let has_class = |c: &str| classes.split_whitespace().any(|x| x == c);
    match sel {
        Sel::Universal => true,
        Sel::Tag(t) => t.eq_ignore_ascii_case(tag),
        Sel::Class(c) => has_class(c),
        Sel::Id(i) => id.map(|x| x == i.as_str()).unwrap_or(false),
        Sel::TagClass(t, c) => t.eq_ignore_ascii_case(tag) && has_class(c),
        Sel::TagId(t, i) => t.eq_ignore_ascii_case(tag) && id.map(|x| x == i.as_str()).unwrap_or(false),
    }
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

/// Parse a color: `#rgb`, `#rrggbb`, `rgb(r,g,b)`, named basics.
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
            // #RGBA → RGB only
            let r = u32::from_str_radix(&hex[0..1], 16).ok()?;
            let g = u32::from_str_radix(&hex[1..2], 16).ok()?;
            let b = u32::from_str_radix(&hex[2..3], 16).ok()?;
            return Some((r << 20) | (r << 16) | (g << 12) | (g << 8) | (b << 4) | b);
        }
        if hex.len() == 6 {
            return u32::from_str_radix(hex, 16).ok();
        }
        if hex.len() == 8 {
            // #RRGGBBAA → RGB
            return u32::from_str_radix(&hex[0..6], 16).ok();
        }
    }
    if let Some(inner) = s.strip_prefix("rgb(").and_then(|x| x.strip_suffix(')')) {
        let mut parts = inner.split(',');
        let r: u32 = parts.next()?.trim().parse().ok()?;
        let g: u32 = parts.next()?.trim().parse().ok()?;
        let b: u32 = parts.next()?.trim().parse().ok()?;
        return Some(((r.min(255)) << 16) | ((g.min(255)) << 8) | b.min(255));
    }
    None
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

fn parse_px(s: &str) -> Option<i32> {
    let s = s.trim().to_ascii_lowercase();
    if let Some(n) = s.strip_suffix("px") {
        return n.trim().parse().ok();
    }
    if let Some(n) = s.strip_suffix("em") {
        let f: i32 = n.trim().parse().ok()?;
        return Some(f * 14);
    }
    if s == "0" {
        return Some(0);
    }
    s.parse().ok()
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
            for tok in value.split_whitespace() {
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
            if let Some(px) = parse_px(value.split_whitespace().next().unwrap_or(value)) {
                st.margin_top = px;
                st.margin_bottom = px;
                st.margin_left = px;
                st.margin_right = px;
            }
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
            if let Some(px) = parse_px(value) {
                st.margin_left = px;
            }
        }
        "margin-right" => {
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
                "flex" | "inline-flex" => DisplayMode::Flex,
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
            // Count columns: `1fr 1fr 1fr` or `repeat(3, 1fr)` or `100px 100px`
            let v = value.trim().to_ascii_lowercase();
            if let Some(rest) = v.strip_prefix("repeat(") {
                let n: u8 = rest
                    .split(',')
                    .next()
                    .and_then(|s| s.trim().parse().ok())
                    .unwrap_or(1);
                st.grid_columns = n.clamp(1, 12);
            } else {
                let n = v.split_whitespace().filter(|t| !t.is_empty()).count();
                st.grid_columns = (n as u8).clamp(1, 12);
            }
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
        "flex" => {
            // flex: grow shrink basis — take first number as grow
            if let Some(g) = value
                .split_whitespace()
                .next()
                .and_then(|t| t.parse::<u32>().ok())
            {
                st.flex_grow = g;
            }
            if value.contains("wrap") {
                st.flex_wrap = super::flex::FlexWrap::Wrap;
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
            }
        }
        "height" => {
            if let Some(px) = parse_px(value) {
                st.height = Some(px);
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
            for tok in value.split_whitespace() {
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
            for tok in value.split_whitespace() {
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
            if let Some(px) = parse_px(value.split_whitespace().next().unwrap_or(value)) {
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
        "animation-fill-mode" | "animation-iteration-count" => {
            // cascade accept
            let _ = value;
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
            let _ = parse_px(value);
        }
        "font-stretch" | "font-variant" | "font-feature-settings" | "font-variation-settings" => {
            let _ = value;
        }
        "object-position" => st.object_position = value.trim().to_string(),
        "mask" => st.mask = value.trim().to_string(),
        "mask-image" => st.mask = value.trim().to_string(),
        "table-layout" => st.table_layout = value.trim().to_ascii_lowercase(),
        "zoom" | "contain" | "color-scheme" | "scroll-behavior" | "backface-visibility"
        | "text-wrap" | "forced-color-adjust" | "container-type" | "overscroll-behavior" => {
            let _ = value;
        }
        "stroke-dasharray" | "stroke-dashoffset" => {
            let _ = value;
        }
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
            let parts: Vec<&str> = value.split_whitespace().collect();
            if let Some(p) = parts.first() {
                st.animation_name = (*p).to_string();
            }
            if let Some(d) = parts.get(1) {
                st.animation_duration = (*d).to_string();
            }
            if let Some(t) = parts.get(2) {
                st.animation_timing = (*t).to_string();
            }
            if let Some(delay) = parts.get(3) {
                st.animation_delay = (*delay).to_string();
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
            for tok in value.split_whitespace() {
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
            for tok in value.split_whitespace() {
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
            }
        }
        "border-bottom-width" => {
            if let Some(px) = parse_px(value) {
                st.border_bottom_width = px;
            }
        }
        "border-left-width" => {
            if let Some(px) = parse_px(value) {
                st.border_left_width = px;
            }
        }
        "border-right-width" => {
            if let Some(px) = parse_px(value) {
                st.border_right_width = px;
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
        "flex-shrink" | "flex-basis" | "align-content" | "align-self" | "order"
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
        }
        "strong" | "b" => {
            st.bold = true;
        }
        "code" | "pre" => {
            st.font_size = parent.font_size.saturating_sub(1).max(10);
        }
        _ => {}
    }
    let mut decls = sheet.matching_decls(tag, id, class);
    if let Some(inl) = inline {
        for d in parse_decls(inl) {
            // Inline style ≈ specificity (1,0,0,0) — applied after author rules
            // in the same important/non-important passes (later wins).
            decls.push((d.name, d.value, d.important));
        }
    }
    apply_decls(&mut st, &decls);
    // `filter` is approximated as a colour transform over the element's own
    // colours (grayscale/invert/brightness/opacity) — blur is not rasterized.
    if !st.filter.is_empty() {
        apply_filter_colors(&mut st);
    }
    st
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
pub fn strip_module_syntax(src: &str) -> (String, Vec<String>) {
    let mut imports = Vec::new();
    let mut out: Vec<String> = Vec::new();
    for line in src.split('\n') {
        let t = line.trim_start();
        let indent = &line[..line.len() - t.len()];
        if starts_kw(t, "import") {
            if let Some(spec) = quoted_spec(t) {
                imports.push(spec);
                if t["import".len()..].trim_start().starts_with('*') {
                    // Namespace imports need a real module record — flag it.
                    out.push(format!(
                        "{indent}// unsupported module form: {}",
                        t.trim_end()
                    ));
                }
                continue; // supported forms: line removed
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
                }
                // `export {a, b};` — bindings already exist as globals.
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

    #[test_case]
    fn parse_color_hex_and_name() {
        assert_eq!(parse_color("#ff0000"), Some(0xff0000));
        assert_eq!(parse_color("#f00"), Some(0xff0000));
        assert_eq!(parse_color("blue"), Some(0x0000ff));
        assert_eq!(parse_color("rgb(1, 2, 3)"), Some(0x010203));
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
        assert_eq!(out, "console.log(X);");
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
