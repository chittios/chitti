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
        "border" | "border-color" | "outline-color" => {
            for tok in value.split_whitespace() {
                if let Some(c) = parse_color(tok) {
                    st.border_color = Some(c);
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
            // Accept and ignore (layout does not clear floats yet).
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
            if let Some(px) = parse_px(value) {
                st.outline_offset = st.outline_offset; // keep
                let _ = px;
            }
            let _ = value;
        }
        "border-spacing" => {
            let _ = parse_px(value);
        }
        "font-stretch" | "font-variant" | "font-feature-settings" | "font-variation-settings" => {
            let _ = value;
        }
        "zoom" | "contain" | "color-scheme" | "scroll-behavior" | "backface-visibility"
        | "object-position" | "mask" | "mask-image" | "text-wrap" | "forced-color-adjust"
        | "table-layout" | "container-type" | "overscroll-behavior" => {
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
            let _ = value;
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
        "filter" | "backdrop-filter" | "content" | "outline" | "outline-width"
        | "outline-style" => {
            let _ = value;
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
            }
        }
        "border-top" | "border-bottom" | "border-left" | "border-right" | "border-style" => {
            for tok in value.split_whitespace() {
                if let Some(c) = parse_color(tok) {
                    st.border_color = Some(c);
                    match name.as_str() {
                        "border-top" => st.border_top_color = Some(c),
                        "border-bottom" => st.border_bottom_color = Some(c),
                        "border-left" => st.border_left_color = Some(c),
                        "border-right" => st.border_right_color = Some(c),
                        _ => {}
                    }
                }
                if let Some(px) = parse_px(tok) {
                    match name.as_str() {
                        "border-top" => st.border_top_width = px,
                        "border-bottom" => st.border_bottom_width = px,
                        "border-left" => st.border_left_width = px,
                        "border-right" => st.border_right_width = px,
                        _ => {}
                    }
                }
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
        "background-image" | "background-size" | "background-position" | "background-repeat"
        | "background-clip" | "background-origin" => {
            let _ = value;
        }
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
    st
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
}
