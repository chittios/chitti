//! HTML element catalog — every tag on MDN’s
//! [HTML elements reference](https://developer.mozilla.org/en-US/docs/Web/HTML/Reference/Elements)
//! is classified for layout (Ladybird maps each to `HTML*Element` +
//! `Layout::Node`; we use a compact `DisplayKind` + category table).
//!
//! Unknown tags still parse as elements (`HTMLUnknownElement` spirit) and lay
//! out as generic blocks/inlines via [`classify`].

use alloc::string::String;

/// How layout treats the element (CSS `display` default + embedding).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DisplayKind {
    /// Not rendered (metadata, scripts, templates).
    None,
    /// Block box (`display:block` default).
    Block,
    /// Inline text-level.
    Inline,
    /// Inline but may wrap like block children (e.g. `button`).
    InlineBlock,
    /// List container.
    List,
    /// List item.
    ListItem,
    /// Table model (simplified: block with row/cell children).
    Table,
    TableRow,
    TableCell,
    /// Replaced / embedded: img, iframe, video, canvas, object, embed.
    Embedded,
    /// Form control widget.
    Control,
    /// Void line break.
    Break,
    /// Horizontal rule.
    Rule,
}

/// MDN-ish category for diagnostics / agent tooling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Category {
    MainRoot,
    DocumentMetadata,
    Sectioning,
    ContentSectioning,
    TextContent,
    InlineText,
    ImageMultimedia,
    EmbeddedContent,
    SvgMath, // not rendered as SVG here
    Scripting,
    DemarcatingEdit,
    Table,
    Forms,
    Interactive,
    WebComponents,
    Obsolete,
    Unknown,
}

/// Metadata for one HTML tag name.
#[derive(Clone, Copy, Debug)]
pub struct ElementInfo {
    pub tag: &'static str,
    pub display: DisplayKind,
    pub category: Category,
    pub void: bool,
}

/// All HTML elements from MDN (127) + a few obsolete still seen in the wild.
/// Order is alphabetical by tag for binary search.
pub static ELEMENTS: &[ElementInfo] = &[
    ElementInfo { tag: "a", display: DisplayKind::Inline, category: Category::InlineText, void: false },
    ElementInfo { tag: "abbr", display: DisplayKind::Inline, category: Category::InlineText, void: false },
    ElementInfo { tag: "acronym", display: DisplayKind::Inline, category: Category::Obsolete, void: false },
    ElementInfo { tag: "address", display: DisplayKind::Block, category: Category::ContentSectioning, void: false },
    ElementInfo { tag: "area", display: DisplayKind::None, category: Category::ImageMultimedia, void: true },
    ElementInfo { tag: "article", display: DisplayKind::Block, category: Category::ContentSectioning, void: false },
    ElementInfo { tag: "aside", display: DisplayKind::Block, category: Category::ContentSectioning, void: false },
    ElementInfo { tag: "audio", display: DisplayKind::Embedded, category: Category::ImageMultimedia, void: false },
    ElementInfo { tag: "b", display: DisplayKind::Inline, category: Category::InlineText, void: false },
    ElementInfo { tag: "base", display: DisplayKind::None, category: Category::DocumentMetadata, void: true },
    ElementInfo { tag: "bdi", display: DisplayKind::Inline, category: Category::InlineText, void: false },
    ElementInfo { tag: "bdo", display: DisplayKind::Inline, category: Category::InlineText, void: false },
    ElementInfo { tag: "big", display: DisplayKind::Inline, category: Category::Obsolete, void: false },
    ElementInfo { tag: "blockquote", display: DisplayKind::Block, category: Category::TextContent, void: false },
    ElementInfo { tag: "body", display: DisplayKind::Block, category: Category::Sectioning, void: false },
    ElementInfo { tag: "br", display: DisplayKind::Break, category: Category::InlineText, void: true },
    ElementInfo { tag: "button", display: DisplayKind::Control, category: Category::Forms, void: false },
    ElementInfo { tag: "canvas", display: DisplayKind::Embedded, category: Category::Scripting, void: false },
    ElementInfo { tag: "caption", display: DisplayKind::Block, category: Category::Table, void: false },
    ElementInfo { tag: "center", display: DisplayKind::Block, category: Category::Obsolete, void: false },
    ElementInfo { tag: "cite", display: DisplayKind::Inline, category: Category::InlineText, void: false },
    ElementInfo { tag: "code", display: DisplayKind::Inline, category: Category::InlineText, void: false },
    ElementInfo { tag: "col", display: DisplayKind::None, category: Category::Table, void: true },
    ElementInfo { tag: "colgroup", display: DisplayKind::None, category: Category::Table, void: false },
    ElementInfo { tag: "data", display: DisplayKind::Inline, category: Category::InlineText, void: false },
    ElementInfo { tag: "datalist", display: DisplayKind::None, category: Category::Forms, void: false },
    ElementInfo { tag: "dd", display: DisplayKind::Block, category: Category::TextContent, void: false },
    ElementInfo { tag: "del", display: DisplayKind::Inline, category: Category::DemarcatingEdit, void: false },
    ElementInfo { tag: "details", display: DisplayKind::Block, category: Category::Interactive, void: false },
    ElementInfo { tag: "dfn", display: DisplayKind::Inline, category: Category::InlineText, void: false },
    ElementInfo { tag: "dialog", display: DisplayKind::Block, category: Category::Interactive, void: false },
    ElementInfo { tag: "dir", display: DisplayKind::List, category: Category::Obsolete, void: false },
    ElementInfo { tag: "div", display: DisplayKind::Block, category: Category::TextContent, void: false },
    ElementInfo { tag: "dl", display: DisplayKind::Block, category: Category::TextContent, void: false },
    ElementInfo { tag: "dt", display: DisplayKind::Block, category: Category::TextContent, void: false },
    ElementInfo { tag: "em", display: DisplayKind::Inline, category: Category::InlineText, void: false },
    ElementInfo { tag: "embed", display: DisplayKind::Embedded, category: Category::EmbeddedContent, void: true },
    ElementInfo { tag: "fencedframe", display: DisplayKind::Embedded, category: Category::EmbeddedContent, void: false },
    ElementInfo { tag: "fieldset", display: DisplayKind::Block, category: Category::Forms, void: false },
    ElementInfo { tag: "figcaption", display: DisplayKind::Block, category: Category::TextContent, void: false },
    ElementInfo { tag: "figure", display: DisplayKind::Block, category: Category::TextContent, void: false },
    ElementInfo { tag: "font", display: DisplayKind::Inline, category: Category::Obsolete, void: false },
    ElementInfo { tag: "footer", display: DisplayKind::Block, category: Category::ContentSectioning, void: false },
    ElementInfo { tag: "form", display: DisplayKind::Block, category: Category::Forms, void: false },
    ElementInfo { tag: "frame", display: DisplayKind::Embedded, category: Category::Obsolete, void: true },
    ElementInfo { tag: "frameset", display: DisplayKind::Block, category: Category::Obsolete, void: false },
    ElementInfo { tag: "h1", display: DisplayKind::Block, category: Category::ContentSectioning, void: false },
    ElementInfo { tag: "h2", display: DisplayKind::Block, category: Category::ContentSectioning, void: false },
    ElementInfo { tag: "h3", display: DisplayKind::Block, category: Category::ContentSectioning, void: false },
    ElementInfo { tag: "h4", display: DisplayKind::Block, category: Category::ContentSectioning, void: false },
    ElementInfo { tag: "h5", display: DisplayKind::Block, category: Category::ContentSectioning, void: false },
    ElementInfo { tag: "h6", display: DisplayKind::Block, category: Category::ContentSectioning, void: false },
    ElementInfo { tag: "head", display: DisplayKind::None, category: Category::DocumentMetadata, void: false },
    ElementInfo { tag: "header", display: DisplayKind::Block, category: Category::ContentSectioning, void: false },
    ElementInfo { tag: "hgroup", display: DisplayKind::Block, category: Category::ContentSectioning, void: false },
    ElementInfo { tag: "hr", display: DisplayKind::Rule, category: Category::TextContent, void: true },
    ElementInfo { tag: "html", display: DisplayKind::Block, category: Category::MainRoot, void: false },
    ElementInfo { tag: "i", display: DisplayKind::Inline, category: Category::InlineText, void: false },
    ElementInfo { tag: "iframe", display: DisplayKind::Embedded, category: Category::EmbeddedContent, void: false },
    ElementInfo { tag: "img", display: DisplayKind::Embedded, category: Category::ImageMultimedia, void: true },
    ElementInfo { tag: "input", display: DisplayKind::Control, category: Category::Forms, void: true },
    ElementInfo { tag: "ins", display: DisplayKind::Inline, category: Category::DemarcatingEdit, void: false },
    ElementInfo { tag: "kbd", display: DisplayKind::Inline, category: Category::InlineText, void: false },
    ElementInfo { tag: "label", display: DisplayKind::Inline, category: Category::Forms, void: false },
    ElementInfo { tag: "legend", display: DisplayKind::Block, category: Category::Forms, void: false },
    ElementInfo { tag: "li", display: DisplayKind::ListItem, category: Category::TextContent, void: false },
    ElementInfo { tag: "link", display: DisplayKind::None, category: Category::DocumentMetadata, void: true },
    ElementInfo { tag: "main", display: DisplayKind::Block, category: Category::ContentSectioning, void: false },
    ElementInfo { tag: "map", display: DisplayKind::Inline, category: Category::ImageMultimedia, void: false },
    ElementInfo { tag: "mark", display: DisplayKind::Inline, category: Category::InlineText, void: false },
    ElementInfo { tag: "marquee", display: DisplayKind::Block, category: Category::Obsolete, void: false },
    ElementInfo { tag: "menu", display: DisplayKind::List, category: Category::Interactive, void: false },
    ElementInfo { tag: "meta", display: DisplayKind::None, category: Category::DocumentMetadata, void: true },
    ElementInfo { tag: "meter", display: DisplayKind::InlineBlock, category: Category::Forms, void: false },
    ElementInfo { tag: "nav", display: DisplayKind::Block, category: Category::ContentSectioning, void: false },
    ElementInfo { tag: "nobr", display: DisplayKind::Inline, category: Category::Obsolete, void: false },
    ElementInfo { tag: "noembed", display: DisplayKind::None, category: Category::Obsolete, void: false },
    ElementInfo { tag: "noframes", display: DisplayKind::None, category: Category::Obsolete, void: false },
    ElementInfo { tag: "noscript", display: DisplayKind::None, category: Category::Scripting, void: false },
    ElementInfo { tag: "object", display: DisplayKind::Embedded, category: Category::EmbeddedContent, void: false },
    ElementInfo { tag: "ol", display: DisplayKind::List, category: Category::TextContent, void: false },
    ElementInfo { tag: "optgroup", display: DisplayKind::None, category: Category::Forms, void: false },
    ElementInfo { tag: "option", display: DisplayKind::None, category: Category::Forms, void: false },
    ElementInfo { tag: "output", display: DisplayKind::Inline, category: Category::Forms, void: false },
    ElementInfo { tag: "p", display: DisplayKind::Block, category: Category::TextContent, void: false },
    ElementInfo { tag: "param", display: DisplayKind::None, category: Category::EmbeddedContent, void: true },
    ElementInfo { tag: "picture", display: DisplayKind::Embedded, category: Category::ImageMultimedia, void: false },
    ElementInfo { tag: "plaintext", display: DisplayKind::Block, category: Category::Obsolete, void: false },
    ElementInfo { tag: "pre", display: DisplayKind::Block, category: Category::TextContent, void: false },
    ElementInfo { tag: "progress", display: DisplayKind::InlineBlock, category: Category::Forms, void: false },
    ElementInfo { tag: "q", display: DisplayKind::Inline, category: Category::InlineText, void: false },
    ElementInfo { tag: "rb", display: DisplayKind::Inline, category: Category::InlineText, void: false },
    ElementInfo { tag: "rp", display: DisplayKind::None, category: Category::InlineText, void: false },
    ElementInfo { tag: "rt", display: DisplayKind::Inline, category: Category::InlineText, void: false },
    ElementInfo { tag: "rtc", display: DisplayKind::Inline, category: Category::InlineText, void: false },
    ElementInfo { tag: "ruby", display: DisplayKind::Inline, category: Category::InlineText, void: false },
    ElementInfo { tag: "s", display: DisplayKind::Inline, category: Category::InlineText, void: false },
    ElementInfo { tag: "samp", display: DisplayKind::Inline, category: Category::InlineText, void: false },
    ElementInfo { tag: "script", display: DisplayKind::None, category: Category::Scripting, void: false },
    ElementInfo { tag: "search", display: DisplayKind::Block, category: Category::ContentSectioning, void: false },
    ElementInfo { tag: "section", display: DisplayKind::Block, category: Category::ContentSectioning, void: false },
    ElementInfo { tag: "select", display: DisplayKind::Control, category: Category::Forms, void: false },
    ElementInfo { tag: "selectedcontent", display: DisplayKind::Inline, category: Category::Forms, void: false },
    ElementInfo { tag: "slot", display: DisplayKind::Inline, category: Category::WebComponents, void: false },
    ElementInfo { tag: "small", display: DisplayKind::Inline, category: Category::InlineText, void: false },
    ElementInfo { tag: "source", display: DisplayKind::None, category: Category::ImageMultimedia, void: true },
    ElementInfo { tag: "span", display: DisplayKind::Inline, category: Category::InlineText, void: false },
    ElementInfo { tag: "strike", display: DisplayKind::Inline, category: Category::Obsolete, void: false },
    ElementInfo { tag: "strong", display: DisplayKind::Inline, category: Category::InlineText, void: false },
    ElementInfo { tag: "style", display: DisplayKind::None, category: Category::DocumentMetadata, void: false },
    ElementInfo { tag: "sub", display: DisplayKind::Inline, category: Category::InlineText, void: false },
    ElementInfo { tag: "summary", display: DisplayKind::Block, category: Category::Interactive, void: false },
    ElementInfo { tag: "sup", display: DisplayKind::Inline, category: Category::InlineText, void: false },
    ElementInfo { tag: "table", display: DisplayKind::Table, category: Category::Table, void: false },
    ElementInfo { tag: "tbody", display: DisplayKind::Table, category: Category::Table, void: false },
    ElementInfo { tag: "td", display: DisplayKind::TableCell, category: Category::Table, void: false },
    ElementInfo { tag: "template", display: DisplayKind::None, category: Category::WebComponents, void: false },
    ElementInfo { tag: "textarea", display: DisplayKind::Control, category: Category::Forms, void: false },
    ElementInfo { tag: "tfoot", display: DisplayKind::Table, category: Category::Table, void: false },
    ElementInfo { tag: "th", display: DisplayKind::TableCell, category: Category::Table, void: false },
    ElementInfo { tag: "thead", display: DisplayKind::Table, category: Category::Table, void: false },
    ElementInfo { tag: "time", display: DisplayKind::Inline, category: Category::InlineText, void: false },
    ElementInfo { tag: "title", display: DisplayKind::None, category: Category::DocumentMetadata, void: false },
    ElementInfo { tag: "tr", display: DisplayKind::TableRow, category: Category::Table, void: false },
    ElementInfo { tag: "track", display: DisplayKind::None, category: Category::ImageMultimedia, void: true },
    ElementInfo { tag: "tt", display: DisplayKind::Inline, category: Category::Obsolete, void: false },
    ElementInfo { tag: "u", display: DisplayKind::Inline, category: Category::InlineText, void: false },
    ElementInfo { tag: "ul", display: DisplayKind::List, category: Category::TextContent, void: false },
    ElementInfo { tag: "var", display: DisplayKind::Inline, category: Category::InlineText, void: false },
    ElementInfo { tag: "video", display: DisplayKind::Embedded, category: Category::ImageMultimedia, void: false },
    ElementInfo { tag: "wbr", display: DisplayKind::Break, category: Category::InlineText, void: true },
    ElementInfo { tag: "xmp", display: DisplayKind::Block, category: Category::Obsolete, void: false },
];

/// Look up a tag (ASCII case-insensitive). `None` → unknown → block.
pub fn lookup(tag: &str) -> Option<&'static ElementInfo> {
    let t = tag.to_ascii_lowercase();
    ELEMENTS
        .binary_search_by(|e| e.tag.cmp(t.as_str()))
        .ok()
        .map(|i| &ELEMENTS[i])
}

pub fn classify(tag: &str) -> DisplayKind {
    lookup(tag).map(|e| e.display).unwrap_or(DisplayKind::Block)
}

pub fn is_void(tag: &str) -> bool {
    lookup(tag).map(|e| e.void).unwrap_or(false)
}

pub fn is_known(tag: &str) -> bool {
    lookup(tag).is_some()
}

/// Number of MDN-listed tags in the table.
pub fn catalog_len() -> usize {
    ELEMENTS.len()
}

/// Human label for agent status.
pub fn coverage_summary() -> String {
    alloc::format!(
        "html elements: {} catalogued (MDN reference); unknown → block (HTMLUnknownElement)",
        ELEMENTS.len()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn catalog_sorted_and_covers_core() {
        assert!(ELEMENTS.len() >= 120, "expected full MDN set, got {}", ELEMENTS.len());
        for w in ELEMENTS.windows(2) {
            assert!(w[0].tag < w[1].tag, "unsorted: {} >= {}", w[0].tag, w[1].tag);
        }
        assert!(is_known("iframe"));
        assert!(is_known("video"));
        assert!(is_known("dialog"));
        assert_eq!(classify("iframe"), DisplayKind::Embedded);
        assert_eq!(classify("div"), DisplayKind::Block);
        assert_eq!(classify("span"), DisplayKind::Inline);
        assert_eq!(classify("table"), DisplayKind::Table);
        assert_eq!(classify("tr"), DisplayKind::TableRow);
        assert_eq!(classify("td"), DisplayKind::TableCell);
        assert!(is_void("img"));
        assert!(is_void("br"));
        assert!(!is_void("div"));
        // Unknown custom element → block (not dropped).
        assert_eq!(classify("my-widget"), DisplayKind::Block);
        assert!(!is_known("my-widget"));
    }
}
