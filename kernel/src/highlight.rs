//! Line-based syntax highlighting for JSON, Markdown, and common programming
//! languages — pure classification logic (bytes in, per-byte [`Class`] out),
//! shared by three surfaces:
//!
//! * the `/open` **editor** (per-cell colours via [`classes`]),
//! * **`/cat`** (ANSI-coloured lines via [`ansi_line`]),
//! * the **chat stream** ([`StreamMd`]): markdown-aware colouring of the
//!   model's streamed reply — headings, fence markers, and fenced code blocks
//!   (lexed per language tag) — without breaking token-by-token streaming:
//!   prose passes straight through; only heading/fence-candidate lines and
//!   fence interiors are held until their newline, then emitted coloured.
//!
//! The lexers are deliberately small: line comments, block comments (state
//! carried across lines), strings with escapes, numbers, keywords, brackets.
//! Good-enough terminal highlighting, not a grammar. ASCII-oriented — any
//! non-ASCII byte is plain text.

use alloc::string::String;
use alloc::vec::Vec;

/// Languages with a dedicated lexer. Everything else renders as plain text.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Lang {
    Json,
    Md,
    Rust,
    Python,
    C,
    Js,
    Toml,
    Sh,
    /// Go (and similar C-family tooling syntax).
    Go,
}

/// Token classes a byte can belong to; the colour mapping is [`rgb`].
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Class {
    Text,
    Keyword,
    /// String literals; in JSON also used for value strings (keys are
    /// [`Class::Keyword`], the classic JSON colour split).
    Str,
    Number,
    Comment,
    Punct,
    /// Markdown heading line.
    Heading,
    /// Markdown code (inline span, untagged fence interior, fence markers).
    Code,
}

/// Index of a class in the [`SYNTAX_COLORS`] palette table.
#[inline]
fn class_index(c: Class) -> usize {
    match c {
        Class::Text => 0,
        Class::Keyword => 1,
        Class::Str => 2,
        Class::Number => 3,
        Class::Comment => 4,
        Class::Punct => 5,
        Class::Heading => 6,
        Class::Code => 7,
    }
}

/// Default syntax palette, from the brand palette (DESIGN.md): terracotta
/// keywords, teal strings/code, amber numbers, muted comments. Order matches
/// [`class_index`]: text, keyword, string, number, comment, punct, heading, code.
pub const DEFAULT_SYNTAX: [(u8, u8, u8); 8] = [
    (250, 249, 245), // Text — cream (chat_fg default)
    (204, 120, 92),  // Keyword — primary terracotta #cc785c
    (93, 184, 166),  // Str — accent-teal #5db8a6
    (232, 165, 90),  // Number — accent-amber #e8a55a
    (108, 106, 100), // Comment — muted #6c6a64
    (142, 139, 130), // Punct — muted-soft #8e8b82
    (204, 120, 92),  // Heading — terracotta
    (93, 184, 166),  // Code — teal
];

/// The live syntax palette, seeded with [`DEFAULT_SYNTAX`] and overridable by a
/// theme via [`set_syntax_colors`]. All three highlight surfaces read it through
/// [`rgb`], so one call reaches the editor, `/cat`, and the chat stream.
static SYNTAX_COLORS: crate::mm::Locked<[(u8, u8, u8); 8]> =
    crate::mm::Locked::new(DEFAULT_SYNTAX);

/// The colour for a token class (theme-driven — see [`set_syntax_colors`]).
pub fn rgb(c: Class) -> (u8, u8, u8) {
    let i = class_index(c);
    SYNTAX_COLORS.with(|p| p[i])
}

/// Map a theme syntax-key (`"keyword"`, `"string"`, `"comment"`, `"number"`,
/// `"text"`, `"punct"`, `"heading"`, `"code"`) to its palette index.
fn syntax_index(name: &str) -> Option<usize> {
    Some(match name {
        "text" => 0,
        "keyword" => 1,
        "string" | "str" => 2,
        "number" | "num" => 3,
        "comment" => 4,
        "punct" | "punctuation" => 5,
        "heading" => 6,
        "code" => 7,
        _ => return None,
    })
}

/// Override the live syntax palette from a theme's `(key, rgb)` pairs. Unknown
/// keys are ignored; unlisted classes keep the default. Pass an empty slice to
/// reset (callers should first copy [`DEFAULT_SYNTAX`]).
pub fn set_syntax_colors(pairs: &[(&str, (u8, u8, u8))]) {
    SYNTAX_COLORS.with(|p| {
        for (name, rgb) in pairs {
            if let Some(i) = syntax_index(name) {
                p[i] = *rgb;
            }
        }
    });
}

/// Reset the syntax palette to the brand defaults.
pub fn reset_syntax_colors() {
    SYNTAX_COLORS.with(|p| *p = DEFAULT_SYNTAX);
}

/// Cross-line lexer state: C-family block comments and Markdown fences.
#[derive(Clone, Copy, Default)]
pub struct State {
    block_comment: bool,
    /// Inside a ``` fence: the tag's language, if recognized.
    fence: Option<Option<Lang>>,
}

/// Pick a language from a file path's extension (case-insensitive).
pub fn lang_for_path(path: &str) -> Option<Lang> {
    let ext = path.rsplit('/').next().unwrap_or(path).rsplit_once('.').map(|(_, e)| e)?;
    lang_for_tag(ext)
}

/// Pick a language from a Markdown fence tag (```rust) or file extension.
pub fn lang_for_tag(tag: &str) -> Option<Lang> {
    let t = tag.trim();
    // Case-insensitive compare without allocating.
    let eq = |s: &str| t.eq_ignore_ascii_case(s);
    Some(match () {
        _ if eq("json") => Lang::Json,
        _ if eq("md") || eq("markdown") => Lang::Md,
        _ if eq("rs") || eq("rust") => Lang::Rust,
        _ if eq("py") || eq("python") => Lang::Python,
        _ if eq("c") || eq("h") || eq("cpp") || eq("hpp") || eq("cc") => Lang::C,
        _ if eq("js") || eq("javascript") || eq("ts") || eq("typescript") || eq("mjs") => Lang::Js,
        _ if eq("toml") || eq("ini") || eq("conf") => Lang::Toml,
        _ if eq("sh") || eq("bash") || eq("zsh") || eq("shell") => Lang::Sh,
        _ if eq("go") || eq("golang") => Lang::Go,
        _ => return None,
    })
}

fn keywords(lang: Lang) -> &'static [&'static str] {
    match lang {
        Lang::Json => &["true", "false", "null"],
        Lang::Md => &[],
        Lang::Rust => &[
            "as", "async", "await", "break", "const", "continue", "crate", "dyn", "else", "enum", "extern", "false",
            "fn", "for", "if", "impl", "in", "let", "loop", "match", "mod", "move", "mut", "pub", "ref", "return",
            "self", "Self", "static", "struct", "super", "trait", "true", "type", "unsafe", "use", "where", "while",
        ],
        Lang::Python => &[
            "and", "as", "assert", "async", "await", "break", "class", "continue", "def", "del", "elif", "else",
            "except", "finally", "for", "from", "global", "if", "import", "in", "is", "lambda", "None", "nonlocal",
            "not", "or", "pass", "raise", "return", "True", "False", "try", "while", "with", "yield",
        ],
        Lang::C => &[
            "break", "case", "char", "const", "continue", "default", "do", "double", "else", "enum", "extern",
            "float", "for", "goto", "if", "inline", "int", "long", "return", "short", "signed", "sizeof", "static",
            "struct", "switch", "typedef", "union", "unsigned", "void", "volatile", "while",
        ],
        Lang::Js => &[
            "async", "await", "break", "case", "catch", "class", "const", "continue", "default", "do", "else",
            "export", "extends", "false", "finally", "for", "from", "function", "if", "import", "in", "instanceof",
            "let", "new", "null", "of", "return", "switch", "this", "throw", "true", "try", "typeof", "undefined",
            "var", "while",
        ],
        Lang::Toml => &["true", "false"],
        Lang::Sh => &[
            "case", "do", "done", "echo", "elif", "else", "esac", "export", "fi", "for", "function", "if", "in",
            "local", "return", "then", "while",
        ],
        Lang::Go => &[
            "break", "case", "chan", "const", "continue", "default", "defer", "else", "fallthrough", "for", "func",
            "go", "goto", "if", "import", "interface", "map", "package", "range", "return", "select", "struct",
            "switch", "type", "var", "true", "false", "nil", "iota", "make", "new", "len", "cap", "append", "copy",
            "delete", "close", "panic", "recover",
        ],
    }
}

/// Does `lang` use `//` line + `/* */` block comments (C family)?
fn c_comments(lang: Lang) -> bool {
    matches!(lang, Lang::Rust | Lang::C | Lang::Js | Lang::Go)
}

/// Does `lang` use `#` line comments?
fn hash_comments(lang: Lang) -> bool {
    matches!(lang, Lang::Python | Lang::Toml | Lang::Sh)
}

/// Classify every byte of `line` (no trailing newline), carrying `st` across
/// lines. The returned Vec has exactly `line.len()` entries.
pub fn classes(lang: Lang, line: &str, st: &mut State) -> Vec<Class> {
    let b = line.as_bytes();
    let mut out = alloc::vec![Class::Text; b.len()];
    match lang {
        Lang::Md => lex_md(b, st, &mut out),
        _ => lex_code(lang, b, st, &mut out),
    }
    out
}

/// Advance the cross-line state over `line` without keeping the classes
/// (used to seed the state at a viewport's first visible line).
pub fn advance(lang: Lang, line: &str, st: &mut State) {
    let _ = classes(lang, line, st);
}

fn find(b: &[u8], from: usize, pat: &[u8]) -> Option<usize> {
    (from..b.len().saturating_sub(pat.len() - 1)).find(|&i| &b[i..i + pat.len()] == pat)
}

fn mark(out: &mut [Class], lo: usize, hi: usize, c: Class) {
    let hi = hi.min(out.len());
    for x in out.iter_mut().take(hi).skip(lo) {
        *x = c;
    }
}

fn lex_code(lang: Lang, b: &[u8], st: &mut State, out: &mut [Class]) {
    let mut i = 0;
    if st.block_comment {
        match find(b, 0, b"*/") {
            Some(e) => {
                mark(out, 0, e + 2, Class::Comment);
                st.block_comment = false;
                i = e + 2;
            }
            None => {
                mark(out, 0, b.len(), Class::Comment);
                return;
            }
        }
    }
    while i < b.len() {
        let c = b[i];
        // Comments.
        if c_comments(lang) && c == b'/' && b.get(i + 1) == Some(&b'*') {
            match find(b, i + 2, b"*/") {
                Some(e) => {
                    mark(out, i, e + 2, Class::Comment);
                    i = e + 2;
                }
                None => {
                    mark(out, i, b.len(), Class::Comment);
                    st.block_comment = true;
                    return;
                }
            }
            continue;
        }
        if (c_comments(lang) && c == b'/' && b.get(i + 1) == Some(&b'/')) || (hash_comments(lang) && c == b'#') {
            mark(out, i, b.len(), Class::Comment);
            return;
        }
        // Strings ('"' everywhere; '\'' where it delimits strings/chars).
        if c == b'"' || (c == b'\'' && matches!(lang, Lang::Python | Lang::Js | Lang::Sh | Lang::C)) {
            let start = i;
            i += 1;
            while i < b.len() {
                if b[i] == b'\\' {
                    i += 2;
                } else if b[i] == c {
                    i += 1;
                    break;
                } else {
                    i += 1;
                }
            }
            let end = i.min(b.len());
            // JSON: a string followed by ':' is an object key.
            let is_key = lang == Lang::Json
                && b[end..].iter().find(|&&x| x != b' ').is_some_and(|&x| x == b':');
            mark(out, start, end, if is_key { Class::Keyword } else { Class::Str });
            continue;
        }
        // Numbers (incl. hex/float/underscore digits).
        if c.is_ascii_digit() {
            let start = i;
            while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'.' || b[i] == b'_') {
                i += 1;
            }
            mark(out, start, i, Class::Number);
            continue;
        }
        // Identifiers / keywords.
        if c.is_ascii_alphabetic() || c == b'_' {
            let start = i;
            while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                i += 1;
            }
            // SAFETY-free: this slice is ASCII by construction.
            let word = core::str::from_utf8(&b[start..i]).unwrap_or("");
            if keywords(lang).contains(&word) {
                mark(out, start, i, Class::Keyword);
            }
            continue;
        }
        if matches!(c, b'{' | b'}' | b'[' | b']' | b'(' | b')' | b',' | b':' | b';') {
            out[i] = Class::Punct;
        }
        i += 1;
    }
}

/// A line's fence marker tag, if the trimmed line opens/closes a ``` fence.
fn fence_tag(line: &[u8]) -> Option<&[u8]> {
    trim_start(line).strip_prefix(b"```".as_slice())
}

fn trim_start(b: &[u8]) -> &[u8] {
    let n = b.iter().take_while(|&&c| c == b' ' || c == b'\t').count();
    &b[n..]
}

fn lex_md(b: &[u8], st: &mut State, out: &mut [Class]) {
    if let Some(tag) = fence_tag(b) {
        // Opening marker records the tag's language; closing clears it.
        st.fence = match st.fence {
            Some(_) => None,
            None => Some(core::str::from_utf8(tag).ok().and_then(lang_for_tag)),
        };
        mark(out, 0, b.len(), Class::Code);
        return;
    }
    if let Some(fl) = st.fence {
        match fl {
            Some(lang) => lex_code(lang, b, st, out),
            None => mark(out, 0, b.len(), Class::Code),
        }
        return;
    }
    let t = trim_start(b);
    if t.first() == Some(&b'#') {
        mark(out, 0, b.len(), Class::Heading);
        return;
    }
    // List bullets: `- `, `* `, `+ `.
    if t.len() >= 2 && matches!(t[0], b'-' | b'*' | b'+') && t[1] == b' ' {
        let at = b.len() - t.len();
        out[at] = Class::Punct;
    }
    // Inline spans: `code` and **bold**.
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'`' {
            if let Some(e) = b[i + 1..].iter().position(|&c| c == b'`') {
                mark(out, i, i + e + 2, Class::Code);
                i += e + 2;
                continue;
            }
        }
        if b[i] == b'*' && b.get(i + 1) == Some(&b'*') {
            if let Some(e) = find(b, i + 2, b"**") {
                mark(out, i, e + 2, Class::Keyword);
                i = e + 2;
                continue;
            }
        }
        i += 1;
    }
}

/// Render one line as ANSI-coloured text (24-bit SGR runs + trailing reset).
/// Bytes classified [`Class::Text`] are left uncoloured, so the surrounding
/// stream's default colour shows through.
pub fn ansi_line(lang: Lang, line: &str, st: &mut State) -> String {
    let cls = classes(lang, line, st);
    let mut out = String::with_capacity(line.len() + 16);
    let mut cur = Class::Text;
    for (i, ch) in line.bytes().enumerate() {
        let c = cls[i];
        if c != cur {
            if c == Class::Text {
                out.push_str("\x1b[0m");
            } else {
                let (r, g, b) = rgb(c);
                out.push_str(&alloc::format!("\x1b[38;2;{};{};{}m", r, g, b));
            }
            cur = c;
        }
        out.push(ch as char);
    }
    if cur != Class::Text {
        out.push_str("\x1b[0m");
    }
    out
}

/// Longest a held-back line can grow before it is flushed raw (a runaway
/// "line" without newlines must not buffer unboundedly).
const HOLD_MAX: usize = 4096;

/// Emit ANSI for a coloured run of `text` as class `c`.
fn sgr_paint(c: Class, text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }
    if c == Class::Text {
        return text.into();
    }
    let (r, g, b) = rgb(c);
    alloc::format!("\x1b[38;2;{r};{g};{b}m{text}\x1b[0m")
}

/// Bold + colour for headings / emphasis (bordered rendered markdown).
fn sgr_bold_paint(c: Class, text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }
    let (r, g, b) = rgb(c);
    alloc::format!("\x1b[1m\x1b[38;2;{r};{g};{b}m{text}\x1b[0m")
}

/// Dim accent gutter used for fenced code (left rail).
fn code_gutter() -> String {
    let (r, g, b) = rgb(Class::Comment);
    alloc::format!("\x1b[38;2;{r};{g};{b}m│\x1b[0m ")
}

/// Render one finished source line of markdown (no trailing newline) to ANSI,
/// updating fence state. bordered: **hides** ``` fence markers, strips
/// heading `#` / bold `**` / inline `` ` `` markers, syntax-highlights code.
fn render_md_line(line: &str, st: &mut State) -> String {
    let b = line.as_bytes();
    // Fence open/close — never print the triple-backtick line (hidden intentionally).
    if let Some(tag_bytes) = fence_tag(b) {
        match st.fence {
            Some(_) => {
                // Closing fence: optional blank visual break already via \n.
                st.fence = None;
                return String::new();
            }
            None => {
                let tag = core::str::from_utf8(tag_bytes).unwrap_or("").trim();
                // Drop optional language info after space (```go title).
                let lang_tag = tag.split(|c: char| c.is_whitespace()).next().unwrap_or("");
                let lang = lang_for_tag(lang_tag);
                st.fence = Some(lang);
                // Dim language label only (no backticks).
                if lang_tag.is_empty() {
                    return String::new();
                }
                let (r, g, b) = rgb(Class::Comment);
                return alloc::format!(
                    "\x1b[38;2;{r};{g};{b}m  {lang_tag}\x1b[0m"
                );
            }
        }
    }
    // Inside a fence: gutter + language lex.
    if let Some(fl) = st.fence {
        let body = match fl {
            Some(lang) => ansi_line(lang, line, st),
            None => sgr_paint(Class::Code, line),
        };
        return alloc::format!("{}{body}", code_gutter());
    }
    // Headings: strip leading #, bold + heading colour (markers removed first
    // so we don't nest SGR inside SGR).
    let t = line.trim_start();
    if t.starts_with('#') {
        let level = t.bytes().take_while(|&c| c == b'#').count();
        if level > 0 && (t.as_bytes().get(level) == Some(&b' ') || t.len() == level) {
            let title = t[level..].trim_start();
            let plain = strip_inline_markers(title);
            return sgr_bold_paint(Class::Heading, &plain);
        }
    }
    // List bullets: colour the marker, render rest inline.
    let trimmed = line.trim_start();
    let indent = line.len() - trimmed.len();
    if let Some(rest) = trimmed
        .strip_prefix("- ")
        .or_else(|| trimmed.strip_prefix("* "))
        .or_else(|| trimmed.strip_prefix("+ "))
    {
        let pad = " ".repeat(indent);
        let bullet = sgr_paint(Class::Punct, "•");
        let body = render_inline_md(rest);
        return alloc::format!("{pad}{bullet} {body}");
    }
    // Numbered list `1. `.
    if let Some(dot) = trimmed.find(". ") {
        if trimmed[..dot].bytes().all(|c| c.is_ascii_digit()) && !trimmed[..dot].is_empty() {
            let pad = " ".repeat(indent);
            let num = sgr_paint(Class::Number, &trimmed[..=dot]);
            let body = render_inline_md(&trimmed[dot + 2..]);
            return alloc::format!("{pad}{num} {body}");
        }
    }
    render_inline_md(line)
}

/// Remove common inline markdown markers, keep plain text only.
fn strip_inline_markers(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'`' {
            if let Some(rel) = b[i + 1..].iter().position(|&c| c == b'`') {
                out.push_str(core::str::from_utf8(&b[i + 1..i + 1 + rel]).unwrap_or(""));
                i += rel + 2;
                continue;
            }
        }
        if b[i] == b'*' && b.get(i + 1) == Some(&b'*') {
            if let Some(e) = find(b, i + 2, b"**") {
                out.push_str(core::str::from_utf8(&b[i + 2..e]).unwrap_or(""));
                i = e + 2;
                continue;
            }
        }
        if b[i] == b'*' {
            // skip unpaired emphasis markers
            i += 1;
            continue;
        }
        out.push(b[i] as char);
        i += 1;
    }
    out
}

/// Inline markdown: strip `**bold**`, `*italic*`, `` `code` `` markers and paint.
fn render_inline_md(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = String::with_capacity(s.len() + 8);
    let mut i = 0;
    while i < b.len() {
        // Inline code `…`
        if b[i] == b'`' {
            if let Some(rel) = b[i + 1..].iter().position(|&c| c == b'`') {
                let content = core::str::from_utf8(&b[i + 1..i + 1 + rel]).unwrap_or("");
                out.push_str(&sgr_paint(Class::Code, content));
                i += rel + 2;
                continue;
            }
        }
        // **bold**
        if b[i] == b'*' && b.get(i + 1) == Some(&b'*') {
            if let Some(e) = find(b, i + 2, b"**") {
                let content = core::str::from_utf8(&b[i + 2..e]).unwrap_or("");
                out.push_str(&sgr_bold_paint(Class::Keyword, content));
                i = e + 2;
                continue;
            }
        }
        // *italic* (single)
        if b[i] == b'*' && b.get(i + 1) != Some(&b'*') {
            if let Some(rel) = b[i + 1..].iter().position(|&c| c == b'*') {
                // Don't treat `*` list-ish mid-word as italic if no close soon.
                let content = core::str::from_utf8(&b[i + 1..i + 1 + rel]).unwrap_or("");
                if !content.is_empty() && !content.contains('\n') {
                    out.push_str(&sgr_paint(Class::Keyword, content));
                    i += rel + 2;
                    continue;
                }
            }
        }
        out.push(b[i] as char);
        i += 1;
    }
    out
}

/// Streaming Markdown renderer for chat (bordered).
///
/// Line-buffered: each complete line is fully rendered (markers stripped,
/// fences hidden, code highlighted) before emit — same checkpoint idea as
/// streaming markdown (stable per line).
pub struct StreamMd {
    st: State,
    hold: String,
}

impl StreamMd {
    pub fn new() -> StreamMd {
        StreamMd {
            st: State::default(),
            hold: String::new(),
        }
    }

    pub fn feed(&mut self, s: &str, emit: &mut dyn FnMut(&str)) {
        for ch in s.chars() {
            if ch == '\n' {
                let line = core::mem::take(&mut self.hold);
                let rendered = render_md_line(&line, &mut self.st);
                // Skip completely empty fence-close lines without a blank gap
                // double; still emit newline so block structure remains.
                emit(&rendered);
                emit("\n");
            } else {
                self.hold.push(ch);
                if self.hold.len() >= HOLD_MAX {
                    let line = core::mem::take(&mut self.hold);
                    emit(&render_md_line(&line, &mut self.st));
                }
            }
        }
    }

    /// Flush anything still held (end of the reply mid-line).
    pub fn finish(&mut self, emit: &mut dyn FnMut(&str)) {
        if !self.hold.is_empty() {
            let line = core::mem::take(&mut self.hold);
            emit(&render_md_line(&line, &mut self.st));
        }
    }
}

/// Render a complete markdown document to ANSI (remote / batch path).
/// Callers should run `shell::chrome::sanitize_chat_text` first when the
/// source may contain fancy Unicode punctuation.
pub fn render_md_document(text: &str) -> String {
    let mut sm = StreamMd::new();
    let mut out = String::new();
    sm.feed(text, &mut |s| out.push_str(s));
    sm.finish(&mut |s| out.push_str(s));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cls(lang: Lang, line: &str) -> Vec<Class> {
        classes(lang, line, &mut State::default())
    }

    #[test_case]
    fn syntax_palette_is_themeable() {
        // Default matches the brand palette.
        assert_eq!(rgb(Class::Keyword), DEFAULT_SYNTAX[1]);
        // A theme override changes the live colour; unlisted keys keep default.
        set_syntax_colors(&[("keyword", (1, 2, 3)), ("string", (4, 5, 6))]);
        assert_eq!(rgb(Class::Keyword), (1, 2, 3));
        assert_eq!(rgb(Class::Str), (4, 5, 6));
        assert_eq!(rgb(Class::Comment), DEFAULT_SYNTAX[4]); // untouched
        // Reset restores the brand defaults (so other tests see a clean palette).
        reset_syntax_colors();
        assert_eq!(rgb(Class::Keyword), DEFAULT_SYNTAX[1]);
    }

    #[test_case]
    fn json_keys_values_numbers() {
        let c = cls(Lang::Json, r#"{"name": "chitti", "port": 8080, "on": true}"#);
        assert_eq!(c[1], Class::Keyword, "object key");
        assert_eq!(c[9], Class::Str, "value string");
        assert_eq!(c[28], Class::Number, "number 8080");
        assert_eq!(c[0], Class::Punct, "open brace");
        assert_eq!(c[39], Class::Keyword, "true constant");
    }

    #[test_case]
    fn rust_keywords_strings_comments() {
        let c = cls(Lang::Rust, r#"pub fn main() { let s = "hi"; } // done"#);
        assert_eq!(c[0], Class::Keyword, "pub");
        assert_eq!(c[4], Class::Keyword, "fn");
        assert_eq!(c[7], Class::Text, "ident main");
        assert_eq!(c[24], Class::Str, "string literal");
        assert_eq!(c[32], Class::Comment, "line comment");
    }

    #[test_case]
    fn block_comment_state_carries_across_lines() {
        let mut st = State::default();
        let c1 = classes(Lang::C, "int x; /* start", &mut st);
        assert_eq!(c1[0], Class::Keyword);
        assert_eq!(*c1.last().unwrap(), Class::Comment);
        let c2 = classes(Lang::C, "still comment */ int y;", &mut st);
        assert_eq!(c2[0], Class::Comment);
        assert_eq!(c2[17], Class::Keyword, "int after the close");
        assert!(!st.block_comment);
    }

    #[test_case]
    fn python_hash_comment_and_quotes() {
        let c = cls(Lang::Python, "x = 'a' # note");
        assert_eq!(c[4], Class::Str);
        assert_eq!(c[8], Class::Comment);
    }

    #[test_case]
    fn md_heading_fence_and_inline_code() {
        let mut st = State::default();
        let h = classes(Lang::Md, "## Title", &mut st);
        assert!(h.iter().all(|&c| c == Class::Heading));
        let f = classes(Lang::Md, "```rust", &mut st);
        assert!(f.iter().all(|&c| c == Class::Code));
        let inside = classes(Lang::Md, "let x = 1;", &mut st);
        assert_eq!(inside[0], Class::Keyword, "fence interior lexed as rust");
        let close = classes(Lang::Md, "```", &mut st);
        assert!(close.iter().all(|&c| c == Class::Code));
        let prose = classes(Lang::Md, "plain `code` text", &mut st);
        assert_eq!(prose[0], Class::Text);
        assert_eq!(prose[7], Class::Code, "inline code span");
    }

    #[test_case]
    fn lang_detection() {
        assert_eq!(lang_for_path("/configs/core/ui.json"), Some(Lang::Json));
        assert_eq!(lang_for_path("SOUL.md"), Some(Lang::Md));
        assert_eq!(lang_for_path("a/b/x.rs"), Some(Lang::Rust));
        assert_eq!(lang_for_path("noext"), None);
        assert_eq!(lang_for_tag("python"), Some(Lang::Python));
        assert_eq!(lang_for_tag("nosuch"), None);
    }

    #[test_case]
    fn ansi_line_wraps_runs_and_resets() {
        let s = ansi_line(Lang::Json, r#"{"k": 1}"#, &mut State::default());
        assert!(s.contains("\x1b[38;2;"), "colour escapes present");
        assert!(s.ends_with("\x1b[0m") || !s.contains("\x1b["), "reset at end");
        // The visible characters survive exactly.
        let stripped: String = {
            let mut out = String::new();
            let mut esc = false;
            for c in s.chars() {
                match (esc, c) {
                    (false, '\x1b') => esc = true,
                    (false, _) => out.push(c),
                    (true, 'm') => esc = false,
                    (true, _) => {}
                }
            }
            out
        };
        assert_eq!(stripped, r#"{"k": 1}"#);
    }

    fn strip_ansi(s: &str) -> String {
        let mut stripped = String::new();
        let mut esc = false;
        for c in s.chars() {
            match (esc, c) {
                (false, '\x1b') => esc = true,
                (false, _) => stripped.push(c),
                (true, 'm') => esc = false,
                (true, _) => {}
            }
        }
        stripped
    }

    #[test_case]
    fn stream_md_hides_fences_and_colours_code() {
        let mut sm = StreamMd::new();
        let mut out = String::new();
        // Line-buffered: incomplete line is held.
        sm.feed("hello ", &mut |s| out.push_str(s));
        assert!(out.is_empty(), "partial line held");
        sm.feed(
            "world\n```rust\nfn main() {}\n```\ndone",
            &mut |s| out.push_str(s),
        );
        sm.finish(&mut |s| out.push_str(s));
        assert!(out.contains("\x1b[38;2;"), "fence content coloured");
        let stripped = strip_ansi(&out);
        // Fence markers hidden (bordered); lang label + code remain.
        assert!(!stripped.contains("```"), "backtick fences must be hidden: {stripped}");
        assert!(stripped.contains("fn main()"), "code body present: {stripped}");
        assert!(stripped.contains("done"), "trailing prose: {stripped}");
        assert!(stripped.contains("rust") || stripped.contains("│"), "lang label or gutter");
    }

    #[test_case]
    fn stream_md_go_fence_and_bold() {
        let out = render_md_document("```go\npackage main\nfunc main() {}\n```\n**bold** tip");
        let stripped = strip_ansi(&out);
        assert!(!stripped.contains("```"));
        assert!(stripped.contains("package main"));
        assert!(stripped.contains("bold tip") || stripped.contains("bold"), "{stripped}");
        assert!(!stripped.contains("**"), "bold markers stripped: {stripped}");
        assert!(out.contains("\x1b[38;2;"), "coloured");
        // Go keywords get keyword colour.
        assert_eq!(lang_for_tag("go"), Some(Lang::Go));
        let c = classes(Lang::Go, "func main()", &mut State::default());
        assert_eq!(c[0], Class::Keyword, "func is a Go keyword");
    }

    #[test_case]
    fn stream_md_heading_and_list() {
        let out = render_md_document("## Title\n- item one\n1. second");
        let stripped = strip_ansi(&out);
        assert!(!stripped.starts_with("##"), "heading hashes stripped: {stripped}");
        assert!(stripped.contains("Title"));
        assert!(stripped.contains("item one"));
        assert!(!stripped.contains("- item"), "dash bullet replaced: {stripped}");
    }
}
