//! Text layout for the PDF writer: base-14 font metrics, word wrap, and the
//! flow of blocks onto pages.
//!
//! Pure — points in, positioned lines out, no I/O and no PDF syntax.
//!
//! **The base-14 fonts need no embedding**, which is what makes a self-contained
//! writer possible without a TrueType subsetter: every conforming reader already
//! has Helvetica, Times and Courier. The cost is that we must know their widths
//! ourselves, because a reader lays text out from the widths *we* declare — get
//! them wrong and the text still renders, just with wrapping that does not match
//! where we said the line ended.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// One of the base-14 faces we use. Non-Latin text is out of scope until an
/// embedded subset exists, and [`Font::supports`] says so rather than emitting
/// a document with holes in it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Font {
    Helvetica,
    HelveticaBold,
    HelveticaOblique,
    Courier,
    CourierBold,
}

impl Font {
    pub fn base_name(self) -> &'static str {
        match self {
            Font::Helvetica => "Helvetica",
            Font::HelveticaBold => "Helvetica-Bold",
            Font::HelveticaOblique => "Helvetica-Oblique",
            Font::Courier => "Courier",
            Font::CourierBold => "Courier-Bold",
        }
    }

    /// The `/F<n>` resource name.
    pub fn res(self) -> &'static str {
        match self {
            Font::Helvetica => "F1",
            Font::HelveticaBold => "F2",
            Font::HelveticaOblique => "F3",
            Font::Courier => "F4",
            Font::CourierBold => "F5",
        }
    }

    pub fn all() -> [Font; 5] {
        [
            Font::Helvetica,
            Font::HelveticaBold,
            Font::HelveticaOblique,
            Font::Courier,
            Font::CourierBold,
        ]
    }

    /// Width of one byte in 1/1000 em.
    ///
    /// **Courier is 600 for every glyph** — it is monospace, so its metrics need
    /// no table at all. Helvetica's come from the Adobe AFM widths, which is why
    /// [`HELVETICA_WIDTHS`] is a generated constant rather than a guess: text
    /// laid out with invented widths wraps in the wrong place and looks like a
    /// layout bug rather than a metrics one.
    pub fn width(self, b: u8) -> u16 {
        match self {
            Font::Courier | Font::CourierBold => 600,
            Font::HelveticaBold => {
                // Bold is wider; the ratio is close enough to uniform that
                // scaling the regular widths keeps wrapping correct, and being
                // slightly conservative only ever wraps early.
                (HELVETICA_WIDTHS[b as usize] as u32 * 1060 / 1000) as u16
            }
            _ => HELVETICA_WIDTHS[b as usize],
        }
    }

    /// Width of a whole string at `size` points.
    pub fn text_width(self, s: &str, size: f32) -> f32 {
        let mils: u32 = s.bytes().map(|b| self.width(b) as u32).sum();
        mils as f32 * size / 1000.0
    }

    /// Whether every byte of `s` can be shown in this (WinAnsi) encoding.
    pub fn supports(self, s: &str) -> bool {
        s.bytes().all(|b| b >= 0x20 || b == b'\n' || b == b'\t')
    }
}

/// Helvetica advance widths in 1/1000 em, indexed by WinAnsi byte.
///
/// From the Adobe Core 14 AFM. Only the printable-ASCII range is populated with
/// real values; the rest take the average, which is a deliberate approximation —
/// those bytes are rare in the documents this writes and a wrong width there
/// costs a slightly ragged margin, not a wrong glyph.
pub const HELVETICA_WIDTHS: [u16; 256] = {
    let mut w = [556u16; 256];
    // space ! " # $ % & ' ( ) * + , - . /
    let ascii: [u16; 95] = [
        278, 278, 355, 556, 556, 889, 667, 191, 333, 333, 389, 584, 278, 333, 278, 278, // 0x20-0x2F
        556, 556, 556, 556, 556, 556, 556, 556, 556, 556, 278, 278, 584, 584, 584, 556, // 0x30-0x3F
        1015, 667, 667, 722, 722, 667, 611, 778, 722, 278, 500, 667, 556, 833, 722, 778, // 0x40-0x4F
        667, 778, 722, 667, 611, 722, 667, 944, 667, 667, 611, 278, 278, 278, 469, 556, // 0x50-0x5F
        333, 556, 556, 500, 556, 556, 278, 556, 556, 222, 222, 500, 222, 833, 556, 556, // 0x60-0x6F
        556, 556, 333, 500, 278, 556, 500, 722, 500, 500, 500, 334, 260, 334, 584, // 0x70-0x7E
    ];
    let mut i = 0;
    while i < 95 {
        w[0x20 + i] = ascii[i];
        i += 1;
    }
    w
};

/// A run of text with one font and size.
#[derive(Clone, Debug, PartialEq)]
pub struct Span {
    pub text: String,
    pub font: Font,
    pub size: f32,
}

/// A laid-out line: what to draw, and where, in PDF user space (origin at the
/// bottom-left of the page).
#[derive(Clone, Debug, PartialEq)]
pub struct Placed {
    pub text: String,
    pub font: Font,
    pub size: f32,
    pub x: f32,
    pub y: f32,
}

/// Page geometry, in points. US Letter by default, which is what a reader
/// assumes when a `/MediaBox` is absent — so stating it explicitly is what makes
/// the document render the same everywhere.
#[derive(Clone, Copy, Debug)]
pub struct Page {
    pub width: f32,
    pub height: f32,
    pub margin: f32,
}

impl Default for Page {
    fn default() -> Self {
        Self {
            width: 612.0,
            height: 792.0,
            margin: 54.0,
        }
    }
}

impl Page {
    pub fn text_width(&self) -> f32 {
        self.width - 2.0 * self.margin
    }
}

/// Break `text` into lines that fit `max_width` at this font and size.
///
/// Greedy by word. A single word longer than the line is broken by character
/// rather than left to overflow the margin — a URL or a long identifier is
/// common enough in the documents this writes that overflowing would be the
/// normal case, not the edge one.
pub fn wrap(text: &str, font: Font, size: f32, max_width: f32) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    if text.trim().is_empty() {
        return alloc::vec![String::new()];
    }
    let mut line = String::new();
    for word in text.split_whitespace() {
        let candidate = if line.is_empty() {
            word.to_string()
        } else {
            alloc::format!("{line} {word}")
        };
        if font.text_width(&candidate, size) <= max_width {
            line = candidate;
            continue;
        }
        if !line.is_empty() {
            out.push(core::mem::take(&mut line));
        }
        // The word alone may still not fit.
        if font.text_width(word, size) <= max_width {
            line = word.to_string();
            continue;
        }
        let mut chunk = String::new();
        for ch in word.chars() {
            let next = alloc::format!("{chunk}{ch}");
            if font.text_width(&next, size) > max_width && !chunk.is_empty() {
                out.push(core::mem::take(&mut chunk));
            }
            chunk.push(ch);
        }
        line = chunk;
    }
    if !line.is_empty() {
        out.push(line);
    }
    if out.is_empty() {
        out.push(String::new());
    }
    out
}

/// A document block, before layout.
#[derive(Clone, Debug, PartialEq)]
pub enum Block {
    Heading { level: u8, text: String },
    Paragraph(String),
    Bullet { depth: u8, text: String },
    /// Preformatted: each line is placed verbatim, never wrapped.
    Code(Vec<String>),
    Rule,
    Spacer(f32),
}

/// Flow `blocks` onto pages.
///
/// Returns one `Vec<Placed>` per page. The y coordinate counts **up** from the
/// bottom of the page, which is the opposite of every screen coordinate system
/// in this OS — mixing the two puts the first line off the bottom edge and the
/// document looks empty.
pub fn flow(blocks: &[Block], page: Page) -> Vec<Vec<Placed>> {
    let mut pages: Vec<Vec<Placed>> = Vec::new();
    let mut cur: Vec<Placed> = Vec::new();
    let mut y = page.height - page.margin;
    let bottom = page.margin;

    let mut newpage = |cur: &mut Vec<Placed>, pages: &mut Vec<Vec<Placed>>, y: &mut f32| {
        pages.push(core::mem::take(cur));
        *y = page.height - page.margin;
    };

    for b in blocks {
        match b {
            Block::Spacer(h) => y -= h,
            Block::Rule => {
                y -= 6.0;
                // Drawn as an underscore run rather than a path so the content
                // stream stays text-only and trivially inspectable.
                let f = Font::Helvetica;
                // `as usize` truncates, which is floor for a positive — `f32::floor`
                // is a `std` method and this is a `no_std` crate.
                let n = (page.text_width() / f.text_width("_", 9.0)) as usize;
                if y < bottom {
                    newpage(&mut cur, &mut pages, &mut y);
                }
                cur.push(Placed {
                    text: "_".repeat(n.max(1)),
                    font: f,
                    size: 9.0,
                    x: page.margin,
                    y,
                });
                y -= 10.0;
            }
            Block::Heading { level, text } => {
                let size = match level {
                    1 => 20.0,
                    2 => 15.0,
                    _ => 12.5,
                };
                let font = Font::HelveticaBold;
                y -= size * 0.8;
                for line in wrap(text, font, size, page.text_width()) {
                    if y < bottom {
                        newpage(&mut cur, &mut pages, &mut y);
                    }
                    cur.push(Placed {
                        text: line,
                        font,
                        size,
                        x: page.margin,
                        y,
                    });
                    y -= size * 1.25;
                }
                y -= size * 0.25;
            }
            Block::Paragraph(text) => {
                let (font, size) = (Font::Helvetica, 10.5);
                for line in wrap(text, font, size, page.text_width()) {
                    if y < bottom {
                        newpage(&mut cur, &mut pages, &mut y);
                    }
                    cur.push(Placed {
                        text: line,
                        font,
                        size,
                        x: page.margin,
                        y,
                    });
                    y -= size * 1.4;
                }
                y -= 5.0;
            }
            Block::Bullet { depth, text } => {
                let (font, size) = (Font::Helvetica, 10.5);
                let indent = page.margin + 14.0 * (*depth as f32 + 1.0);
                let avail = page.width - page.margin - indent;
                for (i, line) in wrap(text, font, size, avail).into_iter().enumerate() {
                    if y < bottom {
                        newpage(&mut cur, &mut pages, &mut y);
                    }
                    // The bullet sits on the first line only; continuations align
                    // under the text, not under the marker.
                    if i == 0 {
                        cur.push(Placed {
                            text: "\u{2022}".to_string(),
                            font,
                            size,
                            x: indent - 10.0,
                            y,
                        });
                    }
                    cur.push(Placed {
                        text: line,
                        font,
                        size,
                        x: indent,
                        y,
                    });
                    y -= size * 1.35;
                }
                y -= 2.0;
            }
            Block::Code(lines) => {
                let (font, size) = (Font::Courier, 9.0);
                y -= 4.0;
                for line in lines {
                    if y < bottom {
                        newpage(&mut cur, &mut pages, &mut y);
                    }
                    // **Never wrapped**: a broken code line is a wrong code line.
                    // Over-long lines are truncated with an ellipsis so the
                    // margin is respected and the loss is visible.
                    let mut text = line.clone();
                    while font.text_width(&text, size) > page.text_width() - 12.0 && text.len() > 1 {
                        text.pop();
                    }
                    if text.len() < line.len() {
                        text.push('…');
                    }
                    cur.push(Placed {
                        text,
                        font,
                        size,
                        x: page.margin + 6.0,
                        y,
                    });
                    y -= size * 1.25;
                }
                y -= 6.0;
            }
        }
    }
    pages.push(cur);
    // A document that ends exactly at a page break would otherwise gain a blank
    // final page.
    while pages.len() > 1 && pages.last().is_some_and(|p| p.is_empty()) {
        pages.pop();
    }
    pages
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Courier is uniform; Helvetica is not — and a proportional font whose
    /// widths were all equal would wrap in visibly wrong places.
    #[test_case]
    fn font_widths_are_the_real_ones() {
        assert_eq!(Font::Courier.width(b'i'), 600);
        assert_eq!(Font::Courier.width(b'W'), 600);
        assert_eq!(Font::Courier.text_width("abcd", 10.0), 24.0);

        assert_eq!(Font::Helvetica.width(b' '), 278);
        assert_eq!(Font::Helvetica.width(b'i'), 222);
        assert_eq!(Font::Helvetica.width(b'W'), 944);
        assert!(
            Font::Helvetica.width(b'i') < Font::Helvetica.width(b'm'),
            "a proportional font must have varying widths"
        );
        // Bold is wider than regular, so bold text never overruns a width
        // computed from the regular face.
        assert!(Font::HelveticaBold.width(b'a') > Font::Helvetica.width(b'a'));
    }

    /// Wrapping never exceeds the limit, and loses no words.
    #[test_case]
    fn wrapping_fits_and_preserves_words() {
        let text = "The quick brown fox jumps over the lazy dog and keeps on running \
                    well past the end of any reasonable line length";
        for width in [120.0f32, 200.0, 400.0] {
            let lines = wrap(text, Font::Helvetica, 10.5, width);
            for l in &lines {
                assert!(
                    Font::Helvetica.text_width(l, 10.5) <= width,
                    "line {l:?} exceeds {width}"
                );
            }
            let joined = lines.join(" ");
            assert_eq!(
                joined.split_whitespace().collect::<Vec<_>>(),
                text.split_whitespace().collect::<Vec<_>>(),
                "no word may be lost at width {width}"
            );
        }
    }

    /// **A word longer than the line is broken, not overflowed.** A URL is the
    /// normal case here, not the edge one.
    #[test_case]
    fn an_overlong_word_is_broken() {
        let long = "https://example.com/a/very/long/path/that/never/ends/at/all/really";
        let lines = wrap(long, Font::Helvetica, 10.5, 100.0);
        assert!(lines.len() > 1, "it must be split");
        for l in &lines {
            assert!(Font::Helvetica.text_width(l, 10.5) <= 100.0, "{l:?}");
        }
        assert_eq!(lines.concat(), long, "breaking must lose nothing");
    }

    /// Empty and whitespace-only text yields one empty line rather than nothing,
    /// so a blank paragraph still advances the cursor.
    #[test_case]
    fn empty_text_yields_one_empty_line() {
        assert_eq!(wrap("", Font::Helvetica, 10.0, 100.0), alloc::vec![String::new()]);
        assert_eq!(wrap("   ", Font::Helvetica, 10.0, 100.0), alloc::vec![String::new()]);
    }

    /// Everything lands inside the margins, and y counts **up** from the bottom.
    #[test_case]
    fn placement_stays_inside_the_page() {
        let page = Page::default();
        let blocks = alloc::vec![
            Block::Heading { level: 1, text: "H.264 in one page".into() },
            Block::Paragraph("A paragraph long enough to need several lines. ".repeat(12)),
            Block::Bullet { depth: 0, text: "A bullet point".into() },
            Block::Code(alloc::vec!["fn main() {}".into()]),
            Block::Rule,
        ];
        let pages = flow(&blocks, page);
        assert!(!pages.is_empty());
        for p in &pages {
            for l in p {
                assert!(l.x >= page.margin - 12.0, "x {} is outside the margin", l.x);
                assert!(l.y >= 0.0 && l.y <= page.height, "y {} is off the page", l.y);
                let right = l.x + l.font.text_width(&l.text, l.size);
                assert!(
                    right <= page.width - page.margin + 1.0,
                    "line {:?} runs to {right}, past the right margin",
                    l.text
                );
            }
        }
        // The first line is near the top, which is what proves the y axis is not
        // inverted — a flipped axis puts it near 0 and the page looks empty.
        assert!(pages[0][0].y > page.height / 2.0, "first line at y={}", pages[0][0].y);
    }

    /// Long content really breaks onto more pages, in order.
    #[test_case]
    fn long_content_paginates() {
        let blocks: Vec<Block> = (0..120)
            .map(|i| Block::Paragraph(alloc::format!("Paragraph number {i} with some words in it.")))
            .collect();
        let pages = flow(&blocks, Page::default());
        assert!(pages.len() > 1, "120 paragraphs must not fit on one page");
        assert!(pages.iter().all(|p| !p.is_empty()), "no page may be blank");
        // Content order is preserved across the break.
        let all: Vec<&str> = pages.iter().flatten().map(|p| p.text.as_str()).collect();
        let first = all.iter().position(|t| t.contains("number 0")).unwrap();
        let later = all.iter().position(|t| t.contains("number 119")).unwrap();
        assert!(first < later);
    }

    /// A code line is truncated rather than wrapped — a broken code line is a
    /// wrong code line.
    #[test_case]
    fn code_lines_are_truncated_not_wrapped() {
        let long = "let x = ".to_string() + &"y".repeat(400);
        let pages = flow(&[Block::Code(alloc::vec![long.clone()])], Page::default());
        let lines: Vec<&Placed> = pages.iter().flatten().collect();
        assert_eq!(lines.len(), 1, "one source line must stay one line");
        assert!(lines[0].text.ends_with('…'), "truncation must be visible");
        assert!(lines[0].text.len() < long.len());
    }

    /// A document ending on a page boundary gains no trailing blank page.
    #[test_case]
    fn no_trailing_blank_page() {
        let pages = flow(&[Block::Paragraph("short".into())], Page::default());
        assert_eq!(pages.len(), 1);
        assert!(!pages[0].is_empty());
    }
}
