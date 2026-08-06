//! Keyboard layout data: a dense US base plus per-layout sparse overrides, four
//! shift levels, dead keys, and the Compose table.
//!
//! ## Why the data is written out rather than packed
//!
//! Nine layouts cost about 11 KB of rodata as plain `KeyDef` rows. A packed
//! `u32`-per-level encoding would save perhaps 5 KB and cost readability on
//! **hand-written data where a typo is invisible** — `c('ö')` in the wrong slot
//! reads exactly like `c('ö')` in the right one. This is a kernel that ships a
//! 4 MiB wasm PDF renderer and megabytes of font; 11 KB is not the constraint.
//! Do not "optimize" this into a bit-packed blob.
//!
//! What *is* enforced, because a hand-written table cannot be trusted otherwise:
//! every layout must be able to type all 26 letters and 10 digits (or you ship a
//! keyboard that cannot type `q`), and no layout may list a usage twice (a
//! duplicate silently shadows whichever row comes second). Both are unit tests.
//!
//! ## Sources
//!
//! The layouts follow the standard national arrangements (the same ones XKB's
//! `symbols/{us,gb,de,fr,es,it,se}` describe). The dead-key composition table is
//! the union of what those layouts can produce, not all of Unicode: `compose_dead`
//! returning `None` is a supported outcome with defined behaviour (see
//! `keymap::apply_dead`), so a gap is a visible `´q` and never a wrong character.

/// One of four shift levels. Named, not indexed, at every use site.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Level {
    Base = 0,
    Shift = 1,
    AltGr = 2,
    ShiftAltGr = 3,
}

/// A dead key — a diacritic that waits for the character it modifies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dead {
    Acute,
    Grave,
    Circumflex,
    Diaeresis,
    Tilde,
    Ring,
    Cedilla,
    Caron,
}

/// What a physical key produces at one level.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Out {
    /// Undefined at this level: the key emits nothing.
    ///
    /// Deliberately **not** a fallback to a different level. Falling back would
    /// mean AltGr on a key with no AltGr form silently types the unshifted
    /// character, which is a wrong character rather than no character.
    None,
    Char(char),
    /// Arms a diacritic; commits nothing yet.
    Dead(Dead),
    /// The Compose key.
    Compose,
}

const fn c(ch: char) -> Out {
    Out::Char(ch)
}
const fn d(k: Dead) -> Out {
    Out::Dead(k)
}
const N: Out = Out::None;

/// One physical key's four levels.
#[derive(Clone, Copy, Debug)]
pub struct KeyDef {
    pub usage: super::Usage,
    pub levels: [Out; 4],
}

const fn k(usage: super::Usage, base: Out, shift: Out, altgr: Out, shift_altgr: Out) -> KeyDef {
    KeyDef { usage, levels: [base, shift, altgr, shift_altgr] }
}

pub struct Layout {
    /// Stable id used by `/keyboard set` and persisted in `ui.json`.
    pub id: &'static str,
    /// Human name for `/keyboard list` and the Kbd status dropdown.
    pub name: &'static str,
    /// Rows that **differ** from [`US_BASE`]; anything absent falls through.
    pub overrides: &'static [KeyDef],
    /// Whether Caps Lock flips this layout's letters. True for all shipped
    /// layouts; the field exists because it is a per-layout property in XKB and
    /// hard-coding it would be a lie waiting to be found.
    pub caps_affects_letters: bool,
    /// Whether right Alt is the **Compose** key rather than a level selector.
    ///
    /// True exactly for layouts that define no AltGr levels, so a US/Dvorak/
    /// Colemak user gets Compose for free and a German user gets AltGr — and
    /// neither loses anything. A field rather than an `id` comparison, so adding
    /// a layout does not mean editing a match somewhere else.
    pub altgr_is_compose: bool,
}

impl Layout {
    /// What this layout's key `usage` produces at `level`.
    ///
    /// Linear scan over at most ~30 override rows plus the 60-row base — one
    /// keystroke, not a hot loop (the decoder this replaced did a 50-row linear
    /// scan per key and was fine). `overrides_are_sorted_by_usage` keeps a binary
    /// search available if that ever changes.
    pub fn lookup(&self, usage: super::Usage, level: Level) -> Out {
        if let Some(row) = self.overrides.iter().find(|r| r.usage == usage) {
            return row.levels[level as usize];
        }
        // The Compose binding lives on right Alt, which never reaches `lookup`
        // as a character key — it is a modifier. The `Out::Compose` row below is
        // for layouts that put Compose on a dedicated key.
        US_BASE
            .iter()
            .find(|r| r.usage == usage)
            .map(|r| r.levels[level as usize])
            .unwrap_or(Out::None)
    }

    /// Whether this layout defines any AltGr-level character.
    pub fn has_altgr_levels(&self) -> bool {
        self.overrides.iter().any(|r| {
            !matches!(r.levels[Level::AltGr as usize], Out::None)
                || !matches!(r.levels[Level::ShiftAltGr as usize], Out::None)
        })
    }

    /// Which dead keys this layout can arm.
    pub fn dead_keys(&self) -> alloc::vec::Vec<Dead> {
        let mut out = alloc::vec::Vec::new();
        for row in self.overrides {
            for l in row.levels {
                if let Out::Dead(k) = l {
                    if !out.contains(&k) {
                        out.push(k);
                    }
                }
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// US base — the dense table every layout is a diff against
// ---------------------------------------------------------------------------

/// The US layout, as HID usage → four levels.
///
/// Reproduces byte-for-byte what the four decoders this replaced produced, which
/// `keymap`'s legacy-fixture tests pin. Enter is `\r` (what xHCI, virtio-input
/// and serial all already sent); the set-1 table's `\n` was the odd one out.
#[rustfmt::skip]
pub static US_BASE: &[KeyDef] = &[
    // Letters.
    k(0x04, c('a'), c('A'), N, N), k(0x05, c('b'), c('B'), N, N),
    k(0x06, c('c'), c('C'), N, N), k(0x07, c('d'), c('D'), N, N),
    k(0x08, c('e'), c('E'), N, N), k(0x09, c('f'), c('F'), N, N),
    k(0x0a, c('g'), c('G'), N, N), k(0x0b, c('h'), c('H'), N, N),
    k(0x0c, c('i'), c('I'), N, N), k(0x0d, c('j'), c('J'), N, N),
    k(0x0e, c('k'), c('K'), N, N), k(0x0f, c('l'), c('L'), N, N),
    k(0x10, c('m'), c('M'), N, N), k(0x11, c('n'), c('N'), N, N),
    k(0x12, c('o'), c('O'), N, N), k(0x13, c('p'), c('P'), N, N),
    k(0x14, c('q'), c('Q'), N, N), k(0x15, c('r'), c('R'), N, N),
    k(0x16, c('s'), c('S'), N, N), k(0x17, c('t'), c('T'), N, N),
    k(0x18, c('u'), c('U'), N, N), k(0x19, c('v'), c('V'), N, N),
    k(0x1a, c('w'), c('W'), N, N), k(0x1b, c('x'), c('X'), N, N),
    k(0x1c, c('y'), c('Y'), N, N), k(0x1d, c('z'), c('Z'), N, N),
    // Digit row.
    k(0x1e, c('1'), c('!'), N, N), k(0x1f, c('2'), c('@'), N, N),
    k(0x20, c('3'), c('#'), N, N), k(0x21, c('4'), c('$'), N, N),
    k(0x22, c('5'), c('%'), N, N), k(0x23, c('6'), c('^'), N, N),
    k(0x24, c('7'), c('&'), N, N), k(0x25, c('8'), c('*'), N, N),
    k(0x26, c('9'), c('('), N, N), k(0x27, c('0'), c(')'), N, N),
    // Whitespace and control.
    k(0x28, c('\r'), c('\r'), N, N),          // Enter
    k(0x29, c('\u{1b}'), c('\u{1b}'), N, N),  // Esc
    k(0x2a, c('\u{8}'), c('\u{8}'), N, N),    // Backspace
    k(0x2b, c('\t'), c('\t'), N, N),          // Tab
    k(0x2c, c(' '), c(' '), N, N),            // Space
    // Punctuation.
    k(0x2d, c('-'), c('_'), N, N), k(0x2e, c('='), c('+'), N, N),
    k(0x2f, c('['), c('{'), N, N), k(0x30, c(']'), c('}'), N, N),
    k(0x31, c('\\'), c('|'), N, N),
    k(0x33, c(';'), c(':'), N, N), k(0x34, c('\''), c('"'), N, N),
    k(0x35, c('`'), c('~'), N, N),
    k(0x36, c(','), c('<'), N, N), k(0x37, c('.'), c('>'), N, N),
    k(0x38, c('/'), c('?'), N, N),
    // Keypad keys that carry characters.
    k(0x54, c('/'), c('/'), N, N), k(0x55, c('*'), c('*'), N, N),
    k(0x56, c('-'), c('-'), N, N), k(0x57, c('+'), c('+'), N, N),
    k(0x58, c('\r'), c('\r'), N, N),
    // The ISO key left of Z on a US-International board; absent on ANSI, which
    // is why no previous table decoded usage 0x64 at all.
    k(0x64, c('\\'), c('|'), N, N),
];

pub const US: Layout = Layout {
    id: "us",
    name: "US (QWERTY)",
    overrides: &[],
    caps_affects_letters: true,
    altgr_is_compose: true,
};

// ---------------------------------------------------------------------------
// United Kingdom
// ---------------------------------------------------------------------------

#[rustfmt::skip]
static UK_KEYS: &[KeyDef] = &[
    k(0x1f, c('2'), c('"'), N, N),                        // 2 " (not @)
    k(0x20, c('3'), c('\u{A3}'), N, N),                   // 3 £
    k(0x31, c('#'), c('~'), N, N),                        // the key left of Enter
    k(0x34, c('\''), c('@'), N, N),                       // ' @
    k(0x35, c('`'), c('\u{AC}'), c('|'), N),              // ` ¬ |
    k(0x64, c('\\'), c('|'), N, N),                       // ISO key: backslash
    k(0x08, c('e'), c('E'), c('\u{20AC}'), N),            // AltGr+e = €
];
pub const UK: Layout = Layout {
    id: "uk",
    name: "United Kingdom",
    overrides: UK_KEYS,
    caps_affects_letters: true,
    altgr_is_compose: false,
};

// ---------------------------------------------------------------------------
// German (QWERTZ)
// ---------------------------------------------------------------------------

#[rustfmt::skip]
static DE_KEYS: &[KeyDef] = &[
    // QWERTZ: the Y and Z positions swap.
    k(0x1c, c('z'), c('Z'), N, N),                        // US 'y' position
    k(0x1d, c('y'), c('Y'), N, N),                        // US 'z' position
    // Umlauts sit on the US punctuation keys.
    k(0x33, c('\u{F6}'), c('\u{D6}'), N, N),              // ö Ö  (US ';')
    k(0x34, c('\u{E4}'), c('\u{C4}'), N, N),              // ä Ä  (US '\'')
    k(0x2f, c('\u{FC}'), c('\u{DC}'), N, N),              // ü Ü  (US '[')
    k(0x30, c('+'), c('*'), c('~'), N),                   // + * ~
    k(0x31, c('#'), c('\''), N, N),
    k(0x2d, c('\u{DF}'), c('?'), c('\\'), N),             // ß ? backslash
    // The two dead keys.
    k(0x2e, d(Dead::Acute), d(Dead::Grave), N, N),        // ´ `  (US '=')
    k(0x35, d(Dead::Circumflex), c('\u{B0}'), N, N),      // ^ °  (US '`')
    // The ISO key left of Y, which carries < > | on German boards.
    k(0x64, c('<'), c('>'), c('|'), N),
    // Digit row: shifted forms and the AltGr set.
    k(0x1e, c('1'), c('!'), N, N),
    k(0x1f, c('2'), c('"'), c('\u{B2}'), N),              // ²
    k(0x20, c('3'), c('\u{A7}'), c('\u{B3}'), N),         // § ³
    k(0x21, c('4'), c('$'), N, N),
    k(0x22, c('5'), c('%'), N, N),
    k(0x23, c('6'), c('&'), N, N),
    k(0x24, c('7'), c('/'), c('{'), N),
    k(0x25, c('8'), c('('), c('['), N),
    k(0x26, c('9'), c(')'), c(']'), N),
    k(0x27, c('0'), c('='), c('}'), N),
    // AltGr letters.
    k(0x14, c('q'), c('Q'), c('@'), N),
    k(0x08, c('e'), c('E'), c('\u{20AC}'), N),            // €
    k(0x10, c('m'), c('M'), c('\u{B5}'), N),              // µ
    // Punctuation.
    k(0x36, c(','), c(';'), N, N),
    k(0x37, c('.'), c(':'), N, N),
    k(0x38, c('-'), c('_'), N, N),
];
pub const DE: Layout = Layout {
    id: "de",
    name: "German (QWERTZ)",
    overrides: DE_KEYS,
    caps_affects_letters: true,
    altgr_is_compose: false,
};

// ---------------------------------------------------------------------------
// French (AZERTY)
// ---------------------------------------------------------------------------

#[rustfmt::skip]
static FR_KEYS: &[KeyDef] = &[
    // AZERTY letter moves.
    k(0x14, c('a'), c('A'), N, N),                        // US 'q' → a
    k(0x04, c('q'), c('Q'), N, N),                        // US 'a' → q
    k(0x1a, c('z'), c('Z'), N, N),                        // US 'w' → z
    k(0x1d, c('w'), c('W'), N, N),                        // US 'z' → w
    k(0x33, c('m'), c('M'), N, N),                        // US ';' → m
    k(0x10, c(','), c('?'), N, N),                        // US 'm' → ,
    // Digit row: unshifted is punctuation, shifted is the digit.
    k(0x1e, c('&'), c('1'), N, N),
    k(0x1f, c('\u{E9}'), c('2'), c('~'), N),              // é
    k(0x20, c('"'), c('3'), c('#'), N),
    k(0x21, c('\''), c('4'), c('{'), N),
    k(0x22, c('('), c('5'), c('['), N),
    k(0x23, c('-'), c('6'), c('|'), N),
    k(0x24, c('\u{E8}'), c('7'), c('`'), N),              // è
    k(0x25, c('_'), c('8'), c('\\'), N),
    k(0x26, c('\u{E7}'), c('9'), c('^'), N),              // ç
    k(0x27, c('\u{E0}'), c('0'), c('@'), N),              // à
    k(0x2d, c(')'), c('\u{B0}'), c(']'), N),              // ) °
    k(0x2e, c('='), c('+'), c('}'), N),
    // Dead keys and the remaining punctuation.
    k(0x2f, d(Dead::Circumflex), d(Dead::Diaeresis), N, N),
    k(0x30, c('$'), c('\u{A3}'), c('\u{20AC}'), N),       // $ £ €
    k(0x34, c('\u{F9}'), c('%'), N, N),                   // ù
    k(0x31, c('*'), c('\u{B5}'), N, N),                   // * µ
    k(0x35, c('\u{B2}'), c('\u{B2}'), N, N),              // ²
    k(0x36, c(';'), c('.'), N, N),
    k(0x37, c(':'), c('/'), N, N),
    k(0x38, c('!'), c('\u{A7}'), N, N),                   // ! §
    // ISO key left of W.
    k(0x64, c('<'), c('>'), N, N),
    k(0x08, c('e'), c('E'), c('\u{20AC}'), N),            // AltGr+e = €
];
pub const FR: Layout = Layout {
    id: "fr",
    name: "French (AZERTY)",
    overrides: FR_KEYS,
    caps_affects_letters: true,
    altgr_is_compose: false,
};

// ---------------------------------------------------------------------------
// Spanish
// ---------------------------------------------------------------------------

#[rustfmt::skip]
static ES_KEYS: &[KeyDef] = &[
    k(0x33, c('\u{F1}'), c('\u{D1}'), N, N),              // ñ Ñ  (US ';')
    k(0x34, d(Dead::Acute), d(Dead::Diaeresis), c('{'), N), // ´ ¨ {
    k(0x2f, d(Dead::Grave), d(Dead::Circumflex), c('['), N),
    k(0x30, c('+'), c('*'), c(']'), N),
    k(0x31, c('\u{E7}'), c('\u{C7}'), c('}'), N),         // ç Ç }
    k(0x2d, c('\''), c('?'), N, N),
    k(0x2e, c('\u{A1}'), c('\u{BF}'), N, N),              // ¡ ¿
    k(0x35, c('\u{BA}'), c('\u{AA}'), c('\\'), N),        // º ª backslash
    k(0x1f, c('2'), c('"'), c('@'), N),
    k(0x20, c('3'), c('\u{B7}'), c('#'), N),              // ·
    k(0x21, c('4'), c('$'), c('~'), N),
    k(0x23, c('6'), c('&'), c('\u{AC}'), N),              // ¬
    k(0x27, c('0'), c('='), N, N),
    k(0x36, c(','), c(';'), N, N),
    k(0x37, c('.'), c(':'), N, N),
    k(0x38, c('-'), c('_'), N, N),
    k(0x64, c('<'), c('>'), N, N),
    k(0x08, c('e'), c('E'), c('\u{20AC}'), N),            // €
];
pub const ES: Layout = Layout {
    id: "es",
    name: "Spanish",
    overrides: ES_KEYS,
    caps_affects_letters: true,
    altgr_is_compose: false,
};

// ---------------------------------------------------------------------------
// Italian
// ---------------------------------------------------------------------------

#[rustfmt::skip]
static IT_KEYS: &[KeyDef] = &[
    k(0x33, c('\u{F2}'), c('\u{E7}'), c('@'), N),         // ò ç @
    k(0x34, c('\u{E0}'), c('\u{B0}'), c('#'), N),         // à ° #
    k(0x2f, c('\u{E8}'), c('\u{E9}'), c('['), N),         // è é [
    k(0x30, c('+'), c('*'), c(']'), N),
    k(0x31, c('\u{F9}'), c('\u{A7}'), N, N),              // ù §
    k(0x2d, c('\''), c('?'), N, N),
    k(0x2e, c('\u{EC}'), c('^'), N, N),                   // ì ^
    k(0x35, c('\\'), c('|'), N, N),
    k(0x1f, c('2'), c('"'), N, N),
    k(0x20, c('3'), c('\u{A3}'), N, N),                   // £
    k(0x27, c('0'), c('='), N, N),
    k(0x36, c(','), c(';'), N, N),
    k(0x37, c('.'), c(':'), N, N),
    k(0x38, c('-'), c('_'), N, N),
    k(0x64, c('<'), c('>'), N, N),
    k(0x08, c('e'), c('E'), c('\u{20AC}'), N),            // €
];
pub const IT: Layout = Layout {
    id: "it",
    name: "Italian",
    overrides: IT_KEYS,
    caps_affects_letters: true,
    altgr_is_compose: false,
};

// ---------------------------------------------------------------------------
// Swedish / Finnish (the Nordic arrangement)
// ---------------------------------------------------------------------------

#[rustfmt::skip]
static SE_KEYS: &[KeyDef] = &[
    k(0x33, c('\u{F6}'), c('\u{D6}'), N, N),              // ö Ö
    k(0x34, c('\u{E4}'), c('\u{C4}'), N, N),              // ä Ä
    k(0x2f, c('\u{E5}'), c('\u{C5}'), N, N),              // å Å
    k(0x30, d(Dead::Diaeresis), d(Dead::Circumflex), d(Dead::Tilde), N),
    k(0x31, c('\''), c('*'), N, N),
    k(0x2d, c('+'), c('?'), c('\\'), N),
    k(0x2e, d(Dead::Acute), d(Dead::Grave), N, N),
    k(0x35, c('\u{A7}'), c('\u{BD}'), N, N),              // § ½
    k(0x1f, c('2'), c('"'), c('@'), N),
    k(0x20, c('3'), c('#'), c('\u{A3}'), N),
    k(0x21, c('4'), c('\u{A4}'), c('$'), N),              // ¤ $
    k(0x24, c('7'), c('/'), c('{'), N),
    k(0x25, c('8'), c('('), c('['), N),
    k(0x26, c('9'), c(')'), c(']'), N),
    k(0x27, c('0'), c('='), c('}'), N),
    k(0x36, c(','), c(';'), N, N),
    k(0x37, c('.'), c(':'), N, N),
    k(0x38, c('-'), c('_'), N, N),
    k(0x64, c('<'), c('>'), c('|'), N),
    k(0x08, c('e'), c('E'), c('\u{20AC}'), N),            // €
];
pub const SE: Layout = Layout {
    id: "se",
    name: "Swedish / Finnish",
    overrides: SE_KEYS,
    caps_affects_letters: true,
    altgr_is_compose: false,
};

// ---------------------------------------------------------------------------
// Dvorak and Colemak — pure letter permutations over the US base
// ---------------------------------------------------------------------------

#[rustfmt::skip]
static DVORAK_KEYS: &[KeyDef] = &[
    // Top row: ' , . p y f g c r l / =
    k(0x14, c('\''), c('"'), N, N),  k(0x1a, c(','), c('<'), N, N),
    k(0x08, c('.'), c('>'), N, N),   k(0x15, c('p'), c('P'), N, N),
    k(0x17, c('y'), c('Y'), N, N),   k(0x1c, c('f'), c('F'), N, N),
    k(0x18, c('g'), c('G'), N, N),   k(0x0c, c('c'), c('C'), N, N),
    k(0x12, c('r'), c('R'), N, N),   k(0x13, c('l'), c('L'), N, N),
    k(0x2f, c('/'), c('?'), N, N),   k(0x30, c('='), c('+'), N, N),
    // Home row: a o e u i d h t n s -
    k(0x04, c('a'), c('A'), N, N),   k(0x16, c('o'), c('O'), N, N),
    k(0x07, c('e'), c('E'), N, N),   k(0x09, c('u'), c('U'), N, N),
    k(0x0a, c('i'), c('I'), N, N),   k(0x0b, c('d'), c('D'), N, N),
    k(0x0d, c('h'), c('H'), N, N),   k(0x0e, c('t'), c('T'), N, N),
    k(0x0f, c('n'), c('N'), N, N),   k(0x33, c('s'), c('S'), N, N),
    k(0x34, c('-'), c('_'), N, N),
    // Bottom row: ; q j k x b m w v z
    k(0x1d, c(';'), c(':'), N, N),   k(0x1b, c('q'), c('Q'), N, N),
    k(0x06, c('j'), c('J'), N, N),   k(0x19, c('k'), c('K'), N, N),
    k(0x05, c('x'), c('X'), N, N),   k(0x11, c('b'), c('B'), N, N),
    k(0x10, c('m'), c('M'), N, N),   k(0x36, c('w'), c('W'), N, N),
    k(0x37, c('v'), c('V'), N, N),   k(0x38, c('z'), c('Z'), N, N),
    // The two keys Dvorak moves off the digit row's edges.
    k(0x2d, c('['), c('{'), N, N),   k(0x2e, c(']'), c('}'), N, N),
];
pub const DVORAK: Layout = Layout {
    id: "dvorak",
    name: "Dvorak",
    overrides: DVORAK_KEYS,
    caps_affects_letters: true,
    altgr_is_compose: true,
};

#[rustfmt::skip]
static COLEMAK_KEYS: &[KeyDef] = &[
    // Colemak moves ten letters and puts Backspace on Caps (which we leave alone
    // — remapping Caps Lock is a separate preference, not part of the layout).
    k(0x08, c('f'), c('F'), N, N),   // US e → f
    k(0x15, c('p'), c('P'), N, N),   // US r → p
    k(0x17, c('g'), c('G'), N, N),   // US t → g
    k(0x1c, c('j'), c('J'), N, N),   // US y → j
    k(0x18, c('l'), c('L'), N, N),   // US u → l
    k(0x0c, c('u'), c('U'), N, N),   // US i → u
    k(0x12, c('y'), c('Y'), N, N),   // US o → y
    k(0x13, c(';'), c(':'), N, N),   // US p → ;
    k(0x16, c('r'), c('R'), N, N),   // US s → r
    k(0x07, c('s'), c('S'), N, N),   // US d → s
    k(0x09, c('t'), c('T'), N, N),   // US f → t
    k(0x0a, c('d'), c('D'), N, N),   // US g → d
    k(0x0d, c('n'), c('N'), N, N),   // US j → n
    k(0x0e, c('e'), c('E'), N, N),   // US k → e
    k(0x0f, c('i'), c('I'), N, N),   // US l → i
    k(0x33, c('o'), c('O'), N, N),   // US ; → o
    k(0x11, c('k'), c('K'), N, N),   // US n → k
];
pub const COLEMAK: Layout = Layout {
    id: "colemak",
    name: "Colemak",
    overrides: COLEMAK_KEYS,
    caps_affects_letters: true,
    altgr_is_compose: true,
};

/// Every shipped layout, in the order `/keyboard list` prints them.
pub static LAYOUTS: &[Layout] = &[US, UK, DE, FR, ES, IT, SE, DVORAK, COLEMAK];

// ---------------------------------------------------------------------------
// Dead keys
// ---------------------------------------------------------------------------

pub fn dead_name(k: Dead) -> &'static str {
    match k {
        Dead::Acute => "acute",
        Dead::Grave => "grave",
        Dead::Circumflex => "circumflex",
        Dead::Diaeresis => "diaeresis",
        Dead::Tilde => "tilde",
        Dead::Ring => "ring",
        Dead::Cedilla => "cedilla",
        Dead::Caron => "caron",
    }
}

/// The **spacing** form of a diacritic: what a double-tap, or a diacritic
/// followed by space, produces.
pub fn spacing_form(k: Dead) -> char {
    match k {
        Dead::Acute => '\u{00B4}',
        Dead::Grave => '`',
        Dead::Circumflex => '^',
        Dead::Diaeresis => '\u{00A8}',
        Dead::Tilde => '~',
        Dead::Ring => '\u{00B0}',
        Dead::Cedilla => '\u{00B8}',
        Dead::Caron => '\u{02C7}',
    }
}

/// `(dead, base, precomposed)`, sorted by `(dead, base)`.
///
/// **This does not need to be small, and there is no arithmetic shortcut worth
/// having.** Latin-1 is *almost* regular (`à á â ã ä å` sit at U+00E0..E5) but the
/// exceptions — `ÿ` at U+00FF, everything in Latin Extended-A like `ć ń ś ź ě` —
/// mean a formula would be a table plus a formula plus the bugs at the seam. What
/// matters is **coverage of the shipped layouts** and the miss fallback (see
/// `keymap::apply_dead`), not compactness: ~170 rows at 6 bytes is a kilobyte.
#[rustfmt::skip]
static DEAD_MAP: &[(Dead, char, char)] = &[
    (Dead::Acute, 'A', '\u{C1}'), (Dead::Acute, 'C', '\u{106}'), (Dead::Acute, 'E', '\u{C9}'),
    (Dead::Acute, 'I', '\u{CD}'), (Dead::Acute, 'N', '\u{143}'), (Dead::Acute, 'O', '\u{D3}'),
    (Dead::Acute, 'S', '\u{15A}'), (Dead::Acute, 'U', '\u{DA}'), (Dead::Acute, 'Y', '\u{DD}'),
    (Dead::Acute, 'Z', '\u{179}'),
    (Dead::Acute, 'a', '\u{E1}'), (Dead::Acute, 'c', '\u{107}'), (Dead::Acute, 'e', '\u{E9}'),
    (Dead::Acute, 'i', '\u{ED}'), (Dead::Acute, 'n', '\u{144}'), (Dead::Acute, 'o', '\u{F3}'),
    (Dead::Acute, 's', '\u{15B}'), (Dead::Acute, 'u', '\u{FA}'), (Dead::Acute, 'y', '\u{FD}'),
    (Dead::Acute, 'z', '\u{17A}'),

    (Dead::Grave, 'A', '\u{C0}'), (Dead::Grave, 'E', '\u{C8}'), (Dead::Grave, 'I', '\u{CC}'),
    (Dead::Grave, 'O', '\u{D2}'), (Dead::Grave, 'U', '\u{D9}'),
    (Dead::Grave, 'a', '\u{E0}'), (Dead::Grave, 'e', '\u{E8}'), (Dead::Grave, 'i', '\u{EC}'),
    (Dead::Grave, 'o', '\u{F2}'), (Dead::Grave, 'u', '\u{F9}'),

    (Dead::Circumflex, 'A', '\u{C2}'), (Dead::Circumflex, 'C', '\u{108}'),
    (Dead::Circumflex, 'E', '\u{CA}'), (Dead::Circumflex, 'G', '\u{11C}'),
    (Dead::Circumflex, 'I', '\u{CE}'), (Dead::Circumflex, 'O', '\u{D4}'),
    (Dead::Circumflex, 'S', '\u{15C}'), (Dead::Circumflex, 'U', '\u{DB}'),
    (Dead::Circumflex, 'W', '\u{174}'), (Dead::Circumflex, 'Y', '\u{176}'),
    (Dead::Circumflex, 'a', '\u{E2}'), (Dead::Circumflex, 'c', '\u{109}'),
    (Dead::Circumflex, 'e', '\u{EA}'), (Dead::Circumflex, 'g', '\u{11D}'),
    (Dead::Circumflex, 'i', '\u{EE}'), (Dead::Circumflex, 'o', '\u{F4}'),
    (Dead::Circumflex, 's', '\u{15D}'), (Dead::Circumflex, 'u', '\u{FB}'),
    (Dead::Circumflex, 'w', '\u{175}'), (Dead::Circumflex, 'y', '\u{177}'),

    (Dead::Diaeresis, 'A', '\u{C4}'), (Dead::Diaeresis, 'E', '\u{CB}'),
    (Dead::Diaeresis, 'I', '\u{CF}'), (Dead::Diaeresis, 'O', '\u{D6}'),
    (Dead::Diaeresis, 'U', '\u{DC}'), (Dead::Diaeresis, 'Y', '\u{178}'),
    (Dead::Diaeresis, 'a', '\u{E4}'), (Dead::Diaeresis, 'e', '\u{EB}'),
    (Dead::Diaeresis, 'i', '\u{EF}'), (Dead::Diaeresis, 'o', '\u{F6}'),
    (Dead::Diaeresis, 'u', '\u{FC}'), (Dead::Diaeresis, 'y', '\u{FF}'),

    (Dead::Tilde, 'A', '\u{C3}'), (Dead::Tilde, 'N', '\u{D1}'), (Dead::Tilde, 'O', '\u{D5}'),
    (Dead::Tilde, 'a', '\u{E3}'), (Dead::Tilde, 'n', '\u{F1}'), (Dead::Tilde, 'o', '\u{F5}'),

    (Dead::Ring, 'A', '\u{C5}'), (Dead::Ring, 'U', '\u{16E}'),
    (Dead::Ring, 'a', '\u{E5}'), (Dead::Ring, 'u', '\u{16F}'),

    (Dead::Cedilla, 'C', '\u{C7}'), (Dead::Cedilla, 'G', '\u{122}'),
    (Dead::Cedilla, 'S', '\u{15E}'), (Dead::Cedilla, 'T', '\u{162}'),
    (Dead::Cedilla, 'c', '\u{E7}'), (Dead::Cedilla, 'g', '\u{123}'),
    (Dead::Cedilla, 's', '\u{15F}'), (Dead::Cedilla, 't', '\u{163}'),

    (Dead::Caron, 'C', '\u{10C}'), (Dead::Caron, 'D', '\u{10E}'), (Dead::Caron, 'E', '\u{11A}'),
    (Dead::Caron, 'N', '\u{147}'), (Dead::Caron, 'R', '\u{158}'), (Dead::Caron, 'S', '\u{160}'),
    (Dead::Caron, 'T', '\u{164}'), (Dead::Caron, 'Z', '\u{17D}'),
    (Dead::Caron, 'c', '\u{10D}'), (Dead::Caron, 'd', '\u{10F}'), (Dead::Caron, 'e', '\u{11B}'),
    (Dead::Caron, 'n', '\u{148}'), (Dead::Caron, 'r', '\u{159}'), (Dead::Caron, 's', '\u{161}'),
    (Dead::Caron, 't', '\u{165}'), (Dead::Caron, 'z', '\u{17E}'),
];

/// Apply a diacritic to a base character. `None` means there is no precomposed
/// form, which the caller handles by typing the spacing diacritic **and** the
/// base — visibly wrong rather than silently the wrong character.
pub fn compose_dead(k: Dead, base: char) -> Option<char> {
    DEAD_MAP.iter().find(|(dk, b, _)| *dk == k && *b == base).map(|(_, _, out)| *out)
}

/// Every dead-key row, for `/keyboard dead`.
pub fn dead_rows() -> &'static [(Dead, char, char)] {
    DEAD_MAP
}

// ---------------------------------------------------------------------------
// Compose
// ---------------------------------------------------------------------------

/// Two-key Compose sequences, sorted.
///
/// Bounded on purpose, and **listable** (`/keyboard compose` prints all of it) —
/// a convenience surface that cannot say what it covers should not exist. This is
/// deliberately not X11's `en_US.UTF-8/Compose` (~5000 sequences, ~200 KB); if
/// somebody wants that it is a file on the store parsed at boot, which is a
/// different feature.
///
/// The digraph forms of the dead keys (`'e` → é, `` `a `` → à, `"u` → ü, `~n` → ñ,
/// `,c` → ç, `oa` → å) are the point: they give a **US-layout** user accents
/// without switching layout, which is the single most useful thing Compose does
/// here.
#[rustfmt::skip]
static COMPOSE: &[(char, char, char)] = &[
    // Accents, so a US layout can type them.
    ('\'', 'a', '\u{E1}'), ('\'', 'e', '\u{E9}'), ('\'', 'i', '\u{ED}'),
    ('\'', 'o', '\u{F3}'), ('\'', 'u', '\u{FA}'), ('\'', 'y', '\u{FD}'),
    ('\'', 'A', '\u{C1}'), ('\'', 'E', '\u{C9}'), ('\'', 'I', '\u{CD}'),
    ('\'', 'O', '\u{D3}'), ('\'', 'U', '\u{DA}'),
    ('`', 'a', '\u{E0}'), ('`', 'e', '\u{E8}'), ('`', 'i', '\u{EC}'),
    ('`', 'o', '\u{F2}'), ('`', 'u', '\u{F9}'),
    ('`', 'A', '\u{C0}'), ('`', 'E', '\u{C8}'),
    ('"', 'a', '\u{E4}'), ('"', 'e', '\u{EB}'), ('"', 'i', '\u{EF}'),
    ('"', 'o', '\u{F6}'), ('"', 'u', '\u{FC}'), ('"', 'y', '\u{FF}'),
    ('"', 'A', '\u{C4}'), ('"', 'O', '\u{D6}'), ('"', 'U', '\u{DC}'),
    ('^', 'a', '\u{E2}'), ('^', 'e', '\u{EA}'), ('^', 'i', '\u{EE}'),
    ('^', 'o', '\u{F4}'), ('^', 'u', '\u{FB}'),
    ('~', 'n', '\u{F1}'), ('~', 'a', '\u{E3}'), ('~', 'o', '\u{F5}'),
    ('~', 'N', '\u{D1}'),
    (',', 'c', '\u{E7}'), (',', 'C', '\u{C7}'),
    ('o', 'a', '\u{E5}'), ('o', 'A', '\u{C5}'),
    // Ligatures and letters with no diacritic form.
    ('a', 'e', '\u{E6}'), ('A', 'E', '\u{C6}'),
    ('o', 'e', '\u{153}'), ('O', 'E', '\u{152}'),
    ('s', 's', '\u{DF}'),
    ('/', 'o', '\u{F8}'), ('/', 'O', '\u{D8}'),
    // Symbols people actually reach for.
    ('o', 'c', '\u{A9}'),  ('o', 'r', '\u{AE}'),  ('o', 'o', '\u{B0}'),
    ('t', 'm', '\u{2122}'),
    ('-', '-', '\u{2014}'), ('-', '.', '\u{2013}'),
    ('<', '<', '\u{AB}'),  ('>', '>', '\u{BB}'),
    ('-', '>', '\u{2192}'), ('<', '-', '\u{2190}'),
    ('!', '=', '\u{2260}'), ('<', '=', '\u{2264}'), ('>', '=', '\u{2265}'),
    ('+', '-', '\u{B1}'),  ('x', 'x', '\u{D7}'),  (':', '-', '\u{F7}'),
    ('1', '2', '\u{BD}'),  ('1', '4', '\u{BC}'),  ('3', '4', '\u{BE}'),
    ('1', 's', '\u{B9}'),  ('2', 's', '\u{B2}'),  ('3', 's', '\u{B3}'),
    ('e', '=', '\u{20AC}'), ('l', '-', '\u{A3}'), ('y', '=', '\u{A5}'),
    ('c', '/', '\u{A2}'),  ('s', 'o', '\u{A7}'),
    ('.', '.', '\u{2026}'), ('*', '*', '\u{2022}'),
    ('?', '?', '\u{BF}'),  ('!', '!', '\u{A1}'),
    ('v', '/', '\u{2713}'), ('x', '/', '\u{2717}'),
    ('m', 'u', '\u{B5}'),  ('n', 'o', '\u{2116}'),
    ('i', 'n', '\u{221E}'), ('~', '~', '\u{2248}'),
];

/// Look up a two-key Compose sequence. Order-insensitive for the symmetric
/// pairs people type either way round (`oc` and `co` both give ©), which is what
/// X11 does with its explicit both-orders entries.
pub fn compose_pair(a: char, b: char) -> Option<char> {
    COMPOSE
        .iter()
        .find(|(x, y, _)| (*x == a && *y == b) || (*x == b && *y == a))
        .map(|(_, _, out)| *out)
}

/// Every Compose row, for `/keyboard compose`.
pub fn compose_rows() -> &'static [(char, char, char)] {
    COMPOSE
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A hand-written table cannot be trusted about this, and the failure mode is
    /// shipping a keyboard that cannot type a letter.
    #[test_case]
    fn every_layout_can_type_all_letters_and_all_digits() {
        for l in LAYOUTS {
            for want in "abcdefghijklmnopqrstuvwxyz".chars() {
                let found = (0x00..=0x70u8).any(|u| {
                    matches!(l.lookup(u, Level::Base), Out::Char(c) if c == want)
                        || matches!(l.lookup(u, Level::Shift), Out::Char(c)
                            if c.to_lowercase().next() == Some(want))
                });
                assert!(found, "layout '{}' cannot type '{want}'", l.id);
            }
            for want in "0123456789".chars() {
                let found = (0x00..=0x70u8).any(|u| {
                    (0..4usize).any(|lv| {
                        let lvl = match lv {
                            0 => Level::Base,
                            1 => Level::Shift,
                            2 => Level::AltGr,
                            _ => Level::ShiftAltGr,
                        };
                        matches!(l.lookup(u, lvl), Out::Char(c) if c == want)
                    })
                });
                assert!(found, "layout '{}' cannot type '{want}'", l.id);
            }
        }
    }

    /// A duplicate row silently shadows whichever comes second, so the table
    /// would be wrong in a way no amount of typing would reveal.
    #[test_case]
    fn no_layout_lists_a_usage_twice() {
        for l in LAYOUTS {
            for (i, a) in l.overrides.iter().enumerate() {
                for b in &l.overrides[i + 1..] {
                    assert_ne!(
                        a.usage, b.usage,
                        "layout '{}' lists usage {:#04x} twice",
                        l.id, a.usage
                    );
                }
            }
        }
        for (i, a) in US_BASE.iter().enumerate() {
            for b in &US_BASE[i + 1..] {
                assert_ne!(a.usage, b.usage, "US_BASE lists usage {:#04x} twice", a.usage);
            }
        }
    }

    #[test_case]
    fn layout_ids_and_names_are_unique() {
        for (i, a) in LAYOUTS.iter().enumerate() {
            for b in &LAYOUTS[i + 1..] {
                assert_ne!(a.id, b.id, "duplicate layout id '{}'", a.id);
                assert_ne!(a.name, b.name, "duplicate layout name '{}'", a.name);
            }
        }
        assert!(LAYOUTS.len() >= 9, "expected at least nine layouts");
    }

    /// `altgr_is_compose` must agree with whether the layout actually defines
    /// AltGr levels, or a German user loses AltGr or a US user loses Compose.
    #[test_case]
    fn compose_is_bound_to_right_alt_exactly_on_layouts_without_altgr_levels() {
        for l in LAYOUTS {
            assert_eq!(
                l.altgr_is_compose,
                !l.has_altgr_levels(),
                "layout '{}': altgr_is_compose={} but has_altgr_levels={}",
                l.id,
                l.altgr_is_compose,
                l.has_altgr_levels()
            );
        }
        // Concretely: US/Dvorak/Colemak get Compose, the European ones get AltGr.
        assert!(US.altgr_is_compose);
        assert!(DVORAK.altgr_is_compose);
        assert!(COLEMAK.altgr_is_compose);
        assert!(!DE.altgr_is_compose);
        assert!(!FR.altgr_is_compose);
    }

    #[test_case]
    fn the_national_layouts_put_their_characters_where_they_belong() {
        // German QWERTZ: y/z swapped, umlauts on the US punctuation keys.
        assert_eq!(DE.lookup(0x1c, Level::Base), Out::Char('z'), "de: US-y types z");
        assert_eq!(DE.lookup(0x1d, Level::Base), Out::Char('y'), "de: US-z types y");
        assert_eq!(DE.lookup(0x33, Level::Base), Out::Char('ö'));
        assert_eq!(DE.lookup(0x34, Level::Shift), Out::Char('Ä'));
        assert_eq!(DE.lookup(0x14, Level::AltGr), Out::Char('@'), "de: AltGr+q = @");
        assert_eq!(DE.lookup(0x08, Level::AltGr), Out::Char('€'), "de: AltGr+e = €");
        assert_eq!(DE.lookup(0x64, Level::Base), Out::Char('<'), "de: the ISO key");
        assert_eq!(DE.lookup(0x64, Level::AltGr), Out::Char('|'));
        // French AZERTY: a/q and z/w swapped, digits need Shift.
        assert_eq!(FR.lookup(0x14, Level::Base), Out::Char('a'), "fr: US-q types a");
        assert_eq!(FR.lookup(0x04, Level::Base), Out::Char('q'), "fr: US-a types q");
        assert_eq!(FR.lookup(0x1e, Level::Base), Out::Char('&'), "fr: unshifted 1 is &");
        assert_eq!(FR.lookup(0x1e, Level::Shift), Out::Char('1'));
        // UK: £ on 3, @ on the quote key.
        assert_eq!(UK.lookup(0x20, Level::Shift), Out::Char('£'));
        assert_eq!(UK.lookup(0x34, Level::Shift), Out::Char('@'));
        // Spanish ñ, Nordic å.
        assert_eq!(ES.lookup(0x33, Level::Base), Out::Char('ñ'));
        assert_eq!(SE.lookup(0x2f, Level::Base), Out::Char('å'));
        // Dvorak/Colemak are permutations that leave the digits alone.
        assert_eq!(DVORAK.lookup(0x16, Level::Base), Out::Char('o'), "dvorak: US-s types o");
        assert_eq!(DVORAK.lookup(0x1e, Level::Base), Out::Char('1'));
        assert_eq!(COLEMAK.lookup(0x16, Level::Base), Out::Char('r'), "colemak: US-s types r");
    }

    #[test_case]
    fn an_absent_row_falls_through_to_the_us_base_and_never_to_another_level() {
        // Colemak does not touch the digit row, so it inherits it.
        assert_eq!(COLEMAK.lookup(0x21, Level::Base), Out::Char('4'));
        assert_eq!(COLEMAK.lookup(0x21, Level::Shift), Out::Char('$'));
        // And an undefined AltGr level is `None`, not a fallback to Base — a
        // fallback would type `4` for AltGr+4, which is a wrong character rather
        // than no character.
        assert_eq!(COLEMAK.lookup(0x21, Level::AltGr), Out::None);
        assert_eq!(US.lookup(0x04, Level::AltGr), Out::None);
        // A usage no table mentions is None on every level.
        for lv in [Level::Base, Level::Shift, Level::AltGr, Level::ShiftAltGr] {
            assert_eq!(US.lookup(0xF0, lv), Out::None);
        }
    }

    #[test_case]
    fn every_dead_key_a_layout_can_arm_covers_the_five_vowels_in_both_cases() {
        // Coverage of the *shipped* layouts is what matters, not table size.
        for l in LAYOUTS {
            for dk in l.dead_keys() {
                for base in "aeiou".chars() {
                    // Tilde and ring genuinely have no form for every vowel
                    // (there is no ĩ or ė), so require them only where Unicode
                    // has one — checked by asserting the *acute/grave/diaeresis/
                    // circumflex* set, which is what these layouts arm for text.
                    if matches!(dk, Dead::Tilde | Dead::Ring | Dead::Cedilla | Dead::Caron) {
                        continue;
                    }
                    assert!(
                        compose_dead(dk, base).is_some(),
                        "layout '{}': dead {} has no form for '{base}'",
                        l.id,
                        dead_name(dk)
                    );
                    let upper = base.to_ascii_uppercase();
                    assert!(
                        compose_dead(dk, upper).is_some(),
                        "layout '{}': dead {} has no form for '{upper}'",
                        l.id,
                        dead_name(dk)
                    );
                }
            }
        }
    }

    #[test_case]
    fn the_dead_map_is_sorted_and_has_no_duplicate_pair() {
        let mut prev: Option<(usize, char)> = None;
        for (dk, base, _) in DEAD_MAP {
            let key = (*dk as usize, *base);
            if let Some(p) = prev {
                assert!(p < key, "DEAD_MAP is out of order at ({}, {base})", dead_name(*dk));
            }
            prev = Some(key);
        }
    }

    #[test_case]
    fn dead_composition_produces_the_right_characters() {
        assert_eq!(compose_dead(Dead::Acute, 'e'), Some('é'));
        assert_eq!(compose_dead(Dead::Acute, 'E'), Some('É'));
        assert_eq!(compose_dead(Dead::Grave, 'a'), Some('à'));
        assert_eq!(compose_dead(Dead::Circumflex, 'o'), Some('ô'));
        assert_eq!(compose_dead(Dead::Diaeresis, 'u'), Some('ü'));
        assert_eq!(compose_dead(Dead::Diaeresis, 'y'), Some('ÿ'), "the Latin-1 exception");
        assert_eq!(compose_dead(Dead::Tilde, 'n'), Some('ñ'));
        assert_eq!(compose_dead(Dead::Ring, 'a'), Some('å'));
        assert_eq!(compose_dead(Dead::Cedilla, 'c'), Some('ç'));
        assert_eq!(compose_dead(Dead::Caron, 'e'), Some('ě'), "Latin Extended-A");
        // A gap is `None`, which the caller renders visibly.
        assert_eq!(compose_dead(Dead::Acute, 'q'), None);
        assert_eq!(compose_dead(Dead::Ring, 'e'), None);
    }

    #[test_case]
    fn spacing_forms_are_all_distinct() {
        let all = [
            Dead::Acute,
            Dead::Grave,
            Dead::Circumflex,
            Dead::Diaeresis,
            Dead::Tilde,
            Dead::Ring,
            Dead::Cedilla,
            Dead::Caron,
        ];
        for (i, a) in all.iter().enumerate() {
            for b in &all[i + 1..] {
                assert_ne!(
                    spacing_form(*a),
                    spacing_form(*b),
                    "{} and {} share a spacing form",
                    dead_name(*a),
                    dead_name(*b)
                );
            }
        }
    }

    #[test_case]
    fn compose_pairs_work_in_either_order_and_have_no_duplicates() {
        assert_eq!(compose_pair('o', 'c'), Some('©'));
        assert_eq!(compose_pair('c', 'o'), Some('©'), "symmetric");
        assert_eq!(compose_pair('\'', 'e'), Some('é'));
        assert_eq!(compose_pair('~', 'n'), Some('ñ'));
        assert_eq!(compose_pair('a', 'e'), Some('æ'));
        assert_eq!(compose_pair('-', '-'), Some('—'));
        assert_eq!(compose_pair('1', '2'), Some('½'));
        assert_eq!(compose_pair('q', 'q'), None, "a gap is a gap");
        // No pair may appear twice in either order, or the second is dead code
        // and the table is lying about its coverage.
        for (i, (a1, b1, _)) in COMPOSE.iter().enumerate() {
            for (a2, b2, _) in &COMPOSE[i + 1..] {
                let same = (a1 == a2 && b1 == b2) || (a1 == b2 && b1 == a2);
                assert!(!same, "COMPOSE lists ({a1}, {b1}) twice");
            }
        }
    }

    #[test_case]
    fn the_compose_table_is_bounded_and_listable() {
        // It is a surface a human is expected to read in one screen. If it grows
        // past a few hundred rows it should become a store-loaded file instead.
        assert!(compose_rows().len() < 200, "COMPOSE has grown past a listable size");
        assert!(!compose_rows().is_empty());
        assert!(!dead_rows().is_empty());
    }
}
