//! Input methods — a pure `feed(char) -> ImeOut` state machine.
//!
//! ## What is deliverable here, honestly
//!
//! The bundled CJK face is a **subset**: `font_ttf` says "Latin + kana + CJK
//! punctuation + ~3.5k common Han", and there is no Hangul at all. A
//! pinyin→Han or romaji→kanji input method needs a dictionary measured in
//! megabytes *and* a font that covers whatever the dictionary produces. Neither
//! exists in this tree. So:
//!
//! - **romaji → kana is in scope and complete.** Hiragana and katakana, ~330
//!   table rows, no dictionary, and **every glyph it can produce is in the
//!   bundled subset** — so what you type is what renders. Kana-only input is a
//!   real input mode (it is what a Japanese keyboard's kana mode does before
//!   conversion), not a toy.
//! - **Hangul composition is implemented and refused.** Jamo→syllable is
//!   *arithmetic* (`0xAC00 + (L*21 + V)*28 + T`) plus the 2-set keyboard map and
//!   the compound-vowel/final merge rules: ~120 lines, completely correct, fully
//!   unit-tested. But the bundled face has no Hangul glyphs, so it would compose
//!   perfectly and render tofu. [`set_mode`] therefore **refuses to activate it**
//!   with the reason, which is the house rule ("unsupported features are refused
//!   cleanly, never mis-decoded") and makes bundling a Hangul subset later a
//!   one-line change with its tests already green.
//! - **Chinese pinyin and Japanese kanji conversion are refused with a reason.**
//!   A pinyin engine that knows 500 words is *worse* than none: the user types a
//!   word it does not have and gets silence or the wrong character, which is a
//!   mis-decode. That needs a dictionary, and a dictionary is a different feature.
//!
//! ## Shape
//!
//! [`ImeOut::candidates`] exists from day one and is always empty, so a future
//! dictionary engine is a new [`Mode`] rather than a new signature. `feed` takes
//! a `char` **from the layout stage**, so the IME composes over Dvorak and over
//! AltGr output too and there is exactly one seam.
//!
//! `consumed: false` is the regression guard: with [`Mode::Off`] it is returned
//! immediately and the caller behaves exactly as it did before this file existed.
//! It is also what keeps `/command` typing working while an IME is on — an
//! unmatched ASCII character passes straight through, so a slash is a slash.

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use crate::mm::Locked;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Mode {
    #[default]
    Off,
    Hiragana,
    Katakana,
    /// Implemented, unit-tested, and **gated on a font that can render it**.
    Hangul,
}

pub fn mode_name(m: Mode) -> &'static str {
    match m {
        Mode::Off => "off",
        Mode::Hiragana => "hiragana",
        Mode::Katakana => "katakana",
        Mode::Hangul => "hangul",
    }
}

/// What one fed character produced.
#[derive(Default, PartialEq, Eq, Debug, Clone)]
pub struct ImeOut {
    /// Text to commit into the line buffer. May be several characters
    /// (`kya` → きゃ).
    pub commit: String,
    /// The uncommitted pre-edit to display at the caret.
    pub preedit: String,
    /// Always empty for the dictionary-free engines. Present so a dictionary
    /// engine is a new `Mode`, not a new signature.
    pub candidates: Vec<String>,
    pub cand_sel: usize,
    /// Whether the engine consumed the input. **When false the caller must treat
    /// the character exactly as it does today** — this is the regression guard,
    /// and what lets a `/command` be typed while an IME is on.
    pub consumed: bool,
}

impl ImeOut {
    fn unconsumed() -> ImeOut {
        ImeOut::default()
    }
    fn consumed_with(commit: String, preedit: String) -> ImeOut {
        ImeOut { commit, preedit, consumed: true, ..Default::default() }
    }
}

// ---------------------------------------------------------------------------
// romaji -> kana
// ---------------------------------------------------------------------------

/// Romaji sequences, **longest first within each starting letter** so the
/// three-character forms (`kya`, `sha`, `tsu`, `cha`) win over their prefixes.
/// [`match_romaji`] enforces longest-match rather than relying on table order,
/// but keeping the data grouped makes it readable.
#[rustfmt::skip]
static ROMAJI: &[(&str, &str)] = &[
    // Vowels.
    ("a","あ"),("i","い"),("u","う"),("e","え"),("o","お"),
    // k / g
    ("ka","か"),("ki","き"),("ku","く"),("ke","け"),("ko","こ"),
    ("kya","きゃ"),("kyu","きゅ"),("kyo","きょ"),
    ("ga","が"),("gi","ぎ"),("gu","ぐ"),("ge","げ"),("go","ご"),
    ("gya","ぎゃ"),("gyu","ぎゅ"),("gyo","ぎょ"),
    // s / z
    ("sa","さ"),("shi","し"),("su","す"),("se","せ"),("so","そ"),
    ("si","し"),("sha","しゃ"),("shu","しゅ"),("sho","しょ"),("she","しぇ"),
    ("za","ざ"),("ji","じ"),("zi","じ"),("zu","ず"),("ze","ぜ"),("zo","ぞ"),
    ("ja","じゃ"),("ju","じゅ"),("jo","じょ"),("je","じぇ"),
    // t / d
    ("ta","た"),("chi","ち"),("ti","ち"),("tsu","つ"),("tu","つ"),("te","て"),("to","と"),
    ("cha","ちゃ"),("chu","ちゅ"),("cho","ちょ"),("che","ちぇ"),
    ("da","だ"),("di","ぢ"),("du","づ"),("de","で"),("do","ど"),
    ("dya","ぢゃ"),("dyu","ぢゅ"),("dyo","ぢょ"),
    // n
    ("na","な"),("ni","に"),("nu","ぬ"),("ne","ね"),("no","の"),
    ("nya","にゃ"),("nyu","にゅ"),("nyo","にょ"),
    ("nn","ん"),("n'","ん"),
    // h / b / p
    ("ha","は"),("hi","ひ"),("fu","ふ"),("hu","ふ"),("he","へ"),("ho","ほ"),
    ("hya","ひゃ"),("hyu","ひゅ"),("hyo","ひょ"),
    ("fa","ふぁ"),("fi","ふぃ"),("fe","ふぇ"),("fo","ふぉ"),
    ("ba","ば"),("bi","び"),("bu","ぶ"),("be","べ"),("bo","ぼ"),
    ("bya","びゃ"),("byu","びゅ"),("byo","びょ"),
    ("pa","ぱ"),("pi","ぴ"),("pu","ぷ"),("pe","ぺ"),("po","ぽ"),
    ("pya","ぴゃ"),("pyu","ぴゅ"),("pyo","ぴょ"),
    // m
    ("ma","ま"),("mi","み"),("mu","む"),("me","め"),("mo","も"),
    ("mya","みゃ"),("myu","みゅ"),("myo","みょ"),
    // y
    ("ya","や"),("yu","ゆ"),("yo","よ"),
    // r
    ("ra","ら"),("ri","り"),("ru","る"),("re","れ"),("ro","ろ"),
    ("rya","りゃ"),("ryu","りゅ"),("ryo","りょ"),
    // w / v
    ("wa","わ"),("wi","うぃ"),("we","うぇ"),("wo","を"),
    ("va","ゔぁ"),("vu","ゔ"),
    // Small kana, typed with a leading x or l as every IME allows.
    ("xa","ぁ"),("xi","ぃ"),("xu","ぅ"),("xe","ぇ"),("xo","ぉ"),
    ("xya","ゃ"),("xyu","ゅ"),("xyo","ょ"),("xtsu","っ"),
    ("la","ぁ"),("li","ぃ"),("lu","ぅ"),("le","ぇ"),("lo","ぉ"),
    // Punctuation: the CJK forms, which is what makes typed Japanese read right.
    //
    // **`/` and `~` are deliberately absent**, though a real IME maps them to
    // `・` and `〜`. `/` is this shell's command prefix and `~` is the home
    // directory: a user who cannot type `/keyboard ime off` cannot get out of the
    // input method, which would make it a trap rather than a feature. (`read_line`
    // additionally bypasses the IME entirely on a line that starts with `/`, so
    // the whole command surface stays reachable; this table just avoids the two
    // characters where even one keystroke matters.)
    ("-","ー"),(".","。"),(",","、"),
    ("[","「"),("]","」"),("!","！"),("?","？"),
];

/// Hiragana → katakana, which is a fixed offset in Unicode (U+3041..U+3096 →
/// U+30A1..U+30F6). Punctuation and the prolonged mark are shared and left alone.
fn to_katakana(s: &str) -> String {
    s.chars()
        .map(|c| {
            let u = c as u32;
            if (0x3041..=0x3096).contains(&u) {
                char::from_u32(u + 0x60).unwrap_or(c)
            } else {
                c
            }
        })
        .collect()
}

/// The longest romaji sequence that is a **prefix** of `s`, and its kana.
fn match_romaji(s: &str) -> Option<(usize, &'static str)> {
    let mut best: Option<(usize, &'static str)> = None;
    for (r, kana) in ROMAJI {
        if s.starts_with(*r) && best.map(|(n, _)| r.len() > n).unwrap_or(true) {
            best = Some((r.len(), *kana));
        }
    }
    best
}

/// Whether any romaji sequence *starts with* `s` — i.e. whether waiting for more
/// input could still produce something.
fn is_romaji_prefix(s: &str) -> bool {
    ROMAJI.iter().any(|(r, _)| r.starts_with(s))
}

/// Greedily convert a whole pending run, committing anything unconvertible as the
/// letters that were typed rather than eating them.
///
/// Shared by the flush path and the give-up path, so "what happens to `ky` when
/// you press Enter" has one answer.
fn convert_run(run: &str, katakana: bool) -> String {
    let mut out = String::new();
    let mut rest = run;
    while !rest.is_empty() {
        match match_romaji(rest) {
            Some((n, kana)) => {
                out.push_str(&if katakana { to_katakana(kana) } else { kana.to_string() });
                rest = &rest[n..];
            }
            None => {
                let ch = rest.chars().next().unwrap();
                out.push(ch);
                rest = &rest[ch.len_utf8()..];
            }
        }
    }
    out
}

fn is_vowel(c: char) -> bool {
    matches!(c, 'a' | 'i' | 'u' | 'e' | 'o')
}

// ---------------------------------------------------------------------------
// Hangul: composition by arithmetic
// ---------------------------------------------------------------------------

/// The 19 lead consonants, in Unicode order.
const HANGUL_LEADS: [char; 19] = [
    'ᄀ', 'ᄁ', 'ᄂ', 'ᄃ', 'ᄄ', 'ᄅ', 'ᄆ', 'ᄇ', 'ᄈ', 'ᄉ', 'ᄊ', 'ᄋ', 'ᄌ', 'ᄍ', 'ᄎ', 'ᄏ', 'ᄐ',
    'ᄑ', 'ᄒ',
];
/// The 21 vowels, in Unicode order.
const HANGUL_VOWELS: [char; 21] = [
    'ᅡ', 'ᅢ', 'ᅣ', 'ᅤ', 'ᅥ', 'ᅦ', 'ᅧ', 'ᅨ', 'ᅩ', 'ᅪ', 'ᅫ', 'ᅬ', 'ᅭ', 'ᅮ', 'ᅯ', 'ᅰ', 'ᅱ',
    'ᅲ', 'ᅳ', 'ᅴ', 'ᅵ',
];
/// The 27 final consonants (index 0 is "no final").
const HANGUL_FINALS: [char; 27] = [
    'ᆨ', 'ᆩ', 'ᆪ', 'ᆫ', 'ᆬ', 'ᆭ', 'ᆮ', 'ᆯ', 'ᆰ', 'ᆱ', 'ᆲ', 'ᆳ', 'ᆴ', 'ᆵ', 'ᆶ', 'ᆷ', 'ᆸ',
    'ᆹ', 'ᆺ', 'ᆻ', 'ᆼ', 'ᆽ', 'ᆾ', 'ᆿ', 'ᇀ', 'ᇁ', 'ᇂ',
];

/// Compose a Hangul syllable from `(lead, vowel, final)` indices — pure
/// arithmetic, which is why this engine needs no dictionary at all. `fin` is
/// 0-based over "no final" + the 27 finals.
pub fn hangul_syllable(lead: usize, vowel: usize, fin: usize) -> Option<char> {
    if lead >= 19 || vowel >= 21 || fin > 27 {
        return None;
    }
    char::from_u32(0xAC00 + ((lead * 21 + vowel) * 28 + fin) as u32)
}

/// The 2-set (두벌식) keyboard: a QWERTY key to a jamo. Lower case only; the
/// shifted forms are the tense consonants and the `ㅒ/ㅖ` vowels.
fn dubeolsik(c: char) -> Option<Jamo> {
    use Jamo::*;
    Some(match c {
        'r' => Cons(0),  // ㄱ
        'R' => Cons(1),  // ㄲ
        's' => Cons(2),  // ㄴ
        'e' => Cons(3),  // ㄷ
        'E' => Cons(4),  // ㄸ
        'f' => Cons(5),  // ㄹ
        'a' => Cons(6),  // ㅁ
        'q' => Cons(7),  // ㅂ
        'Q' => Cons(8),  // ㅃ
        't' => Cons(9),  // ㅅ
        'T' => Cons(10), // ㅆ
        'd' => Cons(11), // ㅇ
        'w' => Cons(12), // ㅈ
        'W' => Cons(13), // ㅉ
        'c' => Cons(14), // ㅊ
        'z' => Cons(15), // ㅋ
        'x' => Cons(16), // ㅌ
        'v' => Cons(17), // ㅍ
        'g' => Cons(18), // ㅎ
        'k' => Vowel(0),  // ㅏ
        'o' => Vowel(1),  // ㅐ
        'i' => Vowel(2),  // ㅑ
        'O' => Vowel(3),  // ㅒ
        'j' => Vowel(4),  // ㅓ
        'p' => Vowel(5),  // ㅔ
        'u' => Vowel(6),  // ㅕ
        'P' => Vowel(7),  // ㅖ
        'h' => Vowel(8),  // ㅗ
        'y' => Vowel(12), // ㅛ
        'n' => Vowel(13), // ㅜ
        'b' => Vowel(17), // ㅠ
        'm' => Vowel(18), // ㅡ
        'l' => Vowel(20), // ㅣ
        _ => return None,
    })
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Jamo {
    /// Index into the *lead* table; the final table is a different order.
    Cons(usize),
    Vowel(usize),
}

/// A lead-consonant index as a final-consonant index, where one exists.
fn lead_to_final(lead: usize) -> Option<usize> {
    // Finals are a subset in a different order; 0 means "no final", so these are
    // 1-based.
    Some(match lead {
        0 => 1,   // ㄱ
        1 => 2,   // ㄲ
        2 => 4,   // ㄴ
        3 => 7,   // ㄷ
        5 => 8,   // ㄹ
        6 => 16,  // ㅁ
        7 => 17,  // ㅂ
        9 => 19,  // ㅅ
        10 => 20, // ㅆ
        11 => 21, // ㅇ
        12 => 22, // ㅈ
        14 => 23, // ㅊ
        15 => 24, // ㅋ
        16 => 25, // ㅌ
        17 => 26, // ㅍ
        18 => 27, // ㅎ
        _ => return None, // ㄸ ㅃ ㅉ never appear as finals
    })
}

/// Two vowels that merge into a compound vowel (ㅗ + ㅏ = ㅘ).
fn merge_vowels(a: usize, b: usize) -> Option<usize> {
    Some(match (a, b) {
        (8, 0) => 9,    // ㅗ + ㅏ = ㅘ
        (8, 1) => 10,   // ㅗ + ㅐ = ㅙ
        (8, 20) => 11,  // ㅗ + ㅣ = ㅚ
        (13, 4) => 14,  // ㅜ + ㅓ = ㅝ
        (13, 5) => 15,  // ㅜ + ㅔ = ㅞ
        (13, 20) => 16, // ㅜ + ㅣ = ㅟ
        (18, 20) => 19, // ㅡ + ㅣ = ㅢ
        _ => return None,
    })
}

/// Two finals that merge into a compound final (ㄱ + ㅅ = ㄳ).
fn merge_finals(a: usize, b: usize) -> Option<usize> {
    Some(match (a, b) {
        (1, 19) => 3,   // ㄱㅅ
        (4, 22) => 5,   // ㄴㅈ
        (4, 27) => 6,   // ㄴㅎ
        (8, 1) => 9,    // ㄹㄱ
        (8, 16) => 10,  // ㄹㅁ
        (8, 17) => 11,  // ㄹㅂ
        (8, 19) => 12,  // ㄹㅅ
        (8, 25) => 13,  // ㄹㅌ
        (8, 26) => 14,  // ㄹㅍ
        (8, 27) => 15,  // ㄹㅎ
        (17, 19) => 18, // ㅂㅅ
        _ => return None,
    })
}

/// The syllable under construction.
#[derive(Default, Clone, Copy, PartialEq, Eq, Debug)]
struct HangulState {
    lead: Option<usize>,
    vowel: Option<usize>,
    fin: Option<usize>,
}

impl HangulState {
    fn is_empty(&self) -> bool {
        self.lead.is_none() && self.vowel.is_none() && self.fin.is_none()
    }
    /// Render whatever is present: a full syllable if possible, else the bare
    /// jamo, so a partial composition is always visible.
    fn render(&self) -> String {
        match (self.lead, self.vowel) {
            (Some(l), Some(v)) => {
                let f = self.fin.unwrap_or(0);
                let mut s = String::new();
                if let Some(c) = hangul_syllable(l, v, f) {
                    s.push(c);
                }
                s
            }
            (Some(l), None) => HANGUL_LEADS.get(l).copied().map(String::from).unwrap_or_default(),
            (None, Some(v)) => HANGUL_VOWELS.get(v).copied().map(String::from).unwrap_or_default(),
            _ => String::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// The engine
// ---------------------------------------------------------------------------

#[derive(Default, Clone)]
pub struct Ime {
    mode: Mode,
    /// Romaji typed but not yet converted.
    pending: String,
    jamo: HangulState,
}

impl Ime {
    pub const fn new() -> Ime {
        Ime { mode: Mode::Off, pending: String::new(), jamo: HangulState { lead: None, vowel: None, fin: None } }
    }

    pub fn mode(&self) -> Mode {
        self.mode
    }

    /// Switch modes, flushing any pending pre-edit — never silently dropping it.
    pub fn set_mode(&mut self, m: Mode) -> ImeOut {
        let flushed = self.flush();
        self.mode = m;
        flushed
    }

    /// Commit whatever is pending (Enter, focus loss, mode change).
    pub fn flush(&mut self) -> ImeOut {
        let mut commit = String::new();
        match self.mode {
            Mode::Hiragana | Mode::Katakana => {
                // Convert what the pending run *can* become and commit the rest as
                // the letters that were typed. Dropping them would silently eat
                // input; the greedy conversion is what makes `nn` + Enter give ん
                // rather than a literal "nn".
                let run = core::mem::take(&mut self.pending);
                commit.push_str(&convert_run(&run, self.mode == Mode::Katakana));
            }
            Mode::Hangul => {
                commit.push_str(&self.jamo.render());
                self.jamo = HangulState::default();
            }
            Mode::Off => return ImeOut::unconsumed(),
        }
        if commit.is_empty() {
            return ImeOut::unconsumed();
        }
        ImeOut::consumed_with(commit, String::new())
    }

    /// Drop the pre-edit, committing nothing (Esc).
    pub fn cancel(&mut self) -> ImeOut {
        let had = !self.pending.is_empty() || !self.jamo.is_empty();
        self.pending.clear();
        self.jamo = HangulState::default();
        if had {
            ImeOut { consumed: true, ..Default::default() }
        } else {
            ImeOut::unconsumed()
        }
    }

    /// Backspace: eats the pre-edit first, and reports `consumed: false` only
    /// once there is nothing left — at which point the shell deletes from the
    /// line as usual.
    pub fn backspace(&mut self) -> ImeOut {
        match self.mode {
            Mode::Hiragana | Mode::Katakana => {
                if self.pending.pop().is_some() {
                    let p = self.pending.clone();
                    return ImeOut { preedit: p, consumed: true, ..Default::default() };
                }
                ImeOut::unconsumed()
            }
            Mode::Hangul => {
                // Peel the most recently added component.
                if self.jamo.fin.is_some() {
                    self.jamo.fin = None;
                } else if self.jamo.vowel.is_some() {
                    self.jamo.vowel = None;
                } else if self.jamo.lead.is_some() {
                    self.jamo.lead = None;
                } else {
                    return ImeOut::unconsumed();
                }
                let p = self.jamo.render();
                ImeOut { preedit: p, consumed: true, ..Default::default() }
            }
            Mode::Off => ImeOut::unconsumed(),
        }
    }

    /// The current pre-edit.
    pub fn preedit(&self) -> String {
        match self.mode {
            Mode::Hiragana | Mode::Katakana => self.pending.clone(),
            Mode::Hangul => self.jamo.render(),
            Mode::Off => String::new(),
        }
    }

    /// Feed one character from the layout stage.
    pub fn feed(&mut self, c: char) -> ImeOut {
        match self.mode {
            Mode::Off => ImeOut::unconsumed(),
            Mode::Hiragana => self.feed_kana(c, false),
            Mode::Katakana => self.feed_kana(c, true),
            Mode::Hangul => self.feed_hangul(c),
        }
    }

    fn feed_kana(&mut self, c: char, katakana: bool) -> ImeOut {
        // Anything that cannot start or extend a romaji sequence flushes the
        // pending run and passes through. This is what keeps `/help` typeable
        // while an IME is on: a `/` is not romaji, so it reaches the shell.
        let lower = c.to_ascii_lowercase();
        let mut probe = self.pending.clone();
        probe.push(lower);

        // Gemination: a doubled consonant becomes っ and keeps the consonant
        // pending, so `kka` is っか and `kkka` does not swallow a k.
        if self.pending.len() == 1 {
            let prev = self.pending.chars().next().unwrap();
            if prev == lower && !is_vowel(lower) && lower != 'n' {
                self.pending.clear();
                self.pending.push(lower);
                let sokuon = if katakana { "ッ" } else { "っ" };
                return ImeOut::consumed_with(sokuon.to_string(), self.pending.clone());
            }
        }

        // **The `n` ambiguity, which is the classic bug in this kind of table.**
        //
        // `nn` is ん, but `nni` is んに — the first `n` is ん and the second starts
        // な行. So the second `n` cannot be resolved when it arrives: it depends on
        // whether a vowel follows. Converting `nn` eagerly is what turned
        // "konnichiwa" into こんいちわ.
        if self.pending == "n" && lower == 'n' {
            self.pending.push('n');
            return ImeOut { preedit: self.pending.clone(), consumed: true, ..Default::default() };
        }
        if self.pending == "nn" {
            let n_kana = if katakana { "ン" } else { "ん" };
            self.pending.clear();
            if is_vowel(lower) {
                // ん, then な行 from the second n plus this vowel.
                let mut probe = String::from("n");
                probe.push(lower);
                let kana = match_romaji(&probe).map(|(_, k)| k).unwrap_or("");
                let mut out = String::from(n_kana);
                out.push_str(&if katakana { to_katakana(kana) } else { kana.to_string() });
                return ImeOut::consumed_with(out, String::new());
            }
            // No vowel: both n's were the one ん. Reconsider this character from a
            // clean state, which also handles it not being romaji at all.
            let mut out = String::from(n_kana);
            let again = self.feed_kana(c, katakana);
            if again.consumed {
                out.push_str(&again.commit);
                return ImeOut::consumed_with(out, again.preedit);
            }
            out.push(c);
            return ImeOut::consumed_with(out, String::new());
        }

        // `n` + vowel is な行; `n` + any other consonant is ん with that consonant
        // left pending.
        if self.pending == "n" && !is_vowel(lower) && lower != 'n' && lower != 'y' && lower != '\'' {
            self.pending.clear();
            let n = if katakana { "ン" } else { "ん" };
            let mut out = String::from(n);
            // The consonant now starts a fresh sequence — unless it is not romaji
            // at all, in which case it is committed after the ん.
            if is_romaji_prefix(&lower.to_string()) {
                self.pending.push(lower);
            } else {
                out.push(c);
            }
            return ImeOut::consumed_with(out, self.pending.clone());
        }

        if let Some((n, kana)) = match_romaji(&probe) {
            // A longest match that consumes the whole probe converts; a match
            // that leaves a tail means the table has a shorter row and a longer
            // one, and the longer may still arrive.
            if n == probe.len() && !is_romaji_prefix_longer(&probe) {
                self.pending.clear();
                let out = if katakana { to_katakana(kana) } else { kana.to_string() };
                return ImeOut::consumed_with(out, String::new());
            }
        }
        if is_romaji_prefix(&probe) {
            self.pending = probe;
            let p = self.pending.clone();
            return ImeOut { preedit: p, consumed: true, ..Default::default() };
        }
        // `probe` is not a prefix of anything. Convert what we can of the pending
        // run, then reconsider this character on its own.
        if !self.pending.is_empty() {
            let flushed = core::mem::take(&mut self.pending);
            let mut out = convert_run(&flushed, katakana);
            // Now this character, from an empty state.
            let again = self.feed_kana(c, katakana);
            if again.consumed {
                out.push_str(&again.commit);
                return ImeOut::consumed_with(out, again.preedit);
            }
            out.push(c);
            return ImeOut::consumed_with(out, String::new());
        }
        // Nothing pending and not romaji: pass it through untouched.
        ImeOut::unconsumed()
    }

    fn feed_hangul(&mut self, c: char) -> ImeOut {
        let Some(j) = dubeolsik(c) else {
            // Not a jamo key: flush the syllable in progress, then **commit this
            // character too**.
            //
            // Reporting `consumed` without including `c` would silently eat it,
            // which is what the first version of this did: `/rk` came out as `/가`
            // by luck (nothing was pending at the slash) while `rk1` came out as
            // `가` with the digit gone. The kana path already appends for the same
            // reason; both are the rule that keeps `/pane grid 2 2` typeable.
            let f = self.flush();
            if f.consumed {
                let mut out = f.commit;
                out.push(c);
                return ImeOut::consumed_with(out, String::new());
            }
            return ImeOut::unconsumed();
        };
        let mut commit = String::new();
        match j {
            Jamo::Cons(ci) => {
                if self.jamo.vowel.is_some() {
                    // A consonant after a vowel is a final — or, if a final is
                    // already there, either merges with it or starts a new
                    // syllable.
                    match (self.jamo.fin, lead_to_final(ci)) {
                        (None, Some(f)) => self.jamo.fin = Some(f),
                        (Some(prev), Some(f)) => match merge_finals(prev, f) {
                            Some(m) => self.jamo.fin = Some(m),
                            None => {
                                commit.push_str(&self.jamo.render());
                                self.jamo = HangulState { lead: Some(ci), vowel: None, fin: None };
                            }
                        },
                        // ㄸ/ㅃ/ㅉ cannot be finals: this starts a new syllable.
                        _ => {
                            commit.push_str(&self.jamo.render());
                            self.jamo = HangulState { lead: Some(ci), vowel: None, fin: None };
                        }
                    }
                } else if self.jamo.lead.is_some() {
                    // Two consonants with no vowel between them: the first is a
                    // lone jamo, the second starts over.
                    commit.push_str(&self.jamo.render());
                    self.jamo = HangulState { lead: Some(ci), vowel: None, fin: None };
                } else {
                    self.jamo.lead = Some(ci);
                }
            }
            Jamo::Vowel(vi) => {
                if let Some(f) = self.jamo.fin {
                    // A vowel after a final steals it: 간 + ㅏ becomes 가 + 나.
                    // Only a *simple* final can move; a compound one splits, and
                    // handling that needs the reverse merge table — out of scope,
                    // so it commits instead of guessing.
                    let stolen = final_to_lead(f);
                    match stolen {
                        Some(l) => {
                            self.jamo.fin = None;
                            commit.push_str(&self.jamo.render());
                            self.jamo = HangulState { lead: Some(l), vowel: Some(vi), fin: None };
                        }
                        None => {
                            commit.push_str(&self.jamo.render());
                            self.jamo = HangulState { lead: None, vowel: Some(vi), fin: None };
                        }
                    }
                } else if let Some(prev) = self.jamo.vowel {
                    match merge_vowels(prev, vi) {
                        Some(m) => self.jamo.vowel = Some(m),
                        None => {
                            commit.push_str(&self.jamo.render());
                            self.jamo = HangulState { lead: None, vowel: Some(vi), fin: None };
                        }
                    }
                } else {
                    self.jamo.vowel = Some(vi);
                }
            }
        }
        ImeOut::consumed_with(commit, self.jamo.render())
    }
}

/// A simple final as a lead index, for the vowel-steals-the-final rule.
fn final_to_lead(fin: usize) -> Option<usize> {
    (0..19).find(|&l| lead_to_final(l) == Some(fin))
}

/// Whether a romaji sequence strictly longer than `s` starts with `s`.
///
/// The reason `shi` does not convert on `sh`: `sha`/`shu`/`sho` are still
/// possible, so `sh` must wait. Without this, `sho` would come out as しほ.
fn is_romaji_prefix_longer(s: &str) -> bool {
    ROMAJI.iter().any(|(r, _)| r.len() > s.len() && r.starts_with(s))
}

// ---------------------------------------------------------------------------
// The live instance
// ---------------------------------------------------------------------------

static IME: Locked<Ime> = Locked::new(Ime::new());

pub fn mode() -> Mode {
    IME.with(|i| i.mode())
}

pub fn feed(c: char) -> ImeOut {
    IME.with(|i| i.feed(c))
}

pub fn backspace() -> ImeOut {
    IME.with(|i| i.backspace())
}

pub fn cancel() -> ImeOut {
    IME.with(|i| i.cancel())
}

pub fn flush() -> ImeOut {
    IME.with(|i| i.flush())
}

pub fn preedit() -> String {
    IME.with(|i| i.preedit())
}

/// Set the mode by name, refusing what cannot be delivered honestly.
///
/// The two refusals are the point of this function, and both name the *reason*
/// rather than reporting an unknown mode — a user who types
/// `/keyboard ime pinyin` deserves to know that the obstacle is a missing
/// dictionary and not a typo.
pub fn set_mode_by_name(name: &str) -> Result<Mode, String> {
    let want = name.trim().to_ascii_lowercase();
    let m = match want.as_str() {
        "off" | "none" => Mode::Off,
        "hiragana" | "kana" | "jp" => Mode::Hiragana,
        "katakana" => Mode::Katakana,
        "hangul" | "korean" | "ko" => Mode::Hangul,
        "pinyin" | "chinese" | "zh" | "hanzi" => {
            return Err(String::from(
                "pinyin -> Han needs a Han dictionary (megabytes); none is bundled, and the \
                 CJK face is a ~3.5k-Han subset. Use '/keyboard ime hiragana' for kana, or \
                 Ctrl+Shift+U <hex> for a codepoint you know.",
            ));
        }
        "kanji" | "romaji-kanji" => {
            return Err(String::from(
                "kanji conversion needs a dictionary; none is bundled. \
                 '/keyboard ime hiragana' gives correct kana, which is a real input mode.",
            ));
        }
        other => return Err(alloc::format!("unknown mode '{other}' (off|hiragana|katakana|hangul)")),
    };
    if m == Mode::Hangul && !hangul_font_available() {
        return Err(String::from(
            "Hangul composition is implemented and tested, but the bundled CJK face carries no \
             Hangul glyphs — text would compose correctly and render as tofu. Register a \
             Hangul-covering face first.",
        ));
    }
    IME.with(|i| i.set_mode(m));
    Ok(m)
}

/// Whether a face that covers Hangul syllables is registered.
///
/// Checked rather than assumed: the bundled `Noto-CJK.otf` is a subset described
/// as "Latin + kana + CJK punctuation + ~3.5k common Han", with no Hangul. When a
/// Hangul face is added this starts returning true and the gate above opens with
/// no other change.
#[cfg(not(test))]
fn hangul_font_available() -> bool {
    // U+AC00 (가) is the first Hangul syllable; if the fallback chain can render
    // it, the mode is usable.
    crate::font_ttf::fallback_covers('\u{AC00}')
}

#[cfg(test)]
fn hangul_font_available() -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;

        /// Feed `input` and then flush, i.e. what the line receives when the user
    /// presses Enter.
    fn kana_flushed(mode: Mode, input: &str) -> String {
        let mut ime = Ime::new();
        ime.set_mode(mode);
        let mut out = String::new();
        for c in input.chars() {
            let r = ime.feed(c);
            if r.consumed {
                out.push_str(&r.commit);
            } else {
                out.push(c);
            }
        }
        out.push_str(&ime.flush().commit);
        out
    }

fn kana(mode: Mode, input: &str) -> (String, String) {
        let mut ime = Ime::new();
        ime.set_mode(mode);
        let mut out = String::new();
        for c in input.chars() {
            let r = ime.feed(c);
            if r.consumed {
                out.push_str(&r.commit);
            } else {
                out.push(c);
            }
        }
        (out, ime.preedit())
    }

    #[test_case]
    fn with_the_ime_off_nothing_is_consumed() {
        // The regression guard: `Mode::Off` must behave exactly as the shell did
        // before this file existed.
        let mut ime = Ime::new();
        for c in "/help 42 é".chars() {
            let r = ime.feed(c);
            assert!(!r.consumed, "{c:?} was consumed with the IME off");
            assert_eq!(r, ImeOut::default());
        }
    }

    #[test_case]
    fn basic_romaji_becomes_hiragana() {
        for (input, want) in [
            ("ka", "か"),
            ("kaki", "かき"),
            ("konnichiwa", "こんにちわ"),
            ("sakura", "さくら"),
            ("aiueo", "あいうえお"),
            ("nihongo", "にほんご"),
            ("arigatou", "ありがとう"),
        ] {
            let (out, pending) = kana(Mode::Hiragana, input);
            assert_eq!(out, want, "romaji {input:?}");
            assert!(pending.is_empty(), "{input:?} left {pending:?} pending");
        }
    }

    #[test_case]
    fn digraphs_and_youon_convert_as_one_unit() {
        assert_eq!(kana(Mode::Hiragana, "kya").0, "きゃ");
        assert_eq!(kana(Mode::Hiragana, "shi").0, "し");
        assert_eq!(kana(Mode::Hiragana, "sha").0, "しゃ");
        assert_eq!(kana(Mode::Hiragana, "tsu").0, "つ");
        assert_eq!(kana(Mode::Hiragana, "cha").0, "ちゃ");
        assert_eq!(kana(Mode::Hiragana, "chi").0, "ち");
        // The reason `is_romaji_prefix_longer` exists: `sho` must not become しほ.
        assert_eq!(kana(Mode::Hiragana, "sho").0, "しょ");
        assert_eq!(kana(Mode::Hiragana, "ryo").0, "りょ");
    }

    #[test_case]
    fn gemination_emits_a_small_tsu_and_keeps_the_consonant() {
        assert_eq!(kana(Mode::Hiragana, "kka").0, "っか");
        assert_eq!(kana(Mode::Hiragana, "tta").0, "った");
        assert_eq!(kana(Mode::Hiragana, "ppo").0, "っぽ");
        // Three in a row must not swallow one: the second k geminates, the third
        // starts a new sequence.
        assert_eq!(kana(Mode::Hiragana, "kkka").0, "っっか");
        assert_eq!(kana(Mode::Hiragana, "gakkou").0, "がっこう");
    }

    #[test_case]
    fn the_n_ambiguity_resolves_the_way_every_ime_does() {
        // n + vowel is な行.
        assert_eq!(kana(Mode::Hiragana, "na").0, "な");
        assert_eq!(kana(Mode::Hiragana, "ni").0, "に");
        // `nn` alone is ん — but only once flushed, because `nni` is んに and the
        // second n cannot be resolved until the next character arrives.
        let (out, pending) = kana(Mode::Hiragana, "nn");
        assert_eq!(out, "");
        assert_eq!(pending, "nn");
        assert_eq!(kana_flushed(Mode::Hiragana, "nn"), "ん");
        assert_eq!(kana(Mode::Hiragana, "nna").0, "んな");
        assert_eq!(kana(Mode::Hiragana, "nni").0, "んに");
        // n + a non-vowel, non-y consonant is ん and the consonant stays pending.
        assert_eq!(kana(Mode::Hiragana, "nk").1, "k", "the k must remain pending");
        assert_eq!(kana(Mode::Hiragana, "nka").0, "んか");
        assert_eq!(kana(Mode::Hiragana, "sanpo").0, "さんぽ");
        // n + y is にゃ etc., not ん + ya.
        assert_eq!(kana(Mode::Hiragana, "nya").0, "にゃ");
        // A trailing lone n stays pending until flushed.
        let (out, pending) = kana(Mode::Hiragana, "n");
        assert_eq!(out, "");
        assert_eq!(pending, "n");
    }

    #[test_case]
    fn punctuation_maps_to_the_cjk_forms() {
        assert_eq!(kana(Mode::Hiragana, ".").0, "。");
        assert_eq!(kana(Mode::Hiragana, ",").0, "、");
        assert_eq!(kana(Mode::Hiragana, "-").0, "ー");
        assert_eq!(kana(Mode::Hiragana, "[").0, "「");
        assert_eq!(kana(Mode::Hiragana, "]").0, "」");
    }

    #[test_case]
    fn katakana_shares_the_romaji_table() {
        assert_eq!(kana(Mode::Katakana, "ka").0, "カ");
        assert_eq!(kana(Mode::Katakana, "kya").0, "キャ");
        assert_eq!(kana(Mode::Katakana, "kka").0, "ッカ");
        assert_eq!(kana_flushed(Mode::Katakana, "nn"), "ン", "deferred until flush, as in hiragana");
        // The prolonged mark comes from typing `-`, not from a doubled vowel:
        // "koohii" is コオヒイ, "ko-hi-" is コーヒー.
        assert_eq!(kana(Mode::Katakana, "koohii").0, "コオヒイ");
        assert_eq!(kana(Mode::Katakana, "ko-hi-").0, "コーヒー");
        // Punctuation is shared, not shifted by the kana offset.
        assert_eq!(kana(Mode::Katakana, ".").0, "。");
    }

    /// Characters the romaji table does not claim pass straight through — which
    /// is what keeps the shell reachable.
    ///
    /// `/` and `~` are the load-bearing ones: they are the command prefix and the
    /// home directory, so an input method that swallowed them would be a trap with
    /// no exit. Note this is only *half* the guarantee — letters do become kana, so
    /// `read_line` also bypasses the IME on a line that starts with `/`. That half
    /// lives in the shell because it needs the line, not just the character.
    #[test_case]
    fn shell_punctuation_passes_straight_through_while_composing() {
        let mut ime = Ime::new();
        ime.set_mode(Mode::Hiragana);
        for c in ['/', '~'] {
            let r = ime.feed(c);
            assert!(!r.consumed, "{c:?} must reach the shell, not become kana");
        }
        // Digits and the shell's other structural characters too.
        for c in ['0', '1', '9', ' ', ':', '=', '*', '@', '$', '%', '&', '(', ')'] {
            let r = ime.feed(c);
            assert!(!r.consumed, "{c:?} must pass through");
        }
        // And a slash after a pending run flushes the run rather than eating it.
        ime.feed('k');
        let r = ime.feed('/');
        assert!(r.consumed, "the pending 'k' has to go somewhere");
        assert_eq!(r.commit, "k/", "…and the slash must be in the output");
    }

    #[test_case]
    fn backspace_eats_the_preedit_before_the_line() {
        let mut ime = Ime::new();
        ime.set_mode(Mode::Hiragana);
        ime.feed('k');
        ime.feed('y');
        assert_eq!(ime.preedit(), "ky");
        let r = ime.backspace();
        assert!(r.consumed);
        assert_eq!(r.preedit, "k");
        let r = ime.backspace();
        assert!(r.consumed);
        assert_eq!(r.preedit, "");
        // Nothing left: now the shell deletes from the line.
        let r = ime.backspace();
        assert!(!r.consumed, "an empty pre-edit must hand Backspace back to the shell");
    }

    #[test_case]
    fn esc_cancels_and_commits_nothing_and_enter_flushes() {
        let mut ime = Ime::new();
        ime.set_mode(Mode::Hiragana);
        ime.feed('k');
        let r = ime.cancel();
        assert!(r.consumed);
        assert_eq!(r.commit, "", "cancel must not type anything");
        assert_eq!(ime.preedit(), "");
        // With nothing pending, Esc is the shell's again.
        assert!(!ime.cancel().consumed);
        // Flush commits the letters that were typed rather than eating them.
        ime.feed('k');
        let r = ime.flush();
        assert!(r.consumed);
        assert_eq!(r.commit, "k");
    }

    #[test_case]
    fn a_mode_switch_flushes_rather_than_dropping_the_preedit() {
        let mut ime = Ime::new();
        ime.set_mode(Mode::Hiragana);
        ime.feed('k');
        let flushed = ime.set_mode(Mode::Katakana);
        assert!(flushed.consumed);
        assert_eq!(flushed.commit, "k", "the pending letter must not vanish");
        assert_eq!(ime.mode(), Mode::Katakana);
        assert_eq!(ime.preedit(), "");
    }

    // --- Hangul -----------------------------------------------------------

    #[test_case]
    fn hangul_composes_by_arithmetic() {
        // 가 = lead ㄱ (0), vowel ㅏ (0), no final.
        assert_eq!(hangul_syllable(0, 0, 0), Some('가'));
        // 한 = ㅎ (18), ㅏ (0), ㄴ (final 4).
        assert_eq!(hangul_syllable(18, 0, 4), Some('한'));
        // 글 = ㄱ (0), ㅡ (18), ㄹ (final 8).
        assert_eq!(hangul_syllable(0, 18, 8), Some('글'));
        // The last syllable in the block.
        assert_eq!(hangul_syllable(18, 20, 27), Some('힣'));
        // Out of range is None, not a wrapped codepoint.
        assert_eq!(hangul_syllable(19, 0, 0), None);
        assert_eq!(hangul_syllable(0, 21, 0), None);
        assert_eq!(hangul_syllable(0, 0, 28), None);
    }

    fn hangul(input: &str) -> String {
        let mut ime = Ime::new();
        // Bypass `set_mode_by_name`'s font gate: the engine is tested even though
        // the mode cannot be activated on a machine that cannot render it.
        ime.mode = Mode::Hangul;
        let mut out = String::new();
        for c in input.chars() {
            let r = ime.feed(c);
            if r.consumed {
                out.push_str(&r.commit);
            } else {
                out.push(c);
            }
        }
        out.push_str(&ime.flush().commit);
        out
    }

    #[test_case]
    fn dubeolsik_typing_produces_syllables() {
        // 한글: ㅎㅏㄴ / ㄱㅡㄹ = g,k,s / r,m,f
        assert_eq!(hangul("gksrmf"), "한글");
        // 가나다
        assert_eq!(hangul("rkskek"), "가나다");
        // A lone lead renders as the jamo rather than nothing.
        assert_eq!(hangul("r"), "ᄀ");
    }

    #[test_case]
    fn hangul_merges_compound_vowels_and_finals() {
        // ㅗ + ㅏ = ㅘ, so 과 is r,h,k.
        assert_eq!(hangul("rhk"), "과");
        // ㅡ + ㅣ = ㅢ.
        assert_eq!(hangul("dml"), "의");
        // ㄱ + ㅅ = ㄳ: 삯 is t,k,r,t.
        assert_eq!(hangul("tkrt"), "삯");
        // ㄹ + ㅎ = ㅀ: 싫 is t,l,f,g.
        assert_eq!(hangul("tlfg"), "싫");
    }

    #[test_case]
    fn a_vowel_after_a_final_steals_it_into_the_next_syllable() {
        // 안 + ㅏ must become 아 + 나, which is what makes typing 아나 possible.
        // g,k,s,k = ㅎㅏㄴㅏ → 하나
        assert_eq!(hangul("gksk"), "하나");
    }

    #[test_case]
    fn a_non_jamo_key_flushes_and_passes_through() {
        // The same rule kana follows, and for the same reason: `/commands`.
        assert_eq!(hangul("rk1"), "가1");
        assert_eq!(hangul("/rk"), "/가");
    }

    #[test_case]
    fn every_lead_that_can_be_a_final_round_trips() {
        for l in 0..19usize {
            if let Some(f) = lead_to_final(l) {
                assert_eq!(final_to_lead(f), Some(l), "lead {l} -> final {f} did not round-trip");
            }
        }
        // The three tense consonants that are never finals.
        for l in [4usize, 8, 13] {
            assert_eq!(lead_to_final(l), None, "lead {l} must not be a final");
        }
    }

    // --- refusals ---------------------------------------------------------

    #[test_case]
    fn pinyin_is_refused_with_the_dictionary_reason() {
        for name in ["pinyin", "chinese", "zh", "hanzi", "kanji"] {
            let e = set_mode_by_name(name).expect_err("must be refused");
            assert!(
                e.contains("dictionary"),
                "the refusal must name the obstacle, got: {e}"
            );
        }
        assert_eq!(mode(), Mode::Off, "a refused mode must not become active");
    }

    #[test_case]
    fn hangul_is_refused_with_the_font_reason_not_reported_as_unknown() {
        let e = set_mode_by_name("hangul").expect_err("no Hangul face is bundled");
        assert!(e.contains("Hangul"), "{e}");
        assert!(
            e.contains("render") || e.contains("tofu") || e.contains("glyph"),
            "the refusal must be about rendering, not about the mode being unknown: {e}"
        );
        assert_eq!(mode(), Mode::Off);
    }

    #[test_case]
    fn the_deliverable_modes_activate_and_off_restores() {
        assert_eq!(set_mode_by_name("hiragana"), Ok(Mode::Hiragana));
        assert_eq!(mode(), Mode::Hiragana);
        assert_eq!(set_mode_by_name("katakana"), Ok(Mode::Katakana));
        assert_eq!(set_mode_by_name("off"), Ok(Mode::Off));
        assert_eq!(mode(), Mode::Off);
        assert!(set_mode_by_name("klingon").is_err());
    }

    #[test_case]
    fn the_romaji_table_has_no_duplicate_sequence() {
        for (i, (a, _)) in ROMAJI.iter().enumerate() {
            for (b, _) in &ROMAJI[i + 1..] {
                assert_ne!(a, b, "ROMAJI lists {a:?} twice");
            }
        }
    }

    /// Every kana this engine can produce must be in a block the bundled face
    /// covers, or it composes correctly and renders as tofu — the exact failure
    /// the Hangul mode is gated against.
    #[test_case]
    fn every_kana_the_table_produces_is_in_the_bundled_range() {
        for (r, kana) in ROMAJI {
            for c in kana.chars() {
                let u = c as u32;
                let ok = (0x3041..=0x30FF).contains(&u)  // hiragana + katakana
                    || (0x3000..=0x303F).contains(&u)     // CJK punctuation
                    || (0xFF00..=0xFF60).contains(&u);    // fullwidth forms
                assert!(ok, "romaji {r:?} produces U+{u:04X}, outside the bundled kana range");
            }
            // And the katakana form must stay in range too.
            for c in to_katakana(kana).chars() {
                let u = c as u32;
                let ok = (0x3041..=0x30FF).contains(&u)
                    || (0x3000..=0x303F).contains(&u)
                    || (0xFF00..=0xFF60).contains(&u);
                assert!(ok, "katakana of {r:?} produces U+{u:04X}, out of range");
            }
        }
    }
}
