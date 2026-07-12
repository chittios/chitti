//! **Minimal Indic text shaping** — pre-base (left-side) matra reordering.
//!
//! This is *not* a full OpenType shaper: there is no GSUB conjunct-ligature
//! substitution and no GPOS mark positioning (above/below vowel signs are not
//! stacked). With a glyph-per-codepoint rasterizer (fontdue) the single biggest
//! legibility win is reordering the **left-side vowel signs**, which Unicode
//! stores *after* their base consonant in logical order but which render to the
//! *left* of the base. Moving them ahead of the base consonant (or the whole
//! consonant conjunct) makes Devanagari/Bengali/Tamil/Gujarati/Gurmukhi/
//! Malayalam words read correctly left-to-right.
//!
//! Scripts whose vowel signs are purely above/below/right (Telugu, Kannada) need
//! no reordering here; they still render as real glyphs via the font fallback
//! chain, just without mark stacking. Full shaping is a documented follow-up.

use alloc::borrow::Cow;
use alloc::string::String;
use alloc::vec::Vec;

/// A pre-base (left-reordering) matra: renders to the LEFT of its base but is
/// encoded after it. Covers the bundled Indic scripts.
fn is_prebase_matra(c: char) -> bool {
    matches!(
        c as u32,
        0x093F                       // Devanagari sign I
        | 0x09BF | 0x09C7 | 0x09C8   // Bengali I, E, AI
        | 0x0A3F                     // Gurmukhi I
        | 0x0ABF                     // Gujarati I
        | 0x0BC6 | 0x0BC7 | 0x0BC8   // Tamil E, EE, AI
        | 0x0D46 | 0x0D47 | 0x0D48   // Malayalam E, EE, AI
    )
}

/// A dependent-consonant joiner (virama / halant) for the bundled scripts.
fn is_virama(c: char) -> bool {
    matches!(
        c as u32,
        0x094D | 0x09CD | 0x0A4D | 0x0ACD | 0x0B4D | 0x0BCD | 0x0C4D | 0x0CCD | 0x0D4D
    )
}

/// A nukta (dot below, modifies the preceding consonant).
fn is_nukta(c: char) -> bool {
    matches!(c as u32, 0x093C | 0x09BC | 0x0A3C | 0x0ABC | 0x0B3C | 0x0CBC)
}

/// A base consonant in one of the bundled Indic blocks. The consonant runs
/// begin at `+0x15` in each block and end at `+0x39` (with script-specific
/// gaps that are simply treated as "not a consonant" — harmless here).
fn is_consonant(c: char) -> bool {
    let u = c as u32;
    let blocks = [
        0x0900u32, // Devanagari
        0x0980,    // Bengali
        0x0A00,    // Gurmukhi
        0x0A80,    // Gujarati
        0x0B00,    // Oriya
        0x0B80,    // Tamil
        0x0C00,    // Telugu
        0x0C80,    // Kannada
        0x0D00,    // Malayalam
    ];
    blocks
        .iter()
        .any(|&b| u >= b + 0x15 && u <= b + 0x39)
        // plus the Devanagari/Bengali/Gurmukhi additional consonants (nukta forms)
        || matches!(u, 0x0958..=0x095F | 0x09DC..=0x09DF | 0x0A59..=0x0A5E)
}

/// Reorder pre-base matras for visual (left-to-right) rendering. Returns the
/// input unchanged (borrowed) when it contains no such matra — the common case,
/// so Latin/CJK/emoji text pays only a single scan.
pub fn shape(text: &str) -> Cow<'_, str> {
    if !text.chars().any(is_prebase_matra) {
        return Cow::Borrowed(text);
    }
    let chars: Vec<char> = text.chars().collect();
    let mut out: Vec<char> = Vec::with_capacity(chars.len());
    for &c in &chars {
        if is_prebase_matra(c) {
            let n = out.len();
            if n >= 1 && (is_consonant(out[n - 1]) || is_nukta(out[n - 1])) {
                // Base consonant is the tail; skip a nukta sitting on it.
                let mut base = n - 1;
                if is_nukta(out[base]) && base > 0 {
                    base -= 1;
                }
                // Extend back over `(virama consonant)` conjunct pairs so the
                // matra lands before the WHOLE cluster (e.g. प्र + ि → ि प्र).
                while base >= 2 && is_virama(out[base - 1]) && is_consonant(out[base - 2]) {
                    base -= 2;
                }
                out.insert(base, c);
            } else {
                // No preceding consonant (malformed) — leave in place.
                out.push(c);
            }
        } else {
            out.push(c);
        }
    }
    Cow::Owned(out.into_iter().collect::<String>())
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    #[test_case]
    fn ascii_and_latin_untouched() {
        assert!(matches!(shape("Hello, world!"), Cow::Borrowed(_)));
        assert_eq!(shape("Google"), "Google");
    }

    #[test_case]
    fn devanagari_i_matra_moves_before_base() {
        // हिन्दी : ह ि न ् द ी  →  ि ह न ् द ी  (the ि reorders before ह).
        let out = shape("\u{0939}\u{093F}\u{0928}\u{094D}\u{0926}\u{0940}").to_string();
        let cs: Vec<char> = out.chars().collect();
        assert_eq!(cs[0] as u32, 0x093F, "i-matra first");
        assert_eq!(cs[1] as u32, 0x0939, "base ह second");
        // The trailing post-base ी (0940) stays put.
        assert_eq!(*cs.last().unwrap() as u32, 0x0940);
    }

    #[test_case]
    fn separate_syllables_not_conjoined() {
        // कमि (क म ि): ि attaches to म only → क ि म, NOT ि क म.
        let out = shape("\u{0915}\u{092E}\u{093F}").to_string();
        let cs: Vec<char> = out.chars().collect();
        assert_eq!(cs[0] as u32, 0x0915, "क stays first");
        assert_eq!(cs[1] as u32, 0x093F, "matra before म");
        assert_eq!(cs[2] as u32, 0x092E, "म last");
    }

    #[test_case]
    fn conjunct_matra_moves_before_cluster() {
        // प्रि (प ् र ि): ि → before the whole प्र conjunct.
        let out = shape("\u{092A}\u{094D}\u{0930}\u{093F}").to_string();
        let cs: Vec<char> = out.chars().collect();
        assert_eq!(cs[0] as u32, 0x093F, "matra before conjunct");
        assert_eq!(cs[1] as u32, 0x092A); // प
        assert_eq!(cs[2] as u32, 0x094D); // ्
        assert_eq!(cs[3] as u32, 0x0930); // र
    }

    #[test_case]
    fn tamil_e_matra_reorders() {
        // தெ (த ெ): ெ (0BC6) is pre-base → ெ த.
        let out = shape("\u{0BA4}\u{0BC6}").to_string();
        let cs: Vec<char> = out.chars().collect();
        assert_eq!(cs[0] as u32, 0x0BC6);
        assert_eq!(cs[1] as u32, 0x0BA4);
    }
}
