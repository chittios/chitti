//! **Grapheme→phoneme (G2P)** for the KittenTTS frontend — turns English text
//! into the phoneme-id sequence the model's `input_ids` expects.
//!
//! KittenTTS phonemizes with eSpeak NG (IPA + stress marks) and maps each
//! phoneme character through a fixed 178-symbol vocabulary. eSpeak NG is a large
//! C engine (and the pure-Rust port is a big std crate that loads data files),
//! neither of which fits a bare-metal `no_std` kernel. Instead we embed a
//! **word→IPA lexicon generated from that same eSpeak oracle** (the top ~8k
//! English words, `en_lexicon.tsv`) and reproduce KittenTTS's exact tokenizer:
//! per-word lexicon lookup yields byte-identical ids to whole-text eSpeak for
//! covered words (verified against the reference), with a crude letter-to-sound
//! fallback for out-of-vocabulary words.

use crate::mm::Locked;
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

/// The exact KittenTTS phoneme vocabulary (index = token id).
static VOCAB: &str = include_str!("testdata/kitten_vocab.txt");
/// `word\tIPA` per line, sorted by word (eSpeak NG en-us, with stress).
static LEXICON: &str = include_str!("testdata/en_lexicon.tsv");

/// char → token id, built once from `VOCAB`.
static CHAR_ID: Locked<Option<BTreeMap<char, i64>>> = Locked::new(None);
/// word → IPA, built once from `LEXICON`.
static LEX: Locked<Option<BTreeMap<&'static str, &'static str>>> = Locked::new(None);

fn with_char_id<R>(f: impl FnOnce(&BTreeMap<char, i64>) -> R) -> R {
    CHAR_ID.with(|slot| {
        if slot.is_none() {
            let mut m = BTreeMap::new();
            for (i, c) in VOCAB.chars().enumerate() {
                m.entry(c).or_insert(i as i64); // first occurrence wins (matches enumerate)
            }
            *slot = Some(m);
        }
        f(slot.as_ref().unwrap())
    })
}

fn with_lex<R>(f: impl FnOnce(&BTreeMap<&'static str, &'static str>) -> R) -> R {
    LEX.with(|slot| {
        if slot.is_none() {
            let mut m = BTreeMap::new();
            for line in LEXICON.lines() {
                if let Some((w, p)) = line.split_once('\t') {
                    m.insert(w, p);
                }
            }
            *slot = Some(m);
        }
        f(slot.as_ref().unwrap())
    })
}

/// Crude letter-to-sound fallback for out-of-vocabulary words (names, etc.):
/// a rough per-letter IPA so unknown words still produce *some* phonemes.
fn letter_to_sound(word: &str, out: &mut String) {
    for c in word.chars() {
        let ipa = match c.to_ascii_lowercase() {
            'a' => "ˈæ",
            'e' => "ɛ",
            'i' => "ɪ",
            'o' => "ɑ",
            'u' => "ʌ",
            'y' => "i",
            'c' => "k",
            'q' => "k",
            'x' => "ks",
            'j' => "ʤ",
            other if other.is_ascii_alphabetic() => {
                out.push(other);
                continue;
            }
            _ => continue,
        };
        out.push_str(ipa);
    }
}

/// Is `c` a `\w` character (Unicode alphanumeric or underscore) — matches the
/// KittenTTS tokenizer's `\w+|[^\w\s]` split. IPA letters are alphabetic, so
/// they count as `\w`; stress marks (ˈˌ), `ː`, punctuation do not.
fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Phonemize `text`: split into `\w+`/punctuation tokens, map each word through
/// the lexicon (LTS fallback), keep punctuation, join with spaces — the same
/// per-word IPA string eSpeak produces for whole text after tokenization.
pub fn phonemize(text: &str) -> String {
    let mut out = String::new();
    with_lex(|lex| {
        let mut word = String::new();
        let mut flush = |word: &mut String, out: &mut String| {
            if !word.is_empty() {
                if !out.is_empty() {
                    out.push(' ');
                }
                match lex.get(word.as_str()) {
                    Some(ipa) => out.push_str(ipa),
                    None => letter_to_sound(word, out),
                }
                word.clear();
            }
        };
        for c in text.chars() {
            if is_word_char(c) {
                word.extend(c.to_lowercase());
            } else if c.is_whitespace() {
                flush(&mut word, &mut out);
            } else {
                flush(&mut word, &mut out);
                if !out.is_empty() {
                    out.push(' ');
                }
                out.push(c); // punctuation kept as its own token
            }
        }
        flush(&mut word, &mut out);
    });
    out
}

/// Full KittenTTS tokenization: `text` → phoneme string → re-tokenized+joined →
/// vocab-char filter → `[0, …, 0]` id sequence (the leading/trailing 0 are the
/// model's `$` boundary token).
pub fn text_to_ids(text: &str) -> Vec<i64> {
    let ph = phonemize(text);
    let joined = rejoin(&ph); // ' '.join(re.findall(r"\w+|[^\w\s]", ph))
    let mut ids = alloc::vec![0i64];
    with_char_id(|cid| {
        for c in joined.chars() {
            if let Some(&id) = cid.get(&c) {
                ids.push(id);
            }
        }
    });
    ids.push(0);
    ids
}

/// `' '.join(re.findall(r"\w+|[^\w\s]", s))` — group word-char runs, emit lone
/// symbols, single-space between tokens.
fn rejoin(s: &str) -> String {
    let mut toks: Vec<String> = Vec::new();
    let mut cur = String::new();
    for c in s.chars() {
        if is_word_char(c) {
            cur.push(c);
        } else {
            if !cur.is_empty() {
                toks.push(core::mem::take(&mut cur));
            }
            if !c.is_whitespace() {
                toks.push(c.into());
            }
        }
    }
    if !cur.is_empty() {
        toks.push(cur);
    }
    toks.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// Text → ids must match the reference KittenTTS tokenizer (whole-text
    /// eSpeak → vocab ids), for lexicon-covered phrases.
    #[test_case]
    fn text_to_ids_matches_reference() {
        assert_eq!(text_to_ids("hello world"), vec![0, 50, 83, 54, 156, 57, 135, 16, 65, 156, 87, 158, 54, 46, 0]);
        assert_eq!(text_to_ids("the voice"), vec![0, 81, 83, 16, 64, 156, 76, 102, 61, 0]);
    }
}
