//! BPE tokenizer with two flavors, dispatched on the GGUF's
//! `tokenizer.ggml.model`, so the shell can turn typed text into token ids
//! for chat -- the encoder counterpart to `detokenize`.
//!
//! - **Gpt2** (`"gpt2"`, Qwen/Llama-3 style): map each input byte through the
//!   GPT-2 byte→unicode table (space→`Ġ`, newline→`Ċ`, ...), then greedily
//!   apply the GGUF's BPE `merges` by rank; every byte-symbol is in-vocab by
//!   construction. Whole-string (no regex pre-tokenization); validated to
//!   reproduce llama.cpp's ids on ordinary text.
//! - **Gemma** (`"gemma4"`): SPM-style BPE on **raw UTF-8** — spaces are
//!   escaped to `▁` (U+2581), the text splits into newline/non-newline runs
//!   (llama.cpp `PRE_TYPE_GEMMA4`: `[^\n]+|[\n]+`, all-newline runs looked up
//!   whole), then the same rank-merge loop runs over characters; a char with
//!   no vocab entry falls back to per-byte `<0xXX>` tokens.
//!
//! Decoding reverses the flavor's mapping and reassembles UTF-8, buffering
//! partial multi-byte sequences so streaming output stays clean.

use crate::cortex::gguf::Gguf;
use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// The tokenizer flavor, from `tokenizer.ggml.model`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    /// GPT-2 byte-level BPE (Qwen; the default).
    Gpt2,
    /// Gemma-4 raw-UTF-8 BPE with `▁` whitespace + `<0xXX>` byte fallback.
    Gemma,
}

pub struct Tokenizer {
    pub kind: Kind,
    byte_to_unicode: [char; 256],
    unicode_to_byte: BTreeMap<char, u8>,
    vocab: BTreeMap<String, u32>, // token string -> id
    ranks: BTreeMap<String, u32>, // "<left> <right>" merge -> priority (lower = earlier)
    pub eos: u32,
    pub im_start: u32,
    pub im_end: u32,
    pub think_open: u32,
    pub think_close: u32,
    /// Qwen tool-call delimiters (`<tool_call>` / `</tool_call>`), or u32::MAX
    /// when the vocab lacks them.
    pub tool_open: u32,
    pub tool_close: u32,
    /// Gemma turn delimiters (`<start_of_turn>` / `<end_of_turn>`), or
    /// u32::MAX when the vocab lacks them.
    pub turn_open: u32,
    pub turn_close: u32,
}

/// GPT-2 byte→unicode table: printable bytes map to themselves, the rest to
/// codepoints 256+ (so every byte becomes a visible char, e.g. space→`Ġ`).
fn build_byte_to_unicode() -> [char; 256] {
    let mut used = [false; 256];
    let mut map = ['\0'; 256];
    for b in 0u32..256 {
        let printable = (0x21..=0x7e).contains(&b) || (0xa1..=0xac).contains(&b) || (0xae..=0xff).contains(&b);
        if printable {
            map[b as usize] = char::from_u32(b).unwrap();
            used[b as usize] = true;
        }
    }
    let mut n = 0u32;
    for b in 0..256 {
        if !used[b] {
            map[b] = char::from_u32(256 + n).unwrap();
            n += 1;
        }
    }
    map
}

impl Tokenizer {
    /// Build the tokenizer from a parsed GGUF's vocab + merges. Allocates owned
    /// lookup maps (~40 MiB for the 9B's 248 K vocab / 247 K merges), so build
    /// it once and keep it.
    pub fn build(gguf: &Gguf) -> Self {
        let byte_to_unicode = build_byte_to_unicode();
        let mut unicode_to_byte = BTreeMap::new();
        for (b, &c) in byte_to_unicode.iter().enumerate() {
            unicode_to_byte.insert(c, b as u8);
        }
        let mut vocab = BTreeMap::new();
        for (i, &t) in gguf.tokens.iter().enumerate() {
            vocab.insert(t.to_string(), i as u32);
        }
        let mut ranks = BTreeMap::new();
        for (i, &m) in gguf.merges.iter().enumerate() {
            ranks.insert(m.to_string(), i as u32);
        }
        let kind = match gguf.tokenizer_model {
            "gemma4" => Kind::Gemma,
            _ => Kind::Gpt2,
        };
        let special = |s: &str, dflt: u32| vocab.get(s).copied().unwrap_or(dflt);
        Self {
            kind,
            eos: gguf.config.eos_token_id,
            // Fallbacks are u32::MAX (an id no model has) so absent special
            // tokens are simply never emitted/matched, rather than colliding
            // with a real token. im_end falls back to eos (the stop set).
            im_start: special("<|im_start|>", u32::MAX),
            im_end: special("<|im_end|>", gguf.config.eos_token_id),
            think_open: special("<think>", u32::MAX),
            think_close: special("</think>", u32::MAX),
            tool_open: special("<tool_call>", u32::MAX),
            tool_close: special("</tool_call>", u32::MAX),
            turn_open: special("<start_of_turn>", u32::MAX),
            turn_close: special("<end_of_turn>", u32::MAX),
            byte_to_unicode,
            unicode_to_byte,
            vocab,
            ranks,
        }
    }

    /// Encode `text` to token ids, per the model's tokenizer flavor.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }
        match self.kind {
            Kind::Gpt2 => self.encode_gpt2(text),
            Kind::Gemma => self.encode_gemma(text),
        }
    }

    /// GPT-2 byte-level BPE (whole-string).
    fn encode_gpt2(&self, text: &str) -> Vec<u32> {
        // Bytes → byte-unicode symbols (one char each).
        let mut syms: Vec<String> = Vec::with_capacity(text.len());
        for &b in text.as_bytes() {
            let mut s = String::new();
            s.push(self.byte_to_unicode[b as usize]);
            syms.push(s);
        }
        self.merge_by_rank(&mut syms);
        // Symbols → ids (byte-level BPE guarantees each byte-symbol is in vocab).
        syms.iter().filter_map(|s| self.vocab.get(s).copied()).collect()
    }

    /// Gemma-4 raw-UTF-8 BPE: `▁` whitespace escaping, newline-run splitting,
    /// rank merges over characters, `<0xXX>` byte fallback for OOV symbols.
    fn encode_gemma(&self, text: &str) -> Vec<u32> {
        // Normalize: escape spaces to ▁ (llama.cpp `escape_whitespaces`).
        let escaped: String = text.chars().map(|c| if c == ' ' { '\u{2581}' } else { c }).collect();
        let mut out = Vec::new();
        // Pre-split `[^\n]+|[\n]+` (PRE_TYPE_GEMMA4): BPE merges never cross a
        // newline-run boundary.
        let bytes = escaped.as_bytes();
        let mut start = 0usize;
        while start < bytes.len() {
            let is_nl = bytes[start] == b'\n';
            let mut end = start + 1;
            while end < bytes.len() && (bytes[end] == b'\n') == is_nl {
                end += 1;
            }
            let word = &escaped[start..end];
            start = end;
            if is_nl {
                // All-newline run: whole-token lookup first (the gemma vocab
                // carries multi-newline tokens; merges assert no newlines).
                if let Some(&id) = self.vocab.get(word) {
                    out.push(id);
                    continue;
                }
            }
            // Characters → symbols, merge by rank, then vocab with byte fallback.
            let mut syms: Vec<String> = word.chars().map(|c| c.to_string()).collect();
            self.merge_by_rank(&mut syms);
            for s in &syms {
                if let Some(&id) = self.vocab.get(s) {
                    out.push(id);
                } else {
                    for b in s.as_bytes() {
                        if let Some(&id) = self.vocab.get(format!("<0x{b:02X}>").as_str()) {
                            out.push(id);
                        }
                    }
                }
            }
        }
        out
    }

    /// Greedily merge the lowest-rank adjacent symbol pair until none remain
    /// (the shared BPE core; the flavors differ only in symbol alphabet).
    fn merge_by_rank(&self, syms: &mut Vec<String>) {
        let mut pair = String::new();
        loop {
            let mut best_rank = u32::MAX;
            let mut best_i = usize::MAX;
            for i in 0..syms.len().saturating_sub(1) {
                pair.clear();
                pair.push_str(&syms[i]);
                pair.push(' ');
                pair.push_str(&syms[i + 1]);
                if let Some(&r) = self.ranks.get(&pair) {
                    if r < best_rank {
                        best_rank = r;
                        best_i = i;
                    }
                }
            }
            if best_i == usize::MAX {
                break;
            }
            let tail = syms.remove(best_i + 1);
            syms[best_i].push_str(&tail);
        }
    }

    /// Map a token id's bytes into `out` (reversing the flavor's encoding).
    /// Bytes are appended raw; the caller reassembles UTF-8 (see [`Stream`]).
    pub fn token_bytes(&self, token: &str, out: &mut Vec<u8>) {
        match self.kind {
            Kind::Gpt2 => {
                for ch in token.chars() {
                    if let Some(&b) = self.unicode_to_byte.get(&ch) {
                        out.push(b);
                    } else {
                        // Not a byte-level char (e.g. a special token like
                        // <|im_end|>): emit its UTF-8 as-is.
                        let mut buf = [0u8; 4];
                        out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                    }
                }
            }
            Kind::Gemma => {
                // `<0xXX>` byte tokens decode to the raw byte; ▁ → space;
                // everything else is already raw UTF-8.
                if let Some(b) = parse_byte_token(token) {
                    out.push(b);
                    return;
                }
                for ch in token.chars() {
                    if ch == '\u{2581}' {
                        out.push(b' ');
                    } else {
                        let mut buf = [0u8; 4];
                        out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                    }
                }
            }
        }
    }
}

/// Parse a `<0xXX>` byte-fallback token into its byte, if it is one.
pub fn parse_byte_token(token: &str) -> Option<u8> {
    let t = token.as_bytes();
    if t.len() == 6 && t.starts_with(b"<0x") && t[5] == b'>' {
        let hex = |c: u8| match c {
            b'0'..=b'9' => Some(c - b'0'),
            b'A'..=b'F' => Some(c - b'A' + 10),
            b'a'..=b'f' => Some(c - b'a' + 10),
            _ => None,
        };
        return Some(hex(t[3])? << 4 | hex(t[4])?);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// Build a tokenizer directly from token/merge lists (no GGUF needed).
    fn tok(kind: Kind, tokens: &[&str], merges: &[&str]) -> Tokenizer {
        let byte_to_unicode = build_byte_to_unicode();
        let mut unicode_to_byte = BTreeMap::new();
        for (b, &c) in byte_to_unicode.iter().enumerate() {
            unicode_to_byte.insert(c, b as u8);
        }
        let mut vocab = BTreeMap::new();
        for (i, t) in tokens.iter().enumerate() {
            vocab.insert(t.to_string(), i as u32);
        }
        let mut ranks = BTreeMap::new();
        for (i, m) in merges.iter().enumerate() {
            ranks.insert(m.to_string(), i as u32);
        }
        Tokenizer {
            kind,
            eos: 0,
            im_start: u32::MAX,
            im_end: 0,
            think_open: u32::MAX,
            think_close: u32::MAX,
            tool_open: u32::MAX,
            tool_close: u32::MAX,
            turn_open: u32::MAX,
            turn_close: u32::MAX,
            byte_to_unicode,
            unicode_to_byte,
            vocab,
            ranks,
        }
    }

    /// Gemma flavor: spaces escape to ▁, rank merges run on raw chars, and a
    /// leading-space word merges into one `▁hello` token.
    #[test_case]
    fn gemma_encodes_with_whitespace_marker() {
        let t = tok(
            Kind::Gemma,
            &["h", "e", "l", "o", "\u{2581}", "he", "ll", "llo", "hello", "\u{2581}hello"],
            &["h e", "l l", "ll o", "he llo", "\u{2581} hello"],
        );
        assert_eq!(t.encode("hello"), vec![8]);
        assert_eq!(t.encode(" hello"), vec![9]);
    }

    /// Gemma flavor: newline runs never merge with text; an all-newline run
    /// resolves as a whole vocab token when present.
    #[test_case]
    fn gemma_newline_runs_split_and_lookup_whole() {
        let t = tok(Kind::Gemma, &["a", "b", "\n", "\n\n"], &[]);
        assert_eq!(t.encode("a\n\nb"), vec![0, 3, 1]);
        // Three newlines: no "\n\n\n" token → falls back to per-char BPE.
        assert_eq!(t.encode("a\n\n\nb"), vec![0, 2, 2, 2, 1]);
    }

    /// Gemma flavor: an out-of-vocab character falls back to <0xXX> byte
    /// tokens (one per UTF-8 byte).
    #[test_case]
    fn gemma_byte_fallback_for_oov() {
        let t = tok(Kind::Gemma, &["a", "<0x7A>", "<0xC3>", "<0xA9>"], &[]);
        assert_eq!(t.encode("az"), vec![0, 1]); // 'z' = 0x7A
        assert_eq!(t.encode("é"), vec![2, 3]); // U+00E9 = C3 A9
    }

    /// Gemma decode: ▁ becomes a space, <0xXX> becomes the raw byte.
    #[test_case]
    fn gemma_token_bytes_roundtrip() {
        let t = tok(Kind::Gemma, &[], &[]);
        let mut out = Vec::new();
        t.token_bytes("\u{2581}hello", &mut out);
        assert_eq!(out, b" hello");
        out.clear();
        t.token_bytes("<0x41>", &mut out);
        assert_eq!(out, b"A");
        assert_eq!(parse_byte_token("<0x7a>"), Some(0x7a));
        assert_eq!(parse_byte_token("<0xZZ>"), None);
        assert_eq!(parse_byte_token("hello"), None);
    }

    /// GPT-2 flavor regression: byte-symbols map through the Ġ table and the
    /// same merge loop (space+h+i merging into one ` hi` token).
    #[test_case]
    fn gpt2_byte_level_bpe_unchanged() {
        let t = tok(Kind::Gpt2, &["h", "i", "Ġ", "hi", "Ġhi"], &["h i", "Ġ hi"]);
        assert_eq!(t.encode("hi"), vec![3]);
        assert_eq!(t.encode(" hi"), vec![4]);
        let mut out = Vec::new();
        t.token_bytes("Ġhi", &mut out);
        assert_eq!(out, b" hi");
    }
}

/// Incremental UTF-8 assembler for streamed decode: token bytes may split a
/// multi-byte char, so buffer bytes and only emit complete UTF-8.
pub struct Stream {
    pending: Vec<u8>,
}

impl Stream {
    pub fn new() -> Self {
        Self { pending: Vec::new() }
    }

    /// Append `token`'s bytes and return the longest now-complete UTF-8 text,
    /// keeping any trailing incomplete byte sequence buffered.
    pub fn push(&mut self, tok: &Tokenizer, token: &str) -> String {
        tok.token_bytes(token, &mut self.pending);
        // Longest valid UTF-8 prefix.
        let valid = match core::str::from_utf8(&self.pending) {
            Ok(_) => self.pending.len(),
            Err(e) => e.valid_up_to(),
        };
        let out = String::from_utf8_lossy(&self.pending[..valid]).into_owned();
        self.pending.drain(..valid);
        out
    }
}

impl Default for Stream {
    fn default() -> Self {
        Self::new()
    }
}
