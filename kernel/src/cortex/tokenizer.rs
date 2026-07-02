//! Byte-level BPE tokenizer (GPT-2 / Qwen style) so the shell can turn typed
//! text into token ids for chat -- the encoder counterpart to `detokenize`.
//!
//! Encoding: map each input byte through the GPT-2 byte→unicode table (which
//! turns a space into `Ġ`, newline into `Ċ`, ...), then greedily apply the
//! GGUF's BPE `merges` by rank until no adjacent pair is mergeable, and look up
//! each resulting symbol in the vocab. This is the whole-string variant (no
//! regex pre-tokenization); validated to reproduce llama.cpp's ids on ordinary
//! text. Decoding reverses the byte map and reassembles UTF-8, buffering
//! partial multi-byte sequences so streaming output stays clean.

use crate::cortex::gguf::Gguf;
use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

pub struct Tokenizer {
    byte_to_unicode: [char; 256],
    unicode_to_byte: BTreeMap<char, u8>,
    vocab: BTreeMap<String, u32>, // token string -> id
    ranks: BTreeMap<String, u32>, // "<left> <right>" merge -> priority (lower = earlier)
    pub eos: u32,
    pub im_start: u32,
    pub im_end: u32,
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
        let special = |s: &str, dflt: u32| vocab.get(s).copied().unwrap_or(dflt);
        Self {
            eos: gguf.config.eos_token_id,
            im_start: special("<|im_start|>", 151644),
            im_end: special("<|im_end|>", gguf.config.eos_token_id),
            byte_to_unicode,
            unicode_to_byte,
            vocab,
            ranks,
        }
    }

    /// Encode `text` to token ids (whole-string byte-level BPE).
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }
        // Bytes → byte-unicode symbols (one char each).
        let mut syms: Vec<String> = Vec::with_capacity(text.len());
        for &b in text.as_bytes() {
            let mut s = String::new();
            s.push(self.byte_to_unicode[b as usize]);
            syms.push(s);
        }
        // Greedily merge the lowest-rank adjacent pair until none remain.
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
        // Symbols → ids (byte-level BPE guarantees each byte-symbol is in vocab).
        syms.iter().filter_map(|s| self.vocab.get(s).copied()).collect()
    }

    /// Map a token id's bytes into `out` (reversing the byte→unicode table).
    /// Bytes are appended raw; the caller reassembles UTF-8 (see [`Stream`]).
    pub fn token_bytes(&self, token: &str, out: &mut Vec<u8>) {
        for ch in token.chars() {
            if let Some(&b) = self.unicode_to_byte.get(&ch) {
                out.push(b);
            } else {
                // Not a byte-level char (e.g. a special token like <|im_end|>):
                // emit its UTF-8 as-is.
                let mut buf = [0u8; 4];
                out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
            }
        }
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
