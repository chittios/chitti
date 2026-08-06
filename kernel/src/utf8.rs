//! Incremental UTF-8 decoding: one byte in, a `char` out when a sequence lands.
//!
//! Extracted from `framebuffer::Pane::feed_utf8` rather than duplicated. That
//! decoder was correct but lived in a module the test build does not compile
//! (`framebuffer` is `#[cfg(not(test))]`), so its invalid-byte and
//! incomplete-sequence branches had zero coverage — and the input side of the
//! OS needs exactly the same logic, because a keypress that emits `é` arrives at
//! `read_line` as two bytes and must insert **one** character.
//!
//! The rule it follows is `core::str::from_utf8`'s own: `error_len() == None`
//! means "incomplete, feed me more", `Some(_)` means "invalid, resynchronise".
//! Hand-rolling the continuation-byte arithmetic instead is where overlong
//! encodings and surrogate halves get let through.

/// A 4-byte incremental UTF-8 decoder.
#[derive(Default, Clone, Copy)]
pub struct Utf8Decoder {
    buf: [u8; 4],
    len: u8,
}

impl Utf8Decoder {
    pub const fn new() -> Self {
        Utf8Decoder { buf: [0; 4], len: 0 }
    }

    /// Feed one byte. Returns the decoded `char` once a full sequence lands,
    /// `None` while one is still arriving, and `U+FFFD` for an invalid byte.
    ///
    /// **Documented limitation:** a byte that *invalidates a partial sequence* is
    /// consumed along with it, so `E6 97 78` yields one `U+FFFD` and the `x` is
    /// lost. WHATWG's decoder would re-emit that byte; doing so here needs an
    /// output queue, since one input byte would then produce two characters, and
    /// a one-in-one-out API is worth more than exact recovery from corrupt input.
    /// Nothing that generates bytes for this decoder produces malformed UTF-8 — a
    /// keymap or IME emits well-formed sequences by construction, and so does a
    /// host terminal — so the only reachable case is a corrupt paste, where the
    /// cost is one dropped character rather than a panic or a corrupted line. An
    /// isolated bad byte (the common case) is recovered exactly.
    pub fn feed(&mut self, b: u8) -> Option<char> {
        if self.len == 0 && b < 0x80 {
            return Some(b as char); // ASCII fast path
        }
        if self.len as usize >= self.buf.len() {
            self.len = 0; // never overflow the buffer
        }
        self.buf[self.len as usize] = b;
        self.len += 1;
        match core::str::from_utf8(&self.buf[..self.len as usize]) {
            Ok(s) => {
                self.len = 0;
                s.chars().next()
            }
            Err(e) if e.error_len().is_none() => None, // incomplete — await more
            Err(_) => {
                // Invalid. Resynchronise by dropping everything buffered: keeping
                // the offending byte would make the *next* good lead byte fail
                // too, turning one bad byte into a run of replacements.
                self.len = 0;
                Some('\u{FFFD}')
            }
        }
    }

    /// Whether a multi-byte sequence is part-way through.
    pub fn pending(&self) -> bool {
        self.len > 0
    }

    pub fn reset(&mut self) {
        self.len = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::String;

    /// Feed every byte of `s` and collect what comes out.
    fn decode(bytes: &[u8]) -> String {
        let mut d = Utf8Decoder::new();
        let mut out = String::new();
        for &b in bytes {
            if let Some(c) = d.feed(b) {
                out.push(c);
            }
        }
        out
    }

    #[test_case]
    fn ascii_passes_through_one_byte_at_a_time() {
        assert_eq!(decode(b"/help 42"), "/help 42");
    }

    #[test_case]
    fn multi_byte_chars_assemble_across_calls() {
        // 2-byte (é), 3-byte (日), 4-byte (emoji) — and nothing is emitted until
        // the last byte of each arrives.
        for s in ["é", "ü", "日", "本", "\u{1F600}", "héllo 日本 ✓"] {
            assert_eq!(decode(s.as_bytes()), s, "round trip failed for {s:?}");
        }
        let mut d = Utf8Decoder::new();
        let bytes = "\u{1F600}".as_bytes();
        assert_eq!(bytes.len(), 4);
        assert_eq!(d.feed(bytes[0]), None);
        assert!(d.pending());
        assert_eq!(d.feed(bytes[1]), None);
        assert_eq!(d.feed(bytes[2]), None);
        assert_eq!(d.feed(bytes[3]), Some('\u{1F600}'));
        assert!(!d.pending());
    }

    #[test_case]
    fn an_invalid_byte_is_a_replacement_and_the_stream_resynchronises() {
        // A stray continuation byte, then good text.
        assert_eq!(decode(&[0x80, b'a', b'b']), "\u{FFFD}ab");
        // A truncated 3-byte sequence followed by ASCII: one replacement, and
        // the `x` is consumed with the invalid sequence — the documented
        // limitation on `feed`. Pinned so nobody "fixes" it by accident, and so
        // the cost is written down rather than discovered.
        assert_eq!(decode(&[0xe6, 0x97, b'x']), "\u{FFFD}");
        // The stream still recovers immediately afterwards.
        assert_eq!(decode(&[0xe6, 0x97, b'x', b'y']), "\u{FFFD}y");
        // An overlong encoding of '/' must not decode as '/'.
        let out = decode(&[0xc0, 0xaf]);
        assert!(!out.contains('/'), "overlong encoding decoded: {out:?}");
        // A surrogate half must not decode.
        let out = decode(&[0xed, 0xa0, 0x80]);
        assert!(!out.is_empty() && out.chars().all(|c| c == '\u{FFFD}'), "{out:?}");
    }

    #[test_case]
    fn a_lone_bad_byte_does_not_poison_the_following_character() {
        // The resynchronise-by-clearing rule: one bad byte must cost exactly one
        // replacement, not a run of them.
        let out = decode(&[0xff, 0xc3, 0xa9]); // garbage, then 'é'
        assert_eq!(out, "\u{FFFD}é");
    }

    #[test_case]
    fn reset_drops_a_partial_sequence() {
        let mut d = Utf8Decoder::new();
        assert_eq!(d.feed(0xe6), None);
        d.reset();
        assert!(!d.pending());
        assert_eq!(d.feed(b'a'), Some('a'), "the buffer must be clean after a reset");
    }

    #[test_case]
    fn a_four_byte_overflow_cannot_wedge_the_decoder() {
        // Five lead bytes in a row: an implementation that only grew the buffer
        // would overflow it. This one restarts, so it keeps making progress —
        // which is the property that matters. (The byte that finally invalidates
        // the run is consumed with it, per `feed`'s documented limitation, so
        // recovery costs one character and not the rest of the line.)
        let mut d = Utf8Decoder::new();
        for _ in 0..5 {
            let _ = d.feed(0xf0);
        }
        let mut got = alloc::string::String::new();
        for &b in b"zz" {
            if let Some(c) = d.feed(b) {
                got.push(c);
            }
        }
        assert!(got.ends_with('z'), "the decoder must recover, got {got:?}");
        assert!(!d.pending(), "and must not be left mid-sequence");
    }
}
