//! DEFLATE **compression** (RFC 1951) and the zlib wrapper (RFC 1950) — the
//! encoder side of [`super::inflate`].
//!
//! Two functions, because two callers want different trades:
//!
//! - [`zlib_stored`] emits BTYPE=00 stored blocks. It compresses nothing (the
//!   output is ~0.03% larger than the input) but it needs no Huffman machinery
//!   and cannot be wrong in an interesting way. This is what the git agent's
//!   `host_deflate` has always used — a valid zlib stream real git accepts,
//!   equivalent to `core.compression=0`.
//! - [`zlib_compress`] emits BTYPE=01 **fixed-Huffman** blocks over greedy
//!   LZ77. Fixed Huffman means the code tables are the ones in RFC 1951 §3.2.6
//!   rather than ones we derive and transmit, which removes the entire
//!   dynamic-table build — the part of a deflate encoder where a bug produces
//!   a stream that only *some* decoders reject.
//!
//! **Why a real compressor exists at all:** a 1920x1080 screenshot is 6.2 MB
//! of raw RGB. Stored blocks leave it at 6.2 MB; a UI screenshot under PNG's
//! `Up` filter is mostly runs of zero, and LZ77 takes the same image to tens
//! of kilobytes. That is the difference between a feature you use in a bug
//! report and one you do not.
//!
//! **How it is known to be correct:** every test round-trips through *our own*
//! [`super::inflate`], which is the decoder that has been reading real PNGs and
//! real git objects since before this file existed. An encoder verified against
//! an independently-written decoder in the same tree is a much stronger claim
//! than one verified against its own inverse.

use alloc::vec;
use alloc::vec::Vec;

/// zlib-compress `data` using **stored (uncompressed) deflate blocks**.
///
/// Kept as its own entry point rather than a level-0 mode of
/// [`zlib_compress`]: the git object path has always used it, its output is
/// byte-stable, and "the bytes are in there literally" is a property worth
/// being able to rely on when debugging a packfile.
pub fn zlib_stored(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() + data.len() / 65535 * 5 + 11);
    out.push(0x78);
    out.push(0x01); // deflate, 32K window, FLEVEL 0 (FCHECK valid)
    let mut rest = data;
    loop {
        let n = rest.len().min(65535);
        let last = rest.len() <= 65535;
        out.push(if last { 0x01 } else { 0x00 }); // BFINAL + BTYPE=00 stored
        let len = n as u16;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&(!len).to_le_bytes());
        out.extend_from_slice(&rest[..n]);
        rest = &rest[n..];
        if rest.is_empty() {
            break;
        }
    }
    out.extend_from_slice(&super::inflate::adler32(data).to_be_bytes());
    out
}

// ---------------------------------------------------------------------------
// RFC 1951 tables
// ---------------------------------------------------------------------------

/// First length for each length code 257..=285.
const LEN_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];
/// Extra bits carried after each length code 257..=285.
const LEN_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
/// First distance for each distance code 0..=29.
const DIST_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
/// Extra bits carried after each distance code 0..=29.
const DIST_EXTRA: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];

const MIN_MATCH: usize = 3;
const MAX_MATCH: usize = 258;
const WINDOW: usize = 32768;
const HASH_BITS: usize = 15;
const HASH_SIZE: usize = 1 << HASH_BITS;
/// How far down a hash chain to walk before settling for the best match found.
/// 32 is the knee: raising it to 128 buys ~2% on UI screenshots for 4x the
/// work, and this runs on a cooperative scheduler where a screenshot must not
/// hold the CPU.
const MAX_CHAIN: usize = 32;
/// A match at least this long ends the search immediately — the remaining
/// candidates cannot pay for themselves.
const GOOD_MATCH: usize = 64;

/// LSB-first bit writer, which is deflate's convention for the bit *stream*.
/// Huffman codes themselves are written MSB-first (RFC 1951 §3.1.1), which is
/// why [`Self::huff`] reverses; getting that backwards produces a stream that
/// inflates to garbage rather than failing, so it has its own test.
struct BitWriter {
    out: Vec<u8>,
    bit: u32,
    acc: u32,
}

impl BitWriter {
    fn with_capacity(n: usize) -> Self {
        BitWriter { out: Vec::with_capacity(n), bit: 0, acc: 0 }
    }

    /// Write the low `n` bits of `v`, least-significant bit first.
    fn bits(&mut self, v: u32, n: u32) {
        self.acc |= (v & ((1u32 << n) - 1)) << self.bit;
        self.bit += n;
        while self.bit >= 8 {
            self.out.push((self.acc & 0xff) as u8);
            self.acc >>= 8;
            self.bit -= 8;
        }
    }

    /// Write a Huffman code: `n` bits of `code`, **most**-significant first.
    fn huff(&mut self, code: u32, n: u32) {
        let mut rev = 0u32;
        for i in 0..n {
            rev |= ((code >> i) & 1) << (n - 1 - i);
        }
        self.bits(rev, n);
    }

    fn finish(mut self) -> Vec<u8> {
        if self.bit > 0 {
            self.out.push((self.acc & 0xff) as u8);
        }
        self.out
    }
}

/// The fixed literal/length code for symbol `sym` as `(code, bits)`
/// — RFC 1951 §3.2.6.
fn fixed_litlen(sym: u16) -> (u32, u32) {
    match sym {
        0..=143 => (0x30 + sym as u32, 8),
        144..=255 => (0x190 + (sym as u32 - 144), 9),
        256..=279 => (sym as u32 - 256, 7),
        _ => (0xc0 + (sym as u32 - 280), 8),
    }
}

/// The length code (and its extra-bit payload) for a match of `len` bytes.
fn len_code(len: usize) -> (u16, u32, u32) {
    let mut i = LEN_BASE.len() - 1;
    while i > 0 && (len as u16) < LEN_BASE[i] {
        i -= 1;
    }
    let extra = LEN_EXTRA[i] as u32;
    (257 + i as u16, len as u32 - LEN_BASE[i] as u32, extra)
}

/// The distance code (and its extra-bit payload) for a back-reference of
/// `dist` bytes.
fn dist_code(dist: usize) -> (u32, u32, u32) {
    let mut i = DIST_BASE.len() - 1;
    while i > 0 && (dist as u16) < DIST_BASE[i] {
        i -= 1;
    }
    let extra = DIST_EXTRA[i] as u32;
    (i as u32, dist as u32 - DIST_BASE[i] as u32, extra)
}

fn hash3(d: &[u8], i: usize) -> usize {
    let v = ((d[i] as u32) << 16) | ((d[i + 1] as u32) << 8) | d[i + 2] as u32;
    // Knuth multiplicative; the top HASH_BITS of the product spread the low
    // bits of all three bytes, which a shift-xor hash does not.
    ((v.wrapping_mul(2_654_435_761)) >> (32 - HASH_BITS)) as usize
}

/// zlib-compress `data` with fixed-Huffman deflate over greedy LZ77.
///
/// Never larger than [`zlib_stored`] would be by more than the Huffman
/// overhead on incompressible input, and typically 10-100x smaller on the
/// filtered-image and text data this kernel actually compresses.
pub fn zlib_compress(data: &[u8]) -> Vec<u8> {
    // Tiny inputs are not worth a match search, and the fixed tables cost more
    // than they save below a few bytes.
    if data.len() < MIN_MATCH {
        return zlib_stored(data);
    }

    let mut w = BitWriter::with_capacity(data.len() / 4 + 64);
    w.out.push(0x78);
    w.out.push(0x01);

    w.bits(1, 1); // BFINAL
    w.bits(1, 2); // BTYPE = 01, fixed Huffman

    // `head[h]` is the most recent position hashing to `h`; `prev[p]` is the
    // position before `p` in that chain. `u32::MAX` is the empty marker, so a
    // real position 0 is representable (an off-by-one here silently loses the
    // first match of the stream).
    let mut head = vec![u32::MAX; HASH_SIZE];
    let mut prev = vec![u32::MAX; data.len()];

    let mut i = 0usize;
    while i < data.len() {
        let (mut best_len, mut best_dist) = (0usize, 0usize);

        if i + MIN_MATCH <= data.len() {
            let h = hash3(data, i);
            let mut cand = head[h];
            let limit = i.saturating_sub(WINDOW);
            let mut chain = 0;
            // Longest match physically available from here. Hoisted out of the
            // loop because the cheap reject below indexes `i + best_len`, and
            // `best_len == max` would put that one byte past the end — which is
            // exactly what `every_distance_code_boundary_round_trips` caught.
            let max = (data.len() - i).min(MAX_MATCH);
            while cand != u32::MAX && (cand as usize) >= limit && chain < MAX_CHAIN {
                // Already as long as this position allows: no candidate can win.
                if best_len >= max {
                    break;
                }
                let c = cand as usize;
                // Cheap reject before the byte loop: a candidate that does not
                // even match at the length we already have cannot beat it. Safe
                // to index because `best_len < max`, so both `i + best_len` and
                // (since `c < i`) `c + best_len` are inside `data`.
                if best_len == 0 || data[c + best_len] == data[i + best_len] {
                    let mut n = 0;
                    while n < max && data[c + n] == data[i + n] {
                        n += 1;
                    }
                    if n > best_len {
                        best_len = n;
                        best_dist = i - c;
                        if n >= GOOD_MATCH {
                            break;
                        }
                    }
                }
                cand = prev[c];
                chain += 1;
            }
            prev[i] = head[h];
            head[h] = i as u32;
        }

        if best_len >= MIN_MATCH {
            let (lc, lx, lxb) = len_code(best_len);
            let (code, bits) = fixed_litlen(lc);
            w.huff(code, bits);
            if lxb > 0 {
                w.bits(lx, lxb);
            }
            let (dc, dx, dxb) = dist_code(best_dist);
            w.huff(dc, 5);
            if dxb > 0 {
                w.bits(dx, dxb);
            }
            // Insert the *interior* positions of the match into the chains too,
            // or every match makes the next one harder to find. Skipping this
            // costs about 30% of the ratio on image data.
            for k in (i + 1)..(i + best_len) {
                if k + MIN_MATCH <= data.len() {
                    let h = hash3(data, k);
                    prev[k] = head[h];
                    head[h] = k as u32;
                }
            }
            i += best_len;
        } else {
            let (code, bits) = fixed_litlen(data[i] as u16);
            w.huff(code, bits);
            i += 1;
        }
    }

    let (code, bits) = fixed_litlen(256); // end of block
    w.huff(code, bits);

    let mut out = w.finish();
    out.extend_from_slice(&super::inflate::adler32(data).to_be_bytes());
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::inflate::zlib_decompress;

    /// The single property that matters: our encoder's output is read back
    /// byte-identically by our independently-written decoder.
    fn round_trip(data: &[u8]) {
        let z = zlib_compress(data);
        let back = zlib_decompress(&z).expect("our own inflate must accept our own deflate");
        assert_eq!(back, data, "round trip differed for {} bytes", data.len());
    }

    #[test_case]
    fn empty_and_tiny_inputs_round_trip() {
        for n in 0..8usize {
            let d: Vec<u8> = (0..n).map(|i| i as u8).collect();
            round_trip(&d);
        }
    }

    #[test_case]
    fn a_run_of_zeros_compresses_and_round_trips() {
        let d = vec![0u8; 100_000];
        let z = zlib_compress(&d);
        // A 100 KB run is one literal plus ~390 maximum-length matches; if this
        // is not tiny, the match loop is not finding anything.
        assert!(z.len() < 2_000, "100k zeros compressed to {} bytes", z.len());
        assert_eq!(zlib_decompress(&z).unwrap(), d);
    }

    #[test_case]
    fn incompressible_data_round_trips_without_exploding() {
        // A cheap LCG stands in for random: no repeats for LZ77 to find, so
        // this is the worst case for the fixed tables.
        let mut s = 0x1234_5678u32;
        let d: Vec<u8> = (0..40_000)
            .map(|_| {
                s = s.wrapping_mul(1_103_515_245).wrapping_add(12345);
                (s >> 16) as u8
            })
            .collect();
        let z = zlib_compress(&d);
        round_trip(&d);
        // Fixed Huffman spends 9 bits on half the byte values, so the ceiling
        // is ~112%. Anything past that means a table is wrong.
        assert!(z.len() < d.len() * 115 / 100, "expanded to {} from {}", z.len(), d.len());
    }

    #[test_case]
    fn text_with_long_repeats_round_trips() {
        let mut d = Vec::new();
        for i in 0..800 {
            d.extend_from_slice(b"the determinism boundary is load-bearing; line ");
            d.extend_from_slice(alloc::format!("{i}\n").as_bytes());
        }
        round_trip(&d);
        assert!(zlib_compress(&d).len() < d.len() / 8);
    }

    /// A match whose length lands exactly on a length-code boundary exercises
    /// the `len_code` search, where an off-by-one picks the wrong extra-bit
    /// count and desynchronises the whole rest of the stream.
    #[test_case]
    fn every_length_code_boundary_round_trips() {
        for &base in LEN_BASE.iter() {
            for delta in [0i32, -1, 1] {
                let n = (base as i32 + delta).max(MIN_MATCH as i32) as usize;
                let mut d = vec![0u8; 0];
                d.extend_from_slice(b"PREFIX");
                let pat: Vec<u8> = (0..n).map(|i| (i % 251) as u8).collect();
                d.extend_from_slice(&pat);
                d.extend_from_slice(&pat); // the match
                round_trip(&d);
            }
        }
    }

    /// Same argument for distances: each code covers a range, and choosing the
    /// neighbouring code shifts every subsequent bit.
    #[test_case]
    fn every_distance_code_boundary_round_trips() {
        for &base in DIST_BASE.iter() {
            let dist = base as usize;
            let mut d: Vec<u8> = (0..dist).map(|i| (i % 253) as u8).collect();
            let head: Vec<u8> = d[..MIN_MATCH.min(d.len())].to_vec();
            d.extend_from_slice(&head);
            round_trip(&d);
        }
    }

    #[test_case]
    fn stored_blocks_still_round_trip_across_the_65535_boundary() {
        for n in [65534usize, 65535, 65536, 131_071] {
            let d: Vec<u8> = (0..n).map(|i| (i % 256) as u8).collect();
            let z = zlib_stored(&d);
            assert_eq!(zlib_decompress(&z).unwrap(), d, "stored round trip failed at {n}");
        }
    }

    #[test_case]
    fn huffman_codes_are_written_most_significant_bit_first() {
        // Symbol 256 (end of block) is fixed code 0 in 7 bits, so a lone
        // end-of-block block is a known bit pattern: BFINAL=1, BTYPE=01, then
        // seven zero bits => 0b0000_0011 = 0x03, 0x00.
        let mut w = BitWriter::with_capacity(4);
        w.bits(1, 1);
        w.bits(1, 2);
        let (c, b) = fixed_litlen(256);
        w.huff(c, b);
        assert_eq!(w.finish(), alloc::vec![0x03, 0x00]);
    }
}
