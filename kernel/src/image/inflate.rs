//! RFC 1951 DEFLATE + RFC 1950 zlib decompression — the pure decoder behind
//! PNG's IDAT stream. Canonical-Huffman decode in the classic bit-serial
//! style (Mark Adler's `puff`), bounds-checked throughout: malformed input
//! returns `Err`, never panics or reads out of bounds.

use alloc::vec::Vec;

/// LSB-first bit reader over the compressed stream.
struct Bits<'a> {
    src: &'a [u8],
    pos: usize, // next byte
    bit: u32,   // bits already consumed from src[pos]
}

impl<'a> Bits<'a> {
    fn bits(&mut self, n: u32) -> Result<u32, &'static str> {
        let mut v = 0u32;
        for i in 0..n {
            let Some(&byte) = self.src.get(self.pos) else { return Err("deflate: truncated stream") };
            v |= (((byte >> self.bit) & 1) as u32) << i;
            self.bit += 1;
            if self.bit == 8 {
                self.bit = 0;
                self.pos += 1;
            }
        }
        Ok(v)
    }

    /// Discard partial bits so the cursor sits on a byte boundary (stored blocks).
    fn align(&mut self) {
        if self.bit != 0 {
            self.bit = 0;
            self.pos += 1;
        }
    }
}

/// A canonical Huffman table: `counts[len]` codes of each bit length and the
/// symbols in code order.
struct Huff {
    counts: [u16; 16],
    symbols: Vec<u16>,
}

impl Huff {
    /// Build from per-symbol code lengths (0 = unused symbol).
    fn build(lengths: &[u8]) -> Result<Huff, &'static str> {
        let mut counts = [0u16; 16];
        for &l in lengths {
            if l > 15 {
                return Err("deflate: code length > 15");
            }
            counts[l as usize] += 1;
        }
        // Over-subscribed tables are invalid (incomplete ones occur for the
        // single-distance-code case and are fine to decode).
        let mut left = 1i32;
        for len in 1..16 {
            left <<= 1;
            left -= counts[len] as i32;
            if left < 0 {
                return Err("deflate: over-subscribed huffman table");
            }
        }
        let mut offs = [0u16; 16];
        for len in 1..15 {
            offs[len + 1] = offs[len] + counts[len];
        }
        let mut symbols = alloc::vec![0u16; lengths.len()];
        for (sym, &l) in lengths.iter().enumerate() {
            if l != 0 {
                symbols[offs[l as usize] as usize] = sym as u16;
                offs[l as usize] += 1;
            }
        }
        counts[0] = 0;
        Ok(Huff { counts, symbols })
    }

    /// Decode one symbol, reading bits MSB-of-code-first (deflate order).
    fn decode(&self, br: &mut Bits<'_>) -> Result<u16, &'static str> {
        let mut code = 0i32;
        let mut first = 0i32;
        let mut index = 0i32;
        for len in 1..16 {
            code |= br.bits(1)? as i32;
            let count = self.counts[len] as i32;
            if code - first < count {
                return self
                    .symbols
                    .get((index + (code - first)) as usize)
                    .copied()
                    .ok_or("deflate: symbol out of range");
            }
            index += count;
            first = (first + count) << 1;
            code <<= 1;
        }
        Err("deflate: invalid huffman code")
    }
}

// Length codes 257..=285: base lengths + extra bits (RFC 1951 §3.2.5).
const LEN_BASE: [u16; 29] = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258];
const LEN_EXTRA: [u8; 29] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
// Distance codes 0..=29.
const DIST_BASE: [u16; 30] =
    [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577];
const DIST_EXTRA: [u8; 30] = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];

/// Sanity cap on the decompressed size (a PNG IDAT for a huge screen is tens
/// of MB; a corrupt stream must not eat the whole heap).
const OUT_MAX: usize = 256 << 20;

/// Decompress a raw DEFLATE stream.
pub fn inflate(src: &[u8]) -> Result<Vec<u8>, &'static str> {
    let mut br = Bits { src, pos: 0, bit: 0 };
    let mut out: Vec<u8> = Vec::new();
    loop {
        let bfinal = br.bits(1)?;
        let btype = br.bits(2)?;
        match btype {
            0 => {
                // Stored: LEN + ~LEN then raw bytes.
                br.align();
                if br.pos + 4 > src.len() {
                    return Err("deflate: truncated stored header");
                }
                let len = src[br.pos] as usize | ((src[br.pos + 1] as usize) << 8);
                let nlen = src[br.pos + 2] as usize | ((src[br.pos + 3] as usize) << 8);
                if len != (!nlen & 0xffff) {
                    return Err("deflate: stored length check failed");
                }
                br.pos += 4;
                if br.pos + len > src.len() {
                    return Err("deflate: truncated stored block");
                }
                out.extend_from_slice(&src[br.pos..br.pos + len]);
                br.pos += len;
            }
            1 => {
                // Fixed tables (RFC 1951 §3.2.6).
                let mut lit = [0u8; 288];
                for (i, l) in lit.iter_mut().enumerate() {
                    *l = match i {
                        0..=143 => 8,
                        144..=255 => 9,
                        256..=279 => 7,
                        _ => 8,
                    };
                }
                let litt = Huff::build(&lit)?;
                let distt = Huff::build(&[5u8; 30])?;
                inflate_block(&mut br, &litt, &distt, &mut out)?;
            }
            2 => {
                // Dynamic tables: code-length code, then lit/dist lengths.
                let hlit = br.bits(5)? as usize + 257;
                let hdist = br.bits(5)? as usize + 1;
                let hclen = br.bits(4)? as usize + 4;
                const ORDER: [usize; 19] = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];
                let mut cl = [0u8; 19];
                for &o in ORDER.iter().take(hclen) {
                    cl[o] = br.bits(3)? as u8;
                }
                let clt = Huff::build(&cl)?;
                let mut lengths = alloc::vec![0u8; hlit + hdist];
                let mut i = 0;
                while i < lengths.len() {
                    let sym = clt.decode(&mut br)?;
                    match sym {
                        0..=15 => {
                            lengths[i] = sym as u8;
                            i += 1;
                        }
                        16 => {
                            if i == 0 {
                                return Err("deflate: repeat with no previous length");
                            }
                            let prev = lengths[i - 1];
                            let n = 3 + br.bits(2)? as usize;
                            for _ in 0..n {
                                if i >= lengths.len() {
                                    return Err("deflate: length repeat overflow");
                                }
                                lengths[i] = prev;
                                i += 1;
                            }
                        }
                        17 | 18 => {
                            let n = if sym == 17 { 3 + br.bits(3)? as usize } else { 11 + br.bits(7)? as usize };
                            if i + n > lengths.len() {
                                return Err("deflate: zero-run overflow");
                            }
                            i += n;
                        }
                        _ => return Err("deflate: bad code-length symbol"),
                    }
                }
                if lengths[256] == 0 {
                    return Err("deflate: no end-of-block code");
                }
                let litt = Huff::build(&lengths[..hlit])?;
                let distt = Huff::build(&lengths[hlit..])?;
                inflate_block(&mut br, &litt, &distt, &mut out)?;
            }
            _ => return Err("deflate: reserved block type"),
        }
        if bfinal == 1 {
            return Ok(out);
        }
    }
}

/// Decode one compressed block's literal/length+distance stream into `out`.
fn inflate_block(br: &mut Bits<'_>, litt: &Huff, distt: &Huff, out: &mut Vec<u8>) -> Result<(), &'static str> {
    loop {
        let sym = litt.decode(br)? as usize;
        match sym {
            0..=255 => out.push(sym as u8),
            256 => return Ok(()),
            257..=285 => {
                let li = sym - 257;
                let len = LEN_BASE[li] as usize + br.bits(LEN_EXTRA[li] as u32)? as usize;
                let d = distt.decode(br)? as usize;
                if d >= 30 {
                    return Err("deflate: bad distance code");
                }
                let dist = DIST_BASE[d] as usize + br.bits(DIST_EXTRA[d] as u32)? as usize;
                if dist == 0 || dist > out.len() {
                    return Err("deflate: distance beyond output");
                }
                if out.len() + len > OUT_MAX {
                    return Err("deflate: output too large");
                }
                // Byte-by-byte on purpose: overlapping copies (dist < len)
                // must replicate the just-written bytes.
                let start = out.len() - dist;
                for k in 0..len {
                    let b = out[start + k];
                    out.push(b);
                }
            }
            _ => return Err("deflate: bad literal/length symbol"),
        }
    }
}

/// Decompress an RFC 1950 zlib stream (2-byte header + deflate + Adler-32).
pub fn zlib_decompress(src: &[u8]) -> Result<Vec<u8>, &'static str> {
    if src.len() < 6 {
        return Err("zlib: stream too short");
    }
    let (cmf, flg) = (src[0], src[1]);
    if cmf & 0x0f != 8 {
        return Err("zlib: not deflate");
    }
    if (cmf as u16 * 256 + flg as u16) % 31 != 0 {
        return Err("zlib: header check failed");
    }
    if flg & 0x20 != 0 {
        return Err("zlib: preset dictionary unsupported");
    }
    let out = inflate(&src[2..src.len() - 4])?;
    // Verify Adler-32 (the trailing 4 bytes, big-endian).
    let tail = &src[src.len() - 4..];
    let want = u32::from_be_bytes([tail[0], tail[1], tail[2], tail[3]]);
    if adler32(&out) != want {
        return Err("zlib: adler-32 mismatch");
    }
    Ok(out)
}

/// Adler-32 checksum (RFC 1950 §8).
pub fn adler32(data: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for chunk in data.chunks(5552) {
        for &x in chunk {
            a += x as u32;
            b += a;
        }
        a %= 65521;
        b %= 65521;
    }
    (b << 16) | a
}

#[cfg(test)]
mod tests {
    use super::*;

    const MSG: &[u8] = &[116, 104, 101, 32, 113, 117, 105, 99, 107, 32, 98, 114, 111, 119, 110, 32, 102, 111, 120, 32, 106, 117, 109, 112, 115, 32, 111, 118, 101, 114, 32, 116, 104, 101, 32, 108, 97, 122, 121, 32, 100, 111, 103, 46, 32, 116, 104, 101, 32, 113, 117, 105, 99, 107, 32, 98, 114, 111, 119, 110, 32, 102, 111, 120, 46];
    // `zlib.compressobj(9, DEFLATED, -15, 9, Z_FIXED)` over MSG (fixed-Huffman raw deflate).
    const FIXED_DEFLATE: &[u8] = &[43, 201, 72, 85, 40, 44, 205, 76, 206, 86, 72, 42, 202, 47, 207, 83, 72, 203, 175, 80, 200, 42, 205, 45, 40, 86, 200, 47, 75, 45, 82, 40, 1, 74, 231, 36, 86, 85, 42, 164, 228, 167, 235, 129, 121, 104, 138, 245, 0];
    // `zlib.compress(MSG * 4, 9)` (dynamic-Huffman zlib stream).
    const DYN_ZLIB: &[u8] = &[120, 218, 237, 213, 215, 90, 8, 0, 0, 128, 209, 74, 202, 172, 200, 142, 84, 168, 204, 108, 153, 165, 73, 139, 10, 89, 41, 171, 80, 25, 133, 180, 19, 162, 61, 208, 210, 162, 105, 71, 217, 209, 178, 10, 105, 209, 144, 81, 168, 236, 246, 66, 92, 248, 95, 194, 247, 245, 10, 231, 230, 8, 136, 138, 13, 30, 57, 102, 194, 52, 77, 61, 227, 181, 27, 183, 237, 218, 239, 29, 20, 22, 147, 120, 49, 237, 206, 243, 151, 175, 63, 124, 105, 236, 16, 24, 36, 37, 55, 126, 234, 156, 69, 154, 107, 204, 45, 119, 238, 115, 62, 228, 29, 157, 112, 33, 245, 118, 214, 227, 231, 239, 63, 55, 180, 255, 17, 233, 63, 72, 81, 105, 246, 66, 13, 93, 163, 53, 54, 123, 157, 60, 188, 2, 67, 163, 175, 222, 202, 124, 148, 255, 162, 242, 125, 91, 87, 207, 126, 146, 35, 100, 21, 23, 168, 235, 24, 174, 54, 179, 176, 57, 120, 44, 32, 36, 42, 254, 252, 213, 135, 207, 74, 94, 85, 127, 170, 111, 235, 59, 112, 184, 140, 194, 148, 89, 11, 150, 155, 108, 216, 106, 109, 231, 120, 240, 100, 100, 220, 185, 43, 55, 51, 30, 86, 84, 213, 253, 104, 253, 45, 220, 119, 180, 252, 228, 153, 243, 213, 150, 46, 223, 98, 101, 235, 224, 126, 212, 255, 228, 217, 148, 27, 247, 30, 60, 45, 174, 248, 222, 242, 171, 71, 159, 1, 195, 70, 207, 152, 183, 120, 201, 178, 85, 166, 91, 14, 184, 121, 250, 157, 56, 117, 230, 236, 221, 251, 79, 138, 202, 223, 213, 126, 23, 234, 45, 49, 84, 122, 220, 164, 25, 218, 6, 43, 215, 111, 222, 177, 231, 128, 239, 241, 136, 211, 201, 151, 175, 223, 45, 44, 123, 91, 243, 173, 249, 167, 208, 144, 81, 99, 39, 78, 159, 171, 170, 189, 110, 211, 246, 221, 246, 174, 71, 124, 99, 147, 46, 93, 75, 207, 201, 43, 252, 248, 181, 169, 83, 176, 151, 248, 144, 9, 211, 148, 85, 180, 244, 87, 172, 219, 181, 223, 229, 176, 79, 112, 120, 108, 218, 157, 236, 220, 130, 210, 55, 31, 59, 4, 254, 65, 45, 2, 234, 16, 80, 143, 129, 234, 15, 148, 17, 80, 161, 64, 85, 2, 37, 11, 148, 5, 80, 231, 129, 170, 7, 106, 22, 80, 142, 64, 101, 0, 37, 12, 212, 82, 160, 252, 129, 42, 6, 106, 24, 80, 166, 64, 157, 1, 170, 22, 168, 73, 64, 237, 1, 234, 58, 80, 63, 129, 82, 5, 234, 8, 80, 121, 64, 137, 3, 181, 2, 168, 112, 160, 222, 0, 53, 6, 168, 109, 64, 93, 4, 170, 17, 168, 57, 64, 57, 3, 149, 5, 148, 8, 80, 186, 64, 5, 2, 245, 2, 168, 17, 64, 153, 1, 21, 15, 212, 39, 160, 166, 0, 101, 7, 212, 77, 160, 126, 3, 165, 6, 212, 81, 160, 158, 2, 53, 0, 168, 85, 64, 157, 2, 234, 29, 80, 227, 128, 218, 1, 212, 101, 160, 154, 129, 154, 11, 148, 43, 80, 57, 64, 245, 2, 74, 31, 168, 96, 160, 74, 129, 26, 9, 212, 70, 160, 18, 129, 250, 2, 212, 84, 160, 246, 1, 117, 27, 168, 63, 64, 105, 0, 229, 5, 84, 62, 80, 146, 64, 173, 6, 42, 10, 168, 106, 160, 20, 128, 178, 6, 234, 10, 80, 173, 64, 205, 7, 202, 29, 168, 7, 64, 245, 1, 106, 25, 80, 39, 128, 42, 7, 74, 26, 168, 205, 64, 37, 3, 245, 13, 168, 233, 64, 217, 3, 149, 14, 148, 32, 80, 90, 64, 249, 0, 85, 0, 212, 96, 160, 214, 2, 21, 3, 212, 7, 160, 198, 3, 181, 19, 168, 84, 160, 218, 129, 90, 8, 148, 7, 80, 143, 128, 234, 7, 148, 33, 80, 33, 64, 189, 2, 74, 6, 168, 173, 64, 157, 3, 234, 7, 80, 51, 129, 114, 0, 234, 30, 80, 61, 128, 90, 2, 148, 31, 80, 69, 64, 13, 5, 106, 61, 80, 167, 129, 170, 1, 106, 34, 80, 187, 129, 186, 6, 84, 39, 80, 42, 64, 29, 6, 42, 23, 40, 49, 160, 140, 129, 10, 3, 234, 53, 80, 114, 64, 89, 2, 117, 1, 168, 6, 160, 102, 3, 229, 4, 84, 38, 80, 61, 129, 210, 1, 42, 0, 168, 18, 160, 134, 3, 181, 1, 168, 56, 160, 234, 128, 154, 12, 148, 45, 80, 55, 128, 250, 5, 212, 98, 160, 60, 129, 122, 2, 148, 4, 80, 43, 129, 138, 0, 234, 45, 80, 99, 129, 218, 14, 212, 37, 160, 154, 128, 82, 6, 202, 5, 168, 108, 160, 104, 77, 153, 214, 92, 104, 45, 155, 214, 68, 105, 77, 143, 214, 130, 104, 237, 37, 173, 73, 209, 154, 57, 173, 37, 208, 218, 103, 90, 83, 162, 181, 189, 180, 118, 139, 214, 186, 104, 77, 157, 214, 142, 209, 218, 51, 90, 27, 72, 107, 38, 180, 22, 73, 107, 85, 180, 38, 79, 107, 86, 180, 150, 66, 107, 45, 180, 54, 143, 214, 220, 104, 237, 62, 173, 245, 166, 53, 3, 90, 59, 78, 107, 101, 180, 54, 138, 214, 54, 209, 90, 18, 173, 125, 21, 232, 254, 191, 251, 255, 238, 255, 255, 187, 255, 255, 2, 200, 165, 214, 112];

    #[test_case]
    fn stored_block_roundtrip() {
        // Hand-built: BFINAL=1 BTYPE=00, align, LEN=5, ~LEN, "hello".
        let src = [0x01, 5, 0, 0xfa, 0xff, b'h', b'e', b'l', b'l', b'o'];
        assert_eq!(inflate(&src).unwrap(), b"hello");
    }

    #[test_case]
    fn fixed_huffman_stream() {
        assert_eq!(inflate(FIXED_DEFLATE).unwrap(), MSG);
    }

    #[test_case]
    fn dynamic_huffman_zlib_with_adler() {
        // 911-byte zlib stream whose first block is BTYPE=2 (dynamic); decodes
        // the synthetic ramp `(i*7 + (i>>3)*13) & 0xff` over 3000 bytes.
        let out = zlib_decompress(DYN_ZLIB).unwrap();
        assert_eq!(out.len(), 3000);
        for (i, &b) in out.iter().enumerate() {
            assert_eq!(b as usize, (i * 7 + (i >> 3) * 13) & 0xff);
        }
    }

    #[test_case]
    fn corrupt_streams_error_not_panic() {
        assert!(inflate(&[]).is_err());
        assert!(inflate(&[0x07]).is_err()); // reserved block type
        let mut bad = DYN_ZLIB.to_vec();
        let n = bad.len();
        bad[n - 1] ^= 0xff; // break the adler
        assert!(zlib_decompress(&bad).is_err());
        let mut trunc = DYN_ZLIB.to_vec();
        trunc.truncate(20);
        assert!(zlib_decompress(&trunc).is_err());
    }
}
