//! H.264 bitstream primitives — the layer every higher stage reads through.
//!
//! Two pure pieces, both spec-faithful (ITU-T H.264, §7.2–7.4 & §9.1) and
//! unit-tested off-hardware:
//!
//! * [`unescape_rbsp`] strips *emulation-prevention* bytes: inside a NAL the
//!   byte sequence `00 00 03` has the `03` inserted purely so the payload can
//!   never contain a start code; the RBSP (raw byte sequence payload) is the
//!   NAL with those `03`s removed.
//! * [`BitReader`] reads the RBSP as a big-endian bit stream with the
//!   Exp-Golomb codes H.264 uses for most syntax elements: `u(n)` fixed-width,
//!   `ue(v)` unsigned Exp-Golomb, `se(v)` signed, `te(v)` truncated.
//!
//! No I/O, no panics on malformed input — every read past the end returns an
//! `Err`, so a truncated/garbage stream is a decode error, never a crash.

use alloc::vec::Vec;

/// Strip H.264 emulation-prevention `0x03` bytes: within the NAL payload, any
/// `00 00 03` becomes `00 00` (the `03` follows two zero bytes and precedes a
/// byte ≤ `0x03`). Returns the RBSP. Input should be the NAL payload *after*
/// the 1-byte NAL header.
pub fn unescape_rbsp(nal: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(nal.len());
    let mut zeros = 0usize;
    let mut i = 0;
    while i < nal.len() {
        let b = nal[i];
        // A 03 that sits right after exactly two 00s and introduces a byte in
        // 00..=03 is an emulation-prevention byte: drop it, keep the rest.
        if zeros >= 2 && b == 0x03 && i + 1 < nal.len() && nal[i + 1] <= 0x03 {
            zeros = 0;
            i += 1;
            continue;
        }
        out.push(b);
        zeros = if b == 0 { zeros + 1 } else { 0 };
        i += 1;
    }
    out
}

/// A big-endian bit reader over an RBSP, with H.264's Exp-Golomb codes.
pub struct BitReader<'a> {
    data: &'a [u8],
    /// Absolute bit position from the start of `data`.
    pos: usize,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        BitReader { data, pos: 0 }
    }

    /// Total bits available.
    fn total_bits(&self) -> usize {
        self.data.len() * 8
    }

    /// Bits not yet consumed.
    pub fn bits_left(&self) -> usize {
        self.total_bits().saturating_sub(self.pos)
    }

    /// Absolute bit position from the start of the data (for handing off to a
    /// byte-aligned consumer, e.g. the CABAC engine after the slice header).
    pub fn bit_pos(&self) -> usize {
        self.pos
    }

    /// Read a single bit (0/1). `Err` past the end.
    pub fn bit(&mut self) -> Result<u32, &'static str> {
        if self.pos >= self.total_bits() {
            return Err("h264 bitreader: read past end");
        }
        let byte = self.data[self.pos >> 3];
        let shift = 7 - (self.pos & 7);
        self.pos += 1;
        Ok(((byte >> shift) & 1) as u32)
    }

    /// Read `n` bits (0..=32) MSB-first as an unsigned integer — `u(n)`.
    pub fn u(&mut self, n: u32) -> Result<u32, &'static str> {
        debug_assert!(n <= 32);
        let mut v = 0u32;
        for _ in 0..n {
            v = (v << 1) | self.bit()?;
        }
        Ok(v)
    }

    /// Read `n` bits (0..=64) as an unsigned integer.
    pub fn u64(&mut self, n: u32) -> Result<u64, &'static str> {
        debug_assert!(n <= 64);
        let mut v = 0u64;
        for _ in 0..n {
            v = (v << 1) | self.bit()? as u64;
        }
        Ok(v)
    }

    /// A 1-bit flag — `u(1)` as a bool.
    pub fn flag(&mut self) -> Result<bool, &'static str> {
        Ok(self.bit()? != 0)
    }

    /// Unsigned Exp-Golomb — `ue(v)` (H.264 §9.1). Count leading zeros
    /// `leadingZeroBits`, then read that many bits; the code number is
    /// `2^lzb - 1 + read`.
    pub fn ue(&mut self) -> Result<u32, &'static str> {
        let mut lzb = 0u32;
        while self.bit()? == 0 {
            lzb += 1;
            if lzb > 32 {
                return Err("h264 bitreader: ue too long");
            }
        }
        if lzb == 0 {
            return Ok(0);
        }
        let rest = self.u(lzb)?;
        Ok((1u32 << lzb) - 1 + rest)
    }

    /// Signed Exp-Golomb — `se(v)` (H.264 §9.1.1): map ue code `k` to
    /// `(-1)^(k+1) * ceil(k/2)`.
    pub fn se(&mut self) -> Result<i32, &'static str> {
        let k = self.ue()? as i64;
        let mag = (k + 1) / 2;
        Ok(if k & 1 == 1 { mag as i32 } else { -(mag as i32) })
    }

    /// Truncated Exp-Golomb — `te(v)`: with `range == 1` it's a single
    /// inverted bit, otherwise identical to `ue`.
    pub fn te(&mut self, range: u32) -> Result<u32, &'static str> {
        if range == 1 {
            Ok(1 - self.bit()?)
        } else {
            self.ue()
        }
    }

    /// Advance to the next byte boundary (used before reading byte-aligned
    /// payloads). No-op if already aligned.
    pub fn byte_align(&mut self) {
        self.pos = (self.pos + 7) & !7;
    }

    /// True while RBSP payload remains before the `rbsp_stop_one_bit` trailer
    /// (H.264 §7.2). Approximated the standard way: there is more data unless
    /// the only remaining bits are a `1` followed by zero-padding to the byte.
    pub fn more_rbsp_data(&mut self) -> bool {
        let left = self.bits_left();
        if left == 0 {
            return false;
        }
        // Find the last set bit in the stream; RBSP data ends just before it.
        let last_one = self.last_set_bit();
        match last_one {
            Some(idx) => self.pos < idx,
            None => false,
        }
    }

    /// Absolute bit index of the final `1` bit in the stream (the stop bit).
    fn last_set_bit(&self) -> Option<usize> {
        for byte_i in (0..self.data.len()).rev() {
            let b = self.data[byte_i];
            if b != 0 {
                // position of the lowest set bit within the byte
                let tz = b.trailing_zeros() as usize;
                let bit_in_byte = 7 - tz;
                return Some(byte_i * 8 + bit_in_byte);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn unescape_removes_emulation_bytes() {
        // 00 00 03 00  →  00 00 00  (the 03 before a <=03 byte is dropped)
        assert_eq!(unescape_rbsp(&[0x00, 0x00, 0x03, 0x00]), alloc::vec![0x00, 0x00, 0x00]);
        // 00 00 03 01 02 03  →  00 00 01 02 03
        assert_eq!(unescape_rbsp(&[0x00, 0x00, 0x03, 0x01, 0x02, 0x03]), alloc::vec![0x00, 0x00, 0x01, 0x02, 0x03]);
        // A 03 not preceded by two zeros is kept verbatim.
        assert_eq!(unescape_rbsp(&[0x01, 0x03, 0x04]), alloc::vec![0x01, 0x03, 0x04]);
        // 00 00 03 04 — 04 > 03, so the 03 is real data, not stripped.
        assert_eq!(unescape_rbsp(&[0x00, 0x00, 0x03, 0x04]), alloc::vec![0x00, 0x00, 0x03, 0x04]);
    }

    #[test_case]
    fn fixed_width_reads() {
        // 1011 0010  0101 ...
        let mut r = BitReader::new(&[0b1011_0010, 0b0101_0000]);
        assert_eq!(r.u(1).unwrap(), 1);
        assert_eq!(r.u(3).unwrap(), 0b011);
        assert_eq!(r.u(4).unwrap(), 0b0010);
        assert_eq!(r.u(4).unwrap(), 0b0101);
        // Past the meaningful bits but still inside the byte is fine (zeros);
        // reading past the buffer errors.
        assert_eq!(r.u(4).unwrap(), 0);
        assert!(r.u(1).is_err());
    }

    #[test_case]
    fn exp_golomb_unsigned() {
        // ue codewords (H.264 Table 9-2):
        //   1            -> 0
        //   010          -> 1
        //   011          -> 2
        //   00100        -> 3
        //   00111        -> 6
        // Pack "1 010 011 00100 00111" MSB-first.
        // bits: 1 0 1 0 0 1 1 0 0 1 0 0 0 0 1 1 1  (17 bits)
        let bytes = [0b1010_0110, 0b0100_0011, 0b1000_0000];
        let mut r = BitReader::new(&bytes);
        assert_eq!(r.ue().unwrap(), 0);
        assert_eq!(r.ue().unwrap(), 1);
        assert_eq!(r.ue().unwrap(), 2);
        assert_eq!(r.ue().unwrap(), 3);
        assert_eq!(r.ue().unwrap(), 6);
    }

    #[test_case]
    fn exp_golomb_signed() {
        // se mapping: ue 0->0, 1->+1, 2->-1, 3->+2, 4->-2 ...
        // ue codes for 0,1,2,3,4 = "1 010 011 00100 00101"
        // bits: 1 0 1 0 0 1 1 0 0 1 0 0 0 0 1 0 1
        let bytes = [0b1010_0110, 0b0100_0010, 0b1000_0000];
        let mut r = BitReader::new(&bytes);
        assert_eq!(r.se().unwrap(), 0);
        assert_eq!(r.se().unwrap(), 1);
        assert_eq!(r.se().unwrap(), -1);
        assert_eq!(r.se().unwrap(), 2);
        assert_eq!(r.se().unwrap(), -2);
    }

    #[test_case]
    fn truncated_exp_golomb() {
        // range==1: a single inverted bit. Stream 0b0100_0000: first bit 0 -> 1,
        // second bit 1 -> 0.
        let mut r = BitReader::new(&[0b0100_0000]);
        assert_eq!(r.te(1).unwrap(), 1);
        assert_eq!(r.te(1).unwrap(), 0);
        // range>1 falls through to ue. "010" is the ue codeword for 1.
        let mut r2 = BitReader::new(&[0b0100_0000]);
        assert_eq!(r2.te(5).unwrap(), 1);
    }

    #[test_case]
    fn more_rbsp_data_tracks_stop_bit() {
        // Payload bits "101" then the rbsp stop bit "1" and zero padding:
        // 1 0 1 1 0 0 0 0  => 0b1011_0000. Stop bit is the 4th bit (index 3).
        let mut r = BitReader::new(&[0b1011_0000]);
        assert!(r.more_rbsp_data()); // at 0, data ends at 3
        r.u(3).unwrap(); // consume "101"
        assert!(!r.more_rbsp_data()); // now at the stop bit
    }
}
