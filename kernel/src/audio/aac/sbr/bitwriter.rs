//! Minimal MSB-first bit writer for re-packing SBR extension payloads.
//!
//! Subset of oxideav-core::bits::BitWriter used by `sbr_element` extended-data
//! capture. MIT port note: oxideav-core (Karpelès Lab Inc.).

use alloc::vec::Vec;

/// Accumulates bits MSB-first into a byte buffer.
pub struct BitWriter {
    data: Vec<u8>,
    acc: u64,
    bits_in_acc: u32,
}

impl BitWriter {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            acc: 0,
            bits_in_acc: 0,
        }
    }

    /// Append `n` bits (0..=32) from the low `n` bits of `value`, MSB first.
    pub fn write_u32(&mut self, value: u32, n: u32) {
        if n == 0 {
            return;
        }
        debug_assert!(n <= 32);
        let mask: u32 = if n == 32 {
            u32::MAX
        } else {
            (1u32 << n) - 1
        };
        let v = (value & mask) as u64;
        let shift = 64 - self.bits_in_acc - n;
        self.acc |= v << shift;
        self.bits_in_acc += n;
        while self.bits_in_acc >= 8 {
            let byte = (self.acc >> 56) as u8;
            self.data.push(byte);
            self.acc <<= 8;
            self.bits_in_acc -= 8;
        }
    }

    /// Pad with zero bits to the next byte boundary, then return the bytes.
    pub fn finish(mut self) -> Vec<u8> {
        if self.bits_in_acc > 0 {
            let byte = (self.acc >> 56) as u8;
            self.data.push(byte);
            self.acc = 0;
            self.bits_in_acc = 0;
        }
        self.data
    }
}

impl Default for BitWriter {
    fn default() -> Self {
        Self::new()
    }
}
