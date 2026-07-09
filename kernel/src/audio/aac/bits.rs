//! MSB-first bit reader for raw AAC access units (no ADTS).
//!
//! Adapted from Symphonia's `BitReaderLtr` (MPL-2.0).

/// Left-to-right (MSB-first) bit reader over a byte slice.
pub struct BitReader<'a> {
    data: &'a [u8],
    /// Absolute bit position into `data`.
    pos: usize,
    /// Total bits available.
    len_bits: usize,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            len_bits: data.len().saturating_mul(8),
        }
    }

    #[inline]
    pub fn bits_left(&self) -> usize {
        self.len_bits.saturating_sub(self.pos)
    }

    /// Absolute bit position (for SBR/PS parsers that size fill regions).
    #[inline]
    pub fn bit_position(&self) -> u64 {
        self.pos as u64
    }

    #[inline]
    pub fn realign(&mut self) {
        let rem = self.pos & 7;
        if rem != 0 {
            self.pos += 8 - rem;
            if self.pos > self.len_bits {
                self.pos = self.len_bits;
            }
        }
    }

    #[inline]
    pub fn read_bit(&mut self) -> Result<u32, &'static str> {
        if self.pos >= self.len_bits {
            return Err("aac: bitstream underrun");
        }
        let byte = self.data[self.pos >> 3];
        let bit = 7 - (self.pos & 7);
        self.pos += 1;
        Ok(((byte >> bit) & 1) as u32)
    }

    #[inline]
    pub fn read_bool(&mut self) -> Result<bool, &'static str> {
        Ok(self.read_bit()? != 0)
    }

    /// Read up to 32 bits MSB-first.
    pub fn read_bits(&mut self, n: u32) -> Result<u32, &'static str> {
        if n == 0 {
            return Ok(0);
        }
        if n > 32 {
            return Err("aac: read_bits > 32");
        }
        if self.bits_left() < n as usize {
            return Err("aac: bitstream underrun");
        }
        let mut v = 0u32;
        let mut left = n;
        while left > 0 {
            let bit_in_byte = self.pos & 7;
            let avail = 8 - bit_in_byte;
            let take = core::cmp::min(left as usize, avail);
            let byte = self.data[self.pos >> 3] as u32;
            let shift = avail - take;
            let mask = (1u32 << take) - 1;
            v = (v << take) | ((byte >> shift) & mask);
            self.pos += take;
            left -= take as u32;
        }
        Ok(v)
    }

    pub fn skip(&mut self, n: u32) -> Result<(), &'static str> {
        if self.bits_left() < n as usize {
            return Err("aac: bitstream underrun");
        }
        self.pos += n as usize;
        Ok(())
    }

    /// Peek `n` bits without advancing (n ≤ 32).
    pub fn peek(&self, n: u32) -> Result<u32, &'static str> {
        if n == 0 {
            return Ok(0);
        }
        if n > 32 {
            return Err("aac: peek > 32");
        }
        if self.bits_left() < n as usize {
            return Err("aac: bitstream underrun");
        }
        let mut pos = self.pos;
        let mut v = 0u32;
        let mut left = n;
        while left > 0 {
            let bit_in_byte = pos & 7;
            let avail = 8 - bit_in_byte;
            let take = core::cmp::min(left as usize, avail);
            let byte = self.data[pos >> 3] as u32;
            let shift = avail - take;
            let mask = (1u32 << take) - 1;
            v = (v << take) | ((byte >> shift) & mask);
            pos += take;
            left -= take as u32;
        }
        Ok(v)
    }

    /// Count leading ones then consume the terminating zero (AAC escape unary).
    pub fn read_unary_ones(&mut self) -> Result<u32, &'static str> {
        let mut n = 0u32;
        loop {
            if self.read_bit()? == 0 {
                return Ok(n);
            }
            n += 1;
            if n > 32 {
                return Err("aac: unary too long");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn read_bits_basic() {
        // 0b1011_0001 0b1100_0000
        let mut br = BitReader::new(&[0xb1, 0xc0]);
        assert_eq!(br.read_bits(4).unwrap(), 0xb);
        assert_eq!(br.read_bits(4).unwrap(), 0x1);
        assert_eq!(br.read_bits(2).unwrap(), 0b11);
        assert_eq!(br.read_bit().unwrap(), 0);
    }

    #[test_case]
    fn underrun_is_err() {
        let mut br = BitReader::new(&[0xff]);
        assert!(br.read_bits(8).is_ok());
        assert!(br.read_bit().is_err());
    }
}
