//! CABAC arithmetic decoding engine (H.264 §9.3) over spec-form tables
//! generated from the FFmpeg sources ([`super::cabac_tables`]).
//!
//! The engine is the bit-serial form of the spec: 9-bit `codIRange` /
//! `codIOffset`, `DecodeDecision` with the rangeTabLPS + transIdx tables,
//! `DecodeBypass`, and `DecodeTerminate`. Contexts are stored s-encoded
//! (`state << 1 | valMPS`) in a fixed 1024-slot array (no per-slice heap).
//! A 32-bit bit reservoir feeds renorm without per-bit slice indexing.
//! Reading past the end of the slice yields 0 bits (the trailing stop bit +
//! alignment guarantee this only happens on the final renorms).

use super::cabac_tables as t;

/// The arithmetic decoder + the full context-model array for one slice.
pub struct Cabac<'a> {
    data: &'a [u8],
    /// Next byte index into `data` for the bit reservoir refill.
    byte_pos: usize,
    /// Bit reservoir: next bits live in the high end of `bits` (left-justified
    /// for the remaining count), consumed MSB-first via `take_bit`.
    bits: u32,
    nbits: u32,
    range: u32,
    offset: u32,
    /// s-encoded context states (`state << 1 | valMPS`), ctxIdx-indexed.
    /// Fixed array — no per-slice `Vec` allocation on 1080p (was ~1kB + free
    /// each slice; hundreds of slices/sec).
    pub ctx: [u8; 1024],
}

impl<'a> Cabac<'a> {
    /// Initialise contexts for the slice (§9.3.1.1) and the engine (§9.3.1.2).
    /// `data` starts at the first byte of the byte-aligned slice data; `qp` is
    /// the slice QP; `init_set` is None for I slices, or `Some(cabac_init_idc)`.
    pub fn new(data: &'a [u8], qp: i32, init_set: Option<u32>) -> Result<Cabac<'a>, &'static str> {
        let table: &[[i8; 2]; 1024] = match init_set {
            None => &t::CTX_INIT_I,
            Some(idc) => t::CTX_INIT_PB.get(idc as usize).ok_or("h264: bad cabac_init_idc")?,
        };
        let qp = qp.clamp(0, 51);
        let mut ctx = [0u8; 1024];
        for (i, &[m, n]) in table.iter().enumerate() {
            let pre = ((m as i32 * qp) >> 4) + n as i32;
            let pre = pre.clamp(1, 126);
            ctx[i] = if pre <= 63 {
                (((63 - pre) as u8) << 1) | 0
            } else {
                (((pre - 64) as u8) << 1) | 1
            };
        }
        let mut c = Cabac {
            data,
            byte_pos: 0,
            bits: 0,
            nbits: 0,
            range: 510,
            offset: 0,
            ctx,
        };
        // Prefill reservoir before the 9-bit init offset.
        c.refill();
        for _ in 0..9 {
            c.offset = (c.offset << 1) | c.take_bit();
        }
        if c.offset >= 510 {
            return Err("h264 cabac: invalid init offset");
        }
        Ok(c)
    }

    /// Keep at least 24 bits in the reservoir (or until EOF, padded with 0).
    #[inline]
    fn refill(&mut self) {
        // Pull whole bytes while we have room; past EOF we still "consume"
        // positions so renorm stays in lockstep with the spec (0 bits).
        while self.nbits <= 24 {
            let b = if self.byte_pos < self.data.len() {
                let v = self.data[self.byte_pos] as u32;
                self.byte_pos += 1;
                v
            } else {
                self.byte_pos = self.byte_pos.saturating_add(1);
                0
            };
            self.bits = (self.bits << 8) | b;
            self.nbits += 8;
            // Cap so we never shift past 32 (nbits can hit 32 exactly).
            if self.nbits >= 32 {
                break;
            }
        }
    }

    #[inline]
    fn take_bit(&mut self) -> u32 {
        if self.nbits == 0 {
            self.refill();
        }
        // Still empty only if data was empty and we just started — treat as 0.
        if self.nbits == 0 {
            return 0;
        }
        self.nbits -= 1;
        (self.bits >> self.nbits) & 1
    }

    /// Initialise for an **HEVC** slice (H.265 §9.3.2.2). The arithmetic engine
    /// and its tables are identical to H.264's — only the context
    /// initialisation differs, so this shares everything below it rather than
    /// growing a second copy of a bit-exact decoder.
    ///
    /// The derivation is a different shape from AVC's: one `initValue` byte per
    /// context yields the slope and offset (`m = (v >> 4) * 5 - 45`,
    /// `n = ((v & 15) << 3) - 16`) instead of a stored `(m, n)` pair.
    ///
    /// `init_type` is `2 - slice_type` (so 2 for I, 1 for P, 0 for B), **XORed
    /// with 3** on a non-I slice when the slice header set `cabac_init_flag` —
    /// which swaps the P and B tables, and is the whole purpose of that flag.
    pub fn new_hevc(
        data: &'a [u8],
        qp: i32,
        init_type: usize,
        init_values: &[[u8; super::super::hevc::cabac_tables::HEVC_CONTEXTS]; 3],
    ) -> Result<Cabac<'a>, &'static str> {
        let table = init_values.get(init_type).ok_or("hevc: bad cabac init type")?;
        let qp = qp.clamp(0, 51);
        let mut ctx = [0u8; 1024];
        for (i, &v) in table.iter().enumerate() {
            let m = (v as i32 >> 4) * 5 - 45;
            let n = ((v as i32 & 15) << 3) - 16;
            let pre = (((m * qp) >> 4) + n).clamp(1, 126);
            ctx[i] = if pre <= 63 {
                ((63 - pre) as u8) << 1
            } else {
                (((pre - 64) as u8) << 1) | 1
            };
        }
        let mut c = Cabac { data, byte_pos: 0, bits: 0, nbits: 0, range: 510, offset: 0, ctx };
        c.refill();
        c.refill();
        c.offset = 0;
        for _ in 0..9 {
            if c.nbits == 0 {
                c.refill();
            }
            c.offset = (c.offset << 1) | c.take_bit();
        }
        Ok(c)
    }

    /// How many bytes of the slice the reservoir has pulled in. Bring-up only:
    /// a parse that ends far from the slice's length has desynchronised, which
    /// distinguishes "read the wrong bins" from "reconstructed them wrongly".
    pub fn byte_pos(&self) -> usize {
        self.byte_pos
    }

    /// DecodeDecision (§9.3.3.2.1) for the context at `ctx_idx`.
    #[inline]
    pub fn decision(&mut self, ctx_idx: usize) -> u32 {
        let s = self.ctx[ctx_idx];
        let state = (s >> 1) as usize;
        let mps = (s & 1) as u32;
        let q = ((self.range >> 6) & 3) as usize;
        let lps = t::RANGE_LPS[state][q] as u32;
        self.range -= lps;
        let bin;
        if self.offset >= self.range {
            bin = mps ^ 1;
            self.offset -= self.range;
            self.range = lps;
            // pStateIdx 0 also flips valMPS (folded into TRANS_LPS derivation).
            let new_mps = if state == 0 { mps ^ 1 } else { mps };
            self.ctx[ctx_idx] = (t::TRANS_LPS[state] << 1) | new_mps as u8;
        } else {
            bin = mps;
            self.ctx[ctx_idx] = (t::TRANS_MPS[state] << 1) | mps as u8;
        }
        // Renorm: typically 0–2 shifts; batch-refill when the reservoir is thin.
        while self.range < 256 {
            if self.nbits < 4 {
                self.refill();
            }
            self.range <<= 1;
            self.offset = (self.offset << 1) | self.take_bit();
        }
        bin
    }

    /// DecodeBypass (§9.3.3.2.3).
    #[inline]
    pub fn bypass(&mut self) -> u32 {
        if self.nbits < 4 {
            self.refill();
        }
        self.offset = (self.offset << 1) | self.take_bit();
        if self.offset >= self.range {
            self.offset -= self.range;
            1
        } else {
            0
        }
    }

    /// DecodeTerminate (§9.3.3.2.4): 1 = end of slice / PCM escape.
    #[inline]
    pub fn terminate(&mut self) -> u32 {
        self.range -= 2;
        if self.offset >= self.range {
            1
        } else {
            while self.range < 256 {
                if self.nbits < 4 {
                    self.refill();
                }
                self.range <<= 1;
                self.offset = (self.offset << 1) | self.take_bit();
            }
            0
        }
    }

    /// Bypass-decode a sign and apply it: returns `-mag` if the sign bit is 1
    /// else `mag`.
    #[inline]
    pub fn bypass_sign(&mut self, mag: i32) -> i32 {
        if self.bypass() != 0 {
            -mag
        } else {
            mag
        }
    }

    /// After `terminate()` returned 1 for a PCM escape: locate the next
    /// byte-aligned raw payload of `n` bytes, return it, and re-arm the
    /// arithmetic engine on the remainder **keeping the context states**.
    ///
    /// Mirrors FFmpeg's `skip_bytes` (contexts persist; only low/range/stream
    /// restart). The reservoir may have prefetched past the terminate bit, so
    /// the start position rewinds unread buffered bytes.
    pub fn take_raw_after_terminate(
        &mut self,
        n: usize,
    ) -> Result<&'a [u8], &'static str> {
        // Bytes still sitting in the reservoir that were never arithmetically
        // consumed (terminate-1 does not renorm).
        let unread = (self.nbits as usize) / 8;
        let mut start = self.byte_pos.saturating_sub(unread);
        // Partial residual bits mean the current byte was only half-used —
        // back up one, matching FFmpeg's `if (low & 1) ptr--`.
        if self.nbits % 8 != 0 || (self.offset & 1) != 0 {
            start = start.saturating_sub(1);
        }
        let end = start.checked_add(n).ok_or("hevc pcm: overflow")?;
        if end > self.data.len() {
            return Err("hevc pcm: sample payload past end of slice");
        }
        let raw = &self.data[start..end];
        let rest = &self.data[end..];
        let ctx = self.ctx;
        // Re-init arithmetic only (same shape as new_hevc's engine half).
        let mut c = Cabac {
            data: rest,
            byte_pos: 0,
            bits: 0,
            nbits: 0,
            range: 510,
            offset: 0,
            ctx,
        };
        c.refill();
        c.refill();
        c.offset = 0;
        for _ in 0..9 {
            if c.nbits == 0 {
                c.refill();
            }
            c.offset = (c.offset << 1) | c.take_bit();
        }
        *self = c;
        Ok(raw)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A hand-checkable engine exercise: encode nothing — just verify context
    // init produces valid s-encodings and the engine consumes bits sanely on
    // an all-zero stream (decisions must not panic and must renormalise).
    #[test_case]
    fn ctx_init_and_decisions_do_not_panic() {
        let data = [0u8; 32];
        let mut c = Cabac::new(&data, 26, None).unwrap();
        for i in 0..1024 {
            // state 0..63 in the high bits, mps in bit 0.
            assert!(c.ctx[i] >> 1 <= 63);
        }
        let mut ones = 0;
        for i in 0..512 {
            ones += c.decision(i % 1024);
        }
        // On a zero stream the exact decisions depend on the models; the
        // invariant that matters is the engine stayed in range.
        assert!(c.range >= 256 && c.range < 512, "range renormalised");
        let _ = ones;
    }

    #[test_case]
    fn bypass_reads_bits_in_order() {
        // First 9 bits = 0b111111101 = 509, the largest *valid* init offset
        // (510/511 are forbidden — an all-0xFF stream must be rejected). With
        // every following bit 1, each bypass computes 509*2+1 = 1019 ≥ 510 →
        // decodes 1 and returns the offset to 509, so the 1s repeat forever.
        assert!(Cabac::new(&[0xffu8; 16], 26, None).is_err(), "offset 511 must be rejected");
        let mut data = [0xffu8; 16];
        data[0] = 0xfe;
        let mut c = Cabac::new(&data, 26, None).unwrap();
        let mut all = 0;
        for _ in 0..8 {
            all += c.bypass();
        }
        assert_eq!(all, 8, "bypass on an all-ones stream is all ones");
    }
}
