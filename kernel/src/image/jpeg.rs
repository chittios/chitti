//! Baseline JPEG decoder (SOF0/SOF1 Huffman sequential): DQT/DHT/SOF/SOS/DRI
//! parsing, entropy decode with byte-stuffing and restart markers, dequant +
//! inverse zigzag, separable f32 IDCT, YCbCr→RGB with generic h/v chroma
//! subsampling (4:4:4, 4:2:2, 4:2:0, …) and grayscale. Progressive (SOF2),
//! arithmetic coding, and CMYK are rejected with clear errors. Malformed
//! input returns `Err`, never panics.

use super::Image;
use alloc::vec::Vec;

/// Zigzag order: `ZIGZAG[k]` is the natural (row-major) index of the k-th
/// coefficient in scan order.
const ZIGZAG: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21,
    28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61,
    54, 47, 55, 62, 63,
];

/// IDCT basis `B[u][x] = 0.5·c(u)·cos((2x+1)uπ/16)` so
/// `f(x,y) = Σu Σv B[u][y]·B[v][x]·F(u,v)` (two separable passes).
const IDCT_B: [[f32; 8]; 8] = [
    [0.353553391, 0.353553391, 0.353553391, 0.353553391, 0.353553391, 0.353553391, 0.353553391, 0.353553391],
    [0.49039264, 0.415734806, 0.277785117, 0.097545161, -0.097545161, -0.277785117, -0.415734806, -0.49039264],
    [0.461939766, 0.191341716, -0.191341716, -0.461939766, -0.461939766, -0.191341716, 0.191341716, 0.461939766],
    [0.415734806, -0.097545161, -0.49039264, -0.277785117, 0.277785117, 0.49039264, 0.097545161, -0.415734806],
    [0.353553391, -0.353553391, -0.353553391, 0.353553391, 0.353553391, -0.353553391, -0.353553391, 0.353553391],
    [0.277785117, -0.49039264, 0.097545161, 0.415734806, -0.415734806, -0.097545161, 0.49039264, -0.277785117],
    [0.191341716, -0.461939766, 0.461939766, -0.191341716, -0.191341716, 0.461939766, -0.461939766, 0.191341716],
    [0.097545161, -0.277785117, 0.415734806, -0.49039264, 0.49039264, -0.415734806, 0.277785117, -0.097545161],
];

/// 8x8 inverse DCT: dequantized coefficients (natural order) → spatial
/// samples, level-shifted (+128) and clamped to 0..=255.
fn idct8x8(coef: &[i32; 64], out: &mut [u8; 64]) {
    let mut tmp = [0f32; 64];
    // Row pass: for each coefficient row u, transform over v → tmp[u][x].
    for u in 0..8 {
        for x in 0..8 {
            let mut s = 0f32;
            for v in 0..8 {
                s += IDCT_B[v][x] * coef[u * 8 + v] as f32;
            }
            tmp[u * 8 + x] = s;
        }
    }
    // Column pass: transform over u → pixel (x, y).
    for y in 0..8 {
        for x in 0..8 {
            let mut s = 0f32;
            for u in 0..8 {
                s += IDCT_B[u][y] * tmp[u * 8 + x];
            }
            let v = (s + 128.5) as i32;
            out[y * 8 + x] = v.clamp(0, 255) as u8;
        }
    }
}

/// A JPEG canonical Huffman table (DHT): 16 length counts + values in order.
struct Huff {
    counts: [u16; 17],
    values: Vec<u8>,
}

impl Huff {
    fn decode(&self, br: &mut Bits<'_>) -> Result<u8, &'static str> {
        let mut code = 0i32;
        let mut first = 0i32;
        let mut index = 0i32;
        for len in 1..=16 {
            code |= br.bit()? as i32;
            let count = self.counts[len] as i32;
            if code - first < count {
                return self.values.get((index + (code - first)) as usize).copied().ok_or("jpeg: huffman value out of range");
            }
            index += count;
            first = (first + count) << 1;
            code <<= 1;
        }
        Err("jpeg: invalid huffman code")
    }
}

/// MSB-first bit reader over entropy-coded data with 0xFF00 byte unstuffing.
/// A bare marker in the stream ends it (`Err`); restart markers are consumed
/// out-of-band by [`Bits::sync_restart`].
struct Bits<'a> {
    src: &'a [u8],
    pos: usize,
    buf: u32,
    n: u32,
}

impl<'a> Bits<'a> {
    fn bit(&mut self) -> Result<u32, &'static str> {
        if self.n == 0 {
            let Some(&b) = self.src.get(self.pos) else { return Err("jpeg: truncated entropy stream") };
            if b == 0xff {
                match self.src.get(self.pos + 1) {
                    Some(0x00) => self.pos += 2, // stuffed 0xFF data byte
                    _ => return Err("jpeg: marker inside entropy stream"),
                }
            } else {
                self.pos += 1;
            }
            self.buf = b as u32;
            self.n = 8;
        }
        self.n -= 1;
        Ok((self.buf >> self.n) & 1)
    }

    /// `n` bits, MSB first (the DC/AC "receive" primitive).
    fn receive(&mut self, n: u32) -> Result<i32, &'static str> {
        let mut v = 0i32;
        for _ in 0..n {
            v = (v << 1) | self.bit()? as i32;
        }
        Ok(v)
    }

    /// Byte-align and consume an expected RSTn marker (restart interval).
    fn sync_restart(&mut self) -> Result<(), &'static str> {
        self.n = 0;
        if self.src.get(self.pos) == Some(&0xff) {
            if let Some(&m) = self.src.get(self.pos + 1) {
                if (0xd0..=0xd7).contains(&m) {
                    self.pos += 2;
                    return Ok(());
                }
            }
        }
        Err("jpeg: expected restart marker")
    }
}

/// JPEG signed-magnitude extension (F.2.2.1 EXTEND).
fn extend(v: i32, n: u32) -> i32 {
    if n == 0 {
        0
    } else if v < (1 << (n - 1)) {
        v - (1 << n) + 1
    } else {
        v
    }
}

#[derive(Clone, Copy, Default)]
struct Component {
    h: usize,
    v: usize,
    tq: usize, // quant table id
    td: usize, // DC huffman id
    ta: usize, // AC huffman id
}

pub fn decode(bytes: &[u8]) -> Result<Image, &'static str> {
    if bytes.len() < 4 || bytes[0] != 0xff || bytes[1] != 0xd8 {
        return Err("jpeg: bad SOI");
    }
    let mut qt: [[u16; 64]; 4] = [[0; 64]; 4];
    let mut dc_tabs: [Option<Huff>; 4] = [None, None, None, None];
    let mut ac_tabs: [Option<Huff>; 4] = [None, None, None, None];
    let (mut w, mut h) = (0usize, 0usize);
    let mut comps: Vec<Component> = Vec::new();
    let mut dri = 0usize;

    let mut i = 2;
    loop {
        // Find the next marker (skip fill bytes).
        while i < bytes.len() && bytes[i] != 0xff {
            i += 1;
        }
        while i < bytes.len() && bytes[i] == 0xff {
            i += 1;
        }
        let Some(&marker) = bytes.get(i) else { return Err("jpeg: truncated") };
        i += 1;
        match marker {
            0xd9 => return Err("jpeg: EOI before SOS"), // end of image
            0xc2 => return Err("jpeg: progressive (SOF2) unsupported"),
            0xc3 | 0xc5..=0xc7 | 0xc9..=0xcb | 0xcd..=0xcf => return Err("jpeg: unsupported SOF type"),
            _ => {}
        }
        // Every remaining marker here carries a 2-byte big-endian length.
        if i + 2 > bytes.len() {
            return Err("jpeg: truncated segment");
        }
        let seg_len = ((bytes[i] as usize) << 8 | bytes[i + 1] as usize).max(2);
        let seg_end = i + seg_len;
        if seg_end > bytes.len() {
            return Err("jpeg: segment past end");
        }
        let mut p = i + 2;
        match marker {
            0xdb => {
                // DQT: one or more (pq/tq, 64 entries) tables.
                while p < seg_end {
                    let pq = bytes[p] >> 4;
                    let tq = (bytes[p] & 15) as usize;
                    p += 1;
                    if tq >= 4 {
                        return Err("jpeg: bad quant table id");
                    }
                    for k in 0..64 {
                        let v = if pq == 0 {
                            let v = *bytes.get(p).ok_or("jpeg: truncated DQT")? as u16;
                            p += 1;
                            v
                        } else {
                            let v = ((*bytes.get(p).ok_or("jpeg: truncated DQT")? as u16) << 8)
                                | *bytes.get(p + 1).ok_or("jpeg: truncated DQT")? as u16;
                            p += 2;
                            v
                        };
                        qt[tq][k] = v;
                    }
                }
            }
            0xc4 => {
                // DHT: one or more (class/id, 16 counts, values) tables.
                while p + 17 <= seg_end {
                    let class = bytes[p] >> 4;
                    let id = (bytes[p] & 15) as usize;
                    p += 1;
                    if id >= 4 || class > 1 {
                        return Err("jpeg: bad huffman table spec");
                    }
                    let mut counts = [0u16; 17];
                    let mut total = 0usize;
                    for (l, c) in counts.iter_mut().skip(1).enumerate() {
                        let _ = l;
                        *c = bytes[p] as u16;
                        total += bytes[p] as usize;
                        p += 1;
                    }
                    if p + total > seg_end {
                        return Err("jpeg: truncated DHT values");
                    }
                    let values = bytes[p..p + total].to_vec();
                    p += total;
                    let t = Huff { counts, values };
                    if class == 0 {
                        dc_tabs[id] = Some(t);
                    } else {
                        ac_tabs[id] = Some(t);
                    }
                }
            }
            0xc0 | 0xc1 => {
                // SOF0/1: precision, dims, components with sampling factors.
                if bytes[p] != 8 {
                    return Err("jpeg: only 8-bit precision supported");
                }
                h = (bytes[p + 1] as usize) << 8 | bytes[p + 2] as usize;
                w = (bytes[p + 3] as usize) << 8 | bytes[p + 4] as usize;
                let nc = bytes[p + 5] as usize;
                if w == 0 || h == 0 || w > 16384 || h > 16384 || w * h > (32 << 20) {
                    return Err("jpeg: unreasonable dimensions");
                }
                if nc != 1 && nc != 3 {
                    return Err("jpeg: only grayscale and YCbCr supported");
                }
                p += 6;
                comps.clear();
                for _ in 0..nc {
                    let hv = bytes[p + 1];
                    let c = Component {
                        h: (hv >> 4) as usize,
                        v: (hv & 15) as usize,
                        tq: bytes[p + 2] as usize,
                        ..Default::default()
                    };
                    if c.h == 0 || c.h > 4 || c.v == 0 || c.v > 4 || c.tq >= 4 {
                        return Err("jpeg: bad component spec");
                    }
                    comps.push(c);
                    p += 3;
                }
            }
            0xdd => {
                dri = (bytes[p] as usize) << 8 | bytes[p + 1] as usize;
            }
            0xda => {
                // SOS: bind huffman tables, then decode the entropy stream.
                let ns = bytes[p] as usize;
                if comps.is_empty() {
                    return Err("jpeg: SOS before SOF");
                }
                if ns != comps.len() {
                    return Err("jpeg: multi-scan baseline unsupported");
                }
                p += 1;
                for _ in 0..ns {
                    let cs = bytes[p] as usize;
                    let idx = (cs.saturating_sub(1)).min(comps.len() - 1);
                    comps[idx].td = (bytes[p + 1] >> 4) as usize;
                    comps[idx].ta = (bytes[p + 1] & 15) as usize;
                    p += 2;
                }
                // Skip Ss/Se/Ah+Al (0, 63, 0 for sequential).
                p += 3;
                return decode_scan(bytes, p, w, h, &comps, &qt, &dc_tabs, &ac_tabs, dri);
            }
            _ => {} // APPn / COM / others: skip
        }
        i = seg_end;
    }
}

/// Decode the interleaved entropy-coded scan into planes, then convert to RGB.
#[allow(clippy::too_many_arguments)]
fn decode_scan(
    bytes: &[u8],
    start: usize,
    w: usize,
    h: usize,
    comps: &[Component],
    qt: &[[u16; 64]; 4],
    dc_tabs: &[Option<Huff>; 4],
    ac_tabs: &[Option<Huff>; 4],
    dri: usize,
) -> Result<Image, &'static str> {
    let hmax = comps.iter().map(|c| c.h).max().unwrap_or(1);
    let vmax = comps.iter().map(|c| c.v).max().unwrap_or(1);
    let mcus_x = w.div_ceil(8 * hmax);
    let mcus_y = h.div_ceil(8 * vmax);

    // Full padded plane per component (MCU-aligned; sampled down later).
    let mut planes: Vec<Vec<u8>> = comps.iter().map(|c| alloc::vec![0u8; mcus_x * c.h * 8 * mcus_y * c.v * 8]).collect();
    let plane_w: Vec<usize> = comps.iter().map(|c| mcus_x * c.h * 8).collect();

    let mut br = Bits { src: bytes, pos: start, buf: 0, n: 0 };
    let mut dc_pred = alloc::vec![0i32; comps.len()];
    let mut block = [0u8; 64];

    for my in 0..mcus_y {
        for mx in 0..mcus_x {
            // Restart interval: byte-align, eat RSTn, reset DC predictors.
            if dri > 0 && (my * mcus_x + mx) > 0 && (my * mcus_x + mx) % dri == 0 {
                br.sync_restart()?;
                dc_pred.iter_mut().for_each(|d| *d = 0);
            }
            for (ci, c) in comps.iter().enumerate() {
                let dct = dc_tabs[c.td].as_ref().ok_or("jpeg: missing DC table")?;
                let act = ac_tabs[c.ta].as_ref().ok_or("jpeg: missing AC table")?;
                let q = &qt[c.tq];
                for by in 0..c.v {
                    for bx in 0..c.h {
                        let mut coef = [0i32; 64];
                        // DC coefficient (differential).
                        let t = dct.decode(&mut br)? as u32;
                        if t > 11 {
                            return Err("jpeg: bad DC category");
                        }
                        dc_pred[ci] += extend(br.receive(t)?, t);
                        coef[0] = dc_pred[ci] * q[0] as i32;
                        // AC coefficients: run/size until EOB.
                        let mut k = 1usize;
                        while k < 64 {
                            let rs = act.decode(&mut br)?;
                            let (r, s) = ((rs >> 4) as usize, (rs & 15) as u32);
                            if s == 0 {
                                if r == 15 {
                                    k += 16; // ZRL: sixteen zeros
                                    continue;
                                }
                                break; // EOB
                            }
                            k += r;
                            if k > 63 {
                                return Err("jpeg: AC run past block end");
                            }
                            coef[ZIGZAG[k]] = extend(br.receive(s)?, s) * q[k] as i32;
                            k += 1;
                        }
                        idct8x8(&coef, &mut block);
                        // Place the 8x8 block into the component plane.
                        let px = (mx * c.h + bx) * 8;
                        let py = (my * c.v + by) * 8;
                        let pw = plane_w[ci];
                        for row in 0..8 {
                            let dst = (py + row) * pw + px;
                            planes[ci][dst..dst + 8].copy_from_slice(&block[row * 8..row * 8 + 8]);
                        }
                    }
                }
            }
        }
    }

    // Colour conversion with generic chroma upsampling (nearest sample).
    let mut pixels: Vec<u32> = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let s = |ci: usize| -> i32 {
                let c = &comps[ci];
                let sx = x * c.h / hmax;
                let sy = y * c.v / vmax;
                planes[ci][sy * plane_w[ci] + sx] as i32
            };
            let (r, g, b) = if comps.len() == 1 {
                let v = s(0);
                (v, v, v)
            } else {
                // ITU-R BT.601 YCbCr→RGB, 16.16 fixed point.
                let (yv, cb, cr) = (s(0), s(1) - 128, s(2) - 128);
                (
                    yv + ((91881 * cr) >> 16),
                    yv - ((22554 * cb + 46802 * cr) >> 16),
                    yv + ((116130 * cb) >> 16),
                )
            };
            let (r, g, b) = (r.clamp(0, 255) as u32, g.clamp(0, 255) as u32, b.clamp(0, 255) as u32);
            pixels.push((r << 16) | (g << 8) | b);
        }
    }
    Ok(Image { w, h, pixels })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn zigzag_is_a_permutation() {
        let mut seen = [false; 64];
        for &z in ZIGZAG.iter() {
            assert!(!seen[z]);
            seen[z] = true;
        }
        assert_eq!(ZIGZAG[1], 1);
        assert_eq!(ZIGZAG[2], 8);
        assert_eq!(ZIGZAG[63], 63);
    }

    #[test_case]
    fn idct_dc_only_is_flat() {
        // F(0,0) = 64 → every pixel 0.125·64·... : 2D IDCT of a DC-only block
        // is flat at DC/8 (basis 0.35355² · 64 = 8), level-shifted to 136.
        let mut coef = [0i32; 64];
        coef[0] = 64;
        let mut out = [0u8; 64];
        idct8x8(&coef, &mut out);
        assert!(out.iter().all(|&v| v == 136), "flat block, got {:?}", &out[..8]);
        // And the extremes clamp.
        coef[0] = 100_000;
        idct8x8(&coef, &mut out);
        assert!(out.iter().all(|&v| v == 255));
        coef[0] = -100_000;
        idct8x8(&coef, &mut out);
        assert!(out.iter().all(|&v| v == 0));
    }

    #[test_case]
    fn extend_matches_spec() {
        // Category 3: raw 0..=3 map to -7..=-4, raw 4..=7 stay 4..=7.
        assert_eq!(extend(0, 3), -7);
        assert_eq!(extend(3, 3), -4);
        assert_eq!(extend(4, 3), 4);
        assert_eq!(extend(7, 3), 7);
        assert_eq!(extend(0, 0), 0);
        assert_eq!(extend(1, 1), 1);
        assert_eq!(extend(0, 1), -1);
    }

    // Real files generated with Pillow on the host (see the task notes).
    const JPG_SOLID: &[u8] = &[255, 216, 255, 224, 0, 16, 74, 70, 73, 70, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 255, 219, 0, 67, 0, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 2, 2, 2, 2, 4, 3, 2, 2, 2, 2, 5, 4, 4, 3, 4, 6, 5, 6, 6, 6, 5, 6, 6, 6, 7, 9, 8, 6, 7, 9, 7, 6, 6, 8, 11, 8, 9, 10, 10, 10, 10, 10, 6, 8, 11, 12, 11, 10, 12, 9, 10, 10, 10, 255, 219, 0, 67, 1, 2, 2, 2, 2, 2, 2, 5, 3, 3, 5, 10, 7, 6, 7, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 255, 192, 0, 17, 8, 0, 16, 0, 16, 3, 1, 17, 0, 2, 17, 1, 3, 17, 1, 255, 196, 0, 31, 0, 0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 255, 196, 0, 181, 16, 0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 125, 1, 2, 3, 0, 4, 17, 5, 18, 33, 49, 65, 6, 19, 81, 97, 7, 34, 113, 20, 50, 129, 145, 161, 8, 35, 66, 177, 193, 21, 82, 209, 240, 36, 51, 98, 114, 130, 9, 10, 22, 23, 24, 25, 26, 37, 38, 39, 40, 41, 42, 52, 53, 54, 55, 56, 57, 58, 67, 68, 69, 70, 71, 72, 73, 74, 83, 84, 85, 86, 87, 88, 89, 90, 99, 100, 101, 102, 103, 104, 105, 106, 115, 116, 117, 118, 119, 120, 121, 122, 131, 132, 133, 134, 135, 136, 137, 138, 146, 147, 148, 149, 150, 151, 152, 153, 154, 162, 163, 164, 165, 166, 167, 168, 169, 170, 178, 179, 180, 181, 182, 183, 184, 185, 186, 194, 195, 196, 197, 198, 199, 200, 201, 202, 210, 211, 212, 213, 214, 215, 216, 217, 218, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 255, 196, 0, 31, 1, 0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 255, 196, 0, 181, 17, 0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 119, 0, 1, 2, 3, 17, 4, 5, 33, 49, 6, 18, 65, 81, 7, 97, 113, 19, 34, 50, 129, 8, 20, 66, 145, 161, 177, 193, 9, 35, 51, 82, 240, 21, 98, 114, 209, 10, 22, 36, 52, 225, 37, 241, 23, 24, 25, 26, 38, 39, 40, 41, 42, 53, 54, 55, 56, 57, 58, 67, 68, 69, 70, 71, 72, 73, 74, 83, 84, 85, 86, 87, 88, 89, 90, 99, 100, 101, 102, 103, 104, 105, 106, 115, 116, 117, 118, 119, 120, 121, 122, 130, 131, 132, 133, 134, 135, 136, 137, 138, 146, 147, 148, 149, 150, 151, 152, 153, 154, 162, 163, 164, 165, 166, 167, 168, 169, 170, 178, 179, 180, 181, 182, 183, 184, 185, 186, 194, 195, 196, 197, 198, 199, 200, 201, 202, 210, 211, 212, 213, 214, 215, 216, 217, 218, 226, 227, 228, 229, 230, 231, 232, 233, 234, 242, 243, 244, 245, 246, 247, 248, 249, 250, 255, 218, 0, 12, 3, 1, 0, 2, 17, 3, 17, 0, 63, 0, 243, 186, 254, 95, 63, 191, 2, 128, 10, 0, 40, 3, 255, 217];
    const JPG_GRAD420: &[u8] = &[255, 216, 255, 224, 0, 16, 74, 70, 73, 70, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 255, 219, 0, 67, 0, 3, 2, 2, 2, 2, 2, 3, 2, 2, 2, 3, 3, 3, 3, 4, 6, 4, 4, 4, 4, 4, 8, 6, 6, 5, 6, 9, 8, 10, 10, 9, 8, 9, 9, 10, 12, 15, 12, 10, 11, 14, 11, 9, 9, 13, 17, 13, 14, 15, 16, 16, 17, 16, 10, 12, 18, 19, 18, 16, 19, 15, 16, 16, 16, 255, 219, 0, 67, 1, 3, 3, 3, 4, 3, 4, 8, 4, 4, 8, 16, 11, 9, 11, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 255, 192, 0, 17, 8, 0, 11, 0, 17, 3, 1, 34, 0, 2, 17, 1, 3, 17, 1, 255, 196, 0, 31, 0, 0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 255, 196, 0, 181, 16, 0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 125, 1, 2, 3, 0, 4, 17, 5, 18, 33, 49, 65, 6, 19, 81, 97, 7, 34, 113, 20, 50, 129, 145, 161, 8, 35, 66, 177, 193, 21, 82, 209, 240, 36, 51, 98, 114, 130, 9, 10, 22, 23, 24, 25, 26, 37, 38, 39, 40, 41, 42, 52, 53, 54, 55, 56, 57, 58, 67, 68, 69, 70, 71, 72, 73, 74, 83, 84, 85, 86, 87, 88, 89, 90, 99, 100, 101, 102, 103, 104, 105, 106, 115, 116, 117, 118, 119, 120, 121, 122, 131, 132, 133, 134, 135, 136, 137, 138, 146, 147, 148, 149, 150, 151, 152, 153, 154, 162, 163, 164, 165, 166, 167, 168, 169, 170, 178, 179, 180, 181, 182, 183, 184, 185, 186, 194, 195, 196, 197, 198, 199, 200, 201, 202, 210, 211, 212, 213, 214, 215, 216, 217, 218, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 255, 196, 0, 31, 1, 0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 255, 196, 0, 181, 17, 0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 119, 0, 1, 2, 3, 17, 4, 5, 33, 49, 6, 18, 65, 81, 7, 97, 113, 19, 34, 50, 129, 8, 20, 66, 145, 161, 177, 193, 9, 35, 51, 82, 240, 21, 98, 114, 209, 10, 22, 36, 52, 225, 37, 241, 23, 24, 25, 26, 38, 39, 40, 41, 42, 53, 54, 55, 56, 57, 58, 67, 68, 69, 70, 71, 72, 73, 74, 83, 84, 85, 86, 87, 88, 89, 90, 99, 100, 101, 102, 103, 104, 105, 106, 115, 116, 117, 118, 119, 120, 121, 122, 130, 131, 132, 133, 134, 135, 136, 137, 138, 146, 147, 148, 149, 150, 151, 152, 153, 154, 162, 163, 164, 165, 166, 167, 168, 169, 170, 178, 179, 180, 181, 182, 183, 184, 185, 186, 194, 195, 196, 197, 198, 199, 200, 201, 202, 210, 211, 212, 213, 214, 215, 216, 217, 218, 226, 227, 228, 229, 230, 231, 232, 233, 234, 242, 243, 244, 245, 246, 247, 248, 249, 250, 255, 218, 0, 12, 3, 1, 0, 2, 17, 3, 17, 0, 63, 0, 249, 139, 193, 223, 179, 231, 220, 255, 0, 65, 244, 254, 26, 247, 79, 7, 126, 207, 159, 115, 253, 7, 211, 248, 107, 222, 60, 29, 161, 105, 31, 39, 250, 4, 93, 187, 87, 185, 248, 59, 66, 210, 62, 79, 244, 8, 187, 118, 175, 157, 203, 120, 130, 182, 135, 202, 120, 89, 226, 230, 101, 238, 111, 211, 169, 242, 167, 252, 51, 231, 253, 56, 255, 0, 227, 180, 87, 221, 127, 216, 122, 79, 252, 248, 69, 249, 81, 95, 67, 254, 176, 214, 63, 167, 63, 226, 46, 102, 94, 127, 121, 255, 217];
    const JPG_GRAY: &[u8] = &[255, 216, 255, 224, 0, 16, 74, 70, 73, 70, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 255, 219, 0, 67, 0, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 2, 2, 2, 2, 4, 3, 2, 2, 2, 2, 5, 4, 4, 3, 4, 6, 5, 6, 6, 6, 5, 6, 6, 6, 7, 9, 8, 6, 7, 9, 7, 6, 6, 8, 11, 8, 9, 10, 10, 10, 10, 10, 6, 8, 11, 12, 11, 10, 12, 9, 10, 10, 10, 255, 192, 0, 11, 8, 0, 9, 0, 9, 1, 1, 17, 0, 255, 196, 0, 31, 0, 0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 255, 196, 0, 181, 16, 0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 125, 1, 2, 3, 0, 4, 17, 5, 18, 33, 49, 65, 6, 19, 81, 97, 7, 34, 113, 20, 50, 129, 145, 161, 8, 35, 66, 177, 193, 21, 82, 209, 240, 36, 51, 98, 114, 130, 9, 10, 22, 23, 24, 25, 26, 37, 38, 39, 40, 41, 42, 52, 53, 54, 55, 56, 57, 58, 67, 68, 69, 70, 71, 72, 73, 74, 83, 84, 85, 86, 87, 88, 89, 90, 99, 100, 101, 102, 103, 104, 105, 106, 115, 116, 117, 118, 119, 120, 121, 122, 131, 132, 133, 134, 135, 136, 137, 138, 146, 147, 148, 149, 150, 151, 152, 153, 154, 162, 163, 164, 165, 166, 167, 168, 169, 170, 178, 179, 180, 181, 182, 183, 184, 185, 186, 194, 195, 196, 197, 198, 199, 200, 201, 202, 210, 211, 212, 213, 214, 215, 216, 217, 218, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 255, 218, 0, 8, 1, 1, 0, 0, 63, 0, 248, 206, 138, 40, 175, 255, 217];

    fn px(img: &Image, x: usize, y: usize) -> (i32, i32, i32) {
        let p = img.pixels[y * img.w + x];
        (((p >> 16) & 255) as i32, ((p >> 8) & 255) as i32, (p & 255) as i32)
    }

    fn close(a: (i32, i32, i32), b: (i32, i32, i32), tol: i32) -> bool {
        (a.0 - b.0).abs() <= tol && (a.1 - b.1).abs() <= tol && (a.2 - b.2).abs() <= tol
    }

    #[test_case]
    fn solid_16x16_444() {
        let img = decode(JPG_SOLID).unwrap();
        assert_eq!((img.w, img.h), (16, 16));
        // Solid (200,80,40) at quality 95; Pillow round-trips it to (199,80,40).
        assert!(close(px(&img, 8, 8), (200, 80, 40), 6), "got {:?}", px(&img, 8, 8));
        assert!(close(px(&img, 0, 0), (200, 80, 40), 6));
        assert!(close(px(&img, 15, 15), (200, 80, 40), 6));
    }

    #[test_case]
    fn gradient_17x11_420_odd_dims() {
        let img = decode(JPG_GRAD420).unwrap();
        assert_eq!((img.w, img.h), (17, 11));
        // Pillow's own decode of the same file: (0,0)→(0,2,0),
        // (8,5)→(121,114,104), (16,10)→(242,226,210). Chroma is 4:2:0 and we
        // upsample nearest (Pillow interpolates), so allow a loose tolerance.
        assert!(close(px(&img, 0, 0), (0, 2, 0), 16), "got {:?}", px(&img, 0, 0));
        assert!(close(px(&img, 8, 5), (121, 114, 104), 16), "got {:?}", px(&img, 8, 5));
        assert!(close(px(&img, 16, 10), (242, 226, 210), 16), "got {:?}", px(&img, 16, 10));
    }

    #[test_case]
    fn grayscale_single_component() {
        let img = decode(JPG_GRAY).unwrap();
        assert_eq!((img.w, img.h), (9, 9));
        assert!(close(px(&img, 4, 4), (77, 77, 77), 3), "got {:?}", px(&img, 4, 4));
    }

    #[test_case]
    fn corrupt_jpeg_errors_not_panics() {
        assert!(decode(&JPG_SOLID[..40]).is_err(), "truncated header");
        let n = JPG_SOLID.len();
        assert!(decode(&JPG_SOLID[..n - 30]).is_err(), "truncated entropy data");
        assert!(decode(b"\xff\xd8\xff\xd9").is_err(), "empty image");
    }
}
