//! PNG decoder: chunk walk (CRC-verified) → zlib-inflated IDAT → per-row
//! unfiltering → RGB pixels. Supports the non-interlaced baseline everything
//! actually ships: bit depth 8 for gray / RGB / gray+alpha / RGBA, palette at
//! 1/2/4/8 bits, gray at 1/2/4 bits, and 16-bit channels (high byte kept).
//! Alpha is composited onto black (the viewer pane is dark). Adam7 interlace
//! is rejected with a clear error. Malformed input returns `Err`, never
//! panics.

use super::inflate::zlib_decompress;
use super::Image;
use alloc::vec::Vec;

/// CRC-32 (IEEE, reflected) over `data` — PNG chunk checksums. Bitwise (no
/// table): chunks are small and this keeps the module self-contained.
/// `pub(crate)` so tests elsewhere can build a valid PNG: the ring-3 differential needs images
/// larger than any sensible checked-in fixture, and a chunk with a wrong CRC is rejected before
/// the interesting code runs.
pub(crate) fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xffff_ffffu32;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xedb8_8320 & mask);
        }
    }
    !crc
}

fn be32(b: &[u8]) -> u32 {
    u32::from_be_bytes([b[0], b[1], b[2], b[3]])
}

/// Paeth predictor (PNG filter type 4).
fn paeth(a: i32, b: i32, c: i32) -> u8 {
    let p = a + b - c;
    let (pa, pb, pc) = ((p - a).abs(), (p - b).abs(), (p - c).abs());
    if pa <= pb && pa <= pc {
        a as u8
    } else if pb <= pc {
        b as u8
    } else {
        c as u8
    }
}

pub fn decode(bytes: &[u8]) -> Result<Image, &'static str> {
    if bytes.len() < 8 || !bytes.starts_with(&[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]) {
        return Err("png: bad signature");
    }
    let (mut w, mut h) = (0usize, 0usize);
    let (mut depth, mut color) = (0u8, 0u8);
    let mut palette: Vec<(u8, u8, u8)> = Vec::new();
    let mut idat: Vec<u8> = Vec::new();
    let mut seen_ihdr = false;

    let mut off = 8;
    while off + 8 <= bytes.len() {
        let len = be32(&bytes[off..]) as usize;
        let typ = &bytes[off + 4..off + 8];
        let data_end = off + 8 + len;
        if data_end + 4 > bytes.len() {
            return Err("png: truncated chunk");
        }
        let data = &bytes[off + 8..data_end];
        if crc32(&bytes[off + 4..data_end]) != be32(&bytes[data_end..]) {
            return Err("png: chunk crc mismatch");
        }
        match typ {
            b"IHDR" => {
                if len != 13 {
                    return Err("png: bad IHDR");
                }
                w = be32(data) as usize;
                h = be32(&data[4..]) as usize;
                depth = data[8];
                color = data[9];
                if data[10] != 0 || data[11] != 0 {
                    return Err("png: unsupported compression/filter method");
                }
                if data[12] != 0 {
                    return Err("png: interlaced (Adam7) unsupported");
                }
                if w == 0 || h == 0 || w > 16384 || h > 16384 || w * h > (32 << 20) {
                    return Err("png: unreasonable dimensions");
                }
                seen_ihdr = true;
            }
            b"PLTE" => {
                if len % 3 != 0 || len > 3 * 256 {
                    return Err("png: bad PLTE");
                }
                palette = data.chunks_exact(3).map(|c| (c[0], c[1], c[2])).collect();
            }
            b"IDAT" => idat.extend_from_slice(data),
            b"IEND" => break,
            _ => {} // ancillary chunks (tRNS, gAMA, tEXt, …) are ignored
        }
        off = data_end + 4;
    }
    if !seen_ihdr {
        return Err("png: missing IHDR");
    }

    // Channels per pixel and legal bit depths per colour type (PNG §11.2.2).
    let channels: usize = match color {
        0 => 1, // gray
        2 => 3, // rgb
        3 => 1, // palette index
        4 => 2, // gray + alpha
        6 => 4, // rgba
        _ => return Err("png: bad colour type"),
    };
    let depth_ok = match color {
        0 => matches!(depth, 1 | 2 | 4 | 8 | 16),
        3 => matches!(depth, 1 | 2 | 4 | 8),
        _ => matches!(depth, 8 | 16),
    };
    if !depth_ok {
        return Err("png: bad bit depth");
    }
    if color == 3 && palette.is_empty() {
        return Err("png: palette image without PLTE");
    }

    let mut raw = zlib_decompress(&idat)?;
    // The compressed bytes are dead the moment they are inflated, and for a large photo they are
    // tens of MB. Dropping them here rather than at the end of the function is what lets the
    // *peak* be the inflated data plus the pixels, instead of all three at once.
    drop(idat);
    let bits_per_px = depth as usize * channels;
    let stride = (w * bits_per_px).div_ceil(8);
    if raw.len() < h * (stride + 1) {
        return Err("png: pixel data too short");
    }
    // Filtering works on byte units of one pixel (min 1).
    let fbpp = bits_per_px.div_ceil(8).max(1);

    // **Unfilter in place, inside `raw`.** A PNG filter references only the bytes to its left and
    // the row above — both already unfiltered once the walk is row-major — so the second buffer
    // this used to build is not needed at all.
    //
    // It used to allocate a `rows` of the whole image *plus a fresh `cur` per scanline*, which is
    // one allocation per row and a full extra copy of the pixel data. That is heap churn the
    // standing performance rule warns about, and in the ring-3 tenant it was worse than churn:
    // the arena there is a bump allocator, so a per-row allocation is never reclaimed and the
    // per-row buffers alone accounted for as much of the arena as the image itself — which is
    // what put a 32 MP PNG over the ceiling and made it refused rather than decoded.
    //
    // Every access is by index rather than by slice: the write target and the three references
    // live in the same buffer, which no pair of slices can express.
    for y in 0..h {
        let base = y * (stride + 1);
        let filter = raw[base];
        // Where the previous row's *data* starts — one row back, past its filter byte. Unused
        // (and deliberately not computed) on the first row, which has no row above it.
        let prev = base.wrapping_sub(stride);
        for x in 0..stride {
            let i = base + 1 + x;
            let rawb = raw[i] as i32;
            let a = if x >= fbpp { raw[i - fbpp] as i32 } else { 0 };
            let b = if y > 0 { raw[prev + x] as i32 } else { 0 };
            let c = if y > 0 && x >= fbpp { raw[prev + x - fbpp] as i32 } else { 0 };
            let v = match filter {
                0 => rawb,
                1 => rawb + a,
                2 => rawb + b,
                3 => rawb + (a + b) / 2,
                4 => rawb + paeth(a, b, c) as i32,
                _ => return Err("png: bad filter type"),
            };
            raw[i] = v as u8;
        }
    }

    // Expand unfiltered scanlines to 0xRRGGBB.
    let mut pixels: Vec<u32> = Vec::with_capacity(w * h);
    // Sample reader for sub-byte depths (gray / palette): value scaled later.
    let sample = |row: &[u8], i: usize| -> u32 {
        match depth {
            16 => row[i * 2] as u32, // high byte
            8 => row[i] as u32,
            d => {
                let per = 8 / d as usize;
                let byte = row[i / per];
                let shift = 8 - d as usize * (i % per + 1);
                ((byte >> shift) as u32) & ((1 << d) - 1)
            }
        }
    };
    // Scale a sub-byte gray sample to 0..=255.
    let scale = |v: u32| -> u32 {
        match depth {
            1 => v * 255,
            2 => v * 85,
            4 => v * 17,
            _ => v,
        }
    };
    let bytes_per_ch = if depth == 16 { 2 } else { 1 };
    for y in 0..h {
        // The unfiltered scanline, in place: past this row's filter byte.
        let row = &raw[y * (stride + 1) + 1..y * (stride + 1) + 1 + stride];
        for x in 0..w {
            let px = match color {
                0 => {
                    let g = scale(sample(row, x));
                    (g << 16) | (g << 8) | g
                }
                3 => {
                    let idx = sample(row, x) as usize;
                    let (r, g, b) = *palette.get(idx).ok_or("png: palette index out of range")?;
                    ((r as u32) << 16) | ((g as u32) << 8) | b as u32
                }
                _ => {
                    // 8/16-bit multichannel: gray+alpha, rgb, rgba.
                    let base = x * channels * bytes_per_ch;
                    let ch = |k: usize| row[base + k * bytes_per_ch] as u32;
                    let (r, g, b, a) = match color {
                        2 => (ch(0), ch(1), ch(2), 255),
                        4 => (ch(0), ch(0), ch(0), ch(1)),
                        _ => (ch(0), ch(1), ch(2), ch(3)), // 6
                    };
                    // Composite onto black: the viewer pane is dark.
                    let (r, g, b) = (r * a / 255, g * a / 255, b * a / 255);
                    (r << 16) | (g << 8) | b
                }
            };
            pixels.push(px);
        }
    }
    Ok(Image { w, h, pixels })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Generated on the host with python (zlib + struct); see the task notes.
    const PNG_RGB: &[u8] = &[137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 3, 0, 0, 0, 2, 8, 2, 0, 0, 0, 18, 22, 241, 77, 0, 0, 0, 24, 73, 68, 65, 84, 120, 218, 99, 248, 207, 192, 192, 0, 193, 92, 34, 114, 26, 70, 54, 191, 126, 126, 5, 0, 55, 98, 6, 184, 0, 92, 11, 101, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130];
    const PNG_RGBA: &[u8] = &[137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 2, 0, 0, 0, 2, 8, 6, 0, 0, 0, 114, 182, 13, 36, 0, 0, 0, 22, 73, 68, 65, 84, 120, 218, 99, 56, 145, 98, 244, 31, 136, 27, 24, 128, 224, 63, 16, 48, 0, 0, 72, 106, 8, 56, 96, 59, 129, 121, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130];
    const PNG_PAL: &[u8] = &[137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 4, 0, 0, 0, 1, 8, 3, 0, 0, 0, 206, 226, 255, 255, 0, 0, 0, 12, 80, 76, 84, 69, 255, 0, 0, 0, 255, 0, 0, 0, 255, 128, 128, 128, 204, 176, 70, 15, 0, 0, 0, 13, 73, 68, 65, 84, 120, 218, 99, 96, 96, 100, 98, 6, 0, 0, 15, 0, 7, 91, 208, 139, 125, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130];
    const PNG_GRAY_SUB: &[u8] = &[137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 3, 0, 0, 0, 1, 8, 0, 0, 0, 0, 62, 139, 75, 104, 0, 0, 0, 12, 73, 68, 65, 84, 120, 218, 99, 76, 49, 58, 7, 0, 2, 102, 1, 102, 132, 148, 253, 40, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130];

    #[test_case]
    fn rgb_3x2_exact_pixels() {
        let img = decode(PNG_RGB).unwrap();
        assert_eq!((img.w, img.h), (3, 2));
        assert_eq!(img.pixels, alloc::vec![0xff0000, 0x00ff00, 0x0000ff, 0x0a141e, 0x28323c, 0xfaf9f5]);
    }

    #[test_case]
    fn rgba_composites_alpha_onto_black() {
        let img = decode(PNG_RGBA).unwrap();
        assert_eq!((img.w, img.h), (2, 2));
        // (200,100,50)@255 stays; @128 halves; opaque black; transparent white.
        assert_eq!(img.pixels[0], 0xc86432);
        assert_eq!(img.pixels[1], 0x643219);
        assert_eq!(img.pixels[2], 0x000000);
        assert_eq!(img.pixels[3], 0x000000);
    }

    #[test_case]
    fn palette_lookup() {
        let img = decode(PNG_PAL).unwrap();
        assert_eq!((img.w, img.h), (4, 1));
        assert_eq!(img.pixels, alloc::vec![0xff0000, 0x00ff00, 0x0000ff, 0x808080]);
    }

    #[test_case]
    fn gray_with_sub_filter() {
        let img = decode(PNG_GRAY_SUB).unwrap();
        assert_eq!((img.w, img.h), (3, 1));
        // Raw samples 100, +50, +206 under filter 1 (Sub): 100, 150, 100.
        assert_eq!(img.pixels, alloc::vec![0x646464, 0x969696, 0x646464]);
    }

    #[test_case]
    fn corrupt_png_errors_not_panics() {
        assert!(decode(&PNG_RGB[..20]).is_err(), "truncated");
        let mut bad = PNG_RGB.to_vec();
        bad[40] ^= 0xff; // flip an IDAT byte: chunk CRC must catch it
        assert!(decode(&bad).is_err());
    }
}
