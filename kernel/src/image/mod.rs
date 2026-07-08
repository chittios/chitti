//! Image decoding for the `/open` viewer: PNG (via the in-tree DEFLATE
//! decoder) and baseline JPEG, both as **pure functions** — bytes in, RGB
//! pixels out, no I/O and no panics on malformed input (every error is a
//! `Result`), so the whole pipeline is unit-testable off-hardware (the
//! standing rule: fiddly logic lives in pure functions; the shell just reads
//! the file and hands the pixels to the compositor).

pub mod inflate;
pub mod jpeg;
pub mod png;

use alloc::vec::Vec;

/// A decoded image: row-major `0x00RRGGBB` pixels (what
/// `framebuffer::present_surface` blits).
pub struct Image {
    pub w: usize,
    pub h: usize,
    pub pixels: Vec<u32>,
}

/// Box-average resize (never upscales past the source; callers pass a target
/// that already fits their pane). Averaging over the source box per target
/// pixel keeps downscaled photos smooth where nearest-neighbour would alias.
pub fn resize(img: &Image, nw: usize, nh: usize) -> Image {
    let (nw, nh) = (nw.max(1), nh.max(1));
    let mut pixels = Vec::with_capacity(nw * nh);
    for y in 0..nh {
        let sy0 = y * img.h / nh;
        let sy1 = (((y + 1) * img.h).div_ceil(nh)).clamp(sy0 + 1, img.h);
        for x in 0..nw {
            let sx0 = x * img.w / nw;
            let sx1 = (((x + 1) * img.w).div_ceil(nw)).clamp(sx0 + 1, img.w);
            let (mut r, mut g, mut b, mut n) = (0u32, 0u32, 0u32, 0u32);
            for sy in sy0..sy1 {
                for sx in sx0..sx1 {
                    let p = img.pixels[sy * img.w + sx];
                    r += (p >> 16) & 255;
                    g += (p >> 8) & 255;
                    b += p & 255;
                    n += 1;
                }
            }
            pixels.push(((r / n) << 16) | ((g / n) << 8) | (b / n));
        }
    }
    Image { w: nw, h: nh, pixels }
}

/// The largest `(w, h)` that fits inside `(maxw, maxh)` preserving aspect
/// ratio, never exceeding the source size (small images stay 1:1 — the
/// compositor integer-upscales them).
pub fn fit(w: usize, h: usize, maxw: usize, maxh: usize) -> (usize, usize) {
    if w <= maxw && h <= maxh {
        return (w, h);
    }
    // Scale by the tighter axis, in 16.16 fixed point.
    let s = ((maxw << 16) / w.max(1)).min((maxh << 16) / h.max(1));
    (((w * s) >> 16).max(1), ((h * s) >> 16).max(1))
}

/// Decode `bytes` by sniffing the container magic (PNG signature / JPEG SOI).
pub fn decode(bytes: &[u8]) -> Result<Image, &'static str> {
    if bytes.starts_with(&[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]) {
        png::decode(bytes)
    } else if bytes.starts_with(&[0xff, 0xd8]) {
        jpeg::decode(bytes)
    } else {
        Err("unknown image format (PNG and JPEG are supported)")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn unknown_magic_is_an_error() {
        assert!(decode(b"GIF89a....").is_err());
        assert!(decode(b"").is_err());
    }

    #[test_case]
    fn fit_preserves_aspect_and_never_upscales() {
        assert_eq!(fit(100, 50, 200, 200), (100, 50), "small image untouched");
        let (w, h) = fit(4000, 3000, 800, 600);
        assert!(w <= 800 && h <= 600);
        // 4:3 stays 4:3 (within a pixel of rounding).
        assert!((w as i64 * 3 - h as i64 * 4).abs() <= 4, "{}x{}", w, h);
        let (w, h) = fit(10_000, 10, 100, 100);
        assert!(w <= 100 && h == 1, "extreme aspect clamps to at least 1: {}x{}", w, h);
    }

    #[test_case]
    fn resize_box_average() {
        // 2x2 → 1x1 averages all four pixels.
        let img = Image { w: 2, h: 2, pixels: alloc::vec![0xff0000, 0x00ff00, 0x0000ff, 0x000000] };
        let out = resize(&img, 1, 1);
        assert_eq!(out.pixels, alloc::vec![(63 << 16) | (63 << 8) | 63]);
        // Identity resize is a copy.
        let same = resize(&img, 2, 2);
        assert_eq!(same.pixels, img.pixels);
    }
}
