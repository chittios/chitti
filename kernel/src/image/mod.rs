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

/// Rotate by `quads`×90° clockwise (`quads` taken mod 4). A pure transform on
/// the pixel grid — the viewer's `r`/`R` keys step this. 90°/270° swap the
/// dimensions; 0° is a plain clone.
pub fn rotate90(img: &Image, quads: u32) -> Image {
    let q = quads % 4;
    if q == 0 {
        return Image { w: img.w, h: img.h, pixels: img.pixels.clone() };
    }
    let (w, h) = (img.w, img.h);
    let (nw, nh) = if q == 2 { (w, h) } else { (h, w) };
    let mut pixels = alloc::vec![0u32; nw * nh];
    for y in 0..h {
        for x in 0..w {
            let p = img.pixels[y * w + x];
            let (nx, ny) = match q {
                1 => (h - 1 - y, x),             // 90° clockwise
                2 => (w - 1 - x, h - 1 - y),     // 180°
                _ => (y, w - 1 - x),             // 270° clockwise (= 90° CCW)
            };
            pixels[ny * nw + nx] = p;
        }
    }
    Image { w: nw, h: nh, pixels }
}

/// Render `src` into a `pane_w`×`pane_h` view buffer for the interactive viewer:
/// rotate by `rot_quads`×90°, scale to `zoom_pct` percent of the fit-to-pane
/// size (100 = fit, 200 = 2×, …), then blit centred plus a `(pan_x, pan_y)`
/// offset in view pixels (+x right, +y down). Anything outside the image is
/// filled with `bg` (the letterbox). The result is exactly pane-sized, so the
/// compositor presents it 1:1. Pure — no I/O, no theme lookup.
pub fn render_view(
    src: &Image,
    pane_w: usize,
    pane_h: usize,
    zoom_pct: u32,
    rot_quads: u32,
    pan_x: i64,
    pan_y: i64,
    bg: u32,
) -> Image {
    let (pane_w, pane_h) = (pane_w.max(1), pane_h.max(1));
    let oriented = rotate90(src, rot_quads);
    let (fw, fh) = fit(oriented.w, oriented.h, pane_w, pane_h);
    let dw = (fw as u64 * zoom_pct as u64 / 100).max(1) as usize;
    let dh = (fh as u64 * zoom_pct as u64 / 100).max(1) as usize;
    let disp = resize(&oriented, dw, dh);
    let mut pixels = alloc::vec![bg; pane_w * pane_h];
    let base_x = (pane_w as i64 - dw as i64) / 2 + pan_x;
    let base_y = (pane_h as i64 - dh as i64) / 2 + pan_y;
    for dy in 0..dh {
        let py = base_y + dy as i64;
        if py < 0 || py >= pane_h as i64 {
            continue;
        }
        for dx in 0..dw {
            let px = base_x + dx as i64;
            if px < 0 || px >= pane_w as i64 {
                continue;
            }
            pixels[py as usize * pane_w + px as usize] = disp.pixels[dy * dw + dx];
        }
    }
    Image { w: pane_w, h: pane_h, pixels }
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
    fn rotate90_quadrants() {
        // 2x3 image, distinct pixels laid out row-major:
        //   0 1
        //   2 3
        //   4 5
        let img = Image { w: 2, h: 3, pixels: alloc::vec![0, 1, 2, 3, 4, 5] };
        // 0° is an identity clone.
        assert_eq!(rotate90(&img, 0).pixels, img.pixels);
        // 90° CW → 3x2:  4 2 0 / 5 3 1
        let r1 = rotate90(&img, 1);
        assert_eq!((r1.w, r1.h), (3, 2));
        assert_eq!(r1.pixels, alloc::vec![4, 2, 0, 5, 3, 1]);
        // 180° → 2x3 reversed.
        let r2 = rotate90(&img, 2);
        assert_eq!((r2.w, r2.h), (2, 3));
        assert_eq!(r2.pixels, alloc::vec![5, 4, 3, 2, 1, 0]);
        // 270° CW → 3x2:  1 3 5 / 0 2 4
        let r3 = rotate90(&img, 3);
        assert_eq!((r3.w, r3.h), (3, 2));
        assert_eq!(r3.pixels, alloc::vec![1, 3, 5, 0, 2, 4]);
        // Four 90° turns come back to the original; mod-4 wraps.
        assert_eq!(rotate90(&img, 4).pixels, img.pixels);
    }

    #[test_case]
    fn render_view_fits_centres_and_letterboxes() {
        // A 2x2 image into a 6x6 pane at zoom 100 fits 1:1 (fit never upscales),
        // centred with a `bg` letterbox all around.
        let img = Image { w: 2, h: 2, pixels: alloc::vec![0x111111, 0x222222, 0x333333, 0x444444] };
        let v = render_view(&img, 6, 6, 100, 0, 0, 0, 0xff00ff);
        assert_eq!((v.w, v.h), (6, 6));
        // Corners are letterbox; the 2x2 lands centred at (2,2)..(4,4).
        assert_eq!(v.pixels[0], 0xff00ff);
        assert_eq!(v.pixels[2 * 6 + 2], 0x111111);
        assert_eq!(v.pixels[2 * 6 + 3], 0x222222);
        assert_eq!(v.pixels[3 * 6 + 2], 0x333333);
        // Zoom 200% doubles it to 4x4 (nearest-neighbour upscale via resize),
        // still centred, still pane-sized.
        let z = render_view(&img, 6, 6, 200, 0, 0, 0, 0);
        assert_eq!((z.w, z.h), (6, 6));
        assert_eq!(z.pixels[1 * 6 + 1], 0x111111); // top-left of the 4x4 block
    }

    #[test_case]
    fn render_view_pan_shifts_the_image() {
        let img = Image { w: 2, h: 2, pixels: alloc::vec![0xaa, 0xbb, 0xcc, 0xdd] };
        // Centred, the top-left source pixel is at pane (2,2). Pan +1,+1 moves it
        // to (3,3).
        let v = render_view(&img, 6, 6, 100, 0, 1, 1, 0);
        assert_eq!(v.pixels[3 * 6 + 3], 0xaa);
        assert_eq!(v.pixels[2 * 6 + 2], 0); // vacated cell is now letterbox
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
