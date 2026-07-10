//! Rasterize a [`Layout`] into an RGB buffer using runtime TTF ([`crate::font_ttf`])
//! and optional decoded images ([`crate::image`]). Draws form controls, a
//! scrollbar, and an optional loading progress bar (Ladybird/Web content chrome).

use super::layout::{ControlKind, FormControl, Layout};
use crate::font_ttf;
use alloc::vec;
use alloc::vec::Vec;

/// Optional chrome overlaid after content paint.
#[derive(Clone, Copy, Debug, Default)]
pub struct Chrome {
    /// 0..=100 loading progress; `None` = hide bar.
    pub progress: Option<u8>,
    /// Draw progress at bottom instead of top.
    pub progress_bottom: bool,
    /// Draw vertical scrollbar when content overflows.
    pub scrollbar: bool,
}

/// Paint `layout` into a `width * height` RGB buffer, scrolled by `scroll_y`.
pub fn paint(layout: &Layout, scroll_y: i32) -> Vec<u32> {
    paint_chrome(layout, scroll_y, Chrome::default())
}

/// Paint with scrollbar / progress chrome.
pub fn paint_chrome(layout: &Layout, scroll_y: i32, chrome: Chrome) -> Vec<u32> {
    let w = layout.width.max(1) as usize;
    let h = layout.height.max(1) as usize;
    let mut buf = vec![layout.bg; w * h];

    // Background rects under content.
    for r in &layout.rects {
        fill_rect(&mut buf, w, h, r.x, r.y - scroll_y, r.w, r.h, r.color);
    }

    // Images (already decoded RGB).
    for im in &layout.images {
        let y0 = im.y - scroll_y;
        if y0 + im.h < 0 || y0 >= layout.height {
            continue;
        }
        if let Some(ref px) = im.pixels {
            blit_image(&mut buf, w, h, im.x, y0, im.w, im.h, px, im.src_w, im.src_h);
        } else {
            fill_rect(&mut buf, w, h, im.x, y0, im.w, im.h, 0xe0e0e0);
            for dx in 0..im.w {
                put(&mut buf, w, h, im.x + dx, y0, 0x888888);
                put(&mut buf, w, h, im.x + dx, y0 + im.h - 1, 0x888888);
            }
            for dy in 0..im.h {
                put(&mut buf, w, h, im.x, y0 + dy, 0x888888);
                put(&mut buf, w, h, im.x + im.w - 1, y0 + dy, 0x888888);
            }
            if !im.alt.is_empty() {
                let _ = font_ttf::blit_run(
                    &mut buf,
                    w,
                    h,
                    im.x + 4,
                    y0 + 4,
                    &im.alt,
                    12.0,
                    0x444444,
                );
            }
        }
    }

    // Nested frames (iframe content or placeholder chrome).
    for fr in &layout.frames {
        let y0 = fr.y - scroll_y;
        if y0 + fr.h < 0 || y0 >= layout.height {
            continue;
        }
        if let Some(ref px) = fr.pixels {
            blit_image(
                &mut buf,
                w,
                h,
                fr.x,
                y0,
                fr.w,
                fr.h,
                px,
                fr.src_w.max(1),
                fr.src_h.max(1),
            );
            // Border
            for dx in 0..fr.w {
                put(&mut buf, w, h, fr.x + dx, y0, 0x888888);
                put(&mut buf, w, h, fr.x + dx, y0 + fr.h - 1, 0x888888);
            }
            for dy in 0..fr.h {
                put(&mut buf, w, h, fr.x, y0 + dy, 0x888888);
                put(&mut buf, w, h, fr.x + fr.w - 1, y0 + dy, 0x888888);
            }
        } else {
            fill_rect(&mut buf, w, h, fr.x, y0, fr.w, fr.h, 0xf0f0f0);
            for dx in 0..fr.w {
                put(&mut buf, w, h, fr.x + dx, y0, 0x888888);
                put(&mut buf, w, h, fr.x + dx, y0 + fr.h - 1, 0x888888);
            }
            for dy in 0..fr.h {
                put(&mut buf, w, h, fr.x, y0 + dy, 0x888888);
                put(&mut buf, w, h, fr.x + fr.w - 1, y0 + dy, 0x888888);
            }
            let label = match fr.kind {
                super::layout::EmbedKind::Video => "[video]",
                super::layout::EmbedKind::Audio => "[audio]",
                super::layout::EmbedKind::Canvas => "[canvas]",
                super::layout::EmbedKind::Iframe if !fr.srcdoc.is_empty() => "[iframe srcdoc]",
                super::layout::EmbedKind::Iframe if !fr.src.is_empty() => "[iframe]",
                _ => "[frame]",
            };
            let _ = font_ttf::blit_run(
                &mut buf,
                w,
                h,
                fr.x + 6,
                y0 + 6,
                label,
                12.0,
                0x555555,
            );
            if !fr.src.is_empty() {
                let _ = font_ttf::blit_run(
                    &mut buf,
                    w,
                    h,
                    fr.x + 6,
                    y0 + 22,
                    &fr.src,
                    11.0,
                    0x1a73e8,
                );
            }
        }
    }

    // Form controls.
    for c in &layout.controls {
        paint_control(&mut buf, w, h, c, scroll_y, layout.height);
    }

    // Text runs (baseline-correct TTF).
    for run in &layout.runs {
        let px = run.font_size.max(8) as f32;
        let line_h = font_ttf::line_height(px) as i32;
        let y = run.y - scroll_y;
        if y + line_h < 0 || y >= layout.height {
            continue;
        }
        let end_x = font_ttf::blit_run(&mut buf, w, h, run.x, y, &run.text, px, run.color);
        if run.bold {
            let _ = font_ttf::blit_run(&mut buf, w, h, run.x + 1, y, &run.text, px, run.color);
        }
        if run.link_href.is_some() {
            let uy = y + line_h - 2;
            if uy >= 0 && uy < layout.height {
                for x in run.x..end_x.max(run.x + 1) {
                    put(&mut buf, w, h, x, uy, run.color);
                }
            }
        }
    }

    if chrome.scrollbar {
        paint_scrollbar(&mut buf, w, h, layout, scroll_y);
    }
    if let Some(pct) = chrome.progress {
        paint_progress(&mut buf, w, h, pct, chrome.progress_bottom);
    }
    buf
}

fn paint_control(
    buf: &mut [u32],
    bw: usize,
    bh: usize,
    c: &FormControl,
    scroll_y: i32,
    view_h: i32,
) {
    if c.kind == ControlKind::Hidden || c.w <= 0 || c.h <= 0 {
        return;
    }
    let y0 = c.y - scroll_y;
    if y0 + c.h < 0 || y0 >= view_h {
        return;
    }
    match c.kind {
        ControlKind::Text | ControlKind::Password | ControlKind::TextArea => {
            let bg = if c.focused { 0xffffff } else { 0xfafafa };
            let border = if c.focused { 0x1a73e8 } else { 0x888888 };
            fill_rect(buf, bw, bh, c.x, y0, c.w, c.h, bg);
            // border
            for dx in 0..c.w {
                put(buf, bw, bh, c.x + dx, y0, border);
                put(buf, bw, bh, c.x + dx, y0 + c.h - 1, border);
            }
            for dy in 0..c.h {
                put(buf, bw, bh, c.x, y0 + dy, border);
                put(buf, bw, bh, c.x + c.w - 1, y0 + dy, border);
            }
            let text_owned: alloc::string::String = if c.value.is_empty() && !c.placeholder.is_empty()
            {
                c.placeholder.clone()
            } else if c.kind == ControlKind::Password {
                let n = c.value.chars().count().min(64);
                core::iter::repeat('*').take(n).collect()
            } else {
                c.value.clone()
            };
            let color = if c.value.is_empty() && !c.placeholder.is_empty() {
                0x999999u32
            } else {
                0x222222u32
            };
            let _ = font_ttf::blit_run(buf, bw, bh, c.x + 6, y0 + 4, &text_owned, 13.0, color);
            if c.focused {
                let tw = font_ttf::measure(&text_owned, 13.0) as i32;
                let cx = c.x + 6 + tw.min(c.w - 10);
                for dy in 4..(c.h - 4).max(5) {
                    put(buf, bw, bh, cx, y0 + dy, 0x1a73e8);
                }
            }
        }
        ControlKind::Submit | ControlKind::Button => {
            let bg = if c.focused { 0x1557b0 } else { 0x1a73e8 };
            fill_rect(buf, bw, bh, c.x, y0, c.w, c.h, bg);
            let label = if c.value.is_empty() {
                if c.kind == ControlKind::Submit {
                    "Submit"
                } else {
                    "Button"
                }
            } else {
                c.value.as_str()
            };
            let tw = font_ttf::measure(label, 13.0) as i32;
            let tx = c.x + ((c.w - tw) / 2).max(4);
            let _ = font_ttf::blit_run(buf, bw, bh, tx, y0 + 5, label, 13.0, 0xffffff);
        }
        ControlKind::Checkbox => {
            fill_rect(buf, bw, bh, c.x, y0, c.w, c.h, 0xffffff);
            for dx in 0..c.w {
                put(buf, bw, bh, c.x + dx, y0, 0x444444);
                put(buf, bw, bh, c.x + dx, y0 + c.h - 1, 0x444444);
            }
            for dy in 0..c.h {
                put(buf, bw, bh, c.x, y0 + dy, 0x444444);
                put(buf, bw, bh, c.x + c.w - 1, y0 + dy, 0x444444);
            }
            if c.checked {
                fill_rect(buf, bw, bh, c.x + 4, y0 + 4, c.w - 8, c.h - 8, 0x1a73e8);
            }
        }
        ControlKind::Hidden => {}
    }
}

fn paint_scrollbar(buf: &mut [u32], w: usize, h: usize, layout: &Layout, scroll_y: i32) {
    let content_h = layout.content_height.max(1);
    let view_h = layout.height.max(1);
    if content_h <= view_h {
        return;
    }
    let track_x = (layout.width - 8).max(0);
    let track_w = 6;
    // Track
    fill_rect(buf, w, h, track_x, 0, track_w, view_h, 0xd0d0d0);
    let max_scroll = (content_h - view_h).max(1);
    let thumb_h = ((view_h as i64 * view_h as i64) / content_h as i64)
        .clamp(16, view_h as i64) as i32;
    let thumb_y = ((scroll_y as i64 * (view_h - thumb_h) as i64) / max_scroll as i64) as i32;
    fill_rect(
        buf,
        w,
        h,
        track_x + 1,
        thumb_y.clamp(0, view_h - thumb_h),
        track_w - 2,
        thumb_h,
        0x666666,
    );
}

/// Thin determinate progress bar (Ladybird load progress affordance).
fn paint_progress(buf: &mut [u32], w: usize, h: usize, pct: u8, bottom: bool) {
    let pct = pct.min(100) as i32;
    let bar_h = 3i32;
    let y = if bottom {
        (h as i32 - bar_h).max(0)
    } else {
        0
    };
    fill_rect(buf, w, h, 0, y, w as i32, bar_h, 0xe8e0d8);
    let fill_w = (w as i32 * pct) / 100;
    if fill_w > 0 {
        fill_rect(buf, w, h, 0, y, fill_w, bar_h, 0xcc785c); // brand terracotta
    }
}

fn blit_image(
    buf: &mut [u32],
    bw: usize,
    bh: usize,
    dx: i32,
    dy: i32,
    dw: i32,
    dh: i32,
    src: &[u32],
    sw: usize,
    sh: usize,
) {
    if sw == 0 || sh == 0 || dw <= 0 || dh <= 0 {
        return;
    }
    for row in 0..dh {
        let sy = (row as usize * sh) / dh as usize;
        for col in 0..dw {
            let sx = (col as usize * sw) / dw as usize;
            let p = src[sy * sw + sx];
            put(buf, bw, bh, dx + col, dy + row, p);
        }
    }
}

fn fill_rect(buf: &mut [u32], w: usize, h: usize, x: i32, y: i32, rw: i32, rh: i32, color: u32) {
    for dy in 0..rh {
        for dx in 0..rw {
            put(buf, w, h, x + dx, y + dy, color);
        }
    }
}

/// FNV-1a checksum of the buffer.
pub fn checksum(buf: &[u32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &px in buf {
        for b in px.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

fn put(buf: &mut [u32], w: usize, h: usize, x: i32, y: i32, color: u32) {
    if x < 0 || y < 0 {
        return;
    }
    let (x, y) = (x as usize, y as usize);
    if x < w && y < h {
        buf[y * w + x] = color & 0x00ff_ffff;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::browser::{html, layout};

    #[test_case]
    fn paint_nonzero_checksum() {
        let doc = html::parse("<html><body><p>Hi</p></body></html>");
        let lay = layout::layout_document_plain(&doc.root, 160, 80);
        let buf = paint(&lay, 0);
        assert_eq!(buf.len(), 160 * 80);
        assert_ne!(checksum(&buf), 0);
        let bg = lay.bg;
        assert!(buf.iter().any(|&p| p != bg), "expected ink pixels");
    }

    #[test_case]
    fn paint_progress_and_form() {
        let doc = html::parse(
            r#"<html><body><form><input name="q" value="ab"><input type="submit"></form></body></html>"#,
        );
        let mut lay = layout::layout_document_plain(&doc.root, 320, 200);
        lay.content_height = 800; // force scrollbar
        let chrome = Chrome {
            progress: Some(40),
            progress_bottom: false,
            scrollbar: true,
        };
        let buf = paint_chrome(&lay, 0, chrome);
        assert_eq!(buf.len(), 320 * 200);
        // Top progress bar uses brand terracotta on some pixels.
        assert!(
            buf.iter().take(320 * 3).any(|&p| p == 0xcc785c),
            "expected progress pixels"
        );
    }
}
