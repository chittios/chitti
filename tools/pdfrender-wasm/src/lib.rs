//! **PDF page rasterizer** for the Chitti `pdf` agent — `assets/render.wasm`.
//!
//! # Why a second module next to `tools/pdf-wasm`
//!
//! `pdf-wasm` (`pdf_digest`) extracts *text*, is `no_std`, and answers the
//! agent's questions about a document. It cannot draw. This module answers the
//! other question — *what does the page look like* — by rasterizing a page with
//! [hayro], a pure-Rust PDF interpreter over the `vello_cpu` CPU rasterizer.
//! They stay separate crates because this one needs `std` (hayro's tree pulls
//! `flate2`, `zune-jpeg`, `brotli`, `pic-scale`), and mixing that into the
//! `no_std` digest module would make the text path drag the renderer with it.
//!
//! # Why wasm and not ring 3
//!
//! A decoder needs no authority, so the standing rule prefers a ring-3 tenant
//! (`userspace/imgdec/`). That is not reachable here: hayro's dependency tree
//! is `std`-bound in three places that are not features (`moxcms`, `pxfm`,
//! `pic-scale`), and `vello_cpu`'s own `no_std` support does not lift them. So
//! the sandbox is **wasm** instead of a ring: this module declares **zero
//! imports**, and the kernel runs it under `agent::wasm_rt`'s page limiter with
//! a fuel bound. A malformed PDF therefore costs a discarded `Session`, not a
//! wild write — the same property the image tenant buys with a page table.
//!
//! # ABI — binary, not the string ABI
//!
//! A page is megabytes of pixels and a PDF is megabytes of bytes, so neither
//! direction goes through JSON/base64 (a 4 MiB page would be a 5.6 MiB string
//! to build, copy and parse). The host writes/reads linear memory directly:
//!
//! | Export | Signature | Meaning |
//! |--------|-----------|---------|
//! | `chitti_alloc` | `(i32) -> i32` | host-write staging (`wasm_rt` looks for this name) |
//! | `pdf_open` | `(ptr, len) -> i32` | page count, or a negative [error code](ERR_PARSE) |
//! | `pdf_page_size` | `(page, _) -> i32` | ptr to `[w, h]` u32 **millipoints**, 0 on error |
//! | `pdf_render` | `(page, scale_permille) -> i32` | ptr to `[w, h, pix_ptr, pix_len]` u32, 0 on error |
//! | `pdf_last_error` | `() -> i32` | why the last call returned 0 |
//!
//! Pixels are `vello_cpu`'s premultiplied RGBA8, 4 bytes per pixel, top-down,
//! no row padding — the host converts to its own `u32` format while copying
//! out (native, and it has to copy anyway).
//!
//! Two protocol rules, both deliberate:
//!
//! * **An oversized render is refused, never attempted.** Growing past
//!   [`MAX_PIXELS`] would hit the host's page limiter as an allocation failure,
//!   which under `panic = "abort"` is a trap that kills the instance — the
//!   document would have to be re-parsed to recover. So the scale is checked
//!   first and [`ERR_TOO_LARGE`] is reported, letting the host clamp and retry
//!   on the same instance (the image tenant's `STATUS_OUT_OF_MEMORY` shape).
//! * **The document and its render cache outlive a call.** Glyph outlines and
//!   decoded images are cached per document, so paging through a file must not
//!   re-parse it. One instance therefore holds one document: `pdf_open` on a
//!   live instance leaks the previous one, and the host is expected to build a
//!   fresh `Session` per file rather than reopening.

use std::sync::Arc;

use hayro::{RenderCache, RenderSettings};
use hayro_interpret::InterpreterSettings;
use hayro_syntax::Pdf;
use hayro_syntax::page::Page;

/// The file did not parse as a PDF.
pub const ERR_PARSE: i32 = -1;
/// No document is open (`pdf_open` was not called, or it failed).
pub const ERR_NO_DOC: i32 = -2;
/// Page index out of range.
pub const ERR_NO_PAGE: i32 = -3;
/// The requested scale would exceed [`MAX_PIXELS`]; retry smaller.
pub const ERR_TOO_LARGE: i32 = -4;
/// The document has no pages.
pub const ERR_EMPTY: i32 = -5;

/// Ceiling on one rendered page, in pixels (~8.3 MP = 33 MiB of RGBA).
///
/// A viewer only ever needs pane-sized pixels times a zoom factor; this exists
/// so a 200x zoom on a poster-sized page is refused rather than trapping.
pub const MAX_PIXELS: u64 = 8_300_000;

struct Doc {
    /// Borrowed from a leaked `Pdf` — see the module docs on one-doc-per-instance.
    pages: &'static [Page<'static>],
    cache: RenderCache<'static>,
}

static mut DOC: Option<Doc> = None;
/// `[w, h, pix_ptr, pix_len]`, read by the host after a successful `pdf_render`.
static mut HDR: [u32; 4] = [0; 4];
/// `[w, h]` millipoints, read after `pdf_page_size`.
static mut SIZE: [u32; 2] = [0; 2];
/// Keeps the last rendered page alive while the host copies it out; dropped at
/// the start of the next render, so at most one page's pixels are resident.
static mut PIX: Option<Vec<u8>> = None;
static mut LAST_ERR: i32 = 0;

fn doc() -> Option<&'static mut Doc> {
    // SAFETY: wasm32-unknown-unknown is single-threaded and the host serializes
    // calls into one instance, so no other reference to DOC can exist here.
    unsafe { (*(&raw mut DOC)).as_mut() }
}

fn fail(code: i32) -> i32 {
    // SAFETY: single-threaded, as above.
    unsafe { LAST_ERR = code };
    0
}

// --- inner API ---------------------------------------------------------------
// The work, addressed by value rather than by pointer. The `extern "C"` exports
// below are thin wrappers over these. Two reasons that split is worth it: the
// host harness (`tools/pdfbench`) can drive the *same* functions natively for a
// pixel-for-pixel differential against the wasm build — pointers in the ABI are
// 32-bit, so a native caller of the exports would truncate them — and the
// fallible logic is reachable from an ordinary `#[test]`.

/// Parse a document and retain it (with its render cache) for later renders.
/// Returns the page count, or one of the `ERR_*` codes.
pub fn open(bytes: Vec<u8>) -> Result<usize, i32> {
    let pdf = Pdf::new(Arc::new(bytes)).map_err(|_| ERR_PARSE)?;
    // Leak so the pages (which borrow the document) can be held across calls.
    let pdf: &'static Pdf = Box::leak(Box::new(pdf));
    let pages: &'static [Page<'static>] = pdf.pages();
    if pages.is_empty() {
        return Err(ERR_EMPTY);
    }
    let n = pages.len();
    // SAFETY: single-threaded, as above.
    unsafe { DOC = Some(Doc { pages, cache: RenderCache::new() }) };
    Ok(n)
}

/// Page size in points (as the PDF declares it, after rotation).
pub fn page_size(page: usize) -> Result<(f32, f32), i32> {
    let doc = doc().ok_or(ERR_NO_DOC)?;
    Ok(doc.pages.get(page).ok_or(ERR_NO_PAGE)?.render_dimensions())
}

/// Rasterize a page at `scale`, returning `(width, height, premultiplied RGBA8)`.
/// The pixels stay owned by the module (the host reads them out of memory before
/// the next call); [`MAX_PIXELS`] is enforced *before* allocating.
pub fn render_page(page: usize, scale: f32) -> Result<(u32, u32, &'static [u8]), i32> {
    let doc = doc().ok_or(ERR_NO_DOC)?;
    let p = doc.pages.get(page).ok_or(ERR_NO_PAGE)?;
    if !(scale.is_finite() && scale > 0.0) {
        return Err(ERR_TOO_LARGE);
    }
    // Refuse before allocating: see the module docs — a page too big to hold is
    // an answerable status, while a trap would cost the parsed document.
    let (pw, ph) = p.render_dimensions();
    let (ow, oh) = ((pw * scale).floor() as u64, (ph * scale).floor() as u64);
    if ow == 0 || oh == 0 || ow.saturating_mul(oh) > MAX_PIXELS {
        return Err(ERR_TOO_LARGE);
    }

    // Drop the previous page's pixels first, so peak use is one page, not two.
    // SAFETY: single-threaded, as above.
    unsafe { PIX = None };

    let settings = RenderSettings {
        x_scale: scale,
        y_scale: scale,
        // Opaque white: a PDF page is paper, and a transparent background would
        // composite the pane's wallpaper through the glyph antialiasing.
        bg_color: hayro::vello_cpu::color::palette::css::WHITE,
        ..Default::default()
    };
    let pixmap = hayro::render(p, &doc.cache, &InterpreterSettings::default(), &settings);
    let (w, h) = (pixmap.width() as u32, pixmap.height() as u32);
    let bytes = premul_rgba_to_bytes(pixmap.take());
    // SAFETY: single-threaded, as above. The slice borrows PIX, which lives
    // until the next `render_page` — the host copies its pixels out first.
    unsafe {
        PIX = Some(bytes);
        let held = (*(&raw const PIX)).as_ref().unwrap();
        Ok((w, h, core::slice::from_raw_parts(held.as_ptr(), held.len())))
    }
}

/// Staging allocation for host writes (`wasm_rt::Session::guest_alloc`).
#[no_mangle]
pub extern "C" fn chitti_alloc(len: i32) -> i32 {
    if len < 0 {
        return -1;
    }
    let mut v = Vec::<u8>::with_capacity(len as usize);
    let p = v.as_mut_ptr();
    std::mem::forget(v);
    p as i32
}

/// Why the last `pdf_page_size` / `pdf_render` returned 0.
#[no_mangle]
pub extern "C" fn pdf_last_error() -> i32 {
    // SAFETY: single-threaded, as above.
    unsafe { LAST_ERR }
}

/// Parse the PDF at `ptr..ptr+len`; returns the page count or a negative error.
#[no_mangle]
pub extern "C" fn pdf_open(ptr: i32, len: i32) -> i32 {
    if ptr <= 0 || len <= 0 {
        return ERR_PARSE;
    }
    // SAFETY: the host wrote `len` bytes at `ptr` (from `chitti_alloc`) before
    // calling; both are bounds-checked on its side against linear memory.
    let bytes = unsafe { std::slice::from_raw_parts(ptr as *const u8, len as usize) }.to_vec();
    match open(bytes) {
        Ok(n) => n as i32,
        Err(e) => e,
    }
}

/// Page dimensions in **millipoints** (points x 1000), so the host can pick a
/// fit-to-pane scale without rendering first. Returns a pointer to `[w, h]`.
#[no_mangle]
pub extern "C" fn pdf_page_size(page: i32, _unused: i32) -> i32 {
    let Ok(page) = usize::try_from(page) else { return fail(ERR_NO_PAGE) };
    match page_size(page) {
        Ok((w, h)) => {
            // SAFETY: single-threaded, as above.
            unsafe {
                SIZE = [(w as f64 * 1000.0) as u32, (h as f64 * 1000.0) as u32];
                LAST_ERR = 0;
                (&raw const SIZE) as i32
            }
        }
        Err(e) => fail(e),
    }
}

/// Rasterize `page` at `scale_permille / 1000`. Returns a pointer to
/// `[w, h, pix_ptr, pix_len]`, or 0 with [`pdf_last_error`] set.
#[no_mangle]
pub extern "C" fn pdf_render(page: i32, scale_permille: i32) -> i32 {
    let Ok(page) = usize::try_from(page) else { return fail(ERR_NO_PAGE) };
    match render_page(page, scale_permille as f32 / 1000.0) {
        Ok((w, h, pixels)) => {
            // SAFETY: single-threaded, as above.
            unsafe {
                HDR = [w, h, pixels.as_ptr() as u32, pixels.len() as u32];
                LAST_ERR = 0;
                (&raw const HDR) as i32
            }
        }
        Err(e) => fail(e),
    }
}

/// Reinterpret the pixmap's `PremulRgba8` vector as bytes — same allocation,
/// 4 bytes per pixel, no copy (the host reads it straight out of memory).
fn premul_rgba_to_bytes(v: Vec<hayro::vello_cpu::color::PremulRgba8>) -> Vec<u8> {
    let mut v = std::mem::ManuallyDrop::new(v);
    let (ptr, len, cap) = (v.as_mut_ptr(), v.len(), v.capacity());
    // SAFETY: `PremulRgba8` is a `#[repr(C)]` 4-byte struct of `u8` fields with
    // no padding and alignment 1, so the buffer is a valid `Vec<u8>` of 4x the
    // element count; the original vector is not dropped (ManuallyDrop).
    unsafe { Vec::from_raw_parts(ptr as *mut u8, len * 4, cap * 4) }
}
