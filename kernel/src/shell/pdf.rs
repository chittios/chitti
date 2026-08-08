//! pdf
//!
//! The **PDF viewer tab**: a real rasterized page in an action pane, with page
//! navigation, zoom and scrolling — the sibling of the image viewer in
//! [`super::media`] and the video player in [`super::video`].
//!
//! # What renders the page
//!
//! [hayro], a pure-Rust PDF interpreter over the `vello_cpu` CPU rasterizer,
//! compiled to wasm (`tools/pdfrender-wasm`, blob at `assets/wasm/pdfrender.wasm`)
//! and run under [`crate::agent::wasm_rt`]. The module declares **zero imports**
//! and gets a memory-page limiter plus a per-call fuel bound, so a malformed
//! document costs a discarded `Session`, never a wild write — the property the
//! image tenant buys with a page table, which is unavailable here because
//! hayro's dependency tree is `std`-bound (see the module docs in
//! `tools/pdfrender-wasm/src/lib.rs` for why ring 3 is not an option).
//!
//! The renderer is **kernel infrastructure, not an agent asset**: it is 4 MiB,
//! and an agent asset is written into the store at every boot and read back on
//! every open. It is a decoder — it needs no capability and no SOUL — so it
//! lives in the image like the in-kernel PNG/JPEG decoders. The `pdf` *agent*
//! still owns the `/open` hook and the text digest it answers questions from.
//!
//! # Performance, and why the numbers shaped the design
//!
//! Measured with `tools/pdfbench` (native and wasmi, same crate, on the host):
//! a dense LaTeX page is ~10-35 ms native and ~0.15-0.4 s under wasmi at a
//! pane-fit scale. The interpreter tax is **3-30x**, which it only is because the
//! guest is built with `simd128` and run on a wasmi with the `simd` feature: as
//! plain scalar wasm on wasmi 0.40 the same pages were 30-90x, and the blend-heavy
//! ones up to 340x. Three consequences, all of which survive the speedup:
//!
//! * **A render is a bounded unit of work with a pump around it.** The kernel is
//!   cooperative, so the viewer renders one page per user action and pumps
//!   [`crate::shell::upkeep`] + polls Ctrl+C *between* renders, the same
//!   granularity as the chunked AAC decode. It does not pump *inside* a render:
//!   wasmi has no yield point to hang that on, and fuel exhaustion is a trap
//!   rather than a resumable stop.
//! * **The rendered page is cached**, keyed by `(page, scale)`. Switching tabs,
//!   scrolling, and panning then cost a memcpy, not a re-render — only a page
//!   change or a zoom change pays the renderer.
//! * **The document stays parsed** inside one wasm instance, so paging through a
//!   35-page paper parses it once. That is why the `Session` is retained rather
//!   than rebuilt per page.

use super::*;

/// Surface id the viewer presents pages on (== framebuffer::PDF_SURFACE).
#[cfg(not(feature = "server"))]
pub(super) const PDF_SURFACE: u32 = u32::MAX - 3;

/// The rasterizer, in the kernel image. See the module docs for why this is not
/// an agent asset.
#[cfg(not(feature = "server"))]
static RENDER_WASM: &[u8] = include_bytes!("../../../assets/wasm/pdfrender.wasm");

/// Linear-memory ceiling for the renderer, in 64 KiB pages (128 MiB).
///
/// Measured on a **real** document, which is the only way this number is worth
/// anything: the Transformer paper (arXiv:1706.03762 — not in `/samples`, its
/// licence is not redistributable; fetch it with `/http -O` to re-measure) peaks
/// at 56 MiB of linear memory at a pane-fit scale and **70 MiB** at full zoom — the
/// glyph and decoded-image caches, not the pixmap. A synthetic one-page fixture
/// had suggested 33 MiB, and a 64 MiB cap sized from it would have failed on
/// page 3 of the first real paper anyone opened. The limiter is what turns "this
/// document wants more memory than we will give it" into a clean refusal rather
/// than kernel heap pressure.
#[cfg(not(feature = "server"))]
const PDF_MEM_PAGES: u32 = 2048;

/// Fuel for one page render.
///
/// Same source: the paper's heaviest page (a full-page attention-matrix figure)
/// costs **2.4 Gfuel** at full zoom, so this is ~3x headroom over the worst thing
/// actually observed, and at the measured ~770 Mfuel/s it bounds a runaway to
/// ~10 s. It stays a real bound rather than a formality — a render is the one
/// call the shell cannot interrupt from outside (wasmi has no yield point), so
/// fuel is the Ctrl+C rule's backstop.
///
/// NB the same page cost 7.7 Gfuel before the guest was built with `simd128`: a
/// vector instruction does the work of several scalar ones and is charged once, so
/// enabling SIMD *lowered* fuel use as well as wall time. A fuel budget is
/// therefore not portable across a change in how the guest is compiled.
#[cfg(not(feature = "server"))]
const PDF_RENDER_FUEL: u64 = 8_000_000_000;

/// Fuel for parsing a document (xref, page tree). Cheap next to a render
/// (~30-45 ms), but a crafted xref loop is exactly the sort of thing that should
/// hit a bound rather than hang the shell.
#[cfg(not(feature = "server"))]
const PDF_OPEN_FUEL: u64 = 2_000_000_000;

/// Function-table ceiling for the renderer: its declared table is 693 entries,
/// so this is ~1.5x headroom for a rebuilt module (each entry is a funcref, i.e.
/// bytes — the bound is about refusing a runaway, not about memory).
#[cfg(not(feature = "server"))]
const PDF_TABLE_ELEMS: u32 = 1024;

/// Cap on the document we will hold: the whole file goes into guest memory, so
/// this is bounded by [`PDF_MEM_PAGES`] together with the page it renders.
#[cfg(not(feature = "server"))]
const MAX_PDF_BYTES: usize = 24 << 20;

/// An open document: the live wasm instance (document parsed, glyph/image caches
/// warm), the view state, and the last rendered page.
#[cfg(not(feature = "server"))]
pub(super) struct PdfTab {
    session: crate::agent::wasm_rt::Session,
    name: String,
    path: String,
    view: crate::pdfview::View,
    /// Page size in millipoints for the *current* page — pages in one document
    /// can differ in size, so this is re-read on every page change.
    page_mpt: (u32, u32),
    /// `(page, permille, w, h, pixels)` — the cache that makes scrolling and tab
    /// switches free. Invalidated by a page or scale change, never by a pan.
    rendered: Option<(usize, u32, u64, u64, alloc::vec::Vec<u32>)>,
    /// Wall-clock ms the last render took, for the HUD.
    last_ms: u64,
    /// The scale the cached page was rendered at, so the HUD reports what is on
    /// screen rather than what was requested.
    permille: u32,
}

#[cfg(not(feature = "server"))]
pub(super) static PDF: crate::mm::Locked<Option<PdfTab>> = crate::mm::Locked::new(None);

/// Whether a document is open (gates the viewer's key handling, so a focused but
/// empty pane never eats keystrokes — the `video_loaded()` rule).
#[cfg(not(feature = "server"))]
pub(super) fn pdf_loaded() -> bool {
    PDF.with(|s| s.is_some())
}
#[cfg(feature = "server")]
pub(super) fn pdf_loaded() -> bool {
    false
}

/// `/open <path>.pdf` — render page 1 into a "pdf" action-pane tab.
///
/// Returns the summary line for the caller to print (this is a tool entry point,
/// reached through the pdf agent's `/open` command hook).
#[cfg(all(not(feature = "server"), not(test)))]
pub(super) fn view_pdf(path: &str) -> String {
    let t0 = crate::arch::now_ms();
    let Some(bytes) = read_mounted(path).or_else(|| crate::synapse::fs::read(path)) else {
        return alloc::format!("error: {path} not found under any mount or in the store");
    };
    if bytes.len() > MAX_PDF_BYTES {
        return alloc::format!(
            "error: {path} is {} MiB — the viewer caps at {} MiB",
            bytes.len() >> 20,
            MAX_PDF_BYTES >> 20
        );
    }
    // Drop any previously open document first: one instance holds one document
    // (the guest leaks the old one on reopen, by design — see its module docs),
    // and this also frees its linear memory before the new one is allocated.
    PDF.with(|s| *s = None);

    let limits = crate::agent::wasm_rt::Limits::default()
        .with_fuel(PDF_OPEN_FUEL)
        .with_pages(PDF_MEM_PAGES)
        // The rasterizer dispatches through trait objects, so its indirect-call
        // table is 693 entries — far past the 256 every app tool needs, and a
        // table that small refuses the module at instantiate time.
        .with_table_elems(PDF_TABLE_ELEMS);
    let mut session = match crate::agent::wasm_rt::Session::instantiate_page(RENDER_WASM, limits) {
        Ok(s) => s,
        Err(e) => return alloc::format!("error: pdf renderer: {e}"),
    };
    let ptr = match session.put_bytes(&bytes) {
        Ok(p) => p,
        Err(e) => return alloc::format!("error: pdf renderer: {e}"),
    };
    let pages = match session.call_i32_i32("pdf_open", ptr as i32, bytes.len() as i32) {
        Ok(n) if n > 0 => n as usize,
        Ok(n) => return alloc::format!("error: cannot read {path}: {}", open_error(n)),
        Err(e) => return alloc::format!("error: pdf renderer: {e}"),
    };

    let mut tab = PdfTab {
        session,
        name: path.rsplit('/').next().unwrap_or(path).to_string(),
        path: path.to_string(),
        view: crate::pdfview::View::new(pages),
        page_mpt: (595_000, 842_000), // replaced by the real size below
        rendered: None,
        last_ms: 0,
        permille: 1000,
    };
    if let Some(sz) = page_size(&mut tab.session, 0) {
        tab.page_mpt = sz;
    }
    PDF.with(|s| *s = Some(tab));

    // Open the tab, then render. Controls activate once the action pane is
    // focused (Ctrl+Tab / click) — the same gating as pane scroll, so typing at
    // the prompt is never eaten.
    crate::framebuffer::set_right(crate::framebuffer::RightMode::Surface(PDF_SURFACE));
    render_pdf();
    let (ms, ok) = PDF.with(|s| s.as_ref().map(|t| (t.last_ms, t.rendered.is_some())).unwrap_or((0, false)));
    if !ok {
        return alloc::format!("error: {path}: {pages} page(s) parsed, but page 1 did not render");
    }
    alloc::format!(
        "ok: pdf {} — {pages} page(s), page 1 rendered in {ms} ms, opened in {} ms  (Ctrl+Tab to focus pane, then PgUp/PgDn pages, +/- zoom, arrows scroll, f fit, 0 reset; Ctrl+C closes)",
        tab_name(),
        crate::arch::now_ms().saturating_sub(t0)
    )
}
#[cfg(not(all(not(feature = "server"), not(test))))]
pub(super) fn view_pdf(path: &str) -> String {
    alloc::format!("error: pdf viewer unavailable in this build ({path})")
}

#[cfg(all(not(feature = "server"), not(test)))]
fn tab_name() -> String {
    PDF.with(|s| s.as_ref().map(|t| t.name.clone()).unwrap_or_default())
}

/// Map `pdf_open`'s negative return to a human reason. Mirrors the `ERR_*`
/// constants in `tools/pdfrender-wasm` — the guest reports *which* failure, and
/// collapsing them all to "cannot open" would throw away the only diagnosis a
/// user gets (an encrypted file and a truncated one need different answers).
#[cfg(not(feature = "server"))]
fn open_error(code: i32) -> &'static str {
    match code {
        -1 => "not a PDF, or its cross-reference table is damaged/encrypted",
        -5 => "the document has no pages",
        _ => "the renderer refused it",
    }
}

/// Ask the guest for a page's size in millipoints.
#[cfg(not(feature = "server"))]
fn page_size(session: &mut crate::agent::wasm_rt::Session, page: usize) -> Option<(u32, u32)> {
    let ptr = session.call_i32_i32("pdf_page_size", page as i32, 0).ok()?;
    if ptr <= 0 {
        return None;
    }
    let v = session.get_u32s(ptr as usize, 2).ok()?;
    (v[0] > 0 && v[1] > 0).then_some((v[0], v[1]))
}

/// Render the current page at the current zoom and present it, using the cache
/// when the page and scale are unchanged. Also the repaint-on-tab-switch path
/// (surfaces are not otherwise backed).
#[cfg(all(not(feature = "server"), not(test)))]
pub(super) fn render_pdf() {
    let hud_h = crate::framebuffer::pdf_hud_height();
    let Some((pane_w, pane_h)) = crate::framebuffer::surface_dims_px(PDF_SURFACE) else {
        return;
    };
    // The page occupies the pane above the reserved status strip.
    let pane = (pane_w as u64, (pane_h as u64).saturating_sub(hud_h).max(1));
    let bg = crate::framebuffer::pane_bg().unwrap_or(0);

    let mut hud = None;
    PDF.with(|slot| {
        let Some(t) = slot.as_mut() else { return };
        let want = crate::pdfview::render_permille(t.page_mpt, pane, t.view.fit, t.view.zoom);
        let page = t.view.page;
        // Re-render only when the page or the scale changed; a scroll or a tab
        // switch reuses the pixels.
        let fresh = matches!(t.rendered, Some((p, s, _, _, _)) if p == page && s == want);
        if !fresh {
            // Say what is happening first. A dense page is ~1 s in the
            // interpreter (990 ms for this repo's own paper, measured in-kernel),
            // and a second of a silent pane is indistinguishable from a hang —
            // the call itself cannot report progress, so the announcement has to
            // come before it.
            crate::framebuffer::draw_pdf_status(&alloc::format!(
                "rendering page {} of {}  at {}%  ...",
                page + 1,
                t.view.pages.max(1),
                want / 10
            ));
            let t0 = crate::arch::now_ms();
            match render_one(&mut t.session, page, want) {
                Ok((w, h, pixels)) => {
                    t.rendered = Some((page, want, w, h, pixels));
                    t.permille = want;
                    t.last_ms = crate::arch::now_ms().saturating_sub(t0);
                }
                Err(e) => {
                    crate::ktrace::log_fmt(format_args!("pdf: {}: page {} render failed: {}", t.path, page + 1, e));
                    serial_println!("pdf> page {} did not render: {e}", page + 1);
                    // Keep whatever was on screen rather than blanking the pane:
                    // a failed page is a page, not a closed document.
                    t.rendered = None;
                }
            }
        }
        let Some((_, _, w, h, pixels)) = t.rendered.as_ref() else { return };
        let (w, h) = (*w, *h);
        // A pan beyond the page (a smaller re-render, or the "bottom of the
        // previous page" sentinel from `pdfview::scroll`) is clamped here, once
        // the real render size is known.
        let (px, py) = crate::pdfview::clamp_pan((w, h), pane, (t.view.pan_x, t.view.pan_y));
        t.view.pan_x = px;
        t.view.pan_y = py;
        let frame = crate::pdfview::compose((w, h), pixels, pane, (px, py), bg);
        crate::framebuffer::present_surface_reserve(PDF_SURFACE, pane.0 as usize, pane.1 as usize, &frame, hud_h);
        hud = Some(crate::pdfview::hud(&t.view, t.permille, t.last_ms));
    });
    if let Some(line) = hud {
        crate::framebuffer::draw_pdf_status(&line);
    }
}
#[cfg(not(all(not(feature = "server"), not(test))))]
pub(super) fn render_pdf() {}

/// One guest render: ask for the page, then copy the pixels out and convert them
/// to the compositor's format.
///
/// Every number the guest reports is bounds-checked by `get_bytes`/`get_u32s`
/// against live linear memory, and the geometry is re-derived from the header
/// rather than assumed from the request — the image tenant's rule, because a
/// guest is the untrusted side even when it is our own code.
#[cfg(all(not(feature = "server"), not(test)))]
fn render_one(
    session: &mut crate::agent::wasm_rt::Session,
    page: usize,
    permille: u32,
) -> Result<(u64, u64, alloc::vec::Vec<u32>), &'static str> {
    session.set_fuel(PDF_RENDER_FUEL)?;
    let hdr = session.call_i32_i32("pdf_render", page as i32, permille as i32)?;
    if hdr <= 0 {
        let code = session.call_i32("pdf_last_error").unwrap_or(0);
        return Err(match code {
            -2 => "no document open",
            -3 => "no such page",
            -4 => "too large to render at this zoom",
            _ => "the renderer refused the page",
        });
    }
    let h = session.get_u32s(hdr as usize, 4)?;
    let (w, ht, ptr, len) = (h[0] as u64, h[1] as u64, h[2] as usize, h[3] as usize);
    if w == 0 || ht == 0 {
        return Err("the renderer produced an empty page");
    }
    // 4 bytes per pixel, and the reported length must match the reported
    // geometry — a mismatch means the header is not describing this buffer.
    if len as u64 != w * ht * 4 {
        return Err("pixel buffer does not match the reported size");
    }
    let rgba = session.get_bytes(ptr, len)?;
    Ok((w, ht, crate::pdfview::rgba_to_rgb(&rgba, (w * ht) as usize)))
}

/// Repaint on tab switch (no re-render: the cache is still valid).
#[cfg(not(test))]
pub(super) fn repaint_pdf() {
    render_pdf();
}

/// Close the viewer and free the document, its wasm instance and its pixels.
#[cfg(not(feature = "server"))]
pub(super) fn close_pdf() {
    PDF.with(|s| *s = None);
}

/// `/pdf` — status, and the human-facing surface for the things the viewer's
/// keys cannot express.
///
/// * `/pdf` — what is open, which page, at what scale.
/// * `/pdf text [path]` — the deterministic wasm **text** digest in an editor
///   tab (the `pdf_text` tool). Without a path it digests the open document, so
///   "show me the text of this" needs no retyping of the filename.
/// * `/pdf page <n>` — jump to a page, which is impractical by key in a long
///   document.
#[cfg(all(not(feature = "server"), not(test)))]

/// `/pdf create <out.pdf> [--from <file>] [--title <t>] [--text <words…>]`
///
/// The source format is taken from the input file's extension — `.md`, `.html`
/// or plain text — because guessing from content is unreliable for short inputs
/// and the extension is what the user already told us.
pub(super) fn run_pdf_create(arg: &str) {
    use alloc::string::{String, ToString};
    let mut out: Option<String> = None;
    let mut from: Option<String> = None;
    let mut title = String::new();
    let mut inline_text = String::new();
    let mut format: Option<String> = None;

    let toks: alloc::vec::Vec<&str> = arg.split_whitespace().collect();
    let mut i = 1; // skip the subcommand
    while i < toks.len() {
        match toks[i] {
            "--from" | "-f" => {
                i += 1;
                from = toks.get(i).map(|s| super::resolve_path(s));
            }
            "--title" | "-t" => {
                i += 1;
                // The title is the rest of the line up to the next flag, so it
                // may contain spaces.
                let mut parts = alloc::vec::Vec::new();
                while i < toks.len() && !toks[i].starts_with("--") {
                    parts.push(toks[i]);
                    i += 1;
                }
                i -= 1;
                title = parts.join(" ");
            }
            "--as" => {
                i += 1;
                format = toks.get(i).map(|s| s.to_ascii_lowercase());
            }
            "--text" => {
                inline_text = toks[i + 1..].join(" ");
                i = toks.len();
            }
            t if out.is_none() && !t.starts_with('-') => out = Some(super::resolve_path(t)),
            t => serial_println!("pdf> ignoring unknown option {t}"),
        }
        i += 1;
    }

    let Some(out_path) = out else {
        serial_println!(
            "pdf> usage: /pdf create <out.pdf> [--from <src.md|src.html|src.txt>] [--text <words>] [--title T]"
        );
        return;
    };
    let (source, ext) = if !inline_text.is_empty() {
        (inline_text, "md".to_string())
    } else if let Some(src) = &from {
        let Some(bytes) = crate::synapse::fs::read(src) else {
            serial_println!("pdf> no such file: {src}");
            return;
        };
        let ext = src.rsplit('.').next().unwrap_or("").to_ascii_lowercase();
        (String::from_utf8_lossy(&bytes).into_owned(), ext)
    } else {
        serial_println!("pdf> nothing to write — give --from <file> or --text <words>");
        return;
    };
    let ext = format.unwrap_or(ext);
    if title.is_empty() {
        title = out_path
            .rsplit('/')
            .next()
            .unwrap_or("Document")
            .trim_end_matches(".pdf")
            .to_string();
    }

    let bytes = match ext.as_str() {
        "html" | "htm" => crate::pdf::from_html(&source, &title),
        "txt" | "text" => crate::pdf::from_text(&source, &title),
        _ => crate::pdf::from_markdown(&source, &title),
    };
    crate::synapse::fs::write(&out_path, &bytes);
    // Read back rather than trusting the write: a file that cannot be reopened
    // is worse than one never written, because the user will go on to /open it.
    if crate::synapse::fs::read(&out_path).is_none() {
        serial_println!("pdf> could not write {out_path}");
        return;
    }
    let pages = alloc::string::String::from_utf8_lossy(&bytes)
        .matches("/Type /Page ")
        .count();
    serial_println!(
        "pdf> wrote {out_path} — {} page(s), {} bytes. /open {out_path} to view it.",
        pages.max(1),
        bytes.len()
    );
}

#[cfg(all(not(feature = "server"), not(test)))]
pub(super) fn run_pdf(arg: &str) {
    let mut it = arg.split_whitespace();
    let sub = it.next().unwrap_or("");
    let rest = it.next().unwrap_or("");
    match sub {
        "" | "status" => {
            let s = PDF.with(|s| {
                s.as_ref().map(|t| {
                    // The render time belongs in the status line, not just the
                    // HUD: it is the answer to "why was that slow", and over a
                    // serial console the HUD is invisible — which is also what
                    // makes the figure testable outside the framebuffer.
                    alloc::format!(
                        "pdf> {} — page {}/{} at {}% ({} ms){}",
                        t.path,
                        t.view.page + 1,
                        t.view.pages,
                        t.permille / 10,
                        t.last_ms,
                        if t.rendered.is_some() { "" } else { " (page not rendered)" }
                    )
                })
            });
            match s {
                Some(line) => serial_println!("{line}"),
                None => serial_println!("pdf> nothing open — /open <file>.pdf   (/pdf text <file> for text only)"),
            }
        }
        "text" => {
            let path = if rest.is_empty() {
                PDF.with(|s| s.as_ref().map(|t| t.path.clone()))
            } else {
                Some(super::resolve_path(rest))
            };
            match path {
                Some(p) => serial_println!("pdf> {}", super::pdf_text(&p)),
                None => serial_println!("pdf> usage: /pdf text <file.pdf>   (or open one first)"),
            }
        }
        "page" => {
            let Ok(n) = rest.parse::<usize>() else {
                serial_println!("pdf> usage: /pdf page <n>");
                return;
            };
            let moved = PDF.with(|s| {
                s.as_mut().map(|t| {
                    let target = n.saturating_sub(1).min(t.view.pages.saturating_sub(1)) as i64;
                    let delta = target - t.view.page as i64;
                    t.view = crate::pdfview::step_page(t.view, delta);
                    // A page change invalidates the page size (pages can differ).
                    if let Some(sz) = page_size(&mut t.session, t.view.page) {
                        t.page_mpt = sz;
                    }
                    t.view.page + 1
                })
            });
            match moved {
                Some(p) => {
                    render_pdf();
                    serial_println!("pdf> page {p}");
                }
                None => serial_println!("pdf> nothing open"),
            }
        }
        "create" | "new" | "write" => run_pdf_create(arg),
        other => serial_println!(
            "pdf> unknown '{other}' — /pdf [status|text [file]|page <n>|create <out.pdf> [--from <src>] [--title T]]"
        ),
    }
}
#[cfg(not(all(not(feature = "server"), not(test))))]
pub(super) fn run_pdf(_arg: &str) {}

#[cfg(not(all(not(feature = "server"), not(test))))]
pub(super) fn run_pdf_create(_arg: &str) {}

/// A **typed** key while the pdf pane is focused: zoom, fit, page stepping.
/// Returns false when the key is not one of ours, so the caller passes it on.
///
/// Deliberately **not** arrows. `media_key` hands this raw typed bytes while
/// `media_nav` hands the *finals* of an escape sequence, and both are `u8` — so a
/// viewer that treated `A`..`D` as arrows here scrolled the page whenever a
/// capital letter was typed. It cost a real bug: with the tab focused,
/// `/cat /samples/README.md` arrived as `/cat /samles/REME.md`, because `p`
/// paged and `A`/`D` "scrolled". Arrows live in [`pdf_nav`], reached only from
/// the escape-sequence path, where an `A` really is an arrow.
#[cfg(all(not(feature = "server"), not(test)))]
pub(super) fn pdf_cmd(c: u8) -> bool {
    match c {
        b'+' | b'=' | b'-' | b'_' | b'f' | b'F' | b'0' | b'n' | b'N' | b'>' | b'p' | b'P' | b'<' | b'g' | b'G' => {
            apply_pdf_key(c)
        }
        _ => false,
    }
}
#[cfg(not(all(not(feature = "server"), not(test))))]
pub(super) fn pdf_cmd(_c: u8) -> bool {
    false
}

/// PgUp/PgDn while the pdf pane is focused: step a page. Returns false when the
/// focused tab is not a loaded document, so the caller falls back to the pane's
/// scrollback — the key's meaning everywhere else.
#[cfg(all(not(feature = "server"), not(test)))]
pub(super) fn pdf_page_key(up: bool) -> bool {
    if !crate::framebuffer::focus_is_action() {
        return false;
    }
    if crate::framebuffer::right_mode() != crate::framebuffer::RightMode::Surface(PDF_SURFACE) || !pdf_loaded() {
        return false;
    }
    apply_pdf_key(if up { b'p' } else { b'n' })
}
#[cfg(not(all(not(feature = "server"), not(test))))]
pub(super) fn pdf_page_key(_up: bool) -> bool {
    false
}

/// An **arrow / Home** key for the focused pdf pane: vertical scroll (which
/// crosses page boundaries) and horizontal pan. Only ever called with the final
/// byte of a CSI sequence.
#[cfg(all(not(feature = "server"), not(test)))]
pub(super) fn pdf_nav(fin: u8) -> bool {
    match fin {
        b'A' | b'B' | b'C' | b'D' => apply_pdf_key(fin),
        // Home = first page, End = last: a long document needs both.
        b'H' => apply_pdf_key(b'g'),
        b'F' => apply_pdf_key(b'G'),
        _ => false,
    }
}
#[cfg(not(all(not(feature = "server"), not(test))))]
pub(super) fn pdf_nav(_fin: u8) -> bool {
    false
}

/// The viewer's key actions, shared by [`pdf_cmd`] and [`pdf_nav`] — which byte
/// is allowed to arrive from where is the callers' business, not this function's.
#[cfg(all(not(feature = "server"), not(test)))]
fn apply_pdf_key(c: u8) -> bool {
    let hud_h = crate::framebuffer::pdf_hud_height();
    let Some((pane_w, pane_h)) = crate::framebuffer::surface_dims_px(PDF_SURFACE) else {
        return false;
    };
    let pane = (pane_w as u64, (pane_h as u64).saturating_sub(hud_h).max(1));
    let mut act = false;
    PDF.with(|slot| {
        let Some(t) = slot.as_mut() else { return };
        // Scroll/pan steps scale with the pane so they feel the same at any
        // resolution — a third of a screen per arrow, like a document reader.
        let vstep = (pane.1 / 3).max(16) as i64;
        let hstep = (pane.0 / 4).max(16) as i64;
        let img = t.rendered.as_ref().map(|(_, _, w, h, _)| (*w, *h)).unwrap_or(pane);
        let before = t.view;
        match c {
            b'+' | b'=' => t.view = crate::pdfview::zoom_by(t.view, crate::pdfview::ZOOM_STEP as i32),
            b'-' | b'_' => t.view = crate::pdfview::zoom_by(t.view, -(crate::pdfview::ZOOM_STEP as i32)),
            // Fit toggles: 'f' cycles page/width, '0' resets zoom *and* fit.
            b'f' | b'F' => {
                t.view.fit = match t.view.fit {
                    crate::pdfview::Fit::Page => crate::pdfview::Fit::Width,
                    crate::pdfview::Fit::Width => crate::pdfview::Fit::Page,
                };
                t.view.pan_x = 0;
                t.view.pan_y = 0;
            }
            b'0' => {
                t.view.zoom = 100;
                t.view.fit = crate::pdfview::Fit::Page;
                t.view.pan_x = 0;
                t.view.pan_y = 0;
            }
            // n/p and PgDn/PgUp (forwarded as '>'/'<' by the key router) step pages.
            b'n' | b'N' | b'>' => t.view = crate::pdfview::step_page(t.view, 1),
            b'p' | b'P' | b'<' => t.view = crate::pdfview::step_page(t.view, -1),
            // Home/End as first/last page — a long document is unusable without them.
            b'g' => t.view = crate::pdfview::step_page(t.view, -(t.view.pages as i64)),
            b'G' => t.view = crate::pdfview::step_page(t.view, t.view.pages as i64),
            // Arrows: vertical scroll crosses page boundaries, horizontal pans.
            b'A' => t.view = crate::pdfview::scroll(t.view, img, pane, -vstep),
            b'B' => t.view = crate::pdfview::scroll(t.view, img, pane, vstep),
            b'C' => t.view.pan_x += hstep,
            b'D' => t.view.pan_x -= hstep,
            _ => return,
        }
        act = true;
        // A page change invalidates the cached page size (pages can differ) —
        // re-read it before the render picks a scale from it.
        if t.view.page != before.page {
            if let Some(sz) = page_size(&mut t.session, t.view.page) {
                t.page_mpt = sz;
            }
        }
    });
    if act {
        render_pdf();
    }
    act
}
