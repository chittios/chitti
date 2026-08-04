//! browser
//!
//! The **browser tab** subsystem carved out of the former 16k-line
//! `shell/mod.rs` monolith: `BrowserSession`, the surface/selection state,
//! load/paint/present, key handling and the in-shell browser tool surface.
//! Moved verbatim; `use super::*` keeps the parent's statics visible, and
//! the parent re-imports this module's items with `use browser::*`.

use super::*;


// Mirror `framebuffer::BROWSER_SURFACE` (module is cfg(not(test))).
pub(super) const BROWSER_SURFACE: u32 = u32::MAX - 2;
pub(super) const BROWSER_BODY_MAX: usize = 1 << 20; // 1 MiB

/// Browser layout/paint viewport width — the action pane's actual pixel width
/// so the page is rendered 1:1 into the pane (no upscaling → crisp text).
/// Falls back to a sane default before the pane exists.
pub(super) fn browser_vw() -> i32 {
    #[cfg(not(test))]
    {
        crate::framebuffer::surface_dims_px(BROWSER_SURFACE)
            .map(|(w, _)| w as i32)
            .unwrap_or(960)
            .clamp(320, 4096)
    }
    #[cfg(test)]
    {
        640
    }
}

/// Browser viewport height — the action pane's pixel height minus the reserved
/// HUD strip, so layout/scroll/paint all agree and present at 1:1.
pub(super) fn browser_vh() -> i32 {
    #[cfg(not(test))]
    {
        let hud = crate::framebuffer::browser_hud_height() as i32;
        crate::framebuffer::surface_dims_px(BROWSER_SURFACE)
            .map(|(_, h)| (h as i32 - hud).max(200))
            .unwrap_or(700)
            .clamp(200, 4096)
    }
    #[cfg(test)]
    {
        400
    }
}

pub(super) struct BrowserSession {
    url: alloc::string::String,
    title: alloc::string::String,
    html: alloc::string::String,
    scroll_y: i32,
    history: alloc::vec::Vec<alloc::string::String>,
    content_height: i32,
    /// Focused form control index (layout.controls).
    focused: Option<usize>,
    /// Live control values (index → value), survives re-layout for typing.
    control_values: alloc::collections::BTreeMap<usize, alloc::string::String>,
    control_checked: alloc::collections::BTreeMap<usize, bool>,
    /// Fetched external `<script src>` bodies (absolute URL → source).
    script_bodies: alloc::collections::BTreeMap<alloc::string::String, alloc::string::String>,
    /// Fetched external stylesheet bodies (absolute URL → CSS, with one level
    /// of `@import` prepended) — repaints re-merge these without refetching.
    css_bodies: alloc::collections::BTreeMap<alloc::string::String, alloc::string::String>,
    /// Decoded CSS background images (absolute URL → (pixels, w, h)).
    bg_pixels:
        alloc::collections::BTreeMap<alloc::string::String, (alloc::vec::Vec<u32>, usize, usize)>,
    /// The script list actually booted into the page JS context (debugging).
    #[allow(dead_code)]
    resolved_scripts: alloc::vec::Vec<alloc::string::String>,
}

pub(super) static BROWSER: crate::mm::Locked<Option<BrowserSession>> = crate::mm::Locked::new(None);

/// Last painted layout (for hover hit-test without re-parse on every mouse move).
pub(super) static BROWSER_LAYOUT: crate::mm::Locked<Option<crate::browser::layout::Layout>> =
    crate::mm::Locked::new(None);

/// Loading progress 0..=100 for the browser chrome bar; 255 = hidden.
pub(super) static BROWSER_PROGRESS: core::sync::atomic::AtomicU8 =
    core::sync::atomic::AtomicU8::new(255);

/// True while `browser_load_method` is mid-navigation (progressive stages).
/// Hover / tab-repaint must not re-layout the *previous* session HTML in this
/// window — that flashed the old page whenever the mouse moved or the action
/// pane was re-composited.
pub(super) static BROWSER_LOADING: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

/// Last pixels we presented for the browser surface (logical RGB, no letterbox).
/// Used to re-blit without re-layout when the action pane is repainted mid-load
/// (tab switch, divider drag, status chrome) so the previous page can't return.
pub(super) struct BrowserPresentCache {
    pixels: alloc::vec::Vec<u32>,
    title: alloc::string::String,
    url: alloc::string::String,
    scroll_y: i32,
    content_h: i32,
}
pub(super) static BROWSER_PRESENT: crate::mm::Locked<Option<BrowserPresentCache>> =
    crate::mm::Locked::new(None);

/// Active drag-to-select in the browser surface (chat pane has its own `textsel`).
/// Press anchors; drag extends; release with a real range copies via OSC 52.
pub(super) struct BrowserTextSel {
    anchor: crate::browser::layout::TextPos,
    head: crate::browser::layout::TextPos,
    press_sx: i32,
    press_sy: i32,
    /// True once the pointer moved past a small threshold (else release = click).
    moved: bool,
}
pub(super) static BROWSER_SEL: crate::mm::Locked<Option<BrowserTextSel>> = crate::mm::Locked::new(None);
/// True while LMB is down after a press that began a browser selection.
pub(super) static BROWSER_SEL_DRAG: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

pub(super) fn browser_sel_range() -> Option<(crate::browser::layout::TextPos, crate::browser::layout::TextPos)> {
    BROWSER_SEL.with(|s| s.as_ref().map(|x| (x.anchor, x.head)))
}

pub(super) fn browser_sel_clear() {
    let had = BROWSER_SEL.with(|s| s.take()).is_some();
    BROWSER_SEL_DRAG.store(false, core::sync::atomic::Ordering::Relaxed);
    if had && browser_loaded() && !browser_is_loading() {
        let _ = browser_repaint();
    }
}

/// Anchor a selection at content point under surface `(sx, sy)`.
pub(super) fn browser_sel_begin(sx: i32, sy: i32) {
    let scroll = BROWSER.with(|s| s.as_ref().map(|b| b.scroll_y).unwrap_or(0));
    let content_y = sy + scroll;
    let pos = BROWSER_LAYOUT.with(|slot| {
        slot.as_ref()
            .and_then(|lay| crate::browser::layout::text_pos_at(lay, sx, content_y))
    });
    BROWSER_SEL.with(|s| {
        *s = pos.map(|p| BrowserTextSel {
            anchor: p,
            head: p,
            press_sx: sx,
            press_sy: sy,
            moved: false,
        });
    });
    BROWSER_SEL_DRAG.store(true, core::sync::atomic::Ordering::Relaxed);
    // Clear a prior highlight if any (repaint only when we had a range).
    let _ = browser_repaint();
}

/// Extend the selection head to surface `(sx, sy)`.
pub(super) fn browser_sel_drag(sx: i32, sy: i32) {
    let scroll = BROWSER.with(|s| s.as_ref().map(|b| b.scroll_y).unwrap_or(0));
    let content_y = sy + scroll;
    let pos = BROWSER_LAYOUT.with(|slot| {
        slot.as_ref()
            .and_then(|lay| crate::browser::layout::text_pos_at(lay, sx, content_y))
    });
    let Some(head) = pos else {
        return;
    };
    let changed = BROWSER_SEL.with(|s| {
        let Some(sel) = s.as_mut() else {
            return false;
        };
        let dx = (sx - sel.press_sx).abs();
        let dy = (sy - sel.press_sy).abs();
        if dx > 3 || dy > 3 {
            sel.moved = true;
        }
        if sel.head != head {
            sel.head = head;
            true
        } else {
            false
        }
    });
    if changed {
        let _ = browser_repaint();
    }
}

/// Finish browser selection on mouse release.
/// - Dragged range → `Some(text)` for clipboard (highlight kept).
/// - Plain click → clear highlight, return `None` so the caller fires `browser_click`.
pub(super) fn browser_sel_end() -> Option<alloc::string::String> {
    let (anchor, head, moved) = BROWSER_SEL.with(|s| {
        s.as_ref()
            .map(|x| (x.anchor, x.head, x.moved))
            .unwrap_or((
                crate::browser::layout::TextPos { run: 0, col: 0 },
                crate::browser::layout::TextPos { run: 0, col: 0 },
                false,
            ))
    });
    if !moved || anchor == head {
        BROWSER_SEL.with(|s| *s = None);
        let _ = browser_repaint();
        return None;
    }
    let text = BROWSER_LAYOUT.with(|slot| {
        slot.as_ref()
            .map(|lay| crate::browser::layout::selection_text(lay, anchor, head))
            .unwrap_or_default()
    });
    if text.is_empty() {
        BROWSER_SEL.with(|s| *s = None);
        let _ = browser_repaint();
        return None;
    }
    // Keep highlight visible until the next press (like chat).
    Some(text)
}

/// Host entry for browser tools (`ToolBinding::Browser`).
pub(crate) fn run_browser_tool(name: &str, args_json: &str) -> alloc::string::String {
    use crate::session::todo::json_str;
    match name {
        "browser_open" | "browser_navigate" => {
            let url = json_str(args_json, "url").unwrap_or_default();
            browser_load(&url, name == "browser_navigate" || name == "browser_open")
        }
        "browser_back" => browser_back(),
        "browser_scroll" => {
            let dy = json_str(args_json, "dy")
                .and_then(|s| s.parse::<i32>().ok())
                .or_else(|| {
                    json_str(args_json, "page")
                        .and_then(|s| s.parse::<i32>().ok())
                        .map(|p| p * (browser_vh() - 40))
                })
                .unwrap_or(browser_vh() / 2);
            browser_scroll(dy)
        }
        "browser_click" => {
            let x = json_str(args_json, "x")
                .and_then(|s| s.parse::<i32>().ok())
                .unwrap_or(0);
            let y = json_str(args_json, "y")
                .and_then(|s| s.parse::<i32>().ok())
                .unwrap_or(0);
            browser_click(x, y)
        }
        "browser_status" => browser_status(),
        "browser_links" => browser_links(),
        "browser_text" => {
            let max = json_str(args_json, "max")
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(4000);
            browser_text(max)
        }
        _ => alloc::format!("error: unknown browser tool '{name}'"),
    }
}

pub(super) fn browser_set_progress(pct: u8) {
    BROWSER_PROGRESS.store(pct.min(100), core::sync::atomic::Ordering::Relaxed);
    #[cfg(not(test))]
    {
        crate::framebuffer::set_cursor_shape(crate::framebuffer::CursorShape::Wait);
    }
}

pub(super) fn browser_clear_progress() {
    BROWSER_PROGRESS.store(255, core::sync::atomic::Ordering::Relaxed);
    BROWSER_LOADING.store(false, core::sync::atomic::Ordering::Relaxed);
    #[cfg(not(test))]
    {
        crate::framebuffer::set_cursor_shape(crate::framebuffer::CursorShape::Arrow);
    }
}

pub(super) fn browser_progress_opt() -> Option<u8> {
    let p = BROWSER_PROGRESS.load(core::sync::atomic::Ordering::Relaxed);
    if p > 100 {
        None
    } else {
        Some(p)
    }
}

pub(super) fn browser_is_loading() -> bool {
    BROWSER_LOADING.load(core::sync::atomic::Ordering::Relaxed)
}

/// Start a progressive navigation: mark loading, drop hover, and forget the
/// previous page's layout so mouse-move hit-tests can't flash it back.
pub(super) fn browser_begin_load() {
    BROWSER_LOADING.store(true, core::sync::atomic::Ordering::Relaxed);
    let _ = crate::browser::set_hover_link(None);
    BROWSER_LAYOUT.with(|s| *s = None);
    // Drop the previous page's present cache so a mid-load `repaint_active_tab`
    // can't re-blit old pixels. The next `browser_present` will refill it.
    BROWSER_PRESENT.with(|s| *s = None);
    // Selection indices are layout-relative — clear on navigation.
    BROWSER_SEL.with(|s| *s = None);
    BROWSER_SEL_DRAG.store(false, core::sync::atomic::Ordering::Relaxed);
}

pub(super) fn browser_load(url: &str, push_hist: bool) -> alloc::string::String {
    // Lazily load the CJK fallback font (off the boot path). Safe to scan disks
    // now — the block probe is idempotent and the FAT directory walk is bounded.
    ensure_disk_fallback_fonts();
    browser_load_method(url, "GET", &[], push_hist)
}

/// Mirror page-JS console lines to the serial console (capped at 50).
pub(super) fn browser_mirror_js_lines(lines: &[alloc::string::String]) {
    for (i, line) in lines.iter().enumerate() {
        if i >= 50 {
            crate::serial_println!("browser> js: … {} more lines", lines.len() - 50);
            break;
        }
        crate::serial_println!("browser> js: {}", line);
    }
}

/// Drain the live page's console log (console.log + uncaught errors) and
/// mirror it to serial — the web-devtools view of what page scripts did.
pub(super) fn browser_mirror_js_log() {
    let lines = crate::browser::js_just::page_with_dom(|d| core::mem::take(&mut d.log))
        .unwrap_or_default();
    browser_mirror_js_lines(&lines);
}

/// After delivering events into page JS: follow a handler-requested
/// navigation (`location.href = …`). Returns `Some(result)` when navigated.
pub(super) fn browser_dispatch_nav(base: &str) -> Option<alloc::string::String> {
    let nav = crate::browser::js_just::page_with_dom(|d| d.navigate.take()).flatten()?;
    let abs = crate::browser::url::resolve(base, &nav).unwrap_or(nav);
    if !crate::browser::url::is_http_url(&abs) {
        return None;
    }
    Some(browser_load(&abs, true))
}

/// Fetch + register `@font-face` web fonts named in `css` (URLs resolved
/// against `base_url`). WOFF is unwrapped to SFNT ([`crate::font_woff`]) and
/// WOFF2 is Brotli-decompressed + glyf/loca-reconstructed
/// ([`crate::font_woff2`]). Failures log, never abort a load.
pub(super) fn browser_load_fonts(css: &str, base_url: &str) {
    let faces = crate::browser::css::scan_font_faces(css);
    if faces.is_empty() {
        return;
    }
    let mut urls: alloc::vec::Vec<alloc::string::String> = alloc::vec::Vec::new();
    let mut wanted: alloc::vec::Vec<(alloc::string::String, alloc::string::String)> =
        alloc::vec::Vec::new();
    for f in &faces {
        if f.family.is_empty() || crate::font_ttf::family_loaded(&f.family) {
            continue;
        }
        if wanted.iter().any(|(fam, _)| fam == &f.family) {
            continue; // first src per family wins
        }
        let abs = crate::browser::url::resolve(base_url, &f.url).unwrap_or_else(|| f.url.clone());
        if !(abs.starts_with("http://") || abs.starts_with("https://")) {
            continue;
        }
        if !urls.contains(&abs) {
            urls.push(abs.clone());
        }
        wanted.push((f.family.clone(), abs));
    }
    if urls.is_empty() {
        return;
    }
    let fetched = crate::browser::worker::fetch_subresources_cooperative(
        &urls,
        crate::browser::loader::Destination::Font,
        1 << 20,
    );
    for (family, abs) in wanted {
        let Some(bytes) = fetched.get(&abs) else {
            crate::serial_println!("browser> font: '{}' not fetched ({})", family, abs);
            continue;
        };
        let res = if crate::font_woff::is_woff2(bytes) {
            // WOFF2: Brotli-decompress + reconstruct the transformed glyf/loca.
            crate::font_woff2::woff2_to_sfnt(bytes)
                .and_then(|sfnt| crate::font_ttf::load_family(&family, &sfnt))
        } else if crate::font_woff::is_woff(bytes) {
            crate::font_woff::woff_to_sfnt(bytes)
                .and_then(|sfnt| crate::font_ttf::load_family(&family, &sfnt))
        } else {
            crate::font_ttf::load_family(&family, bytes)
        };
        match res {
            Ok(()) => {
                crate::serial_println!("browser> font: loaded '{}' ({} B)", family, bytes.len())
            }
            Err(e) => crate::serial_println!("browser> font: '{}' failed: {}", family, e),
        }
    }
}

/// Fetch + decode CSS `background-image: url(…)` targets named in `css`,
/// keyed by absolute URL (resolved against `base_url`). Decodes through the
/// same in-kernel decoders as `fill_image_slot`, kept unscaled (the painter
/// tiles/scales per `background-size`).
pub(super) fn browser_fetch_bg_images(
    css: &str,
    base_url: &str,
) -> alloc::collections::BTreeMap<alloc::string::String, (alloc::vec::Vec<u32>, usize, usize)> {
    let mut out = alloc::collections::BTreeMap::new();
    let mut urls: alloc::vec::Vec<alloc::string::String> = alloc::vec::Vec::new();
    for u in crate::browser::css::scan_css_urls(css) {
        let abs = crate::browser::url::resolve(base_url, &u).unwrap_or(u);
        if (abs.starts_with("http://") || abs.starts_with("https://")) && !urls.contains(&abs) {
            urls.push(abs);
        }
    }
    if urls.is_empty() {
        return out;
    }
    let (loaded, assets) = crate::browser::worker::fetch_images_cooperative(&urls);
    for u in &urls {
        let body: Option<alloc::vec::Vec<u8>> = loaded
            .iter()
            .find(|r| &r.url == u)
            .map(|r| r.body.clone())
            .or_else(|| assets.get(u).map(|(_, b)| b.to_vec()));
        let Some(bytes) = body else { continue };
        // SVG-aware decode (iana.org's icons are SVG); 0 hints → intrinsic size.
        let Some(img) = crate::browser::decode_image_or_svg(&bytes, 0, 0) else {
            crate::serial_println!("browser> bg: decode failed {}", u);
            continue;
        };
        if img.w.saturating_mul(img.h) > 4_000_000 {
            crate::serial_println!("browser> bg: too large ({}x{}) {}", img.w, img.h, u);
            continue;
        }
        out.insert(u.clone(), (img.pixels, img.w, img.h));
    }
    out
}

/// Build the page-boot script list from parsed `<script>` tags, in document
/// order: inline bodies verbatim, external `src` bodies from `script_bodies`
/// (a missing fetch logs `skipped … (not fetched)` and is dropped), module
/// scripts stripped of import/export syntax with their import graph inlined
/// (depth ≤ 3, cycle-safe, post-order so imports run before importers), and
/// `async` tags appended at the very end. Returns `(list, skipped_count)`.
pub(super) fn browser_script_list(
    doc: &crate::browser::html::Document,
    base_url: &str,
    script_bodies: &alloc::collections::BTreeMap<alloc::string::String, alloc::string::String>,
) -> (alloc::vec::Vec<alloc::string::String>, usize) {
    let mut main: alloc::vec::Vec<alloc::string::String> = alloc::vec::Vec::new();
    let mut tail: alloc::vec::Vec<alloc::string::String> = alloc::vec::Vec::new();
    let mut skipped = 0usize;
    let mut visited: alloc::collections::BTreeSet<alloc::string::String> =
        alloc::collections::BTreeSet::new();
    for t in &doc.script_tags {
        let (body, base) = if let Some(src) = &t.src {
            let abs =
                crate::browser::url::resolve(base_url, src).unwrap_or_else(|| src.clone());
            match script_bodies.get(&abs) {
                Some(b) => (b.clone(), abs),
                None => {
                    crate::serial_println!("browser> js: skipped {} (not fetched)", abs);
                    skipped += 1;
                    continue;
                }
            }
        } else {
            if t.body.trim().is_empty() {
                continue;
            }
            (t.body.clone(), alloc::string::String::from(base_url))
        };
        let out = if t.async_ { &mut tail } else { &mut main };
        if t.module {
            visited.insert(base.clone()); // self-import cycle guard
            let (stripped, imports) = crate::browser::css::strip_module_syntax(&body);
            browser_module_graph(stripped, imports, &base, 3, &mut visited, out);
        } else {
            out.push(body);
        }
    }
    main.extend(tail);
    (main, skipped)
}

/// Post-order DFS over an ES-module import graph: fetch each import
/// (depth-limited, cycle-safe), strip its module syntax, recurse into its
/// own imports, then push — so imports execute before their importer.
pub(super) fn browser_module_graph(
    stripped: alloc::string::String,
    imports: alloc::vec::Vec<alloc::string::String>,
    base_url: &str,
    depth: u32,
    visited: &mut alloc::collections::BTreeSet<alloc::string::String>,
    out: &mut alloc::vec::Vec<alloc::string::String>,
) {
    for spec in imports {
        let abs = crate::browser::url::resolve(base_url, &spec).unwrap_or(spec);
        if !(abs.starts_with("http://") || abs.starts_with("https://"))
            || visited.contains(&abs)
        {
            continue;
        }
        visited.insert(abs.clone());
        if depth == 0 {
            crate::serial_println!("browser> js: skipped {} (import depth cap)", abs);
            continue;
        }
        let fetched = crate::browser::worker::fetch_subresources_cooperative(
            core::slice::from_ref(&abs),
            crate::browser::loader::Destination::Script,
            512 * 1024,
        );
        let Some(bytes) = fetched.get(&abs) else {
            crate::serial_println!("browser> js: skipped {} (not fetched)", abs);
            continue;
        };
        let src = alloc::string::String::from_utf8_lossy(bytes).into_owned();
        let (sub_stripped, sub_imports) = crate::browser::css::strip_module_syntax(&src);
        browser_module_graph(sub_stripped, sub_imports, &abs, depth - 1, visited, out);
    }
    out.push(stripped);
}

/// Host tick hook for the `just` JS engine: pump the UI (clock/mouse/net) and
/// report a Ctrl+C so a heavy page's scripts can't freeze the cooperatively-
/// scheduled shell thread. Installed lazily on the first browse; the engine
/// calls it from its hot loops (see `just_engine::runner::host`).
pub(super) fn browser_js_tick() -> bool {
    upkeep();
    poll_interrupt()
}

/// The `host[:port]` of an `http(s)://` URL, or `None`.
pub(super) fn http_host_of(url: &str) -> Option<alloc::string::String> {
    let rest = url
        .strip_prefix("https://")
        .or_else(|| url.strip_prefix("http://"))?;
    let host = rest.split(['/', ':', '?', '#']).next().unwrap_or("");
    if host.is_empty() {
        None
    } else {
        Some(host.to_string())
    }
}

/// **DNS prefetch**: warm the resolver cache for the distinct *cross-origin*
/// hosts among `hrefs` (resolved against `base`), so their subresource fetches
/// skip the DNS round trip. The page's own host is already resolved (document
/// fetch), so only foreign hosts (CDNs) are prefetched — a small, bounded set.
pub(super) fn browser_prefetch_dns<'a>(hrefs: impl Iterator<Item = &'a str>, base: &str) {
    let same = http_host_of(base);
    let mut seen: alloc::collections::BTreeSet<alloc::string::String> =
        alloc::collections::BTreeSet::new();
    for href in hrefs {
        let abs = crate::browser::url::resolve(base, href).unwrap_or_else(|| href.to_string());
        if let Some(host) = http_host_of(&abs) {
            if Some(&host) == same.as_ref() {
                continue; // same origin — already resolved
            }
            if seen.insert(host.clone()) {
                crate::net::prefetch_dns(&host);
                upkeep();
                if poll_interrupt() {
                    break;
                }
            }
        }
    }
}

pub(super) fn browser_load_method(
    url: &str,
    method: &str,
    body: &[u8],
    push_hist: bool,
) -> alloc::string::String {
    let url = url.trim();
    if url.is_empty() {
        return alloc::string::String::from("error: missing url");
    }
    if !crate::browser::url::is_http_url(url) {
        return alloc::string::String::from("error: url must be http:// or https://");
    }
    // Keep the UI alive + Ctrl+C responsive while page scripts run.
    just_engine::runner::host::set_tick_hook(Some(browser_js_tick));
    crate::browser::worker::reset_global();
    // New navigation: clear sessionStorage + session cookies (Web Storage model).
    crate::browser::storage::STORAGE.with(|s| s.end_session());
    crate::browser::storage::load_active();
    crate::browser::events::EVENT_LOOP.with(|el| {
        el.tasks.clear();
        el.microtasks.clear();
        el.queue_load();
    });
    browser_begin_load();
    browser_set_progress(5);

    // Progressive render (stage 1/5): paint a loading screen immediately so the
    // browser tab opens right away instead of a blank pane while the document +
    // subresources fetch.
    {
        let loading =
            crate::browser::layout::layout_reader("Loading\u{2026}", url, browser_vw(), browser_vh());
        browser_paint_stage(&loading, "Loading\u{2026}", url, 8);
    }

    let doc_res = if method.eq_ignore_ascii_case("POST") {
        match crate::net::http::request(
            "POST",
            url,
            &[("Content-Type", "application/x-www-form-urlencoded")],
            body,
            60_000,
        ) {
            Ok(r) => crate::browser::loader::LoadedResource {
                url: url.to_string(),
                status: r.status,
                content_type: r
                    .get("content-type")
                    .map(|s| s.to_string())
                    .unwrap_or_default(),
                headers: r.headers,
                body: r.body,
                from_cache: false,
                redirects: 0,
                destination: crate::browser::loader::Destination::Document,
                cors_opaque: false,
            },
            Err(e) => {
                browser_clear_progress();
                return alloc::format!("error:{e}");
            }
        }
    } else {
        match crate::browser::loader::load_document(url, false) {
            Ok(r) => r,
            Err(e) => {
                browser_clear_progress();
                return alloc::format!("error:{e}");
            }
        }
    };
    browser_set_progress(25);
    if doc_res.status >= 400 {
        browser_clear_progress();
        return alloc::format!(
            "error: HTTP {} at {} (after {} redirect(s))",
            doc_res.status,
            doc_res.url,
            doc_res.redirects
        );
    }
    let final_url = doc_res.url.clone();
    let redirects = doc_res.redirects;
    let status = doc_res.status;
    let from_cache = doc_res.from_cache;
    let mut body_bytes = doc_res.body;
    if body_bytes.len() > BROWSER_BODY_MAX {
        body_bytes.truncate(BROWSER_BODY_MAX);
    }
    let body_html = alloc::string::String::from_utf8_lossy(&body_bytes).into_owned();

    // Progressive render (stage 2/5): DOM paint — lay out the raw HTML with
    // inline CSS only (no external CSS, no scripts) so page structure appears
    // fast, before the heavy script phase.
    {
        let empty: alloc::collections::BTreeMap<alloc::string::String, alloc::string::String> =
            alloc::collections::BTreeMap::new();
        let (dom_doc, dom_lay) = crate::browser::layout_static(
            &body_html,
            browser_vw(),
            browser_vh(),
            &final_url,
            &empty,
        );
        let t = if dom_doc.title.is_empty() {
            final_url.clone()
        } else {
            dom_doc.title.clone()
        };
        browser_paint_stage(&dom_lay, &t, &final_url, 20);
    }

    // --- Subresource discovery: parse once to enumerate external scripts /
    // stylesheets / fonts / background images (the layout parse comes later,
    // via the session path).
    let pre = crate::browser::html::parse(&body_html);
    let is_http = |u: &str| u.starts_with("http://") || u.starts_with("https://");

    // DNS prefetch: resolve the cross-origin hosts of this page's scripts +
    // stylesheets up front so their fetches (below) skip the DNS round trip.
    browser_prefetch_dns(
        pre.script_tags
            .iter()
            .filter_map(|t| t.src.as_deref())
            .chain(pre.styles_ordered.iter().filter_map(|s| match s {
                crate::browser::html::StyleSrc::External(href) => Some(href.as_str()),
                _ => None,
            })),
        &final_url,
    );

    // (a) External scripts.
    let mut script_urls: alloc::vec::Vec<alloc::string::String> = alloc::vec::Vec::new();
    for t in &pre.script_tags {
        if let Some(src) = &t.src {
            let abs =
                crate::browser::url::resolve(&final_url, src).unwrap_or_else(|| src.clone());
            if is_http(&abs) && !script_urls.contains(&abs) {
                script_urls.push(abs);
            }
        }
    }
    let mut script_bodies: alloc::collections::BTreeMap<
        alloc::string::String,
        alloc::string::String,
    > = alloc::collections::BTreeMap::new();
    for (u, body) in crate::browser::worker::fetch_subresources_cooperative(
        &script_urls,
        crate::browser::loader::Destination::Script,
        512 * 1024,
    ) {
        script_bodies.insert(u, alloc::string::String::from_utf8_lossy(&body).into_owned());
    }
    browser_set_progress(32);

    // (b) External stylesheets (document order), plus one level of @import —
    // imports resolve against the *sheet's* URL and prepend to its body.
    let mut css_urls: alloc::vec::Vec<alloc::string::String> = alloc::vec::Vec::new();
    for s in &pre.styles_ordered {
        if let crate::browser::html::StyleSrc::External(href) = s {
            let abs =
                crate::browser::url::resolve(&final_url, href).unwrap_or_else(|| href.clone());
            if is_http(&abs) && !css_urls.contains(&abs) {
                css_urls.push(abs);
            }
        }
    }
    let mut css_bodies: alloc::collections::BTreeMap<
        alloc::string::String,
        alloc::string::String,
    > = alloc::collections::BTreeMap::new();
    for (u, body) in crate::browser::worker::fetch_subresources_cooperative(
        &css_urls,
        crate::browser::loader::Destination::Style,
        256 * 1024,
    ) {
        css_bodies.insert(u, alloc::string::String::from_utf8_lossy(&body).into_owned());
    }
    let mut import_wants: alloc::vec::Vec<(
        alloc::string::String,
        alloc::vec::Vec<alloc::string::String>,
    )> = alloc::vec::Vec::new();
    let mut import_urls: alloc::vec::Vec<alloc::string::String> = alloc::vec::Vec::new();
    for (sheet_url, body) in &css_bodies {
        let imps: alloc::vec::Vec<alloc::string::String> =
            crate::browser::css::scan_imports(body)
                .iter()
                .filter_map(|i| crate::browser::url::resolve(sheet_url, i))
                .filter(|u| is_http(u))
                .collect();
        for u in &imps {
            if !import_urls.contains(u) && !css_bodies.contains_key(u) {
                import_urls.push(u.clone());
            }
        }
        if !imps.is_empty() {
            import_wants.push((sheet_url.clone(), imps));
        }
    }
    if !import_urls.is_empty() {
        let fetched_imports = crate::browser::worker::fetch_subresources_cooperative(
            &import_urls,
            crate::browser::loader::Destination::Style,
            256 * 1024,
        );
        for (sheet_url, imps) in import_wants {
            let mut prefix = alloc::string::String::new();
            for u in &imps {
                if let Some(b) = fetched_imports.get(u) {
                    prefix.push_str(&alloc::string::String::from_utf8_lossy(b));
                    prefix.push('\n');
                }
            }
            if !prefix.is_empty() {
                if let Some(body) = css_bodies.get_mut(&sheet_url) {
                    prefix.push_str(body);
                    *body = prefix;
                }
            }
        }
    }
    browser_set_progress(40);

    // (c) Web fonts + (d) CSS background images, from inline + external CSS.
    let mut all_css = pre.stylesheets.clone();
    for body in css_bodies.values() {
        all_css.push('\n');
        all_css.push_str(body);
    }
    browser_load_fonts(&all_css, &final_url);
    let bg_pixels = browser_fetch_bg_images(&all_css, &final_url);
    browser_set_progress(50);

    // Progressive render (stage 3/5): CSS paint — re-lay out with the fetched
    // external stylesheets applied (still no scripts), so the page is styled
    // before the (potentially heavy) script phase runs.
    {
        let (css_doc, css_lay) = crate::browser::layout_static(
            &body_html,
            browser_vw(),
            browser_vh(),
            &final_url,
            &css_bodies,
        );
        let t = if css_doc.title.is_empty() {
            final_url.clone()
        } else {
            css_doc.title.clone()
        };
        browser_paint_stage(&css_lay, &t, &final_url, 52);
    }

    // --- Boot the persistent page JS context (scripts run ONCE per
    // navigation; repaints re-read the DOM instead of re-running them).
    let (exec_list, _skipped) = browser_script_list(&pre, &final_url, &script_bodies);
    let _parsed = crate::browser::js_just::page_boot(
        &pre,
        &final_url,
        browser_vw(),
        browser_vh(),
        &exec_list,
    );
    browser_mirror_js_log();
    browser_set_progress(55);

    // Script-requested navigation (location.href = …).
    if let Some(nav) =
        crate::browser::js_just::page_with_dom(|d| d.navigate.take()).flatten()
    {
        if crate::browser::url::is_http_url(&nav) && nav != final_url {
            browser_clear_progress();
            return browser_load(&nav, push_hist);
        }
        if let Some(abs) = crate::browser::url::resolve(&final_url, &nav) {
            if abs != final_url {
                browser_clear_progress();
                return browser_load(&abs, push_hist);
            }
        }
    }

    // --- Layout via the session path: live page DOM + merged CSS + bg pixels.
    let (doc, mut lay, js_lines) = {
        let assets = crate::browser::SessionAssets {
            css_external: &css_bodies,
            bg_pixels: &bg_pixels,
        };
        crate::browser::layout_session(&body_html, browser_vw(), browser_vh(), &final_url, &assets)
    };
    browser_mirror_js_lines(&js_lines);

    // Progressive render (stage 4/5): scripts paint — the live page DOM after
    // scripts ran, before images fill in.
    {
        let t = if doc.title.is_empty() {
            final_url.clone()
        } else {
            doc.title.clone()
        };
        browser_paint_stage(&lay, &t, &final_url, 90);
    }

    // Subresource images via cooperative worker pool.
    let mut img_urls = alloc::vec::Vec::new();
    for im in lay.images.iter() {
        if im.src.is_empty() {
            continue;
        }
        let abs =
            crate::browser::url::resolve(&final_url, &im.src).unwrap_or_else(|| im.src.clone());
        if abs.starts_with("http://") || abs.starts_with("https://") {
            img_urls.push(abs);
        }
    }
    let n_total = img_urls.len().max(1);
    let (loaded_imgs, page_assets) =
        crate::browser::worker::fetch_images_cooperative(&img_urls);
    browser_set_progress(40 + (50 * loaded_imgs.len() / n_total) as u8);
    let n_imgs = loaded_imgs.len();
    let mut by_url: alloc::collections::BTreeMap<
        alloc::string::String,
        alloc::vec::Vec<u8>,
    > = alloc::collections::BTreeMap::new();
    for res in &loaded_imgs {
        by_url.insert(res.url.clone(), res.body.clone());
    }
    for u in page_assets.urls() {
        if let Some((_, body)) = page_assets.get(&u) {
            by_url.entry(u).or_insert_with(|| body.to_vec());
        }
    }
    for im in lay.images.iter_mut() {
        if im.src.is_empty() {
            continue;
        }
        let abs =
            crate::browser::url::resolve(&final_url, &im.src).unwrap_or_else(|| im.src.clone());
        if let Some(body) = by_url.get(&abs).or_else(|| by_url.get(&im.src)) {
            crate::browser::fill_image_slot(im, body);
        }
    }

    // Nested iframes / <video> first frames / canvas already has pixels.
    let n_frames = lay.frames.len();
    for fr in lay.frames.iter_mut() {
        crate::shell::upkeep();
        if crate::shell::poll_interrupt() {
            break;
        }
        use crate::browser::layout::EmbedKind;
        match fr.kind {
            EmbedKind::Canvas => {
                // pixels already allocated at layout; JS may redraw later
                continue;
            }
            EmbedKind::Iframe | EmbedKind::Other => {
                if !fr.srcdoc.is_empty() {
                    crate::browser::fill_frame_slot(fr, &fr.srcdoc.clone(), &final_url);
                    continue;
                }
                if fr.src.is_empty() {
                    continue;
                }
                let abs = crate::browser::url::resolve(&final_url, &fr.src)
                    .unwrap_or_else(|| fr.src.clone());
                if !(abs.starts_with("http://") || abs.starts_with("https://")) {
                    continue;
                }
                let req =
                    crate::browser::loader::LoadRequest::iframe(&abs).with_source(&final_url);
                match crate::browser::loader::load(&req) {
                    Ok(res) if res.status < 400 => {
                        let nested =
                            alloc::string::String::from_utf8_lossy(&res.body).into_owned();
                        crate::browser::fill_frame_slot(fr, &nested, &res.url);
                    }
                    Ok(res) => {
                        crate::ktrace::log_fmt(format_args!(
                            "browser:iframe HTTP {} {}",
                            res.status, abs
                        ));
                    }
                    Err(e) => {
                        crate::ktrace::log_fmt(format_args!("browser:iframe error {abs}: {e}"));
                    }
                }
            }
            EmbedKind::Video => {
                if fr.src.is_empty() {
                    continue;
                }
                let abs = crate::browser::url::resolve(&final_url, &fr.src)
                    .unwrap_or_else(|| fr.src.clone());
                // http(s) via loader, or store/mount path via existing readers
                let bytes = if abs.starts_with("http://") || abs.starts_with("https://") {
                    let req = crate::browser::loader::LoadRequest::get(&abs)
                        .with_source(&final_url)
                        .with_timeout(60_000);
                    match crate::browser::loader::load(&req) {
                        Ok(res) if res.status < 400 => res.body,
                        _ => continue,
                    }
                } else {
                    match read_mounted(&abs).or_else(|| crate::synapse::fs::read(&abs)) {
                        Some(b) => b,
                        None => continue,
                    }
                };
                crate::browser::fill_video_slot(fr, bytes);
            }
            EmbedKind::Audio => {
                // HUD-only for now (no waveform in-page).
            }
        }
    }
    let _ = n_frames;

    if let Some(last) = lay.images.last() {
        lay.content_height = lay.content_height.max(last.y + last.h + 16);
    }
    for c in &lay.controls {
        lay.content_height = lay.content_height.max(c.y + c.h + 16);
    }
    for f in &lay.frames {
        lay.content_height = lay.content_height.max(f.y + f.h + 16);
    }

    let mut scroll0 = 0i32;
    if let Some(sy) = crate::browser::js_just::page_with_dom(|d| d.scroll_to.take()).flatten() {
        scroll0 = sy.clamp(0, (lay.content_height - browser_vh()).max(0));
    }

    let mut control_values = alloc::collections::BTreeMap::new();
    let mut control_checked = alloc::collections::BTreeMap::new();
    for c in &lay.controls {
        control_values.insert(c.index, c.value.clone());
        control_checked.insert(c.index, c.checked);
    }

    browser_set_progress(95);
    let (pixels, content_h) =
        crate::browser::paint_layout_chrome(&lay, browser_vh(), scroll0, Some(100));
    let title = doc.title;
    BROWSER.with(|slot| {
        let mut hist = slot
            .as_ref()
            .map(|s| s.history.clone())
            .unwrap_or_default();
        if push_hist {
            if let Some(prev) = slot.as_ref().map(|s| s.url.clone()) {
                if !prev.is_empty() && prev != final_url {
                    hist.push(prev);
                    if hist.len() > 32 {
                        hist.remove(0);
                    }
                }
            }
        }
        *slot = Some(BrowserSession {
            url: final_url.clone(),
            title: title.clone(),
            html: body_html,
            scroll_y: scroll0,
            history: hist,
            content_height: content_h,
            focused: None,
            control_values,
            control_checked,
            script_bodies,
            css_bodies,
            bg_pixels,
            resolved_scripts: exec_list,
        });
    });
    browser_clear_progress();
    crate::browser::events::EVENT_LOOP.with(|el| {
        el.drain(32);
    });
    crate::browser::storage::persist_active();
    BROWSER_LAYOUT.with(|s| *s = Some(lay.clone()));
    browser_present(&pixels, &title, &final_url, scroll0, content_h);
    let chk = crate::browser::paint::checksum(&pixels);
    let (cache_n, cache_b, hits, misses) = crate::browser::loader::cache_stats();
    let (dns_n, dns_hits, dns_miss) = crate::net::dns_cache_stats();
    alloc::format!(
        "ok:title={} url={} redirects={redirects} status={status} cache={} imgs={n_imgs} forms={} iframes={} mem={cache_n}/{cache_b}b hits={hits} misses={misses} dns={dns_n}/{dns_hits}h/{dns_miss}m checksum={chk:016x} size={}x{}",
        title,
        final_url,
        if from_cache { "hit" } else { "miss" },
        lay.controls.len(),
        lay.frames.len(),
        browser_vw(),
        browser_vh()
    )
}

/// Rebuild layout from session HTML, re-apply control state, paint.
pub(super) fn browser_layout_session() -> Option<(
    crate::browser::layout::Layout,
    alloc::string::String,
    alloc::string::String,
    i32,
    Option<usize>,
)> {
    let (html, scroll, url, title, focused, values, checked, css_bodies, bg_pixels) = BROWSER
        .with(|s| {
            s.as_ref().map(|b| {
                (
                    b.html.clone(),
                    b.scroll_y,
                    b.url.clone(),
                    b.title.clone(),
                    b.focused,
                    b.control_values.clone(),
                    b.control_checked.clone(),
                    b.css_bodies.clone(),
                    b.bg_pixels.clone(),
                )
            })
        })?;
    // Session path: re-layout from the LIVE page DOM (no script re-run) with
    // the stored external CSS + background pixels.
    let (doc, mut lay, js_lines) = {
        let assets = crate::browser::SessionAssets {
            css_external: &css_bodies,
            bg_pixels: &bg_pixels,
        };
        crate::browser::layout_session(&html, browser_vw(), browser_vh(), &url, &assets)
    };
    // Handlers may log between repaints — mirror fresh lines to serial.
    browser_mirror_js_lines(&js_lines);
    // The live DOM may have retitled the page (document.title in a handler).
    let title = if doc.title.is_empty() { title } else { doc.title.clone() };
    BROWSER.with(|s| {
        if let Some(b) = s.as_mut() {
            b.title = title.clone();
        }
    });
    for c in &mut lay.controls {
        if let Some(v) = values.get(&c.index) {
            c.value = v.clone();
        }
        if let Some(&k) = checked.get(&c.index) {
            c.checked = k;
        }
        c.focused = focused == Some(c.index);
    }
    if let Some(last) = lay.controls.last() {
        lay.content_height = lay.content_height.max(last.y + last.h + 16);
    }
    // Re-layout rebuilds image/iframe boxes WITHOUT pixels (subresources are
    // only fetched in `browser_load`) — carry the previously decoded pixels
    // over by `src`, otherwise every click/scroll blanked the page's images.
    BROWSER_LAYOUT.with(|prev| {
        if let Some(p) = prev.as_ref() {
            for im in lay.images.iter_mut() {
                if im.pixels.is_none() {
                    if let Some(pim) = p
                        .images
                        .iter()
                        .find(|pi| pi.pixels.is_some() && pi.src == im.src)
                    {
                        im.pixels = pim.pixels.clone();
                        im.w = pim.w;
                        im.h = pim.h;
                        im.src_w = pim.src_w;
                        im.src_h = pim.src_h;
                    }
                }
            }
            for fr in lay.frames.iter_mut() {
                if fr.pixels.is_none() {
                    if let Some(pfr) = p.frames.iter().find(|pf| {
                        pf.pixels.is_some() && pf.src == fr.src && pf.srcdoc == fr.srcdoc
                    }) {
                        fr.pixels = pfr.pixels.clone();
                        fr.src_w = pfr.src_w;
                        fr.src_h = pfr.src_h;
                    }
                }
            }
        }
    });
    Some((lay, url, title, scroll, focused))
}

/// Progressive-render stage paint: render `lay` and blit it to the browser
/// Surface tab immediately, so the page appears in stages (loading → DOM → CSS
/// → scripts → images) instead of a blank pane until the whole pipeline
/// finishes. Pumps `upkeep()` so the clock/mouse stay live between stages.
#[cfg(not(test))]
pub(super) fn browser_paint_stage(
    lay: &crate::browser::layout::Layout,
    title: &str,
    url: &str,
    progress: u8,
) {
    let (pixels, content_h) =
        crate::browser::paint_layout_chrome(lay, browser_vh(), 0, Some(progress));
    BROWSER.with(|s| {
        if let Some(b) = s.as_mut() {
            b.content_height = content_h;
        }
    });
    // Keep hit-testing in sync with the stage just painted so hover never
    // falls back to a stale previous-page layout mid-navigation.
    BROWSER_LAYOUT.with(|s| *s = Some(lay.clone()));
    browser_present(&pixels, title, url, 0, content_h);
    crate::shell::upkeep();
}

#[cfg(test)]
pub(super) fn browser_paint_stage(_lay: &crate::browser::layout::Layout, _t: &str, _u: &str, _p: u8) {}

pub(super) fn browser_present(pixels: &[u32], title: &str, url: &str, scroll_y: i32, content_h: i32) {
    // Always cache what we last put on the surface so mid-load / tab-repaint
    // can re-blit without re-laying out the previous page's HTML.
    BROWSER_PRESENT.with(|s| {
        *s = Some(BrowserPresentCache {
            pixels: pixels.to_vec(),
            title: title.into(),
            url: url.into(),
            scroll_y,
            content_h,
        });
    });
    #[cfg(not(test))]
    {
        // present_surface_reserve already opens/focuses the Surface tab and blits.
        // Do **not** call set_right afterward: that runs repaint_action() and was
        // clearing the just-drawn page (blank black pane).
        // Reserve a video-style HUD strip for title + scroll scrubber + shortcuts.
        let hud = crate::framebuffer::browser_hud_height().max(1);
        crate::framebuffer::present_surface_reserve(
            BROWSER_SURFACE,
            browser_vw() as usize,
            browser_vh() as usize,
            pixels,
            hud,
        );
        let focused = BROWSER.with(|s| s.as_ref().and_then(|b| b.focused).is_some());
        crate::framebuffer::draw_browser_status(
            title,
            url,
            scroll_y,
            content_h,
            browser_vh(),
            focused,
        );
        crate::serial_println!(
            "browser> {} — {}  scroll {}/{}  runs_px={}",
            title,
            url,
            scroll_y,
            (content_h - browser_vh()).max(0),
            pixels.iter().filter(|&&p| p != 0 && p != 0xf5f0e8).count()
        );
    }
    #[cfg(test)]
    {
        let _ = (pixels, title, url, scroll_y, content_h);
    }
}

/// Re-blit the last presented frame without re-layout. Used while loading and
/// by hover when a full layout_session would flash the wrong page.
pub(super) fn browser_represent_cached() -> bool {
    BROWSER_PRESENT.with(|s| {
        if let Some(c) = s.as_ref() {
            let pixels = c.pixels.clone();
            let title = c.title.clone();
            let url = c.url.clone();
            let scroll_y = c.scroll_y;
            let content_h = c.content_h;
            // Don't re-enter the cache writer with the same data via a nested
            // present path that would clone twice — call the FB blit directly.
            #[cfg(not(test))]
            {
                let hud = crate::framebuffer::browser_hud_height().max(1);
                crate::framebuffer::present_surface_reserve(
                    BROWSER_SURFACE,
                    browser_vw() as usize,
                    browser_vh() as usize,
                    &pixels,
                    hud,
                );
                let focused = BROWSER.with(|b| b.as_ref().and_then(|x| x.focused).is_some());
                crate::framebuffer::draw_browser_status(
                    &title,
                    &url,
                    scroll_y,
                    content_h,
                    browser_vh(),
                    focused,
                );
            }
            let _ = (pixels, title, url, scroll_y, content_h);
            true
        } else {
            false
        }
    })
}

pub(super) fn browser_repaint() -> alloc::string::String {
    // Mid-navigation: never re-layout the previous session HTML. Re-blit the
    // last progressive stage (or no-op if nothing has painted yet).
    if browser_is_loading() {
        if browser_represent_cached() {
            return alloc::string::String::from("ok:loading");
        }
        return alloc::string::String::from("ok:loading");
    }
    let Some((lay, url, title, scroll, _focused)) = browser_layout_session() else {
        return alloc::string::String::from("error: no page open (browser_open first)");
    };
    let progress = browser_progress_opt();
    let sel = browser_sel_range();
    let (pixels, content_h) =
        crate::browser::paint_layout_chrome_sel(&lay, browser_vh(), scroll, progress, sel);
    BROWSER.with(|s| {
        if let Some(b) = s.as_mut() {
            b.content_height = content_h;
        }
    });
    BROWSER_LAYOUT.with(|s| *s = Some(lay));
    browser_present(&pixels, &title, &url, scroll, content_h);
    alloc::format!("ok:scroll={scroll} title={title}")
}

pub(super) fn browser_scroll(dy: i32) -> alloc::string::String {
    if browser_is_loading() {
        return alloc::string::String::from("ok:loading");
    }
    let max = BROWSER.with(|s| {
        s.as_ref()
            .map(|b| (b.content_height - browser_vh()).max(0))
            .unwrap_or(0)
    });
    BROWSER.with(|s| {
        if let Some(b) = s.as_mut() {
            b.scroll_y = (b.scroll_y + dy).clamp(0, max);
        }
    });
    browser_repaint()
}

pub(super) fn browser_back() -> alloc::string::String {
    let prev = BROWSER.with(|s| s.as_mut().and_then(|b| b.history.pop()));
    match prev {
        Some(u) => browser_load(&u, false),
        None => alloc::string::String::from("error: no history"),
    }
}

pub(super) fn browser_click(x: i32, y: i32) -> alloc::string::String {
    let Some((lay, base, _title, scroll, _foc)) = browser_layout_session() else {
        return alloc::string::String::from("error: no page open");
    };
    let content_y = y + scroll;
    match crate::browser::layout::hit_test_ex(&lay, x, content_y) {
        crate::browser::layout::Hit::Link(href) => {
            // A covering interactive element gets the click FIRST — its
            // handler may preventDefault() and suppress the navigation.
            if crate::browser::js_just::page_active() {
                let covering = lay
                    .elem_boxes
                    .iter()
                    .rev()
                    .find(|e| {
                        x >= e.x && x < e.x + e.w && content_y >= e.y && content_y < e.y + e.h
                    })
                    .map(|e| e.elem_idx);
                if let Some(ei) = covering {
                    let prevented = crate::browser::js_just::page_dispatch(&[
                        crate::browser::js_just::PageEvent {
                            target: ei,
                            type_: alloc::string::String::from("click"),
                            x,
                            y: content_y,
                        },
                    ]);
                    crate::serial_println!("browser> dispatched click → elem {}", ei);
                    if let Some(out) = browser_dispatch_nav(&base) {
                        return out;
                    }
                    if prevented.first().copied().unwrap_or(false) {
                        browser_repaint();
                        return alloc::string::String::from("ok:click handled (default prevented)");
                    }
                }
            }
            let url = crate::browser::url::resolve(&base, &href).unwrap_or(href);
            browser_load(&url, true)
        }
        crate::browser::layout::Hit::Elem(ei) => {
            // JS-interactive element: deliver the click into page JS, then
            // follow any handler navigation, else repaint the mutated DOM.
            let _prevented = crate::browser::js_just::page_dispatch(&[
                crate::browser::js_just::PageEvent {
                    target: ei,
                    type_: alloc::string::String::from("click"),
                    x,
                    y: content_y,
                },
            ]);
            crate::serial_println!("browser> dispatched click → elem {}", ei);
            if let Some(out) = browser_dispatch_nav(&base) {
                return out;
            }
            browser_repaint();
            alloc::format!("ok:clicked elem {}", ei)
        }
        crate::browser::layout::Hit::Control(idx) => {
            if let Some(c) = lay.controls.get(idx).cloned() {
                use crate::browser::layout::ControlKind;
                // Native focus / check handling first.
                match c.kind {
                    ControlKind::Hidden => {
                        return alloc::string::String::from("ok:hidden");
                    }
                    ControlKind::Submit => {}
                    ControlKind::Checkbox => {
                        BROWSER.with(|s| {
                            if let Some(b) = s.as_mut() {
                                b.focused = Some(idx);
                                let cur = b.control_checked.get(&idx).copied().unwrap_or(false);
                                b.control_checked.insert(idx, !cur);
                            }
                        });
                    }
                    _ => {
                        BROWSER.with(|s| {
                            if let Some(b) = s.as_mut() {
                                b.focused = Some(idx);
                            }
                        });
                    }
                }
                // Then deliver the click into page JS (bubbles to the form).
                let mut prevented = false;
                if crate::browser::js_just::page_active() {
                    if let Some(ei) = c.elem_idx {
                        prevented = crate::browser::js_just::page_dispatch(&[
                            crate::browser::js_just::PageEvent {
                                target: ei,
                                type_: alloc::string::String::from("click"),
                                x,
                                y: content_y,
                            },
                        ])
                        .first()
                        .copied()
                        .unwrap_or(false);
                        crate::serial_println!("browser> dispatched click → elem {}", ei);
                        if let Some(out) = browser_dispatch_nav(&base) {
                            return out;
                        }
                    }
                }
                match c.kind {
                    ControlKind::Submit if !prevented => browser_submit_control(&lay, &c),
                    ControlKind::Submit => {
                        browser_repaint();
                        alloc::string::String::from("ok:submit (default prevented)")
                    }
                    ControlKind::Button => {
                        browser_repaint();
                        alloc::string::String::from("ok:button")
                    }
                    ControlKind::Checkbox => {
                        browser_repaint();
                        alloc::string::String::from("ok:checkbox toggled")
                    }
                    ControlKind::Text | ControlKind::Password | ControlKind::TextArea => {
                        browser_repaint();
                        alloc::format!("ok:focus input {}", c.name)
                    }
                    ControlKind::Hidden => alloc::string::String::from("ok:hidden"),
                }
            } else {
                alloc::string::String::from("ok:no control")
            }
        }
        crate::browser::layout::Hit::Embed(idx) => {
            if let Some(fr) = lay.frames.get(idx).cloned() {
                use crate::browser::layout::EmbedKind;
                match fr.kind {
                    EmbedKind::Video => {
                        if fr.src.is_empty() {
                            return alloc::string::String::from("ok:video (no src)");
                        }
                        let abs = crate::browser::url::resolve(&base, &fr.src)
                            .unwrap_or_else(|| fr.src.clone());
                        browser_play_video_url(&abs, &base)
                    }
                    EmbedKind::Iframe if !fr.src.is_empty() => {
                        let abs = crate::browser::url::resolve(&base, &fr.src)
                            .unwrap_or_else(|| fr.src.clone());
                        browser_load(&abs, true)
                    }
                    EmbedKind::Canvas => {
                        alloc::string::String::from("ok:canvas")
                    }
                    EmbedKind::Audio => {
                        alloc::string::String::from("ok:audio (click play not wired)")
                    }
                    _ => alloc::string::String::from("ok:embed"),
                }
            } else {
                alloc::string::String::from("ok:no embed")
            }
        }
        crate::browser::layout::Hit::Page => {
            BROWSER.with(|s| {
                if let Some(b) = s.as_mut() {
                    b.focused = None;
                }
            });
            browser_repaint();
            alloc::string::String::from("ok:no link at point")
        }
    }
}

/// Fetch (or open local path) video and start the full video player tab.
#[cfg(not(feature = "server"))]
pub(super) fn browser_play_video_url(abs: &str, page_url: &str) -> alloc::string::String {
    let bytes = if abs.starts_with("http://") || abs.starts_with("https://") {
        let req = crate::browser::loader::LoadRequest::get(abs)
            .with_source(page_url)
            .with_timeout(120_000);
        match crate::browser::loader::load(&req) {
            Ok(res) if res.status < 400 => res.body,
            Ok(res) => {
                return alloc::format!("error: video HTTP {}", res.status);
            }
            Err(e) => {
                return alloc::format!("error: video load: {e}");
            }
        }
    } else {
        // Guest path / store
        match read_mounted(abs).or_else(|| crate::synapse::fs::read(abs)) {
            Some(b) => b,
            None => {
                return alloc::format!("error: video not found: {abs}");
            }
        }
    };
    play_video_bytes(abs, bytes);
    alloc::format!("ok:playing video {abs}")
}

#[cfg(feature = "server")]
pub(super) fn browser_play_video_url(_abs: &str, _page_url: &str) -> alloc::string::String {
    alloc::string::String::from("error: video player unavailable in server build")
}

pub(super) fn browser_submit_control(
    lay: &crate::browser::layout::Layout,
    c: &crate::browser::layout::FormControl,
) -> alloc::string::String {
    let form_id = c.form_id;
    // Merge live values from session into a temporary layout clone.
    let mut lay = lay.clone();
    BROWSER.with(|s| {
        if let Some(b) = s.as_ref() {
            for ctl in &mut lay.controls {
                if let Some(v) = b.control_values.get(&ctl.index) {
                    ctl.value = v.clone();
                }
                if let Some(&k) = b.control_checked.get(&ctl.index) {
                    ctl.checked = k;
                }
            }
        }
    });
    let fields = crate::browser::layout::form_fields(&lay, form_id);
    // Include submitter name/value if named.
    let mut fields = fields;
    if !c.name.is_empty() {
        fields.push(crate::browser::form::FormField {
            name: c.name.clone(),
            value: if c.value.is_empty() {
                alloc::string::String::from("Submit")
            } else {
                c.value.clone()
            },
        });
    }
    let base = BROWSER.with(|s| s.as_ref().map(|b| b.url.clone()).unwrap_or_default());
    let sub = crate::browser::form::build_submit(&base, &c.form_action, &c.form_method, &fields);
    if sub.method == "POST" {
        browser_load_method(&sub.url, "POST", sub.body.as_bytes(), true)
    } else {
        browser_load(&sub.url, true)
    }
}

/// Update cursor shape when hovering the browser surface.
pub(super) fn browser_hover(sx: i32, sy: i32) {
    // During progressive load, never re-layout/repaint: `browser_repaint` reads
    // the *previous* page's session HTML and was flashing it back whenever the
    // mouse moved over a link. Keep the Wait cursor and leave the stage pixels.
    if browser_is_loading() {
        #[cfg(not(test))]
        {
            crate::framebuffer::set_cursor_shape(crate::framebuffer::CursorShape::Wait);
        }
        let _ = (sx, sy);
        return;
    }
    let scroll = BROWSER.with(|s| s.as_ref().map(|b| b.scroll_y).unwrap_or(0));
    let content_y = sy + scroll;
    let (kind, link_rect) = BROWSER_LAYOUT.with(|slot| match slot.as_ref() {
        Some(lay) => {
            let kind = crate::browser::layout::cursor_at(lay, sx, content_y);
            // Content-space rect of the link under the cursor, for hover
            // underline (topmost match).
            let rect = lay
                .links
                .iter()
                .rev()
                .find(|b| sx >= b.x && sx < b.x + b.w && content_y >= b.y && content_y < b.y + b.h)
                .map(|b| (b.x, b.y, b.w, b.h));
            (kind, rect)
        }
        None => (crate::browser::layout::CursorKind::Default, None),
    });
    // Underline the hovered link (repaint only when the hovered link changes).
    // Use a paint-from-cached-layout path so we don't re-run the full layout
    // pipeline on every hover change (still needed for underline chrome).
    if crate::browser::set_hover_link(link_rect) {
        browser_repaint_hover();
    }
    #[cfg(not(test))]
    {
        use crate::framebuffer::CursorShape;
        let shape = match kind {
            crate::browser::layout::CursorKind::Pointer => CursorShape::Hand,
            crate::browser::layout::CursorKind::Text => CursorShape::IBeam,
            crate::browser::layout::CursorKind::Default => CursorShape::Arrow,
        };
        crate::framebuffer::set_cursor_shape(shape);
    }
    let _ = (kind, sx, sy);
}

/// Hover-only repaint: paint the cached layout with the new hover underline.
/// Avoids a full `layout_session` (which re-parses HTML) on every mouse move.
pub(super) fn browser_repaint_hover() {
    if browser_is_loading() {
        return;
    }
    let Some(lay) = BROWSER_LAYOUT.with(|s| s.clone()) else {
        return;
    };
    let (url, title, scroll) = BROWSER.with(|s| {
        s.as_ref()
            .map(|b| (b.url.clone(), b.title.clone(), b.scroll_y))
            .unwrap_or_else(|| {
                (
                    alloc::string::String::new(),
                    alloc::string::String::new(),
                    0,
                )
            })
    });
    if url.is_empty() {
        return;
    }
    let progress = browser_progress_opt();
    let sel = browser_sel_range();
    let (pixels, content_h) =
        crate::browser::paint_layout_chrome_sel(&lay, browser_vh(), scroll, progress, sel);
    BROWSER.with(|s| {
        if let Some(b) = s.as_mut() {
            b.content_height = content_h;
        }
    });
    browser_present(&pixels, &title, &url, scroll, content_h);
}

pub(super) fn browser_status() -> alloc::string::String {
    BROWSER.with(|s| match s.as_ref() {
        Some(b) => alloc::format!(
            "ok:url={} title={} scroll={} content_h={} size={}x{}",
            b.url,
            b.title,
            b.scroll_y,
            b.content_height,
            browser_vw(),
            browser_vh()
        ),
        None => alloc::string::String::from("ok:empty"),
    })
}

pub(super) fn browser_links() -> alloc::string::String {
    let html = match BROWSER.with(|s| s.as_ref().map(|b| b.html.clone())) {
        Some(h) => h,
        None => return alloc::string::String::from("error: no page open"),
    };
    let links = crate::browser::page_links(&html);
    if links.is_empty() {
        return alloc::string::String::from("(no links)");
    }
    let mut out = alloc::string::String::new();
    for (i, (h, t)) in links.iter().enumerate().take(64) {
        out.push_str(&alloc::format!("{}. {} — {}\n", i + 1, t, h));
    }
    out
}

pub(super) fn browser_text(max: usize) -> alloc::string::String {
    let html = match BROWSER.with(|s| s.as_ref().map(|b| b.html.clone())) {
        Some(h) => h,
        None => return alloc::string::String::from("error: no page open"),
    };
    let mut t = crate::browser::page_text(&html);
    if t.len() > max {
        t.truncate(max);
        t.push_str("…");
    }
    t
}

/// Whether the browser tab is showing (for key routing).
pub fn browser_loaded() -> bool {
    BROWSER.with(|s| s.is_some())
}

/// Key-path variant of [`browser_dispatch_nav`]: base = current session URL.
#[cfg(not(test))]
pub(super) fn browser_dispatch_nav_key() -> Option<alloc::string::String> {
    let base = BROWSER.with(|s| s.as_ref().map(|b| b.url.clone()))?;
    browser_dispatch_nav(&base)
}

/// Deliver `types` events to the page-JS element behind form control `idx`
/// (when the persistent page is live and the control carries a stamped
/// element index). Syncs the control's live value into the JS DOM first so
/// `input`/`change` handlers read what the user typed. Returns true when any
/// handler called `preventDefault()`.
#[cfg(not(test))]
pub(super) fn browser_control_event(idx: usize, types: &[&str]) -> bool {
    if !crate::browser::js_just::page_active() {
        return false;
    }
    let ei = BROWSER_LAYOUT.with(|s| {
        s.as_ref().and_then(|l| {
            l.controls
                .iter()
                .find(|c| c.index == idx)
                .and_then(|c| c.elem_idx)
        })
    });
    let Some(ei) = ei else { return false };
    let val = BROWSER
        .with(|s| s.as_ref().and_then(|b| b.control_values.get(&idx).cloned()))
        .unwrap_or_default();
    crate::browser::js_just::page_with_dom(|d| {
        if let Some(e) = d.elements.get_mut(ei) {
            e.value = val.clone();
        }
    });
    let evs: alloc::vec::Vec<crate::browser::js_just::PageEvent> = types
        .iter()
        .map(|t| crate::browser::js_just::PageEvent {
            target: ei,
            type_: alloc::string::String::from(*t),
            x: 0,
            y: 0,
        })
        .collect();
    crate::browser::js_just::page_dispatch(&evs)
        .iter()
        .any(|&p| p)
}

/// Handle a key while the browser surface is focused. Returns true if consumed.
#[cfg(not(test))]
pub(super) fn browser_key(byte: u8) -> bool {
    if !browser_loaded() {
        return false;
    }
    // Text entry into focused form control.
    let focused = BROWSER.with(|s| s.as_ref().and_then(|b| b.focused));
    if let Some(idx) = focused {
        match byte {
            0x1b => {
                // Esc clears focus.
                BROWSER.with(|s| {
                    if let Some(b) = s.as_mut() {
                        b.focused = None;
                    }
                });
                let _ = browser_repaint();
                return true;
            }
            0x09 => {
                // Tab → next text control. Guard: `cycle()` + `skip_while` only
                // terminates if the focused index is IN the text-entry list.
                if let Some((lay, ..)) = browser_layout_session() {
                    let in_list = lay
                        .controls
                        .iter()
                        .any(|c| c.index == idx && c.kind.is_text_entry());
                    let next = if !in_list {
                        lay.controls
                            .iter()
                            .filter(|c| c.kind.is_text_entry())
                            .map(|c| c.index)
                            .next()
                    } else {
                        lay.controls
                            .iter()
                            .filter(|c| c.kind.is_text_entry())
                            .map(|c| c.index)
                            .cycle()
                            .skip_while(|&i| i != idx)
                            .nth(1)
                    };
                    BROWSER.with(|s| {
                        if let Some(b) = s.as_mut() {
                            b.focused = next.or(Some(idx));
                        }
                    });
                    let _ = browser_repaint();
                }
                return true;
            }
            0x0d | 0x0a => {
                // Enter → change + submit into page JS, then submit the
                // owning form (unless a handler preventDefault()ed).
                let prevented = browser_control_event(idx, &["change", "submit"]);
                if let Some(out) = browser_dispatch_nav_key() {
                    let _ = out;
                    return true;
                }
                if prevented {
                    let _ = browser_repaint();
                    return true;
                }
                if let Some((lay, ..)) = browser_layout_session() {
                    if let Some(c) = lay.controls.get(idx) {
                        if let Some(sub) = lay
                            .controls
                            .iter()
                            .find(|x| x.form_id == c.form_id && x.kind.is_submit())
                            .cloned()
                        {
                            let _ = browser_submit_control(&lay, &sub);
                            return true;
                        }
                        // Orphan text field: no-op submit.
                    }
                }
                return true;
            }
            0x08 | 0x7f => {
                BROWSER.with(|s| {
                    if let Some(b) = s.as_mut() {
                        if let Some(v) = b.control_values.get_mut(&idx) {
                            v.pop();
                        }
                    }
                });
                let _ = browser_control_event(idx, &["input"]);
                let _ = browser_repaint();
                return true;
            }
            b if b >= 0x20 && b < 0x7f => {
                BROWSER.with(|s| {
                    if let Some(sess) = s.as_mut() {
                        let e = sess.control_values.entry(idx).or_default();
                        if e.len() < 512 {
                            e.push(b as char);
                        }
                    }
                });
                let _ = browser_control_event(idx, &["input"]);
                let _ = browser_repaint();
                return true;
            }
            _ => {}
        }
    }
    match byte {
        b'j' | b'J' => {
            let _ = browser_scroll(browser_vh() / 3);
            true
        }
        b'k' | b'K' => {
            let _ = browser_scroll(-(browser_vh() / 3));
            true
        }
        b' ' if focused.is_none() => {
            let _ = browser_scroll(browser_vh() - 40);
            true
        }
        b'b' | b'B' if focused.is_none() => {
            let _ = browser_back();
            true
        }
        b'r' | b'R' if focused.is_none() => {
            let url = BROWSER.with(|s| s.as_ref().map(|b| b.url.clone()));
            if let Some(u) = url {
                let _ = browser_load(&u, false);
            }
            true
        }
        // PageUp / we only get plain bytes here; Pg keys come as CSI elsewhere.
        _ => false,
    }
}
