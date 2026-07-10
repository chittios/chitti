//! **Browser engine** — inspired by [Ladybird](https://github.com/LadybirdBrowser/ladybird)
//! / LibWeb pipeline (parse → style → layout → paint) and LibJS (sandboxed
//! interpreter). C++ LibWeb/LibJS cannot link into this `no_std` kernel; we
//! reimplement the **same stage split** in pure Rust. Local reference tree:
//! clone Ladybird beside this repo (e.g. `../ladybird-ref`) and open:
//!
//! | Chitti module   | Ladybird reference                                      |
//! |-----------------|---------------------------------------------------------|
//! | `html`/`css`/`layout`/`paint` | `Libraries/LibWeb/{HTML,CSS,Layout,…}`     |
//! | `js`            | `Libraries/LibJS`                                       |
//! | `cache`         | `Libraries/LibHTTP/Cache/{MemoryCache,CacheMode}`       |
//! | `loader`        | `Libraries/LibWeb/Loader/{ResourceLoader,LoadRequest}`  |
//! | `worker`        | `Libraries/LibWeb/HTML/Worker*` + `Services/WebWorker`  |
//! | `elements`      | MDN HTML elements + Ladybird `HTML*Element` table       |
//! | `cors`          | `LibWeb/Fetch/Infrastructure/HTTP/CORS`                 |
//! | iframe/frames   | `HTMLIFrameElement` + nested navigable                  |
//! | postMessage     | `MessageEvent` / Window postMessage                     |
//! | `events`        | HTML event loop + EventTarget                           |
//! | `storage`       | cookies / localStorage / sessionStorage                 |
//! | `js_bc`         | LibJS-style bytecode VM (not a native-code JIT)         |
//! | JS tiers        | Reference: `third_party/just-ref` (applegrew/just) —
//! |                 | tree-walk + stack/register BC; Cranelift JIT is host-
//! |                 | only / x86_64 numeric — we keep no_std dual-arch BC.  |
//! | `flex`          | Flex/Grid placement math                                |
//! | `svg`           | SVG + MathML subset                                     |
//! | `wasm_page`     | Page WASM via wasmi (agent runtime)                     |
//! | `cors` preflight| OPTIONS + Allow-Methods/Headers                         |
//!
//! ```text
//! HTML  → html::parse     (DOM + extract <style>/<script>)     ≈ LibWeb/HTML
//!       → js / js_bc      (scripts + bytecode fast path)         ≈ LibJS
//!       → events::EventLoop (tasks / microtasks / listeners)     ≈ HTML EventLoop
//!       → css::Stylesheet (@layer cascade + flex/grid props)     ≈ LibCSS
//!       → layout + flex   (+ iframe / svg boxes)                 ≈ LibWeb/Layout
//!       → paint → present_surface                                ≈ LibGfx / WebContent
//! assets → loader (+ CORS preflight, cookies)                    ≈ ResourceLoader
//! storage → cookies / localStorage / sessionStorage              ≈ Web Storage
//! ```
//!
//! Honest scope: these are **working foundations** tested in-kernel — not
//! Chrome parity. Native-code JS JIT, full CSS Grid auto-placement, complete
//! SVG filters/SMIL, and every MathML element remain follow-ups. Unknown HTML
//! tags still render as blocks (`HTMLUnknownElement` spirit).
//!
//! Tokenization: `htmlparser` (`no_std`). Engines: first-party pure logic + unit tests.

pub mod cache;
pub mod canvas;
pub mod cors;
pub mod css;
pub mod elements;
pub mod events;
pub mod flex;
pub mod form;
pub mod html;
pub mod httpdate;
pub mod js;
pub mod js_bc;
pub mod js_just;
pub mod layout;
pub mod loader;
pub mod paint;
pub mod psl;
pub mod storage;
pub mod svg;
pub mod url;
pub mod wasm_page;
pub mod worker;

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

/// End-to-end pure render: HTML string → RGB frame + metadata.
pub struct Frame {
    pub title: String,
    pub width: i32,
    pub height: i32,
    pub content_height: i32,
    pub pixels: Vec<u32>,
    pub layout: layout::Layout,
    /// `console.log` lines from page scripts.
    pub js_log: Vec<String>,
}

/// Side effects from page scripts the host may act on.
#[derive(Clone, Debug, Default)]
pub struct ScriptEffects {
    pub log: Vec<String>,
    /// `window.location = …` / `location.href = …`
    pub navigate: Option<String>,
    /// `window.scrollTo(0, y)` / `scrollBy`
    pub scroll_y: Option<i32>,
}

/// Build a frame without painting (so the host can fill images first).
pub fn layout_html(html_src: &str, vw: i32, vh: i32) -> (html::Document, layout::Layout, Vec<String>) {
    let (doc, lay, effects) = layout_html_ex(html_src, vw, vh, "");
    (doc, lay, effects.log)
}

/// Like [`layout_html`] but seeds `location.href` and returns script effects.
pub fn layout_html_ex(
    html_src: &str,
    vw: i32,
    vh: i32,
    location_href: &str,
) -> (html::Document, layout::Layout, ScriptEffects) {
    let mut doc = html::parse(html_src);
    let mut dom = js::JsDom::from_document(&doc);
    dom.location_href = if location_href.is_empty() {
        String::from("about:blank")
    } else {
        location_href.into()
    };
    dom.inner_width = vw;
    dom.inner_height = vh;
    let scripts = doc.scripts.clone();
    let _ = js::run_scripts(&mut dom, &scripts);
    doc.title = dom.title.clone();
    js::commit_to_tree(&mut doc.root, &dom);
    let effects = ScriptEffects {
        log: dom.log.clone(),
        navigate: dom.navigate.clone(),
        scroll_y: dom.scroll_to,
    };
    let sheet = css::Stylesheet::parse(&doc.stylesheets);
    let mut lay = layout::layout_document(&doc.root, &sheet, vw, vh);
    // Apply canvas 2d pixels drawn during script execution.
    js::apply_canvases_to_layout(&dom, &mut lay);
    finish_layout(&doc, &mut lay, vw, vh);
    (doc, lay, effects)
}

/// Shared post-layout fixups: reader-mode fallback for pages whose DOM/CSS
/// path produced no visible geometry, and the dark-page ink fixup (a dark
/// `bg` with mostly-dark ink becomes warm paper + dark ink).
fn finish_layout(doc: &html::Document, lay: &mut layout::Layout, vw: i32, vh: i32) {
    if lay.runs.is_empty()
        && lay.images.is_empty()
        && lay.controls.is_empty()
        && lay.frames.is_empty()
    {
        let plain = html::collect_text(&doc.root);
        if !plain.is_empty() || !doc.title.is_empty() {
            *lay = layout::layout_reader(&doc.title, &plain, vw, vh);
        }
    }
    if is_dark(lay.bg) && !lay.runs.is_empty() {
        let light_ink = lay.runs.iter().filter(|r| !is_dark(r.color)).count();
        if light_ink * 2 < lay.runs.len() {
            lay.bg = 0xf5f0e8;
            for r in &mut lay.runs {
                if is_dark(r.color) {
                    r.color = 0x2a2a2a;
                }
            }
        }
    }
}

/// Session-scoped subresources for the persistent-page render path
/// ([`layout_session`]): fetched external stylesheet bodies and decoded CSS
/// background images, both keyed by **absolute URL** (resolved against the
/// document URL, matching how the host stored them).
pub struct SessionAssets<'a> {
    /// `<link rel=stylesheet>` (+`@import`-expanded) bodies by absolute URL.
    pub css_external: &'a BTreeMap<String, String>,
    /// Decoded `background-image` pixels by absolute URL:
    /// `(0x00RRGGBB row-major, width, height)`.
    pub bg_pixels: &'a BTreeMap<String, (Vec<u32>, usize, usize)>,
}

/// Session render for the **persistent JS page** path: parse, merge inline +
/// external stylesheets in exact document order, commit the live page DOM
/// (if [`js_just::page_active`]) instead of re-running scripts, lay out with
/// the page's interactive element set, fill background-image pixels from
/// `assets`, and apply script-drawn canvases. Returns the doc, the layout,
/// and any **fresh** JS console lines (drained from the page log).
///
/// One-shot callers (iframes, workers, tests, the doc agent) keep using
/// [`layout_html_ex`] — this function never boots or runs scripts itself.
pub fn layout_session(
    html_src: &str,
    vw: i32,
    vh: i32,
    url: &str,
    assets: &SessionAssets<'_>,
) -> (html::Document, layout::Layout, Vec<String>) {
    let mut doc = html::parse(html_src);

    // Merge stylesheets in exact document order (inline bodies verbatim,
    // external hrefs resolved against the document URL and looked up in the
    // fetched set; missing sheets are skipped).
    let mut css_all = String::new();
    for s in &doc.styles_ordered {
        match s {
            html::StyleSrc::Inline(body) => {
                css_all.push_str(body);
                css_all.push('\n');
            }
            html::StyleSrc::External(href) => {
                let abs = url::resolve(url, href).unwrap_or_else(|| href.clone());
                if let Some(body) = assets
                    .css_external
                    .get(&abs)
                    .or_else(|| assets.css_external.get(href))
                {
                    css_all.push_str(body);
                    css_all.push('\n');
                }
            }
        }
    }

    // Live page DOM → tree (title, mutations, JS-created elements), plus the
    // interactive element set for ElemBox hit-testing. No page → stamp only.
    let mut js_log: Vec<String> = Vec::new();
    let mut interactive: Vec<usize> = Vec::new();
    if js_just::page_active() {
        js_just::page_with_dom(|dom| {
            doc.title = dom.title.clone();
            js::commit_full(&mut doc.root, dom);
            js_log = core::mem::take(&mut dom.log);
        });
        interactive = js_just::page_interactive_elems();
    } else {
        js::stamp_elem_indices(&mut doc.root);
    }

    let sheet = css::Stylesheet::parse(&css_all);
    let mut lay = layout::layout_document_ex(&doc.root, &sheet, vw, vh, &interactive);

    // Fill background-image pixels by absolute URL (raw src as fallback).
    for bb in lay.bg_boxes.iter_mut() {
        let abs = url::resolve(url, &bb.src).unwrap_or_else(|| bb.src.clone());
        if let Some((px, w, h)) = assets
            .bg_pixels
            .get(&abs)
            .or_else(|| assets.bg_pixels.get(&bb.src))
        {
            bb.pixels = Some(px.clone());
            bb.src_w = *w;
            bb.src_h = *h;
        }
    }

    // Canvas 2d pixels drawn by page scripts / handlers.
    js_just::page_with_dom(|dom| js::apply_canvases_to_layout(dom, &mut lay));
    finish_layout(&doc, &mut lay, vw, vh);
    (doc, lay, js_log)
}

/// Paint nested HTML into an iframe/frame slot (pure for `srcdoc`; host loads
/// remote `src` then calls this). Depth is capped to avoid recursive iframe bombs.
pub fn fill_frame_slot(fr: &mut layout::FrameBox, html_src: &str, base_url: &str) {
    const MAX_NEST: i32 = 2;
    fill_frame_slot_depth(fr, html_src, base_url, MAX_NEST);
}

fn fill_frame_slot_depth(
    fr: &mut layout::FrameBox,
    html_src: &str,
    base_url: &str,
    depth: i32,
) {
    if depth <= 0 || html_src.is_empty() {
        return;
    }
    let vw = fr.w.max(40);
    let vh = fr.h.max(40);
    let (_doc, mut lay, effects) = layout_html_ex(html_src, vw, vh, base_url);
    // Nested iframes: only fill srcdoc children at reduced depth (no network here).
    if depth > 1 {
        for nested in lay.frames.iter_mut() {
            if !nested.srcdoc.is_empty() {
                fill_frame_slot_depth(nested, &nested.srcdoc.clone(), base_url, depth - 1);
            }
        }
    }
    let _ = effects;
    let pixels = paint::paint(&lay, 0);
    fr.src_w = vw as usize;
    fr.src_h = vh as usize;
    fr.pixels = Some(pixels);
}

/// Decode first video frame into a `<video>` frame box (H.264 path).
pub fn fill_video_slot(fr: &mut layout::FrameBox, bytes: alloc::vec::Vec<u8>) {
    if fr.kind != layout::EmbedKind::Video {
        return;
    }
    let Ok(mut dec) = crate::video::StreamDecoder::open(bytes) else {
        return;
    };
    if !dec.seek_decode(0) {
        return;
    }
    let Some(frame) = dec.cur_frame() else {
        return;
    };
    let tw = fr.w.max(1) as usize;
    let th = fr.h.max(1) as usize;
    if frame.w == 0 || frame.h == 0 || frame.pixels.is_empty() {
        return;
    }
    let (fw, fh) = crate::image::fit(frame.w, frame.h, tw, th);
    let img = crate::image::Image {
        w: frame.w,
        h: frame.h,
        pixels: frame.pixels.clone(),
    };
    let scaled = if fw == frame.w && fh == frame.h {
        img
    } else {
        crate::image::resize(&img, fw, fh)
    };
    fr.src_w = scaled.w;
    fr.src_h = scaled.h;
    fr.pixels = Some(scaled.pixels);
}

/// Decode image bytes into a layout image slot (uses `crate::image`).
pub fn fill_image_slot(im: &mut layout::ImageBox, bytes: &[u8]) {
    let Ok(decoded) = crate::image::decode(bytes) else {
        return;
    };
    let max_w = im.w.max(1) as usize;
    let max_h = im.h.max(1) as usize;
    let (fw, fh) = crate::image::fit(decoded.w, decoded.h, max_w, max_h);
    let scaled = if fw == decoded.w && fh == decoded.h {
        decoded
    } else {
        crate::image::resize(&decoded, fw, fh)
    };
    im.w = scaled.w as i32;
    im.h = scaled.h as i32;
    im.src_w = scaled.w;
    im.src_h = scaled.h;
    im.pixels = Some(scaled.pixels);
}

/// Parse + JS + CSS + layout + paint with `scroll_y`. Pure (no net; images empty).
pub fn render_html(html_src: &str, vw: i32, vh: i32, scroll_y: i32) -> Frame {
    let (doc, lay, effects) = layout_html_ex(html_src, vw, vh, "");
    let max_scroll = (lay.content_height - vh).max(0);
    let sy = scroll_y.clamp(0, max_scroll);
    let chrome = paint::Chrome {
        progress: None,
        progress_bottom: false,
        scrollbar: lay.content_height > vh,
    };
    let pixels = paint::paint_chrome(&lay, sy, chrome);
    Frame {
        title: doc.title,
        width: lay.width,
        height: lay.height,
        content_height: lay.content_height,
        pixels,
        layout: lay,
        js_log: effects.log,
    }
}

/// Paint an existing layout (after host filled images).
pub fn paint_layout(lay: &layout::Layout, vh: i32, scroll_y: i32) -> (Vec<u32>, i32) {
    paint_layout_chrome(lay, vh, scroll_y, None)
}

/// Paint with optional load progress (0..=100).
pub fn paint_layout_chrome(
    lay: &layout::Layout,
    vh: i32,
    scroll_y: i32,
    progress: Option<u8>,
) -> (Vec<u32>, i32) {
    let max_scroll = (lay.content_height - vh).max(0);
    let sy = scroll_y.clamp(0, max_scroll);
    let chrome = paint::Chrome {
        progress,
        progress_bottom: false,
        scrollbar: lay.content_height > vh,
    };
    (paint::paint_chrome(lay, sy, chrome), lay.content_height)
}

fn is_dark(rgb: u32) -> bool {
    let r = ((rgb >> 16) & 0xff) as u32;
    let g = ((rgb >> 8) & 0xff) as u32;
    let b = (rgb & 0xff) as u32;
    // Luma-ish; below ~40 is "near black".
    (r * 3 + g * 6 + b) / 10 < 40
}

/// Plain text for the agent (`browser_text`) — after JS mutations.
pub fn page_text(html_src: &str) -> String {
    let mut doc = html::parse(html_src);
    let mut dom = js::JsDom::from_document(&doc);
    let scripts = doc.scripts.clone();
    let _ = js::run_scripts(&mut dom, &scripts);
    js::commit_to_tree(&mut doc.root, &dom);
    html::collect_text(&doc.root)
}

/// Links as `(href, text)`.
pub fn page_links(html_src: &str) -> Vec<(String, String)> {
    let doc = html::parse(html_src);
    let mut out = Vec::new();
    html::collect_links(&doc.root, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn render_pipeline_smoke() {
        let f = render_html(
            r#"<html><head><title>T</title></head><body>
            <h1>Hello</h1><p><a href="https://ex.com/a">Go</a></p>
            </body></html>"#,
            320,
            200,
            0,
        );
        assert_eq!(f.title, "T");
        assert!(!f.pixels.is_empty());
        assert!(!f.layout.links.is_empty());
        let t = page_text("<html><body><p>X &amp; Y</p></body></html>");
        assert!(t.contains("X") && t.contains("Y"), "{t}");
    }

    #[test_case]
    fn css_and_js_pipeline() {
        // r## so `"#…"` in CSS/JS does not terminate the raw string.
        let html = r##"<!DOCTYPE html><html><head>
            <title>Old</title>
            <style>
              body { background: #112233; }
              #msg { color: #ff0000; font-size: 20px; }
            </style>
            <script>
              document.title = "New";
              var el = document.getElementById("msg");
              el.innerText = "Styled";
              console.log("ran");
            </script>
            </head><body><p id="msg">Hello</p></body></html>"##;
        let doc = html::parse(html);
        assert!(
            doc.stylesheets.contains("112233"),
            "extracted css: {:?}",
            doc.stylesheets
        );
        assert_eq!(doc.scripts.len(), 1);
        let sheet = css::Stylesheet::parse(&doc.stylesheets);
        assert!(sheet.rule_count() >= 1, "rules={}", sheet.rule_count());
        let f = render_html(html, 320, 200, 0);
        assert_eq!(f.title, "New");
        assert_eq!(
            f.layout.bg, 0x112233,
            "page bg; rules={} css={:?}",
            sheet.rule_count(),
            doc.stylesheets
        );
        assert!(f.js_log.iter().any(|l| l.contains("ran")), "{:?}", f.js_log);
        assert!(
            f.layout.runs.iter().any(|r| r.color == 0xff0000 && r.text.contains("Styled")),
            "runs: {:?}",
            f.layout.runs
        );
    }

    #[test_case]
    fn layout_session_merges_external_css_in_order() {
        js_just::page_close(); // isolate from any page another test booted
        let html = r##"<html><head>
            <style>p { color: #00ff00; }</style>
            <link rel="stylesheet" href="/site.css">
            </head><body><p>Styled</p></body></html>"##;
        let mut css_ext = BTreeMap::new();
        css_ext.insert(
            String::from("https://ex.com/site.css"),
            String::from("p { color: #ff0000; }"),
        );
        let bg = BTreeMap::new();
        let assets = SessionAssets {
            css_external: &css_ext,
            bg_pixels: &bg,
        };
        let (_doc, lay, _log) = layout_session(html, 320, 200, "https://ex.com/page", &assets);
        // The later external sheet overrides the earlier inline one
        // (document-order cascade).
        assert!(
            lay.runs.iter().any(|r| r.color == 0xff0000),
            "runs: {:?}",
            lay.runs
        );
        // A missing external sheet is skipped, not fatal: inline still applies.
        let empty = BTreeMap::new();
        let assets2 = SessionAssets {
            css_external: &empty,
            bg_pixels: &bg,
        };
        let (_d2, lay2, _l2) = layout_session(html, 320, 200, "https://ex.com/page", &assets2);
        assert!(lay2.runs.iter().any(|r| r.color == 0x00ff00));
    }

    #[test_case]
    fn layout_session_fills_bg_pixels() {
        js_just::page_close();
        let html = r##"<html><head><style>
            #h { background-image: url("bg.png"); }
            </style></head><body><div id="h">X</div></body></html>"##;
        let css_ext = BTreeMap::new();
        let mut bg = BTreeMap::new();
        bg.insert(
            String::from("https://ex.com/a/bg.png"),
            (alloc::vec![0xff0000u32; 4], 2usize, 2usize),
        );
        let assets = SessionAssets {
            css_external: &css_ext,
            bg_pixels: &bg,
        };
        let (_doc, lay, _log) =
            layout_session(html, 320, 200, "https://ex.com/a/page.html", &assets);
        let bb = lay.bg_boxes.first().expect("bg box");
        assert_eq!(bb.src, "bg.png");
        assert!(bb.pixels.is_some(), "bg pixels filled from assets");
        assert_eq!((bb.src_w, bb.src_h), (2, 2));
    }

    #[test_case]
    fn video_source_child_and_canvas_layout() {
        let html = r#"<html><body>
            <video width="160" height="90"><source src="clip.mp4" type="video/mp4"></video>
            <canvas id="cv" width="40" height="30"></canvas>
            </body></html>"#;
        let (_doc, lay, _) = layout_html_ex(html, 320, 200, "https://ex.com/page");
        let vid = lay
            .frames
            .iter()
            .find(|f| f.kind == layout::EmbedKind::Video)
            .expect("video frame");
        assert_eq!(vid.src, "clip.mp4");
        assert!(lay.frames.iter().any(|f| f.kind == layout::EmbedKind::Canvas));
    }
}
