//! Host harness for the kernel's CSS + layout engine.
//!
//! Mounts `kernel/src/browser/{html,css,elements,flex,layout}.rs` with `#[path]`
//! — the same "one implementation, two build targets" trick `tools/h264diff`,
//! `tools/pngbench` and `tools/cortexdiff` use — so a page can be laid out on
//! the host in milliseconds instead of a two-minute QEMU boot. A fix made here
//! IS the kernel fix; there is no second engine to drift.
//!
//! What it does NOT do: paint. It reports the boxes and text runs layout
//! produced (position, size, colour), which is the layer where "the card has no
//! background" and "the row collapsed to zero height" are decided.
//!
//! Usage:
//!   chitti-csslayout <page.html> [--width N] [--height N] [--filter substr]
//!   chitti-csslayout <page.html> --rects        # background/border boxes only
//!   chitti-csslayout <page.html> --runs         # text runs only
//!
//! `<link rel=stylesheet href=…>` is resolved against the page's directory and
//! read off disk, exactly as the browse host resolves it against the document
//! base.

use std::path::{Path, PathBuf};

// The kernel modules are `no_std` + `alloc`; on the host `alloc` is `std`'s.
extern crate alloc;

/// Font metrics. The kernel measures with the real Geist Mono face; here a
/// proportional-ish approximation is enough to answer layout questions (does a
/// row have height, does a card get a background) without shipping a
/// rasterizer. Anything that depends on exact glyph advances must be checked in
/// the kernel, not here — and the harness says so rather than pretending.
mod font_ttf {
    pub fn line_height(px: f32) -> f32 {
        (px * 1.25).max(1.0)
    }
    /// Canvas text painting is not exercised here.
    pub fn blit_run(
        _fb: &mut [u32],
        _w: usize,
        _h: usize,
        _x: i32,
        _y: i32,
        _text: &str,
        _px: f32,
        _color: u32,
    ) {
    }
    pub fn measure(text: &str, px: f32) -> f32 {
        // Geist Mono is 0.6 em per cell; the browser's default face is
        // proportional, so this is deliberately a rough middle.
        text.chars().count() as f32 * px * 0.55
    }
}

/// `mm::Locked` on the host: layout is single-threaded here too, so a plain
/// `RefCell` behind the same `.with()` shape is faithful.
pub mod mm {
    use std::cell::RefCell;
    pub struct Locked<T>(RefCell<T>);
    // SAFETY: the harness is single-threaded.
    unsafe impl<T> Sync for Locked<T> {}
    impl<T> Locked<T> {
        pub const fn new(v: T) -> Self {
            Self(RefCell::new(v))
        }
        pub fn with<R>(&self, f: impl FnOnce(&mut T) -> R) -> R {
            f(&mut self.0.borrow_mut())
        }
    }
}

pub mod browser;

use browser::{css::Stylesheet, html, layout};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!(
            "usage: chitti-csslayout <page.html> [--width N] [--height N] \
             [--rects|--runs] [--filter substr]"
        );
        std::process::exit(2);
    }
    let mut page = None;
    let mut width = 590i32;
    let mut height = 693i32;
    let mut only_rects = false;
    let mut only_runs = false;
    let mut filter: Option<String> = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--width" => width = it.next().and_then(|v| v.parse().ok()).unwrap_or(width),
            "--height" => height = it.next().and_then(|v| v.parse().ok()).unwrap_or(height),
            "--rects" => only_rects = true,
            "--runs" => only_runs = true,
            "--filter" => filter = it.next().cloned(),
            other => page = Some(PathBuf::from(other)),
        }
    }
    let Some(page) = page else {
        eprintln!("no page given");
        std::process::exit(2);
    };

    let src = match std::fs::read_to_string(&page) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{}: {e}", page.display());
            std::process::exit(1);
        }
    };
    let doc = html::parse(&src);
    let dir = page.parent().unwrap_or(Path::new(".")).to_path_buf();

    // Sheets in exact document order — the cascade depends on it, so an
    // external sheet must not be hoisted above a <style> that follows it.
    let mut css_text = String::new();
    for src in &doc.styles_ordered {
        match src {
            html::StyleSrc::Inline(body) => {
                println!("css: <style> ({} bytes)", body.len());
                css_text.push_str(body);
                css_text.push('\n');
            }
            html::StyleSrc::External(href) => {
                let p = resolve(&dir, href);
                match std::fs::read_to_string(&p) {
                    Ok(body) => {
                        println!("css: {} ({} bytes)", p.display(), body.len());
                        css_text.push_str(&body);
                        css_text.push('\n');
                    }
                    Err(e) => println!("css: {} MISSING ({e})", p.display()),
                }
            }
        }
    }

    let sheet = Stylesheet::parse_with_viewport(&css_text, width);
    let lay = layout::layout_document(&doc.root, &sheet, width, height);

    println!(
        "page: {}  viewport {width}x{height}  content_height={}  bg=#{:06x}",
        page.display(),
        lay.content_height,
        lay.bg
    );
    println!(
        "boxes: {} rects, {} runs, {} controls, {} links, {} images",
        lay.rects.len(),
        lay.runs.len(),
        lay.controls.len(),
        lay.links.len(),
        lay.images.len()
    );

    let keep = |s: &str| filter.as_ref().map(|f| s.contains(f.as_str())).unwrap_or(true);

    if !only_runs {
        println!("\n-- rects (background + border boxes, paint order) --");
        for (i, r) in lay.rects.iter().enumerate() {
            let label = format!("#{:06x}", r.color);
            if !keep(&label) && filter.is_some() {
                continue;
            }
            println!(
                "  [{i:3}] {:>5},{:<5} {:>4}x{:<4}  #{:06x}{}{}",
                r.x,
                r.y,
                r.w,
                r.h,
                r.color,
                if r.radius > 0 { format!("  radius={}", r.radius) } else { String::new() },
                if r.blur > 0 { format!("  blur={}", r.blur) } else { String::new() },
            );
        }
    }
    if !only_rects {
        println!("\n-- text runs --");
        for (i, r) in lay.runs.iter().enumerate() {
            if !keep(&r.text) {
                continue;
            }
            println!(
                "  [{i:3}] {:>5},{:<5} #{:06x} {}px{}{}  {:?}",
                r.x,
                r.y,
                r.color,
                r.font_size,
                if r.bold { " bold" } else { "" },
                if r.underline { " underline" } else { "" },
                r.text
            );
        }
        if !lay.controls.is_empty() {
            println!("\n-- form controls --");
            for (i, c) in lay.controls.iter().enumerate() {
                println!(
                    "  [{i:3}] {:>5},{:<5} {:>4}x{:<4} {:?} value={:?} bg={:?} fg={:?}",
                    c.x, c.y, c.w, c.h, c.kind, c.value, c.bg, c.fg
                );
            }
        }
    }
}

fn resolve(dir: &Path, href: &str) -> PathBuf {
    let href = href.split('#').next().unwrap_or(href);
    if let Some(abs) = href.strip_prefix('/') {
        return PathBuf::from(abs);
    }
    dir.join(href.trim_start_matches("./"))
}
