//! Price the candidate web crates against the engines ChittiOS already has.
//!
//! Same rule as `tools/pngbench`, which measured wasmi at 47-67x native and so
//! settled the "decode PNG in wasm" question before anyone ported anything: a
//! candidate is measured on real inputs, against the incumbent, before it is
//! vendored. `no_std` feasibility and licence are cheap to read off a manifest;
//! **speed and correctness are not**, and they are what decides.
//!
//! Our side is mounted with `#[path]`, so it is the code that ships — not a
//! reimplementation that could flatter or slander itself.
//!
//! Usage: chitti-webbench <corpus-dir>   (defaults to assets/samples/html)

extern crate alloc;

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[path = "browser/mod.rs"]
mod browser;

/// Run `f` enough times to be worth timing, and report the per-iteration mean.
///
/// A single run of a 90 KiB parse is well under a millisecond, and one sample
/// of that is noise; the loop is what makes the two numbers comparable.
fn bench<T>(iters: u32, mut f: impl FnMut() -> T) -> Duration {
    // Warm up: first touch pays for page faults and any lazy init.
    let _ = f();
    let t0 = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(f());
    }
    t0.elapsed() / iters
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

fn main() {
    let root = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("assets/samples/html"));

    let mut pages: Vec<(String, String)> = Vec::new();
    collect(&root, "html", &mut pages);
    // Whatever extra pages were passed alongside (a saved real-world page is
    // worth more than any fixture: it has the malformed markup fixtures lack).
    for extra in std::env::args().skip(2) {
        if let Ok(s) = std::fs::read_to_string(&extra) {
            pages.push((extra, s));
        }
    }
    pages.sort_by_key(|(n, _)| n.clone());

    println!("== HTML: kernel `browser::html` vs `tl` ==");
    println!(
        "{:<34} {:>8} {:>11} {:>11} {:>8}",
        "page", "KiB", "ours ms", "tl ms", "ratio"
    );
    let (mut sum_ours, mut sum_tl) = (0.0, 0.0);
    for (name, src) in &pages {
        let kib = src.len() as f64 / 1024.0;
        let iters = if src.len() > 200_000 { 20 } else { 100 };
        let ours = ms(bench(iters, || browser::html::parse(src)));
        let theirs = ms(bench(iters, || {
            tl::parse(src, tl::ParserOptions::default()).map(|d| d.nodes().len())
        }));
        sum_ours += ours;
        sum_tl += theirs;
        println!(
            "{:<34} {:>8.1} {:>11.3} {:>11.3} {:>7.2}x",
            short(name),
            kib,
            ours,
            theirs,
            ours / theirs.max(f64::MIN_POSITIVE)
        );
    }
    println!(
        "{:<34} {:>8} {:>11.3} {:>11.3} {:>7.2}x",
        "TOTAL", "", sum_ours, sum_tl, sum_ours / sum_tl.max(f64::MIN_POSITIVE)
    );

    // Speed is only half of it: a parser that is fast and builds the wrong tree
    // is not a candidate. These are the cases our own parser gets wrong today
    // (implied end tags), so they are exactly what a replacement has to fix to
    // be worth the port.
    println!("\n== HTML correctness: implied end tags ==");
    for (label, html) in [
        ("<p>one<p>two", "<p>one<p>two"),
        ("<ul><li>a<li>b</ul>", "<ul><li>a<li>b</ul>"),
        ("<table><tr><td>x", "<table><tr><td>x"),
    ] {
        let ours = depth_of_repeated(&browser_tree(html));
        let theirs = tl_depth(html);
        println!(
            "  {label:<24} ours: {ours:<28} tl: {theirs}",
        );
    }

    println!("\n== JS: kernel `just-engine` vs `boa` ==");
    println!(
        "{:<34} {:>8} {:>13} {:>13} {:>8}",
        "script", "KiB", "just ms", "boa ms", "ratio"
    );
    let mut scripts: Vec<(String, String)> = Vec::new();
    collect(&root, "js", &mut scripts);
    scripts.sort_by_key(|(n, _)| n.clone());
    for (name, src) in &scripts {
        let kib = src.len() as f64 / 1024.0;
        let iters = if src.len() > 100_000 { 5 } else { 30 };
        // Parse only: executing needs a DOM neither engine has here, and parse
        // is the phase that dominates a cold page load anyway (measured
        // in-kernel: 64.6 s of a 66 s script phase on a real site).
        let just = ms(bench(iters, || {
            just_engine::parser::JsParser::parse_to_ast_from_str(src).is_ok()
        }));
        let boa = ms(bench(iters, || boa_parse(src)));
        println!(
            "{:<34} {:>8.1} {:>13.3} {:>13.3} {:>7.2}x",
            short(name),
            kib,
            just,
            boa,
            just / boa.max(f64::MIN_POSITIVE)
        );
    }

    // The one that matters most: does the candidate accept code ours rejects?
    println!("\n== JS correctness: does each accept our real bundles? ==");
    for (name, src) in &scripts {
        let just_ok = just_engine::parser::JsParser::parse_to_ast_from_str(src).is_ok();
        let boa_ok = boa_parse(src);
        println!(
            "  {:<40} just: {:<6} boa: {}",
            short(name),
            if just_ok { "ok" } else { "FAIL" },
            if boa_ok { "ok" } else { "FAIL" }
        );
    }
}

/// Parse-only, to match what `JsParser::parse_to_ast_from_str` does on our
/// side. Executing would need a DOM neither engine has here.
fn boa_parse(src: &str) -> bool {
    let mut interner = boa_interner::Interner::default();
    boa_parser::Parser::new(boa_engine::Source::from_bytes(src.as_bytes()))
        .parse_script(&boa_ast::scope::Scope::new_global(), &mut interner)
        .is_ok()
}

fn short(p: &str) -> String {
    Path::new(p)
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| p.to_string())
}

fn collect(dir: &Path, ext: &str, out: &mut Vec<(String, String)>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        if p.extension().and_then(|s| s.to_str()) == Some(ext) {
            if let Ok(s) = std::fs::read_to_string(&p) {
                out.push((p.to_string_lossy().into_owned(), s));
            }
        }
    }
}

fn browser_tree(html: &str) -> browser::html::Document {
    browser::html::parse(html)
}

/// Describe the shape the parser produced for a repeated-tag case: the spec
/// wants siblings, a naive parser nests.
fn depth_of_repeated(doc: &browser::html::Document) -> String {
    fn walk(n: &browser::html::Node, depth: usize, best: &mut usize, count: &mut usize) {
        if let browser::html::NodeKind::Element { tag, .. } = &n.kind {
            if matches!(tag.as_str(), "p" | "li" | "td" | "tr" | "tbody") {
                *count += 1;
                *best = (*best).max(depth);
            }
        }
        for c in &n.children {
            walk(c, depth + 1, best, count);
        }
    }
    let (mut best, mut count) = (0, 0);
    walk(&doc.root, 0, &mut best, &mut count);
    format!("{count} elems, max depth {best}")
}

fn tl_depth(html: &str) -> String {
    let Ok(dom) = tl::parse(html, tl::ParserOptions::default()) else {
        return "parse error".into();
    };
    let p = dom.parser();
    let mut count = 0;
    let mut best = 0;
    fn walk(
        h: &tl::NodeHandle,
        p: &tl::Parser,
        depth: usize,
        best: &mut usize,
        count: &mut usize,
    ) {
        let Some(n) = h.get(p) else { return };
        if let Some(t) = n.as_tag() {
            let name = t.name().as_utf8_str();
            if matches!(&*name, "p" | "li" | "td" | "tr" | "tbody") {
                *count += 1;
                *best = (*best).max(depth);
            }
        }
        if let Some(kids) = n.children() {
            for c in kids.top().iter() {
                walk(c, p, depth + 1, best, count);
            }
        }
    }
    for h in dom.children() {
        walk(h, p, 0, &mut best, &mut count);
    }
    format!("{count} elems, max depth {best}")
}
