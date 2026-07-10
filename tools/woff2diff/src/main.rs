//! Validate the kernel WOFF2 decoder against a fonttools reference.
//!
//! Decodes `geist-ascii.woff2` with the kernel's own `font_woff2` (mounted via
//! `#[path]`), then rasterizes a set of glyphs with both the reconstructed SFNT
//! and the fonttools-decompressed reference TTF and asserts the bitmaps match.
extern crate alloc;

#[path = "../../../kernel/src/font_woff2.rs"]
mod font_woff2;

use std::fs;

fn raster(font_bytes: &[u8], ch: char, px: f32) -> (fontdue::Metrics, Vec<u8>) {
    let font = fontdue::Font::from_bytes(font_bytes, fontdue::FontSettings::default())
        .expect("fontdue parse");
    font.rasterize(ch, px)
}

fn main() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../webcompat/fixtures/woff2");
    let woff2 = fs::read(format!("{dir}/geist-ascii.woff2")).expect("read woff2");
    let reference = fs::read(format!("{dir}/geist-ascii.decompressed.ttf")).expect("read ref");

    let sfnt = font_woff2::woff2_to_sfnt(&woff2).expect("woff2 decode");
    println!("decoded SFNT: {} bytes (reference {} bytes)", sfnt.len(), reference.len());

    // The reconstructed SFNT must itself parse in fontdue.
    let _ = fontdue::Font::from_bytes(sfnt.as_slice(), fontdue::FontSettings::default())
        .expect("reconstructed SFNT parses in fontdue");

    // Rasterize a spread of ASCII glyphs with both fonts; bitmaps must match.
    let px = 40.0;
    let mut checked = 0;
    let mut mismatches = 0;
    for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%&*()".chars() {
        let (m1, b1) = raster(&sfnt, ch, px);
        let (m2, b2) = raster(&reference, ch, px);
        checked += 1;
        if (m1.width, m1.height) != (m2.width, m2.height) || b1 != b2 {
            mismatches += 1;
            if mismatches <= 5 {
                println!(
                    "MISMATCH {ch:?}: mine {}x{} vs ref {}x{}",
                    m1.width, m1.height, m2.width, m2.height
                );
            }
        }
    }
    println!("checked {checked} glyphs, {mismatches} mismatches");
    if mismatches == 0 {
        println!("PASS: WOFF2 reconstruction bit-identical to fonttools reference");
    } else {
        std::process::exit(1);
    }
}
