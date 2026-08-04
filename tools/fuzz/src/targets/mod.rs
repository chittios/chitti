//! Fuzz targets: one per kernel parser, each mounting the real kernel source
//! via `#[path]` (the `pngbench`/`onnxdiff`/`h264diff` pattern) so the bytes
//! exercised are the exact code that runs in the OS.
//!
//! A target is a function `&mut [u8] -> ()` that drives one parser far enough
//! to be interesting. The driver runs it under `catch_unwind`; a panic means a
//! hostile input crashes the kernel, and the input is saved for a regression
//! test.
//!
//! Adding a target:
//!   1. Pick a kernel module that is `no_std` + `alloc`-only (no `crate::`
//!      paths beyond `alloc`/`core`, no smoltcp/hardware). If the module you
//!      want isn't pure yet, extract the pure core into a mountable module
//!      first — that is a genuine confinement win, not just a fuzz convenience.
//!   2. Mount it here with `#[path = "../../../../kernel/src/…"] pub mod …;`
//!      and shim whatever it references (e.g. `Image` for `image/png.rs`).
//!   3. Add a `run_<name>` driving function below and an entry in `TARGETS`.

pub mod json;
pub mod png;
pub mod sha1;

/// Dispatch table: `fuzz run <name>`.
pub const TARGETS: &[&str] = &["json", "png", "sha1"];

pub fn run(name: &str, data: &[u8]) {
    match name {
        "json" => json::run(data),
        "png" => png::run(data),
        "sha1" => sha1::run(data),
        _ => panic!("unknown target {name:?} (try one of {TARGETS:?})"),
    }
}
