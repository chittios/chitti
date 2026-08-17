//! Build `third_party/doomgeneric` twice: a native static library for the
//! baseline, and a wasm module for the interpreter measurement.
//!
//! The build lives here rather than in a makefile for the reason
//! `tools/pdfrender-wasm/.cargo/config.toml` pins its own flags: `-O3` and
//! `-msimd128` are each worth multiples (the PDF renderer measured `opt-level=3`
//! at **8.5x** over `s`, and wasm SIMD at a further **1.5-5.8x**), and a flag
//! passed by hand is a flag a rebuild can silently lose. A measurement that
//! quietly changed its own compiler settings would be worse than no measurement.
//!
//! Both sides compile the **same vendored sources with the same defines**, so the
//! ratio is a property of the interpreter rather than of two different programs.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Doom's native internal resolution. `doomgeneric` defaults to 640x400 (2x), but
/// the renderer's own working size is 320x200 and scaling up costs fill rate for
/// no detail — the game does not draw more at 640x400, it draws the same thing
/// bigger. The compositor already nearest-upscales a surface to the pane, so
/// upscaling here would just do it twice.
const RESX: u32 = 320;
const RESY: u32 = 200;

fn repo_root() -> PathBuf {
    // tools/doombench/ -> repo root
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("repo root")
        .to_path_buf()
}

/// Every vendored translation unit except the ones we replace or must not build.
fn doom_sources(dir: &Path) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = std::fs::read_dir(dir)
        .expect("doomgeneric source dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "c"))
        .filter(|p| {
            let n = p.file_name().unwrap().to_string_lossy().to_string();
            // Three substitutions, each replaced by a file in this crate rather
            // than by an edit upstream (VENDORING.md's rule):
            //   doomgeneric_*.c -- platform ports; ours is src/platform.c.
            //   i_main.c        -- owns `main()`, which would fight the harness
            //                      for the entry point.
            //   w_file_stdc.c   -- defines `stdc_wad_file` as host stdio; ours
            //                      defines the same symbol over a memory buffer,
            //                      because neither a wasm guest nor a ring-3
            //                      tenant has a file descriptor to give Doom.
            //
            // The `i_{allegro,sdl}{sound,music}.c` files are audio *backends* and
            // need their libraries' headers. They are dropped rather than stubbed
            // because `i_sound.c` is already written for their absence: its module
            // table is `#ifdef FEATURE_SOUND`, so with that undefined the list is
            // empty and Doom runs silent by a supported path rather than a hack.
            // The real port defines its own module against `audio_submit` (see the
            // plan's 1e) and will add it back the same way `w_file_memory.c` is
            // added -- as our file, not an edit here.
            const SUBSTITUTED: &[&str] = &["i_main.c", "w_file_stdc.c"];
            const BACKENDS: &[&str] = &[
                "i_allegromusic.c",
                "i_allegrosound.c",
                "i_sdlmusic.c",
                "i_sdlsound.c",
            ];
            !n.starts_with("doomgeneric_")
                && !SUBSTITUTED.contains(&n.as_str())
                && !BACKENDS.contains(&n.as_str())
        })
        .collect();
    // Deterministic order: `read_dir` is filesystem order, and a link line that
    // reshuffles between runs makes two builds hard to compare.
    out.sort();
    out
}

fn brew_prefix(pkg: &str) -> Option<PathBuf> {
    let out = Command::new("brew").arg("--prefix").arg(pkg).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let p = PathBuf::from(String::from_utf8_lossy(&out.stdout).trim().to_string());
    p.exists().then_some(p)
}

fn defines() -> Vec<String> {
    vec![
        // 8-bit paletted, which is Doom's native format. It also quarters the
        // per-frame copy across the wasm boundary: 64 KB rather than 256 KB at
        // 320x200.
        "-DCMAP256".into(),
        format!("-DDOOMGENERIC_RESX={RESX}"),
        format!("-DDOOMGENERIC_RESY={RESY}"),
        // Doom is 1993 C. These are not defects worth patching upstream sources
        // over -- see VENDORING.md's "no source file is modified" rule.
        "-Wno-everything".into(),
    ]
}

fn main() {
    let root = repo_root();
    let dg = root.join("third_party/doomgeneric/doomgeneric");
    if !dg.exists() {
        panic!("vendored doomgeneric not found at {}", dg.display());
    }
    println!("cargo:rerun-if-changed={}", dg.display());
    println!("cargo:rerun-if-changed=src/platform.c");
    println!("cargo:rerun-if-changed=src/w_file_memory.c");

    let srcs = doom_sources(&dg);
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    // ---- native baseline -------------------------------------------------
    // Plain `cc`: whatever the host toolchain is, since this side is only the
    // denominator and must simply be a fair native build.
    let mut objs = Vec::new();
    let ours = [PathBuf::from("src/platform.c"), PathBuf::from("src/w_file_memory.c")];
    for s in srcs.iter().chain(ours.iter()) {
        let o = out.join(format!(
            "native-{}.o",
            s.file_stem().unwrap().to_string_lossy()
        ));
        let st = Command::new("cc")
            .args(["-c", "-O3", "-fno-strict-aliasing"])
            .args(defines())
            .arg("-I")
            .arg(&dg)
            .arg("-o")
            .arg(&o)
            .arg(s)
            .status()
            .expect("run cc");
        assert!(st.success(), "native compile failed: {}", s.display());
        objs.push(o);
    }
    let lib = out.join("libdoomnative.a");
    let st = Command::new("ar")
        .arg("crs")
        .arg(&lib)
        .args(&objs)
        .status()
        .expect("run ar");
    assert!(st.success(), "ar failed");
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=doomnative");

    // ---- wasm module -----------------------------------------------------
    // Apple's clang has no wasm backend, so this needs homebrew llvm plus a
    // sysroot (`wasi-libc`) and a compiler-rt for the target (`wasi-runtimes`).
    // `wasi-runtimes` is a clang **resource dir**, not a sysroot: without
    // `-resource-dir` the link fails with `cannot open .../libclang_rt.builtins.a`
    // naming a path inside the *llvm* cellar, which reads like a broken llvm
    // rather than a missing package. Doom needs those builtins for 64-bit
    // division alone.
    let (Some(llvm), Some(sysroot), Some(resdir)) = (
        brew_prefix("llvm"),
        brew_prefix("wasi-libc"),
        brew_prefix("wasi-runtimes"),
    ) else {
        // A missing wasm toolchain must not fail the build: the native side still
        // answers "is the port correct", which is most of this harness's value.
        // Skipping loudly beats a build error nobody can act on.
        println!("cargo:warning=doombench: no wasm toolchain (need: brew install llvm wasi-libc wasi-runtimes) — wasm side disabled");
        println!("cargo:rustc-cfg=no_wasm_side");
        return;
    };

    let wasm = out.join("doom.wasm");
    let mut cmd = Command::new(llvm.join("bin/clang"));
    cmd.arg("--target=wasm32-wasip1")
        .arg(format!("--sysroot={}", sysroot.join("share/wasi-sysroot").display()))
        .arg(format!("-resource-dir={}", resdir.join("share/wasi-runtimes").display()))
        // Both are load-bearing and measured; see the module doc.
        .args(["-O3", "-msimd128", "-fno-strict-aliasing"])
        .args(defines())
        .arg("-I")
        .arg(&dg)
        // No `main`: the harness drives `dg_*` exports itself.
        .args(["-nostartfiles", "-Wl,--no-entry", "-Wl,--export-dynamic"])
        // Doom's zone allocator wants one big block up front; the default 1 MB
        // stack is also not enough for R_RenderBSPNode's recursion.
        .args(["-Wl,-z,stack-size=1048576", "-Wl,--initial-memory=134217728"])
        .arg("-o")
        .arg(&wasm)
        .args(&srcs)
        .args(&ours);
    let st = cmd.status().expect("run clang for wasm");
    assert!(st.success(), "wasm compile failed");
    println!("cargo:rustc-env=DOOM_WASM={}", wasm.display());
    println!("cargo:rustc-env=DOOM_RESX={RESX}");
    println!("cargo:rustc-env=DOOM_RESY={RESY}");
}
