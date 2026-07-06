//! Build-time metadata: the version and build timestamp shown in the status
//! bar, banner, and `/info`. CI (the release workflow) injects the release tag
//! and time via `CHITTI_VERSION` / `CHITTI_BUILD_TIME`; local builds fall back
//! to the crate version + a stable "dev" stamp so incremental builds stay
//! reproducible (no relink on every build just to bump a timestamp).

fn main() {
    let version = std::env::var("CHITTI_VERSION")
        .ok()
        .filter(|v| !v.is_empty())
        .map(|v| v.trim_start_matches('v').to_string())
        .unwrap_or_else(|| std::env::var("CARGO_PKG_VERSION").unwrap());
    let build_time = std::env::var("CHITTI_BUILD_TIME").ok().filter(|v| !v.is_empty()).unwrap_or_else(|| "dev".into());
    println!("cargo:rustc-env=CHITTI_VERSION={version}");
    println!("cargo:rustc-env=CHITTI_BUILD_TIME={build_time}");
    println!("cargo:rerun-if-env-changed=CHITTI_VERSION");
    println!("cargo:rerun-if-env-changed=CHITTI_BUILD_TIME");

    // The silero-vad model (630 KB) is `include_bytes!`'d into the kernel for the
    // in-kernel VAD. It's gitignored (`assets/voice/`, fetched by
    // `cargo xtask voice-assets`), so it's absent in a fresh clone / CI. Gate the
    // embed on its presence: `voice_vad_embedded` is set only when the file
    // exists, and the code falls back to a no-VAD stub otherwise, so the kernel
    // (and the unit suite) build without the voice assets. Present → embedded and
    // the silero parser/numeric tests run, exactly as before.
    println!("cargo:rustc-check-cfg=cfg(voice_vad_embedded)");
    // build.rs runs with CWD = the crate dir (kernel/); the asset is one up.
    if std::path::Path::new("../assets/voice/silero_vad.onnx").exists() {
        println!("cargo:rustc-cfg=voice_vad_embedded");
    }
    println!("cargo:rerun-if-changed=../assets/voice/silero_vad.onnx");
}
