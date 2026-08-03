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

    // Apple WiFi dongle firmware (~2.5 MiB, from macOS via `cargo xtask
    // wifi-assets`). Embedded so bare m1n1 boots (no ESP disk) can still
    // `/wifi load`. Absent in a fresh clone → stub, no embed.
    println!("cargo:rustc-check-cfg=cfg(wifi_fw_embedded)");
    println!("cargo:rustc-check-cfg=cfg(wifi_nvram_embedded)");
    let wifi_fw = "../assets/wifi/brcm/brcmfmac4388-pcie.apple,miyake.bin";
    let wifi_nv = "../assets/wifi/brcm/brcmfmac4388-pcie.apple,miyake.txt";
    if std::path::Path::new(wifi_fw).exists() {
        println!("cargo:rustc-cfg=wifi_fw_embedded");
    }
    if std::path::Path::new(wifi_nv).exists() {
        println!("cargo:rustc-cfg=wifi_nvram_embedded");
    }
    println!("cargo:rerun-if-changed={wifi_fw}");
    println!("cargo:rerun-if-changed={wifi_nv}");

    embed_sample_files();
}

/// Generate the `/samples/` corpus table (`$OUT_DIR/samples.rs`) that
/// `kernel::samples` includes: one `include_bytes!` per file found under
/// `../assets/samples/`, so the booted OS has openable images/videos/audio/PDFs
/// with no network and no disk.
///
/// Two gates, both required, because the corpus is ~2 MiB of image size nobody
/// asked for by default:
///
/// - `CHITTI_SAMPLE_FILES` must be set to something affirmative (`make run` /
///   `make vbox` set it; empty reads as unset, since `make` passes the variable
///   through unconditionally), and
/// - the files must actually be on disk (`cargo xtask sample-files` fetches
///   them into the gitignored `assets/samples/`, so a fresh clone and CI have
///   none and build exactly as before).
///
/// The directory is **walked**, not listed from a table: xtask owns the corpus
/// definition and duplicating it here would be a second copy to drift. A
/// missing file is therefore not an error — it is simply not embedded.
fn embed_sample_files() {
    println!("cargo:rustc-check-cfg=cfg(samples_embedded)");
    println!("cargo:rerun-if-env-changed=CHITTI_SAMPLE_FILES");

    let root = std::path::Path::new("../assets/samples");
    println!("cargo:rerun-if-changed={}", root.display());

    let want = std::env::var("CHITTI_SAMPLE_FILES")
        .map(|v| {
            let v = v.trim().to_ascii_lowercase();
            !v.is_empty() && !matches!(v.as_str(), "0" | "no" | "off" | "false")
        })
        .unwrap_or(false);

    // `(category, filename, absolute path)`. Category "" = a file at the root of
    // the corpus (the generated README), which lands at `/samples/<name>`.
    let mut found: Vec<(String, String, std::path::PathBuf)> = Vec::new();
    if want && root.is_dir() {
        let mut cats: Vec<(String, std::path::PathBuf)> = Vec::new();
        // One level of categories; the corpus is deliberately flat.
        if let Ok(rd) = std::fs::read_dir(root) {
            for e in rd.flatten() {
                if e.path().is_dir() {
                    cats.push((e.file_name().to_string_lossy().to_string(), e.path()));
                }
            }
        }
        // Sorted — both the categories and the files inside them — so the
        // generated table, and therefore the built kernel, does not depend on
        // directory order.
        cats.sort();
        let mut dirs: Vec<(String, std::path::PathBuf)> = vec![(String::new(), root.to_path_buf())];
        dirs.extend(cats);
        for (cat, dir) in dirs {
            println!("cargo:rerun-if-changed={}", dir.display());
            let Ok(rd) = std::fs::read_dir(&dir) else { continue };
            let mut entries: Vec<std::path::PathBuf> =
                rd.flatten().map(|e| e.path()).filter(|p| p.is_file()).collect();
            // Sorted so the generated table (and thus the built kernel) is
            // reproducible regardless of directory order.
            entries.sort();
            for p in entries {
                let name = p.file_name().unwrap_or_default().to_string_lossy().to_string();
                if name.starts_with('.') {
                    continue;
                }
                let abs = std::fs::canonicalize(&p).unwrap_or(p.clone());
                println!("cargo:rerun-if-changed={}", p.display());
                found.push((cat.clone(), name, abs));
            }
        }
    }

    let out = std::path::Path::new(&std::env::var("OUT_DIR").unwrap()).join("samples.rs");
    let mut src = String::from(
        "// Generated by kernel/build.rs from ../assets/samples — do not edit.\n\
         pub static EMBEDDED: &[(&str, &str, &[u8])] = &[\n",
    );
    let mut bytes = 0u64;
    for (cat, name, path) in &found {
        bytes += std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        // The paths come from a directory walk of our own tree, not from user
        // input, but a quote in a filename would still break the generated
        // source — so refuse such a file rather than emit code that will not
        // compile with a confusing error.
        let p = path.to_string_lossy();
        if p.contains('"') || p.contains('\\') || name.contains('"') {
            println!("cargo:warning=samples: skipping {p} (quote/backslash in path)");
            continue;
        }
        src.push_str(&format!("    (\"{cat}\", \"{name}\", include_bytes!(\"{p}\")),\n"));
    }
    src.push_str("];\n");
    std::fs::write(&out, src).expect("writing samples.rs");

    if !found.is_empty() {
        println!("cargo:rustc-cfg=samples_embedded");
        println!("cargo:warning=samples: embedding {} file(s), {} KiB into the kernel image", found.len(), bytes / 1024);
    }
}
