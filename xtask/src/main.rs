//! Build orchestration for Chitti OS: assembles a bootable Limine image
//! from the kernel and drives QEMU. All project commands go through
//! `cargo xtask <cmd>` (see CHITTI_OS_HANDOFF.md Part 7).

use std::env;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Target architecture, chosen explicitly via `-arch x86_64|aarch64` (never
/// auto-detected from the host): the same unified kernel builds for both.
#[derive(Clone, Copy, PartialEq)]
enum Arch {
    X86_64,
    Aarch64,
}

/// Which bundled model to build/run for, chosen via `-model <name>` (like
/// `-arch`). Selects the kernel memory-layout feature, the GGUF file, and the
/// aarch64 load address + guest RAM. Default is the 0.8B.
#[derive(Clone, Copy, PartialEq)]
enum Model {
    Qwen08B,
    Qwen2B,
    Qwen4B,
    Qwen9B,
}

impl Model {
    /// Cargo features that select this model's heap-size tier in the kernel.
    fn features(self) -> &'static [&'static str] {
        match self {
            Model::Qwen08B => &[],
            Model::Qwen2B => &["model-2b"],
            Model::Qwen4B => &["model-4b"],
            Model::Qwen9B => &["model-9b"],
        }
    }
    /// The GGUF file bundled for this model (relative to the repo root).
    fn gguf_rel(self) -> &'static str {
        match self {
            Model::Qwen08B => "assets/model.gguf",
            Model::Qwen2B => "assets/model-2b.gguf",
            Model::Qwen4B => "assets/model-4b.gguf",
            Model::Qwen9B => "assets/model-9b.gguf",
        }
    }
    /// aarch64 guest-physical load address — one address for every model
    /// (matches `cortex::MODEL_LOAD_ADDR`). The kernel places its heap at the top
    /// of discovered RAM, so the model region `[addr, heap)` sizes itself to `-m`.
    fn aarch64_addr(self) -> &'static str {
        "0x80000000"
    }
    /// QEMU `-m` size: must hold the model (loaded at 0x80000000 = 2 GiB) plus
    /// the heap the kernel places at the top of RAM. The kernel errors clearly if
    /// a model won't fit, so these are simply comfortable sizes per model.
    fn qemu_mem(self) -> &'static str {
        match self {
            // 0.8B (~785 MiB) at 2 GiB + a 512 MiB heap at the top: 3 GiB is ample.
            Model::Qwen08B => "3G",
            // ~1.2 GiB model at 2 GiB + a 512 MiB heap at the top of RAM.
            Model::Qwen2B => "4G",
            // ~2.58 GiB model at 2 GiB + a 512 MiB heap at the top of RAM.
            Model::Qwen4B => "6G",
            Model::Qwen9B => "10G",
        }
    }
    fn label(self) -> &'static str {
        match self {
            Model::Qwen08B => "qwen3.5-0.8b",
            Model::Qwen2B => "qwen3.5-2b",
            Model::Qwen4B => "qwen3.5-4b",
            Model::Qwen9B => "qwen3.5-9b",
        }
    }
}

/// Convert a QEMU `-m` size string (`"3G"`, `"512M"`, or a raw byte count) to
/// bytes, for the `opt/chitti/ramsize` fw_cfg the kernel reads.
fn mem_bytes(m: &str) -> u64 {
    let m = m.trim();
    let (num, mult) = match m.chars().last() {
        Some('G') | Some('g') => (&m[..m.len() - 1], 1u64 << 30),
        Some('M') | Some('m') => (&m[..m.len() - 1], 1u64 << 20),
        Some('K') | Some('k') => (&m[..m.len() - 1], 1u64 << 10),
        _ => (m, 1),
    };
    num.parse::<u64>().unwrap_or(4).saturating_mul(mult)
}

/// Parse `-model <value>` (or `-model=<value>`); default the 0.8B model.
fn parse_model(rest: &[String]) -> Result<Model, String> {
    let mut it = rest.iter();
    while let Some(a) = it.next() {
        let val = if let Some(v) = a.strip_prefix("-model=") {
            Some(v.to_string())
        } else if a == "-model" || a == "--model" {
            it.next().cloned()
        } else {
            None
        };
        if let Some(v) = val {
            return match v.as_str() {
                "qwen3.5-0.8b" | "qwen3.5-0.8B" | "0.8b" | "0.8B" | "qwen0.8b" | "default" => Ok(Model::Qwen08B),
                "qwen3.5-2b" | "qwen3.5-2B" | "2b" | "2B" | "qwen2b" => Ok(Model::Qwen2B),
                "qwen3.5-4b" | "qwen3.5-4B" | "4b" | "4B" | "qwen4b" => Ok(Model::Qwen4B),
                "qwen3.5-9b" | "qwen3.5-9B" | "9b" | "9B" | "qwen9b" => Ok(Model::Qwen9B),
                other => Err(format!("unknown -model '{other}' (expected qwen3.5-0.8b, qwen3.5-2b, qwen3.5-4b, or qwen3.5-9b)")),
            };
        }
    }
    Ok(Model::Qwen08B)
}

/// Parse `-arch <value>` (or `-arch=<value>`) from the args; default x86_64.
fn parse_arch(rest: &[String]) -> Result<Arch, String> {
    let mut it = rest.iter();
    while let Some(a) = it.next() {
        let val = if let Some(v) = a.strip_prefix("-arch=") {
            Some(v.to_string())
        } else if a == "-arch" || a == "--arch" {
            it.next().cloned()
        } else {
            None
        };
        if let Some(v) = val {
            return match v.as_str() {
                "x86_64" | "x86-64" | "x64" => Ok(Arch::X86_64),
                "aarch64" | "arm64" => Ok(Arch::Aarch64),
                other => Err(format!("unknown -arch '{other}' (expected x86_64 or aarch64)")),
            };
        }
    }
    Ok(Arch::X86_64)
}

fn main() {
    let mut args = env::args().skip(1);
    let cmd = args.next().unwrap_or_default();
    let rest: Vec<String> = args.collect();
    let release = rest.iter().any(|a| a == "--release");
    // `--uefi`: boot the hybrid ISO through OVMF (UEFI firmware) instead of the
    // default SeaBIOS, exercising the same UEFI/GOP framebuffer path a real
    // machine uses (the BIOS path uses Limine's VBE setup instead).
    let uefi = rest.iter().any(|a| a == "--uefi");
    // `--disk-only`: boot the installed disk via OVMF with NO ISO attached — the
    // real "boot from disk" path after `/install` (implies UEFI). `--disk <SIZE>`
    // (e.g. `2G`, `1500M`) sizes target/chitti-disk.img so it can hold the ESP +
    // model + data partitions; `--fresh-disk` wipes it first.
    let disk_only = rest.iter().any(|a| a == "--disk-only");
    let fresh_disk = rest.iter().any(|a| a == "--fresh-disk");
    let disk_size = flag_value(&rest, "--disk");
    // `--no-model`: build the ISO with no model module, so `/install` writes only
    // the kernel + config + an empty data partition (fast) — for exercising the
    // install & boot-from-disk flow without the slow ~800 MiB model write.
    let no_model = rest.iter().any(|a| a == "--no-model");
    let arch = match parse_arch(&rest) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("xtask error: {e}");
            std::process::exit(1);
        }
    };
    let model = match parse_model(&rest) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("xtask error: {e}");
            std::process::exit(1);
        }
    };

    let result = match cmd.as_str() {
        "build" => cmd_build(release, arch, model),
        // x86: the classic hybrid BIOS/UEFI ISO. aarch64: a raw GPT disk image
        // (the ARM-world convention — dd/Etcher it to a USB drive, or attach it
        // in UTM/QEMU/VirtualBox-ARM; it boots standalone via the UEFI stub).
        "image" => match arch {
            Arch::Aarch64 => image_aarch64(model),
            Arch::X86_64 => image(release, model),
        },
        "run" => cmd_run(release, arch, model, uefi, disk_only, fresh_disk, disk_size, no_model),
        "test" => cmd_test(),
        "voice-assets" => cmd_voice_assets(),
        // Phase 3 parity gate: build the kernel with the `refcheck` feature,
        // boot the real model, run the acceptance checks, exit pass/fail.
        "ref-check" => cmd_ref_check(),
        // Hidden subcommand: installed as `[target.x86_64-chitti] runner` in
        // kernel/.cargo/config.toml so `cargo test` can boot each compiled
        // test binary in QEMU and translate isa-debug-exit into a real exit
        // code.
        "runner" => cmd_runner(&rest),
        _ => Err(usage()),
    };

    if let Err(e) = result {
        eprintln!("xtask error: {e}");
        std::process::exit(1);
    }
}

fn usage() -> String {
    "usage: cargo xtask <build|image|run|test|ref-check> [-arch x86_64|aarch64] \
     [-model qwen3.5-0.8b|qwen3.5-2b|qwen3.5-4b|qwen3.5-9b] [--release] [--uefi]\n\
     run flags (x86_64): --disk <2G|1500M> size the virtio-blk disk for /install; \
     --disk-only boot the installed disk via UEFI with no ISO; --fresh-disk wipe it first; \
     --no-model install without the model (fast, skips the ~800 MiB write).\n\
     install+boot test:  cargo xtask run --uefi --disk 2G [--no-model]  (type `/install yes`, then quit)\n\
                         cargo xtask run --disk-only                    (boots Chitti from the disk alone)"
        .to_string()
}

/// `cargo xtask build [-arch ...] [-model ...]`: build the unified kernel for
/// the chosen architecture and model memory layout.
fn cmd_build(release: bool, arch: Arch, model: Model) -> Result<(), String> {
    match arch {
        Arch::X86_64 => build_kernel_with(release, model.features()).map(|_| ()),
        Arch::Aarch64 => build_kernel_aarch64(release, model.features()).map(|_| ()),
    }
}

/// Build the unified kernel for aarch64 (`targets/aarch64-chitti.json`) with
/// the given extra cargo `features`, returning the path to the resulting ELF
/// (`-M virt -kernel` bootable).
fn build_kernel_aarch64(release: bool, features: &[&str]) -> Result<PathBuf, String> {
    let kdir = kernel_dir();
    let mut cmd = Command::new("cargo");
    cmd.current_dir(&kdir).args(["build", "--target", "../targets/aarch64-chitti.json"]);
    if release {
        cmd.arg("--release");
    }
    if !features.is_empty() {
        cmd.arg("--features").arg(features.join(","));
    }
    run(&mut cmd)?;
    let profile = if release { "release" } else { "debug" };
    let elf = kdir.join(format!("target/aarch64-chitti/{profile}/chitti-kernel"));
    if !elf.exists() {
        return Err(format!("aarch64 kernel not found at {}", elf.display()));
    }
    Ok(elf)
}

/// `cargo xtask arm64`: build the standalone aarch64 kernel and boot it on
/// `qemu-system-aarch64 -M virt` with `-accel hvf`, so it runs *natively* on
/// Boot the unified kernel built for aarch64 on `qemu-system-aarch64 -M virt`
/// with `-accel hvf`, so it runs *natively* on this Apple Silicon host (no
/// cross-arch emulation) with NEON. Serial to stdio (`-nographic`).
/// Build the Chitti UEFI stub bootloader (`stub/`) for aarch64 — the
/// BOOTAA64.EFI that AAVMF launches from the ESP. It loads the normal identity
/// (`-kernel`) Chitti ELF + model off the ESP and hands off MMU-off via an
/// identity-RAM trampoline, so the kernel boots exactly as under `-kernel`.
fn build_stub_aarch64() -> Result<PathBuf, String> {
    let sdir = repo_root().join("stub");
    let mut cmd = Command::new("cargo");
    cmd.current_dir(&sdir).args(["build", "--release", "--target", "aarch64-unknown-uefi"]);
    run(&mut cmd)?;
    let efi = sdir.join("target/aarch64-unknown-uefi/release/chitti-stub.efi");
    if !efi.exists() {
        return Err(format!("stub not found at {}", efi.display()));
    }
    Ok(efi)
}

/// The framebuffer resolution to hand the guest ramfb, as a `-fw_cfg` argument
/// (the kernel reads `opt/chitti/fbres`). Not a baked constant: honours
/// `$CHITTI_FB_RES=WxH`, else auto-detects the host display, else a safe
/// default. Real hardware / the UEFI path takes its resolution from GOP instead,
/// so this only shapes the QEMU ramfb window.
fn ramfb_res_fw_cfg() -> Vec<String> {
    let (w, h) = std::env::var("CHITTI_FB_RES")
        .ok()
        .and_then(|s| parse_wxh(&s))
        .or_else(detect_host_res)
        .unwrap_or((1600, 1000));
    eprintln!("  framebuffer: {w}x{h} (set CHITTI_FB_RES=WxH to override)");
    vec!["-fw_cfg".into(), format!("name=opt/chitti/fbres,string={w}x{h}")]
}

/// Parse a `WIDTHxHEIGHT` string.
fn parse_wxh(s: &str) -> Option<(u32, u32)> {
    let (a, b) = s.trim().split_once(['x', 'X'])?;
    Some((a.trim().parse().ok()?, b.trim().parse().ok()?))
}

/// Best-effort host main-display resolution on macOS via `system_profiler`.
/// Retina reports are physical pixels, so halve them to a logical size that
/// makes a window roughly filling the screen; leave headroom for chrome.
fn detect_host_res() -> Option<(u32, u32)> {
    let out = Command::new("system_profiler").arg("SPDisplaysDataType").output().ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    for line in text.lines() {
        let Some(rest) = line.trim().strip_prefix("Resolution:") else { continue };
        // e.g. "3456 x 2234 Retina" or "2560 x 1440"
        let parts: Vec<&str> = rest.split_whitespace().collect();
        if parts.len() >= 3 && parts[1] == "x" {
            if let (Ok(mut w), Ok(mut h)) = (parts[0].parse::<u32>(), parts[2].parse::<u32>()) {
                if rest.contains("Retina") {
                    w /= 2;
                    h /= 2;
                }
                return Some((w.max(1024), h.saturating_sub(80).max(720)));
            }
        }
    }
    None
}

/// Guard against artifact mixups: assert `elf` is the identity-map `-kernel`
/// build (entry in low RAM), not a higher-half build sharing the same path.
fn assert_identity_kernel(elf: &Path) -> Result<(), String> {
    let bytes = fs::read(elf).map_err(|e| format!("reading {}: {e}", elf.display()))?;
    let entry = u64::from_le_bytes(bytes[24..32].try_into().unwrap());
    if entry >= 1 << 32 {
        return Err(format!("{} has entry {entry:#x} (higher-half build?) — expected the identity -kernel build", elf.display()));
    }
    Ok(())
}

/// QEMU pflash args for AAVMF (aarch64 UEFI firmware) — the aarch64 analogue of
/// `ovmf_pflash_args`. A per-run writable copy of the vars volume lives under
/// `target/`.
fn aavmf_pflash_args() -> Result<Vec<String>, String> {
    let share = brew_prefix("qemu")?.join("share/qemu");
    let code = share.join("edk2-aarch64-code.fd");
    let vars_src = share.join("edk2-arm-vars.fd");
    if !code.exists() || !vars_src.exists() {
        return Err(format!("AAVMF firmware not found under {} (need edk2-aarch64-code.fd + edk2-arm-vars.fd)", share.display()));
    }
    let vars = repo_root().join("target/aavmf-vars.fd");
    fs::copy(&vars_src, &vars).map_err(|e| format!("copying AAVMF vars: {e}"))?;
    Ok(vec![
        "-drive".into(),
        format!("if=pflash,format=raw,readonly=on,file={}", code.display()),
        "-drive".into(),
        format!("if=pflash,format=raw,file={}", vars.display()),
    ])
}

/// Build a **real FAT32 ESP image** carrying the stub (as BOOTAA64.EFI), the
/// identity kernel, and (optionally) the model, using macOS hdiutil/newfs_msdos.
/// A real image — not VVFAT (sparse-sector stalls, ~504 MiB cap) and not a
/// `-cdrom` El Torito volume — is what boots reliably under AAVMF, and FAT32
/// carries the ~774 MiB model file directly.
fn build_esp_image_aarch64(stub: &Path, kernel: &Path, model: Option<&Path>) -> Result<PathBuf, String> {
    let img = repo_root().join("target/chitti-esp-aa64.img");
    let model_bytes = model.map(|m| fs::metadata(m).map(|md| md.len()).unwrap_or(0)).unwrap_or(0);
    // Voice models bundled onto the ESP too (kernel mounts the FAT ESP and reads
    // them). Grow the image to fit them.
    let voice: Vec<(String, PathBuf)> = voice_model_assets().into_iter().filter(|(_, p)| p.exists()).map(|(n, p)| (n.to_string(), p)).collect();
    let voice_bytes: u64 = voice.iter().map(|(_, p)| fs::metadata(p).map(|m| m.len()).unwrap_or(0)).sum();
    let size_mb = 64 + ((model_bytes + voice_bytes) / (1024 * 1024)) + if model_bytes > 0 { 64 } else { 0 };
    // Recreate the image only when contents changed (cheap heuristic: sizes).
    let f = fs::OpenOptions::new().create(true).write(true).truncate(true).open(&img).map_err(|e| e.to_string())?;
    f.set_len(size_mb * 1024 * 1024).map_err(|e| e.to_string())?;
    drop(f);
    // Attach raw, format FAT32, mount, copy, detach — scripted via /bin/sh.
    let script = format!(
        r#"set -e
DEV=$(hdiutil attach -nomount -imagekey diskimage-class=CRawDiskImage "{img}" | head -1 | awk '{{print $1}}')
newfs_msdos -F 32 -v CHITTI "$DEV" > /dev/null
diskutil mount "$DEV" > /dev/null
MNT=$(diskutil info "$DEV" | awk -F': *' '/Mount Point/{{print $2}}')
mkdir -p "$MNT/EFI/BOOT"
cp "{stub}" "$MNT/EFI/BOOT/BOOTAA64.EFI"
cp "{kernel}" "$MNT/chitti-kernel"
{model_cp}
{voice_cp}
diskutil unmount "$DEV" > /dev/null
hdiutil detach "$DEV" > /dev/null
"#,
        img = img.display(),
        stub = stub.display(),
        kernel = kernel.display(),
        model_cp = model.map(|m| format!("cp \"{}\" \"$MNT/model.gguf.000\"", m.display())).unwrap_or_default(),
        // Voice models at the ESP root, so the kernel's root-file readers find them.
        voice_cp = voice.iter().map(|(n, p)| format!("cp \"{}\" \"$MNT/{}\"", p.display(), n)).collect::<Vec<_>>().join("\n"),
    );
    let status = Command::new("/bin/sh").arg("-c").arg(&script).status().map_err(|e| format!("building ESP image: {e}"))?;
    if !status.success() {
        return Err("ESP image build failed (hdiutil/newfs_msdos)".into());
    }
    eprintln!("  ESP image: {} ({} MiB{})", img.display(), size_mb, if model.is_some() { ", model bundled" } else { "" });
    Ok(img)
}

/// Build a small FAT "voice disk" holding the voice models (`kitten.onnx`,
/// `parakeet.onnx`) so `cargo xtask run` can attach them as an extra virtio-blk
/// disk — the kernel's `find_on_disks` scans it and auto-loads them. Returns
/// `None` if no voice assets are present. Cached; rebuilt when the models change.
fn build_voice_disk() -> Option<PathBuf> {
    let voice: Vec<(String, PathBuf)> = voice_model_assets().into_iter().filter(|(_, p)| p.exists()).map(|(n, p)| (n.to_string(), p)).collect();
    if voice.is_empty() {
        return None;
    }
    let img = repo_root().join("target/chitti-voice.img");
    let total: u64 = voice.iter().map(|(_, p)| fs::metadata(p).map(|m| m.len()).unwrap_or(0)).sum();
    let want = format!("{total}:{}", voice.len());
    let meta = repo_root().join("target/chitti-voice.meta");
    if img.exists() && fs::read_to_string(&meta).map(|s| s.trim() == want).unwrap_or(false) {
        return Some(img);
    }
    let size_mb = 64 + total / (1024 * 1024) + 32;
    let f = fs::OpenOptions::new().create(true).write(true).truncate(true).open(&img).ok()?;
    f.set_len(size_mb * 1024 * 1024).ok()?;
    drop(f);
    let cps = voice.iter().map(|(n, p)| format!("cp \"{}\" \"$MNT/{}\"", p.display(), n)).collect::<Vec<_>>().join("\n");
    let script = format!(
        r#"set -e
DEV=$(hdiutil attach -nomount -imagekey diskimage-class=CRawDiskImage "{img}" | head -1 | awk '{{print $1}}')
newfs_msdos -F 32 -v VOICE "$DEV" > /dev/null
diskutil mount "$DEV" > /dev/null
MNT=$(diskutil info "$DEV" | awk -F': *' '/Mount Point/{{print $2}}')
{cps}
diskutil unmount "$DEV" > /dev/null
hdiutil detach "$DEV" > /dev/null
"#,
        img = img.display(),
    );
    let ok = Command::new("/bin/sh").arg("-c").arg(&script).status().map(|s| s.success()).unwrap_or(false);
    if !ok {
        return None;
    }
    let _ = fs::write(&meta, &want);
    eprintln!("  voice disk: {} ({} MiB, {} model(s))", img.display(), size_mb, voice.len());
    Some(img)
}

/// Boot aarch64 via UEFI firmware (AAVMF) — the Chitti stub loads the normal
/// identity kernel (+ model) off a real FAT ESP and hands off MMU-off through
/// an identity-RAM trampoline, so the kernel boots exactly as under `-kernel`.
/// The data disk is attached first so the in-kernel `probe_disk` (first
/// virtio-mmio slot) targets it for /install + persistence, not the ESP.
fn cmd_run_aarch64_uefi(model: Model, disk: Option<PathBuf>, disk_only: bool, no_model: bool) -> Result<(), String> {
    let elf = build_kernel_aarch64(true, model.features())?;
    assert_identity_kernel(&elf)?;
    let stub = build_stub_aarch64()?;
    let mut qemu = Command::new("qemu-system-aarch64");
    qemu.args(["-M", "virt", "-cpu", "host", "-accel", "hvf", "-smp", "4", "-m", model.qemu_mem()]);
    for a in aavmf_pflash_args()? {
        qemu.arg(a);
    }
    qemu.args([
        "-device", "ramfb", "-device", "virtio-keyboard-device", "-device", "virtio-tablet-device",
        "-display", "cocoa,zoom-to-fit=on", "-serial", "mon:stdio",
    ]);
    // ESP first, data disk LAST: QEMU assigns later virtio-mmio devices to
    // LOWER slots, and the kernel's probe_disk takes the first (lowest) match —
    // so this ordering makes /install + persistence target the data disk, never
    // the boot ESP.
    if disk_only {
        eprintln!("booting aarch64 FROM DISK ONLY via UEFI (no ESP medium)");
        eprintln!("  note: requires an /install that wrote the aarch64 ESP payload to the disk");
    } else {
        let gguf = repo_root().join(model.gguf_rel());
        let model_path = (!no_model && gguf.exists()).then_some(gguf);
        let esp = build_esp_image_aarch64(&stub, &elf, model_path.as_deref())?;
        qemu.arg("-drive").arg(format!("file={},if=none,id=esp,format=raw", esp.display()));
        qemu.args(["-device", "virtio-blk-device,drive=esp"]);
        eprintln!("booting aarch64 via the Chitti UEFI stub (AAVMF) -- firmware loads BOOTAA64.EFI from the ESP");
    }
    if let Some(d) = &disk {
        qemu.arg("-drive").arg(format!("file={},if=none,id=data,format=raw", d.display()));
        qemu.args(["-device", "virtio-blk-device,drive=data"]);
        eprintln!("  data disk: {}", d.display());
    }
    run(&mut qemu)
}

fn cmd_run_aarch64(release: bool, model: Model, disk: Option<PathBuf>, _disk_only: bool) -> Result<(), String> {
    // Native inference on aarch64 is only worthwhile optimized: debug NEON is
    // ~30x slower (no inlining of intrinsics, bounds/overflow checks in the hot
    // matvec loop). So this path defaults to a release build regardless of the
    // `--release` flag; the whole point of `-arch aarch64` is native speed.
    if !release {
        eprintln!("note: building aarch64 in RELEASE (debug NEON inference is ~30x slower)");
    }
    let elf = build_kernel_aarch64(true, model.features())?;
    let mut qemu = Command::new("qemu-system-aarch64");
    // Guest RAM holds the kernel + the model (loaded at `model.aarch64_addr`) +
    // the heap; `model.qemu_mem` is sized for the chosen model's layout (2G for
    // 0.8B, 12G for 9B). `-smp 4`: four vCPUs, which under `-accel hvf` run on
    // four *native* M-series cores in parallel (unlike TCG, where extra vCPUs
    // only contend). Chitti's aarch64 SMP brings the secondaries up via PSCI
    // and splits the hot matvec across them.
    // `-device ramfb`: a simple linear framebuffer the kernel configures via
    // fw_cfg (arch::aarch64::ramfb) and renders the TUI into — the aarch64
    // equivalent of the x86 Limine framebuffer. Dropping `-nographic` lets QEMU
    // open its display window; `-serial mon:stdio` keeps the serial console (and
    // QEMU monitor) on stdio so you still type at the terminal (Ctrl-A X quits,
    // Ctrl-A C for the monitor).
    qemu.args([
        "-M", "virt", "-cpu", "host", "-accel", "hvf", "-smp", "4", "-m", model.qemu_mem(),
        "-device", "ramfb", "-device", "virtio-keyboard-device",
        // A virtio tablet gives the window an absolute-position mouse.
        "-device", "virtio-tablet-device",
        // Resizable graphical window (the ramfb surface scales to fit).
        "-display", "cocoa,zoom-to-fit=on",
        "-serial", "mon:stdio", "-kernel",
    ]);
    qemu.arg(&elf);
    // Hand the guest ramfb the framebuffer resolution to use (the kernel reads
    // opt/chitti/fbres) — derived from the host display, not hardcoded.
    for a in ramfb_res_fw_cfg() {
        qemu.arg(a);
    }
    // Publish the guest RAM size so the kernel places its heap at the top of RAM.
    // On the `-kernel` HVF path QEMU passes no DTB (x0=0), so this fw_cfg file is
    // how the kernel learns how much RAM it has (see `mmu::detect`).
    qemu.arg("-fw_cfg").arg(format!("name=opt/chitti/ramsize,string={}", mem_bytes(model.qemu_mem())));
    // A virtio-net NIC on user-mode networking (built-in DHCP 10.0.2.15 / gw
    // 10.0.2.2 / dns 10.0.2.3) so /network, /ping and /wifi work out of the box.
    qemu.args(["-netdev", "user,id=chittinet", "-device", "virtio-net-device,netdev=chittinet"]);
    // virtio-snd on the host's CoreAudio (mic + speaker) for /voice.
    qemu.args(["-audiodev", "coreaudio,id=chittiaudio", "-device", "virtio-sound-device,audiodev=chittiaudio"]);
    // Attach a virtio-blk disk on the virtio-mmio bus (the aarch64 block driver
    // scans that window) so /disks, /mkext4, /install, and synapse persistence
    // work — the aarch64 counterpart to the x86 virtio-blk-pci drive.
    if let Some(d) = &disk {
        qemu.arg("-drive").arg(format!("file={},if=none,id=chittidisk,format=raw", d.display()));
        qemu.args(["-device", "virtio-blk-device,drive=chittidisk"]);
        eprintln!("  disk: {} (virtio-blk over virtio-mmio)", d.display());
    }
    // Voice models on an extra FAT disk (kernel `find_on_disks` auto-loads them).
    if let Some(vd) = build_voice_disk() {
        qemu.arg("-drive").arg(format!("file={},if=none,id=voicedisk,format=raw", vd.display()));
        qemu.args(["-device", "virtio-blk-device,drive=voicedisk"]);
    }
    // Place the GGUF in guest RAM at the model's load address (where the aarch64
    // `cortex::model_module` looks) -- the equivalent of the x86 Limine boot
    // module, so `infer` works natively.
    let gguf = repo_root().join(model.gguf_rel());
    if gguf.exists() {
        let base = u64::from_str_radix(model.aarch64_addr().trim_start_matches("0x"), 16)
            .map_err(|e| format!("bad model addr {}: {e}", model.aarch64_addr()))?;
        for arg in model_loader_args(&gguf, base)? {
            qemu.arg("-device").arg(arg);
        }
        eprintln!("attaching {} at guest phys {}", model.gguf_rel(), model.aarch64_addr());
    } else {
        eprintln!("note: {} absent -- `infer` will report no model", model.gguf_rel());
    }
    eprintln!("booting aarch64 Chitti ({}) natively via HVF (Ctrl-A X to quit qemu)...", model.label());
    run(&mut qemu)
}

/// Build the QEMU `-device loader` argument(s) that place `gguf` in guest RAM
/// starting at `base_addr`. QEMU's generic loader reads the whole file in one
/// `read(2)`, which on macOS fails with `EINVAL` for images >= 2 GiB (Darwin
/// caps a single read at `INT_MAX`). So any model >= 2 GiB is split into
/// <= 1 GiB chunk files loaded at consecutive addresses -- the guest sees one
/// contiguous blob at `base_addr` regardless. Chunks are cached under `target/`
/// and only rewritten when the model changes.
fn model_loader_args(gguf: &Path, base_addr: u64) -> Result<Vec<String>, String> {
    let size = fs::metadata(gguf).map_err(|e| format!("stat {}: {e}", gguf.display()))?.len();
    const CHUNK: u64 = 1 << 30; // 1 GiB, safely under the loader's per-read limit
    if size < 2 << 30 {
        return Ok(vec![format!("loader,file={},addr={:#x},force-raw=on", gguf.display(), base_addr)]);
    }

    let dir = repo_root().join("target/model-chunks");
    let meta = dir.join("source.meta");
    let want = format!("{}:{size}", gguf.display());
    let n_chunks = size.div_ceil(CHUNK);
    let fresh = fs::read_to_string(&meta).map(|s| s.trim() == want).unwrap_or(false)
        && (0..n_chunks).all(|i| dir.join(format!("c{i}")).exists());
    if !fresh {
        eprintln!("splitting {} ({size} bytes) into {n_chunks} chunks for QEMU loader...", gguf.display());
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).map_err(|e| format!("mkdir {}: {e}", dir.display()))?;
        let mut f = fs::File::open(gguf).map_err(|e| format!("open {}: {e}", gguf.display()))?;
        let mut buf = vec![0u8; 8 << 20];
        for i in 0..n_chunks {
            let mut remaining = CHUNK.min(size - i * CHUNK);
            let mut out = fs::File::create(dir.join(format!("c{i}"))).map_err(|e| e.to_string())?;
            while remaining > 0 {
                let want = buf.len().min(remaining as usize);
                let got = f.read(&mut buf[..want]).map_err(|e| e.to_string())?;
                if got == 0 {
                    break;
                }
                out.write_all(&buf[..got]).map_err(|e| e.to_string())?;
                remaining -= got as u64;
            }
        }
        fs::write(&meta, want).map_err(|e| e.to_string())?;
    }
    Ok((0..n_chunks)
        .map(|i| {
            format!("loader,file={},addr={:#x},force-raw=on", dir.join(format!("c{i}")).display(), base_addr + i * CHUNK)
        })
        .collect())
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask is always under <repo>/xtask")
        .to_path_buf()
}

fn kernel_dir() -> PathBuf {
    repo_root().join("kernel")
}

fn run(cmd: &mut Command) -> Result<(), String> {
    let status = cmd
        .status()
        .map_err(|e| format!("failed to spawn {cmd:?}: {e}"))?;
    if !status.success() {
        return Err(format!("command failed ({status}): {cmd:?}"));
    }
    Ok(())
}

/// As `build_kernel`, but with extra cargo features enabled.
fn build_kernel_with(release: bool, features: &[&str]) -> Result<PathBuf, String> {
    let kdir = kernel_dir();
    let mut cmd = Command::new("cargo");
    cmd.current_dir(&kdir)
        .arg("build")
        .arg("--bin")
        .arg("chitti-kernel");
    if release {
        cmd.arg("--release");
    }
    if !features.is_empty() {
        cmd.arg("--features").arg(features.join(","));
    }
    run(&mut cmd)?;

    let profile = if release { "release" } else { "debug" };
    Ok(kdir
        .join("target/x86_64-chitti")
        .join(profile)
        .join("chitti-kernel"))
}

fn brew_prefix(pkg: &str) -> Result<PathBuf, String> {
    let out = Command::new("brew")
        .args(["--prefix", pkg])
        .output()
        .map_err(|e| format!("failed to run `brew --prefix {pkg}`: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "`brew --prefix {pkg}` failed; install it with `brew install {pkg}`, \
             or point CHITTI_LIMINE_SHARE/CHITTI_LIMINE_BIN at an existing install"
        ));
    }
    Ok(PathBuf::from(
        String::from_utf8_lossy(&out.stdout).trim().to_string(),
    ))
}

fn limine_share_dir() -> Result<PathBuf, String> {
    if let Ok(dir) = env::var("CHITTI_LIMINE_SHARE") {
        return Ok(PathBuf::from(dir));
    }
    Ok(brew_prefix("limine")?.join("share/limine"))
}

fn limine_bin() -> Result<PathBuf, String> {
    if let Ok(bin) = env::var("CHITTI_LIMINE_BIN") {
        return Ok(PathBuf::from(bin));
    }
    Ok(brew_prefix("limine")?.join("bin/limine"))
}

/// Assemble a hybrid BIOS/UEFI ISO around `kernel_bin` with the default (0.8B)
/// model bundled, per Limine's documented `xorriso` + `limine bios-install`
/// recipe (USAGE.md#bios-uefi-hybrid-iso-creation).
/// Split `model` into `<= part_size`-byte files `model.gguf.000`, `.001`, ...
/// under `dir`, streaming (never loading the whole file into memory). Returns
/// the part file names in order. A model that already fits in one part still
/// becomes a single `model.gguf.000`, so the kernel's collect+sort path is
/// uniform.
fn split_model_into_parts(model: &Path, dir: &Path, part_size: u64) -> Result<Vec<String>, String> {
    use std::io::{Read, Write};
    let mut f = fs::File::open(model).map_err(|e| format!("opening model: {e}"))?;
    let mut names = Vec::new();
    let mut buf = vec![0u8; 8 * 1024 * 1024];
    let mut idx = 0usize;
    loop {
        let name = format!("model.gguf.{idx:03}");
        let mut out = fs::File::create(dir.join(&name)).map_err(|e| format!("creating {name}: {e}"))?;
        let mut written: u64 = 0;
        while written < part_size {
            let want = ((part_size - written) as usize).min(buf.len());
            let n = f.read(&mut buf[..want]).map_err(|e| format!("reading model: {e}"))?;
            if n == 0 {
                break;
            }
            out.write_all(&buf[..n]).map_err(|e| format!("writing {name}: {e}"))?;
            written += n as u64;
        }
        names.push(name);
        idx += 1;
        // If we wrote less than a full part, we hit EOF -- done.
        if written < part_size {
            break;
        }
    }
    Ok(names)
}

/// The voice models to bundle into images, as `(bundled-name, repo-relative
/// path)`. Bundled only when present (gitignored; `cargo xtask voice-assets`
/// downloads them). `bundled-name` is what the kernel matches: `find_module`
/// (x86 Limine) and the on-disk filename (aarch64 ESP) both look for
/// "kitten"/"parakeet".
fn voice_model_assets() -> [(&'static str, PathBuf); 2] {
    let root = repo_root();
    [
        ("kitten.onnx", root.join("assets/voice/kitten_tts_mini.onnx")),
        ("parakeet.onnx", root.join("assets/voice/parakeet_ctc_int8.onnx")),
    ]
}

fn assemble_image(kernel_bin: &Path) -> Result<PathBuf, String> {
    assemble_image_opt(kernel_bin, Some("assets/model.gguf"))
}

/// As `assemble_image`, but bundling the model GGUF at `model_rel`.
fn assemble_image_with(kernel_bin: &Path, model_rel: &str) -> Result<PathBuf, String> {
    assemble_image_opt(kernel_bin, Some(model_rel))
}

/// `model_rel`: repo-relative path of the model GGUF to copy into the image, or
/// `None` to bundle no model. The fast test suite (`cargo xtask test`) passes
/// `None` so it never bundles or boots the model it doesn't need;
/// `run`/`image`/`ref-check` pass the selected model.
fn assemble_image_opt(kernel_bin: &Path, model_rel: Option<&str>) -> Result<PathBuf, String> {
    let root = repo_root();
    let iso_root = root.join("target/iso_root");
    if iso_root.exists() {
        fs::remove_dir_all(&iso_root)
            .map_err(|e| format!("removing stale {}: {e}", iso_root.display()))?;
    }
    fs::create_dir_all(iso_root.join("boot/limine")).map_err(|e| e.to_string())?;
    fs::create_dir_all(iso_root.join("EFI/BOOT")).map_err(|e| e.to_string())?;

    fs::copy(kernel_bin, iso_root.join("boot/chitti-kernel"))
        .map_err(|e| format!("copying kernel binary: {e}"))?;
    fs::copy(
        root.join("kernel/limine.conf"),
        iso_root.join("boot/limine/limine.conf"),
    )
    .map_err(|e| format!("copying limine.conf: {e}"))?;

    // The Phase 3 model is loaded as a Limine boot module, not compiled in.
    // It's optional at image-assembly time: the tensor-kernel unit tests
    // (`cargo xtask test`) don't need it, only end-to-end inference does.
    // `limine.conf` references it unconditionally; Limine tolerates a
    // missing module (the ModuleRequest response simply omits it).
    // Bundle the model + declare it in limine.conf ONLY when both requested
    // and present. Limine panics at boot if `module_path` names a module the
    // image lacks, so the declaration must track the actual file.
    let model = model_rel.map(|r| root.join(r)).filter(|p| p.exists());
    if let Some(model) = model {
        // Split the model into <= part-size chunks named model.gguf.000,
        // model.gguf.001, ... and declare each as a Limine module. ISO9660
        // caps a single file at 4 GiB, so a large model (the 9B) must be split
        // to ship inside one ISO; the kernel reassembles the parts. The part
        // size is 3 GiB by default (override with CHITTI_MODEL_PART_MB, used by
        // tests to force multi-part with a small model).
        let part_mb: u64 = std::env::var("CHITTI_MODEL_PART_MB").ok().and_then(|v| v.parse().ok()).unwrap_or(3072);
        let part_size = part_mb * 1024 * 1024;
        let parts = split_model_into_parts(&model, &iso_root.join("boot"), part_size)?;
        let conf_path = iso_root.join("boot/limine/limine.conf");
        let mut conf = fs::read_to_string(&conf_path).map_err(|e| e.to_string())?;
        for name in &parts {
            conf.push_str(&format!("    module_path: boot():/boot/{name}\n"));
        }
        fs::write(&conf_path, conf).map_err(|e| format!("appending module_path: {e}"))?;
        eprintln!("bundled model as {} Limine module part(s)", parts.len());
    } else if let Some(r) = model_rel {
        eprintln!(
            "xtask: note: {r} not present -- run xtask/fetch-model.sh for inference; \
             the image boots without it (no module declared)."
        );
    }

    let share = limine_share_dir()?;
    for f in ["limine-bios.sys", "limine-bios-cd.bin", "limine-uefi-cd.bin"] {
        fs::copy(share.join(f), iso_root.join("boot/limine").join(f))
            .map_err(|e| format!("copying {f} from {}: {e}", share.display()))?;
    }
    for f in ["BOOTX64.EFI", "BOOTIA32.EFI"] {
        fs::copy(share.join(f), iso_root.join("EFI/BOOT").join(f))
            .map_err(|e| format!("copying {f} from {}: {e}", share.display()))?;
    }

    // Installer payload: the files `/install` writes to a target disk to make
    // it boot standalone -- the Limine UEFI binary + the kernel itself -- are
    // bundled as extra Limine modules so the running system can read them from
    // memory and write them to the new disk. (limine.conf for the installed
    // disk is generated in-kernel, not bundled.)
    fs::create_dir_all(iso_root.join("boot/payload")).map_err(|e| e.to_string())?;
    fs::copy(share.join("BOOTX64.EFI"), iso_root.join("boot/payload/BOOTX64.EFI"))
        .map_err(|e| format!("bundling payload BOOTX64.EFI: {e}"))?;
    fs::copy(kernel_bin, iso_root.join("boot/payload/chitti-kernel"))
        .map_err(|e| format!("bundling payload kernel: {e}"))?;
    {
        let conf_path = iso_root.join("boot/limine/limine.conf");
        let mut conf = fs::read_to_string(&conf_path).map_err(|e| e.to_string())?;
        conf.push_str("    module_path: boot():/boot/payload/BOOTX64.EFI\n");
        conf.push_str("    module_path: boot():/boot/payload/chitti-kernel\n");
        fs::write(&conf_path, conf).map_err(|e| format!("declaring payload modules: {e}"))?;
    }

    // Bundle the voice models (STT/TTS) as Limine boot modules when present, so
    // the kernel finds them by name (`find_module`) and auto-loads them — no
    // manual `/voice models load`.
    {
        let conf_path = iso_root.join("boot/limine/limine.conf");
        let mut conf = fs::read_to_string(&conf_path).map_err(|e| e.to_string())?;
        for (name, path) in voice_model_assets() {
            if path.exists() {
                fs::copy(&path, iso_root.join("boot").join(name)).map_err(|e| format!("bundling voice model {name}: {e}"))?;
                conf.push_str(&format!("    module_path: boot():/boot/{name}\n"));
                eprintln!("bundled voice model {name} ({} MiB)", fs::metadata(&path).map(|m| m.len() >> 20).unwrap_or(0));
            }
        }
        fs::write(&conf_path, conf).map_err(|e| format!("declaring voice modules: {e}"))?;
    }

    let iso_path = root.join("target/chitti.iso");
    run(Command::new("xorriso")
        .args([
            "-as",
            "mkisofs",
            "-R",
            "-r",
            "-J",
            "-b",
            "boot/limine/limine-bios-cd.bin",
            "-no-emul-boot",
            "-boot-load-size",
            "4",
            "-boot-info-table",
            "-hfsplus",
            "-apm-block-size",
            "2048",
            "--efi-boot",
            "boot/limine/limine-uefi-cd.bin",
            "-efi-boot-part",
            "--efi-boot-image",
            "--protective-msdos-label",
        ])
        .arg(&iso_root)
        .arg("-o")
        .arg(&iso_path))?;

    run(Command::new(limine_bin()?).arg("bios-install").arg(&iso_path))?;

    Ok(iso_path)
}

/// `cargo xtask image -arch aarch64`: build the **distributable aarch64 disk
/// image** `target/chitti-aa64.img` — a GPT disk with a FAT32 ESP (the Chitti
/// UEFI stub as BOOTAA64.EFI + the kernel + the model) and an ext4 data
/// partition for durable agent state. This is the aarch64 counterpart of the
/// x86 ISO: end users write it to a USB drive (dd / balenaEtcher) and boot any
/// UEFI aarch64 machine, or attach it as the disk of a UTM/QEMU/VirtualBox-ARM
/// VM. It boots standalone — it IS the installed disk.
fn image_aarch64(model: Model) -> Result<(), String> {
    let elf = build_kernel_aarch64(true, model.features())?;
    assert_identity_kernel(&elf)?;
    let stub = build_stub_aarch64()?;
    let gguf = repo_root().join(model.gguf_rel());
    let model_path = gguf.exists().then_some(gguf);
    if model_path.is_none() {
        eprintln!("note: {} absent -- building a model-less image", model.gguf_rel());
    }

    // Layout: GPT (34 + 33 reserved sectors) + ESP (payload + 64 MiB slack) +
    // 256 MiB ext4 data partition.
    let payload: u64 = fs::metadata(&elf).map(|m| m.len()).unwrap_or(0)
        + fs::metadata(&stub).map(|m| m.len()).unwrap_or(0)
        + model_path.as_ref().and_then(|p| fs::metadata(p).ok()).map(|m| m.len()).unwrap_or(0);
    let esp_secs = (payload + 64 * 1024 * 1024).div_ceil(512);
    let data_secs = 256 * 1024 * 1024 / 512u64;
    let total_secs = 34 + esp_secs + data_secs + 34;
    let esp = (34u64, 34 + esp_secs - 1);
    let data = (esp.1 + 1, esp.1 + data_secs);

    let img = repo_root().join("target/chitti-aa64.img");
    let f = fs::OpenOptions::new().create(true).write(true).truncate(true).open(&img).map_err(|e| e.to_string())?;
    f.set_len(total_secs * 512).map_err(|e| e.to_string())?;
    write_gpt_host(&f, total_secs, &[(ESP_GUID_H, esp.0, esp.1, "EFI System"), (LINUX_GUID_H, data.0, data.1, "Chitti Data")])?;
    drop(f);

    // Format + populate the two partitions via macOS hdiutil/diskutil (the GPT
    // is parsed on attach, exposing slice devices s1/s2 owned by the user).
    let mke2fs = brew_prefix("e2fsprogs").map(|p| p.join("sbin/mke2fs")).ok().filter(|p| p.exists())
        .ok_or("mke2fs not found -- brew install e2fsprogs (needed to format the data partition)")?;
    let script = format!(
        r#"set -e
DEV=$(hdiutil attach -nomount -imagekey diskimage-class=CRawDiskImage "{img}" | head -1 | awk '{{print $1}}')
newfs_msdos -F 32 -v CHITTI "${{DEV}}s1" > /dev/null
diskutil mount "${{DEV}}s1" > /dev/null
MNT=$(diskutil info "${{DEV}}s1" | awk -F': *' '/Mount Point/{{print $2}}')
mkdir -p "$MNT/EFI/BOOT"
cp "{stub}" "$MNT/EFI/BOOT/BOOTAA64.EFI"
cp "{kernel}" "$MNT/chitti-kernel"
{model_cp}
diskutil unmount "${{DEV}}s1" > /dev/null
"{mke2fs}" -F -q -t ext4 -b 4096 "${{DEV}}s2"
hdiutil detach "$DEV" > /dev/null
"#,
        img = img.display(),
        stub = stub.display(),
        kernel = elf.display(),
        mke2fs = mke2fs.display(),
        model_cp = model_path.as_ref().map(|m| format!("cp \"{}\" \"$MNT/model.gguf.000\"", m.display())).unwrap_or_default(),
    );
    let status = Command::new("/bin/sh").arg("-c").arg(&script).status().map_err(|e| format!("populating image: {e}"))?;
    if !status.success() {
        return Err("aarch64 image build failed (hdiutil/newfs_msdos/mke2fs)".into());
    }
    println!("image: {} ({} MiB, GPT: ESP {}..{} + ext4 data)", img.display(), total_secs * 512 / (1024 * 1024), esp.0, esp.1);
    println!("  write it to a USB drive (dd/balenaEtcher) or attach as a VM disk -- it boots standalone via UEFI.");
    Ok(())
}

const ESP_GUID_H: [u8; 16] = [0x28, 0x73, 0x2a, 0xc1, 0x1f, 0xf8, 0xd2, 0x11, 0xba, 0x4b, 0x00, 0xa0, 0xc9, 0x3e, 0xc9, 0x3b];
const LINUX_GUID_H: [u8; 16] = [0xaf, 0x3d, 0xc6, 0x0f, 0x83, 0x84, 0x72, 0x47, 0x8e, 0x79, 0x3d, 0x69, 0xd8, 0x47, 0x7d, 0xe4];

/// Host-side GPT writer (a std port of the kernel's `block::gpt::write`):
/// protective MBR + primary/backup headers + entry arrays, IEEE CRC32.
fn write_gpt_host(f: &fs::File, total: u64, parts: &[([u8; 16], u64, u64, &str)]) -> Result<(), String> {
    use std::io::{Seek, SeekFrom, Write};
    use std::os::unix::fs::FileExt;
    fn crc32(data: &[u8]) -> u32 {
        let mut crc: u32 = 0xffff_ffff;
        for &b in data {
            crc ^= b as u32;
            for _ in 0..8 {
                let mask = (crc & 1).wrapping_neg();
                crc = (crc >> 1) ^ (0xedb8_8320 & mask);
            }
        }
        !crc
    }
    let mut w = f;
    let _ = w.seek(SeekFrom::Start(0));
    // Protective MBR.
    let mut mbr = [0u8; 512];
    mbr[0x1be + 4] = 0xee;
    mbr[0x1be + 8..0x1be + 12].copy_from_slice(&1u32.to_le_bytes());
    mbr[0x1be + 12..0x1be + 16].copy_from_slice(&((total - 1).min(0xffff_ffff) as u32).to_le_bytes());
    mbr[510] = 0x55;
    mbr[511] = 0xaa;
    w.write_all(&mbr).map_err(|e| e.to_string())?;
    // Entry array.
    let mut entries = vec![0u8; 128 * 128];
    for (i, (guid, first, last, name)) in parts.iter().enumerate() {
        let e = i * 128;
        entries[e..e + 16].copy_from_slice(guid);
        for b in 0..16 {
            entries[e + 16 + b] = (0xa0 + i as u8).wrapping_add(b as u8);
        }
        entries[e + 32..e + 40].copy_from_slice(&first.to_le_bytes());
        entries[e + 40..e + 48].copy_from_slice(&last.to_le_bytes());
        for (k, c) in name.encode_utf16().take(36).enumerate() {
            entries[e + 56 + k * 2..e + 58 + k * 2].copy_from_slice(&c.to_le_bytes());
        }
    }
    let entries_crc = crc32(&entries);
    let backup_hdr = total - 1;
    let backup_entries = backup_hdr - 32;
    f.write_all_at(&entries, 2 * 512).map_err(|e| e.to_string())?;
    f.write_all_at(&entries, backup_entries * 512).map_err(|e| e.to_string())?;
    // Headers.
    let hdr = |my: u64, alt: u64, ent: u64| -> Vec<u8> {
        let mut h = vec![0u8; 512];
        h[0..8].copy_from_slice(b"EFI PART");
        h[8..12].copy_from_slice(&0x0001_0000u32.to_le_bytes());
        h[12..16].copy_from_slice(&92u32.to_le_bytes());
        h[24..32].copy_from_slice(&my.to_le_bytes());
        h[32..40].copy_from_slice(&alt.to_le_bytes());
        h[40..48].copy_from_slice(&34u64.to_le_bytes());
        h[48..56].copy_from_slice(&(total - 34).to_le_bytes());
        h[56..72].copy_from_slice(b"CHITTI-OS-DISK01");
        h[72..80].copy_from_slice(&ent.to_le_bytes());
        h[80..84].copy_from_slice(&128u32.to_le_bytes());
        h[84..88].copy_from_slice(&128u32.to_le_bytes());
        h[88..92].copy_from_slice(&entries_crc.to_le_bytes());
        let hc = crc32(&h[0..92]);
        h[16..20].copy_from_slice(&hc.to_le_bytes());
        h
    };
    f.write_all_at(&hdr(1, backup_hdr, 2), 512).map_err(|e| e.to_string())?;
    f.write_all_at(&hdr(backup_hdr, 1, backup_entries), backup_hdr * 512).map_err(|e| e.to_string())?;
    Ok(())
}

fn image(release: bool, model: Model) -> Result<(), String> {
    let bin = build_kernel_with(release, model.features())?;
    // Bundle the selected model if present; otherwise a kernel-only bootable
    // ISO (what CI ships -- the model is fetched separately, being large).
    let iso = assemble_image_with(&bin, model.gguf_rel())?;
    println!("image: {}", iso.display());
    Ok(())
}

// `-no-shutdown` deliberately excluded: it makes QEMU pause instead of
// exit on a guest-triggered shutdown, which is exactly what writing to
// isa-debug-exit causes. The test runner needs the process to actually
// exit so it can read the isa-debug-exit status code back.
const QEMU_BASE_ARGS: &[&str] = &[
    "-M",
    "q35",
    // `-cpu max`: expose AVX2 + XSAVE under TCG so the Cortex kernels can
    // use the AVX2/FMA path (the default `qemu64` lacks them, and the kernel
    // then falls back to SSE2 -- correct, just slower).
    "-cpu",
    "max",
    // NB: SMP (`-smp 4`) is added only by the test runner (`cmd_runner`), where
    // the SMP bring-up + spinlock self-test runs. Interactive `run` and
    // `ref-check` stay single-CPU: inference is BSP-bound, and under
    // single-thread TCG extra vCPUs only cost round-robin overhead.
    "-m",
    "2G",
    "-device",
    "isa-debug-exit,iobase=0xf4,iosize=0x04",
    "-no-reboot",
];

fn qemu_base_cmd(iso: &Path) -> Command {
    let mut cmd = Command::new("qemu-system-x86_64");
    cmd.args(QEMU_BASE_ARGS);
    cmd.arg("-cdrom").arg(iso);
    cmd
}

/// `cargo xtask run [-arch ...]`: boot the unified kernel. On aarch64 it runs
/// natively via QEMU + HVF; on x86 it boots the Limine image under
/// qemu-system-x86_64 (TCG on this host).
fn cmd_run(release: bool, arch: Arch, model: Model, uefi: bool, disk_only: bool, fresh_disk: bool, disk_size: Option<String>, no_model: bool) -> Result<(), String> {
    if arch == Arch::Aarch64 {
        let disk = match &disk_size {
            Some(s) => Some(ensure_disk_image(parse_size(s)?, fresh_disk)?),
            // --disk-only with no size: keep the existing (installed) disk
            // AS-IS — wipe only if --fresh-disk was explicitly passed.
            None if fresh_disk || disk_only => Some(ensure_disk_image(0, fresh_disk)?),
            None => None,
        };
        // `--uefi` or `--disk-only` on aarch64 => firmware boot via the Chitti
        // UEFI stub (AAVMF launches BOOTAA64.EFI, which loads the normal
        // identity kernel + model off a real FAT ESP and hands off MMU-off via
        // an identity-RAM trampoline). Otherwise the fast -kernel HVF path.
        if uefi || disk_only {
            return cmd_run_aarch64_uefi(model, disk, disk_only, no_model);
        }
        return cmd_run_aarch64(release, model, disk, disk_only);
    }
    // Disk size: default 4 MiB for a plain run (the SimpleFS boot-counter demo),
    // but an install needs room for the ESP + model + data partitions — pass e.g.
    // `--disk 2G`. `--disk-only` boots whatever is already installed.
    let want_bytes = match &disk_size {
        Some(s) => parse_size(s)?,
        None if disk_only => 0, // keep the existing installed disk as-is
        None => 4 * 1024 * 1024,
    };
    let disk = ensure_disk_image(want_bytes, fresh_disk)?;

    // --- Boot from the installed disk only (no ISO) ------------------------
    if disk_only {
        let mut cmd = Command::new("qemu-system-x86_64");
        // No -cdrom: UEFI finds the bootloader on the installed ESP. More RAM so
        // the ~800 MiB model read off ext4 into contiguous frames has headroom.
        cmd.args(["-M", "q35", "-cpu", "max", "-m", "3G", "-no-reboot"]);
        cmd.args(["-device", "isa-debug-exit,iobase=0xf4,iosize=0x04"]);
        for arg in ovmf_pflash_args()? {
            cmd.arg(arg);
        }
        cmd.args(["-serial", "stdio"]);
        cmd.arg("-drive").arg(format!("file={},if=none,id=chittidisk,format=raw", disk.display()));
        cmd.args(["-device", "virtio-blk-pci,drive=chittidisk,disable-modern=on"]);
        cmd.args(["-device", "qemu-xhci,id=xhci", "-device", "usb-kbd,bus=xhci.0"]);
        // Intel e1000 NIC on user-mode networking (DHCP 10.0.2.15 / gw 10.0.2.2).
        cmd.args(["-netdev", "user,id=chittinet", "-device", "e1000,netdev=chittinet"]);
        eprintln!("booting FROM DISK ONLY via UEFI (OVMF) -- no ISO; the installed Chitti boots itself");
        eprintln!("  disk: {}", disk.display());
        let status = cmd.status().map_err(|e| format!("failed to spawn qemu-system-x86_64: {e}"))?;
        eprintln!("qemu exited: {status}");
        return Ok(());
    }

    // --- Boot the ISO (optionally under UEFI); run `/install` from here ----
    let bin = build_kernel_with(release, model.features())?;
    let iso = if no_model {
        eprintln!("--no-model: ISO has no model module -- `/install` writes kernel + config + an empty data partition (fast)");
        assemble_image_opt(&bin, None)?
    } else {
        assemble_image_with(&bin, model.gguf_rel())?
    };
    let mut cmd = qemu_base_cmd(&iso);
    if uefi {
        for arg in ovmf_pflash_args()? {
            cmd.arg(arg);
        }
        eprintln!("booting via UEFI (OVMF) -- the same GOP framebuffer path real hardware uses");
    }
    cmd.args(["-serial", "stdio"]);
    cmd.arg("-drive").arg(format!("file={},if=none,id=chittidisk,format=raw", disk.display()));
    cmd.args(["-device", "virtio-blk-pci,drive=chittidisk,disable-modern=on"]);
    // A USB keyboard on an xHCI controller, so the xhci/HID driver drives the
    // shell (as a real USB keyboard would); PS/2 also still works.
    cmd.args(["-device", "qemu-xhci,id=xhci", "-device", "usb-kbd,bus=xhci.0"]);
    // Intel e1000 NIC on user-mode networking (DHCP 10.0.2.15 / gw 10.0.2.2).
    cmd.args(["-netdev", "user,id=chittinet", "-device", "e1000,netdev=chittinet"]);
    // virtio-snd on the host's CoreAudio (mic + speaker) for /voice.
    cmd.args(["-audiodev", "coreaudio,id=chittiaudio", "-device", "virtio-sound-pci,audiodev=chittiaudio"]);
    if disk_size.is_some() {
        eprintln!("  disk: {} ({}) -- run `/install yes` at the shell, then reboot with `--disk-only`", disk.display(), disk_size.as_deref().unwrap_or(""));
    }
    let status = cmd
        .status()
        .map_err(|e| format!("failed to spawn qemu-system-x86_64: {e}"))?;
    eprintln!("qemu exited: {status}");
    Ok(())
}

/// Value of a `--flag <value>` option in `args`, if present.
fn flag_value(args: &[String], flag: &str) -> Option<String> {
    args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1).cloned())
}

/// Parse a disk size like `2G`, `1500M`, `800000000` into bytes.
fn parse_size(s: &str) -> Result<u64, String> {
    let s = s.trim();
    let (num, mult) = match s.chars().last() {
        Some('G') | Some('g') => (&s[..s.len() - 1], 1024 * 1024 * 1024),
        Some('M') | Some('m') => (&s[..s.len() - 1], 1024 * 1024),
        Some('K') | Some('k') => (&s[..s.len() - 1], 1024),
        _ => (s, 1),
    };
    num.trim().parse::<u64>().map(|n| n * mult).map_err(|_| format!("bad --disk size {s:?} (use e.g. 2G, 1500M)"))
}

/// QEMU `-drive if=pflash` args to boot via OVMF (UEFI). The code volume is
/// read-only; a per-run writable copy of the vars volume is placed under
/// `target/`. Both come from QEMU's bundled edk2 firmware.
fn ovmf_pflash_args() -> Result<Vec<String>, String> {
    let share = brew_prefix("qemu")?.join("share/qemu");
    let code = share.join("edk2-x86_64-code.fd");
    let vars_src = share.join("edk2-i386-vars.fd");
    if !code.exists() || !vars_src.exists() {
        return Err(format!("OVMF firmware not found under {} (need edk2-x86_64-code.fd + edk2-i386-vars.fd)", share.display()));
    }
    // A fresh writable vars copy per invocation (UEFI writes boot vars).
    let vars = repo_root().join("target/ovmf-vars.fd");
    fs::copy(&vars_src, &vars).map_err(|e| format!("copying OVMF vars: {e}"))?;
    Ok(vec![
        "-drive".into(),
        format!("if=pflash,format=raw,unit=0,readonly=on,file={}", code.display()),
        "-drive".into(),
        format!("if=pflash,format=raw,unit=1,file={}", vars.display()),
    ])
}

/// Ensure `target/chitti-disk.img` exists and is at least `want_bytes` (a sparse
/// raw disk backing the virtio-blk device). Kept across runs so an install — and
/// the SimpleFS/synapse persistence — survives a reboot. Grown (never shrunk) if
/// the existing image is smaller than requested; `fresh` wipes it first.
fn ensure_disk_image(want_bytes: u64, fresh: bool) -> Result<PathBuf, String> {
    let path = repo_root().join("target/chitti-disk.img");
    if fresh && path.exists() {
        fs::remove_file(&path).map_err(|e| format!("removing disk image: {e}"))?;
        eprintln!("--fresh-disk: wiped {}", path.display());
    }
    let cur = fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
    let want = if want_bytes == 0 { cur.max(4 * 1024 * 1024) } else { want_bytes };
    if !path.exists() || cur < want {
        // Sparse allocation: set the length without writing `want` bytes of zeros
        // (a multi-GiB disk would otherwise take real space + time).
        let f = fs::OpenOptions::new().create(true).write(true).open(&path).map_err(|e| format!("creating disk image: {e}"))?;
        f.set_len(want).map_err(|e| format!("sizing disk image: {e}"))?;
        eprintln!("disk image {} sized to {} MiB (sparse)", path.display(), want / (1024 * 1024));
    }
    Ok(path)
}

/// `cargo xtask voice-assets`: download the /voice ONNX models into
/// `assets/voice/` (cached — skips files already present): silero-vad v5
/// (VAD), NeMo parakeet-tdt-ctc-110m int8 (STT, from the sherpa-onnx release
/// bundle, plus its tokens.txt) and KittenTTS mini (TTS).
fn cmd_voice_assets() -> Result<(), String> {
    let dir = repo_root().join("assets/voice");
    fs::create_dir_all(&dir).map_err(|e| format!("mkdir {}: {e}", dir.display()))?;
    let fetch = |name: &str, url: &str| -> Result<(), String> {
        let dst = dir.join(name);
        if dst.exists() {
            eprintln!("voice-assets: {name} already present");
            return Ok(());
        }
        eprintln!("voice-assets: downloading {name}…");
        let st = Command::new("curl")
            .args(["-sL", "-o"])
            .arg(&dst)
            .arg(url)
            .status()
            .map_err(|e| format!("curl: {e}"))?;
        if !st.success() {
            return Err(format!("download failed for {name}"));
        }
        Ok(())
    };
    fetch("silero_vad.onnx", "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx")?;
    fetch(
        "kitten_tts_mini.onnx",
        "https://huggingface.co/KittenML/kitten-tts-mini-0.8/resolve/main/kitten_tts_mini_v0_8.onnx",
    )?;
    // The parakeet STT model ships inside a tar.bz2 bundle with its tokens.
    if !dir.join("parakeet_ctc_int8.onnx").exists() {
        eprintln!("voice-assets: downloading parakeet (STT) bundle…");
        let tmp = std::env::temp_dir().join("parakeet.tar.bz2");
        let st = Command::new("curl")
            .args(["-sL", "-o"])
            .arg(&tmp)
            .arg("https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet_tdt_ctc_110m-en-36000-int8.tar.bz2")
            .status()
            .map_err(|e| format!("curl: {e}"))?;
        if !st.success() {
            return Err("parakeet download failed".into());
        }
        let st = Command::new("tar").args(["-xjf"]).arg(&tmp).arg("-C").arg(std::env::temp_dir()).status().map_err(|e| format!("tar: {e}"))?;
        if !st.success() {
            return Err("parakeet extract failed".into());
        }
        let src = std::env::temp_dir().join("sherpa-onnx-nemo-parakeet_tdt_ctc_110m-en-36000-int8");
        fs::copy(src.join("model.int8.onnx"), dir.join("parakeet_ctc_int8.onnx")).map_err(|e| format!("copy: {e}"))?;
        fs::copy(src.join("tokens.txt"), dir.join("parakeet_tokens.txt")).map_err(|e| format!("copy: {e}"))?;
    } else {
        eprintln!("voice-assets: parakeet already present");
    }
    eprintln!("voice-assets: done — assets/voice/ ready");
    Ok(())
}

/// `cargo xtask test`: run the in-kernel `custom_test_frameworks` test
/// suite via `cargo test --lib`, which cross-compiles each test binary and
/// hands it to the `runner` subcommand below to execute under QEMU.
fn cmd_test() -> Result<(), String> {
    let kdir = kernel_dir();
    let mut cmd = Command::new("cargo");
    cmd.current_dir(&kdir).args(["test", "--lib", "--quiet"]);
    run(&mut cmd)
}

/// Invoked by cargo (via the `runner` config) with the path to a compiled
/// test binary. Boots it in QEMU headlessly and translates the
/// isa-debug-exit exit code `(value << 1) | 1` into this process's exit
/// status, which cargo interprets as the test outcome.
fn cmd_runner(args: &[String]) -> Result<(), String> {
    let bin_path = args
        .first()
        .ok_or_else(|| "runner: missing test binary path argument".to_string())?;
    // Fast test suite: exclude the model so the ISO stays small and boots
    // quickly (the tensor-kernel tests validate against baked-in NumPy
    // reference vectors, not the real model).
    let iso = assemble_image_opt(Path::new(bin_path), None)?;
    let mut cmd = qemu_base_cmd(&iso);
    // The in-kernel test suite includes the Phase 7 SMP bring-up + spinlock
    // self-test, so the harness runs with four vCPUs.
    cmd.args(["-smp", "4", "-serial", "stdio", "-display", "none"]); cmd.args(["-d","int,cpu_reset","-D","/tmp/qint2.log"]);
    let status = cmd
        .status()
        .map_err(|e| format!("failed to spawn qemu-system-x86_64: {e}"))?;
    match status.code() {
        Some(33) => Ok(()), // QemuExitCode::Success (0x10) -> (0x10 << 1) | 1
        Some(other) => Err(format!(
            "QEMU exited with status {other} (expected 33 = isa-debug-exit success)"
        )),
        None => Err("QEMU was terminated by a signal".to_string()),
    }
}

/// `cargo xtask ref-check`: the Phase 3 reference-parity gate. Builds the
/// kernel in release with the `refcheck` feature, boots it with the real
/// model and extra RAM, and lets the in-kernel acceptance routine
/// (`cortex::run_acceptance`) exit QEMU with success (33) or failure. Serial
/// goes to stdio so the `REFCHECK:` lines are visible while it runs.
fn cmd_ref_check() -> Result<(), String> {
    let model = repo_root().join("assets/model.gguf");
    if !model.exists() {
        return Err("assets/model.gguf not present -- run xtask/fetch-model.sh first".to_string());
    }
    let bin = build_kernel_with(true, &["refcheck"])?;
    let iso = assemble_image(&bin)?;
    // CPU inference under QEMU/TCG takes minutes; the model module needs
    // headroom beyond 2 GiB.
    let mut cmd = Command::new("qemu-system-x86_64");
    cmd.args([
        "-M", "q35", "-cpu", "max", "-smp", "4", "-m", "4G", "-device",
        "isa-debug-exit,iobase=0xf4,iosize=0x04", "-no-reboot",
    ]);
    cmd.arg("-cdrom").arg(&iso);
    cmd.args(["-serial", "stdio", "-display", "none"]);
    eprintln!("ref-check: running in-kernel acceptance gate under QEMU (this takes a few minutes)...");
    let status = cmd.status().map_err(|e| format!("failed to spawn qemu-system-x86_64: {e}"))?;
    match status.code() {
        Some(33) => {
            println!("ref-check: PASS (all Phase 3 acceptance checks green)");
            Ok(())
        }
        Some(35) => Err("ref-check: FAIL (QemuExitCode::Failed -- an acceptance check did not match)".to_string()),
        Some(other) => Err(format!("ref-check: QEMU exited with unexpected status {other}")),
        None => Err("ref-check: QEMU was terminated by a signal".to_string()),
    }
}
