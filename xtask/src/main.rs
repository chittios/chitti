//! Build orchestration for ChittiOS: assembles a bootable Limine image
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
    /// Google Gemma 4 E4B instruct (unsloth Q4_K_M GGUF, ~4.6 GiB). Cortex
    /// already speaks the `Gemma4` family; this selects the large heap tier
    /// (same as 9B) and the assets path.
    Gemma4E4B,
    /// PrismML Bonsai-27B **1-bit** (binary) main weights (`Q1_0` GGUF, ~3.8 GiB).
    /// A Qwen3.6-27B hybrid whose GGUF declares `general.architecture = qwen35`
    /// with full DeltaNet SSM keys, so cortex loads it via the existing
    /// `QwenHybrid` family; the only bespoke piece is the `Q1_0` (GGML type 41)
    /// binary dequant (`cortex::tensor::dequant_q1_0_block`, `bit ? +d : −d`).
    /// The smallest/fastest Bonsai — default model for `make run`.
    Bonsai27B,
    /// PrismML Ternary-Bonsai-27B main weights (`Q2_0` ternary GGUF, ~7.17 GiB).
    /// Same `qwen35`/QwenHybrid architecture as the 1-bit build; the bespoke
    /// piece is the `Q2_0` (GGML type 42) ternary dequant. Higher quality than
    /// the 1-bit at ~2× the footprint; selected with `-model bonsai-27b-ternary`.
    Bonsai27BTernary,
    /// Any GGUF by path (`-model path/to/file.gguf`): the kernel derives the
    /// architecture/config from the file itself, so xtask only needs the path
    /// and a derived guest-RAM size (leaked `'static` strs — xtask is a
    /// short-lived host process).
    Custom { path: &'static str, mem: &'static str },
}

impl Model {
    /// Cargo features that select this model's heap-size tier in the kernel.
    fn features(self) -> &'static [&'static str] {
        match self {
            Model::Qwen08B => &[],
            Model::Qwen2B => &["model-2b"],
            Model::Qwen4B => &["model-4b"],
            // ~4.6–5 GiB weights → 1 GiB heap (same tier as 9B).
            Model::Qwen9B | Model::Gemma4E4B => &["model-9b"],
            // ~3.8 GiB Q1_0 / ~7.17 GiB Q2_0 weights → 1 GiB heap (9B tier).
            Model::Bonsai27B | Model::Bonsai27BTernary => &["model-9b"],
            // The default tier (1 GiB heap) fits every model: guest RAM is
            // derived from the file size, and the heap sits at the top of it.
            Model::Custom { .. } => &[],
        }
    }
    /// The GGUF file bundled for this model (relative to the repo root).
    fn gguf_rel(self) -> &'static str {
        match self {
            Model::Qwen08B => "assets/model.gguf",
            Model::Qwen2B => "assets/model-2b.gguf",
            Model::Qwen4B => "assets/model-4b.gguf",
            Model::Qwen9B => "assets/model-9b.gguf",
            Model::Gemma4E4B => "assets/model-gemma4-e4b.gguf",
            Model::Bonsai27B => "assets/model-bonsai-27b-q1.gguf",
            Model::Bonsai27BTernary => "assets/model-bonsai-27b.gguf",
            Model::Custom { path, .. } => path,
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
    /// a model won't fit, so these are simply comfortable sizes per model
    /// (custom paths derive theirs from the file size at parse time).
    fn qemu_mem(self) -> &'static str {
        match self {
            // 0.8B (~785 MiB) at 2 GiB + a 512 MiB heap at the top: 3 GiB is ample.
            Model::Qwen08B => "3G",
            // ~1.2 GiB model at 2 GiB + a 512 MiB heap at the top of RAM.
            Model::Qwen2B => "4G",
            // ~2.58 GiB model at 2 GiB + a 512 MiB heap at the top of RAM.
            Model::Qwen4B => "6G",
            // ~5 GiB Qwen 9B / ~4.6 GiB Gemma 4 E4B Q4_K_M + 1 GiB heap.
            Model::Qwen9B | Model::Gemma4E4B => "10G",
            // ~3.8 GiB Q1_0 model at 2 GiB + a 1 GiB heap at the top of RAM.
            Model::Bonsai27B => "8G",
            // ~7.17 GiB Q2_0 model at 2 GiB + a 1 GiB heap at the top of RAM.
            Model::Bonsai27BTernary => "12G",
            Model::Custom { mem, .. } => mem,
        }
    }
    fn label(self) -> &'static str {
        match self {
            Model::Qwen08B => "qwen3.5-0.8b",
            Model::Qwen2B => "qwen3.5-2b",
            Model::Qwen4B => "qwen3.5-4b",
            Model::Qwen9B => "qwen3.5-9b",
            Model::Gemma4E4B => "gemma-4-e4b",
            Model::Bonsai27B => "bonsai-27b",
            Model::Bonsai27BTernary => "bonsai-27b-ternary",
            Model::Custom { path, .. } => path,
        }
    }
}

/// Convert a QEMU `-m` size string (`"3G"`, `"512M"`, or a raw byte count) to
/// bytes, for the `opt/chitti/ramsize` fw_cfg the kernel reads.
/// Build the slirp user-net `-netdev` value for `id`. If `CHITTI_HOSTFWD=<port>`
/// is set, add a host-forward `tcp:127.0.0.1:<port>-:<port>` so a host process
/// can reach a guest TCP listener (used by the e2e Network-service scenario).
/// Opt-in, so normal `run` invocations keep plain user-mode networking.
/// vCPU count for the aarch64 `run` paths (native cores under HVF):
/// `CHITTI_SMP` overrides, default 8. The kernel parks idle workers in WFE, so
/// extra vCPUs are near-free, and the SDOT matvec/matmul row-split scales with
/// online cores — 8 vCPUs on an 8-core M-series roughly doubles inference over
/// the old `-smp 4`.
fn smp_count() -> String {
    env::var("CHITTI_SMP").ok().filter(|s| !s.trim().is_empty()).unwrap_or_else(|| "8".to_string())
}

fn user_netdev(id: &str) -> String {
    let mut s = format!("user,id={id}");
    if let Ok(ports) = std::env::var("CHITTI_HOSTFWD") {
        for p in ports.split(',').map(str::trim).filter(|p| !p.is_empty()) {
            s.push_str(&format!(",hostfwd=tcp:127.0.0.1:{p}-:{p}"));
        }
    }
    s
}

/// Guest NIC backend for interactive / e2e `run`.
///
/// * `CHITTI_NET_BRIDGE=<ifname>` (e.g. `en0`) — L2 bridge onto that host
///   interface so the guest gets a LAN address via the real DHCP server.
///   macOS uses QEMU's `vmnet-bridged` (Apple vmnet; needs root or the
///   `com.apple.vm.networking` entitlement); other hosts use the classic
///   `bridge` helper (`br=<ifname>`).
/// QEMU storage attachment for the x86 data disk, from `CHITTI_DISK_IF`
/// (default `virtio-blk`).
///
/// Exists so the **real-hardware** storage controllers can be exercised, not just
/// the paravirtual one: `ahci` builds an ich9-ahci HBA (the SATA path most older
/// desktops and laptops use), `nvme` an NVMe controller (what most machines from
/// ~2016 on have), `virtio-blk` the default. Returns the `-device` arguments for
/// drive id `id`; the caller has already added the matching `-drive ...,if=none`.
///
/// `ahci` needs its HBA declared once, so this emits the controller too — pass
/// `first` true for the first disk on the bus. Multiple disks land on separate
/// AHCI ports, which is what exercises the multi-port enumeration.
fn disk_device_args(id: &str, first: bool) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let kind = std::env::var("CHITTI_DISK_IF").unwrap_or_else(|_| "virtio-blk".to_string());
    match kind.as_str() {
        "ahci" | "sata" => {
            if first {
                out.push("-device".into());
                out.push("ich9-ahci,id=ahci".into());
            }
            out.push("-device".into());
            out.push(format!("ide-hd,drive={id},bus=ahci.{}", if first { 0 } else { 1 }));
        }
        "nvme" => {
            out.push("-device".into());
            out.push(format!("nvme,drive={id},serial=chitti-{id}"));
        }
        _ => {
            out.push("-device".into());
            out.push(format!("virtio-blk-pci,drive={id},disable-modern=on"));
        }
    }
    out
}

/// The QEMU NIC device model to attach, from `CHITTI_NIC` (default `e1000`).
///
/// Exists so the by-device-ID NIC dispatch (`net::nic_ids`) can be exercised
/// against every family QEMU can emulate, not just the default:
/// `e1000` (82540EM → the legacy e1000 driver), `e1000e` (82574L → e1000e),
/// `igb` (82576 → the igb/igc driver), `rtl8139`, `virtio-net-pci`.
fn nic_model() -> String {
    std::env::var("CHITTI_NIC").unwrap_or_else(|_| "e1000".to_string())
}

/// * otherwise — slirp user-net via [`user_netdev`] (incl. optional
///   `CHITTI_HOSTFWD`). Hostfwd is ignored when bridging.
fn guest_netdev(id: &str) -> String {
    if let Ok(ifname) = env::var("CHITTI_NET_BRIDGE") {
        let ifname = ifname.trim();
        if !ifname.is_empty() {
            if env::var("CHITTI_HOSTFWD").is_ok() {
                eprintln!(
                    "xtask: note: CHITTI_HOSTFWD is ignored with CHITTI_NET_BRIDGE \
                     (guest is on the LAN; reach it at its DHCP address)"
                );
            }
            #[cfg(target_os = "macos")]
            {
                eprintln!(
                    "  net: vmnet-bridged ifname={ifname} \
                     (may need `sudo` / com.apple.vm.networking)"
                );
                return format!("vmnet-bridged,id={id},ifname={ifname}");
            }
            #[cfg(not(target_os = "macos"))]
            {
                eprintln!("  net: bridge br={ifname}");
                return format!("bridge,id={id},br={ifname}");
            }
        }
    }
    user_netdev(id)
}

/// Boot-time `/model remote` seed for the guest.
///
/// When `CHITTI_REMOTE_URL` is set (optionally `CHITTI_REMOTE_MODEL`,
/// `CHITTI_REMOTE_KEY`), write a small JSON file and hand it to QEMU as
/// `-fw_cfg name=opt/chitti/model,file=…`. The kernel reads it at shell
/// start and activates the hosted backend (same shape as
/// `/configs/core/model.json`). Used by `make run` for LM Studio / Ollama
/// without typing `/model remote` by hand.
///
/// Under slirp user-net the host is `10.0.2.2` (not the host's LAN IP).
fn remote_model_fw_cfg() -> Result<Vec<String>, String> {
    let Ok(url) = env::var("CHITTI_REMOTE_URL") else {
        return Ok(Vec::new());
    };
    let url = url.trim().trim_end_matches('/').to_string();
    if url.is_empty() {
        return Ok(Vec::new());
    }
    if !(url.starts_with("http://") || url.starts_with("https://")) {
        return Err(format!(
            "CHITTI_REMOTE_URL must be http(s)://… (got {url:?})"
        ));
    }
    let model = env::var("CHITTI_REMOTE_MODEL")
        .unwrap_or_else(|_| "default".into())
        .trim()
        .to_string();
    let key = env::var("CHITTI_REMOTE_KEY").unwrap_or_default();
    let key = key.trim();
    // Minimal JSON (no commas in values expected). Written to a file so QEMU's
    // comma-separated -fw_cfg parser never splits the body.
    let json = format!(
        "{{\"mode\":\"remote\",\"url\":\"{}\",\"model\":\"{}\",\"key\":\"{}\"}}",
        json_escape_str(&url),
        json_escape_str(&model),
        json_escape_str(key),
    );
    let path = repo_root().join("target/chitti-fwcfg-model.json");
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("mkdir {}: {e}", parent.display()))?;
    }
    fs::write(&path, json.as_bytes()).map_err(|e| format!("write {}: {e}", path.display()))?;
    eprintln!("  model remote seed: {url} ({model})");
    Ok(vec![
        "-fw_cfg".into(),
        format!("name=opt/chitti/model,file={}", path.display()),
    ])
}

/// Escape a string for embedding in a JSON string value (quotes + backslashes).
fn json_escape_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if c.is_control() => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

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
                // Gemma 4 E4B instruct (unsloth Q4_K_M). Aliases cover the HF
                // repo slug style and a short form.
                "gemma-4-e4b"
                | "gemma-4-E4B"
                | "gemma4-e4b"
                | "gemma4-E4B"
                | "gemma-4-E4B-it"
                | "e4b"
                | "E4B" => Ok(Model::Gemma4E4B),
                // PrismML Bonsai-27B 1-bit (binary Q1_0) — the default.
                "bonsai-27b" | "bonsai27b" | "bonsai" | "bonsai-27b-1bit" | "bonsai-1bit" => Ok(Model::Bonsai27B),
                // PrismML Ternary-Bonsai-27B (Q2_0).
                "bonsai-27b-ternary"
                | "bonsai-ternary"
                | "ternary-bonsai-27b"
                | "Ternary-Bonsai-27B" => Ok(Model::Bonsai27BTernary),
                // Any other value is a GGUF path: the kernel discovers the
                // architecture from the file, so any family/quant works here.
                other if other.ends_with(".gguf") => {
                    let size = fs::metadata(other).map_err(|e| format!("-model {other}: {e}"))?.len();
                    // Model at 2 GiB + heap at the top of RAM: file size plus
                    // ~40% working headroom plus the 2 GiB base offset.
                    let gib = (size as f64 / (1u64 << 30) as f64 * 1.4 + 2.0).ceil() as u64;
                    Ok(Model::Custom {
                        path: Box::leak(v.clone().into_boxed_str()),
                        mem: Box::leak(format!("{}G", gib.max(3)).into_boxed_str()),
                    })
                }
                other => Err(format!(
                    "unknown -model '{other}' (expected qwen3.5-0.8b|2b|4b|9b, gemma-4-e4b, bonsai-27b, bonsai-27b-ternary, or a path to a .gguf)"
                )),
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
    // `-server`: headless server build — no framebuffer console / GUI tools
    // (kernel `server` feature). Applied to build/run/image via the env below.
    if rest.iter().any(|a| a == "-server" || a == "--server") {
        env::set_var("CHITTI_SERVER", "1");
    }
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
        // `--no-model` / `-server` images ship kernel-only (no GGUF module);
        // server also flips the kernel `server` feature via CHITTI_SERVER.
        "image" => match arch {
            Arch::Aarch64 => image_aarch64(model, no_model),
            Arch::X86_64 => image(release, model, no_model),
        },
        "run" => cmd_run(release, arch, model, uefi, disk_only, fresh_disk, disk_size, no_model),
        // Package the aarch64 kernel as a gzip'd arm64 `Image` and (if the m1n1
        // proxy + machine DTB are configured via env) boot it on a tethered
        // Apple Silicon Mac. See `cmd_m1n1`.
        "m1n1" => cmd_m1n1(release),
        "test" => cmd_test(),
        "voice-assets" => cmd_voice_assets(),
        "wifi-assets" => cmd_wifi_assets(),
        // Phase 3 parity gate: build the kernel with the `refcheck` feature,
        // boot the real model, run the acceptance checks, exit pass/fail.
        "ref-check" => cmd_ref_check(arch, model),
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
    "usage: cargo xtask <build|image|run|m1n1|test|ref-check|voice-assets|wifi-assets> [-arch x86_64|aarch64] \
     [-model qwen3.5-0.8b|2b|4b|9b|gemma-4-e4b|bonsai-27b|bonsai-27b-ternary] [--release] [--uefi] [-server]\n\
     wifi-assets: extract Apple FullMAC firmware from macOS into assets/wifi/ (for /wifi load).\n\
     m1n1 (aarch64): package the kernel as a gzip'd arm64 Image and boot it on a \
     tethered Apple Silicon Mac over the m1n1 USB proxy; configure via env \
     CHITTI_M1N1/CHITTI_DTB[/CHITTI_INITRD/CHITTI_BOOTARGS/M1N1DEVICE].\n\
     run flags (x86_64): --disk <2G|1500M> size the virtio-blk disk for /install; \
     --disk-only boot the installed disk via UEFI with no ISO; --fresh-disk wipe it first; \
     --no-model build/boot without a model module (also works with `image`).\n\
     -server: headless kernel (serial only; no framebuffer/GUI tools).\n\
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
    let mut feats: Vec<&str> = features.to_vec();
    if env::var("CHITTI_SERVER").is_ok() {
        feats.push("server");
    }
    let features = &feats[..];
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

/// Locate an `objcopy` that can emit a flat binary from an aarch64 ELF. Prefers
/// `llvm-objcopy` / `rust-objcopy` (arch-neutral, usually already present with
/// the Rust toolchain), then GNU cross binutils. Returns the program name.
fn find_objcopy() -> Result<String, String> {
    let candidates = [
        "llvm-objcopy",
        "rust-objcopy",
        "aarch64-linux-gnu-objcopy",
        "aarch64-elf-objcopy",
        "gobjcopy",
        "objcopy",
    ];
    for c in candidates {
        if Command::new(c).arg("--version").output().map(|o| o.status.success()).unwrap_or(false) {
            return Ok(c.to_string());
        }
    }
    // The `llvm-tools` rustup component ships `llvm-objcopy` inside the toolchain
    // sysroot rather than on PATH — search there before giving up.
    if let Ok(out) = Command::new("rustc").args(["--print", "sysroot"]).output() {
        if out.status.success() {
            let sysroot = String::from_utf8_lossy(&out.stdout);
            let root = Path::new(sysroot.trim());
            if let Ok(rd) = std::fs::read_dir(root.join("lib/rustlib")) {
                for e in rd.flatten() {
                    let cand = e.path().join("bin/llvm-objcopy");
                    if cand.exists() {
                        return Ok(cand.to_string_lossy().into_owned());
                    }
                }
            }
        }
    }
    Err("no objcopy found (install llvm-tools: `rustup component add llvm-tools` \
         then use llvm-objcopy, or a GNU aarch64 binutils)"
        .to_string())
}

/// Sanity-check that a flattened image begins with a valid arm64 `Image` header:
/// the "ARM\x64" magic (little-endian 0x644d5241) at offset 0x38, and a nonzero
/// `code0` (the branch past the header). Guards against a silent objcopy layout
/// change that would make m1n1 reject the payload.
fn verify_image_header(img: &Path) -> Result<(), String> {
    let bytes = std::fs::read(img).map_err(|e| format!("read {}: {e}", img.display()))?;
    if bytes.len() < 64 {
        return Err(format!("{} is only {} bytes — not an arm64 Image", img.display(), bytes.len()));
    }
    let magic = u32::from_le_bytes([bytes[0x38], bytes[0x39], bytes[0x3a], bytes[0x3b]]);
    if magic != 0x644d_5241 {
        return Err(format!("{}: bad arm64 Image magic {:#010x} at 0x38 (expected 0x644d5241)", img.display(), magic));
    }
    if bytes[..4] == [0, 0, 0, 0] {
        return Err(format!("{}: code0 is zero — the header branch is missing", img.display()));
    }
    Ok(())
}

/// `cargo xtask m1n1`: build the aarch64 kernel, flatten it to an arm64 `Image`
/// (the boot header lives at offset 0), gzip it, and — if the m1n1 proxy and a
/// machine device tree are configured — boot it on a tethered Apple Silicon Mac
/// over the m1n1 USB proxy (the ~7 s dev loop). Everything is driven by env so
/// the custom arg parser stays untouched:
///
///   CHITTI_M1N1      path to your m1n1 checkout (uses proxyclient/tools/linux.py)
///   CHITTI_DTB       machine device tree (e.g. apple/t8112-j473.dtb) — required to boot
///   CHITTI_INITRD    optional initramfs / model blob (Stage 1: the GGUF)
///   CHITTI_BOOTARGS  optional kernel bootargs (e.g. "chitti.epoch=1752345600")
///   CHITTI_M1N1_TTY  secondary UART tty for the payload's console after handoff
///                    (the `_03` device, e.g. /dev/cu.usbmodemXXXXD3); without it
///                    linux.py reads the dead proxy device and shows nothing.
///   M1N1DEVICE       proxy control TTY (the `_01` device; read by linux.py itself)
///
/// Without CHITTI_M1N1 + CHITTI_DTB it just builds the Image and prints the exact
/// command to run, so the artifact is always produced.
fn cmd_m1n1(release: bool) -> Result<(), String> {
    let elf = build_kernel_aarch64(release, &[])?;
    let img = elf.with_extension("Image");
    let objcopy = find_objcopy()?;
    run(Command::new(&objcopy).args(["-O", "binary", "--strip-all"]).arg(&elf).arg(&img))?;
    verify_image_header(&img)?;
    let img_len = std::fs::metadata(&img).map(|m| m.len()).unwrap_or(0);

    // gzip (m1n1 auto-detects; convention is a compressed payload). Fall back to
    // the raw Image if `gzip` is unavailable.
    let gz = elf.with_extension("Image.gz");
    let payload = if Command::new("gzip").arg("--version").output().map(|o| o.status.success()).unwrap_or(false) {
        let out = Command::new("gzip")
            .args(["-n", "-9", "-c"])
            .arg(&img)
            .output()
            .map_err(|e| format!("gzip: {e}"))?;
        if !out.status.success() {
            return Err("gzip failed".to_string());
        }
        std::fs::write(&gz, &out.stdout).map_err(|e| format!("write {}: {e}", gz.display()))?;
        gz.clone()
    } else {
        eprintln!("xtask: gzip not found; using the uncompressed Image (linux.py handles it)");
        img.clone()
    };
    println!("m1n1: arm64 Image {} ({img_len} bytes); payload {}", img.display(), payload.display());

    let m1n1 = env::var("CHITTI_M1N1").ok();
    let dtb = env::var("CHITTI_DTB").ok();
    match (m1n1, dtb) {
        (Some(m1n1), Some(dtb)) => {
            let linuxpy = Path::new(&m1n1).join("proxyclient/tools/linux.py");
            if !linuxpy.exists() {
                return Err(format!("CHITTI_M1N1 set but {} not found", linuxpy.display()));
            }
            // The proxyclient needs `construct` + `pyserial`; the host `python3`
            // is often a PEP-668 externally-managed (uv/Homebrew) interpreter
            // where those aren't installed. Prefer a venv at
            // `$CHITTI_M1N1/.venv/bin/python` (see the m1n1 setup steps), then
            // `$CHITTI_M1N1_PYTHON`, else fall back to `python3`.
            let venv_py = Path::new(&m1n1).join(".venv/bin/python");
            let python = env::var("CHITTI_M1N1_PYTHON")
                .ok()
                .filter(|p| !p.is_empty())
                .unwrap_or_else(|| {
                    if venv_py.exists() {
                        venv_py.to_string_lossy().into_owned()
                    } else {
                        "python3".to_string()
                    }
                });
            // Hypervisor boot (CHITTI_M1N1_HV=1): a bare linux.py boot tears down
            // the USB console at handoff (both the `_01` proxy and `_03` UART
            // bridge go away), so nothing is visible. Instead do what
            // run_guest_kernel.sh does — a *nested* m1n1: concatenate a fresh
            // m1n1 + our dtb + Image.gz (+ optional initramfs) into one guest
            // blob, chainload a fresh m1n1 as the resident **hypervisor** (which
            // keeps the USB console alive and traps/forwards the guest UART),
            // then run the combined blob as the guest. The inner m1n1 does the
            // FDT prep and boots ChittiOS; the outer m1n1 lets us SEE it.
            // Optional host-side capture of the forwarded serial console to a
            // logfile (CHITTI_SERIAL_LOG=<path>), so driver bring-up is readable
            // after the fact without a human watching the framebuffer.
            let serial_log = env::var("CHITTI_SERIAL_LOG").ok().filter(|p| !p.is_empty());
            let serial_log = serial_log.as_deref().map(Path::new);
            let hv = env::var("CHITTI_M1N1_HV").map(|v| v == "1" || v == "true").unwrap_or(false);
            if hv {
                let m1n1bin = Path::new(&m1n1).join("build/m1n1.bin");
                let chainload = Path::new(&m1n1).join("proxyclient/tools/chainload.py");
                let runguest = Path::new(&m1n1).join("proxyclient/tools/run_guest.py");
                for p in [&m1n1bin, &chainload, &runguest] {
                    if !p.exists() {
                        return Err(format!("CHITTI_M1N1_HV set but {} missing (build m1n1 first)", p.display()));
                    }
                }
                // Combined guest image: m1n1 + [bootargs line] + dtb + Image.gz [+ initramfs].
                let mut buf = std::fs::read(&m1n1bin).map_err(|e| format!("read {}: {e}", m1n1bin.display()))?;
                if let Ok(ba) = env::var("CHITTI_BOOTARGS") {
                    if !ba.is_empty() {
                        buf.extend_from_slice(format!("chosen.bootargs={ba}\n").as_bytes());
                    }
                }
                buf.extend_from_slice(&std::fs::read(&dtb).map_err(|e| format!("read {dtb}: {e}"))?);
                buf.extend_from_slice(&std::fs::read(&payload).map_err(|e| format!("read {}: {e}", payload.display()))?);
                if let Ok(initrd) = env::var("CHITTI_INITRD") {
                    let data = std::fs::read(&initrd).map_err(|e| format!("read {initrd}: {e}"))?;
                    buf.extend_from_slice(b"m1n1_initramfs");
                    buf.extend_from_slice(&(data.len() as u32).to_le_bytes());
                    buf.extend_from_slice(&data);
                }
                let combined = elf.with_extension("m1n1-guest.bin");
                std::fs::write(&combined, &buf).map_err(|e| format!("write {}: {e}", combined.display()))?;
                println!("m1n1(hv): combined guest image {} ({} bytes)", combined.display(), buf.len());
                println!("m1n1(hv): chainloading a fresh m1n1 as the resident hypervisor…");
                run(Command::new(&python).arg(&chainload).arg("-r").arg(&m1n1bin))?;
                println!("m1n1(hv): starting the ChittiOS guest under the hypervisor (console stays live)…");
                return run_tee(Command::new(&python).arg(&runguest).arg("-r").arg(&combined), serial_log);
            }

            let mut c = Command::new(&python);
            c.arg(&linuxpy).arg(&payload).arg(&dtb);
            if let Ok(initrd) = env::var("CHITTI_INITRD") {
                c.arg(initrd);
            }
            if let Ok(bootargs) = env::var("CHITTI_BOOTARGS") {
                c.args(["-b", &bootargs]);
            }
            // After handoff m1n1 tears down the proxy (`_01`) interface, so the
            // payload's console must be read on the secondary UART bridge
            // (`_03`, e.g. /dev/cu.usbmodemXXXX**D3**). Pass it via CHITTI_M1N1_TTY
            // so linux.py's post-boot `ttymode` reads there and we see ChittiOS's
            // s5l output instead of a dead proxy device.
            if let Ok(tty) = env::var("CHITTI_M1N1_TTY") {
                if !tty.is_empty() {
                    c.args(["-t", &tty]);
                }
            }
            println!("m1n1: booting over the proxy ({:?})…", c);
            run_tee(&mut c, serial_log)
        }
        _ => {
            println!(
                "m1n1: to boot on hardware, set CHITTI_M1N1 (+ CHITTI_DTB, optional \
                 CHITTI_INITRD/CHITTI_BOOTARGS, M1N1DEVICE; CHITTI_SERIAL_LOG=<path> \
                 tees the serial console to a logfile), or run manually:\n  \
                 M1N1DEVICE=/dev/cu.usbmodemXXX <m1n1>/.venv/bin/python \
                 <m1n1>/proxyclient/tools/linux.py \
                 {} <machine.dtb> [initramfs] -b \"chitti.epoch=<unix-secs>\"\n  \
                 (create the venv once: `uv venv <m1n1>/.venv && uv pip install \
                 --python <m1n1>/.venv/bin/python -r <m1n1>/requirements.txt`)",
                payload.display()
            );
            Ok(())
        }
    }
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
    // The stub pins *stable* via `stub/rust-toolchain.toml` (nightly is only
    // for the kernel's `-Z build-std`). Building with `current_dir = stub/`
    // makes rustup pick that file over the repo-root nightly. Ensure the
    // tier-2 UEFI target is installed on stable (idempotent).
    let status = Command::new("rustup")
        .args(["target", "add", "aarch64-unknown-uefi", "--toolchain", "stable"])
        .status()
        .map_err(|e| format!("rustup target add aarch64-unknown-uefi (stable): {e}"))?;
    if !status.success() {
        eprintln!(
            "xtask: warning: rustup target add aarch64-unknown-uefi --toolchain stable \
             failed ({status}); stub build will surface the real error if missing"
        );
    }
    let sdir = repo_root().join("stub");
    let mut cmd = Command::new("cargo");
    // Do not pass `+stable` explicitly: the directory's rust-toolchain.toml
    // selects it. Forcing `+nightly` here would reintroduce the old "nightly
    // without the UEFI target" failure mode.
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
/// Returns **physical pixels** — the same thing VirtualBox's GOP hands the
/// stub (native panel mode). The QEMU window runs `zoom-to-fit=on`, so a
/// full-resolution guest FB scales down cleanly when windowed and is 1:1 in
/// fullscreen — identical rendering across VBox / QEMU / laptop / monitor
/// (halving Retina + subtracting chrome made QEMU pick a different, smaller
/// mode than VBox on the same display, so fonts/layout looked different).
fn detect_host_res() -> Option<(u32, u32)> {
    if !cfg!(target_os = "macos") {
        return None; // Linux hosts: use the default (or CHITTI_FBRES)
    }
    let out = Command::new("system_profiler").arg("SPDisplaysDataType").output().ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    for line in text.lines() {
        let Some(rest) = line.trim().strip_prefix("Resolution:") else { continue };
        // e.g. "3456 x 2234 Retina" or "2560 x 1440"
        let parts: Vec<&str> = rest.split_whitespace().collect();
        if parts.len() >= 3 && parts[1] == "x" {
            if let (Ok(w), Ok(h)) = (parts[0].parse::<u32>(), parts[2].parse::<u32>()) {
                return Some((w.clamp(1024, 3840), h.clamp(720, 2160)));
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
    let (code, vars_src) = if cfg!(target_os = "macos") {
        let share = brew_prefix("qemu")?.join("share/qemu");
        (share.join("edk2-aarch64-code.fd"), share.join("edk2-arm-vars.fd"))
    } else {
        let code = find_path(
            "CHITTI_AAVMF_CODE",
            &["/usr/share/AAVMF/AAVMF_CODE.fd", "/usr/share/qemu-efi-aarch64/QEMU_EFI.fd", "/usr/share/qemu/edk2-aarch64-code.fd"],
        );
        let vars = find_path("CHITTI_AAVMF_VARS", &["/usr/share/AAVMF/AAVMF_VARS.fd", "/usr/share/qemu/edk2-arm-vars.fd"]);
        match (code, vars) {
            (Some(c), Some(v)) => (c, v),
            _ => return Err("AAVMF firmware not found (install qemu-efi-aarch64, or set CHITTI_AAVMF_CODE/CHITTI_AAVMF_VARS)".into()),
        }
    };
    if !code.exists() || !vars_src.exists() {
        return Err(format!("AAVMF firmware not found ({} / {})", code.display(), vars_src.display()));
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
/// Run a host command, failing with a readable error.
fn run_host(prog: &str, args: &[&str]) -> Result<(), String> {
    let st = Command::new(prog).args(args).status().map_err(|e| format!("running {prog}: {e}"))?;
    if !st.success() {
        return Err(format!("{prog} {} failed: {st}", args.join(" ")));
    }
    Ok(())
}

/// Copy `part`'s bytes into `img` starting at byte `offset` (how the Linux
/// image path populates GPT partitions: format a temp file, splice it in).
fn splice_into(img: &Path, offset: u64, part: &Path) -> Result<(), String> {
    use std::io::{Read, Seek, SeekFrom, Write};
    let mut src = fs::File::open(part).map_err(|e| e.to_string())?;
    let mut dst = fs::OpenOptions::new().write(true).open(img).map_err(|e| e.to_string())?;
    dst.seek(SeekFrom::Start(offset)).map_err(|e| e.to_string())?;
    let mut buf = vec![0u8; 8 * 1024 * 1024];
    loop {
        let n = src.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        dst.write_all(&buf[..n]).map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Populate a plain FAT32 image file with `(host_path, image_path)` copies —
/// Linux path via dosfstools/mtools (no root needed). `dirs` are `::/`-style
/// mtools directories created first.
fn populate_fat_linux(img: &Path, label: &str, dirs: &[&str], copies: &[(PathBuf, String)]) -> Result<(), String> {
    let img_s = img.to_string_lossy().to_string();
    run_host("mkfs.vfat", &["-F", "32", "-n", label, &img_s])?;
    for d in dirs {
        run_host("mmd", &["-i", &img_s, d])?;
    }
    for (src, dst) in copies {
        run_host("mcopy", &["-i", &img_s, &src.to_string_lossy(), dst])?;
    }
    Ok(())
}

fn build_esp_image_aarch64(stub: &Path, kernel: &Path, model: Option<&Path>) -> Result<PathBuf, String> {
    let img = repo_root().join("target/chitti-esp-aa64.img");
    let model_bytes = model.map(|m| fs::metadata(m).map(|md| md.len()).unwrap_or(0)).unwrap_or(0);
    // Voice models + WiFi firmware bundled onto the ESP too (kernel mounts the
    // FAT ESP and reads them). Grow the image to fit them.
    let voice: Vec<(String, PathBuf)> = voice_model_assets().into_iter().filter(|(_, p)| p.exists()).map(|(n, p)| (n.to_string(), p)).collect();
    let voice_bytes: u64 = voice.iter().map(|(_, p)| fs::metadata(p).map(|m| m.len()).unwrap_or(0)).sum();
    let wifi = wifi_fw_assets();
    let wifi_bytes: u64 = wifi.iter().map(|(_, p)| fs::metadata(p).map(|m| m.len()).unwrap_or(0)).sum();
    let size_mb = 64
        + ((model_bytes + voice_bytes + wifi_bytes) / (1024 * 1024))
        + if model_bytes > 0 { 64 } else { 0 }
        + if wifi_bytes > 0 { 8 } else { 0 };
    // Recreate the image only when contents changed (cheap heuristic: sizes).
    let f = fs::OpenOptions::new().create(true).write(true).truncate(true).open(&img).map_err(|e| e.to_string())?;
    f.set_len(size_mb * 1024 * 1024).map_err(|e| e.to_string())?;
    drop(f);
    if cfg!(target_os = "linux") {
        let mut copies: Vec<(PathBuf, String)> = vec![
            (stub.to_path_buf(), "::/EFI/BOOT/BOOTAA64.EFI".into()),
            (kernel.to_path_buf(), "::/chitti-kernel".into()),
        ];
        if let Some(m) = model {
            for (src, name) in esp_model_parts(m)? {
                copies.push((src, format!("::/{name}")));
            }
        }
        for (n, pth) in &voice {
            copies.push((pth.clone(), format!("::/{n}")));
        }
        for (n, pth) in &wifi {
            copies.push((pth.clone(), format!("::/{n}")));
        }
        let dirs: &[&str] = if wifi.is_empty() {
            &["::/EFI", "::/EFI/BOOT"]
        } else {
            &["::/EFI", "::/EFI/BOOT", "::/brcm"]
        };
        populate_fat_linux(&img, "CHITTI", dirs, &copies)?;
        eprintln!(
            "  ESP image: {} ({} MiB{}{})",
            img.display(),
            size_mb,
            if model.is_some() { ", model bundled" } else { "" },
            if !wifi.is_empty() { ", wifi fw" } else { "" }
        );
        return Ok(img);
    }
    // macOS: attach raw, format FAT32, mount, copy, detach — via /bin/sh.
    let model_cp = match model {
        Some(m) => esp_model_parts(m)?
            .into_iter()
            .map(|(src, name)| format!("cp \"{}\" \"$MNT/{}\"", src.display(), name))
            .collect::<Vec<_>>()
            .join("\n"),
        None => String::new(),
    };
    let wifi_cp = if wifi.is_empty() {
        String::new()
    } else {
        let mut s = String::from("mkdir -p \"$MNT/brcm\"\n");
        for (n, p) in &wifi {
            s.push_str(&format!("cp \"{}\" \"$MNT/{}\"\n", p.display(), n));
        }
        s
    };
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
{wifi_cp}
diskutil unmount "$DEV" > /dev/null
hdiutil detach "$DEV" > /dev/null
"#,
        img = img.display(),
        stub = stub.display(),
        kernel = kernel.display(),
        model_cp = model_cp,
        // Voice models at the ESP root, so the kernel's root-file readers find them.
        voice_cp = voice.iter().map(|(n, p)| format!("cp \"{}\" \"$MNT/{}\"", p.display(), n)).collect::<Vec<_>>().join("\n"),
        wifi_cp = wifi_cp,
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
/// `None` if no voice assets are present, or if `CHITTI_VOICE_DISK=off`
/// (QEMU write-locks the image, so a second concurrent guest — an e2e
/// scenario booting its own — must skip it or fail to launch).
fn build_voice_disk() -> Option<PathBuf> {
    if env::var("CHITTI_VOICE_DISK").map(|v| v.trim() == "off").unwrap_or(false) {
        return None;
    }
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
    if cfg!(target_os = "linux") {
        let copies: Vec<(PathBuf, String)> = voice.iter().map(|(n, p)| (p.clone(), format!("::/{n}"))).collect();
        if populate_fat_linux(&img, "VOICE", &[], &copies).is_err() {
            return None;
        }
        let _ = fs::write(&meta, &want);
        eprintln!("  voice disk: {} ({} MiB, {} model(s))", img.display(), size_mb, voice.len());
        return Some(img);
    }
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

/// Build a small FAT "model disk" holding the given files — a `.gguf` source
/// lands as `chat.gguf` (the `/model load` runtime-loading path; the e2e
/// model_load scenario), any other file keeps its own name (the e2e `/open`
/// image/audio scenario ships media this way). Opt-in via
/// `CHITTI_MODEL_DISK=<path>[:<path>...]`. Cached; rebuilt when the sources
/// change. FAT32 caps a file at 4 GiB — larger models must be loaded from an
/// ext4 data disk instead.
fn build_model_disk(files: &[PathBuf]) -> Result<PathBuf, String> {
    let mut total = 0u64;
    let mut copies: Vec<(PathBuf, String)> = Vec::new();
    for f in files {
        let size = fs::metadata(f).map_err(|e| format!("CHITTI_MODEL_DISK {}: {e}", f.display()))?.len();
        if size >= 4 * (1u64 << 30) {
            return Err("CHITTI_MODEL_DISK: FAT32 caps a file at 4 GiB; use an ext4 data disk for larger models".into());
        }
        total += size;
        let is_gguf = f.extension().and_then(|e| e.to_str()).map(|e| e.eq_ignore_ascii_case("gguf")).unwrap_or(false);
        let dest = if is_gguf {
            "chat.gguf".to_string()
        } else {
            f.file_name().and_then(|n| n.to_str()).unwrap_or("file.bin").to_string()
        };
        copies.push((f.clone(), dest));
    }
    let img = repo_root().join("target/chitti-model-disk.img");
    let want = copies.iter().map(|(p, d)| format!("{}>{d}", p.display())).collect::<Vec<_>>().join(":") + &format!(":{total}");
    let meta = repo_root().join("target/chitti-model-disk.meta");
    if img.exists() && fs::read_to_string(&meta).map(|s| s.trim() == want).unwrap_or(false) {
        return Ok(img);
    }
    let size_mb = 64 + total / (1024 * 1024) + 32;
    let f = fs::OpenOptions::new().create(true).write(true).truncate(true).open(&img).map_err(|e| e.to_string())?;
    f.set_len(size_mb * 1024 * 1024).map_err(|e| e.to_string())?;
    drop(f);
    if cfg!(target_os = "linux") {
        let cps: Vec<(PathBuf, String)> = copies.iter().map(|(p, d)| (p.clone(), format!("::/{d}"))).collect();
        populate_fat_linux(&img, "MODEL", &[], &cps)?;
    } else {
        let cps = copies.iter().map(|(p, d)| format!("cp \"{}\" \"$MNT/{}\"", p.display(), d)).collect::<Vec<_>>().join("\n");
        let script = format!(
            r#"set -e
DEV=$(hdiutil attach -nomount -imagekey diskimage-class=CRawDiskImage "{img}" | head -1 | awk '{{print $1}}')
newfs_msdos -F 32 -v MODEL "$DEV" > /dev/null
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
            return Err("model disk build failed (hdiutil/newfs_msdos)".into());
        }
    }
    let _ = fs::write(&meta, &want);
    let names = copies.iter().map(|(_, d)| d.as_str()).collect::<Vec<_>>().join(", ");
    eprintln!("  model disk: {} ({} MiB, {names})", img.display(), size_mb);
    Ok(img)
}

/// The `CHITTI_MODEL_DISK` FAT disk, if requested via the environment
/// (colon-separated paths land on one disk).
fn model_disk_from_env() -> Result<Option<PathBuf>, String> {
    match env::var("CHITTI_MODEL_DISK") {
        Ok(p) if !p.trim().is_empty() => {
            let files: Vec<PathBuf> = p.split(':').map(|s| PathBuf::from(s.trim())).filter(|s| s.as_os_str().len() > 0).collect();
            Ok(Some(build_model_disk(&files)?))
        }
        _ => Ok(None),
    }
}

/// Resolve the GGUF for a `run` (not `image`): require the file unless the
/// caller passed `--no-model`. Silent model-less boots after `-model e4b` are
/// how users end up with "no model bundled -- chat unavailable".
fn require_gguf_for_run(model: Model, no_model: bool) -> Result<Option<PathBuf>, String> {
    if no_model {
        eprintln!("--no-model: booting without a model (`/model load` or `/model remote` can add one)");
        return Ok(None);
    }
    let gguf = repo_root().join(model.gguf_rel());
    if gguf.exists() {
        return Ok(Some(gguf));
    }
    Err(format!(
        "{} is not present — cannot boot chat/infer for `-model {}`.\n\
         \n\
           ./xtask/fetch-model.sh {}\n\
           # or: make model MODEL={}\n\
         \n\
         Then re-run. To intentionally boot without a model, pass --no-model.",
        model.gguf_rel(),
        model.label(),
        model.label(),
        model.label(),
    ))
}

/// Boot aarch64 via UEFI firmware (AAVMF) — the Chitti stub loads the normal
/// identity kernel off a real FAT ESP and hands off via an identity-RAM
/// trampoline. **Models are injected with QEMU `-device loader`** at
/// [`Model::aarch64_addr`] (same as the `-kernel` path), not reassembled by
/// the stub from the ESP: multi-GiB GGUFs (e4b / 9b) need a multi-GiB
/// contiguous UEFI `LOADER_DATA` allocation that AAVMF often cannot provide,
/// which used to boot a model-less kernel even when the GGUF was on disk.
///
/// The distributable `image -arch aarch64` still puts the model on the ESP for
/// real hardware; this `run --uefi` path optimises for QEMU correctness.
fn cmd_run_aarch64_uefi(model: Model, disk: Option<PathBuf>, disk_only: bool, no_model: bool) -> Result<(), String> {
    let elf = build_kernel_aarch64(true, model.features())?;
    assert_identity_kernel(&elf)?;
    let stub = build_stub_aarch64()?;
    let mut qemu = Command::new("qemu-system-aarch64");
    qemu.args(["-M", "virt", "-smp", &smp_count(), "-m", model.qemu_mem()]);
    qemu.args(accel_args("aarch64"));
    for a in aavmf_pflash_args()? {
        qemu.arg(a);
    }
    qemu.args(["-device", "ramfb", "-device", "virtio-keyboard-device", "-device", "virtio-tablet-device"]);
    qemu.args(display_args());
    // Same host-derived framebuffer resolution as the `-kernel` path, so the
    // UEFI ramfb fallback matches VBox GOP / QEMU direct (was 1920x1080 fixed).
    for a in ramfb_res_fw_cfg() {
        qemu.arg(a);
    }
    qemu.args(["-serial", "mon:stdio"]);
    // ESP first, data disk LAST: QEMU assigns later virtio-mmio devices to
    // LOWER slots, and the kernel's probe_disk takes the first (lowest) match —
    // so this ordering makes /install + persistence target the data disk, never
    // the boot ESP.
    if disk_only {
        eprintln!("booting aarch64 FROM DISK ONLY via UEFI (no ESP medium)");
        eprintln!("  note: requires an /install that wrote the aarch64 ESP payload to the disk");
    } else {
        // Kernel + stub only on the ESP (fast rebuild). Model comes via loader.
        let esp = build_esp_image_aarch64(&stub, &elf, None)?;
        qemu.arg("-drive").arg(format!("file={},if=none,id=esp,format=raw", esp.display()));
        qemu.args(["-device", "virtio-blk-device,drive=esp"]);
        eprintln!("booting aarch64 via the Chitti UEFI stub (AAVMF) -- firmware loads BOOTAA64.EFI from the ESP");
    }
    // Same QEMU-loader placement as `-kernel`: guest phys MODEL_LOAD_ADDR.
    // Large models are chunked on Darwin (see `model_loader_args`).
    if let Some(gguf) = require_gguf_for_run(model, no_model)? {
        let base = u64::from_str_radix(model.aarch64_addr().trim_start_matches("0x"), 16)
            .map_err(|e| format!("bad model addr {}: {e}", model.aarch64_addr()))?;
        for arg in model_loader_args(&gguf, base)? {
            qemu.arg("-device").arg(arg);
        }
        eprintln!(
            "  model via QEMU loader: {} at guest phys {} ({} — not via ESP/stub reassembly)",
            model.label(),
            model.aarch64_addr(),
            gguf.display()
        );
    }
    if let Some(d) = &disk {
        qemu.arg("-drive").arg(format!("file={},if=none,id=data,format=raw", d.display()));
        qemu.args(["-device", "virtio-blk-device,drive=data"]);
        eprintln!("  data disk: {}", d.display());
    }
    run(&mut qemu)
}

fn cmd_run_aarch64(release: bool, model: Model, disk: Option<PathBuf>, _disk_only: bool, no_model: bool) -> Result<(), String> {
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
    // 0.8B, 12G for 9B). `-smp N` (CHITTI_SMP, default 8): N vCPUs, which under
    // `-accel hvf` run on *native* M-series cores in parallel (unlike TCG,
    // where extra vCPUs only contend). Chitti's aarch64 SMP brings the
    // secondaries up via PSCI and splits the hot matvec/matmul across them.
    // `-device ramfb`: a simple linear framebuffer the kernel configures via
    // fw_cfg (arch::aarch64::ramfb) and renders the TUI into — the aarch64
    // equivalent of the x86 Limine framebuffer. Dropping `-nographic` lets QEMU
    // open its display window; `-serial mon:stdio` keeps the serial console (and
    // QEMU monitor) on stdio so you still type at the terminal (Ctrl-A X quits,
    // Ctrl-A C for the monitor).
    qemu.args(["-M", "virt", "-smp", &smp_count(), "-m", model.qemu_mem()]);
    qemu.args(accel_args("aarch64"));
    qemu.args([
        "-device", "ramfb", "-device", "virtio-keyboard-device",
        // A virtio tablet gives the window an absolute-position mouse.
        "-device", "virtio-tablet-device",
    ]);
    // Resizable graphical window (the ramfb surface scales to fit).
    qemu.args(display_args());
    qemu.args(["-serial", "mon:stdio", "-kernel"]);
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
    // Optional boot seed for /model remote (CHITTI_REMOTE_URL / _MODEL / _KEY).
    for a in remote_model_fw_cfg()? {
        qemu.arg(a);
    }
    // A virtio-net NIC: slirp user-net by default (DHCP 10.0.2.15 / gw 10.0.2.2),
    // or L2-bridged onto a host iface when CHITTI_NET_BRIDGE=<ifname> is set.
    // CHITTI_HOSTFWD only applies to user-net. Host services (LM Studio, …)
    // are reached at 10.0.2.2 under user-net.
    // CHITTI_NIC overrides the transport: virtio-net-device is the virtio-mmio
    // NIC the aarch64 driver prefers; naming a PCI model instead (e1000e, igb,
    // ...) puts it on the GPEX PCIe bus.
    //
    // NB this only finds the NIC on the **UEFI-stub** boot: `crate::pci` gets its
    // ECAM base from the ACPI MCFG in the stub's boot-info page, so on the plain
    // `-kernel` HVF path here there is no PCIe bus to enumerate and a PCI NIC is
    // invisible. Use it with an image/OVMF boot, not `xtask run -arch aarch64`.
    let a64_nic = std::env::var("CHITTI_NIC").unwrap_or_else(|_| "virtio-net-device".to_string());
    qemu.args(["-netdev", &guest_netdev("chittinet"), "-device", &format!("{a64_nic},netdev=chittinet")]);
    // virtio-snd on the host's audio backend (mic + speaker) for /voice.
    qemu.args(audio_args("virtio-sound-device"));
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
    // Opt-in FAT disk carrying a GGUF as `chat.gguf` for the runtime `/model
    // load` path (CHITTI_MODEL_DISK=<path>; the e2e model_load scenario).
    if let Some(md) = model_disk_from_env()? {
        qemu.arg("-drive").arg(format!("file={},if=none,id=modeldisk,format=raw", md.display()));
        qemu.args(["-device", "virtio-blk-device,drive=modeldisk"]);
        eprintln!("  model disk attached (chat.gguf; load with /model load chat.gguf)");
    }
    // Place the GGUF in guest RAM at the model's load address (where the aarch64
    // `cortex::model_module` looks) -- the equivalent of the x86 Limine boot
    // module, so `infer` works natively. `--no-model` skips it (e.g. to prove
    // the runtime `/model load` path starts from nothing).
    if let Some(gguf) = require_gguf_for_run(model, no_model)? {
        let base = u64::from_str_radix(model.aarch64_addr().trim_start_matches("0x"), 16)
            .map_err(|e| format!("bad model addr {}: {e}", model.aarch64_addr()))?;
        for arg in model_loader_args(&gguf, base)? {
            qemu.arg("-device").arg(arg);
        }
        eprintln!("attaching {} at guest phys {}", model.gguf_rel(), model.aarch64_addr());
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

/// Spawn `cmd`, mirroring its stdout to BOTH this terminal and (when `log` is
/// `Some`) an append-to-fresh logfile — so a Mac-mini boot's forwarded serial
/// console is captured to a host file an agent can read afterwards. This is what
/// makes bare-metal driver bring-up self-serve: add `ktrace`/`serial_println!`
/// lines to a driver, boot once, and read the log instead of a human relaying
/// the framebuffer. Works for both the m1n1-hypervisor path (m1n1 forwards the
/// guest VUART to stdout) and the bare `linux.py -t <D3>` path (the payload's
/// own s5l UART, bridged over the `_03` USB serial device). stdin/stderr are
/// inherited, so serial input + Python errors still flow. `PYTHONUNBUFFERED`
/// keeps linux.py from withholding lines once its stdout is a pipe, not a TTY.
fn run_tee(cmd: &mut Command, log: Option<&Path>) -> Result<(), String> {
    use std::process::Stdio;
    let Some(path) = log else { return run(cmd) };
    let mut file =
        fs::File::create(path).map_err(|e| format!("create serial log {}: {e}", path.display()))?;
    println!("xtask: capturing serial console -> {}", path.display());
    cmd.env("PYTHONUNBUFFERED", "1").stdout(Stdio::piped());
    let mut child = cmd.spawn().map_err(|e| format!("failed to spawn {cmd:?}: {e}"))?;
    let mut out = child.stdout.take().ok_or("child produced no stdout pipe")?;
    let stdout = std::io::stdout();
    let mut buf = [0u8; 4096];
    loop {
        match out.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                let mut lock = stdout.lock();
                let _ = lock.write_all(&buf[..n]);
                let _ = lock.flush();
                // Persist immediately — a hang leaves the last line already on disk.
                let _ = file.write_all(&buf[..n]);
                let _ = file.flush();
            }
            Err(e) => return Err(format!("reading serial console: {e}")),
        }
    }
    let status = child.wait().map_err(|e| format!("waiting for {cmd:?}: {e}"))?;
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
    let mut feats: Vec<&str> = features.to_vec();
    if env::var("CHITTI_SERVER").is_ok() {
        feats.push("server");
    }
    let features = &feats[..];
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

/// QEMU audio flags for `device` (virtio-sound-pci / virtio-sound-device).
/// Driver from `CHITTI_AUDIO` (coreaudio|pa|alsa|pipewire|sdl|none|off); the
/// default is the host OS's native backend. `off` omits the device entirely —
/// the escape hatch when the host backend can't open (e.g. macOS mic
/// permission denied: QEMU prints "Can not open `virtio-sound.in'" but boots
/// on with audio input dead).
fn audio_args(device: &str) -> Vec<String> {
    let drv = env::var("CHITTI_AUDIO").unwrap_or_else(|_| {
        if cfg!(target_os = "macos") { "coreaudio".into() } else { "pa".into() }
    });
    if drv == "off" {
        return Vec::new();
    }
    vec![
        "-audiodev".into(),
        format!("{drv},id=chittiaudio"),
        "-device".into(),
        format!("{device},audiodev=chittiaudio"),
    ]
}

/// QEMU display flags: Cocoa on macOS, GTK elsewhere (override: CHITTI_DISPLAY).
fn display_args() -> Vec<String> {
    let d = env::var("CHITTI_DISPLAY").unwrap_or_else(|_| {
        if cfg!(target_os = "macos") { "cocoa,zoom-to-fit=on".into() } else { "gtk,zoom-to-fit=on".into() }
    });
    vec!["-display".into(), d]
}

/// QEMU accelerator for a target arch: HVF on macOS, KVM on a same-arch Linux
/// host, else TCG (override: CHITTI_ACCEL).
fn accel_args(target_arch: &str) -> Vec<String> {
    if let Ok(a) = env::var("CHITTI_ACCEL") {
        return vec!["-accel".into(), a, "-cpu".into(), "host".into()];
    }
    if cfg!(target_os = "macos") && std::env::consts::ARCH == target_arch {
        return vec!["-accel".into(), "hvf".into(), "-cpu".into(), "host".into()];
    }
    // Same-arch Linux host: KVM only if it's actually available (a CI runner
    // may not expose /dev/kvm) — otherwise `-accel kvm` hard-fails, so fall
    // back to TCG (`-cpu max`) rather than refusing to boot.
    if cfg!(target_os = "linux") && std::env::consts::ARCH == target_arch && std::path::Path::new("/dev/kvm").exists() {
        return vec!["-accel".into(), "kvm".into(), "-cpu".into(), "host".into()];
    }
    vec!["-cpu".into(), "max".into()]
}

/// First existing path among `candidates`, with an env-var override.
fn find_path(env_key: &str, candidates: &[&str]) -> Option<PathBuf> {
    if let Ok(p) = env::var(env_key) {
        let p = PathBuf::from(p);
        if p.exists() {
            return Some(p);
        }
    }
    candidates.iter().map(PathBuf::from).find(|p| p.exists())
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
    if cfg!(target_os = "macos") {
        return Ok(brew_prefix("limine")?.join("share/limine"));
    }
    find_path("CHITTI_LIMINE_SHARE", &["/usr/local/share/limine", "/usr/share/limine"])
        .ok_or_else(|| "limine share dir not found (install limine, or set CHITTI_LIMINE_SHARE)".into())
}

fn limine_bin() -> Result<PathBuf, String> {
    if let Ok(bin) = env::var("CHITTI_LIMINE_BIN") {
        return Ok(PathBuf::from(bin));
    }
    if cfg!(target_os = "macos") {
        return Ok(brew_prefix("limine")?.join("bin/limine"));
    }
    find_path("CHITTI_LIMINE_BIN", &["/usr/local/bin/limine", "/usr/bin/limine"])
        .ok_or_else(|| "limine binary not found (install limine, or set CHITTI_LIMINE_BIN)".into())
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

/// Split a bundled model into FAT32-safe parts under `target/esp-model-parts`
/// and return `(part_path, "model.gguf.NNN")` pairs to copy onto the ESP. FAT32
/// caps a single file at 4 GiB, so a large model (the 9B) cannot be copied whole
/// — it is split, and the UEFI stub concatenates the sorted parts back (as every
/// other loader path already does). A model under one part size still becomes a
/// single `model.gguf.000`, so the copy/reassembly path is uniform for all tiers.
fn esp_model_parts(model: &Path) -> Result<Vec<(PathBuf, String)>, String> {
    // 1 GiB: well under FAT32's 4 GiB file cap, and bounds the stub's per-part
    // read buffer (it reads one part at a time into a transient allocation).
    const ESP_PART: u64 = 1 << 30;
    let dir = repo_root().join("target/esp-model-parts");
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    // Drop leftover higher-index parts from a previous larger model so a later
    // boot that scanned the host dir (or a stale ESP) cannot stitch a hybrid.
    if let Ok(rd) = fs::read_dir(&dir) {
        for e in rd.flatten() {
            let n = e.file_name();
            let s = n.to_string_lossy();
            if s.starts_with("model.gguf.") {
                let _ = fs::remove_file(e.path());
            }
        }
    }
    let names = split_model_into_parts(model, &dir, ESP_PART)?;
    Ok(names.into_iter().map(|n| (dir.join(&n), n)).collect())
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

/// Apple FullMAC WiFi firmware to place on the ESP / data image as
/// `brcm/<name>`. Present only after `cargo xtask wifi-assets` (extracts from
/// macOS `/usr/share/firmware/wifi`). The kernel also embeds the `.bin` when
/// present so bare m1n1 boots can `/wifi load` without a disk.
fn wifi_fw_assets() -> Vec<(String, PathBuf)> {
    let dir = repo_root().join("assets/wifi/brcm");
    let names = [
        "brcmfmac4388-pcie.apple,miyake.bin",
        "brcmfmac4388-pcie.apple,miyake.txt",
        "brcmfmac4388-pcie.apple,miyake.clm_blob",
        "brcmfmac4387c2-pcie.apple,miyake.bin",
        "brcmfmac4387c2-pcie.apple,miyake.txt",
    ];
    names
        .into_iter()
        .map(|n| (format!("brcm/{n}"), dir.join(n)))
        .filter(|(_, p)| p.exists())
        .collect()
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
fn image_aarch64(model: Model, no_model: bool) -> Result<(), String> {
    let elf = build_kernel_aarch64(true, model.features())?;
    assert_identity_kernel(&elf)?;
    let stub = build_stub_aarch64()?;
    let gguf = repo_root().join(model.gguf_rel());
    let model_path = if no_model {
        eprintln!("--no-model: building a model-less aarch64 image");
        None
    } else {
        let p = gguf.exists().then_some(gguf);
        if p.is_none() {
            eprintln!("note: {} absent -- building a model-less image", model.gguf_rel());
        }
        p
    };

    // Voice models + WiFi firmware on the ESP so `find_on_disks` auto-loads
    // them (VirtualBox/real-hardware path). Present-only.
    let voice: Vec<(String, PathBuf)> = voice_model_assets().into_iter().filter(|(_, p)| p.exists()).map(|(n, p)| (n.to_string(), p)).collect();
    let voice_bytes: u64 = voice.iter().map(|(_, p)| fs::metadata(p).map(|m| m.len()).unwrap_or(0)).sum();
    let wifi = wifi_fw_assets();
    let wifi_bytes: u64 = wifi.iter().map(|(_, p)| fs::metadata(p).map(|m| m.len()).unwrap_or(0)).sum();
    // Layout: GPT (34 + 33 reserved sectors) + ESP (payload + 64 MiB slack) +
    // 256 MiB ext4 data partition.
    let payload: u64 = fs::metadata(&elf).map(|m| m.len()).unwrap_or(0)
        + fs::metadata(&stub).map(|m| m.len()).unwrap_or(0)
        + model_path.as_ref().and_then(|p| fs::metadata(p).ok()).map(|m| m.len()).unwrap_or(0)
        + voice_bytes
        + wifi_bytes;
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

    if cfg!(target_os = "linux") {
        // Format each partition into a temp file (dosfstools/mtools + e2fsprogs,
        // no root needed), then splice the bytes into the GPT image.
        let esp_tmp = repo_root().join("target/chitti-aa64-esp.tmp");
        let f = fs::OpenOptions::new().create(true).write(true).truncate(true).open(&esp_tmp).map_err(|e| e.to_string())?;
        f.set_len(esp_secs * 512).map_err(|e| e.to_string())?;
        drop(f);
        let mut copies: Vec<(PathBuf, String)> = vec![
            (stub.clone(), "::/EFI/BOOT/BOOTAA64.EFI".into()),
            (elf.clone(), "::/chitti-kernel".into()),
        ];
        if let Some(m) = &model_path {
            for (src, name) in esp_model_parts(m)? {
                copies.push((src, format!("::/{name}")));
            }
        }
        for (n, pth) in &voice {
            copies.push((pth.clone(), format!("::/{n}")));
        }
        for (n, pth) in &wifi {
            copies.push((pth.clone(), format!("::/{n}")));
        }
        let dirs: &[&str] = if wifi.is_empty() {
            &["::/EFI", "::/EFI/BOOT"]
        } else {
            &["::/EFI", "::/EFI/BOOT", "::/brcm"]
        };
        populate_fat_linux(&esp_tmp, "CHITTI", dirs, &copies)?;
        splice_into(&img, esp.0 * 512, &esp_tmp)?;
        let _ = fs::remove_file(&esp_tmp);

        let data_tmp = repo_root().join("target/chitti-aa64-data.tmp");
        let f = fs::OpenOptions::new().create(true).write(true).truncate(true).open(&data_tmp).map_err(|e| e.to_string())?;
        f.set_len(data_secs * 512).map_err(|e| e.to_string())?;
        drop(f);
        run_host("mke2fs", &["-F", "-q", "-t", "ext4", "-b", "4096", &data_tmp.to_string_lossy()])?;
        splice_into(&img, data.0 * 512, &data_tmp)?;
        let _ = fs::remove_file(&data_tmp);

        println!("image: {} ({} MiB, GPT: ESP {}..{} + ext4 data)", img.display(), total_secs * 512 / (1024 * 1024), esp.0, esp.1);
        println!("  write it to a USB drive (dd/balenaEtcher) or attach as a VM disk -- it boots standalone via UEFI.");
        return Ok(());
    }
    // Format + populate the two partitions via macOS hdiutil/diskutil (the GPT
    // is parsed on attach, exposing slice devices s1/s2 owned by the user).
    let mke2fs = brew_prefix("e2fsprogs").map(|p| p.join("sbin/mke2fs")).ok().filter(|p| p.exists())
        .ok_or("mke2fs not found -- brew install e2fsprogs (needed to format the data partition)")?;
    let model_cp = match &model_path {
        Some(m) => esp_model_parts(m)?
            .into_iter()
            .map(|(src, name)| format!("cp \"{}\" \"$MNT/{}\"", src.display(), name))
            .collect::<Vec<_>>()
            .join("\n"),
        None => String::new(),
    };
    let wifi_cp = if wifi.is_empty() {
        String::new()
    } else {
        let mut s = String::from("mkdir -p \"$MNT/brcm\"\n");
        for (n, p) in &wifi {
            s.push_str(&format!("cp \"{}\" \"$MNT/{}\"\n", p.display(), n));
        }
        s
    };
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
{voice_cp}
{wifi_cp}
diskutil unmount "${{DEV}}s1" > /dev/null
"{mke2fs}" -F -q -t ext4 -b 4096 "${{DEV}}s2"
hdiutil detach "$DEV" > /dev/null
"#,
        img = img.display(),
        stub = stub.display(),
        kernel = elf.display(),
        mke2fs = mke2fs.display(),
        model_cp = model_cp,
        voice_cp = voice.iter().map(|(n, p)| format!("cp \"{}\" \"$MNT/{}\"", p.display(), n)).collect::<Vec<_>>().join("\n"),
        wifi_cp = wifi_cp,
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

fn image(release: bool, model: Model, no_model: bool) -> Result<(), String> {
    let bin = build_kernel_with(release, model.features())?;
    // Bundle the selected model if present; otherwise a kernel-only bootable
    // ISO (what CI ships -- the model is fetched separately, being large).
    // `--no-model` forces kernel-only even when a local GGUF exists (server
    // release profile, install-smoke ISOs).
    let iso = if no_model {
        eprintln!("--no-model: ISO has no model module");
        assemble_image_opt(&bin, None)?
    } else {
        assemble_image_with(&bin, model.gguf_rel())?
    };
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

/// Extra x86 `run` args for headless / CI use:
/// - `-display none` when `CHITTI_DISPLAY=none` (the e2e harness + CI set this;
///   otherwise QEMU opens a window that has nowhere to go on a CI runner).
/// - `-accel kvm` when `CHITTI_ACCEL=kvm`, or automatically when `/dev/kvm`
///   exists on a Linux host (GitHub's Linux runners expose it) — TCG otherwise.
fn x86_run_extra_args() -> Vec<String> {
    let mut v = Vec::new();
    if env::var("CHITTI_DISPLAY").as_deref() == Ok("none") {
        v.push("-display".into());
        v.push("none".into());
    }
    let accel = env::var("CHITTI_ACCEL").ok();
    let use_kvm = accel.as_deref() == Some("kvm")
        || (accel.is_none() && cfg!(target_os = "linux") && std::path::Path::new("/dev/kvm").exists());
    if use_kvm {
        v.push("-accel".into());
        v.push("kvm".into());
    }
    v
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
        return cmd_run_aarch64(release, model, disk, disk_only, no_model);
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
        // NIC (user-net or CHITTI_NET_BRIDGE — see guest_netdev); model from CHITTI_NIC.
        cmd.args(["-netdev", &guest_netdev("chittinet"), "-device", &format!("{},netdev=chittinet", nic_model())]);
        eprintln!("booting FROM DISK ONLY via UEFI (OVMF) -- no ISO; the installed Chitti boots itself");
        eprintln!("  disk: {}", disk.display());
        let status = cmd.status().map_err(|e| format!("failed to spawn qemu-system-x86_64: {e}"))?;
        eprintln!("qemu exited: {status}");
        return Ok(());
    }

    // --- Boot the ISO (optionally under UEFI); run `/install` from here ----
    let bin = build_kernel_with(release, model.features())?;
    let model_path = require_gguf_for_run(model, no_model)?;
    let iso = match model_path {
        None => assemble_image_opt(&bin, None)?,
        Some(_) => assemble_image_with(&bin, model.gguf_rel())?,
    };
    let mut cmd = qemu_base_cmd(&iso);
    if uefi {
        for arg in ovmf_pflash_args()? {
            cmd.arg(arg);
        }
        eprintln!("booting via UEFI (OVMF) -- the same GOP framebuffer path real hardware uses");
    }
    cmd.args(["-serial", "stdio"]);
    cmd.args(x86_run_extra_args()); // headless (-display none) + KVM on CI
    for a in remote_model_fw_cfg()? {
        cmd.arg(a);
    }
    cmd.arg("-drive").arg(format!("file={},if=none,id=chittidisk,format=raw", disk.display()));
    for a in disk_device_args("chittidisk", true) {
        cmd.arg(a);
    }
    // Opt-in FAT disk carrying a GGUF as `chat.gguf` for the runtime `/model
    // load` path (CHITTI_MODEL_DISK=<path>; the e2e model_load scenario).
    if let Some(md) = model_disk_from_env()? {
        cmd.arg("-drive").arg(format!("file={},if=none,id=modeldisk,format=raw", md.display()));
        cmd.args(["-device", "virtio-blk-pci,drive=modeldisk,disable-modern=on"]);
        eprintln!("  model disk attached (chat.gguf; load with /model load chat.gguf)");
    }
    // A USB keyboard on an xHCI controller, so the xhci/HID driver drives the
    // shell (as a real USB keyboard would); PS/2 also still works.
    cmd.args(["-device", "qemu-xhci,id=xhci", "-device", "usb-kbd,bus=xhci.0"]);
    // NIC (user-net or CHITTI_NET_BRIDGE — see guest_netdev); model from CHITTI_NIC.
    cmd.args(["-netdev", &guest_netdev("chittinet"), "-device", &format!("{},netdev=chittinet", nic_model())]);
    // virtio-snd on the host's audio backend (mic + speaker) for /voice.
    cmd.args(audio_args("virtio-sound-pci"));
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
    // macOS: the qemu brew keg's edk2 files. Linux: distro OVMF/edk2 packages.
    let (code, vars_src) = if cfg!(target_os = "macos") {
        let share = brew_prefix("qemu")?.join("share/qemu");
        (share.join("edk2-x86_64-code.fd"), share.join("edk2-i386-vars.fd"))
    } else {
        let code = find_path(
            "CHITTI_OVMF_CODE",
            &["/usr/share/OVMF/OVMF_CODE.fd", "/usr/share/OVMF/OVMF_CODE_4M.fd", "/usr/share/edk2/x64/OVMF_CODE.4m.fd", "/usr/share/qemu/edk2-x86_64-code.fd"],
        );
        let vars = find_path(
            "CHITTI_OVMF_VARS",
            &["/usr/share/OVMF/OVMF_VARS.fd", "/usr/share/OVMF/OVMF_VARS_4M.fd", "/usr/share/edk2/x64/OVMF_VARS.4m.fd", "/usr/share/qemu/edk2-i386-vars.fd"],
        );
        match (code, vars) {
            (Some(c), Some(v)) => (c, v),
            _ => return Err("OVMF firmware not found (install ovmf, or set CHITTI_OVMF_CODE/CHITTI_OVMF_VARS)".into()),
        }
    };
    if !code.exists() || !vars_src.exists() {
        return Err(format!("OVMF firmware not found ({} / {})", code.display(), vars_src.display()));
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

/// `cargo xtask wifi-assets`: extract Apple FullMAC dongle firmware from the
/// host macOS tree (`/usr/share/firmware/wifi`) into `assets/wifi/brcm/` in
/// the Asahi/brcmfmac naming layout:
///
/// - `brcmfmac4388-pcie.apple,miyake.bin`  (from `miyake.trx`)
/// - `brcmfmac4388-pcie.apple,miyake.txt`  (NVRAM; prefers antenna X3)
/// - optional `.clm_blob`
///
/// Cached — skips files already present. On non-macOS hosts, points at the
/// expected source layout / Asahi extract path. The kernel embeds the `.bin`
/// when present (m1n1) and ESP images copy the `brcm/` tree for disk boots.
fn cmd_wifi_assets() -> Result<(), String> {
    let out = repo_root().join("assets/wifi/brcm");
    fs::create_dir_all(&out).map_err(|e| format!("mkdir {}: {e}", out.display()))?;

    let bin_name = "brcmfmac4388-pcie.apple,miyake.bin";
    let txt_name = "brcmfmac4388-pcie.apple,miyake.txt";
    let clm_name = "brcmfmac4388-pcie.apple,miyake.clm_blob";
    let bin_dst = out.join(bin_name);
    let txt_dst = out.join(txt_name);
    let clm_dst = out.join(clm_name);

    if bin_dst.exists() && txt_dst.exists() {
        eprintln!("wifi-assets: {bin_name} + NVRAM already present");
        eprintln!("wifi-assets: done — assets/wifi/brcm/ ready (re-run after deleting to refresh)");
        return Ok(());
    }

    // Prefer newer chip steppings when multiple exist (C0 > B0 > C2).
    let mac_roots = [
        "/usr/share/firmware/wifi/C-4388__s-C0",
        "/usr/share/firmware/wifi/C-4388__s-B0",
        "/usr/share/firmware/wifi/C-4388__s-C2",
        // Asahi-style extract destinations (if the user already ran fwextract).
        "/lib/firmware/brcm",
        "/usr/lib/firmware/brcm",
    ];

    let mut trx: Option<PathBuf> = None;
    let mut nvram: Option<PathBuf> = None;
    let mut clmb: Option<PathBuf> = None;
    let mut used_root = String::new();

    for root in &mac_roots {
        let r = Path::new(root);
        if !r.exists() {
            continue;
        }
        // Apple layout: <root>/miyake.trx
        let candidate = r.join("miyake.trx");
        if candidate.exists() {
            trx = Some(candidate);
            used_root = root.to_string();
            // Antenna-specific NVRAM: prefer X3 (j473 reports antenna=X3), then
            // generic miyake, highest m- version wins.
            nvram = pick_miyake_nvram(r);
            let c = r.join("miyake.clmb");
            if c.exists() {
                clmb = Some(c);
            }
            break;
        }
        // Already-converted brcmfmac names (Asahi/Linux tree).
        let asahi_bin = r.join(bin_name);
        if asahi_bin.exists() {
            trx = Some(asahi_bin);
            used_root = root.to_string();
            let t = r.join(txt_name);
            if t.exists() {
                nvram = Some(t);
            }
            let c = r.join(clm_name);
            if c.exists() {
                clmb = Some(c);
            }
            break;
        }
    }

    let Some(trx_src) = trx else {
        return Err(
            "wifi-assets: no miyake firmware found.\n  \
             On the Mac that owns the radio, /usr/share/firmware/wifi/C-4388__s-*/miyake.trx \
             must exist (shipped with macOS).\n  \
             Or place Asahi files under /lib/firmware/brcm/ and re-run.\n  \
             Expected: brcmfmac4388-pcie.apple,miyake.bin"
                .into(),
        );
    };

    if !bin_dst.exists() {
        eprintln!(
            "wifi-assets: copying {} → {bin_name} (from {used_root})",
            trx_src.display()
        );
        fs::copy(&trx_src, &bin_dst).map_err(|e| format!("copy {}: {e}", trx_src.display()))?;
    } else {
        eprintln!("wifi-assets: {bin_name} already present");
    }

    if !txt_dst.exists() {
        if let Some(src) = nvram {
            eprintln!("wifi-assets: NVRAM {} → {txt_name}", src.display());
            let raw = fs::read(&src).map_err(|e| format!("read {}: {e}", src.display()))?;
            let cleaned = clean_nvram_txt(&raw);
            fs::write(&txt_dst, cleaned).map_err(|e| format!("write {}: {e}", txt_dst.display()))?;
        } else {
            eprintln!("wifi-assets: warning — no miyake NVRAM .txt found (OTP-only boot may still work)");
        }
    } else {
        eprintln!("wifi-assets: {txt_name} already present");
    }

    if !clm_dst.exists() {
        if let Some(src) = clmb {
            eprintln!("wifi-assets: CLM {} → {clm_name}", src.display());
            fs::copy(&src, &clm_dst).map_err(|e| format!("copy {}: {e}", src.display()))?;
        }
    }

    let sz = fs::metadata(&bin_dst).map(|m| m.len()).unwrap_or(0);
    eprintln!(
        "wifi-assets: done — {} ({} KiB){}",
        bin_dst.display(),
        sz / 1024,
        if txt_dst.exists() { " + NVRAM" } else { "" }
    );
    eprintln!("wifi-assets: rebuild the kernel so the image embeds (m1n1) / ESP copies (QEMU/VBox)");
    Ok(())
}

/// Prefer antenna-X3 NVRAM (j473), then generic, highest `m-N.N` version.
fn pick_miyake_nvram(dir: &Path) -> Option<PathBuf> {
    let mut best: Option<(u32, PathBuf)> = None;
    let rd = fs::read_dir(dir).ok()?;
    for ent in rd.flatten() {
        let name = ent.file_name();
        let s = name.to_string_lossy();
        // P-miyake-X3_M-…_m-4.7.txt or P-miyake_M-…_m-4.7.txt
        let is_x3 = s.contains("miyake-X3") || s.contains("miyake_X3");
        let is_generic = s.starts_with("P-miyake_M-") && s.ends_with(".txt");
        if !(is_x3 || is_generic) || !s.ends_with(".txt") {
            continue;
        }
        // Score: X3 gets +1000, version m-A.B → A*10+B.
        let mut score = if is_x3 { 1000u32 } else { 0 };
        if let Some(idx) = s.rfind("__m-") {
            let ver = &s[idx + 4..s.len() - 4]; // strip __m- and .txt
            let mut parts = ver.split('.');
            let a: u32 = parts.next().and_then(|x| x.parse().ok()).unwrap_or(0);
            let b: u32 = parts.next().and_then(|x| x.parse().ok()).unwrap_or(0);
            score += a * 10 + b;
        }
        match &best {
            Some((bs, _)) if *bs >= score => {}
            _ => best = Some((score, ent.path())),
        }
    }
    best.map(|(_, p)| p)
}

/// Asahi `process_nvram`: strip spurious whitespace around keys/values.
fn clean_nvram_txt(raw: &[u8]) -> Vec<u8> {
    let text = String::from_utf8_lossy(raw);
    let mut out = String::new();
    for line in text.split('\n') {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some((k, v)) = line.split_once('=') {
            out.push_str(k.trim());
            out.push('=');
            out.push_str(v.trim());
            out.push('\n');
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }
    out.into_bytes()
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

/// `cargo xtask ref-check [-arch …] [-model …]`: the reference-parity gate,
/// for whichever model `-model` selects (parity is keyed on the GGUF's
/// `general.name` against `cortex::refcheck::FIXTURES`; unknown models SKIP
/// parity but still gate the self-consistency checks).
///
/// - x86 (default): boots the ISO under TCG; the kernel exits via
///   isa-debug-exit with success (33) or failure (35). Slow but hosts-anything.
/// - aarch64: boots `-kernel` under HVF (native speed — the recommended gate
///   for the larger models); the kernel powers off via PSCI and the serial
///   output is checked for `REFCHECK: ALL PASS`.
fn cmd_ref_check(arch: Arch, model: Model) -> Result<(), String> {
    let gguf = repo_root().join(model.gguf_rel());
    if !gguf.exists() {
        return Err(format!("{} not present -- run xtask/fetch-model.sh first", model.gguf_rel()));
    }
    // The model's heap-tier feature must match the model (the 4B needs its
    // tier), plus the refcheck entry gate.
    let mut feats: Vec<&str> = model.features().to_vec();
    feats.push("refcheck");

    if matches!(arch, Arch::Aarch64) {
        let elf = build_kernel_aarch64(true, &feats)?;
        let mut cmd = Command::new("qemu-system-aarch64");
        cmd.args(["-M", "virt", "-smp", "4", "-m", model.qemu_mem()]);
        cmd.args(accel_args("aarch64"));
        cmd.args(["-display", "none", "-monitor", "none", "-serial", "stdio", "-no-reboot"]);
        cmd.arg("-kernel").arg(&elf);
        cmd.arg("-fw_cfg").arg(format!("name=opt/chitti/ramsize,string={}", mem_bytes(model.qemu_mem())));
        let base = u64::from_str_radix(model.aarch64_addr().trim_start_matches("0x"), 16)
            .map_err(|e| format!("bad model addr {}: {e}", model.aarch64_addr()))?;
        for arg in model_loader_args(&gguf, base)? {
            cmd.arg("-device").arg(arg);
        }
        eprintln!("ref-check: running the acceptance gate natively under HVF ({})...", model.label());
        // Stream serial live while capturing it — the kernel powers off via
        // PSCI (exit code carries nothing), so PASS/FAIL comes from the log.
        use std::io::BufRead as _;
        let mut child = cmd
            .stdout(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| format!("failed to spawn qemu-system-aarch64: {e}"))?;
        let mut all = String::new();
        if let Some(out) = child.stdout.take() {
            for line in std::io::BufReader::new(out).lines() {
                let line = line.unwrap_or_default();
                println!("{line}");
                all.push_str(&line);
                all.push('\n');
            }
        }
        let _ = child.wait();
        return if all.contains("REFCHECK: ALL PASS") {
            println!("ref-check: PASS (all acceptance checks green)");
            Ok(())
        } else {
            Err("ref-check: FAIL (serial log has no 'REFCHECK: ALL PASS')".to_string())
        };
    }

    let bin = build_kernel_with(true, &feats)?;
    let iso = assemble_image_with(&bin, model.gguf_rel())?;
    // CPU inference under QEMU/TCG takes minutes; the model module needs
    // headroom beyond the file size.
    let mut cmd = Command::new("qemu-system-x86_64");
    cmd.args([
        "-M", "q35", "-cpu", "max", "-smp", "4", "-m", model.qemu_mem(), "-device",
        "isa-debug-exit,iobase=0xf4,iosize=0x04", "-no-reboot",
    ]);
    cmd.arg("-cdrom").arg(&iso);
    cmd.args(["-serial", "stdio", "-display", "none"]);
    eprintln!("ref-check: running in-kernel acceptance gate under QEMU/TCG ({}, this takes a few minutes)...", model.label());
    let status = cmd.status().map_err(|e| format!("failed to spawn qemu-system-x86_64: {e}"))?;
    match status.code() {
        Some(33) => {
            println!("ref-check: PASS (all acceptance checks green)");
            Ok(())
        }
        Some(35) => Err("ref-check: FAIL (QemuExitCode::Failed -- an acceptance check did not match)".to_string()),
        Some(other) => Err(format!("ref-check: QEMU exited with unexpected status {other}")),
        None => Err("ref-check: QEMU was terminated by a signal".to_string()),
    }
}
