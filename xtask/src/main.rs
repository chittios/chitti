//! Build orchestration for ChittiOS: assembles a bootable Limine image
//! from the kernel and drives QEMU. All project commands go through
//! `cargo xtask <cmd>` (see CHITTI_OS_HANDOFF.md Part 7).

use std::env;
mod paper;
mod rings;
mod entry;
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
    /// Liquid LFM2.5-2.6B instruct (`LFM2.5-2.6B-Q4_0.gguf`, ~1.6 GiB). Cortex
    /// `Lfm2` family: 30 layers, 22 recurrent shortconv + 8 attention.
    Lfm2_6B,
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
            // ~1.6 GiB Q4_0 weights → default (1 GiB) heap tier.
            Model::Lfm2_6B => &[],
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
            Model::Lfm2_6B => "assets/model-lfm2-2.6b.gguf",
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
            // ~1.6 GiB Q4_0 at 2 GiB + heap.
            Model::Lfm2_6B => "5G",
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
            Model::Lfm2_6B => "lfm2.5-2.6b",
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

/// QEMU args exporting a **host folder** to the guest over virtio-9p, from
/// `CHITTI_SHARE=<host path>` (and `CHITTI_SHARE_TAG`, default `hostshare`).
///
/// `pci` picks the device model: the aarch64 `-kernel` dev loop has no PCI at
/// all (ECAM comes from the stub's ACPI), so it must use the virtio-mmio
/// `virtio-9p-device`, while x86 and the UEFI paths use `virtio-9p-pci`. The
/// guest driver binds on either.
///
/// `security_model=none` because the alternative, `mapped-xattr`, stores guest
/// ownership in host extended attributes — which makes the shared files awkward
/// to use from the host, and the point of a shared folder is that both sides
/// can read it. Files the guest creates are owned by the QEMU process.
///
/// Empty means unset: `make` passes the variable through unconditionally, the
/// trap `CHITTI_RESOLUTION` and `CHITTI_SAMPLE_FILES` both hit.
fn share_args(pci: bool) -> Vec<String> {
    let Ok(path) = env::var("CHITTI_SHARE") else {
        return Vec::new();
    };
    if path.trim().is_empty() {
        return Vec::new();
    }
    // A missing directory would make QEMU refuse to start at all, which reads
    // as "the kernel will not boot" rather than "that folder is not there".
    if !std::path::Path::new(&path).is_dir() {
        eprintln!("xtask: CHITTI_SHARE={path} is not a directory — not sharing a host folder");
        return Vec::new();
    }
    let tag = env::var("CHITTI_SHARE_TAG").unwrap_or_else(|_| "hostshare".into());
    let dev = if pci { "virtio-9p-pci" } else { "virtio-9p-device" };
    println!("xtask: sharing {path} into the guest as /host (tag {tag})");
    vec![
        "-fsdev".into(),
        format!("local,id=chittishare,path={path},security_model=none"),
        "-device".into(),
        format!("{dev},fsdev=chittishare,mount_tag={tag}"),
    ]
}

/// QEMU args for the **SPICE clipboard agent channel**: a virtio-serial bus
/// with one port named `com.redhat.spice.0`, backed by the `qemu-vdagent`
/// chardev. Enabled by `CHITTI_CLIPBOARD=1`.
///
/// `pci` picks the bus model for the same reason `share_args` does — the
/// aarch64 `-kernel` dev loop has no PCI.
///
/// **What this does and does not buy, by host.** The chardev connects the guest
/// to QEMU's *internal* clipboard manager. QEMU only bridges that to a real
/// system clipboard through a display backend that registers a clipboard peer,
/// and only `gtk` and `dbus` do — **`cocoa` does not**. So on a Linux/GTK host
/// this is window-to-window copy/paste, and on macOS the link is live but ends
/// inside QEMU. The guest says as much in `/clip` rather than implying it
/// worked.
fn clipboard_args(pci: bool) -> Vec<String> {
    let on = env::var("CHITTI_CLIPBOARD").map(|v| {
        let v = v.trim().to_ascii_lowercase();
        !(v.is_empty() || v == "0" || v == "off" || v == "no")
    });
    if on != Ok(true) {
        return Vec::new();
    }
    let bus = if pci { "virtio-serial-pci" } else { "virtio-serial-device" };
    vec![
        "-device".into(),
        format!("{bus},id=chitticlip"),
        "-chardev".into(),
        "qemu-vdagent,id=vdagent,name=vdagent,clipboard=on,mouse=off".into(),
        "-device".into(),
        "virtserialport,bus=chitticlip.0,chardev=vdagent,name=com.redhat.spice.0".into(),
    ]
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

/// Whether an env var is a truthy flag (`1`/`true`/`yes`) or a non-empty value
/// that is not an explicit off (`0`/`false`/`no`/`""`).
fn env_flag_or_value(name: &str) -> Option<String> {
    let Ok(v) = env::var(name) else {
        return None;
    };
    let t = v.trim();
    if t.is_empty() || t == "0" || t.eq_ignore_ascii_case("false") || t.eq_ignore_ascii_case("no") {
        return None;
    }
    Some(t.to_string())
}

fn env_truthy(name: &str) -> bool {
    matches!(
        env::var(name).ok().as_deref().map(str::trim),
        Some("1") | Some("true") | Some("yes") | Some("on")
    )
}

/// Parse `vid:pid` / `0xVVVV:0xPPPP` into QEMU hex strings without `0x`.
fn parse_vid_pid(s: &str) -> Option<(String, String)> {
    let s = s.trim().trim_start_matches("usb-host,");
    let (v, p) = s.split_once(':')?;
    let vid = v.trim().trim_start_matches("0x").trim_start_matches("0X");
    let pid = p.trim().trim_start_matches("0x").trim_start_matches("0X");
    if vid.is_empty() || pid.is_empty() {
        return None;
    }
    // Accept 1–4 hex digits.
    if !vid.chars().all(|c| c.is_ascii_hexdigit()) || !pid.chars().all(|c| c.is_ascii_hexdigit()) {
        return None;
    }
    Some((vid.to_ascii_lowercase(), pid.to_ascii_lowercase()))
}

/// Host USB devices to pass through: `CHITTI_USB_HOST=vid:pid[,…]` plus optional
/// auto-discovery via `CHITTI_USB_BT=1` / `CHITTI_USB_CAM=1` (or explicit
/// `CHITTI_USB_BT=0a12:0001`). Discovery greps macOS `system_profiler` / Linux
/// `lsusb` for Bluetooth / camera product names.
fn collect_usb_host_ids() -> Vec<(String, String, &'static str)> {
    let mut out: Vec<(String, String, &'static str)> = Vec::new();
    let mut push = |vid: String, pid: String, kind: &'static str| {
        if !out.iter().any(|(v, p, _)| v == &vid && p == &pid) {
            out.push((vid, pid, kind));
        }
    };
    if let Some(list) = env_flag_or_value("CHITTI_USB_HOST") {
        for part in list.split(|c| c == ',' || c == ' ') {
            if part.is_empty() {
                continue;
            }
            if let Some((v, p)) = parse_vid_pid(part) {
                push(v, p, "host");
            } else {
                eprintln!("xtask: ignore bad CHITTI_USB_HOST entry {part:?} (want vid:pid)");
            }
        }
    }
    for (var, kind, auto) in [
        ("CHITTI_USB_BT", "bt", true),
        ("CHITTI_USB_CAM", "cam", true),
    ] {
        match env_flag_or_value(var) {
            Some(v) if env_truthy(var) || v == "1" || v.eq_ignore_ascii_case("auto") => {
                for (vid, pid) in discover_usb_ids(kind) {
                    push(vid, pid, kind);
                }
            }
            Some(v) => {
                // Explicit vid:pid (or several).
                for part in v.split(|c| c == ',' || c == ' ') {
                    if part.is_empty() || part == "1" || part.eq_ignore_ascii_case("auto") {
                        continue;
                    }
                    if let Some((vid, pid)) = parse_vid_pid(part) {
                        push(vid, pid, kind);
                    } else if auto {
                        eprintln!("xtask: {var}={part:?} is not vid:pid; try 1 for auto-grep");
                    }
                }
            }
            None => {}
        }
    }
    out
}

/// Grep the host USB tree for Bluetooth (`bt`) or UVC/camera (`cam`) devices.
fn discover_usb_ids(kind: &str) -> Vec<(String, String)> {
    let keywords: &[&str] = match kind {
        "bt" => &[
            "bluetooth",
            " bluetooth",
            "csr8510",
            "btusb",
            "wireless bluetooth",
            "bluetooth radio",
            "bluetooth adapter",
        ],
        "cam" => &[
            "camera",
            "webcam",
            "uvc",
            "imaging",
            "hd web",
            "usb video",
            "facetime",
            "integrated camera",
        ],
        _ => return Vec::new(),
    };
    #[cfg(target_os = "macos")]
    {
        discover_usb_ids_macos(keywords)
    }
    #[cfg(not(target_os = "macos"))]
    {
        discover_usb_ids_lsusb(keywords)
    }
}

#[cfg(target_os = "macos")]
fn discover_usb_ids_macos(keywords: &[&str]) -> Vec<(String, String)> {
    let out = Command::new("system_profiler")
        .args(["SPUSBDataType"])
        .output();
    let Ok(out) = out else {
        eprintln!("xtask: system_profiler SPUSBDataType failed — set CHITTI_USB_HOST=vid:pid by hand");
        return Vec::new();
    };
    let text = String::from_utf8_lossy(&out.stdout);
    parse_system_profiler_usb(&text, keywords)
}

/// Pure parse of `system_profiler SPUSBDataType` text (unit-tested via xtask tests).
fn parse_system_profiler_usb(text: &str, keywords: &[&str]) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut name_hit = false;
    let mut vid: Option<String> = None;
    let mut pid: Option<String> = None;
    let kw_match = |s: &str| {
        let l = s.to_ascii_lowercase();
        keywords.iter().any(|k| l.contains(k))
    };
    for line in text.lines() {
        let t = line.trim();
        // Product name lines look like " equ  Something Bluetooth:" or "FaceTime HD Camera:".
        if t.ends_with(':') && !t.contains("ID:") && !t.starts_with("Product ID") {
            // Flush previous on new node.
            if name_hit {
                if let (Some(v), Some(p)) = (vid.take(), pid.take()) {
                    if !out.iter().any(|(a, b)| a == &v && b == &p) {
                        out.push((v, p));
                    }
                }
            }
            name_hit = kw_match(t);
            vid = None;
            pid = None;
            continue;
        }
        if let Some(rest) = t.strip_prefix("Vendor ID:") {
            // "0x0a12  (Cambridge …)" or "0x0a12"
            let hex = rest
                .split_whitespace()
                .next()
                .unwrap_or("")
                .trim_start_matches("0x")
                .trim_start_matches("0X");
            if !hex.is_empty() {
                vid = Some(hex.to_ascii_lowercase());
            }
        }
        if let Some(rest) = t.strip_prefix("Product ID:") {
            let hex = rest
                .split_whitespace()
                .next()
                .unwrap_or("")
                .trim_start_matches("0x")
                .trim_start_matches("0X");
            if !hex.is_empty() {
                pid = Some(hex.to_ascii_lowercase());
            }
        }
        // Also mark hit if Manufacturer/Serial lines mention keywords.
        if !name_hit && (t.starts_with("Manufacturer:") || t.starts_with("Serial Number:")) && kw_match(t)
        {
            name_hit = true;
        }
        if name_hit {
            if let (Some(v), Some(p)) = (vid.as_ref(), pid.as_ref()) {
                if !out.iter().any(|(a, b)| a == v && b == p) {
                    out.push((v.clone(), p.clone()));
                }
                name_hit = false;
                vid = None;
                pid = None;
            }
        }
    }
    out
}

#[cfg(not(target_os = "macos"))]
fn discover_usb_ids_lsusb(keywords: &[&str]) -> Vec<(String, String)> {
    let out = Command::new("lsusb").output();
    let Ok(out) = out else {
        eprintln!("xtask: lsusb failed — set CHITTI_USB_HOST=vid:pid by hand");
        return Vec::new();
    };
    let text = String::from_utf8_lossy(&out.stdout);
    parse_lsusb(&text, keywords)
}

/// Pure parse of `lsusb` lines (Linux discovery + unit tests on every OS).
#[cfg_attr(target_os = "macos", allow(dead_code))]
fn parse_lsusb(text: &str, keywords: &[&str]) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for line in text.lines() {
        let l = line.to_ascii_lowercase();
        if !keywords.iter().any(|k| l.contains(k)) {
            continue;
        }
        // "Bus 001 Device 004: ID 0a12:0001 Cambridge Silicon Radio …"
        if let Some(id_pos) = l.find(" id ") {
            let rest = line[id_pos + 4..].trim();
            let token = rest.split_whitespace().next().unwrap_or("");
            if let Some((v, p)) = parse_vid_pid(token) {
                if !out.iter().any(|(a, b)| a == &v && b == &p) {
                    out.push((v, p));
                }
            }
        }
    }
    out
}

/// Whether the user asked for host USB passthrough (even if discovery found none).
fn usb_host_requested() -> bool {
    env_flag_or_value("CHITTI_USB_HOST").is_some()
        || env_flag_or_value("CHITTI_USB_BT").is_some()
        || env_flag_or_value("CHITTI_USB_CAM").is_some()
}

/// QEMU `-device usb-host,…` args for passthrough onto bus `bus` (e.g. `xhci.0`).
/// Empty when no CHITTI_USB_* is set. When requested but **nothing was found**,
/// prints a clear warning and continues without attaching (QEMU still boots).
fn usb_host_device_args(bus: &str) -> Vec<String> {
    let requested = usb_host_requested();
    let ids = collect_usb_host_ids();
    if ids.is_empty() {
        if requested {
            eprintln!(
                "xtask: USB_BT/USB_CAM/USB_HOST set but no host device matched — \
                 continuing without passthrough"
            );
            eprintln!("  list candidates:  make usb-list");
            eprintln!("  explicit IDs:     make run USB_BT=0a12:0001 USB_CAM=046d:082d");
            eprintln!(
                "  note: QEMU has no emulated Bluetooth/UVC; macOS may hide \
                 internal FaceTime/BT from system_profiler — use a USB dongle/webcam"
            );
        }
        return Vec::new();
    }
    let mut args = Vec::new();
    for (i, (vid, pid, kind)) in ids.iter().enumerate() {
        eprintln!(
            "  usb-host[{i}]: {kind} vendorid=0x{vid} productid=0x{pid} bus={bus}"
        );
        args.push("-device".into());
        // guest-reset=false: a failed reset on an in-use host device must not
        // kill the whole VM; the guest simply sees no device.
        args.push(format!(
            "usb-host,vendorid=0x{vid},productid=0x{pid},bus={bus},id=usbhost{i},guest-reset=false"
        ));
    }
    eprintln!(
        "  tip: if the guest sees no device, unplug from macOS, grant QEMU USB access, \
         or re-check make usb-list"
    );
    args
}

/// True when the run wants host USB passthrough (so aarch64 must add qemu-xhci).
fn wants_usb_host() -> bool {
    !collect_usb_host_ids().is_empty()
}

/// `cargo xtask usb-ids [bt|cam|all]` — grep host USB and print vid:pid lines.
fn cmd_usb_ids(rest: &[String]) -> Result<(), String> {
    let filter = rest
        .iter()
        .find(|a| !a.starts_with('-'))
        .map(|s| s.as_str())
        .unwrap_or("all");
    // Force auto-grep for the kinds we care about when env is unset, so a bare
    // `cargo xtask usb-ids` is useful without `make usb-list`'s env.
    if matches!(filter, "bt" | "all") && env_flag_or_value("CHITTI_USB_BT").is_none() {
        env::set_var("CHITTI_USB_BT", "1");
    }
    if matches!(filter, "cam" | "all") && env_flag_or_value("CHITTI_USB_CAM").is_none() {
        env::set_var("CHITTI_USB_CAM", "1");
    }
    let ids = collect_usb_host_ids();
    if ids.is_empty() {
        println!("(none found — plug a device or pass vid:pid)");
        return Ok(());
    }
    for (vid, pid, kind) in ids {
        let keep = match filter {
            "bt" => kind == "bt",
            "cam" => kind == "cam",
            _ => true,
        };
        if keep {
            println!("{kind}\t0x{vid}:0x{pid}");
        }
    }
    Ok(())
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
/// `-fw_cfg name=opt/chitti/model,file=…` (the `-kernel` path). The same file
/// is copied onto the ESP as `\chitti-model.json` by the **image** builder, so
/// a UEFI/stub boot (VirtualBox, real hardware) gets the same seed through the
/// boot-info page. The kernel reads it at shell start and activates the hosted
/// backend (same shape as `/configs/core/model.json`).
///
/// Under slirp user-net the host is `10.0.2.2` (not the host's LAN IP).
fn remote_model_json() -> Result<Option<PathBuf>, String> {
    let Ok(url) = env::var("CHITTI_REMOTE_URL") else {
        return Ok(None);
    };
    let url = url.trim().trim_end_matches('/').to_string();
    if url.is_empty() {
        return Ok(None);
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
    Ok(Some(path))
}

/// The `-fw_cfg` seed for QEMU `-kernel` boots (aarch64 + x86), if any.
fn remote_model_fw_cfg() -> Result<Vec<String>, String> {
    let Some(path) = remote_model_json()? else {
        return Ok(Vec::new());
    };
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
                // Liquid LFM2.5-2.6B (Q4_0, cortex Lfm2 family).
                "lfm2.5-2.6b"
                | "lfm2-2.6b"
                | "lfm2.5"
                | "lfm2"
                | "LFM2.5-2.6B" => Ok(Model::Lfm2_6B),
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
        // Fetch the `/samples/` corpus (images/videos/audios/misc). The build
        // paths call this for you when CHITTI_SAMPLE_FILES is set.
        "sample-files" | "samples" => cmd_sample_files(&rest),
        "javy-plugin" => cmd_javy_plugin(),
        "wifi-assets" => cmd_wifi_assets(),
        "iwlwifi-assets" => cmd_iwlwifi_assets(),
        // Phase 3 parity gate: build the kernel with the `refcheck` feature,
        // boot the real model, run the acceptance checks, exit pass/fail.
        "ref-check" => cmd_ref_check(arch, model),
        // Verify the quantitative claims in `paper/main.tex` against the tree.
        // Reports and exits non-zero on drift; never edits the paper.
        "paper-check" => cmd_paper_check(&rest),
        "ring-check" => cmd_ring_check(),
        "imgdec" => cmd_imgdec(),
        // Print host USB vid:pid candidates for BT/camera (used by `make usb-list`).
        "usb-ids" => cmd_usb_ids(&rest),
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
    "usage: cargo xtask <build|image|run|m1n1|test|ref-check|voice-assets|sample-files|wifi-assets|usb-ids> [-arch x86_64|aarch64] \
     [-model qwen3.5-0.8b|2b|4b|9b|gemma-4-e4b|bonsai-27b|bonsai-27b-ternary] [--release] [--uefi] [-server]\n\
     sample-files [--refresh]: fetch the /samples corpus (images/videos/audios/misc) into \
     assets/samples/; embedded into the image and seeded to /samples/ at boot. \
     CHITTI_SAMPLE_FILES=1 (default in `make run` / `make vbox`) fetches + embeds it automatically.\n\
     javy-plugin: rebuild assets/wasm/javy-plugin.wasm (the JS engine + chitti host surface) \
     from tools/javy-plugin; fetches the Javy CLI, needs `rustup target add wasm32-wasip1`.\n\
     wifi-assets: extract Apple FullMAC firmware from macOS into assets/wifi/ (for /wifi load).\n\
     iwlwifi-assets: fetch Intel WiFi firmware from linux-firmware into assets/wifi/iwl/.\n\
     usb-ids [bt|cam|all]: grep host USB for Bluetooth/camera vid:pid (see make usb-list / USB_BT=1).\n\
     m1n1 (aarch64): package the kernel as a gzip'd arm64 Image and boot it on a \
     tethered Apple Silicon Mac over the m1n1 USB proxy; configure via env \
     CHITTI_M1N1/CHITTI_DTB[/CHITTI_INITRD/CHITTI_BOOTARGS/M1N1DEVICE].\n\
     run flags (x86_64): --disk <2G|1500M> size the virtio-blk disk for /install; \
     --disk-only boot the installed disk via UEFI with no ISO; --fresh-disk wipe it first; \
     --no-model build/boot without a model module (also works with `image`).\n\
     run USB: CHITTI_USB_BT=1|vid:pid  CHITTI_USB_CAM=1|vid:pid  CHITTI_USB_HOST=vid:pid,… \
     → QEMU -device usb-host on qemu-xhci (no emulated BT/UVC in QEMU).\n\
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
    // See `build_kernel_with`: samples are embedded at compile time, so fetch
    // before cargo runs. No-op without CHITTI_SAMPLE_FILES.
    ensure_sample_files();
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
/// The display device(s) to give the guest, from `$CHITTI_GPU`.
///
/// * unset / `ramfb` — the simple linear framebuffer the kernel configures via
///   fw_cfg. The default, and the only one that works on the aarch64 `-kernel`
///   path (see below).
/// * `virtio` — `virtio-gpu-pci`, which the in-kernel **KMS** driver binds for real
///   mode setting (`/display set` changes the hardware mode instead of
///   letterboxing).
/// * `vmware` — `vmware-svga`, to exercise the (currently declined) VMSVGA path.
///
/// **`virtio` needs PCI**, and on aarch64 PCI comes from the stub's ACPI — so pass
/// `--uefi` there, or use `-arch x86_64`. On the plain aarch64 `-kernel` path the
/// device is invisible and the driver will not bind.
fn gpu_device_args(arch: &str, uefi: bool) -> Vec<String> {
    let want = std::env::var("CHITTI_GPU").unwrap_or_default();
    let want = want.trim().to_lowercase();
    match want.as_str() {
        "virtio" | "virtio-gpu" => {
            if arch == "aarch64" && !uefi {
                eprintln!(
                    "  CHITTI_GPU=virtio needs PCI: on aarch64 add --uefi (ECAM comes from the stub's ACPI). Using ramfb."
                );
                return vec!["-device".into(), "ramfb".into()];
            }
            // Keep **ramfb as well**. Booting aarch64/HVF with virtio-gpu as the
            // only display puts the firmware's GOP framebuffer inside the device's
            // BAR, and writing there after ExitBootServices aborts QEMU with
            // `hvf: isv` — verified with the KMS probe disabled, so it is the
            // environment, not the driver. With ramfb present the console comes up
            // on safe memory and virtio-gpu is a second device the driver binds; its
            // scanout is DMA RAM, which is writable normally.
            eprintln!("  gpu: ramfb (console) + virtio-gpu-pci (KMS: real mode setting)");
            vec![
                "-device".into(),
                "ramfb".into(),
                "-device".into(),
                "virtio-gpu-pci,id=gpu0".into(),
            ]
        }
        "vmware" | "vmsvga" => {
            eprintln!("  gpu: vmware-svga (VMSVGA driver is detected but declined — see kms/vmsvga.rs)");
            vec!["-device".into(), "vmware-svga,id=gpu0".into()]
        }
        "" | "ramfb" => vec!["-device".into(), "ramfb".into()],
        other => {
            eprintln!("  CHITTI_GPU='{other}' unknown (ramfb|virtio|vmware) — using ramfb");
            vec!["-device".into(), "ramfb".into()]
        }
    }
}

fn ramfb_res_fw_cfg() -> Vec<String> {
    let (w, h) = std::env::var("CHITTI_FB_RES")
        .ok()
        .filter(|v| !v.trim().eq_ignore_ascii_case("max")) // handled by detection
        .and_then(|s| parse_wxh(&s))
        .or_else(detect_host_res)
        .unwrap_or((1600, 1000));
    eprintln!(
        "  framebuffer: {w}x{h} (CHITTI_FB_RES=WxH|max to pin, CHITTI_FB_DISPLAY=<name> to pick a monitor)"
    );
    vec!["-fw_cfg".into(), format!("name=opt/chitti/fbres,string={w}x{h}")]
}

/// Parse a `WIDTHxHEIGHT` string.
fn parse_wxh(s: &str) -> Option<(u32, u32)> {
    let (a, b) = s.trim().split_once(['x', 'X'])?;
    Some((a.trim().parse().ok()?, b.trim().parse().ok()?))
}

/// One attached display as `system_profiler -json` describes it.
#[derive(Debug, PartialEq)]
struct HostDisplay {
    name: String,
    /// `_spdisplays_resolution` — the **desktop** size actually in use. This is the
    /// authoritative value and the one to match; the plain-text `system_profiler`
    /// output omits it for a display at its default scaled mode, which is why the
    /// earlier text-scraping version had to guess (and got a 1440x900 desktop wrong
    /// by halving the 2560x1600 panel to 1280x800).
    desktop: (u32, u32),
    /// `_spdisplays_pixels` — the backing store, i.e. the most pixels this display
    /// can actually show (2880x1800 for a 1440x900 HiDPI desktop). What `max` means.
    native: (u32, u32),
    main: bool,
}

/// Pull `"key" : "value"` out of one JSON object body. Adequate because every field
/// wanted here is a flat string and the objects are machine-generated.
fn json_str_field(body: &str, key: &str) -> Option<String> {
    let pat = format!("\"{key}\"");
    let at = body.find(&pat)?;
    let rest = &body[at + pat.len()..];
    let colon = rest.find(':')?;
    let after = &rest[colon + 1..];
    let open = after.find('"')?;
    let tail = &after[open + 1..];
    let close = tail.find('"')?;
    Some(tail[..close].to_string())
}

/// `"1440 x 900 @ 60.00Hz"` / `"2880 x 1800"` → `(1440, 900)`.
fn parse_res_field(v: &str) -> Option<(u32, u32)> {
    let parts: Vec<&str> = v.split_whitespace().collect();
    if parts.len() >= 3 && parts[1] == "x" {
        return Some((parts[0].parse().ok()?, parts[2].parse().ok()?));
    }
    None
}

/// Split the objects of the first `spdisplays_ndrvs` array, brace-matched.
fn ndrvs_objects(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let Some(at) = text.find("\"spdisplays_ndrvs\"") else { return out };
    let bytes = text.as_bytes();
    let mut i = at;
    // Advance to the '[' that opens the array.
    while i < bytes.len() && bytes[i] != b'[' {
        i += 1;
    }
    let mut depth = 0i32;
    let mut start = 0usize;
    while i < bytes.len() {
        match bytes[i] {
            b'{' => {
                if depth == 0 {
                    start = i;
                }
                depth += 1;
            }
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    out.push(text[start..=i].to_string());
                }
            }
            b']' if depth == 0 => break,
            _ => {}
        }
        i += 1;
    }
    out
}

/// Parse `system_profiler SPDisplaysDataType -json` into the attached displays.
fn parse_displays(text: &str) -> Vec<HostDisplay> {
    ndrvs_objects(text)
        .iter()
        .filter_map(|o| {
            let name = json_str_field(o, "_name")?;
            let desktop = json_str_field(o, "_spdisplays_resolution").and_then(|v| parse_res_field(&v));
            let native = json_str_field(o, "_spdisplays_pixels").and_then(|v| parse_res_field(&v));
            // A display with neither field is not something we can size a window to.
            let desktop = desktop.or(native)?;
            Some(HostDisplay {
                name,
                desktop,
                native: native.unwrap_or(desktop),
                main: json_str_field(o, "spdisplays_main").as_deref() == Some("spdisplays_yes"),
            })
        })
        .collect()
}

/// Choose the display whose size the guest window should match: the one named by
/// `want` (case-insensitive substring), else the main display, else the first.
fn pick_display<'a>(displays: &'a [HostDisplay], want: Option<&str>) -> Option<&'a HostDisplay> {
    if let Some(want) = want.map(str::trim).filter(|w| !w.is_empty()) {
        let lower = want.to_lowercase();
        if let Some(d) = displays.iter().find(|d| d.name.to_lowercase().contains(&lower)) {
            return Some(d);
        }
        eprintln!(
            "  CHITTI_FB_DISPLAY='{want}' matched no display (have: {}) — using the main one",
            displays.iter().map(|d| d.name.as_str()).collect::<Vec<_>>().join(", ")
        );
    }
    displays.iter().find(|d| d.main).or_else(|| displays.first())
}

/// Best-effort host display size on macOS.
///
/// Returns the chosen display's **desktop** size (so the QEMU window fits the screen
/// it opens on), or its full backing-store size when `CHITTI_FB_RES=max`.
fn detect_host_res() -> Option<(u32, u32)> {
    if !cfg!(target_os = "macos") {
        return None; // Linux hosts: use the default (or CHITTI_FB_RES)
    }
    let out = Command::new("system_profiler")
        .args(["SPDisplaysDataType", "-json"])
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let displays = parse_displays(&text);
    if displays.is_empty() {
        return None;
    }
    if displays.len() > 1 {
        let list: Vec<String> = displays
            .iter()
            .map(|d| {
                format!(
                    "{}{} {}x{} (max {}x{})",
                    d.name,
                    if d.main { "*" } else { "" },
                    d.desktop.0,
                    d.desktop.1,
                    d.native.0,
                    d.native.1
                )
            })
            .collect();
        eprintln!("  displays: {} (* = main)", list.join(", "));
    }
    let want = std::env::var("CHITTI_FB_DISPLAY").ok();
    let d = pick_display(&displays, want.as_deref())?;
    let wants_max = std::env::var("CHITTI_FB_RES").map(|v| v.trim().eq_ignore_ascii_case("max")).unwrap_or(false);
    let (w, h) = if wants_max { d.native } else { d.desktop };
    Some((w.clamp(640, 3840), h.clamp(480, 2400)))
}

/// `cargo xtask imgdec` — rebuild the userspace tenant blobs for **both** arches.
///
/// The kernel `include_bytes!`s the flat binaries, so they are checked in like
/// `tools/*-wasm`'s modules. Both arches every time, deliberately: a blob rebuilt for one
/// is the divergence the dual-arch standing rule exists to prevent, and it would not be
/// caught by either build.
fn cmd_imgdec() -> Result<(), String> {
    let repo = repo_root();
    let crate_dir = repo.join("userspace/imgdec");
    let objcopy = find_objcopy()?;
    // `llvm-nm` ships beside `llvm-objcopy` in the same rustup component, so derive it rather
    // than searching again.
    let nm = objcopy.replace("objcopy", "nm");
    for arch in ["x86_64", "aarch64"] {
        let target = format!("../../targets/{arch}-chitti-user.json");
        let st = std::process::Command::new("cargo")
            .current_dir(&crate_dir)
            .args(["build", "--release", "--target", &target])
            .status()
            .map_err(|e| format!("imgdec: cargo: {e}"))?;
        if !st.success() {
            return Err(format!("imgdec: build failed for {arch}"));
        }
        let elf = crate_dir.join(format!("target/{arch}-chitti-user/release/imgdec"));
        let bin = crate_dir.join(format!("imgdec-{arch}.bin"));
        let st = std::process::Command::new(&objcopy)
            .args(["-O", "binary"])
            .arg(&elf)
            .arg(&bin)
            .status()
            .map_err(|e| format!("imgdec: objcopy: {e}"))?;
        if !st.success() {
            return Err(format!("imgdec: objcopy failed for {arch}"));
        }
        // **Record where the entry is, rather than requiring it to be first.** Arranging that in
        // the linker script cost several builds and still failed on x86 (`_start` at
        // `USER_BASE + 0xc` while aarch64 was exact), and the script-level `ASSERT` did not even
        // fire. The kernel `include!`s this number and jumps to `USER_BASE + offset`.
        // From `llvm-nm` plus the linker script, **not** the ELF header: CLAUDE.md's "no ELF
        // loader" is a rule about what this project learns to parse, and it is unnecessary here
        // — two lines of text already carry the answer, and the base comes from the very file
        // the linker used rather than a second copy of the constant.
        let nm_out = std::process::Command::new(&nm)
            .arg(&elf)
            .output()
            .map_err(|e| format!("imgdec: nm: {e}"))?;
        let ld = std::fs::read_to_string(crate_dir.join(format!("link-{arch}.ld")))
            .map_err(|e| format!("imgdec: read linker script: {e}"))?;
        let lay = entry::layout(&String::from_utf8_lossy(&nm_out.stdout), &ld)
            .ok_or_else(|| format!("imgdec: {arch}: could not read the layout via nm + linker script"))?;
        // A tuple expression, so the kernel can `include!` it straight into a `const`.
        let off_file = crate_dir.join(format!("entry-{arch}.in"));
        std::fs::write(&off_file, format!("({}, {}, {})\n", lay.entry, lay.rx, lay.rw))
            .map_err(|e| format!("imgdec: write {}: {e}", off_file.display()))?;
        let n = std::fs::metadata(&bin).map(|m| m.len()).unwrap_or(0);
        println!(
            "imgdec: {arch} -> {} ({n} bytes; entry +{}, rx {} B, rw {} B)",
            bin.display(),
            lay.entry,
            lay.rx,
            lay.rw
        );
    }
    println!("imgdec: both blobs rebuilt -- commit them, the kernel include_bytes! them");
    Ok(())
}

/// `cargo xtask ring-check` — enforce the ring-3 standing rule.
///
/// Fails if any file outside `rings::ALLOWED` calls the Synapse executor directly. See
/// `rings` for why this is a check and not a code review item: a bypass keeps kernel
/// privilege silently, so nothing else goes red.
fn cmd_ring_check() -> Result<(), String> {
    let repo = repo_root();
    let hits = rings::check(&repo).map_err(|e| format!("ring-check: {e}"))?;
    if hits.is_empty() {
        println!("ring-check: ok -- no direct executor calls outside the allowlist");
        println!("  agents and agent-facing commands must use synapse::tenant::invoke_in_userspace");
        return Ok(());
    }
    println!("ring-check: {} direct executor call(s) outside the allowlist:", hits.len());
    for h in &hits {
        println!("  {}:{}  {}", h.file, h.line, h.text);
    }
    println!();
    println!("These keep kernel privilege for work that should run in ring 3. Either route");
    println!("them through `synapse::tenant::invoke_in_userspace` (passing the justification");
    println!("the in-kernel path used), or add the file to `rings::ALLOWED` with a reason.");
    Err(format!("ring-check: {} bypass(es) of the ring-3 rule", hits.len()))
}

/// `cargo xtask paper-check [--ran N]` — compare the paper's derivable claims
/// with the tree. `--ran N` supplies the number of tests the x86 suite executed
/// (from `cargo xtask test`); without it that one claim is skipped rather than
/// guessed at.
fn cmd_paper_check(rest: &[String]) -> Result<(), String> {
    let ran = rest
        .iter()
        .position(|a| a == "--ran")
        .and_then(|i| rest.get(i + 1))
        .and_then(|v| v.parse::<u64>().ok());
    let repo = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .ok_or("cannot locate the repo root")?
        .to_path_buf();
    let claims = paper::check(&repo, ran).map_err(|e| format!("paper-check: {e}"))?;
    if claims.is_empty() {
        return Err("paper-check: found no claims to check -- has the prose changed?".into());
    }
    println!("paper-check: {} derivable claim(s)", claims.len());
    for c in &claims {
        println!("{c}");
    }
    if ran.is_none() {
        println!("  skipped  unit tests run (x86): pass --ran N from `cargo xtask test`");
    }
    println!("  unchecked measured figures (tok/s, gate ns, attack rates) -- these come from");
    println!("            running the kernel (/perf, /bench synapse, /redteam), not the source.");
    let bad = claims.iter().filter(|c| !c.ok()).count();
    if bad > 0 {
        return Err(format!("paper-check: {bad} claim(s) no longer match the code"));
    }
    Ok(())
}

#[cfg(test)]
mod display_tests {
    use super::*;

    /// Real `system_profiler SPDisplaysDataType -json` output (trimmed to the fields
    /// used): an M2 laptop whose **desktop is 1440x900** on a 2880x1800 backing
    /// store, plus an external 1080p monitor. The laptop entry is exactly the case
    /// the old text-scraping version got wrong.
    const TWO_DISPLAYS: &str = r#"{
  "SPDisplaysDataType" : [
    {
      "spdisplays_ndrvs" : [
        {
          "_name" : "Color LCD",
          "_spdisplays_pixels" : "2880 x 1800",
          "_spdisplays_resolution" : "1440 x 900 @ 60.00Hz",
          "spdisplays_main" : "spdisplays_yes"
        },
        {
          "_name" : "DELL P2722HE",
          "_spdisplays_pixels" : "1920 x 1080",
          "_spdisplays_resolution" : "1920 x 1080 @ 60.00Hz"
        }
      ]
    }
  ]
}"#;

    #[test]
    fn parses_the_real_desktop_size_not_the_panel_mode() {
        let d = parse_displays(TWO_DISPLAYS);
        assert_eq!(d.len(), 2, "got {d:?}");
        assert_eq!(d[0].name, "Color LCD");
        // The whole point: 1440x900, NOT the 2560x1600 panel mode and NOT a halving
        // of it (1280x800) — both of which this got wrong before.
        assert_eq!(d[0].desktop, (1440, 900));
        assert_eq!(d[0].native, (2880, 1800), "backing store = what `max` means");
        assert!(d[0].main);
        assert_eq!(d[1].name, "DELL P2722HE");
        assert_eq!(d[1].desktop, (1920, 1080));
        assert_eq!(d[1].native, (1920, 1080));
        assert!(!d[1].main);
    }

    #[test]
    fn picks_the_main_display_by_default() {
        let d = parse_displays(TWO_DISPLAYS);
        assert_eq!(pick_display(&d, None).unwrap().name, "Color LCD");
        assert_eq!(pick_display(&d, Some("dell")).unwrap().name, "DELL P2722HE");
        assert_eq!(pick_display(&d, Some("P2722")).unwrap().name, "DELL P2722HE");
        // An unmatched or blank name falls back to main rather than failing a boot.
        assert_eq!(pick_display(&d, Some("nope")).unwrap().name, "Color LCD");
        assert_eq!(pick_display(&d, Some("  ")).unwrap().name, "Color LCD");
    }

    #[test]
    fn res_fields_parse_with_and_without_refresh() {
        assert_eq!(parse_res_field("1440 x 900 @ 60.00Hz"), Some((1440, 900)));
        assert_eq!(parse_res_field("2880 x 1800"), Some((2880, 1800)));
        assert_eq!(parse_res_field("1440x900"), None, "system_profiler always spaces it");
        assert_eq!(parse_res_field(""), None);
        assert_eq!(parse_res_field("garbage here now"), None);
    }

    #[test]
    fn json_field_extraction_is_exact() {
        let o = r#"{ "_name" : "Color LCD", "spdisplays_main" : "spdisplays_yes" }"#;
        assert_eq!(json_str_field(o, "_name").as_deref(), Some("Color LCD"));
        assert_eq!(json_str_field(o, "spdisplays_main").as_deref(), Some("spdisplays_yes"));
        assert_eq!(json_str_field(o, "absent"), None);
    }

    #[test]
    fn object_splitting_is_brace_matched() {
        let objs = ndrvs_objects(TWO_DISPLAYS);
        assert_eq!(objs.len(), 2);
        assert!(objs[0].contains("Color LCD") && !objs[0].contains("DELL"));
        assert!(objs[1].contains("DELL"));
        // No displays array at all → nothing, not a panic.
        assert!(ndrvs_objects("{}").is_empty());
        assert!(ndrvs_objects("").is_empty());
    }

    #[test]
    fn a_display_with_no_size_is_skipped() {
        let t = r#"{ "spdisplays_ndrvs" : [ { "_name" : "Headless" } ] }"#;
        assert!(parse_displays(t).is_empty());
        // Only `_spdisplays_pixels` present → used for both.
        let t = r#"{ "spdisplays_ndrvs" : [ { "_name" : "P", "_spdisplays_pixels" : "800 x 600" } ] }"#;
        let d = parse_displays(t);
        assert_eq!(d[0].desktop, (800, 600));
        assert_eq!(d[0].native, (800, 600));
    }

    #[test]
    fn empty_or_garbage_input_yields_nothing() {
        assert!(parse_displays("").is_empty());
        assert!(parse_displays("not json at all").is_empty());
        assert!(pick_display(&[], None).is_none());
    }

    #[test]
    fn valid_resolution_accepts_only_wxh() {
        assert!(valid_resolution("1920x1080"));
        assert!(valid_resolution("1920x1080x32"));
        assert!(!valid_resolution("1920"));
        assert!(!valid_resolution("1920x"));
        assert!(!valid_resolution("x1080"));
        assert!(!valid_resolution("1920x1080x"));
        assert!(!valid_resolution("1920X1080"), "Limine wants a lowercase x");
        assert!(!valid_resolution("0x1080"));
        assert!(!valid_resolution("abcxdef"));
        assert!(!valid_resolution(""));
    }

    #[test]
    fn boot_cfg_drops_the_bpp_the_stub_cannot_use() {
        // A GOP mode is chosen by dimensions; Limine's depth component would be
        // written out and silently ignored, so it is stripped here instead.
        let c = boot_cfg_contents("1920x1080x32").expect("valid");
        assert!(c.contains("resolution=1920x1080\n"), "{c}");
        assert!(!c.contains("x32"), "{c}");
        let c = boot_cfg_contents("1280x720").expect("valid");
        assert!(c.contains("resolution=1280x720\n"), "{c}");
        // Every non-comment line must be a key=value the stub's parser accepts.
        for line in c.lines().filter(|l| !l.trim_start().starts_with('#') && !l.trim().is_empty()) {
            assert!(line.contains('='), "unparseable line {line:?}");
        }
        // Junk is a hard error at the call site, not a file with a bad value in it.
        assert!(boot_cfg_contents("1920").is_none());
        assert!(boot_cfg_contents("").is_none());
        assert!(boot_cfg_contents("0x1080").is_none());
    }
}

#[cfg(test)]
mod sample_file_tests {
    use super::*;

    /// The corpus is embedded in the kernel image, so a duplicate destination
    /// would silently shadow one entry (the later fetch overwrites the file) and
    /// a category typo would put a file in a `/samples/` folder nothing lists.
    #[test]
    fn every_sample_has_a_unique_destination_in_a_known_category() {
        let mut seen: Vec<String> = Vec::new();
        for s in SAMPLE_FILES {
            assert!(
                matches!(s.category, "images" | "videos" | "audios" | "misc" | "js" | "html"),
                "{}: unknown category {:?}",
                s.name,
                s.category
            );
            assert!(!s.note.is_empty(), "{}: no provenance note", s.name);
            if s.is_local() {
                assert!(
                    s.local_src().is_file(),
                    "{}: local sample missing at {}",
                    s.name,
                    s.local_src().display()
                );
            } else {
                assert!(
                    s.url.starts_with("http://") || s.url.starts_with("https://"),
                    "{}: not an http(s) url: {}",
                    s.name,
                    s.url
                );
            }
            let dst = format!("{}/{}", s.category, s.name);
            assert!(!seen.contains(&dst), "duplicate sample destination {dst}");
            seen.push(dst);
        }
    }

    /// Every extension claimed openable must be one the OS can actually open —
    /// otherwise a `.flac` sample creeps in as a file that only ever errors. A
    /// format with no decoder yet is allowed, but only when it says so, and its
    /// note must warn the human reading `/samples/README.md`.
    #[test]
    fn every_sample_extension_is_one_the_os_can_open() {
        // Media hooks (agents/media + agents/pdf manifests) plus the text kinds
        // that fall through to the editor, plus `.js` for `/js` (and `/open` → editor).
        const OPENABLE: &[&str] = &[
            "png", "jpg", "jpeg", "wav", "mp3", "aac", "mp4", "mov", "mkv", "webm", "pdf", "txt",
            "json", "csv", "html", "md", "js", "css",
        ];
        for s in SAMPLE_FILES {
            let ext = s.name.rsplit('.').next().unwrap_or_default();
            if s.openable {
                assert!(OPENABLE.contains(&ext), "{}: /open cannot handle .{ext}", s.name);
            } else {
                assert!(
                    !OPENABLE.contains(&ext),
                    "{}: marked unopenable but .{ext} has a decoder",
                    s.name
                );
                assert!(
                    s.note.contains("NO decoder"),
                    "{}: an unopenable sample must say so in its note",
                    s.name
                );
            }
        }
    }

    /// `make` passes `CHITTI_SAMPLE_FILES` through unconditionally, so an empty
    /// value must read as "off" — the trap `CHITTI_RESOLUTION` hit. Explicitly
    /// negative values are off too, so a wrapper can disable without unsetting.
    #[test]
    fn samples_requested_treats_empty_as_unset() {
        // Serialised by construction: one test touches this variable.
        let restore = env::var("CHITTI_SAMPLE_FILES").ok();
        for (val, want) in [
            ("", false),
            ("0", false),
            ("off", false),
            ("no", false),
            ("false", false),
            ("1", true),
            ("yes", true),
            ("true", true),
            (" 1 ", true),
        ] {
            env::set_var("CHITTI_SAMPLE_FILES", val);
            assert_eq!(samples_requested(), want, "CHITTI_SAMPLE_FILES={val:?}");
        }
        env::remove_var("CHITTI_SAMPLE_FILES");
        assert!(!samples_requested(), "unset must be off");
        if let Some(v) = restore {
            env::set_var("CHITTI_SAMPLE_FILES", v);
        }
    }

    /// The README is the provenance record and is itself embedded, so it must
    /// name every file and its source even when nothing has been fetched yet.
    #[test]
    fn readme_lists_every_sample_and_its_source() {
        // Rendered against an empty "present" set: the absent case is the one a
        // fresh clone produces.
        let dir = samples_dir();
        let existing = fs::read_to_string(dir.join("README.md")).ok();
        write_samples_readme(&[]).expect("render readme");
        let md = fs::read_to_string(dir.join("README.md")).expect("readme written");
        for s in SAMPLE_FILES {
            assert!(md.contains(s.name), "readme omits {}", s.name);
            if s.is_local() {
                assert!(
                    md.contains("samples-src") || md.contains(s.category),
                    "readme omits local provenance for {}",
                    s.name
                );
            } else {
                assert!(md.contains(s.url), "readme omits the source of {}", s.name);
            }
        }
        assert!(md.contains("absent"), "sizes should report absent files as such");
        // Leave the tree as it was found (a real fetch rewrites it with sizes).
        match existing {
            Some(prev) => fs::write(dir.join("README.md"), prev).unwrap(),
            None => {
                let _ = fs::remove_file(dir.join("README.md"));
            }
        }
    }
}

#[cfg(test)]
mod usb_host_tests {
    use super::*;

    #[test]
    fn parse_vid_pid_accepts_hex_forms() {
        assert_eq!(
            parse_vid_pid("0a12:0001").as_ref().map(|(a, b)| (a.as_str(), b.as_str())),
            Some(("0a12", "0001"))
        );
        assert_eq!(
            parse_vid_pid("0x0A12:0x0001").as_ref().map(|(a, b)| (a.as_str(), b.as_str())),
            Some(("0a12", "0001"))
        );
        assert!(parse_vid_pid("bad").is_none());
        assert!(parse_vid_pid("zzzz:0001").is_none());
    }

    #[test]
    fn system_profiler_grep_finds_bluetooth_and_camera() {
        let sample = r#"
USB:

    USB 3.1 Bus:

      Host Controller Driver: AppleUSBXHCITR

        CSR8510 A10:

          Product ID: 0x0001
          Vendor ID: 0x0a12  (Cambridge Silicon Radio Ltd.)
          Version: 1.00
          Manufacturer: CSR
          Location ID: 0x00100000

        HD Pro Webcam C920:

          Product ID: 0x082d
          Vendor ID: 0x046d  (Logitech Inc.)
          Version: 0.11
          Serial Number: 1234
          Location ID: 0x00200000
"#;
        let bt = parse_system_profiler_usb(sample, &["bluetooth", "csr8510"]);
        assert_eq!(bt, vec![("0a12".into(), "0001".into())]);
        let cam = parse_system_profiler_usb(sample, &["camera", "webcam"]);
        assert_eq!(cam, vec![("046d".into(), "082d".into())]);
    }

    #[test]
    fn lsusb_grep_finds_by_description() {
        let sample = "\
Bus 001 Device 003: ID 0a12:0001 Cambridge Silicon Radio, Ltd Bluetooth Dongle (HCI mode)
Bus 001 Device 004: ID 046d:082d Logitech, Inc. HD Pro Webcam C920
Bus 001 Device 005: ID 0781:5581 SanDisk Corp. Ultra
";
        let bt = parse_lsusb(sample, &["bluetooth"]);
        assert_eq!(bt, vec![("0a12".into(), "0001".into())]);
        let cam = parse_lsusb(sample, &["webcam", "camera"]);
        assert_eq!(cam, vec![("046d".into(), "082d".into())]);
    }
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
    // Count the payload the ESP must actually hold, kernel + stub included: the
    // kernel is not a fixed size (a debug build, or an embedded `/samples`
    // corpus, adds tens of MiB) and a fixed 64 MiB base silently overflows the
    // volume instead of reporting anything.
    let img_bytes: u64 = fs::metadata(kernel).map(|m| m.len()).unwrap_or(0)
        + fs::metadata(stub).map(|m| m.len()).unwrap_or(0);
    let size_mb = 64
        + ((model_bytes + voice_bytes + wifi_bytes + img_bytes) / (1024 * 1024))
        + if model_bytes > 0 { 64 } else { 0 }
        + if wifi_bytes > 0 { 8 } else { 0 };
    // Recreate the image only when contents changed (cheap heuristic: sizes).
    let f = fs::OpenOptions::new().create(true).write(true).truncate(true).open(&img).map_err(|e| e.to_string())?;
    f.set_len(size_mb * 1024 * 1024).map_err(|e| e.to_string())?;
    drop(f);
    // A crashed run can leave this image attached to a raw /dev/diskN (the
    // model-disk variant of this bug); the next hdiutil attach would return the
    // stale device and newfs_msdos fails on a node the user does not own.
    #[cfg(target_os = "macos")]
    detach_stale_hdiutil(&img);
    let disp_cfg = boot_display_cfg()?;
    // The hosted-model boot seed (`\chitti-model.json`): the stub hands it to
    // the kernel via the boot-info page on UEFI/stub boots (VirtualBox, real
    // hardware) — the image equivalent of the QEMU `-kernel` fw_cfg seed.
    let remote_cfg = remote_model_json()?;
    if cfg!(target_os = "linux") {
        let mut copies: Vec<(PathBuf, String)> = vec![
            (stub.to_path_buf(), "::/EFI/BOOT/BOOTAA64.EFI".into()),
            (kernel.to_path_buf(), "::/chitti-kernel".into()),
        ];
        if let Some(c) = &disp_cfg {
            copies.push((c.clone(), "::/chitti-display.cfg".into()));
        }
        if let Some(rc) = &remote_cfg {
            copies.push((rc.clone(), "::/chitti-model.json".into()));
        }
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
    let remote_cp = remote_cfg
        .as_ref()
        .map(|rc| format!("cp \"{}\" \"$MNT/chitti-model.json\"", rc.display()))
        .unwrap_or_default();
    let script = format!(
        r#"set -e
DEV=$(hdiutil attach -nomount -imagekey diskimage-class=CRawDiskImage "{img}" | head -1 | awk '{{print $1}}')
newfs_msdos -F 32 -v CHITTI "$DEV" > /dev/null
diskutil mount "$DEV" > /dev/null
MNT=$(diskutil info "$DEV" | awk -F': *' '/Mount Point/{{print $2}}')
mkdir -p "$MNT/EFI/BOOT"
cp "{stub}" "$MNT/EFI/BOOT/BOOTAA64.EFI"
cp "{kernel}" "$MNT/chitti-kernel"
{disp_cp}
{model_cp}
{voice_cp}
{wifi_cp}
{remote_cp}
diskutil unmount "$DEV" > /dev/null
hdiutil detach "$DEV" > /dev/null
"#,
        img = img.display(),
        stub = stub.display(),
        kernel = kernel.display(),
        disp_cp = disp_cfg.as_ref().map(|c| format!("cp \"{}\" \"$MNT/chitti-display.cfg\"", c.display())).unwrap_or_default(),
        model_cp = model_cp,
        remote_cp = remote_cp,
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
/// Detach any hdiutil raw-disk attachment of `img` (macOS only). `hdiutil
/// info` groups a `/dev/diskN` line above its `image-path:` line, so track the
/// last device seen and detach it when its image-path matches. A stale
/// attachment makes the next `newfs_msdos` fail on a device the user does not
/// own, and it is exactly what a killed parallel e2e run leaves behind.
#[cfg(target_os = "macos")]
fn detach_stale_hdiutil(img: &std::path::Path) {
    use std::process::Command;
    let Ok(out) = Command::new("hdiutil").arg("info").output() else { return };
    let txt = String::from_utf8_lossy(&out.stdout);
    let img_s = img.to_string_lossy();
    let mut last_dev: Option<String> = None;
    for line in txt.lines() {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix("/dev/disk") {
            let name: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
            last_dev = Some(format!("/dev/disk{name}"));
        } else if t.contains("image-path") && t.contains(img_s.as_ref()) {
            if let Some(dev) = last_dev.take() {
                eprintln!("  hdiutil: detaching stale attachment {dev} of {}", img_s);
                let _ = Command::new("hdiutil").args(["detach", &dev]).status();
            }
        }
    }
}

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
    let want = copies.iter().map(|(p, d)| format!("{}>{d}", p.display())).collect::<Vec<_>>().join(":") + &format!(":{total}");
    // Content-addressed name: under `--jobs`, two shards build *different* model
    // disks concurrently (open_media's media disk vs open_video's clip), and a
    // shared fixed path made them race the same image — one truncated the
    // other's file and the attach/format collided. The hash in the name makes
    // each content's image distinct, so concurrent builds are independent.
    let mut h = std::collections::hash_map::DefaultHasher::new();
    use std::hash::Hasher;
    h.write(want.as_bytes());
    let img = repo_root().join(format!("target/chitti-model-disk-{:016x}.img", h.finish()));
    if img.exists() {
        return Ok(img);
    }
    // A crashed previous run (or a parallel one that got killed) can leave this
    // image attached to a raw /dev/diskN; the next hdiutil attach then returns
    // that *stale* device and newfs_msdos fails with "Permission denied" on a
    // node it does not own. Detach any stale attachment before rebuilding.
    #[cfg(target_os = "macos")]
    detach_stale_hdiutil(&img);
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
/// `CHITTI_GDB=1` opens QEMU's gdbstub on :1234 so a debugger can break into a
/// *hung* guest and get a backtrace. That is the only way to answer "which
/// native loop is the kernel stuck in": a spin inside native code never reaches
/// the cooperative UI pump, so nothing inside the guest can report it — not the
/// script budget, not Ctrl+C, not ktrace. `CHITTI_GDB=wait` also halts at reset
/// so breakpoints can be set before boot.
///
/// Usage: `CHITTI_GDB=1 cargo xtask run -arch aarch64`, then
/// `lldb kernel/target/aarch64-chitti/release/chitti-kernel -o "gdb-remote 1234"`
/// and `process interrupt` + `bt`.
fn gdbstub_args(qemu: &mut Command) {
    match std::env::var("CHITTI_GDB").as_deref() {
        Ok("wait") => {
            qemu.args(["-gdb", "tcp::1234", "-S"]);
        }
        Ok(v) if !v.is_empty() => {
            qemu.args(["-gdb", "tcp::1234"]);
        }
        _ => {}
    }
}

fn cmd_run_aarch64_uefi(model: Model, disk: Option<PathBuf>, disk_only: bool, no_model: bool) -> Result<(), String> {
    let elf = build_kernel_aarch64(true, model.features())?;
    assert_identity_kernel(&elf)?;
    let stub = build_stub_aarch64()?;
    let mut qemu = Command::new("qemu-system-aarch64");
    qemu.args(["-M", "virt", "-smp", &smp_count(), "-m", model.qemu_mem()]);
    qemu.args(accel_args("aarch64"));
    gdbstub_args(&mut qemu);
    for a in aavmf_pflash_args()? {
        qemu.arg(a);
    }
    for a in gpu_device_args("aarch64", true) {
        qemu.arg(a);
    }
    qemu.args(["-device", "virtio-keyboard-device", "-device", "virtio-tablet-device"]);
    if wants_usb_host() {
        qemu.args(["-device", "qemu-xhci,id=xhci"]);
        for a in usb_host_device_args("xhci.0") {
            qemu.arg(a);
        }
    }
    qemu.args(display_args());
    // Same host-derived framebuffer resolution as the `-kernel` path, so the
    // UEFI ramfb fallback matches VBox GOP / QEMU direct (was 1920x1080 fixed).
    for a in ramfb_res_fw_cfg() {
        qemu.arg(a);
    }
    qemu.args(["-serial", "mon:stdio"]);
    for a in share_args(false) {
        qemu.arg(a);
    }
    for a in clipboard_args(false) {
        qemu.arg(a);
    }
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
    gdbstub_args(&mut qemu);
    for a in gpu_device_args("aarch64", false) {
        qemu.arg(a);
    }
    qemu.args([
        "-device", "virtio-keyboard-device",
        // A virtio tablet gives the window an absolute-position mouse.
        "-device", "virtio-tablet-device",
    ]);
    // Host USB BT/cam need an xHCI bus (plain -kernel aarch64 has no USB by default).
    if wants_usb_host() {
        qemu.args(["-device", "qemu-xhci,id=xhci"]);
        for a in usb_host_device_args("xhci.0") {
            qemu.arg(a);
        }
    }
    // Resizable graphical window (the ramfb surface scales to fit).
    qemu.args(display_args());
    qemu.args(["-serial", "mon:stdio", "-kernel"]);
    qemu.arg(&elf);
    for a in share_args(false) {
        qemu.arg(a);
    }
    for a in clipboard_args(false) {
        qemu.arg(a);
    }
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
    // Fetch the /samples corpus first when it was asked for: the kernel's build
    // script embeds whatever is on disk *at compile time*, so a fetch afterwards
    // would silently produce an image with no samples. No-op unless
    // CHITTI_SAMPLE_FILES is set, so `test` / `ref-check` builds are unchanged.
    ensure_sample_files();
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

/// Whether `s` is a Limine `resolution:` value — `<width>x<height>` with an
/// optional `x<bpp>`. Validated rather than passed through, because a typo would
/// otherwise be silently ignored by Limine and read as "EDID detection is broken".
fn valid_resolution(s: &str) -> bool {
    let parts: Vec<&str> = s.split('x').collect();
    if parts.len() != 2 && parts.len() != 3 {
        return false;
    }
    parts.iter().all(|p| {
        !p.is_empty() && p.chars().all(|c| c.is_ascii_digit()) && p.parse::<u32>().is_ok_and(|n| n > 0)
    })
}

/// Body of the loader's display-preference file (`\chitti-display.cfg` on the
/// ESP) for a pinned `CHITTI_RESOLUTION`, or `None` if it names no usable size.
///
/// `CHITTI_RESOLUTION` carries Limine's optional `xBPP` third component, which the
/// stub's parser rejects — a GOP mode is selected by dimensions alone — so the
/// depth is dropped here rather than written out and silently ignored.
fn boot_cfg_contents(res: &str) -> Option<String> {
    if !valid_resolution(res) {
        return None;
    }
    let mut parts = res.split('x');
    let (w, h) = (parts.next()?, parts.next()?);
    Some(format!(
        "# ChittiOS loader display preference — written by `cargo xtask image`.\n\
         # The UEFI stub sets this GOP mode before the kernel starts, so it wins over\n\
         # both the display's EDID-native mode and any hypervisor resolution setting.\n\
         resolution={w}x{h}\n"
    ))
}

/// Write the display-preference file for the ESP, returning its path, or `None`
/// when nothing is pinned.
///
/// This is the aarch64 half of `CHITTI_RESOLUTION` — the x86 half rewrites
/// `limine.conf`. It exists because a hypervisor's own resolution knob cannot be
/// relied on: VirtualBox-ARM stores `VBoxInternal2/EfiGraphicsResolution` and then
/// boots its guest at a different size anyway, leaving no way to ask for a
/// framebuffer that fits the window.
fn boot_display_cfg() -> Result<Option<PathBuf>, String> {
    let Ok(res) = env::var("CHITTI_RESOLUTION") else { return Ok(None) };
    let res = res.trim();
    // Empty means unset: `make vbox` passes `CHITTI_RESOLUTION='$(VBOX_RES)'`
    // unconditionally, and VBOX_RES is empty unless the human named a size.
    if res.is_empty() {
        return Ok(None);
    }
    let Some(body) = boot_cfg_contents(res) else {
        return Err(format!("CHITTI_RESOLUTION='{res}' is not <width>x<height>[x<bpp>] (e.g. 1920x1080)"));
    };
    let path = repo_root().join("target/chitti-display.cfg");
    fs::write(&path, body).map_err(|e| format!("writing {}: {e}", path.display()))?;
    eprintln!("  ESP display pref: resolution={res} (stub sets this GOP mode; unset CHITTI_RESOLUTION for EDID-native)");
    Ok(Some(path))
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

    // Framebuffer mode: `limine.conf` deliberately pins none, so Limine uses the
    // display's EDID-preferred (native) mode — the same discovery the aarch64 stub
    // does. `CHITTI_RESOLUTION=WxH[xBPP]` appends an explicit override for the
    // cases EDID can't answer: a headless VM, or matching a fixed window size.
    // Empty is unset, so a wrapper can pass the variable through unconditionally.
    if let Some(res) = std::env::var("CHITTI_RESOLUTION").ok().map(|v| v.trim().to_string()).filter(|v| !v.is_empty()) {
        let res = res.as_str();
        if !valid_resolution(res) {
            return Err(format!(
                "CHITTI_RESOLUTION='{res}' is not <width>x<height>[x<bpp>] (e.g. 1920x1080)"
            ));
        }
        let conf_path = iso_root.join("boot/limine/limine.conf");
        let mut conf = fs::read_to_string(&conf_path).map_err(|e| e.to_string())?;
        conf.push_str(&format!("    resolution: {res}\n"));
        fs::write(&conf_path, conf).map_err(|e| format!("appending resolution: {e}"))?;
        eprintln!("xtask: pinned framebuffer resolution to {res} (unset CHITTI_RESOLUTION for EDID-native)");
    }

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
    // A crashed run can leave this image attached to a raw /dev/diskN; the next
    // hdiutil attach would return the stale device and newfs_msdos fails.
    #[cfg(target_os = "macos")]
    detach_stale_hdiutil(&img);

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
        if let Some(c) = boot_display_cfg()? {
            copies.push((c, "::/chitti-display.cfg".into()));
        }
        // The hosted-model boot seed (`\chitti-model.json`): the stub hands it
        // to the kernel via the boot-info page on UEFI/stub boots — the image
        // analogue of the QEMU `-kernel` fw_cfg seed.
        if let Some(rc) = remote_model_json()? {
            copies.push((rc, "::/chitti-model.json".into()));
        }
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
    let remote_cp = remote_model_json()?
        .map(|rc| format!("cp \"{}\" \"$MNT/chitti-model.json\"", rc.display()))
        .unwrap_or_default();
    let script = format!(
        r#"set -e
DEV=$(hdiutil attach -nomount -imagekey diskimage-class=CRawDiskImage "{img}" | head -1 | awk '{{print $1}}')
newfs_msdos -F 32 -v CHITTI "${{DEV}}s1" > /dev/null
diskutil mount "${{DEV}}s1" > /dev/null
MNT=$(diskutil info "${{DEV}}s1" | awk -F': *' '/Mount Point/{{print $2}}')
mkdir -p "$MNT/EFI/BOOT"
cp "{stub}" "$MNT/EFI/BOOT/BOOTAA64.EFI"
cp "{kernel}" "$MNT/chitti-kernel"
{disp_cp}
{model_cp}
{voice_cp}
{wifi_cp}
{remote_cp}
diskutil unmount "${{DEV}}s1" > /dev/null
"{mke2fs}" -F -q -t ext4 -b 4096 "${{DEV}}s2"
hdiutil detach "$DEV" > /dev/null
"#,
        img = img.display(),
        stub = stub.display(),
        kernel = elf.display(),
        mke2fs = mke2fs.display(),
        disp_cp = boot_display_cfg()?.map(|c| format!("cp \"{}\" \"$MNT/chitti-display.cfg\"", c.display())).unwrap_or_default(),
        model_cp = model_cp,
        remote_cp = remote_cp,
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
//
// Interactive `run` and `ref-check` stay single-CPU (`-smp` only on the
// test runner): inference is BSP-bound, and under single-thread TCG extra
// vCPUs only cost round-robin overhead. `-cpu max` exposes AVX2/XSAVE.

fn qemu_base_cmd(iso: &Path, mem: &str) -> Command {
    let mut cmd = Command::new("qemu-system-x86_64");
    // `-m` comes from the model (0.8b → 3G, 9b → 10G, …).
    cmd.args([
        "-M",
        "q35",
        "-cpu",
        "max",
        "-m",
        mem,
        "-device",
        "isa-debug-exit,iobase=0xf4,iosize=0x04",
        "-no-reboot",
    ]);
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
        // `mon:stdio` multiplexes QEMU's monitor onto the same stdio as the guest
        // serial (Ctrl+A c toggles), which is what lets a test press hardware buttons:
        // `system_powerdown` sets the ACPI power-button status bit and `system_wakeup`
        // resumes from S3. aarch64 has always run this way; x86 used plain `stdio`, so
        // those scenarios had no monitor to talk to on the arch that implements both.
        cmd.args(["-serial", "mon:stdio"]);
        cmd.arg("-drive").arg(format!("file={},if=none,id=chittidisk,format=raw", disk.display()));
        cmd.args(["-device", "virtio-blk-pci,drive=chittidisk,disable-modern=on"]);
        cmd.args(["-device", "qemu-xhci,id=xhci", "-device", "usb-kbd,bus=xhci.0"]);
        for a in usb_host_device_args("xhci.0") {
            cmd.arg(a);
        }
        // NIC (user-net or CHITTI_NET_BRIDGE — see guest_netdev); model from CHITTI_NIC.
        cmd.args(["-netdev", &guest_netdev("chittinet"), "-device", &format!("{},netdev=chittinet", nic_model())]);
        for a in share_args(true) {
            cmd.arg(a);
        }
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
    let mem = model.qemu_mem();
    let mut cmd = qemu_base_cmd(&iso, mem);
    if uefi {
        for arg in ovmf_pflash_args()? {
            cmd.arg(arg);
        }
        eprintln!("booting via UEFI (OVMF) -- the same GOP framebuffer path real hardware uses");
    }
    cmd.args(["-serial", "mon:stdio"]);
    cmd.args(x86_run_extra_args()); // headless (-display none) + KVM on CI
    // Cocoa/GTK window for interactive use (skip when CHITTI_DISPLAY=none).
    if env::var("CHITTI_DISPLAY").as_deref() != Ok("none") {
        for a in display_args() {
            cmd.arg(a);
        }
    }
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
    // Optional host BT dongle / UVC webcam passthrough (CHITTI_USB_BT / _CAM / _HOST).
    for a in usb_host_device_args("xhci.0") {
        cmd.arg(a);
    }
    // NIC (user-net or CHITTI_NET_BRIDGE — see guest_netdev); model from CHITTI_NIC.
    cmd.args(["-netdev", &guest_netdev("chittinet"), "-device", &format!("{},netdev=chittinet", nic_model())]);
    // Optional host folder over virtio-9p (CHITTI_SHARE), mounted at /host.
    for a in share_args(true) {
        cmd.arg(a);
    }
    // Optional SPICE clipboard agent channel (CHITTI_CLIPBOARD).
    for a in clipboard_args(true) {
        cmd.arg(a);
    }
    // virtio-snd on the host's audio backend (mic + speaker) for /voice.
    // Missing host audio is noisy but non-fatal; use CHITTI_AUDIO=off to silence.
    cmd.args(audio_args("virtio-sound-pci"));
    if disk_size.is_some() {
        eprintln!("  disk: {} ({}) -- run `/install yes` at the shell, then reboot with `--disk-only`", disk.display(), disk_size.as_deref().unwrap_or(""));
    }
    eprintln!(
        "booting x86_64 Chitti ({}) under QEMU -m {mem} (close the window or Ctrl-A X to quit)",
        model.label()
    );
    let status = cmd
        .status()
        .map_err(|e| format!("failed to spawn qemu-system-x86_64: {e}"))?;
    eprintln!("qemu exited: {status}");
    if status.success() {
        eprintln!(
            "  (exit 0 = window closed or guest poweroff; if it quit instantly, \
             check the lines above for USB/audio errors)"
        );
    }
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

// --- sample files (`/samples/…` in the booted OS) --------------------------

/// One bundled sample file: the `/samples/<category>/` folder it lands in, the
/// name it takes there, where it is fetched from, and what it is for.
///
/// `category` is the folder under `assets/samples/` **and** under `/samples/`
/// in the booted OS, so the two cannot drift: the kernel's build script walks
/// the directory rather than carrying a second copy of this table.
struct SampleFile {
    category: &'static str,
    name: &'static str,
    /// Remote `http(s)://…` URL to curl, **or empty** for a **local** sample
    /// checked in at `assets/samples-src/<category>/<name>` (authored scripts
    /// such as the `/js` demos — not fetched, never redistributed from a third
    /// party).
    url: &'static str,
    /// Provenance + what it exercises. Printed by the fetch and written into
    /// the in-OS `/samples/README.md`, because a sample whose origin and
    /// licence are unrecorded is a sample nobody can ship an image with.
    note: &'static str,
    /// False for a format the OS has **no decoder for yet**. Such a file is
    /// deliberately included (it is the next decoder's input), but it is marked
    /// so the corpus cannot quietly accumulate files that only ever produce an
    /// error — `every_sample_extension_is_one_the_os_can_open` holds for the
    /// rest, and the README says which ones will not play.
    openable: bool,
}

impl SampleFile {
    fn is_local(&self) -> bool {
        self.url.is_empty()
    }

    /// Host path of a local sample (`assets/samples-src/<cat>/<name>`).
    fn local_src(&self) -> PathBuf {
        repo_root()
            .join("assets/samples-src")
            .join(self.category)
            .join(self.name)
    }
}

/// Every entry must be **freely redistributable**, and that is a stricter rule
/// than "fetched, never committed" implies. A build with `CHITTI_SAMPLE_FILES`
/// embeds these bytes in a kernel image, and images get passed around — so an
/// arXiv paper under the default non-exclusive distribution licence (which
/// permits arXiv to distribute it, not third parties) does not belong here even
/// though this tree would carry no copy of it. `/http -O <url>` fetches such a
/// document in seconds on the running OS, which is where it should come from.
///
/// The sample corpus. Chosen to cover **every decoder `/open` can reach** —
/// PNG (RGB / RGBA / grayscale), baseline JPEG, H.264+AAC mp4, PCM WAV, MP3
/// (with an ID3v2 tag, which the decoder must skip), ADTS AAC, PDF — plus the
/// text kinds that land in the editor with syntax highlighting.
///
/// Every URL is a **well-known, stable sample source** (upstream project test
/// data, not a random file host), and every file is small on purpose: the whole
/// corpus is embedded in the kernel image, so this is ~2 MiB, not ~200.
const SAMPLE_FILES: &[SampleFile] = &[
    // Images — the PNG + baseline-JPEG paths of `image/`.
    SampleFile {
        category: "images",
        name: "fruits.jpg",
        url: "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/fruits.jpg",
        note: "baseline JPEG, colour photo (OpenCV samples/data, Apache-2.0)",
        openable: true,
    },
    SampleFile {
        category: "images",
        name: "baboon.jpg",
        url: "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/baboon.jpg",
        note: "baseline JPEG, high-detail (OpenCV samples/data, Apache-2.0)",
        openable: true,
    },
    SampleFile {
        category: "images",
        name: "sudoku.png",
        url: "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/sudoku.png",
        note: "PNG photo, large IDAT (OpenCV samples/data, Apache-2.0)",
        openable: true,
    },
    SampleFile {
        category: "images",
        name: "transparency.png",
        url: "https://raw.githubusercontent.com/glennrp/libpng/libpng16/contrib/pngsuite/basn6a08.png",
        note: "PNG RGBA 8-bit, alpha channel (libpng PNGSuite, basn6a08.png)",
        openable: true,
    },
    SampleFile {
        category: "images",
        name: "grayscale.png",
        url: "https://raw.githubusercontent.com/glennrp/libpng/libpng16/contrib/pngsuite/basn0g08.png",
        note: "PNG grayscale 8-bit, colour type 0 (libpng PNGSuite, basn0g08.png)",
        openable: true,
    },
    // Video — H.264 in mp4 **with an AAC audio track**, so `/open` exercises
    // the demuxer, the decoder, the HUD, the audio-locked clock, and the AAC
    // playback path at once (test-videos.co.uk's 1–5 MB clips are video-only,
    // so a "with sound" sample must come from a source that keeps the audio).
    SampleFile {
        category: "videos",
        name: "flower.mp4",
        url: "https://interactive-examples.mdn.mozilla.net/media/cc0-videos/flower.mp4",
        note: "H.264 + AAC in mp4, ~1.1 MB, 30 s (MDN CC0-videos, flower.mp4 — public domain, has sound)",
        openable: true,
    },
    SampleFile {
        category: "videos",
        name: "friday.mp4",
        url: "https://interactive-examples.mdn.mozilla.net/media/cc0-videos/friday.mp4",
        note: "H.264 + AAC in mp4, ~0.5 MB (MDN CC0-videos, friday.mp4 — public domain, has sound)",
        openable: true,
    },
    // Audio — one file per decoder in `audio/`.
    SampleFile {
        category: "audios",
        name: "sample.wav",
        url: "https://raw.githubusercontent.com/rafaelreis-hotmart/Audio-Sample-files/master/sample.wav",
        note: "PCM WAV, 16-bit stereo 44.1 kHz — the stereo-downmix + resample path (rafaelreis-hotmart/Audio-Sample-files)",
        openable: true,
    },
    SampleFile {
        category: "audios",
        name: "sample.mp3",
        url: "https://raw.githubusercontent.com/rafaelreis-hotmart/Audio-Sample-files/master/sample.mp3",
        note: "MPEG Layer III, full-length track with an ID3v2 tag the decoder must skip — the largest sample here (~7 MiB) (rafaelreis-hotmart/Audio-Sample-files)",
        openable: true,
    },
    SampleFile {
        category: "audios",
        name: "sample.aac",
        url: "https://samples.ffmpeg.org/A-codecs/AAC/ct_faac-adts.aac",
        note: "ADTS AAC-LC, 44.1 kHz stereo (FFmpeg sample archive)",
        openable: true,
    },
    SampleFile {
        category: "audios",
        name: "sample.ogg",
        url: "https://raw.githubusercontent.com/rafaelreis-hotmart/Audio-Sample-files/master/sample.ogg",
        // Included on purpose and marked unopenable: there is no Vorbis decoder,
        // and `.ogg` is not in the media agent's `/open` hook, so this one falls
        // through to the **editor** (bytes, not audio) until one exists. Better a
        // labelled gap than a sample that looks broken.
        note: "Ogg Vorbis — NO decoder yet: `/open` puts it in the editor, it does not play (rafaelreis-hotmart/Audio-Sample-files)",
        openable: false,
    },
    // Misc — the pdf agent, and the text kinds that open in the editor.
    SampleFile {
        category: "misc",
        name: "minimal.pdf",
        url: "https://raw.githubusercontent.com/py-pdf/sample-files/main/001-trivial/minimal-document.pdf",
        note: "single-page PDF, classic xref table (py-pdf/sample-files)",
        openable: true,
    },
    SampleFile {
        category: "misc",
        name: "document.pdf",
        url: "https://raw.githubusercontent.com/py-pdf/sample-files/main/002-trivial-libre-office-writer/002-trivial-libre-office-writer.pdf",
        note: "PDF with extractable text, LibreOffice-produced (py-pdf/sample-files)",
        openable: true,
    },
    SampleFile {
        category: "misc",
        name: "pdflatex-4-pages.pdf",
        url: "https://raw.githubusercontent.com/py-pdf/sample-files/main/004-pdflatex-4-pages/pdflatex-4-pages.pdf",
        note: "4-page PDF, 24 KiB — the smallest multi-page document here, so page navigation is testable without a 2 MB download (py-pdf/sample-files, CC-BY-SA-4.0)",
        openable: true,
    },
    SampleFile {
        category: "misc",
        name: "pdflatex-image.pdf",
        url: "https://raw.githubusercontent.com/py-pdf/sample-files/main/003-pdflatex-image/pdflatex-image.pdf",
        note: "PDF with an embedded raster image (DCTDecode) — the renderer's JPEG-inside-PDF path, which no vector-only document reaches (py-pdf/sample-files, CC-BY-SA-4.0)",
        openable: true,
    },
    SampleFile {
        category: "misc",
        name: "geotopo.pdf",
        url: "https://raw.githubusercontent.com/py-pdf/sample-files/main/009-pdflatex-geotopo/GeoTopo-komprimiert.pdf",
        // The "long document with pictures" case, and the only sample where page
        // *navigation* is more than a formality: 117 pages, 19 embedded JPEGs
        // (photographic knot renderings with soft shadows), running headers,
        // boxed theorems and heavy maths. Pages cost ~0.5 s each in the
        // interpreter, so it is also the demonstration that a long document stays
        // usable where a figure-dense paper does not.
        note: "117-page LaTeX book with 19 embedded JPEG figures (Martin Thoma, GeoTopo) — long-document navigation + the raster-image path at scale (py-pdf/sample-files, CC-BY-SA-4.0)",
        openable: true,
    },
    SampleFile {
        category: "misc",
        name: "rfc1951-deflate.txt",
        url: "https://www.rfc-editor.org/rfc/rfc1951.txt",
        note: "plain text — the DEFLATE spec `image/inflate.rs` implements (IETF RFC 1951)",
        openable: true,
    },
    SampleFile {
        category: "misc",
        name: "cars.json",
        url: "https://raw.githubusercontent.com/vega/vega-datasets/main/data/cars.json",
        note: "JSON array, editor syntax highlighting (vega-datasets, BSD-3-Clause)",
        openable: true,
    },
    SampleFile {
        category: "misc",
        name: "seattle-weather.csv",
        url: "https://raw.githubusercontent.com/vega/vega-datasets/main/data/seattle-weather.csv",
        note: "CSV table (vega-datasets, BSD-3-Clause)",
        openable: true,
    },
    SampleFile {
        category: "misc",
        name: "first-web-page.html",
        url: "http://info.cern.ch/hypertext/WWW/TheProject.html",
        note: "HTML for the browser agent — the first web page (CERN)",
        openable: true,
    },
    // JS — scripts for the in-kernel `/js` engine (Node-style CLI). Authored
    // in-tree under `assets/samples-src/js/` and *copied* (not curl'd) so a
    // machine with no network still gets them when samples are requested.
    SampleFile {
        category: "js",
        name: "hello.js",
        url: "",
        note: "minimal /js script — console.log + top-level return (in-tree, ChittiOS)",
        openable: true,
    },
    SampleFile {
        category: "js",
        name: "argv.js",
        url: "",
        note: "prints process.argv / argv — run with `/js /samples/js/argv.js a b` (in-tree, ChittiOS)",
        openable: true,
    },
    SampleFile {
        category: "js",
        name: "fib.js",
        url: "",
        note: "iterative Fibonacci; optional N arg (in-tree, ChittiOS)",
        openable: true,
    },
    SampleFile {
        category: "js",
        name: "math.js",
        url: "",
        note: "Math + Array map/reduce demo (in-tree, ChittiOS)",
        openable: true,
    },
    SampleFile {
        category: "js",
        name: "class.js",
        url: "",
        note: "ES6 class + methods on the just engine (in-tree, ChittiOS)",
        openable: true,
    },
    SampleFile {
        category: "js",
        name: "json.js",
        url: "",
        note: "JSON.stringify / JSON.parse round-trip (in-tree, ChittiOS)",
        openable: true,
    },
    // HTML — pages for `/browse file:///samples/html/…` (authored in-tree;
    // relative CSS/JS resolve as file:/// subresources from the store).
    SampleFile {
        category: "html",
        name: "index.html",
        url: "",
        note: "browse landing page linking CSS/JS sample suites (in-tree, ChittiOS)",
        openable: true,
    },
    SampleFile {
        category: "html",
        name: "hello.html",
        url: "",
        note: "minimal HTML + inline script smoke test for /browse file:/// (in-tree, ChittiOS)",
        openable: true,
    },
    SampleFile {
        category: "html",
        name: "css-demo.html",
        url: "",
        note: "flex/grid/@media/calc/var/::before CSS demo for /browse (in-tree, ChittiOS)",
        openable: true,
    },
    SampleFile {
        category: "html",
        name: "css-suite.html",
        url: "",
        note: "CSS checklist: selectors, float/clear, tables, position, @import (in-tree, ChittiOS)",
        openable: true,
    },
    SampleFile {
        category: "html",
        name: "css-full.html",
        url: "",
        note: "large CSS visual suite (full.css @import theme) for /browse (in-tree, ChittiOS)",
        openable: true,
    },
    SampleFile {
        category: "html",
        name: "css-hn.html",
        url: "",
        note: "HN-like table + link colour cascade for /browse (in-tree, ChittiOS)",
        openable: true,
    },
    SampleFile {
        category: "html",
        name: "js-demo.html",
        url: "",
        note: "page that loads relative styles.css + app.js over file:/// (in-tree, ChittiOS)",
        openable: true,
    },
    SampleFile {
        category: "html",
        name: "js-suite.html",
        url: "",
        note: "self-checking DOM/storage/canvas/events suite for /browse (in-tree, ChittiOS)",
        openable: true,
    },
    SampleFile {
        category: "html",
        name: "js-full.html",
        url: "",
        note: "large self-checking DOM/Promise/Math/storage/canvas suite (in-tree, ChittiOS)",
        openable: true,
    },
    SampleFile {
        category: "html",
        name: "js-fetch.html",
        url: "",
        note: "self-checking fetch over file:///samples JSON (in-tree, ChittiOS)",
        openable: true,
    },
    SampleFile {
        category: "html",
        name: "js-iframe.html",
        url: "",
        note: "iframe src/srcdoc + postMessage self-delivery suite (in-tree, ChittiOS)",
        openable: true,
    },
    SampleFile {
        category: "html",
        name: "iframe-child.html",
        url: "",
        note: "iframe child page for postMessage samples (in-tree, ChittiOS)",
        openable: true,
    },
    SampleFile {
        category: "html",
        name: "js-dom.html",
        url: "",
        note: "interactive DOM list demo (createElement/classList/clicks) (in-tree, ChittiOS)",
        openable: true,
    },
    SampleFile {
        category: "html",
        name: "js-canvas.html",
        url: "",
        note: "canvas 2D paint/clear demo for /browse (in-tree, ChittiOS)",
        openable: true,
    },
    SampleFile {
        category: "html",
        name: "styles.css",
        url: "",
        note: "shared stylesheet for /samples/html (relative link target; in-tree, ChittiOS)",
        openable: true,
    },
    SampleFile {
        category: "html",
        name: "theme.css",
        url: "",
        note: "@import target for suite.css (ok/bad colours; in-tree, ChittiOS)",
        openable: true,
    },
    SampleFile {
        category: "html",
        name: "suite.css",
        url: "",
        note: "css-suite stylesheet with @import theme.css (in-tree, ChittiOS)",
        openable: true,
    },
    SampleFile {
        category: "html",
        name: "full.css",
        url: "",
        note: "css-full stylesheet with @import theme.css (in-tree, ChittiOS)",
        openable: true,
    },
    SampleFile {
        category: "html",
        name: "fetch-data.json",
        url: "",
        note: "JSON fixture for js-fetch suite (in-tree, ChittiOS)",
        openable: true,
    },
    SampleFile {
        category: "html",
        name: "app.js",
        url: "",
        note: "shared page script for /samples/html index + js-demo (in-tree, ChittiOS)",
        openable: true,
    },
    SampleFile {
        category: "html",
        name: "harness.js",
        url: "",
        note: "shared PASS/FAIL harness for large JS suites (in-tree, ChittiOS)",
        openable: true,
    },
    SampleFile {
        category: "html",
        name: "js-suite.js",
        url: "",
        note: "self-checking assertions for js-suite.html (in-tree, ChittiOS)",
        openable: true,
    },
    SampleFile {
        category: "html",
        name: "js-full.js",
        url: "",
        note: "assertions for js-full.html (in-tree, ChittiOS)",
        openable: true,
    },
    SampleFile {
        category: "html",
        name: "js-fetch.js",
        url: "",
        note: "assertions for js-fetch.html (in-tree, ChittiOS)",
        openable: true,
    },
    SampleFile {
        category: "html",
        name: "js-iframe.js",
        url: "",
        note: "assertions for js-iframe.html (in-tree, ChittiOS)",
        openable: true,
    },
    SampleFile {
        category: "html",
        name: "react-tw.html",
        url: "",
        note: "Vite React+Tailwind sample page (built from tools/react-tw; in-tree, ChittiOS)",
        openable: true,
    },
    SampleFile {
        category: "html",
        name: "react-tw.js",
        url: "",
        note: "React 18 IIFE bundle for react-tw.html (npm build artifact; in-tree, ChittiOS)",
        openable: true,
    },
    SampleFile {
        category: "html",
        name: "react-tw.css",
        url: "",
        note: "Tailwind CSS build for react-tw.html (npm build artifact; in-tree, ChittiOS)",
        openable: true,
    },
    SampleFile {
        category: "html",
        name: "shadcn.html",
        url: "",
        note: "shadcn/ui component gallery (built from tools/react-shadcn; in-tree, ChittiOS)",
        openable: true,
    },
    SampleFile {
        category: "html",
        name: "shadcn.js",
        url: "",
        note: "React 18 + shadcn/ui IIFE bundle for shadcn.html (npm build artifact; in-tree, ChittiOS)",
        openable: true,
    },
    SampleFile {
        category: "html",
        name: "shadcn.css",
        url: "",
        note: "Tailwind CSS build for shadcn.html (npm build artifact; in-tree, ChittiOS)",
        openable: true,
    },
    SampleFile {
        category: "html",
        name: "js-dom.js",
        url: "",
        note: "interactive list script for js-dom.html (in-tree, ChittiOS)",
        openable: true,
    },
    SampleFile {
        category: "html",
        name: "js-canvas.js",
        url: "",
        note: "canvas paint script for js-canvas.html (in-tree, ChittiOS)",
        openable: true,
    },
    SampleFile {
        category: "html",
        name: "README.md",
        url: "",
        note: "how to /browse the html samples (in-tree, ChittiOS)",
        openable: true,
    },
];

/// Where the corpus lives on the host. Gitignored: fetched, never committed —
/// the same rule the voice and WiFi assets follow, and the reason this needed
/// no redistribution decision.
fn samples_dir() -> PathBuf {
    repo_root().join("assets/samples")
}

/// Whether this build should embed the sample corpus, from `CHITTI_SAMPLE_FILES`.
///
/// **Empty means unset**: `make` passes the variable through unconditionally, so
/// `CHITTI_SAMPLE_FILES=` must read as "off" rather than as a request (the same
/// trap `CHITTI_RESOLUTION` hit). Anything explicitly negative is off too, so a
/// wrapper can disable it without unsetting.
fn samples_requested() -> bool {
    match env::var("CHITTI_SAMPLE_FILES") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            !v.is_empty() && !matches!(v.as_str(), "0" | "no" | "off" | "false")
        }
        Err(_) => false,
    }
}

/// Sample files already on disk, as `(category, name, bytes)`.
fn samples_present() -> Vec<(&'static str, &'static str, u64)> {
    let dir = samples_dir();
    SAMPLE_FILES
        .iter()
        .filter_map(|s| {
            let p = dir.join(s.category).join(s.name);
            fs::metadata(&p).ok().map(|m| (s.category, s.name, m.len()))
        })
        .collect()
}

/// Called from the build paths: when `CHITTI_SAMPLE_FILES` asks for samples,
/// fetch whatever is missing before the kernel's build script looks for them.
///
/// **Best-effort by design.** A failed download must not fail a build — the
/// kernel is fully functional without samples, and the alternative is a machine
/// with no network being unable to build. It says which files it did not get,
/// and (loudly) when it got none at all, because "the samples are missing" and
/// "the flag did nothing" are different facts.
fn ensure_sample_files() {
    if !samples_requested() {
        return;
    }
    let missing = SAMPLE_FILES.len() - samples_present().len();
    if missing > 0 {
        eprintln!("samples: CHITTI_SAMPLE_FILES set — fetching {missing} missing file(s) into assets/samples/");
        if let Err(e) = cmd_sample_files(&[]) {
            eprintln!("samples: {e}");
        }
    }
    let have = samples_present();
    if have.is_empty() {
        eprintln!(
            "samples: WARNING — CHITTI_SAMPLE_FILES is set but assets/samples/ is empty; \
             building WITHOUT /samples (run `cargo xtask sample-files` with network access)"
        );
    } else {
        let bytes: u64 = have.iter().map(|(_, _, n)| *n).sum();
        eprintln!(
            "samples: embedding {} file(s), {} KiB → /samples/ in the booted OS",
            have.len(),
            bytes / 1024
        );
    }
}

/// `cargo xtask sample-files [--refresh]`: download the `/samples/` corpus into
/// `assets/samples/{images,videos,audios,misc}/` (cached — skips files already
/// present; `--refresh` re-fetches). `make run` / `make vbox` call this for you
/// via `CHITTI_SAMPLE_FILES=1`.
///
/// The kernel embeds whatever is in that directory (see `kernel/build.rs`) and
/// seeds it into the Synapse store at boot, so `/open /samples/images/fruits.jpg`
/// works on a freshly booted machine with no network and no disk.
fn cmd_sample_files(rest: &[String]) -> Result<(), String> {
    let refresh = rest.iter().any(|a| a == "--refresh" || a == "-f");
    let dir = samples_dir();
    let mut failed: Vec<&str> = Vec::new();
    let mut fetched = 0usize;
    for s in SAMPLE_FILES {
        let sub = dir.join(s.category);
        fs::create_dir_all(&sub).map_err(|e| format!("mkdir {}: {e}", sub.display()))?;
        let dst = sub.join(s.name);
        if dst.exists() && !refresh {
            continue;
        }
        if s.is_local() {
            // Authored samples (e.g. /js demos) live in assets/samples-src/ and
            // are copied, not downloaded — no network, always available.
            let src = s.local_src();
            match fs::copy(&src, &dst) {
                Ok(n) if n > 0 => {
                    eprintln!("sample-files: copied {}/{} ({})", s.category, s.name, s.note);
                    fetched += 1;
                }
                _ => {
                    let _ = fs::remove_file(&dst);
                    eprintln!(
                        "sample-files: FAILED {}/{} <- missing local {}",
                        s.category,
                        s.name,
                        src.display()
                    );
                    failed.push(s.name);
                }
            }
            continue;
        }
        // `-f` so an HTTP error is a failure rather than an error page saved as a
        // JPEG (which would then be embedded and fail to decode in the OS).
        let ok = Command::new("curl")
            .args(["-fsSL", "--max-time", "120", "-o"])
            .arg(&dst)
            .arg(s.url)
            .status()
            .map(|st| st.success())
            .unwrap_or(false);
        if ok && fs::metadata(&dst).map(|m| m.len() > 0).unwrap_or(false) {
            eprintln!("sample-files: fetched {}/{} ({})", s.category, s.name, s.note);
            fetched += 1;
        } else {
            // Remove the empty/partial file curl may have left, or the next build
            // embeds a truncated sample.
            let _ = fs::remove_file(&dst);
            eprintln!("sample-files: FAILED {}/{} <- {}", s.category, s.name, s.url);
            failed.push(s.name);
        }
    }
    let have = samples_present();
    write_samples_readme(&have)?;
    let bytes: u64 = have.iter().map(|(_, _, n)| *n).sum();
    eprintln!(
        "sample-files: {} of {} file(s) present ({} KiB) in {}{}",
        have.len(),
        SAMPLE_FILES.len(),
        bytes / 1024,
        dir.display(),
        if fetched > 0 { format!(" — {fetched} newly fetched") } else { String::new() }
    );
    if !failed.is_empty() {
        eprintln!("sample-files: could not fetch: {}", failed.join(", "));
    }
    if have.is_empty() {
        return Err("sample-files: fetched nothing -- check network access".into());
    }
    Ok(())
}

/// Write `assets/samples/README.md` — the provenance record, which is itself
/// embedded (files at the root of the corpus land at `/samples/README.md`), so
/// the booted OS can say where its own samples came from.
fn write_samples_readme(present: &[(&'static str, &'static str, u64)]) -> Result<(), String> {
    let mut md = String::from(
        "# ChittiOS sample files\n\n\
         Fetched by `cargo xtask sample-files` (automatically by `make run` / `make vbox`,\n\
         which set `CHITTI_SAMPLE_FILES=1`), embedded into the kernel image, and written to\n\
         `/samples/` at boot. **Not committed to the repository** — `assets/samples/` is\n\
         gitignored, so nothing here is redistributed by the source tree.\n\n\
         Open one with `/open <path>`; text files land in the editor, media in a player tab.\n\n",
    );
    let mut cats: Vec<&str> = SAMPLE_FILES.iter().map(|s| s.category).collect();
    cats.dedup();
    for cat in cats {
        md.push_str(&format!("## /samples/{cat}\n\n"));
        for s in SAMPLE_FILES.iter().filter(|s| s.category == cat) {
            let size = present.iter().find(|(c, n, _)| *c == s.category && *n == s.name).map(|(_, _, b)| *b);
            let size = match size {
                Some(b) => format!("{} KiB", b.div_ceil(1024)),
                None => "absent".to_string(),
            };
            let flag = if s.openable { "" } else { " **(not playable yet)**" };
            if s.is_local() {
                md.push_str(&format!(
                    "- `{}` — {size}{flag} — {}\n  (local: assets/samples-src/{}/{})\n",
                    s.name, s.note, s.category, s.name
                ));
            } else {
                md.push_str(&format!("- `{}` — {size}{flag} — {}\n  <{}>\n", s.name, s.note, s.url));
            }
        }
        md.push('\n');
    }
    md.push_str(
        "Fetched media keep the licence of their upstream source, listed above.\n\
         The `/samples/js/` scripts are authored in-tree (ChittiOS) for the `/js` CLI.\n",
    );
    let dir = samples_dir();
    fs::create_dir_all(&dir).map_err(|e| format!("mkdir {}: {e}", dir.display()))?;
    fs::write(dir.join("README.md"), md).map_err(|e| format!("writing samples README: {e}"))
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

/// `cargo xtask iwlwifi-assets`: fetch Intel WiFi firmware into `assets/wifi/iwl/`.
///
/// **Fetched, never committed** — the same rule as the Broadcom assets, and the reason
/// this needed no licensing decision: `assets/wifi/` is gitignored, so the repository
/// carries no redistributable blob and each machine pulls what its own hardware needs.
///
/// Takes the family stem to fetch, or every family's newest known image by default. The
/// API version is a property of the file rather than the chip, so this walks candidates
/// newest-first and stops at the first one the upstream tree actually has — exactly what
/// the in-kernel loader does with the local directory.
/// `cargo xtask javy-plugin` — rebuild `assets/wasm/javy-plugin.wasm`.
///
/// The JS engine agents' `tools.js` is compiled by is our own Javy plugin: QuickJS
/// plus the kernel's gated `chitti.host_*` imports. Three steps, each of which fails
/// in a way that does not name its cause, so they live here rather than in a comment:
///
/// 1. `cargo build --target wasm32-wasip1` — the target must be installed
///    (`rustup target add wasm32-wasip1`), and building QuickJS from C pulls a
///    wasi-sdk into the target dir on first run.
/// 2. `javy init-plugin` — **not optional**: it runs the runtime's own
///    initialization and lowers the component to a core module. It also validates
///    through binaryen, which reads the module's `target_features` custom section to
///    decide which wasm features to allow — so the crate must **not** be stripped,
///    or every bulk-memory instruction fails with `[--enable-bulk-memory]`.
/// 3. The result replaces the checked-in blob. Its namespace (`chitti_js_v1`) and
///    size form the stamp `jsmod::emit` writes into every module built against it,
///    so **existing `tools.wasm` artifacts become stale and must be rebuilt** —
///    which the kernel detects and reports rather than failing inside QuickJS.
///
/// The Javy CLI is fetched to the target dir if absent (the `iwlwifi-assets`
/// pattern: fetched, never committed).
fn cmd_javy_plugin() -> Result<(), String> {
    const JAVY_VERSION: &str = "v9.1.0";
    let root = repo_root();
    let dir = root.join("tools/javy-plugin");
    let target_dir = dir.join("target");
    fs::create_dir_all(&target_dir).map_err(|e| format!("mkdir: {e}"))?;

    // 1. The CLI. `javy-plugin-api`'s version must match this release — checked
    // against `crates/plugin/Cargo.toml` at the tag, because a mismatch surfaces
    // only at `init-plugin`.
    let javy = target_dir.join("javy");
    if !javy.exists() {
        let arch = if cfg!(target_arch = "aarch64") { "arm" } else { "x86_64" };
        let os = if cfg!(target_os = "macos") { "macos" } else { "linux" };
        let url = format!(
            "https://github.com/bytecodealliance/javy/releases/download/{JAVY_VERSION}/javy-{arch}-{os}-{JAVY_VERSION}.gz"
        );
        eprintln!("javy-plugin: fetching the Javy CLI {JAVY_VERSION} ({arch}-{os})…");
        let gz = target_dir.join("javy.gz");
        let st = Command::new("curl")
            .args(["-sSL", "-o"])
            .arg(&gz)
            .arg(&url)
            .status()
            .map_err(|e| format!("curl: {e}"))?;
        if !st.success() {
            return Err(format!("could not download {url}"));
        }
        let st = Command::new("gunzip").arg("-f").arg(&gz).status().map_err(|e| format!("gunzip: {e}"))?;
        if !st.success() {
            return Err(String::from("could not gunzip the Javy CLI"));
        }
        let _ = Command::new("chmod").arg("+x").arg(&javy).status();
    }

    // 2. Build the plugin for wasip1.
    eprintln!("javy-plugin: building tools/javy-plugin for wasm32-wasip1…");
    let st = Command::new("cargo")
        .current_dir(&dir)
        .args(["build", "--release", "--target", "wasm32-wasip1"])
        .status()
        .map_err(|e| format!("cargo: {e}"))?;
    if !st.success() {
        return Err(String::from(
            "plugin build failed (is wasm32-wasip1 installed? `rustup target add wasm32-wasip1`)",
        ));
    }

    // 3. Initialize it and replace the blob.
    let built = dir.join("target/wasm32-wasip1/release/chitti_javy_plugin.wasm");
    let out = root.join("assets/wasm/javy-plugin.wasm");
    eprintln!("javy-plugin: javy init-plugin → {}", out.display());
    let st = Command::new(&javy)
        .arg("init-plugin")
        .arg(&built)
        .arg("-o")
        .arg(&out)
        .status()
        .map_err(|e| format!("javy init-plugin: {e}"))?;
    if !st.success() {
        return Err(String::from(
            "javy init-plugin failed (if it complains about bulk memory, the crate was stripped: \
             binaryen reads the `target_features` custom section)",
        ));
    }
    let size = fs::metadata(&out).map(|m| m.len()).unwrap_or(0);
    eprintln!("javy-plugin: wrote {} ({} KiB)", out.display(), size / 1024);
    eprintln!("javy-plugin: NOTE every existing tools.wasm built by /agents build is now stale");
    eprintln!("             (the kernel detects the stamp mismatch and asks for a rebuild)");
    Ok(())
}

fn cmd_iwlwifi_assets() -> Result<(), String> {
    let out = repo_root().join("assets/wifi/iwl");
    fs::create_dir_all(&out).map_err(|e| format!("mkdir {}: {e}", out.display()))?;

    // The kernel's own table, kept in one place: a stem plus the API range to try.
    // Duplicated deliberately as data rather than shared code — xtask is a host binary
    // and does not link the kernel — and `iwlwifi_stems_match_the_kernel` in
    // `drivers::wifi::iwl::fw` is what stops the two drifting.
    let families: &[(&str, u32, u32)] = &[
        ("iwlwifi-cc-a0", 50, 89),        // AX200/AX201
        ("iwlwifi-ty-a0-gf-a0", 50, 89),  // AX210/AX211
        ("iwlwifi-gl-c0-fm-c0", 50, 89),  // BE200
        ("iwlwifi-9260", 30, 46),
        ("iwlwifi-8265", 22, 36),
        ("iwlwifi-7260", 12, 17),
    ];
    const BASE: &str = "https://gitlab.com/kernel-firmware/linux-firmware/-/raw/main";

    let mut got = 0usize;
    for (stem, min_api, max_api) in families {
        if (*min_api..=*max_api).rev().any(|api| {
            let name = format!("{stem}-{api}.ucode");
            let dst = out.join(&name);
            if dst.exists() {
                eprintln!("iwlwifi-assets: {name} already present");
                return true;
            }
            let url = format!("{BASE}/{name}");
            match Command::new("curl")
                .args(["-fsSL", "-o"])
                .arg(&dst)
                .arg(&url)
                .status()
            {
                Ok(st) if st.success() => {
                    eprintln!("iwlwifi-assets: fetched {name}");
                    true
                }
                _ => {
                    // A missing API version is the normal case, not an error: only a few
                    // of the range exist upstream. Remove the empty file curl may leave.
                    let _ = fs::remove_file(&dst);
                    false
                }
            }
        }) {
            got += 1;
        } else {
            eprintln!("iwlwifi-assets: no image found upstream for {stem}");
        }
    }
    if got == 0 {
        return Err("iwlwifi-assets: fetched nothing -- check network access".into());
    }
    eprintln!("iwlwifi-assets: {got} image(s) in {}", out.display());
    eprintln!("iwlwifi-assets: note -- identification only so far; there is no driver to load them yet");
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
    let mut cmd = qemu_base_cmd(&iso, "2G");
    // The in-kernel test suite includes the Phase 7 SMP bring-up + spinlock
    // self-test, so the harness runs with four vCPUs.
    cmd.args(["-smp", "4", "-serial", "stdio", "-display", "none"]);
    // NB: do not add `-d int` here. QEMU dumps the full CPU register state on
    // every interrupt, and with a 1000 Hz APIC tick across these 4 vCPUs that is
    // millions of formatted entries per run -- left on by accident it wrote an
    // 818 MB, 14.2-million-line trace for a single `cargo xtask test`, all of it
    // formatted and written synchronously while the guest waits. Enable it in a
    // local shell for one debugging session, never in the committed runner.
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
