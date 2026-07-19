//! Broadcom **FullMAC PCIe** (brcmfmac-class) bring-up for Apple Silicon.
//!
//! Chip on j473 (Mac mini M2): `pci14e4,4434` / BCM4387, board-type
//! `apple,miyake`. The host interface is the same FullMAC shared-RAM + common
//! rings model Linux `brcmfmac` uses; this module ports the **probe + chip-id +
//! dongle firmware download** path. Scan/connect need the ioctl ring path
//! (next milestone) once `firmware_up` is set.
//!
//! Download flow (Linux `brcmf_pcie_download_fw_nvram`):
//! 1. EROM-scan for the ARM CR4/CA7 wrapbase → halt CPU
//! 2. `memcpy` firmware into BAR2 TCM at `rambase`
//! 3. Optional NVRAM at top of RAM; clear the shared-RAM handshake word
//! 4. Write reset vector to TCM[0], release ARM halt
//! 5. Poll `rambase+ramsize-4` for the firmware-posted shared-RAM address
//!
//! References: m1n1 `proxyclient/hv/trace_wlan.py`, Asahi DT
//! (`brcm,board-type`, `apple,antenna-sku`, `brcm,cal-blob`), Linux
//! `drivers/net/wireless/broadcom/brcm80211/brcmfmac/{pcie,chip}.c`.

use super::proto;
use crate::pci::{self, PciDevice};
use alloc::string::String;
use alloc::vec::Vec;
use core::ptr::{read_volatile, write_volatile};

// assets/ is five levels up from drivers/wifi/brcm/

/// Broadcom PCI vendor.
const VENDOR_BRCM: u16 = 0x14e4;
/// BCM4387 WiFi function on Apple modules (j473 and kin).
const DEV_BCM4387: u16 = 0x4434;
/// Alternate IDs seen on other Apple boards (probe accepts these too).
const DEV_BCM4378: u16 = 0x4425;
const DEV_BCM4377: u16 = 0x4488;

// PCI config offsets (trace_wlan.WLANCfgSpace)
const CFG_BAR0_WINDOW: u16 = 0x80;
const CFG_BAR0_WIN_1000: u16 = 0x70;
const CFG_BAR0_WIN_4000: u16 = 0x74;
const CFG_BAR0_WIN_5000: u16 = 0x78;
const CFG_INTSTATUS: u16 = 0x90;
const CFG_INTMASK: u16 = 0x94;
const CFG_CLK_CTL_ST: u16 = 0xa8;
const CFG_LINK_STATUS_CTRL: u16 = 0xbc;

// BAR0 window layout
const BAR0_WINDOW_OFF: usize = 0x0000;
const BAR0_PCIE2_OFF: usize = 0x2000;
const BAR0_CC_OFF: usize = 0x3000;
/// BAR0 backplane window size (Linux `BRCMF_PCIE_BAR0_REG_SIZE`).
const BAR0_REG_SIZE: u32 = 0x1000;

// Chipcommon inside the fixed 0x3000 window (when BAR0_WINDOW points at CC).
// Chip ID is at CC + 0x00 on these cores when the window is set correctly.
const CC_CHIPID: usize = 0x00;

// AI agent (wrapper) registers — Linux `bcma_regs.h`.
const BCMA_IOCTL: u32 = 0x0408;
const BCMA_IOCTL_CLK: u32 = 0x0001;
const BCMA_IOCTL_FGC: u32 = 0x0002;
const BCMA_RESET_CTL: u32 = 0x0800;
const BCMA_RESET_CTL_RESET: u32 = 0x0001;
const ARMCR4_BCMA_IOCTL_CPUHALT: u32 = 0x0020;

/// How long to wait for the dongle to post the shared-RAM address (ms).
const FW_UP_TIMEOUT_MS: u64 = 5000;
/// Poll period while waiting for firmware init (ms).
const FW_UP_POLL_MS: u64 = 50;

/// Driver state after a successful probe.
pub struct BrcmDevice {
    pub pci: PciDevice,
    pub bar0: u64,
    pub bar2: u64,
    /// PCI-reported BAR2 size (TCM aperture). Caps dongle ramsize.
    pub bar2_size: u64,
    pub mac: [u8; 6],
    pub board_type: String,
    pub antenna_sku: String,
    pub chip_id: u16,
    pub chip_rev: u16,
    pub firmware_stem: String,
    /// True once firmware has been loaded and the shared-RAM handshake
    /// completed (rings/ioctl still pending).
    pub firmware_up: bool,
    /// Dongle RAM window used for the download (TCM offset into BAR2).
    pub rambase: u32,
    pub ramsize: u32,
    /// Shared-RAM address the firmware posted (TCM-relative).
    pub shared_addr: u32,
    /// Shared protocol version from the shared-info header.
    pub shared_version: u8,
    /// ARM core wrapbase from EROM (0 if unknown — download may still work
    /// on a cold-boot halt).
    pub arm_wrap: u32,
    pub arm_core_id: u16,
    /// Cached scan results (filled after a real scan; empty until then).
    pub scan_cache: Vec<proto::BssInfo>,
}

static DEV: crate::mm::Locked<Option<BrcmDevice>> = crate::mm::Locked::new(None);

/// Shared access to the probed device, if any.
pub fn with_dev<R>(f: impl FnOnce(&mut BrcmDevice) -> R) -> Option<R> {
    DEV.with(|d| d.as_mut().map(f))
}

pub fn is_present() -> bool {
    DEV.with(|d| d.is_some())
}

#[inline]
fn mmio_r32(base: u64, off: usize) -> u32 {
    // SAFETY: BAR-mapped Device memory of the WiFi function.
    unsafe { read_volatile((base + off as u64) as *const u32) }
}
#[inline]
fn mmio_w32(base: u64, off: usize, v: u32) {
    // SAFETY: BAR-mapped Device memory of the WiFi function.
    unsafe { write_volatile((base + off as u64) as *mut u32, v) }
}

/// Pull Apple-specific props from the FDT WiFi node (`pci14e4,4434`).
fn fdt_wifi_props() -> (Option<[u8; 6]>, String, String, Vec<u8>) {
    let fdt = crate::arch::aarch64::boot::boot_x0();
    let mut mac = None;
    let mut board = String::new();
    let mut antenna = String::new();
    let mut cal = Vec::new();
    // SAFETY: FDT from boot.
    let _ = unsafe {
        crate::fdt::for_each_prop_of_compatible(fdt, b"pci14e4,4434", &mut |name, val| {
            if name == b"local-mac-address" && val.len() >= 6 {
                let mut m = [0u8; 6];
                m.copy_from_slice(&val[..6]);
                mac = Some(m);
            } else if name == b"brcm,board-type" {
                board = core::str::from_utf8(val)
                    .unwrap_or("")
                    .trim_end_matches('\0')
                    .into();
            } else if name == b"apple,antenna-sku" {
                antenna = core::str::from_utf8(val)
                    .unwrap_or("")
                    .trim_end_matches('\0')
                    .into();
            } else if name == b"brcm,cal-blob" && !val.is_empty() {
                cal.extend_from_slice(val);
            }
        })
    };
    // Fallback: any wifi-compatible if the PCI string differs.
    if mac.is_none() && board.is_empty() {
        let _ = unsafe {
            crate::fdt::for_each_prop_of_compatible(fdt, b"pci14e4,4425", &mut |name, val| {
                if name == b"local-mac-address" && val.len() >= 6 {
                    let mut m = [0u8; 6];
                    m.copy_from_slice(&val[..6]);
                    mac = Some(m);
                } else if name == b"brcm,board-type" {
                    board = core::str::from_utf8(val)
                        .unwrap_or("")
                        .trim_end_matches('\0')
                        .into();
                }
            })
        };
    }
    (mac, board, antenna, cal)
}

/// Find the Broadcom WiFi PCI function (fn0 only — fn1 is Bluetooth).
///
/// WiFi lives on **bus 1** (port0). Touching that ECAM window while the port
/// link is down is a **fatal external abort** on Apple Silicon — we refuse to
/// scan until [`apple_pcie::port0_link_up`] is true, and use recoverable
/// reads as a belt-and-braces.
fn find_wifi_pci() -> Option<PciDevice> {
    if pci::ecam_base() == 0 {
        return None;
    }
    if !crate::arch::aarch64::apple_pcie::port0_link_up() {
        crate::ktrace::log(
            "wifi",
            "port0 link down — not scanning bus 1 (would abort). Power the module (SMC gP0d) first.",
        );
        return None;
    }
    // DT places WiFi at bus 1 / dev 0 / fn 0. Also walk the (now-safe) bus_end
    // range in case a board differs.
    let bus_end = crate::arch::aarch64::apple_pcie::report().bus_end;
    for bus in 0u8..=bus_end {
        for dev in 0u8..32 {
            // Prefer recoverable ECAM for secondary buses.
            let id = if bus == 0 {
                pci::read32(bus, dev, 0, 0x00)
            } else {
                match crate::arch::aarch64::apple_pcie::ecam_read32(bus, dev, 0, 0x00) {
                    Some(v) => v,
                    None => continue, // empty / abort → skip
                }
            };
            let v = (id & 0xffff) as u16;
            if v == 0xffff {
                continue;
            }
            let d = (id >> 16) as u16;
            if v == VENDOR_BRCM && matches!(d, DEV_BCM4387 | DEV_BCM4378 | DEV_BCM4377) {
                return Some(PciDevice {
                    bus,
                    dev,
                    func: 0,
                    vendor: v,
                    device: d,
                });
            }
        }
    }
    None
}

/// Program BAR0_WINDOW and read a 32-bit word from the chipcommon-ish window.
fn read_chipcommon_id(bar0: u64, pci: &PciDevice) -> (u16, u16) {
    // Point the main BAR0 window at the chipcommon core. On Apple FullMAC the
    // fixed CC window at +0x3000 is often already usable; try that first.
    let raw = mmio_r32(bar0, BAR0_CC_OFF + CC_CHIPID);
    let mut chip = (raw & 0xffff) as u16;
    let mut rev = ((raw >> 16) & 0xffff) as u16;
    if chip == 0 || chip == 0xffff {
        // Program BAR0_WINDOW to a typical CC backplane base for 43xx (0x18000000).
        pci::write32(pci.bus, pci.dev, pci.func, CFG_BAR0_WINDOW, 0x1800_0000);
        let raw2 = mmio_r32(bar0, BAR0_WINDOW_OFF + CC_CHIPID);
        chip = (raw2 & 0xffff) as u16;
        rev = ((raw2 >> 16) & 0xffff) as u16;
    }
    (chip, rev)
}

fn stash_fdt_only(reason: &str) {
    let _ = ensure_stub(reason);
}

/// Ensure a device record exists for `/wifi info` even when BAR MMIO failed.
/// Keeps a live probe if BARs are already mapped. Returns true if a stub/live
/// record is present afterward.
pub fn ensure_stub(reason: &str) -> bool {
    if DEV.with(|d| d.as_ref().map(|x| x.bar0 != 0).unwrap_or(false)) {
        return true;
    }
    let (mac_opt, board, antenna, cal) = fdt_wifi_props();
    let mac = mac_opt.unwrap_or([0x00, 0x10, 0x18, 0x00, 0x00, 0x10]);
    let board = if board.is_empty() {
        String::from("apple,miyake")
    } else {
        board
    };
    // Prefer the real PCI BDF if the link is up and the function is visible.
    let pci = find_wifi_pci().unwrap_or(PciDevice {
        bus: 1,
        dev: 0,
        func: 0,
        vendor: VENDOR_BRCM,
        device: DEV_BCM4387,
    });
    crate::ktrace::log_fmt(format_args!(
        "wifi: {reason} — FDT mac={:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x} board={} antenna={} cal={}B pci={:02x}:{:02x}.{}",
        mac[0], mac[1], mac[2], mac[3], mac[4], mac[5],
        board, antenna, cal.len(),
        pci.bus, pci.dev, pci.func
    ));
    DEV.with(|d| {
        *d = Some(BrcmDevice {
            pci,
            bar0: 0,
            bar2: 0,
            bar2_size: 0,
            mac,
            board_type: board.clone(),
            antenna_sku: antenna,
            chip_id: 0,
            chip_rev: 0,
            firmware_stem: proto::firmware_stem(&board, 0x4387),
            firmware_up: false,
            rambase: 0,
            ramsize: 0,
            shared_addr: 0,
            shared_version: 0,
            arm_wrap: 0,
            arm_core_id: 0,
            scan_cache: Vec::new(),
        });
    });
    true
}

/// Probe the WiFi function. Powers the module via SMC if needed, waits for
/// link, then enumerates BAR0/chip id. Returns true on full PCI probe success.
pub fn probe() -> bool {
    // If we already have live BARs, done.
    if DEV.with(|d| d.as_ref().map(|x| x.bar0 != 0).unwrap_or(false)) {
        return true;
    }
    if !crate::arch::aarch64::apple_pcie::ready() {
        crate::ktrace::log("wifi", "Apple PCIe not ready");
        return false;
    }
    // Power + wait for link (no-op if already up).
    if !crate::arch::aarch64::apple_pcie::port0_link_up() {
        crate::ktrace::log("wifi", "link down — asserting SMC power + waiting");
        let _ = crate::arch::aarch64::apple_pcie::retry_wifi_power();
    }
    if !crate::arch::aarch64::apple_pcie::port0_link_up() {
        stash_fdt_only("link still DOWN after SMC power");
        return false;
    }
    // Clear any prior FDT-only stub so we can re-fill with real BARs.
    DEV.with(|d| *d = None);

    let Some(pci_dev) = find_wifi_pci() else {
        crate::ktrace::log("wifi", "no Broadcom FullMAC PCI function (14e4:4434/…) found");
        stash_fdt_only("PCI scan found nothing on bus 1");
        return false;
    };
    crate::ktrace::log_fmt(format_args!(
        "wifi: found {:04x}:{:04x} at {:02x}:{:02x}.{}",
        pci_dev.vendor, pci_dev.device, pci_dev.bus, pci_dev.dev, pci_dev.func
    ));

    // Bring the function to D0 before touching BARs (config works in D3; MEM does not).
    pci_dev.set_power_d0();
    pci_dev.enable_bus_master();

    // Size BARs, then probe which PCI/CPU window the host fabric actually
    // decodes (m1n1 axi2af may not match the Linux DTS ranges).
    let Some((bar0_size, bar0_type, b0_64)) = pci_dev.size_bar(0) else {
        crate::ktrace::log("wifi", "BAR0 not a memory BAR");
        return false;
    };
    let (bar2_size, bar2_type, _) = pci_dev.size_bar(2).unwrap_or((0, 0, false));
    crate::ktrace::log_fmt(format_args!(
        "wifi: BAR0 size={bar0_size:#x} type={bar0_type:#x} 64={b0_64}  BAR2 size={bar2_size:#x}"
    ));

    let Some((bar0, bar2, pci_base)) = crate::arch::aarch64::apple_pcie::find_working_bar_window(
        &pci_dev,
        bar0_size,
        bar0_type,
        bar2_size,
        bar2_type,
    ) else {
        crate::ktrace::log("wifi", "no working BAR outbound window (all candidates aborted)");
        return false;
    };
    crate::ktrace::log_fmt(format_args!(
        "wifi: BAR window pci_base={pci_base:#x} BAR0_cpu={bar0:#x} BAR2_cpu={bar2:#x}"
    ));

    let cmd = pci::read32(pci_dev.bus, pci_dev.dev, pci_dev.func, 0x04);
    crate::ktrace::log_fmt(format_args!(
        "wifi: ep cmd={cmd:#010x} BAR0={:#x} BAR2={:#x}",
        pci_dev.bar(0),
        pci_dev.bar(2)
    ));

    let (chip_id, chip_rev) = read_chipcommon_id(bar0, &pci_dev);
    crate::ktrace::log_fmt(format_args!("wifi: chip_id={chip_id:#x} rev={chip_rev}"));

    let (mac_opt, board, antenna, cal) = fdt_wifi_props();
    let mac = mac_opt.unwrap_or([0x00, 0x10, 0x18, 0x00, 0x00, 0x10]);
    let board = if board.is_empty() {
        String::from("apple,miyake")
    } else {
        board
    };
    let stem = proto::firmware_stem(&board, if chip_id != 0 && chip_id != 0xffff {
        chip_id
    } else {
        0x4387
    });

    if !cal.is_empty() {
        crate::ktrace::log_fmt(format_args!("wifi: cal-blob {} bytes (from FDT/ADT)", cal.len()));
    }

    // Clear host-side interrupt mask for now (poll path later).
    pci::write32(pci_dev.bus, pci_dev.dev, pci_dev.func, CFG_INTMASK, 0);

    let _ = (
        CFG_BAR0_WIN_1000,
        CFG_BAR0_WIN_4000,
        CFG_BAR0_WIN_5000,
        CFG_INTSTATUS,
        CFG_CLK_CTL_ST,
        BAR0_PCIE2_OFF,
        CFG_LINK_STATUS_CTRL,
    );

    let rambase = proto::rambase_for_chip(if chip_id != 0 && chip_id != 0xffff {
        chip_id
    } else {
        0x4388
    })
    .unwrap_or(0x74_0000);

    DEV.with(|d| {
        *d = Some(BrcmDevice {
            pci: pci_dev,
            bar0,
            bar2,
            bar2_size,
            mac,
            board_type: board,
            antenna_sku: antenna,
            chip_id,
            chip_rev,
            firmware_stem: stem,
            firmware_up: false,
            rambase,
            ramsize: 0,
            shared_addr: 0,
            shared_version: 0,
            arm_wrap: 0,
            arm_core_id: 0,
            scan_cache: Vec::new(),
        });
    });
    true
}

// ── Backplane access via BAR0_WINDOW ──────────────────────────────────

fn mdelay(ms: u64) {
    let end = crate::arch::now_ms() + ms;
    while crate::arch::now_ms() < end {
        // status_tick keeps the clock/mouse alive; do not poll Ctrl+C here —
        // short reset delays must not abort mid-sequence. Long waits (TCM
        // copy, FW handshake) call poll_interrupt themselves.
        let _ = crate::shell::status_tick();
        core::hint::spin_loop();
    }
}

/// Program BAR0_WINDOW and return the offset within the 4 KiB window.
fn bar0_window_prep(pci: &PciDevice, addr: u32) -> u32 {
    let off = addr & (BAR0_REG_SIZE - 1);
    let win = addr & !(BAR0_REG_SIZE - 1);
    pci::write32(pci.bus, pci.dev, pci.func, CFG_BAR0_WINDOW, win);
    // Read-back confirms the window latched (some cores need a second write).
    let got = pci::read32(pci.bus, pci.dev, pci.func, CFG_BAR0_WINDOW);
    if got != win {
        pci::write32(pci.bus, pci.dev, pci.func, CFG_BAR0_WINDOW, win);
    }
    off
}

fn bp_read32(bar0: u64, pci: &PciDevice, addr: u32) -> u32 {
    let off = bar0_window_prep(pci, addr);
    mmio_r32(bar0, BAR0_WINDOW_OFF + off as usize)
}

/// Recoverable backplane read — returns `None` on external abort instead of
/// panicking the kernel (Apple APCIE BAR MEM is unforgiving for some cores).
fn bp_read32_probe(bar0: u64, pci: &PciDevice, addr: u32) -> Option<u32> {
    let off = bar0_window_prep(pci, addr);
    crate::arch::aarch64::probe_read32(bar0 + BAR0_WINDOW_OFF as u64 + off as u64)
}

fn bp_write32(bar0: u64, pci: &PciDevice, addr: u32, val: u32) {
    let off = bar0_window_prep(pci, addr);
    mmio_w32(bar0, BAR0_WINDOW_OFF + off as usize, val);
}

fn tcm_read32(tcm: u64, off: u32) -> u32 {
    mmio_r32(tcm, off as usize)
}

fn tcm_write32(tcm: u64, off: u32, val: u32) {
    mmio_w32(tcm, off as usize, val)
}

/// Recoverable BAR2 TCM store — `false` on external abort (does not FATAL).
fn tcm_bar2_probe_write32(tcm: u64, off: u32, val: u32) -> bool {
    crate::arch::aarch64::probe_write32(tcm + off as u64, val)
}

/// Recoverable BAR2 TCM load — `None` on external abort.
fn tcm_bar2_probe_read32(tcm: u64, off: u32) -> Option<u32> {
    crate::arch::aarch64::probe_read32(tcm + off as u64)
}

// ── TCM via BAR0 window (preferred on Apple) ─────────────────────────
//
// On j473, BAR2 (PCI BAR2) accepts stores without abort but **every BAR2 load
// external-aborts**, so we can never verify a download or poll the shared-RAM
// handshake through BAR2. Chipcommon already works through BAR0_WINDOW, so we
// reach dongle RAM the same way: point the 4 KiB BAR0 window at the target
// backplane address and access the offset inside BAR0. Slow but correct.

/// Write 32-bit TCM/backplane via BAR0_WINDOW. Recoverable.
fn tcm_bp_write32(bar0: u64, pci: &PciDevice, addr: u32, val: u32) -> bool {
    let off = bar0_window_prep(pci, addr);
    crate::arch::aarch64::probe_write32(bar0 + BAR0_WINDOW_OFF as u64 + off as u64, val)
}

/// Read 32-bit TCM/backplane via BAR0_WINDOW. Recoverable.
fn tcm_bp_read32(bar0: u64, pci: &PciDevice, addr: u32) -> Option<u32> {
    let off = bar0_window_prep(pci, addr);
    crate::arch::aarch64::probe_read32(bar0 + BAR0_WINDOW_OFF as u64 + off as u64)
}

/// Write+readback probe at `addr` through BAR0 window. Returns true only if
/// the value sticks (proves real RAM, not a posted write into the void).
fn tcm_bp_poke(bar0: u64, pci: &PciDevice, addr: u32) -> bool {
    let magic = 0x4b47_5432u32 ^ addr; // 'KGT2' ^ addr
    if !tcm_bp_write32(bar0, pci, addr, magic) {
        return false;
    }
    // SAFETY: order the store before the load on the same Device location.
    unsafe {
        core::arch::asm!("dsb sy", options(nostack, preserves_flags));
    }
    match tcm_bp_read32(bar0, pci, addr) {
        Some(v) if v == magic => {
            let _ = tcm_bp_write32(bar0, pci, addr, 0);
            true
        }
        other => {
            crate::ktrace::log_fmt(format_args!(
                "wifi: TCM poke @{addr:#x} write_ok read={other:?} (want {magic:#x})"
            ));
            false
        }
    }
}

/// Find a working firmware load base: try known rambase values with poke test.
fn calibrate_rambase(bar0: u64, pci: &PciDevice, preferred: u32) -> Option<u32> {
    let extras = [0x74_0000u32, 0x18_0000, 0x17_0000, 0x35_2000, 0];
    let mut tried = [false; 8];
    let mut try_base = |base: u32, slot: &mut [bool]| -> Option<u32> {
        let idx = match base {
            0 => 0,
            0x74_0000 => 1,
            0x18_0000 => 2,
            0x17_0000 => 3,
            0x35_2000 => 4,
            _ => 5,
        };
        if slot[idx] {
            return None;
        }
        slot[idx] = true;
        if tcm_bp_poke(bar0, pci, base) && tcm_bp_poke(bar0, pci, base.saturating_add(0x1000)) {
            crate::ktrace::log_fmt(format_args!(
                "wifi: TCM calibrate OK at rambase={base:#x} (BAR0 window path)"
            ));
            Some(base)
        } else {
            crate::ktrace::log_fmt(format_args!("wifi: TCM calibrate miss at {base:#x}"));
            None
        }
    };
    if let Some(b) = try_base(preferred, &mut tried) {
        return Some(b);
    }
    for &b in &extras {
        if b == preferred {
            continue;
        }
        if let Some(x) = try_base(b, &mut tried) {
            return Some(x);
        }
    }
    None
}

/// Copy firmware through BAR2 with recoverable stores and per-page readback.
fn tcm_copy_bar2_verified(tcm: u64, rambase: u32, data: &[u8]) -> Result<u32, &'static str> {
    if tcm == 0 {
        return Err("BAR2 not mapped");
    }
    let mut i = 0usize;
    while i + 4 <= data.len() {
        let off = rambase + i as u32;
        if off & 3 != 0 {
            return Err("TCM offset not word-aligned");
        }
        let v = u32::from_le_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]);
        if !tcm_bar2_probe_write32(tcm, off, v) {
            crate::ktrace::log_fmt(format_args!(
                "wifi: BAR2 TCM write abort @{off:#x} after {i} bytes"
            ));
            return Ok(i as u32);
        }
        if i == 0 || (off & 0xfff) == 0 {
            unsafe {
                core::arch::asm!("dsb sy", options(nostack, preserves_flags));
            }
            match tcm_bar2_probe_read32(tcm, off) {
                Some(rb) if rb == v => {}
                Some(rb) => {
                    crate::ktrace::log_fmt(format_args!(
                        "wifi: BAR2 verify fail @{off:#x}: wrote {v:#x} read {rb:#x}"
                    ));
                    return Ok(i as u32);
                }
                None => {
                    crate::ktrace::log_fmt(format_args!(
                        "wifi: BAR2 verify abort @{off:#x} after write"
                    ));
                    return Ok(i as u32);
                }
            }
        }
        i += 4;
        if i & 0xffff == 0 {
            let _ = crate::shell::upkeep();
            if crate::shell::poll_interrupt() {
                return Err("cancelled");
            }
        }
    }
    // Trailing bytes
    while i < data.len() {
        let off = rambase + i as u32;
        let aligned = off & !3;
        let shift = (off & 3) * 8;
        let cur = tcm_bar2_probe_read32(tcm, aligned).unwrap_or(0);
        let nv = (cur & !(0xffu32 << shift)) | ((data[i] as u32) << shift);
        if !tcm_bar2_probe_write32(tcm, aligned, nv) {
            return Ok(i as u32);
        }
        i += 1;
    }
    Ok(i as u32)
}

/// Copy firmware through BAR0 window sliding (4 KiB at a time). Verifies the
/// first and last word of each page. Returns bytes written.
fn tcm_bp_copy(
    bar0: u64,
    pci: &PciDevice,
    rambase: u32,
    data: &[u8],
) -> Result<u32, &'static str> {
    let mut i = 0usize;
    while i + 4 <= data.len() {
        let addr = rambase + i as u32;
        // Keep word alignment on the backplane address.
        if addr & 3 != 0 {
            return Err("TCM copy address not word-aligned");
        }
        let v = u32::from_le_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]);
        if !tcm_bp_write32(bar0, pci, addr, v) {
            crate::ktrace::log_fmt(format_args!(
                "wifi: BAR0-window TCM abort at {addr:#x} after {i} bytes"
            ));
            return Ok(i as u32);
        }
        // Verify first word of each 4 KiB page + the very first word.
        if i == 0 || (addr & 0xfff) == 0 {
            unsafe {
                core::arch::asm!("dsb sy", options(nostack, preserves_flags));
            }
            match tcm_bp_read32(bar0, pci, addr) {
                Some(rb) if rb == v => {}
                Some(rb) => {
                    crate::ktrace::log_fmt(format_args!(
                        "wifi: TCM verify fail @{addr:#x}: wrote {v:#x} read {rb:#x}"
                    ));
                    return Ok(i as u32);
                }
                None => {
                    crate::ktrace::log_fmt(format_args!(
                        "wifi: TCM verify abort @{addr:#x} after write"
                    ));
                    return Ok(i as u32);
                }
            }
        }
        i += 4;
        if i & 0xffff == 0 {
            let _ = crate::shell::upkeep();
            if crate::shell::poll_interrupt() {
                return Err("cancelled");
            }
        }
    }
    // Trailing 1–3 bytes (rare for FW images).
    if i < data.len() {
        let addr = rambase + i as u32;
        let aligned = addr & !3;
        let mut word = tcm_bp_read32(bar0, pci, aligned).unwrap_or(0);
        for (bi, byte) in data[i..].iter().enumerate() {
            let sh = ((addr & 3) as u32 + bi as u32) * 8;
            word = (word & !(0xff << sh)) | ((*byte as u32) << sh);
        }
        if !tcm_bp_write32(bar0, pci, aligned, word) {
            return Ok(i as u32);
        }
        i = data.len();
    }
    Ok(i as u32)
}

/// Pad NVRAM to a 4-byte multiple (Linux/`memcpy_toio` expects aligned DMA
/// windows; misaligned end addresses faulted at `…bffc` on device).
fn pad_nvram_4(nv: &[u8]) -> Vec<u8> {
    let mut v = nv.to_vec();
    while v.len() % 4 != 0 {
        v.push(0);
    }
    v
}

/// Floor-aligned firmware span: the largest 4-byte multiple ≤ `fw_len`.
/// The shared-RAM handshake seed is the **last word inside this span** — on
/// j473 any store past the firmware image external-aborts
/// (`FAR=BAR2+0x961b90` with `fw+4`; earlier NVRAM at `0x967508` / `0x9bfffc`).
fn fw_span(fw_len: u32) -> u32 {
    (fw_len & !3).max(4)
}

/// TCM offset of the handshake word (last full word of the loaded image).
fn handshake_off(rambase: u32, fw_len: u32) -> u32 {
    rambase + fw_span(fw_len) - 4
}

/// Reported dongle RAM size (for logs / later NVRAM). Host **writes** for the
/// handshake always stay inside the firmware image (see [`handshake_off`]).
///
/// Priority for the *reported* size: valid SYS_MEM → SMAR → in-image span.
fn choose_ramsize(
    chip: u16,
    fw: &[u8],
    sysmem_size: u32,
    bar2_size: u64,
    rambase: u32,
) -> u32 {
    let fw_len = fw.len() as u32;
    let in_image = fw_span(fw_len);
    let mut size = in_image;
    let mut src = "in-image";

    // Only trust SYS_MEM if it looks like a real size (not a dead-bus read).
    if sysmem_size > in_image && sysmem_size <= 4 * 1024 * 1024 {
        size = sysmem_size;
        src = "SYS_MEM";
    } else if let Some(smar) = proto::fw_embedded_ramsize(fw) {
        if smar >= in_image && smar <= 0x26_0000 {
            size = smar;
            src = "SMAR";
        }
    }
    let _ = chip;

    if bar2_size != 0 {
        let cap = (bar2_size as u32).saturating_sub(rambase);
        if size > cap {
            crate::ktrace::log_fmt(format_args!(
                "wifi: ramsize {size:#x} > BAR2 window {cap:#x} — capping"
            ));
            size = cap;
        }
    }
    crate::ktrace::log_fmt(format_args!(
        "wifi: ramsize={size:#x} src={src} in_image={in_image:#x} (fw={fw_len:#x} sysmem={sysmem_size:#x})"
    ));
    size.max(in_image)
}

/// AI core: put into reset with `prereset` ioctl bits, then configure `reset`
/// bits while held. Linux `brcmf_chip_ai_coredisable`.
///
/// Reads go through the recoverable probe so a bad wrapbase cannot FATAL. On
/// the Apple backplane a wrap read can transiently abort mid-transition; when
/// that happens we assume the core is **not** already in reset and drive the
/// full assert sequence anyway — the writes are posted and take effect even
/// when a paired read aborts (the old code bailed here and did nothing, which
/// left SYS_MEM stuck in reset with its TCM unreadable).
fn ai_core_disable(bar0: u64, pci: &PciDevice, wrap: u32, prereset: u32, reset: u32) {
    if wrap == 0 {
        return;
    }
    let in_reset = bp_read32_probe(bar0, pci, wrap + BCMA_RESET_CTL)
        .map(|r| r & BCMA_RESET_CTL_RESET != 0)
        .unwrap_or(false);
    if !in_reset {
        // configure reset while clocked
        bp_write32(
            bar0,
            pci,
            wrap + BCMA_IOCTL,
            prereset | BCMA_IOCTL_FGC | BCMA_IOCTL_CLK,
        );
        let _ = bp_read32_probe(bar0, pci, wrap + BCMA_IOCTL);
        // put in reset
        bp_write32(bar0, pci, wrap + BCMA_RESET_CTL, BCMA_RESET_CTL_RESET);
        mdelay(1);
    }
    // in-reset configure
    bp_write32(
        bar0,
        pci,
        wrap + BCMA_IOCTL,
        reset | BCMA_IOCTL_FGC | BCMA_IOCTL_CLK,
    );
    let _ = bp_read32_probe(bar0, pci, wrap + BCMA_IOCTL);
}

/// AI core: take out of reset with `postreset` ioctl bits. Port of Linux
/// `brcmf_chip_ai_resetcore` (Asahi tree): disable first (works for arbitrary
/// current state), then clear `RESET_CTL` in a bounded poll until it reads
/// deasserted, then set `postreset | CLK`. `FGC` (force-gated-clock) is applied
/// in the disable step — that per-core clock force is the *entire* clock setup;
/// there is no chipcommon/PCI-config clock register in this path.
///
/// We always write `RESET_CTL=0` at least once before polling, so a flaky
/// wrap read (which aborts to `None` on the Apple backplane) can never skip the
/// deassert — the bug that left SYS_MEM stuck in reset with its TCM unreadable.
fn ai_core_reset(bar0: u64, pci: &PciDevice, wrap: u32, prereset: u32, reset: u32, postreset: u32) {
    if wrap == 0 {
        return;
    }
    ai_core_disable(bar0, pci, wrap, prereset, reset);
    let mut count = 0u32;
    loop {
        bp_write32(bar0, pci, wrap + BCMA_RESET_CTL, 0);
        let still = bp_read32_probe(bar0, pci, wrap + BCMA_RESET_CTL)
            .map(|r| r & BCMA_RESET_CTL_RESET != 0)
            .unwrap_or(false);
        count += 1;
        if !still || count > 50 {
            break;
        }
        mdelay(1);
    }
    bp_write32(bar0, pci, wrap + BCMA_IOCTL, postreset | BCMA_IOCTL_CLK);
    let _ = bp_read32_probe(bar0, pci, wrap + BCMA_IOCTL);
}

// NB: on these chips the host bring-up does NOT force the clock via any
// clk_ctl_st register (chipcommon 0x1e0 or PCI-config 0xa8) — those are firmware
// state. The backplane ALP clock is already up after link training, and the
// per-core `BCMA_IOCTL_FGC|CLK` in ai_core_disable/reset is the only clock force.

/// Halt the ARM (CR4/CA7) via its wrapper. Linux `brcmf_chip_disable_arm`.
fn arm_halt(bar0: u64, pci: &PciDevice, arm_wrap: u32) {
    if arm_wrap == 0 {
        return;
    }
    // clear all IOCTL bits except HALT, then reset with HALT held
    let val = bp_read32_probe(bar0, pci, arm_wrap + BCMA_IOCTL)
        .unwrap_or(0)
        & ARMCR4_BCMA_IOCTL_CPUHALT;
    ai_core_reset(
        bar0,
        pci,
        arm_wrap,
        val,
        ARMCR4_BCMA_IOCTL_CPUHALT,
        ARMCR4_BCMA_IOCTL_CPUHALT,
    );
}

/// Release ARM halt after firmware is resident. Linux `brcmf_chip_cr4_set_active`
/// / `ca7_set_active`: write rstvec to TCM[0], then resetcore(HALT → 0).
fn arm_run(bar0: u64, pci: &PciDevice, tcm: u64, arm_wrap: u32, rstvec: u32) {
    tcm_write32(tcm, 0, rstvec);
    if arm_wrap != 0 {
        ai_core_reset(
            bar0,
            pci,
            arm_wrap,
            ARMCR4_BCMA_IOCTL_CPUHALT,
            0,
            0,
        );
    }
}

/// Cores we care about from the DMP EROM.
struct ChipCores {
    arm_id: u16,
    arm_base: u32,
    arm_wrap: u32,
    sysmem_base: u32,
    sysmem_wrap: u32,
    sysmem_rev: u8,
}

/// Scan the DMP EROM for ARM + SYS_MEM cores.
///
/// **Do not** size TCM via ARM `BANKIDX`/`BANKINFO` on Apple Silicon: those
/// slave-core MMIO windows external-abort through the APCIE BAR0 aperture
/// (`FAR = BAR0+0x44`). SYS_MEM (0x849) is the Linux path for CA7 chips.
fn discover_cores(bar0: u64, pci: &PciDevice) -> ChipCores {
    let mut out = ChipCores {
        arm_id: 0,
        arm_base: 0,
        arm_wrap: 0,
        sysmem_base: 0,
        sysmem_wrap: 0,
        sysmem_rev: 0,
    };
    let erom_ptr = bp_read32_probe(bar0, pci, proto::SI_ENUM_BASE + proto::CC_EROMPTR)
        .unwrap_or(0);
    if erom_ptr == 0 || erom_ptr == 0xffff_ffff {
        crate::ktrace::log("wifi", "EROM pointer unreadable — ARM wrap unknown");
        return out;
    }
    crate::ktrace::log_fmt(format_args!("wifi: EROM @ {erom_ptr:#x}"));
    let cores = proto::erom_scan(
        |addr| bp_read32_probe(bar0, pci, addr).unwrap_or(0xffff_ffff),
        erom_ptr,
        64,
    );
    crate::ktrace::log_fmt(format_args!("wifi: EROM found {} core(s)", cores.len()));
    for c in cores.iter().take(12) {
        crate::ktrace::log_fmt(format_args!(
            "wifi:   core id={:#x} rev={} base={:#x} wrap={:#x}",
            c.id, c.rev, c.base, c.wrap
        ));
    }
    if let Some(arm) = proto::find_arm_core(&cores) {
        out.arm_id = arm.id;
        out.arm_base = arm.base;
        out.arm_wrap = arm.wrap;
    }
    if let Some(sm) = cores.iter().find(|c| c.id == proto::BCMA_CORE_SYS_MEM) {
        out.sysmem_base = sm.base;
        out.sysmem_wrap = sm.wrap;
        out.sysmem_rev = sm.rev;
    }
    out
}

// SYS_MEM / SOCRAM register offsets (Linux `sbsocramregs`).
const SYSMEM_COREINFO: u32 = 0x00;
const SYSMEM_BANKIDX: u32 = 0x10;
const SYSMEM_BANKINFO: u32 = 0x40;
const SRCI_SRNB_MASK: u32 = 0xf0;
const SRCI_SRNB_MASK_EXT: u32 = 0x100;
const SRCI_SRNB_SHIFT: u32 = 4;
const SOCRAM_BANKINFO_SZMASK: u32 = 0x7f;
const SOCRAM_BANKINFO_SZBASE: u32 = 8192;
const SOCRAM_BANKIDX_MEMTYPE_SHIFT: u32 = 8;
const SOCRAM_MEMTYPE_RAM: u32 = 0;

/// Bring SYS_MEM out of reset and sum bank sizes (Linux `brcmf_chip_sysmem_ramsize`).
/// Returns 0 if the core is missing, dead-bus (`0xffffffff`), or the walk fails.
fn sysmem_ramsize(bar0: u64, pci: &PciDevice, base: u32, wrap: u32, rev: u8) -> u32 {
    if base == 0 {
        return 0;
    }
    if wrap != 0 {
        // Disable then release reset (full AI sequence).
        ai_core_reset(bar0, pci, wrap, 0, 0, 0);
        mdelay(2);
    }
    let Some(coreinfo) = bp_read32_probe(bar0, pci, base + SYSMEM_COREINFO) else {
        crate::ktrace::log("wifi", "SYS_MEM coreinfo unreadable (abort)");
        return 0;
    };
    // Dead BAR window or core still in reset → all-ones / all-zeros.
    if coreinfo == 0 || coreinfo == 0xffff_ffff {
        crate::ktrace::log_fmt(format_args!(
            "wifi: SYS_MEM coreinfo={coreinfo:#x} — core not live, skip bank walk"
        ));
        return 0;
    }
    let mut nb = (coreinfo & SRCI_SRNB_MASK) >> SRCI_SRNB_SHIFT;
    if rev >= 23 {
        nb = (coreinfo & (SRCI_SRNB_MASK | SRCI_SRNB_MASK_EXT)) >> SRCI_SRNB_SHIFT;
    }
    if nb == 0 || nb > 32 {
        crate::ktrace::log_fmt(format_args!(
            "wifi: SYS_MEM coreinfo={coreinfo:#x} nb={nb} — ignoring"
        ));
        return 0;
    }
    let mut memsize = 0u32;
    for idx in 0..nb {
        // Linux: bankidx = (MEMTYPE_RAM << 8) | idx
        let bankidx = (SOCRAM_MEMTYPE_RAM << SOCRAM_BANKIDX_MEMTYPE_SHIFT) | idx;
        bp_write32(bar0, pci, base + SYSMEM_BANKIDX, bankidx);
        let Some(bankinfo) = bp_read32_probe(bar0, pci, base + SYSMEM_BANKINFO) else {
            crate::ktrace::log_fmt(format_args!(
                "wifi: SYS_MEM bank {idx} unreadable — stopping size walk"
            ));
            break;
        };
        if bankinfo == 0xffff_ffff {
            crate::ktrace::log_fmt(format_args!(
                "wifi: SYS_MEM bank {idx} dead (0xffffffff) — stopping size walk"
            ));
            break;
        }
        let bsz = ((bankinfo & SOCRAM_BANKINFO_SZMASK) + 1) * SOCRAM_BANKINFO_SZBASE;
        memsize = memsize.saturating_add(bsz);
    }
    crate::ktrace::log_fmt(format_args!(
        "wifi: SYS_MEM ramsize={memsize:#x} (nb={nb} coreinfo={coreinfo:#x} rev={rev})"
    ));
    memsize
}

/// Full dongle firmware download + shared-RAM handshake.
///
/// `fw` is the raw `.bin`; `nvram` is optional ASCII `.txt` (Linux places it
/// at the top of RAM — many Apple modules boot without it when OTP/cal is
/// present).
fn download_fw_nvram(
    dev: &mut BrcmDevice,
    fw: &[u8],
    nvram: Option<&[u8]>,
) -> Result<(), &'static str> {
    if dev.bar0 == 0 || dev.bar2 == 0 {
        return Err("radio BARs not mapped — cannot download firmware");
    }
    if fw.len() < 4 {
        return Err("firmware image too small");
    }
    let chip = if dev.chip_id != 0 && dev.chip_id != 0xffff {
        dev.chip_id
    } else {
        0x4388
    };
    let preferred = proto::rambase_for_chip(chip).unwrap_or(dev.rambase);
    let rstvec = proto::fw_reset_vector(fw).ok_or("firmware missing reset vector")?;
    let nv_owned: Option<Vec<u8>> = nvram.map(pad_nvram_4);
    let fw_len = fw.len() as u32;

    let cores = discover_cores(dev.bar0, &dev.pci);
    dev.arm_core_id = cores.arm_id;
    dev.arm_wrap = cores.arm_wrap;

    // 1. Halt ARM so TCM is ours (brcmf_chip_set_passive → disable_arm CA7).
    //    The per-core FGC|CLK in ai_core_reset is the only clock force needed;
    //    SYS_MEM is then taken out of reset in step 2 (that is what makes TCM
    //    readable — diagnosed via `/wifi diag`).
    if cores.arm_wrap != 0 {
        crate::ktrace::log("wifi", "halting ARM for download");
        arm_halt(dev.bar0, &dev.pci, cores.arm_wrap);
    } else {
        crate::ktrace::log(
            "wifi",
            "ARM wrap unknown — writing TCM cold (chip should already be passive)",
        );
    }

    // 2. SYS_MEM bring-up (may stay dead on Apple until more PCIE2 init).
    let sysmem_sz = if cores.sysmem_base != 0 {
        crate::ktrace::log_fmt(format_args!(
            "wifi: bringing up SYS_MEM base={:#x} wrap={:#x}",
            cores.sysmem_base, cores.sysmem_wrap
        ));
        sysmem_ramsize(
            dev.bar0,
            &dev.pci,
            cores.sysmem_base,
            cores.sysmem_wrap,
            cores.sysmem_rev,
        )
    } else {
        0
    };

    // 3. Prefer BAR2 TCM when host MEM can **read** it (needs BAR2 inside the
    //    axi2af NP slice — see apple_pcie BAR placement). Fall back to BAR0
    //    window only for SI backplane; TCM RAM is not on the SI bus.
    crate::ktrace::log("wifi", "calibrating TCM…");
    let use_bar2 = {
        let t = dev.bar2;
        if t == 0 {
            false
        } else {
            let magic = 0x4252_5232u32; // 'BRR2'
            let w = tcm_bar2_probe_write32(t, preferred, magic);
            unsafe {
                core::arch::asm!("dsb sy", options(nostack, preserves_flags));
            }
            let r = tcm_bar2_probe_read32(t, preferred);
            let ok = w && r == Some(magic);
            crate::ktrace::log_fmt(format_args!(
                "wifi: BAR2 TCM @{preferred:#x}: write_ok={w} read={r:?} usable={ok}"
            ));
            if ok {
                let _ = tcm_bar2_probe_write32(t, preferred, 0);
            }
            ok
        }
    };

    let rambase = if use_bar2 {
        preferred
    } else {
        // BAR0 window cannot reach ARM TCM (different address space than SI).
        // Still try calibrate for diagnostics; then hard-fail with a clear msg.
        crate::ktrace::log(
            "wifi",
            "BAR2 TCM not readable — check BAR2 is in NP MEM window (pci 0xc0… not 0xc1…)",
        );
        let _ = calibrate_rambase(dev.bar0, &dev.pci, preferred);
        return Err(
            "TCM BAR2 not host-readable (likely BAR2 outside axi2af NP window; rebuild with BAR placement fix)",
        );
    };

    // m1n1 WLANBackplane: SRAM_SIZE = 0x1f9000 for this family.
    const M1N1_SRAM_SIZE: u32 = 0x1f_9000;
    let mut ramsize = choose_ramsize(chip, fw, sysmem_sz, dev.bar2_size, rambase);
    if ramsize > M1N1_SRAM_SIZE {
        crate::ktrace::log_fmt(format_args!(
            "wifi: clamping ramsize {ramsize:#x} → m1n1 SRAM_SIZE {M1N1_SRAM_SIZE:#x}"
        ));
        ramsize = M1N1_SRAM_SIZE;
    }
    // Firmware must fit in SRAM.
    if fw_len > ramsize {
        crate::ktrace::log_fmt(format_args!(
            "wifi: WARNING firmware {fw_len:#x} > ramsize {ramsize:#x} — will write only ramsize bytes"
        ));
    }
    let span = fw_span(fw_len.min(ramsize));
    if let Some(n) = &nv_owned {
        crate::ktrace::log_fmt(format_args!(
            "wifi: deferring NVRAM ({}B) (sysmem={sysmem_sz:#x} span={span:#x})",
            n.len()
        ));
    }

    dev.rambase = rambase;
    dev.ramsize = ramsize;

    crate::ktrace::log_fmt(format_args!(
        "wifi: download begin fw={fw_len}B rambase={rambase:#x} ramsize={ramsize:#x} bar2_size={:#x} rstvec={rstvec:#x} arm_wrap={:#x} path=BAR2",
        dev.bar2_size,
        cores.arm_wrap
    ));

    // 4. Copy firmware via BAR2 with recoverable stores + periodic verify.
    let copy_len = fw_len.min(ramsize);
    let fw_slice = &fw[..copy_len as usize];
    crate::ktrace::log("wifi", "copying firmware into TCM (BAR2)…");
    let written = tcm_copy_bar2_verified(dev.bar2, rambase, fw_slice)?;
    if written < copy_len {
        crate::ktrace::log_fmt(format_args!(
            "wifi: partial firmware write {written:#x}/{copy_len:#x}"
        ));
    } else {
        crate::ktrace::log_fmt(format_args!("wifi: firmware copied+verified ({written} bytes)"));
    }
    if written < 64 {
        return Err("TCM rejected firmware write (almost nothing written)");
    }

    let eff_span = fw_span(written);
    let hs_off = rambase + eff_span - 4;
    // Shared seed at end of **SRAM**, not end of image, when SRAM is larger.
    let hs_off = if ramsize > eff_span {
        rambase + ramsize - 4
    } else {
        hs_off
    };
    dev.ramsize = ramsize.min(written.max(eff_span));

    // 5. Clear handshake (recoverable).
    let shared_written = match tcm_bar2_probe_read32(dev.bar2, hs_off) {
        Some(v) => {
            crate::ktrace::log_fmt(format_args!(
                "wifi: handshake loc {hs_off:#x} = {v:#x} (before seed)"
            ));
            if tcm_bar2_probe_write32(dev.bar2, hs_off, 0) {
                let rb = tcm_bar2_probe_read32(dev.bar2, hs_off).unwrap_or(0xdead_beef);
                crate::ktrace::log_fmt(format_args!(
                    "wifi: handshake cleared at {hs_off:#x}, readback={rb:#x}"
                ));
                0
            } else {
                crate::ktrace::log_fmt(format_args!(
                    "wifi: handshake clear failed — seed={v:#x}"
                ));
                v
            }
        }
        None => {
            return Err("TCM handshake location unreadable after copy");
        }
    };
    let poll_off = hs_off;
    crate::ktrace::log_fmt(format_args!(
        "wifi: polling handshake at TCM {poll_off:#x} seed={shared_written:#x}"
    ));

    // 6. Reset vector at TCM offset 0 (Linux write_tcm32(0, rstvec)) + run ARM.
    crate::ktrace::log("wifi", "releasing ARM — waiting for shared RAM handshake");
    if !tcm_bar2_probe_write32(dev.bar2, 0, rstvec) {
        crate::ktrace::log("wifi", "TCM[0] rstvec store failed");
    }
    if cores.arm_wrap != 0 {
        ai_core_reset(
            dev.bar0,
            &dev.pci,
            cores.arm_wrap,
            ARMCR4_BCMA_IOCTL_CPUHALT,
            0,
            0,
        );
    }

    // 7. Poll handshake via BAR2.
    let deadline = crate::arch::now_ms() + FW_UP_TIMEOUT_MS;
    let mut shared = shared_written;
    while shared == shared_written && crate::arch::now_ms() < deadline {
        mdelay(FW_UP_POLL_MS);
        if crate::shell::poll_interrupt() {
            return Err("cancelled");
        }
        shared = tcm_bar2_probe_read32(dev.bar2, poll_off).unwrap_or(shared);
    }
    if shared == shared_written {
        crate::ktrace::log("wifi", "FW failed to initialize (shared addr unchanged)");
        return Err("firmware did not start — shared-RAM handshake timed out");
    }
    if shared < rambase || shared >= rambase.saturating_add(ramsize.max(0x20_0000)) {
        crate::ktrace::log_fmt(format_args!(
            "wifi: shared addr {shared:#x} (rambase={rambase:#x} ramsize={ramsize:#x})"
        ));
        // Still accept if it looks like a TCM pointer.
        if shared < 0x10_0000 {
            return Err("firmware posted invalid shared-RAM address");
        }
    }

    // 8. Read shared-info header via BAR2.
    let Some(flags) = tcm_bar2_probe_read32(dev.bar2, shared) else {
        return Err("shared-info header unreadable");
    };
    let ver = proto::shared_version(flags);
    crate::ktrace::log_fmt(format_args!(
        "wifi: shared RAM @ {shared:#x} flags={flags:#x} version={ver}"
    ));
    if !proto::shared_version_ok(ver) {
        return Err("unsupported PCIe shared-RAM protocol version");
    }

    dev.shared_addr = shared;
    dev.shared_version = ver;
    dev.firmware_up = true;
    crate::ktrace::log("wifi", "dongle firmware UP — rings/ioctl still pending");
    Ok(())
}

/// Optional build-time embed of the miyake 4388 dongle image (from
/// `assets/wifi/brcm/` via `cargo xtask wifi-assets`). Bare m1n1 has no ESP
/// disk, so this is the path that makes `/wifi load` work there.
#[cfg(wifi_fw_embedded)]
static EMBEDDED_FW: &[u8] =
    include_bytes!("../../../../../assets/wifi/brcm/brcmfmac4388-pcie.apple,miyake.bin");
#[cfg(not(wifi_fw_embedded))]
static EMBEDDED_FW: &[u8] = b"";

#[cfg(wifi_fw_embedded)]
fn embedded_nvram() -> Option<&'static [u8]> {
    // Optional sibling .txt — only compiled in when present next to the .bin.
    #[cfg(wifi_nvram_embedded)]
    {
        static NVRAM: &[u8] =
            include_bytes!("../../../../../assets/wifi/brcm/brcmfmac4388-pcie.apple,miyake.txt");
        Some(NVRAM)
    }
    #[cfg(not(wifi_nvram_embedded))]
    {
        None
    }
}
#[cfg(not(wifi_fw_embedded))]
fn embedded_nvram() -> Option<&'static [u8]> {
    None
}

/// Try to load firmware (embedded → synapse store → on-disk ESP/FAT) and
/// download it into the dongle. Sets [`BrcmDevice::firmware_up`] on success.
pub fn try_load_firmware() -> Result<(), &'static str> {
    // Already up?
    if DEV.with(|d| d.as_ref().map(|x| x.firmware_up).unwrap_or(false)) {
        return Ok(());
    }
    let (stem, bar_ok) = DEV
        .with(|d| {
            d.as_ref()
                .map(|x| (x.firmware_stem.clone(), x.bar0 != 0 && x.bar2 != 0))
        })
        .ok_or("no wifi device")?;
    if !bar_ok {
        return Err("radio BARs not mapped — cannot load firmware yet");
    }

    let mut found_path = String::new();
    let mut fw: Option<Vec<u8>> = None;

    // 1. Build-time embed (m1n1 / no-disk boots).
    if EMBEDDED_FW.len() > 64 {
        crate::ktrace::log_fmt(format_args!(
            "wifi: using embedded firmware ({} bytes)",
            EMBEDDED_FW.len()
        ));
        found_path = String::from("<embedded>");
        fw = Some(EMBEDDED_FW.to_vec());
    }

    // 2. Synapse in-memory store (`/brcm/…`).
    if fw.is_none() {
        let candidates = proto::firmware_search_paths(&stem);
        for path in &candidates {
            if let Ok(bytes) = read_file_bytes(path) {
                crate::ktrace::log_fmt(format_args!(
                    "wifi: found firmware {path} ({} bytes)",
                    bytes.len()
                ));
                found_path = path.clone();
                fw = Some(bytes);
                break;
            }
        }
    }

    // 3. Any FAT/ext4 volume (ESP, voice-style wifi disk, data partition).
    if fw.is_none() {
        let disk_names: &[&str] = &[
            "brcm/brcmfmac4388-pcie.apple,miyake.bin",
            "brcm/brcmfmac4387c2-pcie.apple,miyake.bin",
            "brcmfmac4388-pcie.apple,miyake.bin",
            "brcmfmac4387c2-pcie.apple,miyake.bin",
            "brcm/brcmfmac4388-pcie.bin",
            "brcmfmac4388-pcie.bin",
        ];
        if let Some(bytes) = crate::shell::find_on_disks(disk_names) {
            crate::ktrace::log_fmt(format_args!(
                "wifi: found firmware on disk ({} bytes)",
                bytes.len()
            ));
            // Seed the synapse store so subsequent reads hit /brcm/.
            crate::synapse::fs::write(
                "/brcm/brcmfmac4388-pcie.apple,miyake.bin",
                &bytes,
            );
            found_path = String::from("/brcm/brcmfmac4388-pcie.apple,miyake.bin");
            fw = Some(bytes);
        }
    }

    let fw = match fw {
        Some(b) => b,
        None => {
            return Err(
                "firmware not found — run `make wifi-assets` (or cargo xtask wifi-assets) then rebuild; or place .bin in /brcm/",
            );
        }
    };

    // Optional NVRAM: embed → synapse siblings → disk.
    let mut nvram: Option<Vec<u8>> = None;
    if let Some(n) = embedded_nvram() {
        if n.len() > 8 {
            crate::ktrace::log_fmt(format_args!("wifi: using embedded NVRAM ({} bytes)", n.len()));
            nvram = Some(n.to_vec());
        }
    }
    if nvram.is_none() && found_path != "<embedded>" {
        for np in proto::nvram_paths_for_fw(&found_path) {
            if let Ok(bytes) = read_file_bytes(&np) {
                crate::ktrace::log_fmt(format_args!(
                    "wifi: found NVRAM {np} ({} bytes)",
                    bytes.len()
                ));
                nvram = Some(bytes);
                break;
            }
        }
    }
    if nvram.is_none() {
        if let Some(bytes) = crate::shell::find_on_disks(&[
            "brcm/brcmfmac4388-pcie.apple,miyake.txt",
            "brcmfmac4388-pcie.apple,miyake.txt",
            "brcm/brcmfmac4388-pcie.txt",
        ]) {
            crate::ktrace::log_fmt(format_args!(
                "wifi: found NVRAM on disk ({} bytes)",
                bytes.len()
            ));
            crate::synapse::fs::write(
                "/brcm/brcmfmac4388-pcie.apple,miyake.txt",
                &bytes,
            );
            nvram = Some(bytes);
        }
    }

    with_dev(|dev| download_fw_nvram(dev, &fw, nvram.as_deref()))
        .unwrap_or(Err("no wifi device"))
}

fn read_file_bytes(path: &str) -> Result<Vec<u8>, ()> {
    crate::synapse::fs::read(path).ok_or(())
}

/// Is an AI core out of reset and clocked? Linux `brcmf_chip_ai_iscoreup`:
/// IOCTL has CLK set (and not stuck in FGC-only) and RESET_CTL.RESET clear.
/// Reads recoverably through the BAR0 backplane window; `None` when the wrap
/// registers themselves abort (core unpowered).
fn ai_iscoreup(bar0: u64, pci: &PciDevice, wrap: u32) -> Option<bool> {
    if wrap == 0 {
        return None;
    }
    let ioctl = bp_read32_probe(bar0, pci, wrap + BCMA_IOCTL)?;
    let rst = bp_read32_probe(bar0, pci, wrap + BCMA_RESET_CTL)?;
    Some((ioctl & (BCMA_IOCTL_FGC | BCMA_IOCTL_CLK)) == BCMA_IOCTL_CLK
        && (rst & BCMA_RESET_CTL_RESET) == 0)
}

/// **Decisive BAR2/TCM read-abort diagnostic** — resolves the two candidate
/// root causes of "BAR2 writes stick but every read external-aborts" *without
/// guessing*, in a single boot:
///
/// - **(a) outbound-window / BAR placement**: BAR2 never translates → *every*
///   BAR2 read aborts, **including offset 0**. Fix lives in `apple_pcie`
///   (axi2af window / BAR2 pref-bit / placement).
/// - **(b) dongle RAM not up**: the BAR2 aperture *does* translate (offset 0
///   reads back) but the TCM/SYS_MEM region at `rambase` (`0x740000`) aborts
///   because the CA7's SYS_MEM RAM core is held in reset (`coreinfo=0xffffffff`).
///   The tell-tale: after `ai_core_reset(SYS_MEM)` the same TCM read starts
///   answering. Fix lives here in the chip bring-up (reset SYS_MEM before copy).
///
/// All access is recoverable + bounded; safe to run anytime after `/wifi power`.
pub fn diag() -> Vec<String> {
    with_dev(diag_inner).unwrap_or_else(|| {
        alloc::vec![String::from("no wifi device probed — run /wifi power first")]
    })
}

fn diag_inner(dev: &mut BrcmDevice) -> Vec<String> {
    use alloc::format;
    let mut out = Vec::new();

    // Copy the primitives we need into locals so the probe closures capture by
    // value (no borrow of `dev` held across the later core-reset calls).
    let pci = dev.pci;
    let bar0 = dev.bar0;
    let bar2 = dev.bar2;

    // BAR geometry + type bits straight from config space (pref bit matters:
    // the pref window at 0x6a… L2C-aborts on this SoC, so a pref BAR2 in the
    // non-pref window is a distinct failure mode).
    let bar0_lo = crate::pci::read32(pci.bus, pci.dev, pci.func, 0x10);
    let bar2_lo = crate::pci::read32(pci.bus, pci.dev, pci.func, 0x18);
    out.push(format!(
        "BAR0 cpu={bar0:#x} pci_lo={bar0_lo:#010x} pref={} | BAR2 cpu={bar2:#x} size={:#x} pci_lo={bar2_lo:#010x} pref={} 64b={}",
        bar0_lo & 0x8 != 0,
        dev.bar2_size,
        bar2_lo & 0x8 != 0,
        (bar2_lo >> 1) & 0x3 == 0x2,
    ));

    if bar2 == 0 {
        out.push("BAR2 not mapped — cannot probe TCM (rerun /wifi power)".into());
        return out;
    }
    // map_device_gib covers the whole GiB containing bar2, so every offset
    // below (≤ 64 MiB) is MMU-mapped and a device non-response surfaces as a
    // recoverable external abort (None), not an unrecoverable translation fault.
    crate::arch::aarch64::mmu::map_device_gib(bar2);

    let read_at = |off: u32| crate::arch::aarch64::probe_read32(bar2 + off as u64);
    // Write+readback poke: distinguishes a real backing store (value sticks)
    // from a posted-write-into-void (write "ok", read aborts).
    let poke_at = |off: u32| -> (bool, Option<u32>) {
        let magic = 0x4b4f_5445u32 ^ off; // 'KOTE' ^ off
        let w = tcm_bar2_probe_write32(bar2, off, magic);
        // SAFETY: order the store before the load on the same Device location.
        unsafe {
            core::arch::asm!("dsb sy", options(nostack, preserves_flags));
        }
        let rb = tcm_bar2_probe_read32(bar2, off);
        let _ = tcm_bar2_probe_write32(bar2, off, 0);
        (w, rb)
    };

    let (w0, r0) = poke_at(0);
    let (wr, rr) = poke_at(0x74_0000);
    out.push(format!(
        "BAR2 poke @0: write_ok={w0} read={r0:?} (want {:#x}) | @rambase(0x740000): write_ok={wr} read={rr:?} (want {:#x})",
        0x4b4f_5445u32,
        0x4b4f_5445u32 ^ 0x74_0000,
    ));
    out.push(format!(
        "BAR2 plain read: @0={:?} @1M={:?} @rambase={:?}",
        read_at(0),
        read_at(0x10_0000),
        read_at(0x74_0000),
    ));

    let rambase_ok = rr.is_some();

    // ── DECISIVE: BAR2 relocation sweep ───────────────────────────────────
    // BAR0 reads work at *its* CPU base, so the outbound window is live there.
    // BAR2 maps the dongle's TCM regardless of where its base is programmed, so
    // if BAR2 reads at ANY base inside the proven window, the TCM/device side is
    // fine and the only bug is placement (BAR2 sat at a base the window doesn't
    // translate — e.g. the very bottom edge below m1n1's axi2af start). If BAR2
    // aborts at EVERY base, it is device-side (TCM/SYS_MEM not up).
    // All config writes are restored afterward; reads are recoverable.
    let orig_bar2_lo = crate::pci::read32(pci.bus, pci.dev, pci.func, 0x18);
    let orig_bar2_hi = crate::pci::read32(pci.bus, pci.dev, pci.func, 0x1c);
    let bar2_type = orig_bar2_lo & 0xf;
    // PCI bases inside the non-pref bridge window (0xc000_0000..0xfff0_0000),
    // spread across it, all above BAR0's slot to avoid a collision.
    let sweep_bases = [0xc200_0000u64, 0xc400_0000, 0xc800_0000, 0xe000_0000];
    let mut reloc_hit: Option<(u64, u64)> = None; // (pci_base, cpu_base)
    for &b in &sweep_bases {
        let cmd = crate::pci::read32(pci.bus, pci.dev, pci.func, 0x04);
        crate::pci::write32(pci.bus, pci.dev, pci.func, 0x04, cmd & !0b10);
        pci.program_bar64(2, b, bar2_type);
        crate::pci::write32(pci.bus, pci.dev, pci.func, 0x04, cmd | 0b110);
        let _ = crate::pci::read32(pci.bus, pci.dev, pci.func, 0x04);
        // SAFETY: order the config writes before the MEM probe.
        unsafe {
            core::arch::asm!("dsb sy", options(nostack, preserves_flags));
        }
        let cpu = crate::arch::aarch64::apple_pcie::bar_to_cpu(b);
        crate::arch::aarch64::mmu::map_device_gib(cpu);
        let z = crate::arch::aarch64::probe_read32(cpu);
        let ram = crate::arch::aarch64::probe_read32(cpu + 0x74_0000);
        out.push(format!(
            "BAR2 relocate pci={b:#x} cpu={cpu:#x}: @0={z:?} @rambase={ram:?}"
        ));
        if reloc_hit.is_none() && (z.is_some() || ram.is_some()) {
            reloc_hit = Some((b, cpu));
        }
    }
    // Restore original BAR2 placement so /wifi load keeps working.
    let cmd = crate::pci::read32(pci.bus, pci.dev, pci.func, 0x04);
    crate::pci::write32(pci.bus, pci.dev, pci.func, 0x04, cmd & !0b10);
    crate::pci::write32(pci.bus, pci.dev, pci.func, 0x18, orig_bar2_lo);
    crate::pci::write32(pci.bus, pci.dev, pci.func, 0x1c, orig_bar2_hi);
    crate::pci::write32(pci.bus, pci.dev, pci.func, 0x04, cmd | 0b110);
    let _ = crate::pci::read32(pci.bus, pci.dev, pci.func, 0x04);

    // ── SYS_MEM (CA7 RAM) state via the SI backplane window ────────────────
    let cores = discover_cores(bar0, &pci);
    out.push(format!(
        "cores: arm id={:#x} wrap={:#x} up={:?} | sysmem base={:#x} wrap={:#x} rev={} up={:?}",
        cores.arm_id,
        cores.arm_wrap,
        ai_iscoreup(bar0, &pci, cores.arm_wrap),
        cores.sysmem_base,
        cores.sysmem_wrap,
        cores.sysmem_rev,
        ai_iscoreup(bar0, &pci, cores.sysmem_wrap),
    ));

    // Config-space Broadcom register dump + indirect-backplane self-test. These
    // are always reachable (config space, not the clock-gated MEM window), so
    // they show the chip's true clock/window state and whether the 0x98/0x9c
    // path can poke the backplane (PMU/SYS_MEM) when the MEM window can't.
    let rd = |o: u16| crate::pci::read32(pci.bus, pci.dev, pci.func, o);
    out.push(format!(
        "cfg regs: BAR0_WIN(80)={:#x} BAR1_WIN(84)={:#x} SPROM(88)={:#x} SUBSYS(8c)={:#x} INTMASK(94)={:#x} BP_ADDR(98)={:#x} BP_DATA(9c)={:#x} CLK_CTL(a8)={:#x} LINKCTL(bc)={:#x}",
        rd(0x80), rd(0x84), rd(0x88), rd(0x8c), rd(0x94), rd(0x98), rd(0x9c), rd(0xa8), rd(0xbc),
    ));
    let mut bringup_read: Option<u32> = None;
    if cores.sysmem_base != 0 {
        let ci_before = bp_read32_probe(bar0, &pci, cores.sysmem_base + SYSMEM_COREINFO);
        out.push(format!("SYS_MEM coreinfo(before)={ci_before:?}"));

        // The decisive bring-up experiment (VERDICT b path), per the Asahi
        // brcmfmac sequence: (1) halt the ARM CA7 (set_passive) so we own the
        // RAM, (2) take SYS_MEM out of reset with ai_core_reset (its FGC|CLK is
        // the entire clock force — no separate clock register), then re-read the
        // same TCM location. None→Some ⇒ the sequence works and is in /wifi load.
        crate::ktrace::log("wifi", "diag: CA7 halt + SYS_MEM reset (Asahi seq), re-test TCM");
        if cores.arm_wrap != 0 {
            arm_halt(bar0, &pci, cores.arm_wrap);
        }
        ai_core_reset(bar0, &pci, cores.sysmem_wrap, 0, 0, 0);
        mdelay(3);
        let ci_after = bp_read32_probe(bar0, &pci, cores.sysmem_base + SYSMEM_COREINFO);
        let up_after = ai_iscoreup(bar0, &pci, cores.sysmem_wrap);
        out.push(format!(
            "SYS_MEM coreinfo(after)={ci_after:?} up={up_after:?}"
        ));
        let rr_after = read_at(0x74_0000);
        let r0_after = read_at(0);
        out.push(format!(
            "BAR2 read AFTER bring-up: @0={r0_after:?} @rambase={rr_after:?}"
        ));
        bringup_read = rr_after.or(r0_after);
    }

    // ── Verdict ────────────────────────────────────────────────────────────
    // BAR0 reads at its base prove the outbound window is live, so the sweep is
    // authoritative: if BAR2 reads at some relocated base, TCM is up and the bug
    // is placement; if it aborts everywhere, TCM is device-side down.
    if rambase_ok {
        out.push(
            "VERDICT: BAR2 already readable at rambase — no read-abort blocker; proceed to /wifi load".into(),
        );
    } else if bringup_read.is_some() {
        out.push(
            "VERDICT (b) FIXED: BAR2/TCM answered ONLY after force-clock + CA7-halt + SYS_MEM reset — the RAM was held in reset. The same bring-up is now wired into /wifi load; try it.".into(),
        );
    } else if let Some((b, cpu)) = reloc_hit {
        out.push(format!(
            "VERDICT (a): PLACEMENT — BAR2 reads at pci={b:#x} (cpu={cpu:#x}) but not at its programmed base {bar2:#x}. The outbound window doesn't translate BAR2's current spot; fix = place BAR2 in the proven range (apple_pcie probe_bar_candidates)."
        ));
    } else {
        out.push(
            "VERDICT (b): DEVICE-SIDE — BAR2 aborts everywhere though BAR0 reads fine: window live, dongle RAM not up. force-clock + SYS_MEM reset did NOT wake it (see clk_ctl_st/coreinfo above) — needs PMU power-up or the full chip attach.".into(),
        );
    }

    for l in &out {
        crate::ktrace::log_fmt(format_args!("wifi: diag: {l}"));
    }
    out
}

/// Kick a firmware scan (requires firmware_up). Populates `scan_cache`.
pub fn scan() -> Result<Vec<proto::BssInfo>, &'static str> {
    let up = DEV.with(|d| d.as_ref().map(|x| x.firmware_up).unwrap_or(false));
    if !up {
        return Err("firmware not running — load brcmfmac firmware first");
    }
    // M3: BRCMF_C_SCAN + event ring drain.
    Err("scan ioctl path not yet wired (firmware is up but rings pending)")
}

/// Associate with `ssid` using WPA2-PSK `psk` (passphrase).
pub fn connect(ssid: &str, _psk: &str) -> Result<(), &'static str> {
    if ssid.is_empty() {
        return Err("empty SSID");
    }
    let up = DEV.with(|d| d.as_ref().map(|x| x.firmware_up).unwrap_or(false));
    if !up {
        return Err("firmware not running — load brcmfmac firmware first");
    }
    // M3: SET_SSID + SET_WSEC(AES) + SET_WPA_AUTH(PSK) + SET_WSEC_PMK.
    Err("connect/WPA2 path not yet wired (firmware is up but rings pending)")
}
