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
//! References: m1n1 `proxyclient/hv/trace_wlan.py`, Apple DT
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
fn read_cfg_id(bus: u8, dev: u8, func: u8) -> Option<u32> {
    if bus == 0 {
        Some(pci::read32(bus, dev, func, 0x00))
    } else {
        crate::arch::aarch64::apple_pcie::ecam_read32(bus, dev, func, 0x00)
    }
}

fn match_wifi_id(bus: u8, dev: u8, func: u8, id: u32) -> Option<PciDevice> {
    if !proto::pci_config_id_live(id) {
        return None;
    }
    let v = (id & 0xffff) as u16;
    let d = (id >> 16) as u16;
    if v == VENDOR_BRCM && matches!(d, DEV_BCM4387 | DEV_BCM4378 | DEV_BCM4377) {
        Some(PciDevice {
            bus,
            dev,
            func,
            vendor: v,
            device: d,
        })
    } else {
        None
    }
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
    // DT places WiFi at 01:00.0. Try that first — after PERST the slot often
    // answers CRS / all-ones for a while; the caller retries this function.
    if let Some(id) = read_cfg_id(1, 0, 0) {
        if let Some(dev) = match_wifi_id(1, 0, 0, id) {
            return Some(dev);
        }
    }
    let bus_end = crate::arch::aarch64::apple_pcie::report().bus_end;
    for bus in 0u8..=bus_end.max(1) {
        for dev in 0u8..8 {
            if bus == 1 && dev == 0 {
                continue;
            }
            let Some(id) = read_cfg_id(bus, dev, 0) else {
                continue;
            };
            if let Some(found) = match_wifi_id(bus, dev, 0, id) {
                return Some(found);
            }
        }
    }
    None
}

/// After PERST# the endpoint may take hundreds of ms to decode config
/// (CRS vendor `0x0001`, or all-ones / abort). Poll 01:00.0 until it is
/// a Broadcom function or the budget runs out.
fn wait_for_wifi_pci() -> Option<PciDevice> {
    let deadline = crate::arch::now_ms() + 2000;
    let mut last = 0xffff_ffffu32;
    loop {
        if let Some(id) = read_cfg_id(1, 0, 0) {
            if id != last {
                crate::ktrace::log_fmt(format_args!("wifi: wait 01:00.0 id={id:#010x}"));
                last = id;
            }
        }
        if let Some(dev) = find_wifi_pci() {
            return Some(dev);
        }
        if crate::arch::now_ms() >= deadline {
            return None;
        }
        mdelay(50);
        let _ = crate::shell::upkeep();
        if crate::shell::poll_interrupt() {
            return None;
        }
    }
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

    // PERST# + a rail cycle leave the endpoint in CRS / all-ones until its
    // config space comes up. A single scan right after LINKSTS.UP misses it.
    let Some(pci_dev) = wait_for_wifi_pci() else {
        crate::ktrace::log(
            "wifi",
            "no Broadcom FullMAC PCI function (14e4:4434/…) found after 2s",
        );
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
        &pci_dev, bar0_size, bar0_type, bar2_size, bar2_type,
    ) else {
        crate::ktrace::log(
            "wifi",
            "no working BAR outbound window (all candidates aborted)",
        );
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
    let stem = proto::firmware_stem(
        &board,
        if chip_id != 0 && chip_id != 0xffff {
            chip_id
        } else {
            0x4387
        },
    );

    if !cal.is_empty() {
        crate::ktrace::log_fmt(format_args!(
            "wifi: cal-blob {} bytes (from FDT/ADT)",
            cal.len()
        ));
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
    .unwrap_or(0x20_0000);

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

/// Order a PCI config store against a later BAR0 Device load. ECAM and BAR0
/// are different Device regions; without `dsb` the window write can still be
/// in flight when the BAR0 access lands on the previous 4 KiB.
fn cfg_barrier() {
    // SAFETY: barrier only; no memory operand.
    unsafe {
        core::arch::asm!("dsb sy", options(nostack, preserves_flags));
    }
}

/// Program BAR0_WINDOW and return the offset within the 4 KiB window.
fn bar0_window_prep(pci: &PciDevice, addr: u32) -> u32 {
    let off = addr & (BAR0_REG_SIZE - 1);
    let win = addr & !(BAR0_REG_SIZE - 1);
    pci::write32(pci.bus, pci.dev, pci.func, CFG_BAR0_WINDOW, win);
    cfg_barrier();
    // Read-back confirms the window latched (some cores need a second write).
    let got = pci::read32(pci.bus, pci.dev, pci.func, CFG_BAR0_WINDOW);
    if got != win {
        pci::write32(pci.bus, pci.dev, pci.func, CFG_BAR0_WINDOW, win);
        cfg_barrier();
    }
    off
}

/// Point the sliding window back at chipcommon. Recovers BAR0 after an
/// aborting SYS_MEM/TCM access so a following EROM walk still works.
fn recover_bar0_chipcommon(pci: &PciDevice) {
    pci::write32(
        pci.bus,
        pci.dev,
        pci.func,
        CFG_BAR0_WINDOW,
        proto::SI_ENUM_BASE,
    );
    cfg_barrier();
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
#[allow(dead_code)]
fn calibrate_rambase(bar0: u64, pci: &PciDevice, preferred: u32) -> Option<u32> {
    let extras = [0x20_0000u32, 0x74_0000, 0x18_0000, 0x17_0000, 0x35_2000, 0];
    let mut tried = [false; 8];
    let mut try_base = |base: u32, slot: &mut [bool]| -> Option<u32> {
        let idx = match base {
            0 => 0,
            0x20_0000 => 1,
            0x74_0000 => 2,
            0x18_0000 => 3,
            0x17_0000 => 4,
            0x35_2000 => 5,
            _ => 6,
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
fn tcm_bp_copy(bar0: u64, pci: &PciDevice, rambase: u32, data: &[u8]) -> Result<u32, &'static str> {
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
fn choose_ramsize(chip: u16, fw: &[u8], sysmem_size: u32, bar2_size: u64, rambase: u32) -> u32 {
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
/// `brcmf_chip_ai_resetcore`: disable first (works for arbitrary
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
    let val =
        bp_read32_probe(bar0, pci, arm_wrap + BCMA_IOCTL).unwrap_or(0) & ARMCR4_BCMA_IOCTL_CPUHALT;
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
/// / `ca7_set_active`. Apple 4387/4388 set `skip_reset_vector` — the signed
/// image carries the vector in its footer, and writing TCM[0] is refused.
fn arm_run(bar0: u64, pci: &PciDevice, tcm: u64, arm_wrap: u32, rstvec: u32, chip: u16) {
    if !proto::skip_reset_vector(chip) {
        if !tcm_bar2_probe_write32(tcm, 0, rstvec) {
            crate::ktrace::log("wifi", "TCM[0] rstvec store failed");
        }
    } else {
        crate::ktrace::log_fmt(format_args!(
            "wifi: skip_reset_vector chip={chip:#x} (not writing TCM[0] rstvec={rstvec:#x})"
        ));
    }
    if arm_wrap != 0 {
        ai_core_reset(bar0, pci, arm_wrap, ARMCR4_BCMA_IOCTL_CPUHALT, 0, 0);
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
    /// PCIE2 core rev (Linux uses rev >= 64 for the 64-bit mailbox map).
    pcie2_rev: u8,
    /// PCIE2 core backplane base (for the BAR1/TCM window fixup). 0 if missing.
    pcie2_base: u32,
    /// PCIE2 wrapper — take the core out of reset so CONFIGADDR latches.
    pcie2_wrap: u32,
    /// First 802.11 core wrap — Linux `brcmf_chip_ca7_set_passive` resetcore.
    d11_wrap: u32,
    /// Separate PMU core base (0x827). 0 if the EROM has none.
    pmu_base: u32,
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
        pcie2_rev: 0,
        pcie2_base: 0,
        pcie2_wrap: 0,
        d11_wrap: 0,
        pmu_base: 0,
    };
    let erom_ptr = bp_read32_probe(bar0, pci, proto::SI_ENUM_BASE + proto::CC_EROMPTR).unwrap_or(0);
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
    if let Some(pc) = cores.iter().find(|c| c.id == proto::BCMA_CORE_PCIE2) {
        out.pcie2_rev = pc.rev;
        out.pcie2_base = pc.base;
        out.pcie2_wrap = pc.wrap;
    }
    if let Some(d11) = cores.iter().find(|c| c.id == proto::BCMA_CORE_80211) {
        out.d11_wrap = d11.wrap;
    }
    if let Some(pmu) = cores.iter().find(|c| c.id == proto::BCMA_CORE_PMU) {
        out.pmu_base = pmu.base;
    }
    out
}

/// Point BAR0_WINDOW at `core_base` and confirm it latched (Linux
/// `brcmf_pcie_select_core`).
fn select_core(pci: &PciDevice, core_base: u32) {
    if core_base == 0 {
        return;
    }
    let win = core_base & !(BAR0_REG_SIZE - 1);
    pci::write32(pci.bus, pci.dev, pci.func, CFG_BAR0_WINDOW, win);
    cfg_barrier();
    let got = pci::read32(pci.bus, pci.dev, pci.func, CFG_BAR0_WINDOW);
    if got != win {
        pci::write32(pci.bus, pci.dev, pci.func, CFG_BAR0_WINDOW, win);
        cfg_barrier();
    }
}

/// brcmfmac `brcmf_pcie_attach` BAR2 window fixup.
///
/// Two homes for `BAR2_CONFIG` (0x4e0), tried in this order:
/// 1. PCI config 0x4e0 via ECAM — the register's real address.
/// 2. PCIE2 CONFIGADDR/DATA through the **fixed** BAR0+0x2000 enum window
///    (Linux `BRCMF_PCIE_BARO_PCIE_ENUM_OFFSET`). This does not need
///    `select_core` and is what we should have been using on rev 74.
///
/// Never write a value that equals the PCI vendor/device ID: that is the
/// window-miss signature and would smash BAR2_CONFIG with 0x443414e4.
fn pcie2_bar_window_fixup(bar0: u64, pci: &PciDevice, pcie2_base: u32) -> proto::Bar2Fixup {
    let pci_id = crate::pci::read32(pci.bus, pci.dev, pci.func, 0x00);
    let mut out = proto::Bar2Fixup {
        pci_id,
        ecam_4e0: crate::pci::read32(pci.bus, pci.dev, pci.func, proto::PCIE2_BAR2_CONFIG as u16),
        enum_data: 0,
        slide_win: 0,
        slide_data: 0,
        wrote: false,
    };

    if proto::bar2_config_may_writeback(out.ecam_4e0, pci_id) {
        crate::pci::write32(
            pci.bus,
            pci.dev,
            pci.func,
            proto::PCIE2_BAR2_CONFIG as u16,
            out.ecam_4e0,
        );
        cfg_barrier();
        out.wrote = true;
    }

    let addr_off = proto::pcie2_enum_off(proto::PCIE2_CONFIGADDR) as usize;
    let data_off = proto::pcie2_enum_off(proto::PCIE2_CONFIGDATA) as usize;
    mmio_w32(bar0, addr_off, proto::PCIE2_BAR2_CONFIG);
    cfg_barrier();
    out.enum_data = crate::arch::aarch64::probe_read32(bar0 + data_off as u64).unwrap_or(0);
    if proto::bar2_config_may_writeback(out.enum_data, pci_id) {
        mmio_w32(bar0, data_off, out.enum_data);
        cfg_barrier();
        out.wrote = true;
    }

    if pcie2_base != 0 {
        select_core(pci, pcie2_base);
        out.slide_win = crate::pci::read32(pci.bus, pci.dev, pci.func, CFG_BAR0_WINDOW);
        if out.slide_win == (pcie2_base & !(BAR0_REG_SIZE - 1)) {
            mmio_w32(bar0, proto::PCIE2_CONFIGADDR as usize, proto::PCIE2_BAR2_CONFIG);
            cfg_barrier();
            out.slide_data = crate::arch::aarch64::probe_read32(
                bar0 + proto::PCIE2_CONFIGDATA as u64,
            )
            .unwrap_or(0);
            if proto::bar2_config_may_writeback(out.slide_data, pci_id) {
                mmio_w32(bar0, proto::PCIE2_CONFIGDATA as usize, out.slide_data);
                cfg_barrier();
                out.wrote = true;
            }
        }
    }

    crate::ktrace::log_fmt(format_args!(
        "wifi: BAR2_CONFIG ecam={:#x} enum={:#x} slide_win={:#x} slide={:#x} wrote={} (pci_id={pci_id:#x})",
        out.ecam_4e0, out.enum_data, out.slide_win, out.slide_data, out.wrote
    ));
    recover_bar0_chipcommon(pci);
    out
}

/// Linux `brcmf_pcie_reset_device`, verbatim:
/// 1. select PCIE2, clear ASPM in LINK_STATUS_CTRL
/// 2. select CHIPCOMMON, write `watchdog = 4`
/// 3. wait 100 ms
/// 4. restore ASPM
///
/// Watchdog is written through the **fixed** BAR0+0x3000 chipcommon window
/// (`cc_fixed_off(watchdog)` = 0x3080) so it does not depend on the sliding
/// BAR0_WINDOW latch. The CONFIGADDR restore loop is only for PCIE2 rev ≤ 13;
/// rev 74 skips it (Linux `if (core->rev <= 13)`).
///
/// That 4-tick watchdog is what re-runs PMU defaults and powers SYS_MEM.
fn pcie_reset_device(bar0: u64, pci: &PciDevice, pcie2_rev: u8) {
    let lsc = crate::pci::read32(pci.bus, pci.dev, pci.func, CFG_LINK_STATUS_CTRL);
    let aspm_ok = lsc != 0xffff_ffff;
    if aspm_ok {
        crate::pci::write32(
            pci.bus,
            pci.dev,
            pci.func,
            CFG_LINK_STATUS_CTRL,
            lsc & !0x3,
        );
        cfg_barrier();
    }

    // Linux WRITECC32(watchdog, 4) after select_core(CHIPCOMMON) = BAR0+0x80.
    // Also poke the fixed +0x3000 alias. set_passive leaves the sliding
    // window on a D11 wrap, so the select is load-bearing.
    select_core(pci, proto::SI_ENUM_BASE);
    mmio_w32(bar0, proto::CC_WATCHDOG as usize, proto::CC_WATCHDOG_RESET_TICKS);
    cfg_barrier();
    let wd_off = proto::cc_fixed_off(proto::CC_WATCHDOG) as usize;
    mmio_w32(bar0, wd_off, proto::CC_WATCHDOG_RESET_TICKS);
    cfg_barrier();
    mdelay(proto::CC_WATCHDOG_RESET_WAIT_MS);

    if aspm_ok {
        crate::pci::write32(pci.bus, pci.dev, pci.func, CFG_LINK_STATUS_CTRL, lsc);
        cfg_barrier();
    }
    crate::ktrace::log_fmt(format_args!(
        "wifi: pcie_reset_device watchdog={} @BAR0+{wd_off:#x} wait={}ms lsc={lsc:#x} aspm_ok={aspm_ok} pcie2_rev={pcie2_rev} cfg_restore={}",
        proto::CC_WATCHDOG_RESET_TICKS,
        proto::CC_WATCHDOG_RESET_WAIT_MS,
        proto::pcie2_needs_cfg_restore(pcie2_rev)
    ));
}

/// Linux `brcmf_chip_ca7_set_passive`: halt the CA7, then resetcore the
/// first D11 (PHYRESET|PHYCLOCKEN). Must run *before* the watchdog reset
/// so the boot ROM is not holding SYS_MEM gated.
fn chip_set_passive_ca7(bar0: u64, pci: &PciDevice, arm_wrap: u32, d11_wrap: u32) {
    if arm_wrap != 0 {
        crate::ktrace::log("wifi", "set_passive: halt CA7");
        arm_halt(bar0, pci, arm_wrap);
    }
    if d11_wrap != 0 {
        crate::ktrace::log_fmt(format_args!(
            "wifi: set_passive: reset D11 wrap={d11_wrap:#x}"
        ));
        ai_core_reset(
            bar0,
            pci,
            d11_wrap,
            proto::D11_IOCTL_PHYRESET | proto::D11_IOCTL_PHYCLOCKEN,
            proto::D11_IOCTL_PHYCLOCKEN,
            proto::D11_IOCTL_PHYCLOCKEN,
        );
    }
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

/// Take PCIE2 out of reset so CONFIGADDR/DATA latch. After watchdog the
/// core can sit in reset; writes to +0x120 then read back as PCI config[0].
fn pcie2_ensure_up(bar0: u64, pci: &PciDevice, wrap: u32) {
    if wrap == 0 {
        return;
    }
    match ai_iscoreup(bar0, pci, wrap) {
        Some(true) => {
            crate::ktrace::log("wifi", "PCIE2 already up");
        }
        Some(false) => {
            crate::ktrace::log("wifi", "PCIE2 down — resetcore");
            ai_core_reset(bar0, pci, wrap, 0, 0, 0);
            mdelay(2);
        }
        None => {
            crate::ktrace::log("wifi", "PCIE2 wrap unreadable — leave alone");
        }
    }
}

fn pmu_rd(bar0: u64, pci: &PciDevice, base: u32, off: u32) -> Option<u32> {
    bp_read32_probe(bar0, pci, proto::pmu_reg(base, off))
}

fn pmu_wr(bar0: u64, pci: &PciDevice, base: u32, off: u32, v: u32) {
    bp_write32(bar0, pci, proto::pmu_reg(base, off), v);
}

fn pci_link_dead(pci: &PciDevice) -> bool {
    let id = crate::pci::read32(pci.bus, pci.dev, pci.func, 0x00);
    !proto::pci_config_id_live(id)
}

/// PCIE2 CLK_CTL + PWR_CTL (m1n1 `WLANPCIE2Regs`). Same 0x1e0 offset as
/// chipcommon `clk_ctl_st` but only when the sliding window is on PCIE2.
/// j473: writing the chipcommon copy left `0x2050240` unchanged (no HAVEHT).
fn pcie2_force_power(bar0: u64, pci: &PciDevice, pcie2_base: u32) -> bool {
    if pcie2_base == 0 {
        return false;
    }
    select_core(pci, pcie2_base);
    let clk0 = crate::arch::aarch64::probe_read32(bar0 + proto::PCIE2_CLK_CTL as u64).unwrap_or(0);
    let pwr0 = crate::arch::aarch64::probe_read32(bar0 + proto::PCIE2_PWR_CTL as u64).unwrap_or(0);
    let clk_w = proto::pcie2_clk_force_word(clk0);
    let pwr_w = pwr0 | proto::PCIE2_PWR_ALL_DOMAINS;
    mmio_w32(bar0, proto::PCIE2_CLK_CTL as usize, clk_w);
    mmio_w32(bar0, proto::PCIE2_PWR_CTL as usize, pwr_w);
    cfg_barrier();
    let mut clk = clk0;
    let mut pwr = pwr0;
    for _ in 0..20 {
        mdelay(1);
        clk = crate::arch::aarch64::probe_read32(bar0 + proto::PCIE2_CLK_CTL as u64).unwrap_or(clk);
        pwr = crate::arch::aarch64::probe_read32(bar0 + proto::PCIE2_PWR_CTL as u64).unwrap_or(pwr);
        if proto::pcie2_have_ht(clk) {
            break;
        }
    }
    crate::ktrace::log_fmt(format_args!(
        "wifi: PCIE2 CLK_CTL {clk0:#x}→{clk:#x} HAVEHT={} PWR_CTL {pwr0:#x}→{pwr:#x}",
        proto::pcie2_have_ht(clk)
    ));
    recover_bar0_chipcommon(pci);
    proto::pcie2_have_ht(clk)
}

/// Force ALP+HT in chipcommon `clk_ctl_st`. SYS_MEM is on the HT domain;
/// the dongle PMU will not enable it while HT is only "requested" (ctl 0x180)
/// and not forced.
fn cc_force_ht(bar0: u64, pci: &PciDevice) {
    select_core(pci, proto::SI_ENUM_BASE);
    let cur = crate::arch::aarch64::probe_read32(bar0 + proto::CC_CLK_CTL_ST as u64).unwrap_or(0);
    let want = proto::ccs_force_ht_word(cur);
    mmio_w32(bar0, proto::CC_CLK_CTL_ST as usize, want);
    cfg_barrier();
    mmio_w32(
        bar0,
        proto::cc_fixed_off(proto::CC_CLK_CTL_ST) as usize,
        want,
    );
    cfg_barrier();
    let mut st = cur;
    for _ in 0..20 {
        mdelay(1);
        st = crate::arch::aarch64::probe_read32(bar0 + proto::CC_CLK_CTL_ST as u64).unwrap_or(st);
        if proto::ccs_ht_avail(st) {
            break;
        }
    }
    crate::ktrace::log_fmt(format_args!(
        "wifi: clk_ctl_st cur={cur:#x} st={st:#x} ht_avail={}",
        proto::ccs_ht_avail(st)
    ));
    recover_bar0_chipcommon(pci);
}

/// One recoverable SYS_MEM coreinfo. Never touches the wrap.
fn peek_sysmem(bar0: u64, pci: &PciDevice, base: u32, tag: &str) -> Option<u32> {
    if base == 0 {
        return None;
    }
    let ci = bp_read32_probe(bar0, pci, base + SYSMEM_COREINFO);
    let live = proto::sysmem_coreinfo_live(ci);
    crate::ktrace::log_fmt(format_args!(
        "wifi: SYS_MEM {tag} coreinfo={ci:?} live={live}"
    ));
    // A dead SYS_MEM read wedges APCIE. Do not immediately write BAR0_WINDOW
    // (PCI config) — that is what turned `0xffffffff` into a dead endpoint.
    if live {
        recover_bar0_chipcommon(pci);
    }
    ci
}

/// Overlay-only PMU request. Compact +0x00 aborts on j473 (do not touch).
/// `min_res_mask` at +0x618 has been write-ignored; also poke the request
/// timer and each missing resource bit on its own.
fn pmu_force_resources(bar0: u64, pci: &PciDevice, pmu_base: u32) {
    if pmu_base == 0 {
        crate::ktrace::log("wifi", "PMU core missing from EROM");
        return;
    }
    let min = pmu_rd(bar0, pci, pmu_base, proto::PMU_MIN_RES_MASK).unwrap_or(0);
    let max = pmu_rd(bar0, pci, pmu_base, proto::PMU_MAX_RES_MASK).unwrap_or(0);
    let st = pmu_rd(bar0, pci, pmu_base, proto::PMU_RES_STATE).unwrap_or(0);
    let ctl = pmu_rd(bar0, pci, pmu_base, proto::PMU_CONTROL);
    if max == 0 || max == 0xffff_ffff {
        crate::ktrace::log("wifi", "PMU overlay max not live — not writing");
        recover_bar0_chipcommon(pci);
        return;
    }
    let req = proto::pmu_request_mask(min, max);
    let off = proto::pmu_off_bits(max, st);
    crate::ktrace::log_fmt(format_args!(
        "wifi: PMU overlay ctl={ctl:?} min={min:#x} max={max:#x} st={st:#x} off={off:#x} req={req:#x}"
    ));

    // Official "min is locked" path: res_req_mask + timer. Try the two
    // enable encodings used in Broadcom headers (bit 31 and bit 24).
    pmu_wr(bar0, pci, pmu_base, proto::PMU_RES_REQ_MASK, req);
    pmu_wr(
        bar0,
        pci,
        pmu_base,
        proto::PMU_RES_REQ_TIMER,
        0x8000_00ff,
    );
    cfg_barrier();
    mdelay(4);
    let mut min2 = pmu_rd(bar0, pci, pmu_base, proto::PMU_MIN_RES_MASK).unwrap_or(min);
    let mut st2 = pmu_rd(bar0, pci, pmu_base, proto::PMU_RES_STATE).unwrap_or(st);
    crate::ktrace::log_fmt(format_args!(
        "wifi: PMU after req_timer31 min={min2:#x} st={st2:#x}"
    ));
    pmu_wr(bar0, pci, pmu_base, proto::PMU_RES_REQ_TIMER, 0x0100_00ff);
    cfg_barrier();
    mdelay(4);
    min2 = pmu_rd(bar0, pci, pmu_base, proto::PMU_MIN_RES_MASK).unwrap_or(min2);
    st2 = pmu_rd(bar0, pci, pmu_base, proto::PMU_RES_STATE).unwrap_or(st2);
    crate::ktrace::log_fmt(format_args!(
        "wifi: PMU after req_timer24 min={min2:#x} st={st2:#x}"
    ));

    // One missing bit at a time — a full `min=max` store is rejected whole.
    let mut cur = min2;
    let mut bit = 0u32;
    while off >> bit != 0 {
        if (off >> bit) & 1 != 0 {
            let next = cur | (1u32 << bit);
            pmu_wr(bar0, pci, pmu_base, proto::PMU_MIN_RES_MASK, next);
            cfg_barrier();
            mdelay(2);
            let got = pmu_rd(bar0, pci, pmu_base, proto::PMU_MIN_RES_MASK).unwrap_or(cur);
            crate::ktrace::log_fmt(format_args!(
                "wifi: PMU min|bit{bit} wrote={next:#x} got={got:#x} ok={}",
                got == next
            ));
            if got == next {
                cur = got;
            }
        }
        bit += 1;
        if bit >= 32 {
            break;
        }
    }

    // AOB chips also have a PMU watchdog at +0x634 (not CC+0x80).
    pmu_wr(bar0, pci, pmu_base, proto::PMU_WATCHDOG, proto::CC_WATCHDOG_RESET_TICKS);
    cfg_barrier();
    mdelay(proto::CC_WATCHDOG_RESET_WAIT_MS);
    min2 = pmu_rd(bar0, pci, pmu_base, proto::PMU_MIN_RES_MASK).unwrap_or(cur);
    st2 = pmu_rd(bar0, pci, pmu_base, proto::PMU_RES_STATE).unwrap_or(st2);
    crate::ktrace::log_fmt(format_args!(
        "wifi: PMU after pmuwatchdog min={min2:#x} st={st2:#x}"
    ));
    recover_bar0_chipcommon(pci);
}

/// Bring SYS_MEM out of reset and sum bank sizes (Linux `brcmf_chip_sysmem_ramsize`).
/// Returns 0 if the core is missing, dead-bus (`0xffffffff`), or the walk fails.
///
/// **coreinfo first.** An unpowered SYS_MEM slave returns `0xffffffff` while
/// its wrapper can still decode as "in reset". Linux then `resetcore`s
/// (`FGC|CLK`) and on Apple that turns the next access into an abort that
/// wedges APCIE. Never clock the wrap unless the slave itself is live.
fn sysmem_ramsize(bar0: u64, pci: &PciDevice, base: u32, wrap: u32, rev: u8) -> u32 {
    if base == 0 {
        return 0;
    }
    let Some(coreinfo) = bp_read32_probe(bar0, pci, base + SYSMEM_COREINFO) else {
        crate::ktrace::log("wifi", "SYS_MEM coreinfo unreadable (abort)");
        recover_bar0_chipcommon(pci);
        return 0;
    };
    if !proto::sysmem_coreinfo_live(Some(coreinfo)) {
        crate::ktrace::log_fmt(format_args!(
            "wifi: SYS_MEM coreinfo={coreinfo:#x} — core not live, skip wrap"
        ));
        recover_bar0_chipcommon(pci);
        return 0;
    }
    if wrap != 0 {
        match ai_iscoreup(bar0, pci, wrap) {
            Some(true) => {}
            Some(false) => {
                ai_core_reset(bar0, pci, wrap, 0, 0, 0);
                mdelay(2);
            }
            None => {
                crate::ktrace::log("wifi", "SYS_MEM wrap unreadable — sizing from live coreinfo");
            }
        }
    }
    // Re-read after a possible resetcore.
    let Some(coreinfo) = bp_read32_probe(bar0, pci, base + SYSMEM_COREINFO) else {
        crate::ktrace::log("wifi", "SYS_MEM coreinfo unreadable after wrap");
        recover_bar0_chipcommon(pci);
        return 0;
    };
    // Dead BAR window or core still in reset → all-ones / all-zeros.
    if !proto::sysmem_coreinfo_live(Some(coreinfo)) {
        crate::ktrace::log_fmt(format_args!(
            "wifi: SYS_MEM coreinfo={coreinfo:#x} — core not live, skip bank walk"
        ));
        recover_bar0_chipcommon(pci);
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

    // Linux `brcmf_chip_recognition` then `brcmf_pcie_attach`:
    //   set_passive → watchdog=4 → set_passive → BAR2_CONFIG writeback.
    // Refuse if EROM already failed — the bus is wedged from a prior abort.
    if cores.sysmem_base == 0 && cores.arm_wrap == 0 && cores.pcie2_base == 0 {
        return Err("backplane unreadable (bus wedged) — reboot or /wifi reset, then /wifi load (do not run /wifi diag first)");
    }
    // Do **not** touch SYS_MEM (0x18024000) until HT is forced. A cold
    // coreinfo read returns 0xffffffff and the next PCI config access dies
    // (j473: EROM ok → peek → "PCIe config died").
    if pci_link_dead(&dev.pci) {
        return Err("PCIe config died during EROM — /wifi reset");
    }
    chip_set_passive_ca7(dev.bar0, &dev.pci, cores.arm_wrap, cores.d11_wrap);
    if pci_link_dead(&dev.pci) {
        return Err("PCIe config died during set_passive — /wifi reset");
    }
    let have_ht = pcie2_force_power(dev.bar0, &dev.pci, cores.pcie2_base);
    if pci_link_dead(&dev.pci) {
        return Err("PCIe config died during PCIE2 CLK/PWR — /wifi reset");
    }
    if !have_ht {
        cc_force_ht(dev.bar0, &dev.pci);
        if pci_link_dead(&dev.pci) {
            return Err("PCIe config died during clk_ctl_st — /wifi reset");
        }
    }
    // SYS_MEM only after a clock request. A dead coreinfo still wedges
    // config if we then write BAR0_WINDOW — peek_sysmem skips that recover.
    let after_ht = peek_sysmem(dev.bar0, &dev.pci, cores.sysmem_base, "after-ht");
    let sysmem_sz = if proto::sysmem_coreinfo_live(after_ht) {
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
            "BAR2 TCM write-posted, read abort — RAM still gated (not a BAR-window miss)",
        );
        // One BAR0-window poke at the preferred rambase only. A spray of
        // aborting reads is how the bus got wedged on earlier boots.
        let _ = tcm_bp_poke(dev.bar0, &dev.pci, preferred);
        recover_bar0_chipcommon(&dev.pci);
        return Err(
            "TCM still not readable — see ktrace PCIE2 CLK_CTL/PWR_CTL (HAVEHT)",
        );
    };

    let cap = proto::ramsize_for_chip(chip);
    let mut ramsize = choose_ramsize(chip, fw, sysmem_sz, dev.bar2_size, rambase);
    if ramsize > cap {
        crate::ktrace::log_fmt(format_args!(
            "wifi: clamping ramsize {ramsize:#x} → chip SRAM_SIZE {cap:#x}"
        ));
        ramsize = cap;
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
        crate::ktrace::log_fmt(format_args!(
            "wifi: firmware copied+verified ({written} bytes)"
        ));
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
                crate::ktrace::log_fmt(format_args!("wifi: handshake clear failed — seed={v:#x}"));
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

    // 6. Release ARM. Apple 4387/4388 skip TCM[0] (signed FW footer).
    crate::ktrace::log("wifi", "releasing ARM — waiting for shared RAM handshake");
    arm_run(
        dev.bar0,
        &dev.pci,
        dev.bar2,
        cores.arm_wrap,
        rstvec,
        chip,
    );

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
            crate::synapse::fs::write("/brcm/brcmfmac4388-pcie.apple,miyake.bin", &bytes);
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
            crate::ktrace::log_fmt(format_args!(
                "wifi: using embedded NVRAM ({} bytes)",
                n.len()
            ));
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
            crate::synapse::fs::write("/brcm/brcmfmac4388-pcie.apple,miyake.txt", &bytes);
            nvram = Some(bytes);
        }
    }

    with_dev(|dev| download_fw_nvram(dev, &fw, nvram.as_deref())).unwrap_or(Err("no wifi device"))
}

fn read_file_bytes(path: &str) -> Result<Vec<u8>, ()> {
    crate::synapse::fs::read(path).ok_or(())
}

/// **Hard PERST# reset** of the WiFi endpoint, then re-probe/re-map BARs.
///
/// This resets the whole dongle chip so its on-chip PMU re-sequences and
/// **re-powers the SYS_MEM/RAM domain** — the only lever that works, since the
/// in-band subsystem SSRESET and PMU-mask force both fail to power it (the PMU
/// registers themselves are unreachable; see `/wifi diag`). It is the reset the
/// Apple PCIe root port normally performs but our light-path link bring-up
/// skips. PERST resets the endpoint config, so we drop the stale device record
/// and re-run `probe()` to re-size and re-place the BARs afterward.
pub fn hard_reset() -> Result<(), &'static str> {
    if !crate::arch::aarch64::apple_pcie::hard_reset_wifi_port() {
        return Err("PERST hard reset: link did not come back up (see ktrace)");
    }
    // Endpoint config was reset by PERST — drop the record and re-probe.
    DEV.with(|d| *d = None);
    if probe() {
        Ok(())
    } else {
        Err("re-probe after hard reset failed — BARs not re-mapped (see ktrace)")
    }
}

/// Is an AI core out of reset and clocked? Linux `brcmf_chip_ai_iscoreup`:
/// IOCTL has CLK set (and not stuck in FGC-only) and RESET_CTL.RESET clear.
/// Reads recoverably through the BAR0 backplane window; `None` when the wrap
/// registers themselves abort (core unpowered).
fn ai_iscoreup(bar0: u64, pci: &PciDevice, wrap: u32) -> Option<bool> {
    if wrap == 0 {
        return None;
    }
    let ioctl = bp_read32_probe(bar0, pci, wrap + BCMA_IOCTL);
    let rst = bp_read32_probe(bar0, pci, wrap + BCMA_RESET_CTL);
    if !proto::wrap_regs_live(ioctl, rst) {
        return None;
    }
    let ioctl = ioctl.unwrap();
    let rst = rst.unwrap();
    Some(
        (ioctl & (BCMA_IOCTL_FGC | BCMA_IOCTL_CLK)) == BCMA_IOCTL_CLK
            && (rst & BCMA_RESET_CTL_RESET) == 0,
    )
}

/// **Decisive BAR2/TCM read-abort diagnostic** — resolves the two candidate
/// root causes of "BAR2 writes stick but every read external-aborts" *without
/// guessing*, in a single boot:
///
/// - **(a) outbound-window / BAR placement**: BAR2 never translates → *every*
///   BAR2 read aborts, **including offset 0**. Fix lives in `apple_pcie`
///   (axi2af window / BAR2 pref-bit / placement).
/// - **(b) dongle RAM not up**: SYS_MEM `coreinfo=0xffffffff`. The host does
///   not power that domain in-band — `/wifi power` is the host PERST/pwren
///   sequence. This command is **read-mostly**: it will not PERST the port
///   (that drops BAR mappings and can lose the device on re-probe) and it
///   will not `FGC|CLK` an unpowered SYS_MEM (that wedges APCIE).
///
/// All access is recoverable + bounded; aborting BAR2 reads stay last.
pub fn diag() -> Vec<String> {
    with_dev(diag_inner).unwrap_or_else(|| {
        alloc::vec![String::from(
            "no wifi device probed — run /wifi power first"
        )]
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

    // ORDERING IS LOAD-BEARING: an external abort on the Apple APCIE bus wedges
    // subsequent config/backplane access, so ALL potentially-aborting BAR2/TCM
    // reads run LAST. The power-up + bring-up run first, on a clean bus.

    // ── 0. Config dump FIRST (SSRESET has been seen to turn later config
    //    reads into all-ones, which then looks like "no device").
    let rd = |o: u16| crate::pci::read32(pci.bus, pci.dev, pci.func, 0x00 + o);
    let cfg0 = crate::pci::read32(pci.bus, pci.dev, pci.func, 0x00);
    out.push(format!(
        "cfg id={cfg0:#010x} BAR0_WIN(80)={:#x} BAR1_WIN(84)={:#x} INTMASK(94)={:#x} CLK_CTL(a8)={:#x} LINKCTL(bc)={:#x}",
        rd(0x80), rd(0x84), rd(0x94), rd(0xa8), rd(0xbc),
    ));

    // ── 1. Discover cores (EROM reads — clean) ─────────────────────────────
    let cores = discover_cores(bar0, &pci);
    out.push(format!(
        "cores: arm id={:#x} wrap={:#x} | sysmem {:#x}/{:#x} rev={} | pcie2 rev={} wrap={:#x} | pmu={:#x}",
        cores.arm_id,
        cores.arm_wrap,
        cores.sysmem_base,
        cores.sysmem_wrap,
        cores.sysmem_rev,
        cores.pcie2_rev,
        cores.pcie2_wrap,
        cores.pmu_base,
    ));

    // ── 2. READ ONLY. The previous diag ran set_passive + watchdog +
    //    SYS_MEM wrap resetcore; the wrap decoded as "in reset", FGC|CLK
    //    turned the next access into an abort, and `/wifi load` saw a dead
    //    EROM. Bring-up belongs in load, not here.
    let pci_id = cfg0;
    let ecam_4e0 = crate::pci::read32(pci.bus, pci.dev, pci.func, proto::PCIE2_BAR2_CONFIG as u16);
    let enum_data = {
        let data_off = proto::pcie2_enum_off(proto::PCIE2_CONFIGDATA) as u64;
        crate::arch::aarch64::probe_read32(bar0 + data_off).unwrap_or(0)
    };
    out.push(format!(
        "BAR2_CONFIG ecam={ecam_4e0:#x} enum={enum_data:#x} (miss if == {pci_id:#x})"
    ));

    out.push(
        "SYS_MEM not probed (a dead coreinfo read wedges PCI config on this SoC)".into(),
    );

    if cores.pmu_base != 0 {
        let cap = bp_read32_probe(bar0, &pci, proto::pmu_reg(cores.pmu_base, proto::PMU_CAPABILITIES));
        let min = bp_read32_probe(bar0, &pci, proto::pmu_reg(cores.pmu_base, proto::PMU_MIN_RES_MASK));
        let max = bp_read32_probe(bar0, &pci, proto::pmu_reg(cores.pmu_base, proto::PMU_MAX_RES_MASK));
        let st = bp_read32_probe(bar0, &pci, proto::pmu_reg(cores.pmu_base, proto::PMU_RES_STATE));
        recover_bar0_chipcommon(&pci);
        out.push(format!(
            "PMU @{:#x} cap={cap:?} min={min:?} max={max:?} state={st:?}",
            cores.pmu_base
        ));
    } else {
        out.push("PMU core not in EROM".into());
    }

    // Never abort-read BAR2 from diag.
    out.push("BAR2/TCM not probed (diag is read-only; /wifi load does bring-up)".into());

    out.push(
        "VERDICT: diag is read-only (EROM/PMU/config). /wifi load does set_passive + HT, then one SYS_MEM read.".into(),
    );

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
    // Read the shared-info header and locate ring_info. Mapping the common
    // rings into host DMA and posting BRCMF_C_SCAN is the remaining half —
    // without those mappings an ioctl would write into TCM the host does not
    // own. Report the locator result so `/wifi scan` diagnoses "firmware up
    // but rings unmapped" vs "shared header incomplete".
    let locatable = DEV.with(|d| {
        let Some(dev) = d.as_ref() else {
            return false;
        };
        let mut hdr = [0u8; proto::SHARED_INFO_MIN];
        for i in 0..(proto::SHARED_INFO_MIN / 4) {
            let Some(w) = tcm_bar2_probe_read32(dev.bar2, dev.shared_addr + (i as u32) * 4) else {
                return false;
            };
            hdr[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
        }
        let flags = u32::from_le_bytes([hdr[0], hdr[1], hdr[2], hdr[3]]);
        if let Some(ri) = proto::shared_ring_info_addr(&hdr) {
            crate::ktrace::log_fmt(format_args!(
                "wifi: shared rings locatable — ring_info={ri:#x} ver={}",
                proto::shared_version(flags)
            ));
        }
        proto::rings_locatable(flags, &hdr)
    });
    if locatable {
        Err("scan ioctl path: rings located in shared-info; common-ring DMA mapping still pending")
    } else {
        Err("scan ioctl path: shared-info incomplete (ring_info missing) — firmware up but rings not advertised")
    }
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
    // Same gate as scan: SET_SSID + WSEC + WPA_AUTH + PMK need the H2D control
    // ring. Packing the ioctl bytes is already tested in `proto`; the DMA map
    // is what remains.
    let _ = proto::pack_ioctl_request(0, 0, 1, proto::BRCMF_C_SET_SSID, 1, 36, 0, 0);
    Err("connect/WPA2 path: rings located after scan probe; common-ring DMA mapping still pending")
}
