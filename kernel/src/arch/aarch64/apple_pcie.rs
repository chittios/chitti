//! Apple Silicon **APCIE** host bring-up for bare m1n1 boots.
//!
//! On Apple there is no ACPI MCFG: the PCIe ECAM window is the `config` reg of
//! the FDT `apple,t8112-pcie` / `apple,pcie` node. m1n1 already initialises the
//! root complex + PHY fuses (`src/pcie.c`) before handing off, so we do **not**
//! re-run the full fuse/PHY/tunable dance — we map ECAM, power the WiFi
//! module (SMC), re-train the port link, assign bridge bus numbers, enable the
//! DART for DMA, and leave config-space access to [`crate::pci`].
//!
//! ## What Asahi / Linux actually do (j473 / t8112)
//!
//! From `t8112-j473.dts` + `pcie-apple.c` + `pinctrl-apple-gpio.c`:
//!
//! 1. **pwren** = SMC GPIO 13 (`gP0d` = `0x800001`) — we already do this.
//! 2. **PERST#** is **two** things, both required:
//!    - APCIE `PORT_PERST` (0x814) bit `PERST_OFF`
//!    - **pinctrl_ap GPIO 166**, `GPIO_ACTIVE_LOW` (`reset-gpios` on `port00`)
//! 3. Sequence (Linux `apple_pcie_setup_port`): APPCLK → assert GPIO PERST →
//!    refclk → deassert PORT_PERST + GPIO → 100 ms → LTSSM start.
//! 4. **Bus numbers** are forced by DT (`bus-range = <1 1>` on port00). Without
//!    programming the type-1 bridge's primary/secondary/subordinate (config
//!    0x18), ECAM bus 1 never routes — we saw `sec=0` on all root ports.
//!
//! ## Critical abort rule
//!
//! Secondary-bus ECAM (bus ≥ 1) **external-aborts** while the link is down.
//! Never scan past bus 0 until [`Report::link_up`] is true; use
//! [`super::probe_read32`] for any optional secondary-bus access.
//!
//! Gated on `is_apple()` + the opt-in `chitti.wifi` bootarg.

use core::ptr::{read_volatile, write_volatile};
use super::dart::Dart;

/// FDT compatible for the M2 (t8112) and generic Apple PCIe host.
const PCIE_COMPAT: &[&[u8]] = &[b"apple,t8112-pcie", b"apple,pcie"];

// --- per-port registers (Linux pcie-apple.c / m1n1 APCIE_PORT_*) -----------
const PORT_LTSSMCTL: usize = 0x080;
const PORT_LTSSMCTL_START: u32 = 1 << 0;
const PORT_LINKSTS: usize = 0x208;
const PORT_LINKSTS_UP: u32 = 1 << 0;
const PORT_LINKSTS_BUSY: u32 = 1 << 2;
const PORT_APPCLK: usize = 0x800;
const PORT_APPCLK_EN: u32 = 1 << 0;
const PORT_APPCLK_CGDIS: u32 = 1 << 8;
const PORT_STATUS: usize = 0x804;
const PORT_STATUS_READY: u32 = 1 << 0;
const PORT_REFCLK: usize = 0x810;
const PORT_REFCLK_EN: u32 = 1 << 0;
const PORT_REFCLK_CGDIS: u32 = 1 << 8;
const PORT_PERST: usize = 0x814;
const PORT_PERST_OFF: u32 = 1 << 0; // set = PERST# deasserted
/// Apple port-side **non-pref** MEM window mirrors (1 MiB units).
/// Pref mirrors stay 0 — pref-upper=6 L2C-aborts host MEM on bare m1n1/t8112.
const PORT_MEM_BASE: usize = 0x10;
const PORT_MEM_LIMIT: usize = 0x14;
const PORT_PREF_BASE: usize = 0x18;
const PORT_PREF_LIMIT: usize = 0x20;
const PORT_PREF_BASE_UPPER: usize = 0x24;
const PORT_PREF_LIMIT_UPPER: usize = 0x28;

// --- Apple pinctrl GPIO (drivers/pinctrl/pinctrl-apple-gpio.c) -------------
// j473: reset-gpios = <&pinctrl_ap 166 GPIO_ACTIVE_LOW> for port00.
const PINCTRL_COMPAT: &[&[u8]] = &[b"apple,t8112-pinctrl", b"apple,pinctrl"];
/// Well-known AP pinctrl base on t8112 (FDT `pinctrl@23c100000`).
const PINCTRL_AP_T8112: u64 = 0x2_3c10_0000;
const PERST_GPIO_PORT0: u32 = 166; // pinctrl_ap pin, ACTIVE_LOW
const REG_GPIO_DATA: u32 = 1 << 0;
const REG_GPIO_MODE: u32 = 0x7 << 1; // GENMASK(3,1)
const REG_GPIO_MODE_OUT: u32 = 1 << 1; // FIELD_PREP(MODE, OUT=1)
const REG_GPIO_PERIPH: u32 = 0x3 << 5; // GENMASK(6,5) — clear for GPIO

// PCI config
const PCI_COMMAND: u16 = 0x04;
const PCI_COMMAND_IO: u32 = 1 << 0;
const PCI_COMMAND_MEM: u32 = 1 << 1;
const PCI_COMMAND_MASTER: u32 = 1 << 2;
const PCI_BUS_NUMBER: u16 = 0x18; // primary | sec<<8 | sub<<16
const PCI_EXP_LNKSTA_DLLLA: u16 = 1 << 13;

// --- per-port PHY lane cfg (port_phy base, not RC) -------------------------
const PHY_LANE_CFG: usize = 0x0;
const PHY_LANE_CFG_REFCLK0REQ: u32 = 1 << 0;
const PHY_LANE_CFG_REFCLK1REQ: u32 = 1 << 1;
const PHY_LANE_CFG_REFCLK0ACK: u32 = 1 << 2;
const PHY_LANE_CFG_REFCLK1ACK: u32 = 1 << 3;
const PHY_LANE_CFG_REFCLKEN: u32 = (1 << 9) | (1 << 10);
const PHY_LANE_CFG_REFCLKCGEN: u32 = (1 << 30) | (1 << 31);
const PHY_LANE_CTL: usize = 0x4;
const PHY_LANE_CTL_CFGACC: u32 = 1 << 15;
/// Fallback port0 phy offset from RC (Linux `CORE_PHY_DEFAULT_BASE(0)`).
const PORT0_PHY_OFF: u64 = 0x84000;

/// m1n1 `regs_t8xxx_t600x` / t8112: first 6 `reg` entries are shared
/// (config, rc, phy, phy_ip, axi, fuse). Per-port blocks start at index 6:
///   portN+0 = link/control (APPCLK/STATUS/PERST/LTSSM/LINKSTS)
///   portN+1 = ltssm
///   portN+2 = port phy
/// Earlier code used reg[2] (= global PHY) as "port0" — that is why
/// LTSSM/PERST writes never moved LINKSTS.
const T8XXX_SHARED_REGS: usize = 6;
/// Typical regs per port on t8103/t8112 ADT (m1n1 logs "N reg entries per port").
const T8XXX_PORT_REG_CNT: usize = 4;

/// How long to wait for port0 link after PERST deassert + LTSSM start.
const LINK_WAIT_MS: u64 = 5000;

/// Snapshot of the last bring-up (for `/wifi` / `/lspci` diagnostics).
#[derive(Clone, Copy, Debug, Default)]
pub struct Report {
    pub ready: bool,
    pub ecam: u64,
    pub ecam_size: u64,
    pub port0: u64,
    /// Per-port PHY MMIO (refclk REQ/ACK).
    pub port0_phy: u64,
    /// Root-complex / ctrl MMIO.
    pub rc: u64,
    pub dart: u64,
    pub dart_sid: u32,
    /// AP pinctrl MMIO (for PERST GPIO 166).
    pub pinctrl: u64,
    /// Last bus number passed to [`crate::pci::init`]. **0 when link is down**
    /// so scanners never touch secondary ECAM (which fatal-aborts).
    pub bus_end: u8,
    pub link_up: bool,
}

static REPORT: crate::mm::Locked<Report> = crate::mm::Locked::new(Report {
    ready: false,
    ecam: 0,
    ecam_size: 0,
    port0: 0,
    port0_phy: 0,
    rc: 0,
    dart: 0,
    dart_sid: 0,
    pinctrl: 0,
    bus_end: 0,
    link_up: false,
});

/// Last bring-up snapshot.
pub fn report() -> Report {
    REPORT.with(|r| *r)
}

/// True once ECAM is mapped and root-bus config access is safe.
pub fn ready() -> bool {
    REPORT.with(|r| r.ready)
}

/// True when port0 reports link up (WiFi/BT module is reachable on bus 1).
pub fn port0_link_up() -> bool {
    REPORT.with(|r| r.link_up)
}

fn bootarg_present(needle: &[u8]) -> bool {
    let fdt = super::boot::boot_x0();
    // SAFETY: boot_x0 is the FDT pointer (or non-FDT, rejected by the magic).
    if let Some(c) = unsafe { crate::fdt::chosen(fdt) } {
        if !c.bootargs_ptr.is_null() && c.bootargs_len >= needle.len() {
            // SAFETY: view into the still-mapped FDT.
            let s = unsafe { core::slice::from_raw_parts(c.bootargs_ptr, c.bootargs_len) };
            return s.windows(needle.len()).any(|w| w == needle);
        }
    }
    false
}

#[inline]
fn r32(addr: u64) -> u32 {
    // SAFETY: single 32-bit MMIO read of a mapped APCIE register.
    unsafe { read_volatile(addr as *const u32) }
}
#[inline]
fn w32(addr: u64, v: u32) {
    // SAFETY: single 32-bit MMIO write of a mapped APCIE register.
    unsafe { write_volatile(addr as *mut u32, v) }
}

fn mdelay(ms: u64) {
    let end = crate::arch::now_ms() + ms;
    while crate::arch::now_ms() < end {
        let _ = crate::shell::status_tick();
        core::hint::spin_loop();
    }
}

/// True if `base` looks like a real Apple SoC MMIO window (not FDT garbage).
/// Rejects unaligned junk and the ASCII-in-phys-addr failure mode we hit when
/// `reg_nth` used to read past the property end.
fn is_plausible_mmio(base: u64) -> bool {
    // Apple APCIE/DART windows sit in the high identity-mapped device range.
    // Typical t8112: 0x68x_xxxx_xxxx / 0x69x_xxxx_xxxx. Require 4 KiB alignment
    // and a sane upper bound so a string like "port2" never becomes a FAR.
    base >= 0x2_0000_0000
        && base < 0x10_0000_0000
        && base & 0xfff == 0
}

/// `(ecam, ecam_sz, port0, port0_phy, rc, dart, sid, bus_end)`.
fn discover_pcie() -> Option<(u64, u64, u64, u64, u64, u64, u32, u8)> {
    let fdt = super::boot::boot_x0();
    for &compat in PCIE_COMPAT {
        // SAFETY: FDT from boot_x0.
        let Some((ecam, ecam_sz)) = (unsafe { crate::fdt::reg_nth_of_compatible(fdt, compat, 0) }) else {
            continue;
        };
        let nregs = unsafe { crate::fdt::reg_count_of_compatible(fdt, compat) };
        let rc = unsafe { crate::fdt::reg_nth_of_compatible(fdt, compat, 1) }
            .map(|(b, _)| b)
            .filter(|&b| is_plausible_mmio(b))
            .unwrap_or(0);

        // iommu-map first — on t8112 the pcie0 DART sits at port0+0x8000
        // (0x681008000 next to port0 control 0x681000000), which is our best
        // anchor when the parent FDT only carries config+rc (Linux-style).
        let mut cells = [0u32; 16];
        let n = unsafe { crate::fdt::prop_cells_of_compatible(fdt, compat, b"iommu-map", &mut cells) };
        let (dart, sid) = if n >= 4 {
            let ph = cells[1];
            let sid = cells[2];
            let base = unsafe { crate::fdt::reg_by_phandle(fdt, ph) }
                .map(|(b, _)| b)
                .filter(|&b| is_plausible_mmio(b))
                .unwrap_or(0);
            (base, sid)
        } else {
            (0, 0)
        };

        // --- resolve port0 control MMIO ------------------------------------
        // 1) Full ADT-style reg list (m1n1 bare): port0 at index 6.
        // 2) Scan every reg for a 0x681/2/3_000000-class window.
        // 3) Derive from DART: t8112 pcie0 DART is port0 + 0x8000.
        // 4) Derive from RC: Linux CORE layout port0 = not fixed; use
        //    Asahi/ADT port0 = 0x681000000 when ecam is the t8112 ECAM.
        let mut port0 = 0u64;
        let mut port0_phy = 0u64;
        let mut how = "none";

        if nregs > T8XXX_SHARED_REGS {
            if let Some((b, _)) =
                unsafe { crate::fdt::reg_nth_of_compatible(fdt, compat, T8XXX_SHARED_REGS) }
            {
                if is_plausible_mmio(b) {
                    port0 = b;
                    how = "reg[6]";
                }
            }
            if let Some((b, _)) =
                unsafe { crate::fdt::reg_nth_of_compatible(fdt, compat, T8XXX_SHARED_REGS + 2) }
            {
                if is_plausible_mmio(b) {
                    port0_phy = b;
                }
            }
        }

        if port0 == 0 {
            // Scan all listed regs for a port-control-sized window in the
            // APCIE port aperture (0x6810_0000_0 .. 0x6840_0000_0 on t81xx).
            for i in 0..nregs {
                let Some((b, sz)) = (unsafe { crate::fdt::reg_nth_of_compatible(fdt, compat, i) })
                else {
                    break;
                };
                if !is_plausible_mmio(b) {
                    continue;
                }
                // Port control windows are 0x8000 (Asahi) and live at
                // 0x681/682/683_000000. Skip ECAM (0x690…) and RC (0x6800…).
                let is_port_aperture = (0x6810_0000_0u64..0x6840_0000_0).contains(&b);
                if is_port_aperture && (sz == 0 || sz >= 0x800) {
                    // Prefer the lowest port (WiFi is port0 on mini).
                    if port0 == 0 || b < port0 {
                        port0 = b;
                        how = "reg-scan";
                    }
                }
            }
        }

        if port0 == 0 && dart != 0 {
            // t8112: DART pcie0 at 0x681008000, port0 control at 0x681000000.
            if (dart & 0xffff) == 0x8000 {
                let cand = dart - 0x8000;
                if is_plausible_mmio(cand) {
                    port0 = cand;
                    how = "dart-0x8000";
                }
            } else {
                // Align down to 64 KiB — still lands on 0x681000000 for 0x681008000.
                let cand = dart & !0xffff;
                if is_plausible_mmio(cand) && cand != dart {
                    port0 = cand;
                    how = "dart-align";
                }
            }
        }

        if port0 == 0 && ecam == 0x6900_0000_0 {
            // Last-resort t8112 identity map from Asahi ADT (same SoC as this
            // machine's ECAM). Prefer discovery; this only fires when FDT is
            // Linux-trimmed to config+rc.
            port0 = 0x6810_0000_0;
            how = "t8112-default";
        }

        if port0_phy == 0 && rc != 0 {
            // Linux CORE_PHY_DEFAULT_BASE(0) = rc + 0x84000.
            let cand = rc + PORT0_PHY_OFF;
            if is_plausible_mmio(cand) {
                port0_phy = cand;
            }
        }

        crate::ktrace::log_fmt(format_args!(
            "pcie: nregs={nregs} ecam={ecam:#x} rc={rc:#x} port0={port0:#x} ({how}) phy={port0_phy:#x} dart={dart:#x}"
        ));

        if port0 == 0 || !is_plausible_mmio(port0) {
            crate::ktrace::log("pcie", "could not resolve a plausible port0 MMIO base");
            // Still return ECAM so root-bus access works; link train will no-op.
        }

        let _ = T8XXX_PORT_REG_CNT;

        let mut bus_end = 4u8;
        let mut br = [0u32; 2];
        let bn = unsafe { crate::fdt::prop_cells_of_compatible(fdt, compat, b"bus-range", &mut br) };
        if bn >= 2 && br[1] <= 0xff {
            bus_end = br[1] as u8;
        }
        return Some((ecam, ecam_sz, port0, port0_phy, rc, dart, sid, bus_end));
    }
    None
}

fn port_read_regs(port: u64) -> (u32, u32) {
    (r32(port + PORT_STATUS as u64), r32(port + PORT_LINKSTS as u64))
}

/// m1n1 does **not** require LINKSTS.UP — it waits for STATUS.READY and
/// !LINKSTS.BUSY, then touches config space. We accept either the UP bit or a
/// successful recoverable ECAM read of bus 1 (WiFi lives at 01:00.0).
fn link_seems_up(port: u64) -> (bool, u32, u32) {
    let (st, ls) = port_read_regs(port);
    let ready = (st & PORT_STATUS_READY) != 0;
    let busy = (ls & PORT_LINKSTS_BUSY) != 0;
    let up_bit = (ls & PORT_LINKSTS_UP) != 0;
    if ready && !busy && up_bit {
        return (true, st, ls);
    }
    // Recoverable probe of the secondary bus — only when the port is idle.
    if ready && !busy {
        if let Some(id) = ecam_read32(1, 0, 0, 0x00) {
            let vend = (id & 0xffff) as u16;
            if vend != 0xffff && vend != 0x0000 {
                crate::ktrace::log_fmt(format_args!(
                    "pcie: bus1 probe ok id={id:#010x} (STATUS={st:#x} LINKSTS={ls:#x} up_bit={})",
                    up_bit as u8
                ));
                return (true, st, ls);
            }
        }
    }
    (false, st, ls)
}

// --- pinctrl GPIO PERST ----------------------------------------------------

fn discover_pinctrl_ap() -> u64 {
    let fdt = super::boot::boot_x0();
    for &compat in PINCTRL_COMPAT {
        // Prefer a controller with enough pins for PERST_GPIO_PORT0 (166).
        for node_n in 0..8usize {
            // SAFETY: FDT from boot.
            let Some((base, sz)) =
                (unsafe { crate::fdt::reg_of_nth_node(fdt, compat, node_n, 0) })
            else {
                break;
            };
            if !is_plausible_mmio(base) || sz < 4 * 200 {
                continue;
            }
            // t8112 AP block is at 0x23c100000; accept any large-enough pinctrl
            // in the 0x23cxxxxxx window, or the well-known base.
            if base == PINCTRL_AP_T8112
                || (base >= 0x2_3c00_0000 && base < 0x2_3d00_0000)
                || node_n == 0
            {
                crate::ktrace::log_fmt(format_args!(
                    "pcie: pinctrl_ap @{base:#x} (node {node_n})"
                ));
                return base;
            }
        }
    }
    // t8112 well-known AP pinctrl (Asahi `pinctrl@23c100000`).
    crate::ktrace::log_fmt(format_args!(
        "pcie: pinctrl_ap fallback @{PINCTRL_AP_T8112:#x}"
    ));
    PINCTRL_AP_T8112
}

/// Configure pinctrl pin as GPIO output and drive `level` (1 = high, 0 = low).
/// For ACTIVE_LOW PERST: assert = low (0), deassert = high (1).
fn pinctrl_gpio_out(pinctrl: u64, pin: u32, level_high: bool) {
    if pinctrl == 0 || pin > 512 {
        return;
    }
    let addr = pinctrl + 4 * pin as u64;
    // MODE=OUT, PERIPH=0 (GPIO), DATA=level. Matches apple_gpio_direction_output.
    let mut v = r32(addr);
    v &= !(REG_GPIO_MODE | REG_GPIO_PERIPH | REG_GPIO_DATA);
    v |= REG_GPIO_MODE_OUT;
    if level_high {
        v |= REG_GPIO_DATA;
    }
    w32(addr, v);
}

fn perst_gpio_assert(pinctrl: u64) {
    // ACTIVE_LOW: assert = drive 0.
    pinctrl_gpio_out(pinctrl, PERST_GPIO_PORT0, false);
    crate::ktrace::log_fmt(format_args!(
        "pcie: GPIO PERST pin {PERST_GPIO_PORT0} ASSERT (low)"
    ));
}

fn perst_gpio_deassert(pinctrl: u64) {
    // ACTIVE_LOW: deassert = drive 1.
    pinctrl_gpio_out(pinctrl, PERST_GPIO_PORT0, true);
    crate::ktrace::log_fmt(format_args!(
        "pcie: GPIO PERST pin {PERST_GPIO_PORT0} DEASSERT (high)"
    ));
}

/// Clear sibling roots and configure port0 only (wifi). Shared np windows on
/// every root re-poisoned host MEM on bare m1n1.
fn configure_wifi_root_only() {
    for d in 1u8..4 {
        let id = crate::pci::read32(0, d, 0, 0x00);
        let vend = (id & 0xffff) as u16;
        if vend == 0xffff || vend == 0 {
            continue;
        }
        crate::pci::write32(0, d, 0, PCI_COMMAND, 0);
        crate::pci::write32(0, d, 0, PCI_BUS_NUMBER, 0);
        crate::pci::write32(0, d, 0, 0x20, 0);
        crate::pci::write32(0, d, 0, 0x24, 0x0000_fff0);
        crate::pci::write32(0, d, 0, 0x28, 0);
        crate::pci::write32(0, d, 0, 0x2c, 0);
    }
    configure_root_port(0, 1);
}

/// Program a type-1 root port's bus numbers, command, and MEM windows.
///
/// # Bare m1n1 / t8112 finding (j473, 2026-07)
///
/// Programming **prefetchable upper** `0x28/0x2c = 6` (the Asahi DTS 64-bit
/// pref hole at `0x6_a000_0000`) makes host MEM to that aperture **L2C-abort**
/// (poison). The path that works on bare m1n1:
///
/// - **Non-pref only:** PCI `0xc000_0000..0xfff0_ffff` → CPU `0x6_c000_0000`
/// - Pref window **empty** (base > limit, upper = 0)
/// - Port MMIO mirrors for non-pref only; pref mirrors left 0
/// - Endpoint BAR at PCI `0xc000_0000` (64-bit type, upper dword 0)
///
/// Confirmed on-device: `ldr` @ `0x6c0000000` → chipcommon `0x00018216`.
fn configure_root_port(dev: u8, sec: u8) {
    let id = crate::pci::read32(0, dev, 0, 0x00);
    let vend = (id & 0xffff) as u16;
    if vend == 0xffff || vend == 0 {
        crate::ktrace::log_fmt(format_args!(
            "pcie: configure_root_port: 00:{dev:02x}.0 empty"
        ));
        return;
    }
    // primary=0, secondary=sec, subordinate=sec (single-bus behind the port).
    let bus_reg = 0u32 | ((sec as u32) << 8) | ((sec as u32) << 16);
    crate::pci::write32(0, dev, 0, PCI_BUS_NUMBER, bus_reg);

    // Non-prefetchable 32-bit MEM: base 0xc000_0000, limit 0xfff0_0000.
    // Offset 0x20 = (limit << 16) | base, 1 MiB granularity (top 12 bits).
    crate::pci::write32(0, dev, 0, 0x20, 0xfff0_c000);
    // Pref: **disabled**. Writing upper=6 for the 0x6a hole aborts host MEM
    // on bare m1n1 (reproduced: OPEN/0xffffffff → poison after pref hi write).
    // base > limit empties the window (limit 0x0000, base 0xfff0).
    crate::pci::write32(0, dev, 0, 0x24, 0x0000_fff0);
    crate::pci::write32(0, dev, 0, 0x28, 0);
    crate::pci::write32(0, dev, 0, 0x2c, 0);
    // I/O base/limit: disable (set base > limit).
    crate::pci::write32(0, dev, 0, 0x1c, 0x0000_00f0);

    // Command last so MSE applies after windows stick.
    let cmd = crate::pci::read32(0, dev, 0, PCI_COMMAND);
    crate::pci::write32(
        0,
        dev,
        0,
        PCI_COMMAND,
        cmd | PCI_COMMAND_IO | PCI_COMMAND_MEM | PCI_COMMAND_MASTER,
    );

    // Mirror **non-pref only** into port MMIO. Pref mirrors stay 0.
    if dev == 0 {
        let port = REPORT.with(|r| r.port0);
        if is_plausible_mmio(port) {
            w32(port + PORT_MEM_BASE as u64, 0xc000);
            w32(port + PORT_MEM_LIMIT as u64, 0xfff0);
            w32(port + PORT_PREF_BASE as u64, 0);
            w32(port + PORT_PREF_LIMIT as u64, 0);
            w32(port + PORT_PREF_BASE_UPPER as u64, 0);
            w32(port + PORT_PREF_LIMIT_UPPER as u64, 0);
            crate::ktrace::log_fmt(format_args!(
                "pcie: port0 MEM win np={:#x}/{:#x} pref=0 (no upper-6)",
                r32(port + PORT_MEM_BASE as u64),
                r32(port + PORT_MEM_LIMIT as u64),
            ));
        }
    }

    let bus_rd = crate::pci::read32(0, dev, 0, PCI_BUS_NUMBER);
    let cmd_rd = crate::pci::read32(0, dev, 0, PCI_COMMAND);
    let mem = crate::pci::read32(0, dev, 0, 0x20);
    let pref = crate::pci::read32(0, dev, 0, 0x24);
    let pref_hi_b = crate::pci::read32(0, dev, 0, 0x28);
    let pref_hi_l = crate::pci::read32(0, dev, 0, 0x2c);
    crate::ktrace::log_fmt(format_args!(
        "pcie: root 00:{dev:02x}.0 bus#={bus_rd:#010x} cmd={cmd_rd:#06x} mem={mem:#010x} pref={pref:#010x}/{pref_hi_b:x}:{pref_hi_l:x} (sec={sec})"
    ));
    if mem != 0xfff0_c000 {
        crate::ktrace::log_fmt(format_args!(
            "pcie: root 00:{dev:02x}.0 MEM window write did not stick (want 0xfff0c000 got {mem:#x})"
        ));
    }
}

/// Enable the port's refclk pair (Linux `apple_pcie_setup_refclk`).
/// Soft-fails (logs) if ACKs never arrive — m1n1 may already have the PHY up.
fn setup_port_refclk(port0: u64, port_phy: u64) {
    if port_phy != 0 {
        crate::arch::aarch64::mmu::map_device_gib(port_phy);
        // CFGACC while requesting clocks (t8103/t8112).
        let ctl = r32(port_phy + PHY_LANE_CTL as u64);
        w32(port_phy + PHY_LANE_CTL as u64, ctl | PHY_LANE_CTL_CFGACC);

        let mut cfg = r32(port_phy + PHY_LANE_CFG as u64);
        w32(port_phy + PHY_LANE_CFG as u64, cfg | PHY_LANE_CFG_REFCLK0REQ);
        let t0 = crate::arch::now_ms() + 50;
        while crate::arch::now_ms() < t0 {
            if r32(port_phy + PHY_LANE_CFG as u64) & PHY_LANE_CFG_REFCLK0ACK != 0 {
                break;
            }
            core::hint::spin_loop();
        }
        cfg = r32(port_phy + PHY_LANE_CFG as u64);
        w32(port_phy + PHY_LANE_CFG as u64, cfg | PHY_LANE_CFG_REFCLK1REQ);
        let t1 = crate::arch::now_ms() + 50;
        while crate::arch::now_ms() < t1 {
            if r32(port_phy + PHY_LANE_CFG as u64) & PHY_LANE_CFG_REFCLK1ACK != 0 {
                break;
            }
            core::hint::spin_loop();
        }
        let ctl2 = r32(port_phy + PHY_LANE_CTL as u64);
        w32(port_phy + PHY_LANE_CTL as u64, ctl2 & !PHY_LANE_CTL_CFGACC);
        let cfg2 = r32(port_phy + PHY_LANE_CFG as u64);
        w32(port_phy + PHY_LANE_CFG as u64, cfg2 | PHY_LANE_CFG_REFCLKEN);
        crate::ktrace::log_fmt(format_args!(
            "pcie: port_phy={port_phy:#x} lane_cfg={:#x}",
            r32(port_phy + PHY_LANE_CFG as u64)
        ));
    }

    // Port-side REFCLK enable (t8103/t8112 hw_info.port_refclk).
    let refc = r32(port0 + PORT_REFCLK as u64);
    w32(port0 + PORT_REFCLK as u64, refc | PORT_REFCLK_EN);
}

/// Bring up Apple PCIe ECAM + DART for the WiFi port. Returns true when
/// **root-bus** config-space access is live. Idempotent.
///
/// When the WiFi link is down, `pci` is initialised with `bus_end = 0` so no
/// code path touches secondary ECAM (which would FATAL). Callers that need the
/// WiFi device must check [`port0_link_up`] first.
pub fn init() -> bool {
    if !super::is_apple() {
        return false;
    }
    if REPORT.with(|r| r.ready) {
        // Already inited: if link is still down, re-attempt power (idempotent).
        if !REPORT.with(|r| r.link_up) {
            let _ = power_wifi_and_wait_link(4);
        }
        return true;
    }
    if !bootarg_present(b"chitti.wifi") {
        crate::ktrace::log("pcie", "gated: add `chitti.wifi` bootarg on a bare m1n1 boot");
        return false;
    }

    let Some((ecam, ecam_sz, port0, port0_phy, rc, dart_base, dart_sid, fdt_bus_end)) =
        discover_pcie()
    else {
        crate::ktrace::log("pcie", "no apple,pcie node in the FDT");
        return false;
    };
    crate::ktrace::log_fmt(format_args!(
        "pcie: ECAM {ecam:#x}/{ecam_sz:#x} port0={port0:#x} phy={port0_phy:#x} rc={rc:#x} dart={dart_base:#x} sid={dart_sid}"
    ));

    let pinctrl = discover_pinctrl_ap();

    // Map ECAM (and port regs / RC / DART / pinctrl) as Device — only plausible MMIO.
    crate::arch::aarch64::mmu::map_device_gib(ecam);
    if is_plausible_mmio(port0) {
        crate::arch::aarch64::mmu::map_device_gib(port0);
    }
    if is_plausible_mmio(port0_phy) {
        crate::arch::aarch64::mmu::map_device_gib(port0_phy);
    }
    if is_plausible_mmio(rc) {
        crate::arch::aarch64::mmu::map_device_gib(rc);
    }
    if is_plausible_mmio(dart_base) {
        crate::arch::aarch64::mmu::map_device_gib(dart_base);
    }
    if is_plausible_mmio(pinctrl) {
        crate::arch::aarch64::mmu::map_device_gib(pinctrl);
    }

    // DART stream for bus 1 (WiFi): bypass so identity DRAM DMAs work.
    if is_plausible_mmio(dart_base) {
        // SAFETY: FDT-discovered DART MMIO, Device-mapped above.
        let dart = unsafe { Dart::new(dart_base as usize, dart_sid) };
        if !dart.set_bypass() {
            crate::ktrace::log("pcie", "dart locked — DMA may fail if not already configured");
        }
    }

    // Record bases first so power_wifi_module can use them.
    REPORT.with(|r| {
        r.ecam = ecam;
        r.ecam_size = ecam_sz;
        r.port0 = port0;
        r.port0_phy = port0_phy;
        r.rc = rc;
        r.dart = dart_base;
        r.dart_sid = dart_sid;
        r.pinctrl = pinctrl;
        r.bus_end = 0;
        r.link_up = false;
        r.ready = true; // ECAM root bus is safe
    });
    // Root bus only until link is up (secondary ECAM aborts without link).
    crate::pci::init(ecam, 0);

    // Power the WiFi module (SMC) + train the link, then expand bus_end.
    let link_up = power_wifi_and_wait_link(fdt_bus_end);

    crate::serial_println!(
        "pcie> Apple ECAM {:#x} buses 0..={} dart={:#x}/{} link={}",
        ecam,
        REPORT.with(|r| r.bus_end),
        dart_base,
        dart_sid,
        if link_up { "up" } else { "DOWN" }
    );
    true
}

/// Ensure APPCLK + REFCLK, leave PERST deasserted (m1n1 end-state).
fn port_clocks_on(port: u64, port_phy: u64) {
    let app = r32(port + PORT_APPCLK as u64);
    w32(port + PORT_APPCLK as u64, (app | PORT_APPCLK_EN) & !PORT_APPCLK_CGDIS);
    if is_plausible_mmio(port_phy) {
        setup_port_refclk(port, port_phy);
        let cfg = r32(port_phy + PHY_LANE_CFG as u64);
        w32(port_phy + PHY_LANE_CFG as u64, cfg | PHY_LANE_CFG_REFCLKCGEN);
    } else {
        let refc = r32(port + PORT_REFCLK as u64);
        w32(port + PORT_REFCLK as u64, (refc | PORT_REFCLK_EN) & !PORT_REFCLK_CGDIS);
    }
}

/// Poll until link looks up or `timeout_ms` elapses. Returns (up, last_st, last_ls).
fn wait_link(port: u64, timeout_ms: u64, tag: &str) -> (bool, u32, u32) {
    let deadline = crate::arch::now_ms() + timeout_ms;
    let mut last = (0u32, 0u32);
    loop {
        let (up, st, ls) = link_seems_up(port);
        if (st, ls) != last {
            crate::ktrace::log_fmt(format_args!(
                "pcie: {tag} STATUS={st:#x} LINKSTS={ls:#x}"
            ));
            last = (st, ls);
        }
        if up {
            return (true, st, ls);
        }
        if crate::arch::now_ms() >= deadline {
            return (false, st, ls);
        }
        mdelay(20);
        let _ = crate::shell::upkeep();
    }
}

/// Full PERST cycle + LTSSM start (Linux `apple_pcie_setup_port`).
/// Drives **both** the pinctrl GPIO (ACTIVE_LOW) and `PORT_PERST`.
fn port_perst_cycle_and_ltssm(port: u64, port_phy: u64, pinctrl: u64) {
    // 1) Assert PERST# (GPIO first, then port reg) — device held in reset.
    perst_gpio_assert(pinctrl);
    let perst = r32(port + PORT_PERST as u64);
    w32(port + PORT_PERST as u64, perst & !PORT_PERST_OFF);
    crate::ktrace::log_fmt(format_args!(
        "pcie: {port:#x} PERST assert STATUS={:#x} LINKSTS={:#x}",
        r32(port + PORT_STATUS as u64),
        r32(port + PORT_LINKSTS as u64)
    ));
    mdelay(10);

    // 2) Refclk while held in reset (Linux: assert GPIO, then setup_refclk).
    port_clocks_on(port, port_phy);
    mdelay(1); // Tperst-clk ≥ 100 µs

    // 3) Deassert PORT_PERST then GPIO (Linux order).
    let perst = r32(port + PORT_PERST as u64);
    w32(port + PORT_PERST as u64, perst | PORT_PERST_OFF);
    perst_gpio_deassert(pinctrl);
    crate::ktrace::log_fmt(format_args!(
        "pcie: {port:#x} PERST deassert PERST={:#x}",
        r32(port + PORT_PERST as u64)
    ));
    mdelay(100); // PCIe r5.0 §6.6.1

    // 4) Wait READY
    let deadline = crate::arch::now_ms() + 250;
    while crate::arch::now_ms() < deadline {
        if r32(port + PORT_STATUS as u64) & PORT_STATUS_READY != 0 {
            break;
        }
        core::hint::spin_loop();
    }

    // 5) Kick LTSSM (clear then START so a sticky bit re-arms).
    w32(port + PORT_LTSSMCTL as u64, 0);
    mdelay(1);
    w32(port + PORT_LTSSMCTL as u64, PORT_LTSSMCTL_START);
    // PORT_PREFMEM (0x994) is RAZ/WI on t8112; host MEM uses the non-pref path.
    crate::ktrace::log_fmt(format_args!(
        "pcie: {port:#x} LTSSMCTL={:#x} STATUS={:#x} LINKSTS={:#x}",
        r32(port + PORT_LTSSMCTL as u64),
        r32(port + PORT_STATUS as u64),
        r32(port + PORT_LINKSTS as u64),
    ));
}

/// Dump root-port config space (bus 0) for diagnostics — never touches bus ≥ 1.
fn dump_root_ports() {
    let ecam = REPORT.with(|r| r.ecam);
    if ecam == 0 {
        return;
    }
    // Apple APCIE exposes one type-1 bridge function per port on bus 0.
    for dev in 0u8..8 {
        let id = crate::pci::read32(0, dev, 0, 0x00);
        let vend = (id & 0xffff) as u16;
        if vend == 0xffff || vend == 0 {
            continue;
        }
        let class = crate::pci::read32(0, dev, 0, 0x08);
        let hdr = crate::pci::read8(0, dev, 0, 0x0e);
        let sec = crate::pci::read8(0, dev, 0, 0x19); // secondary bus
        let sub = crate::pci::read8(0, dev, 0, 0x1a);
        // PCIe cap LNKSTA is often at a vendor-specific offset; walk caps for
        // PCI_CAP_ID_EXP (0x10) and read +0x12 (LNKSTA).
        let mut lnksta = 0u16;
        let mut cap = crate::pci::read8(0, dev, 0, 0x34) as u16;
        for _ in 0..16 {
            if cap < 0x40 || cap == 0xff {
                break;
            }
            let w = crate::pci::read16(0, dev, 0, cap);
            let id = (w & 0xff) as u8;
            if id == 0x10 {
                lnksta = crate::pci::read16(0, dev, 0, cap + 0x12);
                break;
            }
            cap = crate::pci::read8(0, dev, 0, cap + 1) as u16;
        }
        crate::ktrace::log_fmt(format_args!(
            "pcie: root 00:{dev:02x}.0 id={id:#010x} class={class:#010x} hdr={hdr:#x} sec={sec} sub={sub} LNKSTA={lnksta:#06x}"
        ));
    }
}

/// Assert SMC WiFi power and bring the WiFi port's link up.
///
/// Bare m1n1 already ran `pcie_init` (axi2af + port APPCLK/PERST deassert).
/// A full GPIO+reg PERST retrain can leave **config** working while **MEM**
/// outbound stays dead — so we prefer a light path when the link is already
/// live, and only fall back to the full Asahi PERST cycle when it is not.
pub fn power_wifi_and_wait_link(fdt_bus_end: u8) -> bool {
    let port0 = REPORT.with(|r| r.port0);
    let port_phy = REPORT.with(|r| r.port0_phy);
    let ecam = REPORT.with(|r| r.ecam);
    let mut pinctrl = REPORT.with(|r| r.pinctrl);
    if ecam == 0 {
        return false;
    }

    crate::ktrace::log("pcie", "link train v6 (bare m1n1: light path if already up)");

    if !is_plausible_mmio(port0) {
        crate::ktrace::log_fmt(format_args!(
            "pcie: refusing link train — port0={port0:#x} not plausible MMIO"
        ));
        return false;
    }

    if pinctrl == 0 || !is_plausible_mmio(pinctrl) {
        pinctrl = discover_pinctrl_ap();
        if is_plausible_mmio(pinctrl) {
            crate::arch::aarch64::mmu::map_device_gib(pinctrl);
            REPORT.with(|r| r.pinctrl = pinctrl);
        }
    }

    dump_root_ports();

    // Bus numbers first — needed for any bus-1 probe even if link is already up.
    // Port0 only for windows; sibling roots cleared (all-roots np re-poisoned MEM).
    configure_wifi_root_only();

    // SMC power always (airplane-mode rail).
    let smc_ok = super::apple_smc::wifi_power_on();
    if !smc_ok {
        crate::ktrace::log("pcie", "SMC wifi power failed — will still try link wait");
    }
    mdelay(100);

    // Light path: m1n1 left APPCLK + PERST deasserted. Ensure clocks, kick
    // LTSSM if needed, do **not** re-assert PERST unless the link is down.
    let (already, st0, ls0) = link_seems_up(port0);
    crate::ktrace::log_fmt(format_args!(
        "pcie: pre-train STATUS={st0:#x} LINKSTS={ls0:#x} already_up={already}"
    ));

    let mut link_up = already;
    if already {
        crate::ktrace::log("pcie", "link already up — skip full PERST cycle (preserve MEM path)");
        // Mild poke: APPCLK + LTSSM start without touching PERST.
        let app = r32(port0 + PORT_APPCLK as u64);
        w32(port0 + PORT_APPCLK as u64, (app | PORT_APPCLK_EN) & !PORT_APPCLK_CGDIS);
        w32(port0 + PORT_LTSSMCTL as u64, PORT_LTSSMCTL_START);
        mdelay(50);
        let (up, st, ls) = link_seems_up(port0);
        link_up = up;
        crate::ktrace::log_fmt(format_args!(
            "pcie: light path STATUS={st:#x} LINKSTS={ls:#x} up={up}"
        ));
    } else {
        crate::ktrace::log("pcie", "link down — full GPIO PERST + LTSSM (Asahi order)");
        let app = r32(port0 + PORT_APPCLK as u64);
        w32(port0 + PORT_APPCLK as u64, app | PORT_APPCLK_EN);
        perst_gpio_assert(pinctrl);
        {
            let perst = r32(port0 + PORT_PERST as u64);
            w32(port0 + PORT_PERST as u64, perst & !PORT_PERST_OFF);
        }
        mdelay(10);
        port_perst_cycle_and_ltssm(port0, port_phy, pinctrl);
        // PERST can clear axi2af / RC state that m1n1 applied at pcie_init.
        reapply_host_tunables();
        let (up, st, ls) = wait_link(port0, LINK_WAIT_MS, "port0");
        link_up = up;
        if link_up {
            crate::ktrace::log_fmt(format_args!(
                "pcie: LINK UP port0 STATUS={st:#x} LINKSTS={ls:#x}"
            ));
        } else {
            crate::ktrace::log_fmt(format_args!(
                "pcie: port0 still DOWN STATUS={st:#x} LINKSTS={ls:#x} smc_ok={smc_ok}"
            ));
        }
    }

    // Bus numbers + MEM windows again after any train / tunable re-apply.
    configure_wifi_root_only();

    // Re-probe bus1 after bus numbers exist — link may already be up with
    // sec=0 so earlier probes couldn't see the endpoint.
    if !link_up {
        mdelay(50);
        let (up, st, ls) = link_seems_up(port0);
        if up {
            crate::ktrace::log_fmt(format_args!(
                "pcie: LINK UP after bus# assign STATUS={st:#x} LINKSTS={ls:#x}"
            ));
            link_up = true;
        }
        // Even without UP bit: if LNKSTA.DLLLA is set on root 00:00.0, trust it.
        if !link_up {
            let mut cap = crate::pci::read8(0, 0, 0, 0x34) as u16;
            for _ in 0..16 {
                if cap < 0x40 || cap == 0xff {
                    break;
                }
                let id = (crate::pci::read16(0, 0, 0, cap) & 0xff) as u8;
                if id == 0x10 {
                    let lnksta = crate::pci::read16(0, 0, 0, cap + 0x12);
                    crate::ktrace::log_fmt(format_args!(
                        "pcie: root0 LNKSTA={lnksta:#06x} DLLLA={}",
                        (lnksta & PCI_EXP_LNKSTA_DLLLA) != 0
                    ));
                    if lnksta & PCI_EXP_LNKSTA_DLLLA != 0 {
                        // Probe bus 1 with numbers programmed.
                        if let Some(id) = ecam_read32(1, 0, 0, 0x00) {
                            let v = (id & 0xffff) as u16;
                            if v != 0xffff && v != 0 {
                                crate::ktrace::log_fmt(format_args!(
                                    "pcie: bus1 id={id:#010x} via DLLLA"
                                ));
                                link_up = true;
                            }
                        }
                    }
                    break;
                }
                cap = crate::pci::read8(0, 0, 0, cap + 1) as u16;
            }
        }
    }

    dump_root_ports();

    let bus_end = if link_up {
        fdt_bus_end.max(1)
    } else {
        // Still open bus 1 after bus# assign so `/wifi` can retry probe
        // (probe_read32 is abort-safe). Safer than leaving sec programmed
        // with bus_end=0 forever.
        0
    };
    // If we programmed sec=1, allow bus_end at least 1 when link is up.
    let bus_end = if link_up { bus_end.max(1) } else { bus_end };
    crate::pci::init(ecam, bus_end);
    REPORT.with(|r| {
        r.bus_end = bus_end;
        r.link_up = link_up;
    });
    link_up
}

/// Re-attempt WiFi power + link (e.g. from `/wifi power` or after a failed probe).
pub fn retry_wifi_power() -> bool {
    if !REPORT.with(|r| r.ready) {
        return init() && REPORT.with(|r| r.link_up);
    }
    // fdt bus end from prior discovery — default 4 for t8112.
    let fdt_end = REPORT.with(|r| if r.bus_end > 0 { r.bus_end } else { 4 });
    power_wifi_and_wait_link(fdt_end)
}

/// **Hard PERST# reset** of the WiFi endpoint, then retrain the link.
///
/// This is the reset the Apple PCIe root port normally performs when the port
/// comes up (`pcie-apple.c`), and it is what leaves the dongle's PMU at defaults
/// with the SYS_MEM/RAM domain **powered**. Our normal bring-up takes the "light
/// path" (skips PERST when the link is already up, to preserve the MEM outbound
/// window), which means the chip is never actually reset — so its RAM domain
/// stays gated off and TCM/SYS_MEM don't decode (diagnosed via `/wifi diag`:
/// SSRESET + PMU-force both fail to power SYS_MEM).
///
/// Forces the full GPIO + PORT_PERST cycle unconditionally, re-applies the
/// axi2af/RC tunables (a PERST retrain clears them, which otherwise leaves MEM
/// outbound dead), waits for link, and reprograms the WiFi root-port windows.
/// The endpoint config is reset by this — the caller MUST re-probe/re-map BARs
/// afterward. Returns true if the link came back up. Bounded + pumps upkeep.
pub fn hard_reset_wifi_port() -> bool {
    if !REPORT.with(|r| r.ready) {
        return init() && REPORT.with(|r| r.link_up);
    }
    let port0 = REPORT.with(|r| r.port0);
    let port_phy = REPORT.with(|r| r.port0_phy);
    let mut pinctrl = REPORT.with(|r| r.pinctrl);
    if !is_plausible_mmio(port0) {
        crate::ktrace::log_fmt(format_args!(
            "pcie: hard reset refused — port0={port0:#x} not plausible MMIO"
        ));
        return false;
    }
    if pinctrl == 0 || !is_plausible_mmio(pinctrl) {
        pinctrl = discover_pinctrl_ap();
        if is_plausible_mmio(pinctrl) {
            crate::arch::aarch64::mmu::map_device_gib(pinctrl);
            REPORT.with(|r| r.pinctrl = pinctrl);
        }
    }
    crate::ktrace::log(
        "pcie",
        "HARD PERST# reset of WiFi endpoint (force full chip reset so PMU re-powers RAM)",
    );

    // SMC power stays on (airplane rail); ensure clocks then full PERST cycle.
    let app = r32(port0 + PORT_APPCLK as u64);
    w32(port0 + PORT_APPCLK as u64, app | PORT_APPCLK_EN);
    port_perst_cycle_and_ltssm(port0, port_phy, pinctrl);
    // A PERST retrain clears the axi2af/RC/config tunables m1n1 applied — without
    // re-applying them MEM outbound stays dead even though config works.
    reapply_host_tunables();
    let (up, st, ls) = wait_link(port0, LINK_WAIT_MS, "port0-hardreset");
    configure_wifi_root_only();
    let bus_end = if up {
        REPORT.with(|r| if r.bus_end > 0 { r.bus_end } else { 4 })
    } else {
        0
    };
    crate::pci::init(REPORT.with(|r| r.ecam), bus_end);
    REPORT.with(|r| {
        r.bus_end = bus_end;
        r.link_up = up;
    });
    crate::ktrace::log_fmt(format_args!(
        "pcie: hard reset done link={} STATUS={st:#x} LINKSTS={ls:#x}",
        if up { "up" } else { "DOWN" }
    ));
    up
}

/// Translate a PCI BAR **bus** address to a CPU physical address using the
/// t8112 `ranges` (Asahi `t8112.dtsi`):
///
/// - 32-bit non-prefetch `0xc000_0000..` → CPU `0x6c000_0000..`  
/// - already-high 64-bit (`≥ 0x6_0000_0000`) → identity  
pub fn bar_to_cpu(bar_bus: u64) -> u64 {
    if bar_bus == 0 {
        return 0;
    }
    if bar_bus >= 0x6_0000_0000 {
        return bar_bus;
    }
    if bar_bus >= 0xc000_0000 {
        return 0x6_0000_0000 + bar_bus;
    }
    bar_bus
}

/// t8112 ADT axi register (m1n1 `regs_t8xxx_t600x.axi_idx` = 4).
const AXI_BASE_T8112: u64 = 0x6_8c00_0000;

/// j473 / `apcie,t8112` ADT `apcie-axi2af-tunables` (dumped via
/// `tools/m1n1_dump_apcie.py`). Re-applied after our PERST cycle — bare m1n1
/// `pcie_init` already ran these once, but link retrain can leave MEM outbound
/// dead while ECAM still works (seen on-device: BAR sticks, `ldr` aborts).
/// Format: (offset, mask, value) for 32-bit RMW.
const AXI2AF_T8112: &[(u32, u32, u32)] = &[
    (0x3c, 0xffff_ffff, 0xffff),
    (0x40, 0xffff_ffff, 0xffff),
    (0x00, 0x203, 0x201),
    (0x0c, 0xfffff, 0x1_6800),
    (0x10, 0xfffff, 0xb400),
    (0x108, 0x13, 0x11),
    (0x10c, 0xfffff, 0x1_6800),
    (0x110, 0xfffff, 0xb400),
    (0x400, 0xc001_03ff, 0xc001_0001),
    (0x600, 0x01ff_ffff, 0x01ff_ffff),
    (0x700, 0x100, 0x0),
    (0x738, 0x03ff_03ff, 0x0020_0020),
    (0x798, 0x03ff_03ff, 0x0080_0010),
    (0x800, 0x100, 0x100),
];

/// j473 `apcie-common-tunables` (RC base).
const COMMON_RC_T8112: &[(u32, u32, u32)] = &[(0x2c, 0xff, 0x1), (0x54, 0xffff_ffff, 0x140)];

/// j473 `/arm-io/apcie/pci-bridge0` `apcie-config-tunables` (port MMIO base).
/// m1n1 applies these **before** PERST deassert; a full PERST cycle clears them.
/// Without 0x140/0x144/0x800, config/link can work while BAR MEM external-aborts
/// (reproduced on bare m1n1: 14e4:4434 + BAR stick, `ldr` poison).
const PORT_CFG_T8112: &[(u32, u32, u32)] = &[
    (0x140, 0x1, 0x1),
    (0x144, 0x00ff_ffff, 0x0025_3770),
    (0x800, 0x00ff_0000, 0x0010_0000), // APPCLK-adjacent bits
];

/// j473 `pcie-rc-tunables` on root-port DBI/ECAM (00:00.0). Same for all bridges.
const PCIE_RC_T8112: &[(u32, u32, u32)] = &[
    (0x078, 0x7000, 0x0),
    (0x194, 0x00fb_ff00, 0x0),
    (0x2a4, 0x8000_0000, 0x0),
    (0x890, 0x1, 0x0),
    (0xb80, 0x3f7f_3f3f, 0x1d1f_3220),
    (0xb84, 0x3f3f, 0x3f3f),
];

/// gen3 + gen4 shadow tunables (merged; gen4 0x890 wins for bit 24).
const PCIE_RC_SHADOW_T8112: &[(u32, u32, u32)] = &[
    (0x154, 0x0f0f, 0x0404),
    (0x178, 0xff, 0x44),
    (0x890, 0x0300_0000, 0x0100_0000), // gen4: set bit 24
    (0x8a8, 0x00ff_ff00, 0x1000),
];

const DWC_DBI_RO_WR: u16 = 0x8bc;

fn rmw_mmio32(base: u64, off: u32, mask: u32, value: u32) {
    let a = base + off as u64;
    let cur = super::probe_read32(a).unwrap_or_else(|| r32(a));
    w32(a, (cur & !mask) | (value & mask));
}

fn rmw_pci32(bus: u8, dev: u8, func: u8, off: u16, mask: u32, value: u32) {
    let cur = crate::pci::read32(bus, dev, func, off);
    crate::pci::write32(bus, dev, func, off, (cur & !mask) | (value & mask));
}

/// Re-apply axi2af + common + **port0 config** + root-port DBI tunables (t8112 j473).
/// Call after any full PERST cycle and before BAR MEM probe.
fn reapply_host_tunables() {
    let axi = AXI_BASE_T8112;
    crate::arch::aarch64::mmu::map_device_gib(axi);
    if super::probe_read32(axi).is_none() {
        crate::ktrace::log("pcie", "axi2af unreadable — skip tunable re-apply");
        return;
    }
    for &(off, mask, val) in AXI2AF_T8112 {
        rmw_mmio32(axi, off, mask, val);
    }

    let rc = REPORT.with(|r| r.rc);
    if is_plausible_mmio(rc) {
        crate::arch::aarch64::mmu::map_device_gib(rc);
        for &(off, mask, val) in COMMON_RC_T8112 {
            rmw_mmio32(rc, off, mask, val);
        }
        w32(rc + 0x4, 0);
        let phyif = r32(rc + 0x24);
        w32(rc + 0x24, phyif | 1);
        let ctl = r32(rc + 0x50);
        w32(rc + 0x50, ctl | 1);
    }

    // Port0 apcie-config-tunables (APPCLK-adjacent + fabric knobs).
    let port0 = REPORT.with(|r| r.port0);
    if is_plausible_mmio(port0) {
        crate::arch::aarch64::mmu::map_device_gib(port0);
        for &(off, mask, val) in PORT_CFG_T8112 {
            rmw_mmio32(port0, off, mask, val);
        }
        crate::ktrace::log_fmt(format_args!(
            "pcie: port0 cfg +0x140={:#x} +0x144={:#x} +0x800={:#x}",
            r32(port0 + 0x140),
            r32(port0 + 0x144),
            r32(port0 + 0x800),
        ));
    }

    // pcie-rc + shadow tunables on root 00:00.0 (and 00:01.0 / 00:02.0).
    // Enable DesignWare RO_WR so protected DBI regs take writes.
    if REPORT.with(|r| r.ecam) != 0 {
        for dev in 0u8..3 {
            let id = crate::pci::read32(0, dev, 0, 0);
            if (id & 0xffff) == 0 || (id & 0xffff) == 0xffff {
                continue;
            }
            let ro = crate::pci::read32(0, dev, 0, DWC_DBI_RO_WR);
            crate::pci::write32(0, dev, 0, DWC_DBI_RO_WR, ro | 1);
            for &(off, mask, val) in PCIE_RC_T8112 {
                rmw_pci32(0, dev, 0, off as u16, mask, val);
            }
            for &(off, mask, val) in PCIE_RC_SHADOW_T8112 {
                rmw_pci32(0, dev, 0, off as u16, mask, val);
            }
            crate::pci::write32(0, dev, 0, DWC_DBI_RO_WR, ro);
        }
    }

    crate::ktrace::log_fmt(format_args!(
        "pcie: re-applied axi2af({}) common({}) port_cfg({}) pcie_rc({}+{})",
        AXI2AF_T8112.len(),
        COMMON_RC_T8112.len(),
        PORT_CFG_T8112.len(),
        PCIE_RC_T8112.len(),
        PCIE_RC_SHADOW_T8112.len(),
    ));
}

/// Ensure the RC fabric is in the m1n1/Linux end-state, re-apply axi2af, map
/// BAR GiBs, and classify ECAM vs BAR-hole probes.
fn prepare_host_mem_path() {
    reapply_host_tunables();

    for pa in [
        0x6_8000_0000u64,
        0x6_a000_0000,
        0x6_c000_0000,
        0x7_0000_0000,
        0x6_9000_0000,
        0x6_b000_0000,
    ] {
        crate::arch::aarch64::mmu::map_device_gib(pa);
    }

    let ecam = REPORT.with(|r| r.ecam);
    let probes = [
        ("ecam", ecam),
        ("bar_hole_6a", 0x6_a000_0000u64),
        ("bar_hole_6c", 0x6_c000_0000u64),
        ("axi", AXI_BASE_T8112),
    ];
    for (tag, a) in probes {
        match super::probe_read32(a) {
            Some(w) => crate::ktrace::log_fmt(format_args!(
                "pcie: decode OK  {tag} @{a:#x} = {w:#010x}"
            )),
            None => crate::ktrace::log_fmt(format_args!(
                "pcie: decode MISS {tag} @{a:#x} (no completer / fabric)"
            )),
        }
    }
}

/// Probe which CPU physical address can actually reach a 64-bit BAR.
///
/// Programs BAR0 (and BAR2 if `bar2_size != 0`) to each candidate **PCI**
/// base, enables MEM, then `probe_read32`s the corresponding CPU address.
/// Returns `(bar0_cpu, bar2_cpu, pci_base_used)` on the first hit.
///
/// Candidates cover the Asahi DTS windows **and** common Apple holes seen on
/// M1/M2 (`0x7_…`) in case m1n1's axi2af tunables wired a different aperture
/// than the Linux `ranges`.
///
/// **Pass 1** skips DesignWare iATU (Asahi Linux never programs it — Apple
/// routes outbound via axi2af). **Pass 2** enables identity iATU regions in
/// case this silicon needs them.
pub fn find_working_bar_window(
    dev: &crate::pci::PciDevice,
    bar0_size: u64,
    bar0_type: u32,
    bar2_size: u64,
    bar2_type: u32,
) -> Option<(u64, u64, u64)> {
    prepare_host_mem_path();

    configure_wifi_root_only();

    // PCIe CEM: ≥100 ms after link up before relying on MEM (OpenBSD waits).
    {
        let end = crate::arch::now_ms() + 100;
        while crate::arch::now_ms() < end {
            let _ = crate::shell::status_tick();
            core::hint::spin_loop();
        }
    }

    // (pci_base_for_bar0, cpu_offset_to_add); cpu = pci + offset.
    // **Non-pref first** — proven on j473 bare m1n1 (chipcommon 0x18216 @ 0x6c).
    // Pref/high-identity candidates stay as fallbacks (may abort on this SoC).
    let candidates: &[(u64, u64)] = &[
        (0xc000_0000, 0x6_0000_0000), // PCI 0xc… → CPU 0x6c… (WORKING path)
        (0xc010_0000, 0x6_0000_0000), // slightly above (BAR2 room)
        (0x6_c000_0000, 0),           // high identity in 0x6c hole
        (0x6_a000_0000, 0),           // DTS pref (often aborts if pref-upper=6)
        (0x7_0000_0000, 0),
        (0xc000_0000, 0),
    ];

    crate::ktrace::log("pcie", "BAR probe (non-pref 0xc→0x6c first)");
    if let Some(hit) =
        probe_bar_candidates(dev, bar0_size, bar0_type, bar2_size, bar2_type, candidates)
    {
        return Some(hit);
    }
    crate::ktrace::log(
        "pcie",
        "BAR probe: ALL miss — MEM outbound not mapped (check pref-upper=0, np window)",
    );
    None
}

fn probe_bar_candidates(
    dev: &crate::pci::PciDevice,
    bar0_size: u64,
    bar0_type: u32,
    bar2_size: u64,
    bar2_type: u32,
    candidates: &[(u64, u64)],
) -> Option<(u64, u64, u64)> {
    for &(pci0, cpu_off) in candidates {
        // ── Placement strategy (j473 bare m1n1) ─────────────────────────
        // The proven host MEM HIT is PCI `0xc000_0000` → CPU `0x6c000_0000`
        // (chipcommon via BAR0). axi2af may only translate a **16 MiB** NP
        // slice there. A 16 MiB BAR2 placed *after* BAR0 lands at
        // `0xc100_0000` → CPU `0x6c100_0000`, which is **outside** that slice:
        // stores appear to succeed (posted) but **loads external-abort** —
        // exactly the WiFi TCM symptom.
        //
        // So put **BAR2 (TCM) first** on the HIT base when it is large, and
        // park the smaller BAR0 after it (still inside the root-port window
        // `0xc000_0000..0xfff0_ffff`). Prefer packing both into the first
        // 16 MiB when BAR2 is small enough.
        let align0 = bar0_size.next_power_of_two().max(0x1000);
        let align2 = if bar2_size != 0 {
            bar2_size.next_power_of_two().max(0x1000)
        } else {
            0x1000
        };
        let hit = (pci0 + align0.max(align2) - 1) & !(align0.max(align2) - 1);

        let (base0, base2) = if bar2_size == 0 {
            let b0 = (pci0 + align0 - 1) & !(align0 - 1);
            (b0, 0u64)
        } else if bar2_size <= 0x100_0000
            && bar0_size + bar2_size <= 0x100_0000
            && align2 <= 0x100_0000
        {
            // Both fit in 16 MiB: BAR0 at hit, BAR2 immediately after (aligned).
            let b0 = (pci0 + align0 - 1) & !(align0 - 1);
            let b2 = (b0 + bar0_size + align2 - 1) & !(align2 - 1);
            if b2 + bar2_size <= b0 + 0x100_0000 {
                (b0, b2)
            } else {
                // BAR2 alone on HIT; BAR0 after the 16 MiB slice.
                let b2 = (pci0 + align2 - 1) & !(align2 - 1);
                let b0 = (b2 + bar2_size + align0 - 1) & !(align0 - 1);
                (b0, b2)
            }
        } else {
            // Large BAR2: claim the HIT base for TCM; BAR0 after it.
            let b2 = (pci0 + align2 - 1) & !(align2 - 1);
            let b0 = (b2 + bar2_size + align0 - 1) & !(align0 - 1);
            (b0, b2)
        };
        let _ = hit;

        // Disable MEM, program, re-enable.
        let cmd = crate::pci::read32(dev.bus, dev.dev, dev.func, 0x04);
        crate::pci::write32(dev.bus, dev.dev, dev.func, 0x04, cmd & !0b10);
        dev.program_bar64(0, base0, bar0_type);
        if bar2_size != 0 {
            dev.program_bar64(2, base2, bar2_type);
        }
        crate::pci::write32(dev.bus, dev.dev, dev.func, 0x04, cmd | 0b110);
        // Flush posted config writes before MEM probe.
        let _ = crate::pci::read32(dev.bus, dev.dev, dev.func, 0x04);
        // SAFETY: DSB so MMIO config is visible to the PCIe fabric.
        unsafe {
            core::arch::asm!("dsb sy", options(nostack, preserves_flags));
        }

        // Verify the BAR write stuck in config space (catches sizing races).
        let bar0_rd = dev.bar(0);
        if bar0_rd & !0xf != base0 & !0xf {
            crate::ktrace::log_fmt(format_args!(
                "pcie: BAR0 config write mismatch want={base0:#x} got={bar0_rd:#x}"
            ));
        }

        let cpu0 = base0.wrapping_add(cpu_off);
        let cpu2 = if base2 != 0 {
            base2.wrapping_add(cpu_off)
        } else {
            0
        };
        crate::arch::aarch64::mmu::map_device_gib(cpu0);
        if cpu2 != 0 {
            crate::arch::aarch64::mmu::map_device_gib(cpu2);
        }
        let end = crate::arch::now_ms() + 10;
        while crate::arch::now_ms() < end {
            core::hint::spin_loop();
        }

        // HIT if BAR0 (regs) **or** BAR2 (TCM base) responds. Prefer a layout
        // where BAR2 reads work — required for firmware download.
        let bar0_word = super::probe_read32(cpu0);
        let bar2_word = if cpu2 != 0 {
            super::probe_read32(cpu2)
        } else {
            None
        };
        // Also try TCM rambase offset on BAR2 (0x740000) when BAR2 is mapped.
        let bar2_ram = if cpu2 != 0 {
            super::probe_read32(cpu2 + 0x74_0000)
        } else {
            None
        };

        crate::ktrace::log_fmt(format_args!(
            "pcie: BAR try pci0={base0:#x} pci2={base2:#x} cpu0={cpu0:#x} cpu2={cpu2:#x} r0={bar0_word:?} r2={bar2_word:?} r2+ram={bar2_ram:?}"
        ));

        if bar0_word.is_some() || bar2_word.is_some() || bar2_ram.is_some() {
            crate::ktrace::log_fmt(format_args!(
                "pcie: BAR window HIT pci0={base0:#x} pci2={base2:#x} cpu0={cpu0:#x} cpu2={cpu2:#x}"
            ));
            return Some((cpu0, cpu2, base0));
        }
        crate::ktrace::log_fmt(format_args!(
            "pcie: BAR window miss pci0={base0:#x} pci2={base2:#x}"
        ));
    }
    None
}

/// Program a DesignWare **outbound iATU** region on root port `00:00.0` so CPU
/// accesses to `[cpu_base, cpu_base+size)` become PCI MEM TLPs (identity).
///
/// **Critical:** `limit_lo` must be ≥ `base_lo` when the hardware compares the
/// low 32 bits alone. A 512 MiB window at `0x6a00_0000_0` has
/// `base_lo=0xa000_0000` and `limit_lo=0x6bff_ffff` — that inverts the range
/// and the region never matches. Keep each region inside one 4 GiB half
/// (e.g. 256 MiB @ `0x6a00_0000_0` → limit `0x6aff_ffff`).
pub fn setup_outbound_mem_window(region: u32, cpu_base: u64, size: u64) {
    let limit = cpu_base.saturating_add(size.saturating_sub(1));
    let base_lo = cpu_base as u32;
    let limit_lo = limit as u32;
    if (cpu_base >> 32) == (limit >> 32) && limit_lo < base_lo {
        crate::ktrace::log_fmt(format_args!(
            "pcie: iATU r{region} SKIP inverted lo range {base_lo:#x}..{limit_lo:#x}"
        ));
        return;
    }

    // Enable DBI writes on the root port (DesignWare RO_WR @ 0x8bc).
    let ro = crate::pci::read32(0, 0, 0, 0x8bc);
    crate::pci::write32(0, 0, 0, 0x8bc, ro | 1);

    // Viewport = outbound region N.
    crate::pci::write32(0, 0, 0, 0x900, region);
    crate::pci::write32(0, 0, 0, 0x904, 0); // CR1: MEM
    crate::pci::write32(0, 0, 0, 0x90c, base_lo);
    crate::pci::write32(0, 0, 0, 0x910, (cpu_base >> 32) as u32);
    crate::pci::write32(0, 0, 0, 0x914, limit_lo);
    // Upper limit (DW ≥ 4.60a viewport) — some builds use 0x924.
    crate::pci::write32(0, 0, 0, 0x920, (limit >> 32) as u32);
    crate::pci::write32(0, 0, 0, 0x924, (limit >> 32) as u32);
    crate::pci::write32(0, 0, 0, 0x918, base_lo); // target = identity
    crate::pci::write32(0, 0, 0, 0x91c, (cpu_base >> 32) as u32);
    crate::pci::write32(0, 0, 0, 0x908, 1u32 << 31); // CR2 enable

    // Unrolled iATU: RC + 0x300000 + region*0x200 (DW 5.x).
    let rc = REPORT.with(|r| r.rc);
    if is_plausible_mmio(rc) {
        let atu = rc + 0x30_0000 + (region as u64) * 0x200;
        crate::arch::aarch64::mmu::map_device_gib(atu);
        let probe = super::probe_read32(atu);
        if probe.is_some() && probe != Some(0xffff_ffff) {
            w32(atu + 0x00, 0); // CTRL1 MEM
            w32(atu + 0x08, base_lo);
            w32(atu + 0x0c, (cpu_base >> 32) as u32);
            w32(atu + 0x10, limit_lo);
            w32(atu + 0x20, (limit >> 32) as u32); // UPPER_LIMIT
            w32(atu + 0x14, base_lo); // target lo (unroll layout varies)
            w32(atu + 0x18, (cpu_base >> 32) as u32);
            // Standard unroll: target at +0x14/+0x18, limit upper +0x20
            w32(atu + 0x04, 1u32 << 31);
            crate::ktrace::log_fmt(format_args!(
                "pcie: iATU unroll r{region} @{atu:#x} {cpu_base:#x}..{limit:#x}"
            ));
        }
    }

    crate::pci::write32(0, 0, 0, 0x8bc, ro);
    crate::ktrace::log_fmt(format_args!(
        "pcie: iATU r{region} identity {cpu_base:#x}..{limit:#x} (lo {base_lo:#x}..{limit_lo:#x})"
    ));
}

/// Recoverable ECAM config read (32-bit). Returns `None` on external abort
/// (unlinked secondary bus). Prefer checking [`port0_link_up`] before scanning
/// bus ≥ 1 — this is a safety net.
pub fn ecam_read32(bus: u8, dev: u8, func: u8, off: u16) -> Option<u32> {
    let ecam = REPORT.with(|r| r.ecam);
    if ecam == 0 {
        return None;
    }
    let addr = ecam
        + ((bus as u64) << 20)
        + ((dev as u64) << 15)
        + ((func as u64) << 12)
        + off as u64;
    super::probe_read32(addr)
}
