//! Apple-Silicon USB **host** bring-up (DART + ATC-PHY + DWC3 → xHCI HID), for
//! USB2 keyboard/mouse when booted via m1n1. This wires together, from the boot
//! device tree:
//!
//! * the **DWC3** controller (`apple,t8112-dwc3`, `usb@382280000`) — its
//!   xHCI-compatible register window is at DWC3_base + 0x0 once the core is in
//!   HOST mode;
//! * the **ATC PHY** (`apple,t8112-atcphy`, `phy@383000000`) `core` +
//!   `pipehandler` register blocks;
//! * the **DART** the controller DMAs through (via the DWC3 node's
//!   `iommus = <&dart SID>`), put into bypass so buffer physical addresses pass
//!   through (ChittiOS runs an identity map).
//!
//! The power/PHY register sequences are ported from m1n1's `src/usb.c`
//! (`usb_phy_bringup`) and `src/usb_dwc3.c` (core/PHY soft-reset). m1n1 tears
//! USB down at handoff and leaves the power domains enabled, so we replay the
//! PHY setup and re-init the controller from scratch — but as an **xHCI host**
//! (`GCTL.PRTCAPDIR = HOST`), which m1n1 never does (it runs a device/gadget).
//!
//! ## Honest status (untested on hardware)
//! m1n1's PHY sequence is the **USB2 *dummy* PHY** it uses as a gadget, where
//! the *host cable* supplies Vbus/CC. As a **host** driving a *downstream*
//! device, the real USB2/Type-C path likely needs (a) a host-mode PHY/mux
//! setting instead of `PIPEHANDLER_MUX_CTRL_DUMMY`, and (b) Type-C orientation +
//! port power via the **TPS6598x PD controller over I²C** — neither of which
//! m1n1 does for the host role, so there is no source to port them from yet.
//! This module therefore brings the stack up as far as m1n1's sources allow and
//! attaches the xHCI core; first-boot enumeration of a plugged-in keyboard is
//! **not** expected to work until the host-PHY + PD-controller pieces land. See
//! the hardware-verification notes. Everything here is gated on `is_apple()` and
//! degrades gracefully (logs + returns false), so it never regresses QEMU.

use core::ptr::{read_volatile, write_volatile};

// --- DWC3 global/device registers (usb_dwc3_regs.h) ----------------------
const DWC3_GSNPSID: usize = 0xc120;
const DWC3_GCTL: usize = 0xc110;
const GCTL_CORESOFTRESET: u32 = 1 << 11;
const GCTL_DISSCRAMBLE: u32 = 1 << 3;
const GCTL_SCALEDOWN_MASK: u32 = 0b11 << 4;
const GCTL_PRTCAPDIR_SHIFT: u32 = 12; // bits[13:12]
const GCTL_PRTCAP_HOST: u32 = 1; // (DEVICE=2, OTG=3)
const DWC3_GSTS: usize = 0xc118;
const DWC3_GUSB2PHYCFG0: usize = 0xc200;
const DWC3_GUSB3PIPECTL0: usize = 0xc2c0;
const PHYCFG_PHYSOFTRST: u32 = 1 << 31;
const DWC3_DCTL: usize = 0xc704;
const DCTL_CSFTRST: u32 = 1 << 30;

// --- pipehandler glue (m1n1 usb.c PIPEHANDLER_*) -------------------------
const PIPEHANDLER_MUX_CTRL: usize = 0x0c;
#[allow(dead_code)]
const PIPEHANDLER_MUX_CTRL_DUMMY: u32 = 0x22; // gadget dummy PHY (m1n1 proxy path)
const PIPEHANDLER_MUX_CTRL_USB3: u32 = 0x08; // USB3 host mux (m1n1 usb.c) — for a downstream device
const PIPEHANDLER_AON_GEN: usize = 0x1c;
const PIPEHANDLER_AON_GEN_DWC3_RESET_N: u32 = 1 << 0;
const PIPEHANDLER_NONSELECTED_OVERRIDE: usize = 0x20;
const PIPEHANDLER_NONSELECTED_VALUE: u32 = 0x9332;

/// Discovered USB register bases + the DART stream, from the FDT.
struct UsbHw {
    dwc3: usize,        // DWC3 core (xHCI window at +0x0 in host mode)
    atc: usize,         // ATC PHY "core"
    pipehandler: usize, // ATC PHY "pipehandler"
    dart_base: usize,   // the DART the controller DMAs through
    dart_sid: u32,      // its stream id (from `iommus`)
}

#[inline]
fn r32(base: usize, off: usize) -> u32 {
    // SAFETY: single 32-bit MMIO read of a Device-mapped USB register.
    unsafe { read_volatile((base + off) as *const u32) }
}
#[inline]
fn w32(base: usize, off: usize, v: u32) {
    // SAFETY: single 32-bit MMIO write of a Device-mapped USB register.
    unsafe { write_volatile((base + off) as *mut u32, v) }
}

/// Busy-wait `ms` milliseconds off the generic timer (valid on Apple; the
/// scheduler is cooperative during boot bring-up).
fn mdelay(ms: u64) {
    let end = crate::arch::now_ms() + ms;
    while crate::arch::now_ms() < end {
        core::hint::spin_loop();
    }
}

/// True if `needle` appears in the FDT `/chosen` `bootargs` (the m1n1 `-b`
/// kernel command line). Used to opt into USB (`chitti.usb`).
fn bootarg_present(needle: &[u8]) -> bool {
    let fdt = super::boot::boot_x0();
    // SAFETY: boot_x0 is the FDT pointer (or non-FDT, rejected by the magic).
    if let Some(c) = unsafe { crate::fdt::chosen(fdt) } {
        if !c.bootargs_ptr.is_null() && c.bootargs_len >= needle.len() {
            // SAFETY: `[bootargs_ptr, +len)` is a view into the still-mapped FDT.
            let s = unsafe { core::slice::from_raw_parts(c.bootargs_ptr, c.bootargs_len) };
            return s.windows(needle.len()).any(|w| w == needle);
        }
    }
    false
}

/// Discover the USB hardware from the boot FDT. `None` if the machine doesn't
/// describe a `apple,*-dwc3` (e.g. QEMU) — a clean skip.
fn discover() -> Option<UsbHw> {
    let fdt = super::boot::boot_x0();
    // DWC3 core = reg[0] of the dwc3 node; ATC core = reg[0], pipehandler =
    // reg[4] ("pipehandler") of the atcphy node.
    // SAFETY: boot_x0 is the FDT pointer (or non-FDT, rejected by the magic).
    let (dwc3, _) = unsafe { crate::fdt::reg_of_compatible(fdt, b"apple,t8112-dwc3") }
        .or_else(|| unsafe { crate::fdt::reg_of_compatible(fdt, b"apple,t8103-dwc3") })?;
    let (atc, _) = unsafe { crate::fdt::reg_of_compatible(fdt, b"apple,t8112-atcphy") }
        .or_else(|| unsafe { crate::fdt::reg_of_compatible(fdt, b"apple,t8103-atcphy") })?;
    let (pipehandler, _) = unsafe { crate::fdt::reg_nth_of_compatible(fdt, b"apple,t8112-atcphy", 4) }
        .or_else(|| unsafe { crate::fdt::reg_nth_of_compatible(fdt, b"apple,t8103-atcphy", 4) })?;
    // DART: the dwc3's `iommus = <&dart0 SID0 &dartUSB SID1>`. m1n1 uses the
    // second (the USB DMA DART at 0x…f80000); take the last pair.
    let mut cells = [0u32; 8];
    let n = unsafe { crate::fdt::prop_cells_of_compatible(fdt, b"apple,t8112-dwc3", b"iommus", &mut cells) }
        .max(unsafe { crate::fdt::prop_cells_of_compatible(fdt, b"apple,t8103-dwc3", b"iommus", &mut cells) });
    let (dart_base, dart_sid) = if n >= 2 {
        let phandle = cells[n - 2];
        let sid = cells[n - 1];
        let (base, _) = unsafe { crate::fdt::reg_by_phandle(fdt, phandle) }?;
        (base as usize, sid)
    } else {
        return None;
    };
    Some(UsbHw { dwc3: dwc3 as usize, atc: atc as usize, pipehandler: pipehandler as usize, dart_base, dart_sid })
}

/// Replay m1n1's USB2 PHY + pipehandler bring-up (`usb.c:usb_phy_bringup`). The
/// power domains are already enabled (m1n1 leaves them on). See the status note:
/// this is the *dummy* PHY sequence.
fn phy_bringup(hw: &UsbHw) {
    // ATC "core" magic sequence (m1n1 usb.c).
    w32(hw.atc, 0x08, 0x01c1000f);
    w32(hw.atc, 0x04, 0x00000003);
    w32(hw.atc, 0x04, 0x00000000);
    w32(hw.atc, 0x1c, 0x008c0813);
    w32(hw.atc, 0x00, 0x00000002);
    // Pipehandler: select the USB3 **host** mux (m1n1 uses DUMMY for its gadget
    // proxy; a downstream device needs the real path), release the DWC3 reset,
    // set override. NB: full host enumeration also needs the ATC-PHY tunable
    // power-on (Asahi atc.c) + cd321x PD orientation/Vbus — see the status note.
    w32(hw.pipehandler, PIPEHANDLER_MUX_CTRL, PIPEHANDLER_MUX_CTRL_USB3);
    w32(hw.pipehandler, PIPEHANDLER_AON_GEN, PIPEHANDLER_AON_GEN_DWC3_RESET_N);
    w32(hw.pipehandler, PIPEHANDLER_NONSELECTED_OVERRIDE, PIPEHANDLER_NONSELECTED_VALUE);
    mdelay(10);
}

/// Reset the DWC3 core + PHYs and put it in **HOST** mode (`usb_dwc3.c` reset
/// sequence, but PRTCAPDIR=HOST instead of DEVICE). Returns false on a bad ID.
fn dwc3_reset_host(hw: &UsbHw) -> bool {
    let id = r32(hw.dwc3, DWC3_GSNPSID);
    if id & 0xffff_0000 != 0x3331_0000 {
        crate::ktrace::log_fmt(format_args!("apple_usb: unexpected DWC3 GSNPSID {id:#x}"));
        return false;
    }
    // Device soft reset (as m1n1 does), then core + PHY soft reset dance.
    w32(hw.dwc3, DWC3_DCTL, r32(hw.dwc3, DWC3_DCTL) | DCTL_CSFTRST);
    let mut spins = 0;
    while r32(hw.dwc3, DWC3_DCTL) & DCTL_CSFTRST != 0 && spins < 100_000 {
        spins += 1;
    }
    w32(hw.dwc3, DWC3_GCTL, r32(hw.dwc3, DWC3_GCTL) | GCTL_CORESOFTRESET);
    w32(hw.dwc3, DWC3_GUSB3PIPECTL0, r32(hw.dwc3, DWC3_GUSB3PIPECTL0) | PHYCFG_PHYSOFTRST);
    w32(hw.dwc3, DWC3_GUSB2PHYCFG0, r32(hw.dwc3, DWC3_GUSB2PHYCFG0) | PHYCFG_PHYSOFTRST);
    mdelay(100);
    w32(hw.dwc3, DWC3_GUSB3PIPECTL0, r32(hw.dwc3, DWC3_GUSB3PIPECTL0) & !PHYCFG_PHYSOFTRST);
    w32(hw.dwc3, DWC3_GUSB2PHYCFG0, r32(hw.dwc3, DWC3_GUSB2PHYCFG0) & !PHYCFG_PHYSOFTRST);
    mdelay(100);
    w32(hw.dwc3, DWC3_GCTL, r32(hw.dwc3, DWC3_GCTL) & !GCTL_CORESOFTRESET);
    mdelay(100);
    // Clear scaledown + descramble, select HOST role.
    let mut gctl = r32(hw.dwc3, DWC3_GCTL);
    gctl &= !(GCTL_SCALEDOWN_MASK | GCTL_DISSCRAMBLE);
    gctl &= !(0b11 << GCTL_PRTCAPDIR_SHIFT);
    gctl |= GCTL_PRTCAP_HOST << GCTL_PRTCAPDIR_SHIFT;
    w32(hw.dwc3, DWC3_GCTL, gctl);
    mdelay(10);
    true
}

/// Dump the DWC3 global + xHCI capability/operational registers to the chat pane
/// (visible on a bare boot). Read-only observation to understand why the xHCI
/// controller won't reach ready. xHCI caps are at `dwc3 + 0`; the op regs at
/// `dwc3 + CAPLENGTH`. GCTL bit[13:12] = PRTCAPDIR (1=host). GSTS[1:0] = current
/// mode. xHCI HCSPARAMS1: slots[7:0], ports[31:24]. USBSTS bit11 = CNR.
fn dump_state(hw: &UsbHw) {
    let gsnpsid = r32(hw.dwc3, DWC3_GSNPSID);
    let gctl = r32(hw.dwc3, DWC3_GCTL);
    let gsts = r32(hw.dwc3, DWC3_GSTS);
    let u2 = r32(hw.dwc3, DWC3_GUSB2PHYCFG0);
    let u3 = r32(hw.dwc3, DWC3_GUSB3PIPECTL0);
    crate::serial_println!(
        "apple_usb: DWC3 id={gsnpsid:#x} gctl={gctl:#x} (prtcap={}) gsts={gsts:#x} u2phy={u2:#x} u3pipe={u3:#x}",
        (gctl >> 12) & 0x3
    );
    let caplen = r32(hw.dwc3, 0) & 0xff;
    let hcs1 = r32(hw.dwc3, 0x04);
    let hcc1 = r32(hw.dwc3, 0x10);
    let op = hw.dwc3 + caplen as usize;
    let usbsts = r32(op, 0x04);
    let usbcmd = r32(op, 0x00);
    crate::serial_println!(
        "apple_usb: xHCI caplen={caplen:#x} slots={} ports={} hcc1={hcc1:#x} usbcmd={usbcmd:#x} usbsts={usbsts:#x} (CNR={})",
        hcs1 & 0xff,
        (hcs1 >> 24) & 0xff,
        (usbsts >> 11) & 1
    );
}

/// Bring up USB HID (keyboard/mouse) on Apple Silicon. No-op (returns false) off
/// Apple or when the FDT has no dwc3. Best-effort + graceful: any failure logs
/// and returns false, never faults. See the module status note on why
/// enumeration may not yet succeed.
pub fn init() -> bool {
    if !super::is_apple() {
        return false;
    }
    // USB is OFF by default and must be opted into with the `chitti.usb` bootarg.
    // Rationale (learned on hardware): the dwc3/DART/ATC MMIO we drive is the
    // SAME USB controller the m1n1 **hypervisor** uses for the proxy console, so
    // resetting it under the hv debug path (CHITTI_M1N1_HV=1) kills the console
    // (proxy "Device not configured"). Enable it only on a **bare** boot for
    // hardware testing: `make m1n1 CHITTI_BOOTARGS="chitti.usb"` (no hv).
    if !bootarg_present(b"chitti.usb") {
        crate::ktrace::log("apple_usb", "USB HID gated (add `chitti.usb` bootarg on a BARE boot to enable; never under the hv)");
        return false;
    }
    let Some(hw) = discover() else {
        crate::serial_println!("apple_usb: no apple,dwc3 in device tree; skipping");
        return false;
    };
    // Step markers go to serial_println! (the chat pane, visible on a bare boot)
    // rather than ktrace (the action/logs pane, closed at boot). Each prints
    // BEFORE the MMIO group that follows, so if a group hangs (e.g. an ungated
    // power domain — touching ungated Apple MMIO stalls the interconnect), the
    // last visible line names the culprit.
    crate::serial_println!(
        "apple_usb: dwc3={:#x} atc={:#x} pipe={:#x} dart={:#x} sid={}",
        hw.dwc3, hw.atc, hw.pipehandler, hw.dart_base, hw.dart_sid
    );
    // Ensure the USB MMIO windows + the DART are mapped (Device).
    crate::serial_println!("apple_usb: [1] mapping MMIO windows");
    super::mmu::map_device_gib(hw.dwc3 as u64);
    super::mmu::map_device_gib(hw.atc as u64);
    super::mmu::map_device_gib(hw.pipehandler as u64);
    super::mmu::map_device_gib(hw.dart_base as u64);
    // Put the controller's DART stream in bypass so DMA uses physical addresses
    // (ChittiOS is identity-mapped, so buffer PA == VA).
    // SAFETY: `dart_base` is the Device-mapped DART; `dart_sid` from the FDT.
    crate::serial_println!("apple_usb: [2] DART bypass (first DART MMIO)");
    let dart = unsafe { super::dart::Dart::new(hw.dart_base, hw.dart_sid) };
    dart.set_bypass();
    // PHY + controller.
    crate::serial_println!("apple_usb: [3] ATC PHY bring-up (first ATC/pipehandler MMIO)");
    phy_bringup(&hw);
    crate::serial_println!("apple_usb: [4] DWC3 host reset (first dwc3 MMIO: GSNPSID)");
    if !dwc3_reset_host(&hw) {
        crate::serial_println!("apple_usb: DWC3 reset failed");
        return false;
    }
    // Observe the controller state before handing off to the xHCI core.
    dump_state(&hw);
    // Drive the xHCI window (DWC3 base + 0x0) with the shared xHCI core.
    crate::serial_println!("apple_usb: [5] xHCI attach at dwc3 window");
    let ok = super::xhci::attach_at(hw.dwc3);
    crate::serial_println!("apple_usb: [6] done (hid up: {ok})");
    ok
}
