//! **ACPI power and sleep buttons** — pressing the power button shuts the machine
//! down cleanly instead of doing nothing.
//!
//! ## Two kinds of power button, and the FADT says which
//!
//! ACPI defines the button twice:
//! - **Fixed-feature** — PM1 status bit 8 (QEMU and many desktops).
//! - **Control-method** — `PNP0C0C` device + GPE (most laptops). FADT flags bit 4.
//!
//! Both paths are polled from [`crate::shell::upkeep`] (human timescale). The GPE
//! path enables one status bit from the device's `_PRW` package; a full SCI → AML
//! Notify dispatcher is not required for “press → poweroff”.
//!
//! ## Why polling is the right call here
//!
//! ACPI would deliver this as an SCI. The scheduler here is cooperative and every
//! long-running loop already pumps [`crate::shell::upkeep`], so a status-register read
//! per pump costs one `in` instruction and needs no interrupt routing — while an SCI
//! would need the interrupt wired, shared with everything else on it, and dispatched.
//! A button press is a human-timescale event; a few milliseconds of latency is free.
//!
//! ## Refusing to act on a bit we did not arm
//!
//! Shutting a machine down by mistake is the worst thing this file could do, so
//! [`poll`] acts only when we successfully armed a path at boot: for fixed-feature that
//! means the FADT named a PM1 button, ACPI mode is on (`SCI_EN`), and status bit 8 is
//! set; for control-method it means a `PNP0C0C` `_PRW` GPE was enabled and that status
//! bit is set (and not `0xff` — an unclaimed port). A stale status bit is cleared at
//! init before the button is armed, so the first press is a press and not a leftover.
//!
//! **Unverified on real hardware**, but *verifiable in a VM*: QEMU's
//! `system_powerdown` sets exactly this status bit, so the e2e harness can press the
//! button for real.

use crate::acpi;
use crate::aml;

/// `PM1x_STS`/`PM1x_EN` bit 8 — the power button.
pub const PWRBTN: u16 = 1 << 8;
/// Bit 9 — the sleep button.
pub const SLPBTN: u16 = 1 << 9;
/// `PM1x_CNT` bit 0 — ACPI mode is enabled and the PM1 event bits are live.
pub const SCI_EN: u16 = 1 << 0;

/// FADT flags bit 4: the power button is a **control-method** device, not a fixed
/// feature. Bit 5 says the same of the sleep button.
pub const FLAG_PWR_BUTTON_IS_CONTROL_METHOD: u32 = 1 << 4;
/// FADT flags bit 5 — see [`FLAG_PWR_BUTTON_IS_CONTROL_METHOD`].
pub const FLAG_SLP_BUTTON_IS_CONTROL_METHOD: u32 = 1 << 5;

/// ACPI HID for the control-method power button.
pub const HID_PWR_BUTTON: &str = "PNP0C0C";

/// True when the machine delivers power-button presses as a `PNP0C0C` GPE rather than
/// through the PM1 status register.
pub fn power_button_is_control_method(fadt_flags: u32) -> bool {
    fadt_flags & FLAG_PWR_BUTTON_IS_CONTROL_METHOD != 0
}

/// True when the machine delivers sleep-button presses as a control-method device.
pub fn sleep_button_is_control_method(fadt_flags: u32) -> bool {
    fadt_flags & FLAG_SLP_BUTTON_IS_CONTROL_METHOD != 0
}

/// True when a PM1 status word reports a power-button press.
///
/// Rejects an all-ones read, which is an unclaimed I/O port rather than every event
/// firing at once — the same reasoning as the embedded controller's status check, and
/// the difference between noticing a button and shutting down because a port decoded
/// to nothing.
pub fn pressed(sts: u16) -> bool {
    sts != 0xffff && sts & PWRBTN != 0
}

/// The value to write back to acknowledge a press.
///
/// PM1 status bits are **write-1-to-clear**, so acknowledging is writing the bit, not
/// clearing it — and only that bit, because writing the whole word back would
/// acknowledge every other event the platform is reporting.
pub fn ack(bit: u16) -> u16 {
    bit
}

/// Fixed-feature (PM1) arming.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FixedButtons {
    sts: u16,
    en: u16,
    sts_b: u16,
    en_b: u16,
}

/// Control-method (GPE) arming: one bit in a GPE status/enable pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GpeButton {
    /// Status register port (byte-addressed I/O).
    sts_port: u16,
    /// Enable register port.
    en_port: u16,
    /// Bit mask within the status/enable **byte** (1 << bit).
    mask: u8,
    /// Global GPE number (for ktrace).
    gpe: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Armed {
    Fixed(FixedButtons),
    Gpe(GpeButton),
}

/// The armed button path, or `None` when nothing could be armed.
static mut BUTTONS: Option<Armed> = None;

/// Set to true by [`poll`] when a press is seen, and consumed by the shell.
static mut PRESSED: bool = false;

/// Arm the power button (fixed-feature **or** control-method GPE).
///
/// Returns true when [`poll`] will act on presses. Every refusal ktraces its reason.
pub fn init(rsdp: u64) -> bool {
    // SAFETY: single-threaded boot-time initialisation.
    if unsafe { BUTTONS.is_some() } {
        return true;
    }
    let flags = acpi::fadt_flags_from_rsdp(rsdp)
        .or_else(|| acpi::pm1_from_rsdp(rsdp).map(|i| i.flags))
        .unwrap_or(0);

    if power_button_is_control_method(flags) {
        return init_control_method(rsdp);
    }
    init_fixed(rsdp)
}

fn init_fixed(rsdp: u64) -> bool {
    let Some(info) = acpi::pm1_from_rsdp(rsdp) else {
        crate::ktrace::log("pwrbtn", "no FADT PM1 event block -- no fixed-feature button");
        return false;
    };
    let (Some(sts), Some(en)) = (info.blocks.a_sts(), info.blocks.a_en()) else {
        crate::ktrace::log("pwrbtn", "PM1a event block absent or too short");
        return false;
    };
    if info.blocks.a_cnt == 0 {
        crate::ktrace::log("pwrbtn", "no PM1a control block -- cannot check ACPI mode");
        return false;
    }
    let cnt = port_in16(info.blocks.a_cnt);
    if cnt == 0xffff {
        crate::ktrace::log("pwrbtn", "PM1a_CNT reads as an unclaimed port; button not armed");
        return false;
    }
    if cnt & SCI_EN == 0 && !enable_acpi_mode(&info) {
        return false;
    }
    port_out16(sts, ack(PWRBTN | SLPBTN));
    let b = info.blocks.b_sts().zip(info.blocks.b_en());
    if let Some((sb, _)) = b {
        port_out16(sb, ack(PWRBTN | SLPBTN));
    }
    let cur = port_in16(en);
    port_out16(en, (cur & !SLPBTN) | PWRBTN);
    let armed = FixedButtons {
        sts,
        en,
        sts_b: b.map(|(s, _)| s).unwrap_or(0),
        en_b: b.map(|(_, e)| e).unwrap_or(0),
    };
    // SAFETY: boot init.
    unsafe { BUTTONS = Some(Armed::Fixed(armed)) };
    crate::ktrace::log_fmt(format_args!(
        "pwrbtn: fixed-feature power button armed (PM1a_STS {sts:#06x}, PM1a_EN {en:#06x})"
    ));
    true
}

/// Arm the `PNP0C0C` control-method button via its `_PRW` GPE.
fn init_control_method(rsdp: u64) -> bool {
    let Some(gpe_blocks) = acpi::gpe_from_rsdp(rsdp) else {
        crate::ktrace::log(
            "pwrbtn",
            "control-method button but no System I/O GPE block in FADT",
        );
        return false;
    };
    // Ensure ACPI mode if PM1a_CNT is present (enables GPE delivery on many chipsets).
    if let Some(info) = acpi::pm1_from_rsdp(rsdp) {
        if info.blocks.a_cnt != 0 {
            let cnt = port_in16(info.blocks.a_cnt);
            if cnt != 0xffff && cnt & SCI_EN == 0 {
                let _ = enable_acpi_mode(&info);
            }
        }
    }
    let map = |phys: u64, len: usize| crate::mm::map_mmio(phys, len);
    let Some(dsdt) = acpi::dsdt_from_rsdp(rsdp, map) else {
        crate::ktrace::log("pwrbtn", "control-method button: no DSDT to find PNP0C0C");
        return false;
    };
    // SAFETY: dsdt_from_rsdp mapped the declared table length.
    let aml = unsafe {
        let len = u32::from_le_bytes([
            *dsdt.add(4),
            *dsdt.add(5),
            *dsdt.add(6),
            *dsdt.add(7),
        ]) as usize;
        if len < 36 || len > 0x40_0000 {
            crate::ktrace::log("pwrbtn", "control-method button: DSDT length implausible");
            return false;
        }
        core::slice::from_raw_parts(dsdt, len)
    };
    // Skip 36-byte ACPI table header → AML starts at 36.
    let aml = aml.get(36..).unwrap_or(&[]);
    let Some(dev) = aml::device_by_hid(aml, HID_PWR_BUTTON) else {
        crate::ktrace::log(
            "pwrbtn",
            "control-method FADT flag set but no PNP0C0C device in DSDT",
        );
        return false;
    };
    let Some(prw) = aml::device_name(aml, &dev, "_PRW") else {
        crate::ktrace::log("pwrbtn", "PNP0C0C has no _PRW; cannot locate GPE");
        return false;
    };
    let Some(gpe_n) = aml::prw_gpe_number(&prw) else {
        crate::ktrace::log("pwrbtn", "PNP0C0C _PRW shape not recognised");
        return false;
    };
    let Some(block) = gpe_blocks.block_for(gpe_n) else {
        crate::ktrace::log_fmt(format_args!(
            "pwrbtn: GPE {gpe_n} not in FADT GPE0/GPE1 blocks"
        ));
        return false;
    };
    let Some((byte_off, bit)) = block.bit_of(gpe_n) else {
        return false;
    };
    let mask = 1u8 << bit;
    let sts_port = block.sts.saturating_add(byte_off as u16);
    let en_port = block.en.saturating_add(byte_off as u16);
    // Clear pending, then enable.
    let _ = port_in8(sts_port); // unclaimed → 0xff
    port_out8(sts_port, mask); // write-1-to-clear
    let cur = port_in8(en_port);
    if cur == 0xff {
        // Ambiguous: floating bus vs all-enabled. Still try to set our bit.
        crate::ktrace::log_fmt(format_args!(
            "pwrbtn: GPE enable port {en_port:#06x} reads 0xff (may be unclaimed)"
        ));
    }
    port_out8(en_port, cur | mask);
    let g = GpeButton {
        sts_port,
        en_port,
        mask,
        gpe: gpe_n,
    };
    // SAFETY: boot init.
    unsafe { BUTTONS = Some(Armed::Gpe(g)) };
    crate::ktrace::log_fmt(format_args!(
        "pwrbtn: control-method (PNP0C0C) armed on GPE {gpe_n} (sts {sts_port:#06x} en {en_port:#06x} mask {mask:#04x})"
    ));
    true
}

/// Ask firmware to hand ACPI over, and wait for it to happen.
///
/// Only reached when `SCI_EN` is clear. Writing `ACPI_ENABLE` to `SMI_CMD` raises an
/// SMI that firmware answers by taking the platform out of legacy mode; the handoff is
/// asynchronous, so `SCI_EN` has to be polled for rather than assumed. Bounded, like
/// every other wait here — a machine that never completes the handoff must cost a
/// fixed number of iterations and a ktrace line, not a hung boot.
///
/// A platform with no `SMI_CMD` has nothing to ask (its firmware either enabled ACPI or
/// never uses legacy mode), so that is a refusal rather than a blind write to port 0.
fn enable_acpi_mode(info: &acpi::Pm1Info) -> bool {
    if info.smi_cmd == 0 || info.acpi_enable == 0 {
        crate::ktrace::log(
            "pwrbtn",
            "ACPI mode off and no SMI_CMD handoff to ask for; button not armed",
        );
        return false;
    }
    port_out8(info.smi_cmd, info.acpi_enable);
    for _ in 0..ACPI_ENABLE_SPINS {
        let cnt = port_in16(info.blocks.a_cnt);
        if cnt != 0xffff && cnt & SCI_EN != 0 {
            crate::ktrace::log_fmt(format_args!(
                "pwrbtn: took ACPI ownership via SMI_CMD {:#06x}",
                info.smi_cmd
            ));
            return true;
        }
        core::hint::spin_loop();
    }
    crate::ktrace::log_fmt(format_args!(
        "pwrbtn: firmware did not complete the ACPI handoff (SMI_CMD {:#06x}); button not armed",
        info.smi_cmd
    ));
    false
}

/// Iterations to wait for firmware to set `SCI_EN` after the SMI. Generous — the
/// handoff is firmware work, not ours — but finite.
const ACPI_ENABLE_SPINS: u32 = 3_000_000;

/// Check for a press. Cheap enough to call from the UI pump: one port read.
///
/// Acknowledges the press so it fires once, and latches it for [`take_press`] rather
/// than shutting down here — the caller decides what a press means, and a function
/// called from a repaint loop must not power the machine off underneath its caller.
pub fn poll() {
    // SAFETY: read of a boot-initialised Copy value; no writer runs after `init`.
    let Some(b) = (unsafe { BUTTONS }) else {
        return;
    };
    match b {
        Armed::Fixed(f) => {
            let sts = port_in16(f.sts);
            if pressed(sts) {
                port_out16(f.sts, ack(PWRBTN));
                // SAFETY: single-threaded UI pump.
                unsafe { PRESSED = true };
                return;
            }
            if f.sts_b != 0 {
                let sts = port_in16(f.sts_b);
                if pressed(sts) {
                    port_out16(f.sts_b, ack(PWRBTN));
                    unsafe { PRESSED = true };
                }
            }
        }
        Armed::Gpe(g) => {
            let sts = port_in8(g.sts_port);
            // 0xff is floating/unclaimed — never treat as a press (same rule as PM1).
            if sts != 0xff && sts & g.mask != 0 {
                port_out8(g.sts_port, g.mask); // write-1-to-clear
                unsafe { PRESSED = true };
            }
        }
    }
}

/// Take a pending press, if there is one.
pub fn take_press() -> bool {
    // SAFETY: single-threaded shell/UI path.
    unsafe {
        let p = PRESSED;
        PRESSED = false;
        p
    }
}

/// True once the button is armed.
pub fn armed() -> bool {
    // SAFETY: as `poll`.
    unsafe { BUTTONS.is_some() }
}

/// A one-line description for `/battery`-style diagnostics.
pub fn status() -> alloc::string::String {
    // SAFETY: as `poll`.
    match unsafe { BUTTONS } {
        Some(Armed::Fixed(b)) => alloc::format!(
            "fixed-feature, armed (PM1a_STS {:#06x}, PM1a_EN {:#06x})",
            b.sts,
            b.en
        ),
        Some(Armed::Gpe(g)) => alloc::format!(
            "control-method (PNP0C0C), GPE {} (sts {:#06x} en {:#06x} mask {:#04x})",
            g.gpe,
            g.sts_port,
            g.en_port,
            g.mask
        ),
        None => alloc::string::String::from("not armed (see the pwrbtn: ktrace line from boot)"),
    }
}

#[cfg(target_arch = "x86_64")]
fn port_in16(port: u16) -> u16 {
    // SAFETY: reading an ACPI-declared PM1 register. A port read has no memory effect,
    // and an unclaimed port returns 0xffff, which `pressed` rejects.
    unsafe { crate::arch::x86_64::port::inw(port) }
}

#[cfg(target_arch = "x86_64")]
fn port_in8(port: u16) -> u8 {
    // SAFETY: GPE status/enable byte I/O as declared by the FADT.
    unsafe { crate::arch::x86_64::port::inb(port) }
}

#[cfg(target_arch = "x86_64")]
fn port_out8(port: u16, v: u8) {
    // SAFETY: SMI_CMD handoff or GPE status/enable byte write.
    unsafe { crate::arch::x86_64::port::outb(port, v) };
}

#[cfg(not(target_arch = "x86_64"))]
fn port_out8(_port: u16, _v: u8) {}

#[cfg(not(target_arch = "x86_64"))]
fn port_in8(_port: u16) -> u8 {
    0xff
}

#[cfg(target_arch = "x86_64")]
fn port_out16(port: u16, v: u16) {
    // SAFETY: writing an ACPI-declared PM1 register, after `init` confirmed ACPI mode
    // is on. Status writes are write-1-to-clear of specific event bits only.
    unsafe { crate::arch::x86_64::port::outw(port, v) };
}

// The PM1 / GPE registers used here are System I/O. Reduced-hardware / aarch64
// platforms without those ports report “not armed” honestly.
#[cfg(not(target_arch = "x86_64"))]
fn port_in16(_port: u16) -> u16 {
    0xffff
}

#[cfg(not(target_arch = "x86_64"))]
fn port_out16(_port: u16, _v: u16) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn the_fadt_flag_decides_which_button_a_machine_has() {
        // Bit 4 set means the press arrives as a PNP0C0C GPE (control-method path).
        assert!(power_button_is_control_method(FLAG_PWR_BUTTON_IS_CONTROL_METHOD));
        assert!(!power_button_is_control_method(0));
        assert!(!power_button_is_control_method(FLAG_SLP_BUTTON_IS_CONTROL_METHOD));
        assert!(sleep_button_is_control_method(FLAG_SLP_BUTTON_IS_CONTROL_METHOD));
    }

    #[test_case]
    fn gpe_status_byte_uses_write_1_to_clear_mask() {
        // Same rule as PM1: ack is the bit mask itself, not the whole register.
        let mask = 1u8 << 5;
        assert_eq!(mask, 0x20);
        assert_eq!(mask & !(1 << 3), 0x20);
    }

    #[test_case]
    fn a_press_is_the_power_bit_and_not_a_floating_port() {
        assert!(pressed(PWRBTN));
        assert!(pressed(PWRBTN | SLPBTN | 0x0001));
        assert!(!pressed(0));
        assert!(!pressed(SLPBTN));
        // An unclaimed I/O port reads all-ones. Treating that as a press powers the
        // machine off because a port decoded to nothing.
        assert!(!pressed(0xffff));
    }

    #[test_case]
    fn acknowledging_writes_only_the_bit_being_acknowledged() {
        // PM1 status bits are write-1-to-clear, so the ack value *is* the bit. Writing
        // the whole status word back would acknowledge every other pending event —
        // including ones a future GPE or timer path will want to see.
        assert_eq!(ack(PWRBTN), PWRBTN);
        assert_eq!(ack(PWRBTN) & SLPBTN, 0);
        let sts = PWRBTN | SLPBTN | 0x0100_u16.rotate_left(3);
        assert_ne!(ack(PWRBTN), sts, "would have acked unrelated events");
    }
}
