//! **ACPI power and sleep buttons** — pressing the power button shuts the machine
//! down cleanly instead of doing nothing.
//!
//! On a laptop or desktop this is the one control a user reaches for when they want the
//! machine off, and until now it did nothing at all: ChittiOS could *perform* an S5
//! transition (`/poweroff`) but never noticed being asked for one.
//!
//! ## Two kinds of power button, and the FADT says which
//!
//! ACPI defines the button twice. The **fixed-feature** button is a bit in the PM1
//! status register, which is what this drives. The **control-method** button is a
//! `PNP0C0C` device delivering a GPE, which needs the general-purpose event machinery
//! and is *not* implemented — but it is also *not guessed at*: FADT flags bit 4 says
//! which form the machine uses, so a control-method machine is reported as such rather
//! than silently polling a register that will never change.
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
//! [`poll`] acts only when *all* of: the FADT described a fixed-feature button, we
//! successfully enabled it, ACPI mode is on (`SCI_EN`), and the status bit is set. A
//! stale status bit from firmware is cleared at init before the button is armed, so the
//! first press is a press and not a leftover.
//!
//! **Unverified on real hardware**, but *verifiable in a VM*: QEMU's
//! `system_powerdown` sets exactly this status bit, so the e2e harness can press the
//! button for real.

use crate::acpi;

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

/// What the machine actually has, decided once at init.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Buttons {
    sts: u16,
    en: u16,
    /// Second PM1 block, when the platform splits its events across two.
    sts_b: u16,
    en_b: u16,
}

/// The armed buttons, or `None` when this machine has no fixed-feature button.
static mut BUTTONS: Option<Buttons> = None;

/// Set to true by [`poll`] when a press is seen, and consumed by the shell.
static mut PRESSED: bool = false;

/// Arm the fixed-feature power button.
///
/// Returns true when the button is armed and [`poll`] will act on it. Every refusal
/// ktraces its reason, because "the power button does nothing" is otherwise
/// indistinguishable between a control-method machine, a memory-mapped PM block and
/// ACPI mode being off.
pub fn init(rsdp: u64) -> bool {
    // SAFETY: single-threaded boot-time initialisation.
    if unsafe { BUTTONS.is_some() } {
        return true;
    }
    let Some(info) = acpi::pm1_from_rsdp(rsdp) else {
        crate::ktrace::log("pwrbtn", "no FADT PM1 event block -- no fixed-feature button");
        return false;
    };
    if power_button_is_control_method(info.flags) {
        // A real laptop often does this. Say so rather than polling a dead bit.
        crate::ktrace::log(
            "pwrbtn",
            "firmware uses a control-method (PNP0C0C) power button; GPE dispatch is not implemented",
        );
        return false;
    }
    let (Some(sts), Some(en)) = (info.blocks.a_sts(), info.blocks.a_en()) else {
        crate::ktrace::log("pwrbtn", "PM1a event block absent or too short");
        return false;
    };
    if info.blocks.a_cnt == 0 {
        crate::ktrace::log("pwrbtn", "no PM1a control block -- cannot check ACPI mode");
        return false;
    }
    // ACPI mode has to be on for the PM1 event bits to mean anything. A UEFI machine
    // boots in ACPI mode already; a legacy-BIOS one starts in SMM's hands and has to be
    // asked, which is what SMI_CMD/ACPI_ENABLE is for.
    let cnt = port_in16(info.blocks.a_cnt);
    if cnt == 0xffff {
        crate::ktrace::log("pwrbtn", "PM1a_CNT reads as an unclaimed port; button not armed");
        return false;
    }
    if cnt & SCI_EN == 0 && !enable_acpi_mode(&info) {
        return false;
    }
    // Clear anything firmware left pending *before* arming, so the first press is a
    // press. Write-1-to-clear, and only the button bits.
    port_out16(sts, ack(PWRBTN | SLPBTN));
    let b = info.blocks.b_sts().zip(info.blocks.b_en());
    if let Some((sb, _)) = b {
        port_out16(sb, ack(PWRBTN | SLPBTN));
    }
    // Arm the power button. The sleep button is deliberately left disabled: there is no
    // S3 implementation to enter, and a button that half-works is worse than one that
    // does nothing.
    let cur = port_in16(en);
    port_out16(en, (cur & !SLPBTN) | PWRBTN);
    let armed = Buttons {
        sts,
        en,
        sts_b: b.map(|(s, _)| s).unwrap_or(0),
        en_b: b.map(|(_, e)| e).unwrap_or(0),
    };
    // SAFETY: as above.
    unsafe { BUTTONS = Some(armed) };
    crate::ktrace::log_fmt(format_args!(
        "pwrbtn: fixed-feature power button armed (PM1a_STS {sts:#06x}, PM1a_EN {en:#06x})"
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
    let sts = port_in16(b.sts);
    if pressed(sts) {
        port_out16(b.sts, ack(PWRBTN));
        // SAFETY: single-threaded UI pump.
        unsafe { PRESSED = true };
        return;
    }
    if b.sts_b != 0 {
        let sts = port_in16(b.sts_b);
        if pressed(sts) {
            port_out16(b.sts_b, ack(PWRBTN));
            // SAFETY: as above.
            unsafe { PRESSED = true };
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
        Some(b) => alloc::format!(
            "fixed-feature, armed (PM1a_STS {:#06x}, PM1a_EN {:#06x})",
            b.sts,
            b.en
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
fn port_out8(port: u16, v: u8) {
    // SAFETY: writing the FADT-declared SMI_CMD port with the FADT-declared
    // ACPI_ENABLE value, only when ACPI mode is off. This is the handoff the
    // specification defines for exactly this situation.
    unsafe { crate::arch::x86_64::port::outb(port, v) };
}

#[cfg(not(target_arch = "x86_64"))]
fn port_out8(_port: u16, _v: u8) {}

#[cfg(target_arch = "x86_64")]
fn port_out16(port: u16, v: u16) {
    // SAFETY: writing an ACPI-declared PM1 register, after `init` confirmed ACPI mode
    // is on. Status writes are write-1-to-clear of specific event bits only.
    unsafe { crate::arch::x86_64::port::outw(port, v) };
}

// The PM1 registers are I/O ports by definition — ACPI's reduced-hardware profile,
// which is what an ARM platform uses, has no fixed-feature registers at all and
// delivers its buttons as GPIO/GPE instead. So there is nothing to port here: `init`
// finds no PM1 block and reports it, which is the truth on such a machine rather than
// a dropped feature.
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
        // Bit 4 set means the press arrives as a PNP0C0C GPE, which is not implemented.
        // Polling PM1 on such a machine watches a bit that never changes.
        assert!(power_button_is_control_method(FLAG_PWR_BUTTON_IS_CONTROL_METHOD));
        assert!(!power_button_is_control_method(0));
        assert!(!power_button_is_control_method(FLAG_SLP_BUTTON_IS_CONTROL_METHOD));
        assert!(sleep_button_is_control_method(FLAG_SLP_BUTTON_IS_CONTROL_METHOD));
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
