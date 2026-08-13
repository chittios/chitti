//! **The Apple SoC watchdog**, whose only job here is to be switched off.
//!
//! A watchdog exists to reset a machine whose software has stopped responding.
//! That is exactly the wrong behaviour while bringing one up: the kernel stops
//! with a syndrome on the screen and the serial console, and a few seconds later
//! the SoC resets — so the symptom a person reports is "it rebooted", the message
//! is gone, and the two very different failures *halted with a diagnosis* and
//! *hung with no output* become indistinguishable.
//!
//! m1n1 disables it in its own `main()` (`src/wdt.c`), so on a clean tethered
//! boot this is a no-op. It is done again here for the cases where that is not
//! true and cannot be checked from the guest: a chainloaded or hypervisor m1n1,
//! a payload booted some other way, or firmware that re-armed it. Writing 0 to a
//! disabled control register costs nothing and removes a variable from every
//! future "why did it reboot" question.
//!
//! Registers are m1n1's (`WDT_CTL` at `+0x1c`), and the base comes from the
//! device tree's `apple,wdt` node — never a constant, per the standing rule. The
//! second watchdog some SoC revisions carry is deliberately **not** touched: its
//! offset comes from `reg[2]`, which the FDT node does not publish (m1n1 reads it
//! from the ADT), and writing a guessed address on a machine like this is how you
//! turn a debugging session into a bricked boot.

use core::sync::atomic::{AtomicBool, Ordering};

/// Control register offset. Writing 0 stops the counter (m1n1 `wdt_disable`).
const WDT_CTL: usize = 0x1c;

/// Set once the write has been made, so `/agx`-style repeat calls and the resume
/// path do not re-walk the tree.
static DISABLED: AtomicBool = AtomicBool::new(false);

/// Turn the watchdog off if this machine has one. Safe to call more than once,
/// and a no-op on anything without an `apple,wdt` node (QEMU, VirtualBox, UEFI).
///
/// Returns the base it wrote to, for the boot log — "no watchdog node" and
/// "disabled the watchdog at 0x…" are different facts and the first is worth
/// seeing on a machine that keeps rebooting.
pub fn disable() -> Option<u64> {
    if DISABLED.load(Ordering::Relaxed) {
        return None;
    }
    let fdt = super::boot::boot_x0();
    // SAFETY: `boot_x0` is the FDT pointer (or not an FDT, rejected by magic).
    let (base, size) = unsafe { crate::fdt::reg_of_compatible(fdt, b"apple,wdt") }?;
    if size <= WDT_CTL as u64 {
        crate::ktrace::log_fmt(format_args!(
            "wdt: node at {base:#x} declares only {size:#x} bytes -- refusing to write past it"
        ));
        return None;
    }
    let va = crate::mm::map_mmio(base, size as usize);
    // SAFETY: `va` maps the watchdog's own register window, sized from its `reg`.
    // A single 32-bit store, the MMIO rule for this arch.
    unsafe {
        core::arch::asm!(
            "str w1, [x0]",
            in("x0") va + WDT_CTL as u64,
            in("w1") 0u32,
            options(nostack, preserves_flags),
        );
    }
    DISABLED.store(true, Ordering::Relaxed);
    Some(base)
}
