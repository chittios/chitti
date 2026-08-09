//! Terminating the guest with a pass/fail verdict the host side (`xtask`) can
//! read — used by the in-kernel test harness and by the `refcheck` gate.
//!
//! The two arches cannot report a status the same way, so this is the facade and
//! each has its own implementation behind it:
//!
//! * **x86_64** writes QEMU's `isa-debug-exit` device, which the host turns into
//!   the process exit code `(value << 1) | 1`. A real exit status, so the runner
//!   never has to parse output.
//! * **aarch64** has no such device — `-M virt` exposes no I/O ports at all —
//!   and PSCI `SYSTEM_OFF` carries nothing back: QEMU exits 0 whether the suite
//!   passed or failed. So the verdict goes out over the serial console as a
//!   sentinel line and `xtask` scans for it, the same arrangement
//!   `cortex::run_acceptance` and `xtask ref-check` already use.
//!
//! Two rules for the aarch64 path, because both failure modes are silent:
//!
//! * The sentinel is printed **before** the poweroff. Nothing runs after `hvc`,
//!   so a verdict emitted afterwards is a verdict that never existed.
//! * A missing sentinel must count as failure, never as success. A guest that
//!   data-aborts, or hangs, or resets, prints neither line — so the runner
//!   requires [`PASS_SENTINEL`] rather than treating its absence as "no news".

/// Printed by [`exit_qemu`] on aarch64 when the suite passed. `xtask`'s runner
/// requires this exact line; changing it means changing `xtask/src/main.rs` too.
pub const PASS_SENTINEL: &str = "CHITTI-TEST: ALL PASS";
/// Printed by [`exit_qemu`] on aarch64 when a test failed. Not load-bearing for
/// the verdict (the absence of [`PASS_SENTINEL`] is already a failure), but it
/// distinguishes "the suite ran and a test failed" from "the guest died".
pub const FAIL_SENTINEL: &str = "CHITTI-TEST: FAILED";

#[repr(u32)]
#[derive(Clone, Copy)]
pub enum QemuExitCode {
    Success = 0x10,
    Failed = 0x11,
}

#[cfg(target_arch = "x86_64")]
const ISA_DEBUG_EXIT_PORT: u16 = 0xf4;

/// Write to the `isa-debug-exit` device, which QEMU is configured (by
/// `xtask`) to translate into the process exit code `(value << 1) | 1`.
/// Never returns: if the device isn't present (e.g. run outside QEMU) the
/// CPU just halts instead.
#[cfg(target_arch = "x86_64")]
pub fn exit_qemu(code: QemuExitCode) -> ! {
    // SAFETY: `xtask` always launches QEMU with
    // `-device isa-debug-exit,iobase=0xf4,iosize=0x04`, so this port is
    // valid for a 32-bit write whenever this binary runs under our runner.
    unsafe { crate::arch::x86_64::port::outl(ISA_DEBUG_EXIT_PORT, code as u32) };
    loop {
        crate::arch::x86_64::hlt();
    }
}

/// Report the verdict on the serial console, then power the machine off via PSCI
/// `SYSTEM_OFF` (which QEMU turns into a process exit).
#[cfg(target_arch = "aarch64")]
pub fn exit_qemu(code: QemuExitCode) -> ! {
    match code {
        QemuExitCode::Success => crate::serial_println!("{}", PASS_SENTINEL),
        QemuExitCode::Failed => crate::serial_println!("{}", FAIL_SENTINEL),
    }
    crate::arch::poweroff()
}
