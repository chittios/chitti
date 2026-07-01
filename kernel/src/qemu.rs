//! `isa-debug-exit` wiring: lets the in-kernel test harness terminate QEMU
//! with a status code the host side (`xtask`) can check, instead of the
//! test runner having to parse serial output to determine pass/fail.

use crate::arch::x86_64::port::outl;

const ISA_DEBUG_EXIT_PORT: u16 = 0xf4;

#[repr(u32)]
#[derive(Clone, Copy)]
pub enum QemuExitCode {
    Success = 0x10,
    Failed = 0x11,
}

/// Write to the `isa-debug-exit` device, which QEMU is configured (by
/// `xtask`) to translate into the process exit code `(value << 1) | 1`.
/// Never returns: if the device isn't present (e.g. run outside QEMU) the
/// CPU just halts instead.
pub fn exit_qemu(code: QemuExitCode) -> ! {
    // SAFETY: `xtask` always launches QEMU with
    // `-device isa-debug-exit,iobase=0xf4,iosize=0x04`, so this port is
    // valid for a 32-bit write whenever this binary runs under our runner.
    unsafe { outl(ISA_DEBUG_EXIT_PORT, code as u32) };
    loop {
        crate::arch::x86_64::hlt();
    }
}
