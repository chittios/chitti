//! x86_64-specific primitives: port I/O, GDT/TSS, IDT + exceptions, the
//! legacy PIC + PIT timer + keyboard IRQs, FPU/SSE init, and 4-level
//! paging.

pub mod ac97;
pub mod ahci;
pub mod apic;
pub mod disk;
pub mod sb16;
pub mod fpu;
pub mod gdt;
pub mod idt;
pub mod interrupts;
pub mod i8042;
pub mod keyboard;
pub mod nvme;
pub mod paging;
pub mod pci;
pub mod pic;
pub mod pit;
pub mod port;
pub mod rtc;
pub mod xhci;

/// Raw cycle counter (TSC) for entropy mixing (see `arch::cycle_count`).
pub fn cycle_count() -> u64 {
    let lo: u32;
    let hi: u32;
    // SAFETY: `rdtsc` reads the timestamp counter into edx:eax; no memory or
    // flag effects. Present on every x86-64 CPU.
    unsafe { core::arch::asm!("rdtsc", out("eax") lo, out("edx") hi, options(nomem, nostack, preserves_flags)) };
    (hi as u64) << 32 | lo as u64
}

/// A hardware random word via `RDRAND` when CPUID reports it, else 0 (see
/// `arch::hw_rand`). CPUID.01H:ECX bit 30 = RDRAND.
pub fn hw_rand() -> u64 {
    let ecx: u32;
    // SAFETY: CPUID leaf 1 is universally available; `rbx` is callee-saved by
    // LLVM so we swap it out around the instruction.
    unsafe {
        core::arch::asm!(
            "mov {tmp:r}, rbx",
            "cpuid",
            "mov rbx, {tmp:r}",
            tmp = out(reg) _,
            inout("eax") 1u32 => _,
            out("ecx") ecx,
            out("edx") _,
            options(nostack, preserves_flags),
        );
    }
    if ecx & (1 << 30) == 0 {
        return 0;
    }
    let v: u64;
    let ok: u8;
    // SAFETY: RDRAND is implemented per the CPUID check; CF=1 => a valid random
    // value was returned. A not-ready result (CF=0) yields 0; the caller mixes
    // multiple samples plus the TSC.
    unsafe {
        core::arch::asm!(
            "rdrand {v}",
            "setc {ok}",
            v = out(reg) v,
            ok = out(reg_byte) ok,
            options(nomem, nostack),
        );
    }
    if ok != 0 {
        v
    } else {
        0
    }
}

/// Halt the CPU until the next interrupt.
#[inline]
pub fn hlt() {
    // SAFETY: `hlt` has no memory-safety implications; it just stops
    // instruction execution until an interrupt arrives.
    unsafe { core::arch::asm!("hlt", options(nomem, nostack, preserves_flags)) }
}

/// Power off / exit QEMU cleanly (so typing `exit` at the shell terminates the
/// emulator instead of leaving it idling). Uses the `isa-debug-exit` device
/// (present in every `xtask` QEMU invocation): a dword write to port 0xf4
/// exits QEMU. Falls back to a halt loop if that somehow returns.
pub fn poweroff() -> ! {
    // SAFETY: 0xf4 is the isa-debug-exit device port; a write exits QEMU.
    unsafe { port::outl(0xf4, 0x10) };
    loop {
        hlt();
    }
}
