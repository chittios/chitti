//! FPU/SSE hardware-state initialization: `CR0`/`CR4` bits, `EFER.NXE`,
//! and `XSAVE`/`XSETBV` scaffolding.
//!
//! Per the locked decision in `CHITTI_OS_HANDOFF.md` Part 2, SIMD
//! *codegen* (the compiler's `sse`/`avx` target features) stays off until
//! Phase 3 — `targets/x86_64-chitti.json` still disables them, so this
//! module changes no generated code. What it does do is put the CPU's
//! FPU/SSE state itself into a well-defined, correctly-enabled shape, so
//! a future `XSAVE`-based context switch (Phase 2) and Phase 3's SIMD
//! tensor kernels have correct hardware to build on rather than an
//! uninitialized/undefined FPU.

use core::arch::asm;

fn read_cr0() -> u64 {
    let v: u64;
    // SAFETY: reading CR0 has no side effects.
    unsafe { asm!("mov {}, cr0", out(reg) v, options(nomem, nostack, preserves_flags)) };
    v
}

fn write_cr0(v: u64) {
    // SAFETY: only clears EM and sets MP/NE (see `init`), all standard,
    // documented bits; run once at boot before any FPU-using code runs.
    unsafe { asm!("mov cr0, {}", in(reg) v, options(nomem, nostack, preserves_flags)) };
}

fn read_cr4() -> u64 {
    let v: u64;
    // SAFETY: reading CR4 has no side effects.
    unsafe { asm!("mov {}, cr4", out(reg) v, options(nomem, nostack, preserves_flags)) };
    v
}

fn write_cr4(v: u64) {
    // SAFETY: only sets OSFXSR/OSXMMEXCPT/OSXSAVE (see `init`), gated on
    // the corresponding CPUID feature bit where applicable (OSXSAVE).
    unsafe { asm!("mov cr4, {}", in(reg) v, options(nomem, nostack, preserves_flags)) };
}

/// Returns `(eax, ecx, edx)`; `ebx` is deliberately not exposed as an
/// operand -- LLVM reserves `rbx` internally, so `cpuid` saves/restores it
/// via the stack instead of naming it in the operand list.
fn cpuid(leaf: u32, subleaf: u32) -> (u32, u32, u32) {
    let (eax, ecx, edx);
    // SAFETY: `cpuid` has no side effects beyond returning feature bits;
    // `rbx` is preserved around the clobber via push/pop.
    unsafe {
        asm!(
            "push rbx",
            "cpuid",
            "pop rbx",
            inout("eax") leaf => eax,
            inout("ecx") subleaf => ecx,
            out("edx") edx,
            options(preserves_flags),
        );
    }
    (eax, ecx, edx)
}

fn read_msr(msr: u32) -> u64 {
    let (lo, hi): (u32, u32);
    // SAFETY: reading the EFER MSR (the only one this module touches) is
    // always valid on any x86_64 CPU in long mode.
    unsafe {
        asm!("rdmsr", in("ecx") msr, out("eax") lo, out("edx") hi, options(nomem, nostack, preserves_flags));
    }
    ((hi as u64) << 32) | lo as u64
}

fn write_msr(msr: u32, value: u64) {
    let lo = value as u32;
    let hi = (value >> 32) as u32;
    // SAFETY: only ever used here to set EFER.NXE (bit 11); every other
    // EFER bit is preserved via read-modify-write in `init`.
    unsafe {
        asm!("wrmsr", in("ecx") msr, in("eax") lo, in("edx") hi, options(nomem, nostack, preserves_flags));
    }
}

const EFER_MSR: u32 = 0xc000_0080;
const EFER_NXE: u64 = 1 << 11;

/// Enable the No-Execute page-table bit at the CPU level. Must run before
/// `mm::paging` maps anything `NO_EXECUTE`: without `EFER.NXE`, bit 63 of
/// a page-table entry is *reserved* and setting it faults instead of
/// denying execution.
pub fn enable_nx() {
    let efer = read_msr(EFER_MSR);
    write_msr(EFER_MSR, efer | EFER_NXE);
}

/// Bring the FPU/SSE unit into a defined, correctly-enabled state:
/// `CR0.EM=0/MP=1/NE=1`, `fninit`, and (when the CPU supports it)
/// `CR4.OSFXSR/OSXMMEXCPT/OSXSAVE` + `XSETBV` enabling the x87 and SSE
/// `XSAVE` state components.
pub fn init() {
    let mut cr0 = read_cr0();
    cr0 &= !(1 << 2); // EM: don't emulate the FPU in software
    cr0 |= 1 << 1; // MP: WAIT/FWAIT instructions respect TS
    cr0 |= 1 << 5; // NE: native (#MF) FPU error reporting, not the legacy IRQ13 path
    write_cr0(cr0);

    // SAFETY: `fninit` only resets FPU state (control word, tags, etc.);
    // safe to run unconditionally once CR0.EM is clear.
    unsafe { asm!("fninit", options(nomem, nostack)) };

    let (_, ecx1, _) = cpuid(1, 0);
    let xsave_supported = ecx1 & (1 << 26) != 0;

    let mut cr4 = read_cr4();
    cr4 |= 1 << 9; // OSFXSR: OS supports FXSAVE/FXRSTOR and SSE
    cr4 |= 1 << 10; // OSXMMEXCPT: OS handles unmasked SIMD FP exceptions
    if xsave_supported {
        cr4 |= 1 << 18; // OSXSAVE: enable XSAVE/XRSTOR/XGETBV/XSETBV
    }
    write_cr4(cr4);

    if xsave_supported {
        // XCR0: enable the x87 (bit 0) and SSE (bit 1) state components.
        // AVX (bit 2) stays disabled -- SIMD codegen is off until Phase 3.
        let xcr0: u64 = 0b11;
        // SAFETY: requires CR4.OSXSAVE=1 (just set above); xcr0=0 is
        // always a valid XCR0 register selector.
        unsafe {
            asm!(
                "xsetbv",
                in("ecx") 0u32,
                in("eax") xcr0 as u32,
                in("edx") (xcr0 >> 32) as u32,
                options(nostack, preserves_flags),
            );
        }
    }

    enable_nx();

    crate::ktrace::log_fmt(format_args!(
        "fpu: CR0/CR4 configured, xsave_supported={xsave_supported}, EFER.NXE enabled"
    ));
}
