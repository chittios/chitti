//! FPU/SSE hardware-state initialization: `CR0`/`CR4` bits, `EFER.NXE`,
//! and `XSAVE`/`XSETBV` scaffolding.
//!
//! As of Phase 3, SIMD *codegen* is on (`targets/x86_64-chitti.json` now
//! sets `+sse,+sse2` and drops soft-float) for the Cortex tensor kernels.
//! The `CR0.EM=0/MP=1/NE=1` + `CR4.OSFXSR/OSXMMEXCPT` state this module
//! establishes is exactly what SSE2 requires, and it enables the
//! `FXSAVE`/`FXRSTOR` that `sched::context` uses to preserve each task's
//! SSE registers across a context switch. `XSAVE`/AVX stay off: the
//! default QEMU CPU reports no `XSAVE` support, and SSE2 needs only
//! `OSFXSR`, so nothing here depends on it.

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

/// Enable SSE at the hardware level: clear `CR0.EM`, set `CR0.MP/NE` and
/// `CR4.OSFXSR/OSXMMEXCPT`. This is the bare minimum for SSE instructions
/// not to fault, and **must run before any SSE-using code executes**. Now
/// that SIMD codegen is on crate-wide (Phase 3), the optimizer emits XMM
/// instructions in ordinary code (vectorized loops, struct moves), so
/// `_start` calls this as its very first action -- long before the fuller
/// `fpu::init` below, which can't run until serial/ktrace are up. Idempotent.
pub fn enable_sse() {
    let mut cr0 = read_cr0();
    cr0 &= !(1 << 2); // EM: don't emulate the FPU in software (EM=1 => SSE #UD)
    cr0 |= 1 << 1; // MP: WAIT/FWAIT instructions respect TS
    cr0 |= 1 << 5; // NE: native (#MF) FPU error reporting, not the legacy IRQ13 path
    write_cr0(cr0);

    let mut cr4 = read_cr4();
    cr4 |= 1 << 9; // OSFXSR: OS supports FXSAVE/FXRSTOR and SSE
    cr4 |= 1 << 10; // OSXMMEXCPT: OS handles unmasked SIMD FP exceptions
    write_cr4(cr4);
}

/// Bring the FPU/SSE unit into a fully defined state: the `enable_sse` bits
/// (idempotently), `fninit`, and (when the CPU supports it) `CR4.OSXSAVE` +
/// `XSETBV` enabling the x87 and SSE `XSAVE` state components. Logs via
/// `ktrace`, so it runs after serial is up (unlike `enable_sse`).
pub fn init() {
    enable_sse();

    // SAFETY: `fninit` only resets FPU state (control word, tags, etc.);
    // safe to run unconditionally once CR0.EM is clear (done in enable_sse).
    unsafe { asm!("fninit", options(nomem, nostack)) };

    let (_, ecx1, _) = cpuid(1, 0);
    let xsave_supported = ecx1 & (1 << 26) != 0;

    // OSFXSR/OSXMMEXCPT are already set by `enable_sse`; only OSXSAVE (gated
    // on CPUID) remains, to unlock XSETBV on CPUs that support XSAVE.
    if xsave_supported {
        let cr4 = read_cr4() | (1 << 18); // OSXSAVE
        write_cr4(cr4);
        // XCR0: enable the x87 (bit 0) and SSE (bit 1) state components.
        // AVX (bit 2) stays disabled -- the default QEMU CPU lacks it.
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
