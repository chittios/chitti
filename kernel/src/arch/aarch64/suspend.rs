//! **PSCI `SYSTEM_SUSPEND`** — the ARM analogue of ACPI S3, and a much smaller job.
//!
//! Where x86 wakes in real mode with nothing set up and needs a trampoline to walk the
//! CPU back into long mode ([`super::super::x86_64::suspend`]), PSCI firmware does most
//! of it: the OS hands `SYSTEM_SUSPEND` an entry point and a context value, the core
//! powers down, and on wake it starts at that entry point with the context in `x0` —
//! the same contract as `CPU_ON`, which SMP bring-up already uses.
//!
//! What firmware does *not* restore is the MMU and the EL1 system registers, so the
//! resume stub puts those back before anything compiled can run. Two facts about this
//! kernel make that short:
//!
//! - The translation map is **identity** (VA == PA), so the stub is executing at the
//!   same address before and after `SCTLR.M` goes on. x86 needed a temporary page table
//!   precisely because that is *not* true there.
//! - **TTBR1 is disabled** ([`super::mmu`] sets `TCR.EPD1`), so there is exactly one
//!   translation base to restore.
//!
//! ## What is saved, and why each one
//!
//! `MAIR`/`TCR`/`TTBR0` are the translation configuration; `SCTLR` turns it on and also
//! carries the **A bit, which must stay clear** — the NEON kernels use unaligned `ldr q`
//! and that is only legal on Normal memory with alignment checking off, so resuming with
//! firmware's `SCTLR` instead of the kernel's would alignment-fault the first matvec.
//! `VBAR` is the exception vector base, without which the first interrupt after resume
//! goes nowhere. `CPACR` re-enables FP/SIMD at EL1; without it the first floating-point
//! instruction traps. Then the callee-saved registers, `SP` and a return address, so the
//! resume looks to the caller like an ordinary function return.
//!
//! **Unverified**: QEMU's `virt` machine does not implement `SYSTEM_SUSPEND` on every
//! version, and where `PSCI_FEATURES` says it is absent [`crate::power::plan`] reports
//! that rather than calling it. The register save/restore arithmetic is unit-tested; the
//! transition is not something this environment can perform.

use crate::power::{
    PsciSavedState as SavedState, S_CPACR, S_MAIR, S_REGS, S_RESUMED, S_SCTLR, S_SP, S_TCR,
    S_TTBR0, S_VBAR,
};

/// The saved state. In RAM, which `SYSTEM_SUSPEND` preserves; its address is handed to
/// firmware as the context value and comes back in `x0`.
static mut STATE: SavedState = SavedState {
    resumed: 0,
    mair: 0,
    tcr: 0,
    ttbr0: 0,
    vbar: 0,
    cpacr: 0,
    sctlr: 0,
    sp: 0,
    regs: [0; 12],
};

core::arch::global_asm!(
    r#"
.section .text
.balign 16
.global chitti_psci_wake
.set SW_RESUMED, 0
.set SW_MAIR,    8
.set SW_TCR,     16
.set SW_TTBR0,   24
.set SW_VBAR,    32
.set SW_CPACR,   40
.set SW_SCTLR,   48
.set SW_SP,      56
.set SW_REGS,    64
chitti_psci_wake:
    // x0 = the context value PSCI was given = physical address of the saved state. The
    // MMU is off, so this is a plain physical access — and because the map is identity,
    // the same address keeps working once it is back on.
    mov     x9, x0
    // Record the resume before anything can go wrong, while the access is trivially
    // valid. `suspend` reads it to tell "came back" from "firmware refused".
    mov     x10, #1
    str     x10, [x9, SW_RESUMED]

    ldr     x1, [x9, SW_MAIR]
    msr     mair_el1, x1
    ldr     x1, [x9, SW_TCR]
    msr     tcr_el1, x1
    ldr     x1, [x9, SW_TTBR0]
    msr     ttbr0_el1, x1
    // Vectors and FP/SIMD before the MMU: an abort taken during the enable sequence with
    // no VBAR would be unrecoverable and undiagnosable.
    ldr     x1, [x9, SW_VBAR]
    msr     vbar_el1, x1
    ldr     x1, [x9, SW_CPACR]
    msr     cpacr_el1, x1
    dsb     ish
    tlbi    vmalle1
    dsb     ish
    isb
    // SCTLR last: it is what turns translation on, and it carries the A bit the NEON
    // kernels depend on being clear.
    ldr     x1, [x9, SW_SCTLR]
    msr     sctlr_el1, x1
    isb

    ldr     x2, [x9, SW_SP]
    mov     sp, x2
    ldp     x19, x20, [x9, SW_REGS]
    ldp     x21, x22, [x9, SW_REGS + 16]
    ldp     x23, x24, [x9, SW_REGS + 32]
    ldp     x25, x26, [x9, SW_REGS + 48]
    ldp     x27, x28, [x9, SW_REGS + 64]
    ldp     x29, x30, [x9, SW_REGS + 80]
    ret
"#
);

extern "C" {
    /// The resume entry point handed to PSCI. Its link address is also its physical
    /// address, because the map is identity.
    fn chitti_psci_wake();
}

/// Suspend the machine. Returns once it has resumed.
///
/// `Err` means firmware declined and nothing happened — the PSCI status is included,
/// because `NOT_SUPPORTED`, `INVALID_ADDRESS` and `DENIED` need different responses.
pub fn suspend() -> Result<(), &'static str> {
    if !super::psci_available() {
        return Err("no PSCI conduit on this machine");
    }
    // SAFETY: single-threaded shell path; `STATE` has no other writer.
    let st = unsafe { core::ptr::addr_of_mut!(STATE) };
    // SAFETY: reads of EL1 system registers, none of which have side effects.
    unsafe {
        let (mut mair, mut tcr, mut ttbr0, mut vbar, mut cpacr, mut sctlr): (u64, u64, u64, u64, u64, u64);
        core::arch::asm!("mrs {}, mair_el1", out(reg) mair, options(nomem, nostack));
        core::arch::asm!("mrs {}, tcr_el1", out(reg) tcr, options(nomem, nostack));
        core::arch::asm!("mrs {}, ttbr0_el1", out(reg) ttbr0, options(nomem, nostack));
        core::arch::asm!("mrs {}, vbar_el1", out(reg) vbar, options(nomem, nostack));
        core::arch::asm!("mrs {}, cpacr_el1", out(reg) cpacr, options(nomem, nostack));
        core::arch::asm!("mrs {}, sctlr_el1", out(reg) sctlr, options(nomem, nostack));
        (*st).mair = mair;
        (*st).tcr = tcr;
        (*st).ttbr0 = ttbr0;
        (*st).vbar = vbar;
        (*st).cpacr = cpacr;
        (*st).sctlr = sctlr;
        (*st).resumed = 0;
    }

    let entry = chitti_psci_wake as usize as u64;
    let ctx = st as u64;
    crate::ktrace::log_fmt(format_args!(
        "psci: SYSTEM_SUSPEND entry {entry:#x} ctx {ctx:#x}"
    ));

    let status: u64;
    // SAFETY: saves the callee-saved registers, SP and a return address into the state
    // block, then asks firmware to suspend. On success the call does not return; the core
    // powers down and later restarts at `chitti_psci_wake`, which restores exactly these
    // and `ret`s to the `3:` label — so control arrives back here as though the call had
    // returned. Every caller-saved register is declared clobbered, because on the resume
    // path they hold whatever the stub left.
    unsafe {
        core::arch::asm!(
            "adr x9, 3f",
            "stp x19, x20, [{s}, {o_regs}]",
            "stp x21, x22, [{s}, {o_regs} + 16]",
            "stp x23, x24, [{s}, {o_regs} + 32]",
            "stp x25, x26, [{s}, {o_regs} + 48]",
            "stp x27, x28, [{s}, {o_regs} + 64]",
            "stp x29, x9,  [{s}, {o_regs} + 80]",
            "mov x9, sp",
            "str x9, [{s}, {o_sp}]",
            "mov x0, {fid}",
            "mov x1, {entry}",
            "mov x2, {ctx}",
            "hvc #0",
            "3:",
            "mov {status}, x0",
            s = in(reg) st,
            fid = in(reg) super::PSCI_SYSTEM_SUSPEND as u64,
            entry = in(reg) entry,
            ctx = in(reg) ctx,
            o_regs = const S_REGS,
            o_sp = const S_SP,
            status = out(reg) status,
            out("x0") _,
            out("x1") _,
            out("x2") _,
            out("x3") _,
            out("x4") _,
            out("x5") _,
            out("x6") _,
            out("x7") _,
            out("x8") _,
            out("x9") _,
            out("x10") _,
            out("x11") _,
            out("x12") _,
            out("x13") _,
            out("x14") _,
            out("x15") _,
            out("x16") _,
            out("x17") _,
        );
    }

    // SAFETY: as above.
    if unsafe { (*st).resumed } != 0 {
        crate::ktrace::log("psci", "resumed from SYSTEM_SUSPEND");
        return Ok(());
    }
    // The call returned without suspending, so `x0` is a PSCI error.
    match status as i64 {
        super::PSCI_NOT_SUPPORTED => Err("firmware does not implement SYSTEM_SUSPEND"),
        -2 => Err("SYSTEM_SUSPEND rejected the entry point (INVALID_ADDRESS)"),
        -3 => Err("SYSTEM_SUSPEND denied: another core is still online"),
        _ => Err("SYSTEM_SUSPEND failed"),
    }
}
