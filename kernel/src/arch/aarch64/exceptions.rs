//! aarch64 EL1 exception vector table + handlers — the aarch64 counterpart of
//! the x86 IDT (`arch::x86_64::idt`). The kernel runs at EL1 using `SP_EL1`
//! (SPSel=1), so every trap arrives through the "Current EL with SP_ELx" group;
//! the IRQ entry there (offset 0x280) is what the GIC-routed generic-timer
//! interrupt hits, driving `sched::on_timer_tick` for preemptive scheduling.
//!
//! Preemption composes with the *cooperative* `context::switch_to` the same way
//! x86 does: the IRQ stub saves the full **caller-saved** state (x0–x30, the FP/
//! SIMD bank q0–q31, and ELR/SPSR) on the interrupted task's stack, and
//! `switch_to` (invoked from `on_timer_tick`) saves the **callee-saved** state —
//! so a preemptive task switch preserves the complete context, and the closing
//! `eret` resumes the interrupted code exactly where it left off.

use core::arch::global_asm;

/// Install the vector table at `VBAR_EL1` on the current core. Call once, on the
/// BSP, before enabling IRQs.
///
/// # Safety
/// Must run at EL1; `vectors` is a valid, 2 KiB-aligned 16-entry table.
pub unsafe fn init() {
    unsafe {
        let base = vectors as *const () as u64;
        core::arch::asm!("msr vbar_el1, {}", "isb", in(reg) base, options(nostack, preserves_flags));
    }
}

unsafe extern "C" {
    /// The vector table symbol defined by the `global_asm!` below.
    fn vectors();
}

/// IRQ dispatch, called from the IRQ vector stub with the full trap frame saved.
/// Routes the generic-timer interrupt to the scheduler; ignores anything else
/// after acknowledging it (so a stray interrupt can't wedge the core).
#[no_mangle]
extern "C" fn aarch64_irq_dispatch() {
    super::gic::handle_irq();
}

/// Synchronous-exception dispatch (Current EL, SP_ELx). `frame` points at the
/// saved trap frame (x0..x30 at [0..248], ELR at [256], SPSR at [264], q0..q31
/// at [272..]). Used for the GIC CPU-interface **UNDEF probe**: while
/// `gic::probing()`, a trapped (undefined) instruction is *recovered* — we note
/// the fault and advance the saved ELR past it so the `eret` resumes at the next
/// instruction (this is how we detect that HVF doesn't expose the `ICC_*`
/// system-register interface without crashing). Any other synchronous exception
/// is a real kernel bug: log the syndrome and halt.
#[no_mangle]
extern "C" fn aarch64_sync_dispatch(frame: *mut u64) {
    let esr: u64;
    // SAFETY: reading ESR_EL1 is always valid at EL1.
    unsafe { core::arch::asm!("mrs {}, esr_el1", out(reg) esr, options(nomem, nostack)) };
    // Recoverable probes: the GIC CPU-interface UNDEF probe (detecting HVF's
    // missing ICC_* sysregs) and the UART MMIO probe (finding the PL011 base on
    // a platform without ACPI SPCR). While either is active, a trapped
    // instruction is *recovered* — note the fault and advance the saved ELR
    // past it (index 32 = byte offset 256; all aarch64 instructions are 4 bytes)
    // so the `eret` resumes at the next instruction. Any other synchronous
    // exception is a real kernel bug: log the syndrome and halt.
    if super::gic::probing() || super::uart_probing() {
        super::gic::note_probe_fault();
        super::note_uart_fault();
        // SAFETY: `frame` is our own trap frame on the current stack.
        unsafe {
            let elr = frame.add(32);
            *elr = (*elr).wrapping_add(4);
        }
        return;
    }
    // SAFETY: `frame` is our own trap frame; index 32 is the saved ELR.
    let elr = unsafe { *frame.add(32) };
    crate::ktrace::log_fmt(format_args!("aarch64 FATAL sync exception: ESR_EL1={:#x} ELR_EL1={:#x}", esr, elr));
    loop {
        super::hlt();
    }
}

/// Fatal-exception handler for the SError / unexpected vectors: log the
/// syndrome + faulting address and halt. A kernel bug lands here instead of
/// silently looping in a vector.
#[no_mangle]
extern "C" fn aarch64_fatal_exception(esr: u64, elr: u64, kind: u64) -> ! {
    crate::ktrace::log_fmt(format_args!("aarch64 FATAL exception: kind={} ESR_EL1={:#x} ELR_EL1={:#x}", kind, esr, elr));
    loop {
        super::hlt();
    }
}

// The vector table. Each of the 16 entries is 0x80 bytes; the table is
// 0x800-aligned. Only the "Current EL SPx" sync + IRQ entries do real work; the
// rest funnel to the fatal handler (they should never fire in this kernel).
//
// The IRQ frame (784 bytes, 16-aligned): x0..x30 at [0..248], ELR/SPSR at
// [256], q0..q31 at [272..784]. FP is saved wholesale (all of q0-q31, incl. the
// caller-saved halves of v8-v15) so preemption is always correct.
global_asm!(
    r#"
.macro FATAL kind
    // Save a couple of scratch regs, gather syndrome, call the Rust handler.
    mrs  x0, esr_el1
    mrs  x1, elr_el1
    mov  x2, #\kind
    b    aarch64_fatal_exception
.endm

.balign 0x800
.global vectors
vectors:
    // --- Current EL with SP0 (unused: we run on SP_ELx) ---
    .balign 0x80
    FATAL 0                      // 0x000 Synchronous
    .balign 0x80
    FATAL 1                      // 0x080 IRQ
    .balign 0x80
    FATAL 2                      // 0x100 FIQ
    .balign 0x80
    FATAL 3                      // 0x180 SError
    // --- Current EL with SP_ELx (our EL1 kernel) ---
    .balign 0x80
    b    sync_current_spx        // 0x200 Synchronous (GIC probe recovery + faults)
    .balign 0x80
    b    irq_current_spx         // 0x280 IRQ  <-- generic-timer interrupt
    .balign 0x80
    FATAL 6                      // 0x300 FIQ
    .balign 0x80
    FATAL 7                      // 0x380 SError
    // --- Lower EL, AArch64 (unused: no EL0) ---
    .balign 0x80
    FATAL 8                      // 0x400 Synchronous
    .balign 0x80
    FATAL 9                      // 0x480 IRQ
    .balign 0x80
    FATAL 10                     // 0x500 FIQ
    .balign 0x80
    FATAL 11                     // 0x580 SError
    // --- Lower EL, AArch32 (unused) ---
    .balign 0x80
    FATAL 12                     // 0x600 Synchronous
    .balign 0x80
    FATAL 13                     // 0x680 IRQ
    .balign 0x80
    FATAL 14                     // 0x700 FIQ
    .balign 0x80
    FATAL 15                     // 0x780 SError

// Save the full caller context (x0-x30, ELR/SPSR, q0-q31) into a 784-byte frame.
.macro SAVE_FRAME
    sub  sp, sp, #784
    stp  x0,  x1,  [sp, #0]
    stp  x2,  x3,  [sp, #16]
    stp  x4,  x5,  [sp, #32]
    stp  x6,  x7,  [sp, #48]
    stp  x8,  x9,  [sp, #64]
    stp  x10, x11, [sp, #80]
    stp  x12, x13, [sp, #96]
    stp  x14, x15, [sp, #112]
    stp  x16, x17, [sp, #128]
    stp  x18, x19, [sp, #144]
    stp  x20, x21, [sp, #160]
    stp  x22, x23, [sp, #176]
    stp  x24, x25, [sp, #192]
    stp  x26, x27, [sp, #208]
    stp  x28, x29, [sp, #224]
    str  x30,      [sp, #240]
    mrs  x0, elr_el1
    mrs  x1, spsr_el1
    stp  x0,  x1,  [sp, #256]
    stp  q0,  q1,  [sp, #272]
    stp  q2,  q3,  [sp, #304]
    stp  q4,  q5,  [sp, #336]
    stp  q6,  q7,  [sp, #368]
    stp  q8,  q9,  [sp, #400]
    stp  q10, q11, [sp, #432]
    stp  q12, q13, [sp, #464]
    stp  q14, q15, [sp, #496]
    stp  q16, q17, [sp, #528]
    stp  q18, q19, [sp, #560]
    stp  q20, q21, [sp, #592]
    stp  q22, q23, [sp, #624]
    stp  q24, q25, [sp, #656]
    stp  q26, q27, [sp, #688]
    stp  q28, q29, [sp, #720]
    stp  q30, q31, [sp, #752]
.endm

.macro RESTORE_FRAME
    ldp  q30, q31, [sp, #752]
    ldp  q28, q29, [sp, #720]
    ldp  q26, q27, [sp, #688]
    ldp  q24, q25, [sp, #656]
    ldp  q22, q23, [sp, #624]
    ldp  q20, q21, [sp, #592]
    ldp  q18, q19, [sp, #560]
    ldp  q16, q17, [sp, #528]
    ldp  q14, q15, [sp, #496]
    ldp  q12, q13, [sp, #464]
    ldp  q10, q11, [sp, #432]
    ldp  q8,  q9,  [sp, #400]
    ldp  q6,  q7,  [sp, #368]
    ldp  q4,  q5,  [sp, #336]
    ldp  q2,  q3,  [sp, #304]
    ldp  q0,  q1,  [sp, #272]
    ldp  x0,  x1,  [sp, #256]
    msr  elr_el1, x0
    msr  spsr_el1, x1
    ldr  x30,      [sp, #240]
    ldp  x28, x29, [sp, #224]
    ldp  x26, x27, [sp, #208]
    ldp  x24, x25, [sp, #192]
    ldp  x22, x23, [sp, #176]
    ldp  x20, x21, [sp, #160]
    ldp  x18, x19, [sp, #144]
    ldp  x16, x17, [sp, #128]
    ldp  x14, x15, [sp, #112]
    ldp  x12, x13, [sp, #96]
    ldp  x10, x11, [sp, #80]
    ldp  x8,  x9,  [sp, #64]
    ldp  x6,  x7,  [sp, #48]
    ldp  x4,  x5,  [sp, #32]
    ldp  x2,  x3,  [sp, #16]
    ldp  x0,  x1,  [sp, #0]
    add  sp, sp, #784
.endm

// Common IRQ handler: save the full caller context, dispatch, restore, eret.
irq_current_spx:
    SAVE_FRAME
    bl   aarch64_irq_dispatch
    RESTORE_FRAME
    eret

// Synchronous handler: save the frame, pass its pointer to the Rust dispatcher
// (which may advance the saved ELR to recover from the GIC probe), restore, eret.
sync_current_spx:
    SAVE_FRAME
    mov  x0, sp
    bl   aarch64_sync_dispatch
    RESTORE_FRAME
    eret
"#
);
