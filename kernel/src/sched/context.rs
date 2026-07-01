//! Stackful coroutine context switch. `switch_to` saves the outgoing
//! task's callee-saved registers and `RFLAGS` onto its own stack, swaps
//! `rsp`, then restores the incoming task's -- symmetric enough that it
//! works identically whether called from ordinary task code (a
//! cooperative `yield_now`) or from inside the PIT timer's interrupt
//! handler (a preemptive one): see `sched::yield_now`'s doc comment for
//! why the latter is safe.
//!
//! Naked so there is no compiler-generated prologue/epilogue to fight
//! with -- this function's body is the entire truth about what a "task"
//! is, register-wise.

use core::arch::{asm, naked_asm};

/// Per-task FPU save area. Sized and 64-byte aligned for `XSAVE` of the
/// x87 + SSE + AVX state (the YMM upper halves the AVX2 kernels use); when
/// AVX2 is unavailable it holds a plain 512-byte `FXSAVE` image instead.
/// 1088 bytes comfortably covers the x87+SSE+AVX `XSAVE` area (~576 B).
///
/// Once Phase 3 turns on SIMD codegen, the tensor kernels keep live `f32`
/// accumulators in vector registers across many instructions, so a
/// preemptive context switch mid-matmul (the timer IRQ can fire at any
/// instruction) has to preserve this state per task -- exactly as it
/// already preserves the GPRs. `FXSAVE` covers only XMM (lower 128 bits),
/// so once AVX2 is on we must use `XSAVE`/`XRSTOR` to also save the YMM
/// upper halves; the choice is made at runtime from `fpu::avx2_enabled()`.
#[repr(C, align(64))]
pub struct FxArea([u8; 1088]);

/// `XSAVE`/`XRSTOR` feature mask (EDX:EAX): x87 (0) | SSE (1) | AVX (2).
const XSAVE_MASK: u32 = 0b111;

impl FxArea {
    /// Zeroed is safe here: an area is only ever restored *after* the owning
    /// task has saved into it at least once (see `sched::yield_now`), so the
    /// zeroed bytes -- an invalid FPU state -- are never actually loaded.
    pub const fn new() -> Self {
        Self([0; 1088])
    }
}

impl Default for FxArea {
    fn default() -> Self {
        Self::new()
    }
}

/// Save the live x87/SSE(/AVX) state into `area` (`XSAVE` when AVX2 is on,
/// else `FXSAVE`).
///
/// # Safety
/// `area` must point at a valid, writable, 64-byte-aligned `FxArea`.
#[inline]
pub unsafe fn save_fpu(area: *mut FxArea) {
    if crate::arch::x86_64::fpu::avx2_enabled() {
        // SAFETY: OSXSAVE + XCR0 were configured in fpu::init; `area` is a
        // valid, 64-aligned region large enough for the AVX XSAVE image.
        unsafe {
            asm!("xsave [{}]", in(reg) area, in("eax") XSAVE_MASK, in("edx") 0u32, options(nostack, preserves_flags))
        };
    } else {
        // SAFETY: `area` is valid/writable/aligned; fxsave writes 512 bytes.
        unsafe { asm!("fxsave [{}]", in(reg) area, options(nostack, preserves_flags)) };
    }
}

/// Restore the state previously written by `save_fpu`.
///
/// # Safety
/// `area` must hold a valid image previously written by `save_fpu` at a
/// valid, 64-byte-aligned address, with the same AVX2-enabled state.
#[inline]
pub unsafe fn restore_fpu(area: *const FxArea) {
    if crate::arch::x86_64::fpu::avx2_enabled() {
        // SAFETY: see `save_fpu`; xrstor only reads `area`.
        unsafe {
            asm!("xrstor [{}]", in(reg) area, in("eax") XSAVE_MASK, in("edx") 0u32, options(nostack, readonly, preserves_flags))
        };
    } else {
        // SAFETY: `area` holds a valid fxsave image; fxrstor only reads it.
        unsafe { asm!("fxrstor [{}]", in(reg) area, options(nostack, readonly, preserves_flags)) };
    }
}

/// Switch from the current stack to `new_rsp`, saving the current stack
/// pointer through `current_rsp`. Returns -- to whichever caller next
/// switches back to this task -- only once this task is rescheduled,
/// possibly long after the textual call site. That delayed return *is*
/// what "this task was descheduled and later resumed" means.
///
/// # Safety
/// `new_rsp` must point at a stack laid out exactly as `switch_to` itself
/// leaves one: `RFLAGS` then six callee-saved GPRs (rbx, rbp, r12-r15)
/// above a return address, all as consecutive `u64`s. That's guaranteed
/// either by a stack `switch_to` previously saved into, or one built by
/// `init_stack` below. `current_rsp` must point at a valid, exclusively-
/// owned `u64` slot to receive the outgoing stack pointer.
#[unsafe(naked)]
pub unsafe extern "C" fn switch_to(current_rsp: *mut u64, new_rsp: u64) {
    naked_asm!(
        "pushfq",
        "push rbx",
        "push rbp",
        "push r12",
        "push r13",
        "push r14",
        "push r15",
        "mov [rdi], rsp",
        "mov rsp, rsi",
        "pop r15",
        "pop r14",
        "pop r13",
        "pop r12",
        "pop rbp",
        "pop rbx",
        "popfq",
        "ret",
    );
}

/// Landing pad for a task's very first switch-in. `switch_to`'s restore
/// path pops the entry function pointer into `r12` and its argument into
/// `r13` (see `init_stack`), which are still live in those physical
/// registers the instant `ret` lands here.
#[unsafe(naked)]
unsafe extern "C" fn trampoline() -> ! {
    naked_asm!("mov rdi, r13", "call r12", "call {exit}", exit = sym super::exit_current_task);
}

/// `RFLAGS.IF` set (bit 9) plus the always-1 reserved bit 1: a fresh task
/// starts with interrupts enabled, matching every other task's steady
/// state.
const INITIAL_RFLAGS: u64 = 0x202;

/// Build the fake `switch_to`-compatible frame for a task that has never
/// run, and return the resulting initial `rsp`.
///
/// # Safety
/// `stack_top` must be the (at-least-16-byte-aligned) top of a freshly
/// allocated, exclusively-owned stack at least 64 bytes deep.
pub unsafe fn init_stack(stack_top: u64, entry: extern "C" fn(u64), arg: u64) -> u64 {
    // SAFETY: caller (`sched::spawn`) guarantees `stack_top` sits atop a
    // freshly allocated, exclusively-owned region at least 8 `u64` slots
    // (64 bytes) deep.
    unsafe {
        let mut sp = stack_top as *mut u64;
        let mut push = |value: u64| {
            sp = sp.sub(1);
            sp.write(value);
        };
        push(trampoline as *const () as u64); // return address for switch_to's `ret`
        push(INITIAL_RFLAGS); // popfq
        push(0); // rbx
        push(0); // rbp
        push(entry as *const () as u64); // r12 -- read by `trampoline`
        push(arg); // r13 -- read by `trampoline`
        push(0); // r14
        push(0); // r15
        sp as u64
    }
}
