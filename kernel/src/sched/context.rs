//! Stackful coroutine context switch, per architecture. `switch_to` saves the
//! outgoing task's callee-saved registers onto its own stack, swaps the stack
//! pointer, and restores the incoming task's -- so a "return" from it happens
//! only once the task is rescheduled. `init_stack` fabricates a first-switch
//! frame that lands a fresh task in `trampoline`, which calls its entry point.
//!
//! x86_64 additionally saves `RFLAGS` and uses `save_fpu`/`restore_fpu`
//! (`FXSAVE`/`XSAVE`) around the switch, because it is *preemptive* (the PIT
//! IRQ can switch mid-computation, so caller-saved vector state must be
//! preserved). aarch64 here is *cooperative* only (no GIC/timer IRQ wired yet),
//! so switches happen at call boundaries: `switch_to` saving the callee-saved
//! GPRs (x19-x30) and FP regs (d8-d15) suffices, and `save_fpu`/`restore_fpu`
//! are no-ops.

#[cfg(target_arch = "x86_64")]
pub use x86::*;
#[cfg(target_arch = "aarch64")]
pub use arm::*;

#[cfg(target_arch = "x86_64")]
mod x86 {
    use core::arch::{asm, naked_asm};

    /// Per-task FPU save area: 64-byte-aligned, large enough for the x87 + SSE
    /// + AVX `XSAVE` image (the AVX2 kernels' YMM state), or a 512-byte
    /// `FXSAVE` image when AVX2 is off.
    #[repr(C, align(64))]
    pub struct FxArea([u8; 1088]);

    /// `XSAVE`/`XRSTOR` feature mask (EDX:EAX): x87 (0) | SSE (1) | AVX (2).
    const XSAVE_MASK: u32 = 0b111;

    impl FxArea {
        pub const fn new() -> Self {
            Self([0; 1088])
        }
    }

    impl Default for FxArea {
        fn default() -> Self {
            Self::new()
        }
    }

    /// Save the live x87/SSE(/AVX) state into `area`.
    ///
    /// # Safety
    /// `area` must point at a valid, writable, 64-byte-aligned `FxArea`.
    #[inline]
    pub unsafe fn save_fpu(area: *mut FxArea) {
        if crate::arch::x86_64::fpu::avx2_enabled() {
            // SAFETY: OSXSAVE + XCR0 configured in fpu::init; `area` is valid.
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
    /// `area` must hold a valid image previously written by `save_fpu`.
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
    /// pointer through `current_rsp`.
    ///
    /// # Safety
    /// `new_rsp` must point at a stack laid out exactly as `switch_to` leaves
    /// one (see `init_stack`); `current_rsp` a valid, owned `u64` slot.
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

    /// Landing pad for a task's first switch-in: entry in `r12`, arg in `r13`.
    #[unsafe(naked)]
    unsafe extern "C" fn trampoline() -> ! {
        naked_asm!("mov rdi, r13", "call r12", "call {exit}", exit = sym crate::sched::exit_current_task);
    }

    /// `RFLAGS.IF` set (bit 9) + reserved bit 1: fresh tasks start with IRQs on.
    const INITIAL_RFLAGS: u64 = 0x202;

    /// Build the first-switch frame and return the initial `rsp`.
    ///
    /// # Safety
    /// `stack_top` must top a fresh, owned stack at least 64 bytes deep.
    pub unsafe fn init_stack(stack_top: u64, entry: extern "C" fn(u64), arg: u64) -> u64 {
        // SAFETY: caller guarantees `stack_top` sits atop an owned region.
        unsafe {
            let mut sp = stack_top as *mut u64;
            let mut push = |value: u64| {
                sp = sp.sub(1);
                sp.write(value);
            };
            push(trampoline as *const () as u64); // return address for `ret`
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
}

#[cfg(target_arch = "aarch64")]
mod arm {
    use core::arch::naked_asm;

    /// FP state is saved inline by `switch_to` (callee-saved d8-d15) for the
    /// cooperative aarch64 scheduler, so the per-task area is empty.
    #[repr(C)]
    pub struct FxArea;

    impl FxArea {
        pub const fn new() -> Self {
            Self
        }
    }

    impl Default for FxArea {
        fn default() -> Self {
            Self::new()
        }
    }

    /// No-op: `switch_to` saves/restores the callee-saved FP regs itself.
    ///
    /// # Safety
    /// `area` must point at a valid `FxArea` (a ZST); nothing is dereferenced.
    #[inline]
    pub unsafe fn save_fpu(_area: *mut FxArea) {}

    /// No-op counterpart to `save_fpu`.
    ///
    /// # Safety
    /// As `save_fpu`.
    #[inline]
    pub unsafe fn restore_fpu(_area: *const FxArea) {}

    /// Switch from the current stack to `new_sp`, saving the current stack
    /// pointer through `current_sp`. Saves the callee-saved GPRs (x19-x30) and
    /// FP regs (d8-d15) onto the current stack, swaps `sp`, and restores.
    ///
    /// # Safety
    /// `new_sp` must point at a stack laid out exactly as this leaves one (see
    /// `init_stack`); `current_sp` a valid, owned `u64` slot.
    #[unsafe(naked)]
    pub unsafe extern "C" fn switch_to(current_sp: *mut u64, new_sp: u64) {
        naked_asm!(
            "stp x19, x20, [sp, #-16]!",
            "stp x21, x22, [sp, #-16]!",
            "stp x23, x24, [sp, #-16]!",
            "stp x25, x26, [sp, #-16]!",
            "stp x27, x28, [sp, #-16]!",
            "stp x29, x30, [sp, #-16]!",
            "stp d8,  d9,  [sp, #-16]!",
            "stp d10, d11, [sp, #-16]!",
            "stp d12, d13, [sp, #-16]!",
            "stp d14, d15, [sp, #-16]!",
            "mov x2, sp",
            "str x2, [x0]", // *current_sp = sp
            "mov sp, x1",   // sp = new_sp
            "ldp d14, d15, [sp], #16",
            "ldp d12, d13, [sp], #16",
            "ldp d10, d11, [sp], #16",
            "ldp d8,  d9,  [sp], #16",
            "ldp x29, x30, [sp], #16",
            "ldp x27, x28, [sp], #16",
            "ldp x25, x26, [sp], #16",
            "ldp x23, x24, [sp], #16",
            "ldp x21, x22, [sp], #16",
            "ldp x19, x20, [sp], #16",
            "ret",
        );
    }

    /// First-switch landing pad: entry in `x19`, arg in `x20` (restored by
    /// `switch_to`'s `ldp`).
    #[unsafe(naked)]
    unsafe extern "C" fn trampoline() -> ! {
        naked_asm!("mov x0, x20", "blr x19", "bl {exit}", exit = sym crate::sched::exit_current_task);
    }

    /// Build the first-switch frame and return the initial `sp`. The 20 saved
    /// slots mirror `switch_to`'s restore order (lowest address first):
    /// d14,d15,d12,d13,d10,d11,d8,d9, x29,x30, x27,x28,x25,x26,x23,x24,x21,x22,
    /// x19,x20 -- so `x30`=trampoline, `x19`=entry, `x20`=arg.
    ///
    /// # Safety
    /// `stack_top` must top a fresh, owned, 16-byte-aligned stack >= 160 bytes.
    pub unsafe fn init_stack(stack_top: u64, entry: extern "C" fn(u64), arg: u64) -> u64 {
        // SAFETY: caller guarantees `stack_top` sits atop an owned region.
        unsafe {
            let mut sp = stack_top as *mut u64;
            let mut push = |value: u64| {
                sp = sp.sub(1);
                sp.write(value);
            };
            // Highest address first (reverse of the restore order).
            push(arg); // x20
            push(entry as *const () as u64); // x19
            push(0); // x22
            push(0); // x21
            push(0); // x24
            push(0); // x23
            push(0); // x26
            push(0); // x25
            push(0); // x28
            push(0); // x27
            push(trampoline as *const () as u64); // x30 (return address)
            push(0); // x29
            push(0); // d9
            push(0); // d8
            push(0); // d11
            push(0); // d10
            push(0); // d13
            push(0); // d12
            push(0); // d15
            push(0); // d14
            sp as u64
        }
    }
}
