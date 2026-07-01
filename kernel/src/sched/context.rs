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

use core::arch::naked_asm;

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
