//! **Surviving a kernel fault in the shell task**: a recovery landmark the REPL
//! arms and a fault handler jumps back to, instead of halting the machine.
//!
//! # Why this exists
//!
//! A CPU fault in a *task* is already contained — [`crate::sched::fault_current_task`]
//! kills it and the machine keeps going. But it deliberately refuses for **task
//! 0, the shell**, on the grounds that isolating a fault there would trade a halt
//! for a machine with no user interface. That is the right call when the
//! alternative is "kill the shell"; it is the wrong one when the alternative is
//! "go back to the prompt", which is what this module makes possible.
//!
//! It matters most on hardware you cannot single-step. On a tethered Apple
//! Silicon boot a fault in a command costs the whole session — the framebuffer,
//! the USB console, the model that took a minute to load, and the state that
//! produced the bug — and the next iteration starts from a cold boot. Coming back
//! to the prompt with the syndrome printed turns that into a line of output.
//!
//! # What it is not
//!
//! **It is not error handling, and it never hides anything.** Every recovery
//! prints the full syndrome first and counts itself; the shell says it happened.
//! A fault is still a kernel bug, and a machine that quietly carried on after one
//! would be worse than one that stopped — the whole reason the fault is
//! interesting is that some invariant is already broken.
//!
//! # When it refuses
//!
//! Recovery abandons every stack frame between the fault and the REPL, so
//! anything those frames were in the middle of stays in the middle of it. Three
//! gates, each of which is a way that would go wrong ([`should_recover`] is the
//! pure decision, so the policy is testable):
//!
//! * **No landmark armed** — a fault before the shell ever ran (device bring-up,
//!   `mm::init`) has nowhere to go back to. Halt, as before.
//! * **A [`crate::mm::Locked`] is held** — the abandoned frames include the one
//!   that would release it, so the next taker spins forever with interrupts
//!   disabled: a machine that stopped with no output at all, which is strictly
//!   worse than halting with a syndrome. `mm::locks_held` is the signal.
//! * **Too many, too fast** — a fault that recurs immediately (a corrupt data
//!   structure the REPL touches on the way back to the prompt) would loop
//!   forever, printing. After [`MAX_RECOVERIES`] consecutive ones the answer is
//!   that this machine is not recoverable; halt and say so.
//!
//! # How it works
//!
//! `save_landmark` / `restore_landmark` are a two-function `setjmp`/`longjmp` in
//! `global_asm!`, per arch, saving only the callee-saved registers and the stack
//! pointer — the ABI's own contract, so nothing else needs to be preserved.
//! Jumping *down* the stack like this is sound in a way jumping up would not be:
//! the frames being abandoned are strictly above the landmark's SP and nothing
//! will return into them.
//!
//! The fault handler calls [`recover`] **from inside the exception handler**,
//! which is safe for the same reason `fault_current_task` is: these handlers run
//! on the faulting task's own stack (on x86 only `#DF` uses an IST), so the trap
//! frame is on the stack being abandoned and nothing will ever `iretq`/`eret`
//! from it. A CPU exception latches no in-service state to acknowledge.
//!
//! # What it does not cover, stated rather than discovered
//!
//! * **A fault inside an interrupt handler** abandons that handler without its
//!   EOI, so on a machine with a live interrupt controller that source can stop
//!   delivering — a dead timer, i.e. no preemption. A prompt with no preemption
//!   still beats a halted machine, so this is not gated on; it is worth knowing
//!   before concluding the timer broke on its own. (It cannot arise on the Apple
//!   path, which is cooperative and takes no interrupts at all.)
//! * **A fault in a task other than the shell** never reaches here:
//!   `fault_current_task` kills that task first, which is the better outcome.
//! * **Leaks.** Every allocation, open handle and borrow owned by the abandoned
//!   frames is lost. Bounded by the recovery count, and the reason the shell
//!   suggests a reboot once you have what you came for.

use core::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

/// Consecutive recoveries before this gives up and halts. Two is enough to
/// survive a bad command and its cleanup; a third in a row means the machine is
/// re-faulting on the way back to the prompt, which recovery cannot fix.
pub const MAX_RECOVERIES: usize = 8;

/// Callee-saved registers + SP + resume address, per arch. aarch64 needs
/// x19-x28, x29, x30, SP and d8-d15 (21 u64); x86-64 SysV needs rbx, rbp,
/// r12-r15, rsp and rip (8). One size for both — it is a handful of bytes and a
/// per-arch constant would be one more thing to keep in step.
const LANDMARK_WORDS: usize = 24;

/// The armed landmark. A single one, for a single recovery point: the shell REPL.
/// Not per task on purpose — every other task already has containment, and the
/// one that does not is the one there is exactly one of.
static LANDMARK: Landmark = Landmark {
    buf: [const { AtomicU64::new(0) }; LANDMARK_WORDS],
    armed: AtomicBool::new(false),
};

/// Consecutive recoveries; reset by [`disarm_reset`] once the shell has
/// completed a command without faulting.
static RECOVERIES: AtomicUsize = AtomicUsize::new(0);

/// Total recoveries this boot, for `/top`-style reporting and for the shell's
/// own "this happened" line.
static TOTAL: AtomicUsize = AtomicUsize::new(0);

/// Why the last recovery happened, as the handler described it.
static LAST_REASON: crate::mm::Locked<Option<&'static str>> = crate::mm::Locked::new(None);

/// Whether IRQs were unmasked in the frame that armed the landmark.
///
/// **Not the same as "enable them".** Exception entry masks IRQs, so a recovery
/// has to put the mask back — and blanket-enabling is wrong on the machine this
/// feature exists for: an Apple Silicon boot has no usable CPU interrupt
/// interface, so `context::initial_daif` masks IRQs for every task deliberately.
/// Unmasking there would let an interrupt reach a vector that dispatches into a
/// GIC that is not present, turning one fault into a fault loop. Recording the
/// landmark's own state and restoring exactly that keeps the recovered prompt
/// identical to the one that armed it.
static LANDMARK_IRQ_ON: AtomicBool = AtomicBool::new(false);

struct Landmark {
    buf: [AtomicU64; LANDMARK_WORDS],
    armed: AtomicBool,
}

// SAFETY: the buffer is only written by `save_landmark` (through a raw pointer,
// while `armed` is false) and only read by `restore_landmark` (while `armed` is
// true), both on the core that armed it.
unsafe impl Sync for Landmark {}

unsafe extern "C" {
    /// Save the callee-saved state into `buf`; returns 0 here, 1 when resumed.
    fn save_landmark(buf: *mut u64) -> u64;
    /// Restore `buf` and resume, as if [`save_landmark`] had returned 1.
    fn restore_landmark(buf: *const u64) -> !;
}

/// What [`arm`] tells its caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arrival {
    /// The landmark is now armed; this is the ordinary path through.
    Armed,
    /// Arrived here by recovering from a fault, with the handler's description.
    Recovered(&'static str),
}

/// Arm the recovery landmark at the caller's frame, or report that we have just
/// come back to it from a fault.
///
/// The caller must be a frame it is safe to return to repeatedly — in practice
/// the top of the REPL loop, above any per-command state. Everything the
/// abandoned frames owned is leaked; that is the price and it is bounded by the
/// recovery count.
pub fn arm() -> Arrival {
    // **The landmark is this call**, so a recovery resumes right here — inside
    // `arm_here`, which returns `true` the second time exactly as `setjmp` does.
    // Anything else (arming in one place and testing a flag in another) reports
    // the arrival at the wrong frame: the jump lands where `save_landmark` was
    // called from and nowhere else.
    if !arm_here() {
        return Arrival::Armed;
    }
    Arrival::Recovered(LAST_REASON.with(|r| r.take()).unwrap_or("kernel fault"))
}

/// Save the caller's frame as the landmark. Returns `false` on the way through
/// and `true` when [`recover`] has jumped back to it.
///
/// Everything that must survive the jump lives in memory rather than in a local:
/// a local written between the save and the jump has an indeterminate value
/// afterwards (it may have been in a callee-saved register that the restore just
/// rewound), which is the oldest `setjmp` trap there is.
fn arm_here() -> bool {
    let mut scratch = [0u64; LANDMARK_WORDS];
    // SAFETY: `scratch` is a live, correctly sized buffer on this stack.
    let resumed = unsafe { save_landmark(scratch.as_mut_ptr()) };
    if resumed != 0 {
        return true;
    }
    for (slot, v) in LANDMARK.buf.iter().zip(scratch.iter()) {
        slot.store(*v, Ordering::Relaxed);
    }
    LANDMARK_IRQ_ON.store(crate::arch::interrupts::are_enabled(), Ordering::Relaxed);
    LANDMARK.armed.store(true, Ordering::Release);
    false
}

/// Forget the landmark (no recovery until the next [`arm`]) and reset the
/// consecutive-fault count. The shell calls this once a command has completed
/// normally: a fault that happens *after* a clean command is a fresh problem, not
/// the continuation of a storm.
pub fn disarm_reset() {
    RECOVERIES.store(0, Ordering::Relaxed);
}

/// Whether a fault should be recovered rather than halted.
///
/// Pure, so the policy can be tested without faulting: `armed` is whether a
/// landmark exists, `locks_held` is [`crate::mm::locks_held`] at the moment of
/// the fault, and `consecutive` is how many recoveries have happened since the
/// last clean command.
pub fn should_recover(armed: bool, locks_held: usize, consecutive: usize) -> bool {
    armed && locks_held == 0 && consecutive < MAX_RECOVERIES
}

/// Why a recovery was refused, in words, for the handler to print before halting.
/// `None` when it was not refused.
pub fn refusal(armed: bool, locks_held: usize, consecutive: usize) -> Option<&'static str> {
    if !armed {
        Some("no recovery point is armed (the fault is before or outside the shell's command loop)")
    } else if locks_held != 0 {
        Some("a kernel lock is held, so unwinding would leave it locked forever")
    } else if consecutive >= MAX_RECOVERIES {
        Some("too many faults in a row; the machine is re-faulting on the way back to the prompt")
    } else {
        None
    }
}

/// Recoveries so far this boot.
pub fn total() -> usize {
    TOTAL.load(Ordering::Relaxed)
}

/// Consecutive recoveries since the last clean command.
pub fn consecutive() -> usize {
    RECOVERIES.load(Ordering::Relaxed)
}

/// Whether a landmark is currently armed.
pub fn is_armed() -> bool {
    LANDMARK.armed.load(Ordering::Acquire)
}

/// Jump back to the armed landmark, reporting `why` at the prompt. **Never
/// returns.** Callers must have checked [`should_recover`] first — this asserts
/// nothing, because the caller is an exception handler with no better option
/// than the one it already decided against.
///
/// # Safety
/// Only from a fault handler running on the faulting task's own stack, with the
/// landmark armed and no kernel lock held. The frames between here and the
/// landmark are abandoned without unwinding.
pub unsafe fn recover(why: &'static str) -> ! {
    RECOVERIES.fetch_add(1, Ordering::Relaxed);
    TOTAL.fetch_add(1, Ordering::Relaxed);
    // `try_with`, never `with`: this runs inside a fault handler, and blocking on
    // a lock here is the deadlock this module exists to avoid. A dropped reason
    // costs a vaguer message, not the recovery.
    LAST_REASON.try_with(|r| *r = Some(why));
    let mut buf = [0u64; LANDMARK_WORDS];
    for (v, slot) in buf.iter_mut().zip(LANDMARK.buf.iter()) {
        *v = slot.load(Ordering::Relaxed);
    }
    // Exception entry masked IRQs. Put back exactly what the landmark's frame
    // had — see `LANDMARK_IRQ_ON` for why "just enable them" is wrong on the
    // machine that needs this most.
    if LANDMARK_IRQ_ON.load(Ordering::Relaxed) {
        crate::arch::interrupts::enable();
    }
    // SAFETY: the caller has established that the landmark is armed, so `buf`
    // holds a frame that `save_landmark` wrote and whose stack is still live
    // (it is below this one).
    unsafe { restore_landmark(buf.as_ptr()) }
}

#[cfg(target_arch = "aarch64")]
core::arch::global_asm!(
    r#"
.global save_landmark
save_landmark:
    stp x19, x20, [x0, #0]
    stp x21, x22, [x0, #16]
    stp x23, x24, [x0, #32]
    stp x25, x26, [x0, #48]
    stp x27, x28, [x0, #64]
    stp x29, x30, [x0, #80]
    mov x1, sp
    str x1, [x0, #96]
    stp d8,  d9,  [x0, #104]
    stp d10, d11, [x0, #120]
    stp d12, d13, [x0, #136]
    stp d14, d15, [x0, #152]
    mov x0, #0
    ret

// **Every load happens before `sp` moves.** The buffer lives on the *faulting*
// stack, which is below the landmark's `sp`; once `sp` is restored, an interrupt
// pushes its frame right over that buffer. Reading d8-d15 after the switch —
// the obvious ordering, since it mirrors the save — is a race that only shows up
// as corrupted floating-point state some time later.
.global restore_landmark
restore_landmark:
    ldp d8,  d9,  [x0, #104]
    ldp d10, d11, [x0, #120]
    ldp d12, d13, [x0, #136]
    ldp d14, d15, [x0, #152]
    ldp x19, x20, [x0, #0]
    ldp x21, x22, [x0, #16]
    ldp x23, x24, [x0, #32]
    ldp x25, x26, [x0, #48]
    ldp x27, x28, [x0, #64]
    ldp x29, x30, [x0, #80]
    ldr x1, [x0, #96]
    mov sp, x1
    mov x0, #1
    ret
"#
);

#[cfg(target_arch = "x86_64")]
core::arch::global_asm!(
    r#"
.global save_landmark
save_landmark:
    mov [rdi + 0],  rbx
    mov [rdi + 8],  rbp
    mov [rdi + 16], r12
    mov [rdi + 24], r13
    mov [rdi + 32], r14
    mov [rdi + 40], r15
    // The SP the caller will have *after* this function returns, i.e. past the
    // return address `call` pushed — restoring the SP we are standing on would
    // put the resumed frame one slot too low and leave the stack misaligned.
    lea rax, [rsp + 8]
    mov [rdi + 48], rax
    mov rax, [rsp]
    mov [rdi + 56], rax
    xor eax, eax
    ret

// The resume address is loaded **before** `rsp` moves: the buffer sits on the
// faulting stack, below the landmark's `rsp`, so once `rsp` is restored an
// interrupt frame lands on top of it.
.global restore_landmark
restore_landmark:
    mov rbx, [rdi + 0]
    mov rbp, [rdi + 8]
    mov r12, [rdi + 16]
    mov r13, [rdi + 24]
    mov r14, [rdi + 32]
    mov r15, [rdi + 40]
    mov rdx, [rdi + 56]
    mov rsp, [rdi + 48]
    mov eax, 1
    jmp rdx
"#
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn recovery_needs_a_landmark_no_locks_and_a_budget() {
        assert!(should_recover(true, 0, 0));
        assert!(should_recover(true, 0, MAX_RECOVERIES - 1));
        // Each gate refuses on its own.
        assert!(!should_recover(false, 0, 0), "nothing to return to");
        assert!(!should_recover(true, 1, 0), "a held lock would never be released");
        assert!(!should_recover(true, 0, MAX_RECOVERIES), "a fault storm");
    }

    #[test_case]
    fn every_refusal_says_which_one_it_was() {
        // The refusals are what a person reads off a serial log on a machine
        // they cannot debug, so each gate must be distinguishable — and a
        // permitted recovery must report nothing at all.
        assert!(refusal(true, 0, 0).is_none());
        let a = refusal(false, 0, 0).expect("unarmed is refused");
        let b = refusal(true, 2, 0).expect("locked is refused");
        let c = refusal(true, 0, MAX_RECOVERIES).expect("a storm is refused");
        assert_ne!(a, b);
        assert_ne!(b, c);
        assert_ne!(a, c);
        // The most dangerous gate is checked before the least: a fault that is
        // both a storm and lock-holding must report the lock.
        assert_eq!(refusal(true, 1, MAX_RECOVERIES), Some(b));
    }

    /// Counted across the jump, so it lives in memory: a local would be
    /// indeterminate afterwards, which is the very trap `arm_here` documents —
    /// and a test that tripped it would fail for a reason unrelated to the asm.
    static PASSES: AtomicUsize = AtomicUsize::new(0);

    #[test_case]
    fn a_landmark_round_trips_through_the_arch_asm() {
        // The one thing no amount of policy testing covers: that save/restore
        // actually work on this arch. Saving returns 0; restoring resumes at the
        // same place with 1, on a stack pointer good enough to keep running on.
        PASSES.store(0, Ordering::Relaxed);
        let mut buf = [0u64; LANDMARK_WORDS];
        // SAFETY: `buf` is live for both calls and `restore_landmark` resumes
        // inside this frame, which is still on the stack.
        let first = unsafe { save_landmark(buf.as_mut_ptr()) };
        PASSES.fetch_add(1, Ordering::Relaxed);
        if first == 0 {
            assert_eq!(PASSES.load(Ordering::Relaxed), 1);
            // SAFETY: as above.
            unsafe { restore_landmark(buf.as_ptr()) };
        }
        assert_eq!(first, 1, "the second arrival reports a resume");
        assert_eq!(PASSES.load(Ordering::Relaxed), 2, "and it re-ran the code after the save");
        // Still able to call, allocate and return on the restored stack — a
        // misaligned or off-by-one SP passes the assertions above and then
        // crashes in the *next* test, which is the worst way to find it.
        let v: alloc::vec::Vec<u64> = (0..64).collect();
        assert_eq!(v.iter().sum::<u64>(), 2016);
    }
}
