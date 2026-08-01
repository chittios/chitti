//! EL0 entry: the aarch64 counterpart of x86's `fastcall`.
//!
//! Same split as there — this is *transport*, and the authority half is
//! [`crate::synapse::abi`]. Same shape too: a tenant runs until it executes `svc`,
//! the trap is dispatched on the kernel stack, and then it is `eret`-ed back with
//! the reply length in x0. Control returns to the kernel context that entered EL0
//! only on a deliberate [`crate::synapse::abi::Entry::Exit`] or on a fault.
//!
//! # Why this is shorter than the x86 path
//!
//! Because AArch64 switches the stack for you. Taking an exception to EL1h makes
//! `SP` become `SP_EL1`, which still holds whatever the kernel left there when it
//! `eret`-ed — so the handler is already on a kernel stack, below the frame of the
//! function that entered EL0, and may `bl` straight into Rust. x86's `syscall`
//! does the opposite: it leaves `RSP` pointing at *user* memory, which is why
//! `fastcall::syscall_entry` has to do a RIP-relative stack switch before it may
//! touch anything. Two designs, and the AArch64 one simply has less to get wrong.
//!
//! # The bits that do have to be right
//!
//! `eret` takes its target from `ELR_EL1` (address) and `SPSR_EL1` (the PSTATE to
//! restore, *including which EL*). `SPSR_EL1.M[3:0] = 0b0000` means EL0t — EL0
//! using `SP_EL0` — so the user stack goes in `SP_EL0`, not `SP`. Writing it to
//! `SP` instead would return to EL0 with the kernel's stack pointer, which is a
//! tenant holding a pointer into kernel memory.
//!
//! `DAIF` is masked in the pushed `SPSR` for the same reason x86 clears `IF`: the
//! first crossing should not have a timer arriving in it, so a failure is the
//! transition and nothing else.

use crate::mm::space::AddressSpace;

/// What a tenant's `svc` carried. Mirrors `fastcall::Trapped` so the neutral
/// facade in [`crate::arch`] can hand back one type.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Trapped {
    pub number: u64,
    pub arg0: u64,
    pub arg1: u64,
}

/// Kernel `SP`/`LR` to resume when a tenant traps. Named (`no_mangle`) because
/// the vector stub addresses them with `adrp`/`:lo12:`, which is PC-relative and
/// so survives the kernel's `.rela.dyn` self-relocation.
#[unsafe(no_mangle)]
pub static mut EL0_RESUME_SP: u64 = 0;
#[unsafe(no_mangle)]
pub static mut EL0_RESUME_LR: u64 = 0;

/// Non-zero when the vector stub should return to the kernel instead of `eret`-ing
/// back to the tenant. Written by the dispatcher, read by the stub — which is why it
/// is a memory flag rather than a return value: the stub has to decide *after* the
/// Rust call has already used x0 for the reply length.
#[unsafe(no_mangle)]
pub static mut EL0_EXIT_TO_KERNEL: u64 = 0;

/// Calls the current tenant has made. Zeroed by `enter_el0` immediately before the
/// crossing, so it measures one run.
static CALLS: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

/// How many calls the tenant made during the last (or current) run.
pub fn call_count() -> u64 {
    CALLS.load(core::sync::atomic::Ordering::Relaxed)
}

/// How a tenant left EL0.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Exit {
    /// It executed `svc` with these arguments.
    Svc(Trapped),
    /// It took a synchronous fault instead — a bad pointer, an unmapped or
    /// non-executable code page, an undefined instruction. `esr` is the syndrome
    /// and `far` the faulting address.
    ///
    /// Reported rather than halting, because the common cause is a mapping the
    /// *kernel* got wrong when setting the tenant up, and a caller that is told
    /// which address faulted can fix it. Once tenants are real tasks this is where
    /// `sched::fault_current_task` belongs.
    Fault { esr: u64, far: u64 },
}

static mut TRAPPED: Trapped = Trapped { number: 0, arg0: 0, arg1: 0 };
static mut LAST_EXIT: Exit = Exit::Svc(Trapped { number: 0, arg0: 0, arg1: 0 });

/// The tenant currently at EL0: its identity and its memory. Mirrors x86's
/// `fastcall::CUR_TASK`/`CUR_SPACE` for the same reasons — the task id is what the
/// capability gate checks and the address space is what a user pointer is
/// validated against, and a tenant supplying either would be choosing whose
/// authority to spend or which page tables the kernel reads through.
static mut CUR_TASK: crate::sched::TaskId = 0;
static mut CUR_SPACE: *const AddressSpace = core::ptr::null();

/// What the last dispatched call replied.
static LAST_REPLY: crate::mm::Locked<Option<alloc::string::String>> = crate::mm::Locked::new(None);

/// The reply text from the last tenant call, if any.
pub fn last_reply() -> Option<alloc::string::String> {
    LAST_REPLY.with(|r| r.clone())
}

/// Called by the `svc` vector stub, on the kernel stack, with the tenant's
/// registers as arguments.
///
/// Hands the call straight to [`crate::synapse::abi::dispatch`], which validates
/// the user pointer against the tenant's own page tables, copies the text in, and
/// runs the four gates.
///
/// **This is what makes the two arches equivalent.** For a while x86 dispatched
/// here and aarch64 only recorded, which is a standing-rule violation ("if a
/// capability exists on one arch, it exists on the other") rather than an
/// unfinished feature: a tenant on aarch64 could trap but never reach the gates,
/// so the same program had different authority depending on the machine.
#[unsafe(no_mangle)]
extern "C" fn aarch64_svc_dispatch(number: u64, arg0: u64, arg1: u64, out_ptr: u64, out_cap: u64) -> u64 {
    // SAFETY: written only from this path with DAIF masked; only one core runs
    // tenants (a user address space is pinned — no TLB shootdown on either arch).
    let (task, space) = unsafe { (CUR_TASK, CUR_SPACE) };
    CALLS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
    unsafe {
        TRAPPED = Trapped { number, arg0, arg1 };
        LAST_EXIT = Exit::Svc(TRAPPED);
        // Decide here, not in the stub: leave EL0 on a deliberate `Exit`, or when
        // there is no tenant registered to resume. Everything else resumes.
        EL0_EXIT_TO_KERNEL = u64::from(
            space.is_null() || crate::synapse::abi::Entry::from_raw(number) == Some(crate::synapse::abi::Entry::Exit),
        );
    }
    if space.is_null() {
        return 0; // no tenant registered: nothing to dispatch against
    }
    // SAFETY: set by `enter_el0` to a borrow that outlives the tenant's time at
    // EL0, and cleared before that borrow ends.
    let space = unsafe { &*space };
    // No register shuffle here, unlike x86: `svc` consumes nothing, so the tenant's
    // x0..x4 are already exactly the C ABI's first five argument registers.
    let reply = crate::synapse::abi::dispatch(task, space, number, arg0, arg1, out_ptr, out_cap);
    let (text, written) = match reply {
        crate::synapse::abi::Reply::Ok { text, written } => (text, written),
        crate::synapse::abi::Reply::Err(e) => (alloc::format!("abi-error:{e:?}"), 0),
        crate::synapse::abi::Reply::Exited => (alloc::string::String::from("exited"), 0),
    };
    LAST_REPLY.with(|r| *r = Some(text));
    written as u64
}

/// Called by the same vector stub when the exception was **not** an `svc`.
#[unsafe(no_mangle)]
extern "C" fn aarch64_el0_fault(esr: u64, far: u64) {
    // SAFETY: as above.
    unsafe {
        LAST_EXIT = Exit::Fault { esr, far };
        // Belt and braces: the stub's fault path returns to the kernel regardless,
        // but leaving a stale zero here would resume a faulted tenant if that path
        // ever grew a flag check.
        EL0_EXIT_TO_KERNEL = 1;
    }
}

/// What the last `svc` carried.
pub fn last_trap() -> Trapped {
    // SAFETY: plain read of a slot written only by `aarch64_svc_dispatch`.
    unsafe { TRAPPED }
}

/// How the tenant last left EL0 — `svc` or a fault.
pub fn last_exit() -> Exit {
    // SAFETY: plain read of a slot written only by the two dispatch functions.
    unsafe { LAST_EXIT }
}

/// Drop to EL0 at `entry_va` with stack `stack_va` in `space`, running as `task`,
/// and return how the tenant left.
///
/// # Safety
/// `entry_va` must be mapped user-executable in `space` and `stack_va`
/// user-writable; `space` must share the kernel mappings (which `AddressSpace`
/// guarantees). Not reentrant.
pub unsafe fn enter_el0(
    task: crate::sched::TaskId,
    space: &AddressSpace,
    entry_va: u64,
    stack_va: u64,
    arg: u64,
) -> Exit {
    let kernel_root = crate::mm::space::kernel_root();
    // SAFETY: caller's contract. DAIF stays masked across the crossing, and the
    // kernel half of `space` keeps this code, its stack and the resume slots
    // mapped — which is what makes switching TTBR0 mid-function sound.
    unsafe {
        crate::arch::interrupts::without_interrupts(|| {
            // The tenant's identity is a parameter, never `current_task_id()`: the
            // caller is the kernel, and checking the gates against the kernel's own
            // task would spend its capabilities on the tenant's behalf.
            CUR_TASK = task;
            CUR_SPACE = space as *const _;
            // Zeroed immediately before the crossing, so they describe this run.
            EL0_EXIT_TO_KERNEL = 0;
            CALLS.store(0, core::sync::atomic::Ordering::Relaxed);
            // Pinned for the same reason as on x86, and deliberately here in both
            // arches rather than inside `AddressSpace::activate`: the pin is a fact
            // about the *task*, and `activate` also runs for the kernel's own root.
            crate::sched::pin_to_boot_cpu(task);
            space.activate();
            core::arch::asm!(
                // **Save the callee-saved registers across the crossing.**
                // `clobber_abi("C")` covers only the caller-saved set, which is right
                // for calling a C function and wrong for entering EL0: a tenant is
                // userspace, honours no ABI, and may freely use x19-x29. Without this
                // the compiler keeps a live value in (say) x19 across the crossing, the
                // tenant overwrites it, and the kernel resumes using whatever userspace
                // left there — which presented as a data abort on the tenant's own
                // argument address, *inside this function*, long after the tenant was
                // gone. A blob that happens to avoid those registers hides it entirely,
                // which is exactly how it got this far.
                //
                // Saved rather than declared clobbered because x19 is reserved by LLVM
                // and cannot be an `asm!` operand at all.
                "sub sp, sp, #96",
                "stp x19, x20, [sp, #0]",
                "stp x21, x22, [sp, #16]",
                "stp x23, x24, [sp, #32]",
                "stp x25, x26, [sp, #48]",
                "stp x27, x28, [sp, #64]",
                "str x29, [sp, #80]",
                // Park the resume point *after* the saves, so the stub's `mov sp, x10`
                // lands with them still on the stack and the pops below find them.
                "adr x9, 1f",
                "adrp x10, EL0_RESUME_LR",
                "add  x10, x10, :lo12:EL0_RESUME_LR",
                "str  x9, [x10]",
                "mov  x9, sp",
                "adrp x10, EL0_RESUME_SP",
                "add  x10, x10, :lo12:EL0_RESUME_SP",
                "str  x9, [x10]",
                // The tenant's stack goes in SP_EL0, because SPSR.M says EL0t.
                // The startup argument, moved in last from a compiler-chosen register.
                // **Not `in("x0")`**: this block also declares `clobber_abi("C")`,
                // which covers x0, and relying on how those two constraints interact
                // is exactly the kind of assumption that silently misplaces an operand.
                "mov x0, {uarg}",
                "msr sp_el0, {ustack}",
                "msr elr_el1, {uentry}",
                // SPSR_EL1: M[3:0]=0 (EL0t) and DAIF masked (bits 9:6).
                "mov x9, #0x3c0",
                "msr spsr_el1, x9",
                "isb",
                "eret",
                "1:",
                "ldp x19, x20, [sp, #0]",
                "ldp x21, x22, [sp, #16]",
                "ldp x23, x24, [sp, #32]",
                "ldp x25, x26, [sp, #48]",
                "ldp x27, x28, [sp, #64]",
                "ldr x29, [sp, #80]",
                "add sp, sp, #96",
                ustack = in(reg) stack_va,
                uentry = in(reg) entry_va,
                uarg = in(reg) arg,
                out("x9") _,
                out("x10") _,
                // **Every callee-saved register is clobbered too.** `clobber_abi("C")`
                // covers only the caller-saved set, which is correct for calling a C
                // function and *wrong* for entering EL0: a tenant is userspace, it
                // honours no ABI, and it may freely use x19-x28. Without these the
                // compiler happily keeps a live value in x19 across the crossing, the
                // tenant overwrites it, and the kernel resumes using whatever userspace
                // left there — which presented as a data abort on the tenant's own
                // argument address, inside this function, long after the tenant was
                // gone. A blob that avoids x19 (the first one did) hides it completely.
                clobber_abi("C"),
            );
            // Back at EL1 on the kernel stack; the tenant's TTBR0 is still live,
            // so restore ours before anything else runs.
            crate::mm::space::activate_kernel(kernel_root);
            // Clear before the borrow ends, so a later stray `svc` cannot be
            // dispatched against a dangling address space.
            CUR_SPACE = core::ptr::null();
            last_exit()
        })
    }
}

/// Prove an EL0 round trip works on *this* machine, once, at boot.
///
/// The aarch64 unit suite does not exist — `cargo xtask test` is x86 only — so
/// unlike the x86 path this cannot be covered by a `#[test_case]`. A boot
/// self-test is the available equivalent, and the same one `mmu::walker_self_test`
/// uses for the same reason: a facility that silently does not work is worse than
/// one that says so at boot.
///
/// Returns `Err` with a short reason rather than panicking; the caller ktraces it.
pub fn self_test() -> Result<(), &'static str> {
    use crate::mm::space::{self, UserPerms};
    // **Three calls, not one — resumption is the part that can silently not work.**
    // Entry 7 is unknown, so the ABI refuses it and the tenant resumes; a broken
    // `eret` path shows up as one call and a stuck or faulting tenant rather than as
    // a wrong answer. The final `Exit` carries arguments so the x1/x2 -> arg0/arg1
    // plumbing is checked on this arch too, and it has to re-load them: the `bl` in
    // the vector stub clobbers the caller-saved registers, which is the contract.
    const BLOB: &[u8] = &[
        0xe0, 0x00, 0x80, 0xd2, // mov x0, #7   (unknown entry: refused, then resume)
        0x01, 0x00, 0x80, 0xd2, // mov x1, #0
        0x02, 0x00, 0x80, 0xd2, // mov x2, #0
        0x01, 0x00, 0x00, 0xd4, // svc #0
        0xe0, 0x00, 0x80, 0xd2, // mov x0, #7   (again, proving the tenant resumed)
        0x01, 0x00, 0x00, 0xd4, // svc #0
        0x40, 0x00, 0x80, 0xd2, // mov x0, #2   (Exit)
        0xa1, 0x00, 0x80, 0xd2, // mov x1, #5
        0x22, 0x01, 0x80, 0xd2, // mov x2, #9
        0x01, 0x00, 0x00, 0xd4, // svc #0
    ];
    // A throwaway identity: the gates must check the *tenant's* authority, not the
    // kernel task running the self-test.
    let tenant = crate::sched::spawn_parked("el0-selftest-tenant");
    let mut sp = AddressSpace::new().ok_or("no address space")?;
    let code_va = space::USER_BASE;
    let code_phys = sp.map_new_page(code_va, UserPerms::RX).map_err(|_| "map code")?;
    // SAFETY: a frame this space owns, reachable by the kernel (VA == PA).
    unsafe {
        core::ptr::copy_nonoverlapping(BLOB.as_ptr(), space::phys_to_kernel(code_phys) as *mut u8, BLOB.len());
    }
    let stack_va = code_va + 0x1000;
    sp.map_new_page(stack_va, UserPerms::RW).map_err(|_| "map stack")?;
    // SAFETY: both pages are mapped with the permissions `enter_el0` requires.
    let got = unsafe { enter_el0(tenant, &sp, code_va, stack_va + 0x1000 - 16, 0) };
    let _ = crate::sched::kill(tenant);
    if call_count() != 3 {
        // Distinguished from a wrong-arguments failure because the cause is
        // different: this is the `eret` resume path, not argument marshalling.
        return Err("the tenant did not resume across all three calls");
    }
    match got {
        Exit::Svc(t) if t == (Trapped { number: 2, arg0: 5, arg1: 9 }) => Ok(()),
        Exit::Svc(_) => Err("svc arrived with the wrong arguments"),
        // Distinguishable from a bad `svc` on purpose: a fault here almost always
        // means the kernel mapped the tenant's pages wrong, and saying so beats
        // reporting "the syscall was odd".
        Exit::Fault { .. } => Err("the tenant faulted instead of calling svc"),
    }
}
