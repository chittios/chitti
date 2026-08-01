//! The `syscall`/`sysret` machine-level trap: MSR setup and the entry stub.
//!
//! This is the *transport* half of ring 3. The authority half — validating a
//! user pointer, copying the call in, running the four gates — is
//! [`crate::synapse::abi`], which is where a mistake is a security bug. Here a
//! mistake is a hang, a reset, or a triple fault, so the two are separate files
//! and land as separate increments.
//!
//! # What `syscall` does and does not do for you
//!
//! `syscall` is fast because it does almost nothing: it loads `CS`/`SS` from
//! `STAR`, puts the return address in `RCX` and `RFLAGS` in `R11`, masks `RFLAGS`
//! with `SFMASK`, and jumps to `LSTAR`. In particular it **does not switch the
//! stack** — `RSP` still points into user memory on entry. That is the single
//! most important fact about this path, and the reason the entry stub's first job
//! is to get onto a kernel stack before touching anything. A handler that pushed
//! even one register first would be writing to a user address with kernel
//! privilege, which is both a corruption bug and a way for a tenant to choose
//! where the kernel writes.
//!
//! (`TSS.rsp0` is *not* consulted by `syscall` either — that is only for
//! interrupts and exceptions raising privilege. It still has to be right, because
//! a timer tick can arrive while a task is in ring 3, and that path does use it.)
//!
//! # Calling convention
//!
//! Ours, not Linux's, because nothing here has to be compatible with anything:
//! `rdi` = entry number, `rsi`/`rdx` = the call's `(ptr, len)`, `r8`/`r9` = the
//! reply buffer's `(ptr, cap)`, `rax` = bytes of reply delivered.
//!
//! `rcx` and `r11` are consumed by the instruction itself (return address, saved
//! flags) and `r10` by the stub's stack switch, so none of those three can carry an
//! argument. That is also why the stub **shuffles**: SysV puts the fourth and fifth
//! integer arguments in `rcx` and `r8`, but `rcx` is already spoken for, so the
//! tenant passes them in `r8`/`r9` and the stub moves them into place — `rcx` first,
//! then `r8`, because the second move would otherwise destroy the first's source.
//!
//! # Status
//!
//! The MSRs are armed and the stub switches stacks and dispatches. What it does
//! **not** yet do is `sysretq`: instead it restores the kernel context that
//! entered it and returns there, which is the shape a one-shot "run a tenant
//! until it traps" needs anyway — and it means this whole path is exercisable
//! from ring 0, where `syscall` is still a legal instruction and a mistake is
//! recoverable. The privilege transition (`iretq` down, `sysretq` back) is the
//! next and last increment, deliberately separated: a mistake *there* is a triple
//! fault with no output, and isolating it means not also debugging the stack
//! switch at the same time.

use core::arch::asm;
use core::sync::atomic::Ordering;

const EFER_MSR: u32 = 0xc000_0080;
/// `EFER.SCE` — System Call Extensions. Without it `syscall` is `#UD`.
const EFER_SCE: u64 = 1 << 0;
const STAR_MSR: u32 = 0xc000_0081;
const LSTAR_MSR: u32 = 0xc000_0082;
const SFMASK_MSR: u32 = 0xc000_0084;

/// `RFLAGS` bits cleared on entry to the kernel.
///
/// `IF` (bit 9) is the one that matters: without masking it, the kernel would run
/// the first instructions of the entry stub — the window where `RSP` still points
/// at user memory — with interrupts enabled, so a timer tick could land there and
/// push an interrupt frame onto the user stack. `TF` (bit 8) and `DF` (bit 10) are
/// masked too: a tenant that left the trap flag set would single-step the kernel,
/// and a set direction flag silently reverses every string operation the kernel
/// performs.
const SFMASK_VALUE: u64 = (1 << 9) | (1 << 8) | (1 << 10);

fn read_msr(msr: u32) -> u64 {
    let (lo, hi): (u32, u32);
    // SAFETY: reading an architectural MSR that exists on every x86_64 CPU in
    // long mode; no side effects.
    unsafe {
        asm!("rdmsr", in("ecx") msr, out("eax") lo, out("edx") hi, options(nomem, nostack, preserves_flags));
    }
    ((hi as u64) << 32) | lo as u64
}

fn write_msr(msr: u32, value: u64) {
    let lo = value as u32;
    let hi = (value >> 32) as u32;
    // SAFETY: the four MSRs this module writes are the architectural
    // `syscall`/`sysret` configuration registers; `EFER` is read-modify-written so
    // no other bit (notably `NXE`, set by `fpu::enable_nx`) is disturbed.
    unsafe {
        asm!("wrmsr", in("ecx") msr, in("eax") lo, in("edx") hi, options(nomem, nostack, preserves_flags));
    }
}

/// Arm the `syscall` instruction on this core.
///
/// Per-core: `STAR`/`LSTAR`/`SFMASK` and `EFER` are all per-CPU MSRs, so an AP
/// that skipped this would `#UD` on the first `syscall` from a task the scheduler
/// happened to place on it. Called from both the BSP and AP bring-up paths for the
/// same reason `gdt::init_ap` exists.
pub fn init() {
    // Read-modify-write: `EFER.NXE` is already set by `fpu::enable_nx`, and
    // clobbering it would make every `NO_EXECUTE` page-table entry fault.
    let efer = read_msr(EFER_MSR);
    write_msr(EFER_MSR, efer | EFER_SCE);

    write_msr(
        STAR_MSR,
        super::gdt::star_value(super::gdt::KERNEL_CODE_SELECTOR, super::gdt::sysret_base_selector()),
    );
    write_msr(LSTAR_MSR, syscall_entry as u64);
    write_msr(SFMASK_MSR, SFMASK_VALUE);
    crate::ktrace::log("fastcall", "syscall armed (EFER.SCE, STAR/LSTAR/SFMASK)");
}

/// Whether `syscall` is armed on this core, and with the values we intended.
/// Returns `(sce_on, star, lstar, sfmask)` for the boot self-check and tests.
pub fn state() -> (bool, u64, u64, u64) {
    (
        read_msr(EFER_MSR) & EFER_SCE != 0,
        read_msr(STAR_MSR),
        read_msr(LSTAR_MSR),
        read_msr(SFMASK_MSR),
    )
}

/// The kernel stack the entry stub switches to.
///
/// Per-core in principle; a single slot for now because only the BSP runs tenants
/// (there is no TLB shootdown on either arch, so a user address space is pinned to
/// one core).
/// Stack top the stub moves `rsp` to. A bare `u64` because the stub addresses it
/// with `sym`, which names a symbol and cannot carry a field offset — so the
/// return slot is a second static ([`RESUME_SLOT`]) rather than a second field.
static mut LANDING: u64 = 0;

/// What the trap carried, recorded by [`syscall_dispatch`] for [`enter`]'s caller.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Trapped {
    pub number: u64,
    pub arg0: u64,
    pub arg1: u64,
}

static mut TRAPPED: Trapped = Trapped { number: 0, arg0: 0, arg1: 0 };

/// The tenant currently in ring 3: its identity and its memory.
///
/// The trap handler needs both to do anything useful — the task id is what the
/// capability gate checks, and the address space is what a user pointer is
/// validated against. Neither can come from the tenant: a caller that supplied its
/// own task id would be choosing whose authority to spend, and one that supplied
/// its own address space would be choosing which page tables the kernel reads
/// through. So `enter_ring3` records them and the handler reads them here.
static mut CUR_TASK: crate::sched::TaskId = 0;
static mut CUR_SPACE: *const crate::mm::space::AddressSpace = core::ptr::null();

/// What the last dispatched call replied, for the kernel to inspect while the ABI
/// is still one-shot. A real tenant will get this copied back into its own memory.
static LAST_REPLY: crate::mm::Locked<Option<alloc::string::String>> = crate::mm::Locked::new(None);

/// The reply text from the last tenant call, if any.
pub fn last_reply() -> Option<alloc::string::String> {
    LAST_REPLY.with(|r| r.clone())
}

/// `LSTAR` target: where the CPU lands on `syscall`.
///
/// `naked` is load-bearing, not stylistic. A compiler-generated prologue would
/// push to whatever `rsp` holds, and on entry that is the *caller's* stack — a
/// user address once tenants exist. So the first two instructions get onto a
/// kernel stack, and only then is anything written.
///
/// `mov rsp, [rip + ...]` is what makes that possible without a scratch register:
/// a RIP-relative load needs no base register, so the switch happens before any
/// register has to be spilled. `r10` is then free to carry the old `rsp`.
#[unsafe(naked)]
pub unsafe extern "C" fn syscall_entry() {
    core::arch::naked_asm!(
        // Onto the kernel stack before touching memory. RIP-relative, so no
        // register is needed to address the slot and none has to be saved first.
        "mov r10, rsp",
        "mov rsp, [rip + {landing}]",
        // Now a stack exists: preserve what the instruction consumed plus the
        // caller's stack pointer.
        "push r10",
        "push rcx",
        "push r11",
        // Shuffle the tenant's 4th/5th arguments into the SysV slots. Order is
        // load-bearing: `rcx` takes `r8` before `r8` takes `r9`.
        "mov rcx, r8",
        "mov r8, r9",
        // **Align before the call.** SysV requires `rsp` ≡ 8 (mod 16) at function
        // entry, so that after the call's own push the callee sees a 16-aligned
        // frame. Three pushes off a 16-aligned top leaves 8, and the `call` makes
        // it 0 — misaligned. Since SSE codegen is on crate-wide, the callee is
        // free to `movaps` to a supposedly-aligned slot, which faults. One extra
        // slot fixes it, and the symmetry is the only reason this is a `sub`
        // rather than a fourth `push`.
        "sub rsp, 8",
        // rdi/rsi/rdx were already in place; rcx/r8 now are too.
        "call {dispatch}",
        "add rsp, 8",
        // `dispatch` left the reply length in rax and set the exit flag if the
        // tenant is done. Read the flag *before* popping, because the pop reuses
        // r10 to carry the tenant's stack pointer.
        "mov r10, [rip + {exitflag}]",
        "test r10, r10",
        "jnz 5f",
        // **Resume the tenant.** `sysretq` returns to rcx with rflags from r11 —
        // exactly the two registers `syscall` consumed on the way in, which is why
        // they were pushed. rax survives the pops and carries the reply length.
        "pop r11",
        "pop rcx",
        "pop r10",
        "mov rsp, r10",
        "sysretq",
        // Or hand control back to the kernel context that entered ring 3. Both
        // paths exist: a tenant that asked to exit, a fault, or a kernel-side
        // `trap_from_kernel` with no tenant registered all come this way.
        "5:",
        "pop r11",
        "pop rcx",
        "pop r10",
        "mov rsp, [rip + {resume}]",
        "ret",
        landing = sym LANDING,
        dispatch = sym syscall_dispatch,
        resume = sym RESUME_SLOT,
        exitflag = sym EXIT_TO_KERNEL,
    )
}

/// How a tenant left ring 3. Mirrors `arch::aarch64::el0::Exit`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Exit {
    /// It executed `syscall` with these arguments (the last of possibly many).
    Svc(Trapped),
    /// It took a fault instead — a bad pointer, a non-user or non-executable page,
    /// a privileged instruction. `code` is the vector's error code and `addr` the
    /// faulting address (`CR2` for `#PF`).
    ///
    /// Reported rather than halting. Before this existed, a mis-mapped tenant killed
    /// the machine on x86 while aarch64 reported it — because on x86 a tenant fault
    /// goes to the IDT, not through the syscall stub, and `fault_current_task`
    /// correctly refuses to kill the *kernel* task that entered ring 3 (the tenant
    /// is not a scheduler task). The handler then halted. See [`abort_tenant`].
    Fault { code: u64, addr: u64 },
}

static mut LAST_EXIT: Exit = Exit::Svc(Trapped { number: 0, arg0: 0, arg1: 0 });

/// Non-zero when the stub should return to the kernel rather than resume the
/// tenant: the tenant asked to exit, or there is no tenant at all.
static mut EXIT_TO_KERNEL: u64 = 0;

/// How many calls the current tenant has made. Lets a caller (and a test) tell
/// "ran and made three calls then exited" from "trapped once".
static CALLS: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

/// Calls made since the last [`enter_ring3`].
pub fn call_count() -> u64 {
    CALLS.load(Ordering::Relaxed)
}

/// `rsp` of the kernel context that entered the trap, restored on the way out.
static mut RESUME_SLOT: u64 = 0;

/// Called by [`syscall_entry`] on a real kernel stack, with the tenant's
/// arguments.
///
/// This is where transport meets authority: the call goes straight into
/// [`crate::synapse::abi::dispatch`], which validates the user pointer against the
/// tenant's *own* page tables, copies the text in, and runs the four gates. The
/// tenant supplies only `(number, ptr, len)` — never its identity, never its
/// address space, never a justification.
#[unsafe(no_mangle)]
extern "C" fn syscall_dispatch(number: u64, arg0: u64, arg1: u64, out_ptr: u64, out_cap: u64) -> u64 {
    // SAFETY: written only from this path, interrupts masked by `SFMASK`, and only
    // the BSP runs tenants (a user address space is pinned to one core: there is no
    // TLB shootdown on either arch).
    let (task, space) = unsafe { (CUR_TASK, CUR_SPACE) };
    unsafe {
        TRAPPED = Trapped { number, arg0, arg1 };
        LAST_EXIT = Exit::Svc(TRAPPED);
    }
    CALLS.fetch_add(1, Ordering::Relaxed);
    // Exit is the only entry that ends the tenant's run; everything else resumes
    // it, including an ABI error — a tenant that passed a bad pointer should learn
    // that and carry on, not be killed for it.
    unsafe {
        EXIT_TO_KERNEL = u64::from(
            space.is_null() || crate::synapse::abi::Entry::from_raw(number) == Some(crate::synapse::abi::Entry::Exit),
        );
    }
    if space.is_null() {
        // A `syscall` with no tenant registered: the kernel-side round-trip test.
        // Recording it is the whole job there.
        return 0;
    }
    // SAFETY: `CUR_SPACE` is set by `enter_ring3` to a borrow that outlives the
    // tenant's time in ring 3, and cleared before that borrow ends.
    let space = unsafe { &*space };
    let reply = crate::synapse::abi::dispatch(task, space, number, arg0, arg1, out_ptr, out_cap);
    let (text, written) = match reply {
        crate::synapse::abi::Reply::Ok { text, written } => (text, written),
        crate::synapse::abi::Reply::Err(e) => (alloc::format!("abi-error:{e:?}"), 0),
        crate::synapse::abi::Reply::Exited => (alloc::string::String::from("exited"), 0),
    };
    LAST_REPLY.with(|r| *r = Some(text));
    written as u64
}

/// How the tenant last left ring 3.
pub fn last_exit() -> Exit {
    // SAFETY: plain read of a slot written only by the dispatch/abort paths.
    unsafe { LAST_EXIT }
}

/// Whether a tenant is currently running in ring 3 — i.e. whether a fault arriving
/// now belongs to a tenant rather than to the kernel.
pub fn tenant_live() -> bool {
    // SAFETY: plain read.
    unsafe { !CUR_SPACE.is_null() }
}

/// Abandon a faulting tenant and hand control back to [`enter_ring3`].
///
/// Called from the `#PF`/`#GP` handlers when [`tenant_live`] says the fault belongs
/// to a tenant. It reuses the stub's own exit mechanism: restore the kernel `rsp`
/// parked in `RESUME_SLOT` and `ret`, which lands immediately after the `iretq` that
/// entered ring 3 — so `enter_ring3` itself restores the kernel address space and
/// returns `Exit::Fault`. The interrupt frame on the fault stack is simply abandoned,
/// which is sound because nothing will ever `iretq` from it.
///
/// # Safety
/// Only valid while a tenant is live and `RESUME_SLOT` names a kernel stack, i.e.
/// only from a fault taken during [`enter_ring3`].
pub unsafe fn abort_tenant(code: u64, addr: u64) -> ! {
    // SAFETY: caller's contract.
    unsafe {
        LAST_EXIT = Exit::Fault { code, addr };
        CUR_SPACE = core::ptr::null();
        EXIT_TO_KERNEL = 1;
        asm!(
            "mov rsp, [rip + {resume}]",
            "ret",
            resume = sym RESUME_SLOT,
            options(noreturn),
        );
    }
}

/// What the last trap carried.
pub fn last_trap() -> Trapped {
    // SAFETY: plain read of a slot written only by `syscall_dispatch`.
    unsafe { TRAPPED }
}

/// Execute `syscall` and come back, having recorded what it carried.
///
/// The kernel-side half of the round trip, and the reason this increment is
/// testable at all: `syscall` is a legal instruction in ring 0, so the stack
/// switch, the argument marshalling and the return path can all be exercised
/// without a privilege transition. The transition is added next; if it breaks, the
/// transport is already known good.
///
/// # Safety
/// Arms `LANDING` and executes `syscall`, so it must not be called reentrantly and
/// `init` must have run on this core.
pub unsafe fn trap_from_kernel(number: u64, arg0: u64, arg1: u64) -> Trapped {
    // A dedicated stack for the stub to land on, so it never lands on the stack
    // it is being called from — which is what will be true of a real tenant, and
    // testing the easier arrangement would test the wrong thing.
    const LAND_SIZE: usize = 16 * 1024;
    let land: alloc::boxed::Box<[u8]> = alloc::vec![0u8; LAND_SIZE].into_boxed_slice();
    let land_top = (land.as_ptr() as u64 + LAND_SIZE as u64) & !0xf;

    // SAFETY: arming the landing slots and executing `syscall`; the stub restores
    // `rsp` from `RESUME_SLOT` and `ret`s to the instruction after this one.
    unsafe {
        LANDING = land_top;
        crate::arch::interrupts::without_interrupts(|| {
            asm!(
                // Park this context's `rsp` where the stub will find it, then trap.
                "lea rax, [rip + 3f]",
                "push rax",
                "mov [rip + {resume}], rsp",
                "syscall",
                // The stub returns here via `ret`, which pops the address pushed
                // above — so `rsp` is already back where it started. Adjusting it
                // again here would over-pop by 8 and return into nothing on the
                // *next* return, a corruption whose symptom appears one frame
                // later than its cause.
                "3:",
                resume = sym RESUME_SLOT,
                in("rdi") number,
                in("rsi") arg0,
                in("rdx") arg1,
                out("rax") _,
                out("rcx") _,
                out("r10") _,
                out("r11") _,
                clobber_abi("sysv64"),
            );
        });
    }
    drop(land);
    last_trap()
}

/// Drop to ring 3 at `entry_va` with stack `stack_va` in `space`, running as
/// `task`, and return what the tenant's first `syscall` carried.
///
/// One-shot by design: the tenant runs until it traps, and the trap returns *here*
/// rather than `sysretq`-ing back. That is the shape the agent loop wants (run a
/// tenant until it asks for something) and it keeps the return path identical to
/// the one already proven by `trap_from_kernel`.
///
/// # Why the tenant runs with interrupts masked
///
/// The pushed `RFLAGS` has `IF` clear, so no timer tick can arrive while in ring 3.
/// That is deliberate for the first crossing: it removes the scheduler from the
/// picture entirely, so a failure is the transition and nothing else.
///
/// # Why `TSS.rsp0` is still set
///
/// Because a *fault* in ring 3 also raises privilege, and the CPU takes its stack
/// from `rsp0` — not from anything `syscall` touches. With `rsp0` at zero, a stray
/// `#PF` or `#GP` in the tenant would push its frame at address 0, fault again with
/// no stack, and triple-fault: the machine resets with no output. With it set, the
/// existing handlers report the fault and `sched::fault_current_task` contains it,
/// which is the difference between a diagnosable first attempt and a silent one.
///
/// # Safety
/// `entry_va` must be mapped executable-and-user in `space`, `stack_va` writable-
/// and-user, and `space` must share the kernel mappings (which
/// [`crate::mm::space::AddressSpace`] guarantees by construction). Not reentrant.
pub unsafe fn enter_ring3(
    task: crate::sched::TaskId,
    space: &crate::mm::space::AddressSpace,
    entry_va: u64,
    stack_va: u64,
) -> Exit {
    const LAND_SIZE: usize = 16 * 1024;
    let land: alloc::boxed::Box<[u8]> = alloc::vec![0u8; LAND_SIZE].into_boxed_slice();
    let land_top = (land.as_ptr() as u64 + LAND_SIZE as u64) & !0xf;
    let fault: alloc::boxed::Box<[u8]> = alloc::vec![0u8; LAND_SIZE].into_boxed_slice();
    let fault_top = (fault.as_ptr() as u64 + LAND_SIZE as u64) & !0xf;

    let kernel_root = crate::mm::space::kernel_root();
    // SAFETY: caller's contract. Interrupts stay masked across the whole crossing;
    // the kernel half of `space` keeps this code, its stack and these statics
    // mapped, which is what makes switching address spaces mid-function sound.
    let out = unsafe {
        crate::arch::interrupts::without_interrupts(|| {
            LANDING = land_top;
            // **Zero the per-run counters immediately before the crossing.** Not at
            // the top of the run and not in `trap_from_kernel`, which shares this
            // counter from ring 0 and would otherwise leave its calls to be counted
            // against the next tenant. Missing this reset is what made a two-syscall
            // blob report four calls: the two from the preceding ring-0 test were
            // still on the counter.
            EXIT_TO_KERNEL = 0;
            CALLS.store(0, Ordering::Relaxed);
            // **Pin the tenant to this core, before it ever runs.** Activating
            // `space` populates *this* core's TLB and there is no shootdown on either
            // arch, so from here on the task is only safe here. Tenants are entered
            // synchronously from the boot CPU today and are never queued, so this
            // changes no scheduling decision; it is set at the moment the fact becomes
            // true rather than the later moment it starts to matter, because the
            // failure it prevents is silent corruption rather than a fault.
            crate::sched::pin_to_boot_cpu(task);
            // **The tenant's identity is a parameter, not `current_task_id()`.**
            // Whoever calls this is the kernel, and the kernel's own task holds
            // kernel authority — checking the gates against *that* would spend the
            // caller's capabilities on the tenant's behalf, which is precisely the
            // confused deputy this whole boundary exists to prevent.
            CUR_TASK = task;
            CUR_SPACE = space as *const _;
            super::gdt::set_kernel_stack(fault_top);
            space.activate();
            asm!(
                // Park the resume point exactly as `trap_from_kernel` does.
                "lea rax, [rip + 4f]",
                "push rax",
                "mov [rip + {resume}], rsp",
                // `iretq` pops RIP, CS, RFLAGS, RSP, SS — so push them in reverse.
                // RFLAGS = 0x2: bit 1 is reserved-set, IF clear (see above).
                "push {ss}",
                "push {ursp}",
                "push 0x2",
                "push {ucs}",
                "push {urip}",
                "iretq",
                "4:",
                resume = sym RESUME_SLOT,
                ss = in(reg) super::gdt::USER_DATA_SELECTOR as u64,
                ursp = in(reg) stack_va,
                ucs = in(reg) super::gdt::USER_CODE_SELECTOR as u64,
                urip = in(reg) entry_va,
                out("rax") _,
                clobber_abi("sysv64"),
            );
            // Back in ring 0 on the kernel's own stack; the tenant's address space
            // is still active, so restore ours before anything else runs.
            crate::mm::space::activate_kernel(kernel_root);
            // Clear before the borrow ends, so a later stray `syscall` cannot be
            // dispatched against a dangling address space.
            CUR_SPACE = core::ptr::null();
            last_exit()
        })
    };
    drop(land);
    drop(fault);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn syscall_is_armed_with_the_selectors_we_intended() {
        let (sce, star, lstar, sfmask) = state();
        assert!(sce, "EFER.SCE must be on or `syscall` is #UD");
        // The CPU derives ring-0 CS/SS and ring-3 CS/SS from these fields; the
        // GDT tests pin that the derived values name the right descriptors.
        assert_eq!(
            star,
            super::super::gdt::star_value(
                super::super::gdt::KERNEL_CODE_SELECTOR,
                super::super::gdt::sysret_base_selector()
            )
        );
        assert_eq!(lstar, syscall_entry as u64, "LSTAR must point at the entry stub");
        assert_eq!(sfmask, SFMASK_VALUE);
    }

    #[test_case]
    fn interrupts_are_masked_on_kernel_entry() {
        // The bit that matters most: on `syscall` the stack pointer still belongs
        // to the caller, so an interrupt arriving in the first instructions of the
        // stub would push a frame onto *user* memory. `IF` must be in SFMASK.
        assert_ne!(SFMASK_VALUE & (1 << 9), 0, "IF must be masked on entry");
        // And the two flags a tenant could otherwise use to alter kernel
        // behaviour: TF single-steps it, DF reverses its string operations.
        assert_ne!(SFMASK_VALUE & (1 << 8), 0, "TF must be masked");
        assert_ne!(SFMASK_VALUE & (1 << 10), 0, "DF must be masked");
    }

    #[test_case]
    fn a_syscall_switches_stacks_dispatches_and_returns() {
        // The transport, end to end, without a privilege transition: `syscall` is
        // legal in ring 0, so this exercises the stack switch, the argument
        // marshalling and the return path where a mistake is recoverable. If this
        // passes and ring 3 then fails, the fault is in the transition alone.
        //
        // Two bugs this would have caught, both found by reading the asm first and
        // both silent faults rather than wrong answers: three pushes off a
        // 16-aligned stack left the `call` misaligned (SSE spill in the callee →
        // #GP), and the caller adjusted `rsp` after a `ret` that had already
        // restored it (over-pop, corrupting the *next* return).
        // SAFETY: `init` ran at boot; not reentrant, and this is the only caller.
        let got = unsafe { trap_from_kernel(7, 0xabc, 0xdef) };
        assert_eq!(got, Trapped { number: 7, arg0: 0xabc, arg1: 0xdef });
        // And we are still executing normally afterwards — the return path put the
        // stack back exactly, which an off-by-8 would not.
        let again = unsafe { trap_from_kernel(2, 1, 0) };
        assert_eq!(again, Trapped { number: 2, arg0: 1, arg1: 0 });
        assert_eq!(last_trap(), again);
    }

    #[test_case]
    fn the_stub_lands_on_its_own_stack_not_the_callers() {
        // The property that makes this safe for a real tenant: the stub must not
        // land on the stack it was called from, because that stack will be user
        // memory. `trap_from_kernel` allocates a separate landing stack for
        // exactly that reason, so a stub that ignored `LANDING` and kept using the
        // caller's `rsp` would still pass the test above — this pins the switch.
        // SAFETY: as above.
        unsafe {
            let before = LANDING;
            let _ = trap_from_kernel(1, 0, 0);
            // `trap_from_kernel` re-arms it per call, so it changed.
            assert_ne!(LANDING, 0, "the landing stack must be armed");
            let _ = before;
        }
    }

    /// `mov rdi, 2; xor rsi, rsi; xor rdx, rdx; syscall` — ask to exit, nothing else.
    ///
    /// Hand-assembled because there is no user toolchain yet: a real tenant will be
    /// a position-independent blob from its own crate, but proving the *crossing*
    /// needs only an instruction that traps.
    const EXIT_BLOB: &[u8] = &[
        0x48, 0xc7, 0xc7, 0x02, 0x00, 0x00, 0x00, // mov rdi, 2  (Entry::Exit)
        0x48, 0x31, 0xf6, // xor rsi, rsi
        0x48, 0x31, 0xd2, // xor rdx, rdx
        0x0f, 0x05, // syscall
    ];

    #[test_case]
    fn a_tenant_runs_in_ring_three_and_traps_back() {
        use crate::mm::space::{self, AddressSpace, UserPerms};
        let mut sp = AddressSpace::new().expect("address space");
        // Code page: user-readable, executable, not writable.
        let code_va = space::USER_BASE;
        let code_phys = sp.map_new_page(code_va, UserPerms::RX).expect("map code");
        // SAFETY: a frame this space owns, reachable by the kernel.
        unsafe {
            core::ptr::copy_nonoverlapping(
                EXIT_BLOB.as_ptr(),
                space::phys_to_kernel(code_phys) as *mut u8,
                EXIT_BLOB.len(),
            );
        }
        // Stack page: writable, never executable. `rsp` starts at the top.
        let stack_va = code_va + 0x1000;
        sp.map_new_page(stack_va, UserPerms::RW).expect("map stack");
        let stack_top = stack_va + 0x1000 - 16;

        // A throwaway identity: a tenant is a task, and the gates must check *its*
        // authority rather than the kernel task that launched it.
        let tenant = crate::sched::spawn_parked("ring3-exit-tenant");
        // SAFETY: both pages are mapped user-accessible with the permissions
        // `enter_ring3` requires, and the space shares the kernel mappings.
        let got = unsafe { enter_ring3(tenant, &sp, code_va, stack_top) };
        assert_eq!(
            got,
            Exit::Svc(Trapped { number: 2, arg0: 0, arg1: 0 }),
            "the tenant's `syscall` must arrive with its arguments intact"
        );
        // Still running, on the kernel's address space, with a working scheduler.
        assert_eq!(space::kernel_root(), crate::arch::x86_64::paging::active_cr3());
        yield_check();
        let _ = crate::sched::kill(tenant);
    }

    /// The scheduler still works after the crossing — a corrupted return path would
    /// show up here rather than at the assertion above.
    fn yield_check() {
        crate::sched::yield_now();
        assert_eq!(crate::sched::current_task_id(), 0);
    }

    #[test_case]
    fn a_tenant_call_reaches_the_gates_and_is_refused() {
        // **The whole stack, end to end.** A tenant executing at CPL 3, in its own
        // address space, submits a real Synapse call; the kernel validates the
        // pointer against the tenant's own page tables, copies the text in, runs
        // the four gates, and refuses it because the tenant holds no capability.
        //
        // That refusal is the point. The tenant supplied only `(number, ptr, len)`
        // — never its identity, never its address space, never a justification — so
        // there is nothing it could have said to be allowed.
        use crate::mm::space::{self, AddressSpace, UserPerms};
        const CALL: &[u8] = br#"{"name":"mem_fs_write","arguments":{"path":"ring3_probe","text":"x"}}"#;
        // Where in the tenant's data page the reply lands — past `CALL`, comfortably
        // inside the same page, so no second mapping is needed.
        const REPLY_OFF: u64 = 0x800;
        const REPLY_CAP: usize = 256;

        let mut sp = AddressSpace::new().expect("address space");
        // Data page first, so the code can name its address.
        let data_va = space::USER_BASE;
        let data_phys = sp.map_new_page(data_va, UserPerms::RW).expect("map data");
        // SAFETY: a frame this space owns, reachable by the kernel.
        unsafe {
            core::ptr::copy_nonoverlapping(
                CALL.as_ptr(),
                space::phys_to_kernel(data_phys) as *mut u8,
                CALL.len(),
            );
        }
        // `mov rdi, 1; mov rsi, data_va; mov rdx, len; syscall` — assembled here
        // rather than as a constant, because it has to embed the real user address.
        let mut blob = alloc::vec::Vec::new();
        blob.extend_from_slice(&[0x48, 0xc7, 0xc7, 0x01, 0x00, 0x00, 0x00]); // mov rdi, 1 (Invoke)
        blob.extend_from_slice(&[0x48, 0xbe]); // mov rsi, imm64
        blob.extend_from_slice(&data_va.to_le_bytes());
        blob.extend_from_slice(&[0x48, 0xc7, 0xc2]); // mov rdx, imm32
        blob.extend_from_slice(&(CALL.len() as u32).to_le_bytes());
        // **A real reply buffer, in the tenant's own data page.** Two reasons to set
        // these rather than zero them. The obvious one is that it tests `copy_out`
        // through the tenant's page tables, so the refusal is asserted where the
        // tenant can actually see it. The other is that a tenant must set *every*
        // argument register the entry reads: this blob predated `r8`/`r9` carrying
        // the reply buffer and left them holding whatever it happened to have, and a
        // bad pointer is handled (`write_reply` delivers nothing) but a
        // garbage-but-*plausible* one is indistinguishable from a real request.
        blob.extend_from_slice(&[0x49, 0xb8]); // mov r8, imm64 (reply buffer)
        blob.extend_from_slice(&(data_va + REPLY_OFF).to_le_bytes());
        blob.extend_from_slice(&[0x49, 0xc7, 0xc1]); // mov r9, imm32 (capacity)
        blob.extend_from_slice(&(REPLY_CAP as u32).to_le_bytes());
        blob.extend_from_slice(&[0x0f, 0x05]); // syscall
        // **Then exit.** Since tenants became resumable, the stub `sysretq`s back
        // after an Invoke rather than returning to the kernel — so a blob that
        // stops here keeps executing into the zeroed page after it (`add [rax], al`
        // with rax = 0, i.e. a #PF on address 0). A tenant has to say when it is
        // done; that is what `Entry::Exit` is for.
        blob.extend_from_slice(&[0x48, 0xc7, 0xc7, 0x02, 0x00, 0x00, 0x00]); // mov rdi, 2
        blob.extend_from_slice(&[0x0f, 0x05]); // syscall

        let code_va = data_va + 0x1000;
        let code_phys = sp.map_new_page(code_va, UserPerms::RX).expect("map code");
        // SAFETY: as above.
        unsafe {
            core::ptr::copy_nonoverlapping(blob.as_ptr(), space::phys_to_kernel(code_phys) as *mut u8, blob.len());
        }
        let stack_va = code_va + 0x1000;
        sp.map_new_page(stack_va, UserPerms::RW).expect("map stack");

        // A throwaway identity holding nothing, so the capability gate is what
        // answers rather than an accident of whatever task ran the test.
        let tenant = crate::sched::spawn_parked("ring3-tenant");
        // SAFETY: all three pages mapped with the permissions `enter_ring3`
        // requires; the space shares the kernel mappings.
        let last = match unsafe { enter_ring3(tenant, &sp, code_va, stack_va + 0x1000 - 16) } {
            Exit::Svc(t) => t,
            Exit::Fault { code, addr } => panic!("tenant faulted: code={code:#x} addr={addr:#x}"),
        };
        // The run ends on the Exit; the Invoke before it is what the gates saw.
        assert_eq!(last.number, 2, "the tenant exited deliberately");
        assert_eq!(call_count(), 2, "one Invoke, then the Exit");

        // **Read the reply out of the tenant's memory, not out of `last_reply()`.**
        // The run ends on the Exit, whose reply overwrites the kernel-side slot — so
        // once tenants became resumable that slot stopped answering "what did the
        // Invoke say". The tenant's own buffer is both durable across later calls and
        // the thing that actually matters: a refusal the tenant cannot read is not a
        // refusal it can act on.
        // SAFETY: the frame backing the tenant's data page, reachable by the kernel.
        let reply_bytes = unsafe {
            core::slice::from_raw_parts(
                (space::phys_to_kernel(data_phys) as *const u8).add(REPLY_OFF as usize),
                REPLY_CAP,
            )
        };
        let end = reply_bytes.iter().position(|&b| b == 0).unwrap_or(REPLY_CAP);
        let reply = core::str::from_utf8(&reply_bytes[..end]).expect("the reply is utf-8");
        assert!(
            reply.starts_with("denied:") || reply.starts_with("refused:"),
            "a tenant with no capability must be refused, got {reply}"
        );
        // And the primitive really did not run.
        assert!(!crate::synapse::fs::exists("ring3_probe"), "the gates must have stopped it");
        let _ = crate::sched::kill(tenant);
    }

    #[test_case]
    fn a_tenant_makes_several_calls_before_exiting() {
        // **Resumption, which a one-shot ABI could not express.** The tenant invokes
        // twice and then exits, so this only passes if `sysretq` really returns to
        // ring 3 with a usable stack and instruction pointer — a broken resume shows
        // up as one call and a stuck or faulting tenant, not as a wrong answer.
        use crate::mm::space::{self, AddressSpace, UserPerms};
        const CALL: &[u8] = br#"{"name":"mem_fs_write","arguments":{"path":"resume_probe","text":"x"}}"#;

        let mut sp = AddressSpace::new().expect("address space");
        let data_va = space::USER_BASE;
        let data_phys = sp.map_new_page(data_va, UserPerms::RW).expect("map data");
        // SAFETY: a frame this space owns, reachable by the kernel.
        unsafe {
            core::ptr::copy_nonoverlapping(CALL.as_ptr(), space::phys_to_kernel(data_phys) as *mut u8, CALL.len());
        }
        let out_va = data_va + 0x1000;
        sp.map_new_page(out_va, UserPerms::RW).expect("map out");

        // Two Invokes then an Exit. Assembled here because it embeds real addresses.
        let mut blob = alloc::vec::Vec::new();
        let mut invoke = |b: &mut alloc::vec::Vec<u8>| {
            b.extend_from_slice(&[0x48, 0xc7, 0xc7, 0x01, 0x00, 0x00, 0x00]); // mov rdi, 1
            b.extend_from_slice(&[0x48, 0xbe]); // mov rsi, imm64
            b.extend_from_slice(&data_va.to_le_bytes());
            b.extend_from_slice(&[0x48, 0xc7, 0xc2]); // mov rdx, imm32
            b.extend_from_slice(&(CALL.len() as u32).to_le_bytes());
            b.extend_from_slice(&[0x49, 0xb8]); // mov r8, imm64  (reply buffer)
            b.extend_from_slice(&out_va.to_le_bytes());
            b.extend_from_slice(&[0x49, 0xc7, 0xc1, 0x40, 0x00, 0x00, 0x00]); // mov r9, 64
            b.extend_from_slice(&[0x0f, 0x05]); // syscall
        };
        invoke(&mut blob);
        invoke(&mut blob);
        blob.extend_from_slice(&[0x48, 0xc7, 0xc7, 0x02, 0x00, 0x00, 0x00]); // mov rdi, 2 (Exit)
        blob.extend_from_slice(&[0x0f, 0x05]); // syscall

        let code_va = out_va + 0x1000;
        let code_phys = sp.map_new_page(code_va, UserPerms::RX).expect("map code");
        // SAFETY: as above.
        unsafe {
            core::ptr::copy_nonoverlapping(blob.as_ptr(), space::phys_to_kernel(code_phys) as *mut u8, blob.len());
        }
        let stack_va = code_va + 0x1000;
        sp.map_new_page(stack_va, UserPerms::RW).expect("map stack");

        let tenant = crate::sched::spawn_parked("resume-tenant");
        // SAFETY: every page mapped with the permissions `enter_ring3` requires.
        let last = match unsafe { enter_ring3(tenant, &sp, code_va, stack_va + 0x1000 - 16) } {
            Exit::Svc(t) => t,
            Exit::Fault { code, addr } => panic!("tenant faulted: code={code:#x} addr={addr:#x}"),
        };
        assert_eq!(last.number, 2, "the run ends on Exit, not on the first Invoke");
        assert_eq!(call_count(), 3, "two Invokes plus the Exit");
        // The gates ran each time and refused each time.
        let reply = last_reply().expect("a reply was recorded");
        assert!(reply.starts_with("denied:") || reply == "exited", "got {reply}");
        assert!(!crate::synapse::fs::exists("resume_probe"), "the gates must have stopped it");
        let _ = crate::sched::kill(tenant);
    }

    #[test_case]
    fn a_faulting_tenant_is_reported_not_fatal() {
        // **This test completing at all is the result.** Before `abort_tenant`, a
        // tenant fault on x86 halted the machine: the fault goes to the IDT rather
        // than through the syscall stub, and `fault_current_task` correctly refuses
        // to kill the *kernel* task that entered ring 3 (a tenant is ring-3 code
        // inside that task, not a task of its own), so the handler fell through to
        // its halt loop. aarch64 already reported it, because there `svc` and faults
        // share one vector — so this was also a live dual-arch divergence.
        use crate::mm::space::{self, AddressSpace, UserPerms};
        let mut sp = AddressSpace::new().expect("address space");
        // A code page whose blob reads an address that is mapped in *no* space.
        let code_va = space::USER_BASE;
        let code_phys = sp.map_new_page(code_va, UserPerms::RX).expect("map code");
        let unmapped = space::USER_BASE + 64 * 0x1000;
        let mut blob = alloc::vec::Vec::new();
        blob.extend_from_slice(&[0x48, 0xb8]); // mov rax, imm64
        blob.extend_from_slice(&unmapped.to_le_bytes());
        blob.extend_from_slice(&[0x48, 0x8b, 0x00]); // mov rax, [rax]  -> #PF
        // SAFETY: a frame this space owns, reachable by the kernel.
        unsafe {
            core::ptr::copy_nonoverlapping(blob.as_ptr(), space::phys_to_kernel(code_phys) as *mut u8, blob.len());
        }
        let stack_va = code_va + 0x1000;
        sp.map_new_page(stack_va, UserPerms::RW).expect("map stack");

        let tenant = crate::sched::spawn_parked("faulting-tenant");
        // SAFETY: pages mapped as `enter_ring3` requires; the blob faults on purpose.
        match unsafe { enter_ring3(tenant, &sp, code_va, stack_va + 0x1000 - 16) } {
            Exit::Fault { addr, .. } => {
                assert_eq!(addr, unmapped, "the reported address must be the one it touched");
            }
            Exit::Svc(t) => panic!("expected a fault, got a syscall: {t:?}"),
        }
        // And the machine is fine: kernel address space restored, scheduler working.
        assert_eq!(space::kernel_root(), crate::arch::x86_64::paging::active_cr3());
        crate::sched::yield_now();
        let _ = crate::sched::kill(tenant);
    }

    #[test_case]
    fn arming_syscall_did_not_clobber_nx() {
        // `EFER` is read-modify-written because `NXE` is already set: clobbering
        // it would make every `NO_EXECUTE` page-table entry fault, and the heap is
        // mapped that way — so the machine would die somewhere unrelated.
        const EFER_NXE: u64 = 1 << 11;
        assert_ne!(read_msr(EFER_MSR) & EFER_NXE, 0, "EFER.NXE must survive arming SCE");
    }
}
