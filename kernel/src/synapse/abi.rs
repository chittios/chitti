//! Synapse's **trap entry**: how a userspace tenant submits a call across the
//! privilege boundary.
//!
//! # Why this lives inside `synapse` and not in a module of its own
//!
//! Because the invariant is *all effects route through Synapse*, and a separate
//! module is where that stops being structural and becomes a rule someone has to
//! remember. The first version of this was `kernel/src/syscall.rs` with three
//! entries, and the middle one — a `Write` that put bytes on the console — was
//! exactly the mistake the arrangement invited: `console_write` is already a
//! gated primitive, so that was a second path to an effect Synapse owns, with no
//! capability check, no audit entry and no taint gate. A file named for a POSIX
//! concept, sitting outside the gate chain, made it feel like plumbing rather
//! than a hole.
//!
//! Here, adding an entry means editing the module tree whose entire purpose is
//! gating, next to the four gates it would have to justify itself against. The
//! test for anything new is not "is it useful" but **"does it convey authority"**.
//! It is also why this is not called a syscall: CLAUDE.md's invariant is that
//! there is no `exec(binary) → trap into syscalls`, and the unit of execution
//! stays an agent planning over capabilities. A trap that carries one Synapse
//! call is transport. A syscall *table* is the thing that invariant forbids.
//!
//! # There is exactly one call that does anything, and that is the design
//!
//! [`executor::execute`] already had the shape of a trap entry before there was
//! one — it takes a task id and an owned string, returns an
//! owned result, and reaches nothing of the caller's state that the caller did
//! not name. So this module wraps it rather than inventing a parallel ABI: the
//! entire kernel surface a tenant can reach is "submit one grammar-checked tool
//! call, get its result". Every effect still routes through the four gates, and
//! there is no second door for a tenant to look for.
//!
//! That is a deliberate contrast with a POSIX-shaped ABI, where the interesting
//! question is which of ~300 syscalls a sandbox forgot to filter. Here the
//! authority question is entirely inside `synapse`, where it was already
//! answered.
//!
//! # The rule: this ABI may add transport, never authority
//!
//! CLAUDE.md's invariant is that **all effects route through Synapse**, and the
//! paper's central claim is that every effect clears four gates. A syscall that
//! reached an effect directly would falsify both — not by argument, but by
//! existing. It would also be a ninth entry in the `security::redteam` census of
//! enforcement sites, which exists precisely because "a new binding that forgets
//! its check is a hole by omission".
//!
//! The first version of this file had three calls, and the middle one was exactly
//! that mistake: a `Write` that put bytes on the console. `console_write` is
//! already a Synapse primitive (registry id 1), so `Write` was a second path to
//! an effect Synapse owns — reachable without holding
//! `InvokePrimitive(console_write)`, with no audit entry, no taint gate and no
//! scope check. It is gone. A tenant that wants the console invokes
//! `console_write` like anything else, and is gated and audited for it.
//!
//! So the test for adding anything here is not "is it useful" but "does it convey
//! authority". A trap that carries a Synapse call across a privilege boundary is
//! transport. A trap that performs an effect is a hole. Note that this is also
//! why a *single* `Invoke` is compatible with "there is no `exec(binary) → trap
//! into syscalls`": the unit of execution is still an agent planning over
//! capabilities, and the trap is only how the plan crosses the ring boundary. A
//! general syscall table would be the thing that invariant forbids.
//!
//! # What the kernel must not trust
//!
//! Everything the caller supplies is an offset and a length in *its own* address
//! space. Two rules, both of which are the whole reason this file exists rather
//! than the executor being called directly:
//!
//! 1. **A user pointer is validated against the caller's own address space**, not
//!    merely range-checked against a constant. A tenant that could name a kernel
//!    address would read or write kernel memory through a copy routine that
//!    believed it was doing the caller a favour — the classic confused-deputy
//!    read, and the reason [`copy_in`] resolves every page through the caller's
//!    page tables before touching it.
//! 2. **The copy is bounded before it is attempted.** A length is a claim; a
//!    caller that claims 4 GiB gets refused, not honoured until the heap runs out.
//!
//! # Not yet reachable from ring 3
//!
//! The machine-level trap (`syscall`/`sysret` MSRs on x86, `svc` on aarch64) and
//! the ring-3/EL0 transition are not here. This is the *authority* half: the
//! validation, the copy, the dispatch and the result marshalling, all callable and
//! testable from kernel context. Wiring a trap to [`dispatch`] adds an entry stub
//! and changes nothing in this file — which is the point of splitting them there,
//! since this half is where a mistake is a security bug and that half is where a
//! mistake is a hang.

use crate::mm::space::{self, AddressSpace};
use crate::sched::TaskId;
use alloc::string::String;
use alloc::vec::Vec;

/// The entry numbers a tenant may pass. Deliberately tiny; see the module doc.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u64)]
pub enum Entry {
    /// Submit one Synapse tool call. Argument: `(ptr, len)` of UTF-8 in the
    /// caller's address space. Result: the executor's response text.
    Invoke = 1,
    /// Relinquish the CPU permanently: the calling task is done.
    ///
    /// The only call besides [`Entry::Invoke`], and the only one that is not a
    /// Synapse primitive. It is not an *effect* on the world — it grants no
    /// authority over anything but the caller itself, changes nothing another
    /// principal can observe except that a task stopped, and writes no state. A
    /// task must also always be able to stop: gating it on a capability would
    /// mean a tenant lacking that capability could never terminate, and the
    /// kernel would have to kill it anyway. It is closer to a `return` than to a
    /// syscall.
    Exit = 2,
}

impl Entry {
    /// Decode a raw entry number. Unknown numbers are refused rather than
    /// treated as anything — an ABI that silently accepted an out-of-range number
    /// as its nearest neighbour would be a way to reach a primitive by guessing.
    pub fn from_raw(n: u64) -> Option<Self> {
        match n {
            1 => Some(Entry::Invoke),
            2 => Some(Entry::Exit),
            _ => None,
        }
    }
}

/// Why an entry was refused before it did anything.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AbiError {
    /// The entry number is not one this kernel implements.
    UnknownCall,
    /// The `(ptr, len)` pair does not lie entirely inside the caller's user
    /// range, or `ptr + len` overflowed.
    BadPointer,
    /// `len` exceeds [`MAX_ARG_LEN`].
    TooLong,
    /// A page in `(ptr, len)` is not mapped in the caller's address space.
    Unmapped,
    /// The bytes were not valid UTF-8. Refused rather than replaced with
    /// replacement characters: a tenant should not be able to make the grammar
    /// see something different from what it sent.
    NotUtf8,
}

/// Longest argument a tenant may submit, in bytes.
///
/// A tool call is JSON naming a primitive and its arguments; 64 KiB is far more
/// than any real call and small enough that a hostile length cannot be used to
/// exhaust the heap. The check happens *before* any allocation, which is the
/// point — a caller's claimed length must never size a buffer.
pub const MAX_ARG_LEN: usize = 64 * 1024;

/// What an entry produced.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Reply {
    /// The call ran. `text` is the response; `written` is how many of its bytes
    /// reached the caller's output buffer.
    ///
    /// The two can differ: a caller that offered a small buffer gets a **truncated**
    /// reply and a `written` smaller than `text.len()`, rather than an error. That is
    /// deliberate — the call itself succeeded and its effect happened, so failing the
    /// whole thing over the reply buffer would make the caller believe nothing ran.
    /// A tenant that cares compares `written` against what it asked for.
    Ok { text: String, written: usize },
    /// The ABI refused it. No primitive ran, and nothing was copied.
    Err(AbiError),
    /// The caller asked to exit.
    Exited,
}

/// Copy `len` bytes from user address `ptr` in `space` into kernel memory.
///
/// This is the confused-deputy boundary. Three checks in this order, and the
/// order matters:
///
/// 1. **length** — before any allocation, so a claimed size cannot be used to
///    exhaust the heap;
/// 2. **range** — the whole span must lie inside the user range, computed with
///    `checked_add` so `ptr + len` cannot wrap into it;
/// 3. **mapping** — every page resolved through *the caller's own tables*, so a
///    tenant naming an address it does not own is refused rather than served.
///
/// Doing (3) via the page tables rather than trusting (2) is what makes the
/// difference: the user *range* is a constant, and a tenant can name an
/// unmapped address inside it as easily as a mapped one.
pub fn copy_in(space: &AddressSpace, ptr: u64, len: usize) -> Result<Vec<u8>, AbiError> {
    if len > MAX_ARG_LEN {
        return Err(AbiError::TooLong);
    }
    let end = ptr.checked_add(len as u64).ok_or(AbiError::BadPointer)?;
    if !space::is_user_range(ptr, end) {
        return Err(AbiError::BadPointer);
    }
    let mut out = Vec::new();
    out.try_reserve_exact(len).map_err(|_| AbiError::TooLong)?;
    let mut va = ptr;
    while va < end {
        let page = va & !(crate::mm::frame::FRAME_SIZE - 1);
        let phys = space.translate(page).ok_or(AbiError::Unmapped)?;
        let off = (va - page) as usize;
        let take = (crate::mm::frame::FRAME_SIZE as usize - off).min((end - va) as usize);
        // SAFETY: `phys + off` is inside a frame the caller's space maps at `va`,
        // reachable by the kernel at `phys_to_kernel`, and `take` stays within
        // that frame by construction.
        let src = unsafe {
            core::slice::from_raw_parts((space::phys_to_kernel(phys) + off as u64) as *const u8, take)
        };
        out.extend_from_slice(src);
        va += take as u64;
    }
    Ok(out)
}

/// Copy `bytes` out to user address `ptr` in `space`, returning how many landed.
///
/// Same validation as [`copy_in`], and the same reason: a tenant naming a kernel
/// address for its *output* buffer would have the kernel scribble on itself.
pub fn copy_out(space: &AddressSpace, ptr: u64, bytes: &[u8]) -> Result<usize, AbiError> {
    let end = ptr.checked_add(bytes.len() as u64).ok_or(AbiError::BadPointer)?;
    if !space::is_user_range(ptr, end) {
        return Err(AbiError::BadPointer);
    }
    let mut va = ptr;
    let mut done = 0usize;
    while va < end {
        let page = va & !(crate::mm::frame::FRAME_SIZE - 1);
        let phys = space.translate(page).ok_or(AbiError::Unmapped)?;
        let off = (va - page) as usize;
        let take = (crate::mm::frame::FRAME_SIZE as usize - off).min(bytes.len() - done);
        // SAFETY: as `copy_in`, and the frame is writable because a user mapping
        // this space created is either RW or RO — a RO target simply gets the
        // bytes, since the kernel is not subject to the user permission bits and
        // the caller asked for this write.
        let dst = unsafe {
            core::slice::from_raw_parts_mut((space::phys_to_kernel(phys) + off as u64) as *mut u8, take)
        };
        dst.copy_from_slice(&bytes[done..done + take]);
        done += take;
        va += take as u64;
    }
    Ok(done)
}

/// Handle one trap entry from `caller`, whose memory is `space`.
///
/// `arg0`/`arg1` are the raw register values a trap would have delivered. No
/// machine-level trap is wired yet (see the module doc); this is callable from
/// kernel context, which is how it is tested.
pub fn dispatch(
    caller: TaskId,
    space: &AddressSpace,
    number: u64,
    arg0: u64,
    arg1: u64,
    out_ptr: u64,
    out_cap: u64,
) -> Reply {
    let Some(call) = Entry::from_raw(number) else {
        return Reply::Err(AbiError::UnknownCall);
    };
    match call {
        Entry::Invoke => {
            let bytes = match copy_in(space, arg0, arg1 as usize) {
                Ok(b) => b,
                Err(e) => return Reply::Err(e),
            };
            let raw = match String::from_utf8(bytes) {
                Ok(s) => s,
                Err(_) => return Reply::Err(AbiError::NotUtf8),
            };
            // **The kernel builds the justification, not the caller.** A tenant
            // submits only the call text; provenance and human-confirmation come
            // from the kernel's own record of this task. That is what stops a
            // tenant asserting `human_confirmed` or `SystemTrusted` and walking
            // through the taint gate and the approval check.
            //
            // The value is whatever the kernel *set before entering userspace*
            // ([`set_run_justification`]), defaulting to fully-tainted when nothing
            // did. That default is the safe one, and it is also what made the value
            // worth parameterising: a caller migrated from ring 0 to ring 3 kept the
            // same identity and the same capabilities but silently became maximally
            // tainted, so every destructive call it used to make legitimately started
            // being refused. Moving code across the ring boundary must not change what
            // it is allowed to do — otherwise "migrate to userspace" is a behaviour
            // change wearing a refactor's clothes. The tenant still cannot influence
            // this: it is set outside ring 3, by the code that chose to run it.
            let justification = run_justification();
            let inv = super::executor::execute_with_justification(caller, &raw, justification);
            // **Record the structured outcome for the kernel, alongside the text for the
            // tenant.** The tenant needs prose; the kernel needs to classify. Letting the
            // kernel re-derive the classification from the prose is what broke the
            // red-team census: a migrated caller's refusals came back through `render`,
            // whose wording differs from the tool router's, so refusals were read as
            // successes and five injected attacks were counted as *permitted*. The
            // refusal text is load-bearing — it is how both the model and the harness
            // tell success from failure — so there must be exactly one authority for
            // what an outcome was, and it is this value, not a string.
            set_last_invocation(inv.clone());
            let text = render(&inv);
            // Hand the answer back through the caller's own page tables. `copy_out`
            // validates the destination exactly as `copy_in` validated the source —
            // a tenant naming a kernel address for its *output* buffer would have
            // the kernel scribble on itself, which is the same confused-deputy bug
            // read in the other direction.
            let written = write_reply(space, out_ptr, out_cap, &text);
            Reply::Ok { text, written }
        }
        Entry::Exit => Reply::Exited,
    }
}

/// The justification the kernel has set for the tenant run now in progress.
///
/// A plain static rather than a parameter because it has to survive the trip through
/// userspace and back: the value is chosen by whoever entered ring 3, and read again
/// when that tenant traps. Single-slot is sufficient — a tenant runs on the boot CPU,
/// pinned, and `enter_tenant` is not reentrant, so exactly one run is ever live.
static RUN_JUSTIFICATION: crate::mm::Locked<Option<crate::security::taint::Justification>> =
    crate::mm::Locked::new(None);

/// Set the justification for the tenant run about to start. Call **before** entering
/// userspace; pair with [`clear_run_justification`] after it returns.
///
/// # Why this is not a hole
/// The tenant never touches it. It is written by kernel code that already knows the
/// provenance of the content motivating the call — for an agent, the `Router`'s
/// computation over its session's resident taint — and a tenant has no way to reach
/// this slot. The unforgeable part of the design is that *the caller cannot choose its
/// own provenance*, and that still holds: the choice is made by the kernel, above the
/// boundary, exactly as it is for an in-kernel caller.
pub fn set_run_justification(j: crate::security::taint::Justification) {
    RUN_JUSTIFICATION.with(|slot| *slot = Some(j));
}

/// Forget any justification set for a finished run, so a later tenant cannot inherit
/// a trust decision made about somebody else's content.
pub fn clear_run_justification() {
    RUN_JUSTIFICATION.with(|slot| *slot = None);
}

/// What the current run's calls are justified by, defaulting to fully tainted.
fn run_justification() -> crate::security::taint::Justification {
    RUN_JUSTIFICATION.with(|slot| *slot).unwrap_or_else(|| {
        crate::security::taint::Justification::from_context(
            crate::security::taint::Provenance::UntrustedIngested,
        )
    })
}

/// The structured outcome of the most recent [`Entry::Invoke`], for the kernel code
/// that asked a tenant to make the call.
///
/// Deliberately *not* something the tenant can read: it is the kernel's own record, so
/// that a caller migrated into userspace is classified from the same value an in-kernel
/// caller would have been, rather than from a re-parse of the reply text.
static LAST_INVOCATION: crate::mm::Locked<Option<super::executor::Invocation>> = crate::mm::Locked::new(None);

fn set_last_invocation(inv: super::executor::Invocation) {
    LAST_INVOCATION.with(|slot| *slot = Some(inv));
}

/// Take the structured outcome of the tenant's last `Invoke`, clearing it.
///
/// Taken rather than peeked so a later caller cannot read someone else's result if the
/// run it expected never happened — an absent value is then a visible `None` rather
/// than a stale success.
pub fn take_last_invocation() -> Option<super::executor::Invocation> {
    LAST_INVOCATION.with(|slot| slot.take())
}

/// Copy as much of `text` as fits into the caller's output buffer, returning how
/// many bytes landed.
///
/// A zero `cap` means the caller wants no reply, which is legal — `Entry::Exit`
/// and fire-and-forget calls have nothing to read. A *bad* pointer is not an error
/// either, for the same reason truncation is not: the primitive has already run, so
/// the honest report is "nothing was delivered", not "the call failed".
fn write_reply(space: &AddressSpace, out_ptr: u64, out_cap: u64, text: &str) -> usize {
    if out_cap == 0 {
        return 0;
    }
    let n = (text.len() as u64).min(out_cap) as usize;
    // Truncate on a char boundary so a tenant never receives half a UTF-8
    // sequence — it would be reading bytes that are not the text we sent.
    let head = crate::tools::pathutil::truncate_on_char_boundary(text, n);
    copy_out(space, out_ptr, head.as_bytes()).unwrap_or(0)
}

/// Render an [`super::executor::Invocation`] as the text a tenant sees.
///
/// Deliberately says *which* gate refused. A tenant learning "denied: no
/// capability" versus "needs human approval" can react sensibly — ask, or stop —
/// and neither tells it anything it could not already infer by trying. Silence
/// would only make a compromised tenant probe blindly, which is noisier.
fn render(inv: &super::executor::Invocation) -> String {
    use super::executor::Invocation;
    match inv {
        Invocation::Executed { result, .. } => alloc::format!("ok:{result}"),
        Invocation::Denied { primitive } => alloc::format!("denied:capability:{primitive}"),
        Invocation::Rejected(err) => alloc::format!("rejected:{err:?}"),
        Invocation::RefusedTainted { primitive } => alloc::format!("refused:tainted:{primitive}"),
        Invocation::DeniedScope { primitive } => alloc::format!("denied:scope:{primitive}"),
        Invocation::NeedsApproval { primitive } => alloc::format!("denied:needs-approval:{primitive}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mm::space::UserPerms;

    const PAGE: u64 = crate::mm::frame::FRAME_SIZE;

    /// A space with one RW user page holding `text`, and its address.
    fn space_with(text: &[u8]) -> (AddressSpace, u64) {
        let mut s = AddressSpace::new().expect("address space");
        let va = space::USER_BASE;
        let phys = s.map_new_page(va, UserPerms::RW).expect("map");
        // SAFETY: a frame this space owns, reachable by the kernel.
        unsafe {
            let dst = core::slice::from_raw_parts_mut(space::phys_to_kernel(phys) as *mut u8, text.len());
            dst.copy_from_slice(text);
        }
        (s, va)
    }

    #[test_case]
    fn an_unknown_syscall_number_is_refused_not_rounded() {
        let (s, _) = space_with(b"");
        // Nothing adjacent, nothing large, nothing zero should reach a primitive.
        for n in [0u64, 4, 99, u64::MAX] {
            assert_eq!(dispatch(0, &s, n, 0, 0, 0, 0), Reply::Err(AbiError::UnknownCall), "number {n}");
        }
    }

    #[test_case]
    fn a_kernel_address_is_refused() {
        // The confused-deputy case: a tenant naming kernel memory must not have
        // the kernel read it on the tenant's behalf.
        let (s, _) = space_with(b"");
        for ptr in [0u64, 0x1000, space::USER_BASE - PAGE, u64::MAX & !0xfff] {
            let r = copy_in(&s, ptr, 8);
            assert!(
                matches!(r, Err(AbiError::BadPointer) | Err(AbiError::Unmapped)),
                "kernel/OOB address {ptr:#x} must be refused, got {r:?}"
            );
        }
    }

    #[test_case]
    fn a_span_that_overflows_is_refused_rather_than_wrapping_into_range() {
        // `ptr + len` wrapping is how a range check gets talked out of its job.
        // Note *which* check fires in each case — the order is part of the
        // contract, and getting the expectation wrong here is how you end up
        // "fixing" a check that was already right.
        let (s, va) = space_with(b"");
        // Absurd length: the length check fires first, before any arithmetic.
        assert_eq!(copy_in(&s, va, usize::MAX), Err(AbiError::TooLong));
        assert_eq!(copy_in(&s, va, MAX_ARG_LEN + 1), Err(AbiError::TooLong));
        // Plausible length near the top of the address space: the length check
        // passes, so it is `checked_add` that catches the wrap.
        assert_eq!(copy_in(&s, u64::MAX - 8, 64), Err(AbiError::BadPointer));
        // And a wrap that would land back inside the user range if it were
        // allowed to happen.
        assert_eq!(copy_in(&s, u64::MAX, 1), Err(AbiError::BadPointer));
    }

    #[test_case]
    fn an_unmapped_user_address_is_refused() {
        // Inside the user *range* but not mapped in this space. The range is a
        // constant, so only consulting the caller's page tables catches this.
        let (s, va) = space_with(b"hi");
        let far = va + 64 * PAGE;
        assert!(space::is_user_addr(far), "test premise: still in the user range");
        assert_eq!(copy_in(&s, far, 8), Err(AbiError::Unmapped));
    }

    #[test_case]
    fn a_valid_copy_in_round_trips() {
        let (s, va) = space_with(b"hello syscall");
        assert_eq!(copy_in(&s, va, 13).unwrap(), b"hello syscall");
        // A zero-length read is legal and yields nothing.
        assert_eq!(copy_in(&s, va, 0).unwrap(), b"");
    }

    #[test_case]
    fn a_copy_spanning_two_pages_needs_both_mapped() {
        // The per-page resolve is not decoration: a span crossing a page boundary
        // where only the first page is mapped must be refused, not truncated.
        let mut s = AddressSpace::new().unwrap();
        let va = space::USER_BASE;
        s.map_new_page(va, UserPerms::RW).unwrap();
        let straddle = va + PAGE - 4;
        assert_eq!(copy_in(&s, straddle, 8), Err(AbiError::Unmapped));
        // With the second page mapped it succeeds.
        s.map_new_page(va + PAGE, UserPerms::RW).unwrap();
        assert_eq!(copy_in(&s, straddle, 8).unwrap().len(), 8);
    }

    #[test_case]
    fn non_utf8_is_refused_not_replaced() {
        // Replacing invalid bytes would let a tenant make the grammar see
        // something other than what it sent.
        let (s, va) = space_with(&[0xff, 0xfe, 0x00, 0x00]);
        assert_eq!(dispatch(0, &s, Entry::Invoke as u64, va, 2, 0, 0), Reply::Err(AbiError::NotUtf8));
    }

    #[test_case]
    fn copy_out_validates_its_destination_too() {
        // A tenant naming a kernel address as its *output* buffer would have the
        // kernel scribble on itself.
        let (s, va) = space_with(b"");
        assert_eq!(copy_out(&s, 0x1000, b"xx"), Err(AbiError::BadPointer));
        assert_eq!(copy_out(&s, va + 64 * PAGE, b"xx"), Err(AbiError::Unmapped));
        assert_eq!(copy_out(&s, va, b"out").unwrap(), 3);
        assert_eq!(copy_in(&s, va, 3).unwrap(), b"out");
    }

    #[test_case]
    fn invoke_reaches_the_gates_and_a_tenant_cannot_claim_trust() {
        // A tenant with no capabilities must be refused by the capability gate —
        // and the justification the kernel builds is UntrustedIngested, so it
        // cannot walk past the taint gate either. The tenant supplies only text.
        let call = br#"{"name":"mem_fs_write","arguments":{"path":"abi_probe","text":"x"}}"#;
        let (s, va) = space_with(call);
        let victim = crate::sched::spawn_parked("abi-tenant");
        let out = dispatch(victim, &s, Entry::Invoke as u64, va, call.len() as u64, 0, 0);
        match out {
            Reply::Ok { text, .. } => {
                assert!(
                    text.starts_with("denied:") || text.starts_with("refused:"),
                    "an uncapable tenant must be refused, got {text}"
                );
            }
            other => panic!("expected a rendered refusal, got {other:?}"),
        }
        assert!(!crate::synapse::fs::exists("abi_probe"), "the primitive must not have run");
        let _ = crate::sched::kill(victim);
    }

    #[test_case]
    fn the_reply_is_delivered_through_the_callers_own_page_tables() {
        // The other half of the round trip, validated the same way as the input: a
        // tenant naming a kernel address for its *output* buffer would have the
        // kernel scribble on itself — the confused deputy read in reverse.
        let call = br#"{"name":"mem_fs_write","arguments":{"path":"abi_reply","text":"x"}}"#;
        let (sp, va) = space_with_two_pages(call);
        let out_va = va + PAGE;
        let victim = crate::sched::spawn_parked("abi-reply-tenant");
        let written = match dispatch(victim, &sp, Entry::Invoke as u64, va, call.len() as u64, out_va, 256) {
            Reply::Ok { text, written } => {
                assert!(text.starts_with("denied:"), "no capability, so a refusal: {text}");
                assert_eq!(written, text.len(), "the whole refusal fitted in 256 bytes");
                written
            }
            other => panic!("expected Ok, got {other:?}"),
        };
        // Read it back *out of tenant memory* — the only thing that proves delivery
        // rather than a plausible return value.
        let got = copy_in(&sp, out_va, written).expect("read the reply back");
        assert!(core::str::from_utf8(&got).unwrap().starts_with("denied:"));
        let _ = crate::sched::kill(victim);
    }

    #[test_case]
    fn a_small_reply_buffer_truncates_rather_than_failing() {
        // By the time the reply is written the primitive has already run, so failing
        // the call over the buffer size would tell the caller nothing happened when
        // something did. Truncate, and report how much landed.
        let call = br#"{"name":"mem_fs_write","arguments":{"path":"abi_trunc","text":"x"}}"#;
        let (sp, va) = space_with_two_pages(call);
        let out_va = va + PAGE;
        let victim = crate::sched::spawn_parked("abi-trunc-tenant");
        match dispatch(victim, &sp, Entry::Invoke as u64, va, call.len() as u64, out_va, 4) {
            Reply::Ok { text, written } => {
                assert_eq!(written, 4, "only what fits");
                assert!(text.len() > 4, "and the kernel still sees the whole text");
            }
            other => panic!("expected Ok, got {other:?}"),
        }
        // Zero capacity means "no reply wanted", which is legal.
        match dispatch(victim, &sp, Entry::Invoke as u64, va, call.len() as u64, out_va, 0) {
            Reply::Ok { written, .. } => assert_eq!(written, 0),
            other => panic!("expected Ok, got {other:?}"),
        }
        // A kernel address as the destination delivers nothing, and is still not an
        // error: the call ran.
        match dispatch(victim, &sp, Entry::Invoke as u64, va, call.len() as u64, 0x1000, 64) {
            Reply::Ok { written, .. } => assert_eq!(written, 0, "kernel address: nothing delivered"),
            other => panic!("expected Ok, got {other:?}"),
        }
        let _ = crate::sched::kill(victim);
    }

    /// A space with `text` in one RW page and a second RW page right after it.
    fn space_with_two_pages(text: &[u8]) -> (AddressSpace, u64) {
        let mut s = AddressSpace::new().expect("address space");
        let va = space::USER_BASE;
        let phys = s.map_new_page(va, UserPerms::RW).expect("map");
        // SAFETY: a frame this space owns, reachable by the kernel.
        unsafe {
            core::ptr::copy_nonoverlapping(text.as_ptr(), space::phys_to_kernel(phys) as *mut u8, text.len());
        }
        s.map_new_page(va + PAGE, UserPerms::RW).expect("map out");
        (s, va)
    }

    #[test_case]
    fn exit_reports_itself_and_console_output_is_not_a_syscall() {
        let (s, _va) = space_with(b"abi\n");
        assert_eq!(dispatch(0, &s, Entry::Exit as u64, 0, 0, 0, 0), Reply::Exited);
        // The ABI must expose exactly two calls. A third would need to justify
        // itself against "does it convey authority" — the question a `Write`
        // syscall failed, since `console_write` is already a gated primitive.
        assert_eq!(Entry::from_raw(3), None, "no third call: effects go through the gates");
        assert!(matches!(Entry::from_raw(1), Some(Entry::Invoke)));
        assert!(matches!(Entry::from_raw(2), Some(Entry::Exit)));
    }
}
