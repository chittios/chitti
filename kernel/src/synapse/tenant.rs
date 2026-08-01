//! Loading and running a userspace tenant.
//!
//! The third piece of the ring-3 story, after the transport ([`crate::arch`]'s
//! `fastcall`/`el0`) and the authority boundary ([`crate::synapse::abi`]): given a
//! blob of position-independent user code, build it an address space, map it, and
//! run it. It lives under `synapse` because what a tenant *is*, from the OS's point
//! of view, is a principal that can only reach the world through the ABI.
//!
//! # Why the code is assembled rather than hand-encoded
//!
//! Every tenant so far was a `&[u8]` of machine code written out by hand, and that
//! cost two real bugs: a register shuffle whose two `mov`s destroyed their own source,
//! and a blob that left the reply-buffer registers holding garbage after the ABI grew
//! them (a *plausible* stray pointer, so the reply landed on the tenant's own code
//! page and it re-executed). Neither was catchable by the type system or by review of
//! the surrounding Rust, because to the compiler a byte array is just data.
//!
//! `global_asm!` hands that work to the assembler: mnemonics instead of opcodes,
//! label arithmetic instead of counted offsets, and a build error instead of a
//! runtime fault when something does not encode. This is the shape the user-visible
//! blobs should take from here on.
//!
//! # Position independence, without a relocation pass
//!
//! The kernel decides where a tenant lands, so the blob must not contain absolute
//! addresses — and rather than fix them up, these blobs simply never form one. Both
//! arches address their own payload PC-relatively (`lea rsi, [rip + …]`, `adr x1, …`)
//! and take their length from assembler label arithmetic, which is an immediate
//! constant rather than an address. So the same bytes work at any base, there is no
//! relocation table to process, and the loader needs to know nothing about the blob's
//! internal layout. That is what makes "embedded PIC blob" a loader you can trust
//! without an ELF parser.
//!
//! # What the tenant deliberately does *not* do
//!
//! It passes a null reply buffer. A tenant reading its reply back is already covered
//! (`fastcall`'s gate test asserts the refusal text lands in the tenant's own page),
//! and adding it here would need a writable page addressed from read-only code —
//! which needs either a relocation or a startup register contract. Both are real
//! design decisions, and neither belongs in the change that proves the loader.

use crate::mm::space::{self, AddressSpace, UserPerms};

/// One gated Synapse call, then exit.
///
/// The call is `mem_fs_read` — a *non-destructive* primitive on purpose. The kernel
/// stamps a tenant's justification [`Provenance::UntrustedIngested`], so a
/// destructive primitive would be stopped by the taint gate and could never
/// demonstrate a completed effect. The refusal paths are tested elsewhere; what has
/// never been shown is a tenant in ring 3 successfully *doing* something.
///
/// Kept in `.rodata` (not a function) because it is data as far as this kernel is
/// concerned: it is never executed here, only copied into a user page.
#[cfg(target_arch = "x86_64")]
core::arch::global_asm!(
    r#"
// **AT&T syntax deliberately.** The length is assembler label arithmetic, and in
// Intel syntax `.Lmsg_end - .Lmsg` parses as a *memory operand* ("cannot use more
// than one symbol in memory operand"); AT&T's `$` states unambiguously that this is
// an immediate. Restored to Intel at the end, since that is what the rest of the
// kernel's x86 asm is written in.
.att_syntax prefix
.section .rodata
.balign 16
.globl chitti_tenant_hello_start
.globl chitti_tenant_hello_end
chitti_tenant_hello_start:
    movq $1, %rdi                       // Entry::Invoke
    leaq .Lmsg(%rip), %rsi              // the call text, addressed PC-relatively
    movq $(.Lmsg_end - .Lmsg), %rdx     // its length: an immediate, not an address
    xorq %r8, %r8                       // no reply buffer ...
    xorq %r9, %r9                       // ... and so a capacity of zero
    syscall
    movq $2, %rdi                       // Entry::Exit
    syscall
    // Unreachable: `Exit` does not return. `ud2` rather than a halt or a jump back,
    // so if the exit path ever *does* return the tenant faults immediately and the
    // kernel reports it, instead of running off into whatever follows.
    ud2
.Lmsg:
    .ascii "{{\"name\":\"mem_fs_read\",\"arguments\":{{\"path\":\"tenant_probe\"}}}}"
.Lmsg_end:
chitti_tenant_hello_end:
.section .text
.intel_syntax noprefix
"#
);

/// The aarch64 twin. Same structure, same guarantees; `adr` is PC-relative within
/// ±1 MiB, which a blob this size cannot exceed.
#[cfg(target_arch = "aarch64")]
core::arch::global_asm!(
    r#"
.section .rodata
.balign 16
.globl chitti_tenant_hello_start
.globl chitti_tenant_hello_end
chitti_tenant_hello_start:
    mov x0, #1                          // Entry::Invoke
    adr x1, .Lmsg64                     // the call text, addressed PC-relatively
    mov x2, #(.Lmsg64_end - .Lmsg64)    // its length: an immediate, not an address
    mov x3, #0                          // no reply buffer ...
    mov x4, #0                          // ... and so a capacity of zero
    svc #0
    mov x0, #2                          // Entry::Exit
    svc #0
    brk #0                              // unreachable; see the x86 note on `ud2`
.Lmsg64:
    .ascii "{{\"name\":\"mem_fs_read\",\"arguments\":{{\"path\":\"tenant_probe\"}}}}"
.Lmsg64_end:
chitti_tenant_hello_end:
.section .text
"#
);

unsafe extern "C" {
    static chitti_tenant_hello_start: u8;
    static chitti_tenant_hello_end: u8;
}

/// The assembled bytes of the one-call tenant.
pub fn hello_blob() -> &'static [u8] {
    // SAFETY: both symbols are defined by the `global_asm!` above, in the same
    // section and in this order, so the difference is the blob's length.
    unsafe {
        let start = &chitti_tenant_hello_start as *const u8;
        let end = &chitti_tenant_hello_end as *const u8;
        core::slice::from_raw_parts(start, end.offset_from(start) as usize)
    }
}

/// Why a tenant could not be loaded. Distinct from how it *exited*: this is the
/// kernel failing to set it up, which is the kernel's bug, not the tenant's.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LoadError {
    /// No address space could be created (out of frames).
    NoAddressSpace,
    /// A page could not be mapped.
    Map,
    /// The blob does not fit the single code page this loader provides.
    TooBig { len: usize },
}

/// A loaded tenant: its address space and the addresses it was laid out at.
///
/// Deliberately a plain struct with public fields rather than a handle: everything
/// about the layout is a decision this loader made, and a caller that wants to
/// inspect or extend it should not have to guess.
pub struct Loaded {
    pub space: AddressSpace,
    pub entry: u64,
    /// Initial stack pointer — the *top* of the stack page, less a little, since both
    /// architectures grow down and neither wants to start exactly at a page boundary.
    pub stack: u64,
}

/// Lay `blob` out as a tenant: one RX code page and one RW stack page.
///
/// The smallest layout that can run anything, and enough for a first-party agent
/// whose data lives behind the ABI rather than in its own memory. A tenant needing
/// more (a heap, a reply buffer, its own read-only data) is a later extension of this
/// function, not of its callers.
pub fn load(blob: &[u8]) -> Result<Loaded, LoadError> {
    const PAGE: u64 = 0x1000;
    if blob.len() as u64 > PAGE {
        return Err(LoadError::TooBig { len: blob.len() });
    }
    let mut space = AddressSpace::new().ok_or(LoadError::NoAddressSpace)?;
    let entry = space::USER_BASE;
    let code_phys = space.map_new_page(entry, UserPerms::RX).map_err(|_| LoadError::Map)?;
    // SAFETY: a freshly mapped frame this space owns, reachable by the kernel, and
    // `blob` is bounded above by the page size.
    unsafe {
        core::ptr::copy_nonoverlapping(blob.as_ptr(), space::phys_to_kernel(code_phys) as *mut u8, blob.len());
    }
    let stack_page = entry + PAGE;
    space.map_new_page(stack_page, UserPerms::RW).map_err(|_| LoadError::Map)?;
    Ok(Loaded { space, entry, stack: stack_page + PAGE - 16 })
}

/// Load and run the one-call tenant, once, at boot.
///
/// Exists for the same reason [`crate::arch::aarch64::el0::self_test`] does: the unit
/// suite is x86 only, so the aarch64 blob and the aarch64 half of the loader would
/// otherwise be *compiled but never executed* — which is precisely the divergence the
/// standing dual-architecture rule forbids, and the kind that stays invisible because
/// both arches build. On x86 the same path is covered by
/// `a_loaded_tenant_runs_in_ring3_and_its_call_reaches_the_gates`.
///
/// Returns `Err` with a short reason rather than panicking; the caller ktraces it.
pub fn self_test() -> Result<(), &'static str> {
    let loaded = load(hello_blob()).map_err(|_| "could not load the tenant")?;
    let tenant = crate::sched::spawn_parked("tenant-selftest");
    let before = crate::synapse::audit::len();
    // SAFETY: `load` mapped the entry RX and the stack RW in a space that shares the
    // kernel mappings.
    let exit = unsafe { crate::arch::enter_tenant(tenant, &loaded.space, loaded.entry, loaded.stack) };
    let saw_call = crate::synapse::audit::snapshot()[before..].iter().any(|e| e.caller == tenant);
    let _ = crate::sched::kill(tenant);
    if !exit.is_deliberate_exit() {
        // A fault here is a loader bug — the mapping, the permissions, or the blob's
        // position independence — so it is worth telling apart from a call that simply
        // never arrived.
        return Err("the tenant faulted or stopped without exiting");
    }
    if !saw_call {
        return Err("the tenant exited but its call never reached the gates");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn the_blob_is_assembled_and_self_contained() {
        let blob = hello_blob();
        // Non-empty and inside one page: `load` refuses anything larger, so a blob
        // that grew past this would fail at run time rather than here.
        assert!(!blob.is_empty(), "the tenant blob must not be empty");
        assert!(blob.len() <= 0x1000, "the tenant blob must fit its code page, is {}", blob.len());
        // The call text travels *with* the code, which is what makes the blob
        // position-independent: nothing outside it has to be mapped or fixed up.
        let text = b"mem_fs_read";
        assert!(
            blob.windows(text.len()).any(|w| w == text),
            "the call text must be embedded in the blob, not referenced from outside it"
        );
    }

    #[test_case]
    fn a_loaded_tenant_runs_in_ring3_and_its_call_reaches_the_gates() {
        // The end-to-end claim of P7/P9: assembled PIC code, mapped by the loader,
        // executed at CPL 3 / EL0, reaching the four gates under its *own* identity.
        use crate::synapse::audit;

        let loaded = load(hello_blob()).expect("load the tenant");
        // A throwaway identity holding nothing: the capability gate must be what
        // answers, not whatever authority the task running this test happens to have.
        let tenant = crate::sched::spawn_parked("hello-tenant");
        let before = audit::len();

        // SAFETY: `load` mapped the entry RX and the stack RW in this space, which
        // shares the kernel mappings.
        let exit = unsafe { crate::arch::enter_tenant(tenant, &loaded.space, loaded.entry, loaded.stack) };

        // It must have left deliberately. A fault here means the *loader* is wrong —
        // the mapping, the permissions, or the blob's position independence — so it is
        // worth distinguishing from a call that merely got refused.
        assert!(exit.is_deliberate_exit(), "the tenant did not exit cleanly: {exit:?}");

        // And its call reached the audit log under its own task id. The outcome is not
        // asserted: without a capability it is refused, and this test's claim is that
        // an untrusted tenant's call *arrives at the boundary* attributed to it. What
        // the gates then decide is the gates' own tests' business.
        let entries = audit::snapshot();
        assert!(
            entries[before..].iter().any(|e| e.caller == tenant),
            "the tenant's call must appear in the audit log under its own identity"
        );
        // Pinned by construction, since it activated a user address space on this core.
        assert!(crate::sched::is_pinned_to_boot_cpu(tenant), "a tenant must be pinned to the boot CPU");
        let _ = crate::sched::kill(tenant);
    }
}
