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
    /// The blob does not fit the single code page this loader provides, or a call's
    /// text does not fit the startup block.
    TooBig { len: usize },
    /// The tenant faulted, or stopped without exiting deliberately. Distinct from a
    /// refusal: this says userspace misbehaved, not that the gates said no.
    Faulted,
    /// The tenant ran and reported its own failure — a malformed input, or a panic it
    /// caught. The interesting case for a decoder, and deliberately not `Faulted`: a
    /// corrupt file is an ordinary answer, not a broken boundary.
    Declined { status: u64 },
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
    /// Base of the stack page, so a caller can map further pages after it without
    /// re-deriving the layout this function chose.
    pub stack_page: u64,
}

impl Loaded {
    /// Map kernel-owned `frames` consecutively at `va` in the tenant.
    ///
    /// The frames stay the **kernel's** — this shares them rather than handing them over,
    /// which is what lets megabytes of input and output cross the boundary with no copy.
    /// The kernel keeps its own alias (`space::phys_to_kernel`) and can read the result
    /// straight out after the tenant exits.
    ///
    /// Read-only for input is not a formality: a decoder must not be able to rewrite the
    /// bitstream it was handed, or a second pass over the "same" data is not the same data.
    pub fn map_shared(&mut self, va: u64, frames: &[u64], perms: UserPerms) -> Result<(), LoadError> {
        for (i, &phys) in frames.iter().enumerate() {
            self.space
                .map_frame(va + (i as u64) * 0x1000, phys, perms)
                .map_err(|_| LoadError::Map)?;
        }
        Ok(())
    }
}

/// A tenant image: its bytes plus where things are inside them.
///
/// The layout comes from `cargo xtask imgdec` (via `llvm-nm` and the linker script), because
/// requiring the linker to put `_start` first and to keep everything read-only did not survive
/// contact with either arch: x86 put the entry 12 bytes in, and a single mutable static landed
/// in the code page and faulted the tenant on its own first instruction.
#[derive(Clone, Copy)]
pub struct Blob {
    pub bytes: &'static [u8],
    /// Entry offset within `bytes`.
    pub entry: u64,
    /// Bytes to map **read-execute** (text + rodata), page-aligned.
    pub rx: u64,
    /// Bytes to map **read-write** (data + bss), page-aligned. Only the `.data` prefix is
    /// present in `bytes`; the rest is `.bss` and is zeroed by the loader.
    pub rw: u64,
}

impl Blob {
    /// A hand-assembled blob: entry first, one code page, no writable data.
    pub const fn flat(bytes: &'static [u8]) -> Self {
        Self { bytes, entry: 0, rx: 0x1000, rw: 0 }
    }
}

/// Lay a [`Blob`] out: its RX pages, then its RW pages, then a stack page.
///
/// The RW pages are the point. A decoder needs statics — inflate's window, Huffman tables — and
/// mapping them into the code page is what made the tenant fault writing its own memory
/// (`error=0x7`: a write to a present read-only page, at the entry instruction).
pub fn load_image(b: Blob) -> Result<Loaded, LoadError> {
    const PAGE: u64 = 0x1000;
    if (b.bytes.len() as u64) > b.rx + b.rw {
        return Err(LoadError::TooBig { len: b.bytes.len() });
    }
    let mut space = AddressSpace::new().ok_or(LoadError::NoAddressSpace)?;
    let base = space::USER_BASE;
    let mut copied = 0usize;
    for (off, perms) in [(0u64, UserPerms::RX), (b.rx, UserPerms::RW)] {
        let len = if off == 0 { b.rx } else { b.rw };
        for p in 0..len / PAGE {
            let va = base + off + p * PAGE;
            let phys = space.map_new_page(va, perms).map_err(|_| LoadError::Map)?;
            // SAFETY: a freshly mapped, zeroed frame this space owns, reachable by the kernel.
            // `take` is bounded by what remains of the image, so `.bss` — which is inside the
            // RW range but absent from the file — simply stays zero.
            let take = b.bytes.len().saturating_sub(copied).min(PAGE as usize);
            if take > 0 {
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        b.bytes.as_ptr().add(copied),
                        space::phys_to_kernel(phys) as *mut u8,
                        take,
                    );
                }
                copied += take;
            }
        }
    }
    let stack_page = base + b.rx + b.rw;
    space.map_new_page(stack_page, UserPerms::RW).map_err(|_| LoadError::Map)?;
    Ok(Loaded { space, entry: base + b.entry, stack: stack_page + PAGE - 16, stack_page })
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
    Ok(Loaded { space, entry, stack: stack_page + PAGE - 16, stack_page })
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
    let exit = unsafe { crate::arch::enter_tenant(tenant, &loaded.space, loaded.entry, loaded.stack, 0) };
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

    // And the *parameterised* tenant, for the same reason this function exists at all:
    // the aarch64 `call_blob` and the startup block would otherwise be compiled and
    // never executed, which both arches building does nothing to catch. Covered on x86
    // by `the_kernel_chooses_a_tenants_work_and_the_gates_still_answer`.
    let caller = crate::sched::spawn_parked("tenant-selftest-call");
    let got = run_call(caller, r#"{"name":"mem_fs_write","arguments":{"path":"selftest_probe","text":"x"}}"#);
    let _ = crate::sched::kill(caller);
    match got {
        // Refused is the expected answer: the throwaway identity holds nothing. What
        // is being checked here is that the reply travelled back through the block.
        Ok(reply) if reply.starts_with("denied:") || reply.starts_with("refused:") => Ok(()),
        Ok(reply) if reply.is_empty() => Err("the parameterised tenant returned no reply"),
        Ok(_) => Err("an unauthorised parameterised call was not refused"),
        Err(_) => Err("the parameterised tenant could not be run"),
    }
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
    fn the_kernel_chooses_a_tenants_work_and_the_gates_still_answer() {
        // P9's migration primitive. Two claims, and the second is the one that makes
        // moving code to userspace safe rather than merely possible:
        //   1. One blob serves different calls — the work arrives through the startup
        //      block, so nothing is assembled in and nothing is relocated.
        //   2. Authority does not travel with it. The tenant holds no capability, so
        //      the call is refused, and the refusal comes back *to userspace*.
        let tenant = crate::sched::spawn_parked("call-tenant");
        let reply = run_call(tenant, r#"{"name":"mem_fs_write","arguments":{"path":"p9_probe","text":"x"}}"#)
            .expect("the tenant must run and exit cleanly");
        assert!(
            reply.starts_with("denied:") || reply.starts_with("refused:"),
            "an unauthorised userspace call must be refused, got {reply}"
        );
        assert!(!crate::synapse::fs::exists("p9_probe"), "the gates must have stopped the write");

        // A *different* call through the same blob, proving the parameterisation is
        // real and not an accident of one payload.
        let reply2 = run_call(tenant, r#"{"name":"mem_fs_read","arguments":{"path":"p9_probe"}}"#)
            .expect("the tenant must run and exit cleanly");
        assert!(!reply2.is_empty(), "the second call must also produce a reply");
        let _ = crate::sched::kill(tenant);
    }

    #[test_case]
    fn the_shared_tenant_really_performs_the_call_in_userspace() {
        // The migration's witness. Both paths yield the same reply and the same audit
        // entry — that equivalence is the goal — so the only way to show the effect
        // actually crossed into ring 3 is to count crossings.
        let tenant = crate::sched::spawn_parked("shared-tenant-probe");
        let before = userspace_calls();
        let j = crate::security::taint::Justification::from_context(
            crate::security::taint::Provenance::UserTyped,
        );
        let reply = call_in_userspace(tenant, r#"{"name":"mem_fs_read","arguments":{"path":"nope"}}"#, j)
            .expect("the shared tenant must run");
        assert!(!reply.is_empty(), "the call must produce a reply");
        assert_eq!(userspace_calls(), before + 1, "the call must have crossed into userspace");

        // Reused, not rebuilt: a second call goes through the same address space.
        let reply2 = call_in_userspace(tenant, r#"{"name":"list","arguments":{}}"#, j)
            .expect("the shared tenant must be reusable");
        assert!(!reply2.is_empty());
        assert_eq!(userspace_calls(), before + 2);
        let _ = crate::sched::kill(tenant);
    }

    #[test_case]
    fn the_kernel_supplied_justification_survives_the_crossing() {
        // **Moving a caller to ring 3 must not change what it may do.** A destructive
        // primitive justified by user-typed content passes the taint gate in the kernel,
        // so it must pass it from userspace too — and it only does because the kernel
        // hands the tenant runner the real justification instead of letting the ABI's
        // fully-tainted default stand. Without this, migrating an agent would quietly
        // refuse every destructive call it legitimately makes.
        //
        // Asserted by contrast so it cannot pass for the wrong reason: the same call, as
        // the same task, differing *only* in the justification the kernel set. Note the
        // primitive has to be a genuinely destructive one — `mem_fs_write` is
        // `Effect::INERT` and the taint gate does not touch it, which is what made a
        // first version of this test pass a tainted write and prove nothing.
        let tenant = crate::sched::spawn_parked("justification-probe");
        let spec = crate::synapse::registry::by_name("mem_fs_delete").expect("mem_fs_delete is registered");
        crate::cap::grant(tenant, crate::cap::Right::InvokePrimitive(spec.id));
        crate::synapse::fs::write("just_probe", b"x");
        let call = r#"{"name":"mem_fs_delete","arguments":{"path":"just_probe"}}"#;

        let tainted = call_in_userspace(
            tenant,
            call,
            crate::security::taint::Justification::from_context(
                crate::security::taint::Provenance::UntrustedIngested,
            ),
        )
        .expect("the tenant must run");
        assert!(tainted.starts_with("refused:"), "untrusted-ingested must be refused, got {tainted}");
        assert!(crate::synapse::fs::exists("just_probe"), "the refused delete must not have happened");

        let typed = call_in_userspace(
            tenant,
            call,
            crate::security::taint::Justification::from_context(
                crate::security::taint::Provenance::UserTyped,
            ),
        )
        .expect("the tenant must run");
        assert!(
            !typed.starts_with("refused:") && !typed.starts_with("denied:"),
            "user-typed must clear the taint gate, got {typed}"
        );
        assert!(!crate::synapse::fs::exists("just_probe"), "the allowed delete must have happened");
        let _ = crate::sched::kill(tenant);
    }

    #[test_case]
    fn a_tenant_computes_over_bulk_shared_buffers_with_no_authority() {
        // **The decoder shape, proven end to end.** Megabytes in and out through shared
        // frames, computed in ring 3, with the tenant holding *no capability at all* —
        // which is why moving PNG/JPEG/H.264 here needs no new Synapse primitive. The
        // payload is a checksum so the assertion can be differential against a trivially
        // correct kernel-side computation; porting a real decoder changes what runs
        // inside, not whether the boundary works.
        let tenant = crate::sched::spawn_parked("bulk-tenant");

        // Spans several pages on purpose: a single-page buffer would pass even if
        // `map_shared` mapped only the first frame, which is the mistake worth catching.
        let input: alloc::vec::Vec<u8> = (0..10_000u32).map(|i| (i % 251) as u8).collect();
        let expected: u64 = input.iter().map(|&b| b as u64).sum();

        let out = run_bulk(tenant, &input).expect("the bulk tenant must run");
        assert_eq!(out.len(), 8, "expected an 8-byte sum, got {}", out.len());
        let got = u64::from_le_bytes(out[..8].try_into().unwrap());
        assert_eq!(got, expected, "ring-3 result must match the kernel's own computation");

        // It really held nothing: the whole point is that a decoder needs no authority, so
        // a capability appearing here would mean the loader was granting one silently.
        assert!(
            !crate::cap::holds(tenant, crate::cap::Right::InvokePrimitive(1)),
            "a compute tenant must hold no capabilities"
        );
        let _ = crate::sched::kill(tenant);
    }

    #[test_case]
    fn the_compiled_userspace_crate_runs_as_a_tenant() {
        // **The crate half of the boundary, proven.** `userspace/imgdec/` is ordinary safe
        // Rust compiled for a freestanding target, linked at `USER_BASE`, objcopy'd to a
        // flat binary and checked in — and here it executes in ring 3 and produces the same
        // answer as the kernel and as the hand-assembled tenant.
        //
        // A *triple* differential, which is stronger than it looks: kernel arithmetic vs an
        // asm tenant vs a compiled-Rust tenant, all over the same bytes through the same
        // loader. Agreement means the crate's startup-block offsets, its entry convention,
        // its linker script's load address and its exit path are all right — any one of
        // which being wrong yields a plausible number rather than an obvious failure.
        let tenant = crate::sched::spawn_parked("imgdec-tenant");
        let input: alloc::vec::Vec<u8> = (0..9_000u32).map(|i| (i % 253) as u8).collect();
        let expected: u64 = input.iter().map(|&b| b as u64).sum();

        let out = run_imgdec(tenant, &input).expect("the compiled tenant must run");
        assert_eq!(out.len(), 8, "expected an 8-byte sum, got {}", out.len());
        assert_eq!(
            u64::from_le_bytes(out[..8].try_into().unwrap()),
            expected,
            "the compiled userspace tenant must agree with the kernel"
        );

        let asm_out = run_bulk(tenant, &input).expect("the asm tenant must run");
        assert_eq!(asm_out, out, "the compiled and assembled tenants must agree");
        let _ = crate::sched::kill(tenant);
    }

    #[test_case]
    fn the_compiled_tenant_is_a_plausible_flat_binary() {
        // Cheap guards on the checked-in artefact, because the failure mode of a stale or
        // wrongly-objcopy'd blob is a tenant that faults at its first instruction — which
        // reads as "ring 3 is broken" rather than "rebuild the blob".
        let b = imgdec_blob();
        assert!(!b.is_empty(), "the compiled tenant blob is empty -- run `cargo xtask imgdec`");
        // Checked against the image's *own* layout, not a one-page assumption: the loader maps
        // `rx` bytes read-execute and `rw` read-write, and the whole point of that split was to
        // let the blob outgrow a page — it is ~11 KiB now that the real decoder is in it.
        let img = imgdec_image();
        assert!(img.rx >= b.len() as u64 || b.len() as u64 <= img.rx + img.rw,
            "blob is {} bytes but the layout only covers rx {} + rw {}", b.len(), img.rx, img.rw);
        assert_eq!(img.rx % 0x1000, 0, "rx must be page-aligned, is {}", img.rx);
        assert!(img.rx > 0, "an image with no executable pages cannot run");
        // A flat binary, not an ELF: objcopy was skipped if this still has a magic number.
        assert_ne!(&b[..4.min(b.len())], b"\x7fELF", "the blob is an ELF, not a flat image");
    }

    #[test_case]
    fn what_ring_three_actually_costs() {
        // The counterpart to `tools/pngbench`'s 38x for wasm: what does the *other* sandbox
        // cost? Measured as two numbers, because they scale differently and quoting one alone
        // is how a per-run setup cost gets mistaken for a per-byte one.
        //
        //   fixed  — build an address space, map the pages, cross, tear down, free the frames.
        //            Paid once per decode.
        //   marginal — the compute itself, which is the *same machine code* either side of the
        //            boundary, so the expectation is ~1x and anything else wants explaining.
        //
        // Coarse on purpose: `now_ms` has millisecond resolution, so each figure is a batch of
        // `N` runs and the per-run cost is a division. Enough to separate "a trap" from "38x".
        const N: usize = 20;
        const SMALL: usize = 0x1000; // 1 page
        const LARGE: usize = 64 * 0x1000; // 64 pages, 256 KiB

        let tenant = crate::sched::spawn_parked("cost-tenant");
        let small = alloc::vec![0xa5u8; SMALL];
        let large = alloc::vec![0x5au8; LARGE];

        // Warm up: the first run grows the heap and faults in the loader's own pages, and
        // charging that to the measurement is how a curve comes out decreasing.
        let _ = run_bulk(tenant, &small).expect("warm-up");

        let t0 = crate::arch::now_ms();
        for _ in 0..N {
            let _ = core::hint::black_box(run_bulk(tenant, &small).expect("small"));
        }
        let t_small = crate::arch::now_ms().saturating_sub(t0);

        let t0 = crate::arch::now_ms();
        for _ in 0..N {
            let _ = core::hint::black_box(run_bulk(tenant, &large).expect("large"));
        }
        let t_large = crate::arch::now_ms().saturating_sub(t0);

        crate::ktrace::log_fmt(format_args!(
            "tenant.cost: {N} runs -- 1 page {t_small}ms, 256KiB {t_large}ms  (per run: fixed ~{}us, +256KiB ~{}us)",
            t_small * 1000 / N as u64,
            t_large.saturating_sub(t_small) * 1000 / N as u64
        ));

        // **No ring-0 comparison here, deliberately.** The obvious thing is to sum the same
        // bytes in the kernel and divide -- and it produces a *flattering lie*: the tenant runs
        // a hand-written asm loop while the kernel side is debug-build Rust with bounds checks,
        // which measured ~9x slower and made ring 3 look faster than ring 0. The same mistake as
        // timing `opt-level="s"` wasm against `opt-level=3` native, in the other direction.
        //
        // The honest position is that the ratio does not need measuring: ring 3 runs the same
        // instructions on the same CPU with the same caches, so marginal cost is native *by
        // construction* -- there is no interpreter to be slow. What is worth measuring is the
        // **fixed** cost above, against the work it protects: ~650 us against a 19 ms PNG decode
        // is ~3%, where wasm's 38x is 3800%.
        //
        // A real ratio needs the identical decoder compiled for both sides, which is what
        // `tools/pngbench` gives once the native tenant runs `image/png.rs`.
        assert!(
            t_large >= t_small,
            "more work took less time: {t_large}ms for 256KiB vs {t_small}ms for one page"
        );
        let _ = crate::sched::kill(tenant);
    }

    #[test_case]
    fn bulk_tenant_frames_are_returned_so_repeated_runs_do_not_leak() {
        // A decoder runs once per frame of video, so a per-run frame leak would exhaust
        // memory in seconds rather than eventually. Asserted across several runs, since a
        // single run could be masked by the allocator's own slack.
        let tenant = crate::sched::spawn_parked("bulk-tenant-leak");
        let input = alloc::vec![7u8; 9000];
        let before = space::free_frames();
        for _ in 0..3 {
            let out = run_bulk(tenant, &input).expect("run");
            assert_eq!(u64::from_le_bytes(out[..8].try_into().unwrap()), 7 * 9000);
        }
        let after = space::free_frames();
        assert_eq!(after, before, "frames must be returned: {before} -> {after}");
        let _ = crate::sched::kill(tenant);
    }

    #[test_case]
    fn a_call_too_large_for_the_startup_block_is_refused_before_running() {
        // Bounded up front rather than truncated: a silently shortened call is a
        // *different* call, and one that might pass gates the original would not.
        let tenant = crate::sched::spawn_parked("call-tenant-big");
        let huge = alloc::string::String::from_utf8(alloc::vec![b'x'; block::CAPACITY + 1]).unwrap();
        assert!(matches!(run_call(tenant, &huge), Err(LoadError::TooBig { .. })));
        let _ = crate::sched::kill(tenant);
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
        let exit = unsafe { crate::arch::enter_tenant(tenant, &loaded.space, loaded.entry, loaded.stack, 0) };

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

// ---------------------------------------------------------------------------
// A tenant whose work is chosen by the kernel
// ---------------------------------------------------------------------------

/// Offsets into the **startup block**: the one piece of ABI between the loader and a
/// parameterised tenant.
///
/// The `hello` blob above has its call assembled into it, which proves the loader but
/// cannot migrate anything: real work is chosen at run time. Rather than relocate a
/// blob or patch its immediates, the loader writes a small header into a page the
/// tenant owns and hands over its address in the startup register. The blob then
/// contains no addresses and no data at all — the same bytes serve every call.
pub mod block {
    /// Pointer to the call text (in the tenant's own memory).
    pub const CALL_PTR: usize = 0;
    /// Length of the call text.
    pub const CALL_LEN: usize = 8;
    /// Where the tenant asks for the reply.
    pub const REPLY_PTR: usize = 16;
    /// How much room the reply may use.
    pub const REPLY_CAP: usize = 24;
    /// Written *by the tenant*: how many bytes of reply it received.
    pub const REPLY_LEN: usize = 32;
    /// Base of the tenant's input buffer (read-only shared frames).
    pub const INPUT_PTR: usize = 40;
    /// How many bytes of input there are.
    pub const INPUT_LEN: usize = 48;
    /// Base of the tenant's output buffer (writable shared frames).
    pub const OUTPUT_PTR: usize = 56;
    /// How much room the output buffer has.
    pub const OUTPUT_CAP: usize = 64;
    /// Written *by the tenant*: how many output bytes it produced.
    pub const OUTPUT_LEN: usize = 72;
    /// Base of the tenant's private heap.
    pub const HEAP_PTR: usize = 80;
    /// How large that heap is.
    pub const HEAP_LEN: usize = 88;
    /// Written *by the tenant*: 0 on success, else its own error code. Distinct from a
    /// fault — a decoder that cleanly rejects a malformed file is not a crash.
    pub const STATUS: usize = 96;
    /// What the loader writes here **before** entry, so the field is fail-closed: a tenant
    /// that exits without claiming success (because it panicked, or never got that far)
    /// leaves this behind rather than an ambiguous zero. It is why the tenant's
    /// `panic_handler` can simply leave, and needs no writable memory to report through.
    pub const STATUS_NOT_RUN: u64 = u64::MAX;
    /// Where the loader puts the call text, clear of the header.
    ///
    /// Moved from 64 when the buffer fields were added. Safe to move because no blob
    /// hard-codes it — they read `CALL_PTR` — so this offset is known only to the loader.
    pub const TEXT: usize = 128;
    /// Where the reply lands. Half a page each way, so neither can reach the other.
    pub const REPLY: usize = 2048;
    /// Room for the reply, and so also the largest call text.
    pub const CAPACITY: usize = 2048 - TEXT;
}

/// The parameterised tenant: read the startup block, submit the call it names, record
/// the reply length, exit.
///
/// `rbx` survives the call because it is callee-saved in the SysV C ABI, and
/// `syscall_dispatch` is an ordinary `extern "C"` function — so the block pointer is
/// still there after the trap. Relying on that is exactly the ABI contract the trap
/// documents, rather than a hopeful assumption about what the handler happens to touch.
#[cfg(target_arch = "x86_64")]
core::arch::global_asm!(
    r#"
.att_syntax prefix
.section .rodata
.balign 16
.globl chitti_tenant_call_start
.globl chitti_tenant_call_end
chitti_tenant_call_start:
    movq %rdi, %rbx                     // the startup block, in a callee-saved register
    movq 0(%rbx), %rsi                  // call text
    movq 8(%rbx), %rdx                  // its length
    movq 16(%rbx), %r8                  // reply buffer
    movq 24(%rbx), %r9                  // reply capacity
    movq $1, %rdi                       // Entry::Invoke
    syscall
    movq %rax, 32(%rbx)                 // how much reply we got
    // **Re-zero the reply registers before Exit.** The trap clobbers the C ABI's
    // caller-saved set, so r8/r9 now hold whatever the handler left there — and
    // passing those as (out_ptr, out_cap) is the exact defect that made the first
    // hand-written blob overwrite its own code page. Exit wants no reply buffer.
    xorq %r8, %r8
    xorq %r9, %r9
    movq $2, %rdi                       // Entry::Exit
    syscall
    ud2                                 // unreachable; Exit does not return
chitti_tenant_call_end:
.section .text
.intel_syntax noprefix
"#
);

/// The aarch64 twin. `x19` is callee-saved in AAPCS64, for the same reason `rbx` is
/// used above.
#[cfg(target_arch = "aarch64")]
core::arch::global_asm!(
    r#"
.section .rodata
.balign 16
.globl chitti_tenant_call_start
.globl chitti_tenant_call_end
chitti_tenant_call_start:
    mov x19, x0                         // the startup block, in a callee-saved register
    ldr x1, [x19, #0]                   // call text
    ldr x2, [x19, #8]                   // its length
    ldr x3, [x19, #16]                  // reply buffer
    ldr x4, [x19, #24]                  // reply capacity
    mov x0, #1                          // Entry::Invoke
    svc #0
    str x0, [x19, #32]                  // how much reply we got
    // Re-zero the reply registers before Exit; see the x86 note. x3/x4 are
    // caller-saved, so after the first trap they hold handler leftovers, and passing
    // those as (out_ptr, out_cap) asks the kernel to write a reply somewhere nobody
    // chose.
    mov x3, #0
    mov x4, #0
    mov x0, #2                          // Entry::Exit
    svc #0
    brk #0                              // unreachable; Exit does not return
chitti_tenant_call_end:
.section .text
"#
);

unsafe extern "C" {
    static chitti_tenant_call_start: u8;
    static chitti_tenant_call_end: u8;
}

/// A tenant that consumes a **bulk input buffer** and produces a **bulk output buffer**,
/// making no Synapse call at all.
///
/// This is the shape every decoder has, and the reason moving PNG/JPEG/H.264 into ring 3
/// needs no new Synapse primitive: a decoder reads bytes, writes pixels, and exits. It
/// requires no authority, so there is nothing for a gate to decide. What it does need is
/// exactly what this proves — bulk data crossing in both directions through shared frames,
/// and the kernel reading the result back out afterwards.
///
/// The payload is a checksum rather than a decode, so the test can be *differential*
/// against a trivially correct kernel-side computation. Porting the real decoder then
/// changes what runs inside, not whether the boundary works.
#[cfg(target_arch = "x86_64")]
core::arch::global_asm!(
    r#"
.att_syntax prefix
.section .rodata
.balign 16
.globl chitti_tenant_bulk_start
.globl chitti_tenant_bulk_end
chitti_tenant_bulk_start:
    movq %rdi, %rbx                     // startup block (callee-saved: survives a trap)
    movq 40(%rbx), %rsi                 // input_ptr
    movq 48(%rbx), %rcx                 // input_len
    xorq %rax, %rax                     // running sum
1:  testq %rcx, %rcx
    jz 2f
    movzbl (%rsi), %edx                 // zero-extends, so the high bits stay clean
    addq %rdx, %rax
    incq %rsi
    decq %rcx
    jmp 1b
2:  movq 56(%rbx), %rdx                 // output_ptr
    movq %rax, (%rdx)                   // the sum, little-endian
    movq $8, 72(%rbx)                   // output_len
    movq $0, 96(%rbx)                   // status: ok
    movq $2, %rdi                       // Entry::Exit
    xorq %r8, %r8
    xorq %r9, %r9
    syscall
    ud2
chitti_tenant_bulk_end:
.section .text
.intel_syntax noprefix
"#
);

/// The aarch64 twin.
#[cfg(target_arch = "aarch64")]
core::arch::global_asm!(
    r#"
.section .rodata
.balign 16
.globl chitti_tenant_bulk_start
.globl chitti_tenant_bulk_end
chitti_tenant_bulk_start:
    mov x19, x0                         // startup block (callee-saved)
    ldr x1, [x19, #40]                  // input_ptr
    ldr x2, [x19, #48]                  // input_len
    mov x3, #0                          // running sum
1:  cbz x2, 2f
    ldrb w4, [x1], #1
    add x3, x3, x4
    sub x2, x2, #1
    b 1b
2:  ldr x5, [x19, #56]                  // output_ptr
    str x3, [x5]
    mov x6, #8
    str x6, [x19, #72]                  // output_len
    str xzr, [x19, #96]                 // status: ok
    mov x0, #2                          // Entry::Exit
    mov x3, #0
    mov x4, #0
    svc #0
    brk #0
chitti_tenant_bulk_end:
.section .text
"#
);

unsafe extern "C" {
    static chitti_tenant_bulk_start: u8;
    static chitti_tenant_bulk_end: u8;
}

/// The assembled bytes of the bulk-buffer tenant.
pub fn bulk_blob() -> &'static [u8] {
    // SAFETY: as the other blobs — two symbols bracketing one `.rodata` run.
    unsafe {
        let start = &chitti_tenant_bulk_start as *const u8;
        let end = &chitti_tenant_bulk_end as *const u8;
        core::slice::from_raw_parts(start, end.offset_from(start) as usize)
    }
}

/// The **Rust userspace tenant**, built from `userspace/imgdec/` and linked at
/// `space::USER_BASE`.
///
/// This is the seam the real decoders arrive through. Unlike the assembled blobs above it
/// is ordinary safe Rust with a `panic_handler` that reports a status word — so a bounds
/// check tripped by a malformed file becomes "this input is corrupt" instead of halting the
/// machine, which is the entire reason for moving a parser across the boundary.
///
/// Checked in and `include_bytes!`d per arch, like `tools/*-wasm`'s modules; rebuild with
/// `cargo xtask imgdec`. Linked at a fixed address rather than made position-independent,
/// which is what keeps the loader free of any relocation handling.
#[cfg(target_arch = "x86_64")]
static IMGDEC_BLOB: &[u8] = include_bytes!("../../../userspace/imgdec/imgdec-x86_64.bin");
#[cfg(target_arch = "aarch64")]
static IMGDEC_BLOB: &[u8] = include_bytes!("../../../userspace/imgdec/imgdec-aarch64.bin");

/// The compiled userspace tenant's bytes.
pub fn imgdec_blob() -> &'static [u8] {
    IMGDEC_BLOB
}

/// Byte offset of the tenant's entry point within [`imgdec_blob`], emitted by
/// `cargo xtask imgdec` from the ELF's own header.
///
/// **Read, not arranged.** The loader used to jump to offset 0 and require the linker to put
/// `_start` there; on x86 it landed at `+0xc` while aarch64 was exact, so a tenant executed into
/// the middle of an unrelated function and faulted reading a kernel address. An
/// `ASSERT(_start == base)` in the linker script did not fire, so the guard was not guarding.
/// The ELF already records the entry and the load base — this is the difference.
#[cfg(target_arch = "x86_64")]
const IMGDEC_LAYOUT: (u64, u64, u64) = include!("../../../userspace/imgdec/entry-x86_64.in");
#[cfg(target_arch = "aarch64")]
const IMGDEC_LAYOUT: (u64, u64, u64) = include!("../../../userspace/imgdec/entry-aarch64.in");

/// The compiled tenant as a loadable image.
pub fn imgdec_image() -> Blob {
    let (entry, rx, rw) = IMGDEC_LAYOUT;
    Blob { bytes: IMGDEC_BLOB, entry, rx, rw }
}

/// Run the compiled userspace tenant over `input`.
pub fn run_imgdec(task: crate::sched::TaskId, input: &[u8]) -> Result<alloc::vec::Vec<u8>, LoadError> {
    run_bulk_image(task, imgdec_image(), input)
}

/// Run the bulk tenant over `input`, returning what it wrote.
///
/// The decoder-shaped path end to end: frames the kernel owns are shared into a tenant
/// (input read-only, output writable), the tenant computes with no authority whatsoever,
/// and the kernel reads the result out of its own alias afterwards.
pub fn run_bulk(task: crate::sched::TaskId, input: &[u8]) -> Result<alloc::vec::Vec<u8>, LoadError> {
    run_bulk_with(task, bulk_blob(), input)
}

/// [`run_bulk`] with the blob named explicitly.
///
/// Parameterised so the hand-assembled tenant and the compiled Rust one run through the
/// *same* loader, mappings and teardown — which is what makes comparing their output a test
/// of the crate rather than a test of two different code paths.
pub fn run_bulk_with(
    task: crate::sched::TaskId,
    blob: &[u8],
    input: &[u8],
) -> Result<alloc::vec::Vec<u8>, LoadError> {
    // The hand-assembled blobs put their entry first by construction; a compiled one says where.
    run_bulk_at(task, blob, 0, input)
}

/// [`run_bulk_with`], entering at `entry_offset` bytes into the blob.
pub fn run_bulk_at(
    task: crate::sched::TaskId,
    blob: &[u8],
    entry_offset: u64,
    input: &[u8],
) -> Result<alloc::vec::Vec<u8>, LoadError> {
    // SAFETY of the lifetime cast: every caller passes either a `'static` blob or a slice that
    // outlives this call, and `load_image` copies the bytes before returning.
    let b = Blob {
        bytes: unsafe { core::slice::from_raw_parts(blob.as_ptr(), blob.len()) },
        entry: entry_offset,
        rx: 0x1000,
        rw: 0,
    };
    run_bulk_image(task, b, input)
}

/// [`run_bulk_at`] for an image with a real layout.
pub fn run_bulk_image(
    task: crate::sched::TaskId,
    image: Blob,
    input: &[u8],
) -> Result<alloc::vec::Vec<u8>, LoadError> {
    const PAGE: usize = 0x1000;
    let in_pages = input.len().div_ceil(PAGE).max(1);
    let mut loaded = load_image(image)?;

    // Frames for the shared buffers, taken from the global allocator and handed back at
    // the end: they stay the *kernel's*, because the kernel reads through its own alias
    // after the tenant's space is gone.
    let mut frames: alloc::vec::Vec<u64> = alloc::vec::Vec::new();
    for _ in 0..in_pages + 1 {
        match space::take_shared_frame() {
            Some(f) => frames.push(f),
            None => {
                for f in frames {
                    space::give_frame(f);
                }
                return Err(LoadError::NoAddressSpace);
            }
        }
    }
    let (in_frames, out_frames) = frames.split_at(in_pages);

    // Copy the input in through the kernel alias. Mapped **read-only** below: a decoder
    // must not be able to rewrite the bitstream it was handed, or a second pass over the
    // "same" data is not the same data.
    for (i, &f) in in_frames.iter().enumerate() {
        let off = i * PAGE;
        let n = (input.len() - off).min(PAGE);
        // SAFETY: a frame the kernel owns, reachable by its alias; `n` is bounded by both
        // the page and the remaining input.
        unsafe {
            core::ptr::copy_nonoverlapping(input.as_ptr().add(off), space::phys_to_kernel(f) as *mut u8, n);
        }
    }

    let args_va = loaded.stack_page + PAGE as u64;
    let in_va = args_va + PAGE as u64;
    let out_va = in_va + (in_pages as u64) * PAGE as u64;

    let run = (|| -> Result<alloc::vec::Vec<u8>, LoadError> {
        let args_phys = loaded.space.map_new_page(args_va, UserPerms::RW).map_err(|_| LoadError::Map)?;
        loaded.map_shared(in_va, in_frames, UserPerms::RO)?;
        loaded.map_shared(out_va, out_frames, UserPerms::RW)?;
        let args_kernel = space::phys_to_kernel(args_phys);
        // SAFETY: one page the space owns, reachable by the kernel; offsets are inside it.
        unsafe {
            let base = args_kernel as *mut u8;
            let put = |off: usize, v: u64| core::ptr::write_unaligned(base.add(off) as *mut u64, v);
            put(block::INPUT_PTR, in_va);
            put(block::INPUT_LEN, input.len() as u64);
            put(block::OUTPUT_PTR, out_va);
            put(block::OUTPUT_CAP, PAGE as u64);
            put(block::STATUS, block::STATUS_NOT_RUN);
        }
        // **No justification is set**, because this tenant makes no Synapse call and so
        // there is nothing for one to justify. That is the decoder shape in one line.
        // SAFETY: code RX, stack RW, block RW, input RO and output RW are all mapped.
        let exit =
            unsafe { crate::arch::enter_tenant(task, &loaded.space, loaded.entry, loaded.stack, args_va) };
        if !exit.is_deliberate_exit() {
            crate::ktrace::log_fmt(format_args!("tenant: bulk tenant did not exit cleanly: {exit:?}"));
            return Err(LoadError::Faulted);
        }
        // SAFETY: the loader's own block page.
        let status = unsafe {
            core::ptr::read_unaligned((args_kernel as *const u8).add(block::STATUS) as *const u64)
        };
        if status != 0 {
            // The tenant declined — a malformed input, or a panic caught by its own handler.
            // Reported as a rejection, **not** a fault: "this file is corrupt" is a different
            // fact from "userspace misbehaved", and conflating them would make a decoder's
            // ordinary error path look like a broken boundary.
            crate::ktrace::log_fmt(format_args!("tenant: bulk tenant declined, status={status:#x}"));
            return Err(LoadError::Declined { status });
        }
        // SAFETY: the loader's own page and frame; the length is clamped to the capacity
        // the tenant was given, so a larger claim cannot walk off the page.
        Ok(unsafe {
            let len = (core::ptr::read_unaligned(
                (args_kernel as *const u8).add(block::OUTPUT_LEN) as *const u64,
            ) as usize)
                .min(PAGE);
            core::slice::from_raw_parts(space::phys_to_kernel(out_frames[0]) as *const u8, len).to_vec()
        })
    })();

    // Drop the space **before** returning the frames. Freeing while a tenant mapping still
    // referenced them would hand a live cross-privilege mapping to the next allocation.
    drop(loaded);
    for &f in in_frames.iter().chain(out_frames.iter()) {
        space::give_frame(f);
    }
    run
}

/// The assembled bytes of the parameterised tenant.
pub fn call_blob() -> &'static [u8] {
    // SAFETY: as `hello_blob` — two symbols bracketing one `.rodata` run.
    unsafe {
        let start = &chitti_tenant_call_start as *const u8;
        let end = &chitti_tenant_call_end as *const u8;
        core::slice::from_raw_parts(start, end.offset_from(start) as usize)
    }
}

/// Run one Synapse call **in userspace**, as `task`, and return what the gates replied.
///
/// This is the migration primitive P9 needs: everything above it — an agent, a shell
/// command — becomes a matter of deciding *which* call to make, with no further
/// userspace machinery. The call still passes all four gates, still under the tenant's
/// own identity, and the kernel builds the justification, so nothing about the
/// authority model changes by moving the caller out of ring 0.
///
/// `Err` means the tenant could not be set up or did not exit cleanly. A *refusal* is
/// a successful run whose reply says so — the distinction matters, because "the gates
/// said no" and "userspace is broken" want different responses.
/// A loaded parameterised tenant, reusable across calls.
///
/// Building an [`AddressSpace`] costs frames and a page-table walk per page, so doing
/// it per call would put that cost on every effect an agent has. The space holds only
/// code, stack and the startup block — no per-agent state — so one can serve every
/// caller.
///
/// **The block page is zeroed before each call.** It is the only page a tenant both
/// reads and writes, so a reused space would otherwise leave one agent's call text and
/// reply readable by the next. That is a cross-agent leak, and the isolation this
/// boundary exists for is between *principals*, not only between userspace and kernel.
pub struct Tenant {
    loaded: Loaded,
    args_va: u64,
    args_kernel: u64,
}

impl Tenant {
    /// Load the parameterised blob and map its startup-block page.
    pub fn new() -> Result<Self, LoadError> {
        let mut loaded = load(call_blob())?;
        let args_va = loaded.stack_page + 0x1000;
        let args_phys = loaded.space.map_new_page(args_va, UserPerms::RW).map_err(|_| LoadError::Map)?;
        let args_kernel = space::phys_to_kernel(args_phys);
        Ok(Self { loaded, args_va, args_kernel })
    }

    /// Run `call` as `task`, justified by `justification`, and return the gates' reply.
    pub fn call(
        &mut self,
        task: crate::sched::TaskId,
        call: &str,
        justification: crate::security::taint::Justification,
    ) -> Result<alloc::string::String, LoadError> {
        if call.len() > block::CAPACITY {
            return Err(LoadError::TooBig { len: call.len() });
        }
        // SAFETY: one page this space owns, reachable by the kernel. Zeroed first so no
        // remnant of a previous caller is visible to this one; then every write is
        // inside the page (header < 64 bytes, text bounded by `CAPACITY`, reply area
        // starting at the midpoint).
        unsafe {
            core::ptr::write_bytes(self.args_kernel as *mut u8, 0, 0x1000);
            let base = self.args_kernel as *mut u8;
            let put = |off: usize, v: u64| core::ptr::write_unaligned(base.add(off) as *mut u64, v);
            put(block::CALL_PTR, self.args_va + block::TEXT as u64);
            put(block::CALL_LEN, call.len() as u64);
            put(block::REPLY_PTR, self.args_va + block::REPLY as u64);
            put(block::REPLY_CAP, block::CAPACITY as u64);
            core::ptr::copy_nonoverlapping(call.as_ptr(), base.add(block::TEXT), call.len());
        }

        crate::synapse::abi::set_run_justification(justification);
        // SAFETY: `new` mapped code RX, stack RW and the block page RW in a space that
        // shares the kernel mappings.
        let exit = unsafe {
            crate::arch::enter_tenant(task, &self.loaded.space, self.loaded.entry, self.loaded.stack, self.args_va)
        };
        // Cleared unconditionally, including on the failure path: a justification left
        // behind would be inherited by the next tenant, handing it a trust decision
        // made about content it never saw.
        crate::synapse::abi::clear_run_justification();

        if !exit.is_deliberate_exit() {
            crate::ktrace::log_fmt(format_args!("tenant: call did not exit cleanly: {exit:?}"));
            return Err(LoadError::Faulted);
        }
        // SAFETY: same page; the length is clamped to the capacity the tenant was
        // given, so a tenant that wrote a larger number cannot make us read past it.
        let reply = unsafe {
            let len = (core::ptr::read_unaligned(
                (self.args_kernel as *const u8).add(block::REPLY_LEN) as *const u64,
            ) as usize)
                .min(block::CAPACITY);
            let bytes = core::slice::from_raw_parts((self.args_kernel as *const u8).add(block::REPLY), len);
            alloc::string::String::from_utf8_lossy(bytes).into_owned()
        };
        Ok(reply)
    }
}

/// The one tenant the system reuses. See [`Tenant`] on why sharing is sound.
///
/// Single-slot because only one tenant can be running: they are pinned to the boot CPU
/// and `enter_tenant` is not reentrant. If that ever changes this becomes a per-CPU
/// slot rather than a lock — a second core entering userspace through it would be a
/// correctness bug, not merely contention.
static SHARED: crate::mm::Locked<Option<Tenant>> = crate::mm::Locked::new(None);

/// Run one Synapse call in userspace and return the **structured** outcome.
///
/// The form every migrated caller should use. A tenant's reply is prose rendered by
/// [`crate::synapse::abi`], whose vocabulary differs from the tool router's — so a
/// caller that re-parses the text to decide what happened will eventually disagree with
/// the kernel about it. That is not hypothetical: doing exactly that made
/// `security::redteam` read refusals as successes and report five injected attacks as
/// permitted. There is one authority for what an outcome was, and it is this value.
///
/// `None` means the tenant never reached the gates (it faulted, or exited without
/// invoking) — deliberately not conflated with a refusal, which *is* an outcome.
pub fn invoke_in_userspace(
    task: crate::sched::TaskId,
    call: &str,
    justification: crate::security::taint::Justification,
) -> Option<crate::synapse::executor::Invocation> {
    match call_in_userspace(task, call, justification) {
        Ok(_prose) => crate::synapse::abi::take_last_invocation(),
        Err(why) => {
            crate::ktrace::log_fmt(format_args!("tenant: userspace invoke failed: {why:?}"));
            None
        }
    }
}

/// How many Synapse calls have been made from userspace.
///
/// Exists so the migration is *observable*: "an agent's effects run in ring 3" is
/// otherwise indistinguishable from the kernel doing it, since both produce the same
/// reply and the same audit entry — which is exactly the equivalence the migration
/// aims for, and exactly why it needs a separate witness.
static USERSPACE_CALLS: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

/// Number of Synapse calls performed from ring 3 / EL0 so far.
pub fn userspace_calls() -> u64 {
    USERSPACE_CALLS.load(core::sync::atomic::Ordering::Relaxed)
}

/// Run one Synapse call in userspace on the shared tenant.
///
/// The entry point a migrated caller uses. Built on first use, so a boot that never
/// runs anything in userspace pays nothing for the capability.
pub fn call_in_userspace(
    task: crate::sched::TaskId,
    call: &str,
    justification: crate::security::taint::Justification,
) -> Result<alloc::string::String, LoadError> {
    // The lock is **not** held across the crossing: `enter_tenant` runs userspace code
    // that traps back into the kernel, where a Synapse primitive may touch anything.
    // Taking the tenant out and putting it back is the same borrow discipline the video
    // player uses when it loans its decoder to a worker.
    let mut tenant = match SHARED.with(|slot| slot.take()) {
        Some(t) => t,
        None => Tenant::new()?,
    };
    let out = tenant.call(task, call, justification);
    SHARED.with(|slot| *slot = Some(tenant));
    if out.is_ok() {
        let n = USERSPACE_CALLS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        if n == 0 {
            crate::ktrace::log("tenant", "first Synapse call performed from userspace");
        }
    }
    out
}

pub fn run_call(task: crate::sched::TaskId, call: &str) -> Result<alloc::string::String, LoadError> {
    if call.len() > block::CAPACITY {
        return Err(LoadError::TooBig { len: call.len() });
    }
    let mut loaded = load(call_blob())?;
    // A third page, after code and stack, for the block plus its text and reply.
    let args_va = loaded.stack_page + 0x1000;
    let args_phys = loaded.space.map_new_page(args_va, UserPerms::RW).map_err(|_| LoadError::Map)?;
    let args_kernel = space::phys_to_kernel(args_phys);

    // SAFETY: a freshly mapped frame this space owns, reachable by the kernel. Every
    // write is inside one page: the header is under 64 bytes, the text is bounded by
    // `CAPACITY` above, and the reply area starts at the page's midpoint.
    unsafe {
        let put = |off: usize, v: u64| core::ptr::write_unaligned((args_kernel as *mut u8).add(off) as *mut u64, v);
        put(block::CALL_PTR, args_va + block::TEXT as u64);
        put(block::CALL_LEN, call.len() as u64);
        put(block::REPLY_PTR, args_va + block::REPLY as u64);
        put(block::REPLY_CAP, block::CAPACITY as u64);
        put(block::REPLY_LEN, 0);
        core::ptr::copy_nonoverlapping(call.as_ptr(), (args_kernel as *mut u8).add(block::TEXT), call.len());
    }

    // SAFETY: `load` mapped code RX and stack RW; the block page is RW above.
    let exit = unsafe {
        crate::arch::enter_tenant(task, &loaded.space, loaded.entry, loaded.stack, args_va)
    };
    if !exit.is_deliberate_exit() {
        crate::ktrace::log_fmt(format_args!("tenant: run_call did not exit cleanly: {exit:?}"));
        return Err(LoadError::Faulted);
    }

    // SAFETY: same page, and the length is clamped to the capacity the tenant was
    // given — a tenant that wrote a larger number cannot make us read past the page.
    let reply = unsafe {
        let len = (core::ptr::read_unaligned((args_kernel as *const u8).add(block::REPLY_LEN) as *const u64) as usize)
            .min(block::CAPACITY);
        let bytes = core::slice::from_raw_parts((args_kernel as *const u8).add(block::REPLY), len);
        alloc::string::String::from_utf8_lossy(bytes).into_owned()
    };
    Ok(reply)
}
