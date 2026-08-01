//! A userspace tenant: runs in **ring 3 / EL0**, loaded by `kernel/src/synapse/tenant.rs`.
//!
//! This is where the attacker-facing decoders belong. A decoder reads a byte buffer and
//! writes a pixel buffer — it needs no authority at all, makes no Synapse call, and so
//! nothing about it has to be gated. That is why moving PNG/JPEG/H.264 out of the kernel
//! costs no new primitive: the only thing that had to be built was a way for bulk data to
//! cross the boundary, and the kernel side of that already exists.
//!
//! The prize is what happens when the input is malformed: a bad file becomes a fault the
//! kernel *reports* about a tenant it can discard, instead of a wild write in ring 0.
//!
//! # Contract with the loader
//!
//! Entered at [`USER_BASE`] with the startup block's address in the C ABI's first argument
//! register (`rdi` / `x0`). The block's layout is `synapse::tenant::block`; the fields this
//! uses are the bulk ones. It must finish by calling `Exit`, and must not assume anything
//! survives a trap beyond the callee-saved registers.
//!
//! # Deliberately minimal
//!
//! No allocator and no `panic` machinery beyond an abort: `panic = "abort"` plus a
//! `panic_handler` that exits with a non-zero status, so a Rust panic inside a decoder is
//! reported to the kernel as a rejected file rather than as a crash. A decoder that needs a
//! heap gets one from the block's `heap_ptr`/`heap_len` — the loader maps it, so the tenant
//! never asks anyone for memory.

#![no_std]
#![no_main]

/// Startup-block offsets. **Must match `synapse::tenant::block`** — the kernel asserts the
/// ones it can see, and the rest are pinned by the differential test: a wrong offset here
/// reads a neighbouring field, which is a plausible number rather than an obvious fault.
mod block {
    pub const INPUT_PTR: usize = 40;
    pub const INPUT_LEN: usize = 48;
    pub const OUTPUT_PTR: usize = 56;
    pub const OUTPUT_CAP: usize = 64;
    pub const OUTPUT_LEN: usize = 72;
    pub const STATUS: usize = 96;
}

/// ABI entry numbers, from `synapse::abi::Entry`.
const ENTRY_EXIT: u64 = 2;

/// Status codes written to the block. 0 is success; everything else says the tenant
/// declined, which is *not* the same as faulting.
const STATUS_OK: u64 = 0;
const STATUS_PANIC: u64 = 1;
const STATUS_OUTPUT_TOO_SMALL: u64 = 2;

/// Leave EL0 for good. Never returns.
fn exit() -> ! {
    // SAFETY: the one ABI call this tenant makes. The reply registers are zeroed because a
    // trap clobbers the caller-saved set, and handing the kernel stale values as a reply
    // buffer is the defect that once had a tenant overwrite its own code page.
    unsafe {
        #[cfg(target_arch = "x86_64")]
        core::arch::asm!(
            "syscall",
            in("rdi") ENTRY_EXIT, in("r8") 0u64, in("r9") 0u64,
            options(noreturn, nostack)
        );
        #[cfg(target_arch = "aarch64")]
        core::arch::asm!(
            "svc #0",
            in("x0") ENTRY_EXIT, in("x3") 0u64, in("x4") 0u64,
            options(noreturn, nostack)
        );
    }
}

/// A panic in a decoder is a **rejected file**, not a crash.
///
/// Rust's bounds checks are a decoder's last line of defence against a malformed input, and
/// in the kernel tripping one halts the machine. Here it just leaves EL0, and the *absence*
/// of a success marker is what the kernel reads as failure — see `STATUS_NOT_RUN`.
///
/// It deliberately writes nothing. An earlier version stashed the block pointer in a
/// `static mut` so it could report a status, and that single mutable global put a writable
/// object in the code page — which the loader maps **RX**, so the tenant faulted on its own
/// first instruction (`PAGE FAULT accessing USER_BASE+104, error=0x7`, a write to a present
/// read-only page). **This blob has no writable data at all**, which is why one RX page
/// plus a stack is a sufficient layout. A decoder that genuinely needs statics or a heap
/// requires the loader to learn a text/data split first; do not reintroduce one by accident.
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    exit()
}

/// # Safety
/// `blk` must point at the startup block and `off` be inside its page.
unsafe fn put(blk: *mut u8, off: usize, v: u64) {
    unsafe { core::ptr::write_unaligned(blk.add(off) as *mut u64, v) }
}

/// # Safety
/// As [`put`].
unsafe fn get(blk: *const u8, off: usize) -> u64 {
    unsafe { core::ptr::read_unaligned(blk.add(off) as *const u64) }
}

#[unsafe(no_mangle)]
pub extern "C" fn _start(blk: *mut u8) -> ! {
    // SAFETY: the loader passes the block's address in the first argument register, mapped
    // read-write for the whole run. The pointer is threaded through as an argument rather
    // than stashed in a static — see the `panic_handler` note on why this blob has no
    // writable data.
    unsafe {
        let input =
            core::slice::from_raw_parts(get(blk, block::INPUT_PTR) as *const u8, get(blk, block::INPUT_LEN) as usize);
        let out_cap = get(blk, block::OUTPUT_CAP) as usize;
        let out = core::slice::from_raw_parts_mut(get(blk, block::OUTPUT_PTR) as *mut u8, out_cap);
        match run(input, out) {
            Ok(n) => {
                put(blk, block::OUTPUT_LEN, n as u64);
                put(blk, block::STATUS, STATUS_OK);
            }
            Err(status) => {
                put(blk, block::OUTPUT_LEN, 0);
                put(blk, block::STATUS, status);
            }
        }
    }
    exit()
}

/// The tenant's actual work: for now a checksum, so the boundary can be tested
/// **differentially** against the kernel computing the same thing over the same bytes.
///
/// This is the seam the PNG decoder drops into. Keeping it a plain
/// `fn(&[u8], &mut [u8]) -> Result<usize, u64>` is the point: it is ordinary safe Rust with
/// no access to anything, which is what the whole boundary exists to arrange.
fn run(input: &[u8], out: &mut [u8]) -> Result<usize, u64> {
    if out.len() < 8 {
        return Err(STATUS_OUTPUT_TOO_SMALL);
    }
    let mut sum: u64 = 0;
    for &b in input {
        sum += b as u64;
    }
    out[..8].copy_from_slice(&sum.to_le_bytes());
    Ok(8)
}
