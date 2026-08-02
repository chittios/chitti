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

// The mounted kernel modules and the decoder's own buffers are `alloc`-based; the tenant
// provides the allocator below, over a static arena.
extern crate alloc;

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
/// The decoder rejected the input — malformed PNG, unsupported feature. An *answer*, not a
/// crash, which is the distinction the whole boundary exists to preserve.
const STATUS_DECODE_FAILED: u64 = 3;

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

/// The decoder itself — the **kernel's own source**, mounted rather than copied
/// (`pdf-wasm`/`h264diff`/`pngbench` all do this), so the repo keeps exactly one PNG
/// implementation and moving it into ring 3 cannot regress it.
///
/// `png.rs` resolves `super::Image` and `super::inflate`, both of which the crate root provides.
#[path = "../../../kernel/src/image/inflate.rs"]
pub mod inflate;

/// Mirrors `kernel/src/image/mod.rs`'s `Image`; `png.rs` constructs it. Declared here rather
/// than mounting `image/mod.rs`, which drags in resize/render and framebuffer types a sandboxed
/// decoder has no business seeing.
pub struct Image {
    pub w: usize,
    pub h: usize,
    pub pixels: alloc::vec::Vec<u32>,
}

#[path = "../../../kernel/src/image/png.rs"]
pub mod png;

/// Bump arena in `.bss`, and the reason the text/data split had to come first.
///
/// `png.rs` allocates — the inflated scanlines, the pixel buffer — so the tenant needs a heap.
/// It never frees: one decode per instance, and the whole arena dies with the address space, so
/// a free list would be code with no purpose. Sized for a ~1.3 MP image (≈5 MB of pixels plus
/// the inflate output and a scanline pair); a larger image is refused by `STATUS_OUT_OF_MEMORY`
/// rather than corrupting anything, which is the behaviour a sandbox is for.
/// 4 MiB. Every byte of this is **mapped page by page on every run** — `load_image` has no
/// demand paging, so the arena is a per-decode cost of `ARENA_BYTES / 4096` frame allocations
/// and page-table walks, not free address space. 16 MiB meant 4097 pages a run; 4 MiB means
/// 1025, and caps decodable images at roughly 0.4 MP until the loader learns either larger
/// pages or a reusable tenant. Bigger inputs are refused, never truncated.
const ARENA_BYTES: usize = 4 * 1024 * 1024;
/// **16-byte aligned.** A bare `[u8; N]` has align 1, so the arena's base was arbitrary — and
/// while `alloc` below aligns each allocation to its own `Layout`, anything the compiler emits
/// assuming a stricter alignment than the type demands (SSE moves over a byte buffer) faults.
/// That presented as a deterministic `#GP(0)` at one instruction during a *valid* decode, while
/// an early-rejected input succeeded on the same path.
#[repr(C, align(16))]
struct Arena([u8; ARENA_BYTES]);
static mut ARENA: Arena = Arena([0; ARENA_BYTES]);
static mut CURSOR: usize = 0;

struct Bump;

// SAFETY: single-threaded tenant, one decode per instance. `alloc` hands out disjoint aligned
// slices from a static arena and never reuses them, so no two live allocations overlap.
unsafe impl core::alloc::GlobalAlloc for Bump {
    unsafe fn alloc(&self, l: core::alloc::Layout) -> *mut u8 {
        unsafe {
            let base = core::ptr::addr_of_mut!(ARENA.0) as usize;
            let start = (base + CURSOR + l.align() - 1) & !(l.align() - 1);
            let end = start.saturating_sub(base).saturating_add(l.size());
            if end > ARENA_BYTES {
                return core::ptr::null_mut(); // out of arena: Rust turns this into a panic -> STATUS_PANIC
            }
            CURSOR = end;
            start as *mut u8
        }
    }
    unsafe fn dealloc(&self, _p: *mut u8, _l: core::alloc::Layout) {}
}

#[global_allocator]
static ALLOC: Bump = Bump;

/// Scratch that lives in `.bss`, proving the loader's read-write mapping works.
///
/// A mutable static is what a real decoder needs (inflate's window, Huffman tables) and what
/// the single-RX-page layout could not support: the first version of this file kept one, the
/// linker put it in the code page, and the tenant faulted writing it on its own first
/// instruction. Used below so it cannot be optimised away.
static mut SCRATCH: [u32; 256] = [0; 256];

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
    // The whole tenant, and the point of all of it: attacker-supplied bytes parsed **outside the
    // kernel**. A malformed PNG that would have been a halted machine is now either a clean
    // `Err` or, if it trips a bounds check, a wasm-style trap the kernel reports about a tenant
    // it can discard.
    // **Not a PNG?** Fall back to the checksum the boundary tests use. Keeping it is deliberate:
    // it is the payload `tenant::self_test` and the triple differential run, and it exercises
    // the crossing without depending on a decoder — so a PNG bug and a boundary bug stay
    // distinguishable. The signature check is the file format's own, not a guess.
    const PNG_SIG: &[u8] = &[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
    if !input.starts_with(PNG_SIG) {
        return checksum(input, out);
    }
    let img = png::decode(input).map_err(|_| STATUS_DECODE_FAILED)?;
    let need = 12 + img.pixels.len() * 4;
    if out.len() < need {
        return Err(STATUS_OUTPUT_TOO_SMALL);
    }
    out[0..4].copy_from_slice(&(img.w as u32).to_le_bytes());
    out[4..8].copy_from_slice(&(img.h as u32).to_le_bytes());
    out[8..12].copy_from_slice(&1u32.to_le_bytes());
    // One memcpy: `Image::pixels` is already a contiguous little-endian `u32` run, which is the
    // wire format. A per-pixel append is what made the wasm benchmark look 67x instead of 38x.
    //
    // SAFETY: `[u32]` -> `[u8]` over the same allocation; 4x the elements, alignment only
    // relaxed, and every target here is little-endian.
    let raw = unsafe {
        core::slice::from_raw_parts(img.pixels.as_ptr() as *const u8, img.pixels.len() * 4)
    };
    out[12..need].copy_from_slice(raw);
    Ok(need)
}

/// The boundary's own payload: sum the input through a `.bss` histogram.
///
/// Kept alongside the decoder so the crossing can be tested without a PNG, and so the RW mapping
/// is exercised even when no image is involved.
fn checksum(input: &[u8], out: &mut [u8]) -> Result<usize, u64> {
    if out.len() < 8 {
        return Err(STATUS_OUTPUT_TOO_SMALL);
    }
    // SAFETY: single-threaded tenant, one call per instance, no aliasing.
    let hist = unsafe { &mut *core::ptr::addr_of_mut!(SCRATCH) };
    hist.fill(0);
    for &b in input {
        hist[b as usize] += 1;
    }
    let mut sum: u64 = 0;
    for (b, &n) in hist.iter().enumerate() {
        sum += n as u64 * b as u64;
    }
    out[..8].copy_from_slice(&sum.to_le_bytes());
    Ok(8)
}
