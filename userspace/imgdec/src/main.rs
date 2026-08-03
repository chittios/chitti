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
//! **It is entered more than once.** The loader keeps a tenant loaded across decodes, because
//! building the address space was the whole measured cost of ring 3 — so every mutable static
//! here is reset at `_start` rather than trusted to be zero. A reused space keeps its `.bss`.
//!
//! Two output modes, chosen by whether the loader mapped an output buffer:
//!
//! * `output_cap > 0` — write into it and report the length. Used by the boundary tests, whose
//!   payload is a checksum rather than a decode.
//! * `output_cap == 0` — **heap-output**: leave the pixels in the arena and report where they
//!   are, plus the image's dimensions. This is what a decoder wants, because how many pixels an
//!   image has is a number inside the file: a loader that had to size an output buffer first
//!   would have to parse the header in the kernel, which is the thing this crate exists to undo.
//!
//! # Deliberately minimal
//!
//! A bump allocator over the loader's heap and no `panic` machinery beyond an abort:
//! `panic = "abort"` plus a `panic_handler` that writes a status word, so a Rust panic inside a
//! decoder — a tripped bounds check on a malformed file — is reported to the kernel as a
//! rejected file rather than as a crash. The heap comes from the block's `heap_ptr`/`heap_len`;
//! the tenant never asks anyone for memory, and when the arena is too small it says so and lets
//! the loader map more.

#![no_std]
#![no_main]
// The tenant reports *why* it stopped, and "the arena was too small" is a status the loader
// acts on by mapping more — so the allocation-failure path is written explicitly rather than
// left to whatever `alloc`'s default handler does with a null (panic in some versions, abort in
// others; an abort would reach the kernel as a **fault**, i.e. as "the decoder is broken").
#![feature(alloc_error_handler)]

// The mounted kernel modules and the decoder's own buffers are `alloc`-based; the tenant
// provides the allocator below, over the heap the loader mapped for it.
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
    pub const HEAP_PTR: usize = 80;
    pub const HEAP_LEN: usize = 88;
    pub const STATUS: usize = 96;
    /// Written by the tenant in the **heap-output** mode: the decoded image's size.
    pub const IMG_W: usize = 104;
    pub const IMG_H: usize = 112;
    /// Written by the tenant **before it decodes**: the arena this input looks like it needs.
    pub const HEAP_WANT: usize = 120;
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
/// The arena ran out. **Distinct from a rejection on purpose**: the kernel cannot know how
/// much heap an image needs before it is parsed (that number is inside the file), so the
/// only honest protocol is for the tenant to say "not enough" and for the loader to map
/// more and try again. Folded into `STATUS_DECODE_FAILED` it would read as "corrupt file",
/// and a perfectly good photo would be reported as broken.
const STATUS_OUT_OF_MEMORY: u64 = 4;

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

/// Baseline JPEG, mounted the same way and for the same reason. It is the larger attacker
/// surface of the two — entropy-coded data, per-MCU Huffman decode, quantisation tables the
/// file chooses — and it arrives here with no authority and no kernel memory in reach.
#[path = "../../../kernel/src/image/jpeg.rs"]
pub mod jpeg;

/// The PNG signature and the JPEG SOI marker, from the formats themselves.
const PNG_SIG: &[u8] = &[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
const JPEG_SOI: &[u8] = &[0xff, 0xd8];

/// Sniff the container and decode. **The same dispatch as `image::decode`**, deliberately: a
/// tenant that accepted a format the kernel's own path does not, or vice versa, would make the
/// differential test compare two different questions.
fn decode_image(input: &[u8]) -> Result<Image, u64> {
    if input.starts_with(PNG_SIG) {
        png::decode(input).map_err(|_| STATUS_DECODE_FAILED)
    } else if input.starts_with(JPEG_SOI) {
        jpeg::decode(input).map_err(|_| STATUS_DECODE_FAILED)
    } else {
        Err(STATUS_DECODE_FAILED)
    }
}

/// How much arena this input looks like it will need, **from its header alone**.
///
/// Reported to the loader before any decoding, and this is the difference between a large photo
/// decoding and being refused. Without it the loader can only *guess* — it doubles the arena and
/// re-enters, so a 32 MP image needs six full attempts to walk from 8 MiB up, each one doing real
/// work before running out; with it there is one retry at the right size. Nothing here decides
/// anything, so a wrong answer costs a retry, never a wrong picture: too small and the loader
/// falls back to doubling, too large and it is capped by the ceiling and the free-frame guard.
///
/// **This is the header parse that belongs in ring 3.** The loader could have read these same
/// bytes itself and skipped the round trip — and that is exactly the mistake: it would put an
/// attacker-shaped parse back in the kernel to save a page-table walk. Every access below is
/// bounds-checked and returns `None` rather than assuming, because an estimate that panicked
/// would reject a file the decoder could have read.
fn arena_estimate(input: &[u8]) -> usize {
    const SLACK: usize = 1 << 20;
    // A doubling `Vec` can hold up to twice what it needs, and both of the big buffers below are
    // grown that way, so the factor is real rather than defensive.
    let two = |n: usize| n.saturating_mul(2);
    if input.starts_with(PNG_SIG) {
        if let Some((w, h, channels, depth)) = png_geometry(input) {
            let stride = w.saturating_mul(channels).saturating_mul(depth).div_ceil(8);
            let raw = h.saturating_mul(stride.saturating_add(1));
            return two(input.len())
                .saturating_add(two(raw))
                .saturating_add(w.saturating_mul(h).saturating_mul(4))
                .saturating_add(SLACK);
        }
    } else if input.starts_with(JPEG_SOI) {
        if let Some((w, h)) = jpeg_geometry(input) {
            // Component planes (~1.5x the pixels at 4:2:0, padded up to whole MCUs) plus the
            // 4-byte-per-pixel output; 8 covers both with room for the padding.
            return two(input.len())
                .saturating_add(w.saturating_mul(h).saturating_mul(8))
                .saturating_add(SLACK);
        }
    }
    // No idea: 0 means "no estimate", and the loader goes back to doubling.
    0
}

/// A PNG's geometry from IHDR, which the format requires to be the first chunk.
fn png_geometry(b: &[u8]) -> Option<(usize, usize, usize, usize)> {
    if b.get(12..16)? != b"IHDR" {
        return None;
    }
    let be = |o: usize| -> Option<usize> {
        let s = b.get(o..o + 4)?;
        Some(u32::from_be_bytes([s[0], s[1], s[2], s[3]]) as usize)
    };
    let (w, h) = (be(16)?, be(20)?);
    let depth = *b.get(24)? as usize;
    let channels = match *b.get(25)? {
        0 | 3 => 1,
        2 => 3,
        4 => 2,
        6 => 4,
        _ => return None,
    };
    // The same bounds `png::decode` enforces, so an absurd header produces no estimate rather
    // than an absurd one.
    if w == 0 || h == 0 || w > 16384 || h > 16384 || !matches!(depth, 1 | 2 | 4 | 8 | 16) {
        return None;
    }
    Some((w, h, channels, depth))
}

/// A baseline JPEG's geometry from SOF0/SOF1, found by walking the marker segments.
fn jpeg_geometry(b: &[u8]) -> Option<(usize, usize)> {
    let mut i = 2usize;
    while i + 4 <= b.len() {
        if *b.get(i)? != 0xff {
            i += 1;
            continue;
        }
        let m = *b.get(i + 1)?;
        match m {
            // Fill byte, or a marker that carries no length.
            0xff => {
                i += 1;
                continue;
            }
            0x01 | 0xd8 | 0xd0..=0xd7 => {
                i += 2;
                continue;
            }
            // The scan, or the end: past here there is no frame header to find.
            0xda | 0xd9 => return None,
            0xc0 | 0xc1 => {
                let h = ((*b.get(i + 5)? as usize) << 8) | *b.get(i + 6)? as usize;
                let w = ((*b.get(i + 7)? as usize) << 8) | *b.get(i + 8)? as usize;
                return if w == 0 || h == 0 { None } else { Some((w, h)) };
            }
            _ => {}
        }
        let len = ((*b.get(i + 2)? as usize) << 8) | *b.get(i + 3)? as usize;
        i += 2 + len.max(2);
    }
    None
}

/// Bump arena over the **loader-mapped heap** named in the startup block.
///
/// `png.rs` allocates — the inflated scanlines, the pixel buffer — so the tenant needs a heap.
/// It never frees: a decode is one pass, and the arena is reset at `_start`, so a free list
/// would be code with no purpose.
///
/// **Why not a `.bss` array, which is what this was.** A static arena is part of the image's
/// RW range, so `load_image` maps every byte of it *on every run* — a 4 MiB arena was 1025
/// frame allocations and page-table walks per decode, and the size was a compile-time constant
/// that had to be big enough for the largest image anyone would open and small enough to map
/// per run. Neither half of that trade is necessary once the loader supplies the heap: it maps
/// it once for a reused tenant, and it can map **more** when the tenant reports
/// [`STATUS_OUT_OF_MEMORY`], which is the only way to size a heap against a number that is
/// inside the file being parsed.
static mut HEAP_BASE: usize = 0;
static mut HEAP_LEN: usize = 0;
/// Bytes handed out so far. **Reset at `_start`, not merely zero-initialised** — that is the
/// one line a reused address space needs and the one nothing about the failure would point to:
/// `.bss` is zeroed by *fresh* frames, so a second decode into a reused space would start with
/// the arena already full and fail as though the image were too large.
static mut CURSOR: usize = 0;
/// Set when the arena could not satisfy an allocation, so the panic that Rust raises out of a
/// null `alloc` can be reported as "map me more heap" rather than as a corrupt file.
static mut OOM: bool = false;
/// The startup block, for the panic handler. See its doc comment: this is safe *now* only
/// because the image has a real RW range; it was not when the whole blob was one RX page.
static mut BLOCK: *mut u8 = core::ptr::null_mut();

struct Bump;

// SAFETY: single-threaded tenant, one decode at a time. `alloc` hands out disjoint aligned
// slices from the loader's heap; the only reuse is of the **top** allocation, which by
// definition nothing later overlaps. Every returned pointer is inside
// `[HEAP_BASE, HEAP_BASE + HEAP_LEN)`, which the loader mapped read-write; an allocation that
// would leave it returns null instead of wrapping.
unsafe impl core::alloc::GlobalAlloc for Bump {
    unsafe fn alloc(&self, l: core::alloc::Layout) -> *mut u8 {
        unsafe {
            let base = HEAP_BASE;
            let start = (base + CURSOR).next_multiple_of(l.align().max(1));
            let end = start.saturating_sub(base).saturating_add(l.size());
            if base == 0 || end > HEAP_LEN {
                // Out of arena. Rust turns a null here into `handle_alloc_error`, and the flag is
                // what tells the handler which status to report.
                OOM = true;
                return core::ptr::null_mut();
            }
            CURSOR = end;
            start as *mut u8
        }
    }

    /// Free **only the top allocation**, by rolling the cursor back.
    ///
    /// The cheapest possible reclaim, and it happens to be the one a decoder needs: scratch is
    /// allocated and dropped in LIFO order (a scanline buffer, an intermediate `Vec` that is
    /// consumed and released), so rolling back turns a permanent leak into free reuse. Without
    /// it a bump arena's size is the *sum* of every temporary a decode ever made rather than its
    /// high-water mark — which for a large image was several times the picture.
    unsafe fn dealloc(&self, p: *mut u8, l: core::alloc::Layout) {
        unsafe {
            if HEAP_BASE != 0 && (p as usize).saturating_add(l.size()) == HEAP_BASE + CURSOR {
                CURSOR = (p as usize) - HEAP_BASE;
            }
        }
    }

    /// Grow or shrink **in place** when the block is on top of the arena.
    ///
    /// This is the other half, and the more valuable one: `Vec` growth doubles, and the default
    /// `realloc` (allocate, copy, free) leaves every intermediate size stranded — a `Vec` grown
    /// to N bytes by doubling costs ~2N of arena and copies N bytes for nothing. The buffers that
    /// grow this way are the largest a decoder has (the accumulated compressed data, the inflate
    /// output), so in-place growth is most of the difference between an arena sized like the
    /// image and an arena sized like several copies of it.
    unsafe fn realloc(&self, p: *mut u8, l: core::alloc::Layout, new_size: usize) -> *mut u8 {
        unsafe {
            let top = HEAP_BASE != 0 && (p as usize).saturating_add(l.size()) == HEAP_BASE + CURSOR;
            if top {
                let end = (p as usize).saturating_sub(HEAP_BASE).saturating_add(new_size);
                if end <= HEAP_LEN {
                    CURSOR = end;
                    return p;
                }
                // A top block that cannot grow will not fit anywhere else either — the arena
                // ends here. Report it rather than falling through to a copy that must fail.
                OOM = true;
                return core::ptr::null_mut();
            }
            // Not on top: the ordinary allocate-copy-free, with the copy bounded by the smaller
            // of the two sizes as `GlobalAlloc` requires.
            let new = self.alloc(core::alloc::Layout::from_size_align_unchecked(new_size, l.align()));
            if !new.is_null() {
                core::ptr::copy_nonoverlapping(p, new, l.size().min(new_size));
                self.dealloc(p, l);
            }
            new
        }
    }
}

#[global_allocator]
static ALLOC: Bump = Bump;

/// An allocation the arena could not satisfy: report it and leave.
///
/// The one failure a decoder has that is **neither** the file's fault nor the decoder's — the
/// loader guessed the arena size, and only the file knows the right one. Reported as its own
/// status so the loader can map more and re-enter, which it does; anything else here (a panic
/// message, an abort) would arrive at the kernel as "corrupt file" or "the tenant faulted", and
/// a good photo would be rejected for want of a page.
#[alloc_error_handler]
fn on_alloc_failure(_layout: core::alloc::Layout) -> ! {
    // SAFETY: single-threaded tenant; `BLOCK` is the loader's block page for this run, or null
    // if the failure somehow preceded `_start`'s first store (the loader's `STATUS_NOT_RUN`
    // then stands, which is still a failure and still fail-closed).
    unsafe {
        if !BLOCK.is_null() {
            put(BLOCK, block::OUTPUT_LEN, 0);
            put(BLOCK, block::STATUS, STATUS_OUT_OF_MEMORY);
        }
    }
    exit()
}

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
/// in the kernel tripping one halts the machine. Here it just leaves EL0, and the kernel reads
/// the status word — which is fail-closed (`STATUS_NOT_RUN`), so even a panic before this
/// handler could write anything is reported as a failure rather than as an empty success.
///
/// It reports *which* failure, and that distinction is load-bearing: a null allocation and a
/// tripped bounds check both arrive here as panics, but "map me more heap" and "this file is
/// corrupt" are opposite answers, and the loader retries only one of them.
///
/// Writing through a `static mut` here was impossible until the image had a text/data split:
/// the first version of this blob was a single RX page, the block pointer went into a mutable
/// global, the linker put it in the code page, and the tenant faulted on its own first
/// instruction writing it (`error=0x7`, a write to a present read-only page).
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    // SAFETY: single-threaded tenant. `BLOCK` is either null (the panic came before `_start`
    // recorded it, in which case the loader's fail-closed `STATUS_NOT_RUN` stands) or the
    // block page the loader mapped read-write for this run.
    unsafe {
        if !BLOCK.is_null() {
            put(BLOCK, block::OUTPUT_LEN, 0);
            put(BLOCK, block::STATUS, if OOM { STATUS_OUT_OF_MEMORY } else { STATUS_PANIC });
        }
    }
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
    // read-write for the whole run, and the heap it names is mapped read-write too.
    unsafe {
        // **Everything mutable is reset here, not merely zero-initialised.** A tenant is
        // entered more than once now — reusing the address space is what removes the
        // per-decode cost of building one — and a reused space keeps its `.bss`. So a
        // cursor left at the end of the previous decode makes the next one fail as though
        // the image were too large, and a stale `OOM` flag mislabels the reason.
        BLOCK = blk;
        HEAP_BASE = get(blk, block::HEAP_PTR) as usize;
        HEAP_LEN = get(blk, block::HEAP_LEN) as usize;
        CURSOR = 0;
        OOM = false;

        let input =
            core::slice::from_raw_parts(get(blk, block::INPUT_PTR) as *const u8, get(blk, block::INPUT_LEN) as usize);
        // Told to the loader **before** decoding, so it is already there if the decode runs out.
        // Computing it in the failure path instead would mean parsing the header while unwinding
        // from an allocation failure, which is the worst place to do anything.
        put(blk, block::HEAP_WANT, arena_estimate(input) as u64);

        let out_cap = get(blk, block::OUTPUT_CAP) as usize;
        if out_cap == 0 {
            // **Heap-output mode.** No output buffer was mapped, so the answer stays where it
            // was built and the tenant reports its address. The loader owns the heap frames and
            // reads the pixels out of its own alias, which means the pixel buffer is never
            // copied inside the tenant and the loader never has to guess how big an output
            // buffer an unparsed image needs.
            match decode_image(input) {
                Ok(img) => {
                    let (w, h) = (img.w, img.h);
                    // Not dropped: `dealloc` is a no-op, but leaking deliberately says that
                    // the buffer outlives the value and is read by the kernel afterwards.
                    let px = core::mem::ManuallyDrop::new(img.pixels);
                    put(blk, block::OUTPUT_PTR, px.as_ptr() as u64);
                    put(blk, block::OUTPUT_LEN, (px.len() * 4) as u64);
                    put(blk, block::IMG_W, w as u64);
                    put(blk, block::IMG_H, h as u64);
                    put(blk, block::STATUS, STATUS_OK);
                }
                Err(status) => {
                    put(blk, block::OUTPUT_LEN, 0);
                    put(blk, block::STATUS, status);
                }
            }
        } else {
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
    }
    exit()
}

/// The buffer-output mode: decode into the loader's output buffer, or — for a payload that is
/// not an image — checksum it, so the boundary can be tested **differentially** against the
/// kernel computing the same thing over the same bytes.
///
/// This is the seam the PNG decoder drops into. Keeping it a plain
/// `fn(&[u8], &mut [u8]) -> Result<usize, u64>` is the point: it is ordinary safe Rust with
/// no access to anything, which is what the whole boundary exists to arrange.
fn run(input: &[u8], out: &mut [u8]) -> Result<usize, u64> {
    // The whole tenant, and the point of all of it: attacker-supplied bytes parsed **outside the
    // kernel**. A malformed PNG that would have been a halted machine is now either a clean
    // `Err` or, if it trips a bounds check, a wasm-style trap the kernel reports about a tenant
    // it can discard.
    // **Not an image?** Fall back to the checksum the boundary tests use. Keeping it is
    // deliberate: it is the payload `tenant::self_test` and the triple differential run, and it
    // exercises the crossing without depending on a decoder — so a decoder bug and a boundary
    // bug stay distinguishable. The signature check is the file format's own, not a guess.
    if !input.starts_with(PNG_SIG) && !input.starts_with(JPEG_SOI) {
        return checksum(input, out);
    }
    let img = decode_image(input)?;
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
