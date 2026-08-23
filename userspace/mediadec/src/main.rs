//! Ring-3 **chunked** media tenant: WAV/MP3/AAC → PCM, H.264/HEVC/VP9 → YUV.
//!
//! Unlike [`imgdec`], `_start` does **not** reset the bump cursor or decoder
//! statics on [`ChunkCmd::Continue`]. Decoder state survives an entry; the
//! kernel pumps `upkeep` / Ctrl+C between crossings. YUV→RGB stays in the
//! kernel. This tenant has no device access and makes one ABI call: Exit.

#![no_std]
#![no_main]
#![feature(alloc_error_handler)]

extern crate alloc;

#[path = "../../../kernel/src/synapse/chunked.rs"]
mod chunked;

/// Kernel audio tree. `hud` is cfg-gated out via `--cfg chitti_tenant`.
#[path = "../../../kernel/src/audio/mod.rs"]
mod audio;

/// Kernel video tree (bitstream → YUV). `frame_from_yuv` is never called.
#[path = "../../../kernel/src/video/mod.rs"]
mod video;

mod session;

/// Host facades the mounted kernel modules resolve via `crate::…`.
#[allow(dead_code)]
mod shell {
    pub fn upkeep() {}
    pub fn poll_interrupt() -> bool {
        false
    }
}

mod sound {
    pub mod mel {
        /// Copied from `kernel/src/sound/mel.rs` — not `f32::cos` (`std`).
        pub fn cosf_pub(x: f32) -> f32 {
            cosf(x)
        }
        fn cosf(x: f32) -> f32 {
            use core::f32::consts::FRAC_PI_2;
            let k = floorf(x / FRAC_PI_2 + 0.5) as i32;
            let r = x - k as f32 * FRAC_PI_2;
            let (c, s) = cos_sin_small(r);
            match k.rem_euclid(4) {
                0 => c,
                1 => -s,
                2 => -c,
                _ => s,
            }
        }
        fn cos_sin_small(r: f32) -> (f32, f32) {
            let r2 = r * r;
            let c = 1.0 - r2 * (0.5 - r2 * (1.0 / 24.0 - r2 / 720.0));
            let s = r * (1.0 - r2 * (1.0 / 6.0 - r2 * (1.0 / 120.0 - r2 / 5040.0)));
            (c, s)
        }
        fn floorf(x: f32) -> f32 {
            let t = x as i64 as f32;
            if t > x { t - 1.0 } else { t }
        }
    }
}

mod cortex {
    pub mod tensor {
        /// Copied from `kernel/src/cortex/tensor.rs`.
        pub fn libm_sqrtf(x: f32) -> f32 {
            #[cfg(target_arch = "x86_64")]
            // SAFETY: `sqrtss` has no side effects; input is non-negative.
            unsafe {
                let mut r = x;
                core::arch::asm!(
                    "sqrtss {r}, {r}",
                    r = inout(xmm_reg) r,
                    options(nomem, nostack, preserves_flags)
                );
                r
            }
            #[cfg(target_arch = "aarch64")]
            // SAFETY: `fsqrt` has no side effects; input is non-negative.
            unsafe {
                let r: f32;
                core::arch::asm!(
                    "fsqrt {r:s}, {x:s}",
                    r = out(vreg) r,
                    x = in(vreg) x,
                    options(nomem, nostack, preserves_flags)
                );
                r
            }
            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            {
                if x <= 0.0 {
                    return 0.0;
                }
                let mut g = x;
                for _ in 0..20 {
                    g = 0.5 * (g + x / g);
                }
                g
            }
        }
    }
}

/// SMP facade so `video::mt` compiles. Always single-threaded: a tenant is
/// pinned to the boot CPU (no TLB shootdown).
#[allow(dead_code)]
mod arch {
    pub fn online_cpus() -> usize {
        1
    }
    /// # Safety
    /// Never called when `online_cpus() == 1`.
    pub unsafe fn parallel_for(
        _n: usize,
        _chunk: usize,
        _f: unsafe fn(usize, usize, *mut u8),
        _ctx: *mut u8,
    ) {
    }
}

use chunked::{ChunkBlock, ChunkCmd, ChunkStatus};

const ENTRY_EXIT: u64 = 2;

fn exit() -> ! {
    // SAFETY: the one ABI call this tenant makes. Reply registers zeroed — a
    // trap clobbers caller-saved and stale values once overwrote a code page.
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

static mut HEAP_BASE: usize = 0;
static mut HEAP_LEN: usize = 0;
static mut CURSOR: usize = 0;
static mut OOM: bool = false;
static mut BLOCK: *mut u8 = core::ptr::null_mut();
static mut LAST_HEAP_WANT: u64 = 0;

struct Bump;

// SAFETY: single-threaded tenant. Allocations are disjoint aligned slices of
// the loader heap; only the top block is reused on free/grow.
unsafe impl core::alloc::GlobalAlloc for Bump {
    unsafe fn alloc(&self, l: core::alloc::Layout) -> *mut u8 {
        unsafe {
            let base = HEAP_BASE;
            let start = (base + CURSOR).next_multiple_of(l.align().max(1));
            let end = start.saturating_sub(base).saturating_add(l.size());
            if base == 0 || end > HEAP_LEN {
                OOM = true;
                return core::ptr::null_mut();
            }
            CURSOR = end;
            start as *mut u8
        }
    }

    unsafe fn dealloc(&self, p: *mut u8, l: core::alloc::Layout) {
        unsafe {
            if HEAP_BASE != 0 && (p as usize).saturating_add(l.size()) == HEAP_BASE + CURSOR {
                CURSOR = (p as usize) - HEAP_BASE;
            }
        }
    }

    unsafe fn realloc(&self, p: *mut u8, l: core::alloc::Layout, new_size: usize) -> *mut u8 {
        unsafe {
            let top = HEAP_BASE != 0 && (p as usize).saturating_add(l.size()) == HEAP_BASE + CURSOR;
            if top {
                let end = (p as usize).saturating_sub(HEAP_BASE).saturating_add(new_size);
                if end <= HEAP_LEN {
                    CURSOR = end;
                    return p;
                }
                OOM = true;
                return core::ptr::null_mut();
            }
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

#[alloc_error_handler]
fn on_alloc_failure(_layout: core::alloc::Layout) -> ! {
    // SAFETY: single-threaded; BLOCK is the loader page or null.
    unsafe {
        if !BLOCK.is_null() {
            write_status(BLOCK, ChunkStatus::OutOfMemory, LAST_HEAP_WANT);
        }
    }
    exit()
}

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    unsafe {
        if !BLOCK.is_null() {
            let st = if OOM { ChunkStatus::OutOfMemory } else { ChunkStatus::Failed };
            write_status(BLOCK, st, LAST_HEAP_WANT);
        }
    }
    exit()
}

/// # Safety
/// `blk` is the loader's block page, at least [`ChunkBlock::BYTES`] long.
unsafe fn write_status(blk: *mut u8, st: ChunkStatus, heap_want: u64) {
    unsafe {
        core::ptr::write_unaligned(blk as *mut u64, st as u64);
        // heap_want is word 7.
        core::ptr::write_unaligned(blk.add(7 * 8) as *mut u64, heap_want);
    }
}

unsafe fn load_block(blk: *mut u8) -> Option<ChunkBlock> {
    let s = unsafe { core::slice::from_raw_parts(blk, ChunkBlock::BYTES) };
    ChunkBlock::unpack(s)
}

unsafe fn store_block(blk: *mut u8, b: &ChunkBlock) {
    let s = unsafe { core::slice::from_raw_parts_mut(blk, ChunkBlock::BYTES) };
    let _ = b.pack(s);
}

#[unsafe(no_mangle)]
pub extern "C" fn _start(blk: *mut u8) -> ! {
    // SAFETY: loader passes the block in the first argument register, mapped
    // read-write; the heap it names is mapped read-write too.
    unsafe {
        BLOCK = blk;
        let Some(mut b) = load_block(blk) else {
            write_status(blk, ChunkStatus::Failed, 0);
            exit();
        };
        HEAP_BASE = b.heap_ptr as usize;
        HEAP_LEN = b.heap_len as usize;

        let cmd = b.cmd().unwrap_or(ChunkCmd::Init);
        match cmd {
            ChunkCmd::Init | ChunkCmd::Cancel => {
                // Drop decoder state **then** reset the arena. Continue must
                // not take this path — a reused space keeps `.bss`.
                session::drop_session();
                CURSOR = 0;
                OOM = false;
            }
            ChunkCmd::Continue | ChunkCmd::Seek | ChunkCmd::Event => {}
        }

        let status = match cmd {
            ChunkCmd::Cancel => ChunkStatus::Cancelled,
            ChunkCmd::Init => {
                if b.input_len == 0 {
                    // Tiny self-check: empty input is a failed init, not Done.
                    ChunkStatus::Failed
                } else {
                    let input = core::slice::from_raw_parts(b.input_ptr as *const u8, b.input_len as usize);
                    LAST_HEAP_WANT = session::heap_guess(input.len());
                    b.heap_want = LAST_HEAP_WANT;
                    session::init(input, &mut b)
                }
            }
            ChunkCmd::Continue => session::cont(b.max_units, &mut b),
            ChunkCmd::Seek => session::seek(b.max_units, &mut b),
            ChunkCmd::Event => ChunkStatus::Failed,
        };
        LAST_HEAP_WANT = b.heap_want;
        b.cmd_or_status = status as u64;
        store_block(blk, &b);
    }
    exit()
}
