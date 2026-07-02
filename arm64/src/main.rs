//! Chitti aarch64 boot + native NEON inference benchmark (Phase 7: fast
//! inference on Apple Silicon).
//!
//! Booted directly by `qemu-system-aarch64 -M virt -kernel` (no bootloader):
//! QEMU loads this ELF and jumps to `_start` in EL1 with the MMU off. Under
//! `-accel hvf` the guest runs *natively* on the M-series CPU via
//! Hypervisor.framework -- no cross-arch translation, and NEON runs on the
//! real vector units. This is the whole point: escape the x86-on-arm TCG
//! emulation the x86 kernel is stuck with.
//!
//! `kmain` runs the same Q8_0 matvec that dominates a token's compute, in
//! NEON, times it with the ARM generic timer, checks it against a scalar
//! reference, and reports throughput -- a concrete measure of how fast Chitti
//! inference *could* run once the full kernel is ported to this arch.
#![no_std]
#![no_main]

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;
use core::alloc::{GlobalAlloc, Layout};
use core::arch::{aarch64::*, asm, global_asm};
use core::panic::PanicInfo;
use core::sync::atomic::{AtomicUsize, Ordering};

// Entry stub: set the stack, enable FP/SIMD (NEON) at EL1, zero .bss, jump to
// Rust. Kept in its own section so the linker places it at the load address.
global_asm!(
    r#"
.section .text.boot
.global _start
_start:
    adrp x0, __stack_top
    add  x0, x0, :lo12:__stack_top
    mov  sp, x0

    // Enable FP/SIMD access at EL1: CPACR_EL1.FPEN = 0b11 (bits [21:20]).
    mrs  x0, cpacr_el1
    orr  x0, x0, #(3 << 20)
    msr  cpacr_el1, x0
    isb

    // Zero .bss.
    adrp x0, __bss_start
    add  x0, x0, :lo12:__bss_start
    adrp x1, __bss_end
    add  x1, x1, :lo12:__bss_end
1:  cmp  x0, x1
    b.hs 2f
    str  xzr, [x0], #8
    b    1b
2:  bl   kmain
3:  wfi
    b    3b
"#
);

// --- console (PL011 UART on the QEMU `virt` machine) ---------------------

const UART0_DR: *mut u32 = 0x0900_0000 as *mut u32;

fn uart_putc(byte: u8) {
    // SAFETY: `virt` PL011 data register; writing a byte transmits it (MMIO).
    unsafe { core::ptr::write_volatile(UART0_DR, byte as u32) };
}
fn puts(s: &str) {
    for b in s.bytes() {
        uart_putc(b);
    }
}
/// Print an `i64` in decimal (with sign).
fn puti(n: i64) {
    if n < 0 {
        uart_putc(b'-');
        putu((-n) as u64);
    } else {
        putu(n as u64);
    }
}
/// Print a `u64` in decimal.
fn putu(mut n: u64) {
    let mut buf = [0u8; 20];
    let mut i = buf.len();
    if n == 0 {
        uart_putc(b'0');
        return;
    }
    while n > 0 {
        i -= 1;
        buf[i] = b'0' + (n % 10) as u8;
        n /= 10;
    }
    for &b in &buf[i..] {
        uart_putc(b);
    }
}

// --- bump allocator over a static arena ----------------------------------

#[repr(align(64))]
struct Arena(#[allow(dead_code)] [u8; ARENA_SIZE]);
const ARENA_SIZE: usize = 64 * 1024 * 1024;
static mut ARENA: Arena = Arena([0; ARENA_SIZE]);
static ARENA_NEXT: AtomicUsize = AtomicUsize::new(0);

struct Bump;
unsafe impl GlobalAlloc for Bump {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let base = core::ptr::addr_of_mut!(ARENA) as usize;
        let cur = ARENA_NEXT.load(Ordering::Relaxed);
        let aligned = (base + cur + layout.align() - 1) & !(layout.align() - 1);
        let end = aligned - base + layout.size();
        if end > ARENA_SIZE {
            return core::ptr::null_mut();
        }
        ARENA_NEXT.store(end, Ordering::Relaxed);
        aligned as *mut u8
    }
    unsafe fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {}
}

#[global_allocator]
static ALLOC: Bump = Bump;

// --- MMU: identity map with normal cacheable RAM ------------------------
//
// QEMU jumps in with the MMU off, so all memory is device-typed and uncached:
// NEON vector loads/stores are unreliable there and everything is slow. This
// sets up a minimal identity map (1 GiB blocks over the low 4 GiB) marking RAM
// as Normal write-back cacheable and the low 1 GiB (MMIO: UART/GIC) as Device,
// then turns on the MMU + I/D caches. After this, NEON is correct and fast.

#[repr(align(4096))]
struct Table(#[allow(dead_code)] [u64; 512]);
static mut L1: Table = Table([0; 512]);

fn mmu_init() {
    // SAFETY: single-core boot; we build a valid identity map and program the
    // standard EL1 translation system registers, then enable the MMU. VA==PA,
    // so the stack/code/UART addresses stay valid across the switch.
    unsafe {
        let l1 = core::ptr::addr_of_mut!(L1) as *mut u64;
        // 1 GiB block descriptors, identity mapping [0, 4 GiB).
        for i in 0..4u64 {
            let pa = i << 30;
            let attr_idx = if i == 0 { 1u64 } else { 0u64 }; // 0: Device MMIO, else Normal
            let sh = if i == 0 { 0u64 } else { 0b11u64 }; // inner-shareable for Normal
            let desc = pa | (attr_idx << 2) | (sh << 8) | (1 << 10) | 0b01; // AF=1, block, valid
            *l1.add(i as usize) = desc;
        }
        // MAIR: attr0 = Normal write-back (0xFF), attr1 = Device nGnRnE (0x00).
        let mair: u64 = 0xFF;
        // TCR: T0SZ=25 (39-bit VA), 4 KiB granule, WB cacheable page walks,
        // inner-shareable, TTBR1 disabled, 40-bit PA.
        let tcr: u64 = 25 | (1 << 8) | (1 << 10) | (3 << 12) | (1 << 23) | (2u64 << 32);
        asm!("msr mair_el1, {}", in(reg) mair, options(nostack));
        asm!("msr tcr_el1, {}", in(reg) tcr, options(nostack));
        asm!("msr ttbr0_el1, {}", in(reg) l1 as u64, options(nostack));
        asm!("dsb ish", "tlbi vmalle1", "dsb ish", "isb", options(nostack));
        let mut sctlr: u64;
        asm!("mrs {}, sctlr_el1", out(reg) sctlr, options(nostack));
        sctlr |= (1 << 0) | (1 << 2) | (1 << 12); // M (MMU), C (data cache), I (instr cache)
        asm!("msr sctlr_el1, {}", in(reg) sctlr, options(nostack));
        asm!("isb", options(nostack));
    }
}

// --- ARM generic timer (for wall-clock measurement) ----------------------

fn cntpct() -> u64 {
    let v: u64;
    // SAFETY: reading the physical counter is always valid at EL1.
    unsafe { asm!("mrs {}, cntpct_el0", out(reg) v, options(nomem, nostack)) };
    v
}
fn cntfrq() -> u64 {
    let v: u64;
    // SAFETY: reading the counter frequency register is always valid.
    unsafe { asm!("mrs {}, cntfrq_el0", out(reg) v, options(nomem, nostack)) };
    v
}

// --- Q8_0 numeric kernels (NEON + scalar reference) ----------------------

const QK: usize = 32;
const Q8_0_BLOCK_BYTES: usize = 2 + QK; // f16 scale + 32 int8

/// `2^e` as an `f32`, built from the IEEE-754 exponent field (no `powi`,
/// which lives in `std`, not `core`). Valid for `e` in the normal range.
fn pow2(e: i32) -> f32 {
    f32::from_bits(((e + 127) as u32) << 23)
}

/// IEEE half -> single. Enough for GGUF Q8_0 block scales (normal values).
fn f16_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) & 1;
    let exp = (bits >> 10) & 0x1f;
    let mant = bits & 0x3ff;
    let s = if sign == 1 { -1.0f32 } else { 1.0f32 };
    if exp == 0 {
        s * (mant as f32) * pow2(-24)
    } else if exp == 0x1f {
        if mant == 0 {
            s * f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        s * (1.0 + (mant as f32) / 1024.0) * pow2(exp as i32 - 15)
    }
}

/// Scalar reference: `y[r] = sum_c W[r,c] * x[c]`, Q8_0 weights.
fn matvec_q8_0_scalar(w: &[u8], x: &[f32], y: &mut [f32], n_rows: usize, n_cols: usize) {
    let blocks = n_cols / QK;
    let row_bytes = blocks * Q8_0_BLOCK_BYTES;
    for r in 0..n_rows {
        let row = &w[r * row_bytes..];
        let mut acc = 0.0f32;
        for b in 0..blocks {
            let base = b * Q8_0_BLOCK_BYTES;
            let d = f16_to_f32(u16::from_le_bytes([row[base], row[base + 1]]));
            for i in 0..QK {
                let q = row[base + 2 + i] as i8 as f32;
                acc += q * d * x[b * QK + i];
            }
        }
        y[r] = acc;
    }
}

/// Fused NEON Q8_0 matvec: per block, widen 8 int8 at a time to f32, scale by
/// the block's f16 `d`, and FMA against `x` into a 4-lane accumulator.
///
/// # Safety
/// `w`/`x`/`y` sized as in `matvec_q8_0_scalar`; runs on any aarch64 core
/// (NEON is baseline). Rows are independent, so this splits across cores.
unsafe fn matvec_q8_0_neon(w: *const u8, x: *const f32, y: *mut f32, n_rows: usize, n_cols: usize) {
    let blocks = n_cols / QK;
    let row_bytes = blocks * Q8_0_BLOCK_BYTES;
    unsafe {
        for r in 0..n_rows {
            let row = w.add(r * row_bytes);
            // Two independent accumulators break the FMA dependency chain so
            // the core's out-of-order FMA units stay busy.
            let mut acc0 = vdupq_n_f32(0.0);
            let mut acc1 = vdupq_n_f32(0.0);
            for b in 0..blocks {
                let base = b * Q8_0_BLOCK_BYTES;
                let d = f16_to_f32(u16::from_le_bytes([*row.add(base), *row.add(base + 1)]));
                let dv = vdupq_n_f32(d);
                let q = row.add(base + 2) as *const i8;
                let xp = x.add(b * QK);
                let mut g = 0;
                while g < QK {
                    let q8 = vld1_s8(q.add(g)); // 8 x i8
                    let q16 = vmovl_s8(q8); // 8 x i16
                    let q32lo = vmovl_s16(vget_low_s16(q16)); // 4 x i32
                    let q32hi = vmovl_s16(vget_high_s16(q16)); // 4 x i32
                    let qflo = vmulq_f32(vcvtq_f32_s32(q32lo), dv);
                    let qfhi = vmulq_f32(vcvtq_f32_s32(q32hi), dv);
                    acc0 = vfmaq_f32(acc0, qflo, vld1q_f32(xp.add(g)));
                    acc1 = vfmaq_f32(acc1, qfhi, vld1q_f32(xp.add(g + 4)));
                    g += 8;
                }
            }
            *y.add(r) = vaddvq_f32(vaddq_f32(acc0, acc1));
        }
    }
}

// --- benchmark -----------------------------------------------------------

/// Rows/cols of the benchmark matvec (cols = model hidden dim).
const ROWS: usize = 4096;
const COLS: usize = 1024;
/// Iterations, so the timed region is long enough to measure accurately.
const ITERS: u64 = 200;
/// Rough Q8_0 MACs per decoded token for the Qwen3.5-0.8B model (24 layers of
/// attention+FFN matvecs plus the vocab output projection). Used only to turn
/// the measured MAC/s into an intuitive tokens/sec estimate.
const MACS_PER_TOKEN: u64 = 750_000_000;

#[no_mangle]
pub extern "C" fn kmain() -> ! {
    puts("Chitti aarch64: boot ok -- running NATIVELY on Apple Silicon via HVF\n");
    mmu_init();
    puts("Chitti aarch64: MMU on (identity map, RAM = normal cacheable)\n");
    puts("Chitti aarch64: NEON Q8_0 matvec benchmark (the kernel that dominates a token)\n\n");

    let blocks = COLS / QK;
    let row_bytes = blocks * Q8_0_BLOCK_BYTES;

    // Build a deterministic Q8_0 weight matrix + f32 activation. Scale bits
    // 0x2c00 = 0.0625 (a valid, finite f16) keep the reference finite.
    let mut w: Vec<u8> = vec![0u8; ROWS * row_bytes];
    let mut lcg: u32 = 0x12345678;
    for r in 0..ROWS {
        for b in 0..blocks {
            let base = r * row_bytes + b * Q8_0_BLOCK_BYTES;
            w[base] = 0x00;
            w[base + 1] = 0x2c; // f16 0.0625
            for i in 0..QK {
                lcg = lcg.wrapping_mul(1664525).wrapping_add(1013904223);
                w[base + 2 + i] = (lcg >> 24) as u8;
            }
        }
    }
    let x: Vec<f32> = (0..COLS).map(|c| ((c % 13) as f32 - 6.0) * 0.1).collect();
    let mut y: Vec<f32> = vec![0.0; ROWS];
    let mut y_ref: Vec<f32> = vec![0.0; ROWS];

    // Correctness: NEON must match the scalar reference.
    matvec_q8_0_scalar(&w, &x, &mut y_ref, ROWS, COLS);
    // SAFETY: slices are correctly sized above.
    unsafe { matvec_q8_0_neon(w.as_ptr(), x.as_ptr(), y.as_mut_ptr(), ROWS, COLS) };
    let mut max_rel = 0.0f32;
    for r in 0..ROWS {
        let denom = if y_ref[r].abs() > 1e-3 { y_ref[r].abs() } else { 1.0 };
        let rel = (y[r] - y_ref[r]).abs() / denom;
        if rel > max_rel {
            max_rel = rel;
        }
    }
    // f32 tolerance: the NEON lane-parallel sum and the scalar sequential sum
    // round differently (float add isn't associative), so a small relative
    // gap is expected and correct -- not a bug.
    let correct = max_rel < 1e-2;
    puts("NEON matches scalar reference: ");
    puts(if correct { "YES" } else { "NO" });
    puts("\n");
    // Diagnostics: MMU state + first-row values + worst relative error.
    let sctlr: u64;
    unsafe { asm!("mrs {}, sctlr_el1", out(reg) sctlr, options(nostack)) };
    puts("  SCTLR.M(mmu)="); putu(sctlr & 1);
    puts(" .C(dcache)="); putu((sctlr >> 2) & 1);
    puts("\n  y_ref[0]*1000="); puti(( y_ref[0] * 1000.0) as i64);
    puts(" y_neon[0]*1000="); puti((y[0] * 1000.0) as i64);
    puts("\n  max_rel*1e6="); putu((max_rel * 1_000_000.0) as u64);
    puts("\n\n");

    // Timed run.
    let start = cntpct();
    for _ in 0..ITERS {
        // SAFETY: as above.
        unsafe { matvec_q8_0_neon(w.as_ptr(), x.as_ptr(), y.as_mut_ptr(), ROWS, COLS) };
    }
    let end = cntpct();
    let freq = cntfrq();
    let ticks = end - start;

    let total_macs = ITERS * (ROWS as u64) * (COLS as u64);
    // MAC/s = total_macs * freq / ticks  (u128 to avoid overflow).
    let mac_per_s = ((total_macs as u128) * (freq as u128) / (ticks.max(1) as u128)) as u64;
    let elapsed_us = (ticks as u128 * 1_000_000 / freq.max(1) as u128) as u64;
    let tok_per_s = mac_per_s / MACS_PER_TOKEN;

    puts("matvec: ");
    putu(ROWS as u64);
    puts(" x ");
    putu(COLS as u64);
    puts(" Q8_0, x");
    putu(ITERS);
    puts(" iters in ");
    putu(elapsed_us);
    puts(" us\n");
    puts("throughput: ");
    putu(mac_per_s / 1_000_000);
    puts(" MMAC/s (");
    putu(mac_per_s / 1_000_000_000);
    puts(" GMAC/s)\n");
    puts("estimated Qwen3.5-0.8B decode: ~");
    putu(tok_per_s);
    puts(" tok/s (native NEON)\n\n");
    puts("Chitti aarch64: done. This is the compute path the full port unlocks.\n");

    loop {
        // SAFETY: idle the core.
        unsafe { asm!("wfi", options(nomem, nostack)) };
    }
}

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    puts("Chitti aarch64: PANIC\n");
    loop {
        unsafe { asm!("wfi", options(nomem, nostack)) };
    }
}
