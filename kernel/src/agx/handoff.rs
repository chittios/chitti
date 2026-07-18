//! Apple AGX **GFXHandoff** — the shared-memory PPL (Page-Protection-Layer)
//! handshake + page-table-flush coordination between the CPU and the gfx-asc
//! firmware. Ported from m1n1's `proxyclient/m1n1/fw/agx/handoff.py`
//! (`GFXHandoffStruct` + `GFXHandoff.initialize`/`lock`).
//!
//! **Why it matters (Milestone 3 unblock):** the firmware's memory manager
//! blocks during boot until the CPU writes `MAGIC_AP` (`PPL_MAGIC`) into the
//! handoff region; it then initialises its MMU/contexts, writes `MAGIC_FW` back,
//! and only *then* proceeds toward power-ON. Without this the coprocessor stalls
//! silently right after the RTKit crashlog buffer request — exactly what we saw.
//!
//! The handoff region is DRAM shared with the coprocessor, so every access is
//! bracketed with cache maintenance (`dc cvac`/`dc civac` + `dsb`) — the
//! proxyclient's "this *absolutely* needs barriers everywhere". aarch64 identity
//! map ⇒ VA == PA.

use core::arch::asm;
use core::ptr::{read_volatile, write_volatile};

/// The PPL init magic both sides write to their MAGIC field (handoff.py).
pub const PPL_MAGIC: u64 = 0x4b1d_0000_0000_0002;

// --- GFXHandoffStruct field offsets (handoff.py) -------------------------
const MAGIC_AP: u64 = 0x0; // u64 — CPU writes PPL_MAGIC
const MAGIC_FW: u64 = 0x8; // u64 — firmware writes PPL_MAGIC back
const LOCK_AP: u64 = 0x10; // u8  — CPU's Dekker lock flag
const LOCK_FW: u64 = 0x11; // u8  — firmware's Dekker lock flag
const TURN: u64 = 0x14; // u32 — Dekker turn
const FLUSH_BASE: u64 = 0x20; // FLUSH_STATE[i] = 0x20 + i*0x18; ADDR +8; SIZE +0x10
const FLUSH_STRIDE: u64 = 0x18;
const FLUSH_COUNT: u64 = 0x41; // 65 contexts
const UNK2: u64 = 0x638; // u8
const UNK3: u64 = 0x640; // u64

// --- cache-maintained shared-memory accessors ----------------------------
// Writes: publish to the coprocessor (`dc cvac` + `dsb`). Reads of
// firmware-written fields: invalidate first so we observe its store
// (`dc civac` + `dsb`, then read).

#[inline]
fn clean(pa: u64) {
    // SAFETY: clean one cache line of mapped Normal DRAM to PoC.
    unsafe { asm!("dc cvac, {p}", "dsb sy", p = in(reg) pa, options(nostack, preserves_flags)) };
}
#[inline]
fn inval(pa: u64) {
    // SAFETY: clean+invalidate one cache line of mapped Normal DRAM to PoC.
    unsafe { asm!("dc civac, {p}", "dsb sy", p = in(reg) pa, options(nostack, preserves_flags)) };
}

#[inline]
fn w64(base: u64, off: u64, v: u64) {
    let a = base + off;
    // SAFETY: single aligned 64-bit write of mapped shared DRAM.
    unsafe { write_volatile(a as *mut u64, v) };
    clean(a);
}
#[inline]
fn r64(base: u64, off: u64) -> u64 {
    let a = base + off;
    inval(a);
    // SAFETY: single aligned 64-bit read of mapped shared DRAM.
    unsafe { read_volatile(a as *const u64) }
}
#[inline]
fn w8(base: u64, off: u64, v: u8) {
    let a = base + off;
    // SAFETY: single 8-bit write of mapped shared DRAM.
    unsafe { write_volatile(a as *mut u8, v) };
    clean(a);
}
#[inline]
fn r8(base: u64, off: u64) -> u8 {
    let a = base + off;
    inval(a);
    // SAFETY: single 8-bit read of mapped shared DRAM.
    unsafe { read_volatile(a as *const u8) }
}
#[inline]
fn w32(base: u64, off: u64, v: u32) {
    let a = base + off;
    // SAFETY: single 32-bit write of mapped shared DRAM.
    unsafe { write_volatile(a as *mut u32, v) };
    clean(a);
}
#[inline]
fn r32(base: u64, off: u64) -> u32 {
    let a = base + off;
    inval(a);
    // SAFETY: single 32-bit read of mapped shared DRAM.
    unsafe { read_volatile(a as *const u32) }
}

/// Take the Dekker's-algorithm lock (handoff.py `GFXHandoff.lock`), bounded +
/// pumping. `pump` returns true to abort (Ctrl+C). Returns false on timeout.
fn lock(base: u64, timeout_ms: u64, pump: &mut dyn FnMut() -> bool) -> bool {
    let deadline = crate::arch::now_ms() + timeout_ms;
    w8(base, LOCK_AP, 1);
    while r8(base, LOCK_FW) != 0 {
        if r32(base, TURN) != 0 {
            w8(base, LOCK_AP, 0);
            while r32(base, TURN) != 0 {
                if crate::arch::now_ms() >= deadline || pump() {
                    return false;
                }
                core::hint::spin_loop();
            }
            w8(base, LOCK_AP, 1);
        }
        if crate::arch::now_ms() >= deadline || pump() {
            return false;
        }
        core::hint::spin_loop();
    }
    true
}

/// Release the Dekker lock (`TURN = 1; LOCK_AP = 0`).
fn unlock(base: u64) {
    w32(base, TURN, 1);
    w8(base, LOCK_AP, 0);
}

/// Run `f` while holding the handoff lock (bounded). Returns false if the lock
/// could not be taken.
pub fn with_lock(base: u64, timeout_ms: u64, pump: &mut dyn FnMut() -> bool, f: impl FnOnce()) -> bool {
    if !lock(base, timeout_ms, pump) {
        return false;
    }
    f();
    unlock(base);
    true
}

/// The PPL init handshake (`GFXHandoff.initialize`): publish `MAGIC_AP`, take
/// the lock, wait for the firmware's `MAGIC_FW`, then zero the flush-state
/// array. This is what unblocks the firmware's MMU init → power-ON. Bounded +
/// pumping; returns false on timeout/abort (`base` is the `handoff` carveout).
pub fn initialize(base: u64, timeout_ms: u64, pump: &mut dyn FnMut() -> bool) -> bool {
    w64(base, MAGIC_AP, PPL_MAGIC);
    w8(base, UNK2, 0xff);
    w64(base, UNK3, 0);

    if !lock(base, timeout_ms, pump) {
        crate::ktrace::log("agx", "handoff: could not take lock for PPL init");
        return false;
    }
    let deadline = crate::arch::now_ms() + timeout_ms;
    let mut ok = true;
    crate::ktrace::log("agx", "handoff: MAGIC_AP written; waiting for FW PPL init (MAGIC_FW)…");
    while r64(base, MAGIC_FW) != PPL_MAGIC {
        if crate::arch::now_ms() >= deadline || pump() {
            ok = false;
            break;
        }
        core::hint::spin_loop();
    }
    unlock(base);
    if !ok {
        crate::ktrace::log_fmt(format_args!("agx: handoff: FW did not ack PPL init (MAGIC_FW={:#018x})", r64(base, MAGIC_FW)));
        return false;
    }
    crate::ktrace::log("agx", "handoff: FW acked PPL init (MAGIC_FW=PPL_MAGIC)");

    // Zero every context's flush-state triple.
    for i in 0..FLUSH_COUNT {
        let o = FLUSH_BASE + i * FLUSH_STRIDE;
        w64(base, o, 0); // FLUSH_STATE
        w64(base, o + 8, 0); // FLUSH_ADDR
        w64(base, o + 0x10, 0); // FLUSH_SIZE
    }
    true
}
