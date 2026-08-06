//! **PS/2 keyboard over the ARM PL050 PrimeCell (KMI)** — the aarch64 analogue
//! of the x86 i8042 PS/2 keyboard (`arch::x86_64::keyboard`). The PL050
//! Keyboard/Mouse Interface is the standard PS/2 controller on ARM development
//! boards (Versatile / RealView / Versatile Express). QEMU's `virt` machine has
//! no PL050, so this is a **safe no-op there** (the PrimeCell-ID probe fails);
//! it activates only where a real PL050 is present, giving aarch64 the same
//! "PS/2 keyboard drives the console" capability x86 has.
//!
//! Polled (like the aarch64 xHCI / virtio-input paths, no IRQ): `poll_key`
//! drains the receive register and turns **scan-code set 2** (what a PS/2
//! keyboard emits natively — the i8042 on a PC translates to set 1, but the
//! PL050 passes set 2 through) into HID usages, which it hands to
//! [`crate::keymap`].
//!
//! It no longer decodes characters. Layout, dead keys, Compose and the arrow→CSI
//! table are shared with the other three transports, which is also what makes the
//! set-2 cross-table testable at all: this module is `cfg`'d out of the test build
//! (`arch::aarch64` only exists on aarch64, and the unit suite is x86), so
//! anything left here could never carry a `#[test_case]`.

use crate::keymap::{self, KeyEvent, Mods, Source};
use crate::mm::Locked;
use core::ptr::{read_volatile, write_volatile};
use core::sync::atomic::{AtomicBool, AtomicU8, Ordering};

// PL050 register offsets (from the KMI base).
const KMICR: u64 = 0x00; // control
const KMISTAT: u64 = 0x04; // status
const KMIDATA: u64 = 0x08; // received / transmit byte

const KMICR_EN: u32 = 1 << 2; // KmiEn: enable the interface
const KMISTAT_RXFULL: u32 = 1 << 4; // a received byte is waiting

// PrimeCell identification (offsets 0xFE0..0xFFC, low byte of each register).
// PL050: peripheral part number 0x050; PrimeCell id 0xB105_F00D.
const PL050_PART: u32 = 0x050;
const PRIMECELL_ID: u32 = 0xB105_F00D;

// Candidate KMI0 (keyboard) base addresses across common ARM machines. Each is
// probed by PrimeCell id, so a wrong guess (or QEMU `virt`, which has none) is
// simply skipped. All sit in the low 1 GiB the identity MMU maps as Device.
const CANDIDATE_BASES: [u64; 2] = [
    0x1c06_0000, // Versatile Express KMI0
    0x1000_6000, // Versatile / RealView KMI0
];

/// The located PL050 base, or 0 if none present. Set once by [`init`].
static BASE: Locked<u64> = Locked::new(0);

/// The PL050 base the keyboard claimed (0 if none), so the mouse KMI driver
/// skips it when scanning.
pub fn keyboard_base() -> u64 {
    BASE.with(|b| *b)
}
/// Live modifier bits as a [`Mods`] bitset. Set 2 delivers *edges* (a break code
/// is `0xF0 <code>`), so converting edge→level stays here and the shared layer
/// takes a level.
static MODS: AtomicU8 = AtomicU8::new(0);
/// Set 2 sends a byte's *break* code as `0xF0 <code>`; this holds across polls.
static BREAK_NEXT: AtomicBool = AtomicBool::new(false);
/// Extended keys are prefixed `0xE0` (right Ctrl/Alt, the GUI keys, the nav block).
static EXT_NEXT: AtomicBool = AtomicBool::new(false);

unsafe fn r32(a: u64) -> u32 {
    unsafe { read_volatile(a as *const u32) }
}
unsafe fn w32(a: u64, v: u32) {
    unsafe { write_volatile(a as *mut u32, v) };
}

/// Read a PrimeCell id block (`id_lo` at 0xFE0 for the peripheral part, or 0xFF0
/// for the cell id): four registers, low byte each, assembled little-endian.
unsafe fn id_block(base: u64, first: u64) -> u32 {
    unsafe {
        (r32(base + first) & 0xff)
            | ((r32(base + first + 4) & 0xff) << 8)
            | ((r32(base + first + 8) & 0xff) << 16)
            | ((r32(base + first + 12) & 0xff) << 24)
    }
}

/// Probe the candidate bases for a real PL050 and, if found, enable it. Safe to
/// call on any platform: reads are to Device-mapped MMIO and a mismatch just
/// skips. Returns true if a PL050 keyboard controller was found + enabled.
pub fn init() -> bool {
    for &base in &CANDIDATE_BASES {
        // SAFETY: `base` is in the identity-mapped low-1 GiB Device region; the
        // PrimeCell id registers are read-only. A missing device reads as 0.
        unsafe {
            let cell = id_block(base, 0xFF0);
            let part = id_block(base, 0xFE0) & 0xfff;
            if cell == PRIMECELL_ID && part == PL050_PART {
                w32(base + KMICR, KMICR_EN); // enable the interface (polled, no IRQ)
                BASE.with(|b| *b = base);
                crate::ktrace::log_fmt(format_args!("pl050: PS/2 keyboard at {:#x} (scan-code set 2)", base));
                return true;
            }
        }
    }
    false
}

fn set_mod(bit: u8, on: bool) {
    let cur = MODS.load(Ordering::Relaxed);
    MODS.store(if on { cur | bit } else { cur & !bit }, Ordering::Relaxed);
}

/// Poll the PL050 and return the next translated input byte.
///
/// Consumes received bytes until one produces output or the receive register
/// drains, tracking the `0xF0` (break) and `0xE0` (extended) prefixes. Everything
/// past "which physical key was that" is [`crate::keymap`]'s job. Non-blocking;
/// `None` if no PL050 or nothing typed.
pub fn poll_key() -> Option<u8> {
    // Anything the shared layer already translated comes out first.
    if let Some(b) = keymap::next_byte() {
        return Some(b);
    }
    let base = BASE.with(|b| *b);
    if base == 0 {
        return None;
    }
    // Bounded so a stuck controller can't spin the caller forever.
    for _ in 0..16 {
        // SAFETY: `base` is the probed, enabled PL050 register block.
        let sc = unsafe {
            if r32(base + KMISTAT) & KMISTAT_RXFULL == 0 {
                return None;
            }
            (r32(base + KMIDATA) & 0xff) as u8
        };
        match sc {
            0xF0 => BREAK_NEXT.store(true, Ordering::Relaxed),
            0xE0 => EXT_NEXT.store(true, Ordering::Relaxed),
            _ => {
                let breaking = BREAK_NEXT.swap(false, Ordering::Relaxed);
                let extended = EXT_NEXT.swap(false, Ordering::Relaxed);
                let pressed = !breaking;
                if let Some(usage) = keymap::usage_from_set2(sc, extended) {
                    if let Some(bit) = keymap::modifier_bit(usage) {
                        set_mod(bit, pressed);
                    } else {
                        let mods = Mods(MODS.load(Ordering::Relaxed));
                        keymap::feed_event(KeyEvent {
                            usage,
                            mods,
                            pressed,
                            src: Source::Ps2Set2,
                        });
                        // The event may have produced bytes; hand back the first.
                        if let Some(b) = keymap::next_byte() {
                            return Some(b);
                        }
                    }
                }
            }
        }
    }
    None
}
