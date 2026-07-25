//! **PS/2 keyboard over the ARM PL050 PrimeCell (KMI)** — the aarch64 analogue
//! of the x86 i8042 PS/2 keyboard (`arch::x86_64::keyboard`). The PL050
//! Keyboard/Mouse Interface is the standard PS/2 controller on ARM development
//! boards (Versatile / RealView / Versatile Express). QEMU's `virt` machine has
//! no PL050, so this is a **safe no-op there** (the PrimeCell-ID probe fails);
//! it activates only where a real PL050 is present, giving aarch64 the same
//! "PS/2 keyboard drives the console" capability x86 has.
//!
//! Polled (like the aarch64 xHCI / virtio-input paths, no IRQ): `poll_key`
//! drains the receive register, decodes **scan-code set 2** (what a PS/2
//! keyboard emits natively — the i8042 on a PC translates to set 1, but the
//! PL050 passes set 2 through), tracks shift/caps, and returns ASCII.

use crate::mm::Locked;
use core::ptr::{read_volatile, write_volatile};
use core::sync::atomic::{AtomicBool, Ordering};

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
static SHIFT_DOWN: AtomicBool = AtomicBool::new(false);
static CTRL_DOWN: AtomicBool = AtomicBool::new(false);
/// Left/Right GUI (⌘ / Super) — set-2 E0 0x1f / E0 0x27.
static GUI_DOWN: AtomicBool = AtomicBool::new(false);
static CAPS_ON: AtomicBool = AtomicBool::new(false);
/// Set 2 sends a byte's *break* code as `0xF0 <code>`; this holds across polls.
static BREAK_NEXT: AtomicBool = AtomicBool::new(false);
/// Extended keys are prefixed `0xE0`; arrows map to ANSI sequences, others are ignored.
static EXT_NEXT: AtomicBool = AtomicBool::new(false);
/// Bytes still owed to the caller (arrow keys expand to a 3-byte ANSI escape,
/// but `poll_key` returns one byte per call).
static PENDING: Locked<alloc::vec::Vec<u8>> = Locked::new(alloc::vec::Vec::new());

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

/// Decode a scan-code-set-2 make code to ASCII, applying shift/caps. `0` means
/// no printable character. Mirrors the x86 set-1 decoder's shift/caps rules.
fn decode(sc: u8) -> Option<u8> {
    let shift = SHIFT_DOWN.load(Ordering::Relaxed);
    // Base (unshifted / shifted) character for this set-2 make code.
    let (lo, hi): (u8, u8) = match sc {
        0x1C => (b'a', b'A'), 0x32 => (b'b', b'B'), 0x21 => (b'c', b'C'), 0x23 => (b'd', b'D'),
        0x24 => (b'e', b'E'), 0x2B => (b'f', b'F'), 0x34 => (b'g', b'G'), 0x33 => (b'h', b'H'),
        0x43 => (b'i', b'I'), 0x3B => (b'j', b'J'), 0x42 => (b'k', b'K'), 0x4B => (b'l', b'L'),
        0x3A => (b'm', b'M'), 0x31 => (b'n', b'N'), 0x44 => (b'o', b'O'), 0x4D => (b'p', b'P'),
        0x15 => (b'q', b'Q'), 0x2D => (b'r', b'R'), 0x1B => (b's', b'S'), 0x2C => (b't', b'T'),
        0x3C => (b'u', b'U'), 0x2A => (b'v', b'V'), 0x1D => (b'w', b'W'), 0x22 => (b'x', b'X'),
        0x35 => (b'y', b'Y'), 0x1A => (b'z', b'Z'),
        0x16 => (b'1', b'!'), 0x1E => (b'2', b'@'), 0x26 => (b'3', b'#'), 0x25 => (b'4', b'$'),
        0x2E => (b'5', b'%'), 0x36 => (b'6', b'^'), 0x3D => (b'7', b'&'), 0x3E => (b'8', b'*'),
        0x46 => (b'9', b'('), 0x45 => (b'0', b')'),
        0x0E => (b'`', b'~'), 0x4E => (b'-', b'_'), 0x55 => (b'=', b'+'),
        0x54 => (b'[', b'{'), 0x5B => (b']', b'}'), 0x5D => (b'\\', b'|'),
        0x4C => (b';', b':'), 0x52 => (b'\'', b'"'),
        0x41 => (b',', b'<'), 0x49 => (b'.', b'>'), 0x4A => (b'/', b'?'),
        0x29 => (b' ', b' '), 0x5A => (b'\n', b'\n'), 0x66 => (0x08, 0x08), 0x0D => (b'\t', b'\t'),
        _ => return None,
    };
    let base = if shift { hi } else { lo };
    // Caps-lock flips letter case (caps + shift = lowercase, as on a PC).
    let ch = if CAPS_ON.load(Ordering::Relaxed) && base.is_ascii_alphabetic() {
        if base.is_ascii_lowercase() { base - 32 } else { base + 32 }
    } else {
        base
    };
    // Ctrl+letter → control code (Ctrl+C = 3 stops generation, etc.).
    if CTRL_DOWN.load(Ordering::Relaxed) && ch.is_ascii_alphabetic() {
        return Some(ch.to_ascii_uppercase() & 0x1f);
    }
    Some(ch)
}

const SC_LSHIFT: u8 = 0x12;
const SC_RSHIFT: u8 = 0x59;
const SC_LCTRL: u8 = 0x14;
const SC_CAPS: u8 = 0x58;

/// Poll the PL050 for the next typed character, decoding set-2 codes + tracking
/// shift/caps across the `0xF0` (break) and `0xE0` (extended) prefixes. Consumes
/// received bytes until it produces a character or the receive register drains.
/// Non-blocking; `None` if no PL050 or nothing typed.
pub fn poll_key() -> Option<u8> {
    // Drain any queued escape-sequence bytes first (arrow-key expansion).
    if let Some(b) = PENDING.with(|p| if p.is_empty() { None } else { Some(p.remove(0)) }) {
        return Some(b);
    }
    let base = BASE.with(|b| *b);
    if base == 0 {
        return None;
    }
    // Bounded so a stuck controller can't spin the caller forever.
    for _ in 0..16 {
        // SAFETY: `base` is the probed, enabled PL050 register block.
        let (has, sc) = unsafe {
            if r32(base + KMISTAT) & KMISTAT_RXFULL == 0 {
                return None;
            }
            (true, (r32(base + KMIDATA) & 0xff) as u8)
        };
        if !has {
            return None;
        }
        match sc {
            0xF0 => {
                BREAK_NEXT.store(true, Ordering::Relaxed);
            }
            0xE0 => {
                EXT_NEXT.store(true, Ordering::Relaxed);
            }
            _ => {
                let breaking = BREAK_NEXT.swap(false, Ordering::Relaxed);
                let extended = EXT_NEXT.swap(false, Ordering::Relaxed);
                if extended {
                    // Extended (E0-prefixed set-2) keys become the ANSI escape
                    // sequences a serial terminal sends — one encoding for every
                    // input path. Arrows + Home/End/PgUp/PgDn/Delete so the
                    // shell's history nav and pane scrollback work from a PS/2
                    // keyboard (VirtualBox-ARM presents one). Ctrl+Tab (E0 0x14)
                    // is the pane-focus toggle. GUI (E0 0x1f / 0x27) tracks ⌘/Super.
                    if !breaking {
                        if sc == 0x14 {
                            CTRL_DOWN.store(true, Ordering::Relaxed);
                            continue;
                        }
                        if sc == 0x1f || sc == 0x27 {
                            GUI_DOWN.store(true, Ordering::Relaxed);
                            continue;
                        }
                        if let Some(seq) = match sc {
                            0x75 => Some(&b"[A"[..]),  // Up
                            0x72 => Some(&b"[B"[..]),  // Down
                            0x74 => Some(&b"[C"[..]),  // Right
                            0x6b => Some(&b"[D"[..]),  // Left
                            0x6c => Some(&b"[H"[..]),  // Home
                            0x69 => Some(&b"[F"[..]),  // End
                            0x7d => Some(&b"[5~"[..]), // Page Up
                            0x7a => Some(&b"[6~"[..]), // Page Down
                            0x71 => Some(&b"[3~"[..]), // Delete
                            _ => None,
                        } {
                            PENDING.with(|p| p.extend_from_slice(seq));
                            return Some(0x1b);
                        }
                    } else if sc == 0x14 {
                        CTRL_DOWN.store(false, Ordering::Relaxed); // right Ctrl release
                    } else if sc == 0x1f || sc == 0x27 {
                        GUI_DOWN.store(false, Ordering::Relaxed);
                    }
                    continue;
                }
                match sc {
                    SC_LSHIFT | SC_RSHIFT => SHIFT_DOWN.store(!breaking, Ordering::Relaxed),
                    SC_LCTRL => CTRL_DOWN.store(!breaking, Ordering::Relaxed),
                    SC_CAPS if !breaking => {
                        CAPS_ON.fetch_xor(true, Ordering::Relaxed);
                    }
                    // Ctrl+Tab: pane-focus toggle, encoded as the private CSI `ESC [ T`.
                    0x0D if !breaking && CTRL_DOWN.load(Ordering::Relaxed) => {
                        PENDING.with(|p| p.extend_from_slice(b"[T"));
                        return Some(0x1b);
                    }
                    // Cmd/Super+Space or Ctrl+Space: Agents browser (`ESC [ g`).
                    // set-2 Space=0x29. Ctrl+Space is the reliable chord when a
                    // macOS host steals ⌘+Space for Spotlight.
                    0x29 if !breaking
                        && (GUI_DOWN.load(Ordering::Relaxed) || CTRL_DOWN.load(Ordering::Relaxed)) =>
                    {
                        PENDING.with(|p| p.extend_from_slice(b"[g"));
                        return Some(0x1b);
                    }
                    _ if !breaking => {
                        if let Some(ch) = decode(sc) {
                            return Some(ch);
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    None
}
