//! **PS/2 mouse over the ARM PL050 PrimeCell (KMI)** — the aarch64 analogue of
//! the x86 i8042 aux mouse. On ARM the pointing device on a PS/2 setup (e.g.
//! VirtualBox-ARM with `hidpointing=ps2mouse`) is a second PL050 KMI, distinct
//! from the keyboard KMI. This driver probes for a PL050 that the keyboard did
//! NOT claim, runs the standard PS/2 mouse init (reset → enable reporting), and
//! on each poll decodes the 3-byte movement packets into [`crate::mouse`].
//!
//! Polled, no IRQ — same as the PL050 keyboard. Safe no-op where no PL050 is
//! present (QEMU `virt`): the PrimeCell-ID probe fails.

use crate::mm::Locked;
use core::ptr::{read_volatile, write_volatile};

const KMICR: u64 = 0x00;
const KMISTAT: u64 = 0x04;
const KMIDATA: u64 = 0x08;
const KMICR_EN: u32 = 1 << 2;
const KMISTAT_RXFULL: u32 = 1 << 4;
const KMISTAT_TXEMPTY: u32 = 1 << 6;

const PL050_PART: u32 = 0x050;
const PRIMECELL_ID: u32 = 0xB105_F00D;

// Candidate KMI bases for the mouse across common ARM machines (KMI1 = the
// second KMI, conventionally the mouse). Each is probed by PrimeCell id, so a
// wrong guess is simply skipped; all sit in the identity-mapped low-1 GiB.
const CANDIDATE_BASES: [u64; 4] = [
    0x1c07_0000, // Versatile Express KMI1 (mouse)
    0x1000_7000, // Versatile / RealView KMI1 (mouse)
    0x1c06_0000, // …or KMI0 if the keyboard is elsewhere (e.g. USB)
    0x1000_6000,
];

static BASE: Locked<u64> = Locked::new(0);
/// PS/2 packet accumulator (4 bytes when the IntelliMouse wheel is enabled).
static PKT: Locked<[u8; 4]> = Locked::new([0; 4]);
static PKT_LEN: Locked<usize> = Locked::new(0);
/// Packet size: 4 once the scroll wheel (IntelliMouse) is negotiated, else 3.
static PKT_SIZE: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(3);

unsafe fn r32(a: u64) -> u32 {
    unsafe { read_volatile(a as *const u32) }
}
unsafe fn w32(a: u64, v: u32) {
    unsafe { write_volatile(a as *mut u32, v) };
}
unsafe fn id_block(base: u64, first: u64) -> u32 {
    unsafe {
        (r32(base + first) & 0xff)
            | ((r32(base + first + 4) & 0xff) << 8)
            | ((r32(base + first + 8) & 0xff) << 16)
            | ((r32(base + first + 12) & 0xff) << 24)
    }
}

/// Send a byte to the PS/2 device and read its one-byte ACK (0xFA). Bounded
/// spins so a missing device can't hang boot.
unsafe fn cmd(base: u64, byte: u8) -> Option<u8> {
    unsafe {
        let mut spins = 0;
        while r32(base + KMISTAT) & KMISTAT_TXEMPTY == 0 {
            spins += 1;
            if spins > 1_000_000 {
                return None;
            }
        }
        w32(base + KMIDATA, byte as u32);
        // Read the ACK.
        spins = 0;
        while r32(base + KMISTAT) & KMISTAT_RXFULL == 0 {
            spins += 1;
            if spins > 2_000_000 {
                return None;
            }
        }
        Some((r32(base + KMIDATA) & 0xff) as u8)
    }
}

/// Probe for a PL050 mouse KMI (not the keyboard's) and initialize the PS/2
/// mouse (enable reporting). Returns true on success.
pub fn init() -> bool {
    let kbd = crate::arch::aarch64::pl050::keyboard_base();
    for &base in &CANDIDATE_BASES {
        if base == kbd {
            continue;
        }
        // SAFETY: identity-mapped Device MMIO; PrimeCell id registers are RO.
        unsafe {
            if id_block(base, 0xFF0) != PRIMECELL_ID || id_block(base, 0xFE0) & 0xfff != PL050_PART {
                continue;
            }
            w32(base + KMICR, KMICR_EN);
            // PS/2 mouse init: reset (0xFF → ACK, self-test 0xAA, id 0x00),
            // then enable data reporting (0xF4 → ACK). Best-effort: some models
            // skip the reset handshake.
            let _ = cmd(base, 0xFF);
            // drain self-test / id bytes
            for _ in 0..8 {
                if r32(base + KMISTAT) & KMISTAT_RXFULL == 0 {
                    break;
                }
                let _ = r32(base + KMIDATA);
            }
            // Enable the IntelliMouse scroll wheel: the magic sample-rate knock
            // (200, 100, 80) then read the device id — 0x03 means the device
            // switched to 4-byte packets whose 4th byte is a signed Z (wheel).
            let _ = cmd(base, 0xF3);
            let _ = cmd(base, 200);
            let _ = cmd(base, 0xF3);
            let _ = cmd(base, 100);
            let _ = cmd(base, 0xF3);
            let _ = cmd(base, 80);
            let _ = cmd(base, 0xF2); // get device id
            let id = if r32(base + KMISTAT) & KMISTAT_RXFULL != 0 { (r32(base + KMIDATA) & 0xff) as u8 } else { 0 };
            let wheel = id == 0x03;
            PKT_SIZE.store(if wheel { 4 } else { 3 }, core::sync::atomic::Ordering::Relaxed);
            if cmd(base, 0xF4).is_none() {
                continue;
            }
            BASE.with(|b| *b = base);
            crate::ktrace::log_fmt(format_args!(
                "pl050: PS/2 mouse at {:#x} (data reporting on{})",
                base,
                if wheel { ", scroll wheel" } else { "" }
            ));
            return true;
        }
    }
    false
}

/// Drain PS/2 mouse packets into [`crate::mouse`]. Each packet is 3 bytes:
/// flags (buttons + movement sign bits), dx, dy (dy is up-positive → screen −).
pub fn poll() {
    let base = BASE.with(|b| *b);
    if base == 0 {
        return;
    }
    for _ in 0..64 {
        // SAFETY: `base` is the probed, enabled PL050 mouse KMI.
        let byte = unsafe {
            if r32(base + KMISTAT) & KMISTAT_RXFULL == 0 {
                return;
            }
            (r32(base + KMIDATA) & 0xff) as u8
        };
        let len = PKT_LEN.with(|l| *l);
        // Resync: the first packet byte always has bit 3 set.
        if len == 0 && byte & 0x08 == 0 {
            continue;
        }
        PKT.with(|p| p[len] = byte);
        let len = len + 1;
        PKT_LEN.with(|l| *l = len);
        let size = PKT_SIZE.load(core::sync::atomic::Ordering::Relaxed);
        if len == size {
            PKT_LEN.with(|l| *l = 0);
            let (flags, bx, by, bz) = PKT.with(|p| (p[0], p[1], p[2], p[3]));
            let dx = bx as i32 - if flags & 0x10 != 0 { 256 } else { 0 };
            let dy = by as i32 - if flags & 0x20 != 0 { 256 } else { 0 };
            crate::mouse::move_rel(dx, -dy); // PS/2 Y is up-positive
            crate::mouse::set_left(flags & 0x01 != 0);
            if size == 4 {
                // 4th byte is a signed Z: +1 = wheel toward the user (scroll
                // down). Negate so "wheel up" is a positive delta (scroll back
                // through history), matching the virtio pointer's convention.
                let dz = bz as i8 as i32;
                if dz != 0 {
                    crate::mouse::add_wheel(-dz);
                }
            }
        }
    }
}
