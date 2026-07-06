//! i8042 PS/2 controller — the **aux (mouse) port**. The PS/2 keyboard is
//! IRQ1-driven in [`super::keyboard`]; this module brings up the controller's
//! second port and polls the standard 3-byte PS/2 mouse protocol into
//! [`crate::mouse`], giving x86 a pointer on VirtualBox, QEMU `q35`, and real
//! PCs without needing a USB tablet. Polled (no IRQ12): the UI idle loops
//! call [`poll_mouse`] via `arch::mouse_poll`, matching the aarch64 PL050
//! mouse driver's model.

use super::port::{inb, outb};
use crate::mm::Locked;
use core::sync::atomic::{AtomicBool, Ordering};

const DATA: u16 = 0x60;
const STATUS: u16 = 0x64; // read: status; write: command
const ST_OUT_FULL: u8 = 1 << 0; // output buffer full (data readable at 0x60)
const ST_IN_FULL: u8 = 1 << 1; // input buffer full (controller busy)
const ST_AUX: u8 = 1 << 5; // the readable byte is from the aux (mouse) port

static UP: AtomicBool = AtomicBool::new(false);
/// Packet size: 4 once the IntelliMouse scroll wheel is negotiated, else 3.
static PKT_SIZE: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(3);

/// PS/2 mouse packet assembly state (4 bytes with the IntelliMouse wheel).
struct Pkt {
    buf: [u8; 4],
    n: usize,
}
static PKT: Locked<Pkt> = Locked::new(Pkt { buf: [0; 4], n: 0 });

/// Wait until the controller can accept a command/data byte. Bounded.
fn wait_write() -> bool {
    for _ in 0..50_000 {
        // SAFETY: status-register read; no side effects.
        if unsafe { inb(STATUS) } & ST_IN_FULL == 0 {
            return true;
        }
        core::hint::spin_loop();
    }
    false
}

/// Wait for (and read) one response byte. Bounded; `None` on timeout.
fn wait_read() -> Option<u8> {
    for _ in 0..50_000 {
        // SAFETY: status read, then a data read only when a byte is pending.
        unsafe {
            if inb(STATUS) & ST_OUT_FULL != 0 {
                return Some(inb(DATA));
            }
        }
        core::hint::spin_loop();
    }
    None
}

/// Send one byte to the **mouse** (aux device) via the 0xD4 controller prefix.
fn aux_write(b: u8) -> bool {
    if !wait_write() {
        return false;
    }
    // SAFETY: 0xD4 = "route next data byte to the aux device" — standard i8042.
    unsafe { outb(STATUS, 0xd4) };
    if !wait_write() {
        return false;
    }
    // SAFETY: the routed data byte.
    unsafe { outb(DATA, b) };
    true
}

/// Bring up the aux (mouse) port: enable it in the controller, reset the
/// mouse, and enable streaming reports. Safe no-op when no i8042/mouse
/// responds (bounded waits) — e.g. a machine without PS/2 at all.
pub fn init() {
    // Enable the aux port (0xA8), then clear the "disable aux clock" bit in
    // the controller config byte. IRQ12 stays disabled — we poll. The
    // keyboard's IRQ1 enable and scancode translation bits are preserved.
    if !wait_write() {
        return;
    }
    // SAFETY: standard i8042 command sequence; each write is gated on the
    // input buffer being empty.
    unsafe {
        outb(STATUS, 0xa8); // enable aux port
        if !wait_write() {
            return;
        }
        outb(STATUS, 0x20); // read config byte
    }
    let Some(mut cfg) = wait_read() else { return };
    cfg &= !(1 << 5); // aux clock enabled
    cfg &= !(1 << 1); // IRQ12 off (polled)
    // SAFETY: write the config byte back (0x60 = write config).
    unsafe {
        if !wait_write() {
            return;
        }
        outb(STATUS, 0x60);
        if !wait_write() {
            return;
        }
        outb(DATA, cfg);
    }
    // Reset the mouse (0xFF -> ACK 0xFA, self-test 0xAA, id 0x00), then set
    // defaults + enable streaming. Any timeout = no mouse present; bail.
    if !aux_write(0xff) || wait_read() != Some(0xfa) {
        return;
    }
    let _ = wait_read(); // 0xAA self-test pass
    let _ = wait_read(); // 0x00 device id
    if !aux_write(0xf6) || wait_read() != Some(0xfa) {
        return;
    }
    // Enable the IntelliMouse scroll wheel: the magic sample-rate knock (200,
    // 100, 80), then read the device id — 0x03 = the mouse switched to 4-byte
    // packets whose 4th byte is a signed Z (wheel).
    let set_rate = |r: u8| aux_write(0xf3) && wait_read() == Some(0xfa) && aux_write(r) && wait_read() == Some(0xfa);
    let _ = set_rate(200) && set_rate(100) && set_rate(80);
    let id = if aux_write(0xf2) && wait_read() == Some(0xfa) { wait_read() } else { None };
    let wheel = id == Some(0x03);
    PKT_SIZE.store(if wheel { 4 } else { 3 }, Ordering::Relaxed);
    if !aux_write(0xf4) || wait_read() != Some(0xfa) {
        return;
    }
    UP.store(true, Ordering::Relaxed);
    crate::ktrace::log(
        "i8042",
        if wheel { "PS/2 mouse up (aux port, polled, 4-byte + scroll wheel)" } else { "PS/2 mouse up (aux port, polled, 3-byte packets)" },
    );
}

/// Drain any pending aux bytes into [`crate::mouse`]. Called from the UI idle
/// loops via `arch::mouse_poll`. Keyboard bytes are left for IRQ1.
pub fn poll_mouse() {
    if !UP.load(Ordering::Relaxed) {
        return;
    }
    // Bounded per call so a wedged device can't stall the UI loop.
    for _ in 0..64 {
        // SAFETY: status read; the data read below only happens when the
        // status says an *aux* byte is pending (keyboard bytes stay for IRQ1).
        let st = unsafe { inb(STATUS) };
        if st & ST_OUT_FULL == 0 || st & ST_AUX == 0 {
            return;
        }
        let b = unsafe { inb(DATA) };
        PKT.with(|p| {
            // Byte 0 of a packet always has bit 3 set; resync on it.
            if p.n == 0 && b & 0x08 == 0 {
                return;
            }
            p.buf[p.n] = b;
            p.n += 1;
            if p.n == PKT_SIZE.load(Ordering::Relaxed) {
                let size = p.n;
                p.n = 0;
                let (flags, dx8, dy8) = (p.buf[0], p.buf[1], p.buf[2]);
                // Overflow packets are garbage; drop them.
                if flags & 0xc0 == 0 {
                    let dx = dx8 as i8 as i32;
                    let dy = dy8 as i8 as i32; // PS/2 y is up-positive
                    crate::mouse::move_rel(dx, -dy);
                }
                crate::mouse::set_left(flags & 0x01 != 0);
                if size == 4 {
                    // 4th byte = signed Z (wheel); +1 = toward the user (scroll
                    // down). Negate so "wheel up" is positive = scroll back.
                    let dz = p.buf[3] as i8 as i32;
                    if dz != 0 {
                        crate::mouse::add_wheel(-dz);
                    }
                }
            }
        });
    }
}
