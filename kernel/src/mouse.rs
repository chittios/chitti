//! Arch-neutral mouse state + the per-idle `tick`. Transport drivers (virtio
//! pointer, USB xHCI/HID boot mouse, PS/2 aux) feed absolute or relative motion
//! and button state here; the UI loops (shell, editor) call [`tick`], which
//! polls the drivers, and act on the returned edge events (move / press /
//! release). Coordinates are framebuffer pixels, clamped to the screen.

use crate::mm::Locked;

struct Mouse {
    x: i32,
    y: i32,
    left: bool,
    prev_left: bool,
    moved: bool,
    /// Accumulated scroll-wheel delta since the last [`tick`] (+ = wheel up /
    /// away from the user). Drained per tick.
    wheel: i32,
}

static M: Locked<Mouse> = Locked::new(Mouse { x: 400, y: 300, left: false, prev_left: false, moved: false, wheel: 0 });

/// Framebuffer size, or `None` in the test build (no framebuffer compiled).
#[cfg(not(test))]
fn screen() -> Option<(u64, u64)> {
    crate::framebuffer::screen_dims()
}
#[cfg(test)]
fn screen() -> Option<(u64, u64)> {
    None
}

/// Feed an absolute position (tablet): raw `(rx, ry)` in `0..=max`, scaled to the
/// framebuffer.
pub fn set_abs(rx: i32, ry: i32, max: i32) {
    if let Some((w, h)) = screen() {
        let m = max.max(1) as i64;
        let x = (rx as i64 * (w as i64 - 1) / m) as i32;
        let y = (ry as i64 * (h as i64 - 1) / m) as i32;
        M.with(|s| {
            if x != s.x || y != s.y {
                s.moved = true;
            }
            s.x = x;
            s.y = y;
        });
    }
}

/// Feed a relative motion (mouse): `dy` is screen-down positive.
pub fn move_rel(dx: i32, dy: i32) {
    if let Some((w, h)) = screen() {
        M.with(|s| {
            let nx = (s.x + dx).clamp(0, w as i32 - 1);
            let ny = (s.y + dy).clamp(0, h as i32 - 1);
            if nx != s.x || ny != s.y {
                s.moved = true;
            }
            s.x = nx;
            s.y = ny;
        });
    }
}

/// Set the left-button state.
pub fn set_left(down: bool) {
    M.with(|s| s.left = down);
}

/// Feed a scroll-wheel delta (+ = wheel up / away). Accumulated until [`tick`].
pub fn add_wheel(dz: i32) {
    M.with(|s| s.wheel += dz);
}

/// A decoded PS/2 mouse packet: screen-oriented motion, button, wheel.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct Ps2Delta {
    /// Horizontal motion (right positive).
    pub dx: i32,
    /// Vertical motion in **screen** orientation (down positive — PS/2's
    /// up-positive Y is already negated here).
    pub dy: i32,
    /// Left button currently down.
    pub left: bool,
    /// Scroll delta (+ = wheel up / away from the user — PS/2's toward-user-
    /// positive Z is negated to match [`add_wheel`]'s convention). 0 for a
    /// 3-byte packet with no wheel.
    pub wheel: i32,
}

/// Decode a standard PS/2 mouse packet (shared by the aarch64 PL050 and x86
/// i8042 aux drivers, and unit-tested — the packet layout is fiddly and easy
/// to break). `size` is 3 (plain) or 4 (IntelliMouse wheel). Byte 0 is the
/// flags/sign byte; on an X/Y **overflow** the motion is dropped (garbage) but
/// the button state is still reported. Returns `None` if `pkt` is too short.
pub fn decode_ps2_packet(pkt: &[u8], size: usize) -> Option<Ps2Delta> {
    if pkt.len() < size || size < 3 {
        return None;
    }
    let flags = pkt[0];
    let left = flags & 0x01 != 0;
    // X/Y overflow bits (0x40/0x80): the movement bytes are meaningless, so
    // report zero motion but keep the button state.
    let (dx, dy) = if flags & 0xc0 != 0 {
        (0, 0)
    } else {
        let dx = pkt[1] as i32 - if flags & 0x10 != 0 { 256 } else { 0 };
        let dy = pkt[2] as i32 - if flags & 0x20 != 0 { 256 } else { 0 };
        (dx, -dy) // PS/2 Y is up-positive; screen is down-positive
    };
    // IntelliMouse Z (byte 3): +1 = wheel toward the user (scroll down); negate
    // so "wheel up" is positive, matching the wheel convention elsewhere.
    let wheel = if size >= 4 { -(pkt[3] as i8 as i32) } else { 0 };
    Some(Ps2Delta { dx, dy, left, wheel })
}

/// One idle tick's worth of mouse activity.
pub struct Tick {
    pub moved: bool,
    pub x: u64,
    pub y: u64,
    pub pressed: bool,  // left button just went down
    pub released: bool, // left button just came up
    pub left: bool,     // current left-button state
    pub wheel: i32,     // scroll delta this tick (+ = up / away from the user)
}

use core::sync::atomic::{AtomicU64, Ordering};

/// `now_ms` of the last mouse motion/click — drives the status-bar indicator.
static ACTIVITY_MS: AtomicU64 = AtomicU64::new(0);

/// When mouse activity was last seen (`arch::now_ms` timebase; 0 = never).
pub fn activity_ms() -> u64 {
    ACTIVITY_MS.load(Ordering::Relaxed)
}

/// Poll the transport drivers and fold the state into edge events for this tick.
pub fn tick() -> Tick {
    crate::arch::mouse_poll();
    M.with(|s| {
        let pressed = s.left && !s.prev_left;
        let released = !s.left && s.prev_left;
        let t = Tick {
            moved: s.moved,
            x: s.x.max(0) as u64,
            y: s.y.max(0) as u64,
            pressed,
            released,
            left: s.left,
            wheel: s.wheel,
        };
        s.prev_left = s.left;
        s.moved = false;
        s.wheel = 0;
        if t.moved || pressed || released || t.wheel != 0 {
            ACTIVITY_MS.store(crate::arch::now_ms(), Ordering::Relaxed);
        }
        t
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Byte 0 always has bit 3 set on a real packet; include it in `flags`.
    #[test_case]
    fn ps2_basic_motion_and_button() {
        // right 5, up 3 (PS/2 up-positive) → screen dy = -3; left down.
        let d = decode_ps2_packet(&[0x09, 5, 3], 3).unwrap();
        assert_eq!(d.dx, 5);
        assert_eq!(d.dy, -3);
        assert!(d.left);
        assert_eq!(d.wheel, 0);
    }

    #[test_case]
    fn ps2_sign_bits() {
        // X sign bit (0x10): byte 0xFE → -2. Y sign bit (0x20): byte 0xFE →
        // raw -2 → screen +2. No button.
        let d = decode_ps2_packet(&[0x08 | 0x10 | 0x20, 0xFE, 0xFE], 3).unwrap();
        assert_eq!(d.dx, -2);
        assert_eq!(d.dy, 2);
        assert!(!d.left);
    }

    #[test_case]
    fn ps2_overflow_drops_motion_keeps_button() {
        // X-overflow (0x40) → motion zeroed, but the left button still reports.
        let d = decode_ps2_packet(&[0x08 | 0x40 | 0x01, 0x7F, 0x7F], 3).unwrap();
        assert_eq!((d.dx, d.dy), (0, 0));
        assert!(d.left);
    }

    #[test_case]
    fn ps2_wheel_sign() {
        // IntelliMouse Z: +1 = toward the user (scroll down) → wheel -1;
        // 0xFF (-1) = away (scroll up) → wheel +1.
        assert_eq!(decode_ps2_packet(&[0x08, 0, 0, 0x01], 4).unwrap().wheel, -1);
        assert_eq!(decode_ps2_packet(&[0x08, 0, 0, 0xFF], 4).unwrap().wheel, 1);
        // A 3-byte packet never yields a wheel delta, even with a 4th byte present.
        assert_eq!(decode_ps2_packet(&[0x08, 0, 0, 0x01], 3).unwrap().wheel, 0);
    }

    #[test_case]
    fn ps2_short_packet_is_rejected() {
        assert!(decode_ps2_packet(&[0x08, 1], 3).is_none());
        assert!(decode_ps2_packet(&[0x08, 1, 2], 4).is_none());
    }
}
