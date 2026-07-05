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
}

static M: Locked<Mouse> = Locked::new(Mouse { x: 400, y: 300, left: false, prev_left: false, moved: false });

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

/// One idle tick's worth of mouse activity.
pub struct Tick {
    pub moved: bool,
    pub x: u64,
    pub y: u64,
    pub pressed: bool,  // left button just went down
    pub released: bool, // left button just came up
    pub left: bool,     // current left-button state
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
        };
        s.prev_left = s.left;
        s.moved = false;
        if t.moved || pressed || released {
            ACTIVITY_MS.store(crate::arch::now_ms(), Ordering::Relaxed);
        }
        t
    })
}
