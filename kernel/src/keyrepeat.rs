//! Software keyboard auto-repeat — the pure timing/state logic behind two
//! seams, kept out of the drivers so it is unit-testable (the standing rule:
//! fiddly logic lives in pure functions, the hardware path just calls it).
//!
//! * [`Typematic`] gives a held key hardware-style repeat where none exists:
//!   USB HID boot keyboards report only press/release edges, so `xhci`
//!   arms a `Typematic` with the byte sequence a press emitted and re-emits it
//!   after an initial delay, at an interval that **accelerates** the longer
//!   the key is held.
//! * [`Accel`] amplifies an already-repeating stream (PS/2 hardware typematic,
//!   a host's virtio-input autorepeat, or `Typematic` itself): the shell and
//!   editor ask it how many steps a repeated erase/nav key is worth, so a
//!   long-held Backspace/Up/Down erases or scrolls progressively faster while
//!   a single tap always means exactly one step.
//!
//! Both are driven by `arch::now_ms()` passed in by the caller — nothing here
//! reads the clock, so tests control time exactly.

/// Milliseconds a key must be held before the first synthesized repeat.
pub const DELAY_MS: u64 = 400;

/// Repeat interval for a key held `held_ms`: starts at a comfortable typing
/// rate and accelerates in two steps the longer the key stays down.
pub fn interval_ms(held_ms: u64) -> u64 {
    if held_ms < 1500 {
        70
    } else if held_ms < 3000 {
        35
    } else {
        18
    }
}

/// Longest byte sequence a single key emits (ESC + `[5~`).
pub const SEQ_MAX: usize = 4;

/// Hold-to-repeat state for one keyboard: the byte sequence the held key
/// emitted on press, and when the next synthesized repeat is due.
pub struct Typematic {
    seq: [u8; SEQ_MAX],
    len: u8,
    pressed_ms: u64,
    due_ms: u64,
    active: bool,
}

impl Typematic {
    pub const fn new() -> Typematic {
        Typematic { seq: [0; SEQ_MAX], len: 0, pressed_ms: 0, due_ms: 0, active: false }
    }

    /// Arm repeat for a key that just emitted `seq` (a new press replaces any
    /// previously held key, like a real keyboard's typematic).
    pub fn press(&mut self, seq: &[u8], now: u64) {
        let n = seq.len().min(SEQ_MAX);
        self.seq[..n].copy_from_slice(&seq[..n]);
        self.len = n as u8;
        self.pressed_ms = now;
        self.due_ms = now + DELAY_MS;
        self.active = n > 0;
    }

    /// Disarm (the held key was released).
    pub fn release(&mut self) {
        self.active = false;
    }

    /// If a repeat is due at `now`, return the byte sequence to re-emit and
    /// schedule the next one (accelerating with hold time). At most one repeat
    /// per call: the caller polls every idle iteration, so a stalled poll loop
    /// never dumps a burst of queued repeats at once.
    pub fn poll(&mut self, now: u64) -> Option<([u8; SEQ_MAX], usize)> {
        if !self.active || now < self.due_ms {
            return None;
        }
        self.due_ms = now + interval_ms(now.saturating_sub(self.pressed_ms));
        Some((self.seq, self.len as usize))
    }
}

/// How closely repeats must follow each other to count as one held-key streak.
/// PS/2 typematic ticks ~92 ms apart, host autorepeat 30–50 ms, [`Typematic`]
/// 18–70 ms — all well inside; deliberate re-taps of the same key are not.
const STREAK_GAP_MS: u64 = 200;

/// Streak amplifier for repeating erase/nav keys: consecutive arrivals of the
/// same key within [`STREAK_GAP_MS`] build a streak, and the streak buys
/// multiple steps per event — so holding Backspace erases 1, then 2, 4, 8
/// characters per repeat the longer it is held.
pub struct Accel {
    key: u8,
    last_ms: u64,
    streak: u32,
}

impl Accel {
    pub const fn new() -> Accel {
        Accel { key: 0, last_ms: 0, streak: 0 }
    }

    /// Record an arrival of `key` at `now` and return how many steps it is
    /// worth (>= 1).
    pub fn steps(&mut self, key: u8, now: u64) -> usize {
        if key == self.key && now.saturating_sub(self.last_ms) <= STREAK_GAP_MS {
            self.streak = self.streak.saturating_add(1);
        } else {
            self.streak = 0;
        }
        self.key = key;
        self.last_ms = now;
        match self.streak {
            0..=11 => 1,
            12..=27 => 2,
            28..=55 => 4,
            _ => 8,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn typematic_delay_then_repeat() {
        let mut t = Typematic::new();
        t.press(b"\x08", 1000);
        assert!(t.poll(1000).is_none(), "no repeat before the initial delay");
        assert!(t.poll(1399).is_none());
        let (seq, n) = t.poll(1400).expect("first repeat at press + DELAY_MS");
        assert_eq!((n, seq[0]), (1, 0x08));
        assert!(t.poll(1401).is_none(), "next repeat not due yet");
        assert!(t.poll(1400 + 70).is_some(), "early-hold interval is 70 ms");
    }

    #[test_case]
    fn typematic_accelerates_with_hold_time() {
        assert_eq!(interval_ms(0), 70);
        assert_eq!(interval_ms(1499), 70);
        assert_eq!(interval_ms(1500), 35);
        assert_eq!(interval_ms(3000), 18);
        let mut t = Typematic::new();
        t.press(b"\x1b[A", 0);
        // Drain until past 3 s of hold; the gap between repeats must shrink.
        let mut now = 0u64;
        let mut prev_gap = u64::MAX;
        let mut last = 0u64;
        while now < 4000 {
            now += 1;
            if t.poll(now).is_some() {
                let gap = now - last;
                if last != 0 {
                    assert!(gap <= prev_gap, "repeat interval must never grow while held");
                    prev_gap = gap;
                }
                last = now;
            }
        }
        assert_eq!(prev_gap, 18, "late-hold interval reaches the fast rate");
    }

    #[test_case]
    fn typematic_release_and_replace() {
        let mut t = Typematic::new();
        t.press(b"a", 0);
        t.release();
        assert!(t.poll(10_000).is_none(), "released key never repeats");
        t.press(b"\x1b[5~", 0);
        let (seq, n) = t.poll(DELAY_MS).expect("re-armed after a new press");
        assert_eq!(&seq[..n], b"\x1b[5~");
    }

    #[test_case]
    fn accel_streak_grows_and_resets() {
        let mut a = Accel::new();
        let mut now = 0u64;
        // A fresh tap is always exactly one step.
        assert_eq!(a.steps(0x08, now), 1);
        // Repeats 50 ms apart build the streak: 1 → 2 → 4 → 8.
        let mut seen = alloc::vec::Vec::new();
        for _ in 0..60 {
            now += 50;
            seen.push(a.steps(0x08, now));
        }
        assert_eq!(seen[0], 1);
        assert!(seen.contains(&2) && seen.contains(&4), "streak passes 2x and 4x");
        assert_eq!(*seen.last().unwrap(), 8, "long hold reaches 8x");
        // A pause resets to a single step; so does switching keys.
        assert_eq!(a.steps(0x08, now + 1000), 1);
        assert_eq!(a.steps(b'A', now + 1050), 1, "different key starts a new streak");
    }
}
