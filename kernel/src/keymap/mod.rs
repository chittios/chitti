//! **The keyboard choke point** — one place that decides what a key means.
//!
//! Before this existed there were four independent scancode→ASCII decoders, one
//! per transport (`arch/x86_64/keyboard.rs` set-1, `arch/aarch64/pl050.rs`
//! set-2, `xhci.rs` HID usages, `arch/aarch64/virtio_input.rs` evdev keycodes),
//! each with its own modifier state, its own caps-lock rule (two had one, two
//! did not), its own copy of the arrow→CSI table, and all four hard-coded to the
//! **US** layout. Adding a layout there would have meant four insertion points
//! and four copies of the dead-key state machine.
//!
//! Four facts about this tree forced the shape:
//!
//! 1. **x86 decodes inside IRQ1.** `keyboard_handler` does its table lookup and
//!    ring push in interrupt context. So layout tables, dead-key state and IME
//!    state cannot live "in the drivers": one of the four cannot allocate. Hence
//!    the split below — the IRQ side only builds a 4-byte [`KeyEvent`] and pushes
//!    it into a fixed ring, and [`translate`] runs on the **drain** side, in task
//!    context, where `Locked` and allocation are ordinary.
//! 2. **`arch::aarch64` is `cfg`'d out of the test build.** Nothing in `pl050.rs`
//!    or `virtio_input.rs` can ever carry a `#[test_case]` — the same
//!    silent-dead-test problem as `framebuffer/`, from a different `cfg`. So the
//!    set-2 and evdev cross-tables **must** live here or they are permanently
//!    untestable.
//! 3. **Dead keys and IME are stateful across keystrokes**, and a machine can
//!    have a USB keyboard *and* a virtio-input window at once (`console.rs` polls
//!    both). Four states would mean `´` on one keyboard and `e` on the other
//!    fails to compose, and caps-lock would not survive switching between them.
//! 4. **`console::read_byte() -> Option<u8>` does not change.** The event type
//!    stops at this boundary; above it the OS still sees a byte stream, so the
//!    shell, the editor, modals, `poll_interrupt`, bracketed paste and every e2e
//!    scenario are untouched.
//!
//! ## Why HID usages are the canonical space
//!
//! A layout maps a *physical position* to a character, so the canonical space has
//! to be positional. HID usages are, and so are set-1, set-2 and evdev — the
//! three cross-tables here are pure relabellings, about 350 bytes of rodata
//! total. Usages are also what `xhci` already speaks, so the one transport that
//! matters on real hardware needs no translation at all. An ASCII-based canonical
//! space could not express "the key left of Y on an ISO board" (usage `0x64`),
//! which is where German puts `<>|` and which no previous table decoded at all.

use alloc::string::String;
use core::sync::atomic::Ordering;
use crate::mm::Locked;

pub mod layouts;

pub use layouts::{Dead, KeyDef, Layout, Level, Out};

/// A physical key, canonically a USB HID Keyboard/Keypad (page 0x07) usage id.
pub type Usage = u8;

/// Modifier state at the moment of a press.
///
/// Left and right Alt are **distinct**. On German/French/Spanish layouts the
/// right one selects a layout level and the left one is a meta modifier, and
/// conflating them is exactly why AltGr did not work: no driver decoded the right
/// Alt bit at all (`xhci` read `report[0] & 0x44`, the OR of both, into nothing).
#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub struct Mods(pub u8);

impl Mods {
    pub const SHIFT: u8 = 1 << 0;
    /// Left Alt / Option — a meta modifier, never a layout level.
    pub const ALT: u8 = 1 << 2;
    pub const CTRL: u8 = 1 << 1;
    /// Right Alt — layout level 3.
    pub const ALTGR: u8 = 1 << 3;
    pub const GUI: u8 = 1 << 4;
    pub const CAPS: u8 = 1 << 5;

    pub const fn new(bits: u8) -> Mods {
        Mods(bits)
    }
    pub const fn has(self, m: u8) -> bool {
        self.0 & m != 0
    }
    pub fn set(&mut self, m: u8, on: bool) {
        if on {
            self.0 |= m;
        } else {
            self.0 &= !m;
        }
    }
    /// Whether this state selects the AltGr level.
    ///
    /// `Ctrl+Alt` counts, exactly as XKB does — many keyboards have no right Alt,
    /// and a macOS host often eats Option before the guest sees it. Applied once,
    /// here, so no driver has to know the convention.
    pub const fn altgr(self) -> bool {
        self.has(Self::ALTGR) || (self.has(Self::CTRL) && self.has(Self::ALT))
    }
}

/// Which transport a press came from.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Source {
    Ps2Set1,
    Ps2Set2,
    Evdev,
    UsbHid,
}

impl Source {
    /// Whether the source delivers its own key repeats.
    ///
    /// USB HID boot keyboards report only press *edges*, so software typematic is
    /// required there. PS/2 hardware and a virtio-input host both repeat on their
    /// own, and stacking our typematic on top of theirs would double every held
    /// key — a latent bug the old code avoided only because `xhci` was the sole
    /// user of [`crate::keyrepeat`].
    pub fn repeats_in_hardware(self) -> bool {
        !matches!(self, Source::UsbHid)
    }
}

/// One normalized key transition: 4 bytes, `Copy`, no allocation — safe to build
/// inside an interrupt handler and push into a fixed-size ring.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct KeyEvent {
    pub usage: Usage,
    pub mods: Mods,
    pub pressed: bool,
    pub src: Source,
}

// ---------------------------------------------------------------------------
// Modifier usages
// ---------------------------------------------------------------------------

pub const U_LCTRL: Usage = 0xE0;
pub const U_LSHIFT: Usage = 0xE1;
pub const U_LALT: Usage = 0xE2;
pub const U_LGUI: Usage = 0xE3;
pub const U_RCTRL: Usage = 0xE4;
pub const U_RSHIFT: Usage = 0xE5;
pub const U_RALT: Usage = 0xE6;
pub const U_RGUI: Usage = 0xE7;
pub const U_CAPSLOCK: Usage = 0x39;

/// The modifier bit a usage toggles, if it is a modifier at all.
pub fn modifier_bit(u: Usage) -> Option<u8> {
    Some(match u {
        U_LSHIFT | U_RSHIFT => Mods::SHIFT,
        U_LCTRL | U_RCTRL => Mods::CTRL,
        U_LALT => Mods::ALT,
        U_RALT => Mods::ALTGR,
        U_LGUI | U_RGUI => Mods::GUI,
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// Cross-tables: transport scancode -> HID usage
// ---------------------------------------------------------------------------

/// PS/2 **scan-code set 1** make code -> HID usage. Index is the make code.
#[rustfmt::skip]
static SET1_TO_USAGE: [Usage; 0x60] = [
/* 00 */ 0,    0x29, 0x1e, 0x1f, 0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x2d, 0x2e, 0x2a, 0x2b,
/* 10 */ 0x14, 0x1a, 0x08, 0x15, 0x17, 0x1c, 0x18, 0x0c, 0x12, 0x13, 0x2f, 0x30, 0x28, 0xE0, 0x04, 0x16,
/* 20 */ 0x07, 0x09, 0x0a, 0x0b, 0x0d, 0x0e, 0x0f, 0x33, 0x34, 0x35, 0xE1, 0x31, 0x1d, 0x1b, 0x06, 0x19,
/* 30 */ 0x05, 0x11, 0x10, 0x36, 0x37, 0x38, 0xE5, 0x55, 0xE2, 0x2c, 0x39, 0x3a, 0x3b, 0x3c, 0x3d, 0x3e,
/* 40 */ 0x3f, 0x40, 0x41, 0x42, 0x43, 0x53, 0x47, 0x5f, 0x60, 0x61, 0x56, 0x5c, 0x5d, 0x5e, 0x57, 0x59,
/* 50 */ 0x5a, 0x5b, 0x62, 0x63, 0,    0,    0x64, 0x44, 0x45, 0,    0,    0,    0,    0,    0,    0,
];

/// PS/2 **scan-code set 2** make code -> HID usage. Index is the make code.
#[rustfmt::skip]
static SET2_TO_USAGE: [Usage; 0x84] = [
/* 00 */ 0,    0x42, 0,    0x3e, 0x3c, 0x3a, 0x3b, 0x45, 0,    0x43, 0x41, 0x3f, 0x3d, 0x2b, 0x35, 0,
/* 10 */ 0,    0xE2, 0xE1, 0,    0xE0, 0x14, 0x1e, 0,    0,    0,    0x1d, 0x16, 0x04, 0x1a, 0x1f, 0,
/* 20 */ 0,    0x06, 0x1b, 0x07, 0x08, 0x21, 0x20, 0,    0,    0x2c, 0x19, 0x09, 0x17, 0x15, 0x22, 0,
/* 30 */ 0,    0x11, 0x05, 0x0b, 0x0a, 0x1c, 0x23, 0,    0,    0,    0x10, 0x0d, 0x0c, 0x18, 0x24, 0x25,
/* 40 */ 0,    0x36, 0x0e, 0x0c, 0x12, 0x27, 0x26, 0,    0,    0x37, 0x38, 0x0f, 0x33, 0x13, 0x2d, 0,
/* 50 */ 0,    0,    0x34, 0,    0x2f, 0x2e, 0,    0,    0x39, 0xE5, 0x28, 0x30, 0,    0x31, 0,    0,
/* 60 */ 0,    0x64, 0,    0,    0,    0,    0x2a, 0,    0,    0x59, 0,    0x5c, 0x5f, 0,    0,    0,
/* 70 */ 0x62, 0x63, 0x5a, 0x5d, 0x5e, 0x60, 0x29, 0x53, 0x44, 0x57, 0x5b, 0x56, 0x55, 0x61, 0x47, 0,
/* 80 */ 0,    0,    0,    0x40,
];

/// Linux **evdev** keycode -> HID usage. Index is the keycode.
#[rustfmt::skip]
static EVDEV_TO_USAGE: [Usage; 128] = [
/*   0 */ 0,    0x29, 0x1e, 0x1f, 0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x2d, 0x2e, 0x2a, 0x2b,
/*  16 */ 0x14, 0x1a, 0x08, 0x15, 0x17, 0x1c, 0x18, 0x0c, 0x12, 0x13, 0x2f, 0x30, 0x28, 0xE0, 0x04, 0x16,
/*  32 */ 0x07, 0x09, 0x0a, 0x0b, 0x0d, 0x0e, 0x0f, 0x33, 0x34, 0x35, 0xE1, 0x31, 0x1d, 0x1b, 0x06, 0x19,
/*  48 */ 0x05, 0x11, 0x10, 0x36, 0x37, 0x38, 0xE5, 0x55, 0xE2, 0x2c, 0x39, 0x3a, 0x3b, 0x3c, 0x3d, 0x3e,
/*  64 */ 0x3f, 0x40, 0x41, 0x42, 0x43, 0x53, 0x47, 0x5f, 0x60, 0x61, 0x56, 0x5c, 0x5d, 0x5e, 0x57, 0x59,
/*  80 */ 0x5a, 0x5b, 0x62, 0x63, 0,    0,    0x64, 0x44, 0x45, 0,    0,    0,    0,    0,    0,    0,
/*  96 */ 0x58, 0xE4, 0x54, 0x46, 0xE6, 0,    0x4a, 0x52, 0x4b, 0x50, 0x4f, 0x4d, 0x51, 0x4e, 0x49, 0x4c,
/* 112 */ 0,    0x7f, 0x81, 0x80, 0x66, 0x67, 0xD7, 0x48, 0,    0x85, 0x90, 0x91, 0x89, 0xE3, 0xE7, 0x65,
];

/// Set-1 make code -> usage. `e0` selects the extended (E0-prefixed) table.
pub fn usage_from_set1(sc: u8, e0: bool) -> Option<Usage> {
    if e0 {
        return Some(match sc {
            0x1d => U_RCTRL,
            0x38 => U_RALT,
            0x5b => U_LGUI,
            0x5c => U_RGUI,
            0x47 => 0x4a, // Home
            0x48 => 0x52, // Up
            0x49 => 0x4b, // PgUp
            0x4b => 0x50, // Left
            0x4d => 0x4f, // Right
            0x4f => 0x4d, // End
            0x50 => 0x51, // Down
            0x51 => 0x4e, // PgDn
            0x52 => 0x49, // Insert
            0x53 => 0x4c, // Delete
            0x35 => 0x54, // keypad /
            0x1c => 0x58, // keypad Enter
            _ => return None,
        });
    }
    let u = *SET1_TO_USAGE.get(sc as usize)?;
    if u == 0 {
        None
    } else {
        Some(u)
    }
}

/// Set-2 make code -> usage. `e0` selects the extended table.
pub fn usage_from_set2(sc: u8, e0: bool) -> Option<Usage> {
    if e0 {
        return Some(match sc {
            0x14 => U_RCTRL,
            0x11 => U_RALT,
            0x1f => U_LGUI,
            0x27 => U_RGUI,
            0x75 => 0x52, // Up
            0x72 => 0x51, // Down
            0x6b => 0x50, // Left
            0x74 => 0x4f, // Right
            0x6c => 0x4a, // Home
            0x69 => 0x4d, // End
            0x7d => 0x4b, // PgUp
            0x7a => 0x4e, // PgDn
            0x70 => 0x49, // Insert
            0x71 => 0x4c, // Delete
            0x4a => 0x54, // keypad /
            0x5a => 0x58, // keypad Enter
            _ => return None,
        });
    }
    let u = *SET2_TO_USAGE.get(sc as usize)?;
    if u == 0 {
        None
    } else {
        Some(u)
    }
}

/// Linux evdev keycode -> usage.
pub fn usage_from_evdev(code: u16) -> Option<Usage> {
    let u = *EVDEV_TO_USAGE.get(code as usize)?;
    if u == 0 {
        None
    } else {
        Some(u)
    }
}

// ---------------------------------------------------------------------------
// Translation state
// ---------------------------------------------------------------------------

/// Everything that persists between keystrokes.
///
/// One instance, here, rather than one per driver: a `´` typed on a USB keyboard
/// and an `e` typed in a virtio-input window must still compose to `é`, and
/// caps-lock must survive switching between them.
#[derive(Default, Clone)]
pub struct State {
    /// A diacritic awaiting its base character.
    pub dead: Option<Dead>,
    /// Compose sequence in progress: the first key, awaiting the second.
    pub compose: Option<char>,
    /// Whether the Compose key itself is armed (no keys typed yet).
    pub compose_armed: bool,
    /// `Ctrl+Shift+U` hex entry: the value accumulated so far.
    pub hex: Option<u32>,
    /// Caps Lock, which is a *toggle* and so cannot live in per-press `Mods`.
    pub caps: bool,
}

/// What one key press produces.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct Emit {
    /// Bytes to hand to `console::read_byte`. UTF-8 for text, an ANSI CSI
    /// sequence for a nav key.
    pub bytes: String,
    /// A pre-edit to show at the caret (a pending dead key, or a hex value being
    /// typed). Empty when nothing is in flight.
    pub preedit: String,
    /// Set when a message should be shown rather than a character typed — an
    /// unknown compose sequence, or a hex value that is not a character.
    pub message: Option<&'static str>,
}

impl Emit {
    fn text(s: &str) -> Emit {
        Emit { bytes: String::from(s), ..Default::default() }
    }
    fn ch(c: char) -> Emit {
        let mut b = String::new();
        b.push(c);
        Emit { bytes: b, ..Default::default() }
    }
    fn nothing() -> Emit {
        Emit::default()
    }
    fn preedit(p: String) -> Emit {
        Emit { preedit: p, ..Default::default() }
    }
}

/// The ANSI sequence a layout-independent nav key produces.
///
/// One table, where there used to be four identical copies. Every input path
/// therefore speaks the same encoding, which is the property the shell, the
/// editor and `media_nav` all rely on.
fn nav_sequence(u: Usage, mods: Mods) -> Option<&'static str> {
    Some(match u {
        0x52 => "\x1b[A", // Up
        0x51 => "\x1b[B", // Down
        0x4f => "\x1b[C", // Right
        0x50 => "\x1b[D", // Left
        0x4a => "\x1b[H", // Home
        0x4d => "\x1b[F", // End
        0x4b => "\x1b[5~", // PgUp
        0x4e => "\x1b[6~", // PgDn
        0x4c => "\x1b[3~", // Delete
        // Function keys, in the VT220/xterm encodings.
        //
        // Standard forms on purpose: a serial terminal sends exactly these, so a
        // shortcut bound to F1 works from a host terminal and from the e2e harness
        // as well as from a physical keyboard. Before this, F-keys were decoded by
        // no driver and produced nothing at all on any transport.
        0x3a => "\x1b[11~", // F1
        0x3b => "\x1b[12~", // F2
        0x3c => "\x1b[13~", // F3
        0x3d => "\x1b[14~", // F4
        0x3e => "\x1b[15~", // F5
        0x3f => "\x1b[17~", // F6  (16 is skipped by the VT220 encoding)
        0x40 => "\x1b[18~", // F7
        0x41 => "\x1b[19~", // F8
        0x42 => "\x1b[20~", // F9
        0x43 => "\x1b[21~", // F10
        0x44 => "\x1b[23~", // F11 (22 is skipped)
        // F12 **and** Print Screen produce the same sequence. Print Screen is the
        // key a human reaches for to take a screenshot and has no terminal
        // encoding at all; F12 is the one a serial terminal can send. Folding them
        // into one code means the shortcut has a single handler and works from
        // both, rather than a physical-only binding nothing can test.
        0x45 | 0x46 => "\x1b[24~",
        // Ctrl+Tab / Ctrl+Shift+Tab: cycle pane focus.
        0x2b if mods.has(Mods::CTRL) && mods.has(Mods::SHIFT) => "\x1b[Z",
        0x2b if mods.has(Mods::CTRL) => "\x1b[T",
        // Cmd/Super+Space (or Ctrl+Space, since a macOS host usually eats
        // ⌘+Space for Spotlight first) opens the Agents browser.
        0x2c if mods.has(Mods::GUI) || mods.has(Mods::CTRL) => "\x1b[g",
        // **Chords for the two global shortcuts, because a Mac keyboard cannot
        // reach the F-key forms without help.** An Apple keyboard has no Print
        // Screen key at all, and sends the *consumer-page* brightness/volume
        // usage for a bare F1/F12 — ChittiOS reads only the boot-keyboard page,
        // so it sees nothing unless Fn is held. Under QEMU on macOS the window
        // server often eats those keys first as well.
        //
        // Both are bound on **Cmd and Ctrl**, which is the same belt-and-braces
        // the Agents browser above uses and for the same reason: the natural
        // chord is the one fingers know, and the Ctrl form is the one that
        // survives a host that steals it.
        //
        // `Cmd+/` (usage 0x38) for help — what Slack, GitHub and half the web
        // use for "show shortcuts", and not a macOS system shortcut.
        0x38 if mods.has(Mods::GUI) || mods.has(Mods::CTRL) => "\x1b[h",
        // `Cmd+Shift+3` (usage 0x20 = the digit 3) for screenshot — macOS's own
        // screenshot chord, so it needs no learning. NB macOS *does* take this
        // one system-wide, so inside a QEMU window the host will grab it and
        // `Ctrl+Shift+3` is the form that reaches the guest; on real hardware
        // (an m1n1 boot) there is no host to intervene and ⌘⇧3 works directly.
        0x20 if mods.has(Mods::SHIFT) && (mods.has(Mods::GUI) || mods.has(Mods::CTRL)) => {
            "\x1b[s"
        }
        // `Cmd+Shift+5` (usage 0x22 = the digit 5) toggles screen recording —
        // macOS's screenshot/recording toolbar key. Same Cmd/Ctrl twinning as
        // the screenshot chord above.
        0x22 if mods.has(Mods::SHIFT) && (mods.has(Mods::GUI) || mods.has(Mods::CTRL)) => {
            "\x1b[r"
        }
        _ => return None,
    })
}

/// The level a modifier state selects.
fn level_for(mods: Mods) -> Level {
    match (mods.altgr(), mods.has(Mods::SHIFT)) {
        (true, true) => Level::ShiftAltGr,
        (true, false) => Level::AltGr,
        (false, true) => Level::Shift,
        (false, false) => Level::Base,
    }
}

/// Translate one key event through a layout and the persistent state.
///
/// Pure with respect to globals: `/keyboard test` and every unit test drive this
/// directly. The precedence order below is not arbitrary — each step is a bug
/// that has happened in some OS:
///
/// 1. **Ctrl without Alt on a letter** becomes a control code, using the
///    *layout's* letter rather than the US position. So Ctrl+C on Dvorak is at
///    the physical `i` key, which is what Dvorak users expect and what XKB does.
/// 2. **AltGr** selects level 2/3.
/// 3. **Shift** selects level 1, then caps-lock flips letters only.
/// 4. Nav keys and the GUI/Ctrl chords produce their CSI sequences.
pub fn translate(st: &mut State, layout: &Layout, ev: KeyEvent) -> Emit {
    if !ev.pressed {
        return Emit::nothing();
    }
    let mut mods = ev.mods;
    if st.caps {
        mods.set(Mods::CAPS, true);
    }

    // Hex entry (Ctrl+Shift+U <hex...>) short-circuits everything: while it is
    // armed, the keys being typed are digits of a codepoint, not text.
    if st.hex.is_some() {
        return hex_step(st, layout, ev, mods);
    }

    // Layout-independent chords and nav keys, before the layout is consulted:
    // an arrow key has no character on any layout.
    if let Some(seq) = nav_sequence(ev.usage, mods) {
        // A pending dead key is dropped by a nav key rather than being applied to
        // whatever is typed after the cursor moves.
        st.dead = None;
        st.compose = None;
        st.compose_armed = false;
        return Emit::text(seq);
    }

    // Esc / Backspace clear a pending composition instead of reaching the line.
    if ev.usage == 0x29 && (st.dead.is_some() || st.compose_armed || st.compose.is_some()) {
        st.dead = None;
        st.compose = None;
        st.compose_armed = false;
        return Emit::nothing();
    }
    if ev.usage == 0x2a && st.dead.is_some() {
        // Backspace with a dead key pending clears the *dead key*, not the
        // previous character — otherwise a mistyped diacritic eats a letter.
        st.dead = None;
        return Emit::nothing();
    }

    // Ctrl+Shift+U arms hex entry.
    if ev.usage == 0x18 && mods.has(Mods::CTRL) && mods.has(Mods::SHIFT) {
        st.hex = Some(0);
        st.dead = None;
        return Emit::preedit(String::from("U+"));
    }

    let out = layout.lookup(ev.usage, level_for(mods));

    // Ctrl+letter -> control code, from the layout's own letter.
    if mods.has(Mods::CTRL) && !mods.has(Mods::ALT) {
        if let Out::Char(c) = layout.lookup(ev.usage, Level::Base) {
            if c.is_ascii_alphabetic() {
                let mut b = String::new();
                b.push((c.to_ascii_uppercase() as u8 & 0x1f) as char);
                st.dead = None;
                return Emit { bytes: b, ..Default::default() };
            }
        }
        // Ctrl with a non-letter and no layout meaning produces nothing, as
        // before — this is not the place to invent Ctrl+punctuation codes.
        if !matches!(out, Out::Char(_)) {
            return Emit::nothing();
        }
    }

    match out {
        Out::None => Emit::nothing(),
        Out::Dead(d) => dead_step(st, d),
        Out::Char(c) => {
            let c = apply_caps(c, mods, layout);
            // The Compose key is bound to right Alt on layouts that define no
            // AltGr levels, so a US user gets Compose for free and a German user
            // gets AltGr — and neither loses anything.
            if st.compose_armed || st.compose.is_some() {
                return compose_step(st, c);
            }
            if let Some(d) = st.dead.take() {
                return apply_dead(d, c);
            }
            Emit::ch(c)
        }
        Out::Compose => {
            st.compose_armed = true;
            st.compose = None;
            Emit::preedit(String::from("\u{25CC}")) // dotted circle: composing
        }
    }
}

/// Caps Lock flips *letters only*, so caps+shift is lowercase, as on a PC.
fn apply_caps(c: char, mods: Mods, layout: &Layout) -> char {
    if !mods.has(Mods::CAPS) || !layout.caps_affects_letters || !c.is_alphabetic() {
        return c;
    }
    // `to_uppercase`/`to_lowercase` return iterators because some characters
    // change length (German ß uppercases to SS). Only the single-char case is
    // taken: a multi-character form does not belong to one keypress, and on a
    // German layout Caps Lock genuinely leaves ß alone rather than typing SS.
    //
    // The two iterators are distinct types, so this cannot be one `if`
    // expression — hence the pair of arms.
    if c.is_lowercase() {
        let mut it = c.to_uppercase();
        match (it.next(), it.next()) {
            (Some(u), None) => u,
            _ => c,
        }
    } else {
        let mut it = c.to_lowercase();
        match (it.next(), it.next()) {
            (Some(u), None) => u,
            _ => c,
        }
    }
}

fn dead_step(st: &mut State, d: Dead) -> Emit {
    match st.dead.take() {
        // Double-tap: emit the spacing form of the diacritic.
        Some(prev) if prev == d => Emit::ch(layouts::spacing_form(d)),
        // A different diacritic: flush the first rather than dropping it, then
        // arm the second.
        Some(prev) => {
            st.dead = Some(d);
            Emit::ch(layouts::spacing_form(prev))
        }
        None => {
            st.dead = Some(d);
            let mut p = String::new();
            p.push(layouts::spacing_form(d));
            Emit::preedit(p)
        }
    }
}

fn apply_dead(d: Dead, base: char) -> Emit {
    if base == ' ' {
        return Emit::ch(layouts::spacing_form(d));
    }
    match layouts::compose_dead(d, base) {
        Some(c) => Emit::ch(c),
        // XKB's behaviour, and the house rule applied to text: emit the spacing
        // diacritic *followed by* the base, so `´` + `q` gives `´q` — visibly
        // wrong rather than silently a bare `q`.
        None => {
            let mut b = String::new();
            b.push(layouts::spacing_form(d));
            b.push(base);
            Emit { bytes: b, ..Default::default() }
        }
    }
}

fn compose_step(st: &mut State, c: char) -> Emit {
    if st.compose_armed {
        st.compose_armed = false;
        st.compose = Some(c);
        let mut p = String::new();
        p.push('\u{25CC}');
        p.push(c);
        return Emit::preedit(p);
    }
    let first = st.compose.take().unwrap_or(' ');
    match layouts::compose_pair(first, c) {
        Some(out) => Emit::ch(out),
        // Deliberately different from the dead-key fallback: a dead key is a
        // *layout* feature whose spacing fallback is standard behaviour a user
        // recognises, while Compose is a convenience — and silently typing `oc`
        // when the user asked for © is a mis-decode. `/keyboard compose` prints
        // the whole table, so the surface cannot lie about its coverage.
        None => Emit { message: Some("no such compose sequence"), ..Default::default() },
    }
}

fn hex_step(st: &mut State, layout: &Layout, ev: KeyEvent, mods: Mods) -> Emit {
    let v = st.hex.unwrap_or(0);
    // Esc cancels.
    if ev.usage == 0x29 {
        st.hex = None;
        return Emit::nothing();
    }
    // Backspace removes a digit, and cancels once empty.
    if ev.usage == 0x2a {
        if v == 0 {
            st.hex = None;
            return Emit::nothing();
        }
        st.hex = Some(v >> 4);
        return Emit::preedit(alloc::format!("U+{:x}", v >> 4));
    }
    let typed = match layout.lookup(ev.usage, level_for(mods)) {
        Out::Char(c) => c,
        _ => return Emit::preedit(alloc::format!("U+{v:x}")),
    };
    if let Some(d) = typed.to_digit(16) {
        // Above 0x10FFFF there is no point accumulating further.
        let next = v.saturating_mul(16).saturating_add(d);
        if next > 0x10_FFFF {
            st.hex = None;
            return Emit {
                message: Some("codepoint above U+10FFFF"),
                ..Default::default()
            };
        }
        st.hex = Some(next);
        return Emit::preedit(alloc::format!("U+{next:x}"));
    }
    // Anything else commits.
    st.hex = None;
    match char::from_u32(v) {
        Some(c) => Emit::ch(c),
        // Surrogates (D800..DFFF) and out-of-range values land here. Refused with
        // a reason rather than substituted with a replacement character.
        None => Emit { message: Some("not a character (surrogate or out of range)"), ..Default::default() },
    }
}

/// A human description of what is in flight, for `/keyboard` status.
pub fn pending_description(st: &State) -> Option<String> {
    if let Some(v) = st.hex {
        return Some(alloc::format!("hex U+{v:x}"));
    }
    if let Some(c) = st.compose {
        return Some(alloc::format!("compose {c}"));
    }
    if st.compose_armed {
        return Some(String::from("compose"));
    }
    st.dead.map(|d| alloc::format!("dead {}", layouts::dead_name(d)))
}

// ---------------------------------------------------------------------------
// The live path: an event ring, drained by `console::read_byte_raw`
// ---------------------------------------------------------------------------

const EVENT_RING: usize = 64;
const BYTE_RING: usize = 256;

/// Fixed-size ring of raw events, written by the drivers (including from IRQ1 on
/// x86) and drained on the console side.
struct EventRing {
    buf: [KeyEvent; EVENT_RING],
    head: usize,
    tail: usize,
}

const EMPTY_EVENT: KeyEvent =
    KeyEvent { usage: 0, mods: Mods(0), pressed: false, src: Source::UsbHid };

impl EventRing {
    const fn new() -> Self {
        EventRing { buf: [EMPTY_EVENT; EVENT_RING], head: 0, tail: 0 }
    }
    fn push(&mut self, ev: KeyEvent) {
        let next = (self.head + 1) % EVENT_RING;
        if next == self.tail {
            return; // full: drop, exactly as the old per-driver rings did
        }
        self.buf[self.head] = ev;
        self.head = next;
    }
    fn pop(&mut self) -> Option<KeyEvent> {
        if self.head == self.tail {
            return None;
        }
        let ev = self.buf[self.tail];
        self.tail = (self.tail + 1) % EVENT_RING;
        Some(ev)
    }
}

/// Output bytes waiting to be read. A single press can produce several
/// (`´`+`q` → two chars, a 4-byte emoji from hex entry), so there is no
/// per-press cap here at all — which is why `keyrepeat::SEQ_MAX` no longer has
/// to bound a keypress, only an escape sequence.
struct ByteRing {
    buf: [u8; BYTE_RING],
    head: usize,
    tail: usize,
}

impl ByteRing {
    const fn new() -> Self {
        ByteRing { buf: [0; BYTE_RING], head: 0, tail: 0 }
    }
    fn push(&mut self, b: u8) {
        let next = (self.head + 1) % BYTE_RING;
        if next == self.tail {
            return;
        }
        self.buf[self.head] = b;
        self.head = next;
    }
    fn pop(&mut self) -> Option<u8> {
        if self.head == self.tail {
            return None;
        }
        let b = self.buf[self.tail];
        self.tail = (self.tail + 1) % BYTE_RING;
        Some(b)
    }
}

/// Which HID usages are physically down, one bit per usage.
///
/// **Why this exists.** `translate` discards every release on its first line
/// (`if !ev.pressed { return Emit::nothing() }`) because the OS above it consumes a
/// *byte stream*, and a byte stream has no concept of a key coming back up. That is
/// right for a shell and an editor, and useless for a game: WASD movement needs to
/// know a key is still held, and `DG_GetKey(int* pressed, ...)` asks for the edge
/// directly.
///
/// The edges already arrive here — [`KeyEvent`] carries `pressed` — so this is a
/// *reader* of state the choke point already sees, not a second decoder. That
/// distinction is the whole reason `keymap/` exists: four independent decoders with
/// four copies of the modifier state is what it replaced, and adding a fifth to
/// serve games would undo it.
///
/// A bitmap rather than a list because `feed_event` runs inside IRQ1 on x86: this
/// allocates nothing, and a set/clear is two instructions.
///
/// It is **physical** state, deliberately unaffected by the layout, dead keys, the
/// IME or Caps Lock. A game wants "is the key left of S held", which is a position;
/// the character it would type is a different question, and `translate` answers
/// that one.
static HELD: [core::sync::atomic::AtomicU32; 8] = [
    core::sync::atomic::AtomicU32::new(0),
    core::sync::atomic::AtomicU32::new(0),
    core::sync::atomic::AtomicU32::new(0),
    core::sync::atomic::AtomicU32::new(0),
    core::sync::atomic::AtomicU32::new(0),
    core::sync::atomic::AtomicU32::new(0),
    core::sync::atomic::AtomicU32::new(0),
    core::sync::atomic::AtomicU32::new(0),
];

/// Record a transition in [`HELD`]. Pure with respect to everything else.
fn held_set(usage: Usage, down: bool) {
    let (w, b) = (usage as usize / 32, usage as usize % 32);
    // A `Usage` is a `u8`, so `w` is always < 8 and this cannot go out of range.
    let cell = &HELD[w];
    if down {
        cell.fetch_or(1 << b, Ordering::Relaxed);
    } else {
        cell.fetch_and(!(1 << b), Ordering::Relaxed);
    }
}

/// Whether the key at `usage` is physically down right now.
pub fn is_held(usage: Usage) -> bool {
    let (w, b) = (usage as usize / 32, usage as usize % 32);
    HELD[w].load(Ordering::Relaxed) & (1 << b) != 0
}

/// Every usage currently down, as a raw bitmap (`usage / 32` indexed, LSB-first).
///
/// Returned as a snapshot so a caller sees one coherent instant rather than a
/// bitmap that shifts under it mid-iteration.
pub fn held_snapshot() -> [u32; 8] {
    core::array::from_fn(|i| HELD[i].load(Ordering::Relaxed))
}

/// Forget every held key.
///
/// Called when input stops being delivered to whoever was reading edges — losing
/// focus, or a game tab closing. Without it a key held at the moment focus moved
/// stays down forever from the game's point of view, and Doom walks into a wall
/// until that key happens to be pressed and released again. The same reason a
/// windowing system synthesises key-up on focus loss.
pub fn clear_held() {
    for c in &HELD {
        c.store(0, Ordering::Relaxed);
    }
}

static EVENTS: Locked<EventRing> = Locked::new(EventRing::new());
static BYTES: Locked<ByteRing> = Locked::new(ByteRing::new());
static STATE: Locked<State> = Locked::new(State {
    dead: None,
    compose: None,
    compose_armed: false,
    hex: None,
    caps: false,
});
/// Software typematic for the one transport that reports only press edges.
static REPEAT: Locked<crate::keyrepeat::Typematic> =
    Locked::new(crate::keyrepeat::Typematic::new());
/// The active layout, by index into [`layouts::LAYOUTS`].
static ACTIVE: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(0);

/// Push one normalized event. **Safe to call from an interrupt handler**: it
/// allocates nothing and takes only the event-ring lock. Translation happens on
/// the drain side ([`next_byte`]).
pub fn feed_event(ev: KeyEvent) {
    // **Modifiers come from `mods`, not only from usages.** A USB boot keyboard
    // reports Ctrl/Shift/Alt as a *bitmask byte*, never as entries in the key
    // array — so a driver reading that byte fills in `ev.mods` and emits no
    // `KeyEvent` whose usage is `U_LCTRL`. Tracking usages alone therefore left
    // every modifier permanently "not held", which is invisible until something
    // asks: Doom's fire button is Ctrl, and it simply never fired.
    //
    // Derived on *every* event so a modifier released while another key is still
    // down is seen. A boot report ORs left and right together, so the side is not
    // recoverable and both are set — "is Ctrl down" is the only question anyone
    // asks, and nothing here needs to tell the two apart.
    for (bit, l, r) in [
        (Mods::CTRL, U_LCTRL, U_RCTRL),
        (Mods::SHIFT, U_LSHIFT, U_RSHIFT),
        (Mods::ALT, U_LALT, U_RALT),
    ] {
        let down = ev.mods.has(bit);
        held_set(l, down);
        held_set(r, down);
    }

    // The event's own usage is applied **last**, so a transport that really does
    // deliver modifier keys as usages (evdev, PS/2) wins over the derivation
    // above. Ordered the other way, the loop clears the very modifier this event
    // is reporting the press of — which is exactly what the first version did.
    //
    // Caps Lock returns early below and the modifier usages never reach
    // `translate` as characters, so tracking here rather than lower down is what
    // makes *every* key observable — including the ones a byte stream cannot
    // express.
    // `usage: 0` is "no key" — a modifier-only report refreshing the derivation
    // above. It has no held bit and no character, so it stops here.
    if ev.usage == 0 {
        return;
    }
    held_set(ev.usage, ev.pressed);

    // Caps Lock is a toggle whose state must outlive any one driver, so it is
    // applied here rather than in whichever driver saw the press.
    if ev.usage == U_CAPSLOCK {
        if ev.pressed {
            STATE.with(|s| s.caps = !s.caps);
        }
        return;
    }
    EVENTS.with(|r| r.push(ev));
}

/// The active layout.
pub fn active_layout() -> &'static Layout {
    let i = ACTIVE.load(core::sync::atomic::Ordering::Relaxed);
    layouts::LAYOUTS.get(i).unwrap_or(&layouts::US)
}

/// Select a layout by id (`us`, `de`, `dvorak`, …). Returns false if unknown.
pub fn set_layout(id: &str) -> bool {
    match layouts::LAYOUTS.iter().position(|l| l.id == id) {
        Some(i) => {
            ACTIVE.store(i, core::sync::atomic::Ordering::Relaxed);
            // A layout switch drops any composition in flight: the pending dead
            // key belonged to the old layout's key positions.
            STATE.with(|s| {
                s.dead = None;
                s.compose = None;
                s.compose_armed = false;
                s.hex = None;
            });
            true
        }
        None => false,
    }
}

/// A snapshot of the persistent state, for `/keyboard` status.
pub fn state_snapshot() -> State {
    STATE.with(|s| s.clone())
}

/// Drain one byte of translated input.
///
/// Runs in task context, so this is where `translate` (which allocates) belongs.
/// Also drives the software repeat for [`Source::UsbHid`].
pub fn next_byte() -> Option<u8> {
    // Anything already translated comes out first, in order.
    if let Some(b) = BYTES.with(|r| r.pop()) {
        return Some(b);
    }
    let now = crate::arch::now_ms();
    let ev = match EVENTS.with(|r| r.pop()) {
        Some(ev) => {
            // Arm or disarm the software repeat. Only for sources that do not
            // repeat themselves — stacking ours on hardware typematic doubles
            // every held key.
            if !ev.src.repeats_in_hardware() {
                REPEAT.with(|t| {
                    if ev.pressed {
                        t.press_event(ev, now);
                    } else {
                        t.release();
                    }
                });
            }
            ev
        }
        None => {
            // No new event: let a held key repeat.
            match REPEAT.with(|t| t.poll_event(now)) {
                Some(ev) => ev,
                None => return None,
            }
        }
    };
    let emit = {
        let layout = active_layout();
        STATE.with(|s| translate(s, layout, ev))
    };
    // The pre-edit is presentation, and the composer owns it. Reported through a
    // separate channel so the byte stream stays a byte stream.
    set_preedit(&emit.preedit, emit.message);
    if emit.bytes.is_empty() {
        return None;
    }
    let mut out = None;
    BYTES.with(|r| {
        for (i, b) in emit.bytes.bytes().enumerate() {
            if i == 0 {
                out = Some(b);
            } else {
                r.push(b);
            }
        }
    });
    out
}

/// The pre-edit run and any refusal message, for the composer to paint.
static PREEDIT: Locked<String> = Locked::new(String::new());
static MESSAGE: Locked<Option<&'static str>> = Locked::new(None);

fn set_preedit(p: &str, msg: Option<&'static str>) {
    PREEDIT.with(|s| {
        if s.as_str() != p {
            s.clear();
            s.push_str(p);
        }
    });
    if msg.is_some() {
        MESSAGE.with(|m| *m = msg);
    }
}

/// The current pre-edit run (empty when nothing is composing).
pub fn preedit() -> String {
    PREEDIT.with(|s| s.clone())
}

/// Take any pending refusal message ("no such compose sequence", …).
pub fn take_message() -> Option<&'static str> {
    MESSAGE.with(|m| m.take())
}

#[cfg(test)]
mod held_tests {
    use super::*;

    fn ev(usage: Usage, pressed: bool) -> KeyEvent {
        KeyEvent { usage, mods: Mods(0), pressed, src: Source::UsbHid }
    }

    /// The property the byte stream cannot express: a key stays held between its
    /// press and its release, so a game can ask "is W down" rather than having to
    /// infer it from repeat arrivals.
    #[test_case]
    fn a_key_is_held_between_press_and_release() {
        clear_held();
        let w = 0x1a; // HID usage for the physical `w` position
        assert!(!is_held(w));
        feed_event(ev(w, true));
        assert!(is_held(w), "a pressed key must read as held");
        feed_event(ev(w, false));
        assert!(!is_held(w), "a released key must not read as held");
        clear_held();
    }

    /// Simultaneous keys, which is the case software typematic structurally cannot
    /// do: `Typematic::press` *replaces* the held key, so W+A for a diagonal is not
    /// merely awkward through the byte stream, it is impossible.
    #[test_case]
    fn several_keys_are_held_at_once() {
        clear_held();
        let (w, a, s, d) = (0x1a, 0x04, 0x16, 0x07);
        for k in [w, a, s, d] {
            feed_event(ev(k, true));
        }
        assert!(is_held(w) && is_held(a) && is_held(s) && is_held(d));
        // Releasing one must not disturb the others.
        feed_event(ev(a, false));
        assert!(is_held(w) && !is_held(a) && is_held(s) && is_held(d));
        clear_held();
    }

    /// Every usage must land in its own bit. Note this also pins the ordering
    /// rule: a modifier pressed *as a usage* with no matching `mods` bit — what
    /// evdev and PS/2 deliver — must stay held, so the event's own usage is
    /// applied after the `mods` derivation rather than before it. A `usage / 32` split with the wrong
    /// shift would alias distinct keys onto one bit — and the symptom would be two
    /// unrelated keys appearing to be the same key, which reads as a stuck input
    /// rather than as an indexing bug.
    #[test_case]
    fn every_usage_gets_its_own_bit() {
        clear_held();
        // Usage 0 is "no key": a modifier-only report uses it to refresh the
        // derivation, so it must claim **no** held bit. Asserted rather than
        // skipped, because a stray bit there would be permanently stuck on.
        clear_held();
        feed_event(ev(0, true));
        assert_eq!(held_snapshot(), [0; 8], "usage 0 must never take a held bit");

        for u in 1..=255u16 {
            let u = u as Usage;
            feed_event(ev(u, true));
            assert!(is_held(u), "usage {u:#04x} did not set");
            // Nothing else may have been disturbed.
            let set = held_snapshot().iter().map(|w| w.count_ones()).sum::<u32>();
            assert_eq!(set, 1, "usage {u:#04x} set {set} bits, not 1");
            feed_event(ev(u, false));
            assert_eq!(held_snapshot(), [0; 8], "usage {u:#04x} did not clear");
        }
        clear_held();
    }

    /// Caps Lock returns early in `feed_event` (it is a toggle, not a character),
    /// and the modifiers never reach `translate` as text — so both are exactly the
    /// keys a byte-stream reader cannot see, and both must still be observable
    /// here. Ctrl and Shift are held modifiers in most games.
    #[test_case]
    fn modifiers_and_caps_lock_are_still_tracked() {
        clear_held();
        for k in [U_LCTRL, U_LSHIFT, U_RALT, U_CAPSLOCK] {
            feed_event(ev(k, true));
            assert!(is_held(k), "modifier {k:#04x} must be observable");
            feed_event(ev(k, false));
            assert!(!is_held(k));
        }
        clear_held();
    }

    /// **The fire-button bug.** A USB boot keyboard reports Ctrl/Shift/Alt as a
    /// bitmask byte, never as entries in the key array — so the driver fills in
    /// `mods` and emits no event whose *usage* is `U_LCTRL`. Tracking usages alone
    /// left every modifier permanently unheld, and Doom's fire button is Ctrl, so
    /// it simply never fired. The earlier test passed only because it synthesised
    /// usage `U_LCTRL` directly, which no real boot keyboard ever sends.
    #[test_case]
    fn a_modifier_reported_only_in_mods_still_reads_as_held() {
        clear_held();
        let w = 0x1a;
        // What a boot keyboard actually delivers: the letter, with CTRL in mods.
        feed_event(KeyEvent { usage: w, mods: Mods(Mods::CTRL), pressed: true, src: Source::UsbHid });
        assert!(is_held(U_LCTRL), "Ctrl must read as held from the mods bits alone");
        assert!(is_held(w));
        // Releasing the modifier while the letter is still down must be seen.
        // (The event's usage is the letter, so the derivation is what decides.)
        feed_event(KeyEvent { usage: w, mods: Mods(0), pressed: true, src: Source::UsbHid });
        assert!(!is_held(U_LCTRL), "a dropped mods bit must clear the held state");
        assert!(is_held(w), "the letter is still down");
        clear_held();
    }

    /// Shift and Alt take the same path, and a mods bit sets both sides: a boot
    /// report ORs left and right together, so the distinction is not recoverable
    /// and a game asking "is Shift down" must still get yes.
    #[test_case]
    fn shift_and_alt_come_through_mods_on_both_sides() {
        clear_held();
        feed_event(KeyEvent { usage: 0x04, mods: Mods(Mods::SHIFT | Mods::ALT), pressed: true, src: Source::UsbHid });
        for k in [U_LSHIFT, U_RSHIFT, U_LALT, U_RALT] {
            assert!(is_held(k), "{k:#04x} must be held");
        }
        for k in [U_LCTRL, U_RCTRL] {
            assert!(!is_held(k), "{k:#04x} must not be held");
        }
        clear_held();
    }

    /// A repeated press with no intervening release (hardware typematic on PS/2 and
    /// virtio-input does exactly this) must leave the key held, not toggle it.
    #[test_case]
    fn a_repeated_press_does_not_toggle() {
        clear_held();
        let w = 0x1a;
        for _ in 0..5 {
            feed_event(ev(w, true));
            assert!(is_held(w));
        }
        feed_event(ev(w, false));
        assert!(!is_held(w));
        clear_held();
    }

    /// Focus loss must forget everything, or a key held as focus moved stays down
    /// forever and the game walks into a wall until that key is pressed again.
    #[test_case]
    fn clear_held_forgets_everything() {
        clear_held();
        for k in [0x1a, 0x04, U_LSHIFT] {
            feed_event(ev(k, true));
        }
        assert!(held_snapshot().iter().any(|&w| w != 0));
        clear_held();
        assert_eq!(held_snapshot(), [0; 8], "focus loss must release every key");
    }

    /// Tracking must not change what the byte stream produces — every existing
    /// consumer (shell, editor, modals, e2e) has to be unaffected.
    #[test_case]
    fn tracking_does_not_disturb_translation() {
        clear_held();
        let mut st = State::default();
        let layout = &layouts::US;
        // HID usage 0x1a is the physical `w` position.
        let down = translate(&mut st, layout, ev(0x1a, true));
        assert_eq!(down.bytes, "w");
        // A release still emits nothing, which is what keeps `read_byte` a byte
        // stream rather than an event stream.
        let up = translate(&mut st, layout, ev(0x1a, false));
        assert!(up.bytes.is_empty(), "a release must still emit no bytes");
        clear_held();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(u: Usage, bits: u8) -> KeyEvent {
        KeyEvent { usage: u, mods: Mods(bits), pressed: true, src: Source::UsbHid }
    }

    fn tr(layout: &Layout, u: Usage, bits: u8) -> String {
        let mut st = State::default();
        translate(&mut st, layout, ev(u, bits)).bytes
    }

    // --- cross-tables -----------------------------------------------------

    /// The three transports must agree about which physical key is which, or a
    /// layout is correct on one keyboard and wrong on another.
    #[test_case]
    fn set1_set2_and_evdev_agree_on_every_graphic_key() {
        // (set-1, set-2, evdev) triples for the whole main block.
        const T: &[(u8, u8, u16)] = &[
            (0x01, 0x76, 1),   // Esc
            (0x02, 0x16, 2),   // 1
            (0x0b, 0x45, 11),  // 0
            (0x0c, 0x4e, 12),  // -
            (0x0d, 0x55, 13),  // =
            (0x0e, 0x66, 14),  // Backspace
            (0x0f, 0x0d, 15),  // Tab
            (0x10, 0x15, 16),  // q
            (0x19, 0x4d, 25),  // p
            (0x1a, 0x54, 26),  // [
            (0x1b, 0x5b, 27),  // ]
            (0x1c, 0x5a, 28),  // Enter
            (0x1e, 0x1c, 30),  // a
            (0x26, 0x4b, 38),  // l
            (0x27, 0x4c, 39),  // ;
            (0x28, 0x52, 40),  // '
            (0x29, 0x0e, 41),  // `
            (0x2b, 0x5d, 43),  // backslash
            (0x2c, 0x1a, 44),  // z
            (0x32, 0x3a, 50),  // m
            (0x33, 0x41, 51),  // ,
            (0x34, 0x49, 52),  // .
            (0x35, 0x4a, 53),  // /
            (0x39, 0x29, 57),  // Space
            (0x56, 0x61, 86),  // the ISO key left of Y
        ];
        for &(s1, s2, ev) in T {
            let a = usage_from_set1(s1, false);
            let b = usage_from_set2(s2, false);
            let c = usage_from_evdev(ev);
            assert!(a.is_some(), "set-1 {s1:#04x} decoded to nothing");
            assert_eq!(a, b, "set-1 {s1:#04x} vs set-2 {s2:#04x}");
            assert_eq!(a, c, "set-1 {s1:#04x} vs evdev {ev}");
        }
    }

    #[test_case]
    fn the_three_transports_agree_on_modifiers_and_nav_keys() {
        // Modifiers, unprefixed.
        for &(s1, s2, ev, want) in &[
            (0x1d, 0x14u8, 29u16, U_LCTRL),
            (0x2a, 0x12, 42, U_LSHIFT),
            (0x36, 0x59, 54, U_RSHIFT),
            (0x38, 0x11, 56, U_LALT),
            (0x3a, 0x58, 58, U_CAPSLOCK),
        ] {
            assert_eq!(usage_from_set1(s1, false), Some(want), "set-1 {s1:#04x}");
            assert_eq!(usage_from_set2(s2, false), Some(want), "set-2 {s2:#04x}");
            assert_eq!(usage_from_evdev(ev), Some(want), "evdev {ev}");
        }
        // Extended (E0-prefixed) on PS/2; plain on evdev.
        for &(s1, s2, ev, want) in &[
            (0x1d, 0x14u8, 97u16, U_RCTRL),
            (0x38, 0x11, 100, U_RALT),
            (0x48, 0x75, 103, 0x52), // Up
            (0x50, 0x72, 108, 0x51), // Down
            (0x4b, 0x6b, 105, 0x50), // Left
            (0x4d, 0x74, 106, 0x4f), // Right
            (0x47, 0x6c, 102, 0x4a), // Home
            (0x4f, 0x69, 107, 0x4d), // End
            (0x49, 0x7d, 104, 0x4b), // PgUp
            (0x51, 0x7a, 109, 0x4e), // PgDn
            (0x53, 0x71, 111, 0x4c), // Delete
            (0x5b, 0x1f, 125, U_LGUI),
        ] {
            assert_eq!(usage_from_set1(s1, true), Some(want), "E0 set-1 {s1:#04x}");
            assert_eq!(usage_from_set2(s2, true), Some(want), "E0 set-2 {s2:#04x}");
            assert_eq!(usage_from_evdev(ev), Some(want), "evdev {ev}");
        }
    }

    #[test_case]
    fn an_unmapped_scancode_is_none_not_a_wrong_key() {
        assert_eq!(usage_from_set1(0xff, false), None);
        assert_eq!(usage_from_set2(0xff, false), None);
        assert_eq!(usage_from_evdev(9999), None);
        assert_eq!(usage_from_set1(0x00, false), None);
        assert_eq!(usage_from_set1(0x77, true), None, "an unknown E0 code");
    }

    // --- the migration gate ----------------------------------------------

    /// **The gate that makes replacing four decoders safe.** Every ASCII byte the
    /// old per-driver decoders produced for US layout must come out of the new
    /// pipeline unchanged, for every scancode and every modifier combination.
    ///
    /// The reference tables are copied verbatim from the four drivers they
    /// replaced; two of those drivers are `cfg`'d out of this build, so this is
    /// the only place their behaviour can be pinned at all.
    #[test_case]
    fn us_layout_reproduces_the_legacy_set1_decoder() {
        // Set-1 make codes 0x00..0x40, unshifted then shifted (verbatim from the
        // deleted `arch::x86_64::keyboard::{UNSHIFTED, SHIFTED}`).
        #[rustfmt::skip]
        const UNSHIFTED: [u8; 0x40] = [
            0,    0x1b, b'1', b'2', b'3', b'4', b'5', b'6', b'7', b'8', b'9', b'0', b'-', b'=', 0x08, b'\t',
            b'q', b'w', b'e', b'r', b't', b'y', b'u', b'i', b'o', b'p', b'[', b']', b'\n', 0,  b'a', b's',
            b'd', b'f', b'g', b'h', b'j', b'k', b'l', b';', b'\'', b'`', 0,  b'\\', b'z', b'x', b'c', b'v',
            b'b', b'n', b'm', b',', b'.', b'/', 0,    b'*', 0,    b' ', 0,    0,    0,    0,    0,   0,
        ];
        #[rustfmt::skip]
        const SHIFTED: [u8; 0x40] = [
            0,    0x1b, b'!', b'@', b'#', b'$', b'%', b'^', b'&', b'*', b'(', b')', b'_', b'+', 0x08, b'\t',
            b'Q', b'W', b'E', b'R', b'T', b'Y', b'U', b'I', b'O', b'P', b'{', b'}', b'\n', 0,  b'A', b'S',
            b'D', b'F', b'G', b'H', b'J', b'K', b'L', b':', b'"',  b'~', 0,  b'|',  b'Z', b'X', b'C', b'V',
            b'B', b'N', b'M', b'<', b'>', b'?', 0,    b'*', 0,    b' ', 0,    0,    0,    0,    0,   0,
        ];
        let us = &layouts::US;
        for sc in 0..0x40u8 {
            let Some(u) = usage_from_set1(sc, false) else {
                // Every code the old table gave a character for must map to a
                // usage; the reverse is fine (the old table had holes we fill).
                assert_eq!(UNSHIFTED[sc as usize], 0, "set-1 {sc:#04x} lost its character");
                continue;
            };
            if modifier_bit(u).is_some() || u == U_CAPSLOCK {
                continue;
            }
            for (shift, table) in [(false, &UNSHIFTED), (true, &SHIFTED)] {
                let want = table[sc as usize];
                if want == 0 {
                    continue;
                }
                let bits = if shift { Mods::SHIFT } else { 0 };
                let got = tr(us, u, bits);
                // The old decoder emitted `\n` for Enter; the new one emits `\r`
                // (what every other path already sent, and what the shell reads).
                let want = if want == b'\n' { b'\r' } else { want };
                assert_eq!(
                    got.as_bytes(),
                    &[want],
                    "set-1 {sc:#04x} usage {u:#04x} shift={shift}: want {want:?}, got {got:?}"
                );
            }
        }
    }

    #[test_case]
    fn us_layout_reproduces_the_legacy_evdev_decoder() {
        // Verbatim from the deleted `virtio_input::keycode_to_ascii` MAP.
        #[rustfmt::skip]
        const MAP: &[(u16, u8, u8)] = &[
            (1, 0x1b, 0x1b),
            (2, b'1', b'!'), (3, b'2', b'@'), (4, b'3', b'#'), (5, b'4', b'$'), (6, b'5', b'%'),
            (7, b'6', b'^'), (8, b'7', b'&'), (9, b'8', b'*'), (10, b'9', b'('), (11, b'0', b')'),
            (12, b'-', b'_'), (13, b'=', b'+'), (14, 0x08, 0x08), (15, b'\t', b'\t'),
            (16, b'q', b'Q'), (17, b'w', b'W'), (18, b'e', b'E'), (19, b'r', b'R'), (20, b't', b'T'),
            (21, b'y', b'Y'), (22, b'u', b'U'), (23, b'i', b'I'), (24, b'o', b'O'), (25, b'p', b'P'),
            (26, b'[', b'{'), (27, b']', b'}'), (28, b'\r', b'\r'),
            (30, b'a', b'A'), (31, b's', b'S'), (32, b'd', b'D'), (33, b'f', b'F'), (34, b'g', b'G'),
            (35, b'h', b'H'), (36, b'j', b'J'), (37, b'k', b'K'), (38, b'l', b'L'), (39, b';', b':'),
            (40, b'\'', b'"'), (41, b'`', b'~'),
            (43, b'\\', b'|'),
            (44, b'z', b'Z'), (45, b'x', b'X'), (46, b'c', b'C'), (47, b'v', b'V'), (48, b'b', b'B'),
            (49, b'n', b'N'), (50, b'm', b'M'), (51, b',', b'<'), (52, b'.', b'>'), (53, b'/', b'?'),
            (57, b' ', b' '),
        ];
        let us = &layouts::US;
        for &(code, base, shifted) in MAP {
            let u = usage_from_evdev(code).unwrap_or_else(|| panic!("evdev {code} lost"));
            assert_eq!(tr(us, u, 0).as_bytes(), &[base], "evdev {code} base");
            assert_eq!(tr(us, u, Mods::SHIFT).as_bytes(), &[shifted], "evdev {code} shift");
        }
    }

    #[test_case]
    fn us_layout_reproduces_the_legacy_hid_decoder() {
        // Verbatim from the deleted `xhci::hid_to_ascii` arms.
        let us = &layouts::US;
        for u in 0x04..=0x1du8 {
            assert_eq!(tr(us, u, 0).as_bytes(), &[b'a' + (u - 0x04)]);
            assert_eq!(tr(us, u, Mods::SHIFT).as_bytes(), &[b'A' + (u - 0x04)]);
        }
        for u in 0x1e..=0x26u8 {
            assert_eq!(tr(us, u, 0).as_bytes(), &[b'1' + (u - 0x1e)]);
        }
        assert_eq!(tr(us, 0x27, 0), "0");
        assert_eq!(tr(us, 0x28, 0), "\r", "Enter");
        assert_eq!(tr(us, 0x29, 0).as_bytes(), &[0x1b], "Esc");
        assert_eq!(tr(us, 0x2a, 0).as_bytes(), &[0x08], "Backspace");
        assert_eq!(tr(us, 0x2b, 0), "\t", "Tab");
        assert_eq!(tr(us, 0x2c, 0), " ", "Space");
        for &(u, base, shifted) in &[
            (0x2du8, '-', '_'),
            (0x2e, '=', '+'),
            (0x2f, '[', '{'),
            (0x30, ']', '}'),
            (0x31, '\\', '|'),
            (0x33, ';', ':'),
            (0x34, '\'', '"'),
            (0x35, '`', '~'),
            (0x36, ',', '<'),
            (0x37, '.', '>'),
            (0x38, '/', '?'),
        ] {
            assert_eq!(tr(us, u, 0).chars().next(), Some(base), "usage {u:#04x} base");
            assert_eq!(tr(us, u, Mods::SHIFT).chars().next(), Some(shifted), "usage {u:#04x} shift");
        }
    }

    #[test_case]
    fn caps_lock_flips_letters_only_and_shift_plus_caps_is_lowercase() {
        let us = &layouts::US;
        let mut st = State { caps: true, ..Default::default() };
        // 'a' with caps → 'A'
        assert_eq!(translate(&mut st, us, ev(0x04, 0)).bytes, "A");
        // caps + shift → lowercase, as on a PC.
        assert_eq!(translate(&mut st, us, ev(0x04, Mods::SHIFT)).bytes, "a");
        // Digits are unaffected by caps.
        assert_eq!(translate(&mut st, us, ev(0x1e, 0)).bytes, "1");
        assert_eq!(translate(&mut st, us, ev(0x1e, Mods::SHIFT)).bytes, "!");
    }

    #[test_case]
    fn caps_state_lives_here_so_it_survives_a_source_switch() {
        // The reason `caps` is in `State` and not in a driver: a keyboard on one
        // transport and a window on another must agree about it.
        let us = &layouts::US;
        let mut st = State::default();
        assert_eq!(translate(&mut st, us, ev(0x04, 0)).bytes, "a");
        st.caps = true;
        let from_other_transport =
            KeyEvent { usage: 0x04, mods: Mods(0), pressed: true, src: Source::Evdev };
        assert_eq!(translate(&mut st, us, from_other_transport).bytes, "A");
    }

    // --- modifiers --------------------------------------------------------

    #[test_case]
    fn ctrl_letter_becomes_a_control_code() {
        let us = &layouts::US;
        assert_eq!(tr(us, 0x06, Mods::CTRL).as_bytes(), &[0x03], "Ctrl+C");
        assert_eq!(tr(us, 0x07, Mods::CTRL).as_bytes(), &[0x04], "Ctrl+D");
        // Shift does not change a control code.
        assert_eq!(tr(us, 0x06, Mods::CTRL | Mods::SHIFT).as_bytes(), &[0x03]);
    }

    /// Ctrl+C must be where the layout puts `c`, not where US puts it.
    #[test_case]
    fn ctrl_letter_uses_the_layouts_letter_not_the_us_position() {
        let dv = layouts::LAYOUTS.iter().find(|l| l.id == "dvorak").expect("dvorak");
        // On Dvorak, `c` sits on the physical key US calls `i` (usage 0x0c).
        assert_eq!(tr(dv, 0x0c, 0), "c", "dvorak: US-i position types c");
        assert_eq!(tr(dv, 0x0c, Mods::CTRL).as_bytes(), &[0x03], "so Ctrl there is Ctrl+C");
        // And the US `c` key types `j` on Dvorak, so Ctrl there is Ctrl+J.
        assert_eq!(tr(dv, 0x06, 0), "j");
        assert_eq!(tr(dv, 0x06, Mods::CTRL).as_bytes(), &[0x0a]);
    }

    #[test_case]
    fn altgr_is_right_alt_or_ctrl_plus_alt_but_never_left_alt_alone() {
        assert!(Mods(Mods::ALTGR).altgr());
        assert!(Mods(Mods::CTRL | Mods::ALT).altgr(), "the XKB convention");
        assert!(!Mods(Mods::ALT).altgr(), "left Alt alone is a meta modifier");
        assert!(!Mods(Mods::CTRL).altgr());
        assert!(!Mods(0).altgr());
        // Level selection follows.
        assert!(matches!(level_for(Mods(0)), Level::Base));
        assert!(matches!(level_for(Mods(Mods::SHIFT)), Level::Shift));
        assert!(matches!(level_for(Mods(Mods::ALTGR)), Level::AltGr));
        assert!(matches!(
            level_for(Mods(Mods::ALTGR | Mods::SHIFT)),
            Level::ShiftAltGr
        ));
    }

    #[test_case]
    fn nav_keys_produce_one_shared_ansi_encoding() {
        let us = &layouts::US;
        for &(u, want) in &[
            (0x52u8, "\x1b[A"),
            (0x51, "\x1b[B"),
            (0x4f, "\x1b[C"),
            (0x50, "\x1b[D"),
            (0x4a, "\x1b[H"),
            (0x4d, "\x1b[F"),
            (0x4b, "\x1b[5~"),
            (0x4e, "\x1b[6~"),
            (0x4c, "\x1b[3~"),
        ] {
            assert_eq!(tr(us, u, 0), want, "usage {u:#04x}");
        }
        assert_eq!(tr(us, 0x2b, Mods::CTRL), "\x1b[T", "Ctrl+Tab");
        assert_eq!(tr(us, 0x2b, Mods::CTRL | Mods::SHIFT), "\x1b[Z");
        assert_eq!(tr(us, 0x2c, Mods::GUI), "\x1b[g", "Cmd+Space");
        assert_eq!(tr(us, 0x2c, Mods::CTRL), "\x1b[g", "Ctrl+Space");
        // Plain Tab and Space are still characters.
        assert_eq!(tr(us, 0x2b, 0), "\t");
        assert_eq!(tr(us, 0x2c, 0), " ");
    }

    /// The two global shortcuts must be reachable without Fn, because an Apple
    /// keyboard has no Print Screen key and sends media usages for bare F-keys.
    #[test_case]
    fn the_global_shortcuts_have_chords_that_need_no_function_key() {
        let us = &layouts::US;
        // Help: Cmd+/ and Ctrl+/.
        assert_eq!(tr(us, 0x38, Mods::GUI), "\x1b[h", "Cmd+/");
        assert_eq!(tr(us, 0x38, Mods::CTRL), "\x1b[h", "Ctrl+/");
        // Screenshot: Cmd+Shift+3 and Ctrl+Shift+3.
        assert_eq!(tr(us, 0x20, Mods::GUI | Mods::SHIFT), "\x1b[s", "Cmd+Shift+3");
        assert_eq!(tr(us, 0x20, Mods::CTRL | Mods::SHIFT), "\x1b[s", "Ctrl+Shift+3");
        // Record: Cmd+Shift+5 and Ctrl+Shift+5 (macOS capture toolbar key).
        assert_eq!(tr(us, 0x22, Mods::GUI | Mods::SHIFT), "\x1b[r", "Cmd+Shift+5");
        assert_eq!(tr(us, 0x22, Mods::CTRL | Mods::SHIFT), "\x1b[r", "Ctrl+Shift+5");

        // And the chords must not eat the ordinary characters they are built on.
        assert_eq!(tr(us, 0x38, 0), "/", "a bare slash is still a slash");
        assert_eq!(tr(us, 0x38, Mods::SHIFT), "?");
        assert_eq!(tr(us, 0x20, 0), "3");
        assert_eq!(tr(us, 0x20, Mods::SHIFT), "#", "Shift+3 is still a hash");
        // Cmd+3 without Shift is not the screenshot chord.
        assert_eq!(tr(us, 0x20, Mods::GUI), "3");
        assert_eq!(tr(us, 0x22, Mods::SHIFT), "%", "Shift+5 is still a percent");

        // They follow the *layout*, so on a layout where `/` sits elsewhere the
        // chord moves with it rather than staying at the US position.
        let fr = layouts::LAYOUTS.iter().find(|l| l.id == "fr").unwrap();
        // French puts `!` on the US `/` key and `:` on the US `.` key; the chord
        // is positional, which is what a keyboard shortcut should be.
        assert_eq!(tr(fr, 0x38, Mods::CTRL), "\x1b[h", "the chord is positional");
    }

    /// F-keys produced **nothing** on every transport before the keymap existed,
    /// which is why they were available to bind shortcuts to.
    #[test_case]
    fn function_keys_use_the_standard_vt220_encodings() {
        let us = &layouts::US;
        for &(u, want) in &[
            (0x3au8, "\x1b[11~"), // F1
            (0x3b, "\x1b[12~"),
            (0x3c, "\x1b[13~"),
            (0x3d, "\x1b[14~"),
            (0x3e, "\x1b[15~"),
            (0x3f, "\x1b[17~"), // F6 — the encoding skips 16
            (0x40, "\x1b[18~"),
            (0x41, "\x1b[19~"),
            (0x42, "\x1b[20~"),
            (0x43, "\x1b[21~"),
            (0x44, "\x1b[23~"), // F11 — and 22
            (0x45, "\x1b[24~"), // F12
        ] {
            assert_eq!(tr(us, u, 0), want, "usage {u:#04x}");
        }
        // Print Screen folds onto F12's code: it has no terminal encoding of its
        // own, so sharing one means the shortcut has a single handler and is
        // reachable from a serial console too.
        assert_eq!(tr(us, 0x46, 0), "\x1b[24~", "Print Screen");
        // They are layout-independent — an F-key has no character on any layout.
        for l in layouts::LAYOUTS {
            assert_eq!(tr(l, 0x3a, 0), "\x1b[11~", "F1 on layout '{}'", l.id);
        }
    }

    /// All three transports must decode the function-key block, or a shortcut
    /// bound to F1 works on one keyboard and not another.
    #[test_case]
    fn every_transport_decodes_the_function_keys_and_print_screen() {
        for &(s1, s2, ev, want) in &[
            (0x3bu8, 0x05u8, 59u16, 0x3au8), // F1
            (0x3c, 0x06, 60, 0x3b),          // F2
            (0x3d, 0x04, 61, 0x3c),          // F3
            (0x44, 0x09, 68, 0x43),          // F10
            (0x57, 0x78, 87, 0x44),          // F11
            (0x58, 0x07, 88, 0x45),          // F12
        ] {
            assert_eq!(usage_from_set1(s1, false), Some(want), "set-1 {s1:#04x}");
            assert_eq!(usage_from_set2(s2, false), Some(want), "set-2 {s2:#04x}");
            assert_eq!(usage_from_evdev(ev), Some(want), "evdev {ev}");
        }
        // Print Screen: evdev calls it SysRq (99). PS/2 sends it as a multi-byte
        // dance that this driver does not reassemble, which is exactly why the
        // shortcut also answers to F12.
        assert_eq!(usage_from_evdev(99), Some(0x46));
    }

    #[test_case]
    fn a_release_produces_nothing() {
        let us = &layouts::US;
        let mut st = State::default();
        let up = KeyEvent { usage: 0x04, mods: Mods(0), pressed: false, src: Source::UsbHid };
        assert_eq!(translate(&mut st, us, up), Emit::default());
    }

    // --- dead keys --------------------------------------------------------

    #[test_case]
    fn a_dead_key_composes_with_the_next_base_character() {
        let de = layouts::LAYOUTS.iter().find(|l| l.id == "de").expect("de");
        let mut st = State::default();
        // The German acute/grave key (US `=` position).
        let armed = translate(&mut st, de, ev(0x2e, 0));
        assert_eq!(armed.bytes, "", "arming a dead key types nothing");
        assert!(!armed.preedit.is_empty(), "but it shows a pre-edit");
        assert_eq!(translate(&mut st, de, ev(0x08, 0)).bytes, "é", "acute + e");
        assert!(st.dead.is_none(), "and the state is cleared");
    }

    #[test_case]
    fn a_dead_key_twice_types_its_spacing_form() {
        let de = layouts::LAYOUTS.iter().find(|l| l.id == "de").unwrap();
        let mut st = State::default();
        translate(&mut st, de, ev(0x2e, 0));
        assert_eq!(translate(&mut st, de, ev(0x2e, 0)).bytes, "\u{00B4}");
        assert!(st.dead.is_none());
    }

    #[test_case]
    fn a_dead_key_then_space_types_its_spacing_form() {
        let de = layouts::LAYOUTS.iter().find(|l| l.id == "de").unwrap();
        let mut st = State::default();
        translate(&mut st, de, ev(0x2e, 0));
        assert_eq!(translate(&mut st, de, ev(0x2c, 0)).bytes, "\u{00B4}");
    }

    #[test_case]
    fn a_dead_key_with_no_precomposed_form_types_both_visibly() {
        // XKB's behaviour, and the house rule: `´` + `q` gives `´q`, which looks
        // wrong, rather than a bare `q`, which looks right and is not.
        let de = layouts::LAYOUTS.iter().find(|l| l.id == "de").unwrap();
        let mut st = State::default();
        translate(&mut st, de, ev(0x2e, 0));
        assert_eq!(translate(&mut st, de, ev(0x14, 0)).bytes, "\u{00B4}q");
    }

    #[test_case]
    fn two_different_dead_keys_flush_the_first_rather_than_dropping_it() {
        let de = layouts::LAYOUTS.iter().find(|l| l.id == "de").unwrap();
        let mut st = State::default();
        translate(&mut st, de, ev(0x2e, 0)); // acute
        // The circumflex key (US backtick position).
        let out = translate(&mut st, de, ev(0x35, 0));
        assert_eq!(out.bytes, "\u{00B4}", "the first diacritic is emitted, not lost");
        assert_eq!(st.dead, Some(Dead::Circumflex), "and the second is armed");
    }

    #[test_case]
    fn esc_and_backspace_clear_a_pending_dead_key_without_eating_a_character() {
        let de = layouts::LAYOUTS.iter().find(|l| l.id == "de").unwrap();
        for clearer in [0x29u8 /* Esc */, 0x2a /* Backspace */] {
            let mut st = State::default();
            translate(&mut st, de, ev(0x2e, 0));
            let out = translate(&mut st, de, ev(clearer, 0));
            assert_eq!(out.bytes, "", "clearing must not send a byte to the line");
            assert!(st.dead.is_none(), "usage {clearer:#04x} did not clear the dead key");
        }
        // Without a pending dead key, Backspace is still a Backspace.
        let mut st = State::default();
        assert_eq!(translate(&mut st, de, ev(0x2a, 0)).as_bytes_len(), 1);
    }

    #[test_case]
    fn an_arrow_key_drops_a_pending_dead_key() {
        // Otherwise the diacritic is applied to whatever is typed after the
        // cursor has moved somewhere else entirely.
        let de = layouts::LAYOUTS.iter().find(|l| l.id == "de").unwrap();
        let mut st = State::default();
        translate(&mut st, de, ev(0x2e, 0));
        assert_eq!(translate(&mut st, de, ev(0x50, 0)).bytes, "\x1b[D");
        assert!(st.dead.is_none());
    }

    // --- compose + hex ----------------------------------------------------

    #[test_case]
    fn compose_two_keys_produce_one_character() {
        let us = &layouts::US;
        let mut st = State::default();
        // Right Alt is Compose on layouts with no AltGr levels.
        assert!(us.altgr_is_compose, "US must bind Compose to right Alt");
        st.compose_armed = true;
        translate(&mut st, us, ev(0x12, 0)); // 'o'
        assert_eq!(translate(&mut st, us, ev(0x06, 0)).bytes, "\u{00A9}", "o + c = ©");
    }

    #[test_case]
    fn an_unknown_compose_sequence_is_refused_and_types_nothing() {
        // Deliberately unlike the dead-key fallback: a dead key's spacing form is
        // standard behaviour a user recognises; typing `qq` because the user asked
        // for something Compose does not have is a mis-decode.
        let us = &layouts::US;
        let mut st = State::default();
        st.compose_armed = true;
        translate(&mut st, us, ev(0x14, 0)); // 'q'
        let out = translate(&mut st, us, ev(0x14, 0)); // 'q'
        assert_eq!(out.bytes, "", "nothing may be typed");
        assert!(out.message.is_some(), "and the refusal must say so");
    }

    #[test_case]
    fn hex_entry_commits_on_a_non_digit_and_cancels_on_esc() {
        let us = &layouts::US;
        let mut st = State::default();
        // Ctrl+Shift+U arms it.
        let armed = translate(&mut st, us, ev(0x18, Mods::CTRL | Mods::SHIFT));
        assert_eq!(st.hex, Some(0));
        assert_eq!(armed.preedit, "U+");
        // 0, 0, e, 9 → U+00E9 = é
        for u in [0x27u8, 0x27, 0x08, 0x26] {
            translate(&mut st, us, ev(u, 0));
        }
        assert_eq!(st.hex, Some(0xe9));
        // Enter (a non-hex key) commits.
        assert_eq!(translate(&mut st, us, ev(0x28, 0)).bytes, "é");
        assert!(st.hex.is_none());
        // Esc cancels without typing.
        translate(&mut st, us, ev(0x18, Mods::CTRL | Mods::SHIFT));
        translate(&mut st, us, ev(0x21, 0)); // '4'
        let out = translate(&mut st, us, ev(0x29, 0));
        assert_eq!(out.bytes, "");
        assert!(st.hex.is_none());
    }

    #[test_case]
    fn hex_entry_refuses_surrogates_and_out_of_range_values() {
        let us = &layouts::US;
        // U+D800 is a surrogate half: `char::from_u32` refuses it, and so must we
        // rather than substituting a replacement character.
        let mut st = State { hex: Some(0xd800), ..Default::default() };
        let out = translate(&mut st, us, ev(0x28, 0));
        assert_eq!(out.bytes, "");
        assert!(out.message.is_some());
        // Beyond U+10FFFF, accumulation stops rather than wrapping.
        let mut st = State { hex: Some(0x10_FFFF), ..Default::default() };
        let out = translate(&mut st, us, ev(0x1e, 0)); // '1' → 0x10FFFF1
        assert!(out.message.is_some(), "must refuse rather than silently truncate");
        assert!(st.hex.is_none());
        // The largest valid codepoint does commit.
        let mut st = State { hex: Some(0x10_FFFF), ..Default::default() };
        assert_eq!(translate(&mut st, us, ev(0x28, 0)).bytes, "\u{10FFFF}");
    }

    #[test_case]
    fn hex_entry_backspace_removes_a_digit_then_cancels() {
        let us = &layouts::US;
        let mut st = State { hex: Some(0xe9), ..Default::default() };
        translate(&mut st, us, ev(0x2a, 0));
        assert_eq!(st.hex, Some(0xe));
        translate(&mut st, us, ev(0x2a, 0));
        assert_eq!(st.hex, Some(0));
        translate(&mut st, us, ev(0x2a, 0));
        assert!(st.hex.is_none(), "backspacing past the start cancels");
    }

    #[test_case]
    fn pending_description_names_what_is_in_flight() {
        let mut st = State::default();
        assert!(pending_description(&st).is_none());
        st.dead = Some(Dead::Acute);
        assert!(pending_description(&st).unwrap().contains("acute"));
        st.dead = None;
        st.hex = Some(0xe9);
        assert!(pending_description(&st).unwrap().contains("e9"));
    }

    // Small helper so the Backspace assertion above reads clearly.
    impl Emit {
        fn as_bytes_len(&self) -> usize {
            self.bytes.len()
        }
    }
}
