//! **virtio-input keyboard** over the virtio-mmio transport, so the QEMU
//! display window (the ramfb TUI) can accept keystrokes on aarch64 — the
//! counterpart to the x86 PS/2 keyboard. The `virt` machine has no PS/2; window
//! key events are delivered by a `-device virtio-keyboard-device`, which QEMU
//! places on a virtio-mmio slot. This driver scans those slots, brings the
//! device up (virtio 1.0, version 2), fills the event virtqueue, and on each
//! poll drains key-press events, maps Linux keycodes → ASCII, and pushes them
//! into a ring the console drains (alongside serial).
//!
//! Polled, not interrupt-driven (the shell already polls `read_byte` in a
//! yielding loop). Queue memory is page-aligned heap on the identity map, so
//! its virtual address is the physical address the device is handed — the same
//! assumption the ramfb framebuffer uses, coherent under HVF and TCG.

use crate::mm::Locked;
use alloc::alloc::{alloc_zeroed, Layout};
use core::ptr::{read_volatile, write_volatile};
use core::sync::atomic::{compiler_fence, Ordering};

// --- virtio-mmio register offsets --------------------------------------
const MAGIC: usize = 0x000; // "virt" = 0x74726976
const VERSION: usize = 0x004;
const DEVICE_ID: usize = 0x008;
const DEVICE_FEATURES: usize = 0x010;
const DEVICE_FEATURES_SEL: usize = 0x014;
const DRIVER_FEATURES: usize = 0x020;
const DRIVER_FEATURES_SEL: usize = 0x024;
const GUEST_PAGE_SIZE: usize = 0x028; // legacy (v1) only
const QUEUE_SEL: usize = 0x030;
const QUEUE_NUM_MAX: usize = 0x034;
const QUEUE_NUM: usize = 0x038;
const QUEUE_ALIGN: usize = 0x03c; // legacy (v1) only
const QUEUE_PFN: usize = 0x040; // legacy (v1) only
const QUEUE_READY: usize = 0x044; // modern (v2) only
const QUEUE_NOTIFY: usize = 0x050;
const STATUS: usize = 0x070;
const QUEUE_DESC_LOW: usize = 0x080;
const QUEUE_DESC_HIGH: usize = 0x084;
const QUEUE_DRIVER_LOW: usize = 0x090;
const QUEUE_DRIVER_HIGH: usize = 0x094;
const QUEUE_DEVICE_LOW: usize = 0x0a0;
const QUEUE_DEVICE_HIGH: usize = 0x0a4;

// Status bits.
const S_ACK: u32 = 1;
const S_DRIVER: u32 = 2;
const S_DRIVER_OK: u32 = 4;
const S_FEATURES_OK: u32 = 8;

const VIRTIO_ID_INPUT: u32 = 18;
const VIRTQ_DESC_F_WRITE: u16 = 2;

// virtio-mmio scan window on QEMU `virt`.
const MMIO_BASE: usize = 0x0a00_0000;
const MMIO_STRIDE: usize = 0x200;
const MMIO_SLOTS: usize = 32;

const QSIZE: usize = 8; // event queue depth (small is fine for a keyboard)

// Linux input event types / codes.
const EV_KEY: u16 = 0x01;
const KEY_LEFTSHIFT: u16 = 42;
const KEY_RIGHTSHIFT: u16 = 54;
const KEY_LEFTCTRL: u16 = 29;

#[inline]
fn dsb() {
    unsafe { core::arch::asm!("dsb sy", options(nomem, nostack, preserves_flags)) };
}

unsafe fn reg_read(base: usize, off: usize) -> u32 {
    unsafe { read_volatile((base + off) as *const u32) }
}
unsafe fn reg_write(base: usize, off: usize, val: u32) {
    unsafe { write_volatile((base + off) as *mut u32, val) };
}

/// One virtio_input_event as the device writes it (little-endian).
#[repr(C)]
#[derive(Clone, Copy)]
struct InputEvent {
    typ: u16,
    code: u16,
    value: u32,
}

/// Page-aligned, zeroed, leaked identity memory (VA == PA on the aarch64 map).
fn alloc_ident(bytes: usize) -> u64 {
    let layout = Layout::from_size_align(bytes.max(1), 4096).unwrap();
    // SAFETY: nonzero layout; memory is leaked and used only as device-shared DMA.
    let p = unsafe { alloc_zeroed(layout) };
    assert!(!p.is_null(), "virtio_input: DMA alloc failed");
    p as u64
}

struct VirtioInput {
    base: usize,
    avail: u64,  // available ring (driver -> device)
    used: u64,   // used ring (device -> driver)
    events: u64, // QSIZE * 8-byte event buffers
    last_used: u16,
    avail_idx: u16,
    shift: bool,
    ctrl: bool,
}

impl VirtioInput {
    /// Read used-ring index (offset 2).
    unsafe fn used_idx(&self) -> u16 {
        unsafe { read_volatile((self.used + 2) as *const u16) }
    }
    /// used.ring[slot].id (each entry 8 bytes at offset 4).
    unsafe fn used_ring_id(&self, slot: u16) -> u32 {
        let off = 4 + (slot as usize % QSIZE) * 8;
        unsafe { read_volatile((self.used + off as u64) as *const u32) }
    }
    /// Offer descriptor `id` on the available ring and bump its index.
    unsafe fn offer(&mut self, id: u16) {
        let ring = self.avail + 4 + (self.avail_idx as usize % QSIZE * 2) as u64;
        unsafe { write_volatile(ring as *mut u16, id) };
        self.avail_idx = self.avail_idx.wrapping_add(1);
        dsb();
        unsafe { write_volatile((self.avail + 2) as *mut u16, self.avail_idx) };
    }
    unsafe fn notify(&self) {
        dsb();
        unsafe { reg_write(self.base, QUEUE_NOTIFY, 0) };
    }
}

static DEV: Locked<Option<VirtioInput>> = Locked::new(None);
/// A small byte ring the poll fills and `read_byte` drains.
static RING: Locked<InputRing> = Locked::new(InputRing::new());

struct InputRing {
    buf: [u8; 64],
    head: usize,
    tail: usize,
}
impl InputRing {
    const fn new() -> Self {
        Self { buf: [0; 64], head: 0, tail: 0 }
    }
    fn push(&mut self, b: u8) {
        let n = (self.head + 1) % self.buf.len();
        if n != self.tail {
            self.buf[self.head] = b;
            self.head = n;
        }
    }
    fn pop(&mut self) -> Option<u8> {
        if self.head == self.tail {
            return None;
        }
        let b = self.buf[self.tail];
        self.tail = (self.tail + 1) % self.buf.len();
        Some(b)
    }
}

/// Scan the virtio-mmio slots for a virtio-input device and bring it up.
/// Returns whether a keyboard was found and initialized.
pub fn init() -> bool {
    let mut base = 0usize;
    let mut version = 0u32;
    for slot in 0..MMIO_SLOTS {
        let b = MMIO_BASE + slot * MMIO_STRIDE;
        // SAFETY: scanning the fixed virtio-mmio window; registers are 32-bit.
        unsafe {
            let v = reg_read(b, VERSION);
            if reg_read(b, MAGIC) == 0x7472_6976 && (v == 1 || v == 2) && reg_read(b, DEVICE_ID) == VIRTIO_ID_INPUT {
                base = b;
                version = v;
                break;
            }
        }
    }
    if base == 0 {
        return false;
    }

    let events = alloc_ident(QSIZE * 8);
    // SAFETY: `base` is a confirmed virtio-input MMIO block; single-core boot.
    let (avail, used) = unsafe {
        // Reset, then ACK + DRIVER (common to both transport versions).
        reg_write(base, STATUS, 0);
        reg_write(base, STATUS, S_ACK);
        reg_write(base, STATUS, S_ACK | S_DRIVER);

        reg_write(base, QUEUE_SEL, 0);
        if reg_read(base, QUEUE_NUM_MAX) == 0 {
            return false;
        }
        reg_write(base, QUEUE_NUM, QSIZE as u32);

        let (desc, avail, used) = if version == 2 {
            // --- modern (virtio 1.0) ---
            reg_write(base, DEVICE_FEATURES_SEL, 1);
            let _ = reg_read(base, DEVICE_FEATURES);
            reg_write(base, DRIVER_FEATURES_SEL, 1);
            reg_write(base, DRIVER_FEATURES, 1); // ack VIRTIO_F_VERSION_1 (bit 32)
            reg_write(base, DRIVER_FEATURES_SEL, 0);
            reg_write(base, DRIVER_FEATURES, 0);
            reg_write(base, STATUS, S_ACK | S_DRIVER | S_FEATURES_OK);
            if reg_read(base, STATUS) & S_FEATURES_OK == 0 {
                return false;
            }
            let desc = alloc_ident(QSIZE * 16);
            let avail = alloc_ident(6 + QSIZE * 2);
            let used = alloc_ident(6 + QSIZE * 8);
            (desc, avail, used)
        } else {
            // --- legacy (version 1): one contiguous ring, addressed by PFN ---
            // (GuestFeatures/Sel share the modern DriverFeatures/Sel offsets.)
            reg_write(base, DRIVER_FEATURES_SEL, 0);
            reg_write(base, DRIVER_FEATURES, 0); // accept no optional features
            reg_write(base, GUEST_PAGE_SIZE, 4096);
            reg_write(base, QUEUE_ALIGN, 4096);
            const N: usize = QSIZE;
            let used_off = align_up(N * 16 + (6 + N * 2), 4096);
            let region = alloc_ident(used_off + 6 + N * 8);
            // QueuePFN is the region's page frame number.
            reg_write(base, QUEUE_PFN, (region >> 12) as u32);
            (region, region + (N * 16) as u64, region + used_off as u64)
        };

        // Each descriptor points at an 8-byte event buffer, device-writable.
        for i in 0..QSIZE {
            let d = desc + (i * 16) as u64;
            write_volatile(d as *mut u64, events + (i * 8) as u64); // addr
            write_volatile((d + 8) as *mut u32, 8); // len
            write_volatile((d + 12) as *mut u16, VIRTQ_DESC_F_WRITE); // flags
            write_volatile((d + 14) as *mut u16, 0); // next
        }

        if version == 2 {
            reg_write(base, QUEUE_DESC_LOW, desc as u32);
            reg_write(base, QUEUE_DESC_HIGH, (desc >> 32) as u32);
            reg_write(base, QUEUE_DRIVER_LOW, avail as u32);
            reg_write(base, QUEUE_DRIVER_HIGH, (avail >> 32) as u32);
            reg_write(base, QUEUE_DEVICE_LOW, used as u32);
            reg_write(base, QUEUE_DEVICE_HIGH, (used >> 32) as u32);
            reg_write(base, QUEUE_READY, 1);
            reg_write(base, STATUS, S_ACK | S_DRIVER | S_FEATURES_OK | S_DRIVER_OK);
        } else {
            reg_write(base, STATUS, S_ACK | S_DRIVER | S_DRIVER_OK);
        }
        (avail, used)
    };

    let mut dev = VirtioInput { base, avail, used, events, last_used: 0, avail_idx: 0, shift: false, ctrl: false };
    // Offer all buffers to the device.
    // SAFETY: rings are set up above.
    unsafe {
        for i in 0..QSIZE as u16 {
            dev.offer(i);
        }
        dev.notify();
    }
    DEV.with(|slot| *slot = Some(dev));
    crate::ktrace::log_fmt(format_args!("virtio-input: keyboard up at {base:#x} (v{version}, window keystrokes enabled)"));
    true
}

fn align_up(x: usize, a: usize) -> usize {
    (x + a - 1) & !(a - 1)
}

/// Drain any pending key events into the input ring, mapping key presses to
/// ASCII bytes. Cheap; called from `read_byte`.
fn poll() {
    DEV.with(|slot| {
        let Some(dev) = slot.as_mut() else {
            return;
        };
        // SAFETY: rings/buffers are the device's, valid for the device lifetime.
        unsafe {
            let uidx = dev.used_idx();
            while dev.last_used != uidx {
                let id = dev.used_ring_id(dev.last_used) as u16;
                compiler_fence(Ordering::Acquire);
                let ev = read_volatile((dev.events + (id as usize % QSIZE * 8) as u64) as *const InputEvent);
                handle_event(dev, ev);
                dev.last_used = dev.last_used.wrapping_add(1);
                // Re-offer the consumed buffer.
                dev.offer(id);
            }
            dev.notify();
        }
    });
}

fn handle_event(dev: &mut VirtioInput, ev: InputEvent) {
    if ev.typ != EV_KEY {
        return;
    }
    let pressed = ev.value != 0; // 1 = press, 2 = autorepeat, 0 = release
    match ev.code {
        KEY_LEFTSHIFT | KEY_RIGHTSHIFT => {
            dev.shift = pressed;
            return;
        }
        KEY_LEFTCTRL => {
            dev.ctrl = pressed;
            return;
        }
        _ => {}
    }
    if ev.value == 0 {
        return; // key release of a normal key: nothing to emit
    }
    if let Some(ascii) = keycode_to_ascii(ev.code, dev.shift, dev.ctrl) {
        RING.with(|r| r.push(ascii));
    }
}

/// The next byte from the window keyboard, if any (drains events first).
pub fn read_byte() -> Option<u8> {
    poll();
    RING.with(|r| r.pop())
}

/// Map a Linux input keycode (US layout) to an ASCII byte. Handles letters,
/// digits, common punctuation, space, enter, tab, backspace; Shift for the
/// upper register; Ctrl+letter → control code (so Ctrl+C=3, Ctrl+D=4 reach the
/// shell). Returns None for keys with no ASCII (function keys, etc.).
fn keycode_to_ascii(code: u16, shift: bool, ctrl: bool) -> Option<u8> {
    // (base, shifted) for codes 1..=57.
    const MAP: &[(u16, u8, u8)] = &[
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
    let (_, base, shifted) = *MAP.iter().find(|(c, _, _)| *c == code)?;
    let ch = if shift { shifted } else { base };
    if ctrl && ch.is_ascii_alphabetic() {
        Some(ch.to_ascii_uppercase() & 0x1f) // Ctrl+letter -> 0x01..=0x1a
    } else {
        Some(ch)
    }
}
