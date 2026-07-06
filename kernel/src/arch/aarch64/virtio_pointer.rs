//! **virtio-input pointer** (tablet or mouse) over virtio-mmio — the aarch64
//! mouse for the ramfb TUI. QEMU delivers window pointer events via
//! `-device virtio-tablet-device` (absolute) or `-device virtio-mouse-device`
//! (relative); both are virtio-input (device id 18), the same transport as the
//! keyboard ([`super::virtio_input`]). This driver scans the mmio slots, uses
//! the virtio-input **config space** to pick the slot that reports pointer axes
//! (EV_ABS / EV_REL) — so it never grabs the keyboard — brings it up, and on
//! each poll folds motion + button events into [`crate::mouse`].
//!
//! Polled, page-aligned identity DMA — same conventions as `virtio_input`.

use crate::mm::Locked;
use alloc::alloc::{alloc_zeroed, Layout};
use core::ptr::{read_volatile, write_volatile};
use core::sync::atomic::{compiler_fence, Ordering};

// virtio-mmio register offsets (subset; see virtio_input for the full list).
const MAGIC: usize = 0x000;
const VERSION: usize = 0x004;
const DEVICE_ID: usize = 0x008;
const DEVICE_FEATURES: usize = 0x010;
const DEVICE_FEATURES_SEL: usize = 0x014;
const DRIVER_FEATURES: usize = 0x020;
const DRIVER_FEATURES_SEL: usize = 0x024;
const GUEST_PAGE_SIZE: usize = 0x028;
const QUEUE_SEL: usize = 0x030;
const QUEUE_NUM_MAX: usize = 0x034;
const QUEUE_NUM: usize = 0x038;
const QUEUE_ALIGN: usize = 0x03c;
const QUEUE_PFN: usize = 0x040;
const QUEUE_READY: usize = 0x044;
const QUEUE_NOTIFY: usize = 0x050;
const STATUS: usize = 0x070;
const QUEUE_DESC_LOW: usize = 0x080;
const QUEUE_DESC_HIGH: usize = 0x084;
const QUEUE_DRIVER_LOW: usize = 0x090;
const QUEUE_DRIVER_HIGH: usize = 0x094;
const QUEUE_DEVICE_LOW: usize = 0x0a0;
const QUEUE_DEVICE_HIGH: usize = 0x0a4;
const CONFIG: usize = 0x100; // virtio_input_config

const S_ACK: u32 = 1;
const S_DRIVER: u32 = 2;
const S_DRIVER_OK: u32 = 4;
const S_FEATURES_OK: u32 = 8;

const VIRTIO_ID_INPUT: u32 = 18;
const VIRTQ_DESC_F_WRITE: u16 = 2;

const MMIO_BASE: usize = 0x0a00_0000;
const MMIO_STRIDE: usize = 0x200;
const MMIO_SLOTS: usize = 32;
const QSIZE: usize = 8;

// Linux input event types / codes.
const EV_SYN: u16 = 0x00;
const EV_KEY: u16 = 0x01;
const EV_REL: u16 = 0x02;
const EV_ABS: u16 = 0x03;
const REL_X: u16 = 0x00;
const REL_Y: u16 = 0x01;
const REL_WHEEL: u16 = 0x08;
const ABS_X: u16 = 0x00;
const ABS_Y: u16 = 0x01;
const BTN_LEFT: u16 = 0x110;

// virtio_input config selects.
const CFG_EV_BITS: u8 = 0x11;
const CFG_ABS_INFO: u8 = 0x12;

#[inline]
fn dsb() {
    unsafe { core::arch::asm!("dsb sy", options(nomem, nostack, preserves_flags)) };
}
unsafe fn rd(base: usize, off: usize) -> u32 {
    unsafe { read_volatile((base + off) as *const u32) }
}
unsafe fn wr(base: usize, off: usize, v: u32) {
    unsafe { write_volatile((base + off) as *mut u32, v) };
}

#[repr(C)]
#[derive(Clone, Copy)]
struct InputEvent {
    typ: u16,
    code: u16,
    value: u32,
}

fn alloc_ident(bytes: usize) -> u64 {
    let layout = Layout::from_size_align(bytes.max(1), 4096).unwrap();
    let p = unsafe { alloc_zeroed(layout) };
    assert!(!p.is_null(), "virtio_pointer: DMA alloc failed");
    p as u64
}

struct Pointer {
    base: usize,
    avail: u64,
    used: u64,
    events: u64,
    last_used: u16,
    avail_idx: u16,
    abs_max: i32,
}

impl Pointer {
    unsafe fn used_idx(&self) -> u16 {
        unsafe { read_volatile((self.used + 2) as *const u16) }
    }
    unsafe fn used_ring_id(&self, slot: u16) -> u32 {
        let off = 4 + (slot as usize % QSIZE) * 8;
        unsafe { read_volatile((self.used + off as u64) as *const u32) }
    }
    unsafe fn offer(&mut self, id: u16) {
        let ring = self.avail + 4 + (self.avail_idx as usize % QSIZE * 2) as u64;
        unsafe { write_volatile(ring as *mut u16, id) };
        self.avail_idx = self.avail_idx.wrapping_add(1);
        dsb();
        unsafe { write_volatile((self.avail + 2) as *mut u16, self.avail_idx) };
    }
    unsafe fn notify(&self) {
        dsb();
        unsafe { wr(self.base, QUEUE_NOTIFY, 0) };
    }
}

static DEV: Locked<Option<Pointer>> = Locked::new(None);

/// Query the size of a virtio-input config subselection (0 if unsupported).
unsafe fn cfg_size(base: usize, select: u8, subsel: u8) -> u8 {
    unsafe {
        write_volatile((base + CONFIG) as *mut u8, select);
        write_volatile((base + CONFIG + 1) as *mut u8, subsel);
        read_volatile((base + CONFIG + 2) as *const u8) // size
    }
}

/// The ABS axis maximum for `axis` (from `abs_info.max` at config+8..12).
unsafe fn abs_max(base: usize, axis: u8) -> i32 {
    unsafe {
        write_volatile((base + CONFIG) as *mut u8, CFG_ABS_INFO);
        write_volatile((base + CONFIG + 1) as *mut u8, axis);
        // abs_info { u32 min; u32 max; ... } at the config union (offset 8).
        read_volatile((base + CONFIG + 8 + 4) as *const u32) as i32
    }
}

/// Find and bring up a virtio-input pointer device. Returns true on success.
pub fn init() -> bool {
    // The keyboard already claimed one virtio-input device; the pointer is a
    // different virtio-input. Skipping the keyboard's base is the reliable way
    // to tell them apart (both share device id 18); the EV-bits probe then only
    // distinguishes an absolute tablet from a relative mouse.
    let kbd_base = crate::arch::aarch64::virtio_input::claimed_base();
    let mut base = 0usize;
    let mut version = 0u32;
    let mut absolute = true;
    for slot in 0..MMIO_SLOTS {
        let b = MMIO_BASE + slot * MMIO_STRIDE;
        if b == kbd_base {
            continue;
        }
        // SAFETY: scanning the fixed virtio-mmio window.
        unsafe {
            let v = rd(b, VERSION);
            if rd(b, MAGIC) != 0x7472_6976 || !(v == 1 || v == 2) || rd(b, DEVICE_ID) != VIRTIO_ID_INPUT {
                continue;
            }
            base = b;
            version = v;
            // A tablet reports EV_ABS; a mouse reports EV_REL only.
            absolute = cfg_size(b, CFG_EV_BITS, EV_ABS as u8) > 0 || cfg_size(b, CFG_EV_BITS, EV_REL as u8) == 0;
            break;
        }
    }
    if base == 0 {
        return false;
    }

    let events = alloc_ident(QSIZE * 8);
    // SAFETY: `base` is a confirmed virtio-input pointer; single-core boot.
    let (avail, used, abs_x_max) = unsafe {
        let abs_x_max = if absolute {
            let m = abs_max(base, ABS_X as u8);
            if m > 1 { m } else { 32767 } // QEMU tablet default range
        } else {
            32767
        };
        wr(base, STATUS, 0);
        wr(base, STATUS, S_ACK);
        wr(base, STATUS, S_ACK | S_DRIVER);
        wr(base, QUEUE_SEL, 0);
        if rd(base, QUEUE_NUM_MAX) == 0 {
            return false;
        }
        wr(base, QUEUE_NUM, QSIZE as u32);

        let (desc, avail, used) = if version == 2 {
            wr(base, DEVICE_FEATURES_SEL, 1);
            let _ = rd(base, DEVICE_FEATURES);
            wr(base, DRIVER_FEATURES_SEL, 1);
            wr(base, DRIVER_FEATURES, 1);
            wr(base, DRIVER_FEATURES_SEL, 0);
            wr(base, DRIVER_FEATURES, 0);
            wr(base, STATUS, S_ACK | S_DRIVER | S_FEATURES_OK);
            if rd(base, STATUS) & S_FEATURES_OK == 0 {
                return false;
            }
            let desc = alloc_ident(QSIZE * 16);
            let avail = alloc_ident(6 + QSIZE * 2);
            let used = alloc_ident(6 + QSIZE * 8);
            (desc, avail, used)
        } else {
            wr(base, DRIVER_FEATURES_SEL, 0);
            wr(base, DRIVER_FEATURES, 0);
            wr(base, GUEST_PAGE_SIZE, 4096);
            wr(base, QUEUE_ALIGN, 4096);
            let used_off = align_up(QSIZE * 16 + (6 + QSIZE * 2), 4096);
            let region = alloc_ident(used_off + 6 + QSIZE * 8);
            wr(base, QUEUE_PFN, (region >> 12) as u32);
            (region, region + (QSIZE * 16) as u64, region + used_off as u64)
        };

        for i in 0..QSIZE {
            let d = desc + (i * 16) as u64;
            write_volatile(d as *mut u64, events + (i * 8) as u64);
            write_volatile((d + 8) as *mut u32, 8);
            write_volatile((d + 12) as *mut u16, VIRTQ_DESC_F_WRITE);
            write_volatile((d + 14) as *mut u16, 0);
        }

        if version == 2 {
            wr(base, QUEUE_DESC_LOW, desc as u32);
            wr(base, QUEUE_DESC_HIGH, (desc >> 32) as u32);
            wr(base, QUEUE_DRIVER_LOW, avail as u32);
            wr(base, QUEUE_DRIVER_HIGH, (avail >> 32) as u32);
            wr(base, QUEUE_DEVICE_LOW, used as u32);
            wr(base, QUEUE_DEVICE_HIGH, (used >> 32) as u32);
            wr(base, QUEUE_READY, 1);
            wr(base, STATUS, S_ACK | S_DRIVER | S_FEATURES_OK | S_DRIVER_OK);
        } else {
            wr(base, STATUS, S_ACK | S_DRIVER | S_DRIVER_OK);
        }
        (avail, used, abs_x_max)
    };

    let mut dev = Pointer { base, avail, used, events, last_used: 0, avail_idx: 0, abs_max: abs_x_max };
    // SAFETY: rings set up above.
    unsafe {
        for i in 0..QSIZE as u16 {
            dev.offer(i);
        }
        dev.notify();
    }
    DEV.with(|slot| *slot = Some(dev));
    crate::ktrace::log_fmt(format_args!(
        "virtio-input: pointer up at {base:#x} (v{version}, {})",
        if absolute { "absolute/tablet" } else { "relative/mouse" }
    ));
    true
}

fn align_up(x: usize, a: usize) -> usize {
    (x + a - 1) & !(a - 1)
}

/// Drain pending pointer events into [`crate::mouse`]. Absolute devices carry a
/// full `(x, y)` per report; relative devices carry deltas. A left click is
/// EV_KEY / BTN_LEFT.
pub fn poll() {
    DEV.with(|slot| {
        let Some(dev) = slot.as_mut() else { return };
        let mut pend_x: Option<i32> = None;
        let mut pend_y: Option<i32> = None;
        // SAFETY: rings/buffers valid for the device lifetime.
        unsafe {
            let uidx = dev.used_idx();
            while dev.last_used != uidx {
                let id = dev.used_ring_id(dev.last_used) as u16;
                compiler_fence(Ordering::Acquire);
                let ev = read_volatile((dev.events + (id as usize % QSIZE * 8) as u64) as *const InputEvent);
                match ev.typ {
                    EV_ABS if ev.code == ABS_X => pend_x = Some(ev.value as i32),
                    EV_ABS if ev.code == ABS_Y => pend_y = Some(ev.value as i32),
                    EV_REL if ev.code == REL_X => crate::mouse::move_rel(ev.value as i32, 0),
                    EV_REL if ev.code == REL_Y => crate::mouse::move_rel(0, ev.value as i32),
                    // Scroll wheel: the value is signed (+ = up/away). Both the
                    // virtio tablet and mouse report it as EV_REL/REL_WHEEL.
                    EV_REL if ev.code == REL_WHEEL => crate::mouse::add_wheel(ev.value as i32),
                    EV_KEY if ev.code == BTN_LEFT => crate::mouse::set_left(ev.value != 0),
                    EV_SYN => {
                        if let (Some(x), Some(y)) = (pend_x.take(), pend_y.take()) {
                            crate::mouse::set_abs(x, y, dev.abs_max);
                        } else if let Some(x) = pend_x.take() {
                            crate::mouse::set_abs(x, 0, dev.abs_max);
                        } else if let Some(y) = pend_y.take() {
                            crate::mouse::set_abs(0, y, dev.abs_max);
                        }
                    }
                    _ => {}
                }
                dev.last_used = dev.last_used.wrapping_add(1);
                dev.offer(id);
            }
            // Absolute report without an explicit SYN pairing (defensive).
            if let (Some(x), Some(y)) = (pend_x, pend_y) {
                crate::mouse::set_abs(x, y, dev.abs_max);
            }
            dev.notify();
        }
    });
}
