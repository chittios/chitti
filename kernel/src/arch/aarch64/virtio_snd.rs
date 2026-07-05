//! **virtio-sound over the virtio-mmio transport** — the PCM device behind the
//! `sound` subsystem on the QEMU `virt` `-kernel` path (`-device
//! virtio-sound-device,audiodev=…`). Four virtqueues: control(0), event(1),
//! tx(2, playback) and rx(3, capture). Requests are descriptor *chains*
//! (message → response; header+payload → status), unlike the single-descriptor
//! net/blk queues.
//!
//! Stream layout: QEMU exposes stream 0 as OUTPUT and stream 1 as INPUT when
//! the audiodev is duplex (coreaudio is); the `streams` count is read from the
//! device config. (A full `PCM_INFO` direction query is the real-hardware
//! hardening step; QEMU's ordering is the virtio-snd reference behaviour.)
//!
//! MMIO register access is a single `ldr`/`str` via inline asm — LLVM otherwise
//! coalesces adjacent volatile accesses into a paired load HVF cannot decode
//! (`hvf: isv`); see the same note in `virtio_net.rs` and CLAUDE.md.

use crate::sound::{proto, SndDevice};
use alloc::alloc::{alloc_zeroed, Layout};
use alloc::collections::VecDeque;
use core::ptr::{read_volatile, write_volatile};
use core::sync::atomic::{fence, Ordering};

use crate::arch::aarch64::dma_to_phys as dma;

// virtio-mmio register offsets (shared layout with virtio_net/virtio_blk).
const MAGIC: usize = 0x000;
const VERSION: usize = 0x004;
const DEVICE_ID: usize = 0x008;
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
const CONFIG: usize = 0x100; // { le32 jacks; le32 streams; le32 chmaps; }

const S_ACK: u32 = 1;
const S_DRIVER: u32 = 2;
const S_DRIVER_OK: u32 = 4;
const S_FEATURES_OK: u32 = 8;

const VIRTIO_ID_SOUND: u32 = 25;
const VIRTQ_DESC_F_NEXT: u16 = 1;
const VIRTQ_DESC_F_WRITE: u16 = 2;

const MMIO_BASE: usize = 0x0a00_0000;
const MMIO_STRIDE: usize = 0x200;
const MMIO_SLOTS: usize = 32;

/// PCM chunk size per TX/RX buffer: 100 ms of S16 mono at 16 kHz.
const PERIOD: usize = 3200;
/// In-flight buffers per direction.
const NBUF: usize = 8;
const QSIZE: usize = 64; // descriptors per queue (chains use 2-3 each)

#[inline]
fn dsb() {
    unsafe { core::arch::asm!("dsb sy", options(nomem, nostack, preserves_flags)) };
}
// Single-instruction MMIO (see module doc — HVF cannot decode merged accesses).
unsafe fn reg_read(base: usize, off: usize) -> u32 {
    let v: u32;
    // SAFETY: `base+off` is a mapped 32-bit virtio-mmio register.
    unsafe {
        core::arch::asm!("ldr {v:w}, [{a}]", v = out(reg) v, a = in(reg) base + off, options(nostack, preserves_flags));
    }
    v
}
unsafe fn reg_write(base: usize, off: usize, val: u32) {
    // SAFETY: `base+off` is a mapped 32-bit virtio-mmio register.
    unsafe {
        core::arch::asm!("str {v:w}, [{a}]", v = in(reg) val, a = in(reg) base + off, options(nostack, preserves_flags));
    }
}
fn align_up(x: usize, a: usize) -> usize {
    (x + a - 1) & !(a - 1)
}
fn alloc_ident(bytes: usize) -> u64 {
    let layout = Layout::from_size_align(bytes.max(1), 4096).unwrap();
    // SAFETY: nonzero layout; leaked, used only as device-shared DMA memory.
    let p = unsafe { alloc_zeroed(layout) };
    assert!(!p.is_null(), "virtio_snd: DMA alloc failed");
    p as u64
}
unsafe fn wr16(a: u64, v: u16) {
    unsafe { write_volatile(a as *mut u16, v) };
}
unsafe fn wr32(a: u64, v: u32) {
    unsafe { write_volatile(a as *mut u32, v) };
}
unsafe fn wr64(a: u64, v: u64) {
    unsafe { write_volatile(a as *mut u64, v) };
}
unsafe fn rd16(a: u64) -> u16 {
    unsafe { read_volatile(a as *const u16) }
}
unsafe fn rd32(a: u64) -> u32 {
    unsafe { read_volatile(a as *const u32) }
}

/// One split virtqueue supporting descriptor chains.
struct Queue {
    idx: u32,
    qsize: u16,
    desc: u64,
    avail: u64,
    used: u64,
    avail_idx: u16,
    used_last: u16,
}

impl Queue {
    unsafe fn setup(base: usize, idx: u32, version: u32) -> Option<Queue> {
        unsafe {
            reg_write(base, QUEUE_SEL, idx);
            let qmax = reg_read(base, QUEUE_NUM_MAX);
            if qmax == 0 {
                return None;
            }
            let qsize = (QSIZE as u32).min(qmax) as u16;
            reg_write(base, QUEUE_NUM, qsize as u32);
            let qs = qsize as usize;
            let (desc, avail, used) = if version == 2 {
                let desc = alloc_ident(qs * 16);
                let avail = alloc_ident(6 + qs * 2);
                let used = alloc_ident(6 + qs * 8);
                let (dp, ap, up) = (dma(desc), dma(avail), dma(used));
                reg_write(base, QUEUE_DESC_LOW, dp as u32);
                reg_write(base, QUEUE_DESC_HIGH, (dp >> 32) as u32);
                reg_write(base, QUEUE_DRIVER_LOW, ap as u32);
                reg_write(base, QUEUE_DRIVER_HIGH, (ap >> 32) as u32);
                reg_write(base, QUEUE_DEVICE_LOW, up as u32);
                reg_write(base, QUEUE_DEVICE_HIGH, (up >> 32) as u32);
                reg_write(base, QUEUE_READY, 1);
                (desc, avail, used)
            } else {
                let used_off = align_up(qs * 16 + (6 + qs * 2), 4096);
                let region = alloc_ident(used_off + 6 + qs * 8);
                reg_write(base, GUEST_PAGE_SIZE, 4096);
                reg_write(base, QUEUE_ALIGN, 4096);
                reg_write(base, QUEUE_PFN, (dma(region) >> 12) as u32);
                (region, region + (qs * 16) as u64, region + used_off as u64)
            };
            Some(Queue { idx, qsize, desc, avail, used, avail_idx: 0, used_last: 0 })
        }
    }

    /// Write descriptor `i` (`addr/len/flags/next`).
    unsafe fn set_desc(&self, i: u16, addr: u64, len: u32, flags: u16, next: u16) {
        unsafe {
            let d = self.desc + i as u64 * 16;
            wr64(d, addr);
            wr32(d + 8, len);
            wr16(d + 12, flags);
            wr16(d + 14, next);
        }
    }

    /// Post the chain headed by descriptor `head` and ring the doorbell.
    unsafe fn post(&mut self, base: usize, head: u16) {
        unsafe {
            let slot = self.avail_idx % self.qsize;
            wr16(self.avail + 4 + slot as u64 * 2, head);
            fence(Ordering::SeqCst);
            self.avail_idx = self.avail_idx.wrapping_add(1);
            wr16(self.avail + 2, self.avail_idx);
            dsb();
            reg_write(base, QUEUE_NOTIFY, self.idx);
        }
    }

    unsafe fn pop_used(&mut self) -> Option<(u16, u32)> {
        unsafe {
            if rd16(self.used + 2) == self.used_last {
                return None;
            }
            fence(Ordering::SeqCst);
            let slot = (self.used_last % self.qsize) as u64;
            let id = rd32(self.used + 4 + slot * 8) as u16;
            let len = rd32(self.used + 4 + slot * 8 + 4);
            self.used_last = self.used_last.wrapping_add(1);
            Some((id, len))
        }
    }
}

/// The poll-driven virtio-sound device (mmio transport).
pub struct VirtioSndMmio {
    base: usize,
    ctrl: Queue,
    tx: Queue,
    rx: Queue,
    // Control-plane DMA buffers.
    ctrl_msg: u64,  // 64 B request
    ctrl_resp: u64, // 64 B response
    // Playback: NBUF slots of [4B stream hdr + PERIOD payload] + 8B status.
    tx_buf: u64,
    tx_inflight: [bool; NBUF],
    out_rate: u32,
    out_running: bool,
    // Capture: NBUF slots of [4B hdr][PERIOD payload][8B status].
    rx_buf: u64,
    in_running: bool,
    /// Captured samples not yet handed to the caller.
    pending: VecDeque<i16>,
}

impl VirtioSndMmio {
    /// Scan the virtio-mmio window for a sound device (id 25) and bring it up.
    pub fn probe() -> Option<VirtioSndMmio> {
        for slot in 0..MMIO_SLOTS {
            let b = MMIO_BASE + slot * MMIO_STRIDE;
            // SAFETY: scanning the fixed virtio-mmio window; 32-bit registers.
            unsafe {
                let v = reg_read(b, VERSION);
                if reg_read(b, MAGIC) == 0x7472_6976 && (v == 1 || v == 2) && reg_read(b, DEVICE_ID) == VIRTIO_ID_SOUND {
                    if let Some(d) = Self::init(b, v) {
                        return Some(d);
                    }
                }
            }
        }
        None
    }

    unsafe fn init(base: usize, version: u32) -> Option<VirtioSndMmio> {
        unsafe {
            reg_write(base, STATUS, 0);
            reg_write(base, STATUS, S_ACK);
            reg_write(base, STATUS, S_ACK | S_DRIVER);
            // Feature negotiation: nothing needed from the low word; ack
            // VERSION_1 on a modern device.
            reg_write(base, DEVICE_FEATURES_SEL, 0);
            if version == 2 {
                reg_write(base, DRIVER_FEATURES_SEL, 0);
                reg_write(base, DRIVER_FEATURES, 0);
                reg_write(base, DRIVER_FEATURES_SEL, 1);
                reg_write(base, DRIVER_FEATURES, 1); // VIRTIO_F_VERSION_1
                reg_write(base, STATUS, S_ACK | S_DRIVER | S_FEATURES_OK);
                if reg_read(base, STATUS) & S_FEATURES_OK == 0 {
                    return None;
                }
            } else {
                reg_write(base, DRIVER_FEATURES_SEL, 0);
                reg_write(base, DRIVER_FEATURES, 0);
            }

            let ctrl = Queue::setup(base, 0, version)?;
            let _event = Queue::setup(base, 1, version)?; // drained never; we poll PCM directly
            let tx = Queue::setup(base, 2, version)?;
            let rx = Queue::setup(base, 3, version)?;

            let ok = if version == 2 { S_ACK | S_DRIVER | S_FEATURES_OK | S_DRIVER_OK } else { S_ACK | S_DRIVER | S_DRIVER_OK };
            reg_write(base, STATUS, ok);

            // Device config (read after DRIVER_OK, like the other drivers).
            let streams = reg_read(base, CONFIG + 4);

            let ctrl_msg = alloc_ident(64);
            let ctrl_resp = alloc_ident(64);
            let tx_buf = alloc_ident(NBUF * (4 + PERIOD + 8));
            let rx_buf = alloc_ident(NBUF * (4 + PERIOD + 8));

            crate::ktrace::log_fmt(format_args!("virtio-snd-mmio: up at {base:#x} (v{version}), {streams} streams"));
            Some(VirtioSndMmio {
                base,
                ctrl,
                tx,
                rx,
                ctrl_msg,
                ctrl_resp,
                tx_buf,
                tx_inflight: [false; NBUF],
                out_rate: 0,
                out_running: false,
                rx_buf,
                in_running: false,
                pending: VecDeque::new(),
            })
        }
    }

    /// Issue one control request and poll for its response status.
    fn ctrl_call(&mut self, msg: &[u8]) -> u32 {
        // SAFETY: ctrl buffers are the driver's DMA regions; 2-desc chain.
        unsafe {
            core::ptr::copy_nonoverlapping(msg.as_ptr(), self.ctrl_msg as *mut u8, msg.len());
            core::ptr::write_bytes(self.ctrl_resp as *mut u8, 0, 64);
            self.ctrl.set_desc(0, dma(self.ctrl_msg), msg.len() as u32, VIRTQ_DESC_F_NEXT, 1);
            self.ctrl.set_desc(1, dma(self.ctrl_resp), 64, VIRTQ_DESC_F_WRITE, 0);
            self.ctrl.post(self.base, 0);
            for _ in 0..2_000_000 {
                if self.ctrl.pop_used().is_some() {
                    return rd32(self.ctrl_resp);
                }
                core::hint::spin_loop();
            }
            0 // timeout: treated as failure (status != S_OK)
        }
    }

    fn tx_slot_addrs(&self, i: usize) -> (u64, u64) {
        let s = self.tx_buf + (i * (4 + PERIOD + 8)) as u64;
        (s, s + 4 + PERIOD as u64) // (hdr+payload, status)
    }
    fn rx_slot_addrs(&self, i: usize) -> (u64, u64, u64) {
        let s = self.rx_buf + (i * (4 + PERIOD + 8)) as u64;
        (s, s + 4, s + 4 + PERIOD as u64) // (hdr, payload, status)
    }

    /// Reclaim completed TX chains.
    fn tx_reclaim(&mut self) {
        // SAFETY: ring accesses on the live queue.
        unsafe {
            while let Some((head, _)) = self.tx.pop_used() {
                let slot = (head as usize) / 2;
                if slot < NBUF {
                    self.tx_inflight[slot] = false;
                }
            }
        }
    }

    /// Post capture buffer `i` (hdr readable, payload+status writable).
    fn rx_post(&mut self, i: usize) {
        let (h, p, st) = self.rx_slot_addrs(i);
        // SAFETY: buffers are the driver's DMA regions; 3-desc chain at 3i.
        unsafe {
            wr32(h, 1); // stream_id 1 = capture
            let d0 = (i * 3) as u16;
            self.rx.set_desc(d0, dma(h), 4, VIRTQ_DESC_F_NEXT, d0 + 1);
            self.rx.set_desc(d0 + 1, dma(p), PERIOD as u32, VIRTQ_DESC_F_WRITE | VIRTQ_DESC_F_NEXT, d0 + 2);
            self.rx.set_desc(d0 + 2, dma(st), 8, VIRTQ_DESC_F_WRITE, 0);
            self.rx.post(self.base, d0);
        }
    }

    /// Configure + start stream `id` at `hz`.
    fn stream_start(&mut self, id: u32, hz: u32) -> Result<(), &'static str> {
        let st = self.ctrl_call(&proto::set_params(id, hz, (NBUF * PERIOD) as u32, PERIOD as u32));
        if st != proto::S_OK {
            return Err("virtio-snd: SET_PARAMS refused");
        }
        if self.ctrl_call(&proto::pcm_op(proto::R_PCM_PREPARE, id)) != proto::S_OK {
            return Err("virtio-snd: PREPARE refused");
        }
        if self.ctrl_call(&proto::pcm_op(proto::R_PCM_START, id)) != proto::S_OK {
            return Err("virtio-snd: START refused");
        }
        Ok(())
    }

    fn stream_stop(&mut self, id: u32) {
        let _ = self.ctrl_call(&proto::pcm_op(proto::R_PCM_STOP, id));
        let _ = self.ctrl_call(&proto::pcm_op(proto::R_PCM_RELEASE, id));
    }
}

impl SndDevice for VirtioSndMmio {
    fn play(&mut self, pcm: &[i16], hz: u32) -> Result<(), &'static str> {
        if !self.out_running || self.out_rate != hz {
            if self.out_running {
                self.stream_stop(0);
            }
            self.stream_start(0, hz)?;
            self.out_running = true;
            self.out_rate = hz;
        }
        // Chunk into PERIOD-byte pieces and enqueue, waiting for free slots.
        let bytes: &[u8] =
            // SAFETY: reinterpreting &[i16] as little-endian bytes (LE targets).
            unsafe { core::slice::from_raw_parts(pcm.as_ptr() as *const u8, pcm.len() * 2) };
        for chunk in bytes.chunks(PERIOD) {
            // Find (wait for) a free slot.
            let slot = loop {
                self.tx_reclaim();
                if let Some(i) = self.tx_inflight.iter().position(|b| !b) {
                    break i;
                }
                crate::sched::yield_now();
            };
            let (hp, st) = self.tx_slot_addrs(slot);
            // SAFETY: slot buffers are the driver's DMA regions; 2-desc chain.
            unsafe {
                wr32(hp, 0); // stream_id 0 = playback
                core::ptr::copy_nonoverlapping(chunk.as_ptr(), (hp + 4) as *mut u8, chunk.len());
                let d0 = (slot * 2) as u16;
                self.tx.set_desc(d0, dma(hp), (4 + chunk.len()) as u32, VIRTQ_DESC_F_NEXT, d0 + 1);
                self.tx.set_desc(d0 + 1, dma(st), 8, VIRTQ_DESC_F_WRITE, 0);
                self.tx_inflight[slot] = true;
                self.tx.post(self.base, d0);
            }
        }
        Ok(())
    }

    fn playing(&mut self) -> bool {
        self.tx_reclaim();
        self.tx_inflight.iter().any(|&b| b)
    }

    fn capture_start(&mut self, hz: u32) -> Result<(), &'static str> {
        if self.in_running {
            return Ok(());
        }
        self.stream_start(1, hz)?;
        self.in_running = true;
        self.pending.clear();
        for i in 0..NBUF {
            self.rx_post(i);
        }
        Ok(())
    }

    fn capture_read(&mut self, out: &mut [i16]) -> usize {
        if self.in_running {
            // SAFETY: ring accesses on the live queue + DMA payload reads.
            unsafe {
                while let Some((head, len)) = self.rx.pop_used() {
                    let slot = (head as usize) / 3;
                    if slot < NBUF {
                        let (_, p, _) = self.rx_slot_addrs(slot);
                        // Written bytes = total used len minus the 8-byte status.
                        let n = (len as usize).saturating_sub(8).min(PERIOD) / 2;
                        for k in 0..n {
                            let lo = read_volatile((p + (k * 2) as u64) as *const u8) as u16;
                            let hi = read_volatile((p + (k * 2 + 1) as u64) as *const u8) as u16;
                            self.pending.push_back((lo | (hi << 8)) as i16);
                        }
                        self.rx_post(slot);
                    }
                }
            }
        }
        let mut n = 0;
        while n < out.len() {
            match self.pending.pop_front() {
                Some(s) => {
                    out[n] = s;
                    n += 1;
                }
                None => break,
            }
        }
        n
    }

    fn capture_stop(&mut self) {
        if self.in_running {
            self.stream_stop(1);
            self.in_running = false;
            self.pending.clear();
        }
    }
}
