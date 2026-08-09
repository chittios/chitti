//! **virtio-sound over modern virtio-PCI** — the PCM device on a real PCI bus
//! (QEMU `-device virtio-sound-pci`, x86 and any machine with ECAM). Same
//! stream model as the aarch64 mmio driver (`arch::aarch64::virtio_snd`):
//! control(0)/event(1)/tx(2)/rx(3) virtqueues, descriptor chains, stream 0 =
//! playback / stream 1 = capture, poll-driven.

use crate::sound::{proto, SndDevice};
use alloc::boxed::Box;
use alloc::collections::VecDeque;
use core::ptr::{read_volatile, write_volatile};
use core::sync::atomic::{fence, Ordering};

#[cfg(target_arch = "aarch64")]
use crate::pci::{self, PciDevice};
#[cfg(target_arch = "x86_64")]
use crate::arch::x86_64::pci::{self, PciDevice};

// virtio PCI capability cfg_type values (as in net/virtio_net_pci.rs).
const CFG_COMMON: u8 = 1;
const CFG_NOTIFY: u8 = 2;
const CAP_VENDOR: u8 = 0x09;

// common-config field offsets (struct virtio_pci_common_cfg).
const DEVICE_STATUS: u64 = 0x14;
const QUEUE_SELECT: u64 = 0x16;
const QUEUE_SIZE: u64 = 0x18;
const QUEUE_ENABLE: u64 = 0x1c;
const QUEUE_NOTIFY_OFF: u64 = 0x1e;
const QUEUE_DESC: u64 = 0x20;
const QUEUE_DRIVER: u64 = 0x28;
const QUEUE_DEVICE: u64 = 0x30;
const DRIVER_FEATURE_SELECT: u64 = 0x08;
const DRIVER_FEATURE: u64 = 0x0c;

const S_ACK: u8 = 1;
const S_DRIVER: u8 = 2;
const S_DRIVER_OK: u8 = 4;
const S_FEATURES_OK: u8 = 8;

const VIRTQ_DESC_F_NEXT: u16 = 1;
const VIRTQ_DESC_F_WRITE: u16 = 2;

// virtio-sound PCI device id: transitional 0x1040+25 (modern).
const VIRTIO_PCI_VENDOR: u16 = 0x1af4;
const VIRTIO_SND_MODERN_DEV: u16 = 0x1040 + 25;

const PERIOD: usize = 3200;
const NBUF: usize = 8;
const QSIZE: u16 = 64;

#[inline]
unsafe fn r8(a: u64) -> u8 {
    unsafe { read_volatile(a as *const u8) }
}
#[inline]
unsafe fn w8(a: u64, v: u8) {
    unsafe { write_volatile(a as *mut u8, v) };
}
#[inline]
unsafe fn r16(a: u64) -> u16 {
    unsafe { read_volatile(a as *const u16) }
}
#[inline]
unsafe fn w16(a: u64, v: u16) {
    unsafe { write_volatile(a as *mut u16, v) };
}
#[inline]
unsafe fn r32(a: u64) -> u32 {
    unsafe { read_volatile(a as *const u32) }
}
#[inline]
unsafe fn w32(a: u64, v: u32) {
    unsafe { write_volatile(a as *mut u32, v) };
}
#[inline]
unsafe fn w64v(a: u64, v: u64) {
    unsafe { write_volatile(a as *mut u64, v) };
}

fn cfg_read8(d: &PciDevice, off: u16) -> u8 {
    (pci::read32(d.bus, d.dev, d.func, off & !3) >> ((off & 3) * 8)) as u8
}
fn cfg_read32_at(d: &PciDevice, off: u16) -> u32 {
    pci::read32(d.bus, d.dev, d.func, off)
}

/// One split virtqueue with chain support (PCI transport).
struct Queue {
    qsize: u16,
    desc: u64,
    avail: u64,
    used: u64,
    notify: u64,
    vqn: u16,
    avail_idx: u16,
    used_last: u16,
}

impl Queue {
    unsafe fn set_desc(&self, i: u16, addr: u64, len: u32, flags: u16, next: u16) {
        unsafe {
            let d = self.desc + i as u64 * 16;
            w64v(d, addr);
            w32(d + 8, len);
            w16(d + 12, flags);
            w16(d + 14, next);
        }
    }
    unsafe fn post(&mut self, head: u16) {
        unsafe {
            let slot = self.avail_idx % self.qsize;
            w16(self.avail + 4 + slot as u64 * 2, head);
            fence(Ordering::SeqCst);
            self.avail_idx = self.avail_idx.wrapping_add(1);
            w16(self.avail + 2, self.avail_idx);
            fence(Ordering::SeqCst);
            w16(self.notify, self.vqn);
        }
    }
    unsafe fn pop_used(&mut self) -> Option<(u16, u32)> {
        unsafe {
            if r16(self.used + 2) == self.used_last {
                return None;
            }
            fence(Ordering::SeqCst);
            let slot = (self.used_last % self.qsize) as u64;
            let id = r32(self.used + 4 + slot * 8) as u16;
            let len = r32(self.used + 4 + slot * 8 + 4);
            self.used_last = self.used_last.wrapping_add(1);
            Some((id, len))
        }
    }
}

unsafe fn setup_queue(c: u64, notify_base: u64, notify_mult: u32, idx: u16) -> Option<Queue> {
    unsafe {
        w16(c + QUEUE_SELECT, idx);
        let qmax = r16(c + QUEUE_SIZE);
        if qmax == 0 {
            return None;
        }
        let qsize = QSIZE.min(qmax);
        w16(c + QUEUE_SIZE, qsize);
        let qs = qsize as usize;
        let (dp, desc) = crate::mm::alloc_dma(qs * 16)?;
        let (ap, avail) = crate::mm::alloc_dma(6 + qs * 2)?;
        let (up, used) = crate::mm::alloc_dma(6 + qs * 8)?;
        w64v(c + QUEUE_DESC, dp);
        w64v(c + QUEUE_DRIVER, ap);
        w64v(c + QUEUE_DEVICE, up);
        let off = r16(c + QUEUE_NOTIFY_OFF);
        w16(c + QUEUE_ENABLE, 1);
        Some(Queue {
            qsize,
            desc,
            avail,
            used,
            notify: notify_base + off as u64 * notify_mult as u64,
            vqn: idx,
            avail_idx: 0,
            used_last: 0,
        })
    }
}

/// The poll-driven virtio-sound device (PCI transport).
pub struct VirtioSndPci {
    ctrl: Queue,
    tx: Queue,
    rx: Queue,
    ctrl_msg: (u64, u64),  // (phys, virt)
    ctrl_resp: (u64, u64),
    tx_buf: (u64, u64),
    tx_inflight: [bool; NBUF],
    out_rate: u32,
    out_running: bool,
    rx_buf: (u64, u64),
    in_running: bool,
    pending: VecDeque<i16>,
}

impl VirtioSndPci {
    /// Find a virtio-sound PCI function and bring it up.
    pub fn probe() -> Option<Box<dyn SndDevice>> {
        // Multimedia audio device: class 0x04, any subclass/prog-if — verify by
        // vendor/device id (modern virtio-snd).
        let d = pci::find_class(0x04, 0x01, 0x00)
            .or_else(|| pci::find_class(0x04, 0x00, 0x00))
            .filter(|d| d.vendor == VIRTIO_PCI_VENDOR && d.device == VIRTIO_SND_MODERN_DEV)?;
        Self::init(d).map(|dev| Box::new(dev) as Box<dyn SndDevice>)
    }

    fn init(d: PciDevice) -> Option<VirtioSndPci> {
        d.enable_bus_master();
        // Walk the capability list for COMMON + NOTIFY structures.
        if cfg_read32_at(&d, 0x04) & (1 << 20) == 0 {
            return None;
        }
        let (mut common, mut notify, mut notify_mult) = (0u64, 0u64, 0u32);
        let mut cap = cfg_read8(&d, 0x34) & 0xfc;
        let mut guard = 0;
        while cap != 0 && guard < 48 {
            guard += 1;
            let vndr = cfg_read8(&d, cap as u16);
            let next = cfg_read8(&d, cap as u16 + 1) & 0xfc;
            if vndr == CAP_VENDOR {
                let cfg_type = cfg_read8(&d, cap as u16 + 3);
                let bar = cfg_read8(&d, cap as u16 + 4);
                let offset = cfg_read32_at(&d, cap as u16 + 8);
                let bar_phys = d.bar(bar);
                if bar_phys != 0 {
                    // Size the mapping by the capability's own length: `offset`
                    // is unbounded within the BAR, and a fixed span stops
                    // covering it as soon as a device puts a structure higher.
                    let length = cfg_read32_at(&d, cap as u16 + 12);
                    let span = (offset as usize).saturating_add(length.max(1) as usize);
                    let virt = crate::mm::map_mmio(bar_phys, span) + offset as u64;
                    match cfg_type {
                        CFG_COMMON => common = virt,
                        CFG_NOTIFY => {
                            notify = virt;
                            notify_mult = cfg_read32_at(&d, cap as u16 + 16);
                        }
                        _ => {}
                    }
                }
            }
            cap = next;
        }
        if common == 0 || notify == 0 {
            return None;
        }
        // SAFETY: HHDM-mapped virtio BAR regions + fresh DMA allocations.
        unsafe {
            w8(common + DEVICE_STATUS, 0);
            while r8(common + DEVICE_STATUS) != 0 {
                core::hint::spin_loop();
            }
            w8(common + DEVICE_STATUS, S_ACK);
            w8(common + DEVICE_STATUS, S_ACK | S_DRIVER);
            w32(common + DRIVER_FEATURE_SELECT, 0);
            w32(common + DRIVER_FEATURE, 0);
            w32(common + DRIVER_FEATURE_SELECT, 1);
            w32(common + DRIVER_FEATURE, 1); // VIRTIO_F_VERSION_1
            w8(common + DEVICE_STATUS, S_ACK | S_DRIVER | S_FEATURES_OK);
            if r8(common + DEVICE_STATUS) & S_FEATURES_OK == 0 {
                return None;
            }
            let ctrl = setup_queue(common, notify, notify_mult, 0)?;
            let _event = setup_queue(common, notify, notify_mult, 1)?;
            let tx = setup_queue(common, notify, notify_mult, 2)?;
            let rx = setup_queue(common, notify, notify_mult, 3)?;
            w8(common + DEVICE_STATUS, S_ACK | S_DRIVER | S_FEATURES_OK | S_DRIVER_OK);

            let ctrl_msg = crate::mm::alloc_dma(64)?;
            let ctrl_resp = crate::mm::alloc_dma(64)?;
            let tx_buf = crate::mm::alloc_dma(NBUF * (4 + PERIOD + 8))?;
            let rx_buf = crate::mm::alloc_dma(NBUF * (4 + PERIOD + 8))?;

            crate::ktrace::log_fmt(format_args!("virtio-snd-pci: up (dev {:04x})", d.device));
            Some(VirtioSndPci {
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

    fn ctrl_call(&mut self, msg: &[u8]) -> u32 {
        // SAFETY: ctrl buffers are the driver's DMA regions; 2-desc chain.
        unsafe {
            core::ptr::copy_nonoverlapping(msg.as_ptr(), self.ctrl_msg.1 as *mut u8, msg.len());
            core::ptr::write_bytes(self.ctrl_resp.1 as *mut u8, 0, 64);
            self.ctrl.set_desc(0, self.ctrl_msg.0, msg.len() as u32, VIRTQ_DESC_F_NEXT, 1);
            self.ctrl.set_desc(1, self.ctrl_resp.0, 64, VIRTQ_DESC_F_WRITE, 0);
            self.ctrl.post(0);
            for _ in 0..2_000_000 {
                if self.ctrl.pop_used().is_some() {
                    return r32(self.ctrl_resp.1);
                }
                core::hint::spin_loop();
            }
            0
        }
    }

    fn tx_slot(&self, i: usize) -> (u64, u64, u64, u64) {
        let off = (i * (4 + PERIOD + 8)) as u64;
        let (p, v) = self.tx_buf;
        (p + off, v + off, p + off + 4 + PERIOD as u64, v + off + 4 + PERIOD as u64)
    }
    fn rx_slot(&self, i: usize) -> (u64, u64) {
        let off = (i * (4 + PERIOD + 8)) as u64;
        (self.rx_buf.0 + off, self.rx_buf.1 + off)
    }

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

    fn rx_post(&mut self, i: usize) {
        let (p, v) = self.rx_slot(i);
        // SAFETY: buffers are the driver's DMA regions; 3-desc chain at 3i.
        unsafe {
            w32(v, 1); // stream_id 1 = capture
            let d0 = (i * 3) as u16;
            self.rx.set_desc(d0, p, 4, VIRTQ_DESC_F_NEXT, d0 + 1);
            self.rx.set_desc(d0 + 1, p + 4, PERIOD as u32, VIRTQ_DESC_F_WRITE | VIRTQ_DESC_F_NEXT, d0 + 2);
            self.rx.set_desc(d0 + 2, p + 4 + PERIOD as u64, 8, VIRTQ_DESC_F_WRITE, 0);
            self.rx.post(d0);
        }
    }

    fn stream_start(&mut self, id: u32, hz: u32) -> Result<(), &'static str> {
        if self.ctrl_call(&proto::set_params(id, hz, (NBUF * PERIOD) as u32, PERIOD as u32)) != proto::S_OK {
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

impl SndDevice for VirtioSndPci {
    fn play(&mut self, pcm: &[i16], hz: u32) -> Result<(), &'static str> {
        if !self.out_running || self.out_rate != hz {
            if self.out_running {
                self.stream_stop(0);
            }
            self.stream_start(0, hz)?;
            self.out_running = true;
            self.out_rate = hz;
        }
        let bytes: &[u8] =
            // SAFETY: reinterpreting &[i16] as little-endian bytes (LE targets).
            unsafe { core::slice::from_raw_parts(pcm.as_ptr() as *const u8, pcm.len() * 2) };
        for chunk in bytes.chunks(PERIOD) {
            let slot = loop {
                self.tx_reclaim();
                if let Some(i) = self.tx_inflight.iter().position(|b| !b) {
                    break i;
                }
                crate::sched::yield_now();
            };
            let (pp, pv, sp, _sv) = self.tx_slot(slot);
            // SAFETY: slot buffers are the driver's DMA regions; 2-desc chain.
            unsafe {
                w32(pv, 0); // stream_id 0 = playback
                core::ptr::copy_nonoverlapping(chunk.as_ptr(), (pv + 4) as *mut u8, chunk.len());
                let d0 = (slot * 2) as u16;
                self.tx.set_desc(d0, pp, (4 + chunk.len()) as u32, VIRTQ_DESC_F_NEXT, d0 + 1);
                self.tx.set_desc(d0 + 1, sp, 8, VIRTQ_DESC_F_WRITE, 0);
                self.tx_inflight[slot] = true;
                self.tx.post(d0);
            }
        }
        Ok(())
    }

    fn out_free_bytes(&mut self) -> usize {
        self.tx_reclaim();
        self.tx_inflight.iter().filter(|&&b| !b).count() * PERIOD
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
                        let (_, v) = self.rx_slot(slot);
                        let n = (len as usize).saturating_sub(8).min(PERIOD) / 2;
                        for k in 0..n {
                            let lo = read_volatile((v + 4 + (k * 2) as u64) as *const u8) as u16;
                            let hi = read_volatile((v + 4 + (k * 2 + 1) as u64) as *const u8) as u16;
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
