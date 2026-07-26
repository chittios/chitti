//! **virtio-gpu** display driver over the modern virtio-PCI transport — real
//! kernel mode setting, the [`DisplayDriver`] backend for QEMU-class machines.
//!
//! PCI rather than virtio-mmio deliberately: `virtio-gpu-pci` exists on **both**
//! `qemu-system-aarch64 -M virt` and `qemu-system-x86_64`, so one transport serves
//! both architectures and the standing no-divergence rule is met without writing
//! the queue code twice. (`pci::PciDevice` is already arch-neutral; this is modelled
//! on `net/virtio_net_pci.rs`, which does the same thing for the NIC.)
//!
//! How a mode set works here, and why it is a genuine one rather than a letterbox:
//!
//! 1. `GET_DISPLAY_INFO` reports each scanout's current rect — the host window.
//! 2. `RESOURCE_CREATE_2D` declares a host-side resource at the wanted size.
//! 3. `RESOURCE_ATTACH_BACKING` hands the device **our** DMA pages, so the
//!    compositor draws straight into the scanout's memory with no copy.
//! 4. `SET_SCANOUT` points the output at it — the display is now that size.
//! 5. `TRANSFER_TO_HOST_2D` + `RESOURCE_FLUSH` present a dirty rect.
//!
//! Step 5 is the one structural difference from a firmware framebuffer: virtio-gpu
//! does not scan out of guest memory continuously, so drawing alone changes
//! nothing on screen. Damage is unioned by [`crate::kms`] and flushed once per idle
//! tick — a flush per glyph would be a queue round trip per glyph.
//!
//! The wire format lives in [`super::virtio_gpu_proto`], pure and unit-tested; this
//! file is the hardware half and is not unit-testable (it needs a device).

use super::virtio_gpu_proto as proto;
use super::{Connector, DisplayDriver, Mode, Scanout};
// PCI lives in a different module per arch (there is no single `crate::pci`), so
// import it the way `net/virtio_net_pci.rs` does — this is what keeps one driver
// serving both architectures.
#[cfg(target_arch = "aarch64")]
use crate::pci::{self, PciDevice};
#[cfg(target_arch = "x86_64")]
use crate::arch::x86_64::pci::{self, PciDevice};
use alloc::vec::Vec;
use core::ptr::{read_volatile, write_volatile};
use core::sync::atomic::{fence, Ordering};

// virtio PCI capability cfg_type values.
const CAP_VENDOR: u8 = 0x09;
const CFG_COMMON: u8 = 1;
const CFG_NOTIFY: u8 = 2;
const CFG_DEVICE: u8 = 4;

// virtio_pci_common_cfg offsets.
const DEVICE_FEATURE_SELECT: u64 = 0x00;
const DEVICE_FEATURE: u64 = 0x04;
const DRIVER_FEATURE_SELECT: u64 = 0x08;
const DRIVER_FEATURE: u64 = 0x0c;
const DEVICE_STATUS: u64 = 0x14;
const QUEUE_SELECT: u64 = 0x16;
const QUEUE_SIZE: u64 = 0x18;
const QUEUE_NOTIFY_OFF: u64 = 0x1e;
const QUEUE_DESC: u64 = 0x20;
const QUEUE_DRIVER: u64 = 0x28;
const QUEUE_DEVICE: u64 = 0x30;
const QUEUE_ENABLE: u64 = 0x1c;

// Device status bits.
const S_ACK: u8 = 1;
const S_DRIVER: u8 = 2;
const S_DRIVER_OK: u8 = 4;
const S_FEATURES_OK: u8 = 8;

const VIRTQ_DESC_F_NEXT: u16 = 1;
const VIRTQ_DESC_F_WRITE: u16 = 2;

/// Control queue depth. Commands here are synchronous and one at a time (this is
/// a console, not a GPU workload), so a small ring is ample.
const QSIZE: u16 = 8;
/// Request and response scratch buffer size. The largest response is
/// `OK_DISPLAY_INFO` (24 + 16*24 = 408 bytes) and the largest request is an
/// attach-backing with a handful of entries.
const BUFSZ: usize = 1024;

/// Resource id for the scanout framebuffer. Any nonzero value works; 0 is
/// reserved by the spec to mean "no resource" (it *disables* a scanout).
const FB_RESOURCE: u32 = 1;

/// Bounded spin for a command response, so a wedged device degrades to "mode set
/// failed" instead of hanging the boot — the same rule every other wait in this
/// tree follows.
const CMD_SPINS: u32 = 2_000_000;

// --- MMIO accessors -----------------------------------------------------------
//
// On aarch64 each access is a **single** `ldr`/`str` via inline asm. Plain
// `read_volatile`/`write_volatile` lets LLVM coalesce adjacent accesses into a
// paired `ldp`/`stp`, which HVF cannot decode — it aborts the VM with
// `Assertion failed: (isv), function hvf_handle_exception`. This driver did exactly
// that and killed the guest right after `boot ok`; the standing rule in CLAUDE.md
// exists for this, and `arch/aarch64/virtio_net.rs` uses the same pattern.
#[cfg(target_arch = "aarch64")]
macro_rules! mmio {
    ($rname:ident, $wname:ident, $ty:ty, $ld:literal, $st:literal, $reg:literal) => {
        #[inline]
        unsafe fn $rname(a: u64) -> $ty {
            let v: $ty;
            unsafe {
                core::arch::asm!(
                    concat!($ld, " ", $reg, ", [{a}]"),
                    v = out(reg) v, a = in(reg) a,
                    options(nostack, preserves_flags)
                )
            };
            v
        }
        #[inline]
        unsafe fn $wname(a: u64, v: $ty) {
            unsafe {
                core::arch::asm!(
                    concat!($st, " ", $reg, ", [{a}]"),
                    v = in(reg) v, a = in(reg) a,
                    options(nostack, preserves_flags)
                )
            };
        }
    };
}

// The instruction must match the **access width** the device decodes. `ldr`/`str`
// with a `w` register is a 32-bit access, so using it for a byte or halfword
// register issues a 4-byte access at that address — which is also undecodable to
// HVF. Byte and halfword need `ldrb`/`strb` and `ldrh`/`strh`.
#[cfg(target_arch = "aarch64")]
mmio!(r8, w8, u8, "ldrb", "strb", "{v:w}");
#[cfg(target_arch = "aarch64")]
mmio!(r16, w16, u16, "ldrh", "strh", "{v:w}");
#[cfg(target_arch = "aarch64")]
mmio!(r32, w32, u32, "ldr", "str", "{v:w}");
#[cfg(target_arch = "aarch64")]
mmio!(r64, w64, u64, "ldr", "str", "{v:x}");

// x86 has no such hazard: a volatile access lowers to one instruction.
#[cfg(not(target_arch = "aarch64"))]
#[inline]
unsafe fn r8(a: u64) -> u8 {
    unsafe { read_volatile(a as *const u8) }
}
#[cfg(not(target_arch = "aarch64"))]
#[inline]
unsafe fn w8(a: u64, v: u8) {
    unsafe { write_volatile(a as *mut u8, v) }
}
#[cfg(not(target_arch = "aarch64"))]
#[inline]
unsafe fn r16(a: u64) -> u16 {
    unsafe { read_volatile(a as *const u16) }
}
#[cfg(not(target_arch = "aarch64"))]
#[inline]
unsafe fn w16(a: u64, v: u16) {
    unsafe { write_volatile(a as *mut u16, v) }
}
#[cfg(not(target_arch = "aarch64"))]
#[inline]
unsafe fn r32(a: u64) -> u32 {
    unsafe { read_volatile(a as *const u32) }
}
#[cfg(not(target_arch = "aarch64"))]
#[inline]
unsafe fn w32(a: u64, v: u32) {
    unsafe { write_volatile(a as *mut u32, v) }
}
#[cfg(not(target_arch = "aarch64"))]
#[inline]
unsafe fn w64(a: u64, v: u64) {
    unsafe { write_volatile(a as *mut u64, v) }
}

fn cfg_read8(d: &PciDevice, off: u16) -> u8 {
    (pci::read32(d.bus, d.dev, d.func, off & !3) >> ((off & 3) * 8)) as u8
}
fn cfg_read32_at(d: &PciDevice, off: u16) -> u32 {
    pci::read32(d.bus, d.dev, d.func, off & !3)
}

/// The control virtqueue: a split ring plus one request and one response buffer.
struct CtrlQueue {
    qsize: u16,
    desc: u64,
    avail: u64,
    used: u64,
    /// Request buffer (device-readable) and response buffer (device-writable).
    req: u64,
    req_phys: u64,
    resp: u64,
    resp_phys: u64,
    notify: u64,
    avail_idx: u16,
    used_last: u16,
}

impl CtrlQueue {
    /// Submit `cmd` as a two-descriptor chain and wait for the response.
    ///
    /// Descriptor 0 carries the request (device-readable) and chains to descriptor
    /// 1 for the reply (device-writable) — the shape every virtio-gpu command
    /// takes. Returns the response bytes actually written.
    unsafe fn exec(&mut self, cmd: &[u8]) -> Option<&[u8]> {
        unsafe {
            if cmd.len() > BUFSZ {
                return None;
            }
            core::ptr::copy_nonoverlapping(cmd.as_ptr(), self.req as *mut u8, cmd.len());
            // Zero the reply so a device that writes nothing can't look like a
            // stale success from the previous command.
            core::ptr::write_bytes(self.resp as *mut u8, 0, proto::CTRL_HDR_LEN);

            let d0 = self.desc;
            w64(d0, self.req_phys);
            w32(d0 + 8, cmd.len() as u32);
            w16(d0 + 12, VIRTQ_DESC_F_NEXT);
            w16(d0 + 14, 1); // -> descriptor 1

            let d1 = self.desc + 16;
            w64(d1, self.resp_phys);
            w32(d1 + 8, BUFSZ as u32);
            w16(d1 + 12, VIRTQ_DESC_F_WRITE);
            w16(d1 + 14, 0);

            let slot = self.avail_idx % self.qsize;
            w16(self.avail + 4 + slot as u64 * 2, 0); // head of the chain
            fence(Ordering::SeqCst);
            self.avail_idx = self.avail_idx.wrapping_add(1);
            w16(self.avail + 2, self.avail_idx);
            fence(Ordering::SeqCst);
            w16(self.notify, 0); // doorbell carries the virtqueue index

            let mut spins = 0u32;
            while r16(self.used + 2) == self.used_last {
                spins += 1;
                if spins > CMD_SPINS {
                    crate::ktrace::log("virtio-gpu", "command timed out");
                    return None;
                }
                core::hint::spin_loop();
            }
            fence(Ordering::SeqCst);
            let s = (self.used_last % self.qsize) as u64;
            let len = r32(self.used + 4 + s * 8 + 4) as usize;
            self.used_last = self.used_last.wrapping_add(1);
            Some(core::slice::from_raw_parts(self.resp as *const u8, len.min(BUFSZ)))
        }
    }
}

/// A virtio-gpu display device.
pub struct VirtioGpu {
    q: CtrlQueue,
    /// Device config space, for `events_read` / `num_scanouts`.
    devcfg: u64,
    num_scanouts: u32,
    /// The current scanout framebuffer, if a mode has been set.
    fb: Option<Fb>,
}

#[derive(Clone, Copy)]
struct Fb {
    virt: u64,
    phys: u64,
    w: u32,
    h: u32,
    pitch: u64,
}

// virtio_gpu_config offsets.
const CFG_EVENTS_READ: u64 = 0;
const CFG_EVENTS_CLEAR: u64 = 4;
const CFG_NUM_SCANOUTS: u64 = 8;

/// `VIRTIO_GPU_EVENT_DISPLAY` — the outputs changed (host window resized, or a
/// display was attached). This is the closest thing to a hot-plug-detect
/// interrupt available here, read from config space rather than an IRQ.
const EVENT_DISPLAY: u32 = 1 << 0;

impl VirtioGpu {
    /// Probe the PCI bus for a virtio-gpu and bring it up. `None` when absent.
    pub fn probe() -> Option<VirtioGpu> {
        // virtio PCI: vendor 0x1af4, modern device id = 0x1040 + virtio device id.
        // Matched by **vendor + device id**, not display class: virtio-gpu-pci
        // reports class 03:00 like every other VGA device, and claiming the wrong
        // one would take over a display we cannot drive. Modern id is
        // `0x1040 + <virtio device id>`; 0x1050 is the transitional alias.
        let want = 0x1040 + proto::VIRTIO_ID_GPU as u16;
        let mut found: Option<PciDevice> = None;
        pci::for_each(&mut |d: PciDevice| {
            if d.vendor == 0x1af4 && (d.device == want || d.device == 0x1050) {
                found = Some(d);
                return false; // stop the walk
            }
            true
        });
        Self::init(found?)
    }

    fn init(d: PciDevice) -> Option<VirtioGpu> {
        d.enable_bus_master();
        if cfg_read32_at(&d, 0x04) & (1 << 20) == 0 {
            return None; // no capability list → not a modern virtio device
        }
        let (mut common, mut notify_virt, mut notify_mult, mut devcfg) = (0u64, 0u64, 0u32, 0u64);
        let mut cap = cfg_read8(&d, 0x34) & 0xfc;
        let mut guard = 0;
        while cap != 0 && guard < 48 {
            guard += 1;
            let next = cfg_read8(&d, cap as u16 + 1) & 0xfc;
            if cfg_read8(&d, cap as u16) == CAP_VENDOR {
                let cfg_type = cfg_read8(&d, cap as u16 + 3);
                let bar = cfg_read8(&d, cap as u16 + 4);
                let offset = cfg_read32_at(&d, cap as u16 + 8);
                let bar_phys = d.bar(bar);
                if bar_phys != 0 {
                    let virt = crate::mm::map_mmio(bar_phys, 0x4000) + offset as u64;
                    match cfg_type {
                        CFG_COMMON => common = virt,
                        CFG_NOTIFY => {
                            notify_virt = virt;
                            notify_mult = cfg_read32_at(&d, cap as u16 + 16);
                        }
                        CFG_DEVICE => devcfg = virt,
                        _ => {}
                    }
                }
            }
            cap = next;
        }
        if common == 0 || notify_virt == 0 {
            return None;
        }

        // SAFETY: `common`/`notify_virt`/`devcfg` are mapped virtio BAR regions.
        unsafe {
            w8(common + DEVICE_STATUS, 0);
            let mut spins = 0u32;
            while r8(common + DEVICE_STATUS) != 0 {
                spins += 1;
                if spins > CMD_SPINS {
                    return None; // reset never completed; leave the device alone
                }
                core::hint::spin_loop();
            }
            w8(common + DEVICE_STATUS, S_ACK);
            w8(common + DEVICE_STATUS, S_ACK | S_DRIVER);

            // Features: only VIRTIO_F_VERSION_1 (bit 32) is required. EDID is
            // offered by some devices but not needed — the mode list comes from
            // GET_DISPLAY_INFO, which every virtio-gpu has.
            w32(common + DEVICE_FEATURE_SELECT, 0);
            let _lo = r32(common + DEVICE_FEATURE);
            w32(common + DRIVER_FEATURE_SELECT, 0);
            w32(common + DRIVER_FEATURE, 0);
            w32(common + DRIVER_FEATURE_SELECT, 1);
            w32(common + DRIVER_FEATURE, 1);
            w8(common + DEVICE_STATUS, S_ACK | S_DRIVER | S_FEATURES_OK);
            if r8(common + DEVICE_STATUS) & S_FEATURES_OK == 0 {
                return None;
            }

            // Control queue (index 0). The cursor queue is not used.
            w16(common + QUEUE_SELECT, 0);
            let qmax = r16(common + QUEUE_SIZE);
            if qmax == 0 {
                return None;
            }
            let qsize = QSIZE.min(qmax);
            w16(common + QUEUE_SIZE, qsize);
            let qs = qsize as usize;
            let (desc_phys, desc) = crate::mm::alloc_dma(qs * 16)?;
            let (avail_phys, avail) = crate::mm::alloc_dma(6 + qs * 2)?;
            let (used_phys, used) = crate::mm::alloc_dma(6 + qs * 8)?;
            let (req_phys, req) = crate::mm::alloc_dma(BUFSZ)?;
            let (resp_phys, resp) = crate::mm::alloc_dma(BUFSZ)?;
            w64(common + QUEUE_DESC, desc_phys);
            w64(common + QUEUE_DRIVER, avail_phys);
            w64(common + QUEUE_DEVICE, used_phys);
            let notify = notify_virt + r16(common + QUEUE_NOTIFY_OFF) as u64 * notify_mult as u64;
            w16(common + QUEUE_ENABLE, 1);
            w8(common + DEVICE_STATUS, S_ACK | S_DRIVER | S_FEATURES_OK | S_DRIVER_OK);

            let num_scanouts = if devcfg != 0 { r32(devcfg + CFG_NUM_SCANOUTS) } else { 1 };
            crate::ktrace::log_fmt(format_args!(
                "virtio-gpu: up (dev {:04x}), q{}, {} scanout(s)",
                d.device, qsize, num_scanouts
            ));
            Some(VirtioGpu {
                q: CtrlQueue {
                    qsize,
                    desc,
                    avail,
                    used,
                    req,
                    req_phys,
                    resp,
                    resp_phys,
                    notify,
                    avail_idx: 0,
                    used_last: 0,
                },
                devcfg,
                num_scanouts: num_scanouts.max(1),
                fb: None,
            })
        }
    }

    /// Run a command that returns no payload, checking the response is an `OK_*`.
    unsafe fn cmd_ok(&mut self, cmd: &[u8]) -> bool {
        unsafe {
            match self.q.exec(cmd) {
                Some(r) => proto::CtrlHdr::parse(r).map(|h| h.is_ok()).unwrap_or(false),
                None => false,
            }
        }
    }

    /// `GET_DISPLAY_INFO`, decoded.
    unsafe fn display_info(&mut self) -> Vec<proto::DisplayOne> {
        unsafe {
            self.q
                .exec(&proto::get_display_info())
                .and_then(proto::parse_display_info)
                .unwrap_or_default()
        }
    }
}

impl DisplayDriver for VirtioGpu {
    fn name(&self) -> &'static str {
        "virtio-gpu"
    }

    fn connectors(&mut self) -> Vec<Connector> {
        // SAFETY: the control queue and its buffers are the live DMA regions.
        let infos = unsafe { self.display_info() };
        proto::connectors_from_display_info(
            &infos,
            self.num_scanouts,
            crate::display::STANDARD_MODES,
        )
    }

    fn set_mode(&mut self, connector: u32, mode: Mode) -> Result<Scanout, &'static str> {
        if mode.w == 0 || mode.h == 0 {
            return Err("zero-sized mode");
        }
        let pitch = mode.w as u64 * 4;
        let bytes = (pitch * mode.h as u64) as usize;
        // A fresh framebuffer per mode set: the old one is a different size, and
        // the resource that references it is released below.
        let (phys, virt) = crate::mm::alloc_dma(bytes).ok_or("framebuffer alloc failed")?;
        // SAFETY: `bytes` freshly-allocated DMA bytes at `virt`; black is what the
        // compositor's redraw expects to paint over.
        unsafe { core::ptr::write_bytes(virt as *mut u8, 0, bytes) };

        let rect = proto::Rect::new(0, 0, mode.w, mode.h);
        // SAFETY: live control queue; each command is validated by its response.
        unsafe {
            // Replacing an existing resource: drop the old one first so the device
            // isn't holding pages we are about to stop using.
            if self.fb.is_some() {
                let _ = self.cmd_ok(&proto::resource_unref(FB_RESOURCE));
            }
            if !self.cmd_ok(&proto::resource_create_2d(FB_RESOURCE, mode.w, mode.h)) {
                return Err("RESOURCE_CREATE_2D refused");
            }
            // One entry: `alloc_dma` is physically contiguous.
            if !self.cmd_ok(&proto::resource_attach_backing(FB_RESOURCE, &[(phys, bytes as u32)])) {
                return Err("RESOURCE_ATTACH_BACKING refused");
            }
            if !self.cmd_ok(&proto::set_scanout(connector, FB_RESOURCE, rect)) {
                return Err("SET_SCANOUT refused");
            }
        }
        self.fb = Some(Fb { virt, phys, w: mode.w, h: mode.h, pitch });
        let (r_shift, g_shift, b_shift) = proto::FORMAT_B8G8R8X8_SHIFTS;
        Ok(Scanout {
            addr: virt as usize,
            pitch,
            w: mode.w,
            h: mode.h,
            bpp_bytes: 4,
            r_shift,
            g_shift,
            b_shift,
        })
    }

    fn flush(&mut self, x: u32, y: u32, w: u32, h: u32) {
        let Some(fb) = self.fb else { return };
        // Clip to the scanout: a damage rect computed against a stale size would
        // otherwise ask the device to read past our backing.
        let x0 = x.min(fb.w);
        let y0 = y.min(fb.h);
        let w = w.min(fb.w - x0);
        let h = h.min(fb.h - y0);
        if w == 0 || h == 0 {
            return;
        }
        let rect = proto::Rect::new(x0, y0, w, h);
        let offset = y0 as u64 * fb.pitch + x0 as u64 * 4;
        // SAFETY: live control queue; the rect is clipped to the attached backing.
        unsafe {
            let _ = self.cmd_ok(&proto::transfer_to_host_2d(FB_RESOURCE, rect, offset));
            let _ = self.cmd_ok(&proto::resource_flush(FB_RESOURCE, rect));
        }
    }

    fn poll_events(&mut self) -> bool {
        if self.devcfg == 0 {
            return false;
        }
        // SAFETY: `devcfg` is the mapped virtio_gpu_config region.
        unsafe {
            let ev = r32(self.devcfg + CFG_EVENTS_READ);
            if ev & EVENT_DISPLAY == 0 {
                return false;
            }
            // Acknowledge by writing the bits back to events_clear, or the device
            // reports the same change forever.
            w32(self.devcfg + CFG_EVENTS_CLEAR, ev);
            true
        }
    }
}
