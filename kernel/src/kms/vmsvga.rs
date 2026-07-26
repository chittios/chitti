//! **VMware SVGA II** (`vmsvga`) display driver — the [`DisplayDriver`] backend
//! for **VirtualBox** and QEMU's `vmware-svga`.
//!
//! Structurally much simpler than virtio-gpu: mode setting is a handful of
//! indexed register writes, and the framebuffer is a **linear surface in the
//! device's VRAM that is scanned out continuously** — so once a mode is set the
//! compositor writes straight at the screen and [`Self::flush`] is a no-op. There
//! is no transfer/present step and therefore no damage tracking, which is the
//! opposite of virtio-gpu.
//!
//! Register access is over BAR0, and *how* depends on the platform:
//!
//! * **x86** exposes BAR0 as an **I/O** BAR — index/value at ports `base+0`/`base+1`
//!   (dword-wide `out`/`in`). This is what QEMU's `vmware-svga` does.
//! * **aarch64** has no port I/O at all, so a VMware-SVGA implementation there must
//!   expose BAR0 as **memory**, and the same index/value pair becomes two dword
//!   MMIO slots.
//!
//! Both are handled by [`Regs`], decided from the BAR's own type bit rather than by
//! `cfg(target_arch)` — a device that reports a memory BAR on x86 works, and so does
//! an ARM hypervisor that reports one. **Caveat:** only the x86/port path has been
//! exercised here (QEMU emulates `vmware-svga` on x86 only), so the MMIO path is
//! reasoned-about, not tested. It logs which transport it picked so a wrong guess is
//! visible in one boot rather than presenting as a black screen.

use super::{Connector, DisplayDriver, Mode, Scanout};
use alloc::vec::Vec;
use core::ptr::{read_volatile, write_volatile};

// Single-instruction MMIO on aarch64. Adjacent `read_volatile`/`write_volatile` can
// be coalesced by LLVM into a paired access HVF cannot decode, aborting the VM with
// `hvf: isv` — and this matters even though the driver is currently declined,
// because **probing still reads these registers**, and VirtualBox-ARM is exactly the
// platform that would expose BAR0 as memory.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn mmio_r32(a: u64) -> u32 {
    let v: u32;
    unsafe {
        core::arch::asm!("ldr {v:w}, [{a}]", v = out(reg) v, a = in(reg) a,
                         options(nostack, preserves_flags))
    };
    v
}
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn mmio_w32(a: u64, v: u32) {
    unsafe {
        core::arch::asm!("str {v:w}, [{a}]", v = in(reg) v, a = in(reg) a,
                         options(nostack, preserves_flags))
    };
}
#[cfg(not(target_arch = "aarch64"))]
#[inline]
unsafe fn mmio_r32(a: u64) -> u32 {
    unsafe { read_volatile(a as *const u32) }
}
#[cfg(not(target_arch = "aarch64"))]
#[inline]
unsafe fn mmio_w32(a: u64, v: u32) {
    unsafe { write_volatile(a as *mut u32, v) }
}

#[cfg(target_arch = "aarch64")]
use crate::pci::{self, PciDevice};
#[cfg(target_arch = "x86_64")]
use crate::arch::x86_64::pci::{self, PciDevice};

/// Whether to drive the device when BAR0 is memory-mapped rather than an I/O BAR.
///
/// Off: that path cannot be tested here (QEMU emulates `vmware-svga` on x86 only,
/// where BAR0 is I/O), and enabling it on a guess mis-programmed a real
/// VirtualBox-ARM display. Verify the register layout on the target first.
const VMSVGA_ALLOW_MMIO: bool = false;

/// VMware's PCI vendor id, and the SVGA II device id.
const VENDOR_VMWARE: u16 = 0x15AD;
const DEVICE_SVGA2: u16 = 0x0405;

// Indexed registers (written to the index port, then read/written via value).
const REG_ID: u32 = 0;
const REG_ENABLE: u32 = 1;
const REG_WIDTH: u32 = 2;
const REG_HEIGHT: u32 = 3;
const REG_MAX_WIDTH: u32 = 4;
const REG_MAX_HEIGHT: u32 = 5;
const REG_BITS_PER_PIXEL: u32 = 7;
const REG_BYTES_PER_LINE: u32 = 12;
const REG_FB_START: u32 = 13;
const REG_FB_OFFSET: u32 = 14;
const REG_VRAM_SIZE: u32 = 15;
const REG_FB_SIZE: u32 = 16;
const REG_CAPABILITIES: u32 = 17;
/// FIFO size in bytes (the BAR2 window).
const REG_MEM_SIZE: u32 = 19;
/// Set once the FIFO area is validly configured. **Without this the device stays in
/// VGA mode and ignores the mode registers entirely.**
const REG_CONFIG_DONE: u32 = 20;
/// Write to ask the device to process the FIFO now.
const REG_SYNC: u32 = 21;
/// Nonzero while the device is still working through the FIFO.
const REG_BUSY: u32 = 22;

/// `SVGA_CAP_EXTENDED_FIFO` — the FIFO carries a register area the device writes
/// into, which the guest must reserve by setting `MIN` past it.
const CAP_EXTENDED_FIFO: u32 = 0x0000_8000;

// FIFO layout, as **dword indices** into the BAR2 window.
const FIFO_MIN: u64 = 0;
const FIFO_MAX: u64 = 1;
const FIFO_NEXT_CMD: u64 = 2;
const FIFO_STOP: u64 = 3;
/// Size of the extended FIFO register area, in dwords (`SVGA_FIFO_NUM_REGS`).
const FIFO_NUM_REGS: u32 = 291;

/// `SVGA_CMD_UPDATE` — "this rectangle of the framebuffer changed". Followed by
/// x, y, width, height.
const CMD_UPDATE: u32 = 1;

/// Bounded spin while waiting for the device to consume FIFO space.
const FIFO_DRAIN_SPINS: u32 = 200_000;

/// The device validates the FIFO before accepting `CONFIG_DONE`, and one of the
/// checks is that there is at least 10 KiB of command space (`max - min`).
const FIFO_MIN_CMD_BYTES: u32 = 10 * 1024;

/// Version handshake values. The driver writes the highest it supports and reads
/// back what the device agreed to; anything below `ID_0` means this is not a
/// VMware SVGA at all.
const SVGA_MAGIC: u32 = 0x0090_0000;
const fn svga_id(ver: u32) -> u32 {
    SVGA_MAGIC << 8 | ver
}
const ID_2: u32 = svga_id(2);
const ID_1: u32 = svga_id(1);
const ID_0: u32 = svga_id(0);

/// How BAR0's index/value pair is reached.
enum Regs {
    /// x86 I/O ports: `base + 0` is the index, `base + 1` the value.
    Port { base: u16 },
    /// Memory-mapped: the same pair as two dword slots.
    Mmio { base: u64 },
}

impl Regs {
    fn write(&self, reg: u32, val: u32) {
        match self {
            #[cfg(target_arch = "x86_64")]
            Regs::Port { base } => {
                // SAFETY: `base` is this device's own I/O BAR; these two ports are
                // the documented SVGA index/value pair.
                unsafe {
                    crate::arch::x86_64::port::outl(*base, reg);
                    crate::arch::x86_64::port::outl(base + 1, val);
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            Regs::Port { .. } => {}
            Regs::Mmio { base } => {
                // SAFETY: `base` is the mapped BAR0 window; index then value.
                unsafe {
                    mmio_w32(*base, reg);
                    mmio_w32(*base + 4, val);
                }
            }
        }
    }

    fn read(&self, reg: u32) -> u32 {
        match self {
            #[cfg(target_arch = "x86_64")]
            Regs::Port { base } => {
                // SAFETY: as `write` — the device's own index/value ports.
                unsafe {
                    crate::arch::x86_64::port::outl(*base, reg);
                    crate::arch::x86_64::port::inl(base + 1)
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            Regs::Port { .. } => 0,
            Regs::Mmio { base } => {
                // SAFETY: as `write`.
                unsafe {
                    mmio_w32(*base, reg);
                    mmio_r32(*base + 4)
                }
            }
        }
    }

    fn kind(&self) -> &'static str {
        match self {
            Regs::Port { .. } => "io-port",
            Regs::Mmio { .. } => "mmio",
        }
    }
}

/// A VMware SVGA II display device.
pub struct VmSvga {
    regs: Regs,
    /// Guest-visible framebuffer (already mapped) and its physical base.
    fb_virt: u64,
    fb_phys: u64,
    fb_size: u32,
    /// Mapped FIFO (BAR2) and the ring bounds, in bytes.
    fifo: u64,
    fifo_min: u32,
    fifo_max: u32,
    /// Whether the ring has been published and `CONFIG_DONE` accepted. Deferred out
    /// of probe so binding the driver cannot disturb a live console.
    configured: bool,
    max: (u32, u32),
    cur: Option<Mode>,
}

impl VmSvga {
    /// Probe the PCI bus for a VMware SVGA II and bring it up.
    pub fn probe() -> Option<VmSvga> {
        let mut found: Option<PciDevice> = None;
        pci::for_each(&mut |d: PciDevice| {
            if d.vendor == VENDOR_VMWARE && d.device == DEVICE_SVGA2 {
                found = Some(d);
                return false;
            }
            true
        });
        Self::init(found?)
    }

    fn init(d: PciDevice) -> Option<VmSvga> {
        d.enable_bus_master();
        // BAR0 raw, so an I/O BAR is visible (`PciDevice::bar` returns 0 for those).
        let bar0 = pci::read32(d.bus, d.dev, d.func, 0x10);
        let regs = if bar0 & 1 != 0 {
            Regs::Port { base: (bar0 & 0xffff_fffc) as u16 }
        } else {
            // **Refused by default.** The index/value pair over MMIO is the branch no
            // emulator here can exercise — QEMU only provides `vmware-svga` on x86,
            // where BAR0 is an I/O BAR — and this register layout is a guess. Acting
            // on a guess drove a VirtualBox-ARM display into a wrong geometry, so the
            // safe answer is to decline and leave the firmware framebuffer alone. The
            // whole KMS layer is optional; losing mode setting costs a feature,
            // getting the registers wrong costs the console.
            //
            // To enable: confirm on the target that BAR0 really is an index/value pair
            // at +0/+4, then flip `VMSVGA_ALLOW_MMIO`.
            if !VMSVGA_ALLOW_MMIO {
                crate::ktrace::log(
                    "vmsvga",
                    "BAR0 is memory-mapped; that transport is unverified — declining",
                );
                crate::serial_println!(
                    "kms> vmsvga found but its BAR0 is MMIO (unverified transport) -- keeping the firmware framebuffer"
                );
                return None;
            }
            let phys = (bar0 & 0xffff_fff0) as u64;
            if phys == 0 {
                return None;
            }
            Regs::Mmio { base: crate::mm::map_mmio(phys, 0x1000) }
        };

        // Version handshake: offer the highest, accept what it agrees to. A device
        // that echoes none of them is not an SVGA II and must be left alone.
        let mut agreed = 0;
        for id in [ID_2, ID_1, ID_0] {
            regs.write(REG_ID, id);
            if regs.read(REG_ID) == id {
                agreed = id;
                break;
            }
        }
        if agreed == 0 {
            crate::ktrace::log("vmsvga", "no ID handshake — not an SVGA II");
            return None;
        }

        let max = (regs.read(REG_MAX_WIDTH), regs.read(REG_MAX_HEIGHT));
        let fb_phys = regs.read(REG_FB_START) as u64;
        let fb_size = regs.read(REG_FB_SIZE).max(regs.read(REG_VRAM_SIZE));
        if fb_phys == 0 || fb_size == 0 || max.0 == 0 || max.1 == 0 {
            crate::ktrace::log("vmsvga", "device reported no framebuffer");
            return None;
        }
        let fb_virt = crate::mm::map_mmio(fb_phys, fb_size as usize);

        // --- FIFO geometry: **read only** -----------------------------------------
        //
        // Probing must not change the device. `CONFIG_DONE` and the ring pointers
        // alter how the device scans out, and at probe time the console is still
        // drawing into the *firmware's* framebuffer — so writing them here moved the
        // scanout out from under a live console and produced a display offset and
        // clipped on VirtualBox. Same rule the I2C/EC drivers follow: identification
        // only ever reads. The ring is configured lazily on the first real mode set
        // (`ensure_fifo`).
        let fifo_phys = d.bar(2);
        let fifo_size = regs.read(REG_MEM_SIZE);
        if fifo_phys == 0 || fifo_size == 0 {
            crate::ktrace::log("vmsvga", "no FIFO (BAR2/MEM_SIZE) — cannot leave VGA mode");
            return None;
        }
        let caps = regs.read(REG_CAPABILITIES);
        // With the extended FIFO the device writes its own registers into the front of
        // the ring, so `MIN` must sit past them or command data and those registers
        // overwrite each other.
        let fifo_min = if caps & CAP_EXTENDED_FIFO != 0 { FIFO_NUM_REGS * 4 } else { 4 * 4 };
        let fifo_max = fifo_size;
        if fifo_max < fifo_min + FIFO_MIN_CMD_BYTES {
            crate::ktrace::log_fmt(format_args!(
                "vmsvga: FIFO too small ({fifo_size} bytes, need {fifo_min} + 10 KiB)"
            ));
            return None;
        }
        let fifo = crate::mm::map_mmio(fifo_phys, fifo_size as usize);

        crate::ktrace::log_fmt(format_args!(
            "vmsvga: up (id {:#x}, {} regs), fb {:#x} ({} KiB), fifo {:#x} ({} KiB, min {}), caps {:#x}, max {}x{}",
            agreed,
            regs.kind(),
            fb_phys,
            fb_size / 1024,
            fifo_phys,
            fifo_size / 1024,
            fifo_min,
            caps,
            max.0,
            max.1
        ));
        // Seed the current mode from the device's own geometry registers. Without
        // this `preferred` fell back to MAX_WIDTH/MAX_HEIGHT — the largest surface
        // the VRAM could hold (an odd 2368x1770 on QEMU), which is a ceiling, not a
        // mode anyone chose, and it would be what a KMS-only boot came up in.
        // Seed the current mode from the framebuffer the console is **actually** using,
        // and only fall back to the device's geometry registers.
        //
        // The order matters and is counter-intuitive: those registers do not report
        // the mode in effect. Before the guest has ever enabled SVGA mode they hold
        // the device's own defaults — QEMU answers 640x480 — while the display is
        // really at whatever the firmware programmed through the VGA path. Trusting
        // them made `preferred` 640x480 on a 1024x768 console, which is the mode a
        // KMS-only boot would then have come up in.
        let cur = crate::framebuffer::physical_size()
            .filter(|&(w, h)| w > 0 && h > 0 && w <= max.0 && h <= max.1)
            .map(|(w, h)| Mode::new(w, h))
            .or_else(|| {
                let (cw, ch) = (regs.read(REG_WIDTH), regs.read(REG_HEIGHT));
                (cw > 0 && ch > 0 && cw <= max.0 && ch <= max.1).then(|| Mode::new(cw, ch))
            });
        Some(VmSvga { regs, fb_virt, fb_phys, fb_size, fifo, fifo_min, fifo_max, configured: false, max, cur })
    }
}

impl DisplayDriver for VmSvga {
    fn name(&self) -> &'static str {
        "vmsvga"
    }

    fn connectors(&mut self) -> Vec<Connector> {
        // One output: SVGA II's multi-monitor support is a separate (screen-object)
        // feature this driver does not use, so reporting one is the honest answer
        // rather than inventing connectors we cannot program.
        let mut modes = Vec::new();
        if let Some(cur) = self.cur {
            modes.push(cur);
        }
        for &(w, h) in crate::display::STANDARD_MODES {
            let m = Mode::new(w, h);
            if Some(m) != self.cur && w <= self.max.0 && h <= self.max.1 {
                modes.push(m);
            }
        }
        alloc::vec![Connector {
            id: 0,
            name: alloc::string::String::from("SVGA-1"),
            connected: true,
            preferred: self.cur,
            modes,
            edid: None,
        }]
    }

    fn set_mode(&mut self, _connector: u32, mode: Mode) -> Result<Scanout, &'static str> {
        if mode.w == 0 || mode.h == 0 {
            return Err("zero-sized mode");
        }
        if mode.w > self.max.0 || mode.h > self.max.1 {
            return Err("mode exceeds the device maximum");
        }
        // Take the device out of VGA mode now, not at probe.
        self.ensure_fifo()?;
        // Disable across the change. Writing geometry to an already-enabled device
        // leaves it scanning out with the *previous* configuration, and the
        // registers read back stale — which is how this first produced a screen
        // showing the console four times side by side.
        self.regs.write(REG_ENABLE, 0);
        self.regs.write(REG_WIDTH, mode.w);
        self.regs.write(REG_HEIGHT, mode.h);
        self.regs.write(REG_BITS_PER_PIXEL, 32);
        self.regs.write(REG_ENABLE, 1);

        // Read the geometry back — the device decides the real pitch, which need not
        // be `width * 4` (VRAM alignment), so computing it would tear.
        let mut pitch = self.regs.read(REG_BYTES_PER_LINE) as u64;
        let offset = self.regs.read(REG_FB_OFFSET) as u64;
        let w = self.regs.read(REG_WIDTH);
        let h = self.regs.read(REG_HEIGHT);
        if w == 0 || h == 0 {
            return Err("device reported no geometry after the mode set");
        }
        // **Validate rather than trust.** A pitch below one row of pixels cannot be
        // right, and using it makes every scanline wrap — the compositor would draw
        // the screen several times across instead of once, which is not obviously a
        // pitch bug when you see it. Fall back to the packed minimum and say so.
        let min_pitch = w as u64 * 4;
        if pitch < min_pitch {
            crate::ktrace::log_fmt(format_args!(
                "vmsvga: bogus BYTES_PER_LINE {pitch} for {w}px — using {min_pitch}"
            ));
            pitch = min_pitch;
        }
        if offset + pitch * h as u64 > self.fb_size as u64 {
            return Err("mode does not fit the device framebuffer");
        }
        self.cur = Some(Mode::new(w, h));
        let _ = self.fb_phys;
        Ok(Scanout {
            addr: (self.fb_virt + offset) as usize,
            pitch,
            w,
            h,
            bpp_bytes: 4,
            // SVGA II's 32-bpp layout is `XRGB` little-endian, the same packing the
            // compositor already uses — so no colour conversion anywhere.
            r_shift: 16,
            g_shift: 8,
            b_shift: 0,
        })
    }

    /// Present a dirty rectangle by queueing `SVGA_CMD_UPDATE`.
    ///
    /// **Not** a no-op, which is the trap here: in VGA mode the device tracks
    /// framebuffer writes, but once it is in SVGA mode (`ENABLE` + `CONFIG_DONE`) it
    /// only repaints from FIFO commands. Drawing without this leaves the screen
    /// frozen on whatever was there when the mode was set.
    fn flush(&mut self, x: u32, y: u32, w: u32, h: u32) {
        let Some(cur) = self.cur else { return };
        // Clip to the scanout: a damage rect computed against a stale size would ask
        // the device to read past the surface.
        let x0 = x.min(cur.w);
        let y0 = y.min(cur.h);
        let w = w.min(cur.w - x0);
        let h = h.min(cur.h - y0);
        if w == 0 || h == 0 {
            return;
        }
        if self.fifo_write(&[CMD_UPDATE, x0, y0, w, h]) {
            // Ask the device to process what we just queued, so the update lands now
            // rather than on its next refresh tick.
            self.regs.write(REG_SYNC, 1);
        }
    }
}

impl VmSvga {
    /// Publish the ring and take the device out of VGA mode. Idempotent.
    ///
    /// Deliberately **not** done at probe: this is the write that changes the
    /// scanout, so it happens only when a mode set is actually requested.
    fn ensure_fifo(&mut self) -> Result<(), &'static str> {
        if self.configured {
            return Ok(());
        }
        // SAFETY: `fifo` is the mapped BAR2 window, at least `fifo_max` bytes.
        unsafe {
            mmio_w32(self.fifo + FIFO_MIN * 4, self.fifo_min);
            mmio_w32(self.fifo + FIFO_MAX * 4, self.fifo_max);
            mmio_w32(self.fifo + FIFO_NEXT_CMD * 4, self.fifo_min);
            mmio_w32(self.fifo + FIFO_STOP * 4, self.fifo_min);
        }
        self.regs.write(REG_CONFIG_DONE, 1);
        if self.regs.read(REG_CONFIG_DONE) == 0 {
            return Err("device refused CONFIG_DONE (FIFO rejected)");
        }
        self.configured = true;
        Ok(())
    }

    /// Bytes free in the command ring, i.e. how far `NEXT_CMD` may advance before it
    /// would run into `STOP` (the device's read pointer).
    fn fifo_free(&self) -> u32 {
        // SAFETY: `fifo` is the mapped BAR2 window.
        let (next, stop) = unsafe {
            (mmio_r32(self.fifo + FIFO_NEXT_CMD * 4), mmio_r32(self.fifo + FIFO_STOP * 4))
        };
        let span = self.fifo_max - self.fifo_min;
        if span == 0 {
            return 0;
        }
        // One word is always left unused so full and empty stay distinguishable.
        if next >= stop {
            span - (next - stop) - 4
        } else {
            (stop - next) - 4
        }
    }

    /// Append `words` to the command ring, wrapping at `MAX`.
    ///
    /// Drains first if the ring is too full, with a **bounded** wait: a wedged device
    /// must cost a dropped frame, not a hung console.
    fn fifo_write(&mut self, words: &[u32]) -> bool {
        let need = (words.len() * 4) as u32;
        let mut spins = 0u32;
        while self.fifo_free() < need {
            if spins == 0 {
                self.regs.write(REG_SYNC, 1); // ask it to consume
            }
            spins += 1;
            if spins > FIFO_DRAIN_SPINS || self.regs.read(REG_BUSY) == 0 {
                // Either it will not drain, or it is idle and still has no room —
                // both mean this update is dropped. The next damage flush retries.
                crate::ktrace::log("vmsvga", "FIFO full; dropping an update");
                return false;
            }
            core::hint::spin_loop();
        }
        // SAFETY: `fifo` is the mapped BAR2 window and every offset below is kept
        // inside [fifo_min, fifo_max) by the wrap.
        unsafe {
            let mut next = mmio_r32(self.fifo + FIFO_NEXT_CMD * 4);
            for &word in words {
                mmio_w32(self.fifo + next as u64, word);
                next += 4;
                if next >= self.fifo_max {
                    next = self.fifo_min;
                }
            }
            // Publish the new write pointer only after the whole command is in
            // place, or the device can consume a half-written command.
            core::sync::atomic::fence(core::sync::atomic::Ordering::SeqCst);
            mmio_w32(self.fifo + FIFO_NEXT_CMD * 4, next);
        }
        true
    }
}
