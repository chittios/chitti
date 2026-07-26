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
        crate::ktrace::log_fmt(format_args!(
            "vmsvga: up (id {:#x}, {} regs), fb {:#x} ({} KiB), max {}x{}",
            agreed,
            regs.kind(),
            fb_phys,
            fb_size / 1024,
            max.0,
            max.1
        ));
        Some(VmSvga { regs, fb_virt, fb_phys, fb_size, max, cur: None })
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
        let cur = self.cur.unwrap_or(Mode::new(self.max.0, self.max.1));
        let mut modes = alloc::vec![cur];
        for &(w, h) in crate::display::STANDARD_MODES {
            let m = Mode::new(w, h);
            if m != cur && w <= self.max.0 && h <= self.max.1 {
                modes.push(m);
            }
        }
        alloc::vec![Connector {
            id: 0,
            name: alloc::string::String::from("SVGA-1"),
            connected: true,
            preferred: Some(cur),
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

    // `flush` is intentionally the default no-op: this framebuffer is scanned out
    // continuously, so a write is on screen already. (The FIFO update command
    // exists for the accelerated paths this driver does not use.)
}
