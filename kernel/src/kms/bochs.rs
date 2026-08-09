//! Bochs VBE (`dispi`) display driver — QEMU's **standard VGA**, PCI `1234:1111`.
//!
//! The KMS backend for the adapter you get by default. Linux's equivalent is
//! `drivers/gpu/drm/tiny/bochs.c`, and this follows it closely because the value
//! of this device is that its behaviour is already pinned down by a driver
//! everyone runs.
//!
//! All the arithmetic and the mode-set ordering live in [`super::bochs_regs`],
//! which is compiled under `cargo xtask test`; this module is the part that
//! needs a device and is therefore `cfg(not(test))`.
//!
//! ## Two transports, and MMIO is the one that matters
//!
//! The dispi registers are reachable either through the x86 I/O ports
//! `0x01CE/0x01CF` or through the device's MMIO BAR. **MMIO is preferred**, and
//! not as a style choice: aarch64 has no I/O ports at all, so the port path
//! cannot exist there, and a driver that only worked on x86 is exactly the
//! divergence the dual-architecture rule forbids.
//!
//! Unlike the VMSVGA backend — which declines its MMIO path because that register
//! layout was a guess that mis-programmed a real display — this layout is not
//! guessed: registers at BAR+0x500 and EDID at the base of the window, as Linux
//! has them.
//! Which transport is used is decided by reading the id register back through it,
//! so a wrong guess about the window declines instead of writing into it.

use super::bochs_regs as reg;
use super::{Connector, DisplayDriver, Mode, Scanout};
use alloc::string::String;
use alloc::vec::Vec;

#[cfg(target_arch = "aarch64")]
use crate::pci::{self, PciDevice};
#[cfg(target_arch = "x86_64")]
use crate::arch::x86_64::pci::{self, PciDevice};

/// 32-bit XRGB is the only depth this drives. The compositor's blitters assume
/// 4-byte pixels throughout, so offering 8/16/24 would mean a second set of them
/// for no gain on any machine we run on.
const BPP: u16 = 32;
const BPP_BYTES: u32 = 4;

// aarch64 MMIO must be a *single* load/store of the width the device decodes.
// LLVM otherwise merges adjacent volatile accesses into a pair, which HVF cannot
// decode (`hvf: isv`) — the same rule every other MMIO site here follows. These
// registers are 16-bit, so `ldrh`/`strh`.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn mmio_r16(a: u64) -> u16 {
    let v: u32;
    unsafe {
        core::arch::asm!("ldrh {v:w}, [{a}]", v = out(reg) v, a = in(reg) a,
                         options(nostack, preserves_flags))
    };
    v as u16
}
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn mmio_w16(a: u64, v: u16) {
    unsafe {
        core::arch::asm!("strh {v:w}, [{a}]", v = in(reg) v as u32, a = in(reg) a,
                         options(nostack, preserves_flags))
    };
}
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn mmio_r8(a: u64) -> u8 {
    let v: u32;
    unsafe {
        core::arch::asm!("ldrb {v:w}, [{a}]", v = out(reg) v, a = in(reg) a,
                         options(nostack, preserves_flags))
    };
    v as u8
}
#[cfg(not(target_arch = "aarch64"))]
#[inline]
unsafe fn mmio_r16(a: u64) -> u16 {
    unsafe { core::ptr::read_volatile(a as *const u16) }
}
#[cfg(not(target_arch = "aarch64"))]
#[inline]
unsafe fn mmio_w16(a: u64, v: u16) {
    unsafe { core::ptr::write_volatile(a as *mut u16, v) }
}
#[cfg(not(target_arch = "aarch64"))]
#[inline]
unsafe fn mmio_r8(a: u64) -> u8 {
    unsafe { core::ptr::read_volatile(a as *const u8) }
}

/// How the dispi index/data pair is reached.
enum Regs {
    /// The device's MMIO BAR; registers at `base + 0x500 + index*2`.
    Mmio { base: u64 },
    /// x86 I/O ports: write the index to 0x01CE, then read/write 0x01CF.
    Port,
}

impl Regs {
    fn read(&self, index: u16) -> u16 {
        match self {
            Regs::Mmio { base } => {
                // SAFETY: `base` is this device's own mapped BAR and `index` is a
                // dispi register, so the address is inside the 0x2000 window.
                unsafe { mmio_r16(base + reg::mmio_offset(index)) }
            }
            #[cfg(target_arch = "x86_64")]
            Regs::Port => {
                // SAFETY: the architectural Bochs VBE port pair.
                unsafe {
                    crate::arch::x86_64::port::outw(reg::IOPORT_INDEX, index);
                    crate::arch::x86_64::port::inw(reg::IOPORT_DATA)
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            Regs::Port => 0,
        }
    }

    fn write(&self, index: u16, val: u16) {
        match self {
            Regs::Mmio { base } => {
                // SAFETY: as `read`.
                unsafe { mmio_w16(base + reg::mmio_offset(index), val) }
            }
            #[cfg(target_arch = "x86_64")]
            Regs::Port => {
                // SAFETY: as `read`.
                unsafe {
                    crate::arch::x86_64::port::outw(reg::IOPORT_INDEX, index);
                    crate::arch::x86_64::port::outw(reg::IOPORT_DATA, val);
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            Regs::Port => {}
        }
    }
}

pub struct Bochs {
    regs: Regs,
    /// Mapped linear framebuffer (BAR0) and its size.
    fb: u64,
    vram: u64,
    /// EDID base block, when the device published a valid one.
    edid: Option<Vec<u8>>,
    /// The display's preferred resolution from that EDID.
    preferred: Option<(u32, u32)>,
    cur: Option<Mode>,
}

impl Bochs {
    pub fn probe() -> Option<Bochs> {
        let mut found: Option<PciDevice> = None;
        pci::for_each(&mut |d: PciDevice| {
            // By vendor+device id, never display class: several unrelated
            // adapters report class 03:00, and driving one of those through these
            // registers writes into whatever they happen to decode there.
            if d.vendor == reg::VENDOR_QEMU && d.device == reg::DEVICE_STDVGA {
                found = Some(d);
                return false;
            }
            true
        });
        Self::init(found?)
    }

    fn init(d: PciDevice) -> Option<Bochs> {
        // BAR0 is the linear framebuffer, BAR2 the register/EDID window.
        let fb_phys = d.bar(0);
        if fb_phys == 0 {
            crate::ktrace::log("bochs", "no linear framebuffer BAR — declining");
            return None;
        }
        // **The transport is chosen by whether the id register reads back through
        // it**, not by measuring the BAR. Sizing a BAR means writing all-ones to
        // it and reading the mask back, and at probe time the console is still
        // drawing into the firmware's framebuffer — the same reason the VMSVGA
        // and I2C drivers identify by reading only. The id check below is a
        // strictly better test anyway: it proves the registers are actually there.
        let mmio_phys = d.bar(2);
        let mut regs = None;
        if mmio_phys != 0 {
            let cand = Regs::Mmio { base: crate::mm::map_mmio(mmio_phys, reg::MMIO_WINDOW_LEN as usize) };
            if reg::is_bochs_id(cand.read(reg::INDEX_ID)) {
                regs = Some(cand);
            }
        }
        if regs.is_none() && cfg!(target_arch = "x86_64") {
            // No usable MMIO window, but on x86 the port pair is architectural.
            let cand = Regs::Port;
            if reg::is_bochs_id(cand.read(reg::INDEX_ID)) {
                regs = Some(cand);
            }
        }
        let Some(regs) = regs else {
            crate::ktrace::log(
                "bochs",
                "no dispi registers over MMIO or ports — declining",
            );
            return None;
        };
        let id = regs.read(reg::INDEX_ID);
        let vram = reg::vram_bytes(regs.read(reg::INDEX_VIDEO_MEMORY_64K));
        if vram == 0 {
            crate::ktrace::log("bochs", "device reports no video memory — declining");
            return None;
        }

        // QEMU publishes the display's EDID in the same BAR when `edid=on` (its
        // default). Reading it means `/display` can name the monitor's own
        // preferred mode instead of guessing from a standard list — the same
        // source the loader used, so the two agree.
        let (edid, preferred) = match &regs {
            Regs::Mmio { base } => read_edid(*base),
            Regs::Port => (None, None),
        };

        let fb = crate::mm::map_mmio(fb_phys, vram as usize);
        crate::ktrace::log_fmt(format_args!(
            "bochs: id {:#06x} vram {} MiB fb {:#x} regs {} edid {}",
            id,
            vram / (1024 * 1024),
            fb_phys,
            match regs {
                Regs::Mmio { .. } => "mmio",
                Regs::Port => "port",
            },
            match preferred {
                Some((w, h)) => alloc::format!("{w}x{h}"),
                None => String::from("none"),
            }
        ));
        Some(Bochs { regs, fb, vram, edid, preferred, cur: None })
    }
}

/// Read and validate the EDID block the device publishes at `base + 0x600`.
///
/// Validated rather than trusted: with `edid=off` that window is whatever the
/// device leaves there, and an unchecked read would hand the mode picker a
/// preferred resolution invented out of uninitialised memory. `edid::is_valid`
/// checks the header and the checksum, which is the same gate the loader applies.
fn read_edid(base: u64) -> (Option<Vec<u8>>, Option<(u32, u32)>) {
    let mut buf = alloc::vec![0u8; crate::edid::BASE_BLOCK_LEN];
    for (i, b) in buf.iter_mut().enumerate() {
        // SAFETY: inside the 0x2000 BAR — 0x600 + 128 is well under it.
        *b = unsafe { mmio_r8(base + reg::MMIO_EDID_BASE + i as u64) };
    }
    if !crate::edid::is_valid(&buf) {
        return (None, None);
    }
    let pref = crate::edid::preferred_resolution(&buf);
    (Some(buf), pref)
}

impl DisplayDriver for Bochs {
    fn name(&self) -> &'static str {
        "bochs"
    }

    fn connectors(&mut self) -> Vec<Connector> {
        // One output. The dispi interface has no multi-head concept at all, so
        // reporting one is the honest answer rather than inventing connectors
        // that cannot be programmed.
        let modes: Vec<Mode> = reg::usable_modes(self.vram, BPP_BYTES, self.preferred)
            .into_iter()
            .map(|(w, h)| Mode::new(w, h))
            .collect();
        alloc::vec![Connector {
            id: 0,
            name: String::from("Virtual-1"),
            connected: true,
            preferred: self.preferred.map(|(w, h)| Mode::new(w, h)),
            modes,
            edid: self.edid.clone(),
        }]
    }

    fn set_mode(&mut self, _connector: u32, mode: Mode) -> Result<Scanout, &'static str> {
        // Refused, not attempted. The device programs an oversized mode happily
        // and then scans out memory that is not there, so the bound has to be
        // checked here.
        if !reg::mode_fits(mode.w, mode.h, BPP_BYTES, self.vram) {
            return Err("mode does not fit the device's video memory");
        }
        for (index, val) in reg::modeset_sequence(mode.w, mode.h, BPP) {
            self.regs.write(index, val);
        }
        // Read the geometry back: the device is entitled to have clamped it, and
        // believing our own write would leave the compositor drawing at a
        // geometry the scanout does not have.
        let w = self.regs.read(reg::INDEX_XRES) as u32;
        let h = self.regs.read(reg::INDEX_YRES) as u32;
        if w == 0 || h == 0 {
            return Err("device reported no geometry after the mode set");
        }
        if (w, h) != (mode.w, mode.h) {
            crate::ktrace::log_fmt(format_args!(
                "bochs: asked for {}x{}, device gave {w}x{h}",
                mode.w, mode.h
            ));
        }
        // The stride is what we asked for in pixels; derive the byte pitch from
        // what the device actually accepted, not from the request.
        let pitch = reg::pitch_bytes(self.regs.read(reg::INDEX_VIRT_WIDTH).max(w as u16) as u32, BPP_BYTES);
        if pitch * h as u64 > self.vram {
            return Err("accepted geometry does not fit video memory");
        }
        self.cur = Some(Mode::new(w, h));
        Ok(Scanout {
            addr: self.fb as usize,
            pitch,
            w,
            h,
            bpp_bytes: BPP_BYTES as u64,
            // The LFB is XRGB8888 little-endian: blue lowest.
            r_shift: 16,
            g_shift: 8,
            b_shift: 0,
        })
    }

    // `flush` stays the default no-op: this device scans out of the linear
    // framebuffer continuously, so a write to it is already on screen. That is
    // the opposite of virtio-gpu (which needs a transfer) and of VMSVGA in SVGA
    // mode (which needs a FIFO update), and it is why neither of those could
    // leave this as the default.
}
