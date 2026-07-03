//! QEMU **ramfb** framebuffer for the aarch64 `virt` machine — the aarch64
//! counterpart to the Limine framebuffer on x86 (`CHITTI_OS_HANDOFF.md` Phase 7
//! framebuffer TUI). The `virt` kernel boots directly (`-kernel`, no
//! bootloader), so there is no Limine framebuffer response; instead QEMU's
//! `ramfb` device gives a simple linear framebuffer that the guest configures
//! through the **fw_cfg** device — no PCI, no bootloader.
//!
//! Flow: read the fw_cfg file directory over its DMA interface, find the
//! `etc/ramfb` entry's selector, allocate an `XRGB8888` framebuffer in
//! (identity-mapped) RAM, and DMA-write a `RAMFBCfg` pointing QEMU's scanout at
//! it. The shared [`framebuffer::Console`](crate::framebuffer) then renders into
//! it exactly as on x86.
//!
//! Cache coherency: under both HVF (shared physical cache hierarchy) and TCG
//! (no emulated cache) QEMU's scanout reads the same RAM the guest wrote, so no
//! explicit cache maintenance is required for the framebuffer.

use alloc::boxed::Box;
use alloc::vec;
use core::ptr::{read_volatile, write_volatile};

/// fw_cfg MMIO base on the QEMU `virt` machine.
const FW_CFG_BASE: usize = 0x0902_0000;
/// DMA address register (write the big-endian physical address of a
/// [`DmaAccess`] here to start a transfer).
const FW_CFG_DMA_ADDR: *mut u64 = (FW_CFG_BASE + 0x10) as *mut u64;

/// The fixed file-directory pseudo-file selector.
const FW_CFG_FILE_DIR: u16 = 0x0019;

// DMA control bits.
const CTL_ERROR: u32 = 0x01;
const CTL_READ: u32 = 0x02;
const CTL_SELECT: u32 = 0x08;
const CTL_WRITE: u32 = 0x10;

/// `DRM_FORMAT_XRGB8888` ('XR24').
const FOURCC_XRGB8888: u32 = 0x3432_5258;

/// Default framebuffer geometry.
const WIDTH: u64 = 1024;
const HEIGHT: u64 = 768;

/// The DMA command block QEMU reads. All fields are big-endian on the wire.
#[repr(C)]
struct DmaAccess {
    control: u32,
    length: u32,
    address: u64,
}

/// The ramfb configuration QEMU reads from the `etc/ramfb` file. Big-endian,
/// 28 bytes on the wire (QEMU reads exactly `length` bytes).
#[repr(C)]
struct RamfbCfg {
    addr: u64,
    fourcc: u32,
    flags: u32,
    width: u32,
    height: u32,
    stride: u32,
}

#[inline]
fn dsb() {
    // Ensure the DMA command block writes are visible before the trigger, and
    // the device's completion write is observed after.
    unsafe { core::arch::asm!("dsb sy", options(nomem, nostack, preserves_flags)) };
}

/// Run one fw_cfg DMA operation. `control` is in native endianness (converted
/// here); `phys` is the (identity-mapped) address of the transfer buffer.
/// Returns false on device error.
///
/// # Safety
/// `phys`/`length` must describe a valid, exclusively-owned buffer.
unsafe fn dma(control: u32, length: u32, phys: u64) -> bool {
    let mut acc = DmaAccess { control: control.to_be(), length: length.to_be(), address: phys.to_be() };
    let acc_pa = &mut acc as *mut DmaAccess as u64;
    dsb();
    // Writing the (big-endian) address of the command block triggers the DMA.
    unsafe { write_volatile(FW_CFG_DMA_ADDR, acc_pa.to_be()) };
    dsb();
    // Poll until QEMU clears control (0 = done) or sets the error bit.
    loop {
        let c = u32::from_be(unsafe { read_volatile(&acc.control as *const u32) });
        if c & CTL_ERROR != 0 {
            return false;
        }
        if c == 0 {
            return true;
        }
        core::hint::spin_loop();
    }
}

/// Scan the fw_cfg file directory for `etc/ramfb` and return its selector.
///
/// # Safety
/// Touches fw_cfg MMIO; call once during single-core boot.
unsafe fn find_ramfb_selector() -> Option<u16> {
    // First 4 bytes of the directory are a big-endian entry count.
    let mut count_be: u32 = 0;
    if !unsafe { dma((FW_CFG_FILE_DIR as u32) << 16 | CTL_SELECT | CTL_READ, 4, &mut count_be as *mut u32 as u64) } {
        return None;
    }
    let count = u32::from_be(count_be) as usize;
    if count == 0 || count > 4096 {
        return None;
    }
    // Each entry: { u32 size; u16 select; u16 reserved; char name[56] } = 64 bytes.
    // Continue reading from the current offset (no re-select).
    let mut entries = vec![0u8; count * 64];
    if !unsafe { dma(CTL_READ, (count * 64) as u32, entries.as_mut_ptr() as u64) } {
        return None;
    }
    for e in entries.chunks_exact(64) {
        let name_len = e[8..64].iter().position(|&b| b == 0).unwrap_or(56);
        if &e[8..8 + name_len] == b"etc/ramfb" {
            return Some(u16::from_be_bytes([e[4], e[5]]));
        }
    }
    None
}

/// Bring up the ramfb framebuffer. Returns `(addr, width, height, pitch)` on
/// success, or `None` if the device is absent (e.g. QEMU launched without
/// `-device ramfb`). The framebuffer is heap-allocated and intentionally leaked
/// (it lives for the life of the system).
///
/// # Safety
/// Must run once, on the boot core, after the heap is initialized.
pub unsafe fn init() -> Option<(usize, u64, u64, u64)> {
    let selector = unsafe { find_ramfb_selector() }?;

    let stride = WIDTH * 4;
    let fb: &'static mut [u8] = Box::leak(vec![0u8; (stride * HEIGHT) as usize].into_boxed_slice());
    // Identity-mapped RAM on `virt`, so the virtual address is the physical one.
    let fb_pa = fb.as_ptr() as u64;

    let mut cfg = RamfbCfg {
        addr: fb_pa.to_be(),
        fourcc: FOURCC_XRGB8888.to_be(),
        flags: 0u32.to_be(),
        width: (WIDTH as u32).to_be(),
        height: (HEIGHT as u32).to_be(),
        stride: (stride as u32).to_be(),
    };
    // 28 bytes on the wire (u64 + 5×u32), regardless of struct padding.
    let cfg_pa = &mut cfg as *mut RamfbCfg as u64;
    if !unsafe { dma((selector as u32) << 16 | CTL_SELECT | CTL_WRITE, 28, cfg_pa) } {
        return None;
    }
    crate::ktrace::log_fmt(format_args!("ramfb: {WIDTH}x{HEIGHT} XRGB8888 framebuffer at {fb_pa:#x} (selector {selector})"));
    Some((fb_pa as usize, WIDTH, HEIGHT, stride))
}
