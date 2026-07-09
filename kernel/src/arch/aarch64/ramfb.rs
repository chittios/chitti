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
use alloc::vec::Vec;
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

/// Fallback framebuffer geometry, used only when the launcher did not provide a
/// resolution (see [`read_fbres`]). ramfb has no preferred-mode/EDID mechanism,
/// so the guest must choose a size; `xtask` passes the host display's size via
/// the `opt/chitti/fbres` fw_cfg file, and this is the last resort if it's
/// absent (e.g. a bare `-kernel` boot). Full HD: QEMU's display scales the
/// window (`zoom-to-fit`), so a large surface beats a tiny one that must be
/// stretched blurry.
const FALLBACK_WIDTH: u64 = 1920;
const FALLBACK_HEIGHT: u64 = 1080;

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

/// Cap on fw_cfg DMA polls. QEMU completes in a handful of iterations; without
/// a bound, platforms that have **no** fw_cfg at `0x09020000` (VirtualBox UEFI,
/// real hardware) spin forever after `write_volatile` to unbacked MMIO — the
/// shell hangs in `remote::boot_seed` / `read_opt_file` and never polls USB,
/// so the host keyboard queue overflows (`VERR_PDM_NO_QUEUE_ITEMS`).
const DMA_MAX_SPINS: u32 = 1_000_000;

/// Run one fw_cfg DMA operation. `control` is in native endianness (converted
/// here); `phys` is the (identity-mapped) address of the transfer buffer.
/// Returns false on device error, timeout, or missing fw_cfg.
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
    // Poll until QEMU clears control (0 = done) or sets the error bit. Bound
    // the wait so non-QEMU hosts return false instead of wedging the kernel.
    let mut spins = 0u32;
    loop {
        let c = u32::from_be(unsafe { read_volatile(&acc.control as *const u32) });
        if c & CTL_ERROR != 0 {
            return false;
        }
        if c == 0 {
            return true;
        }
        spins = spins.saturating_add(1);
        if spins > DMA_MAX_SPINS {
            return false;
        }
        core::hint::spin_loop();
    }
}

/// Scan the fw_cfg file directory for `name`, returning its `(selector, size)`.
///
/// # Safety
/// Touches fw_cfg MMIO; call once during single-core boot.
unsafe fn find_file(name: &[u8]) -> Option<(u16, u32)> {
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
        if &e[8..8 + name_len] == name {
            let size = u32::from_be_bytes([e[0], e[1], e[2], e[3]]);
            return Some((u16::from_be_bytes([e[4], e[5]]), size));
        }
    }
    None
}

/// Read the launcher-supplied RAM size (bytes) from the `opt/chitti/ramsize`
/// fw_cfg file, if present. Heap-free (stack buffers only) so it can run in
/// early boot before the allocator exists, and it scans the file directory one
/// 64-byte entry at a time to avoid a large allocation. `None` if fw_cfg has no
/// such file (e.g. a non-QEMU platform) or the value doesn't parse.
///
/// # Safety
/// Touches fw_cfg MMIO; call once during single-core boot.
pub unsafe fn read_ram_bytes() -> Option<u64> {
    let mut count_be: u32 = 0;
    if !unsafe { dma((FW_CFG_FILE_DIR as u32) << 16 | CTL_SELECT | CTL_READ, 4, &mut count_be as *mut u32 as u64) } {
        return None;
    }
    let count = u32::from_be(count_be) as usize;
    if count == 0 || count > 4096 {
        return None;
    }
    // Directory reads continue from the current offset; pull one 64-byte entry
    // { u32 size; u16 select; u16 rsvd; char name[56] } at a time.
    let mut sel: Option<(u16, u32)> = None;
    let mut e = [0u8; 64];
    for _ in 0..count {
        if !unsafe { dma(CTL_READ, 64, e.as_mut_ptr() as u64) } {
            return None;
        }
        let name_len = e[8..64].iter().position(|&b| b == 0).unwrap_or(56);
        if &e[8..8 + name_len] == b"opt/chitti/ramsize" {
            sel = Some((u16::from_be_bytes([e[4], e[5]]), u32::from_be_bytes([e[0], e[1], e[2], e[3]])));
            break;
        }
    }
    let (selector, size) = sel?;
    if size == 0 || size > 31 {
        return None;
    }
    let mut buf = [0u8; 32];
    if !unsafe { dma((selector as u32) << 16 | CTL_SELECT | CTL_READ, size, buf.as_mut_ptr() as u64) } {
        return None;
    }
    // Parse the leading decimal integer (bytes).
    let mut v: u64 = 0;
    let mut any = false;
    for &b in &buf[..size as usize] {
        if b.is_ascii_digit() {
            v = v.saturating_mul(10) + (b - b'0') as u64;
            any = true;
        } else if any {
            break;
        }
    }
    any.then_some(v)
}

/// Read a whole named fw_cfg file into a buffer.
///
/// # Safety
/// Touches fw_cfg MMIO; call when no concurrent fw_cfg use is in flight
/// (boot + single-threaded shell init are fine).
unsafe fn read_file(name: &[u8]) -> Option<Vec<u8>> {
    let (selector, size) = unsafe { find_file(name) }?;
    if size == 0 || size > 4096 {
        return None;
    }
    let mut buf = vec![0u8; size as usize];
    if !unsafe { dma((selector as u32) << 16 | CTL_SELECT | CTL_READ, size, buf.as_mut_ptr() as u64) } {
        return None;
    }
    Some(buf)
}

/// Read a launcher-supplied `opt/chitti/*` fw_cfg file after the heap is up
/// (e.g. `opt/chitti/model` for a boot-time `/model remote` seed). `None` if
/// the file is absent (non-QEMU, or the launcher did not publish it).
pub fn read_opt_file(name: &[u8]) -> Option<Vec<u8>> {
    // SAFETY: shell init / single-threaded; no concurrent fw_cfg DMA.
    unsafe { read_file(name) }
}

/// The framebuffer resolution the launcher wants, from the `opt/chitti/fbres`
/// fw_cfg file (an ASCII `"WIDTHxHEIGHT"` string `xtask` derives from the host
/// display). `None` if absent/malformed, in which case the fallback is used.
///
/// # Safety
/// Touches fw_cfg MMIO; call once during single-core boot.
unsafe fn read_fbres() -> Option<(u64, u64)> {
    let buf = unsafe { read_file(b"opt/chitti/fbres") }?;
    let s = core::str::from_utf8(&buf).ok()?.trim();
    let (w, h) = s.split_once('x')?;
    let w: u64 = w.trim().parse().ok()?;
    let h: u64 = h.trim().parse().ok()?;
    // Sanity clamp: at least VGA, at most 8K.
    if (320..=7680).contains(&w) && (240..=4320).contains(&h) {
        Some((w, h))
    } else {
        None
    }
}

/// Bring up the ramfb framebuffer. Returns `(addr, width, height, pitch)` on
/// success, or `None` if the device is absent (e.g. QEMU launched without
/// `-device ramfb`). The framebuffer is heap-allocated and intentionally leaked
/// (it lives for the life of the system).
///
/// # Safety
/// Must run once, on the boot core, after the heap is initialized.
pub unsafe fn init() -> Option<(usize, u64, u64, u64)> {
    let (selector, _) = unsafe { find_file(b"etc/ramfb") }?;

    // Resolution from the launcher (host display size) or the fallback — never
    // a single baked-in constant.
    let (width, height) = unsafe { read_fbres() }.unwrap_or((FALLBACK_WIDTH, FALLBACK_HEIGHT));
    let stride = width * 4;
    let fb: &'static mut [u8] = Box::leak(vec![0u8; (stride * height) as usize].into_boxed_slice());
    // Identity-mapped RAM on `virt`, so the virtual address is the physical one.
    let fb_pa = fb.as_ptr() as u64;

    let mut cfg = RamfbCfg {
        addr: fb_pa.to_be(),
        fourcc: FOURCC_XRGB8888.to_be(),
        flags: 0u32.to_be(),
        width: (width as u32).to_be(),
        height: (height as u32).to_be(),
        stride: (stride as u32).to_be(),
    };
    // 28 bytes on the wire (u64 + 5×u32), regardless of struct padding.
    let cfg_pa = &mut cfg as *mut RamfbCfg as u64;
    if !unsafe { dma((selector as u32) << 16 | CTL_SELECT | CTL_WRITE, 28, cfg_pa) } {
        return None;
    }
    crate::ktrace::log_fmt(format_args!("ramfb: {width}x{height} XRGB8888 framebuffer at {fb_pa:#x} (selector {selector})"));
    Some((fb_pa as usize, width, height, stride))
}
