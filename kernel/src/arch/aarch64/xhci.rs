//! aarch64 wrapper for the arch-neutral xHCI core (`crate::xhci`): find the
//! xHCI controller on the PCIe bus (via ACPI-discovered ECAM, `crate::pci`),
//! map its BAR, and hand the core the register base + an identity-map DMA
//! allocator. The USB HID keyboard is the real-hardware input path (ARM has no
//! PS/2); works on any platform that exposes xHCI over PCIe.

use crate::arch::aarch64::dart::{self, Dart};
use crate::arch::aarch64::dma_to_phys;
use crate::mm::Locked;
use crate::pci;
use crate::xhci::Xhci;
use alloc::alloc::{alloc_zeroed, Layout};
use core::ptr::write_volatile;

static XHCI: Locked<Option<Xhci>> = Locked::new(None);

const DART_PAGE: usize = 0x4000; // 16 KiB DART page

/// Apple USB DMA translation state: the DART + its single L2 table (covers IOVA
/// 0..32 MiB, L1 index 0) + an IOVA bump allocator. The DWC3 emits low IOVAs the
/// DART translates to the high physical DMA buffers — bypass, which makes the
/// controller emit the high PA itself, faults the periodic interrupt transfer
/// (Host System Error) even though enumeration's control transfers survive it.
struct UsbDma {
    darts: [(usize, u32); 2], // (base, sid) — all the controller's DARTs
    ndarts: usize,
    l2_va: usize,   // shared 2048-entry L2 table (16 KiB), maps IOVA 0..32 MiB
    next_iova: u64, // bump, 16 KiB pages
}
static USB_DMA: Locked<Option<UsbDma>> = Locked::new(None);

/// Clean a cache range to the Point of Coherency so the DART's (non-coherent)
/// table walker reads the PTEs we just wrote.
fn dcache_clean(va: usize, len: usize) {
    // SAFETY: cache maintenance over a mapped Normal-memory page-table buffer.
    unsafe {
        let mut p = va & !63;
        let end = va + len;
        while p < end {
            core::arch::asm!("dc cvac, {}", in(reg) p, options(nostack, preserves_flags));
            p += 64;
        }
        core::arch::asm!("dsb sy", options(nostack, preserves_flags));
    }
}

/// Set up DART translation for the USB stream: allocate a 2-level page table
/// (one L1 + one L2, covering IOVA 0..32 MiB), point the DART at it, and record
/// state for [`aa_alloc_apple`]. Call before [`attach_at`]. Returns false on
/// allocation failure or a locked DART.
pub fn dma_translate_setup(darts: &[(usize, u32)]) -> bool {
    if darts.is_empty() {
        return false;
    }
    let Ok(layout) = Layout::from_size_align(DART_PAGE, DART_PAGE) else { return false };
    // SAFETY: nonzero 16 KiB layout; leaked, used as DART page tables.
    let l1 = unsafe { alloc_zeroed(layout) } as usize;
    let l2 = unsafe { alloc_zeroed(layout) } as usize;
    if l1 == 0 || l2 == 0 {
        return false;
    }
    let l1_pa = dma_to_phys(l1 as u64);
    let l2_pa = dma_to_phys(l2 as u64);
    // L1[0] → L2 table (a PTE, same encoding as a leaf).
    // SAFETY: l1 is a fresh 16 KiB table.
    unsafe { write_volatile(l1 as *mut u64, dart::make_pte(l2_pa)) };
    dcache_clean(l1, 8);
    // Point EVERY DART at the SAME L1 table (mirrored translation): the dwc3
    // DMAs through all its DARTs, so any one left unconfigured faults.
    let mut saved = [(0usize, 0u32); 2];
    let mut n = 0;
    for &(base, sid) in darts.iter().take(saved.len()) {
        // SAFETY: base is the Device-mapped DART; sid from the FDT.
        let dart = unsafe { Dart::new(base, sid) };
        if !dart.set_translate(l1_pa) {
            crate::serial_println!("apple_usb: DART {base:#x} sid {sid} set_translate failed");
            return false;
        }
        saved[n] = (base, sid);
        n += 1;
    }
    USB_DMA.with(|s| *s = Some(UsbDma { darts: saved, ndarts: n, l2_va: l2, next_iova: DART_PAGE as u64 }));
    crate::serial_println!("apple_usb: DART translate ready ({n} darts, l1={l1_pa:#x} l2={l2_pa:#x})");
    true
}

/// DMA allocator for the Apple USB path: allocate a 16 KiB-aligned buffer, map
/// its physical page(s) to fresh low IOVAs in the USB DART, and return
/// `(IOVA, VA)`. The controller uses the IOVA (translated by the DART); the CPU
/// uses VA (== PA on our identity map). `None` if translation isn't set up.
fn aa_alloc_apple(bytes: usize) -> Option<(u64, usize)> {
    USB_DMA.with(|s| {
        let d = s.as_mut()?;
        let sz = bytes.max(1);
        let layout = Layout::from_size_align(sz, DART_PAGE).ok()?;
        // SAFETY: nonzero layout; leaked, device-shared DMA.
        let va = unsafe { alloc_zeroed(layout) } as usize;
        if va == 0 {
            return None;
        }
        let pa = dma_to_phys(va as u64);
        let pages = sz.div_ceil(DART_PAGE);
        let iova = d.next_iova;
        for i in 0..pages {
            let this = iova + (i * DART_PAGE) as u64;
            let (l1, l2, _) = dart::iova_split(this);
            if l1 != 0 {
                return None; // our single L2 covers only L1==0 (0..32 MiB)
            }
            // SAFETY: l2_va is the live L2 table; entry l2 is in range [0,2048).
            unsafe { write_volatile((d.l2_va + l2 * 8) as *mut u64, dart::make_pte(pa + (i * DART_PAGE) as u64)) };
        }
        dcache_clean(d.l2_va, DART_PAGE); // publish the PTEs to the DART walker
        d.next_iova += (pages * DART_PAGE) as u64;
        // Flush every DART's TLB so the new IOVAs are live in all of them.
        for &(base, sid) in &d.darts[..d.ndarts] {
            // SAFETY: base/sid recorded at setup.
            unsafe { Dart::new(base, sid) }.flush_tlb();
        }
        Some((iova, va))
    })
}

/// Probe + bring up the xHCI controller and enumerate HID keyboard + pointer.
/// No-op if there is no PCIe bus (virtio-mmio-only QEMU) or no xHCI controller.
/// Returns true when **at least one** of keyboard / mouse came up (boot
/// diagnostics use [`has_keyboard`] / [`has_mouse`] separately).
pub fn init_global() -> bool {
    if let Some(mmio) = discover() {
        if let Some(mut x) = Xhci::bringup(mmio, aa_alloc) {
            let ok = x.enumerate_keyboard();
            let kbd = x.has_keyboard();
            let mse = x.has_mouse();
            crate::ktrace::log_fmt(format_args!(
                "xhci: enum done kbd={} mouse={}",
                if kbd { "yes" } else { "no" },
                if mse { "yes" } else { "no" }
            ));
            XHCI.with(|s| *s = Some(x));
            return ok;
        }
    }
    false
}

/// Bring up an xHCI controller at an already-known register base (the Apple
/// dwc3 xHCI window at DWC3_base+0x0, after `apple_usb` has powered the PHY,
/// reset the core into HOST mode, and put the DART in bypass) and enumerate HID.
/// Reuses the same `XHCI` static + `aa_alloc` (identity DMA, valid under DART
/// bypass) so `poll_key`/`poll_mouse` work unchanged. Returns whether a keyboard
/// or mouse came up.
pub fn attach_at(base: usize) -> bool {
    // The Apple DWC3 DMAs through the DART with translated low IOVAs, so use the
    // DART-mapping allocator (set up by apple_usb via dma_translate_setup).
    if let Some(mut x) = Xhci::bringup(base, aa_alloc_apple) {
        // Visible on the chat pane (bare boot has no serial, ktrace pane closed).
        crate::serial_println!(
            "apple_usb: xHCI bringup OK ({} ports); enumerating…",
            x.port_count()
        );
        let ok = x.enumerate_keyboard();
        crate::serial_println!(
            "apple_usb: xHCI enum kbd={} mouse={}",
            if x.has_keyboard() { "yes" } else { "no" },
            if x.has_mouse() { "yes" } else { "no" }
        );
        XHCI.with(|s| *s = Some(x));
        return ok;
    }
    crate::serial_println!("apple_usb: xHCI bringup FAILED (controller not ready) — see ktrace for the exact wait that timed out");
    false
}

/// Whether a USB HID keyboard was enumerated (for the INPUT boot line).
pub fn has_keyboard() -> bool {
    XHCI.with(|s| s.as_ref().map(|x| x.has_keyboard()).unwrap_or(false))
}

/// Whether a USB HID pointer/tablet was enumerated (for the INPUT boot line).
pub fn has_mouse() -> bool {
    XHCI.with(|s| s.as_ref().map(|x| x.has_mouse()).unwrap_or(false))
}

/// The next byte from a USB keyboard, if any.
pub fn poll_key() -> Option<u8> {
    XHCI.with(|s| s.as_mut().and_then(|x| x.poll_key()))
}

/// Drain USB HID mouse reports into `crate::mouse` (no-op if no USB mouse).
pub fn poll_mouse() {
    XHCI.with(|s| {
        if let Some(x) = s.as_mut() {
            x.poll_mouse();
        }
    });
}

/// Page-aligned identity DMA: VA == PA on the aarch64 identity map (or via
/// `dma_to_phys` under the HHDM handoff). Returns `(phys, virt)`.
fn aa_alloc(bytes: usize) -> Option<(u64, usize)> {
    let layout = Layout::from_size_align(bytes.max(1), 4096).ok()?;
    // SAFETY: nonzero layout; leaked, used only as device-shared DMA.
    let va = unsafe { alloc_zeroed(layout) } as u64;
    if va == 0 {
        return None;
    }
    Some((dma_to_phys(va), va as usize))
}

/// Find the xHCI controller (class 0x0C/0x03/0x30) on PCIe, enable it, and map
/// its BAR0's 1 GiB Device block. Returns the register base. `None` if absent.
fn discover() -> Option<usize> {
    let d = pci::find_class(0x0c, 0x03, 0x30)?;
    crate::ktrace::log_fmt(format_args!("xhci: controller at {:02x}:{:02x}.{} on PCIe", d.bus, d.dev, d.func));
    d.enable_bus_master();
    let bar = d.bar(0);
    if bar == 0 {
        return None;
    }
    crate::arch::aarch64::mmu::map_device_gib(bar);
    Some(bar as usize)
}
