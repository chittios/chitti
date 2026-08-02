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
use alloc::vec::Vec;
use core::ptr::write_volatile;

/// Every USB controller we brought up + enumerated (the Mac mini has two dwc3
/// controllers, one per Type-C port; a keyboard on one and a mouse dongle on the
/// other must both stay live). `poll_key`/`poll_mouse` fan out across all of
/// them; a single-controller platform (QEMU) just has one entry.
static XHCI: Locked<Vec<Xhci>> = Locked::new(Vec::new());

const DART_PAGE: usize = 0x4000; // 16 KiB DART page

/// Apple USB DMA translation state: ONE shared L1/L2 table (covers IOVA
/// 0..32 MiB, L1 index 0) + an IOVA bump allocator, plus every DART that points
/// at it. The DWC3 emits low IOVAs the DART translates to the high physical DMA
/// buffers — bypass, which makes the controller emit the high PA itself, faults
/// the periodic interrupt transfer (Host System Error) even though enumeration's
/// control transfers survive it. All controllers share one table (each DART maps
/// the same IOVA→PA; the bump allocator hands out non-overlapping IOVAs), so a
/// second controller reuses the table rather than clobbering the first's.
struct UsbDma {
    darts: [(usize, u32); 4], // (base, sid) — every controller's DARTs (2 each)
    ndarts: usize,
    l1_pa: u64,     // shared L1 table physical (all DARTs' TTBR point here)
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
    USB_DMA.with(|s| {
        // Create the shared L1/L2 table on the first controller; later
        // controllers reuse it (one IOVA→PA map for the whole USB subsystem).
        if s.is_none() {
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
            *s = Some(UsbDma { darts: [(0, 0); 4], ndarts: 0, l1_pa, l2_va: l2, next_iova: DART_PAGE as u64 });
        }
        let d = s.as_mut().unwrap();
        // Point each of THIS controller's DARTs at the shared L1 table (mirrored
        // translation): the dwc3 DMAs through all its DARTs, so any one left
        // unconfigured faults. Record them so `aa_alloc_apple` flushes them all.
        for &(base, sid) in darts {
            if d.ndarts >= d.darts.len() {
                break;
            }
            // SAFETY: base is the Device-mapped DART; sid from the FDT.
            let dart = unsafe { Dart::new(base, sid) };
            if !dart.set_translate(d.l1_pa) {
                crate::ktrace::log_fmt(format_args!("apple_usb: DART {base:#x} sid {sid} set_translate failed"));
                return false;
            }
            d.darts[d.ndarts] = (base, sid);
            d.ndarts += 1;
        }
        crate::ktrace::log_fmt(format_args!("apple_usb: DART translate ready ({} darts total)", d.ndarts));
        true
    })
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
            XHCI.with(|s| s.push(x));
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
        let ok = x.enumerate_keyboard();
        crate::ktrace::log_fmt(format_args!(
            "xhci(apple): {} ports, enum kbd={} mouse={}",
            x.port_count(),
            if x.has_keyboard() { "yes" } else { "no" },
            if x.has_mouse() { "yes" } else { "no" }
        ));
        XHCI.with(|s| s.push(x));
        return ok;
    }
    crate::ktrace::log("xhci(apple)", "controller not ready (bringup timed out)");
    false
}

/// Whether a USB HID keyboard was enumerated on any controller (INPUT boot line).
pub fn has_keyboard() -> bool {
    XHCI.with(|s| s.iter().any(|x| x.has_keyboard()))
}

/// Whether a USB HID pointer/tablet was enumerated on any controller.
pub fn has_mouse() -> bool {
    XHCI.with(|s| s.iter().any(|x| x.has_mouse()))
}

/// The next byte from any USB keyboard, if one is ready (checks every
/// controller — a keyboard may be on a different dwc3 than the mouse).
pub fn poll_key() -> Option<u8> {
    let b = XHCI.with(|s| {
        for x in s.iter_mut() {
            if let Some(b) = x.poll_key() {
                return Some(b);
            }
        }
        None
    });
    maybe_prune_msc_mounts();
    b
}

/// Drain USB HID mouse reports into `crate::mouse` from every controller
/// (no-op if none has a pointer).
pub fn poll_mouse() {
    XHCI.with(|s| {
        for x in s.iter_mut() {
            x.poll_mouse();
        }
    });
    maybe_prune_msc_mounts();
}

/// After releasing the xHCI lock: drop mounts whose USB disk vanished.
fn maybe_prune_msc_mounts() {
    if crate::xhci::take_msc_unplug_prune() {
        let n = crate::fs::mount::prune_missing_disks();
        if n > 0 {
            crate::ktrace::log_fmt(format_args!(
                "xhci: pruned {n} mount(s) after USB MSC disconnect"
            ));
        }
    }
}

/// USB Ethernet bulk transport. aarch64 keeps a **list** of controllers (Apple
/// exposes two dwc3 cores, one per Type-C port), so each of these walks the list and
/// uses whichever controller actually has the bulk endpoints — the adapter could be
/// plugged into either port.
pub fn usb_bulk_ready() -> bool {
    XHCI.with(|s| s.iter().any(|x| x.bulk_role() == Some(crate::xhci::BulkRole::Eth)))
}

/// Whether a USB mass-storage bulk pair is configured on any controller.
pub fn usb_msc_ready() -> bool {
    XHCI.with(|s| s.iter().any(|x| x.bulk_role() == Some(crate::xhci::BulkRole::Msc)))
}

/// Queue a bulk IN transfer on the controller holding the Ethernet adapter.
pub fn usb_bulk_arm_in() {
    XHCI.with(|s| {
        for x in s.iter_mut().filter(|x| x.bulk_role() == Some(crate::xhci::BulkRole::Eth)) {
            x.bulk_arm_in();
        }
    });
}

/// Collect a received frame from whichever Ethernet controller has one.
pub fn usb_bulk_take_in(out: &mut [u8]) -> Option<usize> {
    XHCI.with(|s| {
        for x in s.iter_mut().filter(|x| x.bulk_role() == Some(crate::xhci::BulkRole::Eth)) {
            if let Some(n) = x.bulk_take_in(out) {
                return Some(n);
            }
        }
        None
    })
}

/// Queue a frame on the controller holding the Ethernet adapter.
pub fn usb_bulk_send(data: &[u8]) -> bool {
    XHCI.with(|s| {
        for x in s.iter_mut().filter(|x| x.bulk_role() == Some(crate::xhci::BulkRole::Eth)) {
            if x.bulk_send(data) {
                return true;
            }
        }
        false
    })
}

/// Synchronous bulk OUT for MSC BOT on the controller holding the stick.
pub fn usb_bulk_sync_out(data: &[u8], timeout_ms: u64) -> bool {
    let r = XHCI.with(|s| {
        for x in s.iter_mut().filter(|x| x.bulk_role() == Some(crate::xhci::BulkRole::Msc)) {
            if x.bulk_sync_out(data, timeout_ms) {
                return true;
            }
        }
        false
    });
    maybe_prune_msc_mounts();
    r
}

/// Synchronous bulk IN for MSC BOT.
pub fn usb_bulk_sync_in(out: &mut [u8], timeout_ms: u64) -> Option<usize> {
    let r = XHCI.with(|s| {
        for x in s.iter_mut().filter(|x| x.bulk_role() == Some(crate::xhci::BulkRole::Msc)) {
            if let Some(n) = x.bulk_sync_in(out, timeout_ms) {
                return Some(n);
            }
        }
        None
    });
    maybe_prune_msc_mounts();
    r
}

// ── Bluetooth HCI ────────────────────────────────────────────────────────

pub fn bt_hci_ready() -> bool {
    XHCI.with(|s| s.iter().any(|x| x.has_bluetooth()))
}

pub fn bt_hci_cmd(cmd: &[u8], timeout_ms: u64) -> Option<alloc::vec::Vec<u8>> {
    XHCI.with(|s| {
        for x in s.iter_mut() {
            if x.has_bluetooth() {
                return x.bt_hci_cmd(cmd, timeout_ms);
            }
        }
        None
    })
}

pub fn bt_take_event(out: &mut [u8]) -> Option<usize> {
    XHCI.with(|s| {
        for x in s.iter_mut() {
            if let Some(n) = x.bt_take_event(out) {
                return Some(n);
            }
        }
        None
    })
}

pub fn bt_acl_send_sync(data: &[u8], timeout_ms: u64) -> bool {
    XHCI.with(|s| {
        for x in s.iter_mut() {
            if x.has_bluetooth() {
                return x.bt_acl_send_sync(data, timeout_ms);
            }
        }
        false
    })
}

pub fn bt_acl_recv(out: &mut [u8], timeout_ms: u64) -> Option<usize> {
    XHCI.with(|s| {
        for x in s.iter_mut() {
            if x.has_bluetooth() {
                return x.bt_acl_recv(out, timeout_ms);
            }
        }
        None
    })
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
