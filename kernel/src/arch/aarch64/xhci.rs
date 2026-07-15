//! aarch64 wrapper for the arch-neutral xHCI core (`crate::xhci`): find the
//! xHCI controller on the PCIe bus (via ACPI-discovered ECAM, `crate::pci`),
//! map its BAR, and hand the core the register base + an identity-map DMA
//! allocator. The USB HID keyboard is the real-hardware input path (ARM has no
//! PS/2); works on any platform that exposes xHCI over PCIe.

use crate::arch::aarch64::dma_to_phys;
use crate::mm::Locked;
use crate::pci;
use crate::xhci::Xhci;
use alloc::alloc::{alloc_zeroed, Layout};

static XHCI: Locked<Option<Xhci>> = Locked::new(None);

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
    if let Some(mut x) = Xhci::bringup(base, aa_alloc) {
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
