//! x86 wrapper for the arch-neutral xHCI core (`crate::xhci`): find the PCI
//! xHCI controller via legacy config ports, map its MMIO through the HHDM, and
//! hand the core the register base + `mm::alloc_dma` as the DMA allocator.

use crate::arch::x86_64::port::{inl, outl};
use crate::mm::{alloc_dma, map_mmio_page, Locked};
use crate::xhci::Xhci;

static XHCI: Locked<Option<Xhci>> = Locked::new(None);

/// Probe + bring up the xHCI controller and enumerate HID keyboard + pointer.
/// No-op if absent. Called once at boot on x86.
pub fn init_global() -> bool {
    if let Some(mmio) = discover() {
        if let Some(mut x) = Xhci::bringup(mmio, x86_alloc) {
            let ok = x.enumerate_keyboard();
            XHCI.with(|s| *s = Some(x));
            return ok;
        }
    }
    false
}

/// Whether a USB HID keyboard was enumerated.
pub fn has_keyboard() -> bool {
    XHCI.with(|s| s.as_ref().map(|x| x.has_keyboard()).unwrap_or(false))
}

/// Whether a USB HID pointer/tablet was enumerated.
pub fn has_mouse() -> bool {
    XHCI.with(|s| s.as_ref().map(|x| x.has_mouse()).unwrap_or(false))
}

/// The next byte from a USB keyboard, if any. `None` if no controller/keyboard.
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

/// Whether a USB Ethernet adapter's bulk endpoints are configured.
pub fn usb_bulk_ready() -> bool {
    XHCI.with(|s| {
        s.as_ref()
            .map(|x| x.bulk_role() == Some(crate::xhci::BulkRole::Eth))
            .unwrap_or(false)
    })
}

/// Whether a USB mass-storage bulk pair is configured.
pub fn usb_msc_ready() -> bool {
    XHCI.with(|s| {
        s.as_ref()
            .map(|x| x.bulk_role() == Some(crate::xhci::BulkRole::Msc))
            .unwrap_or(false)
    })
}

/// Queue a bulk IN transfer if none is outstanding, so a frame can arrive.
pub fn usb_bulk_arm_in() {
    XHCI.with(|s| {
        if let Some(x) = s.as_mut() {
            if x.bulk_role() == Some(crate::xhci::BulkRole::Eth) {
                x.bulk_arm_in();
            }
        }
    });
}

/// Collect a received frame, if a bulk IN transfer has completed.
pub fn usb_bulk_take_in(out: &mut [u8]) -> Option<usize> {
    XHCI.with(|s| {
        s.as_mut().and_then(|x| {
            if x.bulk_role() == Some(crate::xhci::BulkRole::Eth) {
                x.bulk_take_in(out)
            } else {
                None
            }
        })
    })
}

/// Queue a frame for transmission; false if a transfer is still outstanding.
pub fn usb_bulk_send(data: &[u8]) -> bool {
    XHCI.with(|s| {
        s.as_mut()
            .map(|x| {
                x.bulk_role() == Some(crate::xhci::BulkRole::Eth) && x.bulk_send(data)
            })
            .unwrap_or(false)
    })
}

/// Synchronous bulk OUT for MSC BOT.
pub fn usb_bulk_sync_out(data: &[u8], timeout_ms: u64) -> bool {
    XHCI.with(|s| {
        s.as_mut()
            .map(|x| {
                x.bulk_role() == Some(crate::xhci::BulkRole::Msc) && x.bulk_sync_out(data, timeout_ms)
            })
            .unwrap_or(false)
    })
}

/// Synchronous bulk IN for MSC BOT.
pub fn usb_bulk_sync_in(out: &mut [u8], timeout_ms: u64) -> Option<usize> {
    XHCI.with(|s| {
        s.as_mut().and_then(|x| {
            if x.bulk_role() == Some(crate::xhci::BulkRole::Msc) {
                x.bulk_sync_in(out, timeout_ms)
            } else {
                None
            }
        })
    })
}

/// `mm::alloc_dma` adapted to the core's `(phys, virt)` allocator shape.
fn x86_alloc(bytes: usize) -> Option<(u64, usize)> {
    alloc_dma(bytes).map(|(pa, va)| (pa, va as usize))
}

fn pci_addr(bus: u8, slot: u8, func: u8, off: u8) -> u32 {
    0x8000_0000 | ((bus as u32) << 16) | ((slot as u32) << 11) | ((func as u32) << 8) | ((off as u32) & 0xfc)
}
fn cfg_read32(bus: u8, slot: u8, func: u8, off: u8) -> u32 {
    // SAFETY: standard PCI config ports.
    unsafe {
        outl(0xcf8, pci_addr(bus, slot, func, off));
        inl(0xcfc)
    }
}
fn cfg_write32(bus: u8, slot: u8, func: u8, off: u8, v: u32) {
    // SAFETY: standard PCI config ports.
    unsafe {
        outl(0xcf8, pci_addr(bus, slot, func, off));
        outl(0xcfc, v);
    }
}

/// Find the xHCI controller (class 0x0C/0x03/0x30), enable it, map its MMIO
/// (32 KiB) through the HHDM, and return the register base. `None` if absent.
fn discover() -> Option<usize> {
    let (bus, slot, func) = find_xhci()?;
    crate::ktrace::log_fmt(format_args!("xhci: controller at {bus:02x}:{slot:02x}.{func}"));
    let bar0 = cfg_read32(bus, slot, func, 0x10);
    if bar0 & 0x1 != 0 {
        return None; // I/O BAR — not xHCI MMIO
    }
    let mut phys = (bar0 & 0xffff_fff0) as u64;
    if (bar0 >> 1) & 0x3 == 0x2 {
        phys |= (cfg_read32(bus, slot, func, 0x14) as u64) << 32;
    }
    let cmd = cfg_read32(bus, slot, func, 0x04);
    cfg_write32(bus, slot, func, 0x04, cmd | 0b110); // memory space + bus master
    let mmio = map_mmio_page(phys) as usize;
    for i in 1u64..8 {
        map_mmio_page(phys + i * 0x1000);
    }
    Some(mmio)
}

/// Scan PCI (buses 0..=1) for an xHCI controller (class 0x0C/0x03/0x30).
fn find_xhci() -> Option<(u8, u8, u8)> {
    for bus in 0u8..=1 {
        for slot in 0u8..32 {
            let id = cfg_read32(bus, slot, 0, 0x00);
            if id == 0xffff_ffff {
                continue;
            }
            let header = (cfg_read32(bus, slot, 0, 0x0c) >> 16) & 0xff;
            let nfuncs = if header & 0x80 != 0 { 8 } else { 1 };
            for func in 0..nfuncs {
                if cfg_read32(bus, slot, func, 0x00) == 0xffff_ffff {
                    continue;
                }
                let class = cfg_read32(bus, slot, func, 0x08);
                if (class >> 24) & 0xff == 0x0c && (class >> 16) & 0xff == 0x03 && (class >> 8) & 0xff == 0x30 {
                    return Some((bus, slot, func));
                }
            }
        }
    }
    None
}
