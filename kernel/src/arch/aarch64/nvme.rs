//! aarch64 discovery wrapper for the shared NVMe core ([`crate::block::nvme`]).
//! Finds the NVMe function on the ACPI-discovered PCIe bus (`crate::pci`, class
//! 0x01/0x08/0x02), maps BAR0 as Device memory, and hands the mapped base + an
//! identity/HHDM DMA allocator to the arch-neutral driver.

use crate::arch::aarch64::dma_to_phys;
use crate::block::nvme::{Nvme, MMIO_SPAN};
use crate::block::Dma;
use crate::pci;
use alloc::alloc::{alloc_zeroed, Layout};

/// aarch64 DMA: a heap `alloc_zeroed` (4 KiB-aligned); `phys` via `dma_to_phys`
/// (identity on `-kernel`, `va - hhdm` under the UEFI stub).
fn dma_alloc(bytes: usize) -> Option<Dma> {
    let layout = Layout::from_size_align(bytes.max(1), 4096).ok()?;
    // SAFETY: non-zero layout; we check the returned pointer.
    let p = unsafe { alloc_zeroed(layout) };
    if p.is_null() {
        return None;
    }
    let virt = p as u64;
    Some(Dma { phys: dma_to_phys(virt), virt })
}

/// Probe the `n`-th NVMe disk on the PCIe bus. One controller is supported, but
/// it may expose several namespaces (VirtualBox presents each attached disk as
/// NSID 1, 2, …); `n` selects namespace `n + 1`. `None` once namespaces run out.
pub fn probe_nth(n: usize) -> Option<Nvme> {
    // NVMe class 0x01 (mass storage) / subclass 0x08 / prog-if 0x02.
    let d = pci::find_class(0x01, 0x08, 0x02)?;
    d.enable_bus_master();
    let regs = d.bar(0);
    if regs == 0 {
        return None;
    }
    let _ = MMIO_SPAN; // the identity Device block covers the whole BAR window
    crate::arch::aarch64::mmu::map_device_gib(regs);
    // SAFETY: `regs` is the BAR-mapped NVMe register block (Device memory).
    unsafe { Nvme::bringup(regs, dma_alloc, (n + 1) as u32) }
}
