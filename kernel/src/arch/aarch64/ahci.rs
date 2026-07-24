//! aarch64 discovery wrapper for the shared AHCI core ([`crate::block::ahci`]).
//! Finds the AHCI function on the ACPI-discovered PCIe bus (`crate::pci`, class
//! 0x01/0x06/0x01), maps ABAR (BAR5) as Device memory, and hands the mapped
//! base + an identity/HHDM DMA allocator to the arch-neutral driver.

use crate::arch::aarch64::dma_to_phys;
use crate::block::ahci::{Ahci, MMIO_SPAN};
use crate::block::Dma;
use crate::pci;
use alloc::alloc::{alloc_zeroed, Layout};

/// aarch64 DMA: a heap `alloc_zeroed` (4 KiB-aligned); `phys` via `dma_to_phys`.
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

/// Probe the `n`-th AHCI **disk**, counted across every AHCI controller on the
/// PCIe bus and every populated port on each — the aarch64 mirror of
/// `arch::x86_64::ahci::probe_nth`; see it for why the count comes first.
pub fn probe_nth(n: usize) -> Option<Ahci> {
    let mut base = 0usize; // global index of this controller's first disk
    let mut c = 0usize;
    while let Some(d) = pci::find_class_nth(0x01, 0x06, 0x01, c) {
        c += 1;
        d.enable_bus_master();
        let abar = d.bar(5); // AHCI ABAR is BAR5
        if abar == 0 {
            continue;
        }
        let _ = MMIO_SPAN; // the identity Device block covers the whole BAR window
        crate::arch::aarch64::mmu::map_device_gib(abar);
        // SAFETY: `abar` is the BAR-mapped HBA register block (Device memory).
        let count = unsafe { Ahci::present_count(abar) };
        if n < base + count {
            // SAFETY: as above; the port index is within the counted range.
            return unsafe { Ahci::bringup_nth(abar, dma_alloc, n - base) };
        }
        base += count;
    }
    None
}
