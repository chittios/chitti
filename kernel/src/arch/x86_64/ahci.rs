//! x86 discovery wrapper for the shared AHCI core ([`crate::block::ahci`]).
//! Finds the AHCI function on the legacy PCI bus (`arch::x86_64::pci`, class
//! 0x01/0x06/0x01), maps ABAR (BAR5) into the HHDM, and hands the mapped base +
//! the frame-allocator DMA source to the arch-neutral driver.

use crate::arch::x86_64::pci;
use crate::block::ahci::{Ahci, MMIO_SPAN};
use crate::block::Dma;

/// x86 DMA: a physically-contiguous frame-allocator region reached via the HHDM.
fn dma_alloc(bytes: usize) -> Option<Dma> {
    crate::mm::alloc_dma(bytes).map(|(phys, virt)| Dma { phys, virt })
}

/// Probe the `n`-th AHCI controller on the PCI bus (only n==0 supported).
pub fn probe_nth(n: usize) -> Option<Ahci> {
    let d = pci::find_class(0x01, 0x06, 0x01)?;
    if n != 0 {
        return None;
    }
    d.enable_bus_master();
    let bar = d.bar(5); // AHCI ABAR is BAR5
    if bar == 0 {
        return None;
    }
    let abar = crate::mm::map_mmio(bar, MMIO_SPAN);
    // SAFETY: `abar` is the HHDM-mapped HBA register block.
    unsafe { Ahci::bringup(abar, dma_alloc) }
}
