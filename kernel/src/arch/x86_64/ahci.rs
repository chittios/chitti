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

/// Probe the `n`-th AHCI **disk**, counted across every AHCI controller on the
/// bus and every populated port on each — not the n-th controller. A desktop
/// with two drives on one HBA, or a chipset exposing two HBAs, presents them all.
///
/// Each controller's populated-port count is read first ([`Ahci::present_count`],
/// pure register reads) so only the port that `n` actually names is brought up;
/// bringing up and discarding the earlier ones would leak their DMA on every
/// enumeration.
pub fn probe_nth(n: usize) -> Option<Ahci> {
    let mut base = 0usize; // global index of this controller's first disk
    let mut c = 0usize;
    while let Some(d) = pci::find_class_nth(0x01, 0x06, 0x01, c) {
        c += 1;
        d.enable_bus_master();
        let bar = d.bar(5); // AHCI ABAR is BAR5
        if bar == 0 {
            continue;
        }
        let abar = crate::mm::map_mmio(bar, MMIO_SPAN);
        // SAFETY: `abar` is the HHDM-mapped HBA register block.
        let count = unsafe { Ahci::present_count(abar) };
        if n < base + count {
            // SAFETY: as above; the port index is within the counted range.
            return unsafe { Ahci::bringup_nth(abar, dma_alloc, n - base) };
        }
        base += count;
    }
    None
}
