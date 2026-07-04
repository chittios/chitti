//! x86 discovery wrapper for the shared NVMe core ([`crate::block::nvme`]).
//! Finds the NVMe function on the legacy PCI bus (`arch::x86_64::pci`, class
//! 0x01/0x08/0x02), maps BAR0 into the HHDM, and hands the mapped base + the
//! frame-allocator DMA source to the arch-neutral driver.

use crate::arch::x86_64::pci;
use crate::block::nvme::{probe_namespace, NvmeNamespace, MMIO_SPAN};
use crate::block::Dma;

/// x86 DMA: a physically-contiguous frame-allocator region reached via the HHDM.
fn dma_alloc(bytes: usize) -> Option<Dma> {
    crate::mm::alloc_dma(bytes).map(|(phys, virt)| Dma { phys, virt })
}

/// Probe the `n`-th NVMe disk on the PCI bus. One controller is supported, but
/// it may expose several namespaces; `n` selects namespace `n + 1`. `None` once
/// namespaces run out.
pub fn probe_nth(n: usize) -> Option<NvmeNamespace> {
    // NVMe class 0x01 (mass storage) / subclass 0x08 / prog-if 0x02.
    let d = pci::find_class(0x01, 0x08, 0x02)?;
    d.enable_bus_master();
    let bar = d.bar(0);
    if bar == 0 {
        return None;
    }
    let regs = crate::mm::map_mmio(bar, MMIO_SPAN);
    // SAFETY: `regs` is the HHDM-mapped NVMe register block. The controller is
    // brought up once; this attaches namespace n+1 through it.
    unsafe { probe_namespace(regs, dma_alloc, n) }
}
