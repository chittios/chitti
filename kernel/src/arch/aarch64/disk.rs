//! The aarch64 boot disk behind the shared [`BlockDevice`] API, selecting the
//! transport at probe time: **virtio-pci** (the real PCIe bus, discovered via
//! ACPI ECAM — real hardware / hypervisors) is preferred; **virtio-mmio** (the
//! QEMU `virt` window) is the fallback. A given platform uses one transport, so
//! `probe_nth` tries PCIe first and falls back to mmio.

use crate::arch::aarch64::virtio_blk::VirtioBlkMmio;
use crate::arch::aarch64::virtio_pci::VirtioBlkPci;
use crate::arch::aarch64::{ahci, nvme};
use crate::block::ahci::Ahci;
use crate::block::nvme::NvmeNamespace;
use crate::block::{BlockDevice, BlockError};

/// The aarch64 boot disk, one variant per real transport. `probe_nth` tries
/// them in order of preference — virtio (para-virtual, fast) first, then the
/// real-hardware controllers NVMe and AHCI, then the QEMU-mmio fallback. A
/// platform typically exposes exactly one, so ordering just skips the absent.
pub enum Disk {
    Pci(VirtioBlkPci),
    Nvme(NvmeNamespace),
    Ahci(Ahci),
    Mmio(VirtioBlkMmio),
}

impl Disk {
    /// The `n`-th block device across ALL transports, counted globally: every
    /// virtio-pci disk, then every NVMe namespace, then AHCI, then virtio-mmio.
    /// (Passing `n` to each transport would let one transport's disks shadow
    /// another's — e.g. two NVMe namespaces hiding the virtio-mmio ESP.)
    pub fn probe_nth(n: usize) -> Option<Disk> {
        // No block transports on Apple Silicon yet: virtio-mmio (0x0a00_0000)
        // and the PCIe-based NVMe/AHCI/virtio-pci probes all read fixed
        // QEMU/SBSA addresses that data-abort under m1n1's hv. Apple's ANS2
        // storage is a follow-up; report no disks so every caller (model load,
        // persistent store, /install) cleanly finds nothing instead of faulting.
        if super::is_apple() {
            return None;
        }
        let mut idx = 0usize;
        macro_rules! scan {
            ($probe:path, $variant:path) => {{
                let mut k = 0usize;
                while let Some(d) = $probe(k) {
                    if idx == n {
                        return Some($variant(d));
                    }
                    idx += 1;
                    k += 1;
                }
            }};
        }
        scan!(VirtioBlkPci::probe_nth, Disk::Pci);
        scan!(nvme::probe_nth, Disk::Nvme);
        scan!(ahci::probe_nth, Disk::Ahci);
        scan!(VirtioBlkMmio::probe_nth, Disk::Mmio);
        None
    }
}

macro_rules! dispatch {
    ($self:ident, $m:ident $(, $a:expr)*) => {
        match $self {
            Disk::Pci(d) => d.$m($($a),*),
            Disk::Nvme(d) => d.$m($($a),*),
            Disk::Ahci(d) => d.$m($($a),*),
            Disk::Mmio(d) => d.$m($($a),*),
        }
    };
}

impl BlockDevice for Disk {
    fn block_count(&self) -> u64 {
        dispatch!(self, block_count)
    }
    fn read_block(&mut self, index: u64, buf: &mut [u8]) -> Result<(), BlockError> {
        dispatch!(self, read_block, index, buf)
    }
    fn write_block(&mut self, index: u64, buf: &[u8]) -> Result<(), BlockError> {
        dispatch!(self, write_block, index, buf)
    }
    fn read_blocks(&mut self, index: u64, buf: &mut [u8]) -> Result<(), BlockError> {
        dispatch!(self, read_blocks, index, buf)
    }
    fn write_blocks(&mut self, index: u64, buf: &[u8]) -> Result<(), BlockError> {
        dispatch!(self, write_blocks, index, buf)
    }
}
