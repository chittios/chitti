//! The aarch64 boot disk behind the shared [`BlockDevice`] API, selecting the
//! transport at probe time: **virtio-pci** (the real PCIe bus, discovered via
//! ACPI ECAM — real hardware / hypervisors) is preferred; **virtio-mmio** (the
//! QEMU `virt` window) is the fallback. A given platform uses one transport, so
//! `probe_nth` tries PCIe first and falls back to mmio.

use crate::arch::aarch64::virtio_blk::VirtioBlkMmio;
use crate::arch::aarch64::virtio_pci::VirtioBlkPci;
use crate::block::{BlockDevice, BlockError};

pub enum Disk {
    Pci(VirtioBlkPci),
    Mmio(VirtioBlkMmio),
}

impl Disk {
    pub fn probe_nth(n: usize) -> Option<Disk> {
        if let Some(d) = VirtioBlkPci::probe_nth(n) {
            return Some(Disk::Pci(d));
        }
        VirtioBlkMmio::probe_nth(n).map(Disk::Mmio)
    }
}

impl BlockDevice for Disk {
    fn block_count(&self) -> u64 {
        match self {
            Disk::Pci(d) => d.block_count(),
            Disk::Mmio(d) => d.block_count(),
        }
    }
    fn read_block(&mut self, index: u64, buf: &mut [u8]) -> Result<(), BlockError> {
        match self {
            Disk::Pci(d) => d.read_block(index, buf),
            Disk::Mmio(d) => d.read_block(index, buf),
        }
    }
    fn write_block(&mut self, index: u64, buf: &[u8]) -> Result<(), BlockError> {
        match self {
            Disk::Pci(d) => d.write_block(index, buf),
            Disk::Mmio(d) => d.write_block(index, buf),
        }
    }
    fn read_blocks(&mut self, index: u64, buf: &mut [u8]) -> Result<(), BlockError> {
        match self {
            Disk::Pci(d) => d.read_blocks(index, buf),
            Disk::Mmio(d) => d.read_blocks(index, buf),
        }
    }
    fn write_blocks(&mut self, index: u64, buf: &[u8]) -> Result<(), BlockError> {
        match self {
            Disk::Pci(d) => d.write_blocks(index, buf),
            Disk::Mmio(d) => d.write_blocks(index, buf),
        }
    }
}
