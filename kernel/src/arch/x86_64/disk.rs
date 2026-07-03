//! The x86 boot disk behind the shared [`BlockDevice`] API, selecting the
//! transport at probe time: **virtio-blk** (legacy PCI I/O — the QEMU/Limine
//! default) is preferred; then the real-hardware controllers **NVMe** and
//! **AHCI** over PCI (for hosts/hypervisors that don't expose virtio). A given
//! platform uses one, so `probe_nth` tries them in order and skips the absent —
//! the x86 mirror of `arch::aarch64::disk::Disk`.

use crate::arch::x86_64::{ahci, nvme};
use crate::block::ahci::Ahci;
use crate::block::nvme::Nvme;
use crate::block::virtio::VirtioBlk;
use crate::block::{BlockDevice, BlockError};

/// The x86 boot disk, one variant per real transport.
pub enum Disk {
    Virtio(VirtioBlk),
    Nvme(Nvme),
    Ahci(Ahci),
}

impl Disk {
    pub fn probe_nth(n: usize) -> Option<Disk> {
        if let Some(d) = VirtioBlk::probe_nth(n) {
            return Some(Disk::Virtio(d));
        }
        if let Some(d) = nvme::probe_nth(n) {
            return Some(Disk::Nvme(d));
        }
        if let Some(d) = ahci::probe_nth(n) {
            return Some(Disk::Ahci(d));
        }
        None
    }
}

macro_rules! dispatch {
    ($self:ident, $m:ident $(, $a:expr)*) => {
        match $self {
            Disk::Virtio(d) => d.$m($($a),*),
            Disk::Nvme(d) => d.$m($($a),*),
            Disk::Ahci(d) => d.$m($($a),*),
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
