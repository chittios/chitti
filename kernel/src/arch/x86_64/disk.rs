//! The x86 boot disk behind the shared [`BlockDevice`] API, selecting the
//! transport at probe time: **virtio-blk** (legacy PCI I/O — the QEMU/Limine
//! default) is preferred; then the real-hardware controllers **NVMe** and
//! **AHCI** over PCI (for hosts/hypervisors that don't expose virtio). A given
//! platform uses one, so `probe_nth` tries them in order and skips the absent —
//! the x86 mirror of `arch::aarch64::disk::Disk`.

use crate::arch::x86_64::{ahci, nvme};
use crate::block::ahci::Ahci;
use crate::block::nvme::NvmeNamespace;
use crate::block::usb_msc::UsbMsc;
use crate::block::virtio::VirtioBlk;
use crate::block::{BlockDevice, BlockError};

/// The x86 boot disk, one variant per real transport.
pub enum Disk {
    Virtio(VirtioBlk),
    Nvme(NvmeNamespace),
    Ahci(Ahci),
    /// USB mass-storage stick (BOT); last so internal disks keep stable indices.
    Usb(UsbMsc),
}

impl Disk {
    /// The `n`-th block device across ALL transports, counted globally: every
    /// virtio-blk disk, then every NVMe namespace, then AHCI, then USB MSC.
    /// (Passing `n` to each transport would let one transport's disks shadow
    /// another's.) USB is last so boot/install disk indices stay stable when a
    /// stick is plugged in.
    pub fn probe_nth(n: usize) -> Option<Disk> {
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
        scan!(VirtioBlk::probe_nth, Disk::Virtio);
        scan!(nvme::probe_nth, Disk::Nvme);
        scan!(ahci::probe_nth, Disk::Ahci);
        scan!(UsbMsc::probe_nth, Disk::Usb);
        None
    }
}

macro_rules! dispatch {
    ($self:ident, $m:ident $(, $a:expr)*) => {
        match $self {
            Disk::Virtio(d) => d.$m($($a),*),
            Disk::Nvme(d) => d.$m($($a),*),
            Disk::Ahci(d) => d.$m($($a),*),
            Disk::Usb(d) => d.$m($($a),*),
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
