//! **Filesystem detection** for internal disks and external/USB drives:
//! identify FAT16/32, exFAT, NTFS, ext2/3/4, and XFS by on-disk signatures,
//! and parse MBR / GPT so per-partition volumes are recognized.
//!
//! This module only **reads** a handful of sectors to classify a volume and
//! pull a label. Read/write happens in [`super::vfs`] via
//! `block::{fat_rw,ext4_rw,ntfs_read}` after an explicit [`super::mount`].

use crate::block::{BlockDevice, BLOCK_SIZE};
use alloc::string::String;
use alloc::vec::Vec;

/// A recognized filesystem type.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FsType {
    Fat16,
    Fat32,
    ExFat,
    Ntfs,
    Ext2,
    Ext3,
    Ext4,
    Xfs,
    Unknown,
}

impl FsType {
    pub fn name(self) -> &'static str {
        match self {
            FsType::Fat16 => "FAT16",
            FsType::Fat32 => "FAT32",
            FsType::ExFat => "exFAT",
            FsType::Ntfs => "NTFS",
            FsType::Ext2 => "ext2",
            FsType::Ext3 => "ext3",
            FsType::Ext4 => "ext4",
            FsType::Xfs => "XFS",
            FsType::Unknown => "unknown",
        }
    }
}

/// A detected volume: where it starts, how big, its filesystem, and (if easily
/// read) its label.
#[derive(Clone, Debug)]
pub struct Volume {
    pub start_lba: u64,
    pub sectors: u64,
    pub fs: FsType,
    pub label: Option<String>,
}

fn le16(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}
fn le32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}
fn le64(b: &[u8], o: usize) -> u64 {
    let mut v = [0u8; 8];
    v.copy_from_slice(&b[o..o + 8]);
    u64::from_le_bytes(v)
}

/// Trim trailing spaces/NULs from a fixed-width label field.
fn trim_label(bytes: &[u8]) -> Option<String> {
    let s: String = bytes.iter().take_while(|&&b| b != 0).map(|&b| b as char).collect();
    let t = s.trim_end_matches(' ').trim();
    if t.is_empty() || t == "NO NAME" {
        None
    } else {
        Some(t.into())
    }
}

/// Probe a whole disk: parse GPT or MBR (or treat as a "super-floppy" with a
/// filesystem directly at LBA 0), and classify each volume. Returns the list of
/// detected volumes (may be empty).
pub fn probe<D: BlockDevice>(dev: &mut D) -> Vec<Volume> {
    let mut out = Vec::new();
    let mut lba0 = [0u8; BLOCK_SIZE];
    if dev.read_block(0, &mut lba0).is_err() {
        return out;
    }

    // GPT? The protective MBR at LBA0 is followed by the GPT header at LBA1.
    let mut gpt = [0u8; BLOCK_SIZE];
    if dev.read_block(1, &mut gpt).is_ok() && &gpt[0..8] == b"EFI PART" {
        parse_gpt(dev, &gpt, &mut out);
        if !out.is_empty() {
            return out;
        }
    }

    // MBR partition table? (0x55AA signature + non-empty entries at 0x1BE.)
    if lba0[510] == 0x55 && lba0[511] == 0xaa {
        let mut any = false;
        for i in 0..4 {
            let e = 0x1be + i * 16;
            let ptype = lba0[e + 4];
            let start = le32(&lba0, e + 8) as u64;
            let count = le32(&lba0, e + 12) as u64;
            if ptype != 0 && count != 0 {
                any = true;
                out.push(classify(dev, start, count));
            }
        }
        if any {
            return out;
        }
    }

    // No partition table: a filesystem may live directly on the device
    // (super-floppy, common on USB sticks). Classify LBA 0 as one volume.
    let total = dev.block_count();
    out.push(classify(dev, 0, total));
    out
}

fn parse_gpt<D: BlockDevice>(dev: &mut D, hdr: &[u8], out: &mut Vec<Volume>) {
    let entry_lba = le64(hdr, 72);
    let num = le32(hdr, 80).min(128);
    let esz = le32(hdr, 84).max(128) as usize;
    let per_sector = BLOCK_SIZE / esz;
    if per_sector == 0 {
        return;
    }
    let mut sector = [0u8; BLOCK_SIZE];
    let mut read = 0u32;
    let mut lba = entry_lba;
    while read < num {
        if dev.read_block(lba, &mut sector).is_err() {
            break;
        }
        for k in 0..per_sector {
            if read >= num {
                break;
            }
            read += 1;
            let e = k * esz;
            // A zero type-GUID means an unused entry.
            if sector[e..e + 16].iter().all(|&b| b == 0) {
                continue;
            }
            let first = le64(&sector, e + 32);
            let last = le64(&sector, e + 40);
            if last >= first {
                out.push(classify(dev, first, last - first + 1));
            }
        }
        lba += 1;
    }
}

/// Classify the volume starting at `start_lba` by reading its first sectors.
fn classify<D: BlockDevice>(dev: &mut D, start_lba: u64, sectors: u64) -> Volume {
    let mut vol = Volume { start_lba, sectors, fs: FsType::Unknown, label: None };
    let mut b0 = [0u8; BLOCK_SIZE];
    if dev.read_block(start_lba, &mut b0).is_err() {
        return vol;
    }

    // exFAT / NTFS: OEM name at offset 3.
    if &b0[3..11] == b"EXFAT   " {
        vol.fs = FsType::ExFat;
        return vol;
    }
    if &b0[3..11] == b"NTFS    " {
        vol.fs = FsType::Ntfs;
        return vol;
    }
    // FAT16: "FAT16   " at 0x36 (54), boot signature 0x55AA — what our own ESP
    // writer produces (label at 0x2b).
    if &b0[54..62] == b"FAT16   " && b0[510] == 0x55 && b0[511] == 0xaa {
        vol.fs = FsType::Fat16;
        vol.label = trim_label(&b0[0x2b..0x36]);
        return vol;
    }
    // FAT32: "FAT32   " at 0x52, boot signature 0x55AA, volume label at 0x47.
    if &b0[0x52..0x5a] == b"FAT32   " && b0[510] == 0x55 && b0[511] == 0xaa {
        vol.fs = FsType::Fat32;
        vol.label = trim_label(&b0[0x47..0x52]);
        return vol;
    }
    // XFS: superblock magic "XFSB" at offset 0 of the volume.
    if &b0[0..4] == b"XFSB" {
        vol.fs = FsType::Xfs;
        // XFS label: 12 bytes at offset 0x6c.
        vol.label = trim_label(&b0[0x6c..0x78]);
        return vol;
    }
    // ext2/3/4: superblock at volume offset 1024 (block 2 of a 512B device),
    // magic 0xEF53 (LE) at superblock offset 0x38.
    let mut sb = [0u8; BLOCK_SIZE];
    if dev.read_block(start_lba + 2, &mut sb).is_ok() && le16(&sb, 0x38) == 0xef53 {
        // feature_compat@0x5c has_journal (0x4) => ext3+; feature_incompat@0x60
        // extents (0x40) => ext4.
        let compat = le32(&sb, 0x5c);
        let incompat = le32(&sb, 0x60);
        vol.fs = if incompat & 0x40 != 0 {
            FsType::Ext4
        } else if compat & 0x4 != 0 {
            FsType::Ext3
        } else {
            FsType::Ext2
        };
        // ext label: 16 bytes at superblock offset 0x78.
        vol.label = trim_label(&sb[0x78..0x88]);
        return vol;
    }

    vol
}
