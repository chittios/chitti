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
    /// A **host shared folder** over 9P2000.L. Unlike every other variant this
    /// is not sniffed from a superblock and has no block device under it — it
    /// is set when a `virtio-9p` export is attached.
    Host,
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
            FsType::Host => "9P (host)",
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

/// Decode a UTF-16 label (surrogate pairs handled) into a `String`.
fn utf16_to_string(units: &[u16]) -> String {
    let mut s = String::new();
    let mut i = 0;
    while i < units.len() {
        let u = units[i];
        if (0xd800..=0xdbff).contains(&u) && i + 1 < units.len() && (0xdc00..=0xdfff).contains(&units[i + 1]) {
            let cp = 0x1_0000 + (((u as u32) - 0xd800) << 10) + ((units[i + 1] as u32) - 0xdc00);
            s.push(char::from_u32(cp).unwrap_or('\u{fffd}'));
            i += 2;
        } else {
            s.push(char::from_u32(u as u32).unwrap_or('\u{fffd}'));
            i += 1;
        }
    }
    s
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

    // A filesystem at LBA 0 (a "super-floppy") before the MBR interpretation.
    // FAT/exFAT/NTFS boot sectors all carry the 0x55AA signature that also
    // marks an MBR, so an MBR check here would misread a filesystem's boot
    // code as partition entries: `newfs_exfat` fills bytes 446-509 with 0xF4,
    // which classifies as an MBR with a garbage first partition and the volume
    // comes out Unknown. Only the absence of any known filesystem at LBA 0
    // lets the bytes be read as an MBR.
    let total = dev.block_count();
    let super_floppy = classify(dev, 0, total);
    if super_floppy.fs != FsType::Unknown {
        out.push(super_floppy);
        return out;
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
    out.push(super_floppy);
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
        // The label lives in a 0x83 volume-label entry in the root directory
        // (its first cluster's first sector), not in the boot sector — two
        // extra reads, so worth doing here where `/disks` lists labels.
        let spc_bits = b0[109] as u64;
        if spc_bits <= 12 {
            let clu_offset = le32(&b0, 88) as u64;
            let root_cluster = le32(&b0, 96) as u64;
            if root_cluster >= 2 {
                let spc = 1u64 << spc_bits;
                let root_sec = start_lba + clu_offset + (root_cluster - 2) * spc;
                let mut r = [0u8; BLOCK_SIZE];
                if dev.read_block(root_sec, &mut r).is_ok() {
                    // The label is normally the first root entry.
                    for e in 0..16 {
                        let o = e * 32;
                        if r[o] == 0x00 {
                            break;
                        }
                        if r[o] == 0x83 {
                            let n = (r[o + 1] as usize).min(11);
                            let mut units = alloc::vec![0u16; n];
                            for (k, u) in units.iter_mut().enumerate() {
                                *u = le16(&r, o + 2 + k * 2);
                            }
                            vol.label = Some(utf16_to_string(&units));
                            break;
                        }
                    }
                }
            }
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::ramdisk::RamDisk;

    #[test_case]
    fn an_exfat_super_floppy_beats_the_mbr_reading_of_its_boot_code() {
        // `newfs_exfat` fills the boot-code region (bytes 446+, which overlap
        // the MBR partition-entry area) with 0xF4. With the MBR check first,
        // that reads as an MBR whose first partition is garbage and the volume
        // comes out Unknown; the filesystem at LBA 0 must win over the MBR
        // interpretation. Found by the e2e interop test against a real
        // macOS-formatted volume.
        let mut disk = RamDisk::new(16384);
        crate::block::exfat_rw::format(&mut disk, "TEST").expect("format");
        let mut sec = [0u8; BLOCK_SIZE];
        disk.read_block(0, &mut sec).unwrap();
        sec[446..510].fill(0xf4); // a real formatter's boot code
        disk.write_block(0, &sec).unwrap();
        let vols = probe(&mut disk);
        assert_eq!(vols.len(), 1);
        assert_eq!(vols[0].fs, FsType::ExFat);
        assert_eq!(vols[0].start_lba, 0);
        assert_eq!(vols[0].label.as_deref(), Some("TEST"));
    }

    #[test_case]
    fn a_real_mbr_still_parses_its_partitions() {
        // The super-floppy-first order must not swallow a genuine MBR disk:
        // LBA 0 is not a known filesystem, so the MBR interpretation applies.
        let mut disk = RamDisk::new(4096);
        let mut mbr = [0u8; BLOCK_SIZE];
        mbr[510] = 0x55;
        mbr[511] = 0xaa;
        // Entry 0: type FAT32 (0x0b), start 64, count 100.
        mbr[0x1be + 4] = 0x0b;
        mbr[0x1be + 8..0x1be + 12].copy_from_slice(&64u32.to_le_bytes());
        mbr[0x1be + 12..0x1be + 16].copy_from_slice(&100u32.to_le_bytes());
        disk.write_block(0, &mbr).unwrap();
        let vols = probe(&mut disk);
        assert_eq!(vols.len(), 1);
        assert_eq!(vols[0].start_lba, 64);
        assert_eq!(vols[0].sectors, 100);
    }
}
