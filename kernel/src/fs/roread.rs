//! Read-only directory reads for detected foreign volumes, "where feasible"
//! (per the locked decision). FAT32 root-directory listing is implemented (it's
//! simple and covers most USB sticks); SimpleFS uses its native listing. exFAT,
//! NTFS, ext4, and XFS are detected (see `fs::detect`) but their directory
//! structures are substantial, so listing them is reported as not-yet-supported
//! rather than pretended. No writes are ever issued to a foreign filesystem.

use crate::block::{BlockDevice, BLOCK_SIZE};
use alloc::string::String;
use alloc::vec::Vec;

fn le16(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}
fn le32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

/// List the root directory of a FAT32 volume starting at `start_lba`. Returns
/// `(name, size, is_dir)` for each live 8.3 entry (LFN/deleted/volume-label
/// entries skipped). Assumes 512-byte sectors (the block device's unit).
pub fn fat32_root_list<D: BlockDevice>(dev: &mut D, start_lba: u64) -> Result<Vec<(String, u32, bool)>, &'static str> {
    let mut bs = [0u8; BLOCK_SIZE];
    dev.read_block(start_lba, &mut bs).map_err(|_| "read boot sector")?;
    let bps = le16(&bs, 0x0b) as u64;
    if bps != BLOCK_SIZE as u64 {
        return Err("non-512-byte sectors unsupported");
    }
    let spc = bs[0x0d] as u64; // sectors per cluster
    let reserved = le16(&bs, 0x0e) as u64;
    let num_fats = bs[0x10] as u64;
    let fat_sz = le32(&bs, 0x24) as u64;
    let root_clus = le32(&bs, 0x2c) as u64;
    if spc == 0 || fat_sz == 0 {
        return Err("bad BPB");
    }
    let first_data = reserved + num_fats * fat_sz; // sectors, volume-relative
    let cluster_lba = |c: u64| start_lba + first_data + (c - 2) * spc;

    let mut out = Vec::new();
    let mut clus = root_clus;
    let mut guard = 0;
    let mut fatbuf = [0u8; BLOCK_SIZE];
    let mut sec = [0u8; BLOCK_SIZE];
    'chain: while clus >= 2 && clus < 0x0fff_fff8 && guard < 4096 {
        guard += 1;
        for s in 0..spc {
            if dev.read_block(cluster_lba(clus) + s, &mut sec).is_err() {
                return Err("read cluster");
            }
            for e in (0..BLOCK_SIZE).step_by(32) {
                let ent = &sec[e..e + 32];
                match ent[0] {
                    0x00 => break 'chain, // end of directory
                    0xe5 => continue,     // deleted
                    _ => {}
                }
                let attr = ent[11];
                if attr == 0x0f || attr & 0x08 != 0 {
                    continue; // LFN fragment or volume-label entry
                }
                let name = fat_83_name(ent);
                if name.is_empty() || name == "." || name == ".." {
                    continue;
                }
                out.push((name, le32(ent, 28), attr & 0x10 != 0));
            }
        }
        // Follow the FAT chain to the next cluster.
        let fat_off = clus * 4;
        let fat_lba = start_lba + reserved + fat_off / bps;
        if dev.read_block(fat_lba, &mut fatbuf).is_err() {
            break;
        }
        clus = le32(&fatbuf, (fat_off % bps) as usize) as u64 & 0x0fff_ffff;
    }
    Ok(out)
}

/// Render an 8.3 directory entry's name as `NAME.EXT` (lowercased).
fn fat_83_name(ent: &[u8]) -> String {
    let base: String = ent[0..8].iter().take_while(|&&b| b != b' ').map(|&b| (b as char).to_ascii_lowercase()).collect();
    let ext: String = ent[8..11].iter().take_while(|&&b| b != b' ').map(|&b| (b as char).to_ascii_lowercase()).collect();
    if ext.is_empty() {
        base
    } else {
        alloc::format!("{base}.{ext}")
    }
}
