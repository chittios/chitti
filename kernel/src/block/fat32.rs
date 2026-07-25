//! **FAT geometry + cluster allocation for writing into an *existing* volume.**
//!
//! [`super::fat`] formats a fresh FAT16 ESP, which is all a whole-disk install
//! needs. Installing *alongside* Windows cannot do that: a PC has exactly one EFI
//! System Partition, and reformatting it removes the Windows boot manager. Our
//! loader has to be added to the ESP that is already there — which means reading
//! its BPB, finding free clusters in its existing FAT, and leaving every other
//! file alone.
//!
//! This module is the pure half of that: geometry arithmetic and free-cluster
//! selection, unit-tested off-hardware. It performs no I/O.
//!
//! Two details in here are the classic FAT bugs:
//!
//! * **FAT type is determined by cluster count, never by a string.** The
//!   `"FAT32   "` / `"FAT16   "` field in the BPB is documentation, not
//!   authority — the specification is explicit that only the count of data
//!   clusters decides, and real formatters do disagree with their own label.
//! * **A FAT32 entry is 28 bits.** The top 4 are reserved and must be masked
//!   before comparing against the end-of-chain and bad-cluster values, or a
//!   perfectly good chain reads as corrupt.

/// Which FAT flavour a volume is, decided by data-cluster count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FatType {
    Fat12,
    Fat16,
    Fat32,
}

/// The geometry of a FAT volume, from its BIOS Parameter Block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bpb {
    pub bytes_per_sector: u32,
    pub sectors_per_cluster: u32,
    pub reserved_sectors: u32,
    pub num_fats: u32,
    /// Root directory entries — 0 on FAT32, where the root is a cluster chain.
    pub root_entries: u32,
    pub fat_size_sectors: u32,
    pub total_sectors: u32,
    /// First cluster of the root directory (FAT32 only; 0 otherwise).
    pub root_cluster: u32,
    pub fat_type: FatType,
    /// Number of addressable data clusters (the value that decided `fat_type`).
    pub data_clusters: u32,
}

/// What a FAT entry says about the cluster it describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClusterState {
    Free,
    /// Marked defective by the formatter; must never be allocated.
    Bad,
    /// Last cluster of a chain.
    End,
    /// Chain continues at this cluster.
    Next(u32),
}

fn le16(b: &[u8], o: usize) -> u32 {
    u16::from_le_bytes([b[o], b[o + 1]]) as u32
}
fn le32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

/// Parse a BPB out of sector 0 of a FAT volume.
///
/// `None` when the sector is not a plausible FAT BPB. Every field that later
/// divides is validated here, so the geometry helpers cannot divide by zero on a
/// hostile or simply non-FAT partition — this parses bytes from a disk that
/// belongs to someone else's operating system.
pub fn parse_bpb(sector: &[u8]) -> Option<Bpb> {
    if sector.len() < 512 {
        return None;
    }
    let bytes_per_sector = le16(sector, 11);
    let sectors_per_cluster = sector[13] as u32;
    let reserved_sectors = le16(sector, 14);
    let num_fats = sector[16] as u32;
    let root_entries = le16(sector, 17);
    let total_16 = le16(sector, 19);
    let fat_16 = le16(sector, 22);
    let total_32 = le32(sector, 32);
    let fat_32 = le32(sector, 36);

    // Reject implausible geometry rather than computing nonsense from it.
    if !matches!(bytes_per_sector, 512 | 1024 | 2048 | 4096) {
        return None;
    }
    if !sectors_per_cluster.is_power_of_two() || sectors_per_cluster == 0 || sectors_per_cluster > 128 {
        return None;
    }
    if reserved_sectors == 0 || num_fats == 0 {
        return None;
    }
    let fat_size_sectors = if fat_16 != 0 { fat_16 } else { fat_32 };
    let total_sectors = if total_16 != 0 { total_16 } else { total_32 };
    if fat_size_sectors == 0 || total_sectors == 0 {
        return None;
    }

    // Data-cluster count — the only thing that decides the FAT type.
    let root_dir_sectors = (root_entries * 32).div_ceil(bytes_per_sector);
    let meta = reserved_sectors + num_fats * fat_size_sectors + root_dir_sectors;
    if total_sectors <= meta {
        return None;
    }
    let data_clusters = (total_sectors - meta) / sectors_per_cluster;
    // Thresholds are from the FAT specification and are exact: < 4085 is FAT12,
    // < 65525 is FAT16, otherwise FAT32.
    let fat_type = if data_clusters < 4085 {
        FatType::Fat12
    } else if data_clusters < 65525 {
        FatType::Fat16
    } else {
        FatType::Fat32
    };
    let root_cluster = if fat_type == FatType::Fat32 { le32(sector, 44) } else { 0 };
    Some(Bpb {
        bytes_per_sector,
        sectors_per_cluster,
        reserved_sectors,
        num_fats,
        root_entries,
        fat_size_sectors,
        total_sectors,
        root_cluster,
        fat_type,
        data_clusters,
    })
}

impl Bpb {
    /// Sectors occupied by the fixed root directory (FAT12/16); 0 on FAT32.
    pub fn root_dir_sectors(&self) -> u32 {
        (self.root_entries * 32).div_ceil(self.bytes_per_sector)
    }

    /// First sector of the data region.
    pub fn first_data_sector(&self) -> u32 {
        self.reserved_sectors + self.num_fats * self.fat_size_sectors + self.root_dir_sectors()
    }

    /// Sector where cluster `n` begins. Data clusters are numbered from **2**;
    /// 0 and 1 are reserved and have no sectors, so they are rejected rather than
    /// silently mapping into the metadata area.
    pub fn cluster_sector(&self, n: u32) -> Option<u32> {
        if n < 2 || n >= self.data_clusters + 2 {
            return None;
        }
        Some(self.first_data_sector() + (n - 2) * self.sectors_per_cluster)
    }

    /// Bytes in one cluster.
    pub fn cluster_bytes(&self) -> u32 {
        self.bytes_per_sector * self.sectors_per_cluster
    }

    /// Clusters needed to hold `bytes`.
    pub fn clusters_for(&self, bytes: u32) -> u32 {
        bytes.div_ceil(self.cluster_bytes().max(1))
    }

    /// Byte offset of cluster `n`'s entry within the FAT.
    pub fn fat_entry_offset(&self, n: u32) -> usize {
        match self.fat_type {
            FatType::Fat32 => n as usize * 4,
            FatType::Fat16 => n as usize * 2,
            // FAT12 packs 1.5 bytes per entry.
            FatType::Fat12 => n as usize + (n as usize / 2),
        }
    }
}

/// Decode a raw FAT entry for `fat_type`.
///
/// The FAT32 mask is the important part: the entry is 28 bits and the top 4 are
/// reserved, so an unmasked comparison against the end-of-chain value makes a
/// healthy chain look corrupt.
pub fn cluster_state(raw: u32, fat_type: FatType) -> ClusterState {
    match fat_type {
        FatType::Fat32 => {
            let v = raw & 0x0fff_ffff;
            match v {
                0 => ClusterState::Free,
                0x0fff_fff7 => ClusterState::Bad,
                v if v >= 0x0fff_fff8 => ClusterState::End,
                v => ClusterState::Next(v),
            }
        }
        FatType::Fat16 => {
            let v = raw & 0xffff;
            match v {
                0 => ClusterState::Free,
                0xfff7 => ClusterState::Bad,
                v if v >= 0xfff8 => ClusterState::End,
                v => ClusterState::Next(v),
            }
        }
        FatType::Fat12 => {
            let v = raw & 0x0fff;
            match v {
                0 => ClusterState::Free,
                0xff7 => ClusterState::Bad,
                v if v >= 0xff8 => ClusterState::End,
                v => ClusterState::Next(v),
            }
        }
    }
}

/// Read cluster `n`'s entry out of a FAT image.
pub fn fat_entry(fat: &[u8], bpb: &Bpb, n: u32) -> Option<u32> {
    let off = bpb.fat_entry_offset(n);
    match bpb.fat_type {
        FatType::Fat32 => {
            if off + 4 > fat.len() {
                return None;
            }
            Some(le32(fat, off))
        }
        FatType::Fat16 => {
            if off + 2 > fat.len() {
                return None;
            }
            Some(le16(fat, off))
        }
        FatType::Fat12 => {
            if off + 2 > fat.len() {
                return None;
            }
            let v = le16(fat, off);
            // Odd clusters take the high 12 bits, even the low 12.
            Some(if n & 1 == 1 { v >> 4 } else { v & 0x0fff })
        }
    }
}

/// Pick `count` free clusters from an existing FAT, lowest-numbered first.
///
/// Returns `None` if the volume does not have that many free — refusing rather
/// than partially allocating, since a half-written chain on someone else's ESP is
/// worse than not installing.
///
/// Clusters marked [`ClusterState::Bad`] are skipped: they are physically
/// defective, and the formatter marked them for a reason.
pub fn find_free_clusters(fat: &[u8], bpb: &Bpb, count: u32) -> Option<alloc::vec::Vec<u32>> {
    if count == 0 {
        return Some(alloc::vec::Vec::new());
    }
    let mut out = alloc::vec::Vec::with_capacity(count as usize);
    // Cluster numbering starts at 2; entries 0 and 1 are reserved media/EOC marks.
    for n in 2..bpb.data_clusters + 2 {
        let Some(raw) = fat_entry(fat, bpb, n) else {
            break; // FAT image shorter than the geometry claims
        };
        if cluster_state(raw, bpb.fat_type) == ClusterState::Free {
            out.push(n);
            if out.len() == count as usize {
                return Some(out);
            }
        }
    }
    None
}

/// Count the free clusters in an existing FAT — for reporting how much room an
/// ESP has before committing to anything.
pub fn count_free_clusters(fat: &[u8], bpb: &Bpb) -> u32 {
    let mut n_free = 0;
    for n in 2..bpb.data_clusters + 2 {
        match fat_entry(fat, bpb, n) {
            Some(raw) if cluster_state(raw, bpb.fat_type) == ClusterState::Free => n_free += 1,
            Some(_) => {}
            None => break,
        }
    }
    n_free
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// Build a BPB sector for a FAT32 volume with `total` sectors.
    fn fat32_bpb(total: u32, spc: u8, fat_sectors: u32) -> vec::Vec<u8> {
        let mut s = vec![0u8; 512];
        s[11..13].copy_from_slice(&512u16.to_le_bytes());
        s[13] = spc;
        s[14..16].copy_from_slice(&32u16.to_le_bytes()); // reserved
        s[16] = 2; // two FATs
        s[17..19].copy_from_slice(&0u16.to_le_bytes()); // FAT32: no fixed root
        s[19..21].copy_from_slice(&0u16.to_le_bytes()); // total_16 unused
        s[22..24].copy_from_slice(&0u16.to_le_bytes()); // fat_16 unused
        s[32..36].copy_from_slice(&total.to_le_bytes());
        s[36..40].copy_from_slice(&fat_sectors.to_le_bytes());
        s[44..48].copy_from_slice(&2u32.to_le_bytes()); // root cluster
        s
    }

    #[test_case]
    fn parses_a_fat32_esp_geometry() {
        // ~100 MiB, 1 sector per cluster -> comfortably above the FAT32 threshold.
        let s = fat32_bpb(204_800, 1, 1600);
        let b = parse_bpb(&s).unwrap();
        assert_eq!(b.fat_type, FatType::Fat32);
        assert_eq!(b.bytes_per_sector, 512);
        assert_eq!(b.root_cluster, 2);
        assert_eq!(b.root_dir_sectors(), 0); // FAT32 root is a chain
        assert_eq!(b.first_data_sector(), 32 + 2 * 1600);
        // Cluster 2 is the first data cluster, by definition.
        assert_eq!(b.cluster_sector(2), Some(b.first_data_sector()));
    }

    #[test_case]
    fn fat_type_comes_from_cluster_count_not_the_label() {
        // Same layout, sized either side of the 65525-cluster FAT16/FAT32 line.
        // A volume labelled "FAT32" can genuinely be FAT16 and vice versa, so the
        // count is the only authority.
        let small = parse_bpb(&fat32_bpb(40_000, 1, 300)).unwrap();
        assert_eq!(small.fat_type, FatType::Fat16);
        assert!(small.data_clusters < 65525);
        let big = parse_bpb(&fat32_bpb(204_800, 1, 1600)).unwrap();
        assert_eq!(big.fat_type, FatType::Fat32);
        assert!(big.data_clusters >= 65525);
    }

    #[test_case]
    fn rejects_implausible_or_non_fat_geometry() {
        // These come off a disk owned by another OS; bad values must not reach the
        // divides in the geometry helpers.
        assert!(parse_bpb(&[0u8; 512]).is_none()); // all zeroes
        assert!(parse_bpb(&[0u8; 100]).is_none()); // short buffer
        let mut s = fat32_bpb(204_800, 1, 1600);
        s[13] = 3; // sectors_per_cluster not a power of two
        assert!(parse_bpb(&s).is_none());
        let mut s = fat32_bpb(204_800, 1, 1600);
        s[11..13].copy_from_slice(&777u16.to_le_bytes()); // silly sector size
        assert!(parse_bpb(&s).is_none());
        let mut s = fat32_bpb(204_800, 1, 1600);
        s[16] = 0; // zero FATs
        assert!(parse_bpb(&s).is_none());
        // Metadata larger than the volume: data_clusters would underflow.
        assert!(parse_bpb(&fat32_bpb(100, 1, 1600)).is_none());
    }

    #[test_case]
    fn fat32_entries_are_masked_to_28_bits() {
        // The reserved top nibble must be ignored. With it included, an
        // end-of-chain marker with the high bits set reads as a Next(...) and the
        // chain walk runs off into nonsense — or a healthy chain looks corrupt.
        assert_eq!(cluster_state(0xffff_ffff, FatType::Fat32), ClusterState::End);
        assert_eq!(cluster_state(0x0fff_fff8, FatType::Fat32), ClusterState::End);
        assert_eq!(cluster_state(0xf000_0000, FatType::Fat32), ClusterState::Free);
        assert_eq!(cluster_state(0xffff_fff7, FatType::Fat32), ClusterState::Bad);
        assert_eq!(cluster_state(0x0000_0005, FatType::Fat32), ClusterState::Next(5));
    }

    #[test_case]
    fn fat16_and_fat12_boundaries() {
        assert_eq!(cluster_state(0xfff8, FatType::Fat16), ClusterState::End);
        assert_eq!(cluster_state(0xfff7, FatType::Fat16), ClusterState::Bad);
        assert_eq!(cluster_state(0x0003, FatType::Fat16), ClusterState::Next(3));
        assert_eq!(cluster_state(0xff8, FatType::Fat12), ClusterState::End);
        assert_eq!(cluster_state(0xff7, FatType::Fat12), ClusterState::Bad);
    }

    #[test_case]
    fn cluster_sector_rejects_reserved_and_out_of_range() {
        let b = parse_bpb(&fat32_bpb(204_800, 1, 1600)).unwrap();
        // 0 and 1 are reserved: mapping them would land in the FAT itself.
        assert_eq!(b.cluster_sector(0), None);
        assert_eq!(b.cluster_sector(1), None);
        assert_eq!(b.cluster_sector(b.data_clusters + 2), None);
        assert!(b.cluster_sector(b.data_clusters + 1).is_some());
    }

    #[test_case]
    fn finds_free_clusters_skipping_used_and_bad() {
        let b = parse_bpb(&fat32_bpb(204_800, 1, 1600)).unwrap();
        let mut fat = vec![0u8; (b.data_clusters as usize + 2) * 4];
        // Entries 0/1 reserved; mark 2 and 3 in use, 4 bad, leave 5+ free.
        fat[2 * 4..3 * 4].copy_from_slice(&0x0fff_ffffu32.to_le_bytes());
        fat[3 * 4..4 * 4].copy_from_slice(&7u32.to_le_bytes());
        fat[4 * 4..5 * 4].copy_from_slice(&0x0fff_fff7u32.to_le_bytes());
        let got = find_free_clusters(&fat, &b, 3).unwrap();
        // A defective cluster must never be handed out.
        assert_eq!(got, vec![5, 6, 7]);
    }

    #[test_case]
    fn refuses_rather_than_partially_allocating() {
        let b = parse_bpb(&fat32_bpb(204_800, 1, 1600)).unwrap();
        // Every cluster in use: asking for one must fail, not return an empty or
        // short list. A half-written chain on someone else's ESP is worse than a
        // refused install.
        let fat = vec![0xffu8; (b.data_clusters as usize + 2) * 4];
        assert!(find_free_clusters(&fat, &b, 1).is_none());
        assert_eq!(count_free_clusters(&fat, &b), 0);
        // Zero clusters is trivially satisfiable.
        assert_eq!(find_free_clusters(&fat, &b, 0), Some(vec![]));
    }

    #[test_case]
    fn counts_free_space_for_reporting() {
        let b = parse_bpb(&fat32_bpb(204_800, 1, 1600)).unwrap();
        let mut fat = vec![0u8; (b.data_clusters as usize + 2) * 4];
        for n in 2..12u32 {
            fat[n as usize * 4..n as usize * 4 + 4].copy_from_slice(&0x0fff_ffffu32.to_le_bytes());
        }
        assert_eq!(count_free_clusters(&fat, &b), b.data_clusters - 10);
    }

    #[test_case]
    fn fat_entry_reads_are_bounds_checked() {
        let b = parse_bpb(&fat32_bpb(204_800, 1, 1600)).unwrap();
        let fat = vec![0u8; 16]; // far shorter than the geometry claims
        assert!(fat_entry(&fat, &b, 3).is_some());
        assert!(fat_entry(&fat, &b, 100).is_none());
        // A truncated FAT must stop the search, not read past the buffer.
        assert!(find_free_clusters(&fat, &b, 1000).is_none());
    }
}
