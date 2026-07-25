//! **Add a loader to an *existing* EFI System Partition**, without disturbing the
//! operating system already installed on it.
//!
//! A PC has exactly one ESP. Installing alongside Windows therefore cannot format
//! it — that would delete the Windows boot manager — so this writes into the
//! filesystem that is already there, using the pure geometry/allocation layer in
//! [`super::fat32`].
//!
//! ## The backup, and why it is a rename
//!
//! We write `/EFI/BOOT/BOOTX64.EFI`, the removable-media fallback path, because it
//! is the only location firmware will boot with no NVRAM boot entry — and we have
//! no way to create one. That file already exists on a Windows ESP (Windows' own
//! fallback loader), so it is **backed up first**.
//!
//! The backup is done by **renaming the existing directory entry**, not by copying
//! its data: the entry keeps pointing at the same cluster chain, so Windows'
//! loader is preserved byte-for-byte with no copying, no second allocation, and no
//! window in which the file is half-written. Restoring it is a rename back.
//!
//! A pre-existing backup is never overwritten — on a second install the first
//! backup is the true Windows original, and clobbering it would lose the only copy.
//!
//! ## What this deliberately does not do
//!
//! It does not create directories. `/EFI/BOOT` exists on every ESP that has ever
//! been booted; if it is absent this refuses rather than synthesising a directory
//! (which needs `.`/`..` entries and is easy to get subtly wrong). It does not
//! touch NVRAM, so on a machine that boots Windows via its own NVRAM entry the
//! firmware boot order still has to be changed by hand.

use super::fat32::{self, Bpb, FatType};
use super::{BlockDevice, BlockError};
use alloc::vec;
use alloc::vec::Vec;

/// Name the displaced Windows loader is renamed to. Valid 8.3, and unlikely to
/// collide with anything a firmware or OS installs.
pub const BACKUP_NAME: &str = "BOOTX64.CHB";
/// The fallback loader path components.
pub const LOADER_DIR: [&str; 2] = ["EFI", "BOOT"];
pub const LOADER_NAME: &str = "BOOTX64.EFI";

#[derive(Debug, PartialEq, Eq)]
pub enum EspError {
    /// Sector 0 is not a plausible FAT BPB.
    NotFat,
    /// FAT12 volumes are not supported (no ESP is FAT12 in practice).
    UnsupportedFat,
    /// `/EFI` or `/EFI/BOOT` is missing.
    NoLoaderDir,
    /// Not enough free clusters for the payload.
    Full,
    /// The directory has no free slot for a new entry.
    DirFull,
    /// A name we need does not fit 8.3.
    BadName,
    Io(BlockError),
}

impl From<BlockError> for EspError {
    fn from(e: BlockError) -> Self {
        EspError::Io(e)
    }
}

/// What an install did, for reporting.
#[derive(Debug, PartialEq, Eq)]
pub struct Installed {
    /// True if an existing loader was renamed out of the way.
    pub backed_up: bool,
    /// True if a backup already existed and was left alone.
    pub backup_preserved: bool,
    pub clusters_used: u32,
}

/// A mounted existing FAT volume, with FAT copy 0 held in memory.
pub struct Esp<'d, D: BlockDevice> {
    dev: &'d mut D,
    bpb: Bpb,
    fat: Vec<u8>,
}

impl<'d, D: BlockDevice> Esp<'d, D> {
    /// Open the FAT volume at the start of `dev`.
    pub fn open(dev: &'d mut D) -> Result<Self, EspError> {
        let mut sector = vec![0u8; 512];
        dev.read_block(0, &mut sector)?;
        let bpb = fat32::parse_bpb(&sector).ok_or(EspError::NotFat)?;
        if bpb.fat_type == FatType::Fat12 {
            return Err(EspError::UnsupportedFat);
        }
        // The FAT is read once and mutated in memory, then written back to every
        // copy at the end — so a failure part-way through leaves the on-disk FAT
        // untouched rather than half-linked.
        let fat_bytes = (bpb.fat_size_sectors * bpb.bytes_per_sector) as usize;
        let mut fat = vec![0u8; fat_bytes];
        dev.read_blocks(bpb.reserved_sectors as u64, &mut fat)?;
        Ok(Esp { dev, bpb, fat })
    }

    pub fn bpb(&self) -> &Bpb {
        &self.bpb
    }

    /// Free space in bytes, for reporting before committing to anything.
    pub fn free_bytes(&self) -> u64 {
        fat32::count_free_clusters(&self.fat, &self.bpb) as u64 * self.bpb.cluster_bytes() as u64
    }

    /// Sectors of the region holding a directory: a cluster chain, or the fixed
    /// root on FAT16.
    fn dir_sectors(&mut self, first_cluster: u32) -> Result<Vec<u32>, EspError> {
        let mut out = Vec::new();
        if first_cluster == 0 {
            // FAT16 fixed root directory.
            let start = self.bpb.reserved_sectors + self.bpb.num_fats * self.bpb.fat_size_sectors;
            for i in 0..self.bpb.root_dir_sectors() {
                out.push(start + i);
            }
            return Ok(out);
        }
        let mut c = first_cluster;
        // Bounded by the cluster count: a corrupted chain that loops must not spin.
        for _ in 0..=self.bpb.data_clusters {
            let base = self.bpb.cluster_sector(c).ok_or(EspError::NoLoaderDir)?;
            for i in 0..self.bpb.sectors_per_cluster {
                out.push(base + i);
            }
            let raw = fat32::fat_entry(&self.fat, &self.bpb, c).ok_or(EspError::NoLoaderDir)?;
            match fat32::cluster_state(raw, self.bpb.fat_type) {
                fat32::ClusterState::Next(n) => c = n,
                _ => return Ok(out),
            }
        }
        Err(EspError::NoLoaderDir)
    }

    /// Read a directory's raw bytes.
    fn read_dir(&mut self, first_cluster: u32) -> Result<(Vec<u32>, Vec<u8>), EspError> {
        let sectors = self.dir_sectors(first_cluster)?;
        let mut buf = vec![0u8; sectors.len() * self.bpb.bytes_per_sector as usize];
        for (i, s) in sectors.iter().enumerate() {
            let off = i * self.bpb.bytes_per_sector as usize;
            self.dev.read_block(*s as u64, &mut buf[off..off + self.bpb.bytes_per_sector as usize])?;
        }
        Ok((sectors, buf))
    }

    /// Write one directory entry back, by absolute byte offset within the
    /// directory's own sector list.
    fn write_dir_entry(&mut self, sectors: &[u32], off: usize, entry: &[u8]) -> Result<(), EspError> {
        let ss = self.bpb.bytes_per_sector as usize;
        let sector = sectors.get(off / ss).copied().ok_or(EspError::DirFull)?;
        let mut buf = vec![0u8; ss];
        self.dev.read_block(sector as u64, &mut buf)?;
        let within = off % ss;
        buf[within..within + fat32::DIR_ENTRY_LEN].copy_from_slice(entry);
        self.dev.write_block(sector as u64, &buf)?;
        Ok(())
    }

    /// Resolve a directory path from the volume root, returning its first cluster.
    fn resolve_dir(&mut self, path: &[&str]) -> Result<u32, EspError> {
        let mut cluster = self.bpb.root_cluster; // 0 on FAT16 = fixed root
        for part in path {
            let name = fat32::short_name(part).ok_or(EspError::BadName)?;
            let (_, dir) = self.read_dir(cluster)?;
            let (_, first, _, attr) = fat32::find_dir_entry(&dir, name).ok_or(EspError::NoLoaderDir)?;
            if attr & fat32::ATTR_DIRECTORY == 0 {
                return Err(EspError::NoLoaderDir); // a file where a directory must be
            }
            cluster = first;
        }
        Ok(cluster)
    }

    /// Install `data` as the fallback loader, backing up any existing one.
    pub fn install_loader(&mut self, data: &[u8]) -> Result<Installed, EspError> {
        let dir_cluster = self.resolve_dir(&LOADER_DIR)?;
        let loader = fat32::short_name(LOADER_NAME).ok_or(EspError::BadName)?;
        let backup = fat32::short_name(BACKUP_NAME).ok_or(EspError::BadName)?;

        let (sectors, dir) = self.read_dir(dir_cluster)?;
        let existing = fat32::find_dir_entry(&dir, loader);
        let had_backup = fat32::find_dir_entry(&dir, backup).is_some();

        // Allocate before mutating anything, so a full volume changes nothing.
        let need = self.bpb.clusters_for(data.len() as u32);
        let clusters =
            fat32::find_free_clusters(&self.fat, &self.bpb, need).ok_or(EspError::Full)?;

        // Back up the displaced loader by renaming its entry: it keeps pointing at
        // the same clusters, so Windows' loader survives byte-for-byte. Never
        // overwrite an existing backup — that one is the true original.
        let mut backed_up = false;
        if let Some((off, first, size, attr)) = existing {
            if !had_backup {
                let renamed = fat32::dir_entry(backup, attr, first, size);
                self.write_dir_entry(&sectors, off, &renamed)?;
                backed_up = true;
            } else {
                // Free the entry we are about to replace; its clusters stay
                // referenced by nothing, but leaking them is far safer than
                // truncating a chain we did not create.
                let mut deleted = dir[off..off + fat32::DIR_ENTRY_LEN].to_vec();
                deleted[0] = 0xe5;
                self.write_dir_entry(&sectors, off, &deleted)?;
            }
        }

        // Write the payload's clusters.
        let cb = self.bpb.cluster_bytes() as usize;
        for (i, &c) in clusters.iter().enumerate() {
            let base = self.bpb.cluster_sector(c).ok_or(EspError::Full)?;
            let start = i * cb;
            let end = (start + cb).min(data.len());
            let mut buf = vec![0u8; cb];
            if start < data.len() {
                buf[..end - start].copy_from_slice(&data[start..end]);
            }
            self.dev.write_blocks(base as u64, &buf)?;
        }

        // Link the chain in the in-memory FAT, then flush to EVERY copy. Copies
        // that disagree are what chkdsk calls corruption, and Windows may "repair"
        // the volume back to the stale copy, silently undoing this install.
        for (cluster, value) in fat32::chain_entries(&clusters, self.bpb.fat_type) {
            let off = self.bpb.fat_entry_offset(cluster);
            match self.bpb.fat_type {
                FatType::Fat32 => {
                    // Preserve the reserved top nibble rather than zeroing it.
                    let old = u32::from_le_bytes([
                        self.fat[off],
                        self.fat[off + 1],
                        self.fat[off + 2],
                        self.fat[off + 3],
                    ]);
                    let merged = (old & 0xf000_0000) | (value & 0x0fff_ffff);
                    self.fat[off..off + 4].copy_from_slice(&merged.to_le_bytes());
                }
                FatType::Fat16 => {
                    self.fat[off..off + 2].copy_from_slice(&(value as u16).to_le_bytes());
                }
                FatType::Fat12 => return Err(EspError::UnsupportedFat),
            }
        }
        for copy in 0..self.bpb.num_fats {
            let start = self.bpb.reserved_sectors + copy * self.bpb.fat_size_sectors;
            self.dev.write_blocks(start as u64, &self.fat)?;
        }

        // Finally the directory entry — last, so the file is fully written and
        // linked before anything points at it.
        let first = clusters.first().copied().unwrap_or(0);
        let entry = fat32::dir_entry(loader, fat32::ATTR_ARCHIVE, first, data.len() as u32);
        let (sectors, dir) = self.read_dir(dir_cluster)?;
        let slot = fat32::find_free_dir_slot(&dir).ok_or(EspError::DirFull)?;
        self.write_dir_entry(&sectors, slot, &entry)?;

        Ok(Installed { backed_up, backup_preserved: had_backup, clusters_used: need })
    }

    /// Undo an install: rename the backup back over the loader entry.
    pub fn restore_backup(&mut self) -> Result<bool, EspError> {
        let dir_cluster = self.resolve_dir(&LOADER_DIR)?;
        let loader = fat32::short_name(LOADER_NAME).ok_or(EspError::BadName)?;
        let backup = fat32::short_name(BACKUP_NAME).ok_or(EspError::BadName)?;
        let (sectors, dir) = self.read_dir(dir_cluster)?;
        let Some((boff, bfirst, bsize, battr)) = fat32::find_dir_entry(&dir, backup) else {
            return Ok(false);
        };
        // Drop our entry, then rename the backup back into place.
        if let Some((off, _, _, _)) = fat32::find_dir_entry(&dir, loader) {
            let mut deleted = dir[off..off + fat32::DIR_ENTRY_LEN].to_vec();
            deleted[0] = 0xe5;
            self.write_dir_entry(&sectors, off, &deleted)?;
        }
        let restored = fat32::dir_entry(loader, battr, bfirst, bsize);
        self.write_dir_entry(&sectors, boff, &restored)?;
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::ramdisk::RamDisk;

    const SS: usize = 512;
    const SPC: u32 = 1;
    const RESERVED: u32 = 32;
    const FAT_SECTORS: u32 = 1600;
    const TOTAL: u32 = 204_800; // ~100 MiB -> FAT32

    fn put(dev: &mut RamDisk, lba: u64, data: &[u8]) {
        let mut buf = [0u8; SS];
        buf[..data.len().min(SS)].copy_from_slice(&data[..data.len().min(SS)]);
        dev.write_block(lba, &buf).unwrap();
    }

    /// Build a minimal but real FAT32 volume: BPB, two FATs, a root directory
    /// containing `EFI`, and `EFI` containing `BOOT`.
    ///
    /// Root = cluster 2, /EFI = cluster 3, /EFI/BOOT = cluster 4.
    fn fat32_volume(with_existing_loader: bool) -> RamDisk {
        let mut dev = RamDisk::new(TOTAL as u64);
        let mut bpb = [0u8; SS];
        bpb[11..13].copy_from_slice(&(SS as u16).to_le_bytes());
        bpb[13] = SPC as u8;
        bpb[14..16].copy_from_slice(&(RESERVED as u16).to_le_bytes());
        bpb[16] = 2; // two FATs — the write-to-every-copy path
        bpb[32..36].copy_from_slice(&TOTAL.to_le_bytes());
        bpb[36..40].copy_from_slice(&FAT_SECTORS.to_le_bytes());
        bpb[44..48].copy_from_slice(&2u32.to_le_bytes()); // root cluster
        dev.write_block(0, &bpb).unwrap();

        let b = fat32::parse_bpb(&bpb).unwrap();
        assert_eq!(b.fat_type, FatType::Fat32);

        // FAT: reserved entries 0/1, then clusters 2,3,4 (root, /EFI, /EFI/BOOT)
        // each a single-cluster chain. Cluster 5 is marked allocated too when the
        // Windows loader is present — a real volume always marks its files'
        // clusters in use, and the allocator's whole protection against
        // overwriting existing data is that the FAT says so. (An earlier version
        // of this fixture left 5 free, and the test correctly caught the install
        // scribbling over Windows' loader.)
        let mut fat = vec![0u8; (FAT_SECTORS * SS as u32) as usize];
        let mut used: vec::Vec<u32> = vec![0, 1, 2, 3, 4];
        if with_existing_loader {
            used.push(5);
        }
        for c in used {
            fat[c as usize * 4..c as usize * 4 + 4].copy_from_slice(&0x0fff_ffffu32.to_le_bytes());
        }
        for copy in 0..2u32 {
            let start = RESERVED + copy * FAT_SECTORS;
            dev.write_blocks(start as u64, &fat).unwrap();
        }

        // Root directory (cluster 2): one entry, the EFI directory at cluster 3.
        let efi = fat32::dir_entry(fat32::short_name("EFI").unwrap(), fat32::ATTR_DIRECTORY, 3, 0);
        put(&mut dev, b.cluster_sector(2).unwrap() as u64, &efi);

        // /EFI (cluster 3): one entry, BOOT at cluster 4.
        let boot = fat32::dir_entry(fat32::short_name("BOOT").unwrap(), fat32::ATTR_DIRECTORY, 4, 0);
        put(&mut dev, b.cluster_sector(3).unwrap() as u64, &boot);

        // /EFI/BOOT (cluster 4): optionally Windows' existing fallback loader,
        // pointed at cluster 5 with some recognisable content.
        if with_existing_loader {
            let mut sec = [0u8; SS];
            let e = fat32::dir_entry(
                fat32::short_name(LOADER_NAME).unwrap(),
                fat32::ATTR_ARCHIVE,
                5,
                4,
            );
            sec[..32].copy_from_slice(&e);
            dev.write_block(b.cluster_sector(4).unwrap() as u64, &sec).unwrap();
            put(&mut dev, b.cluster_sector(5).unwrap() as u64, b"WINL");
        }
        dev
    }

    /// Read a directory cluster's raw bytes back off the device.
    fn dir_bytes(dev: &mut RamDisk, cluster: u32) -> vec::Vec<u8> {
        let mut sector = vec![0u8; SS];
        dev.read_block(0, &mut sector).unwrap();
        let b = fat32::parse_bpb(&sector).unwrap();
        let mut buf = vec![0u8; SS];
        dev.read_block(b.cluster_sector(cluster).unwrap() as u64, &mut buf).unwrap();
        buf
    }

    #[test_case]
    fn installs_the_loader_into_an_existing_esp() {
        let mut dev = fat32_volume(false);
        let payload = vec![0xabu8; 900]; // spans two 512-byte clusters
        let out = {
            let mut esp = Esp::open(&mut dev).unwrap();
            esp.install_loader(&payload).unwrap()
        };
        assert!(!out.backed_up); // nothing was there to displace
        assert_eq!(out.clusters_used, 2);

        // The directory entry must exist, be the right size, and point at a chain.
        let dir = dir_bytes(&mut dev, 4);
        let (_, first, size, _) =
            fat32::find_dir_entry(&dir, fat32::short_name(LOADER_NAME).unwrap()).unwrap();
        assert_eq!(size, 900);
        assert!(first >= 2);

        // The payload must actually be on disk at that cluster.
        let mut sector = vec![0u8; SS];
        dev.read_block(0, &mut sector).unwrap();
        let b = fat32::parse_bpb(&sector).unwrap();
        let mut got = vec![0u8; SS];
        dev.read_block(b.cluster_sector(first).unwrap() as u64, &mut got).unwrap();
        assert_eq!(&got[..], &payload[..SS]);
    }

    #[test_case]
    fn backs_up_the_windows_loader_by_rename_preserving_its_data() {
        let mut dev = fat32_volume(true);
        let out = {
            let mut esp = Esp::open(&mut dev).unwrap();
            esp.install_loader(&vec![0x11u8; 100]).unwrap()
        };
        assert!(out.backed_up);
        assert!(!out.backup_preserved);

        let dir = dir_bytes(&mut dev, 4);
        // The backup entry still points at the ORIGINAL cluster 5 — the rename
        // preserved Windows' loader byte-for-byte with no copying.
        let (_, bfirst, bsize, _) =
            fat32::find_dir_entry(&dir, fat32::short_name(BACKUP_NAME).unwrap()).unwrap();
        assert_eq!((bfirst, bsize), (5, 4));
        let mut sector = vec![0u8; SS];
        dev.read_block(0, &mut sector).unwrap();
        let b = fat32::parse_bpb(&sector).unwrap();
        let mut got = vec![0u8; SS];
        dev.read_block(b.cluster_sector(5).unwrap() as u64, &mut got).unwrap();
        assert_eq!(&got[..4], b"WINL", "Windows loader data was disturbed");

        // And our loader is a separate entry at a different cluster.
        let (_, ours, _, _) =
            fat32::find_dir_entry(&dir, fat32::short_name(LOADER_NAME).unwrap()).unwrap();
        assert_ne!(ours, 5);
    }

    #[test_case]
    fn a_second_install_never_clobbers_the_first_backup() {
        // The first backup is the true Windows original; overwriting it on
        // reinstall would lose the only copy.
        let mut dev = fat32_volume(true);
        {
            let mut esp = Esp::open(&mut dev).unwrap();
            esp.install_loader(&vec![0x11u8; 100]).unwrap();
        }
        let out2 = {
            let mut esp = Esp::open(&mut dev).unwrap();
            esp.install_loader(&vec![0x22u8; 100]).unwrap()
        };
        assert!(out2.backup_preserved);
        assert!(!out2.backed_up);
        let dir = dir_bytes(&mut dev, 4);
        let (_, bfirst, _, _) =
            fat32::find_dir_entry(&dir, fat32::short_name(BACKUP_NAME).unwrap()).unwrap();
        assert_eq!(bfirst, 5, "backup no longer points at Windows' original clusters");
    }

    #[test_case]
    fn the_chain_is_written_to_every_fat_copy() {
        // Copies that disagree are what chkdsk calls corruption, and Windows may
        // "repair" the volume back to the stale copy, undoing the install.
        let mut dev = fat32_volume(false);
        {
            let mut esp = Esp::open(&mut dev).unwrap();
            esp.install_loader(&vec![0x33u8; 2000]).unwrap(); // 4 clusters
        }
        let mut a = vec![0u8; (FAT_SECTORS * SS as u32) as usize];
        let mut b2 = vec![0u8; (FAT_SECTORS * SS as u32) as usize];
        dev.read_blocks(RESERVED as u64, &mut a).unwrap();
        dev.read_blocks((RESERVED + FAT_SECTORS) as u64, &mut b2).unwrap();
        assert_eq!(a, b2, "FAT copies diverged");
    }

    #[test_case]
    fn restore_puts_the_windows_loader_back() {
        let mut dev = fat32_volume(true);
        {
            let mut esp = Esp::open(&mut dev).unwrap();
            esp.install_loader(&vec![0x11u8; 100]).unwrap();
        }
        let restored = {
            let mut esp = Esp::open(&mut dev).unwrap();
            esp.restore_backup().unwrap()
        };
        assert!(restored);
        let dir = dir_bytes(&mut dev, 4);
        // BOOTX64.EFI points at Windows' original clusters again.
        let (_, first, size, _) =
            fat32::find_dir_entry(&dir, fat32::short_name(LOADER_NAME).unwrap()).unwrap();
        assert_eq!((first, size), (5, 4));
        // And the backup name is gone.
        assert!(fat32::find_dir_entry(&dir, fat32::short_name(BACKUP_NAME).unwrap()).is_none());
    }

    #[test_case]
    fn refuses_a_non_fat_or_missing_loader_dir() {
        // A blank device is not a FAT volume.
        let mut blank = RamDisk::new(TOTAL as u64);
        assert_eq!(Esp::open(&mut blank).err(), Some(EspError::NotFat));
        // A valid volume with no /EFI/BOOT must refuse, not synthesise one.
        let mut dev = RamDisk::new(TOTAL as u64);
        let mut bpb = [0u8; SS];
        bpb[11..13].copy_from_slice(&(SS as u16).to_le_bytes());
        bpb[13] = SPC as u8;
        bpb[14..16].copy_from_slice(&(RESERVED as u16).to_le_bytes());
        bpb[16] = 2;
        bpb[32..36].copy_from_slice(&TOTAL.to_le_bytes());
        bpb[36..40].copy_from_slice(&FAT_SECTORS.to_le_bytes());
        bpb[44..48].copy_from_slice(&2u32.to_le_bytes());
        dev.write_block(0, &bpb).unwrap();
        let mut fat = vec![0u8; (FAT_SECTORS * SS as u32) as usize];
        fat[2 * 4..3 * 4].copy_from_slice(&0x0fff_ffffu32.to_le_bytes());
        dev.write_blocks(RESERVED as u64, &fat).unwrap();
        let mut esp = Esp::open(&mut dev).unwrap();
        assert_eq!(esp.install_loader(&[0u8; 10]).err(), Some(EspError::NoLoaderDir));
    }
}
