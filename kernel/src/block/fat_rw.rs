//! **FAT16/FAT32 read-write** for mounted volumes (internal disks and USB).
//!
//! Builds on [`super::fat32`] pure geometry helpers and the same patterns as
//! [`super::esp`] (load FAT → mutate → write every copy). Short 8.3 names only
//! for create (LFN is still read by [`super::fat_read`]); names that need LFN
//! are refused on write rather than silently truncated.
//!
//! ## Ops
//! - `read` / `write` (create or replace file)
//! - `unlink` file
//! - `mkdir`
//! - `readdir` (LFN-aware list via the same walk as the reader)

use crate::block::fat32::{self, Bpb, FatType, ATTR_DIRECTORY, DIR_ENTRY_LEN};
use crate::block::{BlockDevice, BlockError, BLOCK_SIZE};
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

/// Errors from FAT mutations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FatRwError {
    NotFat,
    Unsupported,
    Io,
    NotFound,
    Exists,
    NotAFile,
    NotADir,
    NotEmpty,
    BadName,
    Full,
}

impl From<BlockError> for FatRwError {
    fn from(_: BlockError) -> Self {
        FatRwError::Io
    }
}

/// Live FAT volume handle.
pub struct FatRw<'d, D: BlockDevice> {
    dev: &'d mut D,
    bpb: Bpb,
    /// Full FAT table (all copies start identical; we rewrite every copy).
    fat: Vec<u8>,
    dirty_fat: bool,
}

impl<'d, D: BlockDevice> FatRw<'d, D> {
    pub fn open(dev: &'d mut D) -> Result<Self, FatRwError> {
        let mut sector = [0u8; BLOCK_SIZE];
        dev.read_block(0, &mut sector)?;
        let bpb = fat32::parse_bpb(&sector).ok_or(FatRwError::NotFat)?;
        if bpb.fat_type == FatType::Fat12 {
            return Err(FatRwError::Unsupported);
        }
        if bpb.bytes_per_sector as usize != BLOCK_SIZE {
            return Err(FatRwError::Unsupported);
        }
        let fat_bytes = (bpb.fat_size_sectors * bpb.bytes_per_sector) as usize;
        let mut fat = vec![0u8; fat_bytes];
        dev.read_blocks(bpb.reserved_sectors as u64, &mut fat)?;
        Ok(FatRw {
            dev,
            bpb,
            fat,
            dirty_fat: false,
        })
    }

    /// Flush the in-memory FAT to every on-disk copy.
    pub fn sync_fat(&mut self) -> Result<(), FatRwError> {
        if !self.dirty_fat {
            return Ok(());
        }
        for copy in 0..self.bpb.num_fats {
            let start = self.bpb.reserved_sectors + copy * self.bpb.fat_size_sectors;
            self.dev
                .write_blocks(start as u64, &self.fat)
                .map_err(|_| FatRwError::Io)?;
        }
        self.dirty_fat = false;
        Ok(())
    }

    fn set_fat_entry(&mut self, cluster: u32, value: u32) -> Result<(), FatRwError> {
        let off = self.bpb.fat_entry_offset(cluster);
        match self.bpb.fat_type {
            FatType::Fat32 => {
                if off + 4 > self.fat.len() {
                    return Err(FatRwError::Io);
                }
                let keep = u32::from_le_bytes(self.fat[off..off + 4].try_into().unwrap()) & 0xf000_0000;
                let v = (value & 0x0fff_ffff) | keep;
                self.fat[off..off + 4].copy_from_slice(&v.to_le_bytes());
            }
            FatType::Fat16 => {
                if off + 2 > self.fat.len() {
                    return Err(FatRwError::Io);
                }
                self.fat[off..off + 2].copy_from_slice(&(value as u16).to_le_bytes());
            }
            FatType::Fat12 => return Err(FatRwError::Unsupported),
        }
        self.dirty_fat = true;
        Ok(())
    }

    fn free_chain(&mut self, mut first: u32) -> Result<(), FatRwError> {
        if first < 2 {
            return Ok(());
        }
        for _ in 0..=self.bpb.data_clusters {
            let raw = fat32::fat_entry(&self.fat, &self.bpb, first).ok_or(FatRwError::Io)?;
            self.set_fat_entry(first, 0)?;
            match fat32::cluster_state(raw, self.bpb.fat_type) {
                fat32::ClusterState::Next(n) => first = n,
                _ => return Ok(()),
            }
        }
        Ok(())
    }

    fn dir_sectors(&self, first_cluster: u32) -> Result<Vec<u32>, FatRwError> {
        let mut out = Vec::new();
        if first_cluster == 0 {
            let start = self.bpb.reserved_sectors + self.bpb.num_fats * self.bpb.fat_size_sectors;
            for i in 0..self.bpb.root_dir_sectors() {
                out.push(start + i);
            }
            return Ok(out);
        }
        let mut c = first_cluster;
        for _ in 0..=self.bpb.data_clusters {
            let base = self.bpb.cluster_sector(c).ok_or(FatRwError::Io)?;
            for i in 0..self.bpb.sectors_per_cluster {
                out.push(base + i);
            }
            let raw = fat32::fat_entry(&self.fat, &self.bpb, c).ok_or(FatRwError::Io)?;
            match fat32::cluster_state(raw, self.bpb.fat_type) {
                fat32::ClusterState::Next(n) => c = n,
                _ => return Ok(out),
            }
        }
        Err(FatRwError::Io)
    }

    fn read_dir(&mut self, first_cluster: u32) -> Result<(Vec<u32>, Vec<u8>), FatRwError> {
        let sectors = self.dir_sectors(first_cluster)?;
        let mut buf = vec![0u8; sectors.len() * BLOCK_SIZE];
        for (i, s) in sectors.iter().enumerate() {
            let off = i * BLOCK_SIZE;
            self.dev.read_block(*s as u64, &mut buf[off..off + BLOCK_SIZE])?;
        }
        Ok((sectors, buf))
    }

    fn write_dir_entry(&mut self, sectors: &[u32], off: usize, entry: &[u8]) -> Result<(), FatRwError> {
        let sector = *sectors.get(off / BLOCK_SIZE).ok_or(FatRwError::Full)?;
        let mut buf = [0u8; BLOCK_SIZE];
        self.dev.read_block(sector as u64, &mut buf)?;
        let within = off % BLOCK_SIZE;
        buf[within..within + DIR_ENTRY_LEN].copy_from_slice(entry);
        self.dev.write_block(sector as u64, &buf)?;
        Ok(())
    }

    fn root_cluster(&self) -> u32 {
        self.bpb.root_cluster // 0 on FAT16
    }

    /// Resolve parent directory cluster and basename for `path`.
    fn resolve_parent(&mut self, path: &str) -> Result<(u32, String), FatRwError> {
        let parts: Vec<&str> = path
            .split(|c| c == '/' || c == '\\')
            .filter(|s| !s.is_empty() && *s != ".")
            .collect();
        if parts.is_empty() {
            return Err(FatRwError::BadName);
        }
        let basename = parts[parts.len() - 1].to_string();
        let mut cluster = self.root_cluster();
        for part in &parts[..parts.len() - 1] {
            let (c, _, is_dir) = self.lookup_in(cluster, part)?.ok_or(FatRwError::NotFound)?;
            if !is_dir {
                return Err(FatRwError::NotADir);
            }
            cluster = c;
        }
        Ok((cluster, basename))
    }

    /// Lookup in directory: (first_cluster, size, is_dir).
    fn lookup_in(
        &mut self,
        dir_clus: u32,
        name: &str,
    ) -> Result<Option<(u32, u32, bool)>, FatRwError> {
        let (_, dir) = self.read_dir(dir_clus)?;
        Ok(lookup_name_in_dir_bytes(&dir, name))
    }

    /// Write file contents (create or replace). 8.3 short names only for create.
    pub fn write(&mut self, path: &str, data: &[u8]) -> Result<(), FatRwError> {
        let (parent, base) = self.resolve_parent(path)?;
        let short = fat32::short_name(&base).ok_or(FatRwError::BadName)?;
        let (sectors, dir) = self.read_dir(parent)?;
        // Replace existing file: free old chain.
        if let Some((off, first, _sz, attr)) = fat32::find_dir_entry(&dir, short) {
            if attr & ATTR_DIRECTORY != 0 {
                return Err(FatRwError::NotAFile);
            }
            self.free_chain(first)?;
            let need = self.bpb.clusters_for(data.len() as u32).max(if data.is_empty() { 0 } else { 1 });
            let clusters = if need == 0 {
                Vec::new()
            } else {
                fat32::find_free_clusters(&self.fat, &self.bpb, need).ok_or(FatRwError::Full)?
            };
            for (c, v) in fat32::chain_entries(&clusters, self.bpb.fat_type) {
                self.set_fat_entry(c, v)?;
            }
            let first_c = clusters.first().copied().unwrap_or(0);
            self.write_cluster_chain(&clusters, data)?;
            let entry = fat32::dir_entry(short, 0x20, first_c, data.len() as u32);
            self.write_dir_entry(&sectors, off, &entry)?;
            self.sync_fat()?;
            return Ok(());
        }
        // Create new.
        let need = self.bpb.clusters_for(data.len() as u32).max(if data.is_empty() { 0 } else { 1 });
        let clusters = if need == 0 {
            Vec::new()
        } else {
            fat32::find_free_clusters(&self.fat, &self.bpb, need).ok_or(FatRwError::Full)?
        };
        for (c, v) in fat32::chain_entries(&clusters, self.bpb.fat_type) {
            self.set_fat_entry(c, v)?;
        }
        let first_c = clusters.first().copied().unwrap_or(0);
        self.write_cluster_chain(&clusters, data)?;
        let slot = fat32::find_free_dir_slot(&dir).ok_or(FatRwError::Full)?;
        let entry = fat32::dir_entry(short, 0x20, first_c, data.len() as u32);
        self.write_dir_entry(&sectors, slot, &entry)?;
        self.sync_fat()?;
        Ok(())
    }

    fn write_cluster_chain(&mut self, clusters: &[u32], data: &[u8]) -> Result<(), FatRwError> {
        let cb = self.bpb.cluster_bytes() as usize;
        let mut off = 0usize;
        for &c in clusters {
            let base = self.bpb.cluster_sector(c).ok_or(FatRwError::Io)?;
            for s in 0..self.bpb.sectors_per_cluster {
                let mut sec = [0u8; BLOCK_SIZE];
                if off < data.len() {
                    let n = (data.len() - off).min(BLOCK_SIZE);
                    sec[..n].copy_from_slice(&data[off..off + n]);
                    off += n;
                }
                self.dev.write_block(base as u64 + s as u64, &sec)?;
            }
            let _ = cb;
        }
        Ok(())
    }

    pub fn read(&mut self, path: &str) -> Result<Vec<u8>, FatRwError> {
        let (parent, base) = self.resolve_parent(path)?;
        let (first, size, is_dir) = self.lookup_in(parent, &base)?.ok_or(FatRwError::NotFound)?;
        if is_dir {
            return Err(FatRwError::NotAFile);
        }
        if size == 0 {
            return Ok(Vec::new());
        }
        let mut out = vec![0u8; size as usize];
        self.read_cluster_chain(first, &mut out)?;
        Ok(out)
    }

    fn read_cluster_chain(&mut self, mut first: u32, out: &mut [u8]) -> Result<(), FatRwError> {
        let mut done = 0usize;
        for _ in 0..=self.bpb.data_clusters {
            if first < 2 || done >= out.len() {
                break;
            }
            let base = self.bpb.cluster_sector(first).ok_or(FatRwError::Io)?;
            for s in 0..self.bpb.sectors_per_cluster {
                if done >= out.len() {
                    break;
                }
                let mut sec = [0u8; BLOCK_SIZE];
                self.dev.read_block(base as u64 + s as u64, &mut sec)?;
                let n = (out.len() - done).min(BLOCK_SIZE);
                out[done..done + n].copy_from_slice(&sec[..n]);
                done += n;
            }
            let raw = fat32::fat_entry(&self.fat, &self.bpb, first).ok_or(FatRwError::Io)?;
            match fat32::cluster_state(raw, self.bpb.fat_type) {
                fat32::ClusterState::Next(n) => first = n,
                _ => break,
            }
        }
        Ok(())
    }

    pub fn unlink(&mut self, path: &str) -> Result<(), FatRwError> {
        let (parent, base) = self.resolve_parent(path)?;
        let short = fat32::short_name(&base).ok_or(FatRwError::BadName)?;
        let (sectors, dir) = self.read_dir(parent)?;
        let (off, first, _sz, attr) = fat32::find_dir_entry(&dir, short).ok_or(FatRwError::NotFound)?;
        if attr & ATTR_DIRECTORY != 0 {
            return Err(FatRwError::NotAFile);
        }
        self.free_chain(first)?;
        let mut entry = [0u8; DIR_ENTRY_LEN];
        entry[0] = 0xe5; // deleted
        self.write_dir_entry(&sectors, off, &entry)?;
        self.sync_fat()?;
        Ok(())
    }

    pub fn mkdir(&mut self, path: &str) -> Result<(), FatRwError> {
        let (parent, base) = self.resolve_parent(path)?;
        let short = fat32::short_name(&base).ok_or(FatRwError::BadName)?;
        let (sectors, dir) = self.read_dir(parent)?;
        if fat32::find_dir_entry(&dir, short).is_some() {
            return Err(FatRwError::Exists);
        }
        let clusters = fat32::find_free_clusters(&self.fat, &self.bpb, 1).ok_or(FatRwError::Full)?;
        let c = clusters[0];
        self.set_fat_entry(c, match self.bpb.fat_type {
            FatType::Fat32 => 0x0fff_ffff,
            FatType::Fat16 => 0xffff,
            FatType::Fat12 => 0xfff,
        })?;
        // Initialise directory cluster with . and ..
        let base_sec = self.bpb.cluster_sector(c).ok_or(FatRwError::Io)?;
        let mut sec = [0u8; BLOCK_SIZE];
        let dot = fat32::dir_entry(*b".          ", ATTR_DIRECTORY, c, 0);
        let dotdot = fat32::dir_entry(*b"..         ", ATTR_DIRECTORY, parent, 0);
        sec[0..32].copy_from_slice(&dot);
        sec[32..64].copy_from_slice(&dotdot);
        self.dev.write_block(base_sec as u64, &sec)?;
        for s in 1..self.bpb.sectors_per_cluster {
            let z = [0u8; BLOCK_SIZE];
            self.dev.write_block(base_sec as u64 + s as u64, &z)?;
        }
        let slot = fat32::find_free_dir_slot(&dir).ok_or(FatRwError::Full)?;
        let entry = fat32::dir_entry(short, ATTR_DIRECTORY, c, 0);
        self.write_dir_entry(&sectors, slot, &entry)?;
        self.sync_fat()?;
        Ok(())
    }

    pub fn readdir(&mut self, path: &str) -> Result<Vec<(String, u32, bool)>, FatRwError> {
        let cluster = if path.is_empty() || path == "/" {
            self.root_cluster()
        } else {
            let (parent, base) = self.resolve_parent(path)?;
            let (c, _, is_dir) = self.lookup_in(parent, &base)?.ok_or(FatRwError::NotFound)?;
            if !is_dir {
                return Err(FatRwError::NotADir);
            }
            c
        };
        let (_, dir) = self.read_dir(cluster)?;
        Ok(list_dir_bytes(&dir))
    }
}

fn lookup_name_in_dir_bytes(dir: &[u8], name: &str) -> Option<(u32, u32, bool)> {
    let mut lfn = String::new();
    let mut i = 0usize;
    while i + 32 <= dir.len() {
        let e = &dir[i..i + 32];
        i += 32;
        if e[0] == 0 {
            break;
        }
        if e[0] == 0xe5 {
            lfn.clear();
            continue;
        }
        if e[11] == 0x0f {
            let mut part = String::new();
            for &off in &[1usize, 3, 5, 7, 9, 14, 16, 18, 20, 22, 24, 28, 30] {
                let c = u16::from_le_bytes([e[off], e[off + 1]]);
                if c == 0 || c == 0xffff {
                    break;
                }
                part.push(char::from_u32(c as u32).unwrap_or('?'));
            }
            part.push_str(&lfn);
            lfn = part;
            continue;
        }
        let this = if lfn.is_empty() {
            let base: String = core::str::from_utf8(&e[0..8]).unwrap_or("").trim_end().into();
            let ext: String = core::str::from_utf8(&e[8..11]).unwrap_or("").trim_end().into();
            if ext.is_empty() {
                base
            } else {
                alloc::format!("{base}.{ext}")
            }
        } else {
            core::mem::take(&mut lfn)
        };
        lfn.clear();
        if e[11] & 0x08 != 0 {
            continue;
        }
        if this.eq_ignore_ascii_case(name) {
            let clus = ((u16::from_le_bytes([e[20], e[21]]) as u32) << 16)
                | u16::from_le_bytes([e[26], e[27]]) as u32;
            let size = u32::from_le_bytes([e[28], e[29], e[30], e[31]]);
            return Some((clus, size, e[11] & ATTR_DIRECTORY != 0));
        }
    }
    None
}

fn list_dir_bytes(dir: &[u8]) -> Vec<(String, u32, bool)> {
    let mut out = Vec::new();
    let mut lfn = String::new();
    let mut i = 0usize;
    while i + 32 <= dir.len() {
        let e = &dir[i..i + 32];
        i += 32;
        if e[0] == 0 {
            break;
        }
        if e[0] == 0xe5 {
            lfn.clear();
            continue;
        }
        if e[11] == 0x0f {
            let mut part = String::new();
            for &off in &[1usize, 3, 5, 7, 9, 14, 16, 18, 20, 22, 24, 28, 30] {
                let c = u16::from_le_bytes([e[off], e[off + 1]]);
                if c == 0 || c == 0xffff {
                    break;
                }
                part.push(char::from_u32(c as u32).unwrap_or('?'));
            }
            part.push_str(&lfn);
            lfn = part;
            continue;
        }
        let name = if lfn.is_empty() {
            let base: String = core::str::from_utf8(&e[0..8]).unwrap_or("").trim_end().into();
            let ext: String = core::str::from_utf8(&e[8..11]).unwrap_or("").trim_end().into();
            if ext.is_empty() {
                base
            } else {
                alloc::format!("{base}.{ext}")
            }
        } else {
            core::mem::take(&mut lfn)
        };
        lfn.clear();
        if e[11] & 0x08 != 0 || name == "." || name == ".." {
            continue;
        }
        let size = u32::from_le_bytes([e[28], e[29], e[30], e[31]]);
        out.push((name, size, e[11] & ATTR_DIRECTORY != 0));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::fat::FatWriter;
    use crate::block::ramdisk::RamDisk;

    #[test_case]
    fn fat_rw_write_read_unlink_round_trip() {
        // FatWriter formats FAT16 and needs ≥4085 data clusters (~32 MiB).
        let mut disk = RamDisk::new(65536);
        FatWriter::format(&mut disk).expect("format");
        {
            let mut vol = FatRw::open(&mut disk).expect("open");
            vol.write("HELLO.TXT", b"world").expect("write");
            assert_eq!(vol.read("HELLO.TXT").unwrap(), b"world");
            vol.write("HELLO.TXT", b"longer content").expect("grow");
            assert_eq!(vol.read("HELLO.TXT").unwrap(), b"longer content");
            vol.mkdir("SUB").expect("mkdir");
            vol.write("SUB/A.TXT", b"nested").expect("nested");
            assert_eq!(vol.read("SUB/A.TXT").unwrap(), b"nested");
            let listing = vol.readdir("/").expect("readdir");
            assert!(listing.iter().any(|(n, _, d)| n == "SUB" && *d));
            vol.unlink("HELLO.TXT").expect("unlink");
            assert!(matches!(vol.read("HELLO.TXT"), Err(FatRwError::NotFound)));
        }
        let mut vol = FatRw::open(&mut disk).expect("reopen");
        assert_eq!(vol.read("SUB/A.TXT").unwrap(), b"nested");
    }
}
