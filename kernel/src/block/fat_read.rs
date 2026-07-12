//! A read-only **FAT16/FAT32 reader** — the counterpart to `block::fat`'s
//! writer, used by the aarch64 `/install` to read the installer payload
//! (BOOTAA64.EFI + kernel + model) off the boot ESP it was started from.
//! Handles both FAT variants (detected by cluster count, per the spec), VFAT
//! long file names (the ESP holds `chitti-kernel`, `model.gguf.000` — not
//! 8.3), and subdirectory walks (`EFI/BOOT/BOOTAA64.EFI`).

use crate::block::{BlockDevice, BLOCK_SIZE};
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

fn le16(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}
fn le32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

pub struct FatReader<'d, D: BlockDevice> {
    dev: &'d mut D,
    fat32: bool,
    spc: u64,        // sectors per cluster
    fat_lba: u64,    // first FAT sector
    root_lba: u64,   // FAT16: root-dir region start
    root_secs: u64,  // FAT16: root-dir sectors
    root_clus: u32,  // FAT32: root directory first cluster
    data_lba: u64,   // first sector of cluster 2
    /// Last FAT sector read (`(lba, data)`) — chain walks hit the same sector
    /// ~128 times in a row (FAT32: 128 entries/sector), so this one-line cache
    /// turns per-cluster FAT reads into one read per 64 KiB of file.
    fat_cache: Option<(u64, [u8; BLOCK_SIZE])>,
}

impl<'d, D: BlockDevice> FatReader<'d, D> {
    /// Open a FAT16/FAT32 filesystem at the start of `dev`.
    pub fn open(dev: &'d mut D) -> Option<FatReader<'d, D>> {
        let mut bs = [0u8; BLOCK_SIZE];
        dev.read_block(0, &mut bs).ok()?;
        if bs[510] != 0x55 || bs[511] != 0xAA {
            return None;
        }
        let bytes_per_sec = le16(&bs, 11) as u64;
        if bytes_per_sec != BLOCK_SIZE as u64 {
            return None; // only 512-byte sectors
        }
        let spc = bs[13] as u64;
        let reserved = le16(&bs, 14) as u64;
        let nfats = bs[16] as u64;
        let root_entries = le16(&bs, 17) as u64;
        let total16 = le16(&bs, 19) as u64;
        let fat16_size = le16(&bs, 22) as u64;
        let total32 = le32(&bs, 32) as u64;
        let fat32_size = le32(&bs, 36) as u64;
        let total = if total16 != 0 { total16 } else { total32 };
        let fat_size = if fat16_size != 0 { fat16_size } else { fat32_size };
        if spc == 0 || fat_size == 0 || total == 0 {
            return None;
        }
        let root_secs = (root_entries * 32).div_ceil(BLOCK_SIZE as u64);
        let fat_lba = reserved;
        let root_lba = reserved + nfats * fat_size;
        let data_lba = root_lba + root_secs;
        // FAT type is determined by the count of data clusters (the spec's rule).
        let clusters = (total - data_lba) / spc;
        let fat32 = clusters >= 65525;
        let root_clus = if fat32 { le32(&bs, 44) } else { 0 };
        Some(FatReader { dev, fat32, spc, fat_lba, root_lba, root_secs, root_clus, data_lba, fat_cache: None })
    }

    fn cluster_lba(&self, clus: u32) -> u64 {
        self.data_lba + (clus as u64 - 2) * self.spc
    }

    /// Next cluster in the chain, or None at end-of-chain.
    fn next_cluster(&mut self, clus: u32) -> Option<u32> {
        let off = clus as u64 * if self.fat32 { 4 } else { 2 };
        let lba = self.fat_lba + off / BLOCK_SIZE as u64;
        if self.fat_cache.map(|(l, _)| l) != Some(lba) {
            let mut sec = [0u8; BLOCK_SIZE];
            self.dev.read_block(lba, &mut sec).ok()?;
            self.fat_cache = Some((lba, sec));
        }
        let sec = &self.fat_cache.as_ref().unwrap().1;
        if self.fat32 {
            let n = le32(sec, (off % BLOCK_SIZE as u64) as usize) & 0x0FFF_FFFF;
            (n >= 2 && n < 0x0FFF_FFF8).then_some(n)
        } else {
            let n = le16(sec, (off % BLOCK_SIZE as u64) as usize) as u32;
            (n >= 2 && n < 0xFFF8).then_some(n)
        }
    }

    /// Collect a directory's raw bytes: the FAT16 root region, or a cluster
    /// chain (FAT32 root / any subdirectory).
    fn read_dir(&mut self, first_clus: u32) -> Vec<u8> {
        let mut out = Vec::new();
        let mut sec = [0u8; BLOCK_SIZE];
        if first_clus == 0 && !self.fat32 {
            for s in 0..self.root_secs {
                if self.dev.read_block(self.root_lba + s, &mut sec).is_err() {
                    break;
                }
                out.extend_from_slice(&sec);
            }
            return out;
        }
        let mut clus = if first_clus == 0 { self.root_clus } else { first_clus };
        // Cycle guard: a valid directory chain visits each cluster once. A
        // corrupt or mis-read FAT (which would otherwise loop forever and hang
        // the whole cooperative kernel — this froze boot when `find_on_disks`
        // scanned the ESP) is detected by a revisit and bailed immediately.
        let mut visited: Vec<u32> = Vec::new();
        loop {
            for s in 0..self.spc {
                if self.dev.read_block(self.cluster_lba(clus) + s, &mut sec).is_err() {
                    return out;
                }
                out.extend_from_slice(&sec);
            }
            visited.push(clus);
            match self.next_cluster(clus) {
                Some(n) if !visited.contains(&n) => clus = n,
                _ => return out, // EOC, invalid, or a cycle → stop
            }
        }
    }

    /// Find `name` (case-insensitive; matches LFN or 8.3) in the directory at
    /// `dir_clus` (0 = root). Returns (first_cluster, size, is_dir).
    fn lookup_in(&mut self, dir_clus: u32, name: &str) -> Option<(u32, u32, bool)> {
        let dir = self.read_dir(dir_clus);
        let mut lfn = String::new();
        let mut i = 0usize;
        while i + 32 <= dir.len() {
            let e = &dir[i..i + 32];
            i += 32;
            if e[0] == 0 {
                break; // end of directory
            }
            if e[0] == 0xE5 {
                lfn.clear();
                continue; // deleted
            }
            if e[11] == 0x0F {
                // VFAT LFN entry: 13 UCS-2 chars at fixed offsets; entries are
                // stored last-part-first, so prepend.
                let mut part = String::new();
                for &off in &[1usize, 3, 5, 7, 9, 14, 16, 18, 20, 22, 24, 28, 30] {
                    let c = le16(e, off);
                    if c == 0 || c == 0xFFFF {
                        break;
                    }
                    part.push(char::from_u32(c as u32).unwrap_or('?'));
                }
                part.push_str(&lfn);
                lfn = part;
                continue;
            }
            // Short entry: take the accumulated LFN if present, else decode 8.3.
            let this_name = if lfn.is_empty() {
                let base: String = core::str::from_utf8(&e[0..8]).unwrap_or("").trim_end().into();
                let ext: String = core::str::from_utf8(&e[8..11]).unwrap_or("").trim_end().into();
                if ext.is_empty() { base } else { alloc::format!("{base}.{ext}") }
            } else {
                core::mem::take(&mut lfn)
            };
            lfn.clear();
            if e[11] & 0x08 != 0 {
                continue; // volume label
            }
            if this_name.eq_ignore_ascii_case(name) {
                let clus = ((le16(e, 20) as u32) << 16) | le16(e, 26) as u32;
                let size = le32(e, 28);
                let is_dir = e[11] & 0x10 != 0;
                return Some((clus, size, is_dir));
            }
        }
        None
    }

    /// List the root directory: `(name, size, is_dir)` (LFN-aware).
    pub fn list_root(&mut self) -> Vec<(String, u32, bool)> {
        let dir = self.read_dir(0);
        let mut out = Vec::new();
        let mut lfn = String::new();
        let mut i = 0usize;
        while i + 32 <= dir.len() {
            let e = &dir[i..i + 32];
            i += 32;
            if e[0] == 0 {
                break;
            }
            if e[0] == 0xE5 {
                lfn.clear();
                continue;
            }
            if e[11] == 0x0F {
                let mut part = String::new();
                for &off in &[1usize, 3, 5, 7, 9, 14, 16, 18, 20, 22, 24, 28, 30] {
                    let c = le16(e, off);
                    if c == 0 || c == 0xFFFF {
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
                if ext.is_empty() { base } else { alloc::format!("{base}.{ext}") }
            } else {
                core::mem::take(&mut lfn)
            };
            lfn.clear();
            if e[11] & 0x08 != 0 || name == "." || name == ".." {
                continue; // volume label / dot entries
            }
            out.push((name, le32(e, 28), e[11] & 0x10 != 0));
        }
        out
    }

    /// Read the file at `path` (e.g. `"chitti-kernel"` or `"EFI/BOOT/BOOTAA64.EFI"`,
    /// `/`-separated, from the root) into a Vec. None if any component is missing.
    pub fn read_file(&mut self, path: &str) -> Option<Vec<u8>> {
        let size = self.file_size(path)? as usize;
        let mut out = vec![0u8; size];
        let n = self.read_file_into(path, &mut out)?;
        out.truncate(n);
        Some(out)
    }

    /// As [`read_file`](Self::read_file), but into a caller-provided buffer —
    /// so a multi-GB model can land in DMA frames instead of the kernel heap.
    /// Reads `min(file size, dst.len())` bytes; returns the count.
    pub fn read_file_into(&mut self, path: &str, dst: &mut [u8]) -> Option<usize> {
        let mut dir = 0u32; // root
        let mut parts = path.split('/').filter(|p| !p.is_empty()).peekable();
        while let Some(part) = parts.next() {
            let (clus, size, is_dir) = self.lookup_in(dir, part)?;
            if parts.peek().is_some() {
                if !is_dir {
                    return None;
                }
                dir = clus;
                continue;
            }
            // Final component: read the file's cluster chain. Contiguous
            // cluster runs are coalesced into one `read_blocks` (virtio issues
            // a single multi-sector request) straight into the output buffer —
            // a per-sector loop here made a 131 MB model take minutes to load.
            if is_dir {
                return None;
            }
            let bpc = self.spc as usize * BLOCK_SIZE;
            let want_total = (size as usize).min(dst.len());
            let out = &mut dst[..want_total];
            let mut done = 0usize;
            let mut cur = Some(clus);
            while done < out.len() {
                let c0 = cur?;
                // Extend the run [c0, c0+run) while the chain stays contiguous
                // and more clusters are needed.
                let mut run = 1usize;
                let mut tail = c0;
                cur = None;
                while done + run * bpc < out.len() {
                    match self.next_cluster(tail) {
                        Some(n) if n == tail as u32 + 1 => {
                            tail = n;
                            run += 1;
                        }
                        next => {
                            cur = next;
                            break;
                        }
                    }
                }
                let want = (out.len() - done).min(run * bpc);
                let full = want / BLOCK_SIZE * BLOCK_SIZE;
                let lba = self.cluster_lba(c0);
                if full > 0 {
                    // Read the run in <=2 MiB slices with UI/net upkeep between
                    // them: a 100+ MB contiguous model file is otherwise one
                    // blocking transfer that freezes the clock, mouse, and net
                    // stack for its whole duration (cooperative scheduler).
                    const SLICE: usize = 2 * 1024 * 1024;
                    let mut off = 0usize;
                    while off < full {
                        let n = SLICE.min(full - off);
                        self.dev.read_blocks(lba + (off / BLOCK_SIZE) as u64, &mut out[done + off..done + off + n]).ok()?;
                        off += n;
                        crate::shell::upkeep();
                    }
                    done += full;
                }
                if done < out.len() && want > full {
                    // Trailing partial sector: bounce through a sector buffer.
                    let mut sec = [0u8; BLOCK_SIZE];
                    self.dev.read_block(lba + (full / BLOCK_SIZE) as u64, &mut sec).ok()?;
                    let take = want - full;
                    out[done..done + take].copy_from_slice(&sec[..take]);
                    done += take;
                }
            }
            return Some(done);
        }
        None
    }

    /// The byte size of the file at `path` (from its directory entry), without
    /// reading the data — e.g. to size a region that is already in RAM.
    pub fn file_size(&mut self, path: &str) -> Option<u32> {
        let mut dir = 0u32;
        let mut parts = path.split('/').filter(|p| !p.is_empty()).peekable();
        while let Some(part) = parts.next() {
            let (clus, size, is_dir) = self.lookup_in(dir, part)?;
            if parts.peek().is_some() {
                if !is_dir {
                    return None;
                }
                dir = clus;
            } else {
                return (!is_dir).then_some(size);
            }
        }
        None
    }

    /// Whether the volume contains `path` (used to identify the boot ESP).
    pub fn exists(&mut self, path: &str) -> bool {
        let mut dir = 0u32;
        let mut parts = path.split('/').filter(|p| !p.is_empty()).peekable();
        while let Some(part) = parts.next() {
            match self.lookup_in(dir, part) {
                Some((clus, _, is_dir)) => {
                    if parts.peek().is_some() {
                        if !is_dir {
                            return false;
                        }
                        dir = clus;
                    } else {
                        return true;
                    }
                }
                None => return false,
            }
        }
        false
    }
}
