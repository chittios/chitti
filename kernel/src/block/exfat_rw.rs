//! **exFAT read-write** for mounted volumes (internal disks and USB).
//!
//! Builds on [`super::exfat`] (pure on-disk format) the way `fat_rw` builds on
//! `fat32`: open the volume, load the FAT and the allocation bitmap into
//! memory, and mutate through a thin `BlockDevice`.
//!
//! ## Ops
//! - `format` — create a fresh volume (boot region, FAT, allocation bitmap,
//!   up-case table, root directory)
//! - `read` / `write` (create or replace), `unlink` (files), `mkdir`,
//!   `readdir`, `stat`
//!
//! ## The allocation bitmap is load-bearing, not decorative
//! exFAT allocates by *bitmap* (one bit per cluster), and the FAT only records
//! chains. A writer that updates the FAT but not the bitmap would hand a
//! Windows/Linux reader "free" clusters that are actually in use, and it would
//! overwrite our data the next time it allocates. So every allocation sets
//! **both** — and every free clears both. This is the exFAT analogue of the
//! FAT rule "write every copy": the two structures must never disagree.
//!
//! ## Volume-dirty protocol
//! An RW `open` sets `VOLUME_DIRTY` and clears `CLEAR_TO_ZERO` (the boot
//! checksum skips `vol_flags`, so no checksum update is needed); `sync` clears
//! `VOLUME_DIRTY` and sets `CLEAR_TO_ZERO` again. A volume left dirty on disk
//! is exactly the "was not cleanly unmounted" signal Windows and `fsck_exfat`
//! expect.
//!
//! ## Two honest limitations, both refuse rather than misbehave
//! - Non-ASCII file names are refused on write (their hash cannot be
//!   reproduced by a foreign reader that folds with the volume's up-case
//!   table); reading lists full UTF-16 names.
//! - A volume without an allocation-bitmap entry is refused entirely. Linux
//!   remounts such a volume read-only; we have no read-only-exFAT path that
//!   skips it, and refusing is safer than guessing.

use super::exfat::*;
use super::{BlockDevice, BlockError, BLOCK_SIZE};
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

/// Errors from exFAT mutations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExfatError {
    NotExfat,
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

impl From<BlockError> for ExfatError {
    fn from(_: BlockError) -> Self {
        ExfatError::Io
    }
}

/// File metadata, mirroring the shape the VFS converts from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExfatStat {
    pub mode: u16,
    pub size: u64,
    pub mtime: u32,
    pub is_dir: bool,
}

/// A parsed directory entry (a file primary + its stream extension).
#[derive(Clone, Debug)]
struct Entry {
    /// Byte offset of the primary within the directory buffer.
    offset: usize,
    /// Bytes covered by the whole set (primary + stream + names).
    set_len: usize,
    start_clu: u32,
    size: u64,
    flags: u8,
    is_dir: bool,
    attr: u16,
    name: Vec<u16>,
    /// Packed modify-time fields of the primary.
    mtime: u16,
    mdate: u16,
    mt_cs: u8,
    mt_tz: u8,
}

/// One directory, fully read into memory: its cluster sectors, its bytes, and
/// a copy of the bytes as read (for diffed write-back).
struct Dir {
    start_clu: u32,
    flags: u8,
    sectors: Vec<u32>,
    old: Vec<u8>,
    bytes: Vec<u8>,
}

impl Dir {
    fn len(&self) -> usize {
        self.bytes.len()
    }
}

/// Live exFAT volume handle.
pub struct ExfatRw<'d, D: BlockDevice> {
    dev: &'d mut D,
    bpb: Bpb,
    /// The FAT (first copy), cached in memory; written to every copy on sync.
    fat: Vec<u8>,
    /// The allocation bitmap, one bit per data cluster (bit `c-2` for cluster `c`).
    bitmap: Vec<u8>,
    bitmap_clu: u32,
    bitmap_flags: u8,
    /// FAT or bitmap changed since the last [`Self::sync`].
    dirty: bool,
    writable: bool,
}

impl<'d, D: BlockDevice> ExfatRw<'d, D> {
    /// Open a volume. `writable` performs the volume-dirty handshake and allows
    /// mutation; read-only leaves the flags alone.
    pub fn open(dev: &'d mut D, writable: bool) -> Result<Self, ExfatError> {
        let mut sector = [0u8; BLOCK_SIZE];
        dev.read_block(0, &mut sector)?;
        let bpb = parse_boot(&sector).ok_or(ExfatError::NotExfat)?;

        // Verify the boot-region checksum (strictly, as the Linux driver does:
        // every u32 of sector 11 must hold the value). This catches a
        // half-written format or a corrupted boot region at mount.
        let mut region = vec![0u8; 11 * BLOCK_SIZE];
        for (i, chunk) in region.chunks_mut(BLOCK_SIZE).enumerate() {
            dev.read_block(i as u64, chunk)?;
        }
        let want = boot_checksum(&region);
        let mut cs_sec = [0u8; BLOCK_SIZE];
        dev.read_block(11, &mut cs_sec)?;
        if cs_sec.chunks_exact(4).any(|w| u32::from_le_bytes(w.try_into().unwrap()) != want) {
            return Err(ExfatError::NotExfat);
        }

        // Load the FAT.
        let fat_bytes = bpb.fat_bytes();
        let mut fat = vec![0u8; fat_bytes];
        dev.read_blocks(bpb.fat_offset as u64, &mut fat)?;

        // Load the allocation bitmap (find its entry in the root directory).
        let root = read_chain_bytes(dev, &bpb, bpb.root_cluster, 0)?;
        let (bitmap_clu, bitmap_size, bitmap_flags) =
            find_bitmap_entry(&root).ok_or(ExfatError::Unsupported)?;
        let mut bitmap = vec![0u8; bitmap_size as usize];
        read_chain_into(dev, &bpb, bitmap_clu, bitmap_flags, &mut bitmap)?;

        if writable {
            set_vol_flags(dev, (bpb.vol_flags | VOLUME_DIRTY) & !CLEAR_TO_ZERO)?;
        }

        Ok(ExfatRw {
            dev,
            bpb,
            fat,
            bitmap,
            bitmap_clu,
            bitmap_flags,
            dirty: false,
            writable,
        })
    }

    /// Flush the FAT + bitmap to disk and complete the volume-dirty handshake.
    pub fn sync(&mut self) -> Result<(), ExfatError> {
        if self.dirty {
            // FAT: every copy (a volume whose copies disagree is what fsck
            // reports as corrupt).
            for copy in 0..self.bpb.num_fats {
                let start = self.bpb.fat_offset + copy as u32 * self.bpb.fat_length;
                self.dev.write_blocks(start as u64, &self.fat)?;
            }
            // Bitmap: rewrite its chain (take the in-memory copy to break the
            // borrow — the chain walk needs `&mut self`, the data is `self`).
            let (bm_clu, bm_flags) = (self.bitmap_clu, self.bitmap_flags);
            let bm = core::mem::take(&mut self.bitmap);
            let r = self.write_chain(bm_clu, bm_flags, &bm);
            self.bitmap = bm;
            r?;
            self.dirty = false;
        }
        if self.writable {
            set_vol_flags(self.dev, (self.bpb.vol_flags & !VOLUME_DIRTY) | CLEAR_TO_ZERO)?;
        }
        Ok(())
    }

    fn set_fat(&mut self, cluster: u32, value: u32) -> Result<(), ExfatError> {
        let off = cluster as usize * 4;
        if off + 4 > self.fat.len() {
            return Err(ExfatError::Io);
        }
        self.fat[off..off + 4].copy_from_slice(&value.to_le_bytes());
        self.dirty = true;
        Ok(())
    }

    fn set_bit(&mut self, cluster: u32, used: bool) {
        let bit = (cluster - FIRST_CLUSTER) as usize;
        let (byte, mask) = (bit / 8, 1u8 << (bit % 8));
        if let Some(b) = self.bitmap.get_mut(byte) {
            if used {
                *b |= mask;
            } else {
                *b &= !mask;
            }
            self.dirty = true;
        }
    }

    fn test_bit(&self, cluster: u32) -> bool {
        let bit = (cluster - FIRST_CLUSTER) as usize;
        match self.bitmap.get(bit / 8) {
            Some(b) => b & (1 << (bit % 8)) != 0,
            None => false,
        }
    }

    /// The cluster a chain continues to from `c`, or `None` at end-of-chain.
    /// The caller handles the NoFatChain flag where chains are contiguous.
    fn next_cluster(&self, c: u32) -> Option<u32> {
        if c < FIRST_CLUSTER || c >= self.bpb.cluster_entry_space() {
            return None;
        }
        match fat_entry(&self.fat, c)? {
            EXFAT_FREE | EXFAT_BAD => None,
            v if v >= EXFAT_EOF || v >= self.bpb.cluster_entry_space() => None, // EOC
            v => Some(v),
        }
    }

    /// Allocate `n` free clusters, lowest first, chained in the FAT and marked
    /// in the bitmap.
    fn alloc_clusters(&mut self, n: usize) -> Result<Vec<u32>, ExfatError> {
        let cs = find_free_clusters(&self.fat, self.bpb.cluster_entry_space(), n).ok_or(ExfatError::Full)?;
        for (i, &c) in cs.iter().enumerate() {
            debug_assert!(!self.test_bit(c), "bitmap and FAT disagree at cluster {c}");
            let v = cs.get(i + 1).copied().unwrap_or(EXFAT_EOF);
            self.set_fat(c, v)?;
            self.set_bit(c, true);
        }
        Ok(cs)
    }

    /// Free a cluster chain (FAT entries → free, bitmap bits → clear). The next
    /// link is read **before** clearing the current entry — clearing first makes
    /// `next_cluster` see a free cluster and stop after the first.
    fn free_chain(&mut self, mut first: u32, flags: u8) -> Result<(), ExfatError> {
        if first < FIRST_CLUSTER {
            return Ok(());
        }
        for _ in 0..=self.bpb.clu_count {
            let next = if no_fat_chain(flags) {
                let n = first + 1;
                if n >= self.bpb.cluster_entry_space() { None } else { Some(n) }
            } else {
                self.next_cluster(first)
            };
            self.set_fat(first, EXFAT_FREE)?;
            self.set_bit(first, false);
            match next {
                Some(n) => first = n,
                None => break,
            }
        }
        Ok(())
    }

    // --- chain I/O (through the cached FAT) -------------------------------

    fn write_chain(&mut self, start: u32, flags: u8, data: &[u8]) -> Result<(), ExfatError> {
        let cb = self.bpb.cluster_bytes() as usize;
        let mut c = start;
        let mut off = 0usize;
        for _ in 0..=self.bpb.clu_count {
            if off >= data.len() {
                break;
            }
            let base = self.bpb.cluster_sector(c).ok_or(ExfatError::Io)?;
            let n = (data.len() - off).min(cb);
            // Batched: whole sectors first, then a zero-filled tail sector.
            let full = n / BLOCK_SIZE * BLOCK_SIZE;
            if full > 0 {
                self.dev.write_blocks(base as u64, &data[off..off + full])?;
            }
            if full < n {
                let mut tail = [0u8; BLOCK_SIZE];
                tail[..n - full].copy_from_slice(&data[off + full..off + n]);
                self.dev.write_block(base as u64 + (full / BLOCK_SIZE) as u64, &tail)?;
            }
            off += n;
            if no_fat_chain(flags) {
                c += 1;
            } else {
                c = match self.next_cluster(c) {
                    Some(nc) => nc,
                    None => break,
                };
            }
        }
        Ok(())
    }

    fn read_chain_into(&mut self, start: u32, flags: u8, out: &mut [u8]) -> Result<(), ExfatError> {
        let cb = self.bpb.cluster_bytes() as usize;
        let mut done = 0usize;
        let mut c = start;
        for _ in 0..=self.bpb.clu_count {
            if c < FIRST_CLUSTER || c >= self.bpb.cluster_entry_space() || done >= out.len() {
                break;
            }
            let base = self.bpb.cluster_sector(c).ok_or(ExfatError::Io)?;
            let n = (out.len() - done).min(cb);
            let full = n / BLOCK_SIZE * BLOCK_SIZE;
            if full > 0 {
                self.dev.read_blocks(base as u64, &mut out[done..done + full])?;
            }
            if full < n {
                let mut sec = [0u8; BLOCK_SIZE];
                self.dev.read_block(base as u64 + (full / BLOCK_SIZE) as u64, &mut sec)?;
                out[done + full..done + n].copy_from_slice(&sec[..n - full]);
            }
            done += n;
            if no_fat_chain(flags) {
                c += 1;
            } else {
                c = match self.next_cluster(c) {
                    Some(nc) => nc,
                    None => break,
                };
            }
        }
        Ok(())
    }

    // --- directory machinery ---------------------------------------------

    fn dir_sectors(&self, start: u32, flags: u8) -> Result<Vec<u32>, ExfatError> {
        let mut out = Vec::new();
        let mut c = start;
        for _ in 0..=self.bpb.clu_count {
            let base = self.bpb.cluster_sector(c).ok_or(ExfatError::Io)?;
            for i in 0..self.bpb.sect_per_clus() {
                out.push(base + i);
            }
            if no_fat_chain(flags) {
                c += 1;
            } else {
                match self.next_cluster(c) {
                    Some(n) => c = n,
                    None => break,
                }
            }
        }
        Ok(out)
    }

    fn read_dir(&mut self, start: u32, flags: u8) -> Result<Dir, ExfatError> {
        let sectors = self.dir_sectors(start, flags)?;
        let mut bytes = vec![0u8; sectors.len() * BLOCK_SIZE];
        for (i, &s) in sectors.iter().enumerate() {
            let off = i * BLOCK_SIZE;
            self.dev.read_block(s as u64, &mut bytes[off..off + BLOCK_SIZE])?;
        }
        Ok(Dir { start_clu: start, flags, sectors, old: bytes.clone(), bytes })
    }

    fn flush_dir(&mut self, d: &Dir) -> Result<(), ExfatError> {
        for (i, &s) in d.sectors.iter().enumerate() {
            let off = i * BLOCK_SIZE;
            if d.old[off..off + BLOCK_SIZE] != d.bytes[off..off + BLOCK_SIZE] {
                self.dev.write_block(s as u64, &d.bytes[off..off + BLOCK_SIZE])?;
            }
        }
        Ok(())
    }

    /// Parse the entry at `off` in a directory buffer. `Ok(None)` at the
    /// end-of-directory marker; otherwise `(entry, bytes_to_advance)`.
    fn parse_entry(&self, bytes: &[u8], off: usize) -> Result<Option<(Entry, usize)>, ExfatError> {
        if off + DENTRY_LEN > bytes.len() {
            return Ok(None);
        }
        let t = bytes[off];
        if t == 0x00 || t & 0x80 == 0 {
            // End marker or a deleted entry (in-use bit cleared). A deleted
            // entry advances one slot; the end marker stops the scan.
            return Ok(if t == 0x00 { None } else { Some((empty_entry(off), DENTRY_LEN)) });
        }
        if t != TYPE_FILE {
            // Volume label / bitmap / upcase / guid / padding: not a file.
            return Ok(Some((empty_entry(off), DENTRY_LEN)));
        }
        let num_ext = bytes[off + 1] as usize;
        if num_ext < 1 || num_ext > 32 {
            return Err(ExfatError::Io); // corrupt: refuse rather than skip
        }
        let set_len = (num_ext + 1) * DENTRY_LEN;
        if off + set_len > bytes.len() {
            return Ok(None);
        }
        let s_off = off + DENTRY_LEN;
        if bytes[s_off] != TYPE_STREAM {
            return Err(ExfatError::Io); // file entry with no stream
        }
        let name_len = bytes[s_off + 3] as usize;
        let start_clu = le32(bytes, s_off + 20);
        let size = le64(bytes, s_off + 24);
        let flags = bytes[s_off + 1];
        let attr = le16(bytes, off + 4);
        // Name: `name_len` UTF-16 units across the name entries. The cursor is
        // bounded by the set itself — a lying `name_len` must not walk the
        // cursor past this file's secondaries into whatever follows.
        let mut name = Vec::with_capacity(name_len);
        let mut cursor = off + 2 * DENTRY_LEN;
        while name.len() < name_len {
            if cursor + DENTRY_LEN > off + set_len || bytes[cursor] != TYPE_NAME {
                return Err(ExfatError::Io);
            }
            for slot in bytes[cursor + 2..cursor + DENTRY_LEN].chunks_exact(2) {
                let u = u16::from_le_bytes([slot[0], slot[1]]);
                if u == 0 || name.len() >= name_len {
                    break;
                }
                name.push(u);
            }
            cursor += DENTRY_LEN;
        }
        Ok(Some((
            Entry {
                offset: off,
                set_len,
                start_clu,
                size,
                flags,
                is_dir: attr & ATTR_SUBDIR != 0,
                attr,
                name,
                mtime: le16(bytes, off + 12),
                mdate: le16(bytes, off + 14),
                mt_cs: bytes[off + 21],
                mt_tz: bytes[off + 23],
            },
            set_len,
        )))
    }

    /// Look up `name` in a directory buffer, returning its `Entry`.
    fn lookup(&self, d: &Dir, name: &[u16]) -> Result<Option<Entry>, ExfatError> {
        let mut off = 0usize;
        while off < d.bytes.len() {
            match self.parse_entry(&d.bytes, off)? {
                None => break,
                Some((e, set_len)) => {
                    if !e.name.is_empty() && name_eq(&e.name, name) {
                        return Ok(Some(e));
                    }
                    off += set_len;
                }
            }
        }
        Ok(None)
    }

    /// Resolve a path (split on `/`) into a directory's in-memory view; the
    /// final component must be a directory.
    fn resolve_dir(&mut self, parts: &[&str]) -> Result<Dir, ExfatError> {
        let mut cur = self.read_dir(self.bpb.root_cluster, 0)?;
        for part in parts {
            let name = utf16_from_str(part);
            match self.lookup(&cur, &name)? {
                Some(e) if e.is_dir => cur = self.read_dir(e.start_clu, e.flags)?,
                Some(_) => return Err(ExfatError::NotADir),
                None => return Err(ExfatError::NotFound),
            }
        }
        Ok(cur)
    }

    /// First byte offset of a run of `n` free slots in `d` (end markers and
    /// deleted entries both count as free).
    fn find_empty(&self, d: &Dir, n: usize) -> Option<usize> {
        let mut run = 0usize;
        let mut run_start = 0usize;
        let mut off = 0usize;
        while off + DENTRY_LEN <= d.bytes.len() {
            let t = d.bytes[off];
            if t == 0x00 || t & 0x80 == 0 {
                if run == 0 {
                    run_start = off;
                }
                run += 1;
                if run >= n {
                    return Some(run_start);
                }
            } else {
                run = 0;
            }
            off += DENTRY_LEN;
        }
        None
    }

    /// Append one zeroed cluster to directory `d` (in-memory only; the caller
    /// persists it via [`Self::flush_dir`] and updates the stream size).
    fn grow_dir(&mut self, d: &mut Dir) -> Result<(), ExfatError> {
        let new = self.alloc_clusters(1)?.remove(0);
        // Link the previous tail to the new cluster.
        let mut tail = d.start_clu;
        let mut c = d.start_clu;
        for _ in 0..=self.bpb.clu_count {
            match self.next_cluster(c) {
                Some(n) => {
                    tail = n;
                    c = n;
                }
                None => break,
            }
        }
        self.set_fat(tail, new)?;
        self.set_fat(new, EXFAT_EOF)?;
        let base = self.bpb.cluster_sector(new).ok_or(ExfatError::Io)?;
        let zeros = vec![0u8; self.bpb.cluster_bytes() as usize];
        self.dev.write_blocks(base as u64, &zeros)?;
        for i in 0..self.bpb.sect_per_clus() {
            d.sectors.push(base + i);
        }
        d.bytes.extend_from_slice(&zeros);
        d.old.extend_from_slice(&zeros);
        Ok(())
    }

    /// Update a directory's own stream entry (its `size`/`valid_size`) after it
    /// grew. `dir_parts` is the directory's own path (its set lives in its
    /// parent). The root directory has no stream entry and needs no update.
    fn update_dir_size(&mut self, dir_parts: &[&str], new_size: u64) -> Result<(), ExfatError> {
        if dir_parts.is_empty() {
            return Ok(());
        }
        let gp_parts = &dir_parts[..dir_parts.len() - 1];
        let base_name = utf16_from_str(dir_parts[dir_parts.len() - 1]);
        let mut gp = self.resolve_dir(gp_parts)?;
        let e = self.lookup(&gp, &base_name)?.ok_or(ExfatError::NotFound)?;
        if !e.is_dir {
            return Err(ExfatError::NotADir);
        }
        let s_off = e.offset + DENTRY_LEN;
        gp.bytes[s_off + 8..s_off + 16].copy_from_slice(&new_size.to_le_bytes());
        gp.bytes[s_off + 24..s_off + 32].copy_from_slice(&new_size.to_le_bytes());
        self.rewrite_set_checksum(&mut gp, e.offset, e.set_len)?;
        self.flush_dir(&gp)?;
        Ok(())
    }

    fn rewrite_set_checksum(&self, d: &mut Dir, off: usize, set_len: usize) -> Result<(), ExfatError> {
        let entries: Vec<&[u8]> = (0..set_len / DENTRY_LEN)
            .map(|k| &d.bytes[off + k * DENTRY_LEN..off + (k + 1) * DENTRY_LEN])
            .collect();
        let cs = entry_set_checksum(&entries);
        d.bytes[off + 2..off + 4].copy_from_slice(&cs.to_le_bytes());
        Ok(())
    }

    /// Place an entry set at `off`, writing its checksum into the primary.
    fn place_entry_set(&self, d: &mut Dir, off: usize, set: &[[u8; DENTRY_LEN]]) -> Result<(), ExfatError> {
        if off + set.len() * DENTRY_LEN > d.bytes.len() {
            return Err(ExfatError::Full);
        }
        for (k, e) in set.iter().enumerate() {
            let dst = off + k * DENTRY_LEN;
            d.bytes[dst..dst + DENTRY_LEN].copy_from_slice(e);
        }
        let entries: Vec<&[u8]> = set.iter().map(|e| &e[..]).collect();
        let cs = entry_set_checksum(&entries);
        d.bytes[off + 2..off + 4].copy_from_slice(&cs.to_le_bytes());
        Ok(())
    }

    /// Ensure directory `d` has `n` free slots, growing it (and updating its
    /// own stream size) if it does not. Returns the run offset.
    fn ensure_slots(&mut self, d: &mut Dir, parent_parts: &[&str], n: usize) -> Result<usize, ExfatError> {
        if let Some(off) = self.find_empty(d, n) {
            return Ok(off);
        }
        let before = d.len();
        self.grow_dir(d)?;
        self.update_dir_size(parent_parts, (before + self.bpb.cluster_bytes() as usize) as u64)?;
        self.find_empty(d, n).ok_or(ExfatError::Full)
    }

    // --- public ops --------------------------------------------------------

    pub fn read(&mut self, path: &str) -> Result<Vec<u8>, ExfatError> {
        let (parent, base) = split_path(path)?;
        let parent_dir = self.resolve_dir(&parent)?;
        let e = self.lookup(&parent_dir, &base)?.ok_or(ExfatError::NotFound)?;
        if e.is_dir {
            return Err(ExfatError::NotAFile);
        }
        if e.size == 0 || e.start_clu < FIRST_CLUSTER {
            return Ok(Vec::new());
        }
        let mut out = vec![0u8; e.size as usize];
        self.read_chain_into(e.start_clu, e.flags, &mut out)?;
        Ok(out)
    }

    /// Write `data` to `path`, creating the file or replacing it. Replacing
    /// allocates the new chain **first**, then frees the old one and places a
    /// fresh entry set — so a full-volume error leaves the old file intact
    /// rather than having already freed it. The existing entry is checked for
    /// being a directory *before* any allocation, so that refusal leaks nothing.
    pub fn write(&mut self, path: &str, data: &[u8]) -> Result<(), ExfatError> {
        let (parent, base) = split_path(path)?;
        let mut parent_dir = self.resolve_dir(&parent)?;
        let now = crate::clock::now_unix();

        let existing = self.lookup(&parent_dir, &base)?;
        if let Some(e) = &existing {
            if e.is_dir {
                return Err(ExfatError::NotAFile);
            }
        }

        let need = data.len().div_ceil(self.bpb.cluster_bytes() as usize).max(if data.is_empty() { 0 } else { 1 });
        let clusters = if need == 0 { Vec::new() } else { self.alloc_clusters(need)? };
        if !clusters.is_empty() {
            self.write_chain(clusters[0], 0, data)?;
        }
        let first = clusters.first().copied().unwrap_or(0);
        let set = build_entry_set(&base, false, first, data.len() as u64, now);

        if let Some(e) = existing {
            self.free_chain(e.start_clu, e.flags)?;
            self.mark_deleted(&mut parent_dir, &e);
        }

        let off = self.ensure_slots(&mut parent_dir, &parent, set.len())?;
        self.place_entry_set(&mut parent_dir, off, &set)?;
        self.flush_dir(&parent_dir)?;
        self.sync()?;
        Ok(())
    }

    fn mark_deleted(&self, d: &mut Dir, e: &Entry) {
        for k in 0..e.set_len / DENTRY_LEN {
            d.bytes[e.offset + k * DENTRY_LEN] &= 0x7f;
        }
    }

    pub fn unlink(&mut self, path: &str) -> Result<(), ExfatError> {
        let (parent, base) = split_path(path)?;
        let mut parent_dir = self.resolve_dir(&parent)?;
        let e = self.lookup(&parent_dir, &base)?.ok_or(ExfatError::NotFound)?;
        if e.is_dir {
            return Err(ExfatError::NotAFile);
        }
        self.free_chain(e.start_clu, e.flags)?;
        self.mark_deleted(&mut parent_dir, &e);
        self.flush_dir(&parent_dir)?;
        self.sync()?;
        Ok(())
    }

    pub fn mkdir(&mut self, path: &str) -> Result<(), ExfatError> {
        let (parent, base) = split_path(path)?;
        let mut parent_dir = self.resolve_dir(&parent)?;
        if self.lookup(&parent_dir, &base)?.is_some() {
            return Err(ExfatError::Exists);
        }
        let cb = self.bpb.cluster_bytes() as u64;
        let clu = self.alloc_clusters(1)?.remove(0);
        // Zero the new directory's cluster.
        let base_sec = self.bpb.cluster_sector(clu).ok_or(ExfatError::Io)?;
        let zeros = vec![0u8; cb as usize];
        self.dev.write_blocks(base_sec as u64, &zeros)?;
        let now = crate::clock::now_unix();
        let set = build_entry_set(&base, true, clu, cb, now);
        let off = self.ensure_slots(&mut parent_dir, &parent, set.len())?;
        self.place_entry_set(&mut parent_dir, off, &set)?;
        self.flush_dir(&parent_dir)?;
        self.sync()?;
        Ok(())
    }

    pub fn readdir(&mut self, path: &str) -> Result<Vec<(String, u64, bool)>, ExfatError> {
        let dir = if path.is_empty() || path == "/" {
            self.read_dir(self.bpb.root_cluster, 0)?
        } else {
            let parts: Vec<&str> = path.split('/').filter(|s| !s.is_empty() && *s != ".").collect();
            self.resolve_dir(&parts)?
        };
        let mut out = Vec::new();
        let mut off = 0usize;
        while off < dir.len() {
            match self.parse_entry(&dir.bytes, off)? {
                None => break,
                Some((e, set_len)) => {
                    if !e.name.is_empty() && e.attr & ATTR_VOLUME == 0 {
                        out.push((str_from_utf16(&e.name), e.size, e.is_dir));
                    }
                    off += set_len;
                }
            }
        }
        Ok(out)
    }

    pub fn stat(&mut self, path: &str) -> Result<ExfatStat, ExfatError> {
        let (parent, base) = split_path(path)?;
        let parent_dir = self.resolve_dir(&parent)?;
        let e = self.lookup(&parent_dir, &base)?.ok_or(ExfatError::NotFound)?;
        Ok(ExfatStat {
            mode: if e.is_dir { 0x4000 | 0o755 } else { 0x8000 | 0o644 },
            size: e.size,
            mtime: unpack_time(e.mtime, e.mdate, e.mt_cs, e.mt_tz) as u32,
            is_dir: e.is_dir,
        })
    }

    /// Free space on the volume, in bytes.
    pub fn free_bytes(&self) -> u64 {
        let free = count_free(&self.fat, self.bpb.cluster_entry_space()) as u64;
        free * self.bpb.cluster_bytes() as u64
    }
}

/// Split a `/`-path into `(parent components, base name as UTF-16)`, refusing
/// an empty path or a base name that cannot round-trip. A free function (not a
/// method) so the returned `&str` slices borrow `path`, never the handle.
fn split_path(path: &str) -> Result<(Vec<&str>, Vec<u16>), ExfatError> {
    let parts: Vec<&str> = path
        .split('/')
        .filter(|s| !s.is_empty() && *s != ".")
        .collect();
    if parts.is_empty() {
        return Err(ExfatError::BadName);
    }
    let base = utf16_from_str(parts[parts.len() - 1]);
    if !name_valid(&base) {
        return Err(ExfatError::BadName);
    }
    Ok((parts[..parts.len() - 1].to_vec(), base))
}

/// A placeholder `Entry` for a non-file or deleted directory slot.
fn empty_entry(offset: usize) -> Entry {
    Entry {
        offset,
        set_len: DENTRY_LEN,
        start_clu: 0,
        size: 0,
        flags: 0,
        is_dir: false,
        attr: 0,
        name: Vec::new(),
        mtime: 0,
        mdate: 0,
        mt_cs: 0,
        mt_tz: 0,
    }
}

/// Build an entry set (file primary + stream + name entries).
fn build_entry_set(name: &[u16], is_dir: bool, start_clu: u32, size: u64, unix: i64) -> Vec<[u8; DENTRY_LEN]> {
    let name_entries = name_entry_count(name.len());
    let num_ext = (1 + name_entries) as u8;
    let attr = if is_dir { ATTR_SUBDIR } else { ATTR_ARCHIVE };
    let mut set = Vec::with_capacity(2 + name_entries);
    set.push(file_entry(attr, num_ext, unix));
    set.push(stream_entry(name.len() as u8, name_hash(name), start_clu, size, size, FLAGS_FAT_CHAIN));
    for k in 0..name_entries {
        set.push(name_entry(name, k * NAME_UNITS));
    }
    set
}

// --- free-standing helpers (used during `open`, before a handle exists) ---

fn set_vol_flags<D: BlockDevice>(dev: &mut D, flags: u16) -> Result<(), ExfatError> {
    let mut sector = [0u8; BLOCK_SIZE];
    dev.read_block(0, &mut sector)?;
    sector[106..108].copy_from_slice(&flags.to_le_bytes());
    dev.write_block(0, &sector)?;
    Ok(())
}

/// Read one FAT entry directly from the device (used only by the open-path
/// chain readers, which have no cached FAT yet).
fn fat_entry_on_dev<D: BlockDevice>(dev: &mut D, bpb: &Bpb, cluster: u32) -> Option<u32> {
    let (sec, within) = bpb.fat_entry_location(0, cluster)?;
    let mut sector = [0u8; BLOCK_SIZE];
    dev.read_block(sec as u64, &mut sector).ok()?;
    Some(u32::from_le_bytes([
        sector[within as usize],
        sector[within as usize + 1],
        sector[within as usize + 2],
        sector[within as usize + 3],
    ]))
}

/// Read a whole cluster chain into a fresh `Vec` (open path: root directory,
/// so the chain is short and per-cluster FAT reads are fine).
fn read_chain_bytes<D: BlockDevice>(dev: &mut D, bpb: &Bpb, start: u32, flags: u8) -> Result<Vec<u8>, ExfatError> {
    let mut out = Vec::new();
    let mut c = start;
    for _ in 0..=bpb.clu_count {
        if c < FIRST_CLUSTER || c >= bpb.cluster_entry_space() {
            break;
        }
        let base = bpb.cluster_sector(c).ok_or(ExfatError::Io)?;
        let mut chunk = vec![0u8; bpb.cluster_bytes() as usize];
        dev.read_blocks(base as u64, &mut chunk)?;
        out.extend_from_slice(&chunk);
        if no_fat_chain(flags) {
            c += 1;
        } else {
            match fat_entry_on_dev(dev, bpb, c) {
                Some(v) if v < bpb.cluster_entry_space() && v >= FIRST_CLUSTER => c = v,
                _ => break,
            }
        }
    }
    Ok(out)
}

fn read_chain_into<D: BlockDevice>(
    dev: &mut D,
    bpb: &Bpb,
    start: u32,
    flags: u8,
    out: &mut [u8],
) -> Result<(), ExfatError> {
    let cb = bpb.cluster_bytes() as usize;
    let mut done = 0usize;
    let mut c = start;
    for _ in 0..=bpb.clu_count {
        if c < FIRST_CLUSTER || c >= bpb.cluster_entry_space() || done >= out.len() {
            break;
        }
        let base = bpb.cluster_sector(c).ok_or(ExfatError::Io)?;
        let n = (out.len() - done).min(cb);
        let full = n / BLOCK_SIZE * BLOCK_SIZE;
        if full > 0 {
            dev.read_blocks(base as u64, &mut out[done..done + full])?;
        }
        if full < n {
            let mut sec = [0u8; BLOCK_SIZE];
            dev.read_block(base as u64 + (full / BLOCK_SIZE) as u64, &mut sec)?;
            out[done + full..done + n].copy_from_slice(&sec[..n - full]);
        }
        done += n;
        if no_fat_chain(flags) {
            c += 1;
        } else {
            match fat_entry_on_dev(dev, bpb, c) {
                Some(v) if v < bpb.cluster_entry_space() && v >= FIRST_CLUSTER => c = v,
                _ => break,
            }
        }
    }
    Ok(())
}

/// Locate the allocation-bitmap primary in a root-directory buffer.
fn find_bitmap_entry(root: &[u8]) -> Option<(u32, u64, u8)> {
    let mut off = 0usize;
    while off + DENTRY_LEN <= root.len() {
        let t = root[off];
        if t == 0x00 {
            return None;
        }
        if t == TYPE_BITMAP {
            return Some((le32(root, off + 20), le64(root, off + 24), root[off + 1]));
        }
        off += DENTRY_LEN;
    }
    None
}

// --- format ---------------------------------------------------------------

/// Format `dev` as a fresh exFAT volume (label optional). The whole device
/// becomes one volume. Refuses geometries that cannot fit the format's
/// invariants (cluster count out of range, volume too small).
pub fn format<D: BlockDevice>(dev: &mut D, label: &str) -> Result<(), ExfatError> {
    let total = dev.block_count();
    if total < 32 {
        return Err(ExfatError::Unsupported);
    }

    // Pick a cluster size: 4 KiB up, large enough that the cluster count fits
    // the 32-bit index space, small enough that the volume still has a
    // reasonable number of clusters.
    let mut spc_bits = 3u8;
    let mut fat_len;
    let mut clu_off;
    let mut clu_count;
    loop {
        let spc = 1u64 << spc_bits;
        fat_len = ((total / spc).max(1) + 2) * 4 / BLOCK_SIZE as u64 + 1;
        clu_off = round_up(24 + fat_len, spc);
        clu_count = if clu_off < total { (total - clu_off) / spc } else { 0 };
        // Recompute the FAT length for the actual cluster count, then re-align
        // the cluster heap (it must sit on a cluster boundary).
        let need = ((clu_count + 2) * 4).div_ceil(BLOCK_SIZE as u64);
        if need > fat_len {
            fat_len = need;
            clu_off = round_up(24 + fat_len, spc);
            clu_count = if clu_off < total { (total - clu_off) / spc } else { 0 };
        }
        if clu_count as u32 <= EXFAT_MAX_CLUSTER && clu_count >= 8 {
            break;
        }
        if spc_bits >= 19 {
            return Err(ExfatError::Unsupported);
        }
        spc_bits += 1;
    }
    let fat_len = fat_len as u32;
    let clu_off = clu_off as u32;
    let clu_count = clu_count as u32;

    let bpb = Bpb {
        bytes_per_sector: BLOCK_SIZE as u32,
        sect_per_clus_bits: spc_bits,
        num_fats: 1,
        fat_offset: 24,
        fat_length: fat_len,
        clu_offset: clu_off,
        clu_count,
        root_cluster: FIRST_CLUSTER,
        vol_flags: 0,
    };

    // --- boot region (12 sectors) ---
    let mut s0 = [0u8; BLOCK_SIZE];
    s0[0] = 0xEB;
    s0[1] = 0x76;
    s0[2] = 0x90;
    s0[3..11].copy_from_slice(b"EXFAT   ");
    s0[64..72].copy_from_slice(&0u64.to_le_bytes()); // partition offset (whole disk)
    s0[72..80].copy_from_slice(&total.to_le_bytes());
    s0[80..84].copy_from_slice(&24u32.to_le_bytes()); // fat_offset
    s0[84..88].copy_from_slice(&fat_len.to_le_bytes());
    s0[88..92].copy_from_slice(&clu_off.to_le_bytes());
    s0[92..96].copy_from_slice(&clu_count.to_le_bytes());
    s0[96..100].copy_from_slice(&FIRST_CLUSTER.to_le_bytes());
    s0[100..104].copy_from_slice(&0x12345678u32.to_le_bytes()); // serial
    s0[104] = 0x00;
    s0[105] = 0x01; // fs_revision 1.00
    s0[108] = 9; // 512-byte sectors
    s0[109] = spc_bits;
    s0[110] = 1; // num_fats
    s0[111] = 0x80; // drive select
    s0[510] = 0x55;
    s0[511] = 0xAA;
    dev.write_block(0, &s0)?;
    for sec in 1..=10 {
        let mut s = [0u8; BLOCK_SIZE];
        if sec == 1 || sec == 2 {
            s[3..11].copy_from_slice(b"EXFAT   ");
        }
        if sec == 1 {
            // Extended-boot signature 0xAA550000 (LE) at the tail.
            s[508..512].copy_from_slice(&0xAA550000u32.to_le_bytes());
        }
        if sec == 2 {
            s[510] = 0x55;
            s[511] = 0xAA;
        }
        dev.write_block(sec, &s)?;
    }
    // Boot-checksum sector: the checksum of sectors 0..10, repeated.
    let mut region = vec![0u8; 11 * BLOCK_SIZE];
    for i in 0..11 {
        dev.read_block(i as u64, &mut region[i * BLOCK_SIZE..(i + 1) * BLOCK_SIZE])?;
    }
    let cs = boot_checksum(&region);
    let mut cs_sec = [0u8; BLOCK_SIZE];
    for w in cs_sec.chunks_exact_mut(4) {
        w.copy_from_slice(&cs.to_le_bytes());
    }
    dev.write_block(11, &cs_sec)?;

    // --- FAT ---
    // Build the whole FAT in memory (chain entries set below), then write it
    // once, batched. `vec![0]` gives every free cluster a 0 entry.
    let fat_bytes = bpb.fat_bytes() as usize;
    let mut fat = vec![0u8; fat_bytes];
    fat[0..4].copy_from_slice(&0xFFFF_FFF8u32.to_le_bytes()); // media descriptor
    fat[4..8].copy_from_slice(&EXFAT_EOF.to_le_bytes());

    // --- cluster layout: 2 = root, 3.. = bitmap, then up-case table ---
    let bitmap_bytes = clu_count.div_ceil(8) as u64;
    let bitmap_clusters = bitmap_bytes.div_ceil(bpb.cluster_bytes() as u64).max(1) as u32;
    let upcase_clu = FIRST_CLUSTER + 1 + bitmap_clusters;
    if upcase_clu >= bpb.cluster_entry_space() {
        return Err(ExfatError::Unsupported);
    }

    // Up-case table: the compressed ASCII-fold table (30 units, 60 bytes).
    let table = ascii_upcase_table();
    let mut table_bytes = Vec::with_capacity(table.len() * 2);
    for &v in &table {
        table_bytes.extend_from_slice(&v.to_le_bytes());
    }
    let upcase_checksum = chksum32(&table_bytes);

    // Mark every cluster we lay down (root, bitmap, upcase) used in the FAT
    // and the bitmap. Each allocation is its **own** chain: the root directory
    // is cluster 2 alone, the bitmap chain is 3.., the upcase is `upcase_clu`.
    // Linking them into one chain (as a first cut did) makes the root
    // directory span the bitmap/upcase clusters, whose bytes then parse as
    // garbage directory entries and get overwritten as files are created.
    let last = upcase_clu;
    fat[FIRST_CLUSTER as usize * 4..(FIRST_CLUSTER + 1) as usize * 4].copy_from_slice(&EXFAT_EOF.to_le_bytes());
    if bitmap_clusters >= 1 {
        let bmap_first = FIRST_CLUSTER + 1;
        for c in bmap_first..bmap_first + bitmap_clusters - 1 {
            let off = c as usize * 4;
            fat[off..off + 4].copy_from_slice(&(c + 1).to_le_bytes());
        }
        let last_bmap = bmap_first + bitmap_clusters - 1;
        let off = last_bmap as usize * 4;
        fat[off..off + 4].copy_from_slice(&EXFAT_EOF.to_le_bytes());
    }
    fat[upcase_clu as usize * 4..(upcase_clu + 1) as usize * 4].copy_from_slice(&EXFAT_EOF.to_le_bytes());
    dev.write_blocks(bpb.fat_offset as u64, &fat)?;

    let mut bitmap = vec![0u8; bitmap_bytes as usize];
    for c in FIRST_CLUSTER..=last {
        let bit = (c - FIRST_CLUSTER) as usize;
        bitmap[bit / 8] |= 1 << (bit % 8);
    }
    // Write the bitmap chain (clusters 3..).
    {
        let mut buf = vec![0u8; (bitmap_clusters as usize) * bpb.cluster_bytes() as usize];
        buf[..bitmap.len()].copy_from_slice(&bitmap);
        let base = bpb.cluster_sector(FIRST_CLUSTER + 1).ok_or(ExfatError::Io)?;
        dev.write_blocks(base as u64, &buf)?;
    }
    // Write the up-case table cluster.
    {
        let mut buf = vec![0u8; bpb.cluster_bytes() as usize];
        buf[..table_bytes.len()].copy_from_slice(&table_bytes);
        let base = bpb.cluster_sector(upcase_clu).ok_or(ExfatError::Io)?;
        dev.write_blocks(base as u64, &buf)?;
    }
    // Zero the root directory cluster, then write its entries.
    {
        let base = bpb.cluster_sector(FIRST_CLUSTER).ok_or(ExfatError::Io)?;
        let mut root = vec![0u8; bpb.cluster_bytes() as usize];
        let mut off = 0usize;
        // Volume label (0x83), if any.
        let label_u16: Vec<u16> = label.encode_utf16().take(11).collect();
        if !label_u16.is_empty() {
            let mut e = [0u8; DENTRY_LEN];
            e[0] = TYPE_VOLUME;
            e[1] = label_u16.len() as u8;
            for (k, &c) in label_u16.iter().enumerate() {
                e[2 + k * 2..4 + k * 2].copy_from_slice(&c.to_le_bytes());
            }
            root[off..off + DENTRY_LEN].copy_from_slice(&e);
            off += DENTRY_LEN;
        }
        // Allocation bitmap (0x81).
        let mut e = [0u8; DENTRY_LEN];
        e[0] = TYPE_BITMAP;
        e[20..24].copy_from_slice(&(FIRST_CLUSTER + 1).to_le_bytes());
        e[24..32].copy_from_slice(&bitmap_bytes.to_le_bytes());
        root[off..off + DENTRY_LEN].copy_from_slice(&e);
        off += DENTRY_LEN;
        // Up-case table (0x82).
        let mut e = [0u8; DENTRY_LEN];
        e[0] = TYPE_UPCASE;
        e[4..8].copy_from_slice(&upcase_checksum.to_le_bytes());
        e[20..24].copy_from_slice(&upcase_clu.to_le_bytes());
        e[24..32].copy_from_slice(&(table_bytes.len() as u64).to_le_bytes());
        root[off..off + DENTRY_LEN].copy_from_slice(&e);
        dev.write_blocks(base as u64, &root)?;
    }
    Ok(())
}

fn round_up(v: u64, align: u64) -> u64 {
    if align == 0 {
        v
    } else {
        v.div_ceil(align) * align
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::ramdisk::RamDisk;

    fn small_disk() -> RamDisk {
        // ~8 MiB — plenty of clusters at 4 KiB, tiny FAT.
        RamDisk::new(16384)
    }

    #[test_case]
    fn format_open_round_trip() {
        let mut disk = small_disk();
        format(&mut disk, "CHITTI").expect("format");
        let mut vol = ExfatRw::open(&mut disk, false).expect("open");
        assert!(vol.free_bytes() > 0);
        assert_eq!(vol.readdir("/").unwrap().is_empty(), true);
        // The volume-dirty handshake: a writable open marks dirty, sync cleans.
        let mut vol = ExfatRw::open(&mut disk, true).unwrap();
        vol.sync().unwrap();
    }

    #[test_case]
    fn write_read_unlink_mkdir_round_trip() {
        let mut disk = small_disk();
        format(&mut disk, "CHITTI").expect("format");
        {
            let mut vol = ExfatRw::open(&mut disk, true).expect("open");
            vol.write("HELLO.TXT", b"world").expect("write");
            assert_eq!(vol.read("HELLO.TXT").unwrap(), b"world");
            vol.write("HELLO.TXT", b"longer content here").expect("grow");
            assert_eq!(vol.read("HELLO.TXT").unwrap(), b"longer content here");
            vol.mkdir("SUB").expect("mkdir");
            vol.write("SUB/A.TXT", b"nested file").expect("nested");
            assert_eq!(vol.read("SUB/A.TXT").unwrap(), b"nested file");
            let listing = vol.readdir("/").unwrap();
            assert!(listing.iter().any(|(n, _, d)| n == "SUB" && *d), "{listing:?}");
            assert!(listing.iter().any(|(n, _, d)| n == "HELLO.TXT" && !*d), "{listing:?}");
            let st = vol.stat("SUB/A.TXT").unwrap();
            assert!(!st.is_dir);
            assert_eq!(st.size, 11);
            vol.unlink("HELLO.TXT").expect("unlink");
            assert!(matches!(vol.read("HELLO.TXT"), Err(ExfatError::NotFound)));
            // The freed cluster is reused.
            vol.write("BIG.BIN", &vec![0xabu8; 40_000]).expect("big");
            assert_eq!(vol.read("BIG.BIN").unwrap().len(), 40_000);
        }
        // Remount: the data lives in the device, not in filesystem memory.
        let mut vol = ExfatRw::open(&mut disk, true).expect("reopen");
        assert_eq!(vol.read("SUB/A.TXT").unwrap(), b"nested file");
        assert_eq!(vol.read("BIG.BIN").unwrap().len(), 40_000);
        assert!(vol.read("HELLO.TXT").is_err());
    }

    #[test_case]
    fn overwrite_frees_the_old_chain() {
        let mut disk = small_disk();
        format(&mut disk, "").expect("format");
        let mut vol = ExfatRw::open(&mut disk, true).expect("open");
        vol.write("F", &vec![0u8; 9000]).unwrap(); // 3 clusters
        let before = vol.free_bytes();
        vol.write("F", b"tiny").unwrap(); // 1 cluster
        assert!(vol.free_bytes() > before, "old chain should be freed");
        assert_eq!(vol.read("F").unwrap(), b"tiny");
    }

    #[test_case]
    fn refuses_what_would_not_round_trip() {
        let mut disk = small_disk();
        format(&mut disk, "").expect("format");
        let mut vol = ExfatRw::open(&mut disk, true).expect("open");
        assert_eq!(vol.write("bad/name", b"x"), Err(ExfatError::NotFound));
        assert_eq!(vol.write("trailing ", b"x"), Err(ExfatError::BadName));
        assert_eq!(vol.write("café.txt", b"x"), Err(ExfatError::BadName));
        assert_eq!(vol.write("", b"x"), Err(ExfatError::BadName));
        // mkdir then file-in-place-of-dir is refused, as is unlink-of-dir.
        vol.mkdir("DIR").unwrap();
        assert_eq!(vol.write("DIR", b"x"), Err(ExfatError::NotAFile));
        assert_eq!(vol.unlink("DIR"), Err(ExfatError::NotAFile));
        assert_eq!(vol.mkdir("DIR"), Err(ExfatError::Exists));
    }

    #[test_case]
    fn a_nested_file_grows_its_directory() {
        // Many names -> the parent directory must grow past one cluster.
        let mut disk = small_disk();
        format(&mut disk, "").expect("format");
        let mut vol = ExfatRw::open(&mut disk, true).expect("open");
        for i in 0..60 {
            let name = alloc::format!("file{i:03}.txt");
            vol.write(&name, b"x").unwrap();
        }
        let listing = vol.readdir("/").unwrap();
        assert_eq!(listing.len(), 60);
        // Deep nesting still resolves.
        vol.mkdir("A").unwrap();
        vol.mkdir("A/B").unwrap();
        vol.mkdir("A/B/C").unwrap();
        vol.write("A/B/C/deep.txt", b"deep").unwrap();
        assert_eq!(vol.read("A/B/C/deep.txt").unwrap(), b"deep");
    }

    #[test_case]
    fn volume_label_is_readable_from_the_root() {
        let mut disk = small_disk();
        format(&mut disk, "MYUSB").expect("format");
        let mut vol = ExfatRw::open(&mut disk, false).expect("open");
        let root = vol.read_dir(vol.bpb.root_cluster, 0).unwrap();
        assert_eq!(root.bytes[0], TYPE_VOLUME);
        assert_eq!(root.bytes[1], 5);
        let units: Vec<u16> = (0..5)
            .map(|k| u16::from_le_bytes([root.bytes[2 + k * 2], root.bytes[3 + k * 2]]))
            .collect();
        assert_eq!(str_from_utf16(&units), "MYUSB");
    }

    #[test_case]
    fn a_foreign_no_fat_chain_file_reads_contiguously() {
        // Windows/Linux mark *contiguous* chains with the NoFatChain flag and
        // leave the FAT entries unused. Our own writer always uses the FAT, so
        // nothing it produces exercises the reader's contiguous path — build
        // such a file by hand: data in clusters 5.., stream flag bit 0 set,
        // FAT entries untouched.
        let mut disk = small_disk();
        format(&mut disk, "").expect("format");
        let mut vol = ExfatRw::open(&mut disk, true).expect("open");
        let data = b"hello no-fat chain";
        // The volume's free cluster 5 (2-4 are metadata): write the data there.
        let base = vol.bpb.cluster_sector(5).unwrap();
        let mut buf = vec![0u8; vol.bpb.cluster_bytes() as usize];
        buf[..data.len()].copy_from_slice(data);
        vol.dev.write_blocks(base as u64, &buf).unwrap();
        // Mark 5..7 used in the bitmap (a real owner would), leave the FAT free.
        vol.set_bit(5, true);
        vol.set_bit(6, true);
        vol.set_bit(7, true);
        let name = utf16_from_str("NOFAT.BIN");
        let mut set = build_entry_set(&name, false, 5, data.len() as u64, 0);
        set[1][1] = FLAGS_NO_FAT_CHAIN; // stream flags: contiguous, no FAT chain
        let mut root = vol.read_dir(vol.bpb.root_cluster, 0).unwrap();
        vol.place_entry_set(&mut root, 0, &set).unwrap();
        vol.flush_dir(&root).unwrap();
        vol.sync().unwrap();
        // Read follows contiguity, not the (untouched) FAT.
        assert_eq!(vol.read("NOFAT.BIN").unwrap(), data);
        // And unlink frees the contiguous run.
        vol.unlink("NOFAT.BIN").unwrap();
        assert!(matches!(vol.read("NOFAT.BIN"), Err(ExfatError::NotFound)));
    }

    #[test_case]
    fn a_foreign_volume_without_a_bitmap_is_refused() {
        let mut disk = small_disk();
        format(&mut disk, "").expect("format");
        // Wipe the root directory's bitmap entry (first entry with no label):
        // no allocation bitmap, no RW.
        let bpb = {
            let vol = ExfatRw::open(&mut disk, false).unwrap();
            vol.bpb
        };
        let base = bpb.cluster_sector(2).unwrap();
        let mut sec = [0u8; BLOCK_SIZE];
        disk.read_block(base as u64, &mut sec).unwrap();
        assert_eq!(sec[0], TYPE_BITMAP);
        sec[0] = 0x00; // bitmap entry type -> end marker
        disk.write_block(base as u64, &sec).unwrap();
        assert!(matches!(ExfatRw::open(&mut disk, false), Err(ExfatError::Unsupported)));
    }
}
