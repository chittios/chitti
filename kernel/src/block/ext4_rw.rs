//! Live **read-write ext4** over a volume formatted by [`super::ext4::Ext4Writer`].
//!
//! This is the documented follow-on to `ext4_store`'s rewrite-on-sync path:
//! allocate and free from live bitmaps, create/unlink inodes, and edit
//! directory entries in place — so a single agent-state write is O(file), not
//! O(partition).
//!
//! ## Scope
//! - 4 KiB blocks, 128-byte inodes, block-mapped files (12 direct + single +
//!   double indirect) — the same layout our mkfs produces
//! - Linear directories (no htree); hierarchical paths (`mkdir` + nested write)
//! - **Ordered metadata journal** (PR3): each public mutation is one transaction.
//!   Metadata blocks are staged, written to a reserved journal area, committed,
//!   then installed. Incomplete transactions are discarded on mount; committed
//!   ones are replayed. File *data* is written before the metadata commit
//!   (allocate-new + pointer-flip), so a crash never points an inode at torn
//!   contents.
//!
//! Pure helpers ([`dirent_needed`], dirent walk) are unit-tested off-hardware;
//! the volume path runs on [`super::ramdisk::RamDisk`]. Power-fail injection
//! uses [`Ext4Rw::set_fail_after_device_writes`].

use crate::block::{BlockDevice, BlockError, BLOCK_SIZE};
use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

const EBS: usize = 4096;
const SPB: u64 = (EBS / BLOCK_SIZE) as u64;
const INODE_SIZE: usize = 128;
const BPG: u64 = (EBS * 8) as u64; // 32768
const IPG: u64 = 4096;
const ROOT_INO: u64 = 2;
const PTRS_PER: usize = EBS / 4; // 1024

const S_IFDIR: u16 = 0x4000;
const S_IFREG: u16 = 0x8000;
const FT_REG: u8 = 1;
const FT_DIR: u8 = 2;

// ── journal (Chitti ordered-metadata, not jbd2) ──────────────────────────
// Layout at the end of the volume (`total_blocks - JOURNAL_BLOCKS` ..):
//   [0] header: magic, version, state, n_meta, seq, block_nums[]
//   [1 .. n_meta] staged metadata block bodies
//   [1 + n_meta] commit record (magic + seq)
// Protocol: bodies → header(OPEN) → commit → install finals → header(EMPTY).
// Recovery: matching commit ⇒ replay install; else discard.

/// "C4JL" — Chitti4 JournaL header.
const JOURNAL_MAGIC: u32 = 0x4c_4a_34_43;
/// "C4JC" — commit record.
const COMMIT_MAGIC: u32 = 0x43_4a_34_43;
const JOURNAL_VERSION: u32 = 1;
/// Reserved fs-blocks at the end of the volume for the journal area.
const JOURNAL_BLOCKS: u64 = 32;
const J_STATE_EMPTY: u32 = 0;
const J_STATE_OPEN: u32 = 1;

/// Errors from live ext4 mutations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ext4RwError {
    /// Superblock / feature set we do not write.
    NotSupported,
    Io(BlockError),
    /// Path component missing, or empty path.
    NotFound,
    /// Name already exists where a create was requested / rename target conflict.
    Exists,
    /// Parent is not a directory, or target is a directory when a file was expected.
    NotAFile,
    /// Target is not a directory.
    NotADir,
    /// Directory still has entries other than `.` / `..`.
    NotEmpty,
    /// Name too long for an ext dirent, or path empty.
    BadName,
    /// No free blocks or inodes.
    NoSpace,
    /// Cross-device rename (not supported).
    CrossDevice,
    /// Test-only: device write budget exhausted ([`Ext4Rw::set_fail_after_device_writes`]).
    InjectedFault,
}

impl From<BlockError> for Ext4RwError {
    fn from(e: BlockError) -> Self {
        Ext4RwError::Io(e)
    }
}

/// On-disk size of one directory entry: 8-byte header + name rounded up to 4.
pub fn dirent_needed(name_len: usize) -> usize {
    8 + name_len.div_ceil(4) * 4
}

fn le16(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}
fn le32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}
fn put16(b: &mut [u8], o: usize, v: u16) {
    b[o..o + 2].copy_from_slice(&v.to_le_bytes());
}
fn put32(b: &mut [u8], o: usize, v: u32) {
    b[o..o + 4].copy_from_slice(&v.to_le_bytes());
}
fn le64(b: &[u8], o: usize) -> u64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&b[o..o + 8]);
    u64::from_le_bytes(a)
}
fn put64(b: &mut [u8], o: usize, v: u64) {
    b[o..o + 8].copy_from_slice(&v.to_le_bytes());
}

/// A parsed inode for the RW path (block-mapped only).
#[derive(Clone, Debug)]
struct Inode {
    mode: u16,
    /// Owner: agent id, or `0` for system/orchestrator (not a security gate —
    /// Synapse path scope is).
    uid: u32,
    gid: u32,
    size: u64,
    links: u16,
    blocks_512: u64,
    /// Unix seconds (wall clock); `0` if never stamped.
    atime: u32,
    mtime: u32,
    ctime: u32,
    slots: [u32; 15],
}

impl Inode {
    fn is_dir(&self) -> bool {
        self.mode & S_IFDIR != 0
    }
    fn is_reg(&self) -> bool {
        self.mode & S_IFREG != 0
    }

    /// Stamp mtime/ctime (and atime if unset) from the wall clock.
    fn touch_now(&mut self) {
        let t = wall_secs();
        self.mtime = t;
        self.ctime = t;
        if self.atime == 0 {
            self.atime = t;
        }
    }

    /// Cleared inode record (free slot).
    fn zeroed() -> Self {
        Inode {
            mode: 0,
            uid: 0,
            gid: 0,
            size: 0,
            links: 0,
            blocks_512: 0,
            atime: 0,
            mtime: 0,
            ctime: 0,
            slots: [0; 15],
        }
    }
}

/// Public file metadata (stat(2)-shaped, agent-native meaning for `uid`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FileStat {
    pub ino: u64,
    pub mode: u16,
    pub uid: u32,
    pub gid: u32,
    pub size: u64,
    pub atime: u32,
    pub mtime: u32,
    pub ctime: u32,
    pub nlink: u16,
}

impl FileStat {
    pub fn is_dir(&self) -> bool {
        self.mode & S_IFDIR != 0
    }
    pub fn is_reg(&self) -> bool {
        self.mode & S_IFREG != 0
    }
}

/// Wall-clock seconds for inode stamps; `0` if the clock is still the boot default
/// and no RTC was read (acceptable — mtime is best-effort).
fn wall_secs() -> u32 {
    let t = crate::clock::now_unix();
    if t < 0 {
        0
    } else {
        t as u32
    }
}

/// One group descriptor (the fields we mutate).
#[derive(Clone, Copy, Debug)]
struct GroupDesc {
    block_bitmap: u64,
    inode_bitmap: u64,
    inode_table: u64,
    free_blocks: u16,
    free_inodes: u16,
    used_dirs: u16,
}

/// In-flight transaction: metadata blocks staged until commit.
struct Txn {
    /// Last write wins per fs-block.
    staged: BTreeMap<u64, [u8; EBS]>,
}

/// Live read-write handle on an ext4 volume.
pub struct Ext4Rw<'d, D: BlockDevice> {
    dev: &'d mut D,
    total_blocks: u64,
    total_inodes: u64,
    free_blocks: u64,
    free_inodes: u64,
    ngroups: u64,
    blocks_per_group: u64,
    inodes_per_group: u64,
    inode_size: usize,
    groups: Vec<GroupDesc>,
    /// First fs-block of the journal area (`0` = no journal; commit installs directly).
    journal_start: u64,
    /// Monotonic sequence for commit records (in-memory; on-disk seq is enough).
    journal_seq: u64,
    txn: Option<Txn>,
    /// After this many successful *device* writes, the next fails (tests).
    fail_after: Option<u32>,
    device_writes: u32,
}

impl<'d, D: BlockDevice> Ext4Rw<'d, D> {
    /// Open + validate a volume produced by our mkfs (or a compatible block-mapped
    /// ext2/3/4 with 4 KiB blocks and 128-byte inodes).
    pub fn open(dev: &'d mut D) -> Result<Self, Ext4RwError> {
        let mut s0 = [0u8; BLOCK_SIZE];
        let mut s1 = [0u8; BLOCK_SIZE];
        dev.read_block(2, &mut s0)?;
        dev.read_block(3, &mut s1)?;
        let mut sb = [0u8; 1024];
        sb[..512].copy_from_slice(&s0);
        sb[512..].copy_from_slice(&s1);
        if le16(&sb, 0x38) != 0xef53 {
            return Err(Ext4RwError::NotSupported);
        }
        let block_size = 1024usize << le32(&sb, 0x18);
        if block_size != EBS {
            return Err(Ext4RwError::NotSupported);
        }
        let inode_size = if le32(&sb, 0x4c) >= 1 {
            le16(&sb, 0x58) as usize
        } else {
            128
        };
        // 128 or 256-byte inodes both work for the fields we use (uid/mtime in the
        // base 128). Larger is accepted as long as it is a power of two ≥ 128.
        if inode_size < 128 || inode_size > 256 || EBS % inode_size != 0 {
            return Err(Ext4RwError::NotSupported);
        }
        // Refuse external journal device; we do not follow journal_dev.
        let incompat = le32(&sb, 0x60);
        if incompat & 0x0004 != 0 {
            return Err(Ext4RwError::NotSupported);
        }
        let total_blocks = le32(&sb, 4) as u64;
        let total_inodes = le32(&sb, 0) as u64;
        let free_blocks = le32(&sb, 12) as u64;
        let free_inodes = le32(&sb, 16) as u64;
        let bpg = le32(&sb, 32) as u64;
        let ipg = le32(&sb, 40) as u64;
        // Accept common geometries (including foreign mke2fs). Refuse zero.
        if bpg == 0 || ipg == 0 || bpg > 1_000_000 || ipg > 1_000_000 {
            return Err(Ext4RwError::NotSupported);
        }
        let ngroups = total_blocks.div_ceil(bpg);
        let desc_size = if incompat & 0x80 != 0 {
            le16(&sb, 0xfe) as usize
        } else {
            32
        };
        if desc_size < 32 {
            return Err(Ext4RwError::NotSupported);
        }
        // GDT starts at fs-block 1 (first_data_block is 0 for 4K).
        let mut groups = Vec::with_capacity(ngroups as usize);
        let mut gdt_buf = vec![0u8; EBS];
        for g in 0..ngroups {
            let byte = g * desc_size as u64;
            let gdt_blk = 1 + byte / EBS as u64;
            let off = (byte % EBS as u64) as usize;
            Self::read_eblock_static(dev, gdt_blk, &mut gdt_buf)?;
            let d = &gdt_buf[off..off + 32];
            groups.push(GroupDesc {
                block_bitmap: le32(d, 0) as u64,
                inode_bitmap: le32(d, 4) as u64,
                inode_table: le32(d, 8) as u64,
                free_blocks: le16(d, 12),
                free_inodes: le16(d, 14),
                used_dirs: le16(d, 16),
            });
        }
        let mut vol = Ext4Rw {
            dev,
            total_blocks,
            total_inodes,
            free_blocks,
            free_inodes,
            ngroups,
            blocks_per_group: bpg,
            inodes_per_group: ipg,
            inode_size,
            groups,
            journal_start: 0,
            journal_seq: 1,
            txn: None,
            fail_after: None,
            device_writes: 0,
        };
        // Journal only if already present or the tail is free — never steal
        // blocks that a foreign mke2fs volume already uses.
        vol.ensure_journal_safe()?;
        vol.recover_journal()?;
        Ok(vol)
    }

    /// After `n` successful device writes, the next write returns
    /// [`Ext4RwError::InjectedFault`]. `None` disables. Resets the counter.
    pub fn set_fail_after_device_writes(&mut self, after: Option<u32>) {
        self.fail_after = after;
        self.device_writes = 0;
    }

    /// How many device writes have succeeded since the last fail-budget reset.
    pub fn device_write_count(&self) -> u32 {
        self.device_writes
    }

    fn read_eblock_static(dev: &mut D, eblk: u64, buf: &mut [u8]) -> Result<(), Ext4RwError> {
        if buf.len() != EBS {
            return Err(Ext4RwError::Io(BlockError::BadBufferLen));
        }
        dev.read_blocks(eblk * SPB, buf)?;
        Ok(())
    }

    /// Disk read, ignoring the in-flight transaction stage.
    fn read_eblock_raw(&mut self, eblk: u64, buf: &mut [u8; EBS]) -> Result<(), Ext4RwError> {
        self.dev.read_blocks(eblk * SPB, buf)?;
        Ok(())
    }

    /// Effective read: staged metadata if present, else disk.
    fn read_eblock(&mut self, eblk: u64, buf: &mut [u8; EBS]) -> Result<(), Ext4RwError> {
        if let Some(txn) = self.txn.as_ref() {
            if let Some(staged) = txn.staged.get(&eblk) {
                *buf = *staged;
                return Ok(());
            }
        }
        self.read_eblock_raw(eblk, buf)
    }

    fn fault_gate(&mut self) -> Result<(), Ext4RwError> {
        if let Some(limit) = self.fail_after {
            if self.device_writes >= limit {
                return Err(Ext4RwError::InjectedFault);
            }
        }
        Ok(())
    }

    /// Device write with optional fault injection (counts successes).
    fn write_eblock_raw(&mut self, eblk: u64, buf: &[u8; EBS]) -> Result<(), Ext4RwError> {
        self.fault_gate()?;
        self.dev.write_blocks(eblk * SPB, buf)?;
        self.device_writes = self.device_writes.saturating_add(1);
        Ok(())
    }

    /// Metadata write: stage when a transaction is open, else direct to disk.
    fn write_eblock(&mut self, eblk: u64, buf: &[u8; EBS]) -> Result<(), Ext4RwError> {
        if let Some(txn) = self.txn.as_mut() {
            txn.staged.insert(eblk, *buf);
            return Ok(());
        }
        self.write_eblock_raw(eblk, buf)
    }

    /// File data / indirect pointer blocks: always durable before metadata commit.
    fn write_data_eblock(&mut self, eblk: u64, buf: &[u8; EBS]) -> Result<(), Ext4RwError> {
        self.write_eblock_raw(eblk, buf)
    }

    // ── journal ──────────────────────────────────────────────────────────

    fn ensure_journal_safe(&mut self) -> Result<(), Ext4RwError> {
        if self.journal_start != 0 {
            return Ok(());
        }
        if self.total_blocks <= JOURNAL_BLOCKS + 64 {
            return Err(Ext4RwError::NotSupported);
        }
        let start = self.total_blocks - JOURNAL_BLOCKS;
        let mut hdr = [0u8; EBS];
        self.read_eblock_raw(start, &mut hdr)?;
        if le32(&hdr, 0) == JOURNAL_MAGIC && le32(&hdr, 4) == JOURNAL_VERSION {
            self.journal_start = start;
            let seq = le64(&hdr, 16);
            if seq > 0 {
                self.journal_seq = seq.saturating_add(1);
            }
            return Ok(());
        }
        // Only reserve the tail if every block there is free — otherwise leave
        // journal_start = 0 and commit installs staged blocks directly (foreign
        // mke2fs volumes keep their layout intact).
        for b in start..start + JOURNAL_BLOCKS {
            let g = b / self.blocks_per_group;
            let bit = (b % self.blocks_per_group) as usize;
            let bb = self.groups[g as usize].block_bitmap;
            let mut bm = [0u8; EBS];
            self.read_eblock_raw(bb, &mut bm)?;
            if bm[bit / 8] & (1 << (bit % 8)) != 0 {
                crate::ktrace::log("ext4_rw", "journal tail busy; RW without metadata journal");
                return Ok(());
            }
        }
        for b in start..start + JOURNAL_BLOCKS {
            self.mark_block_used_direct(b)?;
        }
        self.journal_start = start;
        self.write_journal_empty()?;
        crate::ktrace::log_fmt(format_args!(
            "ext4_rw: journal reserved at fs-block {start} ({} blocks)",
            JOURNAL_BLOCKS
        ));
        Ok(())
    }

    /// Set a block-bitmap bit without going through `alloc_block` (journal bootstrap).
    fn mark_block_used_direct(&mut self, blk: u64) -> Result<(), Ext4RwError> {
        if blk == 0 || blk >= self.total_blocks {
            return Ok(());
        }
        let g = blk / self.blocks_per_group;
        let bit = (blk % self.blocks_per_group) as usize;
        let bb = self.groups[g as usize].block_bitmap;
        let mut bm = [0u8; EBS];
        self.read_eblock_raw(bb, &mut bm)?;
        if bm[bit / 8] & (1 << (bit % 8)) != 0 {
            return Ok(()); // already used
        }
        bm[bit / 8] |= 1 << (bit % 8);
        self.write_eblock_raw(bb, &bm)?;
        self.groups[g as usize].free_blocks = self.groups[g as usize].free_blocks.saturating_sub(1);
        self.free_blocks = self.free_blocks.saturating_sub(1);
        // Sync GDT + super free counts directly.
        let gd = self.groups[g as usize];
        let byte = g * 32;
        let gdt_blk = 1 + byte / EBS as u64;
        let off = (byte % EBS as u64) as usize;
        let mut gbuf = [0u8; EBS];
        self.read_eblock_raw(gdt_blk, &mut gbuf)?;
        put16(&mut gbuf, off + 12, gd.free_blocks);
        put16(&mut gbuf, off + 14, gd.free_inodes);
        put16(&mut gbuf, off + 16, gd.used_dirs);
        self.write_eblock_raw(gdt_blk, &gbuf)?;
        let mut sblk = [0u8; EBS];
        self.read_eblock_raw(0, &mut sblk)?;
        put32(&mut sblk, 1024 + 12, self.free_blocks as u32);
        put32(&mut sblk, 1024 + 16, self.free_inodes as u32);
        self.write_eblock_raw(0, &sblk)?;
        Ok(())
    }

    fn write_journal_empty(&mut self) -> Result<(), Ext4RwError> {
        let mut hdr = [0u8; EBS];
        put32(&mut hdr, 0, JOURNAL_MAGIC);
        put32(&mut hdr, 4, JOURNAL_VERSION);
        put32(&mut hdr, 8, J_STATE_EMPTY);
        put32(&mut hdr, 12, 0);
        put64(&mut hdr, 16, self.journal_seq);
        self.write_eblock_raw(self.journal_start, &hdr)
    }

    fn recover_journal(&mut self) -> Result<(), Ext4RwError> {
        if self.journal_start == 0 {
            return Ok(());
        }
        let mut hdr = [0u8; EBS];
        self.read_eblock_raw(self.journal_start, &mut hdr)?;
        if le32(&hdr, 0) != JOURNAL_MAGIC || le32(&hdr, 4) != JOURNAL_VERSION {
            return Ok(());
        }
        let state = le32(&hdr, 8);
        let n = le32(&hdr, 12) as usize;
        let seq = le64(&hdr, 16);
        if state != J_STATE_OPEN || n == 0 || n as u64 + 2 > JOURNAL_BLOCKS {
            if state != J_STATE_EMPTY {
                self.write_journal_empty()?;
            }
            return Ok(());
        }
        // Commit record sits immediately after the staged bodies.
        let cblk = self.journal_start + 1 + n as u64;
        let mut cmt = [0u8; EBS];
        self.read_eblock_raw(cblk, &mut cmt)?;
        if le32(&cmt, 0) == COMMIT_MAGIC && le64(&cmt, 8) == seq {
            // Committed: install every staged body to its final home.
            for i in 0..n {
                let target = le32(&hdr, 32 + i * 4) as u64;
                if target == 0 || target >= self.total_blocks {
                    continue;
                }
                let mut body = [0u8; EBS];
                self.read_eblock_raw(self.journal_start + 1 + i as u64, &mut body)?;
                self.write_eblock_raw(target, &body)?;
            }
            crate::ktrace::log_fmt(format_args!(
                "ext4_rw: journal recovered ({n} metadata block(s), seq {seq})"
            ));
            self.reload_free_counts_from_disk()?;
        } else {
            crate::ktrace::log_fmt(format_args!(
                "ext4_rw: discarded incomplete journal txn (seq {seq}, n={n})"
            ));
        }
        self.journal_seq = seq.saturating_add(1);
        self.write_journal_empty()?;
        Ok(())
    }

    fn reload_free_counts_from_disk(&mut self) -> Result<(), Ext4RwError> {
        let mut s0 = [0u8; BLOCK_SIZE];
        let mut s1 = [0u8; BLOCK_SIZE];
        self.dev.read_block(2, &mut s0)?;
        self.dev.read_block(3, &mut s1)?;
        let mut sb = [0u8; 1024];
        sb[..512].copy_from_slice(&s0);
        sb[512..].copy_from_slice(&s1);
        self.free_blocks = le32(&sb, 12) as u64;
        self.free_inodes = le32(&sb, 16) as u64;
        for g in 0..self.ngroups {
            let byte = g * 32;
            let gdt_blk = 1 + byte / EBS as u64;
            let off = (byte % EBS as u64) as usize;
            let mut gbuf = [0u8; EBS];
            self.read_eblock_raw(gdt_blk, &mut gbuf)?;
            self.groups[g as usize].free_blocks = le16(&gbuf, off + 12);
            self.groups[g as usize].free_inodes = le16(&gbuf, off + 14);
            self.groups[g as usize].used_dirs = le16(&gbuf, off + 16);
        }
        Ok(())
    }

    fn begin_txn(&mut self) {
        debug_assert!(self.txn.is_none(), "nested transaction");
        self.txn = Some(Txn {
            staged: BTreeMap::new(),
        });
    }

    fn abort_txn(&mut self) {
        self.txn = None;
        // In-memory free counts / bitmaps may have diverged; reload from disk.
        let _ = self.reload_free_counts_from_disk();
    }

    fn commit_txn(&mut self) -> Result<(), Ext4RwError> {
        let Some(txn) = self.txn.take() else {
            return Ok(());
        };
        if txn.staged.is_empty() {
            return Ok(());
        }
        let mut pairs: Vec<(u64, [u8; EBS])> = txn.staged.into_iter().collect();
        pairs.sort_by_key(|(b, _)| *b);
        let n = pairs.len();
        if self.journal_start == 0 || n as u64 + 2 > JOURNAL_BLOCKS {
            // No journal (foreign volume) or oversized: install staged blocks directly.
            for (blk, buf) in &pairs {
                self.write_eblock_raw(*blk, buf)?;
            }
            return Ok(());
        }
        let seq = self.journal_seq;
        self.journal_seq = self.journal_seq.saturating_add(1);

        // 1) Bodies into the journal area.
        for (i, (_, buf)) in pairs.iter().enumerate() {
            self.write_eblock_raw(self.journal_start + 1 + i as u64, buf)?;
        }
        // 2) Header describing the set (OPEN).
        let mut hdr = [0u8; EBS];
        put32(&mut hdr, 0, JOURNAL_MAGIC);
        put32(&mut hdr, 4, JOURNAL_VERSION);
        put32(&mut hdr, 8, J_STATE_OPEN);
        put32(&mut hdr, 12, n as u32);
        put64(&mut hdr, 16, seq);
        for (i, (blk, _)) in pairs.iter().enumerate() {
            put32(&mut hdr, 32 + i * 4, *blk as u32);
        }
        self.write_eblock_raw(self.journal_start, &hdr)?;
        // 3) Commit record — after this point, recovery must install.
        let mut cmt = [0u8; EBS];
        put32(&mut cmt, 0, COMMIT_MAGIC);
        put64(&mut cmt, 8, seq);
        self.write_eblock_raw(self.journal_start + 1 + n as u64, &cmt)?;
        // 4) Install to final homes.
        for (blk, buf) in &pairs {
            self.write_eblock_raw(*blk, buf)?;
        }
        // 5) Clear journal.
        self.write_journal_empty()?;
        Ok(())
    }

    /// Run `f` inside a journal transaction; commit on success, abort on error.
    fn with_txn<R>(
        &mut self,
        f: impl FnOnce(&mut Self) -> Result<R, Ext4RwError>,
    ) -> Result<R, Ext4RwError> {
        self.begin_txn();
        match f(self) {
            Ok(r) => match self.commit_txn() {
                Ok(()) => Ok(r),
                Err(e) => {
                    // Commit failed mid-way: recovery on next open will finish or
                    // discard. Drop staged state; counts may be wrong until reload.
                    self.txn = None;
                    let _ = self.reload_free_counts_from_disk();
                    Err(e)
                }
            },
            Err(e) => {
                self.abort_txn();
                Err(e)
            }
        }
    }

    fn blocks_in_group(&self, g: u64) -> u64 {
        (g * self.blocks_per_group + self.blocks_per_group).min(self.total_blocks) - g * self.blocks_per_group
    }

    // ── bitmaps ──────────────────────────────────────────────────────────

    fn alloc_block(&mut self) -> Result<u64, Ext4RwError> {
        if self.free_blocks == 0 {
            return Err(Ext4RwError::NoSpace);
        }
        for g in 0..self.ngroups {
            if self.groups[g as usize].free_blocks == 0 {
                continue;
            }
            let bb = self.groups[g as usize].block_bitmap;
            let mut bm = [0u8; EBS];
            self.read_eblock(bb, &mut bm)?;
            let nb = self.blocks_in_group(g) as usize;
            for bit in 0..nb {
                if bm[bit / 8] & (1 << (bit % 8)) == 0 {
                    bm[bit / 8] |= 1 << (bit % 8);
                    self.write_eblock(bb, &bm)?;
                    self.groups[g as usize].free_blocks -= 1;
                    self.free_blocks -= 1;
                    self.sync_group_desc(g)?;
                    self.sync_super_free()?;
                    return Ok(g * self.blocks_per_group + bit as u64);
                }
            }
        }
        Err(Ext4RwError::NoSpace)
    }

    fn free_block(&mut self, blk: u64) -> Result<(), Ext4RwError> {
        if blk == 0 || blk >= self.total_blocks {
            return Ok(());
        }
        let g = blk / self.blocks_per_group;
        let bit = (blk % self.blocks_per_group) as usize;
        let bb = self.groups[g as usize].block_bitmap;
        let mut bm = [0u8; EBS];
        self.read_eblock(bb, &mut bm)?;
        if bm[bit / 8] & (1 << (bit % 8)) == 0 {
            return Ok(()); // already free
        }
        bm[bit / 8] &= !(1 << (bit % 8));
        self.write_eblock(bb, &bm)?;
        self.groups[g as usize].free_blocks = self.groups[g as usize].free_blocks.saturating_add(1);
        self.free_blocks += 1;
        self.sync_group_desc(g)?;
        self.sync_super_free()?;
        Ok(())
    }

    fn alloc_inode(&mut self) -> Result<u64, Ext4RwError> {
        if self.free_inodes == 0 {
            return Err(Ext4RwError::NoSpace);
        }
        for g in 0..self.ngroups {
            if self.groups[g as usize].free_inodes == 0 {
                continue;
            }
            let ib = self.groups[g as usize].inode_bitmap;
            let mut bm = [0u8; EBS];
            self.read_eblock(ib, &mut bm)?;
            for bit in 0..self.inodes_per_group as usize {
                if bm[bit / 8] & (1 << (bit % 8)) == 0 {
                    bm[bit / 8] |= 1 << (bit % 8);
                    self.write_eblock(ib, &bm)?;
                    self.groups[g as usize].free_inodes -= 1;
                    self.free_inodes -= 1;
                    self.sync_group_desc(g)?;
                    self.sync_super_free()?;
                    return Ok(g * self.inodes_per_group + bit as u64 + 1);
                }
            }
        }
        Err(Ext4RwError::NoSpace)
    }

    fn free_inode_num(&mut self, ino: u64) -> Result<(), Ext4RwError> {
        if ino == 0 || ino > self.total_inodes {
            return Ok(());
        }
        let g = (ino - 1) / self.inodes_per_group;
        let bit = ((ino - 1) % self.inodes_per_group) as usize;
        let ib = self.groups[g as usize].inode_bitmap;
        let mut bm = [0u8; EBS];
        self.read_eblock(ib, &mut bm)?;
        if bm[bit / 8] & (1 << (bit % 8)) == 0 {
            return Ok(());
        }
        bm[bit / 8] &= !(1 << (bit % 8));
        self.write_eblock(ib, &bm)?;
        self.groups[g as usize].free_inodes = self.groups[g as usize].free_inodes.saturating_add(1);
        self.free_inodes += 1;
        self.sync_group_desc(g)?;
        self.sync_super_free()?;
        Ok(())
    }

    fn sync_group_desc(&mut self, g: u64) -> Result<(), Ext4RwError> {
        let gd = self.groups[g as usize];
        // GDT at fs-block 1, 32-byte descriptors (our volumes are not 64-bit).
        let byte = g * 32;
        let gdt_blk = 1 + byte / EBS as u64;
        let off = (byte % EBS as u64) as usize;
        let mut buf = [0u8; EBS];
        self.read_eblock(gdt_blk, &mut buf)?;
        put32(&mut buf, off, gd.block_bitmap as u32);
        put32(&mut buf, off + 4, gd.inode_bitmap as u32);
        put32(&mut buf, off + 8, gd.inode_table as u32);
        put16(&mut buf, off + 12, gd.free_blocks);
        put16(&mut buf, off + 14, gd.free_inodes);
        put16(&mut buf, off + 16, gd.used_dirs);
        self.write_eblock(gdt_blk, &buf)?;
        // Sparse-super backups of the GDT: group 1 (and powers of 3/5/7) hold a
        // copy after their superblock. For correctness under our own remount we
        // only need group 0's GDT; Linux e2fsck may notice backup drift — Phase 2
        // journal work can refresh backups. Leave as-is for PR1.
        Ok(())
    }

    fn sync_super_free(&mut self) -> Result<(), Ext4RwError> {
        // Superblock at byte 1024 of fs-block 0.
        let mut blk = [0u8; EBS];
        self.read_eblock(0, &mut blk)?;
        put32(&mut blk, 1024 + 12, self.free_blocks as u32);
        put32(&mut blk, 1024 + 16, self.free_inodes as u32);
        self.write_eblock(0, &blk)?;
        Ok(())
    }

    // ── inodes ───────────────────────────────────────────────────────────

    fn read_inode(&mut self, ino: u64) -> Result<Inode, Ext4RwError> {
        if ino == 0 || ino > self.total_inodes {
            return Err(Ext4RwError::NotFound);
        }
        let g = (ino - 1) / self.inodes_per_group;
        let index = ((ino - 1) % self.inodes_per_group) as usize;
        let itable = self.groups[g as usize].inode_table;
        let inodes_per_block = EBS / self.inode_size;
        let blk = itable + (index / inodes_per_block) as u64;
        let off = (index % inodes_per_block) * self.inode_size;
        let mut buf = [0u8; EBS];
        self.read_eblock(blk, &mut buf)?;
        let b = &buf[off..off + self.inode_size];
        let mode = le16(b, 0);
        let uid = le16(b, 2) as u32 | ((le16(b, 116) as u32) << 16);
        let size = le32(b, 4) as u64 | ((le32(b, 108) as u64) << 32);
        let atime = le32(b, 8);
        let ctime = le32(b, 12);
        let mtime = le32(b, 16);
        let gid = le16(b, 24) as u32 | ((le16(b, 118) as u32) << 16);
        let links = le16(b, 26);
        let blocks_512 = le32(b, 28) as u64;
        let mut slots = [0u32; 15];
        for k in 0..15 {
            slots[k] = le32(b, 40 + k * 4);
        }
        Ok(Inode {
            mode,
            uid,
            gid,
            size,
            links,
            blocks_512,
            atime,
            mtime,
            ctime,
            slots,
        })
    }

    fn write_inode(&mut self, ino: u64, inode: &Inode) -> Result<(), Ext4RwError> {
        let g = (ino - 1) / self.inodes_per_group;
        let index = ((ino - 1) % self.inodes_per_group) as usize;
        let itable = self.groups[g as usize].inode_table;
        let inodes_per_block = EBS / self.inode_size;
        let blk = itable + (index / inodes_per_block) as u64;
        let off = (index % inodes_per_block) * self.inode_size;
        let mut buf = [0u8; EBS];
        self.read_eblock(blk, &mut buf)?;
        let b = &mut buf[off..off + self.inode_size];
        // Clear then encode — avoids stale high-size / flag bits from a prior use.
        for x in b.iter_mut() {
            *x = 0;
        }
        put16(b, 0, inode.mode);
        put16(b, 2, (inode.uid & 0xffff) as u16);
        put32(b, 4, (inode.size & 0xffff_ffff) as u32);
        put32(b, 8, inode.atime);
        put32(b, 12, inode.ctime);
        put32(b, 16, inode.mtime);
        put16(b, 24, (inode.gid & 0xffff) as u16);
        put16(b, 26, inode.links);
        put32(b, 28, inode.blocks_512 as u32);
        for k in 0..15 {
            put32(b, 40 + k * 4, inode.slots[k]);
        }
        if inode.mode & S_IFDIR == 0 && inode.size >= (1 << 31) {
            put32(b, 108, (inode.size >> 32) as u32);
        }
        // i_uid_high / i_gid_high at 116/118 (rev-1 inode).
        put16(b, 116, (inode.uid >> 16) as u16);
        put16(b, 118, (inode.gid >> 16) as u16);
        self.write_eblock(blk, &buf)?;
        Ok(())
    }

    // ── block mapping ────────────────────────────────────────────────────

    fn map_block(&mut self, inode: &Inode, lblk: u64) -> Result<Option<u64>, Ext4RwError> {
        let ib = |i: usize| inode.slots[i] as u64;
        if lblk < 12 {
            let p = ib(lblk as usize);
            return Ok(if p == 0 { None } else { Some(p) });
        }
        let mut buf = [0u8; EBS];
        let l = lblk - 12;
        if l < PTRS_PER as u64 {
            let si = ib(12);
            if si == 0 {
                return Ok(None);
            }
            self.read_eblock(si, &mut buf)?;
            let p = le32(&buf, (l as usize) * 4) as u64;
            return Ok(if p == 0 { None } else { Some(p) });
        }
        let l2 = l - PTRS_PER as u64;
        if l2 < (PTRS_PER as u64) * (PTRS_PER as u64) {
            let di = ib(13);
            if di == 0 {
                return Ok(None);
            }
            self.read_eblock(di, &mut buf)?;
            let si = le32(&buf, ((l2 / PTRS_PER as u64) as usize) * 4) as u64;
            if si == 0 {
                return Ok(None);
            }
            self.read_eblock(si, &mut buf)?;
            let p = le32(&buf, ((l2 % PTRS_PER as u64) as usize) * 4) as u64;
            return Ok(if p == 0 { None } else { Some(p) });
        }
        Err(Ext4RwError::NotSupported) // triple-indirect
    }

    /// Free every data block of `inode` and its single/double-indirect pointer
    /// blocks. Leaves slots uncleared — the caller rewrites the inode.
    ///
    /// Order matters: data is freed through the live map first, then the
    /// pointer blocks themselves. `slots[12]` is the first single-indirect and
    /// is **not** listed again under the double-indirect (`slots[13]`), matching
    /// [`Self::build_iblocks`].
    fn free_inode_blocks(&mut self, inode: &Inode) -> Result<(), Ext4RwError> {
        let nblocks = (inode.size as usize).div_ceil(EBS);
        for lb in 0..nblocks as u64 {
            if let Some(pb) = self.map_block(inode, lb)? {
                self.free_block(pb)?;
            }
        }
        let si = inode.slots[12] as u64;
        if si != 0 {
            self.free_block(si)?;
        }
        let di = inode.slots[13] as u64;
        if di != 0 {
            let mut buf = [0u8; EBS];
            self.read_eblock(di, &mut buf)?;
            for k in 0..PTRS_PER {
                let s = le32(&buf, k * 4) as u64;
                if s != 0 {
                    self.free_block(s)?;
                }
            }
            self.free_block(di)?;
        }
        Ok(())
    }

    /// Build i_block slots for `data_blocks`, allocating indirects as needed.
    /// Returns (slots, total 512-byte sectors used including indirects).
    fn build_iblocks(&mut self, data_blocks: &[u64]) -> Result<([u32; 15], u64), Ext4RwError> {
        let mut slots = [0u32; 15];
        let n = data_blocks.len();
        let mut meta = 0u64;
        for k in 0..n.min(12) {
            slots[k] = data_blocks[k] as u32;
        }
        let mut idx = 12usize;
        if n > idx {
            let si = self.alloc_block()?;
            meta += 1;
            let arr = &data_blocks[12..(12 + PTRS_PER).min(n)];
            self.write_ptr_block(si, arr)?;
            slots[12] = si as u32;
            idx = 12 + arr.len();
        }
        if n > idx {
            let di = self.alloc_block()?;
            meta += 1;
            let mut singles: Vec<u64> = Vec::new();
            let rem = &data_blocks[idx..];
            let mut c = 0;
            while c < rem.len() && singles.len() < PTRS_PER {
                let chunk = &rem[c..(c + PTRS_PER).min(rem.len())];
                let si = self.alloc_block()?;
                meta += 1;
                self.write_ptr_block(si, chunk)?;
                singles.push(si);
                c += chunk.len();
            }
            if c < rem.len() {
                return Err(Ext4RwError::NoSpace); // beyond double-indirect
            }
            self.write_ptr_block(di, &singles)?;
            slots[13] = di as u32;
        }
        let sectors = (data_blocks.len() as u64 + meta) * SPB;
        Ok((slots, sectors))
    }

    fn write_ptr_block(&mut self, b: u64, ptrs: &[u64]) -> Result<(), Ext4RwError> {
        let mut buf = [0u8; EBS];
        for (k, &p) in ptrs.iter().enumerate() {
            put32(&mut buf, k * 4, p as u32);
        }
        // Pointer blocks for file data must hit the disk before the inode that
        // names them is committed (ordered mode).
        self.write_data_eblock(b, &buf)
    }

    fn stream_data(&mut self, data: &[u8], blocks: &[u64]) -> Result<(), Ext4RwError> {
        let mut bi = 0usize;
        while bi < blocks.len() {
            let mut run = 1usize;
            while bi + run < blocks.len() && blocks[bi + run] == blocks[bi] + run as u64 {
                run += 1;
            }
            // Write one fs-block at a time through the fault-gated data path so
            // power-fail injection is uniform (batched multi-block writes would
            // skip the counter).
            for r in 0..run {
                let mut ebuf = [0u8; EBS];
                let start = (bi + r) * EBS;
                let end = (start + EBS).min(data.len());
                if end > start {
                    ebuf[..end - start].copy_from_slice(&data[start..end]);
                }
                self.write_data_eblock(blocks[bi + r], &ebuf)?;
            }
            bi += run;
        }
        Ok(())
    }

    /// Replace the entire contents of inode `ino` (must be a regular file).
    ///
    /// **Allocate-new + pointer-flip:** new data blocks are written first, then
    /// the inode is updated (journaled metadata), then the old blocks are freed.
    /// A crash never leaves the inode pointing at a half-written new body.
    fn set_file_contents(&mut self, ino: u64, data: &[u8]) -> Result<(), Ext4RwError> {
        let old = self.read_inode(ino)?;
        if !old.is_reg() {
            return Err(Ext4RwError::NotAFile);
        }
        let nblocks = data.len().div_ceil(EBS);
        let mut data_blocks = Vec::with_capacity(nblocks);
        for _ in 0..nblocks {
            data_blocks.push(self.alloc_block()?);
        }
        self.stream_data(data, &data_blocks)?;
        let (slots, sectors) = self.build_iblocks(&data_blocks)?;
        let mut new_inode = old.clone();
        new_inode.size = data.len() as u64;
        new_inode.blocks_512 = sectors;
        new_inode.slots = slots;
        new_inode.touch_now();
        self.write_inode(ino, &new_inode)?;
        // Free the previous mapping only after the new inode is staged/committed.
        self.free_inode_blocks(&old)?;
        Ok(())
    }

    // ── directories ──────────────────────────────────────────────────────

    /// Walk a directory inode's data blocks and return `(name, ino, file_type)`.
    fn list_dir_ino(&mut self, dir_ino: u64) -> Result<Vec<(String, u64, u8)>, Ext4RwError> {
        let inode = self.read_inode(dir_ino)?;
        if !inode.is_dir() {
            return Err(Ext4RwError::NotADir);
        }
        let mut out = Vec::new();
        let nblocks = (inode.size as usize).div_ceil(EBS);
        let mut buf = [0u8; EBS];
        for lb in 0..nblocks as u64 {
            let Some(pb) = self.map_block(&inode, lb)? else {
                break;
            };
            self.read_eblock(pb, &mut buf)?;
            let mut off = 0usize;
            while off + 8 <= EBS {
                let ino = le32(&buf, off) as u64;
                let rec_len = le16(&buf, off + 4) as usize;
                let name_len = buf[off + 6] as usize;
                let ftype = buf[off + 7];
                if rec_len < 8 {
                    break;
                }
                if ino != 0 && name_len > 0 && off + 8 + name_len <= EBS {
                    let name = String::from_utf8_lossy(&buf[off + 8..off + 8 + name_len]).into_owned();
                    out.push((name, ino, ftype));
                }
                off += rec_len;
            }
        }
        Ok(out)
    }

    fn lookup_in_dir(&mut self, dir_ino: u64, name: &str) -> Result<Option<(u64, u8)>, Ext4RwError> {
        for (n, ino, ft) in self.list_dir_ino(dir_ino)? {
            if n == name {
                return Ok(Some((ino, ft)));
            }
        }
        Ok(None)
    }

    /// Insert `name → (ino, ftype)` into directory `dir_ino`.
    fn dir_add(&mut self, dir_ino: u64, name: &str, child_ino: u64, ftype: u8) -> Result<(), Ext4RwError> {
        if name.is_empty() || name.len() > 255 || name.contains('/') {
            return Err(Ext4RwError::BadName);
        }
        if self.lookup_in_dir(dir_ino, name)?.is_some() {
            return Err(Ext4RwError::Exists);
        }
        let need = dirent_needed(name.len());
        let mut inode = self.read_inode(dir_ino)?;
        if !inode.is_dir() {
            return Err(Ext4RwError::NotADir);
        }
        let nblocks = (inode.size as usize).div_ceil(EBS).max(1);
        let mut buf = [0u8; EBS];
        // Try to fit into an existing block.
        for lb in 0..nblocks as u64 {
            let Some(pb) = self.map_block(&inode, lb)? else {
                continue;
            };
            self.read_eblock(pb, &mut buf)?;
            if try_insert_dirent(&mut buf, name.as_bytes(), child_ino, ftype, need) {
                self.write_eblock(pb, &buf)?;
                return Ok(());
            }
        }
        // Need a new directory block.
        let new_blk = self.alloc_block()?;
        let mut newbuf = [0u8; EBS];
        // Single entry spanning the whole block.
        put32(&mut newbuf, 0, child_ino as u32);
        put16(&mut newbuf, 4, EBS as u16);
        newbuf[6] = name.len() as u8;
        newbuf[7] = ftype;
        newbuf[8..8 + name.len()].copy_from_slice(name.as_bytes());
        self.write_eblock(new_blk, &newbuf)?;

        // Rebuild directory block list: old data blocks + new.
        let mut data_blocks = Vec::new();
        let old_n = (inode.size as usize).div_ceil(EBS);
        for lb in 0..old_n as u64 {
            if let Some(pb) = self.map_block(&inode, lb)? {
                data_blocks.push(pb);
            }
        }
        data_blocks.push(new_blk);
        // Free old indirects only (data blocks are kept).
        let si = inode.slots[12] as u64;
        let di = inode.slots[13] as u64;
        // Clear slots so free of indirects doesn't touch data; free indirects.
        if si != 0 {
            self.free_block(si)?;
        }
        if di != 0 {
            let mut pbuf = [0u8; EBS];
            self.read_eblock(di, &mut pbuf)?;
            for k in 0..PTRS_PER {
                let s = le32(&pbuf, k * 4) as u64;
                if s != 0 {
                    self.free_block(s)?;
                }
            }
            self.free_block(di)?;
        }
        let (slots, sectors) = self.build_iblocks(&data_blocks)?;
        inode.slots = slots;
        inode.size = (data_blocks.len() * EBS) as u64;
        inode.blocks_512 = sectors;
        self.write_inode(dir_ino, &inode)?;
        Ok(())
    }

    /// Remove the dirent named `name` from `dir_ino`. Does not free the child inode.
    fn dir_remove(&mut self, dir_ino: u64, name: &str) -> Result<(), Ext4RwError> {
        let inode = self.read_inode(dir_ino)?;
        if !inode.is_dir() {
            return Err(Ext4RwError::NotADir);
        }
        let nblocks = (inode.size as usize).div_ceil(EBS);
        let mut buf = [0u8; EBS];
        for lb in 0..nblocks as u64 {
            let Some(pb) = self.map_block(&inode, lb)? else {
                continue;
            };
            self.read_eblock(pb, &mut buf)?;
            if try_remove_dirent(&mut buf, name.as_bytes()) {
                self.write_eblock(pb, &buf)?;
                return Ok(());
            }
        }
        Err(Ext4RwError::NotFound)
    }

    // ── path resolution ──────────────────────────────────────────────────

    fn normalize_parts(path: &str) -> Result<Vec<&str>, Ext4RwError> {
        let p = path.trim().trim_start_matches('/');
        if p.is_empty() {
            return Ok(Vec::new());
        }
        let mut parts = Vec::new();
        for seg in p.split('/') {
            if seg.is_empty() || seg == "." {
                continue;
            }
            if seg == ".." {
                parts.pop();
                continue;
            }
            if seg.len() > 255 {
                return Err(Ext4RwError::BadName);
            }
            parts.push(seg);
        }
        Ok(parts)
    }

    /// Resolve `path` to an inode, or `NotFound`.
    pub fn lookup(&mut self, path: &str) -> Result<u64, Ext4RwError> {
        let parts = Self::normalize_parts(path)?;
        if parts.is_empty() {
            return Ok(ROOT_INO);
        }
        let mut ino = ROOT_INO;
        for (i, seg) in parts.iter().enumerate() {
            let Some((child, ft)) = self.lookup_in_dir(ino, seg)? else {
                return Err(Ext4RwError::NotFound);
            };
            if i + 1 < parts.len() && ft != FT_DIR {
                return Err(Ext4RwError::NotADir);
            }
            ino = child;
        }
        Ok(ino)
    }

    /// Ensure every directory component of `path` exists; return (parent_ino, basename).
    fn ensure_parent(&mut self, path: &str) -> Result<(u64, String), Ext4RwError> {
        let parts = Self::normalize_parts(path)?;
        if parts.is_empty() {
            return Err(Ext4RwError::BadName);
        }
        let basename = parts[parts.len() - 1].to_string();
        let mut ino = ROOT_INO;
        for seg in &parts[..parts.len() - 1] {
            match self.lookup_in_dir(ino, seg)? {
                Some((child, ft)) if ft == FT_DIR => ino = child,
                Some(_) => return Err(Ext4RwError::NotADir),
                None => {
                    // mkdir -p
                    let child = self.create_dir_inode(ino)?;
                    self.dir_add(ino, seg, child, FT_DIR)?;
                    ino = child;
                }
            }
        }
        Ok((ino, basename))
    }

    fn create_dir_inode(&mut self, parent_ino: u64) -> Result<u64, Ext4RwError> {
        let ino = self.alloc_inode()?;
        let blk = self.alloc_block()?;
        // `.` and `..`
        let entries = [
            (ino, b".".as_slice(), FT_DIR),
            (parent_ino, b"..".as_slice(), FT_DIR),
        ];
        let mut dirbuf = [0u8; EBS];
        let mut off = 0usize;
        for (j, (i, name, ft)) in entries.iter().enumerate() {
            let reclen = if j + 1 == entries.len() {
                EBS - off
            } else {
                dirent_needed(name.len())
            };
            put32(&mut dirbuf, off, *i as u32);
            put16(&mut dirbuf, off + 4, reclen as u16);
            dirbuf[off + 6] = name.len() as u8;
            dirbuf[off + 7] = *ft;
            dirbuf[off + 8..off + 8 + name.len()].copy_from_slice(name);
            off += reclen;
        }
        self.write_eblock(blk, &dirbuf)?;
        let (slots, sectors) = self.build_iblocks(&[blk])?;
        let t = wall_secs();
        let inode = Inode {
            mode: S_IFDIR | 0o755,
            uid: 0,
            gid: 0,
            size: EBS as u64,
            links: 2,
            blocks_512: sectors,
            atime: t,
            mtime: t,
            ctime: t,
            slots,
        };
        self.write_inode(ino, &inode)?;
        // parent link count + used_dirs
        let mut parent = self.read_inode(parent_ino)?;
        parent.links = parent.links.saturating_add(1);
        parent.ctime = t;
        self.write_inode(parent_ino, &parent)?;
        let g = (ino - 1) / self.inodes_per_group;
        self.groups[g as usize].used_dirs = self.groups[g as usize].used_dirs.saturating_add(1);
        self.sync_group_desc(g)?;
        Ok(ino)
    }

    // ── public API ───────────────────────────────────────────────────────

    /// Create a directory at `path` (parents created as needed). Idempotent if
    /// the path already names a directory.
    pub fn mkdir(&mut self, path: &str) -> Result<(), Ext4RwError> {
        match self.lookup(path) {
            Ok(ino) => {
                let i = self.read_inode(ino)?;
                if i.is_dir() {
                    return Ok(());
                }
                return Err(Ext4RwError::Exists);
            }
            Err(Ext4RwError::NotFound) => {}
            Err(e) => return Err(e),
        }
        let path = path.to_string();
        self.with_txn(move |vol| {
            let (parent, name) = vol.ensure_parent(&path)?;
            if vol.lookup_in_dir(parent, &name)?.is_some() {
                return Ok(());
            }
            let child = vol.create_dir_inode(parent)?;
            vol.dir_add(parent, &name, child, FT_DIR)?;
            Ok(())
        })
    }

    /// Create or replace a regular file at `path` with `data`. Intermediate
    /// directories are created (`mkdir -p` semantics). One journal transaction.
    pub fn write(&mut self, path: &str, data: &[u8]) -> Result<(), Ext4RwError> {
        let path = path.to_string();
        let data = data.to_vec();
        self.with_txn(move |vol| {
            let (parent, name) = vol.ensure_parent(&path)?;
            match vol.lookup_in_dir(parent, &name)? {
                Some((_ino, ft)) if ft == FT_DIR => Err(Ext4RwError::NotAFile),
                Some((ino, _)) => vol.set_file_contents(ino, &data),
                None => {
                    let ino = vol.alloc_inode()?;
                    let t = wall_secs();
                    let inode = Inode {
                        mode: S_IFREG | 0o644,
                        uid: 0,
                        gid: 0,
                        size: 0,
                        links: 1,
                        blocks_512: 0,
                        atime: t,
                        mtime: t,
                        ctime: t,
                        slots: [0; 15],
                    };
                    vol.write_inode(ino, &inode)?;
                    vol.dir_add(parent, &name, ino, FT_REG)?;
                    vol.set_file_contents(ino, &data)
                }
            }
        })
    }

    /// Read the whole file at `path` into a new buffer.
    pub fn read(&mut self, path: &str) -> Result<Vec<u8>, Ext4RwError> {
        let ino = self.lookup(path)?;
        let inode = self.read_inode(ino)?;
        if !inode.is_reg() {
            return Err(Ext4RwError::NotAFile);
        }
        let mut out = vec![0u8; inode.size as usize];
        let mut done = 0usize;
        let mut lb = 0u64;
        let mut buf = [0u8; EBS];
        while done < out.len() {
            let Some(pb) = self.map_block(&inode, lb)? else {
                break;
            };
            self.read_eblock(pb, &mut buf)?;
            let take = (out.len() - done).min(EBS);
            out[done..done + take].copy_from_slice(&buf[..take]);
            done += take;
            lb += 1;
        }
        out.truncate(done);
        Ok(out)
    }

    /// Whether a path exists (file or directory).
    pub fn exists(&mut self, path: &str) -> bool {
        self.lookup(path).is_ok()
    }

    /// Unlink a regular file. Directories use [`Self::rmdir`].
    pub fn unlink(&mut self, path: &str) -> Result<(), Ext4RwError> {
        let path = path.to_string();
        self.with_txn(move |vol| {
            let parts = Self::normalize_parts(&path)?;
            if parts.is_empty() {
                return Err(Ext4RwError::BadName);
            }
            let name = parts[parts.len() - 1];
            let parent_ino = if parts.len() == 1 {
                ROOT_INO
            } else {
                let parent = parts[..parts.len() - 1].join("/");
                vol.lookup(&parent)?
            };
            let Some((ino, ft)) = vol.lookup_in_dir(parent_ino, name)? else {
                return Err(Ext4RwError::NotFound);
            };
            if ft == FT_DIR {
                return Err(Ext4RwError::NotAFile);
            }
            let inode = vol.read_inode(ino)?;
            vol.free_inode_blocks(&inode)?;
            vol.write_inode(ino, &Inode::zeroed())?;
            vol.free_inode_num(ino)?;
            vol.dir_remove(parent_ino, name)?;
            Ok(())
        })
    }

    /// Remove an empty directory.
    pub fn rmdir(&mut self, path: &str) -> Result<(), Ext4RwError> {
        let path = path.to_string();
        self.with_txn(move |vol| {
            let parts = Self::normalize_parts(&path)?;
            if parts.is_empty() {
                return Err(Ext4RwError::BadName);
            }
            let name = parts[parts.len() - 1];
            let parent_ino = if parts.len() == 1 {
                ROOT_INO
            } else {
                let parent = parts[..parts.len() - 1].join("/");
                vol.lookup(&parent)?
            };
            let Some((ino, ft)) = vol.lookup_in_dir(parent_ino, name)? else {
                return Err(Ext4RwError::NotFound);
            };
            if ft != FT_DIR {
                return Err(Ext4RwError::NotADir);
            }
            for (n, _, _) in vol.list_dir_ino(ino)? {
                if n != "." && n != ".." {
                    return Err(Ext4RwError::NotEmpty);
                }
            }
            let inode = vol.read_inode(ino)?;
            vol.free_inode_blocks(&inode)?;
            vol.write_inode(ino, &Inode::zeroed())?;
            vol.free_inode_num(ino)?;
            vol.dir_remove(parent_ino, name)?;
            let mut parent = vol.read_inode(parent_ino)?;
            parent.links = parent.links.saturating_sub(1);
            parent.ctime = wall_secs();
            vol.write_inode(parent_ino, &parent)?;
            let g = (ino - 1) / vol.inodes_per_group;
            vol.groups[g as usize].used_dirs = vol.groups[g as usize].used_dirs.saturating_sub(1);
            vol.sync_group_desc(g)?;
            Ok(())
        })
    }

    /// Stat a path (file or directory).
    pub fn stat(&mut self, path: &str) -> Result<FileStat, Ext4RwError> {
        let ino = self.lookup(path)?;
        let i = self.read_inode(ino)?;
        Ok(FileStat {
            ino,
            mode: i.mode,
            uid: i.uid,
            gid: i.gid,
            size: i.size,
            atime: i.atime,
            mtime: i.mtime,
            ctime: i.ctime,
            nlink: i.links,
        })
    }

    /// Set the owner field (agent id). Cosmetic / audit — not an authority gate.
    pub fn chown(&mut self, path: &str, uid: u32) -> Result<(), Ext4RwError> {
        let path = path.to_string();
        self.with_txn(move |vol| {
            let ino = vol.lookup(&path)?;
            let mut i = vol.read_inode(ino)?;
            i.uid = uid;
            i.ctime = wall_secs();
            vol.write_inode(ino, &i)
        })
    }

    /// Set permission bits (low 12 bits of mode). Does **not** change file type.
    /// Cosmetic for `/ls -l`; Synapse scope remains the real gate.
    pub fn chmod(&mut self, path: &str, mode_bits: u16) -> Result<(), Ext4RwError> {
        let path = path.to_string();
        self.with_txn(move |vol| {
            let ino = vol.lookup(&path)?;
            let mut i = vol.read_inode(ino)?;
            let ftype = i.mode & 0xf000;
            i.mode = ftype | (mode_bits & 0x0fff);
            i.ctime = wall_secs();
            vol.write_inode(ino, &i)
        })
    }

    /// Rename/move a file or empty directory within this volume (one transaction).
    ///
    /// - Same volume only.
    /// - If `to` exists as a regular file, it is replaced.
    /// - If `to` exists as a directory, refused (`Exists`).
    /// - Directory `from` must be empty (no children other than `.`/`..`).
    pub fn rename(&mut self, from: &str, to: &str) -> Result<(), Ext4RwError> {
        let from = from.to_string();
        let to = to.to_string();
        self.with_txn(move |vol| {
            let from_parts = Self::normalize_parts(&from)?;
            let to_parts = Self::normalize_parts(&to)?;
            if from_parts.is_empty() || to_parts.is_empty() {
                return Err(Ext4RwError::BadName);
            }
            if from_parts == to_parts {
                return Ok(());
            }
            // Refuse moving a directory into itself.
            if to_parts.len() > from_parts.len() && to_parts[..from_parts.len()] == from_parts[..] {
                return Err(Ext4RwError::BadName);
            }
            let from_name = from_parts[from_parts.len() - 1];
            let from_parent = if from_parts.len() == 1 {
                ROOT_INO
            } else {
                vol.lookup(&from_parts[..from_parts.len() - 1].join("/"))?
            };
            // ensure_parent creates intermediate dirs for `to` (same txn).
            let (to_parent, to_name) = vol.ensure_parent(&to)?;
            let Some((ino, ft)) = vol.lookup_in_dir(from_parent, from_name)? else {
                return Err(Ext4RwError::NotFound);
            };
            // Destination conflict.
            if let Some((dst_ino, dst_ft)) = vol.lookup_in_dir(to_parent, &to_name)? {
                if dst_ft == FT_DIR {
                    return Err(Ext4RwError::Exists);
                }
                // Replace regular file.
                let old = vol.read_inode(dst_ino)?;
                vol.free_inode_blocks(&old)?;
                vol.write_inode(dst_ino, &Inode::zeroed())?;
                vol.free_inode_num(dst_ino)?;
                vol.dir_remove(to_parent, &to_name)?;
            }
            if ft == FT_DIR {
                // Only empty dirs (other than . / ..).
                for (n, _, _) in vol.list_dir_ino(ino)? {
                    if n != "." && n != ".." {
                        return Err(Ext4RwError::NotEmpty);
                    }
                }
            }
            vol.dir_remove(from_parent, from_name)?;
            vol.dir_add(to_parent, &to_name, ino, ft)?;
            if ft == FT_DIR && from_parent != to_parent {
                // Update `..` to the new parent and fix link counts.
                vol.dir_remove(ino, "..")?;
                vol.dir_add(ino, "..", to_parent, FT_DIR)?;
                let mut old_p = vol.read_inode(from_parent)?;
                old_p.links = old_p.links.saturating_sub(1);
                old_p.ctime = wall_secs();
                vol.write_inode(from_parent, &old_p)?;
                let mut new_p = vol.read_inode(to_parent)?;
                new_p.links = new_p.links.saturating_add(1);
                new_p.ctime = wall_secs();
                vol.write_inode(to_parent, &new_p)?;
            }
            let mut i = vol.read_inode(ino)?;
            i.ctime = wall_secs();
            vol.write_inode(ino, &i)?;
            Ok(())
        })
    }

    /// List one directory: `(name, is_dir)`, excluding `.` / `..`.
    pub fn readdir(&mut self, path: &str) -> Result<Vec<(String, bool)>, Ext4RwError> {
        let ino = self.lookup(path)?;
        let mut out = Vec::new();
        for (n, _, ft) in self.list_dir_ino(ino)? {
            if n == "." || n == ".." {
                continue;
            }
            out.push((n, ft == FT_DIR));
        }
        Ok(out)
    }

    /// Depth-first list of every regular-file path under `path` (absolute-style,
    /// no leading slash), for the store cache rebuild.
    pub fn list_files_recursive(&mut self, path: &str) -> Result<Vec<String>, Ext4RwError> {
        let mut out = Vec::new();
        self.walk_files(path, &mut out)?;
        Ok(out)
    }

    fn walk_files(&mut self, path: &str, out: &mut Vec<String>) -> Result<(), Ext4RwError> {
        let entries = self.readdir(path)?;
        for (name, is_dir) in entries {
            let child = if path == "/" || path.is_empty() {
                name.clone()
            } else {
                alloc::format!("{}/{}", path.trim_start_matches('/'), name)
            };
            if is_dir {
                self.walk_files(&child, out)?;
            } else {
                out.push(child);
            }
        }
        Ok(())
    }
}

// ── pure dirent helpers ──────────────────────────────────────────────────

/// Try to insert a dirent into a single directory block. Returns true on success.
pub fn try_insert_dirent(buf: &mut [u8; EBS], name: &[u8], ino: u64, ftype: u8, need: usize) -> bool {
    let mut off = 0usize;
    while off + 8 <= EBS {
        let ent_ino = le32(buf, off) as u64;
        let rec_len = le16(buf, off + 4) as usize;
        let name_len = buf[off + 6] as usize;
        if rec_len < 8 || off + rec_len > EBS {
            break;
        }
        // Reuse a deleted entry (ino == 0) that is large enough.
        if ent_ino == 0 && rec_len >= need {
            // Claim the whole rec_len (simple; spare room stays inside this entry
            // until a later split or a new block).
            put32(buf, off, ino as u32);
            buf[off + 6] = name.len() as u8;
            buf[off + 7] = ftype;
            buf[off + 8..off + 8 + name.len()].copy_from_slice(name);
            return true;
        }
        let real = dirent_needed(name_len);
        // Split a live entry that has spare rec_len.
        if ent_ino != 0 && rec_len >= real + need {
            let new_off = off + real;
            let new_rec = rec_len - real;
            put16(buf, off + 4, real as u16);
            for x in &mut buf[new_off..new_off + new_rec] {
                *x = 0;
            }
            put32(buf, new_off, ino as u32);
            put16(buf, new_off + 4, new_rec as u16);
            buf[new_off + 6] = name.len() as u8;
            buf[new_off + 7] = ftype;
            buf[new_off + 8..new_off + 8 + name.len()].copy_from_slice(name);
            return true;
        }
        off += rec_len;
    }
    false
}

/// Zero the inode of the dirent named `name` (merge into previous rec_len when
/// possible). Returns true if found.
pub fn try_remove_dirent(buf: &mut [u8; EBS], name: &[u8]) -> bool {
    let mut off = 0usize;
    let mut prev: Option<usize> = None;
    while off + 8 <= EBS {
        let ent_ino = le32(buf, off) as u64;
        let rec_len = le16(buf, off + 4) as usize;
        let name_len = buf[off + 6] as usize;
        if rec_len < 8 || off + rec_len > EBS {
            break;
        }
        if ent_ino != 0 && name_len == name.len() && &buf[off + 8..off + 8 + name_len] == name {
            put32(buf, off, 0);
            buf[off + 6] = 0;
            buf[off + 7] = 0;
            // Absorb into previous entry's rec_len when present.
            if let Some(p) = prev {
                let prev_rec = le16(buf, p + 4) as usize;
                put16(buf, p + 4, (prev_rec + rec_len) as u16);
            }
            return true;
        }
        prev = Some(off);
        off += rec_len;
    }
    false
}

// ── tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::ext4::{Ext4Writer, FileSpec};
    use crate::block::ext4_read::Ext4Reader;
    use crate::block::ramdisk::RamDisk;
    use alloc::vec;

    fn disk() -> RamDisk {
        // 8 MiB — matches the existing ext4 layout tests.
        RamDisk::new(16384)
    }

    fn fresh() -> RamDisk {
        let mut d = disk();
        Ext4Writer::format(&mut d, &[]).expect("format");
        d
    }

    #[test_case]
    fn dirent_needed_rounds_name_to_4() {
        assert_eq!(dirent_needed(0), 8);
        assert_eq!(dirent_needed(1), 12);
        assert_eq!(dirent_needed(4), 12);
        assert_eq!(dirent_needed(5), 16);
    }

    #[test_case]
    fn try_insert_and_remove_dirent_round_trip() {
        // One spanning empty-ish block: a single deleted-style pad after "." .
        let mut buf = [0u8; EBS];
        // Entry "." occupying minimal, rest free via large rec_len on a real entry.
        put32(&mut buf, 0, 2);
        put16(&mut buf, 4, 12);
        buf[6] = 1;
        buf[7] = FT_DIR;
        buf[8] = b'.';
        // Pad entry ino=0 covering the rest — classic after delete, or we split.
        put32(&mut buf, 12, 0);
        put16(&mut buf, 16, (EBS - 12) as u16);
        assert!(try_insert_dirent(&mut buf, b"a.txt", 11, FT_REG, dirent_needed(5)));
        // Find it.
        let mut found = false;
        let mut off = 0usize;
        while off + 8 <= EBS {
            let ino = le32(&buf, off) as u64;
            let rec = le16(&buf, off + 4) as usize;
            let nl = buf[off + 6] as usize;
            if ino == 11 && nl == 5 && &buf[off + 8..off + 13] == b"a.txt" {
                found = true;
                break;
            }
            if rec < 8 {
                break;
            }
            off += rec;
        }
        assert!(found, "inserted dirent not found");
        assert!(try_remove_dirent(&mut buf, b"a.txt"));
        assert!(!try_remove_dirent(&mut buf, b"a.txt"), "second remove must miss");
    }

    #[test_case]
    fn write_read_unlink_round_trip() {
        let mut d = fresh();
        {
            let mut vol = Ext4Rw::open(&mut d).expect("open");
            vol.write("hello.txt", b"world").expect("write");
            assert_eq!(vol.read("hello.txt").unwrap(), b"world");
            vol.write("hello.txt", b"longer contents here").expect("grow");
            assert_eq!(vol.read("hello.txt").unwrap(), b"longer contents here");
            vol.write("hello.txt", b"x").expect("shrink");
            assert_eq!(vol.read("hello.txt").unwrap(), b"x");
            vol.unlink("hello.txt").expect("unlink");
            assert!(matches!(vol.read("hello.txt"), Err(Ext4RwError::NotFound)));
        }
        // Remount — still consistent, file gone.
        let mut vol = Ext4Rw::open(&mut d).expect("reopen");
        assert!(!vol.exists("hello.txt"));
    }

    #[test_case]
    fn nested_path_mkdir_p_and_recursive_list() {
        let mut d = fresh();
        let mut vol = Ext4Rw::open(&mut d).expect("open");
        vol.write("agent/1/SOUL.md", b"# soul").expect("nested write");
        vol.write("agent/1/memory/note", b"n").expect("deeper");
        vol.write("agent/2/SOUL.md", b"# two").expect("sib");
        assert_eq!(vol.read("agent/1/SOUL.md").unwrap(), b"# soul");
        let mut files = vol.list_files_recursive("/").unwrap();
        files.sort();
        assert_eq!(
            files,
            vec![
                String::from("agent/1/SOUL.md"),
                String::from("agent/1/memory/note"),
                String::from("agent/2/SOUL.md"),
            ]
        );
    }

    #[test_case]
    fn rmdir_refuses_nonempty_then_succeeds() {
        let mut d = fresh();
        let mut vol = Ext4Rw::open(&mut d).expect("open");
        vol.write("d/f", b"1").expect("w");
        assert_eq!(vol.rmdir("d"), Err(Ext4RwError::NotEmpty));
        vol.unlink("d/f").expect("u");
        vol.rmdir("d").expect("rmdir");
        assert!(!vol.exists("d"));
    }

    #[test_case]
    fn multi_block_file_and_reader_compat() {
        // Larger than one block; also visible to the RO Ext4Reader after close.
        let mut d = fresh();
        let payload: Vec<u8> = (0..10000u32).map(|i| (i % 251) as u8).collect();
        {
            let mut vol = Ext4Rw::open(&mut d).expect("open");
            vol.write("big.bin", &payload).expect("write big");
            assert_eq!(vol.read("big.bin").unwrap(), payload);
        }
        let mut r = Ext4Reader::open(&mut d).expect("ro open");
        let mut buf = vec![0u8; payload.len()];
        let n = r.read_root_file("big.bin", &mut buf).expect("ro read");
        assert_eq!(&buf[..n], &payload[..]);
    }

    #[test_case]
    fn many_files_do_not_require_reformat() {
        // The property that replaces O(partition) sync: N creates leave the
        // original free-block pool depleted by O(N), not a full rewrite.
        let mut d = fresh();
        let free_before = {
            let vol = Ext4Rw::open(&mut d).expect("open");
            vol.free_blocks
        };
        {
            let mut vol = Ext4Rw::open(&mut d).expect("open");
            for i in 0..40 {
                let name = alloc::format!("f{i:02}.txt");
                vol.write(&name, b"payload").expect("w");
            }
            assert!(vol.free_blocks < free_before);
            assert!(vol.free_blocks + 200 > free_before, "should not burn the whole volume");
            for i in 0..40 {
                let name = alloc::format!("f{i:02}.txt");
                assert_eq!(vol.read(&name).unwrap(), b"payload");
            }
        }
    }

    #[test_case]
    fn preseeded_format_files_still_readable_after_rw_open() {
        let a = b"alpha";
        let files = [FileSpec { name: "pre", data: a.as_slice() }];
        let mut d = disk();
        Ext4Writer::format(&mut d, &files).unwrap();
        let mut vol = Ext4Rw::open(&mut d).expect("open");
        assert_eq!(vol.read("pre").unwrap(), a);
        vol.write("post", b"beta").unwrap();
        assert_eq!(vol.read("post").unwrap(), b"beta");
        assert_eq!(vol.read("pre").unwrap(), a);
    }

    #[test_case]
    fn rename_file_same_dir_and_across_dirs() {
        let mut d = fresh();
        let mut vol = Ext4Rw::open(&mut d).expect("open");
        vol.write("a.txt", b"one").unwrap();
        vol.rename("a.txt", "b.txt").unwrap();
        assert!(!vol.exists("a.txt"));
        assert_eq!(vol.read("b.txt").unwrap(), b"one");
        vol.rename("b.txt", "sub/c.txt").unwrap();
        assert!(!vol.exists("b.txt"));
        assert_eq!(vol.read("sub/c.txt").unwrap(), b"one");
        // Remount preserves rename.
        drop(vol);
        let mut vol = Ext4Rw::open(&mut d).expect("reopen");
        assert_eq!(vol.read("sub/c.txt").unwrap(), b"one");
    }

    #[test_case]
    fn rename_replaces_existing_file_not_dir() {
        let mut d = fresh();
        let mut vol = Ext4Rw::open(&mut d).expect("open");
        vol.write("old", b"AAA").unwrap();
        vol.write("new", b"BBB").unwrap();
        vol.rename("old", "new").unwrap();
        assert_eq!(vol.read("new").unwrap(), b"AAA");
        assert!(!vol.exists("old"));
        vol.mkdir("d").unwrap();
        vol.write("x", b"x").unwrap();
        assert_eq!(vol.rename("x", "d"), Err(Ext4RwError::Exists));
    }

    #[test_case]
    fn stat_and_chown_chmod_and_mtime_update() {
        let mut d = fresh();
        let mut vol = Ext4Rw::open(&mut d).expect("open");
        vol.write("f", b"hi").unwrap();
        let s0 = vol.stat("f").unwrap();
        assert!(s0.is_reg());
        assert_eq!(s0.size, 2);
        assert_eq!(s0.mode & 0o777, 0o644);
        vol.chown("f", 7).unwrap();
        vol.chmod("f", 0o600).unwrap();
        let s1 = vol.stat("f").unwrap();
        assert_eq!(s1.uid, 7);
        assert_eq!(s1.mode & 0o777, 0o600);
        // Content write bumps mtime (may equal if wall clock is frozen at default).
        let before = s1.mtime;
        vol.write("f", b"hello").unwrap();
        let s2 = vol.stat("f").unwrap();
        assert_eq!(s2.size, 5);
        assert!(s2.mtime >= before);
    }

    #[test_case]
    fn rename_empty_dir() {
        let mut d = fresh();
        let mut vol = Ext4Rw::open(&mut d).expect("open");
        vol.mkdir("d").unwrap();
        vol.rename("d", "e").unwrap();
        assert!(!vol.exists("d"));
        assert!(vol.stat("e").unwrap().is_dir());
        vol.write("e/f", b"x").unwrap();
        assert_eq!(vol.rename("e", "f"), Err(Ext4RwError::NotEmpty));
    }

    #[test_case]
    fn journal_survives_clean_remount() {
        let mut d = fresh();
        {
            let mut vol = Ext4Rw::open(&mut d).expect("open");
            vol.write("a.txt", b"one").unwrap();
            vol.write("b/c.txt", b"two").unwrap();
        }
        let mut vol = Ext4Rw::open(&mut d).expect("reopen");
        assert_eq!(vol.read("a.txt").unwrap(), b"one");
        assert_eq!(vol.read("b/c.txt").unwrap(), b"two");
    }

    #[test_case]
    fn power_fail_during_write_never_corrupts_sibling() {
        // For every prefix of the device-write stream of a create, killing the
        // volume mid-op and remounting must leave the volume mountable and must
        // not damage an already-committed sibling file. The new file is either
        // fully present or fully absent — never half-applied metadata.
        let writes_needed = {
            let mut d = fresh();
            let mut vol = Ext4Rw::open(&mut d).unwrap();
            vol.write("keep.txt", b"stable").unwrap();
            let base = vol.device_write_count();
            vol.write("new.txt", b"hello-world").unwrap();
            vol.device_write_count() - base
        };
        assert!(writes_needed > 2, "expected a multi-write transaction, got {writes_needed}");

        for fail_at in 0..writes_needed {
            let mut d = fresh();
            {
                let mut vol = Ext4Rw::open(&mut d).unwrap();
                vol.write("keep.txt", b"stable").unwrap();
                let base = vol.device_write_count();
                vol.set_fail_after_device_writes(Some(base + fail_at));
                let _ = vol.write("new.txt", b"hello-world");
            }
            let mut vol = Ext4Rw::open(&mut d).unwrap_or_else(|e| {
                panic!("remount failed after fault at write {fail_at}: {e:?}");
            });
            assert_eq!(
                vol.read("keep.txt").expect("sibling lost"),
                b"stable",
                "sibling corrupted at fail_at={fail_at}"
            );
            match vol.read("new.txt") {
                Ok(v) => assert_eq!(&v[..], b"hello-world", "torn new file at fail_at={fail_at}"),
                Err(Ext4RwError::NotFound) => {}
                Err(e) => panic!("unexpected read error at fail_at={fail_at}: {e:?}"),
            }
        }
    }

    #[test_case]
    fn power_fail_mid_overwrite_keeps_old_or_new_not_mix() {
        let mut d = fresh();
        {
            let mut vol = Ext4Rw::open(&mut d).unwrap();
            vol.write("f", b"AAAAAAAA").unwrap();
        }
        let writes_needed = {
            let mut d2 = fresh();
            let mut vol = Ext4Rw::open(&mut d2).unwrap();
            vol.write("f", b"AAAAAAAA").unwrap();
            let base = vol.device_write_count();
            vol.write("f", b"BBBBBBBBBBBB").unwrap();
            vol.device_write_count() - base
        };
        for fail_at in 0..writes_needed {
            let mut d = fresh();
            {
                let mut vol = Ext4Rw::open(&mut d).unwrap();
                vol.write("f", b"AAAAAAAA").unwrap();
                let base = vol.device_write_count();
                vol.set_fail_after_device_writes(Some(base + fail_at));
                let _ = vol.write("f", b"BBBBBBBBBBBB");
            }
            let mut vol = Ext4Rw::open(&mut d).expect("remount");
            let got = vol.read("f").expect("file must exist");
            assert!(
                got == b"AAAAAAAA" || got == b"BBBBBBBBBBBB",
                "mixed contents at fail_at={fail_at}: {got:?}"
            );
        }
    }
}
