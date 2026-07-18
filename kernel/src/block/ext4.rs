//! A from-scratch **ext4 write driver** (`/install`'s primary filesystem):
//! `mkfs` + write a set of files into the root directory. The on-disk layout is
//! the ext2/3/4 family with a minimal, correctness-first feature set —
//! `filetype + large_file + sparse_super`, 128-byte inodes, block-mapped files
//! (12 direct + single/double indirect, so large files like the model fit).
//! Linux's ext4 driver mounts it and Limine boots from it. The layout was
//! validated against `e2fsck`/`debugfs` in `tools/mkext4_ref.py` before this
//! port.
//!
//! Unlike the host prototype (which built the whole image in RAM), the kernel
//! **streams** file data directly to device blocks — the model is far larger
//! than the heap — and builds only the small metadata (bitmaps, GDT,
//! superblock) a block at a time.

use crate::block::{BlockDevice, BlockError, BLOCK_SIZE};
use alloc::vec;
use alloc::vec::Vec;

const EBS: usize = 4096; // ext block size
const SPB: u64 = (EBS / BLOCK_SIZE) as u64; // 512B sectors per ext block = 8
const INODE_SIZE: usize = 128;
const BPG: u64 = (EBS * 8) as u64; // blocks per group = 32768
const IPG: u64 = 4096; // inodes per group
const FIRST_INO: u64 = 11;
const ROOT_INO: u64 = 2;
const PTRS_PER: usize = EBS / 4; // 1024

const S_IFDIR: u16 = 0x4000;
const S_IFREG: u16 = 0x8000;

const INCOMPAT_FILETYPE: u32 = 0x0002;
/// Declared so the volume classifies (and mounts) as ext4 everywhere; legal
/// with zero extent-mapped inodes — the per-inode EXTENTS_FL decides, and all
/// our files are block-mapped (which ext4 drivers read fine).
const INCOMPAT_EXTENTS: u32 = 0x0040;
const ROCOMPAT_SPARSE_SUPER: u32 = 0x0001;
const ROCOMPAT_LARGE_FILE: u32 = 0x0002;

/// A file to write into the new filesystem's root: a name + its data (a slice
/// into module memory — the kernel binary, a model part, or generated text).
pub struct FileSpec<'a> {
    pub name: &'a str,
    pub data: &'a [u8],
}

impl FileSpec<'_> {
    fn len(&self) -> usize {
        self.data.len()
    }
}

fn sparse_has_super(g: u64) -> bool {
    if g == 0 || g == 1 {
        return true;
    }
    let is_pow = |base: u64| {
        let mut x = base;
        while x < g {
            x *= base;
        }
        x == g
    };
    is_pow(3) || is_pow(5) || is_pow(7)
}

/// The ext4 mkfs+writer over a block device (typically a `Partition`).
pub struct Ext4Writer<'d, D: BlockDevice> {
    dev: &'d mut D,
    total_blocks: u64,
    ngroups: u64,
    itable_blocks: u64,
    gdt_blocks: u64,
    meta_overhead: u64, // super + gdt, in sparse-super groups
    g_block_bitmap: Vec<u64>,
    g_inode_bitmap: Vec<u64>,
    g_inode_table: Vec<u64>,
    g_first_data: Vec<u64>,
    block_alloc: Vec<u64>, // per-group next-free absolute block
    used_blocks: u64,      // count of allocated data/indirect blocks
    next_ino: u64,
}

/// A finished inode to write into the table.
struct InodeRec {
    ino: u64,
    mode: u16,
    size: u64,
    blocks_512: u64,
    slots: [u32; 15],
    links: u16,
}

impl<'d, D: BlockDevice> Ext4Writer<'d, D> {
    fn write_eblock(&mut self, eblk: u64, buf: &[u8; EBS]) -> Result<(), BlockError> {
        self.dev.write_blocks(eblk * SPB, buf)
    }

    fn group_start(g: u64) -> u64 {
        g * BPG
    }
    fn blocks_in_group(&self, g: u64) -> u64 {
        (Self::group_start(g) + BPG).min(self.total_blocks) - Self::group_start(g)
    }

    fn alloc_block(&mut self, prefer: Option<u64>) -> Option<u64> {
        let order: Vec<u64> = match prefer {
            Some(p) => core::iter::once(p).chain((0..self.ngroups).filter(move |g| *g != p)).collect(),
            None => (0..self.ngroups).collect(),
        };
        for g in order {
            let end = Self::group_start(g) + self.blocks_in_group(g);
            if self.block_alloc[g as usize] < end {
                let b = self.block_alloc[g as usize];
                self.block_alloc[g as usize] += 1;
                self.used_blocks += 1;
                return Some(b);
            }
        }
        None
    }

    /// Format `dev` as ext4 and write `files` into the root directory.
    pub fn format(dev: &'d mut D, files: &[FileSpec]) -> Result<(), BlockError> {
        let total_blocks = dev.block_count() / SPB;
        if total_blocks < 64 {
            return Err(BlockError::OutOfRange);
        }
        let ngroups = total_blocks.div_ceil(BPG);
        let itable_blocks = (IPG * INODE_SIZE as u64).div_ceil(EBS as u64);
        let gdt_blocks = (ngroups * 32).div_ceil(EBS as u64);
        let meta_overhead = 1 + gdt_blocks;

        let mut w = Ext4Writer {
            dev,
            total_blocks,
            ngroups,
            itable_blocks,
            gdt_blocks,
            meta_overhead,
            g_block_bitmap: Vec::new(),
            g_inode_bitmap: Vec::new(),
            g_inode_table: Vec::new(),
            g_first_data: Vec::new(),
            block_alloc: Vec::new(),
            used_blocks: 0,
            next_ino: FIRST_INO,
        };
        w.layout();
        w.write_all(files)
    }

    fn layout(&mut self) {
        for g in 0..self.ngroups {
            let start = Self::group_start(g);
            let mut off = 0u64;
            if sparse_has_super(g) {
                off += self.meta_overhead;
            }
            let bb = start + off;
            off += 1;
            let ib = start + off;
            off += 1;
            let it = start + off;
            off += self.itable_blocks;
            self.g_block_bitmap.push(bb);
            self.g_inode_bitmap.push(ib);
            self.g_inode_table.push(it);
            self.g_first_data.push(start + off);
            self.block_alloc.push(start + off);
        }
    }

    /// Build the 15 i_block slots for `data_blocks`, allocating indirect blocks
    /// as needed. Returns (slots, total_meta_blocks_allocated).
    fn build_iblocks(&mut self, data_blocks: &[u64]) -> Result<([u32; 15], u64), BlockError> {
        let mut slots = [0u32; 15];
        let n = data_blocks.len();
        let mut indirect = 0u64;
        for k in 0..n.min(12) {
            slots[k] = data_blocks[k] as u32;
        }
        let mut idx = 12usize;
        if n > idx {
            let si = self.alloc_block(None).ok_or(BlockError::OutOfRange)?;
            indirect += 1;
            let arr = &data_blocks[12..(12 + PTRS_PER).min(n)];
            self.write_ptr_block(si, arr)?;
            slots[12] = si as u32;
            idx = 12 + arr.len();
        }
        if n > idx {
            let di = self.alloc_block(None).ok_or(BlockError::OutOfRange)?;
            indirect += 1;
            let mut singles: Vec<u64> = Vec::new();
            let rem = &data_blocks[idx..];
            let mut c = 0;
            while c < rem.len() && singles.len() < PTRS_PER {
                let chunk = &rem[c..(c + PTRS_PER).min(rem.len())];
                let si = self.alloc_block(None).ok_or(BlockError::OutOfRange)?;
                indirect += 1;
                self.write_ptr_block(si, chunk)?;
                singles.push(si);
                c += chunk.len();
            }
            self.write_ptr_block(di, &singles)?;
            slots[13] = di as u32;
        }
        Ok((slots, indirect))
    }

    fn write_ptr_block(&mut self, b: u64, ptrs: &[u64]) -> Result<(), BlockError> {
        let mut buf = [0u8; EBS];
        for (k, &p) in ptrs.iter().enumerate() {
            buf[k * 4..k * 4 + 4].copy_from_slice(&(p as u32).to_le_bytes());
        }
        self.write_eblock(b, &buf)
    }

    fn write_all(&mut self, files: &[FileSpec]) -> Result<(), BlockError> {
        let mut inodes: Vec<InodeRec> = Vec::new();
        let mut dir_entries: Vec<(u64, Vec<u8>, u8)> = Vec::new();
        dir_entries.push((ROOT_INO, b".".to_vec(), 2));
        dir_entries.push((ROOT_INO, b"..".to_vec(), 2));

        // Regular files: stream chunk data into freshly allocated blocks.
        for f in files {
            let size = f.len();
            let nblocks = size.div_ceil(EBS);
            let mut data_blocks = Vec::with_capacity(nblocks);
            for _ in 0..nblocks {
                data_blocks.push(self.alloc_block(None).ok_or(BlockError::OutOfRange)?);
            }
            self.stream_file(f, &data_blocks)?;
            let (slots, indirect) = self.build_iblocks(&data_blocks)?;
            let ino = self.next_ino;
            self.next_ino += 1;
            inodes.push(InodeRec {
                ino,
                mode: S_IFREG | 0o644,
                size: size as u64,
                blocks_512: (data_blocks.len() as u64 + indirect) * SPB,
                slots,
                links: 1,
            });
            dir_entries.push((ino, f.name.as_bytes().to_vec(), 1));
        }

        // Root directory: pack the (flat) entries into as many 4 KiB blocks as
        // they need — one dirent block per group. A single block overflowed
        // once the app-agent file set's names exceeded 4096 B of dirents
        // (KERNEL PANIC in `make_dir_block`). The reader already walks a
        // multi-block directory (`ext4_read::list_root` iterates
        // `size / block_size` blocks via `map_block`), so this stays readable.
        let groups = partition_dir_entries(&dir_entries);
        let mut root_blocks = Vec::with_capacity(groups.len());
        for _ in 0..groups.len() {
            root_blocks.push(self.alloc_block(Some(0)).ok_or(BlockError::OutOfRange)?);
        }
        for (g, &blk) in groups.iter().zip(root_blocks.iter()) {
            let dirbuf = make_dir_block(&dir_entries[g.clone()]);
            self.write_eblock(blk, &dirbuf)?;
        }
        let (slots, indirect) = self.build_iblocks(&root_blocks)?;
        inodes.push(InodeRec {
            ino: ROOT_INO,
            mode: S_IFDIR | 0o755,
            size: (root_blocks.len() * EBS) as u64,
            blocks_512: (root_blocks.len() as u64 + indirect) * SPB,
            slots,
            links: 2,
        });

        self.write_inode_table(&inodes)?;
        self.write_bitmaps_gdt_super(&inodes)
    }

    /// Write a file's bytes across its data blocks (zero-padding the tail).
    /// Blocks from `alloc_block` are sequential, so consecutive runs are
    /// written with batched multi-sector requests (vs one 512 B sector each).
    fn stream_file(&mut self, f: &FileSpec, data_blocks: &[u64]) -> Result<(), BlockError> {
        let data = f.data;
        let mut bi = 0usize;
        while bi < data_blocks.len() {
            // Extend the run while blocks stay consecutive.
            let mut run = 1usize;
            while bi + run < data_blocks.len() && data_blocks[bi + run] == data_blocks[bi] + run as u64 {
                run += 1;
            }
            let start = bi * EBS;
            let end = (start + run * EBS).min(data.len());
            let full = if end > start { (end - start) / BLOCK_SIZE * BLOCK_SIZE } else { 0 };
            let sector0 = data_blocks[bi] * SPB;
            if full > 0 {
                self.dev.write_blocks(sector0, &data[start..start + full])?;
            }
            // Zero-padded tail of the final partial sector + trailing sectors.
            let written_secs = (full / BLOCK_SIZE) as u64;
            let total_secs = run as u64 * SPB;
            if written_secs < total_secs {
                let mut tail = [0u8; BLOCK_SIZE];
                let rem = end - (start + full);
                tail[..rem].copy_from_slice(&data[start + full..end]);
                self.dev.write_block(sector0 + written_secs, &tail)?;
                let zero = [0u8; BLOCK_SIZE];
                for sx in written_secs + 1..total_secs {
                    self.dev.write_block(sector0 + sx, &zero)?;
                }
            }
            bi += run;
        }
        Ok(())
    }

    /// Write the inodes into their inode-table blocks (all our inodes land in
    /// group 0). Zeroes **every** reserved inode-table block in **every** group
    /// before overlaying the real inodes: we do not mark groups `INODE_UNINIT`,
    /// so e2fsck scans every inode slot — any block we leave unwritten would be
    /// read as a stale inode (e.g. leftover bytes from a prior filesystem).
    /// A correct mkfs must therefore initialise the whole inode table, not just
    /// the blocks that happen to hold a live inode.
    fn write_inode_table(&mut self, inodes: &[InodeRec]) -> Result<(), BlockError> {
        let inodes_per_block = (EBS / INODE_SIZE) as u64; // 32
        for g in 0..self.ngroups {
            let it0 = self.g_inode_table[g as usize];
            for blk in 0..self.itable_blocks {
                let mut buf = [0u8; EBS];
                // Group 0 holds inodes 1..=IPG; overlay any that fall in `blk`.
                if g == 0 {
                    for rec in inodes.iter() {
                        let idx = rec.ino - 1; // 0-based inode index
                        if idx / inodes_per_block == blk {
                            let off = (idx % inodes_per_block) as usize * INODE_SIZE;
                            encode_inode(&mut buf[off..off + INODE_SIZE], rec);
                        }
                    }
                }
                self.write_eblock(it0 + blk, &buf)?;
            }
        }
        Ok(())
    }

    fn write_bitmaps_gdt_super(&mut self, inodes: &[InodeRec]) -> Result<(), BlockError> {
        let used_inodes = inode_used_set(inodes);
        let mut total_free_blocks = 0u64;
        let mut total_free_inodes = 0u64;

        // GDT (built in RAM; small).
        let mut gdt = vec![0u8; (self.gdt_blocks * EBS as u64) as usize];

        for g in 0..self.ngroups {
            let start = Self::group_start(g);
            let nb = self.blocks_in_group(g);
            // Block bitmap for group g.
            let mut bm = [0u8; EBS];
            let mut used_b = 0u64;
            for b in start..start + nb {
                let is_meta = (sparse_has_super(g) && b < start + self.meta_overhead)
                    || b == self.g_block_bitmap[g as usize]
                    || b == self.g_inode_bitmap[g as usize]
                    || (self.g_inode_table[g as usize]..self.g_inode_table[g as usize] + self.itable_blocks).contains(&b);
                // A data/indirect block is allocated iff below the group's cursor
                // and at/after first-data (alloc is sequential from first-data).
                let is_data = b >= self.g_first_data[g as usize] && b < self.block_alloc[g as usize];
                if is_meta || is_data {
                    let bit = (b - start) as usize;
                    bm[bit / 8] |= 1 << (bit % 8);
                    used_b += 1;
                }
            }
            for bit in nb as usize..EBS * 8 {
                bm[bit / 8] |= 1 << (bit % 8);
            }
            self.write_eblock(self.g_block_bitmap[g as usize], &bm)?;
            let free_b = nb - used_b;

            // Inode bitmap for group g.
            let mut ibm = [0u8; EBS];
            let base = g * IPG;
            let mut used_i = 0u64;
            for within in 0..IPG {
                let ino = base + within + 1;
                if used_inodes.contains(&ino) {
                    ibm[(within / 8) as usize] |= 1 << (within % 8);
                    used_i += 1;
                }
            }
            for bit in IPG as usize..EBS * 8 {
                ibm[bit / 8] |= 1 << (bit % 8);
            }
            self.write_eblock(self.g_inode_bitmap[g as usize], &ibm)?;
            let free_i = IPG - used_i;

            total_free_blocks += free_b;
            total_free_inodes += free_i;
            let e = (g * 32) as usize;
            gdt[e..e + 4].copy_from_slice(&(self.g_block_bitmap[g as usize] as u32).to_le_bytes());
            gdt[e + 4..e + 8].copy_from_slice(&(self.g_inode_bitmap[g as usize] as u32).to_le_bytes());
            gdt[e + 8..e + 12].copy_from_slice(&(self.g_inode_table[g as usize] as u32).to_le_bytes());
            gdt[e + 12..e + 14].copy_from_slice(&(free_b as u16).to_le_bytes());
            gdt[e + 14..e + 16].copy_from_slice(&(free_i as u16).to_le_bytes());
            let dirs: u16 = if g == 0 { 1 } else { 0 };
            gdt[e + 16..e + 18].copy_from_slice(&dirs.to_le_bytes());
        }

        // Write GDT into every sparse-super group (right after its superblock).
        for g in 0..self.ngroups {
            if sparse_has_super(g) {
                let gdt_start = Self::group_start(g) + 1;
                for (i, chunk) in gdt.chunks(EBS).enumerate() {
                    let mut buf = [0u8; EBS];
                    buf[..chunk.len()].copy_from_slice(chunk);
                    self.write_eblock(gdt_start + i as u64, &buf)?;
                }
            }
        }

        // Superblock (+ sparse backups).
        let total_inodes = IPG * self.ngroups;
        for g in 0..self.ngroups {
            if !sparse_has_super(g) {
                continue;
            }
            let mut sb = [0u8; EBS];
            encode_superblock(&mut sb, self.total_blocks, total_inodes, total_free_blocks, total_free_inodes, g as u16);
            // Group 0's superblock sits at byte offset 1024 of block 0; backups
            // sit at the very start of the group's first block.
            if g == 0 {
                // Block 0 holds the 1024-byte boot area then the superblock at
                // byte offset 1024.
                let mut real = [0u8; EBS];
                real[1024..2048].copy_from_slice(&sb[0..1024]);
                self.write_eblock(0, &real)?;
            } else {
                self.write_eblock(Self::group_start(g), &sb)?;
            }
        }
        Ok(())
    }
}

fn inode_used_set(inodes: &[InodeRec]) -> Vec<u64> {
    // inodes 1..=10 reserved, plus root(2, in that range) and our files.
    let mut set: Vec<u64> = (1..=10).collect();
    for r in inodes {
        if !set.contains(&r.ino) {
            set.push(r.ino);
        }
    }
    set
}

fn encode_inode(buf: &mut [u8], r: &InodeRec) {
    buf[0..2].copy_from_slice(&r.mode.to_le_bytes());
    buf[4..8].copy_from_slice(&((r.size & 0xffff_ffff) as u32).to_le_bytes());
    buf[26..28].copy_from_slice(&r.links.to_le_bytes());
    buf[28..32].copy_from_slice(&(r.blocks_512 as u32).to_le_bytes());
    for k in 0..15 {
        buf[40 + k * 4..40 + k * 4 + 4].copy_from_slice(&r.slots[k].to_le_bytes());
    }
    // size_high (dir_acl) for large regular files (large_file feature).
    if r.mode & S_IFDIR == 0 && r.size >= (1 << 31) {
        buf[108..112].copy_from_slice(&((r.size >> 32) as u32).to_le_bytes());
    }
}

fn encode_superblock(sb: &mut [u8], total_blocks: u64, total_inodes: u64, free_b: u64, free_i: u64, group_nr: u16) {
    let put32 = |sb: &mut [u8], o: usize, v: u32| sb[o..o + 4].copy_from_slice(&v.to_le_bytes());
    let put16 = |sb: &mut [u8], o: usize, v: u16| sb[o..o + 2].copy_from_slice(&v.to_le_bytes());
    put32(sb, 0, total_inodes as u32); // s_inodes_count
    put32(sb, 4, total_blocks as u32); // s_blocks_count
    put32(sb, 8, (total_blocks / 20) as u32); // r_blocks
    put32(sb, 12, free_b as u32); // free blocks
    put32(sb, 16, free_i as u32); // free inodes
    put32(sb, 20, 0); // first_data_block (0 for 4K)
    put32(sb, 24, 2); // log_block_size => 4K
    put32(sb, 28, 2); // log_cluster_size
    put32(sb, 32, BPG as u32); // blocks_per_group
    put32(sb, 36, BPG as u32); // clusters_per_group
    put32(sb, 40, IPG as u32); // inodes_per_group
    put16(sb, 56, 0xEF53); // magic
    put16(sb, 58, 1); // state = clean
    put16(sb, 60, 1); // errors = continue
    put16(sb, 90, group_nr); // s_block_group_nr
    put32(sb, 76, 1); // rev_level = dynamic
    put32(sb, 84, FIRST_INO as u32); // first_ino
    put16(sb, 88, INODE_SIZE as u16); // inode_size
    put32(sb, 92, 0); // feature_compat
    put32(sb, 96, INCOMPAT_FILETYPE | INCOMPAT_EXTENTS); // feature_incompat
    put32(sb, 100, ROCOMPAT_SPARSE_SUPER | ROCOMPAT_LARGE_FILE); // feature_ro_compat
}

/// On-disk size of one directory entry: an 8-byte header + the name rounded up
/// to 4 bytes.
fn dirent_len(name_len: usize) -> usize {
    8 + name_len.div_ceil(4) * 4
}

/// Split directory entries into contiguous groups that each fit in one 4 KiB
/// dirent block (each becomes one block of the directory inode, in order).
/// Greedy: start a new block whenever the next entry would overflow the current
/// one. A directory that used to be a single [`make_dir_block`] call now spans
/// `groups.len()` blocks — the writer must give the inode all of them.
fn partition_dir_entries(entries: &[(u64, Vec<u8>, u8)]) -> Vec<core::ops::Range<usize>> {
    let mut groups = Vec::new();
    let mut start = 0usize;
    let mut used = 0usize;
    for (i, e) in entries.iter().enumerate() {
        let need = dirent_len(e.1.len());
        if i > start && used + need > EBS {
            groups.push(start..i);
            start = i;
            used = 0;
        }
        used += need;
    }
    groups.push(start..entries.len());
    groups
}

/// Pack directory entries into one 4 KiB block; the last entry's rec_len spans
/// to the block end (classic ext2 linked directory). Callers must ensure the
/// entries fit (`partition_dir_entries`); `make_dir_block` on an over-full
/// group would write past the block.
fn make_dir_block(entries: &[(u64, Vec<u8>, u8)]) -> [u8; EBS] {
    let mut buf = [0u8; EBS];
    let mut off = 0usize;
    for (j, (ino, name, ft)) in entries.iter().enumerate() {
        let reclen = if j == entries.len() - 1 { EBS - off } else { dirent_len(name.len()) };
        buf[off..off + 4].copy_from_slice(&(*ino as u32).to_le_bytes());
        buf[off + 4..off + 6].copy_from_slice(&(reclen as u16).to_le_bytes());
        buf[off + 6] = name.len() as u8;
        buf[off + 7] = *ft;
        buf[off + 8..off + 8 + name.len()].copy_from_slice(name);
        off += reclen;
    }
    buf
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;
    use alloc::vec::Vec;

    /// A flat file set whose dirents exceed one 4 KiB block must split into
    /// multiple contiguous groups, each within `EBS`, covering every entry in
    /// order — the fix for the `make_dir_block` overflow panic (real hardware
    /// hit it once the app-agent set grew past a single root dir block).
    #[test_case]
    fn partition_dir_entries_splits_and_covers() {
        let entries: Vec<(u64, Vec<u8>, u8)> =
            (0..300).map(|i| (2u64, format!("agent/{i:04}/SOUL.md").into_bytes(), 1u8)).collect();
        let groups = partition_dir_entries(&entries);
        assert!(groups.len() >= 2, "should span multiple blocks, got {}", groups.len());
        let mut expect = 0usize;
        for g in &groups {
            assert_eq!(g.start, expect, "groups must be contiguous");
            assert!(g.end > g.start, "empty group");
            let used: usize = entries[g.clone()].iter().map(|e| dirent_len(e.1.len())).sum();
            assert!(used <= EBS, "group {g:?} uses {used} > {EBS}");
            expect = g.end;
        }
        assert_eq!(expect, entries.len(), "all entries covered");
    }

    /// `make_dir_block` on a partitioned group stays within the block and its
    /// rec_len chain reaches exactly the block end — exactly what the reader
    /// (`ext4_read::list_root`) walks, so every entry is readable.
    #[test_case]
    fn make_dir_block_reclen_chain_reaches_end() {
        let entries: Vec<(u64, Vec<u8>, u8)> =
            (0..100).map(|i| (2u64, format!("file{i:03}.txt").into_bytes(), 1u8)).collect();
        let groups = partition_dir_entries(&entries);
        let blk = make_dir_block(&entries[groups[0].clone()]);
        let mut off = 0usize;
        let mut count = 0usize;
        while off + 8 <= EBS {
            let rec_len = u16::from_le_bytes([blk[off + 4], blk[off + 5]]) as usize;
            assert!(rec_len >= 8, "rec_len {rec_len} at off {off}");
            let nl = blk[off + 6] as usize;
            assert!(off + 8 + nl <= EBS, "name overruns block at off {off}");
            count += 1;
            off += rec_len;
        }
        assert_eq!(off, EBS, "rec_len chain must reach block end exactly");
        assert_eq!(count, groups[0].len(), "walked every entry in the block");
    }
}
