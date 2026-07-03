//! A minimal **FAT16 writer** for the EFI System Partition. UEFI firmware only
//! reads FAT to find `/EFI/BOOT/BOOTX64.EFI`, so the ESP just needs to carry
//! the Limine loader; everything else (limine.conf, kernel, model) lives on the
//! ext4 partition (`block::ext4`), which Limine reads. Scope is exactly that:
//! format FAT16 + create directories + write files, using only 8.3 names
//! (`EFI`, `BOOT`, `BOOTX64.EFI` all qualify), so no long-filename entries are
//! needed. Not a general FAT implementation.

use crate::block::{BlockDevice, BlockError, BLOCK_SIZE};

/// Pick the smallest sectors-per-cluster keeping the cluster count within
/// FAT16's 4085..65525 range for `total` sectors — 2 KiB clusters for a small
/// ESP, up to 32 KiB for a model-carrying (~1 GiB) one.
fn pick_spc(total: u64) -> Option<u64> {
    for spc in [4u64, 8, 16, 32, 64] {
        let approx = (total - RESERVED - ROOT_DIR_SECTORS) / spc;
        let fat_sectors = ((approx + 2) * 2).div_ceil(BLOCK_SIZE as u64);
        let data_sectors = total - RESERVED - NUM_FATS * fat_sectors - ROOT_DIR_SECTORS;
        let clusters = data_sectors / spc;
        if (4085..65525).contains(&clusters) {
            return Some(spc);
        }
    }
    None
}
const RESERVED: u64 = 1;
const NUM_FATS: u64 = 2;
const ROOT_ENTRIES: u64 = 512;
const ROOT_DIR_SECTORS: u64 = ROOT_ENTRIES * 32 / BLOCK_SIZE as u64; // 32
const EOC: u16 = 0xffff;

const ATTR_DIR: u8 = 0x10;
const ATTR_ARCHIVE: u8 = 0x20;

pub struct FatWriter<'d, D: BlockDevice> {
    dev: &'d mut D,
    fat_start: u64,      // sector of FAT #0
    fat_sectors: u64,    // per FAT
    root_start: u64,     // sector of the fixed root directory
    data_start: u64,     // sector of cluster 2
    spc: u64,            // sectors per cluster (chosen per volume size)
    next_free_clus: u16, // sequential allocator (fresh FS)
}

impl<'d, D: BlockDevice> FatWriter<'d, D> {
    /// Format `dev` as FAT16 and return a writer positioned on the empty FS.
    pub fn format(dev: &'d mut D) -> Result<FatWriter<'d, D>, BlockError> {
        let total = dev.block_count();
        // Iterate FAT size vs cluster count to a consistent solution.
        let Some(spc) = pick_spc(total) else {
            // No cluster size puts this geometry in FAT16's range.
            return Err(BlockError::OutOfRange);
        };
        let approx_clusters = (total - RESERVED - ROOT_DIR_SECTORS) / spc;
        let fat_sectors = ((approx_clusters + 2) * 2).div_ceil(BLOCK_SIZE as u64);
        let fat_start = RESERVED;
        let root_start = RESERVED + NUM_FATS * fat_sectors;
        let data_start = root_start + ROOT_DIR_SECTORS;

        // Boot sector / BPB.
        let mut bs = [0u8; BLOCK_SIZE];
        bs[0] = 0xeb;
        bs[1] = 0x3c;
        bs[2] = 0x90; // jmp
        bs[3..11].copy_from_slice(b"MSWIN4.1");
        le16(&mut bs, 11, BLOCK_SIZE as u16); // bytes/sector
        bs[13] = spc as u8;
        le16(&mut bs, 14, RESERVED as u16);
        bs[16] = NUM_FATS as u8;
        le16(&mut bs, 17, ROOT_ENTRIES as u16);
        le16(&mut bs, 19, if total < 0x10000 { total as u16 } else { 0 });
        bs[21] = 0xf8; // media = fixed disk
        le16(&mut bs, 22, fat_sectors as u16);
        le16(&mut bs, 24, 32); // sectors/track
        le16(&mut bs, 26, 64); // heads
        le32(&mut bs, 32, if total >= 0x10000 { total as u32 } else { 0 }); // total_sectors_32
        bs[36] = 0x80; // drive number
        bs[38] = 0x29; // extended boot signature
        le32(&mut bs, 39, 0x1234_5678); // volume id
        bs[43..54].copy_from_slice(b"CHITTI ESP ");
        bs[54..62].copy_from_slice(b"FAT16   ");
        bs[510] = 0x55;
        bs[511] = 0xaa;
        dev.write_block(0, &bs)?;

        // Zero both FATs, then set the two reserved entries.
        let zero = [0u8; BLOCK_SIZE];
        for f in 0..NUM_FATS {
            for s in 0..fat_sectors {
                dev.write_block(fat_start + f * fat_sectors + s, &zero)?;
            }
        }
        // Zero the root directory region.
        for s in 0..ROOT_DIR_SECTORS {
            dev.write_block(root_start + s, &zero)?;
        }

        let mut w = FatWriter { dev, fat_start, fat_sectors, root_start, data_start, spc, next_free_clus: 2 };
        // FAT[0]=media|0xFF00, FAT[1]=EOC.
        w.set_fat(0, 0xfff8)?;
        w.set_fat(1, 0xffff)?;
        Ok(w)
    }

    fn set_fat(&mut self, clus: u16, val: u16) -> Result<(), BlockError> {
        let off = clus as u64 * 2;
        let sec = self.fat_start + off / BLOCK_SIZE as u64;
        let within = (off % BLOCK_SIZE as u64) as usize;
        // Update in both FAT copies.
        for f in 0..NUM_FATS {
            let s = sec + f * self.fat_sectors;
            let mut buf = [0u8; BLOCK_SIZE];
            self.dev.read_block(s, &mut buf)?;
            le16(&mut buf, within, val);
            self.dev.write_block(s, &buf)?;
        }
        Ok(())
    }

    fn cluster_sector(&self, clus: u16) -> u64 {
        self.data_start + (clus as u64 - 2) * self.spc
    }

    /// Allocate a chain of `n` clusters, linking them in the FAT (last = EOC).
    /// Allocation is sequential on this fresh-FS writer, so the chain is built
    /// a FAT sector at a time (read-modify-write once per affected sector per
    /// FAT copy) instead of once per cluster — ~256x fewer IOs for a large file.
    fn alloc_chain(&mut self, n: u64) -> Result<u16, BlockError> {
        let first = self.next_free_clus;
        let last = first as u64 + n - 1;
        let ents_per_sec = (BLOCK_SIZE / 2) as u64;
        let sec_lo = first as u64 / ents_per_sec;
        let sec_hi = last / ents_per_sec;
        for sec in sec_lo..=sec_hi {
            let mut buf = [0u8; BLOCK_SIZE];
            self.dev.read_block(self.fat_start + sec, &mut buf)?;
            for e in 0..ents_per_sec {
                let c = sec * ents_per_sec + e;
                if c < first as u64 || c > last {
                    continue;
                }
                let val = if c == last { EOC } else { (c + 1) as u16 };
                le16(&mut buf, (e * 2) as usize, val);
            }
            for f in 0..NUM_FATS {
                self.dev.write_block(self.fat_start + f * self.fat_sectors + sec, &buf)?;
            }
        }
        self.next_free_clus = (last + 1) as u16;
        Ok(first)
    }

    /// Write `data` into `first_clus`'s chain (chain must already be sized).
    /// Clusters from this writer are sequential, so the data region is one
    /// contiguous sector run — written with batched multi-sector requests.
    fn write_clusters(&mut self, first_clus: u16, data: &[u8]) -> Result<(), BlockError> {
        let base = self.cluster_sector(first_clus);
        let full = data.len() / BLOCK_SIZE * BLOCK_SIZE;
        if full > 0 {
            self.dev.write_blocks(base, &data[..full])?;
        }
        if full < data.len() {
            let mut tail = [0u8; BLOCK_SIZE];
            tail[..data.len() - full].copy_from_slice(&data[full..]);
            self.dev.write_block(base + (full / BLOCK_SIZE) as u64, &tail)?;
        }
        Ok(())
    }

    /// Create a subdirectory named `name` (8.3) whose entry goes in the
    /// directory at `parent_sector`/`parent_sectors` (root region or a cluster
    /// chain). Returns the new directory's first cluster.
    fn mkdir_in_root(&mut self, name: &str) -> Result<u16, BlockError> {
        let clus = self.alloc_chain(1)?;
        // Initialize the directory cluster with "." and ".." entries.
        let base = self.cluster_sector(clus);
        let zero = [0u8; BLOCK_SIZE];
        for s in 0..self.spc {
            self.dev.write_block(base + s, &zero)?;
        }
        let mut first = [0u8; BLOCK_SIZE];
        write_dir_entry(&mut first, 0, ".", ATTR_DIR, clus, 0);
        write_dir_entry(&mut first, 1, "..", ATTR_DIR, 0, 0); // ".." -> root (0)
        self.dev.write_block(base, &first)?;
        // Add the entry in the fixed root directory.
        self.add_root_entry(name, ATTR_DIR, clus, 0)?;
        Ok(clus)
    }

    /// Add an entry into the fixed root directory region (writes LFN entries +
    /// the 8.3 entry at consecutive free slots for long names).
    fn add_root_entry(&mut self, name: &str, attr: u8, first_clus: u16, size: u32) -> Result<(), BlockError> {
        let entries = make_entries(name, attr, first_clus, size);
        // Root dir spans a contiguous sector region; place all entries in order.
        let mut slot = 0usize;
        for s in 0..ROOT_DIR_SECTORS {
            let mut buf = [0u8; BLOCK_SIZE];
            self.dev.read_block(self.root_start + s, &mut buf)?;
            let mut dirty = false;
            for e in 0..(BLOCK_SIZE / 32) {
                if slot < entries.len() && (buf[e * 32] == 0x00 || buf[e * 32] == 0xe5) {
                    buf[e * 32..e * 32 + 32].copy_from_slice(&entries[slot]);
                    slot += 1;
                    dirty = true;
                }
            }
            if dirty {
                self.dev.write_block(self.root_start + s, &buf)?;
            }
            if slot >= entries.len() {
                return Ok(());
            }
        }
        Err(BlockError::OutOfRange)
    }

    /// Add an entry into a subdirectory (single-cluster dir) at `dir_clus`.
    fn add_dir_entry(&mut self, dir_clus: u16, name: &str, attr: u8, first_clus: u16, size: u32) -> Result<(), BlockError> {
        let entries = make_entries(name, attr, first_clus, size);
        let base = self.cluster_sector(dir_clus);
        let mut slot = 0usize;
        for s in 0..self.spc {
            let mut buf = [0u8; BLOCK_SIZE];
            self.dev.read_block(base + s, &mut buf)?;
            let mut dirty = false;
            for e in 0..(BLOCK_SIZE / 32) {
                if slot < entries.len() && (buf[e * 32] == 0x00 || buf[e * 32] == 0xe5) {
                    buf[e * 32..e * 32 + 32].copy_from_slice(&entries[slot]);
                    slot += 1;
                    dirty = true;
                }
            }
            if dirty {
                self.dev.write_block(base + s, &buf)?;
            }
            if slot >= entries.len() {
                return Ok(());
            }
        }
        Err(BlockError::OutOfRange)
    }

    /// Write a file `name` (8.3) into the FAT root directory with `data`.
    pub fn write_root_file(&mut self, name: &str, data: &[u8]) -> Result<(), BlockError> {
        let nclus = (data.len() as u64).div_ceil(self.spc * BLOCK_SIZE as u64).max(1);
        let fclus = self.alloc_chain(nclus)?;
        self.write_clusters(fclus, data)?;
        self.add_root_entry(name, ATTR_ARCHIVE, fclus, data.len() as u32)
    }

    /// Write `/EFI/BOOT/<name>` with `data` (the common case: the Limine EFI).
    pub fn write_efi_boot_file(&mut self, name: &str, data: &[u8]) -> Result<(), BlockError> {
        let efi = self.mkdir_in_root("EFI")?;
        let boot = self.alloc_chain(1)?;
        // init BOOT dir cluster
        let base = self.cluster_sector(boot);
        let zero = [0u8; BLOCK_SIZE];
        for s in 0..self.spc {
            self.dev.write_block(base + s, &zero)?;
        }
        let mut firstblk = [0u8; BLOCK_SIZE];
        write_dir_entry(&mut firstblk, 0, ".", ATTR_DIR, boot, 0);
        write_dir_entry(&mut firstblk, 1, "..", ATTR_DIR, efi, 0);
        self.dev.write_block(base, &firstblk)?;
        self.add_dir_entry(efi, "BOOT", ATTR_DIR, boot, 0)?;
        // The file itself.
        let nclus = (data.len() as u64).div_ceil(self.spc * BLOCK_SIZE as u64).max(1);
        let fclus = self.alloc_chain(nclus)?;
        self.write_clusters(fclus, data)?;
        self.add_dir_entry(boot, name, ATTR_ARCHIVE, fclus, data.len() as u32)?;
        Ok(())
    }
}

fn le16(buf: &mut [u8], off: usize, v: u16) {
    buf[off..off + 2].copy_from_slice(&v.to_le_bytes());
}
fn le32(buf: &mut [u8], off: usize, v: u32) {
    buf[off..off + 4].copy_from_slice(&v.to_le_bytes());
}

/// Write a 32-byte 8.3 directory entry at slot `idx` of `buf` (used only for the
/// `.` and `..` entries, which are always 8.3).
fn write_dir_entry(buf: &mut [u8], idx: usize, name: &str, attr: u8, first_clus: u16, size: u32) {
    let e = idx * 32;
    let mut short = [b' '; 11];
    if name == "." {
        short[0] = b'.';
    } else if name == ".." {
        short[0] = b'.';
        short[1] = b'.';
    } else {
        short = short_83(name).0;
    }
    buf[e..e + 32].copy_from_slice(&short_entry(&short, attr, first_clus, size));
}

/// Build a 32-byte 8.3 entry from an 11-byte packed name.
fn short_entry(short: &[u8; 11], attr: u8, first_clus: u16, size: u32) -> [u8; 32] {
    let mut e = [0u8; 32];
    e[0..11].copy_from_slice(short);
    e[11] = attr;
    e[20..22].copy_from_slice(&0u16.to_le_bytes()); // first_clus hi (0 for FAT16)
    e[26..28].copy_from_slice(&first_clus.to_le_bytes());
    e[28..32].copy_from_slice(&size.to_le_bytes());
    e
}

/// True if `name` fits a plain 8.3 short name (≤8 base, ≤3 ext, no lowercase
/// forcing — we uppercase, so accept any case; but reject if base>8 or ext>3).
fn is_8_3(name: &str) -> bool {
    let (base, ext) = name.split_once('.').unwrap_or((name, ""));
    base.len() <= 8 && ext.len() <= 3 && !name.is_empty() && !base.is_empty()
}

/// Pack `name` into an 11-byte 8.3 short name (returns the bytes + whether it
/// was lossy — i.e. an alias with `~1` was needed).
fn short_83(name: &str) -> ([u8; 11], bool) {
    let (base, ext) = name.split_once('.').unwrap_or((name, ""));
    let mut short = [b' '; 11];
    let clean = |c: u8| -> u8 {
        let c = c.to_ascii_uppercase();
        if c.is_ascii_alphanumeric() || b"$%'-_@~`!(){}^#&".contains(&c) {
            c
        } else {
            b'_'
        }
    };
    let lossy = base.len() > 8 || ext.len() > 3;
    if lossy {
        // "BASE~1" alias: first 6 cleaned base chars + "~1".
        let mut i = 0;
        for c in base.bytes().take(6) {
            short[i] = clean(c);
            i += 1;
        }
        short[6] = b'~';
        short[7] = b'1';
    } else {
        for (i, c) in base.bytes().take(8).enumerate() {
            short[i] = clean(c);
        }
    }
    for (i, c) in ext.bytes().take(3).enumerate() {
        short[8 + i] = clean(c);
    }
    (short, lossy)
}

/// VFAT LFN checksum of an 11-byte 8.3 short name.
fn lfn_checksum(short: &[u8; 11]) -> u8 {
    let mut sum: u8 = 0;
    for &b in short {
        sum = ((sum & 1) << 7).wrapping_add(sum >> 1).wrapping_add(b);
    }
    sum
}

/// Produce the directory entries for `name`: for a plain 8.3 name, one entry;
/// for a long name, the VFAT LFN entries (in reverse order) followed by the
/// 8.3 alias entry.
fn make_entries(name: &str, attr: u8, first_clus: u16, size: u32) -> alloc::vec::Vec<[u8; 32]> {
    use alloc::vec::Vec;
    let (short, lossy) = short_83(name);
    if is_8_3(name) && !lossy {
        return alloc::vec![short_entry(&short, attr, first_clus, size)];
    }
    let cksum = lfn_checksum(&short);
    let utf16: Vec<u16> = name.encode_utf16().collect();
    let nparts = utf16.len().div_ceil(13);
    let mut entries: Vec<[u8; 32]> = Vec::new();
    for seq in (1..=nparts).rev() {
        let mut e = [0u8; 32];
        e[0] = seq as u8 | if seq == nparts { 0x40 } else { 0 };
        e[11] = 0x0f; // LFN attribute
        e[13] = cksum;
        // 13 UTF-16 units at byte offsets 1,3,5,7,9 / 14,16,18,20,22,24 / 28,30.
        let offs = [1usize, 3, 5, 7, 9, 14, 16, 18, 20, 22, 24, 28, 30];
        for (k, &o) in offs.iter().enumerate() {
            let ci = (seq - 1) * 13 + k;
            let val: u16 = if ci < utf16.len() {
                utf16[ci]
            } else if ci == utf16.len() {
                0x0000
            } else {
                0xffff
            };
            e[o..o + 2].copy_from_slice(&val.to_le_bytes());
        }
        entries.push(e);
    }
    entries.push(short_entry(&short, attr, first_clus, size));
    entries
}
