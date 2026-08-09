//! **Pure exFAT on-disk format** — the spec halves of the read/write driver
//! (`super::exfat_rw`), all byte-arithmetic and no I/O, so they can be
//! unit-tested off-hardware (the standing rule for anything a transcription
//! slip would corrupt silently).
//!
//! Every constant and layout here is pinned against the Linux `exfat` driver
//! (`fs/exfat/{exfat_raw.h,super.c,misc.c,nls.c,dir.c}`), **fetched, never
//! recalled** — the exFAT spec's own constants are the sort of thing a memory
//! "cleaned up" into a confident wrong answer (e.g. the boot checksum skips
//! `percent_in_use` at offset 112 as well as `vol_flags`, and missing the skip
//! makes every volume with a non-zero percent read as corrupt).
//!
//! Two design decisions are worth stating before the code:
//!
//! - **Names are folded ASCII-only.** exFAT case-insensitivity runs through the
//!   volume's up-case table; the recommended table folds ASCII exactly as
//!   `a-z → A-Z` and everything else to itself. Our hash and comparison do the
//!   same, and the formatter writes a *compressed table that encodes exactly
//!   that* — so hashes match on our own volumes and on any volume carrying the
//!   recommended table, which is every volume Windows or mkfs.exfat has made.
//!   Non-ASCII names are **refused on write** (a name whose hash nobody else
//!   can reproduce is a file nobody else can find), while reading lists full
//!   UTF-16 names. The up-case table itself lives on the volume; we never load
//!   it because we never need more than ASCII folding.
//! - **A stream's `NoFatChain` flag is always cleared by our writer** (use the
//!   FAT), the always-valid choice: the flag claims contiguity, which our
//!   lowest-free-cluster allocator does not guarantee. The reader honours the
//!   flag when *other* writers set it (Windows/Linux mark contiguous chains).

use crate::block::BLOCK_SIZE;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

// --- cluster values (exfat_raw.h) ---------------------------------------

/// End of a cluster chain.
pub const EXFAT_EOF: u32 = 0xFFFF_FFFF;
/// Defective cluster; never allocated.
pub const EXFAT_BAD: u32 = 0xFFFF_FFF7;
/// Free cluster.
pub const EXFAT_FREE: u32 = 0;
/// Highest allocatable cluster (the top 10 values are reserved markers).
pub const EXFAT_MAX_CLUSTER: u32 = 0xFFFF_FFF5;
/// Clusters 0 and 1 are reserved; the data heap starts at 2.
pub const FIRST_CLUSTER: u32 = 2;

// --- directory entry types (exfat_raw.h) --------------------------------
//
// The high bit is InUse; a deleted entry is `type & 0x7F` (the `EXFAT_DELETE`
// mask). The value 0x00 means "end of directory".

/// File / directory primary entry (in use).
pub const TYPE_FILE: u8 = 0x85;
/// Stream-extension secondary entry.
pub const TYPE_STREAM: u8 = 0xC0;
/// File-name secondary entry (15 UTF-16 units each).
pub const TYPE_NAME: u8 = 0xC1;
/// Volume-label primary.
pub const TYPE_VOLUME: u8 = 0x83;
/// Allocation-bitmap primary.
pub const TYPE_BITMAP: u8 = 0x81;
/// Up-case-table primary.
pub const TYPE_UPCASE: u8 = 0x82;
/// Volume-GUID primary (benign; skipped, not required).
pub const TYPE_GUID: u8 = 0xA0;
/// TexFAT padding primary.
pub const TYPE_PADDING: u8 = 0xA1;

/// File attributes (the primary's `attr` field).
pub const ATTR_READONLY: u16 = 0x0001;
pub const ATTR_HIDDEN: u16 = 0x0002;
pub const ATTR_SYSTEM: u16 = 0x0004;
pub const ATTR_VOLUME: u16 = 0x0008;
pub const ATTR_SUBDIR: u16 = 0x0010;
pub const ATTR_ARCHIVE: u16 = 0x0020;

/// Volume flags (`vol_flags` in the boot sector).
pub const VOLUME_DIRTY: u16 = 0x0002;
pub const MEDIA_FAILURE: u16 = 0x0004;
pub const CLEAR_TO_ZERO: u16 = 0x0008;

pub const DENTRY_LEN: usize = 32;
/// UTF-16 units per file-name entry.
pub const NAME_UNITS: usize = 15;
/// Maximum file-name length in UTF-16 units (excl. NUL).
pub const MAX_NAME: usize = 255;
/// Stream `GeneralSecondaryFlags` for an allocation that uses the FAT chain.
/// Bit 0 (AllocationPossible) is **set** — `fsck_exfat` reports a stream whose
/// data has no allocation as "File has no stream allocation" (Linux's
/// `ALLOC_FAT_CHAIN = 0x01`).
pub const FLAGS_FAT_CHAIN: u8 = 0x01;
/// Stream flags claiming a **contiguous** chain (no FAT entries used):
/// Linux's `ALLOC_NO_FAT_CHAIN = 0x03`. A reader must treat *exactly this
/// value* as contiguous and everything else as a FAT chain.
pub const FLAGS_NO_FAT_CHAIN: u8 = 0x03;

/// Whether a stream's flags claim a contiguous (NoFatChain) allocation. This is
/// a value comparison, not a bit test: Linux's `exfat_chain_advance` only skips
/// the FAT walk when `flags == ALLOC_NO_FAT_CHAIN` (0x03), and `0x01` means
/// "FAT chain in use".
pub fn no_fat_chain(flags: u8) -> bool {
    flags == FLAGS_NO_FAT_CHAIN
}

pub fn le16(b: &[u8], o: usize) -> u16 {
    if o + 2 > b.len() {
        return 0;
    }
    u16::from_le_bytes([b[o], b[o + 1]])
}
pub fn le32(b: &[u8], o: usize) -> u32 {
    if o + 4 > b.len() {
        return 0;
    }
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}
pub fn le64(b: &[u8], o: usize) -> u64 {
    if o + 8 > b.len() {
        return 0;
    }
    let mut v = [0u8; 8];
    v.copy_from_slice(&b[o..o + 8]);
    u64::from_le_bytes(v)
}

/// The decoded geometry of an exFAT volume (from its main boot sector).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bpb {
    pub bytes_per_sector: u32,
    pub sect_per_clus_bits: u8,
    pub num_fats: u8,
    /// First FAT sector.
    pub fat_offset: u32,
    /// Sectors per FAT (the copies are equal).
    pub fat_length: u32,
    /// First data-sector (start of the cluster heap).
    pub clu_offset: u32,
    /// Number of data clusters (clusters 0 and 1 are reserved).
    pub clu_count: u32,
    /// First cluster of the root directory (always ≥ 2).
    pub root_cluster: u32,
    pub vol_flags: u16,
}

impl Bpb {
    pub fn sect_per_clus(&self) -> u32 {
        1 << self.sect_per_clus_bits
    }
    pub fn cluster_bytes(&self) -> u32 {
        self.bytes_per_sector << self.sect_per_clus_bits
    }
    /// Entries per directory cluster.
    pub fn dentries_per_clu(&self) -> u32 {
        self.cluster_bytes() / DENTRY_LEN as u32
    }
    /// Bytes of one FAT copy.
    pub fn fat_bytes(&self) -> usize {
        (self.fat_length as usize) * (self.bytes_per_sector as usize)
    }
    /// Total clusters including the two reserved ones.
    pub fn num_clusters(&self) -> u32 {
        self.cluster_entry_space()
    }
    /// The cluster index space is `clu_count + 2` (0 and 1 reserved), and the
    /// FAT must hold one entry for each.
    pub fn cluster_entry_space(&self) -> u32 {
        self.clu_count + 2
    }
    /// First sector of data cluster `n`. Clusters 0/1 have no sectors; the
    /// mapping is the exFAT one — data cluster 2 starts at `clu_offset`.
    pub fn cluster_sector(&self, n: u32) -> Option<u32> {
        if n < FIRST_CLUSTER || n >= self.cluster_entry_space() {
            return None;
        }
        Some(self.clu_offset + (n - FIRST_CLUSTER) * self.sect_per_clus())
    }
    /// `(sector, byte-within-sector)` of cluster `n`'s entry in FAT `copy`.
    pub fn fat_entry_location(&self, copy: u8, n: u32) -> Option<(u32, u32)> {
        if copy >= self.num_fats {
            return None;
        }
        let byte = n as usize * 4;
        if byte as u32 >= self.fat_length * self.bytes_per_sector {
            return None; // entry beyond this FAT's extent
        }
        let base = self.fat_offset + copy as u32 * self.fat_length;
        Some((base + byte as u32 / self.bytes_per_sector, (byte as u32) % self.bytes_per_sector))
    }
}

/// Parse + validate a main boot sector. `None` on anything a hostile or
/// foreign partition could plausibly produce — geometry that later divides.
///
/// Mirrors `exfat_read_boot_sector`'s checks exactly (signature, OEM name, the
/// `must_be_zero` region that keeps a FAT volume from mounting, FAT count,
/// sector/cluster sizes, and the two consistency inequalities). We additionally
/// require 512-byte sectors, since `BlockDevice` is 512-byte addressed.
pub fn parse_boot(sector: &[u8]) -> Option<Bpb> {
    if sector.len() < BLOCK_SIZE {
        return None;
    }
    if le16(sector, 510) != 0xAA55 {
        return None;
    }
    if &sector[3..11] != b"EXFAT   " {
        return None;
    }
    // All 53 bytes of the legacy-BPB area must be zero; this is what keeps a
    // FAT volume (whose BPB lives there) from being misread as exFAT.
    if sector[11..64].iter().any(|&b| b != 0) {
        return None;
    }
    let num_fats = sector[110];
    if num_fats != 1 && num_fats != 2 {
        return None;
    }
    let sect_size_bits = sector[108];
    if !(9..=12).contains(&sect_size_bits) {
        return None;
    }
    // The spec caps clusters at 32 MiB (`sect_per_clus_bits ≤ 25 - sect_size_bits`).
    let spc_bits = sector[109];
    if spc_bits > 25 - sect_size_bits {
        return None;
    }
    // We only mount 512-byte-sector volumes (the block layer is fixed there).
    if sect_size_bits != 9 {
        return None;
    }
    let bytes_per_sector = 1u32 << sect_size_bits;
    let fat_length = le32(sector, 84);
    let clu_offset = le32(sector, 88);
    let clu_count = le32(sector, 92);
    if fat_length == 0 || clu_count == 0 || clu_count > EXFAT_MAX_CLUSTER {
        return None;
    }
    // Consistency inequalities, exactly as the driver checks them.
    if (fat_length as u64) * (bytes_per_sector as u64) < (clu_count as u64 + 2) * 4 {
        return None; // FAT too small to index every cluster
    }
    let fat_start = le32(sector, 80) as u64;
    if (clu_offset as u64) < fat_start + (fat_length as u64) * (num_fats as u64) {
        return None; // cluster heap overlapping the FAT(s)
    }
    let root_cluster = le32(sector, 96);
    if root_cluster < FIRST_CLUSTER || root_cluster >= clu_count + FIRST_CLUSTER {
        return None;
    }
    Some(Bpb {
        bytes_per_sector,
        sect_per_clus_bits: spc_bits,
        num_fats,
        fat_offset: le32(sector, 80),
        fat_length,
        clu_offset,
        clu_count,
        root_cluster,
        vol_flags: le16(sector, 106),
    })
}

// --- checksums (misc.c) --------------------------------------------------
//
// Both are "rotate right by 1, add byte". The directory-set checksum skips the
// primary's bytes 2-3 (where the checksum itself lives); the boot checksum
// skips byte 106/107 (`vol_flags`) and 112 (`percent_in_use`) — the fields the
// driver updates in place without rewriting the checksum sector.

fn rot16(c: u16) -> u16 {
    c.rotate_right(1)
}
fn rot32(c: u32) -> u32 {
    c.rotate_right(1)
}

/// 16-bit checksum over `data`; `skip_set_checksum` skips bytes 2-3 (the
/// SetChecksum field of the primary file entry).
pub fn chksum16(data: &[u8], skip_set_checksum: bool) -> u16 {
    let mut c = 0u16;
    for (i, &b) in data.iter().enumerate() {
        if skip_set_checksum && (i == 2 || i == 3) {
            continue;
        }
        c = rot16(c).wrapping_add(b as u16);
    }
    c
}

/// 32-bit checksum over `data`, skipping byte indices 106/107/112 (only
/// meaningful for the main boot sector, but applied uniformly as the driver
/// does).
pub fn chksum32(data: &[u8]) -> u32 {
    let mut c = 0u32;
    for (i, &b) in data.iter().enumerate() {
        if i == 106 || i == 107 || i == 112 {
            continue;
        }
        c = rot32(c).wrapping_add(b as u32);
    }
    c
}

/// Boot checksum over the 11-sector boot region (main + extended + OEM +
/// reserved). Sectors 1..10 are checksummed whole; the skips land in sector 0
/// only because those byte offsets are inside it.
pub fn boot_checksum(region: &[u8]) -> u32 {
    debug_assert!(region.len() >= 11 * BLOCK_SIZE);
    let mut c = 0u32;
    for (i, &b) in region[..11 * BLOCK_SIZE].iter().enumerate() {
        if i == 106 || i == 107 || i == 112 {
            continue;
        }
        c = rot32(c).wrapping_add(b as u32);
    }
    c
}

/// Checksum of a directory entry set: the primary (bytes 2-3 skipped, its own
/// checksum field) followed by every secondary.
pub fn entry_set_checksum(entries: &[&[u8]]) -> u16 {
    let mut c = 0u16;
    for (i, e) in entries.iter().enumerate() {
        let skip = i == 0;
        for (j, &b) in e.iter().enumerate() {
            if skip && (j == 2 || j == 3) {
                continue;
            }
            c = rot16(c).wrapping_add(b as u16);
        }
    }
    c
}

// --- names ---------------------------------------------------------------

/// exFAT case folding, reduced to ASCII (the recommended up-case table folds
/// `a-z` to `A-Z` and every other code point to itself).
pub fn upcase(c: u16) -> u16 {
    if (0x61..=0x7a).contains(&c) {
        c - 0x20
    } else {
        c
    }
}

/// The `NameHash` of a UTF-16 name: the 16-bit checksum over the **up-cased**
/// name's little-endian bytes (`exfat_calc_chksum16(upname, len << 1, 0,
/// CS_DEFAULT)` in `nls.c`).
pub fn name_hash(name: &[u16]) -> u16 {
    let mut h = 0u16;
    for &c in name {
        let c = upcase(c);
        h = rot16(h).wrapping_add((c & 0xff) as u16);
        h = rot16(h).wrapping_add((c >> 8) as u16);
    }
    h
}

/// Case-insensitive name comparison under the same ASCII fold.
pub fn name_eq(a: &[u16], b: &[u16]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(&x, &y)| upcase(x) == upcase(y))
}

/// Characters exFAT forbids in a name (the `bad_uni_chars` list).
const FORBIDDEN: &[u16] = &[0x22, 0x2a, 0x2f, 0x3a, 0x3c, 0x3e, 0x3f, 0x5c, 0x7c];

/// Whether a UTF-16 name is writable: 1..=255 units, no control chars, none of
/// the forbidden characters, no trailing space or dot, and **ASCII only** (a
/// non-ASCII name whose hash no other implementation reproduces is a file
/// nobody else can find — refused rather than stored that way).
pub fn name_valid(name: &[u16]) -> bool {
    let n = name.len();
    if n == 0 || n > MAX_NAME {
        return false;
    }
    for (i, &c) in name.iter().enumerate() {
        if c >= 0x80 || c < 0x20 {
            return false;
        }
        if FORBIDDEN.contains(&c) {
            return false;
        }
        // A trailing space or period is legal to store but unreachable on
        // Windows; refuse rather than create a name that round-trips badly.
        if i + 1 == n && (c == b' ' as u16 || c == b'.' as u16) {
            return false;
        }
    }
    true
}

/// Decode a UTF-16 string (surrogate pairs handled) into a `String`.
pub fn str_from_utf16(units: &[u16]) -> String {
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

/// Encode a `str` as UTF-16 code units.
pub fn utf16_from_str(s: &str) -> Vec<u16> {
    s.encode_utf16().collect()
}

// --- timestamps ----------------------------------------------------------

/// Pack a Unix timestamp into exFAT's `(time, date, centisec)` fields. The
/// stored time is **UTC** (tz byte = `EXFAT_TZ_VALID`, i.e. offset 0), which is
/// the same choice the driver makes on a `sys_tz`-off system. Out-of-range
/// timestamps clamp to 1980-01-01 00:00 (the format's minimum).
pub fn pack_time(unix: i64) -> (u16, u16, u8) {
    let (y, mo, d, h, mi, s, _) = crate::clock::civil_from_unix(unix);
    if !(1980..=2107).contains(&y) {
        return (0, 0x21, 0); // 1980-01-01 00:00:00
    }
    let time = ((h as u16) << 11) | ((mi as u16) << 5) | ((s as u16 >> 1) as u16);
    let date = (((y - 1980) as u16) << 9) | ((mo as u16) << 5) | (d as u16);
    // The centisec field carries the odd-second + sub-second part.
    let cs = ((s & 1) as u8) * 100;
    (time, date, cs)
}

/// Unpack an exFAT timestamp into Unix seconds, honouring the UTC-offset
/// field. Mirrors `exfat_get_entry_time`: the packed fields are read as if UTC,
/// then the offset field moves them (valid offsets are `0x00-0x3F` east,
/// `0x40-0x7F` west).
pub fn unpack_time(time: u16, date: u16, cs: u8, tz: u8) -> i64 {
    let y = 1980 + (date >> 9) as i64;
    let mo = ((date >> 5) & 0x0f) as i64;
    let d = (date & 0x1f) as i64;
    let h = (time >> 11) as i64;
    let mi = ((time >> 5) & 0x3f) as i64;
    let s = ((time & 0x1f) << 1) as i64 + (cs / 100) as i64;
    let mut secs = crate::clock::unix_from_civil(y, mo, d, h, mi, s);
    if tz & 0x80 != 0 {
        let off = (tz & 0x7f) as i64;
        let mins = if off <= 0x3f { -off } else { 0x80 - off };
        secs += mins * 900; // 900 s per 15-min unit
    }
    secs
}

// --- FAT + bitmap arithmetic ---------------------------------------------

/// Cluster `n`'s FAT entry out of a loaded FAT image.
pub fn fat_entry(fat: &[u8], n: u32) -> Option<u32> {
    let off = n as usize * 4;
    if off + 4 > fat.len() {
        return None;
    }
    Some(le32(fat, off))
}

/// Pick `count` free clusters (FAT entry == 0), lowest first, skipping
/// defective ones. `None` rather than a partial allocation, so a caller never
/// commits a half chain.
pub fn find_free_clusters(fat: &[u8], cluster_entry_space: u32, count: usize) -> Option<Vec<u32>> {
    if count == 0 {
        return Some(Vec::new());
    }
    let mut out = Vec::with_capacity(count);
    for n in FIRST_CLUSTER..cluster_entry_space {
        match fat_entry(fat, n) {
            Some(v) if v == EXFAT_FREE => out.push(n),
            Some(v) if v == EXFAT_BAD => {}
            Some(_) => {}
            None => break, // FAT image shorter than the geometry claims
        }
        if out.len() == count {
            return Some(out);
        }
    }
    None
}

/// Count free clusters (for `free_bytes` reporting).
pub fn count_free(fat: &[u8], cluster_entry_space: u32) -> u32 {
    let mut n = 0;
    for c in FIRST_CLUSTER..cluster_entry_space {
        if fat_entry(fat, c) == Some(EXFAT_FREE) {
            n += 1;
        }
    }
    n
}

// --- directory entry builders ---------------------------------------------

/// A 32-byte primary file/directory entry. `tz` byte is `EXFAT_TZ_VALID` (UTC).
pub fn file_entry(attr: u16, num_ext: u8, unix: i64) -> [u8; DENTRY_LEN] {
    let (t, d, cs) = pack_time(unix);
    let mut e = [0u8; DENTRY_LEN];
    e[0] = TYPE_FILE;
    e[1] = num_ext;
    e[4..6].copy_from_slice(&attr.to_le_bytes());
    for off in [8usize, 12, 16] {
        e[off..off + 2].copy_from_slice(&t.to_le_bytes());
        e[off + 2..off + 4].copy_from_slice(&d.to_le_bytes());
    }
    e[20] = cs;
    e[21] = cs;
    e[22] = 0x80; // create_tz: offset valid, 0
    e[23] = 0x80; // modify_tz
    e[24] = 0x80; // access_tz
    e
}

/// A stream-extension secondary entry. `flags` is [`FLAGS_FAT_CHAIN`] for our
/// own files; read back from disk for foreign ones.
pub fn stream_entry(name_len: u8, hash: u16, start_clu: u32, size: u64, valid: u64, flags: u8) -> [u8; DENTRY_LEN] {
    let mut e = [0u8; DENTRY_LEN];
    e[0] = TYPE_STREAM;
    e[1] = flags;
    e[3] = name_len;
    e[4..6].copy_from_slice(&hash.to_le_bytes());
    e[8..16].copy_from_slice(&valid.to_le_bytes());
    e[20..24].copy_from_slice(&start_clu.to_le_bytes());
    e[24..32].copy_from_slice(&size.to_le_bytes());
    e
}

/// One file-name secondary entry carrying `name[offset .. offset+15]`, padded
/// with NULs (the reader stops at the first `0x0000`).
pub fn name_entry(name: &[u16], offset: usize) -> [u8; DENTRY_LEN] {
    let mut e = [0u8; DENTRY_LEN];
    e[0] = TYPE_NAME;
    for (k, slot) in e[2..].chunks_exact_mut(2).enumerate() {
        let c = name.get(offset + k).copied().unwrap_or(0);
        slot.copy_from_slice(&c.to_le_bytes());
    }
    e
}

/// Number of file-name entries a name needs (1..=17 for 1..=255 units).
pub fn name_entry_count(name_len: usize) -> usize {
    name_len.div_ceil(NAME_UNITS)
}

/// The compressed up-case table our formatter writes: identity runs encoded as
/// `0xFFFF, count`, explicit mappings as the value. Covers the whole 0x10000
/// code space and folds ASCII exactly, so every implementation using it agrees
/// with our name hashes.
pub fn ascii_upcase_table() -> Vec<u16> {
    let mut t = Vec::new();
    t.push(0xFFFF);
    t.push(0x61); // indices 0..=0x60 are identity (0x61 entries)
    for c in b'A'..=b'Z' {
        t.push(c as u16); // index 0x61+i → 0x41+i
    }
    // Indices 0x7B..=0xFFFF are identity: 65536 - 0x7B(123) = 65413 = 0xFF85.
    // Getting this count wrong stops the decode at 0xFFF8 and the table is
    // rejected by Linux's loader ("failed to load upcase table").
    t.push(0xFFFF);
    t.push(0xFF85);
    t
}

/// Decode a compressed up-case table into `table[index] = upcased[index]`
/// (identity where unmapped). Mirrors `exfat_load_upcase_table`.
pub fn decode_upcase_table(raw: &[u16], out: &mut [u16]) -> bool {
    let mut index: u32 = 0;
    let mut skip: u32 = 0;
    for &uni in raw {
        let uni = uni as u32;
        if skip != 0 {
            index += uni;
            skip = 0;
            continue;
        }
        if uni == index {
            index += 1;
        } else if uni == 0xFFFF {
            skip = 1;
        } else {
            if index < out.len() as u32 {
                out[index as usize] = uni as u16;
            }
            index += 1;
        }
    }
    index >= 0xFFFF
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a plausible 512-byte boot sector for a `(total, spc_bits)`
    /// volume, with geometry computed the same way the formatter does (FAT
    /// sized to the cluster count, cluster heap aligned to the cluster size).
    fn boot(total: u64, spc_bits: u8, num_fats: u8) -> Vec<u8> {
        let spc = 1u64 << spc_bits;
        let mut fat_len = ((total / spc).max(1) + 2) * 4 / BLOCK_SIZE as u64 + 1;
        let mut clu_off = round_up(24 + num_fats as u64 * fat_len, spc);
        let mut clu_count = if clu_off < total { (total - clu_off) / spc } else { 0 };
        let need = ((clu_count + 2) * 4).div_ceil(BLOCK_SIZE as u64);
        if need > fat_len {
            fat_len = need;
            clu_off = round_up(24 + num_fats as u64 * fat_len, spc);
            clu_count = if clu_off < total { (total - clu_off) / spc } else { 0 };
        }
        let mut s = vec![0u8; BLOCK_SIZE];
        s[0] = 0xEB;
        s[1] = 0x76;
        s[2] = 0x90;
        s[3..11].copy_from_slice(b"EXFAT   ");
        s[64..72].copy_from_slice(&0u64.to_le_bytes()); // partition offset
        s[72..80].copy_from_slice(&total.to_le_bytes()); // volume length
        s[80..84].copy_from_slice(&24u32.to_le_bytes());
        s[84..88].copy_from_slice(&(fat_len as u32).to_le_bytes());
        s[88..92].copy_from_slice(&(clu_off as u32).to_le_bytes());
        s[92..96].copy_from_slice(&(clu_count as u32).to_le_bytes());
        s[96..100].copy_from_slice(&2u32.to_le_bytes()); // root cluster
        s[100..104].copy_from_slice(&0x12345678u32.to_le_bytes());
        s[104..106].copy_from_slice(&0x0100u16.to_le_bytes()); // fs_revision
        s[106..108].copy_from_slice(&0u16.to_le_bytes()); // vol_flags
        s[108] = 9; // 512-byte sectors
        s[109] = spc_bits;
        s[110] = num_fats;
        s[111] = 0x80; // drive select
        s[112] = 0; // percent in use
        s[510] = 0x55;
        s[511] = 0xAA;
        s
    }

    fn round_up(v: u64, align: u64) -> u64 {
        v.div_ceil(align) * align
    }

    #[test_case]
    fn parses_an_exfat_boot_sector() {
        let b = parse_boot(&boot(204_800, 3, 1)).unwrap();
        assert_eq!(b.bytes_per_sector, 512);
        assert_eq!(b.sect_per_clus(), 8);
        assert_eq!(b.cluster_bytes(), 4096);
        assert_eq!(b.num_fats, 1);
        assert_eq!(b.root_cluster, 2);
        // Geometry is self-consistent: the FAT covers every cluster, and the
        // cluster heap sits after the FAT(s) on a cluster boundary.
        assert!(b.fat_entry_location(0, b.cluster_entry_space() - 1).is_some());
        assert!(b.cluster_sector(b.cluster_entry_space() - 1).is_some());
        assert!(b.clu_offset as u64 >= 24 + b.fat_length as u64 * b.num_fats as u64);
        assert_eq!(b.clu_offset % b.sect_per_clus(), 0);
        // Cluster 2 is the first data cluster, by definition.
        assert_eq!(b.cluster_sector(2), Some(b.clu_offset));
        // Cluster 0/1 are reserved and must not map into the metadata area.
        assert_eq!(b.cluster_sector(0), None);
        assert_eq!(b.cluster_sector(1), None);
        assert_eq!(b.cluster_sector(b.cluster_entry_space()), None);
    }

    #[test_case]
    fn fat_entry_location_covers_both_copies() {
        let b = parse_boot(&boot(204_800, 3, 2)).unwrap();
        let (s0, o0) = b.fat_entry_location(0, 3).unwrap();
        assert_eq!(o0, 12);
        let (s1, _) = b.fat_entry_location(1, 3).unwrap();
        assert_eq!(s1, s0 + b.fat_length); // second copy one FAT further in
        assert_eq!(b.fat_entry_location(2, 3), None); // no third copy
        assert_eq!(b.fat_entry_location(0, u32::MAX), None); // past the FAT
    }

    #[test_case]
    fn rejects_foreign_or_hostile_boot_sectors() {
        assert!(parse_boot(&[0u8; BLOCK_SIZE]).is_none());
        assert!(parse_boot(&[0u8; 100]).is_none()); // short
        // A FAT BPB has a nonzero legacy-BPB region -> must be refused, or a
        // FAT volume would mount as exFAT and read garbage.
        let mut s = boot(204_800, 3, 1);
        s[36] = 0x29;
        assert!(parse_boot(&s).is_none());
        // num_fats must be 1 or 2.
        let mut s = boot(204_800, 3, 0);
        assert!(parse_boot(&s).is_none());
        // Sector size not 512 is refused by us (block layer is 512-fixed).
        let mut s = boot(204_800, 3, 1);
        s[108] = 10;
        assert!(parse_boot(&s).is_none());
        // Cluster size past the 32 MiB cap.
        let mut s = boot(204_800, 17, 1);
        assert!(parse_boot(&s).is_none());
        // FAT too small for the cluster count.
        let mut s = boot(204_800, 3, 1);
        s[84..88].copy_from_slice(&1u32.to_le_bytes());
        assert!(parse_boot(&s).is_none());
        // Cluster heap overlapping the FAT(s).
        let mut s = boot(204_800, 3, 1);
        s[88..92].copy_from_slice(&25u32.to_le_bytes());
        assert!(parse_boot(&s).is_none());
    }

    #[test_case]
    fn boot_checksum_skips_vol_flags_and_percent() {
        let mut region = vec![0u8; 11 * BLOCK_SIZE];
        region[0..BLOCK_SIZE].copy_from_slice(&boot(204_800, 3, 1));
        let c1 = boot_checksum(&region);
        // The fields the driver updates in place must not change the checksum.
        region[106] = 0x02;
        region[107] = 0x00;
        region[112] = 0x64;
        let c2 = boot_checksum(&region);
        assert_eq!(c1, c2);
        // Any other byte does.
        region[100] ^= 0xff;
        assert_ne!(c1, boot_checksum(&region));
    }

    #[test_case]
    fn chksum16_matches_reference_hand_computation() {
        // Hand-computed: checksum over {1,2,3,4} with no skips.
        let mut c = 0u16;
        for &b in &[1u8, 2, 3, 4] {
            c = c.rotate_right(1).wrapping_add(b as u16);
        }
        assert_eq!(chksum16(&[1, 2, 3, 4], false), c);
        // The primary skips its own checksum bytes (2,3), so a checksum written
        // into them must be idempotent.
        let mut entries = vec![file_entry(ATTR_ARCHIVE, 2, 0), stream_entry(1, 0, 0, 0, 0, FLAGS_FAT_CHAIN)];
        let cs = entry_set_checksum(&entries.iter().map(|e| &e[..]).collect::<Vec<_>>());
        entries[0][2..4].copy_from_slice(&cs.to_le_bytes());
        assert_eq!(entry_set_checksum(&entries.iter().map(|e| &e[..]).collect::<Vec<_>>()), cs);
    }

    #[test_case]
    fn name_hash_is_over_the_upcased_name() {
        // "A" = 0x41, hashed as u16 LE bytes {0x41, 0x00} with the rotate-add:
        //   0 + 0x41 = 0x41; rot16(0x41) = 0x8020; + 0x00 = 0x8020.
        assert_eq!(name_hash(&[0x41]), 0x8020);
        // Folding makes lowercase and uppercase collide.
        assert_eq!(name_hash(&[b'a' as u16, b'b' as u16]), name_hash(&[b'A' as u16, b'B' as u16]));
        // A non-ASCII unit folds to itself.
        assert_eq!(name_hash(&[0xE9]), name_hash(&[0xE9]));
    }

    #[test_case]
    fn upcase_folds_ascii_only() {
        assert_eq!(upcase(b'a' as u16), b'A' as u16);
        assert_eq!(upcase(b'z' as u16), b'Z' as u16);
        assert_eq!(upcase(b'A' as u16), b'A' as u16);
        assert_eq!(upcase(0xE9), 0xE9); // é
        assert_eq!(upcase(0xDF), 0xDF); // ß (the recommended table maps this to
                                        // Ÿ, a quirk we deliberately do not share)
        assert_eq!(name_eq(&[b'a' as u16, 0xE9], &[b'A' as u16, 0xE9]), true);
        assert_eq!(name_eq(&[b'a' as u16], &[b'B' as u16]), false);
    }

    #[test_case]
    fn name_validation_refuses_what_would_not_round_trip() {
        assert!(name_valid(&utf16_from_str("hello.txt")));
        assert!(name_valid(&utf16_from_str("CAPS and spaces")));
        assert!(!name_valid(&[]));
        assert!(!name_valid(&utf16_from_str(".")));
        assert!(!name_valid(&utf16_from_str("..")));
        assert!(!name_valid(&utf16_from_str("has/slash")));
        assert!(!name_valid(&utf16_from_str("has:colon")));
        assert!(!name_valid(&utf16_from_str("trailing ")));
        assert!(!name_valid(&utf16_from_str("trailing.")));
        assert!(!name_valid(&utf16_from_str("ctrl\x01")));
        assert!(!name_valid(&utf16_from_str("café.txt"))); // non-ASCII refused
        // 255 units is the max; 256 is not.
        assert!(name_valid(&vec![b'a' as u16; 255]));
        assert!(!name_valid(&vec![b'a' as u16; 256]));
    }

    #[test_case]
    fn utf16_round_trips_including_surrogates() {
        assert_eq!(str_from_utf16(&utf16_from_str("hello world")), "hello world");
        // U+1F600 😀 needs a surrogate pair.
        let units = utf16_from_str("a\u{1f600}b");
        assert_eq!(str_from_utf16(&units), "a\u{1f600}b");
        assert_eq!(name_hash(&units), name_hash(&units));
    }

    #[test_case]
    fn timestamps_round_trip_utc() {
        for unix in [315_532_800i64, 1_600_000_000, 1_500_000_000, 1_700_000_000] {
            let (t, d, cs) = pack_time(unix);
            // The odd second rides in the centisec field.
            let got = unpack_time(t, d, cs, 0x80);
            let want = unix - (unix % 2);
            assert!((got - want).abs() <= 1, "{unix}: got {got}, want ~{want}");
        }
        // Out of range clamps to 1980-01-01 rather than wrapping.
        let (t, d, cs) = pack_time(-100);
        assert_eq!(unpack_time(t, d, cs, 0x80), crate::clock::unix_from_civil(1980, 1, 1, 0, 0, 0));
        // A west-of-UTC offset shifts the reading by the offset. The offset
        // field needs the VALID bit (0x80): a raw 0x60 means "local time with
        // no stored offset", which is not adjusted.
        let (t, d, cs) = pack_time(1_600_000_000);
        let local = unpack_time(t, d, cs, 0xE0); // -8h (0x80-0x60=0x20 -> 8h)
        assert_eq!(local, 1_600_000_000 + 8 * 3600);
        let no_adjust = unpack_time(t, d, cs, 0x60);
        assert_eq!(no_adjust, 1_600_000_000);
    }

    #[test_case]
    fn ascii_upcase_table_decodes_to_ascii_fold() {
        let table = ascii_upcase_table();
        // 0xFFFF,0x61 + 26 mappings + 0xFFFF,0xFF85.
        assert_eq!(table.len(), 30);
        let mut full = [0u16; 0x10000];
        assert!(decode_upcase_table(&table, &mut full));
        // Explicit mappings land; everything else is 0, meaning "identity"
        // (the driver's `vol_utbl[a] ? vol_utbl[a] : a` convention).
        assert_eq!(full[b'a' as usize], b'A' as u16);
        assert_eq!(full[b'z' as usize], b'Z' as u16);
        assert_eq!(full[b'A' as usize], 0); // 'A' folds to itself
        assert_eq!(full[0x60], 0);
        assert_eq!(full[0x7b], 0);
        assert_eq!(full[0xE9], 0);
        assert_eq!(full[0xFFFF], 0);
    }

    #[test_case]
    fn fat_helpers_find_free_and_count() {
        // 10 data clusters, 2 and 3 used, 4 bad, rest free.
        let mut fat = vec![0u8; 12 * 4];
        fat[2 * 4..3 * 4].copy_from_slice(&EXFAT_EOF.to_le_bytes());
        fat[3 * 4..4 * 4].copy_from_slice(&7u32.to_le_bytes());
        fat[4 * 4..5 * 4].copy_from_slice(&EXFAT_BAD.to_le_bytes());
        assert_eq!(find_free_clusters(&fat, 12, 3), Some(vec![5, 6, 7]));
        assert_eq!(find_free_clusters(&fat, 12, 0), Some(vec![]));
        let full = vec![0xffu8; 12 * 4];
        assert_eq!(find_free_clusters(&full, 12, 1), None);
        assert_eq!(count_free(&full, 12), 0);
        assert_eq!(count_free(&fat, 12), 7);
        assert!(fat_entry(&fat, 2) == Some(EXFAT_EOF));
        assert!(fat_entry(&fat, 99).is_none()); // bounds-checked
    }

    #[test_case]
    fn entry_builders_layout_bytes_correctly() {
        let fe = file_entry(ATTR_SUBDIR, 2, 0);
        assert_eq!(fe[0], TYPE_FILE);
        assert_eq!(fe[1], 2); // stream + one name entry
        assert_eq!(u16::from_le_bytes([fe[4], fe[5]]), ATTR_SUBDIR);
        let se = stream_entry(3, 0x1234, 9, 12345, 12345, FLAGS_FAT_CHAIN);
        assert_eq!(se[0], TYPE_STREAM);
        assert_eq!(se[1], FLAGS_FAT_CHAIN);
        assert_eq!(se[3], 3);
        assert_eq!(u16::from_le_bytes([se[4], se[5]]), 0x1234);
        assert_eq!(u32::from_le_bytes([se[20], se[21], se[22], se[23]]), 9);
        assert_eq!(u64::from_le_bytes(se[24..32].try_into().unwrap()), 12345);
        let ne = name_entry(&utf16_from_str("abc"), 0);
        assert_eq!(ne[0], TYPE_NAME);
        assert_eq!(u16::from_le_bytes([ne[2], ne[3]]), b'a' as u16);
        assert_eq!(u16::from_le_bytes([ne[6], ne[7]]), b'c' as u16);
        assert_eq!(u16::from_le_bytes([ne[8], ne[9]]), 0); // first NUL pad slot
        assert_eq!(name_entry_count(1), 1);
        assert_eq!(name_entry_count(15), 1);
        assert_eq!(name_entry_count(16), 2);
        assert_eq!(name_entry_count(255), 17);
    }
}
