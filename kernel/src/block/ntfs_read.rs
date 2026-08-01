//! **NTFS read-only** volume access (internal disks and USB MSC).
//!
//! Scope is intentionally narrow and fail-closed:
//! - Parse the boot sector / BPB
//! - Read MFT records (with multi-sector transfer fixups)
//! - Walk `$I30` directory indexes (INDEX_ROOT + INDEX_ALLOCATION)
//! - Read unnamed `$DATA` (resident or non-resident runlist)
//!
//! **Writes are not implemented.** A wrong NTFS mutation silently corrupts the
//! volume; mount policy keeps NTFS read-only until a verified writer lands.
//!
//! Pure parse helpers are unit-tested; end-to-end needs a real NTFS image.

use crate::block::{BlockDevice, BLOCK_SIZE};
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

fn le16(b: &[u8], o: usize) -> u16 {
    if o + 2 > b.len() {
        return 0;
    }
    u16::from_le_bytes([b[o], b[o + 1]])
}
fn le32(b: &[u8], o: usize) -> u32 {
    if o + 4 > b.len() {
        return 0;
    }
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}
fn le64(b: &[u8], o: usize) -> u64 {
    if o + 8 > b.len() {
        return 0;
    }
    let mut v = [0u8; 8];
    v.copy_from_slice(&b[o..o + 8]);
    u64::from_le_bytes(v)
}

/// Signed little-endian integer of `n` bytes (1–8), used by runlists.
pub fn sle(bytes: &[u8]) -> i64 {
    if bytes.is_empty() || bytes.len() > 8 {
        return 0;
    }
    let mut v = [0u8; 8];
    v[..bytes.len()].copy_from_slice(bytes);
    // Sign-extend from the high bit of the last supplied byte.
    if bytes[bytes.len() - 1] & 0x80 != 0 {
        for b in &mut v[bytes.len()..] {
            *b = 0xff;
        }
    }
    i64::from_le_bytes(v)
}

/// Boot-sector geometry.
#[derive(Clone, Copy, Debug)]
pub struct NtfsBoot {
    pub bytes_per_sector: u16,
    pub sectors_per_cluster: u8,
    pub mft_lcn: u64,
    pub mft_record_size: u32,
    pub index_record_size: u32,
    pub cluster_bytes: u64,
}

/// Parse an NTFS boot sector (first sector of the volume).
pub fn parse_boot(sector: &[u8]) -> Option<NtfsBoot> {
    if sector.len() < 0x54 {
        return None;
    }
    if &sector[3..11] != b"NTFS    " {
        return None;
    }
    let bps = le16(sector, 0x0b);
    if bps as usize != BLOCK_SIZE {
        return None;
    }
    let spc = sector[0x0d];
    if spc == 0 {
        return None;
    }
    let mft_lcn = le64(sector, 0x30);
    if mft_lcn == 0 {
        return None;
    }
    // Clusters per MFT record: positive = clusters; negative = 2^(-n) bytes.
    let cpm = sector[0x40] as i8;
    let mft_record_size = if cpm > 0 {
        (cpm as u32) * (spc as u32) * (bps as u32)
    } else if cpm < 0 {
        1u32 << ((-cpm) as u32)
    } else {
        return None;
    };
    if mft_record_size < 512 || mft_record_size > 4096 {
        return None;
    }
    let cpi = sector[0x44] as i8;
    let index_record_size = if cpi > 0 {
        (cpi as u32) * (spc as u32) * (bps as u32)
    } else if cpi < 0 {
        1u32 << ((-cpi) as u32)
    } else {
        4096
    };
    Some(NtfsBoot {
        bytes_per_sector: bps,
        sectors_per_cluster: spc,
        mft_lcn,
        mft_record_size,
        index_record_size,
        cluster_bytes: bps as u64 * spc as u64,
    })
}

/// Apply the multi-sector transfer (USA) fixup in place. Returns false if the
/// record is corrupt or too short.
pub fn apply_fixup(rec: &mut [u8]) -> bool {
    if rec.len() < 0x30 || (&rec[0..4] != b"FILE" && &rec[0..4] != b"INDX") {
        return false;
    }
    let usa_off = le16(rec, 0x04) as usize;
    let usa_count = le16(rec, 0x06) as usize; // includes the USN word
    if usa_count < 2 || usa_off + usa_count * 2 > rec.len() {
        return false;
    }
    let usn = [rec[usa_off], rec[usa_off + 1]];
    for i in 1..usa_count {
        let sector_end = i * BLOCK_SIZE;
        if sector_end > rec.len() || sector_end < 2 {
            return false;
        }
        let pos = sector_end - 2;
        if rec[pos] != usn[0] || rec[pos + 1] != usn[1] {
            return false;
        }
        let src = usa_off + i * 2;
        rec[pos] = rec[src];
        rec[pos + 1] = rec[src + 1];
    }
    true
}

/// One decoded data run: `lcn` is absolute; `None` means a sparse hole.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DataRun {
    pub lcn: Option<u64>,
    pub clusters: u64,
}

/// Decode a non-resident runlist. `runs` starts at the mapping-pairs offset.
pub fn decode_runlist(runs: &[u8]) -> Option<Vec<DataRun>> {
    let mut out = Vec::new();
    let mut i = 0usize;
    let mut cur_lcn: i64 = 0;
    while i < runs.len() {
        let header = runs[i];
        i += 1;
        if header == 0 {
            break;
        }
        let len_size = (header & 0x0f) as usize;
        let off_size = (header >> 4) as usize;
        if len_size == 0 || i + len_size + off_size > runs.len() {
            return None;
        }
        let clusters = sle(&runs[i..i + len_size]) as u64;
        i += len_size;
        if off_size == 0 {
            // Sparse.
            out.push(DataRun {
                lcn: None,
                clusters,
            });
        } else {
            let delta = sle(&runs[i..i + off_size]);
            i += off_size;
            cur_lcn = cur_lcn.checked_add(delta)?;
            if cur_lcn < 0 {
                return None;
            }
            out.push(DataRun {
                lcn: Some(cur_lcn as u64),
                clusters,
            });
        }
    }
    Some(out)
}

/// Attribute type codes we care about.
pub const AT_STANDARD_INFO: u32 = 0x10;
pub const AT_FILE_NAME: u32 = 0x30;
pub const AT_DATA: u32 = 0x80;
pub const AT_INDEX_ROOT: u32 = 0x90;
pub const AT_INDEX_ALLOCATION: u32 = 0xa0;
pub const AT_END: u32 = 0xffff_ffff;

/// Slice one attribute header; returns (type, total_len, non_resident, body_range).
#[derive(Clone, Copy, Debug)]
pub struct AttrView<'a> {
    pub ty: u32,
    pub non_resident: bool,
    pub name: &'a [u8], // UTF-16LE
    /// For resident: value bytes. For non-resident: full attribute body from
    /// the start of the attribute (for runlist/offset fields).
    pub raw: &'a [u8],
    pub value: &'a [u8],
    pub data_size: u64,
    pub runlist: &'a [u8],
}

/// Iterate attributes inside a fixed-up FILE record.
pub fn iter_attrs(rec: &[u8]) -> impl Iterator<Item = AttrView<'_>> {
    let first = if rec.len() >= 0x16 {
        le16(rec, 0x14) as usize
    } else {
        rec.len()
    };
    AttrIter { rec, off: first }
}

struct AttrIter<'a> {
    rec: &'a [u8],
    off: usize,
}

impl<'a> Iterator for AttrIter<'a> {
    type Item = AttrView<'a>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.off + 8 > self.rec.len() {
            return None;
        }
        let ty = le32(self.rec, self.off);
        if ty == AT_END || ty == 0 {
            return None;
        }
        let len = le32(self.rec, self.off + 4) as usize;
        if len < 16 || self.off + len > self.rec.len() {
            return None;
        }
        let raw = &self.rec[self.off..self.off + len];
        let non_res = raw[8] != 0;
        let name_len = raw[9] as usize;
        let name_off = le16(raw, 10) as usize;
        let name = if name_len > 0 && name_off + name_len * 2 <= raw.len() {
            &raw[name_off..name_off + name_len * 2]
        } else {
            &[][..]
        };
        let (value, data_size, runlist) = if !non_res {
            if raw.len() < 0x18 {
                self.off += len;
                return None;
            }
            let vlen = le32(raw, 0x10) as usize;
            let voff = le16(raw, 0x14) as usize;
            if voff + vlen > raw.len() {
                self.off += len;
                return None;
            }
            (&raw[voff..voff + vlen], vlen as u64, &[][..])
        } else {
            if raw.len() < 0x40 {
                self.off += len;
                return None;
            }
            let data_size = le64(raw, 0x30);
            let run_off = le16(raw, 0x20) as usize;
            let runlist = if run_off < raw.len() {
                &raw[run_off..]
            } else {
                &[][..]
            };
            (&[][..], data_size, runlist)
        };
        self.off += len;
        Some(AttrView {
            ty,
            non_resident: non_res,
            name,
            raw,
            value,
            data_size,
            runlist,
        })
    }
}

/// Decode a UTF-16LE name (no surrogates needed for typical ASCII/BMP names).
pub fn utf16le_name(bytes: &[u8]) -> String {
    let mut s = String::new();
    let mut i = 0;
    while i + 1 < bytes.len() {
        let c = u16::from_le_bytes([bytes[i], bytes[i + 1]]);
        i += 2;
        if c == 0 {
            break;
        }
        if let Some(ch) = char::from_u32(c as u32) {
            s.push(ch);
        }
    }
    s
}

/// One directory entry from an $I30 index.
#[derive(Clone, Debug)]
pub struct DirEnt {
    pub name: String,
    pub mft_ref: u64,
    pub is_dir: bool,
}

/// Parse INDEX_ROOT / INDEX_ALLOCATION node body (starts at the INDEX_HEADER).
pub fn parse_index_entries(body: &[u8]) -> Vec<DirEnt> {
    let mut out = Vec::new();
    if body.len() < 16 {
        return out;
    }
    // INDEX_HEADER: entries_offset@0, index_length@4, allocated@8, flags@12
    let ent_off = le32(body, 0) as usize;
    let index_len = le32(body, 4) as usize;
    if ent_off >= body.len() || index_len > body.len() {
        return out;
    }
    let end = index_len.min(body.len());
    let mut off = ent_off;
    while off + 16 <= end {
        let entry_len = le16(body, off + 8) as usize;
        let key_len = le16(body, off + 10) as usize;
        let flags = le32(body, off + 12);
        if entry_len < 16 || off + entry_len > body.len() {
            break;
        }
        // Last entry has no FILE_NAME key (flags bit 1).
        if flags & 2 == 0 && key_len >= 0x42 {
            let key = &body[off + 16..off + 16 + key_len];
            // FILE_NAME: parent@0 … flags@56, name_len@64, name_type@65, name@66
            let name_len = key[64] as usize;
            let want = name_len * 2;
            if 66 + want <= key.len() {
                let name = utf16le_name(&key[66..66 + want]);
                if !name.is_empty() && name != "." && name != ".." {
                    let mft_ref = le64(body, off) & 0x0000_ffff_ffff_ffff;
                    let fn_flags = le32(key, 56);
                    let is_dir = fn_flags & 0x1000_0000 != 0;
                    // name_type: 0=POSIX, 1=Win32, 2=DOS, 3=Win32+DOS.
                    // Prefer longer (Win32) over short DOS for the same MFT ref.
                    let ntype = key[65];
                    if let Some(prev) = out.iter_mut().find(|e| e.mft_ref == mft_ref) {
                        if ntype != 2 && (prev.name.len() < name.len() || ntype == 1 || ntype == 3)
                        {
                            prev.name = name;
                            prev.is_dir = is_dir;
                        }
                    } else {
                        out.push(DirEnt {
                            name,
                            mft_ref,
                            is_dir,
                        });
                    }
                }
            }
        }
        if flags & 2 != 0 {
            break; // last entry
        }
        if entry_len == 0 {
            break;
        }
        off += entry_len;
    }
    out
}

/// Live NTFS volume handle (read-only).
pub struct NtfsReader<'d, D: BlockDevice> {
    dev: &'d mut D,
    boot: NtfsBoot,
}

impl<'d, D: BlockDevice> NtfsReader<'d, D> {
    pub fn open(dev: &'d mut D) -> Option<Self> {
        let mut sec = [0u8; BLOCK_SIZE];
        dev.read_block(0, &mut sec).ok()?;
        let boot = parse_boot(&sec)?;
        Some(NtfsReader { dev, boot })
    }

    fn cluster_lba(&self, lcn: u64) -> u64 {
        lcn * self.boot.sectors_per_cluster as u64
    }

    fn read_clusters(&mut self, lcn: u64, n: u64, out: &mut [u8]) -> Option<()> {
        let start = self.cluster_lba(lcn);
        let sectors = n * self.boot.sectors_per_cluster as u64;
        let bytes = sectors as usize * BLOCK_SIZE;
        if out.len() < bytes {
            return None;
        }
        self.dev.read_blocks(start, &mut out[..bytes]).ok()?;
        Some(())
    }

    /// Read one MFT record by number (0 = $MFT, 5 = root).
    pub fn read_mft_record(&mut self, num: u64) -> Option<Vec<u8>> {
        let rec_size = self.boot.mft_record_size as usize;
        // MFT itself starts at mft_lcn; each record is mft_record_size bytes.
        // For records beyond the first cluster of $MFT we still use the
        // contiguous layout of $MFT's first extent — enough for root + small
        // volumes. Large volumes need $DATA runlist of record 0; upgrade path.
        let byte_off = num * rec_size as u64;
        let lba = self.cluster_lba(self.boot.mft_lcn) + byte_off / BLOCK_SIZE as u64;
        let mut buf = vec![0u8; rec_size];
        // May span multiple sectors.
        let nsec = rec_size / BLOCK_SIZE;
        self.dev.read_blocks(lba, &mut buf[..nsec * BLOCK_SIZE]).ok()?;
        if !apply_fixup(&mut buf) {
            return None;
        }
        // In-use flag bit 0 of flags at 0x16.
        if le16(&buf, 0x16) & 1 == 0 {
            return None;
        }
        Some(buf)
    }

    /// Unnamed $DATA content for an MFT record.
    pub fn read_data(&mut self, mft_num: u64) -> Option<Vec<u8>> {
        let rec = self.read_mft_record(mft_num)?;
        for a in iter_attrs(&rec) {
            if a.ty != AT_DATA || !a.name.is_empty() {
                continue;
            }
            if !a.non_resident {
                return Some(a.value.to_vec());
            }
            let runs = decode_runlist(a.runlist)?;
            let mut out = vec![0u8; a.data_size as usize];
            let mut done = 0usize;
            let cb = self.boot.cluster_bytes as usize;
            for run in runs {
                let take_clusters = run.clusters as usize;
                for k in 0..take_clusters {
                    if done >= out.len() {
                        return Some(out);
                    }
                    let mut clus = vec![0u8; cb];
                    if let Some(lcn) = run.lcn {
                        self.read_clusters(lcn + k as u64, 1, &mut clus)?;
                    }
                    // else sparse: already zero
                    let n = (out.len() - done).min(cb);
                    out[done..done + n].copy_from_slice(&clus[..n]);
                    done += n;
                }
            }
            out.truncate(a.data_size as usize);
            return Some(out);
        }
        None
    }

    /// List the root directory (MFT #5).
    pub fn list_root(&mut self) -> Option<Vec<DirEnt>> {
        self.list_dir(5)
    }

    pub fn list_dir(&mut self, mft_num: u64) -> Option<Vec<DirEnt>> {
        let rec = self.read_mft_record(mft_num)?;
        let mut ents = Vec::new();
        // INDEX_ROOT ($I30)
        for a in iter_attrs(&rec) {
            if a.ty != AT_INDEX_ROOT {
                continue;
            }
            // Resident value: attr type, collation, size, clusters, then INDEX_HEADER
            if a.value.len() < 16 + 16 {
                continue;
            }
            // Skip 16-byte index root header → INDEX_HEADER
            ents.extend(parse_index_entries(&a.value[16..]));
        }
        // INDEX_ALLOCATION ($I30) — large directories
        for a in iter_attrs(&rec) {
            if a.ty != AT_INDEX_ALLOCATION || !a.non_resident {
                continue;
            }
            let runs = decode_runlist(a.runlist)?;
            let irec = self.boot.index_record_size as usize;
            for run in runs {
                let Some(lcn0) = run.lcn else { continue };
                for k in 0..run.clusters {
                    let mut buf = vec![0u8; irec.max(self.boot.cluster_bytes as usize)];
                    let nclus = (irec as u64).div_ceil(self.boot.cluster_bytes).max(1);
                    // Index records may be multi-cluster; read one record's clusters.
                    self.read_clusters(lcn0 + k * nclus, nclus, &mut buf)?;
                    if buf.len() < irec {
                        continue;
                    }
                    buf.truncate(irec);
                    if !apply_fixup(&mut buf) {
                        continue;
                    }
                    // INDX: header then INDEX_HEADER at 0x18 (typical)
                    let ih_off = 0x18usize;
                    if ih_off < buf.len() {
                        ents.extend(parse_index_entries(&buf[ih_off..]));
                    }
                }
            }
        }
        // Dedup by mft_ref preferring longer names
        let mut dedup: Vec<DirEnt> = Vec::new();
        for e in ents {
            if let Some(p) = dedup.iter_mut().find(|x| x.mft_ref == e.mft_ref) {
                if e.name.len() > p.name.len() {
                    *p = e;
                }
            } else {
                dedup.push(e);
            }
        }
        Some(dedup)
    }

    /// Resolve a path relative to root (`foo/bar.txt`) and read file bytes.
    pub fn read_file(&mut self, path: &str) -> Option<Vec<u8>> {
        let parts: Vec<&str> = path
            .split(|c| c == '/' || c == '\\')
            .filter(|s| !s.is_empty() && *s != ".")
            .collect();
        if parts.is_empty() {
            return None;
        }
        let mut mft = 5u64;
        for (i, part) in parts.iter().enumerate() {
            let ents = self.list_dir(mft)?;
            let ent = ents
                .iter()
                .find(|e| e.name.eq_ignore_ascii_case(part))?;
            if i + 1 == parts.len() {
                if ent.is_dir {
                    return None;
                }
                return self.read_data(ent.mft_ref);
            }
            if !ent.is_dir {
                return None;
            }
            mft = ent.mft_ref;
        }
        None
    }

    /// Directory listing for a relative path ("" or "/" = root).
    pub fn readdir(&mut self, path: &str) -> Option<Vec<(String, u64, bool)>> {
        let parts: Vec<&str> = path
            .split(|c| c == '/' || c == '\\')
            .filter(|s| !s.is_empty() && *s != ".")
            .collect();
        let mut mft = 5u64;
        for part in &parts {
            let ents = self.list_dir(mft)?;
            let ent = ents
                .iter()
                .find(|e| e.name.eq_ignore_ascii_case(part))?;
            if !ent.is_dir {
                return None;
            }
            mft = ent.mft_ref;
        }
        let ents = self.list_dir(mft)?;
        Some(
            ents.into_iter()
                .map(|e| (e.name, 0u64, e.is_dir))
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn parse_boot_requires_ntfs_oem() {
        let mut s = [0u8; 512];
        s[3..11].copy_from_slice(b"NTFS    ");
        s[0x0b..0x0d].copy_from_slice(&512u16.to_le_bytes());
        s[0x0d] = 8; // 4K clusters
        s[0x30..0x38].copy_from_slice(&4u64.to_le_bytes()); // MFT at LCN 4
        s[0x40] = (-10i8) as u8; // 2^10 = 1024-byte MFT records
        s[0x44] = (-12i8) as u8; // 4096-byte index
        let b = parse_boot(&s).expect("boot");
        assert_eq!(b.mft_lcn, 4);
        assert_eq!(b.mft_record_size, 1024);
        assert_eq!(b.cluster_bytes, 4096);
    }

    #[test_case]
    fn sle_sign_extends() {
        assert_eq!(sle(&[0x01]), 1);
        assert_eq!(sle(&[0xff]), -1);
        assert_eq!(sle(&[0x00, 0x01]), 256);
        assert_eq!(sle(&[0xff, 0xff]), -1);
    }

    #[test_case]
    fn decode_runlist_sparse_and_data() {
        // header 0x11: 1-byte length, 1-byte offset; length=2, offset=+5
        // then sparse 0x01 length=3 offset empty; then end 0
        let runs = [0x11, 0x02, 0x05, 0x01, 0x03, 0x00];
        let d = decode_runlist(&runs).unwrap();
        assert_eq!(d.len(), 2);
        assert_eq!(d[0].clusters, 2);
        assert_eq!(d[0].lcn, Some(5));
        assert_eq!(d[1].clusters, 3);
        assert_eq!(d[1].lcn, None);
    }

    #[test_case]
    fn apply_fixup_restores_sector_tails() {
        // Minimal 1024-byte FILE record with 2 sectors of USA.
        let mut rec = vec![0u8; 1024];
        rec[0..4].copy_from_slice(b"FILE");
        // usa_off = 0x30, usa_count = 3 (USN + 2 sector words)
        rec[0x04..0x06].copy_from_slice(&0x30u16.to_le_bytes());
        rec[0x06..0x08].copy_from_slice(&3u16.to_le_bytes());
        // USN = 0xAB 0xCD
        rec[0x30] = 0xab;
        rec[0x31] = 0xcd;
        // Original words stored in USA array
        rec[0x32] = 0x11;
        rec[0x33] = 0x22;
        rec[0x34] = 0x33;
        rec[0x35] = 0x44;
        // Sector ends currently hold USN
        rec[510] = 0xab;
        rec[511] = 0xcd;
        rec[1022] = 0xab;
        rec[1023] = 0xcd;
        assert!(apply_fixup(&mut rec));
        assert_eq!(rec[510], 0x11);
        assert_eq!(rec[511], 0x22);
        assert_eq!(rec[1022], 0x33);
        assert_eq!(rec[1023], 0x44);
    }

    #[test_case]
    fn utf16le_name_decodes_ascii() {
        let bytes = b"H\0e\0l\0l\0o\0";
        assert_eq!(utf16le_name(bytes), "Hello");
    }
}
