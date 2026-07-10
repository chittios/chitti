//! **WOFF1 → SFNT converter** — unwraps a `wOFF` web-font container back into
//! the plain TTF/OTF byte layout that [`crate::font_ttf`] (fontdue) parses.
//! Pure and allocation-only: every read is bounds-checked and malformed input
//! returns `Err`, never panics. Compressed tables are RFC 1950 zlib streams,
//! inflated via [`crate::image::inflate::zlib_decompress`] (which also
//! verifies the Adler-32 trailer).
//!
//! **WOFF2 is detected but not supported** ([`is_woff2`]): its payload is
//! Brotli-compressed with a transformed `glyf`/`loca`, i.e. a whole second
//! decompressor plus a glyph re-encoder — out of scope until a no_std Brotli
//! lands. Callers should log "woff2 unsupported (brotli)" and fall through to
//! the next `src` in a CSS `@font-face` list.

use crate::image::inflate::zlib_decompress;
use alloc::borrow::Cow;
use alloc::vec::Vec;

/// WOFF1 signature `wOFF`.
const WOFF1_SIG: u32 = 0x774F_4646;
/// WOFF2 signature `wOF2`.
const WOFF2_SIG: u32 = 0x774F_4632;
/// Fixed WOFF1 header size (bytes).
const WOFF_HDR: usize = 44;
/// WOFF1 table-directory entry size (bytes).
const WOFF_DIR_ENTRY: usize = 20;
/// SFNT table-directory entry size (bytes).
const SFNT_DIR_ENTRY: usize = 16;
/// Sanity cap on the rebuilt SFNT (a corrupt origLength must not eat the
/// heap; real web fonts are single-digit MB).
const SFNT_MAX: usize = 64 << 20;
/// Sanity cap on numTables (also keeps `numTables * 16` inside u16 for the
/// binary-search fields; real fonts have a few dozen tables).
const MAX_TABLES: usize = 4095;

/// True if `data` starts with the WOFF1 signature `wOFF`.
pub fn is_woff(data: &[u8]) -> bool {
    data.len() >= 4 && data[..4] == WOFF1_SIG.to_be_bytes()
}

/// True if `data` starts with the WOFF2 signature `wOF2`. WOFF2 is **not**
/// convertible here (Brotli + transformed glyf) — see the module docs.
pub fn is_woff2(data: &[u8]) -> bool {
    data.len() >= 4 && data[..4] == WOFF2_SIG.to_be_bytes()
}

/// Convert a WOFF1 container into the equivalent SFNT (TTF/OTF) byte stream.
///
/// Emits the 12-byte offset table (flavor copied from the WOFF header,
/// binary-search fields recomputed), a directory in the WOFF's own entry
/// order (the spec requires it tag-ascending already), and each table's
/// original bytes 4-byte aligned. `head.checkSumAdjustment` is left as
/// stored (fontdue does not verify it). Every offset/length is validated;
/// malformed input returns `Err`.
pub fn woff_to_sfnt(woff: &[u8]) -> Result<Vec<u8>, &'static str> {
    if is_woff2(woff) {
        return Err("woff2 unsupported (brotli)");
    }
    if !is_woff(woff) {
        return Err("woff: bad signature");
    }
    if woff.len() < WOFF_HDR {
        return Err("woff: truncated header");
    }
    let flavor = be32(woff, 4)?;
    let num_tables = be16(woff, 12)? as usize;
    if num_tables == 0 || num_tables > MAX_TABLES {
        return Err("woff: bad table count");
    }
    let dir_end = WOFF_HDR + num_tables * WOFF_DIR_ENTRY;
    if dir_end > woff.len() {
        return Err("woff: truncated directory");
    }

    // Pass 1: parse + decode every table before emitting anything, so a bad
    // entry anywhere rejects the whole font.
    let mut tables: Vec<(u32, u32, Cow<[u8]>)> = Vec::with_capacity(num_tables);
    let mut total = 12usize + num_tables * SFNT_DIR_ENTRY;
    for i in 0..num_tables {
        let e = WOFF_HDR + i * WOFF_DIR_ENTRY;
        let tag = be32(woff, e)?;
        let off = be32(woff, e + 4)? as usize;
        let comp_len = be32(woff, e + 8)? as usize;
        let orig_len = be32(woff, e + 12)? as usize;
        let checksum = be32(woff, e + 16)?;
        if comp_len > orig_len {
            return Err("woff: compLength exceeds origLength");
        }
        let end = off.checked_add(comp_len).ok_or("woff: table offset overflow")?;
        if end > woff.len() {
            return Err("woff: table out of bounds");
        }
        total = total
            .checked_add(pad4(orig_len))
            .ok_or("woff: sfnt too large")?;
        if total > SFNT_MAX {
            return Err("woff: sfnt too large");
        }
        // compLength < origLength ⇒ zlib stream; equal ⇒ stored raw.
        let data: Cow<[u8]> = if comp_len < orig_len {
            let raw =
                zlib_decompress(&woff[off..end]).map_err(|_| "woff: table inflate failed")?;
            if raw.len() != orig_len {
                return Err("woff: inflated length mismatch");
            }
            Cow::Owned(raw)
        } else {
            Cow::Borrowed(&woff[off..end])
        };
        tables.push((tag, checksum, data));
    }

    // 12-byte offset table: flavor, numTables, then the binary-search fields
    // (searchRange = maxPow2(numTables)*16, entrySelector = log2 of that
    // power, rangeShift = numTables*16 - searchRange).
    let mut out: Vec<u8> = Vec::with_capacity(total);
    let mut pow2 = 1usize;
    let mut entry_selector = 0u16;
    while pow2 * 2 <= num_tables {
        pow2 *= 2;
        entry_selector += 1;
    }
    let search_range = (pow2 * SFNT_DIR_ENTRY) as u16;
    out.extend_from_slice(&flavor.to_be_bytes());
    out.extend_from_slice(&(num_tables as u16).to_be_bytes());
    out.extend_from_slice(&search_range.to_be_bytes());
    out.extend_from_slice(&entry_selector.to_be_bytes());
    out.extend_from_slice(&((num_tables * SFNT_DIR_ENTRY) as u16 - search_range).to_be_bytes());

    // Directory (WOFF entry order preserved), offsets 4-byte aligned.
    let mut data_off = 12 + num_tables * SFNT_DIR_ENTRY;
    for (tag, checksum, data) in &tables {
        out.extend_from_slice(&tag.to_be_bytes());
        out.extend_from_slice(&checksum.to_be_bytes());
        out.extend_from_slice(&(data_off as u32).to_be_bytes());
        out.extend_from_slice(&(data.len() as u32).to_be_bytes());
        data_off += pad4(data.len());
    }
    for (_, _, data) in &tables {
        out.extend_from_slice(data);
        out.resize(pad4(out.len()), 0);
    }
    Ok(out)
}

/// Round `n` up to the next multiple of 4 (SFNT tables are long-aligned).
fn pad4(n: usize) -> usize {
    (n + 3) & !3
}

fn be16(d: &[u8], off: usize) -> Result<u16, &'static str> {
    if off.checked_add(2).is_none_or(|end| end > d.len()) {
        return Err("woff: truncated");
    }
    Ok(u16::from_be_bytes([d[off], d[off + 1]]))
}

fn be32(d: &[u8], off: usize) -> Result<u32, &'static str> {
    if off.checked_add(4).is_none_or(|end| end > d.len()) {
        return Err("woff: truncated");
    }
    Ok(u32::from_be_bytes([d[off], d[off + 1], d[off + 2], d[off + 3]]))
}

#[cfg(test)]
mod tests {
    use super::*;

    static GEIST_TTF: &[u8] = include_bytes!("../../assets/GeistMono-Regular.ttf");

    /// `zlib.compress(b"CHITTIOSFONTTABLE!!" * 8, 9)` — 152 bytes → 30
    /// (compressed at test-authoring time; the kernel has no deflate encoder).
    const ZLIB_FIXTURE: &[u8] = &[
        0x78, 0xda, 0x73, 0xf6, 0xf0, 0x0c, 0x09, 0xf1, 0xf4, 0x0f, 0x76, 0xf3, 0xf7, 0x0b, 0x09,
        0x71, 0x74, 0xf2, 0x71, 0x55, 0x54, 0x74, 0x1e, 0x0c, 0x42, 0x00, 0xba, 0x24, 0x2a, 0x41,
    ];
    const ZLIB_ORIG_LEN: usize = 152;
    const TEST_TAG: u32 = 0x7465_7374; // "test"

    fn rd32(d: &[u8], off: usize) -> u32 {
        u32::from_be_bytes([d[off], d[off + 1], d[off + 2], d[off + 3]])
    }

    /// Build a WOFF1 container from `(tag, checksum, payload, origLength)`
    /// entries; a payload shorter than origLength is a zlib stream, equal is
    /// stored.
    fn build_woff(flavor: u32, tables: &[(u32, u32, &[u8], u32)]) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(&WOFF1_SIG.to_be_bytes());
        v.extend_from_slice(&flavor.to_be_bytes());
        v.extend_from_slice(&0u32.to_be_bytes()); // length (unused here)
        v.extend_from_slice(&(tables.len() as u16).to_be_bytes());
        v.extend_from_slice(&0u16.to_be_bytes()); // reserved
        v.extend_from_slice(&0u32.to_be_bytes()); // totalSfntSize (unused)
        v.extend_from_slice(&1u16.to_be_bytes()); // majorVersion
        v.extend_from_slice(&0u16.to_be_bytes()); // minorVersion
        v.extend_from_slice(&[0u8; 20]); // meta/priv offsets+lengths
        assert_eq!(v.len(), WOFF_HDR);
        let mut off = WOFF_HDR + tables.len() * WOFF_DIR_ENTRY;
        for (tag, sum, comp, orig) in tables {
            v.extend_from_slice(&tag.to_be_bytes());
            v.extend_from_slice(&(off as u32).to_be_bytes());
            v.extend_from_slice(&(comp.len() as u32).to_be_bytes());
            v.extend_from_slice(&orig.to_be_bytes());
            v.extend_from_slice(&sum.to_be_bytes());
            off += super::pad4(comp.len());
        }
        for (_, _, comp, _) in tables {
            v.extend_from_slice(comp);
            v.resize(super::pad4(v.len()), 0);
        }
        v
    }

    /// Wrap the embedded Geist SFNT into a WOFF, all tables stored.
    fn wrap_geist() -> Vec<u8> {
        let flavor = rd32(GEIST_TTF, 0);
        let n = u16::from_be_bytes([GEIST_TTF[4], GEIST_TTF[5]]) as usize;
        let mut tabs: Vec<(u32, u32, &[u8], u32)> = Vec::new();
        for i in 0..n {
            let e = 12 + i * 16;
            let tag = rd32(GEIST_TTF, e);
            let sum = rd32(GEIST_TTF, e + 4);
            let off = rd32(GEIST_TTF, e + 8) as usize;
            let len = rd32(GEIST_TTF, e + 12) as usize;
            tabs.push((tag, sum, &GEIST_TTF[off..off + len], len as u32));
        }
        build_woff(flavor, &tabs)
    }

    #[test_case]
    fn woff_stored_roundtrip_parses_with_fontdue() {
        let woff = wrap_geist();
        assert!(is_woff(&woff));
        let sfnt = woff_to_sfnt(&woff).expect("convert");
        // Directory matches the original face table-for-table, byte-for-byte.
        let n = u16::from_be_bytes([sfnt[4], sfnt[5]]) as usize;
        assert_eq!(n, u16::from_be_bytes([GEIST_TTF[4], GEIST_TTF[5]]) as usize);
        for i in 0..n {
            let e = 12 + i * 16;
            assert_eq!(rd32(&sfnt, e), rd32(GEIST_TTF, e), "tag order");
            let off = rd32(&sfnt, e + 8) as usize;
            let len = rd32(&sfnt, e + 12) as usize;
            let ooff = rd32(GEIST_TTF, e + 8) as usize;
            let olen = rd32(GEIST_TTF, e + 12) as usize;
            assert_eq!(len, olen);
            assert_eq!(&sfnt[off..off + len], &GEIST_TTF[ooff..ooff + olen]);
        }
        assert!(fontdue::Font::from_bytes(sfnt.as_slice(), fontdue::FontSettings::default()).is_ok());
    }

    #[test_case]
    fn woff_zlib_table_inflates() {
        let woff = build_woff(0x0001_0000, &[(TEST_TAG, 0, ZLIB_FIXTURE, ZLIB_ORIG_LEN as u32)]);
        let sfnt = woff_to_sfnt(&woff).expect("convert");
        assert_eq!(rd32(&sfnt, 12), TEST_TAG);
        let off = rd32(&sfnt, 12 + 8) as usize;
        let len = rd32(&sfnt, 12 + 12) as usize;
        assert_eq!(len, ZLIB_ORIG_LEN);
        let mut want = Vec::new();
        for _ in 0..8 {
            want.extend_from_slice(b"CHITTIOSFONTTABLE!!");
        }
        assert_eq!(&sfnt[off..off + len], want.as_slice());
    }

    #[test_case]
    fn woff_malformed_inputs_rejected() {
        // Bad signature / truncated header.
        assert!(woff_to_sfnt(b"nope").is_err());
        assert!(woff_to_sfnt(&[0x77, 0x4F, 0x46, 0x46, 0, 0]).is_err());
        let good = build_woff(0x0001_0000, &[(TEST_TAG, 7, b"0123", 4)]);
        assert!(woff_to_sfnt(&good).is_ok());
        // Truncated directory.
        assert!(woff_to_sfnt(&good[..50]).is_err());
        // compLength > origLength (origLength dir field patched to 1 < 4).
        let mut bad = good.clone();
        bad[WOFF_HDR + 12..WOFF_HDR + 16].copy_from_slice(&1u32.to_be_bytes());
        assert!(woff_to_sfnt(&bad).is_err());
        // Table offset out of bounds.
        let mut oob = good.clone();
        oob[WOFF_HDR + 4..WOFF_HDR + 8].copy_from_slice(&0xffff_0000u32.to_be_bytes());
        assert!(woff_to_sfnt(&oob).is_err());
        // Zero tables.
        let mut none = good.clone();
        none[12..14].copy_from_slice(&0u16.to_be_bytes());
        assert!(woff_to_sfnt(&none).is_err());
        // Zlib table whose claimed origLength disagrees with the inflated size.
        let lied = build_woff(
            0x0001_0000,
            &[(TEST_TAG, 0, ZLIB_FIXTURE, (ZLIB_ORIG_LEN + 1) as u32)],
        );
        assert!(woff_to_sfnt(&lied).is_err());
    }

    #[test_case]
    fn woff_signature_detection() {
        assert!(is_woff(b"wOFF\x00\x01\x00\x00"));
        assert!(!is_woff(b"wOF2aaaa"));
        assert!(!is_woff(b"wO"));
        assert!(is_woff2(b"wOF2aaaa"));
        assert!(!is_woff2(b"wOFF\x00\x01\x00\x00"));
        assert_eq!(woff_to_sfnt(b"wOF2aaaa").unwrap_err(), "woff2 unsupported (brotli)");
    }
}
