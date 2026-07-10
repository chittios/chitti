//! WOFF2 → SFNT converter (ISO/W3C WOFF2).
//!
//! A WOFF2 font is a Brotli-compressed SFNT whose `glyf`/`loca` tables are
//! stored in a compact **transformed** form. This module reverses that:
//!
//! 1. parse the WOFF2 header + table directory (known-tag table, `UIntBase128`
//!    and `255UInt16` varints);
//! 2. Brotli-decompress the payload (via the vendored `brotli-decompressor`
//!    crate, driven with a small `alloc`-backed allocator so it never touches
//!    the kernel stack the way the crate's bundled no_std wrapper would);
//! 3. reconstruct the standard `glyf` table (triplet-encoded points, 255UInt16
//!    contour/instruction counts, per-glyph bbox bitmap, composites) and the
//!    matching `loca`;
//! 4. reassemble a valid SFNT (offset table + tag-sorted directory + padded
//!    table data + table checksums) that `fontdue` can parse.
//!
//! The `glyf`/`loca` transform-reversal is a faithful port of fontTools'
//! `WOFF2GlyfTable.reconstruct` / `_decodeTriplets` (BSD/MIT — see
//! THIRDPARTY-LICENSES.md). The **hmtx** transform (version 1) is uncommon and
//! not implemented — such a font is rejected cleanly (never mis-decoded), same
//! posture as the WOFF1 path.
//!
//! Everything below the Brotli call is a set of pure functions over byte
//! slices, unit-tested off-hardware against a fonttools-produced fixture (see
//! `tools/woff2diff` and the in-kernel `#[test_case]`s).

use alloc::vec::Vec;

use brotli_decompressor::{
    BrotliDecompressStream, BrotliResult, BrotliState, SliceWrapper, SliceWrapperMut,
};

/// WOFF2 signature `wOF2`.
const WOFF2_SIG: u32 = 0x774F_4632;

/// The 63 well-known table tags addressable by a 6-bit index in a directory
/// entry's flags byte (index 63 = an explicit 4-byte tag follows). Verbatim
/// from the WOFF2 spec (Table 6).
const KNOWN_TAGS: [&[u8; 4]; 63] = [
    b"cmap", b"head", b"hhea", b"hmtx", b"maxp", b"name", b"OS/2", b"post", b"cvt ", b"fpgm",
    b"glyf", b"loca", b"prep", b"CFF ", b"VORG", b"EBDT", b"EBLC", b"gasp", b"hdmx", b"kern",
    b"LTSH", b"PCLT", b"VDMX", b"vhea", b"vmtx", b"BASE", b"GDEF", b"GPOS", b"GSUB", b"EBSC",
    b"JSTF", b"MATH", b"CBDT", b"CBLC", b"COLR", b"CPAL", b"SVG ", b"sbix", b"acnt", b"avar",
    b"bdat", b"bloc", b"bsln", b"cvar", b"fdsc", b"feat", b"fmtx", b"fvar", b"gvar", b"hsty",
    b"just", b"lcar", b"mort", b"morx", b"opbd", b"prop", b"trak", b"Zapf", b"Silf", b"Glat",
    b"Gloc", b"Feat", b"Sill",
];

/// True if `data` starts with the WOFF2 signature `wOF2`.
pub fn is_woff2(data: &[u8]) -> bool {
    data.len() >= 4 && data[..4] == WOFF2_SIG.to_be_bytes()
}

// ---- byte cursor + varints -------------------------------------------------

struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Cursor { data, pos: 0 }
    }
    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }
    fn u8(&mut self) -> Result<u8, &'static str> {
        let b = *self.data.get(self.pos).ok_or("woff2: truncated")?;
        self.pos += 1;
        Ok(b)
    }
    fn u16(&mut self) -> Result<u16, &'static str> {
        Ok(((self.u8()? as u16) << 8) | self.u8()? as u16)
    }
    fn u32(&mut self) -> Result<u32, &'static str> {
        Ok(((self.u16()? as u32) << 16) | self.u16()? as u32)
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], &'static str> {
        let end = self.pos.checked_add(n).ok_or("woff2: overflow")?;
        let s = self.data.get(self.pos..end).ok_or("woff2: truncated")?;
        self.pos = end;
        Ok(s)
    }
    /// `UIntBase128` (1–5 bytes, big-endian, 7 bits/byte).
    fn base128(&mut self) -> Result<u32, &'static str> {
        let mut result: u32 = 0;
        // A leading 0x80 byte (leading zero) is invalid per spec.
        if *self.data.get(self.pos).ok_or("woff2: truncated")? == 0x80 {
            return Err("woff2: base128 leading zero");
        }
        for _ in 0..5 {
            let code = self.u8()?;
            if result & 0xFE00_0000 != 0 {
                return Err("woff2: base128 overflow");
            }
            result = (result << 7) | (code & 0x7F) as u32;
            if code & 0x80 == 0 {
                return Ok(result);
            }
        }
        Err("woff2: base128 too long")
    }
}

/// Read a `255UInt16` value (1–3 bytes) from a byte slice cursor.
fn read_255ushort(data: &[u8], pos: &mut usize) -> Result<u16, &'static str> {
    let code = *data.get(*pos).ok_or("woff2: 255ushort truncated")?;
    *pos += 1;
    match code {
        253 => {
            let hi = *data.get(*pos).ok_or("woff2: 255ushort truncated")? as u16;
            let lo = *data.get(*pos + 1).ok_or("woff2: 255ushort truncated")? as u16;
            *pos += 2;
            Ok((hi << 8) | lo)
        }
        254 => {
            let v = *data.get(*pos).ok_or("woff2: 255ushort truncated")? as u16;
            *pos += 1;
            Ok(v + 506)
        }
        255 => {
            let v = *data.get(*pos).ok_or("woff2: 255ushort truncated")? as u16;
            *pos += 1;
            Ok(v + 253)
        }
        _ => Ok(code as u16),
    }
}

// ---- table directory -------------------------------------------------------

struct TableEntry {
    tag: [u8; 4],
    transformed: bool,
    /// Length of this table's bytes in the decompressed stream (transformLength
    /// if transformed, else origLength).
    stream_length: u32,
    /// Offset of this table's bytes within the decompressed stream.
    stream_offset: usize,
}

// ---- brotli allocator (alloc-backed) ---------------------------------------

/// A `SliceWrapper`-conforming heap buffer backed by `Vec` (the crate's own
/// `HeapAlloc` is gated behind its `std` feature).
struct VecMem<T>(Vec<T>);
impl<T> Default for VecMem<T> {
    fn default() -> Self {
        VecMem(Vec::new())
    }
}
impl<T> SliceWrapper<T> for VecMem<T> {
    fn slice(&self) -> &[T] {
        &self.0
    }
}
impl<T> SliceWrapperMut<T> for VecMem<T> {
    fn slice_mut(&mut self) -> &mut [T] {
        &mut self.0
    }
}

struct VecAlloc;
impl<T: Clone + Default> brotli_decompressor::Allocator<T> for VecAlloc {
    type AllocatedMemory = VecMem<T>;
    fn alloc_cell(&mut self, len: usize) -> VecMem<T> {
        VecMem(alloc::vec![T::default(); len])
    }
    fn free_cell(&mut self, _data: VecMem<T>) {}
}

/// Brotli-decompress `input` into a buffer of exactly `expected` bytes.
fn brotli_decompress(input: &[u8], expected: usize) -> Result<Vec<u8>, &'static str> {
    let mut output = alloc::vec![0u8; expected];
    let mut state = BrotliState::new(VecAlloc, VecAlloc, VecAlloc);
    let mut available_in = input.len();
    let mut input_offset = 0usize;
    let mut available_out = output.len();
    let mut output_offset = 0usize;
    let mut total_out = 0usize;
    let result = BrotliDecompressStream(
        &mut available_in,
        &mut input_offset,
        input,
        &mut available_out,
        &mut output_offset,
        &mut output,
        &mut total_out,
        &mut state,
    );
    match result {
        BrotliResult::ResultSuccess => {
            if output_offset != expected {
                return Err("woff2: brotli size mismatch");
            }
            Ok(output)
        }
        _ => Err("woff2: brotli decode failed"),
    }
}

// ---- glyf / loca reconstruction --------------------------------------------

const HAVE_OVERLAP_SIMPLE_BITMAP: u16 = 0x0001;
const FLAG_OVERLAP_SIMPLE: u8 = 0x40;

// Composite component flags.
const ARG_1_AND_2_ARE_WORDS: u16 = 0x0001;
const WE_HAVE_A_SCALE: u16 = 0x0008;
const MORE_COMPONENTS: u16 = 0x0020;
const WE_HAVE_AN_X_AND_Y_SCALE: u16 = 0x0040;
const WE_HAVE_A_TWO_BY_TWO: u16 = 0x0080;
const WE_HAVE_INSTRUCTIONS: u16 = 0x0100;

fn be_i16(v: i32) -> [u8; 2] {
    (v as i16 as u16).to_be_bytes()
}
fn be_u16(v: u16) -> [u8; 2] {
    v.to_be_bytes()
}

/// Reconstruct the standard `glyf` table + the loca offset array from the
/// transformed WOFF2 `glyf` data. Returns `(glyf_bytes, loca_offsets,
/// index_format)`.
fn reconstruct_glyf(data: &[u8]) -> Result<(Vec<u8>, Vec<u32>, u16), &'static str> {
    let mut c = Cursor::new(data);
    let _version = c.u16()?;
    let option_flags = c.u16()?;
    let num_glyphs = c.u16()? as usize;
    let index_format = c.u16()?;
    let n_contour_size = c.u32()? as usize;
    let n_points_size = c.u32()? as usize;
    let flag_size = c.u32()? as usize;
    let glyph_size = c.u32()? as usize;
    let composite_size = c.u32()? as usize;
    let bbox_size = c.u32()? as usize;
    let instruction_size = c.u32()? as usize;

    let n_contour_stream = c.take(n_contour_size)?;
    let n_points_stream = c.take(n_points_size)?;
    let flag_stream = c.take(flag_size)?;
    let glyph_stream = c.take(glyph_size)?;
    let composite_stream = c.take(composite_size)?;
    let bbox_block = c.take(bbox_size)?;
    let instruction_stream = c.take(instruction_size)?;

    let overlap_bitmap: &[u8] = if option_flags & HAVE_OVERLAP_SIMPLE_BITMAP != 0 {
        c.take((num_glyphs + 7) >> 3)?
    } else {
        &[]
    };

    // bbox stream = bitmap (one bit per glyph) followed by the bbox values.
    let bbox_bitmap_size = ((num_glyphs + 31) >> 5) << 2;
    if bbox_block.len() < bbox_bitmap_size {
        return Err("woff2: bbox bitmap truncated");
    }
    let bbox_bitmap = &bbox_block[..bbox_bitmap_size];
    let mut bbox_pos = bbox_bitmap_size; // cursor into bbox_block for values

    // Cursors into the streams that are consumed sequentially per glyph.
    let mut npoints_pos = 0usize;
    let mut flag_pos = 0usize;
    let mut glyph_pos = 0usize;
    let mut composite_pos = 0usize;
    let mut instr_pos = 0usize;

    let mut glyf: Vec<u8> = Vec::new();
    let mut loca: Vec<u32> = Vec::with_capacity(num_glyphs + 1);
    loca.push(0);

    for gid in 0..num_glyphs {
        let nc_off = gid * 2;
        let n_contours = i16::from_be_bytes([
            *n_contour_stream.get(nc_off).ok_or("woff2: nContour truncated")?,
            *n_contour_stream.get(nc_off + 1).ok_or("woff2: nContour truncated")?,
        ]);

        let have_bbox = bbox_bitmap
            .get(gid >> 3)
            .map(|b| b & (0x80 >> (gid & 7)) != 0)
            .unwrap_or(false);

        let mut glyph_bytes: Vec<u8> = Vec::new();

        if n_contours == 0 {
            // Empty glyph — no data.
        } else if n_contours < 0 {
            // Composite glyph. Components are stored in standard on-disk form.
            let comp_start = composite_pos;
            let mut have_instructions = false;
            loop {
                let flags = u16::from_be_bytes([
                    *composite_stream.get(composite_pos).ok_or("woff2: comp truncated")?,
                    *composite_stream.get(composite_pos + 1).ok_or("woff2: comp truncated")?,
                ]);
                composite_pos += 4; // flags + glyphIndex
                composite_pos += if flags & ARG_1_AND_2_ARE_WORDS != 0 { 4 } else { 2 };
                if flags & WE_HAVE_A_SCALE != 0 {
                    composite_pos += 2;
                } else if flags & WE_HAVE_AN_X_AND_Y_SCALE != 0 {
                    composite_pos += 4;
                } else if flags & WE_HAVE_A_TWO_BY_TWO != 0 {
                    composite_pos += 8;
                }
                if flags & WE_HAVE_INSTRUCTIONS != 0 {
                    have_instructions = true;
                }
                if flags & MORE_COMPONENTS == 0 {
                    break;
                }
            }
            if composite_pos > composite_stream.len() {
                return Err("woff2: comp overrun");
            }
            let comp_bytes = &composite_stream[comp_start..composite_pos];
            if !have_bbox {
                return Err("woff2: composite without bbox");
            }
            // numberOfContours = -1, bbox, components, [instructions]
            glyph_bytes.extend_from_slice(&be_i16(-1));
            glyph_bytes.extend_from_slice(bbox_block.get(bbox_pos..bbox_pos + 8).ok_or("woff2: bbox truncated")?);
            bbox_pos += 8;
            glyph_bytes.extend_from_slice(comp_bytes);
            if have_instructions {
                let ilen = read_255ushort(glyph_stream, &mut glyph_pos)? as usize;
                let instr = instruction_stream
                    .get(instr_pos..instr_pos + ilen)
                    .ok_or("woff2: instr truncated")?;
                instr_pos += ilen;
                glyph_bytes.extend_from_slice(&be_u16(ilen as u16));
                glyph_bytes.extend_from_slice(instr);
            }
        } else {
            // Simple glyph.
            let ncont = n_contours as usize;
            let mut end_pts: Vec<u16> = Vec::with_capacity(ncont);
            let mut end_point: i32 = -1;
            for _ in 0..ncont {
                let n = read_255ushort(n_points_stream, &mut npoints_pos)? as i32;
                end_point += n;
                if end_point < 0 || end_point > 0xFFFF {
                    return Err("woff2: bad point count");
                }
                end_pts.push(end_point as u16);
            }
            let n_points = (end_point + 1) as usize;

            // Flags (one byte per point).
            let flags = flag_stream
                .get(flag_pos..flag_pos + n_points)
                .ok_or("woff2: flagStream truncated")?;
            flag_pos += n_points;

            // Triplet-decode coordinates.
            let (xs, ys, on_curve, consumed) =
                decode_triplets(&flags[..n_points], &glyph_stream[glyph_pos..])?;
            glyph_pos += consumed;

            // Instructions.
            let ilen = read_255ushort(glyph_stream, &mut glyph_pos)? as usize;
            let instr = instruction_stream
                .get(instr_pos..instr_pos + ilen)
                .ok_or("woff2: instr truncated")?;
            instr_pos += ilen;

            // Bounding box: read or recompute.
            let (xmin, ymin, xmax, ymax) = if have_bbox {
                let b = bbox_block.get(bbox_pos..bbox_pos + 8).ok_or("woff2: bbox truncated")?;
                bbox_pos += 8;
                (
                    i16::from_be_bytes([b[0], b[1]]) as i32,
                    i16::from_be_bytes([b[2], b[3]]) as i32,
                    i16::from_be_bytes([b[4], b[5]]) as i32,
                    i16::from_be_bytes([b[6], b[7]]) as i32,
                )
            } else {
                let mut xmn = i32::MAX;
                let mut ymn = i32::MAX;
                let mut xmx = i32::MIN;
                let mut ymx = i32::MIN;
                for i in 0..n_points {
                    xmn = xmn.min(xs[i]);
                    ymn = ymn.min(ys[i]);
                    xmx = xmx.max(xs[i]);
                    ymx = ymx.max(ys[i]);
                }
                if n_points == 0 {
                    (0, 0, 0, 0)
                } else {
                    (xmn, ymn, xmx, ymx)
                }
            };

            // Emit standard simple glyph.
            glyph_bytes.extend_from_slice(&be_i16(n_contours as i32));
            glyph_bytes.extend_from_slice(&be_i16(xmin));
            glyph_bytes.extend_from_slice(&be_i16(ymin));
            glyph_bytes.extend_from_slice(&be_i16(xmax));
            glyph_bytes.extend_from_slice(&be_i16(ymax));
            for &e in &end_pts {
                glyph_bytes.extend_from_slice(&be_u16(e));
            }
            glyph_bytes.extend_from_slice(&be_u16(ilen as u16));
            glyph_bytes.extend_from_slice(instr);
            // Flags: no repeat/compression; ON_CURVE (0x01) + OVERLAP_SIMPLE on
            // the first point when the bitmap says so. Coordinates are emitted
            // as plain 16-bit deltas (no SHORT/SAME bits set).
            let overlap_first = overlap_bitmap
                .get(gid >> 3)
                .map(|b| b & (0x80 >> (gid & 7)) != 0)
                .unwrap_or(false);
            for i in 0..n_points {
                let mut f = if on_curve[i] { 0x01u8 } else { 0x00u8 };
                if i == 0 && overlap_first {
                    f |= FLAG_OVERLAP_SIMPLE;
                }
                glyph_bytes.push(f);
            }
            let mut prev = 0i32;
            for i in 0..n_points {
                glyph_bytes.extend_from_slice(&be_i16(xs[i] - prev));
                prev = xs[i];
            }
            let mut prev = 0i32;
            for i in 0..n_points {
                glyph_bytes.extend_from_slice(&be_i16(ys[i] - prev));
                prev = ys[i];
            }
        }

        // Pad each glyph to an even length so short-loca offsets stay valid.
        if glyph_bytes.len() % 2 != 0 {
            glyph_bytes.push(0);
        }
        glyf.extend_from_slice(&glyph_bytes);
        loca.push(glyf.len() as u32);
    }

    let _ = c.remaining();
    Ok((glyf, loca, index_format))
}

/// Decode the WOFF2 triplet-encoded coordinates for a simple glyph. Returns
/// absolute `(xs, ys, on_curve, bytes_consumed_from_glyph_stream)`. Faithful
/// port of fontTools `_decodeTriplets`.
fn decode_triplets(
    flags: &[u8],
    triplets: &[u8],
) -> Result<(Vec<i32>, Vec<i32>, Vec<bool>, usize), &'static str> {
    let n = flags.len();
    let mut xs = alloc::vec![0i32; n];
    let mut ys = alloc::vec![0i32; n];
    let mut on = alloc::vec![false; n];
    let with_sign = |flag: u8, base: i32| -> i32 {
        if flag & 1 != 0 {
            base
        } else {
            -base
        }
    };
    let mut x = 0i32;
    let mut y = 0i32;
    let mut ti = 0usize;
    let g = |idx: usize| -> Result<i32, &'static str> {
        triplets.get(idx).map(|b| *b as i32).ok_or("woff2: glyphStream truncated")
    };
    for i in 0..n {
        let flag_full = flags[i];
        let on_curve = flag_full >> 7 == 0;
        let flag = flag_full & 0x7F;
        let n_bytes = if flag < 84 {
            1
        } else if flag < 120 {
            2
        } else if flag < 124 {
            3
        } else {
            4
        };
        if ti + n_bytes > triplets.len() {
            return Err("woff2: glyphStream truncated");
        }
        let (dx, dy);
        if flag < 10 {
            dx = 0;
            dy = with_sign(flag, (((flag & 14) as i32) << 7) + g(ti)?);
        } else if flag < 20 {
            dx = with_sign(flag, ((((flag - 10) & 14) as i32) << 7) + g(ti)?);
            dy = 0;
        } else if flag < 84 {
            let b0 = (flag - 20) as i32;
            let b1 = g(ti)?;
            dx = with_sign(flag, 1 + (b0 & 0x30) + (b1 >> 4));
            dy = with_sign(flag >> 1, 1 + ((b0 & 0x0C) << 2) + (b1 & 0x0F));
        } else if flag < 120 {
            let b0 = (flag - 84) as i32;
            dx = with_sign(flag, 1 + ((b0 / 12) << 8) + g(ti)?);
            dy = with_sign(flag >> 1, 1 + (((b0 % 12) >> 2) << 8) + g(ti + 1)?);
        } else if flag < 124 {
            let b1 = g(ti + 1)?;
            dx = with_sign(flag, (g(ti)? << 4) + (b1 >> 4));
            dy = with_sign(flag >> 1, ((b1 & 0x0F) << 8) + g(ti + 2)?);
        } else {
            dx = with_sign(flag, (g(ti)? << 8) + g(ti + 1)?);
            dy = with_sign(flag >> 1, (g(ti + 2)? << 8) + g(ti + 3)?);
        }
        ti += n_bytes;
        x += dx;
        y += dy;
        xs[i] = x;
        ys[i] = y;
        on[i] = on_curve;
    }
    Ok((xs, ys, on, ti))
}

/// Encode the loca table from glyph offsets, in the given index format
/// (0 = short/`u16` halved offsets, 1 = long/`u32`).
fn build_loca(offsets: &[u32], index_format: u16) -> Vec<u8> {
    let mut out = Vec::new();
    if index_format == 0 {
        for &o in offsets {
            out.extend_from_slice(&be_u16((o / 2) as u16));
        }
    } else {
        for &o in offsets {
            out.extend_from_slice(&o.to_be_bytes());
        }
    }
    out
}

// ---- SFNT reassembly -------------------------------------------------------

fn table_checksum(data: &[u8]) -> u32 {
    let mut sum: u32 = 0;
    let mut i = 0;
    while i < data.len() {
        let mut word = [0u8; 4];
        for j in 0..4 {
            if i + j < data.len() {
                word[j] = data[i + j];
            }
        }
        sum = sum.wrapping_add(u32::from_be_bytes(word));
        i += 4;
    }
    sum
}

/// Convert a WOFF2 font to an SFNT (TrueType/OpenType) byte blob.
pub fn woff2_to_sfnt(woff2: &[u8]) -> Result<Vec<u8>, &'static str> {
    if !is_woff2(woff2) {
        return Err("woff2: bad signature");
    }
    let mut c = Cursor::new(woff2);
    let _sig = c.u32()?;
    let flavor = c.u32()?;
    let _length = c.u32()?;
    let num_tables = c.u16()? as usize;
    let _reserved = c.u16()?;
    let _total_sfnt_size = c.u32()?;
    let total_compressed_size = c.u32()? as usize;
    let _major = c.u16()?;
    let _minor = c.u16()?;
    let _meta_offset = c.u32()?;
    let _meta_length = c.u32()?;
    let _meta_orig_length = c.u32()?;
    let _priv_offset = c.u32()?;
    let _priv_length = c.u32()?;

    // Table directory.
    let mut entries: Vec<TableEntry> = Vec::with_capacity(num_tables);
    let mut stream_cursor: usize = 0;
    for _ in 0..num_tables {
        let flags = c.u8()?;
        let tag_index = flags & 0x3F;
        let transform_version = flags >> 6;
        let tag: [u8; 4] = if tag_index == 0x3F {
            let t = c.take(4)?;
            [t[0], t[1], t[2], t[3]]
        } else {
            *KNOWN_TAGS[tag_index as usize]
        };
        let is_glyf_or_loca = &tag == b"glyf" || &tag == b"loca";
        // Transformed iff: glyf/loca with version 0, OR any other table with a
        // non-zero version.
        let transformed = if is_glyf_or_loca {
            transform_version == 0
        } else {
            transform_version != 0
        };
        // Reject transforms we don't implement (only glyf/loca version 0).
        if transformed && !is_glyf_or_loca {
            return Err("woff2: unsupported table transform");
        }
        let orig_length = c.base128()?;
        let stream_length = if transformed { c.base128()? } else { orig_length };
        entries.push(TableEntry {
            tag,
            transformed,
            stream_length,
            stream_offset: stream_cursor,
        });
        stream_cursor += stream_length as usize;
    }

    // Decompress the Brotli payload (exactly the sum of the stream lengths).
    let compressed = c.take(total_compressed_size)?;
    let decompressed = brotli_decompress(compressed, stream_cursor)?;

    // Reconstruct each table's final (SFNT) bytes.
    // glyf must be reconstructed before loca (loca is derived from it).
    let mut glyf_bytes: Option<Vec<u8>> = None;
    let mut loca_bytes: Option<Vec<u8>> = None;
    let mut index_format: u16 = 1;
    for e in &entries {
        if &e.tag == b"glyf" && e.transformed {
            let raw = &decompressed[e.stream_offset..e.stream_offset + e.stream_length as usize];
            let (g, offsets, ifmt) = reconstruct_glyf(raw)?;
            index_format = ifmt;
            loca_bytes = Some(build_loca(&offsets, ifmt));
            glyf_bytes = Some(g);
        }
    }

    struct OutTable {
        tag: [u8; 4],
        data: Vec<u8>,
    }
    let mut out_tables: Vec<OutTable> = Vec::with_capacity(num_tables);
    for e in &entries {
        let data: Vec<u8> = if &e.tag == b"glyf" && e.transformed {
            glyf_bytes.take().ok_or("woff2: glyf missing")?
        } else if &e.tag == b"loca" && e.transformed {
            loca_bytes.take().ok_or("woff2: loca before glyf")?
        } else {
            decompressed
                .get(e.stream_offset..e.stream_offset + e.stream_length as usize)
                .ok_or("woff2: table slice oob")?
                .to_vec()
        };
        out_tables.push(OutTable { tag: e.tag, data });
    }

    // Patch head.indexToLocFormat (offset 50, u16) to match the loca we built.
    for t in &mut out_tables {
        if &t.tag == b"head" && t.data.len() >= 52 {
            t.data[50..52].copy_from_slice(&index_format.to_be_bytes());
        }
    }

    // Assemble the SFNT: offset table + tag-sorted directory + padded data.
    out_tables.sort_by(|a, b| a.tag.cmp(&b.tag));
    let n = out_tables.len() as u16;
    // searchRange/entrySelector/rangeShift.
    let mut entry_selector = 0u16;
    let mut search_range = 16u32;
    while search_range * 2 <= (n as u32) * 16 {
        search_range *= 2;
        entry_selector += 1;
    }
    let range_shift = (n as u32) * 16 - search_range;

    let mut out = Vec::new();
    out.extend_from_slice(&flavor.to_be_bytes());
    out.extend_from_slice(&be_u16(n));
    out.extend_from_slice(&be_u16(search_range as u16));
    out.extend_from_slice(&be_u16(entry_selector));
    out.extend_from_slice(&be_u16(range_shift as u16));

    let dir_size = 12 + out_tables.len() * 16;
    // Compute offsets (each table 4-byte aligned).
    let mut offset = dir_size;
    let mut offsets = Vec::with_capacity(out_tables.len());
    for t in &out_tables {
        offsets.push(offset);
        offset += t.data.len();
        offset = (offset + 3) & !3;
    }

    // Directory records.
    for (i, t) in out_tables.iter().enumerate() {
        out.extend_from_slice(&t.tag);
        out.extend_from_slice(&table_checksum(&t.data).to_be_bytes());
        out.extend_from_slice(&(offsets[i] as u32).to_be_bytes());
        out.extend_from_slice(&(t.data.len() as u32).to_be_bytes());
    }
    // Table data (padded to 4 bytes).
    for t in &out_tables {
        out.extend_from_slice(&t.data);
        while out.len() % 4 != 0 {
            out.push(0);
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    // A fonttools-produced WOFF2 of an ASCII subset of Geist Mono (glyf/loca
    // transformed). The reconstruction is validated bit-for-bit against the
    // fonttools-decompressed reference on the host in `tools/woff2diff`.
    static GEIST_WOFF2: &[u8] =
        include_bytes!("../../tools/webcompat/fixtures/woff2/geist-ascii.woff2");

    #[test_case]
    fn woff2_signature_detection() {
        assert!(is_woff2(b"wOF2\x00\x01\x00\x00"));
        assert!(!is_woff2(b"wOFF\x00\x01\x00\x00"));
        assert!(!is_woff2(b"wO"));
    }

    #[test_case]
    fn read_255ushort_encodings() {
        // Values < 253 are a single byte; 255→+253, 254→+506, 253→2-byte word.
        assert_eq!(read_255ushort(&[252], &mut 0).unwrap(), 252);
        assert_eq!(read_255ushort(&[255, 0], &mut 0).unwrap(), 253);
        assert_eq!(read_255ushort(&[254, 0], &mut 0).unwrap(), 506);
        assert_eq!(read_255ushort(&[253, 0x01, 0x02], &mut 0).unwrap(), 0x0102);
    }

    #[test_case]
    fn woff2_decodes_to_parseable_sfnt() {
        let sfnt = woff2_to_sfnt(GEIST_WOFF2).expect("woff2 decode");
        // A valid TrueType SFNT begins with 0x00010000 or 'true'/'OTTO'.
        let ver = u32::from_be_bytes([sfnt[0], sfnt[1], sfnt[2], sfnt[3]]);
        assert!(ver == 0x0001_0000 || ver == 0x4F54_544F || ver == 0x7472_7565);
        // Directory must contain glyf + loca (reconstructed) and head.
        let num_tables = u16::from_be_bytes([sfnt[4], sfnt[5]]) as usize;
        let mut has_glyf = false;
        let mut has_loca = false;
        for i in 0..num_tables {
            let o = 12 + i * 16;
            let tag = &sfnt[o..o + 4];
            if tag == b"glyf" {
                has_glyf = true;
            }
            if tag == b"loca" {
                has_loca = true;
            }
        }
        assert!(has_glyf && has_loca);

        // fontdue must parse it and rasterize a glyph to a non-empty bitmap.
        let font = fontdue::Font::from_bytes(sfnt.as_slice(), fontdue::FontSettings::default())
            .expect("reconstructed SFNT parses in fontdue");
        let (metrics, bitmap) = font.rasterize('A', 40.0);
        assert!(metrics.width > 0 && metrics.height > 0);
        assert!(bitmap.iter().any(|&p| p != 0));
    }

    #[test_case]
    fn woff2_rejects_garbage() {
        assert!(woff2_to_sfnt(b"wOF2short").is_err());
        assert!(woff2_to_sfnt(b"notawoff2blob!!!").is_err());
    }
}
