//! VP9 bitstream layer: superframe index, the boolean (arithmetic) decoder, and
//! the uncompressed frame header (VP9 Bitstream & Decoding Process
//! Specification v0.6, §6.1–6.2 and §9.2).
//!
//! VP9 shares nothing with the H.26x layer below the container, which is why
//! none of [`super::bits`] is reused:
//!
//! * **There are no NAL units and no start codes.** A coded frame is just
//!   bytes; a container sample may hold *several* of them glued together with a
//!   trailing **superframe index** — typically an invisible ALTREF plus the
//!   frame that shows it. A demuxer that treats a sample as one frame decodes
//!   the ALTREF and displays it, which looks like the video jumping ahead.
//! * **There is no Exp-Golomb.** The uncompressed header is plain fixed-width
//!   bits ([`BitReader`]); everything after it is arithmetic-coded through
//!   [`BoolDecoder`], including the *probabilities* used to decode the rest.
//! * **Keyframes are not a container property.** `is_sync` in an mp4/mkv sample
//!   table is a hint; the authority is `frame_type` in the header, and a
//!   `show_existing_frame` sample codes no picture at all.
//!
//! Pure + panic-free: malformed input returns `Err`, never crashes.

use alloc::vec::Vec;

pub mod decoder;
pub mod header;
pub mod idct_kernels;
pub mod loopfilter;
pub mod inter;
pub mod intra;
pub mod tables;
pub mod tile;
pub mod tokens;
pub mod transform;

/// Frames a superframe index may hold (3 bits + 1).
pub const MAX_FRAMES_PER_SUPERFRAME: usize = 8;
/// Reference-frame slots a VP9 decoder keeps.
pub const NUM_REF_FRAMES: usize = 8;
/// References a single inter frame may name.
pub const REFS_PER_FRAME: usize = 3;
/// Segments the segmentation map may distinguish.
pub const MAX_SEGMENTS: usize = 8;
/// Segment-level features (`ALT_Q`, `ALT_L`, `REF_FRAME`, `SKIP`).
pub const SEG_LVL_MAX: usize = 4;
/// Loop-filter deltas: one per reference frame kind (intra + 3 inter).
pub const MAX_REF_LF_DELTAS: usize = 4;
/// Loop-filter deltas: one per prediction mode class.
pub const MAX_MODE_LF_DELTAS: usize = 2;
/// Tile-width bounds in units of 64x64 superblocks (§6.2.14).
const MIN_TILE_WIDTH_B64: u32 = 4;
const MAX_TILE_WIDTH_B64: u32 = 64;

/// `segmentation_feature_bits` (§6.2.11) — how many bits each segment feature's
/// value takes: enough for its maximum (255 quantiser, 63 loop-filter level, 3
/// reference frames, and the skip flag which carries no value at all).
const SEG_FEATURE_BITS: [u32; SEG_LVL_MAX] = [8, 6, 2, 0];
/// `segmentation_feature_signed` — whether a sign bit follows the value. Only
/// the quantiser and loop-filter deltas are signed.
const SEG_FEATURE_SIGNED: [bool; SEG_LVL_MAX] = [true, true, false, false];
/// The largest value each feature may legally carry. A field wide enough to
/// code 255 can also code numbers a conforming stream never uses, and these
/// values go on to index quantiser and loop-filter tables — so they are clamped
/// on the way in, as libvpx does, rather than trusted and bounds-checked at
/// every later use.
const SEG_FEATURE_MAX: [i32; SEG_LVL_MAX] = [255, 63, 3, 0];

/// A plain MSB-first bit reader for the *uncompressed* header. VP9 has no
/// emulation prevention and no Exp-Golomb, so this is deliberately not
/// [`super::bits::BitReader`] — sharing one would invite calling `ue()` on a
/// VP9 stream, which reads a valid-looking number from the wrong syntax.
pub struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        BitReader { data, pos: 0 }
    }

    /// Bit position from the start (so the caller can find the byte at which
    /// the arithmetic-coded compressed header begins).
    pub fn bit_pos(&self) -> usize {
        self.pos
    }

    pub fn bit(&mut self) -> Result<u32, &'static str> {
        if self.pos >= self.data.len() * 8 {
            return Err("vp9: header read past end");
        }
        let b = self.data[self.pos >> 3];
        let shift = 7 - (self.pos & 7);
        self.pos += 1;
        Ok(((b >> shift) & 1) as u32)
    }

    /// `f(n)` — `n` bits MSB-first.
    pub fn f(&mut self, n: u32) -> Result<u32, &'static str> {
        let mut v = 0u32;
        for _ in 0..n {
            v = (v << 1) | self.bit()?;
        }
        Ok(v)
    }

    pub fn flag(&mut self) -> Result<bool, &'static str> {
        Ok(self.bit()? != 0)
    }

    /// `s(n)` — an `n`-bit magnitude followed by a sign bit. The sign is
    /// **after** the value, not a two's-complement high bit; reading it the
    /// other way round gives a plausible wrong number for every delta in the
    /// loop-filter and quantiser headers.
    pub fn s(&mut self, n: u32) -> Result<i32, &'static str> {
        let v = self.f(n)? as i32;
        Ok(if self.flag()? { -v } else { v })
    }

    /// Advance to the next byte boundary.
    pub fn byte_align(&mut self) {
        self.pos = (self.pos + 7) & !7;
    }
}

// ---------------------------------------------------------------------------
// Boolean (arithmetic) decoder — §9.2
// ---------------------------------------------------------------------------

/// VP9's boolean decoder, in libvpx's windowed form: `value` holds the coded
/// bits left-aligned in a 64-bit register so renormalisation shifts many bits at
/// once instead of looping a bit at a time.
///
/// The spec presents the same coder bit-serially
/// (`split = 1 + (((range - 1) * p) >> 8)`); this is the identical arithmetic
/// written as `(range * p + (256 - p)) >> 8`, which is where libvpx's form comes
/// from — but the coefficient layer calls this millions of times per frame, and
/// a bit-serial renormalise is the difference between a decoder that plays and
/// one that does not.
pub struct BoolDecoder<'a> {
    data: &'a [u8],
    pos: usize,
    value: u64,
    range: u32,
    /// Valid bits held in `value` beyond the top byte. Negative means refill.
    count: i32,
}

const BD_VALUE_BITS: i32 = 64;
/// libvpx `LOTS_OF_BITS`. When the buffer runs out, `count` is bumped by this
/// so the decoder can keep consuming *virtual zero bits* — which a conforming
/// stream legitimately does, because the arithmetic coder is pre-loaded up to
/// 56 bits ahead of what it has actually consumed. Treating the first read past
/// the last byte as an error rejects every valid frame.
const LOTS_OF_BITS: i32 = 0x4000_0000;

impl<'a> BoolDecoder<'a> {
    /// Start decoding at `data[0]`. The first bit is a marker that must be 0
    /// (§9.2.1); a non-zero marker means this is not a VP9 arithmetic stream —
    /// most often a mis-computed header size, which otherwise decodes garbage
    /// silently for a whole frame.
    pub fn new(data: &'a [u8]) -> Result<BoolDecoder<'a>, &'static str> {
        if data.is_empty() {
            return Err("vp9 bool: empty partition");
        }
        let mut d = BoolDecoder { data, pos: 0, value: 0, range: 255, count: -8 };
        d.fill();
        if d.read_bool(128) != 0 {
            return Err("vp9 bool: marker bit set (bad partition size?)");
        }
        Ok(d)
    }

    fn fill(&mut self) {
        let bits_left = (self.data.len() - self.pos) as i32 * 8;
        let mut shift = BD_VALUE_BITS - 8 - (self.count + 8);
        if bits_left > BD_VALUE_BITS {
            while shift >= 0 && self.pos < self.data.len() {
                self.count += 8;
                self.value |= (self.data[self.pos] as u64) << shift;
                self.pos += 1;
                shift -= 8;
            }
            return;
        }
        // Near the end. `bits_over` is how far the pre-load would reach past
        // the last byte; once it does, `count` takes LOTS_OF_BITS and the
        // remaining bits read as zero.
        let bits_over = shift + 8 - bits_left;
        let mut loop_end = 0;
        if bits_over >= 0 {
            self.count += LOTS_OF_BITS;
            loop_end = bits_over;
        }
        if bits_over < 0 || bits_left > 0 {
            while shift >= loop_end && self.pos < self.data.len() {
                self.count += 8;
                self.value |= (self.data[self.pos] as u64) << shift;
                self.pos += 1;
                shift -= 8;
            }
        }
    }

    /// True once the decoder has consumed materially more than its partition
    /// held — libvpx `vpx_reader_has_error`.
    ///
    /// Note what this is *not*: it is not "a read touched the last byte". The
    /// coder pre-loads up to 56 bits ahead, so every well-formed partition ends
    /// with the reader holding virtual zeros; flagging that rejects every valid
    /// frame (it rejected the first real keyframe this decoder was pointed at).
    pub fn exhausted(&self) -> bool {
        self.count > BD_VALUE_BITS && self.count < LOTS_OF_BITS
    }

    /// Decode one boolean with probability `prob`/256 of being 0.
    pub fn read_bool(&mut self, prob: u8) -> u32 {
        let split = (self.range * prob as u32 + (256 - prob as u32)) >> 8;
        if self.count < 0 {
            self.fill();
        }
        let bigsplit = (split as u64) << (BD_VALUE_BITS - 8);
        let mut bit = 0u32;
        let mut range = split;
        if self.value >= bigsplit {
            range = self.range - split;
            self.value -= bigsplit;
            bit = 1;
        }
        // Renormalise: `range` is in 1..=255 here, and for a byte
        // `leading_zeros()` is exactly libvpx's `vpx_norm` table.
        let shift = (range as u8).leading_zeros();
        self.range = range << shift;
        self.value <<= shift;
        self.count = self.count.saturating_sub(shift as i32);
        bit
    }

    /// `L(n)` — an `n`-bit literal, each bit at probability 128.
    pub fn read_literal(&mut self, n: u32) -> u32 {
        let mut v = 0u32;
        for _ in 0..n {
            v = (v << 1) | self.read_bool(128);
        }
        v
    }

    /// Read a value that is present only when a flag bit says so, else 0.
    pub fn read_bool_or(&mut self, prob: u8, n: u32) -> u32 {
        if self.read_bool(prob) != 0 {
            self.read_literal(n)
        } else {
            0
        }
    }

    /// Decode a symbol from a VP9 probability tree. A tree is a flat array of
    /// `i8` where a non-positive entry is `-symbol` and a positive entry is the
    /// index of the next node pair.
    pub fn read_tree(&mut self, tree: &[i8], probs: &[u8]) -> u32 {
        let mut i: usize = 0;
        loop {
            let p = *probs.get(i >> 1).unwrap_or(&128);
            let n = tree[i + self.read_bool(p) as usize];
            if n <= 0 {
                return (-n) as u32;
            }
            i = n as usize;
        }
    }

    /// A signed value: magnitude then sign, as in the uncompressed header.
    pub fn read_signed_literal(&mut self, n: u32) -> i32 {
        let v = self.read_literal(n) as i32;
        if self.read_bool(128) != 0 {
            -v
        } else {
            v
        }
    }
}

// ---------------------------------------------------------------------------
// Superframe index — Annex B
// ---------------------------------------------------------------------------

/// Split one container sample into its coded frames.
///
/// A VP9 sample may be a *superframe*: several coded frames concatenated, with a
/// trailing index whose first and last bytes are identical markers
/// (`110xxxxx`). Almost every libvpx-encoded stream uses this to carry an
/// invisible ALTREF alongside the frame that displays it, so treating a sample
/// as a single frame is not a rare edge case — it shows the wrong picture on
/// ordinary content.
///
/// Returns byte ranges into `data`, in decode order. A sample with no index is
/// one frame, so the common case allocates a single entry.
pub fn split_superframe(data: &[u8]) -> Result<Vec<(usize, usize)>, &'static str> {
    if data.is_empty() {
        return Err("vp9: empty sample");
    }
    let marker = data[data.len() - 1];
    if marker & 0xe0 == 0xc0 {
        let frames = ((marker & 0x7) as usize + 1).min(MAX_FRAMES_PER_SUPERFRAME);
        let mag = ((marker >> 3) & 0x3) as usize + 1;
        let index_sz = 2 + mag * frames;
        if data.len() >= index_sz && data[data.len() - index_sz] == marker {
            // The index is well-formed: read the per-frame sizes (little-endian).
            let mut out = Vec::with_capacity(frames);
            let mut p = data.len() - index_sz + 1;
            let mut off = 0usize;
            for _ in 0..frames {
                let mut sz = 0usize;
                for k in 0..mag {
                    sz |= (data[p + k] as usize) << (8 * k);
                }
                p += mag;
                if off + sz > data.len() - index_sz {
                    return Err("vp9: superframe size overruns the sample");
                }
                out.push((off, off + sz));
                off += sz;
            }
            return Ok(out);
        }
    }
    Ok(alloc::vec![(0, data.len())])
}

// ---------------------------------------------------------------------------
// Uncompressed frame header — §6.2
// ---------------------------------------------------------------------------

/// VP9 colour spaces (§6.2.2).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ColorSpace {
    Unknown = 0,
    Bt601 = 1,
    Bt709 = 2,
    Smpte170 = 3,
    Smpte240 = 4,
    Bt2020 = 5,
    Reserved = 6,
    Rgb = 7,
}

impl ColorSpace {
    fn from_code(v: u32) -> ColorSpace {
        match v {
            1 => ColorSpace::Bt601,
            2 => ColorSpace::Bt709,
            3 => ColorSpace::Smpte170,
            4 => ColorSpace::Smpte240,
            5 => ColorSpace::Bt2020,
            6 => ColorSpace::Reserved,
            7 => ColorSpace::Rgb,
            _ => ColorSpace::Unknown,
        }
    }
}

/// Sub-pixel interpolation filter (§6.2.6). The bitstream codes these in a
/// *different* order than the enum, via `LITERAL_TO_FILTER`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum InterpFilter {
    EightTap = 0,
    EightTapSmooth = 1,
    EightTapSharp = 2,
    Bilinear = 3,
    Switchable = 4,
}

/// §6.2.6: the two-bit literal does **not** index the filter enum directly.
const LITERAL_TO_FILTER: [InterpFilter; 4] = [
    InterpFilter::EightTapSmooth,
    InterpFilter::EightTap,
    InterpFilter::EightTapSharp,
    InterpFilter::Bilinear,
];

/// libvpx `setup_past_independence`: the reference-frame loop-filter deltas do
/// **not** start at zero. Intra is +1, last 0, golden and altref -1, and every
/// intra frame resets to these before the header's own updates.
///
/// It matters on the very first frame: an intra frame with deltas enabled
/// filters one level *above* the transmitted `loop_filter_level`, so starting
/// from zero leaves a picture that is close and a reference that drifts.
pub const DEFAULT_REF_LF_DELTAS: [i32; MAX_REF_LF_DELTAS] = [1, 0, -1, -1];

/// Loop-filter parameters (§6.2.8).
#[derive(Clone, Copy, Debug)]
pub struct LoopFilterParams {
    pub level: u32,
    pub sharpness: u32,
    pub delta_enabled: bool,
    pub delta_update: bool,
    pub ref_deltas: [i32; MAX_REF_LF_DELTAS],
    pub mode_deltas: [i32; MAX_MODE_LF_DELTAS],
    /// Which `ref_deltas` this header actually transmitted (the rest persist
    /// from the previous frame — they are *not* reset to zero).
    pub ref_delta_updated: [bool; MAX_REF_LF_DELTAS],
    pub mode_delta_updated: [bool; MAX_MODE_LF_DELTAS],
}

impl Default for LoopFilterParams {
    fn default() -> Self {
        LoopFilterParams {
            level: 0,
            sharpness: 0,
            delta_enabled: false,
            delta_update: false,
            ref_deltas: DEFAULT_REF_LF_DELTAS,
            mode_deltas: [0; MAX_MODE_LF_DELTAS],
            ref_delta_updated: [false; MAX_REF_LF_DELTAS],
            mode_delta_updated: [false; MAX_MODE_LF_DELTAS],
        }
    }
}

impl LoopFilterParams {
    /// `vp9_loop_filter_frame_init`: the per-(reference, mode) filter level.
    /// Index `[0][*]` is intra; `[1..=3][0..=1]` are the inter references with
    /// the ZEROMV / moving mode classes.
    pub fn level_table(&self) -> [[u8; 2]; 4] {
        let mut out = [[self.level as u8; 2]; 4];
        if !self.delta_enabled {
            return out;
        }
        let scale = 1i32 << (self.level >> 5);
        let base = self.level as i32;
        out[0] = [(base + self.ref_deltas[0] * scale).clamp(0, 63) as u8; 2];
        for rf in 1..4 {
            for m in 0..2 {
                out[rf][m] =
                    (base + self.ref_deltas[rf] * scale + self.mode_deltas[m] * scale).clamp(0, 63) as u8;
            }
        }
        out
    }

    /// The filter level an **intra** block runs at: the transmitted level plus
    /// the intra reference delta, scaled by `1 << (level >> 5)` and clamped.
    pub fn intra_level(&self) -> u8 {
        if !self.delta_enabled {
            return self.level as u8;
        }
        let scale = 1i32 << (self.level >> 5);
        (self.level as i32 + self.ref_deltas[0] * scale).clamp(0, 63) as u8
    }
}

/// Quantiser parameters (§6.2.9).
#[derive(Clone, Copy, Debug, Default)]
pub struct QuantParams {
    pub base_q_idx: u32,
    pub delta_q_y_dc: i32,
    pub delta_q_uv_dc: i32,
    pub delta_q_uv_ac: i32,
}

impl QuantParams {
    /// True when every plane's quantiser is the identity, so the frame is coded
    /// losslessly and the transform is the Walsh-Hadamard rather than the DCT.
    pub fn lossless(&self) -> bool {
        self.base_q_idx == 0 && self.delta_q_y_dc == 0 && self.delta_q_uv_dc == 0 && self.delta_q_uv_ac == 0
    }
}

/// Segmentation parameters (§6.2.11).
#[derive(Clone, Copy, Debug)]
pub struct SegmentationParams {
    pub enabled: bool,
    pub update_map: bool,
    pub temporal_update: bool,
    pub tree_probs: [u8; 7],
    pub pred_probs: [u8; 3],
    pub update_data: bool,
    pub abs_or_delta_update: bool,
    pub feature_enabled: [[bool; SEG_LVL_MAX]; MAX_SEGMENTS],
    pub feature_data: [[i32; SEG_LVL_MAX]; MAX_SEGMENTS],
}

impl Default for SegmentationParams {
    fn default() -> Self {
        SegmentationParams {
            enabled: false,
            update_map: false,
            temporal_update: false,
            // 255 is "always take this branch" — the value the spec assigns to
            // a probability the header chose not to code.
            tree_probs: [255; 7],
            pred_probs: [255; 3],
            update_data: false,
            abs_or_delta_update: false,
            feature_enabled: [[false; SEG_LVL_MAX]; MAX_SEGMENTS],
            feature_data: [[0; SEG_LVL_MAX]; MAX_SEGMENTS],
        }
    }
}

/// Tile layout (§6.2.14).
#[derive(Clone, Copy, Debug, Default)]
pub struct TileInfo {
    pub cols_log2: u32,
    pub rows_log2: u32,
}

/// A parsed VP9 uncompressed frame header.
#[derive(Clone, Debug)]
pub struct FrameHeader {
    pub profile: u8,
    /// A `show_existing_frame` header codes no picture: it just re-displays a
    /// reference slot. Everything below is meaningless when this is `Some`.
    pub show_existing_frame: Option<u8>,
    pub key_frame: bool,
    pub show_frame: bool,
    pub error_resilient: bool,
    pub intra_only: bool,
    pub reset_frame_context: u32,
    pub refresh_frame_flags: u8,
    pub ref_frame_idx: [u8; REFS_PER_FRAME],
    pub ref_frame_sign_bias: [bool; REFS_PER_FRAME],
    pub allow_high_precision_mv: bool,
    pub interp_filter: InterpFilter,
    pub refresh_frame_context: bool,
    pub frame_parallel_decoding_mode: bool,
    pub frame_context_idx: u32,
    pub bit_depth: u8,
    pub color_space: ColorSpace,
    pub color_range_full: bool,
    pub subsampling_x: bool,
    pub subsampling_y: bool,
    pub width: u32,
    pub height: u32,
    pub render_width: u32,
    pub render_height: u32,
    pub loop_filter: LoopFilterParams,
    pub quant: QuantParams,
    pub segmentation: SegmentationParams,
    pub tiles: TileInfo,
    /// Size in bytes of the arithmetic-coded compressed header that follows.
    pub header_size_in_bytes: u32,
    /// Byte offset in the frame at which the compressed header begins.
    pub uncompressed_header_bytes: usize,
}

impl FrameHeader {
    /// True when nothing in this frame predicts from another picture.
    pub fn is_intra(&self) -> bool {
        self.key_frame || self.intra_only
    }
    /// Frame size in 8x8 blocks.
    pub fn mi_cols(&self) -> u32 {
        (self.width + 7) / 8
    }
    pub fn mi_rows(&self) -> u32 {
        (self.height + 7) / 8
    }
    /// Frame size in 64x64 superblocks.
    pub fn sb64_cols(&self) -> u32 {
        (self.mi_cols() + 7) / 8
    }
    pub fn sb64_rows(&self) -> u32 {
        (self.mi_rows() + 7) / 8
    }
    /// Byte offset at which the tile data begins (after the compressed header).
    pub fn tile_data_offset(&self) -> usize {
        self.uncompressed_header_bytes + self.header_size_in_bytes as usize
    }
}

/// Dimensions previously stored in each reference slot, so
/// `frame_size_with_refs` can adopt one. A decoder keeps this alongside the
/// reference pictures themselves.
pub type RefSizes = [(u32, u32); NUM_REF_FRAMES];

/// Parse the uncompressed header of one coded frame (§6.2).
///
/// `ref_sizes` supplies the dimensions of the eight reference slots; an inter
/// frame usually **does not code its own size** and instead says "same as
/// reference *n*". Passing zeros makes such a frame parse to a zero size rather
/// than failing, which is why an unknown slot is rejected explicitly here.
pub fn parse_frame_header(data: &[u8], ref_sizes: &RefSizes) -> Result<FrameHeader, &'static str> {
    let mut r = BitReader::new(data);
    if r.f(2)? != 2 {
        return Err("vp9: bad frame marker");
    }
    let profile_low = r.bit()?;
    let profile_high = r.bit()?;
    let profile = ((profile_high << 1) | profile_low) as u8;
    if profile == 3 && r.flag()? {
        return Err("vp9: reserved bit set in profile 3");
    }
    let mut h = FrameHeader {
        profile,
        show_existing_frame: None,
        key_frame: false,
        show_frame: false,
        error_resilient: false,
        intra_only: false,
        reset_frame_context: 0,
        refresh_frame_flags: 0,
        ref_frame_idx: [0; REFS_PER_FRAME],
        ref_frame_sign_bias: [false; REFS_PER_FRAME],
        allow_high_precision_mv: false,
        interp_filter: InterpFilter::EightTap,
        refresh_frame_context: false,
        frame_parallel_decoding_mode: false,
        frame_context_idx: 0,
        bit_depth: 8,
        color_space: ColorSpace::Bt601,
        color_range_full: false,
        subsampling_x: true,
        subsampling_y: true,
        width: 0,
        height: 0,
        render_width: 0,
        render_height: 0,
        loop_filter: LoopFilterParams::default(),
        quant: QuantParams::default(),
        segmentation: SegmentationParams::default(),
        tiles: TileInfo::default(),
        header_size_in_bytes: 0,
        uncompressed_header_bytes: 0,
    };

    if r.flag()? {
        // show_existing_frame: the whole rest of the header is absent.
        h.show_existing_frame = Some(r.f(3)? as u8);
        h.show_frame = true;
        r.byte_align();
        h.uncompressed_header_bytes = r.bit_pos() / 8;
        return Ok(h);
    }
    h.key_frame = r.bit()? == 0;
    h.show_frame = r.flag()?;
    h.error_resilient = r.flag()?;

    if h.key_frame {
        frame_sync_code(&mut r)?;
        color_config(&mut r, &mut h)?;
        frame_size(&mut r, &mut h)?;
        render_size(&mut r, &mut h)?;
        h.refresh_frame_flags = 0xff;
        h.intra_only = false;
    } else {
        h.intra_only = if !h.show_frame { r.flag()? } else { false };
        h.reset_frame_context = if h.error_resilient { 0 } else { r.f(2)? };
        if h.intra_only {
            frame_sync_code(&mut r)?;
            if profile > 0 {
                color_config(&mut r, &mut h)?;
            } else {
                // Profile 0 intra-only frames are 8-bit 4:2:0 BT.601 by
                // definition — the header codes none of it.
                h.color_space = ColorSpace::Bt601;
                h.subsampling_x = true;
                h.subsampling_y = true;
                h.bit_depth = 8;
            }
            h.refresh_frame_flags = r.f(8)? as u8;
            frame_size(&mut r, &mut h)?;
            render_size(&mut r, &mut h)?;
        } else {
            h.refresh_frame_flags = r.f(8)? as u8;
            for i in 0..REFS_PER_FRAME {
                h.ref_frame_idx[i] = r.f(3)? as u8;
                h.ref_frame_sign_bias[i] = r.flag()?;
            }
            frame_size_with_refs(&mut r, &mut h, ref_sizes)?;
            h.allow_high_precision_mv = r.flag()?;
            h.interp_filter = read_interp_filter(&mut r)?;
        }
    }

    if !h.error_resilient {
        h.refresh_frame_context = r.flag()?;
        h.frame_parallel_decoding_mode = r.flag()?;
    } else {
        // An error-resilient frame may not carry probabilities forward.
        h.refresh_frame_context = false;
        h.frame_parallel_decoding_mode = true;
    }
    h.frame_context_idx = r.f(2)?;

    loop_filter_params(&mut r, &mut h.loop_filter)?;
    quantization_params(&mut r, &mut h.quant)?;
    segmentation_params(&mut r, &mut h.segmentation)?;
    tile_info(&mut r, &mut h)?;
    h.header_size_in_bytes = r.f(16)?;
    if h.header_size_in_bytes == 0 {
        return Err("vp9: zero-length compressed header");
    }
    r.byte_align();
    h.uncompressed_header_bytes = r.bit_pos() / 8;
    if h.tile_data_offset() > data.len() {
        return Err("vp9: compressed header overruns the frame");
    }
    Ok(h)
}

fn frame_sync_code(r: &mut BitReader) -> Result<(), &'static str> {
    if r.f(8)? != 0x49 || r.f(8)? != 0x83 || r.f(8)? != 0x42 {
        return Err("vp9: bad frame sync code");
    }
    Ok(())
}

fn color_config(r: &mut BitReader, h: &mut FrameHeader) -> Result<(), &'static str> {
    h.bit_depth = if h.profile >= 2 {
        if r.flag()? {
            12
        } else {
            10
        }
    } else {
        8
    };
    h.color_space = ColorSpace::from_code(r.f(3)?);
    if h.color_space != ColorSpace::Rgb {
        h.color_range_full = r.flag()?;
        if h.profile == 1 || h.profile == 3 {
            h.subsampling_x = r.flag()?;
            h.subsampling_y = r.flag()?;
            if r.flag()? {
                return Err("vp9: reserved bit set in color config");
            }
        } else {
            h.subsampling_x = true;
            h.subsampling_y = true;
        }
    } else {
        // 4:4:4 RGB is always full range and never subsampled.
        h.color_range_full = true;
        if h.profile == 1 || h.profile == 3 {
            h.subsampling_x = false;
            h.subsampling_y = false;
            if r.flag()? {
                return Err("vp9: reserved bit set in RGB color config");
            }
        } else {
            return Err("vp9: RGB requires profile 1 or 3");
        }
    }
    Ok(())
}

fn frame_size(r: &mut BitReader, h: &mut FrameHeader) -> Result<(), &'static str> {
    h.width = r.f(16)? + 1;
    h.height = r.f(16)? + 1;
    Ok(())
}

fn render_size(r: &mut BitReader, h: &mut FrameHeader) -> Result<(), &'static str> {
    if r.flag()? {
        h.render_width = r.f(16)? + 1;
        h.render_height = r.f(16)? + 1;
    } else {
        h.render_width = h.width;
        h.render_height = h.height;
    }
    Ok(())
}

fn frame_size_with_refs(
    r: &mut BitReader,
    h: &mut FrameHeader,
    ref_sizes: &RefSizes,
) -> Result<(), &'static str> {
    let mut found = false;
    for i in 0..REFS_PER_FRAME {
        if r.flag()? {
            let slot = h.ref_frame_idx[i] as usize;
            let (w, hh) = *ref_sizes.get(slot).ok_or("vp9: ref slot out of range")?;
            if w == 0 || hh == 0 {
                return Err("vp9: frame sizes from an unpopulated reference slot");
            }
            h.width = w;
            h.height = hh;
            found = true;
            break;
        }
    }
    if !found {
        frame_size(r, h)?;
    }
    render_size(r, h)
}

fn read_interp_filter(r: &mut BitReader) -> Result<InterpFilter, &'static str> {
    if r.flag()? {
        Ok(InterpFilter::Switchable)
    } else {
        Ok(LITERAL_TO_FILTER[r.f(2)? as usize])
    }
}

fn loop_filter_params(r: &mut BitReader, lf: &mut LoopFilterParams) -> Result<(), &'static str> {
    lf.level = r.f(6)?;
    lf.sharpness = r.f(3)?;
    lf.delta_enabled = r.flag()?;
    if lf.delta_enabled {
        lf.delta_update = r.flag()?;
        if lf.delta_update {
            for i in 0..MAX_REF_LF_DELTAS {
                if r.flag()? {
                    lf.ref_deltas[i] = r.s(6)?;
                    lf.ref_delta_updated[i] = true;
                }
            }
            for i in 0..MAX_MODE_LF_DELTAS {
                if r.flag()? {
                    lf.mode_deltas[i] = r.s(6)?;
                    lf.mode_delta_updated[i] = true;
                }
            }
        }
    }
    Ok(())
}

fn read_delta_q(r: &mut BitReader) -> Result<i32, &'static str> {
    if r.flag()? {
        r.s(4)
    } else {
        Ok(0)
    }
}

fn quantization_params(r: &mut BitReader, q: &mut QuantParams) -> Result<(), &'static str> {
    q.base_q_idx = r.f(8)?;
    q.delta_q_y_dc = read_delta_q(r)?;
    q.delta_q_uv_dc = read_delta_q(r)?;
    q.delta_q_uv_ac = read_delta_q(r)?;
    Ok(())
}

fn segmentation_params(r: &mut BitReader, s: &mut SegmentationParams) -> Result<(), &'static str> {
    s.enabled = r.flag()?;
    if !s.enabled {
        return Ok(());
    }
    s.update_map = r.flag()?;
    if s.update_map {
        for i in 0..7 {
            s.tree_probs[i] = if r.flag()? { r.f(8)? as u8 } else { 255 };
        }
        s.temporal_update = r.flag()?;
        for i in 0..3 {
            let coded = if s.temporal_update { r.flag()? } else { false };
            s.pred_probs[i] = if coded { r.f(8)? as u8 } else { 255 };
        }
    }
    s.update_data = r.flag()?;
    if s.update_data {
        s.abs_or_delta_update = r.flag()?;
        for i in 0..MAX_SEGMENTS {
            for j in 0..SEG_LVL_MAX {
                let enabled = r.flag()?;
                s.feature_enabled[i][j] = enabled;
                let mut value = 0i32;
                if enabled {
                    let bits = SEG_FEATURE_BITS[j];
                    if bits > 0 {
                        value = (r.f(bits)? as i32).min(SEG_FEATURE_MAX[j]);
                    }
                    if SEG_FEATURE_SIGNED[j] && r.flag()? {
                        value = -value;
                    }
                }
                s.feature_data[i][j] = value;
            }
        }
    }
    Ok(())
}

/// `calc_min_log2_tile_cols` (§6.2.14): a tile column may not be wider than
/// `MAX_TILE_WIDTH_B64`, so a wide frame is *forced* to split.
pub fn min_log2_tile_cols(sb64_cols: u32) -> u32 {
    let mut n = 0u32;
    while (MAX_TILE_WIDTH_B64 << n) < sb64_cols {
        n += 1;
    }
    n
}

/// `calc_max_log2_tile_cols` (§6.2.14): a tile column may not be narrower than
/// `MIN_TILE_WIDTH_B64`.
pub fn max_log2_tile_cols(sb64_cols: u32) -> u32 {
    let mut n = 1u32;
    while (sb64_cols >> n) >= MIN_TILE_WIDTH_B64 {
        n += 1;
    }
    n - 1
}

fn tile_info(r: &mut BitReader, h: &mut FrameHeader) -> Result<(), &'static str> {
    let sb64_cols = h.sb64_cols();
    let min_log2 = min_log2_tile_cols(sb64_cols);
    let max_log2 = max_log2_tile_cols(sb64_cols);
    let mut cols_log2 = min_log2;
    while cols_log2 < max_log2 {
        if r.flag()? {
            cols_log2 += 1;
        } else {
            break;
        }
    }
    h.tiles.cols_log2 = cols_log2;
    let mut rows_log2 = r.f(1)?;
    if rows_log2 > 0 {
        rows_log2 += r.f(1)?;
    }
    h.tiles.rows_log2 = rows_log2;
    Ok(())
}

/// True iff a container sample codes a keyframe — the authority for the sample
/// table's `is_sync`, which a muxer may get wrong and which a superframe makes
/// ambiguous (the *first* frame of a superframe is the one that matters, and a
/// `show_existing_frame` sample codes no picture at all).
pub fn sample_is_keyframe(data: &[u8]) -> bool {
    let Ok(ranges) = split_superframe(data) else {
        return false;
    };
    let Some(&(s, e)) = ranges.first() else {
        return false;
    };
    let Some(chunk) = data.get(s..e) else {
        return false;
    };
    if chunk.len() < 2 {
        return false;
    }
    // frame_marker(2) profile_low(1) profile_high(1) [reserved(1)]
    // show_existing_frame(1) frame_type(1)
    let mut r = BitReader::new(chunk);
    let Ok(marker) = r.f(2) else { return false };
    if marker != 2 {
        return false;
    }
    let (Ok(lo), Ok(hi)) = (r.bit(), r.bit()) else {
        return false;
    };
    let profile = (hi << 1) | lo;
    if profile == 3 && r.bit().is_err() {
        return false;
    }
    match r.bit() {
        Ok(0) => {}
        _ => return false, // show_existing_frame, or read error
    }
    matches!(r.bit(), Ok(0)) // frame_type == KEY_FRAME
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn bool_decoder_matches_the_spec_split() {
        // The windowed form must agree with the spec's bit-serial definition,
        // `split = 1 + (((range - 1) * p) >> 8)`, for every (range, prob) pair
        // a decoder can reach: range 128..=255 after renormalisation, any prob.
        for range in 128u32..=255 {
            for prob in 1u32..=255 {
                let libvpx = (range * prob + (256 - prob)) >> 8;
                let spec = 1 + (((range - 1) * prob) >> 8);
                assert_eq!(libvpx, spec);
                // The split must leave both branches a non-empty sub-range, or
                // renormalisation would shift on a zero range.
                assert!(libvpx >= 1 && libvpx < range);
            }
        }
    }

    #[test_case]
    fn bool_decoder_rejects_a_set_marker_bit() {
        // The first decoded bool must be 0. 0xff decodes to 1 → not a VP9
        // partition (most often a wrong compressed-header size).
        assert!(BoolDecoder::new(&[0xff, 0xff, 0xff, 0xff]).is_err());
        assert!(BoolDecoder::new(&[]).is_err());
        assert!(BoolDecoder::new(&[0x00, 0x00, 0x00, 0x00]).is_ok());
    }

    #[test_case]
    fn bool_decoder_reads_literals_back() {
        // A stream of zero bytes decodes to zeros at p=128; a stream of 0x2a…
        // is checked only for self-consistency of range/renormalisation (it
        // never leaves 1..=255 and never traps).
        let mut d = BoolDecoder::new(&[0x00; 16]).unwrap();
        assert_eq!(d.read_literal(8), 0);
        let mut d = BoolDecoder::new(&[0x2a; 32]).unwrap();
        for _ in 0..64 {
            let _ = d.read_literal(4);
            assert!(d.range >= 128 && d.range <= 255, "range renormalised");
        }
    }

    #[test_case]
    fn a_short_read_past_the_end_is_not_an_error() {
        // The coder pre-loads ~56 bits, so a well-formed partition always ends
        // with virtual zeros in the register. Flagging that as an error is what
        // made the first real keyframe fail to decode.
        let mut d = BoolDecoder::new(&[0x00; 8]).unwrap();
        let _ = d.read_literal(8);
        assert!(!d.exhausted(), "reading inside a partition must not flag an error");
    }

    #[test_case]
    fn a_long_read_past_the_end_is_an_error() {
        let mut d = BoolDecoder::new(&[0x00, 0x00]).unwrap();
        for _ in 0..64 {
            let _ = d.read_literal(8);
        }
        assert!(d.exhausted(), "a decode that ran well off the end must say so");
    }

    #[test_case]
    fn superframe_index_splits_a_sample() {
        // Two frames of 3 and 4 bytes, 1-byte sizes.
        // marker = 110 | mag-1 (0) << 3 | frames-1 (1) = 0b1100_0001 = 0xc1.
        let mut data = alloc::vec![0xaau8; 3];
        data.extend_from_slice(&[0xbb; 4]);
        data.push(0xc1);
        data.push(3);
        data.push(4);
        data.push(0xc1);
        let r = split_superframe(&data).unwrap();
        assert_eq!(r, alloc::vec![(0, 3), (3, 7)]);
    }

    #[test_case]
    fn a_sample_without_an_index_is_one_frame() {
        let data = [0x82u8, 0x49, 0x83, 0x42, 0x00];
        assert_eq!(split_superframe(&data).unwrap(), alloc::vec![(0, 5)]);
        // A trailing byte that merely *looks* like a marker but whose mirror
        // byte disagrees is data, not an index.
        let data = [0x00u8, 0x00, 0x00, 0xc1];
        assert_eq!(split_superframe(&data).unwrap(), alloc::vec![(0, 4)]);
    }

    #[test_case]
    fn superframe_with_lying_sizes_is_refused() {
        // An index that claims more bytes than the sample holds must error, not
        // hand out a range that overruns the buffer.
        let data = [0xaau8, 0xbb, 0xc1, 200, 200, 0xc1];
        assert!(split_superframe(&data).is_err());
    }

    #[test_case]
    fn tile_column_bounds_match_the_spec() {
        // A 64x64-superblock-wide frame (4096 px) must split into at least 2
        // tile columns, since MAX_TILE_WIDTH_B64 is 64.
        assert_eq!(min_log2_tile_cols(64), 0);
        assert_eq!(min_log2_tile_cols(65), 1);
        assert_eq!(min_log2_tile_cols(129), 2);
        // …and may not split so far that a column is under 4 superblocks.
        assert_eq!(max_log2_tile_cols(1), 0);
        assert_eq!(max_log2_tile_cols(4), 0);
        assert_eq!(max_log2_tile_cols(8), 1);
        assert_eq!(max_log2_tile_cols(16), 2);
        assert_eq!(max_log2_tile_cols(64), 4);
    }

    #[test_case]
    fn interp_filter_literal_is_not_the_enum_order() {
        // Literal 0 is EIGHTTAP_SMOOTH, not EIGHTTAP — reading the literal as
        // the enum swaps the two commonest filters and softens every frame.
        assert_eq!(LITERAL_TO_FILTER[0], InterpFilter::EightTapSmooth);
        assert_eq!(LITERAL_TO_FILTER[1], InterpFilter::EightTap);
        assert_eq!(LITERAL_TO_FILTER[2], InterpFilter::EightTapSharp);
        assert_eq!(LITERAL_TO_FILTER[3], InterpFilter::Bilinear);
    }

    /// The first 104 bytes of a real libvpx-vp9 keyframe (176x144, profile 0,
    /// `-g 5 -deadline realtime`), taken from the first sample of a `vp09` mp4.
    ///
    /// It is exactly the 18-byte uncompressed header plus the whole 82-byte
    /// compressed header plus 4 bytes of tile data — a *shorter* fixture is
    /// rejected by `parse_frame_header`'s own bounds check, which is the right
    /// behaviour and is why the header sizes are asserted below.
    const LIBVPX_KEYFRAME: [u8; 104] = [
        0x82, 0x49, 0x83, 0x42, 0x00, 0x0a, 0xf0, 0x08, 0xf6, 0x06, 0x38, 0x24, 0x1c, 0x18, 0x4a,
        0x00, 0x05, 0x20, 0x50, 0x60, 0x0f, 0xcf, 0xff, 0x9e, 0xfc, 0x7f, 0xf0, 0x3f, 0xa3, 0xf8,
        0xff, 0xf1, 0x1f, 0x9f, 0xf8, 0xfe, 0x4f, 0xf0, 0x7c, 0x5e, 0xf9, 0xef, 0xee, 0xf5, 0x9f,
        0xb3, 0xd8, 0x3e, 0x87, 0xf8, 0xfd, 0x87, 0xd7, 0x7e, 0x8f, 0x1d, 0xfa, 0x3b, 0x7f, 0xcf,
        0xfe, 0x0f, 0x0d, 0x8c, 0x78, 0x2c, 0x87, 0x57, 0xcf, 0xab, 0xe9, 0xdf, 0xd1, 0xdd, 0xcb,
        0x04, 0xba, 0xbf, 0x83, 0x71, 0x5d, 0x14, 0xf8, 0xdb, 0xf7, 0xb3, 0x6a, 0x9f, 0xf5, 0x91,
        0x60, 0x5a, 0x9e, 0xeb, 0x7d, 0x80, 0x99, 0xd2, 0xa0, 0x00, 0x7e, 0xa2, 0x8d, 0xed,
    ];

    #[test_case]
    fn parses_a_real_libvpx_keyframe() {
        let refs: RefSizes = [(0, 0); NUM_REF_FRAMES];
        let h = parse_frame_header(&LIBVPX_KEYFRAME, &refs).unwrap();
        assert_eq!(h.profile, 0);
        assert!(h.key_frame);
        assert!(h.show_frame);
        assert!(!h.error_resilient);
        assert!(h.is_intra());
        assert_eq!(h.width, 176);
        assert_eq!(h.height, 144);
        assert_eq!(h.render_width, 176);
        assert_eq!(h.render_height, 144);
        assert_eq!(h.bit_depth, 8);
        assert!(h.subsampling_x && h.subsampling_y, "4:2:0");
        assert_eq!(h.color_space, ColorSpace::Unknown);
        assert!(!h.color_range_full);
        // A keyframe refreshes every reference slot.
        assert_eq!(h.refresh_frame_flags, 0xff);
        assert_eq!(h.quant.base_q_idx, 37);
        assert!(!h.quant.lossless());
        assert_eq!(h.loop_filter.level, 3);
        assert_eq!(h.loop_filter.sharpness, 0);
        assert!(!h.segmentation.enabled);
        assert_eq!(h.tiles.cols_log2, 0);
        assert_eq!(h.tiles.rows_log2, 0);
        assert_eq!(h.header_size_in_bytes, 82);
        assert_eq!(h.uncompressed_header_bytes, 18);
        assert_eq!(h.tile_data_offset(), 100);
        // The size is right iff a bool decoder starts cleanly at that offset:
        // its first bit is a marker that must be 0, so a mis-read header size
        // fails here instead of decoding a frame of noise.
        let cs = h.uncompressed_header_bytes;
        assert!(BoolDecoder::new(&LIBVPX_KEYFRAME[cs..h.tile_data_offset()]).is_ok());
        // Frame geometry in the units the CTU/superblock layer works in.
        assert_eq!(h.mi_cols(), 22);
        assert_eq!(h.mi_rows(), 18);
        assert_eq!(h.sb64_cols(), 3);
        assert_eq!(h.sb64_rows(), 3);
    }

    #[test_case]
    fn keyframe_detection_reads_the_bitstream_not_the_container() {
        assert!(sample_is_keyframe(&LIBVPX_KEYFRAME));
        // Byte 0 is marker(2) profile_lo profile_hi show_existing frame_type
        // show_frame error_res, so `frame_type` is bit 5 counting from the MSB:
        // 0x82 → 0x86 is the same frame declared inter.
        let mut inter = LIBVPX_KEYFRAME;
        inter[0] = 0x86;
        assert!(!sample_is_keyframe(&inter));
        // A show_existing_frame sample codes no picture, so it is not a sync
        // point even though it may be the frame a seek wants to land on.
        // marker=10, profile=00, show_existing=1, idx=010, then padding.
        assert!(!sample_is_keyframe(&[0b1000_1010, 0x00]));
    }

    #[test_case]
    fn a_frame_that_names_an_empty_reference_slot_is_refused() {
        // An inter frame usually takes its size from a reference rather than
        // coding one. With no reference decoded yet that is not a zero-sized
        // frame, it is a stream that cannot be started here.
        // marker=10 profile=00 show_existing=0 frame_type=1 show_frame=1
        // error_res=0 | reset_ctx=00 refresh=0x00 | 3×(ref_idx=000,sign=0)
        // | found_ref=1  → "my size is reference slot 0's size".
        let bits = [0x86u8, 0x00, 0x00, 0x02];
        let empty: RefSizes = [(0, 0); NUM_REF_FRAMES];
        assert!(parse_frame_header(&bits, &empty).is_err());
        // With that slot populated the same bytes parse and adopt its size,
        // which is what proves the failure above was the empty slot and not a
        // mis-built fixture that never reached `frame_size_with_refs`.
        let mut refs: RefSizes = [(0, 0); NUM_REF_FRAMES];
        refs[0] = (320, 240);
        let h = parse_frame_header(&bits, &refs).unwrap_err();
        // It gets past the size and fails later (the fixture is truncated),
        // never on the reference slot.
        assert!(h.contains("read past end"), "unexpected error: {}", h);
    }

    #[test_case]
    fn rejects_garbage() {
        let refs: RefSizes = [(0, 0); NUM_REF_FRAMES];
        assert!(parse_frame_header(&[], &refs).is_err());
        assert!(parse_frame_header(&[0x00, 0x00, 0x00, 0x00], &refs).is_err(), "bad frame marker");
        // A keyframe whose sync code is wrong is not silently accepted.
        let mut bad = LIBVPX_KEYFRAME;
        bad[1] = 0x00;
        assert!(parse_frame_header(&bad, &refs).is_err());
        assert!(!sample_is_keyframe(&[]));
    }
}
