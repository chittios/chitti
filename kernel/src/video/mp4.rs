//! ISO base media file format (ISO/IEC 14496-12) demuxer — the container for
//! `.mp4` and `.mov`. Walks the box tree, finds the first video track it
//! recognises, reads its decoder configuration box (`avcC` / `hvcC` / `vpcC` →
//! [`CodecConfig`]), and assembles the per-sample table (offset/size/DTS/sync)
//! from `stsz`/`stsc`/`stco`/`stts`/`stss`, so the decoder can pull one access
//! unit at a time.
//!
//! It also owns the configuration-record parsers themselves, because those are
//! ISO-BMFF structures wherever they appear — Matroska's `CodecPrivate` carries
//! the very same records, so [`super::mkv`] parses them through here rather than
//! growing a second copy that can drift.
//!
//! Pure over an in-memory `&[u8]` (the file is loaded whole, like the image
//! decoders) — no I/O, no panics on malformed input.

use alloc::vec::Vec;

/// A big-endian reader with bounds checks — every past-end read is an `Err`.
struct Reader<'a> {
    d: &'a [u8],
    p: usize,
}
impl<'a> Reader<'a> {
    fn new(d: &'a [u8]) -> Self {
        Reader { d, p: 0 }
    }
    fn remaining(&self) -> usize {
        self.d.len().saturating_sub(self.p)
    }
    fn u8(&mut self) -> Result<u8, &'static str> {
        let b = *self.d.get(self.p).ok_or("mp4: eof")?;
        self.p += 1;
        Ok(b)
    }
    fn u16(&mut self) -> Result<u16, &'static str> {
        Ok(((self.u8()? as u16) << 8) | self.u8()? as u16)
    }
    fn u24(&mut self) -> Result<u32, &'static str> {
        Ok(((self.u8()? as u32) << 16) | ((self.u8()? as u32) << 8) | self.u8()? as u32)
    }
    fn u32(&mut self) -> Result<u32, &'static str> {
        Ok(((self.u16()? as u32) << 16) | self.u16()? as u32)
    }
    fn u64(&mut self) -> Result<u64, &'static str> {
        Ok(((self.u32()? as u64) << 32) | self.u32()? as u64)
    }
    fn skip(&mut self, n: usize) -> Result<(), &'static str> {
        if n > self.remaining() {
            return Err("mp4: skip past eof");
        }
        self.p += n;
        Ok(())
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], &'static str> {
        if n > self.remaining() {
            return Err("mp4: take past eof");
        }
        let s = &self.d[self.p..self.p + n];
        self.p += n;
        Ok(s)
    }
}

/// A box header: 4-char type + the slice of its *contents* (after the header).
struct BoxHeader<'a> {
    typ: [u8; 4],
    body: &'a [u8],
}

/// Iterate the boxes directly inside `data`. Handles 32-bit `size`, the
/// `size==1` 64-bit largesize form, and `size==0` (extends to end).
fn boxes(data: &[u8]) -> Vec<BoxHeader<'_>> {
    let mut out = Vec::new();
    let mut r = Reader::new(data);
    while r.remaining() >= 8 {
        let start = r.p;
        let size32 = match r.u32() {
            Ok(v) => v,
            Err(_) => break,
        };
        let typ = match r.take(4) {
            Ok(t) => [t[0], t[1], t[2], t[3]],
            Err(_) => break,
        };
        let header_len;
        let box_size;
        if size32 == 1 {
            let large = match r.u64() {
                Ok(v) => v,
                Err(_) => break,
            };
            header_len = 16;
            box_size = large as usize;
        } else if size32 == 0 {
            header_len = 8;
            box_size = data.len() - start;
        } else {
            header_len = 8;
            box_size = size32 as usize;
        }
        if box_size < header_len || start + box_size > data.len() {
            break;
        }
        let body = &data[start + header_len..start + box_size];
        out.push(BoxHeader { typ, body });
        r.p = start + box_size;
    }
    out
}

fn find<'a>(list: &'a [BoxHeader<'a>], typ: &[u8; 4]) -> Option<&'a BoxHeader<'a>> {
    list.iter().find(|b| &b.typ == typ)
}

/// The H.264 decoder config from an `avcC` box.
#[derive(Clone, Debug, Default)]
pub struct AvcC {
    /// NAL length prefix size in bytes (1..=4) for AVCC sample framing.
    pub length_size: u8,
    pub sps: Vec<Vec<u8>>, // each entry includes the NAL header byte
    pub pps: Vec<Vec<u8>>,
}

/// Parse an `AVCDecoderConfigurationRecord` (ISO 14496-15 §5.2.4.1). Public so
/// the Matroska demuxer can parse its `CodecPrivate` (same record).
pub fn parse_avcc(body: &[u8]) -> Result<AvcC, &'static str> {
    let mut r = Reader::new(body);
    let _version = r.u8()?; // configurationVersion (1)
    let _profile = r.u8()?;
    let _profile_compat = r.u8()?;
    let _level = r.u8()?;
    let length_size = (r.u8()? & 0x03) + 1; // lengthSizeMinusOne
    let mut a = AvcC { length_size, ..Default::default() };
    let num_sps = r.u8()? & 0x1f;
    for _ in 0..num_sps {
        let len = r.u16()? as usize;
        a.sps.push(r.take(len)?.to_vec());
    }
    let num_pps = r.u8()?;
    for _ in 0..num_pps {
        let len = r.u16()? as usize;
        a.pps.push(r.take(len)?.to_vec());
    }
    Ok(a)
}

/// The H.265 decoder config from an `hvcC` box.
///
/// Structurally an AVCC record with the parameter sets moved into a
/// *typed array* list — HEVC has three kinds (VPS/SPS/PPS) plus SEI, so a
/// record cannot use AVCC's fixed "SPS list then PPS list" shape.
#[derive(Clone, Debug, Default)]
pub struct HvcC {
    /// NAL length prefix size in bytes (1..=4) for HVCC sample framing.
    pub length_size: u8,
    pub general_profile_idc: u8,
    pub general_tier_high: bool,
    pub general_level_idc: u8,
    pub chroma_format_idc: u8,
    pub bit_depth_luma: u8,
    pub bit_depth_chroma: u8,
    pub vps: Vec<Vec<u8>>, // each entry includes the 2-byte NAL header
    pub sps: Vec<Vec<u8>>,
    pub pps: Vec<Vec<u8>>,
}

/// Parse an `HEVCDecoderConfigurationRecord` (ISO 14496-15 §8.3.3.1.2). Public
/// so the Matroska demuxer can parse its `CodecPrivate`, which is the same
/// record.
///
/// The trap versus `avcC` is that the fixed part is **22 bytes** before the
/// array count — it carries the whole `profile_tier_level` (12 bytes of profile
/// space/compatibility/constraint flags) plus parallelism, chroma format and bit
/// depths. Reading it with `avcC`'s 5-byte preamble lands the NAL arrays inside
/// the constraint flags, where the length fields are large plausible numbers.
pub fn parse_hvcc(body: &[u8]) -> Result<HvcC, &'static str> {
    let mut r = Reader::new(body);
    let version = r.u8()?; // configurationVersion (1)
    if version != 1 {
        return Err("mp4: unsupported hvcC version");
    }
    let b = r.u8()?;
    let mut h = HvcC {
        general_tier_high: b & 0x20 != 0,
        general_profile_idc: b & 0x1f,
        ..Default::default()
    };
    let _general_profile_compatibility = r.u32()?;
    let _general_constraint_hi = r.u32()?; // 48 constraint bits
    let _general_constraint_lo = r.u16()?;
    h.general_level_idc = r.u8()?;
    let _min_spatial_segmentation = r.u16()?; // 4 reserved + 12
    let _parallelism_type = r.u8()?; // 6 reserved + 2
    h.chroma_format_idc = r.u8()? & 0x03;
    h.bit_depth_luma = (r.u8()? & 0x07) + 8;
    h.bit_depth_chroma = (r.u8()? & 0x07) + 8;
    let _avg_frame_rate = r.u16()?;
    let flags = r.u8()?; // constantFrameRate(2) numTemporalLayers(3) nested(1) lengthSizeMinusOne(2)
    h.length_size = (flags & 0x03) + 1;
    let num_arrays = r.u8()?;
    for _ in 0..num_arrays {
        let nal_type = r.u8()? & 0x3f;
        let count = r.u16()?;
        for _ in 0..count {
            let len = r.u16()? as usize;
            let nal = r.take(len)?.to_vec();
            match nal_type {
                32 => h.vps.push(nal),
                33 => h.sps.push(nal),
                34 => h.pps.push(nal),
                _ => {} // SEI and reserved arrays are carried, not needed
            }
        }
    }
    if h.sps.is_empty() {
        return Err("mp4: hvcC has no SPS");
    }
    Ok(h)
}

/// The VP9 decoder config from a `vpcC` box (VP Codec ISO Media File Format
/// Binding). Unlike `avcC`/`hvcC` it carries **no parameter sets** — VP9 has
/// none; every frame's header is self-describing — so this is metadata only and
/// the decoder never depends on it.
#[derive(Clone, Debug, Default)]
pub struct VpcC {
    pub profile: u8,
    /// 10× the level, i.e. 10 = level 1.0, 51 = level 5.1.
    pub level: u8,
    pub bit_depth: u8,
    /// 0 = 4:2:0 vertically co-sited, 1 = 4:2:0 co-located, 2 = 4:2:2, 3 = 4:4:4.
    pub chroma_subsampling: u8,
    pub full_range: bool,
    pub colour_primaries: u8,
    pub transfer_characteristics: u8,
    pub matrix_coefficients: u8,
}

/// Parse a `vpcC` box (a *full* box: 1 version byte + 3 flag bytes first).
pub fn parse_vpcc(body: &[u8]) -> Result<VpcC, &'static str> {
    let mut r = Reader::new(body);
    let _ver_flags = r.u32()?;
    let mut v = VpcC { profile: r.u8()?, level: r.u8()?, ..Default::default() };
    let b = r.u8()?;
    v.bit_depth = b >> 4;
    v.chroma_subsampling = (b >> 1) & 0x07;
    v.full_range = b & 1 != 0;
    v.colour_primaries = r.u8()?;
    v.transfer_characteristics = r.u8()?;
    v.matrix_coefficients = r.u8()?;
    Ok(v)
}

/// Which codec a demuxed video track carries, with its decoder configuration.
///
/// This is the seam that keeps the demuxers codec-agnostic: `mp4`/`mkv` decide
/// *what* the track is, and [`super::StreamDecoder`] decides what to do with it.
#[derive(Clone, Debug)]
pub enum CodecConfig {
    Avc(AvcC),
    Hevc(HvcC),
    /// VP9 (and VP8, which shares the container binding but is not decoded).
    Vp9(VpcC),
}

impl CodecConfig {
    /// NAL length-prefix size for length-framed codecs. **Zero for VP9**, whose
    /// samples are raw frames with no framing at all — a caller that assumes a
    /// prefix reads the first four bytes of the frame header as a length.
    pub fn length_size(&self) -> u8 {
        match self {
            CodecConfig::Avc(a) => a.length_size,
            CodecConfig::Hevc(h) => h.length_size,
            CodecConfig::Vp9(_) => 0,
        }
    }

    /// Short name for the probe line and the player HUD.
    pub fn codec_name(&self) -> &'static str {
        match self {
            CodecConfig::Avc(_) => "H.264",
            CodecConfig::Hevc(_) => "H.265",
            CodecConfig::Vp9(_) => "VP9",
        }
    }
}

impl Default for CodecConfig {
    fn default() -> Self {
        CodecConfig::Avc(AvcC::default())
    }
}

/// One coded sample (access unit) in the file: byte range + timing + keyframe.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sample {
    pub offset: usize,
    pub size: usize,
    pub dts: u64,
    /// Composition (display) timestamp: `dts + ctts offset` (== dts without ctts).
    pub cts: u64,
    pub is_sync: bool,
}

/// The sample-table inputs, decoded from the `stbl` child boxes.
pub struct SampleTables {
    /// Per-sample sizes (`stsz`, or expanded from a fixed default size).
    pub sizes: Vec<u32>,
    /// Chunk file offsets (`stco`/`co64`).
    pub chunk_offsets: Vec<u64>,
    /// `stsc` runs: (first_chunk_index_0based, samples_per_chunk).
    pub stsc: Vec<(u32, u32)>,
    /// `stts` runs: (sample_count, delta).
    pub stts: Vec<(u32, u32)>,
    /// `ctts` runs (sample_count, composition offset), empty when absent.
    pub ctts: Vec<(u32, i64)>,
    /// Sync-sample indices (`stss`, 1-based in the file); empty ⇒ all sync.
    pub sync: Vec<u32>,
}

/// Assemble the flat per-sample list (offset, size, DTS, sync) from the decoded
/// `stbl` tables. This is the fiddly index math (ISO 14496-12 §8.7), pulled
/// out as a pure function so it can be unit-tested directly.
pub fn build_samples(t: &SampleTables) -> Vec<Sample> {
    let n = t.sizes.len();
    let mut samples = Vec::with_capacity(n);
    // Expand stsc → samples-per-chunk for every chunk.
    let num_chunks = t.chunk_offsets.len();
    let mut per_chunk = Vec::with_capacity(num_chunks);
    for c in 0..num_chunks {
        // The applicable run is the last one whose first_chunk <= c.
        let mut spc = 0u32;
        for &(first, count) in &t.stsc {
            if first as usize <= c {
                spc = count;
            } else {
                break;
            }
        }
        per_chunk.push(spc);
    }
    // Walk chunks, laying samples end-to-end within each chunk's offset.
    let mut si = 0usize; // global sample index
    let sync_set = &t.sync;
    let mut next_sync = 0usize;
    // DTS accumulation from stts runs.
    let mut dts = 0u64;
    let mut stts_run = 0usize;
    let mut stts_left = t.stts.first().map(|&(c, _)| c).unwrap_or(0);
    for c in 0..num_chunks {
        let mut off = t.chunk_offsets[c] as usize;
        for _ in 0..per_chunk[c] {
            if si >= n {
                break;
            }
            let size = t.sizes[si] as usize;
            // sync?
            let is_sync = if sync_set.is_empty() {
                true
            } else {
                let one_based = (si + 1) as u32;
                while next_sync < sync_set.len() && sync_set[next_sync] < one_based {
                    next_sync += 1;
                }
                next_sync < sync_set.len() && sync_set[next_sync] == one_based
            };
            // Composition offset from the ctts runs (0 when absent).
            let mut coff = 0i64;
            if !t.ctts.is_empty() {
                let mut acc = 0usize;
                for &(cnt, o) in &t.ctts {
                    if si < acc + cnt as usize {
                        coff = o;
                        break;
                    }
                    acc += cnt as usize;
                }
            }
            let cts = (dts as i64 + coff).max(0) as u64;
            samples.push(Sample { offset: off, size, dts, cts, is_sync });
            off += size;
            // advance DTS
            while stts_left == 0 && stts_run + 1 < t.stts.len() {
                stts_run += 1;
                stts_left = t.stts[stts_run].0;
            }
            if let Some(&(_, delta)) = t.stts.get(stts_run) {
                dts += delta as u64;
                if stts_left > 0 {
                    stts_left -= 1;
                }
            }
            si += 1;
        }
    }
    samples
}

/// Absolute max table entries from an untrusted box (allocation-bomb guard).
const MAX_TABLE_ENTRIES: usize = 2_000_000;

/// Cap `count` against remaining body bytes for fixed-width records.
fn capped_count(count: usize, body_remaining: usize, rec_size: usize) -> Result<usize, &'static str> {
    if count > MAX_TABLE_ENTRIES {
        return Err("mp4: table count too large");
    }
    if rec_size > 0 && count > body_remaining / rec_size {
        return Err("mp4: table count exceeds box size");
    }
    Ok(count)
}

fn parse_stsz(body: &[u8]) -> Result<Vec<u32>, &'static str> {
    let mut r = Reader::new(body);
    let _ver_flags = r.u32()?;
    let default_size = r.u32()?;
    let count = capped_count(r.u32()? as usize, r.remaining(), if default_size != 0 { 0 } else { 4 })?;
    if default_size != 0 {
        return Ok(alloc::vec![default_size; count]);
    }
    let mut v = Vec::with_capacity(count);
    for _ in 0..count {
        v.push(r.u32()?);
    }
    Ok(v)
}

fn parse_offsets(body: &[u8], wide: bool) -> Result<Vec<u64>, &'static str> {
    let mut r = Reader::new(body);
    let _ver_flags = r.u32()?;
    let rec = if wide { 8 } else { 4 };
    let count = capped_count(r.u32()? as usize, r.remaining(), rec)?;
    let mut v = Vec::with_capacity(count);
    for _ in 0..count {
        v.push(if wide { r.u64()? } else { r.u32()? as u64 });
    }
    Ok(v)
}

fn parse_stsc(body: &[u8]) -> Result<Vec<(u32, u32)>, &'static str> {
    let mut r = Reader::new(body);
    let _ver_flags = r.u32()?;
    let count = capped_count(r.u32()? as usize, r.remaining(), 12)?;
    let mut v = Vec::with_capacity(count);
    for _ in 0..count {
        let first_chunk = r.u32()?; // 1-based in file
        let samples_per_chunk = r.u32()?;
        let _desc = r.u32()?;
        v.push((first_chunk.saturating_sub(1), samples_per_chunk));
    }
    Ok(v)
}

fn parse_stts(body: &[u8]) -> Result<Vec<(u32, u32)>, &'static str> {
    let mut r = Reader::new(body);
    let _ver_flags = r.u32()?;
    let count = capped_count(r.u32()? as usize, r.remaining(), 8)?;
    let mut v = Vec::with_capacity(count);
    for _ in 0..count {
        let sc = r.u32()?;
        let delta = r.u32()?;
        v.push((sc, delta));
    }
    Ok(v)
}

/// `ctts`: composition offset runs (v0 unsigned / v1 signed offsets).
fn parse_ctts(body: &[u8]) -> Result<Vec<(u32, i64)>, &'static str> {
    let mut r = Reader::new(body);
    let ver = r.u32()? >> 24;
    let n = capped_count(r.u32()? as usize, r.remaining(), 8)?;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let cnt = r.u32()?;
        let off = r.u32()?;
        let off = if ver == 1 { off as i32 as i64 } else { off as i64 };
        out.push((cnt, off));
    }
    Ok(out)
}

fn parse_stss(body: &[u8]) -> Result<Vec<u32>, &'static str> {
    let mut r = Reader::new(body);
    let _ver_flags = r.u32()?;
    let count = capped_count(r.u32()? as usize, r.remaining(), 4)?;
    let mut v = Vec::with_capacity(count);
    for _ in 0..count {
        v.push(r.u32()?);
    }
    Ok(v)
}

/// A demuxed video track: geometry, timing, decoder config, and samples.
pub struct VideoTrack {
    pub width: u32,
    pub height: u32,
    pub timescale: u32,
    pub duration: u64,
    pub config: CodecConfig,
    pub samples: Vec<Sample>,
}

impl VideoTrack {
    /// Track duration in milliseconds.
    pub fn duration_ms(&self) -> u64 {
        if self.timescale == 0 {
            0
        } else {
            self.duration * 1000 / self.timescale as u64
        }
    }
    pub fn frame_count(&self) -> usize {
        self.samples.len()
    }
}

/// Parse the whole file and return its first decodable video track, if any.
pub fn parse(data: &[u8]) -> Result<VideoTrack, &'static str> {
    let top = boxes(data);
    let moov = find(&top, b"moov").ok_or("mp4: no moov box")?;
    let moov_boxes = boxes(moov.body);
    for trak in moov_boxes.iter().filter(|b| &b.typ == b"trak") {
        if let Some(t) = parse_trak(trak.body)? {
            return Ok(t);
        }
    }
    Err("mp4: no video track found")
}

fn parse_trak(trak_body: &[u8]) -> Result<Option<VideoTrack>, &'static str> {
    let trak = boxes(trak_body);
    // tkhd for the display width/height (16.16 fixed) — a fallback for geometry.
    let (mut tk_w, mut tk_h) = (0u32, 0u32);
    if let Some(tkhd) = find(&trak, b"tkhd") {
        let mut r = Reader::new(tkhd.body);
        let ver_flags = r.u32()?;
        let version = ver_flags >> 24;
        // creation/modification/track_id/reserved/duration vary by version
        if version == 1 {
            r.skip(8 + 8 + 4 + 4 + 8)?;
        } else {
            r.skip(4 + 4 + 4 + 4 + 4)?;
        }
        r.skip(8)?; // reserved[2]
        r.skip(2 + 2 + 2 + 2)?; // layer, alt group, volume, reserved
        r.skip(36)?; // matrix
        tk_w = r.u32()? >> 16;
        tk_h = r.u32()? >> 16;
    }
    let mdia = match find(&trak, b"mdia") {
        Some(m) => boxes(m.body),
        None => return Ok(None),
    };
    // Handler must be 'vide'.
    if let Some(hdlr) = find(&mdia, b"hdlr") {
        let mut r = Reader::new(hdlr.body);
        r.u32()?; // ver/flags
        r.u32()?; // pre_defined
        let handler = r.take(4)?;
        if handler != b"vide" {
            return Ok(None);
        }
    }
    // mdhd timescale + duration.
    let (mut timescale, mut duration) = (0u32, 0u64);
    if let Some(mdhd) = find(&mdia, b"mdhd") {
        let mut r = Reader::new(mdhd.body);
        let ver_flags = r.u32()?;
        if ver_flags >> 24 == 1 {
            r.skip(16)?; // creation+modification (64-bit each)
            timescale = r.u32()?;
            duration = r.u64()?;
        } else {
            r.skip(8)?; // creation+modification (32-bit each)
            timescale = r.u32()?;
            duration = r.u32()? as u64;
        }
    }
    let minf = match find(&mdia, b"minf") {
        Some(m) => boxes(m.body),
        None => return Ok(None),
    };
    let stbl = match find(&minf, b"stbl") {
        Some(s) => boxes(s.body),
        None => return Ok(None),
    };
    // stsd → the video sample entry → its decoder configuration box.
    let stsd = find(&stbl, b"stsd").ok_or("mp4: no stsd")?;
    let (config, (sw, sh)) = parse_stsd(stsd.body)?;

    let sizes = parse_stsz(find(&stbl, b"stsz").ok_or("mp4: no stsz")?.body)?;
    let chunk_offsets = if let Some(co64) = find(&stbl, b"co64") {
        parse_offsets(co64.body, true)?
    } else {
        parse_offsets(find(&stbl, b"stco").ok_or("mp4: no stco/co64")?.body, false)?
    };
    let stsc = parse_stsc(find(&stbl, b"stsc").ok_or("mp4: no stsc")?.body)?;
    let stts = parse_stts(find(&stbl, b"stts").ok_or("mp4: no stts")?.body)?;
    let sync = match find(&stbl, b"stss") {
        Some(s) => parse_stss(s.body)?,
        None => Vec::new(),
    };
    let ctts = match find(&stbl, b"ctts") {
        Some(c) => parse_ctts(c.body)?,
        None => Vec::new(),
    };
    let samples = build_samples(&SampleTables { sizes, chunk_offsets, stsc, stts, ctts, sync });

    let width = if sw != 0 { sw } else { tk_w };
    let height = if sh != 0 { sh } else { tk_h };
    Ok(Some(VideoTrack { width, height, timescale, duration, config, samples }))
}

/// Parse `stsd` → the first video sample entry's dimensions + decoder config.
///
/// The recognised four-character codes and their configuration boxes:
///
/// | entry            | config | codec |
/// |------------------|--------|-------|
/// | `avc1` / `avc3`  | `avcC` | H.264 |
/// | `hvc1` / `hev1`  | `hvcC` | H.265 |
/// | `vp09`           | `vpcC` | VP9   |
///
/// `hvc1` vs `hev1` and `avc1` vs `avc3` differ only in whether parameter sets
/// may also appear in the samples; both carry them in the config box, so both
/// are read the same way here and in-band sets simply override later.
fn parse_stsd(body: &[u8]) -> Result<(CodecConfig, (u32, u32)), &'static str> {
    let mut r = Reader::new(body);
    r.u32()?; // version/flags
    let entry_count = r.u32()?;
    if entry_count == 0 {
        return Err("mp4: empty stsd");
    }
    // Each entry begins with a box header (size, type).
    let rest = &body[8..];
    for b in boxes(rest) {
        let want: Option<(&[u8; 4], u8)> = match &b.typ {
            b"avc1" | b"avc3" => Some((b"avcC", 0)),
            b"hvc1" | b"hev1" => Some((b"hvcC", 1)),
            b"vp09" => Some((b"vpcC", 2)),
            _ => None,
        };
        let Some((cfg_type, which)) = want else { continue };
        // VisualSampleEntry: 6 reserved + 2 data_ref_idx + 16 predefined/
        // reserved, then width(16) height(16) at offset 24.
        let mut vr = Reader::new(b.body);
        vr.skip(24)?;
        let w = vr.u16()? as u32;
        let h = vr.u16()? as u32;
        // Skip to the child boxes: 14 fixed fields (horiz/vert res, reserved,
        // frame_count) + 32-byte compressorname + depth(2) + predefined(2).
        vr.skip(14 + 32 + 2 + 2)?;
        let child_boxes = boxes(&b.body[vr.p..]);
        let cfg = find(&child_boxes, cfg_type);
        let config = match (which, cfg) {
            (0, Some(c)) => CodecConfig::Avc(parse_avcc(c.body)?),
            (1, Some(c)) => CodecConfig::Hevc(parse_hvcc(c.body)?),
            (2, Some(c)) => CodecConfig::Vp9(parse_vpcc(c.body)?),
            // A `vp09` entry without `vpcC` is still decodable — VP9 frames are
            // self-describing, so the box is metadata. The H.26x entries are
            // not: without their parameter sets there is nothing to decode.
            (2, None) => CodecConfig::Vp9(VpcC::default()),
            _ => return Err("mp4: video sample entry has no decoder config box"),
        };
        return Ok((config, (w, h)));
    }
    Err("mp4: no supported video sample entry (avc1/avc3, hvc1/hev1, vp09)")
}

/// A demuxed audio track: the AAC `AudioSpecificConfig` plus the sample table
/// (each sample is one raw AAC access unit). Enough to feed the AAC-LC decoder.
pub struct AudioTrack {
    pub sample_rate: u32,
    pub channels: u8,
    /// The MPEG-4 `AudioSpecificConfig` (object type, freq index, channel cfg).
    pub asc: Vec<u8>,
    pub timescale: u32,
    pub duration: u64,
    pub samples: Vec<Sample>,
}

impl AudioTrack {
    pub fn duration_ms(&self) -> u64 {
        if self.timescale == 0 { 0 } else { self.duration * 1000 / self.timescale as u64 }
    }
}

/// Parse the file and return its first AAC audio track, if any.
pub fn parse_audio(data: &[u8]) -> Result<Option<AudioTrack>, &'static str> {
    let top = boxes(data);
    let moov = find(&top, b"moov").ok_or("mp4: no moov box")?;
    let moov_boxes = boxes(moov.body);
    for trak in moov_boxes.iter().filter(|b| &b.typ == b"trak") {
        if let Some(t) = parse_trak_audio(trak.body)? {
            return Ok(Some(t));
        }
    }
    Ok(None)
}

fn parse_trak_audio(trak_body: &[u8]) -> Result<Option<AudioTrack>, &'static str> {
    let trak = boxes(trak_body);
    let mdia = match find(&trak, b"mdia") {
        Some(m) => boxes(m.body),
        None => return Ok(None),
    };
    // Handler must be 'soun'.
    if let Some(hdlr) = find(&mdia, b"hdlr") {
        let mut r = Reader::new(hdlr.body);
        r.u32()?; // ver/flags
        r.u32()?; // pre_defined
        if r.take(4)? != b"soun" {
            return Ok(None);
        }
    } else {
        return Ok(None);
    }
    let (mut timescale, mut duration) = (0u32, 0u64);
    if let Some(mdhd) = find(&mdia, b"mdhd") {
        let mut r = Reader::new(mdhd.body);
        if r.u32()? >> 24 == 1 {
            r.skip(16)?;
            timescale = r.u32()?;
            duration = r.u64()?;
        } else {
            r.skip(8)?;
            timescale = r.u32()?;
            duration = r.u32()? as u64;
        }
    }
    let minf = match find(&mdia, b"minf") {
        Some(m) => boxes(m.body),
        None => return Ok(None),
    };
    let stbl = match find(&minf, b"stbl") {
        Some(s) => boxes(s.body),
        None => return Ok(None),
    };
    let stsd = find(&stbl, b"stsd").ok_or("mp4: no stsd")?;
    let (sample_rate, channels, asc) = parse_stsd_audio(stsd.body)?;

    let sizes = parse_stsz(find(&stbl, b"stsz").ok_or("mp4: no stsz")?.body)?;
    let chunk_offsets = if let Some(co64) = find(&stbl, b"co64") {
        parse_offsets(co64.body, true)?
    } else {
        parse_offsets(find(&stbl, b"stco").ok_or("mp4: no stco/co64")?.body, false)?
    };
    let stsc = parse_stsc(find(&stbl, b"stsc").ok_or("mp4: no stsc")?.body)?;
    let stts = parse_stts(find(&stbl, b"stts").ok_or("mp4: no stts")?.body)?;
    // Audio is all keyframes (no stss) — every access unit is independently
    // decodable.
    let samples = build_samples(&SampleTables { sizes, chunk_offsets, stsc, stts, ctts: Vec::new(), sync: Vec::new() });
    Ok(Some(AudioTrack { sample_rate, channels, asc, timescale, duration, samples }))
}

/// Parse `stsd` → the first AAC (`mp4a`) entry's sample rate/channels + the
/// `AudioSpecificConfig` from its `esds` (ISO or QuickTime `wave` nesting).
fn parse_stsd_audio(body: &[u8]) -> Result<(u32, u8, Vec<u8>), &'static str> {
    let mut r = Reader::new(body);
    r.u32()?; // version/flags
    if r.u32()? == 0 {
        return Err("mp4: empty stsd");
    }
    let rest = &body[8..];
    for b in boxes(rest) {
        if &b.typ == b"mp4a" {
            return parse_mp4a_entry(b.body);
        }
    }
    Err("mp4: no mp4a sample entry (unsupported audio codec)")
}

/// Decode one `mp4a` AudioSampleEntry body (ISO v0 or QuickTime v1/v2).
fn parse_mp4a_entry(body: &[u8]) -> Result<(u32, u8, Vec<u8>), &'static str> {
    // SampleEntry: 6 reserved + 2 data_reference_index.
    if body.len() < 28 {
        return Err("mp4: mp4a entry too short");
    }
    // Bytes 8..10 are version for QuickTime SoundDescription; 0 for pure ISO.
    let version = u16::from_be_bytes([body[8], body[9]]);
    let channels = u16::from_be_bytes([body[16], body[17]]) as u8;
    let sample_rate = u32::from_be_bytes([body[24], body[25], body[26], body[27]]) >> 16;
    // ISO v0: children start at 28. QT v1: +16 bytes of packet info. QT v2: +36.
    let mut child_off = 28usize;
    match version {
        0 => {}
        1 => child_off = 28 + 16,
        2 => child_off = 28 + 36,
        _ => {
            // Unknown — try 28 first, then scan for esds.
            child_off = 28;
        }
    }
    if child_off > body.len() {
        child_off = 28.min(body.len());
    }
    let asc = find_esds_asc(&body[child_off..])
        .or_else(|_| find_esds_asc_anywhere(body))
        .map_err(|_| "mp4: no esds/AudioSpecificConfig in mp4a")?;
    Ok((sample_rate, channels.max(1), asc))
}

/// Look for `esds` (or QuickTime `wave` → `esds`) among child boxes of `mp4a`.
fn find_esds_asc(children: &[u8]) -> Result<Vec<u8>, &'static str> {
    let child_boxes = boxes(children);
    if let Some(esds) = find(&child_boxes, b"esds") {
        return parse_esds_asc(esds.body);
    }
    // QuickTime: esds lives under a `wave` atom (alongside `frma`/`mp4a`).
    if let Some(wave) = find(&child_boxes, b"wave") {
        let wave_boxes = boxes(wave.body);
        if let Some(esds) = find(&wave_boxes, b"esds") {
            return parse_esds_asc(esds.body);
        }
        // Some files nest another mp4a inside wave.
        if let Some(inner) = find(&wave_boxes, b"mp4a") {
            if let Ok(a) = find_esds_asc(inner.body) {
                return Ok(a);
            }
        }
    }
    Err("mp4: no esds")
}

/// Last-resort scan: any top-level-looking `esds` box anywhere in the entry.
fn find_esds_asc_anywhere(body: &[u8]) -> Result<Vec<u8>, &'static str> {
    // Walk every 4-byte-aligned "esds" tag candidate.
    let mut i = 0;
    while i + 8 <= body.len() {
        if &body[i + 4..i + 8] == b"esds" {
            let size = u32::from_be_bytes([body[i], body[i + 1], body[i + 2], body[i + 3]]) as usize;
            if size >= 8 && i + size <= body.len() {
                if let Ok(asc) = parse_esds_asc(&body[i + 8..i + size]) {
                    return Ok(asc);
                }
            }
        }
        i += 1;
    }
    Err("mp4: esds not found")
}

/// Expandable MPEG-4 descriptor length (7 bits/byte, MSB=continue).
fn esds_read_len(r: &mut Reader) -> Result<usize, &'static str> {
    let mut len = 0usize;
    for _ in 0..4 {
        let b = r.u8()?;
        len = (len << 7) | (b & 0x7f) as usize;
        if b & 0x80 == 0 {
            break;
        }
    }
    Ok(len)
}

/// Walk the `esds` MPEG-4 descriptor chain and return the `AudioSpecificConfig`
/// (DecoderSpecificInfo, tag 0x05). Tolerates version/flags prefix and nested
/// descriptors that aren't strictly ordered.
fn parse_esds_asc(body: &[u8]) -> Result<Vec<u8>, &'static str> {
    // Skip optional 4-byte version/flags when present (fullbox style).
    let start = if body.len() >= 4 && body[0] == 0 && body[1] == 0 && body[2] == 0 {
        4
    } else {
        0
    };
    // Prefer structured walk; fall back to a linear tag scan for tag 0x05.
    if let Ok(asc) = parse_esds_structured(&body[start..]) {
        return Ok(asc);
    }
    parse_esds_scan(&body[start..])
}

fn parse_esds_structured(body: &[u8]) -> Result<Vec<u8>, &'static str> {
    let mut r = Reader::new(body);
    // ES_Descriptor (tag 0x03) — optional wrapper.
    let tag = r.u8()?;
    if tag == 0x03 {
        esds_read_len(&mut r)?;
        r.skip(2)?; // ES_ID
        let flags = r.u8()?;
        if flags & 0x80 != 0 {
            r.skip(2)?;
        }
        if flags & 0x40 != 0 {
            let url_len = r.u8()? as usize;
            r.skip(url_len)?;
        }
        if flags & 0x20 != 0 {
            r.skip(2)?;
        }
        if r.u8()? != 0x04 {
            return Err("mp4: no DecoderConfigDescriptor");
        }
    } else if tag != 0x04 {
        return Err("mp4: esds unexpected tag");
    }
    esds_read_len(&mut r)?;
    r.skip(1 + 1 + 3 + 4 + 4)?; // objectType, streamType, bufferSize, max/avg bitrate
    if r.u8()? != 0x05 {
        return Err("mp4: no DecoderSpecificInfo (AudioSpecificConfig)");
    }
    let len = esds_read_len(&mut r)?;
    Ok(r.take(len)?.to_vec())
}

/// Linear scan for tag 0x05 (DecoderSpecificInfo) — handles odd descriptor
/// layouts some remuxers produce.
fn parse_esds_scan(body: &[u8]) -> Result<Vec<u8>, &'static str> {
    let mut i = 0;
    while i < body.len() {
        let tag = body[i];
        i += 1;
        if i >= body.len() {
            break;
        }
        // Expandable length
        let mut len = 0usize;
        for _ in 0..4 {
            if i >= body.len() {
                return Err("mp4: esds truncated");
            }
            let b = body[i];
            i += 1;
            len = (len << 7) | (b & 0x7f) as usize;
            if b & 0x80 == 0 {
                break;
            }
        }
        if tag == 0x05 {
            if i + len <= body.len() && len > 0 && len < 64 {
                return Ok(body[i..i + len].to_vec());
            }
            return Err("mp4: bad ASC length in esds");
        }
        // Skip this descriptor's payload and continue scanning inside it too
        // (nested) by not jumping — just advance past header; nested tags will
        // be seen as we walk byte-by-byte only at tag boundaries. Jump payload:
        if i + len > body.len() {
            break;
        }
        // Recurse into payload for nested descriptors.
        if let Ok(asc) = parse_esds_scan(&body[i..i + len]) {
            return Ok(asc);
        }
        i += len;
    }
    Err("mp4: AudioSpecificConfig not found in esds")
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test_case]
    fn box_walker_handles_sizes_and_children() {
        // A tiny tree: ftyp(8 bytes body "isom") then moov containing one child
        // "abcd" with a 4-byte body.
        let mut f = Vec::new();
        let push_box = |f: &mut Vec<u8>, typ: &[u8; 4], body: &[u8]| {
            let size = (8 + body.len()) as u32;
            f.extend_from_slice(&size.to_be_bytes());
            f.extend_from_slice(typ);
            f.extend_from_slice(body);
        };
        push_box(&mut f, b"ftyp", b"isom");
        // moov body = one child box
        let mut moov_body = Vec::new();
        push_box(&mut moov_body, b"abcd", &[1, 2, 3, 4]);
        push_box(&mut f, b"moov", &moov_body);

        let top = boxes(&f);
        assert_eq!(top.len(), 2);
        assert_eq!(&top[0].typ, b"ftyp");
        assert_eq!(top[0].body, b"isom");
        let children = boxes(top[1].body);
        assert_eq!(children.len(), 1);
        assert_eq!(&children[0].typ, b"abcd");
        assert_eq!(children[0].body, &[1, 2, 3, 4]);
    }

    #[test_case]
    fn build_samples_single_chunk_all_sync() {
        // 3 samples of sizes 10,20,30 in one chunk at offset 100; 1 stsc run
        // (chunk 0 → 3 samples); stts one run (3 samples, delta 40); no stss.
        let t = SampleTables {
            sizes: vec![10, 20, 30],
            chunk_offsets: vec![100],
            stsc: vec![(0, 3)],
            stts: vec![(3, 40)],
            ctts: vec![],
            sync: vec![],
        };
        let s = build_samples(&t);
        assert_eq!(s.len(), 3);
        assert_eq!(s[0], Sample { offset: 100, size: 10, dts: 0, cts: 0, is_sync: true });
        assert_eq!(s[1], Sample { offset: 110, size: 20, dts: 40, cts: 40, is_sync: true });
        assert_eq!(s[2], Sample { offset: 130, size: 30, dts: 80, cts: 80, is_sync: true });
    }

    #[test_case]
    fn build_samples_multi_chunk_with_stss() {
        // 4 samples across 2 chunks (2 each) at offsets 0 and 500; only sample
        // 1 (1-based) is a sync sample.
        let t = SampleTables {
            sizes: vec![5, 5, 5, 5],
            chunk_offsets: vec![0, 500],
            stsc: vec![(0, 2)],
            stts: vec![(4, 100)],
            ctts: vec![],
            sync: vec![1],
        };
        let s = build_samples(&t);
        assert_eq!(s.len(), 4);
        assert_eq!(s[0].offset, 0);
        assert!(s[0].is_sync);
        assert_eq!(s[1].offset, 5);
        assert!(!s[1].is_sync);
        assert_eq!(s[2].offset, 500); // second chunk
        assert!(!s[2].is_sync);
        assert_eq!(s[3].offset, 505);
    }

    #[test_case]
    fn avcc_parse_extracts_sps_pps() {
        // A minimal avcC: version=1, profile/compat/level=66/0/11, lenSize=4
        // (0xff → 0x03+1), 1 SPS of 3 bytes, 1 PPS of 2 bytes.
        let body = [
            0x01, 66, 0x00, 11, 0xff, // config + lengthSizeMinusOne=3
            0xe1, // reserved(3 bits)=111 + numSPS=1
            0x00, 0x03, 0x67, 0xaa, 0xbb, // SPS len 3
            0x01, // numPPS=1
            0x00, 0x02, 0x68, 0xcc, // PPS len 2
        ];
        let a = parse_avcc(&body).unwrap();
        assert_eq!(a.length_size, 4);
        assert_eq!(a.sps.len(), 1);
        assert_eq!(a.sps[0], vec![0x67, 0xaa, 0xbb]);
        assert_eq!(a.pps.len(), 1);
        assert_eq!(a.pps[0], vec![0x68, 0xcc]);
    }

    #[test_case]
    fn esds_extracts_audio_specific_config() {
        // The real `esds` body from a stereo 44.1 kHz AAC-LC mp4 (echo-hereweare):
        // ES_Descriptor → DecoderConfig (objType 0x40 = AAC) → DecoderSpecificInfo
        // whose AudioSpecificConfig is 0x1210 (AOT 2 = LC, freq idx 4 = 44100,
        // channel cfg 2 = stereo). Exercises the expandable-length descriptor walk.
        let body = [
            0x00, 0x00, 0x00, 0x00, // version/flags
            0x03, 0x80, 0x80, 0x80, 0x22, 0x00, 0x00, 0x00, // ES_Descriptor tag+len, ES_ID, flags
            0x04, 0x80, 0x80, 0x80, 0x14, // DecoderConfigDescriptor tag+len
            0x40, 0x15, 0x00, 0x18, 0x00, 0x00, 0x00, 0x01, 0xf4, 0x00, 0x00, 0x01, 0xf4, // cfg fields
            0x05, 0x80, 0x80, 0x80, 0x02, 0x12, 0x10, // DecoderSpecificInfo (ASC=0x1210)
            0x06, 0x80, 0x80, 0x80, 0x01, 0x02, // SLConfigDescriptor
        ];
        assert_eq!(parse_esds_asc(&body).unwrap(), vec![0x12, 0x10]);
    }
}
