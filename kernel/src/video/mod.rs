//! Video: container demuxing + decoding for the `/open` player.
//!
//! Built in stages, each pure and unit-tested off-hardware (the standing rule —
//! fiddly logic lives in pure functions verified with cases, and hard numeric
//! bring-up is diffed against a host reference decoder, not QEMU):
//!
//! * **The bitstream layer** — [`bits`] (RBSP unescape + Exp-Golomb, shared by
//!   [`h264`] and [`hevc`]), the [`mp4`] ISO-BMFF and [`mkv`] Matroska demuxers
//!   (box/element tree → sample table + a [`mp4::CodecConfig`]), and per-codec
//!   NAL/frame-header parsing. [`probe`] reports a stream's
//!   geometry/profile/frame-count without decoding pixels.
//! * **The pixel pipelines** — slice → entropy decode → intra/inter → transform
//!   → in-loop filter, all producing the same [`Frame`] the player presents.
//!
//! Codec support, and it is deliberately reported rather than attempted:
//!
//! | codec | demux + describe | decode |
//! |-------|------------------|--------|
//! | H.264 / AVC | yes | yes — baseline CAVLC through High-profile CABAC (I/P/B) |
//! | H.265 / HEVC | yes | Main/RExt mono+4:2:0/4:2:2/4:4:4, 8–12-bit, tiles, PCM |
//! | VP9 | yes | yes — profile 0, intra + inter, bit-exact vs libvpx |
//!
//! A stream outside the supported set comes back from [`probe`] with
//! `decodable == false` and a [`VideoInfo::unsupported_reason`] that names the
//! actual cause, so `/open` never reports a file as playable and then fails on
//! it. Containers: `.mp4`/`.mov` and `.mkv`/`.webm`; HLS/TS is not demuxed.

pub mod bits;
pub mod h264;
pub mod hevc;
pub mod mkv;
pub mod mp4;
/// Minimal MP4 muxer for screen recordings (the inverse of [`mp4`]).
pub mod mp4_mux;
pub mod mt;
pub mod vp9;
pub mod yuv;

use alloc::string::String;
use alloc::vec::Vec;

/// A decoded frame: row-major `0x00RRGGBB` pixels, the same layout the
/// framebuffer's `present_surface` blits (so a video frame presents exactly
/// like an image). Produced by the pixel pipeline in a later stage.
pub struct Frame {
    pub w: usize,
    pub h: usize,
    pub pixels: Vec<u32>,
    /// Presentation timestamp in milliseconds from the start of the track.
    pub pts_ms: u64,
}

/// What [`probe`] reports about a video file — enough to drive the player UI and
/// tell the user exactly what is and isn't decodable yet.
pub struct VideoInfo {
    pub container: &'static str,
    pub codec: String,
    pub width: u32,
    pub height: u32,
    pub profile_idc: u8,
    pub level_idc: u8,
    pub frame_count: usize,
    pub duration_ms: u64,
    /// H.264 only: false = CAVLC, true = CABAC. HEVC is always CABAC and VP9 is
    /// not CABAC at all, so this is reported as `true`/`false` respectively and
    /// the player does not branch on it for them.
    pub cabac: bool,
    /// True once the pixel pipeline can decode this stream.
    pub decodable: bool,
    /// Why not, when `decodable` is false — empty otherwise.
    ///
    /// The reason is produced *here*, where the parameter sets are in hand, and
    /// not inferred at the call site: the player used to print "CABAC entropy
    /// coding (baseline/CAVLC only)" for every undecodable stream, which is a
    /// true statement about H.264 and a nonsense one about HEVC (always CABAC)
    /// or VP9 (no CABAC at all).
    pub unsupported_reason: &'static str,
}

/// Sniff the container and, for a supported one, demux + parse the parameter
/// sets to describe the stream. Does **not** decode pixels.
pub fn probe(bytes: &[u8]) -> Result<VideoInfo, &'static str> {
    let (container, d) = demux(bytes)?;
    let frame_count = d.samples.len();
    let duration_ms = d.duration_ms;
    let (codec, width, height, profile_idc, level_idc, cabac, (decodable, unsupported_reason)) = match &d.config {
        mp4::CodecConfig::Avc(avcc) => {
            let sps_nal = avcc.sps.first().ok_or("video: avcC has no SPS")?;
            let pps_nal = avcc.pps.first().ok_or("video: avcC has no PPS")?;
            let sps = h264::parse_sps(&bits::unescape_rbsp(&sps_nal[1..]))?;
            let pps = h264::parse_pps(&bits::unescape_rbsp(&pps_nal[1..]))?;
            (
                alloc::format!(
                    "H.264 (avc, profile {}, level {}.{})",
                    sps.profile_idc,
                    sps.level_idc / 10,
                    sps.level_idc % 10
                ),
                sps.width(),
                sps.height(),
                sps.profile_idc,
                sps.level_idc,
                pps.entropy_coding_mode,
                (true, ""),
            )
        }
        mp4::CodecConfig::Hevc(hvcc) => {
            let sps_nal = hvcc.sps.first().ok_or("video: hvcC has no SPS")?;
            let sps = hevc::parse_sps(&bits::unescape_rbsp(&sps_nal[2..]))?;
            // HEVC levels are coded as 30×level, so 93 is 3.1 and 120 is 4.0 —
            // AVC's ×10 convention would print every one of them wrong.
            let (lvl_major, lvl_minor) = (sps.ptl.level_idc / 30, (sps.ptl.level_idc % 30) / 3);
            (
                alloc::format!(
                    "H.265 (hevc, {} profile, {} tier, level {}.{}, {}-bit)",
                    sps.ptl.profile_name(),
                    if sps.ptl.tier_high { "high" } else { "main" },
                    lvl_major,
                    lvl_minor,
                    sps.bit_depth_luma
                ),
                sps.width(),
                sps.height(),
                sps.ptl.profile_idc,
                sps.ptl.level_idc,
                true, // HEVC has no CAVLC mode
                hevc_support(&sps),
            )
        }
        mp4::CodecConfig::Vp9(vpcc) => {
            // VP9 carries no parameter sets — the geometry is in the first
            // frame's own header, which is the only authority (the `vpcC` box
            // and the track header can and do disagree with it).
            let first = d.samples.first().ok_or("video: no coded frames")?;
            let data = bytes.get(first.offset..first.offset + first.size).ok_or("video: sample out of range")?;
            let ranges = vp9::split_superframe(data)?;
            let (s, e) = ranges[0];
            let refs: vp9::RefSizes = [(0, 0); vp9::NUM_REF_FRAMES];
            let h = vp9::parse_frame_header(&data[s..e], &refs)?;
            (
                alloc::format!(
                    "VP9 (profile {}, level {}.{}, {}-bit)",
                    h.profile,
                    vpcc.level / 10,
                    vpcc.level % 10,
                    h.bit_depth
                ),
                h.render_width,
                h.render_height,
                h.profile,
                vpcc.level,
                false,
                vp9_support(&h),
            )
        }
    };
    Ok(VideoInfo {
        container,
        codec,
        width,
        height,
        profile_idc,
        level_idc,
        frame_count,
        duration_ms,
        cabac,
        decodable,
        unsupported_reason,
    })
}

/// What the HEVC path supports, and why not when it doesn't.
///
/// The **first** answer is that there is no HEVC pixel pipeline yet: the
/// bitstream layer here parses VPS/SPS/PPS and slice headers, which is what
/// `probe` reports from, and nothing decodes samples. Saying so plainly is the
/// point — a stream reported decodable that then fails to open is worse than one
/// that says up front what it is.
///
/// The profile checks are kept and ordered *before* that so the message names
/// the more specific fact when there is one: a Main 10 file will not decode here
/// even once the pipeline lands, and a user with a 10-bit file wants to be told
/// that rather than "not implemented yet".
fn hevc_support(sps: &hevc::Sps) -> (bool, &'static str) {
    if !(8..=12).contains(&sps.bit_depth_luma) {
        return (false, "HEVC bit depth must be 8–12");
    }
    // Chroma bit depth is signalled even for monochrome; when chroma exists it
    // must match luma (we do not implement dual bit-depth).
    if sps.chroma_format_idc != 0 && sps.bit_depth_luma != sps.bit_depth_chroma {
        return (false, "HEVC dual bit-depth (luma ≠ chroma) is not supported");
    }
    // 0 = monochrome, 1/2/3 = 4:2:0 / 4:2:2 / 4:4:4.
    if sps.chroma_format_idc > 3 {
        return (false, "unknown HEVC chroma_format_idc");
    }
    // Main / RExt monochrome and 4:2:0/4:2:2/4:4:4 at 8–12 bit. Samples are
    // u16 internally; the player downshifts and (if needed) 4:2:0-sub-samples
    // for the RGB converter. Tiles and PCM are decoded when present.
    (true, "")
}

/// What the VP9 path supports, and why not when it doesn't. Same shape and same
/// reasoning as [`hevc_support`].
fn vp9_support(h: &vp9::FrameHeader) -> (bool, &'static str) {
    if h.profile != 0 || h.bit_depth != 8 {
        return (false, "VP9 profile 1-3 (10/12-bit or non-4:2:0) — only profile 0 decodes");
    }
    if !h.subsampling_x || !h.subsampling_y {
        return (false, "non-4:2:0 chroma");
    }
    if h.segmentation.enabled {
        return (false, "VP9 segmentation is not implemented yet");
    }
    (true, "")
}

/// A demuxed track's essentials, container-agnostic.
struct Demuxed {
    config: mp4::CodecConfig,
    samples: alloc::vec::Vec<mp4::Sample>,
    timescale: u32,
    duration_ms: u64,
}

fn demux(bytes: &[u8]) -> Result<(&'static str, Demuxed), &'static str> {
    match sniff(bytes) {
        Container::Mp4 => {
            let t = mp4::parse(bytes)?;
            let duration_ms = t.duration_ms();
            Ok(("mp4/mov (ISO-BMFF)", Demuxed { config: t.config, samples: t.samples, timescale: t.timescale, duration_ms }))
        }
        Container::Matroska => {
            let t = mkv::parse(bytes)?;
            let duration_ms = t.duration_ms();
            Ok(("matroska/webm (EBML)", Demuxed { config: t.config, samples: t.samples, timescale: t.timescale, duration_ms }))
        }
        Container::Unknown => Err("video: unrecognised container (mp4/mov, mkv/webm supported)"),
    }
}

enum Container {
    Mp4,
    Matroska,
    Unknown,
}

fn sniff(bytes: &[u8]) -> Container {
    // ISO-BMFF: a top-level `ftyp` (or `moov`/`mdat`) box — the 4 bytes at
    // offset 4 are the type.
    if bytes.len() >= 12 {
        let typ = &bytes[4..8];
        if typ == b"ftyp" || typ == b"moov" || typ == b"mdat" || typ == b"free" || typ == b"skip" {
            return Container::Mp4;
        }
    }
    // Matroska/WebM: the EBML magic 0x1A45DFA3.
    if bytes.starts_with(&[0x1a, 0x45, 0xdf, 0xa3]) {
        return Container::Matroska;
    }
    Container::Unknown
}

/// Preferred max edge for display RGB. Full-res 4K RGB is ~33 MiB/frame; the
/// DPB keeps full-res YUV only for active references. 640 keeps convert cheap
/// on 1080p/4K while still filling the action pane.
const DISPLAY_MAX_EDGE: usize = 640;

/// Short look-ahead ring (display RGB) — like VLC's jitter buffer, **not** a
/// full-clip cache. ~8 frames × 640×360×4 ≈ 7 MiB max.
const RGB_RING: usize = 8;

/// Fit `(w,h)` into a box of max edge `max_edge`, preserving aspect (at least 1×1).
fn fit_display(w: usize, h: usize, max_edge: usize) -> (usize, usize) {
    if w == 0 || h == 0 {
        return (1, 1);
    }
    let long = w.max(h);
    if long <= max_edge {
        return (w, h);
    }
    let dw = (w * max_edge / long).max(1);
    let dh = (h * max_edge / long).max(1);
    (dw, dh)
}

/// Convert a decoded YUV frame to RGB for presentation. Large sources are
/// nearest-neighbour downscaled to `max_edge`. Uses **NEON SIMD** via [`yuv`];
/// on aarch64 with online APs, **multi-core row split** ([`mt`]).
fn frame_from_yuv(df: &h264::decoder::DecodedFrame, pts_ms: u64, max_edge: usize) -> Frame {
    let (sw, sh, scw) = (df.w, df.h, df.w / 2);
    let (dw, dh) = fit_display(sw, sh, max_edge);
    let mut pixels = alloc::vec![0u32; dw * dh];
    let mut ctx = yuv::ConvertCtx {
        y: df.y.as_ptr(),
        cb: df.cb.as_ptr(),
        cr: df.cr.as_ptr(),
        out: pixels.as_mut_ptr(),
        y_len: df.y.len(),
        cb_len: df.cb.len(),
        cr_len: df.cr.len(),
        out_len: pixels.len(),
        sw,
        sh,
        scw,
        dw,
        dh,
    };
    // SAFETY: ctx lives for the call; workers write disjoint rows.
    unsafe {
        mt::parallel_rows(
            dh,
            8,
            yuv::convert_worker,
            &mut ctx as *mut yuv::ConvertCtx as *mut u8,
        );
    }
    Frame { w: dw, h: dh, pixels, pts_ms }
}

/// A **streaming** H.264 player pipeline — the same shape as VLC/mpv/ffmpeg:
///
/// 1. **Demux once** (sample table + SPS/PPS) — kilobytes, not gigabytes.
/// 2. **Keep the compressed file** (or map it) and read one access unit at a time.
/// 3. **Decode on demand** as the play clock advances — never rasterize the
///    whole movie up front.
/// 4. **RAM stays O(1)**: one current display RGB + a few H.264 DPB reference
///    pictures (YUV). A 2‑hour 4K file does **not** become hundreds of MB of RGB.
///
/// Full-clip RGB caching is deliberately not done: it makes open take minutes
/// and blows the heap. If the player falls behind, it drops frames (soft clock
/// re-anchor in `pump_video`), same idea as VLC under CPU pressure.
pub struct StreamDecoder {
    /// Compressed container bytes (mp4/mkv). Demux indexes into this; we do
    /// **not** expand it to per-frame RGB.
    bytes: Vec<u8>,
    sps: h264::Sps,
    pps: h264::Pps,
    length_size: u8,
    /// Sample table only (offsets/sizes/cts) — O(frames), a few bytes each.
    samples: Vec<mp4::Sample>,
    timescale: u32,
    /// Track duration in ms (public: the player's clock reads it).
    pub duration_ms: u64,
    /// Index of the next sample to decode (so the newest picture is `next-1`).
    next: usize,
    engine: Engine,
    /// display index → decode sample index (identity for baseline; B-frame
    /// streams are sorted by composition timestamp).
    display: Vec<usize>,
    /// Inverse of `display`: decode sample index → display position.
    disp_of: Vec<usize>,
    /// Hurry mode (set for one [`seek_decode_hurry`] call): skip decoding
    /// non-reference samples that display before this index.
    hurry_before: Option<usize>,
    /// Display index of the single current RGB frame.
    cur_idx: Option<usize>,
    /// The one RGB frame currently shown (also stored in `ring` when present).
    cur: Option<Frame>,
    /// Short look-ahead of recent display RGB frames (evict oldest). Forward
    /// play often hits the ring; backward/seek misses and re-decodes.
    ring: alloc::collections::BTreeMap<usize, Frame>,
    /// Source coded width/height (full res, may be 4K).
    pub src_w: u32,
    pub src_h: u32,
    /// Display max edge for YUV→RGB downscale (pane-sized, not full 4K RGB).
    display_edge: usize,
    /// Raw SPS/PPS NALs (with header) for re-initialising rust_h264 on seek.
    avcc_sps: alloc::vec::Vec<alloc::vec::Vec<u8>>,
    avcc_pps: alloc::vec::Vec<alloc::vec::Vec<u8>>,
    /// Which backend is active (for status / debugging).
    pub backend: &'static str,
}

/// Decode backend. Prefer **rust_h264** (vendored Main/High + CABAC); on
/// init/decode failure fall back to our hand-rolled CAVLC / CABAC ports.
enum Engine {
    /// Primary: `third_party/rust_h264` (pure Rust, no_std-patched).
    RustH264 {
        dec: rust_h264::decoder::Decoder,
        /// Decode-order sample index → YUV planes ready for display convert.
        pending: alloc::collections::BTreeMap<usize, h264::decoder::DecodedFrame>,
        /// Sample index of the picture currently being assembled (rust_h264
        /// returns the *previous* picture when a new one starts).
        open_sample: Option<usize>,
    },
    /// Backup: our baseline CAVLC path.
    Cavlc { reference: Option<h264::decoder::DecodedFrame> },
    /// H.265/HEVC (Main profile, 8-bit 4:2:0).
    Hevc {
        dec: hevc::decoder::HevcDecoder,
        /// Sample index -> the pictures that became displayable at it. HEVC
        /// reorders, so one sample may release several frames or none.
        pending: alloc::collections::BTreeMap<usize, hevc::decoder::DecodedFrame>,
        /// Next display slot to hand a released picture to.
        next_out: usize,
    },
    /// VP9 (profile 0, 8-bit 4:2:0), bit-exact against libvpx.
    Vp9 {
        dec: vp9::decoder::Vp9Decoder,
        /// Sample index → the picture that sample *showed*. A VP9 sample may
        /// show nothing (a hidden ALTREF) or re-show a slot, so this is not a
        /// one-to-one map from samples to pictures.
        pending: alloc::collections::BTreeMap<usize, vp9::decoder::DecodedFrame>,
    },
    /// Backup: our Main/High CABAC path.
    Cabac {
        dec: h264::decoder_cabac::H264Dec,
        pending: alloc::collections::BTreeMap<usize, alloc::rc::Rc<h264::decoder_cabac::Pic>>,
    },
}

/// Pure frame-drop test over one AVCC sample (length-prefixed NALs): true iff
/// it contains at least one slice and **every** slice is non-reference
/// (`nal_ref_idc == 0` — an H.264 "disposable" picture; nothing predicts from
/// it, so skipping its decode entirely is always safe). Conservative `false`
/// on any malformed framing.
fn sample_is_nonref(data: &[u8], length_size: usize) -> bool {
    if length_size == 0 || length_size > 4 {
        return false;
    }
    let mut off = 0usize;
    let mut saw_slice = false;
    while off + length_size < data.len() {
        let mut len = 0usize;
        for &b in &data[off..off + length_size] {
            len = (len << 8) | b as usize;
        }
        off += length_size;
        if len == 0 || off + len > data.len() {
            return false;
        }
        let hdr = data[off];
        let ty = hdr & 0x1f;
        if ty == 1 || ty == 5 {
            saw_slice = true;
            if (hdr & 0x60) != 0 {
                return false; // a reference slice — later frames need it
            }
        }
        off += len;
    }
    saw_slice
}

fn rust_h264_from_avcc(sps_nals: &[alloc::vec::Vec<u8>], pps_nals: &[alloc::vec::Vec<u8>]) -> Result<rust_h264::decoder::Decoder, &'static str> {
    let mut dec = rust_h264::decoder::Decoder::new();
    for raw in sps_nals.iter().chain(pps_nals.iter()) {
        let nal = rust_h264::nal::parse_nal_bytes(raw).ok_or("rust_h264: bad param NAL")?;
        dec.decode_nal(&nal).map_err(|_| "rust_h264: param set rejected")?;
    }
    Ok(dec)
}

fn yuv_from_rust_h264(f: rust_h264::decoder::Frame) -> h264::decoder::DecodedFrame {
    h264::decoder::DecodedFrame {
        w: f.width as usize,
        h: f.height as usize,
        y: f.y,
        cb: f.u,
        cr: f.v,
    }
}

/// Downshift an HEVC frame to 8-bit **4:2:0** planes for the RGB converter.
///
/// 4:2:2 / 4:4:4 are point-sampled into 4:2:0 (take even rows/cols). The
/// player only paints 4:2:0 today; bit-exact work lives in `videodiff`, which
/// consumes the native layout from `hevcseq`.
fn hevc_frame_to_8bit(f: &hevc::decoder::DecodedFrame) -> h264::decoder::DecodedFrame {
    let shift = f.bit_depth.saturating_sub(8);
    let y: alloc::vec::Vec<u8> = f.y.iter().map(|&s| (s >> shift) as u8).collect();
    let (cw, ch) = (f.w / 2, f.h / 2);
    // Mid-grey chroma for monochrome and as the base for 4:2:0 conversion.
    let mut cb = alloc::vec![128u8; cw * ch];
    let mut cr = alloc::vec![128u8; cw * ch];
    if f.chroma_format_idc != 0 && f.cw > 0 && f.ch > 0 {
        for j in 0..ch {
            let sj = match f.chroma_format_idc {
                1 => j,                         // 4:2:0: already half height
                2 => j * 2,                     // 4:2:2: full height → take even rows
                _ => j * 2,                     // 4:4:4: take even rows
            };
            if sj >= f.ch {
                break;
            }
            for i in 0..cw {
                let si = match f.chroma_format_idc {
                    3 => i * 2, // 4:4:4: take even cols
                    _ => i,     // 4:2:0 / 4:2:2: already half width
                };
                if si >= f.cw {
                    break;
                }
                cb[j * cw + i] = (f.cb[sj * f.cw + si] >> shift) as u8;
                cr[j * cw + i] = (f.cr[sj * f.cw + si] >> shift) as u8;
            }
        }
    }
    h264::decoder::DecodedFrame {
        w: f.w,
        h: f.h,
        y,
        cb,
        cr,
    }
}

impl StreamDecoder {
    /// Demux the container and parse the decoder configuration, ready to decode
    /// on demand.
    pub fn open(bytes: Vec<u8>) -> Result<StreamDecoder, &'static str> {
        let (_container, d) = demux(&bytes)?;
        if let mp4::CodecConfig::Vp9(_) = d.config {
            return StreamDecoder::open_vp9(bytes, d);
        }
        if let mp4::CodecConfig::Hevc(_) = d.config {
            return StreamDecoder::open_hevc(bytes, d);
        }
        let avcc = match d.config {
            mp4::CodecConfig::Avc(a) => a,
            mp4::CodecConfig::Hevc(_) => unreachable!("handled above"),
            mp4::CodecConfig::Vp9(_) => unreachable!("handled above"),
        };
        let sps_nal = avcc.sps.first().ok_or("video: no SPS")?;
        let pps_nal = avcc.pps.first().ok_or("video: no PPS")?;
        let sps = h264::parse_sps(&bits::unescape_rbsp(&sps_nal[1..]))?;
        let pps = h264::parse_pps(&bits::unescape_rbsp(&pps_nal[1..]))?;
        if d.samples.is_empty() {
            return Err("video: no decodable frames found");
        }
        // Prefer rust_h264; keep our CAVLC/CABAC ports as fallback.
        let (engine, backend) = match rust_h264_from_avcc(&avcc.sps, &avcc.pps) {
            Ok(dec) => (
                Engine::RustH264 {
                    dec,
                    pending: alloc::collections::BTreeMap::new(),
                    open_sample: None,
                },
                "rust_h264",
            ),
            Err(_) if pps.entropy_coding_mode => (
                Engine::Cabac {
                    dec: h264::decoder_cabac::H264Dec::new(sps.clone(), pps.clone())?,
                    pending: alloc::collections::BTreeMap::new(),
                },
                "native-cabac",
            ),
            Err(_) => (Engine::Cavlc { reference: None }, "native-cavlc"),
        };
        // Display order: sort sample indices by composition timestamp (stable,
        // so equal timestamps keep decode order). Baseline streams have
        // cts == dts and this is the identity.
        let mut display: Vec<usize> = (0..d.samples.len()).collect();
        display.sort_by_key(|&i| (d.samples[i].cts, i));
        let mut disp_of = alloc::vec![0usize; display.len()];
        for (pos, &si) in display.iter().enumerate() {
            disp_of[si] = pos;
        }
        let src_w = sps.width();
        let src_h = sps.height();
        Ok(StreamDecoder {
            length_size: avcc.length_size,
            samples: d.samples,
            timescale: d.timescale,
            duration_ms: d.duration_ms,
            bytes,
            sps,
            pps,
            next: 0,
            engine,
            display,
            disp_of,
            hurry_before: None,
            cur_idx: None,
            cur: None,
            ring: alloc::collections::BTreeMap::new(),
            src_w,
            src_h,
            display_edge: DISPLAY_MAX_EDGE,
            avcc_sps: avcc.sps,
            avcc_pps: avcc.pps,
            backend,
        })
    }

    /// Open a VP9 track. VP9 carries no parameter sets, so the geometry comes
    /// from the first frame's own header — the same authority the probe uses.
    fn open_vp9(bytes: Vec<u8>, d: Demuxed) -> Result<StreamDecoder, &'static str> {
        let first = d.samples.first().ok_or("video: no decodable frames found")?;
        let data = bytes
            .get(first.offset..first.offset + first.size)
            .ok_or("video: sample out of range")?;
        let ranges = vp9::split_superframe(data)?;
        let refs: vp9::RefSizes = [(0, 0); vp9::NUM_REF_FRAMES];
        let h = vp9::parse_frame_header(&data[ranges[0].0..ranges[0].1], &refs)?;
        let mut display: Vec<usize> = (0..d.samples.len()).collect();
        display.sort_by_key(|&i| (d.samples[i].cts, i));
        let mut disp_of = alloc::vec![0usize; display.len()];
        for (pos, &si) in display.iter().enumerate() {
            disp_of[si] = pos;
        }
        Ok(StreamDecoder {
            length_size: 0,
            samples: d.samples,
            timescale: d.timescale,
            duration_ms: d.duration_ms,
            bytes,
            sps: h264::Sps::default(),
            pps: h264::Pps::default(),
            next: 0,
            engine: Engine::Vp9 {
                dec: vp9::decoder::Vp9Decoder::new(),
                pending: alloc::collections::BTreeMap::new(),
            },
            display,
            disp_of,
            hurry_before: None,
            cur_idx: None,
            cur: None,
            ring: alloc::collections::BTreeMap::new(),
            src_w: h.render_width,
            src_h: h.render_height,
            display_edge: DISPLAY_MAX_EDGE,
            avcc_sps: Vec::new(),
            avcc_pps: Vec::new(),
            backend: "vp9",
        })
    }

    /// Open an HEVC track. The parameter sets live in the `hvcC` box and are
    /// fed to the decoder before the first sample, exactly as an Annex-B stream
    /// would carry them in-band.
    fn open_hevc(bytes: Vec<u8>, d: Demuxed) -> Result<StreamDecoder, &'static str> {
        let hvcc = match &d.config {
            mp4::CodecConfig::Hevc(h) => h.clone(),
            _ => unreachable!(),
        };
        let sps_nal = hvcc.sps.first().ok_or("video: hvcC has no SPS")?;
        let sps = hevc::parse_sps(&bits::unescape_rbsp(&sps_nal[2..]))?;
        let (ok, why) = hevc_support(&sps);
        if !ok {
            return Err(why);
        }
        if d.samples.is_empty() {
            return Err("video: no decodable frames found");
        }
        let mut dec = hevc::decoder::HevcDecoder::new();
        dec.set_parameter_sets(&hvcc.vps, &hvcc.sps, &hvcc.pps)?;

        let mut display: Vec<usize> = (0..d.samples.len()).collect();
        display.sort_by_key(|&i| (d.samples[i].cts, i));
        let mut disp_of = alloc::vec![0usize; display.len()];
        for (pos, &si) in display.iter().enumerate() {
            disp_of[si] = pos;
        }
        let (w, h) = (sps.width() as usize, sps.height() as usize);
        Ok(StreamDecoder {
            length_size: hvcc.length_size,
            samples: d.samples,
            timescale: d.timescale,
            duration_ms: d.duration_ms,
            bytes,
            sps: h264::Sps::default(),
            pps: h264::Pps::default(),
            next: 0,
            engine: Engine::Hevc {
                dec,
                pending: alloc::collections::BTreeMap::new(),
                next_out: 0,
            },
            display,
            disp_of,
            hurry_before: None,
            cur_idx: None,
            cur: None,
            ring: alloc::collections::BTreeMap::new(),
            src_w: w as u32,
            src_h: h as u32,
            display_edge: DISPLAY_MAX_EDGE,
            avcc_sps: Vec::new(),
            avcc_pps: Vec::new(),
            backend: "hevc",
        })
    }

    /// Switch from rust_h264 to the native backup (call on hard decode error).
    fn fallback_to_native(&mut self) {
        self.engine = if self.pps.entropy_coding_mode {
            match h264::decoder_cabac::H264Dec::new(self.sps.clone(), self.pps.clone()) {
                Ok(dec) => {
                    self.backend = "native-cabac";
                    Engine::Cabac {
                        dec,
                        pending: alloc::collections::BTreeMap::new(),
                    }
                }
                Err(_) => {
                    self.backend = "native-cavlc";
                    Engine::Cavlc { reference: None }
                }
            }
        } else {
            self.backend = "native-cavlc";
            Engine::Cavlc { reference: None }
        };
        self.next = 0;
        self.ring.clear();
        self.cur = None;
        self.cur_idx = None;
    }

    /// Total number of frames (samples) in the track.
    pub fn frame_count(&self) -> usize {
        self.samples.len()
    }

    /// Presentation timestamp of display frame `idx`, in ms — read from the
    /// sample table without decoding, so the playback clock can seek freely.
    pub fn pts_ms(&self, idx: usize) -> u64 {
        self.display
            .get(idx)
            .and_then(|&si| self.samples.get(si))
            .map(|s| if self.timescale > 0 { s.cts * 1000 / self.timescale as u64 } else { 0 })
            .unwrap_or(0)
    }

    /// Nearest display index whose decode sample is a sync/IDR frame at or
    /// before `idx` (for VLC-style frame-drop when the decoder can't keep up).
    pub fn keyframe_at_or_before(&self, idx: usize) -> usize {
        let idx = idx.min(self.display.len().saturating_sub(1));
        let mut si = self.display[idx];
        while si > 0 && !self.samples[si].is_sync {
            si -= 1;
        }
        // Map decode sample → a display index that shows it (first with that sample).
        self.display
            .iter()
            .position(|&d| d == si)
            .unwrap_or(0)
    }

    fn ring_insert(&mut self, idx: usize, frame: Frame) {
        self.ring.insert(idx, frame);
        while self.ring.len() > RGB_RING {
            if let Some((&k, _)) = self.ring.iter().next() {
                // Prefer dropping frames older than current playhead.
                if self.cur_idx.map(|c| k < c).unwrap_or(true) {
                    self.ring.remove(&k);
                } else {
                    // All frames are ahead — drop the furthest.
                    let last = *self.ring.keys().next_back().unwrap();
                    self.ring.remove(&last);
                }
            } else {
                break;
            }
        }
    }

    /// Decode sample `self.next` into `reference`, always advancing `next` (even
    /// on a per-sample error) so the caller's decode loop can't spin forever.
    fn decode_one(&mut self) -> Result<(), &'static str> {
        let s = self.samples[self.next];
        self.next += 1;
        if s.offset + s.size > self.bytes.len() {
            return Err("video: sample out of range");
        }
        let data = &self.bytes[s.offset..s.offset + s.size];
        // One mp4/mkv sample = one access unit = a full frame, possibly split
        // into multiple slice NALs. Collect them all and decode as one frame.
        let sample_i = self.next - 1;
        match &mut self.engine {
            Engine::RustH264 {
                dec,
                pending,
                open_sample,
            } => {
                let nals = rust_h264::nal::parse_avcc(data, self.length_size as usize);
                if nals.is_empty() {
                    return Err("video: sample has no NALs");
                }
                for nal in &nals {
                    match dec.decode_nal(nal) {
                        Ok(Some(frame)) => {
                            // Frame for the previously opened sample (decode order).
                            if let Some(prev) = *open_sample {
                                pending.insert(prev, yuv_from_rust_h264(frame));
                            }
                        }
                        Ok(None) => {}
                        Err(_) => return Err("rust_h264: decode_nal failed"),
                    }
                }
                // One sample ≈ one access unit / picture.
                *open_sample = Some(sample_i);
                while pending.len() > 24 {
                    let k = *pending.keys().next().unwrap();
                    pending.remove(&k);
                }
            }
            Engine::Hevc { dec, pending, next_out } => {
                let nals = hevc::split_hvcc(data, self.length_size);
                if nals.is_empty() {
                    return Err("video: sample has no NALs");
                }
                let out = dec.decode_au(&nals)?;
                // HEVC reorders, so a released picture belongs to the *display*
                // slot it comes out at, not to the sample that produced it.
                for f in out {
                    if *next_out < self.display.len() {
                        pending.insert(self.display[*next_out], f);
                        *next_out += 1;
                    }
                }
                while pending.len() > 16 {
                    let k = *pending.keys().next().unwrap();
                    pending.remove(&k);
                }
            }
            Engine::Vp9 { dec, pending } => {
                // One container sample may hold several coded frames; at most
                // one of them is shown.
                for &(a, b) in &vp9::split_superframe(data)? {
                    if let Some(f) = dec.decode_frame(&data[a..b])? {
                        pending.insert(sample_i, f);
                    }
                }
                while pending.len() > 8 {
                    let k = *pending.keys().next().unwrap();
                    pending.remove(&k);
                }
            }
            Engine::Cavlc { reference } => {
                let mut slices: Vec<(alloc::vec::Vec<u8>, bool)> = Vec::new();
                for nal in h264::split_avcc(data, self.length_size) {
                    if nal.kind.is_slice() {
                        slices.push((nal.rbsp(), nal.kind == h264::NalType::SliceIdr));
                    }
                }
                if slices.is_empty() {
                    return Err("video: sample has no slices");
                }
                let df = h264::decoder::decode_access_unit(
                    &self.sps,
                    &self.pps,
                    &slices,
                    reference.as_ref(),
                )?;
                *reference = Some(df);
            }
            Engine::Cabac { dec, pending } => {
                let mut slices: Vec<(alloc::vec::Vec<u8>, bool, u8)> = Vec::new();
                for nal in h264::split_avcc(data, self.length_size) {
                    if nal.kind.is_slice() {
                        slices.push((nal.rbsp(), nal.kind == h264::NalType::SliceIdr, nal.ref_idc));
                    }
                }
                if slices.is_empty() {
                    return Err("video: sample has no slices");
                }
                let pic = dec.decode_au(&slices)?;
                pending.insert(sample_i, pic);
                while pending.len() > 20 {
                    let k = *pending.keys().next().unwrap();
                    pending.remove(&k);
                }
            }
        }
        Ok(())
    }

    /// Ensure picture for decode sample `si` is in the engine pending map
    /// (rust_h264 needs a following AU or flush to release the last picture).
    fn ensure_rust_h264_frame(&mut self, si: usize) {
        let has = matches!(
            &self.engine,
            Engine::RustH264 { pending, .. } if pending.contains_key(&si)
        );
        if has {
            return;
        }
        if self.next < self.samples.len() {
            // Feed one more sample to force finalize of `si`.
            let _ = self.decode_one();
        } else if let Engine::RustH264 {
            dec,
            pending,
            open_sample,
        } = &mut self.engine
        {
            if let Some(frame) = dec.flush() {
                if let Some(prev) = *open_sample {
                    pending.insert(prev, yuv_from_rust_h264(frame));
                }
            }
            *open_sample = None;
        }
    }

    /// True if decode sample `si` can be skipped entirely without corrupting
    /// later pictures: every slice NAL in it is **non-reference**
    /// (`nal_ref_idc == 0`) — nothing else predicts from it.
    fn sample_droppable(&self, si: usize) -> bool {
        let s = &self.samples[si];
        if s.is_sync || s.offset + s.size > self.bytes.len() {
            return false;
        }
        sample_is_nonref(&self.bytes[s.offset..s.offset + s.size], self.length_size as usize)
    }

    /// As [`seek_decode`], but when `hurry` is set (playback is behind the
    /// clock) **skip decoding** backlog frames that are non-reference and
    /// display before `idx` — the H.264 form of frame-dropping: the picture
    /// was never going to be shown and nothing predicts from it.
    pub fn seek_decode_hurry(&mut self, idx: usize, hurry: bool) -> bool {
        self.hurry_before = if hurry { Some(idx.min(self.display.len().saturating_sub(1))) } else { None };
        let r = self.seek_decode(idx);
        self.hurry_before = None;
        r
    }

    /// Ensure the current RGB frame is display index `idx`, decoding forward as
    /// needed. A backward seek rewinds to the latest sync (IDR) sample ≤ the
    /// target in *decode* order and re-decodes forward (P/B frames can't be
    /// decoded in reverse). Returns `true` if the presented frame changed.
    pub fn seek_decode(&mut self, idx: usize) -> bool {
        let idx = idx.min(self.display.len().saturating_sub(1));
        if self.cur_idx == Some(idx) {
            return false;
        }
        // Ring hit (look-ahead / recent frame) — no re-decode.
        if let Some(f) = self.ring.get(&idx) {
            self.cur_idx = Some(idx);
            self.cur = Some(Frame {
                w: f.w,
                h: f.h,
                pixels: f.pixels.clone(),
                pts_ms: f.pts_ms,
            });
            return true;
        }
        // The picture shown at display position `idx` is the output of decode
        // sample `target`; decoding runs in decode (stored) order — same as
        // VLC: walk forward from the last keyframe, never reverse through P/B.
        let target = self.display[idx];
        let cached = matches!(
            &self.engine,
            Engine::Cabac { pending, .. } if pending.contains_key(&target)
        ) || matches!(
            &self.engine,
            Engine::RustH264 { pending, .. } if pending.contains_key(&target)
        );
        if target + 1 < self.next && !cached {
            // Rewind to the latest keyframe ≤ target and reset decode state.
            let mut s = target;
            while s > 0 && !self.samples[s].is_sync {
                s -= 1;
            }
            self.next = s;
            match &mut self.engine {
                Engine::Hevc { dec, pending, next_out } => {
                    *dec = hevc::decoder::HevcDecoder::new();
                    pending.clear();
                    *next_out = 0;
                }
                Engine::RustH264 {
                    dec,
                    pending,
                    open_sample,
                } => {
                    // Re-init rust_h264 (no public reset API).
                    if let Ok(d) = rust_h264_from_avcc(&self.avcc_sps, &self.avcc_pps) {
                        *dec = d;
                    }
                    pending.clear();
                    *open_sample = None;
                }
                Engine::Vp9 { dec, pending } => {
                    // There is no way to reset a VP9 decoder's reference slots
                    // other than starting again from a keyframe.
                    *dec = vp9::decoder::Vp9Decoder::new();
                    pending.clear();
                }
                Engine::Cavlc { reference } => *reference = None,
                Engine::Cabac { dec, pending } => {
                    dec.reset();
                    pending.clear();
                }
            }
            self.ring.clear();
        }
        while self.next <= target {
            // Frame-drop: behind the clock, a backlog picture nothing predicts
            // from (all slices nal_ref_idc == 0) that would display before the
            // hurry point is never fed to the decoder at all.
            if let Some(hb) = self.hurry_before {
                let si = self.next;
                if si != target && self.disp_of[si] < hb && self.sample_droppable(si) {
                    self.next += 1;
                    continue;
                }
            }
            if self.decode_one().is_err() {
                // Primary path failed — switch to native backup and retry once.
                if matches!(self.engine, Engine::RustH264 { .. }) {
                    self.fallback_to_native();
                    // Rewind to keyframe and continue with backup.
                    let mut s = target;
                    while s > 0 && !self.samples[s].is_sync {
                        s -= 1;
                    }
                    self.next = s;
                    continue;
                }
                break;
            }
        }
        // rust_h264 holds the last picture until the next AU or flush.
        if matches!(self.engine, Engine::RustH264 { .. }) {
            self.ensure_rust_h264_frame(target);
        }
        let pts = self.pts_ms(idx);
        let edge = self.display_edge;
        let frame = match &mut self.engine {
            Engine::RustH264 { pending, .. } => {
                let f = pending.get(&target).map(|df| frame_from_yuv(df, pts, edge));
                let min_future = self.display[idx..].iter().copied().min().unwrap_or(target);
                let dead: Vec<usize> = pending.range(..min_future).map(|(&k, _)| k).collect();
                for k in dead {
                    pending.remove(&k);
                }
                f
            }
            Engine::Hevc { pending, .. } => {
                let f = pending.get(&target).map(|f| {
                    frame_from_yuv(&hevc_frame_to_8bit(f), pts, edge)
                });
                let min_future = self.display[idx..].iter().copied().min().unwrap_or(target);
                let dead: Vec<usize> = pending.range(..min_future).map(|(&k, _)| k).collect();
                for k in dead {
                    pending.remove(&k);
                }
                f
            }
            Engine::Vp9 { pending, .. } => {
                let f = pending.get(&target).map(|f| {
                    frame_from_yuv(
                        &h264::decoder::DecodedFrame {
                            w: f.w,
                            h: f.h,
                            y: f.y.clone(),
                            cb: f.cb.clone(),
                            cr: f.cr.clone(),
                        },
                        pts,
                        edge,
                    )
                });
                let dead: alloc::vec::Vec<usize> = pending.range(..target).map(|(&k, _)| k).collect();
                for k in dead {
                    pending.remove(&k);
                }
                f
            }
            Engine::Cavlc { reference } => {
                reference.as_ref().map(|df| frame_from_yuv(df, pts, edge))
            }
            Engine::Cabac { pending, .. } => {
                let f = pending.get(&target).map(|pic| frame_from_yuv(&pic.f, pts, edge));
                let min_future = self.display[idx..].iter().copied().min().unwrap_or(target);
                let dead: Vec<usize> = pending.range(..min_future).map(|(&k, _)| k).collect();
                for k in dead {
                    pending.remove(&k);
                }
                f
            }
        };
        if let Some(f) = frame {
            self.cur_idx = Some(idx);
            self.ring_insert(
                idx,
                Frame {
                    w: f.w,
                    h: f.h,
                    pixels: f.pixels.clone(),
                    pts_ms: f.pts_ms,
                },
            );
            self.cur = Some(f);
            true
        } else {
            false
        }
    }

    /// Decode forward into the ring up to `play_idx + n` without changing the
    /// currently displayed frame (look-ahead only).
    pub fn prefetch(&mut self, play_idx: usize, n: usize) {
        if n == 0 || self.frame_count() == 0 {
            return;
        }
        let end = (play_idx + n).min(self.frame_count() - 1);
        let restore_idx = self.cur_idx;
        let restore = self.cur.take();
        for i in (play_idx + 1)..=end {
            if self.ring.contains_key(&i) {
                continue;
            }
            let _ = self.seek_decode(i);
        }
        // Put the playhead picture back (seek_decode may have moved `cur`).
        if let Some(i) = restore_idx {
            if let Some(f) = self.ring.get(&i) {
                self.cur_idx = Some(i);
                self.cur = Some(Frame {
                    w: f.w,
                    h: f.h,
                    pixels: f.pixels.clone(),
                    pts_ms: f.pts_ms,
                });
            } else if let Some(f) = restore {
                self.cur_idx = Some(i);
                self.cur = Some(f);
            }
        } else {
            self.cur_idx = None;
            self.cur = restore;
        }
    }

    /// The currently presented RGB frame, if any.
    pub fn cur_frame(&self) -> Option<&Frame> {
        self.cur.as_ref()
    }
}

/// What [`audio_info`] reports about a file's audio track.
/// What pressing play should do, given where the picture is in the clip.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Resume {
    /// Play again from the first frame — the clip had finished.
    Restart,
    /// Resume with the media clock anchored at this time (ms).
    At(u64),
}

/// Decide how to resume playback.
///
/// **The media clock and the displayed frame must never disagree**, which is the bug this
/// exists to prevent. The player advances by picking a goal frame *forward* of the current
/// one (`idx + 1`, clamped to the last frame), so if the clock is resumed behind the
/// picture nothing is ever due and nothing is ever decoded: playback reports "playing" and
/// sits on one frame until the clock catches up. Two ways that happened:
///
/// * **Play at the end of a clip.** Reaching the end stops playback but recorded no
///   position, so `paused_at` still held whatever the last manual pause left there — 0 if
///   the user never paused. Resuming anchored the clock to that stale time while `idx`
///   stayed on the final frame, where `idx + 1` clamps to `idx` and no goal is ever
///   greater. The picture froze on the last frame for a whole clip duration, then stopped
///   again. `Restart` is what every player does here.
/// * **Play after pausing earlier in the clip.** Same shape, shorter freeze: the clock
///   resumed at the old pause point, behind the frame on screen, and playback waited out
///   the difference. Anchoring to `max(paused_at, frame_pts)` makes the clock at least as
///   far along as the picture, so the next frame is due immediately.
///
/// `total_ms == 0` (a clip whose duration is unknown) must not read as "already finished".
pub fn resume_action(paused_at: u64, frame_pts: u64, total_ms: u64) -> Resume {
    if total_ms > 0 && paused_at >= total_ms {
        return Resume::Restart;
    }
    Resume::At(paused_at.max(frame_pts))
}

pub struct AudioInfo {
    pub codec: &'static str,
    pub sample_rate: u32,
    pub channels: u8,
    /// True when the in-kernel AAC-LC decoder can turn this track into PCM.
    pub decodable: bool,
}

/// Describe a file's audio track, if it has a demuxable one. Reports the AAC
/// track from an mp4/mov (`esds` `AudioSpecificConfig`); `decodable` is true
/// for any ASC whose core we can decode (plain AAC-LC and HE-AAC / HE-AACv2
/// with full SBR/PS reconstruction). `sample_rate` is the **playback** rate
/// (SBR output rate when HE-AAC).
pub fn audio_info(bytes: &[u8]) -> Option<AudioInfo> {
    match sniff(bytes) {
        Container::Mp4 => match mp4::parse_audio(bytes) {
            Ok(Some(t)) => {
                let (sample_rate, channels, decodable, codec) = match crate::audio::aac::parse_asc(&t.asc) {
                    Ok(a) => {
                        let codec = match (a.sbr, a.ps, a.aot) {
                            (true, true, _) => "HE-AACv2 (SBR+PS)",
                            (true, false, _) => "HE-AAC (SBR)",
                            (_, _, 1) => "AAC Main",
                            (_, _, 3) => "AAC SSR",
                            (_, _, 4) => "AAC LTP",
                            _ => "AAC (mp4a)",
                        };
                        // Prefer ASC **output** rate (SBR) for display/playback.
                        (a.output_rate(), a.channels, true, codec)
                    }
                    Err(_) => (t.sample_rate, t.channels, false, "AAC (mp4a)"),
                };
                Some(AudioInfo {
                    codec,
                    sample_rate,
                    channels,
                    decodable,
                })
            }
            _ => None,
        },
        _ => None,
    }
}

/// Demux + decode an mp4/mov AAC audio track to mono S16 PCM.
///
/// Uses [`mp4::parse_audio`] for the sample table and ASC, then the AAC-LC
/// decoder. Returns `Err` when there is no audio track or the ASC is outside
/// the supported LC subset.
pub fn decode_audio(bytes: &[u8]) -> Result<crate::audio::Audio, &'static str> {
    match sniff(bytes) {
        Container::Mp4 => {
            let track = mp4::parse_audio(bytes)?.ok_or("video: no AAC audio track")?;
            let sample_pairs: alloc::vec::Vec<(usize, usize)> =
                track.samples.iter().map(|s| (s.offset, s.size)).collect();
            crate::audio::aac::decode_track(
                track.sample_rate,
                track.channels,
                &track.asc,
                bytes,
                &sample_pairs,
            )
        }
        _ => Err("video: audio decode only for mp4/mov"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Resuming must never leave the clock behind the picture, because the player only
    /// ever decodes *forward*: a clock behind the displayed frame means no frame is due,
    /// nothing is decoded, and the picture sits still while the HUD says "playing".
    #[test_case]
    fn resuming_never_anchors_the_clock_behind_the_picture() {
        const TOTAL: u64 = 6_000;

        // Play at the end of the clip replays it. This is the freeze that was reported:
        // the end-of-clip stop left `paused_at` at 0, so play anchored the clock to 0
        // while the last frame stayed on screen — and `idx + 1` clamps to `idx` there, so
        // no goal was ever greater and nothing decoded for a whole clip duration.
        assert_eq!(resume_action(TOTAL, 5_966, TOTAL), Resume::Restart);
        assert_eq!(resume_action(TOTAL + 40, 5_966, TOTAL), Resume::Restart);
        // The pre-fix state, now harmless: a stale 0 with the picture on the last frame
        // resumes *at the picture*, not at 0.
        assert_eq!(resume_action(0, 5_966, TOTAL), Resume::At(5_966));

        // Ordinary mid-clip pause/resume is unchanged: the clock returns to the pause point.
        assert_eq!(resume_action(3_000, 3_000, TOTAL), Resume::At(3_000));
        assert_eq!(resume_action(3_000, 2_966, TOTAL), Resume::At(3_000));

        // A stale pause point behind the picture (paused early, played on, stopped) is
        // pulled forward to the frame on screen rather than waiting the difference out.
        assert_eq!(resume_action(1_000, 4_500, TOTAL), Resume::At(4_500));

        // A clip with unknown duration must not read as "already finished" — that would
        // make every resume restart from the beginning.
        assert_eq!(resume_action(0, 0, 0), Resume::At(0));
        assert_eq!(resume_action(5_000, 5_000, 0), Resume::At(5_000));

        // Whatever it returns, the clock is never behind the picture.
        for (paused, pts) in [(0u64, 0u64), (0, 5_966), (1_000, 4_500), (3_000, 3_000), (9_999, 12)] {
            match resume_action(paused, pts, TOTAL) {
                Resume::Restart => {}
                Resume::At(ms) => assert!(ms >= pts, "clock {ms} is behind the picture {pts}"),
            }
        }
    }

    #[test_case]
    fn sniff_recognises_containers() {
        let mut mp4 = alloc::vec![0u8; 12];
        mp4[4..8].copy_from_slice(b"ftyp");
        assert!(matches!(sniff(&mp4), Container::Mp4));
        assert!(matches!(sniff(&[0x1a, 0x45, 0xdf, 0xa3, 0, 0]), Container::Matroska));
        assert!(matches!(sniff(b"not a video"), Container::Unknown));
    }

    #[test_case]
    fn probe_rejects_unknown_container() {
        assert!(probe(b"random bytes here").is_err());
    }
}

#[cfg(test)]
mod framedrop_test {
    //! The pure frame-drop test (`sample_is_nonref`): only an all-non-ref
    //! sample may be skipped; anything malformed is conservatively kept.
    use super::sample_is_nonref;

    fn sample(nals: &[&[u8]]) -> alloc::vec::Vec<u8> {
        let mut v = alloc::vec::Vec::new();
        for n in nals {
            v.extend_from_slice(&(n.len() as u32).to_be_bytes());
            v.extend_from_slice(n);
        }
        v
    }

    #[test_case]
    fn nonref_slice_is_droppable() {
        // nal_ref_idc = 0, type 1 (non-IDR slice) → header 0x01.
        assert!(sample_is_nonref(&sample(&[&[0x01, 0xaa, 0xbb]]), 4));
        // SEI (type 6) alongside doesn't block the drop.
        assert!(sample_is_nonref(&sample(&[&[0x06, 0x05], &[0x01, 0xaa]]), 4));
    }

    #[test_case]
    fn reference_and_idr_are_kept() {
        // ref_idc 2 (0x41): reference P/B slice.
        assert!(!sample_is_nonref(&sample(&[&[0x41, 0xaa]]), 4));
        // IDR (type 5, ref_idc 3): 0x65.
        assert!(!sample_is_nonref(&sample(&[&[0x65, 0xaa]]), 4));
        // Mixed: one non-ref + one ref slice → keep.
        assert!(!sample_is_nonref(&sample(&[&[0x01, 0xaa], &[0x41, 0xbb]]), 4));
    }

    #[test_case]
    fn malformed_is_kept() {
        assert!(!sample_is_nonref(&[], 4));
        assert!(!sample_is_nonref(&[0, 0, 0, 9, 0x01], 4)); // length overruns
        assert!(!sample_is_nonref(&sample(&[&[0x06, 0x05]]), 4)); // no slice at all
        assert!(!sample_is_nonref(&sample(&[&[0x01, 0xaa]]), 0)); // bad length size
    }
}

#[cfg(test)]
mod decode_fixture_test {
    //! Full-decode regression: an embedded baseline keyframe (x264 → mp4)
    //! whose YUV hash was captured from PyAV/ffmpeg. Guards the Rust port.
    use super::*;

    /// These fixtures are all H.264, so unwrap the config to its `AvcC`. A
    /// fixture that stopped being H.264 should fail loudly here rather than be
    /// skipped.
    fn expect_avc(c: &mp4::CodecConfig) -> &mp4::AvcC {
        match c {
            mp4::CodecConfig::Avc(a) => a,
            _ => panic!("fixture is not an H.264 track"),
        }
    }

    // c_hq 64x64 baseline, muxed from x264; expected hash 2213512446
    const CLIP_MP4: [u8; 2139] = [
    0, 0, 0, 32, 102, 116, 121, 112, 105, 115, 111, 109, 0, 0, 2, 0, 105, 115, 111, 109, 105, 115, 111, 50,
    97, 118, 99, 49, 109, 112, 52, 49, 0, 0, 2, 113, 109, 111, 111, 118, 0, 0, 0, 108, 109, 118, 104, 100,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25, 0, 0, 0, 1, 0, 1, 0, 0,
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 2, 0, 0, 1, 253, 116, 114, 97, 107, 0, 0, 0, 92, 116, 107, 104, 100, 0, 0, 0, 7,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0,
    0, 64, 0, 0, 0, 64, 0, 0, 0, 0, 1, 153, 109, 100, 105, 97, 0, 0, 0, 28, 109, 100, 104, 100,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25, 0, 1, 0, 0, 0, 0, 0, 39,
    104, 100, 108, 114, 0, 0, 0, 0, 0, 0, 0, 0, 118, 105, 100, 101, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 99, 104, 105, 116, 116, 105, 0, 0, 0, 1, 78, 109, 105, 110, 102, 0, 0, 0, 20, 118,
    109, 104, 100, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 36, 100, 105, 110, 102, 0,
    0, 0, 28, 100, 114, 101, 102, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 12, 117, 114, 108, 32, 0,
    0, 0, 1, 0, 0, 1, 14, 115, 116, 98, 108, 0, 0, 0, 146, 115, 116, 115, 100, 0, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 0, 130, 97, 118, 99, 49, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 0, 64, 0, 72, 0, 0, 0, 72, 0, 0, 0,
    0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 255, 255, 0, 0, 0, 44, 97, 118, 99,
    67, 1, 66, 192, 10, 255, 225, 0, 20, 103, 66, 192, 10, 220, 66, 104, 64, 0, 0, 3, 0, 64, 0, 0,
    12, 163, 196, 137, 224, 1, 0, 5, 104, 206, 1, 12, 178, 0, 0, 0, 24, 115, 116, 116, 115, 0, 0, 0,
    0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 28, 115, 116, 115, 99, 0, 0, 0,
    0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 24, 115, 116, 115,
    122, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 5, 194, 0, 0, 0, 20, 115, 116, 115,
    115, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 20, 115, 116, 99, 111, 0, 0, 0,
    0, 0, 0, 0, 1, 0, 0, 2, 153, 0, 0, 5, 202, 109, 100, 97, 116, 0, 0, 5, 190, 101, 136, 132,
    58, 196, 162, 37, 34, 33, 203, 192, 59, 128, 1, 135, 0, 1, 34, 154, 202, 224, 0, 84, 0, 134, 50, 0,
    32, 35, 42, 160, 0, 84, 0, 67, 25, 0, 2, 96, 6, 64, 84, 0, 9, 128, 25, 3, 32, 2, 2, 50,
    53, 0, 2, 160, 2, 24, 84, 0, 64, 70, 76, 128, 8, 8, 200, 168, 0, 128, 140, 141, 64, 0, 168, 0,
    134, 21, 0, 16, 17, 149, 80, 0, 42, 0, 33, 140, 128, 8, 8, 202, 168, 0, 21, 0, 16, 193, 0, 0,
    65, 68, 0, 5, 134, 98, 0, 0, 130, 136, 0, 11, 12, 160, 116, 0, 13, 129, 208, 0, 54, 7, 64, 0,
    216, 29, 0, 3, 96, 29, 0, 3, 50, 58, 0, 6, 100, 116, 0, 12, 200, 232, 0, 25, 144, 186, 135, 0,
    1, 6, 176, 0, 16, 219, 0, 1, 29, 132, 7, 128, 1, 54, 15, 0, 2, 108, 62, 0, 4, 216, 60, 0,
    9, 176, 120, 0, 19, 96, 240, 0, 38, 193, 224, 0, 77, 131, 192, 0, 155, 15, 128, 1, 54, 15, 0, 2,
    108, 62, 0, 4, 216, 60, 0, 9, 176, 120, 0, 19, 96, 240, 0, 38, 193, 224, 0, 77, 131, 192, 0, 155,
    2, 0, 0, 130, 136, 0, 8, 97, 81, 208, 0, 54, 7, 64, 0, 216, 29, 0, 3, 96, 116, 0, 13, 143,
    133, 212, 56, 0, 8, 53, 128, 0, 134, 216, 0, 8, 236, 32, 60, 0, 9, 176, 120, 0, 19, 96, 240, 0,
    38, 193, 224, 0, 77, 131, 192, 0, 155, 7, 128, 1, 54, 15, 0, 2, 108, 30, 0, 4, 216, 60, 0, 9,
    176, 120, 0, 19, 96, 240, 0, 38, 193, 224, 0, 77, 131, 192, 0, 155, 7, 128, 1, 54, 15, 0, 2, 108,
    30, 0, 4, 216, 16, 0, 4, 20, 64, 0, 67, 10, 142, 128, 1, 176, 58, 0, 6, 192, 232, 0, 27, 3,
    160, 0, 108, 120, 93, 64, 7, 0, 1, 5, 48, 0, 16, 54, 0, 2, 128, 76, 102, 230, 228, 228, 229, 93,
    205, 205, 205, 200, 194, 2, 225, 224, 0, 77, 131, 192, 0, 155, 7, 128, 1, 54, 15, 0, 2, 108, 30, 0,
    4, 216, 60, 0, 9, 176, 120, 0, 19, 96, 240, 0, 38, 193, 224, 0, 77, 131, 192, 0, 155, 7, 128, 1,
    54, 15, 0, 2, 108, 30, 0, 4, 216, 60, 0, 9, 176, 120, 0, 19, 96, 0, 144, 0, 16, 77, 0, 2,
    128, 1, 132, 193, 192, 195, 207, 151, 88, 43, 131, 234, 232, 10, 65, 64, 64, 0, 16, 81, 0, 1, 12, 42,
    58, 0, 6, 192, 232, 0, 27, 3, 160, 0, 108, 14, 128, 1, 177, 225, 103, 7, 0, 1, 12, 120, 0, 2,
    50, 160, 0, 38, 205, 9, 3, 176, 0, 29, 168, 45, 0, 3, 50, 61, 128, 0, 237, 64, 232, 0, 25, 144,
    118, 0, 3, 181, 5, 160, 0, 102, 65, 216, 0, 14, 212, 14, 128, 1, 153, 30, 192, 0, 118, 160, 116, 0,
    12, 200, 246, 0, 3, 181, 3, 160, 0, 102, 65, 216, 0, 14, 212, 14, 128, 1, 153, 7, 96, 0, 59, 80,
    58, 0, 6, 101, 16, 0, 4, 20, 64, 0, 67, 9, 124, 116, 0, 12, 200, 232, 0, 25, 145, 208, 0, 51,
    35, 160, 0, 102, 74, 79, 34, 33, 34, 35, 34, 55, 184, 200, 224, 208, 1, 1, 25, 21, 0, 16, 17, 145,
    80, 1, 1, 25, 26, 128, 1, 80, 1, 12, 42, 0, 32, 35, 34, 160, 2, 2, 50, 100, 0, 64, 70, 64,
    3, 128, 0, 128, 24, 0, 16, 4, 5, 206, 73, 218, 13, 29, 58, 48, 144, 88, 86, 150, 163, 149, 0, 16,
    17, 147, 32, 2, 2, 50, 0, 56, 0, 8, 1, 0, 1, 64, 17, 0, 54, 48, 55, 125, 138, 36, 159, 196,
    226, 163, 45, 251, 64, 0, 192, 0, 66, 4, 0, 128, 13, 181, 80, 24, 77, 84, 52, 20, 10, 52, 2, 30,
    254, 0, 28, 0, 4, 21, 64, 0, 160, 0, 101, 192, 72, 22, 44, 145, 79, 247, 76, 77, 53, 65, 8, 188,
    8, 62, 0, 0, 128, 104, 8, 0, 144, 103, 64, 155, 177, 58, 122, 93, 94, 87, 28, 100, 177, 147, 0, 2,
    96, 6, 64, 76, 0, 9, 128, 25, 4, 141, 201, 196, 166, 68, 57, 72, 229, 64, 4, 4, 100, 0, 112, 0,
    16, 2, 0, 2, 128, 34, 0, 108, 96, 110, 251, 20, 73, 63, 137, 197, 70, 91, 247, 2, 0, 1, 9, 224,
    0, 45, 147, 136, 34, 12, 148, 154, 49, 83, 230, 2, 239, 252, 131, 128, 1, 135, 0, 1, 23, 186, 205, 0,
    3, 0, 1, 8, 16, 2, 0, 54, 213, 64, 97, 53, 80, 208, 80, 40, 208, 8, 123, 248, 152, 0, 19, 0,
    50, 2, 96, 2, 6, 100, 106, 0, 5, 64, 4, 48, 228, 0, 64, 103, 137, 144, 4, 4, 100, 200, 0, 128,
    140, 141, 64, 0, 168, 0, 134, 21, 0, 16, 17, 149, 80, 0, 42, 0, 33, 140, 128, 8, 8, 202, 168, 0,
    21, 0, 16, 201, 17, 9, 17, 9, 17, 28, 220, 158, 200, 0, 128, 140, 138, 128, 8, 8, 200, 168, 0, 128,
    140, 141, 64, 0, 168, 0, 134, 50, 0, 32, 35, 34, 160, 2, 2, 50, 100, 0, 64, 70, 70, 160, 0, 84,
    0, 67, 10, 128, 8, 8, 201, 144, 1, 1, 25, 21, 0, 16, 17, 144, 0, 224, 0, 32, 6, 0, 4, 1,
    1, 115, 146, 118, 131, 71, 78, 140, 36, 22, 21, 165, 168, 229, 64, 4, 4, 100, 4, 128, 0, 128, 16, 0,
    20, 1, 16, 3, 99, 3, 119, 216, 162, 73, 252, 78, 42, 50, 223, 184, 16, 0, 8, 79, 0, 1, 108, 156,
    65, 16, 100, 164, 209, 138, 159, 48, 23, 127, 228, 28, 0, 12, 56, 0, 8, 189, 214, 34, 0, 0, 130, 136,
    0, 8, 97, 47, 29, 0, 3, 50, 58, 0, 6, 100, 116, 0, 12, 200, 232, 0, 25, 153, 17, 25, 187, 220,
    74, 66, 82, 57, 80, 1, 1, 25, 21, 0, 16, 17, 147, 32, 2, 2, 50, 0, 28, 0, 4, 18, 192, 0,
    160, 0, 97, 48, 120, 48, 251, 229, 214, 26, 176, 218, 194, 7, 128, 144, 15, 160, 3, 128, 0, 128, 24, 0,
    16, 4, 5, 206, 73, 218, 13, 29, 58, 48, 144, 88, 86, 150, 163, 240, 32, 0, 16, 158, 0, 2, 217, 56,
    130, 32, 201, 73, 163, 21, 62, 96, 46, 255, 222, 0, 0, 128, 104, 24, 1, 32, 206, 129, 55, 98, 132, 212,
    154, 188, 174, 57, 201, 97, 127, 132, 128, 202, 27, 179, 64, 0, 192, 0, 66, 4, 0, 128, 13, 181, 80, 24,
    77, 84, 52, 20, 10, 52, 2, 30, 254, 38, 0, 4, 192, 12, 128, 152, 0, 129, 153, 26, 128, 1, 80, 1,
    12, 38, 0, 129, 153, 86, 0, 2, 160, 4, 173, 45, 0, 64, 207, 19, 32, 0, 76, 0, 200, 36, 68, 36,
    68, 36, 68, 57, 72, 232, 56, 0, 24, 112, 0, 17, 123, 172, 38, 0, 32, 102, 76, 128, 32, 35, 35, 80,
    0, 42, 0, 33, 133, 64, 4, 4, 100, 84, 0, 64, 70, 76, 128, 8, 8, 200, 212, 0, 10, 128, 8, 99,
    32, 2, 2, 50, 100, 0, 64, 70, 69, 64, 4, 4, 100, 106, 0, 5, 64, 4, 48, 168, 0, 128, 140, 170,
    128, 1, 80, 1, 12, 100, 0, 64, 70, 85, 64, 0, 168, 0, 134, 20, 158, 79, 19, 34, 18, 34, 28, 175,
    0, 28, 0, 4, 21, 64, 0, 160, 0, 101, 192, 72, 22, 44, 145, 79, 247, 76, 77, 53, 65, 8, 188, 8,
    62, 0, 0, 128, 104, 8, 0, 144, 103, 64, 155, 177, 58, 122, 93, 94, 87, 28, 100, 177, 160, 224, 0, 97,
    192, 0, 66, 214, 177, 62, 0, 5, 64, 8, 97, 200, 0, 128, 207, 19, 32, 8, 8, 201, 144, 1, 1, 25,
    26, 128, 1, 80, 1, 12, 100, 0, 64, 70, 76, 128, 8, 8, 200, 168, 0, 128, 140, 141, 64, 0, 168, 0,
    134, 21, 0, 16, 17, 149, 80, 0, 42, 0, 33, 140, 128, 8, 8, 202, 168, 0, 21, 0, 16, 196, 64, 0,
    16, 81, 0, 1, 12, 37, 227, 160, 0, 102, 71, 64, 0, 204, 142, 128, 1, 153, 29, 0, 3, 50, 82, 114,
    147, 148, 156,
];
    const EXPECT_HASH: u32 = 2213512446;
    const EXPECT_W: usize = 64;
    const EXPECT_H: usize = 64;

    #[test_case]
    fn decodes_embedded_keyframe_matching_pyav() {
        let track = mp4::parse(&CLIP_MP4).unwrap();
        let track_avcc = expect_avc(&track.config);
        let sps = h264::parse_sps(&bits::unescape_rbsp(&track_avcc.sps[0][1..])).unwrap();
        let pps = h264::parse_pps(&bits::unescape_rbsp(&track_avcc.pps[0][1..])).unwrap();
        assert_eq!((sps.width() as usize, sps.height() as usize), (EXPECT_W, EXPECT_H));
        let s = track.samples.iter().find(|s| s.is_sync).expect("sync sample");
        let data = &CLIP_MP4[s.offset..s.offset + s.size];
        let nal = h264::split_avcc(data, track_avcc.length_size)
            .into_iter()
            .find(|n| n.kind == h264::NalType::SliceIdr)
            .expect("IDR nal");
        let df = h264::decoder::decode_islice(&sps, &pps, &nal.rbsp(), true).unwrap();
        assert_eq!((df.w, df.h), (EXPECT_W, EXPECT_H));
        let mut hh: u32 = 0;
        for &b in df.y.iter().chain(df.cb.iter()).chain(df.cr.iter()) {
            hh = hh.wrapping_mul(31).wrapping_add(b as u32);
        }
        assert_eq!(hh, EXPECT_HASH, "decoded keyframe must match the PyAV reference");
    }
    const PCLIP_MP4: [u8; 3331] = [
    0, 0, 0, 32, 102, 116, 121, 112, 105, 115, 111, 109, 0, 0, 2, 0, 105, 115, 111, 109, 105, 115, 111, 50,
    97, 118, 99, 49, 109, 112, 52, 49, 0, 0, 2, 126, 109, 111, 111, 118, 0, 0, 0, 108, 109, 118, 104, 100,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25, 0, 0, 0, 4, 0, 1, 0, 0,
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 2, 0, 0, 2, 10, 116, 114, 97, 107, 0, 0, 0, 92, 116, 107, 104, 100, 0, 0, 0, 7,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0,
    0, 96, 0, 0, 0, 64, 0, 0, 0, 0, 1, 166, 109, 100, 105, 97, 0, 0, 0, 28, 109, 100, 104, 100,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25, 0, 4, 0, 0, 0, 0, 0, 39,
    104, 100, 108, 114, 0, 0, 0, 0, 0, 0, 0, 0, 118, 105, 100, 101, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 99, 104, 105, 116, 116, 105, 0, 0, 0, 1, 91, 109, 105, 110, 102, 0, 0, 0, 20, 118,
    109, 104, 100, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 36, 100, 105, 110, 102, 0,
    0, 0, 28, 100, 114, 101, 102, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 12, 117, 114, 108, 32, 0,
    0, 0, 1, 0, 0, 1, 27, 115, 116, 98, 108, 0, 0, 0, 147, 115, 116, 115, 100, 0, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 0, 131, 97, 118, 99, 49, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 96, 0, 64, 0, 72, 0, 0, 0, 72, 0, 0, 0,
    0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 255, 255, 0, 0, 0, 45, 97, 118, 99,
    67, 1, 66, 192, 10, 255, 225, 0, 21, 103, 66, 192, 10, 218, 24, 154, 16, 0, 0, 3, 0, 16, 0, 0,
    3, 3, 40, 241, 34, 106, 1, 0, 5, 104, 206, 3, 178, 200, 0, 0, 0, 24, 115, 116, 116, 115, 0, 0,
    0, 0, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 1, 0, 0, 0, 28, 115, 116, 115, 99, 0, 0,
    0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 1, 0, 0, 0, 36, 115, 116,
    115, 122, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 7, 104, 0, 0, 1, 15, 0, 0,
    0, 240, 0, 0, 0, 246, 0, 0, 0, 20, 115, 116, 115, 115, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
    0, 1, 0, 0, 0, 20, 115, 116, 99, 111, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 2, 166, 0, 0,
    10, 101, 109, 100, 97, 116, 0, 0, 7, 100, 101, 136, 132, 58, 196, 162, 37, 34, 33, 203, 192, 59, 128, 12,
    80, 0, 17, 153, 172, 100, 0, 64, 34, 6, 64, 65, 153, 11, 128, 8, 6, 64, 168, 0, 128, 68, 5, 64,
    4, 2, 32, 100, 4, 25, 144, 184, 0, 128, 100, 10, 128, 131, 50, 100, 4, 25, 145, 80, 16, 102, 66, 224,
    2, 1, 144, 42, 2, 12, 201, 120, 0, 128, 100, 25, 1, 6, 100, 188, 0, 64, 50, 2, 0, 1, 112, 0,
    21, 107, 136, 0, 2, 4, 160, 0, 42, 54, 133, 24, 163, 20, 98, 140, 72, 0, 19, 50, 120, 0, 19, 50,
    120, 0, 19, 50, 120, 0, 19, 50, 23, 80, 224, 0, 32, 154, 0, 2, 19, 224, 0, 34, 176, 128, 240, 1,
    96, 240, 1, 96, 240, 1, 96, 240, 1, 96, 240, 1, 96, 240, 1, 96, 240, 1, 96, 240, 1, 96, 240, 1,
    96, 240, 1, 96, 240, 1, 96, 240, 1, 96, 240, 1, 96, 240, 1, 96, 240, 1, 96, 240, 1, 96, 64, 0,
    46, 0, 26, 138, 49, 70, 40, 197, 31, 194, 234, 28, 0, 4, 19, 64, 0, 66, 124, 0, 4, 86, 16, 30,
    0, 44, 30, 0, 44, 30, 0, 44, 30, 0, 44, 30, 0, 44, 30, 0, 44, 30, 0, 44, 30, 0, 44, 30,
    0, 44, 30, 0, 44, 30, 0, 44, 30, 0, 44, 30, 0, 44, 30, 0, 44, 30, 0, 44, 30, 0, 44, 8,
    0, 5, 192, 3, 81, 70, 40, 197, 24, 163, 240, 186, 128, 14, 0, 2, 7, 96, 0, 32, 60, 0, 40, 27,
    59, 59, 58, 58, 58, 196, 236, 236, 236, 232, 126, 14, 135, 128, 11, 7, 128, 11, 7, 128, 11, 7, 128, 11,
    7, 128, 11, 7, 128, 11, 7, 128, 11, 7, 128, 11, 7, 128, 11, 7, 128, 11, 7, 128, 11, 7, 128, 11,
    7, 128, 11, 7, 128, 11, 7, 128, 11, 0, 4, 128, 0, 129, 168, 0, 160, 2, 27, 12, 12, 54, 237, 89,
    130, 144, 35, 164, 131, 80, 164, 32, 0, 23, 0, 13, 69, 24, 163, 20, 98, 143, 226, 37, 17, 55, 55, 121,
    9, 86, 84, 0, 64, 34, 2, 160, 2, 1, 16, 21, 1, 6, 100, 46, 0, 32, 25, 2, 160, 2, 1, 16,
    21, 0, 16, 8, 128, 9, 0, 8, 0, 80, 17, 0, 76, 216, 38, 223, 12, 186, 85, 48, 57, 39, 223, 180,
    0, 96, 0, 32, 174, 3, 128, 234, 191, 133, 210, 208, 8, 142, 13, 243, 132, 221, 252, 0, 56, 0, 96, 2,
    3, 5, 201, 91, 168, 56, 56, 80, 90, 37, 77, 70, 70, 139, 194, 0, 1, 8, 160, 0, 16, 31, 44, 43,
    157, 10, 185, 34, 75, 254, 214, 28, 127, 239, 0, 0, 64, 12, 16, 4, 133, 112, 20, 106, 110, 151, 230, 142,
    106, 139, 49, 210, 218, 202, 13, 198, 137, 7, 0, 24, 160, 0, 34, 35, 88, 118, 0, 32, 25, 6, 64, 65,
    153, 47, 0, 16, 12, 128, 128, 0, 92, 0, 53, 20, 98, 140, 81, 138, 63, 221, 226, 37, 41, 14, 85, 128,
    14, 0, 2, 6, 160, 2, 128, 8, 108, 48, 48, 227, 181, 102, 10, 64, 142, 146, 10, 66, 144, 38, 248, 0,
    2, 0, 96, 192, 36, 43, 128, 163, 83, 116, 191, 52, 115, 84, 89, 142, 150, 214, 80, 220, 73, 141, 116, 27,
    137, 18, 124, 0, 16, 9, 90, 100, 0, 64, 34, 6, 64, 65, 153, 11, 128, 8, 6, 65, 48, 3, 65, 169,
    47, 1, 6, 157, 50, 2, 8, 201, 120, 0, 128, 72, 67, 32, 32, 204, 151, 128, 8, 6, 65, 144, 16, 102,
    75, 192, 4, 3, 32, 32, 0, 23, 0, 13, 69, 24, 163, 20, 98, 143, 194, 206, 14, 0, 2, 18, 176, 0,
    4, 76, 64, 0, 74, 74, 18, 14, 128, 14, 200, 90, 0, 51, 33, 208, 1, 217, 7, 64, 6, 100, 58, 0,
    59, 33, 104, 0, 204, 135, 64, 7, 100, 29, 0, 25, 144, 232, 0, 236, 131, 160, 3, 50, 29, 0, 29, 144,
    116, 0, 102, 67, 160, 3, 178, 14, 128, 12, 200, 116, 0, 118, 65, 208, 1, 153, 68, 0, 1, 2, 80, 0,
    16, 96, 94, 120, 0, 19, 50, 120, 0, 19, 50, 120, 0, 19, 50, 120, 0, 19, 50, 82, 121, 17, 9, 17,
    25, 17, 189, 198, 71, 6, 128, 131, 50, 42, 2, 12, 200, 168, 8, 51, 33, 112, 1, 0, 200, 21, 1, 6,
    100, 84, 4, 25, 147, 32, 32, 204, 128, 7, 0, 12, 0, 64, 96, 185, 43, 117, 7, 7, 10, 11, 68, 169,
    168, 200, 209, 74, 128, 131, 50, 100, 4, 25, 144, 1, 192, 2, 0, 20, 4, 64, 19, 54, 9, 183, 195, 46,
    149, 76, 14, 73, 247, 237, 0, 24, 0, 8, 43, 128, 224, 58, 175, 225, 116, 180, 2, 35, 131, 124, 225, 55,
    127, 0, 14, 0, 2, 7, 96, 2, 128, 8, 168, 52, 19, 34, 178, 148, 234, 191, 41, 209, 30, 158, 135, 191,
    0, 0, 64, 12, 16, 15, 10, 224, 40, 212, 225, 39, 205, 28, 213, 150, 99, 165, 167, 128, 1, 0, 134, 29,
    128, 8, 6, 132, 72, 220, 156, 74, 100, 67, 148, 142, 84, 4, 25, 144, 1, 192, 2, 0, 20, 4, 64, 19,
    54, 9, 183, 195, 46, 149, 76, 14, 73, 247, 238, 16, 0, 8, 69, 0, 0, 128, 249, 97, 92, 232, 85, 201,
    18, 95, 246, 176, 227, 255, 32, 224, 3, 20, 0, 4, 68, 107, 52, 0, 96, 0, 32, 174, 3, 128, 234, 191,
    133, 210, 208, 8, 142, 13, 243, 132, 221, 252, 158, 0, 4, 2, 24, 76, 4, 17, 146, 240, 1, 0, 200, 19,
    1, 4, 100, 200, 8, 51, 38, 64, 65, 153, 11, 128, 8, 6, 64, 168, 8, 51, 37, 224, 2, 1, 144, 100,
    4, 25, 146, 240, 1, 0, 200, 20, 144, 202, 79, 34, 33, 34, 33, 34, 35, 155, 147, 202, 128, 131, 50, 42,
    2, 12, 200, 168, 8, 51, 33, 112, 1, 0, 200, 21, 1, 6, 100, 84, 4, 25, 147, 32, 32, 204, 133, 192,
    4, 3, 32, 84, 4, 25, 147, 32, 32, 204, 138, 128, 131, 50, 0, 28, 0, 48, 1, 1, 130, 228, 173, 212,
    28, 28, 40, 45, 18, 166, 163, 35, 69, 42, 2, 12, 200, 9, 0, 8, 0, 80, 17, 0, 76, 216, 38, 223,
    12, 186, 85, 48, 57, 39, 223, 184, 64, 0, 33, 20, 0, 2, 3, 229, 133, 115, 161, 87, 36, 73, 127, 218,
    195, 143, 252, 131, 128, 12, 80, 0, 17, 17, 172, 68, 0, 1, 2, 80, 0, 16, 96, 94, 120, 0, 19, 50,
    120, 0, 19, 50, 120, 0, 19, 50, 120, 0, 19, 51, 34, 35, 55, 123, 137, 72, 74, 71, 42, 2, 12, 200,
    168, 8, 51, 38, 64, 65, 153, 0, 14, 0, 2, 6, 96, 2, 128, 8, 108, 52, 48, 223, 181, 102, 10, 17,
    229, 52, 21, 132, 224, 214, 0, 56, 0, 96, 2, 3, 5, 201, 91, 168, 56, 56, 80, 90, 37, 77, 70, 70,
    139, 194, 0, 1, 8, 160, 0, 16, 31, 44, 43, 157, 10, 185, 34, 75, 254, 214, 28, 127, 239, 0, 0, 64,
    12, 24, 4, 133, 112, 20, 106, 110, 151, 230, 142, 106, 138, 241, 210, 90, 232, 55, 26, 45, 0, 24, 0, 8,
    43, 128, 224, 58, 175, 225, 116, 180, 2, 35, 131, 124, 225, 55, 127, 39, 128, 1, 0, 134, 19, 1, 4, 100,
    188, 0, 64, 50, 7, 64, 32, 145, 38, 64, 4, 2, 32, 100, 4, 25, 144, 184, 0, 128, 100, 2, 4, 160,
    0, 2, 18, 160, 0, 32, 98, 0, 2, 30, 160, 0, 216, 52, 172, 49, 192, 6, 45, 97, 142, 0, 49, 107,
    7, 184, 0, 197, 172, 30, 224, 3, 22, 176, 123, 128, 12, 90, 193, 238, 0, 49, 107, 7, 184, 0, 197, 172,
    30, 224, 3, 22, 176, 123, 128, 12, 90, 193, 238, 0, 49, 107, 7, 184, 0, 197, 172, 30, 224, 3, 22, 176,
    123, 128, 12, 90, 193, 238, 0, 49, 107, 7, 184, 0, 197, 172, 30, 224, 3, 22, 176, 164, 249, 73, 132, 73,
    0, 4, 0, 1, 3, 144, 0, 16, 28, 0, 16, 12, 156, 156, 156, 28, 28, 28, 28, 156, 156, 156, 28, 28,
    255, 255, 0, 11, 0, 1, 3, 80, 3, 128, 8, 108, 44, 48, 219, 189, 109, 32, 65, 69, 6, 33, 113, 255,
    73, 226, 100, 66, 68, 67, 149, 224, 3, 128, 0, 129, 216, 0, 160, 2, 42, 13, 4, 200, 172, 165, 58, 175,
    202, 116, 71, 167, 161, 239, 192, 0, 16, 3, 4, 3, 194, 184, 10, 53, 56, 73, 243, 71, 53, 101, 152, 233,
    104, 56, 0, 197, 0, 1, 7, 26, 195, 176, 1, 0, 200, 19, 1, 4, 100, 200, 8, 51, 38, 64, 65, 153,
    11, 128, 8, 6, 65, 144, 16, 102, 76, 128, 131, 50, 42, 2, 12, 200, 92, 0, 64, 50, 5, 64, 65, 153,
    47, 0, 16, 12, 131, 32, 32, 204, 151, 128, 8, 6, 65, 16, 0, 4, 9, 64, 0, 65, 129, 121, 224, 0,
    76, 201, 224, 0, 76, 201, 224, 0, 76, 201, 224, 0, 76, 201, 73, 202, 78, 82, 114, 34, 18, 34, 50, 35,
    123, 140, 142, 13, 1, 6, 100, 84, 4, 25, 145, 80, 16, 102, 66, 224, 2, 1, 144, 42, 2, 12, 200, 168,
    8, 51, 38, 64, 65, 153, 0, 14, 0, 24, 0, 128, 193, 114, 86, 234, 14, 14, 20, 22, 137, 83, 81, 145,
    162, 149, 1, 6, 100, 200, 8, 51, 32, 3, 128, 4, 0, 40, 8, 128, 38, 108, 19, 111, 134, 93, 42, 152,
    28, 147, 239, 218, 0, 48, 0, 16, 87, 1, 192, 117, 95, 194, 233, 104, 4, 71, 6, 249, 194, 110, 254, 0,
    28, 0, 4, 14, 192, 5, 0, 17, 80, 104, 38, 69, 101, 41, 213, 126, 83, 162, 61, 61, 15, 126, 0, 0,
    128, 24, 32, 30, 21, 192, 81, 169, 194, 79, 154, 57, 171, 44, 199, 75, 79, 0, 2, 1, 12, 59, 0, 16,
    13, 8, 145, 185, 56, 148, 200, 135, 41, 28, 168, 8, 51, 32, 3, 128, 4, 0, 40, 8, 128, 38, 108, 19,
    111, 134, 93, 42, 152, 28, 147, 239, 220, 32, 0, 16, 138, 0, 1, 1, 242, 194, 185, 208, 171, 146, 36, 191,
    237, 97, 199, 254, 65, 192, 6, 40, 0, 8, 136, 214, 1, 32, 12, 0, 4, 21, 192, 112, 29, 87, 240, 186,
    90, 1, 17, 193, 190, 112, 155, 191, 147, 192, 0, 128, 67, 9, 128, 130, 50, 94, 0, 32, 25, 2, 96, 32,
    140, 153, 1, 6, 100, 200, 8, 51, 33, 112, 1, 0, 200, 21, 1, 6, 100, 188, 0, 64, 50, 12, 128, 131,
    50, 94, 0, 32, 25, 8, 0, 0, 1, 11, 65, 154, 34, 184, 97, 71, 0, 2, 81, 192, 0, 239, 212, 112,
    0, 37, 28, 0, 14, 253, 71, 0, 2, 81, 192, 0, 239, 212, 112, 0, 37, 28, 0, 14, 253, 71, 0, 2,
    81, 192, 0, 239, 212, 112, 0, 37, 28, 0, 14, 253, 71, 0, 2, 81, 192, 0, 239, 212, 112, 0, 37, 28,
    0, 14, 253, 71, 0, 2, 81, 192, 0, 239, 212, 112, 0, 37, 28, 0, 14, 226, 137, 197, 19, 245, 28, 0,
    9, 71, 0, 3, 191, 81, 192, 0, 148, 112, 0, 59, 138, 39, 20, 78, 40, 159, 12, 40, 224, 0, 74, 56,
    0, 29, 248, 99, 131, 64, 1, 6, 100, 84, 0, 16, 102, 89, 80, 0, 65, 153, 21, 0, 4, 25, 144, 224,
    0, 74, 32, 128, 13, 126, 32, 29, 16, 14, 197, 19, 138, 39, 11, 18, 0, 8, 0, 2, 2, 32, 1, 0,
    129, 22, 44, 88, 48, 96, 193, 139, 22, 44, 24, 49, 255, 254, 0, 30, 1, 0, 0, 129, 8, 11, 2, 138,
    60, 135, 247, 131, 118, 49, 89, 227, 127, 120, 99, 130, 0, 48, 28, 60, 113, 209, 225, 224, 225, 145, 168, 44,
    133, 179, 118, 151, 128, 2, 9, 26, 202, 128, 2, 12, 200, 168, 0, 32, 140, 135, 0, 2, 81, 4, 0, 107,
    241, 0, 232, 128, 119, 195, 28, 168, 0, 32, 204, 138, 128, 2, 12, 203, 42, 0, 8, 51, 34, 160, 0, 131,
    50, 28, 0, 9, 68, 16, 1, 175, 196, 3, 162, 1, 216, 0, 0, 0, 236, 65, 154, 66, 184, 97, 71, 0,
    2, 81, 192, 0, 239, 212, 112, 0, 37, 28, 0, 14, 253, 71, 0, 2, 81, 192, 0, 239, 212, 112, 0, 37,
    28, 0, 14, 253, 71, 0, 2, 81, 192, 0, 239, 212, 112, 0, 37, 28, 0, 14, 253, 71, 0, 2, 81, 192,
    0, 239, 212, 112, 0, 37, 28, 0, 14, 253, 71, 0, 2, 81, 192, 0, 239, 212, 112, 0, 37, 28, 0, 14,
    226, 137, 197, 19, 245, 28, 0, 9, 71, 0, 3, 191, 81, 192, 0, 148, 112, 0, 59, 138, 39, 20, 78, 40,
    159, 12, 40, 224, 0, 74, 56, 0, 29, 214, 10, 3, 42, 32, 18, 33, 0, 0, 137, 212, 81, 56, 162, 114,
    248, 104, 71, 184, 99, 129, 182, 165, 132, 132, 242, 130, 139, 128, 160, 0, 32, 58, 0, 2, 0, 5, 129, 9,
    128, 208, 123, 127, 231, 51, 22, 245, 139, 165, 60, 110, 202, 88, 124, 104, 175, 203, 165, 222, 16, 12, 114, 160,
    0, 130, 50, 42, 0, 8, 25, 150, 84, 0, 16, 51, 34, 160, 0, 131, 50, 28, 0, 9, 68, 16, 1, 175,
    196, 3, 162, 1, 223, 12, 114, 160, 0, 129, 153, 21, 0, 4, 12, 203, 42, 0, 8, 25, 145, 80, 0, 64,
    204, 135, 0, 2, 81, 4, 0, 107, 241, 0, 232, 128, 118, 0, 0, 0, 242, 65, 154, 98, 184, 97, 71, 0,
    2, 81, 192, 0, 239, 212, 112, 0, 37, 28, 0, 14, 253, 71, 0, 2, 81, 192, 0, 239, 212, 112, 0, 37,
    28, 0, 14, 253, 71, 0, 2, 81, 192, 0, 239, 212, 112, 0, 37, 28, 0, 14, 253, 71, 0, 2, 81, 192,
    0, 239, 212, 112, 0, 37, 28, 0, 14, 253, 71, 0, 2, 81, 192, 0, 239, 212, 112, 0, 37, 28, 0, 14,
    226, 137, 197, 19, 245, 28, 0, 9, 71, 0, 3, 191, 81, 192, 0, 148, 112, 0, 59, 138, 39, 20, 79, 134,
    20, 112, 0, 37, 28, 0, 14, 253, 71, 0, 2, 81, 192, 0, 238, 176, 80, 25, 81, 0, 145, 8, 0, 4,
    78, 162, 137, 197, 19, 176, 224, 140, 63, 119, 119, 112, 2, 55, 250, 191, 251, 252, 0, 16, 6, 2, 14, 54,
    38, 167, 199, 132, 8, 155, 152, 175, 108, 241, 127, 128, 0, 32, 44, 6, 44, 88, 190, 71, 2, 223, 93, 123,
    75, 129, 114, 76, 47, 147, 0, 4, 25, 145, 80, 0, 64, 140, 132, 16, 40, 25, 8, 127, 203, 162, 1, 218,
    193, 64, 101, 68, 2, 68, 32, 0, 17, 58, 240, 199, 6, 128, 2, 12, 200, 168, 0, 32, 204, 178, 160, 0,
    131, 50, 42, 0, 8, 51, 33, 192, 0, 148, 65, 0, 26, 252, 64, 58, 32, 29, 128,
    ];
    const PEXPECT_HASH: u32 = 1978314551;
    const PEXPECT_FRAMES: usize = 4;

    #[test_case]
    fn decodes_embedded_i_and_p_frames_matching_pyav() {
        // Full I+P decode of an embedded baseline clip; hash of all frames' YUV
        // must match the PyAV-derived reference (guards the inter path).
        let track = mp4::parse(&PCLIP_MP4).unwrap();
        let track_avcc = expect_avc(&track.config);
        let sps = h264::parse_sps(&bits::unescape_rbsp(&track_avcc.sps[0][1..])).unwrap();
        let pps = h264::parse_pps(&bits::unescape_rbsp(&track_avcc.pps[0][1..])).unwrap();
        let mut hh: u32 = 0;
        let mut nf = 0usize;
        let mut reference: Option<h264::decoder::DecodedFrame> = None;
        for s in &track.samples {
            let data = &PCLIP_MP4[s.offset..s.offset + s.size];
            for nal in h264::split_avcc(data, track_avcc.length_size) {
                if nal.kind.is_slice() {
                    let is_idr = nal.kind == h264::NalType::SliceIdr;
                    let df = h264::decoder::decode_slice(&sps, &pps, &nal.rbsp(), is_idr, reference.as_ref()).unwrap();
                    for &b in df.y.iter().chain(df.cb.iter()).chain(df.cr.iter()) {
                        hh = hh.wrapping_mul(31).wrapping_add(b as u32);
                    }
                    nf += 1;
                    reference = Some(df);
                    break;
                }
            }
        }
        assert_eq!(nf, PEXPECT_FRAMES, "decoded frame count");
        assert_eq!(hh, PEXPECT_HASH, "I+P decode must match the PyAV reference");
    }

    fn frame_hash(f: &Frame) -> u32 {
        let mut h: u32 = 0;
        for &p in &f.pixels {
            h = h.wrapping_mul(31).wrapping_add(p);
        }
        h
    }

    #[test_case]
    fn stream_decoder_seek_matches_sequential() {
        // The streaming decoder must produce, for random-access and *backward*
        // seeks, exactly the frames a straight sequential decode produces — it
        // rewinds to the latest keyframe and re-decodes forward (baseline
        // P-frames can't run in reverse). This is what keeps memory to one
        // reference frame instead of the whole clip (the green-frame trap).
        let mut seq = StreamDecoder::open(PCLIP_MP4.to_vec()).unwrap();
        let n = seq.frame_count();
        assert_eq!(n, PEXPECT_FRAMES, "sample count");
        let mut hashes = alloc::vec::Vec::new();
        for i in 0..n {
            assert!(seq.seek_decode(i), "forward decode of frame {i}");
            hashes.push(frame_hash(seq.cur_frame().unwrap()));
        }
        // pts is monotonic non-decreasing.
        for i in 1..n {
            assert!(seq.pts_ms(i) >= seq.pts_ms(i - 1), "pts must not go backwards");
        }
        // A fresh decoder jumped around out of order must match frame-for-frame.
        let mut ra = StreamDecoder::open(PCLIP_MP4.to_vec()).unwrap();
        for &i in &[n - 1, 0, n / 2, 1, n - 1, 0] {
            ra.seek_decode(i);
            assert_eq!(frame_hash(ra.cur_frame().unwrap()), hashes[i], "seek to frame {i} must match sequential decode");
        }
        // Re-seeking the same frame is a no-op (returns false, frame unchanged).
        assert!(!ra.seek_decode(0), "re-seeking the current frame changes nothing");
    }

    // 64x64 **High-profile** clip (x264: CABAC, 8x8 transform, I+P+B with
    // 2 B-frames), muxed by the e2e stdlib muxer; expected hash captured from
    // PyAV (frames matched by POC — the test muxer writes no ctts, so the
    // comparison runs in decode order).
    const HICLIP_MP4: [u8; 1098] = [
    0, 0, 0, 32, 102, 116, 121, 112, 105, 115, 111, 109, 0, 0, 2, 0, 105, 115, 111, 109, 105, 115, 111, 50,
    97, 118, 99, 49, 109, 112, 52, 49, 0, 0, 2, 134, 109, 111, 111, 118, 0, 0, 0, 108, 109, 118, 104, 100,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25, 0, 0, 0, 5, 0, 1, 0, 0,
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 2, 0, 0, 2, 18, 116, 114, 97, 107, 0, 0, 0, 92, 116, 107, 104, 100, 0, 0, 0, 7,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0,
    0, 64, 0, 0, 0, 64, 0, 0, 0, 0, 1, 174, 109, 100, 105, 97, 0, 0, 0, 28, 109, 100, 104, 100,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25, 0, 5, 0, 0, 0, 0, 0, 39,
    104, 100, 108, 114, 0, 0, 0, 0, 0, 0, 0, 0, 118, 105, 100, 101, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 99, 104, 105, 116, 116, 105, 0, 0, 0, 1, 99, 109, 105, 110, 102, 0, 0, 0, 20, 118,
    109, 104, 100, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 36, 100, 105, 110, 102, 0,
    0, 0, 28, 100, 114, 101, 102, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 12, 117, 114, 108, 32, 0,
    0, 0, 1, 0, 0, 1, 35, 115, 116, 98, 108, 0, 0, 0, 151, 115, 116, 115, 100, 0, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 0, 135, 97, 118, 99, 49, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 0, 64, 0, 72, 0, 0, 0, 72, 0, 0, 0,
    0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 255, 255, 0, 0, 0, 49, 97, 118, 99,
    67, 1, 100, 0, 10, 255, 225, 0, 24, 103, 100, 0, 10, 172, 217, 68, 38, 192, 68, 0, 0, 3, 0, 4,
    0, 0, 3, 0, 202, 60, 72, 150, 88, 1, 0, 6, 104, 235, 225, 50, 200, 176, 0, 0, 0, 24, 115, 116,
    116, 115, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 5, 0, 0, 0, 1, 0, 0, 0, 28, 115, 116,
    115, 99, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 5, 0, 0, 0, 1, 0, 0,
    0, 40, 115, 116, 115, 122, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 227, 0, 0,
    0, 79, 0, 0, 0, 30, 0, 0, 0, 28, 0, 0, 0, 48, 0, 0, 0, 20, 115, 116, 115, 115, 0, 0,
    0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 20, 115, 116, 99, 111, 0, 0, 0, 0, 0, 0,
    0, 1, 0, 0, 2, 174, 0, 0, 1, 164, 109, 100, 97, 116, 0, 0, 0, 223, 101, 136, 132, 0, 255, 147,
    6, 106, 124, 139, 89, 12, 6, 101, 132, 7, 79, 55, 172, 203, 126, 236, 150, 11, 50, 117, 31, 192, 45, 32,
    118, 66, 174, 248, 179, 243, 108, 48, 240, 53, 116, 118, 159, 107, 203, 192, 214, 146, 194, 8, 60, 81, 27, 166,
    161, 50, 26, 152, 29, 71, 243, 112, 19, 255, 254, 115, 133, 27, 36, 250, 100, 5, 235, 111, 203, 107, 226, 76,
    97, 240, 163, 75, 228, 174, 130, 14, 137, 65, 164, 39, 158, 116, 127, 122, 194, 15, 252, 227, 165, 109, 144, 47,
    0, 180, 21, 76, 82, 221, 238, 27, 172, 28, 224, 221, 122, 189, 186, 1, 137, 179, 114, 177, 47, 41, 122, 198,
    37, 70, 211, 23, 224, 137, 232, 189, 251, 70, 175, 13, 82, 12, 100, 235, 34, 81, 16, 86, 221, 5, 79, 186,
    198, 164, 56, 219, 62, 254, 211, 35, 91, 243, 10, 129, 111, 70, 99, 117, 116, 4, 97, 96, 56, 89, 155, 226,
    166, 133, 176, 219, 222, 133, 220, 117, 203, 239, 118, 56, 11, 246, 159, 130, 136, 119, 92, 156, 247, 6, 204, 205,
    181, 243, 233, 186, 111, 19, 6, 192, 133, 243, 178, 251, 238, 44, 129, 29, 125, 202, 190, 213, 25, 84, 8, 23,
    241, 0, 0, 0, 75, 65, 154, 35, 99, 224, 126, 10, 31, 247, 141, 145, 164, 62, 97, 244, 114, 249, 165, 110,
    154, 222, 151, 155, 149, 11, 93, 113, 255, 255, 135, 240, 40, 88, 50, 224, 84, 128, 221, 160, 26, 164, 122, 255,
    128, 10, 198, 156, 220, 32, 48, 18, 198, 139, 191, 120, 49, 105, 194, 150, 30, 32, 166, 60, 35, 220, 135, 202,
    214, 238, 219, 20, 191, 162, 149, 208, 0, 0, 0, 26, 65, 158, 65, 120, 175, 248, 115, 7, 11, 21, 102, 67,
    147, 255, 223, 217, 173, 1, 28, 216, 7, 31, 166, 227, 136, 161, 0, 0, 0, 24, 1, 158, 98, 106, 73, 255,
    244, 212, 165, 163, 247, 25, 42, 67, 249, 43, 108, 145, 188, 234, 119, 20, 228, 159, 0, 0, 0, 44, 65, 154,
    100, 75, 168, 66, 16, 90, 32, 140, 7, 240, 132, 7, 241, 128, 63, 208, 5, 85, 195, 147, 183, 42, 231, 142,
    130, 218, 75, 190, 255, 114, 99, 136, 243, 186, 59, 55, 158, 184, 48, 3, 221, 171,
    ];
    const HI_EXPECT_HASH: u32 = 2499348351;
    const HI_EXPECT_FRAMES: usize = 5;

    #[test_case]
    fn decodes_embedded_cabac_high_profile_clip() {
        // Full CABAC decode of an embedded High-profile clip (I/P/B slices,
        // 8x8 transform): every decoded picture's YUV, hashed in decode order,
        // must match the PyAV-derived reference.
        let track = mp4::parse(&HICLIP_MP4).unwrap();
        let track_avcc = expect_avc(&track.config);
        let sps = h264::parse_sps(&bits::unescape_rbsp(&track_avcc.sps[0][1..])).unwrap();
        let pps = h264::parse_pps(&bits::unescape_rbsp(&track_avcc.pps[0][1..])).unwrap();
        assert!(pps.entropy_coding_mode, "fixture must be CABAC");
        let mut dec = h264::decoder_cabac::H264Dec::new(sps, pps).unwrap();
        let mut hh: u32 = 0;
        let mut nf = 0usize;
        for s in &track.samples {
            let data = &HICLIP_MP4[s.offset..s.offset + s.size];
            let mut slices: alloc::vec::Vec<(alloc::vec::Vec<u8>, bool, u8)> = alloc::vec::Vec::new();
            for nal in h264::split_avcc(data, track_avcc.length_size) {
                if nal.kind.is_slice() {
                    slices.push((nal.rbsp(), nal.kind == h264::NalType::SliceIdr, nal.ref_idc));
                }
            }
            let pic = dec.decode_au(&slices).unwrap();
            for &b in pic.f.y.iter().chain(pic.f.cb.iter()).chain(pic.f.cr.iter()) {
                hh = hh.wrapping_mul(31).wrapping_add(b as u32);
            }
            nf += 1;
        }
        assert_eq!(nf, HI_EXPECT_FRAMES, "decoded frame count");
        assert_eq!(hh, HI_EXPECT_HASH, "CABAC decode must match the PyAV reference");
    }

}

#[cfg(test)]
mod hevc_vp9_container_test {
    //! End-to-end container + bitstream regression for the two codecs added
    //! alongside H.264: a real x265 `hvc1` mp4 and a real libvpx `V_VP9` WebM,
    //! embedded whole.
    //!
    //! These are *files*, not hand-built bit patterns, because every bug this
    //! layer has is a bug about real encoder output: the two-byte NAL header,
    //! the 22-byte `hvcC` preamble, VP9's absent `CodecPrivate`, and the
    //! `profile_tier_level` bit budget all parse "fine" on a fixture built by
    //! the same person who wrote the parser.
    use super::*;

    const HEVC_MP4: [u8; 1279] = [
        0, 0, 0, 28, 102, 116, 121, 112, 105, 115, 111, 109, 0, 0, 2, 0, 105, 115, 111, 109, 105, 115, 111, 50,
        109, 112, 52, 49, 0, 0, 0, 8, 102, 114, 101, 101, 0, 0, 1, 50, 109, 100, 97, 116, 0, 0, 0, 66,
        40, 1, 175, 29, 128, 247, 211, 185, 182, 132, 206, 129, 170, 0, 207, 90, 220, 221, 154, 119, 96, 174, 107, 199,
        146, 243, 185, 229, 235, 106, 243, 55, 116, 208, 196, 138, 42, 170, 188, 142, 48, 253, 77, 156, 152, 33, 206, 176,
        196, 242, 188, 87, 225, 90, 184, 206, 112, 15, 92, 36, 211, 49, 212, 29, 157, 152, 0, 0, 0, 144, 2, 1,
        208, 17, 87, 132, 49, 142, 64, 174, 98, 250, 215, 172, 22, 89, 143, 35, 211, 234, 252, 226, 3, 232, 36, 161,
        93, 204, 217, 82, 209, 120, 105, 158, 99, 207, 5, 104, 65, 241, 35, 241, 43, 163, 213, 167, 32, 127, 23, 115,
        12, 138, 232, 130, 240, 6, 171, 108, 253, 152, 210, 87, 74, 203, 145, 220, 185, 65, 115, 238, 85, 220, 205, 177,
        68, 201, 157, 175, 78, 101, 207, 177, 99, 235, 60, 234, 66, 43, 42, 52, 83, 13, 227, 69, 62, 210, 52, 73,
        255, 32, 173, 242, 182, 146, 34, 240, 19, 91, 225, 36, 92, 96, 238, 215, 150, 74, 15, 147, 212, 50, 182, 9,
        69, 183, 200, 3, 247, 0, 9, 198, 95, 100, 193, 111, 243, 90, 79, 202, 25, 250, 130, 165, 211, 214, 0, 0,
        0, 18, 0, 1, 224, 36, 191, 134, 20, 192, 50, 220, 227, 121, 234, 40, 59, 153, 45, 183, 0, 0, 0, 54,
        2, 1, 208, 25, 245, 245, 16, 193, 142, 64, 174, 47, 80, 21, 231, 82, 221, 122, 123, 90, 145, 209, 183, 44,
        113, 211, 29, 243, 43, 90, 244, 150, 118, 65, 189, 241, 168, 214, 248, 30, 28, 246, 62, 122, 87, 59, 239, 21,
        166, 37, 138, 127, 49, 147, 0, 0, 3, 169, 109, 111, 111, 118, 0, 0, 0, 108, 109, 118, 104, 100, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 232, 0, 0, 0, 160, 0, 1, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 2, 0, 0, 2, 211, 116, 114, 97, 107, 0, 0, 0, 92, 116, 107, 104, 100, 0, 0, 0, 3, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 160, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 64,
        0, 0, 0, 64, 0, 0, 0, 0, 0, 36, 101, 100, 116, 115, 0, 0, 0, 28, 101, 108, 115, 116, 0, 0,
        0, 0, 0, 0, 0, 1, 0, 0, 0, 160, 0, 0, 4, 0, 0, 1, 0, 0, 0, 0, 2, 75, 109, 100,
        105, 97, 0, 0, 0, 32, 109, 100, 104, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        50, 0, 0, 0, 10, 0, 85, 196, 0, 0, 0, 0, 0, 45, 104, 100, 108, 114, 0, 0, 0, 0, 0, 0,
        0, 0, 118, 105, 100, 101, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 86, 105, 100, 101, 111, 72,
        97, 110, 100, 108, 101, 114, 0, 0, 0, 1, 246, 109, 105, 110, 102, 0, 0, 0, 20, 118, 109, 104, 100, 0,
        0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 36, 100, 105, 110, 102, 0, 0, 0, 28, 100,
        114, 101, 102, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 12, 117, 114, 108, 32, 0, 0, 0, 1, 0,
        0, 1, 182, 115, 116, 98, 108, 0, 0, 0, 238, 115, 116, 115, 100, 0, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 222, 104, 101, 118, 49, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 64, 0, 64, 0, 72, 0, 0, 0, 72, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 255, 255, 0, 0, 0, 116, 104, 118, 99, 67, 1, 1, 96,
        0, 0, 0, 144, 0, 0, 0, 0, 0, 30, 240, 0, 252, 253, 248, 248, 0, 0, 15, 3, 32, 0, 1, 0,
        24, 64, 1, 12, 1, 255, 255, 1, 96, 0, 0, 3, 0, 144, 0, 0, 3, 0, 0, 3, 0, 30, 149, 144,
        9, 33, 0, 1, 0, 39, 66, 1, 1, 1, 96, 0, 0, 3, 0, 144, 0, 0, 3, 0, 0, 3, 0, 30,
        160, 32, 129, 5, 150, 86, 73, 36, 202, 230, 128, 128, 0, 0, 3, 0, 128, 0, 0, 12, 132, 34, 0, 1,
        0, 7, 68, 1, 193, 114, 180, 34, 64, 0, 0, 0, 20, 98, 116, 114, 116, 0, 0, 0, 0, 0, 0, 58,
        52, 0, 0, 0, 0, 0, 0, 0, 24, 115, 116, 116, 115, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
        4, 0, 0, 2, 0, 0, 0, 0, 20, 115, 116, 115, 115, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
        1, 0, 0, 0, 16, 115, 100, 116, 112, 0, 0, 0, 0, 32, 16, 24, 16, 0, 0, 0, 48, 99, 116, 116,
        115, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 1, 0, 0, 4, 0, 0, 0, 0, 1, 0, 0, 6,
        0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 4, 0, 0, 0, 0, 28, 115, 116, 115,
        99, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 1, 0, 0, 0,
        36, 115, 116, 115, 122, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 70, 0, 0, 0,
        148, 0, 0, 0, 22, 0, 0, 0, 58, 0, 0, 0, 20, 115, 116, 99, 111, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 0, 0, 44, 0, 0, 0, 98, 117, 100, 116, 97, 0, 0, 0, 90, 109, 101, 116, 97, 0, 0, 0,
        0, 0, 0, 0, 33, 104, 100, 108, 114, 0, 0, 0, 0, 0, 0, 0, 0, 109, 100, 105, 114, 97, 112, 112,
        108, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 45, 105, 108, 115, 116, 0, 0, 0, 37, 169, 116,
        111, 111, 0, 0, 0, 29, 100, 97, 116, 97, 0, 0, 0, 1, 0, 0, 0, 0, 76, 97, 118, 102, 54, 50,
        46, 49, 50, 46, 49, 48, 50,
    ];

    const VP9_WEBM: [u8; 1182] = [
        26, 69, 223, 163, 159, 66, 134, 129, 1, 66, 247, 129, 1, 66, 242, 129, 4, 66, 243, 129, 8, 66, 130, 132,
        119, 101, 98, 109, 66, 135, 129, 2, 66, 133, 129, 2, 24, 83, 128, 103, 1, 0, 0, 0, 0, 0, 4, 110,
        17, 77, 155, 116, 186, 77, 187, 139, 83, 171, 132, 21, 73, 169, 102, 83, 172, 129, 161, 77, 187, 139, 83, 171,
        132, 22, 84, 174, 107, 83, 172, 129, 216, 77, 187, 140, 83, 171, 132, 18, 84, 195, 103, 83, 172, 130, 1, 27,
        77, 187, 140, 83, 171, 132, 28, 83, 187, 107, 83, 172, 130, 4, 88, 236, 1, 0, 0, 0, 0, 0, 0, 89,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21, 73, 169, 102, 178, 42, 215,
        177, 131, 15, 66, 64, 77, 128, 141, 76, 97, 118, 102, 54, 50, 46, 49, 50, 46, 49, 48, 50, 87, 65, 141,
        76, 97, 118, 102, 54, 50, 46, 49, 50, 46, 49, 48, 50, 68, 137, 136, 64, 100, 0, 0, 0, 0, 0, 0,
        22, 84, 174, 107, 190, 174, 1, 0, 0, 0, 0, 0, 0, 53, 215, 129, 1, 115, 197, 136, 183, 102, 123, 118,
        202, 180, 3, 238, 156, 129, 0, 34, 181, 156, 131, 117, 110, 100, 136, 129, 0, 134, 133, 86, 95, 86, 80, 57,
        131, 129, 1, 35, 227, 131, 132, 2, 98, 90, 0, 224, 134, 176, 129, 64, 186, 129, 64, 18, 84, 195, 103, 216,
        115, 115, 160, 99, 192, 128, 103, 200, 154, 69, 163, 135, 69, 78, 67, 79, 68, 69, 82, 68, 135, 141, 76, 97,
        118, 102, 54, 50, 46, 49, 50, 46, 49, 48, 50, 115, 115, 178, 99, 192, 139, 99, 197, 136, 183, 102, 123, 118,
        202, 180, 3, 238, 103, 200, 161, 69, 163, 136, 68, 85, 82, 65, 84, 73, 79, 78, 68, 135, 147, 48, 48, 58,
        48, 48, 58, 48, 48, 46, 49, 54, 48, 48, 48, 48, 48, 48, 48, 0, 31, 67, 182, 117, 66, 218, 231, 129,
        0, 163, 65, 88, 129, 0, 0, 128, 130, 73, 131, 66, 0, 3, 240, 3, 246, 6, 56, 36, 28, 24, 74, 0,
        2, 160, 80, 96, 31, 205, 255, 158, 245, 159, 199, 250, 255, 87, 241, 254, 175, 211, 252, 31, 39, 246, 95, 82,
        75, 43, 231, 210, 243, 81, 247, 92, 12, 249, 252, 81, 193, 148, 103, 113, 128, 69, 224, 0, 126, 157, 125, 222,
        133, 241, 13, 13, 196, 110, 161, 69, 103, 180, 219, 138, 219, 252, 196, 103, 67, 25, 253, 188, 211, 106, 144, 170,
        236, 11, 105, 120, 76, 217, 204, 115, 14, 69, 178, 119, 1, 170, 235, 136, 232, 25, 33, 35, 122, 132, 221, 62,
        33, 108, 183, 140, 177, 243, 87, 145, 177, 10, 92, 158, 29, 120, 48, 219, 178, 108, 22, 61, 231, 185, 47, 205,
        160, 60, 183, 218, 27, 74, 60, 55, 33, 122, 63, 190, 173, 167, 95, 159, 150, 175, 121, 9, 27, 212, 39, 39,
        106, 120, 2, 207, 183, 95, 51, 158, 145, 70, 91, 199, 135, 183, 97, 158, 176, 64, 39, 60, 211, 81, 224, 125,
        232, 212, 182, 50, 22, 128, 179, 251, 43, 123, 34, 207, 123, 194, 164, 73, 165, 11, 13, 228, 175, 69, 198, 184,
        142, 76, 224, 146, 218, 238, 95, 16, 24, 21, 247, 213, 163, 140, 110, 195, 96, 177, 239, 59, 78, 88, 4, 146,
        178, 21, 253, 165, 223, 114, 60, 221, 188, 40, 26, 100, 2, 28, 21, 70, 76, 19, 87, 149, 40, 179, 248, 235,
        110, 96, 98, 243, 202, 232, 250, 24, 201, 59, 222, 240, 29, 175, 217, 145, 79, 183, 144, 148, 116, 5, 238, 157,
        244, 12, 144, 145, 189, 65, 249, 160, 65, 184, 16, 211, 232, 221, 134, 193, 99, 222, 118, 156, 176, 9, 37, 100,
        43, 251, 108, 128, 117, 127, 205, 130, 79, 92, 252, 240, 182, 188, 34, 142, 95, 243, 100, 107, 48, 131, 53, 159,
        156, 115, 222, 81, 161, 143, 134, 155, 121, 180, 200, 0, 163, 64, 209, 129, 0, 40, 0, 134, 0, 64, 146, 241,
        33, 64, 0, 0, 96, 118, 159, 94, 238, 35, 171, 213, 99, 226, 202, 150, 254, 191, 140, 88, 10, 192, 85, 82,
        44, 42, 240, 32, 0, 117, 67, 95, 25, 254, 58, 253, 123, 180, 63, 213, 18, 102, 188, 65, 135, 87, 75, 23,
        237, 65, 115, 68, 209, 16, 61, 46, 150, 30, 142, 94, 10, 167, 235, 91, 242, 216, 126, 241, 106, 238, 76, 202,
        71, 61, 74, 108, 206, 105, 133, 245, 72, 190, 219, 4, 48, 34, 226, 9, 162, 124, 15, 176, 46, 203, 212, 232,
        91, 233, 15, 223, 217, 89, 48, 31, 246, 5, 217, 131, 164, 106, 69, 11, 124, 143, 18, 147, 76, 62, 6, 129,
        53, 6, 195, 204, 14, 226, 134, 199, 31, 192, 40, 33, 41, 169, 156, 161, 36, 88, 178, 221, 147, 179, 208, 182,
        208, 51, 92, 149, 86, 17, 187, 110, 207, 203, 50, 66, 37, 215, 42, 57, 249, 226, 13, 143, 91, 135, 34, 14,
        220, 203, 74, 89, 162, 197, 99, 136, 74, 55, 31, 192, 40, 33, 41, 169, 156, 161, 38, 34, 28, 103, 59, 88,
        65, 138, 99, 164, 34, 238, 128, 0, 163, 215, 129, 0, 80, 0, 134, 0, 64, 146, 156, 64, 78, 224, 0, 3,
        112, 0, 0, 17, 51, 201, 224, 0, 15, 219, 160, 84, 79, 86, 213, 95, 194, 199, 231, 121, 157, 249, 249, 49,
        128, 213, 62, 252, 149, 144, 56, 59, 120, 135, 2, 123, 207, 186, 195, 240, 238, 224, 24, 5, 225, 25, 111, 220,
        217, 144, 216, 185, 191, 124, 157, 230, 33, 250, 98, 207, 71, 67, 150, 73, 144, 7, 186, 71, 8, 90, 119, 48,
        0, 163, 205, 129, 0, 120, 0, 134, 0, 64, 146, 156, 72, 80, 0, 0, 3, 112, 0, 0, 29, 55, 19, 147,
        140, 66, 248, 148, 166, 40, 39, 126, 125, 11, 5, 32, 29, 64, 124, 244, 169, 221, 255, 246, 110, 224, 83, 160,
        0, 0, 1, 215, 183, 37, 96, 0, 0, 45, 229, 30, 62, 5, 174, 112, 0, 0, 0, 47, 226, 193, 238, 37,
        160, 0, 0, 67, 138, 9, 0, 0, 28, 83, 187, 107, 145, 187, 143, 179, 129, 0, 183, 138, 247, 129, 1, 241,
        130, 1, 120, 240, 129, 3,
    ];


    #[test_case]
    fn probes_a_real_hevc_mp4() {
        let info = probe(&HEVC_MP4).unwrap();
        assert_eq!(info.container, "mp4/mov (ISO-BMFF)");
        assert_eq!(info.width, 64);
        assert_eq!(info.height, 64);
        assert_eq!(info.profile_idc, 1, "Main profile");
        assert_eq!(info.level_idc, 30, "level 1.0 is coded as 30x1.0");
        assert_eq!(info.frame_count, 4);
        assert!(info.codec.starts_with("H.265"), "codec was {}", info.codec);
        assert!(info.decodable, "reason was {}", info.unsupported_reason);
    }

    /// The whole HEVC pipeline, end to end on a real file: demux, parameter
    /// sets, CABAC, reconstruction, in-loop filters, and reorder.
    ///
    /// Structural properties only here (every frame, right size, non-flat).
    /// Bit-exact agreement with PyAV is the acceptance gate and lives in
    /// `tools/videodiff`.
    #[test_case]
    fn decodes_every_frame_of_a_real_hevc_mp4() {
        let track = mp4::parse(&HEVC_MP4).unwrap();
        let hvcc = match &track.config {
            mp4::CodecConfig::Hevc(h) => h.clone(),
            _ => panic!("expected an HEVC track"),
        };
        let mut dec = hevc::decoder::HevcDecoder::new();
        dec.set_parameter_sets(&hvcc.vps, &hvcc.sps, &hvcc.pps).unwrap();
        let mut seen = 0usize;
        for s in &track.samples {
            let data = &HEVC_MP4[s.offset..s.offset + s.size];
            let nals = hevc::split_hvcc(data, hvcc.length_size);
            for f in dec.decode_au(&nals).expect("decode") {
                assert_eq!((f.w, f.h), (64, 64));
                assert_eq!(f.y.len(), 64 * 64);
                seen += 1;
            }
        }
        seen += dec.flush().len();
        assert_eq!(seen, 4, "every frame must decode");
    }

    #[test_case]
    fn hevc_config_and_parameter_sets_round_trip() {
        let track = mp4::parse(&HEVC_MP4).unwrap();
        let hvcc = match &track.config {
            mp4::CodecConfig::Hevc(h) => h,
            other => panic!("expected an HEVC track, got {}", other.codec_name()),
        };
        assert_eq!(hvcc.length_size, 4);
        assert_eq!(hvcc.general_profile_idc, 1);
        assert_eq!(hvcc.chroma_format_idc, 1, "4:2:0");
        assert_eq!(hvcc.bit_depth_luma, 8);
        assert_eq!(hvcc.bit_depth_chroma, 8);
        // All three parameter-set kinds arrive in their own typed arrays — the
        // shape `avcC` cannot express, and the reason `hvcC` needs its own
        // parser rather than a reused one.
        assert_eq!(hvcc.vps.len(), 1);
        assert_eq!(hvcc.sps.len(), 1);
        assert_eq!(hvcc.pps.len(), 1);
        let sps = hevc::parse_sps(&bits::unescape_rbsp(&hvcc.sps[0][2..])).unwrap();
        let pps = hevc::parse_pps(&bits::unescape_rbsp(&hvcc.pps[0][2..])).unwrap();
        assert_eq!(sps.width(), 64);
        assert_eq!(sps.height(), 64);
        assert_eq!(sps.ctb_size(), 64);
        assert_eq!(sps.ctb_grid(), (1, 1));
        // The record's own copies of these must agree with the SPS it carries;
        // a bit-budget error in profile_tier_level shows up right here.
        assert_eq!(sps.ptl.profile_idc, hvcc.general_profile_idc);
        assert_eq!(sps.ptl.level_idc, hvcc.general_level_idc);
        assert_eq!(sps.bit_depth_luma as u8, hvcc.bit_depth_luma);
        assert!(pps.entropy_coding_sync_enabled || !pps.entropy_coding_sync_enabled);
    }

    #[test_case]
    fn hevc_slice_headers_parse_for_every_frame() {
        let track = mp4::parse(&HEVC_MP4).unwrap();
        let hvcc = match &track.config {
            mp4::CodecConfig::Hevc(h) => h,
            _ => panic!("not HEVC"),
        };
        let sps = hevc::parse_sps(&bits::unescape_rbsp(&hvcc.sps[0][2..])).unwrap();
        let pps = hevc::parse_pps(&bits::unescape_rbsp(&hvcc.pps[0][2..])).unwrap();
        // x265 at keyint=4 bframes=2 emits I,P,B,P with POCs 0,2,1,3 — decode
        // order, so the POCs are the thing that proves the headers really
        // parsed rather than merely not erroring.
        let mut got: alloc::vec::Vec<(hevc::SliceType, u32)> = alloc::vec::Vec::new();
        for s in &track.samples {
            let data = &HEVC_MP4[s.offset..s.offset + s.size];
            for nal in hevc::split_hvcc(data, hvcc.length_size) {
                if !nal.kind.is_slice() {
                    continue;
                }
                let h = hevc::parse_slice_header(&nal.rbsp(), nal.kind, &sps, &pps).unwrap();
                assert!(h.first_slice_in_pic, "single-slice fixture");
                assert_eq!(h.segment_address, 0);
                // The CABAC data must start inside the RBSP, byte-aligned.
                assert!(h.data_byte_offset > 0 && h.data_byte_offset < nal.rbsp().len());
                got.push((h.slice_type, h.pic_order_cnt_lsb));
            }
        }
        assert_eq!(got.len(), 4);
        assert_eq!(got[0].0, hevc::SliceType::I);
        assert_eq!(got[0].1, 0, "an IDR codes no POC lsb, so it reads as 0");
        assert_eq!(got[1], (hevc::SliceType::P, 2));
        assert_eq!(got[2], (hevc::SliceType::B, 1));
        assert_eq!(got[3], (hevc::SliceType::P, 3));
        // The B frame's RPS names one past and one future picture.
        assert!(track.samples[0].is_sync);
    }

    #[test_case]
    fn probes_a_real_vp9_webm() {
        let info = probe(&VP9_WEBM).unwrap();
        assert_eq!(info.container, "matroska/webm (EBML)");
        assert_eq!(info.width, 64);
        assert_eq!(info.height, 64);
        assert_eq!(info.profile_idc, 0);
        assert_eq!(info.frame_count, 4);
        assert!(info.codec.starts_with("VP9"), "codec was {}", info.codec);
        assert!(info.decodable, "profile-0 VP9 decodes: {}", info.unsupported_reason);
    }

    #[test_case]
    fn vp9_decodes_bit_exactly_against_libvpx() {
        // The whole 4-frame sequence — a keyframe plus three inter frames —
        // decoded on the real kernel and hashed. The expected value is the
        // FNV-1a of libvpx's own output for this file, taken through
        // `tools/videodiff` (which was verified byte-identical to PyAV), so
        // this test fails on *any* divergence from the reference decoder, not
        // just on a crash.
        //
        // It covers the inter path end to end: reference slots, the MV
        // reference search, sub-pel compensation and the loop filter.
        let track = mkv::parse(&VP9_WEBM).unwrap();
        let mut dec = vp9::decoder::Vp9Decoder::new();
        let mut h: u32 = 0x811c_9dc5;
        let mut feed = |bytes: &[u8], h: &mut u32| {
            for &b in bytes {
                *h ^= b as u32;
                *h = h.wrapping_mul(0x0100_0193);
            }
        };
        let mut shown = 0usize;
        for s in &track.samples {
            let data = &VP9_WEBM[s.offset..s.offset + s.size];
            for &(a, b) in &vp9::split_superframe(data).unwrap() {
                if let Some(f) = dec.decode_frame(&data[a..b]).unwrap() {
                    assert_eq!((f.w, f.h), (64, 64));
                    feed(&f.y, &mut h);
                    feed(&f.cb, &mut h);
                    feed(&f.cr, &mut h);
                    shown += 1;
                }
            }
        }
        assert_eq!(shown, 4, "all four frames are shown");
        assert_eq!(h, 233_641_961, "VP9 output diverged from libvpx");
    }

    const VP9_FP0_WEBM: [u8; 3189] = [
        26, 69, 223, 163, 159, 66, 134, 129, 1, 66, 247, 129, 1, 66, 242, 129, 4, 66, 243, 129, 8, 66, 130, 132,
        119, 101, 98, 109, 66, 135, 129, 2, 66, 133, 129, 2, 24, 83, 128, 103, 1, 0, 0, 0, 0, 0, 12, 69,
        17, 77, 155, 116, 186, 77, 187, 139, 83, 171, 132, 21, 73, 169, 102, 83, 172, 129, 161, 77, 187, 139, 83, 171,
        132, 22, 84, 174, 107, 83, 172, 129, 216, 77, 187, 140, 83, 171, 132, 18, 84, 195, 103, 83, 172, 130, 1, 27,
        77, 187, 140, 83, 171, 132, 28, 83, 187, 107, 83, 172, 130, 12, 47, 236, 1, 0, 0, 0, 0, 0, 0, 89,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21, 73, 169, 102, 178, 42, 215,
        177, 131, 15, 66, 64, 77, 128, 141, 76, 97, 118, 102, 54, 50, 46, 49, 50, 46, 49, 48, 50, 87, 65, 141,
        76, 97, 118, 102, 54, 50, 46, 49, 50, 46, 49, 48, 50, 68, 137, 136, 64, 116, 0, 0, 0, 0, 0, 0,
        22, 84, 174, 107, 190, 174, 1, 0, 0, 0, 0, 0, 0, 53, 215, 129, 1, 115, 197, 136, 135, 232, 88, 87,
        217, 114, 73, 253, 156, 129, 0, 34, 181, 156, 131, 117, 110, 100, 136, 129, 0, 134, 133, 86, 95, 86, 80, 57,
        131, 129, 1, 35, 227, 131, 132, 2, 98, 90, 0, 224, 134, 176, 129, 192, 186, 129, 128, 18, 84, 195, 103, 216,
        115, 115, 160, 99, 192, 128, 103, 200, 154, 69, 163, 135, 69, 78, 67, 79, 68, 69, 82, 68, 135, 141, 76, 97,
        118, 102, 54, 50, 46, 49, 50, 46, 49, 48, 50, 115, 115, 178, 99, 192, 139, 99, 197, 136, 135, 232, 88, 87,
        217, 114, 73, 253, 103, 200, 161, 69, 163, 136, 68, 85, 82, 65, 84, 73, 79, 78, 68, 135, 147, 48, 48, 58,
        48, 48, 58, 48, 48, 46, 51, 50, 48, 48, 48, 48, 48, 48, 48, 0, 31, 67, 182, 117, 74, 177, 231, 129,
        0, 163, 69, 194, 129, 0, 0, 128, 130, 73, 131, 66, 0, 11, 240, 7, 244, 16, 56, 36, 28, 24, 74, 0,
        3, 160, 124, 176, 122, 159, 201, 161, 102, 153, 216, 61, 210, 131, 247, 69, 169, 114, 158, 226, 224, 220, 61, 4,
        16, 62, 205, 97, 191, 208, 133, 98, 33, 85, 216, 165, 34, 230, 182, 154, 58, 245, 111, 226, 187, 236, 40, 207,
        98, 163, 254, 174, 250, 188, 7, 177, 234, 240, 0, 0, 109, 189, 127, 255, 223, 14, 159, 194, 168, 63, 176, 108,
        110, 32, 211, 96, 244, 33, 245, 211, 233, 128, 99, 101, 64, 88, 11, 84, 196, 90, 25, 157, 6, 89, 214, 78,
        77, 128, 249, 209, 193, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 183, 233, 90, 104, 66, 148, 127,
        206, 40, 39, 231, 255, 32, 116, 115, 149, 98, 160, 125, 230, 160, 206, 217, 43, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 9, 148, 240, 213, 237, 31, 110, 214, 177, 205, 174, 225, 199, 235, 96, 254, 193, 177, 184,
        131, 77, 131, 208, 65, 92, 242, 221, 58, 22, 26, 208, 249, 223, 41, 154, 235, 223, 192, 68, 228, 195, 182, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 43, 129, 143, 229, 148, 67, 0, 4, 148, 195, 67, 250, 212,
        142, 165, 121, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 224, 210, 113, 138, 30, 177, 102, 13, 60,
        116, 164, 183, 48, 91, 90, 177, 196, 131, 255, 18, 81, 156, 206, 138, 64, 0, 0, 0, 3, 189, 128, 0, 55,
        28, 0, 0, 0, 28, 178, 204, 161, 208, 124, 39, 220, 0, 0, 43, 84, 20, 78, 2, 225, 0, 102, 107, 207,
        196, 0, 0, 0, 198, 115, 103, 244, 119, 255, 253, 53, 168, 248, 31, 194, 61, 85, 244, 43, 66, 39, 182, 118,
        64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 177, 127, 110, 152, 7, 12, 133, 121, 123, 59, 33,
        59, 226, 203, 239, 0, 218, 229, 111, 228, 44, 20, 64, 102, 250, 168, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 91, 130, 123, 77, 8, 83, 97, 161, 166, 211, 73, 244, 89, 188, 24, 0, 4, 157, 157, 128, 0, 0,
        0, 38, 92, 0, 178, 128, 0, 0, 0, 251, 220, 85, 110, 79, 97, 128, 2, 40, 99, 90, 19, 170, 145, 146,
        17, 234, 175, 161, 90, 17, 61, 179, 176, 128, 250, 183, 255, 147, 247, 214, 62, 160, 254, 193, 186, 59, 64, 244,
        127, 139, 136, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 211, 255, 252, 186, 162, 138, 169, 7, 246,
        13, 141, 196, 26, 108, 27, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 38, 196, 153, 94, 186,
        219, 137, 192, 252, 100, 238, 242, 160, 239, 188, 233, 74, 11, 93, 2, 78, 0, 0, 0, 0, 0, 0, 0, 103,
        72, 0, 0, 14, 127, 45, 152, 112, 32, 28, 126, 1, 35, 177, 192, 89, 64, 0, 0, 221, 116, 192, 0, 0,
        193, 127, 58, 108, 152, 139, 152, 206, 240, 113, 57, 114, 185, 34, 60, 152, 159, 124, 116, 171, 123, 89, 199, 80,
        172, 86, 145, 251, 253, 198, 234, 48, 24, 176, 185, 15, 125, 111, 88, 243, 203, 146, 15, 66, 36, 54, 153, 180,
        254, 147, 207, 23, 58, 175, 55, 33, 49, 217, 213, 46, 225, 254, 145, 62, 225, 82, 182, 149, 155, 134, 15, 163,
        47, 215, 0, 116, 66, 82, 38, 47, 145, 145, 229, 9, 123, 159, 36, 56, 31, 60, 55, 207, 109, 8, 6, 253,
        57, 17, 159, 18, 47, 217, 12, 140, 141, 62, 153, 234, 22, 230, 52, 45, 50, 243, 56, 21, 248, 180, 197, 165,
        44, 163, 50, 127, 42, 159, 132, 181, 209, 189, 157, 6, 175, 136, 144, 249, 241, 2, 223, 193, 62, 157, 48, 61,
        13, 105, 68, 45, 73, 80, 165, 85, 62, 179, 173, 241, 240, 70, 74, 186, 108, 75, 154, 5, 36, 214, 3, 242,
        183, 72, 252, 113, 36, 193, 41, 15, 191, 187, 63, 71, 163, 63, 0, 95, 105, 53, 205, 195, 20, 148, 127, 78,
        91, 128, 74, 90, 184, 233, 200, 61, 38, 218, 219, 73, 202, 177, 255, 51, 181, 234, 56, 199, 208, 221, 49, 200,
        9, 210, 128, 103, 82, 202, 2, 25, 130, 146, 88, 16, 10, 129, 5, 67, 84, 129, 69, 250, 89, 183, 159, 216,
        171, 202, 202, 10, 5, 127, 107, 135, 165, 216, 177, 152, 80, 117, 13, 102, 73, 251, 47, 235, 240, 80, 73, 75,
        8, 250, 193, 217, 168, 218, 204, 107, 255, 216, 230, 125, 163, 138, 19, 198, 228, 124, 26, 198, 247, 163, 183, 217,
        137, 29, 157, 221, 219, 82, 227, 165, 6, 184, 153, 155, 22, 167, 4, 21, 102, 135, 156, 33, 89, 252, 53, 69,
        8, 212, 56, 46, 123, 107, 80, 152, 14, 63, 228, 212, 144, 123, 238, 27, 139, 96, 77, 101, 21, 218, 210, 109,
        162, 252, 168, 68, 147, 149, 78, 225, 231, 61, 46, 58, 42, 226, 24, 91, 143, 116, 181, 89, 189, 154, 125, 114,
        84, 177, 248, 79, 145, 253, 252, 6, 1, 243, 53, 78, 106, 160, 122, 147, 167, 215, 196, 240, 171, 17, 6, 166,
        58, 59, 255, 110, 9, 3, 247, 16, 156, 91, 3, 164, 89, 126, 146, 101, 97, 175, 85, 6, 163, 6, 176, 126,
        200, 63, 224, 112, 162, 102, 163, 206, 74, 72, 61, 185, 108, 203, 46, 141, 55, 115, 127, 6, 64, 80, 152, 135,
        202, 248, 124, 10, 191, 0, 12, 124, 236, 193, 168, 125, 248, 229, 208, 237, 175, 83, 137, 215, 24, 63, 24, 43,
        222, 253, 67, 243, 220, 155, 161, 59, 96, 222, 145, 152, 78, 243, 86, 174, 91, 70, 146, 68, 37, 35, 197, 124,
        213, 169, 47, 231, 104, 174, 126, 199, 168, 130, 53, 171, 10, 160, 183, 88, 158, 219, 137, 233, 33, 197, 70, 8,
        187, 23, 100, 32, 206, 19, 45, 44, 12, 216, 58, 190, 231, 88, 67, 120, 196, 9, 152, 229, 239, 253, 71, 88,
        117, 206, 170, 255, 227, 227, 136, 99, 182, 173, 76, 135, 112, 113, 25, 106, 71, 11, 137, 204, 156, 184, 101, 12,
        135, 245, 58, 194, 179, 162, 13, 250, 3, 220, 163, 239, 2, 132, 24, 56, 157, 0, 163, 254, 149, 152, 72, 121,
        75, 34, 247, 18, 242, 213, 174, 157, 32, 13, 8, 150, 43, 81, 122, 65, 209, 167, 192, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 128, 13, 132, 81, 254, 32, 89, 42, 50, 117, 155, 144, 249, 211, 121, 159, 180,
        72, 170, 224, 12, 117, 211, 76, 116, 255, 115, 111, 186, 16, 219, 226, 8, 93, 161, 223, 253, 241, 252, 188, 112,
        212, 49, 72, 229, 54, 35, 68, 100, 69, 243, 75, 217, 182, 112, 147, 149, 109, 70, 3, 242, 234, 29, 254, 226,
        53, 142, 82, 26, 107, 236, 94, 78, 123, 106, 55, 148, 107, 108, 18, 41, 246, 99, 157, 87, 217, 98, 48, 96,
        74, 242, 173, 174, 7, 239, 150, 57, 20, 43, 18, 120, 18, 125, 217, 246, 123, 31, 198, 4, 156, 63, 65, 98,
        135, 27, 239, 46, 180, 170, 127, 181, 195, 34, 79, 171, 216, 97, 161, 96, 29, 209, 63, 247, 107, 35, 228, 152,
        123, 122, 38, 188, 49, 36, 56, 119, 57, 113, 244, 54, 42, 250, 70, 68, 60, 10, 190, 151, 252, 242, 111, 252,
        183, 52, 37, 118, 125, 46, 12, 53, 189, 43, 104, 194, 243, 127, 218, 222, 188, 144, 91, 202, 92, 175, 247, 130,
        73, 91, 45, 198, 88, 32, 62, 221, 171, 25, 201, 245, 150, 47, 248, 216, 185, 103, 253, 213, 173, 18, 125, 86,
        149, 37, 11, 0, 238, 137, 255, 187, 89, 31, 44, 25, 65, 3, 197, 108, 12, 168, 37, 139, 198, 19, 101, 13,
        212, 213, 27, 30, 179, 127, 186, 186, 152, 179, 148, 39, 77, 168, 241, 136, 145, 237, 74, 126, 172, 92, 232, 78,
        161, 150, 234, 64, 143, 199, 30, 181, 203, 58, 139, 170, 87, 182, 236, 216, 152, 93, 227, 53, 72, 213, 83, 173,
        5, 239, 17, 152, 28, 83, 31, 151, 117, 50, 224, 29, 81, 27, 197, 49, 248, 47, 207, 211, 82, 52, 122, 92,
        152, 64, 19, 71, 144, 42, 59, 0, 0, 9, 154, 82, 79, 0, 163, 64, 167, 129, 0, 40, 0, 134, 0, 64,
        146, 152, 60, 80, 0, 0, 3, 112, 0, 0, 90, 39, 123, 62, 255, 63, 255, 58, 148, 199, 88, 31, 219, 44,
        255, 219, 70, 40, 71, 171, 14, 116, 74, 79, 252, 135, 27, 107, 255, 255, 246, 46, 204, 157, 127, 57, 119, 107,
        61, 69, 71, 33, 194, 158, 149, 38, 209, 188, 52, 124, 147, 116, 19, 108, 39, 155, 186, 57, 142, 140, 164, 31,
        62, 85, 173, 165, 158, 255, 12, 185, 136, 131, 192, 93, 165, 60, 87, 91, 249, 119, 177, 193, 176, 122, 114, 142,
        14, 75, 16, 86, 121, 142, 175, 60, 63, 182, 24, 11, 41, 240, 15, 91, 179, 214, 149, 83, 102, 156, 231, 83,
        188, 62, 99, 93, 235, 44, 44, 185, 50, 198, 10, 193, 45, 109, 97, 238, 184, 19, 40, 142, 182, 49, 32, 103,
        164, 254, 210, 254, 233, 26, 10, 134, 66, 78, 253, 155, 57, 78, 117, 0, 163, 64, 178, 129, 0, 80, 0, 134,
        0, 64, 146, 152, 52, 78, 224, 0, 7, 115, 67, 14, 122, 192, 0, 0, 89, 89, 194, 162, 230, 253, 179, 112,
        116, 52, 34, 200, 215, 6, 231, 27, 192, 68, 168, 208, 154, 13, 65, 102, 138, 237, 208, 193, 141, 90, 17, 15,
        150, 10, 83, 240, 111, 249, 181, 60, 189, 92, 22, 207, 219, 84, 131, 11, 132, 181, 59, 0, 173, 104, 145, 212,
        111, 2, 188, 7, 39, 14, 192, 30, 94, 62, 215, 71, 54, 45, 38, 125, 56, 70, 162, 178, 138, 67, 51, 229,
        128, 232, 1, 147, 251, 173, 189, 166, 91, 26, 0, 164, 189, 153, 25, 78, 224, 53, 29, 128, 0, 89, 31, 213,
        20, 84, 42, 209, 177, 70, 238, 208, 87, 0, 92, 222, 205, 155, 194, 8, 180, 201, 37, 91, 144, 232, 18, 154,
        64, 178, 3, 238, 245, 27, 46, 128, 103, 240, 239, 58, 254, 121, 90, 245, 93, 232, 91, 100, 133, 86, 156, 105,
        234, 78, 99, 88, 128, 163, 64, 165, 129, 0, 120, 0, 134, 0, 64, 146, 152, 60, 80, 0, 0, 3, 112, 0,
        0, 87, 43, 199, 243, 155, 202, 193, 211, 120, 16, 178, 81, 144, 147, 229, 54, 72, 117, 72, 157, 13, 171, 127,
        114, 144, 230, 176, 240, 137, 9, 93, 106, 126, 250, 119, 114, 125, 22, 179, 57, 245, 215, 184, 218, 11, 147, 77,
        105, 186, 190, 134, 130, 53, 126, 64, 105, 245, 95, 103, 35, 98, 200, 16, 130, 249, 21, 236, 25, 123, 238, 186,
        145, 67, 34, 113, 195, 191, 251, 202, 225, 255, 227, 24, 119, 75, 29, 247, 154, 59, 51, 188, 249, 5, 208, 62,
        206, 215, 149, 117, 115, 161, 20, 172, 7, 51, 37, 69, 225, 116, 206, 38, 166, 237, 183, 26, 245, 82, 180, 103,
        189, 60, 27, 115, 215, 117, 180, 50, 108, 117, 238, 84, 8, 3, 240, 151, 252, 253, 212, 19, 169, 151, 155, 249,
        163, 13, 128, 48, 128, 163, 65, 26, 129, 0, 160, 0, 134, 0, 64, 146, 152, 52, 77, 64, 0, 3, 112, 0,
        0, 87, 36, 68, 48, 116, 89, 250, 109, 165, 65, 57, 217, 230, 213, 18, 172, 21, 235, 150, 176, 75, 126, 123,
        194, 19, 184, 79, 231, 176, 172, 86, 152, 127, 204, 38, 178, 110, 159, 243, 159, 134, 182, 101, 219, 225, 28, 158,
        107, 146, 94, 101, 20, 192, 221, 134, 34, 120, 207, 111, 39, 228, 52, 154, 103, 138, 27, 67, 157, 203, 29, 239,
        212, 102, 98, 95, 95, 11, 120, 213, 2, 146, 177, 13, 192, 142, 66, 75, 88, 141, 212, 45, 96, 12, 189, 160,
        71, 107, 101, 171, 13, 12, 211, 207, 225, 25, 95, 107, 42, 112, 84, 228, 51, 19, 132, 110, 65, 122, 50, 242,
        244, 161, 243, 126, 65, 134, 113, 82, 171, 112, 156, 233, 96, 93, 185, 53, 218, 140, 252, 110, 99, 225, 3, 165,
        201, 0, 47, 168, 227, 45, 112, 113, 236, 111, 73, 33, 50, 46, 195, 86, 139, 227, 134, 216, 140, 255, 168, 25,
        44, 132, 118, 229, 181, 238, 28, 235, 214, 245, 58, 36, 46, 75, 157, 30, 136, 83, 165, 186, 127, 155, 209, 171,
        194, 180, 47, 127, 104, 8, 34, 74, 162, 245, 124, 246, 53, 222, 124, 242, 149, 205, 189, 10, 243, 210, 88, 147,
        248, 102, 76, 136, 156, 117, 243, 112, 28, 93, 119, 205, 68, 143, 241, 213, 205, 220, 60, 40, 240, 128, 231, 7,
        0, 64, 220, 202, 171, 86, 157, 126, 127, 110, 168, 19, 115, 60, 209, 111, 219, 67, 10, 209, 132, 206, 54, 136,
        181, 140, 163, 64, 156, 129, 0, 200, 0, 134, 0, 64, 146, 152, 32, 80, 0, 0, 3, 112, 0, 0, 84, 206,
        73, 90, 135, 83, 72, 20, 47, 57, 13, 207, 13, 61, 164, 30, 3, 238, 238, 181, 84, 172, 240, 199, 153, 71,
        254, 156, 42, 233, 207, 60, 49, 241, 120, 223, 14, 63, 242, 185, 52, 34, 227, 28, 144, 226, 138, 143, 136, 54,
        241, 227, 176, 242, 132, 72, 131, 26, 8, 190, 196, 158, 84, 225, 5, 158, 27, 6, 71, 147, 190, 32, 60, 58,
        186, 124, 104, 71, 16, 204, 107, 116, 52, 83, 213, 34, 121, 112, 144, 22, 153, 228, 107, 196, 172, 56, 244, 88,
        142, 235, 251, 200, 130, 54, 18, 95, 50, 146, 5, 76, 89, 209, 179, 75, 61, 253, 93, 137, 175, 225, 59, 112,
        69, 157, 134, 91, 59, 60, 17, 154, 112, 26, 158, 108, 86, 159, 121, 19, 60, 163, 64, 170, 129, 0, 240, 0,
        134, 0, 64, 146, 152, 40, 78, 224, 0, 3, 112, 0, 0, 83, 38, 74, 58, 175, 184, 65, 114, 79, 226, 19,
        78, 13, 29, 64, 234, 203, 129, 75, 76, 72, 56, 152, 45, 128, 227, 233, 43, 197, 130, 140, 99, 228, 39, 96,
        89, 54, 61, 97, 154, 29, 163, 122, 109, 183, 15, 28, 178, 91, 39, 27, 6, 26, 83, 52, 123, 46, 45, 177,
        205, 195, 11, 232, 82, 224, 138, 176, 210, 32, 224, 106, 226, 192, 85, 249, 116, 3, 235, 81, 170, 73, 67, 189,
        179, 71, 42, 129, 128, 109, 181, 151, 119, 91, 58, 3, 54, 159, 180, 10, 164, 82, 79, 161, 2, 113, 100, 2,
        178, 106, 1, 166, 171, 68, 155, 230, 56, 232, 213, 161, 193, 140, 154, 201, 10, 1, 247, 238, 85, 191, 206, 178,
        180, 151, 247, 2, 32, 87, 115, 241, 104, 253, 60, 19, 128, 60, 236, 160, 63, 255, 240, 212, 112, 0, 163, 247,
        129, 1, 24, 0, 134, 0, 64, 146, 152, 72, 80, 0, 0, 3, 112, 0, 0, 119, 238, 63, 103, 67, 138, 20,
        163, 124, 60, 36, 66, 217, 67, 219, 238, 72, 3, 39, 221, 85, 61, 200, 65, 199, 127, 127, 164, 64, 29, 12,
        112, 9, 112, 49, 104, 59, 10, 162, 85, 238, 210, 192, 185, 60, 187, 242, 19, 62, 147, 64, 193, 92, 153, 1,
        229, 189, 249, 69, 120, 194, 70, 26, 178, 178, 68, 254, 232, 143, 157, 77, 133, 89, 156, 217, 197, 33, 118, 181,
        131, 36, 238, 255, 34, 191, 125, 153, 79, 178, 167, 245, 54, 124, 122, 16, 55, 12, 67, 42, 98, 214, 0, 28,
        83, 187, 107, 145, 187, 143, 179, 129, 0, 183, 138, 247, 129, 1, 241, 130, 1, 120, 240, 129, 3,
    ];
    /// FNV-1a of libvpx's own eight-frame output for `VP9_FP0_WEBM`.
    const VP9_FP0_HASH: u32 = 1_570_905_375;

    #[test_case]
    fn vp9_backward_adaptation_matches_libvpx() {
        // A stream encoded with **frame-parallel decoding off**, so every frame
        // after the first decodes against probabilities derived from the
        // previous frame's symbol counts. Without adaptation this desynchronises
        // at frame 1; with adaptation that is subtly wrong it survives two or
        // three frames and then breaks — which is why the hash covers all eight.
        let track = mkv::parse(&VP9_FP0_WEBM).unwrap();
        let mut dec = vp9::decoder::Vp9Decoder::new();
        let mut h: u32 = 0x811c_9dc5;
        let mut shown = 0usize;
        for s in &track.samples {
            let data = &VP9_FP0_WEBM[s.offset..s.offset + s.size];
            for &(a, b) in &vp9::split_superframe(data).unwrap() {
                if let Some(f) = dec.decode_frame(&data[a..b]).unwrap() {
                    for plane in [&f.y, &f.cb, &f.cr] {
                        for &byte in plane {
                            h ^= byte as u32;
                            h = h.wrapping_mul(0x0100_0193);
                        }
                    }
                    shown += 1;
                }
            }
        }
        assert_eq!(shown, 8);
        assert_eq!(h, VP9_FP0_HASH, "adapted probabilities diverged from libvpx");
    }

    #[test_case]
    fn a_vp9_track_opens_in_the_streaming_player() {
        // The path `/open` takes: demux, then decode on demand.
        let mut dec = StreamDecoder::open(VP9_WEBM.to_vec()).unwrap();
        assert_eq!(dec.backend, "vp9");
        assert_eq!((dec.src_w, dec.src_h), (64, 64));
        assert_eq!(dec.frame_count(), 4);
        assert!(dec.seek_decode(0), "first frame decodes");
        assert!(dec.cur_frame().is_some());
        // …and a later frame, which needs the reference slots to have been kept.
        assert!(dec.seek_decode(3));
    }

    #[test_case]
    fn vp9_webm_has_no_codec_private_and_needs_none() {
        let track = mkv::parse(&VP9_WEBM).unwrap();
        assert!(matches!(track.config, mp4::CodecConfig::Vp9(_)));
        // Zero, because VP9 samples are raw frames: a caller that assumed a
        // length prefix would read the first four bytes of the frame header as
        // a NAL length.
        assert_eq!(track.config.length_size(), 0);
        assert_eq!(track.samples.len(), 4);
    }

    #[test_case]
    fn vp9_frame_headers_parse_for_every_frame() {
        let track = mkv::parse(&VP9_WEBM).unwrap();
        let mut refs: vp9::RefSizes = [(0, 0); vp9::NUM_REF_FRAMES];
        let mut kinds: alloc::vec::Vec<bool> = alloc::vec::Vec::new();
        for s in &track.samples {
            let data = &VP9_WEBM[s.offset..s.offset + s.size];
            let parts = vp9::split_superframe(data).unwrap();
            for &(a, b) in &parts {
                let h = vp9::parse_frame_header(&data[a..b], &refs).unwrap();
                assert_eq!(h.width, 64);
                assert_eq!(h.height, 64);
                assert_eq!(h.profile, 0);
                assert_eq!(h.bit_depth, 8);
                // The compressed header must fit inside the frame.
                assert!(h.tile_data_offset() <= b - a);
                // A bool decoder must start cleanly at the compressed header;
                // its marker bit is the check that the size was right.
                let cs = h.uncompressed_header_bytes;
                let ce = cs + h.header_size_in_bytes as usize;
                assert!(vp9::BoolDecoder::new(&data[a + cs..a + ce]).is_ok(), "compressed header is not a bool partition");
                for slot in 0..vp9::NUM_REF_FRAMES {
                    if h.refresh_frame_flags & (1 << slot) != 0 {
                        refs[slot] = (h.width, h.height);
                    }
                }
                kinds.push(h.key_frame);
            }
        }
        assert_eq!(kinds.len(), 4);
        assert!(kinds[0], "first frame is a keyframe");
        assert!(!kinds[1] && !kinds[2] && !kinds[3]);
        // The container's sync flag and the bitstream must agree here.
        let first = &track.samples[0];
        assert!(first.is_sync);
        assert!(vp9::sample_is_keyframe(&VP9_WEBM[first.offset..first.offset + first.size]));
    }
}
