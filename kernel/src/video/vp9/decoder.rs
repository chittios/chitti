//! VP9 frame decode: tile layout, the superblock loop, and the YUV output.
//!
//! Scope today is **intra frames** — keyframes and `intra_only` frames. Inter
//! prediction (reference motion vectors, 8-tap compensation, compound
//! prediction) and backward probability adaptation are not built, and a frame
//! that needs them is refused rather than decoded without them: an inter frame
//! decoded as if it were intra produces a full-frame garbage picture, which
//! looks like a decoder crash rather than a missing feature.
//!
//! Two structural notes:
//!
//! * **Tiles are independent by construction.** Each resets the above contexts
//!   over its own column range and the left contexts every superblock row, and
//!   each gets its own arithmetic decoder over its own byte range. That is what
//!   makes them parallelisable — and it means a tile-boundary bug shows up as a
//!   vertical seam rather than as noise.
//! * **All tiles but the last carry a 4-byte big-endian length.** The last one
//!   runs to the end of the frame. Reading a length for it consumes four bytes
//!   of coefficients.

use super::header::{read_compressed_header, FrameContext, TxMode};
use super::tile::{decode_partition, FrameDecodeState};
use super::{max_log2_tile_cols, min_log2_tile_cols, BoolDecoder, FrameHeader, NUM_REF_FRAMES};
use alloc::vec::Vec;

/// A decoded frame's planes, cropped to the visible size.
pub struct DecodedFrame {
    pub w: usize,
    pub h: usize,
    pub y: Vec<u8>,
    pub cb: Vec<u8>,
    pub cr: Vec<u8>,
}

/// A VP9 sequence decoder: the eight reference slots and the eight persistent
/// probability contexts, plus the bookkeeping that connects frames.
///
/// This is the state that makes VP9 a *sequence* rather than a series of
/// pictures, and three parts of it are load-bearing:
///
/// * **Eight reference slots, refreshed by a bitmask.** A frame names three of
///   them to predict from and may write itself into any subset. A slot is not
///   "the previous frame" — an ALTREF may sit in a slot for dozens of frames.
/// * **Eight probability contexts.** The compressed header's forward updates
///   are applied to a *copy* of context `frame_context_idx`, and only written
///   back when `refresh_frame_context` says so.
/// * **`setup_past_independence`** on every keyframe, intra-only or
///   error-resilient frame resets all eight contexts and the loop-filter
///   deltas. Skipping it makes the first frame after a mid-stream keyframe
///   decode against stale probabilities.
pub struct Vp9Decoder {
    refs: [Option<alloc::rc::Rc<super::inter::RefFrame>>; NUM_REF_FRAMES],
    contexts: [FrameContext; 4],
    ref_sizes: super::RefSizes,
    /// Whether each reference slot's picture points forward in time.
    ref_sign_bias: [bool; 4],
    /// The previously decoded picture, for the temporal MV candidate. This is
    /// **not** a reference slot: it is whatever was decoded last, regardless of
    /// which slots it refreshed.
    prev_frame: Option<alloc::rc::Rc<super::inter::RefFrame>>,
    /// `(width, height, intra_only, show_frame, key_frame)` of the last frame.
    last: Option<(u32, u32, bool, bool, bool)>,
}

impl Default for Vp9Decoder {
    fn default() -> Self {
        Vp9Decoder::new()
    }
}

impl Vp9Decoder {
    pub fn new() -> Vp9Decoder {
        Vp9Decoder {
            refs: Default::default(),
            contexts: core::array::from_fn(|_| FrameContext::default()),
            ref_sizes: [(0, 0); NUM_REF_FRAMES],
            ref_sign_bias: [false; 4],
            prev_frame: None,
            last: None,
        }
    }

    /// Decode one coded frame. Returns the picture when the frame is shown;
    /// `None` for a hidden frame (an ALTREF), which still updates the state.
    pub fn decode_frame(&mut self, data: &[u8]) -> Result<Option<DecodedFrame>, &'static str> {
        self.decode_frame_inner(data, &mut None)
    }

    /// As [`Self::decode_frame`], but also hands back the frame's mode-info
    /// grid. Bring-up only: "the modes are wrong" and "the pixels are wrong"
    /// look identical from the outside and need different fixes.
    pub fn decode_frame_debug(
        &mut self,
        data: &[u8],
        mi_out: &mut Option<(usize, usize, Vec<super::tile::ModeInfo>)>,
    ) -> Result<Option<DecodedFrame>, &'static str> {
        self.decode_frame_inner(data, mi_out)
    }

    fn decode_frame_inner(
        &mut self,
        data: &[u8],
        mi_out: &mut Option<(usize, usize, Vec<super::tile::ModeInfo>)>,
    ) -> Result<Option<DecodedFrame>, &'static str> {
        let h = super::parse_frame_header(data, &self.ref_sizes)?;

        // A `show_existing_frame` frame codes no picture at all: it re-displays
        // a slot and changes nothing else.
        if let Some(slot) = h.show_existing_frame {
            let f = self.refs[slot as usize % NUM_REF_FRAMES]
                .as_ref()
                .ok_or("vp9: show_existing_frame names an empty slot")?;
            return Ok(Some(crop_ref(f)));
        }

        if h.is_intra() || h.error_resilient {
            for c in self.contexts.iter_mut() {
                *c = FrameContext::default();
            }
        }
        // The context the frame decodes against, and a copy of it *before* the
        // header's forward updates — backward adaptation merges the counts into
        // that pre-update copy, not into the updated one.
        let pre = self.contexts[h.frame_context_idx as usize % 4].clone();
        let mut fc = pre.clone();

        // Sign bias is per *reference frame kind*, taken from this frame's
        // slot assignment.
        let mut sign_bias = [false; 4];
        for i in 0..super::REFS_PER_FRAME {
            sign_bias[i + 1] = h.ref_frame_sign_bias[i];
        }

        let mut tx_mode = super::header::TxMode::Only4x4;
        let state = if h.is_intra() {
            decode_frame_state_tx(data, &h, &mut fc, None, &mut tx_mode)?
        } else {
            let (fixed, var) = super::tile::FrameRefs::compound_refs(sign_bias);
            let bufs = core::array::from_fn(|i| {
                self.refs[h.ref_frame_idx[i] as usize % NUM_REF_FRAMES]
                    .as_deref()
            });
            // A reference of a different size would need the scaled prediction
            // path, which is not built; refusing is better than predicting from
            // the wrong pixels.
            for b in bufs.iter().flatten() {
                if b.planes[0].width != h.width as usize || b.planes[0].height != h.height as usize {
                    return Err("vp9: scaled reference frames are not implemented yet");
                }
            }
            let refs = super::tile::FrameRefs {
                bufs,
                sign_bias,
                comp_fixed_ref: fixed,
                comp_var_ref: var,
                reference_mode: super::header::ReferenceMode::Single,
                interp_filter: if h.interp_filter == super::InterpFilter::Switchable {
                    None
                } else {
                    Some(h.interp_filter as u8)
                },
                allow_high_precision_mv: h.allow_high_precision_mv,
                // `use_prev_frame_mvs`: the previous decoded frame must be a
                // shown, same-sized, non-intra, non-error-resilient picture.
                // Notably it is *false* for the frame right after a keyframe,
                // which is why a decoder without the temporal candidate gets
                // frames 0 and 1 right and fails at frame 2.
                prev_mvs: if !h.error_resilient
                    && self.last.map(|(w, ht, intra, shown, key)| {
                        w == h.width && ht == h.height && !intra && shown && !key
                    }) == Some(true)
                {
                    self.prev_frame.as_deref().map(|f| f.mvs.as_slice())
                } else {
                    None
                },
            };
            let mut refs = refs;
            decode_frame_state_tx(data, &h, &mut fc, Some(&mut refs), &mut tx_mode)?
        };

        // Backward adaptation: when a frame is neither error-resilient nor
        // frame-parallel, the probabilities the *next* frames start from are
        // this frame's symbol counts merged into `pre`. Skipping it leaves the
        // following frame decoding against the wrong probabilities — a desync
        // one frame removed from the omission.
        if !h.error_resilient && !h.frame_parallel_decoding_mode {
            if let Some(counts) = state.counts.as_deref() {
                super::header::adapt(
                    &mut fc,
                    &pre,
                    counts,
                    h.is_intra(),
                    self.last.map(|l| l.4).unwrap_or(false),
                    tx_mode,
                    h.interp_filter == super::InterpFilter::Switchable,
                    h.allow_high_precision_mv,
                );
            }
        }
        if h.refresh_frame_context {
            self.contexts[h.frame_context_idx as usize % 4] = fc;
        }

        *mi_out = Some((state.mi_cols, state.mi_rows, state.mi.clone()));
        // Publish into the refreshed slots.
        let picture = alloc::rc::Rc::new(super::inter::RefFrame {
            mvs: state
                .mi
                .iter()
                .map(|m| (m.mv, m.ref_frame))
                .collect(),
            mi_cols: state.mi_cols,
            mi_rows: state.mi_rows,
            planes: state.planes,
        });
        for slot in 0..NUM_REF_FRAMES {
            if h.refresh_frame_flags & (1 << slot) != 0 {
                self.refs[slot] = Some(picture.clone());
                self.ref_sizes[slot] = (h.width, h.height);
            }
        }
        self.ref_sign_bias = sign_bias;
        self.prev_frame = Some(picture.clone());
        self.last = Some((h.width, h.height, h.intra_only, h.show_frame, h.key_frame));
        Ok(if h.show_frame { Some(crop_ref(&picture)) } else { None })
    }
}

/// Copy a reference picture's planes out to their visible size.
fn crop_ref(f: &super::inter::RefFrame) -> DecodedFrame {
    let take = |p: &super::tile::Plane| -> Vec<u8> {
        let mut out = Vec::with_capacity(p.width * p.height);
        for y in 0..p.height {
            let row = y * p.stride;
            out.extend_from_slice(&p.data[row..row + p.width]);
        }
        out
    };
    DecodedFrame {
        w: f.planes[0].width,
        h: f.planes[0].height,
        y: take(&f.planes[0]),
        cb: take(&f.planes[1]),
        cr: take(&f.planes[2]),
    }
}

/// `get_tile_offset`: where tile `idx` starts, in 8x8 units.
fn tile_offset(idx: usize, mis: usize, log2: u32) -> usize {
    let sb_cols = (mis + 7) >> 3;
    let offset = ((idx * sb_cols) >> log2) << 3;
    offset.min(mis)
}

/// Decode one **intra** VP9 frame into planes.
///
/// `fc` is the frame context to decode against; it is updated in place by the
/// compressed header, and the caller decides whether to keep it (per
/// `refresh_frame_context`).
pub fn decode_intra_frame(
    data: &[u8],
    h: &FrameHeader,
    fc: &mut FrameContext,
) -> Result<DecodedFrame, &'static str> {
    Ok(crop(decode_intra_frame_state(data, h, fc)?, h))
}

/// As [`decode_intra_frame`] but returning the whole decode state, so a
/// bring-up harness can inspect the partition/mode grid separately from the
/// pixels. "The modes are wrong" and "the residual is wrong" produce the same
/// bad picture and need different fixes.
pub fn decode_intra_frame_state(
    data: &[u8],
    h: &FrameHeader,
    fc: &mut FrameContext,
) -> Result<FrameDecodeState, &'static str> {
    decode_frame_state(data, h, fc, None)
}

/// Decode any frame; `refs` is `None` for an intra frame.
pub fn decode_frame_state(
    data: &[u8],
    h: &FrameHeader,
    fc: &mut FrameContext,
    refs: Option<&mut super::tile::FrameRefs>,
) -> Result<FrameDecodeState, &'static str> {
    let mut tx = super::header::TxMode::Only4x4;
    decode_frame_state_tx(data, h, fc, refs, &mut tx)
}

/// As [`decode_frame_state`], also reporting the transform mode the compressed
/// header chose — backward adaptation needs it to know whether the per-block
/// transform-size probabilities were used at all.
pub fn decode_frame_state_tx(
    data: &[u8],
    h: &FrameHeader,
    fc: &mut FrameContext,
    refs: Option<&mut super::tile::FrameRefs>,
    tx_mode_out: &mut super::header::TxMode,
) -> Result<FrameDecodeState, &'static str> {
    if !h.is_intra() && refs.is_none() {
        return Err("vp9: an inter frame needs its reference pictures");
    }
    if h.profile != 0 || h.bit_depth != 8 || !h.subsampling_x || !h.subsampling_y {
        return Err("vp9: only profile 0 (8-bit 4:2:0) is implemented");
    }
    if h.segmentation.enabled {
        // Segmentation changes the quantiser and loop filter per block and adds
        // a coded map; decoding without it would apply the wrong quantiser to
        // most of the frame.
        return Err("vp9: segmentation is not implemented yet");
    }

    let comp = read_compressed_header(
        data.get(h.uncompressed_header_bytes..h.tile_data_offset())
            .ok_or("vp9: compressed header out of range")?,
        fc,
        h.quant.lossless(),
        h.is_intra(),
        h.interp_filter == super::InterpFilter::Switchable,
        h.allow_high_precision_mv,
        // `vp9_compound_reference_allowed`: some reference must point the other
        // way in time, or the reference mode is not coded at all.
        refs.as_ref()
            .map(|r| r.sign_bias[2] != r.sign_bias[1] || r.sign_bias[3] != r.sign_bias[1])
            .unwrap_or(false),
    )?;
    // The reference mode is decoded *by* the compressed header, and the tile
    // data then needs it — so it is written back here rather than guessed
    // before the header is read.
    *tx_mode_out = comp.tx_mode;
    let mut refs = refs;
    if let Some(fr) = refs.as_mut() {
        fr.reference_mode = comp.reference_mode;
    }
    let refs = refs.map(|r| &*r);

    let mut s = FrameDecodeState::new(
        h.width as usize,
        h.height as usize,
        h.quant.base_q_idx as i32,
        h.quant.delta_q_y_dc,
        h.quant.delta_q_uv_dc,
        h.quant.delta_q_uv_ac,
        h.quant.lossless(),
        comp.tx_mode,
        h.subsampling_x,
        h.subsampling_y,
        h.is_intra(),
    );

    // The tile layout must agree with what the uncompressed header committed
    // to; a mismatch means the header was mis-parsed and the byte ranges below
    // would be nonsense.
    let sb64_cols = h.sb64_cols();
    if h.tiles.cols_log2 < min_log2_tile_cols(sb64_cols)
        || h.tiles.cols_log2 > max_log2_tile_cols(sb64_cols)
    {
        return Err("vp9: tile column count outside the legal range for this width");
    }
    let tile_cols = 1usize << h.tiles.cols_log2;
    let tile_rows = 1usize << h.tiles.rows_log2;

    let mut counts = super::header::FrameCounts::default();
    let mut p = h.tile_data_offset();
    for tr in 0..tile_rows {
        for tc in 0..tile_cols {
            let last = tr == tile_rows - 1 && tc == tile_cols - 1;
            let size = if last {
                data.len().saturating_sub(p)
            } else {
                if p + 4 > data.len() {
                    return Err("vp9: truncated tile size");
                }
                let n = u32::from_be_bytes([data[p], data[p + 1], data[p + 2], data[p + 3]]) as usize;
                p += 4;
                n
            };
            if p + size > data.len() {
                return Err("vp9: tile overruns the frame");
            }
            let tile = &data[p..p + size];
            p += size;

            let mi_row_start = tile_offset(tr, s.mi_rows, h.tiles.rows_log2);
            let mi_row_end = tile_offset(tr + 1, s.mi_rows, h.tiles.rows_log2);
            let mi_col_start = tile_offset(tc, s.mi_cols, h.tiles.cols_log2);
            let mi_col_end = tile_offset(tc + 1, s.mi_cols, h.tiles.cols_log2);

            let mut r = BoolDecoder::new(tile)?;
            s.start_tile(mi_col_start, mi_col_end);
            let mut mi_row = mi_row_start;
            while mi_row < mi_row_end {
                s.start_sb_row();
                let mut mi_col = mi_col_start;
                while mi_col < mi_col_end {
                    // BLOCK_64X64 is size index 12; its width is 16 4x4 units,
                    // so `n4x4_l2` is 4.
                    decode_partition(&mut r, &mut s, fc, &mut counts, refs, mi_row, mi_col, 12, 4);
                    mi_col += 8;
                }
                mi_row += 8;
            }
            if r.exhausted() {
                return Err("vp9: tile data ran past its partition");
            }
        }
    }

    // Deblock in place: later frames predict from the *filtered* picture, so
    // this is part of decoding, not of presentation.
    let lvl_table = h.loop_filter.level_table();
    super::loopfilter::loop_filter_frame(
        &mut s,
        h.loop_filter.level,
        h.loop_filter.intra_level(),
        h.loop_filter.sharpness,
        if h.is_intra() { None } else { Some(&lvl_table) },
    );
    s.counts = Some(alloc::boxed::Box::new(counts));
    Ok(s)
}

/// Copy the superblock-aligned planes down to the visible frame size.
fn crop(s: FrameDecodeState, h: &FrameHeader) -> DecodedFrame {
    let take = |p: &super::tile::Plane| -> Vec<u8> {
        let mut out = Vec::with_capacity(p.width * p.height);
        for y in 0..p.height {
            let row = y * p.stride;
            out.extend_from_slice(&p.data[row..row + p.width]);
        }
        out
    };
    DecodedFrame {
        w: h.width as usize,
        h: h.height as usize,
        y: take(&s.planes[0]),
        cb: take(&s.planes[1]),
        cr: take(&s.planes[2]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn tile_offsets_partition_the_frame_exactly() {
        // Every tile boundary must be superblock-aligned, tiles must not
        // overlap, and the last must reach the end — the property that makes a
        // tile-boundary bug a seam rather than silent overlap.
        for &mis in &[1usize, 22, 80, 160, 256] {
            for log2 in 0..3u32 {
                let n = 1usize << log2;
                let mut prev = 0;
                for i in 0..n {
                    let a = tile_offset(i, mis, log2);
                    let b = tile_offset(i + 1, mis, log2);
                    assert_eq!(a, prev, "tile {} starts where {} ended", i, i.wrapping_sub(1));
                    assert!(b >= a, "tile {} runs backwards", i);
                    assert!(a % 8 == 0 || a == mis, "tile start is not superblock-aligned");
                    prev = b;
                }
                assert_eq!(prev, mis, "tiles must cover the frame");
            }
        }
    }

    #[test_case]
    fn an_inter_frame_is_refused_not_decoded_as_intra() {
        // Decoding an inter frame with the intra path produces a full frame of
        // garbage, which reads as a broken decoder rather than a missing
        // feature — so it must be an error.
        let refs: super::super::RefSizes = [(0, 0); super::super::NUM_REF_FRAMES];
        let bits = [0x86u8, 0x00, 0x00, 0x02];
        let mut refs2 = refs;
        refs2[0] = (64, 64);
        if let Ok(h) = super::super::parse_frame_header(&bits, &refs2) {
            let mut fc = FrameContext::default();
            // `unwrap_err` would need `DecodedFrame: Debug`; match instead so
            // the frame type stays free of derives it does not otherwise want
            // (it holds three whole planes).
            match decode_intra_frame(&bits, &h, &mut fc) {
                Err(e) => assert!(e.contains("inter"), "unexpected error: {}", e),
                Ok(_) => panic!("an inter frame must not decode on the intra path"),
            }
        }
    }
}
