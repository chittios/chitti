//! The HEVC picture decoder: the CTU quadtree walk, reconstruction, and the
//! in-loop filters.
//!
//! This is the glue that turns the pure stages next door into a frame. It owns
//! the things a *picture* has and a block does not:
//!
//! - **Neighbour grids.** Motion, intra modes, skip flags, coding depth, QP and
//!   coefficient-presence, each at the granularity the specification indexes
//!   them by. Almost every context derivation upstream reads one of these, so
//!   their update points are as load-bearing as their contents.
//! - **Availability.** A neighbour exists only if it is inside the picture, in
//!   the same slice, and **already decoded** — which for a quadtree is not
//!   "above or left" but a z-scan order comparison. Getting this wrong reads
//!   uninitialised motion, which decodes to a plausible picture that drifts.
//! - **The two-pass in-loop filter.** Deblocking runs over the whole picture
//!   after every CTU is reconstructed, then SAO runs over the deblocked result
//!   from a *copy* — an in-place SAO would classify each sample against its
//!   already-offset neighbour.
//!
//! Scope: 4:2:0, 8/10/12-bit, tiles and PCM supported. Range extensions
//! (4:2:2/4:4:4) are refused by name rather than mis-decoded.

use alloc::rc::Rc;
use alloc::vec;
use alloc::vec::Vec;

use super::super::h264::cabac::Cabac;
use super::ctu::{PartMode, EdgeSide};
use super::inter::{AmvpNeighbour, MergeNeighbours, MvField, Plane, MAX_PB};
use super::syntax::{self as syn, Bin, BinSource};
use super::{
    cabac_tables as ct, ctu, dpb, inter, intra, residual, sao, tiles, transform, NalType, Pps,
    SliceHeader, SliceType, Sps,
};

/// A decoded picture, in the shape the player consumes.
pub struct DecodedFrame {
    pub w: usize,
    pub h: usize,
    /// Bit depth of the sample arrays (8, 10 or 12).
    pub bit_depth: u32,
    pub y: Vec<u16>,
    pub cb: Vec<u16>,
    pub cr: Vec<u16>,
}

/// A reconstructed reference picture.
pub struct RefFrame {
    pub poc: i32,
    /// The motion field, kept for the temporal merge candidate. HEVC stores it
    /// at **16x16** granularity (`x &= ~15`), which is a real compression of
    /// the field, not an implementation shortcut — reading it at 4x4 gives a
    /// different candidate.
    pub mvf: Vec<PuInfo>,
    pub mvf_w: usize,
    pub w: usize,
    pub h: usize,
    pub cw: usize,
    pub chh: usize,
    pub bit_depth: u32,
    pub y: Vec<u16>,
    pub cb: Vec<u16>,
    pub cr: Vec<u16>,
}

/// Per-4x4 motion and prediction record — what every neighbour derivation
/// upstream reads.
#[derive(Clone, Copy, Default)]
pub struct PuInfo {
    intra: bool,
    mvf: MvField,
    /// Resolved POCs, so a comparison never has to go back through a list.
    ref_poc: [i32; 2],
    ref_lt: [bool; 2],
}

/// Edge-boundary bits on the 8x8 deblocking grid.
const EDGE_V_TU: u8 = 1;
const EDGE_V_PU: u8 = 2;
const EDGE_H_TU: u8 = 4;
const EDGE_H_PU: u8 = 8;

/// SAO parameters for one CTB.
#[derive(Clone, Copy, Default)]
struct SaoCtb {
    /// 0 = off, 1 = band, 2 = edge.
    type_idx: [u8; 3],
    eo_class: [u8; 3],
    band_position: [u8; 3],
    offsets: [sao::SaoOffsets; 3],
}

/// Everything that lives for one picture.
struct Pic {
    poc: i32,
    w: usize,
    h: usize,
    cw: usize,
    ch: usize,
    y: Vec<u16>,
    cb: Vec<u16>,
    cr: Vec<u16>,

    // --- neighbour grids ---
    /// Per min-PU (4x4).
    pu: Vec<PuInfo>,
    /// Per min-PU luma intra mode, kept beside `pu` because the MPM derivation
    /// reads a neighbour's mode even when the block itself is inter.
    ipm: Vec<u8>,
    pu_w: usize,
    /// Per min-TB (4x4): does the containing transform block carry luma
    /// coefficients? Read only by boundary-strength derivation.
    cbf: Vec<bool>,
    /// Per min-CB.
    skip: Vec<bool>,
    ct_depth: Vec<u8>,
    cb_w: usize,
    /// Per min-TB: luma QP, for the deblocking filter's per-edge average.
    qp_y: Vec<i8>,
    /// Per min-TB: the z-scan address, so availability is one comparison.
    zscan: Vec<u32>,
    tb_w: usize,
    tb_h: usize,
    /// Per CTB: which slice it belongs to, and its SAO parameters.
    slice_idx: Vec<i32>,
    sao_par: Vec<SaoCtb>,
    ctb_w: usize,
    ctb_h: usize,
    /// Per 8x8: transform / prediction boundary bits.
    edges: Vec<u8>,
    edge_w: usize,
    /// Set where deblocking must not run (lossless or PCM blocks).
    no_deblock: Vec<bool>,
    /// Per-CTB deblocking parameters, because they are signalled per *slice*
    /// and the filter runs over the whole picture afterwards.
    db_beta: Vec<i8>,
    db_tc: Vec<i8>,
    db_off: Vec<bool>,
    log2_min_tb: u32,
    log2_ctb: u32,
    bit_depth_luma: u32,
    bit_depth_chroma: u32,
}

impl Pic {
    #[inline]
    fn pu_at(&self, x: usize, y: usize) -> PuInfo {
        self.pu[(y >> 2) * self.pu_w + (x >> 2)]
    }
}

/// The sequence decoder.
pub struct HevcDecoder {
    spss: alloc::collections::BTreeMap<u32, Sps>,
    ppss: alloc::collections::BTreeMap<u32, Pps>,
    dpb: Vec<(dpb::FrameFlags, Rc<RefFrame>)>,
    poc_tid0: i32,
    pic: Option<Pic>,
    /// Pictures finished and awaiting output, lowest POC first.
    ready: alloc::collections::BTreeMap<i32, Rc<RefFrame>>,
    /// Bring-up trace: `(x, y, log2_size, c_idx, mode, cbf)` per transform
    /// unit, and the CTU count each slice consumed. Empty unless `trace` is on.
    pub trace_on: bool,
    pub trace: Vec<(u16, u16, u8, u8, u8, u8)>,
    pub ctus: usize,
}

impl Default for HevcDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl HevcDecoder {
    pub fn new() -> HevcDecoder {
        HevcDecoder {
            spss: alloc::collections::BTreeMap::new(),
            ppss: alloc::collections::BTreeMap::new(),
            dpb: Vec::new(),
            poc_tid0: 0,
            pic: None,
            ready: alloc::collections::BTreeMap::new(),
            trace_on: false,
            trace: Vec::new(),
            ctus: 0,
        }
    }

    /// Load the parameter sets from an `hvcC` box, before any sample.
    ///
    /// The NAL payloads there carry their two-byte header, which the RBSP
    /// unescape must skip — feeding it in makes the SPS's first field the
    /// header's own bits and yields a plausible, wrong geometry.
    pub fn set_parameter_sets(
        &mut self,
        _vps: &[Vec<u8>],
        spss: &[Vec<u8>],
        ppss: &[Vec<u8>],
    ) -> Result<(), &'static str> {
        for nal in spss {
            let rbsp = super::super::bits::unescape_rbsp(nal.get(2..).unwrap_or(&[]));
            let s = super::parse_sps(&rbsp)?;
            self.spss.insert(s.id, s);
        }
        for nal in ppss {
            let rbsp = super::super::bits::unescape_rbsp(nal.get(2..).unwrap_or(&[]));
            let p = super::parse_pps(&rbsp)?;
            self.ppss.insert(p.id, p);
        }
        Ok(())
    }

    /// Feed one access unit's NAL units (already length-prefix-split).
    ///
    /// Returns the pictures that became displayable, lowest POC first.
    pub fn decode_au(&mut self, nals: &[super::Nal]) -> Result<Vec<DecodedFrame>, &'static str> {
        for n in nals {
            let rbsp = n.rbsp();
            match n.kind {
                NalType::Sps => {
                    let s = super::parse_sps(&rbsp)?;
                    self.spss.insert(s.id, s);
                }
                NalType::Pps => {
                    let p = super::parse_pps(&rbsp)?;
                    self.ppss.insert(p.id, p);
                }
                k if k.is_slice() => self.decode_slice(k, &rbsp)?,
                _ => {}
            }
        }
        self.finish_picture();
        // **Release in display order, not decode order.** A picture is held
        // until more than `sps_max_num_reorder_pics` are pending, and then the
        // *lowest POC* leaves — which is the whole of C.5.2.2 bumping and the
        // only thing that turns a B-pyramid's decode order back into display
        // order. Draining eagerly instead emits decode order, which is right
        // for the first picture of every stream and wrong for the rest.
        let reorder = self
            .spss
            .values()
            .next()
            .map(|s| s.max_num_reorder_pics as usize)
            .unwrap_or(0);
        let mut out = Vec::new();
        while self.ready.len() > reorder {
            let k = *self.ready.keys().next().unwrap();
            let f = self.ready.remove(&k).unwrap();
            out.push(DecodedFrame {
                w: f.w,
                h: f.h,
                bit_depth: f.bit_depth,
                y: f.y.clone(),
                cb: f.cb.clone(),
                cr: f.cr.clone(),
            });
        }
        Ok(out)
    }

    /// Flush the buffer at end of stream.
    pub fn flush(&mut self) -> Vec<DecodedFrame> {
        self.finish_picture();
        let mut out = Vec::new();
        let keys: Vec<i32> = self.ready.keys().copied().collect();
        for k in keys {
            let f = self.ready.remove(&k).unwrap();
            out.push(DecodedFrame {
                w: f.w,
                h: f.h,
                bit_depth: f.bit_depth,
                y: f.y.clone(),
                cb: f.cb.clone(),
                cr: f.cr.clone(),
            });
        }
        out
    }

    fn finish_picture(&mut self) {
        let Some(mut p) = self.pic.take() else { return };
        // The in-loop filters run over the whole picture, after every CTU is
        // reconstructed — deblocking first, then SAO over a *copy* of the
        // deblocked result.
        deblock_picture(&mut p);
        sao_picture(&mut p);
        // Sub-sample the motion field to 16x16 as the specification stores it.
        let mw = p.w.div_ceil(16);
        let mh = p.h.div_ceil(16);
        let mut mvf = vec![PuInfo::default(); mw * mh];
        for j in 0..mh {
            for i in 0..mw {
                mvf[j * mw + i] = p.pu_at((i * 16).min(p.w - 1), (j * 16).min(p.h - 1));
            }
        }
        let f = Rc::new(RefFrame {
            poc: p.poc,
            mvf,
            mvf_w: mw,
            w: p.w,
            h: p.h,
            cw: p.cw,
            chh: p.ch,
            bit_depth: p.bit_depth_luma,
            y: p.y,
            cb: p.cb,
            cr: p.cr,
        });
        self.dpb.push((
            dpb::FrameFlags { short_ref: true, long_ref: false, output: false },
            f.clone(),
        ));
        // Bound the buffer: a reference that no later RPS names has already
        // been unmarked, so anything not marked is free.
        self.dpb.retain(|(fl, _)| fl.short_ref || fl.long_ref);
        while self.dpb.len() > 16 {
            self.dpb.remove(0);
        }
        self.ready.insert(p.poc, f);
    }

    fn decode_slice(&mut self, nal: NalType, rbsp: &[u8]) -> Result<(), &'static str> {
        // The slice header names its PPS, which names its SPS.
        // Peek the PPS id: first_slice flag (+ no_output for IRAP), then ue().
        let pps_id = super::peek_slice_pps_id(rbsp, nal)?;
        let pps = self.ppss.get(&pps_id).ok_or("hevc: slice names an unknown PPS")?.clone();
        let sps = self.spss.get(&pps.sps_id).ok_or("hevc: PPS names an unknown SPS")?.clone();
        let sh = super::parse_slice_header(rbsp, nal, &sps, &pps)?;

        if sps.chroma_format_idc != 1 {
            return Err("hevc: only 4:2:0 is supported");
        }
        // 8/10/12-bit Main family; tiles and PCM handled in the walk.

        if sh.first_slice_in_pic {
            self.finish_picture();
            let poc = if nal.is_idr() {
                0
            } else {
                dpb::compute_poc(sps.log2_max_poc_lsb, self.poc_tid0, sh.pic_order_cnt_lsb as i32, nal)
            };
            self.poc_tid0 = poc;
            self.pic = Some(Pic::new(&sps, poc));
            // Apply the RPS: unmark everything this picture does not reference.
            let sets = dpb::derive_rps(poc, Some(&sh.st_rps), &dpb::LongTermRps::default());
            for (fl, f) in self.dpb.iter_mut() {
                let named = sets.st_curr_before.contains(&f.poc)
                    || sets.st_curr_after.contains(&f.poc)
                    || sets.st_foll.contains(&f.poc);
                fl.short_ref = named;
            }
            // IDR and BLA always empty the DPB (NoRaslOutputFlag = 1). CRA does
            // *not* when it is mid-stream: its RASL leading pictures predict from
            // the previous GOP, so wiping the buffer here turns a continuous
            // keyint into three broken frames before every CRA. At true random
            // access the DPB is already empty, so treating CRA like a normal
            // picture is also correct there (missing refs fall back below).
            if nal.is_idr() || nal.is_bla() {
                for (fl, _) in self.dpb.iter_mut() {
                    fl.short_ref = false;
                    fl.long_ref = false;
                }
            }
            self.dpb.retain(|(fl, _)| fl.short_ref || fl.long_ref);
        }
        let pic_poc = self.pic.as_ref().ok_or("hevc: slice before the first picture")?.poc;

        // Reference lists, as POCs, resolved to frames.
        let sets = dpb::derive_rps(pic_poc, Some(&sh.st_rps), &dpb::LongTermRps::default());
        let mut lists: [Vec<Rc<RefFrame>>; 2] = [Vec::new(), Vec::new()];
        let mut list_poc: [Vec<i32>; 2] = [Vec::new(), Vec::new()];
        if sh.slice_type != SliceType::I {
            for li in 0..2usize {
                let n = if li == 0 { sh.num_ref_idx_l0 } else { sh.num_ref_idx_l1 } as usize;
                if li == 1 && sh.slice_type != SliceType::B {
                    continue;
                }
                let modif = if li == 0 {
                    if sh.list_entry_l0.is_empty() {
                        None
                    } else {
                        Some(sh.list_entry_l0.as_slice())
                    }
                } else if sh.list_entry_l1.is_empty() {
                    None
                } else {
                    Some(sh.list_entry_l1.as_slice())
                };
                for e in dpb::build_ref_list(&sets, li, n, modif) {
                    // A missing reference is replaced by the nearest available
                    // picture rather than failing: a stream that starts on a
                    // CRA legitimately references pictures that were never sent.
                    let f = self
                        .dpb
                        .iter()
                        .find(|(_, f)| f.poc == e.poc)
                        .or_else(|| {
                            self.dpb.iter().min_by_key(|(_, f)| (f.poc - e.poc).abs())
                        })
                        .map(|(_, f)| f.clone());
                    if let Some(f) = f {
                        list_poc[li].push(f.poc);
                        lists[li].push(f);
                    }
                }
                if lists[li].is_empty() && !self.dpb.is_empty() {
                    let f = self.dpb[self.dpb.len() - 1].1.clone();
                    list_poc[li].push(f.poc);
                    lists[li].push(f);
                }
                if lists[li].is_empty() {
                    return Err("hevc: an inter slice with no reference pictures");
                }
            }
        }

        // The collocated picture: `collocated_from_l0` picks the list, and the
        // index is a slice-header field. Getting the *list* wrong silently
        // reads a different picture's motion, which is a plausible candidate
        // and therefore a drift rather than a failure.
        let col = if sh.temporal_mvp_enabled && sh.slice_type != SliceType::I {
            let li = if sh.collocated_from_l0 { 0usize } else { 1 };
            lists[li].get(sh.collocated_ref_idx as usize).cloned()
        } else {
            None
        };

        let data = rbsp.get(sh.data_byte_offset..).ok_or("hevc: slice data past the end")?;
        let init_type = super::cabac_init_type(sh.slice_type, sh.cabac_init);
        let qp = sh.qp.clamp(0, 51);
        let cabac = Cabac::new_hevc(data, qp, init_type, &ct::INIT_VALUES)?;

        let mut sd = SliceDecoder {
            sps: &sps,
            pps: &pps,
            sh: &sh,
            c: cabac,
            pic: self.pic.as_mut().unwrap(),
            lists: &lists,
            list_poc: &list_poc,
            col,
            qp_y: qp,
            qp_pred: qp,
            qp_y_prev: qp,
            stat_coeff: [0; 4],
            cu: CuState::default(),
            slice_index: sh.segment_address as i32,
            ctb_addr: sh.segment_address as usize,
            data,
            qp_init: qp,
            init_type,
            trace_on: self.trace_on,
            trace: Vec::new(),
            ctus: 0,
            tile_map: None,
        };
        let dlen = data.len();
        let r = sd.run();
        if self.trace_on {
            self.trace.push((
                0xFFFD,
                sd.c.byte_pos() as u16,
                (dlen & 0xff) as u8,
                (dlen >> 8) as u8,
                sh.slice_type as u8,
                0,
            ));
        }
        let (t, n) = (core::mem::take(&mut sd.trace), sd.ctus);
        self.trace.extend(t);
        self.ctus += n;
        r
    }
}

impl Pic {
    fn new(sps: &Sps, poc: i32) -> Pic {
        let w = sps.pic_width_in_luma_samples as usize;
        let h = sps.pic_height_in_luma_samples as usize;
        let cw = w / 2;
        let ch = h / 2;
        let log2_ctb = sps.log2_ctb_size as usize;
        let ctb = 1usize << log2_ctb;
        let ctb_w = w.div_ceil(ctb);
        let ctb_h = h.div_ceil(ctb);
        let log2_min_tb = sps.log2_min_tb_size as usize;
        let tb_w = w.div_ceil(1 << log2_min_tb);
        let tb_h = h.div_ceil(1 << log2_min_tb);
        let pu_w = w.div_ceil(4);
        let pu_h = h.div_ceil(4);
        let cb_w = w.div_ceil(1 << sps.log2_min_cb_size);
        let cb_h = h.div_ceil(1 << sps.log2_min_cb_size);
        let edge_w = w.div_ceil(8);
        let edge_h = h.div_ceil(8);

        // The z-scan address table (§6.5.2): CTB raster order, then Morton
        // order inside a CTB. Availability is a single comparison against it,
        // which is the only way to answer "has this block been decoded" for a
        // quadtree — "above or left" is true of blocks that come *later*.
        let ld = log2_ctb - log2_min_tb;
        let mut zscan = vec![0u32; tb_w * tb_h];
        for y in 0..tb_h {
            for x in 0..tb_w {
                let mut val = (((y >> ld) * ctb_w + (x >> ld)) as u32) << (ld * 2);
                for i in 0..ld {
                    let m = 1usize << i;
                    val += ((x & m != 0) as u32) * (m * m) as u32
                        + ((y & m != 0) as u32) * (2 * m * m) as u32;
                }
                zscan[y * tb_w + x] = val;
            }
        }

        Pic {
            poc,
            w,
            h,
            cw,
            ch,
            y: vec![0u16; w * h],
            cb: vec![1u16 << (sps.bit_depth_chroma - 1); cw * ch],
            cr: vec![1u16 << (sps.bit_depth_chroma - 1); cw * ch],
            pu: vec![PuInfo::default(); pu_w * pu_h],
            ipm: vec![intra::DC; pu_w * pu_h],
            pu_w,
            cbf: vec![false; tb_w * tb_h],
            skip: vec![false; cb_w * cb_h],
            ct_depth: vec![0u8; cb_w * cb_h],
            cb_w,
            qp_y: vec![0i8; tb_w * tb_h],
            zscan,
            tb_w,
            tb_h,
            slice_idx: vec![-1; ctb_w * ctb_h],
            sao_par: vec![SaoCtb::default(); ctb_w * ctb_h],
            ctb_w,
            ctb_h,
            edges: vec![0u8; edge_w * edge_h],
            edge_w,
            no_deblock: vec![false; edge_w * edge_h],
            db_beta: vec![0i8; ctb_w * ctb_h],
            db_tc: vec![0i8; ctb_w * ctb_h],
            db_off: vec![false; ctb_w * ctb_h],
            log2_min_tb: sps.log2_min_tb_size,
            log2_ctb: sps.log2_ctb_size,
            bit_depth_luma: sps.bit_depth_luma,
            bit_depth_chroma: sps.bit_depth_chroma,
        }
    }
}

/// State that lives for one coding unit.
#[derive(Clone, Copy, Default)]
struct CuState {
    x: usize,
    y: usize,
    log2_size: u32,
    intra: bool,
    skip: bool,
    part_mode_2nx2n: bool,
    part_mode: Option<PartMode>,
    transquant_bypass: bool,
    intra_split: bool,
    max_trafo_depth: u32,
    ct_depth: u8,
    /// The four luma intra modes (one per NxN quadrant; index 0 otherwise).
    ipm: [u8; 4],
    ipm_c: [u8; 4],
    merge_2nx2n: bool,
    qp_delta_coded: bool,
    qp_delta: i32,
}

struct SliceDecoder<'a> {
    sps: &'a Sps,
    pps: &'a Pps,
    sh: &'a SliceHeader,
    c: Cabac<'a>,
    pic: &'a mut Pic,
    lists: &'a [Vec<Rc<RefFrame>>; 2],
    list_poc: &'a [Vec<i32>; 2],
    /// The picture the temporal merge candidate reads its motion from.
    col: Option<Rc<RefFrame>>,
    qp_y: i32,
    qp_pred: i32,
    /// The QP of the last coding unit in decoding order — `qPY_PREV`.
    qp_y_prev: i32,
    stat_coeff: [u8; 4],
    cu: CuState,
    slice_index: i32,
    ctb_addr: usize,
    /// The slice's CABAC data, kept so a WPP substream can re-enter it.
    data: &'a [u8],
    qp_init: i32,
    init_type: usize,
    trace_on: bool,
    trace: Vec<(u16, u16, u8, u8, u8, u8)>,
    ctus: usize,
    tile_map: Option<tiles::TileMap>,
}

impl<'a> SliceDecoder<'a> {
    #[inline]
    fn bd_luma(&self) -> u32 { self.sps.bit_depth_luma }
    #[inline]
    fn bd_chroma(&self) -> u32 { self.sps.bit_depth_chroma }
    #[inline]
    fn bd_for(&self, c_idx: usize) -> u32 {
        if c_idx == 0 { self.bd_luma() } else { self.bd_chroma() }
    }
    #[inline]
    fn sample_max(&self, c_idx: usize) -> i32 {
        (1i32 << self.bd_for(c_idx)) - 1
    }

    fn run(&mut self) -> Result<(), &'static str> {
        let log2_ctb = self.sps.log2_ctb_size;
        let ctb = 1usize << log2_ctb;
        let total = self.pic.ctb_w * self.pic.ctb_h;
        let ctb_w = self.pic.ctb_w;
        let ctb_h = self.pic.ctb_h;

        // Tile map: single-tile is the identity (rs == ts). Multi-tile walks
        // CTBs in tile-scan order and restarts CABAC at every tile boundary.
        let tile_map = if self.pps.tiles_enabled {
            tiles::TileMap::from_pps(
                ctb_w,
                ctb_h,
                self.pps.num_tile_columns as usize,
                self.pps.num_tile_rows as usize,
                self.pps.uniform_spacing,
                &self.pps.column_width,
                &self.pps.row_height,
            )?
        } else {
            tiles::TileMap::single(ctb_w, ctb_h)
        };
        self.tile_map = Some(tile_map.clone());

        // WPP: each CTB row is a substream that inherits contexts saved after
        // the second CTB of the row above (x265 default for multi-row pictures).
        let wpp = self.pps.entropy_coding_sync_enabled && !self.pps.tiles_enabled;
        let tiles_on = self.pps.tiles_enabled;
        let save_col = if ctb_w > 1 { 1usize } else { 0 };
        let mut saved: Option<([u8; 1024], [u8; 4])> = None;
        let mut starts: Vec<usize> = alloc::vec![0usize];
        for &o in self.sh.entry_point_offsets.iter() {
            let last = *starts.last().unwrap();
            starts.push(last + o as usize);
        }
        let mut entry_i = 0usize;

        let start_rs = self.sh.segment_address as usize;
        let mut ts = tile_map.rs_to_ts.get(start_rs).copied().unwrap_or(start_rs);

        loop {
            if ts >= total {
                break;
            }
            let rs = tile_map.ts_to_rs[ts];
            let rx = rs % ctb_w;
            let ry = rs / ctb_w;
            self.ctb_addr = rs;

            let is_first = ts == tile_map.rs_to_ts[start_rs];
            let new_tile = tiles_on
                && !is_first
                && tile_map.tile_id[ts] != tile_map.tile_id[ts - 1];
            let new_wpp_row = wpp && rx == 0 && !is_first;

            if new_tile || new_wpp_row {
                entry_i += 1;
                let off = starts.get(entry_i).copied().unwrap_or(0);
                let Some(rest) = self.data.get(off..) else {
                    break;
                };
                let mut nc =
                    Cabac::new_hevc(rest, self.qp_init, self.init_type, &ct::INIT_VALUES)?;
                if new_wpp_row {
                    // WPP inherits the previous row's context snapshot.
                    if let Some((ctx, st)) = saved {
                        nc.ctx = ctx;
                        self.stat_coeff = st;
                    }
                } else {
                    // Tile boundary: contexts re-initialised (already done by
                    // new_hevc) and rice stats reset.
                    self.stat_coeff = [0; 4];
                }
                self.c = nc;
                self.qp_y = self.qp_init;
                self.qp_y_prev = self.qp_init;
            }

            self.pic.slice_idx[rs] = self.slice_index;
            self.pic.db_beta[rs] = self.sh.beta_offset_div2 as i8;
            self.pic.db_tc[rs] = self.sh.tc_offset_div2 as i8;
            self.pic.db_off[rs] = self.sh.deblocking_filter_disabled;
            self.qp_pred = self.qp_y;

            self.sao_param(rx, ry);
            self.coding_quadtree(rx * ctb, ry * ctb, log2_ctb, 0)?;

            if wpp && rx == save_col {
                saved = Some((self.c.ctx, self.stat_coeff));
            }
            ts += 1;
            self.ctus += 1;
            if self.c.terminate() != 0 {
                break;
            }
        }
        Ok(())
    }

    // -- availability ------------------------------------------------------

    /// Is `(xn, yn)` decoded, inside the picture, and in this slice?
    fn available(&self, xc: usize, yc: usize, xn: isize, yn: isize) -> bool {
        if xn < 0 || yn < 0 {
            return false;
        }
        let (xn, yn) = (xn as usize, yn as usize);
        if xn >= self.pic.w || yn >= self.pic.h {
            return false;
        }
        let sh = self.sps.log2_min_tb_size as usize;
        let zc = self.pic.zscan[(yc >> sh) * self.pic.tb_w + (xc >> sh)];
        let zn = self.pic.zscan[(yn >> sh) * self.pic.tb_w + (xn >> sh)];
        if zn >= zc {
            return false;
        }
        let lc = self.sps.log2_ctb_size as usize;
        let cn = (yn >> lc) * self.pic.ctb_w + (xn >> lc);
        if self.pic.slice_idx[cn] != self.slice_index {
            return false;
        }
        // Neighbours in a different tile are unavailable (H.265 §6.4.1).
        if let Some(ref tm) = self.tile_map {
            if tm.tile_id_rs(cn) != tm.tile_id_rs(self.ctb_addr) {
                return false;
            }
        }
        true
    }

    /// The same, additionally requiring the neighbour to be inter-coded — what
    /// merge and AMVP need.
    fn avail_inter(&self, xc: usize, yc: usize, xn: isize, yn: isize) -> Option<PuInfo> {
        if !self.available(xc, yc, xn, yn) {
            return None;
        }
        let p = self.pic.pu_at(xn as usize, yn as usize);
        if p.intra {
            None
        } else {
            Some(p)
        }
    }

    // -- SAO ---------------------------------------------------------------

    fn sao_param(&mut self, rx: usize, ry: usize) {
        if !self.sh.sao_luma && !self.sh.sao_chroma {
            return;
        }
        let idx = ry * self.pic.ctb_w + rx;
        let mut merge_left = false;
        let mut merge_up = false;
        if rx > 0 && self.pic.slice_idx[idx - 1] == self.slice_index {
            merge_left = self.c.decision(ct::SAO_MERGE_FLAG) != 0;
        }
        if !merge_left && ry > 0 && self.pic.slice_idx[idx - self.pic.ctb_w] == self.slice_index {
            merge_up = self.c.decision(ct::SAO_MERGE_FLAG) != 0;
        }
        if merge_left {
            self.pic.sao_par[idx] = self.pic.sao_par[idx - 1];
            return;
        }
        if merge_up {
            self.pic.sao_par[idx] = self.pic.sao_par[idx - self.pic.ctb_w];
            return;
        }
        let mut s = SaoCtb::default();
        for c_idx in 0..3usize {
            let on = if c_idx == 0 { self.sh.sao_luma } else { self.sh.sao_chroma };
            if !on {
                continue;
            }
            if c_idx == 2 {
                // Cr shares Cb's type and edge class but has its own offsets.
                s.type_idx[2] = s.type_idx[1];
                s.eo_class[2] = s.eo_class[1];
            } else if self.c.decision(ct::SAO_TYPE_IDX) == 0 {
                s.type_idx[c_idx] = 0;
            } else if self.c.bypass() == 0 {
                s.type_idx[c_idx] = 1; // band
            } else {
                s.type_idx[c_idx] = 2; // edge
            }
            if s.type_idx[c_idx] == 0 {
                continue;
            }
            let cmax = (1u32 << (self.sps.bit_depth_luma.min(10) - 5)) - 1;
            let mut abs = [0u32; 4];
            for a in abs.iter_mut() {
                let mut i = 0u32;
                while i < cmax && self.c.bypass() != 0 {
                    i += 1;
                }
                *a = i;
            }
            let mut vals: sao::SaoOffsets = [0; 5];
            if s.type_idx[c_idx] == 1 {
                let mut sign = [false; 4];
                for i in 0..4 {
                    if abs[i] != 0 {
                        sign[i] = self.c.bypass() != 0;
                    }
                }
                let mut bp = self.c.bypass();
                for _ in 0..4 {
                    bp = (bp << 1) | self.c.bypass();
                }
                s.band_position[c_idx] = bp as u8;
                for i in 0..4 {
                    vals[i + 1] = if sign[i] { -(abs[i] as i16) } else { abs[i] as i16 };
                }
            } else {
                if c_idx != 2 {
                    let e = (self.c.bypass() << 1) | self.c.bypass();
                    s.eo_class[c_idx] = e as u8;
                }
                // Categories 1 and 2 are positive, 3 and 4 negative — the sign
                // is *implied* by the category, not signalled, because an edge
                // offset only ever moves a sample towards its neighbours.
                for i in 0..4 {
                    vals[i + 1] = if i > 1 { -(abs[i] as i16) } else { abs[i] as i16 };
                }
            }
            s.offsets[c_idx] = vals;
        }
        self.pic.sao_par[idx] = s;
    }

    // -- the quadtree ------------------------------------------------------

    fn coding_quadtree(
        &mut self,
        x0: usize,
        y0: usize,
        log2_cb_size: u32,
        depth: u8,
    ) -> Result<(), &'static str> {
        let size = 1usize << log2_cb_size;
        let split = if x0 + size <= self.pic.w
            && y0 + size <= self.pic.h
            && log2_cb_size > self.sps.log2_min_cb_size
        {
            let lg = self.sps.log2_min_cb_size as usize;
            let left = if self.available(x0, y0, x0 as isize - 1, y0 as isize) {
                Some(self.pic.ct_depth[(y0 >> lg) * self.pic.cb_w + ((x0 - 1) >> lg)])
            } else {
                None
            };
            let above = if self.available(x0, y0, x0 as isize, y0 as isize - 1) {
                Some(self.pic.ct_depth[((y0 - 1) >> lg) * self.pic.cb_w + (x0 >> lg)])
            } else {
                None
            };
            let inc = syn::split_cu_ctx(depth, left, above);
            self.c.decision(ct::SPLIT_CODING_UNIT_FLAG + inc) != 0
        } else {
            log2_cb_size > self.sps.log2_min_cb_size
        };

        if self.pps.cu_qp_delta_enabled
            && log2_cb_size >= self.sps.log2_ctb_size - self.pps.diff_cu_qp_delta_depth
        {
            self.cu.qp_delta_coded = false;
            self.cu.qp_delta = 0;
            // A new quantisation group: its predictor is the **average of the
            // left and above neighbours inside this CTB** (H.265 §8.6.1),
            // falling back to the previous CU's QP where either is absent —
            // *not* simply the running QP. The difference is a slow drift that
            // grows toward the bottom-right of every picture, because the error
            // compounds through each group's predictor and then through intra
            // prediction from the blocks it mis-quantised.
            self.qp_pred = self.derive_qp_pred(x0, y0);
            self.qp_y = self.qp_pred;
        }

        if split {
            let half = size >> 1;
            self.coding_quadtree(x0, y0, log2_cb_size - 1, depth + 1)?;
            if x0 + half < self.pic.w {
                self.coding_quadtree(x0 + half, y0, log2_cb_size - 1, depth + 1)?;
            }
            if y0 + half < self.pic.h {
                self.coding_quadtree(x0, y0 + half, log2_cb_size - 1, depth + 1)?;
            }
            if x0 + half < self.pic.w && y0 + half < self.pic.h {
                self.coding_quadtree(x0 + half, y0 + half, log2_cb_size - 1, depth + 1)?;
            }
            return Ok(());
        }
        self.coding_unit(x0, y0, log2_cb_size, depth)
    }

    fn coding_unit(
        &mut self,
        x0: usize,
        y0: usize,
        log2_cb_size: u32,
        depth: u8,
    ) -> Result<(), &'static str> {
        let size = 1usize << log2_cb_size;
        self.cu = CuState {
            x: x0,
            y: y0,
            log2_size: log2_cb_size,
            intra: true,
            ct_depth: depth,
            qp_delta_coded: self.cu.qp_delta_coded,
            qp_delta: self.cu.qp_delta,
            ipm: [1; 4],
            ipm_c: [1; 4],
            ..CuState::default()
        };

        if self.trace_on {
            let bp = self.c.byte_pos() as u16;
            self.trace.push((0xFFFC, bp, x0 as u8, y0 as u8, log2_cb_size as u8, depth));
        }
        if self.pps.transquant_bypass_enabled {
            self.cu.transquant_bypass = self.c.decision(ct::CU_TRANSQUANT_BYPASS_FLAG) != 0;
        }

        let lg = self.sps.log2_min_cb_size as usize;
        if self.sh.slice_type != SliceType::I {
            let left = if self.available(x0, y0, x0 as isize - 1, y0 as isize) {
                Some(self.pic.skip[(y0 >> lg) * self.pic.cb_w + ((x0 - 1) >> lg)])
            } else {
                None
            };
            let above = if self.available(x0, y0, x0 as isize, y0 as isize - 1) {
                Some(self.pic.skip[((y0 - 1) >> lg) * self.pic.cb_w + (x0 >> lg)])
            } else {
                None
            };
            let inc = syn::skip_flag_ctx(left, above);
            self.cu.skip = self.c.decision(ct::SKIP_FLAG + inc) != 0;
            self.cu.intra = false;
        }
        // The grids are written *before* the CU body, because a later block in
        // the same CU reads them as its own neighbour.
        self.mark_cb(x0, y0, size, self.cu.skip, depth);

        if self.cu.skip {
            self.cu.part_mode = Some(PartMode::Part2Nx2N);
            self.prediction_unit(x0, y0, size, size, 0, true)?;
            self.mark_edges(x0, y0, size, size, true);
            self.set_qp(x0, y0, size);
            return Ok(());
        }

        if self.sh.slice_type != SliceType::I {
            self.cu.intra = self.c.decision(ct::PRED_MODE_FLAG) != 0;
        }
        let mut part = PartMode::Part2Nx2N;
        if !self.cu.intra || log2_cb_size == self.sps.log2_min_cb_size {
            part = syn::part_mode(
                &mut self.c,
                self.cu.intra,
                log2_cb_size,
                self.sps.log2_min_cb_size,
                self.sps.amp_enabled,
            );
        }
        self.cu.part_mode = Some(part);
        self.cu.intra_split = part == PartMode::PartNxN && self.cu.intra;

        // PCM: only on an intra 2Nx2N CU whose size is in the SPS PCM range.
        // The flag is a terminate bin (not a regular decision).
        if self.cu.intra
            && part == PartMode::Part2Nx2N
            && self.sps.pcm_enabled
            && log2_cb_size >= self.sps.log2_min_pcm_cb_size
            && log2_cb_size <= self.sps.log2_max_pcm_cb_size
            && self.c.terminate() != 0
        {
            self.pcm_sample(x0, y0, log2_cb_size)?;
            self.mark_intra_default(x0, y0, size);
            if self.sps.pcm_loop_filter_disabled {
                self.cu.transquant_bypass = true;
            }
            self.set_qp(x0, y0, size);
            return Ok(());
        }

        if self.cu.intra {
            self.intra_modes(x0, y0, log2_cb_size, part)?;
        } else {
            for i in 0..part.num_pus() {
                let (dx, dy, w, h) = part.pu_rect(i, size);
                self.prediction_unit(x0 + dx, y0 + dy, w, h, i, false)?;
            }
            for i in 0..part.num_pus() {
                let (dx, dy, w, h) = part.pu_rect(i, size);
                self.mark_edges(x0 + dx, y0 + dy, w, h, true);
            }
        }

        let mut rqt_root_cbf = true;
        if !self.cu.intra && !(part == PartMode::Part2Nx2N && self.cu.merge_2nx2n) {
            rqt_root_cbf = syn::rqt_root_cbf(&mut self.c);
        }
        if rqt_root_cbf {
            self.cu.max_trafo_depth = if self.cu.intra {
                self.sps.max_transform_hierarchy_depth_intra + self.cu.intra_split as u32
            } else {
                self.sps.max_transform_hierarchy_depth_inter
            };
            self.transform_tree(x0, y0, x0, y0, log2_cb_size, log2_cb_size, 0, 0, [false; 2])?;
        } else if self.cu.intra {
            // An intra CU always predicts, even with no residual.
            self.transform_tree(x0, y0, x0, y0, log2_cb_size, log2_cb_size, 0, 0, [false; 2])?;
        }
        self.set_qp(x0, y0, size);
        Ok(())
    }

    fn pcm_sample(&mut self, x0: usize, y0: usize, log2_cb_size: u32) -> Result<(), &'static str> {
        let size = 1usize << log2_cb_size;
        let bd_y = self.sps.pcm_bit_depth_luma;
        let bd_c = self.sps.pcm_bit_depth_chroma;
        let n_y = size * size * bd_y as usize;
        let n_c = if self.sps.chroma_format_idc != 0 {
            2 * (size / 2) * (size / 2) * bd_c as usize
        } else {
            0
        };
        let n_bits = n_y + n_c;
        let n_bytes = (n_bits + 7) / 8;
        let raw = self.c.take_raw_after_terminate(n_bytes)?;
        // Big-endian bit pack into samples, left-aligned into the picture bit depth.
        let mut bit = 0usize;
        let shift_y = self.sps.bit_depth_luma.saturating_sub(bd_y);
        let shift_c = self.sps.bit_depth_chroma.saturating_sub(bd_c);
        for j in 0..size {
            if y0 + j >= self.pic.h {
                break;
            }
            for i in 0..size {
                if x0 + i >= self.pic.w {
                    break;
                }
                let s = pcm_take_bits(raw, &mut bit, bd_y);
                self.pic.y[(y0 + j) * self.pic.w + x0 + i] = s << shift_y;
            }
        }
        if self.sps.chroma_format_idc != 0 {
            let (cx, cy, cw, ch) = (x0 / 2, y0 / 2, size / 2, size / 2);
            for plane in 0..2usize {
                let dst = if plane == 0 {
                    &mut self.pic.cb
                } else {
                    &mut self.pic.cr
                };
                let stride = self.pic.cw;
                for j in 0..ch {
                    if cy + j >= self.pic.ch {
                        break;
                    }
                    for i in 0..cw {
                        if cx + i >= self.pic.cw {
                            break;
                        }
                        let s = pcm_take_bits(raw, &mut bit, bd_c);
                        dst[(cy + j) * stride + cx + i] = s << shift_c;
                    }
                }
            }
        }
        Ok(())
    }

    fn mark_intra_default(&mut self, x0: usize, y0: usize, size: usize) {
        let n = (size / 4).max(1);
        for j in 0..n {
            for i in 0..n {
                let yy = y0 / 4 + j;
                let xx = x0 / 4 + i;
                if yy * self.pic.pu_w + xx < self.pic.pu.len() && xx < self.pic.pu_w {
                    self.pic.pu[yy * self.pic.pu_w + xx].intra = true;
                    self.pic.ipm[yy * self.pic.pu_w + xx] = intra::DC;
                }
            }
        }
    }

    fn mark_cb(&mut self, x0: usize, y0: usize, size: usize, skip: bool, depth: u8) {
        let lg = self.sps.log2_min_cb_size as usize;
        let n = (size >> lg).max(1);
        for j in 0..n {
            for i in 0..n {
                let yy = (y0 >> lg) + j;
                let xx = (x0 >> lg) + i;
                if yy * self.pic.cb_w + xx < self.pic.skip.len() && xx < self.pic.cb_w {
                    self.pic.skip[yy * self.pic.cb_w + xx] = skip;
                    self.pic.ct_depth[yy * self.pic.cb_w + xx] = depth;
                }
            }
        }
    }

    /// `qPY_PRED` for a quantisation group at `(x, y)` (H.265 §8.6.1).
    fn derive_qp_pred(&self, x: usize, y: usize) -> i32 {
        let prev = self.qp_y_prev;
        let lc = self.sps.log2_ctb_size as usize;
        let ctb = (y >> lc) * self.pic.ctb_w + (x >> lc);
        let lg = self.sps.log2_min_tb_size as usize;
        // A neighbour outside the current CTB does not count — the predictor is
        // deliberately CTB-local so a lost CTB cannot corrupt the next one's QP.
        let pick = |nx: isize, ny: isize| -> i32 {
            if nx < 0 || ny < 0 {
                return prev;
            }
            let (nx, ny) = (nx as usize, ny as usize);
            if nx >= self.pic.w || ny >= self.pic.h {
                return prev;
            }
            if (ny >> lc) * self.pic.ctb_w + (nx >> lc) != ctb {
                return prev;
            }
            if !self.available(x, y, nx as isize, ny as isize) {
                return prev;
            }
            self.pic.qp_y[(ny >> lg) * self.pic.tb_w + (nx >> lg)] as i32
        };
        let a = pick(x as isize - 1, y as isize);
        let b = pick(x as isize, y as isize - 1);
        (a + b + 1) >> 1
    }

    fn set_qp(&mut self, x0: usize, y0: usize, size: usize) {
        self.qp_y_prev = self.qp_y;
        let lg = self.sps.log2_min_tb_size as usize;
        let n = (size >> lg).max(1);
        for j in 0..n {
            for i in 0..n {
                let yy = (y0 >> lg) + j;
                let xx = (x0 >> lg) + i;
                if yy < self.pic.tb_h && xx < self.pic.tb_w {
                    self.pic.qp_y[yy * self.pic.tb_w + xx] = self.qp_y as i8;
                }
            }
        }
        if self.cu.transquant_bypass || self.sps.pcm_loop_filter_disabled && false {
            let n = (size / 8).max(1);
            for j in 0..n {
                for i in 0..n {
                    let yy = y0 / 8 + j;
                    let xx = x0 / 8 + i;
                    if yy * self.pic.edge_w + xx < self.pic.no_deblock.len() {
                        self.pic.no_deblock[yy * self.pic.edge_w + xx] = true;
                    }
                }
            }
        }
    }

    /// Mark a rectangle's edges as prediction (`pu`) or transform boundaries on
    /// the 8x8 deblocking grid.
    ///
    /// **All four sides are marked for transform edges**, not only left and
    /// top. The edge table is indexed by the *q* side of each edge (the sample
    /// to the right of a vertical edge, or below a horizontal one), so a TU at
    /// `(x0,y0)` must also write its right edge at `x0+w` and its bottom edge
    /// at `y0+h`. Marking only left/top was enough when every neighbour also
    /// ran a transform tree — the neighbour's left/top covered the shared edge
    /// — but a residual CU next to a skip (or any inter CU with
    /// `rqt_root_cbf = 0`) left the shared edge flagged as a pure prediction
    /// boundary. Coefficient-based bS then never fired, and the deblocking
    /// filter under-smoothed every such edge by a sample or two.
    ///
    /// Prediction edges still only need left/top from each PU: every coded PU
    /// marks its own left/top, so a shared PU boundary is always written by
    /// the right or lower neighbour. The four-side rule is only load-bearing
    /// for transform flags.
    fn mark_edges(&mut self, x0: usize, y0: usize, w: usize, h: usize, is_pu: bool) {
        let (vb, hb) = if is_pu { (EDGE_V_PU, EDGE_H_PU) } else { (EDGE_V_TU, EDGE_H_TU) };
        let pic_w = self.pic.w;
        let pic_h = self.pic.h;
        let edge_w = self.pic.edge_w;
        let edge_h = self.pic.edges.len() / edge_w.max(1);
        let edges = &mut self.pic.edges;

        // Vertical edge at picture-x `x` (q is the column to its right).
        let mark_v = |edges: &mut [u8], x: usize, y0: usize, h: usize, bit: u8| {
            if x % 8 != 0 || x == 0 || x >= pic_w {
                return;
            }
            let ex = x / 8;
            if ex >= edge_w {
                return;
            }
            for y in (y0..y0 + h).step_by(8) {
                let ey = y / 8;
                if ey < edge_h {
                    edges[ey * edge_w + ex] |= bit;
                }
            }
        };
        // Horizontal edge at picture-y `y` (q is the row below it).
        let mark_h = |edges: &mut [u8], y: usize, x0: usize, w: usize, bit: u8| {
            if y % 8 != 0 || y == 0 || y >= pic_h {
                return;
            }
            let ey = y / 8;
            if ey >= edge_h {
                return;
            }
            for x in (x0..x0 + w).step_by(8) {
                let ex = x / 8;
                if ex < edge_w {
                    edges[ey * edge_w + ex] |= bit;
                }
            }
        };
        // Left and top — both PU and TU.
        mark_v(edges, x0, y0, h, vb);
        mark_h(edges, y0, x0, w, hb);
        // Right and bottom — TU only (see doc comment).
        if !is_pu {
            mark_v(edges, x0 + w, y0, h, vb);
            mark_h(edges, y0 + h, x0, w, hb);
        }
    }
}

// ---------------------------------------------------------------------------
// Prediction units and modes
// ---------------------------------------------------------------------------

impl<'a> SliceDecoder<'a> {
    /// Parse the luma and chroma intra modes for a CU (§7.3.8.5).
    ///
    /// **Every `prev_intra_luma_pred_flag` is read before any `mpm_idx` or
    /// remainder.** The flags are context-coded and the rest is bypass, so the
    /// bitstream groups them — reading them per-partition interleaves the two
    /// and desynchronises the coder on any NxN CU.
    fn intra_modes(
        &mut self,
        x0: usize,
        y0: usize,
        log2_cb_size: u32,
        part: PartMode,
    ) -> Result<(), &'static str> {
        let split = part == PartMode::PartNxN;
        let pb = (1usize << log2_cb_size) >> split as usize;
        let side = 1 + split as usize;

        let mut prev = [false; 4];
        for i in 0..side * side {
            prev[i] = self.c.decision(ct::PREV_INTRA_LUMA_PRED_FLAG) != 0;
        }
        let mut mpm = [0usize; 4];
        let mut rem = [0u8; 4];
        for i in 0..side * side {
            if prev[i] {
                mpm[i] = syn::mpm_idx(&mut self.c);
            } else {
                rem[i] = syn::rem_intra_luma_pred_mode(&mut self.c);
            }
        }

        for j in 0..side {
            for i in 0..side {
                let k = j * side + i;
                let (px, py) = (x0 + i * pb, y0 + j * pb);
                let cand_left = self.intra_neighbour_mode(px, py, -1, 0);
                let cand_above = self.intra_neighbour_mode(px, py, 0, -1);
                let cands = ctu::mpm_candidates(cand_left, cand_above);
                let mode = ctu::luma_intra_mode(cands, prev[k], mpm[k], rem[k]);
                self.cu.ipm[k] = mode;
                // Written immediately: the *next* partition of this same CU
                // reads it as its left or above neighbour.
                self.set_ipm(px, py, pb, mode);
            }
        }
        // **One chroma mode per CU in 4:2:0**, however many luma partitions
        // there are — only 4:4:4 signals one per partition. Reading `side*side`
        // of them consumes three extra syntax elements on every NxN intra CU
        // and desynchronises everything after it.
        let sig = syn::intra_chroma_pred_mode(&mut self.c);
        for k in 0..side * side {
            self.cu.ipm_c[k] = ctu::chroma_intra_mode(sig, self.cu.ipm[k]);
        }
        if !split {
            self.cu.ipm[1] = self.cu.ipm[0];
            self.cu.ipm[2] = self.cu.ipm[0];
            self.cu.ipm[3] = self.cu.ipm[0];
            self.cu.ipm_c[1] = self.cu.ipm_c[0];
            self.cu.ipm_c[2] = self.cu.ipm_c[0];
            self.cu.ipm_c[3] = self.cu.ipm_c[0];
        }
        Ok(())
    }

    /// A neighbour's luma intra mode, substituted with DC where it is
    /// unavailable, inter-coded, or **above the current CTB row** — intra mode
    /// prediction does not cross a horizontal CTB boundary, which is what makes
    /// wavefront processing possible and is invisible in a one-CTB-row picture.
    fn intra_neighbour_mode(&self, x: usize, y: usize, dx: isize, dy: isize) -> u8 {
        let (nx, ny) = (x as isize + dx, y as isize + dy);
        if !self.available(x, y, nx, ny) {
            return intra::DC;
        }
        if dy < 0 {
            let ctb = 1usize << self.sps.log2_ctb_size;
            if (ny as usize) < (y / ctb) * ctb {
                return intra::DC;
            }
        }
        let p = self.pic.pu_at(nx as usize, ny as usize);
        if !p.intra {
            return intra::DC;
        }
        self.pic.ipm[(ny as usize >> 2) * self.pic.pu_w + (nx as usize >> 2)]
    }

    fn set_ipm(&mut self, x0: usize, y0: usize, size: usize, mode: u8) {
        let n = (size >> 2).max(1);
        for j in 0..n {
            for i in 0..n {
                let (xx, yy) = ((x0 >> 2) + i, (y0 >> 2) + j);
                if xx < self.pic.pu_w && yy * self.pic.pu_w + xx < self.pic.pu.len() {
                    self.pic.ipm[yy * self.pic.pu_w + xx] = mode;
                    self.pic.pu[yy * self.pic.pu_w + xx] =
                        PuInfo { intra: true, ..PuInfo::default() };
                }
            }
        }
    }

    fn set_pu(&mut self, x0: usize, y0: usize, w: usize, h: usize, info: PuInfo) {
        for j in 0..(h >> 2).max(1) {
            for i in 0..(w >> 2).max(1) {
                let (xx, yy) = ((x0 >> 2) + i, (y0 >> 2) + j);
                if xx < self.pic.pu_w && yy * self.pic.pu_w + xx < self.pic.pu.len() {
                    self.pic.pu[yy * self.pic.pu_w + xx] = info;
                }
            }
        }
    }

    /// Parse and reconstruct one inter prediction unit.
    fn prediction_unit(
        &mut self,
        x0: usize,
        y0: usize,
        w: usize,
        h: usize,
        part_idx: usize,
        is_skip: bool,
    ) -> Result<(), &'static str> {
        let max_cand = self.sh.max_num_merge_cand() as usize;
        let mut mvf = MvField::default();

        let merge = if is_skip {
            true
        } else {
            self.c.decision(ct::MERGE_FLAG) != 0
        };
        if merge && self.cu.part_mode == Some(PartMode::Part2Nx2N) {
            self.cu.merge_2nx2n = true;
        }

        if merge {
            let idx = if max_cand > 1 { syn::merge_idx(&mut self.c, max_cand) } else { 0 };
            let list = self.merge_list(x0, y0, w, h, part_idx, max_cand);
            mvf = *list.get(idx).unwrap_or(&MvField::default());
        } else {
            let is_b = self.sh.slice_type == SliceType::B;
            let idc = if is_b {
                syn::inter_pred_idc(&mut self.c, w, h, self.cu.ct_depth as usize)
            } else {
                0
            };
            mvf.pred_flag = match idc {
                0 => 1,
                1 => 2,
                _ => 3,
            };
            for l in 0..2usize {
                if !mvf.uses(l) {
                    continue;
                }
                // **The bin count comes from the slice header, not from the
                // list I managed to resolve.** `build_ref_list` drops entries
                // whose picture is missing (a stream opened on a CRA really has
                // some), so the two can differ — and then `ref_idx` reads the
                // wrong number of bins and desynchronises the coder from the
                // first prediction unit that uses AMVP.
                let nrefs =
                    if l == 0 { self.sh.num_ref_idx_l0 } else { self.sh.num_ref_idx_l1 } as usize;
                mvf.ref_idx[l] = syn::ref_idx(&mut self.c, l, nrefs) as i8;
                // `mvd_l1_zero_flag` says list 1's difference is zero and is
                // not coded at all — reading it anyway steals a bin.
                let mvd = if l == 1 && self.sh.mvd_l1_zero && mvf.pred_flag == 3 {
                    (0i16, 0i16)
                } else {
                    syn::mvd_coding(&mut self.c)
                };
                let mvp_flag = self.c.decision(ct::MVP_LX_FLAG) != 0;
                let target = *self.list_poc[l]
                    .get(mvf.ref_idx[l] as usize)
                    .unwrap_or(&self.pic.poc);
                let cands = self.amvp_list(x0, y0, w, h, l, target);
                let p = cands[mvp_flag as usize];
                mvf.mv[l] = (p.0.wrapping_add(mvd.0), p.1.wrapping_add(mvd.1));
            }
        }

        let mut info = PuInfo { intra: false, mvf, ..PuInfo::default() };
        for l in 0..2usize {
            if mvf.uses(l) {
                info.ref_poc[l] =
                    *self.list_poc[l].get(mvf.ref_idx[l] as usize).unwrap_or(&self.pic.poc);
            }
        }
        if self.trace_on {
            self.trace.push((
                0xFFFE,
                x0 as u16 | ((y0 as u16) << 8),
                mvf.pred_flag,
                mvf.mv[0].0 as u8,
                mvf.mv[0].1 as u8,
                merge as u8,
            ));
        }
        self.set_pu(x0, y0, w, h, info);
        self.motion_compensate(x0, y0, w, h, &info);
        Ok(())
    }

    /// The five spatial merge neighbours, gathered with the partition-shape
    /// exclusions that stop a PU merging with its own other half (which would
    /// make the split pointless and is therefore forbidden).
    fn merge_list(
        &self,
        x0: usize,
        y0: usize,
        w: usize,
        h: usize,
        part_idx: usize,
        max_cand: usize,
    ) -> Vec<MvField> {
        let part = self.cu.part_mode.unwrap_or(PartMode::Part2Nx2N);
        let vert_split = matches!(
            part,
            PartMode::PartNx2N | PartMode::PartnLx2N | PartMode::PartnRx2N
        );
        let horz_split = matches!(
            part,
            PartMode::Part2NxN | PartMode::Part2NxnU | PartMode::Part2NxnD
        );

        let a1 = if part_idx == 1 && vert_split {
            None
        } else {
            self.avail_inter(x0, y0, x0 as isize - 1, (y0 + h) as isize - 1)
        };
        let b1 = if part_idx == 1 && horz_split {
            None
        } else {
            self.avail_inter(x0, y0, (x0 + w) as isize - 1, y0 as isize - 1)
        };
        let b0 = self.avail_inter(x0, y0, (x0 + w) as isize, y0 as isize - 1);
        let a0 = self.avail_inter(x0, y0, x0 as isize - 1, (y0 + h) as isize);
        let b2 = self.avail_inter(x0, y0, x0 as isize - 1, y0 as isize - 1);

        let n = MergeNeighbours {
            a1: a1.map(|p| p.mvf),
            b1: b1.map(|p| p.mvf),
            b0: b0.map(|p| p.mvf),
            a0: a0.map(|p| p.mvf),
            b2: b2.map(|p| p.mvf),
        };
        let nb_refs = if self.sh.slice_type == SliceType::B {
            (self.sh.num_ref_idx_l0).min(self.sh.num_ref_idx_l1) as usize
        } else {
            self.sh.num_ref_idx_l0 as usize
        };
        let is_b = self.sh.slice_type == SliceType::B;
        // The temporal candidate always uses reference index 0 of each list.
        let target = [
            self.list_poc[0].first().copied().unwrap_or(self.pic.poc),
            self.list_poc[1].first().copied().unwrap_or(self.pic.poc),
        ];
        let temporal = if self.sh.temporal_mvp_enabled {
            self.temporal_candidate(x0, y0, w, h, target, if is_b { 3 } else { 1 })
        } else {
            None
        };
        inter::merge_candidates(&n, temporal, is_b, max_cand, nb_refs.max(1))
    }

    /// The temporal merge candidate (H.265 §8.5.3.2.8).
    ///
    /// Two positions are tried: the **bottom-right** neighbour of the
    /// prediction unit, then its **centre**. The bottom-right is only legal
    /// while it stays inside the picture *and inside the current CTB row* —
    /// otherwise it would reference motion from a CTB row that a
    /// wavefront-parallel decoder has not finished. Both are then snapped to
    /// the 16x16 grid the motion field is stored on.
    fn temporal_candidate(
        &self,
        x0: usize,
        y0: usize,
        w: usize,
        h: usize,
        target: [i32; 2],
        pred_flag: u8,
    ) -> Option<MvField> {
        let col = self.col.as_ref()?;
        let lc = self.sps.log2_ctb_size as usize;

        let read = |x: usize, y: usize| -> PuInfo {
            let (mx, my) = ((x & !15) / 16, (y & !15) / 16);
            let i = my * col.mvf_w + mx;
            col.mvf.get(i).copied().unwrap_or_default()
        };
        let br = (x0 + w, y0 + h);
        let mut temp = None;
        if (y0 >> lc) == (br.1 >> lc) && br.1 < self.pic.h && br.0 < self.pic.w {
            let p = read(br.0, br.1);
            if !p.intra {
                temp = Some(p);
            }
        }
        if temp.is_none() {
            let p = read(x0 + (w >> 1), y0 + (h >> 1));
            if !p.intra {
                temp = Some(p);
            }
        }
        let tc = temp?;

        let mut out = MvField { pred_flag, ref_idx: [0, 0], mv: [(0, 0); 2] };
        let mut any = false;
        for lx in 0..2usize {
            if pred_flag & (1 << lx) == 0 {
                continue;
            }
            // Which of the collocated block's lists to read: its only used one
            // when it is uni-predicted, and otherwise the one the specification
            // names from `collocated_from_l0` / the reference ordering.
            let l_col = if !tc.mvf.uses(0) {
                1
            } else if !tc.mvf.uses(1) {
                0
            } else {
                // Bi-predicted: if no reference is later than the current
                // picture, take list X; otherwise the *opposite* list from the
                // collocated one.
                let any_future = self.list_poc[0].iter().chain(self.list_poc[1].iter()).any(|&p| p > self.pic.poc);
                if !any_future {
                    lx
                } else if self.sh.collocated_from_l0 {
                    1
                } else {
                    0
                }
            };
            let col_poc_diff = col.poc - tc.ref_poc[l_col];
            let cur_poc_diff = self.pic.poc - target[lx];
            out.mv[lx] = if col_poc_diff == cur_poc_diff || col_poc_diff == 0 {
                tc.mvf.mv[l_col]
            } else {
                inter::mv_scale(tc.mvf.mv[l_col], col_poc_diff, cur_poc_diff)
            };
            any = true;
        }
        any.then_some(out)
    }

    fn amvp_list(
        &self,
        x0: usize,
        y0: usize,
        w: usize,
        h: usize,
        l: usize,
        target_poc: i32,
    ) -> [inter::Mv; 2] {
        let to_n = |p: Option<PuInfo>| -> Option<AmvpNeighbour> {
            p.map(|p| AmvpNeighbour { mvf: p.mvf, ref_poc: p.ref_poc, long_term: p.ref_lt })
        };
        let a0 = to_n(self.avail_inter(x0, y0, x0 as isize - 1, (y0 + h) as isize));
        let a1 = to_n(self.avail_inter(x0, y0, x0 as isize - 1, (y0 + h) as isize - 1));
        let b0 = to_n(self.avail_inter(x0, y0, (x0 + w) as isize, y0 as isize - 1));
        let b1 = to_n(self.avail_inter(x0, y0, (x0 + w) as isize - 1, y0 as isize - 1));
        let b2 = to_n(self.avail_inter(x0, y0, x0 as isize - 1, y0 as isize - 1));
        // AMVP takes the temporal candidate too, and it matters most exactly
        // where the spatial ones are absent — the first prediction unit of a
        // picture, where falling back to a zero vector is a whole-frame shift.
        let temporal = if self.sh.temporal_mvp_enabled {
            let mut t = [0i32; 2];
            t[l] = target_poc;
            self.temporal_candidate(x0, y0, w, h, t, 1 << l).map(|m| m.mv[l])
        } else {
            None
        };
        inter::amvp_candidates(a0, a1, b0, b1, b2, l, target_poc, false, self.pic.poc, temporal)
    }

    /// Motion-compensate one prediction unit into the picture.
    fn motion_compensate(&mut self, x0: usize, y0: usize, w: usize, h: usize, info: &PuInfo) {
        let mut mid: [Vec<i16>; 2] = [Vec::new(), Vec::new()];
        let mut used = [false; 2];
        for l in 0..2usize {
            if !info.mvf.uses(l) {
                continue;
            }
            let Some(f) = self.lists[l].get(info.mvf.ref_idx[l].max(0) as usize).cloned() else {
                continue;
            };
            let mv = info.mvf.mv[l];
            let mut buf = vec![0i16; MAX_PB * MAX_PB];
            let p = Plane { data: &f.y, stride: f.w, width: f.w, height: f.h };
            inter::put_luma(
                &mut buf,
                &p,
                x0 as i32 + (mv.0 >> 2) as i32,
                y0 as i32 + (mv.1 >> 2) as i32,
                w,
                h,
                (mv.0 & 3) as usize,
                (mv.1 & 3) as usize,
                self.bd_luma(),
            );
            mid[l] = buf;
            used[l] = true;
        }
        let stride = self.pic.w;
        let bd = self.bd_luma();
        match (used[0], used[1]) {
            (true, true) => {
                let a = core::mem::take(&mut mid[0]);
                let b = core::mem::take(&mut mid[1]);
                inter::bi_pred(
                    &mut self.pic.y[y0 * stride + x0..],
                    stride,
                    &a,
                    &b,
                    w,
                    h,
                    bd,
                );
            }
            (true, false) => {
                let a = core::mem::take(&mut mid[0]);
                inter::uni_pred(&mut self.pic.y[y0 * stride + x0..], stride, &a, w, h, bd);
            }
            (false, true) => {
                let b = core::mem::take(&mut mid[1]);
                inter::uni_pred(&mut self.pic.y[y0 * stride + x0..], stride, &b, w, h, bd);
            }
            (false, false) => {}
        }

        // Chroma: half the resolution, so eighth-pel phases and half-sized
        // blocks. A prediction unit narrower than 8 luma samples still has a
        // chroma block, of 2 or 4 samples.
        let (cx, cy, cw, chh) = (x0 / 2, y0 / 2, (w / 2).max(1), (h / 2).max(1));
        for plane in 0..2usize {
            let mut cmid: [Vec<i16>; 2] = [Vec::new(), Vec::new()];
            let mut cused = [false; 2];
            for l in 0..2usize {
                if !info.mvf.uses(l) {
                    continue;
                }
                let Some(f) = self.lists[l].get(info.mvf.ref_idx[l].max(0) as usize).cloned()
                else {
                    continue;
                };
                let mv = info.mvf.mv[l];
                let src = if plane == 0 { &f.cb } else { &f.cr };
                let p = Plane { data: src, stride: f.cw, width: f.cw, height: f.chh };
                let mut buf = vec![0i16; MAX_PB * MAX_PB];
                inter::put_chroma(
                    &mut buf,
                    &p,
                    cx as i32 + (mv.0 >> 3) as i32,
                    cy as i32 + (mv.1 >> 3) as i32,
                    cw,
                    chh,
                    (mv.0 & 7) as usize,
                    (mv.1 & 7) as usize,
                    self.bd_chroma(),
                );
                cmid[l] = buf;
                cused[l] = true;
            }
            let cstride = self.pic.cw;
            let bd = self.bd_chroma();
            let dst = if plane == 0 { &mut self.pic.cb } else { &mut self.pic.cr };
            let off = cy * cstride + cx;
            match (cused[0], cused[1]) {
                (true, true) => {
                    inter::bi_pred(&mut dst[off..], cstride, &cmid[0], &cmid[1], cw, chh, bd)
                }
                (true, false) => inter::uni_pred(&mut dst[off..], cstride, &cmid[0], cw, chh, bd),
                (false, true) => inter::uni_pred(&mut dst[off..], cstride, &cmid[1], cw, chh, bd),
                (false, false) => {}
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The transform tree
// ---------------------------------------------------------------------------

impl<'a> SliceDecoder<'a> {
    /// Recurse the residual quadtree (§7.3.8.8).
    ///
    /// Three things decide `split_transform_flag` when it is *not* coded, and
    /// all three are inferences a decoder must make rather than read: a block
    /// larger than the maximum transform must split; an intra NxN CU must split
    /// once so each partition gets its own transform; and an inter CU with a
    /// non-2Nx2N partition must split when the hierarchy depth is zero, so the
    /// transform never straddles two prediction units.
    #[allow(clippy::too_many_arguments)]
    fn transform_tree(
        &mut self,
        x0: usize,
        y0: usize,
        xbase: usize,
        ybase: usize,
        log2_cb_size: u32,
        log2_trafo_size: u32,
        depth: u32,
        blk_idx: usize,
        base_cbf: [bool; 2],
    ) -> Result<(), &'static str> {
        let mut cbf_cb = base_cbf[0];
        let mut cbf_cr = base_cbf[1];

        let split = if log2_trafo_size <= self.sps.log2_max_tb_size
            && log2_trafo_size > self.sps.log2_min_tb_size
            && depth < self.cu.max_trafo_depth
            && !(self.cu.intra_split && depth == 0)
        {
            syn::split_transform_flag(&mut self.c, log2_trafo_size)
        } else {
            let inter_split = self.sps.max_transform_hierarchy_depth_inter == 0
                && !self.cu.intra
                && self.cu.part_mode != Some(PartMode::Part2Nx2N)
                && depth == 0;
            log2_trafo_size > self.sps.log2_max_tb_size
                || (self.cu.intra_split && depth == 0)
                || inter_split
        };

        // Chroma CBFs are only coded while the block is above 4x4 — below that
        // the four luma sub-blocks share one chroma transform, carried by the
        // parent's flag.
        if log2_trafo_size > 2 {
            if depth == 0 || base_cbf[0] {
                cbf_cb = syn::cbf_chroma(&mut self.c, depth as usize);
            }
            if depth == 0 || base_cbf[1] {
                cbf_cr = syn::cbf_chroma(&mut self.c, depth as usize);
            }
        }

        if split {
            let half = 1usize << (log2_trafo_size - 1);
            for (i, (dx, dy)) in [(0, 0), (half, 0), (0, half), (half, half)].iter().enumerate() {
                self.transform_tree(
                    x0 + dx,
                    y0 + dy,
                    x0,
                    y0,
                    log2_cb_size,
                    log2_trafo_size - 1,
                    depth + 1,
                    i,
                    [cbf_cb, cbf_cr],
                )?;
            }
            return Ok(());
        }

        // A luma CBF is inferred for an inter CU at the root of a tree that was
        // reached at all: something must be coded, or `rqt_root_cbf` would have
        // been zero.
        let cbf_luma = if self.cu.intra || depth != 0 || cbf_cb || cbf_cr {
            syn::cbf_luma(&mut self.c, depth as usize)
        } else {
            true
        };
        self.transform_unit(
            x0, y0, xbase, ybase, log2_trafo_size, depth, blk_idx, cbf_luma, cbf_cb, cbf_cr,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn transform_unit(
        &mut self,
        x0: usize,
        y0: usize,
        xbase: usize,
        ybase: usize,
        log2_trafo_size: u32,
        depth: u32,
        blk_idx: usize,
        cbf_luma: bool,
        cbf_cb: bool,
        cbf_cr: bool,
    ) -> Result<(), &'static str> {
        let mode = if self.cu.intra_split { self.cu.ipm[blk_idx] } else { self.cu.ipm[0] };
        let mode_c = if self.cu.intra_split { self.cu.ipm_c[blk_idx] } else { self.cu.ipm_c[0] };

        // The QP delta is coded once per quantisation group, in the first
        // transform unit that carries any coefficient at all.
        if (cbf_luma || cbf_cb || cbf_cr)
            && self.pps.cu_qp_delta_enabled
            && !self.cu.qp_delta_coded
        {
            let abs = syn::cu_qp_delta_abs(&mut self.c).ok_or("hevc: corrupt cu_qp_delta_abs")?;
            let mut d = abs as i32;
            // `cu_qp_delta_sign_flag` is a **bypass** bin, and it is only
            // present when the magnitude is non-zero. Reading a context-coded
            // bin here instead consumes one that belongs to the residual that
            // immediately follows, so the very first coefficient block of the
            // picture decodes from the wrong bit onward.
            if abs != 0 && self.c.bypass() != 0 {
                d = -d;
            }
            self.cu.qp_delta = d;
            self.cu.qp_delta_coded = true;
            self.qp_y = (self.qp_pred + d).rem_euclid(52);
        }

        if self.trace_on {
            self.trace.push((
                x0 as u16,
                y0 as u16,
                log2_trafo_size as u8,
                0,
                mode,
                cbf_luma as u8 | (cbf_cb as u8) << 1 | (cbf_cr as u8) << 2,
            ));
        }
        self.mark_edges(x0, y0, 1 << log2_trafo_size, 1 << log2_trafo_size, false);

        // Luma: intra prediction happens per transform block, because it reads
        // neighbours this same CU has just reconstructed.
        if self.cu.intra {
            self.intra_predict(x0, y0, log2_trafo_size, 0, mode);
        }
        if cbf_luma {
            let scan = ctu::scan_order(self.cu.intra, log2_trafo_size, mode);
            self.residual_block(x0, y0, log2_trafo_size, 0, scan, mode)?;
            self.mark_cbf(x0, y0, 1 << log2_trafo_size);
        }

        // Chroma follows at half the size; a 4x4 luma block has no chroma
        // transform of its own — the fourth sub-block carries the parent's.
        if log2_trafo_size > 2 {
            let l2c = log2_trafo_size - 1;
            for (c_idx, present) in [(1usize, cbf_cb), (2usize, cbf_cr)] {
                if self.cu.intra {
                    self.intra_predict(x0 / 2, y0 / 2, l2c, c_idx, mode_c);
                }
                if present {
                    let scan = ctu::scan_order(self.cu.intra, l2c, mode_c);
                    self.residual_block(x0 / 2, y0 / 2, l2c, c_idx, scan, mode_c)?;
                }
            }
        } else if blk_idx == 3 {
            let l2c = log2_trafo_size;
            for (c_idx, present) in [(1usize, cbf_cb), (2usize, cbf_cr)] {
                if self.cu.intra {
                    self.intra_predict(xbase / 2, ybase / 2, l2c, c_idx, mode_c);
                }
                if present {
                    let scan = ctu::scan_order(self.cu.intra, l2c, mode_c);
                    self.residual_block(xbase / 2, ybase / 2, l2c, c_idx, scan, mode_c)?;
                }
            }
        }
        let _ = depth;
        Ok(())
    }

    fn mark_cbf(&mut self, x0: usize, y0: usize, size: usize) {
        let lg = self.pic.log2_min_tb as usize;
        for j in 0..(size >> lg).max(1) {
            for i in 0..(size >> lg).max(1) {
                let (xx, yy) = ((x0 >> lg) + i, (y0 >> lg) + j);
                if xx < self.pic.tb_w && yy < self.pic.tb_h {
                    self.pic.cbf[yy * self.pic.tb_w + xx] = true;
                }
            }
        }
    }

    /// Parse, dequantise, inverse-transform and add one residual block.
    fn residual_block(
        &mut self,
        x0: usize,
        y0: usize,
        log2_size: u32,
        c_idx: usize,
        scan_idx: usize,
        mode: u8,
    ) -> Result<(), &'static str> {
        let size = 1usize << log2_size;
        let qp = if c_idx == 0 {
            self.qp_y
        } else {
            let off = if c_idx == 1 {
                self.pps.cb_qp_offset + self.sh.cb_qp_offset
            } else {
                self.pps.cr_qp_offset + self.sh.cr_qp_offset
            };
            transform::chroma_qp((self.qp_y + off).clamp(0, 57), 1)
        };

        let bd = if c_idx == 0 {
            self.sps.bit_depth_luma
        } else {
            self.sps.bit_depth_chroma
        };
        // Dequant uses the bit-depth-offset QP (H.265 §8.6.1).
        let qp_bd = qp + 6 * (bd as i32 - 8);
        let p = residual::ResidualParams {
            log2_size,
            c_idx,
            scan_idx,
            intra: self.cu.intra,
            pred_mode_intra: mode,
            transquant_bypass: self.cu.transquant_bypass,
            transform_skip_enabled: self.pps.transform_skip_enabled,
            log2_max_transform_skip: 2,
            explicit_rdpcm_enabled: false,
            implicit_rdpcm_enabled: false,
            sign_data_hiding: self.pps.sign_data_hiding_enabled,
            persistent_rice: false,
            transform_skip_context: false,
            qp: qp_bd,
            bit_depth: bd,
            scale_matrix: None,
            dc_scale: 16,
        };
        let mut coeffs = [0i16; 32 * 32];
        let r = residual::residual_coding(
            &mut self.c,
            &mut coeffs[..size * size],
            &mut self.stat_coeff,
            &p,
        );
        if self.trace_on && self.trace.len() < 4 {
            self.trace.push((
                0xFFFF,
                coeffs[0] as u16,
                r.max_xy as u8,
                c_idx as u8,
                qp as u8,
                r.transform_skip as u8,
            ));
        }

        if !self.cu.transquant_bypass {
            if r.transform_skip {
                transform::transform_skip_scale(&mut coeffs[..size * size], log2_size, bd);
            } else if self.cu.intra && c_idx == 0 && log2_size == 2 {
                transform::inverse_transform(&mut coeffs[..size * size], log2_size, bd, true);
            } else if r.max_xy == 0 {
                transform::inverse_transform_dc(&mut coeffs[..size * size], log2_size, bd);
            } else {
                transform::inverse_transform(&mut coeffs[..size * size], log2_size, bd, false);
            }
        }

        let max = self.sample_max(c_idx);
        let (dst, stride, pw, ph) = match c_idx {
            0 => (&mut self.pic.y, self.pic.w, self.pic.w, self.pic.h),
            1 => (&mut self.pic.cb, self.pic.cw, self.pic.cw, self.pic.ch),
            _ => (&mut self.pic.cr, self.pic.cw, self.pic.cw, self.pic.ch),
        };
        for j in 0..size {
            if y0 + j >= ph {
                break;
            }
            for i in 0..size {
                if x0 + i >= pw {
                    break;
                }
                let k = (y0 + j) * stride + x0 + i;
                let v = dst[k] as i32 + coeffs[j * size + i] as i32;
                dst[k] = v.clamp(0, max) as u16;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Intra prediction against the reconstructed picture
// ---------------------------------------------------------------------------

impl<'a> SliceDecoder<'a> {
    /// Gather reference samples, substitute the missing ones, filter, predict.
    ///
    /// The availability test runs in **luma** coordinates even for a chroma
    /// block, because the z-scan table and slice map are luma-indexed; scaling
    /// the coordinates instead of the answer is how a chroma block ends up
    /// consulting a differently-shaped neighbourhood from its luma.
    fn intra_predict(&mut self, x0: usize, y0: usize, log2_size: u32, c_idx: usize, mode: u8) {
        let n = 1usize << log2_size;
        let scale = if c_idx == 0 { 1usize } else { 2 };
        let (pw, ph, stride) = if c_idx == 0 {
            (self.pic.w, self.pic.h, self.pic.w)
        } else {
            (self.pic.cw, self.pic.ch, self.pic.cw)
        };

        let mut refs = intra::Refs::default();
        let total = 4 * n + 1;
        let mut avail = vec![false; total];

        // Scan order: up the left edge from the bottom, the corner, then along
        // the top. `substitute` depends on exactly this ordering.
        let mut put = |i: usize, sx: isize, sy: isize, refs: &mut intra::Refs, av: &mut [bool]| {
            let ok = sx >= 0
                && sy >= 0
                && (sx as usize) < pw
                && (sy as usize) < ph
                && self.available(
                    x0 * scale,
                    y0 * scale,
                    sx * scale as isize,
                    sy * scale as isize,
                )
                && (!self.pps.constrained_intra_pred
                    || self.pic.pu_at((sx as usize) * scale, (sy as usize) * scale).intra);
            if ok {
                let src = match c_idx {
                    0 => &self.pic.y,
                    1 => &self.pic.cb,
                    _ => &self.pic.cr,
                };
                let v = src[(sy as usize) * stride + sx as usize];
                if i < 2 * n {
                    refs.left[2 * n - i] = v;
                } else if i == 2 * n {
                    refs.top[0] = v;
                    refs.left[0] = v;
                } else {
                    refs.top[i - 2 * n] = v;
                }
                av[i] = true;
            }
        };
        for i in 0..2 * n {
            put(
                i,
                x0 as isize - 1,
                y0 as isize + (2 * n - 1 - i) as isize,
                &mut refs,
                &mut avail,
            );
        }
        put(2 * n, x0 as isize - 1, y0 as isize - 1, &mut refs, &mut avail);
        for i in 0..2 * n {
            put(2 * n + 1 + i, x0 as isize + i as isize, y0 as isize - 1, &mut refs, &mut avail);
        }

        intra::substitute(&mut refs, n, &avail, self.bd_for(c_idx));
        intra::filter_refs(
            &mut refs,
            n,
            mode,
            log2_size,
            c_idx,
            self.sps.strong_intra_smoothing,
            self.bd_for(c_idx),
        );

        let mut block = [0u16; 32 * 32];
        intra::predict(&mut block[..n * n], n, &refs, mode, log2_size, c_idx, self.bd_for(c_idx));

        let dst = match c_idx {
            0 => &mut self.pic.y,
            1 => &mut self.pic.cb,
            _ => &mut self.pic.cr,
        };
        for j in 0..n {
            if y0 + j >= ph {
                break;
            }
            for i in 0..n {
                if x0 + i >= pw {
                    break;
                }
                dst[(y0 + j) * stride + x0 + i] = block[j * n + i];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// In-loop filters, over the whole picture
// ---------------------------------------------------------------------------

/// One side of an edge, read out of the picture's neighbour grids.
fn side_at(p: &Pic, x: usize, y: usize) -> EdgeSide {
    let pu = p.pu_at(x, y);
    let lg = p.log2_min_tb as usize;
    let cbf = p.cbf[(y >> lg) * p.tb_w + (x >> lg)];
    EdgeSide {
        intra: pu.intra,
        cbf,
        refs: [
            pu.mvf.uses(0).then_some(pu.ref_poc[0]),
            pu.mvf.uses(1).then_some(pu.ref_poc[1]),
        ],
        mvs: pu.mvf.mv,
    }
}

/// Pull `n` big-endian bits from a raw PCM payload.
fn pcm_take_bits(raw: &[u8], bit: &mut usize, n: u32) -> u16 {
    let mut v = 0u16;
    for _ in 0..n {
        let b = *bit / 8;
        let s = 7 - (*bit % 8);
        let bitv = if b < raw.len() {
            (raw[b] >> s) & 1
        } else {
            0
        };
        v = (v << 1) | bitv as u16;
        *bit += 1;
    }
    v
}

/// Deblock the whole picture: vertical edges first, then horizontal.
///
/// The order is normative, not an implementation choice — the horizontal pass
/// reads samples the vertical pass has already modified, so swapping them is a
/// different picture.
fn deblock_picture(p: &mut Pic) {
    let lg = p.log2_min_tb as usize;
    let lc = p.log2_ctb as usize;
    let qp_of = |p: &Pic, x: usize, y: usize| p.qp_y[(y >> lg) * p.tb_w + (x >> lg)] as i32;

    for pass in 0..2usize {
        let vertical = pass == 0;
        let (tu_bit, pu_bit) = if vertical { (EDGE_V_TU, EDGE_V_PU) } else { (EDGE_H_TU, EDGE_H_PU) };
        let outer = if vertical { p.w } else { p.h };
        let inner = if vertical { p.h } else { p.w };

        for a in (8..outer).step_by(8) {
            for b in (0..inner).step_by(8) {
                let (ex, ey) = if vertical { (a / 8, b / 8) } else { (b / 8, a / 8) };
                if ey * p.edge_w + ex >= p.edges.len() {
                    continue;
                }
                let f = p.edges[ey * p.edge_w + ex];
                if f & (tu_bit | pu_bit) == 0 {
                    continue;
                }
                let ctb = ((if vertical { b } else { a }) >> lc) * p.ctb_w
                    + ((if vertical { a } else { b }) >> lc);
                if p.db_off.get(ctb).copied().unwrap_or(false) {
                    continue;
                }
                let beta_off = p.db_beta.get(ctb).copied().unwrap_or(0) as i32 * 2;
                let tc_off = p.db_tc.get(ctb).copied().unwrap_or(0) as i32 * 2;

                // bS is derived per four samples along the edge, and the two
                // halves of an 8-sample segment decide independently.
                let mut tcs = [0i32; 2];
                let mut any = false;
                let mut qp_sum = 0i32;
                for half in 0..2usize {
                    let along = b + half * 4;
                    if along >= inner {
                        continue;
                    }
                    let (px, py, qx, qy) = if vertical {
                        (a - 1, along, a, along)
                    } else {
                        (along, a - 1, along, a)
                    };
                    let is_tu = f & tu_bit != 0;
                    let bs = ctu::boundary_strength(&side_at(p, px, py), &side_at(p, qx, qy), is_tu);
                    if bs == 0 {
                        continue;
                    }
                    let qp = (qp_of(p, px, py) + qp_of(p, qx, qy) + 1) >> 1;
                    tcs[half] = super::deblock::tc(qp, bs, tc_off);
                    qp_sum = qp;
                    any = true;
                }
                if !any {
                    continue;
                }
                let beta = super::deblock::beta(qp_sum, beta_off);
                let (base, xs, ys) = if vertical {
                    ((b * p.w + a) as isize, 1isize, p.w as isize)
                } else {
                    ((a * p.w + b) as isize, p.w as isize, 1isize)
                };
                // Only run where all eight samples across the edge exist.
                if vertical && (a < 4 || a + 4 > p.w) {
                    continue;
                }
                if !vertical && (a < 4 || a + 4 > p.h) {
                    continue;
                }
                if b + 8 > inner {
                    continue;
                }
                let shift = p.bit_depth_luma.saturating_sub(8);
                let beta = beta << shift;
                let tcs = [tcs[0] << shift, tcs[1] << shift];
                let max = (1i32 << p.bit_depth_luma) - 1;
                super::deblock::filter_luma_edge(
                    &mut p.y,
                    base,
                    xs,
                    ys,
                    beta,
                    tcs,
                    [false; 2],
                    [false; 2],
                    max,
                );
            }
        }

        // Chroma: only intra edges (bS == 2) filter, on a 16-luma-sample grid.
        for a in (16..outer).step_by(16) {
            for b in (0..inner).step_by(16) {
                let (ex, ey) = if vertical { (a / 8, b / 8) } else { (b / 8, a / 8) };
                if ey * p.edge_w + ex >= p.edges.len() {
                    continue;
                }
                if p.edges[ey * p.edge_w + ex] & (tu_bit | pu_bit) == 0 {
                    continue;
                }
                let ctb = ((if vertical { b } else { a }) >> lc) * p.ctb_w
                    + ((if vertical { a } else { b }) >> lc);
                if p.db_off.get(ctb).copied().unwrap_or(false) {
                    continue;
                }
                let tc_off = p.db_tc.get(ctb).copied().unwrap_or(0) as i32 * 2;
                let mut tcs = [0i32; 2];
                for half in 0..2usize {
                    let along = b + half * 8;
                    if along >= inner {
                        continue;
                    }
                    let (px, py, qx, qy) = if vertical {
                        (a - 1, along, a, along)
                    } else {
                        (along, a - 1, along, a)
                    };
                    let bs = ctu::boundary_strength(&side_at(p, px, py), &side_at(p, qx, qy), true);
                    if bs != 2 {
                        continue;
                    }
                    let qp_l = (qp_of(p, px, py) + qp_of(p, qx, qy) + 1) >> 1;
                    let qp_c = transform::chroma_qp(qp_l, 1);
                    tcs[half] = super::deblock::tc(qp_c, bs, tc_off);
                }
                if tcs[0] == 0 && tcs[1] == 0 {
                    continue;
                }
                let (ca, cb_) = (a / 2, b / 2);
                if vertical && (ca < 2 || ca + 2 > p.cw) {
                    continue;
                }
                if !vertical && (ca < 2 || ca + 2 > p.ch) {
                    continue;
                }
                if cb_ + 8 > (if vertical { p.ch } else { p.cw }) {
                    continue;
                }
                let (base, xs, ys) = if vertical {
                    ((cb_ * p.cw + ca) as isize, 1isize, p.cw as isize)
                } else {
                    ((ca * p.cw + cb_) as isize, p.cw as isize, 1isize)
                };
                let shift = p.bit_depth_chroma.saturating_sub(8);
                let tcs = [tcs[0] << shift, tcs[1] << shift];
                let max = (1i32 << p.bit_depth_chroma) - 1;
                for plane in 0..2usize {
                    let dst = if plane == 0 { &mut p.cb } else { &mut p.cr };
                    super::deblock::filter_chroma_edge(
                        dst, base, xs, ys, tcs, [false; 2], [false; 2], max,
                    );
                }
            }
        }
    }
}

/// Apply SAO over the deblocked picture.
///
/// The source is a **copy**: the edge classifier compares each sample against
/// its neighbours' *pre-SAO* values, so filtering in place makes column `x`'s
/// category depend on column `x-1`'s offset — a directional bias that reads as
/// a motion-compensation bug.
fn sao_picture(p: &mut Pic) {
    if p.sao_par.iter().all(|s| s.type_idx.iter().all(|&t| t == 0)) {
        return;
    }
    let src_y = p.y.clone();
    let src_cb = p.cb.clone();
    let src_cr = p.cr.clone();
    let ctb = 1usize << p.log2_ctb;

    for ry in 0..p.ctb_h {
        for rx in 0..p.ctb_w {
            let s = p.sao_par[ry * p.ctb_w + rx];
            for c_idx in 0..3usize {
                if s.type_idx[c_idx] == 0 {
                    continue;
                }
                let sh = if c_idx == 0 { 0 } else { 1 };
                let (pw, ph, stride) = if c_idx == 0 {
                    (p.w, p.h, p.w)
                } else {
                    (p.cw, p.ch, p.cw)
                };
                let bs = ctb >> sh;
                let x0 = rx * bs;
                let y0 = ry * bs;
                if x0 >= pw || y0 >= ph {
                    continue;
                }
                let w = bs.min(pw - x0);
                let h = bs.min(ph - y0);
                let src: &[u16] = match c_idx {
                    0 => &src_y,
                    1 => &src_cb,
                    _ => &src_cr,
                };
                let mut tmp = vec![0u16; w * h];
                if s.type_idx[c_idx] == 1 {
                    sao::band_filter(
                        &mut tmp,
                        w,
                        &src[y0 * stride + x0..],
                        stride,
                        &s.offsets[c_idx],
                        s.band_position[c_idx] as usize,
                        w,
                        h,
                        if c_idx == 0 { p.bit_depth_luma } else { p.bit_depth_chroma },
                    );
                } else {
                    // Seed with the deblocked samples so the border rows the
                    // classifier must skip keep their value.
                    for j in 0..h {
                        tmp[j * w..j * w + w]
                            .copy_from_slice(&src[(y0 + j) * stride + x0..(y0 + j) * stride + x0 + w]);
                    }
                    let borders = sao::Borders {
                        left: x0 > 0,
                        right: x0 + w < pw,
                        above: y0 > 0,
                        below: y0 + h < ph,
                    };
                    sao::edge_filter(
                        &mut tmp,
                        w,
                        src,
                        stride,
                        y0 * stride + x0,
                        &s.offsets[c_idx],
                        s.eo_class[c_idx] as usize,
                        w,
                        h,
                        &borders,
                        if c_idx == 0 { p.bit_depth_luma } else { p.bit_depth_chroma },
                    );
                }
                let dst: &mut [u16] = match c_idx {
                    0 => &mut p.y,
                    1 => &mut p.cb,
                    _ => &mut p.cr,
                };
                for j in 0..h {
                    dst[(y0 + j) * stride + x0..(y0 + j) * stride + x0 + w]
                        .copy_from_slice(&tmp[j * w..j * w + w]);
                }
            }
        }
    }
}
