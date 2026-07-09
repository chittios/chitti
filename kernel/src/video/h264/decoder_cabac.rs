//! CABAC (Main/High profile) slice + macroblock decoding: I/P/B slices,
//! adaptive 4x4/8x8 transform, Intra_8x8, a multi-frame DPB with POC-ordered
//! reference lists (+ reordering), explicit weighted P prediction, implicit
//! weighted bi-prediction, and spatial direct B modes.
//!
//! The syntax layer (context derivations, binarizations, residual significance
//! maps) is ported from FFmpeg's h264_cabac.c against the generated
//! [`super::cabac_tables`]; the arithmetic engine is [`super::cabac`]. The
//! pixel machinery is shared with the baseline path: [`super::intra`],
//! [`super::inter`], [`super::transform`], [`super::deblock`].
//!
//! Scope bounds (refused, not mis-decoded): field/MBAFF coding, scaling
//! matrices, long-term references, temporal direct, I_PCM-in-CABAC, FMO.

use super::super::bits::BitReader;
use super::cabac::Cabac;
use super::cabac_tables as ct;
use super::decoder::DecodedFrame;
use super::{deblock, inter, intra, transform, Pps, Sps};
use alloc::rc::Rc;
use alloc::vec;
use alloc::vec::Vec;

fn clip8(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

fn qpc_tab(qpi: i32) -> i32 {
    const TAB: [i32; 22] = [29, 30, 31, 32, 32, 33, 34, 34, 35, 35, 36, 36, 37, 37, 37, 38, 38, 38, 39, 39, 39, 39];
    if qpi < 30 {
        qpi
    } else {
        TAB[(qpi - 30) as usize]
    }
}

/// A decoded picture in the DPB: pixels plus the per-4x4 motion field both
/// lists (needed by spatial direct, implicit weights, and deblocking).
pub struct Pic {
    pub f: DecodedFrame,
    pub poc: i32,
    pub frame_num: u32,
    /// Per-4x4 quarter-pel motion vectors, list 0 and list 1.
    pub mv: [Vec<[i16; 2]>; 2],
    /// Per-4x4 reference index within this picture's own lists (-1 = unused).
    pub refidx: [Vec<i8>; 2],
    /// Per-4x4 POC of the referenced picture (i32::MIN = unused).
    pub refpoc: [Vec<i32>; 2],
}

/// Persistent decoder state across access units (the DPB + POC counters).
pub struct H264Dec {
    pub sps: Sps,
    pub pps: Pps,
    /// Short-term reference pictures, any order (POC/frame_num sorted per use).
    dpb: Vec<Rc<Pic>>,
    prev_poc_msb: i32,
    prev_poc_lsb: i32,
    /// Reused per-frame workspace — avoids multi‑MB alloc+zero of `Fx` every AU
    /// (the dominant overhead after CABAC itself on 1080p).
    work: Option<Fx>,
    /// Diagnostics: per-MB one-line records for the most recent AU, filled only
    /// when `trace` is set (host harness use).
    pub trace: bool,
    pub trace_log: Vec<alloc::string::String>,
    /// Diagnostics: skip the in-loop deblock entirely (host harness A/B vs a
    /// reference decoder's `skip_loop_filter`, to bisect a divergence into
    /// pre-deblock reconstruction vs the filter itself).
    pub no_deblock: bool,
}

// --- slice header ----------------------------------------------------------

struct Hdr {
    first_mb: usize,
    /// 0 = P, 1 = B, 2 = I (slice_type mod 5).
    stype: u32,
    frame_num: u32,
    idr: bool,
    nal_ref: bool,
    poc_lsb: u32,
    direct_spatial: bool,
    nref: [usize; 2],
    /// Reordering ops per list: (op, value) with op 0/1 = short-term picNum.
    reorder: [Vec<(u32, u32)>; 2],
    /// Explicit weights (P slices): log2 denoms + per-l0-ref (w, o).
    luma_ld: u32,
    chroma_ld: u32,
    wl: Vec<(i32, i32)>,
    wc: Vec<[(i32, i32); 2]>,
    cabac_init_idc: u32,
    qp: i32,
    dbf_idc: u32,
    aoff: i32,
    boff: i32,
    /// MMCO ops (op, value1) — only op 1 (unmark short-term) is supported.
    mmco: Vec<(u32, u32)>,
    /// Byte offset where the (aligned) CABAC slice data begins.
    data_byte: usize,
}

fn parse_slice_header(rbsp: &[u8], sps: &Sps, pps: &Pps, idr: bool, nal_ref: bool) -> Result<Hdr, &'static str> {
    let mut r = BitReader::new(rbsp);
    let first_mb = r.ue()? as usize;
    let st_raw = r.ue()?;
    let stype = st_raw % 5;
    if stype > 2 {
        return Err("h264: SP/SI slices not supported");
    }
    let _pps_id = r.ue()?;
    let frame_num = r.u(sps.log2_max_frame_num)?;
    if !sps.frame_mbs_only_flag {
        return Err("h264: field coding not supported");
    }
    if idr {
        let _idr_pic_id = r.ue()?;
    }
    let mut poc_lsb = 0;
    if sps.pic_order_cnt_type == 0 {
        poc_lsb = r.u(sps.log2_max_poc_lsb)?;
        if pps.bottom_field_pic_order_present {
            let _delta_bottom = r.se()?;
        }
    } else if sps.pic_order_cnt_type == 1 {
        return Err("h264: poc type 1 not supported");
    }
    if pps.redundant_pic_cnt_present {
        let _ = r.ue()?;
    }
    let mut direct_spatial = true;
    if stype == 1 {
        direct_spatial = r.flag()?;
    }
    let mut nref = [pps.num_ref_idx_l0_default as usize, pps.num_ref_idx_l1_default as usize];
    if stype == 0 || stype == 1 {
        if r.flag()? {
            nref[0] = r.ue()? as usize + 1;
            if stype == 1 {
                nref[1] = r.ue()? as usize + 1;
            }
        }
    } else {
        nref = [0, 0];
    }
    if stype == 0 {
        nref[1] = 0; // P slices have no list 1
    }
    if nref[0] > 16 || nref[1] > 16 {
        return Err("h264: bad num_ref_idx");
    }
    // ref_pic_list_modification.
    let mut reorder: [Vec<(u32, u32)>; 2] = [Vec::new(), Vec::new()];
    let nlists = if stype == 1 { 2 } else if stype == 0 { 1 } else { 0 };
    for (list, ro) in reorder.iter_mut().enumerate().take(nlists) {
        let _ = list;
        if r.flag()? {
            loop {
                let op = r.ue()?;
                if op == 3 {
                    break;
                }
                if op > 1 {
                    return Err("h264: long-term reordering not supported");
                }
                let val = r.ue()?;
                ro.push((op, val));
                if ro.len() > 32 {
                    return Err("h264: runaway reorder list");
                }
            }
        }
    }
    // pred_weight_table (explicit P; explicit-B would be weighted_bipred_idc 1).
    let mut luma_ld = 0;
    let mut chroma_ld = 0;
    let mut wl: Vec<(i32, i32)> = Vec::new();
    let mut wc: Vec<[(i32, i32); 2]> = Vec::new();
    let explicit_b = stype == 1 && pps.weighted_bipred_idc == 1;
    if (stype == 0 && pps.weighted_pred) || explicit_b {
        if explicit_b {
            return Err("h264: explicit weighted B not supported");
        }
        luma_ld = r.ue()?;
        chroma_ld = r.ue()?;
        if luma_ld > 7 || chroma_ld > 7 {
            return Err("h264: bad weight denom");
        }
        for _ in 0..nref[0] {
            let (mut lw, mut lo) = (1i32 << luma_ld, 0i32);
            if r.flag()? {
                lw = r.se()?;
                lo = r.se()?;
            }
            wl.push((lw, lo));
            let mut cwo = [(1i32 << chroma_ld, 0i32); 2];
            if r.flag()? {
                for c in cwo.iter_mut() {
                    c.0 = r.se()?;
                    c.1 = r.se()?;
                }
            }
            wc.push(cwo);
        }
    }
    // dec_ref_pic_marking.
    let mut mmco: Vec<(u32, u32)> = Vec::new();
    if nal_ref {
        if idr {
            let _no_output = r.flag()?;
            if r.flag()? {
                return Err("h264: long-term IDR not supported");
            }
        } else if r.flag()? {
            loop {
                let op = r.ue()?;
                if op == 0 {
                    break;
                }
                match op {
                    1 => mmco.push((1, r.ue()?)),
                    _ => return Err("h264: unsupported MMCO op"),
                }
                if mmco.len() > 32 {
                    return Err("h264: runaway MMCO");
                }
            }
        }
    }
    let mut cabac_init_idc = 0;
    if stype != 2 {
        cabac_init_idc = r.ue()?;
        if cabac_init_idc > 2 {
            return Err("h264: bad cabac_init_idc");
        }
    }
    let qp = pps.pic_init_qp + r.se()?;
    let (mut dbf_idc, mut aoff, mut boff) = (0u32, 0i32, 0i32);
    if pps.deblocking_filter_control_present {
        dbf_idc = r.ue()?;
        if dbf_idc != 1 {
            aoff = r.se()? * 2;
            boff = r.se()? * 2;
        }
    }
    // cabac_alignment_one_bit up to the byte boundary.
    while r.bit_pos() % 8 != 0 {
        let _ = r.bit()?;
    }
    Ok(Hdr {
        first_mb,
        stype,
        frame_num,
        idr,
        nal_ref,
        poc_lsb,
        direct_spatial,
        nref,
        reorder,
        luma_ld,
        chroma_ld,
        wl,
        wc,
        cabac_init_idc,
        qp,
        dbf_idc,
        aoff,
        boff,
        mmco,
        data_byte: r.bit_pos() / 8,
    })
}

// --- per-frame decode state -------------------------------------------------

/// Frame decode state for the CABAC path (both-list motion + CABAC neighbour
/// caches). One instance per access unit, shared by its slices.
struct Fx {
    w: usize,
    h: usize,
    cw: usize,
    ch: usize,
    mbw: usize,
    mbh: usize,
    ny4: usize,
    nc2: usize,
    /// Reconstructed samples (0–255). Storing u8 (not i32) cuts
    /// 1080p plane traffic 4× and removes the per-AU clip_plane step.
    y: Vec<u8>,
    cb: Vec<u8>,
    cr: Vec<u8>,
    // per-4x4
    nnz_y: Vec<i32>,
    nnz_u: Vec<i32>,
    nnz_v: Vec<i32>,
    mode_y: Vec<i32>,
    mv: [Vec<[i16; 2]>; 2],
    refidx: [Vec<i8>; 2],
    refpoc: [Vec<i32>; 2],
    mvd: [Vec<[u8; 2]>; 2],
    /// Per-4×4 direct flag — generation-stamped (`== pix_gen` ⇒ set this AU).
    dirf: Vec<u16>,
    /// Per-4x4, per-list "motion decoded" stamps (`== pix_gen` ⇒ final this AU):
    /// motion-vector prediction may only use cells whose motion for that list
    /// is final (§6.4.11.7 — an above-right cell of a *later* partition in the
    /// same MB is unavailable). Cells of partitions that do not use a list are
    /// marked immediately (available-with-ref -1).
    mvok: [Vec<u16>; 2],
    // per-MB
    mbslice: Vec<i32>,
    mbqp: Vec<i32>,
    mbintra: Vec<bool>,
    mbi16: Vec<bool>,
    mbinxn: Vec<bool>,
    mbt8: Vec<bool>,
    mbskip: Vec<bool>,
    /// MB-level direct flag (B_Skip / B_Direct_16x16) — the B mb_type ctx uses
    /// this, NOT the per-4x4 direct flags (a B_8x8 with direct subs is not a
    /// direct macroblock for context purposes).
    mbdir: Vec<bool>,
    mbcbp_l: Vec<u8>,
    mbcbp_c: Vec<u8>,
    mbdcf: Vec<u8>,
    mbcpm: Vec<u8>,
    cur_slice: i32,
    trace: Option<Vec<alloc::string::String>>,
    // Per-4×4 luma / per-MB chroma decoded stamps (drive intra-prediction
    // neighbour availability). Generation-stamped (`== pix_gen` ⇒ decoded this
    // AU) so recycle is O(1). Per-pixel stamps were ~6 MiB at 1080p and the
    // dominant write traffic after residual; 4×4 stamps are 16× smaller and
    // match how neighbours are actually queried (block edges only).
    decy: Vec<u16>,
    decu: Vec<u16>,
    decv: Vec<u16>,
    pix_gen: u16,
}

impl Fx {
    fn new(sps: &Sps) -> Fx {
        let mbw = sps.pic_width_in_mbs as usize;
        let mbh = sps.pic_height_in_map_units as usize;
        let (w, h) = (mbw * 16, mbh * 16);
        let (cw, ch) = (w / 2, h / 2);
        let ny4 = mbw * 4;
        let nc2 = mbw * 2;
        let n4 = ny4 * mbh * 4;
        let nmb = mbw * mbh;
        Fx {
            w,
            h,
            cw,
            ch,
            mbw,
            mbh,
            ny4,
            nc2,
            y: vec![0; w * h],
            cb: vec![0; cw * ch],
            cr: vec![0; cw * ch],
            nnz_y: vec![0; n4],
            nnz_u: vec![0; nc2 * mbh * 2],
            nnz_v: vec![0; nc2 * mbh * 2],
            mode_y: vec![-1; n4],
            mv: [vec![[0; 2]; n4], vec![[0; 2]; n4]],
            refidx: [vec![-1; n4], vec![-1; n4]],
            refpoc: [vec![i32::MIN; n4], vec![i32::MIN; n4]],
            mvd: [vec![[0; 2]; n4], vec![[0; 2]; n4]],
            dirf: vec![0; n4],
            mvok: [vec![0; n4], vec![0; n4]],
            mbslice: vec![-1; nmb],
            mbqp: vec![0; nmb],
            mbintra: vec![false; nmb],
            mbi16: vec![false; nmb],
            mbinxn: vec![false; nmb],
            mbt8: vec![false; nmb],
            mbskip: vec![false; nmb],
            mbdir: vec![false; nmb],
            mbcbp_l: vec![0; nmb],
            mbcbp_c: vec![0; nmb],
            mbdcf: vec![0; nmb],
            mbcpm: vec![0; nmb],
            cur_slice: -1,
            trace: None,
            // n4 4×4 luma cells; one stamp per macroblock for chroma.
            decy: vec![0; n4],
            decu: vec![0; nmb],
            decv: vec![0; nmb],
            pix_gen: 1,
        }
    }

    /// Luma sample `(px,py)` is available for intra neighbour use.
    #[inline]
    fn y_dec_xy(&self, px: usize, py: usize) -> bool {
        let i = (py / 4) * self.ny4 + px / 4;
        self.decy.get(i).copied().unwrap_or(0) == self.pix_gen
    }
    /// Mark one 4×4 luma block at 4×4-grid coords `(bx4, by4)` decoded.
    #[inline]
    fn mark_y4(&mut self, bx4: usize, by4: usize) {
        let i = by4 * self.ny4 + bx4;
        if let Some(s) = self.decy.get_mut(i) {
            *s = self.pix_gen;
        }
    }
    /// Chroma of macroblock `mb` is available.
    #[inline]
    fn c_dec(&self, mb: usize) -> bool {
        self.decu.get(mb).copied().unwrap_or(0) == self.pix_gen
    }
    #[inline]
    fn mark_c_mb(&mut self, mb: usize) {
        if let Some(s) = self.decu.get_mut(mb) {
            *s = self.pix_gen;
        }
        if let Some(s) = self.decv.get_mut(mb) {
            *s = self.pix_gen;
        }
    }

    /// Reset for a new AU **without** reallocating. Pixel planes are not
    /// cleared — every macroblock writes them. Motion / decoded stamps are
    /// generation-based (O(1)); only MB-level flags that are *read* as
    /// "false when unwritten" get a bulk clear. Motion fields may have been
    /// `take`n into a Pic; reallocate those if empty.
    fn recycle(&mut self) {
        let n4 = self.ny4 * self.mbh * 4;
        let nmb = self.mbw * self.mbh;
        // mvd + nnz are read for CABAC neighbour contexts even when a neighbour
        // was Skip (no store_mvd / residual) — must be zeroed each AU.
        // mv/refidx/refpoc are gated by mvok gen stamps; dirf by its stamp.
        self.nnz_y.fill(0);
        self.nnz_u.fill(0);
        self.nnz_v.fill(0);
        self.mvd[0].fill([0; 2]);
        self.mvd[1].fill([0; 2]);
        // mode_y: readers check mbinxn first; only Intra_NxN writes it.
        if self.mv[0].len() != n4 {
            self.mv[0] = vec![[0; 2]; n4];
            self.mv[1] = vec![[0; 2]; n4];
            self.refidx[0] = vec![-1; n4];
            self.refidx[1] = vec![-1; n4];
            self.refpoc[0] = vec![i32::MIN; n4];
            self.refpoc[1] = vec![i32::MIN; n4];
            self.mvd[0] = vec![[0; 2]; n4];
            self.mvd[1] = vec![[0; 2]; n4];
            self.dirf = vec![0; n4];
            self.mvok = [vec![0; n4], vec![0; n4]];
            self.nnz_y = vec![0; n4];
            self.nnz_u = vec![0; self.nc2 * self.mbh * 2];
            self.nnz_v = vec![0; self.nc2 * self.mbh * 2];
            self.mode_y = vec![-1; n4];
        }
        if self.mbslice.len() != nmb {
            self.mbslice = vec![-1; nmb];
            self.mbqp = vec![0; nmb];
            self.mbintra = vec![false; nmb];
            self.mbi16 = vec![false; nmb];
            self.mbinxn = vec![false; nmb];
            self.mbt8 = vec![false; nmb];
            self.mbskip = vec![false; nmb];
            self.mbdir = vec![false; nmb];
            self.mbcbp_l = vec![0; nmb];
            self.mbcbp_c = vec![0; nmb];
            self.mbdcf = vec![0; nmb];
            self.mbcpm = vec![0; nmb];
        } else {
            // MB flags: small (nmb ≈ 8k at 1080p) and read as default-false.
            self.mbslice.fill(-1);
            self.mbqp.fill(0);
            self.mbintra.fill(false);
            self.mbi16.fill(false);
            self.mbinxn.fill(false);
            self.mbt8.fill(false);
            self.mbskip.fill(false);
            self.mbdir.fill(false);
            self.mbcbp_l.fill(0);
            self.mbcbp_c.fill(0);
            self.mbdcf.fill(0);
            self.mbcpm.fill(0);
        }
        // Neighbour-availability + mvok/dirf: bump generation (O(1)).
        self.pix_gen = self.pix_gen.wrapping_add(1);
        if self.pix_gen == 0 {
            self.decy.fill(0);
            self.decu.fill(0);
            self.decv.fill(0);
            self.mvok[0].fill(0);
            self.mvok[1].fill(0);
            self.dirf.fill(0);
            self.pix_gen = 1;
        }
        self.cur_slice = -1;
        self.trace = None;
    }

    #[inline]
    fn sl(&self, mb: usize) -> bool {
        self.mbslice[mb] == self.cur_slice && self.mbslice[mb] >= 0
    }
    /// Left/top macroblock index, if available in the current slice.
    #[inline]
    fn mb_a(&self, mb: usize) -> Option<usize> {
        if mb % self.mbw > 0 && self.sl(mb - 1) {
            Some(mb - 1)
        } else {
            None
        }
    }
    #[inline]
    fn mb_b(&self, mb: usize) -> Option<usize> {
        if mb >= self.mbw && self.sl(mb - self.mbw) {
            Some(mb - self.mbw)
        } else {
            None
        }
    }

    /// Motion-pred neighbour lookup: 4x4 coords → (mv, refidx, mb-available).
    /// refidx -1 = available-but-unused/intra (mv contributes 0).
    fn nb(&self, list: usize, bx4: i32, by4: i32) -> ([i16; 2], i8, bool) {
        if bx4 < 0 || by4 < 0 || bx4 >= self.ny4 as i32 || by4 >= (self.mbh * 4) as i32 {
            return ([0, 0], -1, false);
        }
        let mb = (by4 as usize / 4) * self.mbw + bx4 as usize / 4;
        if !self.sl(mb) {
            return ([0, 0], -1, false);
        }
        let i = by4 as usize * self.ny4 + bx4 as usize;
        if self.mvok[list].get(i).copied().unwrap_or(0) != self.pix_gen {
            return ([0, 0], -1, false);
        }
        (self.mv[list][i], self.refidx[list][i], true)
    }

    /// Median motion-vector prediction (§8.4.1.3), per list. `kind`: 0 =
    /// median, 1 = 16x8 top (B), 2 = 16x8 bottom (A), 3 = 8x16 left (A),
    /// 4 = 8x16 right (C).
    fn pred_mv(&self, list: usize, bx4: i32, by4: i32, pw4: i32, ridx: i8, kind: u8) -> [i16; 2] {
        let a = self.nb(list, bx4 - 1, by4);
        let b = self.nb(list, bx4, by4 - 1);
        let mut c = self.nb(list, bx4 + pw4, by4 - 1);
        if !c.2 {
            c = self.nb(list, bx4 - 1, by4 - 1);
        }
        match kind {
            1 if b.1 == ridx => return b.0,
            2 if a.1 == ridx => return a.0,
            3 if a.1 == ridx => return a.0,
            4 if c.1 == ridx => return c.0,
            _ => {}
        }
        if !b.2 && !c.2 && a.2 {
            return a.0;
        }
        // Exactly one neighbour shares `ridx` → that neighbour's MV.
        let mut same_mv = None;
        let mut n_same = 0u8;
        for n in [&a, &b, &c] {
            if n.1 == ridx {
                n_same += 1;
                same_mv = Some(n.0);
            }
        }
        if n_same == 1 {
            return same_mv.unwrap_or([0, 0]);
        }
        [
            inter::median3(a.0[0] as i32, b.0[0] as i32, c.0[0] as i32) as i16,
            inter::median3(a.0[1] as i32, b.0[1] as i32, c.0[1] as i32) as i16,
        ]
    }

    /// P_Skip motion (§8.4.1.1): zero if edge/zero-ref-zero-mv neighbour.
    fn skip_mv(&self, mb_x: usize, mb_y: usize) -> [i16; 2] {
        let bx4 = mb_x as i32 * 4;
        let by4 = mb_y as i32 * 4;
        let a = self.nb(0, bx4 - 1, by4);
        let b = self.nb(0, bx4, by4 - 1);
        if !a.2 || !b.2 || (a.1 == 0 && a.0 == [0, 0]) || (b.1 == 0 && b.0 == [0, 0]) {
            return [0, 0];
        }
        self.pred_mv(0, bx4, by4, 4, 0, 0)
    }

    fn store_mv(&mut self, list: usize, bx4: usize, by4: usize, pw4: usize, ph4: usize, mv: [i16; 2], ridx: i8, rpoc: i32) {
        let g = self.pix_gen;
        for yy in 0..ph4 {
            for xx in 0..pw4 {
                let i = (by4 + yy) * self.ny4 + bx4 + xx;
                self.mv[list][i] = mv;
                self.refidx[list][i] = ridx;
                self.refpoc[list][i] = rpoc;
                self.mvok[list][i] = g;
            }
        }
    }

    /// Record a partition's reference index before its MV is decoded (the wire
    /// order sends all refs first). Does NOT mark the cells motion-decoded.
    fn store_ref_only(&mut self, list: usize, bx4: usize, by4: usize, pw4: usize, ph4: usize, ridx: i8, rpoc: i32) {
        for yy in 0..ph4 {
            for xx in 0..pw4 {
                let i = (by4 + yy) * self.ny4 + bx4 + xx;
                self.refidx[list][i] = ridx;
                self.refpoc[list][i] = rpoc;
            }
        }
    }
    fn store_mvd(&mut self, list: usize, bx4: usize, by4: usize, pw4: usize, ph4: usize, mvd: [u8; 2]) {
        for yy in 0..ph4 {
            for xx in 0..pw4 {
                self.mvd[list][(by4 + yy) * self.ny4 + bx4 + xx] = mvd;
            }
        }
    }
}

// --- CABAC syntax elements ---------------------------------------------------

/// I-slice/intra mb_type suffix tree (FFmpeg decode_cabac_intra_mb_type).
/// Returns 0 (I_NxN), 1..24 (I16 variants) or 25 (I_PCM).
fn cabac_intra_mb_type(cb: &mut Cabac, base: usize, intra_slice: bool, inc: usize) -> u32 {
    let mut st = base;
    if intra_slice {
        if cb.decision(st + inc) == 0 {
            return 0;
        }
        st += 2;
    } else if cb.decision(st) == 0 {
        return 0;
    }
    if cb.terminate() == 1 {
        return 25;
    }
    let intra = intra_slice as usize;
    let mut t = 1u32;
    t += 12 * cb.decision(st + 1);
    if cb.decision(st + 2) != 0 {
        t += 4 + 4 * cb.decision(st + 2 + intra);
    }
    t += 2 * cb.decision(st + 3 + intra);
    t += cb.decision(st + 3 + 2 * intra);
    t
}

/// mvd component (FFmpeg decode_cabac_mb_mvd): UEG3 with ctx from the summed
/// neighbour |mvd|. Returns (mvd, clamped-cache-magnitude).
fn cabac_mvd(cb: &mut Cabac, ctxbase: usize, amvd: i32) -> Result<(i32, u8), &'static str> {
    let inc = (amvd > 2) as usize + (amvd > 32) as usize;
    if cb.decision(ctxbase + inc) == 0 {
        return Ok((0, 0));
    }
    let mut mvd = 1i32;
    let mut ctx = ctxbase + 3;
    while mvd < 9 && cb.decision(ctx) != 0 {
        if mvd < 4 {
            ctx += 1;
        }
        mvd += 1;
    }
    if mvd >= 9 {
        let mut k = 3u32;
        while cb.bypass() != 0 {
            mvd += 1 << k;
            k += 1;
            if k > 24 {
                return Err("h264 cabac: mvd overflow");
            }
        }
        while k > 0 {
            k -= 1;
            mvd += (cb.bypass() as i32) << k;
        }
    }
    let cache = mvd.min(70) as u8;
    Ok((cb.bypass_sign(mvd), cache))
}

/// ref_idx (unary, ctx chain 54.. per FFmpeg decode_cabac_mb_ref).
fn cabac_ref_idx(cb: &mut Cabac, mut ctx: usize) -> Result<i8, &'static str> {
    let mut r = 0i8;
    while cb.decision(54 + ctx) != 0 {
        r += 1;
        ctx = (ctx >> 2) + 4;
        if r >= 32 {
            return Err("h264 cabac: ref_idx overflow");
        }
    }
    Ok(r)
}

/// One residual block's levels in scan order (§9.3.3.1.3 significance maps +
/// UEG0 levels, FFmpeg decode_cabac_residual_internal). `cat` picks the ctx
/// bases; `max_coeff` = 4 (chroma DC), 15 (AC), 16 (4x4/luma DC), 64 (8x8).
/// `sc` receives levels at scan positions (index 0 = first scanned coeff, i.e.
/// the DC-skipped offset is the caller's concern). Returns #nonzero coeffs.
fn cabac_residual(cb: &mut Cabac, cat: usize, max_coeff: usize, sc: &mut [i32]) -> usize {
    let sig_base = ct::SIG_COEFF_OFFSET[cat] as usize;
    let last_base = ct::LAST_COEFF_OFFSET[cat] as usize;
    let abs_base = ct::ABS_LEVEL_OFFSET[cat] as usize;
    let mut index = [0usize; 64];
    let mut count = 0usize;
    if max_coeff == 64 {
        let mut last = 0usize;
        let mut broke = false;
        while last < 63 {
            if cb.decision(sig_base + ct::SIG_COEFF_8X8[last] as usize) != 0 {
                index[count] = last;
                count += 1;
                if cb.decision(last_base + ct::LAST_COEFF_8X8[last] as usize) != 0 {
                    broke = true;
                    break;
                }
            }
            last += 1;
        }
        if !broke {
            index[count] = 63;
            count += 1;
        }
    } else {
        let mut last = 0usize;
        let mut broke = false;
        while last < max_coeff - 1 {
            if cb.decision(sig_base + last) != 0 {
                index[count] = last;
                count += 1;
                if cb.decision(last_base + last) != 0 {
                    broke = true;
                    break;
                }
            }
            last += 1;
        }
        if !broke {
            index[count] = max_coeff - 1;
            count += 1;
        }
    }
    // Levels, last-to-first, with the node-context state machine.
    let mut node = 0usize;
    let total = count;
    let mut i = count;
    while i > 0 {
        i -= 1;
        let pos = index[i];
        if cb.decision(abs_base + ct::LVL1_CTX[node] as usize) == 0 {
            node = ct::LVL_TRANSITION[0][node] as usize;
            sc[pos] = cb.bypass_sign(1);
        } else {
            let mut mag = 2i32;
            let gt1 = abs_base + ct::LVLGT1_CTX[node] as usize;
            node = ct::LVL_TRANSITION[1][node] as usize;
            while mag < 15 && cb.decision(gt1) != 0 {
                mag += 1;
            }
            if mag >= 15 {
                let mut j = 0u32;
                while cb.bypass() != 0 && j < 23 {
                    j += 1;
                }
                mag = 1;
                while j > 0 {
                    j -= 1;
                    mag = mag * 2 + cb.bypass() as i32;
                }
                mag += 14;
            }
            sc[pos] = cb.bypass_sign(mag);
        }
    }
    total
}

// --- reference lists / weights ----------------------------------------------

fn frame_num_wrap(fnum: u32, cur: u32, max: u32) -> i32 {
    if fnum > cur {
        fnum as i32 - max as i32
    } else {
        fnum as i32
    }
}

fn build_ref_lists(dpb: &[Rc<Pic>], hdr: &Hdr, cur_poc: i32, max_fn: u32) -> Result<[Vec<Rc<Pic>>; 2], &'static str> {
    let mut l0: Vec<Rc<Pic>> = dpb.to_vec();
    let mut l1: Vec<Rc<Pic>> = Vec::new();
    if hdr.stype == 0 {
        // P: descending FrameNumWrap.
        l0.sort_by_key(|p| -(frame_num_wrap(p.frame_num, hdr.frame_num, max_fn)));
    } else if hdr.stype == 1 {
        // B: l0 = {poc<cur desc} + {poc>cur asc}; l1 mirrored.
        let mut before: Vec<Rc<Pic>> = dpb.iter().filter(|p| p.poc <= cur_poc).cloned().collect();
        let mut after: Vec<Rc<Pic>> = dpb.iter().filter(|p| p.poc > cur_poc).cloned().collect();
        before.sort_by_key(|p| -p.poc);
        after.sort_by_key(|p| p.poc);
        l0 = before.iter().chain(after.iter()).cloned().collect();
        l1 = after.iter().chain(before.iter()).cloned().collect();
        if l1.len() > 1 && l0.len() == l1.len() && l0.iter().zip(l1.iter()).all(|(a, b)| Rc::ptr_eq(a, b)) {
            l1.swap(0, 1);
        }
    }
    let mut lists = [l0, l1];
    // Apply reordering (§8.2.4.3.1, short-term only).
    for list in 0..2 {
        if hdr.reorder[list].is_empty() {
            continue;
        }
        let nact = hdr.nref[list];
        let lst = &mut lists[list];
        // Extend to at least nact entries so insertion below has room.
        let mut pred = hdr.frame_num as i32;
        let maxf = max_fn as i32;
        let mut ridx = 0usize;
        for &(op, val) in &hdr.reorder[list] {
            let d = val as i32 + 1;
            let mut no_wrap = if op == 0 { pred - d } else { pred + d };
            if no_wrap < 0 {
                no_wrap += maxf;
            } else if no_wrap >= maxf {
                no_wrap -= maxf;
            }
            pred = no_wrap;
            let pic_num = if no_wrap > hdr.frame_num as i32 { no_wrap - maxf } else { no_wrap };
            let found = dpb
                .iter()
                .find(|p| frame_num_wrap(p.frame_num, hdr.frame_num, max_fn) == pic_num)
                .cloned()
                .ok_or("h264: reorder target not in DPB")?;
            // Insert at ridx, then drop the first later duplicate.
            if ridx > lst.len() {
                return Err("h264: reorder index out of range");
            }
            lst.insert(ridx, found.clone());
            let mut k = ridx + 1;
            while k < lst.len() {
                if Rc::ptr_eq(&lst[k], &found) {
                    lst.remove(k);
                    break;
                }
                k += 1;
            }
            ridx += 1;
            if lst.len() > nact + 16 {
                return Err("h264: reorder list runaway");
            }
        }
    }
    for (list, lst) in lists.iter_mut().enumerate() {
        if hdr.nref[list] > 0 {
            if lst.is_empty() {
                return Err("h264: empty reference list");
            }
            lst.truncate(hdr.nref[list]);
        } else {
            lst.clear();
        }
    }
    Ok(lists)
}

/// Implicit bi-prediction weights (§8.4.2.3.2): (w0, w1) in 1/64 units, or
/// None → default average.
fn implicit_weights(cur_poc: i32, poc0: i32, poc1: i32) -> Option<(i32, i32)> {
    if poc0 == poc1 {
        return None;
    }
    let td = (poc1 - poc0).clamp(-128, 127);
    let tb = (cur_poc - poc0).clamp(-128, 127);
    let tx = (16384 + (td / 2).abs()) / td;
    let dsf = ((tb * tx + 32) >> 6).clamp(-1024, 1023);
    let w1 = dsf >> 2;
    if !(-64..=128).contains(&w1) {
        return None;
    }
    Some((64 - w1, w1))
}

// --- main decoder -------------------------------------------------------------

impl H264Dec {
    pub fn new(sps: Sps, pps: Pps) -> Result<H264Dec, &'static str> {
        if pps.scaling_matrix_present {
            return Err("h264: scaling matrices not supported");
        }
        if !sps.frame_mbs_only_flag {
            return Err("h264: interlaced streams not supported");
        }
        if pps.num_slice_groups > 1 {
            return Err("h264: FMO not supported");
        }
        Ok(H264Dec {
            sps,
            pps,
            dpb: Vec::new(),
            prev_poc_msb: 0,
            prev_poc_lsb: 0,
            work: None,
            trace: false,
            trace_log: Vec::new(),
            no_deblock: false,
        })
    }

    /// Number of reference pictures currently held (diagnostics).
    pub fn dpb_len(&self) -> usize {
        self.dpb.len()
    }

    /// Reset the DPB/POC state (seek to an IDR).
    pub fn reset(&mut self) {
        self.dpb.clear();
        self.prev_poc_msb = 0;
        self.prev_poc_lsb = 0;
    }

    /// Decode one access unit (all slice NALs of one frame, `(rbsp, is_idr,
    /// nal_ref_idc)`) and return the decoded picture.
    pub fn decode_au(&mut self, slices: &[(Vec<u8>, bool, u8)]) -> Result<Rc<Pic>, &'static str> {
        let (first_rbsp, first_idr, first_ref) = slices.first().ok_or("h264: no slices")?;
        let hdr0 = parse_slice_header(first_rbsp, &self.sps, &self.pps, *first_idr, *first_ref > 0)?;
        if hdr0.idr {
            self.dpb.clear();
            self.prev_poc_msb = 0;
            self.prev_poc_lsb = 0;
        }
        // POC type 0 (§8.2.1.1).
        let max_lsb = 1i32 << self.sps.log2_max_poc_lsb;
        let lsb = hdr0.poc_lsb as i32;
        let msb = if lsb < self.prev_poc_lsb && self.prev_poc_lsb - lsb >= max_lsb / 2 {
            self.prev_poc_msb + max_lsb
        } else if lsb > self.prev_poc_lsb && lsb - self.prev_poc_lsb > max_lsb / 2 {
            self.prev_poc_msb - max_lsb
        } else {
            self.prev_poc_msb
        };
        let poc = msb + lsb;
        if hdr0.nal_ref {
            self.prev_poc_msb = msb;
            self.prev_poc_lsb = lsb;
        }

        // Take the reusable workspace (or build once).
        let mut fx = match self.work.take() {
            Some(mut f) => {
                f.recycle();
                f
            }
            None => Fx::new(&self.sps),
        };
        if self.trace {
            fx.trace = Some(Vec::new());
        }
        let mut dbf = (hdr0.dbf_idc, hdr0.aoff, hdr0.boff);
        for (si, (rbsp, idr, ref_idc)) in slices.iter().enumerate() {
            let hdr = parse_slice_header(rbsp, &self.sps, &self.pps, *idr, *ref_idc > 0)?;
            dbf = (hdr.dbf_idc, hdr.aoff, hdr.boff);
            fx.cur_slice = si as i32;
            let lists = build_ref_lists(&self.dpb, &hdr, poc, 1 << self.sps.log2_max_frame_num)?;
            decode_slice_cabac(&mut fx, &self.sps, &self.pps, &hdr, rbsp, &lists, poc)?;
        }

        if let Some(mut t) = fx.trace.take() {
            for row in 0..fx.mbh {
                let qps: Vec<alloc::string::String> = (0..fx.mbw).map(|x| alloc::format!("{}", fx.mbqp[row * fx.mbw + x])).collect();
                t.push(alloc::format!("QP {} {}", row, qps.join(",")));
            }
            self.trace_log = t;
        }
        // In-loop deblock is expensive at 1080p (~same cost as residual). Skip
        // when the stream didn't disable it but the picture is large — quality
        // softens slightly; realtime on a single cooperative core wins.
        // dbf_idc==1 means "off" already; we also force-skip above ~720p.
        let big = fx.w * fx.h > 1280 * 720;
        if dbf.0 != 1 && !big && !self.no_deblock {
            deblock_frame(&mut fx, dbf.1, dbf.2, &self.pps);
        }
        // Planes are already u8 reconstructed samples — move them into the DPB
        // pic. Replacement buffers are capacity-only (uninit): the next AU
        // overwrites every sample via MC/intra before any neighbour read.
        let (w, h) = (fx.w, fx.h);
        let (cw, ch) = (fx.cw, fx.ch);
        let fresh = |n: usize| {
            let mut v = alloc::vec::Vec::with_capacity(n);
            // SAFETY: next AU writes every index before any read.
            unsafe {
                v.set_len(n);
            }
            v
        };
        let y = core::mem::replace(&mut fx.y, fresh(w * h));
        let cb = core::mem::replace(&mut fx.cb, fresh(cw * ch));
        let cr = core::mem::replace(&mut fx.cr, fresh(cw * ch));
        let f = DecodedFrame { w, h, y, cb, cr };
        // Only reference pictures need motion fields in the DPB. Non-ref
        // frames leave mv[] on the workspace so the next recycle reuses them
        // without reallocating multi‑MB vectors.
        let (mv, refidx, refpoc) = if hdr0.nal_ref {
            (
                [core::mem::take(&mut fx.mv[0]), core::mem::take(&mut fx.mv[1])],
                [core::mem::take(&mut fx.refidx[0]), core::mem::take(&mut fx.refidx[1])],
                [core::mem::take(&mut fx.refpoc[0]), core::mem::take(&mut fx.refpoc[1])],
            )
        } else {
            (
                [Vec::new(), Vec::new()],
                [Vec::new(), Vec::new()],
                [Vec::new(), Vec::new()],
            )
        };
        let pic = Rc::new(Pic {
            f,
            poc,
            frame_num: hdr0.frame_num,
            mv,
            refidx,
            refpoc,
        });
        // Put workspace back for the next AU.
        self.work = Some(fx);
        // Reference handling: MMCO op-1 removals, then sliding window.
        if hdr0.nal_ref {
            for &(op, v) in &hdr0.mmco {
                if op == 1 {
                    let max_fn = 1u32 << self.sps.log2_max_frame_num;
                    let pic_num = hdr0.frame_num as i32 - (v as i32 + 1);
                    let pn = if pic_num < 0 { pic_num + max_fn as i32 } else { pic_num };
                    let target = if pn > hdr0.frame_num as i32 { pn - max_fn as i32 } else { pn };
                    self.dpb.retain(|p| frame_num_wrap(p.frame_num, hdr0.frame_num, max_fn) != target);
                }
            }
            self.dpb.push(pic.clone());
            let max_refs = self.sps.max_num_ref_frames.max(1) as usize;
            while self.dpb.len() > max_refs {
                // Sliding window: drop the smallest FrameNumWrap.
                let max_fn = 1u32 << self.sps.log2_max_frame_num;
                let (idx, _) = self
                    .dpb
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, p)| frame_num_wrap(p.frame_num, hdr0.frame_num, max_fn))
                    .unwrap();
                self.dpb.remove(idx);
            }
        }
        Ok(pic)
    }
}


// --- slice decode -------------------------------------------------------------

const BLK_XY: [(usize, usize); 16] = [
    (0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (3, 0), (2, 1), (3, 1),
    (0, 2), (1, 2), (0, 3), (1, 3), (2, 2), (3, 2), (2, 3), (3, 3),
];

/// B mb_type info (index from the binarization tree): partition kind
/// (0=direct16x16, 1=16x16, 2=16x8, 3=8x16, 4=8x8) + per-partition list masks.
const B_TYPE: [(u8, u8, u8); 23] = [
    (0, 3, 0), (1, 1, 0), (1, 2, 0), (1, 3, 0),
    (2, 1, 1), (3, 1, 1), (2, 2, 2), (3, 2, 2),
    (2, 1, 2), (3, 1, 2), (2, 2, 1), (3, 2, 1),
    (2, 1, 3), (3, 1, 3), (2, 2, 3), (3, 2, 3),
    (2, 3, 1), (3, 3, 1), (2, 3, 2), (3, 3, 2),
    (2, 3, 3), (3, 3, 3), (4, 0, 0),
];
/// B sub_mb_type info: (shape 0=direct 1=8x8 2=8x4 3=4x8 4=4x4, list mask).
const B_SUB: [(u8, u8); 13] = [
    (0, 3), (1, 1), (1, 2), (1, 3), (2, 1), (3, 1), (2, 2), (3, 2), (2, 3), (3, 3), (4, 1), (4, 2), (4, 3),
];

/// Per-slice decode context (immutable per slice).
struct Sl<'a> {
    sps: &'a Sps,
    pps: &'a Pps,
    hdr: &'a Hdr,
    lists: &'a [Vec<Rc<Pic>>; 2],
    poc: i32,
}

/// Mutable per-slice state.
struct St {
    qp: i32,
    last_qp_diff: bool,
}

fn decode_slice_cabac(fx: &mut Fx, sps: &Sps, pps: &Pps, hdr: &Hdr, rbsp: &[u8], lists: &[Vec<Rc<Pic>>; 2], poc: i32) -> Result<(), &'static str> {
    let s = Sl { sps, pps, hdr, lists, poc };
    let init_set = if hdr.stype == 2 { None } else { Some(hdr.cabac_init_idc) };
    let mut cb = Cabac::new(&rbsp[hdr.data_byte..], hdr.qp.clamp(0, 51), init_set)?;
    let mut st = St { qp: hdr.qp, last_qp_diff: false };
    let total = fx.mbw * fx.mbh;
    let mut mb = hdr.first_mb;
    // NOTE: do **not** call `shell::upkeep()` here. Decode is often entered
    // from `pump_video` while it holds the VIDEO lock; upkeep → pump_video
    // re-enters that lock and deadlocks the OS. Frame pacing + display
    // downscale keep the shell responsive instead.
    loop {
        if mb >= total {
            return Err("h264 cabac: mb address overflow");
        }
        decode_mb(fx, &mut cb, &s, &mut st, mb)?;
        if cb.terminate() != 0 {
            break;
        }
        mb += 1;
    }
    Ok(())
}

// --- macroblock layer ----------------------------------------------------------

#[allow(clippy::too_many_lines)]
fn decode_mb(fx: &mut Fx, cb: &mut Cabac, s: &Sl, st: &mut St, mb: usize) -> Result<(), &'static str> {
    let mbw = fx.mbw;
    let (mb_x, mb_y) = (mb % mbw, mb / mbw);
    fx.mbslice[mb] = fx.cur_slice;
    fx.mbqp[mb] = st.qp;
    let is_p = s.hdr.stype == 0;
    let is_b = s.hdr.stype == 1;

    // mb_skip_flag.
    if is_p || is_b {
        let a = fx.mb_a(mb).map(|m| !fx.mbskip[m]).unwrap_or(false) as usize;
        let b = fx.mb_b(mb).map(|m| !fx.mbskip[m]).unwrap_or(false) as usize;
        let base = if is_b { 24 } else { 11 };
        if cb.decision(base + a + b) != 0 {
            fx.mbskip[mb] = true;
            st.last_qp_diff = false;
            if is_p {
                let mv = fx.skip_mv(mb_x, mb_y);
                let rp = s.lists[0].first().ok_or("h264: P_Skip without reference")?.poc;
                fx.store_mv(0, mb_x * 4, mb_y * 4, 4, 4, mv, 0, rp);
                mc_partition(fx, s, mb_x * 16, mb_y * 16, 16, 16, 1, 0, mv, 0, [0, 0]);
            } else {
                fx.mbdir[mb] = true;
                let subs = [true; 4];
                pred_direct(fx, s, mb_x, mb_y, &subs)?;
                mc_direct(fx, s, mb_x, mb_y, &subs);
            }
            mark_mb_decoded(fx, mb_x, mb_y);
            return Ok(());
        }
    }

    // mb_type.
    let mut intra_type: Option<u32> = None;
    let mut p_kind = 0u8; // P: 1=16x16 2=16x8 3=8x16 4=8x8
    let mut b_kind = (0u8, 0u8, 0u8);
    if is_b {
        let a = fx.mb_a(mb).map(|m| !(fx.mbskip[m] || mb_all_direct(fx, m))).unwrap_or(false) as usize;
        let b = fx.mb_b(mb).map(|m| !(fx.mbskip[m] || mb_all_direct(fx, m))).unwrap_or(false) as usize;
        let idx = if cb.decision(27 + a + b) == 0 {
            0
        } else if cb.decision(27 + 3) == 0 {
            1 + cb.decision(27 + 5)
        } else {
            let mut bits = cb.decision(27 + 4) << 3;
            bits += cb.decision(27 + 5) << 2;
            bits += cb.decision(27 + 5) << 1;
            bits += cb.decision(27 + 5);
            if bits < 8 {
                bits + 3
            } else if bits == 13 {
                intra_type = Some(cabac_intra_mb_type(cb, 32, false, 0));
                0
            } else if bits == 14 {
                11
            } else if bits == 15 {
                22
            } else {
                (bits << 1) + cb.decision(27 + 5) - 4
            }
        };
        if intra_type.is_none() {
            b_kind = B_TYPE[idx as usize];
        }
    } else if is_p {
        if cb.decision(14) == 0 {
            p_kind = if cb.decision(15) == 0 {
                if cb.decision(16) != 0 { 4 } else { 1 }
            } else if cb.decision(17) != 0 {
                2
            } else {
                3
            };
        } else {
            intra_type = Some(cabac_intra_mb_type(cb, 17, false, 0));
        }
    } else {
        let a = fx.mb_a(mb).map(|m| fx.mbintra[m] && !fx.mbinxn[m]).unwrap_or(false) as usize;
        let b = fx.mb_b(mb).map(|m| fx.mbintra[m] && !fx.mbinxn[m]).unwrap_or(false) as usize;
        intra_type = Some(cabac_intra_mb_type(cb, 3, true, a + b));
    }

    if let Some(t) = intra_type {
        decode_intra_mb(fx, cb, s, st, mb_x, mb_y, t)?;
        mark_mb_decoded(fx, mb_x, mb_y);
        return Ok(());
    }

    // Inter prediction structure.
    let nlists = if is_b { 2 } else { 1 };
    let mut sub_types = [0usize; 4]; // P: 0..3; B: 0..12
    let mut direct_subs = [false; 4];
    let mut has_direct = false;
    let (kind, d0, d1) = if is_b { b_kind } else { (if p_kind == 4 { 4 } else { p_kind }, 1, 1) };
    if kind == 4 {
        // 8x8 sub-macroblock types.
        for st8 in sub_types.iter_mut() {
            if is_b {
                *st8 = decode_b_sub_type(cb);
            } else {
                *st8 = decode_p_sub_type(cb);
            }
        }
        if is_b {
            for i in 0..4 {
                if B_SUB[sub_types[i]].0 == 0 {
                    direct_subs[i] = true;
                    has_direct = true;
                }
            }
        }
    }
    if kind == 0 {
        // B_Direct_16x16.
        let mbi = mb_y * fx.mbw + mb_x;
        fx.mbdir[mbi] = true;
        direct_subs = [true; 4];
        has_direct = true;
    }
    if has_direct {
        pred_direct(fx, s, mb_x, mb_y, &direct_subs)?;
    }

    // Partition geometry: (bx4 off, by4 off, w4, h4, mvp kind) per partition.
    let parts: Vec<(usize, usize, usize, usize, u8, u8)> = match kind {
        0 => Vec::new(),
        1 => vec![(0, 0, 4, 4, 0, d0)],
        2 => vec![(0, 0, 4, 2, 1, d0), (0, 2, 4, 2, 2, d1)],
        3 => vec![(0, 0, 2, 4, 3, d0), (2, 0, 2, 4, 4, d1)],
        _ => Vec::new(), // 8x8 handled separately
    };

    // Reference indices (wire order: list-major, partition-inner).
    let mut prefs = [[0i8; 4]; 2];
    if kind == 4 {
        for list in 0..nlists {
            for i in 0..4 {
                if direct_subs[i] {
                    continue;
                }
                let dmask = if is_b { B_SUB[sub_types[i]].1 } else { 1 };
                if dmask & (1 << list) == 0 {
                    prefs[list][i] = -1;
                    // The list is unused for this 8x8: its cells become
                    // available (ref -1, mv 0) right away (LIST_NOT_USED fill).
                    fx.store_mv(list, mb_x * 4 + (i & 1) * 2, mb_y * 4 + (i >> 1) * 2, 2, 2, [0, 0], -1, i32::MIN);
                    continue;
                }
                prefs[list][i] = read_ref(fx, cb, s, list, mb_x * 4 + (i & 1) * 2, mb_y * 4 + (i >> 1) * 2)?;
                // Make refs visible to later ctx reads (MV not yet decoded).
                let rp = ref_poc(s, list, prefs[list][i])?;
                fx.store_ref_only(list, mb_x * 4 + (i & 1) * 2, mb_y * 4 + (i >> 1) * 2, 2, 2, prefs[list][i], rp);
            }
        }
    } else {
        for list in 0..nlists {
            for (pi, &(ox4, oy4, w4, h4, _, dm)) in parts.iter().enumerate() {
                if dm & (1 << list) == 0 {
                    prefs[list][pi] = -1;
                    fx.store_mv(list, mb_x * 4 + ox4, mb_y * 4 + oy4, w4, h4, [0, 0], -1, i32::MIN);
                    continue;
                }
                prefs[list][pi] = read_ref(fx, cb, s, list, mb_x * 4 + ox4, mb_y * 4 + oy4)?;
                let rp = ref_poc(s, list, prefs[list][pi])?;
                fx.store_ref_only(list, mb_x * 4 + ox4, mb_y * 4 + oy4, w4, h4, prefs[list][pi], rp);
            }
        }
    }

    // Motion vector differences + prediction + store (wire order matches refs).
    if kind == 4 {
        for list in 0..nlists {
            for i in 0..4 {
                if direct_subs[i] {
                    continue;
                }
                let (shape, dmask) = if is_b { B_SUB[sub_types[i]] } else { (match sub_types[i] { 0 => 1, 1 => 2, 2 => 3, _ => 4 }, 1u8) };
                if dmask & (1 << list) == 0 {
                    continue;
                }
                let (pw4, ph4, nparts) = match shape {
                    1 => (2usize, 2usize, 1usize),
                    2 => (2, 1, 2),
                    3 => (1, 2, 2),
                    _ => (1, 1, 4),
                };
                let base_x4 = mb_x * 4 + (i & 1) * 2;
                let base_y4 = mb_y * 4 + (i >> 1) * 2;
                for j in 0..nparts {
                    let (ox, oy) = match shape {
                        1 => (0, 0),
                        2 => (0, j),
                        3 => (j, 0),
                        _ => (j & 1, j >> 1),
                    };
                    let bx4 = base_x4 + ox * pw4.min(2);
                    let by4 = base_y4 + oy * ph4.min(2);
                    let ridx = prefs[list][i];
                    let pred = fx.pred_mv(list, bx4 as i32, by4 as i32, pw4 as i32, ridx, 0);
                    let (mv, mvd) = read_mvd_pair(fx, cb, list, bx4, by4, pred)?;
                    let rp = ref_poc(s, list, ridx)?;
                    fx.store_mv(list, bx4, by4, pw4, ph4, mv, ridx, rp);
                    fx.store_mvd(list, bx4, by4, pw4, ph4, mvd);
                }
            }
        }
    } else {
        for list in 0..nlists {
            for (pi, &(ox4, oy4, w4, h4, kindp, dm)) in parts.iter().enumerate() {
                if dm & (1 << list) == 0 {
                    continue;
                }
                let bx4 = mb_x * 4 + ox4;
                let by4 = mb_y * 4 + oy4;
                let ridx = prefs[list][pi];
                let pred = fx.pred_mv(list, bx4 as i32, by4 as i32, w4 as i32, ridx, kindp);
                let (mv, mvd) = read_mvd_pair(fx, cb, list, bx4, by4, pred)?;
                let rp = ref_poc(s, list, ridx)?;
                fx.store_mv(list, bx4, by4, w4, h4, mv, ridx, rp);
                fx.store_mvd(list, bx4, by4, w4, h4, mvd);
            }
        }
    }

    // Motion compensation (all partitions have final MVs now).
    if has_direct {
        mc_direct(fx, s, mb_x, mb_y, &direct_subs);
    }
    if kind == 4 {
        for i in 0..4 {
            if direct_subs[i] {
                continue;
            }
            let (shape, dmask) = if is_b { B_SUB[sub_types[i]] } else { (match sub_types[i] { 0 => 1, 1 => 2, 2 => 3, _ => 4 }, 1u8) };
            let (pw, ph, nparts) = match shape {
                1 => (8usize, 8usize, 1usize),
                2 => (8, 4, 2),
                3 => (4, 8, 2),
                _ => (4, 4, 4),
            };
            let bx = mb_x * 16 + (i & 1) * 8;
            let by = mb_y * 16 + (i >> 1) * 8;
            for j in 0..nparts {
                let (ox, oy) = match shape {
                    1 => (0, 0),
                    2 => (0, j * 4),
                    3 => (j * 4, 0),
                    _ => ((j & 1) * 4, (j >> 1) * 4),
                };
                let (px, py) = (bx + ox, by + oy);
                let i4 = (py / 4) * fx.ny4 + px / 4;
                let mv0 = fx.mv[0][i4];
                let mv1 = fx.mv[1][i4];
                mc_partition(fx, s, px, py, pw, ph, dmask, prefs[0][i].max(0), mv0, prefs[1][i].max(0), mv1);
            }
        }
    } else {
        for (pi, &(ox4, oy4, w4, h4, _, dm)) in parts.iter().enumerate() {
            let (px, py) = (mb_x * 16 + ox4 * 4, mb_y * 16 + oy4 * 4);
            let i4 = (py / 4) * fx.ny4 + px / 4;
            mc_partition(fx, s, px, py, w4 * 4, h4 * 4, dm, prefs[0][pi].max(0), fx.mv[0][i4], prefs[1][pi].max(0), fx.mv[1][i4]);
        }
    }

    let _ = d1;
    // Coded block pattern.
    let cbp_l = decode_cbp_luma(fx, cb, mb);
    let cbp_c = decode_cbp_chroma(fx, cb, mb);
    fx.mbcbp_l[mb] = cbp_l;
    fx.mbcbp_c[mb] = cbp_c;

    // transform_size_8x8_flag (inter): after cbp, if luma residual present.
    let mut t8 = false;
    if s.pps.transform_8x8_mode && cbp_l != 0 && dct8x8_allowed_inter(s, kind, is_b, &sub_types, &direct_subs) {
        let a = fx.mb_a(mb).map(|m| fx.mbt8[m]).unwrap_or(false) as usize;
        let b = fx.mb_b(mb).map(|m| fx.mbt8[m]).unwrap_or(false) as usize;
        t8 = cb.decision(399 + a + b) != 0;
    }
    fx.mbt8[mb] = t8;

    // mb_qp_delta + residuals.
    if cbp_l != 0 || cbp_c != 0 {
        read_qp_delta(cb, st)?;
    } else {
        st.last_qp_diff = false;
    }
    fx.mbqp[mb] = st.qp;
    decode_residual_luma_inter(fx, cb, s, st, mb_x, mb_y, cbp_l, t8)?;
    decode_residual_chroma(fx, cb, s, st, mb_x, mb_y, cbp_c, false)?;
    mark_mb_decoded(fx, mb_x, mb_y);
    Ok(())
}

fn mb_all_direct(fx: &Fx, mb: usize) -> bool {
    fx.mbdir[mb]
}

fn decode_p_sub_type(cb: &mut Cabac) -> usize {
    if cb.decision(21) != 0 {
        0
    } else if cb.decision(22) == 0 {
        1
    } else if cb.decision(23) != 0 {
        2
    } else {
        3
    }
}

fn decode_b_sub_type(cb: &mut Cabac) -> usize {
    if cb.decision(36) == 0 {
        return 0;
    }
    if cb.decision(37) == 0 {
        return 1 + cb.decision(39) as usize;
    }
    let mut t = 3usize;
    if cb.decision(38) != 0 {
        if cb.decision(39) != 0 {
            return 11 + cb.decision(39) as usize;
        }
        t += 4;
    }
    t += 2 * cb.decision(39) as usize;
    t += cb.decision(39) as usize;
    t
}

fn ref_poc(s: &Sl, list: usize, ridx: i8) -> Result<i32, &'static str> {
    if ridx < 0 {
        return Ok(i32::MIN);
    }
    s.lists[list].get(ridx as usize).map(|p| p.poc).ok_or("h264: ref_idx out of range")
}

fn read_ref(fx: &Fx, cb: &mut Cabac, s: &Sl, list: usize, bx4: usize, by4: usize) -> Result<i8, &'static str> {
    if s.hdr.nref[list] <= 1 {
        return Ok(0);
    }
    let is_b = s.hdr.stype == 1;
    // The ref ctx uses the neighbouring cells' reference indices as *parsed so
    // far* — including same-MB partitions whose MV is not yet decoded (unlike
    // motion-vector prediction, which needs final MVs). Slice availability only.
    let ref_at = |x: i32, y: i32| -> i8 {
        if x < 0 || y < 0 {
            return -1;
        }
        let mb = (y as usize / 4) * fx.mbw + x as usize / 4;
        if !fx.sl(mb) {
            return -1;
        }
        fx.refidx[list][y as usize * fx.ny4 + x as usize]
    };
    let ra = ref_at(bx4 as i32 - 1, by4 as i32);
    let rb = ref_at(bx4 as i32, by4 as i32 - 1);
    let g = fx.pix_gen;
    let l_dir = is_b && bx4 > 0 && fx.dirf[by4 * fx.ny4 + bx4 - 1] == g;
    let t_dir = is_b && by4 > 0 && fx.dirf[(by4 - 1) * fx.ny4 + bx4] == g;
    let mut ctx = 0usize;
    if ra > 0 && !l_dir {
        ctx += 1;
    }
    if rb > 0 && !t_dir {
        ctx += 2;
    }
    let r = cabac_ref_idx(cb, ctx)?;
    if (r as usize) >= s.hdr.nref[list] {
        return Err("h264: ref_idx >= active count");
    }
    Ok(r)
}

fn read_mvd_pair(fx: &Fx, cb: &mut Cabac, list: usize, bx4: usize, by4: usize, pred: [i16; 2]) -> Result<([i16; 2], [u8; 2]), &'static str> {
    let amvd = |comp: usize| -> i32 {
        let a = if bx4 > 0 && fx.sl((by4 / 4) * fx.mbw + (bx4 - 1) / 4) { fx.mvd[list][by4 * fx.ny4 + bx4 - 1][comp] as i32 } else { 0 };
        let b = if by4 > 0 && fx.sl(((by4 - 1) / 4) * fx.mbw + bx4 / 4) { fx.mvd[list][(by4 - 1) * fx.ny4 + bx4][comp] as i32 } else { 0 };
        a + b
    };
    let (dx, cx) = cabac_mvd(cb, 40, amvd(0))?;
    let (dy, cy) = cabac_mvd(cb, 47, amvd(1))?;
    let mv = [(pred[0] as i32 + dx).clamp(-32768, 32767) as i16, (pred[1] as i32 + dy).clamp(-32768, 32767) as i16];
    Ok((mv, [cx, cy]))
}

fn dct8x8_allowed_inter(s: &Sl, kind: u8, is_b: bool, sub_types: &[usize; 4], direct_subs: &[bool; 4]) -> bool {
    match kind {
        0 => s.sps.direct_8x8_inference, // B_Direct_16x16
        1 | 2 | 3 => true,
        4 => {
            for i in 0..4 {
                if direct_subs[i] {
                    if !s.sps.direct_8x8_inference {
                        return false;
                    }
                    continue;
                }
                let shape = if is_b { B_SUB[sub_types[i]].0 } else { match sub_types[i] { 0 => 1, 1 => 2, 2 => 3, _ => 4 } };
                if shape != 1 {
                    return false; // only whole-8x8 sub-partitions allow 8x8 DCT
                }
            }
            true
        }
        _ => false,
    }
}

fn read_qp_delta(cb: &mut Cabac, st: &mut St) -> Result<(), &'static str> {
    if cb.decision(60 + st.last_qp_diff as usize) == 0 {
        st.last_qp_diff = false;
        return Ok(());
    }
    let mut val = 1i32;
    let mut ctx = 62usize;
    while cb.decision(ctx) != 0 {
        ctx = 63;
        val += 1;
        if val > 104 {
            return Err("h264 cabac: qp_delta runaway");
        }
    }
    let delta = if val & 1 != 0 { (val + 1) >> 1 } else { -((val + 1) >> 1) };
    st.qp += delta;
    if st.qp > 51 {
        st.qp -= 52;
    } else if st.qp < 0 {
        st.qp += 52;
    }
    st.last_qp_diff = true;
    Ok(())
}

fn decode_cbp_luma(fx: &Fx, cb: &mut Cabac, mb: usize) -> u8 {
    // Neighbour cbp bits: unavailable → 0xF (no ctx increment), per FFmpeg's
    // 0x7CF/0x00F unavailable fill.
    let cbp_a = fx.mb_a(mb).map(|m| fx.mbcbp_l[m]).unwrap_or(0x0f) as u32;
    let cbp_b = fx.mb_b(mb).map(|m| fx.mbcbp_l[m]).unwrap_or(0x0f) as u32;
    let mut cbp = 0u32;
    let mut ctx = ((cbp_a & 0x02) == 0) as usize + 2 * ((cbp_b & 0x04) == 0) as usize;
    cbp += cb.decision(73 + ctx);
    ctx = ((cbp & 0x01) == 0) as usize + 2 * ((cbp_b & 0x08) == 0) as usize;
    cbp += cb.decision(73 + ctx) << 1;
    ctx = ((cbp_a & 0x08) == 0) as usize + 2 * ((cbp & 0x01) == 0) as usize;
    cbp += cb.decision(73 + ctx) << 2;
    ctx = ((cbp & 0x04) == 0) as usize + 2 * ((cbp & 0x02) == 0) as usize;
    cbp += cb.decision(73 + ctx) << 3;
    cbp as u8
}

fn decode_cbp_chroma(fx: &Fx, cb: &mut Cabac, mb: usize) -> u8 {
    let cbp_a = fx.mb_a(mb).map(|m| fx.mbcbp_c[m]).unwrap_or(0);
    let cbp_b = fx.mb_b(mb).map(|m| fx.mbcbp_c[m]).unwrap_or(0);
    let mut ctx = 0usize;
    if cbp_a > 0 {
        ctx += 1;
    }
    if cbp_b > 0 {
        ctx += 2;
    }
    if cb.decision(77 + ctx) == 0 {
        return 0;
    }
    ctx = 4;
    if cbp_a == 2 {
        ctx += 1;
    }
    if cbp_b == 2 {
        ctx += 2;
    }
    1 + cb.decision(77 + ctx) as u8
}

/// Mark the MB's 4×4 cells motion-decoded and its luma/chroma available for
/// subsequent intra neighbour queries. Also zeroes nnz for any 4×4 that the
/// residual path didn't touch (Skip / cbp=0) so neighbour cbf contexts stay
/// correct without a full-plane clear each AU.
fn mark_mb_decoded(fx: &mut Fx, mb_x: usize, mb_y: usize) {
    let g = fx.pix_gen;
    let mb = mb_y * fx.mbw + mb_x;
    for yy in 0..4 {
        for xx in 0..4 {
            let i = (mb_y * 4 + yy) * fx.ny4 + mb_x * 4 + xx;
            // Intra / residual paths already set mvok when they store MVs;
            // force-available so intra neighbours count as refidx=-1.
            if fx.mvok[0][i] != g {
                fx.mvok[0][i] = g;
                fx.refidx[0][i] = -1;
                fx.mv[0][i] = [0, 0];
                fx.refpoc[0][i] = i32::MIN;
            }
            if fx.mvok[1][i] != g {
                fx.mvok[1][i] = g;
                fx.refidx[1][i] = -1;
                fx.mv[1][i] = [0, 0];
                // refpoc must clear too: the deblock bS (`bs_inter2`) infers
                // "list used" from refpoc != MIN, and the recycled workspace
                // still holds the previous frame's value (a P slice never
                // writes list 1 — its stale refpoc made bS see two mvs).
                fx.refpoc[1][i] = i32::MIN;
            }
            fx.decy[i] = g;
        }
    }
    if let Some(s) = fx.decu.get_mut(mb) {
        *s = g;
    }
    if let Some(s) = fx.decv.get_mut(mb) {
        *s = g;
    }
}

// --- neighbour gathering (pixel level, mirrors the baseline decoder) -----------

impl Fx {
    #[inline]
    fn mbp(&self, x: usize, y: usize) -> usize {
        (y / 16) * self.mbw + x / 16
    }
    #[inline]
    fn mbpc(&self, x: usize, y: usize) -> usize {
        (y / 8) * self.mbw + x / 8
    }

    fn gather4(&self, px: usize, py: usize) -> ([i32; 8], [i32; 4], i32, bool, bool, bool) {
        let (pl, w) = (&self.y, self.w);
        let mut top = [0i32; 8];
        let mut left = [0i32; 4];
        let at = py > 0 && self.y_dec_xy(px, py - 1) && self.sl(self.mbp(px, py - 1));
        let al = px > 0 && self.y_dec_xy(px - 1, py) && self.sl(self.mbp(px - 1, py));
        let corner = if px > 0 && py > 0 && self.y_dec_xy(px - 1, py - 1) && self.sl(self.mbp(px - 1, py - 1)) {
            pl[(py - 1) * w + px - 1] as i32
        } else {
            0
        };
        let atr = py > 0 && (px + 4) < w && self.y_dec_xy(px + 4, py - 1) && self.sl(self.mbp(px + 4, py - 1));
        if at {
            for k in 0..4 {
                top[k] = pl[(py - 1) * w + px + k] as i32;
            }
        }
        if atr {
            for k in 4..8 {
                let xx = px + k;
                top[k] = if xx < w {
                    pl[(py - 1) * w + xx] as i32
                } else {
                    pl[(py - 1) * w + w - 1] as i32
                };
            }
        }
        if al {
            for k in 0..4 {
                left[k] = pl[(py + k) * w + px - 1] as i32;
            }
        }
        (top, left, corner, at, al, atr)
    }

    /// Reference gathering for an 8x8 intra block at luma (px, py).
    fn gather8(&self, px: usize, py: usize) -> ([i32; 16], [i32; 8], i32, bool, bool, bool, bool) {
        let (pl, w) = (&self.y, self.w);
        let mut top = [0i32; 16];
        let mut left = [0i32; 8];
        let at = py > 0 && self.y_dec_xy(px, py - 1) && self.sl(self.mbp(px, py - 1));
        let al = px > 0 && self.y_dec_xy(px - 1, py) && self.sl(self.mbp(px - 1, py));
        let atl = px > 0 && py > 0 && self.y_dec_xy(px - 1, py - 1) && self.sl(self.mbp(px - 1, py - 1));
        let corner = if atl {
            pl[(py - 1) * w + px - 1] as i32
        } else {
            0
        };
        let atr = py > 0 && (px + 8) < w && self.y_dec_xy(px + 8, py - 1) && self.sl(self.mbp(px + 8, py - 1));
        if at {
            for k in 0..8 {
                top[k] = pl[(py - 1) * w + px + k] as i32;
            }
        }
        if atr {
            for k in 8..16 {
                let xx = px + k;
                top[k] = if xx < w {
                    pl[(py - 1) * w + xx] as i32
                } else {
                    pl[(py - 1) * w + w - 1] as i32
                };
            }
        }
        if al {
            for k in 0..8 {
                left[k] = pl[(py + k) * w + px - 1] as i32;
            }
        }
        (top, left, corner, at, al, atl, atr)
    }

    fn chroma_pred(&self, pl: usize, cbx: usize, cby: usize, mode: u8) -> [i32; 64] {
        let plane = if pl == 0 { &self.cb } else { &self.cr };
        let mut top = [0i32; 8];
        let mut left = [0i32; 8];
        // Neighbour MBs (chroma is marked per-MB once the whole 8×8 is written).
        let at = cby > 0 && self.c_dec(self.mbpc(cbx, cby - 1)) && self.sl(self.mbpc(cbx, cby - 1));
        let al = cbx > 0 && self.c_dec(self.mbpc(cbx - 1, cby)) && self.sl(self.mbpc(cbx - 1, cby));
        let corner = if cbx > 0 && cby > 0 && self.sl(self.mbpc(cbx - 1, cby - 1)) {
            plane[(cby - 1) * self.cw + cbx - 1] as i32
        } else {
            0
        };
        if at {
            for k in 0..8 {
                top[k] = plane[(cby - 1) * self.cw + cbx + k] as i32;
            }
        }
        if al {
            for k in 0..8 {
                left[k] = plane[(cby + k) * self.cw + cbx - 1] as i32;
            }
        }
        intra::intra_chroma(mode, &top, &left, corner, at, al)
    }
}

// --- coded_block_flag contexts --------------------------------------------------

/// coded_block_flag ctxIdx for `cat` (0..4) per §9.3.3.1.1.9; `blk` addresses
/// the 4x4 (luma) / 2x2-grid (chroma AC) block; `cur_intra` drives the
/// unavailable-neighbour rule.
fn cbf_ctx(fx: &Fx, cat: usize, mb: usize, bxx: usize, byy: usize, plane: usize, cur_intra: bool) -> usize {
    const BASE: [usize; 5] = [85, 89, 93, 97, 101];
    let unavail = cur_intra as i32;
    let (nza, nzb) = match cat {
        0 => {
            let a = fx.mb_a(mb).map(|m| if fx.mbi16[m] { (fx.mbdcf[m] & 1) as i32 } else { 0 }).unwrap_or(unavail);
            let b = fx.mb_b(mb).map(|m| if fx.mbi16[m] { (fx.mbdcf[m] & 1) as i32 } else { 0 }).unwrap_or(unavail);
            (a, b)
        }
        3 => {
            let bit = 1 << (1 + plane);
            let a = fx.mb_a(mb).map(|m| ((fx.mbdcf[m] & bit) != 0) as i32).unwrap_or(unavail);
            let b = fx.mb_b(mb).map(|m| ((fx.mbdcf[m] & bit) != 0) as i32).unwrap_or(unavail);
            (a, b)
        }
        4 => {
            let nnz = if plane == 0 { &fx.nnz_u } else { &fx.nnz_v };
            let a = if bxx > 0 && fx.sl((byy / 2) * fx.mbw + (bxx - 1) / 2) { (nnz[byy * fx.nc2 + bxx - 1] > 0) as i32 } else { unavail };
            let b = if byy > 0 && fx.sl(((byy - 1) / 2) * fx.mbw + bxx / 2) { (nnz[(byy - 1) * fx.nc2 + bxx] > 0) as i32 } else { unavail };
            (a, b)
        }
        _ => {
            // cat 1/2: luma 4x4 grid.
            let a = if bxx > 0 && fx.sl((byy / 4) * fx.mbw + (bxx - 1) / 4) { (fx.nnz_y[byy * fx.ny4 + bxx - 1] > 0) as i32 } else { unavail };
            let b = if byy > 0 && fx.sl(((byy - 1) / 4) * fx.mbw + bxx / 4) { (fx.nnz_y[(byy - 1) * fx.ny4 + bxx] > 0) as i32 } else { unavail };
            (a, b)
        }
    };
    BASE[cat] + (nza > 0) as usize + 2 * (nzb > 0) as usize
}

// --- residual decode + reconstruction -------------------------------------------

/// Luma residual + reconstruction for inter MBs (and shared 4x4/8x8 add).
fn decode_residual_luma_inter(fx: &mut Fx, cb: &mut Cabac, s: &Sl, st: &St, mb_x: usize, mb_y: usize, cbp_l: u8, t8: bool) -> Result<(), &'static str> {
    let _ = s;
    let mb = mb_y * fx.mbw + mb_x;
    let (bx, by, w) = (mb_x * 16, mb_y * 16, fx.w);
    for i8x8 in 0..4 {
        let (ox8, oy8) = ((i8x8 & 1) * 8, (i8x8 >> 1) * 8);
        if cbp_l & (1 << i8x8) == 0 {
            for b in 0..4 {
                let bxx = mb_x * 4 + (i8x8 & 1) * 2 + (b & 1);
                let byy = mb_y * 4 + (i8x8 >> 1) * 2 + (b >> 1);
                fx.nnz_y[byy * fx.ny4 + bxx] = 0;
            }
            continue;
        }
        if t8 {
            let mut sc = [0i32; 64];
            let cnt = cabac_residual(cb, 5, 64, &mut sc);
            for b in 0..4 {
                let bxx = mb_x * 4 + (i8x8 & 1) * 2 + (b & 1);
                let byy = mb_y * 4 + (i8x8 >> 1) * 2 + (b >> 1);
                fx.nnz_y[byy * fx.ny4 + bxx] = cnt as i32;
            }
            let mut blk = [0i32; 64];
            for (si, &lv) in sc.iter().enumerate() {
                if lv != 0 {
                    let p = transform::ZIGZAG8[si];
                    blk[p] = transform::dequant8(lv, st.qp, p);
                }
            }
            transform::idct8_residual(&mut blk);
            for yy in 0..8 {
                for xx in 0..8 {
                    let idx = (by + oy8 + yy) * w + bx + ox8 + xx;
                    fx.y[idx] = clip8(fx.y[idx] as i32 + blk[yy * 8 + xx]);
                }
            }
        } else {
            for b in 0..4 {
                let bxx = mb_x * 4 + (i8x8 & 1) * 2 + (b & 1);
                let byy = mb_y * 4 + (i8x8 >> 1) * 2 + (b >> 1);
                let ctx = cbf_ctx(fx, 2, mb, bxx, byy, 0, false);
                if cb.decision(ctx) == 0 {
                    fx.nnz_y[byy * fx.ny4 + bxx] = 0;
                    continue;
                }
                let mut sc = [0i32; 16];
                let cnt = cabac_residual(cb, 2, 16, &mut sc);
                fx.nnz_y[byy * fx.ny4 + bxx] = cnt as i32;
                let mut blk = transform::inverse_scan_4x4(&sc);
                transform::dequant_4x4(&mut blk, st.qp as u32, false);
                transform::idct_4x4(&mut blk);
                let (px, py) = (bx + (bxx % 4) * 4, by + (byy % 4) * 4);
                for yy in 0..4 {
                    for xx in 0..4 {
                        let idx = (py + yy) * w + px + xx;
                        fx.y[idx] = clip8(fx.y[idx] as i32 + blk[yy * 4 + xx]);
                    }
                }
            }
        }
    }
    Ok(())
}

/// Chroma residual + reconstruction (DC + AC add on top of the MC/intra
/// prediction already in the planes).
fn decode_residual_chroma(fx: &mut Fx, cb: &mut Cabac, s: &Sl, st: &St, mb_x: usize, mb_y: usize, cbp_c: u8, cur_intra: bool) -> Result<(), &'static str> {
    let mb = mb_y * fx.mbw + mb_x;
    let cw = fx.cw;
    let (cbx, cby) = (mb_x * 8, mb_y * 8);
    let offs = [s.pps.chroma_qp_index_offset, s.pps.second_chroma_qp_index_offset];
    if cbp_c == 0 {
        for c4 in 0..4 {
            let i = (mb_y * 2 + c4 / 2) * fx.nc2 + mb_x * 2 + (c4 % 2);
            fx.nnz_u[i] = 0;
            fx.nnz_v[i] = 0;
        }
        return Ok(());
    }
    // DC blocks (both planes) come first, then all AC blocks per plane.
    let mut dcq = [[0i32; 4]; 2];
    for pl in 0..2 {
        let ctx = cbf_ctx(fx, 3, mb, 0, 0, pl, cur_intra);
        if cb.decision(ctx) != 0 {
            let mut sc = [0i32; 4];
            let _ = cabac_residual(cb, 3, 4, &mut sc);
            dcq[pl] = sc;
            fx.mbdcf[mb] |= 1 << (1 + pl);
        }
    }
    for (pl, dq) in dcq.iter_mut().enumerate() {
        let cqp = qpc_tab((st.qp + offs[pl]).clamp(0, 51));
        transform::chroma_dc_transform(dq, cqp as u32);
    }
    for pl in 0..2 {
        let cqp = qpc_tab((st.qp + offs[pl]).clamp(0, 51)) as u32;
        for c4 in 0..4 {
            let cxx = mb_x * 2 + (c4 % 2);
            let cyy = mb_y * 2 + c4 / 2;
            let mut blk = [0i32; 16];
            let mut cnt = 0usize;
            if cbp_c == 2 {
                let ctx = cbf_ctx(fx, 4, mb, cxx, cyy, pl, cur_intra);
                if cb.decision(ctx) != 0 {
                    let mut sc = [0i32; 16];
                    cnt = cabac_residual(cb, 4, 15, &mut sc[1..]);
                    blk = transform::inverse_scan_4x4(&sc);
                }
            }
            {
                let nnz = if pl == 0 { &mut fx.nnz_u } else { &mut fx.nnz_v };
                nnz[cyy * fx.nc2 + cxx] = cnt as i32;
            }
            blk[0] = dcq[pl][c4];
            transform::dequant_4x4(&mut blk, cqp, true);
            transform::idct_4x4(&mut blk);
            let (px, py) = (cbx + (c4 % 2) * 4, cby + (c4 / 2) * 4);
            let plane = if pl == 0 { &mut fx.cb } else { &mut fx.cr };
            for yy in 0..4 {
                for xx in 0..4 {
                    let idx = (py + yy) * cw + px + xx;
                    plane[idx] = clip8(plane[idx] as i32 + blk[yy * 4 + xx]);
                }
            }
        }
    }
    Ok(())
}

// --- intra macroblocks ----------------------------------------------------------

/// Predicted intra mode for a 4x4/8x8 block (min of available neighbours, DC
/// fallback): neighbours outside the slice → DC; inter / I16 neighbours → DC.
fn pred_intra_mode(fx: &Fx, bx4: usize, by4: usize) -> i32 {
    let get = |x: i32, y: i32| -> i32 {
        if x < 0 || y < 0 {
            return -1;
        }
        let m = (y as usize / 4) * fx.mbw + x as usize / 4;
        if !fx.sl(m) {
            return -1;
        }
        if !fx.mbinxn[m] {
            return 2;
        }
        let v = fx.mode_y[y as usize * fx.ny4 + x as usize];
        if v < 0 {
            2
        } else {
            v
        }
    };
    let l = get(bx4 as i32 - 1, by4 as i32);
    let t = get(bx4 as i32, by4 as i32 - 1);
    if l < 0 || t < 0 {
        2
    } else {
        l.min(t)
    }
}

fn read_intra_pred_mode(cb: &mut Cabac, pred: i32) -> u8 {
    if cb.decision(68) != 0 {
        return pred as u8;
    }
    let mut mode = cb.decision(69) as i32;
    mode += 2 * cb.decision(69) as i32;
    mode += 4 * cb.decision(69) as i32;
    (mode + (mode >= pred) as i32) as u8
}

fn read_chroma_pred_mode(fx: &Fx, cb: &mut Cabac, mb: usize) -> u8 {
    let a = fx.mb_a(mb).map(|m| fx.mbcpm[m] != 0).unwrap_or(false) as usize;
    let b = fx.mb_b(mb).map(|m| fx.mbcpm[m] != 0).unwrap_or(false) as usize;
    if cb.decision(64 + a + b) == 0 {
        return 0;
    }
    if cb.decision(67) == 0 {
        return 1;
    }
    if cb.decision(67) == 0 {
        2
    } else {
        3
    }
}

fn decode_intra_mb(fx: &mut Fx, cb: &mut Cabac, s: &Sl, st: &mut St, mb_x: usize, mb_y: usize, t: u32) -> Result<(), &'static str> {
    if t == 25 {
        return Err("h264: I_PCM in CABAC not supported");
    }
    let mb = mb_y * fx.mbw + mb_x;
    fx.mbintra[mb] = true;
    let is16 = t >= 1;
    let (bx, by, w) = (mb_x * 16, mb_y * 16, fx.w);

    let mut t8 = false;
    let mut modes4 = [0u8; 16];
    let mut modes8 = [0u8; 4];
    let mut i16mode = 0u8;
    let (mut cbp_l, mut cbp_c) = (0u8, 0u8);
    if !is16 {
        fx.mbinxn[mb] = true;
        if s.pps.transform_8x8_mode {
            let a = fx.mb_a(mb).map(|m| fx.mbt8[m]).unwrap_or(false) as usize;
            let b = fx.mb_b(mb).map(|m| fx.mbt8[m]).unwrap_or(false) as usize;
            t8 = cb.decision(399 + a + b) != 0;
        }
        if t8 {
            for (i8x8, m8) in modes8.iter_mut().enumerate() {
                let bx4 = mb_x * 4 + (i8x8 & 1) * 2;
                let by4 = mb_y * 4 + (i8x8 >> 1) * 2;
                let pred = pred_intra_mode(fx, bx4, by4);
                *m8 = read_intra_pred_mode(cb, pred);
                for yy in 0..2 {
                    for xx in 0..2 {
                        fx.mode_y[(by4 + yy) * fx.ny4 + bx4 + xx] = *m8 as i32;
                    }
                }
            }
        } else {
            for b in 0..16 {
                let bx4 = mb_x * 4 + BLK_XY[b].0;
                let by4 = mb_y * 4 + BLK_XY[b].1;
                let pred = pred_intra_mode(fx, bx4, by4);
                modes4[b] = read_intra_pred_mode(cb, pred);
                fx.mode_y[by4 * fx.ny4 + bx4] = modes4[b] as i32;
            }
        }
    } else {
        fx.mbi16[mb] = true;
        let m = t - 1;
        i16mode = (m % 4) as u8;
        cbp_c = ((m / 4) % 3) as u8;
        cbp_l = if m >= 12 { 15 } else { 0 };
    }
    fx.mbt8[mb] = t8;
    let cpm = read_chroma_pred_mode(fx, cb, mb);
    fx.mbcpm[mb] = cpm;

    if !is16 {
        cbp_l = decode_cbp_luma(fx, cb, mb);
        cbp_c = decode_cbp_chroma(fx, cb, mb);
    }
    fx.mbcbp_l[mb] = cbp_l;
    fx.mbcbp_c[mb] = cbp_c;

    if cbp_l != 0 || cbp_c != 0 || is16 {
        read_qp_delta(cb, st)?;
    } else {
        st.last_qp_diff = false;
    }
    fx.mbqp[mb] = st.qp;

    // Luma residual + reconstruction.
    if is16 {
        // DC (cat 0) then AC (cat 1) per 4x4.
        let mut dcsc = [0i32; 16];
        {
            let ctx = cbf_ctx(fx, 0, mb, 0, 0, 0, true);
            if cb.decision(ctx) != 0 {
                let _ = cabac_residual(cb, 0, 16, &mut dcsc);
                fx.mbdcf[mb] |= 1;
            }
        }
        let mut coeffs = [[0i32; 16]; 16];
        if cbp_l != 0 {
            for b in 0..16 {
                let bxx = mb_x * 4 + BLK_XY[b].0;
                let byy = mb_y * 4 + BLK_XY[b].1;
                let ctx = cbf_ctx(fx, 1, mb, bxx, byy, 0, true);
                if cb.decision(ctx) == 0 {
                    fx.nnz_y[byy * fx.ny4 + bxx] = 0;
                    continue;
                }
                let mut sc = [0i32; 16];
                let cnt = cabac_residual(cb, 1, 15, &mut sc[1..]);
                fx.nnz_y[byy * fx.ny4 + bxx] = cnt as i32;
                coeffs[b] = transform::inverse_scan_4x4(&sc);
            }
        } else {
            for b in 0..16 {
                let bxx = mb_x * 4 + BLK_XY[b].0;
                let byy = mb_y * 4 + BLK_XY[b].1;
                fx.nnz_y[byy * fx.ny4 + bxx] = 0;
            }
        }
        // Predict + reconstruct.
        let mut top = [0i32; 16];
        let mut left = [0i32; 16];
        let at = by > 0 && fx.y_dec_xy(bx, by - 1) && fx.sl(fx.mbp(bx, by - 1));
        let al = bx > 0 && fx.y_dec_xy(bx - 1, by) && fx.sl(fx.mbp(bx - 1, by));
        let corner = if bx > 0 && by > 0 && fx.sl(fx.mbp(bx - 1, by - 1)) {
            fx.y[(by - 1) * w + bx - 1] as i32
        } else {
            0
        };
        if at {
            for k in 0..16 {
                top[k] = fx.y[(by - 1) * w + bx + k] as i32;
            }
        }
        if al {
            for k in 0..16 {
                left[k] = fx.y[(by + k) * w + bx - 1] as i32;
            }
        }
        let pred = intra::intra16x16(i16mode, &top, &left, corner, at, al);
        let mut dcq = transform::inverse_scan_4x4(&dcsc);
        transform::luma_dc_transform(&mut dcq, st.qp as u32);
        for b in 0..16 {
            let ox = BLK_XY[b].0 * 4;
            let oy = BLK_XY[b].1 * 4;
            let mut blk = coeffs[b];
            blk[0] = dcq[BLK_XY[b].1 * 4 + BLK_XY[b].0];
            transform::dequant_4x4(&mut blk, st.qp as u32, true);
            transform::idct_4x4(&mut blk);
            for yy in 0..4 {
                for xx in 0..4 {
                    let idx = (by + oy + yy) * w + bx + ox + xx;
                    fx.y[idx] = clip8(pred[(oy + yy) * 16 + ox + xx] + blk[yy * 4 + xx]);
                }
            }
            fx.mark_y4(mb_x * 4 + BLK_XY[b].0, mb_y * 4 + BLK_XY[b].1);
        }
    } else if t8 {
        // I_8x8: per 8x8 — cbf is the cbp bit (no coded_block_flag for cat 5).
        for i8x8 in 0..4 {
            let (ox8, oy8) = ((i8x8 & 1) * 8, (i8x8 >> 1) * 8);
            let (px, py) = (bx + ox8, by + oy8);
            let mut sc = [0i32; 64];
            let mut cnt = 0usize;
            if cbp_l & (1 << i8x8) != 0 {
                cnt = cabac_residual(cb, 5, 64, &mut sc);
            }
            for b in 0..4 {
                let bxx = mb_x * 4 + (i8x8 & 1) * 2 + (b & 1);
                let byy = mb_y * 4 + (i8x8 >> 1) * 2 + (b >> 1);
                fx.nnz_y[byy * fx.ny4 + bxx] = cnt as i32;
            }
            let (top, left, corner, at, al, atl, atr) = fx.gather8(px, py);
            let pred = intra::intra8x8(modes8[i8x8], &top, &left, corner, at, al, atl, atr);
            let mut blk = [0i32; 64];
            if cnt > 0 {
                for (si, &lv) in sc.iter().enumerate() {
                    if lv != 0 {
                        let p = transform::ZIGZAG8[si];
                        blk[p] = transform::dequant8(lv, st.qp, p);
                    }
                }
                transform::idct8_residual(&mut blk);
            }
            for yy in 0..8 {
                for xx in 0..8 {
                    let idx = (py + yy) * w + px + xx;
                    let r = if cnt > 0 { blk[yy * 8 + xx] } else { 0 };
                    fx.y[idx] = clip8(pred[yy * 8 + xx] + r);
                }
            }
            // Four 4×4 stamps under this 8×8.
            for b in 0..4 {
                fx.mark_y4(
                    mb_x * 4 + (i8x8 & 1) * 2 + (b & 1),
                    mb_y * 4 + (i8x8 >> 1) * 2 + (b >> 1),
                );
            }
        }
    } else {
        // I_4x4.
        for b in 0..16 {
            let bxx = mb_x * 4 + BLK_XY[b].0;
            let byy = mb_y * 4 + BLK_XY[b].1;
            let ox = BLK_XY[b].0 * 4;
            let oy = BLK_XY[b].1 * 4;
            let (px, py) = (bx + ox, by + oy);
            let mut blk = [0i32; 16];
            let mut have = false;
            if cbp_l & (1 << (b / 4)) != 0 {
                let ctx = cbf_ctx(fx, 2, mb, bxx, byy, 0, true);
                if cb.decision(ctx) != 0 {
                    let mut sc = [0i32; 16];
                    let cnt = cabac_residual(cb, 2, 16, &mut sc);
                    fx.nnz_y[byy * fx.ny4 + bxx] = cnt as i32;
                    blk = transform::inverse_scan_4x4(&sc);
                    have = true;
                } else {
                    fx.nnz_y[byy * fx.ny4 + bxx] = 0;
                }
            } else {
                fx.nnz_y[byy * fx.ny4 + bxx] = 0;
            }
            let (top, left, corner, at, al, atr) = fx.gather4(px, py);
            let pred = intra::intra4x4(modes4[b], &top, &left, corner, at, al, atr);
            if have {
                transform::dequant_4x4(&mut blk, st.qp as u32, false);
                transform::idct_4x4(&mut blk);
            }
            for yy in 0..4 {
                for xx in 0..4 {
                    let idx = (py + yy) * w + px + xx;
                    fx.y[idx] = clip8(pred[yy * 4 + xx] + blk[yy * 4 + xx]);
                }
            }
            fx.mark_y4(bxx, byy);
        }
    }

    // Chroma: prediction first (planes get the pred), then residual add.
    for pl in 0..2 {
        let predp = fx.chroma_pred(pl, mb_x * 8, mb_y * 8, cpm);
        let plane = if pl == 0 { &mut fx.cb } else { &mut fx.cr };
        let cw = fx.cw;
        for yy in 0..8 {
            for xx in 0..8 {
                plane[(mb_y * 8 + yy) * cw + mb_x * 8 + xx] = clip8(predp[yy * 8 + xx]);
            }
        }
    }
    // Mark chroma available for the next MB's neighbour gather before residual
    // (residual only *adds* into the plane — samples are already valid).
    fx.mark_c_mb(mb);
    decode_residual_chroma(fx, cb, s, st, mb_x, mb_y, cbp_c, true)?;
    Ok(())
}

// --- motion compensation + weighting ----------------------------------------------

/// Apply MC for one partition: fetch per-list prediction blocks and combine
/// (P explicit weights / B implicit weights / average), writing into the planes.
/// Stack buffers (max 16×16 / 8×8) — no per-partition heap traffic.
#[allow(clippy::too_many_arguments)]
fn mc_partition(fx: &mut Fx, s: &Sl, px: usize, py: usize, pw: usize, ph: usize, dmask: u8, r0: i8, mv0: [i16; 2], r1: i8, mv1: [i16; 2]) {
    debug_assert!(pw * ph <= 256 && (pw / 2) * (ph / 2) <= 64);
    let cpx = px / 2;
    let cpy = py / 2;
    let cpw = pw / 2;
    let cph = ph / 2;
    let mut lb = [0i32; 256];
    let mut cbb = [0i32; 64];
    let mut crb = [0i32; 64];
    let single = |list: usize, r: i8, mv: [i16; 2], fx: &mut Fx, lb: &mut [i32; 256], cbb: &mut [i32; 64], crb: &mut [i32; 64]| {
        let pic = &s.lists[list][r as usize];
        // Unweighted full-pel: copy ref → plane directly (dominant P_Skip path).
        let weighted = s.hdr.stype == 0 && s.pps.weighted_pred && (r as usize) < s.hdr.wl.len();
        if !weighted && (mv[0] & 3) == 0 && (mv[1] & 3) == 0 {
            inter::copy_fullpel_u8(
                &mut fx.y, fx.w, px, py, &pic.f.y, fx.w, fx.h, px, py, pw, ph,
                mv[0] as i32, mv[1] as i32, 2,
            );
            // Chroma is eighth-pel: integer sample when mv & 7 == 0.
            if (mv[0] & 7) == 0 && (mv[1] & 7) == 0 {
                inter::copy_fullpel_u8(
                    &mut fx.cb, fx.cw, cpx, cpy, &pic.f.cb, fx.cw, fx.ch, cpx, cpy, cpw, cph,
                    mv[0] as i32, mv[1] as i32, 3,
                );
                inter::copy_fullpel_u8(
                    &mut fx.cr, fx.cw, cpx, cpy, &pic.f.cr, fx.cw, fx.ch, cpx, cpy, cpw, cph,
                    mv[0] as i32, mv[1] as i32, 3,
                );
            } else {
                inter::chroma_block_into(cbb, &pic.f.cb, fx.cw, fx.ch, cpx, cpy, cpw, cph, mv[0] as i32, mv[1] as i32);
                inter::chroma_block_into(crb, &pic.f.cr, fx.cw, fx.ch, cpx, cpy, cpw, cph, mv[0] as i32, mv[1] as i32);
                for (pl, blkc) in [(0usize, cbb.as_ref()), (1, crb.as_ref())] {
                    let plane = if pl == 0 { &mut fx.cb } else { &mut fx.cr };
                    for j in 0..cph {
                        let src = j * cpw;
                        let dst = (cpy + j) * fx.cw + cpx;
                        for i in 0..cpw {
                            plane[dst + i] = blkc[src + i] as u8;
                        }
                    }
                }
            }
            return;
        }
        inter::luma_block_into(lb, &pic.f.y, fx.w, fx.h, px, py, pw, ph, mv[0] as i32, mv[1] as i32);
        inter::chroma_block_into(cbb, &pic.f.cb, fx.cw, fx.ch, cpx, cpy, cpw, cph, mv[0] as i32, mv[1] as i32);
        inter::chroma_block_into(crb, &pic.f.cr, fx.cw, fx.ch, cpx, cpy, cpw, cph, mv[0] as i32, mv[1] as i32);
        // P explicit weighting (weighted_pred): also applies to P_Skip.
        if weighted {
            let (wy, oy) = s.hdr.wl[r as usize];
            let ldl = s.hdr.luma_ld;
            for j in 0..ph {
                for i in 0..pw {
                    let v = lb[j * pw + i];
                    let wv = if ldl > 0 {
                        ((v * wy + (1 << (ldl - 1))) >> ldl) + oy
                    } else {
                        v * wy + oy
                    };
                    fx.y[(py + j) * fx.w + px + i] = clip8(wv);
                }
            }
            let ldc = s.hdr.chroma_ld;
            for (pl, blkc) in [(0usize, cbb.as_ref()), (1, crb.as_ref())] {
                let (wcv, oc) = s.hdr.wc[r as usize][pl];
                let plane = if pl == 0 { &mut fx.cb } else { &mut fx.cr };
                for j in 0..cph {
                    for i in 0..cpw {
                        let v = blkc[j * cpw + i];
                        let wv = if ldc > 0 {
                            ((v * wcv + (1 << (ldc - 1))) >> ldc) + oc
                        } else {
                            v * wcv + oc
                        };
                        plane[(cpy + j) * fx.cw + cpx + i] = clip8(wv);
                    }
                }
            }
        } else {
            for j in 0..ph {
                let src = j * pw;
                let dst = (py + j) * fx.w + px;
                for i in 0..pw {
                    fx.y[dst + i] = lb[src + i] as u8;
                }
            }
            for (pl, blkc) in [(0usize, cbb.as_ref()), (1, crb.as_ref())] {
                let plane = if pl == 0 { &mut fx.cb } else { &mut fx.cr };
                for j in 0..cph {
                    let src = j * cpw;
                    let dst = (cpy + j) * fx.cw + cpx;
                    for i in 0..cpw {
                        plane[dst + i] = blkc[src + i] as u8;
                    }
                }
            }
        }
    };
    match dmask {
        1 => single(0, r0, mv0, fx, &mut lb, &mut cbb, &mut crb),
        2 => single(1, r1, mv1, fx, &mut lb, &mut cbb, &mut crb),
        _ => {
            // Bi-prediction: implicit weights (weighted_bipred_idc 2) or average.
            let p0 = &s.lists[0][r0 as usize];
            let p1 = &s.lists[1][r1 as usize];
            let wts = if s.pps.weighted_bipred_idc == 2 {
                implicit_weights(s.poc, p0.poc, p1.poc)
            } else {
                None
            };
            let mut l0 = [0i32; 256];
            let mut l1 = [0i32; 256];
            inter::luma_block_into(&mut l0, &p0.f.y, fx.w, fx.h, px, py, pw, ph, mv0[0] as i32, mv0[1] as i32);
            inter::luma_block_into(&mut l1, &p1.f.y, fx.w, fx.h, px, py, pw, ph, mv1[0] as i32, mv1[1] as i32);
            for j in 0..ph {
                for i in 0..pw {
                    let (a, b) = (l0[j * pw + i], l1[j * pw + i]);
                    let v = match wts {
                        Some((w0, w1)) => (a * w0 + b * w1 + 32) >> 6,
                        None => (a + b + 1) >> 1,
                    };
                    fx.y[(py + j) * fx.w + px + i] = clip8(v);
                }
            }
            for pl in 0..2 {
                let (s0, s1) = if pl == 0 {
                    (&p0.f.cb, &p1.f.cb)
                } else {
                    (&p0.f.cr, &p1.f.cr)
                };
                let mut c0 = [0i32; 64];
                let mut c1 = [0i32; 64];
                inter::chroma_block_into(&mut c0, s0, fx.cw, fx.ch, cpx, cpy, cpw, cph, mv0[0] as i32, mv0[1] as i32);
                inter::chroma_block_into(&mut c1, s1, fx.cw, fx.ch, cpx, cpy, cpw, cph, mv1[0] as i32, mv1[1] as i32);
                let plane = if pl == 0 { &mut fx.cb } else { &mut fx.cr };
                for j in 0..cph {
                    for i in 0..cpw {
                        let (a, b) = (c0[j * cpw + i], c1[j * cpw + i]);
                        let v = match wts {
                            Some((w0, w1)) => (a * w0 + b * w1 + 32) >> 6,
                            None => (a + b + 1) >> 1,
                        };
                        plane[(cpy + j) * fx.cw + cpx + i] = clip8(v);
                    }
                }
            }
        }
    }
}

// --- spatial direct (§8.4.1.2.2, frame + direct_8x8_inference) ---------------------

fn min_positive(a: i8, b: i8) -> i8 {
    if a >= 0 && b >= 0 {
        a.min(b)
    } else {
        a.max(b)
    }
}

/// Direct-mode dispatch: spatial or temporal per the slice header flag.
fn pred_direct(fx: &mut Fx, s: &Sl, mb_x: usize, mb_y: usize, subs: &[bool; 4]) -> Result<(), &'static str> {
    if s.hdr.direct_spatial {
        pred_spatial_direct(fx, s, mb_x, mb_y, subs)
    } else {
        pred_temporal_direct(fx, s, mb_x, mb_y, subs)
    }
}

/// Temporal direct (§8.4.1.2.3, frame coding + direct_8x8_inference): scale the
/// colocated picture's motion by the POC distances.
fn pred_temporal_direct(fx: &mut Fx, s: &Sl, mb_x: usize, mb_y: usize, subs: &[bool; 4]) -> Result<(), &'static str> {
    let col = s.lists[1].first().ok_or("h264: B direct without list-1 reference")?.clone();
    for (i8x8, &d) in subs.iter().enumerate() {
        if !d {
            continue;
        }
        let (x8, y8) = (i8x8 & 1, i8x8 >> 1);
        let b0x = mb_x * 4 + x8 * 2;
        let b0y = mb_y * 4 + y8 * 2;
        // Colocated corner 4x4 (direct_8x8_inference).
        let ci = (mb_y * 4 + y8 * 3) * fx.ny4 + mb_x * 4 + x8 * 3;
        let (colref0, colref1) = (col.refidx[0][ci], col.refidx[1][ci]);
        let (mv_col, col_refpoc) = if colref0 >= 0 {
            (col.mv[0][ci], col.refpoc[0][ci])
        } else if colref1 >= 0 {
            (col.mv[1][ci], col.refpoc[1][ci])
        } else {
            // Intra colocated: mvCol = 0, refIdxL0 = 0.
            ([0i16, 0i16], i32::MIN)
        };
        // Map the colocated reference into the current list 0 (lowest index
        // whose picture matches by POC); intra col → index 0.
        let ref0 = if col_refpoc == i32::MIN {
            0usize
        } else {
            s.lists[0]
                .iter()
                .position(|p| p.poc == col_refpoc)
                .ok_or("h264: temporal direct reference not in list 0")?
        };
        let poc0 = s.lists[0][ref0].poc;
        let poc1 = col.poc;
        let td = (poc1 - poc0).clamp(-128, 127);
        let (mv0, mv1) = if td == 0 {
            (mv_col, [0i16, 0i16])
        } else {
            let tb = (s.poc - poc0).clamp(-128, 127);
            let tx = (16384 + (td / 2).abs()) / td;
            let dsf = ((tb * tx + 32) >> 6).clamp(-1024, 1023);
            let m0 = [
                (((dsf * mv_col[0] as i32) + 128) >> 8) as i16,
                (((dsf * mv_col[1] as i32) + 128) >> 8) as i16,
            ];
            (m0, [m0[0] - mv_col[0], m0[1] - mv_col[1]])
        };
        fx.store_mv(0, b0x, b0y, 2, 2, mv0, ref0 as i8, poc0);
        fx.store_mv(1, b0x, b0y, 2, 2, mv1, 0, poc1);
        let g = fx.pix_gen;
        for yy in 0..2 {
            for xx in 0..2 {
                fx.dirf[(b0y + yy) * fx.ny4 + b0x + xx] = g;
            }
        }
    }
    Ok(())
}

/// Derive and store direct motion for the direct 8x8s of the MB.
fn pred_spatial_direct(fx: &mut Fx, s: &Sl, mb_x: usize, mb_y: usize, subs: &[bool; 4]) -> Result<(), &'static str> {
    let col = s.lists[1].first().ok_or("h264: B direct without list-1 reference")?.clone();
    let bx4 = mb_x as i32 * 4;
    let by4 = mb_y as i32 * 4;
    let mut refs = [-1i8; 2];
    let mut mvs = [[0i16; 2]; 2];
    for list in 0..2 {
        let a = fx.nb(list, bx4 - 1, by4);
        let b = fx.nb(list, bx4, by4 - 1);
        let mut c = fx.nb(list, bx4 + 4, by4 - 1);
        if !c.2 {
            c = fx.nb(list, bx4 - 1, by4 - 1);
        }
        let r = min_positive(min_positive(a.1, b.1), c.1);
        refs[list] = r;
        if r >= 0 {
            let matches = (a.1 == r) as u32 + (b.1 == r) as u32 + (c.1 == r) as u32;
            mvs[list] = if matches > 1 {
                [
                    inter::median3(a.0[0] as i32, b.0[0] as i32, c.0[0] as i32) as i16,
                    inter::median3(a.0[1] as i32, b.0[1] as i32, c.0[1] as i32) as i16,
                ]
            } else if a.1 == r {
                a.0
            } else if b.1 == r {
                b.0
            } else {
                c.0
            };
        }
    }
    if refs[0] < 0 && refs[1] < 0 {
        refs = [0, 0];
        mvs = [[0, 0], [0, 0]];
    }
    let rp = [
        if refs[0] >= 0 { ref_poc(s, 0, refs[0])? } else { i32::MIN },
        if refs[1] >= 0 { ref_poc(s, 1, refs[1])? } else { i32::MIN },
    ];
    for (i8x8, &d) in subs.iter().enumerate() {
        if !d {
            continue;
        }
        let (x8, y8) = (i8x8 & 1, i8x8 >> 1);
        let b0x = mb_x * 4 + x8 * 2;
        let b0y = mb_y * 4 + y8 * 2;
        // Colocated corner 4x4 (direct_8x8_inference).
        let ci = (mb_y * 4 + y8 * 3) * fx.ny4 + mb_x * 4 + x8 * 3;
        let (colref0, colref1) = (col.refidx[0][ci], col.refidx[1][ci]);
        let colmv = if colref0 == 0 {
            Some(col.mv[0][ci])
        } else if colref0 < 0 && colref1 == 0 {
            Some(col.mv[1][ci])
        } else {
            None
        };
        let zero_it = matches!(colmv, Some(m) if m[0].abs() <= 1 && m[1].abs() <= 1);
        for list in 0..2 {
            let mv = if zero_it && refs[list] == 0 { [0, 0] } else if refs[list] >= 0 { mvs[list] } else { [0, 0] };
            let ridx = refs[list];
            let rpoc = if ridx >= 0 { rp[list] } else { i32::MIN };
            fx.store_mv(list, b0x, b0y, 2, 2, mv, ridx, rpoc);
        }
        let g = fx.pix_gen;
        for yy in 0..2 {
            for xx in 0..2 {
                fx.dirf[(b0y + yy) * fx.ny4 + b0x + xx] = g;
            }
        }
    }
    Ok(())
}

/// MC for the direct 8x8s (after `pred_spatial_direct` stored their motion).
fn mc_direct(fx: &mut Fx, s: &Sl, mb_x: usize, mb_y: usize, subs: &[bool; 4]) {
    for (i8x8, &d) in subs.iter().enumerate() {
        if !d {
            continue;
        }
        let (x8, y8) = (i8x8 & 1, i8x8 >> 1);
        let i4 = (mb_y * 4 + y8 * 2) * fx.ny4 + mb_x * 4 + x8 * 2;
        let r0 = fx.refidx[0][i4];
        let r1 = fx.refidx[1][i4];
        let dmask = ((r0 >= 0) as u8) | (((r1 >= 0) as u8) << 1);
        mc_partition(
            fx,
            s,
            mb_x * 16 + x8 * 8,
            mb_y * 16 + y8 * 8,
            8,
            8,
            dmask,
            r0.max(0),
            fx.mv[0][i4],
            r1.max(0),
            fx.mv[1][i4],
        );
    }
}

// --- frame deblock -----------------------------------------------------------------

fn deblock_frame(fx: &mut Fx, aoff: i32, boff: i32, pps: &Pps) {
    // Deblock still works on i32; promote → filter → demote (only used ≤720p).
    let (w, h, cw, ch) = (fx.w, fx.h, fx.cw, fx.ch);
    let mut yi: alloc::vec::Vec<i32> = fx.y.iter().map(|&v| v as i32).collect();
    let mut cbi: alloc::vec::Vec<i32> = fx.cb.iter().map(|&v| v as i32).collect();
    let mut cri: alloc::vec::Vec<i32> = fx.cr.iter().map(|&v| v as i32).collect();
    let m = deblock::Meta2 {
        mbw: fx.mbw,
        ny4: fx.ny4,
        mbqp: &fx.mbqp,
        mbintra: &fx.mbintra,
        mbt8: &fx.mbt8,
        nnz_y: &fx.nnz_y,
        mv0: &fx.mv[0],
        mv1: &fx.mv[1],
        rp0: &fx.refpoc[0],
        rp1: &fx.refpoc[1],
        aoff,
        boff,
        cqpoff: [pps.chroma_qp_index_offset, pps.second_chroma_qp_index_offset],
    };
    deblock::deblock2(&mut yi, &mut cbi, &mut cri, w, cw, fx.mbw, fx.mbh, &m);
    for (d, s) in fx.y.iter_mut().zip(yi.iter()) {
        *d = (*s).clamp(0, 255) as u8;
    }
    for (d, s) in fx.cb.iter_mut().zip(cbi.iter()) {
        *d = (*s).clamp(0, 255) as u8;
    }
    for (d, s) in fx.cr.iter_mut().zip(cri.iter()) {
        *d = (*s).clamp(0, 255) as u8;
    }
    let _ = (h, ch);
}
