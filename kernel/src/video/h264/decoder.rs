//! H.264 slice + macroblock decoding → reconstructed YUV frames. Ports the
//! reference decoder validated bit-exact against PyAV (see `tools/h264diff`).
//!
//! Scope: **I and P slices** (I_4x4/I_16x16/I_PCM; P_L0_16x16/16x8/8x16/8x8/Skip),
//! CAVLC, 4:2:0, **multiple slices per frame** (real encoders split a frame into
//! slices; cross-slice neighbours are unavailable — enforced for nC / intra /
//! MV prediction). Intra ([`super::intra`]) + inverse transform
//! ([`super::transform`]) + CAVLC ([`super::cavlc`]) + inter ([`super::inter`]) +
//! in-loop deblocking ([`super::deblock`]).

use super::super::bits::BitReader;
use super::{cavlc, inter, intra, transform, Pps, Sps};
use alloc::vec;
use alloc::vec::Vec;

const BLK_XY: [(usize, usize); 16] = [
    (0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (3, 0), (2, 1), (3, 1),
    (0, 2), (1, 2), (0, 3), (1, 3), (2, 2), (3, 2), (2, 3), (3, 3),
];
const CBP_INTRA: [u8; 48] = [
    47, 31, 15, 0, 23, 27, 29, 30, 7, 11, 13, 14, 39, 43, 45, 46, 16, 3, 5, 10, 12, 19, 21, 26, 28, 35, 37, 42, 44, 1,
    2, 4, 8, 17, 18, 20, 24, 6, 9, 22, 25, 32, 33, 34, 36, 40, 38, 41,
];
const CBP_INTER: [u8; 48] = [
    0, 16, 1, 2, 4, 8, 32, 3, 5, 10, 12, 15, 47, 7, 11, 13, 14, 6, 9, 31, 35, 37, 42, 44, 33, 34, 36, 40, 39, 43, 45, 46,
    17, 18, 20, 24, 19, 21, 26, 28, 23, 27, 29, 30, 22, 25, 38, 41,
];
const SUB: [(usize, usize, usize); 4] = [(1, 2, 2), (2, 2, 1), (2, 1, 2), (4, 1, 1)];

fn qpc(qpi: i32) -> i32 {
    const TAB: [i32; 22] = [29, 30, 31, 32, 32, 33, 34, 34, 35, 35, 36, 36, 37, 37, 37, 38, 38, 38, 39, 39, 39, 39];
    if qpi < 30 {
        qpi
    } else {
        TAB[(qpi - 30) as usize]
    }
}
fn clip8(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

/// A decoded frame as YUV 4:2:0 planes.
pub struct DecodedFrame {
    pub w: usize,
    pub h: usize,
    pub y: Vec<u8>,
    pub cb: Vec<u8>,
    pub cr: Vec<u8>,
}

/// Per-frame decode state, shared across the frame's slices.
struct Ctx<'a> {
    w: usize,
    h: usize,
    cw: usize,
    ch: usize,
    mbw: usize,
    mbh: usize,
    ny4: usize,
    nc2: usize,
    y: Vec<i32>,
    cb: Vec<i32>,
    cr: Vec<i32>,
    decy: Vec<bool>,
    decu: Vec<bool>,
    decv: Vec<bool>,
    nnz_y: Vec<i32>,
    nnz_u: Vec<i32>,
    nnz_v: Vec<i32>,
    mode_y: Vec<i32>,
    mvx: Vec<i32>,
    mvy: Vec<i32>,
    refi: Vec<i32>,
    mbqp: Vec<i32>,
    mbintra: Vec<bool>,
    mbslice: Vec<i32>,
    cur_slice: i32,
    refr: Option<&'a DecodedFrame>,
    qp: i32,
    chroma_qp_off: i32,
    dbf_idc: u32,
    aoff: i32,
    boff: i32,
}

impl<'a> Ctx<'a> {
    fn new(sps: &Sps, pps: &Pps, refr: Option<&'a DecodedFrame>) -> Ctx<'a> {
        let mbw = sps.pic_width_in_mbs as usize;
        let mbh = sps.pic_height_in_map_units as usize;
        let (w, h) = (mbw * 16, mbh * 16);
        let (cw, ch) = (w / 2, h / 2);
        let ny4 = mbw * 4;
        let nc2 = mbw * 2;
        Ctx {
            w, h, cw, ch, mbw, mbh, ny4, nc2,
            y: vec![0; w * h],
            cb: vec![0; cw * ch],
            cr: vec![0; cw * ch],
            decy: vec![false; w * h],
            decu: vec![false; cw * ch],
            decv: vec![false; cw * ch],
            nnz_y: vec![0; ny4 * mbh * 4],
            nnz_u: vec![0; nc2 * mbh * 2],
            nnz_v: vec![0; nc2 * mbh * 2],
            mode_y: vec![-1; ny4 * mbh * 4],
            mvx: vec![0; ny4 * mbh * 4],
            mvy: vec![0; ny4 * mbh * 4],
            refi: vec![-1; ny4 * mbh * 4],
            mbqp: vec![0; mbw * mbh],
            mbintra: vec![false; mbw * mbh],
            mbslice: vec![-1; mbw * mbh],
            cur_slice: -1,
            refr,
            qp: 0,
            chroma_qp_off: pps.chroma_qp_index_offset,
            dbf_idc: 0,
            aoff: 0,
            boff: 0,
        }
    }
    #[inline]
    fn mb4(&self, bx4: usize, by4: usize) -> usize {
        (by4 / 4) * self.mbw + bx4 / 4
    }
    #[inline]
    fn mbp(&self, x: usize, y: usize) -> usize {
        (y / 16) * self.mbw + x / 16
    }
    #[inline]
    fn mbpc(&self, x: usize, y: usize) -> usize {
        (y / 8) * self.mbw + x / 8
    }
    #[inline]
    fn sl(&self, mb_idx: usize) -> bool {
        self.mbslice[mb_idx] == self.cur_slice
    }
    fn nc_l(&self, bxx: usize, byy: usize) -> i32 {
        let a = bxx > 0 && self.sl(self.mb4(bxx - 1, byy));
        let b = byy > 0 && self.sl(self.mb4(bxx, byy - 1));
        let na = if a { self.nnz_y[byy * self.ny4 + bxx - 1] } else { -1 };
        let nb = if b { self.nnz_y[(byy - 1) * self.ny4 + bxx] } else { -1 };
        if a && b {
            (na + nb + 1) >> 1
        } else if a {
            na
        } else if b {
            nb
        } else {
            0
        }
    }
    fn nc_c(&self, nnz: &[i32], cxx: usize, cyy: usize) -> i32 {
        let a = cxx > 0 && self.sl((cyy / 2) * self.mbw + (cxx - 1) / 2);
        let b = cyy > 0 && self.sl(((cyy - 1) / 2) * self.mbw + cxx / 2);
        let na = if a { nnz[cyy * self.nc2 + cxx - 1] } else { -1 };
        let nb = if b { nnz[(cyy - 1) * self.nc2 + cxx] } else { -1 };
        if a && b {
            (na + nb + 1) >> 1
        } else if a {
            na
        } else if b {
            nb
        } else {
            0
        }
    }
    fn nb(&self, bx4: i32, by4: i32) -> (i32, i32, i32, bool) {
        if bx4 < 0 || by4 < 0 || bx4 >= self.ny4 as i32 || by4 >= (self.h / 4) as i32 {
            return (0, 0, -1, false);
        }
        if !self.decy[(by4 as usize * 4) * self.w + bx4 as usize * 4] || !self.sl(self.mb4(bx4 as usize, by4 as usize)) {
            return (0, 0, -1, false);
        }
        let i = by4 as usize * self.ny4 + bx4 as usize;
        (self.mvx[i], self.mvy[i], self.refi[i], true)
    }
    fn predict_mv(&self, bx4: i32, by4: i32, pw4: i32, ridx: i32, kind: u8) -> (i32, i32) {
        let a = self.nb(bx4 - 1, by4);
        let b = self.nb(bx4, by4 - 1);
        let mut c = self.nb(bx4 + pw4, by4 - 1);
        if !c.3 {
            c = self.nb(bx4 - 1, by4 - 1);
        }
        match kind {
            1 if b.2 == ridx => return (b.0, b.1),
            2 if a.2 == ridx => return (a.0, a.1),
            3 if a.2 == ridx => return (a.0, a.1),
            4 if c.2 == ridx => return (c.0, c.1),
            _ => {}
        }
        if !b.3 && !c.3 && a.3 {
            return (a.0, a.1);
        }
        let same: Vec<&(i32, i32, i32, bool)> = [&a, &b, &c].into_iter().filter(|n| n.2 == ridx).collect();
        if same.len() == 1 {
            return (same[0].0, same[0].1);
        }
        (inter::median3(a.0, b.0, c.0), inter::median3(a.1, b.1, c.1))
    }
    fn skip_mv(&self, mb_x: usize, mb_y: usize) -> (i32, i32) {
        let bx4 = mb_x as i32 * 4;
        let by4 = mb_y as i32 * 4;
        let a = self.nb(bx4 - 1, by4);
        let b = self.nb(bx4, by4 - 1);
        if !a.3 || !b.3 || (a.2 == 0 && a.0 == 0 && a.1 == 0) || (b.2 == 0 && b.0 == 0 && b.1 == 0) {
            return (0, 0);
        }
        self.predict_mv(bx4, by4, 4, 0, 0)
    }
    fn store_mv(&mut self, bx4: usize, by4: usize, pw4: usize, ph4: usize, vx: i32, vy: i32, ridx: i32) {
        for yy in 0..ph4 {
            for xx in 0..pw4 {
                let i = (by4 + yy) * self.ny4 + (bx4 + xx);
                self.mvx[i] = vx;
                self.mvy[i] = vy;
                self.refi[i] = ridx;
            }
        }
    }
    fn mc(&mut self, px: usize, py: usize, pw: usize, ph: usize, vx: i32, vy: i32) {
        let rf = self.refr.expect("P slice without reference");
        let lb = inter::luma_block(&rf.y, self.w, self.h, px, py, pw, ph, vx, vy);
        for j in 0..ph {
            for i in 0..pw {
                self.y[(py + j) * self.w + px + i] = lb[j * pw + i];
                self.decy[(py + j) * self.w + px + i] = true;
            }
        }
        let (cpx, cpy, cpw, cph) = (px / 2, py / 2, pw / 2, ph / 2);
        for pl in 0..2 {
            let src = if pl == 0 { &rf.cb } else { &rf.cr };
            let cbk = inter::chroma_block(src, self.cw, self.ch, cpx, cpy, cpw, cph, vx, vy);
            let (plane, dec) = if pl == 0 { (&mut self.cb, &mut self.decu) } else { (&mut self.cr, &mut self.decv) };
            for j in 0..cph {
                for i in 0..cpw {
                    plane[(cpy + j) * self.cw + cpx + i] = cbk[j * cpw + i];
                    dec[(cpy + j) * self.cw + cpx + i] = true;
                }
            }
        }
    }
    /// Intra-4×4 neighbour gather (slice-aware).
    fn gather4(&self, px: usize, py: usize) -> ([i32; 8], [i32; 4], i32, bool, bool, bool) {
        let (pl, dec, w) = (&self.y, &self.decy, self.w);
        let mut top = [0i32; 8];
        let mut left = [0i32; 4];
        let at = py > 0 && dec[(py - 1) * w + px] && self.sl(self.mbp(px, py - 1));
        let al = px > 0 && dec[py * w + px - 1] && self.sl(self.mbp(px - 1, py));
        let corner = if px > 0 && py > 0 && dec[(py - 1) * w + px - 1] && self.sl(self.mbp(px - 1, py - 1)) {
            pl[(py - 1) * w + px - 1]
        } else {
            0
        };
        let atr = py > 0 && (px + 4) < w && dec[(py - 1) * w + px + 4] && self.sl(self.mbp(px + 4, py - 1));
        if at {
            for k in 0..4 {
                top[k] = pl[(py - 1) * w + px + k];
            }
        }
        if atr {
            for k in 4..8 {
                let xx = px + k;
                top[k] = if xx < w { pl[(py - 1) * w + xx] } else { pl[(py - 1) * w + w - 1] };
            }
        }
        if al {
            for k in 0..4 {
                left[k] = pl[(py + k) * w + px - 1];
            }
        }
        (top, left, corner, at, al, atr)
    }
    fn clear_chroma_nnz(&mut self, mb_x: usize, mb_y: usize) {
        for c4 in 0..4 {
            let i = (mb_y * 2 + c4 / 2) * self.nc2 + mb_x * 2 + (c4 % 2);
            self.nnz_u[i] = 0;
            self.nnz_v[i] = 0;
        }
    }
    fn set_mb_mode(&mut self, mb_x: usize, mb_y: usize, v: i32) {
        for yy in 0..4 {
            for xx in 0..4 {
                self.mode_y[(mb_y * 4 + yy) * self.ny4 + mb_x * 4 + xx] = v;
            }
        }
    }
    fn clear_luma_nnz(&mut self, mb_x: usize, mb_y: usize) {
        for yy in 0..4 {
            for xx in 0..4 {
                self.nnz_y[(mb_y * 4 + yy) * self.ny4 + mb_x * 4 + xx] = 0;
            }
        }
    }
    fn recon_chroma(&mut self, r: &mut BitReader, mb_x: usize, mb_y: usize, cbp_chroma: usize, chroma_mode: u8, is_intra: bool) -> Result<(), &'static str> {
        let (cbx, cby) = (mb_x * 8, mb_y * 8);
        let cqp = qpc((self.qp + self.chroma_qp_off).clamp(0, 51));
        let mut cdc = [[0i32; 4]; 2];
        let mut cac = [[[0i32; 16]; 4]; 2];
        if cbp_chroma != 0 {
            for pl in 0..2 {
                let (dc, _) = cavlc::residual_block(r, 4, -1)?;
                cdc[pl] = [dc[0], dc[1], dc[2], dc[3]];
            }
            if cbp_chroma == 2 {
                for pl in 0..2 {
                    for c4 in 0..4 {
                        let cxx = mb_x * 2 + (c4 % 2);
                        let cyy = mb_y * 2 + (c4 / 2);
                        let nc = {
                            let nnz_ref: &[i32] = if pl == 0 { &self.nnz_u } else { &self.nnz_v };
                            self.nc_c(nnz_ref, cxx, cyy)
                        };
                        let (ac, tc) = cavlc::residual_block(r, 15, nc)?;
                        let mut blk = [0i32; 16];
                        blk[1..16].copy_from_slice(&ac[0..15]);
                        cac[pl][c4] = transform::inverse_scan_4x4(&blk);
                        if pl == 0 {
                            self.nnz_u[cyy * self.nc2 + cxx] = tc as i32;
                        } else {
                            self.nnz_v[cyy * self.nc2 + cxx] = tc as i32;
                        }
                    }
                }
            } else {
                self.clear_chroma_nnz(mb_x, mb_y);
            }
        } else {
            self.clear_chroma_nnz(mb_x, mb_y);
        }
        for pl in 0..2 {
            let pred = if is_intra {
                self.chroma_intra_pred(pl, cbx, cby, chroma_mode)
            } else {
                [0i32; 64]
            };
            let mut dcq = cdc[pl];
            if cbp_chroma != 0 {
                transform::chroma_dc_transform(&mut dcq, cqp as u32);
            } else {
                dcq = [0; 4];
            }
            let (plane, dec) = if pl == 0 { (&mut self.cb, &mut self.decu) } else { (&mut self.cr, &mut self.decv) };
            for c4 in 0..4 {
                let ox = (c4 % 2) * 4;
                let oy = (c4 / 2) * 4;
                let mut blk = if cbp_chroma == 2 { cac[pl][c4] } else { [0i32; 16] };
                blk[0] = dcq[c4];
                transform::dequant_4x4(&mut blk, cqp as u32, true);
                transform::idct_4x4(&mut blk);
                for yy in 0..4 {
                    for xx in 0..4 {
                        let px = cbx + ox + xx;
                        let py = cby + oy + yy;
                        let base = if is_intra { pred[(oy + yy) * 8 + ox + xx] } else { plane[py * self.cw + px] };
                        plane[py * self.cw + px] = clip8(base + blk[yy * 4 + xx]) as i32;
                        dec[py * self.cw + px] = true;
                    }
                }
            }
        }
        Ok(())
    }
    /// Intra chroma prediction (slice-aware neighbours; per-quadrant DC).
    fn chroma_intra_pred(&self, pl: usize, cbx: usize, cby: usize, chroma_mode: u8) -> [i32; 64] {
        let (plane, dec) = if pl == 0 { (&self.cb, &self.decu) } else { (&self.cr, &self.decv) };
        let mut top = [0i32; 8];
        let mut left = [0i32; 8];
        let at = cby > 0 && dec[(cby - 1) * self.cw + cbx] && self.sl(self.mbpc(cbx, cby - 1));
        let al = cbx > 0 && dec[cby * self.cw + cbx - 1] && self.sl(self.mbpc(cbx - 1, cby));
        let corner = if cbx > 0 && cby > 0 && self.sl(self.mbpc(cbx - 1, cby - 1)) { plane[(cby - 1) * self.cw + cbx - 1] } else { 0 };
        if at {
            for k in 0..8 {
                top[k] = plane[(cby - 1) * self.cw + cbx + k];
            }
        }
        if al {
            for k in 0..8 {
                left[k] = plane[(cby + k) * self.cw + cbx - 1];
            }
        }
        intra::intra_chroma(chroma_mode, &top, &left, corner, at, al)
    }
    fn into_frame(self) -> DecodedFrame {
        DecodedFrame {
            w: self.w,
            h: self.h,
            y: self.y.iter().map(|&v| v as u8).collect(),
            cb: self.cb.iter().map(|&v| v as u8).collect(),
            cr: self.cr.iter().map(|&v| v as u8).collect(),
        }
    }
}

/// Decode a full access unit (all its slice RBSPs, in order) into one frame,
/// then deblock. `slices` is `(rbsp, is_idr)` per slice. `ref_frame` is the
/// previous decoded frame (for P slices).
pub fn decode_access_unit(sps: &Sps, pps: &Pps, slices: &[(Vec<u8>, bool)], ref_frame: Option<&DecodedFrame>) -> Result<DecodedFrame, &'static str> {
    let mut c = Ctx::new(sps, pps, ref_frame);
    for (si, (rbsp, is_idr)) in slices.iter().enumerate() {
        c.cur_slice = si as i32;
        decode_slice_into(&mut c, sps, pps, rbsp, *is_idr)?;
    }
    // Deblock the assembled frame (disable_deblocking_filter_idc==1 → skip).
    if c.dbf_idc != 1 {
        let m = super::deblock::Meta {
            mbw: c.mbw,
            ny4: c.ny4,
            mbqp: &c.mbqp,
            mbintra: &c.mbintra,
            nnz_y: &c.nnz_y,
            mvx: &c.mvx,
            mvy: &c.mvy,
            refi: &c.refi,
            aoff: c.aoff,
            boff: c.boff,
            chroma_qp_off: pps.chroma_qp_index_offset,
        };
        super::deblock::deblock(&mut c.y, &mut c.cb, &mut c.cr, c.w, c.h, c.cw, c.mbw, c.mbh, &m);
    }
    Ok(c.into_frame())
}

/// Decode one slice's macroblocks into `c` (from its `first_mb` until the slice
/// RBSP ends). Parses the slice header (with the pic-order and redundant fields
/// real streams carry).
fn decode_slice_into(c: &mut Ctx, sps: &Sps, pps: &Pps, rbsp: &[u8], is_idr: bool) -> Result<(), &'static str> {
    let mbw = c.mbw;
    let ny4 = c.ny4;
    let w = c.w;
    let cw = c.cw;
    let mut r = BitReader::new(rbsp);
    let first_mb = r.ue()? as usize;
    let slice_type = r.ue()? % 5;
    let _pps_id = r.ue()?;
    let _frame_num = r.u(sps.log2_max_frame_num)?;
    if is_idr {
        let _idr = r.ue()?;
    }
    if sps.pic_order_cnt_type == 0 {
        let _poc = r.u(sps.log2_max_poc_lsb)?;
        if pps.bottom_field_pic_order_present {
            let _delta_bottom = r.se()?;
        }
    }
    if pps.redundant_pic_cnt_present {
        let _redundant = r.ue()?;
    }
    let is_p = slice_type == 0;
    let mut num_ref = 1u32;
    if is_p {
        if r.bit()? == 1 {
            num_ref = r.ue()? + 1;
        }
        if r.bit()? == 1 {
            loop {
                let idc = r.ue()?;
                if idc == 0 || idc == 1 || idc == 2 {
                    let _ = r.ue()?;
                }
                if idc == 3 {
                    break;
                }
            }
        }
    }
    if is_idr {
        let _ = r.bit()?;
        let _ = r.bit()?;
    } else {
        // dec_ref_pic_marking for a reference (non-IDR) slice.
        if r.bit()? == 1 {
            loop {
                let op = r.ue()?;
                if op == 1 || op == 3 {
                    let _ = r.ue()?;
                }
                if op == 2 {
                    let _ = r.ue()?;
                }
                if op == 3 || op == 6 {
                    let _ = r.ue()?;
                }
                if op == 4 {
                    let _ = r.ue()?;
                }
                if op == 0 {
                    break;
                }
            }
        }
    }
    c.qp = pps.pic_init_qp + r.se()?;
    let mut dbf_idc = 0u32;
    let mut aoff = 0i32;
    let mut boff = 0i32;
    if pps.deblocking_filter_control_present {
        dbf_idc = r.ue()?;
        if dbf_idc != 1 {
            aoff = r.se()? * 2;
            boff = r.se()? * 2;
        }
    }
    // The frame's deblock params come from its first slice.
    if c.cur_slice == 0 {
        c.dbf_idc = dbf_idc;
        c.aoff = aoff;
        c.boff = boff;
    }
    if is_p && c.refr.is_none() {
        return Err("video: P slice without a reference frame");
    }

    let total = mbw * c.mbh;
    let mut mb = first_mb;
    let mut skip = 0u32;
    let mut pending = false;
    while mb < total && r.more_rbsp_data() {
        let mb_x = mb % mbw;
        let mb_y = mb / mbw;
        let bx = mb_x * 16;
        let by = mb_y * 16;
        c.mbslice[mb] = c.cur_slice;
        if is_p && !pending {
            skip = r.ue()?;
            pending = true;
        }
        if is_p && skip > 0 {
            skip -= 1;
            let (vx, vy) = c.skip_mv(mb_x, mb_y);
            c.store_mv(mb_x * 4, mb_y * 4, 4, 4, vx, vy, 0);
            c.mc(bx, by, 16, 16, vx, vy);
            c.set_mb_mode(mb_x, mb_y, 2);
            c.clear_luma_nnz(mb_x, mb_y);
            c.clear_chroma_nnz(mb_x, mb_y);
            c.mbqp[mb] = c.qp;
            c.mbintra[mb] = false;
            mb += 1;
            continue;
        }
        pending = false;
        let mut mb_type = r.ue()? as usize;
        let inter = is_p && mb_type < 5;
        if is_p && !inter {
            mb_type -= 5;
        }
        if inter {
            c.set_mb_mode(mb_x, mb_y, 2);
            decode_inter_mb(c, &mut r, mb_x, mb_y, mb_type, num_ref)?;
            c.mbqp[mb] = c.qp;
            c.mbintra[mb] = false;
            mb += 1;
            continue;
        }
        // Intra macroblock.
        let is16 = mb_type >= 1;
        let mut i16mode = 0u8;
        let cbp_luma;
        let cbp_chroma;
        let mut modes = [0u8; 16];
        let chroma_mode;
        if mb_type == 25 {
            r.byte_align();
            for yy in 0..16 {
                for xx in 0..16 {
                    c.y[(by + yy) * w + bx + xx] = r.u(8)? as i32;
                    c.decy[(by + yy) * w + bx + xx] = true;
                }
            }
            for (pl, plane, dec) in [(0usize, &mut c.cb, &mut c.decu), (1, &mut c.cr, &mut c.decv)] {
                let _ = pl;
                for yy in 0..8 {
                    for xx in 0..8 {
                        plane[(mb_y * 8 + yy) * cw + mb_x * 8 + xx] = r.u(8)? as i32;
                        dec[(mb_y * 8 + yy) * cw + mb_x * 8 + xx] = true;
                    }
                }
            }
            for yy in 0..4 {
                for xx in 0..4 {
                    c.nnz_y[(mb_y * 4 + yy) * ny4 + mb_x * 4 + xx] = 16;
                    c.mode_y[(mb_y * 4 + yy) * ny4 + mb_x * 4 + xx] = 2;
                }
            }
            c.mbqp[mb] = c.qp;
            c.mbintra[mb] = true;
            mb += 1;
            continue;
        } else if is16 {
            let mt = mb_type - 1;
            i16mode = (mt % 4) as u8;
            cbp_chroma = (mt / 4) % 3;
            cbp_luma = if mt >= 12 { 15 } else { 0 };
            c.set_mb_mode(mb_x, mb_y, 2);
            chroma_mode = r.ue()? as u8;
        } else {
            for b in 0..16 {
                let bxx = mb_x * 4 + BLK_XY[b].0;
                let byy = mb_y * 4 + BLK_XY[b].1;
                let la = if bxx > 0 && c.sl(c.mb4(bxx - 1, byy)) { c.mode_y[byy * ny4 + bxx - 1] } else { -1 };
                let ta = if byy > 0 && c.sl(c.mb4(bxx, byy - 1)) { c.mode_y[(byy - 1) * ny4 + bxx] } else { -1 };
                let pm = if la < 0 || ta < 0 { 2 } else { la.min(ta) };
                let m = if r.bit()? == 1 {
                    pm
                } else {
                    let rem = r.u(3)? as i32;
                    if rem < pm {
                        rem
                    } else {
                        rem + 1
                    }
                };
                modes[b] = m as u8;
                c.mode_y[byy * ny4 + bxx] = m;
            }
            chroma_mode = r.ue()? as u8;
            let cbp = CBP_INTRA[r.ue()? as usize] as usize;
            cbp_luma = cbp & 15;
            cbp_chroma = cbp >> 4;
        }
        if is16 || cbp_luma != 0 || cbp_chroma != 0 {
            c.qp = (c.qp + r.se()? + 52) % 52;
        }
        recon_intra_luma(c, &mut r, mb_x, mb_y, is16, i16mode, cbp_luma, &modes)?;
        c.recon_chroma(&mut r, mb_x, mb_y, cbp_chroma, chroma_mode, true)?;
        c.mbqp[mb] = c.qp;
        c.mbintra[mb] = true;
        mb += 1;
    }
    Ok(())
}

/// Thin wrapper: decode a single-slice frame (fixture tests / simple clips).
pub fn decode_slice(sps: &Sps, pps: &Pps, rbsp: &[u8], is_idr: bool, ref_frame: Option<&DecodedFrame>) -> Result<DecodedFrame, &'static str> {
    decode_access_unit(sps, pps, &[(rbsp.to_vec(), is_idr)], ref_frame)
}
pub fn decode_islice(sps: &Sps, pps: &Pps, rbsp: &[u8], is_idr: bool) -> Result<DecodedFrame, &'static str> {
    decode_slice(sps, pps, rbsp, is_idr, None)
}

fn decode_inter_mb(c: &mut Ctx, r: &mut BitReader, mb_x: usize, mb_y: usize, mb_type: usize, num_ref: u32) -> Result<(), &'static str> {
    let (bx, by) = (mb_x * 16, mb_y * 16);
    let parts: &[(usize, usize, usize, usize, u8)] = match mb_type {
        0 => &[(0, 0, 16, 16, 0)],
        1 => &[(0, 0, 16, 8, 1), (0, 8, 16, 8, 2)],
        2 => &[(0, 0, 8, 16, 3), (8, 0, 8, 16, 4)],
        _ => &[],
    };
    if mb_type < 3 {
        if num_ref > 1 {
            for _ in 0..parts.len() {
                let _ = r.u(1)?;
            }
        }
        for &(ox, oy, pw, ph, kind) in parts {
            let bx4 = mb_x * 4 + ox / 4;
            let by4 = mb_y * 4 + oy / 4;
            let (pw4, ph4) = (pw / 4, ph / 4);
            let (mpx, mpy) = c.predict_mv(bx4 as i32, by4 as i32, pw4 as i32, 0, kind);
            let vx = mpx + r.se()?;
            let vy = mpy + r.se()?;
            c.store_mv(bx4, by4, pw4, ph4, vx, vy, 0);
            c.mc(bx + ox, by + oy, pw, ph, vx, vy);
        }
    } else {
        let mut subs = [0usize; 4];
        for s in subs.iter_mut() {
            *s = r.ue()? as usize & 3;
        }
        if num_ref > 1 {
            for _ in 0..4 {
                let _ = r.u(1)?;
            }
        }
        for s8 in 0..4 {
            let s8x = (s8 % 2) * 8;
            let s8y = (s8 / 2) * 8;
            let (nsub, spw4, sph4) = SUB[subs[s8]];
            let (spw, sph) = (spw4 * 4, sph4 * 4);
            for sp in 0..nsub {
                let (sox, soy) = match subs[s8] {
                    0 => (0, 0),
                    1 => (0, sp * 4),
                    2 => (sp * 4, 0),
                    _ => ((sp % 2) * 4, (sp / 2) * 4),
                };
                let ox = s8x + sox;
                let oy = s8y + soy;
                let bx4 = mb_x * 4 + ox / 4;
                let by4 = mb_y * 4 + oy / 4;
                let (mpx, mpy) = c.predict_mv(bx4 as i32, by4 as i32, spw4 as i32, 0, 0);
                let vx = mpx + r.se()?;
                let vy = mpy + r.se()?;
                c.store_mv(bx4, by4, spw4, sph4, vx, vy, 0);
                c.mc(bx + ox, by + oy, spw, sph, vx, vy);
            }
        }
    }
    let cbp = CBP_INTER[r.ue()? as usize] as usize;
    let cbp_luma = cbp & 15;
    let cbp_chroma = cbp >> 4;
    if cbp_luma != 0 || cbp_chroma != 0 {
        c.qp = (c.qp + r.se()? + 52) % 52;
    }
    for b in 0..16 {
        let bxx = mb_x * 4 + BLK_XY[b].0;
        let byy = mb_y * 4 + BLK_XY[b].1;
        let ox = BLK_XY[b].0 * 4;
        let oy = BLK_XY[b].1 * 4;
        if cbp_luma & (1 << (b / 4)) != 0 {
            let nc = c.nc_l(bxx, byy);
            let (sc, tc) = cavlc::residual_block(r, 16, nc)?;
            let mut blk = transform::inverse_scan_4x4(&sc);
            transform::dequant_4x4(&mut blk, c.qp as u32, false);
            transform::idct_4x4(&mut blk);
            for yy in 0..4 {
                for xx in 0..4 {
                    let idx = (by + oy + yy) * c.w + bx + ox + xx;
                    c.y[idx] = clip8(c.y[idx] + blk[yy * 4 + xx]) as i32;
                }
            }
            c.nnz_y[byy * c.ny4 + bxx] = tc as i32;
        } else {
            c.nnz_y[byy * c.ny4 + bxx] = 0;
        }
    }
    c.recon_chroma(r, mb_x, mb_y, cbp_chroma, 0, false)
}

fn recon_intra_luma(c: &mut Ctx, r: &mut BitReader, mb_x: usize, mb_y: usize, is16: bool, i16mode: u8, cbp_luma: usize, modes: &[u8; 16]) -> Result<(), &'static str> {
    let (bx, by, w) = (mb_x * 16, mb_y * 16, c.w);
    if is16 {
        let nc = c.nc_l(mb_x * 4, mb_y * 4);
        let (dcsc, _) = cavlc::residual_block(r, 16, nc)?;
        let dcras = transform::inverse_scan_4x4(&dcsc);
        let mut coeffs = [[0i32; 16]; 16];
        for b in 0..16 {
            let bxx = mb_x * 4 + BLK_XY[b].0;
            let byy = mb_y * 4 + BLK_XY[b].1;
            if cbp_luma & (1 << (b / 4)) != 0 {
                let (ac, tc) = cavlc::residual_block(r, 15, c.nc_l(bxx, byy))?;
                let mut blk = [0i32; 16];
                blk[1..16].copy_from_slice(&ac[0..15]);
                coeffs[b] = transform::inverse_scan_4x4(&blk);
                c.nnz_y[byy * c.ny4 + bxx] = tc as i32;
            } else {
                c.nnz_y[byy * c.ny4 + bxx] = 0;
            }
        }
        let mut top = [0i32; 16];
        let mut left = [0i32; 16];
        let at = by > 0 && c.decy[(by - 1) * w + bx] && c.sl(c.mbp(bx, by - 1));
        let al = bx > 0 && c.decy[by * w + bx - 1] && c.sl(c.mbp(bx - 1, by));
        let corner = if bx > 0 && by > 0 && c.sl(c.mbp(bx - 1, by - 1)) { c.y[(by - 1) * w + bx - 1] } else { 0 };
        if at {
            for k in 0..16 {
                top[k] = c.y[(by - 1) * w + bx + k];
            }
        }
        if al {
            for k in 0..16 {
                left[k] = c.y[(by + k) * w + bx - 1];
            }
        }
        let pred = intra::intra16x16(i16mode, &top, &left, corner, at, al);
        let mut dcq = dcras;
        transform::luma_dc_transform(&mut dcq, c.qp as u32);
        for b in 0..16 {
            let ox = BLK_XY[b].0 * 4;
            let oy = BLK_XY[b].1 * 4;
            let mut blk = coeffs[b];
            blk[0] = dcq[BLK_XY[b].1 * 4 + BLK_XY[b].0];
            transform::dequant_4x4(&mut blk, c.qp as u32, true);
            transform::idct_4x4(&mut blk);
            for yy in 0..4 {
                for xx in 0..4 {
                    let idx = (by + oy + yy) * w + bx + ox + xx;
                    c.y[idx] = clip8(pred[(oy + yy) * 16 + ox + xx] + blk[yy * 4 + xx]) as i32;
                    c.decy[idx] = true;
                }
            }
        }
    } else {
        for b in 0..16 {
            let bxx = mb_x * 4 + BLK_XY[b].0;
            let byy = mb_y * 4 + BLK_XY[b].1;
            let ox = BLK_XY[b].0 * 4;
            let oy = BLK_XY[b].1 * 4;
            let px = bx + ox;
            let py = by + oy;
            let mut coeff = [0i32; 16];
            if cbp_luma & (1 << (b / 4)) != 0 {
                let (sc, tc) = cavlc::residual_block(r, 16, c.nc_l(bxx, byy))?;
                coeff = transform::inverse_scan_4x4(&sc);
                c.nnz_y[byy * c.ny4 + bxx] = tc as i32;
            } else {
                c.nnz_y[byy * c.ny4 + bxx] = 0;
            }
            let (top, left, corner, at, al, atr) = c.gather4(px, py);
            let pred = intra::intra4x4(modes[b], &top, &left, corner, at, al, atr);
            let mut blk = coeff;
            transform::dequant_4x4(&mut blk, c.qp as u32, false);
            transform::idct_4x4(&mut blk);
            for yy in 0..4 {
                for xx in 0..4 {
                    let idx = (py + yy) * w + px + xx;
                    c.y[idx] = clip8(pred[yy * 4 + xx] + blk[yy * 4 + xx]) as i32;
                    c.decy[idx] = true;
                }
            }
        }
    }
    Ok(())
}
