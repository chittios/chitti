//! MPEG-1/2/2.5 **Layer III** decoder — a faithful `no_std` Rust port of
//! [minimp3](https://github.com/lieff/minimp3) (CC0 / public domain), scalar
//! path only, Layer III only (Layer I/II frames are skipped). The numeric
//! tables live in [`super::mp3_tables`], generated verbatim from minimp3.h by
//! `tools/gen_mp3_tables.py`. Covers mono + stereo (MS and intensity),
//! long/short/mixed blocks, the bit reservoir, free-format streams, and the
//! MPEG-2/2.5 low-sample-rate profile.
//!
//! Pure function of its input: bytes in, mono S16 PCM out — no I/O, no
//! panics on malformed input (garbage is skipped by frame sync, a truncated
//! tail just ends the stream). The unit tests pin decoded output against
//! minimp3's own scalar decode of the same fixture files.

use super::mp3_tables as t;
use alloc::boxed::Box;
use alloc::vec::Vec;

pub const MAX_SAMPLES_PER_FRAME: usize = 1152 * 2;
const MAX_FREE_FORMAT_FRAME_SIZE: usize = 2304;
const MAX_FRAME_SYNC_MATCHES: usize = 10;
const MAX_L3_FRAME_PAYLOAD_BYTES: usize = MAX_FREE_FORMAT_FRAME_SIZE;
const MAX_BITRESERVOIR_BYTES: usize = 511;
const SHORT_BLOCK_TYPE: u8 = 2;
const STOP_BLOCK_TYPE: u8 = 3;
const HDR_SIZE: usize = 4;
// MAX_SCF = 255 + BITS_DEQUANTIZER_OUT*4 - 210 = 41 (BITS_DEQUANTIZER_OUT = -1);
// MAX_SCFI = (MAX_SCF + 3) & ~3.
const MAX_SCFI: i32 = 44;

// --- frame-header accessors (h = the 4 header bytes) -----------------------

fn hdr_is_mono(h: &[u8]) -> bool {
    h[3] & 0xC0 == 0xC0
}
fn hdr_is_ms_stereo(h: &[u8]) -> bool {
    h[3] & 0xE0 == 0x60
}
fn hdr_is_free_format(h: &[u8]) -> bool {
    h[2] & 0xF0 == 0
}
fn hdr_is_crc(h: &[u8]) -> bool {
    h[1] & 1 == 0
}
fn hdr_test_padding(h: &[u8]) -> bool {
    h[2] & 0x2 != 0
}
fn hdr_test_mpeg1(h: &[u8]) -> bool {
    h[1] & 0x8 != 0
}
fn hdr_test_not_mpeg25(h: &[u8]) -> bool {
    h[1] & 0x10 != 0
}
fn hdr_test_i_stereo(h: &[u8]) -> bool {
    h[3] & 0x10 != 0
}
fn hdr_test_ms_stereo(h: &[u8]) -> bool {
    h[3] & 0x20 != 0
}
fn hdr_get_layer(h: &[u8]) -> u8 {
    (h[1] >> 1) & 3
}
fn hdr_get_bitrate(h: &[u8]) -> u8 {
    h[2] >> 4
}
fn hdr_get_sample_rate(h: &[u8]) -> u8 {
    (h[2] >> 2) & 3
}
fn hdr_get_my_sample_rate(h: &[u8]) -> usize {
    let sr = hdr_get_sample_rate(h) as usize;
    sr + (((h[1] >> 3) & 1) as usize + ((h[1] >> 4) & 1) as usize) * 3
}
fn hdr_is_frame_576(h: &[u8]) -> bool {
    h[1] & 14 == 2
}
fn hdr_is_layer_1(h: &[u8]) -> bool {
    h[1] & 6 == 6
}

fn hdr_valid(h: &[u8]) -> bool {
    h[0] == 0xff
        && ((h[1] & 0xF0) == 0xf0 || (h[1] & 0xFE) == 0xe2)
        && hdr_get_layer(h) != 0
        && hdr_get_bitrate(h) != 15
        && hdr_get_sample_rate(h) != 3
}

fn hdr_compare(h1: &[u8], h2: &[u8]) -> bool {
    hdr_valid(h2)
        && ((h1[1] ^ h2[1]) & 0xFE) == 0
        && ((h1[2] ^ h2[2]) & 0x0C) == 0
        && !(hdr_is_free_format(h1) ^ hdr_is_free_format(h2))
}

fn hdr_bitrate_kbps(h: &[u8]) -> u32 {
    2 * t::HALFRATE[hdr_test_mpeg1(h) as usize][hdr_get_layer(h) as usize - 1][hdr_get_bitrate(h) as usize] as u32
}

fn hdr_sample_rate_hz(h: &[u8]) -> u32 {
    const HZ: [u32; 3] = [44100, 48000, 32000];
    HZ[hdr_get_sample_rate(h) as usize] >> (!hdr_test_mpeg1(h) as u32) >> (!hdr_test_not_mpeg25(h) as u32)
}

fn hdr_frame_samples(h: &[u8]) -> usize {
    if hdr_is_layer_1(h) {
        384
    } else {
        1152 >> (hdr_is_frame_576(h) as usize)
    }
}

fn hdr_frame_bytes(h: &[u8], free_format_size: i32) -> i32 {
    let frame_bytes = (hdr_frame_samples(h) as u32 * hdr_bitrate_kbps(h) * 125 / hdr_sample_rate_hz(h)) as i32;
    let frame_bytes = if hdr_is_layer_1(h) { frame_bytes & !3 } else { frame_bytes };
    if frame_bytes != 0 {
        frame_bytes
    } else {
        free_format_size
    }
}

fn hdr_padding(h: &[u8]) -> i32 {
    if hdr_test_padding(h) {
        if hdr_is_layer_1(h) {
            4
        } else {
            1
        }
    } else {
        0
    }
}

// --- bitstream --------------------------------------------------------------

/// Bit reader with minimp3's exact semantics: reading past the limit returns
/// 0 (the position still advances, so overrun is detectable via `pos`).
struct Bs<'a> {
    buf: &'a [u8],
    pos: i32,   // in bits
    limit: i32, // in bits
}

impl<'a> Bs<'a> {
    fn new(buf: &'a [u8], bytes: usize) -> Bs<'a> {
        Bs { buf, pos: 0, limit: bytes as i32 * 8 }
    }

    fn byte(&self, i: i32) -> u32 {
        *self.buf.get(i as usize).unwrap_or(&0) as u32
    }

    fn get_bits(&mut self, n: u32) -> u32 {
        let s = (self.pos & 7) as u32;
        let mut shl = n as i32 + s as i32;
        let mut p = self.pos >> 3;
        self.pos += n as i32;
        if self.pos > self.limit {
            return 0;
        }
        let mut cache: u32 = 0;
        let mut next = self.byte(p) & (255 >> s);
        p += 1;
        loop {
            shl -= 8;
            if shl <= 0 {
                break;
            }
            cache |= next << shl;
            next = self.byte(p);
            p += 1;
        }
        cache | (next >> -shl)
    }
}

// --- side info / granule ----------------------------------------------------

#[derive(Clone, Copy)]
struct GrInfo {
    sfbtab: &'static [u8],
    part_23_length: u16,
    big_values: u16,
    scalefac_compress: u16,
    global_gain: u8,
    block_type: u8,
    mixed_block_flag: u8,
    n_long_sfb: u8,
    n_short_sfb: u8,
    table_select: [u8; 3],
    region_count: [u8; 3],
    subblock_gain: [u8; 3],
    preflag: u8,
    scalefac_scale: u8,
    count1_table: u8,
    scfsi: u8,
}

impl GrInfo {
    const fn zero() -> GrInfo {
        GrInfo {
            sfbtab: &[],
            part_23_length: 0,
            big_values: 0,
            scalefac_compress: 0,
            global_gain: 0,
            block_type: 0,
            mixed_block_flag: 0,
            n_long_sfb: 0,
            n_short_sfb: 0,
            table_select: [0; 3],
            region_count: [0; 3],
            subblock_gain: [0; 3],
            preflag: 0,
            scalefac_scale: 0,
            count1_table: 0,
            scfsi: 0,
        }
    }
}

fn l3_read_side_info(bs: &mut Bs<'_>, gr: &mut [GrInfo; 4], hdr: &[u8]) -> i32 {
    let mut sr_idx = hdr_get_my_sample_rate(hdr);
    sr_idx -= (sr_idx != 0) as usize;
    let mut gr_count: usize = if hdr_is_mono(hdr) { 1 } else { 2 };
    let main_data_begin;
    let mut scfsi: i32;

    if hdr_test_mpeg1(hdr) {
        gr_count *= 2;
        main_data_begin = bs.get_bits(9) as i32;
        scfsi = bs.get_bits(7 + gr_count as u32) as i32;
    } else {
        main_data_begin = (bs.get_bits(8 + gr_count as u32) >> gr_count) as i32;
        scfsi = 0;
    }

    let mut part_23_sum: i32 = 0;
    for g in gr.iter_mut().take(gr_count) {
        if hdr_is_mono(hdr) {
            scfsi <<= 4;
        }
        g.part_23_length = bs.get_bits(12) as u16;
        part_23_sum += g.part_23_length as i32;
        g.big_values = bs.get_bits(9) as u16;
        if g.big_values > 288 {
            return -1;
        }
        g.global_gain = bs.get_bits(8) as u8;
        g.scalefac_compress = bs.get_bits(if hdr_test_mpeg1(hdr) { 4 } else { 9 }) as u16;
        g.sfbtab = &t::SCF_LONG[sr_idx];
        g.n_long_sfb = 22;
        g.n_short_sfb = 0;
        let tables: u32;
        if bs.get_bits(1) != 0 {
            g.block_type = bs.get_bits(2) as u8;
            if g.block_type == 0 {
                return -1;
            }
            g.mixed_block_flag = bs.get_bits(1) as u8;
            g.region_count[0] = 7;
            g.region_count[1] = 255;
            if g.block_type == SHORT_BLOCK_TYPE {
                scfsi &= 0x0F0F;
                if g.mixed_block_flag == 0 {
                    g.region_count[0] = 8;
                    g.sfbtab = &t::SCF_SHORT[sr_idx];
                    g.n_long_sfb = 0;
                    g.n_short_sfb = 39;
                } else {
                    g.sfbtab = &t::SCF_MIXED[sr_idx];
                    g.n_long_sfb = if hdr_test_mpeg1(hdr) { 8 } else { 6 };
                    g.n_short_sfb = 30;
                }
            }
            tables = bs.get_bits(10) << 5;
            g.subblock_gain[0] = bs.get_bits(3) as u8;
            g.subblock_gain[1] = bs.get_bits(3) as u8;
            g.subblock_gain[2] = bs.get_bits(3) as u8;
        } else {
            g.block_type = 0;
            g.mixed_block_flag = 0;
            tables = bs.get_bits(15);
            g.region_count[0] = bs.get_bits(4) as u8;
            g.region_count[1] = bs.get_bits(3) as u8;
            g.region_count[2] = 255;
        }
        g.table_select[0] = (tables >> 10) as u8;
        g.table_select[1] = ((tables >> 5) & 31) as u8;
        g.table_select[2] = (tables & 31) as u8;
        g.preflag = if hdr_test_mpeg1(hdr) { bs.get_bits(1) as u8 } else { (g.scalefac_compress >= 500) as u8 };
        g.scalefac_scale = bs.get_bits(1) as u8;
        g.count1_table = bs.get_bits(1) as u8;
        g.scfsi = ((scfsi >> 12) & 15) as u8;
        scfsi <<= 4;
    }

    if part_23_sum + bs.pos > bs.limit + main_data_begin * 8 {
        return -1;
    }
    main_data_begin
}

// --- scalefactors -------------------------------------------------------------

fn l3_read_scalefactors(scf: &mut [u8], ist_pos: &mut [u8], scf_size: &[u8; 4], scf_count: &[u8], bs: &mut Bs<'_>, mut scfsi: i32) {
    let mut at = 0usize;
    for i in 0..4 {
        let cnt = scf_count[i] as usize;
        if cnt == 0 {
            break;
        }
        if scfsi & 8 != 0 {
            let (a, b) = (at, at + cnt);
            scf[a..b].copy_from_slice(&ist_pos[a..b]);
        } else {
            let bits = scf_size[i] as u32;
            if bits == 0 {
                scf[at..at + cnt].fill(0);
                ist_pos[at..at + cnt].fill(0);
            } else {
                let max_scf: i32 = if scfsi < 0 { (1 << bits) - 1 } else { -1 };
                for k in 0..cnt {
                    let s = bs.get_bits(bits) as i32;
                    ist_pos[at + k] = if s == max_scf { 255 } else { s as u8 };
                    scf[at + k] = s as u8;
                }
            }
        }
        at += cnt;
        scfsi = scfsi.wrapping_mul(2);
    }
    scf[at] = 0;
    scf[at + 1] = 0;
    scf[at + 2] = 0;
}

fn l3_ldexp_q2(mut y: f32, mut exp_q2: i32) -> f32 {
    loop {
        let e = (30 * 4).min(exp_q2);
        y *= t::EXPFRAC[(e & 3) as usize] * (1 << 30 >> (e >> 2)) as f32;
        exp_q2 -= e;
        if exp_q2 <= 0 {
            return y;
        }
    }
}

fn l3_decode_scalefactors(hdr: &[u8], ist_pos: &mut [u8], bs: &mut Bs<'_>, gr: &GrInfo, scf: &mut [f32], ch: usize) {
    let mut scf_partition: &[u8] = &t::SCF_PARTITIONS[((gr.n_short_sfb != 0) as usize) + ((gr.n_long_sfb == 0) as usize)];
    let mut scf_size = [0u8; 4];
    let mut iscf = [0u8; 40];
    let scf_shift = gr.scalefac_scale as i32 + 1;
    let mut scfsi = gr.scfsi as i32;

    if hdr_test_mpeg1(hdr) {
        let part = t::SCFC_DECODE[gr.scalefac_compress as usize] as u8;
        scf_size[0] = part >> 2;
        scf_size[1] = part >> 2;
        scf_size[2] = part & 3;
        scf_size[3] = part & 3;
    } else {
        let ist = (hdr_test_i_stereo(hdr) && ch != 0) as i32;
        let mut sfc = (gr.scalefac_compress as i32) >> ist;
        let mut k = (ist * 3 * 4) as usize;
        while sfc >= 0 {
            let mut modprod = 1i32;
            for i in (0..4).rev() {
                scf_size[i] = ((sfc / modprod) % t::SCF_MOD[k + i] as i32) as u8;
                modprod *= t::SCF_MOD[k + i] as i32;
            }
            sfc -= modprod;
            k += 4;
        }
        scf_partition = &scf_partition[k..];
        scfsi = -16;
    }
    l3_read_scalefactors(&mut iscf, ist_pos, &scf_size, scf_partition, bs, scfsi);

    if gr.n_short_sfb != 0 {
        let sh = 3 - scf_shift;
        let base = gr.n_long_sfb as usize;
        let mut i = 0usize;
        while i < gr.n_short_sfb as usize {
            iscf[base + i] = iscf[base + i].wrapping_add(((gr.subblock_gain[0] as i32) << sh) as u8);
            iscf[base + i + 1] = iscf[base + i + 1].wrapping_add(((gr.subblock_gain[1] as i32) << sh) as u8);
            iscf[base + i + 2] = iscf[base + i + 2].wrapping_add(((gr.subblock_gain[2] as i32) << sh) as u8);
            i += 3;
        }
    } else if gr.preflag != 0 {
        for i in 0..10 {
            iscf[11 + i] = iscf[11 + i].wrapping_add(t::PREAMP[i]);
        }
    }

    let gain_exp = gr.global_gain as i32 + (-1) * 4 - 210 - if hdr_is_ms_stereo(hdr) { 2 } else { 0 };
    let gain = l3_ldexp_q2((1i64 << (MAX_SCFI / 4)) as f32, MAX_SCFI - gain_exp);
    for i in 0..(gr.n_long_sfb as usize + gr.n_short_sfb as usize) {
        scf[i] = l3_ldexp_q2(gain, (iscf[i] as i32) << scf_shift);
    }
}

fn l3_pow_43(mut x: i32) -> f32 {
    if x < 129 {
        return t::POW43[(16 + x) as usize];
    }
    let mut mult = 256;
    if x < 1024 {
        mult = 16;
        x <<= 3;
    }
    let sign = 2 * x & 64;
    let frac = ((x & 63) - sign) as f32 / ((x & !63) + sign) as f32;
    t::POW43[(16 + ((x + sign) >> 6)) as usize] * (1.0 + frac * ((4.0 / 3.0) + frac * (2.0 / 9.0))) * mult as f32
}

// --- huffman ------------------------------------------------------------------

fn l3_huffman(dst: &mut [f32], bs: &mut Bs<'_>, gr_info: &GrInfo, scf: &[f32], layer3gr_limit: i32) {
    let mut one = 0.0f32;
    let mut ireg = 0usize;
    let mut big_val_cnt = gr_info.big_values as i32;
    let mut sfb_i = 0usize; // cursor into gr_info.sfbtab
    let mut scf_i = 0usize; // cursor into scf
    let mut d = 0usize; // cursor into dst

    let mut ptr = (bs.pos / 8) as usize; // byte cursor into bs.buf
    let nextb = |p: usize| -> u32 { *bs.buf.get(p).unwrap_or(&0) as u32 };
    let mut bs_cache: u32 =
        (((nextb(ptr) * 256 + nextb(ptr + 1)) * 256 + nextb(ptr + 2)) * 256 + nextb(ptr + 3)) << (bs.pos & 7);
    let mut bs_sh: i32 = (bs.pos & 7) - 8;
    ptr += 4;

    macro_rules! peek {
        ($n:expr) => {
            (bs_cache >> (32 - $n)) as i32
        };
    }
    macro_rules! flush {
        ($n:expr) => {
            bs_cache = bs_cache.wrapping_shl($n as u32);
            bs_sh += $n;
        };
    }
    macro_rules! check {
        () => {
            while bs_sh >= 0 {
                bs_cache |= nextb(ptr) << bs_sh;
                ptr += 1;
                bs_sh -= 8;
            }
        };
    }

    while big_val_cnt > 0 {
        // A corrupt stream can run the region/band cursors off their tables
        // (C reads adjacent statics; we stop the granule instead).
        if ireg >= 3 {
            break;
        }
        let tab_num = gr_info.table_select[ireg] as usize;
        let mut sfb_cnt = gr_info.region_count[ireg] as i32;
        ireg += 1;
        let codebook = &t::HUFF_TABS[t::HUFF_TABINDEX[tab_num] as usize..];
        let linbits = t::HUFF_LINBITS[tab_num] as i32;
        loop {
            if sfb_i >= gr_info.sfbtab.len() || scf_i >= scf.len() || d + 2 * big_val_cnt.max(0) as usize > dst.len() {
                bs.pos = layer3gr_limit;
                return;
            }
            let np = (gr_info.sfbtab[sfb_i] / 2) as i32;
            sfb_i += 1;
            if np == 0 {
                bs.pos = layer3gr_limit;
                return;
            }
            let mut pairs_to_decode = big_val_cnt.min(np);
            one = scf[scf_i];
            scf_i += 1;
            loop {
                let mut w = 5;
                let mut leaf = codebook[peek!(w) as usize] as i32;
                while leaf < 0 {
                    flush!(w);
                    w = leaf & 7;
                    leaf = codebook[(peek!(w) - (leaf >> 3)) as usize] as i32;
                }
                flush!(leaf >> 8);

                for _ in 0..2 {
                    let lsb = leaf & 0x0F;
                    if lsb == 15 && linbits != 0 {
                        let big = lsb + peek!(linbits);
                        flush!(linbits);
                        check!();
                        dst[d] = one * l3_pow_43(big) * if (bs_cache as i32) < 0 { -1.0 } else { 1.0 };
                        flush!(1);
                    } else {
                        dst[d] = t::POW43[(16 + lsb - 16 * (bs_cache >> 31) as i32) as usize] * one;
                        flush!(if lsb != 0 { 1 } else { 0 });
                    }
                    d += 1;
                    leaf >>= 4;
                }
                check!();
                pairs_to_decode -= 1;
                if pairs_to_decode == 0 {
                    break;
                }
            }
            big_val_cnt -= np;
            sfb_cnt -= 1;
            if !(big_val_cnt > 0 && sfb_cnt >= 0) {
                break;
            }
        }
    }

    // count1 region: quads of ±1/0.
    let mut np = 1 - big_val_cnt;
    loop {
        let codebook_count1: &[u8] = if gr_info.count1_table != 0 { &t::HUFF_TAB33 } else { &t::HUFF_TAB32 };
        let mut leaf = codebook_count1[peek!(4) as usize] as i32;
        if leaf & 8 == 0 {
            leaf = codebook_count1[((leaf >> 3) + (bs_cache.wrapping_shl(4) >> (32 - (leaf & 3))) as i32) as usize] as i32;
        }
        flush!(leaf & 7);
        let bspos = (ptr as i32) * 8 - 24 + bs_sh;
        if bspos > layer3gr_limit || d + 4 > dst.len() {
            break;
        }
        macro_rules! reload_scalefactor {
            () => {
                np -= 1;
                if np == 0 {
                    if sfb_i >= gr_info.sfbtab.len() || scf_i >= scf.len() {
                        break;
                    }
                    np = (gr_info.sfbtab[sfb_i] / 2) as i32;
                    sfb_i += 1;
                    if np == 0 {
                        break;
                    }
                    one = scf[scf_i];
                    scf_i += 1;
                }
            };
        }
        macro_rules! deq_count1 {
            ($s:expr) => {
                if leaf & (128 >> $s) != 0 {
                    dst[d + $s] = if (bs_cache as i32) < 0 { -one } else { one };
                    flush!(1);
                }
            };
        }
        reload_scalefactor!();
        deq_count1!(0);
        deq_count1!(1);
        reload_scalefactor!();
        deq_count1!(2);
        deq_count1!(3);
        check!();
        d += 4;
    }

    bs.pos = layer3gr_limit;
}

// --- stereo -------------------------------------------------------------------

/// `buf` is both channels: left `[0..576]`, right `[576..1152]`.
fn l3_midside_stereo(buf: &mut [f32], at: usize, n: usize) {
    for i in 0..n {
        let a = buf[at + i];
        let b = buf[at + 576 + i];
        buf[at + i] = a + b;
        buf[at + 576 + i] = a - b;
    }
}

fn l3_intensity_stereo_band(buf: &mut [f32], at: usize, n: usize, kl: f32, kr: f32) {
    for i in 0..n {
        buf[at + 576 + i] = buf[at + i] * kr;
        buf[at + i] *= kl;
    }
}

fn l3_stereo_top_band(right: &[f32], sfb: &[u8], nbands: usize, max_band: &mut [i32; 3]) {
    *max_band = [-1, -1, -1];
    let mut at = 0usize;
    for i in 0..nbands {
        let len = sfb[i] as usize;
        let mut k = 0;
        while k < len {
            if right[at + k] != 0.0 || right[at + k + 1] != 0.0 {
                max_band[i % 3] = i as i32;
                break;
            }
            k += 2;
        }
        at += len;
    }
}

fn l3_stereo_process(buf: &mut [f32], ist_pos: &[u8], sfb: &[u8], hdr: &[u8], max_band: &[i32; 3], mpeg2_sh: i32) {
    let max_pos: u32 = if hdr_test_mpeg1(hdr) { 7 } else { 64 };
    let mut at = 0usize;
    let mut i = 0usize;
    while sfb[i] != 0 {
        let ipos = ist_pos[i] as u32;
        if (i as i32) > max_band[i % 3] && ipos < max_pos {
            let s = if hdr_test_ms_stereo(hdr) { 1.41421356f32 } else { 1.0 };
            let (kl, kr);
            if hdr_test_mpeg1(hdr) {
                kl = t::PAN[2 * ipos as usize];
                kr = t::PAN[2 * ipos as usize + 1];
            } else {
                let mut l = 1.0f32;
                let mut r = l3_ldexp_q2(1.0, ((ipos as i32 + 1) >> 1) << mpeg2_sh);
                if ipos & 1 != 0 {
                    l = r;
                    r = 1.0;
                }
                kl = l;
                kr = r;
            }
            l3_intensity_stereo_band(buf, at, sfb[i] as usize, kl * s, kr * s);
        } else if hdr_test_ms_stereo(hdr) {
            l3_midside_stereo(buf, at, sfb[i] as usize);
        }
        at += sfb[i] as usize;
        i += 1;
    }
}

fn l3_intensity_stereo(buf: &mut [f32], ist_pos: &mut [u8], gr: &[GrInfo], hdr: &[u8]) {
    let n_sfb = gr[0].n_long_sfb as usize + gr[0].n_short_sfb as usize;
    let max_blocks = if gr[0].n_short_sfb != 0 { 3 } else { 1 };
    let mut max_band = [0i32; 3];
    l3_stereo_top_band(&buf[576..], gr[0].sfbtab, n_sfb, &mut max_band);
    if gr[0].n_long_sfb != 0 {
        let m = max_band[0].max(max_band[1]).max(max_band[2]);
        max_band = [m, m, m];
    }
    for i in 0..max_blocks {
        let default_pos = if hdr_test_mpeg1(hdr) { 3 } else { 0 };
        let itop = n_sfb - max_blocks + i;
        let prev = itop as i32 - max_blocks as i32;
        ist_pos[itop] = if max_band[i] >= prev { default_pos } else { ist_pos[prev as usize] };
    }
    l3_stereo_process(buf, ist_pos, gr[0].sfbtab, hdr, &max_band, (gr[1].scalefac_compress & 1) as i32);
}

// --- reorder / antialias / imdct ------------------------------------------------

fn l3_reorder(grbuf: &mut [f32], scratch: &mut [f32], sfb: &[u8]) {
    let mut src = 0usize;
    let mut dst = 0usize;
    let mut si = 0usize;
    loop {
        let len = sfb[si] as usize;
        if len == 0 {
            break;
        }
        for i in 0..len {
            scratch[dst] = grbuf[src + i];
            dst += 1;
            scratch[dst] = grbuf[src + i + len];
            dst += 1;
            scratch[dst] = grbuf[src + i + 2 * len];
            dst += 1;
        }
        si += 3;
        src += 3 * len;
    }
    grbuf[..dst].copy_from_slice(&scratch[..dst]);
}

fn l3_antialias(grbuf: &mut [f32], mut at: usize, mut nbands: i32) {
    while nbands > 0 {
        for i in 0..8 {
            let u = grbuf[at + 18 + i];
            let d = grbuf[at + 17 - i];
            grbuf[at + 18 + i] = u * t::AA[0][i] - d * t::AA[1][i];
            grbuf[at + 17 - i] = u * t::AA[1][i] + d * t::AA[0][i];
        }
        nbands -= 1;
        at += 18;
    }
}

fn l3_dct3_9(y: &mut [f32; 9]) {
    let (mut s0, s1, mut s2, s3, mut s4, s5, mut s6, s7, mut s8);
    s0 = y[0];
    s2 = y[2];
    s4 = y[4];
    s6 = y[6];
    s8 = y[8];
    let t0 = s0 + s6 * 0.5;
    s0 -= s6;
    let t4 = (s4 + s2) * 0.93969262;
    let t2 = (s8 + s2) * 0.76604444;
    s6 = (s4 - s8) * 0.17364818;
    s4 += s8 - s2;

    s2 = s0 - s4 * 0.5;
    y[4] = s4 + s0;
    s8 = t0 - t2 + s6;
    s0 = t0 - t4 + t2;
    s4 = t0 + t4 - s6;

    s1 = y[1];
    s3 = y[3] * 0.86602540;
    s5 = y[5];
    s7 = y[7];

    let t0 = (s5 + s1) * 0.98480775;
    let t4 = (s5 - s7) * 0.34202014;
    let t2 = (s1 + s7) * 0.64278761;
    let s1n = (s1 - s5 - s7) * 0.86602540;

    let s5n = t0 - s3 - t2;
    let s7n = t4 - s3 - t0;
    let s3n = t4 + s3 - t2;

    y[0] = s4 - s7n;
    y[1] = s2 + s1n;
    y[2] = s0 - s3n;
    y[3] = s8 + s5n;
    y[5] = s8 - s5n;
    y[6] = s0 + s3n;
    y[7] = s2 - s1n;
    y[8] = s4 + s7n;
}

fn l3_imdct36(grbuf: &mut [f32], mut gat: usize, overlap: &mut [f32], mut oat: usize, window: &[f32; 18], nbands: usize) {
    for _ in 0..nbands {
        let mut co = [0f32; 9];
        let mut si = [0f32; 9];
        co[0] = -grbuf[gat];
        si[0] = grbuf[gat + 17];
        for i in 0..4 {
            si[8 - 2 * i] = grbuf[gat + 4 * i + 1] - grbuf[gat + 4 * i + 2];
            co[1 + 2 * i] = grbuf[gat + 4 * i + 1] + grbuf[gat + 4 * i + 2];
            si[7 - 2 * i] = grbuf[gat + 4 * i + 4] - grbuf[gat + 4 * i + 3];
            co[2 + 2 * i] = -(grbuf[gat + 4 * i + 3] + grbuf[gat + 4 * i + 4]);
        }
        l3_dct3_9(&mut co);
        l3_dct3_9(&mut si);

        si[1] = -si[1];
        si[3] = -si[3];
        si[5] = -si[5];
        si[7] = -si[7];

        for i in 0..9 {
            let ovl = overlap[oat + i];
            let sum = co[i] * t::TWID9[9 + i] + si[i] * t::TWID9[i];
            overlap[oat + i] = co[i] * t::TWID9[i] - si[i] * t::TWID9[9 + i];
            grbuf[gat + i] = ovl * window[i] - sum * window[9 + i];
            grbuf[gat + 17 - i] = ovl * window[9 + i] + sum * window[i];
        }
        gat += 18;
        oat += 9;
    }
}

fn l3_idct3(x0: f32, x1: f32, x2: f32, dst: &mut [f32; 3]) {
    let m1 = x1 * 0.86602540;
    let a1 = x0 - x2 * 0.5;
    dst[1] = x0 + x2;
    dst[0] = a1 + m1;
    dst[2] = a1 - m1;
}

fn l3_imdct12(x: &[f32], dst: &mut [f32], dat: usize, overlap: &mut [f32], oat: usize) {
    let mut co = [0f32; 3];
    let mut si = [0f32; 3];
    l3_idct3(-x[0], x[6] + x[3], x[12] + x[9], &mut co);
    l3_idct3(x[15], x[12] - x[9], x[6] - x[3], &mut si);
    si[1] = -si[1];
    for i in 0..3 {
        let ovl = overlap[oat + i];
        let sum = co[i] * t::TWID3[3 + i] + si[i] * t::TWID3[i];
        overlap[oat + i] = co[i] * t::TWID3[i] - si[i] * t::TWID3[3 + i];
        dst[dat + i] = ovl * t::TWID3[2 - i] - sum * t::TWID3[5 - i];
        dst[dat + 5 - i] = ovl * t::TWID3[5 - i] + sum * t::TWID3[2 - i];
    }
}

fn l3_imdct_short(grbuf: &mut [f32], mut gat: usize, overlap: &mut [f32], mut oat: usize, nbands: usize) {
    for _ in 0..nbands {
        let mut tmp = [0f32; 18];
        tmp.copy_from_slice(&grbuf[gat..gat + 18]);
        grbuf[gat..gat + 6].copy_from_slice(&overlap[oat..oat + 6]);
        l3_imdct12(&tmp, grbuf, gat + 6, overlap, oat + 6);
        l3_imdct12(&tmp[1..], grbuf, gat + 12, overlap, oat + 6);
        // The third 12-point IMDCT writes into the overlap itself.
        {
            let mut co = [0f32; 3];
            let mut si = [0f32; 3];
            let x = &tmp[2..];
            l3_idct3(-x[0], x[6] + x[3], x[12] + x[9], &mut co);
            l3_idct3(x[15], x[12] - x[9], x[6] - x[3], &mut si);
            si[1] = -si[1];
            for i in 0..3 {
                let ovl = overlap[oat + 6 + i];
                let sum = co[i] * t::TWID3[3 + i] + si[i] * t::TWID3[i];
                overlap[oat + 6 + i] = co[i] * t::TWID3[i] - si[i] * t::TWID3[3 + i];
                overlap[oat + i] = ovl * t::TWID3[2 - i] - sum * t::TWID3[5 - i];
                overlap[oat + 5 - i] = ovl * t::TWID3[5 - i] + sum * t::TWID3[2 - i];
            }
        }
        oat += 9;
        gat += 18;
    }
}

fn l3_change_sign(grbuf: &mut [f32]) {
    let mut at = 18;
    let mut b = 0;
    while b < 32 {
        let mut i = 1;
        while i < 18 {
            grbuf[at + i] = -grbuf[at + i];
            i += 2;
        }
        b += 2;
        at += 36;
    }
}

fn l3_imdct_gr(grbuf: &mut [f32], overlap: &mut [f32], block_type: u8, n_long_bands: usize) {
    let mut gat = 0usize;
    let mut oat = 0usize;
    if n_long_bands != 0 {
        l3_imdct36(grbuf, gat, overlap, oat, &t::MDCT_WINDOW[0], n_long_bands);
        gat += 18 * n_long_bands;
        oat += 9 * n_long_bands;
    }
    if block_type == SHORT_BLOCK_TYPE {
        l3_imdct_short(grbuf, gat, overlap, oat, 32 - n_long_bands);
    } else {
        l3_imdct36(grbuf, gat, overlap, oat, &t::MDCT_WINDOW[(block_type == STOP_BLOCK_TYPE) as usize], 32 - n_long_bands);
    }
}

// --- reservoir ------------------------------------------------------------------

/// Per-decode working memory (~12 KiB): the assembled main-data (reservoir +
/// frame payload), side info, granule buffers, and the synthesis scratch.
pub struct Scratch {
    maindata: [u8; MAX_BITRESERVOIR_BYTES + MAX_L3_FRAME_PAYLOAD_BYTES],
    maindata_len: usize,
    bs_pos: i32, // bit position within maindata
    gr_info: [GrInfo; 4],
    grbuf: [f32; 2 * 576],
    scf: [f32; 40],
    syn: [f32; 33 * 64],
    ist_pos: [[u8; 39]; 2],
}

impl Scratch {
    fn new() -> Box<Scratch> {
        Box::new(Scratch {
            maindata: [0; MAX_BITRESERVOIR_BYTES + MAX_L3_FRAME_PAYLOAD_BYTES],
            maindata_len: 0,
            bs_pos: 0,
            gr_info: [GrInfo::zero(); 4],
            grbuf: [0.0; 2 * 576],
            scf: [0.0; 40],
            syn: [0.0; 33 * 64],
            ist_pos: [[0; 39]; 2],
        })
    }
}

/// Persistent decoder state across frames (bit reservoir + overlap + QMF).
pub struct Mp3Dec {
    mdct_overlap: [[f32; 9 * 32]; 2],
    qmf_state: [f32; 15 * 64],
    reserv: i32,
    free_format_bytes: i32,
    header: [u8; 4],
    reserv_buf: [u8; MAX_BITRESERVOIR_BYTES],
}

impl Mp3Dec {
    pub fn new() -> Box<Mp3Dec> {
        Box::new(Mp3Dec {
            mdct_overlap: [[0.0; 9 * 32]; 2],
            qmf_state: [0.0; 15 * 64],
            reserv: 0,
            free_format_bytes: 0,
            header: [0; 4],
            reserv_buf: [0; MAX_BITRESERVOIR_BYTES],
        })
    }

    fn reset(&mut self) {
        self.mdct_overlap = [[0.0; 9 * 32]; 2];
        self.qmf_state = [0.0; 15 * 64];
        self.reserv = 0;
        self.free_format_bytes = 0;
        self.header = [0; 4];
    }
}

fn l3_save_reservoir(h: &mut Mp3Dec, s: &mut Scratch) {
    let mut pos = ((s.bs_pos + 7) / 8) as usize;
    let mut remains = s.maindata_len as i32 - pos as i32;
    if remains > MAX_BITRESERVOIR_BYTES as i32 {
        pos += (remains - MAX_BITRESERVOIR_BYTES as i32) as usize;
        remains = MAX_BITRESERVOIR_BYTES as i32;
    }
    if remains > 0 {
        h.reserv_buf[..remains as usize].copy_from_slice(&s.maindata[pos..pos + remains as usize]);
    }
    h.reserv = remains.max(0);
}

fn l3_restore_reservoir(h: &mut Mp3Dec, frame: &[u8], frame_pos_bits: i32, s: &mut Scratch, main_data_begin: i32) -> bool {
    let frame_bytes = ((frame.len() as i32 * 8 - frame_pos_bits) / 8).max(0) as usize;
    let bytes_have = h.reserv.min(main_data_begin).max(0) as usize;
    let from = (h.reserv - main_data_begin).max(0) as usize;
    s.maindata[..bytes_have].copy_from_slice(&h.reserv_buf[from..from + bytes_have]);
    let at = (frame_pos_bits / 8) as usize;
    s.maindata[bytes_have..bytes_have + frame_bytes].copy_from_slice(&frame[at..at + frame_bytes]);
    s.maindata_len = bytes_have + frame_bytes;
    s.bs_pos = 0;
    h.reserv >= main_data_begin
}

// --- L3 decode ------------------------------------------------------------------

fn l3_decode(h: &mut Mp3Dec, s: &mut Scratch, gr_at: usize, nch: usize) {
    let hdr = h.header;
    for ch in 0..nch {
        let layer3gr_limit = s.bs_pos + s.gr_info[gr_at + ch].part_23_length as i32;
        let gr = s.gr_info[gr_at + ch];
        {
            let mut bs = Bs { buf: &s.maindata[..s.maindata_len], pos: s.bs_pos, limit: s.maindata_len as i32 * 8 };
            l3_decode_scalefactors(&hdr, &mut s.ist_pos[ch], &mut bs, &gr, &mut s.scf, ch);
            l3_huffman(&mut s.grbuf[ch * 576..(ch + 1) * 576], &mut bs, &gr, &s.scf, layer3gr_limit);
            s.bs_pos = bs.pos;
        }
    }

    if hdr_test_i_stereo(&hdr) {
        let (grbuf, gr_pair) = (&mut s.grbuf, &s.gr_info[gr_at..gr_at + 2]);
        let mut ist = s.ist_pos[1];
        l3_intensity_stereo(grbuf, &mut ist, gr_pair, &hdr);
        s.ist_pos[1] = ist;
    } else if hdr_is_ms_stereo(&hdr) {
        l3_midside_stereo(&mut s.grbuf, 0, 576);
    }

    for ch in 0..nch {
        let gr = s.gr_info[gr_at + ch];
        let mut aa_bands = 31i32;
        let n_long_bands = (if gr.mixed_block_flag != 0 { 2 } else { 0 }) << ((hdr_get_my_sample_rate(&hdr) == 2) as usize);

        if gr.n_short_sfb != 0 {
            aa_bands = n_long_bands as i32 - 1;
            let base = ch * 576 + n_long_bands * 18;
            let (grbuf, syn) = (&mut s.grbuf, &mut s.syn);
            l3_reorder(&mut grbuf[base..(ch + 1) * 576], syn, &gr.sfbtab[gr.n_long_sfb as usize..]);
        }

        l3_antialias(&mut s.grbuf, ch * 576, aa_bands);
        l3_imdct_gr(&mut s.grbuf[ch * 576..(ch + 1) * 576], &mut h.mdct_overlap[ch], gr.block_type, n_long_bands);
        l3_change_sign(&mut s.grbuf[ch * 576..(ch + 1) * 576]);
    }
}

// --- synthesis ---------------------------------------------------------------------

fn mp3d_dct_ii(grbuf: &mut [f32], at: usize, n: usize) {
    for k in 0..n {
        let y = at + k;
        let mut tt = [[0f32; 8]; 4];
        for i in 0..8 {
            let x0 = grbuf[y + i * 18];
            let x1 = grbuf[y + (15 - i) * 18];
            let x2 = grbuf[y + (16 + i) * 18];
            let x3 = grbuf[y + (31 - i) * 18];
            let t0 = x0 + x3;
            let t1 = x1 + x2;
            let t2 = (x1 - x2) * t::DCT2_SEC[3 * i];
            let t3 = (x0 - x3) * t::DCT2_SEC[3 * i + 1];
            tt[0][i] = t0 + t1;
            tt[1][i] = (t0 - t1) * t::DCT2_SEC[3 * i + 2];
            tt[2][i] = t3 + t2;
            tt[3][i] = (t3 - t2) * t::DCT2_SEC[3 * i + 2];
        }
        for x in tt.iter_mut() {
            let (mut x0, mut x1, mut x2, mut x3, mut x4, mut x5, mut x6, mut x7) =
                (x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]);
            let mut xt = x0 - x7;
            x0 += x7;
            x7 = x1 - x6;
            x1 += x6;
            x6 = x2 - x5;
            x2 += x5;
            x5 = x3 - x4;
            x3 += x4;
            x4 = x0 - x3;
            x0 += x3;
            x3 = x1 - x2;
            x1 += x2;
            x[0] = x0 + x1;
            x[4] = (x0 - x1) * 0.70710677;
            x5 += x6;
            x6 = (x6 + x7) * 0.70710677;
            x7 += xt;
            x3 = (x3 + x4) * 0.70710677;
            x5 -= x7 * 0.198912367; /* rotate by PI/8 */
            x7 += x5 * 0.382683432;
            x5 -= x7 * 0.198912367;
            x0 = xt - x6;
            xt += x6;
            x[1] = (xt + x7) * 0.50979561;
            x[2] = (x4 + x3) * 0.54119611;
            x[3] = (x0 - x5) * 0.60134488;
            x[5] = (x0 + x5) * 0.89997619;
            x[6] = (x4 - x3) * 1.30656302;
            x[7] = (xt - x7) * 2.56291556;
        }
        let mut yy = y;
        for i in 0..7 {
            grbuf[yy] = tt[0][i];
            grbuf[yy + 18] = tt[2][i] + tt[3][i] + tt[3][i + 1];
            grbuf[yy + 2 * 18] = tt[1][i] + tt[1][i + 1];
            grbuf[yy + 3 * 18] = tt[2][i + 1] + tt[3][i] + tt[3][i + 1];
            yy += 4 * 18;
        }
        grbuf[yy] = tt[0][7];
        grbuf[yy + 18] = tt[2][7] + tt[3][7];
        grbuf[yy + 2 * 18] = tt[1][7];
        grbuf[yy + 3 * 18] = tt[3][7];
    }
}

fn mp3d_scale_pcm(sample: f32) -> i16 {
    if sample >= 32766.5 {
        return 32767;
    }
    if sample <= -32767.5 {
        return -32768;
    }
    let mut s = (sample + 0.5) as i16;
    s -= (s < 0) as i16; /* away from zero, to be compliant */
    s
}

fn mp3d_synth_pair(pcm: &mut [i16], pat: usize, nch: usize, lins: &[f32], z: usize) {
    let mut a: f32;
    a = (lins[z + 14 * 64] - lins[z]) * 29.0;
    a += (lins[z + 64] + lins[z + 13 * 64]) * 213.0;
    a += (lins[z + 12 * 64] - lins[z + 2 * 64]) * 459.0;
    a += (lins[z + 3 * 64] + lins[z + 11 * 64]) * 2037.0;
    a += (lins[z + 10 * 64] - lins[z + 4 * 64]) * 5153.0;
    a += (lins[z + 5 * 64] + lins[z + 9 * 64]) * 6574.0;
    a += (lins[z + 8 * 64] - lins[z + 6 * 64]) * 37489.0;
    a += lins[z + 7 * 64] * 75038.0;
    pcm[pat] = mp3d_scale_pcm(a);

    let z = z + 2;
    a = lins[z + 14 * 64] * 104.0;
    a += lins[z + 12 * 64] * 1567.0;
    a += lins[z + 10 * 64] * 9727.0;
    a += lins[z + 8 * 64] * 64019.0;
    a += lins[z + 6 * 64] * -9975.0;
    a += lins[z + 4 * 64] * -45.0;
    a += lins[z + 2 * 64] * 146.0;
    a += lins[z] * -5.0;
    pcm[pat + 16 * nch] = mp3d_scale_pcm(a);
}

/// One 32-sample synthesis step over two subbands: grbuf columns `xat`
/// (left) / `xat + 576*(nch-1)` (right), writing 64 interleaved samples.
fn mp3d_synth(grbuf: &[f32], xat: usize, pcm: &mut [i16], pat: usize, nch: usize, lins: &mut [f32], lat: usize) {
    let xr = xat + 576 * (nch - 1);
    let dstr = pat + (nch - 1); // pcm right-channel offset
    let dstl = pat;

    let zlin = lat + 15 * 64;
    lins[zlin + 4 * 15] = grbuf[xat + 18 * 16];
    lins[zlin + 4 * 15 + 1] = grbuf[xr + 18 * 16];
    lins[zlin + 4 * 15 + 2] = grbuf[xat];
    lins[zlin + 4 * 15 + 3] = grbuf[xr];

    lins[zlin + 4 * 31] = grbuf[xat + 1 + 18 * 16];
    lins[zlin + 4 * 31 + 1] = grbuf[xr + 1 + 18 * 16];
    lins[zlin + 4 * 31 + 2] = grbuf[xat + 1];
    lins[zlin + 4 * 31 + 3] = grbuf[xr + 1];

    mp3d_synth_pair(pcm, dstr, nch, lins, lat + 4 * 15 + 1);
    mp3d_synth_pair(pcm, dstr + 32 * nch, nch, lins, lat + 4 * 15 + 64 + 1);
    mp3d_synth_pair(pcm, dstl, nch, lins, lat + 4 * 15);
    mp3d_synth_pair(pcm, dstl + 32 * nch, nch, lins, lat + 4 * 15 + 64);

    let mut w = 0usize; // cursor into SYNTH_WIN
    for i in (0..15).rev() {
        let mut a = [0f32; 4];
        let mut b = [0f32; 4];

        lins[zlin + 4 * i] = grbuf[xat + 18 * (31 - i)];
        lins[zlin + 4 * i + 1] = grbuf[xr + 18 * (31 - i)];
        lins[zlin + 4 * i + 2] = grbuf[xat + 1 + 18 * (31 - i)];
        lins[zlin + 4 * i + 3] = grbuf[xr + 1 + 18 * (31 - i)];
        lins[zlin + 4 * (i + 16)] = grbuf[xat + 1 + 18 * (1 + i)];
        lins[zlin + 4 * (i + 16) + 1] = grbuf[xr + 1 + 18 * (1 + i)];
        // 4*(i - 16) + 2 == (4*i + 2) - 64, always >= 0 relative to `lat`.
        lins[zlin + 4 * i + 2 - 64] = grbuf[xat + 18 * (1 + i)];
        lins[zlin + 4 * i + 3 - 64] = grbuf[xr + 18 * (1 + i)];

        // S0/S1/S2 accumulation over k = 0..7 (V-window MAC ladder).
        for k in 0..8usize {
            let w0 = t::SYNTH_WIN[w];
            w += 1;
            let w1 = t::SYNTH_WIN[w];
            w += 1;
            let vz = zlin + 4 * i - k * 64;
            let vy = zlin + 4 * i - (15 - k) * 64;
            match k {
                0 => {
                    for j in 0..4 {
                        b[j] = lins[vz + j] * w1 + lins[vy + j] * w0;
                        a[j] = lins[vz + j] * w0 - lins[vy + j] * w1;
                    }
                }
                2 | 4 | 6 => {
                    for j in 0..4 {
                        b[j] += lins[vz + j] * w1 + lins[vy + j] * w0;
                        a[j] += lins[vz + j] * w0 - lins[vy + j] * w1;
                    }
                }
                _ => {
                    for j in 0..4 {
                        b[j] += lins[vz + j] * w1 + lins[vy + j] * w0;
                        a[j] += lins[vy + j] * w1 - lins[vz + j] * w0;
                    }
                }
            }
        }

        pcm[dstr + (15 - i) * nch] = mp3d_scale_pcm(a[1]);
        pcm[dstr + (17 + i) * nch] = mp3d_scale_pcm(b[1]);
        pcm[dstl + (15 - i) * nch] = mp3d_scale_pcm(a[0]);
        pcm[dstl + (17 + i) * nch] = mp3d_scale_pcm(b[0]);
        pcm[dstr + (47 - i) * nch] = mp3d_scale_pcm(a[3]);
        pcm[dstr + (49 + i) * nch] = mp3d_scale_pcm(b[3]);
        pcm[dstl + (47 - i) * nch] = mp3d_scale_pcm(a[2]);
        pcm[dstl + (49 + i) * nch] = mp3d_scale_pcm(b[2]);
    }
}

fn mp3d_synth_granule(qmf_state: &mut [f32; 15 * 64], grbuf: &mut [f32], nbands: usize, nch: usize, pcm: &mut [i16], pat: usize, lins: &mut [f32]) {
    for i in 0..nch {
        mp3d_dct_ii(grbuf, 576 * i, nbands);
    }

    lins[..15 * 64].copy_from_slice(qmf_state);

    let mut i = 0usize;
    while i < nbands {
        mp3d_synth(grbuf, i, pcm, pat + 32 * nch * i, nch, lins, i * 64);
        i += 2;
    }

    if nch == 1 {
        let mut i = 0usize;
        while i < 15 * 64 {
            qmf_state[i] = lins[nbands * 64 + i];
            i += 2;
        }
    } else {
        qmf_state.copy_from_slice(&lins[nbands * 64..nbands * 64 + 15 * 64]);
    }
}

// --- frame sync -------------------------------------------------------------------

fn mp3d_match_frame(hdr: &[u8], mp3_bytes: usize, frame_bytes: i32) -> bool {
    let mut i = 0usize;
    for nmatch in 0..MAX_FRAME_SYNC_MATCHES {
        i += (hdr_frame_bytes(&hdr[i..], frame_bytes) + hdr_padding(&hdr[i..])) as usize;
        if i + HDR_SIZE > mp3_bytes {
            return nmatch > 0;
        }
        if !hdr_compare(hdr, &hdr[i..]) {
            return false;
        }
    }
    true
}

fn mp3d_find_frame(mp3: &[u8], free_format_bytes: &mut i32, ptr_frame_bytes: &mut i32) -> usize {
    let mp3_bytes = mp3.len();
    let end = mp3_bytes.saturating_sub(HDR_SIZE);
    for i in 0..end {
        let h = &mp3[i..];
        if hdr_valid(h) {
            let mut frame_bytes = hdr_frame_bytes(h, *free_format_bytes);
            let mut frame_and_padding = frame_bytes + hdr_padding(h);

            let mut k = HDR_SIZE;
            while frame_bytes == 0 && k < MAX_FREE_FORMAT_FRAME_SIZE && i + 2 * k < mp3_bytes - HDR_SIZE {
                if hdr_compare(h, &h[k..]) {
                    let fb = k as i32 - hdr_padding(h);
                    let nextfb = fb + hdr_padding(&h[k..]);
                    if i + k + nextfb as usize + HDR_SIZE <= mp3_bytes && hdr_compare(h, &h[k + nextfb as usize..]) {
                        frame_and_padding = k as i32;
                        frame_bytes = fb;
                        *free_format_bytes = fb;
                    }
                }
                k += 1;
            }
            if (frame_bytes != 0
                && i + frame_and_padding as usize <= mp3_bytes
                && mp3d_match_frame(h, mp3_bytes - i, frame_bytes))
                || (i == 0 && frame_and_padding as usize == mp3_bytes)
            {
                *ptr_frame_bytes = frame_and_padding;
                return i;
            }
            *free_format_bytes = 0;
        }
    }
    *ptr_frame_bytes = 0;
    mp3_bytes
}

/// Frame metadata returned alongside decoded samples.
pub struct FrameInfo {
    pub frame_bytes: usize,
    pub channels: usize,
    pub hz: u32,
    pub layer: u8,
}

/// Decode one frame from the head of `mp3` into `pcm` (interleaved S16).
/// Returns samples-per-channel (0 = frame skipped/insufficient data; check
/// `info.frame_bytes` to advance).
pub fn decode_frame(dec: &mut Mp3Dec, scratch: &mut Scratch, mp3: &[u8], pcm: &mut [i16; MAX_SAMPLES_PER_FRAME], info: &mut FrameInfo) -> usize {
    let mp3_bytes = mp3.len();
    let mut i = 0usize;
    let mut frame_size: i32 = 0;

    if mp3_bytes > 4 && dec.header[0] == 0xff && hdr_compare(&dec.header, mp3) {
        frame_size = hdr_frame_bytes(mp3, dec.free_format_bytes) + hdr_padding(mp3);
        if frame_size != mp3_bytes as i32 && (frame_size + HDR_SIZE as i32 > mp3_bytes as i32 || !hdr_compare(mp3, &mp3[frame_size as usize..])) {
            frame_size = 0;
        }
    }
    if frame_size == 0 {
        dec.reset();
        i = mp3d_find_frame(mp3, &mut dec.free_format_bytes, &mut frame_size);
        if frame_size == 0 || i + frame_size as usize > mp3_bytes {
            info.frame_bytes = i;
            return 0;
        }
    }

    let hdr = &mp3[i..];
    dec.header.copy_from_slice(&hdr[..4]);
    info.frame_bytes = i + frame_size as usize;
    info.channels = if hdr_is_mono(hdr) { 1 } else { 2 };
    info.hz = hdr_sample_rate_hz(hdr);
    info.layer = 4 - hdr_get_layer(hdr);

    // Layer I/II: skip the frame (Layer III only, like MINIMP3_ONLY_MP3).
    if info.layer != 3 {
        return 0;
    }

    let frame = &hdr[..frame_size as usize];
    let mut bs_pos: i32 = HDR_SIZE as i32 * 8;
    if hdr_is_crc(hdr) {
        bs_pos += 16;
    }

    let main_data_begin = {
        let mut bs = Bs { buf: frame, pos: bs_pos, limit: frame_size * 8 };
        let r = l3_read_side_info(&mut bs, &mut scratch.gr_info, hdr);
        bs_pos = bs.pos;
        r
    };
    if main_data_begin < 0 || bs_pos > frame_size * 8 {
        dec.reset();
        return 0;
    }
    let success = l3_restore_reservoir(dec, frame, bs_pos, scratch, main_data_begin);
    let mut samples = 0usize;
    if success {
        let ngr = if hdr_test_mpeg1(hdr) { 2 } else { 1 };
        let nch = info.channels;
        for igr in 0..ngr {
            scratch.grbuf.fill(0.0);
            l3_decode(dec, scratch, igr * nch, nch);
            mp3d_synth_granule(&mut dec.qmf_state, &mut scratch.grbuf, 18, nch, pcm, igr * 576 * nch, &mut scratch.syn);
        }
        samples = hdr_frame_samples(&dec.header);
    }
    l3_save_reservoir(dec, scratch);
    if success {
        samples
    } else {
        0
    }
}

/// Decode a whole MP3 stream to mono S16 at the stream's sample rate
/// (stereo is downmixed `(l+r)/2`). ID3v2 tags are skipped; garbage between
/// frames is resynced past; a rate change mid-stream ends the decode.
pub fn decode(bytes: &[u8]) -> Result<super::Audio, &'static str> {
    // ID3v2: "ID3" + version(2) + flags + syncsafe u28 size.
    let mut at = 0usize;
    if bytes.len() > 10 && &bytes[0..3] == b"ID3" {
        let sz = ((bytes[6] as usize & 0x7f) << 21)
            | ((bytes[7] as usize & 0x7f) << 14)
            | ((bytes[8] as usize & 0x7f) << 7)
            | (bytes[9] as usize & 0x7f);
        at = (10 + sz).min(bytes.len());
    }

    let mut dec = Mp3Dec::new();
    let mut scratch = Scratch::new();
    let mut pcm = [0i16; MAX_SAMPLES_PER_FRAME];
    let mut out: Vec<i16> = Vec::new();
    let mut rate: u32 = 0;
    const MAX_SAMPLES: usize = 128 << 20; // ~48 min at 44.1 kHz — heap guard

    while at < bytes.len() {
        let mut info = FrameInfo { frame_bytes: 0, channels: 0, hz: 0, layer: 0 };
        let samples = decode_frame(&mut dec, &mut scratch, &bytes[at..], &mut pcm, &mut info);
        if info.frame_bytes == 0 {
            break; // no more frames found
        }
        at += info.frame_bytes;
        if samples == 0 {
            continue; // skipped frame (garbage, L1/L2, or reservoir priming)
        }
        if rate == 0 {
            rate = info.hz;
        } else if rate != info.hz {
            break; // rate change mid-stream: keep what we have
        }
        if out.len() + samples > MAX_SAMPLES {
            break;
        }
        if info.channels == 2 {
            out.extend(pcm[..samples * 2].chunks_exact(2).map(|c| ((c[0] as i32 + c[1] as i32) / 2) as i16));
        } else {
            out.extend_from_slice(&pcm[..samples]);
        }
    }

    if out.is_empty() || rate == 0 {
        return Err("mp3: no decodable Layer III frames");
    }
    Ok(super::Audio { rate, pcm: out })
}

/// Fresh per-decode scratch (public so `decode_frame` is testable directly).
pub fn new_scratch() -> Box<Scratch> {
    Scratch::new()
}

#[cfg(test)]
mod tests {
    use super::super::mp3_testdata as td;
    use super::*;

    /// Compare a decode against the pinned reference (minimp3 scalar output,
    /// stride-97 sampled), allowing ±2 for float association differences.
    fn check(bytes: &[u8], rate: u32, len: usize, ref97: &[i16]) {
        let a = decode(bytes).expect("decodes");
        assert_eq!(a.rate, rate);
        assert_eq!(a.pcm.len(), len);
        for (k, &want) in ref97.iter().enumerate() {
            let got = a.pcm[k * 97];
            assert!((got as i32 - want as i32).abs() <= 2, "sample {}: got {} want {}", k * 97, got, want);
        }
    }

    #[test_case]
    fn joint_stereo_short_blocks_match_reference() {
        // lame -b 96 -m j over a sweep + click train: MS stereo, long and
        // short blocks, bit reservoir all exercised.
        check(td::MP3_ST, td::MP3_ST_RATE, td::MP3_ST_LEN, td::MP3_ST_REF97);
    }

    #[test_case]
    fn mpeg2_lsf_16khz_matches_reference() {
        // MPEG-2 low-sample-rate profile (16 kHz mono): the LSF scalefactor path.
        check(td::MP3_M16, td::MP3_M16_RATE, td::MP3_M16_LEN, td::MP3_M16_REF97);
    }

    #[test_case]
    fn id3_and_garbage_are_skipped() {
        // ID3v2 header (syncsafe size 100) + junk + a real stream still decodes.
        let mut b = alloc::vec![0u8; 0];
        b.extend_from_slice(b"ID3\x04\x00\x00\x00\x00\x00\x64");
        b.extend_from_slice(&[0xAA; 100]);
        b.extend_from_slice(td::MP3_M16);
        let a = decode(&b).expect("decodes past ID3");
        assert_eq!(a.rate, td::MP3_M16_RATE);
        assert_eq!(a.pcm.len(), td::MP3_M16_LEN);
    }

    #[test_case]
    fn corrupt_input_errors_not_panics() {
        assert!(decode(&[0u8; 64]).is_err(), "no frames");
        assert!(decode(&td::MP3_ST[..40]).is_err(), "truncated head");
        // Bit-flip the middle of the stream: must not panic (frames resync).
        let mut bad = td::MP3_ST.to_vec();
        let n = bad.len();
        for i in (n / 3..2 * n / 3).step_by(7) {
            bad[i] ^= 0x5a;
        }
        let _ = decode(&bad); // Ok or Err both fine — just no panic
    }
}
