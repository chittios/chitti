//! CAVLC residual-block decoding (H.264 §9.2). The VLC tables are the
//! authoritative FFmpeg ones (parsed from `libavcodec/h264_cavlc.c` /
//! `h264data.c`, never hand-transcribed); the decode logic is validated
//! bit-exact against PyAV/ffmpeg in the host reference before this port.
//!
//! `residual_block` reads one block and returns its coefficients in **scan
//! order** (0 = lowest frequency) plus the total non-zero count (the `nnz`
//! the neighbour-context `nC` is built from). Pure + panic-free.

use super::super::bits::BitReader;

pub const COEFF_TOKEN_LEN: [[u8; 68]; 4] = [
    [1, 0, 0, 0, 6, 2, 0, 0, 8, 6, 3, 0, 9, 8, 7, 5, 10, 9, 8, 6, 11, 10, 9, 7, 13, 11, 10, 8, 13, 13, 11, 9, 13, 13, 13, 10, 14, 14, 13, 11, 14, 14, 14, 13, 15, 15, 14, 14, 15, 15, 15, 14, 16, 15, 15, 15, 16, 16, 16, 15, 16, 16, 16, 16, 16, 16, 16, 16],
    [2, 0, 0, 0, 6, 2, 0, 0, 6, 5, 3, 0, 7, 6, 6, 4, 8, 6, 6, 4, 8, 7, 7, 5, 9, 8, 8, 6, 11, 9, 9, 6, 11, 11, 11, 7, 12, 11, 11, 9, 12, 12, 12, 11, 12, 12, 12, 11, 13, 13, 13, 12, 13, 13, 13, 13, 13, 14, 13, 13, 14, 14, 14, 13, 14, 14, 14, 14],
    [4, 0, 0, 0, 6, 4, 0, 0, 6, 5, 4, 0, 6, 5, 5, 4, 7, 5, 5, 4, 7, 5, 5, 4, 7, 6, 6, 4, 7, 6, 6, 4, 8, 7, 7, 5, 8, 8, 7, 6, 9, 8, 8, 7, 9, 9, 8, 8, 9, 9, 9, 8, 10, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
    [6, 0, 0, 0, 6, 6, 0, 0, 6, 6, 6, 0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
];
pub const COEFF_TOKEN_BITS: [[u8; 68]; 4] = [
    [1, 0, 0, 0, 5, 1, 0, 0, 7, 4, 1, 0, 7, 6, 5, 3, 7, 6, 5, 3, 7, 6, 5, 4, 15, 6, 5, 4, 11, 14, 5, 4, 8, 10, 13, 4, 15, 14, 9, 4, 11, 10, 13, 12, 15, 14, 9, 12, 11, 10, 13, 8, 15, 1, 9, 12, 11, 14, 13, 8, 7, 10, 9, 12, 4, 6, 5, 8],
    [3, 0, 0, 0, 11, 2, 0, 0, 7, 7, 3, 0, 7, 10, 9, 5, 7, 6, 5, 4, 4, 6, 5, 6, 7, 6, 5, 8, 15, 6, 5, 4, 11, 14, 13, 4, 15, 10, 9, 4, 11, 14, 13, 12, 8, 10, 9, 8, 15, 14, 13, 12, 11, 10, 9, 12, 7, 11, 6, 8, 9, 8, 10, 1, 7, 6, 5, 4],
    [15, 0, 0, 0, 15, 14, 0, 0, 11, 15, 13, 0, 8, 12, 14, 12, 15, 10, 11, 11, 11, 8, 9, 10, 9, 14, 13, 9, 8, 10, 9, 8, 15, 14, 13, 13, 11, 14, 10, 12, 15, 10, 13, 12, 11, 14, 9, 12, 8, 10, 13, 8, 13, 7, 9, 12, 9, 12, 11, 10, 5, 8, 7, 6, 1, 4, 3, 2],
    [3, 0, 0, 0, 0, 1, 0, 0, 4, 5, 6, 0, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63],
];
pub const CHROMA_DC_CT_LEN: [u8; 20] = [2, 0, 0, 0, 6, 1, 0, 0, 6, 6, 3, 0, 6, 7, 7, 6, 6, 8, 8, 7];
pub const CHROMA_DC_CT_BITS: [u8; 20] = [1, 0, 0, 0, 7, 1, 0, 0, 4, 6, 1, 0, 3, 3, 2, 5, 2, 3, 2, 0];
pub const TOTAL_ZEROS_LEN: [[u8; 16]; 15] = [
    [1, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 9],
    [3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6, 6, 6, 6, 0],
    [4, 3, 3, 3, 4, 4, 3, 3, 4, 5, 5, 6, 5, 6, 0, 0],
    [5, 3, 4, 4, 3, 3, 3, 4, 3, 4, 5, 5, 5, 0, 0, 0],
    [4, 4, 4, 3, 3, 3, 3, 3, 4, 5, 4, 5, 0, 0, 0, 0],
    [6, 5, 3, 3, 3, 3, 3, 3, 4, 3, 6, 0, 0, 0, 0, 0],
    [6, 5, 3, 3, 3, 2, 3, 4, 3, 6, 0, 0, 0, 0, 0, 0],
    [6, 4, 5, 3, 2, 2, 3, 3, 6, 0, 0, 0, 0, 0, 0, 0],
    [6, 6, 4, 2, 2, 3, 2, 5, 0, 0, 0, 0, 0, 0, 0, 0],
    [5, 5, 3, 2, 2, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [4, 4, 3, 3, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [4, 4, 2, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 3, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
];
pub const TOTAL_ZEROS_BITS: [[u8; 16]; 15] = [
    [1, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 1],
    [7, 6, 5, 4, 3, 5, 4, 3, 2, 3, 2, 3, 2, 1, 0, 0],
    [5, 7, 6, 5, 4, 3, 4, 3, 2, 3, 2, 1, 1, 0, 0, 0],
    [3, 7, 5, 4, 6, 5, 4, 3, 3, 2, 2, 1, 0, 0, 0, 0],
    [5, 4, 3, 7, 6, 5, 4, 3, 2, 1, 1, 0, 0, 0, 0, 0],
    [1, 1, 7, 6, 5, 4, 3, 2, 1, 1, 0, 0, 0, 0, 0, 0],
    [1, 1, 5, 4, 3, 3, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 3, 3, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 3, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 3, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 2, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
];
pub const CHROMA_DC_TZ_LEN: [[u8; 4]; 3] = [
    [1, 2, 3, 3],
    [1, 2, 2, 0],
    [1, 1, 0, 0],
];
pub const CHROMA_DC_TZ_BITS: [[u8; 4]; 3] = [
    [1, 1, 1, 0],
    [1, 1, 0, 0],
    [1, 0, 0, 0],
];
pub const RUN_LEN: [[u8; 16]; 7] = [
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 2, 2, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 2, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 3, 3, 3, 3, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0],
];
pub const RUN_BITS: [[u8; 16]; 7] = [
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 2, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 0, 1, 3, 2, 5, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [7, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
];

/// Read one prefix-free VLC codeword by matching accumulated bits against the
/// `(len, code)` table rows. `len == 0` entries are padding and never match.
fn read_vlc(r: &mut BitReader, lens: &[u8], bits: &[u8]) -> Result<usize, &'static str> {
    let mut code: u32 = 0;
    let mut l: u32 = 0;
    while l < 24 {
        code = (code << 1) | r.bit()?;
        l += 1;
        for idx in 0..lens.len() {
            if lens[idx] as u32 == l && bits[idx] as u32 == code {
                return Ok(idx);
            }
        }
    }
    Err("cavlc: vlc decode failed")
}

/// Decode a residual block: `max_coeff` coefficients (16 luma 4×4, 15 for the
/// AC of an Intra_16x16 or a chroma AC block, 4 for chroma DC) with neighbour
/// context `n_c` (`-1` selects the chroma-DC table). Returns the coefficients in
/// **scan order** and the total non-zero count.
pub fn residual_block(
    r: &mut BitReader,
    max_coeff: usize,
    n_c: i32,
) -> Result<([i32; 16], usize), &'static str> {
    let mut coeffs = [0i32; 16];
    let tt = if n_c == -1 {
        read_vlc(r, &CHROMA_DC_CT_LEN, &CHROMA_DC_CT_BITS)?
    } else {
        let tab = if n_c < 2 {
            0
        } else if n_c < 4 {
            1
        } else if n_c < 8 {
            2
        } else {
            3
        };
        read_vlc(r, &COEFF_TOKEN_LEN[tab], &COEFF_TOKEN_BITS[tab])?
    };
    let total_coeff = tt >> 2;
    let trailing_ones = tt & 3;
    if total_coeff == 0 {
        return Ok((coeffs, 0));
    }
    if total_coeff > max_coeff {
        return Err("cavlc: total_coeff exceeds max");
    }
    let mut level = [0i32; 16];
    for lv in level.iter_mut().take(trailing_ones) {
        *lv = 1 - 2 * (r.bit()? as i32);
    }
    let mut suffix_length: u32 = if total_coeff > 10 && trailing_ones < 3 { 1 } else { 0 };
    for i in trailing_ones..total_coeff {
        let mut level_prefix: u32 = 0;
        while r.bit()? == 0 {
            level_prefix += 1;
            if level_prefix > 60 {
                return Err("cavlc: level_prefix overflow");
            }
        }
        let level_suffix_size = if level_prefix == 14 && suffix_length == 0 {
            4
        } else if level_prefix >= 15 {
            level_prefix - 3
        } else {
            suffix_length
        };
        let level_suffix = if level_suffix_size > 0 { r.u(level_suffix_size)? as i32 } else { 0 };
        let mut level_code = ((level_prefix.min(15) << suffix_length) as i32) + level_suffix;
        if level_prefix >= 15 && suffix_length == 0 {
            level_code += 15;
        }
        if level_prefix >= 16 {
            level_code += (1i32 << (level_prefix - 3)) - 4096;
        }
        if i == trailing_ones && trailing_ones < 3 {
            level_code += 2;
        }
        level[i] = if level_code % 2 == 0 { (level_code + 2) >> 1 } else { (-level_code - 1) >> 1 };
        if suffix_length == 0 {
            suffix_length = 1;
        }
        if level[i].abs() > (3 << (suffix_length - 1)) && suffix_length < 6 {
            suffix_length += 1;
        }
    }
    let mut zeros_left = if total_coeff < max_coeff {
        if n_c == -1 {
            read_vlc(r, &CHROMA_DC_TZ_LEN[total_coeff - 1], &CHROMA_DC_TZ_BITS[total_coeff - 1])?
        } else {
            read_vlc(r, &TOTAL_ZEROS_LEN[total_coeff - 1], &TOTAL_ZEROS_BITS[total_coeff - 1])?
        }
    } else {
        0
    };
    let mut pos = zeros_left + total_coeff - 1;
    coeffs[pos] = level[0];
    let mut i = 1;
    while i < total_coeff && zeros_left > 0 {
        let run_row = zeros_left.min(7) - 1;
        let rb = read_vlc(r, &RUN_LEN[run_row], &RUN_BITS[run_row])?;
        zeros_left -= rb;
        pos -= 1 + rb;
        coeffs[pos] = level[i];
        i += 1;
    }
    while i < total_coeff {
        pos -= 1;
        coeffs[pos] = level[i];
        i += 1;
    }
    Ok((coeffs, total_coeff))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn coeff_token_table0_basics() {
        // Table 0 (0<=nC<2): "1" -> total_coeff 0; "01" (len2) -> tc1,t1... the
        // (len,code) rows come from FFmpeg. Decode a lone "1" bit → tc=0.
        let data = [0b1000_0000u8];
        let mut r = BitReader::new(&data);
        let idx = read_vlc(&mut r, &COEFF_TOKEN_LEN[0], &COEFF_TOKEN_BITS[0]).unwrap();
        assert_eq!(idx >> 2, 0, "leading 1 decodes to total_coeff 0");
    }

    #[test_case]
    fn empty_block_reads_one_bit() {
        // total_coeff 0 → residual_block returns zeros and consumes just the
        // coeff_token "1".
        let data = [0b1000_0000u8];
        let mut r = BitReader::new(&data);
        let (c, tc) = residual_block(&mut r, 16, 0).unwrap();
        assert_eq!(tc, 0);
        assert!(c.iter().all(|&x| x == 0));
    }
}
