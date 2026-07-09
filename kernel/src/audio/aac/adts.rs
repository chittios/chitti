//! ADTS (Audio Data Transport Stream) demux — bare `.aac` files.
//!
//! ISO/IEC 13818-7 §1.A.2 fixed header + raw_data_block frames. Each frame
//! yields one raw AAC access unit (or several when `number_of_raw_data_blocks`
//! > 0; we treat multi-RDB frames as a single concatenated payload the core
//! decoder walks until TERM).

use super::{parse_asc, Asc, Decoder};
use alloc::vec::Vec;

/// One ADTS frame: ASC-equivalent config + raw payload (no ADTS header).
pub struct AdtsFrame<'a> {
    pub sample_rate: u32,
    pub channels: u8,
    pub profile_aot: u8,
    pub payload: &'a [u8],
}

/// Parse the fixed ADTS header at `data[0..]`. Returns `(header_len, frame_len, info)`.
pub fn parse_header(data: &[u8]) -> Result<(usize, usize, AdtsInfo), &'static str> {
    if data.len() < 7 {
        return Err("adts: truncated header");
    }
    if data[0] != 0xff || (data[1] & 0xf0) != 0xf0 {
        return Err("adts: bad syncword");
    }
    let protection_absent = (data[1] & 0x01) != 0;
    let profile = (data[2] >> 6) & 0x03; // 0=Main, 1=LC, 2=SSR, 3=LTP (MPEG-2 ADTS)
    // MPEG-4 ADTS uses profile+1 as AOT for LC (profile 1 → AOT 2).
    let aot = profile + 1;
    let freq_idx = (data[2] >> 2) & 0x0f;
    let sample_rate = super::tables::SAMPLE_RATES
        .get(freq_idx as usize)
        .copied()
        .filter(|&r| r != 0)
        .ok_or("adts: bad sample rate index")?;
    let chan_cfg = ((data[2] & 0x01) << 2) | ((data[3] >> 6) & 0x03);
    let channels = match chan_cfg {
        0 => 0, // PCE in stream
        1..=7 => chan_cfg,
        _ => return Err("adts: bad channel config"),
    };
    let frame_len =
        (((data[3] as usize) & 0x03) << 11) | ((data[4] as usize) << 3) | ((data[5] as usize) >> 5);
    let hdr = if protection_absent { 7 } else { 9 };
    if frame_len < hdr {
        return Err("adts: frame_length < header");
    }
    // number_of_raw_data_blocks_in_frame is low 2 bits of byte 6
    let _n_rdb = data[6] & 0x03;
    Ok((
        hdr,
        frame_len,
        AdtsInfo {
            sample_rate,
            channels,
            aot,
            freq_idx,
            chan_cfg,
        },
    ))
}

#[derive(Clone, Copy, Debug)]
pub struct AdtsInfo {
    pub sample_rate: u32,
    pub channels: u8,
    pub aot: u8,
    pub freq_idx: u8,
    pub chan_cfg: u8,
}

impl AdtsInfo {
    /// Build a synthetic 2-byte (or longer) ASC for the LC/Main/LTP core.
    pub fn to_asc_bytes(&self) -> Vec<u8> {
        // Minimal GA ASC: AOT(5) + freq(4) + ch(4) + frameLengthFlag(1)=0
        // + dependsOnCore(1)=0 + extensionFlag(1)=0
        let mut bits: u32 = 0;
        let mut n = 0u32;
        let push = |bits: &mut u32, n: &mut u32, v: u32, w: u32| {
            *bits = (*bits << w) | (v & ((1 << w) - 1));
            *n += w;
        };
        let aot = if self.aot == 0 { 2 } else { self.aot };
        push(&mut bits, &mut n, aot as u32, 5);
        push(&mut bits, &mut n, self.freq_idx as u32, 4);
        push(&mut bits, &mut n, self.chan_cfg as u32, 4);
        push(&mut bits, &mut n, 0, 3); // frameLength, depends, ext = 0
        // Pack into bytes
        let nbytes = ((n + 7) / 8) as usize;
        let mut out = alloc::vec![0u8; nbytes.max(2)];
        let shift = nbytes as u32 * 8 - n;
        let v = bits << shift;
        for i in 0..nbytes {
            out[i] = ((v >> (8 * (nbytes as u32 - 1 - i as u32))) & 0xff) as u8;
        }
        out
    }
}

/// Find the next ADTS sync in `data`, returning byte offset or None.
pub fn find_sync(data: &[u8]) -> Option<usize> {
    let mut i = 0;
    while i + 2 < data.len() {
        if data[i] == 0xff && (data[i + 1] & 0xf0) == 0xf0 {
            return Some(i);
        }
        i += 1;
    }
    None
}

/// Decode a whole ADTS `.aac` file (or ADTS byte stream) to mono S16 PCM.
pub fn decode_file(data: &[u8]) -> Result<crate::audio::Audio, &'static str> {
    let start = find_sync(data).ok_or("adts: no syncword")?;
    let mut pos = start;
    let mut pcm = Vec::new();
    let mut rate = 0u32;
    let mut dec: Option<Decoder> = None;
    let mut frames = 0usize;

    while pos + 7 < data.len() {
        // Resync if needed
        if data[pos] != 0xff || (data[pos + 1] & 0xf0) != 0xf0 {
            match find_sync(&data[pos..]) {
                Some(o) => pos += o,
                None => break,
            }
        }
        let (hdr, frame_len, info) = match parse_header(&data[pos..]) {
            Ok(v) => v,
            Err(_) => {
                pos += 1;
                continue;
            }
        };
        if pos + frame_len > data.len() {
            break; // truncated tail
        }
        let payload = &data[pos + hdr..pos + frame_len];
        pos += frame_len;

        if dec.is_none() {
            let asc_bytes = info.to_asc_bytes();
            let mut asc = parse_asc(&asc_bytes).unwrap_or(Asc {
                sample_rate: info.sample_rate,
                output_sample_rate: info.sample_rate,
                channels: if info.channels == 0 { 2 } else { info.channels },
                frame_length: 1024,
                sbr: false,
                ps: false,
                aot: info.aot,
                need_pce: info.channels == 0,
            });
            // ADTS profile already maps to AOT
            asc.aot = info.aot;
            asc.sample_rate = info.sample_rate;
            asc.output_sample_rate = info.sample_rate;
            if info.channels != 0 {
                asc.channels = info.channels;
            } else {
                asc.need_pce = true;
            }
            rate = asc.output_rate();
            dec = Some(Decoder::new(&asc));
        }
        let d = dec.as_mut().unwrap();
        match d.decode_raw(payload) {
            Ok(frame) => {
                frames += 1;
                pcm.extend_from_slice(&frame);
                if frames & 31 == 0 {
                    crate::shell::upkeep();
                }
            }
            Err(_) => {
                // Keep stream going — one bad frame of silence
                let n = d.frame_samples();
                pcm.extend(core::iter::repeat(0i16).take(n));
            }
        }
    }
    if frames == 0 {
        return Err("adts: no frames decoded");
    }
    Ok(crate::audio::Audio {
        rate: if rate != 0 {
            rate
        } else {
            dec.as_ref().map(|d| d.output_rate()).unwrap_or(44100)
        },
        pcm,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn parse_header_lc_stereo() {
        // Craft minimal ADTS: sync, MPEG4, LC profile, 44100, stereo, len=7+1
        let mut h = [0u8; 8];
        h[0] = 0xff;
        h[1] = 0xf1; // MPEG-4, layer 0, CRC absent
        // profile=1 (LC) → bits 01xxxxxx, freq_idx=4 (44100)=0100, chan starts
        // byte2: profile(2) sampling(4) private(1) chan_hi(1)
        // profile=1 → 01, freq=4 → 0100, private=0, chan_hi=1 (for ch=2 → 010)
        h[2] = 0b01_0100_0_1;
        // byte3: chan_lo(2)=10, original, home, copyright, copy_start, frame_len hi(2)
        // ch=2 → 10; frame_len=8 → ...
        h[3] = 0b10_0_0_0_0_00;
        h[4] = 0x01; // frame_len mid: (1<<3)=8 with lo=0
        h[5] = 0x00;
        h[6] = 0x00;
        h[7] = 0x00;
        let (hdr, flen, info) = parse_header(&h).unwrap();
        assert_eq!(hdr, 7);
        assert_eq!(flen, 8);
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
        assert_eq!(info.aot, 2); // LC
    }
}
