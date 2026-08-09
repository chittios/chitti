//! RIFF/WAVE decoder: chunk walk → `fmt ` (rate/channels/format) + `data` →
//! mono S16. Handles PCM at 8 (unsigned), 16, 24, and 32 bits, IEEE float32,
//! and `WAVE_FORMAT_EXTENSIBLE` wrappers of both; any channel count is
//! downmixed by averaging. Malformed input returns `Err`, never panics.

use super::Audio;
use alloc::vec::Vec;

const FMT_PCM: u16 = 1;
const FMT_FLOAT: u16 = 3;
const FMT_EXTENSIBLE: u16 = 0xFFFE;

fn le16(b: &[u8]) -> u16 {
    u16::from_le_bytes([b[0], b[1]])
}
fn le32(b: &[u8]) -> u32 {
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

pub fn decode(bytes: &[u8]) -> Result<Audio, &'static str> {
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err("wav: not a RIFF/WAVE file");
    }
    let mut pos = 12usize;
    let mut format = 0u16;
    let mut channels = 0usize;
    let mut rate = 0u32;
    let mut bits = 0usize;
    let mut data: Option<&[u8]> = None;

    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let sz = le32(&bytes[pos + 4..]) as usize;
        // A sloppy writer may declare a data chunk running to EOF; clamp.
        let body = &bytes[pos + 8..(pos + 8 + sz).min(bytes.len())];
        match id {
            b"fmt " if body.len() >= 16 => {
                format = le16(body);
                channels = le16(&body[2..]) as usize;
                rate = le32(&body[4..]);
                bits = le16(&body[14..]) as usize;
                if format == FMT_EXTENSIBLE && body.len() >= 26 {
                    // The real format is the first two GUID bytes of SubFormat.
                    format = le16(&body[24..]);
                }
            }
            b"data" => data = Some(body),
            _ => {}
        }
        pos += 8 + sz + (sz & 1); // chunks are word-aligned
    }

    let data = data.ok_or("wav: no data chunk")?;
    if channels == 0 || rate == 0 {
        return Err("wav: missing or bad fmt chunk");
    }
    if rate > 384_000 {
        return Err("wav: unreasonable sample rate");
    }
    let bytes_per = bits / 8;
    if !matches!((format, bits), (FMT_PCM, 8 | 16 | 24 | 32) | (FMT_FLOAT, 32)) {
        return Err("wav: unsupported sample format (PCM 8/16/24/32 or float32)");
    }

    let frame = bytes_per * channels;
    let n = data.len() / frame.max(1);
    // Stereo is kept as-is (interleaved L R); anything wider is folded to
    // stereo-from-mono, because the devices here run one or two channels and a
    // 5.1 file dropped to its first two channels would lose the centre — which
    // on a film is most of the dialogue.
    let out_ch: usize = if channels == 1 { 1 } else { 2 };
    let mut pcm: Vec<i16> = Vec::with_capacity(n * out_ch);
    let sample = |s: &[u8]| -> i32 {
        match (format, bits) {
            (FMT_PCM, 8) => (s[0] as i32 - 128) << 8, // u8 is unsigned, offset-binary
            (FMT_PCM, 16) => i16::from_le_bytes([s[0], s[1]]) as i32,
            (FMT_PCM, 24) => (i32::from_le_bytes([0, s[0], s[1], s[2]]) >> 8) >> 8,
            (FMT_PCM, 32) => i32::from_le_bytes([s[0], s[1], s[2], s[3]]) >> 16,
            (FMT_FLOAT, 32) => {
                let x = f32::from_bits(le32(s));
                (x.clamp(-1.0, 1.0) * 32767.0) as i32
            }
            _ => 0,
        }
    };
    for f in 0..n {
        let at = |c: usize| sample(&data[f * frame + c * bytes_per..]);
        match (channels, out_ch) {
            (1, _) => pcm.push(at(0).clamp(-32768, 32767) as i16),
            (2, _) => {
                pcm.push(at(0).clamp(-32768, 32767) as i16);
                pcm.push(at(1).clamp(-32768, 32767) as i16);
            }
            // More than two channels: average the odd-indexed into right and
            // the even-indexed into left, so centre and LFE reach both sides
            // rather than vanishing.
            _ => {
                let (mut l, mut r, mut nl, mut nr) = (0i64, 0i64, 0i64, 0i64);
                for c in 0..channels {
                    let v = at(c) as i64;
                    if c % 2 == 0 {
                        l += v;
                        nl += 1;
                    } else {
                        r += v;
                        nr += 1;
                    }
                }
                pcm.push((l / nl.max(1)).clamp(-32768, 32767) as i16);
                pcm.push((r / nr.max(1)).clamp(-32768, 32767) as i16);
            }
        }
    }
    Ok(Audio { rate, pcm, channels: out_ch as u8 })
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    /// Build a minimal WAV: fmt (`format`, `channels`, `rate`, `bits`) + data.
    fn wav(format: u16, channels: u16, rate: u32, bits: u16, data: &[u8]) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(b"RIFF");
        b.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
        b.extend_from_slice(b"WAVE");
        b.extend_from_slice(b"fmt ");
        b.extend_from_slice(&16u32.to_le_bytes());
        b.extend_from_slice(&format.to_le_bytes());
        b.extend_from_slice(&channels.to_le_bytes());
        b.extend_from_slice(&rate.to_le_bytes());
        let block = channels as u32 * bits as u32 / 8;
        b.extend_from_slice(&(rate * block).to_le_bytes());
        b.extend_from_slice(&(block as u16).to_le_bytes());
        b.extend_from_slice(&bits.to_le_bytes());
        b.extend_from_slice(b"data");
        b.extend_from_slice(&(data.len() as u32).to_le_bytes());
        b.extend_from_slice(data);
        b
    }

    /// **Stereo survives decoding**, interleaved. It used to be averaged here,
    /// which is what made every stereo file play in mono.
    #[test_case]
    fn pcm16_stereo_is_preserved_interleaved() {
        let mut d = Vec::new();
        for v in [1000i16, 3000, -400, -600] {
            d.extend_from_slice(&v.to_le_bytes());
        }
        let a = decode(&wav(1, 2, 44100, 16, &d)).unwrap();
        assert_eq!(a.rate, 44100);
        assert_eq!(a.channels, 2);
        assert_eq!(a.pcm, alloc::vec![1000, 3000, -400, -600]);
        // Two frames, not four samples' worth of time — the distinction the
        // whole `frames()` API exists for.
        assert_eq!(a.frames(), 2);
        // And folding down still gives what the old decoder produced.
        let m = a.into_mono();
        assert_eq!(m.channels, 1);
        assert_eq!(m.pcm, alloc::vec![2000, -500]);
    }

    /// A mono file is byte-identical to before — the regression guard for every
    /// caller that predates channels existing.
    #[test_case]
    fn mono_is_unchanged() {
        let mut d = Vec::new();
        for v in [1000i16, 3000, -400] {
            d.extend_from_slice(&v.to_le_bytes());
        }
        let a = decode(&wav(1, 1, 44100, 16, &d)).unwrap();
        assert_eq!(a.channels, 1);
        assert_eq!(a.pcm, alloc::vec![1000, 3000, -400]);
        assert_eq!(a.frames(), 3);
    }

    /// More than two channels folds to **stereo**, not to the first two: taking
    /// channels 0 and 1 of a 5.1 stream drops the centre, which on a film is
    /// most of the dialogue.
    #[test_case]
    fn more_than_two_channels_folds_to_stereo_without_losing_centre() {
        // One 6-channel frame; the centre channel is the only loud one.
        let mut d = Vec::new();
        for v in [0i16, 0, 30000, 0, 0, 0] {
            d.extend_from_slice(&v.to_le_bytes());
        }
        let a = decode(&wav(1, 6, 48000, 16, &d)).unwrap();
        assert_eq!(a.channels, 2);
        assert_eq!(a.frames(), 1);
        assert_ne!(a.pcm[0], 0, "the centre channel must reach the left");
        assert_eq!(a.pcm.len(), 2);
    }

    #[test_case]
    fn pcm8_offset_binary() {
        // 128 = silence, 255 ~ +max, 0 = -max.
        let a = decode(&wav(1, 1, 8000, 8, &[128, 255, 0])).unwrap();
        assert_eq!(a.pcm, alloc::vec![0, 127 << 8, -128 << 8]);
    }

    #[test_case]
    fn pcm24_and_32_scale_down() {
        let mut d = Vec::new();
        d.extend_from_slice(&0x123456i32.to_le_bytes()[..3]); // 24-bit LE
        let a = decode(&wav(1, 1, 48000, 24, &d)).unwrap();
        assert_eq!(a.pcm, alloc::vec![0x1234]);
        let a = decode(&wav(1, 1, 48000, 32, &0x1234_5678i32.to_le_bytes())).unwrap();
        assert_eq!(a.pcm, alloc::vec![0x1234]);
    }

    #[test_case]
    fn float32_clamps() {
        let mut d = Vec::new();
        for v in [0.5f32, -2.0, 1.0] {
            d.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        let a = decode(&wav(3, 1, 16000, 32, &d)).unwrap();
        assert_eq!(a.pcm, alloc::vec![16383, -32767, 32767]);
    }

    #[test_case]
    fn malformed_errors_not_panics() {
        assert!(decode(b"RIFF").is_err());
        assert!(decode(&wav(1, 0, 44100, 16, &[])).is_err(), "zero channels");
        assert!(decode(&wav(7, 1, 44100, 16, &[0, 0])).is_err(), "unknown format tag");
        // Truncated data chunk decodes what's there (clamped), no panic.
        let mut w = wav(1, 1, 8000, 16, &[1, 0, 2, 0]);
        w.truncate(w.len() - 2);
        assert_eq!(decode(&w).unwrap().pcm.len(), 1);
    }
}
