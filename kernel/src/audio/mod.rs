//! Audio decoding for the `/open` player: RIFF/WAVE, MPEG Layer III (MP3),
//! and ADTS AAC (`.aac`), all **pure functions** — bytes in, mono S16 PCM +
//! sample rate out, no I/O and no panics on malformed input. The shell reads
//! the file, this module decodes it, and `sound::play` drains it chunk by
//! chunk (pumping `shell::upkeep` and honouring Ctrl+C between chunks).

pub mod aac;
pub mod mp3;
pub mod mp3_tables;
#[cfg(test)]
pub mod mp3_testdata;
pub mod wav;

use alloc::vec::Vec;

/// Decoded audio: mono signed-16-bit samples at `rate` Hz (what the sound
/// devices play; stereo sources are downmixed `(l+r)/2`).
pub struct Audio {
    pub rate: u32,
    pub pcm: Vec<i16>,
}

impl Audio {
    /// Duration in whole milliseconds.
    pub fn duration_ms(&self) -> u64 {
        if self.rate == 0 {
            return 0;
        }
        self.pcm.len() as u64 * 1000 / self.rate as u64
    }
}

/// Default number of peak buckets for the audio-player waveform visualizer.
pub const WAVEFORM_BINS: usize = 256;

/// Downsample mono S16 PCM into `n` peak magnitudes in `0..=255` (one max-abs
/// sample per equal-length chunk). Used by the audio-player UI so a track can
/// paint a wave without re-scanning multi-megabyte PCM every refresh.
pub fn waveform_peaks(pcm: &[i16], n: usize) -> Vec<u8> {
    if n == 0 {
        return Vec::new();
    }
    let mut out = alloc::vec![0u8; n];
    if pcm.is_empty() {
        return out;
    }
    let chunk = (pcm.len() + n - 1) / n;
    for (i, slot) in out.iter_mut().enumerate() {
        let start = i * chunk;
        if start >= pcm.len() {
            break;
        }
        let end = (start + chunk).min(pcm.len());
        let mut peak: u32 = 0;
        for &s in &pcm[start..end] {
            let a = (s as i32).unsigned_abs();
            if a > peak {
                peak = a;
            }
        }
        // Map 0..32768 → 0..255 (keep a floor of 1 when any energy is present
        // so a quiet bucket still shows a 1px tick).
        *slot = if peak == 0 {
            0
        } else {
            ((peak * 255) / 32768).max(1).min(255) as u8
        };
    }
    out
}

/// Decode `bytes` by sniffing the container: RIFF/WAVE, MP3 (ID3 / frame
/// sync), or ADTS AAC (`.aac`, syncword `0xFFF`).
pub fn decode(bytes: &[u8]) -> Result<Audio, &'static str> {
    if bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WAVE" {
        wav::decode(bytes)
    } else if is_adts(bytes) {
        aac::decode_adts(bytes)
    } else if bytes.len() >= 3
        && (&bytes[0..3] == b"ID3" || (bytes[0] == 0xff && bytes[1] & 0xe0 == 0xe0))
    {
        mp3::decode(bytes)
    } else {
        Err("unknown audio format (WAV, MP3, and ADTS AAC are supported)")
    }
}

/// True when `bytes` look like ADTS AAC (not MP3): sync `0xFFF` with layer
/// bits zero (MP3 layer is non-zero in the same header region).
fn is_adts(bytes: &[u8]) -> bool {
    // Scan a little for a sync (ID3-prefixed ADTS is rare; bare ADTS common).
    let n = bytes.len().min(64);
    let mut i = 0;
    while i + 2 < n {
        if bytes[i] == 0xff && (bytes[i + 1] & 0xf0) == 0xf0 {
            // ADTS: layer bits (byte1 bits 3-2) must be 00; MP3 uses 01/10/11.
            let layer = (bytes[i + 1] >> 1) & 0x03;
            if layer == 0 {
                return true;
            }
        }
        i += 1;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn unknown_container_is_an_error() {
        assert!(decode(b"OggS....").is_err());
        assert!(decode(b"").is_err());
    }

    #[test_case]
    fn duration_math() {
        let a = Audio { rate: 16000, pcm: alloc::vec![0i16; 16000 * 3 + 8000] };
        assert_eq!(a.duration_ms(), 3500);
        assert_eq!(Audio { rate: 0, pcm: Vec::new() }.duration_ms(), 0);
    }

    #[test_case]
    fn waveform_peaks_empty_and_constant() {
        assert!(waveform_peaks(&[], 8).iter().all(|&p| p == 0));
        // Full-scale constant → peak 255 in every bin.
        let loud = alloc::vec![i16::MAX; 1000];
        let p = waveform_peaks(&loud, 10);
        assert_eq!(p.len(), 10);
        assert!(p.iter().all(|&x| x >= 254), "got {p:?}");
        // Silence → zeros.
        let quiet = alloc::vec![0i16; 500];
        assert!(waveform_peaks(&quiet, 5).iter().all(|&x| x == 0));
    }

    #[test_case]
    fn waveform_peaks_locates_loud_region() {
        // First half quiet, second half loud — later bins must dominate.
        let mut pcm = alloc::vec![0i16; 200];
        for s in &mut pcm[100..] {
            *s = 20000;
        }
        let p = waveform_peaks(&pcm, 8);
        let early: u32 = p[..4].iter().map(|&x| x as u32).sum();
        let late: u32 = p[4..].iter().map(|&x| x as u32).sum();
        assert!(late > early, "loud tail should dominate: {p:?}");
    }
}
