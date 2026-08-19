//! Audio decoding for the `/open` player: RIFF/WAVE, MPEG Layer III (MP3),
//! and ADTS AAC (`.aac`), all **pure functions** — bytes in, S16 PCM +
//! sample rate + channel count out, no I/O and no panics on malformed input.
//! The shell reads the file, this module decodes it, and `sound::play_ch`
//! drains it chunk by chunk (pumping `shell::upkeep` and honouring Ctrl+C
//! between chunks).
//!
//! ## Samples, frames and channels
//!
//! Stereo is **preserved and interleaved** (`L R L R …`), so `pcm.len()` is
//! `frames * channels` and is no longer a sample count you can divide by the
//! rate. Everything time-related — duration, seeking, the waveform — must go
//! through [`Audio::frames`], and mixing the two is the bug this layout
//! invites: on a stereo track it makes every duration twice its real value and
//! every seek land at half its intended position, both of which look like a
//! plausible clock rather than an error.
//!
//! The voice pipeline wants mono and says so explicitly with
//! [`Audio::into_mono`] — VAD, STT and the mel front end are all mono by
//! construction, so a silent stereo buffer reaching them would be interpreted
//! as audio at twice the rate.

pub mod aac;
pub mod hud;
pub mod mp3;
pub mod mp3_tables;
#[cfg(test)]
pub mod mp3_testdata;
pub mod wav;

use alloc::vec::Vec;

/// Decoded audio: signed-16-bit samples at `rate` Hz, `channels` of them
/// interleaved per frame.
pub struct Audio {
    pub rate: u32,
    pub pcm: Vec<i16>,
    /// 1 = mono, 2 = stereo interleaved. Never 0 — a decoder that cannot
    /// determine the channel count fails instead.
    pub channels: u8,
}

impl Audio {
    /// A mono buffer, the shape everything assumed before stereo existed.
    pub fn mono(rate: u32, pcm: Vec<i16>) -> Audio {
        Audio { rate, pcm, channels: 1 }
    }

    /// Frames — one per instant of time, whatever the channel count. This, not
    /// `pcm.len()`, is what durations and seek positions are measured in.
    pub fn frames(&self) -> usize {
        self.pcm.len() / self.channels.max(1) as usize
    }

    /// Duration in whole milliseconds.
    pub fn duration_ms(&self) -> u64 {
        if self.rate == 0 {
            return 0;
        }
        self.frames() as u64 * 1000 / self.rate as u64
    }

    /// Collapse to mono, averaging the channels. For the voice path, which is
    /// mono end to end.
    pub fn into_mono(self) -> Audio {
        if self.channels <= 1 {
            return self;
        }
        Audio { rate: self.rate, pcm: to_mono(&self.pcm, self.channels), channels: 1 }
    }
}

/// Average `channels` interleaved channels down to one.
///
/// Accumulates in `i32` before dividing: summing two near-full-scale samples in
/// `i16` wraps, which turns a loud passage into loud noise of the opposite sign
/// rather than clipping — audible, and easy to mistake for a decoder bug.
pub fn to_mono(pcm: &[i16], channels: u8) -> Vec<i16> {
    let ch = channels.max(1) as usize;
    if ch == 1 {
        return pcm.to_vec();
    }
    pcm.chunks_exact(ch)
        .map(|f| {
            let sum: i32 = f.iter().map(|&s| s as i32).sum();
            (sum / ch as i32).clamp(-32768, 32767) as i16
        })
        .collect()
}

/// Duplicate a mono buffer into `channels` interleaved channels.
///
/// Used when the device is running a stereo stream and the source is mono (a
/// TTS clip mixed with a stereo track), so the stream format never has to
/// change mid-playback.
pub fn from_mono(pcm: &[i16], channels: u8) -> Vec<i16> {
    let ch = channels.max(1) as usize;
    if ch == 1 {
        return pcm.to_vec();
    }
    let mut out = Vec::with_capacity(pcm.len() * ch);
    for &s in pcm {
        for _ in 0..ch {
            out.push(s);
        }
    }
    out
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
        let a = Audio::mono(16000, alloc::vec![0i16; 16000 * 3 + 8000]);
        assert_eq!(a.duration_ms(), 3500);
        assert_eq!(Audio::mono(0, Vec::new()).duration_ms(), 0);
    }

    /// **Duration is measured in frames, not samples.** A stereo track holds
    /// twice as many samples for the same wall time, so dividing `pcm.len()` by
    /// the rate reports double — a plausible clock that simply runs at 2x, which
    /// is exactly the kind of wrongness nothing flags.
    #[test_case]
    fn duration_counts_frames_not_samples() {
        let stereo = Audio { rate: 16000, pcm: alloc::vec![0i16; 16000 * 2 * 3], channels: 2 };
        assert_eq!(stereo.frames(), 16000 * 3);
        assert_eq!(stereo.duration_ms(), 3000, "3 s of stereo is 3 s, not 6");
        // Same wall time, same duration, whatever the channel count.
        let mono = Audio::mono(16000, alloc::vec![0i16; 16000 * 3]);
        assert_eq!(mono.duration_ms(), stereo.duration_ms());
    }

    #[test_case]
    fn mono_and_stereo_conversions_round_trip() {
        // Averaging accumulates in i32: two near-full-scale samples summed in
        // i16 would wrap, turning a loud passage into loud noise of the
        // opposite sign.
        assert_eq!(to_mono(&[i16::MAX, i16::MAX], 2), alloc::vec![i16::MAX]);
        assert_eq!(to_mono(&[i16::MIN, i16::MIN], 2), alloc::vec![i16::MIN]);
        assert_eq!(to_mono(&[1000, 3000, -400, -600], 2), alloc::vec![2000, -500]);
        // Mono in, mono out, untouched.
        assert_eq!(to_mono(&[1, 2, 3], 1), alloc::vec![1, 2, 3]);
        // Duplication is the exact inverse for an already-mono source.
        assert_eq!(from_mono(&[5, -5], 2), alloc::vec![5, 5, -5, -5]);
        assert_eq!(to_mono(&from_mono(&[5, -5], 2), 2), alloc::vec![5, -5]);
        // A trailing partial frame is dropped rather than read past.
        assert_eq!(to_mono(&[10, 20, 30], 2), alloc::vec![15]);
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
