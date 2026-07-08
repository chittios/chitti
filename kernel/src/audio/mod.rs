//! Audio decoding for the `/open` player: RIFF/WAVE and MPEG Layer III
//! (MP3), both **pure functions** — bytes in, mono S16 PCM + sample rate out,
//! no I/O and no panics on malformed input. The shell reads the file, this
//! module decodes it, and `sound::play` drains it chunk by chunk (pumping
//! `shell::upkeep` and honouring Ctrl+C between chunks).

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

/// Decode `bytes` by sniffing the container: RIFF/WAVE, or MP3 (ID3 tag /
/// frame sync).
pub fn decode(bytes: &[u8]) -> Result<Audio, &'static str> {
    if bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WAVE" {
        wav::decode(bytes)
    } else if bytes.len() >= 3 && (&bytes[0..3] == b"ID3" || (bytes[0] == 0xff && bytes[1] & 0xe0 == 0xe0)) {
        mp3::decode(bytes)
    } else {
        Err("unknown audio format (WAV and MP3 are supported)")
    }
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
}
