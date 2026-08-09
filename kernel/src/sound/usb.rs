//! **USB audio playback** — a [`SndDevice`] over the xHCI isochronous OUT path.
//!
//! The device side lives in [`crate::xhci`] (endpoint configuration, the ring,
//! the per-interval pump) and [`crate::drivers::uac`] (descriptors); this is the
//! thin adapter that makes a USB headset look like every other backend.
//!
//! ## Why this one is fed differently
//!
//! Every other backend here is single-shot: hand it a buffer, it DMAs the whole
//! thing, `playing()` goes false. **An isochronous endpoint cannot work that
//! way** — the device consumes exactly one packet per service interval whether
//! or not the host supplied one, so audio has to be delivered as a continuous
//! trickle. Falling behind is a gap in the middle of a track, not a slower
//! track.
//!
//! So [`SndDevice::play_ch`] only *queues*, and `xhci::uac_pump` (called from
//! the idle tick, alongside every other upkeep) moves one interval at a time.
//! `out_free_bytes` reports the room left so the chunked-TTS speech pump and the
//! media players throttle themselves against it exactly as they do for HDA.
//!
//! ## Format
//!
//! The stream runs at whatever rate and channel count the descriptor walk chose
//! — commonly 48 kHz stereo. Callers pass their own rate and channel count and
//! this converts: [`crate::sound::resample_ch`] for the rate (frame-aware, so
//! the channels cannot swap) and duplicate/fold for the channel count. Sending
//! a mono buffer into a stereo stream unconverted is the classic version of this
//! bug and plays at half speed in the left channel.

use crate::sound::SndDevice;
use alloc::boxed::Box;
use alloc::vec::Vec;

pub struct UsbAudio {
    hz: u32,
    channels: u8,
}

impl UsbAudio {
    /// Adopt the playback stream the xHCI enumeration configured, if any.
    pub fn probe() -> Option<Box<dyn SndDevice>> {
        if !crate::arch::uac_available() {
            return None;
        }
        // **The claim happens here, not at enumeration.** Selecting the
        // streaming alt starts an endpoint the xHC services every interval, so
        // it is taken at the moment this device is about to become *the* sound
        // device — which is also the moment we know nothing else will be.
        if !crate::arch::uac_start(48_000) {
            crate::ktrace::log("sound", "USB audio device present but its stream could not be started");
            return None;
        }
        let (hz, channels) = crate::arch::uac_format()?;
        if hz == 0 || channels == 0 {
            return None;
        }
        crate::ktrace::log_fmt(format_args!(
            "sound: USB audio device adopted ({hz} Hz, {channels} ch, isochronous)"
        ));
        Some(Box::new(UsbAudio { hz, channels }))
    }

    /// Convert `pcm` from `(hz, channels)` to the stream's format and return the
    /// little-endian bytes the endpoint expects.
    fn to_device(&self, pcm: &[i16], hz: u32, channels: u8) -> Vec<u8> {
        // Channel count first, then rate: resampling picks whole frames, so it
        // has to know how wide a frame is. Doing it the other way round means
        // resampling at the wrong frame width, which is the channel-swap bug in
        // a different disguise.
        let matched: Vec<i16> = match (channels, self.channels) {
            (a, b) if a == b => pcm.to_vec(),
            (1, _) => crate::audio::from_mono(pcm, self.channels),
            (_, 1) => crate::audio::to_mono(pcm, channels),
            // Neither side is mono and they differ: fold to mono and fan back
            // out, which is lossy but never mis-interleaves.
            _ => crate::audio::from_mono(&crate::audio::to_mono(pcm, channels), self.channels),
        };
        let resampled = crate::sound::resample_ch(&matched, hz, self.hz, self.channels);
        let mut out = Vec::with_capacity(resampled.len() * 2);
        for s in resampled {
            out.extend_from_slice(&s.to_le_bytes());
        }
        out
    }
}

impl SndDevice for UsbAudio {
    fn play(&mut self, pcm: &[i16], hz: u32) -> Result<(), &'static str> {
        self.play_ch(pcm, hz, 1)
    }

    fn play_ch(&mut self, pcm: &[i16], hz: u32, channels: u8) -> Result<(), &'static str> {
        let bytes = self.to_device(pcm, hz, channels);
        if bytes.is_empty() {
            return Ok(());
        }
        if crate::arch::uac_queue(&bytes) {
            Ok(())
        } else {
            // The queue is full — the caller is ahead of the device. Reported
            // rather than blocked on: the pump runs from the idle tick, so
            // spinning here would stop the very thing that drains it.
            Err("usb audio: queue full")
        }
    }

    fn out_channels(&self) -> u8 {
        self.channels
    }

    fn out_free_bytes(&mut self) -> usize {
        // Pump first: the caller is asking how much room there is precisely
        // because it wants to refill, and room only appears when a packet
        // leaves.
        crate::arch::uac_pump();
        crate::arch::uac_free_bytes()
    }

    /// **This query pumps, deliberately.**
    ///
    /// Callers wait for playback by spinning on `playing()`, and they do not all
    /// spin on the same tick — `voice_test` uses `ui_tick`, the speech pump uses
    /// `upkeep`, the media players use their own. `upkeep` pumps too, but a loop
    /// that does not reach it would spin here forever: the queue only drains
    /// from the pump, so `playing()` would stay true for good and the shell
    /// would hang mid-tone. That is exactly what happened.
    ///
    /// Pumping here makes progress a property of *asking*, so every waiter is
    /// correct regardless of which tick it runs. The two calls are sequential,
    /// never nested, so the controller lock is taken twice rather than
    /// re-entered.
    fn playing(&mut self) -> bool {
        crate::arch::uac_pump();
        crate::arch::uac_busy()
    }

    fn capture_start(&mut self, _hz: u32) -> Result<(), &'static str> {
        // A UAC capture stream is a separate AudioStreaming interface with an
        // isochronous IN endpoint. `find_output_stream` deliberately only looks
        // for output, so there is nothing to start — said plainly rather than
        // returning Ok and delivering silence, which would make `/voice` wait
        // for an utterance that can never arrive.
        Err("usb audio: capture not implemented")
    }

    fn capture_read(&mut self, _out: &mut [i16]) -> usize {
        0
    }

    fn capture_stop(&mut self) {}
}
