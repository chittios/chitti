//! **Sound** — PCM audio in and out for the voice pipeline (`/voice`). Drivers
//! implement [`SndDevice`] (16-bit signed PCM, mono); three back-ends cover
//! every target the standing rule requires:
//! - **virtio-sound over virtio-mmio** — aarch64 QEMU `virt` (`-kernel`),
//! - **virtio-sound over PCI** — QEMU x86/aarch64 with a PCI bus,
//! - **Intel HDA** ([`hda`]) — **VirtualBox** (x86 *and* ARM) and real Intel/ARM
//!   machines, plus QEMU's `intel-hda` for testing.
//! `autodetect` tries them in that order, so the same image gets audio on QEMU,
//! VirtualBox, and bare metal.
//!
//! The virtio-snd protocol (virtio spec §5.14) rides four virtqueues:
//! control(0) / event(1) / tx(2, playback) / rx(3, capture). We poll — no
//! interrupts — and run one output stream and one input stream, S16 mono, at
//! the rate the caller asks for (16 kHz for VAD/STT, 22.05 kHz for TTS).

use crate::mm::Locked;
use alloc::boxed::Box;
use alloc::vec::Vec;

pub mod usb;

pub mod proto {
    //! virtio-snd control-plane constants + tiny message builders (all
    //! little-endian on the wire, which every supported target is).

    // Control request codes.
    pub const R_PCM_INFO: u32 = 0x0100;
    pub const R_PCM_SET_PARAMS: u32 = 0x0101;
    pub const R_PCM_PREPARE: u32 = 0x0102;
    pub const R_PCM_RELEASE: u32 = 0x0103;
    pub const R_PCM_START: u32 = 0x0104;
    pub const R_PCM_STOP: u32 = 0x0105;
    /// First "success" status code (`VIRTIO_SND_S_OK`).
    pub const S_OK: u32 = 0x8000;

    // PCM sample formats / rates (spec enumerations).
    pub const FMT_S16: u8 = 5;
    pub fn rate_code(hz: u32) -> u8 {
        match hz {
            5512 => 0,
            8000 => 1,
            11025 => 2,
            16000 => 3,
            22050 => 4,
            32000 => 5,
            44100 => 6,
            48000 => 7,
            _ => 3, // default 16 kHz
        }
    }

    /// `virtio_snd_pcm_set_params` for stream `id`: S16, mono, `hz`.
    pub fn set_params(id: u32, hz: u32, buffer_bytes: u32, period_bytes: u32) -> [u8; 24] {
        let mut m = [0u8; 24];
        m[0..4].copy_from_slice(&R_PCM_SET_PARAMS.to_le_bytes());
        m[4..8].copy_from_slice(&id.to_le_bytes());
        m[8..12].copy_from_slice(&buffer_bytes.to_le_bytes());
        m[12..16].copy_from_slice(&period_bytes.to_le_bytes());
        // features = 0
        m[20] = 1; // channels
        m[21] = FMT_S16;
        m[22] = rate_code(hz);
        m
    }

    /// A simple `virtio_snd_pcm_hdr` request (prepare/start/stop/release).
    pub fn pcm_op(code: u32, id: u32) -> [u8; 8] {
        let mut m = [0u8; 8];
        m[0..4].copy_from_slice(&code.to_le_bytes());
        m[4..8].copy_from_slice(&id.to_le_bytes());
        m
    }
}

/// Resample mono PCM from `hz` to a device's fixed `out_hz` by nearest-neighbor
/// (exact sample-repetition for the integer ratios in use: 16→48 kHz ×3,
/// 24→48 kHz ×2). Fixed-rate devices (HDA @48 k, AC'97 @48 k) call this instead
/// of assuming the input rate — playing 24 kHz TTS through a hardcoded 16 kHz
/// assumption is how "hello there" became "helllloooo theeeere" (1.5× slow).
pub fn resample(pcm: &[i16], hz: u32, out_hz: u32) -> Vec<i16> {
    if hz == out_hz || hz == 0 {
        return pcm.to_vec();
    }
    let out_len = (pcm.len() as u64 * out_hz as u64 / hz as u64) as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        out.push(pcm[(i as u64 * hz as u64 / out_hz as u64) as usize]);
    }
    out
}

/// [`resample`] for interleaved multi-channel PCM: picks whole **frames**, so
/// the channels stay in step.
///
/// Running interleaved stereo through the mono [`resample`] picks nearest
/// *samples*, and whenever the chosen index has the opposite parity to its slot
/// the two channels swap — so the image flips back and forth through the track
/// and the sum is close enough to right that it reads as a bad encode rather
/// than a resampler bug.
pub fn resample_ch(pcm: &[i16], hz: u32, out_hz: u32, channels: u8) -> Vec<i16> {
    let ch = channels.max(1) as usize;
    if ch == 1 {
        return resample(pcm, hz, out_hz);
    }
    if hz == out_hz || hz == 0 {
        return pcm.to_vec();
    }
    let in_frames = pcm.len() / ch;
    let out_frames = (in_frames as u64 * out_hz as u64 / hz as u64) as usize;
    let mut out = Vec::with_capacity(out_frames * ch);
    for f in 0..out_frames {
        let src = (f as u64 * hz as u64 / out_hz as u64) as usize;
        let src = src.min(in_frames.saturating_sub(1));
        for c in 0..ch {
            out.push(pcm[src * ch + c]);
        }
    }
    out
}

/// A PCM sound device: play and capture 16-bit signed mono samples. Poll-driven.
pub trait SndDevice {
    /// Start (or restart) the output stream at `hz`, then queue `pcm` for
    /// playback. Blocks only to enqueue (the device drains asynchronously).
    /// Implementations **must honor `hz`** — set the hardware rate to it, or
    /// [`resample`] to the device's fixed rate. Callers pass 16 kHz (mic/test
    /// tones) *and* 24 kHz (KittenTTS).
    fn play(&mut self, pcm: &[i16], hz: u32) -> Result<(), &'static str>;
    /// Start (or restart) the output stream at `hz` with `channels` channels and
    /// queue `pcm`, **interleaved** when `channels > 1`.
    ///
    /// The default folds to mono and calls [`Self::play`], so a driver that has
    /// not been taught stereo keeps working byte-identically — and a caller
    /// never has to ask whether the device can do it. Override this to run a
    /// real multi-channel stream.
    ///
    /// A driver that overrides it **must still honour `channels == 1`** by
    /// running a mono stream or duplicating: the voice pipeline plays mono into
    /// the same device, and a mono buffer fed to a stereo stream plays at half
    /// speed in the left channel, which sounds like a broken decoder rather
    /// than a stream-format bug.
    fn play_ch(&mut self, pcm: &[i16], hz: u32, channels: u8) -> Result<(), &'static str> {
        if channels <= 1 {
            return self.play(pcm, hz);
        }
        self.play(&crate::audio::to_mono(pcm, channels), hz)
    }
    /// Channels the device can actually run (1 or 2). Reported so `/voice` and
    /// the players can say what they got rather than guessing.
    fn out_channels(&self) -> u8 {
        1
    }
    /// How many PCM **bytes** [`Self::play`] can currently enqueue without
    /// blocking (free device periods). 0 = unknown/none — callers that need a
    /// non-blocking feed (the chunked-TTS speech pump) then only refill when
    /// the queue has fully drained. Default keeps legacy drivers working.
    fn out_free_bytes(&mut self) -> usize {
        0
    }
    /// True while queued playback is still draining.
    fn playing(&mut self) -> bool;
    /// Start the capture stream at `hz` (idempotent).
    fn capture_start(&mut self, hz: u32) -> Result<(), &'static str>;
    /// Pop captured samples into `out`; returns how many were written.
    fn capture_read(&mut self, out: &mut [i16]) -> usize;
    /// Stop capturing.
    fn capture_stop(&mut self);
}

static SND: Locked<Option<Box<dyn SndDevice>>> = Locked::new(None);

/// Software output volume in percent (`0..=100`). Applied in [`play`] so every
/// backend (virtio-snd, HDA, AC'97, SB16) gets the same gain without per-driver
/// wiring. Adjusted by the media players' ↑/↓ keys.
static VOLUME: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(100);
/// Global mute — when set, [`play`] queues silence. Toggled by `m` on the
/// audio/video tabs (shared so muting one surface mutes the device).
static MUTED: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

/// Current software volume percent (`0..=100`).
pub fn volume() -> u32 {
    VOLUME.load(core::sync::atomic::Ordering::Relaxed).min(100)
}

/// Set software volume percent, clamped to `0..=100`.
pub fn set_volume(v: u32) {
    VOLUME.store(v.min(100), core::sync::atomic::Ordering::Relaxed);
}

/// Adjust volume by `delta` percent points (e.g. `+5` / `-5`). Returns the new
/// level. Unmutes when raising volume from a muted state so ↑ is useful after
/// `m`.
pub fn volume_adjust(delta: i32) -> u32 {
    let cur = volume() as i32;
    let next = (cur + delta).clamp(0, 100) as u32;
    set_volume(next);
    if delta > 0 && muted() {
        set_muted(false);
    }
    next
}

/// Whether global software mute is on.
pub fn muted() -> bool {
    MUTED.load(core::sync::atomic::Ordering::Relaxed)
}

/// Set global software mute.
pub fn set_muted(m: bool) {
    MUTED.store(m, core::sync::atomic::Ordering::Relaxed);
}

/// Toggle global mute; returns the new muted state.
pub fn toggle_mute() -> bool {
    let n = !muted();
    set_muted(n);
    n
}

/// Apply [`volume`]/[`muted`] to an S16 mono buffer (in place, clamped).
pub fn apply_output_gain(pcm: &mut [i16]) {
    if muted() || volume() == 0 {
        for s in pcm.iter_mut() {
            *s = 0;
        }
        return;
    }
    let g = volume() as i32;
    if g >= 100 {
        return;
    }
    for s in pcm.iter_mut() {
        *s = ((*s as i32) * g / 100) as i16;
    }
}

/// Bring the sound subsystem up on `dev`.
pub fn init(dev: Box<dyn SndDevice>) {
    SND.with(|s| *s = Some(dev));
    crate::ktrace::log("sound", "PCM device up (S16 mono, poll-driven)");
}

/// True once a sound device has been brought up.
pub fn is_up() -> bool {
    SND.with(|s| s.is_some())
}

/// Re-establish the sound device after a suspend.
///
/// Every backend here is **polled and stateful in the controller** — HDA's
/// CORB/RIRB rings, the codec's widget power state and amp settings, virtio's
/// negotiated queues. S3 resets all of it, so the retained handle keeps
/// answering and simply never produces sound again: not an error, just silence,
/// which is the hardest kind of resume failure to attribute.
///
/// So the device is dropped and re-probed from scratch rather than poked back
/// into life. Re-probing is what every one of these drivers already does
/// correctly at boot, and reusing that path is far safer than a second,
/// resume-only bring-up sequence that nothing exercises.
///
/// **This leaks the previous instance's DMA pages** (the frame allocator has no
/// free path for them). That is bounded — a few hundred KiB per resume — and is
/// the honest trade against a silent audio device; a machine suspended enough
/// times for it to matter has other problems.
pub fn resume() {
    let had = is_up();
    SND.with(|s| *s = None);
    autodetect();
    crate::ktrace::log_fmt(format_args!(
        "sound: resume re-probe {} (was {})",
        if is_up() { "found a device" } else { "found nothing" },
        if had { "up" } else { "down" }
    ));
}

/// Discover and bring up the first available sound device: virtio-snd over
/// mmio (aarch64 QEMU `-kernel`), virtio-snd over PCI (QEMU), then **Intel
/// HDA** — VirtualBox (x86 + ARM) and real machines. No-op if none is present.
pub fn autodetect() {
    if is_up() {
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if let Some(dev) = crate::arch::aarch64::virtio_snd::VirtioSndMmio::probe() {
            init(Box::new(dev));
            return;
        }
    }
    if let Some(dev) = virtio_snd_pci::VirtioSndPci::probe() {
        init(dev);
        return;
    }
    if let Some(dev) = hda::Hda::probe() {
        init(dev);
        return;
    }
    // Legacy x86 audio: AC'97 (VirtualBox/ICH), then Sound Blaster 16 (ISA).
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(dev) = crate::arch::x86_64::ac97::Ac97::probe() {
            init(dev);
            return;
        }
        if let Some(dev) = crate::arch::x86_64::sb16::Sb16::probe() {
            init(dev);
            return;
        }
    }
    // **USB audio last, not first**, and the reason is a trade worth stating.
    //
    // A plugged-in headset is a deliberate act, so preferring it looks right —
    // and it is what a desktop OS does. But this driver implements **output
    // only**: adopting a headset as *the* sound device would take the
    // microphone away, and `/voice` is a mic-to-model loop. On a laptop with
    // both HDA and a USB headset that trades a working voice assistant for a
    // different set of speakers, which is a bad deal.
    //
    // So USB audio serves the machines that would otherwise have **no** audio
    // at all — a desktop whose only output is a USB DAC, a laptop with no
    // codec — and stays out of the way where a full-duplex device exists.
    // Revisit when the capture side lands: a UAC capture stream is a second
    // AudioStreaming interface with an isochronous IN endpoint, at which point
    // preferring the headset becomes the right default.
    if let Some(dev) = usb::UsbAudio::probe() {
        init(dev);
        return;
    }
    // Nothing matched: dump the multimedia-class PCI devices so an unsupported
    // controller (or a PCI-discovery gap) is diagnosable from the boot log.
    crate::ktrace::log("sound", "no audio device matched — multimedia (class 0x04) PCI devices:");
    #[cfg(target_arch = "aarch64")]
    crate::pci::log_class(0x04);
    #[cfg(target_arch = "x86_64")]
    crate::arch::x86_64::pci::log_class(0x04);
}

/// Queue `pcm` (S16 mono at `hz`) for playback. Applies the global software
/// volume/mute (see [`volume`] / [`muted`]) so media-player ↑/↓/`m` take effect
/// on every device backend.
pub fn play(pcm: &[i16], hz: u32) -> Result<(), &'static str> {
    play_ch(pcm, hz, 1)
}

/// Queue `pcm` (S16, `channels` interleaved, at `hz`) for playback.
///
/// The gain is applied per sample, so it is channel-agnostic and needs no
/// separate stereo path — which is the reason volume/mute keep working
/// unchanged for both.
pub fn play_ch(pcm: &[i16], hz: u32, channels: u8) -> Result<(), &'static str> {
    SND.with(|s| match s.as_mut() {
        Some(d) => {
            if muted() || volume() < 100 {
                let mut buf = pcm.to_vec();
                apply_output_gain(&mut buf);
                d.play_ch(&buf, hz, channels)
            } else {
                d.play_ch(pcm, hz, channels)
            }
        }
        None => Err("no sound device"),
    })
}

/// Channels the active device will run (1 or 2); 1 when none is up.
pub fn out_channels() -> u8 {
    SND.with(|s| s.as_ref().map(|d| d.out_channels()).unwrap_or(1))
}

/// True while playback is draining. Poll this (with `sched::yield_now`) to wait.
pub fn playing() -> bool {
    SND.with(|s| s.as_mut().map(|d| d.playing()).unwrap_or(false))
}

/// PCM bytes [`play`] can enqueue right now without blocking (0 when unknown —
/// see [`SndDevice::out_free_bytes`]). The chunked-TTS speech pump uses this to
/// keep the device fed from `ui_tick` while synthesis runs on the SMP fleet.
pub fn out_free_bytes() -> usize {
    SND.with(|s| s.as_mut().map(|d| d.out_free_bytes()).unwrap_or(0))
}

/// Start capturing at `hz`.
pub fn capture_start(hz: u32) -> Result<(), &'static str> {
    SND.with(|s| match s.as_mut() {
        Some(d) => d.capture_start(hz),
        None => Err("no sound device"),
    })
}

/// Read captured samples; returns the count written into `out`.
pub fn capture_read(out: &mut [i16]) -> usize {
    SND.with(|s| s.as_mut().map(|d| d.capture_read(out)).unwrap_or(0))
}

/// Stop capturing.
pub fn capture_stop() {
    SND.with(|s| {
        if let Some(d) = s.as_mut() {
            d.capture_stop();
        }
    });
}

/// RMS level of a PCM frame, normalized to 0.0..=1.0 — drives the `/voice`
/// waveform animation.
pub fn rms(pcm: &[i16]) -> f32 {
    if pcm.is_empty() {
        return 0.0;
    }
    let sum: f64 = pcm.iter().map(|&s| (s as f64) * (s as f64)).sum();
    let rms = libm_sqrt(sum / pcm.len() as f64) / 32768.0;
    rms as f32
}

fn libm_sqrt(x: f64) -> f64 {
    // Newton's method is plenty for a UI level meter.
    if x <= 0.0 {
        return 0.0;
    }
    let mut r = x;
    for _ in 0..24 {
        r = 0.5 * (r + x / r);
    }
    r
}

pub mod g2p;
pub mod hda;
pub mod mel;
pub mod model_store;
pub mod stt;
pub mod tts;
pub mod vad;
pub mod virtio_snd_pci;

/// A short test tone (sine-ish square blend) for `/voice test`.
pub fn test_tone(hz_tone: u32, ms: u32, rate: u32) -> Vec<i16> {
    let n = (rate * ms / 1000) as usize;
    let mut v = Vec::with_capacity(n);
    let period = (rate / hz_tone.max(1)).max(2) as usize;
    for i in 0..n {
        // Triangle wave: soft on the ears, no float trig needed.
        let ph = i % period;
        let half = period / 2;
        let amp = if ph < half { (ph * 2 * 20000 / half.max(1)) as i32 - 10000 } else { 10000 - ((ph - half) * 2 * 20000 / half.max(1)) as i32 };
        // Fade in/out over 10 ms to avoid clicks.
        let fade = (rate / 100) as usize;
        let g = if i < fade { i * 256 / fade } else if n - i < fade { (n - i) * 256 / fade } else { 256 };
        v.push((amp * g as i32 / 256) as i16);
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **Interleaved stereo must be resampled by frame.** The mono resampler
    /// picks nearest *samples*; whenever the chosen index has the opposite
    /// parity to its slot, left and right swap — so the stereo image flips back
    /// and forth through the track while the sum stays close enough to right
    /// that it reads as a bad encode rather than a resampler bug.
    #[test_case]
    fn stereo_resampling_keeps_the_channels_in_step() {
        // Left channel is always positive, right always negative, so a swap is
        // unmistakable in the output's signs.
        let src: Vec<i16> = (0..8).flat_map(|i| [100 + i, -(100 + i)]).collect();
        let up = resample_ch(&src, 8000, 12000, 2);
        assert_eq!(up.len() % 2, 0, "output must be whole frames");
        for f in up.chunks_exact(2) {
            assert!(f[0] > 0, "left stayed left: {up:?}");
            assert!(f[1] < 0, "right stayed right: {up:?}");
        }
        // 1.5x the frames, not 1.5x the samples.
        assert_eq!(up.len() / 2, 12);

        // Downsampling too.
        let down = resample_ch(&src, 12000, 8000, 2);
        for f in down.chunks_exact(2) {
            assert!(f[0] > 0 && f[1] < 0);
        }

        // Mono is delegated unchanged, and an equal rate is a passthrough.
        assert_eq!(resample_ch(&[1, 2, 3], 8000, 16000, 1), resample(&[1, 2, 3], 8000, 16000));
        assert_eq!(resample_ch(&src, 8000, 8000, 2), src);
    }

    /// The resampler is what made "hello there" play as "helllloooo theeeere"
    /// when a driver ignored the rate — assert the exact-ratio behaviour.
    #[test_case]
    fn resample_integer_ratios() {
        // Identity: same rate returns the input unchanged.
        assert_eq!(resample(&[1, 2, 3], 16_000, 16_000), alloc::vec![1, 2, 3]);
        // 16k -> 48k is x3 upsampling: length triples (nearest-neighbour).
        let up = resample(&[10, 20], 16_000, 48_000);
        assert_eq!(up.len(), 6);
        assert_eq!(up[0], 10);
        assert_eq!(up[5], 20);
        // 24k -> 48k is x2.
        assert_eq!(resample(&[5, 6, 7], 24_000, 48_000).len(), 6);
        // Downsample 48k -> 16k halves-and-thirds (len scales by ratio).
        assert_eq!(resample(&[0; 6], 48_000, 16_000).len(), 2);
        // hz == 0 is a no-op guard (never divide by zero).
        assert_eq!(resample(&[9], 0, 48_000), alloc::vec![9]);
    }

    #[test_case]
    fn volume_adjust_clamps_and_unmutes() {
        set_volume(100);
        set_muted(false);
        assert_eq!(volume_adjust(50), 100, "cannot exceed 100");
        assert_eq!(volume_adjust(-30), 70);
        assert_eq!(volume_adjust(-1000), 0);
        set_muted(true);
        assert!(muted());
        assert_eq!(volume_adjust(10), 10);
        assert!(!muted(), "raising volume unmutes");
        set_volume(100);
        set_muted(false);
    }

    #[test_case]
    fn apply_output_gain_scales_and_mutes() {
        set_volume(50);
        set_muted(false);
        let mut pcm = [1000i16, -2000, 0];
        apply_output_gain(&mut pcm);
        assert_eq!(pcm, [500, -1000, 0]);
        set_muted(true);
        apply_output_gain(&mut pcm);
        assert_eq!(pcm, [0, 0, 0]);
        set_volume(100);
        set_muted(false);
    }

    #[test_case]
    fn rms_bounds() {
        // Silence is 0.
        assert_eq!(rms(&[0, 0, 0, 0]), 0.0);
        // Empty is 0 (no divide-by-zero).
        assert_eq!(rms(&[]), 0.0);
        // Full-scale square wave ~= 1.0 (within the Newton-sqrt tolerance).
        let full = [i16::MAX, i16::MIN, i16::MAX, i16::MIN];
        let r = rms(&full);
        assert!(r > 0.99 && r <= 1.01, "full-scale RMS ~= 1.0, got {}", r);
    }

    #[test_case]
    fn test_tone_length_matches_duration() {
        // A 200 ms tone at 16 kHz is 3200 samples.
        assert_eq!(test_tone(440, 200, 16_000).len(), 3200);
    }
}
