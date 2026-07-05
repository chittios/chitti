//! **Sound** — PCM audio in and out for the voice pipeline (`/voice`): a
//! virtio-snd NIC-style driver captures microphone frames and plays synthesized
//! speech. Drivers implement [`SndDevice`] (16-bit signed PCM, mono); the
//! transports are **virtio-sound over virtio-mmio** (aarch64 QEMU `virt`) and
//! **virtio-sound over PCI** (x86 QEMU/VBox), discovered like the net drivers.
//!
//! The virtio-snd protocol (virtio spec §5.14) rides four virtqueues:
//! control(0) / event(1) / tx(2, playback) / rx(3, capture). We poll — no
//! interrupts — and run one output stream and one input stream, S16 mono, at
//! the rate the caller asks for (16 kHz for VAD/STT, 22.05 kHz for TTS).

use crate::mm::Locked;
use alloc::boxed::Box;
use alloc::vec::Vec;

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

/// A PCM sound device: play and capture 16-bit signed mono samples. Poll-driven.
pub trait SndDevice {
    /// Start (or restart) the output stream at `hz`, then queue `pcm` for
    /// playback. Blocks only to enqueue (the device drains asynchronously).
    fn play(&mut self, pcm: &[i16], hz: u32) -> Result<(), &'static str>;
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

/// Bring the sound subsystem up on `dev`.
pub fn init(dev: Box<dyn SndDevice>) {
    SND.with(|s| *s = Some(dev));
    crate::ktrace::log("sound", "PCM device up (S16 mono, poll-driven)");
}

/// True once a sound device has been brought up.
pub fn is_up() -> bool {
    SND.with(|s| s.is_some())
}

/// Discover and bring up the first available sound device (virtio-snd over
/// mmio, then over PCI). No-op if none is present. Called once at boot.
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
    }
}

/// Queue `pcm` (S16 mono at `hz`) for playback.
pub fn play(pcm: &[i16], hz: u32) -> Result<(), &'static str> {
    SND.with(|s| match s.as_mut() {
        Some(d) => d.play(pcm, hz),
        None => Err("no sound device"),
    })
}

/// True while playback is draining. Poll this (with `sched::yield_now`) to wait.
pub fn playing() -> bool {
    SND.with(|s| s.as_mut().map(|d| d.playing()).unwrap_or(false))
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
