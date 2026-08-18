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

/// Set once [`autodetect`] has run to completion (success or not).
static DISCOVERY_DONE: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

/// Status-bar chip state: Ready once a PCM device is up, Pending until the
/// first probe, Disabled if that probe found nothing (or the built-in amp
/// refused).
pub fn device_status() -> crate::icons::DeviceStatus {
    crate::icons::device_status(
        is_up(),
        DISCOVERY_DONE.load(core::sync::atomic::Ordering::Relaxed),
    )
}

/// Milliseconds between re-probes when no device is up. A human plugging
/// something in cannot notice this delay; what it stops is a caller in a loop
/// re-walking PCI (and re-selecting a USB alternate setting) on a machine that
/// genuinely has no audio. The best-effort paths — the notification chime, a
/// wasm app's `play` — deliberately stay on [`is_up`] and never probe at all.
const REPROBE_MS: u64 = 1500;

/// When the last re-probe ran, so [`should_reprobe`] can rate-limit it.
static LAST_PROBE_MS: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

/// Whether a play attempt at `now_ms` should re-run discovery.
///
/// Pure so the rate limit is testable (nothing else in this decision is worth a
/// test; the *timing* is the part that would silently regress into either a
/// per-chime PCI scan or a device that is never found).
pub fn should_reprobe(is_up: bool, last_ms: u64, now_ms: u64) -> bool {
    !is_up && (last_ms == 0 || now_ms.saturating_sub(last_ms) >= REPROBE_MS)
}

/// Ensure a sound device is up, re-running discovery if one has appeared since
/// boot. Returns whether there is one now.
///
/// **Discovery used to happen only at boot**, and every play path then asked
/// `is_up()` — so a USB DAC plugged in after boot, or enumerated after
/// `autodetect()` ran, reported "no sound device" for the rest of the session
/// even though a re-probe would have adopted it immediately. That is the whole
/// difference between "this machine has no audio" and "this machine has audio"
/// on hardware whose only output is a USB device, which is exactly the position
/// an Apple Silicon Mac is in here (see [`autodetect`]).
///
/// Cheap when a device is already up (one atomic load and a `Locked` peek), so
/// the play paths can call it unconditionally in place of `is_up()`.
pub fn ensure_up() -> bool {
    if is_up() {
        return true;
    }
    let now = crate::arch::now_ms();
    if should_reprobe(false, LAST_PROBE_MS.load(core::sync::atomic::Ordering::Relaxed), now) {
        LAST_PROBE_MS.store(now.max(1), core::sync::atomic::Ordering::Relaxed);
        autodetect();
    }
    is_up()
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

/// A sound backend, and — the part that matters here — **what probing it
/// touches**.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Backend {
    /// virtio-snd over the `virt` machine's fixed virtio-mmio window.
    VirtioMmio,
    /// virtio-snd as a PCI function.
    VirtioPci,
    /// Intel HDA (PCI).
    Hda,
    /// AC'97 (PCI, x86).
    Ac97,
    /// Sound Blaster 16 (ISA ports, x86).
    Sb16,
    /// USB audio class, over a controller this kernel brought up itself.
    Usb,
}

impl Backend {
    /// Whether probing this backend reads a **PC-shaped address that was never
    /// discovered** — QEMU `virt`'s virtio-mmio window at a compiled-in
    /// `0x0a00_0000`, PCI configuration space, an ISA port.
    ///
    /// On a PC or a VM those reads are harmless: an absent device answers
    /// all-ones. **On Apple Silicon they are fatal.** There is no PCI to
    /// enumerate unless `chitti.wifi` brought APCIE up, and the low GiB the
    /// virtio window lives in is Device-typed with nothing behind it, so the
    /// access takes an external abort — which is not a failed probe, it is the
    /// machine going down. The boot path has always known this (`main.rs` runs
    /// the whole `net`/`sound`/`clipboard` block only in its non-Apple arm);
    /// what it could not do is stop a *later* caller from reaching the same
    /// probes, which is exactly what `ensure_up` then did.
    pub fn probes_undiscovered_hardware(self) -> bool {
        !matches!(self, Backend::Usb)
    }
}

/// The backends [`autodetect`] may probe, in order, on this machine.
///
/// Pure and tested, because the rule it encodes is one a future backend can
/// silently break: adding a probe to `autodetect` without classifying it here is
/// a machine-killer on one platform and invisible on every other.
pub fn probe_order(apple: bool) -> &'static [Backend] {
    use Backend::*;
    if apple {
        // USB first (a DAC on the dwc3 we already brought up). Built-in
        // speaker follow-up is `try_apple_builtin`, gated on the tree, not a
        // Backend probe — it must not appear here or the "no undiscovered
        // hardware" test would have to special-case it.
        &[Usb]
    } else {
        &[VirtioMmio, VirtioPci, Hda, Ac97, Sb16, Usb]
    }
}

/// Discover and bring up the first available sound device: virtio-snd over
/// mmio (aarch64 QEMU `-kernel`), virtio-snd over PCI (QEMU), then **Intel
/// HDA** — VirtualBox (x86 + ARM) and real machines. No-op if none is present.
///
/// The probe list comes from [`probe_order`] rather than being written out
/// inline; see [`Backend::probes_undiscovered_hardware`] for why that is a
/// safety property and not tidiness.
pub fn autodetect() {
    if is_up() {
        DISCOVERY_DONE.store(true, core::sync::atomic::Ordering::Relaxed);
        return;
    }
    #[cfg(target_arch = "aarch64")]
    let apple = crate::arch::aarch64::is_apple();
    #[cfg(not(target_arch = "aarch64"))]
    let apple = false;
    let order = probe_order(apple);

    #[cfg(target_arch = "aarch64")]
    if order.contains(&Backend::VirtioMmio) {
        if let Some(dev) = crate::arch::aarch64::virtio_snd::VirtioSndMmio::probe() {
            init(Box::new(dev));
            DISCOVERY_DONE.store(true, core::sync::atomic::Ordering::Relaxed);
            return;
        }
    }
    if !order.contains(&Backend::VirtioPci) {
        // Apple: skip straight to the portable backends. Written as an early
        // exit rather than wrapping each probe, so a probe added below without a
        // guard cannot be reached here at all.
        if let Some(dev) = usb::UsbAudio::probe() {
            init(dev);
            DISCOVERY_DONE.store(true, core::sync::atomic::Ordering::Relaxed);
            return;
        }
        try_apple_builtin();
        DISCOVERY_DONE.store(true, core::sync::atomic::Ordering::Relaxed);
        return;
    }
    if let Some(dev) = virtio_snd_pci::VirtioSndPci::probe() {
        init(dev);
        DISCOVERY_DONE.store(true, core::sync::atomic::Ordering::Relaxed);
        return;
    }
    if let Some(dev) = hda::Hda::probe() {
        init(dev);
        DISCOVERY_DONE.store(true, core::sync::atomic::Ordering::Relaxed);
        return;
    }
    // Legacy x86 audio: AC'97 (VirtualBox/ICH), then Sound Blaster 16 (ISA).
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(dev) = crate::arch::x86_64::ac97::Ac97::probe() {
            init(dev);
            DISCOVERY_DONE.store(true, core::sync::atomic::Ordering::Relaxed);
            return;
        }
        if let Some(dev) = crate::arch::x86_64::sb16::Sb16::probe() {
            init(dev);
            DISCOVERY_DONE.store(true, core::sync::atomic::Ordering::Relaxed);
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
        DISCOVERY_DONE.store(true, core::sync::atomic::Ordering::Relaxed);
        return;
    }
    // Nothing matched. Everywhere the list above ran, audio is a PCI function,
    // so an unsupported controller (or a PCI-discovery gap) is diagnosable from
    // the class-4 list. (Apple Silicon returned above, before any of it.)
    DISCOVERY_DONE.store(true, core::sync::atomic::Ordering::Relaxed);
    crate::ktrace::log("sound", "no audio device matched — multimedia (class 0x04) PCI devices:");
    #[cfg(target_arch = "aarch64")]
    crate::pci::log_class(0x04);
    #[cfg(target_arch = "x86_64")]
    crate::arch::x86_64::pci::log_class(0x04);
}

/// What an Apple Silicon Mac has instead of an audio controller, read off its
/// own device tree — and which piece of it is unimplemented here.
///
/// **Apple Silicon has no PCI audio function at all**, so the class-4 dump that
/// diagnoses every other machine prints an empty list and reads as "the OS
/// cannot see your sound card". Nothing is missing: built-in audio on these
/// machines is an I2S controller (`apple,mca`) fed by a DMA engine, clocked by
/// `apple,nco`, driving an amplifier over I2C — five separate drivers, none of
/// which exists here. Which DMA engine it is differs by generation and is the
/// part that decides how much work this is: the M1 uses **ADMAC**, a plain MMIO
/// engine (m1n1's `proxyclient/experiments/speaker_amp.py` drives the whole
/// stack in 148 lines), while the M2 routes audio through **SIO**, an RTKit
/// coprocessor that has to be given firmware parameters and a mailbox before it
/// will move a byte (`third_party/m1n1/src/sio.c`).
///
/// So this is a report, not a probe: it only ever *reads* the device tree, and
/// touches no register of any of it. A USB audio device works today through the
/// same dwc3/xHCI path the keyboard and mouse use.
#[cfg(target_arch = "aarch64")]
fn report_apple_audio() {
    // Once per boot: `ensure_up` re-probes on every play attempt, and a machine
    // with no audio would otherwise repeat the whole census into the log each
    // time somebody opened a file.
    static REPORTED: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);
    if REPORTED.swap(true, core::sync::atomic::Ordering::Relaxed) {
        return;
    }
    let dtb = crate::arch::aarch64::boot::boot_x0();
    crate::ktrace::log("sound", "no audio device: this is an Apple Silicon Mac, which has no PCI audio function");
    // `(compatible, what it is, what using it would need)`. Reported in the
    // order the signal flows, so the list reads as the stack that is missing.
    const BLOCKS: &[(&[u8], &str)] = &[
        (b"apple,nco", "audio clock generator (NCO)"),
        (b"apple,mca", "I2S controller (MCA)"),
        (b"apple,admac", "audio DMA engine (ADMAC — plain MMIO)"),
        (b"apple,sio", "the SIO coprocessor (RTKit; DMA for other peripherals)"),
        (b"ti,tas2764", "built-in speaker amplifier, over I2C"),
        (b"cirrus,cs42l84", "headphone jack codec, over I2C"),
        (b"apple,dpaudio", "HDMI/DisplayPort audio (needs the DCP display coprocessor)"),
        (b"apple,aop-audio", "microphone, via the always-on processor"),
    ];
    for (compat, what) in BLOCKS {
        // SAFETY: `boot_x0` is the FDT pointer (or not an FDT, rejected by magic).
        if let Some((base, _)) = unsafe { crate::fdt::reg_of_compatible(dtb, compat) } {
            crate::ktrace::log_fmt(format_args!(
                "sound:   {base:#x} {what} — no driver",
            ));
        }
    }
    // **Which DMA engine feeds the I2S is a property of the machine, not of its
    // generation.** Both an ADMAC and a SIO are present here, and guessing from
    // the SoC ("the M2 moved audio to SIO") got it backwards — the M2 mini's MCA
    // is wired to ADMAC exactly as the M1's is, which is the difference between
    // porting m1n1's 148-line `speaker_amp.py` recipe and bringing up an RTKit
    // coprocessor with firmware. The tree answers it directly: `dmas`' first cell
    // is the engine's phandle. Ask, do not infer.
    let mut dmas = [0u32; 2];
    // SAFETY: as above.
    let n = unsafe { crate::fdt::prop_cells_of_compatible(dtb, b"apple,mca", b"dmas", &mut dmas) };
    if n > 0 {
        // SAFETY: as above.
        match unsafe { crate::fdt::reg_by_phandle(dtb, dmas[0]) } {
            Some((base, _)) => crate::ktrace::log_fmt(format_args!(
                "sound:   the I2S controller's DMA is the engine at {base:#x} (its `dmas` phandle {:#x})",
                dmas[0]
            )),
            None => crate::ktrace::log_fmt(format_args!(
                "sound:   the I2S controller names DMA phandle {:#x}, which resolves to no node",
                dmas[0]
            )),
        }
    }
    crate::ktrace::log(
        "sound",
        "no built-in speaker amp in the tree; a USB audio device works through the same \
         dwc3/xHCI path as the keyboard (chitti.usb)",
    );
}

/// Bring up the Mac's own speaker if the tree names it. Tried **once**: the
/// first write is an amplifier leaving shutdown, and a loop of that is how a
/// failed boot would keep clicking the speaker. USB already ran; this is the
/// fallback, not a re-probe of undiscovered hardware.
fn try_apple_builtin() {
    if is_up() {
        return;
    }
    if !apple::builtin_present() {
        #[cfg(target_arch = "aarch64")]
        report_apple_audio();
        return;
    }
    static TRIED: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);
    if TRIED.swap(true, core::sync::atomic::Ordering::Relaxed) {
        return;
    }
    #[cfg(target_arch = "aarch64")]
    {
        match apple::up() {
            Ok(()) => crate::ktrace::log("sound", "built-in speaker up"),
            Err(e) => crate::ktrace::log_fmt(format_args!("sound: built-in speaker bring-up failed: {e}")),
        }
    }
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
/// Built-in audio on Apple Silicon (NCO -> MCA -> ADMAC -> TAS2764).
pub mod apple;
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
mod submit_tests {
    use super::*;
    use alloc::vec;

    /// A staged block is consumed by its submission, so a second call with
    /// nothing staged is an error rather than a silent replay of the previous
    /// block — "the guest sent nothing" and "the guest sent the same thing" are
    /// different facts, and replaying would stutter rather than go quiet.
    #[test_case]
    fn pcm_is_consumed_by_its_submission() {
        let t = crate::sched::current_task_id();
        discard_staged_pcm(t);
        stage_pcm(t, vec![0i16; 64]);
        // Whether playback itself succeeds depends on a device being up, which a
        // test guest has not got; what is under test is the validation and the
        // consume. Either way the stage must be gone.
        let _ = submit_staged(t, 64, 16_000, 1);
        assert!(
            submit_staged(t, 64, 16_000, 1).is_err(),
            "the stage must not survive a submission"
        );
    }

    /// Every number is re-checked against what was actually staged. A block
    /// played at the wrong rate or channel count is not slightly wrong — it is a
    /// different pitch and duration, which reads as a broken decoder rather than
    /// a bad argument.
    #[test_case]
    fn a_shape_mismatch_is_refused() {
        let t = crate::sched::current_task_id();

        // Sample count disagrees with frames x channels.
        discard_staged_pcm(t);
        stage_pcm(t, vec![0i16; 64]);
        assert!(submit_staged(t, 64, 16_000, 2).is_err(), "64 frames stereo needs 128");

        discard_staged_pcm(t);
        stage_pcm(t, vec![0i16; 10]);
        assert!(submit_staged(t, 64, 16_000, 1).is_err());

        // Channel counts we cannot mix.
        discard_staged_pcm(t);
        stage_pcm(t, vec![0i16; 64]);
        assert!(submit_staged(t, 64, 16_000, 0).is_err());
        discard_staged_pcm(t);
        stage_pcm(t, vec![0i16; 64]);
        assert!(submit_staged(t, 8, 16_000, 8).is_err());

        // Rates outside anything a device accepts. A plausible-but-wrong rate
        // would play, at the wrong pitch.
        for bad in [0u32, 100, 1_000_000] {
            discard_staged_pcm(t);
            stage_pcm(t, vec![0i16; 64]);
            assert!(submit_staged(t, 64, bad, 1).is_err(), "rate {bad} must be refused");
        }
        discard_staged_pcm(t);
    }

    /// A submission with nothing staged must not play whatever was left behind.
    #[test_case]
    fn submitting_without_staging_is_an_error() {
        let t = crate::sched::current_task_id();
        discard_staged_pcm(t);
        assert!(submit_staged(t, 64, 16_000, 1).is_err());
    }

    /// The block bound is what stops one call handing over an unbounded
    /// allocation; a game submits about a frame of audio, far below it.
    #[test_case]
    fn an_oversized_block_is_refused() {
        let t = crate::sched::current_task_id();
        discard_staged_pcm(t);
        let n = MAX_SUBMIT_SAMPLES + 2;
        stage_pcm(t, vec![0i16; n]);
        assert!(submit_staged(t, n, 44_100, 1).is_err());
        discard_staged_pcm(t);
    }

    /// Stages are per-task, so one guest cannot play another's audio — the same
    /// reason capability slots and frame stages are per-task.
    #[test_case]
    fn stages_do_not_leak_between_tasks() {
        let a = crate::sched::current_task_id();
        let b = a + 1;
        discard_staged_pcm(a);
        discard_staged_pcm(b);
        stage_pcm(a, vec![0i16; 64]);
        assert!(submit_staged(b, 64, 16_000, 1).is_err(), "b staged nothing");
        discard_staged_pcm(a);
    }

    /// Free space is reported in samples, not bytes — a guest mixes samples, and
    /// bytes would make every one of them divide by two and get it wrong once.
    #[test_case]
    fn free_space_is_in_samples() {
        assert_eq!(out_free_samples(), out_free_bytes() / 2);
    }
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

    #[test_case]
    fn apple_probes_nothing_it_has_not_discovered() {
        // The rule this pins: on Apple Silicon no probe may read a compiled-in
        // PC address. Those reads are harmless everywhere else — an absent PCI
        // function answers all-ones — and on that machine they take an external
        // abort, which is not a failed probe but the machine going down. It is
        // how `/open <audio>` came to reset a Mac mini.
        for b in probe_order(true) {
            assert!(
                !b.probes_undiscovered_hardware(),
                "{b:?} may not be probed on Apple Silicon"
            );
        }
        assert!(probe_order(true).contains(&Backend::Usb), "USB audio is the one that works there");
        // Everywhere else the full list still runs, in the documented order,
        // with USB last so a headset never displaces a full-duplex codec.
        let pc = probe_order(false);
        assert_eq!(pc.first(), Some(&Backend::VirtioMmio));
        assert_eq!(pc.last(), Some(&Backend::Usb));
        assert!(pc.iter().filter(|b| b.probes_undiscovered_hardware()).count() >= 4);
    }

    #[test_case]
    fn a_device_that_is_up_is_never_re_probed() {
        // The whole point of the rate limit is that the common case costs
        // nothing: with a device up, no elapsed time makes a probe due.
        assert!(!should_reprobe(true, 0, 0));
        assert!(!should_reprobe(true, 1, 10_000_000));
    }

    #[test_case]
    fn the_first_play_probes_and_the_next_ones_wait() {
        // Never probed (0 is the "never" sentinel, and a machine really can be
        // at now_ms 0): probe. Then hold off until the interval has passed, so
        // a caller in a loop does not re-walk the bus per call.
        assert!(should_reprobe(false, 0, 0));
        assert!(!should_reprobe(false, 1_000, 1_000));
        assert!(!should_reprobe(false, 1_000, 1_000 + REPROBE_MS - 1));
        assert!(should_reprobe(false, 1_000, 1_000 + REPROBE_MS));
        // A clock that went backwards (a resume re-anchor) must not lock
        // discovery out forever — saturating, so it simply waits.
        assert!(!should_reprobe(false, 10_000, 5));
    }
}

// ---------------------------------------------------------------------------
// Guest-submitted PCM (AUDIO_SUBMIT)
// ---------------------------------------------------------------------------

/// PCM a guest has produced but the gate has not yet approved.
///
/// Same shape and same reason as `synapse::ui`'s frame staging: the primitive
/// executes in ring 3 and a tenant cannot reach a wasm guest's linear memory,
/// while the call block that carries the arguments is 1920 bytes. So the samples
/// are read on the kernel side — where wasmi bounds-checks the read — parked
/// against the calling task, and the gated call carries only their shape.
/// Reading is not an effect; **playing** is.
static STAGED_PCM: crate::mm::Locked<alloc::collections::BTreeMap<crate::sched::TaskId, alloc::vec::Vec<i16>>> =
    crate::mm::Locked::new(alloc::collections::BTreeMap::new());

/// Longest block a single submission may carry.
///
/// One second at 44.1 kHz stereo. The point is not the duration but that a guest
/// cannot hand over an unbounded allocation in one call; a game submits ~1 frame
/// of audio at a time, three orders of magnitude below this.
pub const MAX_SUBMIT_SAMPLES: usize = 88_200;

/// Park a block of PCM for `task`, replacing any previous one.
pub fn stage_pcm(task: crate::sched::TaskId, pcm: alloc::vec::Vec<i16>) {
    STAGED_PCM.with(|m| {
        m.insert(task, pcm);
    });
}

/// Drop a task's staged PCM, if any.
pub fn discard_staged_pcm(task: crate::sched::TaskId) {
    STAGED_PCM.with(|m| {
        m.remove(&task);
    });
}

/// Play the PCM `task` staged.
///
/// Every number is re-checked against what was actually staged rather than
/// trusted from the call — the image tenant's rule. A mismatch is refused rather
/// than played at the wrong rate: audio at the wrong sample rate is not slightly
/// wrong, it is a different pitch and duration, which reads as a broken decoder.
pub fn submit_staged(
    task: crate::sched::TaskId,
    frames: usize,
    rate: u32,
    channels: u8,
) -> Result<usize, &'static str> {
    let staged = STAGED_PCM.with(|m| m.remove(&task));
    let Some(pcm) = staged else {
        return Err("no pcm staged for this task");
    };
    if channels == 0 || channels > 2 {
        return Err("channels must be 1 or 2");
    }
    // A plausible-but-wrong rate is worse than a refusal: it plays, at the wrong
    // pitch, and sounds like a decoder bug rather than a bad argument.
    if !(4_000..=192_000).contains(&rate) {
        return Err("rate out of range");
    }
    if pcm.len() != frames.saturating_mul(channels as usize) {
        return Err("staged pcm does not match the submitted shape");
    }
    if pcm.len() > MAX_SUBMIT_SAMPLES {
        return Err("block too large");
    }
    play_ch(&pcm, rate, channels).map(|()| frames)
}

/// How much room the output queue has, in **samples** (not bytes).
///
/// The pacing signal a guest needs, and the same one `speech_pump` and the video
/// player use: query, drain that much, submit. Reported in samples because that
/// is what the caller mixes in; bytes would make every guest divide by two and
/// get it wrong once.
pub fn out_free_samples() -> usize {
    out_free_bytes() / 2
}
