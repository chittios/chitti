//! **Intel High Definition Audio (HDA) controller driver** — the audio path on
//! VirtualBox (x86 *and* ARM, "Intel HD Audio" controller), real Intel/ARM
//! machines, and QEMU's `intel-hda`/`ich9-intel-hda`. virtio-sound only exists
//! under QEMU, so per the real-hardware rule this is the driver that makes
//! `/voice` work everywhere else.
//!
//! Poll-driven like the other sound drivers (no interrupts):
//! - controller reset (GCTL.CRST), then **CORB/RIRB** ring buffers for codec
//!   verbs — the robust command path on real hardware (immediate-command
//!   registers are optional in the spec).
//! - a codec-graph walk: find the Audio Function Group, **rank** the output pin
//!   complexes by `CONFIG_DEFAULT` ([`rank_output_pin`] — speaker, then
//!   headphone, then line-out, refusing pins the board wired nowhere and any
//!   SPDIF/HDMI pin belonging to the graphics device), then **search the graph**
//!   from that pin back to a DAC ([`Hda::find_output_path`]). The search matters:
//!   a pin's connection list usually does not contain the DAC directly — the
//!   common shape is `pin <- mixer <- dac` — so pointing the pin straight at "the
//!   first DAC we saw" leaves the path unconnected and the codec mute. Every
//!   widget along the discovered path gets its input select pointed at the next
//!   hop, its amp unmuted (0 dB offset) and power set to D0; the pin also gets
//!   output-enable + EAPD. The input side takes the first wired line-in/mic pin.
//! - one output stream (first output SD, stream tag 1) and one input stream
//!   (SD 0, tag 2). 16-bit mono; native 16 kHz when the converter's PCM caps
//!   allow it, otherwise the stream runs at 48 kHz and this driver repeats
//!   (playback) / averages (capture) 3:1 so the `SndDevice` contract stays
//!   16 kHz.
//!
//! MMIO uses single-instruction accesses on aarch64 (the `hvf: isv` lesson —
//! see CLAUDE.md); DMA memory comes from `mm::alloc_dma` on both arches.

use crate::sound::SndDevice;
use alloc::collections::VecDeque;
use alloc::vec::Vec;

#[cfg(target_arch = "aarch64")]
use crate::pci::{self, PciDevice};
#[cfg(target_arch = "x86_64")]
use crate::arch::x86_64::pci::{self, PciDevice};

// --- controller registers (offsets into BAR0) ----------------------------
const GCAP: u64 = 0x00;
const GCTL: u64 = 0x08;
const STATESTS: u64 = 0x0e;
const ICOI: u64 = 0x60; // immediate command output
const ICII: u64 = 0x64; // immediate command input (response)
const ICIS: u64 = 0x68; // immediate command status (bit0 ICB, bit1 IRV)
const CORBLBASE: u64 = 0x40;
const CORBUBASE: u64 = 0x44;
const CORBWP: u64 = 0x48;
const CORBRP: u64 = 0x4a;
const CORBCTL: u64 = 0x4c;
const CORBSIZE: u64 = 0x4e;
const RIRBLBASE: u64 = 0x50;
const RIRBUBASE: u64 = 0x54;
const RIRBWP: u64 = 0x58;
const RINTCNT: u64 = 0x5a;
const RIRBCTL: u64 = 0x5c;
const RIRBSTS: u64 = 0x5d;
const RIRBSIZE: u64 = 0x5e;
const SD_BASE: u64 = 0x80;
const SD_STRIDE: u64 = 0x20;
// Stream-descriptor register offsets.
const SD_CTL: u64 = 0x00; // 24-bit (byte 2 carries the stream tag)
const SD_STS: u64 = 0x03;
const SD_LPIB: u64 = 0x04;
const SD_CBL: u64 = 0x08;
const SD_LVI: u64 = 0x0c;
const SD_FMT: u64 = 0x12;
const SD_BDPL: u64 = 0x18;
const SD_BDPU: u64 = 0x1c;

const SDCTL_SRST: u32 = 1 << 0;
const SDCTL_RUN: u32 = 1 << 1;
const SDCTL_IOCE: u32 = 1 << 2;
const SDSTS_BCIS: u8 = 1 << 2;

// --- codec verbs ----------------------------------------------------------
const V_GET_PARAM: u32 = 0xf00;
const V_CONN_SELECT: u32 = 0x701;
const V_GET_CONN_LIST: u32 = 0xf02;
const V_SET_POWER: u32 = 0x705;
const V_SET_STREAM: u32 = 0x706;
const V_SET_PIN_CTL: u32 = 0x707;
const V_SET_EAPD: u32 = 0x70c;
const V_SET_FORMAT: u32 = 0x200; // 4-bit verb with 16-bit payload
const V_SET_AMP: u32 = 0x300; // 4-bit verb with 16-bit payload
// GET_PARAM parameter ids.
const P_SUB_NODES: u32 = 0x04;
const P_FG_TYPE: u32 = 0x05;
const P_WIDGET_CAP: u32 = 0x09;
const P_PCM_CAPS: u32 = 0x0a;
const P_CONN_LEN: u32 = 0x0e;
const P_OUT_AMP: u32 = 0x12;
/// Pin capabilities (GET_PARAM 0x0c).
const P_PIN_CAPS: u32 = 0x0c;
/// PIN_CAP bit 4: this pin complex can drive an output.
const PIN_CAP_OUTPUT: u32 = 1 << 4;
/// GET_CONFIG_DEFAULT verb — the pin's platform wiring description.
const V_GET_CONFIG_DEFAULT: u32 = 0xf1c;

/// Widget types (WIDGET_CAP bits 23:20).
const WIDGET_AUDIO_OUT: u32 = 0x0;
const WIDGET_MIXER: u32 = 0x2;
const WIDGET_SELECTOR: u32 = 0x3;
const WIDGET_PIN: u32 = 0x4;

/// How far to search the widget graph from a pin back to a DAC. Real codecs need
/// 2-3 hops (`pin <- mixer <- dac`); the bound stops a malformed or cyclic
/// connection list from recursing without end.
const MAX_PATH_DEPTH: usize = 6;

const PIN_OUT_EN: u32 = 0x40;
const PIN_IN_EN: u32 = 0x20;

/// Rank an output pin from its `CONFIG_DEFAULT`, lower being more desirable, or
/// `None` if it is not an output pin we should use.
///
/// Two things beyond the device type matter, and both are why picking the
/// lowest-numbered output pin misbehaves on real machines:
///
/// * **Connectivity** (bits 31:30). `1` means "no physical connection" — the
///   codec exposes the pin but the board wired it nowhere. Selecting one of those
///   gives a path that configures cleanly and produces silence, so it is refused
///   outright.
/// * **Device type** (bits 23:20). A laptop's speaker is the right default; a
///   headphone jack is next; a line-out is last, since many laptops report one
///   they do not physically have. HDMI/SPDIF outputs belong to the graphics
///   device and must never be chosen as the system output.
pub(crate) fn rank_output_pin(cfg: u32) -> Option<u8> {
    let connectivity = (cfg >> 30) & 0x3;
    if connectivity == 1 {
        return None; // no physical connection
    }
    match (cfg >> 20) & 0xf {
        0x1 => Some(0), // speaker
        0x2 => Some(1), // headphone out
        0x0 => Some(2), // line out
        _ => None,      // 3..7 = SPDIF/HDMI/other digital, 8.. = inputs
    }
}

/// Whether a pin's `CONFIG_DEFAULT` describes an input we can capture from
/// (line-in or mic), and that is actually wired up.
pub(crate) fn is_input_pin(cfg: u32) -> bool {
    if (cfg >> 30) & 0x3 == 1 {
        return false; // no physical connection
    }
    matches!((cfg >> 20) & 0xf, 0x8 | 0xa)
}

/// PCM chunk for the capture ring: 100 ms of S16 mono at 48 kHz (worst case).
const CAP_CHUNK: usize = 9600;
const CAP_SLOTS: usize = 4;

// --- MMIO access: single-instruction on aarch64 ---------------------------
#[cfg(target_arch = "aarch64")]
mod mmio {
    pub unsafe fn r8(a: u64) -> u8 {
        let v: u32;
        unsafe { core::arch::asm!("ldrb {v:w}, [{a}]", v = out(reg) v, a = in(reg) a, options(nostack, preserves_flags)) };
        v as u8
    }
    pub unsafe fn w8(a: u64, v: u8) {
        unsafe { core::arch::asm!("strb {v:w}, [{a}]", v = in(reg) v as u32, a = in(reg) a, options(nostack, preserves_flags)) };
    }
    pub unsafe fn r16(a: u64) -> u16 {
        let v: u32;
        unsafe { core::arch::asm!("ldrh {v:w}, [{a}]", v = out(reg) v, a = in(reg) a, options(nostack, preserves_flags)) };
        v as u16
    }
    pub unsafe fn w16(a: u64, v: u16) {
        unsafe { core::arch::asm!("strh {v:w}, [{a}]", v = in(reg) v as u32, a = in(reg) a, options(nostack, preserves_flags)) };
    }
    pub unsafe fn r32(a: u64) -> u32 {
        let v: u32;
        unsafe { core::arch::asm!("ldr {v:w}, [{a}]", v = out(reg) v, a = in(reg) a, options(nostack, preserves_flags)) };
        v
    }
    pub unsafe fn w32(a: u64, v: u32) {
        unsafe { core::arch::asm!("str {v:w}, [{a}]", v = in(reg) v, a = in(reg) a, options(nostack, preserves_flags)) };
    }
}
#[cfg(target_arch = "x86_64")]
mod mmio {
    use core::ptr::{read_volatile, write_volatile};
    pub unsafe fn r8(a: u64) -> u8 {
        unsafe { read_volatile(a as *const u8) }
    }
    pub unsafe fn w8(a: u64, v: u8) {
        unsafe { write_volatile(a as *mut u8, v) };
    }
    pub unsafe fn r16(a: u64) -> u16 {
        unsafe { read_volatile(a as *const u16) }
    }
    pub unsafe fn w16(a: u64, v: u16) {
        unsafe { write_volatile(a as *mut u16, v) };
    }
    pub unsafe fn r32(a: u64) -> u32 {
        unsafe { read_volatile(a as *const u32) }
    }
    pub unsafe fn w32(a: u64, v: u32) {
        unsafe { write_volatile(a as *mut u32, v) };
    }
}
use mmio::*;

fn spin_wait(mut tries: u32, mut done: impl FnMut() -> bool) -> bool {
    while tries > 0 {
        if done() {
            return true;
        }
        core::hint::spin_loop();
        tries -= 1;
    }
    false
}

/// The poll-driven HDA device.
pub struct Hda {
    regs: u64,
    codec: u32, // codec address
    // CORB/RIRB rings (DMA): 256 entries each.
    corb: (u64, u64), // (phys, virt)
    rirb: (u64, u64),
    corb_wp: u16,
    // Command transport: some controllers implement only one. QEMU's ich9 uses
    // the immediate-command registers; VirtualBox's ICH6 uses CORB/RIRB. We
    // detect which returns a sane codec id and use that.
    imm: bool,
    // Codec graph results.
    dac: u32,
    adc: u32,
    // Stream descriptor register bases.
    out_sd: u64,
    in_sd: u64,
    // Whether the converters do native 16 kHz (else 48 kHz + 3:1 in software).
    native_16k: bool,
    // Playback state.
    play_open: bool,
    // Capture state.
    cap_ring: Option<(u64, u64)>, // (phys, virt) CAP_SLOTS*CAP_CHUNK ring
    cap_pos: usize,               // last LPIB byte position we consumed
    cap_on: bool,
    pending: VecDeque<i16>,
    cap_decim: [i32; 2], // (accumulated sum, count) for 48k->16k averaging
}

impl Hda {
    /// Find an HDA controller (PCI class 04.03) and bring it up.
    pub fn probe() -> Option<alloc::boxed::Box<dyn SndDevice>> {
        let d = match pci::find_class_sub(0x04, 0x03) {
            Some(d) => d,
            None => {
                crate::ktrace::log("hda", "probe: no PCI class 04.03 device found");
                return None;
            }
        };
        crate::ktrace::log_fmt(format_args!("hda: found controller {:04x}:{:04x} at {}:{}.{}", d.vendor, d.device, d.bus, d.dev, d.func));
        match Self::init(d) {
            Some(h) => Some(alloc::boxed::Box::new(h) as alloc::boxed::Box<dyn SndDevice>),
            None => {
                crate::ktrace::log("hda", "init failed (reset/codec bring-up)");
                None
            }
        }
    }

    fn init(d: PciDevice) -> Option<Hda> {
        d.enable_bus_master();
        let bar = d.bar(0);
        if bar == 0 {
            return None;
        }
        let regs = crate::mm::map_mmio(bar, 0x4000);
        // SAFETY: `regs` is the mapped HDA register block; standard bring-up.
        unsafe {
            // Reset the controller: clear then set CRST, waiting on each edge.
            w32(regs + GCTL, r32(regs + GCTL) & !1);
            if !spin_wait(5_000_000, || r32(regs + GCTL) & 1 == 0) {
                crate::ktrace::log("hda", "reset: CRST clear timed out");
                return None;
            }
            w32(regs + GCTL, r32(regs + GCTL) | 1);
            if !spin_wait(5_000_000, || r32(regs + GCTL) & 1 == 1) {
                crate::ktrace::log("hda", "reset: CRST set timed out");
                return None;
            }
            // Codecs report presence in STATESTS shortly after reset.
            spin_wait(5_000_000, || r16(regs + STATESTS) != 0);
            let statests = r16(regs + STATESTS);
            let codec = match (0..15).find(|&c| statests & (1 << c) != 0) {
                Some(c) => c as u32,
                None => {
                    crate::ktrace::log("hda", "no codec present in STATESTS");
                    return None;
                }
            };

            // CORB/RIRB: 256 entries (CORB 1 KiB, RIRB 2 KiB).
            let corb = crate::mm::alloc_dma(1024)?;
            let rirb = crate::mm::alloc_dma(2048)?;
            // Stop DMA engines while programming.
            w8(regs + CORBCTL, 0);
            w8(regs + RIRBCTL, 0);
            w32(regs + CORBLBASE, corb.0 as u32);
            w32(regs + CORBUBASE, (corb.0 >> 32) as u32);
            w8(regs + CORBSIZE, 0x02); // 256 entries
            // Reset CORB read pointer.
            w16(regs + CORBRP, 1 << 15);
            spin_wait(100_000, || r16(regs + CORBRP) & (1 << 15) != 0);
            w16(regs + CORBRP, 0);
            spin_wait(100_000, || r16(regs + CORBRP) & (1 << 15) == 0);
            w16(regs + CORBWP, 0);
            w32(regs + RIRBLBASE, rirb.0 as u32);
            w32(regs + RIRBUBASE, (rirb.0 >> 32) as u32);
            w8(regs + RIRBSIZE, 0x02);
            w16(regs + RIRBWP, 1 << 15); // reset write pointer
            w16(regs + RINTCNT, 1);
            // Run both rings.
            w8(regs + CORBCTL, 0x02);
            w8(regs + RIRBCTL, 0x02);

            let gcap = r16(regs + GCAP);
            let iss = ((gcap >> 8) & 0xf) as u64; // input streams come first
            let out_sd = regs + SD_BASE + iss * SD_STRIDE; // first output SD
            let in_sd = regs + SD_BASE; // first input SD

            let mut hda = Hda {
                regs,
                codec,
                corb,
                rirb,
                corb_wp: 0,
                imm: false,
                dac: 0,
                adc: 0,
                out_sd,
                in_sd,
                native_16k: false,
                play_open: false,
                cap_ring: None,
                cap_pos: 0,
                cap_on: false,
                pending: VecDeque::new(),
                cap_decim: [0; 2],
            };
            // Detect the command transport. Try the immediate-command registers
            // first (QEMU ich9 answers there and its CORB is quirky); if they
            // time out (VirtualBox's ICH6 doesn't implement them) fall back to
            // CORB/RIRB. A valid codec vendor id is non-zero and not all-ones.
            hda.imm = true;
            let mut vid = hda.param(0, 0x00);
            if vid == 0 || vid == 0xffff_ffff {
                hda.imm = false;
                vid = hda.param(0, 0x00);
            }
            crate::ktrace::log_fmt(format_args!("hda: transport={} codec vid={vid:#x}", if hda.imm { "immediate" } else { "CORB/RIRB" }));
            if !hda.codec_setup() {
                crate::ktrace::log("hda", "codec_setup failed (no DAC/output pin)");
                return None;
            }
            crate::ktrace::log_fmt(format_args!(
                "hda: up (vendor {:04x} dev {:04x}), codec {} dac {} adc {}, {}",
                d.vendor,
                d.device,
                codec,
                hda.dac,
                hda.adc,
                if hda.native_16k { "native 16k" } else { "48k + 3:1" }
            ));
            Some(hda)
        }
    }

    /// Send one verb through CORB and pull its response from RIRB.
    fn verb(&mut self, nid: u32, verb: u32, payload: u32) -> u32 {
        // 12-bit verb + 8-bit payload, or 4-bit verb + 16-bit payload.
        let cmd = if verb & 0xf00 == verb && verb <= 0xf00 || verb > 0xff {
            // 12-bit verb id (e.g. 0xf00, 0x701…), 8-bit payload
            (self.codec << 28) | (nid << 20) | (verb << 8) | (payload & 0xff)
        } else {
            (self.codec << 28) | (nid << 20) | (verb << 8) | (payload & 0xff)
        };
        self.push_verb(cmd)
    }
    /// 4-bit verb (SET_FORMAT/SET_AMP) with a 16-bit payload.
    fn verb16(&mut self, nid: u32, verb4: u32, payload: u32) -> u32 {
        let cmd = (self.codec << 28) | (nid << 20) | ((verb4 >> 8) << 16) | (payload & 0xffff);
        self.push_verb(cmd)
    }

    /// Send `cmd` and return the codec response, over whichever transport this
    /// controller implements (`imm` selects immediate-command registers vs the
    /// CORB/RIRB DMA rings). QEMU's ich9 only answers on the immediate registers;
    /// VirtualBox's ICH6 only on CORB/RIRB — so we support both.
    fn push_verb(&mut self, cmd: u32) -> u32 {
        if self.imm {
            self.push_verb_imm(cmd)
        } else {
            self.push_verb_corb(cmd)
        }
    }

    /// Immediate-command path (ICOI/ICII/ICIS).
    fn push_verb_imm(&mut self, cmd: u32) -> u32 {
        // SAFETY: immediate-command registers on the live controller.
        unsafe {
            let s = self.regs;
            spin_wait(1_000_000, || r16(s + ICIS) & 0x1 == 0);
            w32(s + ICOI, cmd);
            w16(s + ICIS, 0b11); // set ICB, write-1-clear IRV
            if !spin_wait(2_000_000, || r16(s + ICIS) & 0x2 != 0) {
                return 0;
            }
            let resp = r32(s + ICII);
            w16(s + ICIS, 0x2);
            resp
        }
    }

    /// CORB/RIRB DMA-ring path. CORB and RIRB stay in lockstep (one response per
    /// command), so the response for the command at write-pointer `i` lands in
    /// `RIRB[i]`.
    fn push_verb_corb(&mut self, cmd: u32) -> u32 {
        // SAFETY: ring DMA memory + controller registers, live for its lifetime.
        unsafe {
            let s = self.regs;
            self.corb_wp = if self.corb_wp >= 255 { 1 } else { self.corb_wp + 1 };
            core::ptr::write_volatile((self.corb.1 + self.corb_wp as u64 * 4) as *mut u32, cmd);
            w16(s + CORBWP, self.corb_wp);
            if !spin_wait(2_000_000, || (r16(s + RIRBWP) & 0xff) == self.corb_wp) {
                return 0;
            }
            core::ptr::read_volatile((self.rirb.1 + self.corb_wp as u64 * 8) as *const u32)
        }
    }

    fn param(&mut self, nid: u32, p: u32) -> u32 {
        self.verb(nid, V_GET_PARAM, p)
    }

    /// Walk the codec: locate the AFG, first DAC + output pin, first ADC +
    /// input pin; wire and unmute the paths.
    fn codec_setup(&mut self) -> bool {
        // Root node 0 → function groups.
        let sub = self.param(0, P_SUB_NODES);
        let (fg_start, fg_count) = ((sub >> 16) & 0xff, sub & 0xff);
        let mut afg = 0;
        for fg in fg_start..fg_start + fg_count {
            if self.param(fg, P_FG_TYPE) & 0xff == 0x01 {
                afg = fg;
                break;
            }
        }
        if afg == 0 {
            return false;
        }
        self.verb(afg, V_SET_POWER, 0); // D0

        let sub = self.param(afg, P_SUB_NODES);
        let (w_start, w_count) = ((sub >> 16) & 0xff, sub & 0xff);
        let (mut adc, mut in_pin) = (0u32, 0u32);
        // Output pin: rank the candidates instead of taking the lowest node id.
        // A real codec exposes several output pins and the first one in node
        // order is regularly the wrong one — a rear line-out on a laptop that has
        // none, or an HDMI pin belonging to the graphics device.
        let mut best_out: Option<(u8, u32)> = None; // (rank, nid)
        for nid in w_start..w_start + w_count {
            let cap = self.param(nid, P_WIDGET_CAP);
            match (cap >> 20) & 0xf {
                0x1 if adc == 0 => adc = nid, // audio input (ADC)
                0x4 => {
                    let cfg = self.verb(nid, V_GET_CONFIG_DEFAULT, 0);
                    if let Some(rank) = rank_output_pin(cfg) {
                        // Only consider pins the codec says can actually drive
                        // output (PIN_CAP bit 4).
                        if self.param(nid, P_PIN_CAPS) & PIN_CAP_OUTPUT != 0
                            && best_out.is_none_or(|(r, _)| rank < r)
                        {
                            best_out = Some((rank, nid));
                        }
                    } else if is_input_pin(cfg) && in_pin == 0 {
                        in_pin = nid;
                    }
                }
                _ => {}
            }
        }
        let Some((_, out_pin)) = best_out else {
            crate::ktrace::log("hda", "no usable output pin complex on this codec");
            return false;
        };
        // Walk the widget graph from the pin back to a DAC. The DAC is very often
        // NOT directly in the pin's connection list — Realtek and friends put a
        // mixer or selector in between — so pointing the pin straight at "the
        // first DAC we saw" silently leaves the path unconnected and the codec
        // mute. `path` comes back as [pin, .., dac].
        let mut path = alloc::vec::Vec::new();
        if !self.find_output_path(out_pin, 0, &mut path) {
            crate::ktrace::log_fmt(format_args!(
                "hda: no DAC reachable from output pin {out_pin} (searched {} levels)",
                MAX_PATH_DEPTH
            ));
            return false;
        }
        let dac = *path.last().unwrap();
        crate::ktrace::log_fmt(format_args!("hda: output path {path:?} (pin -> dac)"));
        self.dac = dac;
        self.adc = adc;

        // 16 kHz native? PCM caps bit 2 (rates: 8, 11.025, 16, 22.05, …).
        let pcm = {
            let p = self.param(dac, P_PCM_CAPS);
            if p != 0 { p } else { self.param(afg, P_PCM_CAPS) }
        };
        self.native_16k = pcm & (1 << 2) != 0;

        // Power up and wire every widget along the discovered path, not just the
        // two ends: each selector/mixer in between needs its input select pointed
        // at the next hop and its amp unmuted, or the signal stops there.
        for &n in &path {
            self.verb(n, V_SET_POWER, 0);
            self.unmute(n);
        }
        for pair in path.windows(2) {
            self.select_connection(pair[0], pair[1]);
        }
        self.verb(out_pin, V_SET_PIN_CTL, PIN_OUT_EN);
        self.verb(out_pin, V_SET_EAPD, 0x02);

        // Input path (optional — capture still degrades gracefully without).
        if adc != 0 && in_pin != 0 {
            for &n in &[adc, in_pin] {
                self.verb(n, V_SET_POWER, 0);
            }
            self.select_connection(adc, in_pin);
            self.verb(in_pin, V_SET_PIN_CTL, PIN_IN_EN);
            self.unmute(adc);
            self.unmute(in_pin);
        }
        true
    }

    /// Depth-first search from a pin complex back to an audio-output (DAC)
    /// widget, through selectors and mixers, appending `[pin, .., dac]` to
    /// `chain`. Returns false if no DAC is reachable within
    /// [`MAX_PATH_DEPTH`] hops.
    ///
    /// This exists because a pin's connection list usually does **not** contain
    /// the DAC directly: the common shape is `pin <- mixer <- dac` or
    /// `pin <- selector <- mixer <- dac`.
    fn find_output_path(&mut self, nid: u32, depth: usize, chain: &mut alloc::vec::Vec<u32>) -> bool {
        if depth >= MAX_PATH_DEPTH || chain.contains(&nid) {
            return false; // too deep, or a cycle in the graph
        }
        chain.push(nid);
        let wtype = (self.param(nid, P_WIDGET_CAP) >> 20) & 0xf;
        if wtype == WIDGET_AUDIO_OUT {
            return true; // reached a DAC
        }
        // Pin complexes, selectors and mixers are all worth descending through;
        // anything else (power widget, beep generator, vendor) is a dead end.
        if !matches!(wtype, WIDGET_PIN | WIDGET_SELECTOR | WIDGET_MIXER) {
            chain.pop();
            return false;
        }
        let len = self.param(nid, P_CONN_LEN) & 0x7f;
        for i in 0..len {
            let resp = self.verb(nid, V_GET_CONN_LIST, i & !3);
            let entry = (resp >> ((i & 3) * 8)) & 0xff;
            if entry == 0 {
                continue;
            }
            if self.find_output_path(entry, depth + 1, chain) {
                return true;
            }
        }
        chain.pop();
        false
    }

    /// Point `widget`'s connection selector at `target` if it appears in the
    /// widget's connection list.
    fn select_connection(&mut self, widget: u32, target: u32) {
        let len = self.param(widget, P_CONN_LEN) & 0x7f;
        for i in 0..len {
            // Short-form entries: 4 per response, 8 bits each.
            let resp = self.verb(widget, V_GET_CONN_LIST, i & !3);
            let entry = (resp >> ((i & 3) * 8)) & 0xff;
            if entry == target {
                self.verb(widget, V_CONN_SELECT, i);
                return;
            }
        }
    }

    /// Unmute a widget's output amp at 0 dB (the amp-cap offset step).
    fn unmute(&mut self, nid: u32) {
        let cap = {
            let c = self.param(nid, P_OUT_AMP);
            if c != 0 { c } else { 0x1f }
        };
        let gain = cap & 0x7f;
        // Set output amp, left+right: 0xB000 | gain.
        self.verb16(nid, V_SET_AMP, 0xb000 | gain);
    }

    /// Stream format word: 16-bit mono at 16 kHz (native) or 48 kHz.
    fn fmt(&self) -> u16 {
        if self.native_16k {
            // base 48k, mult 1, div 3 (48/3=16): div field = 2; bits=16 (01).
            (2 << 8) | (1 << 4)
        } else {
            1 << 4 // 48 kHz, 16-bit, mono
        }
    }
    fn hw_rate(&self) -> usize {
        if self.native_16k {
            16000
        } else {
            48000
        }
    }

    /// Reset a stream descriptor.
    unsafe fn sd_reset(&self, sd: u64) {
        unsafe {
            w32(sd + SD_CTL, SDCTL_SRST);
            spin_wait(100_000, || r32(sd + SD_CTL) & SDCTL_SRST != 0);
            w32(sd + SD_CTL, 0);
            spin_wait(100_000, || r32(sd + SD_CTL) & SDCTL_SRST == 0);
        }
    }
}

impl SndDevice for Hda {
    fn play(&mut self, pcm: &[i16], hz: u32) -> Result<(), &'static str> {
        // Playback always runs the stream at 48 kHz (every codec's base rate)
        // and resamples the input to it — `hz` varies by caller: 16 kHz test
        // tones/mic loops, 24 kHz KittenTTS. (Capture keeps the native-16k
        // path; see `fmt`/`hw_rate`.)
        const PLAY_HZ: u32 = 48_000;
        const PLAY_FMT: u16 = 1 << 4; // base 48 kHz, 16-bit, mono
        // Single-shot DMA: refuse to start a new transfer while one is running
        // (speech_pump waits for !playing() via out_free_bytes).
        if self.playing() {
            return Err("hda: still playing");
        }
        let samples: Vec<i16> = crate::sound::resample(pcm, hz, PLAY_HZ);
        let bytes = samples.len() * 2;
        if bytes == 0 {
            return Ok(());
        }
        let buf = crate::mm::alloc_dma(bytes).ok_or("hda: DMA alloc failed")?;
        let bdl = crate::mm::alloc_dma(32).ok_or("hda: BDL alloc failed")?;
        // SAFETY: fresh DMA regions; stream registers are the device's.
        unsafe {
            core::ptr::copy_nonoverlapping(samples.as_ptr() as *const u8, buf.1 as *mut u8, bytes);
            // Two BDL entries (the spec minimum), IOC on the last.
            let half = (bytes / 2) & !1;
            let e = bdl.1;
            core::ptr::write_volatile(e as *mut u64, buf.0);
            core::ptr::write_volatile((e + 8) as *mut u32, half as u32);
            core::ptr::write_volatile((e + 12) as *mut u32, 0);
            core::ptr::write_volatile((e + 16) as *mut u64, buf.0 + half as u64);
            core::ptr::write_volatile((e + 24) as *mut u32, (bytes - half) as u32);
            core::ptr::write_volatile((e + 28) as *mut u32, 1); // IOC
            let sd = self.out_sd;
            self.sd_reset(sd);
            w8(sd + SD_STS, SDSTS_BCIS); // clear stale completion
            w32(sd + SD_BDPL, bdl.0 as u32);
            w32(sd + SD_BDPU, (bdl.0 >> 32) as u32);
            w32(sd + SD_CBL, bytes as u32);
            w16(sd + SD_LVI, 1);
            w16(sd + SD_FMT, PLAY_FMT);
            // Bind the converter to stream tag 1, channel 0 + format.
            self.verb(self.dac, V_SET_STREAM, 1 << 4);
            self.verb16(self.dac, V_SET_FORMAT, PLAY_FMT as u32);
            // Tag 1 into CTL byte 2, IOC enable, run.
            w32(sd + SD_CTL, (1 << 20) | SDCTL_IOCE | SDCTL_RUN);
        }
        self.play_open = true;
        Ok(())
    }

    /// Free bytes the speech pump can enqueue: whole second when idle, 0 while
    /// the single DMA buffer is still running.
    fn out_free_bytes(&mut self) -> usize {
        if self.playing() {
            0
        } else {
            // 1 s of 24 kHz mono S16 — matches the speech_pump fallback batch.
            24_000 * 2
        }
    }

    fn playing(&mut self) -> bool {
        if !self.play_open {
            return false;
        }
        // SAFETY: live stream registers.
        unsafe {
            let sd = self.out_sd;
            // Buffer completion (IOC) *or* stream no longer RUN: some firmwares
            // clear RUN without setting BCIS — treat either as done so we never
            // hang the speech drain loop.
            let sts = r8(sd + SD_STS);
            let ctl = r32(sd + SD_CTL);
            if sts & SDSTS_BCIS != 0 || ctl & SDCTL_RUN == 0 {
                w32(sd + SD_CTL, 0);
                w8(sd + SD_STS, SDSTS_BCIS);
                self.play_open = false;
                return false;
            }
        }
        true
    }

    fn capture_start(&mut self, _hz: u32) -> Result<(), &'static str> {
        if self.cap_on {
            return Ok(());
        }
        if self.adc == 0 {
            return Err("hda: codec has no ADC/mic path");
        }
        let ring = match self.cap_ring {
            Some(r) => r,
            None => {
                let r = crate::mm::alloc_dma(CAP_SLOTS * CAP_CHUNK).ok_or("hda: DMA alloc failed")?;
                self.cap_ring = Some(r);
                r
            }
        };
        let bdl = crate::mm::alloc_dma(CAP_SLOTS * 16).ok_or("hda: BDL alloc failed")?;
        // SAFETY: fresh DMA + live registers.
        unsafe {
            for i in 0..CAP_SLOTS {
                let e = bdl.1 + (i * 16) as u64;
                core::ptr::write_volatile(e as *mut u64, ring.0 + (i * CAP_CHUNK) as u64);
                core::ptr::write_volatile((e + 8) as *mut u32, CAP_CHUNK as u32);
                core::ptr::write_volatile((e + 12) as *mut u32, 0);
            }
            let sd = self.in_sd;
            self.sd_reset(sd);
            w32(sd + SD_BDPL, bdl.0 as u32);
            w32(sd + SD_BDPU, (bdl.0 >> 32) as u32);
            w32(sd + SD_CBL, (CAP_SLOTS * CAP_CHUNK) as u32);
            w16(sd + SD_LVI, (CAP_SLOTS - 1) as u16);
            w16(sd + SD_FMT, self.fmt());
            self.verb(self.adc, V_SET_STREAM, 2 << 4);
            self.verb16(self.adc, V_SET_FORMAT, self.fmt() as u32);
            w32(sd + SD_CTL, (2 << 20) | SDCTL_RUN);
        }
        self.cap_pos = 0;
        self.cap_on = true;
        self.pending.clear();
        self.cap_decim = [0; 2];
        Ok(())
    }

    fn capture_read(&mut self, out: &mut [i16]) -> usize {
        if self.cap_on {
            let ring = self.cap_ring.unwrap();
            // SAFETY: live registers + the capture ring DMA region.
            unsafe {
                let lpib = r32(self.in_sd + SD_LPIB) as usize % (CAP_SLOTS * CAP_CHUNK);
                let total = CAP_SLOTS * CAP_CHUNK;
                let avail = (lpib + total - self.cap_pos) % total & !1;
                let decim = self.hw_rate() / 16000; // 1 or 3
                let mut read = 0usize;
                while read < avail {
                    let b = (self.cap_pos + read) % total;
                    let lo = core::ptr::read_volatile((ring.1 + b as u64) as *const u8) as u16;
                    let hi = core::ptr::read_volatile((ring.1 + b as u64 + 1) as *const u8) as u16;
                    let s = (lo | (hi << 8)) as i16;
                    // Average `decim` samples down to one (48 kHz → 16 kHz).
                    self.cap_decim[0] += s as i32;
                    self.cap_decim[1] += 1;
                    if self.cap_decim[1] as usize == decim {
                        self.pending.push_back((self.cap_decim[0] / decim as i32) as i16);
                        self.cap_decim = [0; 2];
                    }
                    read += 2;
                }
                self.cap_pos = (self.cap_pos + read) % total;
            }
        }
        let mut n = 0;
        while n < out.len() {
            match self.pending.pop_front() {
                Some(s) => {
                    out[n] = s;
                    n += 1;
                }
                None => break,
            }
        }
        n
    }

    fn capture_stop(&mut self) {
        if self.cap_on {
            // SAFETY: live stream registers.
            unsafe {
                w32(self.in_sd + SD_CTL, 0);
            }
            self.cap_on = false;
            self.pending.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a CONFIG_DEFAULT with the given connectivity and device type.
    fn cfg(connectivity: u32, device: u32) -> u32 {
        (connectivity << 30) | (device << 20)
    }

    #[test_case]
    fn output_pin_ranking_prefers_speaker_then_headphone_then_line_out() {
        let spk = rank_output_pin(cfg(2, 0x1)).unwrap(); // fixed internal speaker
        let hp = rank_output_pin(cfg(0, 0x2)).unwrap(); // headphone jack
        let line = rank_output_pin(cfg(0, 0x0)).unwrap(); // line out
        assert!(spk < hp, "speaker must outrank headphone");
        assert!(hp < line, "headphone must outrank line-out");
    }

    #[test_case]
    fn unconnected_pins_are_refused() {
        // Connectivity 1 = "no physical connection": the codec reports the pin
        // but the board wired it nowhere. Choosing it yields a path that
        // configures fine and plays silence.
        assert_eq!(rank_output_pin(cfg(1, 0x1)), None);
        assert_eq!(rank_output_pin(cfg(1, 0x2)), None);
        assert!(!is_input_pin(cfg(1, 0xa)));
        // The same device types ARE accepted when connectivity says otherwise.
        assert!(rank_output_pin(cfg(2, 0x1)).is_some());
        assert!(is_input_pin(cfg(0, 0xa)));
    }

    #[test_case]
    fn digital_and_input_pins_are_not_output_candidates() {
        // SPDIF / HDMI belong to the graphics device — never the system output.
        for dev in [0x3, 0x4, 0x5, 0x6, 0x7] {
            assert_eq!(rank_output_pin(cfg(0, dev)), None, "device {dev:#x}");
        }
        // Input device types must not be picked as outputs either.
        assert_eq!(rank_output_pin(cfg(0, 0x8)), None); // line in
        assert_eq!(rank_output_pin(cfg(0, 0xa)), None); // mic
    }

    #[test_case]
    fn input_pin_accepts_line_in_and_mic_only() {
        assert!(is_input_pin(cfg(0, 0x8))); // line in
        assert!(is_input_pin(cfg(0, 0xa))); // mic
        assert!(!is_input_pin(cfg(0, 0x1))); // speaker is not an input
        assert!(!is_input_pin(cfg(0, 0x9))); // 0x9 is not a type we capture from
    }
}
