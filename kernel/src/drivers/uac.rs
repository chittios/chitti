//! **USB Audio Class 1.0** — descriptor parse, stream selection, control shapes.
//!
//! A USB headset or DAC is the one audio device this OS could not use at all:
//! HDA covers built-in codecs, but anything on a USB port was invisible. This
//! This module is the pure layer — descriptors in, a [`StreamPlan`] out —
//! mirroring [`crate::drivers::uvc`], which does the same job for video over the
//! same isochronous transport. The stream itself is configured and fed by
//! `xhci::configure_uac` / `xhci::uac_pump`, and surfaced as a `SndDevice` by
//! [`crate::sound::usb`].
//!
//! ## Alternate settings are the whole protocol
//!
//! An AudioStreaming interface always exposes **alt 0 with no endpoints at
//! all**. That is not a degenerate case, it is the spec's way of saying "I am
//! claiming no bus bandwidth right now": a device is parked there and moved to
//! alt 1 or higher when the host wants to stream.
//!
//! So a driver that configures the interface and starts writing gets a
//! perfectly successful `SET_INTERFACE`, an endpoint that does not exist, and
//! silence. **Selecting a non-zero alt is the single most important thing in
//! this file**, and [`find_output_stream`] refuses to return alt 0.
//!
//! ## Traps that produce plausible wrong answers
//!
//! * **An audio endpoint descriptor is 9 bytes, not 7.** UAC1 appends
//!   `bRefresh` and `bSynchAddress`. A walk that steps by a hardcoded 7 lands
//!   two bytes into the next descriptor and reads its `bLength` as an endpoint
//!   address — so the walk keeps going and finds nonsense. Everything here
//!   steps by the descriptor's own `bLength`.
//! * **Sample frequencies are 24-bit little-endian**, three bytes each. Read as
//!   a `u32` they consume a byte of the next entry, so a device offering
//!   44100/48000 reports the first rate correctly and the second as garbage.
//! * **The sampling-frequency control is addressed to the ENDPOINT**, not the
//!   interface. Sent to the interface it is accepted by some firmware and
//!   ignored by the rest, leaving the device converting at whatever rate it
//!   defaulted to — audio that plays at the wrong speed rather than not at all.

use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, AtomicU16, AtomicU32, AtomicU8, Ordering};

/// USB Audio class.
pub const USB_CLASS_AUDIO: u8 = 0x01;
/// AudioControl interface — the function's control surface.
pub const SC_AUDIOCONTROL: u8 = 0x01;
/// AudioStreaming interface — carries the isochronous endpoint.
pub const SC_AUDIOSTREAMING: u8 = 0x02;

/// Class-specific descriptor types.
pub const CS_INTERFACE: u8 = 0x24;
pub const CS_ENDPOINT: u8 = 0x25;

/// AudioStreaming class-specific interface descriptor subtypes.
pub const AS_GENERAL: u8 = 0x01;
pub const AS_FORMAT_TYPE: u8 = 0x02;

/// `wFormatTag` for uncompressed PCM. The only format we can feed.
pub const FORMAT_PCM: u16 = 0x0001;
/// `bFormatType` I — the fixed-size-sample formats, which PCM is.
pub const FORMAT_TYPE_I: u8 = 0x01;

/// Class request codes.
pub const SET_CUR: u8 = 0x01;
pub const GET_CUR: u8 = 0x81;
/// Endpoint control selector for the sample rate.
pub const SAMPLING_FREQ_CONTROL: u8 = 0x01;

/// `bmRequestType` host → device, class, recipient **endpoint**.
pub const BM_OUT_CLASS_ENDPOINT: u8 = 0x22;
/// `bmRequestType` host → device, standard, recipient interface — `SET_INTERFACE`.
pub const BM_OUT_STD_IFACE: u8 = 0x01;
pub const REQ_SET_INTERFACE: u8 = 0x0b;

/// True when an interface is a UAC AudioControl interface.
pub fn is_audio_control(class: u8, subclass: u8, _proto: u8) -> bool {
    class == USB_CLASS_AUDIO && subclass == SC_AUDIOCONTROL
}

/// True when an interface is a UAC AudioStreaming interface.
pub fn is_audio_streaming(class: u8, subclass: u8, _proto: u8) -> bool {
    class == USB_CLASS_AUDIO && subclass == SC_AUDIOSTREAMING
}

/// One playable AudioStreaming alternate setting.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StreamPlan {
    /// Interface number to `SET_INTERFACE`.
    pub iface: u8,
    /// **Non-zero** alternate setting. Alt 0 claims no bandwidth and exposes no
    /// endpoint; see the module docs.
    pub alt: u8,
    /// Isochronous OUT endpoint address (no direction bit — OUT is bit 7 clear).
    pub ep: u8,
    /// `wMaxPacketSize` — the most the device accepts per service interval.
    pub mps: u16,
    /// `bInterval`, in the encoding the endpoint descriptor uses.
    pub interval: u8,
    pub channels: u8,
    /// Bytes per sample per channel (`bSubframeSize`).
    pub subframe: u8,
    /// Bits actually used within each subframe (`bBitResolution`).
    pub bits: u8,
    /// Sample rates the alt setting offers. A continuous range is represented
    /// by its two endpoints.
    pub rates: Vec<u32>,
    /// True when the rates are a continuous range rather than a discrete list.
    pub continuous: bool,
}

impl StreamPlan {
    /// Whether this alt can carry `hz`, either as a listed rate or inside a
    /// continuous range.
    pub fn supports_rate(&self, hz: u32) -> bool {
        if self.continuous {
            match (self.rates.first(), self.rates.last()) {
                (Some(&lo), Some(&hi)) => hz >= lo && hz <= hi,
                _ => false,
            }
        } else {
            self.rates.contains(&hz)
        }
    }

    /// The rate to actually run: `want` when the device offers it, otherwise the
    /// closest one it does.
    ///
    /// Falling back to "the first rate offered" is the tempting alternative and
    /// is wrong in an audible way — a device listing 8000 first would play every
    /// track at a sixth speed, which reads as a broken decoder.
    pub fn pick_rate(&self, want: u32) -> Option<u32> {
        if self.rates.is_empty() {
            return None;
        }
        if self.supports_rate(want) {
            return Some(want);
        }
        if self.continuous {
            let lo = *self.rates.first()?;
            let hi = *self.rates.last()?;
            return Some(want.clamp(lo, hi));
        }
        self.rates.iter().copied().min_by_key(|&r| r.abs_diff(want))
    }

    /// Bytes one service interval carries at `hz` — what `wMaxPacketSize` has to
    /// cover.
    ///
    /// At the 1 ms frame of a full-speed device this is a millisecond of audio.
    /// A packet larger than `mps` is not truncated politely: the device drops
    /// the whole transfer, so the stream gaps rather than distorts.
    pub fn bytes_per_interval(&self, hz: u32) -> usize {
        let frames = hz.div_ceil(1000) as usize;
        frames * self.channels.max(1) as usize * self.subframe.max(1) as usize
    }

    /// Whether the endpoint can carry `hz` at this alt's format.
    pub fn fits(&self, hz: u32) -> bool {
        self.bytes_per_interval(hz) <= self.mps as usize
    }
}

/// Decode a 24-bit little-endian sample frequency.
///
/// Three bytes, not four. Read as a `u32` it swallows a byte of whatever comes
/// next, so a device offering 44100 then 48000 reports the first correctly and
/// the second as a large wrong number that no range check rejects.
pub fn freq24(b: &[u8]) -> Option<u32> {
    let s = b.get(..3)?;
    Some(u32::from(s[0]) | (u32::from(s[1]) << 8) | (u32::from(s[2]) << 16))
}

/// Walk a configuration descriptor and pick the best AudioStreaming alternate
/// setting for **output** at or near `want_hz`.
///
/// "Best" is: PCM Format Type I, an isochronous OUT endpoint, a non-zero alt,
/// and among those the one whose channel count is closest to `want_channels`
/// without exceeding it — preferring stereo for music while still finding a
/// mono-only headset. Ties break toward the alt that can actually carry the
/// requested rate.
pub fn find_output_stream(desc: &[u8], want_hz: u32, want_channels: u8) -> Option<StreamPlan> {
    let mut best: Option<StreamPlan> = None;
    let mut cur: Option<StreamPlan> = None;
    let mut in_as = false;

    let mut i = 0usize;
    while i + 2 <= desc.len() {
        let len = desc[i] as usize;
        // A zero or overlong length would spin or read past the buffer, and
        // these bytes came off a bus we do not control.
        if len < 2 || i + len > desc.len() {
            break;
        }
        let d = &desc[i..i + len];
        match d[1] {
            // Standard interface descriptor.
            0x04 if len >= 9 => {
                // A new interface ends the previous candidate.
                if let Some(p) = cur.take() {
                    consider(&mut best, p, want_hz, want_channels);
                }
                let (number, alt, class, subclass) = (d[2], d[3], d[5], d[6]);
                in_as = is_audio_streaming(class, subclass, d[7]);
                // Alt 0 exposes no endpoint by construction: it is the parked
                // setting. Tracking it would produce a plan whose `ep` stays 0.
                if in_as && alt != 0 {
                    cur = Some(StreamPlan {
                        iface: number,
                        alt,
                        ep: 0,
                        mps: 0,
                        interval: 1,
                        channels: 0,
                        subframe: 0,
                        bits: 0,
                        rates: Vec::new(),
                        continuous: false,
                    });
                } else {
                    cur = None;
                }
            }
            // Class-specific AS interface descriptor.
            CS_INTERFACE if in_as && len >= 3 => {
                if let Some(p) = cur.as_mut() {
                    match d[2] {
                        AS_GENERAL if len >= 7 => {
                            let tag = u16::from_le_bytes([d[5], d[6]]);
                            if tag != FORMAT_PCM {
                                // Not PCM (MPEG, AC-3, …): we cannot feed it, so
                                // drop the candidate rather than plan a stream
                                // whose bytes the device will reinterpret.
                                cur = None;
                            }
                        }
                        AS_FORMAT_TYPE if len >= 8 => {
                            if d[3] != FORMAT_TYPE_I {
                                cur = None;
                            } else {
                                p.channels = d[4];
                                p.subframe = d[5];
                                p.bits = d[6];
                                let n = d[7];
                                if n == 0 {
                                    // Continuous: lower then upper bound.
                                    p.continuous = true;
                                    if let (Some(lo), Some(hi)) =
                                        (freq24(&d[8..]), freq24(&d[11..]))
                                    {
                                        p.rates.push(lo);
                                        p.rates.push(hi);
                                    }
                                } else {
                                    // A discrete table: `n` entries of **three**
                                    // bytes each.
                                    for k in 0..n as usize {
                                        let at = 8 + k * 3;
                                        match freq24(d.get(at..).unwrap_or(&[])) {
                                            Some(f) => p.rates.push(f),
                                            None => break,
                                        }
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
            // Standard endpoint descriptor. **9 bytes for audio**, but the walk
            // uses `bLength`, so both sizes work.
            0x05 if in_as && len >= 7 => {
                if let Some(p) = cur.as_mut() {
                    let addr = d[2];
                    let attrs = d[3];
                    let mps = u16::from_le_bytes([d[4], d[5]]);
                    let is_isoc = attrs & 0x03 == 0x01;
                    let is_out = addr & 0x80 == 0;
                    if is_isoc && is_out {
                        p.ep = addr;
                        p.mps = mps;
                        p.interval = d[6];
                    }
                }
            }
            _ => {}
        }
        i += len;
    }
    if let Some(p) = cur.take() {
        consider(&mut best, p, want_hz, want_channels);
    }
    best
}

/// Keep `p` if it beats the incumbent.
fn consider(best: &mut Option<StreamPlan>, p: StreamPlan, want_hz: u32, want_channels: u8) {
    // A candidate with no endpoint never had one — alt 0, or an IN-only alt.
    if p.ep == 0 || p.mps == 0 || p.rates.is_empty() || p.channels == 0 {
        return;
    }
    // Ranked, most significant first:
    //   1. can it carry the requested rate at all;
    //   2. does it fit within the channel count asked for — an alt with *more*
    //      channels than wanted is usable but wasteful (it needs the source
    //      duplicated up and burns bus bandwidth), so it loses to any alt that
    //      fits;
    //   3. among those that fit, the most channels; among those that do not,
    //      the fewest excess.
    //
    // Packet size is deliberately **not** a tie-break. It was, and it meant a
    // request for mono chose a stereo alt purely because its endpoint was
    // bigger — the device then ran a stereo stream fed mono frames, which plays
    // at half speed in one channel.
    let score = |c: &StreamPlan| -> (u8, u8, u8) {
        // 2 = carries exactly the rate asked for; 1 = carries a nearby one it
        // would have to substitute; 0 = cannot carry its own best rate at all.
        // Collapsing the first two — asking only "does some rate fit" — makes an
        // alt that would resample to 48 kHz tie with one that natively does 96,
        // and the tie then falls to whatever came first in the descriptor.
        let rate = match c.pick_rate(want_hz) {
            Some(r) if c.fits(r) && r == want_hz => 2,
            Some(r) if c.fits(r) => 1,
            _ => 0,
        };
        if c.channels <= want_channels {
            (rate, 2, c.channels)
        } else {
            (rate, 1, u8::MAX - c.channels)
        }
    };
    match best {
        Some(b) if score(b) >= score(&p) => {}
        _ => *best = Some(p),
    }
}

/// `(bmRequestType, bRequest, wValue, wIndex, wLength)` for `SET_INTERFACE`,
/// which is what actually claims the isochronous bandwidth.
pub fn set_interface_setup(iface: u8, alt: u8) -> (u8, u8, u16, u16, u16) {
    (BM_OUT_STD_IFACE, REQ_SET_INTERFACE, alt as u16, iface as u16, 0)
}

/// `(bmRequestType, bRequest, wValue, wIndex, wLength)` for setting the sample
/// rate, plus the three data bytes.
///
/// Addressed to the **endpoint** (`wIndex` is the endpoint address, and the
/// recipient bits say endpoint). Sent to the interface it is accepted by some
/// firmware and ignored by the rest, leaving the device converting at its
/// default rate — audio at the wrong speed rather than no audio.
pub fn set_sample_rate_setup(ep: u8, hz: u32) -> ((u8, u8, u16, u16, u16), [u8; 3]) {
    let setup = (
        BM_OUT_CLASS_ENDPOINT,
        SET_CUR,
        (SAMPLING_FREQ_CONTROL as u16) << 8,
        ep as u16,
        3,
    );
    let data = [(hz & 0xff) as u8, ((hz >> 8) & 0xff) as u8, ((hz >> 16) & 0xff) as u8];
    (setup, data)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn iface(number: u8, alt: u8, class: u8, sub: u8, n_ep: u8) -> [u8; 9] {
        [9, 0x04, number, alt, n_ep, class, sub, 0, 0]
    }
    /// A UAC1 endpoint descriptor is **9** bytes: the standard 7 plus
    /// `bRefresh` and `bSynchAddress`.
    fn audio_ep(addr: u8, attrs: u8, mps: u16, interval: u8) -> [u8; 9] {
        let m = mps.to_le_bytes();
        [9, 0x05, addr, attrs, m[0], m[1], interval, 0, 0]
    }
    fn as_general(tag: u16) -> [u8; 7] {
        let t = tag.to_le_bytes();
        [7, CS_INTERFACE, AS_GENERAL, 1, 0, t[0], t[1]]
    }
    /// Format Type I with a discrete rate table.
    fn format_type_i(channels: u8, subframe: u8, bits: u8, rates: &[u32]) -> Vec<u8> {
        let mut d = alloc::vec![
            (8 + rates.len() * 3) as u8,
            CS_INTERFACE,
            AS_FORMAT_TYPE,
            FORMAT_TYPE_I,
            channels,
            subframe,
            bits,
            rates.len() as u8,
        ];
        for r in rates {
            d.push((r & 0xff) as u8);
            d.push(((r >> 8) & 0xff) as u8);
            d.push(((r >> 16) & 0xff) as u8);
        }
        d
    }

    /// A minimal but realistic headset: AC interface, AS alt 0 (parked, no
    /// endpoints) and AS alt 1 (stereo 16-bit, 44.1/48 kHz).
    fn headset_desc() -> Vec<u8> {
        let mut d = Vec::new();
        d.extend_from_slice(&iface(0, 0, USB_CLASS_AUDIO, SC_AUDIOCONTROL, 0));
        d.extend_from_slice(&iface(1, 0, USB_CLASS_AUDIO, SC_AUDIOSTREAMING, 0));
        d.extend_from_slice(&iface(1, 1, USB_CLASS_AUDIO, SC_AUDIOSTREAMING, 1));
        d.extend_from_slice(&as_general(FORMAT_PCM));
        d.extend_from_slice(&format_type_i(2, 2, 16, &[44100, 48000]));
        d.extend_from_slice(&audio_ep(0x01, 0x01, 200, 1));
        d
    }

    /// **Alt 0 exposes no endpoint by design** — it is the spec's "claiming no
    /// bandwidth" state. A driver that selects it gets a successful
    /// `SET_INTERFACE`, an endpoint that does not exist, and silence.
    #[test_case]
    fn the_selected_alternate_setting_is_never_zero() {
        let p = find_output_stream(&headset_desc(), 48000, 2).expect("a stream must be found");
        assert_eq!(p.iface, 1);
        assert_ne!(p.alt, 0, "alt 0 claims no bandwidth and has no endpoint");
        assert_eq!(p.alt, 1);
        assert_eq!(p.ep, 0x01);
        assert_eq!(p.mps, 200);
    }

    /// **Sample frequencies are 24-bit little-endian**, three bytes each. Read
    /// four at a time, the second rate swallows a byte of the next field and
    /// comes out as a large wrong number nothing range-checks.
    #[test_case]
    fn sample_rates_are_three_byte_little_endian() {
        assert_eq!(freq24(&[0x44, 0xac, 0x00]), Some(44100));
        assert_eq!(freq24(&[0x80, 0xbb, 0x00]), Some(48000));
        assert_eq!(freq24(&[0x00, 0x77, 0x01]), Some(96000));
        assert_eq!(freq24(&[1, 2]), None, "a short tail is refused, not padded");

        let p = find_output_stream(&headset_desc(), 48000, 2).unwrap();
        assert_eq!(p.rates, alloc::vec![44100, 48000], "both rates, both correct");
        assert_eq!(p.channels, 2);
        assert_eq!(p.subframe, 2);
        assert_eq!(p.bits, 16);
    }

    /// A continuous range is two bounds, not a list — and membership is a range
    /// test rather than an equality test.
    #[test_case]
    fn a_continuous_rate_range_is_a_range() {
        let mut d = Vec::new();
        d.extend_from_slice(&iface(1, 1, USB_CLASS_AUDIO, SC_AUDIOSTREAMING, 1));
        d.extend_from_slice(&as_general(FORMAT_PCM));
        // bSamFreqType = 0 → continuous, 8000..96000.
        let mut ft = alloc::vec![14u8, CS_INTERFACE, AS_FORMAT_TYPE, FORMAT_TYPE_I, 2, 2, 16, 0];
        for r in [8000u32, 96000] {
            ft.push((r & 0xff) as u8);
            ft.push(((r >> 8) & 0xff) as u8);
            ft.push(((r >> 16) & 0xff) as u8);
        }
        d.extend_from_slice(&ft);
        d.extend_from_slice(&audio_ep(0x02, 0x01, 400, 1));

        let p = find_output_stream(&d, 44100, 2).unwrap();
        assert!(p.continuous);
        assert!(p.supports_rate(44100), "inside the range");
        assert!(p.supports_rate(8000) && p.supports_rate(96000), "the bounds themselves");
        assert!(!p.supports_rate(192000), "outside");
        assert_eq!(p.pick_rate(44100), Some(44100));
        // Out of range clamps into it rather than failing.
        assert_eq!(p.pick_rate(192000), Some(96000));
        assert_eq!(p.pick_rate(4000), Some(8000));
    }

    /// An unsupported rate picks the **closest**, not the first offered. A
    /// device listing 8000 first would otherwise play every track at a sixth
    /// speed, which sounds like a broken decoder rather than a rate mismatch.
    #[test_case]
    fn an_unsupported_rate_picks_the_closest_not_the_first() {
        let mut d = Vec::new();
        d.extend_from_slice(&iface(1, 1, USB_CLASS_AUDIO, SC_AUDIOSTREAMING, 1));
        d.extend_from_slice(&as_general(FORMAT_PCM));
        d.extend_from_slice(&format_type_i(2, 2, 16, &[8000, 44100, 48000]));
        d.extend_from_slice(&audio_ep(0x01, 0x01, 400, 1));
        let p = find_output_stream(&d, 22050, 2).unwrap();
        assert_eq!(p.pick_rate(22050), Some(8000), "8000 is nearer 22050 than 44100");
        assert_eq!(p.pick_rate(46000), Some(44100));
        assert_eq!(p.pick_rate(48000), Some(48000), "an exact match wins outright");
    }

    /// A packet bigger than `wMaxPacketSize` is dropped whole by the device, so
    /// the stream gaps rather than distorts — the size must be checked before
    /// the alt is used.
    #[test_case]
    fn the_packet_size_must_cover_one_interval() {
        let p = find_output_stream(&headset_desc(), 48000, 2).unwrap();
        // 48 kHz stereo 16-bit = 48 frames/ms * 2ch * 2B = 192 bytes ≤ 200.
        assert_eq!(p.bytes_per_interval(48000), 192);
        assert!(p.fits(48000));
        // 96 kHz would need 384 and does not fit this endpoint.
        assert_eq!(p.bytes_per_interval(96000), 384);
        assert!(!p.fits(96000));
        // A rate that is not a whole number of frames per ms rounds **up** —
        // rounding down under-provisions one millisecond in ten and clips.
        assert_eq!(p.bytes_per_interval(44100), 45 * 2 * 2);
    }

    /// Non-PCM formats are refused rather than planned: the bytes we would send
    /// are PCM, and a device told to expect AC-3 reinterprets them as noise.
    #[test_case]
    fn a_non_pcm_format_is_not_claimed() {
        let mut d = Vec::new();
        d.extend_from_slice(&iface(1, 1, USB_CLASS_AUDIO, SC_AUDIOSTREAMING, 1));
        d.extend_from_slice(&as_general(0x1001)); // MPEG
        d.extend_from_slice(&format_type_i(2, 2, 16, &[48000]));
        d.extend_from_slice(&audio_ep(0x01, 0x01, 400, 1));
        assert_eq!(find_output_stream(&d, 48000, 2), None);
    }

    /// An IN-only alt (a microphone) must not be taken for an output.
    #[test_case]
    fn a_capture_only_interface_is_not_an_output() {
        let mut d = Vec::new();
        d.extend_from_slice(&iface(2, 1, USB_CLASS_AUDIO, SC_AUDIOSTREAMING, 1));
        d.extend_from_slice(&as_general(FORMAT_PCM));
        d.extend_from_slice(&format_type_i(1, 2, 16, &[48000]));
        d.extend_from_slice(&audio_ep(0x81, 0x01, 200, 1)); // IN
        assert_eq!(find_output_stream(&d, 48000, 2), None);

        // ...and a bulk OUT endpoint is not isochronous.
        let mut d2 = Vec::new();
        d2.extend_from_slice(&iface(2, 1, USB_CLASS_AUDIO, SC_AUDIOSTREAMING, 1));
        d2.extend_from_slice(&as_general(FORMAT_PCM));
        d2.extend_from_slice(&format_type_i(2, 2, 16, &[48000]));
        d2.extend_from_slice(&audio_ep(0x02, 0x02, 200, 1)); // bulk
        assert_eq!(find_output_stream(&d2, 48000, 2), None);
    }

    /// Given several usable alts, the one nearest the requested channel count
    /// wins — a headset commonly offers mono and stereo alts on one interface.
    #[test_case]
    fn the_best_alt_is_chosen_among_several() {
        let mut d = Vec::new();
        d.extend_from_slice(&iface(1, 0, USB_CLASS_AUDIO, SC_AUDIOSTREAMING, 0));
        // alt 1: mono.
        d.extend_from_slice(&iface(1, 1, USB_CLASS_AUDIO, SC_AUDIOSTREAMING, 1));
        d.extend_from_slice(&as_general(FORMAT_PCM));
        d.extend_from_slice(&format_type_i(1, 2, 16, &[48000]));
        d.extend_from_slice(&audio_ep(0x01, 0x01, 200, 1));
        // alt 2: stereo.
        d.extend_from_slice(&iface(1, 2, USB_CLASS_AUDIO, SC_AUDIOSTREAMING, 1));
        d.extend_from_slice(&as_general(FORMAT_PCM));
        d.extend_from_slice(&format_type_i(2, 2, 16, &[48000]));
        d.extend_from_slice(&audio_ep(0x01, 0x01, 400, 1));

        assert_eq!(find_output_stream(&d, 48000, 2).unwrap().alt, 2, "stereo wanted");
        // Asking for mono takes the mono alt rather than half-using the stereo
        // one. This is the case a packet-size tie-break got wrong: the stereo
        // alt has the larger endpoint, so it won on size while being the wrong
        // format — a stereo stream fed mono frames plays at half speed in one
        // channel.
        assert_eq!(find_output_stream(&d, 48000, 1).unwrap().alt, 1);
        // An alt with more channels than asked for is still usable when it is
        // the only one — better a working stream than none.
        let mut only_stereo = Vec::new();
        only_stereo.extend_from_slice(&iface(1, 1, USB_CLASS_AUDIO, SC_AUDIOSTREAMING, 1));
        only_stereo.extend_from_slice(&as_general(FORMAT_PCM));
        only_stereo.extend_from_slice(&format_type_i(2, 2, 16, &[48000]));
        only_stereo.extend_from_slice(&audio_ep(0x01, 0x01, 400, 1));
        assert_eq!(find_output_stream(&only_stereo, 48000, 1).unwrap().channels, 2);
    }

    /// **The 9-byte audio endpoint descriptor must not desynchronise the walk.**
    /// Stepping by a hardcoded 7 lands two bytes in and reads the next
    /// descriptor's `bLength` as an endpoint address.
    #[test_case]
    fn a_nine_byte_endpoint_descriptor_does_not_desync_the_walk() {
        let mut d = headset_desc();
        // Append a second interface after the 9-byte endpoint. If the walk
        // mis-stepped, this would not parse and the assertion below would see
        // the wrong interface number.
        d.extend_from_slice(&iface(7, 1, USB_CLASS_AUDIO, SC_AUDIOSTREAMING, 1));
        d.extend_from_slice(&as_general(FORMAT_PCM));
        d.extend_from_slice(&format_type_i(2, 2, 16, &[96000]));
        d.extend_from_slice(&audio_ep(0x03, 0x01, 800, 1));

        // Asking for 96 kHz must find the *second* interface, which is only
        // reachable if the walk stepped over the first 9-byte endpoint exactly.
        let p = find_output_stream(&d, 96000, 2).unwrap();
        assert_eq!(p.iface, 7);
        assert_eq!(p.ep, 0x03);
        assert_eq!(p.rates, alloc::vec![96000]);
    }

    /// A truncated or zero-length descriptor terminates the walk instead of
    /// spinning or reading past the buffer.
    #[test_case]
    fn a_malformed_descriptor_terminates_the_walk() {
        assert_eq!(find_output_stream(&[], 48000, 2), None);
        assert_eq!(find_output_stream(&[0, 0x04, 1, 1, 1, 1, 2, 0, 0], 48000, 2), None);
        assert_eq!(find_output_stream(&[9, 0x04, 1], 48000, 2), None);
        // A format descriptor claiming more rates than it carries stops at the
        // ones actually present rather than reading past the end.
        let mut d = Vec::new();
        d.extend_from_slice(&iface(1, 1, USB_CLASS_AUDIO, SC_AUDIOSTREAMING, 1));
        d.extend_from_slice(&as_general(FORMAT_PCM));
        d.extend_from_slice(&[11, CS_INTERFACE, AS_FORMAT_TYPE, FORMAT_TYPE_I, 2, 2, 16, 5, 0x80, 0xbb, 0x00]);
        d.extend_from_slice(&audio_ep(0x01, 0x01, 400, 1));
        let p = find_output_stream(&d, 48000, 2).unwrap();
        assert_eq!(p.rates, alloc::vec![48000], "only the rate that was present");
    }

    /// **The sample-rate control goes to the endpoint**, not the interface —
    /// sent to the interface it is silently ignored by much firmware, leaving
    /// the device converting at its default rate.
    #[test_case]
    fn the_sample_rate_request_targets_the_endpoint() {
        let ((bm, req, val, idx, len), data) = set_sample_rate_setup(0x01, 48000);
        assert_eq!(bm, 0x22, "host->device, class, recipient ENDPOINT");
        assert_eq!(bm & 0x1f, 0x02, "recipient bits say endpoint");
        assert_eq!(req, SET_CUR);
        assert_eq!(val, 0x0100, "SAMPLING_FREQ_CONTROL in the high byte");
        assert_eq!(idx, 0x01, "wIndex is the endpoint address");
        assert_eq!(len, 3);
        assert_eq!(data, [0x80, 0xbb, 0x00], "24-bit little-endian");
        assert_eq!(freq24(&data), Some(48000), "round-trips");

        // SET_INTERFACE is a *standard* request to the interface.
        let (bm, req, val, idx, len) = set_interface_setup(1, 2);
        assert_eq!(bm, 0x01, "host->device, standard, recipient interface");
        assert_eq!(req, REQ_SET_INTERFACE);
        assert_eq!((val, idx, len), (2, 1, 0));
    }
}

// ── live inventory ───────────────────────────────────────────────────────
//
// What the last enumeration saw, so `/voice` and `sound` can report a USB audio
// device honestly — present, and what it would play — rather than the shell
// having to guess from silence. The streaming path is not built yet (see the
// module status below), so this is deliberately *reporting* only: nothing here
// claims the device on the bus or takes it away from anything else.

static SEEN: AtomicBool = AtomicBool::new(false);
static ROOT_PORT: AtomicU8 = AtomicU8::new(0);
static SLOT: AtomicU8 = AtomicU8::new(0);
static PLAN_IFACE: AtomicU8 = AtomicU8::new(0);
static PLAN_ALT: AtomicU8 = AtomicU8::new(0);
static PLAN_EP: AtomicU8 = AtomicU8::new(0);
static PLAN_MPS: AtomicU16 = AtomicU16::new(0);
static PLAN_CH: AtomicU8 = AtomicU8::new(0);
static PLAN_BITS: AtomicU8 = AtomicU8::new(0);
static PLAN_RATE: AtomicU32 = AtomicU32::new(0);

/// Record a USB audio function found during enumeration, and the output stream
/// it would use. `desc` is the whole configuration descriptor.
pub fn note_usb_device(root_port: u8, slot: u8, desc: &[u8]) {
    SEEN.store(true, Ordering::Release);
    ROOT_PORT.store(root_port, Ordering::Relaxed);
    SLOT.store(slot, Ordering::Relaxed);
    // 48 kHz stereo is what the media path wants; `find_output_stream` falls
    // back to whatever the device does offer.
    let Some(p) = find_output_stream(desc, 48_000, 2) else {
        crate::ktrace::log(
            "uac",
            "USB audio device present, but no PCM output alt it can stream (capture-only, or a non-PCM format)",
        );
        return;
    };
    let rate = p.pick_rate(48_000).unwrap_or(0);
    PLAN_IFACE.store(p.iface, Ordering::Relaxed);
    PLAN_ALT.store(p.alt, Ordering::Relaxed);
    PLAN_EP.store(p.ep, Ordering::Relaxed);
    PLAN_MPS.store(p.mps, Ordering::Relaxed);
    PLAN_CH.store(p.channels, Ordering::Relaxed);
    PLAN_BITS.store(p.bits, Ordering::Relaxed);
    PLAN_RATE.store(rate, Ordering::Relaxed);
    crate::ktrace::log_fmt(format_args!(
        "uac: USB audio out on slot {slot} port {root_port} -- iface {} alt {} ep {:#04x}, {} ch {}-bit {} Hz, mps {}",
        p.iface, p.alt, p.ep, p.channels, p.bits, rate, p.mps
    ));
}

/// True when a USB audio device was seen during enumeration.
pub fn present() -> bool {
    SEEN.load(Ordering::Acquire)
}

/// Human-readable status lines for `/voice` and the sound status commands.
pub fn status_lines() -> Vec<alloc::string::String> {
    use alloc::string::ToString;
    let mut v = Vec::new();
    if !present() {
        v.push("no USB audio device enumerated".to_string());
        return v;
    }
    v.push(alloc::format!(
        "device on slot {} (root port {})",
        SLOT.load(Ordering::Relaxed),
        ROOT_PORT.load(Ordering::Relaxed)
    ));
    let ep = PLAN_EP.load(Ordering::Relaxed);
    if ep == 0 {
        v.push("no streamable PCM output alternate setting".to_string());
    } else {
        v.push(alloc::format!(
            "output: iface {} alt {} ep {ep:#04x}, {} ch {}-bit {} Hz, {} B/packet",
            PLAN_IFACE.load(Ordering::Relaxed),
            PLAN_ALT.load(Ordering::Relaxed),
            PLAN_CH.load(Ordering::Relaxed),
            PLAN_BITS.load(Ordering::Relaxed),
            PLAN_RATE.load(Ordering::Relaxed),
            PLAN_MPS.load(Ordering::Relaxed)
        ));
    }
    v.push(alloc::format!(
        "playback: {}",
        if crate::arch::uac_ready() {
            "streaming (isochronous OUT, pumped from the idle tick)"
        } else if crate::arch::uac_available() {
            "ready -- claimed only when this becomes the sound device (see sound::autodetect)"
        } else {
            "no stream could be configured -- see the ktrace for why"
        }
    ));
    v
}
