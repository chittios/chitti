//! **USB Video Class (UVC)** — pure descriptor parse + identify-only presence.
//!
//! ## Stages (hardware plan PR7)
//!
//! 1. **Identify + parse** (this file): note Video Control / Streaming interfaces
//!    during xHCI config walk; parse VS format/frame descriptors for MJPEG and
//!    uncompressed (YUY2) geometries. Unit-tested off the bus.
//! 2. **Payload path** — bulk stills first where the camera offers them; full
//!    isochronous needs xHCI isoc rings (not yet).
//! 3. **`/camera` capture** — frame → image path / raw buffer (later).
//!
//! ## Spec traps encoded here
//!
//! - Interface class `0x0E` (Video); subclass `1` = VC, `2` = VS.
//! - Format descriptors are **class-specific** (`CS_INTERFACE` = 0x24) with
//!   subtypes Format Uncompressed (0x04), Format MJPEG (0x06), Frame Uncompressed
//!   (0x05), Frame MJPEG (0x07).
//! - Lengths are claims: a lying `bLength` is refused, never clamped into the
//!   next descriptor.

use core::sync::atomic::{AtomicBool, AtomicU16, AtomicU8, Ordering};

/// USB Video class.
pub const USB_CLASS_VIDEO: u8 = 0x0e;
pub const SC_VIDEOCONTROL: u8 = 0x01;
pub const SC_VIDEOSTREAMING: u8 = 0x02;

/// Class-specific interface descriptor type.
pub const CS_INTERFACE: u8 = 0x24;

pub const VS_FORMAT_UNCOMPRESSED: u8 = 0x04;
pub const VS_FRAME_UNCOMPRESSED: u8 = 0x05;
pub const VS_FORMAT_MJPEG: u8 = 0x06;
pub const VS_FRAME_MJPEG: u8 = 0x07;

/// GUID for YUY2 (YUYV) in UVC uncompressed format descriptors.
pub const GUID_YUY2: [u8; 16] = [
    0x59, 0x55, 0x59, 0x32, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71,
];

pub fn is_video_control(class: u8, subclass: u8, _proto: u8) -> bool {
    class == USB_CLASS_VIDEO && subclass == SC_VIDEOCONTROL
}

pub fn is_video_streaming(class: u8, subclass: u8, _proto: u8) -> bool {
    class == USB_CLASS_VIDEO && subclass == SC_VIDEOSTREAMING
}

/// Pixel format we understand enough to name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PixelFormat {
    Mjpeg,
    Yuy2,
    UncompressedOther,
}

impl PixelFormat {
    pub fn name(self) -> &'static str {
        match self {
            PixelFormat::Mjpeg => "MJPEG",
            PixelFormat::Yuy2 => "YUY2",
            PixelFormat::UncompressedOther => "uncompressed",
        }
    }
}

/// One frame size under a format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameSize {
    pub width: u16,
    pub height: u16,
}

/// Parsed VS format + its frames.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FormatInfo {
    pub format: PixelFormat,
    pub frames: alloc::vec::Vec<FrameSize>,
}

// ── live inventory ───────────────────────────────────────────────────────

static SEEN: AtomicBool = AtomicBool::new(false);
static ROOT_PORT: AtomicU8 = AtomicU8::new(0);
static SLOT: AtomicU8 = AtomicU8::new(0);
static N_VC: AtomicU8 = AtomicU8::new(0);
static N_VS: AtomicU8 = AtomicU8::new(0);
static BEST_W: AtomicU16 = AtomicU16::new(0);
static BEST_H: AtomicU16 = AtomicU16::new(0);
static BEST_FMT: AtomicU8 = AtomicU8::new(0); // 0 none, 1 mjpeg, 2 yuy2, 3 other

/// Note a VC or VS interface.
pub fn note_usb_iface(root_port: u8, slot: u8, class: u8, sub: u8, _proto: u8) {
    SEEN.store(true, Ordering::Release);
    ROOT_PORT.store(root_port, Ordering::Relaxed);
    SLOT.store(slot, Ordering::Relaxed);
    if is_video_control(class, sub, 0) {
        N_VC.fetch_add(1, Ordering::Relaxed);
    }
    if is_video_streaming(class, sub, 0) {
        N_VS.fetch_add(1, Ordering::Relaxed);
    }
    crate::ktrace::log_fmt(format_args!(
        "uvc: Video interface noted (port {root_port} slot {slot} sub={sub}) — identify only"
    ));
}

pub fn seen() -> bool {
    SEEN.load(Ordering::Acquire)
}

pub fn clear_if_port(root_port: u8) {
    if SEEN.load(Ordering::Acquire) && ROOT_PORT.load(Ordering::Relaxed) == root_port {
        SEEN.store(false, Ordering::Release);
        N_VC.store(0, Ordering::Relaxed);
        N_VS.store(0, Ordering::Relaxed);
        BEST_W.store(0, Ordering::Relaxed);
        BEST_H.store(0, Ordering::Relaxed);
        BEST_FMT.store(0, Ordering::Relaxed);
        crate::ktrace::log("uvc", "camera gone with root port");
    }
}

/// Parse a full configuration descriptor for UVC formats; updates BEST_*.
pub fn try_parse_config(config: &[u8]) {
    let formats = parse_vs_formats(config);
    let mut best: Option<(u32, PixelFormat, u16, u16)> = None;
    for f in &formats {
        for fr in &f.frames {
            let area = fr.width as u32 * fr.height as u32;
            let rank = match f.format {
                PixelFormat::Mjpeg => 2,
                PixelFormat::Yuy2 => 1,
                PixelFormat::UncompressedOther => 0,
            };
            let score = (area << 2) | rank;
            match best {
                Some((s, _, _, _)) if s >= score => {}
                _ => best = Some((score, f.format, fr.width, fr.height)),
            }
        }
    }
    if let Some((_, fmt, w, h)) = best {
        BEST_W.store(w, Ordering::Relaxed);
        BEST_H.store(h, Ordering::Relaxed);
        BEST_FMT.store(
            match fmt {
                PixelFormat::Mjpeg => 1,
                PixelFormat::Yuy2 => 2,
                PixelFormat::UncompressedOther => 3,
            },
            Ordering::Relaxed,
        );
        crate::ktrace::log_fmt(format_args!(
            "uvc: best format {} {}x{} ({} format(s) parsed)",
            fmt.name(),
            w,
            h,
            formats.len()
        ));
    }
}

pub fn status_lines() -> alloc::vec::Vec<alloc::string::String> {
    use alloc::format;
    use alloc::string::String;
    let mut v = alloc::vec::Vec::new();
    if !seen() {
        v.push(String::from("usb: no UVC interface noted at enumeration"));
        v.push(String::from(
            "next: plug a webcam; capture needs isoc/bulk payload path (not yet)",
        ));
        return v;
    }
    v.push(format!(
        "usb: present (root port {}, slot {}, VC×{} VS×{})",
        ROOT_PORT.load(Ordering::Relaxed),
        SLOT.load(Ordering::Relaxed),
        N_VC.load(Ordering::Relaxed),
        N_VS.load(Ordering::Relaxed),
    ));
    let w = BEST_W.load(Ordering::Relaxed);
    let h = BEST_H.load(Ordering::Relaxed);
    let fmt = match BEST_FMT.load(Ordering::Relaxed) {
        1 => "MJPEG",
        2 => "YUY2",
        3 => "uncompressed",
        _ => "unknown",
    };
    if w > 0 && h > 0 {
        v.push(format!("format: {fmt} {w}x{h} (largest/preferred from descriptors)"));
    } else {
        v.push(String::from(
            "format: descriptors not parsed or no frame sizes",
        ));
    }
    v.push(String::from(
        "capture: not yet — needs xHCI isoc (or bulk still) + MJPEG/YUY2 path",
    ));
    v
}

// ── pure parse ───────────────────────────────────────────────────────────

/// Walk a configuration (or VS interface) blob for Format + Frame descriptors.
pub fn parse_vs_formats(buf: &[u8]) -> alloc::vec::Vec<FormatInfo> {
    let mut out = alloc::vec::Vec::new();
    let mut i = 0usize;
    let mut current: Option<FormatInfo> = None;

    let flush = |current: &mut Option<FormatInfo>, out: &mut alloc::vec::Vec<FormatInfo>| {
        if let Some(f) = current.take() {
            if !f.frames.is_empty() {
                out.push(f);
            }
        }
    };

    while i + 3 <= buf.len() {
        let blen = buf[i] as usize;
        if blen < 3 || i + blen > buf.len() {
            break;
        }
        let btype = buf[i + 1];
        let subtype = buf[i + 2];
        if btype == CS_INTERFACE {
            match subtype {
                VS_FORMAT_MJPEG => {
                    flush(&mut current, &mut out);
                    current = Some(FormatInfo {
                        format: PixelFormat::Mjpeg,
                        frames: alloc::vec::Vec::new(),
                    });
                }
                VS_FORMAT_UNCOMPRESSED => {
                    flush(&mut current, &mut out);
                    let fmt = if blen >= 3 + 16 + 1 {
                        // bFormatIndex @3, guid @4..20
                        let guid = &buf[i + 4..i + 20];
                        if guid == GUID_YUY2 {
                            PixelFormat::Yuy2
                        } else {
                            PixelFormat::UncompressedOther
                        }
                    } else {
                        PixelFormat::UncompressedOther
                    };
                    current = Some(FormatInfo {
                        format: fmt,
                        frames: alloc::vec::Vec::new(),
                    });
                }
                VS_FRAME_MJPEG | VS_FRAME_UNCOMPRESSED => {
                    // bFrameIndex @3, bmCapabilities @4, wWidth @5, wHeight @7
                    if blen >= 10 {
                        let w = u16::from_le_bytes([buf[i + 5], buf[i + 6]]);
                        let h = u16::from_le_bytes([buf[i + 7], buf[i + 8]]);
                        if w > 0 && h > 0 {
                            if let Some(ref mut f) = current {
                                f.frames.push(FrameSize {
                                    width: w,
                                    height: h,
                                });
                            }
                        }
                    }
                }
                _ => {}
            }
        } else if btype == 0x04 {
            // New interface — finish the previous format group.
            flush(&mut current, &mut out);
        }
        i += blen;
    }
    flush(&mut current, &mut out);
    out
}

/// GUID equality helper for tests.
pub fn guid_is_yuy2(g: &[u8]) -> bool {
    g.len() >= 16 && g[..16] == GUID_YUY2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn video_class_helpers() {
        assert!(is_video_control(0x0e, 0x01, 0));
        assert!(is_video_streaming(0x0e, 0x02, 0));
        assert!(!is_video_control(0x0e, 0x02, 0));
        assert!(!is_usb_bt_like(0x0e, 0x01));
    }

    fn is_usb_bt_like(c: u8, s: u8) -> bool {
        c == 0xe0 && s == 0x01
    }

    /// Synthetic VS blob: MJPEG format + two frames (640x480, 1280x720).
    #[test_case]
    fn parse_mjpeg_format_and_frames() {
        let mut d = alloc::vec::Vec::new();
        // Format MJPEG: bLength=11, type=0x24, subtype=0x06, bFormatIndex=1, …
        d.extend_from_slice(&[11, 0x24, 0x06, 1, 1, 0, 0, 0, 0, 0, 0]);
        // Frame MJPEG 640x480
        d.extend_from_slice(&[
            30, 0x24, 0x07, 1, 0, // hdr
            0x80, 0x02, // w=640
            0xe0, 0x01, // h=480
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]);
        // Frame 1280x720
        d.extend_from_slice(&[
            30, 0x24, 0x07, 2, 0, 0x00, 0x05, // 1280
            0xd0, 0x02, // 720
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]);
        let f = parse_vs_formats(&d);
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].format, PixelFormat::Mjpeg);
        assert_eq!(
            f[0].frames,
            alloc::vec![
                FrameSize {
                    width: 640,
                    height: 480
                },
                FrameSize {
                    width: 1280,
                    height: 720
                }
            ]
        );
    }

    #[test_case]
    fn parse_yuy2_guid() {
        let mut d = alloc::vec::Vec::new();
        // Format Uncompressed with YUY2 GUID (bLength = 27 typical min)
        let mut fmt = alloc::vec![0u8; 27];
        fmt[0] = 27;
        fmt[1] = 0x24;
        fmt[2] = VS_FORMAT_UNCOMPRESSED;
        fmt[3] = 1; // format index
        fmt[4..20].copy_from_slice(&GUID_YUY2);
        d.extend_from_slice(&fmt);
        // One frame 320x240
        d.extend_from_slice(&[
            30, 0x24, 0x05, 1, 0, 0x40, 0x01, // 320
            0xf0, 0x00, // 240
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]);
        let f = parse_vs_formats(&d);
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].format, PixelFormat::Yuy2);
        assert_eq!(f[0].frames[0].width, 320);
        assert_eq!(f[0].frames[0].height, 240);
        assert!(guid_is_yuy2(&GUID_YUY2));
    }

    #[test_case]
    fn malformed_length_stops_walk() {
        // bLength 0 would spin forever without the max(1)/break guards.
        assert!(parse_vs_formats(&[0, 0x24, 0x06]).is_empty());
        assert!(parse_vs_formats(&[5, 0x24, 0x06]).is_empty()); // claims 5, only 3 bytes
        assert!(parse_vs_formats(&[]).is_empty());
    }
}
