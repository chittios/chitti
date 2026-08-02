//! **USB Video Class (UVC)** — descriptor parse, stream setup helpers, frame reassembly.
//!
//! ## Stages
//!
//! 1. **Identify + parse** — VC/VS interfaces, MJPEG/YUY2 formats + frame sizes.
//! 2. **Payload path** — PROBE/COMMIT wire shapes, bulk or isoc endpoint pick,
//!    payload-header strip + frame assemble (pure); xHCI drives the rings.
//! 3. **`/camera grab`** — one still → `/downloads/camera-*.jpg` (or `.yuy2`).
//!
//! Isochronous uses xHCI Isoch TRBs; bulk webcams use the same Normal-TRB path
//! as MSC. Prefer **MJPEG** and a modest frame (≤ 640 wide) so a grab fits the
//! DMA bounce buffers.

use core::sync::atomic::{AtomicBool, AtomicU16, AtomicU32, AtomicU8, Ordering};

/// USB Video class.
pub const USB_CLASS_VIDEO: u8 = 0x0e;
pub const SC_VIDEOCONTROL: u8 = 0x01;
pub const SC_VIDEOSTREAMING: u8 = 0x02;

pub const CS_INTERFACE: u8 = 0x24;
pub const VS_FORMAT_UNCOMPRESSED: u8 = 0x04;
pub const VS_FRAME_UNCOMPRESSED: u8 = 0x05;
pub const VS_FORMAT_MJPEG: u8 = 0x06;
pub const VS_FRAME_MJPEG: u8 = 0x07;

/// Class request codes (UVC).
pub const SET_CUR: u8 = 0x01;
pub const GET_CUR: u8 = 0x81;
/// VS interface control selectors.
pub const VS_PROBE_CONTROL: u8 = 0x01;
pub const VS_COMMIT_CONTROL: u8 = 0x02;

pub const PROBE_LEN: usize = 26;

pub const GUID_YUY2: [u8; 16] = [
    0x59, 0x55, 0x59, 0x32, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71,
];

pub fn is_video_control(class: u8, subclass: u8, _proto: u8) -> bool {
    class == USB_CLASS_VIDEO && subclass == SC_VIDEOCONTROL
}

pub fn is_video_streaming(class: u8, subclass: u8, _proto: u8) -> bool {
    class == USB_CLASS_VIDEO && subclass == SC_VIDEOSTREAMING
}

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameSize {
    pub frame_index: u8,
    pub width: u16,
    pub height: u16,
    /// 100 ns units; 0 if unknown (caller may use 333333 ≈ 30 fps).
    pub default_interval: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FormatInfo {
    pub format: PixelFormat,
    pub format_index: u8,
    pub frames: alloc::vec::Vec<FrameSize>,
}

/// Streaming endpoint on a VS alternate setting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamEp {
    pub vs_iface: u8,
    pub alt: u8,
    pub ep_addr: u8,
    pub mps: u16,
    pub interval: u8,
    pub bulk: bool,
}

/// Everything needed to start a still capture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamPlan {
    pub ep: StreamEp,
    pub format_index: u8,
    pub frame_index: u8,
    pub width: u16,
    pub height: u16,
    pub format: PixelFormat,
    pub frame_interval: u32,
}

// ── live inventory ───────────────────────────────────────────────────────

static SEEN: AtomicBool = AtomicBool::new(false);
static TRANSPORT: AtomicBool = AtomicBool::new(false);
static ROOT_PORT: AtomicU8 = AtomicU8::new(0);
static SLOT: AtomicU8 = AtomicU8::new(0);
static N_VC: AtomicU8 = AtomicU8::new(0);
static N_VS: AtomicU8 = AtomicU8::new(0);
static BEST_W: AtomicU16 = AtomicU16::new(0);
static BEST_H: AtomicU16 = AtomicU16::new(0);
static BEST_FMT: AtomicU8 = AtomicU8::new(0);
static LAST_GRAB_MS: AtomicU32 = AtomicU32::new(0);

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
        "uvc: Video interface noted (port {root_port} slot {slot} sub={sub})"
    ));
}

pub fn note_transport_ready() {
    TRANSPORT.store(true, Ordering::Release);
    crate::ktrace::log("uvc", "stream transport ready (PROBE/COMMIT + ep configured)");
}

pub fn seen() -> bool {
    SEEN.load(Ordering::Acquire)
}

pub fn transport_ready() -> bool {
    TRANSPORT.load(Ordering::Acquire) && crate::arch::uvc_ready()
}

pub fn clear_if_port(root_port: u8) {
    if SEEN.load(Ordering::Acquire) && ROOT_PORT.load(Ordering::Relaxed) == root_port {
        SEEN.store(false, Ordering::Release);
        TRANSPORT.store(false, Ordering::Release);
        N_VC.store(0, Ordering::Relaxed);
        N_VS.store(0, Ordering::Relaxed);
        BEST_W.store(0, Ordering::Relaxed);
        BEST_H.store(0, Ordering::Relaxed);
        BEST_FMT.store(0, Ordering::Relaxed);
        crate::ktrace::log("uvc", "camera gone with root port");
    }
}

pub fn try_parse_config(config: &[u8]) {
    if let Some(plan) = plan_stream(config) {
        BEST_W.store(plan.width, Ordering::Relaxed);
        BEST_H.store(plan.height, Ordering::Relaxed);
        BEST_FMT.store(
            match plan.format {
                PixelFormat::Mjpeg => 1,
                PixelFormat::Yuy2 => 2,
                PixelFormat::UncompressedOther => 3,
            },
            Ordering::Relaxed,
        );
        crate::ktrace::log_fmt(format_args!(
            "uvc: plan {} {}x{} iface {} alt {} {} ep {:#x} mps {}",
            plan.format.name(),
            plan.width,
            plan.height,
            plan.ep.vs_iface,
            plan.ep.alt,
            if plan.ep.bulk { "bulk" } else { "isoc" },
            plan.ep.ep_addr,
            plan.ep.mps
        ));
    }
}

pub fn status_lines() -> alloc::vec::Vec<alloc::string::String> {
    use alloc::format;
    use alloc::string::String;
    let mut v = alloc::vec::Vec::new();
    if !seen() {
        v.push(String::from("usb: no UVC interface noted at enumeration"));
        v.push(String::from("grab: plug a webcam, then /camera grab"));
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
        v.push(format!("format: {fmt} {w}x{h}"));
    }
    if transport_ready() {
        v.push(String::from(
            "stream: ready — /camera grab  (still → /downloads/camera-*.jpg|.yuy2)",
        ));
    } else {
        v.push(String::from(
            "stream: not configured (no usable isoc/bulk alt, or xHCI setup failed)",
        ));
    }
    let g = LAST_GRAB_MS.load(Ordering::Relaxed);
    if g > 0 {
        v.push(format!("last grab: ok at {g} ms uptime"));
    }
    v
}

pub fn mark_grab_ok() {
    LAST_GRAB_MS.store(crate::arch::now_ms() as u32, Ordering::Relaxed);
}

// ── pure: formats ────────────────────────────────────────────────────────

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
                VS_FORMAT_MJPEG if blen >= 4 => {
                    flush(&mut current, &mut out);
                    current = Some(FormatInfo {
                        format: PixelFormat::Mjpeg,
                        format_index: buf[i + 3],
                        frames: alloc::vec::Vec::new(),
                    });
                }
                VS_FORMAT_UNCOMPRESSED if blen >= 4 => {
                    flush(&mut current, &mut out);
                    let fmt = if blen >= 20 {
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
                        format_index: buf[i + 3],
                        frames: alloc::vec::Vec::new(),
                    });
                }
                VS_FRAME_MJPEG | VS_FRAME_UNCOMPRESSED if blen >= 10 => {
                    let frame_index = buf[i + 3];
                    let w = u16::from_le_bytes([buf[i + 5], buf[i + 6]]);
                    let h = u16::from_le_bytes([buf[i + 7], buf[i + 8]]);
                    let default_interval = if blen >= 26 {
                        u32::from_le_bytes([
                            buf[i + 21],
                            buf[i + 22],
                            buf[i + 23],
                            buf[i + 24],
                        ])
                    } else {
                        0
                    };
                    if w > 0 && h > 0 {
                        if let Some(ref mut f) = current {
                            f.frames.push(FrameSize {
                                frame_index,
                                width: w,
                                height: h,
                                default_interval,
                            });
                        }
                    }
                }
                _ => {}
            }
        } else if btype == 0x04 {
            flush(&mut current, &mut out);
        }
        i += blen;
    }
    flush(&mut current, &mut out);
    out
}

/// Find VS alternate settings that expose an IN isoc or bulk streaming endpoint.
pub fn find_stream_eps(desc: &[u8]) -> alloc::vec::Vec<StreamEp> {
    const DT_INTERFACE: u8 = 0x04;
    const DT_ENDPOINT: u8 = 0x05;
    let mut out = alloc::vec::Vec::new();
    let mut i = 0usize;
    let mut vs_iface = 0u8;
    let mut alt = 0u8;
    let mut in_vs = false;
    while i + 2 <= desc.len() {
        let len = desc[i] as usize;
        if len < 2 || i + len > desc.len() {
            break;
        }
        match desc[i + 1] {
            DT_INTERFACE if len >= 9 => {
                vs_iface = desc[i + 2];
                alt = desc[i + 3];
                let class = desc[i + 5];
                let sub = desc[i + 6];
                in_vs = is_video_streaming(class, sub, 0);
            }
            DT_ENDPOINT if len >= 7 && in_vs && alt > 0 => {
                let addr = desc[i + 2];
                let attrs = desc[i + 3];
                let mps = u16::from_le_bytes([desc[i + 4], desc[i + 5]]) & 0x07ff;
                let ivl = desc[i + 6];
                let xfer = attrs & 0x03;
                if addr & 0x80 != 0 && mps > 0 {
                    if xfer == 0x01 {
                        // isoc
                        out.push(StreamEp {
                            vs_iface,
                            alt,
                            ep_addr: addr,
                            mps,
                            interval: ivl.max(1),
                            bulk: false,
                        });
                    } else if xfer == 0x02 {
                        out.push(StreamEp {
                            vs_iface,
                            alt,
                            ep_addr: addr,
                            mps,
                            interval: ivl.max(1),
                            bulk: true,
                        });
                    }
                }
            }
            _ => {}
        }
        i += len;
    }
    out
}

/// Pick a stream plan: prefer MJPEG ≤ 640 wide, then any MJPEG, then YUY2 small.
pub fn plan_stream(desc: &[u8]) -> Option<StreamPlan> {
    let formats = parse_vs_formats(desc);
    let eps = find_stream_eps(desc);
    if formats.is_empty() || eps.is_empty() {
        return None;
    }
    // Prefer bulk (simpler host path) when MPS is large enough; else isoc.
    let ep = eps
        .iter()
        .copied()
        .filter(|e| e.bulk && e.mps >= 512)
        .max_by_key(|e| e.mps)
        .or_else(|| {
            eps.iter()
                .copied()
                .filter(|e| !e.bulk)
                .max_by_key(|e| e.mps)
        })
        .or_else(|| eps.iter().copied().max_by_key(|e| e.mps))?;

    let mut candidates: alloc::vec::Vec<(i32, PixelFormat, u8, FrameSize)> =
        alloc::vec::Vec::new();
    for f in &formats {
        for fr in &f.frames {
            let area = fr.width as i32 * fr.height as i32;
            // Score: MJPEG preferred; prefer width in 320..=640; avoid huge.
            let mut score = match f.format {
                PixelFormat::Mjpeg => 1_000_000,
                PixelFormat::Yuy2 => 100_000,
                PixelFormat::UncompressedOther => 0,
            };
            if fr.width <= 640 && fr.width >= 160 {
                score += 50_000 - (640 - fr.width as i32).abs() * 10;
            } else if fr.width > 640 {
                score -= (fr.width as i32 - 640) * 20;
            }
            score -= area / 1000; // slight preference for smaller stills
            candidates.push((score, f.format, f.format_index, *fr));
        }
    }
    candidates.sort_by(|a, b| b.0.cmp(&a.0));
    let (_, format, format_index, fr) = candidates.into_iter().next()?;
    let frame_interval = if fr.default_interval != 0 {
        fr.default_interval
    } else {
        333_333 // ~30 fps in 100ns units
    };
    Some(StreamPlan {
        ep,
        format_index,
        frame_index: fr.frame_index,
        width: fr.width,
        height: fr.height,
        format,
        frame_interval,
    })
}

// ── pure: PROBE/COMMIT + payload ─────────────────────────────────────────

/// Build a 26-byte VS_PROBE/COMMIT control structure.
pub fn build_probe(
    format_index: u8,
    frame_index: u8,
    frame_interval: u32,
    max_frame: u32,
    max_payload: u32,
) -> [u8; PROBE_LEN] {
    let mut b = [0u8; PROBE_LEN];
    b[0] = 0x01; // bmHint: dwFrameInterval
    b[1] = 0x00;
    b[2] = format_index;
    b[3] = frame_index;
    b[4..8].copy_from_slice(&frame_interval.to_le_bytes());
    b[18..22].copy_from_slice(&max_frame.to_le_bytes());
    b[22..26].copy_from_slice(&max_payload.to_le_bytes());
    b
}

/// Read format/frame indices back from a PROBE structure.
pub fn parse_probe_indices(b: &[u8]) -> Option<(u8, u8, u32, u32)> {
    if b.len() < PROBE_LEN {
        return None;
    }
    let interval = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
    let max_payload = u32::from_le_bytes([b[22], b[23], b[24], b[25]]);
    Some((b[2], b[3], interval, max_payload))
}

/// UVC payload header bitfield (byte 1 of header when HLE ≥ 2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PayloadHdr {
    pub header_len: usize,
    pub fid: bool,
    pub eof: bool,
    pub err: bool,
}

/// Parse UVC payload header at the start of an isoc/bulk packet.
pub fn parse_payload_header(pkt: &[u8]) -> Option<PayloadHdr> {
    if pkt.is_empty() {
        return None;
    }
    let hle = pkt[0] as usize;
    if hle < 2 || hle > pkt.len() || hle > 12 {
        return None;
    }
    let bfh = pkt[1];
    Some(PayloadHdr {
        header_len: hle,
        fid: bfh & 0x01 != 0,
        eof: bfh & 0x02 != 0,
        err: bfh & 0x40 != 0,
    })
}

/// Incremental frame assembler for UVC payloads.
#[derive(Clone, Debug, Default)]
pub struct FrameAssembler {
    buf: alloc::vec::Vec<u8>,
    fid: Option<bool>,
    done: bool,
}

impl FrameAssembler {
    pub fn new() -> Self {
        Self {
            buf: alloc::vec::Vec::new(),
            fid: None,
            done: false,
        }
    }

    pub fn reset(&mut self) {
        self.buf.clear();
        self.fid = None;
        self.done = false;
    }

    /// Feed one USB packet. Returns `Some(frame)` when EOF completes a frame.
    pub fn push(&mut self, pkt: &[u8]) -> Option<alloc::vec::Vec<u8>> {
        if self.done {
            self.reset();
        }
        let hdr = parse_payload_header(pkt)?;
        if hdr.err {
            self.reset();
            return None;
        }
        if let Some(fid) = self.fid {
            if fid != hdr.fid && !self.buf.is_empty() {
                // FID toggle without EOF — drop partial, start new.
                self.buf.clear();
            }
        }
        self.fid = Some(hdr.fid);
        let data = &pkt[hdr.header_len..];
        if self.buf.len() + data.len() > 2 * 1024 * 1024 {
            self.reset();
            return None;
        }
        self.buf.extend_from_slice(data);
        if hdr.eof && !self.buf.is_empty() {
            self.done = true;
            return Some(core::mem::take(&mut self.buf));
        }
        None
    }

    pub fn len(&self) -> usize {
        self.buf.len()
    }
}

pub fn guid_is_yuy2(g: &[u8]) -> bool {
    g.len() >= 16 && g[..16] == GUID_YUY2
}

/// Estimate max frame buffer for a plan (MJPEG soft cap / YUY2 exact).
pub fn max_frame_bytes(plan: &StreamPlan) -> u32 {
    match plan.format {
        PixelFormat::Mjpeg => {
            // Soft bound: ~1.5 bpp worst-case still for moderate sizes.
            let area = plan.width as u32 * plan.height as u32;
            (area + area / 2).max(64 * 1024).min(1024 * 1024)
        }
        PixelFormat::Yuy2 | PixelFormat::UncompressedOther => {
            plan.width as u32 * plan.height as u32 * 2
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn video_class_helpers() {
        assert!(is_video_control(0x0e, 0x01, 0));
        assert!(is_video_streaming(0x0e, 0x02, 0));
    }

    #[test_case]
    fn parse_mjpeg_with_indices() {
        let mut d = alloc::vec::Vec::new();
        d.extend_from_slice(&[11, 0x24, 0x06, 1, 1, 0, 0, 0, 0, 0, 0]);
        let mut fr = alloc::vec![0u8; 30];
        fr[0] = 30;
        fr[1] = 0x24;
        fr[2] = 0x07;
        fr[3] = 2; // frame index
        fr[5] = 0x80;
        fr[6] = 0x02; // 640
        fr[7] = 0xe0;
        fr[8] = 0x01; // 480
        fr[21..25].copy_from_slice(&333_333u32.to_le_bytes());
        d.extend_from_slice(&fr);
        let f = parse_vs_formats(&d);
        assert_eq!(f[0].format_index, 1);
        assert_eq!(f[0].frames[0].frame_index, 2);
        assert_eq!(f[0].frames[0].width, 640);
        assert_eq!(f[0].frames[0].default_interval, 333_333);
    }

    #[test_case]
    fn find_isoc_and_bulk_stream_eps() {
        let mut d = alloc::vec::Vec::new();
        // VS iface 1 alt 0 (no ep)
        d.extend_from_slice(&[9, 0x04, 1, 0, 0, 0x0e, 0x02, 0, 0]);
        // VS iface 1 alt 1 isoc
        d.extend_from_slice(&[9, 0x04, 1, 1, 1, 0x0e, 0x02, 0, 0]);
        d.extend_from_slice(&[7, 0x05, 0x81, 0x05, 0x00, 0x14, 1]); // isoc 5120? mps 0x1400 & 7ff = 1024
        // fix mps 1024 = 0x0400
        d.clear();
        d.extend_from_slice(&[9, 0x04, 1, 0, 0, 0x0e, 0x02, 0, 0]);
        d.extend_from_slice(&[9, 0x04, 1, 1, 1, 0x0e, 0x02, 0, 0]);
        d.extend_from_slice(&[7, 0x05, 0x81, 0x05, 0x00, 0x04, 1]); // isoc IN 1024
        d.extend_from_slice(&[9, 0x04, 1, 2, 1, 0x0e, 0x02, 0, 0]);
        d.extend_from_slice(&[7, 0x05, 0x82, 0x02, 0x00, 0x02, 0]); // bulk IN 512
        let eps = find_stream_eps(&d);
        assert!(eps.iter().any(|e| !e.bulk && e.mps == 1024 && e.alt == 1));
        assert!(eps.iter().any(|e| e.bulk && e.mps == 512 && e.alt == 2));
    }

    #[test_case]
    fn probe_roundtrip_and_payload_assemble() {
        let p = build_probe(1, 2, 333_333, 100_000, 1024);
        let (fi, fri, iv, mp) = parse_probe_indices(&p).unwrap();
        assert_eq!((fi, fri, iv, mp), (1, 2, 333_333, 1024));

        // Two packets: data + EOF
        let mut a = FrameAssembler::new();
        let mut p1 = alloc::vec![2u8, 0x01]; // HLE=2, FID=1
        p1.extend_from_slice(b"hello");
        assert!(a.push(&p1).is_none());
        let mut p2 = alloc::vec![2u8, 0x03]; // FID=1 EOF
        p2.extend_from_slice(b" world");
        let frame = a.push(&p2).unwrap();
        assert_eq!(&frame, b"hello world");
    }

    #[test_case]
    fn payload_rejects_bad_header() {
        assert!(parse_payload_header(&[]).is_none());
        assert!(parse_payload_header(&[1, 0]).is_none()); // HLE < 2
        assert!(parse_payload_header(&[20, 0]).is_none()); // HLE > len
    }

    #[test_case]
    fn plan_prefers_modest_mjpeg() {
        let mut d = alloc::vec::Vec::new();
        d.extend_from_slice(&[9, 0x04, 1, 0, 0, 0x0e, 0x02, 0, 0]);
        d.extend_from_slice(&[9, 0x04, 1, 1, 1, 0x0e, 0x02, 0, 0]);
        d.extend_from_slice(&[7, 0x05, 0x81, 0x05, 0x00, 0x04, 1]);
        // format 1 MJPEG with 320 and 1920 frames
        d.extend_from_slice(&[11, 0x24, 0x06, 1, 1, 0, 0, 0, 0, 0, 0]);
        let mut fr = alloc::vec![0u8; 30];
        fr[0] = 30;
        fr[1] = 0x24;
        fr[2] = 0x07;
        fr[3] = 1;
        fr[5..7].copy_from_slice(&320u16.to_le_bytes());
        fr[7..9].copy_from_slice(&240u16.to_le_bytes());
        d.extend_from_slice(&fr);
        fr[3] = 2;
        fr[5..7].copy_from_slice(&1920u16.to_le_bytes());
        fr[7..9].copy_from_slice(&1080u16.to_le_bytes());
        d.extend_from_slice(&fr);
        let plan = plan_stream(&d).unwrap();
        assert_eq!(plan.format, PixelFormat::Mjpeg);
        assert_eq!(plan.width, 320);
    }
}
