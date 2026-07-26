//! virtio-gpu **wire protocol** — pure encode/decode, no MMIO.
//!
//! Split out deliberately, following the same discipline as `agx/proto.rs`: the
//! byte layouts and the response parsing are the part that is easy to get subtly
//! wrong and impossible to debug on a screen that has gone black, so they live
//! here where `cargo xtask test` covers them. The transport half (virtqueue setup
//! over MMIO and PCI, doorbells, used-ring polling) is per-arch hardware code and
//! stays out of this file.
//!
//! Layouts are from the virtio 1.x spec, §5.7 (GPU Device). Every structure is
//! little-endian and every command is prefixed by a [`CtrlHdr`].

use alloc::vec::Vec;

/// virtio device id for the GPU.
pub const VIRTIO_ID_GPU: u32 = 16;

/// Feature bit: the device can report an output's EDID (`GET_EDID`).
pub const F_EDID: u32 = 1;

// --- command / response types ------------------------------------------------

pub const CMD_GET_DISPLAY_INFO: u32 = 0x0100;
pub const CMD_RESOURCE_CREATE_2D: u32 = 0x0101;
pub const CMD_RESOURCE_UNREF: u32 = 0x0102;
pub const CMD_SET_SCANOUT: u32 = 0x0103;
pub const CMD_RESOURCE_FLUSH: u32 = 0x0104;
pub const CMD_TRANSFER_TO_HOST_2D: u32 = 0x0105;
pub const CMD_RESOURCE_ATTACH_BACKING: u32 = 0x0106;
pub const CMD_GET_EDID: u32 = 0x010a;

pub const RESP_OK_NODATA: u32 = 0x1100;
pub const RESP_OK_DISPLAY_INFO: u32 = 0x1101;
pub const RESP_OK_EDID: u32 = 0x1104;

/// `B8G8R8X8_UNORM` — bytes B,G,R,X in memory, i.e. the little-endian `XRGB8888`
/// the compositor already packs (red at bit 16). Chosen so a mode set does not
/// change the pixel format and every existing blit keeps working.
pub const FORMAT_B8G8R8X8: u32 = 2;

/// Shifts implied by [`FORMAT_B8G8R8X8`]: `(r, g, b)`.
pub const FORMAT_B8G8R8X8_SHIFTS: (u32, u32, u32) = (16, 8, 0);

/// The maximum scanouts the spec allows a device to report.
pub const MAX_SCANOUTS: usize = 16;

/// Size of `virtio_gpu_ctrl_hdr`.
pub const CTRL_HDR_LEN: usize = 24;

/// `virtio_gpu_ctrl_hdr` — the prefix on every request and response.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CtrlHdr {
    pub ty: u32,
    pub flags: u32,
    pub fence_id: u64,
    pub ctx_id: u32,
    pub ring_idx: u8,
}

impl CtrlHdr {
    /// A plain request header: no fence, no context, ring 0.
    pub const fn cmd(ty: u32) -> CtrlHdr {
        CtrlHdr { ty, flags: 0, fence_id: 0, ctx_id: 0, ring_idx: 0 }
    }

    fn encode_into(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.ty.to_le_bytes());
        out.extend_from_slice(&self.flags.to_le_bytes());
        out.extend_from_slice(&self.fence_id.to_le_bytes());
        out.extend_from_slice(&self.ctx_id.to_le_bytes());
        out.push(self.ring_idx);
        out.extend_from_slice(&[0u8; 3]); // padding
    }

    /// Parse a response header. `None` if the buffer is too short.
    pub fn parse(b: &[u8]) -> Option<CtrlHdr> {
        if b.len() < CTRL_HDR_LEN {
            return None;
        }
        let u32at = |o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
        Some(CtrlHdr {
            ty: u32at(0),
            flags: u32at(4),
            fence_id: u64::from_le_bytes(b[8..16].try_into().ok()?),
            ctx_id: u32at(16),
            ring_idx: b[20],
        })
    }

    /// Whether this response reports success (any `OK_*` type).
    pub fn is_ok(&self) -> bool {
        (0x1100..0x1200).contains(&self.ty)
    }
}

/// `virtio_gpu_rect`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

impl Rect {
    pub const fn new(x: u32, y: u32, w: u32, h: u32) -> Rect {
        Rect { x, y, w, h }
    }
    fn encode_into(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.x.to_le_bytes());
        out.extend_from_slice(&self.y.to_le_bytes());
        out.extend_from_slice(&self.w.to_le_bytes());
        out.extend_from_slice(&self.h.to_le_bytes());
    }
    fn parse(b: &[u8]) -> Option<Rect> {
        if b.len() < 16 {
            return None;
        }
        let at = |o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
        Some(Rect { x: at(0), y: at(4), w: at(8), h: at(12) })
    }
}

// --- request encoders --------------------------------------------------------

/// `GET_DISPLAY_INFO` — header only.
pub fn get_display_info() -> Vec<u8> {
    let mut v = Vec::with_capacity(CTRL_HDR_LEN);
    CtrlHdr::cmd(CMD_GET_DISPLAY_INFO).encode_into(&mut v);
    v
}

/// `GET_EDID` for one scanout.
pub fn get_edid(scanout: u32) -> Vec<u8> {
    let mut v = Vec::with_capacity(CTRL_HDR_LEN + 8);
    CtrlHdr::cmd(CMD_GET_EDID).encode_into(&mut v);
    v.extend_from_slice(&scanout.to_le_bytes());
    v.extend_from_slice(&0u32.to_le_bytes()); // padding
    v
}

/// `RESOURCE_CREATE_2D` — declare a host-side resource of `w`x`h`.
pub fn resource_create_2d(resource_id: u32, w: u32, h: u32) -> Vec<u8> {
    let mut v = Vec::with_capacity(CTRL_HDR_LEN + 16);
    CtrlHdr::cmd(CMD_RESOURCE_CREATE_2D).encode_into(&mut v);
    v.extend_from_slice(&resource_id.to_le_bytes());
    v.extend_from_slice(&FORMAT_B8G8R8X8.to_le_bytes());
    v.extend_from_slice(&w.to_le_bytes());
    v.extend_from_slice(&h.to_le_bytes());
    v
}

/// `RESOURCE_UNREF` — drop a resource (used when replacing one on a mode set).
pub fn resource_unref(resource_id: u32) -> Vec<u8> {
    let mut v = Vec::with_capacity(CTRL_HDR_LEN + 8);
    CtrlHdr::cmd(CMD_RESOURCE_UNREF).encode_into(&mut v);
    v.extend_from_slice(&resource_id.to_le_bytes());
    v.extend_from_slice(&0u32.to_le_bytes()); // padding
    v
}

/// `RESOURCE_ATTACH_BACKING` — give the host our guest pages for `resource_id`.
///
/// `entries` are `(guest physical address, length)`. This is what makes the
/// framebuffer *ours*: the compositor writes straight into these pages and a
/// later `TRANSFER_TO_HOST_2D` tells the device to pick the bytes up.
pub fn resource_attach_backing(resource_id: u32, entries: &[(u64, u32)]) -> Vec<u8> {
    let mut v = Vec::with_capacity(CTRL_HDR_LEN + 8 + entries.len() * 16);
    CtrlHdr::cmd(CMD_RESOURCE_ATTACH_BACKING).encode_into(&mut v);
    v.extend_from_slice(&resource_id.to_le_bytes());
    v.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    for &(addr, len) in entries {
        v.extend_from_slice(&addr.to_le_bytes());
        v.extend_from_slice(&len.to_le_bytes());
        v.extend_from_slice(&0u32.to_le_bytes()); // padding
    }
    v
}

/// `SET_SCANOUT` — point output `scanout_id` at `resource_id`'s `rect`.
///
/// A `resource_id` of 0 *disables* the scanout, which is how the spec says to turn
/// an output off; passing 0 by accident blanks the screen.
pub fn set_scanout(scanout_id: u32, resource_id: u32, rect: Rect) -> Vec<u8> {
    let mut v = Vec::with_capacity(CTRL_HDR_LEN + 24);
    CtrlHdr::cmd(CMD_SET_SCANOUT).encode_into(&mut v);
    rect.encode_into(&mut v);
    v.extend_from_slice(&scanout_id.to_le_bytes());
    v.extend_from_slice(&resource_id.to_le_bytes());
    v
}

/// `TRANSFER_TO_HOST_2D` — copy a dirty rect out of our pages into the resource.
pub fn transfer_to_host_2d(resource_id: u32, rect: Rect, offset: u64) -> Vec<u8> {
    let mut v = Vec::with_capacity(CTRL_HDR_LEN + 32);
    CtrlHdr::cmd(CMD_TRANSFER_TO_HOST_2D).encode_into(&mut v);
    rect.encode_into(&mut v);
    v.extend_from_slice(&offset.to_le_bytes());
    v.extend_from_slice(&resource_id.to_le_bytes());
    v.extend_from_slice(&0u32.to_le_bytes()); // padding
    v
}

/// `RESOURCE_FLUSH` — present a rect of the resource on screen.
pub fn resource_flush(resource_id: u32, rect: Rect) -> Vec<u8> {
    let mut v = Vec::with_capacity(CTRL_HDR_LEN + 24);
    CtrlHdr::cmd(CMD_RESOURCE_FLUSH).encode_into(&mut v);
    rect.encode_into(&mut v);
    v.extend_from_slice(&resource_id.to_le_bytes());
    v.extend_from_slice(&0u32.to_le_bytes()); // padding
    v
}

// --- response decoders -------------------------------------------------------

/// One output as `GET_DISPLAY_INFO` reports it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct DisplayOne {
    pub rect: Rect,
    pub enabled: bool,
    pub flags: u32,
}

/// Parse an `OK_DISPLAY_INFO` response into its scanouts.
///
/// The response is a fixed array of [`MAX_SCANOUTS`] entries regardless of how
/// many the device actually has — the caller pairs it with `num_scanouts` from
/// config space. A short buffer yields the entries that *are* present rather than
/// failing, since a device reporting fewer is legitimate.
pub fn parse_display_info(b: &[u8]) -> Option<Vec<DisplayOne>> {
    let hdr = CtrlHdr::parse(b)?;
    if hdr.ty != RESP_OK_DISPLAY_INFO {
        return None;
    }
    let mut out = Vec::new();
    for i in 0..MAX_SCANOUTS {
        let at = CTRL_HDR_LEN + i * 24;
        if at + 24 > b.len() {
            break;
        }
        let rect = Rect::parse(&b[at..])?;
        let u32at = |o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
        out.push(DisplayOne { rect, enabled: u32at(at + 16) != 0, flags: u32at(at + 20) });
    }
    Some(out)
}

/// Parse an `OK_EDID` response into the EDID bytes.
pub fn parse_edid(b: &[u8]) -> Option<Vec<u8>> {
    let hdr = CtrlHdr::parse(b)?;
    if hdr.ty != RESP_OK_EDID {
        return None;
    }
    // Layout: hdr, size u32, padding u32, then `size` bytes of EDID.
    let size = u32::from_le_bytes(b.get(CTRL_HDR_LEN..CTRL_HDR_LEN + 4)?.try_into().ok()?) as usize;
    let start = CTRL_HDR_LEN + 8;
    let end = start.checked_add(size)?;
    if size == 0 || end > b.len() {
        return None;
    }
    Some(b[start..end].to_vec())
}

/// Turn a `GET_DISPLAY_INFO` reply into KMS connectors.
///
/// A scanout's reported rect is its *current* mode, which the device also treats
/// as preferred — so it becomes the connector's preferred mode and heads the mode
/// list. The rest of the list is the standard sizes that fit within it, because
/// virtio-gpu has no EDID-style mode table of its own: any size up to the host
/// window is valid, which is the opposite problem from a fixed panel.
pub fn connectors_from_display_info(
    infos: &[DisplayOne],
    num_scanouts: u32,
    standard: &[(u32, u32)],
) -> Vec<super::Connector> {
    let n = (num_scanouts as usize).min(infos.len());
    let mut out = Vec::with_capacity(n);
    for (i, d) in infos.iter().take(n).enumerate() {
        let cur = super::Mode::new(d.rect.w, d.rect.h);
        let mut modes = Vec::new();
        if cur.w > 0 && cur.h > 0 {
            modes.push(cur);
        }
        for &(w, h) in standard {
            let m = super::Mode::new(w, h);
            if m != cur && w <= cur.w.max(1) && h <= cur.h.max(1) {
                modes.push(m);
            }
        }
        out.push(super::Connector {
            id: i as u32,
            // DRM-style naming: virtio-gpu outputs are virtual.
            name: alloc::format!("Virtual-{}", i + 1),
            connected: d.enabled,
            preferred: (cur.w > 0 && cur.h > 0).then_some(cur),
            modes,
            edid: None,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn ctrl_hdr_round_trips_with_correct_padding() {
        let mut v = Vec::new();
        CtrlHdr::cmd(CMD_GET_DISPLAY_INFO).encode_into(&mut v);
        // 24 bytes exactly: the 3 padding bytes after ring_idx are part of the
        // struct, and omitting them shifts every following field.
        assert_eq!(v.len(), CTRL_HDR_LEN);
        let back = CtrlHdr::parse(&v).unwrap();
        assert_eq!(back.ty, CMD_GET_DISPLAY_INFO);
        assert_eq!(back.fence_id, 0);
        assert!(CtrlHdr::parse(&v[..CTRL_HDR_LEN - 1]).is_none(), "short buffer must not parse");
    }

    #[test_case]
    fn response_ok_range_is_recognised() {
        assert!(CtrlHdr::cmd(RESP_OK_NODATA).is_ok());
        assert!(CtrlHdr::cmd(RESP_OK_DISPLAY_INFO).is_ok());
        assert!(CtrlHdr::cmd(RESP_OK_EDID).is_ok());
        // Error responses start at 0x1200.
        assert!(!CtrlHdr::cmd(0x1200).is_ok());
        assert!(!CtrlHdr::cmd(0x1201).is_ok());
        // A request type is not a success response.
        assert!(!CtrlHdr::cmd(CMD_SET_SCANOUT).is_ok());
    }

    #[test_case]
    fn command_lengths_match_the_spec_structs() {
        assert_eq!(get_display_info().len(), CTRL_HDR_LEN);
        assert_eq!(get_edid(0).len(), CTRL_HDR_LEN + 8);
        assert_eq!(resource_create_2d(1, 640, 480).len(), CTRL_HDR_LEN + 16);
        assert_eq!(resource_unref(1).len(), CTRL_HDR_LEN + 8);
        assert_eq!(set_scanout(0, 1, Rect::new(0, 0, 640, 480)).len(), CTRL_HDR_LEN + 24);
        assert_eq!(resource_flush(1, Rect::new(0, 0, 640, 480)).len(), CTRL_HDR_LEN + 24);
        assert_eq!(transfer_to_host_2d(1, Rect::new(0, 0, 640, 480), 0).len(), CTRL_HDR_LEN + 32);
        // One backing entry is 16 bytes (u64 addr, u32 len, u32 pad).
        assert_eq!(resource_attach_backing(1, &[(0x1000, 4096)]).len(), CTRL_HDR_LEN + 8 + 16);
        assert_eq!(
            resource_attach_backing(1, &[(0x1000, 4096), (0x2000, 4096)]).len(),
            CTRL_HDR_LEN + 8 + 32
        );
    }

    #[test_case]
    fn resource_create_2d_carries_the_expected_format() {
        let v = resource_create_2d(7, 1920, 1080);
        let at = |o: usize| u32::from_le_bytes(v[o..o + 4].try_into().unwrap());
        assert_eq!(at(CTRL_HDR_LEN), 7, "resource id");
        assert_eq!(at(CTRL_HDR_LEN + 4), FORMAT_B8G8R8X8);
        assert_eq!(at(CTRL_HDR_LEN + 8), 1920);
        assert_eq!(at(CTRL_HDR_LEN + 12), 1080);
        // The format must be the one whose shifts match what the compositor packs,
        // or every colour comes out swapped.
        assert_eq!(FORMAT_B8G8R8X8_SHIFTS, (16, 8, 0));
    }

    #[test_case]
    fn set_scanout_puts_the_rect_before_the_ids() {
        let v = set_scanout(3, 9, Rect::new(1, 2, 800, 600));
        let at = |o: usize| u32::from_le_bytes(v[o..o + 4].try_into().unwrap());
        assert_eq!((at(CTRL_HDR_LEN), at(CTRL_HDR_LEN + 4)), (1, 2), "rect x,y");
        assert_eq!((at(CTRL_HDR_LEN + 8), at(CTRL_HDR_LEN + 12)), (800, 600), "rect w,h");
        assert_eq!(at(CTRL_HDR_LEN + 16), 3, "scanout id follows the rect");
        assert_eq!(at(CTRL_HDR_LEN + 20), 9, "resource id last");
    }

    #[test_case]
    fn attach_backing_encodes_each_entry() {
        let v = resource_attach_backing(5, &[(0xDEAD_BEEF_0000, 8192), (0x1_0000, 4096)]);
        let u32at = |o: usize| u32::from_le_bytes(v[o..o + 4].try_into().unwrap());
        let u64at = |o: usize| u64::from_le_bytes(v[o..o + 8].try_into().unwrap());
        assert_eq!(u32at(CTRL_HDR_LEN), 5);
        assert_eq!(u32at(CTRL_HDR_LEN + 4), 2, "nr_entries");
        assert_eq!(u64at(CTRL_HDR_LEN + 8), 0xDEAD_BEEF_0000);
        assert_eq!(u32at(CTRL_HDR_LEN + 16), 8192);
        assert_eq!(u64at(CTRL_HDR_LEN + 24), 0x1_0000);
        assert_eq!(u32at(CTRL_HDR_LEN + 32), 4096);
    }

    /// Build an `OK_DISPLAY_INFO` reply with `rects` as the enabled scanouts.
    fn display_info_reply(rects: &[(u32, u32, bool)]) -> Vec<u8> {
        let mut v = Vec::new();
        CtrlHdr::cmd(RESP_OK_DISPLAY_INFO).encode_into(&mut v);
        for i in 0..MAX_SCANOUTS {
            let (w, h, on) = rects.get(i).copied().unwrap_or((0, 0, false));
            Rect::new(0, 0, w, h).encode_into(&mut v);
            v.extend_from_slice(&(on as u32).to_le_bytes());
            v.extend_from_slice(&0u32.to_le_bytes());
        }
        v
    }

    #[test_case]
    fn parse_display_info_reads_every_scanout() {
        let b = display_info_reply(&[(1920, 1080, true), (1280, 720, false)]);
        let d = parse_display_info(&b).expect("parses");
        assert_eq!(d.len(), MAX_SCANOUTS);
        assert_eq!(d[0].rect, Rect::new(0, 0, 1920, 1080));
        assert!(d[0].enabled);
        assert_eq!(d[1].rect, Rect::new(0, 0, 1280, 720));
        assert!(!d[1].enabled);
        // Wrong response type is refused rather than misread.
        let mut bad = b.clone();
        bad[0..4].copy_from_slice(&RESP_OK_NODATA.to_le_bytes());
        assert!(parse_display_info(&bad).is_none());
        // A truncated reply yields the entries that are present.
        let short = &b[..CTRL_HDR_LEN + 24];
        assert_eq!(parse_display_info(short).unwrap().len(), 1);
    }

    #[test_case]
    fn parse_edid_extracts_only_the_reported_size() {
        let mut v = Vec::new();
        CtrlHdr::cmd(RESP_OK_EDID).encode_into(&mut v);
        v.extend_from_slice(&128u32.to_le_bytes());
        v.extend_from_slice(&0u32.to_le_bytes());
        v.extend_from_slice(&[0xAB; 200]); // more bytes present than reported
        let e = parse_edid(&v).unwrap();
        assert_eq!(e.len(), 128, "must honour the reported size, not the buffer");
        // A size that runs past the buffer is refused rather than read OOB.
        let mut bad = v[..CTRL_HDR_LEN + 8 + 16].to_vec();
        bad[CTRL_HDR_LEN..CTRL_HDR_LEN + 4].copy_from_slice(&9999u32.to_le_bytes());
        assert!(parse_edid(&bad).is_none());
        // Zero size is "no EDID", not an empty success.
        let mut zero = v.clone();
        zero[CTRL_HDR_LEN..CTRL_HDR_LEN + 4].copy_from_slice(&0u32.to_le_bytes());
        assert!(parse_edid(&zero).is_none());
    }

    #[test_case]
    fn connectors_report_the_current_mode_as_preferred() {
        let infos = parse_display_info(&display_info_reply(&[(1920, 1080, true), (1280, 720, true)]))
            .unwrap();
        let standard = [(1920u32, 1080u32), (1280, 720), (1024, 768), (2560, 1440)];
        let cs = super::connectors_from_display_info(&infos, 2, &standard);
        assert_eq!(cs.len(), 2, "only num_scanouts connectors, not all 16");
        assert_eq!(cs[0].name, "Virtual-1");
        assert!(cs[0].connected);
        assert_eq!(cs[0].preferred, Some(crate::kms::Mode::new(1920, 1080)));
        assert_eq!(cs[0].modes[0], crate::kms::Mode::new(1920, 1080), "current mode heads the list");
        // Nothing larger than the current mode is offered, and no duplicate.
        assert!(!cs[0].modes.contains(&crate::kms::Mode::new(2560, 1440)));
        assert_eq!(
            cs[0].modes.iter().filter(|m| **m == crate::kms::Mode::new(1920, 1080)).count(),
            1
        );
        assert!(cs[0].modes.contains(&crate::kms::Mode::new(1280, 720)));
        // The second output is scaled to its own current mode.
        assert_eq!(cs[1].preferred, Some(crate::kms::Mode::new(1280, 720)));
        assert!(!cs[1].modes.contains(&crate::kms::Mode::new(1920, 1080)));
    }

    #[test_case]
    fn connectors_tolerate_a_disabled_or_zero_sized_scanout() {
        let infos = parse_display_info(&display_info_reply(&[(0, 0, false)])).unwrap();
        let cs = super::connectors_from_display_info(&infos, 1, &[(1024, 768)]);
        assert_eq!(cs.len(), 1);
        assert!(!cs[0].connected);
        assert_eq!(cs[0].preferred, None, "a 0x0 scanout has no preferred mode");
        // num_scanouts larger than the reply cannot over-read.
        let cs = super::connectors_from_display_info(&infos, 99, &[]);
        assert!(cs.len() <= MAX_SCANOUTS);
    }
}
