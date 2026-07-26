//! EDID (E-DID 1.3/1.4 base block) parsing — how a display reports its own
//! **native** resolution.
//!
//! This is the standards-based answer to "what resolution should the console
//! be?", and it replaces two pieces of guesswork that both broke on real
//! platforms: hardcoding a mode (the x86 Limine config pinned 2560x1440) and
//! picking the *largest* mode the firmware advertises (the aarch64 UEFI stub).
//! A hypervisor typically advertises a long mode list up to some maximum that
//! has nothing to do with the display the user configured, so "largest" is not
//! "native" — it just silently ignored the VM's setting.
//!
//! Deliberately parse-only and self-contained (no `crate::` imports): the UEFI
//! stub mounts this file directly with `#[path]`, so it must compile in both
//! crates, and the fiddly bit-packing gets unit-tested under
//! `cargo xtask test` rather than only on hardware we can't easily inspect.

/// Length of the EDID base block. Extension blocks (CEA-861 etc.) follow it and
/// are not needed to find the preferred timing.
pub const BASE_BLOCK_LEN: usize = 128;

/// The fixed 8-byte EDID header every valid base block starts with.
const HEADER: [u8; 8] = [0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00];

/// Offset of the first of four 18-byte detailed timing descriptors.
const DTD0: usize = 54;
const DTD_LEN: usize = 18;
const DTD_COUNT: usize = 4;

/// Whether `edid` is a structurally valid base block: long enough, correct
/// header, and a base block whose 128 bytes sum to zero mod 256.
///
/// Both checks matter. A bus read that returns all `0x00` or all `0xFF` (an
/// absent or unpowered display, or a hypervisor with no EDID at all) passes a
/// naive "is it non-empty" test and would otherwise yield a plausible-looking
/// resolution built from garbage.
pub fn is_valid(edid: &[u8]) -> bool {
    if edid.len() < BASE_BLOCK_LEN || edid[..8] != HEADER {
        return false;
    }
    checksum_ok(&edid[..BASE_BLOCK_LEN])
}

/// True if a 128-byte block's bytes sum to 0 mod 256 (EDID's checksum rule).
fn checksum_ok(block: &[u8]) -> bool {
    block.iter().fold(0u8, |a, &b| a.wrapping_add(b)) == 0
}

/// The active resolution of one 18-byte detailed timing descriptor, or `None`
/// when the slot is unused or holds a *display* descriptor rather than a timing.
///
/// The dimensions are split across bytes: the low 8 bits of each are their own
/// byte, and the high 4 bits are packed into the **top nibble** of a shared byte
/// with the blanking interval's high bits. Forgetting the shift is how a
/// 2560-wide panel reads back as 0 (2560 = `0xA00`, whose low byte is `0x00`).
pub fn dtd_resolution(dtd: &[u8]) -> Option<(u32, u32)> {
    if dtd.len() < DTD_LEN {
        return None;
    }
    // Bytes 0..2 are the pixel clock in 10 kHz units; zero marks the slot as a
    // display descriptor (monitor name, range limits, …), not a timing.
    if dtd[0] == 0 && dtd[1] == 0 {
        return None;
    }
    let w = dtd[2] as u32 | ((dtd[4] as u32 & 0xF0) << 4);
    let h = dtd[5] as u32 | ((dtd[7] as u32 & 0xF0) << 4);
    if w == 0 || h == 0 {
        return None;
    }
    Some((w, h))
}

/// The display's **preferred** (native) resolution from a valid EDID.
///
/// The first detailed timing descriptor is the preferred one by convention (and
/// by requirement in EDID 1.4). The later slots are searched only as a fallback
/// for displays that leave the first empty; `None` means "this EDID does not say"
/// — a caller must then keep whatever mode the firmware chose rather than invent
/// one.
pub fn preferred_resolution(edid: &[u8]) -> Option<(u32, u32)> {
    if !is_valid(edid) {
        return None;
    }
    (0..DTD_COUNT).find_map(|i| {
        let at = DTD0 + i * DTD_LEN;
        dtd_resolution(edid.get(at..at + DTD_LEN)?)
    })
}

/// Pick the best display mode for a panel whose native size is `native`, from
/// `modes` given as `(index, width, height)`.
///
/// Prefers an exact match; otherwise the largest mode that fits **inside** the
/// native size, so the picture is never scaled up or cropped by the display.
/// `None` when nothing fits, which tells the caller to leave the mode alone.
pub fn best_mode_for<I>(native: (u32, u32), modes: I) -> Option<usize>
where
    I: IntoIterator<Item = (usize, u32, u32)>,
{
    let (nw, nh) = native;
    let mut fit: Option<(usize, u64)> = None;
    for (i, w, h) in modes {
        if w == nw && h == nh {
            return Some(i);
        }
        if w <= nw && h <= nh {
            let area = w as u64 * h as u64;
            if fit.is_none_or(|(_, best)| area > best) {
                fit = Some((i, area));
            }
        }
    }
    fit.map(|(i, _)| i)
}

/// A display's identity, as it identifies *itself* in its EDID.
///
/// This is what makes per-monitor settings possible: it is stable across reboots
/// and cable swaps, and different for two different panels — the same thing
/// GNOME's `monitors.xml` keys its per-output configuration on, and what DRM
/// exposes per connector.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct DisplayId {
    /// PnP manufacturer code, three uppercase letters (e.g. `DEL`, `APP`).
    pub mfg: [u8; 3],
    /// Vendor product code.
    pub product: u16,
    /// Vendor serial number (`0` when the display reports none — common).
    pub serial: u32,
}

impl DisplayId {
    /// A stable, filesystem/JSON-safe key like `DEL-A1B2-00034F1C`.
    pub fn key(&self) -> alloc::string::String {
        alloc::format!(
            "{}{}{}-{:04X}-{:08X}",
            self.mfg[0] as char,
            self.mfg[1] as char,
            self.mfg[2] as char,
            self.product,
            self.serial
        )
    }
}

/// The display's self-reported identity, or `None` if the EDID is not valid.
///
/// The manufacturer code is three **five-bit** letters packed big-endian into
/// bytes 8..10 (`1 = 'A'`), which is the one field here that is not a plain
/// little-endian integer — reading it as a `u16` yields nonsense.
pub fn identity(edid: &[u8]) -> Option<DisplayId> {
    if !is_valid(edid) {
        return None;
    }
    let raw = u16::from_be_bytes([edid[8], edid[9]]);
    let letter = |shift: u16| -> u8 {
        let v = ((raw >> shift) & 0x1F) as u8;
        if (1..=26).contains(&v) {
            b'A' + v - 1
        } else {
            b'?'
        }
    };
    let mfg = [letter(10), letter(5), letter(0)];
    Some(DisplayId {
        mfg,
        product: u16::from_le_bytes([edid[10], edid[11]]),
        serial: u32::from_le_bytes([edid[12], edid[13], edid[14], edid[15]]),
    })
}

/// The human-readable monitor name from the EDID product-name descriptor
/// (`0xFC`), e.g. `DELL P2722HE`. `None` when the display doesn't publish one.
///
/// A *display* descriptor is distinguished from a *timing* descriptor by a zero
/// pixel clock, with the type in byte 3; the text runs to a `0x0A` terminator and
/// is space-padded.
pub fn monitor_name(edid: &[u8]) -> Option<alloc::string::String> {
    if !is_valid(edid) {
        return None;
    }
    for i in 0..DTD_COUNT {
        let d = edid.get(DTD0 + i * DTD_LEN..DTD0 + (i + 1) * DTD_LEN)?;
        if d[0] != 0 || d[1] != 0 || d[3] != 0xFC {
            continue;
        }
        let mut s = alloc::string::String::new();
        for &b in &d[5..DTD_LEN] {
            if b == 0x0A {
                break;
            }
            // Printable ASCII only: a garbled descriptor must not inject control
            // characters into a name that ends up in a log line or a JSON key.
            if (0x20..0x7F).contains(&b) {
                s.push(b as char);
            }
        }
        let t = s.trim_end();
        if !t.is_empty() {
            return Some(alloc::string::String::from(t));
        }
    }
    None
}

/// What is known about one graphics output while choosing which display the
/// console should use. A real laptop with a monitor attached exposes **one output
/// per display**, so "the first one" is a coin flip.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct OutputInfo {
    /// The firmware installed its console-out marker on this output — i.e. this
    /// is where the firmware itself draws, so it is where the user is looking.
    pub console_out: bool,
    /// Native resolution from this output's EDID, if a display reported one.
    /// `None` means nothing is plugged in, or it reported no usable EDID.
    pub edid_native: Option<(u32, u32)>,
}

/// Choose which graphics output to drive, from `outputs` in firmware order.
///
/// The order is deliberate and standards-based rather than a guess:
///
/// 1. The output carrying the firmware's **console-out** marker. That is the
///    display the firmware's own boot messages went to, which is by definition
///    the one the user is watching.
/// 2. Failing that, any output whose **EDID** we could read — proof that a
///    display is actually connected, rather than a dark or absent port.
/// 3. Failing that, the first output, so a headless or EDID-less machine still
///    gets a console instead of nothing.
///
/// Returns an index into `outputs`, or `None` if it is empty.
pub fn pick_output(outputs: &[OutputInfo]) -> Option<usize> {
    if outputs.is_empty() {
        return None;
    }
    // Prefer a console-out output that also has an EDID, then any console-out.
    if let Some(i) = outputs
        .iter()
        .position(|o| o.console_out && o.edid_native.is_some())
    {
        return Some(i);
    }
    if let Some(i) = outputs.iter().position(|o| o.console_out) {
        return Some(i);
    }
    if let Some(i) = outputs.iter().position(|o| o.edid_native.is_some()) {
        return Some(i);
    }
    Some(0)
}

/// Whether a mode the firmware left set is implausibly small for a console —
/// the classic "UEFI came up at 800x600" case on real hardware.
///
/// Used as the *last* resort before falling back to the largest advertised mode:
/// with no EDID we must trust the firmware's choice (that is what a hypervisor's
/// configured resolution looks like), but a sub-VGA-era surface is a default
/// nobody chose, not a setting.
pub fn is_implausibly_small(w: u32, h: u32) -> bool {
    w < 1024 || h < 768
}

/// The name of the loader's display-preference file on the ESP.
///
/// It lives on the **FAT ESP** rather than with the rest of the settings on the
/// ext4 data partition for one reason: the stub runs before any of the kernel
/// exists and can only read FAT, so this is the one channel a human has to the
/// mode the framebuffer is *created* in.
pub const BOOT_CFG_PATH: &str = "chitti-display.cfg";

/// Parse the loader display-preference file: `resolution=<W>x<H>` lines, `#`
/// comments, blank lines and unknown keys ignored.
///
/// Deliberately tolerant about everything except the value itself: a file a
/// human hand-edited on the ESP of a machine that will not boot should not brick
/// the boot, so a malformed line is skipped rather than failing the parse. A
/// malformed *resolution* returns `None` (keep the firmware's mode) instead of a
/// guess — the whole point of this file is that the guess was wrong.
///
/// The last valid `resolution` wins, so appending a line overrides an earlier one.
pub fn parse_boot_cfg(text: &str) -> Option<(u32, u32)> {
    let mut found = None;
    for line in text.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        let Some((key, val)) = line.split_once('=') else { continue };
        if key.trim() != "resolution" {
            continue;
        }
        if let Some(dims) = parse_dims(val.trim()) {
            found = Some(dims);
        }
    }
    found
}

/// `<W>x<H>` → dimensions, rejecting anything a framebuffer could not be.
///
/// The ceiling is not arbitrary politeness: the surface is `w * h * 4` bytes of
/// contiguous framebuffer, so an absurd value asks the firmware for a mode it
/// will refuse (or, worse, one that will not fit VRAM).
pub fn parse_dims(s: &str) -> Option<(u32, u32)> {
    let (w, h) = s.split_once(['x', 'X'])?;
    let w: u32 = w.trim().parse().ok()?;
    let h: u32 = h.trim().parse().ok()?;
    if w < 320 || h < 200 || w > 16384 || h > 16384 {
        return None;
    }
    Some((w, h))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a base block with `dtds` placed in the four timing slots and a
    /// correct trailing checksum.
    fn synth(dtds: &[[u8; DTD_LEN]]) -> alloc::vec::Vec<u8> {
        let mut e = alloc::vec![0u8; BASE_BLOCK_LEN];
        e[..8].copy_from_slice(&HEADER);
        for (i, d) in dtds.iter().enumerate().take(DTD_COUNT) {
            let at = DTD0 + i * DTD_LEN;
            e[at..at + DTD_LEN].copy_from_slice(d);
        }
        // Checksum byte makes the block sum to 0 mod 256.
        let sum = e[..BASE_BLOCK_LEN - 1].iter().fold(0u8, |a, &b| a.wrapping_add(b));
        e[BASE_BLOCK_LEN - 1] = 0u8.wrapping_sub(sum);
        e
    }

    /// A detailed timing descriptor for `w x h` with a nonzero pixel clock.
    fn dtd(w: u32, h: u32) -> [u8; DTD_LEN] {
        let mut d = [0u8; DTD_LEN];
        d[0] = 0x01; // nonzero pixel clock → this slot IS a timing
        d[2] = (w & 0xFF) as u8;
        d[4] = (((w >> 8) & 0x0F) << 4) as u8;
        d[5] = (h & 0xFF) as u8;
        d[7] = (((h >> 8) & 0x0F) << 4) as u8;
        d
    }

    #[test_case]
    fn preferred_resolution_reads_the_first_timing() {
        let e = synth(&[dtd(1920, 1080), dtd(1280, 720)]);
        assert_eq!(preferred_resolution(&e), Some((1920, 1080)));
    }

    #[test_case]
    fn preferred_resolution_handles_high_bits() {
        // These all have a zero or small low byte, so a parser that forgets the
        // packed high nibble reads them as 0 (or a tiny width).
        for &(w, h) in &[(2560u32, 1440u32), (3840, 2160), (1024, 768), (1280, 1024)] {
            let e = synth(&[dtd(w, h)]);
            assert_eq!(preferred_resolution(&e), Some((w, h)), "{}x{}", w, h);
        }
    }

    #[test_case]
    fn preferred_resolution_skips_display_descriptors() {
        // Slot 0 has a zero pixel clock (a monitor-name descriptor, not a
        // timing) → the first real timing wins.
        let mut name = [0u8; DTD_LEN];
        name[3] = 0xFC; // display product name tag
        let e = synth(&[name, dtd(1600, 900)]);
        assert_eq!(preferred_resolution(&e), Some((1600, 900)));
    }

    #[test_case]
    fn invalid_edid_says_nothing_rather_than_guessing() {
        // All zeros / all ones: an absent display or a platform with no EDID.
        assert_eq!(preferred_resolution(&[0u8; BASE_BLOCK_LEN]), None);
        assert_eq!(preferred_resolution(&[0xFFu8; BASE_BLOCK_LEN]), None);
        // Too short.
        assert_eq!(preferred_resolution(&[]), None);
        assert_eq!(preferred_resolution(&[0u8; 64]), None);
        // Right header, broken checksum → refused.
        let mut e = synth(&[dtd(1920, 1080)]);
        e[BASE_BLOCK_LEN - 1] = e[BASE_BLOCK_LEN - 1].wrapping_add(1);
        assert!(!is_valid(&e));
        assert_eq!(preferred_resolution(&e), None);
        // Right checksum, wrong header → refused.
        let mut e = synth(&[dtd(1920, 1080)]);
        e[0] = 0x12;
        let sum = e[..BASE_BLOCK_LEN - 1].iter().fold(0u8, |a, &b| a.wrapping_add(b));
        e[BASE_BLOCK_LEN - 1] = 0u8.wrapping_sub(sum);
        assert!(!is_valid(&e));
        assert_eq!(preferred_resolution(&e), None);
        // A valid EDID with no timings at all also says nothing.
        assert_eq!(preferred_resolution(&synth(&[])), None);
    }

    #[test_case]
    fn best_mode_prefers_exact_then_largest_that_fits() {
        let modes = [(0usize, 800u32, 600u32), (1, 1920, 1080), (2, 2560, 1440), (3, 1280, 720)];
        // Exact match wins even though a bigger mode exists.
        assert_eq!(best_mode_for((1920, 1080), modes), Some(1));
        // No exact match → largest that fits inside the panel.
        assert_eq!(best_mode_for((2000, 1200), modes), Some(1));
        assert_eq!(best_mode_for((1300, 800), modes), Some(3));
        assert_eq!(best_mode_for((1024, 768), modes), Some(0));
        // Nothing fits → leave the mode alone.
        assert_eq!(best_mode_for((640, 480), modes), None);
        assert_eq!(best_mode_for((1920, 1080), []), None);
    }

    /// A display descriptor (zero pixel clock) of `tag` carrying `text`.
    fn desc(tag: u8, text: &str) -> [u8; DTD_LEN] {
        let mut d = [0x20u8; DTD_LEN];
        d[0] = 0;
        d[1] = 0;
        d[2] = 0;
        d[3] = tag;
        d[4] = 0;
        let b = text.as_bytes();
        for i in 0..(DTD_LEN - 5) {
            d[5 + i] = if i < b.len() {
                b[i]
            } else if i == b.len() {
                0x0A
            } else {
                0x20
            };
        }
        d
    }

    /// Set the identity fields on a synthesised EDID and fix the checksum.
    fn with_identity(mut e: alloc::vec::Vec<u8>, mfg: &str, product: u16, serial: u32) -> alloc::vec::Vec<u8> {
        let b = mfg.as_bytes();
        let code = |c: u8| ((c - b'A' + 1) as u16) & 0x1F;
        let raw = (code(b[0]) << 10) | (code(b[1]) << 5) | code(b[2]);
        e[8..10].copy_from_slice(&raw.to_be_bytes());
        e[10..12].copy_from_slice(&product.to_le_bytes());
        e[12..16].copy_from_slice(&serial.to_le_bytes());
        let sum = e[..BASE_BLOCK_LEN - 1].iter().fold(0u8, |a, &b| a.wrapping_add(b));
        e[BASE_BLOCK_LEN - 1] = 0u8.wrapping_sub(sum);
        e
    }

    #[test_case]
    fn identity_unpacks_the_five_bit_manufacturer_letters() {
        let e = with_identity(synth(&[dtd(1920, 1080)]), "DEL", 0xA1B2, 0x00034F1C);
        let id = identity(&e).expect("valid edid");
        assert_eq!(&id.mfg, b"DEL", "manufacturer letters are 5-bit big-endian packed");
        assert_eq!(id.product, 0xA1B2);
        assert_eq!(id.serial, 0x00034F1C);
        assert_eq!(id.key(), "DEL-A1B2-00034F1C");
        // Another vendor gives a different key — the whole point.
        let e2 = with_identity(synth(&[dtd(2560, 1600)]), "APP", 0x1234, 0);
        let id2 = identity(&e2).unwrap();
        assert_eq!(id2.key(), "APP-1234-00000000");
        assert_ne!(id.key(), id2.key());
    }

    #[test_case]
    fn identity_needs_a_valid_edid() {
        assert!(identity(&[0u8; BASE_BLOCK_LEN]).is_none());
        assert!(identity(&[0xFFu8; BASE_BLOCK_LEN]).is_none());
        assert!(identity(&[]).is_none());
        // An out-of-range letter code degrades to '?' rather than a wild byte.
        let mut e = synth(&[dtd(1920, 1080)]);
        e[8] = 0x7F;
        e[9] = 0xFF;
        let sum = e[..BASE_BLOCK_LEN - 1].iter().fold(0u8, |a, &b| a.wrapping_add(b));
        e[BASE_BLOCK_LEN - 1] = 0u8.wrapping_sub(sum);
        let id = identity(&e).unwrap();
        assert!(id.mfg.iter().all(|&c| c.is_ascii_uppercase() || c == b'?'), "{:?}", id.mfg);
    }

    #[test_case]
    fn monitor_name_reads_the_product_name_descriptor() {
        let e = synth(&[dtd(1920, 1080), desc(0xFC, "DELL P2722HE")]);
        assert_eq!(monitor_name(&e).as_deref(), Some("DELL P2722HE"));
        // No name descriptor → None, not an empty string.
        assert_eq!(monitor_name(&synth(&[dtd(1920, 1080)])), None);
        // A different descriptor type is not mistaken for the name.
        assert_eq!(monitor_name(&synth(&[desc(0xFF, "SERIAL123")])), None);
        // Control bytes in a garbled descriptor are dropped.
        let mut d = desc(0xFC, "OK");
        d[7] = 0x01;
        let got = monitor_name(&synth(&[d])).unwrap_or_default();
        assert!(got.chars().all(|c| c.is_ascii_graphic() || c == ' '), "{got:?}");
    }

    #[test_case]
    fn pick_output_prefers_the_firmware_console_display() {
        let lid = |c, e| OutputInfo { console_out: c, edid_native: e };
        // A laptop panel plus an attached monitor: the console-out one wins even
        // when it is listed second and the other is bigger.
        let outs = [lid(false, Some((3840, 2160))), lid(true, Some((1920, 1080)))];
        assert_eq!(pick_output(&outs), Some(1));
        // Console-out with no EDID loses to console-out with one.
        let outs = [lid(true, None), lid(true, Some((1920, 1080)))];
        assert_eq!(pick_output(&outs), Some(1));
        // No console-out marker anywhere → the connected display (has EDID) wins,
        // not the dark port listed first.
        let outs = [lid(false, None), lid(false, Some((2560, 1440)))];
        assert_eq!(pick_output(&outs), Some(1));
        // Nothing known at all → the first output, so we still get a console.
        let outs = [lid(false, None), lid(false, None)];
        assert_eq!(pick_output(&outs), Some(0));
        // A single output is always the answer, whatever it reports.
        assert_eq!(pick_output(&[lid(false, None)]), Some(0));
        assert_eq!(pick_output(&[]), None);
    }

    #[test_case]
    fn implausibly_small_only_flags_unchosen_defaults() {
        assert!(is_implausibly_small(800, 600));
        assert!(is_implausibly_small(640, 480));
        assert!(!is_implausibly_small(1024, 768));
        assert!(!is_implausibly_small(1280, 800));
        assert!(!is_implausibly_small(1920, 1080));
    }

    #[test_case]
    fn boot_cfg_reads_a_resolution_and_tolerates_the_rest() {
        assert_eq!(parse_boot_cfg("resolution=1920x1080\n"), Some((1920, 1080)));
        // Comments, blank lines, whitespace and unknown keys are all survivable —
        // this file gets hand-edited on the ESP of a machine that will not boot.
        let cfg = "# ChittiOS display\n\n  resolution = 1280x720  # comment\nscale=2\n";
        assert_eq!(parse_boot_cfg(cfg), Some((1280, 720)));
        // Last valid wins, so a line can be appended to override.
        assert_eq!(parse_boot_cfg("resolution=800x600\nresolution=1920x1200\n"), Some((1920, 1200)));
        // A malformed value keeps the firmware's mode rather than guessing.
        assert_eq!(parse_boot_cfg("resolution=lots\n"), None);
        assert_eq!(parse_boot_cfg("resolution=\n"), None);
        assert_eq!(parse_boot_cfg(""), None);
        assert_eq!(parse_boot_cfg("scale=2\n"), None);
        // A bad line does not discard a good one, whichever order they come in.
        assert_eq!(parse_boot_cfg("resolution=1600x900\nresolution=nonsense\n"), Some((1600, 900)));
    }

    #[test_case]
    fn parse_dims_rejects_what_cannot_be_a_framebuffer() {
        assert_eq!(parse_dims("1920x1080"), Some((1920, 1080)));
        assert_eq!(parse_dims("1920X1080"), Some((1920, 1080)));
        assert_eq!(parse_dims("320x200"), Some((320, 200)));
        // Below any console's floor, and past what VRAM would hold.
        assert_eq!(parse_dims("319x200"), None);
        assert_eq!(parse_dims("320x199"), None);
        assert_eq!(parse_dims("99999x1080"), None);
        // Not two numbers at all.
        assert_eq!(parse_dims("1920"), None);
        assert_eq!(parse_dims("x1080"), None);
        assert_eq!(parse_dims("-1920x1080"), None);
        assert_eq!(parse_dims("1920x1080x60"), None);
    }
}
