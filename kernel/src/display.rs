//! Display configuration: the offerable mode list, the **runtime** logical
//! desktop, and the **next-boot** physical mode preference.
//!
//! Two mechanisms, because they answer different questions:
//!
//! * **Logical desktop** (`logical`) — the compositor lays out against this size
//!   and blits it as a centred viewport inside the physical framebuffer, letterboxed.
//!   It applies *immediately* and identically on both arches, and text stays crisp
//!   because glyphs are still rasterised at physical pixels (nothing is scaled).
//!   This is what "change the resolution" can mean at runtime at all: once the
//!   loader has exited boot services there is no GOP left to call, so the
//!   hardware mode is fixed for the life of the boot.
//! * **Boot mode** (`boot_mode`) — a preference the *loader* applies on the next
//!   boot, which is the only way to actually change the hardware mode and use
//!   every physical pixel. Costs a reboot.
//!
//! The geometry and parsing live here, pure and unit-tested, so the fiddly parts
//! (clamping, centring, config round-tripping) are covered off-hardware.

use alloc::string::String;
use alloc::vec::Vec;

/// Smallest logical desktop that still fits the chrome (frames, status bar, a
/// usable composer). Below this the UI is unusable rather than merely small.
pub const MIN_LOGICAL_W: u32 = 640;
pub const MIN_LOGICAL_H: u32 = 400;

/// Desktop sizes offered by `/display list`, largest first.
///
/// A fixed table rather than a probe: after boot services are gone there is no
/// mode list to enumerate, and these are logical viewport sizes, not hardware
/// modes — any size that fits inside the framebuffer is valid. The table exists
/// only so the common choices are one command away; `/display set` accepts an
/// arbitrary `WxH` too.
pub const STANDARD_MODES: &[(u32, u32)] = &[
    (3840, 2160),
    (3440, 1440),
    (2560, 1600),
    (2560, 1440),
    (1920, 1200),
    (1920, 1080),
    (1680, 1050),
    (1600, 900),
    (1440, 900),
    (1366, 768),
    (1280, 800),
    (1280, 720),
    (1024, 768),
];

/// The logical sizes selectable on a `phys`-sized framebuffer: the panel's own
/// size ("native") first, then every standard mode that fits strictly inside it.
///
/// A mode larger than the framebuffer is not offered — the viewport is a window
/// *into* the framebuffer, so there is nothing to show outside it.
pub fn modes_for(phys: (u32, u32)) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    if phys.0 >= MIN_LOGICAL_W && phys.1 >= MIN_LOGICAL_H {
        out.push(phys);
    }
    for &m in STANDARD_MODES {
        if m == phys {
            continue; // already listed as native
        }
        if m.0 <= phys.0 && m.1 <= phys.1 && m.0 >= MIN_LOGICAL_W && m.1 >= MIN_LOGICAL_H {
            out.push(m);
        }
    }
    out
}

/// Clamp a requested logical desktop into what the framebuffer can host.
///
/// Never returns a size larger than `phys` (there are no pixels there) nor
/// smaller than the usable minimum — and if the framebuffer itself is below the
/// minimum, `phys` wins, because showing a cramped UI beats showing none.
pub fn clamp_logical(phys: (u32, u32), want: (u32, u32)) -> (u32, u32) {
    let w = want.0.clamp(MIN_LOGICAL_W.min(phys.0), phys.0.max(1));
    let h = want.1.clamp(MIN_LOGICAL_H.min(phys.1), phys.1.max(1));
    (w, h)
}

/// The centred viewport `(x, y, w, h)` for a logical desktop inside `phys`.
///
/// Centring (rather than top-left) puts the letterbox evenly around the desktop,
/// which is what every display does when it pillarboxes a smaller mode.
pub fn viewport(phys: (u32, u32), logical: (u32, u32)) -> (u32, u32, u32, u32) {
    let (w, h) = clamp_logical(phys, logical);
    ((phys.0 - w) / 2, (phys.1 - h) / 2, w, h)
}

/// Largest font scale offered (cells are `8*scale` x `16*scale` pixels).
pub const MAX_FONT_SCALE: u64 = 4;

/// The automatic font scale for a desktop `height` pixels tall.
///
/// Thresholds rather than a division, because the old formula
/// (`(height + 550) / 1100`) needed **1650** px before it reached scale 2 — so a
/// 2560x1440 panel rendered at scale 1, i.e. 8x16 px cells giving 320 columns.
/// That is not "dense", it is unreadable, and it was the real reason a 2K display
/// looked broken. A 1440p screen wants scale 2; 4K wants 3.
///
/// Only the *automatic* choice; an explicit `font_scale` in `ui.json` (or
/// `/display scale <n>`) always wins.
pub fn auto_font_scale(height: u64) -> u64 {
    match height {
        0..=1199 => 1,  // 720p / 800p / 1080p
        1200..=1799 => 2, // 1200p / 1440p / 1600p
        1800..=2599 => 3, // 2160p (4K)
        _ => 4,           // 5K+
    }
}

/// Clamp a requested font scale; `0` means "automatic".
pub fn clamp_font_scale(n: u64) -> u64 {
    if n == 0 {
        0
    } else {
        n.clamp(1, MAX_FONT_SCALE)
    }
}

/// Parse `WIDTHxHEIGHT` (also accepts `X`, and spaces around either number).
pub fn parse_wxh(s: &str) -> Option<(u32, u32)> {
    let (a, b) = s.trim().split_once(['x', 'X'])?;
    let w: u32 = a.trim().parse().ok()?;
    let h: u32 = b.trim().parse().ok()?;
    (w > 0 && h > 0).then_some((w, h))
}

/// Format a mode as `WIDTHxHEIGHT`.
pub fn format_wxh(m: (u32, u32)) -> String {
    alloc::format!("{}x{}", m.0, m.1)
}

/// The active display's EDID base block, captured by the loader before boot
/// services went away and handed over in the boot-info page.
static EDID: crate::mm::Locked<Option<Vec<u8>>> = crate::mm::Locked::new(None);

/// Record the loader-supplied EDID for the active display (called once at boot).
pub fn set_edid(bytes: &[u8]) {
    if crate::edid::is_valid(bytes) {
        EDID.with(|e| *e = Some(bytes[..crate::edid::BASE_BLOCK_LEN].to_vec()));
    }
}

/// The active display's EDID, if the firmware published one.
pub fn edid_bytes() -> Option<Vec<u8>> {
    EDID.with(|e| e.clone())
}

/// The settings-profile key for a display — what per-monitor settings are stored
/// under, the way GNOME's `monitors.xml` keys per-output configuration.
///
/// Prefers the display's own EDID identity (vendor/product/serial), which is
/// stable across reboots and distinct per panel. With no EDID — a hypervisor, or
/// the x86/Limine path where the loader does not pass one — it falls back to the
/// framebuffer geometry, so two monitors of *different* size still get separate
/// profiles even though they can't be told apart by name.
pub fn profile_key(edid: Option<&[u8]>, phys: Option<(u32, u32)>) -> String {
    if let Some(id) = edid.and_then(crate::edid::identity) {
        return id.key();
    }
    match phys {
        Some((w, h)) => alloc::format!("fb-{w}x{h}"),
        None => String::from("default"),
    }
}

/// A human label for the active display: its EDID product name, else its size.
pub fn profile_name(edid: Option<&[u8]>, phys: Option<(u32, u32)>) -> String {
    if let Some(n) = edid.and_then(crate::edid::monitor_name) {
        return n;
    }
    match phys {
        Some(p) => alloc::format!("display {}", format_wxh(p)),
        None => String::from("display"),
    }
}

/// The persisted display settings (`/configs/core/display.json`).
#[derive(Clone, PartialEq, Debug, Default)]
pub struct DisplayCfg {
    /// Runtime logical desktop. `None` = native, i.e. use the whole framebuffer.
    ///
    /// `None` rather than "the current physical size" on purpose: a saved config
    /// outlives the monitor it was written on, and "follow the panel" stays right
    /// when a different display is attached.
    pub logical: Option<(u32, u32)>,
    /// Physical mode for the loader to set on the next boot. `None` = auto, i.e.
    /// let the display's EDID decide (see [`crate::edid`]).
    pub boot_mode: Option<(u32, u32)>,
    /// Font scale; `0` = automatic from the desktop height
    /// ([`auto_font_scale`]). This is the knob that actually answers "everything
    /// is too small on a high-resolution screen" — a smaller *desktop* only
    /// letterboxes, it does not enlarge anything.
    pub font_scale: u64,
}

impl DisplayCfg {
    /// Read from parsed JSON. Unknown/garbage values fall back to the default
    /// (native / auto) rather than failing, so a hand-edited file can't brick the
    /// console — the whole point of this config is to be recoverable.
    pub fn from_json(j: &crate::json::Json) -> DisplayCfg {
        let mode = |key: &str| -> Option<(u32, u32)> {
            let v = j.get(key)?;
            let s = v.as_str()?;
            let t = s.trim();
            // "native" / "auto" / "" all mean "no preference".
            if t.is_empty() || t.eq_ignore_ascii_case("native") || t.eq_ignore_ascii_case("auto") {
                return None;
            }
            parse_wxh(t)
        };
        let font_scale = j
            .get("font_scale")
            .and_then(|v| v.as_i64())
            .map(|n| clamp_font_scale(n.max(0) as u64))
            .unwrap_or(0);
        DisplayCfg { logical: mode("logical"), boot_mode: mode("boot_mode"), font_scale }
    }

    /// Serialize to the on-disk JSON text.
    pub fn to_json_string(&self) -> String {
        let s = |m: Option<(u32, u32)>, dflt: &str| -> String {
            m.map(format_wxh).unwrap_or_else(|| String::from(dflt))
        };
        alloc::format!(
            "{{\n  \"logical\": \"{}\",\n  \"boot_mode\": \"{}\",\n  \"font_scale\": {}\n}}\n",
            s(self.logical, "native"),
            s(self.boot_mode, "auto"),
            self.font_scale
        )
    }
}

/// The whole display settings file: a **per-display** profile map plus the
/// global next-boot mode.
///
/// Per-display because that is what makes the settings actually right on a
/// machine with more than one screen — the same model as GNOME's `monitors.xml`.
/// Plug in a 1080p monitor and it gets its own scale and desktop; go back to the
/// laptop panel and it gets its own. A single global profile is wrong the moment
/// two displays differ, which is exactly the case that started this.
#[derive(Clone, PartialEq, Debug, Default)]
pub struct DisplaySettings {
    /// `(profile key, profile)` pairs. A `Vec` rather than a map so the file
    /// round-trips in a stable order.
    pub displays: Vec<(String, DisplayCfg)>,
    /// The loader hint, which is not per-display: whichever screen the firmware
    /// brings up, there is only one mode request to make.
    pub boot_mode: Option<(u32, u32)>,
}

impl DisplaySettings {
    /// Parse the settings file.
    ///
    /// Accepts the **older flat shape** (`{logical, font_scale, boot_mode}`) and
    /// adopts it as `key`'s profile, so an existing config keeps working and is
    /// migrated on the next save instead of being silently discarded.
    pub fn from_json(j: &crate::json::Json, key: &str) -> DisplaySettings {
        let mut out = DisplaySettings {
            displays: Vec::new(),
            boot_mode: DisplayCfg::from_json(j).boot_mode,
        };
        match j.get("displays").and_then(|d| d.as_object()) {
            Some(entries) => {
                for (k, v) in entries {
                    out.displays.push((k.clone(), DisplayCfg::from_json(v)));
                }
            }
            None => {
                // Legacy flat file: keep it, attributed to the display in use.
                let flat = DisplayCfg::from_json(j);
                if flat.logical.is_some() || flat.font_scale != 0 {
                    out.displays.push((String::from(key), flat));
                }
            }
        }
        out
    }

    /// The profile for `key`, or defaults if this display hasn't been configured.
    pub fn profile(&self, key: &str) -> DisplayCfg {
        self.displays
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, p)| p.clone())
            .unwrap_or_default()
    }

    /// Store `profile` for `key`, replacing any existing entry.
    pub fn set_profile(&mut self, key: &str, profile: DisplayCfg) {
        match self.displays.iter_mut().find(|(k, _)| k == key) {
            Some(slot) => slot.1 = profile,
            None => self.displays.push((String::from(key), profile)),
        }
    }

    /// Serialize to the on-disk JSON text.
    pub fn to_json_string(&self) -> String {
        let mut s = String::from("{\n  \"boot_mode\": \"");
        s.push_str(&self.boot_mode.map(format_wxh).unwrap_or_else(|| String::from("auto")));
        s.push_str("\",\n  \"displays\": {");
        for (i, (k, p)) in self.displays.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str(&alloc::format!(
                "\n    \"{}\": {{ \"logical\": \"{}\", \"font_scale\": {} }}",
                k,
                p.logical.map(format_wxh).unwrap_or_else(|| String::from("native")),
                p.font_scale
            ));
        }
        s.push_str(if self.displays.is_empty() { "}\n}\n" } else { "\n  }\n}\n" });
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn modes_for_lists_native_first_then_smaller() {
        let m = modes_for((2560, 1440));
        assert_eq!(m[0], (2560, 1440), "native first");
        assert!(m.contains(&(1920, 1080)));
        assert!(m.contains(&(1280, 720)));
        // Nothing bigger than the framebuffer in either axis.
        assert!(!m.contains(&(3840, 2160)));
        assert!(!m.contains(&(2560, 1600)), "taller than the panel");
        // Native is not duplicated.
        assert_eq!(m.iter().filter(|&&x| x == (2560, 1440)).count(), 1);
        // Descending by width.
        for i in 1..m.len() {
            assert!(m[i].0 <= m[i - 1].0, "not descending at {i}: {m:?}");
        }
    }

    #[test_case]
    fn modes_for_exact_standard_panel_has_no_duplicate() {
        let m = modes_for((1920, 1080));
        assert_eq!(m[0], (1920, 1080));
        assert_eq!(m.iter().filter(|&&x| x == (1920, 1080)).count(), 1);
        assert!(m.contains(&(1280, 720)));
        assert!(!m.contains(&(1920, 1200)));
    }

    #[test_case]
    fn modes_for_tiny_framebuffer_offers_nothing_unusable() {
        // Below the usable minimum: no options at all rather than a broken UI.
        assert!(modes_for((320, 200)).is_empty());
        // Just at the minimum: native only.
        assert_eq!(modes_for((640, 400)), alloc::vec![(640, 400)]);
    }

    #[test_case]
    fn clamp_logical_never_exceeds_the_framebuffer() {
        assert_eq!(clamp_logical((1920, 1080), (3840, 2160)), (1920, 1080));
        assert_eq!(clamp_logical((1920, 1080), (1280, 720)), (1280, 720));
        // Absurdly small requests are floored to the usable minimum.
        assert_eq!(clamp_logical((1920, 1080), (1, 1)), (MIN_LOGICAL_W, MIN_LOGICAL_H));
        assert_eq!(clamp_logical((1920, 1080), (0, 0)), (MIN_LOGICAL_W, MIN_LOGICAL_H));
        // A framebuffer smaller than the minimum still yields itself, not zero.
        assert_eq!(clamp_logical((320, 200), (1280, 720)), (320, 200));
        assert_eq!(clamp_logical((320, 200), (1, 1)), (320, 200));
    }

    #[test_case]
    fn viewport_is_centred_and_inside_the_framebuffer() {
        assert_eq!(viewport((1920, 1080), (1920, 1080)), (0, 0, 1920, 1080));
        assert_eq!(viewport((2560, 1440), (1920, 1080)), (320, 180, 1920, 1080));
        // Odd leftovers split down (never negative, never past the edge).
        let (x, y, w, h) = viewport((1921, 1081), (1920, 1080));
        assert_eq!((x, y, w, h), (0, 0, 1920, 1080));
        // Whatever the request, the viewport fits.
        for &phys in &[(800u32, 600u32), (1366, 768), (1920, 1080), (3840, 2160)] {
            for &want in &[(640u32, 400u32), (1280, 720), (1920, 1080), (9999, 9999)] {
                let (x, y, w, h) = viewport(phys, want);
                assert!(x + w <= phys.0, "phys={phys:?} want={want:?} overflows x");
                assert!(y + h <= phys.1, "phys={phys:?} want={want:?} overflows y");
                assert!(w > 0 && h > 0);
            }
        }
    }

    #[test_case]
    fn wxh_round_trips() {
        assert_eq!(parse_wxh("1920x1080"), Some((1920, 1080)));
        assert_eq!(parse_wxh("1920X1080"), Some((1920, 1080)));
        assert_eq!(parse_wxh(" 1280 x 720 "), Some((1280, 720)));
        assert_eq!(parse_wxh("1920"), None);
        assert_eq!(parse_wxh("1920x"), None);
        assert_eq!(parse_wxh("x1080"), None);
        assert_eq!(parse_wxh("0x1080"), None, "zero is not a size");
        assert_eq!(parse_wxh("1920x0"), None);
        assert_eq!(parse_wxh("axb"), None);
        assert_eq!(parse_wxh(""), None);
        for &m in STANDARD_MODES {
            assert_eq!(parse_wxh(&format_wxh(m)), Some(m));
        }
    }

    #[test_case]
    fn auto_font_scale_makes_high_res_readable() {
        // The bug this replaced: the old `(h + 550) / 1100` needed 1650px for
        // scale 2, so a 1440p panel rendered 8x16 cells — 320 columns.
        assert_eq!(auto_font_scale(1440), 2, "a 2K panel must not render at scale 1");
        assert_eq!(auto_font_scale(720), 1);
        assert_eq!(auto_font_scale(1000), 1);
        assert_eq!(auto_font_scale(1080), 1);
        assert_eq!(auto_font_scale(1200), 2);
        assert_eq!(auto_font_scale(1600), 2);
        assert_eq!(auto_font_scale(2160), 3, "4K");
        assert_eq!(auto_font_scale(2880), 4);
        // Never zero (that would mean a zero-size cell), and monotonic.
        let mut prev = 0;
        for h in (0..4000).step_by(40) {
            let s = auto_font_scale(h);
            assert!(s >= 1 && s <= MAX_FONT_SCALE, "h={h} → {s}");
            assert!(s >= prev, "not monotonic at h={h}");
            prev = s;
        }
    }

    #[test_case]
    fn clamp_font_scale_keeps_zero_as_automatic() {
        assert_eq!(clamp_font_scale(0), 0, "0 means automatic, not 1");
        assert_eq!(clamp_font_scale(1), 1);
        assert_eq!(clamp_font_scale(4), 4);
        assert_eq!(clamp_font_scale(99), MAX_FONT_SCALE);
    }

    #[test_case]
    fn config_round_trips_through_json() {
        let cfg = DisplayCfg { logical: Some((1920, 1080)), boot_mode: Some((2560, 1440)), font_scale: 2 };
        let text = cfg.to_json_string();
        let back = DisplayCfg::from_json(&crate::json::Json::parse(&text).unwrap());
        assert_eq!(back, cfg);
        // Defaults serialize to the readable sentinels and parse back as None.
        let dflt = DisplayCfg::default();
        let text = dflt.to_json_string();
        assert!(text.contains("\"native\"") && text.contains("\"auto\""), "{text}");
        let back = DisplayCfg::from_json(&crate::json::Json::parse(&text).unwrap());
        assert_eq!(back, dflt);
    }

    #[test_case]
    fn settings_keep_a_profile_per_display() {
        let mut s = DisplaySettings::default();
        s.set_profile("DEL-A1B2-00034F1C", DisplayCfg { logical: None, boot_mode: None, font_scale: 2 });
        s.set_profile("APP-1234-00000000", DisplayCfg { logical: Some((1280, 800)), boot_mode: None, font_scale: 3 });
        s.boot_mode = Some((1920, 1080));
        // Each display gets its own answer; an unknown one gets defaults.
        assert_eq!(s.profile("DEL-A1B2-00034F1C").font_scale, 2);
        assert_eq!(s.profile("APP-1234-00000000").logical, Some((1280, 800)));
        assert_eq!(s.profile("nope"), DisplayCfg::default());
        // Re-setting replaces rather than duplicating.
        s.set_profile("DEL-A1B2-00034F1C", DisplayCfg { logical: None, boot_mode: None, font_scale: 4 });
        assert_eq!(s.displays.len(), 2);
        assert_eq!(s.profile("DEL-A1B2-00034F1C").font_scale, 4);
        // Round-trips, and one display's change does not disturb the other.
        let text = s.to_json_string();
        let back = DisplaySettings::from_json(&crate::json::Json::parse(&text).expect(&text), "x");
        assert_eq!(back.boot_mode, Some((1920, 1080)));
        assert_eq!(back.profile("APP-1234-00000000").font_scale, 3);
        assert_eq!(back.profile("DEL-A1B2-00034F1C").font_scale, 4);
    }

    #[test_case]
    fn settings_migrate_the_old_flat_file() {
        // The pre-per-display shape must be adopted for the display in use, not
        // silently dropped.
        let j = crate::json::Json::parse(
            "{\"logical\": \"1600x900\", \"font_scale\": 3, \"boot_mode\": \"1920x1080\"}",
        )
        .unwrap();
        let s = DisplaySettings::from_json(&j, "DEL-A1B2-00034F1C");
        assert_eq!(s.boot_mode, Some((1920, 1080)));
        let p = s.profile("DEL-A1B2-00034F1C");
        assert_eq!(p.logical, Some((1600, 900)));
        assert_eq!(p.font_scale, 3);
        // A flat file with nothing set adds no profile at all.
        let j = crate::json::Json::parse("{\"boot_mode\": \"auto\"}").unwrap();
        assert!(DisplaySettings::from_json(&j, "k").displays.is_empty());
        // An empty settings object round-trips.
        let text = DisplaySettings::default().to_json_string();
        let back = DisplaySettings::from_json(&crate::json::Json::parse(&text).expect(&text), "k");
        assert_eq!(back, DisplaySettings::default());
    }

    #[test_case]
    fn profile_key_prefers_edid_identity_then_geometry() {
        let e = with_edid_identity("DEL", 0xA1B2, 0x00034F1C);
        // EDID identity wins, and is independent of the current resolution.
        assert_eq!(profile_key(Some(&e), Some((1920, 1080))), "DEL-A1B2-00034F1C");
        assert_eq!(profile_key(Some(&e), None), "DEL-A1B2-00034F1C");
        // No EDID (a hypervisor, or the x86 loader) → keyed by size, so two
        // differently-sized monitors still get separate profiles.
        assert_eq!(profile_key(None, Some((1920, 1080))), "fb-1920x1080");
        assert_ne!(profile_key(None, Some((2560, 1440))), profile_key(None, Some((1920, 1080))));
        // Nothing known at all still yields a usable key.
        assert_eq!(profile_key(None, None), "default");
        // Invalid EDID is treated as absent, never as an identity of garbage.
        assert_eq!(profile_key(Some(&[0u8; 128]), Some((800, 600))), "fb-800x600");
    }

    #[test_case]
    fn profile_name_prefers_the_monitor_name() {
        assert_eq!(profile_name(None, Some((1920, 1080))), "display 1920x1080");
        assert_eq!(profile_name(None, None), "display");
        assert_eq!(profile_name(Some(&[0u8; 128]), Some((640, 480))), "display 640x480");
    }

    /// A valid EDID carrying just an identity (no timings needed here).
    fn with_edid_identity(mfg: &str, product: u16, serial: u32) -> alloc::vec::Vec<u8> {
        let mut e = alloc::vec![0u8; crate::edid::BASE_BLOCK_LEN];
        e[..8].copy_from_slice(&[0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00]);
        let b = mfg.as_bytes();
        let code = |c: u8| ((c - b'A' + 1) as u16) & 0x1F;
        let raw = (code(b[0]) << 10) | (code(b[1]) << 5) | code(b[2]);
        e[8..10].copy_from_slice(&raw.to_be_bytes());
        e[10..12].copy_from_slice(&product.to_le_bytes());
        e[12..16].copy_from_slice(&serial.to_le_bytes());
        let sum = e[..127].iter().fold(0u8, |a, &b| a.wrapping_add(b));
        e[127] = 0u8.wrapping_sub(sum);
        e
    }

    #[test_case]
    fn config_survives_a_hand_edited_file() {
        // Garbage, wrong types, and missing keys all mean "no preference" — a bad
        // edit must not be able to leave the console unusable.
        for text in [
            "{}",
            "{\"logical\": \"potato\"}",
            "{\"logical\": \"\", \"boot_mode\": \"  \"}",
            "{\"logical\": 1080}",
            "{\"logical\": null, \"boot_mode\": \"NATIVE\"}",
            "{\"boot_mode\": \"Auto\"}",
        ] {
            let j = crate::json::Json::parse(text).expect(text);
            assert_eq!(DisplayCfg::from_json(&j), DisplayCfg::default(), "{text}");
        }
        // A valid value alongside a bad one is still honoured.
        let j = crate::json::Json::parse("{\"logical\": \"1600x900\", \"boot_mode\": \"junk\"}").unwrap();
        assert_eq!(
            DisplayCfg::from_json(&j),
            DisplayCfg { logical: Some((1600, 900)), boot_mode: None, font_scale: 0 }
        );
        // An out-of-range font scale clamps rather than producing a giant cell.
        let j = crate::json::Json::parse("{\"font_scale\": 99}").unwrap();
        assert_eq!(DisplayCfg::from_json(&j).font_scale, MAX_FONT_SCALE);
    }
}
