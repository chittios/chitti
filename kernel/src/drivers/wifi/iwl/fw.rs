//! **Intel WiFi firmware images and device identification** — the pure half of
//! `iwlwifi`.
//!
//! Every Intel WiFi part refuses to do anything until the host has loaded a firmware
//! image into it, and *which* image depends on the chip family and on a firmware API
//! version that Linux discovers by trying filenames from the newest backwards. So before
//! any register is touched, two questions have to be answered off-device: which family is
//! this, and is a usable `.ucode` for it present?
//!
//! Both are pure functions here, and unit-tested — which is the only verification
//! available: QEMU emulates no Intel WiFi part, so unlike the Ethernet drivers there is
//! not even an emulated device to try. Getting the identification wrong would mean
//! loading an AX210 image into an AX200, which the device answers by failing its own
//! signature check.
//!
//! ## The image format
//!
//! A modern `.ucode` is a TLV container: an 88-byte header (a zero word, the `IWL\n`
//! magic, a 64-byte human-readable version, then version/build words) followed by
//! type/length/value records, each padded to a 4-byte boundary. The sections the device
//! actually needs are among those records; everything else is skipped **by length**,
//! which is what lets one parser read images from firmware releases newer than this code.
//!
//! ## What this is not
//!
//! Identification and image parsing only. Nothing here brings a radio up: that needs the
//! device's register interface, its command rings, the firmware handshake, and then a
//! 802.11 state machine plus WPA2 — the same staging the Broadcom driver is going
//! through. `iwl::device` is where that starts; this file is what it will load.

use alloc::string::String;
use alloc::vec::Vec;

/// PCI vendor id for Intel.
pub const VENDOR_INTEL: u16 = 0x8086;

/// Chip families, which decide the firmware image and the register interface.
///
/// Coarser than Linux's `iwl_cfg` table on purpose: the distinctions that matter before
/// a device is touched are the firmware name and the broad generation. Anything finer is
/// a property of the running device, not of its PCI id.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Family {
    /// 7260/7265 — the first generation this could plausibly support.
    Iwl7000,
    /// 8260/8265.
    Iwl8000,
    /// 9260/9461/9560.
    Iwl9000,
    /// AX200/AX201 (the `cc-a0` images).
    Ax200,
    /// AX210/AX211/AX411 (`ty-a0-gf-a0` and relatives).
    Ax210,
    /// BE200 — Wi-Fi 7.
    Be200,
}

impl Family {
    /// The firmware filename stem Linux looks for, without the API version or extension.
    ///
    /// The full name is `<stem>-<api>.ucode`, and the API number is *not* derivable from
    /// the hardware — Linux tries the highest it knows and counts down until a file
    /// exists. [`firmware_candidates`] does the same.
    pub fn firmware_stem(&self) -> &'static str {
        match self {
            Family::Iwl7000 => "iwlwifi-7260",
            Family::Iwl8000 => "iwlwifi-8265",
            Family::Iwl9000 => "iwlwifi-9260",
            Family::Ax200 => "iwlwifi-cc-a0",
            Family::Ax210 => "iwlwifi-ty-a0-gf-a0",
            Family::Be200 => "iwlwifi-gl-c0-fm-c0",
        }
    }

    /// A short human name for logs.
    pub fn label(&self) -> &'static str {
        match self {
            Family::Iwl7000 => "7260/7265",
            Family::Iwl8000 => "8260/8265",
            Family::Iwl9000 => "9260/9560",
            Family::Ax200 => "AX200/AX201",
            Family::Ax210 => "AX210/AX211",
            Family::Be200 => "BE200",
        }
    }

    /// Highest firmware API version to look for. Linux counts down from its maximum;
    /// starting too high merely costs a few missing-file checks, starting too low means
    /// never finding a current image.
    pub fn max_api(&self) -> u32 {
        match self {
            Family::Iwl7000 => 17,
            Family::Iwl8000 => 36,
            Family::Iwl9000 => 46,
            Family::Ax200 | Family::Ax210 | Family::Be200 => 89,
        }
    }

    /// Lowest API version worth trying.
    pub fn min_api(&self) -> u32 {
        match self {
            Family::Iwl7000 => 12,
            Family::Iwl8000 => 22,
            Family::Iwl9000 => 30,
            Family::Ax200 | Family::Ax210 | Family::Be200 => 50,
        }
    }
}

// PCI device ids, grouped by family. Deliberately a table rather than range checks:
// Intel's ids are not ordered by generation, and a range would silently claim a future
// part whose register interface differs.
const IDS_7000: &[u16] = &[0x08b1, 0x08b2, 0x08b3, 0x08b4, 0x095a, 0x095b];
const IDS_8000: &[u16] = &[0x24f3, 0x24f4, 0x24fd, 0x24f5, 0x24f6];
const IDS_9000: &[u16] = &[
    0x2526, 0x271b, 0x271c, 0x30dc, 0x31dc, 0x9df0, 0xa370, 0x9df1,
];
const IDS_AX200: &[u16] = &[0x2723, 0x02f0, 0x4df0, 0x34f0, 0xa0f0, 0x43f0, 0x3df0];
const IDS_AX210: &[u16] = &[0x2725, 0x7a70, 0x7af0, 0x51f0, 0x54f0, 0x7e40, 0x2726];
const IDS_BE200: &[u16] = &[0x272b];

/// Which family a PCI id belongs to, or `None` for anything not recognised.
///
/// `None` is the important answer: an unknown Intel WiFi id must **not** fall back to a
/// nearby family. Broadcom taught this the hard way in the Ethernet dispatcher — all
/// Intel Ethernet reports the same class, and driving one family through another's
/// register map produces a device that links and never receives. WiFi is worse, because
/// the wrong firmware fails a signature check inside the device with no useful error.
pub fn family_for(vendor: u16, device: u16) -> Option<Family> {
    if vendor != VENDOR_INTEL {
        return None;
    }
    for (ids, fam) in [
        (IDS_7000, Family::Iwl7000),
        (IDS_8000, Family::Iwl8000),
        (IDS_9000, Family::Iwl9000),
        (IDS_AX200, Family::Ax200),
        (IDS_AX210, Family::Ax210),
        (IDS_BE200, Family::Be200),
    ] {
        if ids.contains(&device) {
            return Some(fam);
        }
    }
    None
}

/// Firmware filenames to try, newest API first.
///
/// The API version is a property of the *image*, not the chip, so this enumerates rather
/// than computes — exactly as Linux does.
pub fn firmware_candidates(f: Family) -> Vec<String> {
    let stem = f.firmware_stem();
    (f.min_api()..=f.max_api())
        .rev()
        .map(|api| alloc::format!("{stem}-{api}.ucode"))
        .collect()
}

// --- the .ucode TLV container --------------------------------------------

/// `IWL\n`, little-endian, at offset 4 of a TLV-format image.
pub const UCODE_MAGIC: u32 = 0x0a4c_5749;

/// Bytes before the first TLV: a zero word, the magic, 64 bytes of version text, then
/// version, build and an ignored quadword.
pub const UCODE_HEADER_LEN: usize = 88;

/// Offset of the human-readable version string.
const OFF_HUMAN: usize = 8;
/// Length of that string field.
const HUMAN_LEN: usize = 64;

// TLV record types this code names. Everything else is skipped by length, which is what
// lets one parser read images from releases newer than itself.
/// Runtime firmware section (secure-boot images).
pub const TLV_SEC_RT: u32 = 19;
/// Init-phase section.
pub const TLV_SEC_INIT: u32 = 20;
/// Default calibration set.
pub const TLV_DEF_CALIB: u32 = 22;
/// PHY SKU descriptor.
pub const TLV_PHY_SKU: u32 = 23;
/// Firmware API bitmap.
pub const TLV_API_CHANGES_SET: u32 = 29;
/// Firmware capability bitmap.
pub const TLV_ENABLED_CAPABILITIES: u32 = 30;
/// A version string record.
pub const TLV_FW_VERSION: u32 = 36;
/// The command-version table: which version of each command *this* firmware speaks.
pub const TLV_CMD_VERSIONS: u32 = 48;

/// One entry of the command-version table: four bytes, `cmd`, `group`, `cmd_ver`,
/// `notif_ver`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CmdVersion {
    pub cmd: u8,
    pub group: u8,
    /// Version of the command's *request* structure.
    pub cmd_ver: u8,
    /// Version of its notification/response structure.
    pub notif_ver: u8,
}

/// Bytes per entry. The record's length divided by this is the entry count — a length that
/// is not a multiple of four is a record this parser does not understand, not one to read
/// as far as it goes.
pub const CMD_VERSION_ENTRY_LEN: usize = 4;

/// Parse the command-version table.
///
/// This is the table that makes it *safe* to add configuration commands later. Their
/// request structures differ between firmware API versions — the same command id takes a
/// different layout in different releases — and the firmware image says which one it
/// expects. So a driver can either check, or send a plausible structure the firmware reads
/// as something else. There is no third option and no error path from the device.
pub fn parse_cmd_versions(payload: &[u8]) -> Option<Vec<CmdVersion>> {
    if payload.is_empty() || payload.len() % CMD_VERSION_ENTRY_LEN != 0 {
        return None;
    }
    Some(
        payload
            .chunks_exact(CMD_VERSION_ENTRY_LEN)
            .map(|c| CmdVersion {
                cmd: c[0],
                group: c[1],
                cmd_ver: c[2],
                notif_ver: c[3],
            })
            .collect(),
    )
}

/// One record in the container: its type and the byte range of its payload.
///
/// A range rather than a copy, so parsing a multi-megabyte image allocates a handful of
/// descriptors instead of duplicating it — the kernel allocator punishes churn, and a
/// firmware image is exactly the size that makes copying it expensive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Section {
    pub kind: u32,
    pub range: core::ops::Range<usize>,
}

/// A parsed firmware image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FirmwareImage {
    /// The human-readable version the header carries, trimmed at its first NUL.
    pub version: String,
    /// Version and build words from the header.
    pub ver: u32,
    pub build: u32,
    /// Every record, in file order.
    pub sections: Vec<Section>,
}

impl FirmwareImage {
    /// The payload range of the first record of `kind`.
    pub fn section(&self, kind: u32) -> Option<core::ops::Range<usize>> {
        self.sections
            .iter()
            .find(|s| s.kind == kind)
            .map(|s| s.range.clone())
    }

    /// Whether the image carries anything that could be loaded as runtime firmware.
    ///
    /// An image whose records are all metadata is well-formed and useless, and finding
    /// that out here is much better than after the device has been reset.
    pub fn has_runtime_section(&self) -> bool {
        self.sections.iter().any(|s| s.kind == TLV_SEC_RT)
    }

    /// The command-version table, if the image carries one, parsed from `blob`.
    ///
    /// Takes the bytes the image was parsed from because [`Section`] holds ranges rather
    /// than copies — the image is megabytes and the allocator punishes churn.
    pub fn cmd_versions(&self, blob: &[u8]) -> Option<Vec<CmdVersion>> {
        let r = self.section(TLV_CMD_VERSIONS)?;
        parse_cmd_versions(blob.get(r)?)
    }

    /// Which version of `(group, cmd)` this firmware speaks, if it says.
    ///
    /// `None` means the table is absent or silent about this command — which is not the same
    /// as version 0, and callers must treat it as "unknown" rather than assume the layout
    /// they happen to implement.
    pub fn cmd_version(&self, blob: &[u8], group: u8, cmd: u8) -> Option<CmdVersion> {
        self.cmd_versions(blob)?
            .into_iter()
            .find(|v| v.group == group && v.cmd == cmd)
    }
}

fn le32(d: &[u8], at: usize) -> Option<u32> {
    let b = d.get(at..at + 4)?;
    Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

/// Parse a `.ucode` image.
///
/// `None` when it is not a TLV-format Intel image or is malformed. Three checks decide
/// that, and each rejects a real mistake: the leading word must be **zero** (a non-zero
/// value there is one of the pre-TLV formats, which this cannot read and must not
/// misinterpret), the magic must be `IWL\n` (so a truncated download or an entirely
/// different file is refused rather than fed to the device), and every record's declared
/// length must fit inside the file.
pub fn parse_image(d: &[u8]) -> Option<FirmwareImage> {
    if d.len() < UCODE_HEADER_LEN {
        return None;
    }
    if le32(d, 0)? != 0 {
        return None; // a pre-TLV format
    }
    if le32(d, 4)? != UCODE_MAGIC {
        return None;
    }
    let human = &d[OFF_HUMAN..OFF_HUMAN + HUMAN_LEN];
    let end = human.iter().position(|&c| c == 0).unwrap_or(HUMAN_LEN);
    let mut version = String::new();
    for &c in &human[..end] {
        // The field is documented as printable ASCII; anything else is a corrupt
        // download, and substituting a placeholder keeps the string safe to log.
        version.push(if c.is_ascii_graphic() || c == b' ' {
            c as char
        } else {
            '?'
        });
    }
    let ver = le32(d, OFF_HUMAN + HUMAN_LEN)?;
    let build = le32(d, OFF_HUMAN + HUMAN_LEN + 4)?;

    let mut sections = Vec::new();
    let mut i = UCODE_HEADER_LEN;
    while i + 8 <= d.len() {
        let kind = le32(d, i)?;
        let len = le32(d, i + 4)? as usize;
        let start = i + 8;
        let end = start.checked_add(len)?;
        if end > d.len() {
            return None; // a record claiming more than the file holds
        }
        sections.push(Section {
            kind,
            range: start..end,
        });
        // Records are padded to a 4-byte boundary; without rounding up, one odd-length
        // record misaligns every record after it and the walk reads garbage types.
        let advance = 8 + ((len + 3) & !3);
        i = i.checked_add(advance)?;
        if advance == 8 && len == 0 {
            // A zero-length record is legal, but a stream of them would not terminate
            // if the advance were ever computed as zero. Bounded by construction here;
            // this guard documents the intent.
            continue;
        }
    }
    Some(FirmwareImage {
        version,
        ver,
        build,
        sections,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal TLV image with the given records.
    fn image(version: &str, records: &[(u32, &[u8])]) -> Vec<u8> {
        let mut d = alloc::vec![0u8; UCODE_HEADER_LEN];
        d[4..8].copy_from_slice(&UCODE_MAGIC.to_le_bytes());
        let v = version.as_bytes();
        d[OFF_HUMAN..OFF_HUMAN + v.len()].copy_from_slice(v);
        d[OFF_HUMAN + HUMAN_LEN..OFF_HUMAN + HUMAN_LEN + 4].copy_from_slice(&7u32.to_le_bytes());
        d[OFF_HUMAN + HUMAN_LEN + 4..OFF_HUMAN + HUMAN_LEN + 8]
            .copy_from_slice(&99u32.to_le_bytes());
        for (kind, payload) in records {
            d.extend_from_slice(&kind.to_le_bytes());
            d.extend_from_slice(&(payload.len() as u32).to_le_bytes());
            d.extend_from_slice(payload);
            while d.len() % 4 != 0 {
                d.push(0); // records are 4-byte aligned
            }
        }
        d
    }

    #[test_case]
    fn the_firmware_says_which_version_of_each_command_it_speaks() {
        // This table is what makes adding a configuration command safe rather than a guess:
        // the same command id takes a different request structure in different firmware
        // releases, and the image itself says which. Four bytes per entry — `cmd`, `group`,
        // `cmd_ver`, `notif_ver` — from Linux's `iwl_fw_cmd_version`.
        let table: &[u8] = &[
            0x02, 0x0c, 0x04, 0x00, // NVM_GET_INFO in the regulatory group, request v4
            0x0d, 0x01, 0x0e, 0x02, // SCAN_REQ_UMAC in the long group, request v14
        ];
        let d = image(
            "v",
            &[(TLV_SEC_RT, &[1, 2, 3, 4]), (TLV_CMD_VERSIONS, table)],
        );
        let img = parse_image(&d).unwrap();
        let v = img.cmd_versions(&d).expect("the table must parse");
        assert_eq!(v.len(), 2);
        assert_eq!(
            img.cmd_version(&d, 0x0c, 0x02),
            Some(CmdVersion {
                cmd: 0x02,
                group: 0x0c,
                cmd_ver: 4,
                notif_ver: 0
            })
        );
        assert_eq!(img.cmd_version(&d, 0x01, 0x0d).unwrap().cmd_ver, 14);

        // Silence about a command is not version 0 — a caller must be able to tell "this
        // firmware wants v0" from "it did not say", or it will send the layout it happens to
        // implement and be read as something else.
        assert_eq!(
            img.cmd_version(&d, 0x01, 0x02),
            None,
            "cmd/group were matched crosswise"
        );
        let bare = image("v", &[(TLV_SEC_RT, &[1])]);
        let bare_img = parse_image(&bare).unwrap();
        assert!(bare_img.cmd_versions(&bare).is_none());
        assert!(bare_img.cmd_version(&bare, 0x0c, 0x02).is_none());

        // A length that is not a whole number of entries is a record this parser does not
        // understand, not one to read as far as it goes.
        assert!(parse_cmd_versions(&[0x02, 0x0c, 0x04]).is_none());
        assert!(parse_cmd_versions(&[]).is_none());
        assert_eq!(parse_cmd_versions(table).unwrap().len(), 2);
    }

    #[test_case]
    fn parses_a_tlv_image_and_finds_its_sections() {
        let d = image(
            "iwlwifi-cc-a0-77",
            &[(TLV_SEC_RT, &[1, 2, 3, 4]), (TLV_PHY_SKU, &[9])],
        );
        let img = parse_image(&d).unwrap();
        assert_eq!(img.version, "iwlwifi-cc-a0-77");
        assert_eq!((img.ver, img.build), (7, 99));
        assert_eq!(img.sections.len(), 2);
        let rt = img.section(TLV_SEC_RT).unwrap();
        assert_eq!(&d[rt], &[1, 2, 3, 4]);
        assert!(img.has_runtime_section());
    }

    #[test_case]
    fn odd_length_records_are_padded_to_four_bytes() {
        // The trap: a 1-byte record followed by another. Without rounding the advance up,
        // the walk lands mid-record and reads a length out of payload bytes — which then
        // either overruns the file or invents hundreds of records.
        let d = image("v", &[(TLV_PHY_SKU, &[0xaa]), (TLV_SEC_RT, &[0xbb, 0xcc])]);
        let img = parse_image(&d).unwrap();
        assert_eq!(img.sections.len(), 2);
        assert_eq!(img.sections[0].kind, TLV_PHY_SKU);
        assert_eq!(img.sections[1].kind, TLV_SEC_RT);
        assert_eq!(&d[img.section(TLV_SEC_RT).unwrap()], &[0xbb, 0xcc]);
    }

    #[test_case]
    fn a_pre_tlv_image_is_refused_rather_than_misread() {
        // Older formats put a version number where the TLV format has a zero. Reading one
        // as TLV would walk its body as records; refusing is the only safe answer, since
        // this cannot load them.
        let mut d = image("v", &[(TLV_SEC_RT, &[1])]);
        d[0..4].copy_from_slice(&3u32.to_le_bytes());
        assert!(parse_image(&d).is_none());
    }

    #[test_case]
    fn a_wrong_magic_or_truncated_file_is_refused() {
        let mut d = image("v", &[(TLV_SEC_RT, &[1])]);
        d[4..8].copy_from_slice(&0xdead_beefu32.to_le_bytes());
        assert!(parse_image(&d).is_none(), "wrong magic accepted");

        // A partial download: the header survives but a record claims more than is there.
        let mut d = image("v", &[(TLV_SEC_RT, &[1, 2, 3, 4])]);
        let n = d.len();
        d.truncate(n - 2);
        assert!(parse_image(&d).is_none(), "truncated record accepted");

        assert!(parse_image(&[]).is_none());
        assert!(parse_image(&alloc::vec![0u8; UCODE_HEADER_LEN - 1]).is_none());
    }

    #[test_case]
    fn an_image_with_no_runtime_section_is_well_formed_but_unusable() {
        // Worth distinguishing: this parses cleanly, so the only thing that stops it
        // being loaded into a freshly reset device is asking the question.
        let d = image("meta only", &[(TLV_API_CHANGES_SET, &[0, 0, 0, 1])]);
        let img = parse_image(&d).unwrap();
        assert!(!img.has_runtime_section());
        assert!(img.section(TLV_SEC_RT).is_none());
    }

    #[test_case]
    fn a_corrupt_version_string_stays_safe_to_log() {
        let mut d = image("ok", &[(TLV_SEC_RT, &[1])]);
        d[OFF_HUMAN] = 0x01; // not printable
        let img = parse_image(&d).unwrap();
        assert_eq!(img.version, "?k");
    }

    #[test_case]
    fn known_intel_parts_map_to_their_families() {
        assert_eq!(family_for(VENDOR_INTEL, 0x2723), Some(Family::Ax200));
        assert_eq!(family_for(VENDOR_INTEL, 0x2725), Some(Family::Ax210));
        assert_eq!(family_for(VENDOR_INTEL, 0x24fd), Some(Family::Iwl8000));
        assert_eq!(family_for(VENDOR_INTEL, 0x2526), Some(Family::Iwl9000));
        assert_eq!(family_for(VENDOR_INTEL, 0x08b1), Some(Family::Iwl7000));
        assert_eq!(family_for(VENDOR_INTEL, 0x272b), Some(Family::Be200));
    }

    #[test_case]
    fn an_unknown_id_is_refused_not_guessed_into_a_nearby_family() {
        // The Ethernet dispatcher's lesson, and worse here: the wrong firmware fails a
        // signature check *inside* the device, with no error the host can read.
        assert_eq!(family_for(VENDOR_INTEL, 0xffff), None);
        assert_eq!(
            family_for(VENDOR_INTEL, 0x2724),
            None,
            "adjacent id must not be claimed"
        );
        // And another vendor's device is never ours, whatever its id.
        assert_eq!(family_for(0x10ec, 0x2723), None);
    }

    #[test_case]
    fn no_device_id_belongs_to_two_families() {
        // A duplicate would make the family depend on table order, which is exactly the
        // kind of thing that works until the tables are reordered.
        let all = [
            IDS_7000, IDS_8000, IDS_9000, IDS_AX200, IDS_AX210, IDS_BE200,
        ];
        for (i, a) in all.iter().enumerate() {
            for b in all.iter().skip(i + 1) {
                for id in a.iter() {
                    assert!(!b.contains(id), "device {id:#06x} is in two families");
                }
            }
        }
    }

    #[test_case]
    fn iwlwifi_stems_match_the_kernel() {
        // `cargo xtask iwlwifi-assets` duplicates this table as data, because xtask is a
        // host binary and cannot link the kernel. This pins the kernel side so a change
        // here fails visibly and points at the other copy — it cannot check xtask itself,
        // and saying so is more useful than implying it does.
        let expect = [
            (Family::Ax200, "iwlwifi-cc-a0", 50u32, 89u32),
            (Family::Ax210, "iwlwifi-ty-a0-gf-a0", 50, 89),
            (Family::Be200, "iwlwifi-gl-c0-fm-c0", 50, 89),
            (Family::Iwl9000, "iwlwifi-9260", 30, 46),
            (Family::Iwl8000, "iwlwifi-8265", 22, 36),
            (Family::Iwl7000, "iwlwifi-7260", 12, 17),
        ];
        for (f, stem, min, max) in expect {
            assert_eq!(f.firmware_stem(), stem, "{} stem changed", f.label());
            assert_eq!(f.min_api(), min, "{} min api changed", f.label());
            assert_eq!(f.max_api(), max, "{} max api changed", f.label());
        }
    }

    #[test_case]
    fn firmware_candidates_run_newest_first_and_are_bounded() {
        let c = firmware_candidates(Family::Ax200);
        assert_eq!(c.first().unwrap(), "iwlwifi-cc-a0-89.ucode");
        assert_eq!(c.last().unwrap(), "iwlwifi-cc-a0-50.ucode");
        assert_eq!(c.len(), 40);
        // Every family produces a non-empty, ordered list.
        for f in [
            Family::Iwl7000,
            Family::Iwl8000,
            Family::Iwl9000,
            Family::Ax200,
            Family::Ax210,
            Family::Be200,
        ] {
            let c = firmware_candidates(f);
            assert!(!c.is_empty(), "{} has no candidates", f.label());
            assert!(f.min_api() <= f.max_api());
            assert!(c[0].starts_with(f.firmware_stem()));
        }
    }
}
