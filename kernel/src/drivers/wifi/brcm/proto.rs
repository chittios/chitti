//! Pure **brcmfmac / FullMAC** wire helpers: ioctl command codes, message
//! header layouts, and SSID parsing. Side-effect-free so unit tests run under
//! `cargo xtask test` (x86, no Apple hardware).

#![allow(dead_code)]

/// Common brcmfmac ioctl / dongle commands (from Linux `brcmu_wifi.h` /
/// `brcmfmac` — the subset we need for scan/connect).
pub const BRCMF_C_GET_VERSION: u32 = 1;
pub const BRCMF_C_UP: u32 = 2;
pub const BRCMF_C_DOWN: u32 = 3;
pub const BRCMF_C_SET_PROMISC: u32 = 10;
pub const BRCMF_C_GET_RATE: u32 = 12;
pub const BRCMF_C_GET_INFRA: u32 = 19;
pub const BRCMF_C_SET_INFRA: u32 = 20;
pub const BRCMF_C_GET_AUTH: u32 = 21;
pub const BRCMF_C_SET_AUTH: u32 = 22;
pub const BRCMF_C_GET_BSSID: u32 = 23;
pub const BRCMF_C_GET_SSID: u32 = 25;
pub const BRCMF_C_SET_SSID: u32 = 26;
pub const BRCMF_C_TERMINATED: u32 = 28;
pub const BRCMF_C_GET_CHANNEL: u32 = 29;
pub const BRCMF_C_SET_CHANNEL: u32 = 30;
pub const BRCMF_C_GET_SRL: u32 = 31;
pub const BRCMF_C_GET_LRL: u32 = 33;
pub const BRCMF_C_GET_RADIO: u32 = 37;
pub const BRCMF_C_SET_RADIO: u32 = 38;
pub const BRCMF_C_GET_PHYTYPE: u32 = 39;
pub const BRCMF_C_SCAN: u32 = 50;
pub const BRCMF_C_SCAN_RESULTS: u32 = 51;
pub const BRCMF_C_DISASSOC: u32 = 52;
pub const BRCMF_C_REASSOC: u32 = 53;
pub const BRCMF_C_SET_ROAM_TRIGGER: u32 = 55;
pub const BRCMF_C_SET_ROAM_DELTA: u32 = 57;
pub const BRCMF_C_GET_DTIMPRD: u32 = 89;
pub const BRCMF_C_SET_COUNTRY: u32 = 98;
pub const BRCMF_C_GET_PM: u32 = 49;
pub const BRCMF_C_SET_PM: u32 = 49; // same cmd, set via length
pub const BRCMF_C_SET_WSEC: u32 = 133;
pub const BRCMF_C_GET_WSEC: u32 = 133;
pub const BRCMF_C_SET_WPA_AUTH: u32 = 164;
pub const BRCMF_C_GET_WPA_AUTH: u32 = 165;
pub const BRCMF_C_SET_SCB_TIMEOUT: u32 = 28; // placeholder — real value differs per tree

/// WPA auth modes (`WPA_AUTH_*` in brcmfmac).
pub const WPA_AUTH_DISABLED: u32 = 0x0000;
pub const WPA_AUTH_NONE: u32 = 0x0001;
pub const WPA_AUTH_UNSPECIFIED: u32 = 0x0002;
pub const WPA_AUTH_PSK: u32 = 0x0004;
pub const WPA2_AUTH_UNSPECIFIED: u32 = 0x0040;
pub const WPA2_AUTH_PSK: u32 = 0x0080;

/// `wsec` bit flags.
pub const WSEC_NONE: u32 = 0;
pub const WSEC_WEP: u32 = 0x0001;
pub const WSEC_TKIP: u32 = 0x0002;
pub const WSEC_AES: u32 = 0x0004;

/// Msg types on the common control rings (m1n1 `trace_wlan.py`).
pub const MSG_TYPE_IOCTLPTR_REQ: u8 = 0x09;
pub const MSG_TYPE_IOCTLPTR_REQ_ACK: u8 = 0x0a;
pub const MSG_TYPE_IOCTL_RESP: u8 = 0x0c;
pub const MSG_TYPE_EVENT: u8 = 0x0e;
pub const MSG_TYPE_H2D_MAILBOX_DATA: u8 = 0x23;
pub const MSG_TYPE_D2H_MAILBOX_DATA: u8 = 0x24;

/// 802.11 max SSID length.
pub const DOT11_MAX_SSID_LEN: usize = 32;

/// A scanned BSS the driver reports to `/wifi scan`.
#[derive(Clone, Debug)]
pub struct BssInfo {
    pub ssid: alloc::string::String,
    pub bssid: [u8; 6],
    pub channel: u16,
    pub rssi: i16,
    pub privacy: bool,
}

/// Pack a `wl_ssid` (len le32 + up to 32 bytes). Pure.
pub fn pack_ssid(ssid: &str, out: &mut [u8]) -> Option<usize> {
    let b = ssid.as_bytes();
    if b.len() > DOT11_MAX_SSID_LEN || out.len() < 4 + b.len() {
        return None;
    }
    out[0..4].copy_from_slice(&(b.len() as u32).to_le_bytes());
    out[4..4 + b.len()].copy_from_slice(b);
    Some(4 + b.len())
}

/// Unpack a `wl_ssid` into a UTF-8-lossy String. Pure.
pub fn unpack_ssid(buf: &[u8]) -> Option<alloc::string::String> {
    if buf.len() < 4 {
        return None;
    }
    let n = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
    if n > DOT11_MAX_SSID_LEN || 4 + n > buf.len() {
        return None;
    }
    Some(alloc::string::String::from_utf8_lossy(&buf[4..4 + n]).into_owned())
}

/// Encode a little-endian MsgHeader (8 bytes). Pure.
pub fn pack_msg_header(msg_type: u8, if_id: i8, flags: u8, epoch: u8, request_id: u32) -> [u8; 8] {
    let mut h = [0u8; 8];
    h[0] = msg_type;
    h[1] = if_id as u8;
    h[2] = flags;
    h[3] = epoch;
    h[4..8].copy_from_slice(&request_id.to_le_bytes());
    h
}

/// Decode a MsgHeader. Pure.
pub fn unpack_msg_header(h: &[u8]) -> Option<(u8, i8, u8, u8, u32)> {
    if h.len() < 8 {
        return None;
    }
    let rid = u32::from_le_bytes([h[4], h[5], h[6], h[7]]);
    Some((h[0], h[1] as i8, h[2], h[3], rid))
}

/// Encode an IOCTL pointer request payload (after MsgHeader). Pure.
/// Layout: cmd u32, trans_id u16, input_len u16, output_len u16, rsvd[3] u16,
/// host_input_buf_addr u64.
pub fn pack_ioctl_ptr_req(
    cmd: u32,
    trans_id: u16,
    input_len: u16,
    output_len: u16,
    host_buf: u64,
) -> [u8; 24] {
    let mut p = [0u8; 24];
    p[0..4].copy_from_slice(&cmd.to_le_bytes());
    p[4..6].copy_from_slice(&trans_id.to_le_bytes());
    p[6..8].copy_from_slice(&input_len.to_le_bytes());
    p[8..10].copy_from_slice(&output_len.to_le_bytes());
    // rsvd 10..16 already zero
    p[16..24].copy_from_slice(&host_buf.to_le_bytes());
    p
}

/// Apple board-type → firmware basename stem (without path). Pure.
///
/// j473 Mac mini M2 (`pci14e4,4434`) chipcommon CHIPID is **0x4388** (BAR0
/// +0x3000). Asahi firmware names:
/// - `brcmfmac4388-pcie.apple,miyake.bin` (matches chipcommon)
/// - also accept `brcmfmac4387c2-pcie.apple,miyake.bin` (older trees / PCI naming)
pub fn firmware_stem(board_type: &str, chip_id: u16) -> alloc::string::String {
    use alloc::format;
    let chip: &str = match chip_id {
        // PCI device id if ever passed instead of chipcommon.
        0x4434 | 0x4387 => "4387c2",
        // Live chipcommon on j473 miyake (0x10044388 → chip 0x4388, rev 4).
        0x4388 => "4388",
        0x4378 => "4378b1",
        0x4377 => "4377b3",
        other => {
            return if board_type.is_empty() {
                format!("brcmfmac{other:04x}-pcie")
            } else {
                format!("brcmfmac{other:04x}-pcie.{board_type}")
            };
        }
    };
    if board_type.is_empty() {
        format!("brcmfmac{chip}-pcie")
    } else {
        format!("brcmfmac{chip}-pcie.{board_type}")
    }
}

// ── Dongle firmware download geometry (Linux brcmfmac/pcie.c + chip.c) ──

/// Magic `SMAR` at firmware offset [`RAMSIZE_OFFSET`] — embeds dongle RAM size.
pub const RAMSIZE_MAGIC: u32 = 0x534d_4152; // 'SMAR'
/// Offset of the optional ramsize magic + size pair inside the `.bin`.
pub const RAMSIZE_OFFSET: usize = 0x6c;
/// Shared-protocol version window (Linux `BRCMF_PCIE_MIN/MAX_SHARED_VERSION`).
pub const SHARED_VERSION_MIN: u8 = 5;
pub const SHARED_VERSION_MAX: u8 = 7;
pub const SHARED_VERSION_MASK: u32 = 0x00ff;
/// SI enum / chipcommon default base (AXI AI chips).
pub const SI_ENUM_BASE: u32 = 0x1800_0000;
/// Chipcommon `eromptr` register offset.
pub const CC_EROMPTR: u32 = 0xfc;
/// Chipcommon `watchdog` register offset.
pub const CC_WATCHDOG: u32 = 0x80;

/// Dongle TCM / sysmem base for a chipcommon chip id. Pure table from
/// `brcmf_chip_tcm_rambase`. `0x4388` (j473 chipcommon) shares the 4387 base.
pub fn rambase_for_chip(chip_id: u16) -> Option<u32> {
    Some(match chip_id {
        0x4377 => 0x17_0000,
        0x4378 => 0x35_2000,
        // BCM4387 + Apple chipcommon 0x4388 (same die family).
        0x4387 | 0x4388 => 0x74_0000,
        // Older FullMAC still listed for completeness.
        0x4364 => 0x16_0000,
        0x4355 | 0x4359 => 0x16_0000,
        0x4365 | 0x4366 => 0x20_0000,
        0x4350 | 0x4354 | 0x4356 | 0x4358 | 0x4360 | 0x4362 | 0x4371 => 0x18_0000,
        _ => return None,
    })
}

/// Soft upper bound used only as a sanity cap (not the primary size source).
/// Download prefers a tight `fw+nv+4` pack — the old 0x280000 default placed
/// NVRAM past mapped TCM on j473 and external-aborted.
pub fn default_ramsize_hint(chip_id: u16) -> u32 {
    match chip_id {
        0x4387 | 0x4388 => 0x26_0000,
        0x4378 => 0x20_0000,
        0x4377 => 0x18_0000,
        _ => 0x20_0000,
    }
}

/// Parse optional ramsize from firmware (`SMAR` at [`RAMSIZE_OFFSET`]). Pure.
pub fn fw_embedded_ramsize(data: &[u8]) -> Option<u32> {
    if data.len() < RAMSIZE_OFFSET + 8 {
        return None;
    }
    let magic = u32::from_le_bytes([
        data[RAMSIZE_OFFSET],
        data[RAMSIZE_OFFSET + 1],
        data[RAMSIZE_OFFSET + 2],
        data[RAMSIZE_OFFSET + 3],
    ]);
    if magic != RAMSIZE_MAGIC {
        return None;
    }
    let size = u32::from_le_bytes([
        data[RAMSIZE_OFFSET + 4],
        data[RAMSIZE_OFFSET + 5],
        data[RAMSIZE_OFFSET + 6],
        data[RAMSIZE_OFFSET + 7],
    ]);
    if size == 0 || size > 16 * 1024 * 1024 {
        return None;
    }
    Some(size)
}

/// Reset vector written into TCM[0] before releasing the ARM — first LE word
/// of the firmware image (Linux `get_unaligned_le32(fw->data)`).
pub fn fw_reset_vector(data: &[u8]) -> Option<u32> {
    if data.len() < 4 {
        return None;
    }
    Some(u32::from_le_bytes([data[0], data[1], data[2], data[3]]))
}

/// Shared-RAM pointer the firmware posts at `rambase + ramsize - 4` after init.
pub fn shared_addr_valid(addr: u32, rambase: u32, ramsize: u32) -> bool {
    if ramsize < 4 {
        return false;
    }
    addr >= rambase && addr < rambase.saturating_add(ramsize)
}

/// Decode shared-info version from the first word at the shared address.
pub fn shared_version(flags: u32) -> u8 {
    (flags & SHARED_VERSION_MASK) as u8
}

pub fn shared_version_ok(version: u8) -> bool {
    (SHARED_VERSION_MIN..=SHARED_VERSION_MAX).contains(&version)
}

/// Build firmware search paths for a stem (with Asahi alternate names). Pure.
pub fn firmware_search_paths(stem: &str) -> alloc::vec::Vec<alloc::string::String> {
    use alloc::{format, string::String, vec::Vec};
    let mut stems: Vec<String> = Vec::new();
    stems.push(String::from(stem));
    // Always try the j473 / miyake Asahi names as fallbacks.
    for alt in [
        "brcmfmac4388-pcie.apple,miyake",
        "brcmfmac4387c2-pcie.apple,miyake",
        "brcmfmac4388-pcie",
        "brcmfmac4387c2-pcie",
    ] {
        if !stems.iter().any(|s| s == alt) {
            stems.push(String::from(alt));
        }
    }
    let mut out = Vec::new();
    for s in &stems {
        out.push(format!("/brcm/{s}.bin"));
        out.push(format!("/firmware/brcm/{s}.bin"));
        out.push(format!("/vendorfw/firmware/brcm/{s}.bin"));
    }
    out
}

/// NVRAM path siblings for a firmware `.bin` path (optional). Pure.
pub fn nvram_paths_for_fw(fw_path: &str) -> alloc::vec::Vec<alloc::string::String> {
    use alloc::{format, string::String, vec::Vec};
    let mut out = Vec::new();
    if let Some(stem) = fw_path.strip_suffix(".bin") {
        out.push(format!("{stem}.txt"));
        // board-stripped: brcmfmac4388-pcie.apple,miyake → brcmfmac4388-pcie.txt
        if let Some(dot) = stem.rfind(".apple,") {
            out.push(format!("{}.txt", &stem[..dot]));
        }
    }
    // Also generic board nvram next to the file.
    if let Some(slash) = fw_path.rfind('/') {
        let dir = &fw_path[..slash];
        out.push(format!("{dir}/brcmfmac4388-pcie.apple,miyake.txt"));
        out.push(format!("{dir}/brcmfmac4387c2-pcie.apple,miyake.txt"));
    }
    // Dedup while preserving order.
    let mut dedup = Vec::new();
    for p in out {
        if !dedup.contains(&p) {
            dedup.push(p);
        }
    }
    let _ = String::new();
    dedup
}

// ── PL-368 DMP EROM (pure walker; hardware supplies the word reader) ──

/// One AXI core from the EROM table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EromCore {
    pub id: u16,
    pub rev: u8,
    pub base: u32,
    pub wrap: u32,
}

// DMP descriptor bits (Linux chip.c).
const DMP_DESC_TYPE_MSK: u32 = 0x0000_000f;
const DMP_DESC_VALID: u32 = 0x0000_0001;
const DMP_DESC_EMPTY: u32 = 0x0000_0000;
const DMP_DESC_COMPONENT: u32 = 0x0000_0001;
const DMP_DESC_MASTER_PORT: u32 = 0x0000_0003;
const DMP_DESC_ADDRESS: u32 = 0x0000_0005;
const DMP_DESC_ADDRSIZE_GT32: u32 = 0x0000_0008;
const DMP_DESC_EOT: u32 = 0x0000_000f;
const DMP_COMP_PARTNUM: u32 = 0x000f_ff00;
const DMP_COMP_PARTNUM_S: u32 = 8;
const DMP_COMP_REVISION: u32 = 0xff00_0000;
const DMP_COMP_REVISION_S: u32 = 24;
const DMP_COMP_NUM_SWRAP: u32 = 0x00f8_0000;
const DMP_COMP_NUM_SWRAP_S: u32 = 19;
const DMP_COMP_NUM_MWRAP: u32 = 0x0007_c000;
const DMP_COMP_NUM_MWRAP_S: u32 = 14;
const DMP_SLAVE_ADDR_BASE: u32 = 0xffff_f000;
const DMP_SLAVE_TYPE: u32 = 0x0000_00c0;
const DMP_SLAVE_TYPE_S: u32 = 6;
const DMP_SLAVE_TYPE_SLAVE: u32 = 0;
const DMP_SLAVE_TYPE_SWRAP: u32 = 2;
const DMP_SLAVE_TYPE_MWRAP: u32 = 3;
const DMP_SLAVE_SIZE_TYPE: u32 = 0x0000_0030;
const DMP_SLAVE_SIZE_TYPE_S: u32 = 4;
const DMP_SLAVE_SIZE_4K: u32 = 0;
const DMP_SLAVE_SIZE_8K: u32 = 1;
const DMP_SLAVE_SIZE_DESC: u32 = 3;

/// BCMA core part numbers we care about for download.
pub const BCMA_CORE_CHIPCOMMON: u16 = 0x800;
pub const BCMA_CORE_ARM_CR4: u16 = 0x83e;
pub const BCMA_CORE_ARM_CA7: u16 = 0x847;
pub const BCMA_CORE_PCIE2: u16 = 0x83c;
pub const BCMA_CORE_80211: u16 = 0x812;
pub const BCMA_CORE_SYS_MEM: u16 = 0x849;
pub const BCMA_CORE_INTERNAL_MEM: u16 = 0x80e;

/// Walk a PL-368 EROM, calling `read32(addr)` for each word. Pure relative to
/// the reader. Caps at `max_cores` to bound work.
pub fn erom_scan(mut read32: impl FnMut(u32) -> u32, erom_base: u32, max_cores: usize) -> alloc::vec::Vec<EromCore> {
    use alloc::vec::Vec;
    let mut cores = Vec::new();
    let mut erom = erom_base;
    let mut steps = 0u32;
    const MAX_STEPS: u32 = 4096; // defensive bound on descriptor walks

    while cores.len() < max_cores && steps < MAX_STEPS {
        steps += 1;
        let val = read32(erom);
        erom = erom.wrapping_add(4);
        if val & DMP_DESC_VALID == 0 {
            continue;
        }
        let mut desc_type = val & DMP_DESC_TYPE_MSK;
        if desc_type == DMP_DESC_EMPTY {
            continue;
        }
        if desc_type == DMP_DESC_EOT {
            break;
        }
        if desc_type != DMP_DESC_COMPONENT {
            continue;
        }
        let id = ((val & DMP_COMP_PARTNUM) >> DMP_COMP_PARTNUM_S) as u16;

        // Second component descriptor carries wrap counts + rev.
        let val2 = read32(erom);
        erom = erom.wrapping_add(4);
        steps += 1;
        if (val2 & DMP_DESC_TYPE_MSK) != DMP_DESC_COMPONENT {
            break;
        }
        let nmw = (val2 & DMP_COMP_NUM_MWRAP) >> DMP_COMP_NUM_MWRAP_S;
        let nsw = (val2 & DMP_COMP_NUM_SWRAP) >> DMP_COMP_NUM_SWRAP_S;
        let rev = ((val2 & DMP_COMP_REVISION) >> DMP_COMP_REVISION_S) as u8;
        if nmw + nsw == 0 && id != 0x827 && id != 0x840 {
            // no ports (except PMU/GCI) — skip
            continue;
        }

        // Parse register/wrap addresses for this component.
        let wraptype = {
            // Peek next descriptor type without consuming if we need to rewind.
            let peek = read32(erom);
            let peek_type = peek & DMP_DESC_TYPE_MSK;
            if peek_type == DMP_DESC_MASTER_PORT {
                erom = erom.wrapping_add(4);
                steps += 1;
                DMP_SLAVE_TYPE_MWRAP
            } else if peek_type == DMP_DESC_ADDRESS
                || (peek_type & !DMP_DESC_ADDRSIZE_GT32) == DMP_DESC_ADDRESS
            {
                DMP_SLAVE_TYPE_SWRAP
            } else {
                continue;
            }
        };

        let mut regbase = 0u32;
        let mut wrapbase = 0u32;
        loop {
            if steps >= MAX_STEPS {
                break;
            }
            let mut val = read32(erom);
            erom = erom.wrapping_add(4);
            steps += 1;
            let mut desc = val & DMP_DESC_TYPE_MSK;
            if (desc & !DMP_DESC_ADDRSIZE_GT32) == DMP_DESC_ADDRESS {
                desc = DMP_DESC_ADDRESS;
            }
            if desc == DMP_DESC_EOT {
                erom = erom.wrapping_sub(4);
                break;
            }
            if desc == DMP_DESC_COMPONENT {
                erom = erom.wrapping_sub(4);
                break;
            }
            if desc != DMP_DESC_ADDRESS {
                continue;
            }
            if val & DMP_DESC_ADDRSIZE_GT32 != 0 {
                let _ = read32(erom);
                erom = erom.wrapping_add(4);
                steps += 1;
            }
            let sztype = (val & DMP_SLAVE_SIZE_TYPE) >> DMP_SLAVE_SIZE_TYPE_S;
            if sztype == DMP_SLAVE_SIZE_DESC {
                let szdesc = read32(erom);
                erom = erom.wrapping_add(4);
                steps += 1;
                if szdesc & DMP_DESC_ADDRSIZE_GT32 != 0 {
                    let _ = read32(erom);
                    erom = erom.wrapping_add(4);
                    steps += 1;
                }
            }
            if sztype != DMP_SLAVE_SIZE_4K && sztype != DMP_SLAVE_SIZE_8K {
                continue;
            }
            let stype = (val & DMP_SLAVE_TYPE) >> DMP_SLAVE_TYPE_S;
            if regbase == 0 && stype == DMP_SLAVE_TYPE_SLAVE {
                regbase = val & DMP_SLAVE_ADDR_BASE;
            }
            if wrapbase == 0 && stype == wraptype {
                wrapbase = val & DMP_SLAVE_ADDR_BASE;
            }
            if regbase != 0 && wrapbase != 0 {
                break;
            }
        }

        if regbase != 0 {
            cores.push(EromCore {
                id,
                rev,
                base: regbase,
                wrap: wrapbase,
            });
        }
    }
    cores
}

/// Pick the ARM CPU core used to halt/run the dongle. Prefer CA7 then CR4.
pub fn find_arm_core(cores: &[EromCore]) -> Option<EromCore> {
    cores
        .iter()
        .copied()
        .find(|c| c.id == BCMA_CORE_ARM_CA7)
        .or_else(|| cores.iter().copied().find(|c| c.id == BCMA_CORE_ARM_CR4))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn ssid_roundtrip() {
        let mut buf = [0u8; 40];
        let n = pack_ssid("chitti-lan", &mut buf).unwrap();
        assert_eq!(n, 4 + 10);
        assert_eq!(unpack_ssid(&buf[..n]).unwrap(), "chitti-lan");
        assert!(pack_ssid(&"x".repeat(33), &mut buf).is_none());
    }

    #[test_case]
    fn msg_header_roundtrip() {
        let h = pack_msg_header(MSG_TYPE_IOCTLPTR_REQ, 0, 0, 1, 0xabcd_1234);
        let (t, ifid, f, e, rid) = unpack_msg_header(&h).unwrap();
        assert_eq!(t, MSG_TYPE_IOCTLPTR_REQ);
        assert_eq!(ifid, 0);
        assert_eq!(f, 0);
        assert_eq!(e, 1);
        assert_eq!(rid, 0xabcd_1234);
    }

    #[test_case]
    fn ioctl_ptr_layout() {
        let p = pack_ioctl_ptr_req(BRCMF_C_GET_VERSION, 7, 0, 64, 0x10_0000_4000);
        assert_eq!(u32::from_le_bytes([p[0], p[1], p[2], p[3]]), BRCMF_C_GET_VERSION);
        assert_eq!(u16::from_le_bytes([p[4], p[5]]), 7);
        assert_eq!(u64::from_le_bytes(p[16..24].try_into().unwrap()), 0x10_0000_4000);
    }

    #[test_case]
    fn firmware_stem_miyake() {
        assert_eq!(
            firmware_stem("apple,miyake", 0x4387),
            "brcmfmac4387c2-pcie.apple,miyake"
        );
        // j473 live chipcommon id
        assert_eq!(
            firmware_stem("apple,miyake", 0x4388),
            "brcmfmac4388-pcie.apple,miyake"
        );
        assert_eq!(firmware_stem("", 0x4378), "brcmfmac4378b1-pcie");
        // unknown still keeps board suffix
        assert_eq!(
            firmware_stem("apple,miyake", 0x1234),
            "brcmfmac1234-pcie.apple,miyake"
        );
    }

    #[test_case]
    fn rambase_table() {
        assert_eq!(rambase_for_chip(0x4388), Some(0x74_0000));
        assert_eq!(rambase_for_chip(0x4387), Some(0x74_0000));
        assert_eq!(rambase_for_chip(0x4378), Some(0x35_2000));
        assert_eq!(rambase_for_chip(0x4377), Some(0x17_0000));
        assert_eq!(rambase_for_chip(0x1234), None);
    }

    #[test_case]
    fn fw_ramsize_and_rstvec() {
        let mut fw = [0u8; RAMSIZE_OFFSET + 8];
        fw[0..4].copy_from_slice(&0x0010_0200u32.to_le_bytes()); // fake rstvec
        fw[RAMSIZE_OFFSET..RAMSIZE_OFFSET + 4]
            .copy_from_slice(&RAMSIZE_MAGIC.to_le_bytes());
        fw[RAMSIZE_OFFSET + 4..RAMSIZE_OFFSET + 8]
            .copy_from_slice(&0x28_0000u32.to_le_bytes());
        assert_eq!(fw_reset_vector(&fw), Some(0x0010_0200));
        assert_eq!(fw_embedded_ramsize(&fw), Some(0x28_0000));
        // wrong magic
        fw[RAMSIZE_OFFSET] = 0;
        assert_eq!(fw_embedded_ramsize(&fw), None);
        assert!(fw_reset_vector(&[]).is_none());
    }

    #[test_case]
    fn shared_addr_and_version() {
        let base = 0x74_0000;
        let size = 0x28_0000;
        assert!(shared_addr_valid(base + 0x1000, base, size));
        assert!(shared_addr_valid(base, base, size));
        assert!(!shared_addr_valid(base + size, base, size));
        assert!(!shared_addr_valid(base - 1, base, size));
        assert!(!shared_addr_valid(0, base, size));
        assert_eq!(shared_version(0x0001_0007), 7);
        assert!(shared_version_ok(5));
        assert!(shared_version_ok(7));
        assert!(!shared_version_ok(4));
        assert!(!shared_version_ok(8));
    }

    #[test_case]
    fn firmware_paths_include_alts() {
        let paths = firmware_search_paths("brcmfmac4388-pcie.apple,miyake");
        assert!(paths.iter().any(|p| p == "/brcm/brcmfmac4388-pcie.apple,miyake.bin"));
        assert!(paths.iter().any(|p| p == "/brcm/brcmfmac4387c2-pcie.apple,miyake.bin"));
        assert!(paths.iter().any(|p| p.contains("/vendorfw/")));
        let nv = nvram_paths_for_fw("/brcm/brcmfmac4388-pcie.apple,miyake.bin");
        assert!(nv.iter().any(|p| p.ends_with(".txt")));
        assert!(nv.iter().any(|p| p.contains("brcmfmac4388-pcie.txt")
            || p.ends_with("brcmfmac4388-pcie.apple,miyake.txt")));
    }

    #[test_case]
    fn erom_scan_synthetic() {
        // Minimal synthetic EROM: one COMPONENT (ARM_CR4) with slave + wrap
        // address descriptors, then EOT.
        // Component A: partnum=0x83e, valid component type.
        // Component B: nmw=1, nsw=1, rev=4.
        // Address slave 4K @ 0x18102000, wrap MWRAP 4K @ 0x18103000.
        let mut words = alloc::vec::Vec::new();
        let push = |v: &mut alloc::vec::Vec<u32>, w: u32| v.push(w);
        // component desc 1: valid|component, partnum 0x83e
        push(
            &mut words,
            DMP_DESC_VALID
                | DMP_DESC_COMPONENT
                | ((0x83eu32) << DMP_COMP_PARTNUM_S),
        );
        // component desc 2: nmw=1, nsw=1, rev=4
        push(
            &mut words,
            DMP_DESC_VALID
                | DMP_DESC_COMPONENT
                | (1 << DMP_COMP_NUM_MWRAP_S)
                | (1 << DMP_COMP_NUM_SWRAP_S)
                | (4 << DMP_COMP_REVISION_S),
        );
        // master port (selects MWRAP wraptype)
        push(&mut words, DMP_DESC_VALID | DMP_DESC_MASTER_PORT);
        // slave address 4K @ 0x18102000
        push(
            &mut words,
            DMP_DESC_VALID
                | DMP_DESC_ADDRESS
                | (DMP_SLAVE_TYPE_SLAVE << DMP_SLAVE_TYPE_S)
                | (DMP_SLAVE_SIZE_4K << DMP_SLAVE_SIZE_TYPE_S)
                | 0x1810_2000,
        );
        // wrap address 4K MWRAP @ 0x18103000
        push(
            &mut words,
            DMP_DESC_VALID
                | DMP_DESC_ADDRESS
                | (DMP_SLAVE_TYPE_MWRAP << DMP_SLAVE_TYPE_S)
                | (DMP_SLAVE_SIZE_4K << DMP_SLAVE_SIZE_TYPE_S)
                | 0x1810_3000,
        );
        // EOT
        push(&mut words, DMP_DESC_VALID | DMP_DESC_EOT);

        let map = words;
        let cores = erom_scan(
            |addr| {
                let idx = ((addr - 0x1000) / 4) as usize;
                map.get(idx).copied().unwrap_or(0)
            },
            0x1000,
            16,
        );
        assert_eq!(cores.len(), 1);
        assert_eq!(cores[0].id, BCMA_CORE_ARM_CR4);
        assert_eq!(cores[0].rev, 4);
        assert_eq!(cores[0].base, 0x1810_2000);
        assert_eq!(cores[0].wrap, 0x1810_3000);
        assert_eq!(find_arm_core(&cores).map(|c| c.wrap), Some(0x1810_3000));
        assert!(find_arm_core(&[]).is_none());
    }
}
