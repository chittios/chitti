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
/// +0x3000). brcmfmac firmware names:
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

/// Dongle TCM / sysmem base for a chipcommon chip id.
///
/// Linux `brcmf_chip_tcm_rambase` has 4387 → `0x740000`. Apple chipcommon
/// `0x4388` (j473) is **not** that entry: the live backplane table in
/// m1n1 `WLANBackplane4388` is `SRAM_BASE=0x200000` / `SRAM_SIZE=0x2e0000`.
/// Probing 4388 at `0x740000` is past the end of SRAM (`0x200000+0x2e0000`
/// = `0x4e0000`) and aborts even when RAM is up.
pub fn rambase_for_chip(chip_id: u16) -> Option<u32> {
    Some(match chip_id {
        0x4377 => 0x17_0000,
        0x4378 => 0x35_2000,
        0x4387 => 0x74_0000,
        0x4388 => 0x20_0000,
        0x4364 => 0x16_0000,
        0x4355 | 0x4359 => 0x16_0000,
        0x4365 | 0x4366 => 0x20_0000,
        0x4350 | 0x4354 | 0x4356 | 0x4358 | 0x4360 | 0x4362 | 0x4371 => 0x18_0000,
        _ => return None,
    })
}

/// Known SRAM span for a chip id (m1n1 backplane tables / Linux raminfo).
pub fn ramsize_for_chip(chip_id: u16) -> u32 {
    match chip_id {
        0x4388 => 0x2e_0000,
        0x4387 => 0x1f_9000,
        0x4378 => 0x20_0000,
        0x4377 => 0x18_0000,
        _ => 0x20_0000,
    }
}

/// Apple signed images carry the reset vector in the firmware footer.
/// Writing TCM[0] (`brcmf_chip_cr4_set_active`) is the Linux default, but
/// Apple signed images set `skip_reset_vector` — the host must not poke
/// the vector slot.
pub fn skip_reset_vector(chip_id: u16) -> bool {
    matches!(chip_id, 0x4387 | 0x4388)
}

/// SYS_MEM `coreinfo` is live (powered + decoding). `None` is an external
/// abort; `0` / `0xffffffff` are the unpowered/unclocked AXI-slave signatures.
/// Clocking (`FGC|CLK`) a dead core turns `0xffffffff` into an abort and
/// wedges the Apple APCIE bus — never `ai_core_reset` unless this is true.
pub fn sysmem_coreinfo_live(coreinfo: Option<u32>) -> bool {
    matches!(coreinfo, Some(v) if v != 0 && v != 0xffff_ffff)
}

/// Whether the host may force a clock into the SYS_MEM wrapper.
pub fn may_clock_sysmem(coreinfo: Option<u32>) -> bool {
    sysmem_coreinfo_live(coreinfo)
}

/// A PCI config Vendor/Device dword is a live function — not an empty slot,
/// a floating bus, or a Configuration Request Retry (CRS = vendor `0x0001`).
/// After PERST# the endpoint answers CRS or all-ones until its config space
/// is up; those must be retried, not treated as "no device".
pub fn pci_config_id_live(id: u32) -> bool {
    let vend = (id & 0xffff) as u16;
    vend != 0 && vend != 0xffff && vend != 0x0001
}

/// Linux `brcmf_pcie_reset_device`: write this value to chipcommon `watchdog`
/// then wait 100 ms. It is a 4-tick watchdog reset, **not** the F0 SSRESET
/// enable mask (`0x10000000`). That mask is a different register encoding
/// and does not re-run the PMU resource defaults that power SYS_MEM.
pub const CC_WATCHDOG_RESET_TICKS: u32 = 4;
pub const CC_WATCHDOG_RESET_WAIT_MS: u64 = 100;

/// PCIE2 config-indirection pair (`brcmf_pcie_pcie2reg_configaddr/data`).
pub const PCIE2_CONFIGADDR: u32 = 0x120;
pub const PCIE2_CONFIGDATA: u32 = 0x124;
/// `BRCMF_PCIE_CFGREG_REG_BAR2_CONFIG` — the BAR2 size/window fixup.
pub const PCIE2_BAR2_CONFIG: u32 = 0x4e0;
/// PCI config `RBAR_CTRL` (Linux `BRCMF_PCIE_CFGREG_RBAR_CTRL`).
pub const PCIE2_RBAR_CTRL: u32 = 0x228;

/// BAR0 is a 32 KiB aperture (Linux `BRCMF_PCIE_REG_MAP_SIZE`).
/// 0x0000 sliding window (config 0x80), 0x1000 wrapper (config 0x70),
/// 0x2000 PCIE2 enum (fixed), 0x3000 chipcommon (fixed).
pub const BAR0_MAP_SIZE: u32 = 32 * 1024;
/// Linux `BRCMF_PCIE_BARO_PCIE_ENUM_OFFSET` — PCIE2 registers without
/// depending on `BAR0_WINDOW` latching.
pub const BAR0_PCIE2_ENUM_OFF: u32 = 0x2000;
/// Fixed chipcommon window inside BAR0.
pub const BAR0_CC_FIXED_OFF: u32 = 0x3000;

/// Offset of a PCIE2 register through the fixed BAR0 enum window.
pub fn pcie2_enum_off(reg: u32) -> u32 {
    BAR0_PCIE2_ENUM_OFF.saturating_add(reg)
}

/// Offset of a chipcommon register through the fixed BAR0 window.
pub fn cc_fixed_off(reg: u32) -> u32 {
    BAR0_CC_FIXED_OFF.saturating_add(reg)
}

/// Linux `brcmf_pcie_reset_device` only rewrites PCIE2 CONFIGADDR/DATA
/// for `rev <= 13`. Rev 74 (j473) uses the attach-time 0x4e0 writeback
/// only, and never that restore loop.
pub fn pcie2_needs_cfg_restore(rev: u8) -> bool {
    rev <= 13
}

/// CONFIGDATA / ECAM 0x4e0 that still equals the PCI vendor/device ID
/// (or all-ones / zero) means the window missed — writing it back would
/// smash `BAR2_CONFIG` with the PCI ID.
pub fn bar2_config_is_window_miss(value: u32, pci_id: u32) -> bool {
    value == pci_id || value == 0 || value == 0xffff_ffff
}

/// Whether a BAR2_CONFIG read is safe to write back (Linux attach).
pub fn bar2_config_may_writeback(value: u32, pci_id: u32) -> bool {
    !bar2_config_is_window_miss(value, pci_id)
}

/// Result of the BAR2_CONFIG attach-time writeback (for `/wifi diag`).
#[derive(Clone, Copy, Debug, Default)]
pub struct Bar2Fixup {
    pub pci_id: u32,
    pub ecam_4e0: u32,
    pub enum_data: u32,
    pub slide_win: u32,
    pub slide_data: u32,
    pub wrote: bool,
}

/// Wrapper IOCTL / RESET_CTL of all-ones is a dead AXI slave, not
/// "core down". Clocking (`FGC|CLK`) that turns the next access into
/// an abort and wedges APCIE.
pub fn wrap_regs_live(ioctl: Option<u32>, reset_ctl: Option<u32>) -> bool {
    match (ioctl, reset_ctl) {
        (Some(i), Some(r)) => i != 0xffff_ffff && r != 0xffff_ffff,
        _ => false,
    }
}

/// D11 (802.11) wrapper ioctl bits — Linux `D11_BCMA_IOCTL_*`.
pub const D11_IOCTL_PHYCLOCKEN: u32 = 0x0004;
pub const D11_IOCTL_PHYRESET: u32 = 0x0008;

// ── Apple host port bring-up (pure, from pcie-apple.c + pinctrl-apple-gpio.c)

/// t8112 `pcie_pins`: CLKREQ on pinctrl_ap pins 162/163/164, function 1
/// (`APPLE_PINMUX(n, 1)` → `periph1`). Linux applies this via `pinctrl-0`
/// before `apple_pcie_setup_link`; without it the endpoint cannot assert
/// CLKREQ# and the dongle PMU never starts the RAM domain.
pub const PCIE_CLKREQ_PINS: &[u32] = &[162, 163, 164];
/// Peripheral function for those CLKREQ pins (`periph1`).
pub const PCIE_CLKREQ_FUNC: u32 = 1;
/// j473 `reset-gpios = <&pinctrl_ap 166 GPIO_ACTIVE_LOW>` (port00 PERST#).
pub const PCIE_PERST_GPIO: u32 = 166;

/// pinctrl register fields — `drivers/pinctrl/pinctrl-apple-gpio.c`.
pub const PINCTRL_REG_DATA: u32 = 1 << 0;
pub const PINCTRL_REG_MODE: u32 = 0x7 << 1;
pub const PINCTRL_REG_MODE_OUT: u32 = 1 << 1;
pub const PINCTRL_REG_PERIPH: u32 = 0x3 << 5;
pub const PINCTRL_REG_INPUT_ENABLE: u32 = 1 << 9;

/// `apple_gpio_pinmux_set`: replace PERIPH, set INPUT_ENABLE, leave
/// the rest (MODE/DATA/PULL) alone.
pub fn pinctrl_pinmux_word(cur: u32, func: u32) -> u32 {
    (cur & !PINCTRL_REG_PERIPH) | ((func & 0x3) << 5) | PINCTRL_REG_INPUT_ENABLE
}

/// `apple_gpio_direction_output`: PERIPH=GPIO, MODE=OUT, DATA=level.
/// ACTIVE_LOW PERST: assert = `level_high=false`, deassert = `true`.
pub fn pinctrl_gpio_out_word(cur: u32, level_high: bool) -> u32 {
    let mut v = cur & !(PINCTRL_REG_MODE | PINCTRL_REG_PERIPH | PINCTRL_REG_DATA);
    v |= PINCTRL_REG_MODE_OUT;
    if level_high {
        v |= PINCTRL_REG_DATA;
    }
    v
}

/// Host-side steps of Linux `apple_pcie_setup_link` + `setup_port` tail,
/// plus the CLKREQ pinmux pinctrl applies before the PCIe driver runs.
/// `cycle_power` inserts a rail-off **after** PERST is asserted (never
/// before — powering the module with PERST# deasserted is how the dongle
/// PMU samples straps wrong and leaves SYS_MEM gated).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PortBringupStep {
    PinmuxClkreq,
    AppclkOn,
    PerstAssert,
    PowerOffWhilePerst,
    PowerOnWhilePerst,
    RefclkOn,
    SettlePwrenMs,
    PerstDeassert,
    SettlePerstMs,
    WaitReady,
    ClearCgdis,
    LtssmStart,
}

/// Canonical host-port bring-up. The two 100 ms settles are `SettlePwrenMs`
/// (Tpvperl, only meaningful when pwren is present — it always is on j473)
/// and `SettlePerstMs` (PCIe r5.0 §6.6.1 after PERST# deassert).
pub fn apple_port_bringup_steps(cycle_power: bool) -> &'static [PortBringupStep] {
    use PortBringupStep::*;
    if cycle_power {
        &[
            PinmuxClkreq,
            AppclkOn,
            PerstAssert,
            PowerOffWhilePerst,
            PowerOnWhilePerst,
            RefclkOn,
            SettlePwrenMs,
            PerstDeassert,
            SettlePerstMs,
            WaitReady,
            ClearCgdis,
            LtssmStart,
        ]
    } else {
        &[
            PinmuxClkreq,
            AppclkOn,
            PerstAssert,
            PowerOnWhilePerst,
            RefclkOn,
            SettlePwrenMs,
            PerstDeassert,
            SettlePerstMs,
            WaitReady,
            ClearCgdis,
            LtssmStart,
        ]
    }
}

/// Soft upper bound used only as a sanity cap (not the primary size source).
/// Download prefers a tight `fw+nv+4` pack — the old 0x280000 default placed
/// NVRAM past mapped TCM on j473 and external-aborted.
pub fn default_ramsize_hint(chip_id: u16) -> u32 {
    match chip_id {
        0x4388 => 0x2e_0000,
        0x4387 => 0x26_0000,
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

// ── Shared-info + common-ring layouts (Linux brcmfmac/pcie.c) ──────────────
//
// Once firmware posts `shared_addr`, the host reads this header to locate the
// H2D/D2H control rings that carry ioctl pointer requests. Pure so the offsets
// are pinned before any MMIO write.

/// Byte offset of `ring_info_addr` in the shared-info header (LE u32).
pub const SHARED_RING_INFO_OFF: usize = 0x34;
/// Byte offset of `hto_d_mb_data_addr` (host→device mailbox).
pub const SHARED_HTOD_MB_OFF: usize = 0x2c;
/// Byte offset of `dto_h_mb_data_addr` (device→host mailbox).
pub const SHARED_DTOH_MB_OFF: usize = 0x30;

/// Minimum shared-info bytes we must be able to read for ring bring-up.
pub const SHARED_INFO_MIN: usize = SHARED_RING_INFO_OFF + 4;

/// Decode the ring-info TCM address from a shared-info header blob.
pub fn shared_ring_info_addr(shared: &[u8]) -> Option<u32> {
    if shared.len() < SHARED_INFO_MIN {
        return None;
    }
    let addr = u32::from_le_bytes([
        shared[SHARED_RING_INFO_OFF],
        shared[SHARED_RING_INFO_OFF + 1],
        shared[SHARED_RING_INFO_OFF + 2],
        shared[SHARED_RING_INFO_OFF + 3],
    ]);
    if addr == 0 {
        None
    } else {
        Some(addr)
    }
}

/// Decode host→device mailbox data address.
pub fn shared_htod_mb_addr(shared: &[u8]) -> Option<u32> {
    if shared.len() < SHARED_DTOH_MB_OFF {
        return None;
    }
    let addr = u32::from_le_bytes([
        shared[SHARED_HTOD_MB_OFF],
        shared[SHARED_HTOD_MB_OFF + 1],
        shared[SHARED_HTOD_MB_OFF + 2],
        shared[SHARED_HTOD_MB_OFF + 3],
    ]);
    if addr == 0 {
        None
    } else {
        Some(addr)
    }
}

/// A complete ioctl-pointer request: 8-byte msg header + 24-byte payload.
pub fn pack_ioctl_request(
    if_id: i8,
    epoch: u8,
    request_id: u32,
    cmd: u32,
    trans_id: u16,
    input_len: u16,
    output_len: u16,
    host_buf: u64,
) -> [u8; 32] {
    let mut out = [0u8; 32];
    out[..8].copy_from_slice(&pack_msg_header(
        MSG_TYPE_IOCTLPTR_REQ,
        if_id,
        0,
        epoch,
        request_id,
    ));
    out[8..].copy_from_slice(&pack_ioctl_ptr_req(
        cmd, trans_id, input_len, output_len, host_buf,
    ));
    out
}

/// True when the shared-info header has enough structure to attempt ring
/// init (version OK + non-zero ring_info pointer). Does not touch hardware.
pub fn rings_locatable(shared_flags: u32, shared_header: &[u8]) -> bool {
    shared_version_ok(shared_version(shared_flags))
        && shared_ring_info_addr(shared_header).is_some()
}

/// Build firmware search paths for a stem (with brcmfmac alternate names). Pure.
pub fn firmware_search_paths(stem: &str) -> alloc::vec::Vec<alloc::string::String> {
    use alloc::{format, string::String, vec::Vec};
    let mut stems: Vec<String> = Vec::new();
    stems.push(String::from(stem));
    // Always try the j473 / miyake brcmfmac names as fallbacks.
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
/// Separate PMU core (`brcmf_chip_get_pmu` when AOB is present).
pub const BCMA_CORE_PMU: u16 = 0x827;

/// PMU register offsets from the PMU core base (Linux `chipcregs`, 0x600…).
/// On AOB chips these are **not** at chipcommon+0x600 — that overlay aborts.
pub const PMU_CONTROL: u32 = 0x600;
pub const PMU_CAPABILITIES: u32 = 0x604;
pub const PMU_STATUS: u32 = 0x608;
pub const PMU_RES_STATE: u32 = 0x60c;
pub const PMU_MIN_RES_MASK: u32 = 0x618;
pub const PMU_MAX_RES_MASK: u32 = 0x61c;
pub const PMU_WATCHDOG: u32 = 0x634;
pub const PMU_RES_REQ_TIMER: u32 = 0x644;
/// Resource-request mask (`chipcregs.res_req_mask`). Newer PMUs honour this
/// when `min_res_mask` is locked by OTP.
pub const PMU_RES_REQ_MASK: u32 = 0x648;
pub const PMU_CAPABILITIES_EXT: u32 = 0x64c;

/// `pmuregs_t` layout used when the PMU is its own core (`si_setcore(PMU)`).
/// `pmucontrol` is at **+0x00**, not +0x600. The 0x600 block can be a
/// read-only mirror — j473 ignored every store at +0x618.
pub const PMU_COMPACT_DELTA: u32 = 0x600;

/// Overlay (chipcregs) offset → compact `pmuregs_t` offset.
pub fn pmu_compact_off(overlay_off: u32) -> u32 {
    overlay_off.saturating_sub(PMU_COMPACT_DELTA)
}

/// `res_req_timer` enable + a short ILP-tick timeout (bit 24 + 0xff).
pub const PMU_RES_REQ_TIMER_GO: u32 = 0x0100_00ff;

/// Chipcommon `clk_ctl_st` (0x1e0).
pub const CC_CLK_CTL_ST: u32 = 0x1e0;
pub const CCS_FORCEALP: u32 = 1 << 0;
pub const CCS_FORCEHT: u32 = 1 << 1;
pub const CCS_ALPAVAIL: u32 = 1 << 16;
pub const CCS_HTAVAIL: u32 = 1 << 17;

pub fn ccs_force_ht_word(cur: u32) -> u32 {
    cur | CCS_FORCEALP | CCS_FORCEHT
}

pub fn ccs_ht_avail(st: u32) -> bool {
    st & CCS_HTAVAIL != 0
}

/// PCIE2 `CLK_CTL` / `PWR_CTL` — m1n1 `WLANPCIE2Regs` on 4388.
/// These are **not** chipcommon `clk_ctl_st`. Same offset (0x1e0) but only
/// when BAR0_WINDOW is the PCIE2 core (0x18001000).
pub const PCIE2_CLK_CTL: u32 = 0x1e0;
pub const PCIE2_PWR_CTL: u32 = 0x1e8;
pub const PCIE2_CLK_FORCEALP: u32 = 1 << 0;
pub const PCIE2_CLK_FORCEHT: u32 = 1 << 1;
pub const PCIE2_CLK_HAVEALPREQ: u32 = 1 << 3;
pub const PCIE2_CLK_HAVEHTREQ: u32 = 1 << 4;
pub const PCIE2_CLK_HQCLKREQ: u32 = 1 << 6;
pub const PCIE2_CLK_HAVEHT: u32 = 1 << 17;
/// Request every power domain + the PWRON bits (DMN0–4, PWRON_DMN0–4).
pub const PCIE2_PWR_ALL_DOMAINS: u32 = 0x1f | (0x1f << 8);

pub fn pcie2_clk_force_word(cur: u32) -> u32 {
    cur | PCIE2_CLK_FORCEALP
        | PCIE2_CLK_FORCEHT
        | PCIE2_CLK_HAVEALPREQ
        | PCIE2_CLK_HAVEHTREQ
        | PCIE2_CLK_HQCLKREQ
}

pub fn pcie2_have_ht(st: u32) -> bool {
    st & PCIE2_CLK_HAVEHT != 0
}

/// PMU revision from `pmucapabilities` (low 8 bits).
pub fn pmu_rev(cap: u32) -> u8 {
    (cap & 0xff) as u8
}

/// Mask the host should write to `min_res_mask` to request every resource
/// the chip advertises. Bits outside `max` make the PMU **reject the whole
/// write** and leave `min` unchanged (j473: `0xffffffff` was a no-op).
pub fn pmu_request_mask(min: u32, max: u32) -> u32 {
    if max == 0 || max == 0xffff_ffff {
        min
    } else {
        max
    }
}

/// Resources allowed by `max` but not currently in `res_state`.
pub fn pmu_off_bits(max: u32, state: u32) -> u32 {
    max & !state
}

/// Backplane address of a PMU register.
pub fn pmu_reg(pmu_base: u32, off: u32) -> u32 {
    pmu_base.saturating_add(off)
}

/// A PMU register read is live (powered + decoding).
pub fn pmu_reg_live(v: Option<u32>) -> bool {
    matches!(v, Some(x) if x != 0xffff_ffff)
}

/// Walk a PL-368 EROM, calling `read32(addr)` for each word. Pure relative to
/// the reader. Caps at `max_cores` to bound work.
pub fn erom_scan(
    mut read32: impl FnMut(u32) -> u32,
    erom_base: u32,
    max_cores: usize,
) -> alloc::vec::Vec<EromCore> {
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
        assert_eq!(
            u32::from_le_bytes([p[0], p[1], p[2], p[3]]),
            BRCMF_C_GET_VERSION
        );
        assert_eq!(u16::from_le_bytes([p[4], p[5]]), 7);
        assert_eq!(
            u64::from_le_bytes(p[16..24].try_into().unwrap()),
            0x10_0000_4000
        );
    }

    #[test_case]
    fn shared_ring_info_offsets_and_ioctl_request() {
        let mut hdr = [0u8; SHARED_INFO_MIN];
        // flags / version word is at 0; ring_info at SHARED_RING_INFO_OFF.
        hdr[0] = 6; // version 6
        hdr[SHARED_RING_INFO_OFF..SHARED_RING_INFO_OFF + 4]
            .copy_from_slice(&0x0012_3400u32.to_le_bytes());
        hdr[SHARED_HTOD_MB_OFF..SHARED_HTOD_MB_OFF + 4]
            .copy_from_slice(&0x0011_0000u32.to_le_bytes());
        assert_eq!(shared_ring_info_addr(&hdr), Some(0x0012_3400));
        assert_eq!(shared_htod_mb_addr(&hdr), Some(0x0011_0000));
        assert!(rings_locatable(6, &hdr));
        assert!(!rings_locatable(6, &[0u8; 8]), "too short");
        assert!(!rings_locatable(99, &hdr), "bad version");

        let req = pack_ioctl_request(0, 1, 42, BRCMF_C_SCAN, 3, 0, 256, 0x1000);
        assert_eq!(req[0], MSG_TYPE_IOCTLPTR_REQ);
        assert_eq!(u32::from_le_bytes([req[4], req[5], req[6], req[7]]), 42);
        assert_eq!(
            u32::from_le_bytes([req[8], req[9], req[10], req[11]]),
            BRCMF_C_SCAN
        );
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
        assert_eq!(rambase_for_chip(0x4388), Some(0x20_0000));
        assert_eq!(rambase_for_chip(0x4387), Some(0x74_0000));
        assert_eq!(ramsize_for_chip(0x4388), 0x2e_0000);
        assert_eq!(ramsize_for_chip(0x4387), 0x1f_9000);
        assert!(0x20_0000 + ramsize_for_chip(0x4388) <= 0x4e_0000);
        assert!(0x74_0000 >= 0x20_0000 + ramsize_for_chip(0x4388));
        assert_eq!(rambase_for_chip(0x4378), Some(0x35_2000));
        assert_eq!(rambase_for_chip(0x4377), Some(0x17_0000));
        assert_eq!(rambase_for_chip(0x1234), None);
    }

    #[test_case]
    fn fw_ramsize_and_rstvec() {
        let mut fw = [0u8; RAMSIZE_OFFSET + 8];
        fw[0..4].copy_from_slice(&0x0010_0200u32.to_le_bytes()); // fake rstvec
        fw[RAMSIZE_OFFSET..RAMSIZE_OFFSET + 4].copy_from_slice(&RAMSIZE_MAGIC.to_le_bytes());
        fw[RAMSIZE_OFFSET + 4..RAMSIZE_OFFSET + 8].copy_from_slice(&0x28_0000u32.to_le_bytes());
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
        assert!(paths
            .iter()
            .any(|p| p == "/brcm/brcmfmac4388-pcie.apple,miyake.bin"));
        assert!(paths
            .iter()
            .any(|p| p == "/brcm/brcmfmac4387c2-pcie.apple,miyake.bin"));
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
            DMP_DESC_VALID | DMP_DESC_COMPONENT | ((0x83eu32) << DMP_COMP_PARTNUM_S),
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

    #[test_case]
    fn apple_4388_skips_reset_vector_and_sysmem_liveness() {
        assert!(skip_reset_vector(0x4388));
        assert!(skip_reset_vector(0x4387));
        assert!(!skip_reset_vector(0x4378));
        assert!(!sysmem_coreinfo_live(None));
        assert!(!sysmem_coreinfo_live(Some(0)));
        assert!(!sysmem_coreinfo_live(Some(0xffff_ffff)));
        assert!(sysmem_coreinfo_live(Some(0x0001_00c0)));
        assert!(!may_clock_sysmem(Some(0xffff_ffff)));
        assert!(may_clock_sysmem(Some(0x12)));
        assert!(!pci_config_id_live(0xffff_ffff));
        assert!(!pci_config_id_live(0));
        assert!(!pci_config_id_live(0x0000_0001), "CRS vendor 0x0001");
        assert!(pci_config_id_live(0x4434_14e4));
        assert_eq!(CC_WATCHDOG_RESET_TICKS, 4);
        assert_eq!(CC_WATCHDOG_RESET_WAIT_MS, 100);
        assert_ne!(CC_WATCHDOG_RESET_TICKS, 0x1000_0000);
        assert_eq!(PCIE2_BAR2_CONFIG, 0x4e0);
        assert_eq!(pcie2_enum_off(PCIE2_CONFIGADDR), 0x2120);
        assert_eq!(pcie2_enum_off(PCIE2_CONFIGDATA), 0x2124);
        assert_eq!(cc_fixed_off(CC_WATCHDOG), 0x3080);
        assert!(!pcie2_needs_cfg_restore(74));
        assert!(pcie2_needs_cfg_restore(13));
        assert!(!pcie2_needs_cfg_restore(14));
        // Writing 0x443414e4 back into BAR2_CONFIG is the window-miss bug.
        assert!(bar2_config_is_window_miss(0x4434_14e4, 0x4434_14e4));
        assert!(bar2_config_is_window_miss(0, 0x4434_14e4));
        assert!(bar2_config_is_window_miss(0xffff_ffff, 0x4434_14e4));
        assert!(!bar2_config_is_window_miss(0x0000_1800, 0x4434_14e4));
        assert!(bar2_config_may_writeback(0x0000_1800, 0x4434_14e4));
        assert!(!bar2_config_may_writeback(0x4434_14e4, 0x4434_14e4));
        assert!(!wrap_regs_live(None, Some(1)));
        assert!(!wrap_regs_live(Some(0xffff_ffff), Some(0xffff_ffff)));
        assert!(wrap_regs_live(Some(0x1), Some(0x1)));
        assert!(wrap_regs_live(Some(0), Some(0)));
        assert_eq!(D11_IOCTL_PHYRESET | D11_IOCTL_PHYCLOCKEN, 0x000c);
        assert_eq!(BCMA_CORE_PMU, 0x827);
        assert_eq!(pmu_reg(0x1801_2000, PMU_CAPABILITIES), 0x1801_2604);
        assert_eq!(pmu_reg(0x1801_2000, PMU_MIN_RES_MASK), 0x1801_2618);
        assert_eq!(pmu_reg(0x1801_2000, PMU_MAX_RES_MASK), 0x1801_261c);
        assert_ne!(pmu_reg(0x1801_2000, PMU_MIN_RES_MASK), 0x1800_0618);
        assert!(!pmu_reg_live(None));
        assert!(!pmu_reg_live(Some(0xffff_ffff)));
        assert!(pmu_reg_live(Some(0x2b)));
        assert_eq!(PMU_MIN_RES_MASK, 0x618);
        assert_eq!(PMU_MAX_RES_MASK, 0x61c);
        assert_eq!(PMU_RES_REQ_MASK, 0x648);
        // Live j473 values: all-ones was rejected; request must be ⊆ max.
        let min = 0x0607_7eed;
        let max = 0x0e4f_7fff;
        let st = 0x064e_7fed;
        assert_eq!(pmu_rev(0x0456_6b2b), 43);
        assert_eq!(pmu_request_mask(min, max), max);
        assert_eq!(pmu_request_mask(min, max) & !max, 0);
        assert_eq!(pmu_request_mask(min, 0xffff_ffff), min);
        assert_eq!(pmu_off_bits(max, st), 0x0801_0012);
        assert_ne!(pmu_request_mask(min, max), 0xffff_ffff);
        assert_eq!(pmu_compact_off(PMU_MIN_RES_MASK), 0x18);
        assert_eq!(pmu_compact_off(PMU_MAX_RES_MASK), 0x1c);
        assert_eq!(pmu_compact_off(PMU_RES_STATE), 0x0c);
        assert_eq!(pmu_compact_off(PMU_RES_REQ_MASK), 0x48);
        assert_eq!(pmu_compact_off(PMU_RES_REQ_TIMER), 0x44);
        assert_eq!(pmu_reg(0x1801_2000, pmu_compact_off(PMU_MIN_RES_MASK)), 0x1801_2018);
        assert_eq!(ccs_force_ht_word(0), CCS_FORCEALP | CCS_FORCEHT);
        assert!(!ccs_ht_avail(0));
        assert!(ccs_ht_avail(CCS_HTAVAIL | CCS_ALPAVAIL));
        assert_eq!(CC_CLK_CTL_ST, 0x1e0);
        assert_eq!(PCIE2_CLK_CTL, 0x1e0);
        assert_eq!(PCIE2_PWR_CTL, 0x1e8);
        assert_eq!(pcie2_clk_force_word(0) & PCIE2_CLK_FORCEHT, PCIE2_CLK_FORCEHT);
        assert!(pcie2_have_ht(PCIE2_CLK_HAVEHT));
        assert!(!pcie2_have_ht(0x0205_0240), "CC-looking status is not PCIE2 HAVEHT");
        assert_eq!(PCIE2_PWR_ALL_DOMAINS, 0x1f1f);
    }

    #[test_case]
    fn pinctrl_words_match_linux() {
        // apple_gpio_pinmux_set(func=1): PERIPH=1, INPUT_ENABLE, rest kept.
        let cur = 0x0000_0005; // leftover DATA + MODE noise
        let mux = pinctrl_pinmux_word(cur, PCIE_CLKREQ_FUNC);
        assert_eq!(mux & PINCTRL_REG_PERIPH, PCIE_CLKREQ_FUNC << 5);
        assert_ne!(mux & PINCTRL_REG_INPUT_ENABLE, 0);
        assert_eq!(
            mux & !(PINCTRL_REG_PERIPH | PINCTRL_REG_INPUT_ENABLE),
            cur & !(PINCTRL_REG_PERIPH | PINCTRL_REG_INPUT_ENABLE)
        );
        // apple_gpio_direction_output(0): GPIO, MODE=OUT, DATA=0 (PERST assert).
        let out_lo = pinctrl_gpio_out_word(mux, false);
        assert_eq!(out_lo & PINCTRL_REG_PERIPH, 0);
        assert_eq!(out_lo & PINCTRL_REG_MODE, PINCTRL_REG_MODE_OUT);
        assert_eq!(out_lo & PINCTRL_REG_DATA, 0);
        let out_hi = pinctrl_gpio_out_word(0, true);
        assert_ne!(out_hi & PINCTRL_REG_DATA, 0);
        assert_eq!(PCIE_CLKREQ_PINS, &[162, 163, 164]);
        assert_eq!(PCIE_PERST_GPIO, 166);
    }

    #[test_case]
    fn port_bringup_asserts_perst_before_power() {
        use PortBringupStep::*;
        let cyc = apple_port_bringup_steps(true);
        let cold = apple_port_bringup_steps(false);
        // CLKREQ pinmux is first — Linux pinctrl runs before pcie-apple.
        assert_eq!(cyc[0], PinmuxClkreq);
        assert_eq!(cold[0], PinmuxClkreq);
        let perst = cyc.iter().position(|&s| s == PerstAssert).unwrap();
        let off = cyc.iter().position(|&s| s == PowerOffWhilePerst).unwrap();
        let on = cyc.iter().position(|&s| s == PowerOnWhilePerst).unwrap();
        let refc = cyc.iter().position(|&s| s == RefclkOn).unwrap();
        let de = cyc.iter().position(|&s| s == PerstDeassert).unwrap();
        assert!(perst < off, "PERST# must be held before the rail drops");
        assert!(off < on, "rail must drain before it comes back");
        assert!(on < refc, "pwren before endpoint refclk (setup_link)");
        assert!(refc < de, "refclk before PERST# deassert (Tperst-clk)");
        // Cold start never drops the rail (pwren is already the power-on).
        assert!(!cold.iter().any(|&s| s == PowerOffWhilePerst));
        assert!(cold.iter().any(|&s| s == PowerOnWhilePerst));
        // Both 100 ms settles present, in order.
        let s0 = cyc.iter().position(|&s| s == SettlePwrenMs).unwrap();
        let s1 = cyc.iter().position(|&s| s == SettlePerstMs).unwrap();
        assert!(s0 < de && de < s1);
    }
}
