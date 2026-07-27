//! **Wi-Fi** — real Broadcom FullMAC on Apple Silicon, facade on emulators.
//!
//! Driver code lives under [`brcm`] (`drivers/wifi/brcm/`). This module is the
//! shell/net-facing facade:
//!
//! On a bare m1n1 boot of a Mac mini M2 (`chitti.wifi` bootarg):
//! 1. [`crate::arch::aarch64::apple_pcie`] maps the APCIE ECAM + DART.
//! 2. `brcm` probes `pci14e4,4434` (BCM4387/4388), reads chip id + FDT MAC.
//! 3. `/wifi load` downloads the Asahi `.bin` into dongle TCM (BAR2) and waits
//!    for the shared-RAM handshake. Scan/connect still need the common-ring
//!    ioctl path (next). Until then `/wifi info` shows BAR + firmware state.
//!
//! On QEMU/VBox the historical **wired facade** remains: scan shows a fake
//! SSID and connect runs DHCP on the virtio/e1000 NIC.

pub mod brcm;
pub mod iwl;
pub mod wpa;

/// Pure brcmfmac helpers (always available for unit tests).
pub use brcm::proto;

use alloc::string::String;
use alloc::vec::Vec;

/// Bring up Apple PCIe + SMC power + probe the WiFi function.
/// Returns true if the radio has live BARs (link up + enumerated).
pub fn init_apple() -> bool {
    #[cfg(all(target_arch = "aarch64", not(test)))]
    {
        if !crate::arch::aarch64::is_apple() {
            return false;
        }
        if !crate::arch::aarch64::apple_pcie::init() {
            return false;
        }
        // probe() asserts SMC power and waits for link.
        return brcm::probe();
    }
    #[cfg(not(all(target_arch = "aarch64", not(test))))]
    {
        false
    }
}

/// Re-run SMC power + link wait + probe (e.g. `/wifi power`).
pub fn power_on() -> Result<(), &'static str> {
    // An Intel radio needs no board-level power sequencing — it is an ordinary PCIe
    // function — so `up` on such a machine means bring-up, which `/wifi up` routes to
    // `iwl::bring_up`. Handled by the caller so the message can carry detail.

    #[cfg(all(target_arch = "aarch64", not(test)))]
    {
        if !crate::arch::aarch64::is_apple() {
            return Err("not Apple Silicon");
        }
        if !crate::arch::aarch64::apple_pcie::ready()
            && !crate::arch::aarch64::apple_pcie::init()
        {
            return Err("PCIe init failed");
        }
        if !crate::arch::aarch64::apple_pcie::retry_wifi_power() {
            return Err("link still down after SMC gP0d power");
        }
        if brcm::probe() {
            Ok(())
        } else {
            // Still publish FDT/link state so `/wifi info` is useful.
            let _ = brcm::ensure_stub("probe incomplete (BAR/MMIO)");
            Err("link up but BAR MMIO not reachable — see ktrace (BAR window miss/HIT)")
        }
    }
    #[cfg(not(all(target_arch = "aarch64", not(test))))]
    {
        Err("Apple Wi-Fi is aarch64-only")
    }
}

/// True when we have **any** Apple WiFi state (live BARs **or** FDT stub after
/// a partial probe). Used by `/wifi info` so the user never sees a blank
/// "no adapter" once `chitti.wifi` has touched the radio.
pub fn hardware_present() -> bool {
    #[cfg(all(target_arch = "aarch64", not(test)))]
    {
        if brcm::is_present() {
            return true;
        }
        // PCIe host is up even if the endpoint BAR window failed.
        return crate::arch::aarch64::is_apple()
            && crate::arch::aarch64::apple_pcie::ready();
    }
    #[cfg(not(all(target_arch = "aarch64", not(test))))]
    {
        false
    }
}

/// True when BARs mapped and we can talk to the chip (not merely link-up).
pub fn radio_ready() -> bool {
    #[cfg(all(target_arch = "aarch64", not(test)))]
    {
        return brcm::with_dev(|d| d.bar0 != 0).unwrap_or(false);
    }
    #[cfg(not(all(target_arch = "aarch64", not(test))))]
    {
        false
    }
}

/// Human-readable status lines for `/wifi info`.
pub fn info_lines() -> Vec<String> {
    use alloc::format;
    let mut lines = Vec::new();
    // Intel WiFi, if the machine has one. Identification only — say so on the same line
    // as the part name, so nobody reads "AX200 found" as "WiFi works".
    if let Some(f) = iwl::probe() {
        lines.push(format!(
            "adapter: Intel {} (device {:#06x}) -- identified only, no driver yet",
            f.family.label(),
            f.device_id
        ));
        lines.push(format!(
            "  firmware needed: {} (would try {} API versions down to {})",
            f.firmware.first().map(|s| s.as_str()).unwrap_or("?"),
            f.firmware.len(),
            f.family.min_api()
        ));
        lines.push(
            "  fetch with `cargo xtask iwlwifi-assets`; scan/associate is not implemented".into(),
        );
    }
    #[cfg(all(target_arch = "aarch64", not(test)))]
    {
        if crate::arch::aarch64::is_apple() {
            // Always surface PCIe host state on Apple.
            let r = crate::arch::aarch64::apple_pcie::report();
            if r.ready {
                lines.push(format!(
                    "pcie: ECAM {:#x} bus_end={} link={} port0={:#x} dart={:#x}/{}",
                    r.ecam,
                    r.bus_end,
                    if r.link_up { "up" } else { "DOWN" },
                    r.port0,
                    r.dart,
                    r.dart_sid
                ));
            } else {
                lines.push(
                    "pcie: not ready — boot with `chitti.wifi` on bare m1n1".into(),
                );
            }

            if let Some(()) = brcm::with_dev(|d| {
                lines.push(format!(
                    "adapter: Broadcom FullMAC {:04x}:{:04x}  bus {:02x}:{:02x}.{}",
                    d.pci.vendor, d.pci.device, d.pci.bus, d.pci.dev, d.pci.func
                ));
                lines.push(format!(
                    "  chip_id={:#x} rev={}  board={}  antenna={}",
                    d.chip_id, d.chip_rev, d.board_type, d.antenna_sku
                ));
                lines.push(format!(
                    "  mac={:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
                    d.mac[0], d.mac[1], d.mac[2], d.mac[3], d.mac[4], d.mac[5]
                ));
                lines.push(format!("  BAR0={:#x} BAR2={:#x}", d.bar0, d.bar2));
                lines.push(format!(
                    "  firmware: stem={}  loaded={}",
                    d.firmware_stem, d.firmware_up
                ));
                if d.rambase != 0 {
                    lines.push(format!(
                        "  dongle: rambase={:#x} ramsize={:#x} shared={:#x} ver={}",
                        d.rambase, d.ramsize, d.shared_addr, d.shared_version
                    ));
                }
                if d.bar0 == 0 {
                    lines.push(
                        "  status: PCIe link may be up but BAR MMIO not mapped — /wifi power"
                            .into(),
                    );
                    lines.push(
                        "  next: check ktrace for 'BAR window HIT'; need host outbound".into(),
                    );
                } else if !d.firmware_up {
                    lines.push(
                        "  status: radio enumerated (BAR MEM OK); firmware not loaded".into(),
                    );
                    lines.push(format!(
                        "  next: host `make wifi-assets` then rebuild (embeds {}.bin), or place it in /brcm/",
                        d.firmware_stem
                    ));
                    lines.push(
                        "  then: /wifi load — downloads the dongle image over BAR2 TCM".into(),
                    );
                } else {
                    lines.push(
                        "  status: firmware up (shared-RAM handshake OK); rings/ioctl next"
                            .into(),
                    );
                    lines.push(
                        "  next: /wifi scan once common rings are wired".into(),
                    );
                }
            }) {
                let _ = ();
            } else if r.ready {
                lines.push(
                    "adapter: not probed yet — run /wifi power".into(),
                );
            } else {
                lines.push("adapter: none".into());
            }
            return lines;
        }
    }
    #[cfg(all(target_arch = "aarch64", not(test)))]
    {
        lines.push("adapter: none (not Apple Silicon)".into());
    }
    #[cfg(not(all(target_arch = "aarch64", not(test))))]
    {
        lines.push("adapter: none (Apple Wi-Fi is aarch64-only)".into());
    }
    lines
}

/// Attempt firmware load from the store.
pub fn load_firmware() -> Result<(), &'static str> {
    #[cfg(all(target_arch = "aarch64", not(test)))]
    {
        if !radio_ready() {
            power_on().map_err(|e| e)?;
        }
        if !radio_ready() {
            return Err("radio BARs not mapped — cannot load firmware yet");
        }
        return brcm::try_load_firmware();
    }
    #[cfg(not(all(target_arch = "aarch64", not(test))))]
    {
        Err("Apple Wi-Fi firmware load is aarch64-only")
    }
}

/// Hard PERST# reset of the WiFi endpoint (forces a full chip reset so the
/// dongle PMU re-powers the SYS_MEM/RAM domain), then re-probe/re-map BARs.
pub fn hard_reset() -> Result<(), &'static str> {
    #[cfg(all(target_arch = "aarch64", not(test)))]
    {
        if crate::arch::aarch64::is_apple() {
            return brcm::hard_reset();
        }
    }
    Err("Apple Wi-Fi hard reset is aarch64-only")
}

/// Decisive on-hardware diagnostic for the BAR2/TCM read-abort blocker.
/// Returns human-readable lines (also mirrored to ktrace). Apple-only.
pub fn diag() -> Vec<String> {
    #[cfg(all(target_arch = "aarch64", not(test)))]
    {
        if crate::arch::aarch64::is_apple() {
            if !radio_ready() {
                let _ = power_on();
            }
            if !radio_ready() {
                return alloc::vec![String::from(
                    "radio BARs not mapped — run /wifi power first (see /wifi info + ktrace)"
                )];
            }
            return brcm::diag();
        }
    }
    alloc::vec![String::from("/wifi diag is Apple-Silicon only")]
}

/// Scan for networks (real radio) or return the facade list.
pub fn scan() -> Result<Vec<proto::BssInfo>, &'static str> {
    #[cfg(all(target_arch = "aarch64", not(test)))]
    {
        if crate::arch::aarch64::is_apple() {
            if !crate::arch::aarch64::apple_pcie::port0_link_up() || !radio_ready() {
                let _ = power_on();
            }
            if !crate::arch::aarch64::apple_pcie::port0_link_up() {
                return Err("WiFi PCIe link down — /wifi power failed");
            }
            if !radio_ready() {
                return Err(
                    "BAR MMIO not mapped yet — cannot RF scan (see /wifi info + ktrace)",
                );
            }
            if brcm::with_dev(|d| !d.firmware_up).unwrap_or(true) {
                match brcm::try_load_firmware() {
                    Ok(()) => {}
                    Err(e) => {
                        return Err(e);
                    }
                }
            }
            return brcm::scan();
        }
    }
    // Emulator facade.
    Ok(alloc::vec![proto::BssInfo {
        ssid: "chitti-lan".into(),
        bssid: [0x02, 0, 0, 0, 0, 1],
        channel: 0,
        rssi: -40,
        privacy: true,
    }])
}

/// Connect to `ssid` with password `psk`.
pub fn connect(ssid: &str, psk: &str) -> Result<(), &'static str> {
    #[cfg(all(target_arch = "aarch64", not(test)))]
    {
        if crate::arch::aarch64::is_apple() {
            if !crate::arch::aarch64::apple_pcie::port0_link_up() || !radio_ready() {
                power_on()?;
            }
            if !radio_ready() {
                return Err(
                    "BAR MMIO not mapped yet — cannot associate (see /wifi info + ktrace)",
                );
            }
            if brcm::with_dev(|d| !d.firmware_up).unwrap_or(true) {
                brcm::try_load_firmware()?;
            }
            return brcm::connect(ssid, psk);
        }
    }
    // Emulator: rename + DHCP (caller runs dhcp after Ok).
    let _ = psk;
    crate::net::set_ifname("wlan0");
    Ok(())
}
