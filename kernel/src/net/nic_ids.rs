//! **Which driver claims a given PCI NIC** — pure vendor/device-ID → driver
//! classification, no MMIO, unit-tested off-hardware.
//!
//! This exists because matching on *vendor alone* is wrong on real hardware in a
//! way emulators never expose. Every Intel Ethernet controller reports vendor
//! `0x8086` and PCI class `02:00:00`, but the register layout and descriptor
//! format split into four incompatible families:
//!
//! | family | ring registers | descriptors | typical machine |
//! |---|---|---|---|
//! | [`NicKind::E1000`] (82540-82547) | `RDBAL 0x2800` / `TDBAL 0x3800` | legacy | QEMU default, VirtualBox, ~2003 PCI cards |
//! | [`NicKind::E1000e`] (82571-I219) | same as above | legacy | **most business laptops** (I217/I218/I219) |
//! | [`NicKind::Igb`] (82575-I350) | `RDBAL 0xC000` / `TDBAL 0xE000` | advanced | servers, I210/I211 desktop boards |
//! | [`NicKind::Igc`] (I225/I226) | as igb, different init | advanced | 2.5GbE desktop boards, 2020+ |
//!
//! Driving an I219 with the legacy-e1000 init happens to *mostly* work (the ring
//! registers are at the same offsets), which is why the bug was invisible; an
//! igb/igc has its rings at completely different offsets and would simply never
//! receive a frame.
//!
//! ## The unknown-ID policy
//! The legacy `e1000`, `igb` and `igc` ID sets are **closed** — those families
//! are finished, no new IDs will appear. The `e1000e` set is **not**: Intel adds
//! I219 IDs with every new PCH generation (there are already ~40, and an
//! exhaustive list here would be stale within a year). So the tables below
//! enumerate the closed families exhaustively and an unrecognised Intel Ethernet
//! device **defaults to [`NicKind::E1000e`]** — the overwhelmingly likely family
//! for anything newer than what we can name — with its ID logged so it can be
//! reported and pinned down. Attempting the most probable driver and saying so
//! beats refusing to bring the NIC up at all.

/// The driver family that should claim a PCI network function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NicKind {
    /// Intel 82540/82544/82545/82546-class legacy gigabit — [`super::e1000`].
    E1000,
    /// Intel PCIe gigabit, 82571 through I219 — [`super::e1000`] in e1000e mode.
    E1000e,
    /// Intel 82575/82576/82580/I210/I211/I350 — [`super::igb`], advanced descriptors.
    Igb,
    /// Intel I225/I226 2.5GbE — [`super::igb`] in igc mode.
    Igc,
    /// Realtek RTL8139 — [`super::rtl8139`].
    Rtl8139,
    /// Realtek RTL8168/8111/8101/8125 — [`super::r8169`].
    R8169,
    /// virtio-net over PCI — [`super::virtio_net_pci`].
    VirtioNet,
}

impl NicKind {
    /// A short human name for boot logs / `/network`.
    pub fn name(self) -> &'static str {
        match self {
            NicKind::E1000 => "e1000",
            NicKind::E1000e => "e1000e",
            NicKind::Igb => "igb",
            NicKind::Igc => "igc",
            NicKind::Rtl8139 => "rtl8139",
            NicKind::R8169 => "r8169",
            NicKind::VirtioNet => "virtio-net",
        }
    }
}

/// PCI vendor IDs.
pub const VENDOR_INTEL: u16 = 0x8086;
pub const VENDOR_REALTEK: u16 = 0x10ec;
pub const VENDOR_VIRTIO: u16 = 0x1af4;

/// Intel **legacy e1000** (Linux `e1000`): 82542 … 82547. Closed set.
const INTEL_E1000: &[u16] = &[
    0x1000, 0x1001, 0x1004, 0x1008, 0x1009, 0x100c, 0x100d, 0x100e, 0x100f, 0x1010, 0x1011, 0x1012,
    0x1013, 0x1014, 0x1015, 0x1016, 0x1017, 0x1018, 0x1019, 0x101a, 0x101d, 0x101e, 0x1026, 0x1027,
    0x1028, 0x1075, 0x1076, 0x1077, 0x1078, 0x1079, 0x107a, 0x107b, 0x107c, 0x108a, 0x1099, 0x10b5,
];

/// Intel **igb** (Linux `igb`): 82575/82576/82580/I210/I211/I350. Closed set.
/// Ring registers at `0xC000`/`0xE000`, advanced descriptors — driving one of
/// these as an e1000 silently receives nothing.
const INTEL_IGB: &[u16] = &[
    0x10a7, 0x10a9, 0x10c9, 0x10d6, 0x10e6, 0x10e7, 0x10e8, 0x1518, 0x150a, 0x150d, 0x1526, 0x150e,
    0x150f, 0x1510, 0x1511, 0x1516, 0x1527, 0x1521, 0x1522, 0x1523, 0x1524, 0x1533, 0x1536, 0x1537,
    0x1538, 0x1539, 0x157b, 0x157c,
];

/// Intel **igc** (Linux `igc`): I225/I226 2.5GbE. Closed set.
const INTEL_IGC: &[u16] = &[
    0x0d9f, 0x125b, 0x125c, 0x125d, 0x125e, 0x125f, 0x15f2, 0x15f3, 0x15f4, 0x15f5, 0x15f8, 0x3100,
    0x3101, 0x5502, 0x5503,
];

/// Realtek **RTL8139**-class fast ethernet (Linux `8139too`).
const REALTEK_8139: &[u16] = &[0x8129, 0x8138, 0x8139];

/// Realtek **r8169**-class gigabit/2.5G (Linux `r8169`): RTL8169/8168/8111/8101/
/// 8125. `0x8168` is the single most common Ethernet controller in consumer PCs.
/// The ids dispatched to [`super::r8169`], so that driver's own tests can assert every
/// one of them lands in a known register layout rather than a default that suits neither
/// generation.
pub fn realtek_r8169_ids() -> &'static [u16] {
    REALTEK_R8169
}

const REALTEK_R8169: &[u16] = &[0x3000, 0x8125, 0x8136, 0x8161, 0x8162, 0x8167, 0x8168, 0x8169];

/// Classify a PCI network function by vendor/device ID.
///
/// `None` means "no driver here" — the caller should keep scanning for another
/// NIC rather than claiming this one. An unrecognised **Intel** Ethernet device
/// resolves to [`NicKind::E1000e`] by policy (see the module docs), so `None` is
/// returned only for vendors we have no driver for at all (Broadcom `tg3`,
/// Atheros/Killer `alx`, Marvell, Aquantia, VMware `vmxnet3`).
pub fn driver_for(vendor: u16, device: u16) -> Option<NicKind> {
    match vendor {
        VENDOR_INTEL => Some(if INTEL_E1000.contains(&device) {
            NicKind::E1000
        } else if INTEL_IGB.contains(&device) {
            NicKind::Igb
        } else if INTEL_IGC.contains(&device) {
            NicKind::Igc
        } else {
            // Unknown Intel Ethernet: assume the open-ended family. See docs.
            NicKind::E1000e
        }),
        VENDOR_REALTEK => {
            if REALTEK_8139.contains(&device) {
                Some(NicKind::Rtl8139)
            } else if REALTEK_R8169.contains(&device) {
                Some(NicKind::R8169)
            } else {
                None
            }
        }
        // virtio-net: transitional (0x1000) and modern (0x1041) device IDs.
        VENDOR_VIRTIO if device == 0x1000 || device == 0x1041 => Some(NicKind::VirtioNet),
        _ => None,
    }
}

/// True when [`driver_for`] resolved by the unknown-Intel fallback rather than an
/// explicit table hit — the probe logs this so an unsupported ID is reportable
/// from one boot log instead of looking like a working NIC that never receives.
pub fn is_intel_guess(vendor: u16, device: u16) -> bool {
    vendor == VENDOR_INTEL
        && !INTEL_E1000.contains(&device)
        && !INTEL_IGB.contains(&device)
        && !INTEL_IGC.contains(&device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn intel_families_do_not_overlap() {
        // The three closed sets must be disjoint — an ID in two of them would
        // make `driver_for`'s if/else-if order silently decide the family.
        for &d in INTEL_E1000 {
            assert!(!INTEL_IGB.contains(&d), "e1000/igb overlap at {d:#06x}");
            assert!(!INTEL_IGC.contains(&d), "e1000/igc overlap at {d:#06x}");
        }
        for &d in INTEL_IGB {
            assert!(!INTEL_IGC.contains(&d), "igb/igc overlap at {d:#06x}");
        }
        for &d in REALTEK_8139 {
            assert!(!REALTEK_R8169.contains(&d), "realtek overlap at {d:#06x}");
        }
    }

    #[test_case]
    fn qemu_models_map_to_their_drivers() {
        // Every NIC QEMU can emulate — these are the ones we can actually test.
        assert_eq!(driver_for(0x8086, 0x100e), Some(NicKind::E1000)); // -device e1000
        assert_eq!(driver_for(0x8086, 0x100f), Some(NicKind::E1000)); // e1000-82545em
        assert_eq!(driver_for(0x8086, 0x1008), Some(NicKind::E1000)); // e1000-82544gc
        assert_eq!(driver_for(0x8086, 0x10d3), Some(NicKind::E1000e)); // -device e1000e (82574L)
        assert_eq!(driver_for(0x8086, 0x10c9), Some(NicKind::Igb)); // -device igb (82576)
        assert_eq!(driver_for(0x10ec, 0x8139), Some(NicKind::Rtl8139)); // -device rtl8139
        assert_eq!(driver_for(0x1af4, 0x1000), Some(NicKind::VirtioNet)); // virtio-net-pci
        assert_eq!(driver_for(0x1af4, 0x1041), Some(NicKind::VirtioNet)); // modern virtio
    }

    #[test_case]
    fn real_laptop_nics_map_to_their_drivers() {
        // The controllers actually found in machines people would install on.
        assert_eq!(driver_for(0x8086, 0x1502), Some(NicKind::E1000e)); // 82579LM
        assert_eq!(driver_for(0x8086, 0x153a), Some(NicKind::E1000e)); // I217-LM
        assert_eq!(driver_for(0x8086, 0x155a), Some(NicKind::E1000e)); // I218-LM
        assert_eq!(driver_for(0x8086, 0x156f), Some(NicKind::E1000e)); // I219-LM
        assert_eq!(driver_for(0x8086, 0x1570), Some(NicKind::E1000e)); // I219-V
        assert_eq!(driver_for(0x8086, 0x1533), Some(NicKind::Igb)); // I210
        assert_eq!(driver_for(0x8086, 0x1539), Some(NicKind::Igb)); // I211
        assert_eq!(driver_for(0x8086, 0x15f3), Some(NicKind::Igc)); // I225-V
        assert_eq!(driver_for(0x8086, 0x125c), Some(NicKind::Igc)); // I226-V
        assert_eq!(driver_for(0x10ec, 0x8168), Some(NicKind::R8169)); // RTL8111/8168
        assert_eq!(driver_for(0x10ec, 0x8125), Some(NicKind::R8169)); // RTL8125 2.5G
    }

    #[test_case]
    fn legacy_e1000_ids_are_not_driven_as_e1000e() {
        // The regression guard for the bug this table fixes: an 82540EM must NOT
        // fall through to the e1000e path, and an I210 must NOT be claimed by the
        // legacy e1000 driver (its rings live at 0xC000, not 0x2800).
        assert_eq!(driver_for(0x8086, 0x100e), Some(NicKind::E1000));
        assert_ne!(driver_for(0x8086, 0x1533), Some(NicKind::E1000));
        assert_ne!(driver_for(0x8086, 0x15f3), Some(NicKind::E1000));
    }

    #[test_case]
    fn unknown_intel_guesses_e1000e_and_is_flagged() {
        // A future I219 successor we have never seen: claimed as e1000e, and
        // `is_intel_guess` marks it so the probe can log the unknown ID.
        assert_eq!(driver_for(0x8086, 0xabcd), Some(NicKind::E1000e));
        assert!(is_intel_guess(0x8086, 0xabcd));
        // Known IDs are never flagged as guesses.
        assert!(!is_intel_guess(0x8086, 0x100e));
        assert!(!is_intel_guess(0x8086, 0x1533));
        assert!(!is_intel_guess(0x8086, 0x15f3));
        // Non-Intel is never a "guess" — those tables are exact-match only.
        assert!(!is_intel_guess(0x10ec, 0x8168));
    }

    #[test_case]
    fn vendors_without_a_driver_return_none() {
        // Must be None, not a wrong-driver guess: claiming these would take the
        // NIC away from a later working one and then never receive a frame.
        assert_eq!(driver_for(0x14e4, 0x1682), None); // Broadcom tg3
        assert_eq!(driver_for(0x1969, 0xe0b1), None); // Atheros/Killer alx
        assert_eq!(driver_for(0x15ad, 0x07b0), None); // VMware vmxnet3
        assert_eq!(driver_for(0x1d6a, 0x94c0), None); // Aquantia
        assert_eq!(driver_for(0x10ec, 0x5289), None); // Realtek, but a card reader
        assert_eq!(driver_for(0x1af4, 0x1042), None); // virtio, but a block device
    }
}
