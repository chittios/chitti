//! **Bluetooth (staged)** — identify USB Bluetooth controllers; pure HCI codec.
//!
//! ## Stages (hardware plan PR6)
//!
//! 1. **Identify** — USB class `E0/01/01` (Wireless Controller / RF / Bluetooth)
//!    noted during xHCI config walk; `/bluetooth status` reports it.
//! 2. **HCI codec** — packet framing, HCI_Reset, Read Local Name (this module's
//!    [`hci`]; unit-tested). No firmware download yet.
//! 3. **Transport + HID host** — HCI over USB bulk/interrupt, then BLE HOGP or
//!    BR/EDR HID (not yet).
//! 4. **Pairing** — human modal + store bond (not yet).
//!
//! A USB Bluetooth dongle must **not** steal the single bulk pair used by MSC
//! or Ethernet; presence is recorded without configuring endpoints.

pub mod hci;

use core::sync::atomic::{AtomicBool, AtomicU8, Ordering};

/// USB class triple for Bluetooth primary controller (USB IF).
pub const USB_CLASS_WIRELESS: u8 = 0xe0;
pub const USB_SUBCLASS_RF: u8 = 0x01;
pub const USB_PROTO_BLUETOOTH: u8 = 0x01;

/// True when an interface matches the Bluetooth USB class triple.
pub fn is_usb_bluetooth(class: u8, subclass: u8, protocol: u8) -> bool {
    class == USB_CLASS_WIRELESS && subclass == USB_SUBCLASS_RF && protocol == USB_PROTO_BLUETOOTH
}

static SEEN: AtomicBool = AtomicBool::new(false);
static ROOT_PORT: AtomicU8 = AtomicU8::new(0);
static SLOT: AtomicU8 = AtomicU8::new(0);
static IFACE_CLASS: AtomicU8 = AtomicU8::new(0);
static IFACE_SUB: AtomicU8 = AtomicU8::new(0);
static IFACE_PROTO: AtomicU8 = AtomicU8::new(0);

/// Record a USB Bluetooth interface found during enumeration.
pub fn note_usb_device(root_port: u8, slot: u8, class: u8, sub: u8, proto: u8) {
    SEEN.store(true, Ordering::Release);
    ROOT_PORT.store(root_port, Ordering::Relaxed);
    SLOT.store(slot, Ordering::Relaxed);
    IFACE_CLASS.store(class, Ordering::Relaxed);
    IFACE_SUB.store(sub, Ordering::Relaxed);
    IFACE_PROTO.store(proto, Ordering::Relaxed);
    crate::ktrace::log_fmt(format_args!(
        "bluetooth: USB BT noted (root port {root_port} slot {slot} {class:02x}/{sub:02x}/{proto:02x}) — identify only, no HCI transport yet"
    ));
}

/// Forget BT claim when its root port is unplugged.
pub fn clear_if_port(root_port: u8) {
    if SEEN.load(Ordering::Acquire) && ROOT_PORT.load(Ordering::Relaxed) == root_port {
        SEEN.store(false, Ordering::Release);
        crate::ktrace::log("bluetooth", "USB BT gone with root port");
    }
}

/// Whether any Bluetooth interface has been noted since boot (or last clear).
pub fn present() -> bool {
    SEEN.load(Ordering::Acquire)
}

/// Snapshot for `/bluetooth status`.
pub fn status_lines() -> alloc::vec::Vec<alloc::string::String> {
    use alloc::format;
    use alloc::string::String;
    let mut v = alloc::vec::Vec::new();
    if present() {
        v.push(format!(
            "usb: present (root port {}, slot {}, class {:02x}/{:02x}/{:02x})",
            ROOT_PORT.load(Ordering::Relaxed),
            SLOT.load(Ordering::Relaxed),
            IFACE_CLASS.load(Ordering::Relaxed),
            IFACE_SUB.load(Ordering::Relaxed),
            IFACE_PROTO.load(Ordering::Relaxed),
        ));
    } else {
        v.push(String::from("usb: no Bluetooth interface noted at enumeration"));
    }
    v.push(String::from(
        "hci: codec ready (HCI_Reset / Read Local Name pure); USB transport not wired",
    ));
    v.push(String::from(
        "next: HCI over USB bulk/interrupt, then BLE HOGP or BR/EDR HID host",
    ));
    // Prove the pure codec is linked and self-consistent.
    let reset = hci::cmd_reset();
    v.push(format!(
        "hci sample: Reset command {} bytes (ogf={:#x} ocf={:#x})",
        reset.len(),
        hci::OGF_CONTROLLER_BASEBAND,
        hci::OCF_RESET
    ));
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn usb_bt_class_triple_is_exact() {
        assert!(is_usb_bluetooth(0xe0, 0x01, 0x01));
        assert!(!is_usb_bluetooth(0xe0, 0x01, 0x00)); // not Bluetooth proto
        assert!(!is_usb_bluetooth(0x08, 0x06, 0x50)); // MSC
        assert!(!is_usb_bluetooth(0x0e, 0x01, 0x00)); // UVC
    }
}
