//! **Bluetooth (staged)** — USB HCI transport, classic host ops, PIN pairing.
//!
//! ## Stages (hardware plan PR6)
//!
//! 1. **Identify** — USB class `E0/01/01` noted at enumeration.
//! 2. **HCI codec** — [`hci`] pure builders/parsers (unit-tested).
//! 3. **Transport + HID host** — HCI commands via USB class control, events on
//!    interrupt IN, ACL on a dedicated bulk pair ([`usb`] + xHCI `BtHci`);
//!    classic HID opens L2CAP PSM 0x11/0x13 after ACL + auth ([`l2cap`]/[`hidp`]).
//! 4. **Pairing** — PIN modal + durable [`bond`] store under `/configs/core/`.
//!
//! ACL bulk is **separate** from the MSC/Ethernet bulk claim so a stick and a
//! dongle can coexist.

pub mod bond;
pub mod hci;
pub mod hidp;
pub mod host;
pub mod l2cap;
pub mod usb;

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
static TRANSPORT: AtomicBool = AtomicBool::new(false);
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
        "bluetooth: USB BT noted (root port {root_port} slot {slot} {class:02x}/{sub:02x}/{proto:02x})"
    ));
}

/// Called when xHCI finished configuring HCI endpoints.
pub fn note_transport_ready(root_port: u8, slot: u8) {
    TRANSPORT.store(true, Ordering::Release);
    ROOT_PORT.store(root_port, Ordering::Relaxed);
    SLOT.store(slot, Ordering::Relaxed);
    crate::ktrace::log("bluetooth", "HCI USB transport ready");
}

/// Forget BT claim when its root port is unplugged.
pub fn clear_if_port(root_port: u8) {
    if SEEN.load(Ordering::Acquire) && ROOT_PORT.load(Ordering::Relaxed) == root_port {
        SEEN.store(false, Ordering::Release);
        TRANSPORT.store(false, Ordering::Release);
        host::on_transport_lost();
        crate::ktrace::log("bluetooth", "USB BT gone with root port");
    }
}

/// Whether any Bluetooth interface has been noted since boot (or last clear).
pub fn present() -> bool {
    SEEN.load(Ordering::Acquire)
}

/// Whether the HCI USB transport is live (endpoints configured).
pub fn transport_ready() -> bool {
    TRANSPORT.load(Ordering::Acquire) && crate::arch::bt_hci_ready()
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
    if transport_ready() {
        v.push(String::from(
            "hci: USB transport up (class control + interrupt events + ACL bulk)",
        ));
    } else if present() {
        v.push(String::from(
            "hci: interface seen but transport not configured (endpoint layout?)",
        ));
    } else {
        v.push(String::from("hci: idle"));
    }
    for line in host::status_extra() {
        v.push(line);
    }
    let bonds = bond::load();
    v.push(format!("bonds: {} stored ({})", bonds.len(), bond::BOND_PATH));
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
