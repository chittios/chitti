//! **Bond store** — durable link keys / PIN history for classic pairing.
//!
//! Stored as a simple text file under the Synapse store so a reboot keeps
//! pairings the human already approved. Format (one bond per line):
//!
//! ```text
//! AA:BB:CC:DD:EE:FF name_with_underscores
//! ```
//!
//! Link keys are not always available without SSP; we store the address and
//! display name so `/bluetooth bonds` and auto-reconnect hints work. When a
//! link key is obtained later it can be appended as a third field (hex).

use alloc::string::{String, ToString};
use alloc::vec::Vec;

pub const BOND_PATH: &str = "/configs/core/bluetooth_bonds.txt";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Bond {
    /// Display form `AA:BB:…`.
    pub addr: String,
    pub name: String,
    /// Optional 16-byte link key as 32 hex chars.
    pub link_key_hex: Option<String>,
}

/// Parse the bond file body (pure).
pub fn parse_bonds(text: &str) -> Vec<Bond> {
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.split_whitespace();
        let Some(addr) = parts.next() else { continue };
        if crate::drivers::bluetooth::hci::parse_bd_addr(addr).is_none() {
            continue;
        }
        let name = parts.next().unwrap_or("?").replace('_', " ");
        let link_key_hex = parts.next().map(|s| s.to_string());
        out.push(Bond {
            addr: addr.to_ascii_uppercase(),
            name,
            link_key_hex,
        });
    }
    out
}

/// Serialise bonds for write-back.
pub fn format_bonds(bonds: &[Bond]) -> String {
    let mut s = String::from("# ChittiOS Bluetooth bonds — address name [link_key_hex]\n");
    for b in bonds {
        let name = b.name.replace(' ', "_");
        s.push_str(&b.addr);
        s.push(' ');
        s.push_str(&name);
        if let Some(ref k) = b.link_key_hex {
            s.push(' ');
            s.push_str(k);
        }
        s.push('\n');
    }
    s
}

/// Load bonds from the store (empty if missing).
pub fn load() -> Vec<Bond> {
    match crate::synapse::fs::read(BOND_PATH) {
        Some(bytes) => {
            let text = core::str::from_utf8(&bytes).unwrap_or("");
            parse_bonds(text)
        }
        None => Vec::new(),
    }
}

/// Save the full bond list (overwrite).
pub fn save(bonds: &[Bond]) -> Result<(), &'static str> {
    let body = format_bonds(bonds);
    crate::synapse::fs::write(BOND_PATH, body.as_bytes());
    Ok(())
}

/// Upsert one bond by address.
pub fn upsert(addr: &str, name: &str, link_key_hex: Option<&str>) -> Result<(), &'static str> {
    let addr = addr.to_ascii_uppercase();
    let mut bonds = load();
    if let Some(b) = bonds.iter_mut().find(|b| b.addr == addr) {
        b.name = name.to_string();
        if link_key_hex.is_some() {
            b.link_key_hex = link_key_hex.map(|s| s.to_string());
        }
    } else {
        bonds.push(Bond {
            addr,
            name: name.to_string(),
            link_key_hex: link_key_hex.map(|s| s.to_string()),
        });
    }
    save(&bonds)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn parse_and_format_roundtrip() {
        let text = "# comment\nAA:BB:CC:DD:EE:FF My_Keyboard\n11:22:33:44:55:66 Mouse deadbeef\n";
        let b = parse_bonds(text);
        assert_eq!(b.len(), 2);
        assert_eq!(b[0].name, "My Keyboard");
        assert_eq!(b[1].link_key_hex.as_deref(), Some("deadbeef"));
        let again = parse_bonds(&format_bonds(&b));
        assert_eq!(again.len(), 2);
        assert_eq!(again[0].addr, "AA:BB:CC:DD:EE:FF");
    }

    #[test_case]
    fn rejects_bad_address_lines() {
        assert!(parse_bonds("not-an-addr foo\n").is_empty());
    }
}
