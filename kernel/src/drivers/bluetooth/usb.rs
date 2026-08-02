//! **USB Bluetooth endpoint layout** — pure config-descriptor walk.
//!
//! A primary controller interface (`E0/01/01`) exposes:
//! - **Interrupt IN** — HCI events
//! - **Bulk IN + Bulk OUT** — ACL data
//!
//! HCI commands use the **default control pipe** (class request), not bulk.

/// Endpoints for one Bluetooth HCI interface.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BtEndpoints {
    pub iface: u8,
    pub evt_ep: u8,
    pub evt_mps: u16,
    pub evt_interval: u8,
    pub acl_in_ep: u8,
    pub acl_out_ep: u8,
    pub acl_in_mps: u16,
    pub acl_out_mps: u16,
}

/// Locate the first Bluetooth interface with interrupt IN + bulk IN/OUT.
pub fn find_bt_endpoints(desc: &[u8]) -> Option<BtEndpoints> {
    const DT_INTERFACE: u8 = 0x04;
    const DT_ENDPOINT: u8 = 0x05;
    let mut i = 0usize;
    let mut cur_iface = 0u8;
    let mut match_iface = false;
    let mut evt_ep = 0u8;
    let mut evt_mps = 0u16;
    let mut evt_ivl = 0u8;
    let mut acl_in = 0u8;
    let mut acl_out = 0u8;
    let mut acl_in_mps = 0u16;
    let mut acl_out_mps = 0u16;

    let try_done = |evt_ep: u8,
                    evt_mps: u16,
                    evt_ivl: u8,
                    acl_in: u8,
                    acl_out: u8,
                    acl_in_mps: u16,
                    acl_out_mps: u16,
                    cur_iface: u8|
     -> Option<BtEndpoints> {
        if evt_ep != 0 && acl_in != 0 && acl_out != 0 {
            Some(BtEndpoints {
                iface: cur_iface,
                evt_ep,
                evt_mps: evt_mps.max(16),
                evt_interval: evt_ivl.max(1),
                acl_in_ep: acl_in,
                acl_out_ep: acl_out,
                acl_in_mps: acl_in_mps.max(64),
                acl_out_mps: acl_out_mps.max(64),
            })
        } else {
            None
        }
    };

    while i + 2 <= desc.len() {
        let len = desc[i] as usize;
        if len < 2 || i + len > desc.len() {
            break;
        }
        match desc[i + 1] {
            DT_INTERFACE if len >= 9 => {
                if let Some(ep) = try_done(
                    evt_ep, evt_mps, evt_ivl, acl_in, acl_out, acl_in_mps, acl_out_mps, cur_iface,
                ) {
                    return Some(ep);
                }
                cur_iface = desc[i + 2];
                let class = desc[i + 5];
                let sub = desc[i + 6];
                let proto = desc[i + 7];
                match_iface = super::is_usb_bluetooth(class, sub, proto);
                evt_ep = 0;
                acl_in = 0;
                acl_out = 0;
            }
            DT_ENDPOINT if len >= 7 && match_iface => {
                let addr = desc[i + 2];
                let attrs = desc[i + 3];
                let mps = u16::from_le_bytes([desc[i + 4], desc[i + 5]]);
                let ivl = desc[i + 6];
                let xfer = attrs & 0x03;
                if xfer == 0x03 && addr & 0x80 != 0 && evt_ep == 0 {
                    // Interrupt IN
                    evt_ep = addr;
                    evt_mps = mps;
                    evt_ivl = ivl;
                } else if xfer == 0x02 {
                    if addr & 0x80 != 0 {
                        acl_in = addr;
                        acl_in_mps = mps;
                    } else {
                        acl_out = addr;
                        acl_out_mps = mps;
                    }
                }
            }
            _ => {}
        }
        i += len;
    }
    try_done(
        evt_ep, evt_mps, evt_ivl, acl_in, acl_out, acl_in_mps, acl_out_mps, cur_iface,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn iface(n: u8, class: u8, sub: u8, proto: u8) -> [u8; 9] {
        [9, 0x04, n, 0, 3, class, sub, proto, 0]
    }
    fn ep(addr: u8, attrs: u8, mps: u16, ivl: u8) -> [u8; 7] {
        [
            7,
            0x05,
            addr,
            attrs,
            mps.to_le_bytes()[0],
            mps.to_le_bytes()[1],
            ivl,
        ]
    }

    #[test_case]
    fn finds_classic_bt_usb_layout() {
        let mut d = alloc::vec::Vec::new();
        d.extend_from_slice(&iface(0, 0xe0, 0x01, 0x01));
        d.extend_from_slice(&ep(0x81, 0x03, 16, 1)); // interrupt IN
        d.extend_from_slice(&ep(0x02, 0x02, 64, 0)); // bulk OUT
        d.extend_from_slice(&ep(0x82, 0x02, 64, 0)); // bulk IN
        let e = find_bt_endpoints(&d).unwrap();
        assert_eq!(e.iface, 0);
        assert_eq!(e.evt_ep, 0x81);
        assert_eq!(e.acl_out_ep, 0x02);
        assert_eq!(e.acl_in_ep, 0x82);
    }

    #[test_case]
    fn ignores_hid_and_incomplete_bt() {
        let mut d = alloc::vec::Vec::new();
        d.extend_from_slice(&iface(0, 0x03, 0x01, 0x01));
        d.extend_from_slice(&ep(0x81, 0x03, 8, 1));
        assert!(find_bt_endpoints(&d).is_none());
        // BT with only interrupt — not enough
        let mut d2 = alloc::vec::Vec::new();
        d2.extend_from_slice(&iface(0, 0xe0, 0x01, 0x01));
        d2.extend_from_slice(&ep(0x81, 0x03, 16, 1));
        assert!(find_bt_endpoints(&d2).is_none());
    }
}
