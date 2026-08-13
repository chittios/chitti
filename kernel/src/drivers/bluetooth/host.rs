//! **Classic Bluetooth host** — reset, identity, inquiry, PIN pair, HID open.
//!
//! Talks only through [`crate::arch`] HCI USB helpers so the policy stays arch-
//! neutral. Fail closed: every HCI failure is a string the shell prints.

use super::{bond, hci, hidp, l2cap};
use alloc::string::String;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU16, AtomicU8, Ordering};

static SIG_ID: AtomicU8 = AtomicU8::new(1);
static LOCAL_CID: AtomicU16 = AtomicU16::new(0x40);
static ACL_HANDLE: AtomicU16 = AtomicU16::new(0);
static HID_CTRL_DCID: AtomicU16 = AtomicU16::new(0);
static HID_INTR_DCID: AtomicU16 = AtomicU16::new(0);
static HID_CTRL_SCID: AtomicU16 = AtomicU16::new(0);
static HID_INTR_SCID: AtomicU16 = AtomicU16::new(0);
static PAIRED_ADDR: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

fn next_sig_id() -> u8 {
    SIG_ID.fetch_add(1, Ordering::Relaxed).max(1)
}

fn next_cid() -> u16 {
    let c = LOCAL_CID.fetch_add(1, Ordering::Relaxed);
    if c < 0x40 {
        LOCAL_CID.store(0x41, Ordering::Relaxed);
        0x40
    } else {
        c
    }
}

fn pack_addr(le: &[u8; 6]) -> u64 {
    u64::from_le_bytes([le[0], le[1], le[2], le[3], le[4], le[5], 0, 0])
}

fn unpack_addr(v: u64) -> [u8; 6] {
    let b = v.to_le_bytes();
    [b[0], b[1], b[2], b[3], b[4], b[5]]
}

pub fn on_transport_lost() {
    ACL_HANDLE.store(0, Ordering::Relaxed);
    HID_CTRL_DCID.store(0, Ordering::Relaxed);
    HID_INTR_DCID.store(0, Ordering::Relaxed);
    PAIRED_ADDR.store(0, Ordering::Relaxed);
}

pub fn status_extra() -> Vec<String> {
    let mut v = Vec::new();
    let h = ACL_HANDLE.load(Ordering::Relaxed);
    if h != 0 {
        v.push(alloc::format!("link: ACL handle {h:#x}"));
        let a = unpack_addr(PAIRED_ADDR.load(Ordering::Relaxed));
        if a != [0; 6] {
            v.push(alloc::format!("peer: {}", hci::format_bd_addr(&a)));
        }
        let ic = HID_INTR_DCID.load(Ordering::Relaxed);
        if ic != 0 {
            v.push(alloc::format!("hid: interrupt dcid {ic:#x} (boot reports → console)"));
        } else {
            v.push(String::from("hid: ACL up, L2CAP HID not open — try /bluetooth hid"));
        }
    } else {
        v.push(String::from("link: no ACL connection"));
    }
    v
}

fn cmd(c: &[u8]) -> Result<Vec<u8>, &'static str> {
    crate::arch::bt_hci_cmd(c, 3000).ok_or("HCI command failed or timed out")
}

/// HCI_Reset + read local identity.
pub fn reset_and_info() -> Result<String, &'static str> {
    if !crate::arch::bt_hci_ready() {
        return Err("no HCI USB transport");
    }
    let ret = cmd(&hci::cmd_reset_usb())?;
    if ret.first().copied().unwrap_or(1) != 0 {
        return Err("HCI_Reset status non-zero");
    }
    // Controller needs a settle after reset.
    for _ in 0..50_000 {
        core::hint::spin_loop();
    }
    let name_ret = cmd(&hci::cmd_read_local_name_usb()).unwrap_or_default();
    let name = hci::local_name_from_return(&name_ret).unwrap_or("(unnamed)");
    let bd_ret = cmd(&hci::cmd_read_bd_addr_usb())?;
    let bd = hci::bd_addr_from_return(&bd_ret).ok_or("no BD_ADDR")?;
    let _ = cmd(&hci::cmd_write_scan_enable_usb(0)); // not discoverable by default
    Ok(alloc::format!(
        "reset ok — local '{}'  {}",
        name,
        hci::format_bd_addr(&bd)
    ))
}

/// Inquiry for `length_slots` × 1.28 s (cap 10 ≈ 12.8 s).
pub fn scan(length_slots: u8) -> Result<Vec<hci::InquiryEntry>, &'static str> {
    if !crate::arch::bt_hci_ready() {
        return Err("no HCI USB transport");
    }
    let slots = length_slots.clamp(1, 10);
    // Inquiry returns Command Status, then results, then Inquiry Complete.
    let body = hci::cmd_inquiry_usb(slots, 0);
    // Send without waiting for Command Complete — wait for status.
    let _ = crate::arch::bt_hci_cmd(&body, 500); // may fail if only status returned
    let mut found: Vec<hci::InquiryEntry> = Vec::new();
    let start = crate::arch::now_ms();
    let limit_ms = (slots as u64) * 1280 + 2000;
    let mut buf = [0u8; 256];
    while crate::arch::now_ms().wrapping_sub(start) < limit_ms {
        crate::shell::upkeep();
        if let Some(n) = crate::arch::bt_take_event(&mut buf) {
            if let Some(ev) = hci::parse_event_usb(&buf[..n]) {
                match ev.code {
                    hci::EVT_INQUIRY_RESULT
                    | hci::EVT_INQUIRY_RESULT_WITH_RSSI
                    | hci::EVT_EXTENDED_INQUIRY_RESULT => {
                        // Standard inquiry result layout for first form.
                        if ev.code == hci::EVT_INQUIRY_RESULT {
                            for e in hci::parse_inquiry_result(ev.params) {
                                if !found.iter().any(|f| f.bd_addr == e.bd_addr) {
                                    found.push(e);
                                }
                            }
                        } else if ev.params.len() >= 15 {
                            // RSSI / extended: num + 6 addr + …
                            let mut e = hci::parse_inquiry_result(ev.params);
                            if e.is_empty() && ev.params[0] >= 1 && ev.params.len() >= 15 {
                                let mut bd = [0u8; 6];
                                bd.copy_from_slice(&ev.params[1..7]);
                                e.push(hci::InquiryEntry {
                                    bd_addr: bd,
                                    page_scan_rep_mode: ev.params[7],
                                    class_of_device: u32::from_le_bytes([
                                        ev.params[9],
                                        ev.params[10],
                                        ev.params[11],
                                        0,
                                    ]),
                                });
                            }
                            for ent in e {
                                if !found.iter().any(|f| f.bd_addr == ent.bd_addr) {
                                    found.push(ent);
                                }
                            }
                        }
                    }
                    hci::EVT_INQUIRY_COMPLETE => break,
                    _ => {}
                }
            }
        }
    }
    let _ = crate::arch::bt_hci_cmd(&hci::cmd_inquiry_cancel_usb(), 500);
    Ok(found)
}

/// Create ACL and pair, using Secure Simple Pairing when the peer supports it.
///
/// The PIN is retained strictly as a legacy fallback.  Modern keyboards and
/// mice normally request SSP I/O capabilities followed by numeric comparison
/// or passkey entry; treating those events as unknown is why they previously
/// timed out after connecting.
pub fn pair(addr_str: &str, pin: Option<&str>) -> Result<String, &'static str> {
    if !crate::arch::bt_hci_ready() {
        return Err("no HCI USB transport");
    }
    let bd = hci::parse_bd_addr(addr_str).ok_or("bad address (use AA:BB:CC:DD:EE:FF)")?;
    let pin = pin.unwrap_or("0000");
    if pin.len() > 16 {
        return Err("PIN too long (max 16)");
    }

    // Create Connection → Command Status, then Connection Complete event.
    let _ = crate::arch::bt_hci_cmd(&hci::cmd_create_connection_usb(&bd), 500);
    let start = crate::arch::now_ms();
    let mut handle = 0u16;
    let mut buf = [0u8; 256];
    while crate::arch::now_ms().wrapping_sub(start) < 15_000 {
        crate::shell::upkeep();
        // Answer PIN requests while waiting.
        if let Some(n) = crate::arch::bt_take_event(&mut buf) {
            if let Some(ev) = hci::parse_event_usb(&buf[..n]) {
                match ev.code {
                    hci::EVT_PIN_CODE_REQUEST => {
                        if let Some(req_bd) = hci::parse_pin_code_request(ev.params) {
                            let _ = crate::arch::bt_hci_cmd(
                                &hci::cmd_pin_code_reply_usb(&req_bd, pin.as_bytes()),
                                2000,
                            );
                        }
                    }
                    hci::EVT_CONNECTION_COMPLETE => {
                        if let Some(cc) = hci::parse_connection_complete(ev.params) {
                            if cc.bd_addr == bd {
                                if cc.status != 0 {
                                    return Err("connection failed");
                                }
                                handle = cc.handle;
                                break;
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    if handle == 0 {
        return Err("connection timed out");
    }
    ACL_HANDLE.store(handle, Ordering::Relaxed);
    PAIRED_ADDR.store(pack_addr(&bd), Ordering::Relaxed);

    // Request authentication if not already done.  From here controllers may
    // choose legacy PIN pairing or SSP; handle both paths until completion.
    let _ = crate::arch::bt_hci_cmd(&hci::cmd_auth_requested_usb(handle), 2000);
    let start = crate::arch::now_ms();
    let mut link_key = None;
    let mut used_ssp = false;
    let mut auth_complete = false;
    while crate::arch::now_ms().wrapping_sub(start) < 10_000 {
        crate::shell::upkeep();
        if let Some(n) = crate::arch::bt_take_event(&mut buf) {
            if let Some(ev) = hci::parse_event_usb(&buf[..n]) {
                match ev.code {
                    hci::EVT_PIN_CODE_REQUEST => {
                        if let Some(req_bd) = hci::parse_pin_code_request(ev.params) {
                            let _ = crate::arch::bt_hci_cmd(
                                &hci::cmd_pin_code_reply_usb(&req_bd, pin.as_bytes()), 2000,
                            );
                        }
                    }
                    hci::EVT_IO_CAPABILITY_REQUEST => {
                        let req_bd = hci::parse_bd_addr_event(ev.params).ok_or("malformed SSP I/O-capability request")?;
                        used_ssp = true;
                        let _ = crate::arch::bt_hci_cmd(
                            &hci::cmd_io_capability_reply_usb(
                                &req_bd, hci::IO_CAP_DISPLAY_YES_NO,
                                hci::OOB_DATA_NOT_PRESENT, hci::AUTH_REQ_GENERAL_BONDING_MITM,
                            ), 2000,
                        );
                    }
                    hci::EVT_USER_CONFIRMATION_REQUEST => {
                        let (req_bd, value) = hci::parse_user_confirmation_request(ev.params)
                            .ok_or("malformed SSP numeric-comparison request")?;
                        used_ssp = true;
                        let shown = crate::modal::input(
                            "Bluetooth pairing confirmation",
                            &alloc::format!("Does {} show {:06}? Type yes to approve", hci::format_bd_addr(&req_bd), value),
                            false,
                        );
                        let reply = if shown.trim().eq_ignore_ascii_case("yes") {
                            hci::cmd_user_confirmation_reply_usb(&req_bd)
                        } else {
                            hci::cmd_user_confirmation_neg_reply_usb(&req_bd)
                        };
                        let _ = crate::arch::bt_hci_cmd(&reply, 2000);
                        if !shown.trim().eq_ignore_ascii_case("yes") {
                            return Err("SSP numeric comparison declined");
                        }
                    }
                    hci::EVT_USER_PASSKEY_REQUEST => {
                        let req_bd = hci::parse_bd_addr_event(ev.params).ok_or("malformed SSP passkey request")?;
                        used_ssp = true;
                        let text = crate::modal::input(
                            "Bluetooth passkey",
                            &alloc::format!("Enter the six-digit passkey for {}", hci::format_bd_addr(&req_bd)), true,
                        );
                        let reply = text.parse::<u32>().ok().and_then(|v| hci::cmd_user_passkey_reply_usb(&req_bd, v))
                            .unwrap_or_else(|| hci::cmd_user_passkey_neg_reply_usb(&req_bd));
                        let _ = crate::arch::bt_hci_cmd(&reply, 2000);
                        if text.parse::<u32>().ok().filter(|v| *v <= 999_999).is_none() {
                            return Err("SSP passkey declined or invalid");
                        }
                    }
                    hci::EVT_LINK_KEY_NOTIFICATION if ev.params.len() >= 22 => {
                        if ev.params[..6] == bd {
                            let mut hex = String::new();
                            for b in &ev.params[6..22] { hex.push_str(&alloc::format!("{b:02x}")); }
                            link_key = Some(hex);
                        }
                    }
                    hci::EVT_SIMPLE_PAIRING_COMPLETE => {
                        let (status, peer) = hci::parse_simple_pairing_complete(ev.params)
                            .ok_or("malformed SSP completion event")?;
                        if peer == bd {
                            if status != 0 { return Err("Secure Simple Pairing failed"); }
                            auth_complete = true;
                            break;
                        }
                    }
                    hci::EVT_AUTH_COMPLETE => {
                        if ev.params.first().copied().unwrap_or(1) != 0 { return Err("authentication failed"); }
                        auth_complete = true;
                        if !used_ssp { break; }
                    }
                    _ => {}
                }
            }
        }
    }

    if !auth_complete { return Err("pairing timed out"); }

    let name = addr_str;
    let _ = bond::upsert(&hci::format_bd_addr(&bd), name, link_key.as_deref());
    Ok(alloc::format!(
        "paired {} handle {handle:#x} ({}; bond saved)",
        hci::format_bd_addr(&bd), if used_ssp { "Secure Simple Pairing" } else { "legacy PIN" }
    ))
}

/// Open HID Control + Interrupt L2CAP channels on the current ACL link.
pub fn open_hid() -> Result<String, &'static str> {
    let handle = ACL_HANDLE.load(Ordering::Relaxed);
    if handle == 0 {
        return Err("no ACL — /bluetooth pair first");
    }
    let ctrl_scid = next_cid();
    let intr_scid = next_cid();
    HID_CTRL_SCID.store(ctrl_scid, Ordering::Relaxed);
    HID_INTR_SCID.store(intr_scid, Ordering::Relaxed);

    // Connection request control, then interrupt.
    send_l2cap(handle, &l2cap::connection_request(next_sig_id(), l2cap::PSM_HID_CONTROL, ctrl_scid))?;
    let ctrl_dcid = wait_conn_rsp(ctrl_scid, 5000)?;
    HID_CTRL_DCID.store(ctrl_dcid, Ordering::Relaxed);
    send_l2cap(handle, &l2cap::config_request(next_sig_id(), ctrl_dcid, 0))?;
    let _ = wait_config_rsp(5000);

    send_l2cap(
        handle,
        &l2cap::connection_request(next_sig_id(), l2cap::PSM_HID_INTERRUPT, intr_scid),
    )?;
    let intr_dcid = wait_conn_rsp(intr_scid, 5000)?;
    HID_INTR_DCID.store(intr_dcid, Ordering::Relaxed);
    send_l2cap(handle, &l2cap::config_request(next_sig_id(), intr_dcid, 0))?;
    let _ = wait_config_rsp(5000);

    // SET_PROTOCOL(boot) on control channel.
    let payload = hidp::set_protocol_boot();
    let pdu = l2cap::pdu(ctrl_dcid, &payload);
    send_acl(handle, &pdu)?;

    Ok(alloc::format!(
        "HID channels open (ctrl dcid {ctrl_dcid:#x}, intr dcid {intr_dcid:#x})"
    ))
}

/// Poll ACL for HID interrupt reports → keyboard bytes.
pub fn poll_hid_input() {
    let handle = ACL_HANDLE.load(Ordering::Relaxed);
    let intr = HID_INTR_DCID.load(Ordering::Relaxed);
    if handle == 0 || intr == 0 {
        return;
    }
    let mut buf = [0u8; 1024];
    // Non-blocking-ish short timeout.
    let Some(n) = crate::arch::bt_acl_recv(&mut buf, 5) else {
        return;
    };
    if n < 4 {
        return;
    }
    let Some((h, _pb, dlen)) = hci::parse_acl_header(&buf[..n]) else {
        return;
    };
    if h != handle {
        return;
    }
    let data = &buf[4..n.min(4 + dlen as usize)];
    // May be L2CAP signalling (answer config etc.) or HID data.
    if let Some((cid, payload)) = l2cap::parse_pdu(data) {
        if cid == l2cap::CID_SIGNALING {
            handle_signalling(handle, payload);
            return;
        }
        // Device→host uses our SCID as the CID in their PDUs.
        let our_intr = HID_INTR_SCID.load(Ordering::Relaxed);
        if cid == our_intr || cid == intr {
            if let Some(rep) = hidp::parse_data_input(payload) {
                if hidp::is_boot_keyboard_report(rep) {
                    feed_boot_keyboard(rep);
                }
            } else if hidp::is_boot_keyboard_report(payload) {
                // Some stacks omit HIDP header on interrupt.
                feed_boot_keyboard(payload);
            }
        }
    }
}

fn feed_boot_keyboard(rep: &[u8]) {
    // Reuse USB HID mapping via a tiny local table for a–z / digits / enter.
    if rep.len() < 8 {
        return;
    }
    let shift = rep[0] & 0x22 != 0;
    let ctrl = rep[0] & 0x11 != 0;
    for &u in &rep[2..8] {
        if u == 0 {
            continue;
        }
        if let Some(b) = boot_usage_ascii(u, shift, ctrl) {
            crate::console::unread(b);
        }
    }
}

fn boot_usage_ascii(usage: u8, shift: bool, ctrl: bool) -> Option<u8> {
    let base: u8 = match usage {
        0x04..=0x1d => b'a' + (usage - 0x04),
        0x1e..=0x26 => b'1' + (usage - 0x1e),
        0x27 => b'0',
        0x28 => b'\r',
        0x2a => 0x08,
        0x2b => b'\t',
        0x2c => b' ',
        0x2d => b'-',
        0x2e => b'=',
        _ => return None,
    };
    if ctrl && (b'a'..=b'z').contains(&base) {
        return Some(base - b'a' + 1);
    }
    if shift && (b'a'..=b'z').contains(&base) {
        return Some(base - b'a' + b'A');
    }
    Some(base)
}

fn handle_signalling(handle: u16, payload: &[u8]) {
    let Some(cmd) = l2cap::parse_signalling(payload) else {
        return;
    };
    match cmd.code {
        l2cap::SIG_CONNECTION_REQUEST => {
            if let Some((psm, scid)) = l2cap::parse_connection_request(cmd.data) {
                let dcid = next_cid();
                let rsp = l2cap::connection_response(
                    cmd.id,
                    dcid,
                    scid,
                    l2cap::CONN_SUCCESS,
                    0,
                );
                let _ = send_acl(handle, &rsp);
                if psm == l2cap::PSM_HID_INTERRUPT {
                    HID_INTR_DCID.store(scid, Ordering::Relaxed); // peer's cid for our TX
                    HID_INTR_SCID.store(dcid, Ordering::Relaxed);
                }
                if psm == l2cap::PSM_HID_CONTROL {
                    HID_CTRL_DCID.store(scid, Ordering::Relaxed);
                    HID_CTRL_SCID.store(dcid, Ordering::Relaxed);
                }
            }
        }
        l2cap::SIG_CONFIG_REQUEST => {
            if cmd.data.len() >= 2 {
                let dcid = u16::from_le_bytes([cmd.data[0], cmd.data[1]]);
                let rsp = l2cap::config_response(cmd.id, dcid, 0, 0);
                let _ = send_acl(handle, &rsp);
            }
        }
        _ => {}
    }
}

fn send_l2cap(handle: u16, pdu: &[u8]) -> Result<(), &'static str> {
    send_acl(handle, pdu)
}

fn send_acl(handle: u16, l2cap_pdu: &[u8]) -> Result<(), &'static str> {
    let mut pkt = Vec::with_capacity(4 + l2cap_pdu.len());
    // PB=0b10 (first flushable), BC=00
    pkt.extend_from_slice(&hci::acl_header(handle, 0x2, l2cap_pdu.len() as u16));
    pkt.extend_from_slice(l2cap_pdu);
    if crate::arch::bt_acl_send_sync(&pkt, 2000) {
        Ok(())
    } else {
        Err("ACL send failed")
    }
}

fn wait_conn_rsp(scid: u16, timeout_ms: u64) -> Result<u16, &'static str> {
    let start = crate::arch::now_ms();
    let mut buf = [0u8; 1024];
    while crate::arch::now_ms().wrapping_sub(start) < timeout_ms {
        crate::shell::upkeep();
        if let Some(n) = crate::arch::bt_acl_recv(&mut buf, 100) {
            if n < 4 {
                continue;
            }
            let Some((h, _, dlen)) = hci::parse_acl_header(&buf[..n]) else {
                continue;
            };
            if h != ACL_HANDLE.load(Ordering::Relaxed) {
                continue;
            }
            let data = &buf[4..n.min(4 + dlen as usize)];
            if let Some((cid, payload)) = l2cap::parse_pdu(data) {
                if cid == l2cap::CID_SIGNALING {
                    if let Some(cmd) = l2cap::parse_signalling(payload) {
                        if cmd.code == l2cap::SIG_CONNECTION_RESPONSE {
                            if let Some((dcid, rscid, result, _)) =
                                l2cap::parse_connection_response(cmd.data)
                            {
                                if rscid == scid {
                                    if result != l2cap::CONN_SUCCESS {
                                        return Err("L2CAP connect rejected");
                                    }
                                    return Ok(dcid);
                                }
                            }
                        }
                        if cmd.code == l2cap::SIG_CONFIG_REQUEST {
                            handle_signalling(h, payload);
                        }
                    }
                }
            }
        }
    }
    Err("L2CAP connect timeout")
}

fn wait_config_rsp(timeout_ms: u64) -> Result<(), &'static str> {
    let start = crate::arch::now_ms();
    let mut buf = [0u8; 1024];
    while crate::arch::now_ms().wrapping_sub(start) < timeout_ms {
        crate::shell::upkeep();
        if let Some(n) = crate::arch::bt_acl_recv(&mut buf, 100) {
            if n < 4 {
                continue;
            }
            if let Some((_, _, dlen)) = hci::parse_acl_header(&buf[..n]) {
                let data = &buf[4..n.min(4 + dlen as usize)];
                if let Some((cid, payload)) = l2cap::parse_pdu(data) {
                    if cid == l2cap::CID_SIGNALING {
                        if let Some(cmd) = l2cap::parse_signalling(payload) {
                            if cmd.code == l2cap::SIG_CONFIG_RESPONSE {
                                return Ok(());
                            }
                            if cmd.code == l2cap::SIG_CONFIG_REQUEST {
                                handle_signalling(ACL_HANDLE.load(Ordering::Relaxed), payload);
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(()) // best-effort
}

/// Disconnect ACL if any.
pub fn disconnect() -> Result<(), &'static str> {
    let h = ACL_HANDLE.load(Ordering::Relaxed);
    if h == 0 {
        return Err("no connection");
    }
    let _ = cmd(&hci::cmd_disconnect_usb(h, 0x13)); // remote user terminated
    on_transport_lost();
    Ok(())
}
