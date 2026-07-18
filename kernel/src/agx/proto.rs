//! Apple **RTKit** / **ASC-mailbox** wire protocol — the *pure* half of the AGX
//! GPU coprocessor bring-up. Every message the AGX `gfx-asc` exchanges over the
//! ASC mailbox is a `{msg0: u64, msg1: u32}` pair; `msg1` is the endpoint and
//! `msg0` packs a management/endpoint message in bitfields. This module is the
//! bit-twiddling: field encode/decode, HELLO version negotiation, the endpoint
//! map, buffer-request parsing, and the small received-message → action state
//! machine.
//!
//! It is **arch-neutral and side-effect-free** so it lives outside
//! `arch/aarch64/` and is exercised by the host unit suite (`cargo xtask test`,
//! x86 — where `arch::aarch64` is not even compiled). The hardware half (MMIO
//! FIFO, `cpu_start`, DMA-buffer allocation, the cooperative pump) lives in
//! `agx::asc` + `agx::mod` and is gated to aarch64.
//!
//! Field positions and message constants are taken verbatim from m1n1's
//! `src/rtkit.c` (`rtkit_boot`, `rtkit_recv`, `rtkit_handle_buffer_request`),
//! vendored at `third_party/m1n1/src/rtkit.c`.

#![allow(dead_code)] // a few constants document the wire format but aren't read

// --- RTKit system endpoints (rtkit.c:22-27) ------------------------------
pub const EP_MGMT: u8 = 0;
pub const EP_CRASHLOG: u8 = 1;
pub const EP_SYSLOG: u8 = 2;
pub const EP_DEBUG: u8 = 3;
pub const EP_IOREPORT: u8 = 4;
pub const EP_OSLOG: u8 = 8;
/// The first *app* (non-system) endpoint — messages here are forwarded to the
/// caller rather than handled as management traffic.
pub const EP_APP_BASE: u8 = 0x20;

// --- management message types, in MGMT_TYPE (rtkit.c) --------------------
pub const MGMT_MSG_HELLO: u64 = 1;
pub const MGMT_MSG_HELLO_ACK: u64 = 2;
pub const MGMT_MSG_START_EP: u64 = 5;
pub const MGMT_MSG_IOP_PWR_STATE: u64 = 6;
pub const MGMT_MSG_IOP_PWR_STATE_ACK: u64 = 7;
pub const MGMT_MSG_EPMAP: u64 = 8;
pub const MGMT_MSG_AP_PWR_STATE: u64 = 0xb;
/// The per-endpoint "give me a shared buffer" request (syslog/crashlog/ioreport).
pub const MSG_BUFFER_REQUEST: u64 = 1;
/// syslog "here is my ring geometry" init message.
pub const MSG_SYSLOG_INIT: u64 = 8;
/// syslog "a line was written" — must be ACKed by echoing the message back.
pub const MSG_SYSLOG_LOG: u64 = 5;

// --- RTKit power states (rtkit.c:75-81) ----------------------------------
pub const POWER_OFF: u64 = 0x00;
pub const POWER_SLEEP: u64 = 0x01;
pub const POWER_QUIESCED: u64 = 0x10;
pub const POWER_ON: u64 = 0x20;
pub const POWER_INIT: u64 = 0x220;

// --- version window we advertise (rtkit.c:70-71) -------------------------
pub const RTKIT_MIN_VERSION: u64 = 11;
pub const RTKIT_MAX_VERSION: u64 = 12;

/// Inclusive bit-mask `[hi:lo]` (m1n1's `GENMASK`). Pure `const`.
pub const fn gen_mask(hi: u32, lo: u32) -> u64 {
    let width = hi - lo + 1;
    (if width == 64 { u64::MAX } else { (1u64 << width) - 1 }) << lo
}

/// Extract the field `[hi:lo]` from `v` (m1n1's `FIELD_GET`). Pure.
#[inline]
pub const fn field_get(v: u64, hi: u32, lo: u32) -> u64 {
    (v & gen_mask(hi, lo)) >> lo
}

/// Place `val` into the field `[hi:lo]` (m1n1's `FIELD_PREP`). Pure.
#[inline]
pub const fn field_prep(val: u64, hi: u32, lo: u32) -> u64 {
    (val << lo) & gen_mask(hi, lo)
}

// Field spans (hi, lo) — named to mirror the m1n1 `#define`s.
const MGMT_TYPE: (u32, u32) = (59, 52);
const MGMT_PWR_STATE: (u32, u32) = (15, 0);
const HELLO_MINVER: (u32, u32) = (15, 0);
const HELLO_MAXVER: (u32, u32) = (31, 16);
const EPMAP_BASE: (u32, u32) = (34, 32);
const EPMAP_BITMAP: (u32, u32) = (31, 0);
const EPMAP_DONE_BIT: u32 = 51;
const EPMAP_REPLY_MORE_BIT: u32 = 0;
const START_EP_IDX: (u32, u32) = (39, 32);
const START_EP_FLAG_BIT: u32 = 1;
const BUFFER_REQUEST_SIZE: (u32, u32) = (51, 44); // in 4 KiB pages
// The buffer DVA field. Generic RTKit uses [41:0], but the AGX crashlog message
// (crash.py `CrashLogMessage.DVA`) is [43:0] — needed for high-half TTBR1 kernel
// VAs (e.g. 0xfae00000000, whose bits [43:40] are set). Use the wider field.
const BUFFER_REQUEST_IOVA: (u32, u32) = (43, 0);

/// The management/endpoint message type in `msg0` (bits [59:52]).
#[inline]
pub fn mgmt_type(msg0: u64) -> u64 {
    field_get(msg0, MGMT_TYPE.0, MGMT_TYPE.1)
}

/// The power-state value carried by an IOP/AP power ACK (bits [15:0]).
#[inline]
pub fn pwr_state(msg0: u64) -> u64 {
    field_get(msg0, MGMT_PWR_STATE.0, MGMT_PWR_STATE.1)
}

/// HELLO's advertised `(min_ver, max_ver)` version window.
#[inline]
pub fn hello_versions(msg0: u64) -> (u64, u64) {
    (field_get(msg0, HELLO_MINVER.0, HELLO_MINVER.1), field_get(msg0, HELLO_MAXVER.0, HELLO_MAXVER.1))
}

/// Negotiate the RTKit version: `min(MAX, iop_max)` if our `[MIN,MAX]` window
/// overlaps the IOP's `[min,max]`, else `None` (incompatible — abort boot).
/// Mirrors `rtkit_boot`'s check + `want_ver` (rtkit.c:529-538).
pub fn negotiate_version(iop_min: u64, iop_max: u64) -> Option<u64> {
    if iop_min > RTKIT_MAX_VERSION || iop_max < RTKIT_MIN_VERSION {
        return None;
    }
    Some(if iop_max < RTKIT_MAX_VERSION { iop_max } else { RTKIT_MAX_VERSION })
}

/// Decoded endpoint-map message: the 32-bit presence `bitmap` at `base*32`,
/// and whether this is the final EPMAP fragment (`done`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EpMap {
    pub bitmap: u32,
    pub base: u64,
    pub done: bool,
}

/// Decode an EPMAP message (`msg0`) from EP_MGMT (rtkit.c:576-608).
pub fn epmap(msg0: u64) -> EpMap {
    EpMap {
        bitmap: field_get(msg0, EPMAP_BITMAP.0, EPMAP_BITMAP.1) as u32,
        base: field_get(msg0, EPMAP_BASE.0, EPMAP_BASE.1),
        done: msg0 & (1u64 << EPMAP_DONE_BIT) != 0,
    }
}

/// A parsed buffer request: `n_pages` × 4 KiB, and a preallocated `addr` (0 =
/// the IOP wants *us* to allocate and reply with the address).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferRequest {
    pub n_pages: u64,
    pub addr: u64,
}

/// Decode a per-endpoint buffer request (rtkit.c:282-284).
pub fn buffer_request(msg0: u64) -> BufferRequest {
    BufferRequest {
        n_pages: field_get(msg0, BUFFER_REQUEST_SIZE.0, BUFFER_REQUEST_SIZE.1),
        addr: field_get(msg0, BUFFER_REQUEST_IOVA.0, BUFFER_REQUEST_IOVA.1),
    }
}

// --- message builders (all target EP_MGMT unless noted) ------------------

/// `IOP_PWR_STATE` with `state` (rtkit.c:503-505) — wake/init the IOP.
pub fn msg_iop_pwr_state(state: u64) -> u64 {
    field_prep(MGMT_MSG_IOP_PWR_STATE, MGMT_TYPE.0, MGMT_TYPE.1)
        | field_prep(state, MGMT_PWR_STATE.0, MGMT_PWR_STATE.1)
}

/// `AP_PWR_STATE` with `state` (rtkit.c:648-649) — e.g. → ON to reach RUNNING.
pub fn msg_ap_pwr_state(state: u64) -> u64 {
    field_prep(MGMT_MSG_AP_PWR_STATE, MGMT_TYPE.0, MGMT_TYPE.1)
        | field_prep(state, MGMT_PWR_STATE.0, MGMT_PWR_STATE.1)
}

/// `HELLO_ACK` echoing the negotiated `ver` in both min/max (rtkit.c:542-544).
pub fn msg_hello_ack(ver: u64) -> u64 {
    field_prep(MGMT_MSG_HELLO_ACK, MGMT_TYPE.0, MGMT_TYPE.1)
        | field_prep(ver, HELLO_MINVER.0, HELLO_MINVER.1)
        | field_prep(ver, HELLO_MAXVER.0, HELLO_MAXVER.1)
}

/// `EPMAP` reply for fragment `base`; sets DONE when we've seen the last
/// fragment, else MORE (rtkit.c:610-615).
pub fn msg_epmap_reply(base: u64, done: bool) -> u64 {
    let mut m = field_prep(MGMT_MSG_EPMAP, MGMT_TYPE.0, MGMT_TYPE.1) | field_prep(base, EPMAP_BASE.0, EPMAP_BASE.1);
    if done {
        m |= 1u64 << EPMAP_DONE_BIT;
    } else {
        m |= 1u64 << EPMAP_REPLY_MORE_BIT;
    }
    m
}

/// `START_EP` for endpoint `ep` (rtkit.c:483-485).
pub fn msg_start_ep(ep: u8) -> u64 {
    field_prep(MGMT_MSG_START_EP, MGMT_TYPE.0, MGMT_TYPE.1)
        | (1u64 << START_EP_FLAG_BIT)
        | field_prep(ep as u64, START_EP_IDX.0, START_EP_IDX.1)
}

/// A `BUFFER_REQUEST` reply granting `n_pages` at device address `dva` (already
/// OR'd with any `dva_base`). rtkit.c:316-321 — the `MGMT_TYPE` field reuses the
/// `MSG_BUFFER_REQUEST` code and the size/iova fields carry the grant.
pub fn msg_buffer_reply(n_pages: u64, dva: u64) -> u64 {
    field_prep(MSG_BUFFER_REQUEST, MGMT_TYPE.0, MGMT_TYPE.1)
        | field_prep(n_pages, BUFFER_REQUEST_SIZE.0, BUFFER_REQUEST_SIZE.1)
        | field_prep(dva, BUFFER_REQUEST_IOVA.0, BUFFER_REQUEST_IOVA.1)
}

// --- endpoint set + state ------------------------------------------------

/// Which system endpoints the IOP advertised in its endpoint map. `START_EP` is
/// sent for each present one before the power-up pump (rtkit.c:625-635).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EndpointSet {
    pub crashlog: bool,
    pub debug: bool,
    pub ioreport: bool,
    pub syslog: bool,
    pub oslog: bool,
}

impl EndpointSet {
    /// Record a present endpoint index from an EPMAP bitmap bit (rtkit.c:584-603).
    /// App endpoints (`>= 0x20`) and EP_MGMT are ignored.
    pub fn record(&mut self, ep_idx: u8) {
        match ep_idx {
            EP_CRASHLOG => self.crashlog = true,
            EP_DEBUG => self.debug = true,
            EP_IOREPORT => self.ioreport = true,
            EP_SYSLOG => self.syslog = true,
            EP_OSLOG => self.oslog = true,
            _ => {}
        }
    }

    /// Fold every set bit of an EPMAP `bitmap` at fragment `base` into the set.
    pub fn record_bitmap(&mut self, bitmap: u32, base: u64) {
        for i in 0..32u8 {
            if bitmap & (1u32 << i) != 0 {
                let ep_idx = (32 * base) as u8 + i;
                if ep_idx < EP_APP_BASE {
                    self.record(ep_idx);
                }
            }
        }
    }

    /// The endpoints to `START_EP`, in m1n1's order (debug, crashlog, syslog,
    /// ioreport, oslog — rtkit.c:626-635).
    pub fn to_start(&self) -> impl Iterator<Item = u8> {
        let list = [
            (self.debug, EP_DEBUG),
            (self.crashlog, EP_CRASHLOG),
            (self.syslog, EP_SYSLOG),
            (self.ioreport, EP_IOREPORT),
            (self.oslog, EP_OSLOG),
        ];
        list.into_iter().filter_map(|(present, ep)| present.then_some(ep))
    }
}

/// Live RTKit boot state (the mutable half of `rtkit_dev`). The pure
/// [`handle_system_msg`] transition folds received system messages into this.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RtkitState {
    pub iop_power: u64,
    pub ap_power: u64,
    pub crashed: bool,
    /// Set once the crashlog endpoint's shared buffer has been granted — a
    /// *second* crashlog buffer request then means the IOP crashed (rtkit.c:437).
    pub have_crashlog_buffer: bool,
}

/// A shared-buffer kind, so the orchestrator can remember which DMA region it
/// handed to which endpoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BufferKind {
    Syslog,
    Crashlog,
    Ioreport,
}

/// What the hardware orchestrator must do after a received system message. The
/// pure transition can't touch MMIO or allocate DMA, so it returns an intent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    /// Nothing to send — state (power/flags) already updated.
    None,
    /// Send this raw `(msg0, ep)` verbatim (e.g. ACK a syslog/ioreport message).
    Send(u64, u8),
    /// Allocate a `n_pages`×4 KiB shared buffer for `ep`'s `kind`; if `addr != 0`
    /// the IOP pre-allocated it (record only, no reply), else reply with the DVA.
    AllocBuffer { ep: u8, kind: BufferKind, n_pages: u64, addr: u64 },
    /// The IOP crashed (a second crashlog buffer request). Abort the boot.
    Crashed,
    /// A message we don't handle (logged by the caller); no reply.
    Unhandled,
}

/// Fold one received **system** message `(msg0, ep)` into `state`, returning the
/// hardware action to take. Pure port of `rtkit_recv`'s per-endpoint switch
/// (rtkit.c:393-468); app endpoints (`ep >= 0x20`) are the caller's concern and
/// must not be passed here.
pub fn handle_system_msg(state: &mut RtkitState, msg0: u64, ep: u8) -> Action {
    let ty = mgmt_type(msg0);
    match ep {
        EP_MGMT => match ty {
            MGMT_MSG_IOP_PWR_STATE_ACK => {
                state.iop_power = pwr_state(msg0);
                Action::None
            }
            MGMT_MSG_AP_PWR_STATE => {
                // AP_PWR_STATE and its ACK share the type code 0xb.
                state.ap_power = pwr_state(msg0);
                Action::None
            }
            _ => Action::Unhandled,
        },
        EP_SYSLOG => match ty {
            MSG_BUFFER_REQUEST => {
                let br = buffer_request(msg0);
                Action::AllocBuffer { ep, kind: BufferKind::Syslog, n_pages: br.n_pages, addr: br.addr }
            }
            MSG_SYSLOG_INIT => Action::None,
            MSG_SYSLOG_LOG => Action::Send(msg0, ep), // ACK by echoing
            _ => Action::Unhandled,
        },
        EP_CRASHLOG => match ty {
            MSG_BUFFER_REQUEST => {
                if state.have_crashlog_buffer {
                    Action::Crashed
                } else {
                    let br = buffer_request(msg0);
                    Action::AllocBuffer { ep, kind: BufferKind::Crashlog, n_pages: br.n_pages, addr: br.addr }
                }
            }
            _ => Action::Unhandled,
        },
        EP_IOREPORT => match ty {
            MSG_BUFFER_REQUEST => {
                let br = buffer_request(msg0);
                Action::AllocBuffer { ep, kind: BufferKind::Ioreport, n_pages: br.n_pages, addr: br.addr }
            }
            // "unknown but must be ACKed" (rtkit.c:454-457).
            0x8 | 0xc => Action::Send(msg0, ep),
            _ => Action::Unhandled,
        },
        _ => Action::Unhandled,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn genmask_and_fields_roundtrip() {
        assert_eq!(gen_mask(59, 52), 0x0ff0_0000_0000_0000);
        assert_eq!(gen_mask(15, 0), 0xffff);
        assert_eq!(gen_mask(0, 0), 1);
        // prep then get is identity within the field.
        let m = field_prep(0x20, 15, 0);
        assert_eq!(field_get(m, 15, 0), 0x20);
        let m = field_prep(7, 59, 52);
        assert_eq!(field_get(m, 59, 52), 7);
    }

    #[test_case]
    fn hello_decode_and_version_negotiation() {
        // Build a HELLO advertising [11,12] with type=1.
        let msg0 = field_prep(MGMT_MSG_HELLO, 59, 52) | field_prep(11, 15, 0) | field_prep(12, 31, 16);
        assert_eq!(mgmt_type(msg0), MGMT_MSG_HELLO);
        assert_eq!(hello_versions(msg0), (11, 12));
        assert_eq!(negotiate_version(11, 12), Some(12));
        // IOP maxes at 11 → we come down to 11.
        assert_eq!(negotiate_version(9, 11), Some(11));
        // Disjoint windows → None.
        assert_eq!(negotiate_version(13, 20), None); // min > our MAX
        assert_eq!(negotiate_version(1, 10), None); // max < our MIN
    }

    #[test_case]
    fn hello_ack_is_wellformed() {
        let ack = msg_hello_ack(12);
        assert_eq!(mgmt_type(ack), MGMT_MSG_HELLO_ACK);
        assert_eq!(hello_versions(ack), (12, 12));
    }

    #[test_case]
    fn epmap_decode_and_reply() {
        // bitmap has EP_CRASHLOG(1), EP_SYSLOG(2), EP_IOREPORT(4) set, base 0, DONE.
        let bitmap = (1u32 << EP_CRASHLOG) | (1u32 << EP_SYSLOG) | (1u32 << EP_IOREPORT);
        let msg0 = field_prep(MGMT_MSG_EPMAP, 59, 52) | field_prep(bitmap as u64, 31, 0) | (1u64 << 51);
        let em = epmap(msg0);
        assert_eq!(em, EpMap { bitmap, base: 0, done: true });

        let mut set = EndpointSet::default();
        set.record_bitmap(em.bitmap, em.base);
        assert_eq!(set, EndpointSet { crashlog: true, syslog: true, ioreport: true, ..Default::default() });
        // START order: debug, crashlog, syslog, ioreport, oslog.
        let starts: alloc::vec::Vec<u8> = set.to_start().collect();
        assert_eq!(starts, alloc::vec![EP_CRASHLOG, EP_SYSLOG, EP_IOREPORT]);

        // Reply: MORE when not done, DONE when done.
        let more = msg_epmap_reply(0, false);
        assert_eq!(mgmt_type(more), MGMT_MSG_EPMAP);
        assert_eq!(more & (1 << 0), 1); // MORE
        assert_eq!(more & (1 << 51), 0);
        let done = msg_epmap_reply(0, true);
        assert_eq!(done & (1 << 51), 1 << 51); // DONE
    }

    #[test_case]
    fn epmap_base_offsets_endpoint_indices() {
        // Fragment base=1 → bit i means endpoint 32+i. Bit for oslog(8) would be
        // ep 40 (app range, ignored); a base-0 bit 8 is the real oslog.
        let mut set = EndpointSet::default();
        set.record_bitmap(1 << 8, 0);
        assert!(set.oslog);
        let mut set2 = EndpointSet::default();
        set2.record_bitmap(1 << 8, 1); // ep 40 → app range → ignored
        assert_eq!(set2, EndpointSet::default());
    }

    #[test_case]
    fn start_ep_encoding() {
        let m = msg_start_ep(EP_IOREPORT);
        assert_eq!(mgmt_type(m), MGMT_MSG_START_EP);
        assert_eq!(m & (1 << 1), 1 << 1); // FLAG
        assert_eq!(field_get(m, 39, 32), EP_IOREPORT as u64);
    }

    #[test_case]
    fn power_acks_update_state() {
        let mut st = RtkitState::default();
        // IOP power ACK → ON.
        let iop = field_prep(MGMT_MSG_IOP_PWR_STATE_ACK, 59, 52) | field_prep(POWER_ON, 15, 0);
        assert_eq!(handle_system_msg(&mut st, iop, EP_MGMT), Action::None);
        assert_eq!(st.iop_power, POWER_ON);
        // AP power ACK → ON.
        let ap = field_prep(MGMT_MSG_AP_PWR_STATE, 59, 52) | field_prep(POWER_ON, 15, 0);
        assert_eq!(handle_system_msg(&mut st, ap, EP_MGMT), Action::None);
        assert_eq!(st.ap_power, POWER_ON);
    }

    #[test_case]
    fn buffer_request_alloc_and_crash_path() {
        let mut st = RtkitState::default();
        // syslog buffer request for 4 pages, IOP wants us to allocate (addr 0).
        let msg0 = field_prep(MSG_BUFFER_REQUEST, 59, 52) | field_prep(4, 51, 44);
        assert_eq!(
            handle_system_msg(&mut st, msg0, EP_SYSLOG),
            Action::AllocBuffer { ep: EP_SYSLOG, kind: BufferKind::Syslog, n_pages: 4, addr: 0 }
        );
        // First crashlog request → allocate.
        assert!(matches!(
            handle_system_msg(&mut st, msg0, EP_CRASHLOG),
            Action::AllocBuffer { kind: BufferKind::Crashlog, .. }
        ));
        // Once the crashlog buffer exists, a second request means a crash.
        st.have_crashlog_buffer = true;
        assert_eq!(handle_system_msg(&mut st, msg0, EP_CRASHLOG), Action::Crashed);
    }

    #[test_case]
    fn buffer_reply_carries_dva_and_size() {
        // A high-half TTBR1 kernel VA (bits [43:40] set) must survive the reply,
        // which the old [41:0] field truncated.
        let dva = 0xfae00000000u64;
        let reply = msg_buffer_reply(4, dva);
        assert_eq!(mgmt_type(reply), MSG_BUFFER_REQUEST);
        assert_eq!(field_get(reply, 51, 44), 4);
        assert_eq!(field_get(reply, 43, 0), dva); // full 44-bit DVA preserved
        assert_eq!(field_get(reply, 41, 0), dva & gen_mask(41, 0));
    }

    #[test_case]
    fn preallocated_buffer_request_carries_addr() {
        // IOP provides its own address → we record, don't reply.
        let msg0 = field_prep(MSG_BUFFER_REQUEST, 59, 52) | field_prep(2, 51, 44) | field_prep(0x1_0000, 41, 0);
        let br = buffer_request(msg0);
        assert_eq!(br, BufferRequest { n_pages: 2, addr: 0x1_0000 });
    }

    #[test_case]
    fn ioreport_ack_and_syslog_echo() {
        let mut st = RtkitState::default();
        // ioreport type 0x8 / 0xc must be ACKed by echo.
        let m8 = field_prep(0x8, 59, 52);
        assert_eq!(handle_system_msg(&mut st, m8, EP_IOREPORT), Action::Send(m8, EP_IOREPORT));
        // syslog LOG must be ACKed by echo.
        let mlog = field_prep(MSG_SYSLOG_LOG, 59, 52);
        assert_eq!(handle_system_msg(&mut st, mlog, EP_SYSLOG), Action::Send(mlog, EP_SYSLOG));
    }
}
