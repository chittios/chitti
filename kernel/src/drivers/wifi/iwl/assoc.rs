//! **The association sequence** — the order the six commands go out in.
//!
//! [`super::ctxt`], [`super::sta`] and [`super::tx`] each build one command;
//! nothing said what order they belong in, and the order is not obvious from
//! any of them. This is that order, as a pure state machine: it emits commands,
//! is told what happened, and emits the next one. No hardware, so the whole
//! join is testable.
//!
//! It is the radio-side twin of [`super::super::link`], which sequences the
//! 802.11 *frames*. The two interleave — the frames are carried by the commands
//! — and [`Stage::AuthAssoc`] and [`Stage::Handshake`] are exactly the points
//! where this machine waits for that one.
//!
//! ## The order, and the two places it is not what you would guess
//!
//! ```text
//! 1  PHY_CONTEXT      ADD      tune to the channel
//! 2  MAC_CONTEXT      ADD      the interface, NOT yet associated
//! 3  BINDING_CONTEXT  ADD      tie the MAC to the PHY
//! 4  ADD_STA                   the access point, BEFORE authenticating
//! 5  (802.11 auth + association exchange)
//! 6  MAC_CONTEXT      MODIFY   now associated, carrying the AID
//! 7  (four-way handshake)
//! 8  ADD_STA_KEY               the pairwise key
//! 9  ADD_STA_KEY               the group key, if the AP sent one
//! ```
//!
//! * **The station is added before authentication, not after.** Management
//!   frames are transmitted *through* a station entry, so a driver that waits
//!   until it is associated has nothing to send the authentication frame with.
//!   The station exists first and is told about the association later.
//!
//! * **The MAC context is sent twice.** Once to create the interface and again,
//!   as a modify, once the association response has supplied an AID and the
//!   beacon timing. Sending it once with a zero AID leaves the firmware unable
//!   to track the beacon, so power save and DTIM never work and the link drops
//!   at the first idle period.
//!
//! Keys come **last**, after the handshake, for the obvious reason: they do not
//! exist until it completes. A driver that installs a key earlier installs
//! whatever its PTK buffer happened to contain.

use alloc::vec::Vec;

/// Where the join has got to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// Nothing sent.
    Idle,
    /// `PHY_CONTEXT_CMD` add.
    PhyContext,
    /// `MAC_CONTEXT_CMD` add — not yet associated.
    MacContext,
    /// `BINDING_CONTEXT_CMD` add.
    Binding,
    /// `ADD_STA` for the access point.
    AddStation,
    /// Waiting on the 802.11 authentication and association exchange, which
    /// [`super::super::link`] drives.
    AuthAssoc,
    /// `MAC_CONTEXT_CMD` modify, now carrying the association id.
    MacContextAssoc,
    /// Waiting on the four-way handshake.
    Handshake,
    /// `ADD_STA_KEY` for the pairwise key, then the group key.
    InstallKeys,
    /// Traffic can flow.
    Ready,
    /// Terminal.
    Failed,
}

/// A command to transmit: `(group, id, payload)`.
pub type Command = (u8, u8, Vec<u8>);

/// What the sequencer needs to know about the network being joined.
#[derive(Debug, Clone)]
pub struct Target {
    pub bssid: [u8; 6],
    pub our_mac: [u8; 6],
    pub channel: u8,
    /// Filled in from the association response.
    pub aid: u16,
    pub beacon_interval: u16,
    pub dtim_period: u8,
}

/// Context identifiers this interface occupies. Ids are a small fixed space the
/// driver allocates; one interface uses one of each.
#[derive(Debug, Clone, Copy)]
pub struct Ids {
    pub phy_id: u8,
    pub phy_color: u8,
    pub mac_id: u8,
    pub mac_color: u8,
    pub binding_id: u8,
    pub binding_color: u8,
}

impl Default for Ids {
    fn default() -> Self {
        Ids {
            phy_id: 0,
            phy_color: 0,
            mac_id: 0,
            mac_color: 0,
            binding_id: 0,
            binding_color: 0,
        }
    }
}

/// Drives the association commands in order.
pub struct Sequence {
    stage: Stage,
    target: Target,
    ids: Ids,
    /// Transmit queues the station may use.
    tfd_queue_msk: u32,
}

impl Sequence {
    pub fn new(target: Target, ids: Ids, tfd_queue_msk: u32) -> Sequence {
        Sequence {
            stage: Stage::Idle,
            target,
            ids,
            tfd_queue_msk,
        }
    }

    pub fn stage(&self) -> Stage {
        self.stage
    }
    pub fn ready(&self) -> bool {
        self.stage == Stage::Ready
    }

    /// The association response supplies the AID and the beacon timing, which
    /// the second MAC context needs.
    pub fn on_associated(&mut self, aid: u16, beacon_interval: u16, dtim_period: u8) {
        self.target.aid = aid;
        self.target.beacon_interval = beacon_interval;
        self.target.dtim_period = dtim_period;
    }

    /// The next command to send, or `None` when this stage is waiting on the
    /// 802.11 exchange (or the join is finished).
    ///
    /// Advancing is [`Self::advance`]'s job, not this one's: a caller that fails
    /// to send must be able to ask again and get the same command rather than
    /// skipping it.
    pub fn next_command(
        &self,
        keys: Option<(&[u8; 16], u64)>,
        gtk: Option<(u8, &[u8])>,
    ) -> Option<Command> {
        use super::{ctxt, sta};
        match self.stage {
            Stage::Idle | Stage::PhyContext => Some((
                super::proto::GROUP_LEGACY,
                ctxt::PHY_CONTEXT_CMD,
                ctxt::phy_context_20mhz(
                    self.ids.phy_id,
                    self.ids.phy_color,
                    self.target.channel,
                    ctxt::ACTION_ADD,
                ),
            )),
            Stage::MacContext => Some((
                super::proto::GROUP_LEGACY,
                ctxt::MAC_CONTEXT_CMD,
                ctxt::mac_context_sta(
                    self.ids.mac_id,
                    self.ids.mac_color,
                    &self.target.our_mac,
                    &self.target.bssid,
                    ctxt::ACTION_ADD,
                    // **Not associated yet** — the AID does not exist until the
                    // association response arrives.
                    None,
                ),
            )),
            Stage::Binding => Some((
                super::proto::GROUP_LEGACY,
                ctxt::BINDING_CONTEXT_CMD,
                ctxt::binding(
                    self.ids.binding_id,
                    self.ids.binding_color,
                    ctxt::id_and_color(self.ids.mac_id, self.ids.mac_color),
                    ctxt::id_and_color(self.ids.phy_id, self.ids.phy_color),
                    ctxt::ACTION_ADD,
                ),
            )),
            Stage::AddStation => Some((
                super::proto::GROUP_LEGACY,
                sta::ADD_STA,
                // The AID is zero here and that is correct: the station exists
                // so management frames have something to go out through, and is
                // told about the association afterwards.
                sta::add_ap(
                    self.ids.mac_id,
                    self.ids.mac_color,
                    &self.target.bssid,
                    0,
                    self.tfd_queue_msk,
                ),
            )),
            Stage::MacContextAssoc => Some((
                super::proto::GROUP_LEGACY,
                ctxt::MAC_CONTEXT_CMD,
                ctxt::mac_context_sta(
                    self.ids.mac_id,
                    self.ids.mac_color,
                    &self.target.our_mac,
                    &self.target.bssid,
                    ctxt::ACTION_MODIFY,
                    Some((
                        self.target.aid,
                        self.target.beacon_interval,
                        self.target.dtim_period,
                    )),
                ),
            )),
            Stage::InstallKeys => {
                let (tk, pn) = keys?;
                // The pairwise key first: it protects everything unicast. The
                // group key follows only if the AP sent one.
                let _ = gtk;
                Some((
                    super::proto::GROUP_LEGACY,
                    sta::ADD_STA_KEY,
                    sta::add_pairwise_key(sta::AP_STA_ID, 0, tk, pn),
                ))
            }
            // These wait on the 802.11 exchange, and `Ready`/`Failed` are done.
            Stage::AuthAssoc | Stage::Handshake | Stage::Ready | Stage::Failed => None,
        }
    }

    /// The group-key command, once the pairwise key is in. Separate because the
    /// AP may not have sent one.
    pub fn group_key_command(&self, gtk: Option<(u8, &[u8])>) -> Option<Command> {
        let (id, key) = gtk?;
        Some((
            super::proto::GROUP_LEGACY,
            super::sta::ADD_STA_KEY,
            super::sta::add_group_key(super::sta::AP_STA_ID, id, key)?,
        ))
    }

    /// Move to the next stage after the current command succeeded.
    pub fn advance(&mut self) {
        self.stage = match self.stage {
            Stage::Idle | Stage::PhyContext => Stage::MacContext,
            Stage::MacContext => Stage::Binding,
            Stage::Binding => Stage::AddStation,
            // The station is up; now the 802.11 exchange runs.
            Stage::AddStation => Stage::AuthAssoc,
            Stage::AuthAssoc => Stage::MacContextAssoc,
            Stage::MacContextAssoc => Stage::Handshake,
            Stage::Handshake => Stage::InstallKeys,
            Stage::InstallKeys => Stage::Ready,
            Stage::Ready => Stage::Ready,
            Stage::Failed => Stage::Failed,
        };
    }

    /// Abandon the join.
    pub fn fail(&mut self) {
        self.stage = Stage::Failed;
    }
}

#[cfg(test)]
mod tests {
    use super::super::{ctxt, sta};
    use super::*;

    fn target() -> Target {
        Target {
            bssid: [0xaa; 6],
            our_mac: [0x02; 6],
            channel: 6,
            aid: 0,
            beacon_interval: 0,
            dtim_period: 0,
        }
    }

    fn drive() -> Sequence {
        Sequence::new(target(), Ids::default(), 0xff)
    }

    /// The order, end to end. Each command is identified by its id so a
    /// reordering shows up as the wrong command rather than as a subtle
    /// difference in bytes.
    #[test_case]
    fn the_commands_go_out_in_order() {
        let mut s = drive();
        let tk = [0x11u8; 16];

        let mut seen = alloc::vec::Vec::new();
        for _ in 0..12 {
            if let Some((_, id, _)) = s.next_command(Some((&tk, 1)), None) {
                seen.push(id);
            }
            if s.stage() == Stage::AuthAssoc {
                s.on_associated(7, 100, 3);
            }
            s.advance();
            if s.ready() {
                break;
            }
        }
        assert_eq!(
            seen,
            alloc::vec![
                ctxt::PHY_CONTEXT_CMD,
                ctxt::MAC_CONTEXT_CMD,
                ctxt::BINDING_CONTEXT_CMD,
                sta::ADD_STA,
                ctxt::MAC_CONTEXT_CMD, // again, associated
                sta::ADD_STA_KEY,
            ]
        );
        assert!(s.ready());
    }

    /// **The station is added before authentication.** Management frames are
    /// transmitted through a station entry, so a driver that waits until it is
    /// associated has nothing to send the authentication frame with.
    #[test_case]
    fn the_station_exists_before_the_authentication_exchange() {
        let mut s = drive();
        // Walk to the station stage.
        while s.stage() != Stage::AddStation {
            assert_ne!(s.stage(), Stage::AuthAssoc, "auth must not come first");
            s.advance();
        }
        let (_, id, body) = s.next_command(None, None).expect("ADD_STA");
        assert_eq!(id, sta::ADD_STA);
        // The AID is zero at this point and that is correct.
        assert_eq!(u16::from_le_bytes([body[36], body[37]]), 0, "no AID yet");
        // Only *after* the station does the 802.11 exchange run.
        s.advance();
        assert_eq!(s.stage(), Stage::AuthAssoc);
        assert_eq!(
            s.next_command(None, None),
            None,
            "this stage waits on frames"
        );
    }

    /// **The MAC context is sent twice**, and the second carries the AID and the
    /// beacon timing. Sending it once with a zero AID leaves the firmware unable
    /// to track the beacon, so DTIM and power save never work and the link drops
    /// at the first idle period.
    #[test_case]
    fn the_mac_context_is_sent_again_once_associated() {
        let mut s = drive();
        s.advance(); // past phy
        let (_, _, first) = s.next_command(None, None).unwrap();
        let u = ctxt::MAC_UNION_OFF;
        assert_eq!(
            u32::from_le_bytes(first[u..u + 4].try_into().unwrap()),
            0,
            "not associated"
        );
        let a1 = u32::from_le_bytes(first[4..8].try_into().unwrap());
        assert_eq!(a1, ctxt::ACTION_ADD);

        // Walk to the post-association context.
        while s.stage() != Stage::AuthAssoc {
            s.advance();
        }
        s.on_associated(9, 100, 2);
        s.advance();
        assert_eq!(s.stage(), Stage::MacContextAssoc);
        let (_, id, second) = s.next_command(None, None).unwrap();
        assert_eq!(id, ctxt::MAC_CONTEXT_CMD, "the same command, a second time");
        assert_eq!(
            u32::from_le_bytes(second[u..u + 4].try_into().unwrap()),
            1,
            "associated"
        );
        let a2 = u32::from_le_bytes(second[4..8].try_into().unwrap());
        assert_eq!(a2, ctxt::ACTION_MODIFY, "modify, not add");
        assert_eq!(
            u32::from_le_bytes(second[u + 36..u + 40].try_into().unwrap()),
            9,
            "the AID"
        );
    }

    /// **Keys come last**, after the handshake — they do not exist until it
    /// completes. Asked earlier the sequencer emits no key command at all
    /// rather than one built from an empty buffer.
    #[test_case]
    fn no_key_is_installed_before_the_handshake_completes() {
        let mut s = drive();
        let tk = [0x22u8; 16];
        // At every stage before InstallKeys, nothing emits ADD_STA_KEY.
        for _ in 0..8 {
            if s.stage() == Stage::InstallKeys {
                break;
            }
            if let Some((_, id, _)) = s.next_command(Some((&tk, 1)), None) {
                assert_ne!(id, sta::ADD_STA_KEY, "a key before the handshake");
            }
            s.advance();
        }
        assert_eq!(s.stage(), Stage::InstallKeys);
        // And with no key material it emits nothing rather than an empty key.
        assert_eq!(s.next_command(None, None), None);
        let (_, id, _) = s.next_command(Some((&tk, 1)), None).unwrap();
        assert_eq!(id, sta::ADD_STA_KEY);
    }

    /// The group key is separate because the access point may not have sent
    /// one, and a network with no group key is legitimate.
    #[test_case]
    fn the_group_key_is_optional() {
        let s = drive();
        assert_eq!(s.group_key_command(None), None, "no GTK, no command");
        let gtk = [0x33u8; 16];
        let (_, id, body) = s.group_key_command(Some((2, &gtk))).expect("a GTK");
        assert_eq!(id, sta::ADD_STA_KEY);
        // It really is the group key: the multicast bit is set.
        let flags = u16::from_le_bytes([body[2], body[3]]);
        assert_ne!(flags & sta::KEY_MULTICAST, 0);
        // A key that is not CCMP-128 is refused rather than truncated.
        assert_eq!(s.group_key_command(Some((1, &[0u8; 32]))), None);
    }

    /// A command is not consumed by asking for it: a caller that fails to send
    /// must be able to ask again and get the same command rather than skipping
    /// a stage.
    #[test_case]
    fn asking_twice_yields_the_same_command() {
        let s = drive();
        let a = s.next_command(None, None);
        let b = s.next_command(None, None);
        assert_eq!(a, b, "asking does not advance");
        assert_eq!(s.stage(), Stage::Idle);
    }

    /// A failed join stops emitting, so a caller that ignores the error does
    /// not continue configuring a radio it has given up on.
    #[test_case]
    fn a_failed_sequence_emits_nothing_further() {
        let mut s = drive();
        s.fail();
        assert_eq!(s.stage(), Stage::Failed);
        assert_eq!(s.next_command(Some((&[0; 16], 1)), None), None);
        s.advance();
        assert_eq!(s.stage(), Stage::Failed, "and stays failed");
    }
}
