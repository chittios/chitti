//! **Host/firmware wire formats** — commands out, notifications in.
//!
//! Once firmware is running, everything between host and device is a framed message on a
//! ring: a command header plus payload going out, and a receive packet carrying a
//! notification coming back. Both formats are small and both are entirely
//! unit-testable, which is why they live here rather than inside the ring plumbing —
//! the same split as [`super::csr`] and [`super::context`].
//!
//! The reason this layer matters more than its size suggests: it is what turns "firmware
//! was handed over" into "firmware answered". Until a notification can be parsed, a load
//! that silently failed and a load that worked are indistinguishable from the host.
//!
//! Layouts from Linux's `iwl-trans.h`, `iwl-fh.h` and `commands.h`. **Unverified against
//! silicon** — no emulator provides an Intel WiFi device.

use alloc::vec::Vec;

// --- command identity -----------------------------------------------------

/// Command groups. A command is identified by the pair (group, id), and the *same* id
/// means different things in different groups — so a command sent with the wrong group is
/// not rejected, it is obeyed as something else.
pub const GROUP_LEGACY: u8 = 0x0;
pub const GROUP_LONG: u8 = 0x1;
pub const GROUP_SYSTEM: u8 = 0x2;
pub const GROUP_MAC_CONF: u8 = 0x3;
pub const GROUP_PHY_OPS: u8 = 0x4;
pub const GROUP_DATA_PATH: u8 = 0x5;

/// Regulatory and NVM group — where the device's own configuration is read from.
pub const GROUP_REGULATORY_NVM: u8 = 0xc;
/// Read the NVM's summary. **Read-only**, which is why it is the first command sent: a
/// wrong guess about a configuration command's payload misconfigures a radio, while a wrong
/// guess here can only be answered with an error.
pub const NVM_GET_INFO: u8 = 0x02;

/// `SCAN_REQ_UMAC` — start a scan. Group `LONG` (0x1), command 0x0d.
///
/// **Not sent by this driver**, and the reason is the point: the request
/// structure is versioned and has changed repeatedly (adaptive dwell, then a
/// v8 rewrite, then per-band channel configuration), so the *same* command id
/// takes a different layout on different firmware. `SCAN_REQ_UMAC_VERSIONS`
/// records which layouts are implemented here — currently none — and
/// `iwl::scan_supported` refuses anything absent from it.
///
/// Sending a plausible-looking structure at the wrong version is the failure
/// mode this whole module is written to avoid: the radio accepts the command,
/// interprets our fields as different ones, and either scans nothing or
/// configures itself in a way no host-visible error reports.
pub const SCAN_REQ_UMAC: u8 = 0x0d;

/// Versions of `SCAN_REQ_UMAC` whose layout this driver implements.
///
/// v17 is implemented in [`super::scan`], with every offset pinned against
/// `scan.h`'s own field list. Adding another version means adding its struct
/// too — the test in `super::scan` fails if this list names a version no
/// builder exists for.
///
/// **Still unverified against a radio.** No emulator provides an Intel WiFi
/// part, so the layout is checked against the header's arithmetic and nothing
/// more; the refusal path above remains the honest answer for every other
/// version.
pub const SCAN_REQ_UMAC_VERSIONS: &[u8] = &[17];

/// Firmware's first word after a successful load: it is alive.
pub const UCODE_ALIVE_NTFY: u8 = 0x01;
/// Initialisation finished.
pub const INIT_COMPLETE_NOTIF: u8 = 0x04;
/// Firmware failed and is describing why.
pub const UCODE_ERROR_NTFY: u8 = 0x02;

/// The wide command header every gen2 message carries.
///
/// `#[repr(C, packed)]` because the device reads it byte for byte; a padded version would
/// place the length two bytes from where firmware looks for it.
#[repr(C, packed)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct CmdHeaderWide {
    pub cmd: u8,
    pub group_id: u8,
    pub sequence: u16,
    pub length: u16,
    pub reserved: u8,
    pub version: u8,
}

/// Size of the wide header, which every payload offset is measured from.
pub const CMD_HEADER_WIDE_LEN: usize = 8;

/// Build a command header.
///
/// `payload_len` is the payload alone — firmware adds the header itself. Passing the
/// whole frame length here makes firmware read eight bytes past the end of the payload,
/// which it answers with an error notification at best and silence at worst.
pub fn cmd_header(group: u8, cmd: u8, sequence: u16, payload_len: u16) -> CmdHeaderWide {
    CmdHeaderWide {
        cmd,
        group_id: group,
        sequence,
        length: payload_len,
        reserved: 0,
        version: 0,
    }
}

/// Pack a queue index and slot into the `sequence` field.
///
/// Firmware echoes this back on the response, and it is the only way a reply is matched
/// to its request — two commands in flight with the same sequence make one of the
/// answers land on the wrong waiter.
pub fn make_sequence(queue: u8, slot: u8) -> u16 {
    ((queue as u16 & 0x1f) << 8) | slot as u16
}

/// The queue a sequence came from.
pub fn sequence_queue(seq: u16) -> u8 {
    ((seq >> 8) & 0x1f) as u8
}

/// The slot a sequence came from.
pub fn sequence_slot(seq: u16) -> u8 {
    (seq & 0xff) as u8
}

// --- transmit descriptors -------------------------------------------------

/// One transfer buffer: where a piece of the frame is and how long.
#[repr(C, packed)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct TransferBuffer {
    pub len: u16,
    pub addr: u64,
}

/// Maximum transfer buffers one descriptor can chain.
pub const MAX_TBS: usize = 25;

/// A gen2 transmit descriptor: a count, then that many transfer buffers.
///
/// The count is what the device trusts — not the array length — so a descriptor whose
/// `num_tbs` exceeds the buffers actually filled makes the device DMA from uninitialised
/// memory, and one that undercounts silently truncates the frame.
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct Tfd {
    pub num_tbs: u16,
    pub tbs: [TransferBuffer; MAX_TBS],
    pub _pad: u32,
}

impl Default for Tfd {
    fn default() -> Self {
        Tfd {
            num_tbs: 0,
            tbs: [TransferBuffer::default(); MAX_TBS],
            _pad: 0,
        }
    }
}

/// Build a descriptor over `bufs`, each `(physical address, length)`.
///
/// `None` when there are too many buffers to chain, or when any is empty or unaddressed —
/// both are descriptors the device would act on and neither is expressible as an error
/// once it has.
pub fn build_tfd(bufs: &[(u64, u16)]) -> Option<Tfd> {
    if bufs.is_empty() || bufs.len() > MAX_TBS {
        return None;
    }
    let mut t = Tfd::default();
    for (i, &(addr, len)) in bufs.iter().enumerate() {
        if addr == 0 || len == 0 {
            return None;
        }
        t.tbs[i] = TransferBuffer { len, addr };
    }
    t.num_tbs = bufs.len() as u16;
    Some(t)
}

/// Serialise a command: the wide header followed by its payload.
///
/// Kept pure and separate from the queue plumbing because this is the byte layout firmware
/// parses, and the two mistakes it can contain — a length that counts the header, and a
/// little-endian field written big-endian — both produce a command that is *obeyed*, with
/// the wrong size or the wrong contents.
pub fn build_command(group: u8, cmd: u8, sequence: u16, payload: &[u8]) -> Vec<u8> {
    let mut v = Vec::with_capacity(CMD_HEADER_WIDE_LEN + payload.len());
    v.push(cmd);
    v.push(group);
    v.extend_from_slice(&sequence.to_le_bytes());
    v.extend_from_slice(&(payload.len() as u16).to_le_bytes());
    v.push(0); // reserved
    v.push(0); // version
    v.extend_from_slice(payload);
    v
}

/// Size of the first transfer buffer, which the device fetches into a small internal
/// staging buffer rather than reading in place.
///
/// This is why a command cannot simply be pointed at as one buffer: the first 20 bytes must
/// live in a separate, aligned allocation. Getting it wrong does not fail cleanly — the
/// device reads a partly-stale header.
pub const FIRST_TB_SIZE: usize = 20;
/// The alignment that staging buffer needs.
pub const FIRST_TB_ALIGN: usize = 64;

/// How a command's bytes are split across transfer buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommandSplit {
    /// Bytes copied into the aligned first-TB staging buffer.
    pub first: usize,
    /// Bytes left in the command buffer, described by a second TB — zero for a short
    /// command that fits entirely in the first.
    pub rest: usize,
}

/// Split a command of `len` bytes the way the device expects to fetch it.
pub fn split_command(len: usize) -> CommandSplit {
    let first = core::cmp::min(len, FIRST_TB_SIZE);
    CommandSplit {
        first,
        rest: len - first,
    }
}

/// The host's view of a command queue: where the next command goes and what is in flight.
///
/// Pure bookkeeping, deliberately separate from the DMA and the doorbell. The failure it
/// exists to prevent is a wrapped write index overwriting a descriptor the device has not
/// read yet — which loses a command already counted as sent, so the driver waits forever
/// for a response to something the device never saw.
#[derive(Debug, Clone)]
pub struct CmdQueue {
    /// Number of slots, a power of two.
    pub slots: usize,
    /// Next slot the host will write.
    write: usize,
    /// Sequence occupying each slot, or `None` when free.
    inflight: Vec<Option<u16>>,
    /// Queue id, which goes into both the doorbell and every sequence number.
    pub id: u8,
}

impl CmdQueue {
    /// `slots` must be a power of two — the device wraps the index by masking, so anything
    /// else makes the host and the device disagree about which descriptor is which.
    pub fn new(id: u8, slots: usize) -> Option<CmdQueue> {
        if slots == 0 || !slots.is_power_of_two() || slots > 256 {
            return None;
        }
        Some(CmdQueue {
            slots,
            write: 0,
            inflight: alloc::vec![None; slots],
            id,
        })
    }

    /// Log2 of the slot count, which is what the context info wants.
    pub fn size_log2(&self) -> u8 {
        self.slots.trailing_zeros() as u8
    }

    /// Claim the next slot, returning `(slot, sequence)`.
    ///
    /// `None` when that slot is still in flight: the queue is full, and reusing the slot
    /// would overwrite a descriptor the device may not have read.
    pub fn claim(&mut self) -> Option<(usize, u16)> {
        if self.inflight[self.write].is_some() {
            return None;
        }
        let slot = self.write;
        let seq = make_sequence(self.id, slot as u8);
        self.inflight[slot] = Some(seq);
        self.write = (self.write + 1) % self.slots;
        Some((slot, seq))
    }

    /// The write index to ring the doorbell with after filling a slot.
    pub fn write_index(&self) -> u16 {
        self.write as u16
    }

    /// Retire a response's sequence, freeing its slot.
    ///
    /// Returns false for a sequence that is not in flight — a duplicate response, or one
    /// for a command this driver never sent. Both happen (firmware retransmits), and
    /// freeing a slot on the strength of one would release a slot still in use.
    pub fn retire(&mut self, seq: u16) -> bool {
        let slot = sequence_slot(seq) as usize;
        if slot >= self.slots || self.inflight[slot] != Some(seq) {
            return false;
        }
        self.inflight[slot] = None;
        true
    }

    /// Whether `seq` is a command still awaiting its response.
    pub fn is_inflight(&self, seq: u16) -> bool {
        let slot = sequence_slot(seq) as usize;
        slot < self.slots && self.inflight[slot] == Some(seq)
    }
}

// --- receive packets ------------------------------------------------------

/// The frame-size field of `len_n_flags` — the rest are flags, and masking them off is
/// not optional: used unmasked, the length is enormous and every bounds check passes.
pub const RX_FRAME_SIZE_MASK: u32 = 0x0000_3fff;

/// A notification from firmware.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RxPacket<'a> {
    pub group_id: u8,
    pub cmd: u8,
    pub sequence: u16,
    pub payload: &'a [u8],
}

/// Firmware's own verdict on its startup, in the first halfword of the alive notification.
pub const ALIVE_STATUS_OK: u16 = 0xcafe;
/// The value firmware writes when it started but is not usable.
pub const ALIVE_STATUS_ERR: u16 = 0xdead;

impl RxPacket<'_> {
    /// Whether this is firmware announcing it is alive.
    pub fn is_alive(&self) -> bool {
        self.group_id == GROUP_LEGACY && self.cmd == UCODE_ALIVE_NTFY
    }

    /// The status word out of an alive notification.
    ///
    /// Only the status is read. The rest of that structure has gone through many revisions
    /// with fields at different offsets per firmware version, and there is no honest way to
    /// pick one here without a device to check against — whereas the status is the field
    /// that decides whether to continue, and an *unusable* firmware that announced itself is
    /// otherwise taken for a working one.
    pub fn alive_status(&self) -> Option<u16> {
        if !self.is_alive() || self.payload.len() < 2 {
            return None;
        }
        Some(u16::from_le_bytes([self.payload[0], self.payload[1]]))
    }

    /// Whether this packet is the response to a command sent with `sequence`.
    ///
    /// Group and command must match too. Firmware sends unsolicited notifications
    /// constantly, and a sequence number is only 13 meaningful bits — so matching on it
    /// alone eventually hands a scan notification to whatever is waiting for a reply.
    pub fn answers(&self, group: u8, cmd: u8, sequence: u16) -> bool {
        self.group_id == group && self.cmd == cmd && self.sequence == sequence
    }

    /// Whether this is firmware reporting its own failure.
    pub fn is_error(&self) -> bool {
        self.group_id == GROUP_LEGACY && self.cmd == UCODE_ERROR_NTFY
    }
}

/// Parse one receive packet out of a buffer the device wrote.
///
/// The layout is a 4-byte `len_n_flags`, then the wide command header, then payload. Three
/// things are checked, and each rejects a real failure rather than a hypothetical one: the
/// declared frame size must be masked before use (unmasked it is huge and every bounds
/// check trivially passes), it must be at least a header (a shorter one would make the
/// payload slice underflow), and it must fit the buffer the device was given (a longer one
/// means the device wrote more than the ring can hold, and trusting it reads whatever
/// follows).
pub fn parse_rx(buf: &[u8]) -> Option<RxPacket<'_>> {
    if buf.len() < 4 + CMD_HEADER_WIDE_LEN {
        return None;
    }
    let len_n_flags = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
    let size = (len_n_flags & RX_FRAME_SIZE_MASK) as usize;
    if size < CMD_HEADER_WIDE_LEN || 4 + size > buf.len() {
        return None;
    }
    let h = &buf[4..4 + CMD_HEADER_WIDE_LEN];
    Some(RxPacket {
        cmd: h[0],
        group_id: h[1],
        sequence: u16::from_le_bytes([h[2], h[3]]),
        payload: &buf[4 + CMD_HEADER_WIDE_LEN..4 + size],
    })
}

/// What the NVM info response says about the device.
///
/// Field widths are **not** uniform, and that is the whole reason this struct exists rather
/// than a slice of `u32`s: the general section is `u32, u16, u8, u8`. Read as four words —
/// which is what a plausible-looking guess produces — `nvm_version` swallows the board type
/// and address count, `board_type` reads the SKU flags and `n_hw_addrs` reads the transmit
/// chain mask, which is a small number and passes any sanity check. Taken from Linux's
/// `iwl_nvm_get_info_rsp` in `fw/api/nvm-reg.h`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NvmInfo {
    pub flags: u32,
    pub nvm_version: u16,
    pub board_type: u8,
    /// How many MAC addresses the device is provisioned with. Zero would mean it cannot
    /// address a frame, so it is a useful check on the whole exchange.
    pub n_hw_addrs: u8,
    /// Transmit and receive chain masks from the PHY section — the radio's antenna
    /// configuration, and `None` on a response too short to carry it (older API versions
    /// return less, and reporting the absence beats inventing a chain count).
    pub chains: Option<(u32, u32)>,
}

/// Length of the response's general section: `flags`, `nvm_version`, `board_type`,
/// `n_hw_addrs`.
pub const NVM_GENERAL_LEN: usize = 8;
/// Offset of the PHY section: past the general section and the 4-byte SKU section.
pub const NVM_PHY_OFFSET: usize = NVM_GENERAL_LEN + 4;

impl NvmInfo {
    /// Parse an NVM info response.
    pub fn parse(payload: &[u8]) -> Option<NvmInfo> {
        if payload.len() < NVM_GENERAL_LEN {
            return None;
        }
        let le32 = |at: usize| {
            u32::from_le_bytes([
                payload[at],
                payload[at + 1],
                payload[at + 2],
                payload[at + 3],
            ])
        };
        let info = NvmInfo {
            flags: le32(0),
            nvm_version: u16::from_le_bytes([payload[4], payload[5]]),
            board_type: payload[6],
            n_hw_addrs: payload[7],
            chains: if payload.len() >= NVM_PHY_OFFSET + 8 {
                Some((le32(NVM_PHY_OFFSET), le32(NVM_PHY_OFFSET + 4)))
            } else {
                None
            },
        };
        // An all-ones response is a floating read dressed up as data, and a device
        // provisioned with no addresses did not answer this command.
        if info.nvm_version == u16::MAX || info.n_hw_addrs == 0 || info.n_hw_addrs > 16 {
            return None;
        }
        Some(info)
    }
}

/// Build the free-receive-buffer list the device consumes.
///
/// Each entry is a physical address of a receive buffer. Pure because the failure is
/// arithmetic and invisible: a list with a zero entry gives the device a buffer at
/// physical address 0 to write a packet into.
pub fn build_rbd_list(bufs: &[u64]) -> Option<Vec<u64>> {
    if bufs.is_empty() || bufs.iter().any(|&b| b == 0 || b % 256 != 0) {
        return None;
    }
    Some(bufs.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn the_command_header_is_packed_as_firmware_reads_it() {
        // Padding here would move `length` two bytes from where firmware looks, and a
        // command whose length reads as garbage is obeyed with a garbage payload size.
        assert_eq!(core::mem::size_of::<CmdHeaderWide>(), CMD_HEADER_WIDE_LEN);
        assert_eq!(core::mem::offset_of!(CmdHeaderWide, cmd), 0);
        assert_eq!(core::mem::offset_of!(CmdHeaderWide, group_id), 1);
        assert_eq!(core::mem::offset_of!(CmdHeaderWide, sequence), 2);
        assert_eq!(core::mem::offset_of!(CmdHeaderWide, length), 4);
        assert_eq!(core::mem::offset_of!(CmdHeaderWide, version), 7);
    }

    #[test_case]
    fn a_header_carries_the_payload_length_not_the_frame_length() {
        // Firmware adds the header itself. Passing the whole frame length makes it read
        // eight bytes past the payload.
        let h = cmd_header(GROUP_SYSTEM, 0x1c, 0x0102, 40);
        assert_eq!(h.group_id, GROUP_SYSTEM);
        assert_eq!(h.cmd, 0x1c);
        assert_eq!({ h.length }, 40);
        assert_eq!({ h.sequence }, 0x0102);
    }

    #[test_case]
    fn a_sequence_round_trips_its_queue_and_slot() {
        // The response is matched to its request by this field alone, so a collision
        // delivers one command's answer to another's waiter.
        for q in [0u8, 1, 4, 31] {
            for slot in [0u8, 1, 7, 255] {
                let s = make_sequence(q, slot);
                assert_eq!(sequence_queue(s), q, "queue lost for {q}/{slot}");
                assert_eq!(sequence_slot(s), slot, "slot lost for {q}/{slot}");
            }
        }
        // The queue field is five bits; a larger value must not bleed into the slot.
        assert_eq!(sequence_slot(make_sequence(0xff, 3)), 3);
    }

    #[test_case]
    fn a_descriptor_trusts_its_count_so_the_count_must_match() {
        // `num_tbs` is what the device acts on, not the array length: over-count and it
        // DMAs from uninitialised memory, under-count and the frame is truncated.
        let t = build_tfd(&[(0x1000, 8), (0x2000, 64)]).unwrap();
        assert_eq!({ t.num_tbs }, 2);
        assert_eq!({ t.tbs[0].addr }, 0x1000);
        assert_eq!({ t.tbs[0].len }, 8);
        assert_eq!({ t.tbs[1].len }, 64);
        // Entries past the count stay zeroed, so an over-count would at least fault
        // rather than read a stale address from a previous frame.
        assert_eq!({ t.tbs[2].addr }, 0);
    }

    #[test_case]
    fn a_descriptor_refuses_what_the_device_would_act_on_regardless() {
        assert!(build_tfd(&[]).is_none(), "empty descriptor accepted");
        assert!(build_tfd(&[(0, 8)]).is_none(), "null buffer address accepted");
        assert!(build_tfd(&[(0x1000, 0)]).is_none(), "zero-length buffer accepted");
        let many: alloc::vec::Vec<(u64, u16)> = (1..=MAX_TBS as u64 + 1).map(|i| (i * 0x1000, 4)).collect();
        assert!(build_tfd(&many).is_none(), "over-long chain accepted");
        let exact: alloc::vec::Vec<(u64, u16)> = (1..=MAX_TBS as u64).map(|i| (i * 0x1000, 4)).collect();
        // Copy the field out before comparing: a reference into a packed struct is
        // undefined behaviour even when it is never dereferenced.
        let n = { build_tfd(&exact).unwrap().num_tbs };
        assert_eq!(n, MAX_TBS as u16);
    }

    /// A receive buffer as the device would have written it.
    fn rx_bytes(group: u8, cmd: u8, seq: u16, payload: &[u8], flags: u32) -> alloc::vec::Vec<u8> {
        let size = CMD_HEADER_WIDE_LEN + payload.len();
        let mut v = alloc::vec::Vec::new();
        v.extend_from_slice(&((size as u32) | flags).to_le_bytes());
        v.push(cmd);
        v.push(group);
        v.extend_from_slice(&seq.to_le_bytes());
        v.extend_from_slice(&(payload.len() as u16).to_le_bytes());
        v.push(0);
        v.push(0);
        v.extend_from_slice(payload);
        v
    }

    #[test_case]
    fn parses_a_notification_and_recognises_alive() {
        // The whole point of this layer: until a notification can be read, a firmware load
        // that silently failed and one that worked look identical from the host.
        let b = rx_bytes(GROUP_LEGACY, UCODE_ALIVE_NTFY, 0x0203, &[1, 2, 3, 4], 0);
        let p = parse_rx(&b).unwrap();
        assert!(p.is_alive());
        assert!(!p.is_error());
        assert_eq!(p.sequence, 0x0203);
        assert_eq!(p.payload, &[1, 2, 3, 4]);

        let e = rx_bytes(GROUP_LEGACY, UCODE_ERROR_NTFY, 0, &[], 0);
        assert!(parse_rx(&e).unwrap().is_error());
    }

    #[test_case]
    fn the_frame_size_is_masked_before_it_is_trusted() {
        // The upper bits of `len_n_flags` are flags. Used unmasked the length is enormous
        // and every bounds check passes trivially — so the parse would hand out a slice
        // running off the end of the ring.
        let b = rx_bytes(GROUP_LEGACY, UCODE_ALIVE_NTFY, 0, &[9, 9], 0xffff_0000);
        let p = parse_rx(&b).expect("flags must not make the packet unparseable");
        assert_eq!(p.payload, &[9, 9]);
    }

    #[test_case]
    fn a_packet_shorter_than_its_header_or_longer_than_its_buffer_is_refused() {
        // Both are things a confused device really does, and both would otherwise read
        // past what it was given.
        let mut short = rx_bytes(GROUP_LEGACY, UCODE_ALIVE_NTFY, 0, &[], 0);
        short[0] = 4; // declares less than one header
        assert!(parse_rx(&short).is_none());

        let mut over = rx_bytes(GROUP_LEGACY, UCODE_ALIVE_NTFY, 0, &[1, 2], 0);
        over[0] = 0x40; // declares 64 bytes in a buffer holding 14
        assert!(parse_rx(&over).is_none());

        assert!(parse_rx(&[]).is_none());
        assert!(parse_rx(&[0; 8]).is_none());
    }

    #[test_case]
    fn a_group_is_part_of_a_commands_identity() {
        // The same id means different things per group, so a command sent with the wrong
        // group is not rejected — it is obeyed as something else.
        let a = rx_bytes(GROUP_LEGACY, 0x01, 0, &[], 0);
        let b = rx_bytes(GROUP_DATA_PATH, 0x01, 0, &[], 0);
        assert!(parse_rx(&a).unwrap().is_alive());
        assert!(!parse_rx(&b).unwrap().is_alive(), "group ignored when matching ALIVE");
    }

    #[test_case]
    fn a_command_is_a_header_then_its_payload() {
        let c = build_command(GROUP_SYSTEM, 0x1c, 0x0304, &[0xaa, 0xbb, 0xcc]);
        assert_eq!(c.len(), CMD_HEADER_WIDE_LEN + 3);
        assert_eq!(c[0], 0x1c, "cmd id first");
        assert_eq!(c[1], GROUP_SYSTEM);
        assert_eq!(&c[2..4], &0x0304u16.to_le_bytes(), "sequence is little-endian");
        assert_eq!(&c[4..6], &3u16.to_le_bytes(), "the length counts the payload only");
        assert_eq!(&c[6..8], &[0, 0]);
        assert_eq!(&c[8..], &[0xaa, 0xbb, 0xcc]);

        // It must parse as the packet firmware would echo, so the encoder and the decoder
        // cannot drift apart.
        let mut framed = ((c.len() as u32).to_le_bytes()).to_vec();
        framed.extend_from_slice(&c);
        let p = parse_rx(&framed).unwrap();
        assert_eq!((p.group_id, p.cmd, p.sequence), (GROUP_SYSTEM, 0x1c, 0x0304));
        assert_eq!(p.payload, &[0xaa, 0xbb, 0xcc]);
    }

    #[test_case]
    fn a_command_is_split_so_its_first_twenty_bytes_are_staged() {
        // The device fetches the first transfer buffer into an internal staging buffer
        // instead of reading it in place, so those bytes must come from a separate aligned
        // allocation. A command described as one buffer is read with a partly-stale header.
        assert_eq!(split_command(8), CommandSplit { first: 8, rest: 0 });
        assert_eq!(split_command(20), CommandSplit { first: 20, rest: 0 });
        assert_eq!(split_command(21), CommandSplit { first: 20, rest: 1 });
        assert_eq!(split_command(200), CommandSplit { first: 20, rest: 180 });
        // Every split covers the whole command exactly.
        for len in 1..300usize {
            let s = split_command(len);
            assert_eq!(s.first + s.rest, len, "len {len} lost bytes in the split");
            assert!(s.first <= FIRST_TB_SIZE);
        }
        assert_eq!(FIRST_TB_ALIGN % FIRST_TB_SIZE.next_power_of_two(), 0);
    }

    #[test_case]
    fn the_doorbell_packs_the_queue_and_index_without_overlap() {
        // One register serves every queue, so an index bleeding into the id field rings a
        // different queue's doorbell — leaving this command unqueued while another queue is
        // told to run.
        assert_eq!(csr_doorbell(0, 1), 0x0000_0001);
        assert_eq!(csr_doorbell(4, 0x10), 0x0004_0010);
        assert_eq!(csr_doorbell(31, 0xffff), 0x001f_ffff);
        for q in [0u8, 1, 9, 31] {
            for idx in [0u16, 1, 15, 0xffff] {
                let v = csr_doorbell(q, idx);
                assert_eq!(v >> 16, q as u32, "queue lost for {q}/{idx}");
                assert_eq!(v & 0xffff, idx as u32, "index lost for {q}/{idx}");
            }
        }
    }

    /// The doorbell encoder lives in `csr` with the register it writes; re-exported here so
    /// the packing test sits next to the sequence packing it has to stay distinct from.
    fn csr_doorbell(queue: u8, idx: u16) -> u32 {
        super::super::csr::txq_doorbell(queue, idx)
    }

    #[test_case]
    fn the_queue_refuses_to_reuse_a_slot_still_in_flight() {
        // The failure this prevents: a wrapped write index overwriting a descriptor the
        // device has not read, which loses a command already counted as sent — so the driver
        // waits forever for a response to something the device never saw.
        let mut q = CmdQueue::new(4, 4).unwrap();
        assert_eq!(q.size_log2(), 2);
        let mut seqs = alloc::vec::Vec::new();
        for expect_slot in 0..4 {
            let (slot, seq) = q.claim().expect("a fresh queue has room");
            assert_eq!(slot, expect_slot);
            assert_eq!(sequence_queue(seq), 4, "the sequence must name its queue");
            assert_eq!(sequence_slot(seq) as usize, slot);
            seqs.push(seq);
        }
        assert!(q.claim().is_none(), "a full queue handed out a fifth slot");

        // Retiring the oldest frees exactly its slot, and the index wraps to it.
        assert!(q.retire(seqs[0]));
        assert_eq!(q.write_index(), 0);
        let (slot, _) = q.claim().unwrap();
        assert_eq!(slot, 0);
        assert!(q.claim().is_none());
    }

    #[test_case]
    fn a_duplicate_or_unknown_response_does_not_free_a_slot() {
        // Firmware retransmits, and a sequence is only 13 meaningful bits. Freeing a slot on
        // a response we cannot account for releases one that is still in use.
        let mut q = CmdQueue::new(0, 8).unwrap();
        let (_, seq) = q.claim().unwrap();
        assert!(q.is_inflight(seq));
        assert!(q.retire(seq));
        assert!(!q.is_inflight(seq));
        assert!(!q.retire(seq), "a duplicate response retired the slot twice");
        assert!(!q.retire(make_sequence(0, 7)), "an unsent sequence was retired");
        // A sequence naming a slot past the end of the queue must not index out of bounds.
        assert!(!q.retire(make_sequence(0, 200)));
        assert!(!q.is_inflight(make_sequence(0, 200)));

        // Slot counts the device cannot wrap by masking.
        assert!(CmdQueue::new(0, 0).is_none());
        assert!(CmdQueue::new(0, 6).is_none());
        assert!(CmdQueue::new(0, 512).is_none());
    }

    #[test_case]
    fn a_response_is_matched_by_group_and_command_as_well_as_sequence() {
        // Firmware sends unsolicited notifications constantly; matching on the sequence
        // alone eventually hands one to whatever is waiting for a reply.
        let b = rx_bytes(GROUP_SYSTEM, 0x1c, 0x0405, &[1], 0);
        let p = parse_rx(&b).unwrap();
        assert!(p.answers(GROUP_SYSTEM, 0x1c, 0x0405));
        assert!(!p.answers(GROUP_SYSTEM, 0x1d, 0x0405), "wrong command matched");
        assert!(!p.answers(GROUP_MAC_CONF, 0x1c, 0x0405), "wrong group matched");
        assert!(!p.answers(GROUP_SYSTEM, 0x1c, 0x0406), "wrong sequence matched");
    }

    #[test_case]
    fn firmware_that_announces_itself_unusable_is_not_taken_for_working() {
        // The alive notification carries firmware's own verdict, and 0xdead is a firmware
        // that started and cannot be used — indistinguishable from a good one if only the
        // notification's arrival is checked.
        let ok = rx_bytes(
            GROUP_LEGACY,
            UCODE_ALIVE_NTFY,
            0,
            &ALIVE_STATUS_OK.to_le_bytes(),
            0,
        );
        assert_eq!(parse_rx(&ok).unwrap().alive_status(), Some(ALIVE_STATUS_OK));
        let bad = rx_bytes(
            GROUP_LEGACY,
            UCODE_ALIVE_NTFY,
            0,
            &ALIVE_STATUS_ERR.to_le_bytes(),
            0,
        );
        assert_eq!(parse_rx(&bad).unwrap().alive_status(), Some(ALIVE_STATUS_ERR));
        // A notification too short to carry a status, and a packet that is not one at all.
        let runt = rx_bytes(GROUP_LEGACY, UCODE_ALIVE_NTFY, 0, &[1], 0);
        assert_eq!(parse_rx(&runt).unwrap().alive_status(), None);
        let other = rx_bytes(GROUP_LEGACY, 0x33, 0, &[0xfe, 0xca], 0);
        assert_eq!(parse_rx(&other).unwrap().alive_status(), None);
    }

    #[test_case]
    fn the_nvm_response_is_decoded_with_its_real_field_widths() {
        // The general section is u32, u16, u8, u8 — not four words. This shipped as four
        // words first, which is a mistake with no symptom: `nvm_version` swallows the board
        // type and address count, `board_type` reads the SKU flags, and `n_hw_addrs` reads
        // the transmit chain mask, which is a small number and passes any sanity check. The
        // layout is Linux's `iwl_nvm_get_info_rsp`; this test is what pins it.
        let mut p = alloc::vec::Vec::new();
        p.extend_from_slice(&0x1u32.to_le_bytes()); // flags
        p.extend_from_slice(&0x0c11u16.to_le_bytes()); // nvm_version
        p.push(0x12); // board_type
        p.push(2); // n_hw_addrs
        p.extend_from_slice(&0xdead_beefu32.to_le_bytes()); // mac_sku_flags
        p.extend_from_slice(&3u32.to_le_bytes()); // tx_chains
        p.extend_from_slice(&2u32.to_le_bytes()); // rx_chains
        p.extend_from_slice(&[0xab; 16]); // regulatory, not decoded

        let n = NvmInfo::parse(&p).expect("a well-formed response must parse");
        assert_eq!(n.flags, 1);
        assert_eq!(n.nvm_version, 0x0c11);
        assert_eq!(n.board_type, 0x12);
        assert_eq!(n.n_hw_addrs, 2);
        assert_eq!(n.chains, Some((3, 2)), "the PHY section was read from the wrong offset");

        // A response carrying only the general section is legal on older API versions, and
        // the absence of a chain count is reported rather than invented.
        let short = NvmInfo::parse(&p[..NVM_GENERAL_LEN]).expect("the general section suffices");
        assert_eq!(short.n_hw_addrs, 2);
        assert_eq!(short.chains, None);

        // A response that is really a floating read, or one describing a device that cannot
        // address a frame: both mean the command was not answered, not that it was.
        let ones = alloc::vec![0xffu8; 32];
        assert!(NvmInfo::parse(&ones).is_none(), "a floating read was accepted as NVM data");
        let mut zero_addrs = p.clone();
        zero_addrs[7] = 0;
        assert!(NvmInfo::parse(&zero_addrs).is_none());
        let mut absurd = p.clone();
        absurd[7] = 99;
        assert!(NvmInfo::parse(&absurd).is_none());
        for n in 0..NVM_GENERAL_LEN {
            assert!(NvmInfo::parse(&p[..n]).is_none(), "a truncated response parsed");
        }
    }

    #[test_case]
    fn the_receive_buffer_list_refuses_addresses_the_device_would_write_to() {
        // A zero entry hands the device physical address 0 to write a packet into.
        assert!(build_rbd_list(&[]).is_none());
        assert!(build_rbd_list(&[0x1000, 0]).is_none());
        assert!(build_rbd_list(&[0x1001]).is_none(), "unaligned buffer accepted");
        assert_eq!(build_rbd_list(&[0x1000, 0x2000]).unwrap(), alloc::vec![0x1000, 0x2000]);
    }
}
