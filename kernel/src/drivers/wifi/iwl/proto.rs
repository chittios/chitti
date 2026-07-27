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

impl RxPacket<'_> {
    /// Whether this is firmware announcing it is alive.
    pub fn is_alive(&self) -> bool {
        self.group_id == GROUP_LEGACY && self.cmd == UCODE_ALIVE_NTFY
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
    fn the_receive_buffer_list_refuses_addresses_the_device_would_write_to() {
        // A zero entry hands the device physical address 0 to write a packet into.
        assert!(build_rbd_list(&[]).is_none());
        assert!(build_rbd_list(&[0x1000, 0]).is_none());
        assert!(build_rbd_list(&[0x1001]).is_none(), "unaligned buffer accepted");
        assert_eq!(build_rbd_list(&[0x1000, 0x2000]).unwrap(), alloc::vec![0x1000, 0x2000]);
    }
}
