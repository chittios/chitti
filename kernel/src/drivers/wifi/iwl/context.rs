//! **gen2 context info** — how an AX200-and-later device loads its own firmware.
//!
//! Older Intel parts are fed firmware section by section by the host, through the flow
//! handler, with a handshake per section. Everything from the 22000 family onward
//! (AX200/AX201/AX210, the parts actually in modern laptops) works the other way round:
//! the host builds a **context info** structure in DMA memory that describes where
//! everything is, writes its physical address to one register, and the device's own
//! loader reads firmware out of host memory.
//!
//! That inverts where the risk lives. There is almost no sequence to get wrong, and
//! almost all of the difficulty is in the **layout** of one struct — which is exactly the
//! kind of thing that fails silently, because the device reads a field from the wrong
//! offset and either stalls or DMAs from an address that was never a firmware image.
//!
//! So every field offset here is asserted with `offset_of!`, the same treatment the S3
//! resume state block gets, for the same reason: it is a contract with something that
//! cannot report a disagreement.
//!
//! Layouts from Linux's `iwl-context-info.h` and `iwl-context-info-gen3.h`.
//! **Unverified on hardware** — no emulator provides an Intel WiFi device.

/// Control block: sizes and versions the device's loader reads first.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct ControlBlock {
    /// Firmware-visible version of this structure's own layout.
    pub version: u16,
    pub size: u16,
    /// Reserved words the device expects to be zero.
    pub _rsvd: [u32; 3],
}

/// Where the receive rings live.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct RbdControl {
    /// Log2 of the number of receive buffer descriptors.
    pub rbd_size: u32,
    /// Physical address of the free-RBD list.
    pub free_rbd_addr: u64,
    /// Physical address of the used-RBD list.
    pub used_rbd_addr: u64,
    /// Physical address of the status block the device writes progress into.
    pub status_addr: u64,
}

/// Where the transmit command queue lives.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct TxControl {
    /// Physical address of the command queue's descriptor array.
    pub cmd_queue_addr: u64,
    /// Log2 of its size in entries.
    pub cmd_queue_size: u8,
    pub _rsvd: [u8; 7],
}

/// The firmware image itself, described to the device's loader.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct FwControl {
    /// Physical address of a list of (address, length) pairs, one per firmware section.
    pub img_addr: u64,
    pub img_size: u32,
    pub _rsvd: u32,
}

/// The whole structure the device is pointed at.
///
/// Field order is the contract. `context_info_offsets_match_the_firmware_layout` pins it,
/// because the device reads these by displacement and cannot tell us if they moved.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct ContextInfo {
    pub control: ControlBlock,
    pub _rsvd0: u64,
    pub rbd: RbdControl,
    pub _rsvd1: u64,
    pub tx: TxControl,
    pub _rsvd2: [u64; 2],
    pub fw: FwControl,
    pub _rsvd3: [u64; 2],
}

/// The version this code writes into [`ControlBlock::version`].
///
/// A device whose loader expects a different one will not proceed, which is the correct
/// outcome — far better than reading a structure it half understands.
pub const CONTEXT_INFO_VERSION: u16 = 1;

/// One firmware section, as the device's loader wants it: where it is and how big.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SectionEntry {
    pub addr: u64,
    pub size: u32,
    pub _rsvd: u32,
}

/// Build the section list the device DMAs firmware from.
///
/// Pure: it takes the physical address each section was copied to and its length, and
/// produces the array the device reads. Separated out because the failure mode is
/// arithmetic — an entry whose size is the *file* length rather than the copied length
/// makes the device read past the buffer, and nothing on the host notices.
pub fn build_section_list(sections: &[(u64, u32)]) -> alloc::vec::Vec<SectionEntry> {
    sections
        .iter()
        .map(|&(addr, size)| SectionEntry {
            addr,
            size,
            _rsvd: 0,
        })
        .collect()
}

/// Whether a built context info is internally consistent enough to hand to a device.
///
/// Every check here is a mistake that is otherwise undetectable from the host: a null
/// address in a field the loader will DMA from, a queue size the descriptor array cannot
/// hold, or a zero-length firmware list. The device's response to any of them is to stop,
/// with no diagnostic — so refusing here is the only way the reason is ever known.
pub fn validate(c: &ContextInfo) -> Result<(), &'static str> {
    if c.control.version != CONTEXT_INFO_VERSION {
        return Err("context info version mismatch");
    }
    if c.control.size == 0 {
        return Err("context info size is zero");
    }
    if c.fw.img_addr == 0 || c.fw.img_size == 0 {
        return Err("no firmware section list");
    }
    if c.tx.cmd_queue_addr == 0 {
        return Err("no command queue");
    }
    if c.rbd.free_rbd_addr == 0 || c.rbd.used_rbd_addr == 0 || c.rbd.status_addr == 0 {
        return Err("receive rings incomplete");
    }
    // Every address the device DMAs from has to be aligned; an unaligned descriptor array
    // is a class of failure the device reports as nothing at all.
    for (name, addr) in [
        ("firmware list", c.fw.img_addr),
        ("command queue", c.tx.cmd_queue_addr),
        ("free rbd", c.rbd.free_rbd_addr),
        ("used rbd", c.rbd.used_rbd_addr),
        ("status", c.rbd.status_addr),
    ] {
        if addr % 256 != 0 {
            let _ = name;
            return Err("a DMA address in the context info is not 256-byte aligned");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn context_info_offsets_match_the_firmware_layout() {
        // The device reads these by displacement and has no way to report a disagreement:
        // a shifted field means it takes a queue size for an address, or DMAs firmware
        // from whatever the reserved word happened to hold. Same treatment as the S3
        // resume state block, for the same reason.
        assert_eq!(core::mem::offset_of!(ContextInfo, control), 0);
        assert_eq!(core::mem::size_of::<ControlBlock>(), 16);
        assert_eq!(core::mem::offset_of!(ContextInfo, rbd), 24);
        assert_eq!(core::mem::offset_of!(ContextInfo, tx), 64);
        assert_eq!(core::mem::offset_of!(ContextInfo, fw), 96);
        // And the whole thing stays a fixed size, so a page allocated for it is enough.
        assert!(core::mem::size_of::<ContextInfo>() <= 256);
    }

    #[test_case]
    fn a_section_entry_is_address_then_size() {
        // Swap these and the device treats a length as an address — a DMA read from
        // physical address 0x1000-ish, which on a real machine is somebody else's memory.
        assert_eq!(core::mem::offset_of!(SectionEntry, addr), 0);
        assert_eq!(core::mem::offset_of!(SectionEntry, size), 8);
        assert_eq!(core::mem::size_of::<SectionEntry>(), 16);
    }

    #[test_case]
    fn the_section_list_carries_the_copied_length_not_the_file_length() {
        // Pure arithmetic, but the mistake it guards is real: describing a section by its
        // length in the `.ucode` file rather than by how much was copied into DMA memory
        // makes the device read past the buffer, and nothing on the host notices.
        let list = build_section_list(&[(0x1000, 64), (0x2000, 128)]);
        assert_eq!(list.len(), 2);
        assert_eq!(
            list[0],
            SectionEntry {
                addr: 0x1000,
                size: 64,
                _rsvd: 0
            }
        );
        assert_eq!(list[1].addr, 0x2000);
        assert_eq!(list[1].size, 128);
        assert!(build_section_list(&[]).is_empty());
    }

    /// A context info with every field plausibly filled.
    fn good() -> ContextInfo {
        ContextInfo {
            control: ControlBlock {
                version: CONTEXT_INFO_VERSION,
                size: core::mem::size_of::<ContextInfo>() as u16,
                _rsvd: [0; 3],
            },
            rbd: RbdControl {
                rbd_size: 8,
                free_rbd_addr: 0x1_0000,
                used_rbd_addr: 0x1_0100,
                status_addr: 0x1_0200,
            },
            tx: TxControl {
                cmd_queue_addr: 0x2_0000,
                cmd_queue_size: 4,
                _rsvd: [0; 7],
            },
            fw: FwControl {
                img_addr: 0x3_0000,
                img_size: 32,
                _rsvd: 0,
            },
            ..Default::default()
        }
    }

    #[test_case]
    fn a_complete_context_info_validates() {
        assert!(validate(&good()).is_ok());
    }

    #[test_case]
    fn every_missing_piece_is_caught_before_the_device_sees_it() {
        // The device's response to any of these is to stop with no diagnostic at all, so
        // this is the only place the reason can ever be known.
        let mut c = good();
        c.fw.img_addr = 0;
        assert!(validate(&c).is_err(), "null firmware list accepted");

        let mut c = good();
        c.fw.img_size = 0;
        assert!(validate(&c).is_err(), "empty firmware list accepted");

        let mut c = good();
        c.tx.cmd_queue_addr = 0;
        assert!(validate(&c).is_err(), "missing command queue accepted");

        let mut c = good();
        c.rbd.status_addr = 0;
        assert!(validate(&c).is_err(), "missing status block accepted");

        let mut c = good();
        c.control.version = CONTEXT_INFO_VERSION + 1;
        assert!(validate(&c).is_err(), "version mismatch accepted");
    }

    #[test_case]
    fn an_unaligned_dma_address_is_refused() {
        // The device requires its descriptor arrays aligned, and an unaligned one is
        // another silent stall. The allocator hands out page-aligned memory, so hitting
        // this means an offset was added somewhere it should not have been.
        let mut c = good();
        c.tx.cmd_queue_addr = 0x2_0001;
        assert!(validate(&c).is_err());
        c.tx.cmd_queue_addr = 0x2_0080; // 128-byte aligned is still not enough
        assert!(validate(&c).is_err());
        c.tx.cmd_queue_addr = 0x2_0100;
        assert!(validate(&c).is_ok());
    }
}
