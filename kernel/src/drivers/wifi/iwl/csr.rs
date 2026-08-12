//! **Intel WiFi control and status registers** — the register map, and the pure
//! predicates over it.
//!
//! Everything an `iwlwifi` bring-up does is a read-modify-write of one of these, gated on
//! a bit becoming set. The offsets and bit positions live here on their own, away from the
//! sequences that use them, for the reason the FADT taught this codebase: a wrong offset
//! does not fail loudly, it reads a plausible value from the neighbouring register — and
//! the only defence is a named constant with a test on it.
//!
//! Offsets are from Linux's `iwl-csr.h`. **Unverified on hardware**: QEMU emulates no
//! Intel WiFi part, so nothing here has ever addressed a real device, and the value of the
//! tests is that they pin the numbers rather than that they prove them.
//!
//! ## Two register spaces, not one
//!
//! The device presents its own CSRs directly in BAR0, but the *peripheral* registers
//! (`prph`) behind them are reached indirectly, by writing an address to
//! [`HBUS_TARG_PRPH_WADDR`] and then the data — and the read side needs a matching
//! address/data pair. Confusing the two is the classic `iwlwifi` porting bug: a prph
//! offset written as if it were a CSR lands in whatever CSR shares that number.

/// Hardware revision. Read first — its family bits decide which bring-up path applies.
pub const CSR_HW_REV: usize = 0x028;

/// Interface configuration: bus type and which silicon step this is.
pub const CSR_HW_IF_CONFIG_REG: usize = 0x000;

/// Interrupt status and mask. The driver polls, so the mask stays clear.
pub const CSR_INT: usize = 0x008;
pub const CSR_INT_MASK: usize = 0x00c;
/// Flow-handler interrupt status — DMA completions land here.
pub const CSR_FH_INT_STATUS: usize = 0x010;

/// Reset register.
pub const CSR_RESET: usize = 0x020;
/// General-purpose control: the clock and MAC-access handshakes live here.
pub const CSR_GP_CNTRL: usize = 0x024;

/// EEPROM/OTP general purpose.
pub const CSR_EEPROM_GP: usize = 0x030;
/// Analogue PLL configuration.
pub const CSR_ANA_PLL_CFG: usize = 0x20c;
/// Chicken bits — one of them disables an L1 power-saving behaviour that breaks DMA.
pub const CSR_GIO_CHICKEN_BITS: usize = 0x100;
/// Mailbox set register, used to tell firmware the host is alive.
pub const CSR_MBOX_SET_REG: usize = 0x088;

/// Doorbell that starts a gen2 firmware load once the context info is in place.
pub const CSR_CTXT_INFO_BA: usize = 0x40;
/// gen3 devices take the context info address in a different pair of registers.
pub const CSR_CTXT_INFO_ADDR: usize = 0x118;
pub const CSR_IML_DATA_ADDR: usize = 0x120;
pub const CSR_IML_SIZE_ADDR: usize = 0x128;

// --- indirect (peripheral) register access --------------------------------

/// Write address, then data, to reach a `prph` register.
pub const HBUS_TARG_PRPH_WADDR: usize = 0x44c;
pub const HBUS_TARG_PRPH_WDAT: usize = 0x450;
pub const HBUS_TARG_PRPH_RADDR: usize = 0x448;
pub const HBUS_TARG_PRPH_RDAT: usize = 0x454;

/// The `prph` write address register wants the target address with this bit set, which is
/// what marks it a 4-byte access. Writing the bare address reads back zero and the write
/// silently goes nowhere.
///
/// The address field is **20 bits, and the truncation is deliberate**: peripheral
/// addresses are written in Linux as full `0x00Axxxxx` values (`WFPM_GP2` is `0xA030B4`),
/// but the hardware supplies the `0xA00000` base itself, so only the low 20 bits travel.
/// Widening this mask to "fix" an address that looks truncated would put base bits into
/// the access-size field and change the width of every access.
pub const PRPH_ADDR_MASK: u32 = 0x000f_ffff;
pub const PRPH_WRITE_ENABLE: u32 = 0x3 << 24;
pub const PRPH_READ_ENABLE: u32 = 0x3 << 24;

// --- the device's own MAC address -----------------------------------------

/// The MAC address as strapped on the board, and as burned into the part's OTP.
///
/// Two sources because either can be blank: a board that straps the address leaves the OTP
/// words zero, and vice versa. Preferring one unconditionally gets an all-zero address on
/// half the machines.
pub const CSR_MAC_ADDR0_STRAP: usize = 0x380;
pub const CSR_MAC_ADDR1_STRAP: usize = 0x384;
pub const CSR_MAC_ADDR0_OTP: usize = 0x388;
pub const CSR_MAC_ADDR1_OTP: usize = 0x38c;

/// Assemble a MAC address from the two register words.
///
/// The byte order is the part that cannot be guessed: the first word holds the leading four
/// bytes **big-endian** while the register itself reads little-endian, and the second word
/// contributes only its low two bytes, also reversed. A straightforward
/// `to_le_bytes`-and-concatenate produces a plausible-looking address that is wrong in a way
/// nothing local detects — the traffic simply never comes back.
///
/// `None` when the words describe no address: all-zero (unwritten), all-ones (a floating
/// read), or a multicast address, which a station's own address never is.
pub fn mac_from_words(w0: u32, w1: u32) -> Option<[u8; 6]> {
    let mac = [
        (w0 >> 24) as u8,
        (w0 >> 16) as u8,
        (w0 >> 8) as u8,
        w0 as u8,
        (w1 >> 8) as u8,
        w1 as u8,
    ];
    if mac.iter().all(|&b| b == 0) || mac.iter().all(|&b| b == 0xff) {
        return None;
    }
    if mac[0] & 1 != 0 {
        return None; // a multicast address is not an interface's own
    }
    Some(mac)
}

// --- transmit and receive doorbells ---------------------------------------

/// The transmit-queue doorbell. One register serves every queue: the queue id and the new
/// write index are packed into the same word.
pub const HBUS_TARG_WRPTR: usize = 0x460;

/// Peripheral address of a receive queue's free-buffer write index, per queue.
///
/// A `prph` address, not a CSR — so it goes through [`prph_write_addr`], and using it as a
/// direct BAR0 offset lands in an unrelated register a long way away.
pub const RFH_Q_FRBDCB_WIDX_TRANS: u32 = 0x00A0_8080;

/// Pack a queue id and write index for [`HBUS_TARG_WRPTR`].
///
/// The index is what tells the device how far the host has filled the ring, and the queue
/// id selects which ring — in the same word, so an unmasked index bleeding into the id
/// field rings a *different* queue's doorbell. That is not a lost command: it is a command
/// left unqueued while some other queue is told to run.
pub fn txq_doorbell(queue: u8, write_index: u16) -> u32 {
    ((queue as u32) << 16) | (write_index as u32 & 0xffff)
}

/// Peripheral address of the free-buffer write index for `queue`.
pub fn frbdcb_widx(queue: u8) -> u32 {
    RFH_Q_FRBDCB_WIDX_TRANS + (queue as u32) * 4
}

// --- CSR_GP_CNTRL bits ----------------------------------------------------

/// Ask for access to the MAC's clock domain.
pub const GP_CNTRL_MAC_ACCESS_REQ: u32 = 1 << 3;
/// Access granted — the bit to wait for.
pub const GP_CNTRL_MAC_ACCESS_EN: u32 = 1 << 0;
/// The MAC clock is running.
pub const GP_CNTRL_MAC_CLOCK_READY: u32 = 1 << 0;
/// Initialisation is complete.
pub const GP_CNTRL_INIT_DONE: u32 = 1 << 2;
/// Device is in low power; must be cleared before anything else works.
pub const GP_CNTRL_GOING_TO_SLEEP: u32 = 1 << 4;
/// Reset the device's power management state.
pub const GP_CNTRL_SW_RESET: u32 = 1 << 31;

// --- CSR_RESET bits -------------------------------------------------------

/// Software reset. Self-clearing.
pub const RESET_SW: u32 = 1 << 7;
/// Stop the master DMA engine.
pub const RESET_STOP_MASTER: u32 = 1 << 9;
/// Master is now idle — the bit to wait for after asking it to stop.
pub const RESET_MASTER_DISABLED: u32 = 1 << 8;

// --- CSR_HW_IF_CONFIG_REG bits -------------------------------------------

/// Tell the device this is a PCIe host and prepare it.
pub const HW_IF_CONFIG_PREPARE: u32 = 1 << 27;
/// Set while preparation is in progress; must go clear.
pub const HW_IF_CONFIG_NIC_READY: u32 = 1 << 22;
/// Reflects that firmware owns the device.
pub const HW_IF_CONFIG_HAP_WAKE: u32 = 1 << 23;

// --- CSR_GIO_CHICKEN_BITS ------------------------------------------------

/// Disable the L1-active retry behaviour. Left enabled, DMA can stall on a link that
/// enters L1 mid-transfer — the kind of fault that presents as an occasional hang rather
/// than a clean failure.
pub const GIO_CHICKEN_L1A_NO_L0S_RX: u32 = 1 << 23;

/// Encode a context-info physical address for the gen2 doorbell.
///
/// The register holds the address **shifted right by 4** — it has no room for the low
/// bits, so they must already be zero. `None` when they are not, or when the shifted
/// value will not fit 32 bits: both would hand the device's loader a *different* address
/// than the structure is at, and its response is to stop with nothing reported.
///
/// Pure and tested because a bare `>> 4` in the middle of a bring-up sequence is the shape
/// of bug this codebase keeps paying for — correct today, silently wrong the moment
/// someone adds an offset to the allocation.
pub fn ctxt_info_ba(phys: u64) -> Option<u32> {
    if phys == 0 || phys % 16 != 0 {
        return None;
    }
    let shifted = phys >> 4;
    if shifted > u32::MAX as u64 {
        return None;
    }
    Some(shifted as u32)
}

/// All-ones is what an unmapped or powered-down BAR reads back.
///
/// The same rule as the embedded controller's `0xff` and the PM1 status word: a register
/// space that answers every bit set has not answered at all, and treating it as data is
/// how a driver decides a dead device is ready.
pub fn is_floating(v: u32) -> bool {
    v == u32::MAX
}

/// Whether the hardware revision looks like a real device rather than a floating bus.
///
/// `CSR_HW_REV` is the first read of a bring-up, so it is also the cheapest place to find
/// out the BAR did not map. Zero is as suspect as all-ones: a mapped-but-unpowered
/// function reads zeroes.
pub fn hw_rev_plausible(v: u32) -> bool {
    !is_floating(v) && v != 0
}

/// The device family, from `CSR_HW_REV`'s step/dash bits.
///
/// Only the distinction that changes the bring-up path: gen2-and-later devices load
/// firmware by being handed a *context info* structure, where earlier ones are fed section
/// by section. Getting this wrong means starting the wrong load sequence entirely.
pub fn hw_rev_step(hw_rev: u32) -> u32 {
    // Linux: `CSR_HW_REV_STEP(hw_rev)` — bits 2..4.
    (hw_rev >> 2) & 0x3
}

/// Encode an address for the indirect `prph` write path.
///
/// Pure because the encoding is the bug: the address needs the access-size bits, and a
/// bare address makes the write vanish with no error anywhere.
pub fn prph_write_addr(addr: u32) -> u32 {
    (addr & PRPH_ADDR_MASK) | PRPH_WRITE_ENABLE
}

/// Encode an address for the indirect `prph` read path.
pub fn prph_read_addr(addr: u32) -> u32 {
    (addr & PRPH_ADDR_MASK) | PRPH_READ_ENABLE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn a_floating_bus_is_not_a_device() {
        // The rule this codebase keeps relearning: all-ones means nothing answered.
        assert!(is_floating(u32::MAX));
        assert!(!is_floating(0x0000_0100));
        // And zero is equally suspect for a revision register — a mapped but unpowered
        // function reads zeroes, which `hw_rev_plausible` must not accept as a chip id.
        assert!(!hw_rev_plausible(u32::MAX));
        assert!(!hw_rev_plausible(0));
        assert!(hw_rev_plausible(0x0000_0350));
    }

    #[test_case]
    fn prph_addresses_carry_their_access_size_bits() {
        // A bare address in the prph write-address register makes the following data write
        // go nowhere, with nothing reported. The encoding is the whole difference between
        // a configured device and a silent one.
        assert_ne!(
            prph_write_addr(0x1234) & PRPH_WRITE_ENABLE,
            0,
            "access-size bits missing"
        );
        assert_ne!(prph_read_addr(0x1234) & PRPH_READ_ENABLE, 0);
        // An address that already fits 20 bits survives intact.
        assert_eq!(prph_write_addr(0x0003_0b4) & PRPH_ADDR_MASK, 0x0003_0b4);
    }

    #[test_case]
    fn a_peripheral_address_is_truncated_to_twenty_bits_on_purpose() {
        // Linux writes these as full `0x00Axxxxx` values — `WFPM_GP2` is `0xA030B4` — and
        // masks to 20 bits, because the hardware supplies the `0xA00000` base itself. So a
        // truncated-looking result is correct, and this is the test that stops someone
        // "fixing" the mask: a wider one would put base bits into the access-size field and
        // silently change the width of every peripheral access.
        assert_eq!(prph_write_addr(0x00a0_30b4) & PRPH_ADDR_MASK, 0x0003_0b4);
        assert_eq!(prph_read_addr(0x00a0_30b4) & PRPH_ADDR_MASK, 0x0003_0b4);
        // And nothing above the field ever reaches the control bits.
        assert_eq!(
            prph_write_addr(0xffff_ffff),
            PRPH_ADDR_MASK | PRPH_WRITE_ENABLE
        );
    }

    #[test_case]
    fn a_mac_address_is_assembled_in_the_order_the_registers_hold_it() {
        // The first word's four bytes are big-endian and the second contributes only its low
        // two, reversed. A `to_le_bytes`-and-concatenate gives a plausible address that is
        // wrong in a way nothing local detects — traffic just never comes back — so the byte
        // order is pinned here rather than discovered on a laptop.
        assert_eq!(
            mac_from_words(0x0011_2233, 0x0000_4455),
            Some([0x00, 0x11, 0x22, 0x33, 0x44, 0x55])
        );
        // The high half of the second word is not part of the address.
        assert_eq!(
            mac_from_words(0x0011_2233, 0xdead_4455),
            Some([0x00, 0x11, 0x22, 0x33, 0x44, 0x55])
        );
    }

    #[test_case]
    fn an_absent_or_impossible_mac_address_is_refused() {
        // Both sources exist because either can be blank, so "unwritten" has to be
        // distinguishable from an address — otherwise the interface comes up as 00:00:00:…
        // and every frame it sends is dropped by the access point.
        assert_eq!(
            mac_from_words(0, 0),
            None,
            "an unwritten address was accepted"
        );
        assert_eq!(
            mac_from_words(u32::MAX, u32::MAX),
            None,
            "a floating read was accepted as an address"
        );
        // Bit 0 of the first byte is the multicast bit; an interface's own address is never
        // multicast, so this is a misread rather than a valid value.
        assert_eq!(mac_from_words(0x0111_2233, 0x0000_4455), None);
        // A locally-administered but unicast address is legitimate.
        assert!(mac_from_words(0x0211_2233, 0x0000_4455).is_some());
    }

    #[test_case]
    fn the_two_register_spaces_do_not_share_offsets_by_accident() {
        // The classic porting bug is using a prph offset as a CSR. These are different
        // spaces, so an overlap in the *numbers* is not itself wrong — but the indirect
        // access registers must not collide with the CSRs the reset sequence touches, or
        // one sequence would clobber the other's state.
        let csrs = [
            CSR_HW_REV,
            CSR_HW_IF_CONFIG_REG,
            CSR_INT,
            CSR_INT_MASK,
            CSR_FH_INT_STATUS,
            CSR_RESET,
            CSR_GP_CNTRL,
            CSR_EEPROM_GP,
            CSR_GIO_CHICKEN_BITS,
            CSR_MBOX_SET_REG,
        ];
        let hbus = [
            HBUS_TARG_PRPH_WADDR,
            HBUS_TARG_PRPH_WDAT,
            HBUS_TARG_PRPH_RADDR,
            HBUS_TARG_PRPH_RDAT,
        ];
        for c in csrs {
            for h in hbus {
                assert_ne!(c, h, "CSR {c:#x} collides with an HBUS register");
            }
        }
        // And every CSR offset is distinct: a duplicate would mean two names for one
        // register, which is how a sequence ends up writing the wrong one.
        for (i, a) in csrs.iter().enumerate() {
            for b in csrs.iter().skip(i + 1) {
                assert_ne!(a, b, "duplicate CSR offset {a:#x}");
            }
        }
    }

    #[test_case]
    fn the_reset_and_access_bits_are_the_spec_positions() {
        // Pinned individually because a wrong bit here waits forever for a flag that will
        // never set — and a bounded wait then reports "device did not respond" for a
        // device that was responding perfectly.
        assert_eq!(RESET_SW, 0x80);
        assert_eq!(RESET_STOP_MASTER, 0x200);
        assert_eq!(RESET_MASTER_DISABLED, 0x100);
        assert_eq!(GP_CNTRL_MAC_ACCESS_REQ, 0x8);
        assert_eq!(GP_CNTRL_MAC_ACCESS_EN, 0x1);
        assert_eq!(GP_CNTRL_INIT_DONE, 0x4);
        assert_eq!(HW_IF_CONFIG_PREPARE, 0x0800_0000);
        assert_eq!(HW_IF_CONFIG_NIC_READY, 0x0040_0000);
    }

    #[test_case]
    fn the_gen2_doorbell_address_is_shifted_and_checked() {
        // The register has no room for the low four bits, so an address that has any is
        // not representable — and silently truncating it points the device's loader at a
        // structure that is not there.
        assert_eq!(ctxt_info_ba(0x1234_0000), Some(0x0123_4000));
        assert_eq!(ctxt_info_ba(0x10), Some(1));
        assert_eq!(ctxt_info_ba(0), None, "a null address is not a location");
        assert_eq!(
            ctxt_info_ba(0x1234_0008),
            None,
            "unaligned address accepted"
        );
        // The shifted value still has to fit the register: 36 bits of address does, more
        // does not, and a wrapped write is another wrong address.
        assert_eq!(ctxt_info_ba(0xf_ffff_fff0), Some(0xffff_ffff));
        assert_eq!(ctxt_info_ba(0x10_0000_0000), None);
    }

    #[test_case]
    fn the_hardware_step_is_read_from_the_documented_bits() {
        // Step 2..3 of CSR_HW_REV. It selects the firmware-load path, so reading the wrong
        // bits starts the wrong sequence on a real device.
        assert_eq!(hw_rev_step(0b0000), 0);
        assert_eq!(hw_rev_step(0b0100), 1);
        assert_eq!(hw_rev_step(0b1000), 2);
        assert_eq!(hw_rev_step(0b1100), 3);
        // Bits above the field must not leak in.
        assert_eq!(hw_rev_step(0xffff_fff0 | 0b0100), 1);
    }
}
