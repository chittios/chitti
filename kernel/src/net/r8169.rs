//! **Realtek RTL8169/8168/8111/8101/8125 over PCI** (the Linux `r8169` family) —
//! the most common Ethernet controller in consumer PCs and laptops. If someone
//! installs ChittiOS on an existing Windows desktop, this is very likely the NIC
//! it has.
//!
//! Realtek's descriptor model is nothing like Intel's: 16-byte descriptors whose
//! **ownership bit lives in the descriptor itself** (`DescOwn`, bit 31 of
//! `opts1`) rather than in a head/tail register pair. There is no tail pointer to
//! advance — the driver hands a descriptor over by setting `DescOwn` and the NIC
//! clears it on completion; transmission is kicked with a one-byte write to
//! `TxPoll`. The last descriptor of each ring must carry the `RingEnd` bit.
//!
//! Register block: `BAR2` on the PCIe parts (8168/8101/8125), `BAR1` on the
//! original PCI 8169. Rather than keep a per-chip table, [`Rtl8169::init`] takes
//! the first BAR that decodes as memory.
//!
//! ## Verification status
//! **This driver is unverified on hardware.** QEMU emulates `rtl8139` but no
//! r8169-family part, so there is no way to exercise it in this tree's test
//! setup. The register offsets, bit definitions and bring-up order are
//! transcribed from the RTL8168 datasheet and Linux `drivers/net/ethernet/
//! realtek/r8169_main.c`; the descriptor handling mirrors `rtl8169_start_xmit` /
//! `rtl_rx`. Every step logs through `ktrace`, and a failure to reset or link is
//! reported explicitly, so a first boot on real hardware should be diagnosable
//! from one log rather than presenting as a silent NIC.
//!
//! ## Known omissions
//! * **PHY firmware.** Linux loads `rtl_nic/rtl8168*.fw` per chip revision. It is
//!   an erratum/performance patch, not required to link (Linux continues without
//!   it), so it is not loaded here.
//! * **OCP register access** used by RTL8168g-and-later for ASPM/EEE tuning.
//!   Absent; on a laptop with aggressive ASPM this could affect throughput or, in
//!   the worst case, link stability.
//! * **RTL8125 (2.5GbE)** is dispatched here because Linux drives it with the same
//!   driver, but it uses an extended descriptor layout on some paths. Treat 8125
//!   support as the least certain part of this file.
//! * No interrupts (poll-only), no checksum offload, no VLAN, one ring pair.

use crate::net::NetDevice;
use core::ptr::{read_volatile, write_volatile};
use core::sync::atomic::{fence, Ordering};

#[cfg(target_arch = "aarch64")]
use crate::pci::PciDevice;
#[cfg(target_arch = "x86_64")]
use crate::arch::x86_64::pci::PciDevice;

// --- register offsets ----------------------------------------------------
const MAC0: usize = 0x00; // 6 bytes of station address
const MAR0: usize = 0x08; // multicast filter, 8 bytes
const TX_DESC_LO: usize = 0x20;
const TX_DESC_HI: usize = 0x24;
const CHIP_CMD: usize = 0x37; // u8
const TX_POLL: usize = 0x38; // u8   (8168; the 8125 moves it — see `RegMap`)
const INTR_MASK: usize = 0x3c; // u16  (8168)
const INTR_STATUS: usize = 0x3e; // u16 (8168)

// --- the three registers the RTL8125 moves --------------------------------
//
// 2.5GbE parts are dispatched to this driver because Linux drives them with it, but
// they are not register-compatible everywhere: the interrupt mask and status widen to
// 32 bits and move to 0x38/0x3c, and the transmit doorbell moves to 0x90. The 8168
// offsets overlap those new positions, so driving an 8125 with them writes the transmit
// doorbell into the interrupt mask — which is why this is a per-chip map rather than a
// comment saying "treat 8125 with caution".
//
// Offsets from Linux's `r8169_main.c` register map. **Unverified on hardware**: QEMU
// models no r8169-family part at all, so the value of splitting them is that an 8125 is
// now driven with its own offsets instead of another chip's, not that it is proven.

/// RTL8125 interrupt mask — 32-bit, at the offset the 8168 uses for the doorbell.
const INTR_MASK_8125: usize = 0x38;
/// RTL8125 interrupt status — 32-bit.
const INTR_STATUS_8125: usize = 0x3c;
/// RTL8125 transmit doorbell.
const TX_POLL_8125: usize = 0x90;

/// Which registers this chip actually has, and how wide.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegMap {
    pub intr_mask: usize,
    pub intr_status: usize,
    pub tx_poll: usize,
    /// True on the 8125, whose interrupt registers are 32 bits rather than 16.
    pub wide_intr: bool,
}

/// PCI device ids of the 2.5GbE parts.
const RTL8125_IDS: &[u16] = &[0x8125, 0x3000];

/// The register map for a Realtek device id.
///
/// Pure, so the one thing that decides whether an 8125 is driven correctly is testable
/// without the card — which matters more than usual here, because there is no emulated
/// r8169 anywhere to catch a mistake.
pub fn reg_map(device: u16) -> RegMap {
    if RTL8125_IDS.contains(&device) {
        RegMap {
            intr_mask: INTR_MASK_8125,
            intr_status: INTR_STATUS_8125,
            tx_poll: TX_POLL_8125,
            wide_intr: true,
        }
    } else {
        RegMap {
            intr_mask: INTR_MASK,
            intr_status: INTR_STATUS,
            tx_poll: TX_POLL,
            wide_intr: false,
        }
    }
}
const TX_CONFIG: usize = 0x40;
const RX_CONFIG: usize = 0x44;
const CFG_9346: usize = 0x50; // u8 — register write lock
const PHY_STATUS: usize = 0x6c; // u8
const RX_MAX_SIZE: usize = 0xda; // u16
const CPLUS_CMD: usize = 0xe0; // u16
const RX_DESC_LO: usize = 0xe4;
const RX_DESC_HI: usize = 0xe8;
const MAX_TX_PACKET_SIZE: usize = 0xec; // u8, in 128-byte units

// ChipCmd (0x37) bits
const CMD_RESET: u8 = 0x10;
const CMD_RX_ENB: u8 = 0x08;
const CMD_TX_ENB: u8 = 0x04;

// Cfg9346 (0x50) — unlock before writing config registers, lock after.
const CFG9346_LOCK: u8 = 0x00;
const CFG9346_UNLOCK: u8 = 0xc0;

// TxPoll (0x38)
const TX_POLL_NPQ: u8 = 0x40; // kick the normal-priority queue

// RxConfig (0x44) accept bits
const ACCEPT_BROADCAST: u32 = 0x08;
const ACCEPT_MULTICAST: u32 = 0x04;
const ACCEPT_MY_PHYS: u32 = 0x02;
/// FIFO threshold (7 = no threshold) at bit 13, DMA burst (7 = unlimited) at bit 8.
const RX_FIFO_THRESH: u32 = 7 << 13;
const RX_DMA_BURST: u32 = 7 << 8;
/// TX DMA burst (7 = unlimited) at bit 8, plus the standard inter-frame gap.
const TX_DMA_BURST: u32 = 7 << 8;
const TX_INTERFRAME_GAP: u32 = 0x0300_0000;

// CPlusCmd (0xE0) — clear the debug/test/force-mode bits that must not be left
// set, and nothing else. Bit positions per the RTL8168 datasheet §CPlusCmd and
// Linux's `rtl_register_content` enum.
//
// Deliberately NOT cleared: `PCIDAC` (bit 4) and `PCIMulRW` (bit 3) govern DMA
// addressing and multiple read/write, and `Normal_mode` (bit 13) must stay set —
// clearing any of them breaks DMA rather than tidying it up. Whatever firmware
// left in the checksum/VLAN-offload bits (5, 6) is harmless to us because smoltcp
// validates checksums itself.
const CPCMD_CLEAR_MASK: u16 = (1 << 15) // EnableBist
    | (1 << 14) // Mac_dbgo_oe
    | (1 << 12) // Force_half_dup
    | (1 << 11) // Force_rxflow_en
    | (1 << 10) // Force_txflow_en
    | (1 << 9)  // Cxpl_dbg_sel
    | (1 << 8)  // ASF (manageability)
    | 0x001c; // Mac_dbgo_sel
/// `Normal_mode` must be set for ordinary operation (bit 13).
const CPCMD_NORMAL_MODE: u16 = 1 << 13;

// PHYstatus (0x6C) bits
const PHY_LINK_OK: u8 = 0x02;
const PHY_FULL_DUP: u8 = 0x01;
const PHY_10: u8 = 0x04;
const PHY_100: u8 = 0x08;
const PHY_1000: u8 = 0x10;

// Descriptor opts1 bits
const DESC_OWN: u32 = 1 << 31; // owned by the NIC
const DESC_RING_END: u32 = 1 << 30; // last descriptor in the ring
const DESC_FIRST_FRAG: u32 = 1 << 29;
const DESC_LAST_FRAG: u32 = 1 << 28;
/// RX error summary; the frame in this descriptor is not usable.
const RX_RES: u32 = 0x0020_0000;
/// RX frame length lives in the low 14 bits and **includes the 4-byte CRC**.
const RX_LEN_MASK: u32 = 0x0000_3fff;

const NRX: usize = 32;
const NTX: usize = 32;
const BUFSZ: usize = 2048;
const DESC_SZ: usize = 16;
const MMIO_SPAN: usize = 0x1000;

/// Mask every interrupt and acknowledge whatever was pending, at whatever width and
/// offset this chip keeps those registers.
///
/// A 16-bit write to the 8125's 32-bit mask would leave the upper half whatever firmware
/// left it as — and this driver polls, so an unmasked source has nothing to service it.
fn mask_intr(regs: u64, map: &RegMap) {
    if map.wide_intr {
        w32(regs, map.intr_mask, 0);
        w32(regs, map.intr_status, 0xffff_ffff);
    } else {
        w16(regs, map.intr_mask, 0);
        w16(regs, map.intr_status, 0xffff);
    }
}

/// A poll-driven Realtek r8169-family NIC.
pub struct Rtl8169 {
    regs: u64,
    /// Where this chip's moving registers actually are.
    map: RegMap,
    mac: [u8; 6],
    rx_ring: u64,
    tx_ring: u64,
    rx_bufs: u64,
    tx_bufs: u64,
    tx_bufs_phys: u64,
    rx_cur: usize,
    tx_cur: usize,
}

#[inline]
fn r8(base: u64, off: usize) -> u8 {
    // SAFETY: `base` is the mapped register block; `off` within MMIO_SPAN.
    unsafe { read_volatile((base + off as u64) as *const u8) }
}
#[inline]
fn w8(base: u64, off: usize, v: u8) {
    // SAFETY: as `r8`; 8-bit register write.
    unsafe { write_volatile((base + off as u64) as *mut u8, v) };
}
#[inline]
fn r16(base: u64, off: usize) -> u16 {
    // SAFETY: as `r8`; 16-bit aligned register.
    unsafe { read_volatile((base + off as u64) as *const u16) }
}
#[inline]
fn w16(base: u64, off: usize, v: u16) {
    // SAFETY: as `r8`; 16-bit register write.
    unsafe { write_volatile((base + off as u64) as *mut u16, v) };
}
#[inline]
fn w32(base: u64, off: usize, v: u32) {
    // SAFETY: as `r8`; 32-bit register write.
    unsafe { write_volatile((base + off as u64) as *mut u32, v) };
}

impl Rtl8169 {
    /// Bring up the located Realtek NIC. Returns `None` if no memory BAR decodes,
    /// the chip refuses to reset, or DMA memory can't be allocated.
    pub fn init(d: PciDevice) -> Option<Rtl8169> {
        d.enable_bus_master();
        // Register window: BAR2 on the PCIe parts, BAR1 on the original PCI 8169.
        // `bar()` yields 0 for an I/O BAR, so take the first that decodes.
        let (bar, which) = [2u8, 1, 0]
            .into_iter()
            .find_map(|i| {
                let b = d.bar(i);
                if b != 0 {
                    Some((b, i))
                } else {
                    None
                }
            })?;
        crate::ktrace::log_fmt(format_args!("r8169: registers in BAR{which} at {bar:#x}"));
        let regs = crate::mm::map_mmio(bar, MMIO_SPAN);

        // Which registers this chip has. Done before the first write, because two of the
        // three that move are ones this very sequence touches.
        let map = reg_map(d.device);
        if map.wide_intr {
            crate::ktrace::log("r8169", "RTL8125-class part: 32-bit interrupt registers at 0x38/0x3c, doorbell at 0x90");
        }

        // Mask interrupts and clear any pending status — this driver polls.
        mask_intr(regs, &map);

        // Soft reset: self-clearing, spec allows ~100 us.
        w8(regs, CHIP_CMD, CMD_RESET);
        let mut reset_ok = false;
        for _ in 0..1_000_000 {
            if r8(regs, CHIP_CMD) & CMD_RESET == 0 {
                reset_ok = true;
                break;
            }
            core::hint::spin_loop();
        }
        if !reset_ok {
            crate::ktrace::log("r8169", "chip never cleared CmdReset -- aborting bring-up");
            return None;
        }

        // Station address is loaded from the EEPROM by the chip itself.
        let mut mac = [0u8; 6];
        for (i, b) in mac.iter_mut().enumerate() {
            *b = r8(regs, MAC0 + i);
        }

        // Descriptor rings + buffer pools.
        let (rx_ring_phys, rx_ring) = crate::mm::alloc_dma(NRX * DESC_SZ)?;
        let (tx_ring_phys, tx_ring) = crate::mm::alloc_dma(NTX * DESC_SZ)?;
        let (rx_bufs_phys, rx_bufs) = crate::mm::alloc_dma(NRX * BUFSZ)?;
        let (tx_bufs_phys, tx_bufs) = crate::mm::alloc_dma(NTX * BUFSZ)?;

        // RX descriptors: hand every one to the NIC immediately (DescOwn set),
        // buffer size in the low bits, RingEnd on the last.
        for i in 0..NRX {
            let p = rx_ring + (i * DESC_SZ) as u64;
            let mut opts1 = DESC_OWN | BUFSZ as u32;
            if i == NRX - 1 {
                opts1 |= DESC_RING_END;
            }
            // SAFETY: `p` is within the RX ring DMA region; 16-byte descriptor
            // laid out opts1, opts2, addr(64).
            unsafe {
                write_volatile(p as *mut u32, opts1);
                write_volatile((p + 4) as *mut u32, 0);
                write_volatile((p + 8) as *mut u64, rx_bufs_phys + (i * BUFSZ) as u64);
            }
        }
        // TX descriptors start unowned (free); RingEnd on the last.
        for i in 0..NTX {
            let p = tx_ring + (i * DESC_SZ) as u64;
            let opts1 = if i == NTX - 1 { DESC_RING_END } else { 0 };
            // SAFETY: within the TX ring DMA region.
            unsafe {
                write_volatile(p as *mut u32, opts1);
                write_volatile((p + 4) as *mut u32, 0);
                write_volatile((p + 8) as *mut u64, tx_bufs_phys + (i * BUFSZ) as u64);
            }
        }

        // Config registers are write-protected until Cfg9346 is unlocked.
        w8(regs, CFG_9346, CFG9346_UNLOCK);

        // Clear the debug/test bits out of CPlusCmd, keep everything else, and
        // make sure Normal_mode is on.
        let cp = (r16(regs, CPLUS_CMD) & !CPCMD_CLEAR_MASK) | CPCMD_NORMAL_MODE;
        w16(regs, CPLUS_CMD, cp);
        crate::ktrace::log_fmt(format_args!("r8169: CPlusCmd {:#06x}", cp));

        // Largest accepted RX frame, and the TX packet-size limit (128-byte units).
        w16(regs, RX_MAX_SIZE, BUFSZ as u16);
        w8(regs, MAX_TX_PACKET_SIZE, (BUFSZ / 128) as u8);

        // Ring base addresses must be programmed before Tx/Rx are enabled.
        w32(regs, TX_DESC_LO, tx_ring_phys as u32);
        w32(regs, TX_DESC_HI, (tx_ring_phys >> 32) as u32);
        w32(regs, RX_DESC_LO, rx_ring_phys as u32);
        w32(regs, RX_DESC_HI, (rx_ring_phys >> 32) as u32);

        // Enable the engines, then configure DMA bursts and the receive filter —
        // this order is what the datasheet's initialisation flow specifies.
        w8(regs, CHIP_CMD, CMD_TX_ENB | CMD_RX_ENB);
        w32(regs, TX_CONFIG, TX_DMA_BURST | TX_INTERFRAME_GAP);
        w32(
            regs,
            RX_CONFIG,
            RX_FIFO_THRESH | RX_DMA_BURST | ACCEPT_BROADCAST | ACCEPT_MULTICAST | ACCEPT_MY_PHYS,
        );
        // Accept all multicast groups (smoltcp does its own filtering).
        w32(regs, MAR0, 0xffff_ffff);
        w32(regs, MAR0 + 4, 0xffff_ffff);

        mask_intr(regs, &map); // still polling
        w8(regs, CFG_9346, CFG9346_LOCK);

        // Link state, bounded — auto-negotiation on a gigabit PHY takes a while.
        let mut phy = 0u8;
        for _ in 0..2_000_000 {
            phy = r8(regs, PHY_STATUS);
            if phy & PHY_LINK_OK != 0 {
                break;
            }
            core::hint::spin_loop();
        }
        let speed = if phy & PHY_1000 != 0 {
            "1000"
        } else if phy & PHY_100 != 0 {
            "100"
        } else if phy & PHY_10 != 0 {
            "10"
        } else {
            "?"
        };
        crate::ktrace::log_fmt(format_args!(
            "r8169: up (vendor {:04x} dev {:04x}), MAC {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}, link={} {}Mb/s {}",
            d.vendor,
            d.device,
            mac[0],
            mac[1],
            mac[2],
            mac[3],
            mac[4],
            mac[5],
            phy & PHY_LINK_OK != 0,
            speed,
            if phy & PHY_FULL_DUP != 0 { "full" } else { "half" }
        ));
        if phy & PHY_LINK_OK == 0 {
            crate::ktrace::log("r8169", "link down after bring-up -- check the cable; this driver is unverified on hardware, see the module docs");
        }

        Some(Rtl8169 { regs, map, mac, rx_ring, tx_ring, rx_bufs, tx_bufs, tx_bufs_phys, rx_cur: 0, tx_cur: 0 })
    }
}

impl NetDevice for Rtl8169 {
    fn mac(&self) -> [u8; 6] {
        self.mac
    }

    fn receive(&mut self, out: &mut [u8]) -> Option<usize> {
        let p = self.rx_ring + (self.rx_cur * DESC_SZ) as u64;
        // SAFETY: `p` addresses the current RX descriptor in DMA memory.
        let opts1 = unsafe { read_volatile(p as *const u32) };
        if opts1 & DESC_OWN != 0 {
            return None; // still owned by the NIC — nothing received here yet
        }
        fence(Ordering::Acquire);
        // Length includes the 4-byte CRC; a too-short or errored frame is dropped
        // but its descriptor is still recycled.
        let raw = (opts1 & RX_LEN_MASK) as usize;
        let n = if opts1 & RX_RES != 0 || raw < 4 {
            0
        } else {
            (raw - 4).min(out.len())
        };
        if n > 0 {
            let src = self.rx_bufs + (self.rx_cur * BUFSZ) as u64;
            // SAFETY: `src` is this descriptor's 2 KiB buffer; `n < raw <= BUFSZ`.
            unsafe { core::ptr::copy_nonoverlapping(src as *const u8, out.as_mut_ptr(), n) };
        }
        // Give the descriptor back: restore the buffer size and set DescOwn,
        // preserving RingEnd on the final slot.
        let mut back = DESC_OWN | BUFSZ as u32;
        if self.rx_cur == NRX - 1 {
            back |= DESC_RING_END;
        }
        fence(Ordering::Release);
        // SAFETY: handing the descriptor we just consumed back to the NIC.
        unsafe { write_volatile(p as *mut u32, back) };
        self.rx_cur = (self.rx_cur + 1) % NRX;
        Some(n)
    }

    fn transmit(&mut self, frame: &[u8]) {
        if frame.len() > BUFSZ {
            return;
        }
        let i = self.tx_cur;
        let p = self.tx_ring + (i * DESC_SZ) as u64;
        // SAFETY: reading the descriptor's ownership bit.
        if unsafe { read_volatile(p as *const u32) } & DESC_OWN != 0 {
            return; // ring full — the NIC still owns this slot; drop the frame
        }
        let buf = self.tx_bufs + (i * BUFSZ) as u64;
        let mut opts1 = DESC_OWN | DESC_FIRST_FRAG | DESC_LAST_FRAG | frame.len() as u32;
        if i == NTX - 1 {
            opts1 |= DESC_RING_END;
        }
        // SAFETY: `buf` is this slot's 2 KiB TX buffer; `p` its descriptor.
        unsafe {
            core::ptr::copy_nonoverlapping(frame.as_ptr(), buf as *mut u8, frame.len());
            write_volatile((p + 8) as *mut u64, self.tx_bufs_phys + (i * BUFSZ) as u64);
            write_volatile((p + 4) as *mut u32, 0);
            // DescOwn must be published last: the NIC may start reading the
            // moment it is set.
            fence(Ordering::Release);
            write_volatile(p as *mut u32, opts1);
        }
        fence(Ordering::Release);
        self.tx_cur = (i + 1) % NTX;
        // Realtek has no tail register — poke TxPoll to tell the NIC to look.
        w8(self.regs, self.map.tx_poll, TX_POLL_NPQ);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn the_8125_gets_its_own_register_offsets() {
        // The reason this is a map and not a comment: the 8168's transmit doorbell (0x38)
        // is the 8125's interrupt *mask*, and the 8168's mask (0x3c) is the 8125's
        // status. Driving an 8125 with 8168 offsets writes the doorbell into the mask —
        // a NIC that neither transmits nor stays quiet.
        let m = reg_map(0x8125);
        assert!(m.wide_intr);
        assert_eq!(m.intr_mask, 0x38);
        assert_eq!(m.intr_status, 0x3c);
        assert_eq!(m.tx_poll, 0x90);
        assert_ne!(m.tx_poll, TX_POLL, "the doorbell must not stay at the 8168 offset");
    }

    #[test_case]
    fn the_8168_family_keeps_the_classic_layout() {
        // 0x8168 is the single most common Ethernet controller in consumer PCs; nothing
        // about adding 8125 support may move its registers.
        for id in [0x8168u16, 0x8169, 0x8136, 0x8161, 0x8162, 0x8167] {
            let m = reg_map(id);
            assert!(!m.wide_intr, "{id:#06x} must not be treated as 8125-class");
            assert_eq!(m.intr_mask, 0x3c);
            assert_eq!(m.intr_status, 0x3e);
            assert_eq!(m.tx_poll, 0x38);
        }
    }

    #[test_case]
    fn every_id_the_dispatcher_sends_here_gets_a_map() {
        // `nic_ids` claims a fixed list for this driver; each one has to land in exactly
        // one of the two layouts, or a card would be driven with a default that suits
        // neither.
        for &id in crate::net::nic_ids::realtek_r8169_ids() {
            let m = reg_map(id);
            let is_8125 = RTL8125_IDS.contains(&id);
            assert_eq!(m.wide_intr, is_8125, "{id:#06x} classified inconsistently");
        }
    }
}
