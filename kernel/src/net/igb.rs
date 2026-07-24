//! **Intel igb / igc NIC over PCI** — 82575/82576/82580/I210/I211/I350 (igb) and
//! I225/I226 2.5GbE (igc). These are the Ethernet controllers on desktop
//! motherboards and servers; the I210/I211 in particular is extremely common on
//! enthusiast and small-server boards, and I225/I226 on anything 2.5GbE.
//!
//! This is a *separate driver* from [`super::e1000`] because 82575 moved almost
//! everything that matters:
//!
//! | | e1000/e1000e | igb/igc |
//! |---|---|---|
//! | RX ring base | `RDBAL 0x2800` | `RDBAL(0) 0xC000` |
//! | TX ring base | `TDBAL 0x3800` | `TDBAL(0) 0xE000` |
//! | interrupt regs | `ICR 0x00C0` | `ICR 0x1500`, plus extended `EIMC 0x1528` |
//! | descriptors | legacy 16-byte | **advanced** 16-byte (different field layout) |
//! | queue enable | implicit in `RCTL`/`TCTL` | explicit `RXDCTL/TXDCTL.ENABLE`, polled |
//!
//! Pointing the e1000 driver at one of these — which is what matching on vendor
//! `0x8086` alone did — writes the rings into reserved space at `0x2800`, so the
//! NIC comes up, reports a link, and never receives a single frame.
//!
//! **Descriptor choice.** Both families are driven with *advanced* descriptors
//! (RX: one-buffer, `SRRCTL.DESCTYPE = 001`; TX: advanced data descriptors,
//! `DTYP = 0x3`), matching what Linux `igb`/`igc` do. 82576 also supports the
//! legacy formats, but I225/I226 effectively does not, so using the advanced
//! layout for both keeps one code path across the whole range.
//!
//! Poll-driven, one RX and one TX queue — no interrupts, no RSS, no multi-queue.
//! Verified against QEMU's `-device igb` (82576) model; I210/I225 register layout
//! comes from the datasheets and Linux `e1000_regs.h`.

use crate::net::nic_ids::NicKind;
use crate::net::NetDevice;
use core::ptr::{read_volatile, write_volatile};
use core::sync::atomic::{fence, Ordering};

#[cfg(target_arch = "aarch64")]
use crate::pci::PciDevice;
#[cfg(target_arch = "x86_64")]
use crate::arch::x86_64::pci::PciDevice;

// --- general registers ---------------------------------------------------
const CTRL: usize = 0x0000;
const STATUS: usize = 0x0008;
const CTRL_EXT: usize = 0x0018;
const RCTL: usize = 0x0100;
const TCTL: usize = 0x0400;
const MTA: usize = 0x5200; // 128 * u32 multicast table
const RAL0: usize = 0x5400;
const RAH0: usize = 0x5404;

// --- interrupt registers (82575+ moved these out of the 0x00C0 block) ----
const ICR: usize = 0x1500;
const IMC: usize = 0x150c;
const EIMC: usize = 0x1528;

// --- per-queue ring registers (queue 0; stride 0x40 per queue) -----------
const RDBAL0: usize = 0xc000;
const RDBAH0: usize = 0xc004;
const RDLEN0: usize = 0xc008;
const SRRCTL0: usize = 0xc00c;
const RDH0: usize = 0xc010;
const RDT0: usize = 0xc018;
const RXDCTL0: usize = 0xc028;
const TDBAL0: usize = 0xe000;
const TDBAH0: usize = 0xe004;
const TDLEN0: usize = 0xe008;
const TDH0: usize = 0xe010;
const TDT0: usize = 0xe018;
const TXDCTL0: usize = 0xe028;

// CTRL bits
const CTRL_SLU: u32 = 1 << 6; // set link up
const CTRL_RST: u32 = 1 << 26; // device reset (self-clearing)
const CTRL_EXT_DRV_LOAD: u32 = 1 << 28;
// STATUS bits
const STATUS_LU: u32 = 1 << 1;
// RCTL bits
const RCTL_EN: u32 = 1 << 1;
const RCTL_BAM: u32 = 1 << 15; // broadcast accept
const RCTL_SECRC: u32 = 1 << 26; // strip Ethernet CRC
// TCTL bits
const TCTL_EN: u32 = 1 << 1;
const TCTL_PSP: u32 = 1 << 3; // pad short packets
// SRRCTL: buffer size in 1 KiB units, and the descriptor type field (27:25).
const SRRCTL_BSIZEPKT_2K: u32 = 2; // 2 KiB packet buffer
const SRRCTL_DESCTYPE_ADV_ONEBUF: u32 = 1 << 25;
const SRRCTL_DROP_EN: u32 = 1 << 31; // drop rather than stall when the ring is full
// RXDCTL/TXDCTL queue enable.
const XDCTL_ENABLE: u32 = 1 << 25;

// Advanced RX descriptor write-back: status/error dword at byte offset 8.
const RXD_STAT_DD: u32 = 1 << 0;
const RXD_STAT_EOP: u32 = 1 << 1;
// Advanced TX data descriptor: cmd_type_len fields.
const TXD_DTYP_DATA: u32 = 0x0030_0000;
const TXD_DCMD_EOP: u32 = 0x0100_0000;
const TXD_DCMD_IFCS: u32 = 0x0200_0000;
const TXD_DCMD_RS: u32 = 0x0800_0000;
const TXD_DCMD_DEXT: u32 = 0x2000_0000;
/// `olinfo_status` carries the payload length starting at bit 14.
const TXD_PAYLEN_SHIFT: u32 = 14;
const TXD_STAT_DD: u32 = 1 << 0;

const NRX: usize = 32;
const NTX: usize = 32;
const BUFSZ: usize = 2048;
const DESC_SZ: usize = 16;
const MMIO_SPAN: usize = 0x2_0000; // 128 KiB register block

/// A poll-driven Intel igb/igc NIC.
pub struct Igb {
    regs: u64, // HHDM-mapped BAR0
    mac: [u8; 6],
    rx_ring: u64,
    tx_ring: u64,
    rx_bufs: u64,
    rx_bufs_phys: u64,
    tx_bufs: u64,
    tx_bufs_phys: u64,
    rx_cur: usize,
    tx_cur: usize,
}

#[inline]
fn mmio_r(base: u64, off: usize) -> u32 {
    // SAFETY: `base` is the HHDM-mapped register block; `off` within MMIO_SPAN.
    unsafe { read_volatile((base + off as u64) as *const u32) }
}
#[inline]
fn mmio_w(base: u64, off: usize, v: u32) {
    // SAFETY: as `mmio_r`; 32-bit register write.
    unsafe { write_volatile((base + off as u64) as *mut u32, v) };
}

impl Igb {
    /// Bring up the located igb/igc NIC. `kind` is carried only for logging — the
    /// two families share this bring-up sequence. Returns `None` if BAR0 is
    /// unusable, DMA memory can't be allocated, or a queue refuses to enable.
    pub fn init(d: PciDevice, kind: NicKind) -> Option<Igb> {
        let name = kind.name();
        d.enable_bus_master();
        let bar0 = d.bar(0);
        if bar0 == 0 {
            crate::ktrace::log(name, "BAR0 is not a memory BAR -- cannot map registers");
            return None;
        }
        let regs = crate::mm::map_mmio(bar0, MMIO_SPAN);

        // Mask every interrupt source (both the legacy and the extended block)
        // and clear the cause register — this driver strictly polls.
        mmio_w(regs, IMC, 0xffff_ffff);
        mmio_w(regs, EIMC, 0xffff_ffff);
        let _ = mmio_r(regs, ICR);

        // Reset, then re-mask: firmware may have left the rings live and pointed
        // at memory we are about to reuse.
        mmio_w(regs, CTRL, mmio_r(regs, CTRL) | CTRL_RST);
        for _ in 0..1_000_000 {
            if mmio_r(regs, CTRL) & CTRL_RST == 0 {
                break;
            }
            core::hint::spin_loop();
        }
        mmio_w(regs, IMC, 0xffff_ffff);
        mmio_w(regs, EIMC, 0xffff_ffff);
        let _ = mmio_r(regs, ICR);
        // `CTRL.RST` preserves PCI config space on paper, but firmware and real
        // parts vary; re-asserting bus mastering costs one config write and
        // removes a class of "rings programmed, no DMA ever happens" bug.
        d.enable_bus_master();
        // Tell the manageability firmware a driver owns the part.
        mmio_w(regs, CTRL_EXT, mmio_r(regs, CTRL_EXT) | CTRL_EXT_DRV_LOAD);

        // Zero the multicast table filter.
        for i in 0..128 {
            mmio_w(regs, MTA + i * 4, 0);
        }

        let mac = read_mac(regs, name);

        // Descriptor rings + buffer pools (physically contiguous DMA memory).
        let (rx_ring_phys, rx_ring) = crate::mm::alloc_dma(NRX * DESC_SZ)?;
        let (tx_ring_phys, tx_ring) = crate::mm::alloc_dma(NTX * DESC_SZ)?;
        let (rx_bufs_phys, rx_bufs) = crate::mm::alloc_dma(NRX * BUFSZ)?;
        let (tx_bufs_phys, tx_bufs) = crate::mm::alloc_dma(NTX * BUFSZ)?;

        // Advanced RX descriptors, read format: packet buffer address then header
        // buffer address (0 in one-buffer mode).
        for i in 0..NRX {
            let p = rx_ring + (i * DESC_SZ) as u64;
            // SAFETY: `p` is within the RX ring DMA region; 16-byte descriptor.
            unsafe {
                write_volatile(p as *mut u64, rx_bufs_phys + (i * BUFSZ) as u64);
                write_volatile((p + 8) as *mut u64, 0);
            }
        }
        // TX descriptors start idle with DD set so the ring reads as all-free.
        for i in 0..NTX {
            let p = tx_ring + (i * DESC_SZ) as u64;
            // SAFETY: within the TX ring DMA region.
            unsafe {
                write_volatile(p as *mut u64, 0);
                write_volatile((p + 8) as *mut u32, 0);
                write_volatile((p + 12) as *mut u32, TXD_STAT_DD);
            }
        }

        // --- RX queue 0 ---
        mmio_w(regs, RDBAL0, rx_ring_phys as u32);
        mmio_w(regs, RDBAH0, (rx_ring_phys >> 32) as u32);
        mmio_w(regs, RDLEN0, (NRX * DESC_SZ) as u32);
        mmio_w(regs, SRRCTL0, SRRCTL_BSIZEPKT_2K | SRRCTL_DESCTYPE_ADV_ONEBUF | SRRCTL_DROP_EN);
        mmio_w(regs, RDH0, 0);
        mmio_w(regs, RDT0, (NRX - 1) as u32);
        mmio_w(regs, RXDCTL0, mmio_r(regs, RXDCTL0) | XDCTL_ENABLE);
        if !wait_enabled(regs, RXDCTL0) {
            crate::ktrace::log(name, "RX queue 0 never reported enabled (RXDCTL.ENABLE stuck clear)");
            return None;
        }
        mmio_w(regs, RCTL, RCTL_EN | RCTL_BAM | RCTL_SECRC);

        // --- TX queue 0 ---
        mmio_w(regs, TDBAL0, tx_ring_phys as u32);
        mmio_w(regs, TDBAH0, (tx_ring_phys >> 32) as u32);
        mmio_w(regs, TDLEN0, (NTX * DESC_SZ) as u32);
        mmio_w(regs, TDH0, 0);
        mmio_w(regs, TDT0, 0);
        mmio_w(regs, TXDCTL0, mmio_r(regs, TXDCTL0) | XDCTL_ENABLE);
        if !wait_enabled(regs, TXDCTL0) {
            crate::ktrace::log(name, "TX queue 0 never reported enabled (TXDCTL.ENABLE stuck clear)");
            return None;
        }
        mmio_w(regs, TCTL, TCTL_EN | TCTL_PSP);

        // Link up. Auto-negotiation takes a moment; give it a bounded chance.
        mmio_w(regs, CTRL, mmio_r(regs, CTRL) | CTRL_SLU);
        let mut link = false;
        for _ in 0..200_000 {
            if mmio_r(regs, STATUS) & STATUS_LU != 0 {
                link = true;
                break;
            }
            core::hint::spin_loop();
        }
        crate::ktrace::log_fmt(format_args!(
            "{name}: up (vendor {:04x} dev {:04x}), MAC {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}, link={link}",
            d.vendor, d.device, mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]
        ));

        Some(Igb {
            regs,
            mac,
            rx_ring,
            tx_ring,
            rx_bufs,
            rx_bufs_phys,
            tx_bufs,
            tx_bufs_phys,
            rx_cur: 0,
            tx_cur: 0,
        })
    }
}

/// Poll a queue's `*XDCTL.ENABLE` until the hardware acknowledges it. The
/// datasheet requires this handshake before touching the ring; a queue that never
/// enables means the ring programming was rejected, and continuing would silently
/// never transfer.
fn wait_enabled(regs: u64, off: usize) -> bool {
    for _ in 0..1_000_000 {
        if mmio_r(regs, off) & XDCTL_ENABLE != 0 {
            return true;
        }
        core::hint::spin_loop();
    }
    false
}

/// Read the MAC from the Receive Address registers, which firmware pre-loads from
/// the NVM. igb/igc NVM access (EERD / the I210 flashless iNVM) is not
/// implemented; an empty RAL/RAH is reported rather than papered over.
fn read_mac(regs: u64, name: &str) -> [u8; 6] {
    let ral = mmio_r(regs, RAL0);
    let rah = mmio_r(regs, RAH0);
    if ral == 0 && (rah & 0xffff) == 0 {
        crate::ktrace::log(name, "RAL/RAH empty -- MAC unknown (NVM read not implemented)");
        return [0; 6];
    }
    [ral as u8, (ral >> 8) as u8, (ral >> 16) as u8, (ral >> 24) as u8, rah as u8, (rah >> 8) as u8]
}

impl NetDevice for Igb {
    fn mac(&self) -> [u8; 6] {
        self.mac
    }

    fn receive(&mut self, out: &mut [u8]) -> Option<usize> {
        let p = self.rx_ring + (self.rx_cur * DESC_SZ) as u64;
        // Advanced RX write-back: status_error at +8, length at +12.
        // SAFETY: `p` addresses the current RX descriptor in DMA memory.
        let (status, len) = unsafe {
            let status = read_volatile((p + 8) as *const u32);
            let len = read_volatile((p + 12) as *const u16);
            (status, len)
        };
        if status & RXD_STAT_DD == 0 {
            return None; // hardware hasn't filled this descriptor yet
        }
        fence(Ordering::Acquire);
        let n = if status & RXD_STAT_EOP != 0 { (len as usize).min(out.len()) } else { 0 };
        if n > 0 {
            let src = self.rx_bufs + (self.rx_cur * BUFSZ) as u64;
            // SAFETY: `src` is this descriptor's 2 KiB buffer; `n <= len <= BUFSZ`.
            unsafe { core::ptr::copy_nonoverlapping(src as *const u8, out.as_mut_ptr(), n) };
        }
        // Hand the descriptor back: the write-back clobbered the read-format
        // addresses, so both must be rewritten (unlike the legacy descriptor,
        // where buffer_addr survives).
        // SAFETY: resetting the descriptor we just consumed.
        unsafe {
            write_volatile(p as *mut u64, self.rx_bufs_phys + (self.rx_cur * BUFSZ) as u64);
            write_volatile((p + 8) as *mut u64, 0);
        }
        fence(Ordering::Release);
        mmio_w(self.regs, RDT0, self.rx_cur as u32);
        self.rx_cur = (self.rx_cur + 1) % NRX;
        Some(n)
    }

    fn transmit(&mut self, frame: &[u8]) {
        if frame.len() > BUFSZ {
            return;
        }
        let i = self.tx_cur;
        let p = self.tx_ring + (i * DESC_SZ) as u64;
        let buf = self.tx_bufs + (i * BUFSZ) as u64;
        let len = frame.len() as u32;
        // SAFETY: `buf` is this slot's 2 KiB TX buffer; `p` its descriptor.
        unsafe {
            core::ptr::copy_nonoverlapping(frame.as_ptr(), buf as *mut u8, frame.len());
            write_volatile(p as *mut u64, self.tx_bufs_phys + (i * BUFSZ) as u64);
            write_volatile(
                (p + 8) as *mut u32,
                len | TXD_DTYP_DATA | TXD_DCMD_EOP | TXD_DCMD_IFCS | TXD_DCMD_RS | TXD_DCMD_DEXT,
            );
            // olinfo_status: payload length at bit 14, status (DD) cleared.
            write_volatile((p + 12) as *mut u32, len << TXD_PAYLEN_SHIFT);
        }
        fence(Ordering::Release);
        self.tx_cur = (i + 1) % NTX;
        mmio_w(self.regs, TDT0, self.tx_cur as u32);
    }
}
