//! **Intel gigabit NIC over PCI** — two families behind one driver:
//!
//! * [`NicKind::E1000`] — 82540/82544/82545/82546 legacy PCI (QEMU `-device
//!   e1000`, VirtualBox's default adapter, ~2003-era cards).
//! * [`NicKind::E1000e`] — 82571 through **I217/I218/I219** PCIe (QEMU `-device
//!   e1000e`, and the Ethernet controller in most business laptops).
//!
//! Both use the same 16-byte **legacy** descriptors and the same ring registers
//! (`RDBAL 0x2800` / `TDBAL 0x3800`), which is why pointing the legacy init at an
//! I219 mostly worked and hid the family distinction. The bring-up sequences do
//! differ, and this driver now selects between them ([`Family`]):
//!
//! | | legacy e1000 | e1000e |
//! |---|---|---|
//! | device reset | optional | **required** — firmware may leave the ring registers live |
//! | MAC fallback | EEPROM via `EERD` | RAL/RAH only; ICH8+ has no `EERD` (NVM is behind the FLASH BAR) |
//! | `CTRL_EXT.DRV_LOAD` | n/a | **set** — tells the ME/firmware the driver owns the PHY |
//! | `TIPG` IPGT | 10 | 8 (PCIe copper) |
//! | `TCTL.COLD` | 0x40 | 0x3F (full duplex) |
//!
//! Poll-driven (no interrupts): legacy RX/TX descriptor rings in DMA memory and a
//! per-descriptor buffer pool. Dual-arch: the PCI config surface is
//! `crate::arch::x86_64::pci` (I/O ports) on x86 and `crate::pci` (ECAM) on
//! aarch64 — identical `PciDevice` API.
//!
//! **Known limit on I217/I219:** Linux's `e1000e` carries a long tail of
//! PHY-specific errata workarounds for these parts (ULP disable, K1/LPT
//! workarounds, SMBus/ME handoff). None of that is here. Basic ring operation
//! after a clean firmware handoff is expected to work; a link that never comes up
//! on a specific PCH generation is most likely one of those workarounds missing,
//! and the boot log records the device ID and link state to make that diagnosable.

use crate::net::nic_ids::NicKind;
use crate::net::NetDevice;
use core::ptr::{read_volatile, write_volatile};
use core::sync::atomic::{fence, Ordering};

#[cfg(target_arch = "aarch64")]
use crate::pci::PciDevice;
#[cfg(target_arch = "x86_64")]
use crate::arch::x86_64::pci::PciDevice;

// --- register offsets (bytes into BAR0) ---------------------------------
const CTRL: usize = 0x0000;
const STATUS: usize = 0x0008;
const EERD: usize = 0x0014;
const CTRL_EXT: usize = 0x0018;
const ICR: usize = 0x00c0;
const IMC: usize = 0x00d8;
const RCTL: usize = 0x0100;
const TCTL: usize = 0x0400;
const TIPG: usize = 0x0410;
const RDBAL: usize = 0x2800;
const RDBAH: usize = 0x2804;
const RDLEN: usize = 0x2808;
const RDH: usize = 0x2810;
const RDT: usize = 0x2818;
const TDBAL: usize = 0x3800;
const TDBAH: usize = 0x3804;
const TDLEN: usize = 0x3808;
const TDH: usize = 0x3810;
const TDT: usize = 0x3818;
const MTA: usize = 0x5200; // 128 * u32 multicast table
const RAL0: usize = 0x5400;
const RAH0: usize = 0x5404;

// CTRL bits
const CTRL_SLU: u32 = 1 << 6; // set link up
const CTRL_ASDE: u32 = 1 << 5; // auto-speed detect enable
const CTRL_RST: u32 = 1 << 26; // device reset (self-clearing)
// CTRL_EXT bits
const CTRL_EXT_DRV_LOAD: u32 = 1 << 28; // driver loaded; firmware/ME hands off the PHY
// STATUS bits
const STATUS_LU: u32 = 1 << 1; // link up
// RCTL bits
const RCTL_EN: u32 = 1 << 1;
const RCTL_BAM: u32 = 1 << 15; // broadcast accept
const RCTL_SECRC: u32 = 1 << 26; // strip Ethernet CRC
const RCTL_BSIZE_2048: u32 = 0; // 2048-byte buffers (BSIZE=00, no BSEX)
// TCTL bits
const TCTL_EN: u32 = 1 << 1;
const TCTL_PSP: u32 = 1 << 3; // pad short packets
// RX status
const RXD_STAT_DD: u8 = 1 << 0; // descriptor done
const RXD_STAT_EOP: u8 = 1 << 1; // end of packet
// TX command / status
const TXD_CMD_EOP: u8 = 1 << 0;
const TXD_CMD_IFCS: u8 = 1 << 1; // insert FCS
const TXD_CMD_RS: u8 = 1 << 3; // report status
const TXD_STAT_DD: u8 = 1 << 0;

const NRX: usize = 32;
const NTX: usize = 32;
const BUFSZ: usize = 2048;
const DESC_SZ: usize = 16;
const MMIO_SPAN: usize = 0x2_0000; // 128 KiB register block

/// Which Intel gigabit family this instance is driving — see the module table.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Family {
    /// 82540/82544/82545/82546 legacy PCI.
    Legacy,
    /// 82571 … I219 PCIe.
    Pcie,
}

/// A poll-driven Intel gigabit NIC (legacy e1000 or e1000e).
pub struct E1000 {
    regs: u64, // HHDM-mapped BAR0
    mac: [u8; 6],
    rx_ring: u64, // virt of RX descriptor ring
    tx_ring: u64, // virt of TX descriptor ring
    rx_bufs: u64, // virt of RX buffer pool (phys handed to descriptors)
    tx_bufs: u64,
    tx_bufs_phys: u64,
    rx_cur: usize,
    tx_cur: usize,
}

#[inline]
fn mmio_r(base: u64, off: usize) -> u32 {
    // SAFETY: `base` is the HHDM-mapped e1000 register block; `off` in range.
    unsafe { read_volatile((base + off as u64) as *const u32) }
}
#[inline]
fn mmio_w(base: u64, off: usize, v: u32) {
    // SAFETY: as `mmio_r`; 32-bit register write.
    unsafe { write_volatile((base + off as u64) as *mut u32, v) };
}

impl E1000 {
    /// Bring up the located Intel NIC. `kind` selects the family-specific parts
    /// of the sequence (see the module table). Returns `None` if BAR0 is unusable
    /// or DMA memory can't be allocated.
    pub fn init(d: PciDevice, kind: NicKind) -> Option<E1000> {
        let family = match kind {
            NicKind::E1000 => Family::Legacy,
            // Anything dispatched here that isn't the legacy family is PCIe —
            // `net::pci` only routes E1000/E1000e to this driver.
            _ => Family::Pcie,
        };
        d.enable_bus_master();
        let bar0 = d.bar(0);
        if bar0 == 0 {
            return None;
        }
        let regs = crate::mm::map_mmio(bar0, MMIO_SPAN);

        // Mask all interrupts and clear any pending — we strictly poll.
        mmio_w(regs, IMC, 0xffff_ffff);
        let _ = mmio_r(regs, ICR);

        // Reset the device before touching the rings. On a real machine the
        // firmware (or a previous OS on a warm reboot) may have left the
        // descriptor rings enabled and pointed at memory we now own; programming
        // RDBAL/TDBAL under a live RCTL.EN would have the NIC DMA into whatever
        // the old ring addresses were. QEMU hands over a quiescent device, which
        // is why this was never needed before.
        if family == Family::Pcie {
            reset(regs);
            // `CTRL.RST` preserves PCI config space on paper, but firmware and
            // real parts vary; re-asserting bus mastering costs one config write
            // and removes a class of "rings programmed, no DMA ever happens" bug.
            d.enable_bus_master();
            // Tell the manageability firmware / ME that a driver owns the part.
            // Without this, ICH-based PHYs (I217/I219) can stay under firmware
            // control and never link up.
            mmio_w(regs, CTRL_EXT, mmio_r(regs, CTRL_EXT) | CTRL_EXT_DRV_LOAD);
        }

        // Zero the multicast table filter.
        for i in 0..128 {
            mmio_w(regs, MTA + i * 4, 0);
        }

        let mac = read_mac(regs, family);

        // Descriptor rings + buffer pools (physically contiguous DMA memory).
        let (rx_ring_phys, rx_ring) = crate::mm::alloc_dma(NRX * DESC_SZ)?;
        let (tx_ring_phys, tx_ring) = crate::mm::alloc_dma(NTX * DESC_SZ)?;
        let (rx_bufs_phys, rx_bufs) = crate::mm::alloc_dma(NRX * BUFSZ)?;
        let (tx_bufs_phys, tx_bufs) = crate::mm::alloc_dma(NTX * BUFSZ)?;

        // Populate RX descriptors: buffer_addr + zeroed status.
        for i in 0..NRX {
            let d = rx_ring + (i * DESC_SZ) as u64;
            // SAFETY: `d` is within the RX ring DMA region; 16-byte descriptor.
            unsafe {
                write_volatile(d as *mut u64, rx_bufs_phys + (i * BUFSZ) as u64);
                write_volatile((d + 8) as *mut u64, 0);
            }
        }
        for i in 0..NTX {
            let d = tx_ring + (i * DESC_SZ) as u64;
            // SAFETY: within the TX ring DMA region.
            unsafe {
                write_volatile(d as *mut u64, tx_bufs_phys + (i * BUFSZ) as u64);
                // status DD=1 so the ring reads as "all free" initially.
                write_volatile((d + 8) as *mut u64, (TXD_STAT_DD as u64) << 32);
            }
        }

        // Program the RX ring.
        mmio_w(regs, RDBAL, rx_ring_phys as u32);
        mmio_w(regs, RDBAH, (rx_ring_phys >> 32) as u32);
        mmio_w(regs, RDLEN, (NRX * DESC_SZ) as u32);
        mmio_w(regs, RDH, 0);
        mmio_w(regs, RDT, (NRX - 1) as u32);
        mmio_w(regs, RCTL, RCTL_EN | RCTL_BAM | RCTL_SECRC | RCTL_BSIZE_2048);

        // Program the TX ring.
        mmio_w(regs, TDBAL, tx_ring_phys as u32);
        mmio_w(regs, TDBAH, (tx_ring_phys >> 32) as u32);
        mmio_w(regs, TDLEN, (NTX * DESC_SZ) as u32);
        mmio_w(regs, TDH, 0);
        mmio_w(regs, TDT, 0);
        // CT=0x10 (collision threshold), COLD = full-duplex collision distance
        // (0x40 half / 0x3F full — the PCIe parts are always full duplex).
        let cold: u32 = if family == Family::Pcie { 0x3f } else { 0x40 };
        mmio_w(regs, TCTL, TCTL_EN | TCTL_PSP | (0x10 << 4) | (cold << 12));
        // IPGT/IPGR1/IPGR2. IPGT is 10 for legacy copper, 8 for PCIe copper.
        let ipgt: u32 = if family == Family::Pcie { 8 } else { 10 };
        mmio_w(regs, TIPG, 0x0060_2000 | ipgt);

        // Link up + auto-speed.
        let ctrl = mmio_r(regs, CTRL);
        mmio_w(regs, CTRL, ctrl | CTRL_SLU | CTRL_ASDE);

        // Report the link state. Auto-negotiation takes a moment, so a `false`
        // here is not necessarily a failure — but on a real machine it is the
        // first thing to look at, so give it a bounded chance to come up and log
        // what we ended with.
        let mut link = false;
        for _ in 0..200_000 {
            if mmio_r(regs, STATUS) & STATUS_LU != 0 {
                link = true;
                break;
            }
            core::hint::spin_loop();
        }
        crate::ktrace::log_fmt(format_args!(
            "{}: up (vendor {:04x} dev {:04x}), MAC {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}, link={}",
            if family == Family::Pcie { "e1000e" } else { "e1000" },
            d.vendor,
            d.device,
            mac[0],
            mac[1],
            mac[2],
            mac[3],
            mac[4],
            mac[5],
            link
        ));
        if !link {
            crate::ktrace::log("e1000", "link down after bring-up -- check the cable; on I217/I219 this can also be a missing PHY errata workaround (see the module docs)");
        }

        Some(E1000 { regs, mac, rx_ring, tx_ring, rx_bufs, tx_bufs, tx_bufs_phys, rx_cur: 0, tx_cur: 0 })
    }
}

/// Issue a device reset and wait for it to complete. `CTRL.RST` is self-clearing;
/// the datasheet allows up to 1 ms plus a PCI-config reload, so the spin is
/// generous but bounded — a NIC that never clears RST is left alone rather than
/// hanging the boot.
fn reset(regs: u64) {
    mmio_w(regs, CTRL, mmio_r(regs, CTRL) | CTRL_RST);
    for _ in 0..1_000_000 {
        if mmio_r(regs, CTRL) & CTRL_RST == 0 {
            break;
        }
        core::hint::spin_loop();
    }
    // The reset re-enables interrupts and clears the cause register; re-mask.
    mmio_w(regs, IMC, 0xffff_ffff);
    let _ = mmio_r(regs, ICR);
}

/// Read the MAC: prefer the Receive Address registers, which both firmware and
/// QEMU pre-load from the NVM.
///
/// The `EERD` EEPROM fallback is **legacy-only**. On ICH8-and-later parts
/// (82577/82579/I217/I218/I219) there is no EEPROM behind `EERD` at all — the NVM
/// lives in the platform SPI flash reached through a second BAR and a shadow-RAM
/// protocol. Polling `EERD.DONE` there spins the full bound and yields zeros, so
/// for the PCIe family we report the failure instead of pretending.
fn read_mac(regs: u64, family: Family) -> [u8; 6] {
    let ral = mmio_r(regs, RAL0);
    let rah = mmio_r(regs, RAH0);
    if ral != 0 || (rah & 0xffff) != 0 {
        return [ral as u8, (ral >> 8) as u8, (ral >> 16) as u8, (ral >> 24) as u8, rah as u8, (rah >> 8) as u8];
    }
    if family == Family::Pcie {
        crate::ktrace::log("e1000e", "RAL/RAH empty and EERD is not present on this family -- MAC unknown (NVM is behind the FLASH BAR; not implemented)");
        return [0; 6];
    }
    // EEPROM: three 16-bit words at addresses 0,1,2 via the EERD register.
    let mut mac = [0u8; 6];
    for word in 0u32..3 {
        mmio_w(regs, EERD, (word << 8) | 1); // START | addr
        // Poll for DONE (bit 4).
        let mut data = 0u32;
        for _ in 0..100_000 {
            let v = mmio_r(regs, EERD);
            if v & (1 << 4) != 0 {
                data = v >> 16;
                break;
            }
            core::hint::spin_loop();
        }
        mac[(word * 2) as usize] = data as u8;
        mac[(word * 2 + 1) as usize] = (data >> 8) as u8;
    }
    mac
}

impl NetDevice for E1000 {
    fn mac(&self) -> [u8; 6] {
        self.mac
    }

    fn receive(&mut self, out: &mut [u8]) -> Option<usize> {
        let d = self.rx_ring + (self.rx_cur * DESC_SZ) as u64;
        // SAFETY: `d` addresses the current RX descriptor in DMA memory.
        let (len, status) = unsafe {
            let len = read_volatile((d + 8) as *const u16);
            let status = read_volatile((d + 12) as *const u8);
            (len, status)
        };
        if status & RXD_STAT_DD == 0 {
            return None; // hardware hasn't filled this descriptor yet
        }
        fence(Ordering::Acquire);
        let n = if status & RXD_STAT_EOP != 0 { (len as usize).min(out.len()) } else { 0 };
        if n > 0 {
            let src = self.rx_bufs + (self.rx_cur * BUFSZ) as u64;
            // SAFETY: `src` is this descriptor's 2 KiB buffer; `n <= len`.
            unsafe { core::ptr::copy_nonoverlapping(src as *const u8, out.as_mut_ptr(), n) };
        }
        // Hand the descriptor back to hardware: clear length/status (resets DD)
        // while leaving the buffer_addr in place, then advance the tail.
        // SAFETY: resetting the descriptor we just consumed.
        unsafe {
            write_volatile((d + 8) as *mut u64, 0);
        }
        mmio_w(self.regs, RDT, self.rx_cur as u32);
        self.rx_cur = (self.rx_cur + 1) % NRX;
        Some(n)
    }

    fn transmit(&mut self, frame: &[u8]) {
        if frame.len() > BUFSZ {
            return;
        }
        let i = self.tx_cur;
        let d = self.tx_ring + (i * DESC_SZ) as u64;
        let buf = self.tx_bufs + (i * BUFSZ) as u64;
        // SAFETY: `buf` is this slot's 2 KiB TX buffer; `d` its descriptor.
        unsafe {
            core::ptr::copy_nonoverlapping(frame.as_ptr(), buf as *mut u8, frame.len());
            write_volatile(d as *mut u64, self.tx_bufs_phys + (i * BUFSZ) as u64);
            // length | cmd(EOP|IFCS|RS) in byte 11, status byte 12 cleared.
            let lower = frame.len() as u32 & 0xffff;
            let cmd = (TXD_CMD_EOP | TXD_CMD_IFCS | TXD_CMD_RS) as u32;
            write_volatile((d + 8) as *mut u32, lower | (cmd << 24));
            write_volatile((d + 12) as *mut u32, 0);
        }
        fence(Ordering::Release);
        self.tx_cur = (i + 1) % NTX;
        mmio_w(self.regs, TDT, self.tx_cur as u32);
    }
}
