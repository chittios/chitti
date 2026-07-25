//! **AHCI (SATA) block driver core** — arch-neutral, the storage controller
//! VirtualBox defaults to and most SATA hardware exposes. Polled, one command
//! slot per port; 48-bit LBA READ/WRITE DMA EXT; presented behind the shared
//! [`BlockDevice`] API.
//!
//! **Every populated port is a disk.** A real SATA HBA has up to 32 ports and
//! they are not densely packed — a drive on port 3 with ports 0-2 empty is
//! completely normal, and desktops routinely have two or three drives on one
//! controller. So enumeration is two-level: [`Ahci::present_count`] reports how
//! many ports are implemented *and* populated (pure register reads, no
//! allocation), and [`Ahci::bringup_nth`] brings up one of them. Callers map a
//! global disk index onto (controller, port) using the count, so skipping past a
//! disk costs nothing — bringing a port up allocates DMA, and a
//! bring-up-then-discard loop would leak it on every enumeration.
//!
//! This module is the *logic*; device discovery is per-arch (each arch finds
//! the AHCI functions on its PCI bus, maps ABAR/BAR5, and hands the mapped MMIO
//! base + a [`DmaAlloc`] to the bring-up). x86 (`arch::x86_64::ahci`) and
//! aarch64 (`arch::aarch64::ahci`) are thin wrappers over this one core.
//!
//! Still one command slot (slot 0) per port and strictly polled: I/O on a port is
//! serialized. That bounds throughput, not correctness.

use crate::block::{BlockDevice, BlockError, Dma, DmaAlloc, BLOCK_SIZE};
use core::ptr::{read_volatile, write_volatile};
use core::sync::atomic::{fence, Ordering};

// HBA global registers.
const HBA_GHC: u64 = 0x04;
const HBA_PI: u64 = 0x0c;
const GHC_AE: u32 = 1 << 31; // AHCI enable

// Port registers (port base = ABAR + 0x100 + port*0x80).
const P_CLB: u64 = 0x00; // command list base (u64)
const P_FB: u64 = 0x08; // FIS base (u64)
const P_IS: u64 = 0x10; // interrupt status
const P_CMD: u64 = 0x18; // command + status
const P_TFD: u64 = 0x20; // task file data
const P_SSTS: u64 = 0x28; // SATA status
const P_CI: u64 = 0x38; // command issue

const CMD_ST: u32 = 1 << 0; // start
const CMD_FRE: u32 = 1 << 4; // FIS receive enable
const CMD_FR: u32 = 1 << 14; // FIS receive running
const CMD_CR: u32 = 1 << 15; // command list running
const TFD_BSY: u32 = 1 << 7;
const TFD_DRQ: u32 = 1 << 3;

const DATA_MAX: usize = 64 * 1024;

/// MMIO span the ABAR register block reaches (generic HBA regs + 32 ports x
/// 0x80): the arch wrapper maps at least this much.
pub const MMIO_SPAN: usize = 0x1100;

unsafe fn r32(a: u64) -> u32 {
    unsafe { read_volatile(a as *const u32) }
}
unsafe fn w32(a: u64, v: u32) {
    unsafe { write_volatile(a as *mut u32, v) };
}
unsafe fn w8(a: u64, v: u8) {
    unsafe { write_volatile(a as *mut u8, v) };
}

/// A polled AHCI port with a single command slot. Buffers are [`Dma`] pairs:
/// the CPU fills the command list/table via `.virt`, the HBA reads them by
/// `.phys`.
pub struct Ahci {
    port: u64,     // port register base (virtual)
    clb: Dma,      // command list
    ctba: Dma,     // command table
    data_buf: Dma, // 64 KiB bounce buffer
    capacity: u64,
}

impl Ahci {
    /// Bring up the first present SATA port on the HBA at the already-mapped
    /// ABAR `abar`, using `alloc` for DMA memory. `None` if no device is
    /// attached or IDENTIFY fails.
    ///
    /// # Safety
    /// `abar` must be a valid, mapped AHCI ABAR (at least [`MMIO_SPAN`] bytes),
    /// and `alloc` must return real physically-contiguous DMA memory.
    pub unsafe fn bringup(abar: u64, alloc: DmaAlloc) -> Option<Ahci> {
        // SAFETY: forwarded to `bringup_nth` under the caller's contract.
        unsafe { Self::bringup_nth(abar, alloc, 0) }
    }

    /// How many ports on this HBA are both implemented (`PI`) and have a device
    /// attached (`SSTS.DET == 3`) — i.e. how many disks this controller offers.
    ///
    /// Pure register reads, no allocation and no state change beyond the
    /// idempotent `GHC.AE` enable, so a caller can cheaply work out which
    /// controller/port a global disk index lands on before committing to a
    /// bring-up.
    ///
    /// # Safety
    /// `abar` must be a valid, mapped AHCI ABAR (at least [`MMIO_SPAN`] bytes).
    pub unsafe fn present_count(abar: u64) -> usize {
        unsafe {
            w32(abar + HBA_GHC, r32(abar + HBA_GHC) | GHC_AE);
            let pi = r32(abar + HBA_PI);
            (0..32u32)
                .filter(|p| pi & (1 << p) != 0)
                .filter(|p| r32(abar + 0x100 + *p as u64 * 0x80 + P_SSTS) & 0xf == 3)
                .count()
        }
    }

    /// Bring up the **`nth`** port on this HBA that is both implemented (`PI`) and
    /// has a device attached (`SSTS.DET == 3`), counting only such ports.
    /// `None` once `nth` runs past the last one, which is what lets a caller
    /// enumerate every disk on the controller by counting up from 0.
    ///
    /// A desktop SATA HBA routinely has several ports populated, and they are
    /// **not** contiguous — a drive on port 3 with ports 0-2 empty is normal.
    /// Only bringing up the first present port made every disk but one invisible.
    ///
    /// Ports are independent register blocks, so bringing up port 3 does not
    /// disturb an already-running port 0: `GHC.AE` is idempotent and everything
    /// else written here is inside the selected port's own 0x80-byte window.
    ///
    /// # Safety
    /// As [`Ahci::bringup`].
    pub unsafe fn bringup_nth(abar: u64, alloc: DmaAlloc, nth: usize) -> Option<Ahci> {
        unsafe {
            w32(abar + HBA_GHC, r32(abar + HBA_GHC) | GHC_AE);
            let pi = r32(abar + HBA_PI);
            // Walk implemented ports, counting the ones with a device present
            // (SSTS DET=3) until we reach `nth`.
            let mut port = 0u64;
            let mut seen = 0usize;
            for p in 0..32u32 {
                if pi & (1 << p) == 0 {
                    continue;
                }
                let base = abar + 0x100 + p as u64 * 0x80;
                if r32(base + P_SSTS) & 0xf != 3 {
                    continue;
                }
                if seen == nth {
                    port = base;
                    break;
                }
                seen += 1;
            }
            if port == 0 {
                return None;
            }

            // Stop the port before reprogramming CLB/FB.
            let mut cmd = r32(port + P_CMD);
            w32(port + P_CMD, cmd & !(CMD_ST | CMD_FRE));
            let mut g = 0;
            while r32(port + P_CMD) & (CMD_CR | CMD_FR) != 0 && g < 1_000_000 {
                g += 1;
            }

            // Command list (1 KiB, 32 headers), received-FIS area (256 B), and one
            // command table (CFIS + PRDT).
            let clb = alloc(1024)?;
            let fb = alloc(256)?;
            let ctba = alloc(4096)?;
            w32(port + P_CLB, clb.phys as u32);
            w32(port + P_CLB + 4, (clb.phys >> 32) as u32);
            w32(port + P_FB, fb.phys as u32);
            w32(port + P_FB + 4, (fb.phys >> 32) as u32);
            // Slot 0 header -> the command table.
            write_volatile((clb.virt + 8) as *mut u32, ctba.phys as u32); // CTBA
            write_volatile((clb.virt + 12) as *mut u32, (ctba.phys >> 32) as u32);

            // Clear errors, start (FRE then ST).
            w32(port + P_IS, 0xffff_ffff);
            cmd = r32(port + P_CMD);
            w32(port + P_CMD, cmd | CMD_FRE);
            w32(port + P_CMD, r32(port + P_CMD) | CMD_ST);

            let mut a = Ahci { port, clb, ctba, data_buf: alloc(DATA_MAX)?, capacity: 0 };

            // IDENTIFY DEVICE (0xEC) into the bounce buffer.
            if !a.issue(0xEC, 0, 0, 512, false) {
                crate::ktrace::log("ahci", "IDENTIFY failed");
                return None;
            }
            // 48-bit LBA sector count at words 100..103 (byte offset 200).
            let buf = a.data_buf.virt;
            let lba48 = (read_volatile((buf + 200) as *const u16) as u64)
                | ((read_volatile((buf + 202) as *const u16) as u64) << 16)
                | ((read_volatile((buf + 204) as *const u16) as u64) << 32);
            let lba28 = read_volatile((buf + 120) as *const u32) as u64; // words 60..61
            a.capacity = if lba48 != 0 { lba48 } else { lba28 };
            crate::ktrace::log_fmt(format_args!("ahci: SATA disk up, {} sectors ({} MiB)", a.capacity, a.capacity * 512 / (1024 * 1024)));
            Some(a)
        }
    }

    /// Issue one command on slot 0 and poll for completion. `cmd`: ATA command
    /// (0xEC IDENTIFY, 0x25 READ DMA EXT, 0x35 WRITE DMA EXT). Data goes through
    /// the bounce buffer; `bytes` is the transfer size; `write` sets direction.
    unsafe fn issue(&mut self, cmd: u8, lba: u64, count: u16, bytes: u32, write: bool) -> bool {
        unsafe {
            // Command table: CFIS @0 (H2D register FIS), PRDT @0x80.
            let ct = self.ctba.virt;
            core::ptr::write_bytes(ct as *mut u8, 0, 0x80 + 16);
            // H2D register FIS.
            w8(ct, 0x27); // FIS type
            w8(ct + 1, 0x80); // C=1 (command)
            w8(ct + 2, cmd);
            w8(ct + 4, lba as u8);
            w8(ct + 5, (lba >> 8) as u8);
            w8(ct + 6, (lba >> 16) as u8);
            w8(ct + 7, 0x40); // device: LBA mode
            w8(ct + 8, (lba >> 24) as u8);
            w8(ct + 9, (lba >> 32) as u8);
            w8(ct + 10, (lba >> 40) as u8);
            w8(ct + 12, count as u8);
            w8(ct + 13, (count >> 8) as u8);
            // PRDT entry 0 -> bounce buffer.
            let prdt = ct + 0x80;
            let db = self.data_buf.phys;
            w32(prdt, db as u32);
            w32(prdt + 4, (db >> 32) as u32);
            w32(prdt + 12, (bytes.max(1) - 1) | (1 << 31)); // DBC (0-based) + I

            // Command header slot 0: CFL=5 dwords, W bit, PRDTL=1.
            let cfl = 5u32;
            let flags = cfl | (if write { 1 << 6 } else { 0 }) | (1u32 << 16);
            write_volatile(self.clb.virt as *mut u32, flags);
            write_volatile((self.clb.virt + 4) as *mut u32, 0); // PRDBC

            // Wait for the port to be idle, then issue slot 0.
            let mut g = 0;
            while r32(self.port + P_TFD) & (TFD_BSY | TFD_DRQ) != 0 && g < 1_000_000 {
                g += 1;
            }
            fence(Ordering::SeqCst);
            w32(self.port + P_CI, 1);

            let mut spins = 0u64;
            while r32(self.port + P_CI) & 1 != 0 {
                if r32(self.port + P_IS) & (1 << 30) != 0 {
                    return false; // Task File Error
                }
                spins += 1;
                if spins > 2_000_000_000 {
                    crate::ktrace::log("ahci", "command never completed (2e9 spins) -- giving up");
                    return false;
                }
                // Answer Ctrl+C. The bound above is generous enough to be many
                // seconds on a failing drive, during which the machine would
                // otherwise be unescapable — the standing rule is that every
                // blocking operation stays interruptible. Checked periodically,
                // not every spin, so it costs nothing on a healthy device.
                //
                // Only `poll_interrupt` (a non-blocking console-ring read that
                // pushes back anything that is not Ctrl+C), never `upkeep()`:
                // block I/O runs from inside the UI pump and from early boot, so
                // pumping the UI here would risk exactly the re-entrancy hang that
                // `fix(ui): stop package-app open hang on surface HUD reentrancy`
                // addressed.
                if spins % 0x10_0000 == 0 && crate::shell::poll_interrupt() {
                    crate::ktrace::log("ahci", "command cancelled by Ctrl+C");
                    return false;
                }
                core::hint::spin_loop();
            }
            fence(Ordering::SeqCst);
            r32(self.port + P_TFD) & (TFD_BSY | 1) == 0 // not busy, no ERR
        }
    }

    fn rw(&mut self, write: bool, index: u64, ptr: *mut u8, len: usize) -> Result<(), BlockError> {
        let count = (len / BLOCK_SIZE) as u16;
        // SAFETY: bounce buffer + command table from bringup.
        unsafe {
            if write {
                core::ptr::copy_nonoverlapping(ptr as *const u8, self.data_buf.virt as *mut u8, len);
            }
            let cmd = if write { 0x35 } else { 0x25 }; // WRITE/READ DMA EXT
            if !self.issue(cmd, index, count, len as u32, write) {
                return Err(BlockError::DeviceError);
            }
            if !write {
                core::ptr::copy_nonoverlapping(self.data_buf.virt as *const u8, ptr, len);
            }
        }
        Ok(())
    }
}

impl BlockDevice for Ahci {
    fn block_count(&self) -> u64 {
        self.capacity
    }
    fn read_block(&mut self, index: u64, buf: &mut [u8]) -> Result<(), BlockError> {
        if buf.len() != BLOCK_SIZE {
            return Err(BlockError::BadBufferLen);
        }
        self.read_blocks(index, buf)
    }
    fn write_block(&mut self, index: u64, buf: &[u8]) -> Result<(), BlockError> {
        if buf.len() != BLOCK_SIZE {
            return Err(BlockError::BadBufferLen);
        }
        let mut tmp = [0u8; BLOCK_SIZE];
        tmp.copy_from_slice(buf);
        self.rw(true, index, tmp.as_mut_ptr(), BLOCK_SIZE)
    }
    fn read_blocks(&mut self, index: u64, buf: &mut [u8]) -> Result<(), BlockError> {
        if buf.len() % BLOCK_SIZE != 0 {
            return Err(BlockError::BadBufferLen);
        }
        let mut off = 0usize;
        while off < buf.len() {
            let take = (buf.len() - off).min(DATA_MAX);
            self.rw(false, index + (off / BLOCK_SIZE) as u64, buf[off..].as_mut_ptr(), take)?;
            off += take;
        }
        Ok(())
    }
    fn write_blocks(&mut self, index: u64, buf: &[u8]) -> Result<(), BlockError> {
        if buf.len() % BLOCK_SIZE != 0 {
            return Err(BlockError::BadBufferLen);
        }
        let mut off = 0usize;
        while off < buf.len() {
            let take = (buf.len() - off).min(DATA_MAX);
            self.rw(true, index + (off / BLOCK_SIZE) as u64, buf[off..].as_ptr() as *mut u8, take)?;
            off += take;
        }
        Ok(())
    }
}
