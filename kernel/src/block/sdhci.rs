//! **SDHCI** — the SD Host Controller, and through it SD cards and **eMMC**.
//!
//! This is the storage on machines that have no SATA and no NVMe: tablets,
//! Chromebook-class laptops, and most single-board computers. Without it those
//! machines boot ChittiOS and find no disk at all.
//!
//! Discovered as PCI class `08:05` (SD Host controller), which is what QEMU's
//! `sdhci-pci` presents and what the SDHCI-over-PCI parts in real machines use.
//! Transfers go through the **PIO data port** rather than SDMA/ADMA: a
//! descriptor engine is a second thing to get wrong for a driver whose job here
//! is to make the disk visible, and every wait in this file is bounded anyway.
//!
//! ## The card initialisation sequence, and why the order is not negotiable
//!
//! A card powers up in "idle" and has to be walked through a fixed ladder; a
//! step done out of order does not error, it produces a card that answers some
//! commands and not others.
//!
//! 1. `CMD0 GO_IDLE_STATE` — reset to idle.
//! 2. `CMD8 SEND_IF_COND` — **this is the version probe.** A v2.00+ SD card
//!    echoes the check pattern back; a v1.x card times out, which is a
//!    *successful* negative answer and not a failure. Skipping CMD8 makes every
//!    modern card come up as v1.x, which caps it at 2 GB and mis-addresses
//!    everything above that.
//! 3. `ACMD41` (SD) or `CMD1` (eMMC) — repeated until the card leaves busy. The
//!    `HCS` bit here is only honoured if CMD8 succeeded, which is why 2 comes
//!    first.
//! 4. `CMD2 ALL_SEND_CID`, `CMD3 SEND_RELATIVE_ADDR` — get an address.
//! 5. `CMD9 SEND_CSD` — capacity.
//! 6. `CMD7 SELECT_CARD`, then `CMD16 SET_BLOCKLEN 512`.
//!
//! ## Byte addressing versus block addressing
//!
//! **A standard-capacity card is addressed in bytes and a high-capacity one in
//! blocks**, and nothing in the command reports which. The distinction comes
//! from the OCR's `CCS` bit at step 3, and getting it wrong does not fail: a
//! byte-addressed read of block 1 lands at byte 1, i.e. inside block 0, so the
//! filesystem reads plausible, wrongly-offset data. [`SdCard::lba_arg`] is the
//! single place that decision is applied.

use crate::block::{BlockDevice, BlockError, BLOCK_SIZE};

#[cfg(target_arch = "aarch64")]
use crate::pci::{self, PciDevice};
#[cfg(target_arch = "x86_64")]
use crate::arch::x86_64::pci::{self, PciDevice};

// --- register offsets (SD Host Controller Simplified Spec 3.00, part 2) ----

const SDHCI_BLOCK_SIZE: u64 = 0x04; // u16
const SDHCI_BLOCK_COUNT: u64 = 0x06; // u16
const SDHCI_ARGUMENT: u64 = 0x08;
const SDHCI_TRANSFER_MODE: u64 = 0x0c; // u16
const SDHCI_COMMAND: u64 = 0x0e; // u16
const SDHCI_RESPONSE: u64 = 0x10; // 4 x u32
const SDHCI_BUFFER: u64 = 0x20; // PIO data port
const SDHCI_PRESENT_STATE: u64 = 0x24;
const SDHCI_HOST_CONTROL: u64 = 0x28; // u8
const SDHCI_POWER_CONTROL: u64 = 0x29; // u8
const SDHCI_CLOCK_CONTROL: u64 = 0x2c; // u16
const SDHCI_TIMEOUT_CONTROL: u64 = 0x2e; // u8
const SDHCI_SOFTWARE_RESET: u64 = 0x2f; // u8
const SDHCI_INT_STATUS: u64 = 0x30;
const SDHCI_INT_ENABLE: u64 = 0x34;
const SDHCI_SIGNAL_ENABLE: u64 = 0x38;
const SDHCI_CAPABILITIES: u64 = 0x40;

// Present state.
const STATE_CMD_INHIBIT: u32 = 1 << 0;
const STATE_DAT_INHIBIT: u32 = 1 << 1;
const STATE_CARD_INSERTED: u32 = 1 << 16;

// Interrupt status bits.
const INT_CMD_COMPLETE: u32 = 1 << 0;
const INT_XFER_COMPLETE: u32 = 1 << 1;
const INT_BUF_WRITE_READY: u32 = 1 << 4;
const INT_BUF_READ_READY: u32 = 1 << 5;
/// Any bit at or above 15 is an error interrupt.
const INT_ERROR: u32 = 1 << 15;

// Software reset.
const RESET_ALL: u8 = 1 << 0;
const RESET_CMD: u8 = 1 << 1;
const RESET_DATA: u8 = 1 << 2;

// Clock control.
const CLOCK_INTERNAL_EN: u16 = 1 << 0;
const CLOCK_INTERNAL_STABLE: u16 = 1 << 1;
const CLOCK_SD_EN: u16 = 1 << 2;

// Power control.
const POWER_ON: u8 = 1 << 0;

// Transfer mode.
const XFER_DMA: u16 = 1 << 0;
const XFER_BLOCK_COUNT_EN: u16 = 1 << 1;
const XFER_AUTO_CMD12: u16 = 1 << 2;
const XFER_READ: u16 = 1 << 4;
const XFER_MULTI_BLOCK: u16 = 1 << 5;

// Command register fields.
const CMD_RESP_NONE: u16 = 0;
const CMD_RESP_136: u16 = 1;
const CMD_RESP_48: u16 = 2;
const CMD_RESP_48_BUSY: u16 = 3;
const CMD_CRC_CHECK: u16 = 1 << 3;
const CMD_INDEX_CHECK: u16 = 1 << 4;
const CMD_DATA_PRESENT: u16 = 1 << 5;

/// The response shape a command expects — part of the command word, and the
/// thing that decides how many response registers are meaningful.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Resp {
    None,
    /// R1/R3/R6/R7 — a single 32-bit response.
    Short,
    /// R1b — short, with the card holding DAT low while busy.
    ShortBusy,
    /// R2 — the 136-bit CID/CSD.
    Long,
}

/// Pack the SDHCI command register: index, response type, and the CRC/index
/// checks the spec ties to each response shape.
///
/// The checks are **not free choices**: R2 carries no command index and R3
/// (the OCR) carries no CRC, so enabling those checks makes a correct card
/// report an error interrupt on a command that actually succeeded. Pure so the
/// table is pinned by tests rather than by a card.
pub fn command_word(index: u8, resp: Resp, data: bool, crc: bool, idx_check: bool) -> u16 {
    let r = match resp {
        Resp::None => CMD_RESP_NONE,
        Resp::Short => CMD_RESP_48,
        Resp::ShortBusy => CMD_RESP_48_BUSY,
        Resp::Long => CMD_RESP_136,
    };
    let mut w = ((index as u16) << 8) | r;
    if crc {
        w |= CMD_CRC_CHECK;
    }
    if idx_check {
        w |= CMD_INDEX_CHECK;
    }
    if data {
        w |= CMD_DATA_PRESENT;
    }
    w
}

/// Divider for the 8-bit clock-divisor field (SDHCI 2.0 "version 1" mode):
/// the register holds `base / (2 * div)`, so a divisor of 0 means "base
/// clock, undivided" rather than a division by zero.
///
/// Returns the register's `(low, high)` halves already positioned — the field
/// is **split**, low 8 bits at 8..15 and the upper 2 bits at 6..7, and packing
/// it as one contiguous field silently caps the divider at 255, which on a
/// 200 MHz base clock is a 390 kHz floor rather than the 100 kHz identification
/// speed a card must be initialised at.
pub fn clock_divider(base_hz: u32, target_hz: u32) -> u16 {
    if target_hz == 0 || base_hz <= target_hz {
        return 0;
    }
    let mut div = 1u32;
    // The field counts half-dividers: base / (2 * div).
    while base_hz / (2 * div) > target_hz && div < 0x3ff {
        div += 1;
    }
    let lo = (div & 0xff) as u16;
    let hi = ((div >> 8) & 0x3) as u16;
    (lo << 8) | (hi << 6)
}

/// Capacity in 512-byte blocks from a CSD register.
///
/// The two CSD versions compute capacity completely differently and are told
/// apart by the top two bits:
///
/// * **v1.0** (standard capacity): `(C_SIZE + 1) * 2^(C_SIZE_MULT + 2)` blocks
///   of `2^READ_BL_LEN` bytes, normalised here to 512-byte blocks.
/// * **v2.0** (high capacity): `(C_SIZE + 1) * 1024` blocks of 512 bytes, with
///   `C_SIZE` in a different place and 22 bits wide.
///
/// Applying the v1 formula to a v2 card yields a number that is small and
/// plausible — a 64 GB card reads as a few hundred megabytes — so the
/// filesystem mounts and then fails only past the phantom end.
///
/// `csd` is the 128-bit register as four big-endian words, `csd[0]` holding
/// bits 127..96, which is the order the response registers are read back in.
pub fn csd_capacity_blocks(csd: &[u32; 4]) -> Option<u64> {
    let version = csd[0] >> 30;
    match version {
        0 => {
            // v1.0. C_SIZE is bits 73..62, split across words.
            let read_bl_len = ((csd[1] >> 16) & 0xf) as u32;
            let c_size = ((csd[1] & 0x3ff) << 2) | ((csd[2] >> 30) & 0x3);
            let c_size_mult = ((csd[2] >> 15) & 0x7) as u32;
            if read_bl_len < 9 || read_bl_len > 11 {
                return None; // 512/1024/2048 are the only legal block lengths
            }
            let blocks = (c_size as u64 + 1) * (1u64 << (c_size_mult + 2));
            // Normalise the card's own block length to our 512-byte blocks.
            Some(blocks << (read_bl_len - 9))
        }
        1 => {
            // v2.0 (SDHC/SDXC). C_SIZE is bits 69..48, 22 bits.
            let c_size = ((csd[1] & 0x3f) << 16) | ((csd[2] >> 16) & 0xffff);
            Some((c_size as u64 + 1) * 1024)
        }
        // v3.0 (SDUC) uses a 28-bit C_SIZE in the same place; anything else is
        // a version this driver has not been written against, and guessing a
        // capacity is worse than reporting none.
        _ => None,
    }
}

/// The argument for a block command.
///
/// **A standard-capacity card takes a byte offset and a high-capacity card
/// takes a block index**, and the command itself carries no hint. A byte
/// address computed as a block index reads 512x too low — inside block 0 for
/// any small index — so the data is real, wrongly-offset bytes rather than an
/// error.
pub fn lba_arg(lba: u64, high_capacity: bool) -> u32 {
    if high_capacity {
        lba as u32
    } else {
        (lba.saturating_mul(BLOCK_SIZE as u64)) as u32
    }
}

// --- MMIO helpers ---------------------------------------------------------

/// Single-instruction MMIO, per the aarch64 rule in CLAUDE.md: LLVM otherwise
/// coalesces adjacent volatile accesses into a paired load HVF cannot decode.
#[inline(always)]
unsafe fn r32(base: u64, off: u64) -> u32 {
    // SAFETY: `base + off` is inside the mapped BAR of a claimed controller.
    unsafe { core::ptr::read_volatile((base + off) as *const u32) }
}
#[inline(always)]
unsafe fn w32(base: u64, off: u64, v: u32) {
    // SAFETY: as above.
    unsafe { core::ptr::write_volatile((base + off) as *mut u32, v) }
}
#[inline(always)]
unsafe fn r16(base: u64, off: u64) -> u16 {
    // SAFETY: as above.
    unsafe { core::ptr::read_volatile((base + off) as *const u16) }
}
#[inline(always)]
unsafe fn w16(base: u64, off: u64, v: u16) {
    // SAFETY: as above.
    unsafe { core::ptr::write_volatile((base + off) as *mut u16, v) }
}
#[inline(always)]
unsafe fn r8(base: u64, off: u64) -> u8 {
    // SAFETY: as above.
    unsafe { core::ptr::read_volatile((base + off) as *const u8) }
}
#[inline(always)]
unsafe fn w8(base: u64, off: u64, v: u8) {
    // SAFETY: as above.
    unsafe { core::ptr::write_volatile((base + off) as *mut u8, v) }
}

/// Spin until `f` is false or the bound is hit. Every wait here is bounded: a
/// card that never answers must fail the probe, never hang the boot.
fn spin_until(mut spins: u32, mut f: impl FnMut() -> bool) -> bool {
    while spins > 0 {
        if !f() {
            return true;
        }
        spins -= 1;
        core::hint::spin_loop();
    }
    false
}

/// An SD card or eMMC device behind one SDHCI controller.
pub struct SdCard {
    base: u64,
    rca: u16,
    blocks: u64,
    /// Block-addressed (SDHC/SDXC/eMMC > 2 GB) rather than byte-addressed.
    high_capacity: bool,
}

impl SdCard {
    /// Issue a command and wait for completion. Returns the response registers.
    ///
    /// Errors are read from the interrupt status: an error interrupt leaves the
    /// controller wedged for the next command, so the command and data lines
    /// are reset before returning. Skipping that makes one bad command turn
    /// into an endlessly failing controller, which reads as a dead card.
    unsafe fn cmd(&mut self, index: u8, arg: u32, resp: Resp, data: bool) -> Result<[u32; 4], BlockError> {
        let b = self.base;
        // SAFETY: `b` is the mapped BAR of a claimed controller throughout.
        unsafe {
            // Wait for the command line, and the data line too when this
            // command uses it — issuing over an active transfer aborts it.
            let mask = if data || resp == Resp::ShortBusy {
                STATE_CMD_INHIBIT | STATE_DAT_INHIBIT
            } else {
                STATE_CMD_INHIBIT
            };
            if !spin_until(1_000_000, || r32(b, SDHCI_PRESENT_STATE) & mask != 0) {
                return Err(BlockError::DeviceError);
            }
            // Clear stale status so this command's bits are unambiguous.
            w32(b, SDHCI_INT_STATUS, 0xffff_ffff);
            w32(b, SDHCI_ARGUMENT, arg);
            // R2 carries no command index and R3 no CRC, so those checks are
            // enabled per response shape rather than always.
            let (crc, idx) = match (index, resp) {
                (_, Resp::None) => (false, false),
                (_, Resp::Long) => (true, false),  // R2: CRC yes, index no
                (41, _) => (false, false),         // ACMD41 returns R3 (OCR)
                (1, _) => (false, false),          // CMD1 returns R3 (eMMC OCR)
                _ => (true, true),
            };
            w16(b, SDHCI_COMMAND, command_word(index, resp, data, crc, idx));

            if !spin_until(1_000_000, || {
                r32(b, SDHCI_INT_STATUS) & (INT_CMD_COMPLETE | INT_ERROR) == 0
            }) {
                self.reset_lines();
                return Err(BlockError::DeviceError);
            }
            let st = r32(b, SDHCI_INT_STATUS);
            if st & INT_ERROR != 0 {
                self.reset_lines();
                return Err(BlockError::DeviceError);
            }
            w32(b, SDHCI_INT_STATUS, INT_CMD_COMPLETE);
            Ok([
                r32(b, SDHCI_RESPONSE),
                r32(b, SDHCI_RESPONSE + 4),
                r32(b, SDHCI_RESPONSE + 8),
                r32(b, SDHCI_RESPONSE + 12),
            ])
        }
    }

    /// Reset the command and data lines after an error, leaving the card and
    /// the clock alone.
    unsafe fn reset_lines(&self) {
        let b = self.base;
        // SAFETY: mapped BAR.
        unsafe {
            w8(b, SDHCI_SOFTWARE_RESET, RESET_CMD | RESET_DATA);
            spin_until(100_000, || r8(b, SDHCI_SOFTWARE_RESET) & (RESET_CMD | RESET_DATA) != 0);
            w32(b, SDHCI_INT_STATUS, 0xffff_ffff);
        }
    }

    /// An application command: `CMD55` addressed to the card, then the ACMD.
    unsafe fn acmd(&mut self, index: u8, arg: u32, resp: Resp) -> Result<[u32; 4], BlockError> {
        // SAFETY: forwarded.
        unsafe {
            self.cmd(55, (self.rca as u32) << 16, Resp::Short, false)?;
            self.cmd(index, arg, resp, false)
        }
    }

    /// Move one block between the PIO data port and `buf`.
    ///
    /// The buffer-ready interrupt must be waited for **per block**: the port is
    /// one block deep, and writing ahead of it silently drops words rather than
    /// stalling.
    unsafe fn pio_block(&mut self, buf: &mut [u8], read: bool) -> Result<(), BlockError> {
        let b = self.base;
        let want = if read { INT_BUF_READ_READY } else { INT_BUF_WRITE_READY };
        // SAFETY: mapped BAR; `buf` is one BLOCK_SIZE block, checked by callers.
        unsafe {
            if !spin_until(1_000_000, || r32(b, SDHCI_INT_STATUS) & (want | INT_ERROR) == 0) {
                self.reset_lines();
                return Err(BlockError::DeviceError);
            }
            if r32(b, SDHCI_INT_STATUS) & INT_ERROR != 0 {
                self.reset_lines();
                return Err(BlockError::DeviceError);
            }
            w32(b, SDHCI_INT_STATUS, want);
            for chunk in buf.chunks_mut(4) {
                if read {
                    let w = r32(b, SDHCI_BUFFER).to_le_bytes();
                    chunk.copy_from_slice(&w[..chunk.len()]);
                } else {
                    let mut w = [0u8; 4];
                    w[..chunk.len()].copy_from_slice(chunk);
                    w32(b, SDHCI_BUFFER, u32::from_le_bytes(w));
                }
            }
        }
        Ok(())
    }

    /// Read or write `n` consecutive blocks through PIO.
    unsafe fn transfer(&mut self, lba: u64, buf: &mut [u8], read: bool) -> Result<(), BlockError> {
        if buf.len() % BLOCK_SIZE != 0 || buf.is_empty() {
            return Err(BlockError::BadBufferLen);
        }
        let n = (buf.len() / BLOCK_SIZE) as u16;
        if lba.saturating_add(n as u64) > self.blocks {
            return Err(BlockError::OutOfRange);
        }
        let b = self.base;
        // SAFETY: mapped BAR; bounds checked above.
        unsafe {
            w16(b, SDHCI_BLOCK_SIZE, BLOCK_SIZE as u16);
            w16(b, SDHCI_BLOCK_COUNT, n);
            let mut mode = if read { XFER_READ } else { 0 };
            if n > 1 {
                // A multi-block transfer must be stopped. AUTO_CMD12 makes the
                // controller send CMD12 itself; without it the card stays in
                // the transfer state and every later command fails.
                mode |= XFER_MULTI_BLOCK | XFER_BLOCK_COUNT_EN | XFER_AUTO_CMD12;
            }
            // No DMA: the PIO port is the transport here.
            mode &= !XFER_DMA;
            w16(b, SDHCI_TRANSFER_MODE, mode);

            let index = match (read, n > 1) {
                (true, false) => 17,  // READ_SINGLE_BLOCK
                (true, true) => 18,   // READ_MULTIPLE_BLOCK
                (false, false) => 24, // WRITE_BLOCK
                (false, true) => 25,  // WRITE_MULTIPLE_BLOCK
            };
            self.cmd(index, lba_arg(lba, self.high_capacity), Resp::Short, true)?;

            for chunk in buf.chunks_mut(BLOCK_SIZE) {
                self.pio_block(chunk, read)?;
            }
            // The transfer is only done when the controller says so — the last
            // buffer-ready does not mean the card has finished programming.
            if !spin_until(4_000_000, || {
                r32(b, SDHCI_INT_STATUS) & (INT_XFER_COMPLETE | INT_ERROR) == 0
            }) {
                self.reset_lines();
                return Err(BlockError::DeviceError);
            }
            let st = r32(b, SDHCI_INT_STATUS);
            w32(b, SDHCI_INT_STATUS, INT_XFER_COMPLETE);
            if st & INT_ERROR != 0 {
                self.reset_lines();
                return Err(BlockError::DeviceError);
            }
        }
        Ok(())
    }
}

impl BlockDevice for SdCard {
    fn block_count(&self) -> u64 {
        self.blocks
    }

    fn read_block(&mut self, index: u64, buf: &mut [u8]) -> Result<(), BlockError> {
        if buf.len() != BLOCK_SIZE {
            return Err(BlockError::BadBufferLen);
        }
        // SAFETY: the controller is claimed and mapped for this card's lifetime.
        unsafe { self.transfer(index, buf, true) }
    }

    fn write_block(&mut self, index: u64, buf: &[u8]) -> Result<(), BlockError> {
        if buf.len() != BLOCK_SIZE {
            return Err(BlockError::BadBufferLen);
        }
        let mut tmp = [0u8; BLOCK_SIZE];
        tmp.copy_from_slice(buf);
        // SAFETY: as above.
        unsafe { self.transfer(index, &mut tmp, false) }
    }

    /// Multi-block read in one command — the batching the standing performance
    /// rule requires (one polled round trip per request, not per 512 bytes).
    fn read_blocks(&mut self, index: u64, buf: &mut [u8]) -> Result<(), BlockError> {
        // SAFETY: as above.
        unsafe { self.transfer(index, buf, true) }
    }

    fn write_blocks(&mut self, index: u64, buf: &[u8]) -> Result<(), BlockError> {
        let mut tmp = alloc::vec![0u8; buf.len()];
        tmp.copy_from_slice(buf);
        // SAFETY: as above.
        unsafe { self.transfer(index, &mut tmp, false) }
    }
}

/// Probe the `n`-th SD host controller and bring up the card in it.
///
/// `None` when there is no such controller, no card in it, or the card does not
/// complete initialisation — each logged, because "no disk" otherwise has
/// several indistinguishable causes on a machine whose only storage this is.
pub fn probe_nth(n: usize) -> Option<SdCard> {
    // PCI class 08 (base system peripheral), subclass 05 (SD host controller).
    let dev: PciDevice = pci::find_class_sub_nth(0x08, 0x05, n)?;
    let bar = dev.bar(0);
    if bar == 0 {
        crate::ktrace::log("sdhci", "controller has no BAR0 -- skipped");
        return None;
    }
    dev.enable_bus_master();
    let base = crate::mm::map_mmio(bar, 0x1000);
    crate::ktrace::log_fmt(format_args!("sdhci: controller {n} at {bar:#x}"));

    // SAFETY: `base` is the freshly mapped BAR of a controller we just claimed.
    unsafe {
        // Full reset, then wait for it to clear.
        w8(base, SDHCI_SOFTWARE_RESET, RESET_ALL);
        if !spin_until(1_000_000, || r8(base, SDHCI_SOFTWARE_RESET) & RESET_ALL != 0) {
            crate::ktrace::log("sdhci", "reset never completed");
            return None;
        }
        // A card must be present before anything else is worth doing.
        if r32(base, SDHCI_PRESENT_STATE) & STATE_CARD_INSERTED == 0 {
            crate::ktrace::log("sdhci", "no card inserted");
            return None;
        }

        // Base clock from the capabilities register (MHz in bits 8..15 for
        // spec 2.0; 0 means "ask the platform", which we cannot, so assume the
        // common 50 MHz rather than dividing by zero).
        let caps = r32(base, SDHCI_CAPABILITIES);
        let base_mhz = ((caps >> 8) & 0xff).max(1);
        let base_hz = base_mhz * 1_000_000;

        // Identification runs at 400 kHz or below — a card will not answer
        // CMD0/CMD8 at full speed.
        w16(base, SDHCI_CLOCK_CONTROL, 0);
        let div = clock_divider(base_hz, 400_000);
        w16(base, SDHCI_CLOCK_CONTROL, div | CLOCK_INTERNAL_EN);
        if !spin_until(1_000_000, || {
            r16(base, SDHCI_CLOCK_CONTROL) & CLOCK_INTERNAL_STABLE == 0
        }) {
            crate::ktrace::log("sdhci", "internal clock never stabilised");
            return None;
        }
        w16(base, SDHCI_CLOCK_CONTROL, div | CLOCK_INTERNAL_EN | CLOCK_SD_EN);

        // Power: pick the highest voltage the controller advertises.
        let v = if caps & (1 << 24) != 0 {
            0x0e // 3.3 V
        } else if caps & (1 << 25) != 0 {
            0x0c // 3.0 V
        } else {
            0x0a // 1.8 V
        };
        w8(base, SDHCI_POWER_CONTROL, v | POWER_ON);
        w8(base, SDHCI_TIMEOUT_CONTROL, 0x0e); // max timeout
        // Poll rather than interrupt: enable the status bits, mask the signals.
        w32(base, SDHCI_INT_ENABLE, 0xffff_ffff);
        w32(base, SDHCI_SIGNAL_ENABLE, 0);
        w8(base, SDHCI_HOST_CONTROL, 0); // 1-bit bus, the always-safe width

        let mut card = SdCard { base, rca: 0, blocks: 0, high_capacity: false };

        // 1. GO_IDLE_STATE.
        card.cmd(0, 0, Resp::None, false).ok()?;

        // 2. SEND_IF_COND. `0x1AA` = 2.7-3.6 V, check pattern 0xAA. A v2.00+
        //    card echoes it; a v1.x card times out, which is a *negative
        //    answer*, not a failure — so the error is swallowed deliberately
        //    and only the echo decides.
        let v2 = match card.cmd(8, 0x1aa, Resp::Short, false) {
            Ok(r) => r[0] & 0xfff == 0x1aa,
            Err(_) => {
                card.reset_lines();
                false
            }
        };

        // 3. Leave busy. SD first (ACMD41); an eMMC does not answer CMD55 and
        //    we fall back to CMD1.
        let hcs = if v2 { 1 << 30 } else { 0 };
        let mut ocr = 0u32;
        let mut is_sd = true;
        let mut ready = false;
        for _ in 0..1000 {
            match card.acmd(41, hcs | 0x00ff_8000, Resp::Short) {
                Ok(r) => {
                    ocr = r[0];
                    if ocr & (1 << 31) != 0 {
                        ready = true;
                        break;
                    }
                }
                Err(_) => {
                    card.reset_lines();
                    is_sd = false;
                    break;
                }
            }
        }
        if !is_sd {
            for _ in 0..1000 {
                match card.cmd(1, 0x40ff_8000, Resp::Short, false) {
                    Ok(r) => {
                        ocr = r[0];
                        if ocr & (1 << 31) != 0 {
                            ready = true;
                            break;
                        }
                    }
                    Err(_) => {
                        card.reset_lines();
                        break;
                    }
                }
            }
        }
        if !ready {
            crate::ktrace::log("sdhci", "card never left busy (ACMD41/CMD1)");
            return None;
        }
        // **CCS decides byte- vs block-addressing.** Nothing later reports it.
        card.high_capacity = ocr & (1 << 30) != 0;

        // 4. Address the card.
        card.cmd(2, 0, Resp::Long, false).ok()?;
        if is_sd {
            let r = card.cmd(3, 0, Resp::Short, false).ok()?;
            card.rca = (r[0] >> 16) as u16;
        } else {
            // eMMC is *assigned* an address by the host rather than proposing
            // one; 1 is as good as any non-zero value.
            card.rca = 1;
            card.cmd(3, (card.rca as u32) << 16, Resp::Short, false).ok()?;
        }

        // 5. Capacity from the CSD.
        let csd = card.cmd(9, (card.rca as u32) << 16, Resp::Long, false).ok()?;
        // The response registers hold bits 127..8 shifted down by 8 (the CRC
        // byte is not returned), so realign before decoding — reading them raw
        // puts every CSD field 8 bits out and yields a plausible wrong size.
        let aligned = [
            (csd[3] << 8) | (csd[2] >> 24),
            (csd[2] << 8) | (csd[1] >> 24),
            (csd[1] << 8) | (csd[0] >> 24),
            csd[0] << 8,
        ];
        let blocks = csd_capacity_blocks(&aligned).or_else(|| {
            crate::ktrace::log("sdhci", "unrecognised CSD version -- refusing to guess a capacity");
            None
        })?;
        card.blocks = blocks;

        // 6. Select the card and pin the block length. CMD16 is a no-op on a
        //    high-capacity card (fixed 512) but harmless, and required below it.
        card.cmd(7, (card.rca as u32) << 16, Resp::ShortBusy, false).ok()?;
        card.cmd(16, BLOCK_SIZE as u32, Resp::Short, false).ok()?;

        // Identification is done: raise the clock to 25 MHz for transfers.
        w16(base, SDHCI_CLOCK_CONTROL, 0);
        let fast = clock_divider(base_hz, 25_000_000);
        w16(base, SDHCI_CLOCK_CONTROL, fast | CLOCK_INTERNAL_EN);
        spin_until(1_000_000, || {
            r16(base, SDHCI_CLOCK_CONTROL) & CLOCK_INTERNAL_STABLE == 0
        });
        w16(base, SDHCI_CLOCK_CONTROL, fast | CLOCK_INTERNAL_EN | CLOCK_SD_EN);

        crate::ktrace::log_fmt(format_args!(
            "sdhci: {} card up, rca {:#x}, {} blocks ({} MiB), {}-addressed",
            if is_sd { "SD" } else { "eMMC" },
            card.rca,
            card.blocks,
            card.blocks / 2048,
            if card.high_capacity { "block" } else { "byte" }
        ));
        Some(card)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The CRC and index checks are tied to the response shape by the spec, not
    /// chosen: R2 carries no command index and R3 no CRC, so enabling those
    /// checks makes a *correct* card raise an error interrupt on a command that
    /// succeeded.
    #[test_case]
    fn command_word_packs_index_and_response_type() {
        // CMD17 READ_SINGLE_BLOCK: R1, data present, both checks.
        let w = command_word(17, Resp::Short, true, true, true);
        assert_eq!(w >> 8, 17, "command index");
        assert_eq!(w & 0x3, CMD_RESP_48, "R1 is a 48-bit response");
        assert_ne!(w & CMD_DATA_PRESENT, 0);
        assert_ne!(w & CMD_CRC_CHECK, 0);
        assert_ne!(w & CMD_INDEX_CHECK, 0);

        // CMD0 GO_IDLE_STATE: no response at all.
        let w = command_word(0, Resp::None, false, false, false);
        assert_eq!(w, 0, "no index, no response, no checks");

        // CMD2 ALL_SEND_CID: R2, 136-bit, no index check.
        let w = command_word(2, Resp::Long, false, true, false);
        assert_eq!(w & 0x3, CMD_RESP_136);
        assert_eq!(w & CMD_INDEX_CHECK, 0, "R2 carries no command index");

        // CMD7 SELECT_CARD uses R1b — busy is part of the response type.
        assert_eq!(command_word(7, Resp::ShortBusy, false, true, true) & 0x3, CMD_RESP_48_BUSY);
    }

    /// The clock divisor field is **split** — low 8 bits at 8..15, upper 2 at
    /// 6..7. Packed as one contiguous field it caps at 255, which on a 200 MHz
    /// base clock is a 390 kHz floor rather than the ≤400 kHz identification
    /// speed a card needs; the card then never answers CMD0.
    #[test_case]
    fn clock_divider_splits_its_field_and_reaches_identification_speed() {
        // Base at or below the target needs no division at all.
        assert_eq!(clock_divider(25_000_000, 25_000_000), 0);
        assert_eq!(clock_divider(400_000, 25_000_000), 0);
        assert_eq!(clock_divider(50_000_000, 0), 0);

        // 50 MHz down to 400 kHz: div = 63 (50e6 / 126 = 396 kHz), one byte.
        let d = clock_divider(50_000_000, 400_000);
        let div = ((d >> 8) & 0xff) | (((d >> 6) & 0x3) << 8);
        assert!(50_000_000 / (2 * div as u32) <= 400_000, "must not exceed 400 kHz");

        // 200 MHz down to 400 kHz needs div = 250, still one byte...
        let d = clock_divider(200_000_000, 400_000);
        let div = ((d >> 8) & 0xff) | (((d >> 6) & 0x3) << 8);
        assert!(200_000_000 / (2 * div as u32) <= 400_000);

        // ...and 400 MHz needs 500, which does NOT fit in the low byte. This is
        // the case a contiguous packing gets wrong.
        let d = clock_divider(400_000_000, 400_000);
        let div = ((d >> 8) & 0xff) | (((d >> 6) & 0x3) << 8);
        assert!(div > 255, "the upper bits must be used, got {div}");
        assert!(400_000_000 / (2 * div as u32) <= 400_000);
        assert_ne!(d & (0x3 << 6), 0, "upper divisor bits must be populated");
    }

    /// **The two CSD versions compute capacity completely differently.** Reading
    /// a v2 card with the v1 formula gives a small, plausible number — a 64 GB
    /// card reads as a few hundred MB — so the filesystem mounts and only fails
    /// past the phantom end.
    #[test_case]
    fn csd_capacity_distinguishes_the_two_versions() {
        // v2.0 (CSD_STRUCTURE = 1), C_SIZE = 0x00_F000 → (61440+1)*1024 blocks
        // = ~30 GiB, a real 32 GB SDHC card.
        let mut csd = [0u32; 4];
        csd[0] = 1 << 30;
        let c_size: u32 = 0xf000;
        csd[1] = (c_size >> 16) & 0x3f;
        csd[2] = (c_size & 0xffff) << 16;
        let blocks = csd_capacity_blocks(&csd).unwrap();
        assert_eq!(blocks, (0xf000 + 1) * 1024);
        assert!(blocks * 512 / (1 << 30) >= 29, "about 30 GiB");

        // v1.0 (CSD_STRUCTURE = 0): READ_BL_LEN = 9 (512 B), C_SIZE = 3751,
        // C_SIZE_MULT = 7 → (3752) * 512 blocks ≈ 1 GB.
        let mut csd = [0u32; 4];
        csd[0] = 0;
        let c_size: u32 = 3751;
        csd[1] = (9 << 16) | (c_size >> 2);
        csd[2] = ((c_size & 0x3) << 30) | (7 << 15);
        let blocks = csd_capacity_blocks(&csd).unwrap();
        assert_eq!(blocks, 3752 * 512);

        // A v1 card whose native block length is 1024 must be normalised to our
        // 512-byte blocks, not reported as-is.
        let mut big_bl = csd;
        big_bl[1] = (10 << 16) | (c_size >> 2);
        assert_eq!(csd_capacity_blocks(&big_bl).unwrap(), 3752 * 512 * 2);

        // An illegal block length is refused rather than shifted by a negative.
        let mut bad = csd;
        bad[1] = (15 << 16) | (c_size >> 2);
        assert_eq!(csd_capacity_blocks(&bad), None);

        // An unknown CSD version reports nothing rather than guessing: a wrong
        // capacity is worse than an unusable card, because it corrupts.
        let mut v3 = [0u32; 4];
        v3[0] = 2 << 30;
        assert_eq!(csd_capacity_blocks(&v3), None);
        v3[0] = 3 << 30;
        assert_eq!(csd_capacity_blocks(&v3), None);
    }

    /// **Standard-capacity cards are byte-addressed and high-capacity ones
    /// block-addressed**, and the command carries no hint. Getting it wrong does
    /// not error: a byte address used as a block index reads 512x too low, which
    /// for any small index lands inside block 0 — real bytes, wrong offset.
    #[test_case]
    fn block_addressing_follows_the_capacity_class() {
        assert_eq!(lba_arg(0, true), 0);
        assert_eq!(lba_arg(0, false), 0, "block 0 is byte 0 either way");
        assert_eq!(lba_arg(1, true), 1);
        assert_eq!(lba_arg(1, false), 512, "byte-addressed: block 1 is byte 512");
        assert_eq!(lba_arg(1000, true), 1000);
        assert_eq!(lba_arg(1000, false), 512_000);
        // A byte address saturates rather than wrapping: a standard-capacity
        // card cannot exceed 2 GB anyway, and a wrapped address would name a
        // real, wrong block instead of failing.
        assert_eq!(lba_arg(u64::MAX, false), u32::MAX);
    }
}
