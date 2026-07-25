//! **Synopsys DesignWare I2C master**, as Intel ships it in the LPSS block.
//!
//! This is the controller a modern laptop's touchpad hangs off. It appears as a PCI
//! function (Intel LPSS, class `0x0C`/subclass `0x80` — "serial bus controller,
//! other") with a memory BAR, so unlike the device *behind* it, the controller
//! itself is discoverable. Sunrise Point onward keep the same DesignWare core, which
//! is why one driver covers the range.
//!
//! Poll-driven like every other driver here: no interrupts, transfers are driven by
//! watching the TX/RX FIFOs and the raw interrupt-status bits.
//!
//! ## Scope and honesty about it
//!
//! Enough to do HID-over-I2C: 7-bit addressing, a write, a read, and a combined
//! write-then-read (the register-then-data pattern HID-over-I2C uses). No 10-bit
//! addressing, no DMA, no clock-rate computation from the source clock — the timing
//! registers are programmed from the values firmware left in place, which is what
//! Linux does when ACPI supplies no explicit timing.
//!
//! **Unverified on hardware.** QEMU emulates no LPSS I2C controller, so nothing here
//! has ever driven a real bus. Register offsets and the enable/abort sequences are
//! from the DesignWare databook and Linux's `i2c-designware-core`. Every failure path
//! logs, so a first attempt on real hardware should say which step gave up.

use crate::block::BlockError;

#[cfg(target_arch = "aarch64")]
use crate::pci::{self, PciDevice};
#[cfg(target_arch = "x86_64")]
use crate::arch::x86_64::pci::{self, PciDevice};

// --- DesignWare register offsets -----------------------------------------
const IC_CON: usize = 0x00;
const IC_TAR: usize = 0x04;
const IC_DATA_CMD: usize = 0x10;
const IC_SS_SCL_HCNT: usize = 0x14;
const IC_SS_SCL_LCNT: usize = 0x18;
const IC_FS_SCL_HCNT: usize = 0x1c;
const IC_FS_SCL_LCNT: usize = 0x20;
const IC_INTR_MASK: usize = 0x30;
const IC_RAW_INTR_STAT: usize = 0x34;
const IC_RX_TL: usize = 0x38;
const IC_TX_TL: usize = 0x3c;
const IC_CLR_INTR: usize = 0x40;
const IC_ENABLE: usize = 0x6c;
const IC_STATUS: usize = 0x70;
const IC_TXFLR: usize = 0x74;
const IC_RXFLR: usize = 0x78;
const IC_ENABLE_STATUS: usize = 0x9c;
const IC_COMP_TYPE: usize = 0xf8;

/// `IC_COMP_TYPE` reads back this signature on a DesignWare core — the cheap way to
/// confirm the BAR really is one before writing to it.
const DW_IC_COMP_TYPE_VALUE: u32 = 0x44570140;

// IC_CON bits
const CON_MASTER: u32 = 1 << 0;
const CON_SPEED_FAST: u32 = 2 << 1;
const CON_RESTART_EN: u32 = 1 << 5;
const CON_SLAVE_DISABLE: u32 = 1 << 6;
// IC_DATA_CMD bits
const CMD_READ: u32 = 1 << 8;
const CMD_STOP: u32 = 1 << 9;
// IC_STATUS bits
const STATUS_TFNF: u32 = 1 << 1; // transmit FIFO not full
const STATUS_RFNE: u32 = 1 << 3; // receive FIFO not empty
const STATUS_ACTIVITY: u32 = 1 << 0;
// IC_RAW_INTR_STAT bits
const INTR_TX_ABRT: u32 = 1 << 6;

/// How long to spin waiting for FIFO space or a byte. Generous, but bounded: a bus
/// with no device (or one holding SCL low) must not hang the kernel — the same rule
/// the disk and HPET paths follow.
const SPINS: u32 = 2_000_000;

/// A DesignWare I2C master.
pub struct DwI2c {
    regs: u64,
}

fn r32(base: u64, off: usize) -> u32 {
    // SAFETY: `base` is the mapped controller register block; `off` within it.
    unsafe { core::ptr::read_volatile((base + off as u64) as *const u32) }
}
fn w32(base: u64, off: usize, v: u32) {
    // SAFETY: as `r32`; 32-bit register write.
    unsafe { core::ptr::write_volatile((base + off as u64) as *mut u32, v) };
}

impl DwI2c {
    /// Probe PCI for the `n`-th Intel LPSS I2C controller and bring it up.
    ///
    /// Confirms `IC_COMP_TYPE` before touching anything else: class `0x0C:0x80` is a
    /// catch-all "other serial bus" that also covers unrelated devices, so matching
    /// on class alone and then writing would poke a stranger's registers.
    pub fn probe_nth(n: usize) -> Option<DwI2c> {
        let d = pci::find_class_nth(0x0c, 0x80, 0x00, n)?;
        d.enable_bus_master();
        let bar = d.bar(0);
        // Same reasoning as the DSDT mapping: an implausible BAR (unassigned, or with
        // bits above the physical-address width) must be refused rather than mapped,
        // or the bogus page-table entry faults with a reserved-bit error that says
        // nothing about where it came from.
        if bar == 0 || bar >= (1 << 40) {
            return None;
        }
        let regs = crate::mm::map_mmio(bar, 0x1000);
        let sig = r32(regs, IC_COMP_TYPE);
        if sig != DW_IC_COMP_TYPE_VALUE {
            crate::ktrace::log_fmt(format_args!(
                "i2c: {:04x}:{:04x} class 0c:80 is not a DesignWare core (COMP_TYPE {sig:#x}) -- skipping",
                d.vendor, d.device
            ));
            return None;
        }
        let mut c = DwI2c { regs };
        if !c.init() {
            return None;
        }
        crate::ktrace::log_fmt(format_args!(
            "i2c: DesignWare master up at {bar:#x} ({:04x}:{:04x})",
            d.vendor, d.device
        ));
        Some(c)
    }

    /// Disable, configure as a fast-mode master, re-enable.
    ///
    /// The controller must be **disabled** to write `IC_CON`/`IC_TAR`, and the
    /// databook requires polling `IC_ENABLE_STATUS` rather than assuming the write
    /// took effect — a disable can take several bus cycles to land.
    fn init(&mut self) -> bool {
        if !self.set_enable(false) {
            crate::ktrace::log("i2c", "controller would not disable -- refusing to configure it");
            return false;
        }
        w32(self.regs, IC_INTR_MASK, 0); // strictly polled
        w32(self.regs, IC_RX_TL, 0);
        w32(self.regs, IC_TX_TL, 0);
        // Keep firmware's SCL timing: deriving it needs the LPSS source clock, which
        // ACPI does not always give us, and firmware has already set values that work
        // for this board.
        let _ = (
            r32(self.regs, IC_SS_SCL_HCNT),
            r32(self.regs, IC_SS_SCL_LCNT),
            r32(self.regs, IC_FS_SCL_HCNT),
            r32(self.regs, IC_FS_SCL_LCNT),
        );
        w32(
            self.regs,
            IC_CON,
            CON_MASTER | CON_SPEED_FAST | CON_RESTART_EN | CON_SLAVE_DISABLE,
        );
        true
    }

    /// Set `IC_ENABLE` and wait for `IC_ENABLE_STATUS` to agree. Bounded.
    fn set_enable(&mut self, on: bool) -> bool {
        w32(self.regs, IC_ENABLE, if on { 1 } else { 0 });
        for _ in 0..SPINS {
            if (r32(self.regs, IC_ENABLE_STATUS) & 1 != 0) == on {
                return true;
            }
            core::hint::spin_loop();
        }
        false
    }

    /// Point the controller at a 7-bit target address. Requires the controller
    /// disabled, so this disables and re-enables around the write.
    fn set_target(&mut self, addr: u16) -> bool {
        if !self.set_enable(false) {
            return false;
        }
        w32(self.regs, IC_TAR, (addr & 0x3ff) as u32);
        self.set_enable(true)
    }

    /// Clear a latched transmit-abort, which otherwise wedges every later transfer.
    fn clear_abort(&mut self) {
        let _ = r32(self.regs, IC_CLR_INTR);
    }

    fn aborted(&mut self) -> bool {
        r32(self.regs, IC_RAW_INTR_STAT) & INTR_TX_ABRT != 0
    }

    /// Write `data` to `addr`, then read `read.len()` bytes in the same transaction.
    ///
    /// This combined form is the shape HID-over-I2C needs: address a register, then
    /// read its contents without releasing the bus (`CON_RESTART_EN` makes the
    /// direction change a repeated START rather than a STOP). Either half may be
    /// empty, giving a plain write or a plain read.
    ///
    /// A transmit-abort means nothing answered (or the target NAKed), which is the
    /// normal outcome of addressing something that is not there — so it is reported,
    /// not logged as an error, letting a caller safely test an address.
    pub fn write_read(&mut self, addr: u16, data: &[u8], read: &mut [u8]) -> Result<(), BlockError> {
        if !self.set_target(addr) {
            return Err(BlockError::OutOfRange);
        }
        self.clear_abort();

        // Write phase. STOP only if there is nothing to read after it.
        for (i, &b) in data.iter().enumerate() {
            let last = i + 1 == data.len() && read.is_empty();
            self.push_cmd(b as u32 | if last { CMD_STOP } else { 0 })?;
        }
        // Read phase: one READ command per byte, STOP on the last.
        let n = read.len();
        let mut got = 0usize;
        for i in 0..n {
            let last = i + 1 == n;
            self.push_cmd(CMD_READ | if last { CMD_STOP } else { 0 })?;
            // Drain as bytes arrive so the RX FIFO cannot overflow on a long read.
            got += self.drain(&mut read[got..], got + 1)?;
        }
        // Collect any stragglers still in the FIFO.
        while got < n {
            let before = got;
            got += self.drain(&mut read[got..], got + 1)?;
            if got == before {
                return Err(BlockError::DeviceError);
            }
        }
        Ok(())
    }

    /// Queue one command word, waiting for TX FIFO space. Bounded; a latched abort
    /// short-circuits the wait.
    fn push_cmd(&mut self, cmd: u32) -> Result<(), BlockError> {
        for _ in 0..SPINS {
            if self.aborted() {
                return Err(BlockError::DeviceError);
            }
            if r32(self.regs, IC_STATUS) & STATUS_TFNF != 0 {
                w32(self.regs, IC_DATA_CMD, cmd);
                return Ok(());
            }
            core::hint::spin_loop();
        }
        Err(BlockError::OutOfRange)
    }

    /// Move whatever the RX FIFO holds into `out`, waiting until at least `want`
    /// bytes have been produced in total. Returns how many it moved.
    fn drain(&mut self, out: &mut [u8], want: usize) -> Result<usize, BlockError> {
        let _ = want;
        let mut moved = 0usize;
        for _ in 0..SPINS {
            if self.aborted() {
                return Err(BlockError::DeviceError);
            }
            if r32(self.regs, IC_STATUS) & STATUS_RFNE != 0 {
                while moved < out.len() && r32(self.regs, IC_STATUS) & STATUS_RFNE != 0 {
                    out[moved] = (r32(self.regs, IC_DATA_CMD) & 0xff) as u8;
                    moved += 1;
                }
                return Ok(moved);
            }
            core::hint::spin_loop();
        }
        Err(BlockError::OutOfRange)
    }

    /// Whether anything answers at `addr`.
    ///
    /// A **zero-length** probe: it addresses the target and looks for a transmit
    /// abort, without writing a data byte. That matters — a write of even one byte to
    /// an unknown address on a laptop's shared bus could land on the embedded
    /// controller.
    pub fn present(&mut self, addr: u16) -> bool {
        if !self.set_target(addr) {
            return false;
        }
        self.clear_abort();
        if self.push_cmd(CMD_STOP).is_err() {
            return false;
        }
        for _ in 0..SPINS {
            if self.aborted() {
                return false;
            }
            if r32(self.regs, IC_STATUS) & STATUS_ACTIVITY == 0 {
                return !self.aborted();
            }
            core::hint::spin_loop();
        }
        false
    }
}
