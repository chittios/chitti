//! The **Apple I2C master** (a PASemi-derived controller), which is how the
//! audio amplifier and the headphone codec are configured.
//!
//! Not the DesignWare controller [`crate::drivers::i2c`] drives on x86 laptops —
//! same bus, completely different registers. The transfer model is a pair of
//! FIFOs: every byte written to `MTXFIFO` carries flags saying whether to issue
//! a START before it or a STOP after it, so a whole transaction is queued as
//! words and the controller runs it.
//!
//! Ported from m1n1's `proxyclient/m1n1/hw/i2c.py` (`write_reg`/`read_reg`) and
//! `src/i2c.c`. **The python is the reference here, not the C**: `i2c.c`
//! implements *SMBus block* transfers, which insert a length byte after the
//! register address. An audio codec is a plain register device — it would read
//! that length byte as data and write it into the register.
//!
//! Every wait is bounded and every failure says which step it was: on a bus that
//! also carries the embedded controller and the PMU, "no answer" and "the device
//! NAKed" are different facts, and the second one is what an absent or
//! unpowered chip looks like.

/// Register offsets (m1n1 `I2CRegs`).
pub mod reg {
    pub const MTXFIFO: u64 = 0x00;
    pub const MRXFIFO: u64 = 0x04;
    pub const SMSTA: u64 = 0x14;
    pub const CTL: u64 = 0x1c;
}

/// `MTXFIFO` flags: what to do around the byte in `DATA`.
pub const TX_READ: u32 = 1 << 10;
pub const TX_STOP: u32 = 1 << 9;
pub const TX_START: u32 = 1 << 8;
/// `MRXFIFO`: no byte available.
pub const RX_EMPTY: u32 = 1 << 8;
/// `SMSTA` bits worth naming.
pub const SMSTA_XEN: u32 = 1 << 27; // transaction ended
pub const SMSTA_MTN: u32 = 1 << 21; // master received a NACK
/// `CTL` bits: the controller enable and the FIFO resets.
pub const CTL_ENABLE: u32 = 1 << 11;
pub const CTL_MRR: u32 = 1 << 10; // reset master RX FIFO
pub const CTL_MTR: u32 = 1 << 9; // reset master TX FIFO
/// Clock divider m1n1 uses for these buses.
pub const CTL_CLK: u32 = 0x4;

/// The word that starts a transaction addressed to `addr`.
///
/// I2C's 7-bit address sits in bits 7:1 with the direction in bit 0, which is
/// why every one of these is `addr << 1`. Writing the address unshifted is the
/// classic version of this bug: it addresses a device at half the address, and
/// on a bus that carries a power controller that is not a harmless mistake.
pub fn addr_word(addr: u8, read: bool) -> u32 {
    TX_START | ((addr as u32) << 1) | read as u32
}

/// The word that ends a write transaction with `byte`.
pub fn data_word(byte: u8, stop: bool) -> u32 {
    byte as u32 | if stop { TX_STOP } else { 0 }
}

/// The word that asks the controller to clock in `len` bytes and then stop.
pub fn read_word(len: u8) -> u32 {
    TX_READ | TX_STOP | len as u32
}

/// What went wrong, in terms a person reading a serial log can act on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum I2cError {
    /// The device did not acknowledge its address — absent, or powered down.
    Nak,
    /// The controller never reported the transaction ending.
    Timeout,
    /// The receive FIFO stayed empty; the device stopped clocking data out.
    ReadTimeout,
}

impl I2cError {
    pub fn as_str(self) -> &'static str {
        match self {
            I2cError::Nak => "no acknowledgement from the device (absent or powered down)",
            I2cError::Timeout => "the controller never reported the transfer ending",
            I2cError::ReadTimeout => "the device stopped sending before the requested length",
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod hw {
    use super::*;

    /// Bound on a single transaction. An I2C byte at 100 kHz is ~90 us, so a
    /// register write is well under a millisecond; 50 ms is a bound rather than
    /// an expectation, and reaching it means the bus is wedged.
    const TIMEOUT_MS: u64 = 50;

    /// An Apple I2C master, already powered and mapped.
    pub struct I2c {
        base: u64,
    }

    impl I2c {
        /// Adopt the controller at physical `base`.
        ///
        /// # Safety
        /// `base` must be an Apple I2C register block from the device tree, and
        /// its power domain must already be on — a gated block reads all-ones,
        /// which this cannot distinguish from a wedged bus.
        pub unsafe fn new(base: u64, size: usize) -> Self {
            I2c { base: crate::mm::map_mmio(base, size) }
        }

        fn r(&self, off: u64) -> u32 {
            let v: u32;
            // SAFETY: `base + off` is inside the mapped register block.
            unsafe {
                core::arch::asm!("ldr {0:w}, [{1}]", out(reg) v, in(reg) self.base + off, options(nostack))
            };
            v
        }

        fn w(&self, off: u64, v: u32) {
            // SAFETY: as `r`.
            unsafe {
                core::arch::asm!("str {0:w}, [{1}]", in(reg) v, in(reg) self.base + off, options(nostack))
            };
        }

        /// Reset both FIFOs and clear the status bits, so a previous wedged
        /// transaction cannot be mistaken for this one's result.
        fn begin(&self) {
            self.w(reg::CTL, CTL_MTR | CTL_MRR);
            self.w(reg::SMSTA, 0xffff_ffff);
            self.w(reg::CTL, CTL_ENABLE | CTL_CLK);
        }

        fn end(&self) {
            self.w(reg::CTL, CTL_CLK);
        }

        /// Wait for the transaction-ended bit, treating a NAK as its own answer.
        fn wait_done(&self) -> Result<(), I2cError> {
            let deadline = crate::arch::now_ms() + TIMEOUT_MS;
            loop {
                let s = self.r(reg::SMSTA);
                if s & SMSTA_MTN != 0 {
                    return Err(I2cError::Nak);
                }
                if s & SMSTA_XEN != 0 {
                    return Ok(());
                }
                if crate::arch::now_ms() >= deadline {
                    return Err(I2cError::Timeout);
                }
                core::hint::spin_loop();
            }
        }

        /// Write `data` to register `reg_addr` of the device at `addr`.
        pub fn write_reg(&self, addr: u8, reg_addr: u8, data: &[u8]) -> Result<(), I2cError> {
            self.begin();
            self.w(reg::MTXFIFO, addr_word(addr, false));
            // The register address is just the first data byte of the write.
            let last = data.len();
            self.w(reg::MTXFIFO, data_word(reg_addr, last == 0));
            for (i, &b) in data.iter().enumerate() {
                self.w(reg::MTXFIFO, data_word(b, i + 1 == last));
            }
            let r = self.wait_done();
            self.end();
            r
        }

        /// Read `out.len()` bytes from register `reg_addr` of `addr`.
        ///
        /// The read is the two-transaction form every register device uses:
        /// write the register address with **no STOP**, then a repeated START
        /// with the read bit. Issuing a STOP in between makes the device forget
        /// which register was selected, and it answers from wherever its pointer
        /// happens to be — plausible data, wrong register.
        pub fn read_reg(&self, addr: u8, reg_addr: u8, out: &mut [u8]) -> Result<(), I2cError> {
            if out.is_empty() || out.len() > 255 {
                return Err(I2cError::ReadTimeout);
            }
            self.begin();
            self.w(reg::MTXFIFO, addr_word(addr, false));
            self.w(reg::MTXFIFO, data_word(reg_addr, false));
            self.w(reg::MTXFIFO, addr_word(addr, true));
            self.w(reg::MTXFIFO, read_word(out.len() as u8));
            let deadline = crate::arch::now_ms() + TIMEOUT_MS;
            for slot in out.iter_mut() {
                loop {
                    let v = self.r(reg::MRXFIFO);
                    if v & RX_EMPTY == 0 {
                        *slot = v as u8;
                        break;
                    }
                    if self.r(reg::SMSTA) & SMSTA_MTN != 0 {
                        self.end();
                        return Err(I2cError::Nak);
                    }
                    if crate::arch::now_ms() >= deadline {
                        self.end();
                        return Err(I2cError::ReadTimeout);
                    }
                    core::hint::spin_loop();
                }
            }
            self.end();
            Ok(())
        }

        /// Read one register byte.
        pub fn read_reg8(&self, addr: u8, reg_addr: u8) -> Result<u8, I2cError> {
            let mut b = [0u8; 1];
            self.read_reg(addr, reg_addr, &mut b)?;
            Ok(b[0])
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub use hw::I2c;

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn an_address_word_shifts_the_address_and_carries_the_direction() {
        // The amp is at 0x38, so a write must put 0x70 on the bus and a read
        // 0x71. An unshifted address would address device 0x38>>1 = 0x1c, which
        // on this bus is somebody else's chip.
        assert_eq!(addr_word(0x38, false), TX_START | 0x70);
        assert_eq!(addr_word(0x38, true), TX_START | 0x71);
        assert_eq!(addr_word(0x4b, false), TX_START | 0x96);
    }

    #[test_case]
    fn only_the_last_byte_of_a_write_carries_stop() {
        assert_eq!(data_word(0xa5, false), 0xa5);
        assert_eq!(data_word(0xa5, true), 0xa5 | TX_STOP);
        // A read request is a length, not a byte, and always ends the
        // transaction.
        assert_eq!(read_word(4), TX_READ | TX_STOP | 4);
        assert_eq!(read_word(1) & 0xff, 1);
    }

    #[test_case]
    fn the_flag_bits_do_not_overlap_the_data_field() {
        // Every flag lives above the byte, so a data byte of 0xff cannot set
        // one by accident — which would turn a data byte into a spurious STOP.
        for f in [TX_READ, TX_STOP, TX_START, RX_EMPTY] {
            assert_eq!(f & 0xff, 0, "flag {f:#x} overlaps the data field");
        }
        assert_eq!(data_word(0xff, false) & (TX_READ | TX_STOP | TX_START), 0);
    }

    #[test_case]
    fn every_failure_reads_differently() {
        // A person on a serial console has to tell "the chip is not there" from
        // "the bus is wedged"; a single "i2c failed" would not.
        assert_ne!(I2cError::Nak.as_str(), I2cError::Timeout.as_str());
        assert_ne!(I2cError::Timeout.as_str(), I2cError::ReadTimeout.as_str());
    }
}
