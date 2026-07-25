//! **HID over I2C** — the transport a modern laptop touchpad uses.
//!
//! Sits on [`super::i2c`] (the DesignWare/LPSS master) and feeds
//! [`crate::mouse`], so a touchpad ends up in the same place as a USB or PS/2
//! pointer and the rest of the UI needs no changes.
//!
//! ## Why this is not simply "read the reports"
//!
//! An I2C-HID device is not enumerable. Three facts have to come from elsewhere:
//!
//! * its **I2C address** — from `_CRS` ([`crate::acpi::i2c_resources`]);
//! * the **HID descriptor register** — from `_DSM`, which needs AML evaluation. It
//!   is `0x0020` on the large majority of devices, so that is the default, and the
//!   descriptor read is validated (see [`HidDesc::parse`]) so a wrong guess is
//!   *detected* rather than silently producing garbage;
//! * the **report layout** — from the HID report descriptor, parsed by the existing
//!   [`crate::xhci::parse_report_layout`], which is already used for USB pointers.
//!   Reusing it means precision touchpads and simple mice decode through one
//!   code path.
//!
//! ## Verification status
//!
//! The descriptor parsing and register framing are unit-tested. The bus interaction
//! is **unverified on hardware** — QEMU has no LPSS I2C controller — so `probe`
//! validates aggressively and refuses rather than guessing: a device whose HID
//! descriptor does not look like one is left alone, which matters because the same
//! bus can host the embedded controller.

use super::i2c::DwI2c;
use crate::block::BlockError;

/// Default HID descriptor register. `_DSM` is authoritative, but this is the value
/// on the large majority of devices and is what Linux falls back to.
pub const DEFAULT_HID_DESC_REG: u16 = 0x0020;

/// The ACPI id every HID-over-I2C device reports in `_HID` or `_CID`.
pub const HID_I2C_PNP_ID: &str = "PNP0C50";

/// Length of the HID descriptor, per the HID-over-I2C specification.
pub const HID_DESC_LEN: usize = 30;
/// The `bcdVersion` a conforming device reports.
pub const HID_BCD_V1: u16 = 0x0100;

/// The fields of the HID descriptor this driver needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HidDesc {
    pub desc_len: u16,
    pub bcd_version: u16,
    pub report_desc_len: u16,
    pub report_desc_reg: u16,
    pub input_reg: u16,
    pub max_input_len: u16,
    pub command_reg: u16,
    pub data_reg: u16,
}

impl HidDesc {
    /// Parse and **validate** a HID descriptor read.
    ///
    /// Validation is the point, not a formality: the descriptor register address is a
    /// guess when `_DSM` cannot be evaluated, so this is the check that tells a real
    /// I2C-HID device from an arbitrary register on some other chip. A device that
    /// does not report the specified length and version is refused.
    pub fn parse(d: &[u8]) -> Option<HidDesc> {
        if d.len() < HID_DESC_LEN {
            return None;
        }
        let u16at = |o: usize| u16::from_le_bytes([d[o], d[o + 1]]);
        let desc_len = u16at(0);
        let bcd_version = u16at(2);
        // A conforming device reports exactly this length and version 1.00. Anything
        // else means we are not looking at a HID descriptor.
        if desc_len as usize != HID_DESC_LEN || bcd_version != HID_BCD_V1 {
            return None;
        }
        let report_desc_len = u16at(4);
        let max_input_len = u16at(10);
        // A report descriptor of zero length, or an input report that cannot even
        // hold its own 2-byte length prefix, is not usable.
        if report_desc_len == 0 || max_input_len < 2 {
            return None;
        }
        Some(HidDesc {
            desc_len,
            bcd_version,
            report_desc_len,
            report_desc_reg: u16at(6),
            input_reg: u16at(8),
            max_input_len,
            command_reg: u16at(22),
            data_reg: u16at(24),
        })
    }
}

/// Split an input report into `(report_id, payload)`.
///
/// HID-over-I2C prefixes every input report with a **2-byte little-endian length that
/// includes the prefix itself**. Getting that wrong is the classic bug: treating the
/// length as payload-only overruns, and ignoring it entirely reads stale FIFO bytes as
/// report data.
///
/// A length of 0 is the device's documented way of saying "no report pending" (it
/// signals this after a reset), and is not an error.
pub fn split_input_report(buf: &[u8]) -> Option<&[u8]> {
    if buf.len() < 2 {
        return None;
    }
    let len = u16::from_le_bytes([buf[0], buf[1]]) as usize;
    if len == 0 {
        return None; // nothing pending
    }
    if len < 2 || len > buf.len() {
        return None; // malformed: shorter than its own prefix, or beyond what we read
    }
    Some(&buf[2..len])
}

/// A HID-over-I2C device on a bus.
pub struct I2cHid {
    bus: DwI2c,
    addr: u16,
    desc: HidDesc,
    /// Report layout decoded from the HID report descriptor, shared with the USB
    /// pointer path.
    layout: crate::xhci::PtrLayout,
    /// Budget for logging the first few reports, so a mis-decoded layout is
    /// diagnosable on hardware this cannot be tested against.
    dbg: u8,
}

impl I2cHid {
    /// Read a 16-bit register and return `len` bytes from it.
    fn read_reg(bus: &mut DwI2c, addr: u16, reg: u16, out: &mut [u8]) -> Result<(), BlockError> {
        bus.write_read(addr, &reg.to_le_bytes(), out)
    }

    /// Try to bring up a HID-over-I2C device at `addr`.
    ///
    /// Returns `None` for anything that does not answer as one. This is deliberately
    /// the only way in: the caller may be walking addresses from `_CRS` that belong to
    /// sensors or the embedded controller, and nothing here writes to the device
    /// beyond the register *reads* needed to identify it.
    pub fn probe(mut bus: DwI2c, addr: u16, desc_reg: u16) -> Option<I2cHid> {
        let mut raw = [0u8; HID_DESC_LEN];
        if Self::read_reg(&mut bus, addr, desc_reg, &mut raw).is_err() {
            return None; // nothing there, or it NAKed
        }
        let desc = HidDesc::parse(&raw)?;
        crate::ktrace::log_fmt(format_args!(
            "i2c_hid: device at {addr:#x}: report desc {} bytes at reg {:#x}, input reg {:#x}, max input {}",
            desc.report_desc_len, desc.report_desc_reg, desc.input_reg, desc.max_input_len
        ));
        // Fetch and decode the report descriptor with the same parser the USB pointer
        // path uses, so both decode identically.
        let mut rd = alloc::vec![0u8; desc.report_desc_len as usize];
        if Self::read_reg(&mut bus, addr, desc.report_desc_reg, &mut rd).is_err() {
            crate::ktrace::log("i2c_hid", "report descriptor read failed");
            return None;
        }
        // SAFETY: `rd` is a live buffer of exactly the length passed.
        let layout = unsafe { crate::xhci::parse_report_layout(rd.as_ptr() as usize, rd.len()) }?;
        crate::ktrace::log("i2c_hid", "report layout decoded; touchpad ready");
        Some(I2cHid { bus, addr, desc, layout, dbg: 5 })
    }

    /// Poll once for an input report and feed [`crate::mouse`].
    ///
    /// Non-blocking in the sense that matters: a device with nothing to say answers a
    /// zero length, which costs one short transfer and no waiting.
    pub fn poll(&mut self) {
        let n = self.desc.max_input_len as usize;
        let mut buf = alloc::vec![0u8; n.min(64)];
        // The input register is read *without* a register write — the device presents
        // the pending report directly.
        if self.bus.write_read(self.addr, &[], &mut buf).is_err() {
            return;
        }
        let Some(report) = split_input_report(&buf) else {
            return;
        };
        crate::xhci::feed_pointer_report(&self.layout, report, Some(&mut self.dbg));
    }

    pub fn address(&self) -> u16 {
        self.addr
    }
}

// --- boot bring-up --------------------------------------------------------

/// The touchpad, once found.
static HID: crate::mm::Locked<Option<I2cHid>> = crate::mm::Locked::new(None);

/// Find and bring up a HID-over-I2C pointer, if this machine has one.
///
/// Walks the I2C connections the DSDT declares ([`crate::acpi::i2c_resources`]) and
/// asks each whether it answers as HID. Nothing is written to any address during
/// identification — see [`I2cHid::probe`] — because those addresses may belong to the
/// embedded controller or a sensor, and a laptop is not a safe place to guess.
///
/// A no-op where there is no DesignWare controller (every non-Intel machine, and
/// QEMU), so it costs one PCI scan and returns.
pub fn init(rsdp: u64) {
    if HID.with(|h| h.is_some()) {
        return;
    }
    let Some(aml) = crate::acpi::dsdt_bytes(rsdp, crate::mm::map_mmio) else {
        return;
    };
    // Ask the namespace which device claims to be a HID-over-I2C touchpad and read
    // *its* `_CRS`, rather than trying every I2C connection in the table. That is the
    // difference the AML walk buys: no addressing of devices that are not the
    // touchpad, on a bus that also carries the embedded controller.
    let mut conns = alloc::vec::Vec::new();
    if let Some(dev) = crate::aml::device_by_hid(aml, HID_I2C_PNP_ID) {
        if let Some(crs) = crate::aml::device_name(aml, &dev, "_CRS") {
            if let Some(r) = crs.as_buffer().and_then(crate::acpi::parse_i2c_serial_bus) {
                crate::ktrace::log_fmt(format_args!(
                    "i2c_hid: {} declares {} at {:#x} ({} Hz)",
                    dev.path, HID_I2C_PNP_ID, r.address, r.speed_hz
                ));
                conns.push(r);
            }
        }
    }
    if conns.is_empty() {
        // No device claims the touchpad ID (or its `_CRS` is not a static buffer), so
        // fall back to every declared I2C connection. Identification is read-only, so
        // this is safe — just less targeted.
        conns = crate::acpi::i2c_resources(aml.as_ptr());
        if conns.is_empty() {
            return;
        }
        crate::ktrace::log("i2c_hid", "no PNP0C50 device in the namespace; trying every declared I2C connection");
    }
    // Try each controller in turn: a machine can have several I2C buses, and the
    // touchpad is on exactly one of them.
    let mut ctrl = 0usize;
    while let Some(mut bus) = super::i2c::DwI2c::probe_nth(ctrl) {
        ctrl += 1;
        for c in conns.iter().filter(|c| !c.controller_mode) {
            // Cheap zero-length check first, so a full descriptor read is only
            // attempted where something actually answers.
            if !bus.present(c.address) {
                continue;
            }
            match I2cHid::probe(bus, c.address, DEFAULT_HID_DESC_REG) {
                Some(dev) => {
                    crate::ktrace::log_fmt(format_args!(
                        "i2c_hid: touchpad at {:#x} on controller {}",
                        dev.address(),
                        ctrl - 1
                    ));
                    HID.with(|h| *h = Some(dev));
                    return;
                }
                // `probe` consumes the bus, so re-acquire it to keep looking. The
                // controller is stateless between transfers, so re-probing is safe.
                None => match super::i2c::DwI2c::probe_nth(ctrl - 1) {
                    Some(b) => bus = b,
                    None => return,
                },
            }
        }
    }
    if ctrl > 0 {
        crate::ktrace::log("i2c_hid", "no HID-over-I2C device answered on any I2C bus");
    }
}

/// Whether a HID-over-I2C pointer was found.
pub fn present() -> bool {
    HID.with(|h| h.is_some())
}

/// Drain any pending touchpad report. Pumped from `arch::mouse_poll` alongside the
/// USB and PS/2 pointers.
pub fn poll() {
    HID.with(|h| {
        if let Some(d) = h.as_mut() {
            d.poll();
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// A conforming HID descriptor.
    fn desc_bytes() -> vec::Vec<u8> {
        let mut d = vec![0u8; HID_DESC_LEN];
        d[0..2].copy_from_slice(&(HID_DESC_LEN as u16).to_le_bytes());
        d[2..4].copy_from_slice(&HID_BCD_V1.to_le_bytes());
        d[4..6].copy_from_slice(&180u16.to_le_bytes()); // report desc len
        d[6..8].copy_from_slice(&0x0021u16.to_le_bytes()); // report desc reg
        d[8..10].copy_from_slice(&0x0022u16.to_le_bytes()); // input reg
        d[10..12].copy_from_slice(&64u16.to_le_bytes()); // max input len
        d[22..24].copy_from_slice(&0x0023u16.to_le_bytes()); // command reg
        d[24..26].copy_from_slice(&0x0024u16.to_le_bytes()); // data reg
        d
    }

    #[test_case]
    fn parses_a_conforming_hid_descriptor() {
        let d = HidDesc::parse(&desc_bytes()).unwrap();
        assert_eq!(d.report_desc_len, 180);
        assert_eq!(d.report_desc_reg, 0x0021);
        assert_eq!(d.input_reg, 0x0022);
        assert_eq!(d.max_input_len, 64);
        assert_eq!(d.command_reg, 0x0023);
        assert_eq!(d.data_reg, 0x0024);
    }

    #[test_case]
    fn refuses_anything_that_is_not_a_hid_descriptor() {
        // This validation is what distinguishes a real device from an arbitrary
        // register on another chip, since the descriptor register is a guess when
        // _DSM cannot be evaluated. It must be strict.
        let mut wrong_len = desc_bytes();
        wrong_len[0] = 0x1f; // claims 31 bytes
        assert_eq!(HidDesc::parse(&wrong_len), None);
        let mut wrong_ver = desc_bytes();
        wrong_ver[2..4].copy_from_slice(&0x0200u16.to_le_bytes());
        assert_eq!(HidDesc::parse(&wrong_ver), None);
        // All zeroes: what a register that is not a HID descriptor often reads as.
        assert_eq!(HidDesc::parse(&[0u8; HID_DESC_LEN]), None);
        // 0xFF everywhere: what an unclaimed/NAKing bus often reads as.
        assert_eq!(HidDesc::parse(&[0xffu8; HID_DESC_LEN]), None);
        // Short read.
        assert_eq!(HidDesc::parse(&desc_bytes()[..HID_DESC_LEN - 1]), None);
    }

    #[test_case]
    fn refuses_unusable_report_geometry() {
        // A zero-length report descriptor, or an input report too small to hold its
        // own length prefix, cannot be driven.
        let mut no_rd = desc_bytes();
        no_rd[4..6].copy_from_slice(&0u16.to_le_bytes());
        assert_eq!(HidDesc::parse(&no_rd), None);
        let mut tiny_in = desc_bytes();
        tiny_in[10..12].copy_from_slice(&1u16.to_le_bytes());
        assert_eq!(HidDesc::parse(&tiny_in), None);
    }

    #[test_case]
    fn input_report_length_prefix_includes_itself() {
        // The 2-byte prefix counts itself. Treating it as payload-only would overrun;
        // ignoring it would hand stale bytes to the decoder.
        let mut buf = vec![0u8; 8];
        buf[0..2].copy_from_slice(&6u16.to_le_bytes()); // total 6 => 4 payload bytes
        buf[2..6].copy_from_slice(&[0xa, 0xb, 0xc, 0xd]);
        assert_eq!(split_input_report(&buf), Some(&[0xa, 0xb, 0xc, 0xd][..]));
    }

    #[test_case]
    fn zero_length_means_no_report_not_an_error() {
        // A device signals "nothing pending" with a zero length, notably after reset.
        let buf = vec![0u8; 8];
        assert_eq!(split_input_report(&buf), None);
    }

    #[test_case]
    fn malformed_report_lengths_are_refused() {
        // Shorter than its own prefix.
        let mut b1 = vec![0u8; 8];
        b1[0..2].copy_from_slice(&1u16.to_le_bytes());
        assert_eq!(split_input_report(&b1), None);
        // Longer than what was actually read — must not slice past the buffer.
        let mut b2 = vec![0u8; 8];
        b2[0..2].copy_from_slice(&64u16.to_le_bytes());
        assert_eq!(split_input_report(&b2), None);
        // Too short to even hold a prefix.
        assert_eq!(split_input_report(&[0x06]), None);
    }
}
