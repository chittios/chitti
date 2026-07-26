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
//! * the **HID descriptor register** — from `_DSM`, which is now evaluated
//!   ([`descriptor_register`]). `0x0020` remains the fallback for a `_DSM` this AML
//!   subset cannot evaluate, since it is the value on the large majority of devices,
//!   and the descriptor read is validated (see [`HidDesc::parse`]) so a wrong
//!   register is *detected* rather than silently producing garbage;
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

// --- HID-over-I2C commands ------------------------------------------------
//
// A command is a write of the 2-byte command-register address followed by two command
// bytes: the low byte carries the report type/ID (and, for SET_POWER, the power state),
// the high byte the opcode. Nothing here reads a report — these are the writes that make
// a device start producing them at all.

/// `RESET` — the device re-initialises and then signals completion by presenting a
/// zero-length input report.
pub const HID_OP_RESET: u8 = 0x01;
/// `SET_POWER` — the power state travels in the command's low byte.
pub const HID_OP_SET_POWER: u8 = 0x08;

/// `SET_POWER` argument: fully on.
pub const HID_POWER_ON: u8 = 0x00;
/// `SET_POWER` argument: sleep. What a suspend should leave the touchpad in.
pub const HID_POWER_SLEEP: u8 = 0x01;

/// Encode one HID-over-I2C command as the bytes to write.
///
/// Register address little-endian, then the command's low byte (argument) and high byte
/// (opcode) — in that order, which is the part worth pinning: swapping them sends
/// `SET_POWER` as report-type 8 of opcode 0, which a device answers by doing nothing at
/// all rather than by NAKing.
pub fn encode_command(cmd_reg: u16, opcode: u8, arg: u8) -> [u8; 4] {
    let r = cmd_reg.to_le_bytes();
    [r[0], r[1], arg, opcode]
}

/// Spins to wait for a reset to complete. A device answers within milliseconds; this is
/// generous and finite, because a touchpad that never finishes resetting must not hang
/// the boot.
const RESET_SPINS: u32 = 200_000;

/// The ACPI id every HID-over-I2C device reports in `_HID` or `_CID`.
pub const HID_I2C_PNP_ID: &str = "PNP0C50";

/// The HID-over-I2C `_DSM` UUID, `3CDFF6F7-4267-4555-AD05-B30A3D8938DE`, in the
/// mixed-endian byte order ACPI buffers use (first three groups little-endian, the
/// rest big-endian). Getting that order wrong makes the `LEqual` against the table's
/// own buffer fail and the method return its "unsupported" branch — a silent
/// fallback to the default, not an error, which is why the order is spelled out here.
pub const HID_DSM_UUID: [u8; 16] = [
    0xf7, 0xf6, 0xdf, 0x3c, // 3CDFF6F7 (LE)
    0x67, 0x42, // 4267 (LE)
    0x55, 0x45, // 4555 (LE)
    0xad, 0x05, // AD05 (BE)
    0xb3, 0x0a, 0x3d, 0x89, 0x38, 0xde, // B30A3D8938DE (BE)
];

/// `_DSM` function index that returns the HID descriptor register address.
pub const HID_DSM_FN_DESC_REG: u64 = 1;

/// Ask the firmware where this device's HID descriptor register is.
///
/// `_DSM(uuid, revision, function, args)` with function 1 returns the register
/// address. Now that the evaluator exists this is answered rather than assumed — which
/// matters for the minority of devices that do not use `0x0020`, where the old default
/// meant the descriptor read failed validation and the touchpad was simply skipped.
///
/// Returns `None` when there is no `_DSM`, when it uses AML beyond this subset, or when
/// it answers with something that cannot be a register (zero, or past the 16-bit
/// register space). Each of those falls back to [`DEFAULT_HID_DESC_REG`], which is what
/// shipped before — so this can only improve on the previous behaviour, never regress
/// it.
pub fn descriptor_register(aml: &[u8], dev: &crate::aml::DeviceNode) -> Option<u16> {
    use crate::aml::Value;
    let v = crate::aml::eval_device_method(
        aml,
        dev,
        "_DSM",
        &[
            Value::Buffer(HID_DSM_UUID.to_vec()),
            Value::Integer(1), // revision
            Value::Integer(HID_DSM_FN_DESC_REG),
            Value::Package(alloc::vec::Vec::new()),
        ],
    )?;
    let reg = match v {
        Value::Integer(i) => i,
        _ => return None,
    };
    // Zero is `_DSM`'s "function not supported" answer, and a register cannot exceed
    // the 16-bit I2C register space.
    if reg == 0 || reg > u16::MAX as u64 {
        return None;
    }
    Some(reg as u16)
}

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

    /// Send one HID-over-I2C command.
    fn command(bus: &mut DwI2c, addr: u16, cmd_reg: u16, opcode: u8, arg: u8) -> Result<(), BlockError> {
        let bytes = encode_command(cmd_reg, opcode, arg);
        // A write with nothing to read: `write_read` puts STOP on the last data byte
        // when the read buffer is empty, which is exactly a write-only transfer.
        bus.write_read(addr, &bytes, &mut [])
    }

    /// Power the device on and reset it, so it starts producing input reports.
    ///
    /// **This is what was missing.** The HID descriptor is readable from a device that
    /// is still powered down — which is why the descriptor phase appeared to work — but
    /// no report ever arrives until `SET_POWER(ON)` and `RESET` have been sent. A
    /// touchpad without this reads perfectly and then does nothing, forever.
    ///
    /// Deliberately only reachable *after* [`HidDesc::parse`] has validated the
    /// descriptor. That ordering is the safety property the probe path depends on: the
    /// caller may be walking `_CRS` addresses that belong to the embedded controller or
    /// a sensor, and these are the first *writes* this driver makes to the device.
    fn power_on_and_reset(bus: &mut DwI2c, addr: u16, cmd_reg: u16, input_reg_len: usize) -> bool {
        if Self::command(bus, addr, cmd_reg, HID_OP_SET_POWER, HID_POWER_ON).is_err() {
            crate::ktrace::log("i2c_hid", "SET_POWER(ON) was not acknowledged");
            return false;
        }
        if Self::command(bus, addr, cmd_reg, HID_OP_RESET, 0).is_err() {
            crate::ktrace::log("i2c_hid", "RESET was not acknowledged");
            return false;
        }
        // A reset completes when the device presents a report — a zero-length one, which
        // is its documented way of saying "nothing pending". Either answer means it is
        // talking to us; only silence for the whole budget is a failure.
        let mut buf = alloc::vec![0u8; input_reg_len.min(64).max(2)];
        for _ in 0..RESET_SPINS {
            if bus.write_read(addr, &[], &mut buf).is_ok() {
                crate::ktrace::log("i2c_hid", "powered on and reset");
                return true;
            }
            core::hint::spin_loop();
        }
        crate::ktrace::log("i2c_hid", "device never answered after RESET");
        false
    }

    /// Put the device to sleep, for a suspend.
    pub fn sleep(&mut self) -> bool {
        let r = Self::command(&mut self.bus, self.addr, self.desc.command_reg, HID_OP_SET_POWER, HID_POWER_SLEEP);
        r.is_ok()
    }

    /// Power the device back on after a resume. The controller and the device both lose
    /// their state across S3, so this is the same sequence boot uses.
    pub fn resume(&mut self) -> bool {
        Self::power_on_and_reset(
            &mut self.bus,
            self.addr,
            self.desc.command_reg,
            self.desc.max_input_len as usize,
        )
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
        // Now that the descriptor has validated — so we know this is a HID device and
        // not the embedded controller sharing the bus — power it on and reset it. Until
        // this happens the device answers register reads but produces no reports.
        if !Self::power_on_and_reset(&mut bus, addr, desc.command_reg, desc.max_input_len as usize) {
            return None;
        }
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

/// Put the touchpad to sleep before a suspend, if there is one.
pub fn suspend() {
    HID.with(|h| {
        if let Some(d) = h.as_mut() {
            if !d.sleep() {
                crate::ktrace::log("i2c_hid", "SET_POWER(SLEEP) failed; suspending anyway");
            }
        }
    });
}

/// Power the touchpad back on after a resume.
///
/// The device loses its state across S3 exactly as the controller does, so this is the
/// same power-on-and-reset boot uses. Without it a resumed machine has a touchpad that
/// answers register reads and produces no reports — the same failure the boot path had
/// before the sequence existed.
pub fn resume() {
    HID.with(|h| {
        if let Some(d) = h.as_mut() {
            if d.resume() {
                crate::ktrace::log("i2c_hid", "touchpad powered back on after resume");
            } else {
                crate::ktrace::log("i2c_hid", "touchpad did not come back after resume");
            }
        }
    });
}

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
    let mut desc_reg = DEFAULT_HID_DESC_REG;
    if let Some(dev) = crate::aml::device_by_hid(aml, HID_I2C_PNP_ID) {
        if let Some(crs) = crate::aml::device_name(aml, &dev, "_CRS") {
            if let Some(r) = crs.as_buffer().and_then(crate::acpi::parse_i2c_serial_bus) {
                crate::ktrace::log_fmt(format_args!(
                    "i2c_hid: {} declares {} at {:#x} ({} Hz)",
                    dev.path, HID_I2C_PNP_ID, r.address, r.speed_hz
                ));
                conns.push(r);
            }
            // Ask `_DSM` for the descriptor register instead of assuming it. A device
            // that answers is read at the register firmware names; anything else keeps
            // the majority default.
            match descriptor_register(aml, &dev) {
                Some(reg) => {
                    crate::ktrace::log_fmt(format_args!(
                        "i2c_hid: {} _DSM gives descriptor register {:#x}",
                        dev.path, reg
                    ));
                    desc_reg = reg;
                }
                None => crate::ktrace::log_fmt(format_args!(
                    "i2c_hid: {} _DSM not evaluable; using the default register {:#x}",
                    dev.path, DEFAULT_HID_DESC_REG
                )),
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
            match I2cHid::probe(bus, c.address, desc_reg) {
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

    // --- _DSM: where the HID descriptor register comes from -----------------

    /// PkgLength for a body of `n` bytes (the encoding counts its own size).
    fn pkglen(n: usize) -> vec::Vec<u8> {
        let one = n + 1;
        if one <= 0x3f {
            return alloc::vec![one as u8];
        }
        let two = n + 2;
        alloc::vec![0x40 | (two & 0x0f) as u8, (two >> 4) as u8]
    }

    /// `Buffer (len) { bytes }`
    fn buffer(bytes: &[u8]) -> vec::Vec<u8> {
        let mut inner = alloc::vec![0x0au8, bytes.len() as u8]; // BytePrefix size
        inner.extend_from_slice(bytes);
        let mut v = alloc::vec![0x11u8]; // BufferOp
        v.extend_from_slice(&pkglen(inner.len()));
        v.extend_from_slice(&inner);
        v
    }

    /// `If (cond) { body }`
    fn if_block(cond: &[u8], body: &[u8]) -> vec::Vec<u8> {
        let mut inner = cond.to_vec();
        inner.extend_from_slice(body);
        let mut v = alloc::vec![0xa0u8]; // IfOp
        v.extend_from_slice(&pkglen(inner.len()));
        v.extend_from_slice(&inner);
        v
    }

    /// `Method (_DSM, 4) { body }`, wrapped in a `DeviceNode` spanning it.
    fn dsm_method(body: &[u8]) -> vec::Vec<u8> {
        let mut inner: vec::Vec<u8> = "_DSM".bytes().collect();
        inner.push(4); // 4 args
        inner.extend_from_slice(body);
        let mut v = alloc::vec![0x14u8]; // MethodOp
        v.extend_from_slice(&pkglen(inner.len()));
        v.extend_from_slice(&inner);
        v
    }

    fn node(len: usize) -> crate::aml::DeviceNode {
        crate::aml::DeviceNode {
            path: alloc::string::String::from("\\_SB.PCI0.I2C1.TPD0"),
            body: 0..len,
            extent: 0..len,
        }
    }

    /// The body every vendor writes: check the UUID, check the function, return the
    /// register.
    fn conforming_dsm(reg: u8) -> vec::Vec<u8> {
        // If (LEqual (Arg0, Buffer(16){uuid})) { If (LEqual (Arg2, One)) { Return (reg) } }
        let mut cond = alloc::vec![0x93u8, 0x68]; // LEqual, Arg0
        cond.extend_from_slice(&buffer(&HID_DSM_UUID));
        let inner_cond = alloc::vec![0x93u8, 0x6a, 0x01]; // LEqual, Arg2, One
        let ret = alloc::vec![0xa4u8, 0x0a, reg]; // Return (BytePrefix reg)
        let outer_body = if_block(&inner_cond, &ret);
        let mut body = if_block(&cond, &outer_body);
        body.extend_from_slice(&[0xa4, 0x00]); // Return (Zero)
        dsm_method(&body)
    }

    #[test_case]
    fn dsm_supplies_the_descriptor_register() {
        // The point of evaluating `_DSM` at all: a device that does not use 0x0020 was
        // previously read at the wrong register, failed descriptor validation, and was
        // skipped entirely.
        let aml = conforming_dsm(0x21);
        assert_eq!(descriptor_register(&aml, &node(aml.len())), Some(0x21));
    }

    #[test_case]
    fn the_uuid_byte_order_is_the_mixed_endian_one() {
        // ACPI buffers store the first three UUID groups little-endian. With the order
        // wrong the LEqual fails, `_DSM` takes its unsupported branch and returns zero,
        // and the result is a *silent* fallback rather than an error — so this is worth
        // pinning.
        assert_eq!(HID_DSM_UUID[0], 0xf7);
        assert_eq!(HID_DSM_UUID[3], 0x3c);
        assert_eq!(HID_DSM_UUID[15], 0xde);

        // A table whose UUID differs by one byte must not match.
        let mut wrong = HID_DSM_UUID;
        wrong[0] ^= 1;
        let mut cond = alloc::vec![0x93u8, 0x68];
        cond.extend_from_slice(&buffer(&wrong));
        let ret = alloc::vec![0xa4u8, 0x0a, 0x21];
        let mut body = if_block(&cond, &ret);
        body.extend_from_slice(&[0xa4, 0x00]); // Return (Zero)
        let aml = dsm_method(&body);
        assert_eq!(descriptor_register(&aml, &node(aml.len())), None);
    }

    #[test_case]
    fn an_unsupported_or_absent_dsm_falls_back_rather_than_failing() {
        // Zero is `_DSM`'s "function not supported" answer; it is not a register.
        let aml = dsm_method(&[0xa4, 0x00]); // Return (Zero)
        assert_eq!(descriptor_register(&aml, &node(aml.len())), None);

        // No `_DSM` at all.
        assert_eq!(descriptor_register(&[], &node(0)), None);

        // A register past the 16-bit register space cannot be one.
        let mut body = alloc::vec![0xa4u8, 0x0c]; // Return (DWordPrefix ...)
        body.extend_from_slice(&0x1_0000u32.to_le_bytes());
        let aml = dsm_method(&body);
        assert_eq!(descriptor_register(&aml, &node(aml.len())), None);
    }

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
    fn a_command_is_the_register_then_argument_then_opcode() {
        // Byte order is the whole encoding. Swapping the last two sends SET_POWER as
        // report-type 8 of opcode 0, which a device answers by doing nothing at all
        // rather than by NAKing — so the failure looks like a dead touchpad, not an
        // error.
        assert_eq!(
            encode_command(0x0023, HID_OP_SET_POWER, HID_POWER_ON),
            [0x23, 0x00, 0x00, 0x08]
        );
        assert_eq!(
            encode_command(0x0023, HID_OP_SET_POWER, HID_POWER_SLEEP),
            [0x23, 0x00, 0x01, 0x08]
        );
        assert_eq!(encode_command(0x0023, HID_OP_RESET, 0), [0x23, 0x00, 0x00, 0x01]);
    }

    #[test_case]
    fn the_command_register_address_is_little_endian() {
        // A two-byte register address, low byte first — the same convention every other
        // register access here uses, and a device NAKs a swapped one.
        assert_eq!(encode_command(0x1234, HID_OP_RESET, 0)[..2], [0x34, 0x12]);
        assert_eq!(encode_command(0x00ff, HID_OP_RESET, 0)[..2], [0xff, 0x00]);
    }

    #[test_case]
    fn the_reset_and_power_opcodes_are_the_spec_values() {
        // From the HID-over-I2C specification. Pinned because a wrong opcode is
        // indistinguishable at runtime from a device that simply does not respond.
        assert_eq!(HID_OP_RESET, 0x01);
        assert_eq!(HID_OP_SET_POWER, 0x08);
        assert_eq!(HID_POWER_ON, 0x00);
        assert_eq!(HID_POWER_SLEEP, 0x01);
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
