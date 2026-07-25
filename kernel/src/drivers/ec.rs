//! **ACPI embedded controller** — the microcontroller a laptop hides its battery,
//! lid, thermal and keyboard-backlight state behind.
//!
//! There is no way to read a battery on a real laptop without this. The AML in the
//! DSDT declares an `OperationRegion` in the `EmbeddedControl` address space and
//! reads fields out of it; every one of those reads is a two-port handshake with a
//! separate microcontroller. So `_BST` is only answerable once this driver works.
//!
//! ## Shape
//!
//! Three layers, in the order that keeps the logic testable:
//!
//! 1. **Pure protocol** — status-bit predicates and the transaction state machine,
//!    with no port access anywhere in them. Driven in tests by a simulated
//!    controller (including a deliberately rude one) via [`EcIo`].
//! 2. **Transport** ([`EcTransport`]) — the arch-specific register access. x86 uses
//!    I/O ports; the memory-mapped form (what ACPI's reduced-hardware profile
//!    describes, and the only form reachable on aarch64) uses single-byte MMIO.
//! 3. **Discovery** ([`init`]) — find `PNP0C09` in the namespace, read its `_CRS`,
//!    and take the first two declared registers as data and command.
//!
//! ## Bounded, always
//!
//! Every wait is a bounded spin. An absent or wedged controller must cost a fixed
//! number of iterations and then report failure — the alternative is a boot that
//! hangs on a laptop whose EC is behind a different port pair, which is exactly the
//! failure mode the HPET probe already taught us to design out. A floating bus reads
//! `0xff` (every status bit set, including two that cannot be true together), so a
//! liveness check rejects that before any command is issued.
//!
//! **Unverified on hardware.** QEMU emulates no ACPI EC. The protocol is from ACPI
//! 6.5 §12 and Linux's `drivers/acpi/ec.c`; the failure paths all log, so a first
//! run on a real laptop should say which step gave up.

use crate::acpi;
use crate::aml;
use alloc::string::String;
use alloc::vec::Vec;

// --- Status register bits (read from the command/status port) -------------
/// Output buffer full: a byte is waiting for us in the data register.
pub const STS_OBF: u8 = 1 << 0;
/// Input buffer full: the controller has not yet consumed our last write.
pub const STS_IBF: u8 = 1 << 1;
/// The byte in the data register is being interpreted as a command.
pub const STS_CMD: u8 = 1 << 3;
/// Burst mode active.
pub const STS_BURST: u8 = 1 << 4;
/// An SCI event is pending; `QUERY` will name it.
pub const STS_SCI_EVT: u8 = 1 << 5;
/// An SMI event is pending (firmware's, not ours).
pub const STS_SMI_EVT: u8 = 1 << 6;

// --- Commands ------------------------------------------------------------
/// Read a byte from the EC address space.
pub const CMD_READ: u8 = 0x80;
/// Write a byte to the EC address space.
pub const CMD_WRITE: u8 = 0x81;
/// Enter burst mode.
pub const CMD_BURST_ENABLE: u8 = 0x82;
/// Leave burst mode.
pub const CMD_BURST_DISABLE: u8 = 0x83;
/// Ask which event fired.
pub const CMD_QUERY: u8 = 0x84;

/// The data register at its spec-defined fixed address.
pub const DEFAULT_DATA_PORT: u16 = 0x62;
/// The command/status register at its spec-defined fixed address.
pub const DEFAULT_CMD_PORT: u16 = 0x66;

/// ACPI hardware ID of the embedded controller device.
pub const EC_HID: &str = "PNP0C09";

/// True when the controller has consumed our last write and will accept another.
pub fn input_ready(status: u8) -> bool {
    status & STS_IBF == 0
}

/// True when a byte is waiting in the data register.
pub fn output_ready(status: u8) -> bool {
    status & STS_OBF != 0
}

/// True when an event is pending, which `CMD_QUERY` will identify.
pub fn event_pending(status: u8) -> bool {
    status & STS_SCI_EVT != 0
}

/// True when the status byte cannot have come from a real controller.
///
/// `0xff` is an unclaimed x86 I/O port (and an unmapped MMIO read). It is also
/// self-contradictory as a status: OBF and IBF both set means the controller is
/// simultaneously waiting for us to read and waiting to consume a write, and the
/// reserved bit 2 is set on top. Treating it as a live status is how a machine with
/// no EC at the probed address ends up reporting a battery.
pub fn status_is_implausible(status: u8) -> bool {
    status == 0xff
}

/// Byte-level access to the two EC registers.
///
/// Split out as a trait so the transaction sequences below can be exercised against
/// a simulated controller — the sequences are where the bugs live, and they must not
/// need hardware to test.
pub trait EcIo {
    /// Read the command/status register.
    fn status(&mut self) -> u8;
    /// Write the command register.
    fn write_cmd(&mut self, v: u8);
    /// Read the data register.
    fn read_data(&mut self) -> u8;
    /// Write the data register.
    fn write_data(&mut self, v: u8);
}

/// Iterations a single handshake wait may spin before giving up.
///
/// A live controller answers in microseconds. This is large enough that a slow one
/// still succeeds and small enough that a full read against dead ports costs
/// milliseconds, not a hung boot.
const MAX_SPIN: u32 = 200_000;

/// Why a transaction failed. Distinct variants because they point at different
/// causes on a real machine, and "it didn't work" is not a useful ktrace line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EcError {
    /// The status register never showed the controller accepting our write.
    InputTimeout,
    /// The controller never produced the byte we asked for.
    OutputTimeout,
    /// The status register read back as something no controller would report.
    NotPresent,
}

/// An embedded controller reached through some [`EcIo`].
pub struct Ec<T: EcIo> {
    io: T,
}

impl<T: EcIo> Ec<T> {
    /// Wrap a transport.
    pub fn new(io: T) -> Self {
        Self { io }
    }

    /// The transport back, for callers that own the device.
    pub fn io_mut(&mut self) -> &mut T {
        &mut self.io
    }

    /// Spin until the controller will accept an input byte.
    fn wait_input(&mut self) -> Result<(), EcError> {
        for _ in 0..MAX_SPIN {
            let st = self.io.status();
            if status_is_implausible(st) {
                return Err(EcError::NotPresent);
            }
            if input_ready(st) {
                return Ok(());
            }
            core::hint::spin_loop();
        }
        Err(EcError::InputTimeout)
    }

    /// Spin until a byte is waiting to be read.
    fn wait_output(&mut self) -> Result<(), EcError> {
        for _ in 0..MAX_SPIN {
            let st = self.io.status();
            if status_is_implausible(st) {
                return Err(EcError::NotPresent);
            }
            if output_ready(st) {
                return Ok(());
            }
            core::hint::spin_loop();
        }
        Err(EcError::OutputTimeout)
    }

    /// Discard a byte the controller left in the data register.
    ///
    /// Firmware hands over with an unread byte often enough that Linux does this
    /// before every transaction. Skipping it makes the *next* read return the
    /// *previous* value — an off-by-one-transaction error that looks like a
    /// plausible-but-wrong battery reading rather than a failure.
    fn drain_stale(&mut self) {
        let st = self.io.status();
        if !status_is_implausible(st) && output_ready(st) {
            let _ = self.io.read_data();
        }
    }

    /// Check the controller is there at all, without issuing a command.
    ///
    /// Read-only on purpose: writing a command to whatever device actually owns a
    /// wrongly-guessed port pair is not an acceptable way to find out it is not an
    /// EC.
    pub fn probe(&mut self) -> Result<(), EcError> {
        let st = self.io.status();
        if status_is_implausible(st) {
            return Err(EcError::NotPresent);
        }
        // A controller that never clears IBF is wedged; say so now rather than at the
        // first battery read.
        self.wait_input()
    }

    /// Read one byte from EC address `offset`.
    pub fn read(&mut self, offset: u8) -> Result<u8, EcError> {
        self.drain_stale();
        self.wait_input()?;
        self.io.write_cmd(CMD_READ);
        self.wait_input()?;
        self.io.write_data(offset);
        self.wait_output()?;
        Ok(self.io.read_data())
    }

    /// Write one byte to EC address `offset`.
    pub fn write(&mut self, offset: u8, value: u8) -> Result<(), EcError> {
        self.wait_input()?;
        self.io.write_cmd(CMD_WRITE);
        self.wait_input()?;
        self.io.write_data(offset);
        self.wait_input()?;
        self.io.write_data(value);
        // The controller acknowledges by consuming the value; without waiting for
        // that, a following read races the write.
        self.wait_input()
    }

    /// Read a run of bytes starting at `offset`.
    ///
    /// One transaction per byte, as the spec defines it — burst mode would be faster
    /// but holds the controller hostage, and nothing here is on a hot path.
    pub fn read_bytes(&mut self, offset: u8, out: &mut [u8]) -> Result<(), EcError> {
        for (i, slot) in out.iter_mut().enumerate() {
            let off = offset.checked_add(i as u8).ok_or(EcError::InputTimeout)?;
            *slot = self.read(off)?;
        }
        Ok(())
    }

    /// Ask which event fired. Returns `0` when the controller has nothing to report.
    pub fn query(&mut self) -> Result<u8, EcError> {
        self.wait_input()?;
        self.io.write_cmd(CMD_QUERY);
        self.wait_output()?;
        Ok(self.io.read_data())
    }
}

/// How the two EC registers are actually reached on this machine.
///
/// Both forms exist in ACPI: the fixed/legacy profile uses I/O ports, the
/// reduced-hardware profile a memory-mapped pair. Which one a machine declares is a
/// firmware fact, not an arch fact — so the enum carries both and the arch only
/// decides which arms it can perform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EcTransport {
    /// x86 I/O ports.
    Port { data: u16, cmd: u16 },
    /// Memory-mapped registers.
    Mmio { data: u64, cmd: u64 },
}

impl EcIo for EcTransport {
    fn status(&mut self) -> u8 {
        match *self {
            EcTransport::Port { cmd, .. } => port_in(cmd),
            EcTransport::Mmio { cmd, .. } => mmio_in(cmd),
        }
    }

    fn write_cmd(&mut self, v: u8) {
        match *self {
            EcTransport::Port { cmd, .. } => port_out(cmd, v),
            EcTransport::Mmio { cmd, .. } => mmio_out(cmd, v),
        }
    }

    fn read_data(&mut self) -> u8 {
        match *self {
            EcTransport::Port { data, .. } => port_in(data),
            EcTransport::Mmio { data, .. } => mmio_in(data),
        }
    }

    fn write_data(&mut self, v: u8) {
        match *self {
            EcTransport::Port { data, .. } => port_out(data, v),
            EcTransport::Mmio { data, .. } => mmio_out(data, v),
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn port_in(port: u16) -> u8 {
    // SAFETY: reading an ACPI-declared EC status/data port. A read of an x86 I/O port
    // has no side effect on memory; an unclaimed port returns 0xff, which
    // `status_is_implausible` rejects.
    unsafe { crate::arch::x86_64::port::inb(port) }
}

#[cfg(target_arch = "x86_64")]
fn port_out(port: u16, v: u8) {
    // SAFETY: writing an ACPI-declared EC register, only after `probe` established a
    // plausible status there.
    unsafe { crate::arch::x86_64::port::outb(port, v) };
}

// aarch64 has no port I/O instruction. A firmware that describes its EC with
// SystemIO resources is therefore unreachable here, and the honest answer is to
// refuse rather than to poke an address that means something else — `init` reports
// exactly that.
#[cfg(not(target_arch = "x86_64"))]
fn port_in(_port: u16) -> u8 {
    0xff
}

#[cfg(not(target_arch = "x86_64"))]
fn port_out(_port: u16, _v: u8) {}

fn mmio_in(addr: u64) -> u8 {
    // SAFETY: single-byte read of an ACPI-declared, identity-mapped device register.
    // A byte access needs no alignment care, and a single `ldrb` is what the aarch64
    // MMIO rule requires — nothing here can be coalesced into a paired load.
    unsafe { core::ptr::read_volatile(addr as *const u8) }
}

fn mmio_out(addr: u64, v: u8) {
    // SAFETY: as `mmio_in`, and only reached after `probe` found a plausible status.
    unsafe { core::ptr::write_volatile(addr as *mut u8, v) };
}

/// The system's embedded controller, once found.
static mut EC: Option<EcTransport> = None;

/// Read the two EC registers out of a `PNP0C09` device's `_CRS`.
///
/// Pure so the interesting part — that the *first* declared register is data and the
/// second is command, and that a single-register or empty `_CRS` is a refusal rather
/// than a guess — is testable.
pub fn transport_from_crs(crs: &[u8]) -> Option<EcTransport> {
    let r = acpi::parse_resources(crs);
    if r.io.len() >= 2 {
        return Some(EcTransport::Port {
            data: r.io[0].base,
            cmd: r.io[1].base,
        });
    }
    if r.mem.len() >= 2 {
        return Some(EcTransport::Mmio {
            data: r.mem[0].base as u64,
            cmd: r.mem[1].base as u64,
        });
    }
    None
}

/// Find the embedded controller and prove it answers.
///
/// Namespace first: `PNP0C09`'s `_CRS` is authoritative and is what a machine with a
/// non-standard port pair needs. Only if the namespace yields nothing does x86 fall
/// back to the spec's fixed `0x62`/`0x66` — and even then the controller has to pass
/// [`Ec::probe`] before it is recorded, so a machine without one is left with no EC
/// rather than a phantom.
///
/// Returns the transport that was accepted, or `None` with a ktrace line naming the
/// step that failed.
pub fn init(rsdp: u64) -> Option<EcTransport> {
    let mut candidates: Vec<(EcTransport, String)> = Vec::new();

    if let Some(dsdt) = acpi::dsdt_bytes(rsdp, crate::mm::map_mmio) {
        if let Some(dev) = aml::device_by_hid(dsdt, EC_HID) {
            match aml::device_name(dsdt, &dev, "_CRS") {
                Some(aml::Value::Buffer(b)) => match transport_from_crs(&b) {
                    Some(t) => candidates.push((t, alloc::format!("{} _CRS", dev.path))),
                    None => crate::ktrace::log_fmt(format_args!(
                        "ec: {} _CRS declares fewer than two registers; ignoring",
                        dev.path
                    )),
                },
                _ => crate::ktrace::log_fmt(format_args!("ec: {} has no _CRS buffer", dev.path)),
            }
        } else {
            crate::ktrace::log_fmt(format_args!("ec: no {} device in the namespace", EC_HID));
        }
    } else {
        crate::ktrace::log_fmt(format_args!("ec: no DSDT"));
    }

    if candidates.is_empty() && cfg!(target_arch = "x86_64") {
        candidates.push((
            EcTransport::Port {
                data: DEFAULT_DATA_PORT,
                cmd: DEFAULT_CMD_PORT,
            },
            String::from("fixed 0x62/0x66"),
        ));
    }

    for (t, how) in candidates {
        if let EcTransport::Port { .. } = t {
            if !cfg!(target_arch = "x86_64") {
                crate::ktrace::log_fmt(format_args!("ec: {} needs port I/O, unavailable on this arch", how));
                continue;
            }
        }
        let mut ec = Ec::new(t);
        match ec.probe() {
            Ok(()) => {
                crate::ktrace::log_fmt(format_args!("ec: present via {}", how));
                // SAFETY: single-threaded boot-time initialisation, before any task
                // can call `read`.
                unsafe { EC = Some(t) };
                return Some(t);
            }
            Err(e) => crate::ktrace::log_fmt(format_args!("ec: {} did not answer ({:?})", how, e)),
        }
    }
    None
}

/// True once [`init`] has accepted a controller.
pub fn present() -> bool {
    // SAFETY: read of a boot-initialised `Option<EcTransport>`; `EcTransport` is Copy
    // and no writer runs after `init`.
    unsafe { EC.is_some() }
}

/// Read one byte from the system EC, or `None` if there is none.
pub fn read(offset: u8) -> Option<u8> {
    // SAFETY: as `present`.
    let t = unsafe { EC }?;
    Ec::new(t).read(offset).ok()
}

/// Read a run of bytes from the system EC.
pub fn read_bytes(offset: u8, out: &mut [u8]) -> Option<()> {
    // SAFETY: as `present`.
    let t = unsafe { EC }?;
    Ec::new(t).read_bytes(offset, out).ok()
}

/// Write one byte to the system EC.
pub fn write(offset: u8, value: u8) -> Option<()> {
    // SAFETY: as `present`.
    let t = unsafe { EC }?;
    Ec::new(t).write(offset, value).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// A simulated controller with a byte-addressed store, driven by the same
    /// two-register protocol as the real thing.
    struct FakeEc {
        space: [u8; 256],
        status: u8,
        out: Option<u8>,
        pending_cmd: Option<u8>,
        /// Bytes written to the data register since the command.
        args: vec::Vec<u8>,
        /// Iterations of IBF-set to make the host wait through, per write.
        busy: u32,
        busy_left: u32,
    }

    impl FakeEc {
        fn new() -> Self {
            let mut space = [0u8; 256];
            for (i, b) in space.iter_mut().enumerate() {
                *b = i as u8 ^ 0x5a;
            }
            Self {
                space,
                status: 0,
                out: None,
                pending_cmd: None,
                args: vec::Vec::new(),
                busy: 0,
                busy_left: 0,
            }
        }

        /// Make every write take `n` status reads to be consumed.
        fn slow(mut self, n: u32) -> Self {
            self.busy = n;
            self
        }

        fn accept(&mut self) {
            self.status &= !STS_IBF;
            match self.pending_cmd {
                Some(CMD_READ) if self.args.len() == 1 => {
                    self.out = Some(self.space[self.args[0] as usize]);
                    self.status |= STS_OBF;
                    self.pending_cmd = None;
                    self.args.clear();
                }
                Some(CMD_WRITE) if self.args.len() == 2 => {
                    self.space[self.args[0] as usize] = self.args[1];
                    self.pending_cmd = None;
                    self.args.clear();
                }
                Some(CMD_QUERY) => {
                    self.out = Some(0x11);
                    self.status |= STS_OBF;
                    self.pending_cmd = None;
                    self.args.clear();
                }
                _ => {}
            }
        }
    }

    impl EcIo for FakeEc {
        fn status(&mut self) -> u8 {
            if self.status & STS_IBF != 0 {
                if self.busy_left > 0 {
                    self.busy_left -= 1;
                } else {
                    self.accept();
                }
            }
            self.status
        }
        fn write_cmd(&mut self, v: u8) {
            self.pending_cmd = Some(v);
            self.args.clear();
            self.status |= STS_IBF | STS_CMD;
            self.busy_left = self.busy;
        }
        fn read_data(&mut self) -> u8 {
            self.status &= !STS_OBF;
            self.out.take().unwrap_or(0)
        }
        fn write_data(&mut self, v: u8) {
            self.args.push(v);
            self.status |= STS_IBF;
            self.busy_left = self.busy;
        }
    }

    #[test_case]
    fn status_predicates_match_the_bit_definitions() {
        assert!(input_ready(0));
        assert!(!input_ready(STS_IBF));
        assert!(output_ready(STS_OBF));
        assert!(!output_ready(0));
        assert!(event_pending(STS_SCI_EVT));
        assert!(!event_pending(STS_OBF | STS_IBF));
    }

    #[test_case]
    fn a_floating_bus_is_not_a_controller() {
        // 0xff is an unclaimed port. It also claims OBF and IBF at once, which no
        // real controller reports — accepting it invents a battery on a desktop.
        assert!(status_is_implausible(0xff));
        assert!(!status_is_implausible(0));
        assert!(!status_is_implausible(STS_OBF | STS_IBF | STS_CMD));

        struct Dead;
        impl EcIo for Dead {
            fn status(&mut self) -> u8 {
                0xff
            }
            fn write_cmd(&mut self, _v: u8) {
                panic!("wrote a command to a dead bus");
            }
            fn read_data(&mut self) -> u8 {
                0xff
            }
            fn write_data(&mut self, _v: u8) {
                panic!("wrote data to a dead bus");
            }
        }
        let mut ec = Ec::new(Dead);
        assert_eq!(ec.probe(), Err(EcError::NotPresent));
        assert_eq!(ec.read(0x00), Err(EcError::NotPresent));
    }

    #[test_case]
    fn reads_and_writes_complete_against_a_simulated_controller() {
        let mut ec = Ec::new(FakeEc::new());
        ec.probe().unwrap();
        assert_eq!(ec.read(0x00), Ok(0x5a));
        assert_eq!(ec.read(0x0f), Ok(0x0f ^ 0x5a));
        ec.write(0x20, 0xa5).unwrap();
        assert_eq!(ec.read(0x20), Ok(0xa5));
        assert_eq!(ec.query(), Ok(0x11));
    }

    #[test_case]
    fn a_slow_controller_still_completes() {
        // The handshake must tolerate a controller that takes its time consuming
        // writes, not just one that answers on the first status read.
        let mut ec = Ec::new(FakeEc::new().slow(500));
        assert_eq!(ec.read(0x07), Ok(0x07 ^ 0x5a));
    }

    #[test_case]
    fn a_stale_output_byte_does_not_shift_every_later_read() {
        // Firmware commonly hands over with an unread byte in the data register. If
        // it is not drained, the first read returns *it* and every subsequent read
        // returns the previous transaction's value — wrong-but-plausible numbers.
        let mut fake = FakeEc::new();
        fake.out = Some(0xde);
        fake.status |= STS_OBF;
        let mut ec = Ec::new(fake);
        assert_eq!(ec.read(0x01), Ok(0x01 ^ 0x5a), "stale byte was not drained");
        assert_eq!(ec.read(0x02), Ok(0x02 ^ 0x5a));
    }

    #[test_case]
    fn a_wedged_controller_times_out_instead_of_spinning_forever() {
        // IBF stuck set: the controller never consumes anything. The wait is bounded,
        // so this returns rather than hanging the boot.
        struct Wedged;
        impl EcIo for Wedged {
            fn status(&mut self) -> u8 {
                STS_IBF
            }
            fn write_cmd(&mut self, _v: u8) {}
            fn read_data(&mut self) -> u8 {
                0
            }
            fn write_data(&mut self, _v: u8) {}
        }
        let mut ec = Ec::new(Wedged);
        assert_eq!(ec.probe(), Err(EcError::InputTimeout));
        assert_eq!(ec.read(0), Err(EcError::InputTimeout));

        // IBF clears but a byte never arrives: a different bound, a different error.
        struct Mute;
        impl EcIo for Mute {
            fn status(&mut self) -> u8 {
                0
            }
            fn write_cmd(&mut self, _v: u8) {}
            fn read_data(&mut self) -> u8 {
                0
            }
            fn write_data(&mut self, _v: u8) {}
        }
        let mut ec = Ec::new(Mute);
        assert_eq!(ec.read(0), Err(EcError::OutputTimeout));
    }

    #[test_case]
    fn read_bytes_walks_consecutive_offsets_and_refuses_to_wrap() {
        let mut ec = Ec::new(FakeEc::new());
        let mut buf = [0u8; 4];
        ec.read_bytes(0x10, &mut buf).unwrap();
        assert_eq!(buf, [0x10 ^ 0x5a, 0x11 ^ 0x5a, 0x12 ^ 0x5a, 0x13 ^ 0x5a]);

        // The EC address space is 8-bit; a run that would wrap past 0xff must fail
        // rather than silently re-read from the bottom.
        let mut over = [0u8; 4];
        assert!(ec.read_bytes(0xfe, &mut over).is_err());
    }

    #[test_case]
    fn transport_comes_from_the_crs_in_declaration_order() {
        // Data register first, command second — swapping them makes every status read
        // return data and every transaction fail.
        let crs = vec![
            0x47, 0x01, 0x62, 0x00, 0x62, 0x00, 0x01, 0x01, // IO(0x62, 1)
            0x47, 0x01, 0x66, 0x00, 0x66, 0x00, 0x01, 0x01, // IO(0x66, 1)
            0x79, 0x00,
        ];
        assert_eq!(
            transport_from_crs(&crs),
            Some(EcTransport::Port {
                data: 0x62,
                cmd: 0x66
            })
        );

        // A machine whose EC is at a non-standard pair is exactly why _CRS wins over
        // the fixed addresses.
        let odd = vec![
            0x47, 0x01, 0x30, 0x09, 0x30, 0x09, 0x01, 0x01, // IO(0x930, 1)
            0x47, 0x01, 0x34, 0x09, 0x34, 0x09, 0x01, 0x01, // IO(0x934, 1)
            0x79, 0x00,
        ];
        assert_eq!(
            transport_from_crs(&odd),
            Some(EcTransport::Port {
                data: 0x930,
                cmd: 0x934
            })
        );
    }

    #[test_case]
    fn a_crs_without_two_registers_is_refused_not_guessed() {
        // One register, or none, means we do not know where the command port is.
        // Filling in a guess would write a command to an unknown device.
        let one = vec![0x47, 0x01, 0x62, 0x00, 0x62, 0x00, 0x01, 0x01, 0x79, 0x00];
        assert_eq!(transport_from_crs(&one), None);
        assert_eq!(transport_from_crs(&[0x79, 0x00]), None);
        assert_eq!(transport_from_crs(&[]), None);
    }

    #[test_case]
    fn memory_mapped_registers_are_accepted_when_that_is_what_firmware_declares() {
        // The reduced-hardware profile describes the same controller with memory
        // ranges; both forms have to work from one API.
        let mut crs = vec![0x86, 0x09, 0x00, 0x01];
        crs.extend_from_slice(&0x4000_0000u32.to_le_bytes());
        crs.extend_from_slice(&1u32.to_le_bytes());
        crs.extend_from_slice(&[0x86, 0x09, 0x00, 0x01]);
        crs.extend_from_slice(&0x4000_0004u32.to_le_bytes());
        crs.extend_from_slice(&1u32.to_le_bytes());
        crs.extend_from_slice(&[0x79, 0x00]);
        assert_eq!(
            transport_from_crs(&crs),
            Some(EcTransport::Mmio {
                data: 0x4000_0000,
                cmd: 0x4000_0004
            })
        );
    }
}
