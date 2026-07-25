//! **ACPI control-method battery** — a real percentage, from the firmware's own AML.
//!
//! This is the payoff of the AML work. A laptop's battery is not a register anywhere:
//! firmware exposes it as `_BST` (current state) and `_BIF`/`_BIX` (static
//! information) methods on a `PNP0C0A` device, whose bodies read named fields that
//! alias an `OperationRegion` — almost always in the `EmbeddedControl` space, i.e. a
//! two-port handshake with a separate microcontroller. So a percentage needs, in
//! order: the namespace walk, the evaluator, `OperationRegion`/`Field`, and the EC
//! driver. All four now exist, and this is the layer that composes them.
//!
//! ## Where the numbers come from
//!
//! `_BST` returns a package of four: state flags, present rate, **remaining
//! capacity**, present voltage. `_BIF` (or `_BIX`) gives **last full charge
//! capacity**. The percentage is remaining ÷ last-full, and it is deliberately taken
//! against *last full charge* rather than *design* capacity — that is what every
//! other OS reports, and on a worn battery the design figure reads permanently below
//! 100%.
//!
//! ## Fail closed, everywhere
//!
//! Every step can legitimately be unavailable — a desktop has no battery, a machine
//! may describe its EC at ports we cannot reach, `_BST` may use AML this subset does
//! not implement. Each of those yields `None`, never a guess: the status bar showing
//! nothing is correct, and showing an invented "100%" is not. ACPI's own
//! "unknown" sentinel (`0xffff_ffff`) is honoured as unknown rather than treated as a
//! four-billion-mWh capacity.
//!
//! **Unverified on hardware.** Nothing in this environment has an ACPI battery. The
//! pure arithmetic and packaging is unit-tested against the shapes the specification
//! defines, and every failure path logs which step gave up.

use crate::aml::{self, FieldUnit, Value};
use alloc::string::String;

/// ACPI hardware ID of a control-method battery.
pub const BATTERY_HID: &str = "PNP0C0A";

/// ACPI's "this value is unknown" sentinel for a 32-bit capacity or rate.
pub const UNKNOWN: u64 = 0xffff_ffff;

// `_BST` state flags.
/// The battery is discharging.
pub const BST_DISCHARGING: u64 = 1 << 0;
/// The battery is charging.
pub const BST_CHARGING: u64 = 1 << 1;
/// The battery is in a critical energy state.
pub const BST_CRITICAL: u64 = 1 << 2;

/// A decoded `_BST` result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Bst {
    /// Raw state flags.
    pub state: u64,
    /// Present rate, or [`UNKNOWN`].
    pub rate: u64,
    /// Remaining capacity, or [`UNKNOWN`].
    pub remaining: u64,
    /// Present voltage, or [`UNKNOWN`].
    pub voltage: u64,
}

impl Bst {
    /// True while the battery is being charged.
    pub fn charging(&self) -> bool {
        self.state & BST_CHARGING != 0
    }

    /// True while the battery is supplying the machine.
    pub fn discharging(&self) -> bool {
        self.state & BST_DISCHARGING != 0
    }

    /// True when firmware has flagged a critical energy level.
    pub fn critical(&self) -> bool {
        self.state & BST_CRITICAL != 0
    }
}

/// What a caller — the status bar — needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatteryState {
    /// Charge as a percentage of last-full capacity, 0–100.
    pub percent: u8,
    /// True while charging.
    pub charging: bool,
    /// True while running on the battery.
    pub discharging: bool,
    /// True when firmware says the level is critical.
    pub critical: bool,
}

/// Decode a `_BST` return value.
///
/// Requires all four elements: a shorter package means the method returned something
/// this does not understand, and reading three of four values would silently take the
/// voltage slot as a capacity.
pub fn parse_bst(v: &Value) -> Option<Bst> {
    let items = match v {
        Value::Package(p) => p,
        _ => return None,
    };
    if items.len() < 4 {
        return None;
    }
    let n = |k: usize| -> Option<u64> {
        match items.get(k)? {
            Value::Integer(i) => Some(*i),
            _ => None,
        }
    };
    Some(Bst {
        state: n(0)?,
        rate: n(1)?,
        remaining: n(2)?,
        voltage: n(3)?,
    })
}

/// Pull last-full-charge capacity out of a `_BIF` or `_BIX` return value.
///
/// The two packages differ by one leading element — `_BIX` starts with a revision —
/// so the field sits at index 2 in `_BIF` and index 3 in `_BIX`. They are told apart
/// by length rather than by which method was called, so a firmware that returns the
/// wrong one for the name is still read correctly: `_BIF` has 13 elements, `_BIX` at
/// least 20.
pub fn full_capacity(v: &Value) -> Option<u64> {
    let items = match v {
        Value::Package(p) => p,
        _ => return None,
    };
    let idx = if items.len() >= 20 {
        3
    } else if items.len() >= 13 {
        2
    } else {
        return None;
    };
    match items.get(idx)? {
        // An unknown last-full capacity makes a percentage impossible, not zero.
        Value::Integer(c) if *c != 0 && *c != UNKNOWN => Some(*c),
        _ => None,
    }
}

/// Combine a `_BST` reading with a last-full capacity into a percentage.
///
/// Clamped to 100: a freshly-calibrated battery genuinely reports a remaining
/// capacity above its last-full figure, and "104%" in a status bar reads as a bug.
pub fn percent(bst: &Bst, full: u64) -> Option<u8> {
    if full == 0 || full == UNKNOWN || bst.remaining == UNKNOWN {
        return None;
    }
    // u64 maths on values that are milliwatt-hours: no overflow, no float.
    let p = bst.remaining.saturating_mul(100) / full;
    Some(p.min(100) as u8)
}

/// Read a field unit's value by fetching the bytes it spans.
///
/// Pure, and separated from any address space on purpose — the bit arithmetic is
/// where a field read goes subtly wrong, and it must be testable without hardware.
/// `read_byte` is given an **absolute** address in the field's own address space.
///
/// Handles unaligned and sub-byte fields by fetching every byte the field touches and
/// shifting/masking, rather than assuming byte alignment: a 4-bit field at bit 12 must
/// not return its neighbours' bits.
pub fn read_field(
    f: &FieldUnit,
    region_offset: u64,
    read_byte: &dyn Fn(u64) -> Option<u8>,
) -> Option<u64> {
    // A wider-than-64-bit field is a buffer field, not an integer; refuse rather than
    // truncate it into a plausible-looking number.
    if f.bit_width == 0 || f.bit_width > 64 {
        return None;
    }
    let shift = f.bit_offset % 8;
    let first = region_offset.checked_add(f.bit_offset / 8)?;
    let nbytes = ((shift + f.bit_width) as usize + 7) / 8;
    // u128 because shift + width can reach 71 bits before the shift back down.
    let mut raw: u128 = 0;
    for k in 0..nbytes {
        let b = read_byte(first.checked_add(k as u64)?)? as u128;
        raw |= b << (8 * k);
    }
    let v = raw >> shift;
    let mask: u128 = if f.bit_width == 64 {
        u64::MAX as u128
    } else {
        (1u128 << f.bit_width) - 1
    };
    Some((v & mask) as u64)
}

/// Read a byte from whichever address space a region names.
///
/// Only the spaces a battery actually uses are implemented. `PciConfig` and `SMBus`
/// regions return `None` — a wrong byte from the wrong space would produce a
/// confident nonsense reading, which is the failure mode worth avoiding most.
fn read_space_byte(space: u8, addr: u64) -> Option<u8> {
    match space {
        aml::SPACE_EMBEDDED_CONTROL => {
            // The EC address space is 8-bit; an address past it is a table bug, not a
            // wrap-around read.
            if addr > 0xff {
                return None;
            }
            super::ec::read(addr as u8)
        }
        aml::SPACE_SYSTEM_MEMORY => {
            let va = crate::mm::map_mmio(addr & !0xfff, 0x1000);
            if va == 0 {
                return None;
            }
            // SAFETY: single-byte read of an ACPI-declared region that `map_mmio` just
            // mapped; a byte access has no alignment requirement.
            Some(unsafe { core::ptr::read_volatile((va + (addr & 0xfff)) as *const u8) })
        }
        #[cfg(target_arch = "x86_64")]
        aml::SPACE_SYSTEM_IO => {
            if addr > 0xffff {
                return None;
            }
            // SAFETY: reading an ACPI-declared SystemIO register. Reads of x86 I/O
            // ports have no memory effect.
            Some(unsafe { crate::arch::x86_64::port::inb(addr as u16) })
        }
        _ => None,
    }
}

/// Evaluate one method on the battery device, resolving its field reads against
/// hardware.
fn eval(dsdt: &'static [u8], dev: &aml::DeviceNode, method: &str) -> Option<Value> {
    let resolve = |name: &str| -> Option<u64> {
        let (unit, region) = aml::find_field(dsdt, name)?;
        read_field(&unit, region.offset, &|addr| {
            read_space_byte(region.space, addr)
        })
    };
    aml::eval_device_method_with_fields(dsdt, dev, method, &[], &resolve)
}

/// Read the machine's battery, if it has one that answers.
///
/// Returns `None` on a desktop, on a machine whose `_BST` uses AML beyond this
/// subset, and whenever any single reading is unavailable. A ktrace line names the
/// step that gave up, so a first run on real hardware is diagnosable.
pub fn read() -> Option<BatteryState> {
    #[cfg(target_arch = "x86_64")]
    let rsdp = crate::arch::x86_64::rsdp_address()?;
    #[cfg(target_arch = "aarch64")]
    let rsdp = crate::arch::aarch64::rsdp_address()?;
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let rsdp = return None;

    let dsdt = crate::acpi::dsdt_bytes(rsdp, crate::mm::map_mmio)?;
    let dev = aml::device_by_hid(dsdt, BATTERY_HID)?;

    let bst = match eval(dsdt, &dev, "_BST").as_ref().and_then(parse_bst) {
        Some(b) => b,
        None => {
            crate::ktrace::log_fmt(format_args!(
                "battery: {} _BST did not evaluate to a 4-element package",
                dev.path
            ));
            return None;
        }
    };
    // `_BIX` first: it is the newer method, and a machine providing both should be
    // read through the one with more information.
    let full = eval(dsdt, &dev, "_BIX")
        .as_ref()
        .and_then(full_capacity)
        .or_else(|| eval(dsdt, &dev, "_BIF").as_ref().and_then(full_capacity));
    let Some(full) = full else {
        crate::ktrace::log_fmt(format_args!(
            "battery: {} gave no usable last-full capacity (_BIX/_BIF)",
            dev.path
        ));
        return None;
    };
    let percent = percent(&bst, full)?;
    Some(BatteryState {
        percent,
        charging: bst.charging(),
        discharging: bst.discharging(),
        critical: bst.critical(),
    })
}

/// How long a reading is reused before another `_BST` evaluation.
///
/// Every read is an AML evaluation plus a handful of EC transactions, each a bounded
/// spin on a slow microcontroller. The status bar repaints many times a second; doing
/// this per frame would spend the machine's time watching a battery.
const REFRESH_MS: u64 = 5_000;

/// Attempts allowed before concluding the machine has no readable battery.
///
/// A desktop has none and never will, so retrying forever is pure waste. More than one
/// attempt because an EC can legitimately be busy during early boot.
const MAX_ATTEMPTS: u32 = 3;

/// Delay between failed attempts.
const RETRY_MS: u64 = 10_000;

/// Cached state: the last reading, when it was taken, and how many attempts failed.
static mut CACHE: (Option<BatteryState>, u64, u32) = (None, 0, 0);

/// The battery state, refreshed at most every [`REFRESH_MS`].
///
/// Safe to call from the status-bar repaint path. Returns `None` on a machine with no
/// readable battery — and after [`MAX_ATTEMPTS`] failures stops trying, so a desktop
/// pays for three DSDT walks in total rather than one per repaint.
pub fn cached() -> Option<BatteryState> {
    let now = crate::arch::now_ms();
    // SAFETY: the cache is only touched from the single-threaded UI/status path. A
    // torn read would at worst reuse a stale percentage for one frame.
    let (last, at, fails) = unsafe { CACHE };
    if fails >= MAX_ATTEMPTS {
        return None;
    }
    let due = match last {
        Some(_) => now.saturating_sub(at) >= REFRESH_MS,
        // Not yet succeeded: back off harder, since the likely answer is "no battery".
        None => at == 0 || now.saturating_sub(at) >= RETRY_MS,
    };
    if !due {
        return last;
    }
    match read() {
        Some(b) => {
            // SAFETY: as above.
            unsafe { CACHE = (Some(b), now.max(1), 0) };
            Some(b)
        }
        None => {
            // SAFETY: as above.
            unsafe { CACHE = (None, now.max(1), fails + 1) };
            None
        }
    }
}

/// Format a battery state for the status bar.
///
/// Pure so the rendering is testable: a charging battery is marked, a critical one is
/// marked differently, and the percentage never gains a leading space that would
/// shift the rest of the bar.
pub fn format(b: &BatteryState) -> String {
    let mark = if b.charging {
        "+"
    } else if b.critical {
        "!"
    } else {
        ""
    };
    alloc::format!("{}{}%", mark, b.percent)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn pkg(v: &[u64]) -> Value {
        Value::Package(v.iter().map(|x| Value::Integer(*x)).collect())
    }

    #[test_case]
    fn decodes_a_bst_package() {
        let v = pkg(&[BST_DISCHARGING, 1500, 3200, 11000]);
        let b = parse_bst(&v).unwrap();
        assert_eq!(b.remaining, 3200);
        assert_eq!(b.voltage, 11000);
        assert!(b.discharging() && !b.charging() && !b.critical());

        let c = parse_bst(&pkg(&[BST_CHARGING | BST_CRITICAL, 0, 100, 11000])).unwrap();
        assert!(c.charging() && c.critical());
    }

    #[test_case]
    fn a_short_bst_package_is_refused_not_partially_read() {
        // Reading three of four elements would take the voltage slot as a capacity.
        assert!(parse_bst(&pkg(&[0, 0, 3200])).is_none());
        assert!(parse_bst(&Value::Integer(0)).is_none());
        assert!(parse_bst(&Value::Package(vec![])).is_none());
    }

    #[test_case]
    fn full_capacity_comes_from_the_right_slot_for_bif_and_bix() {
        // _BIF: 13 elements, last-full at index 2. _BIX: 20+, at index 3 (a revision
        // shifts everything). Using one index for both reads the design capacity as
        // the last-full figure, which reads as a permanently-below-100% battery.
        let mut bif = vec![0u64; 13];
        bif[1] = 5000; // design capacity
        bif[2] = 4200; // last full charge
        assert_eq!(full_capacity(&pkg(&bif)), Some(4200));

        let mut bix = vec![0u64; 20];
        bix[2] = 5000;
        bix[3] = 4200;
        assert_eq!(full_capacity(&pkg(&bix)), Some(4200));
    }

    #[test_case]
    fn an_unknown_or_zero_capacity_yields_no_percentage() {
        let mut bif = vec![0u64; 13];
        bif[2] = UNKNOWN;
        assert_eq!(full_capacity(&pkg(&bif)), None);
        bif[2] = 0;
        assert_eq!(full_capacity(&pkg(&bif)), None);

        let bst = Bst {
            remaining: UNKNOWN,
            ..Default::default()
        };
        assert_eq!(percent(&bst, 4200), None);
        assert_eq!(percent(&Bst { remaining: 100, ..Default::default() }, 0), None);
    }

    #[test_case]
    fn percentage_is_against_last_full_and_clamped() {
        let half = Bst {
            remaining: 2100,
            ..Default::default()
        };
        assert_eq!(percent(&half, 4200), Some(50));
        // A just-calibrated battery reports above its last-full figure; 104% is a bug
        // to a reader, so clamp.
        let over = Bst {
            remaining: 4400,
            ..Default::default()
        };
        assert_eq!(percent(&over, 4200), Some(100));
        assert_eq!(percent(&Bst { remaining: 0, ..Default::default() }, 4200), Some(0));
    }

    #[test_case]
    fn a_byte_aligned_field_reads_little_endian() {
        // Battery capacities are 16-bit fields split across two EC bytes; getting the
        // byte order wrong turns 3200 mWh into 32780.
        let f = FieldUnit {
            name: String::from("BRC0"),
            region: String::from("ECR"),
            bit_offset: 16,
            bit_width: 16,
        };
        let mem = |a: u64| -> Option<u8> { [0u8, 0, 0x80, 0x0c, 0].get(a as usize).copied() };
        assert_eq!(read_field(&f, 0, &mem), Some(0x0c80));
    }

    #[test_case]
    fn a_sub_byte_field_masks_out_its_neighbours() {
        // 4 bits at bit 12 of 0xab, 0xcd -> the 'c' nibble, not the whole byte.
        let f = FieldUnit {
            name: String::from("NIBL"),
            region: String::from("ECR"),
            bit_offset: 12,
            bit_width: 4,
        };
        let mem = |a: u64| -> Option<u8> { [0xabu8, 0xcd].get(a as usize).copied() };
        assert_eq!(read_field(&f, 0, &mem), Some(0xc));

        // A single flag bit, and one that spans a byte boundary.
        let flag = FieldUnit {
            name: String::from("FLAG"),
            region: String::from("ECR"),
            bit_offset: 7,
            bit_width: 1,
        };
        assert_eq!(read_field(&flag, 0, &mem), Some(1));
        let span = FieldUnit {
            name: String::from("SPAN"),
            region: String::from("ECR"),
            bit_offset: 4,
            bit_width: 8,
        };
        assert_eq!(read_field(&span, 0, &mem), Some(0xda));
    }

    #[test_case]
    fn the_region_offset_is_added_to_the_field_offset() {
        // A region based at 0x20 with a field at bit 8 reads absolute byte 0x21.
        let f = FieldUnit {
            name: String::from("BYTE"),
            region: String::from("ECR"),
            bit_offset: 8,
            bit_width: 8,
        };
        let seen = core::cell::Cell::new(0u64);
        let mem = |a: u64| -> Option<u8> {
            seen.set(a);
            Some(0x5a)
        };
        assert_eq!(read_field(&f, 0x20, &mem), Some(0x5a));
        assert_eq!(seen.get(), 0x21);
    }

    #[test_case]
    fn an_unreadable_byte_fails_the_whole_field() {
        // Half a capacity is worse than none: if the EC does not answer for one byte
        // of a 16-bit field, the field has no value.
        let f = FieldUnit {
            name: String::from("BRC0"),
            region: String::from("ECR"),
            bit_offset: 0,
            bit_width: 16,
        };
        let half = |a: u64| -> Option<u8> { if a == 0 { Some(0x80) } else { None } };
        assert_eq!(read_field(&f, 0, &half), None);
    }

    #[test_case]
    fn oversized_and_empty_fields_are_refused() {
        let mem = |_: u64| Some(0xff);
        let wide = FieldUnit {
            name: String::from("BUFF"),
            region: String::from("ECR"),
            bit_offset: 0,
            bit_width: 128,
        };
        assert_eq!(read_field(&wide, 0, &mem), None);
        let empty = FieldUnit {
            name: String::from("NONE"),
            region: String::from("ECR"),
            bit_offset: 0,
            bit_width: 0,
        };
        assert_eq!(read_field(&empty, 0, &mem), None);

        // A full 64-bit field is the widest that is still an integer.
        let full = FieldUnit {
            name: String::from("WIDE"),
            region: String::from("ECR"),
            bit_offset: 0,
            bit_width: 64,
        };
        assert_eq!(read_field(&full, 0, &mem), Some(u64::MAX));
    }

    #[test_case]
    fn status_text_marks_charging_and_critical() {
        let base = BatteryState {
            percent: 42,
            charging: false,
            discharging: true,
            critical: false,
        };
        assert_eq!(format(&base), "42%");
        assert_eq!(
            format(&BatteryState {
                charging: true,
                ..base
            }),
            "+42%"
        );
        assert_eq!(
            format(&BatteryState {
                critical: true,
                percent: 4,
                ..base
            }),
            "!4%"
        );
        // Charging wins over critical: it is the more actionable fact.
        assert_eq!(
            format(&BatteryState {
                charging: true,
                critical: true,
                percent: 4,
                ..base
            }),
            "+4%"
        );
    }
}
