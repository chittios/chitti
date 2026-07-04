//! MC146818 CMOS real-time clock (I/O ports 0x70/0x71). Read once at boot to
//! seed the wall clock ([`crate::clock`]). Values are BCD by default; we wait
//! out any update-in-progress, read twice for a stable snapshot, and convert to
//! a Unix timestamp via the shared civil-date math.

use crate::arch::x86_64::port::{inb, outb};

const ADDR: u16 = 0x70;
const DATA: u16 = 0x71;

fn cmos(reg: u8) -> u8 {
    // SAFETY: standard CMOS access — select the register (bit 7 keeps NMI
    // disabled during the pair), then read the data port.
    unsafe {
        outb(ADDR, reg | 0x80);
        inb(DATA)
    }
}

fn update_in_progress() -> bool {
    cmos(0x0A) & 0x80 != 0
}

fn bcd_to_bin(v: u8) -> u8 {
    (v & 0x0f) + ((v >> 4) * 10)
}

/// Read the CMOS RTC as a Unix timestamp (UTC). `None` if the clock never
/// leaves update-in-progress (absent/broken RTC).
pub fn read_unix() -> Option<u64> {
    let mut guard = 0;
    while update_in_progress() {
        guard += 1;
        if guard > 1_000_000 {
            return None;
        }
    }
    let read = || (cmos(0x00), cmos(0x02), cmos(0x04), cmos(0x07), cmos(0x08), cmos(0x09));
    // Read twice; retry if a rollover happened mid-read.
    let mut last = read();
    for _ in 0..8 {
        let cur = read();
        if cur == last {
            break;
        }
        last = cur;
    }
    let (mut sec, mut min, mut hour, mut day, mut month, mut year) = last;
    let status_b = cmos(0x0B);
    if status_b & 0x04 == 0 {
        // BCD mode (the common default): convert every field.
        sec = bcd_to_bin(sec);
        min = bcd_to_bin(min);
        // Preserve the 12h PM flag (bit 7) across the BCD conversion of the low bits.
        hour = (bcd_to_bin(hour & 0x7f)) | (hour & 0x80);
        day = bcd_to_bin(day);
        month = bcd_to_bin(month);
        year = bcd_to_bin(year);
    }
    if status_b & 0x02 == 0 && hour & 0x80 != 0 {
        // 12-hour mode, PM: 12→12, else +12.
        hour = ((hour & 0x7f) + 12) % 24;
    }
    let full_year = 2000 + year as i64; // CMOS year is 2-digit; assume 21st century.
    let unix = crate::clock::unix_from_civil(full_year, month as i64, day as i64, hour as i64, min as i64, sec as i64);
    if unix > 0 {
        Some(unix as u64)
    } else {
        None
    }
}
