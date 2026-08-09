//! **QEMU fw_cfg on x86** — reading the launcher's `opt/chitti/*` files.
//!
//! The aarch64 side has had this since ramfb (`arch::aarch64::ramfb`), and x86
//! had nothing — so `xtask` wrote the hosted-model seed, passed
//! `-fw_cfg name=opt/chitti/model,file=…`, and the kernel silently ignored it.
//! `shell::remote::boot_seed` was `#[cfg(target_arch = "aarch64")]` with an
//! `None` arm for everything else, which is precisely the divergence the
//! dual-architecture rule exists to prevent: the capability existed on one arch
//! and not the other, and nothing reported it — `/model` just said `local`.
//!
//! ## Ports, not MMIO, and no DMA
//!
//! On `virt` the interface is MMIO with a DMA descriptor; on a PC it is two I/O
//! ports, and the plain data port is enough here:
//!
//! - **0x510** — selector, a 16-bit write.
//! - **0x511** — data, read a byte at a time; the selector auto-advances.
//!
//! The DMA interface (0x514) exists on x86 too and would be faster, but the
//! files this reads are a few hundred bytes and DMA adds a physical-address
//! round trip plus a completion spin for no gain. The aarch64 side uses DMA
//! because its non-DMA path is far slower, not because it is required.
//!
//! ## The signature check is load-bearing
//!
//! Ports 0x510/0x511 are not claimed by anything on a machine without fw_cfg,
//! so they float. Reading the directory off a floating bus yields a huge entry
//! count and then a walk over garbage. Selector 0x0000 returns the ASCII
//! signature `"QEMU"`, and nothing proceeds until that matches.

use super::port::{inb, outw};
use alloc::vec;
use alloc::vec::Vec;

const PORT_SELECTOR: u16 = 0x510;
const PORT_DATA: u16 = 0x511;

/// `FW_CFG_SIGNATURE` — reads the four bytes `QEMU`.
const SEL_SIGNATURE: u16 = 0x0000;
/// `FW_CFG_FILE_DIR`.
const SEL_FILE_DIR: u16 = 0x0019;

/// A directory entry is a fixed 64 bytes: size, select, reserved, name[56].
const ENTRY_LEN: usize = 64;
const NAME_LEN: usize = 56;

/// Largest file this will read. The seeds are JSON of a few hundred bytes; a
/// bound keeps a garbage size from allocating wildly on a machine with no
/// fw_cfg that got past the signature check.
const MAX_FILE: u32 = 64 * 1024;

/// Select `key` and read `n` bytes from the data port.
///
/// # Safety
/// Touches fw_cfg I/O ports; call from single-threaded boot or shell init with
/// no other fw_cfg access in flight — the selector is global state.
unsafe fn read_sel(key: u16, n: usize) -> Vec<u8> {
    // SAFETY: the caller guarantees exclusive fw_cfg use.
    unsafe {
        outw(PORT_SELECTOR, key);
        let mut out = vec![0u8; n];
        for b in out.iter_mut() {
            *b = inb(PORT_DATA);
        }
        out
    }
}

/// Whether a fw_cfg interface is actually present.
///
/// # Safety
/// As [`read_sel`].
unsafe fn present() -> bool {
    // SAFETY: forwarded.
    unsafe { read_sel(SEL_SIGNATURE, 4) == b"QEMU" }
}

/// Find `name` in the file directory, returning `(selector, size)`.
///
/// # Safety
/// As [`read_sel`].
unsafe fn find_file(name: &[u8]) -> Option<(u16, u32)> {
    // SAFETY: forwarded; the signature check has already run.
    unsafe {
        outw(PORT_SELECTOR, SEL_FILE_DIR);
        // The count is **big-endian**, like every multi-byte field in the
        // fw_cfg directory — little-endian here reads 0x01000000 entries and
        // walks off the end of the world.
        let mut cnt = [0u8; 4];
        for b in cnt.iter_mut() {
            *b = inb(PORT_DATA);
        }
        let count = u32::from_be_bytes(cnt);
        // A machine with no fw_cfg that somehow passed the signature check, or
        // a corrupt read: refuse rather than walk.
        if count > 4096 {
            return None;
        }
        for _ in 0..count {
            let mut e = [0u8; ENTRY_LEN];
            for b in e.iter_mut() {
                *b = inb(PORT_DATA);
            }
            let size = u32::from_be_bytes([e[0], e[1], e[2], e[3]]);
            let select = u16::from_be_bytes([e[4], e[5]]);
            // The name is NUL-padded to 56 bytes.
            let raw = &e[8..8 + NAME_LEN];
            let end = raw.iter().position(|&c| c == 0).unwrap_or(NAME_LEN);
            if &raw[..end] == name {
                return Some((select, size));
            }
        }
        None
    }
}

/// Read a launcher-supplied `opt/chitti/*` fw_cfg file.
///
/// `None` when there is no fw_cfg interface (not QEMU) or the launcher did not
/// publish that file — both ordinary, neither an error.
pub fn read_opt_file(name: &[u8]) -> Option<Vec<u8>> {
    // SAFETY: called from shell init, single-threaded, no concurrent fw_cfg.
    unsafe {
        if !present() {
            return None;
        }
        let (selector, size) = find_file(name)?;
        if size == 0 || size > MAX_FILE {
            return None;
        }
        Some(read_sel(selector, size as usize))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The directory's multi-byte fields are **big-endian**. Read
    /// little-endian, a count of 1 becomes 16,777,216 and the walk reads far
    /// past the directory — so this pins the decode rather than the I/O.
    #[test_case]
    fn directory_fields_are_big_endian() {
        assert_eq!(u32::from_be_bytes([0, 0, 0, 1]), 1);
        assert_ne!(u32::from_le_bytes([0, 0, 0, 1]), 1, "the wrong reading is huge");
        assert_eq!(u32::from_le_bytes([0, 0, 0, 1]), 16_777_216);
        // A selector likewise.
        assert_eq!(u16::from_be_bytes([0x00, 0x19]), SEL_FILE_DIR);
    }

    /// An entry is 64 bytes with a NUL-padded 56-byte name, and the name is
    /// compared to its NUL, not to the whole field — otherwise nothing ever
    /// matches.
    #[test_case]
    fn an_entry_name_ends_at_its_nul() {
        let mut e = [0u8; ENTRY_LEN];
        e[8..8 + b"opt/chitti/model".len()].copy_from_slice(b"opt/chitti/model");
        let raw = &e[8..8 + NAME_LEN];
        let end = raw.iter().position(|&c| c == 0).unwrap_or(NAME_LEN);
        assert_eq!(&raw[..end], b"opt/chitti/model");
        assert_ne!(raw, b"opt/chitti/model".as_slice(), "the field is padded");
        assert_eq!(ENTRY_LEN, 8 + NAME_LEN);
    }

    /// The size bound keeps a garbage directory from allocating wildly on a
    /// machine whose 0x510/0x511 are floating.
    #[test_case]
    fn an_absurd_file_size_is_refused() {
        assert!(MAX_FILE < u32::MAX / 2);
        assert!(MAX_FILE >= 4096, "the seeds are only a few hundred bytes");
    }
}
