//! Where does a tenant blob's entry sit, without parsing ELF?
//!
//! CLAUDE.md is explicit: **no ELF loader**. A first attempt read the ELF64 header for
//! `e_entry`, which — even living in the host build tool rather than the kernel — is exactly
//! the format knowledge this OS does not want to acquire. It is also unnecessary.
//!
//! Two pieces of text already hold the answer:
//!
//! * `llvm-nm` prints `_start`'s address.
//! * the linker script says where the image was based, and *we wrote it*.
//!
//! Their difference is the byte offset of the entry within the flat binary. No structures, no
//! spec, and the base is read from the same file the linker used rather than duplicated as a
//! constant here — so it cannot drift out of step with the script.
//!
//! Why an offset at all: the loader used to jump to offset 0 and require the linker to put
//! `_start` there. On x86 it landed at `+0xc` while aarch64 was exact, so a tenant executed into
//! the middle of an unrelated function and faulted reading a kernel address — an arch-dependent
//! offset nobody chose, and an `ASSERT` in the script that failed to fire. Reading the number is
//! more robust than arranging it.

/// `_start`'s address, from `llvm-nm` output.
///
/// The format is `<hex address> <type> <name>`; a global text symbol is `T`, a local one `t`,
/// and both are accepted because the visibility of `_start` is not this function's business.
pub fn parse_nm_start(out: &str) -> Option<u64> {
    for line in out.lines() {
        let mut it = line.split_whitespace();
        let (addr, ty, name) = (it.next()?, it.next()?, it.next()?);
        if name == "_start" && (ty == "T" || ty == "t") {
            return u64::from_str_radix(addr, 16).ok();
        }
    }
    None
}

/// Any symbol's address from `llvm-nm` output, by name.
///
/// Type letter ignored: linker-defined symbols like `__rw_start` show up as `A`/`D`/`B`
/// depending on the section they fall in, and which one is not the caller's business.
pub fn parse_nm_symbol(out: &str, want: &str) -> Option<u64> {
    for line in out.lines() {
        let mut it = line.split_whitespace();
        let (addr, _ty, name) = (it.next()?, it.next()?, it.next()?);
        if name == want {
            return u64::from_str_radix(addr, 16).ok();
        }
    }
    None
}

/// How a tenant image is laid out, in bytes from the start of the flat binary.
#[derive(Debug, PartialEq, Eq)]
pub struct Layout {
    /// Entry point, from the start of the image.
    pub entry: u64,
    /// Bytes of read-execute image (text + rodata). Page-aligned by the linker script.
    pub rx: u64,
    /// Bytes of read-write image (data + bss). The flat binary holds only the `.data` part;
    /// the loader zeroes the rest, which is `.bss`.
    pub rw: u64,
}

/// Read the whole layout from `nm` output plus the linker script.
pub fn layout(nm_out: &str, ld: &str) -> Option<Layout> {
    let base = parse_link_base(ld)?;
    let rw_start = parse_nm_symbol(nm_out, "__rw_start")?;
    let rw_end = parse_nm_symbol(nm_out, "__rw_end")?;
    Some(Layout {
        entry: parse_nm_start(nm_out)?.checked_sub(base)?,
        rx: rw_start.checked_sub(base)?,
        rw: rw_end.checked_sub(rw_start)?,
    })
}

/// The load address a linker script sets, from its `. = <addr>;` line.
///
/// Read rather than duplicated: the script is the authority on where the image is based, and a
/// second copy of that number in the build tool is a drift waiting to happen.
pub fn parse_link_base(ld: &str) -> Option<u64> {
    for line in ld.lines() {
        // `continue`, **not** `?`. A `?` here returns from the whole function on the first line
        // that is not the assignment — which in a real script is line 1 (`ENTRY(_start)`), so it
        // found nothing and the caller silently got `None`.
        let Some(rest) = line.trim().strip_prefix(". =") else { continue };
        let rest = rest.trim().trim_end_matches(';').trim();
        let Some(hex) = rest.strip_prefix("0x") else { continue };
        if let Ok(v) = u64::from_str_radix(hex, 16) {
            return Some(v);
        }
    }
    None
}

/// Entry offset within the flat image: where `_start` is, less where the image starts.
///
/// `None` when `_start` precedes the base, which would mean the entry is not in the image —
/// refused rather than clamped to 0, because a silent 0 is the bug this replaces.
pub fn entry_offset(nm_out: &str, ld: &str) -> Option<u64> {
    parse_nm_start(nm_out)?.checked_sub(parse_link_base(ld)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    const LD: &str = "ENTRY(_start)\nSECTIONS {\n  . = 0x40000000;\n  .text : { *(.text) }\n}\n";

    #[test]
    fn the_offset_is_start_minus_the_scripts_base() {
        // The real x86 case: `_start` 12 bytes into the image.
        let nm = "000000004000000c T _start\n0000000040000100 T other\n";
        assert_eq!(entry_offset(nm, LD), Some(12));
        // And the aarch64 case, which was exact — which is why the bug hid on one arch.
        let a64 = "0000004000000000 T _start\n";
        let ld64 = "SECTIONS {\n  . = 0x4000000000;\n}\n";
        assert_eq!(entry_offset(a64, ld64), Some(0));
    }

    #[test]
    fn a_local_start_still_counts_and_other_symbols_do_not() {
        // Note this returns the *address*, not an offset — the subtraction is `entry_offset`'s.
        assert_eq!(parse_nm_start("0000000040000004 t _start\n"), Some(0x4000_0004));
        // `_start_of_something` must not match, or the offset silently comes from elsewhere.
        assert_eq!(parse_nm_start("0000000040000010 T _start_helper\n"), None);
        assert_eq!(parse_nm_start("0000000040000010 D _start\n"), None, "data symbol is not the entry");
    }

    #[test]
    fn the_layout_splits_read_execute_from_read_write() {
        // The split that lets a tenant have a mutable static at all: everything below
        // `__rw_start` is mapped RX, everything above RW.
        let nm = "000000004000000c T _start\n0000000040001000 A __rw_start\n0000000040003000 A __rw_end\n";
        assert_eq!(layout(nm, LD), Some(Layout { entry: 12, rx: 0x1000, rw: 0x2000 }));
    }

    #[test]
    fn a_layout_missing_its_markers_is_refused() {
        // Defaulting `rw` to 0 would map the data page RX and reproduce the original fault, so
        // an image without the markers must be rejected rather than guessed at.
        assert_eq!(layout("000000004000000c T _start\n", LD), None);
    }

    #[test]
    fn missing_or_malformed_input_is_refused_not_defaulted() {
        assert_eq!(entry_offset("", LD), None);
        assert_eq!(entry_offset("000000004000000c T _start\n", "SECTIONS { }"), None);
        // Entry below the base cannot be an offset into the image.
        assert_eq!(entry_offset("0000000000001000 T _start\n", LD), None);
    }
}
