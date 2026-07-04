//! Minimal flattened-device-tree (FDT/DTB) reader — just enough to discover the
//! physical RAM window (`/memory` `reg`) that QEMU `-M virt -kernel` passes in
//! x0 (Linux boot convention). This lets the kernel place its heap at the top of
//! *actual* RAM instead of a hardcoded per-model address (see [`super::mmu`] /
//! [`crate::mm`]).
//!
//! It runs with the **MMU off** (before `mmu::init`), so it must be pure: no
//! heap, no `Locked`/atomics, only direct big-endian pointer reads. It parses
//! only what it needs (`#address-cells`/`#size-cells` from the root and the
//! first `/memory` node's `reg`) and bounds every read to the blob's declared
//! `totalsize`, returning `None` on anything malformed or non-FDT.

const FDT_MAGIC: u32 = 0xd00d_feed;
const FDT_BEGIN_NODE: u32 = 1;
const FDT_END_NODE: u32 = 2;
const FDT_PROP: u32 = 3;
const FDT_NOP: u32 = 4;
const FDT_END: u32 = 9;

/// Read a big-endian u32 at byte offset `off` from `base`, if `[off, off+4)`
/// fits within `total`.
///
/// # Safety
/// `base` must point at a readable blob of at least `total` bytes.
#[inline]
unsafe fn be32(base: *const u8, off: usize, total: usize) -> Option<u32> {
    if off + 4 > total {
        return None;
    }
    // SAFETY: bounds checked against `total`; caller guarantees the blob.
    let p = unsafe { base.add(off) };
    // SAFETY: 4 in-bounds bytes; unaligned reads are fine on aarch64 normal mem.
    Some(u32::from_be_bytes(unsafe { [*p, *p.add(1), *p.add(2), *p.add(3)] }))
}

/// Read `cells` big-endian 32-bit cells starting at `off` as one u64 (FDT packs
/// wide addresses/sizes as consecutive cells, most significant first).
///
/// # Safety
/// As [`be32`].
unsafe fn read_cells(base: *const u8, off: usize, cells: u32, total: usize) -> Option<u64> {
    let mut v: u64 = 0;
    for i in 0..cells as usize {
        // SAFETY: delegated to `be32`, which bounds-checks.
        let c = unsafe { be32(base, off + i * 4, total)? };
        v = (v << 32) | c as u64;
    }
    Some(v)
}

/// A NUL-terminated C string starts at `off` — return its length (excluding the
/// NUL) if it terminates within `total`.
///
/// # Safety
/// As [`be32`].
unsafe fn cstr_len(base: *const u8, off: usize, total: usize) -> Option<usize> {
    let mut i = off;
    while i < total {
        // SAFETY: `i < total`, in bounds.
        if unsafe { *base.add(i) } == 0 {
            return Some(i - off);
        }
        i += 1;
    }
    None
}

/// True if the property name at `name_off` in the strings block equals `want`.
///
/// # Safety
/// As [`be32`]; `str_base = base + off_strings`.
unsafe fn prop_name_is(base: *const u8, off_strings: usize, name_off: usize, want: &[u8], total: usize) -> bool {
    let start = off_strings + name_off;
    // SAFETY: bounded by `cstr_len`.
    let Some(len) = (unsafe { cstr_len(base, start, total) }) else { return false };
    if len != want.len() {
        return false;
    }
    for (i, &b) in want.iter().enumerate() {
        // SAFETY: `start + i < start + len < total`.
        if unsafe { *base.add(start + i) } != b {
            return false;
        }
    }
    true
}

/// Parse the first `/memory` node's `(base, size)` from the FDT at physical
/// address `dtb_pa`. Returns `None` if `dtb_pa` is 0, not a valid FDT (bad
/// magic), or has no parseable memory node.
///
/// Cell widths come from the root `#address-cells`/`#size-cells` (default 2/2,
/// as on the `virt` machine). Only the root's cells are tracked — sufficient for
/// `/memory`, which is a direct child of the root.
///
/// # Safety
/// `dtb_pa`, if non-zero and a valid FDT, must point at a readable DTB blob.
pub unsafe fn memory_region(dtb_pa: u64) -> Option<(u64, u64)> {
    if dtb_pa == 0 {
        return None;
    }
    let base = dtb_pa as *const u8;
    // Header: magic@0, totalsize@4, off_dt_struct@8, off_dt_strings@12.
    // Read magic with a small fixed bound first, then trust totalsize.
    if unsafe { be32(base, 0, 16)? } != FDT_MAGIC {
        return None;
    }
    let total = unsafe { be32(base, 4, 16)? } as usize;
    let off_struct = unsafe { be32(base, 8, total)? } as usize;
    let off_strings = unsafe { be32(base, 12, total)? } as usize;

    let mut addr_cells: u32 = 2;
    let mut size_cells: u32 = 2;
    let mut depth: i32 = 0;
    // Set while walking the immediate children of the root whose node name
    // begins with "memory" (e.g. "memory@40000000").
    let mut in_memory = false;

    let mut off = off_struct;
    loop {
        let tok = unsafe { be32(base, off, total)? };
        off += 4;
        match tok {
            FDT_BEGIN_NODE => {
                depth += 1;
                // Node name: NUL-terminated string, then pad to 4 bytes.
                let name_off = off;
                let len = unsafe { cstr_len(base, name_off, total)? };
                // A memory node is a direct child of the root (depth == 2 once
                // we've entered it: root is depth 1). Match the "memory" prefix.
                if depth == 2 && len >= 6 {
                    let m = b"memory";
                    let mut matches = true;
                    for (i, &b) in m.iter().enumerate() {
                        if unsafe { *base.add(name_off + i) } != b {
                            matches = false;
                            break;
                        }
                    }
                    in_memory = matches;
                }
                off += (len + 1 + 3) & !3;
            }
            FDT_END_NODE => {
                if depth == 2 {
                    in_memory = false;
                }
                depth -= 1;
                if depth < 0 {
                    return None;
                }
            }
            FDT_PROP => {
                let len = unsafe { be32(base, off, total)? } as usize;
                let name_off = unsafe { be32(base, off + 4, total)? } as usize;
                let data_off = off + 8;
                // Root-level cell counts.
                if depth == 1 {
                    if unsafe { prop_name_is(base, off_strings, name_off, b"#address-cells", total) } {
                        if let Some(c) = unsafe { be32(base, data_off, total) } {
                            addr_cells = c;
                        }
                    } else if unsafe { prop_name_is(base, off_strings, name_off, b"#size-cells", total) } {
                        if let Some(c) = unsafe { be32(base, data_off, total) } {
                            size_cells = c;
                        }
                    }
                }
                // The memory node's `reg` = <base size> in the root's cells.
                if in_memory && unsafe { prop_name_is(base, off_strings, name_off, b"reg", total) } {
                    let mem_base = unsafe { read_cells(base, data_off, addr_cells, total)? };
                    let mem_size = unsafe { read_cells(base, data_off + addr_cells as usize * 4, size_cells, total)? };
                    if mem_size != 0 {
                        return Some((mem_base, mem_size));
                    }
                }
                off += (len + 3) & !3;
            }
            FDT_NOP => {}
            FDT_END => return None,
            _ => return None, // unknown token: malformed
        }
    }
}
