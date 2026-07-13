//! Flattened-device-tree (FDT/DTB) reader. Arch-neutral pure logic — a bootloader
//! (QEMU `-M virt -kernel`, or **m1n1** on real Apple Silicon) hands a DTB in x0
//! (Linux boot convention) describing RAM, the console UART, the interrupt
//! controller, DMA engines, and the display; this parses it so nothing is
//! hardcoded to one platform. The aarch64 boot path reaches it via
//! [`crate::arch::aarch64::dtb`], which re-exports this module.
//!
//! Everything here runs with the **MMU off** (before `mmu::init`) on the boot
//! path, so it must be pure: no heap, no `Locked`/atomics, only direct
//! big-endian pointer reads, every access bounded by the blob's declared
//! `totalsize`, `None` on anything malformed or non-FDT. Being arch-neutral and
//! pointer-only, it is exercised by the host unit suite (`cargo xtask test`)
//! against blobs built in-memory.

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
                off += 8 + ((len + 3) & !3); // skip len(4)+nameoff(4)+padded value
            }
            FDT_NOP => {}
            FDT_END => return None,
            _ => return None, // unknown token: malformed
        }
    }
}

// ---------------------------------------------------------------------------
// General FDT traversal (extends the `/memory`-only reader above so the same
// blob m1n1 hands us in x0 can describe every device we discover — UART, AIC,
// DART, PMGR, USB, the simple framebuffer — instead of hardcoding QEMU-virt
// addresses). Still MMU-off-safe: no heap, no atomics, only bounded big-endian
// pointer reads. `#address-cells`/`#size-cells` are tracked **per parent** via a
// small fixed depth stack, so a device `reg` decodes against its own bus's cell
// widths (the root-only tracking above is insufficient for `/soc`-nested nodes).
// ---------------------------------------------------------------------------

/// Max node nesting we track cell widths for. Real Apple/QEMU trees are shallow.
const MAX_DEPTH: usize = 32;

/// Parsed FDT header offsets, validated against the magic + declared total size.
struct Header {
    total: usize,
    off_struct: usize,
    off_strings: usize,
}

/// Validate the FDT magic and read the struct/strings block offsets.
///
/// # Safety
/// `dtb_pa`, if a valid FDT, must point at a readable blob.
unsafe fn header(dtb_pa: u64) -> Option<(*const u8, Header)> {
    if dtb_pa == 0 {
        return None;
    }
    let base = dtb_pa as *const u8;
    if unsafe { be32(base, 0, 16)? } != FDT_MAGIC {
        return None;
    }
    let total = unsafe { be32(base, 4, 16)? } as usize;
    let off_struct = unsafe { be32(base, 8, total)? } as usize;
    let off_strings = unsafe { be32(base, 12, total)? } as usize;
    Some((base, Header { total, off_struct, off_strings }))
}

/// True if the `compatible` property value at `[data_off, data_off+len)` — a
/// sequence of NUL-terminated strings — contains an exact match for `want`.
///
/// # Safety
/// As [`be32`].
unsafe fn compat_has(base: *const u8, data_off: usize, len: usize, want: &[u8], total: usize) -> bool {
    let end = (data_off + len).min(total);
    let mut i = data_off;
    while i < end {
        // SAFETY: `i < end <= total`.
        let Some(slen) = (unsafe { cstr_len(base, i, total) }) else { return false };
        if slen == want.len() {
            let mut eq = true;
            for (k, &b) in want.iter().enumerate() {
                // SAFETY: `i + k < i + slen < total`.
                if unsafe { *base.add(i + k) } != b {
                    eq = false;
                    break;
                }
            }
            if eq {
                return true;
            }
        }
        i += slen + 1; // skip this string + its NUL
    }
    false
}

/// The physical `reg` `(base, size)` of the first node whose `compatible` list
/// contains `want` (e.g. `b"apple,s5l-uart"`, `b"apple,t8110-dart"`). Decodes the
/// first `reg` pair using the node's **parent** cell widths. Returns `None` if
/// absent. This is the generic device-base primitive the Apple drivers use.
///
/// # Safety
/// `dtb_pa`, if a valid FDT, must point at a readable blob.
pub unsafe fn reg_of_compatible(dtb_pa: u64, want: &[u8]) -> Option<(u64, u64)> {
    // SAFETY: delegated to `header`.
    let (base, h) = unsafe { header(dtb_pa)? };
    let mut acells = [2u32; MAX_DEPTH];
    let mut scells = [2u32; MAX_DEPTH];
    let mut matched = [false; MAX_DEPTH];
    let mut reg_off = [0usize; MAX_DEPTH]; // 0 = no reg seen for this node
    let mut depth: usize = 0;
    let mut off = h.off_struct;
    loop {
        // SAFETY: `off` is bounded by every `be32` against `h.total`.
        let tok = unsafe { be32(base, off, h.total)? };
        off += 4;
        match tok {
            FDT_BEGIN_NODE => {
                depth += 1;
                if depth >= MAX_DEPTH {
                    return None; // deeper than we track — bail rather than corrupt
                }
                acells[depth] = 2;
                scells[depth] = 2;
                matched[depth] = false;
                reg_off[depth] = 0;
                let name_off = off;
                // SAFETY: bounded.
                let len = unsafe { cstr_len(base, name_off, h.total)? };
                off += (len + 1 + 3) & !3;
            }
            FDT_END_NODE => {
                if depth == 0 {
                    return None;
                }
                if matched[depth] && reg_off[depth] != 0 {
                    let pa = acells[depth - 1];
                    let ps = scells[depth - 1];
                    // SAFETY: bounded.
                    let b = unsafe { read_cells(base, reg_off[depth], pa, h.total)? };
                    let s = unsafe { read_cells(base, reg_off[depth] + pa as usize * 4, ps, h.total)? };
                    return Some((b, s));
                }
                depth -= 1;
            }
            FDT_PROP => {
                // SAFETY: bounded.
                let len = unsafe { be32(base, off, h.total)? } as usize;
                let name_off = unsafe { be32(base, off + 4, h.total)? } as usize;
                let data_off = off + 8;
                if depth < MAX_DEPTH {
                    if unsafe { prop_name_is(base, h.off_strings, name_off, b"#address-cells", h.total) } {
                        if let Some(c) = unsafe { be32(base, data_off, h.total) } {
                            acells[depth] = c;
                        }
                    } else if unsafe { prop_name_is(base, h.off_strings, name_off, b"#size-cells", h.total) } {
                        if let Some(c) = unsafe { be32(base, data_off, h.total) } {
                            scells[depth] = c;
                        }
                    } else if unsafe { prop_name_is(base, h.off_strings, name_off, b"compatible", h.total) } {
                        if unsafe { compat_has(base, data_off, len, want, h.total) } {
                            matched[depth] = true;
                        }
                    } else if unsafe { prop_name_is(base, h.off_strings, name_off, b"reg", h.total) } {
                        reg_off[depth] = data_off;
                    }
                }
                off += 8 + ((len + 3) & !3); // skip len(4)+nameoff(4)+padded value
            }
            FDT_NOP => {}
            FDT_END => return None,
            _ => return None,
        }
    }
}

/// A simple linear framebuffer as described by a `simple-framebuffer` FDT node
/// (iBoot sets it up; m1n1 leaves it lit and passes it through). All the console
/// needs to light up on the real display.
#[derive(Clone, Copy)]
pub struct Fb {
    pub base: u64,
    pub size: u64,
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    /// Per-channel bit shift + bytes-per-pixel, decoded from the `format` string.
    pub r_shift: u8,
    pub g_shift: u8,
    pub b_shift: u8,
    pub bpp: u8,
}

/// Map a `simple-framebuffer` `format` string to `(r_shift, g_shift, b_shift,
/// bytes_per_pixel)`. Covers the formats Apple/iBoot and QEMU use; unknown
/// formats fall back to 32bpp XRGB (`x8r8g8b8`). Pure — unit-tested.
pub fn format_shifts(fmt: &[u8]) -> (u8, u8, u8, u8) {
    match fmt {
        b"x8r8g8b8" | b"a8r8g8b8" => (16, 8, 0, 4),
        b"x8b8g8r8" | b"a8b8g8r8" => (0, 8, 16, 4),
        // Apple's 30-bit mode: 10-bit channels packed in 32 bits, red high.
        // Our console writes 8-bit channel values at the top of each field.
        b"x2r10g10b10" => (22, 12, 2, 4),
        b"x2b10g10r10" => (2, 12, 22, 4),
        b"r5g6b5" => (11, 5, 0, 2),
        _ => (16, 8, 0, 4),
    }
}

/// Find the `simple-framebuffer` node and read its geometry + pixel format.
///
/// # Safety
/// `dtb_pa`, if a valid FDT, must point at a readable blob.
pub unsafe fn find_framebuffer(dtb_pa: u64) -> Option<Fb> {
    // SAFETY: delegated to `header`.
    let (base, h) = unsafe { header(dtb_pa)? };
    let mut acells = [2u32; MAX_DEPTH];
    let mut scells = [2u32; MAX_DEPTH];
    let mut depth: usize = 0;
    // Per-current-node accumulation (framebuffer nodes are leaves; commit on
    // END_NODE using these scalars, reset at each BEGIN_NODE).
    let mut is_fb = false;
    let mut reg_off = 0usize;
    let mut width = 0u32;
    let mut height = 0u32;
    let mut stride = 0u32;
    let mut fmt = [0u8; 16];
    let mut fmt_len = 0usize;
    let mut off = h.off_struct;
    loop {
        // SAFETY: bounded by `be32`.
        let tok = unsafe { be32(base, off, h.total)? };
        off += 4;
        match tok {
            FDT_BEGIN_NODE => {
                depth += 1;
                if depth >= MAX_DEPTH {
                    return None;
                }
                acells[depth] = 2;
                scells[depth] = 2;
                is_fb = false;
                reg_off = 0;
                width = 0;
                height = 0;
                stride = 0;
                fmt_len = 0;
                // SAFETY: bounded.
                let len = unsafe { cstr_len(base, off, h.total)? };
                off += (len + 1 + 3) & !3;
            }
            FDT_END_NODE => {
                if depth == 0 {
                    return None;
                }
                if is_fb && reg_off != 0 {
                    let pa = acells[depth - 1];
                    let ps = scells[depth - 1];
                    // SAFETY: bounded.
                    let b = unsafe { read_cells(base, reg_off, pa, h.total)? };
                    let s = unsafe { read_cells(base, reg_off + pa as usize * 4, ps, h.total)? };
                    let (r_shift, g_shift, b_shift, bpp) = format_shifts(&fmt[..fmt_len]);
                    // Fall back to a byte-derived stride if the node omitted one.
                    let stride = if stride != 0 { stride } else { width * bpp as u32 };
                    return Some(Fb { base: b, size: s, width, height, stride, r_shift, g_shift, b_shift, bpp });
                }
                depth -= 1;
            }
            FDT_PROP => {
                // SAFETY: bounded.
                let len = unsafe { be32(base, off, h.total)? } as usize;
                let name_off = unsafe { be32(base, off + 4, h.total)? } as usize;
                let data_off = off + 8;
                if depth < MAX_DEPTH {
                    if unsafe { prop_name_is(base, h.off_strings, name_off, b"#address-cells", h.total) } {
                        if let Some(c) = unsafe { be32(base, data_off, h.total) } {
                            acells[depth] = c;
                        }
                    } else if unsafe { prop_name_is(base, h.off_strings, name_off, b"#size-cells", h.total) } {
                        if let Some(c) = unsafe { be32(base, data_off, h.total) } {
                            scells[depth] = c;
                        }
                    } else if unsafe { prop_name_is(base, h.off_strings, name_off, b"compatible", h.total) } {
                        if unsafe { compat_has(base, data_off, len, b"simple-framebuffer", h.total) } {
                            is_fb = true;
                        }
                    } else if unsafe { prop_name_is(base, h.off_strings, name_off, b"reg", h.total) } {
                        reg_off = data_off;
                    } else if unsafe { prop_name_is(base, h.off_strings, name_off, b"width", h.total) } {
                        width = unsafe { be32(base, data_off, h.total) }.unwrap_or(0);
                    } else if unsafe { prop_name_is(base, h.off_strings, name_off, b"height", h.total) } {
                        height = unsafe { be32(base, data_off, h.total) }.unwrap_or(0);
                    } else if unsafe { prop_name_is(base, h.off_strings, name_off, b"stride", h.total) } {
                        stride = unsafe { be32(base, data_off, h.total) }.unwrap_or(0);
                    } else if unsafe { prop_name_is(base, h.off_strings, name_off, b"format", h.total) } {
                        let n = len.min(fmt.len());
                        // A `format` value is a NUL-terminated string; copy sans NUL.
                        let mut k = 0;
                        while k < n {
                            // SAFETY: `data_off + k < data_off + len <= total`.
                            let c = unsafe { *base.add(data_off + k) };
                            if c == 0 {
                                break;
                            }
                            fmt[k] = c;
                            k += 1;
                        }
                        fmt_len = k;
                    }
                }
                off += 8 + ((len + 3) & !3); // skip len(4)+nameoff(4)+padded value
            }
            FDT_NOP => {}
            FDT_END => return None,
            _ => return None,
        }
    }
}

/// The RAM blob m1n1 loaded for us (our model vehicle) + kernel boot arguments,
/// read from `/chosen`. `initrd` is `(start, end)` physical; `bootargs` is a
/// `(ptr, len)` view into the still-mapped FDT (parsed later for `chitti.epoch`).
#[derive(Clone, Copy)]
pub struct Chosen {
    pub initrd_start: u64,
    pub initrd_end: u64,
    pub bootargs_ptr: *const u8,
    pub bootargs_len: usize,
}

/// Read `/chosen`'s `linux,initrd-start`/`-end` and `bootargs`.
///
/// # Safety
/// `dtb_pa`, if a valid FDT, must point at a readable blob.
pub unsafe fn chosen(dtb_pa: u64) -> Option<Chosen> {
    // SAFETY: delegated to `header`.
    let (base, h) = unsafe { header(dtb_pa)? };
    let mut depth: usize = 0;
    let mut in_chosen = false;
    let mut out = Chosen { initrd_start: 0, initrd_end: 0, bootargs_ptr: core::ptr::null(), bootargs_len: 0 };
    let mut off = h.off_struct;
    loop {
        // SAFETY: bounded by `be32`.
        let tok = unsafe { be32(base, off, h.total)? };
        off += 4;
        match tok {
            FDT_BEGIN_NODE => {
                depth += 1;
                let name_off = off;
                // SAFETY: bounded.
                let len = unsafe { cstr_len(base, name_off, h.total)? };
                if depth == 2 && len == 6 {
                    let want = b"chosen";
                    let mut eq = true;
                    for (i, &b) in want.iter().enumerate() {
                        // SAFETY: `name_off + i < name_off + len < total`.
                        if unsafe { *base.add(name_off + i) } != b {
                            eq = false;
                            break;
                        }
                    }
                    in_chosen = eq;
                }
                off += (len + 1 + 3) & !3;
            }
            FDT_END_NODE => {
                if depth == 2 && in_chosen {
                    return Some(out);
                }
                if depth == 0 {
                    return None;
                }
                depth -= 1;
            }
            FDT_PROP => {
                // SAFETY: bounded.
                let len = unsafe { be32(base, off, h.total)? } as usize;
                let name_off = unsafe { be32(base, off + 4, h.total)? } as usize;
                let data_off = off + 8;
                if in_chosen {
                    // initrd start/end may be `<u32>` or `<u64>` — read `len` cells.
                    if unsafe { prop_name_is(base, h.off_strings, name_off, b"linux,initrd-start", h.total) } {
                        out.initrd_start = unsafe { read_cells(base, data_off, (len / 4) as u32, h.total) }.unwrap_or(0);
                    } else if unsafe { prop_name_is(base, h.off_strings, name_off, b"linux,initrd-end", h.total) } {
                        out.initrd_end = unsafe { read_cells(base, data_off, (len / 4) as u32, h.total) }.unwrap_or(0);
                    } else if unsafe { prop_name_is(base, h.off_strings, name_off, b"bootargs", h.total) } {
                        // SAFETY: `data_off + len <= total`; view into the blob.
                        out.bootargs_ptr = unsafe { base.add(data_off) };
                        out.bootargs_len = len.saturating_sub(1); // drop trailing NUL
                    }
                }
                off += 8 + ((len + 3) & !3); // skip len(4)+nameoff(4)+padded value
            }
            FDT_NOP => {}
            FDT_END => {
                return if out.initrd_start != 0 || !out.bootargs_ptr.is_null() { Some(out) } else { None };
            }
            _ => return None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    /// A tiny builder that emits a valid v17 FDT blob for the parser tests.
    struct FdtBuild {
        st: Vec<u8>,   // struct block
        strs: Vec<u8>, // strings block
    }
    impl FdtBuild {
        fn new() -> Self {
            FdtBuild { st: Vec::new(), strs: Vec::new() }
        }
        fn tok(&mut self, t: u32) {
            self.st.extend_from_slice(&t.to_be_bytes());
        }
        fn align(&mut self) {
            while self.st.len() % 4 != 0 {
                self.st.push(0);
            }
        }
        fn begin(&mut self, name: &str) {
            self.tok(FDT_BEGIN_NODE);
            self.st.extend_from_slice(name.as_bytes());
            self.st.push(0);
            self.align();
        }
        fn end(&mut self) {
            self.tok(FDT_END_NODE);
        }
        fn stroff(&mut self, name: &str) -> u32 {
            let off = self.strs.len() as u32;
            self.strs.extend_from_slice(name.as_bytes());
            self.strs.push(0);
            off
        }
        fn prop(&mut self, name: &str, val: &[u8]) {
            let no = self.stroff(name);
            self.tok(FDT_PROP);
            self.st.extend_from_slice(&(val.len() as u32).to_be_bytes());
            self.st.extend_from_slice(&no.to_be_bytes());
            self.st.extend_from_slice(val);
            self.align();
        }
        fn prop_u32(&mut self, name: &str, v: u32) {
            self.prop(name, &v.to_be_bytes());
        }
        fn prop_pair64(&mut self, name: &str, a: u64, b: u64) {
            let mut v = Vec::new();
            v.extend_from_slice(&a.to_be_bytes());
            v.extend_from_slice(&b.to_be_bytes());
            self.prop(name, &v);
        }
        fn prop_str(&mut self, name: &str, s: &str) {
            let mut v = Vec::from(s.as_bytes());
            v.push(0);
            self.prop(name, &v);
        }
        fn build(mut self) -> Vec<u8> {
            self.tok(FDT_END);
            let rsvmap = 16usize; // one terminating (0,0) reserve entry
            let off_struct = 40 + rsvmap;
            let off_strings = off_struct + self.st.len();
            let total = off_strings + self.strs.len();
            let mut out = Vec::with_capacity(total);
            let mut h = [0u32; 10];
            h[0] = FDT_MAGIC;
            h[1] = total as u32;
            h[2] = off_struct as u32;
            h[3] = off_strings as u32;
            h[4] = 40; // off_mem_rsvmap
            h[5] = 17; // version
            h[6] = 16; // last_comp_version
            h[7] = 0; // boot_cpuid_phys
            h[8] = self.strs.len() as u32;
            h[9] = self.st.len() as u32;
            for w in h {
                out.extend_from_slice(&w.to_be_bytes());
            }
            out.extend_from_slice(&[0u8; 16]); // reserve map terminator
            out.extend_from_slice(&self.st);
            out.extend_from_slice(&self.strs);
            out
        }
    }

    /// A representative Apple-ish tree: root cells, /memory, /chosen, an
    /// s5l UART under /soc, and a simple-framebuffer.
    fn sample() -> Vec<u8> {
        let mut f = FdtBuild::new();
        f.begin(""); // root
        f.prop_u32("#address-cells", 2);
        f.prop_u32("#size-cells", 2);

        f.begin("memory@800000000");
        f.prop_str("device_type", "memory");
        f.prop_pair64("reg", 0x8_0000_0000, 0x2_0000_0000);
        f.end();

        f.begin("chosen");
        f.prop_str("bootargs", "chitti.epoch=1700000000");
        // initrd start/end are each a single value (here 8-byte u64, as m1n1 emits).
        f.prop("linux,initrd-start", &0x8_1000_0000u64.to_be_bytes());
        f.prop("linux,initrd-end", &0x8_1400_0000u64.to_be_bytes());
        f.end();

        f.begin("soc");
        f.prop_u32("#address-cells", 2);
        f.prop_u32("#size-cells", 2);
        f.begin("serial@235200000");
        f.prop_str("compatible", "apple,s5l-uart");
        f.prop_pair64("reg", 0x2_3520_0000, 0x1000);
        f.end();
        f.end();

        f.begin("framebuffer");
        f.prop_str("compatible", "simple-framebuffer");
        f.prop_pair64("reg", 0x9_A000_0000, 0x00A0_0000);
        f.prop_u32("width", 2560);
        f.prop_u32("height", 1600);
        f.prop_u32("stride", 2560 * 4);
        f.prop_str("format", "x8r8g8b8");
        f.end();

        f.end(); // root
        f.build()
    }

    #[test_case]
    fn memory_region_reads_apple_base() {
        let blob = sample();
        let got = unsafe { memory_region(blob.as_ptr() as u64) };
        assert_eq!(got, Some((0x8_0000_0000, 0x2_0000_0000)));
    }

    #[test_case]
    fn reg_of_compatible_finds_uart_under_soc() {
        let blob = sample();
        let got = unsafe { reg_of_compatible(blob.as_ptr() as u64, b"apple,s5l-uart") };
        assert_eq!(got, Some((0x2_3520_0000, 0x1000)));
        // A compatible that isn't present returns None.
        assert_eq!(unsafe { reg_of_compatible(blob.as_ptr() as u64, b"apple,t8110-dart") }, None);
    }

    #[test_case]
    fn find_framebuffer_reads_geometry_and_format() {
        let blob = sample();
        let fb = unsafe { find_framebuffer(blob.as_ptr() as u64) }.expect("fb present");
        assert_eq!(fb.base, 0x9_A000_0000);
        assert_eq!(fb.size, 0x00A0_0000);
        assert_eq!(fb.width, 2560);
        assert_eq!(fb.height, 1600);
        assert_eq!(fb.stride, 2560 * 4);
        assert_eq!((fb.r_shift, fb.g_shift, fb.b_shift, fb.bpp), (16, 8, 0, 4));
    }

    #[test_case]
    fn chosen_reads_initrd_and_bootargs() {
        let blob = sample();
        let c = unsafe { chosen(blob.as_ptr() as u64) }.expect("chosen present");
        assert_eq!(c.initrd_start, 0x8_1000_0000);
        assert_eq!(c.initrd_end, 0x8_1400_0000);
        // bootargs points into the blob; reconstruct and check.
        let s = unsafe { core::slice::from_raw_parts(c.bootargs_ptr, c.bootargs_len) };
        assert_eq!(s, b"chitti.epoch=1700000000");
    }

    #[test_case]
    fn rejects_non_fdt() {
        assert_eq!(unsafe { memory_region(0) }, None);
        let junk = [0u8; 64];
        assert!(unsafe { find_framebuffer(junk.as_ptr() as u64) }.is_none());
        assert!(unsafe { chosen(junk.as_ptr() as u64) }.is_none());
        assert_eq!(unsafe { reg_of_compatible(junk.as_ptr() as u64, b"anything") }, None);
    }

    #[test_case]
    fn format_shifts_known_and_fallback() {
        assert_eq!(format_shifts(b"x8r8g8b8"), (16, 8, 0, 4));
        assert_eq!(format_shifts(b"a8b8g8r8"), (0, 8, 16, 4));
        assert_eq!(format_shifts(b"r5g6b5"), (11, 5, 0, 2));
        assert_eq!(format_shifts(b"x2r10g10b10"), (22, 12, 2, 4));
        // Unknown → 32bpp XRGB fallback.
        assert_eq!(format_shifts(b"weird"), (16, 8, 0, 4));
    }
}
