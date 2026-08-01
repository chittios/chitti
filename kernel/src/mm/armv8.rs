//! Pure ARMv8-A (VMSAv8-64) translation-table descriptor encoding: index math,
//! block/page/table descriptor construction, and the block-split arithmetic a
//! 4 KiB page-table walker needs.
//!
//! **Why this lives under `mm/` and not `arch/aarch64/`.** The unit suite
//! (`cargo xtask test`) runs on x86 only, and `arch::aarch64` is `cfg`-gated out
//! of that build — so aarch64 logic placed there is untestable by construction.
//! Descriptor encoding is exactly the kind of thing that must be tested rather
//! than eyeballed: every field is a bit range, a wrong one produces a
//! *plausible* descriptor, and the failure mode is the hardware's page-table
//! walker silently missing our entry — an unbacked mapping, not an error. So the
//! bit packing lives here, compiled on both arches, covered by tests that run on
//! x86; [`crate::arch::aarch64::mmu`] holds only the MMIO/barrier/allocation
//! half. Same trick as [`crate::mm::ramlayout`], [`crate::edid`] and
//! `acpi::fadt_dsdt`.
//!
//! Geometry is the one the kernel actually programs in `TCR_EL1` (`T0SZ = 25`,
//! 4 KiB granule): a **39-bit** VA resolved over three levels — L1 selects a
//! 1 GiB block, L2 a 2 MiB block, L3 a 4 KiB page. There is no L0, which is why
//! `arch::aarch64::mmu`'s root table is genuinely level 1.

/// Levels this geometry has, coarsest first. L1 = 1 GiB, L2 = 2 MiB, L3 = 4 KiB.
pub const TOP_LEVEL: u32 = 1;
/// The level whose descriptors are 4 KiB pages (no finer level exists).
pub const PAGE_LEVEL: u32 = 3;
/// Descriptors per table at every level (4 KiB / 8 bytes).
pub const ENTRIES: usize = 512;
/// Virtual-address bits `TCR_EL1.T0SZ = 25` yields.
pub const VA_BITS: u32 = 39;

// --- descriptor type bits [1:0] -----------------------------------------

/// An invalid (unmapped) descriptor.
pub const INVALID: u64 = 0b00;
/// Type bits for a **block** descriptor — legal only at L1 and L2.
pub const BLOCK: u64 = 0b01;
/// Type bits for a **table** descriptor (L1/L2) and, at L3, for a **page**.
/// The reuse of `0b11` is why [`descriptor`] takes the level: emitting `BLOCK`
/// at L3 produces a *reserved* encoding the walker treats as a fault.
pub const TABLE: u64 = 0b11;

// --- lower attributes [11:2] --------------------------------------------

/// MAIR index for Normal write-back cacheable memory (`attr0`; see
/// `arch::aarch64::mmu`'s `MAIR_EL1` value).
pub const ATTR_NORMAL: u64 = 0 << 2;
/// MAIR index for Device-nGnRnE memory (`attr1`).
pub const ATTR_DEVICE: u64 = 1 << 2;
/// `AP[1]`: the mapping is accessible at **EL0** as well as EL1. Absent from
/// every descriptor the boot identity map builds — the kernel has no userspace
/// yet — and the bit a user mapping must set.
pub const AP_USER: u64 = 1 << 6;
/// `AP[2]`: read-only (at both exception levels the mapping is accessible from).
pub const AP_RO: u64 = 1 << 7;
/// Inner-shareable. Required on Normal cacheable memory for the cores to see
/// each other's writes; meaningless (and conventionally 0) on Device memory.
pub const SH_INNER: u64 = 0b11 << 8;
/// Access flag. **A descriptor without it faults on first touch**, and nothing
/// in this kernel handles that fault, so every mapping sets it.
pub const AF: u64 = 1 << 10;
/// Not-global: the entry is tagged with the current ASID rather than shared by
/// all address spaces. For per-process mappings.
pub const NG: u64 = 1 << 11;

// --- upper attributes [63:51] ------------------------------------------

/// Privileged-execute-never.
pub const PXN: u64 = 1 << 53;
/// Unprivileged-execute-never. (At EL1 with a single translation regime this is
/// the plain `XN`.)
pub const UXN: u64 = 1 << 54;
/// The **Contiguous** hint: this entry is one of an aligned run of 16 that the
/// TLB may cache as a single larger entry. Never set by this kernel, and
/// [`split_child`] deliberately clears it — see there for why.
pub const CONTIGUOUS: u64 = 1 << 52;

/// Output-address field, bits `47:12`.
pub const ADDR_MASK: u64 = 0x0000_ffff_ffff_f000;

/// Every attribute bit a descriptor carries: the lower block `11:2` and the
/// upper block `63:52`. Deliberately excludes bit 51 (`DBM`, ARMv8.1) — this
/// kernel never sets it, and a mask that claimed it would silently propagate a
/// bit whose meaning depends on `TCR_EL1.HD` being enabled.
pub const ATTR_MASK: u64 = 0xfff0_0000_0000_0000 | 0x0000_0000_0000_0ffc;

/// The Normal-memory attributes the boot identity map uses for RAM.
pub const fn normal_attrs() -> u64 {
    ATTR_NORMAL | SH_INNER | AF
}

/// The Device-memory attributes the boot identity map uses for MMIO.
pub const fn device_attrs() -> u64 {
    ATTR_DEVICE | AF
}

/// Bytes one descriptor at `level` spans: 1 GiB at L1, 2 MiB at L2, 4 KiB at L3.
pub const fn level_size(level: u32) -> u64 {
    // L1 -> 30, L2 -> 21, L3 -> 12.
    1u64 << level_shift(level)
}

/// The VA bit position where `level`'s index field starts.
pub const fn level_shift(level: u32) -> u32 {
    12 + 9 * (PAGE_LEVEL - level)
}

/// Index into the table at `level` that `va` selects. L1 reads VA bits `38:30`,
/// L2 `29:21`, L3 `20:12`.
pub const fn table_index(va: u64, level: u32) -> usize {
    ((va >> level_shift(level)) & 0x1ff) as usize
}

/// A **block** (L1/L2) or **page** (L3) descriptor mapping `pa` with `attrs`.
///
/// The type bits depend on the level — `0b01` for a block, `0b11` for a page —
/// which is the whole reason this takes `level` rather than being two
/// constants. `attrs` is masked to the real attribute fields so a caller cannot
/// accidentally inject address or type bits.
pub const fn descriptor(level: u32, pa: u64, attrs: u64) -> u64 {
    let ty = if level == PAGE_LEVEL { TABLE } else { BLOCK };
    (pa & ADDR_MASK) | (attrs & ATTR_MASK) | ty
}

/// A **table** descriptor pointing at the next-level table at physical `pa`.
/// Carries no memory attributes: the leaf descriptor's attributes are what
/// apply (this kernel leaves the table-descriptor hierarchical permission bits
/// `APTable`/`XNTable`/`PXNTable` at 0, i.e. "no restriction added").
pub const fn table_descriptor(pa: u64) -> u64 {
    (pa & ADDR_MASK) | TABLE
}

/// Whether `desc` is a valid (mapped) descriptor of any kind.
pub const fn is_valid(desc: u64) -> bool {
    desc & 0b11 != INVALID
}

/// Whether `desc` at `level` points at a next-level table. Only true below
/// [`PAGE_LEVEL`]: at L3 the same `0b11` encoding is a page, and reading it as a
/// table would walk into the mapped memory as if it held descriptors.
pub const fn is_table(desc: u64, level: u32) -> bool {
    level < PAGE_LEVEL && desc & 0b11 == TABLE
}

/// Whether `desc` at `level` maps memory directly (a block at L1/L2, a page at
/// L3) rather than pointing at another table.
pub const fn is_leaf(desc: u64, level: u32) -> bool {
    if level == PAGE_LEVEL {
        desc & 0b11 == TABLE
    } else {
        desc & 0b11 == BLOCK
    }
}

/// The output address a valid descriptor carries: the next-level table for a
/// table descriptor, the mapped memory for a block/page.
pub const fn descriptor_addr(desc: u64) -> u64 {
    desc & ADDR_MASK
}

/// The attributes a leaf descriptor carries, ready to hand back to
/// [`descriptor`].
pub const fn descriptor_attrs(desc: u64) -> u64 {
    desc & ATTR_MASK
}

/// What a walker heading for a 4 KiB page must do with the descriptor it found.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Step {
    /// Descend into the next-level table at this physical address.
    Descend(u64),
    /// Nothing is mapped here: a fresh zeroed table must be installed.
    NeedTable,
    /// A block maps this address at the current level. Reaching a finer granule
    /// means breaking it into a next-level table first.
    SplitBlock,
}

/// Classify `desc` at `level` for a walk towards [`PAGE_LEVEL`].
///
/// Only meaningful for `level < PAGE_LEVEL` — at L3 there is nothing to descend
/// into, and the `0b11` that means "table" above would mean "page" there. That
/// asymmetry is the whole reason this is a named function rather than an inline
/// `match`: a walker that treated an L3 page as a table would walk *into the
/// mapped memory* and read whatever it holds as descriptors.
pub const fn walk_step(desc: u64, level: u32) -> Step {
    if !is_valid(desc) {
        Step::NeedTable
    } else if is_table(desc, level) {
        Step::Descend(descriptor_addr(desc))
    } else {
        Step::SplitBlock
    }
}

/// Child `i` of a `level` block descriptor being split into a `level + 1`
/// table: the same memory, same attributes, one level finer.
///
/// This is the arithmetic behind splitting a live 2 MiB block into 4 KiB pages
/// so a single page inside it can be remapped. Two things it must get right,
/// both of which produce a working-looking descriptor when wrong:
///
/// * the type bits are re-derived for the child's level, so a 2 MiB block
///   (`0b01`) becomes a 4 KiB **page** (`0b11`) — carrying `0b01` down to L3 is
///   a *reserved* encoding, so every one of the 512 children would fault;
/// * the [`CONTIGUOUS`] hint is dropped. It is a claim about an aligned run of
///   16 entries *at the child's level*, which the parent's bit says nothing
///   about; propagating it would license the TLB to cache a larger entry than
///   the table describes.
pub const fn split_child(level: u32, desc: u64, i: usize) -> u64 {
    let child = level + 1;
    let pa = descriptor_addr(desc) + (i as u64) * level_size(child);
    descriptor(child, pa, descriptor_attrs(desc) & !CONTIGUOUS)
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;
    const MIB2: u64 = 1 << 21;

    #[test_case]
    fn level_geometry_matches_the_tcr_the_kernel_programs() {
        // T0SZ=25 with a 4 KiB granule: three levels, 1 GiB / 2 MiB / 4 KiB.
        assert_eq!(level_size(1), GIB);
        assert_eq!(level_size(2), MIB2);
        assert_eq!(level_size(3), 4096);
        // 512 entries at the top level times 1 GiB is the whole 39-bit VA space.
        assert_eq!(ENTRIES as u64 * level_size(TOP_LEVEL), 1u64 << VA_BITS);
    }

    #[test_case]
    fn table_index_reads_the_right_va_bit_field() {
        // L1 = bits 38:30, L2 = 29:21, L3 = 20:12. Build a VA with a distinct
        // index at each level so a shifted-by-9 mistake cannot pass.
        let va = (5u64 << 30) | (17u64 << 21) | (300u64 << 12) | 0xabc;
        assert_eq!(table_index(va, 1), 5);
        assert_eq!(table_index(va, 2), 17);
        assert_eq!(table_index(va, 3), 300);
    }

    #[test_case]
    fn table_index_ignores_bits_above_the_39_bit_va() {
        // A 40-bit address's bit 39 must not leak into the L1 index (it would
        // alias entry 0 onto entry 256 and quietly map the wrong gigabyte).
        assert_eq!(table_index(1u64 << 39, 1), 0);
        assert_eq!(table_index((1u64 << 39) | (3 << 30), 1), 3);
    }

    #[test_case]
    fn a_page_descriptor_is_0b11_and_a_block_is_0b01() {
        // The encoding trap: `0b01` at L3 is reserved, so a splitter that
        // carried the block type down would produce 512 faulting entries.
        let page = descriptor(PAGE_LEVEL, 0x4000_1000, normal_attrs());
        assert_eq!(page & 0b11, TABLE, "an L3 leaf is a page descriptor (0b11)");
        assert!(is_leaf(page, PAGE_LEVEL));
        assert!(!is_table(page, PAGE_LEVEL), "0b11 at L3 is a page, never a table");

        let block = descriptor(2, 0x4020_0000, normal_attrs());
        assert_eq!(block & 0b11, BLOCK);
        assert!(is_leaf(block, 2));
        assert!(!is_table(block, 2));
    }

    #[test_case]
    fn the_encoders_reproduce_the_boot_identity_maps_descriptors() {
        // These are the exact values `arch::aarch64::mmu` has always written
        // (`normal_block`/`device_block`/`normal_l2_block`/`device_l2_block`).
        // Pinning them means the walker cannot introduce a *different* encoding
        // of "the same" mapping, which is how a split block silently changes
        // memory type under a running driver.
        let pa = 4 * GIB;
        assert_eq!(descriptor(1, pa, normal_attrs()), pa | (0b11 << 8) | (1 << 10) | 0b01);
        assert_eq!(descriptor(1, pa, device_attrs()), pa | (1 << 2) | (1 << 10) | 0b01);
        assert_eq!(descriptor(2, MIB2, normal_attrs()), MIB2 | (0b11 << 8) | (1 << 10) | 0b01);
        assert_eq!(descriptor(2, MIB2, device_attrs()), MIB2 | (1 << 2) | (1 << 10) | 0b01);
    }

    #[test_case]
    fn descriptor_masks_out_stray_caller_bits() {
        // `attrs` must not be able to smuggle in an address or type bits: a
        // caller passing a whole descriptor back in as attributes would
        // otherwise relocate the mapping.
        let d = descriptor(3, 0x1000, u64::MAX);
        assert_eq!(descriptor_addr(d), 0x1000, "address comes only from `pa`");
        assert_eq!(d & 0b11, TABLE, "type bits come only from the level");
        // An unaligned `pa` is truncated to its frame, never allowed to bleed
        // into the attribute bits.
        assert_eq!(descriptor_addr(descriptor(3, 0x1fff, 0)), 0x1000);
    }

    #[test_case]
    fn a_table_descriptor_carries_no_memory_attributes() {
        let t = table_descriptor(0x4321_f000);
        assert_eq!(descriptor_addr(t), 0x4321_f000);
        assert_eq!(descriptor_attrs(t), 0, "attributes live on the leaf, not the table");
        assert!(is_valid(t));
        assert!(is_table(t, 1) && is_table(t, 2));
    }

    #[test_case]
    fn invalid_descriptors_are_recognised_as_unmapped() {
        assert!(!is_valid(0));
        // Bit pattern with every attribute set but the type bits clear: still
        // invalid. (Zeroed tables read as 0, but a broken-for-BBM entry can
        // retain attributes while its type bits are cleared.)
        assert!(!is_valid(ATTR_MASK | ADDR_MASK));
        assert!(!is_leaf(0, 3) && !is_table(0, 1));
    }

    #[test_case]
    fn walk_step_distinguishes_the_three_things_a_descriptor_can_be() {
        assert_eq!(walk_step(0, 1), Step::NeedTable);
        assert_eq!(walk_step(table_descriptor(0x9000), 1), Step::Descend(0x9000));
        assert_eq!(walk_step(table_descriptor(0x9000), 2), Step::Descend(0x9000));
        assert_eq!(walk_step(descriptor(1, GIB, normal_attrs()), 1), Step::SplitBlock);
        assert_eq!(walk_step(descriptor(2, MIB2, device_attrs()), 2), Step::SplitBlock);
    }

    #[test_case]
    fn splitting_a_2mib_block_yields_512_contiguous_pages() {
        let base = 0x4020_0000;
        let block = descriptor(2, base, normal_attrs());
        for i in [0usize, 1, 255, 511] {
            let child = split_child(2, block, i);
            assert_eq!(descriptor_addr(child), base + i as u64 * 4096, "child {i} address");
            assert_eq!(child & 0b11, TABLE, "child {i} must be a page, not a block");
            assert_eq!(descriptor_attrs(child), normal_attrs(), "child {i} keeps the memory type");
        }
        // The split covers exactly the parent's range: no gap, no overlap.
        let last = split_child(2, block, ENTRIES - 1);
        assert_eq!(descriptor_addr(last) + 4096, base + MIB2);
    }

    #[test_case]
    fn splitting_a_1gib_block_yields_512_2mib_blocks() {
        let base = 2 * GIB;
        let block = descriptor(1, base, device_attrs());
        let child = split_child(1, block, 3);
        assert_eq!(descriptor_addr(child), base + 3 * MIB2);
        assert_eq!(child & 0b11, BLOCK, "an L2 child of an L1 block is still a block");
        assert_eq!(descriptor_attrs(child), device_attrs(), "Device memory stays Device");
    }

    #[test_case]
    fn splitting_drops_the_contiguous_hint() {
        // The parent's Contiguous bit describes an aligned run of 16 entries at
        // the *parent's* level. Carried down it would tell the TLB it may cache
        // 16 child entries as one, which the table does not guarantee.
        let block = descriptor(2, 0x4020_0000, normal_attrs() | CONTIGUOUS);
        assert_eq!(split_child(2, block, 0) & CONTIGUOUS, 0);
    }

    #[test_case]
    fn user_and_permission_bits_are_the_documented_positions() {
        // AP[1] = bit 6 (EL0 access), AP[2] = bit 7 (read-only). Getting these
        // swapped makes a "read-only user page" a read-write kernel-only one —
        // no fault, no error, just no isolation.
        assert_eq!(AP_USER, 1 << 6);
        assert_eq!(AP_RO, 1 << 7);
        let user_ro = descriptor(3, 0x1000, normal_attrs() | AP_USER | AP_RO | UXN | PXN);
        assert_ne!(user_ro & AP_USER, 0);
        assert_ne!(user_ro & AP_RO, 0);
        // Both execute-never bits survive the attribute mask (they are upper
        // attributes, bits 53/54 — a mask covering only bits 11:2 would drop
        // them and hand userspace an executable data page).
        assert_ne!(user_ro & PXN, 0);
        assert_ne!(user_ro & UXN, 0);
    }

    #[test_case]
    fn attributes_survive_a_descriptor_round_trip() {
        // `descriptor_attrs` feeding `descriptor` must be the identity for every
        // attribute the kernel sets — that round trip is what `split_child` and
        // any future permission change rely on.
        let attrs = normal_attrs() | AP_USER | AP_RO | NG | PXN | UXN;
        let d = descriptor(3, 0x7fff_f000, attrs);
        assert_eq!(descriptor_attrs(d), attrs);
        assert_eq!(descriptor(3, descriptor_addr(d), descriptor_attrs(d)), d);
    }
}
