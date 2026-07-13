//! Pure physical-RAM-layout math for the identity map: given the firmware's
//! actual RAM regions, decide how each 1 GiB block (and, inside a *mixed*
//! block, each 2 MiB chunk) must be typed. Real machines and VirtualBox-ARM
//! interleave RAM and MMIO inside one GiB block — e.g. VBox puts the tail of
//! low RAM (where the stub legitimately allocates the model), the GOP
//! framebuffer aperture (`0xd8000000`) and the PCIe ECAM (`0xfeddd000`) all in
//! the `0xC0000000` block — so typing whole GiB blocks is not expressible:
//! Normal over MMIO breaks/asserts, Device over RAM alignment-faults the
//! vector loads of the SDOT matvecs (the "/perf FATAL at FAR=0xc0000000").
//! Arch-neutral so the x86 unit suite exercises it.

/// How a 1 GiB block must be mapped, from the RAM regions that intersect it.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum BlockKind {
    /// Entirely RAM → one Normal cacheable block descriptor.
    Normal,
    /// No RAM at all → Device (MMIO may live anywhere in it).
    Device,
    /// RAM and non-RAM interleave → needs an L2 table of 2 MiB chunks.
    Mixed,
}

/// True if `[base, base+len)` lies entirely inside one of `regions`
/// (`(base, size)` pairs; need not be sorted). Zero-size regions are ignored.
pub fn range_is_ram(base: u64, len: u64, regions: &[(u64, u64)]) -> bool {
    let end = match base.checked_add(len) {
        Some(e) => e,
        None => return false,
    };
    regions
        .iter()
        .filter(|(_, s)| *s > 0)
        .any(|&(rb, rs)| base >= rb && end <= rb.saturating_add(rs))
}

/// True if `[base, base+len)` overlaps any of `regions` at all.
pub fn range_touches_ram(base: u64, len: u64, regions: &[(u64, u64)]) -> bool {
    let end = base.saturating_add(len);
    regions
        .iter()
        .filter(|(_, s)| *s > 0)
        .any(|&(rb, rs)| base < rb.saturating_add(rs) && rb < end)
    }

/// Classify the 1 GiB block starting at `block_base`.
pub fn classify_gib(block_base: u64, regions: &[(u64, u64)]) -> BlockKind {
    const GIB: u64 = 1 << 30;
    if range_is_ram(block_base, GIB, regions) {
        BlockKind::Normal
    } else if range_touches_ram(block_base, GIB, regions) {
        BlockKind::Mixed
    } else {
        BlockKind::Device
    }
}

/// For a chunk (2 MiB granule) inside a mixed block: Normal iff **fully** RAM.
/// A partially-RAM 2 MiB chunk types as Device — RAM regions are page-granular
/// so at worst <2 MiB of RAM at a region edge becomes uncached-but-correct,
/// while the reverse (Normal over MMIO) would be broken.
pub fn chunk_is_normal(chunk_base: u64, regions: &[(u64, u64)]) -> bool {
    const CHUNK: u64 = 2 << 20;
    range_is_ram(chunk_base, CHUNK, regions)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The VirtualBox-ARM shape that broke: low RAM 0x4000_0000..0xD200_0000
    /// (contains the model tail crossing 0xC000_0000), MMIO hole with the fb
    /// (0xd800_0000) + ECAM (0xfedd_d000), more RAM above 4 GiB.
    const VBOX: [(u64, u64); 2] = [
        (0x4000_0000, 0x9200_0000),   // → 0xD200_0000
        (0x1_0000_0000, 0x3C00_0000), // ~960 MiB high
    ];

    #[test_case]
    fn vbox_blocks_classify() {
        assert_eq!(classify_gib(0x0000_0000, &VBOX), BlockKind::Device); // MMIO low
        assert_eq!(classify_gib(0x4000_0000, &VBOX), BlockKind::Normal);
        assert_eq!(classify_gib(0x8000_0000, &VBOX), BlockKind::Normal);
        assert_eq!(classify_gib(0xC000_0000, &VBOX), BlockKind::Mixed); // RAM tail + fb + ECAM
        assert_eq!(classify_gib(0x1_0000_0000, &VBOX), BlockKind::Mixed); // high RAM, partial GiB
        assert_eq!(classify_gib(0x2_0000_0000, &VBOX), BlockKind::Device);
    }

    #[test_case]
    fn vbox_mixed_block_chunks() {
        // Model tail at 0xC000_0000 → RAM → Normal (the /perf crash site).
        assert!(chunk_is_normal(0xC000_0000, &VBOX));
        assert!(chunk_is_normal(0xCFE0_0000, &VBOX));
        // Framebuffer + ECAM chunks → Device.
        assert!(!chunk_is_normal(0xD800_0000, &VBOX));
        assert!(!chunk_is_normal(0xFEC0_0000, &VBOX));
        // The 2 MiB-aligned region edge itself is fully RAM → Normal.
        assert!(chunk_is_normal(0xD1E0_0000, &VBOX));
        // With a page-granular (non-2 MiB-aligned) region end, the straddling
        // chunk types Device — the safe side.
        let ragged = [(0x4000_0000u64, 0x91FF_0000u64)]; // ends 0xD1FF_0000
        assert!(!chunk_is_normal(0xD1E0_0000, &ragged));
        assert!(chunk_is_normal(0xD1C0_0000, &ragged));
    }

    #[test_case]
    fn apple_silicon_high_ram_base() {
        // Apple Silicon (via m1n1) places system RAM at 0x8_0000_0000 (32 GiB),
        // not QEMU's 0x4000_0000. The ~32 GiB of address space below it is
        // unbacked / low MMIO and MUST type as Device — mapping it Normal would
        // let the core speculatively touch an unbacked address and fault. The
        // RAM blocks themselves stay Normal so code/heap/stack run cacheable.
        let m2: [(u64, u64); 1] = [(0x8_0000_0000, 16u64 << 30)]; // 16 GiB Mac Mini
        // Below the base: Device, never Normal.
        assert_eq!(classify_gib(0x0000_0000, &m2), BlockKind::Device);
        assert_eq!(classify_gib(0x4000_0000, &m2), BlockKind::Device); // QEMU's base — unbacked here
        assert_eq!(classify_gib(0x2_0000_0000, &m2), BlockKind::Device);
        assert_eq!(classify_gib(0x7_C000_0000, &m2), BlockKind::Device); // last GiB before RAM
        // The RAM span [32 GiB, 48 GiB): all Normal.
        assert_eq!(classify_gib(0x8_0000_0000, &m2), BlockKind::Normal);
        assert_eq!(classify_gib(0xB_8000_0000, &m2), BlockKind::Normal);
        assert_eq!(classify_gib(0xB_C000_0000, &m2), BlockKind::Normal); // last RAM GiB
        // Past the top: Device again.
        assert_eq!(classify_gib(0xC_0000_0000, &m2), BlockKind::Device);
    }

    #[test_case]
    fn contiguous_qemu_layout_stays_all_normal() {
        // QEMU virt: one contiguous clump — every covered block is Normal, so
        // the mixed path never engages (legacy behaviour preserved).
        let qemu = [(0x4000_0000u64, 3u64 << 30)];
        assert_eq!(classify_gib(0x4000_0000, &qemu), BlockKind::Normal);
        assert_eq!(classify_gib(0x8000_0000, &qemu), BlockKind::Normal);
        assert_eq!(classify_gib(0xC000_0000, &qemu), BlockKind::Normal);
        assert_eq!(classify_gib(0x1_0000_0000, &qemu), BlockKind::Device);
    }

    #[test_case]
    fn edge_cases() {
        assert!(!range_is_ram(0, 0x1000, &[]));
        assert!(!range_touches_ram(0, 0x1000, &[]));
        // Zero-size region ignored; overflow-safe.
        assert!(!range_is_ram(u64::MAX - 4096, 8192, &[(0, u64::MAX)]));
        assert!(range_is_ram(0x4000_0000, 0x1000, &[(0x4000_0000, 0x1000), (0, 0)]));
    }
}
