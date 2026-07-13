//! aarch64's view of the flattened-device-tree reader. The parsing itself is
//! arch-neutral pure logic in [`crate::fdt`] (so the host unit suite can cover
//! it); this module just re-exports it under the `dtb::` name the aarch64 boot
//! path (`mmu`, `boot`) has always used, and is where any aarch64-specific DTB
//! glue would live.

pub use crate::fdt::*;
