//! Minimal aarch64 MMU bring-up: an identity map over the low 4 GiB with 1 GiB
//! blocks (RAM = Normal write-back cacheable, low 1 GiB = Device for the PL011
//! UART / GIC), then MMU + I/D caches on. QEMU enters with the MMU off, where
//! all memory is device-typed and uncached -- NEON is unreliable and slow
//! there -- so this must run before any NEON/cached work.

use core::arch::asm;

#[repr(align(4096))]
struct Table(#[allow(dead_code)] [u64; 512]);
static mut L1: Table = Table([0; 512]);

/// Set up the identity map and enable the MMU + caches. Idempotent-ish; call
/// once, early, on the boot core.
pub fn init() {
    // SAFETY: single-core boot; builds a valid identity map and programs the
    // standard EL1 translation registers. VA==PA, so stack/code/UART stay valid.
    unsafe {
        let l1 = core::ptr::addr_of_mut!(L1) as *mut u64;
        for i in 0..4u64 {
            let pa = i << 30; // 1 GiB blocks
            let attr_idx = if i == 0 { 1u64 } else { 0u64 }; // 0: Device MMIO, else Normal
            let sh = if i == 0 { 0u64 } else { 0b11u64 }; // inner-shareable for Normal
            let desc = pa | (attr_idx << 2) | (sh << 8) | (1 << 10) | 0b01; // AF=1, block, valid
            *l1.add(i as usize) = desc;
        }
        enable_mmu(l1);
    }
}

/// Enable the MMU + caches on a secondary core, reusing the BSP's already-built
/// identity map (`L1`). A secondary starts (via PSCI `CPU_ON`) with the MMU
/// off, where RAM is Device-typed and atomics/`Locked` can't complete -- so
/// this must run before the core touches any shared, lock-guarded structure.
/// The translation table is shared read-only across cores; only the per-core
/// system registers are programmed here.
///
/// # Safety
/// Must run exactly once per secondary core, before any cached/atomic access,
/// with the `L1` table already initialized by the BSP's `init`.
pub unsafe fn enable_secondary() {
    // SAFETY: `L1` is a valid, BSP-initialized identity map; programming the
    // per-core translation registers to it keeps VA==PA (stack/code stay live).
    unsafe { enable_mmu(core::ptr::addr_of_mut!(L1) as *mut u64) };
}

/// Program the EL1 translation registers to `l1` and turn the MMU + I/D caches
/// on. Shared by the BSP (`init`, after building the table) and each secondary
/// (`enable_secondary`, reusing it). The register values are identical on every
/// core (the map is global), so this is deterministic.
///
/// # Safety
/// `l1` must point at a valid, populated L1 translation table for a 39-bit
/// identity map; caller ensures VA==PA so the running stack/code stay mapped.
unsafe fn enable_mmu(l1: *mut u64) {
    // SAFETY: caller's contract; these are the standard EL1 MMU registers.
    unsafe {
        // MAIR: attr0 = Normal write-back (0xFF), attr1 = Device nGnRnE (0x00).
        let mair: u64 = 0xFF;
        // TCR: T0SZ=25 (39-bit VA), 4 KiB granule, WB cacheable walks,
        // inner-shareable, TTBR1 disabled, 40-bit PA.
        let tcr: u64 = 25 | (1 << 8) | (1 << 10) | (3 << 12) | (1 << 23) | (2u64 << 32);
        asm!("msr mair_el1, {}", in(reg) mair, options(nostack));
        asm!("msr tcr_el1, {}", in(reg) tcr, options(nostack));
        asm!("msr ttbr0_el1, {}", in(reg) l1 as u64, options(nostack));
        asm!("dsb ish", "tlbi vmalle1", "dsb ish", "isb", options(nostack));
        let mut sctlr: u64;
        asm!("mrs {}, sctlr_el1", out(reg) sctlr, options(nostack));
        sctlr |= (1 << 0) | (1 << 2) | (1 << 12); // M (MMU), C (data cache), I (instr cache)
        asm!("msr sctlr_el1, {}", in(reg) sctlr, options(nostack));
        asm!("isb", options(nostack));
    }
}
