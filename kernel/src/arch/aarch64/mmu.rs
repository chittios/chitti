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
