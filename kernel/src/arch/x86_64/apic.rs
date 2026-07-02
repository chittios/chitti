//! Local APIC (xAPIC), the per-core interrupt controller
//! (`CHITTI_OS_HANDOFF.md` Phase 7: "APIC per core"). Limine starts every
//! core with its local APIC hardware-enabled (`IA32_APIC_BASE.EN`); this
//! module reads the per-core APIC id and *software*-enables the APIC (sets
//! the spurious-interrupt-vector register), which every core does as it comes
//! online.
//!
//! Only the pieces SMP bring-up needs are here. The application processors run
//! a cooperative, interrupts-disabled worker on each core (`crate::smp`), so
//! no IO-APIC redirection or per-core APIC-timer programming is required yet;
//! the legacy PIC/PIT (`pic.rs`/`pit.rs`) still drives the BSP's timer and
//! keyboard IRQs, untouched. Wiring the APIC timer / IPIs is future work.

use core::arch::asm;
use core::sync::atomic::{AtomicU64, Ordering};

/// `IA32_APIC_BASE` MSR: bits 12..=35 hold the local APIC's physical base
/// (0xFEE00000 on reset), bit 11 is the global enable Limine leaves set.
const IA32_APIC_BASE: u32 = 0x1b;

/// Register offsets within the local APIC's 4 KiB MMIO page.
const REG_ID: u64 = 0x20;
const REG_SPURIOUS: u64 = 0xf0;

/// Spurious-interrupt vector we point the APIC at; the vector is never
/// expected to fire (APs keep interrupts disabled), but the register's bit 8
/// is the APIC software-enable, so it must be written to bring the APIC fully
/// online.
const SPURIOUS_VECTOR: u32 = 0xff;
const APIC_SOFTWARE_ENABLE: u32 = 1 << 8;

/// Virtual address the local APIC MMIO page is mapped at, cached after
/// `init_mapping`. The local APIC lives at the same physical address on every
/// core, so one mapping (in the page tables all cores share) serves them all.
static APIC_VIRT: AtomicU64 = AtomicU64::new(0);

fn read_msr(msr: u32) -> u64 {
    let (lo, hi): (u32, u32);
    // SAFETY: reading IA32_APIC_BASE is valid on any x86_64 CPU in long mode.
    unsafe { asm!("rdmsr", in("ecx") msr, out("eax") lo, out("edx") hi, options(nomem, nostack, preserves_flags)) };
    ((hi as u64) << 32) | lo as u64
}

/// Map the local APIC MMIO page (it sits in a hole the HHDM does not cover)
/// and cache its virtual address. Must run once on the BSP after `mm::init`
/// and before any `local_id`/`software_enable`; APs reuse the shared mapping.
pub fn init_mapping() {
    let phys = read_msr(IA32_APIC_BASE) & 0xffff_f000;
    let virt = crate::mm::map_mmio_page(phys);
    APIC_VIRT.store(virt, Ordering::SeqCst);
}

fn read_reg(offset: u64) -> u32 {
    let base = APIC_VIRT.load(Ordering::SeqCst);
    debug_assert!(base != 0, "apic: used before init_mapping");
    // SAFETY: `base + offset` is the mapped MMIO address of a valid local-APIC
    // register; APIC registers are 32-bit and must be accessed volatile.
    unsafe { core::ptr::read_volatile((base + offset) as *const u32) }
}

fn write_reg(offset: u64, value: u32) {
    let base = APIC_VIRT.load(Ordering::SeqCst);
    debug_assert!(base != 0, "apic: used before init_mapping");
    // SAFETY: as `read_reg`; the spurious-vector register is writable.
    unsafe { core::ptr::write_volatile((base + offset) as *mut u32, value) };
}

/// This core's local-APIC id (bits 24..=31 of the id register, for xAPIC).
pub fn local_id() -> u32 {
    read_reg(REG_ID) >> 24
}

/// Software-enable this core's local APIC. Idempotent; called by every core
/// (BSP and APs) as it comes online. Requires `init_mapping` to have run.
pub fn software_enable() {
    let spurious = read_reg(REG_SPURIOUS);
    write_reg(REG_SPURIOUS, spurious | APIC_SOFTWARE_ENABLE | SPURIOUS_VECTOR);
}
