//! 4-level (PML4) paging: page-table entry/table types, a walk-and-map
//! function that allocates missing intermediate tables on demand, and the
//! higher-half direct map (HHDM) offset used to reach physical frames by
//! virtual address.
//!
//! Limine hands the kernel control with paging already enabled (a
//! higher-half mapping for the kernel image, plus the HHDM covering all
//! usable physical memory). This module *extends* those tables — walking
//! the current `CR3` and inserting new mappings — rather than replacing
//! them from scratch; Phase 1 only needs this to back the kernel heap
//! (`mm::heap`) with real physical frames, not a from-scratch address
//! space (that's a Phase 2 concern, once tasks need isolated spaces).

use core::arch::asm;
use core::sync::atomic::{AtomicU64, Ordering};

static HHDM_OFFSET: AtomicU64 = AtomicU64::new(u64::MAX);

/// Record the HHDM offset Limine reported (see `chitti_kernel::init`).
/// Must be called before any other function in this module.
pub fn set_hhdm_offset(offset: u64) {
    HHDM_OFFSET.store(offset, Ordering::SeqCst);
}

fn hhdm_offset() -> u64 {
    let offset = HHDM_OFFSET.load(Ordering::SeqCst);
    debug_assert_ne!(offset, u64::MAX, "paging: used before set_hhdm_offset");
    offset
}

/// Translate a physical address to the virtual address it's reachable at
/// through the HHDM.
pub fn phys_to_virt(phys: u64) -> u64 {
    phys + hhdm_offset()
}

pub const PRESENT: u64 = 1 << 0;
pub const WRITABLE: u64 = 1 << 1;
/// Requires `arch::x86_64::fpu::enable_nx()` (`EFER.NXE`) to have run
/// first: without it this bit is reserved and using it faults.
pub const NO_EXECUTE: u64 = 1 << 63;

const ADDR_MASK: u64 = 0x000f_ffff_ffff_f000;

#[repr(transparent)]
#[derive(Clone, Copy)]
struct PageTableEntry(u64);

impl PageTableEntry {
    const fn empty() -> Self {
        Self(0)
    }

    fn is_present(&self) -> bool {
        self.0 & PRESENT != 0
    }

    fn addr(&self) -> u64 {
        self.0 & ADDR_MASK
    }

    fn set(&mut self, addr: u64, flags: u64) {
        self.0 = (addr & ADDR_MASK) | flags;
    }
}

#[repr(C, align(4096))]
struct PageTable {
    entries: [PageTableEntry; 512],
}

/// A source of fresh, zeroed physical frames for new page-table levels.
/// Implemented by `mm::frame::BitmapFrameAllocator` (via an adapter in
/// `mm::heap`) so this module stays decoupled from the concrete
/// allocator's locking strategy.
pub trait FrameAllocator {
    fn allocate_frame(&mut self) -> Option<u64>;
}

fn table_at(phys: u64) -> &'static mut PageTable {
    // SAFETY: `phys` is always either the current CR3 or a present page
    // table entry's address, i.e. a physical frame Limine or this module
    // itself allocated as a page table; `phys_to_virt` reaches it through
    // the HHDM, which covers all usable physical memory.
    unsafe { &mut *(phys_to_virt(phys) as *mut PageTable) }
}

fn zero_table(table: &mut PageTable) {
    for entry in table.entries.iter_mut() {
        *entry = PageTableEntry::empty();
    }
}

fn read_cr3() -> u64 {
    let v: u64;
    // SAFETY: reading CR3 has no side effects.
    unsafe { asm!("mov {}, cr3", out(reg) v, options(nomem, nostack, preserves_flags)) };
    v & ADDR_MASK
}

fn invlpg(virt: u64) {
    // SAFETY: invalidates the TLB entry for `virt`; always safe, at worst
    // costs a future TLB miss.
    unsafe { asm!("invlpg [{}]", in(reg) virt, options(nostack, preserves_flags)) };
}

/// Index into the page-table level `level` (3=PML4, 2=PDPT, 1=PD, 0=PT)
/// that `virt` falls under.
fn table_index(virt: u64, level: u64) -> usize {
    ((virt >> (12 + 9 * level)) & 0x1ff) as usize
}

/// Map the single page at `virt` (must be 4 KiB-aligned) to `phys` with
/// `flags`, allocating any missing PML4/PDPT/PD entries via `alloc`.
/// Walks from the *current* `CR3`.
pub fn map_page(virt: u64, phys: u64, flags: u64, alloc: &mut impl FrameAllocator) {
    debug_assert_eq!(virt % 4096, 0);
    debug_assert_eq!(phys % 4096, 0);

    let mut table = table_at(read_cr3());
    for level in (1..=3u64).rev() {
        let idx = table_index(virt, level);
        let entry = &mut table.entries[idx];
        if !entry.is_present() {
            let new_frame = alloc
                .allocate_frame()
                .expect("map_page: out of physical frames for page tables");
            zero_table(table_at(new_frame));
            entry.set(new_frame, PRESENT | WRITABLE);
        }
        table = table_at(entry.addr());
    }

    let idx = table_index(virt, 0);
    table.entries[idx].set(phys, flags);
    invlpg(virt);
}
