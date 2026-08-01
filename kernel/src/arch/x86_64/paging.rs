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
/// Ring 3 may access the mapping. Absent from every flag combination the kernel
/// maps today (heap and MMIO are both supervisor-only), and the counterpart of
/// aarch64's [`crate::mm::armv8::AP_USER`].
///
/// **It must be set on every level of the walk, not just the leaf.** The x86
/// permission check ANDs `U/S` down the whole path, so a user page reached
/// through a supervisor-only PDPT/PD/PT is unreachable from ring 3 — a
/// page fault with no indication of which level refused. [`map_page`] therefore
/// propagates this bit into the intermediate tables it creates.
pub const USER_ACCESSIBLE: u64 = 1 << 2;
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

/// A source of fresh, zeroed physical frames for new page-table levels. Now
/// [`crate::mm::frame::TableFrames`], shared with aarch64's walker so the two
/// arches allocate intermediate tables through one trait; re-exported here under
/// the name this module has always used.
pub use crate::mm::frame::TableFrames as FrameAllocator;

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

/// The active page-table root.
///
/// Public because the S3 resume path has to save it before firmware destroys `CR3` and
/// put it back afterwards — and it must be the *same* value the kernel is running on,
/// not a re-derived one.
pub fn active_cr3() -> u64 {
    read_cr3()
}

fn read_cr3() -> u64 {
    let v: u64;
    // SAFETY: reading CR3 has no side effects.
    unsafe { asm!("mov {}, cr3", out(reg) v, options(nomem, nostack, preserves_flags)) };
    v & ADDR_MASK
}

/// Build a fresh top-level table that **shares every kernel mapping** and has an
/// empty user half, returning its physical address.
///
/// On x86 that is exactly a PML4 whose upper 256 entries are copied from the
/// kernel's: the kernel image and Limine's HHDM both live in the higher half, so
/// copying those entries shares the whole kernel address space by reference —
/// later kernel mappings appear in every space automatically, because they are
/// installed *below* the copied entries. The lower 256 entries start empty and
/// belong to the task.
///
/// # Safety
/// `frame` must be an exclusively-owned 4 KiB frame reachable through the HHDM.
pub unsafe fn new_root_sharing_kernel(frame: u64) -> u64 {
    let src = table_at(read_cr3());
    let dst = table_at(frame);
    for i in 0..512 {
        // Upper half = kernel (higher-half image + HHDM); lower half = user.
        dst.entries[i] = if i >= 256 { src.entries[i] } else { PageTableEntry::empty() };
    }
    frame
}

/// The physical address `virt` translates to in the tree rooted at physical
/// `root`, or `None` if nothing maps it. 4 KiB pages only — this kernel maps
/// nothing with the 2 MiB/1 GiB page-size bit, so a large page here would be a
/// bug rather than a case to resolve. Mirrors aarch64's
/// [`crate::mm::walk::translate`], and exists for the same reason: confirming a
/// mapping took, rather than discovering it did not via a fault.
pub fn translate_in(root: u64, virt: u64) -> Option<u64> {
    let mut table = table_at(root);
    for level in (1..=3u64).rev() {
        let entry = table.entries[table_index(virt, level)];
        if !entry.is_present() {
            return None;
        }
        table = table_at(entry.addr());
    }
    let leaf = table.entries[table_index(virt, 0)];
    if !leaf.is_present() {
        return None;
    }
    Some(leaf.addr() | (virt & 0xfff))
}

/// Switch the active address space to the tree rooted at physical `root`.
///
/// # Safety
/// `root` must be a valid top-level table that maps all currently-executing
/// kernel code, data and the current stack — which
/// [`new_root_sharing_kernel`] guarantees by copying the kernel half.
pub unsafe fn activate_root(root: u64) {
    // SAFETY: caller's contract; writing CR3 replaces the address space and
    // flushes non-global TLB entries.
    unsafe { asm!("mov cr3, {}", in(reg) root, options(nostack, preserves_flags)) };
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
///
/// When `flags` requests [`USER_ACCESSIBLE`], the bit is also set on every
/// intermediate entry along the walk — created or pre-existing — because the
/// hardware ANDs `U/S` down the path (see that constant). Widening an
/// intermediate is safe: the leaf's own `U/S` still decides, so kernel pages
/// sharing the table stay supervisor-only.
pub fn map_page(virt: u64, phys: u64, flags: u64, alloc: &mut impl FrameAllocator) {
    map_page_in(read_cr3(), virt, phys, flags, alloc)
}

/// Like [`map_page`], but into the tree rooted at physical `root` rather than the
/// one `CR3` currently points at. This is what per-task address spaces are built
/// with ([`crate::mm::space`]): a fresh PML4 that shares the kernel's higher-half
/// entries and gets its own lower-half user mappings.
pub fn map_page_in(root: u64, virt: u64, phys: u64, flags: u64, alloc: &mut impl FrameAllocator) {
    debug_assert_eq!(virt % 4096, 0);
    debug_assert_eq!(phys % 4096, 0);

    let intermediate = PRESENT | WRITABLE | (flags & USER_ACCESSIBLE);
    let mut table = table_at(root);
    for level in (1..=3u64).rev() {
        let idx = table_index(virt, level);
        let entry = &mut table.entries[idx];
        if !entry.is_present() {
            let new_frame = alloc
                .allocate_frame()
                .expect("map_page: out of physical frames for page tables");
            zero_table(table_at(new_frame));
            entry.set(new_frame, intermediate);
        } else {
            entry.0 |= intermediate & USER_ACCESSIBLE;
        }
        table = table_at(entry.addr());
    }

    let idx = table_index(virt, 0);
    table.entries[idx].set(phys, flags);
    invlpg(virt);
}
