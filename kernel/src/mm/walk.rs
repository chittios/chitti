//! A 4 KiB page-table walker over ARMv8-A translation tables: map, unmap and
//! translate a single page, allocating intermediate levels on demand and
//! splitting a block descriptor into a finer table when one is in the way.
//!
//! The x86 counterpart is [`crate::arch::x86_64::paging::map_page`], which walks
//! the live `CR3` directly. This one is written against a trait instead, for one
//! reason worth the indirection: the unit suite runs on x86 only, so a walker
//! living in `arch/aarch64` could be compiled but never *executed* by a test.
//! Here the descriptor manipulation, the allocation of new levels and — above
//! all — the **break-before-make ordering** are exercised against real tables
//! built in the heap, on x86, by `cargo xtask test`.
//!
//! That ordering is the reason this is not a place to be clever. Replacing a
//! live valid descriptor with a different valid one in place is a TLB conflict
//! abort on real Apple cores while QEMU and hypervisors quietly tolerate it —
//! the exact trap [`crate::arch::aarch64::mmu::map_device_gib`] was fixed for,
//! found only on a bare Mac. A test that asserts the sequence is the only way
//! that stays fixed.
//!
//! Physical addresses are turned into pointers by [`Mmu::table_ptr`], whose
//! default is the identity — true for aarch64's VA == PA map today, and the seam
//! to change if the kernel ever moves to a TTBR1 high-half mapping.

use super::armv8::{self, Step};
use super::frame::TableFrames;

/// The barriers, TLB maintenance and physical-to-pointer translation a walker
/// needs from the platform.
///
/// Split out so the walk itself is testable: the test implementation records the
/// calls instead of executing them, which is what makes break-before-make
/// *ordering* assertable rather than merely reviewed.
pub trait Mmu {
    /// Where the translation table at physical address `pa` is readable and
    /// writable. Identity by default (aarch64 runs VA == PA).
    ///
    /// Not a formality even on an identity map: a descriptor stores only bits
    /// 47:12 of an address ([`armv8::ADDR_MASK`]), so any address that does not
    /// fit that field comes back out of [`armv8::descriptor_addr`] silently
    /// truncated. The tests below rely on this hook for exactly that reason —
    /// their tables are x86 heap allocations above 2^48 — and a TTBR1 high-half
    /// kernel would too.
    fn table_ptr(&self, pa: u64) -> *mut u64 {
        pa as *mut u64
    }
    /// Make prior descriptor writes visible to the hardware table walker,
    /// before any TLB operation that depends on them (`dsb ishst`).
    fn publish(&self);
    /// Invalidate the TLB for the single page at `va`.
    fn invalidate_page(&self, va: u64);
    /// Invalidate every TLB entry. Needed after a **block** split, which
    /// changes the translation of a whole 1 GiB or 2 MiB range at once — more
    /// virtual addresses than a by-address invalidate covers.
    fn invalidate_all(&self);
}

/// Whether the walker may break a live block descriptor apart to install a finer
/// mapping inside it.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Split {
    /// Split it. Only correct on a table the caller owns exclusively and is not
    /// executing out of: between the break and the make, the block's whole range
    /// is unmapped, so any access to it — from this core or another, code, stack
    /// or heap — faults.
    Allow,
    /// Refuse, reporting [`MapError::WouldSplitBlock`]. The right answer for the
    /// live kernel identity map, where the range being briefly unmapped is
    /// running code.
    Refuse,
}

/// Why a [`map_page`] failed. Every variant is a refusal to do something unsafe
/// or impossible, never a partially-applied mapping: on error the table is
/// exactly as it was, except for intermediate levels that were legitimately
/// created on the way down (which are correct, empty, and reused next time).
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum MapError {
    /// `va` is outside the virtual-address width the translation regime covers.
    VaOutOfRange,
    /// `va` or `pa` was not 4 KiB-aligned.
    Misaligned,
    /// The frame allocator has no frame left for an intermediate table.
    OutOfFrames,
    /// There is no frame allocator at all. Reachable on aarch64, whose boot
    /// paths do not all yield a trustworthy free-memory pool (see
    /// `mm::aarch64_frames`) — so a platform wrapper may have no frame source to
    /// pass in, which is a refusal rather than an exhaustion.
    NoFrameAllocator,
    /// A block descriptor covers `va` and [`Split::Refuse`] was in force.
    /// `base`/`level` name the block, so the log says which mapping is in the way.
    WouldSplitBlock { level: u32, base: u64 },
}

/// Allocate and zero a frame to use as a translation table.
fn fresh_table(mmu: &impl Mmu, alloc: &mut impl TableFrames) -> Result<u64, MapError> {
    let pa = alloc.allocate_frame().ok_or(MapError::OutOfFrames)?;
    // SAFETY: a freshly-allocated, exclusively-owned 4 KiB frame, reachable at
    // `table_ptr(pa)` by this trait's contract. A table must start zeroed: any
    // stale nonzero word would read as a valid descriptor pointing at whatever
    // used the frame before.
    unsafe { core::ptr::write_bytes(mmu.table_ptr(pa) as *mut u8, 0, 4096) };
    Ok(pa)
}

/// Read entry `idx` of the table at physical `pa`.
///
/// # Safety
/// `pa` must be a live translation table reachable through `mmu`, `idx < 512`.
unsafe fn get(mmu: &impl Mmu, pa: u64, idx: usize) -> u64 {
    // SAFETY: caller's contract; a table is 512 u64s.
    unsafe { mmu.table_ptr(pa).add(idx).read_volatile() }
}

/// Write entry `idx` of the table at physical `pa`.
///
/// # Safety
/// As [`get`].
unsafe fn put(mmu: &impl Mmu, pa: u64, idx: usize, desc: u64) {
    // SAFETY: caller's contract.
    unsafe { mmu.table_ptr(pa).add(idx).write_volatile(desc) }
}

/// Replace the block descriptor at `table[idx]` with a table of `ENTRIES`
/// next-level descriptors covering exactly the same memory with exactly the same
/// attributes, and return the new table's physical address.
///
/// **Break-before-make, in this order and no other.** The child table is built
/// completely first, so the window in which anything is unmapped is as short as
/// possible; then the parent entry is invalidated, the TLB is flushed *wholly*
/// (the block spans more addresses than a by-address invalidate reaches), and
/// only then is the table descriptor published. Writing the table descriptor
/// over the live block in place — the obvious shortcut — is a valid-to-valid
/// change of a live translation, which real Apple cores fault and emulators do
/// not.
///
/// The "make" needs no TLB operation of its own: the entry it overwrites was
/// just made invalid and flushed, so this is an invalid-to-valid transition, and
/// the descriptor it installs describes the *same* translation the block did.
/// Ordering the store is therefore enough, and the caller's own terminal
/// [`Mmu::invalidate_page`] provides the context synchronization.
///
/// # Safety
/// `table[idx]` must be a valid block descriptor at `level`, and the caller must
/// have established that unmapping its range for the duration is safe (see
/// [`Split`]).
unsafe fn split_block(
    mmu: &impl Mmu,
    table: u64,
    idx: usize,
    level: u32,
    alloc: &mut impl TableFrames,
) -> Result<u64, MapError> {
    // SAFETY: caller's contract.
    let block = unsafe { get(mmu, table, idx) };
    let child = fresh_table(mmu, alloc)?;
    for i in 0..armv8::ENTRIES {
        // SAFETY: `child` is a fresh 4 KiB table; `i < ENTRIES`.
        unsafe { put(mmu, child, i, armv8::split_child(level, block, i)) };
    }
    mmu.publish();
    // Break.
    // SAFETY: caller's contract.
    unsafe { put(mmu, table, idx, armv8::INVALID) };
    mmu.publish();
    mmu.invalidate_all();
    // Make.
    // SAFETY: caller's contract.
    unsafe { put(mmu, table, idx, armv8::table_descriptor(child)) };
    mmu.publish();
    Ok(child)
}

/// Map the 4 KiB page at `va` to `pa` with `attrs` in the tree rooted at
/// physical `root`, creating intermediate tables through `alloc`.
///
/// Replacing an existing page mapping is break-before-make too, for the same
/// reason as a block split — except that here one by-address invalidate is
/// exactly the right scope. An identical descriptor is a no-op: not merely an
/// optimisation, since breaking and remaking a mapping that was already correct
/// would open a fault window for nothing.
///
/// # Safety
/// `root` must be a live [`armv8::TOP_LEVEL`] translation table reachable
/// through `mmu`, and the caller must be able to tolerate `va`'s translation
/// changing (with interrupts disabled, if the tree is the one it is running on).
pub unsafe fn map_page(
    mmu: &impl Mmu,
    root: u64,
    va: u64,
    pa: u64,
    attrs: u64,
    split: Split,
    alloc: &mut impl TableFrames,
) -> Result<(), MapError> {
    if va >= 1u64 << armv8::VA_BITS {
        return Err(MapError::VaOutOfRange);
    }
    if va % 4096 != 0 || pa % 4096 != 0 {
        return Err(MapError::Misaligned);
    }

    let mut table = root;
    for level in armv8::TOP_LEVEL..armv8::PAGE_LEVEL {
        let idx = armv8::table_index(va, level);
        // SAFETY: `table` is a live table (the root, or a descendant reached
        // through a table descriptor); `idx < ENTRIES`.
        let desc = unsafe { get(mmu, table, idx) };
        table = match armv8::walk_step(desc, level) {
            Step::Descend(next) => next,
            Step::NeedTable => {
                let child = fresh_table(mmu, alloc)?;
                // Invalid to valid: no break-before-make needed (there was no
                // translation to conflict with), but the write must still be
                // published and any cached faulting entry invalidated.
                // SAFETY: as above.
                unsafe { put(mmu, table, idx, armv8::table_descriptor(child)) };
                mmu.publish();
                mmu.invalidate_page(va);
                child
            }
            Step::SplitBlock => {
                if split == Split::Refuse {
                    return Err(MapError::WouldSplitBlock { level, base: armv8::descriptor_addr(desc) });
                }
                // SAFETY: `desc` is a valid block at `level` (that is what
                // `SplitBlock` means), and the caller allowed the split.
                unsafe { split_block(mmu, table, idx, level, alloc)? }
            }
        };
    }

    let idx = armv8::table_index(va, armv8::PAGE_LEVEL);
    let desc = armv8::descriptor(armv8::PAGE_LEVEL, pa, attrs);
    // SAFETY: `table` is the live L3 table for `va`.
    let old = unsafe { get(mmu, table, idx) };
    if old == desc {
        return Ok(()); // already exactly this mapping
    }
    if armv8::is_valid(old) {
        // SAFETY: as above.
        unsafe { put(mmu, table, idx, armv8::INVALID) };
        mmu.publish();
        mmu.invalidate_page(va);
    }
    // SAFETY: as above.
    unsafe { put(mmu, table, idx, desc) };
    mmu.publish();
    mmu.invalidate_page(va);
    Ok(())
}

/// Unmap the 4 KiB page at `va`, returning whether anything was mapped there.
///
/// Only ever clears an L3 **page**: a `va` covered by a block is left alone and
/// reported as unmapped-by-this-call (`false`), because zeroing the block would
/// unmap up to a gigabyte the caller did not ask about. Intermediate tables are
/// kept — freeing an emptied table needs a per-table live count, which belongs
/// with per-task address-space teardown rather than here.
///
/// # Safety
/// As [`map_page`].
pub unsafe fn unmap_page(mmu: &impl Mmu, root: u64, va: u64) -> bool {
    let mut table = root;
    for level in armv8::TOP_LEVEL..armv8::PAGE_LEVEL {
        // SAFETY: caller's contract; `table` is live at every step.
        let desc = unsafe { get(mmu, table, armv8::table_index(va, level)) };
        match armv8::walk_step(desc, level) {
            Step::Descend(next) => table = next,
            Step::NeedTable | Step::SplitBlock => return false,
        }
    }
    let idx = armv8::table_index(va, armv8::PAGE_LEVEL);
    // SAFETY: caller's contract.
    if !armv8::is_valid(unsafe { get(mmu, table, idx) }) {
        return false;
    }
    // SAFETY: caller's contract.
    unsafe { put(mmu, table, idx, armv8::INVALID) };
    mmu.publish();
    mmu.invalidate_page(va);
    true
}

/// The physical address `va` translates to in the tree rooted at `root`, or
/// `None` if nothing maps it. Resolves blocks as well as pages, adding the
/// offset within whichever granule was found — so it reports what the hardware
/// walker would, which is the point of having it (a mapping that reads back
/// wrong is otherwise only observable as a fault somewhere else).
///
/// # Safety
/// `root` must be a live [`armv8::TOP_LEVEL`] table reachable through `mmu`.
pub unsafe fn translate(mmu: &impl Mmu, root: u64, va: u64) -> Option<u64> {
    let mut table = root;
    for level in armv8::TOP_LEVEL..=armv8::PAGE_LEVEL {
        // SAFETY: caller's contract.
        let desc = unsafe { get(mmu, table, armv8::table_index(va, level)) };
        if !armv8::is_valid(desc) {
            return None;
        }
        if armv8::is_leaf(desc, level) {
            let size = armv8::level_size(level);
            return Some(armv8::descriptor_addr(desc) | (va & (size - 1)));
        }
        table = armv8::descriptor_addr(desc);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::alloc::{alloc_zeroed, Layout};
    use alloc::vec::Vec;

    /// Synthetic physical address the first test table is handed out at. Low
    /// enough to fit a descriptor's 47:12 address field, and far from every
    /// address the tests *map*, so a table can never be confused with a mapping
    /// target.
    const PHYS_BASE: u64 = 0x20_0000_0000; // 128 GiB

    /// Synthetic-physical to real-pointer table, shared between the frame source
    /// (which populates it) and the [`Mmu`] (which resolves through it).
    type Frames = alloc::rc::Rc<core::cell::RefCell<Vec<(u64, *mut u64)>>>;

    /// Table frames for the tests: real 4 KiB-aligned heap blocks, handed out
    /// under **synthetic** low physical addresses.
    ///
    /// Using the heap pointers directly as physical addresses is the obvious
    /// thing and it does not work: the x86 kernel heap lives above 2^48, and a
    /// descriptor holds only bits 47:12, so `descriptor_addr` hands back a
    /// truncated address and the next dereference faults. That is not
    /// hypothetical — it is what the first version of these tests did, and it
    /// general-protection-faulted on the first descent into an allocated table.
    struct HeapFrames {
        frames: Frames,
        next_phys: u64,
        limit: usize,
    }

    impl HeapFrames {
        fn new() -> Self {
            Self { frames: Frames::default(), next_phys: PHYS_BASE, limit: usize::MAX }
        }
        /// A source that runs dry after `n` frames, for the OOM path.
        fn limited(n: usize) -> Self {
            Self { limit: n, ..Self::new() }
        }
        fn count(&self) -> usize {
            self.frames.borrow().len()
        }
    }

    impl TableFrames for HeapFrames {
        fn allocate_frame(&mut self) -> Option<u64> {
            if self.count() >= self.limit {
                return None;
            }
            let layout = Layout::from_size_align(4096, 4096).unwrap();
            // SAFETY: nonzero, aligned layout; leaked for the test's duration and
            // only ever addressed as a translation table through the walker.
            let p = unsafe { alloc_zeroed(layout) } as *mut u64;
            assert!(!p.is_null(), "test heap exhausted");
            let phys = self.next_phys;
            self.next_phys += 4096;
            self.frames.borrow_mut().push((phys, p));
            Some(phys)
        }
    }

    /// What the walker asked the platform to do, in order. Recording rather than
    /// performing is what makes break-before-make ordering assertable.
    #[derive(Debug, PartialEq, Eq, Clone, Copy)]
    enum Op {
        Publish,
        InvalidatePage(u64),
        InvalidateAll,
    }

    struct RecordingMmu {
        frames: Frames,
        ops: core::cell::RefCell<Vec<Op>>,
    }

    impl RecordingMmu {
        fn ops(&self) -> Vec<Op> {
            self.ops.borrow().clone()
        }
        fn clear(&self) {
            self.ops.borrow_mut().clear();
        }
    }

    impl Mmu for RecordingMmu {
        fn table_ptr(&self, pa: u64) -> *mut u64 {
            match self.frames.borrow().iter().find(|&&(p, _)| p == pa) {
                Some(&(_, ptr)) => ptr,
                // Strict on purpose: the walker must only ever dereference a
                // table it was handed. Reaching here means it derived a table
                // address from something that was not a table descriptor.
                None => panic!("table_ptr: {pa:#x} is not a table this harness handed out"),
            }
        }
        fn publish(&self) {
            self.ops.borrow_mut().push(Op::Publish);
        }
        fn invalidate_page(&self, va: u64) {
            self.ops.borrow_mut().push(Op::InvalidatePage(va));
        }
        fn invalidate_all(&self) {
            self.ops.borrow_mut().push(Op::InvalidateAll);
        }
    }

    fn fixture() -> (RecordingMmu, HeapFrames, u64) {
        let mut frames = HeapFrames::new();
        let mmu = RecordingMmu { frames: frames.frames.clone(), ops: core::cell::RefCell::new(Vec::new()) };
        let root = frames.allocate_frame().unwrap();
        (mmu, frames, root)
    }

    const PAGE: u64 = 4096;
    const MIB2: u64 = 2 << 20;
    const GIB: u64 = 1 << 30;

    #[test_case]
    fn map_then_translate_round_trips_through_three_fresh_levels() {
        let (mmu, mut f, root) = fixture();
        let va = (5 * GIB) | (7 * MIB2) | (9 * PAGE);
        // SAFETY: `root` is a table this test owns; nothing executes out of it.
        unsafe {
            map_page(&mmu, root, va, 0x1234_5000, armv8::normal_attrs(), Split::Allow, &mut f).unwrap();
            assert_eq!(translate(&mmu, root, va), Some(0x1234_5000));
            // Neighbouring pages are untouched.
            assert_eq!(translate(&mmu, root, va + PAGE), None);
            assert_eq!(translate(&mmu, root, va - PAGE), None);
        }
        // Three levels: root existed, so two intermediate tables plus the L3.
        assert_eq!(f.count(), 3, "L2, L3 and the root");
    }

    #[test_case]
    fn translate_reports_the_offset_within_a_page() {
        let (mmu, mut f, root) = fixture();
        // SAFETY: a table this test owns.
        unsafe {
            map_page(&mmu, root, PAGE, 0x9000, armv8::normal_attrs(), Split::Allow, &mut f).unwrap();
            assert_eq!(translate(&mmu, root, PAGE + 0xabc), Some(0x9abc));
        }
    }

    #[test_case]
    fn a_second_page_in_the_same_region_reuses_the_intermediate_tables() {
        let (mmu, mut f, root) = fixture();
        // SAFETY: a table this test owns.
        unsafe {
            map_page(&mmu, root, 0, 0x1000, armv8::normal_attrs(), Split::Allow, &mut f).unwrap();
            let after_first = f.count();
            map_page(&mmu, root, PAGE, 0x2000, armv8::normal_attrs(), Split::Allow, &mut f).unwrap();
            assert_eq!(f.count(), after_first, "no new tables for a sibling page");
            assert_eq!(translate(&mmu, root, 0), Some(0x1000));
            assert_eq!(translate(&mmu, root, PAGE), Some(0x2000));
        }
    }

    #[test_case]
    fn remapping_a_live_page_breaks_before_it_makes() {
        let (mmu, mut f, root) = fixture();
        // SAFETY: a table this test owns.
        unsafe {
            map_page(&mmu, root, 0, 0x1000, armv8::normal_attrs(), Split::Allow, &mut f).unwrap();
            mmu.clear();
            map_page(&mmu, root, 0, 0x5000, armv8::normal_attrs(), Split::Allow, &mut f).unwrap();
            assert_eq!(translate(&mmu, root, 0), Some(0x5000));
        }
        // Break (publish + invalidate) strictly before make (publish +
        // invalidate). Writing the new descriptor over the live one in place is
        // a valid-to-valid change real Apple cores fault on.
        assert_eq!(
            mmu.ops(),
            [
                Op::Publish,
                Op::InvalidatePage(0),
                Op::Publish,
                Op::InvalidatePage(0)
            ]
        );
    }

    #[test_case]
    fn remapping_a_page_to_exactly_what_it_already_says_touches_nothing() {
        let (mmu, mut f, root) = fixture();
        // SAFETY: a table this test owns.
        unsafe {
            map_page(&mmu, root, 0, 0x1000, armv8::normal_attrs(), Split::Allow, &mut f).unwrap();
            mmu.clear();
            map_page(&mmu, root, 0, 0x1000, armv8::normal_attrs(), Split::Allow, &mut f).unwrap();
        }
        // No break, no make: an identical rewrite would otherwise open a fault
        // window over a mapping that was already correct.
        assert!(mmu.ops().is_empty(), "an identical mapping must be a no-op, got {:?}", mmu.ops());
    }

    #[test_case]
    fn changing_only_the_attributes_of_a_live_page_still_uses_bbm() {
        let (mmu, mut f, root) = fixture();
        // SAFETY: a table this test owns.
        unsafe {
            map_page(&mmu, root, 0, 0x1000, armv8::normal_attrs(), Split::Allow, &mut f).unwrap();
            mmu.clear();
            let user = armv8::normal_attrs() | armv8::AP_USER;
            map_page(&mmu, root, 0, 0x1000, user, Split::Allow, &mut f).unwrap();
            assert_eq!(translate(&mmu, root, 0), Some(0x1000));
        }
        assert_eq!(mmu.ops().len(), 4, "same address, different attributes is still valid-to-valid");
    }

    #[test_case]
    fn splitting_a_2mib_block_preserves_every_other_page_in_it() {
        let (mmu, mut f, root) = fixture();
        // Hand-build an L1 -> L2 tree whose L2 entry is a 2 MiB block, the shape
        // `mmu::init` produces for a mixed RAM/MMIO gigabyte.
        let l2 = f.allocate_frame().unwrap();
        let block_pa = 16 * MIB2;
        // SAFETY: tables this test owns.
        unsafe {
            put(&mmu, root, 0, armv8::table_descriptor(l2));
            put(&mmu, l2, 1, armv8::descriptor(2, block_pa, armv8::normal_attrs()));
            // Before: the block translates its whole 2 MiB range.
            assert_eq!(translate(&mmu, root, MIB2), Some(block_pa));
            assert_eq!(translate(&mmu, root, MIB2 + 0x1000), Some(block_pa + 0x1000));

            mmu.clear();
            // Remap one page inside it to somewhere else entirely.
            map_page(&mmu, root, MIB2 + 0x3000, 0xdead_0000, armv8::normal_attrs(), Split::Allow, &mut f)
                .unwrap();

            // The one page moved...
            assert_eq!(translate(&mmu, root, MIB2 + 0x3000), Some(0xdead_0000));
            // ...and all 511 others still resolve exactly where the block put
            // them. This is what `split_child` is for, and a wrong child address
            // or type here would silently relocate or unmap them.
            for i in 0..512u64 {
                if i == 3 {
                    continue;
                }
                assert_eq!(
                    translate(&mmu, root, MIB2 + i * PAGE),
                    Some(block_pa + i * PAGE),
                    "page {i} of the split block"
                );
            }
        }
    }

    #[test_case]
    fn splitting_a_block_invalidates_the_whole_tlb_between_break_and_make() {
        let (mmu, mut f, root) = fixture();
        let l2 = f.allocate_frame().unwrap();
        // SAFETY: tables this test owns.
        unsafe {
            put(&mmu, root, 0, armv8::table_descriptor(l2));
            put(&mmu, l2, 0, armv8::descriptor(2, 0, armv8::normal_attrs()));
            mmu.clear();
            map_page(&mmu, root, 0x4000, 0x7000, armv8::normal_attrs(), Split::Allow, &mut f).unwrap();
        }
        let ops = mmu.ops();
        // A by-address invalidate cannot cover a 2 MiB block, so the split must
        // use the whole-TLB form; and it must sit between the break and the make.
        let all = ops.iter().position(|o| *o == Op::InvalidateAll).expect("block split must flush the whole TLB");
        assert!(all > 0 && matches!(ops[all - 1], Op::Publish), "break is published before the flush");
        assert!(all + 1 < ops.len(), "the make follows the flush");
    }

    #[test_case]
    fn splitting_a_1gib_block_goes_through_two_levels() {
        let (mmu, mut f, root) = fixture();
        // SAFETY: a table this test owns.
        unsafe {
            // An L1 1 GiB block, as `mmu::init` writes for a whole-RAM gigabyte.
            put(&mmu, root, 1, armv8::descriptor(1, GIB, armv8::normal_attrs()));
            map_page(&mmu, root, GIB + 0x2000, 0xabc_0000, armv8::normal_attrs(), Split::Allow, &mut f)
                .unwrap();
            assert_eq!(translate(&mmu, root, GIB + 0x2000), Some(0xabc_0000));
            // Both a neighbouring 4 KiB page inside the split 2 MiB chunk and a
            // far-away page still under an untouched 2 MiB child resolve.
            assert_eq!(translate(&mmu, root, GIB + 0x3000), Some(GIB + 0x3000));
            assert_eq!(translate(&mmu, root, GIB + 700 * MIB2 / 2), Some(GIB + 700 * MIB2 / 2));
        }
    }

    #[test_case]
    fn a_refused_split_leaves_the_block_exactly_as_it_was() {
        let (mmu, mut f, root) = fixture();
        // SAFETY: a table this test owns.
        unsafe {
            put(&mmu, root, 2, armv8::descriptor(1, 2 * GIB, armv8::normal_attrs()));
            let before = get(&mmu, root, 2);
            mmu.clear();
            let e = map_page(&mmu, root, 2 * GIB, 0x1000, armv8::normal_attrs(), Split::Refuse, &mut f);
            assert_eq!(e, Err(MapError::WouldSplitBlock { level: 1, base: 2 * GIB }));
            // The refusal must be total: no descriptor touched, no TLB op, so
            // the live kernel map is not left half-broken.
            assert_eq!(get(&mmu, root, 2), before);
            assert!(mmu.ops().is_empty());
            // And the block still translates.
            assert_eq!(translate(&mmu, root, 2 * GIB + 0x555), Some(2 * GIB + 0x555));
        }
    }

    #[test_case]
    fn unmap_clears_a_page_and_reports_whether_there_was_one() {
        let (mmu, mut f, root) = fixture();
        // SAFETY: a table this test owns.
        unsafe {
            map_page(&mmu, root, PAGE, 0x1000, armv8::normal_attrs(), Split::Allow, &mut f).unwrap();
            assert!(unmap_page(&mmu, root, PAGE));
            assert_eq!(translate(&mmu, root, PAGE), None);
            // Idempotent: a second unmap reports there was nothing.
            assert!(!unmap_page(&mmu, root, PAGE));
            // An address whose tables do not even exist is not an error.
            assert!(!unmap_page(&mmu, root, 300 * GIB));
        }
    }

    #[test_case]
    fn unmap_refuses_to_tear_down_a_block() {
        let (mmu, _f, root) = fixture();
        // SAFETY: a table this test owns.
        unsafe {
            put(&mmu, root, 0, armv8::descriptor(1, 0, armv8::normal_attrs()));
            // Asking to unmap one page must not zero a descriptor covering a
            // gigabyte the caller said nothing about.
            assert!(!unmap_page(&mmu, root, 0x5000));
            assert_eq!(translate(&mmu, root, 0x5000), Some(0x5000));
        }
    }

    #[test_case]
    fn running_out_of_frames_is_an_error_not_a_panic() {
        // One frame for the root, then nothing: the first intermediate table
        // cannot be allocated.
        let mut f = HeapFrames::limited(1);
        let mmu = RecordingMmu { frames: f.frames.clone(), ops: core::cell::RefCell::new(Vec::new()) };
        let root = f.allocate_frame().unwrap();
        // SAFETY: a table this test owns.
        let e = unsafe { map_page(&mmu, root, 0, 0x1000, armv8::normal_attrs(), Split::Allow, &mut f) };
        assert_eq!(e, Err(MapError::OutOfFrames));
        // SAFETY: as above.
        assert_eq!(unsafe { translate(&mmu, root, 0) }, None, "no half-built mapping");
    }

    #[test_case]
    fn bad_arguments_are_rejected_before_anything_is_written() {
        let (mmu, mut f, root) = fixture();
        // Beyond the 39-bit VA the kernel's TCR configures: mapping it would
        // silently alias a low address, since the index math masks to 9 bits.
        // SAFETY: a table this test owns.
        unsafe {
            assert_eq!(
                map_page(&mmu, root, 1 << armv8::VA_BITS, 0, armv8::normal_attrs(), Split::Allow, &mut f),
                Err(MapError::VaOutOfRange)
            );
            assert_eq!(
                map_page(&mmu, root, 0x800, 0, armv8::normal_attrs(), Split::Allow, &mut f),
                Err(MapError::Misaligned)
            );
            assert_eq!(
                map_page(&mmu, root, 0, 0x800, armv8::normal_attrs(), Split::Allow, &mut f),
                Err(MapError::Misaligned)
            );
        }
        assert!(mmu.ops().is_empty());
        assert_eq!(f.count(), 1, "only the root; a rejected call allocates nothing");
    }

    #[test_case]
    fn a_user_page_carries_the_user_bit_all_the_way_into_the_descriptor() {
        let (mmu, mut f, root) = fixture();
        let attrs = armv8::normal_attrs() | armv8::AP_USER | armv8::UXN | armv8::PXN;
        // SAFETY: a table this test owns.
        unsafe {
            map_page(&mmu, root, 0x1000, 0x2000, attrs, Split::Allow, &mut f).unwrap();
            // Read the leaf back through the tables rather than trusting the
            // return value: unlike x86, an ARMv8 table descriptor adds no
            // permission of its own, so the leaf is where AP[1] must land.
            let l2 = armv8::descriptor_addr(get(&mmu, root, 0));
            let l3 = armv8::descriptor_addr(get(&mmu, l2, 0));
            let leaf = get(&mmu, l3, 1);
            assert_eq!(armv8::descriptor_attrs(leaf), attrs);
            assert_ne!(leaf & armv8::AP_USER, 0);
        }
    }
}
