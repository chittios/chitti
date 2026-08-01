//! Per-task address spaces: a translation-table root that **shares every kernel
//! mapping** and owns a private user range.
//!
//! # The shape, and why it is the same on both arches
//!
//! An [`AddressSpace`] is a fresh top-level table whose kernel entries are copied
//! from the live kernel root, plus whatever the task maps into the user range.
//! Because the kernel half is present in every space, the kernel can be executing
//! — on its own stack, with its own heap and MMIO reachable — in *any* address
//! space, so switching is a `CR3`/`TTBR0_EL1` write and nothing more. There is no
//! trampoline, no "switch to a neutral space first", and no per-arch dance.
//!
//! The two arches divide the address space differently, which is the only
//! asymmetry and is confined to [`USER_BASE`]:
//!
//! * **x86_64** — the kernel lives in the higher half (its image and Limine's
//!   HHDM), so the *lower* half is free and is where user memory goes. Copying
//!   PML4 entries 256..512 shares the kernel by reference: a later kernel mapping
//!   is installed *below* those entries and so appears in every space
//!   automatically.
//! * **aarch64** — the kernel is an identity map over `[0, mapped_bytes)`, so user
//!   memory goes *above* it, in the rest of the 512 GiB the 39-bit VA covers.
//!   Here the L1 entries are copied by value, so a kernel mapping made *after* a
//!   space exists would not appear in it. Device discovery all happens at boot,
//!   before any user space is created, which is what makes that acceptable — and
//!   the one assumption to revisit if it stops being true.
//!
//! # Why the kernel did not move to TTBR1
//!
//! The alternative was relocating the aarch64 kernel to a high half, as Linux
//! does. It would reduce cross-arch divergence, and the walker is already written
//! against a trait that would absorb it ([`super::walk::Mmu::table_ptr`]) — but
//! VA == PA is assumed across roughly ten aarch64 driver files, and it is
//! boot-critical on m1n1, VirtualBox-ARM and UTM. Per-task address spaces do not
//! need it: sharing the kernel by copied entries gets the same isolation for user
//! memory. So that change stays available and unmade.

use super::frame::FRAME_SIZE;
use super::walk::MapError;
use alloc::vec::Vec;

/// First virtual address available to a task, and the size of that range.
///
/// x86: the lower half, which no kernel mapping occupies. Deliberately not
/// address 0, so a null dereference in a user task still faults.
#[cfg(target_arch = "x86_64")]
pub const USER_BASE: u64 = 0x0000_0000_4000_0000; // 1 GiB
#[cfg(target_arch = "x86_64")]
pub const USER_SIZE: u64 = 0x0000_7000_0000_0000; // to the top of the lower half

/// aarch64: above the boot identity map, inside the 39-bit VA. 256 GiB is past
/// any plausible `mapped_bytes` (which tracks installed RAM) while leaving 256
/// GiB of user range; [`AddressSpace::new`] checks the machine agrees rather
/// than trusting the constant.
#[cfg(not(target_arch = "x86_64"))]
pub const USER_BASE: u64 = 256 << 30;
#[cfg(not(target_arch = "x86_64"))]
pub const USER_SIZE: u64 = 256 << 30;

/// Whether `va` is a page-aligned address inside the user range.
pub fn is_user_addr(va: u64) -> bool {
    va % FRAME_SIZE == 0 && is_user_range(va, va.saturating_add(1))
}

/// Whether the half-open span `[start, end)` lies entirely inside the user range.
///
/// Unaligned addresses are fine — a syscall argument is a byte pointer, not a
/// page. What is *not* fine is a span that leaves the range at either end, so
/// this is the check [`crate::synapse::abi::copy_in`] uses after `checked_add`.
pub fn is_user_range(start: u64, end: u64) -> bool {
    let top = USER_BASE.saturating_add(USER_SIZE);
    start >= USER_BASE && end <= top && start <= end
}

/// Where the kernel can read physical address `phys`. Public because the syscall
/// layer copies through it; see [`AddressSpace`] for why this is the identity on
/// aarch64 and the HHDM on x86.
pub fn phys_to_kernel(phys: u64) -> u64 {
    phys_to_ptr(phys)
}

/// What a user mapping may do.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct UserPerms {
    pub write: bool,
    pub exec: bool,
}

impl UserPerms {
    pub const RO: Self = Self { write: false, exec: false };
    pub const RW: Self = Self { write: true, exec: false };
    pub const RX: Self = Self { write: false, exec: true };
}

/// A task's address space: kernel mappings shared, user mappings private.
///
/// Owns every frame it allocated — page tables *and* user pages — and returns
/// them on drop. That is the whole reason the frames are tracked rather than
/// merely mapped: an address space that leaked its tables would leak a few pages
/// per task forever, which is the P1 mistake in a new place.
pub struct AddressSpace {
    root: u64,
    /// Frames to hand back on drop, `root` included.
    owned: Vec<u64>,
}

/// Take a zeroed frame from the global allocator.
fn take_frame() -> Option<u64> {
    let phys = super::FRAME_ALLOCATOR.with(|slot| slot.as_mut().and_then(|a| a.allocate()))?;
    let virt = phys_to_ptr(phys);
    // SAFETY: a freshly-allocated, exclusively-owned frame, reachable at `virt`.
    // Zeroing matters: a stale word in a table reads as a valid descriptor.
    unsafe { core::ptr::write_bytes(virt as *mut u8, 0, FRAME_SIZE as usize) };
    Some(phys)
}

/// Where physical `phys` is readable by the kernel.
fn phys_to_ptr(phys: u64) -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        crate::arch::x86_64::paging::phys_to_virt(phys)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        phys // identity map
    }
}

impl AddressSpace {
    /// Create a space that shares the kernel's mappings and has an empty user
    /// range. `None` if no frame is available (or, on aarch64, if the boot
    /// identity map reaches into [`USER_BASE`], which would mean user memory
    /// aliasing kernel memory).
    pub fn new() -> Option<Self> {
        #[cfg(not(target_arch = "x86_64"))]
        {
            let mapped = crate::arch::aarch64::mmu::mapped_bytes();
            if mapped > USER_BASE {
                crate::ktrace::log_fmt(format_args!(
                    "mm::space: identity map reaches {mapped:#x}, past USER_BASE {USER_BASE:#x} -- refusing"
                ));
                return None;
            }
        }
        let frame = take_frame()?;
        // SAFETY: `frame` is an owned, zeroed, kernel-reachable 4 KiB frame.
        // Both take and return a *physical* address; each reaches the table the
        // way its arch does (HHDM / identity).
        let root = unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                crate::arch::x86_64::paging::new_root_sharing_kernel(frame)
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                crate::arch::aarch64::mmu::new_root_sharing_kernel(frame)
            }
        };
        Some(Self { root, owned: alloc::vec![frame] })
    }

    /// The root's physical address, for [`Self::activate`] and diagnostics.
    pub fn root(&self) -> u64 {
        self.root
    }

    /// How many frames this space owns (tables plus user pages).
    pub fn frame_count(&self) -> usize {
        self.owned.len()
    }

    /// The physical address `va` maps to **in this space**, or `None`.
    ///
    /// This is what makes a user pointer trustworthy: the user range is a
    /// constant, so range-checking an address says only that a tenant *could*
    /// have owned it. Resolving it through the caller's own tables says it does.
    pub fn translate(&self, va: u64) -> Option<u64> {
        #[cfg(target_arch = "x86_64")]
        {
            crate::arch::x86_64::paging::translate_in(self.root, va)
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            // SAFETY: `self.root` is a live table this space owns, reachable at
            // its physical address (VA == PA).
            unsafe { super::walk::translate(&HwMmu, self.root, va) }
        }
    }

    /// Map one 4 KiB page of *fresh, zeroed* memory at user address `va`.
    ///
    /// The frame is allocated here and owned by this space, so a task cannot be
    /// handed a page that something else is still using — which is what "isolated"
    /// has to mean before ring 3 exists to enforce it.
    pub fn map_new_page(&mut self, va: u64, perms: UserPerms) -> Result<u64, MapError> {
        if !is_user_addr(va) {
            return Err(MapError::VaOutOfRange);
        }
        let frame = take_frame().ok_or(MapError::OutOfFrames)?;
        self.owned.push(frame);
        self.map_frame(va, frame, perms)?;
        Ok(frame)
    }

    /// Map an existing frame at user address `va`. The frame is **not** taken
    /// over: use this to share something the kernel owns (a shared buffer),
    /// and [`Self::map_new_page`] for a task's own memory.
    pub fn map_frame(&mut self, va: u64, phys: u64, perms: UserPerms) -> Result<(), MapError> {
        if !is_user_addr(va) {
            return Err(MapError::VaOutOfRange);
        }
        // Tables created during the walk belong to this space too. The walker
        // takes frames from the global allocator, so they are collected after the
        // fact by watching the free count — see `map_via_walker`.
        self.map_via_walker(va, phys, perms)
    }

    #[cfg(target_arch = "x86_64")]
    fn map_via_walker(&mut self, va: u64, phys: u64, perms: UserPerms) -> Result<(), MapError> {
        use crate::arch::x86_64::paging::{self, NO_EXECUTE, PRESENT, USER_ACCESSIBLE, WRITABLE};
        let mut flags = PRESENT | USER_ACCESSIBLE;
        if perms.write {
            flags |= WRITABLE;
        }
        if !perms.exec {
            flags |= NO_EXECUTE;
        }
        // Intermediate tables the walk creates are tracked by recording the
        // frames handed out during it, so `Drop` returns them.
        let mut tracker = TrackingFrames { owned: &mut self.owned };
        paging::map_page_in(self.root, va, phys, flags, &mut tracker);
        Ok(())
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn map_via_walker(&mut self, va: u64, phys: u64, perms: UserPerms) -> Result<(), MapError> {
        use super::armv8;
        use super::walk::{self, Split};
        let mut attrs = armv8::normal_attrs() | armv8::AP_USER | armv8::NG;
        if !perms.write {
            attrs |= armv8::AP_RO;
        }
        // Execute-never for both levels unless asked; `PXN` always, since user
        // memory must never be executable at EL1 whatever the task may run.
        attrs |= armv8::PXN;
        if !perms.exec {
            attrs |= armv8::UXN;
        }
        let mut tracker = TrackingFrames { owned: &mut self.owned };
        // `Split::Allow` is safe here in a way it is not for the kernel root:
        // this table is owned by this space and nothing is executing out of its
        // user range, so a break-before-make window unmaps nothing live.
        // SAFETY: `self.root` is a live L1 table this space owns, reachable at its
        // physical address (VA == PA).
        unsafe { walk::map_page(&HwMmu, self.root, va, phys, attrs, Split::Allow, &mut tracker) }
    }

    /// Make this space the active one.
    ///
    /// # Safety
    /// Kernel code, data and the current stack must be mapped here — which
    /// construction guarantees — and the caller must restore the previous root
    /// before this space is dropped, or the machine will be running on freed
    /// tables.
    pub unsafe fn activate(&self) {
        // SAFETY: caller's contract.
        unsafe {
            #[cfg(target_arch = "x86_64")]
            crate::arch::x86_64::paging::activate_root(self.root);
            #[cfg(not(target_arch = "x86_64"))]
            crate::arch::aarch64::mmu::activate_root(self.root);
        }
    }
}

/// The kernel's own root, to switch back to.
pub fn kernel_root() -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        crate::arch::x86_64::paging::active_cr3()
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        crate::arch::aarch64::mmu::kernel_root()
    }
}

/// Restore the kernel's address space.
///
/// # Safety
/// Only meaningful while the kernel root still exists, which it always does.
pub unsafe fn activate_kernel(root: u64) {
    // SAFETY: `root` came from `kernel_root()`, which maps everything.
    unsafe {
        #[cfg(target_arch = "x86_64")]
        crate::arch::x86_64::paging::activate_root(root);
        #[cfg(not(target_arch = "x86_64"))]
        crate::arch::aarch64::mmu::activate_root(root);
    }
}

/// aarch64's barrier/TLB implementation for [`super::walk`], mirroring the one in
/// `arch::aarch64::mmu` (which is private to that module).
#[cfg(not(target_arch = "x86_64"))]
struct HwMmu;

#[cfg(not(target_arch = "x86_64"))]
impl super::walk::Mmu for HwMmu {
    fn publish(&self) {
        // SAFETY: a store barrier.
        unsafe { core::arch::asm!("dsb ishst", options(nostack, preserves_flags)) };
    }
    fn invalidate_page(&self, va: u64) {
        // SAFETY: TLB maintenance is always architecturally safe.
        unsafe {
            core::arch::asm!("tlbi vaae1is, {}", "dsb ish", "isb", in(reg) va >> 12, options(nostack, preserves_flags));
        }
    }
    fn invalidate_all(&self) {
        // SAFETY: as above.
        unsafe { core::arch::asm!("tlbi vmalle1is", "dsb ish", "isb", options(nostack, preserves_flags)) };
    }
}

/// A frame source that records what it hands out, so an [`AddressSpace`] can
/// return the intermediate tables a walk created — not just the pages it asked
/// for. Without this the tables leak, a few pages per task, forever.
struct TrackingFrames<'a> {
    owned: &'a mut Vec<u64>,
}

impl super::frame::TableFrames for TrackingFrames<'_> {
    fn allocate_frame(&mut self) -> Option<u64> {
        let phys = super::FRAME_ALLOCATOR.with(|slot| slot.as_mut().and_then(|a| a.allocate()))?;
        self.owned.push(phys);
        Some(phys)
    }
}

impl Drop for AddressSpace {
    fn drop(&mut self) {
        // Never free the space we are running in: that would pull the tables out
        // from under the CPU, and the fault would come later, somewhere else.
        if kernel_root() == self.root {
            crate::ktrace::log("mm::space", "BUG: dropping the active address space -- leaking it instead");
            return;
        }
        super::FRAME_ALLOCATOR.with(|slot| {
            if let Some(a) = slot.as_mut() {
                for &f in self.owned.iter() {
                    a.free(f);
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn free_frames() -> u64 {
        super::super::FRAME_ALLOCATOR.with(|slot| slot.as_ref().map(|a| a.free_frame_count()).unwrap_or(0))
    }

    #[test_case]
    fn a_new_space_shares_the_kernel_mappings() {
        // The property everything else rests on: the kernel must be reachable in
        // a task's space, or switching to it would fault on the next instruction.
        // Checked by translating a *kernel* address through the new root.
        let space = AddressSpace::new().expect("a space needs a frame allocator");
        let probe = &space as *const _ as u64; // a live kernel address (this stack)
        let via_kernel = translate_in(kernel_root(), probe);
        let via_space = translate_in(space.root(), probe);
        assert!(via_kernel.is_some(), "test premise: the probe address is mapped");
        assert_eq!(via_space, via_kernel, "a kernel address must resolve identically in a task space");
    }

    #[test_case]
    fn user_pages_are_private_to_their_space() {
        // Two spaces mapping the same user address must get different memory —
        // that is the whole point, and it is what a shared table would break.
        let mut a = AddressSpace::new().unwrap();
        let mut b = AddressSpace::new().unwrap();
        let fa = a.map_new_page(USER_BASE, UserPerms::RW).expect("map a");
        let fb = b.map_new_page(USER_BASE, UserPerms::RW).expect("map b");
        assert_ne!(fa, fb, "the same user VA in two spaces must be different frames");
        assert_eq!(translate_in(a.root(), USER_BASE), Some(fa));
        assert_eq!(translate_in(b.root(), USER_BASE), Some(fb));
        // And the kernel's own space must not have acquired that mapping.
        assert_eq!(translate_in(kernel_root(), USER_BASE), None, "user memory must not leak into the kernel space");
    }

    #[test_case]
    fn addresses_outside_the_user_range_are_refused() {
        let mut s = AddressSpace::new().unwrap();
        // Kernel range, address 0, and unaligned addresses are all rejected
        // rather than mapped somewhere surprising.
        assert_eq!(s.map_new_page(0, UserPerms::RW), Err(MapError::VaOutOfRange));
        assert_eq!(s.map_new_page(USER_BASE - FRAME_SIZE, UserPerms::RW), Err(MapError::VaOutOfRange));
        assert_eq!(s.map_new_page(USER_BASE + 0x800, UserPerms::RW), Err(MapError::VaOutOfRange));
        assert!(!is_user_addr(0));
        assert!(is_user_addr(USER_BASE));
    }

    #[test_case]
    fn dropping_a_space_returns_every_frame_it_owned() {
        // Tables as well as pages. An address space that freed only its pages
        // would leak a few frames per task forever — the P1 mistake in a new
        // place, and just as invisible.
        let before = free_frames();
        {
            let mut s = AddressSpace::new().unwrap();
            for i in 0..4 {
                s.map_new_page(USER_BASE + i * FRAME_SIZE, UserPerms::RW).unwrap();
            }
            // Root + intermediate tables + 4 pages: strictly more than the pages.
            assert!(s.frame_count() > 4, "intermediate tables must be tracked, got {}", s.frame_count());
            assert!(free_frames() < before, "test premise: frames were consumed");
        }
        assert_eq!(free_frames(), before, "every frame must come back on drop");
    }

    #[test_case]
    fn a_second_page_in_the_same_region_reuses_tables() {
        let mut s = AddressSpace::new().unwrap();
        s.map_new_page(USER_BASE, UserPerms::RW).unwrap();
        let after_first = s.frame_count();
        s.map_new_page(USER_BASE + FRAME_SIZE, UserPerms::RW).unwrap();
        assert_eq!(
            s.frame_count(),
            after_first + 1,
            "a sibling page costs one frame, not a fresh table chain"
        );
    }

    /// Resolve `va` in the tree rooted at `root`, for assertions. Uses each
    /// arch's own walker so the test checks the real tables.
    fn translate_in(root: u64, va: u64) -> Option<u64> {
        #[cfg(target_arch = "x86_64")]
        {
            crate::arch::x86_64::paging::translate_in(root, va)
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            // SAFETY: `root` is a live table reachable at its physical address.
            unsafe { super::super::walk::translate(&HwMmu, root, va) }
        }
    }
}
