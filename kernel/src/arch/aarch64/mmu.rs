//! Minimal aarch64 MMU bring-up: an identity map over the low 4 GiB with 1 GiB
//! blocks (RAM = Normal write-back cacheable, low 1 GiB = Device for the PL011
//! UART / GIC), then MMU + I/D caches on. QEMU enters with the MMU off, where
//! all memory is device-typed and uncached -- NEON is unreliable and slow
//! there -- so this must run before any NEON/cached work.

use core::arch::asm;
use core::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

#[repr(align(4096))]
struct Table(#[allow(dead_code)] [u64; 512]);
static mut L1: Table = Table([0; 512]);
/// L2 tables for **mixed** 1 GiB blocks (RAM and MMIO interleaved — real
/// firmware and VirtualBox-ARM do this; see [`crate::mm::ramlayout`]). Each
/// maps its GiB at 2 MiB granularity. Statically sized: a machine has only a
/// handful of RAM-clump boundary blocks.
const MAX_L2: usize = 8;
static mut L2S: [Table; MAX_L2] = [const { Table([0; 512]) }; MAX_L2];

/// Top of the RAM/heap region the kernel may use, discovered at boot: on
/// `-kernel` the top of DTB `/memory`; on UEFI `heap_base + heap_size` reported
/// by the stub. Bounds the identity map. 0 until [`init`] runs.
static RAM_END: AtomicU64 = AtomicU64::new(0);
/// Extent (bytes) of the identity map [`init`] built — `MAP_GIB` 1-GiB blocks.
static MAPPED_BYTES: AtomicU64 = AtomicU64::new(0);
/// On the UEFI path, the physical base of the heap the stub pre-allocated
/// (AnyPages) and reported in boot-info; the kernel places its heap there
/// instead of at the top of RAM (the top is where UEFI parks ACPI/runtime data).
/// 0 on the `-kernel` path (heap goes at the top of RAM instead).
static UEFI_HEAP_BASE: AtomicU64 = AtomicU64::new(0);
/// On the UEFI path, the physical base/size of the model the stub loaded
/// (AnyPages) and reported in boot-info; the kernel reads the model there rather
/// than at a fixed address (fixed addresses aren't reliably free under UEFI
/// firmware). Both 0 on the `-kernel` path (model at `cortex::MODEL_LOAD_ADDR`).
static UEFI_MODEL_BASE: AtomicU64 = AtomicU64::new(0);
static UEFI_MODEL_SIZE: AtomicU64 = AtomicU64::new(0);

/// Conservative fallback RAM if neither a DTB nor a stub boot-info field is
/// present (an unknown boot path): assume the `virt` base + 4 GiB. A model that
/// needs more will fail the fit check in `mm::init` with a clear message rather
/// than faulting on an unbacked heap.
const FALLBACK_RAM_END: u64 = 0x4000_0000 + (4 << 30);

/// Max RAM regions carried in boot-info (matches the stub's table size).
const MAX_REGIONS: usize = 16;

/// The RAM extents [`init`] typed its identity map from, republished so
/// [`crate::mm`] can build a frame allocator without re-parsing the boot path.
/// An array of atomics rather than a `static mut`: `init` writes them with the
/// MMU already on, and every reader runs later.
static RAM_REGIONS: [(AtomicU64, AtomicU64); MAX_REGIONS] =
    [const { (AtomicU64::new(0), AtomicU64::new(0)) }; MAX_REGIONS];
static N_RAM_REGIONS: AtomicUsize = AtomicUsize::new(0);

/// The **free** extents the UEFI stub reported (`CONVENTIONAL` memory only) —
/// a strictly different list from [`RAM_REGIONS`], which is every DRAM address
/// including firmware-owned memory. Empty on the `-kernel` path and on an older
/// stub that does not publish the field.
static FREE_REGIONS: [(AtomicU64, AtomicU64); MAX_REGIONS] =
    [const { (AtomicU64::new(0), AtomicU64::new(0)) }; MAX_REGIONS];
static N_FREE_REGIONS: AtomicUsize = AtomicUsize::new(0);

/// The stub's GOP framebuffer `(base, bytes)`, 0 on the `-kernel` path. Read by
/// `mm` so a framebuffer that happens to live in RAM is never handed out as a
/// free frame (on the UEFI path it is usually an MMIO aperture instead, in which
/// case reserving it is a harmless no-op).
static FB_REGION: (AtomicU64, AtomicU64) = (AtomicU64::new(0), AtomicU64::new(0));

/// Whether a valid boot-info page was present, i.e. whether this is the
/// UEFI/stub path. Distinguishes "no firmware, the whole machine is ours" from
/// "firmware owns memory we cannot enumerate" — a distinction the frame
/// allocator's safety depends on, and which `ram_end != 0` cannot make.
static HAVE_BOOTINFO: AtomicBool = AtomicBool::new(false);

/// What [`detect`] found: the usable top address, (UEFI only) the stub's
/// pre-allocated heap and model regions, and the machine's actual RAM extents
/// (`(base, size)` pairs; `n_regions == 0` when the boot path can't provide
/// them — the map then falls back to Normal-to-`ram_end` blocks).
struct RamInfo {
    ram_end: u64,
    uefi_heap_base: u64,
    uefi_model: (u64, u64),
    regions: [(u64, u64); MAX_REGIONS],
    n_regions: usize,
    /// UEFI only: `CONVENTIONAL` extents (see [`FREE_REGIONS`]).
    free: [(u64, u64); MAX_REGIONS],
    n_free: usize,
    /// UEFI only: the GOP framebuffer `(base, bytes)`.
    fb: (u64, u64),
    have_bootinfo: bool,
}

impl RamInfo {
    fn with_regions(mut self, regs: &[(u64, u64)]) -> RamInfo {
        for &r in regs.iter().take(MAX_REGIONS) {
            self.regions[self.n_regions] = r;
            self.n_regions += 1;
            // RAM above a hole can end past any size-derived estimate.
            self.ram_end = self.ram_end.max(r.0.saturating_add(r.1));
        }
        self
    }
    fn bare(ram_end: u64, uefi_heap_base: u64, uefi_model: (u64, u64)) -> RamInfo {
        RamInfo {
            ram_end,
            uefi_heap_base,
            uefi_model,
            regions: [(0, 0); MAX_REGIONS],
            n_regions: 0,
            free: [(0, 0); MAX_REGIONS],
            n_free: 0,
            fb: (0, 0),
            have_bootinfo: false,
        }
    }
}

/// Discover RAM at boot. Runs with the MMU **off** (before the map is built), so
/// it only does pure reads — no atomics, no heap.
fn detect() -> RamInfo {
    // `-kernel`: QEMU passes the DTB in x0; parse its `/memory` node.
    // SAFETY: `boot_x0()` is the DTB pointer (or non-FDT, rejected by magic).
    if let Some((b, s)) = unsafe { super::dtb::memory_region(super::boot::boot_x0()) } {
        return RamInfo::bare(b.saturating_add(s), 0, (0, 0)).with_regions(&[(b, s)]);
    }
    // UEFI/stub: the boot-info page carries the heap and model regions the stub
    // allocated (AnyPages), plus total installed RAM. The map must cover the
    // full machine — not just the stub's two LOADER_DATA regions — so a
    // QEMU-loader-placed model at MODEL_LOAD_ADDR (2 GiB) is identity-mapped
    // even when the stub failed to load a multi-GiB GGUF from the ESP.
    if let Some((hb, hs, mb, ms, total_ram)) = bootinfo_regions(super::boot::boot_x1()) {
        // QEMU `virt` (and the stub's handoff) place system RAM at 0x4000_0000.
        let from_total = if total_ram > 0 { 0x4000_0000u64.saturating_add(total_ram) } else { 0 };
        let ram_end = hb
            .saturating_add(hs)
            .max(mb.saturating_add(ms))
            .max(from_total);
        // The stub also writes the machine's actual RAM extents (count@104,
        // (base,size) pairs@112) — RAM and MMIO interleave inside GiB blocks
        // on VirtualBox/real hardware, and only the true extents let the map
        // type them correctly. Absent on an older stub → n_regions 0 → the
        // legacy Normal-to-ram_end map.
        let mut info = RamInfo::bare(ram_end, hb, (mb, ms));
        info.have_bootinfo = true;
        let bi = super::boot::boot_x1() as *const u8;
        // SAFETY: same validated CHITTIBI page bootinfo_regions just read.
        let rd = |off: usize| -> u64 {
            let b = unsafe { core::slice::from_raw_parts(bi.add(off), 8) };
            u64::from_le_bytes(b.try_into().unwrap())
        };
        let n = rd(104) as usize;
        if n > 0 && n <= MAX_REGIONS {
            let mut regs = [(0u64, 0u64); MAX_REGIONS];
            for (i, r) in regs.iter_mut().enumerate().take(n) {
                *r = (rd(112 + i * 16), rd(112 + i * 16 + 8));
            }
            info = info.with_regions(&regs[..n]);
        }
        // Free (`CONVENTIONAL`) extents at 520/528.., and the GOP framebuffer at
        // 8/24/32. Both are for the frame allocator, not the map: the free list
        // is what it may hand out, the framebuffer is what it must not. Absent
        // (count 0) on a stub older than the field — deliberately not inferred
        // from the RAM extents, which include firmware-owned memory.
        let nf = rd(520) as usize;
        if nf > 0 && nf <= MAX_REGIONS {
            for i in 0..nf {
                info.free[i] = (rd(528 + i * 16), rd(528 + i * 16 + 8));
            }
            info.n_free = nf;
        }
        info.fb = (rd(8), rd(24).saturating_mul(rd(32)));
        return info;
    }
    // QEMU `-kernel`: no DTB in x0 under HVF and no stub, so read the RAM size the
    // launcher (`xtask`) published via fw_cfg (`opt/chitti/ramsize`). RAM on the
    // `virt` machine starts at 0x40000000.
    // SAFETY: single-core early boot; fw_cfg MMIO + stack buffers only.
    if let Some(bytes) = unsafe { super::ramfb::read_ram_bytes() } {
        return RamInfo::bare(0x4000_0000 + bytes, 0, (0, 0)).with_regions(&[(0x4000_0000, bytes)]);
    }
    RamInfo::bare(FALLBACK_RAM_END, 0, (0, 0))
}

/// Read `(heap_base, heap_size, model_base, model_size, total_ram)` from the
/// UEFI stub's boot-info page, if present. Layout: magic "CHITTIBI"@0,
/// heap_base@60, heap_size@68, model_base@76, model_size@84, total_ram@92
/// (little-endian; see `stub`). Pure reads (MMU may be off). Returns None
/// unless the heap region is present.
fn bootinfo_regions(bi: u64) -> Option<(u64, u64, u64, u64, u64)> {
    if bi == 0 {
        return None;
    }
    let p = bi as *const u8;
    // SAFETY: identity/flat address; read the 8-byte magic then the u64 fields.
    let magic = unsafe { core::slice::from_raw_parts(p, 8) };
    if magic != b"CHITTIBI" {
        return None;
    }
    let rd = |off: usize| -> u64 {
        // SAFETY: within the 4 KiB boot-info page.
        let b = unsafe { core::slice::from_raw_parts(p.add(off), 8) };
        u64::from_le_bytes(b.try_into().unwrap())
    };
    let (hb, hs) = (rd(60), rd(68));
    if hb == 0 || hs == 0 {
        return None;
    }
    Some((hb, hs, rd(76), rd(84), rd(92)))
}

/// Set up the identity map and enable the MMU + caches. Idempotent-ish; call
/// once, early, on the boot core. The map spans **exactly the discovered RAM**
/// (rounded up to a 1-GiB block), so it never over-maps into unbacked physical
/// space — Apple's hypervisor asserts (`isv`) on a speculative/actual access to
/// an unbacked *Normal* mapping, which a generous over-map would invite on a VM
/// with less RAM. Device MMIO above RAM (framebuffer, PCIe ECAM) is mapped
/// on demand via [`map_device_gib`]. The rounding gives up-to-1 GiB of headroom
/// above `ram_end`, covering a framebuffer that sits just past the heap.
pub fn init() {
    let info = detect();
    // Map exactly enough 1-GiB blocks to cover RAM (>=2: base is at 1 GiB), capped
    // at the single L1 table's 512 GiB.
    let map_gib = info.ram_end.div_ceil(1 << 30).clamp(2, 512);
    let regions = &info.regions[..info.n_regions];
    let mut l2_used = 0usize;
    let mut l2_overflow = false;
    // SAFETY: single-core boot; builds a valid identity map and programs the
    // standard EL1 translation registers. VA==PA, so stack/code/UART stay valid.
    unsafe {
        let l1 = core::ptr::addr_of_mut!(L1) as *mut u64;
        for i in 0..map_gib {
            let pa = i << 30; // 1 GiB blocks
            // Block 0 (below the 0x4000_0000 RAM base) is the UART/GIC MMIO.
            let kind = if i == 0 || regions.is_empty() {
                // Legacy shape (no extents known): block 0 Device, rest Normal
                // up to ram_end — correct only when RAM is one contiguous clump.
                if i == 0 { crate::mm::ramlayout::BlockKind::Device } else { crate::mm::ramlayout::BlockKind::Normal }
            } else {
                crate::mm::ramlayout::classify_gib(pa, regions)
            };
            let desc = match kind {
                crate::mm::ramlayout::BlockKind::Normal => normal_block(pa),
                crate::mm::ramlayout::BlockKind::Device => device_block(pa),
                crate::mm::ramlayout::BlockKind::Mixed => {
                    // RAM and MMIO share this GiB (VirtualBox: the model tail,
                    // GOP framebuffer and ECAM all in the 0xC000_0000 block) —
                    // split it into 2 MiB chunks typed from the real extents.
                    if l2_used < MAX_L2 {
                        let l2 = core::ptr::addr_of_mut!(L2S[l2_used]) as *mut u64;
                        l2_used += 1;
                        for c in 0..512u64 {
                            let cpa = pa + (c << 21);
                            *l2.add(c as usize) = if crate::mm::ramlayout::chunk_is_normal(cpa, regions) {
                                normal_l2_block(cpa)
                            } else {
                                device_l2_block(cpa)
                            };
                        }
                        (l2 as u64) | 0b11 // table descriptor
                    } else {
                        // Out of static L2 tables: degrade to the legacy Normal
                        // block (never silently unmapped) and say so below.
                        l2_overflow = true;
                        normal_block(pa)
                    }
                }
            };
            *l1.add(i as usize) = desc;
        }
        enable_mmu(l1);
    }
    // MMU is on now, so atomics work; publish what we discovered/mapped.
    RAM_END.store(info.ram_end, Ordering::Relaxed);
    UEFI_HEAP_BASE.store(info.uefi_heap_base, Ordering::Relaxed);
    UEFI_MODEL_BASE.store(info.uefi_model.0, Ordering::Relaxed);
    UEFI_MODEL_SIZE.store(info.uefi_model.1, Ordering::Relaxed);
    MAPPED_BYTES.store(map_gib << 30, Ordering::Relaxed);
    // Republish the extents the map was typed from, plus the boot path's free
    // list and framebuffer, so `mm` can build a frame allocator without
    // re-parsing the DTB / boot-info page (which `detect` may have read with the
    // MMU off, and which the framebuffer's own GiB may no longer be mapped for).
    HAVE_BOOTINFO.store(info.have_bootinfo, Ordering::Relaxed);
    for (i, &(b, s)) in info.regions[..info.n_regions].iter().enumerate() {
        RAM_REGIONS[i].0.store(b, Ordering::Relaxed);
        RAM_REGIONS[i].1.store(s, Ordering::Relaxed);
    }
    N_RAM_REGIONS.store(info.n_regions, Ordering::Relaxed);
    for (i, &(b, s)) in info.free[..info.n_free].iter().enumerate() {
        FREE_REGIONS[i].0.store(b, Ordering::Relaxed);
        FREE_REGIONS[i].1.store(s, Ordering::Relaxed);
    }
    N_FREE_REGIONS.store(info.n_free, Ordering::Relaxed);
    FB_REGION.0.store(info.fb.0, Ordering::Relaxed);
    FB_REGION.1.store(info.fb.1, Ordering::Relaxed);
    if l2_overflow {
        crate::ktrace::log("mmu", "too many mixed RAM/MMIO GiB blocks; some mapped Normal (raise MAX_L2)");
    }
}

/// 1 GiB **Normal** (write-back cacheable, inner-shareable) L1 block at `pa`.
fn normal_block(pa: u64) -> u64 {
    pa | (0b11 << 8) | (1 << 10) | 0b01 // attr_idx 0, SH=inner, AF=1, block
}
/// 1 GiB **Device** L1 block at `pa`.
fn device_block(pa: u64) -> u64 {
    pa | (1 << 2) | (1 << 10) | 0b01 // attr_idx 1 (Device), AF=1, block
}
/// 2 MiB **Normal** L2 block at `pa` (same encoding, level-2 granularity).
fn normal_l2_block(pa: u64) -> u64 {
    pa | (0b11 << 8) | (1 << 10) | 0b01
}
/// 2 MiB **Device** L2 block at `pa`.
fn device_l2_block(pa: u64) -> u64 {
    pa | (1 << 2) | (1 << 10) | 0b01
}

/// Top usable RAM address discovered at boot. On `-kernel` the heap is placed
/// just below this (see [`crate::mm`]).
pub fn ram_end() -> u64 {
    RAM_END.load(Ordering::Relaxed)
}

/// The UEFI stub's pre-allocated heap base, or 0 on the `-kernel` path (where the
/// kernel places the heap at the top of RAM itself).
pub fn uefi_heap_base() -> u64 {
    UEFI_HEAP_BASE.load(Ordering::Relaxed)
}

/// The UEFI stub's loaded model `(base, size)`, or `None` on the `-kernel` path
/// (where the model is at `cortex::MODEL_LOAD_ADDR`). Lets `cortex` read the
/// model wherever the firmware placed it instead of at a fixed address.
pub fn uefi_model() -> Option<(usize, usize)> {
    let base = UEFI_MODEL_BASE.load(Ordering::Relaxed);
    let size = UEFI_MODEL_SIZE.load(Ordering::Relaxed);
    if base != 0 && size != 0 {
        Some((base as usize, size as usize))
    } else {
        None
    }
}

/// Extent (bytes) of the identity map — the upper bound for any physical address
/// the kernel may dereference (e.g. the framebuffer must lie below this).
pub fn mapped_bytes() -> u64 {
    MAPPED_BYTES.load(Ordering::Relaxed)
}

/// Copy up to `out.len()` published extents out of `src`, returning the count.
fn load_regions(
    src: &[(AtomicU64, AtomicU64); MAX_REGIONS],
    n: &AtomicUsize,
    out: &mut [(u64, u64); MAX_REGIONS],
) -> usize {
    let n = n.load(Ordering::Relaxed).min(MAX_REGIONS);
    for (o, s) in out.iter_mut().zip(src.iter()).take(n) {
        *o = (s.0.load(Ordering::Relaxed), s.1.load(Ordering::Relaxed));
    }
    n
}

/// The machine's DRAM extents, as the identity map was typed from them. Written
/// into `out` (no heap: the first caller runs during `mm::init`); returns how
/// many are valid.
///
/// **This is every DRAM address, not free memory.** On the UEFI path it includes
/// firmware runtime data, ACPI tables, the loaded kernel and the stub's own
/// allocations — all correctly DRAM, none of it allocatable. Use
/// [`free_regions`] to decide what may be handed out.
pub fn ram_regions(out: &mut [(u64, u64); MAX_REGIONS]) -> usize {
    load_regions(&RAM_REGIONS, &N_RAM_REGIONS, out)
}

/// The extents the UEFI stub reported as **free** (`CONVENTIONAL`) memory — the
/// aarch64 counterpart of x86's `MEMMAP_USABLE` entries. 0 on the `-kernel` path
/// (no firmware to ask) and on a stub predating the field; a caller that gets 0
/// on the UEFI path must not substitute [`ram_regions`].
pub fn free_regions(out: &mut [(u64, u64); MAX_REGIONS]) -> usize {
    load_regions(&FREE_REGIONS, &N_FREE_REGIONS, out)
}

/// The stub's GOP framebuffer `(base, bytes)`, or `None` outside the UEFI path.
pub fn framebuffer_region() -> Option<(u64, u64)> {
    let (b, s) = (FB_REGION.0.load(Ordering::Relaxed), FB_REGION.1.load(Ordering::Relaxed));
    (b != 0 && s != 0).then_some((b, s))
}

/// Whether the kernel was entered through the UEFI stub (a valid boot-info page
/// in x1) rather than a firmware-less `-kernel`/m1n1 handoff. The two differ in
/// who owns physical memory, which is why this is reported separately from
/// whether any *particular* boot-info field was present.
pub fn have_bootinfo() -> bool {
    HAVE_BOOTINFO.load(Ordering::Relaxed)
}

/// Add a 1 GiB **Device** identity mapping for the block containing `pa`, live.
/// Used to reach PCIe ECAM config space (discovered from ACPI MCFG at a
/// high physical address outside the initial low identity map). Idempotent.
pub fn map_device_gib(pa: u64) {
    let idx = (pa >> 30) as usize;
    if idx >= 512 {
        return; // beyond the single-level L1 (512 GiB)
    }
    // SAFETY: L1 is the live TTBR0 table; writing one block descriptor + a TLB
    // invalidate publishes the new Device mapping. Device attr_idx=1, non-shareable.
    unsafe {
        let l1 = core::ptr::addr_of_mut!(L1) as *mut u64;
        let cur = *l1.add(idx);
        if cur & 0b11 == 0b11 {
            // A **mixed** block already split into 2 MiB chunks from the real
            // RAM extents: its non-RAM chunks are Device by construction, so
            // the MMIO here is reachable as-is. Never collapse the table back
            // to a Device block — that's exactly the bug that Device-typed the
            // model tail sharing the ECAM's GiB on VirtualBox.
            return;
        }
        let desc = ((idx as u64) << 30) | (1u64 << 2) | (1 << 10) | 0b01; // AF=1, Device, block, valid
        if cur == desc {
            return; // already the exact Device block — no live change, no BBM.
        }
        // Break-before-make: `mmu::init` already mapped this GiB (as a Normal or
        // Device block), so this is a *live* valid->valid entry change. Real
        // Apple cores fault such a change done in place (a TLB conflict abort) —
        // the QEMU/hv paths are lenient (the hv's stage-2 masks the stage-1 TLB),
        // which is why this only bit the bare Mac boot. Invalidate first, then
        // write the new block. Safe because no code/stack/framebuffer lives in
        // the MMIO GiB being remapped.
        *l1.add(idx) = 0; // break: make the entry invalid
        asm!("dsb ish", "tlbi vmalle1", "dsb ish", options(nostack));
        *l1.add(idx) = desc; // make: publish the Device block
        asm!("dsb ish", "isb", options(nostack));
    }
}

/// Identity-map the 1 GiB **Normal** RAM block containing `pa`, but only if that
/// block is currently **unmapped** — never disturbing a live mapping (so it can't
/// retype the running stack/heap/code). Used to reach RAM that lies above the
/// `/memory`-reported top the initial map covered: m1n1 parks its DTB in its own
/// heap, which can sit above the RAM top it hands the payload, so the FDT is
/// readable with the MMU off (flat physical) yet faults once the MMU is on. This
/// makes the FDT's GiB reachable for any MMU-on parse. Invalid->valid needs no
/// break-before-make; a TLBI covers a possibly-cached faulting entry.
pub fn map_ram_gib_if_unmapped(pa: u64) {
    let idx = (pa >> 30) as usize;
    if idx >= 512 {
        return;
    }
    // SAFETY: L1 is the live TTBR0 table; we only write an entry that is
    // currently invalid, so no live translation changes (no BBM needed).
    unsafe {
        let l1 = core::ptr::addr_of_mut!(L1) as *mut u64;
        if *l1.add(idx) & 0b11 != 0 {
            return; // already a valid block or table — leave it untouched
        }
        *l1.add(idx) = normal_block((idx as u64) << 30);
        asm!("dsb ish", "tlbi vmalle1", "dsb ish", "isb", options(nostack));
    }
}

// --- 4 KiB page mapping over the boot identity map -----------------------

/// The hardware's half of [`crate::mm::walk`]: barriers and TLB maintenance.
/// The descriptor manipulation, block splitting and break-before-make ordering
/// live there, arch-neutral, so the x86-only unit suite actually executes them.
struct HwMmu;

impl crate::mm::walk::Mmu for HwMmu {
    // `table_ptr` is the default identity: this kernel runs VA == PA, so a
    // table's physical address *is* where it is readable.

    fn publish(&self) {
        // SAFETY: a store barrier. The table walker's accesses are
        // inner-shareable cacheable (see `enable_mmu`'s TCR), so they are
        // coherent with these stores once ordered — no cache maintenance needed.
        unsafe { asm!("dsb ishst", options(nostack, preserves_flags)) };
    }

    fn invalidate_page(&self, va: u64) {
        // SAFETY: TLB maintenance is always architecturally safe. `vaae1is` is
        // the by-address, all-ASID, EL1, **inner-shareable** form: the shareable
        // domain is what makes the other cores drop the entry too, which the
        // non-shareable `tlbi vmalle1` this file used to reach for does not do.
        // The operand is the VA in units of 4 KiB pages, not bytes.
        unsafe {
            asm!("tlbi vaae1is, {}", "dsb ish", "isb", in(reg) va >> 12, options(nostack, preserves_flags));
        }
    }

    fn invalidate_all(&self) {
        // SAFETY: as above, for every entry — the scope a block split needs.
        unsafe { asm!("tlbi vmalle1is", "dsb ish", "isb", options(nostack, preserves_flags)) };
    }
}

/// The kernel's root (level 1) translation table, by physical address. This is
/// the table `TTBR0_EL1` points at on every core.
pub fn kernel_root() -> u64 {
    core::ptr::addr_of!(L1) as u64
}

/// Map the 4 KiB page at `va` to `pa` with `attrs` (see [`crate::mm::armv8`]) in
/// the live kernel translation table, taking intermediate tables from the
/// physical frame allocator.
///
/// **Blocks are never split here.** The boot identity map is built from 1 GiB and
/// 2 MiB blocks, and every one of them covers running code, the stack, the heap,
/// or MMIO a driver is mid-transaction with — while a split leaves that whole
/// range unmapped between the break and the make, for every core, not just this
/// one. So a `va` already covered by a block is refused
/// ([`crate::mm::walk::MapError::WouldSplitBlock`]) rather than silently
/// risked; splitting is for a table the caller owns exclusively, which is what
/// per-task address spaces will build.
///
/// In practice that costs nothing: the addresses a new mapping wants are the ones
/// the boot map left *invalid*, where the walk allocates tables and no split
/// arises.
pub fn map_page(va: u64, pa: u64, attrs: u64) -> Result<(), crate::mm::walk::MapError> {
    use crate::mm::walk::{self, MapError, Split};
    // `Locked::with` runs with interrupts disabled, which is also what the walk
    // needs: it must not be interrupted between a break and its make.
    crate::mm::FRAME_ALLOCATOR.with(|slot| match slot.as_mut() {
        // SAFETY: `kernel_root` is the live L1 table `TTBR0_EL1` uses, reachable
        // at its physical address (VA == PA), and `Split::Refuse` keeps the walk
        // from unmapping any range this core is running out of.
        Some(alloc) => unsafe {
            walk::map_page(&HwMmu, kernel_root(), va, pa, attrs, Split::Refuse, alloc)
        },
        None => Err(MapError::NoFrameAllocator),
    })
}

/// Unmap the 4 KiB page at `va` from the live kernel table; `true` if something
/// was mapped there. A `va` covered by a **block** is left alone (reported
/// `false`) — see [`crate::mm::walk::unmap_page`].
pub fn unmap_page(va: u64) -> bool {
    // SAFETY: the live L1 table, reachable at its physical address.
    crate::arch::interrupts::without_interrupts(|| unsafe {
        crate::mm::walk::unmap_page(&HwMmu, kernel_root(), va)
    })
}

/// The physical address `va` currently translates to, resolving blocks as well
/// as pages — i.e. what the hardware walker would find. For diagnostics and for
/// confirming a mapping took, rather than discovering it did not via a fault.
pub fn translate(va: u64) -> Option<u64> {
    // SAFETY: the live L1 table, reachable at its physical address.
    unsafe { crate::mm::walk::translate(&HwMmu, kernel_root(), va) }
}

/// Prove the 4 KiB walker works on *this* machine, once, at boot.
///
/// [`crate::mm::walk`]'s logic is covered by the x86 unit suite, but its
/// platform half is not: the `tlbi vaae1is` operand scaling, where the `dsb`s
/// sit, and whether a fresh mapping is actually usable are properties of the
/// hardware, and nothing else in the kernel calls [`map_page`] yet. So this maps
/// a frame at an unused high virtual address, writes a pattern through the new
/// mapping and reads it back, then unmaps it — the same posture as the SMP wake
/// self-test, and for the same reason: a mapping facility that silently does not
/// work is far worse than one that says so at boot.
///
/// The write is the load-bearing part. Installing a descriptor over a previously
/// *invalid* entry needs no break-before-make, but the architecture does permit a
/// cached faulting entry — so the store only lands if the invalidate after the
/// make is right. A wrong `tlbi` operand shows up here as a fault or a stale
/// read, not as a compile error.
///
/// Returns `Err` with a short reason instead of panicking; the caller ktraces it.
pub fn walker_self_test() -> Result<(), &'static str> {
    use crate::mm::armv8;
    // The last gigabyte of the 39-bit VA space: past anything `init` mapped on
    // any plausible machine, and checked for emptiness rather than assumed.
    let va = (1u64 << armv8::VA_BITS) - (1 << 30);
    if translate(va).is_some() {
        return Err("test address is already mapped");
    }
    let phys = crate::mm::FRAME_ALLOCATOR
        .with(|slot| slot.as_mut().and_then(|a| a.allocate()))
        .ok_or("no frame to map")?;
    let attrs = armv8::normal_attrs() | armv8::UXN | armv8::PXN;
    let outcome = map_page(va, phys, attrs)
        .map_err(|_| "map_page refused")
        .and_then(|()| {
            if translate(va) != Some(phys) {
                return Err("mapping does not read back");
            }
            const PATTERN: u64 = 0x0123_4567_89ab_cdef;
            // SAFETY: `va` was unmapped a moment ago and now maps `phys`, a
            // freshly-allocated frame owned by nothing else. Volatile so the
            // write and the read cannot be folded away.
            let seen = unsafe {
                core::ptr::write_volatile(va as *mut u64, PATTERN);
                core::ptr::read_volatile(va as *const u64)
            };
            if seen != PATTERN {
                return Err("write through the new mapping did not land");
            }
            // The same bytes must be visible at the frame's identity address —
            // proof the mapping points where it claims rather than somewhere
            // that merely happens to be writable.
            // SAFETY: `phys` is an owned frame inside the identity map.
            if unsafe { core::ptr::read_volatile(phys as *const u64) } != PATTERN {
                return Err("the mapping does not alias the frame it names");
            }
            Ok(())
        });
    // Always tear down, whatever happened: leaving a stray high mapping (and a
    // leaked frame) behind would be a worse legacy than a failed self-test.
    let unmapped = unmap_page(va);
    crate::mm::FRAME_ALLOCATOR.with(|slot| {
        if let Some(a) = slot.as_mut() {
            a.free(phys)
        }
    });
    outcome?;
    if !unmapped || translate(va).is_some() {
        return Err("unmap did not clear the mapping");
    }
    Ok(())
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
        // A=0 (bit 1): alignment checking OFF — a load-bearing invariant, not
        // an inherited default. The NEON kernels use unaligned `ldr q` inline
        // asm (the +strict-align rule), which is only architecturally legal on
        // Normal memory with A=0; on the UEFI path the firmware's SCTLR is
        // whatever it left behind (VirtualBox-ARM can hand off with A=1, which
        // alignment-faults the first SDOT matvec).
        sctlr &= !(1 << 1);
        asm!("msr sctlr_el1, {}", in(reg) sctlr, options(nostack));
        asm!("isb", options(nostack));
    }
}
