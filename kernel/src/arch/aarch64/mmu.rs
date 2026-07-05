//! Minimal aarch64 MMU bring-up: an identity map over the low 4 GiB with 1 GiB
//! blocks (RAM = Normal write-back cacheable, low 1 GiB = Device for the PL011
//! UART / GIC), then MMU + I/D caches on. QEMU enters with the MMU off, where
//! all memory is device-typed and uncached -- NEON is unreliable and slow
//! there -- so this must run before any NEON/cached work.

use core::arch::asm;
use core::sync::atomic::{AtomicU64, Ordering};

#[repr(align(4096))]
struct Table(#[allow(dead_code)] [u64; 512]);
static mut L1: Table = Table([0; 512]);
/// L2 tables for GiB blocks that must be **split** (a device window inside a
/// RAM block — VirtualBox puts the GOP framebuffer at 0xE000_0000, inside
/// mapped RAM, and remapping that whole GiB as Device turned the tail of the
/// model region into Device memory: unaligned SIMD loads alignment-faulted).
/// A small fixed pool; each entry maps one GiB as 512 × 2 MiB blocks.
static mut L2_POOL: [Table; 4] = [const { Table([0; 512]) }; 4];
/// Which L1 index each pool slot serves (u64::MAX = free). Single-core boot
/// path; no locking needed (map_device_gib runs during init, BSP only).
static L2_OWNER: [AtomicU64; 4] = [const { AtomicU64::new(u64::MAX) }; 4];

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

/// What [`detect`] found: the usable top address, and (UEFI only) the stub's
/// pre-allocated heap and model regions.
struct RamInfo {
    ram_end: u64,
    uefi_heap_base: u64,
    uefi_model: (u64, u64),
}

/// Discover RAM at boot. Runs with the MMU **off** (before the map is built), so
/// it only does pure reads — no atomics, no heap.
fn detect() -> RamInfo {
    // `-kernel`: QEMU passes the DTB in x0; parse its `/memory` node.
    // SAFETY: `boot_x0()` is the DTB pointer (or non-FDT, rejected by magic).
    if let Some((b, s)) = unsafe { super::dtb::memory_region(super::boot::boot_x0()) } {
        return RamInfo { ram_end: b.saturating_add(s), uefi_heap_base: 0, uefi_model: (0, 0) };
    }
    // UEFI/stub: the boot-info page carries the heap and model regions the stub
    // allocated (AnyPages). The map must cover both.
    if let Some((hb, hs, mb, ms)) = bootinfo_regions(super::boot::boot_x1()) {
        let ram_end = hb.saturating_add(hs).max(mb.saturating_add(ms));
        return RamInfo { ram_end, uefi_heap_base: hb, uefi_model: (mb, ms) };
    }
    // QEMU `-kernel`: no DTB in x0 under HVF and no stub, so read the RAM size the
    // launcher (`xtask`) published via fw_cfg (`opt/chitti/ramsize`). RAM on the
    // `virt` machine starts at 0x40000000.
    // SAFETY: single-core early boot; fw_cfg MMIO + stack buffers only.
    if let Some(bytes) = unsafe { super::ramfb::read_ram_bytes() } {
        return RamInfo { ram_end: 0x4000_0000 + bytes, uefi_heap_base: 0, uefi_model: (0, 0) };
    }
    RamInfo { ram_end: FALLBACK_RAM_END, uefi_heap_base: 0, uefi_model: (0, 0) }
}

/// Read `(heap_base, heap_size, model_base, model_size)` from the UEFI stub's
/// boot-info page, if present. Layout: magic "CHITTIBI"@0, heap_base@60,
/// heap_size@68, model_base@76, model_size@84 (little-endian; see `stub`). Pure
/// reads (MMU may be off). Returns None unless the heap region is present.
fn bootinfo_regions(bi: u64) -> Option<(u64, u64, u64, u64)> {
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
    Some((hb, hs, rd(76), rd(84)))
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
    // SAFETY: single-core boot; builds a valid identity map and programs the
    // standard EL1 translation registers. VA==PA, so stack/code/UART stay valid.
    unsafe {
        let l1 = core::ptr::addr_of_mut!(L1) as *mut u64;
        for i in 0..map_gib {
            let pa = i << 30; // 1 GiB blocks
            let attr_idx = if i == 0 { 1u64 } else { 0u64 }; // 0: Device MMIO, else Normal
            let sh = if i == 0 { 0u64 } else { 0b11u64 }; // inner-shareable for Normal
            let desc = pa | (attr_idx << 2) | (sh << 8) | (1 << 10) | 0b01; // AF=1, block, valid
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

/// Add a 1 GiB **Device** identity mapping for the block containing `pa`, live.
/// Used to reach PCIe ECAM config space (discovered from ACPI MCFG at a
/// high physical address outside the initial low identity map). Idempotent.
pub fn map_device_gib(pa: u64) {
    let idx = (pa >> 30) as usize;
    if idx >= 512 {
        return; // beyond the single-level L1 (512 GiB)
    }
    // A device inside a **RAM** GiB block (VirtualBox's GOP framebuffer sits at
    // 0xE000_0000, below RAM's end) must not demote the whole block: the model
    // and heap can share it, and Device-typed RAM alignment-faults every
    // unaligned SIMD access. Split such a block into 2 MiB L2 entries and punch
    // a 64 MiB Device window (covers any framebuffer) around `pa` instead.
    let in_ram = idx >= 1 && ((idx as u64) << 30) < MAPPED_BYTES.load(Ordering::Relaxed);
    if in_ram {
        map_device_window_2m(pa, 64 << 20);
        return;
    }
    // SAFETY: L1 is the live TTBR0 table; writing one block descriptor + a TLB
    // invalidate publishes the new Device mapping. Device attr_idx=1, non-shareable.
    unsafe {
        let l1 = core::ptr::addr_of_mut!(L1) as *mut u64;
        let desc = ((idx as u64) << 30) | (1u64 << 2) | (1 << 10) | 0b01; // AF=1, Device, block, valid
        *l1.add(idx) = desc;
        asm!("dsb ish", "tlbi vmalle1", "dsb ish", "isb", options(nostack));
    }
}

/// Split the GiB block containing `pa` into 512 × 2 MiB L2 blocks (RAM stays
/// Normal) and mark `[pa & !2MiB-1, +len)` Device. Idempotent per block; falls
/// back to the whole-GiB Device demotion if the small L2 pool is exhausted
/// (previous behavior — correct for the device, slow if RAM shares the block).
fn map_device_window_2m(pa: u64, len: u64) {
    let idx = (pa >> 30) as usize;
    // Find (or claim) the pool slot for this L1 index.
    let mut slot = usize::MAX;
    for (i, owner) in L2_OWNER.iter().enumerate() {
        let o = owner.load(Ordering::Relaxed);
        if o == idx as u64 {
            slot = i;
            break;
        }
        if o == u64::MAX && slot == usize::MAX {
            slot = i;
        }
    }
    if slot == usize::MAX {
        // Pool exhausted: previous whole-GiB behavior.
        // SAFETY: same single-descriptor write as map_device_gib's device path.
        unsafe {
            let l1 = core::ptr::addr_of_mut!(L1) as *mut u64;
            *l1.add(idx) = ((idx as u64) << 30) | (1u64 << 2) | (1 << 10) | 0b01;
            asm!("dsb ish", "tlbi vmalle1", "dsb ish", "isb", options(nostack));
        }
        return;
    }
    // SAFETY: single-core init path (BSP); the pool table is exclusively ours
    // once L2_OWNER is claimed, and the L1 slot swap below is a single aligned
    // 64-bit store followed by TLB invalidation.
    unsafe {
        let l2 = core::ptr::addr_of_mut!(L2_POOL[slot]) as *mut u64;
        if L2_OWNER[slot].load(Ordering::Relaxed) != idx as u64 {
            // Fresh split: fill all 512 entries as Normal 2 MiB blocks.
            for j in 0..512 {
                let base = ((idx as u64) << 30) | ((j as u64) << 21);
                *l2.add(j) = base | (0u64 << 2) | (0b11 << 8) | (1 << 10) | 0b01; // Normal, inner-shareable
            }
            L2_OWNER[slot].store(idx as u64, Ordering::Relaxed);
        }
        // Punch the Device window (2 MiB granules).
        let start = (pa & !((1 << 21) - 1)).max((idx as u64) << 30);
        let end = (pa + len).min(((idx as u64) + 1) << 30);
        let mut a = start;
        while a < end {
            let j = ((a >> 21) & 0x1ff) as usize;
            *l2.add(j) = a | (1u64 << 2) | (1 << 10) | 0b01; // Device, non-shareable
            a += 1 << 21;
        }
        // Swap the L1 entry to the table descriptor and publish.
        let l1 = core::ptr::addr_of_mut!(L1) as *mut u64;
        *l1.add(idx) = (l2 as u64) | 0b11; // table descriptor
        asm!("dsb ish", "tlbi vmalle1", "dsb ish", "isb", options(nostack));
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
        // A (alignment check) must be OFF: the SIMD hot loops (tensor.rs
        // `ldq_*`/`dot_f32_neon` — `ldr q`/`ldp q` on 1-/2-byte-aligned Q8_0
        // block data) rely on Normal-memory unaligned access. QEMU's EDK2
        // hands off with A=0, but VirtualBox's EFI leaves A=1 — inheriting it
        // made the first quantized matvec alignment-fault (ESR DFSC 0x21).
        sctlr &= !(1 << 1);
        asm!("msr sctlr_el1, {}", in(reg) sctlr, options(nostack));
        asm!("isb", options(nostack));
    }
}
