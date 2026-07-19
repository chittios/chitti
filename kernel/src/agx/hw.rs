//! AGX hardware orchestration (aarch64 / real Apple Silicon): FDT discovery, the
//! minimal PMGR power-domain enable, and the RTKit boot handshake that ties the
//! pure [`super::proto`] codecs to the [`super::asc`] mailbox transport. Ported
//! from m1n1's `rtkit_boot` (`third_party/m1n1/src/rtkit.c:496-657`) but
//! cooperative + bounded: every wait pumps the UI/clock/net and answers Ctrl+C.

use super::asc::{Asc, Message};
use super::handoff;
use super::proto::{self, Action, BufferKind, EndpointSet, RtkitState};
use super::uat;
use alloc::alloc::{alloc_zeroed, Layout};

/// t8112 GPU node: `gpu@206400000`, compatible `apple,agx-t8112`. Its `reg[0]`
/// ("asc") is the coprocessor CPU base; the mailbox FIFO sits at +0x8000
/// (`asc.rs` folds that in). Verified against `third_party/dtb/t8112-j473.dtb`.
const AGX_COMPAT: &[u8] = b"apple,agx-t8112";

/// The `gfx` power domain's PMGR pwrstate register, absolute. From
/// `third_party/dtb/t8112-j473.dtb`: `power-management@23b700000` (the t8112
/// PMGR, `reg = <0x2 0x3b700000 …>`) + its child `power-controller@430`
/// (`label = "gfx"`). t8112-specific — only ever reached after the
/// `apple,agx-t8112` GPU node is confirmed present, so it can't misfire on
/// another SoC. Mandatory before touching ASC MMIO (m1n1 `kboot_gpu.c`).
const GFX_PWRSTATE: u64 = 0x2_3b70_0430;
const PMGR_PS_TARGET: u32 = 0xf; // bits[3:0]  — TARGET field
const PMGR_PS_ACTUAL_SHIFT: u32 = 4; // bits[7:4] — ACTUAL field
const PMGR_PS_ACTIVE: u32 = 0xf;
/// Bits `pmgr_set_mode` clears before setting TARGET (auto-enable + the sticky
/// was-clock/power-gated bits): `AUTO_ENABLE(28) | WAS_CLKGATED(9) | WAS_PWRGATED(8) | TARGET(3:0)`.
const PMGR_CLEAR: u32 = (1 << 28) | (1 << 9) | (1 << 8) | 0xf;

/// Timeouts (ms) — generous, cooperative, always bounded.
const HELLO_TIMEOUT_MS: u64 = 1000;
const STEP_TIMEOUT_MS: u64 = 1000;
const SEND_TIMEOUT_MS: u64 = 200;
/// Overall budget for the "wait until the IOP powers ON" phase. The IOP may take
/// several seconds to init its logs + power on after we grant its buffers, so
/// this is a *total* deadline (not per-message) — a slow-but-progressing IOP is
/// not cut off, and only true silence for the whole budget is a timeout.
const POWERUP_BUDGET_MS: u64 = 15000;
/// Per-poll recv window inside the power-up budget (short, so the heartbeat +
/// interrupt stay responsive).
const POWERUP_POLL_MS: u64 = 500;

/// The outcome of a bring-up attempt (drives the `/agx` report + message).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Outcome {
    #[default]
    NotRun,
    /// No `apple,agx-t8112` node — not a t8112, or no FDT (QEMU/VBox).
    NoGpu,
    /// `cpu_start` done but no HELLO in ~1 s — **firmware not resident** (the
    /// external Asahi GPU-firmware-provisioning blocker).
    NoHello,
    BadHello,
    VersionMismatch,
    EpmapFail,
    SendFail,
    /// Timed out waiting for the IOP to reach power ON.
    Timeout,
    /// The IOP reported a crash during boot.
    Crashed,
    /// Full handshake complete — coprocessor RUNNING.
    Running,
}

/// Snapshot of the last bring-up, for `/agx status`.
#[derive(Clone, Copy, Default)]
struct Report {
    outcome: Outcome,
    asc_base: u64,
    version: u64,
    eps: EndpointSet,
    iop_power: u64,
    ap_power: u64,
    n_buffers: u32,
    cpu_running: bool,
    crashlog_phys: u64,
}

static REPORT: crate::mm::Locked<Report> = crate::mm::Locked::new(Report {
    outcome: Outcome::NotRun,
    asc_base: 0,
    version: 0,
    eps: EndpointSet { crashlog: false, debug: false, ioreport: false, syslog: false, oslog: false },
    iop_power: 0,
    ap_power: 0,
    n_buffers: 0,
    cpu_running: false,
    crashlog_phys: 0,
});

/// True if `needle` appears in the FDT `/chosen bootargs` (the m1n1 `-b` command
/// line). Gates the bring-up on `chitti.agx`, mirroring `apple_usb`'s
/// `chitti.usb`.
fn bootarg_present(needle: &[u8]) -> bool {
    let fdt = crate::arch::aarch64::boot::boot_x0();
    // SAFETY: boot_x0 is the FDT pointer (or non-FDT, rejected by the magic).
    if let Some(c) = unsafe { crate::fdt::chosen(fdt) } {
        if !c.bootargs_ptr.is_null() && c.bootargs_len >= needle.len() {
            // SAFETY: `[bootargs_ptr, +len)` views the still-mapped FDT.
            let s = unsafe { core::slice::from_raw_parts(c.bootargs_ptr, c.bootargs_len) };
            return s.windows(needle.len()).any(|w| w == needle);
        }
    }
    false
}

/// Discover the AGX ASC CPU base from the boot FDT (`reg[0]` of the
/// `apple,agx-t8112` GPU node). `None` when absent (QEMU/VBox/other SoC).
fn discover_asc_base() -> Option<u64> {
    let fdt = crate::arch::aarch64::boot::boot_x0();
    // SAFETY: boot_x0 is the FDT pointer (or non-FDT, rejected by the magic).
    let (base, _size) = unsafe { crate::fdt::reg_of_compatible(fdt, AGX_COMPAT) }?;
    Some(base)
}

/// Read/modify/write a 32-bit PMGR register (single `ldr`/`str`, the MMIO rule).
fn pmgr_r32(addr: u64) -> u32 {
    // SAFETY: single 32-bit MMIO read of a mapped PMGR register.
    unsafe { core::ptr::read_volatile(addr as *const u32) }
}
fn pmgr_w32(addr: u64, v: u32) {
    // SAFETY: single 32-bit MMIO write of a mapped PMGR register.
    unsafe { core::ptr::write_volatile(addr as *mut u32, v) }
}

/// Enable the `gfx` power domain (m1n1 `pmgr_set_mode` → ACTIVE): clear the
/// auto/gated bits + TARGET, set TARGET=ACTIVE, poll ACTUAL==ACTIVE. Bounded.
/// Returns false if the domain never reaches ACTIVE.
fn pmgr_enable_gfx() -> bool {
    crate::arch::aarch64::mmu::map_device_gib(GFX_PWRSTATE);
    let cur = pmgr_r32(GFX_PWRSTATE);
    let next = (cur & !PMGR_CLEAR) | PMGR_PS_TARGET;
    pmgr_w32(GFX_PWRSTATE, next);
    let deadline = crate::arch::now_ms() + 100;
    while crate::arch::now_ms() < deadline {
        let actual = (pmgr_r32(GFX_PWRSTATE) >> PMGR_PS_ACTUAL_SHIFT) & 0xf;
        if actual == PMGR_PS_ACTIVE {
            crate::ktrace::log_fmt(format_args!("agx: gfx power domain ACTIVE (pmgr {:#x}={:#x})", GFX_PWRSTATE, pmgr_r32(GFX_PWRSTATE)));
            return true;
        }
        core::hint::spin_loop();
    }
    crate::ktrace::log_fmt(format_args!("agx: gfx power domain did NOT reach ACTIVE (pmgr {:#x}={:#x})", GFX_PWRSTATE, pmgr_r32(GFX_PWRSTATE)));
    false
}

/// Allocate a zeroed, 16 KiB-aligned DMA buffer of `pages`×4 KiB for an RTKit
/// shared region. aarch64 identity map ⇒ phys == the returned pointer. Leaked
/// for the coprocessor's lifetime. `None` on OOM.
fn alloc_shared(pages: u64) -> Option<u64> {
    let bytes = (pages.max(1) as usize) * 4096;
    let layout = Layout::from_size_align(bytes, 16384).ok()?;
    // SAFETY: nonzero, 16 KiB-aligned; leaked as device-shared DMA (VA == PA).
    let p = unsafe { alloc_zeroed(layout) } as u64;
    (p != 0).then_some(p)
}

/// Latched Ctrl+C during a bring-up (reset at the start of every `up()`). Once
/// set, every subsequent `pump()` reports abort, so a multi-second wait bails
/// promptly on the first Ctrl+C rather than only shortening the current poll.
static ABORT: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

// ===================== Milestone 2: minimal GPU UAT ======================
//
// The gfx-asc addresses DRAM only through its UAT (ARMv8 16 KiB page tables),
// so an RTKit shared buffer must be mapped into the kernel context (ctx 0) and
// handed back as a GPU VA — that is what unblocks the power-ON stall. We reuse
// the physical carveouts m1n1 already allocated (ttbs / pagetables) and program
// ctx-0's TTBR0 to a fresh L1 table of our own, keeping TTBR1 = the firmware's
// shared-region tables (its own FW-side mappings). Mirrors the m1n1 proxyclient
// (`m1n1/hw/uat.py`, `m1n1/agx/__init__.py`).

/// GPU VA space we hand RTKit shared buffers out of. RTKit buffers MUST live in
/// the shared **TTBR1** kernel range, not per-context TTBR0: during boot the
/// firmware runs in its own context (ctx-0 TTBR0 isn't active), but TTBR1 is
/// shared across all contexts (the firmware's `pagetables` region), so a TTBR1
/// buffer is reachable from the firmware's boot context. Set to the SAME VA the
/// working proxyclient uses — `kern_va_base + 0x80000000` (kern_va_base =
/// rtkit-private-vm-region-base 0xffffff8000000000 + size 0x2000000000) — so the
/// crashlog/RTKit buffers land exactly where the replayed initdata expects them.
const BUF_VA_BASE: u64 = 0xffffffa080000000;

struct UatState {
    ready: bool,
    ttbs: u64,       // gpu-region carveout phys (per-context TTBR pairs)
    pagetables: u64, // gfx-shared-region carveout phys (firmware TTBR1)
    handoff: u64,    // GFXHandoff carveout phys (PPL lock for TTBR writes)
    l1: u64,         // our ctx-0 TTBR0 L1 table phys
    va_next: u64,    // bump allocator for buffer GPU VAs
}

static UAT: crate::mm::Locked<UatState> =
    crate::mm::Locked::new(UatState { ready: false, ttbs: 0, pagetables: 0, handoff: 0, l1: 0, va_next: BUF_VA_BASE });

/// Clean `len` bytes of DRAM from `pa` to the point of coherency so the GPU's
/// MMU walker sees our freshly-written PTEs (the xhci DART-publish pattern:
/// `dc cvac` per line + `dsb sy`). aarch64 identity map ⇒ VA == PA.
fn dcache_clean(pa: u64, len: u64) {
    // SAFETY: cache maintenance over mapped Normal DRAM (a page table / TTBR).
    unsafe {
        let mut p = pa & !63;
        let end = pa + len;
        while p < end {
            core::arch::asm!("dc cvac, {}", in(reg) p, options(nostack, preserves_flags));
            p += 64;
        }
        core::arch::asm!("dsb sy", options(nostack, preserves_flags));
    }
}

/// Invalidate `len` bytes of DRAM from `pa` (clean+invalidate) so a subsequent
/// CPU read observes what the **GPU/firmware** wrote (not a stale cache line).
/// The mirror of [`dcache_clean`] for the firmware→AP direction.
fn dcache_invalidate(pa: u64, len: u64) {
    // SAFETY: cache maintenance over mapped Normal DRAM the firmware wrote.
    unsafe {
        core::arch::asm!("dsb sy", options(nostack, preserves_flags));
        let mut p = pa & !63;
        let end = pa + len;
        while p < end {
            core::arch::asm!("dc civac, {}", in(reg) p, options(nostack, preserves_flags));
            p += 64;
        }
        core::arch::asm!("dsb sy", options(nostack, preserves_flags));
    }
}

// ---- Replay range map: GPU-VA → identity-mapped phys, for reading back what
// the firmware writes into shared memory (channel rings, status flags). Recorded
// once during `replay_initdata`; read-only afterward. Single-threaded within the
// `/agx` command, but Locked keeps it sound. ----
const MAX_REPLAY_RANGES: usize = 96;

struct ReplayMap {
    /// (gpu_va, phys, size) for each replayed DRAM range. phys == CPU VA (identity).
    ranges: [(u64, u64, u64); MAX_REPLAY_RANGES],
    n: usize,
}

static REPLAY_MAP: crate::mm::Locked<ReplayMap> =
    crate::mm::Locked::new(ReplayMap { ranges: [(0, 0, 0); MAX_REPLAY_RANGES], n: 0 });

fn replay_map_reset() {
    REPLAY_MAP.with(|m| m.n = 0);
}

fn replay_map_record(va: u64, phys: u64, size: u64) {
    REPLAY_MAP.with(|m| {
        if m.n < MAX_REPLAY_RANGES {
            let i = m.n;
            m.ranges[i] = (va, phys, size);
            m.n += 1;
        }
    });
}

/// Translate a GPU VA back to the identity-mapped phys we replayed it at, if it
/// falls inside a recorded range. Returns `(phys, bytes_left_in_range)`.
fn gpu_va_to_phys(va: u64) -> Option<(u64, u64)> {
    REPLAY_MAP.with(|m| {
        for &(base, phys, size) in &m.ranges[..m.n] {
            if va >= base && va < base + size {
                let off = va - base;
                return Some((phys + off, size - off));
            }
        }
        None
    })
}

/// Read up to `buf.len()` bytes of GPU shared memory at `va` into `buf`, first
/// invalidating the cache so the firmware's writes are visible. Returns how many
/// bytes were read (0 if `va` isn't in a replayed range). Never crosses a range.
fn gpu_read(va: u64, buf: &mut [u8]) -> usize {
    let Some((phys, avail)) = gpu_va_to_phys(va) else {
        return 0;
    };
    let n = buf.len().min(avail as usize);
    dcache_invalidate(phys, n as u64);
    // SAFETY: `phys` is our own identity-mapped, replayed DRAM range; `n` stays
    // within the range's bounds (clamped by `avail`).
    unsafe { core::ptr::copy_nonoverlapping(phys as *const u8, buf.as_mut_ptr(), n) };
    n
}

fn gpu_read_u32(va: u64) -> Option<u32> {
    let mut b = [0u8; 4];
    (gpu_read(va, &mut b) == 4).then(|| u32::from_le_bytes(b))
}

/// Write `buf` into GPU shared memory at `va` (translating via the replay map)
/// and clean it to the point of coherency so the firmware's UAT walker + reads
/// observe it. Returns bytes written (0 if `va` isn't in a replayed range).
fn gpu_write(va: u64, buf: &[u8]) -> usize {
    let Some((phys, avail)) = gpu_va_to_phys(va) else {
        return 0;
    };
    let n = buf.len().min(avail as usize);
    // SAFETY: `phys` is our own identity-mapped, replayed DRAM range; `n` clamped
    // to the range's remaining bytes.
    unsafe { core::ptr::copy_nonoverlapping(buf.as_ptr(), phys as *mut u8, n) };
    dcache_clean(phys, n as u64);
    n
}

fn gpu_write_u32(va: u64, val: u32) -> bool {
    gpu_write(va, &val.to_le_bytes()) == 4
}

/// Read a 64-bit page-table entry at identity-mapped physical `pa`.
fn rd64(pa: u64) -> u64 {
    // SAFETY: single aligned 64-bit read of mapped DRAM (a page-table slot).
    unsafe { core::ptr::read_volatile(pa as *const u64) }
}
/// Write a 64-bit page-table entry at identity-mapped physical `pa`.
fn wr64(pa: u64, v: u64) {
    // SAFETY: single aligned 64-bit write of mapped DRAM (a page-table slot).
    unsafe { core::ptr::write_volatile(pa as *mut u64, v) }
}

/// Resolve a GPU node `memory-region` carveout by its `memory-region-names`
/// entry (e.g. `b"ttbs"`, `b"pagetables"`) to its live `(base, size)`. Returns
/// `None` if absent/unpopulated.
fn discover_carveout(want: &[u8]) -> Option<(u64, u64)> {
    let fdt = crate::arch::aarch64::boot::boot_x0();
    let mut phandles = [0u32; 8];
    // SAFETY: boot_x0 is the FDT pointer (or non-FDT, rejected by the magic).
    let n = unsafe { crate::fdt::prop_cells_of_compatible(fdt, AGX_COMPAT, b"memory-region", &mut phandles) };
    // Find `want`'s index in memory-region-names, then resolve that phandle.
    let mut idx = None;
    let mut collected = 0usize;
    // SAFETY: boot_x0 is the FDT pointer (or non-FDT, rejected by the magic).
    unsafe {
        crate::fdt::for_each_prop_of_compatible(fdt, AGX_COMPAT, &mut |name, val| {
            if name == b"memory-region-names" {
                for (i, s) in val.split(|&b| b == 0).filter(|s| !s.is_empty()).enumerate() {
                    if s == want {
                        idx = Some(i);
                    }
                    collected = i + 1;
                }
            }
        });
    }
    let _ = collected;
    let i = idx?;
    if i >= n {
        return None;
    }
    // SAFETY: boot_x0 is the FDT pointer (or non-FDT, rejected by the magic).
    let (b, s) = unsafe { crate::fdt::reg_by_phandle(fdt, phandles[i])? };
    (b != 0 && s != 0).then_some((b, s))
}

/// Bring up the ctx-0 kernel page tables. The firmware is **already running**
/// (it did HELLO/EPMAP), so iBoot has set up its ctx-0 TTBR0 with the firmware's
/// own mappings — we must **reuse** that live L1, not replace it, or the
/// firmware loses its mappings and hangs. Read the existing ctx-0 TTBR0 from the
/// `ttbs` carveout and adopt its L1 table; only if it is empty do we allocate a
/// fresh one and program the TTBR pair. Idempotent; false if a carveout is
/// missing.
fn uat_init() -> bool {
    if UAT.with(|u| u.ready) {
        return true;
    }
    let Some((ttbs, ttbs_sz)) = discover_carveout(b"ttbs") else {
        crate::ktrace::log("agx", "uat: no `ttbs` (gpu-region) carveout — cannot map GPU buffers");
        return false;
    };
    let Some((pagetables, pt_sz)) = discover_carveout(b"pagetables") else {
        crate::ktrace::log("agx", "uat: no `pagetables` (gfx-shared) carveout");
        return false;
    };
    let Some((handoff, ho_sz)) = discover_carveout(b"handoff") else {
        crate::ktrace::log("agx", "uat: no `handoff` carveout — cannot PPL-init the firmware");
        return false;
    };
    crate::ktrace::log_fmt(format_args!(
        "agx: uat carveouts — ttbs={ttbs:#x}/{ttbs_sz:#x} pagetables={pagetables:#x}/{pt_sz:#x} handoff={handoff:#x}/{ho_sz:#x}"
    ));

    // The unblock (Milestone 3): the firmware's memory manager blocks its MMU
    // init until we complete the GFXHandoff PPL handshake (write MAGIC_AP, wait
    // for MAGIC_FW). Do it first — without it the coprocessor never powers on.
    if !handoff::initialize(handoff, 3000, &mut pump) {
        crate::ktrace::log("agx", "uat: GFXHandoff PPL init failed/timed out");
        return false;
    }

    // Read the firmware's ctx-0 TTBR pair (diagnostic — it was 0/0 as booted).
    let fw_ttbr0 = rd64(ttbs + uat::ttbr_slot_offset(0, 0));
    let fw_ttbr1 = rd64(ttbs + uat::ttbr_slot_offset(0, 1));
    crate::ktrace::log_fmt(format_args!("agx: uat ttbs ctx-0 (as booted) TTBR0={fw_ttbr0:#018x} TTBR1={fw_ttbr1:#018x}"));

    // ctx-0 kernel context: a fresh L1 for TTBR0 (our buffer mappings), TTBR1 =
    // the firmware's shared-region tables. Reuse the firmware's L1 if it already
    // installed one (it hasn't — 0/0). Program the pair UNDER the handoff lock
    // (proxyclient `UAT.init`: `with self.handoff.lock(): set_l0(0,0); set_l0(0,1)`).
    let l1 = if uat::is_valid(fw_ttbr0) {
        uat::ttbr_base(fw_ttbr0)
    } else {
        let Some(l1) = alloc_shared(1) else {
            crate::ktrace::log("agx", "uat: L1 table alloc failed");
            return false;
        };
        l1
    };
    let ttbr0 = uat::ttbr(l1, 0, true);
    let ttbr1 = uat::ttbr(pagetables, 0, true);
    let locked = handoff::with_lock(handoff, 2000, &mut pump, || {
        wr64(ttbs + uat::ttbr_slot_offset(0, 0), ttbr0);
        wr64(ttbs + uat::ttbr_slot_offset(0, 1), ttbr1);
        dcache_clean(ttbs, uat::TTBR_PAIR_BYTES);
        dcache_clean(l1, uat::PAGE_SIZE);
    });
    if !locked {
        crate::ktrace::log("agx", "uat: could not take handoff lock to set ctx-0 TTBRs");
        return false;
    }
    crate::ktrace::log_fmt(format_args!("agx: uat ctx-0 TTBR0={ttbr0:#018x} (L1 {l1:#x}) TTBR1={ttbr1:#018x} (fw {pagetables:#x})"));

    UAT.with(|u| {
        u.ttbs = ttbs;
        u.pagetables = pagetables;
        u.handoff = handoff;
        u.l1 = l1;
        u.va_next = BUF_VA_BASE;
        u.ready = true;
    });
    true
}

/// Fetch the next-level table a descriptor points at, allocating + linking a
/// fresh zeroed table if the slot is empty. `table` is the current level's
/// physical base; `idx` the entry index. Returns the next table's phys, or
/// `None` on OOM.
fn get_or_alloc_table(table: u64, idx: usize) -> Option<u64> {
    let slot = table + idx as u64 * 8;
    let e = rd64(slot);
    if uat::is_valid(e) {
        return Some(uat::pte_output(e));
    }
    let next = alloc_shared(1)?;
    // Clean the ENTIRE new table page to PoC, not just the parent slot — the
    // GPU MMU walker reads the table from DRAM and must see our zeros (Grok/
    // proxyclient: full-page dc cvac before installing leaves), else it walks
    // stale/garbage entries.
    dcache_clean(next, uat::PAGE_SIZE);
    wr64(slot, uat::table_pte(next));
    dcache_clean(slot, 8);
    Some(next)
}

/// Map one 16 KiB `pa` page at GPU VA `va`. `l1_ttbr0`/`l1_ttbr1` are the L1
/// table bases for each half; the VA's bit-39 select picks which — TTBR1 (the
/// firmware's shared `pagetables`) for RTKit kernel buffers, so they're reachable
/// from the firmware's boot context.
fn uat_map_page(l1_ttbr0: u64, l1_ttbr1: u64, va: u64, pa: u64, mmio: bool) -> bool {
    let (l0, i1, i2, i3, _) = uat::split_va(va);
    let l1 = if l0 == 1 { l1_ttbr1 } else { l1_ttbr0 };
    let Some(l2) = get_or_alloc_table(l1, i1) else { return false };
    let Some(l3) = get_or_alloc_table(l2, i2) else { return false };
    let slot = l3 + i3 as u64 * 8;
    let pte = if mmio { uat::mmio_page_pte(pa) } else { uat::kernel_buffer_pte(pa) };
    wr64(slot, pte);
    dcache_clean(slot, 8);
    true
}

/// Map a physically-contiguous RTKit buffer (`phys`, `bytes`) into the shared
/// TTBR1 kernel range and return its GPU VA (bump-allocated from `BUF_VA_BASE`).
/// `None` on failure.
fn uat_map_buffer(phys: u64, bytes: u64) -> Option<u64> {
    if !uat_init() {
        return None;
    }
    let npages = bytes.div_ceil(uat::PAGE_SIZE).max(1);
    let va = UAT.with(|u| {
        let va = u.va_next;
        u.va_next += npages * uat::PAGE_SIZE;
        va
    });
    uat_map_range(va, phys, npages * uat::PAGE_SIZE).then_some(va)
}

/// Map `[va, va+bytes)` → `[phys, phys+bytes)` page-by-page into ctx-0 (TTBR0/1
/// by the VA's bit-39 select). `bytes` is rounded up to the 16 KiB page. Used to
/// place a buffer at a *specific* GPU VA (the initdata replay). Returns false on
/// OOM/failure.
fn uat_map_range(va: u64, phys: u64, bytes: u64) -> bool {
    uat_map_range_kind(va, phys, bytes, false)
}

/// Like [`uat_map_range`] but `mmio` selects the Device leaf attribute (for the
/// AGX IOMappings — GPU-register MMIO mapped into the kernel VM).
fn uat_map_range_kind(va: u64, phys: u64, bytes: u64, mmio: bool) -> bool {
    let (l1_ttbr0, l1_ttbr1) = UAT.with(|u| (u.l1, u.pagetables));
    let npages = bytes.div_ceil(uat::PAGE_SIZE).max(1);
    for i in 0..npages {
        let off = i * uat::PAGE_SIZE;
        if !uat_map_page(l1_ttbr0, l1_ttbr1, (va + off) & !uat::PAGE_MASK, phys + off, mmio) {
            crate::ktrace::log("agx", "uat: page map failed (OOM)");
            return false;
        }
    }
    true
}

// ===================== Layer 1: per-agent GPU context =====================
//
// A GPU *context* is its own TTBR0 address space (a fresh L1) bound into the ttbs
// per-context slot, sharing the firmware's TTBR1 (`pagetables`). Command buffers
// and their data live in a context; a submission's `context_id` selects it.
// Reference: m1n1 `agx/context.py GPUContext` — a zeroed L1,
// `bind_context(ctx_id, ttbr0)`, object mappings via `iomap_at`. VA layout
// (proxyclient): pipelines 0x1100000000, GEM 0x1500000000, userspace 0x1600000000.

/// Standard GPU-VA region bases inside a context (proxyclient `GPUContext`).
#[allow(dead_code)] // pipeline/userspace regions used by Layer 2, landing next
const CTX_PIPELINE_BASE: u64 = 0x1100000000;
const CTX_GEM_BASE: u64 = 0x1500000000;
#[allow(dead_code)]
const CTX_USERSPACE_BASE: u64 = 0x1600000000;

/// A live GPU context: its numeric id (ASID) + the phys of its TTBR0 L1 table.
#[derive(Clone, Copy)]
struct GpuContext {
    ctx_id: u16,
    l1: u64,
}

/// Create + bind a fresh GPU context: allocate a zeroed TTBR0 L1 and program the
/// ttbs pair for `ctx_id` (TTBR0 = our L1, TTBR1 = the shared firmware
/// `pagetables`) under the GFXHandoff PPL lock — exactly as ctx-0 is set in
/// `uat_init`, and as the proxyclient's `bind_context` does. `None` on OOM / lock
/// failure. `ctx_id` 0 is the kernel context and is never re-bound here.
fn create_context(ctx_id: u16) -> Option<GpuContext> {
    if ctx_id == 0 || !uat_init() {
        return None;
    }
    let (ttbs, pagetables, handoff) = UAT.with(|u| (u.ttbs, u.pagetables, u.handoff));
    let l1 = alloc_shared(1)?;
    dcache_clean(l1, uat::PAGE_SIZE);
    let ttbr0 = uat::ttbr(l1, ctx_id as u64, true);
    let ttbr1 = uat::ttbr(pagetables, ctx_id as u64, true);
    let base = ttbs + uat::ttbr_slot_offset(ctx_id as usize, 0);
    let locked = handoff::with_lock(handoff, 2000, &mut pump, || {
        wr64(ttbs + uat::ttbr_slot_offset(ctx_id as usize, 0), ttbr0);
        wr64(ttbs + uat::ttbr_slot_offset(ctx_id as usize, 1), ttbr1);
        dcache_clean(base, uat::TTBR_PAIR_BYTES);
        dcache_clean(l1, uat::PAGE_SIZE);
    });
    if !locked {
        crate::ktrace::log("agx", "ctx: could not take handoff lock to bind context");
        return None;
    }
    crate::ktrace::log_fmt(format_args!(
        "agx: ctx {ctx_id} bound TTBR0={ttbr0:#018x} (L1 {l1:#x}) TTBR1={ttbr1:#018x}"
    ));
    Some(GpuContext { ctx_id, l1 })
}

/// Map one 16 KiB context object page into `ctx`'s TTBR0 (low-half VA; `pipeline`
/// picks the shader AP). Intermediate tables are allocated in the context's own L1
/// subtree. False if the VA is in the high half (bit 39 set) or on OOM.
fn ctx_map_page(ctx: &GpuContext, va: u64, pa: u64, pipeline: bool) -> bool {
    let (l0, i1, i2, i3, _) = uat::split_va(va);
    if l0 != 0 {
        return false; // context objects live in TTBR0
    }
    let Some(l2) = get_or_alloc_table(ctx.l1, i1) else { return false };
    let Some(l3) = get_or_alloc_table(l2, i2) else { return false };
    let slot = l3 + i3 as u64 * 8;
    wr64(slot, uat::context_page_pte(pa, pipeline));
    dcache_clean(slot, 8);
    true
}

/// Map `[va, va+bytes)` → `[phys, …)` into `ctx` page-by-page. `bytes` rounds up
/// to 16 KiB. False on OOM.
#[allow(dead_code)] // used by Layer 2 (command-buffer objects), landing next
fn ctx_map_range(ctx: &GpuContext, va: u64, phys: u64, bytes: u64, pipeline: bool) -> bool {
    let npages = bytes.div_ceil(uat::PAGE_SIZE).max(1);
    for i in 0..npages {
        let off = i * uat::PAGE_SIZE;
        if !ctx_map_page(ctx, (va + off) & !uat::PAGE_MASK, phys + off, pipeline) {
            return false;
        }
    }
    true
}

/// Allocate `bytes` of zeroed DRAM and map it into `ctx` at `va` (bump-agnostic —
/// caller picks the VA within a region). Returns the backing phys, or `None`.
#[allow(dead_code)] // used by Layer 2, landing next
fn ctx_alloc_at(ctx: &GpuContext, va: u64, bytes: u64, pipeline: bool) -> Option<u64> {
    let npages = bytes.div_ceil(uat::PAGE_SIZE).max(1);
    let phys = alloc_shared(npages)?;
    dcache_clean(phys, npages * uat::PAGE_SIZE);
    ctx_map_range(ctx, va, phys, npages * uat::PAGE_SIZE, pipeline).then_some(phys)
}

/// Smoke-test the context layer on hardware: create a fresh context, map one test
/// page, and read the ttbs bind + L1 chain back to confirm the descriptors landed
/// correctly (and that binding a new context doesn't fault the firmware). Does not
/// submit work — that is Layer 2.
fn context_selftest() {
    let Some(ctx) = create_context(64) else {
        crate::ktrace::log("agx", "ctx: selftest — create_context failed");
        return;
    };
    let (ttbs,) = UAT.with(|u| (u.ttbs,));
    let bound0 = rd64(ttbs + uat::ttbr_slot_offset(64, 0));
    let bound1 = rd64(ttbs + uat::ttbr_slot_offset(64, 1));
    crate::ktrace::log_fmt(format_args!(
        "agx: ctx selftest — ttbs[64] TTBR0={bound0:#018x} TTBR1={bound1:#018x} (L1 {:#x})",
        ctx.l1
    ));
    // Map a test page at the GEM base and verify the walk resolves to it.
    let Some(phys) = alloc_shared(1) else { return };
    if !ctx_map_page(&ctx, CTX_GEM_BASE, phys, false) {
        crate::ktrace::log("agx", "ctx: selftest — ctx_map_page failed");
        return;
    }
    let (_, i1, i2, i3, _) = uat::split_va(CTX_GEM_BASE);
    let l2 = uat::pte_output(rd64(ctx.l1 + i1 as u64 * 8));
    let l3 = uat::pte_output(rd64(l2 + i2 as u64 * 8));
    let leaf = rd64(l3 + i3 as u64 * 8);
    let ok = uat::is_valid(leaf) && uat::pte_output(leaf) == phys;
    crate::ktrace::log_fmt(format_args!(
        "agx: ctx selftest — GEM {CTX_GEM_BASE:#x} -> leaf {leaf:#018x} (phys {phys:#x}) walk_ok={ok}"
    ));
}

/// The AGX **IOMappings** — GPU-register MMIO the firmware needs mapped into its
/// kernel VM (initdata RegionC iomappings). Captured from the live-M2 struct
/// dump (t8112); fixed SoC physical addresses. `(gpu_va, phys, size)`.
const IOMAPPINGS: &[(u64, u64, u64)] = &[
    (0xffffffa068000000, 0x204d00000, 0x14000),
    (0xffffffa068018000, 0x20e100000, 0x4000),
    (0xffffffa068020000, 0x23b0c4000, 0x4000),
    (0xffffffa068028000, 0x204000000, 0x20000),
    (0xffffffa06804c000, 0x23b2c0000, 0x1000),
    (0xffffffa068054000, 0x204d80000, 0x8000),
    (0xffffffa068061000, 0x204d61000, 0x1000),
    (0xffffffa068068000, 0x200000000, 0xd6400),
    (0xffffffa068144000, 0x204e00000, 0x10000),
    (0xffffffa068158000, 0x27d050000, 0x4000),
    (0xffffffa068160000, 0x23b3d0000, 0x1000),
    (0xffffffa068168000, 0x23b3c0000, 0x1000),
];
/// The GPU timestamp region (initdata `timestamp_region_base`) — a DRAM buffer in
/// the kgpurw allocator that the memmap scan didn't reach. Map zeroed DRAM here.
const TIMESTAMP_REGION_VA: u64 = 0xffffffa071000000;
const TIMESTAMP_REGION_SIZE: u64 = 0x10000;

/// Replay the captured ctx-0 GPU memory map from the live M2: for every range,
/// allocate DRAM, map it at the *identical* GPU VA (so the embedded pointers in
/// the config bytes stay valid), and copy the captured bytes (or zero-fill).
/// Returns the initdata GPU VA on success. Because it's the same machine, this
/// reproduces exactly what the working driver hands the firmware.
fn replay_initdata() -> Option<u64> {
    use super::initdata_blob::{INITDATA_VA, RANGES, REPLAY_DATA};
    if !uat_init() {
        return None;
    }
    crate::ktrace::log_fmt(format_args!("agx: replaying {} initdata ranges ({} B config data)", RANGES.len(), REPLAY_DATA.len()));
    replay_map_reset();
    let mut mapped_bytes: u64 = 0;
    for &(va, size, data_off) in RANGES {
        let size = size as u64;
        // Page-aligned physical DRAM for this range.
        let Some(phys) = alloc_shared(size.div_ceil(4096)) else {
            crate::ktrace::log_fmt(format_args!("agx: replay OOM at va {va:#x} size {size:#x}"));
            return None;
        };
        replay_map_record(va, phys, size);
        if data_off != u32::MAX {
            // Copy the captured config bytes into the fresh DRAM (VA==PA).
            let src = &REPLAY_DATA[data_off as usize..data_off as usize + size as usize];
            // SAFETY: `phys` is our own alloc_shared'd, size-byte, identity-mapped buffer.
            unsafe { core::ptr::copy_nonoverlapping(src.as_ptr(), phys as *mut u8, size as usize) };
        }
        // Clean to PoC so the GPU sees it, then map at the exact VA.
        dcache_clean(phys, size);
        if !uat_map_range(va, phys, size) {
            return None;
        }
        mapped_bytes += size;
    }

    // Map the GPU-register IOMappings (MMIO → GPU VA, Device attr) at their exact
    // VAs. These aren't DRAM — the physical is the SoC register block itself.
    for &(gpu_va, phys, size) in IOMAPPINGS {
        crate::arch::aarch64::mmu::map_device_gib(phys);
        if !uat_map_range_kind(gpu_va, phys, size, true) {
            crate::ktrace::log_fmt(format_args!("agx: IOMapping {gpu_va:#x}->{phys:#x} failed"));
            return None;
        }
    }
    // The timestamp region (DRAM, zeroed) that the memmap scan didn't reach.
    if let Some(ts) = alloc_shared(TIMESTAMP_REGION_SIZE / 4096) {
        dcache_clean(ts, TIMESTAMP_REGION_SIZE);
        let _ = uat_map_range(TIMESTAMP_REGION_VA, ts, TIMESTAMP_REGION_SIZE);
    }

    crate::ktrace::log_fmt(format_args!(
        "agx: initdata replay done — {mapped_bytes:#x} DRAM + {} IOMappings, initdata VA {INITDATA_VA:#x}",
        IOMAPPINGS.len()
    ));
    Some(INITDATA_VA)
}

/// Read + log the SGX GPU-MMU fault-status register, *recoverably* (the block
/// external-aborts until the GPU powers on, so we arm the sync-handler probe and
/// treat a fault as "SGX unpowered" instead of crashing). `FAULTED`/`REASON`/
/// `CONTEXT`/`ADDR<<6` say whether the firmware's MMU faulted (e.g. on a buffer
/// we mapped) and where. Offsets from m1n1 `hw/agx.py` (`SGXRegs.FAULT_INFO`).
fn sgx_fault_check() {
    let fdt = crate::arch::aarch64::boot::boot_x0();
    // SAFETY: boot_x0 is the FDT pointer (or non-FDT, rejected by the magic).
    let Some((sgx, _)) = (unsafe { crate::fdt::reg_nth_of_compatible(fdt, AGX_COMPAT, 1) }) else {
        return;
    };
    crate::arch::aarch64::mmu::map_device_gib(sgx);
    crate::arch::aarch64::set_agx_probing(true);
    // SAFETY: recoverable read — if the SGX block is unpowered the sync handler
    // skips this `ldr` and flags the fault (agx_probe_faulted).
    let fi = unsafe { core::ptr::read_volatile((sgx + 0x17030) as *const u64) };
    crate::arch::aarch64::set_agx_probing(false);
    if crate::arch::aarch64::agx_probe_faulted() {
        crate::ktrace::log("agx", "SGX FAULT_INFO unreadable (external abort) → GPU core still unpowered");
        return;
    }
    let reasons = ["INVALID", "AF_FAULT", "WRITE_ONLY", "READ_ONLY", "NO_ACCESS", "UNK", "UNK", "UNK"];
    crate::ktrace::log_fmt(format_args!(
        "agx: SGX FAULT_INFO={fi:#018x} faulted={} reason={} rw={} ctx={} addr={:#x}",
        fi & 1,
        reasons[((fi >> 1) & 0x7) as usize],
        if (fi >> 4) & 1 == 1 { "R" } else { "W" },
        (fi >> 17) & 0x3f,
        (fi >> 30) << 6
    ));
}

/// SGX liveness poke (proxyclient `AGX.poke_sgx`): read-modify-write `0x70001`
/// into sgx reg[1] + 0xd14000. Recoverable — if the SGX block is still gated the
/// access external-aborts and we skip it (the sync-handler probe swallows the
/// fault) rather than crashing.
fn sgx_poke() {
    let fdt = crate::arch::aarch64::boot::boot_x0();
    // SAFETY: boot_x0 is the FDT pointer (or non-FDT, rejected by the magic).
    let Some((sgx, _)) = (unsafe { crate::fdt::reg_nth_of_compatible(fdt, AGX_COMPAT, 1) }) else {
        return;
    };
    crate::arch::aarch64::mmu::map_device_gib(sgx);
    let addr = sgx + 0xd14000;
    crate::arch::aarch64::set_agx_probing(true);
    // SAFETY: recoverable MMIO read; a fault is swallowed by the sync handler.
    let v = unsafe { core::ptr::read_volatile(addr as *const u32) };
    if !crate::arch::aarch64::agx_probe_faulted() {
        // SAFETY: recoverable MMIO write of the SGX liveness bit.
        unsafe { core::ptr::write_volatile(addr as *mut u32, 0x70001 | v) };
    }
    crate::arch::aarch64::set_agx_probing(false);
    if crate::arch::aarch64::agx_probe_faulted() {
        crate::ktrace::log("agx", "SGX poke skipped — sgx block still gated (external abort)");
    } else {
        crate::ktrace::log_fmt(format_args!("agx: SGX poked @ {addr:#x} (was {v:#x})"));
    }
}

/// Dump the head of the crashlog shared buffer (identity-mapped DRAM we own). A
/// firmware crash writes a `CLHE` (`'C''L''H''E'`) record here; a still-zero
/// buffer means the firmware never touched it (stuck before any crash path).
fn dump_crashlog(phys: u64) {
    // SAFETY: `phys` is our own alloc_shared'd DMA buffer, identity-mapped.
    let w0 = unsafe { core::ptr::read_volatile(phys as *const u32) };
    let clhe = u32::from_le_bytes(*b"CLHE");
    if w0 == 0 {
        crate::ktrace::log_fmt(format_args!("agx: crashlog buffer @ {phys:#x} is all-zero (FW wrote nothing — stuck, not crashed)"));
        return;
    }
    let tag = if w0 == clhe { " (CLHE crash record!)" } else { "" };
    let mut line = alloc::string::String::new();
    use core::fmt::Write;
    for i in 0..16u64 {
        // SAFETY: 16 words within our 16 KiB crashlog buffer.
        let w = unsafe { core::ptr::read_volatile((phys + i * 4) as *const u32) };
        let _ = write!(line, " {w:08x}");
    }
    crate::ktrace::log_fmt(format_args!("agx: crashlog @ {phys:#x}{tag}:{line}"));

    // Scan the first ~2 KiB for printable ASCII runs — the crashlog `Cstr`
    // records carry the firmware's own crash message (often naming exactly what
    // it faulted on), far more useful than the raw hex.
    let mut s = alloc::string::String::new();
    let mut run = alloc::string::String::new();
    for i in 0..2048u64 {
        // SAFETY: within our 16 KiB crashlog buffer.
        let b = unsafe { core::ptr::read_volatile((phys + i) as *const u8) };
        if (0x20..0x7f).contains(&b) {
            run.push(b as char);
        } else {
            if run.len() >= 4 {
                s.push_str(&run);
                s.push('|');
            }
            run.clear();
        }
    }
    if !s.is_empty() {
        crate::ktrace::log_fmt(format_args!("agx: crashlog strings: {s}"));
    }
}

/// The cooperative pump run between every mailbox poll: keep the UI/clock/net
/// alive and report Ctrl+C (returns true to abort). Mirrors the inference/IO
/// loops' `upkeep()` + interrupt-poll discipline (CLAUDE.md standing rule).
fn pump() -> bool {
    crate::shell::upkeep();
    if crate::shell::poll_interrupt() {
        ABORT.store(true, core::sync::atomic::Ordering::Relaxed);
    }
    ABORT.load(core::sync::atomic::Ordering::Relaxed)
}

/// Send `(msg0, ep)`, logging + returning false on failure.
fn send(asc: &Asc, msg0: u64, ep: u8) -> bool {
    let ok = asc.send(&Message { msg0, msg1: ep as u32 }, SEND_TIMEOUT_MS, &mut pump);
    if !ok {
        crate::ktrace::log_fmt(format_args!("agx: send failed (ep {ep:#x}, msg {msg0:#018x}) — mailbox stuck?"));
    }
    ok
}

/// Drive the RTKit handshake to RUNNING (port of `rtkit_boot`). Fills `rep` and
/// returns the outcome.
fn rtkit_boot(asc: &Asc, rep: &mut Report) -> Outcome {
    use proto::*;

    // Boot the IOP and nudge it awake (INIT). Unconditional, per m1n1.
    asc.cpu_start();
    if !send(asc, msg_iop_pwr_state(POWER_INIT), EP_MGMT) {
        return Outcome::SendFail;
    }

    // Initialise the UAT (GFXHandoff PPL handshake + ctx-0 tables) BEFORE the
    // RTKit power handshake — drm/asahi does the whole UAT setup in `new()`
    // before `rtk.boot()`, so the firmware's memory manager is up before the
    // power handshake rather than being set up mid-way through it (which we did
    // before, and the firmware then never reached power-ON). Non-fatal: if the
    // firmware's PPL isn't ready this early, `uat_map_buffer` retries it when the
    // first buffer is requested.
    if uat_init() {
        crate::ktrace::log("agx", "uat/PPL initialised before RTKit handshake (drm/asahi order)");
    } else {
        crate::ktrace::log("agx", "early uat/PPL init not ready — will retry at first buffer request");
    }

    // --- Phase A decisive test: HELLO within ~1 s ⇒ firmware resident. --------
    let Some(hello) = asc.recv_blocking(HELLO_TIMEOUT_MS, &mut pump) else {
        crate::ktrace::log("agx", "no HELLO within 1s after cpu_start — gfx-asc firmware not resident (blocked on Asahi GPU-firmware provisioning)");
        return Outcome::NoHello;
    };
    crate::ktrace::log_fmt(format_args!("agx: got mailbox msg0={:#018x} ep={:#x}", hello.msg0, hello.msg1));
    if hello.msg1 != EP_MGMT as u32 || mgmt_type(hello.msg0) != MGMT_MSG_HELLO {
        crate::ktrace::log("agx", "first message was not HELLO on EP_MGMT");
        return Outcome::BadHello;
    }
    let (min_ver, max_ver) = hello_versions(hello.msg0);
    let Some(ver) = negotiate_version(min_ver, max_ver) else {
        crate::ktrace::log_fmt(format_args!("agx: version mismatch (IOP [{min_ver},{max_ver}] vs [{RTKIT_MIN_VERSION},{RTKIT_MAX_VERSION}])"));
        return Outcome::VersionMismatch;
    };
    rep.version = ver;
    crate::ktrace::log_fmt(format_args!("agx: HELLO ok — booting RTKit version {ver}"));
    if !send(asc, msg_hello_ack(ver), EP_MGMT) {
        return Outcome::SendFail;
    }

    // --- endpoint map loop ---------------------------------------------------
    let mut eps = EndpointSet::default();
    loop {
        let Some(m) = asc.recv_blocking(STEP_TIMEOUT_MS, &mut pump) else {
            crate::ktrace::log("agx", "timeout waiting for endpoint map");
            return Outcome::EpmapFail;
        };
        if m.msg1 != EP_MGMT as u32 || mgmt_type(m.msg0) != MGMT_MSG_EPMAP {
            crate::ktrace::log_fmt(format_args!("agx: expected EPMAP, got msg0={:#018x} ep={:#x}", m.msg0, m.msg1));
            return Outcome::EpmapFail;
        }
        let em = epmap(m.msg0);
        crate::ktrace::log_fmt(format_args!("agx: EPMAP raw msg0={:#018x} bitmap={:#010x} base={} done={}", m.msg0, em.bitmap, em.base, em.done));
        eps.record_bitmap(em.bitmap, em.base);
        if !send(asc, msg_epmap_reply(em.base, em.done), EP_MGMT) {
            return Outcome::SendFail;
        }
        if em.done {
            break;
        }
    }
    rep.eps = eps;
    crate::ktrace::log_fmt(format_args!(
        "agx: endpoints — crashlog={} debug={} syslog={} ioreport={} oslog={}",
        eps.crashlog, eps.debug, eps.syslog, eps.ioreport, eps.oslog
    ));

    // --- start every present system endpoint ---------------------------------
    for ep in eps.to_start() {
        if !send(asc, msg_start_ep(ep), EP_MGMT) {
            return Outcome::SendFail;
        }
    }

    // --- request AP power ON *immediately* (the AGX `boot_done`) --------------
    // AGX-specific ordering (proxyclient `mgmt.py`: on the last EPMAP fragment it
    // starts the endpoints and *then* `boot_done()` = SetAPPower(ON), and only
    // after that `wait_boot` for BOTH iop and ap power == ON). The AGX firmware
    // waits for this AP-power-ON before it transitions the IOP to ON — the
    // generic `rtkit.c` order (wait for iop ON, *then* send AP ON) deadlocks it,
    // which is exactly the silent-after-buffer-grant stall we saw.
    if !send(asc, msg_ap_pwr_state(POWER_ON), EP_MGMT) {
        return Outcome::SendFail;
    }
    crate::ktrace::log("agx", "sent AP_PWR_STATE=ON (boot_done); waiting for iop+ap power ON");

    // --- pump until BOTH iop and ap power reach ON, servicing buffer requests -
    // Total-budget wait: poll in short windows against one overall deadline so a
    // slow-but-progressing IOP is not cut off. Every received message is logged.
    let mut state = RtkitState::default();
    let powerup_deadline = crate::arch::now_ms() + POWERUP_BUDGET_MS;
    let mut last_beat = crate::arch::now_ms();
    while state.iop_power != POWER_ON || state.ap_power != POWER_ON {
        if crate::arch::now_ms() >= powerup_deadline {
            crate::ktrace::log_fmt(format_args!(
                "agx: timed out after {POWERUP_BUDGET_MS}ms (iop_power={:#x} ap_power={:#x}, buffers granted={})",
                state.iop_power, state.ap_power, rep.n_buffers
            ));
            // Diagnostics: the handoff state (did the FW touch it post-PPL?) and
            // the SGX GPU-MMU fault register (recoverable — the block
            // external-aborts while the GPU is unpowered).
            if let Some((ho, _)) = discover_carveout(b"handoff") {
                handoff::dump(ho);
            }
            sgx_fault_check();
            // Dump the crashlog buffer — if the firmware hit an internal error it
            // may have written a 'CLHE' record here (plain DRAM we own).
            if rep.crashlog_phys != 0 {
                dump_crashlog(rep.crashlog_phys);
            }
            return Outcome::Timeout;
        }
        let Some(m) = asc.recv_blocking(POWERUP_POLL_MS, &mut pump) else {
            if ABORT.load(core::sync::atomic::Ordering::Relaxed) {
                crate::ktrace::log("agx", "power-up wait cancelled (Ctrl+C)");
                return Outcome::Timeout;
            }
            // No message this window. Heartbeat ~1/s so the human sees progress,
            // then keep waiting until the overall deadline.
            let now = crate::arch::now_ms();
            if now.saturating_sub(last_beat) >= 1000 {
                last_beat = now;
                crate::ktrace::log_fmt(format_args!("agx: waiting for power ON… ({}ms left, iop={:#x} ap={:#x})", powerup_deadline.saturating_sub(now), state.iop_power, state.ap_power));
            }
            continue;
        };
        crate::ktrace::log_fmt(format_args!("agx: pump msg0={:#018x} ep={:#x}", m.msg0, m.msg1));
        let ep = m.msg1 as u8;
        if ep >= EP_APP_BASE {
            crate::ktrace::log_fmt(format_args!("agx: unexpected app-endpoint msg during boot (ep {ep:#x})"));
            continue;
        }
        match handle_system_msg(&mut state, m.msg0, ep) {
            Action::None => {}
            Action::Send(msg0, e) => {
                if !send(asc, msg0, e) {
                    return Outcome::SendFail;
                }
            }
            Action::AllocBuffer { ep, kind, n_pages, addr } => {
                crate::ktrace::log_fmt(format_args!("agx: buffer request ep={ep:#x} kind={kind:?} pages={n_pages} addr={addr:#x}"));
                if addr != 0 {
                    // IOP pre-allocated (SRAM/handoff) — record, no reply.
                    rep.n_buffers += 1;
                    if kind == BufferKind::Crashlog {
                        state.have_crashlog_buffer = true;
                    }
                } else {
                    // 16 KiB-align the buffer (the UAT page granule) and reply
                    // the aligned page count, exactly as the proxyclient crashlog
                    // handler does (`align(0x1000*SIZE, 0x4000)`, reply
                    // `SIZE=aligned/0x1000`) — a 2-page (8 KiB) request becomes a
                    // 4-page (16 KiB) grant.
                    let aligned_bytes = (n_pages * 4096).next_multiple_of(uat::PAGE_SIZE);
                    let aligned_pages = aligned_bytes / 4096;
                    let Some(phys) = alloc_shared(aligned_pages) else {
                        crate::ktrace::log("agx", "shared-buffer allocation failed (OOM)");
                        return Outcome::Timeout;
                    };
                    // Map into the GPU's ctx-0 page tables and reply with the GPU
                    // VA (not raw phys) — the coprocessor reads DRAM only through
                    // the UAT.
                    let Some(dva) = uat_map_buffer(phys, aligned_bytes) else {
                        crate::ktrace::log("agx", "uat map failed");
                        return Outcome::Timeout;
                    };
                    let reply = msg_buffer_reply(aligned_pages, dva);
                    crate::ktrace::log_fmt(format_args!("agx: buffer reply ep={ep:#x} pages={aligned_pages} dva={dva:#x} msg0={reply:#018x}"));
                    if !send(asc, reply, ep) {
                        return Outcome::SendFail;
                    }
                    rep.n_buffers += 1;
                    if kind == BufferKind::Crashlog {
                        state.have_crashlog_buffer = true;
                        rep.crashlog_phys = phys;
                    }
                    // NB: the proxyclient sends SetAPPower(ON) exactly ONCE (at
                    // boot_done, before this). No re-send — a second AP-power
                    // transition can confuse the firmware's power FSM.
                }
            }
            Action::Crashed => {
                crate::ktrace::log("agx", "IOP crashed during boot (second crashlog buffer request)");
                return Outcome::Crashed;
            }
            Action::Unhandled => {
                crate::ktrace::log_fmt(format_args!("agx: unhandled system msg ep={ep:#x} msg0={:#018x}", m.msg0));
            }
        }
    }
    // Both iop and ap power reached ON — the coprocessor is RUNNING.
    rep.iop_power = state.iop_power;
    rep.ap_power = state.ap_power;
    crate::ktrace::log("agx", "power ON reached (iop+ap) — coprocessor RUNNING");
    Outcome::Running
}

/// Compute-milestone step 1: start the app endpoints (0x20 firmware/KMD, 0x21
/// doorbell) on a RUNNING coprocessor and observe for a few seconds. Logs every
/// message; services any further system BUFFER_REQUESTs through the UAT (as
/// during boot). App-endpoint traffic (ep >= 0x20) is just logged — handling it
/// needs the initdata + channel rings (the next sub-step).
fn probe_app_endpoints(asc: &Asc, rep: &mut Report) {
    use proto::*;
    crate::ktrace::log("agx", "compute: starting app endpoints 0x20 (firmware) + 0x21 (doorbell)");
    for ep in [EP_APP_BASE, EP_APP_BASE + 1] {
        if !send(asc, msg_start_ep(ep), EP_MGMT) {
            crate::ktrace::log_fmt(format_args!("agx: compute: START_EP {ep:#x} failed"));
            return;
        }
    }
    let deadline = crate::arch::now_ms() + 3000;
    // The crashlog buffer was already granted during boot, so a further crashlog
    // message is a CRASH notification (not a getbuf).
    let mut state = RtkitState { iop_power: POWER_ON, ap_power: POWER_ON, crashed: false, have_crashlog_buffer: true };
    let mut any = false;
    while crate::arch::now_ms() < deadline {
        let Some(m) = asc.recv_blocking(500, &mut pump) else {
            if ABORT.load(core::sync::atomic::Ordering::Relaxed) {
                break;
            }
            continue;
        };
        any = true;
        let ep = m.msg1 as u8;
        crate::ktrace::log_fmt(format_args!("agx: compute: msg0={:#018x} ep={ep:#x}", m.msg0));
        if ep >= EP_APP_BASE {
            crate::ktrace::log("agx", "compute: app-endpoint message (needs initdata/channels — next step)");
            continue;
        }
        // Further system messages (e.g. channel-ring buffer requests) — service
        // buffer requests through the shared TTBR1 UAT, ACK the rest.
        match handle_system_msg(&mut state, m.msg0, ep) {
            Action::Crashed => {
                // Expected without initdata — but it PROVES the buffer is
                // reachable (the FW wrote crash data there). Dump the record.
                crate::ktrace::log("agx", "compute: firmware CRASHED (expected without initdata) — dumping crash record:");
                if rep.crashlog_phys != 0 {
                    dump_crashlog(rep.crashlog_phys);
                }
                break;
            }
            Action::AllocBuffer { ep, n_pages, addr, .. } if addr == 0 => {
                let aligned = (n_pages * 4096).next_multiple_of(uat::PAGE_SIZE);
                if let Some(phys) = alloc_shared(aligned / 4096) {
                    if let Some(dva) = uat_map_buffer(phys, aligned) {
                        let _ = send(asc, msg_buffer_reply(aligned / 4096, dva), ep);
                        rep.n_buffers += 1;
                        crate::ktrace::log_fmt(format_args!("agx: compute: served buffer ep={ep:#x} -> dva={dva:#x}"));
                    }
                }
            }
            Action::Send(msg0, e) => {
                let _ = send(asc, msg0, e);
            }
            _ => {}
        }
    }
    if !any {
        crate::ktrace::log("agx", "compute: firmware idle after app-endpoint start (expected — it awaits MSG_INIT/initdata)");
    }
}

// Firmware→AP channel ring/state GPU VAs, from the live-M2 initdata dump
// (`InitData_RegionB.channels`, tools/agx-extract). The firmware writes these
// shared-memory rings autonomously once it accepts initdata — reading them back
// is the real proof-of-life (the ASC mailbox stays quiet post-MSG_INIT).
struct ChannelVa {
    name: &'static str,
    state: u64,
    ring: u64,
    ring_size: u64,
}
const CHANNELS: &[ChannelVa] = &[
    ChannelVa { name: "FWLog", state: 0xffffffa0403ffee0, ring: 0xffffffa040630000, ring_size: 0x150000 },
    ChannelVa { name: "KTrace", state: 0xffffffa0404d7fd0, ring: 0xffffffa040519000, ring_size: 0x8000 },
    ChannelVa { name: "Stats", state: 0xffffffa040563fd0, ring: 0xffffffa0405a6000, ring_size: 0x8000 },
    ChannelVa { name: "Event", state: 0xffffffa040377fd0, ring: 0xffffffa0403b8800, ring_size: 0x4000 },
    ChannelVa { name: "DevCtrl", state: 0xffffffa040333fd0, ring: 0xffffffa000330000, ring_size: 0x4000 },
];
const NCHAN: usize = 5;

/// Read the `(read_ptr@0x0, write_ptr@0x20)` cursor pair of every channel's state
/// header. Snapshotted *before* MSG_INIT (the replayed baseline) and again after,
/// so movement is unambiguous — a replayed-but-stale non-zero value (e.g. DevCtrl
/// left at 0x2/0x2 by the proxy capture) is NOT mistaken for a fresh write.
fn read_channel_cursors() -> [(u32, u32); NCHAN] {
    let mut out = [(0u32, 0u32); NCHAN];
    for (i, ch) in CHANNELS.iter().enumerate() {
        let mut st = [0u8; 64];
        let _ = gpu_read(ch.state, &mut st);
        let c0 = u32::from_le_bytes([st[0], st[1], st[2], st[3]]);
        let c1 = u32::from_le_bytes([st[0x20], st[0x21], st[0x22], st[0x23]]);
        out[i] = (c0, c1);
    }
    out
}

/// Compare post-MSG_INIT channel state against the replayed `baseline` and report
/// whether the firmware wrote anything — the decisive liveness signal. A cursor
/// that **moved** (or fresh printable log text) is proof our firmware is
/// executing; a merely-nonzero-but-unchanged cursor is stale replay data and does
/// NOT count. Purely read-only.
fn probe_firmware_liveness(baseline: &[(u32, u32); NCHAN]) -> bool {
    let now = read_channel_cursors();
    let mut alive = false;
    for (i, ch) in CHANNELS.iter().enumerate() {
        let (b0, b1) = baseline[i];
        let (c0, c1) = now[i];
        let moved = c0 != b0 || c1 != b1;
        crate::ktrace::log_fmt(format_args!(
            "agx: chan {:>7} cur0 {b0:#x}->{c0:#x} cur1 {b1:#x}->{c1:#x} {}",
            ch.name,
            if moved { "MOVED" } else { "(unchanged)" }
        ));
        if moved {
            alive = true;
        }
        // Scan the head of the ring for printable ASCII (FWLog/KTrace carry text).
        let scan = (ch.ring_size as usize).min(4096);
        let mut ring = alloc::vec![0u8; scan];
        let rn = gpu_read(ch.ring, &mut ring);
        if scan_ascii_report(ch.name, &ring[..rn]) {
            alive = true;
        }
    }
    alive
}

/// Log the longest printable-ASCII run (≥6 chars) found in `data`. Returns true
/// if any such run exists (the firmware wrote human-readable log text).
fn scan_ascii_report(name: &str, data: &[u8]) -> bool {
    let mut best_start = 0usize;
    let mut best_len = 0usize;
    let mut run_start = 0usize;
    let mut run = 0usize;
    for (i, &b) in data.iter().enumerate() {
        if (0x20..0x7f).contains(&b) {
            if run == 0 {
                run_start = i;
            }
            run += 1;
            if run > best_len {
                best_len = run;
                best_start = run_start;
            }
        } else {
            run = 0;
        }
    }
    if best_len >= 6 {
        let mut s = alloc::string::String::new();
        for &b in &data[best_start..best_start + best_len.min(80)] {
            s.push(b as char);
        }
        crate::ktrace::log_fmt(format_args!("agx: {name} ring text @+{best_start:#x}: {s:?}"));
        true
    } else {
        false
    }
}

// DevCtrl channel geometry, from the proxyclient (fw/agx/channels.py, G14/V13.5):
// item size 0x40, ring at the initdata's DevCtrl ringbuffer_addr, cursor pair in
// the state header (READ_PTR@0x0 / WRITE_PTR@0x20). The device-control channel is
// AP→FW; the AP writes a message at slot WRITE_PTR, bumps it, and rings doorbell
// 0x11. The firmware drains READ_PTR→WRITE_PTR and processes each message.
const DEVCTRL_STATE_VA: u64 = 0xffffffa040333fd0;
const DEVCTRL_RING_VA: u64 = 0xffffffa000330000;
const DEVCTRL_ITEM: u64 = 0x40;
const DEVCTRL_RING_ITEMS: u32 = 0x100;
const STATE_WRITE_PTR: u64 = 0x20;
/// Device-control message types (V >= V13_3): DC_Init begins channel operation;
/// DC_UpdateIdleTS follows it in the proxyclient boot sequence.
const DC_INIT: u32 = 0x1a;
const DC_UPDATE_IDLE_TS: u32 = 0x23;

/// Append one device-control message (`msg_type` + zero body, 0x40 bytes) to the
/// DevCtrl ring at the current WRITE_PTR, advance WRITE_PTR, and ring the DevCtrl
/// doorbell — the exact `GPUTXChannel.send_message` sequence. Returns false only
/// if the ring VA isn't mapped or the doorbell send fails.
fn devctrl_send(asc: &Asc, msg_type: u32) -> bool {
    let wptr = gpu_read_u32(DEVCTRL_STATE_VA + STATE_WRITE_PTR).unwrap_or(0);
    let mut msg = [0u8; DEVCTRL_ITEM as usize];
    msg[..4].copy_from_slice(&msg_type.to_le_bytes());
    let slot = DEVCTRL_RING_VA + DEVCTRL_ITEM * (wptr as u64);
    if gpu_write(slot, &msg) != msg.len() {
        crate::ktrace::log_fmt(format_args!("agx: devctrl_send: ring slot {slot:#x} unmapped"));
        return false;
    }
    let next = (wptr + 1) % DEVCTRL_RING_ITEMS;
    gpu_write_u32(DEVCTRL_STATE_VA + STATE_WRITE_PTR, next);
    crate::ktrace::log_fmt(format_args!(
        "agx: devctrl_send type={msg_type:#x} slot={wptr} -> WRITE_PTR={next}, ringing doorbell 0x11"
    ));
    send(asc, proto::msg_doorbell(proto::DOORBELL_DEVCTRL), proto::EP_DOORBELL)
}

/// Kick the firmware into full operation after MSG_INIT: send DC_Init then
/// DC_UpdateIdleTS on the DevCtrl channel (the proxyclient's post-initdata boot
/// steps). This is what makes the firmware begin driving FWLog/Event/etc. — the
/// channels the liveness probe showed still idle. Read-back reports the result.
fn kick_devctrl(asc: &Asc, baseline: &[(u32, u32); NCHAN]) {
    use proto::*;
    crate::ktrace::log("agx", "compute: kicking DevCtrl (DC_Init + DC_UpdateIdleTS)…");
    if !devctrl_send(asc, DC_INIT) {
        crate::ktrace::log("agx", "compute: DC_Init doorbell send failed");
        return;
    }
    // Give the firmware a moment to consume DC_Init and start its channels.
    let mut pump_fn = pump;
    let _ = asc.recv_blocking(500, &mut pump_fn);
    if !devctrl_send(asc, DC_UPDATE_IDLE_TS) {
        crate::ktrace::log("agx", "compute: DC_UpdateIdleTS doorbell send failed");
    }
    // Observe the mailbox briefly (the firmware fires an EP_FIRMWARE 0x42 event
    // when it posts to a channel) then re-probe the rings.
    let deadline = crate::arch::now_ms() + 3000;
    let mut state = RtkitState { iop_power: POWER_ON, ap_power: POWER_ON, crashed: false, have_crashlog_buffer: true };
    while crate::arch::now_ms() < deadline {
        let Some(m) = asc.recv_blocking(500, &mut pump_fn) else {
            if ABORT.load(core::sync::atomic::Ordering::Relaxed) {
                break;
            }
            continue;
        };
        let ep = m.msg1 as u8;
        crate::ktrace::log_fmt(format_args!("agx: post-DC msg0={:#018x} ep={ep:#x}", m.msg0));
        if ep < EP_APP_BASE {
            let _ = handle_system_msg(&mut state, m.msg0, ep);
        }
    }
    // Did the kick move any firmware→AP channel past its replayed baseline?
    if probe_firmware_liveness(baseline) {
        crate::ktrace::log("agx", "compute: DevCtrl kick WORKED — firmware channels advanced (FWLog/Event/Stats live)");
    } else {
        crate::ktrace::log("agx", "compute: no further channel movement after DevCtrl kick — inspect DC_Init handling");
    }
}

/// Compute milestone: replay the captured initdata memory map, start the app
/// endpoints, hand the firmware the initdata pointer (MSG_INIT), and observe.
/// If the firmware accepts it (no crash + it starts driving its channels), the
/// GPU is initialised and ready for command submission (the next step).
fn boot_firmware(asc: &Asc, rep: &mut Report) {
    use proto::*;
    let Some(initdata_va) = replay_initdata() else {
        crate::ktrace::log("agx", "compute: initdata replay failed — aborting");
        return;
    };
    // Snapshot channel cursors as replayed (pre-MSG_INIT) so the liveness probe
    // can detect real firmware writes as movement, not stale replay data.
    let baseline = read_channel_cursors();
    // Start the app endpoints then immediately MSG_INIT (drm/asahi order:
    // start_ep(0x20/0x21) → send_message(MSG_INIT | initdata)).
    for ep in [EP_FIRMWARE, EP_DOORBELL] {
        if !send(asc, msg_start_ep(ep), EP_MGMT) {
            crate::ktrace::log_fmt(format_args!("agx: compute: START_EP {ep:#x} failed"));
            return;
        }
    }
    let init = msg_fw_init(initdata_va);
    crate::ktrace::log_fmt(format_args!("agx: compute: MSG_INIT initdata={initdata_va:#x} msg0={init:#018x}"));
    if !send(asc, init, EP_FIRMWARE) {
        return;
    }
    // NB: do NOT ring the DevCtrl doorbell yet — that requires a proper DC_Init
    // command written into the devctrl ring (a replayed ring is stale/empty), and
    // kicking it processes garbage. Just MSG_INIT and observe.
    let _ = msg_doorbell; // (retained; used once the command ring is built)

    // Observe: accepted → the firmware drives its channels (fw-endpoint events /
    // channel-ring buffer requests); rejected → a crashlog message.
    crate::ktrace::log("agx", "compute: MSG_INIT sent — observing firmware response");
    let deadline = crate::arch::now_ms() + 5000;
    let mut state = RtkitState { iop_power: POWER_ON, ap_power: POWER_ON, crashed: false, have_crashlog_buffer: true };
    let mut msgs = 0u32;
    while crate::arch::now_ms() < deadline {
        let Some(m) = asc.recv_blocking(500, &mut pump) else {
            if ABORT.load(core::sync::atomic::Ordering::Relaxed) {
                break;
            }
            continue;
        };
        msgs += 1;
        let ep = m.msg1 as u8;
        crate::ktrace::log_fmt(format_args!("agx: compute: msg0={:#018x} ep={ep:#x}", m.msg0));
        if ep >= EP_APP_BASE {
            // Firmware endpoint event — the firmware is alive and driving its
            // channels. This is the success signal.
            continue;
        }
        match handle_system_msg(&mut state, m.msg0, ep) {
            Action::Crashed => {
                crate::ktrace::log("agx", "compute: firmware CRASHED after MSG_INIT — dumping crash record:");
                if rep.crashlog_phys != 0 {
                    dump_crashlog(rep.crashlog_phys);
                }
                return;
            }
            Action::AllocBuffer { ep, n_pages, addr, .. } if addr == 0 => {
                let aligned = (n_pages * 4096).next_multiple_of(uat::PAGE_SIZE);
                if let Some(phys) = alloc_shared(aligned / 4096) {
                    if let Some(dva) = uat_map_buffer(phys, aligned) {
                        let _ = send(asc, msg_buffer_reply(aligned / 4096, dva), ep);
                        crate::ktrace::log_fmt(format_args!("agx: compute: served buffer ep={ep:#x} -> {dva:#x}"));
                    }
                }
            }
            Action::Send(msg0, e) => {
                let _ = send(asc, msg0, e);
            }
            _ => {}
        }
    }
    if msgs == 0 {
        crate::ktrace::log("agx", "compute: no mailbox msgs post-MSG_INIT (expected — FW drives shared-memory rings). Probing channels…");
    } else {
        crate::ktrace::log_fmt(format_args!("agx: compute: firmware sent {msgs} mailbox message(s) after MSG_INIT"));
    }
    // The decisive check: did the firmware write its shared-memory channel rings?
    // (A cursor that MOVED past its replayed baseline / fresh boot-log text =
    // initdata accepted + firmware executing. A stale-but-nonzero cursor is not.)
    let alive = probe_firmware_liveness(&baseline);
    if alive {
        crate::ktrace::log("agx", "compute: FIRMWARE ALIVE — channel cursor advanced past replay baseline (initdata accepted, GPU executing)");
    } else {
        crate::ktrace::log("agx", "compute: no channel movement vs baseline — firmware parsed initdata but hasn't driven rings yet (may need DevCtrl kick)");
    }
    // Snapshot the post-MSG_INIT cursors, then kick DevCtrl (DC_Init +
    // DC_UpdateIdleTS) to bring the firmware into full operation — this is what
    // starts the FWLog/Event channels the probe showed idle. Movement measured
    // against this fresh baseline is attributable to the kick, not to MSG_INIT.
    let post_init = read_channel_cursors();
    kick_devctrl(asc, &post_init);
}

// The compute (CP) cmdqueue channel. drm/asahi's run_job rings
// MSG_TX_DOORBELL | pipe_type | (priority<<2), where the index is PRIORITY (not a
// queue index). Compute priority 0 → channel (0<<2)|2 = 2 = CL_0, the canonical
// default. (An earlier experiment used CL_2/chan 10; the doorbell was never the
// blocker.) Ring/state VAs from the initdata dump (replayed → gpu_read/write).
const CL_STATE_VA: u64 = 0xffffffa04008bfd0; // CL_0 state
const CL_RING_VA: u64 = 0xffffffa000088000; // CL_0 ring
const CL_ITEM: u64 = 0x40; // RunCmdQueueMsg size
const CL_CHANNEL: u16 = 2; // (priority 0 << 2) | compute(2)
// Stats channel state (firmware→AP) — re-read to confirm liveness during dispatch.
const STATS_STATE_VA: u64 = 0xffffffa040563fd0;
// Event channel state (firmware→AP) — the firmware posts job completion/error
// events here. Movement ⇒ the firmware actually PROCESSED our submission (even if
// it errored); no movement + Stats moving ⇒ it never scanned the work channel.
const EVENT_STATE_VA: u64 = 0xffffffa040377fd0;

/// Round `n` up to the 16 KiB UAT page.
fn page_up(n: u64) -> u64 {
    (n + uat::PAGE_MASK) & !uat::PAGE_MASK
}

/// Place `data` at context VA `va` (alloc zeroed DRAM, map it, copy, clean).
/// Returns the backing phys (identity-mapped, CPU-readable) or `None`.
fn place(ctx: &GpuContext, va: u64, data: &[u8], pipeline: bool) -> Option<u64> {
    let size = page_up((data.len() as u64).max(1));
    let phys = ctx_alloc_at(ctx, va, size, pipeline)?;
    if !data.is_empty() {
        // SAFETY: `phys` is our identity-mapped, zeroed, page-sized alloc; the copy
        // stays within it (data.len() <= size).
        unsafe { core::ptr::copy_nonoverlapping(data.as_ptr(), phys as *mut u8, data.len()) };
        dcache_clean(phys, data.len() as u64);
    }
    Some(phys)
}

/// **First compute dispatch attempt** — build the full submission (context +
/// shader + USC + CDM stream + microsequence + WorkCommandCP + command queue),
/// submit it on the CP channel, and read back the output. A best-effort shot to
/// get a hardware signal: the known-answer [`shaders::HELLO_COMPUTE`] kernel
/// writes `0xCAFEF00D` to `out[0]`. Success = that value appears; otherwise the
/// SGX fault register + channel state are dumped as the diagnostic. Several struct
/// tails are still reverse-engineering-uncertain (see `workcmd.rs`), so a fault is
/// an expected, informative outcome — not a regression.
fn dispatch_hello_compute() {
    use super::{cdm, shaders, workcmd};
    use workcmd::{ComputeCmd, MicroSeqRefs, QueueRefs};
    if !uat_init() {
        crate::ktrace::log("agx", "compute: uat not ready — run /agx up first");
        return;
    }
    let Some(ctx) = create_context(3) else {
        crate::ktrace::log("agx", "compute: create_context failed");
        return;
    };
    // --- assign context VAs (page-aligned; pre-assigned so cross-refs resolve) ---
    let g = CTX_GEM_BASE;
    let (va_out, va_arg, va_cdm, va_ms, va_wc, va_rs, va_ring, va_gbuf, va_nl, va_gctx, va_qi) = (
        g, g + 0x04000, g + 0x08000, g + 0x0c000, g + 0x10000, g + 0x14000, g + 0x18000,
        g + 0x1c000, g + 0x20000, g + 0x24000, g + 0x28000,
    );
    // Event/notifier objects (a work queue the firmware will service needs these).
    let (va_notifier, va_threshold) = (g + 0x2c000, g + 0x30000);
    let p = CTX_PIPELINE_BASE + 0x10000;
    let (va_shader, va_usc) = (p, p + 0x04000);

    // --- build object contents (VAs known, so refs resolve) ---
    // USC: bind arg buffer → uniforms, shared_none, shader code, reg count.
    let mut usc = cdm::UscStream::new();
    usc.uniform(0, shaders::HELLO_COMPUTE_UNIFORM_HALFS, va_arg);
    usc.shared_none();
    usc.shader((va_shader - CTX_PIPELINE_BASE) as u32);
    usc.registers(shaders::HELLO_COMPUTE_GPRS, 0);
    usc.no_preshader();

    // CDM command stream: launch 1×1×1 thread of the USC pipeline.
    let mut cdm_buf = alloc::vec::Vec::new();
    cdm_buf.extend_from_slice(&cdm::cdm_launch_word0(shaders::HELLO_COMPUTE_UNIFORM_HALFS, 0, 0, 0, cdm::MODE_DIRECT).to_le_bytes());
    cdm_buf.extend_from_slice(&cdm::cdm_launch_word1((va_usc & 0xffff_ffff) as u32).to_le_bytes());
    for w in cdm::cdm_size(1, 1, 1) {
        cdm_buf.extend_from_slice(&w.to_le_bytes());
    }
    for w in cdm::cdm_size(1, 1, 1) {
        cdm_buf.extend_from_slice(&w.to_le_bytes());
    }
    cdm_buf.extend_from_slice(&cdm::cdm_barrier().to_le_bytes());

    // Microsequence (references the WorkCommandCP's inline params + the queue).
    let ms = workcmd::microseq_compute(&MicroSeqRefs {
        job_params1: va_wc + 0x70,
        job_params2: va_wc + 0x1fc,
        work_queue: va_qi,
        vm_slot: ctx.ctx_id as u32,
        uuid: 1,
        notifier_buf: va_notifier + 0xa8, // NotifierState.unk_buf
        ..Default::default()
    });

    // WorkCommandCP.
    let wc = workcmd::run_compute(&ComputeCmd {
        vm_slot: ctx.ctx_id as u32,
        notifier: va_notifier,
        encoder: va_cdm,
        pipeline_base: CTX_PIPELINE_BASE,
        encoder_end: va_cdm + cdm_buf.len() as u64,
        encoder_id: 1,
        microsequence: va_ms,
        microsequence_size: ms.len() as u32,
        uuid: 1,
        stamp_value: 0,
    });

    // Command queue: RingState (wptr=1), a ring whose entry 0 = the WorkCommandCP.
    let rs = workcmd::ring_state(1, 0x80);
    let mut ring = alloc::vec![0u8; 0x80 * 8];
    ring[..8].copy_from_slice(&va_wc.to_le_bytes());
    let qi = workcmd::queue_info(&QueueRefs {
        state: va_rs,
        ring: va_ring,
        notifier_list: va_nl,
        gpu_buf: va_gbuf,
        gpu_context: va_gctx,
        uuid: 1,
        event_id: 0,
    });
    let arg = va_out.to_le_bytes();

    // Valid event/notifier/context objects (the work-queue scheduler prerequisite).
    let nl = workcmd::notifier_list(va_nl); // empty circular list (self-linked)
    let notif = workcmd::notifier(va_threshold, ctx.ctx_id as u32);
    let thr = workcmd::threshold();
    let gctx = workcmd::gpu_context_data();

    // --- place everything (capture RingState phys for post-run read-back) ---
    let rs_phys = place(&ctx, va_rs, &rs, false);
    let ok = place(&ctx, va_shader, shaders::HELLO_COMPUTE, true).is_some()
        && place(&ctx, va_usc, &usc.bytes, true).is_some()
        && place(&ctx, va_cdm, &cdm_buf, false).is_some()
        && place(&ctx, va_ms, &ms, false).is_some()
        && place(&ctx, va_wc, &wc, false).is_some()
        && rs_phys.is_some()
        && place(&ctx, va_ring, &ring, false).is_some()
        && place(&ctx, va_gbuf, &[], false).is_some()
        && place(&ctx, va_nl, &nl, false).is_some()
        && place(&ctx, va_gctx, &gctx, false).is_some()
        && place(&ctx, va_notifier, &notif, false).is_some()
        && place(&ctx, va_threshold, &thr, false).is_some()
        && place(&ctx, va_qi, &qi, false).is_some();
    let out_phys = place(&ctx, va_out, &arg, false); // out buffer (also holds nothing yet)
    let Some(out_phys) = out_phys else {
        crate::ktrace::log("agx", "compute: out buffer alloc failed");
        return;
    };
    // Overwrite the out buffer with the arg (out_ptr) at va_arg, keep va_out zeroed.
    let Some(_arg_phys) = place(&ctx, va_arg, &arg, false) else { return };
    if !ok {
        crate::ktrace::log("agx", "compute: object placement failed (OOM/map)");
        return;
    }
    // Zero the output slot so a stale value can't masquerade as success.
    // SAFETY: out_phys is our identity-mapped page.
    unsafe { core::ptr::write_volatile(out_phys as *mut u32, 0) };
    dcache_clean(out_phys, 4);
    crate::ktrace::log_fmt(format_args!(
        "agx: compute: ctx {} objects placed — shader@{va_shader:#x} usc@{va_usc:#x} cdm@{va_cdm:#x} wc@{va_wc:#x} qi@{va_qi:#x}",
        ctx.ctx_id
    ));

    // Snapshot the Stats cursor so we can tell if the firmware is even alive
    // during the dispatch (it advanced during the DevCtrl kick), and the Event
    // cursor to tell if the firmware PROCESSED our job (posts completion/error).
    let stats_before = gpu_read_u32(STATS_STATE_VA + STATE_WRITE_PTR).unwrap_or(0);
    let event_before = gpu_read_u32(EVENT_STATE_VA + STATE_WRITE_PTR).unwrap_or(0);

    // --- submit RunCmdQueueMsg on the CP channel (CL_0, id 2) + doorbell ---
    let msg = workcmd::run_cmd_queue_msg(workcmd::QUEUE_COMPUTE, va_qi, 1, 1, true);
    let wptr = gpu_read_u32(CL_STATE_VA + STATE_WRITE_PTR).unwrap_or(0);
    let slot = CL_RING_VA + CL_ITEM * (wptr as u64);
    if gpu_write(slot, &msg) != msg.len() {
        crate::ktrace::log("agx", "compute: CL ring slot unmapped");
        return;
    }
    gpu_write_u32(CL_STATE_VA + STATE_WRITE_PTR, wptr + 1);
    crate::ktrace::log_fmt(format_args!("agx: compute: submitted on CP channel CL_0 — WRITE_PTR {}->{}, doorbell {CL_CHANNEL:#x}", wptr, wptr + 1));

    let Some(base) = discover_asc_base() else {
        crate::ktrace::log("agx", "compute: ASC not available");
        return;
    };
    // SAFETY: `base` is the discovered ASC MMIO base (mapped during `up`).
    let asc = unsafe { Asc::new(base as usize) };
    let _ = send(&asc, proto::msg_doorbell(CL_CHANNEL), proto::EP_DOORBELL);
    // A general kick (0x10) nudges the firmware to pump its channels.
    let _ = send(&asc, proto::msg_doorbell(0x10), proto::EP_DOORBELL);

    // --- poll the output buffer for the magic + watch for a GPU fault ---
    let deadline = crate::arch::now_ms() + 3000;
    let mut got = 0u32;
    while crate::arch::now_ms() < deadline {
        pump();
        dcache_invalidate(out_phys, 4);
        // SAFETY: out_phys is our identity-mapped page.
        got = unsafe { core::ptr::read_volatile(out_phys as *const u32) };
        if got == shaders::HELLO_COMPUTE_MAGIC {
            break;
        }
        if ABORT.load(core::sync::atomic::Ordering::Relaxed) {
            break;
        }
    }
    if got == shaders::HELLO_COMPUTE_MAGIC {
        crate::ktrace::log_fmt(format_args!(
            "agx: compute: *** GPU DISPATCH SUCCEEDED *** out[0]={got:#x} (== HELLO_COMPUTE_MAGIC)"
        ));
    } else {
        crate::ktrace::log_fmt(format_args!(
            "agx: compute: no result — out[0]={got:#x} (want {:#x}). Dumping GPU fault state:",
            shaders::HELLO_COMPUTE_MAGIC
        ));
        sgx_fault_check();
        let cl_rd = gpu_read_u32(CL_STATE_VA).unwrap_or(0);
        let cl_wr = gpu_read_u32(CL_STATE_VA + STATE_WRITE_PTR).unwrap_or(0);
        crate::ktrace::log_fmt(format_args!("agx: compute: CL_0 cursors read={cl_rd:#x} write={cl_wr:#x} (read==write ⇒ firmware consumed the submit)"));
        // Firmware liveness during the dispatch: did Stats advance?
        let stats_after = gpu_read_u32(STATS_STATE_VA + STATE_WRITE_PTR).unwrap_or(0);
        crate::ktrace::log_fmt(format_args!(
            "agx: compute: Stats cursor {stats_before:#x}->{stats_after:#x} ({}), fw {}",
            if stats_after != stats_before { "MOVED" } else { "unchanged" },
            if stats_after != stats_before { "alive during dispatch" } else { "may be idle/stuck" }
        ));
        // Did the firmware PROCESS the job at all? The Event channel is where it
        // posts completion/error. MOVED ⇒ it read+processed (content bug, keep
        // fixing); unchanged (+Stats moving) ⇒ it never scanned the work channel
        // (firmware-internal activation prereq → needs a reference capture).
        let event_after = gpu_read_u32(EVENT_STATE_VA + STATE_WRITE_PTR).unwrap_or(0);
        crate::ktrace::log_fmt(format_args!(
            "agx: compute: Event cursor {event_before:#x}->{event_after:#x} ({}) ⇒ firmware {}",
            if event_after != event_before { "MOVED" } else { "unchanged" },
            if event_after != event_before {
                "PROCESSED the job (read+errored ⇒ content bug)"
            } else {
                "never scanned the work channel (activation prereq ⇒ capture needed)"
            }
        ));
        // The queue's own RingState: did the firmware bump gpu_rptr/gpu_doneptr?
        if let Some(rsp) = rs_phys {
            dcache_invalidate(rsp, 0x70);
            // SAFETY: rsp is our identity-mapped RingState page.
            let doneptr = unsafe { core::ptr::read_volatile(rsp as *const u32) };
            let gpu_rptr = unsafe { core::ptr::read_volatile((rsp + 0x30) as *const u32) };
            let cpu_wptr = unsafe { core::ptr::read_volatile((rsp + 0x40) as *const u32) };
            crate::ktrace::log_fmt(format_args!(
                "agx: compute: queue RingState gpu_doneptr={doneptr:#x} gpu_rptr={gpu_rptr:#x} cpu_wptr={cpu_wptr:#x} (gpu_rptr>0 ⇒ firmware read the queue ring)"
            ));
        }
    }
}

/// Bring up the AGX coprocessor. Gated on `is_apple()` + the `chitti.agx`
/// bootarg; a clean no-op otherwise. Returns the human-facing summary line.
fn up() -> Outcome {
    if !crate::arch::aarch64::is_apple() {
        crate::serial_println!("agx> not Apple Silicon — nothing to bring up");
        return Outcome::NoGpu;
    }
    if !bootarg_present(b"chitti.agx") {
        crate::serial_println!("agx> gated: add the `chitti.agx` bootarg on a BARE boot (never under the m1n1 hv)");
        return Outcome::NotRun;
    }
    let Some(base) = discover_asc_base() else {
        crate::serial_println!("agx> no `apple,agx-t8112` GPU node in the device tree");
        return Outcome::NoGpu;
    };
    ABORT.store(false, core::sync::atomic::Ordering::Relaxed);
    // Bare Apple boot has no serial; mirror the handshake trace to the chat pane.
    crate::ktrace::set_console_echo(true);
    crate::ktrace::log_fmt(format_args!("agx: GPU ASC base {base:#x}"));

    // PMGR gfx power on, then map the ASC MMIO window.
    if !pmgr_enable_gfx() {
        crate::serial_println!("agx> gfx power domain would not enable — aborting");
        return Outcome::SendFail;
    }
    // SGX liveness poke every working proxyclient path does before ASC boot
    // (`AGX.poke_sgx`): read then write `0x70001` to sgx reg[1] + 0xd14000.
    // drm/asahi doesn't spell it out (genpd/firmware cover it), but bare
    // bring-up treats it as load-bearing. After gfx is ACTIVE + the window mapped.
    sgx_poke();
    crate::arch::aarch64::mmu::map_device_gib(base);

    let mut rep = Report { asc_base: base, ..Default::default() };
    // SAFETY: `base` is the FDT-discovered, Device-mapped ASC CPU window.
    let asc = unsafe { Asc::new(base as usize) };
    let outcome = rtkit_boot(&asc, &mut rep);
    rep.outcome = outcome;
    rep.cpu_running = asc.cpu_running();
    REPORT.with(|r| *r = rep);

    // Compute milestone: replay the captured initdata memory map, start the app
    // endpoints, MSG_INIT the firmware, and observe.
    let _ = probe_app_endpoints; // (retained for on-demand diagnosis)
    if outcome == Outcome::Running {
        boot_firmware(&asc, &mut rep);
    }

    match outcome {
        Outcome::Running => crate::serial_println!("agx> coprocessor RUNNING — RTKit v{} up, {} endpoint(s) started", rep.version, count_eps(&rep.eps)),
        Outcome::NoHello => crate::serial_println!(
            "agx> NO HELLO after cpu_start — the gfx-asc control firmware is not resident.\n\
             agx> Milestone 1 is BLOCKED on Asahi GPU-firmware provisioning for this machine."
        ),
        Outcome::VersionMismatch => crate::serial_println!("agx> RTKit version mismatch — see ktrace"),
        Outcome::BadHello => crate::serial_println!("agx> first mailbox message was not HELLO — see ktrace"),
        Outcome::EpmapFail => crate::serial_println!("agx> endpoint-map exchange failed — see ktrace"),
        Outcome::SendFail => crate::serial_println!("agx> mailbox send failed (ASC stuck?) — see ktrace"),
        Outcome::Timeout => crate::serial_println!("agx> timed out reaching power ON — see ktrace"),
        Outcome::Crashed => crate::serial_println!("agx> IOP crashed during boot — see ktrace"),
        Outcome::NoGpu | Outcome::NotRun => {}
    }
    outcome
}

fn count_eps(e: &EndpointSet) -> u32 {
    e.crashlog as u32 + e.debug as u32 + e.syslog as u32 + e.ioreport as u32 + e.oslog as u32
}

/// `/agx status` — dump the last bring-up snapshot.
fn status() {
    let r = REPORT.with(|r| *r);
    crate::serial_println!("agx status:");
    crate::serial_println!("  outcome:     {:?}", r.outcome);
    crate::serial_println!("  asc base:    {:#x}", r.asc_base);
    crate::serial_println!("  cpu running: {}", r.cpu_running);
    crate::serial_println!("  rtkit ver:   {}", r.version);
    crate::serial_println!(
        "  endpoints:   crashlog={} debug={} syslog={} ioreport={} oslog={}",
        r.eps.crashlog, r.eps.debug, r.eps.syslog, r.eps.ioreport, r.eps.oslog
    );
    crate::serial_println!("  iop power:   {:#x}", r.iop_power);
    crate::serial_println!("  ap power:    {:#x}", r.ap_power);
    crate::serial_println!("  buffers:     {}", r.n_buffers);
}

/// Print one FDT property: name, byte length, and the value decoded as
/// big-endian u32 cells (addresses/phandles) plus an ASCII rendering when it
/// looks like a string list.
fn dump_prop(name: &[u8], val: &[u8]) {
    let n = core::str::from_utf8(name).unwrap_or("<name?>");
    // ASCII string-ish values (all printable/NUL) — show them as text.
    let stringish = !val.is_empty() && val.iter().all(|&b| b == 0 || (0x20..0x7f).contains(&b));
    if stringish && val.len() <= 96 {
        let s = core::str::from_utf8(&val[..val.len().saturating_sub(1)]).unwrap_or("");
        crate::serial_println!("  {n} ({} B) = \"{}\"", val.len(), s.replace('\0', "\",\""));
        return;
    }
    if val.len() % 4 == 0 && !val.is_empty() {
        // Decode as up to 16 BE u32 cells (pairs read as u64 for addresses).
        let mut line = alloc::string::String::new();
        use core::fmt::Write;
        let ncells = (val.len() / 4).min(16);
        for i in 0..ncells {
            let c = u32::from_be_bytes([val[i * 4], val[i * 4 + 1], val[i * 4 + 2], val[i * 4 + 3]]);
            let _ = write!(line, " {c:#x}");
        }
        crate::serial_println!("  {n} ({} B) =<{} >", val.len(), line);
    } else {
        crate::serial_println!("  {n} ({} B) = <{} bytes>", val.len(), val.len());
    }
}

/// `/agx sgx` — dump the AGX GPU node's every property + the resolved
/// `memory-region` carveouts (ttbs/pagetables/handoff/…) AS FILLED on the live
/// machine. This is the discovery step for Milestone 2's GPU-VA layout: it shows
/// whether m1n1 populated the `uat-*` carveouts (they are `reg=<0,0,0,0>` in the
/// static DTB) and surfaces any `rtkit-private-vm-region`-style property.
fn sgx_dump() {
    use alloc::vec::Vec;
    if !crate::arch::aarch64::is_apple() {
        crate::serial_println!("agx> not Apple Silicon");
        return;
    }
    let fdt = crate::arch::aarch64::boot::boot_x0();
    crate::serial_println!("agx sgx: GPU node ({})", core::str::from_utf8(AGX_COMPAT).unwrap_or("?"));
    // reg[0]=asc, reg[1]=sgx.
    for (i, nm) in [(0usize, "asc"), (1, "sgx")] {
        // SAFETY: boot_x0 is the FDT pointer (or non-FDT, rejected by the magic).
        if let Some((b, s)) = unsafe { crate::fdt::reg_nth_of_compatible(fdt, AGX_COMPAT, i) } {
            crate::serial_println!("  reg[{i}] {nm}: base={b:#x} size={s:#x}");
        }
    }
    crate::serial_println!("agx sgx: properties —");
    let mut memregion: Vec<u32> = Vec::new();
    let mut memnames: Vec<u8> = Vec::new();
    // SAFETY: boot_x0 is the FDT pointer (or non-FDT, rejected by the magic).
    let found = unsafe {
        crate::fdt::for_each_prop_of_compatible(fdt, AGX_COMPAT, &mut |name, val| {
            dump_prop(name, val);
            if name == b"memory-region" {
                for ch in val.chunks_exact(4) {
                    memregion.push(u32::from_be_bytes([ch[0], ch[1], ch[2], ch[3]]));
                }
            } else if name == b"memory-region-names" {
                memnames = val.to_vec();
            }
        })
    };
    if !found {
        crate::serial_println!("agx sgx: no {} node in the device tree", core::str::from_utf8(AGX_COMPAT).unwrap_or("?"));
        return;
    }
    // Resolve each memory-region phandle → its carveout (base,size) as filled.
    if !memregion.is_empty() {
        crate::serial_println!("agx sgx: memory-region carveouts (resolved) —");
        let names: Vec<&[u8]> = memnames.split(|&b| b == 0).filter(|s| !s.is_empty()).collect();
        for (i, &ph) in memregion.iter().enumerate() {
            let nm = names.get(i).and_then(|s| core::str::from_utf8(s).ok()).unwrap_or("?");
            // SAFETY: boot_x0 is the FDT pointer (or non-FDT, rejected by the magic).
            match unsafe { crate::fdt::reg_by_phandle(fdt, ph) } {
                Some((b, s)) => crate::serial_println!("  [{i}] {nm:14} phandle={ph:#x} base={b:#x} size={s:#x}"),
                None => crate::serial_println!("  [{i}] {nm:14} phandle={ph:#x} <unresolved>"),
            }
        }
    }
}

/// `/agx` command entry (aarch64).
pub fn command(arg: &str) {
    match arg.trim() {
        "" | "up" | "init" => {
            up();
        }
        "status" | "info" => status(),
        "sgx" | "dump" => sgx_dump(),
        "ctx" => context_selftest(),
        "compute" => dispatch_hello_compute(),
        other => {
            crate::serial_println!("agx> unknown subcommand '{other}' (try: up | status | sgx | ctx | compute)");
        }
    }
}
