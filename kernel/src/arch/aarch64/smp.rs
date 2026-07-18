//! aarch64 SMP (Phase 7): bring the secondary cores online and split the
//! hottest kernel across them. Under `qemu-system-aarch64 -accel hvf -smp N`
//! the vCPUs run on *native* M-series cores in parallel -- unlike x86 under
//! TCG, where extra vCPUs only contend for one host thread -- so a data-parallel
//! matvec is a real speedup here.
//!
//! Bring-up is via PSCI `CPU_ON` (the same HVC conduit as `poweroff`): the BSP
//! launches each secondary at [`smp_secondary_entry`], handing it a private
//! stack through the PSCI `context_id`. Each secondary enables its MMU (reusing
//! the BSP's identity map), claims a worker slot, and parks in [`ap_rust_entry`]
//! spinning on a job descriptor.
//!
//! The work model is a static-partition barrier, not a scheduler: for one
//! matvec the BSP writes the shared operands + each worker's disjoint row range,
//! bumps a generation counter (release), computes its own range, then waits for
//! every worker to report that generation done (acquire). Weights/activation
//! are read-only during the pass and each core writes a disjoint slice of `y`,
//! so it is race-free without per-element locking.
//!
//! The wait is **bounded** (see [`barrier`]): a worker that never wakes — a
//! hypervisor that parks a trapped `WFE` until an interrupt, as VirtualBox-ARM
//! does — has its range recomputed on the BSP and the fleet degrades to
//! single-core for the session. Workers also enable the counter **event
//! stream** so `WFE` self-wakes even without `SEV`, and a boot-time self-test
//! in [`init`] detects a broken wake path before the first real job.

use crate::cortex::tensor;
use core::arch::global_asm;
use core::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering};

/// Max cores we size the static tables for (QEMU `-smp` is 4 here; the M2 has
/// 8). Only cores actually brought online are used.
const MAX_CPUS: usize = 8;
/// Per-secondary stack. The worker loop is shallow (it just calls the matvec
/// kernel), so 64 KiB is ample.
const AP_STACK_SIZE: usize = 64 * 1024;
/// Below this row count a matvec isn't worth the cross-core sync; the BSP does
/// it alone.
const PARALLEL_MIN_ROWS: usize = 256;

/// Total cores (BSP + workers) a **decode** (single-vector) matvec may use.
/// Decode re-streams the whole model per token, so it is memory-bandwidth-
/// bound: it saturates DRAM well before all 8 cores are busy, and past ~4 the
/// extra threads only contend on the bus while the slow (E-)cores drag the
/// barrier. Measured on this M2: llama.cpp decodes *faster* on 4 threads than
/// 8 (4.37 vs 3.18 t/s). Prefill (batched matmul) is compute-bound and keeps
/// the full fleet.
const DECODE_MAX_PARTS: usize = 4;

/// Workers for a bandwidth-bound decode matvec — [`fleet_workers`] capped so
/// BSP + workers ≤ [`DECODE_MAX_PARTS`]. Reserved: a hard core cap on decode
/// measured *worse* here (our weighted-barrier row split degrades at small
/// part counts, unlike llama.cpp's independent thread pool), so decode keeps
/// the full fleet + speed-weighted rows; kept for future revisit.
#[allow(dead_code)]
fn decode_workers() -> usize {
    fleet_workers().min(DECODE_MAX_PARTS.saturating_sub(1))
}

#[repr(C, align(16))]
struct ApStack([u8; AP_STACK_SIZE]);
/// One private stack per potential secondary core (BSS; never zeroed-critical).
static mut AP_STACKS: [ApStack; MAX_CPUS] = [const { ApStack([0; AP_STACK_SIZE]) }; MAX_CPUS];

/// Number of secondaries that have come online and claimed a worker slot.
static N_WORKERS: AtomicUsize = AtomicUsize::new(0);

/// Set when a worker missed a barrier deadline: its wake mechanism (SEV /
/// counter event stream) evidently doesn't reach a `WFE`-parked vCPU on this
/// hypervisor (observed on VirtualBox-ARM, which blocks a trapped `WFE` until
/// an *interrupt* — and no interrupts are routed to the secondaries). Once
/// set, every dispatcher runs single-core forever: slow beats a silent hang.
static DEGRADED: AtomicBool = AtomicBool::new(false);

/// Workers the dispatchers may use: 0 once [`DEGRADED`] (single-core fallback).
fn active_workers() -> usize {
    if DEGRADED.load(Ordering::Relaxed) { 0 } else { N_WORKERS.load(Ordering::Relaxed) }
}

/// Disable multi-core dispatch permanently and say why (once).
fn degrade(reason: &str, slot: usize) {
    if !DEGRADED.swap(true, Ordering::AcqRel) {
        crate::ktrace::log_fmt(format_args!(
            "smp: worker {slot} {reason} -- disabling multi-core dispatch (single-core fallback; hypervisor WFE wake broken?)"
        ));
    }
}

/// The single matvec job descriptor all workers watch. Shared operands plus a
/// per-worker-slot row range; `go` is the generation counter / release point.
struct Job {
    w: AtomicPtr<u8>,
    xq: AtomicPtr<i8>,
    xs: AtomicPtr<f32>,
    y: AtomicPtr<f32>,
    n_cols: AtomicUsize,
    m_count: AtomicUsize, // activation columns (1 = decode matvec, >1 = batched prefill)
    n_rows: AtomicUsize,  // total rows (the `y` column stride)
    // Kernel selector, decoupled from `qtype`: 0 = Q8_0 SDOT matmul (xq/xs),
    // 1 = Q4_0 SDOT (xq/xs), 2 = generic dequant-and-dot over the f32 `xf`
    // (using `qtype` as the dequant type), 3 = Q4_K SDOT (xq/xs),
    // 4 = generic row work (`fn_ptr(ctx)` over `[row_start, row_end)` — video
    // YUV→RGB and other data-parallel kernels).
    mode: AtomicUsize,
    qtype: AtomicUsize,
    xf: AtomicPtr<f32>,
    /// Function pointer for mode 4: `unsafe fn(start, end, ctx)`.
    fn_ptr: AtomicUsize,
    /// Opaque context pointer for mode 4.
    ctx: AtomicPtr<u8>,
    row_start: [AtomicUsize; MAX_CPUS],
    row_end: [AtomicUsize; MAX_CPUS],
    go: AtomicU64,
}

static JOB: Job = Job {
    w: AtomicPtr::new(core::ptr::null_mut()),
    xq: AtomicPtr::new(core::ptr::null_mut()),
    xs: AtomicPtr::new(core::ptr::null_mut()),
    y: AtomicPtr::new(core::ptr::null_mut()),
    n_cols: AtomicUsize::new(0),
    m_count: AtomicUsize::new(1),
    n_rows: AtomicUsize::new(0),
    mode: AtomicUsize::new(0),
    qtype: AtomicUsize::new(0),
    xf: AtomicPtr::new(core::ptr::null_mut()),
    fn_ptr: AtomicUsize::new(0),
    ctx: AtomicPtr::new(core::ptr::null_mut()),
    row_start: [const { AtomicUsize::new(0) }; MAX_CPUS],
    row_end: [const { AtomicUsize::new(0) }; MAX_CPUS],
    go: AtomicU64::new(0),
};
/// Per-worker-slot "generation last completed", so the BSP can barrier on it.
static DONE: [AtomicU64; MAX_CPUS] = [const { AtomicU64::new(0) }; MAX_CPUS];
/// Per-worker-slot "generation picked up": a worker stamps this the moment it
/// wakes for a generation, *before* reading its row range. Lets the barrier
/// tell "never woke" (safe to retract the range and recompute on the BSP)
/// apart from "in flight" (give it more time).
static CLAIM: [AtomicU64; MAX_CPUS] = [const { AtomicU64::new(0) }; MAX_CPUS];

/// Cumulative cycles (CNTVCT_EL0) each core has spent *computing* a matmul
/// chunk — the numerator for `/top`'s per-core utilisation. Index = core id
/// (0 = BSP). Workers accumulate in [`worker_loop`]; the BSP accumulates its
/// own chunk in the `matmul_sdot`/`matvec_*` drivers. Since CNTVCT is a
/// wall-clock counter shared by all cores, a core's busy% over a window is
/// `(busy_delta) / (cntvct window)` — no frequency needed.
static CORE_BUSY: [AtomicU64; MAX_CPUS] = [const { AtomicU64::new(0) }; MAX_CPUS];

/// Cumulative compute-cycles for `core` (see [`CORE_BUSY`]). 0 for an unknown
/// core. Read by the shell's `/top` to derive per-core busy percentages.
pub fn core_busy_cycles(core: usize) -> u64 {
    CORE_BUSY.get(core).map(|a| a.load(Ordering::Relaxed)).unwrap_or(0)
}

/// Add `cycles` to `core`'s busy accumulator (the BSP records its own chunk).
#[inline]
fn add_busy(core: usize, cycles: u64) {
    if let Some(a) = CORE_BUSY.get(core) {
        a.fetch_add(cycles, Ordering::Relaxed);
    }
    // Remember the last slice cost for inverse-speed row weighting. A zero
    // sample means "no data yet" → equal split; store at least 1.
    if let Some(a) = LAST_SLICE.get(core) {
        a.store(cycles.max(1), Ordering::Relaxed);
    }
}

/// Cycles spent on the most recent matvec/matmul slice per core (0 = unknown).
/// Used by [`row_boundary`] so faster cores (P) absorb more rows and the
/// barrier no longer waits on equal-share E-core stragglers under HVF.
static LAST_SLICE: [AtomicU64; MAX_CPUS] = [const { AtomicU64::new(0) }; MAX_CPUS];

/// Exclusive end of part `k` in a `n_parts`-way split of `n_rows` (part 0 is
/// the BSP). When every core has a [`LAST_SLICE`] sample, rows are allotted
/// ∝ 1/cycles (faster cores get more work); otherwise equal split.
#[inline]
fn row_boundary(n_rows: usize, n_parts: usize, k: usize) -> usize {
    if k == 0 {
        return 0;
    }
    if k >= n_parts {
        return n_rows;
    }
    let mut w = [0u64; MAX_CPUS];
    let mut have = true;
    for i in 0..n_parts {
        let c = LAST_SLICE[i].load(Ordering::Relaxed);
        if c == 0 {
            have = false;
            break;
        }
        // weight ∝ 1/cost; scale so small cycle counts stay integral.
        w[i] = (1_000_000u64 / c).max(1);
    }
    if !have {
        return k * n_rows / n_parts;
    }
    let sum: u64 = w[..n_parts].iter().sum::<u64>().max(1);
    let mut acc_w = 0u64;
    for i in 0..k {
        acc_w += w[i];
    }
    let end = ((n_rows as u64 * acc_w) / sum) as usize;
    end.min(n_rows)
}

extern "C" {
    /// The secondary-core entry stub (in `global_asm!` below). Its address is
    /// the PSCI `CPU_ON` entry point; it sets SP from the PSCI `context_id`,
    /// enables FP/SIMD, and calls [`ap_rust_entry`].
    fn smp_secondary_entry();
}

global_asm!(
    r#"
.section .text
.global smp_secondary_entry
smp_secondary_entry:
    mov  sp, x0                 // x0 = PSCI context_id = this core's stack top
    mrs  x1, cpacr_el1          // enable FP/SIMD (NEON) at EL1
    orr  x1, x1, #(3 << 20)
    msr  cpacr_el1, x1
    isb
    bl   ap_rust_entry          // never returns
1:  wfi
    b    1b
"#
);

/// PSCI `CPU_ON` (64-bit function id `0xC400_0003`) via the HVC conduit the
/// QEMU `virt` machine advertises. Starts `target`'s core at `entry` with
/// `ctx` delivered in `x0`. Returns the PSCI status (0 = success).
fn psci_cpu_on(target: u64, entry: u64, ctx: u64) -> i64 {
    let ret: u64;
    // SAFETY: PSCI CPU_ON has no memory effects in this core; per SMCCC it
    // returns in x0 and preserves x4+. We mark x1-x3 clobbered.
    unsafe {
        core::arch::asm!(
            "hvc #0",
            inout("x0") 0xC400_0003u64 => ret,
            in("x1") target,
            in("x2") entry,
            in("x3") ctx,
            options(nostack),
        );
    }
    ret as i64
}

/// PSCI status for "no such target CPU" — how `CPU_ON` reports an index past the
/// last core, which is how we discover the CPU count (cores are numbered
/// contiguously from 0 on the `virt` machine and under VirtualBox/UEFI).
const PSCI_INVALID_PARAMETERS: i64 = -2;

/// Bring **every available** secondary core online (BSP + secondaries). Runs on
/// the BSP after the MMU is on. The core count is discovered dynamically, not
/// hardcoded: we `CPU_ON` indices 1, 2, … until PSCI reports no such core
/// (`INVALID_PARAMETERS`), capped at `MAX_CPUS` (the per-core stack array bound).
/// This tracks the actual `-smp`/hardware CPU count — the parity match to x86's
/// `smp::init`, which brings up whatever APs Limine's SMP response lists. The BSP
/// then waits briefly for the launched cores to register worker slots.
pub fn init() {
    // PSCI is the ARM/SBSA (QEMU, VirtualBox, UEFI) way to start secondaries,
    // over an `hvc`/`smc` conduit. Apple Silicon has **no PSCI** — its FDT has
    // no `arm,psci-*` node and cores start via Apple's CPU-start MMIO (a
    // follow-up). Issuing `hvc` there traps to EL2 (m1n1) and halts the guest,
    // so skip PSCI **when a valid FDT explicitly lacks it** (the m1n1 path
    // always passes one). Boots with *no* FDT in x0 — QEMU/VBox `-kernel`
    // ELF and the UEFI stub — are SBSA platforms where PSCI is the norm, so
    // they keep the PSCI bring-up (gating those on FDT contents once turned
    // SMP off on QEMU entirely: "no FDT" is not "FDT says no PSCI").
    let fdt = super::boot::boot_x0();
    // SAFETY: `boot_x0` is the FDT pointer (or non-FDT/0, rejected by the magic).
    let fdt_present = unsafe { crate::fdt::present(fdt) };
    // SAFETY: as above; only consulted when the header validated.
    let has_psci = fdt_present
        && unsafe {
            crate::fdt::has_compatible(fdt, b"arm,psci-1.0")
                || crate::fdt::has_compatible(fdt, b"arm,psci-0.2")
                || crate::fdt::has_compatible(fdt, b"arm,psci")
        };
    if fdt_present && !has_psci {
        crate::ktrace::log("smp", "FDT advertises no PSCI (Apple Silicon) -- single-core; Apple CPU-start is a follow-up");
        return;
    }
    let entry_fn: unsafe extern "C" fn() = smp_secondary_entry;
    let entry = entry_fn as usize as u64;
    let mut started = 0usize;
    for i in 1..MAX_CPUS {
        // SAFETY: `AP_STACKS[i]` is a distinct static stack region; we hand its
        // top to core `i` as the PSCI context_id (the asm stub loads it into SP).
        let stack_top = unsafe {
            let base = core::ptr::addr_of!(AP_STACKS[i]) as u64;
            base + AP_STACK_SIZE as u64
        };
        // QEMU `virt` numbers cores' MPIDR affinity 0..N, so target == index.
        let rc = psci_cpu_on(i as u64, entry, stack_top);
        if rc == 0 {
            started += 1;
        } else if rc == PSCI_INVALID_PARAMETERS {
            break; // no core at this index -> we've enumerated them all
        } else {
            crate::ktrace::log_fmt(format_args!("smp: CPU_ON core {i} failed (psci={rc})"));
            break;
        }
    }

    // Wait (bounded) for the launched secondaries to enable their MMU and register.
    let deadline = crate::arch::now_ms() + 1000;
    while N_WORKERS.load(Ordering::Acquire) < started && crate::arch::now_ms() < deadline {
        core::hint::spin_loop();
    }
    let online = N_WORKERS.load(Ordering::Acquire) + 1;
    crate::ktrace::log_fmt(format_args!("smp: {online} cores online (BSP + {} workers, discovered via PSCI)", online - 1));
    // Whether the int8 matrix-multiply (FEAT_I8MM / `vmmlaq_s32`) is available
    // to the guest gates the Q1_0/Q2_0 i8mm matmul fast path; log it once so a
    // slow prefill can be diagnosed as "HVF masked I8MM → vdotq fallback".
    crate::ktrace::log_fmt(format_args!(
        "smp: FEAT_I8MM {} (Q1_0/Q2_0 batched matmul uses {})",
        if super::has_i8mm() { "yes" } else { "no" },
        if super::has_i8mm() { "vmmla i8mm" } else { "vdotq" }
    ));

    // Wake self-test: push one trivial generation through the real job/barrier
    // machinery. A hypervisor that never wakes a `WFE`-parked vCPU (VirtualBox-
    // ARM blocks it until an interrupt, and secondaries get none) degrades to
    // single-core HERE, with one clear line — instead of hanging the first
    // inference matvec mid-prefill.
    if online > 1 {
        unsafe fn nop_rows(_s: usize, _e: usize, _ctx: *mut u8) {}
        // SAFETY: `nop_rows` touches no memory; ctx is unused (null).
        unsafe { parallel_for(4096, 1, nop_rows, core::ptr::null_mut()) };
        if DEGRADED.load(Ordering::Relaxed) {
            crate::ktrace::log_fmt(format_args!("smp: wake self-test FAILED -- workers online but unwakeable; running single-core"));
        } else {
            crate::ktrace::log_fmt(format_args!("smp: wake self-test ok ({} workers)", online - 1));
        }
    }
}

/// Number of cores currently participating (BSP + registered workers).
pub fn online_cpus() -> usize {
    N_WORKERS.load(Ordering::Relaxed) + 1
}

/// Secondary-core Rust entry (called from the asm stub with SP already set).
/// Enables the MMU from the shared identity map (so atomics work), claims a
/// worker slot, and runs the worker loop forever.
#[no_mangle]
extern "C" fn ap_rust_entry() -> ! {
    // MMU first: before it, RAM is Device-typed and the atomics below can't
    // complete. `enable_secondary` is pure asm (no atomics), so it is safe here.
    // SAFETY: the BSP built `L1` before launching us; we only program our own
    // per-core translation registers to it (VA==PA keeps our stack live).
    unsafe { super::mmu::enable_secondary() };

    let slot = N_WORKERS.fetch_add(1, Ordering::AcqRel);
    if slot >= MAX_CPUS {
        // More cores than we sized for: park.
        loop {
            crate::arch::hlt();
        }
    }
    worker_loop(slot);
}

/// Publish prior stores to other cores, then wake any `WFE`-parked workers.
/// `SEV` sets a sticky per-core event flag on every core, so there's no
/// lost-wakeup race with a worker about to `WFE` (its `WFE` consumes the flag
/// and returns immediately).
#[inline]
fn signal_workers() {
    // SAFETY: `dsb ishst` orders the preceding job stores before the event;
    // `sev` has no memory effects.
    unsafe { core::arch::asm!("dsb ishst", "sev", options(nomem, nostack, preserves_flags)) };
}

/// Enable the virtual-counter **event stream** on this core (CNTKCTL_EL1:
/// EVNTEN, EVNTI = bit 12), so a `WFE` self-wakes every 2^13 counter ticks
/// (~340 µs at 24 MHz) even if a cross-core `SEV` never arrives — the standard
/// hypervisor-/hardware-proof bound on `WFE` parking (Linux does the same).
/// Read-modify-write so the EL0 counter-access bits are preserved.
fn enable_wfe_event_stream() {
    let mut v: u64;
    // SAFETY: CNTKCTL_EL1 is EL1-writable; setting EVNTEN/EVNTI only starts
    // event generation for WFE, with no memory effects.
    unsafe {
        core::arch::asm!("mrs {}, cntkctl_el1", out(reg) v, options(nomem, nostack, preserves_flags));
        v = (v & !(0xf << 4) & !(1 << 3)) | (1 << 2) | (12 << 4); // EVNTEN=1, EVNTDIR=0, EVNTI=12
        core::arch::asm!("msr cntkctl_el1, {}", in(reg) v, options(nomem, nostack, preserves_flags));
    }
}

// --- fire-and-forget async job (one slot) ------------------------------------
//
// A long-running task (tens of ms — video decode-ahead) must not ride the
// matvec barrier: it would stall inference for its whole duration. Instead the
// **last** worker slot can be temporarily reserved for one async job; the
// dispatchers exclude it from the fleet while the job is active. Submission
// and matvec dispatch both happen on the BSP, so there is no submit/dispatch
// race by construction.

const ASYNC_IDLE: usize = 0;
const ASYNC_SUBMITTED: usize = 1;
const ASYNC_DONE: usize = 2;
static ASYNC_STATE: AtomicUsize = AtomicUsize::new(ASYNC_IDLE);
static ASYNC_FN: AtomicUsize = AtomicUsize::new(0);
static ASYNC_CTX: AtomicPtr<u8> = AtomicPtr::new(core::ptr::null_mut());
static ASYNC_SLOT: AtomicUsize = AtomicUsize::new(usize::MAX);

/// Submit `f(ctx)` to run once on the reserved worker. `false` (caller runs it
/// synchronously instead) when there are no usable workers or a job is already
/// active. BSP-only.
///
/// # Safety
/// `ctx` must stay valid and untouched by the caller until [`async_take_done`]
/// returns true; `f` must be safe to run on another core with that contract.
pub unsafe fn async_submit(f: unsafe fn(*mut u8), ctx: *mut u8) -> bool {
    let workers = fleet_workers();
    if workers == 0 || ASYNC_STATE.load(Ordering::Acquire) != ASYNC_IDLE {
        return false;
    }
    ASYNC_FN.store(f as usize, Ordering::Relaxed);
    ASYNC_CTX.store(ctx, Ordering::Relaxed);
    ASYNC_SLOT.store(workers - 1, Ordering::Relaxed);
    ASYNC_STATE.store(ASYNC_SUBMITTED, Ordering::Release);
    signal_workers();
    true
}

/// Poll for completion; consumes the done state (job slot returns to the
/// fleet). BSP-only.
pub fn async_take_done() -> bool {
    if ASYNC_STATE.load(Ordering::Acquire) == ASYNC_DONE {
        ASYNC_STATE.store(ASYNC_IDLE, Ordering::Release);
        return true;
    }
    false
}

/// True while an async job is submitted or running (its slot is reserved).
pub fn async_active() -> bool {
    ASYNC_STATE.load(Ordering::Acquire) != ASYNC_IDLE
}

/// Workers available to the barrier fleet: the async slot (always the last)
/// drops out while a job is active. Dispatchers must also zero the excluded
/// slot's row range so a late generation-claim by that worker no-ops.
fn fleet_workers() -> usize {
    let w = active_workers();
    if w > 0 && async_active() {
        w - 1
    } else {
        w
    }
}

/// Retract row ranges of slots the current dispatch does not use, so a worker
/// that joins late (e.g. finishing an async job mid-generation) computes
/// nothing instead of stale ranges against fresh operands.
fn clear_unused_slots(used: usize) {
    let total = N_WORKERS.load(Ordering::Relaxed).min(MAX_CPUS);
    for s in used..total {
        JOB.row_start[s].store(0, Ordering::Relaxed);
        JOB.row_end[s].store(0, Ordering::Relaxed);
    }
}

/// How long the barrier waits for all workers before treating a slot as a
/// straggler. Generations complete in µs–ms; 500 ms of silence means the wake
/// never happened (or the host scheduler is pathologically starved).
const BARRIER_WAIT_MS: u64 = 500;
/// Extra time granted to a worker that *claimed* the generation (it woke and
/// may be mid-compute) before the BSP gives up and recomputes its range.
const BARRIER_GRACE_MS: u64 = 1500;

/// Wait for every worker to finish generation `g` — **bounded**. A slot that
/// misses the deadline is handled per its [`CLAIM`] stamp:
///
/// - never claimed: the worker never woke (hypervisor `WFE` wake broken). Its
///   range is retracted (a late wake then no-ops) and recomputed on the BSP.
/// - claimed but not done: it woke and is computing; wait a further grace
///   window, then recompute anyway.
///
/// Either way the fleet is [`degrade`]d so no future job depends on broken
/// wakes. If a straggler turns out to be merely late and computes its original
/// range concurrently with the BSP's recompute, both write the *same values*
/// to the same disjoint `y` rows (same operands, deterministic kernel), so the
/// overlap is benign. Deadlines are absolute from barrier entry, so the total
/// bound is ~WAIT+GRACE regardless of worker count.
fn barrier(workers: usize, g: u64, recompute: &dyn Fn(usize, usize)) {
    let t0 = crate::arch::now_ms();
    let mut spins = 0u32;
    for s in 0..workers {
        loop {
            if DONE[s].load(Ordering::Acquire) == g {
                break;
            }
            if crate::arch::now_ms().saturating_sub(t0) > BARRIER_WAIT_MS {
                handle_straggler(s, g, t0, recompute);
                break;
            }
            // Re-issue SEV occasionally: heals a hypervisor that loses rare
            // wake events without pegging the interconnect.
            spins = spins.wrapping_add(1);
            if spins % 8192 == 0 {
                signal_workers();
            }
            core::hint::spin_loop();
        }
    }
}

/// Deadline-miss path of [`barrier`] — see there for the protocol.
#[cold]
fn handle_straggler(s: usize, g: u64, t0: u64, recompute: &dyn Fn(usize, usize)) {
    let rs = JOB.row_start[s].load(Ordering::Relaxed);
    let re = JOB.row_end[s].load(Ordering::Relaxed);
    if CLAIM[s].load(Ordering::Acquire) != g {
        // Never woke. Retract the range first so a late wake no-ops, then do
        // its work here. (A wake racing this retraction computes the original
        // range — identical values, see the barrier doc.)
        JOB.row_start[s].store(0, Ordering::Relaxed);
        JOB.row_end[s].store(0, Ordering::Relaxed);
        degrade("never woke for a job", s);
        recompute(rs, re);
        return;
    }
    // Claimed: it's computing. Give it the grace window before stepping in.
    while DONE[s].load(Ordering::Acquire) != g {
        if crate::arch::now_ms().saturating_sub(t0) > BARRIER_WAIT_MS + BARRIER_GRACE_MS {
            degrade("wedged mid-job", s);
            recompute(rs, re);
            return;
        }
        core::hint::spin_loop();
    }
}

/// A worker core: **park on `WFE`** until the BSP publishes a new job
/// generation, run this slot's row range through the SDOT matvec, mark the
/// generation done, repeat. `WFE` (not a busy `spin_loop`) is essential on
/// hypervisors: a busy-spinning idle secondary pegs a host core, and with many
/// vCPUs (e.g. an 8-CPU VirtualBox VM) the spinning secondaries starve the boot
/// core during time-sensitive work like USB xHCI enumeration — which then times
/// out, leaving no keyboard and (with the timer IRQ also cooperative) a frozen
/// console. Parked workers cost nothing until the BSP `signal_workers()`.
fn worker_loop(slot: usize) -> ! {
    enable_wfe_event_stream();
    let mut last = 0u64;
    loop {
        // Long-running fire-and-forget job addressed to this slot (video
        // decode-ahead). Runs outside the barrier fleet: while it is active
        // the dispatchers exclude this slot (see `async_reserved`).
        if ASYNC_STATE.load(Ordering::Acquire) == ASYNC_SUBMITTED && ASYNC_SLOT.load(Ordering::Relaxed) == slot {
            let f = ASYNC_FN.load(Ordering::Relaxed);
            let ctx = ASYNC_CTX.load(Ordering::Relaxed);
            if f != 0 {
                let t0 = super::cycle_count();
                // SAFETY: the submitter published fn/ctx before the state
                // (release) and won't touch ctx again until it observes DONE.
                unsafe {
                    let f: unsafe fn(*mut u8) = core::mem::transmute(f);
                    f(ctx);
                }
                add_busy(slot + 1, super::cycle_count().wrapping_sub(t0));
            }
            ASYNC_STATE.store(ASYNC_DONE, Ordering::Release);
            continue;
        }
        let g = JOB.go.load(Ordering::Acquire);
        if g == last {
            // SAFETY: `wfe` just parks until an event/IRQ; spurious wakes are
            // fine — we re-check `go` on the next iteration.
            unsafe { core::arch::asm!("wfe", options(nomem, nostack, preserves_flags)) };
            continue;
        }
        last = g;
        CLAIM[slot].store(g, Ordering::Release);
        let rs = JOB.row_start[slot].load(Ordering::Relaxed);
        let re = JOB.row_end[slot].load(Ordering::Relaxed);
        if re > rs {
            let t0 = super::cycle_count();
            let w = JOB.w.load(Ordering::Relaxed);
            let xq = JOB.xq.load(Ordering::Relaxed);
            let xs = JOB.xs.load(Ordering::Relaxed);
            let y = JOB.y.load(Ordering::Relaxed);
            let n_cols = JOB.n_cols.load(Ordering::Relaxed);
            let m_count = JOB.m_count.load(Ordering::Relaxed);
            let n_rows = JOB.n_rows.load(Ordering::Relaxed);
            let qtype = JOB.qtype.load(Ordering::Relaxed);
            let mode = JOB.mode.load(Ordering::Relaxed);
            // SAFETY: the BSP guarantees (via the `go` release/acquire) that the
            // operands are published and this slot's [rs, re) is in bounds and
            // disjoint from every other core's range, so the writes to `y` don't
            // alias. The weights/activation are read-only during the pass.
            unsafe {
                match mode {
                    0 => tensor::matmul_q8_0_sdot_rows(w, xq as *const i8, xs as *const f32, y, m_count, n_rows, rs, re, n_cols),
                    1 => tensor::matvec_q4_0_sdot_rows(w, xq as *const i8, xs as *const f32, y, rs, re, n_cols),
                    3 => tensor::matvec_q4_k_sdot_rows(w, xq as *const i8, xs as *const f32, y, rs, re, n_cols),
                    // Q1/Q2: static contiguous row ranges (not work-steal).
                    // Fine-grained steal re-ran Q1 act-sum precompute per slab
                    // and shredded sequential weight locality — measured ~1 t/s
                    // tg on 8 vCPUs, i.e. near single-core. Contiguous split
                    // restores L2 streaming + one precompute per core.
                    5 => tensor::matvec_q2_0_sdot_rows(w, xq as *const i8, xs as *const f32, y, rs, re, n_cols),
                    6 => tensor::matvec_q1_0_sdot_rows(w, xq as *const i8, xs as *const f32, y, rs, re, n_cols),
                    7 => tensor::matmul_q1_0_sdot_rows(w, xq as *const i8, xs as *const f32, y, m_count, n_rows, rs, re, n_cols),
                    8 => tensor::matmul_q2_0_sdot_rows(w, xq as *const i8, xs as *const f32, y, m_count, n_rows, rs, re, n_cols),
                    // 9/10 = Q1_0 / Q8_0 i8mm matmul. Only dispatched when
                    // has_i8mm() is true, so the kernels' FEAT_I8MM precondition holds.
                    9 => tensor::matmul_q1_0_i8mm_rows(w, xq as *const i8, xs as *const f32, y, m_count, n_rows, rs, re, n_cols),
                    10 => tensor::matmul_q8_0_i8mm_rows(w, xq as *const i8, xs as *const f32, y, m_count, n_rows, rs, re, n_cols),
                    11 => tensor::matmul_q4_0_i8mm_rows(w, xq as *const i8, xs as *const f32, y, m_count, n_rows, rs, re, n_cols),
                    4 => {
                        // Generic row work (video YUV→RGB, etc.).
                        let fp = JOB.fn_ptr.load(Ordering::Relaxed);
                        let ctx = JOB.ctx.load(Ordering::Relaxed);
                        if fp != 0 {
                            let f: unsafe fn(usize, usize, *mut u8) = core::mem::transmute(fp);
                            f(rs, re, ctx);
                        }
                    }
                    _ => tensor::matvec_quant_rows(qtype as u32, w, JOB.xf.load(Ordering::Relaxed), y, rs, re, n_cols),
                }
            }
            // Worker slot s runs on core s+1 (core 0 is the BSP).
            add_busy(slot + 1, super::cycle_count().wrapping_sub(t0));
        }
        DONE[slot].store(g, Ordering::Release);
    }
}

/// Run a `Q8_0` SDOT matmul (`y[m][r] = W[r] · xq[m]` for all `m` in
/// `0..m_count`, `r` in `0..n_rows`) across all online cores by row range. The
/// activations must already be quantized (`xq`/`xs`, `m_count` columns);
/// `m_count == 1` is the decode matvec. Falls back to a single-core pass when
/// only the BSP is online or the matrix is small. This is what
/// `tensor::matvec_q8_0_fast` (decode) and the batched prefill call on aarch64.
///
/// # Safety
/// Same contract as `tensor::matmul_q8_0_sdot_rows` over `[0, n_rows)`.
pub unsafe fn matmul_sdot(
    w: *const u8,
    xq: *const i8,
    xs: *const f32,
    y: *mut f32,
    m_count: usize,
    n_rows: usize,
    n_cols: usize,
) {
    let workers = fleet_workers();
    if workers == 0 || n_rows < PARALLEL_MIN_ROWS {
        // SAFETY: caller's contract; whole range on this core.
        unsafe { tensor::matmul_q8_0_sdot_rows(w, xq, xs, y, m_count, n_rows, 0, n_rows, n_cols) };
        return;
    }
    let n_parts = workers + 1; // BSP + workers
    let boundary = |k: usize| k * n_rows / n_parts;

    // Publish shared operands, then each worker's chunk (chunk k+1 for slot k).
    JOB.w.store(w as *mut u8, Ordering::Relaxed);
    JOB.xq.store(xq as *mut i8, Ordering::Relaxed);
    JOB.xs.store(xs as *mut f32, Ordering::Relaxed);
    JOB.y.store(y, Ordering::Relaxed);
    JOB.n_cols.store(n_cols, Ordering::Relaxed);
    JOB.m_count.store(m_count, Ordering::Relaxed);
    JOB.n_rows.store(n_rows, Ordering::Relaxed);
    JOB.mode.store(0, Ordering::Relaxed); // Q8_0 SDOT matmul
    for s in 0..workers {
        JOB.row_start[s].store(boundary(s + 1), Ordering::Relaxed);
        JOB.row_end[s].store(boundary(s + 2), Ordering::Relaxed);
    }
    // Release the job: bump the generation (BSP is the only writer of `go`).
    clear_unused_slots(workers);
    let g = JOB.go.load(Ordering::Relaxed) + 1;
    JOB.go.store(g, Ordering::Release);
    signal_workers(); // wake WFE-parked workers

    // BSP computes chunk 0 while the workers run theirs.
    // SAFETY: disjoint row range [0, boundary(1)); caller's contract.
    let t0 = super::cycle_count();
    unsafe { tensor::matmul_q8_0_sdot_rows(w, xq, xs, y, m_count, n_rows, 0, boundary(1), n_cols) };
    add_busy(0, super::cycle_count().wrapping_sub(t0));

    // Bounded barrier; a straggler's range is recomputed here (see `barrier`).
    // SAFETY: same contract as the chunk above, over the straggler's range.
    barrier(workers, g, &|rs, re| unsafe { tensor::matmul_q8_0_sdot_rows(w, xq, xs, y, m_count, n_rows, rs, re, n_cols) });
}

/// Run a generic (non-Q8_0) `Q*`-quant matvec `y = W · x` (f32 activation `x`)
/// across all online cores by row range, dequantizing each block on the fly
/// (`tensor::matvec_quant_rows`). Used for the mixed-quant 9B's Q4_0/Q4_1/Q5_K/
/// Q6_K tensors. Falls back to single-core when only the BSP is online or the
/// matrix is small.
///
/// # Safety
/// Same contract as `tensor::matvec_quant_rows` over `[0, n_rows)`.
pub unsafe fn matvec_quant(qt: u32, w: *const u8, x: *const f32, y: *mut f32, n_rows: usize, n_cols: usize) {
    let workers = fleet_workers();
    if workers == 0 || n_rows < PARALLEL_MIN_ROWS {
        // SAFETY: caller's contract; whole range on this core.
        unsafe { tensor::matvec_quant_rows(qt, w, x, y, 0, n_rows, n_cols) };
        return;
    }
    let n_parts = workers + 1;
    let boundary = |k: usize| k * n_rows / n_parts;
    JOB.w.store(w as *mut u8, Ordering::Relaxed);
    JOB.xf.store(x as *mut f32, Ordering::Relaxed);
    JOB.y.store(y, Ordering::Relaxed);
    JOB.n_cols.store(n_cols, Ordering::Relaxed);
    JOB.n_rows.store(n_rows, Ordering::Relaxed);
    JOB.qtype.store(qt as usize, Ordering::Relaxed);
    JOB.mode.store(2, Ordering::Relaxed); // generic dequant+dot
    for s in 0..workers {
        JOB.row_start[s].store(boundary(s + 1), Ordering::Relaxed);
        JOB.row_end[s].store(boundary(s + 2), Ordering::Relaxed);
    }
    clear_unused_slots(workers);
    let g = JOB.go.load(Ordering::Relaxed) + 1;
    JOB.go.store(g, Ordering::Release);
    signal_workers(); // wake WFE-parked workers
    // SAFETY: disjoint row range [0, boundary(1)); caller's contract.
    let t0 = super::cycle_count();
    unsafe { tensor::matvec_quant_rows(qt, w, x, y, 0, boundary(1), n_cols) };
    add_busy(0, super::cycle_count().wrapping_sub(t0));
    // SAFETY: same contract as the chunk above, over the straggler's range.
    barrier(workers, g, &|rs, re| unsafe { tensor::matvec_quant_rows(qt, w, x, y, rs, re, n_cols) });
}

/// Run a `Q4_0` SDOT matvec (int8-quantized activation `xq`/`xs`) across all
/// online cores by row range -- the fast Q4_0 path (`matvec_q4_0_sdot_rows`),
/// used for the 9B's many Q4_0 tensors. Falls back to single-core when only the
/// BSP is online or the matrix is small.
///
/// # Safety
/// Same contract as `tensor::matvec_q4_0_sdot_rows` over `[0, n_rows)`.
pub unsafe fn matvec_q4_0_sdot(w: *const u8, xq: *const i8, xs: *const f32, y: *mut f32, n_rows: usize, n_cols: usize) {
    let workers = fleet_workers();
    if workers == 0 || n_rows < PARALLEL_MIN_ROWS {
        // SAFETY: caller's contract; whole range on this core.
        unsafe { tensor::matvec_q4_0_sdot_rows(w, xq, xs, y, 0, n_rows, n_cols) };
        return;
    }
    let n_parts = workers + 1;
    let boundary = |k: usize| k * n_rows / n_parts;
    JOB.w.store(w as *mut u8, Ordering::Relaxed);
    JOB.xq.store(xq as *mut i8, Ordering::Relaxed);
    JOB.xs.store(xs as *mut f32, Ordering::Relaxed);
    JOB.y.store(y, Ordering::Relaxed);
    JOB.n_cols.store(n_cols, Ordering::Relaxed);
    JOB.mode.store(1, Ordering::Relaxed); // Q4_0 SDOT
    for s in 0..workers {
        JOB.row_start[s].store(boundary(s + 1), Ordering::Relaxed);
        JOB.row_end[s].store(boundary(s + 2), Ordering::Relaxed);
    }
    clear_unused_slots(workers);
    let g = JOB.go.load(Ordering::Relaxed) + 1;
    JOB.go.store(g, Ordering::Release);
    signal_workers(); // wake WFE-parked workers
    // SAFETY: disjoint row range [0, boundary(1)); caller's contract.
    let t0 = super::cycle_count();
    unsafe { tensor::matvec_q4_0_sdot_rows(w, xq, xs, y, 0, boundary(1), n_cols) };
    add_busy(0, super::cycle_count().wrapping_sub(t0));
    // SAFETY: same contract as the chunk above, over the straggler's range.
    barrier(workers, g, &|rs, re| unsafe { tensor::matvec_q4_0_sdot_rows(w, xq, xs, y, rs, re, n_cols) });
}

/// Run a generic data-parallel job over `[0, n)` split across online cores.
/// Each core invokes `f(start, end, ctx)` on a disjoint half-open range.
/// Falls back to a single call on the BSP when only one core is online or
/// `n < min_chunk * 2`. Used by the video player for YUV→RGB row convert.
///
/// # Safety
/// `f` must be safe for concurrent calls on disjoint `[start, end)` ranges
/// with the same `ctx`; `ctx` must remain live until this returns.
pub unsafe fn parallel_for(
    n: usize,
    min_chunk: usize,
    f: unsafe fn(usize, usize, *mut u8),
    ctx: *mut u8,
) {
    let workers = fleet_workers();
    if workers == 0 || n < min_chunk.saturating_mul(2).max(16) {
        f(0, n, ctx);
        return;
    }
    let n_parts = workers + 1;
    let boundary = |k: usize| k * n / n_parts;
    JOB.fn_ptr.store(f as usize, Ordering::Relaxed);
    JOB.ctx.store(ctx, Ordering::Relaxed);
    JOB.mode.store(4, Ordering::Relaxed);
    for s in 0..workers {
        JOB.row_start[s].store(boundary(s + 1), Ordering::Relaxed);
        JOB.row_end[s].store(boundary(s + 2), Ordering::Relaxed);
    }
    clear_unused_slots(workers);
    let g = JOB.go.load(Ordering::Relaxed) + 1;
    JOB.go.store(g, Ordering::Release);
    signal_workers();
    let t0 = super::cycle_count();
    f(0, boundary(1), ctx);
    add_busy(0, super::cycle_count().wrapping_sub(t0));
    // SAFETY: same contract as the chunk above, over the straggler's range.
    barrier(workers, g, &|rs, re| unsafe { f(rs, re, ctx) });
}

/// Run a `Q4_K` SDOT matvec (int8-quantized activation `xq`/`xs`) across all
/// online cores by row range -- the fast path for Q4_K_M-dominant models
/// (Ornith, Qwen3.6, Gemma-4 quants). Falls back to single-core when only the
/// BSP is online or the matrix is small.
///
/// # Safety
/// Same contract as `tensor::matvec_q4_k_sdot_rows` over `[0, n_rows)`.
pub unsafe fn matvec_q4_k_sdot(w: *const u8, xq: *const i8, xs: *const f32, y: *mut f32, n_rows: usize, n_cols: usize) {
    let workers = fleet_workers();
    if workers == 0 || n_rows < PARALLEL_MIN_ROWS {
        // SAFETY: caller's contract; whole range on this core.
        unsafe { tensor::matvec_q4_k_sdot_rows(w, xq, xs, y, 0, n_rows, n_cols) };
        return;
    }
    let n_parts = workers + 1;
    let boundary = |k: usize| k * n_rows / n_parts;
    JOB.w.store(w as *mut u8, Ordering::Relaxed);
    JOB.xq.store(xq as *mut i8, Ordering::Relaxed);
    JOB.xs.store(xs as *mut f32, Ordering::Relaxed);
    JOB.y.store(y, Ordering::Relaxed);
    JOB.n_cols.store(n_cols, Ordering::Relaxed);
    JOB.mode.store(3, Ordering::Relaxed); // Q4_K SDOT
    for s in 0..workers {
        JOB.row_start[s].store(boundary(s + 1), Ordering::Relaxed);
        JOB.row_end[s].store(boundary(s + 2), Ordering::Relaxed);
    }
    clear_unused_slots(workers);
    let g = JOB.go.load(Ordering::Relaxed) + 1;
    JOB.go.store(g, Ordering::Release);
    signal_workers(); // wake WFE-parked workers
    // SAFETY: disjoint row range [0, boundary(1)); caller's contract.
    let t0 = super::cycle_count();
    unsafe { tensor::matvec_q4_k_sdot_rows(w, xq, xs, y, 0, boundary(1), n_cols) };
    add_busy(0, super::cycle_count().wrapping_sub(t0));
    // SAFETY: same contract as the chunk above, over the straggler's range.
    barrier(workers, g, &|rs, re| unsafe { tensor::matvec_q4_k_sdot_rows(w, xq, xs, y, rs, re, n_cols) });
}

/// `y = W·x` for Q2_0 weights split across online cores by **contiguous**
/// row range — the fast PrismML ternary path (`matvec_q2_0_sdot_rows`,
/// Bonsai-27B). Falls back to single-core when only the BSP is online or the
/// matrix is small.
///
/// # Safety
/// Same contract as `tensor::matvec_q2_0_sdot_rows` over `[0, n_rows)`.
pub unsafe fn matvec_q2_0_sdot(w: *const u8, xq: *const i8, xs: *const f32, y: *mut f32, n_rows: usize, n_cols: usize) {
    let workers = fleet_workers();
    if workers == 0 || n_rows < PARALLEL_MIN_ROWS {
        // SAFETY: caller's contract; whole range on this core.
        unsafe { tensor::matvec_q2_0_sdot_rows(w, xq, xs, y, 0, n_rows, n_cols) };
        return;
    }
    let n_parts = workers + 1;
    let b0 = row_boundary(n_rows, n_parts, 1);
    JOB.w.store(w as *mut u8, Ordering::Relaxed);
    JOB.xq.store(xq as *mut i8, Ordering::Relaxed);
    JOB.xs.store(xs as *mut f32, Ordering::Relaxed);
    JOB.y.store(y, Ordering::Relaxed);
    JOB.n_cols.store(n_cols, Ordering::Relaxed);
    JOB.mode.store(5, Ordering::Relaxed); // Q2_0 SDOT
    for s in 0..workers {
        JOB.row_start[s].store(row_boundary(n_rows, n_parts, s + 1), Ordering::Relaxed);
        JOB.row_end[s].store(row_boundary(n_rows, n_parts, s + 2), Ordering::Relaxed);
    }
    clear_unused_slots(workers);
    let g = JOB.go.load(Ordering::Relaxed) + 1;
    JOB.go.store(g, Ordering::Release);
    signal_workers();
    let t0 = super::cycle_count();
    // SAFETY: disjoint row range [0, b0); caller's contract.
    unsafe { tensor::matvec_q2_0_sdot_rows(w, xq, xs, y, 0, b0, n_cols) };
    add_busy(0, super::cycle_count().wrapping_sub(t0));
    barrier(workers, g, &|rs, re| unsafe { tensor::matvec_q2_0_sdot_rows(w, xq, xs, y, rs, re, n_cols) });
}

/// `y = W·x` for Q1_0 (binary) weights split across online cores by
/// contiguous row range — the fast PrismML 1-bit path
/// (`matvec_q1_0_sdot_rows`, Bonsai-27B). Falls back to single-core when only
/// the BSP is online or the matrix is small.
///
/// # Safety
/// Same contract as `tensor::matvec_q1_0_sdot_rows` over `[0, n_rows)`.
pub unsafe fn matvec_q1_0_sdot(w: *const u8, xq: *const i8, xs: *const f32, y: *mut f32, n_rows: usize, n_cols: usize) {
    let workers = fleet_workers();
    if workers == 0 || n_rows < PARALLEL_MIN_ROWS {
        // SAFETY: caller's contract; whole range on this core.
        unsafe { tensor::matvec_q1_0_sdot_rows(w, xq, xs, y, 0, n_rows, n_cols) };
        return;
    }
    let n_parts = workers + 1;
    let b0 = row_boundary(n_rows, n_parts, 1);
    JOB.w.store(w as *mut u8, Ordering::Relaxed);
    JOB.xq.store(xq as *mut i8, Ordering::Relaxed);
    JOB.xs.store(xs as *mut f32, Ordering::Relaxed);
    JOB.y.store(y, Ordering::Relaxed);
    JOB.n_cols.store(n_cols, Ordering::Relaxed);
    JOB.mode.store(6, Ordering::Relaxed); // Q1_0 SDOT
    for s in 0..workers {
        JOB.row_start[s].store(row_boundary(n_rows, n_parts, s + 1), Ordering::Relaxed);
        JOB.row_end[s].store(row_boundary(n_rows, n_parts, s + 2), Ordering::Relaxed);
    }
    clear_unused_slots(workers);
    let g = JOB.go.load(Ordering::Relaxed) + 1;
    JOB.go.store(g, Ordering::Release);
    signal_workers();
    let t0 = super::cycle_count();
    // SAFETY: disjoint row range [0, b0); caller's contract.
    unsafe { tensor::matvec_q1_0_sdot_rows(w, xq, xs, y, 0, b0, n_cols) };
    add_busy(0, super::cycle_count().wrapping_sub(t0));
    barrier(workers, g, &|rs, re| unsafe { tensor::matvec_q1_0_sdot_rows(w, xq, xs, y, rs, re, n_cols) });
}

/// Batched weight-stationary `Q1_0` matmul (`y[m][r] = W[r]·xq[m]`) split
/// across online cores by contiguous row range — the batched-prefill kernel
/// for the 1-bit Bonsai (`matmul_q1_0_sdot_rows`).
///
/// # Safety
/// Same contract as `tensor::matmul_q1_0_sdot_rows` over `[0, n_rows)`.
pub unsafe fn matmul_q1_0_sdot(w: *const u8, xq: *const i8, xs: *const f32, y: *mut f32, m_count: usize, n_rows: usize, n_cols: usize) {
    let workers = fleet_workers();
    if workers == 0 || n_rows < PARALLEL_MIN_ROWS {
        // SAFETY: caller's contract; whole range on this core.
        unsafe { tensor::matmul_q1_0_sdot_rows(w, xq, xs, y, m_count, n_rows, 0, n_rows, n_cols) };
        return;
    }
    let n_parts = workers + 1;
    let b0 = row_boundary(n_rows, n_parts, 1);
    JOB.w.store(w as *mut u8, Ordering::Relaxed);
    JOB.xq.store(xq as *mut i8, Ordering::Relaxed);
    JOB.xs.store(xs as *mut f32, Ordering::Relaxed);
    JOB.y.store(y, Ordering::Relaxed);
    JOB.n_cols.store(n_cols, Ordering::Relaxed);
    JOB.m_count.store(m_count, Ordering::Relaxed);
    JOB.n_rows.store(n_rows, Ordering::Relaxed);
    JOB.mode.store(7, Ordering::Relaxed); // Q1_0 SDOT matmul
    for s in 0..workers {
        JOB.row_start[s].store(row_boundary(n_rows, n_parts, s + 1), Ordering::Relaxed);
        JOB.row_end[s].store(row_boundary(n_rows, n_parts, s + 2), Ordering::Relaxed);
    }
    clear_unused_slots(workers);
    let g = JOB.go.load(Ordering::Relaxed) + 1;
    JOB.go.store(g, Ordering::Release);
    signal_workers();
    let t0 = super::cycle_count();
    // SAFETY: disjoint row range [0, b0); caller's contract.
    unsafe { tensor::matmul_q1_0_sdot_rows(w, xq, xs, y, m_count, n_rows, 0, b0, n_cols) };
    add_busy(0, super::cycle_count().wrapping_sub(t0));
    barrier(workers, g, &|rs, re| unsafe {
        tensor::matmul_q1_0_sdot_rows(w, xq, xs, y, m_count, n_rows, rs, re, n_cols)
    });
}

/// Batched weight-stationary `Q1_0` matmul via **i8mm** (`vmmlaq_s32`) split
/// across online cores — the fast prefill path when the CPU has FEAT_I8MM.
/// Same shape as [`matmul_q1_0_sdot`] but mode 9 / `matmul_q1_0_i8mm_rows`.
///
/// # Safety
/// The CPU must implement FEAT_I8MM (caller gates on `super::has_i8mm()`);
/// otherwise same contract as `tensor::matmul_q1_0_i8mm_rows` over `[0,n_rows)`.
pub unsafe fn matmul_q1_0_i8mm(w: *const u8, xq: *const i8, xs: *const f32, y: *mut f32, m_count: usize, n_rows: usize, n_cols: usize) {
    let workers = fleet_workers();
    if workers == 0 || n_rows < PARALLEL_MIN_ROWS {
        // SAFETY: caller guarantees FEAT_I8MM; whole range on this core.
        unsafe { tensor::matmul_q1_0_i8mm_rows(w, xq, xs, y, m_count, n_rows, 0, n_rows, n_cols) };
        return;
    }
    let n_parts = workers + 1;
    let b0 = row_boundary(n_rows, n_parts, 1);
    JOB.w.store(w as *mut u8, Ordering::Relaxed);
    JOB.xq.store(xq as *mut i8, Ordering::Relaxed);
    JOB.xs.store(xs as *mut f32, Ordering::Relaxed);
    JOB.y.store(y, Ordering::Relaxed);
    JOB.n_cols.store(n_cols, Ordering::Relaxed);
    JOB.m_count.store(m_count, Ordering::Relaxed);
    JOB.n_rows.store(n_rows, Ordering::Relaxed);
    JOB.mode.store(9, Ordering::Relaxed); // Q1_0 i8mm matmul
    for s in 0..workers {
        JOB.row_start[s].store(row_boundary(n_rows, n_parts, s + 1), Ordering::Relaxed);
        JOB.row_end[s].store(row_boundary(n_rows, n_parts, s + 2), Ordering::Relaxed);
    }
    clear_unused_slots(workers);
    let g = JOB.go.load(Ordering::Relaxed) + 1;
    JOB.go.store(g, Ordering::Release);
    signal_workers();
    let t0 = super::cycle_count();
    // SAFETY: FEAT_I8MM per caller; disjoint row range [0, b0).
    unsafe { tensor::matmul_q1_0_i8mm_rows(w, xq, xs, y, m_count, n_rows, 0, b0, n_cols) };
    add_busy(0, super::cycle_count().wrapping_sub(t0));
    barrier(workers, g, &|rs, re| unsafe {
        tensor::matmul_q1_0_i8mm_rows(w, xq, xs, y, m_count, n_rows, rs, re, n_cols)
    });
}

/// Batched weight-stationary `Q8_0` matmul via **i8mm** (`vmmlaq_s32`) split
/// across online cores — the fast prefill path for Q8_0 (the 0.8B) when the CPU
/// has FEAT_I8MM. Same shape as [`matmul_sdot`] but mode 10 / the i8mm kernel.
///
/// # Safety
/// The CPU must implement FEAT_I8MM (caller gates on `super::has_i8mm()`);
/// otherwise same contract as `tensor::matmul_q8_0_i8mm_rows` over `[0,n_rows)`.
pub unsafe fn matmul_q8_0_i8mm(w: *const u8, xq: *const i8, xs: *const f32, y: *mut f32, m_count: usize, n_rows: usize, n_cols: usize) {
    let workers = fleet_workers();
    if workers == 0 || n_rows < PARALLEL_MIN_ROWS {
        // SAFETY: caller guarantees FEAT_I8MM; whole range on this core.
        unsafe { tensor::matmul_q8_0_i8mm_rows(w, xq, xs, y, m_count, n_rows, 0, n_rows, n_cols) };
        return;
    }
    let n_parts = workers + 1;
    let b0 = row_boundary(n_rows, n_parts, 1);
    JOB.w.store(w as *mut u8, Ordering::Relaxed);
    JOB.xq.store(xq as *mut i8, Ordering::Relaxed);
    JOB.xs.store(xs as *mut f32, Ordering::Relaxed);
    JOB.y.store(y, Ordering::Relaxed);
    JOB.n_cols.store(n_cols, Ordering::Relaxed);
    JOB.m_count.store(m_count, Ordering::Relaxed);
    JOB.n_rows.store(n_rows, Ordering::Relaxed);
    JOB.mode.store(10, Ordering::Relaxed); // Q8_0 i8mm matmul
    for s in 0..workers {
        JOB.row_start[s].store(row_boundary(n_rows, n_parts, s + 1), Ordering::Relaxed);
        JOB.row_end[s].store(row_boundary(n_rows, n_parts, s + 2), Ordering::Relaxed);
    }
    clear_unused_slots(workers);
    let g = JOB.go.load(Ordering::Relaxed) + 1;
    JOB.go.store(g, Ordering::Release);
    signal_workers();
    let t0 = super::cycle_count();
    // SAFETY: FEAT_I8MM per caller; disjoint row range [0, b0).
    unsafe { tensor::matmul_q8_0_i8mm_rows(w, xq, xs, y, m_count, n_rows, 0, b0, n_cols) };
    add_busy(0, super::cycle_count().wrapping_sub(t0));
    barrier(workers, g, &|rs, re| unsafe {
        tensor::matmul_q8_0_i8mm_rows(w, xq, xs, y, m_count, n_rows, rs, re, n_cols)
    });
}

/// Batched weight-stationary `Q4_0` matmul via **i8mm** split across online
/// cores — batched prefill for the Q4_0 models (2B/4B) when the CPU has
/// FEAT_I8MM. Mode 11 / `matmul_q4_0_i8mm_rows`.
///
/// # Safety
/// The CPU must implement FEAT_I8MM (caller gates on `super::has_i8mm()`);
/// otherwise same contract as `tensor::matmul_q4_0_i8mm_rows` over `[0,n_rows)`.
pub unsafe fn matmul_q4_0_i8mm(w: *const u8, xq: *const i8, xs: *const f32, y: *mut f32, m_count: usize, n_rows: usize, n_cols: usize) {
    let workers = fleet_workers();
    if workers == 0 || n_rows < PARALLEL_MIN_ROWS {
        // SAFETY: caller guarantees FEAT_I8MM; whole range on this core.
        unsafe { tensor::matmul_q4_0_i8mm_rows(w, xq, xs, y, m_count, n_rows, 0, n_rows, n_cols) };
        return;
    }
    let n_parts = workers + 1;
    let b0 = row_boundary(n_rows, n_parts, 1);
    JOB.w.store(w as *mut u8, Ordering::Relaxed);
    JOB.xq.store(xq as *mut i8, Ordering::Relaxed);
    JOB.xs.store(xs as *mut f32, Ordering::Relaxed);
    JOB.y.store(y, Ordering::Relaxed);
    JOB.n_cols.store(n_cols, Ordering::Relaxed);
    JOB.m_count.store(m_count, Ordering::Relaxed);
    JOB.n_rows.store(n_rows, Ordering::Relaxed);
    JOB.mode.store(11, Ordering::Relaxed); // Q4_0 i8mm matmul
    for s in 0..workers {
        JOB.row_start[s].store(row_boundary(n_rows, n_parts, s + 1), Ordering::Relaxed);
        JOB.row_end[s].store(row_boundary(n_rows, n_parts, s + 2), Ordering::Relaxed);
    }
    clear_unused_slots(workers);
    let g = JOB.go.load(Ordering::Relaxed) + 1;
    JOB.go.store(g, Ordering::Release);
    signal_workers();
    let t0 = super::cycle_count();
    // SAFETY: FEAT_I8MM per caller; disjoint row range [0, b0).
    unsafe { tensor::matmul_q4_0_i8mm_rows(w, xq, xs, y, m_count, n_rows, 0, b0, n_cols) };
    add_busy(0, super::cycle_count().wrapping_sub(t0));
    barrier(workers, g, &|rs, re| unsafe {
        tensor::matmul_q4_0_i8mm_rows(w, xq, xs, y, m_count, n_rows, rs, re, n_cols)
    });
}

/// Batched weight-stationary `Q2_0` matmul split across online cores by
/// contiguous row range — the batched-prefill kernel for Ternary-Bonsai
/// (`matmul_q2_0_sdot_rows`).
///
/// # Safety
/// Same contract as `tensor::matmul_q2_0_sdot_rows` over `[0, n_rows)`.
pub unsafe fn matmul_q2_0_sdot(w: *const u8, xq: *const i8, xs: *const f32, y: *mut f32, m_count: usize, n_rows: usize, n_cols: usize) {
    let workers = fleet_workers();
    if workers == 0 || n_rows < PARALLEL_MIN_ROWS {
        // SAFETY: caller's contract; whole range on this core.
        unsafe { tensor::matmul_q2_0_sdot_rows(w, xq, xs, y, m_count, n_rows, 0, n_rows, n_cols) };
        return;
    }
    let n_parts = workers + 1;
    let b0 = row_boundary(n_rows, n_parts, 1);
    JOB.w.store(w as *mut u8, Ordering::Relaxed);
    JOB.xq.store(xq as *mut i8, Ordering::Relaxed);
    JOB.xs.store(xs as *mut f32, Ordering::Relaxed);
    JOB.y.store(y, Ordering::Relaxed);
    JOB.n_cols.store(n_cols, Ordering::Relaxed);
    JOB.m_count.store(m_count, Ordering::Relaxed);
    JOB.n_rows.store(n_rows, Ordering::Relaxed);
    JOB.mode.store(8, Ordering::Relaxed); // Q2_0 SDOT matmul
    for s in 0..workers {
        JOB.row_start[s].store(row_boundary(n_rows, n_parts, s + 1), Ordering::Relaxed);
        JOB.row_end[s].store(row_boundary(n_rows, n_parts, s + 2), Ordering::Relaxed);
    }
    clear_unused_slots(workers);
    let g = JOB.go.load(Ordering::Relaxed) + 1;
    JOB.go.store(g, Ordering::Release);
    signal_workers();
    let t0 = super::cycle_count();
    // SAFETY: disjoint row range [0, b0); caller's contract.
    unsafe { tensor::matmul_q2_0_sdot_rows(w, xq, xs, y, m_count, n_rows, 0, b0, n_cols) };
    add_busy(0, super::cycle_count().wrapping_sub(t0));
    barrier(workers, g, &|rs, re| unsafe {
        tensor::matmul_q2_0_sdot_rows(w, xq, xs, y, m_count, n_rows, rs, re, n_cols)
    });
}
