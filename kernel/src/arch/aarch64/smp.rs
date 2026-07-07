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

use crate::cortex::tensor;
use core::arch::global_asm;
use core::sync::atomic::{AtomicPtr, AtomicU64, AtomicUsize, Ordering};

/// Max cores we size the static tables for (QEMU `-smp` is 4 here; the M2 has
/// 8). Only cores actually brought online are used.
const MAX_CPUS: usize = 8;
/// Per-secondary stack. The worker loop is shallow (it just calls the matvec
/// kernel), so 64 KiB is ample.
const AP_STACK_SIZE: usize = 64 * 1024;
/// Below this row count a matvec isn't worth the cross-core sync; the BSP does
/// it alone.
const PARALLEL_MIN_ROWS: usize = 256;

#[repr(C, align(16))]
struct ApStack([u8; AP_STACK_SIZE]);
/// One private stack per potential secondary core (BSS; never zeroed-critical).
static mut AP_STACKS: [ApStack; MAX_CPUS] = [const { ApStack([0; AP_STACK_SIZE]) }; MAX_CPUS];

/// Number of secondaries that have come online and claimed a worker slot.
static N_WORKERS: AtomicUsize = AtomicUsize::new(0);

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
    // (using `qtype` as the dequant type), 3 = Q4_K SDOT (xq/xs). Keeping this
    // separate from `qtype` avoids the collision where a quant type could mean
    // either the SDOT or the generic path.
    mode: AtomicUsize,
    qtype: AtomicUsize,
    xf: AtomicPtr<f32>,
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
    row_start: [const { AtomicUsize::new(0) }; MAX_CPUS],
    row_end: [const { AtomicUsize::new(0) }; MAX_CPUS],
    go: AtomicU64::new(0),
};
/// Per-worker-slot "generation last completed", so the BSP can barrier on it.
static DONE: [AtomicU64; MAX_CPUS] = [const { AtomicU64::new(0) }; MAX_CPUS];

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

/// A worker core: **park on `WFE`** until the BSP publishes a new job
/// generation, run this slot's row range through the SDOT matvec, mark the
/// generation done, repeat. `WFE` (not a busy `spin_loop`) is essential on
/// hypervisors: a busy-spinning idle secondary pegs a host core, and with many
/// vCPUs (e.g. an 8-CPU VirtualBox VM) the spinning secondaries starve the boot
/// core during time-sensitive work like USB xHCI enumeration — which then times
/// out, leaving no keyboard and (with the timer IRQ also cooperative) a frozen
/// console. Parked workers cost nothing until the BSP `signal_workers()`.
fn worker_loop(slot: usize) -> ! {
    let mut last = 0u64;
    loop {
        let g = JOB.go.load(Ordering::Acquire);
        if g == last {
            // SAFETY: `wfe` just parks until an event/IRQ; spurious wakes are
            // fine — we re-check `go` on the next iteration.
            unsafe { core::arch::asm!("wfe", options(nomem, nostack, preserves_flags)) };
            continue;
        }
        last = g;
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
    let workers = N_WORKERS.load(Ordering::Relaxed);
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
    let g = JOB.go.load(Ordering::Relaxed) + 1;
    JOB.go.store(g, Ordering::Release);
    signal_workers(); // wake WFE-parked workers

    // BSP computes chunk 0 while the workers run theirs.
    // SAFETY: disjoint row range [0, boundary(1)); caller's contract.
    let t0 = super::cycle_count();
    unsafe { tensor::matmul_q8_0_sdot_rows(w, xq, xs, y, m_count, n_rows, 0, boundary(1), n_cols) };
    add_busy(0, super::cycle_count().wrapping_sub(t0));

    // Barrier: wait for every worker to finish this generation.
    for s in 0..workers {
        while DONE[s].load(Ordering::Acquire) != g {
            core::hint::spin_loop();
        }
    }
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
    let workers = N_WORKERS.load(Ordering::Relaxed);
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
    let g = JOB.go.load(Ordering::Relaxed) + 1;
    JOB.go.store(g, Ordering::Release);
    signal_workers(); // wake WFE-parked workers
    // SAFETY: disjoint row range [0, boundary(1)); caller's contract.
    let t0 = super::cycle_count();
    unsafe { tensor::matvec_quant_rows(qt, w, x, y, 0, boundary(1), n_cols) };
    add_busy(0, super::cycle_count().wrapping_sub(t0));
    for s in 0..workers {
        while DONE[s].load(Ordering::Acquire) != g {
            core::hint::spin_loop();
        }
    }
}

/// Run a `Q4_0` SDOT matvec (int8-quantized activation `xq`/`xs`) across all
/// online cores by row range -- the fast Q4_0 path (`matvec_q4_0_sdot_rows`),
/// used for the 9B's many Q4_0 tensors. Falls back to single-core when only the
/// BSP is online or the matrix is small.
///
/// # Safety
/// Same contract as `tensor::matvec_q4_0_sdot_rows` over `[0, n_rows)`.
pub unsafe fn matvec_q4_0_sdot(w: *const u8, xq: *const i8, xs: *const f32, y: *mut f32, n_rows: usize, n_cols: usize) {
    let workers = N_WORKERS.load(Ordering::Relaxed);
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
    let g = JOB.go.load(Ordering::Relaxed) + 1;
    JOB.go.store(g, Ordering::Release);
    signal_workers(); // wake WFE-parked workers
    // SAFETY: disjoint row range [0, boundary(1)); caller's contract.
    let t0 = super::cycle_count();
    unsafe { tensor::matvec_q4_0_sdot_rows(w, xq, xs, y, 0, boundary(1), n_cols) };
    add_busy(0, super::cycle_count().wrapping_sub(t0));
    for s in 0..workers {
        while DONE[s].load(Ordering::Acquire) != g {
            core::hint::spin_loop();
        }
    }
}

/// Run a `Q4_K` SDOT matvec (int8-quantized activation `xq`/`xs`) across all
/// online cores by row range -- the fast path for Q4_K_M-dominant models
/// (Ornith, Qwen3.6, Gemma-4 quants). Falls back to single-core when only the
/// BSP is online or the matrix is small.
///
/// # Safety
/// Same contract as `tensor::matvec_q4_k_sdot_rows` over `[0, n_rows)`.
pub unsafe fn matvec_q4_k_sdot(w: *const u8, xq: *const i8, xs: *const f32, y: *mut f32, n_rows: usize, n_cols: usize) {
    let workers = N_WORKERS.load(Ordering::Relaxed);
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
    let g = JOB.go.load(Ordering::Relaxed) + 1;
    JOB.go.store(g, Ordering::Release);
    signal_workers(); // wake WFE-parked workers
    // SAFETY: disjoint row range [0, boundary(1)); caller's contract.
    let t0 = super::cycle_count();
    unsafe { tensor::matvec_q4_k_sdot_rows(w, xq, xs, y, 0, boundary(1), n_cols) };
    add_busy(0, super::cycle_count().wrapping_sub(t0));
    for s in 0..workers {
        while DONE[s].load(Ordering::Acquire) != g {
            core::hint::spin_loop();
        }
    }
}
