//! Symmetric multiprocessing bring-up (`CHITTI_OS_HANDOFF.md` Phase 7).
//!
//! Limine discovers the application processors (APs) via ACPI/MADT and parks
//! each one spinning on a per-CPU `goto_address`; [`init`] writes that field
//! to launch each AP into [`ap_entry`]. Each core then sets up its own GDT/TSS
//! (`gdt::init_ap`), loads the shared IDT (`idt::load_ap`), software-enables
//! its local APIC (`apic`), runs a bounded self-test, then parks.
//!
//! The self-test is also the **lock-discipline** proof the phase calls for:
//! every online core (BSP + APs) hammers one shared counter through the
//! kernel's `Locked` spinlock. If the spinlock provides real cross-core
//! mutual exclusion, the counter lands on exactly `cpus * WORK_PER_CPU` with
//! no lost updates; a broken lock would drop increments under contention.
//! Per-core counters prove the work genuinely ran on more than one core.
//!
//! After the self-test each AP joins the **compute fleet** rather than halting
//! forever: it claims a dense worker index and parks in `hlt` with interrupts
//! enabled, waiting for a wake IPI. Idle workers therefore cost nothing (no busy
//! spin stealing vCPU time from the BSP, which matters under QEMU TCG), but are
//! available the moment there is data-parallel work.
//!
//! [`parallel_for`] is that work: a static-partition job barrier, matching
//! `arch::aarch64::smp::parallel_for` in both signature and semantics, reached
//! through the arch-neutral `arch::parallel_for`. It backs the ONNX hot ops, the
//! video row conversion, and the Cortex matvec row split — all of which
//! previously ran on a single core here while aarch64 used the whole machine.
//!
//! Two safety properties are load-bearing, both learned on aarch64: the barrier
//! is **bounded** and distinguishes a worker that never woke (whose range the BSP
//! then runs itself) from one that is merely slow (which must never be
//! duplicated); and a boot-time [`wake_self_test`] proves the IPI wake actually
//! works, disabling the fleet if it does not. Preemptive per-core run queues
//! remain future work — this is fan-out, not scheduling.

use crate::arch::x86_64::{apic, gdt, idt};
use crate::limine_protocol::SmpInfo;
use crate::mm::Locked;
use core::sync::atomic::{AtomicU64, Ordering};

/// Upper bound on cores we track per-core state for.
pub const MAX_CPUS: usize = 64;

/// How many locked increments each core contributes to the shared counter.
const WORK_PER_CPU: u64 = 5000;

/// Cores that have come online (starts at 1 for the BSP; each AP adds itself).
static CPUS_ONLINE: AtomicU64 = AtomicU64::new(1);
/// Total cores Limine reported (1 until `init` learns otherwise).
static EXPECTED_CPUS: AtomicU64 = AtomicU64::new(1);
/// APs that have finished their self-test work chunk.
static WORKERS_DONE: AtomicU64 = AtomicU64::new(0);

/// The shared counter every core increments through the spinlock -- the
/// lock-correctness target.
static SHARED_COUNTER: Locked<u64> = Locked::new(0);

const ZERO: AtomicU64 = AtomicU64::new(0);
/// Per-core work counters (indexed by the slot assigned at launch), so the
/// self-test can show work happened on more than one core.
static PERCPU_WORK: [AtomicU64; MAX_CPUS] = [ZERO; MAX_CPUS];
/// Per-core local-APIC id + 1 (0 = slot unused), recorded as each core comes
/// online so the BSP can print one tidy summary instead of the cores' log
/// lines interleaving on the shared serial port.
static PERCPU_LAPIC: [AtomicU64; MAX_CPUS] = [ZERO; MAX_CPUS];

/// Snapshot of the SMP self-test, read by the acceptance test.
pub struct SmpStats {
    pub cpus_online: u64,
    pub expected_cpus: u64,
    pub shared_counter: u64,
    pub expected_counter: u64,
    pub cpus_that_worked: usize,
}

/// Bring up the APs, run the self-test, and park them. Called once on the BSP
/// from `chitti_kernel::init`, after `mm::init`. A no-op beyond the BSP's own
/// chunk when Limine reports a single CPU.
pub fn init() {
    // Map the local APIC MMIO page (the HHDM doesn't cover it) into the shared
    // page tables, then software-enable the BSP's own local APIC. APs reuse
    // this mapping and enable their own APICs as they come online.
    apic::init_mapping();
    apic::software_enable();

    let Some(resp) = crate::SMP_REQUEST.response() else {
        crate::ktrace::log("smp", "no MP response from Limine; running single-core");
        EXPECTED_CPUS.store(1, Ordering::SeqCst);
        do_work_chunk(0);
        return;
    };

    let cpus = resp.cpus();
    let bsp = resp.bsp_lapic_id();
    let n = cpus.len().min(MAX_CPUS) as u64;
    EXPECTED_CPUS.store(n, Ordering::SeqCst);
    PERCPU_LAPIC[0].store(bsp as u64 + 1, Ordering::SeqCst); // BSP occupies slot 0
    crate::ktrace::log_fmt(format_args!(
        "smp: Limine reports {} cpu(s); BSP lapic {} apic_id {}",
        cpus.len(),
        bsp,
        apic::local_id()
    ));

    // Launch each AP by handing it a slot and writing its goto_address.
    let ap_fn: extern "C" fn(*const SmpInfo) -> ! = ap_entry;
    let mut slot: u64 = 1; // slot 0 is the BSP
    for info in cpus {
        if info.lapic_id == bsp || slot as usize >= MAX_CPUS {
            continue;
        }
        info.extra_argument.store(slot, Ordering::SeqCst);
        slot += 1;
        // The AP is spinning on goto_address; this store launches it.
        info.goto_address.store(ap_fn as u64, Ordering::SeqCst);
    }

    // The BSP contributes its own chunk (slot 0), concurrently with the APs.
    do_work_chunk(0);

    // Wait for every AP to finish its chunk (bounded, so a wedged AP can't
    // hang the whole boot forever).
    let expected_aps = n.saturating_sub(1);
    let mut spins: u64 = 0;
    while WORKERS_DONE.load(Ordering::Acquire) < expected_aps {
        core::hint::spin_loop();
        spins += 1;
        if spins > 5_000_000_000 {
            crate::ktrace::log("smp", "timed out waiting for APs to finish self-test");
            break;
        }
    }

    // Now that the APs have all reported in, print one tidy per-core summary.
    for slot in 0..n as usize {
        let lapic = PERCPU_LAPIC[slot].load(Ordering::SeqCst);
        if lapic != 0 {
            crate::ktrace::log_fmt(format_args!(
                "smp:   cpu slot {slot}: lapic {}{}",
                lapic - 1,
                if slot == 0 { " (BSP)" } else { " (AP, online + self-test done)" }
            ));
        }
    }

    let total = SHARED_COUNTER.with(|c| *c);
    crate::ktrace::log_fmt(format_args!(
        "smp: {} cpu(s) online; shared counter = {} (expected {}) -- spinlock {}",
        CPUS_ONLINE.load(Ordering::SeqCst),
        total,
        n * WORK_PER_CPU,
        if total == n * WORK_PER_CPU { "OK" } else { "LOST UPDATES" }
    ));

    // Prove the compute fleet's wake path before anything relies on it.
    wake_self_test();
}

/// AP entry point. Limine jumps here (with `rdi` = this core's `SmpInfo`) on a
/// bootloader-provided stack. Sets up this core, runs its work chunk, then
/// parks forever.
extern "C" fn ap_entry(info: *const SmpInfo) -> ! {
    // Must be first: SSE codegen is on crate-wide, and Limine does not
    // guarantee SSE is enabled on an AP (same reason `_start` does this).
    crate::arch::x86_64::fpu::enable_sse();

    gdt::init_ap();
    idt::load_ap();
    apic::software_enable();

    // SAFETY: `info` is this core's valid `SmpInfo` (Limine passed it in rdi);
    // `extra_argument`/`lapic_id` were set before we were launched.
    let slot = unsafe { (*info).extra_argument.load(Ordering::SeqCst) as usize };
    let lapic = unsafe { (*info).lapic_id };

    // Record identity for the BSP's summary (avoid logging here: many cores
    // hit the shared serial port at once, garbling each other's lines).
    if slot < MAX_CPUS {
        PERCPU_LAPIC[slot].store(lapic as u64 + 1, Ordering::SeqCst);
    }
    CPUS_ONLINE.fetch_add(1, Ordering::SeqCst);

    do_work_chunk(slot);
    WORKERS_DONE.fetch_add(1, Ordering::SeqCst);

    // Join the compute fleet rather than halting forever. Claim a dense worker
    // index (independent of the sparse Limine slot, so `parallel_for` can address
    // workers 0..WORKERS without gaps) and park waiting for work.
    let worker = WORKERS.fetch_add(1, Ordering::AcqRel) as usize;
    if worker >= MAX_CPUS - 1 {
        // More cores than the job tables address: park this one for good rather
        // than index out of range.
        loop {
            crate::arch::x86_64::hlt();
        }
    }
    fleet_worker(worker)
}

/// One core's contribution to the shared-counter self-test: `WORK_PER_CPU`
/// increments, each taking the spinlock, plus a per-core tally.
fn do_work_chunk(slot: usize) {
    for _ in 0..WORK_PER_CPU {
        SHARED_COUNTER.with(|c| *c += 1);
    }
    if slot < MAX_CPUS {
        PERCPU_WORK[slot].store(WORK_PER_CPU, Ordering::SeqCst);
    }
}

/// Total CPUs Limine reported.
pub fn cpu_count() -> u64 {
    EXPECTED_CPUS.load(Ordering::SeqCst)
}

/// Result of the boot-time SMP self-test.
pub fn stats() -> SmpStats {
    let expected = EXPECTED_CPUS.load(Ordering::SeqCst);
    SmpStats {
        cpus_online: CPUS_ONLINE.load(Ordering::SeqCst),
        expected_cpus: expected,
        shared_counter: SHARED_COUNTER.with(|c| *c),
        expected_counter: expected * WORK_PER_CPU,
        cpus_that_worked: PERCPU_WORK.iter().filter(|w| w.load(Ordering::SeqCst) > 0).count(),
    }
}

// --- compute fleet: parallel_for across the application processors ---------
//
// Until now the APs booted, ran a self-test and halted forever, so every
// compute-heavy loop in the kernel ran on one core on x86 while the identical
// workload was split across all cores on aarch64. That is the dual-architecture
// rule's "capability exists on one arch but not the other" case, and on an 8-core
// laptop it means inference used an eighth of the machine.
//
// Design mirrors `arch::aarch64::smp`: a static-partition job barrier, not a
// scheduler. The BSP publishes a row range per worker, bumps a generation
// counter, wakes the fleet, computes its own share, then waits.
//
// Wake mechanism differs by necessity. aarch64 parks workers in `WFE` and relies
// on the event stream; x86 has no equivalent, so workers sit in `hlt` with
// interrupts enabled and the BSP pokes them with an IPI. A `pause` spin would
// also work and need no interrupt plumbing, but it costs a core of power per idle
// AP — unacceptable on a laptop.

use core::sync::atomic::{AtomicPtr, AtomicUsize};

/// Vector the wake IPI is delivered on. Above the PIC's 32..47 range and the
/// APIC timer's 0x40.
const WAKE_VECTOR: u8 = 0x41;

/// A unit of fan-out work: `f(start, end, ctx)` over a disjoint index range.
struct Job {
    /// Generation counter; incremented to publish new work.
    go: AtomicU64,
    fn_ptr: AtomicUsize,
    ctx: AtomicPtr<u8>,
    start: [AtomicUsize; MAX_CPUS],
    end: [AtomicUsize; MAX_CPUS],
    /// Generation this worker has *begun* — set before it touches the range, so
    /// the BSP can tell "never started" (safe to take over) from "still running"
    /// (must not duplicate).
    claimed: [AtomicU64; MAX_CPUS],
    /// Generation this worker has finished.
    done: [AtomicU64; MAX_CPUS],
}

const AZERO: AtomicUsize = AtomicUsize::new(0);
static JOB: Job = Job {
    go: AtomicU64::new(0),
    fn_ptr: AtomicUsize::new(0),
    ctx: AtomicPtr::new(core::ptr::null_mut()),
    start: [AZERO; MAX_CPUS],
    end: [AZERO; MAX_CPUS],
    claimed: [ZERO; MAX_CPUS],
    done: [ZERO; MAX_CPUS],
};

/// Number of APs that have registered as fleet workers.
static WORKERS: AtomicU64 = AtomicU64::new(0);

/// Workers available for fan-out (online APs). The BSP is always an extra
/// participant on top of this.
pub fn online_cpus() -> usize {
    WORKERS.load(Ordering::Acquire) as usize + 1
}

/// The wake-IPI handler: nothing to do but acknowledge. Its only purpose is to
/// bring the core out of `hlt` so the park loop re-checks the generation.
extern "x86-interrupt" fn wake_handler(_frame: crate::arch::x86_64::idt::InterruptStackFrame) {
    crate::arch::x86_64::apic::eoi();
}

/// Park this AP as a fleet worker: wait for a job generation, run this worker's
/// range, mark it done, sleep again.
///
/// Never returns. Interrupts are enabled here (they are not during bring-up) so
/// the wake IPI can land; the handler does nothing but EOI.
fn fleet_worker(worker: usize) -> ! {
    crate::arch::x86_64::idt::set_irq_handler(WAKE_VECTOR, wake_handler);
    crate::arch::x86_64::interrupts::enable();
    let mut last = JOB.go.load(Ordering::Acquire);
    loop {
        let g = JOB.go.load(Ordering::Acquire);
        if g != last {
            last = g;
            let f = JOB.fn_ptr.load(Ordering::Acquire);
            let s = JOB.start[worker].load(Ordering::Relaxed);
            let e = JOB.end[worker].load(Ordering::Relaxed);
            if f != 0 && e > s {
                // Publish "started" before doing any work, so a straggler check
                // cannot conclude this range is untouched and run it as well.
                JOB.claimed[worker].store(g, Ordering::Release);
                // SAFETY: the BSP published `f` as a `unsafe fn(usize, usize,
                // *mut u8)` together with a range disjoint from every other
                // worker's, and keeps `ctx` alive across the barrier.
                unsafe {
                    let f: unsafe fn(usize, usize, *mut u8) = core::mem::transmute(f);
                    f(s, e, JOB.ctx.load(Ordering::Acquire));
                }
            } else {
                JOB.claimed[worker].store(g, Ordering::Release);
            }
            JOB.done[worker].store(g, Ordering::Release);
        }
        // Sleep until the next IPI. `hlt` with interrupts enabled costs nothing
        // while idle, unlike a spin.
        crate::arch::x86_64::hlt();
    }
}

/// How long the BSP waits for the fleet before treating a worker as a straggler.
/// Generous: this is a bound against broken wake delivery, not a scheduling
/// deadline.
const BARRIER_SPINS: u64 = 200_000_000;

/// Run `f` over `[0, n)` split across the BSP and every online AP.
///
/// Falls back to running inline on the calling core when there are no workers or
/// the range is too small to be worth splitting — the same shape, and the same
/// signature, as `arch::aarch64::smp::parallel_for`, so callers need no `cfg`.
///
/// # Safety
/// `f` must be safe to call concurrently on disjoint sub-ranges of `[0, n)`
/// sharing `ctx`; `ctx` must outlive the call.
pub unsafe fn parallel_for(n: usize, min_chunk: usize, f: unsafe fn(usize, usize, *mut u8), ctx: *mut u8) {
    let workers = (WORKERS.load(Ordering::Acquire) as usize).min(MAX_CPUS - 1);
    if workers == 0 || n < min_chunk.saturating_mul(2).max(16) {
        // SAFETY: caller's contract, over the whole range on this core.
        unsafe { f(0, n, ctx) };
        return;
    }
    let n_parts = workers + 1; // + the BSP
    let boundary = |k: usize| k * n / n_parts;
    JOB.fn_ptr.store(f as usize, Ordering::Relaxed);
    JOB.ctx.store(ctx, Ordering::Relaxed);
    for w in 0..workers {
        JOB.start[w].store(boundary(w + 1), Ordering::Relaxed);
        JOB.end[w].store(boundary(w + 2), Ordering::Relaxed);
    }
    let g = JOB.go.load(Ordering::Relaxed) + 1;
    JOB.go.store(g, Ordering::Release);
    crate::arch::x86_64::apic::send_wake_ipi(WAKE_VECTOR);

    // The BSP takes the first partition while the fleet works.
    // SAFETY: caller's contract, over a range disjoint from every worker's.
    unsafe { f(0, boundary(1), ctx) };

    // Bounded barrier. A worker that never *claimed* its range cannot have
    // touched it, so the BSP runs it rather than returning a partial result — the
    // failure mode to avoid is silently-wrong output, and slow beats stuck.
    for w in 0..workers {
        let mut spins = 0u64;
        while JOB.done[w].load(Ordering::Acquire) != g {
            if spins > BARRIER_SPINS {
                if JOB.claimed[w].load(Ordering::Acquire) != g {
                    let (s, e) = (JOB.start[w].load(Ordering::Relaxed), JOB.end[w].load(Ordering::Relaxed));
                    crate::ktrace::log_fmt(format_args!(
                        "smp: worker {w} never woke for generation {g}; BSP running rows {s}..{e}"
                    ));
                    // SAFETY: `claimed != g` proves the worker never entered this
                    // range, so running it here cannot race.
                    unsafe { f(s, e, ctx) };
                    JOB.done[w].store(g, Ordering::Release);
                } else {
                    // Claimed but unfinished: it IS running, so duplicating would
                    // corrupt the output. Keep waiting, but say so.
                    crate::ktrace::log_fmt(format_args!(
                        "smp: worker {w} claimed generation {g} but has not finished -- still waiting"
                    ));
                    spins = 0;
                    continue;
                }
                break;
            }
            spins += 1;
            core::hint::spin_loop();
        }
    }
}

/// Scratch buffer for the wake self-test: one slot per index, each written by
/// whichever core owns that index.
static SELFTEST_MARKS: [AtomicU64; SELFTEST_N] = [ZERO; SELFTEST_N];
const SELFTEST_N: usize = 4096;

/// Self-test body: stamp `gen` into every index of this core's range.
///
/// # Safety
/// Ranges handed to it are disjoint, so the writes never overlap.
unsafe fn selftest_mark(start: usize, end: usize, ctx: *mut u8) {
    let gen = ctx as u64;
    for i in start..end.min(SELFTEST_N) {
        SELFTEST_MARKS[i].store(gen, Ordering::Relaxed);
    }
}

/// Push one real job through the fleet to prove the wake path works, and fall
/// back to single-core if it does not.
///
/// This mirrors aarch64's `wake self-test`, and exists for the same hard-won
/// reason: a hypervisor can accept the parked state and never deliver the wake.
/// (On aarch64 that was VirtualBox holding a trapped `WFE` until an interrupt,
/// which hung the first prefill forever.) x86 wakes with an IPI, which a
/// hypervisor could equally drop — so verify it once, at boot, on the real job
/// and barrier path rather than trusting it.
///
/// A failure is not fatal: `WORKERS` is zeroed, so `parallel_for` runs everything
/// inline from then on. Slow beats stuck, and slow beats silently wrong.
pub fn wake_self_test() {
    let workers = WORKERS.load(Ordering::Acquire) as usize;
    if workers == 0 {
        return; // single-core boot; nothing to prove
    }
    for m in SELFTEST_MARKS.iter() {
        m.store(0, Ordering::Relaxed);
    }
    let gen = 0xa5u64;
    // SAFETY: `selftest_mark` only writes its own disjoint range of a static
    // array, and `ctx` is a plain integer, not a pointer that must outlive.
    unsafe { parallel_for(SELFTEST_N, 1, selftest_mark, gen as *mut u8) };

    // Did every worker actually run its own range, or did the BSP have to cover?
    let g = JOB.go.load(Ordering::Acquire);
    let unwoken = (0..workers).filter(|&w| JOB.claimed[w].load(Ordering::Acquire) != g).count();
    // And is the output complete regardless of who computed it?
    let unmarked = SELFTEST_MARKS.iter().filter(|m| m.load(Ordering::Relaxed) != gen).count();

    if unmarked != 0 {
        crate::ktrace::log_fmt(format_args!(
            "smp: wake self-test FAILED -- {unmarked}/{SELFTEST_N} indices unwritten; disabling the compute fleet"
        ));
        WORKERS.store(0, Ordering::Release);
        return;
    }
    if unwoken != 0 {
        crate::ktrace::log_fmt(format_args!(
            "smp: wake self-test degraded -- {unwoken}/{workers} worker(s) never woke (IPI not delivered); disabling the compute fleet"
        ));
        WORKERS.store(0, Ordering::Release);
        return;
    }
    crate::ktrace::log_fmt(format_args!("smp: wake self-test ok ({workers} compute worker(s))"));
}

#[cfg(test)]
mod tests {
    use super::*;

    const TN: usize = 4096;
    static TEST_OUT: [AtomicU64; TN] = [ZERO; TN];

    /// Write an index-derived value, so a wrong split shows up as wrong *data*,
    /// not merely a missing write.
    ///
    /// # Safety
    /// Ranges are disjoint, so no two calls touch the same slot.
    unsafe fn fill(start: usize, end: usize, _ctx: *mut u8) {
        for i in start..end.min(TN) {
            TEST_OUT[i].store((i as u64) * 3 + 1, Ordering::Relaxed);
        }
    }

    #[test_case]
    fn parallel_for_matches_the_serial_result() {
        // The property the fan-out must have: the output is identical however the
        // range is partitioned. A dropped or overlapping partition shows up here.
        for slot in TEST_OUT.iter() {
            slot.store(u64::MAX, Ordering::Relaxed);
        }
        // SAFETY: `fill` only writes its own disjoint range of a static array.
        unsafe { parallel_for(TN, 32, fill, core::ptr::null_mut()) };
        for i in 0..TN {
            assert_eq!(
                TEST_OUT[i].load(Ordering::Relaxed),
                (i as u64) * 3 + 1,
                "index {i} wrong after parallel_for -- partition dropped or overlapped"
            );
        }
    }

    #[test_case]
    fn parallel_for_runs_small_ranges_inline_and_still_covers_them() {
        // Below ~2*min_chunk the whole range runs on the calling core. It must
        // still be fully computed — an early return here would silently produce
        // zeros for small matrices.
        for slot in TEST_OUT.iter().take(8) {
            slot.store(u64::MAX, Ordering::Relaxed);
        }
        // SAFETY: as above.
        unsafe { parallel_for(8, 32, fill, core::ptr::null_mut()) };
        for i in 0..8 {
            assert_eq!(TEST_OUT[i].load(Ordering::Relaxed), (i as u64) * 3 + 1);
        }
    }

    #[test_case]
    fn parallel_for_handles_a_zero_length_range() {
        // n == 0 must be a no-op, not a division by n_parts producing an empty
        // barrier the BSP waits on forever.
        // SAFETY: as above.
        unsafe { parallel_for(0, 32, fill, core::ptr::null_mut()) };
    }
}
