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
//! APs do a fixed chunk of work and then `hlt` forever (interrupts stay
//! disabled, as Limine started them), so they never busy-spin stealing vCPU
//! time from the BSP -- important under QEMU's cross-arch TCG. This establishes
//! the substrate: multiple cores executing kernel code concurrently under
//! correct locks.
//!
//! Data-parallel work (splitting the Cortex `Q8_0` matvec's rows across cores)
//! is a real speedup on native multi-core x86, but measured a net *loss* under
//! QEMU TCG -- `thread=multi` taxes every emulated instruction and idle worker
//! cores contend for host CPU -- so inference is kept single-core here and the
//! APs park. The row-range kernel (`tensor::matvec_q8_0_rows`) keeps that split
//! a drop-in away for a real-hardware target. Preemptive per-core run queues +
//! IPIs remain future work.

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

    // Parked: halt forever with interrupts disabled. The core is done; it
    // consumes no vCPU time (unlike a busy spin) until the next reset.
    loop {
        crate::arch::x86_64::hlt();
    }
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
