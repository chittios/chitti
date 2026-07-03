//! **GICv3 interrupt controller + generic-timer tick** — the aarch64 analogue
//! of the x86 PIC + PIT (`arch::x86_64::{pic,pit}`). It wires the EL1 physical
//! timer's private peripheral interrupt (PPI, INTID 30) through the GIC to the
//! CPU so `sched::on_timer_tick` fires periodically, giving aarch64 the same
//! timer-preemptive scheduling x86 has (previously it was cooperative-only).
//!
//! GICv3 is what QEMU `virt` exposes under `-accel hvf` on Apple Silicon (and
//! what modern real ARM hardware uses). The distributor + redistributor are
//! MMIO (already in the Device-mapped low 1 GiB); the CPU interface is the
//! `ICC_*_EL1` system registers. Only the **BSP** is wired — like x86, the
//! secondary cores park (they do bounded inference work, not scheduled tasks),
//! so preemption is BSP-driven.

use core::arch::asm;
use core::ptr::{read_volatile, write_volatile};
use core::sync::atomic::{AtomicU64, Ordering};

// QEMU `virt` GICv3 MMIO bases (also the SBSA/typical real-hardware layout is
// discovered from the DTB/ACPI; these match QEMU virt, our test target).
const GICD_BASE: u64 = 0x0800_0000; // distributor
const GICR_BASE: u64 = 0x080A_0000; // redistributor frame 0 (the BSP's)

// Distributor registers.
const GICD_CTLR: u64 = 0x0000;

// Redistributor: RD_base at GICR_BASE, SGI_base at +0x10000.
const GICR_WAKER: u64 = 0x0014; // in RD_base
const GICR_SGI: u64 = 0x1_0000; // SGI_base offset
const GICR_IGROUPR0: u64 = GICR_SGI + 0x0080;
const GICR_ISENABLER0: u64 = GICR_SGI + 0x0100;
const GICR_IPRIORITYR: u64 = GICR_SGI + 0x0400;

// Generic-timer PPIs: the EL1 physical timer is INTID 30; the EL2 physical
// timer is INTID 26. Under HVF + `-cpu host` the guest runs at EL2 with VHE
// (E2H=1), so `CNTP_*_EL0` drives the *EL2* timer (INTID 26); on plain EL1
// (real hardware / no-VHE) it's INTID 30. We enable + accept both, so the same
// code is correct regardless of which EL we booted at.
const TIMER_PPI_EL1: u32 = 30;
const TIMER_PPI_EL2: u32 = 26;
const TICK_HZ: u64 = 100; // scheduler tick frequency (cf. x86 PIT 1000 Hz)

/// Current exception level (1, 2, or 3), from `CurrentEL[3:2]`.
fn current_el() -> u64 {
    let el: u64;
    // SAFETY: reading CurrentEL is always valid.
    unsafe { asm!("mrs {}, CurrentEL", out(reg) el, options(nomem, nostack, preserves_flags)) };
    (el >> 2) & 0x3
}

/// Ticks since boot (the aarch64 counterpart of `pit::TICKS`).
pub static TICKS: AtomicU64 = AtomicU64::new(0);

pub fn ticks() -> u64 {
    TICKS.load(Ordering::Relaxed)
}

// --- UNDEF-probe state (see `init_bsp`) ---------------------------------
// The GICv3 CPU interface is the `ICC_*` system registers. On TCG/KVM/real ARM
// hardware they work at EL1 once `ICC_SRE_EL1.SRE=1`; under Apple-Silicon HVF
// the emulated GICv3 does NOT expose them to a bare-metal EL1 guest (access is
// UNDEFINED) and HVF also refuses to provide EL2. So before committing to the
// interrupt path we *probe* one CPU-interface access under the recoverable sync
// handler (`exceptions::aarch64_sync_dispatch`): if it UNDEFs, we note it and
// fall back to cooperative scheduling instead of crashing the boot.
static PROBING: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);
static PROBE_FAULTED: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

/// True while a CPU-interface access is being probed (read by the sync handler).
pub fn probing() -> bool {
    PROBING.load(Ordering::Acquire)
}

/// Called by the sync handler when the probed instruction UNDEFs.
pub fn note_probe_fault() {
    PROBE_FAULTED.store(true, Ordering::Release);
}

unsafe fn mmio_w32(base: u64, off: u64, v: u32) {
    unsafe { write_volatile((base + off) as *mut u32, v) };
}
unsafe fn mmio_r32(base: u64, off: u64) -> u32 {
    unsafe { read_volatile((base + off) as *const u32) }
}

fn cntfrq() -> u64 {
    let v: u64;
    // SAFETY: reading the counter frequency register is valid at EL1.
    unsafe { asm!("mrs {}, cntfrq_el0", out(reg) v, options(nomem, nostack, preserves_flags)) };
    v
}

/// Program the EL1 physical timer to fire one tick-interval from now.
fn timer_reload() {
    let interval = (cntfrq() / TICK_HZ).max(1);
    // SAFETY: CNTP_TVAL/CTL are EL0/EL1-accessible timer registers.
    unsafe {
        asm!("msr cntp_tval_el0, {}", in(reg) interval, options(nomem, nostack, preserves_flags));
        asm!("msr cntp_ctl_el0, {}", in(reg) 1u64, options(nomem, nostack, preserves_flags)); // ENABLE, IMASK=0
    }
}

/// Bring up the GICv3 (distributor + this core's redistributor + CPU interface)
/// and start the periodic timer, on the **BSP**. Returns `true` if the CPU
/// interface is usable (→ enable IRQs for preemptive scheduling) or `false` if
/// it UNDEFs (Apple-Silicon HVF → the caller stays cooperative). Call once,
/// before enabling IRQs; the vector table (`exceptions::init`) must already be
/// installed (the probe relies on its recoverable sync handler).
///
/// # Safety
/// Runs at EL1 on the BSP; the GIC MMIO windows are Device-mapped.
pub unsafe fn init_bsp() -> bool {
    unsafe {
        // --- Distributor: affinity routing + Group1 enable. ---
        let ctlr = mmio_r32(GICD_BASE, GICD_CTLR);
        mmio_w32(GICD_BASE, GICD_CTLR, ctlr | (1 << 4)); // ARE_NS
        mmio_w32(GICD_BASE, GICD_CTLR, mmio_r32(GICD_BASE, GICD_CTLR) | (1 << 1)); // EnableGrp1NS

        // --- Redistributor: wake it, then configure the timer PPI. ---
        // Clear ProcessorSleep, wait for ChildrenAsleep to clear.
        let waker = mmio_r32(GICR_BASE, GICR_WAKER);
        mmio_w32(GICR_BASE, GICR_WAKER, waker & !(1 << 1));
        let mut g = 0;
        while mmio_r32(GICR_BASE, GICR_WAKER) & (1 << 2) != 0 && g < 1_000_000 {
            g += 1;
        }
        // PPIs/SGIs to Group1 (bit per INTID).
        mmio_w32(GICR_BASE, GICR_IGROUPR0, 0xFFFF_FFFF);
        // Priority for both timer PPIs (byte-addressed): highest usable (0x00).
        write_volatile((GICR_BASE + GICR_IPRIORITYR + TIMER_PPI_EL1 as u64) as *mut u8, 0x00);
        write_volatile((GICR_BASE + GICR_IPRIORITYR + TIMER_PPI_EL2 as u64) as *mut u8, 0x00);
        // Enable both timer PPIs (EL1=30, EL2=26).
        mmio_w32(GICR_BASE, GICR_ISENABLER0, (1 << TIMER_PPI_EL1) | (1 << TIMER_PPI_EL2));

        // --- CPU interface (system registers). ---
        // Enable the system-register interface for our EL. When we run at EL2
        // (VHE, under HVF), ICC access is governed by ICC_SRE_EL2, which must
        // have SRE (bit0) + Enable (bit3) set before ICC_SRE_EL1/PMR/etc. become
        // usable; on plain EL1, ICC_SRE_EL1.SRE alone suffices.
        if current_el() == 2 {
            let mut sre2: u64;
            asm!("mrs {}, ICC_SRE_EL2", out(reg) sre2, options(nomem, nostack));
            sre2 |= (1 << 0) | (1 << 3); // SRE | Enable
            asm!("msr ICC_SRE_EL2, {}", "isb", in(reg) sre2, options(nostack));
        }
        let mut sre: u64;
        asm!("mrs {}, ICC_SRE_EL1", out(reg) sre, options(nomem, nostack));
        sre |= 1;
        asm!("msr ICC_SRE_EL1, {}", "isb", in(reg) sre, options(nostack));
        // --- Probe the CPU interface before relying on it. --- Attempt the
        // first ICC access (PMR) under the recoverable sync handler; if it
        // UNDEFs (HVF), bail out and let the caller stay cooperative.
        PROBE_FAULTED.store(false, Ordering::Release);
        PROBING.store(true, Ordering::Release);
        core::sync::atomic::compiler_fence(Ordering::SeqCst);
        asm!("msr ICC_PMR_EL1, {}", "isb", in(reg) 0xFFu64, options(nostack)); // priority mask = allow all
        core::sync::atomic::compiler_fence(Ordering::SeqCst);
        PROBING.store(false, Ordering::Release);
        if PROBE_FAULTED.load(Ordering::Acquire) {
            crate::ktrace::log("gic", "CPU interface (ICC_* sysregs) unavailable -- staying cooperative (HVF has no GICv3 sysreg interface for bare-metal EL1)");
            return false;
        }

        // Enable Group1 delivery to this CPU, then start the periodic timer.
        asm!("msr ICC_IGRPEN1_EL1, {}", in(reg) 1u64, options(nomem, nostack));
        timer_reload();
        crate::ktrace::log_fmt(format_args!("gic: GICv3 up at EL{}, timer @ {} Hz (cntfrq {})", current_el(), TICK_HZ, cntfrq()));
        true
    }
}

/// Handle one IRQ: acknowledge via `ICC_IAR1_EL1`, service the timer, complete
/// via `ICC_EOIR1_EL1`. Called from the IRQ vector (`exceptions`), with the full
/// trap frame already saved, so `on_timer_tick` may safely switch tasks.
pub fn handle_irq() {
    // SAFETY: EL1 GIC CPU-interface reads/writes; IAR ack then EOI is the
    // required GICv3 handshake.
    let intid: u64;
    unsafe { asm!("mrs {}, ICC_IAR1_EL1", out(reg) intid, options(nomem, nostack)) };
    let id = (intid & 0xffffff) as u32;

    if id == TIMER_PPI_EL1 || id == TIMER_PPI_EL2 {
        TICKS.fetch_add(1, Ordering::Relaxed);
        timer_reload(); // rearm before EOI
        unsafe { asm!("msr ICC_EOIR1_EL1, {}", in(reg) intid, options(nomem, nostack)) };
        crate::sched::on_timer_tick();
        return;
    }

    // Unknown/spurious (1023 = no pending): still EOI real ones.
    if id < 1020 {
        unsafe { asm!("msr ICC_EOIR1_EL1, {}", in(reg) intid, options(nomem, nostack)) };
    }
}
