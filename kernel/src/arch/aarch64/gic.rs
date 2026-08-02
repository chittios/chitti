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

// GICv3 MMIO bases are **discovered**, never assumed: from the device tree's
// `arm,gic-v3` node `reg` (QEMU `virt`, any FDT platform) or, when UEFI booted us
// with no device tree, from the ACPI MADT (`acpi::gic_from_rsdp` — VirtualBox-ARM,
// UTM, real SBSA machines). A real server's distributor is nowhere near QEMU's
// address, and writing GICD_CTLR at a guessed base pokes unrelated MMIO.
//
// These defaults are the QEMU `virt` layout, used only as a last resort when a
// device tree advertises `arm,gic-v3` but carries no usable `reg` (malformed, or
// a truncated FDT) — the platform is then almost certainly QEMU-like anyway.
const GICD_BASE_DEFAULT: u64 = 0x0800_0000; // distributor
const GICR_BASE_DEFAULT: u64 = 0x080A_0000; // redistributor frame 0 (the BSP's)

/// The resolved distributor base, set by [`init_bsp`] before any GIC access.
static GICD: AtomicU64 = AtomicU64::new(GICD_BASE_DEFAULT);
/// The resolved redistributor base for the BSP, set by [`init_bsp`].
static GICR: AtomicU64 = AtomicU64::new(GICR_BASE_DEFAULT);

#[inline]
fn gicd_base() -> u64 {
    GICD.load(Ordering::Relaxed)
}
#[inline]
fn gicr_base() -> u64 {
    GICR.load(Ordering::Relaxed)
}

// Distributor registers.
const GICD_CTLR: u64 = 0x0000;

// Redistributor: RD_base at `gicr_base()`, SGI_base at +0x10000.
const GICR_WAKER: u64 = 0x0014; // in RD_base
const GICR_SGI: u64 = 0x1_0000; // SGI_base offset
const GICR_IGROUPR0: u64 = GICR_SGI + 0x0080;
const GICR_ISENABLER0: u64 = GICR_SGI + 0x0100;
const GICR_IPRIORITYR: u64 = GICR_SGI + 0x0400;

// Generic-timer PPIs. We drive the **virtual timer** (CNTV, INTID 27): it is the
// timer a guest is allowed to use — the *physical* timer (CNTP, INTID 30) is
// commonly trapped/denied by the hypervisor (VirtualBox raises a GP on a guest
// `msr CNTP_TVAL_EL0`; KVM/HVF reserve it for the host). The virtual timer works
// at EL1 under every hypervisor and on real hardware. We also enable + accept
// the physical (30) and EL2 (26) PPIs so we still tick on a platform that only
// routes those, but we only *program* CNTV.
const TIMER_PPI_VIRT: u32 = 27; // virtual timer (CNTV) -- the one we program
const TIMER_PPI_EL1: u32 = 30; // EL1 physical timer (CNTP)
const TIMER_PPI_EL2: u32 = 26; // EL2 physical timer (CNTHP)
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
/// Set by `init_bsp`: whether the GIC + timer came up (→ preemption available).
/// Read by `start_preemption`, which actually unmasks IRQs — deferred until the
/// framebuffer + devices are up, so device bring-up runs with IRQs masked (as in
/// the cooperative path) and the display is never left uninitialized.
static PREEMPTIVE: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);
/// True only after [`start_preemption`] has *observed* a timer tick. Distinct
/// from [`PREEMPTIVE`]: init can arm the timer while delivery still fails (HVF
/// `-kernel` with no GIC), and then `wfi` in the idle path never wakes — the
/// shell looks frozen. [`crate::power::idle::halt`] consults this.
static TIMER_LIVE: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

/// Whether a timer IRQ has been observed (safe to `wfi` / halt the BSP).
pub fn timer_live() -> bool {
    TIMER_LIVE.load(Ordering::Acquire)
}

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

/// Program the **virtual timer** (CNTV) to fire one tick-interval from now.
/// The virtual timer is the guest-safe generic timer (see `TIMER_PPI_VIRT`).
fn timer_reload() {
    let interval = (cntfrq() / TICK_HZ).max(1);
    // SAFETY: CNTV_TVAL/CTL are EL0/EL1-accessible timer registers.
    unsafe {
        asm!("msr cntv_tval_el0, {}", in(reg) interval, options(nomem, nostack, preserves_flags));
        asm!("msr cntv_ctl_el0, {}", in(reg) 1u64, options(nomem, nostack, preserves_flags)); // ENABLE, IMASK=0
    }
}

/// This core's `MPIDR_EL1` (affinity bits), for matching the ACPI MADT's per-CPU
/// GICC entries.
fn mpidr() -> u64 {
    let v: u64;
    // SAFETY: MPIDR_EL1 is readable at EL1.
    unsafe { asm!("mrs {}, mpidr_el1", out(reg) v, options(nomem, nostack, preserves_flags)) };
    v
}

/// Discover the GICv3 distributor/redistributor bases from firmware and store
/// them in [`GICD`]/[`GICR`]. Returns `false` when this platform has no GICv3 we
/// can locate, in which case the caller must stay cooperative.
///
/// Order matters, and each branch encodes a real platform:
///
/// 1. **A device tree that advertises `arm,gic-v3`** (QEMU `virt` via `-kernel`,
///    most FDT platforms) — take `reg[0]` as the distributor and `reg[1]` as the
///    redistributor. Falls back to the QEMU-`virt` constants only if `reg` is
///    unreadable.
/// 2. **A device tree that does not** — Apple Silicon. There is no GIC; return
///    false. Critically this must *not* fall through to ACPI: Apple has no ACPI,
///    and probing a guessed base is an uncatchable data abort.
/// 3. **No device tree at all** (UEFI stub boot: VirtualBox-ARM, UTM, real SBSA
///    hardware) — parse the ACPI MADT. This is the case that previously left
///    every UEFI-booted ARM machine with no timer interrupt at all, silently
///    cooperative, because the old code required an FDT node that never existed.
fn resolve_gic_bases() -> bool {
    let fdt = super::boot::boot_x0();
    // SAFETY: `boot_x0` is the FDT pointer (or a non-FDT value the magic rejects).
    if unsafe { crate::fdt::present(fdt) } {
        // SAFETY: as above; the FDT is mapped and its magic checked.
        if !unsafe { crate::fdt::has_compatible(fdt, b"arm,gic-v3") } {
            crate::ktrace::log("gic", "device tree advertises no GICv3 (Apple AIC?) -- cooperative scheduling; AIC is a follow-up");
            return false;
        }
        // SAFETY: as above. reg[0] = distributor, reg[1] = redistributor.
        let d = unsafe { crate::fdt::reg_nth_of_compatible(fdt, b"arm,gic-v3", 0) };
        let r = unsafe { crate::fdt::reg_nth_of_compatible(fdt, b"arm,gic-v3", 1) };
        match (d, r) {
            (Some((dbase, _)), Some((rbase, _))) if dbase != 0 && rbase != 0 => {
                GICD.store(dbase, Ordering::Relaxed);
                GICR.store(rbase, Ordering::Relaxed);
                super::mmu::map_device_gib(dbase);
                super::mmu::map_device_gib(rbase);
                crate::ktrace::log_fmt(format_args!(
                    "gic: GICv3 from device tree -- GICD {dbase:#x} GICR {rbase:#x}"
                ));
            }
            _ => {
                crate::ktrace::log_fmt(format_args!(
                    "gic: device tree has arm,gic-v3 but no usable reg -- assuming the QEMU virt layout (GICD {GICD_BASE_DEFAULT:#x} GICR {GICR_BASE_DEFAULT:#x})"
                ));
            }
        }
        return true;
    }
    // No device tree: UEFI handed us ACPI tables instead.
    let bi = super::boot::boot_x1();
    if bi == 0 {
        crate::ktrace::log("gic", "no device tree and no boot-info page -- cannot locate a GIC; cooperative scheduling");
        return false;
    }
    // SAFETY: `bi` is the stub's identity-mapped boot-info page; verify the magic
    // before trusting the RSDP field at offset 40.
    let magic = unsafe { core::slice::from_raw_parts(bi as *const u8, 8) };
    if magic != b"CHITTIBI" {
        crate::ktrace::log("gic", "boot-info magic mismatch -- cannot locate a GIC; cooperative scheduling");
        return false;
    }
    // SAFETY: boot-info verified; RSDP pointer is at offset 40.
    let rsdp = unsafe { core::ptr::read_volatile((bi + 40) as *const u64) };
    match crate::acpi::gic_from_rsdp(rsdp, mpidr()) {
        Some(g) => {
            GICD.store(g.gicd, Ordering::Relaxed);
            GICR.store(g.gicr, Ordering::Relaxed);
            // A real machine's GIC is nowhere near the low Device block the
            // identity map covers by default — map both windows explicitly.
            super::mmu::map_device_gib(g.gicd);
            super::mmu::map_device_gib(g.gicr);
            crate::ktrace::log_fmt(format_args!(
                "gic: GICv{} from ACPI MADT -- GICD {:#x} GICR {:#x}",
                g.version, g.gicd, g.gicr
            ));
            if g.version != 0 && g.version < 3 {
                crate::ktrace::log_fmt(format_args!(
                    "gic: MADT reports GICv{} -- this driver is GICv3-only; cooperative scheduling",
                    g.version
                ));
                return false;
            }
            true
        }
        None => {
            crate::ktrace::log_fmt(format_args!(
                "gic: no usable GIC in the ACPI MADT (RSDP {rsdp:#x}) -- cooperative scheduling"
            ));
            false
        }
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
    // Locate the GIC before touching it. Apple Silicon has **no GIC** at all (it
    // uses AIC at a wholly different base) and touching a phantom distributor is
    // a data abort the sysreg probe below cannot catch — it is MMIO, not an UNDEF.
    // So the base must come from firmware, and if firmware doesn't describe a
    // GICv3 we stay cooperative rather than guess.
    if !resolve_gic_bases() {
        PREEMPTIVE.store(false, Ordering::Release);
        return false;
    }
    let (gicd_base, gicr_base) = (gicd_base(), gicr_base());
    unsafe {
        // --- Distributor: affinity routing + Group1 enable. ---
        let ctlr = mmio_r32(gicd_base, GICD_CTLR);
        mmio_w32(gicd_base, GICD_CTLR, ctlr | (1 << 4)); // ARE_NS
        mmio_w32(gicd_base, GICD_CTLR, mmio_r32(gicd_base, GICD_CTLR) | (1 << 1)); // EnableGrp1NS

        // --- Redistributor: wake it, then configure the timer PPI. ---
        // Clear ProcessorSleep, wait for ChildrenAsleep to clear.
        let waker = mmio_r32(gicr_base, GICR_WAKER);
        mmio_w32(gicr_base, GICR_WAKER, waker & !(1 << 1));
        let mut g = 0;
        while mmio_r32(gicr_base, GICR_WAKER) & (1 << 2) != 0 && g < 1_000_000 {
            g += 1;
        }
        // PPIs/SGIs to Group1 (bit per INTID).
        mmio_w32(gicr_base, GICR_IGROUPR0, 0xFFFF_FFFF);
        // Priority for all timer PPIs (byte-addressed): highest usable (0x00).
        write_volatile((gicr_base + GICR_IPRIORITYR + TIMER_PPI_VIRT as u64) as *mut u8, 0x00);
        write_volatile((gicr_base + GICR_IPRIORITYR + TIMER_PPI_EL1 as u64) as *mut u8, 0x00);
        write_volatile((gicr_base + GICR_IPRIORITYR + TIMER_PPI_EL2 as u64) as *mut u8, 0x00);
        // Enable the timer PPIs (virtual=27, physical EL1=30, EL2=26). We only
        // *program* the virtual timer, but enabling all is harmless.
        mmio_w32(gicr_base, GICR_ISENABLER0, (1 << TIMER_PPI_VIRT) | (1 << TIMER_PPI_EL1) | (1 << TIMER_PPI_EL2));

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
        // --- Probe every system register the interrupt path needs, before
        // relying on any of it. --- Each of these can be trapped/UNDEF on a
        // given hypervisor (HVF UNDEFs the ICC_* CPU interface entirely;
        // VirtualBox raised a GP on the *physical* timer). We run them under the
        // recoverable sync handler, and if ANY faults we bail to cooperative
        // scheduling rather than guru-crash: PMR + Group1 enable + start the
        // virtual timer. (The redistributor/distributor above are MMIO, which
        // fault differently and are always present, so they stay outside.)
        PROBE_FAULTED.store(false, Ordering::Release);
        PROBING.store(true, Ordering::Release);
        core::sync::atomic::compiler_fence(Ordering::SeqCst);
        asm!("msr ICC_PMR_EL1, {}", "isb", in(reg) 0xFFu64, options(nostack)); // priority mask = allow all
        // EOImode = 0: a write to ICC_EOIR1 both drops priority AND deactivates
        // the interrupt. The reset value is implementation-defined; if it were 1,
        // our EOI would leave the PPI active forever (exactly one tick, then
        // silence). Set it explicitly (still inside the recoverable probe).
        let mut ctlr: u64;
        asm!("mrs {}, ICC_CTLR_EL1", out(reg) ctlr, options(nomem, nostack));
        ctlr &= !(1 << 1); // EOImode = 0
        asm!("msr ICC_CTLR_EL1, {}", "isb", in(reg) ctlr, options(nostack));
        asm!("msr ICC_IGRPEN1_EL1, {}", "isb", in(reg) 1u64, options(nostack)); // enable Group1 delivery
        timer_reload(); // start the virtual timer (CNTV)
        core::sync::atomic::compiler_fence(Ordering::SeqCst);
        PROBING.store(false, Ordering::Release);
        if PROBE_FAULTED.load(Ordering::Acquire) {
            crate::ktrace::log("gic", "CPU interface / timer sysreg unavailable -- staying cooperative (hypervisor doesn't expose the GICv3 sysreg interface or virtual timer to a bare-metal EL1 guest)");
            PREEMPTIVE.store(false, Ordering::Release);
            return false;
        }
        crate::ktrace::log_fmt(format_args!("gic: GICv3 up at EL{}, virtual timer @ {} Hz (cntfrq {}); IRQs still masked until devices are up", current_el(), TICK_HZ, cntfrq()));
        PREEMPTIVE.store(true, Ordering::Release);
        true
    }
}

/// Unmask IRQs to begin timer-preemptive scheduling — called **after** the
/// framebuffer + devices are brought up (they initialize with IRQs masked, like
/// the cooperative path, so the display is never left half-initialized). No-op
/// if the GIC/timer weren't available (`init_bsp` returned false → cooperative).
pub fn start_preemption() {
    if !PREEMPTIVE.load(Ordering::Acquire) {
        crate::ktrace::log("gic", "no timer IRQ -- cooperative scheduling");
        return;
    }
    super::interrupts::enable();
    // Warmup: confirm timer IRQs are actually *delivered* before committing to
    // preemptive mode. Deliberately a PURE MEMORY spin on the tick counter --
    // no counter/system-register reads, because on some hypervisors (VirtualBox)
    // CNTPCT reads trap to the VMM (making each iteration microseconds) and the
    // virtual counter may not advance for a bare-metal guest at all, which
    // turned a "bounded" wait into tens of minutes. ~50M cheap iterations is
    // 50-200 ms of real time -- several 100 Hz tick periods -- on any host.
    let mut iters = 0u64;
    while ticks() == 0 && iters < 50_000_000 {
        iters += 1;
        core::hint::spin_loop();
    }
    if ticks() == 0 {
        super::interrupts::disable();
        TIMER_LIVE.store(false, Ordering::Release);
        crate::ktrace::log("gic", "timer IRQs not delivered -- re-masked, staying cooperative");
        return;
    }
    TIMER_LIVE.store(true, Ordering::Release);
    crate::ktrace::log_fmt(format_args!("gic: timer delivering IRQs ({} ticks) -- preemptive scheduling", ticks()));
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

    if id == TIMER_PPI_VIRT || id == TIMER_PPI_EL1 || id == TIMER_PPI_EL2 {
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
