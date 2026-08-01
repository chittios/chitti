//! Architecture-specific code lives under `arch/<name>/`, and the rest of the
//! kernel reaches it only through the small **facade** re-exported here --
//! `arch::interrupts` and `arch::hlt` -- never `arch::x86_64::...` directly.
//! That is what lets the arch-independent layers (mm, sched, ktrace, cortex,
//! synapse, persona, ...) compile unchanged on either target; each supported
//! architecture provides the same facade surface.
//!
//! `x86_64` is the mature port (Limine boot, the full OS). `aarch64` is the
//! native Apple-Silicon port (QEMU + HVF), brought up incrementally; the two
//! are being collapsed into this single dual-arch tree.

#[cfg(target_arch = "x86_64")]
pub mod x86_64;

#[cfg(target_arch = "x86_64")]
pub use x86_64::{hlt, interrupts, poweroff, reboot};

#[cfg(target_arch = "aarch64")]
pub mod aarch64;

#[cfg(target_arch = "aarch64")]
pub use aarch64::{has_i8mm, hlt, interrupts, poweroff, reboot};

/// Milliseconds since boot -- the PIT tick counter on x86, the generic timer
/// on aarch64. Used for inference throughput timing.
#[cfg(target_arch = "x86_64")]
pub fn now_ms() -> u64 {
    x86_64::pit::ticks()
}

#[cfg(target_arch = "aarch64")]
pub fn now_ms() -> u64 {
    aarch64::time_ms()
}

/// Current wall-clock time as a Unix timestamp read from the hardware RTC, or
/// `None` if no RTC is readable (the wall clock then falls back to a default
/// until `/datetime` sets it). CMOS on x86, PL031 on aarch64.
#[cfg(target_arch = "x86_64")]
pub fn rtc_unix() -> Option<u64> {
    x86_64::rtc::read_unix()
}

#[cfg(target_arch = "aarch64")]
pub fn rtc_unix() -> Option<u64> {
    // Apple Silicon has no PL031 at 0x0901_0000 — its RTC is behind the PMU/SMC.
    // Reading the phantom PL031 there data-aborts under m1n1's hv, so skip it;
    // the wall clock uses the `chitti.epoch=` bootarg / `/datetime` instead.
    if aarch64::is_apple() {
        return None;
    }
    aarch64::rtc::read_unix()
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub fn rtc_unix() -> Option<u64> {
    None
}

/// Poll every present mouse transport, feeding motion/buttons into
/// [`crate::mouse`]. aarch64: virtio pointer + PL050 PS/2 + USB (xHCI/HID).
/// x86: i8042 PS/2 aux + USB (xHCI/HID). Cheap; called from the UI idle
/// loops via `mouse::tick`.
#[cfg(target_arch = "aarch64")]
pub fn mouse_poll() {
    aarch64::virtio_pointer::poll();
    aarch64::pl050_mouse::poll();
    aarch64::xhci::poll_mouse();
}

#[cfg(target_arch = "x86_64")]
pub fn mouse_poll() {
    x86_64::i8042::poll_mouse();
    x86_64::xhci::poll_mouse();
    // HID-over-I2C touchpad (Intel LPSS). A no-op unless one was found at boot;
    // most laptops from ~2016 have no PS/2 aux port, so this is their only pointer.
    crate::drivers::i2c_hid::poll();
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub fn mouse_poll() {}

/// A best-effort hardware entropy word, for seeding the CSPRNG (TLS handshake
/// keys). x86: `RDRAND` when the CPU reports it, else 0. aarch64: `RNDR`
/// (FEAT_RNG) when present, else 0. `net::tls::seed_rng` mixes several of these
/// with the cycle counter, so a 0 (facility absent — QEMU/HVF often lack both)
/// degrades to counter-jitter entropy rather than failing. Not audited crypto
/// entropy; adequate for a research OS talking to a model server over the LAN.
#[cfg(target_arch = "x86_64")]
pub fn hw_rand() -> u64 {
    x86_64::hw_rand()
}
#[cfg(target_arch = "aarch64")]
pub fn hw_rand() -> u64 {
    aarch64::hw_rand()
}
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub fn hw_rand() -> u64 {
    0
}

/// Number of CPU cores online — x86 SMP APs (`smp`) or aarch64 PSCI-started
/// secondaries (`arch::aarch64::smp`), behind one API for the status bar and
/// `/top`.
#[cfg(target_arch = "x86_64")]
pub fn cpu_count() -> u64 {
    crate::smp::cpu_count()
}
#[cfg(target_arch = "aarch64")]
pub fn cpu_count() -> u64 {
    aarch64::smp::online_cpus() as u64
}
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub fn cpu_count() -> u64 {
    1
}

/// Cumulative compute-cycles core `core` has spent in the parallel matmul
/// workers, for `/top`'s per-core utilisation. aarch64 splits inference across
/// cores (`aarch64::smp`); x86 runs it single-core, so only core 0 is ever
/// busy there (returns 0 for the rest).
#[cfg(target_arch = "aarch64")]
pub fn core_busy_cycles(core: usize) -> u64 {
    aarch64::smp::core_busy_cycles(core)
}
#[cfg(not(target_arch = "aarch64"))]
pub fn core_busy_cycles(_core: usize) -> u64 {
    0
}

/// A monotonically-advancing cycle/tick counter for entropy mixing (finer than
/// `now_ms`): the TSC on x86, `CNTVCT_EL0` on aarch64.
#[cfg(target_arch = "x86_64")]
pub fn cycle_count() -> u64 {
    x86_64::cycle_count()
}
#[cfg(target_arch = "aarch64")]
pub fn cycle_count() -> u64 {
    aarch64::cycle_count()
}
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub fn cycle_count() -> u64 {
    now_ms()
}

/// Cores available to fan work out across, counting the caller's own core.
///
/// Arch-neutral so compute-heavy code does not have to `cfg` on the arch to find
/// out whether parallelism exists — which is exactly how x86 ended up running
/// every parallel loop on one core while aarch64 split it across the fleet.
#[cfg(target_arch = "x86_64")]
pub fn online_cpus() -> usize {
    crate::smp::online_cpus()
}
#[cfg(target_arch = "aarch64")]
pub fn online_cpus() -> usize {
    aarch64::smp::online_cpus()
}
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub fn online_cpus() -> usize {
    1
}

/// Run `f` over `[0, n)`, split across every online core.
///
/// `min_chunk` is the smallest range worth handing to a worker; below roughly
/// twice that the whole range runs inline on the calling core, since the barrier
/// would cost more than the work. Both arches implement the same static-partition
/// barrier and the same fallback, so a caller gets parallelism wherever it exists
/// and correct serial behaviour where it does not.
///
/// # Safety
/// `f` must be safe to call concurrently on disjoint sub-ranges of `[0, n)`
/// sharing `ctx`, and `ctx` must outlive the call.
#[cfg(target_arch = "x86_64")]
pub unsafe fn parallel_for(n: usize, min_chunk: usize, f: unsafe fn(usize, usize, *mut u8), ctx: *mut u8) {
    // SAFETY: forwarded under the caller's contract.
    unsafe { crate::smp::parallel_for(n, min_chunk, f, ctx) }
}
#[cfg(target_arch = "aarch64")]
pub unsafe fn parallel_for(n: usize, min_chunk: usize, f: unsafe fn(usize, usize, *mut u8), ctx: *mut u8) {
    // SAFETY: forwarded under the caller's contract.
    unsafe { aarch64::smp::parallel_for(n, min_chunk, f, ctx) }
}
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub unsafe fn parallel_for(n: usize, _min_chunk: usize, f: unsafe fn(usize, usize, *mut u8), ctx: *mut u8) {
    // SAFETY: forwarded under the caller's contract; single-core.
    unsafe { f(0, n, ctx) }
}

/// Whether a USB Ethernet adapter's bulk endpoints are configured.
#[cfg(target_arch = "x86_64")]
pub fn usb_bulk_ready() -> bool {
    x86_64::xhci::usb_bulk_ready()
}
#[cfg(target_arch = "aarch64")]
pub fn usb_bulk_ready() -> bool {
    aarch64::xhci::usb_bulk_ready()
}
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub fn usb_bulk_ready() -> bool {
    false
}

/// USB Ethernet bulk transport, arch-neutral (the controller lives under the
/// per-arch xHCI wrapper, exactly like `mouse_poll`). Used by `net::usb_eth`.
#[cfg(target_arch = "x86_64")]
pub fn usb_bulk_arm_in() {
    x86_64::xhci::usb_bulk_arm_in()
}
#[cfg(target_arch = "aarch64")]
pub fn usb_bulk_arm_in() {
    aarch64::xhci::usb_bulk_arm_in()
}
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub fn usb_bulk_arm_in() {}

#[cfg(target_arch = "x86_64")]
pub fn usb_bulk_take_in(out: &mut [u8]) -> Option<usize> {
    x86_64::xhci::usb_bulk_take_in(out)
}
#[cfg(target_arch = "aarch64")]
pub fn usb_bulk_take_in(out: &mut [u8]) -> Option<usize> {
    aarch64::xhci::usb_bulk_take_in(out)
}
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub fn usb_bulk_take_in(_out: &mut [u8]) -> Option<usize> {
    None
}

#[cfg(target_arch = "x86_64")]
pub fn usb_bulk_send(data: &[u8]) -> bool {
    x86_64::xhci::usb_bulk_send(data)
}
#[cfg(target_arch = "aarch64")]
pub fn usb_bulk_send(data: &[u8]) -> bool {
    aarch64::xhci::usb_bulk_send(data)
}
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub fn usb_bulk_send(_data: &[u8]) -> bool {
    false
}

/// This core's hardware identity, cheap and MMIO-free.
///
/// x86 reads the *initial* APIC id out of `CPUID.01:EBX[31:24]` rather than the
/// local-APIC `ID` register, because this has to work before `apic::init` has
/// mapped that MMIO — and on x86 `sched::init` runs *before* the APIC is up.
/// aarch64 reads `MPIDR_EL1`'s affinity bits, which are always available at EL1.
#[cfg(target_arch = "x86_64")]
pub fn hw_cpu_id() -> u64 {
    let ebx: u32;
    // SAFETY: CPUID leaf 1 exists on every x86_64 CPU and has no side effects.
    unsafe {
        core::arch::asm!(
            "push rbx", "mov eax, 1", "cpuid", "mov {out:e}, ebx", "pop rbx",
            out = out(reg) ebx,
            out("eax") _, out("ecx") _, out("edx") _,
            options(nostack, preserves_flags),
        );
    }
    (ebx >> 24) as u64
}
#[cfg(target_arch = "aarch64")]
pub fn hw_cpu_id() -> u64 {
    let v: u64;
    // SAFETY: MPIDR_EL1 is readable at EL1 and has no side effects.
    unsafe { core::arch::asm!("mrs {}, mpidr_el1", out(reg) v, options(nomem, nostack, preserves_flags)) };
    v & 0xff_ffff // Aff2:Aff1:Aff0
}
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub fn hw_cpu_id() -> u64 {
    0
}

/// The core the scheduler was brought up on. `u64::MAX` until recorded.
static BOOT_CPU: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(u64::MAX);

/// Record the current core as the one that owns the scheduler. Called once from
/// `sched::init`.
pub fn claim_boot_cpu() {
    BOOT_CPU.store(hw_cpu_id(), core::sync::atomic::Ordering::SeqCst);
}

/// Whether this core is the one the scheduler was brought up on.
///
/// # Why the scheduler is single-core, and why that is load-bearing
///
/// There is **no TLB shootdown on either arch**, so a `mm::space::AddressSpace` is
/// only valid on the core that activated it. Today that is safe by construction
/// rather than by policy: application processors never touch `sched` — they park
/// waiting for IPI-driven `parallel_for` work and arm no timer, so `reschedule`
/// only ever runs here. A future change that gave APs a timer tick to get real SMP
/// scheduling would reintroduce the hazard silently, and the symptom would be
/// stale-TLB memory corruption rather than a fault. Hence the assertion in
/// `sched::reschedule` — this is a tripwire for that change, not a mechanism.
pub fn is_boot_cpu() -> bool {
    match BOOT_CPU.load(core::sync::atomic::Ordering::SeqCst) {
        u64::MAX => true, // not yet recorded: early boot, single-threaded
        id => id == hw_cpu_id(),
    }
}

/// How a userspace tenant left ring 3 / EL0, in arch-neutral terms.
///
/// The two transports report the same two outcomes with different syndromes — x86 an
/// interrupt vector plus `CR2`, aarch64 `ESR_EL1` plus `FAR_EL1` — so this is a
/// translation, not a lowest common denominator: both fields survive, only their
/// names generalise. Callers above the transport (the loader, the tests, eventually
/// the scheduler) go through this so a tenant behaves identically on both machines,
/// which is the standing dual-architecture rule applied to the newest subsystem
/// rather than retrofitted to it later.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TenantExit {
    /// It submitted an ABI call. `number` is the [`crate::synapse::abi::Entry`].
    Called { number: u64, arg0: u64, arg1: u64 },
    /// It took a fault instead. `syndrome` is the vector (x86) or `ESR_EL1`
    /// (aarch64); `address` is `CR2` or `FAR_EL1`.
    Faulted { syndrome: u64, address: u64 },
}

impl TenantExit {
    /// Did the tenant stop because it asked to?
    ///
    /// The distinction that matters to a loader: any other outcome — a fault, or a
    /// call that is not `Exit` — means the tenant was interrupted rather than
    /// finished, and for a fault the likeliest cause is that the *kernel* mapped it
    /// wrong.
    pub fn is_deliberate_exit(&self) -> bool {
        matches!(self, TenantExit::Called { number, .. } if crate::synapse::abi::Entry::from_raw(*number) == Some(crate::synapse::abi::Entry::Exit))
    }
}

/// Run `task` in userspace at `entry_va` with stack `stack_va` in `space`, and report
/// how it left. The arch-neutral entry point for [`crate::synapse::tenant`].
///
/// # Safety
/// `entry_va` must be mapped user-executable in `space` and `stack_va` user-writable;
/// `space` must share the kernel mappings. Not reentrant.
pub unsafe fn enter_tenant(
    task: crate::sched::TaskId,
    space: &crate::mm::space::AddressSpace,
    entry_va: u64,
    stack_va: u64,
) -> TenantExit {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: forwarded verbatim; the caller carries the contract.
        match unsafe { x86_64::fastcall::enter_ring3(task, space, entry_va, stack_va) } {
            x86_64::fastcall::Exit::Svc(t) => {
                TenantExit::Called { number: t.number, arg0: t.arg0, arg1: t.arg1 }
            }
            x86_64::fastcall::Exit::Fault { code, addr } => {
                TenantExit::Faulted { syndrome: code, address: addr }
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: as above.
        match unsafe { aarch64::el0::enter_el0(task, space, entry_va, stack_va) } {
            aarch64::el0::Exit::Svc(t) => TenantExit::Called { number: t.number, arg0: t.arg0, arg1: t.arg1 },
            aarch64::el0::Exit::Fault { esr, far } => TenantExit::Faulted { syndrome: esr, address: far },
        }
    }
}
