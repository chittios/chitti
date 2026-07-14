//! aarch64 architecture support (Phase 7): the native Apple-Silicon port.
//!
//! Booted directly by `qemu-system-aarch64 -M virt -kernel` (no Limine); under
//! `-accel hvf` the guest runs natively on the M-series CPU. This module
//! provides the same facade the rest of the kernel expects from an arch
//! (`interrupts`, `hlt`) plus the aarch64-specific bring-up (boot stub, MMU,
//! PL011 UART, generic timer) the shared upper layers build on.

pub mod apple_usb;
pub mod boot;
pub mod dart;
pub mod dtb;
pub mod mmu;
pub mod ahci;
pub mod disk;
pub mod exceptions;
pub mod gic;
pub mod nvme;
pub mod pl050;
pub mod pl050_mouse;
pub mod ramfb;
pub mod rtc;
pub mod smp;
pub mod virtio_blk;
pub mod virtio_input;
pub mod virtio_net;
pub mod virtio_pointer;
pub mod virtio_snd;
pub mod virtio_pci;
pub mod xhci;

use core::arch::asm;

/// The Limine HHDM offset, set once at `limine_start`. On the `-kernel` build,
/// RAM is identity-mapped (VA == PA) so this stays 0 and `dma_to_phys` is the
/// identity. On the Limine build, heap RAM lives at `phys + hhdm`, so a device
/// (virtio) handed a heap address needs `va - hhdm` — its physical address.
#[cfg(feature = "boot-limine")]
static HHDM_OFFSET: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

#[cfg(feature = "boot-limine")]
pub fn set_hhdm(offset: u64) {
    HHDM_OFFSET.store(offset, core::sync::atomic::Ordering::Relaxed);
}

/// Translate a CPU (heap) virtual address to the physical address a device sees.
/// Identity on the `-kernel` build; `va - hhdm` on the Limine build.
#[inline]
pub fn dma_to_phys(va: u64) -> u64 {
    #[cfg(feature = "boot-limine")]
    {
        va - HHDM_OFFSET.load(core::sync::atomic::Ordering::Relaxed)
    }
    #[cfg(not(feature = "boot-limine"))]
    {
        va
    }
}

/// Halt the core until an interrupt (wait-for-interrupt).
#[inline]
pub fn hlt() {
    // SAFETY: `wfi` only idles the core; no memory-safety implications.
    unsafe { asm!("wfi", options(nomem, nostack, preserves_flags)) };
}

/// Power off via PSCI `SYSTEM_OFF` (function id 0x84000008) through the HVC
/// conduit the QEMU `virt` machine advertises -- QEMU turns it into a clean
/// process exit, so typing `exit` at the shell quits the emulator.
pub fn poweroff() -> ! {
    // SAFETY: PSCI SYSTEM_OFF has no memory effects; it does not return.
    unsafe {
        asm!(
            "mov w0, #0x0008",
            "movk w0, #0x8400, lsl #16", // w0 = 0x84000008 (PSCI_SYSTEM_OFF)
            "hvc #0",
            options(nomem, nostack, noreturn),
        );
    }
}

/// Reboot via PSCI `SYSTEM_RESET` (function id 0x84000009). On real hardware
/// and most hypervisors this cold-resets the board; under QEMU with
/// `-no-reboot` the process exits instead (same as SYSTEM_OFF for the host).
pub fn reboot() -> ! {
    // SAFETY: PSCI SYSTEM_RESET has no memory effects; it does not return.
    unsafe {
        asm!(
            "mov w0, #0x0009",
            "movk w0, #0x8400, lsl #16", // w0 = 0x84000009 (PSCI_SYSTEM_RESET)
            "hvc #0",
            options(nomem, nostack, noreturn),
        );
    }
}

/// CPU interrupt masking via `DAIF.I` (the IRQ mask bit), mirroring the x86
/// `interrupts` facade so `mm::Locked`, `ktrace`, and the scheduler are
/// arch-agnostic.
pub mod interrupts {
    use core::arch::asm;
    use core::sync::atomic::{AtomicBool, Ordering};

    /// Truthful-logging shadow of the IRQ state (see the x86 counterpart);
    /// not used for correctness.
    pub static INTERRUPTS_ENABLED: AtomicBool = AtomicBool::new(false);

    #[inline]
    pub fn enable() {
        // SAFETY: unmasking IRQs is safe once the vector table + timer are set.
        unsafe { asm!("msr daifclr, #2", options(nomem, nostack, preserves_flags)) };
        INTERRUPTS_ENABLED.store(true, Ordering::Relaxed);
    }

    #[inline]
    pub fn disable() {
        // SAFETY: masking IRQs has no memory-safety implications.
        unsafe { asm!("msr daifset, #2", options(nomem, nostack, preserves_flags)) };
        INTERRUPTS_ENABLED.store(false, Ordering::Relaxed);
    }

    #[inline]
    fn irqs_masked() -> bool {
        let daif: u64;
        // SAFETY: reading DAIF is always valid.
        unsafe { asm!("mrs {}, daif", out(reg) daif, options(nomem, nostack, preserves_flags)) };
        (daif >> 7) & 1 != 0 // bit 7 = I (IRQ mask)
    }

    /// Run `f` with IRQs masked, restoring the prior mask state so nested
    /// critical sections compose (the aarch64 analogue of the x86 version).
    pub fn without_interrupts<T>(f: impl FnOnce() -> T) -> T {
        let was_masked = irqs_masked();
        if !was_masked {
            disable();
        }
        let r = f();
        if !was_masked {
            enable();
        }
        r
    }
}

// --- PL011 UART ----------------------------------------------------------
// The base defaults to QEMU `virt`'s 0x09000000 but is overridden at boot from
// the ACPI SPCR table (`init_uart`) so we hit the right MMIO on platforms with
// a different map (e.g. VirtualBox's PL011 at 0xFFDDF000). DR is at base+0x00,
// FR at base+0x18.
use core::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
static UART_BASE: AtomicUsize = AtomicUsize::new(0x0900_0000);
const UART_DR: usize = 0x00;
const UART_FR: usize = 0x18;
const UART_FR_RXFE: u32 = 1 << 4; // receive FIFO empty
const UART_FR_TXFF: u32 = 1 << 5; // transmit FIFO full

// The console UART flavour. QEMU `virt` / SBSA / VirtualBox use a PL011; Apple
// Silicon (booted via m1n1) has no PL011 at all — it uses a Samsung `s5l` UART
// whose base comes from the boot FDT (`apple,s5l-uart`). Selected by
// `init_uart_apple` before the first print; defaults to PL011 everywhere else.
const UART_KIND_PL011: u32 = 0;
const UART_KIND_S5L: u32 = 1;
static UART_KIND: AtomicU32 = AtomicU32::new(UART_KIND_PL011);
// Samsung s5l register offsets + status bits (matches m1n1's `uart.c`).
const S5L_UTRSTAT: usize = 0x10; // TX/RX status
const S5L_UTXH: usize = 0x20; // transmit holding
const S5L_URXH: usize = 0x24; // receive holding
const S5L_UTRSTAT_RXD: u32 = 1 << 0; // RX data ready
const S5L_UTRSTAT_TXBE: u32 = 1 << 1; // TX buffer empty

/// True on Apple Silicon (the boot FDT's root is `apple,arm-platform`). Gates
/// the QEMU-virt MMIO probes (fw_cfg/ramfb, PL031 RTC, virtio-mmio, PL050): on
/// Apple those fixed low addresses are unbacked, and under m1n1's hypervisor a
/// read there is a fatal "unmapped IPA" data abort — whereas QEMU/VBox merely
/// return garbage a magic check rejects. Set by [`init_uart_apple`] (which runs
/// before any device probe) and read across the aarch64 boot path.
static IS_APPLE: AtomicBool = AtomicBool::new(false);

/// Are we on Apple Silicon, per the boot device tree? See [`IS_APPLE`].
pub fn is_apple() -> bool {
    IS_APPLE.load(Ordering::Relaxed)
}

// Recoverable-probe flags for the PL011 MMIO scan (read by the sync handler in
// `exceptions`), so probing an unbacked candidate address can't crash the boot.
static UART_PROBING: AtomicBool = AtomicBool::new(false);
static UART_PROBE_FAULTED: AtomicBool = AtomicBool::new(false);

// Whether the console UART is a *real, verified* PL011 we may read RX from.
// Defaults true (QEMU `virt` `-kernel` has a PL011 at the default base); set
// authoritatively by `init_uart`. On a platform with NO PL011 (e.g. VirtualBox
// with the serial port disabled) reading the phantom base returns garbage whose
// "RX not empty" flag is set, which would flood the shell with spurious
// keystrokes — so `uart_getb` returns `None` unless a PL011 is verified.
static UART_RX_OK: AtomicBool = AtomicBool::new(true);

/// True while probing a candidate UART address (read by the sync handler).
pub fn uart_probing() -> bool {
    UART_PROBING.load(Ordering::Acquire)
}
/// Called by the sync handler when a probed UART read faults.
pub fn note_uart_fault() {
    UART_PROBE_FAULTED.store(true, Ordering::Release);
}

#[inline]
fn uart_base() -> usize {
    UART_BASE.load(Ordering::Relaxed)
}

/// Candidate PL011 base addresses to probe when ACPI SPCR doesn't name one.
/// Each is verified by PrimeCell id before use, so a wrong guess is skipped —
/// this is discovery-by-probe, not per-hypervisor hardcoding. 0x09000000 is
/// QEMU `virt` (the default); 0xFFDDF000 is where VirtualBox-ARM maps its PL011.
const UART_CANDIDATES: [u64; 2] = [0x0900_0000, 0xFFDD_F000];

/// Read a PrimeCell id block (peripheral part @0xFE0, or cell id @0xFF0): four
/// registers, low byte each, assembled little-endian. Same layout as PL050.
unsafe fn primecell_block(base: u64, first: u64) -> u32 {
    use core::ptr::read_volatile;
    unsafe {
        (read_volatile((base + first) as *const u32) & 0xff)
            | ((read_volatile((base + first + 4) as *const u32) & 0xff) << 8)
            | ((read_volatile((base + first + 8) as *const u32) & 0xff) << 16)
            | ((read_volatile((base + first + 12) as *const u32) & 0xff) << 24)
    }
}

/// Is there a PL011 UART at `base`? Checks the PrimeCell id (0xB105_F00D) and
/// peripheral part number (0x011). Reads are guarded by the recoverable sync
/// handler, so an unbacked address just reports "no" instead of faulting.
fn is_pl011(base: u64) -> bool {
    UART_PROBE_FAULTED.store(false, Ordering::Release);
    UART_PROBING.store(true, Ordering::Release);
    core::sync::atomic::compiler_fence(Ordering::SeqCst);
    // SAFETY: reads are recovered by the sync handler if `base` is unbacked.
    let (cell, part) = unsafe { (primecell_block(base, 0xFF0), primecell_block(base, 0xFE0) & 0xfff) };
    core::sync::atomic::compiler_fence(Ordering::SeqCst);
    UART_PROBING.store(false, Ordering::Release);
    !UART_PROBE_FAULTED.load(Ordering::Acquire) && cell == 0xB105_F00D && part == 0x011
}

/// Discover the console UART base and switch the driver to it. First ACPI SPCR
/// (via the stub's boot-info RSDP — the firmware-blessed way); then a
/// PrimeCell-id probe of known PL011 bases (for platforms like VirtualBox that
/// place the PL011 off QEMU's 0x09000000 and provide no SPCR). Maps the chosen
/// base's GiB block as Device. A no-op on the `-kernel` path (no boot-info) and
/// where nothing is found, so QEMU `virt` keeps the 0x09000000 default.
///
/// Must run AFTER the exception vectors are installed (the MMIO probe relies on
/// the recoverable sync handler) and before the first output we want captured.
pub fn init_uart() {
    // Apple's s5l console was already selected from the FDT (init_uart_apple);
    // the PL011 discovery below doesn't apply (Apple has no PL011) and its
    // RX-probe would wrongly disable the working s5l RX.
    if UART_KIND.load(Ordering::Relaxed) == UART_KIND_S5L {
        return;
    }
    let mut chosen: Option<u64> = None;
    // 1. ACPI SPCR, if the boot-info carries an RSDP.
    let bi = boot::boot_x1();
    if bi != 0 {
        // SAFETY: `bi` is the stub's identity-mapped boot-info; check the magic.
        let magic = unsafe { core::slice::from_raw_parts(bi as *const u8, 8) };
        if magic == b"CHITTIBI" {
            let rsdp = unsafe { core::ptr::read_volatile((bi + 40) as *const u64) };
            if let Some((base, iface)) = crate::acpi::uart_from_rsdp(rsdp) {
                if iface == 0x03 || iface == 0x0e {
                    chosen = Some(base); // PL011 / ARM SBSA (shared DR/FR layout)
                }
            }
        }
    }
    // 2. Probe known PL011 bases by PrimeCell id (skip the current default).
    if chosen.is_none() {
        for &base in &UART_CANDIDATES {
            if base != uart_base() as u64 && is_pl011(base) {
                chosen = Some(base);
                break;
            }
        }
    }
    if let Some(base) = chosen {
        if base as usize != uart_base() {
            mmu::map_device_gib(base);
            UART_BASE.store(base as usize, Ordering::Relaxed);
            crate::ktrace::log_fmt(format_args!("uart: PL011 console at {:#x}", base));
        }
    }
    // Serial RX is the QEMU `-kernel` dev console (input on stdio). On the
    // UEFI/stub path (boot-info present — VirtualBox, UTM, real hardware) there is
    // no interactive serial console; input comes from USB/PS-2, and reading a
    // flaky or absent platform UART's RX can flood the shell with garbage
    // "keystrokes". So enable RX only on `-kernel` (no boot-info) and only for a
    // PrimeCell-verified PL011.
    let effective = chosen.unwrap_or(uart_base() as u64);
    let rx_ok = boot::boot_x1() == 0 && is_pl011(effective);
    UART_RX_OK.store(rx_ok, Ordering::Relaxed);
    if !rx_ok {
        crate::ktrace::log_fmt(format_args!("uart: serial RX disabled at {effective:#x} (input via USB/PS-2)"));
    }
}

/// Write one byte to the console (blocks briefly until the TX path is ready).
pub fn uart_putb(byte: u8) {
    let base = uart_base();
    if UART_KIND.load(Ordering::Relaxed) == UART_KIND_S5L {
        // Samsung s5l (Apple): spin until the TX buffer is empty, then write UTXH.
        // SAFETY: `base` is the Device-mapped s5l register block.
        unsafe {
            while core::ptr::read_volatile((base + S5L_UTRSTAT) as *const u32) & S5L_UTRSTAT_TXBE == 0 {}
            core::ptr::write_volatile((base + S5L_UTXH) as *mut u32, byte as u32);
        }
        return;
    }
    // SAFETY: `base` is a Device-mapped PL011 register block; the flag register
    // gates the data write.
    unsafe {
        while core::ptr::read_volatile((base + UART_FR) as *const u32) & UART_FR_TXFF != 0 {}
        core::ptr::write_volatile((base + UART_DR) as *mut u32, byte as u32);
    }
}

/// Read one byte from the console if one is waiting, else `None`.
pub fn uart_getb() -> Option<u8> {
    // No verified console RX => never read (a phantom UART's flags/data are
    // garbage and would flood the shell). See `UART_RX_OK`.
    if !UART_RX_OK.load(Ordering::Relaxed) {
        return None;
    }
    let base = uart_base();
    if UART_KIND.load(Ordering::Relaxed) == UART_KIND_S5L {
        // Samsung s5l: read URXH only when UTRSTAT reports RX data ready.
        // SAFETY: Device-mapped s5l register block.
        unsafe {
            return if core::ptr::read_volatile((base + S5L_UTRSTAT) as *const u32) & S5L_UTRSTAT_RXD != 0 {
                Some((core::ptr::read_volatile((base + S5L_URXH) as *const u32) & 0xff) as u8)
            } else {
                None
            };
        }
    }
    // SAFETY: PL011 MMIO; only reads DR once the "RX empty" flag is clear.
    unsafe {
        if core::ptr::read_volatile((base + UART_FR) as *const u32) & UART_FR_RXFE != 0 {
            None
        } else {
            Some((core::ptr::read_volatile((base + UART_DR) as *const u32) & 0xff) as u8)
        }
    }
}

/// Select the Apple Samsung `s5l` console UART from the boot device tree (m1n1
/// hands the FDT in x0). Apple Silicon has no PL011, so without this the console
/// writes into an unbacked QEMU address and nothing appears. Pure FDT walk — no
/// MMIO probing — so it is safe to call *before* the exception vectors exist
/// (unlike [`init_uart`]'s PL011 probe) and thus before the very first print.
/// A no-op on QEMU/SBSA (no `apple,s5l-uart` node), leaving the PL011 path.
pub fn init_uart_apple() {
    let fdt = boot::boot_x0();
    // SAFETY: `boot_x0` is the FDT pointer (or non-FDT, rejected by the magic).
    IS_APPLE.store(unsafe { crate::fdt::has_compatible(fdt, b"apple,arm-platform") }, Ordering::Relaxed);
    if let Some((base, _size)) = unsafe { crate::fdt::reg_of_compatible(fdt, b"apple,s5l-uart") } {
        mmu::map_device_gib(base);
        UART_BASE.store(base as usize, Ordering::Relaxed);
        UART_KIND.store(UART_KIND_S5L, Ordering::Relaxed);
        UART_RX_OK.store(true, Ordering::Relaxed); // real UART; m1n1 hv forwards RX
        crate::ktrace::log_fmt(format_args!("uart: Apple s5l console at {base:#x}"));
    }
}

// --- generic timer (millisecond clock) -----------------------------------

fn cntvct() -> u64 {
    let v: u64;
    // SAFETY: reading the virtual counter is valid at EL1.
    unsafe { asm!("mrs {}, cntvct_el0", out(reg) v, options(nomem, nostack, preserves_flags)) };
    v
}

/// Raw cycle/tick counter for entropy mixing (see `arch::cycle_count`).
pub fn cycle_count() -> u64 {
    cntvct()
}

/// A hardware random word via `RNDR` (FEAT_RNG), or 0 when the feature is
/// absent (QEMU `virt`/HVF usually lack it). See `arch::hw_rand`.
pub fn hw_rand() -> u64 {
    // ID_AA64ISAR0_EL1.RNDR is bits [63:60]; nonzero => RNDR/RNDRRS present.
    let isar0: u64;
    // SAFETY: reading the ID register is valid at EL1.
    unsafe { asm!("mrs {}, id_aa64isar0_el1", out(reg) isar0, options(nomem, nostack, preserves_flags)) };
    if (isar0 >> 60) & 0xf == 0 {
        return 0;
    }
    let v: u64;
    let ok: u64;
    // SAFETY: RNDR (s3_3_c2_c4_0) is implemented per the ID check above; it sets
    // NZCV — PSTATE.Z=1 means the RNG wasn't ready, so we read the flags and
    // treat a not-ready result as 0 (the caller mixes multiple samples).
    unsafe {
        asm!(
            "mrs {v}, s3_3_c2_c4_0",  // RNDR
            "cset {ok}, ne",          // ok = 1 if Z clear (value valid)
            v = out(reg) v,
            ok = out(reg) ok,
            options(nomem, nostack),
        );
    }
    if ok != 0 {
        v
    } else {
        0
    }
}

fn cntfrq() -> u64 {
    let v: u64;
    // SAFETY: reading the counter frequency register is valid.
    unsafe { asm!("mrs {}, cntfrq_el0", out(reg) v, options(nomem, nostack, preserves_flags)) };
    v
}

/// Milliseconds since boot, from the ARM generic timer (the aarch64 analogue
/// of the x86 PIT tick counter used for inference timing).
///
/// Uses the **virtual** counter `CNTVCT_EL0`, the counter a guest is meant to
/// read: HVF-based hypervisors (QEMU-HVF, VirtualBox) virtualize it against host
/// time, whereas the *physical* counter `CNTPCT_EL0` is frozen/denied to a
/// bare-metal guest on VirtualBox (so the clock — and with it the blinking caret
/// and status-bar datetime — froze). VirtualBox also leaves `CNTFRQ_EL0` reading
/// 0, so fall back to the Apple-Silicon host frequency (24 MHz) then.
pub fn time_ms() -> u64 {
    let f = if cntfrq() != 0 { cntfrq() } else { 24_000_000 };
    cntvct() * 1000 / f
}

/// PSCI `SYSTEM_OFF` (function 0x8400_0008): power the machine off — how a
/// `refcheck` build terminates QEMU on aarch64 (the x86 analogue is the
/// isa-debug-exit device). QEMU `virt` exposes PSCI over HVC to an EL1 guest
/// (the HVF/KVM conduit); platforms that use the SMC conduit fall through to
/// the SMC call; if both return (no PSCI at all), park the core.
pub fn psci_system_off() -> ! {
    // SAFETY: PSCI calls take the function id in x0; per SMCCC the callee may
    // clobber x0-x3 (declared). SYSTEM_OFF does not return on success, and an
    // unimplemented conduit just returns NOT_SUPPORTED in x0 — attempting
    // both conduits is safe.
    unsafe {
        let mut f: u64 = 0x8400_0008;
        asm!("hvc #0", inout("x0") f, lateout("x1") _, lateout("x2") _, lateout("x3") _, options(nomem, nostack));
        f = 0x8400_0008;
        asm!("smc #0", inout("x0") f, lateout("x1") _, lateout("x2") _, lateout("x3") _, options(nomem, nostack));
        let _ = f;
    }
    loop {
        hlt();
    }
}
