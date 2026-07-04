//! aarch64 architecture support (Phase 7): the native Apple-Silicon port.
//!
//! Booted directly by `qemu-system-aarch64 -M virt -kernel` (no Limine); under
//! `-accel hvf` the guest runs natively on the M-series CPU. This module
//! provides the same facade the rest of the kernel expects from an arch
//! (`interrupts`, `hlt`) plus the aarch64-specific bring-up (boot stub, MMU,
//! PL011 UART, generic timer) the shared upper layers build on.

pub mod boot;
pub mod mmu;
pub mod ahci;
pub mod disk;
pub mod exceptions;
pub mod gic;
pub mod nvme;
pub mod pl050;
pub mod ramfb;
pub mod rtc;
pub mod smp;
pub mod virtio_blk;
pub mod virtio_input;
pub mod virtio_pointer;
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
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
static UART_BASE: AtomicUsize = AtomicUsize::new(0x0900_0000);
const UART_DR: usize = 0x00;
const UART_FR: usize = 0x18;
const UART_FR_RXFE: u32 = 1 << 4; // receive FIFO empty
const UART_FR_TXFF: u32 = 1 << 5; // transmit FIFO full

// Recoverable-probe flags for the PL011 MMIO scan (read by the sync handler in
// `exceptions`), so probing an unbacked candidate address can't crash the boot.
static UART_PROBING: AtomicBool = AtomicBool::new(false);
static UART_PROBE_FAULTED: AtomicBool = AtomicBool::new(false);

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
}

/// Write one byte to the console (blocks briefly if the TX FIFO is full).
pub fn uart_putb(byte: u8) {
    let base = uart_base();
    // SAFETY: `base` is a Device-mapped PL011 register block; the flag register
    // gates the data write.
    unsafe {
        while core::ptr::read_volatile((base + UART_FR) as *const u32) & UART_FR_TXFF != 0 {}
        core::ptr::write_volatile((base + UART_DR) as *mut u32, byte as u32);
    }
}

/// Read one byte from the console if the RX FIFO has one, else `None`.
pub fn uart_getb() -> Option<u8> {
    let base = uart_base();
    // SAFETY: PL011 MMIO; only reads DR once the "RX empty" flag is clear.
    unsafe {
        if core::ptr::read_volatile((base + UART_FR) as *const u32) & UART_FR_RXFE != 0 {
            None
        } else {
            Some((core::ptr::read_volatile((base + UART_DR) as *const u32) & 0xff) as u8)
        }
    }
}

// --- generic timer (millisecond clock) -----------------------------------

fn cntpct() -> u64 {
    let v: u64;
    // SAFETY: reading the physical counter is valid at EL1.
    unsafe { asm!("mrs {}, cntpct_el0", out(reg) v, options(nomem, nostack, preserves_flags)) };
    v
}

fn cntfrq() -> u64 {
    let v: u64;
    // SAFETY: reading the counter frequency register is valid.
    unsafe { asm!("mrs {}, cntfrq_el0", out(reg) v, options(nomem, nostack, preserves_flags)) };
    v
}

/// Milliseconds since boot, from the ARM generic timer (the aarch64 analogue
/// of the x86 PIT tick counter used for inference timing).
pub fn time_ms() -> u64 {
    let f = cntfrq();
    if f == 0 {
        0
    } else {
        cntpct() * 1000 / f
    }
}
