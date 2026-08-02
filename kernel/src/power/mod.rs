//! **Power management** — CPU idle, suspend/resume.
//!
//! - [`idle`] — `hlt` / `wfi` when the scheduler has no runnable work.
//! - Suspend/resume — closing a laptop's lid and having it come back.
//!
//! Powering *off* already worked ([`crate::arch::poweroff`], ACPI S5). Suspending is a
//! different problem in one specific way: the machine has to come **back**, through a
//! path that runs before anything the kernel normally relies on exists.
//!
//! ## Two mechanisms, one API
//!
//! - **x86: ACPI S3 (suspend-to-RAM).** RAM stays refreshed, almost everything else
//!   loses power. The OS publishes a resume entry point in the FACS **waking vector**,
//!   saves its own CPU state, then writes the `\_S3` sleep type to `PM1a_CNT`. On wake,
//!   firmware jumps to that vector **in real mode**, so resuming means a 16-bit → long
//!   mode trampoline before a single line of Rust can run.
//! - **aarch64: PSCI `SYSTEM_SUSPEND`.** Far less work for the OS, because firmware
//!   saves and restores the CPU context itself: the kernel hands it an entry point and
//!   a context value, and wakes up with registers already restored.
//!
//! They are genuinely different mechanisms rather than one ported to two places, so
//! each arch implements `probe`/`enter` and everything above them is shared.
//!
//! ## Why this reports before it acts
//!
//! A suspend that does not resume is the worst failure this kernel can have: the
//! machine is not damaged, but everything unsaved is gone and the only way out is
//! holding the power button. So [`plan`] is read-only and enumerates every
//! precondition, `/suspend plan` shows it, and the transition refuses unless all of
//! them hold. On a machine that cannot suspend, saying so is the correct outcome.

pub mod idle;

use alloc::string::String;
use alloc::vec::Vec;

// --- PSCI SYSTEM_SUSPEND state layout -------------------------------------
//
// Lives here rather than in `arch::aarch64::suspend` for one specific reason: the unit
// suite (`cargo xtask test`) is an **x86** build, so `arch::aarch64` is cfg-gated out of
// it entirely and anything defined there is untestable. The offsets below are the one
// place the Rust struct and the resume stub's assembly have to agree, which makes them
// exactly the thing that must be covered — so they sit in a module both arches compile.

/// State firmware does not restore. `#[repr(C)]`, and the offsets are mirrored in the
/// assembly — `state_offsets_match_the_assembly` pins them.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct PsciSavedState {
    /// Set by the resume stub, so [`suspend`] can tell a resume from a refusal.
    pub resumed: u64,
    pub mair: u64,
    pub tcr: u64,
    pub ttbr0: u64,
    pub vbar: u64,
    pub cpacr: u64,
    pub sctlr: u64,
    pub sp: u64,
    /// `x19`–`x28`, then `x29` (frame pointer) and `x30` (the return address).
    pub regs: [u64; 12],
}

/// Offsets the assembly uses. Mirrored, not derived — hence the test.
pub const S_RESUMED: usize = 0;
pub const S_MAIR: usize = 8;
pub const S_TCR: usize = 16;
pub const S_TTBR0: usize = 24;
pub const S_VBAR: usize = 32;
pub const S_CPACR: usize = 40;
pub const S_SCTLR: usize = 48;
pub const S_SP: usize = 56;
pub const S_REGS: usize = 64;


/// How this machine suspends, if it can.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SuspendKind {
    /// ACPI S3 via the PM1 control registers and the FACS waking vector.
    AcpiS3,
    /// PSCI `SYSTEM_SUSPEND` — firmware handles the CPU context.
    PsciSystemSuspend,
}

impl SuspendKind {
    /// A short name for a report line.
    pub fn label(&self) -> &'static str {
        match self {
            SuspendKind::AcpiS3 => "ACPI S3 (suspend-to-RAM)",
            SuspendKind::PsciSystemSuspend => "PSCI SYSTEM_SUSPEND",
        }
    }
}

/// The result of examining the machine: what it can do, and what is missing.
#[derive(Debug, Clone)]
pub struct Plan {
    /// The mechanism this machine would use, if any is available.
    pub kind: Option<SuspendKind>,
    /// True only when every precondition holds.
    pub ready: bool,
    /// One line per precondition, in the order they are checked. This is the whole
    /// value of the command on a machine that will not suspend.
    pub report: Vec<String>,
}

impl Plan {
    fn unsupported(reason: &str) -> Plan {
        Plan {
            kind: None,
            ready: false,
            report: alloc::vec![String::from(reason)],
        }
    }
}

/// Examine the machine's suspend support without changing anything.
pub fn plan() -> Plan {
    probe()
}

/// Suspend the machine, returning only once it has resumed.
///
/// `Err` means the transition was **not attempted** — a missing precondition, named.
/// A transition that is attempted and fails to resume does not return at all, which is
/// exactly why [`plan`] exists and why this refuses on anything less than a complete
/// set of preconditions.
pub fn suspend() -> Result<(), String> {
    let p = plan();
    if !p.ready {
        // Hand back the first thing that is missing rather than a generic failure: on a
        // real laptop that line is the entire diagnosis.
        let why = p
            .report
            .iter()
            .find(|l| l.contains("MISSING") || l.contains("no "))
            .cloned()
            .unwrap_or_else(|| String::from("preconditions not met"));
        return Err(why);
    }
    enter()
}

// --- x86: ACPI S3 ---------------------------------------------------------

#[cfg(target_arch = "x86_64")]
fn probe() -> Plan {
    use crate::acpi;
    let mut report = Vec::new();

    let Some(rsdp) = crate::arch::x86_64::rsdp_address() else {
        return Plan::unsupported("acpi: no RSDP -- cannot suspend");
    };
    report.push(alloc::format!("acpi: RSDP at {rsdp:#x}"));

    // The `\_S3` package is the platform's own declaration that it supports
    // suspend-to-RAM; there is no FADT flag for it.
    let sleep = acpi::sleep_from_rsdp(rsdp, 3, |phys| crate::mm::map_mmio(phys, 0x4_0000));
    let Some(sleep) = sleep else {
        report.push(String::from(
            "MISSING _S3: firmware declares no \\_S3 package -- this machine cannot suspend to RAM",
        ));
        return Plan {
            kind: None,
            ready: false,
            report,
        };
    };
    report.push(alloc::format!(
        "_S3:  SLP_TYPa {} SLP_TYPb {} via PM1a_CNT {:#06x}{}",
        sleep.slp_typa,
        sleep.slp_typb,
        sleep.pm1a_cnt,
        if sleep.pm1b_cnt != 0 {
            alloc::format!(" PM1b_CNT {:#06x}", sleep.pm1b_cnt)
        } else {
            String::new()
        }
    ));

    // ACPI mode has to be on, or the sleep-enable write goes to a register firmware
    // still owns.
    // SAFETY: reading the FADT-declared PM1a control port has no memory effect.
    let cnt = unsafe { crate::arch::x86_64::port::inw(sleep.pm1a_cnt) };
    if cnt == 0xffff {
        report.push(String::from(
            "MISSING acpi mode: PM1a_CNT reads as an unclaimed port",
        ));
    } else if cnt & acpi::SCI_EN != 0 {
        report.push(String::from("mode: ACPI mode on (SCI_EN)"));
    } else if sleep.smi_cmd != 0 && sleep.acpi_enable != 0 {
        report.push(alloc::format!(
            "mode: ACPI mode off; would take ownership via SMI_CMD {:#06x}",
            sleep.smi_cmd
        ));
    } else {
        report.push(String::from(
            "MISSING acpi mode: off, and no SMI_CMD handoff to ask for",
        ));
    }

    // The FACS is where the resume entry point gets published. Without it firmware has
    // nowhere to jump and the machine would sleep permanently.
    match acpi::facs_from_rsdp(rsdp, crate::mm::map_mmio) {
        Some(f) => report.push(alloc::format!(
            "facs: at {:#x}, version {}, hardware signature {:#010x}{}",
            f.addr,
            f.version,
            f.hardware_signature,
            if f.has_extended_waking_vector() {
                ""
            } else {
                " (32-bit waking vector only)"
            }
        )),
        None => report.push(String::from(
            "MISSING facs: no valid FACS -- nowhere to publish a resume entry point",
        )),
    }

    // The resume trampoline has to be reachable by a real-mode `CS:IP`, so it must come
    // out of the first MiB — and whether the firmware's memory map leaves anything
    // allocatable down there is a property of the machine, not something the kernel can
    // arrange. Checked *here*, because a plan that reports "ready" and then refuses at
    // the transition is a plan that lied.
    let low = crate::mm::low_memory();
    if let Some((phys, _)) = crate::mm::s3_page() {
        // Reserved at boot, before the heap could take it — which it would have, since
        // the heap maps a gigabyte upward from the lowest free frame.
        report.push(alloc::format!(
            "low:  trampoline page reserved at boot ({phys:#x}); first MiB had {} KiB usable",
            low.usable / 1024
        ));
    } else if let Some(base) = low.reclaimable_base {
        // Boot is long over by the time anyone types `/suspend`, and "reclaimable" is
        // precisely the firmware's statement that this range is the OS's afterwards.
        report.push(alloc::format!(
            "low:  no usable frame below 1 MiB; would reclaim bootloader memory at {base:#x} ({} KiB available)",
            low.reclaimable / 1024
        ));
    } else {
        report.push(alloc::format!(
            "MISSING low memory: the firmware leaves nothing below 1 MiB for the resume trampoline \
             ({} KiB usable, {} KiB reclaimable, {} KiB reserved)",
            low.usable / 1024,
            low.reclaimable / 1024,
            low.other / 1024
        ));
    }

    let ready = !report.iter().any(|l| l.contains("MISSING"));
    Plan {
        kind: Some(SuspendKind::AcpiS3),
        ready,
        report,
    }
}

// --- aarch64: PSCI SYSTEM_SUSPEND -----------------------------------------

#[cfg(target_arch = "aarch64")]
fn probe() -> Plan {
    use crate::arch::aarch64;
    let mut report = Vec::new();

    // Not merely "is there PSCI" — on Apple Silicon the probe call itself would halt
    // the guest, so the gate has to come first and it is the one SMP already uses.
    if !aarch64::psci_available() {
        return Plan::unsupported(
            "psci: no PSCI on this machine (Apple Silicon) -- cannot suspend",
        );
    }
    report.push(String::from("psci: conduit available"));

    let rc = aarch64::psci_features(aarch64::PSCI_SYSTEM_SUSPEND);
    if rc == aarch64::PSCI_NOT_SUPPORTED || rc < 0 {
        report.push(alloc::format!(
            "MISSING psci: firmware does not implement SYSTEM_SUSPEND (PSCI_FEATURES = {rc})"
        ));
        return Plan {
            kind: None,
            ready: false,
            report,
        };
    }
    report.push(alloc::format!(
        "psci: SYSTEM_SUSPEND implemented (PSCI_FEATURES = {rc})"
    ));

    let ready = !report.iter().any(|l| l.contains("MISSING"));
    Plan {
        kind: Some(SuspendKind::PsciSystemSuspend),
        ready,
        report,
    }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn probe() -> Plan {
    Plan::unsupported("suspend: unsupported architecture")
}

/// Perform the transition. Split from [`probe`] so the read-only path can never
/// accidentally enter it.
#[cfg(target_arch = "x86_64")]
fn enter() -> Result<(), String> {
    let rsdp = crate::arch::x86_64::rsdp_address().ok_or_else(|| String::from("no RSDP"))?;
    let sleep = crate::acpi::sleep_from_rsdp(rsdp, 3, |phys| crate::mm::map_mmio(phys, 0x4_0000))
        .ok_or_else(|| String::from("no \\_S3 package"))?;
    let facs = crate::acpi::facs_from_rsdp(rsdp, crate::mm::map_mmio)
        .ok_or_else(|| String::from("no FACS"))?;
    // Ask the touchpad to sleep before the machine does: a device left powered while the
    // bus controller loses its state is the one most likely to come back confused.
    crate::drivers::i2c_hid::suspend();
    crate::arch::x86_64::suspend::suspend(&sleep, &facs).map_err(String::from)?;
    resume_devices();
    Ok(())
}

#[cfg(target_arch = "aarch64")]
fn enter() -> Result<(), String> {
    crate::arch::aarch64::suspend::suspend().map_err(String::from)?;
    resume_devices();
    Ok(())
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn enter() -> Result<(), String> {
    Err(String::from("no suspend mechanism on this architecture"))
}

/// Put back the device state firmware destroyed.
///
/// Deliberately minimal and explicit rather than a general "re-init everything": each
/// call here is something the machine visibly loses across S3, and each is bounded. A
/// broad re-probe would be slower *and* riskier — resume is not a good moment to
/// discover a driver's init path is not idempotent.
///
/// What is **not** restored yet is honestly incomplete: the NIC, xHCI, AHCI/NVMe and the
/// sound device all lose their controller state, and a follow-up gives each driver a
/// `resume` of its own. Until then a resumed machine has a working console and scheduler
/// but may need `/network dhcp` to get back on the network.
fn resume_devices() {
    // aarch64 needs much less: the resume stub already put back the MMU registers, the
    // exception vectors and FP/SIMD, because nothing compiled can run without them. What
    // is left is the interrupt controller and the generic timer, whose programming PSCI
    // does not preserve — so this re-runs the real bring-up rather than only the unmask.
    //
    // Note what the log after a resume does *not* prove: `start_preemption`'s warmup
    // waits for the tick counter to become non-zero, and after a resume it already is
    // (it is a static that survived), so the check passes vacuously. On this path
    // "preemptive scheduling" means "interrupts unmasked", not "timer IRQs observed".
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: at EL1 on the BSP with the vectors restored by the resume stub, which
        // is the same precondition boot has — and `init_bsp` probes the CPU interface
        // through the recoverable sync handler rather than assuming it exists.
        unsafe { crate::arch::aarch64::gic::init_bsp() };
        crate::arch::aarch64::gic::start_preemption();
        crate::ktrace::log("resume", "GIC and generic timer re-programmed");
    }
    #[cfg(target_arch = "x86_64")]
    {
        use crate::arch::x86_64;
        // The local APIC loses its software-enable and its timer; without this the
        // scheduler silently stops preempting, which looks like a machine that resumed
        // but hangs on the first long operation.
        x86_64::apic::software_enable();
        if let Some(rsdp) = x86_64::rsdp_address() {
            x86_64::pit::try_apic_timer(rsdp);
        }
        // The i8042 controller comes back with its configuration byte reset, so a
        // keyboard that worked before suspend types nothing after it.
        x86_64::keyboard::init();
        // And the HID-over-I2C touchpad, which loses its power state the same way — it
        // would otherwise answer register reads and produce no reports.
        crate::drivers::i2c_hid::resume();
        // The trampoline resumed with interrupts masked.
        x86_64::interrupts::enable();
        crate::ktrace::log("resume", "APIC timer, i8042 and interrupts re-armed");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn psci_state_offsets_match_the_assembly() {
        // The resume stub reads every one of these by displacement. A mismatch restores
        // `TCR` from the `MAIR` slot, which is a translation fault the instant the MMU
        // comes on — with no diagnostic, because the vectors are not up yet either.
        assert_eq!(core::mem::offset_of!(PsciSavedState, resumed), S_RESUMED);
        assert_eq!(core::mem::offset_of!(PsciSavedState, mair), S_MAIR);
        assert_eq!(core::mem::offset_of!(PsciSavedState, tcr), S_TCR);
        assert_eq!(core::mem::offset_of!(PsciSavedState, ttbr0), S_TTBR0);
        assert_eq!(core::mem::offset_of!(PsciSavedState, vbar), S_VBAR);
        assert_eq!(core::mem::offset_of!(PsciSavedState, cpacr), S_CPACR);
        assert_eq!(core::mem::offset_of!(PsciSavedState, sctlr), S_SCTLR);
        assert_eq!(core::mem::offset_of!(PsciSavedState, sp), S_SP);
        assert_eq!(core::mem::offset_of!(PsciSavedState, regs), S_REGS);
    }

    #[test_case]
    fn the_psci_register_area_holds_every_callee_saved_register() {
        // x19-x28, plus the frame pointer and the return address: six `stp` pairs, and
        // the stub's `ldp` offsets run to S_REGS + 80. One short and the resume would
        // return to a stale address.
        assert_eq!(core::mem::size_of::<[u64; 12]>(), 96);
        assert_eq!(S_REGS + 80 + 16, core::mem::size_of::<PsciSavedState>());
    }

    #[test_case]
    fn psci_stp_offsets_stay_within_the_immediate_range() {
        // `stp`/`ldp` scale their immediate by 8 and allow -512..504. Every offset the
        // assembly uses has to fit, or it does not assemble — but it also must not be
        // silently rewritten into a different addressing form on a future edit.
        for off in [S_REGS, S_REGS + 16, S_REGS + 32, S_REGS + 48, S_REGS + 64, S_REGS + 80] {
            assert!(off % 8 == 0 && off <= 504, "stp offset {off} out of range");
        }
    }
}
