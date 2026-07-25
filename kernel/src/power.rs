//! **Suspend and resume** — closing a laptop's lid and having it come back.
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

use alloc::string::String;
use alloc::vec::Vec;

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
fn enter() -> Result<(), String> {
    Err(String::from(
        "the transition itself is not wired up yet -- `/suspend plan` reports what this machine can do",
    ))
}
