//! **AGX** — Apple-Silicon GPU bring-up (Milestone 1: the coprocessor).
//!
//! ChittiOS runs LLM inference entirely on the CPU. The M2's real compute
//! horsepower is the **AGX GPU**, reachable only on bare Apple Silicon (via the
//! `cargo xtask m1n1` boot). Before any GPU *compute* can happen, the GPU's
//! control coprocessor (`gfx-asc`) must be booted and driven through Apple's
//! generic **RTKit** handshake to a RUNNING state. That handshake is this
//! module; the compute path (UAT page tables, the firmware command ring, a GEMM
//! microkernel hooked into `cortex`) is future milestones.
//!
//! Layering mirrors the determinism rule of the rest of the kernel:
//! * [`proto`] — the **pure** RTKit/ASC wire protocol (field codecs, version
//!   negotiation, the received-message state machine). Arch-neutral, unit-tested
//!   under `cargo xtask test` (x86, no hardware).
//! * [`asc`] — the **ASC mailbox** MMIO transport (aarch64 only).
//! * this file — discovery (FDT), the minimal PMGR power-domain enable, the
//!   RTKit boot orchestration tying `proto` + `asc` together, and the `/agx`
//!   shell command.
//!
//! Everything hardware-touching is gated on [`arch::aarch64::is_apple`] and an
//! opt-in `chitti.agx` bootarg (same rationale as `chitti.usb` in `apple_usb` —
//! never perturb hardware the m1n1 hypervisor may share); off Apple / on
//! QEMU/VBox the whole thing is a clean no-op.
//!
//! **De-risking note (see the plan):** m1n1 never boots the GPU `gfx-asc` over
//! RTKit, so it is unproven that the control firmware is resident in DRAM at
//! handoff. The boot therefore does `cpu_start` and waits ~1 s for a **HELLO**
//! *first*: HELLO ⇒ firmware resident, milestone achievable; no HELLO ⇒ blocked
//! on external Asahi GPU-firmware provisioning (reported, not silently hung).

pub mod proto;
pub mod uat;

/// ASC mailbox transport — shared by AGX and the SMC client (WiFi power).
#[cfg(target_arch = "aarch64")]
pub mod asc;

#[cfg(target_arch = "aarch64")]
mod handoff;

#[cfg(target_arch = "aarch64")]
mod hw;

/// The `/agx` shell command: `up` runs the coprocessor bring-up, `status` dumps
/// the last result. Off aarch64 (or off Apple) it explains why it's a no-op.
pub fn command(arg: &str) {
    #[cfg(target_arch = "aarch64")]
    {
        hw::command(arg);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = arg;
        crate::serial_println!("agx> Apple AGX GPU is aarch64-only (build/boot the aarch64 kernel on a real M2)");
    }
}
