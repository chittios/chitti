//! Apple **PMGR** power domains, addressed by the name the device tree gives
//! them.
//!
//! Every block in the audio path is power-gated at reset — the I2C controller,
//! the DMA engine, the I2S clusters — and reading a gated block's registers
//! returns nothing useful while writing to one does nothing at all. So this is
//! the first step of the bring-up and the one whose failure explains every later
//! symptom.
//!
//! **The domain is found by `label`, never by index.** A t8112 has around ninety
//! `apple,pmgr-pwrstate` children under the PMGR, distinguished only by that
//! string (`audio_p`, `sio_adma`, `mca0`…), and each one's `reg` is its
//! *offset within the PMGR*, not an address. Picking by position would depend on
//! the order a tree happens to list them in.
//!
//! Ported from m1n1 `src/pmgr.c` (`pmgr_set_mode`) and the existing in-kernel
//! [`crate::agx`] bring-up, which does the same dance against one hardcoded
//! address; this is that logic with the address discovered.

/// Bits cleared to bring a domain up: `AUTO_PM_EN(28) | WAS_CLKGATED(9) |
/// WAS_PWRGATED(8) | TARGET(3:0)` — m1n1's `PMGR_PS_*`. Target 0xf is ACTIVE, so
/// "clear then set the target" is how a domain is asked for.
const PMGR_CLEAR: u32 = (1 << 28) | (1 << 9) | (1 << 8) | 0xf;
/// Target power state field, and the value meaning "on".
const PS_TARGET_ACTIVE: u32 = 0xf;
/// Where the hardware reports the state it has actually reached (bits 7:4).
const PS_ACTUAL_SHIFT: u32 = 4;

/// How long to wait for a domain to report ACTIVE. Domains come up in
/// microseconds; this is a bound, not an expectation.
const TIMEOUT_MS: u64 = 50;

/// The value to write to a pwrstate register to request ACTIVE, given what it
/// currently reads.
///
/// Pure so the bit twiddling is testable — the failure mode of getting it wrong
/// is a domain that stays gated and a driver that then reads all-ones from a
/// block that is simply switched off, which looks like missing hardware.
pub fn active_request(current: u32) -> u32 {
    (current & !PMGR_CLEAR) | PS_TARGET_ACTIVE
}

/// Whether a pwrstate register reads as fully on.
pub fn is_active(v: u32) -> bool {
    (v >> PS_ACTUAL_SHIFT) & 0xf == PS_TARGET_ACTIVE
}

#[cfg(target_arch = "aarch64")]
mod hw {
    use super::*;

    /// Read the PMGR block's base from the device tree.
    fn pmgr_base() -> Option<u64> {
        let dtb = crate::arch::aarch64::boot::boot_x0();
        // SAFETY: `boot_x0` is the FDT pointer (or not an FDT, rejected by magic).
        let (base, size) = unsafe { crate::fdt::reg_of_compatible(dtb, b"apple,pmgr") }?;
        Some(crate::mm::map_mmio(base, size as usize))
    }

    /// Bring up the power domain a **phandle** names, and its parents first.
    ///
    /// This is the form that actually works, and the label form below is the
    /// convenience wrapper. A device node states its own domains
    /// (`power-domains = <&ps_audio_p &ps_mca0 …>`), so nothing has to know a
    /// machine's names — and a domain whose *parent* is still gated will not come
    /// up, which is why this walks up first. Depth-capped because the recursion
    /// is over data from outside this kernel.
    pub fn enable_phandle(ph: u32, depth: u32) -> bool {
        if depth > 4 {
            return false;
        }
        let dtb = crate::arch::aarch64::boot::boot_x0();
        // Parents first.
        let mut parents = [0u32; 4];
        // SAFETY: `boot_x0` is the FDT pointer (or not an FDT, rejected by magic).
        let n = unsafe { crate::fdt::prop_cells_by_phandle(dtb, ph, b"power-domains", &mut parents) };
        for p in &parents[..n.min(parents.len())] {
            if *p != 0 && *p != ph {
                enable_phandle(*p, depth + 1);
            }
        }
        // SAFETY: as above.
        let Some((off, _)) = (unsafe { crate::fdt::reg_by_phandle(dtb, ph) }) else {
            crate::ktrace::log_fmt(format_args!("audio: pmgr: phandle {ph:#x} names no node"));
            return false;
        };
        set_active(off, ph)
    }

    /// Bring up every power domain the node with `compat` declares.
    ///
    /// **A block whose domain is off does not read as zero, it does not decode
    /// at all** — the access takes a synchronous external abort, which on this
    /// machine is a fault rather than a bad value. So this runs before the first
    /// register touch of every block, and the DART in front of the DMA engine is
    /// as much a block as the engine is: reading its lock register with its
    /// domain gated is exactly how the first bring-up faulted.
    pub fn enable_domains_of(compat: &[u8]) -> bool {
        let dtb = crate::arch::aarch64::boot::boot_x0();
        let mut doms = [0u32; 8];
        // SAFETY: `boot_x0` is the FDT pointer (or not an FDT, rejected by magic).
        let n = unsafe { crate::fdt::prop_cells_of_compatible(dtb, compat, b"power-domains", &mut doms) };
        // No `power-domains` is not a failure: a clock generator inside an
        // already-powered block legitimately declares none.
        doms[..n.min(doms.len())].iter().all(|&d| d == 0 || enable_phandle(d, 0))
    }

    /// Bring the power domain named `label` up, and report whether it reached
    /// ACTIVE. A domain that is already on is a no-op that returns `true`.
    pub fn enable(label: &[u8]) -> bool {
        if pmgr_base().is_none() {
            crate::ktrace::log("audio", "pmgr: no apple,pmgr node in the device tree");
            return false;
        }
        let dtb = crate::arch::aarch64::boot::boot_x0();
        // SAFETY: as above.
        let Some((off, _)) = (unsafe { crate::fdt::reg_of_labeled_node(dtb, b"apple,pmgr-pwrstate", label) })
        else {
            crate::ktrace::log_fmt(format_args!(
                "audio: pmgr: no power domain labelled '{}'",
                core::str::from_utf8(label).unwrap_or("?")
            ));
            return false;
        };
        set_active(off, 0)
    }

    /// Ask the domain at PMGR offset `off` for ACTIVE and wait for it. `who` is
    /// only for the log line.
    fn set_active(off: u64, who: u32) -> bool {
        let Some(base) = pmgr_base() else { return false };
        let reg = base + off;
        // SAFETY: `reg` is a PMGR pwrstate register, inside the block the device
        // tree sized; single 32-bit accesses, the aarch64 MMIO rule.
        unsafe {
            let cur = read32(reg);
            if is_active(cur) {
                return true;
            }
            write32(reg, active_request(cur));
        }
        let deadline = crate::arch::now_ms() + TIMEOUT_MS;
        while crate::arch::now_ms() < deadline {
            // SAFETY: as above.
            if is_active(unsafe { read32(reg) }) {
                return true;
            }
            core::hint::spin_loop();
        }
        // SAFETY: as above.
        let v = unsafe { read32(reg) };
        crate::ktrace::log_fmt(format_args!(
            "audio: pmgr: domain at pmgr+{off:#x} (phandle {who:#x}) did not reach ACTIVE, reads {v:#x}"
        ));
        false
    }

    /// # Safety
    /// `addr` must be a mapped device register.
    unsafe fn read32(addr: u64) -> u32 {
        let v: u32;
        unsafe { core::arch::asm!("ldr {0:w}, [{1}]", out(reg) v, in(reg) addr, options(nostack)) };
        v
    }

    /// # Safety
    /// `addr` must be a mapped device register.
    unsafe fn write32(addr: u64, v: u32) {
        unsafe { core::arch::asm!("str {0:w}, [{1}]", in(reg) v, in(reg) addr, options(nostack)) };
    }
}

#[cfg(target_arch = "aarch64")]
pub use hw::{enable, enable_domains_of, enable_phandle};

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn requesting_active_clears_the_gate_history_and_keeps_the_rest() {
        // A gated domain: target 0 (off), and the "was clock/power gated" bits
        // set. The request must clear those and ask for target 0xf, while
        // leaving every unrelated bit exactly as the hardware had it.
        let gated = (1 << 28) | (1 << 9) | (1 << 8) | (1 << 20) | 0x0;
        let req = active_request(gated);
        assert_eq!(req & 0xf, 0xf, "target ACTIVE");
        assert_eq!(req & (1 << 28), 0, "auto-PM cleared");
        assert_eq!(req & (1 << 9), 0);
        assert_eq!(req & (1 << 8), 0);
        assert_eq!(req & (1 << 20), 1 << 20, "unrelated bits preserved");
    }

    #[test_case]
    fn active_is_read_from_the_actual_field_not_the_target() {
        // The distinction that matters: writing a target does not make a domain
        // on. Bits 7:4 are what the hardware reached, 3:0 what was asked for —
        // reading the request back would report success the instant it was made.
        assert!(is_active(0xff));
        assert!(is_active(0x00f0));
        assert!(!is_active(0x000f), "asked for on, not yet on");
        assert!(!is_active(0));
    }
}
