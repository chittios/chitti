//! **Intel WiFi bring-up** — mapping the device, waking it, resetting it, and pointing it
//! at firmware.
//!
//! The sequences here are ports of Linux's `iwl_pcie_prepare_card_hw`,
//! `iwl_pcie_apm_init`, `iwl_pcie_sw_reset` and `iwl_pcie_ctxt_info_init`. The pure parts
//! they depend on live next door in [`super::csr`] and [`super::context`], which is where
//! the offsets and layouts are pinned by tests; this file is the ordering and the waiting.
//!
//! ## Every wait is bounded, and says which one gave up
//!
//! A device that never sets `NIC_READY`, never grants MAC access, or never finishes its
//! reset must cost a fixed number of iterations and a log line — not a hung boot. That
//! is the same rule the embedded controller, the HPET probe and the power button follow,
//! and it matters more here because there is no emulated Intel WiFi part anywhere: the
//! first time this code runs will be on somebody's laptop, and the only thing it can
//! offer them is an honest account of how far it got.
//!
//! ## What this does not do
//!
//! It stops after firmware is handed over. There is no receive path, no command
//! round-trip, no 802.11 state machine and no WPA2 — so the radio does not associate and
//! `/wifi connect` still cannot work. Those are further stages; see [`super`].

use super::{context, csr, fw};

#[cfg(target_arch = "x86_64")]
use crate::arch::x86_64::pci::PciDevice;

/// How long to spin on a hardware handshake before giving up. Generous — these are
/// microsecond-scale in hardware — and finite, because the alternative is a boot that
/// stops here on a machine whose radio is asleep.
const HANDSHAKE_SPINS: u32 = 500_000;

/// BAR0 window. The CSRs and the indirect-access registers all live inside it.
const MMIO_SPAN: usize = 0x2000;

/// A mapped, reset Intel WiFi device.
pub struct IwlDevice {
    regs: u64,
    pub family: fw::Family,
    pub hw_rev: u32,
    /// Physical address of the context info handed to the device, once firmware is loaded.
    pub ctxt_phys: u64,
}

fn r32(base: u64, off: usize) -> u32 {
    // SAFETY: `base` is the mapped BAR0 window and `off` is inside `MMIO_SPAN`.
    unsafe { core::ptr::read_volatile((base + off as u64) as *const u32) }
}

fn w32(base: u64, off: usize, v: u32) {
    // SAFETY: as `r32`; these are the device's own control registers.
    unsafe { core::ptr::write_volatile((base + off as u64) as *mut u32, v) };
}

/// Wait for `mask` to reach `want` in the register at `off`.
///
/// Returns false on timeout *or* on a floating read — a BAR that answers all-ones has not
/// answered, and spinning the full budget on it wastes the time twice over.
fn wait_bits(base: u64, off: usize, mask: u32, want: u32) -> bool {
    for _ in 0..HANDSHAKE_SPINS {
        let v = r32(base, off);
        if csr::is_floating(v) {
            return false;
        }
        if v & mask == want {
            return true;
        }
        core::hint::spin_loop();
    }
    false
}

impl IwlDevice {
    /// Read an indirect peripheral register.
    ///
    /// Two writes and a read, not one access: the address goes to one register and the
    /// data comes back from another. Using a `prph` offset directly as a CSR is the
    /// classic porting bug, and it lands in whatever CSR shares that number.
    pub fn prph_read(&self, addr: u32) -> u32 {
        w32(self.regs, csr::HBUS_TARG_PRPH_RADDR, csr::prph_read_addr(addr));
        r32(self.regs, csr::HBUS_TARG_PRPH_RDAT)
    }

    /// Write an indirect peripheral register.
    pub fn prph_write(&self, addr: u32, v: u32) {
        w32(self.regs, csr::HBUS_TARG_PRPH_WADDR, csr::prph_write_addr(addr));
        w32(self.regs, csr::HBUS_TARG_PRPH_WDAT, v);
    }

    /// Take access to the MAC clock domain, which most registers need.
    ///
    /// Ask, then wait for the grant. Proceeding without the grant is not a crash — it is
    /// worse: reads return stale values and writes are dropped, so the bring-up appears
    /// to work and the device never starts.
    fn grab_nic_access(&self) -> bool {
        let v = r32(self.regs, csr::CSR_GP_CNTRL);
        w32(self.regs, csr::CSR_GP_CNTRL, v | csr::GP_CNTRL_MAC_ACCESS_REQ);
        if !wait_bits(
            self.regs,
            csr::CSR_GP_CNTRL,
            csr::GP_CNTRL_MAC_ACCESS_EN,
            csr::GP_CNTRL_MAC_ACCESS_EN,
        ) {
            crate::ktrace::log("iwlwifi", "MAC access was never granted");
            return false;
        }
        true
    }

    /// Tell the device a PCIe host is present and wait for it to be ready.
    fn prepare_card(regs: u64) -> bool {
        let v = r32(regs, csr::CSR_HW_IF_CONFIG_REG);
        if csr::is_floating(v) {
            crate::ktrace::log("iwlwifi", "BAR0 reads all-ones -- the window did not map");
            return false;
        }
        w32(
            regs,
            csr::CSR_HW_IF_CONFIG_REG,
            v | csr::HW_IF_CONFIG_PREPARE,
        );
        // NIC_READY going *clear* is the signal preparation finished — the polarity is
        // worth stating, because waiting for it to set never completes.
        if !wait_bits(regs, csr::CSR_HW_IF_CONFIG_REG, csr::HW_IF_CONFIG_NIC_READY, 0) {
            crate::ktrace::log("iwlwifi", "card never reported ready after PREPARE");
            return false;
        }
        true
    }

    /// The APM (advanced power management) init sequence.
    fn apm_init(regs: u64) -> bool {
        // Disable the L1-active behaviour that can stall DMA mid-transfer. Left enabled,
        // the symptom is an occasional hang rather than a clean failure, which is much
        // harder to attribute.
        let v = r32(regs, csr::CSR_GIO_CHICKEN_BITS);
        w32(
            regs,
            csr::CSR_GIO_CHICKEN_BITS,
            v | csr::GIO_CHICKEN_L1A_NO_L0S_RX,
        );

        // Wake the device out of its low-power state and wait for the clock.
        let v = r32(regs, csr::CSR_GP_CNTRL);
        w32(
            regs,
            csr::CSR_GP_CNTRL,
            (v | csr::GP_CNTRL_INIT_DONE) & !csr::GP_CNTRL_GOING_TO_SLEEP,
        );
        if !wait_bits(
            regs,
            csr::CSR_GP_CNTRL,
            csr::GP_CNTRL_MAC_CLOCK_READY,
            csr::GP_CNTRL_MAC_CLOCK_READY,
        ) {
            crate::ktrace::log("iwlwifi", "MAC clock never came up");
            return false;
        }
        true
    }

    /// Stop the DMA master and software-reset the device.
    ///
    /// Order matters: resetting while the master is still running can leave a DMA in
    /// flight against memory the host is about to reuse.
    fn sw_reset(regs: u64) -> bool {
        let v = r32(regs, csr::CSR_RESET);
        w32(regs, csr::CSR_RESET, v | csr::RESET_STOP_MASTER);
        if !wait_bits(
            regs,
            csr::CSR_RESET,
            csr::RESET_MASTER_DISABLED,
            csr::RESET_MASTER_DISABLED,
        ) {
            crate::ktrace::log("iwlwifi", "DMA master never went idle; not resetting");
            return false;
        }
        w32(regs, csr::CSR_RESET, csr::RESET_SW);
        // Self-clearing.
        if !wait_bits(regs, csr::CSR_RESET, csr::RESET_SW, 0) {
            crate::ktrace::log("iwlwifi", "software reset never completed");
            return false;
        }
        true
    }

    /// Map and reset an identified device, leaving it ready for firmware.
    ///
    /// `None` at any step, each logged. Nothing here is speculative about *what* the
    /// device is — [`fw::family_for`] has already refused anything unrecognised, because
    /// the wrong bring-up path on a WiFi part fails inside the device with no error the
    /// host can read.
    #[cfg(target_arch = "x86_64")]
    pub fn open(d: PciDevice, family: fw::Family) -> Option<IwlDevice> {
        d.enable_bus_master();
        let bar = d.bar(0);
        if bar == 0 {
            crate::ktrace::log("iwlwifi", "BAR0 does not decode as memory");
            return None;
        }
        let regs = crate::mm::map_mmio(bar, MMIO_SPAN);
        if regs == 0 {
            crate::ktrace::log("iwlwifi", "could not map BAR0");
            return None;
        }

        let hw_rev = r32(regs, csr::CSR_HW_REV);
        if !csr::hw_rev_plausible(hw_rev) {
            crate::ktrace::log_fmt(format_args!(
                "iwlwifi: implausible CSR_HW_REV {hw_rev:#010x} -- not proceeding"
            ));
            return None;
        }
        crate::ktrace::log_fmt(format_args!(
            "iwlwifi: {} at BAR0 {bar:#x}, HW_REV {hw_rev:#010x} (step {})",
            family.label(),
            csr::hw_rev_step(hw_rev)
        ));

        // Mask interrupts before anything can raise one: this driver polls, and an
        // unmasked source with no handler installed is a fault, not a missed event.
        w32(regs, csr::CSR_INT_MASK, 0);
        w32(regs, csr::CSR_INT, u32::MAX);

        if !Self::prepare_card(regs) || !Self::apm_init(regs) || !Self::sw_reset(regs) {
            return None;
        }
        // The reset clears the clock state, so bring it back before handing over.
        if !Self::apm_init(regs) {
            return None;
        }
        crate::ktrace::log("iwlwifi", "prepared, APM up, reset complete");

        let dev = IwlDevice {
            regs,
            family,
            hw_rev,
            ctxt_phys: 0,
        };
        if !dev.grab_nic_access() {
            return None;
        }
        Some(dev)
    }

    /// Copy the firmware's sections into DMA memory and hand the device its context info.
    ///
    /// This is the whole gen2 load: the device's own loader reads the image out of host
    /// memory once it has the structure's address. `image` is the parsed `.ucode` and
    /// `blob` the bytes it was parsed from.
    ///
    /// Returns the physical address handed over, or an error naming the step. It does
    /// **not** wait for firmware to come alive — that needs the receive path to see the
    /// alive notification, which does not exist yet, so the honest end of this stage is
    /// "the device has been told where firmware is".
    pub fn load_firmware(
        &mut self,
        image: &fw::FirmwareImage,
        blob: &[u8],
    ) -> Result<u64, &'static str> {
        let rt = image
            .section(fw::TLV_SEC_RT)
            .ok_or("firmware image has no runtime section")?;
        let bytes = blob.get(rt.clone()).ok_or("runtime section out of range")?;

        // One DMA buffer for the firmware itself.
        let (fw_phys, fw_virt) =
            crate::mm::alloc_dma(bytes.len().max(4096)).ok_or("no DMA memory for firmware")?;
        // SAFETY: `fw_virt` maps at least `bytes.len()` freshly-allocated owned bytes.
        unsafe { core::ptr::copy_nonoverlapping(bytes.as_ptr(), fw_virt as *mut u8, bytes.len()) };

        // The section list the device reads to find it. Described by the *copied* length,
        // not the file length — the difference is a DMA read past the buffer.
        let list = context::build_section_list(&[(fw_phys, bytes.len() as u32)]);
        let (list_phys, list_virt) =
            crate::mm::alloc_dma(4096).ok_or("no DMA memory for the section list")?;
        // SAFETY: `list_virt` is a whole owned page and the list is far smaller.
        unsafe {
            core::ptr::copy_nonoverlapping(
                list.as_ptr() as *const u8,
                list_virt as *mut u8,
                list.len() * core::mem::size_of::<context::SectionEntry>(),
            )
        };

        // Rings the device wants addresses for even before it is asked to receive
        // anything: it validates the structure as a whole.
        let (free_rbd, _) = crate::mm::alloc_dma(4096).ok_or("no DMA memory for the free RBD list")?;
        let (used_rbd, _) = crate::mm::alloc_dma(4096).ok_or("no DMA memory for the used RBD list")?;
        let (status, _) = crate::mm::alloc_dma(4096).ok_or("no DMA memory for the status block")?;
        let (cmdq, _) = crate::mm::alloc_dma(4096).ok_or("no DMA memory for the command queue")?;

        let ctxt = context::ContextInfo {
            control: context::ControlBlock {
                version: context::CONTEXT_INFO_VERSION,
                size: core::mem::size_of::<context::ContextInfo>() as u16,
                _rsvd: [0; 3],
            },
            rbd: context::RbdControl {
                rbd_size: 8,
                free_rbd_addr: free_rbd,
                used_rbd_addr: used_rbd,
                status_addr: status,
            },
            tx: context::TxControl {
                cmd_queue_addr: cmdq,
                cmd_queue_size: 4,
                _rsvd: [0; 7],
            },
            fw: context::FwControl {
                img_addr: list_phys,
                img_size: (list.len() * core::mem::size_of::<context::SectionEntry>()) as u32,
                _rsvd: 0,
            },
            ..Default::default()
        };
        // Validate before the device sees it: its answer to a malformed structure is to
        // stop, with nothing reported anywhere.
        context::validate(&ctxt)?;

        let (ctxt_phys, ctxt_virt) =
            crate::mm::alloc_dma(4096).ok_or("no DMA memory for the context info")?;
        // SAFETY: a whole owned page, and `ContextInfo` is far smaller than one.
        unsafe { core::ptr::write(ctxt_virt as *mut context::ContextInfo, ctxt) };

        // Hand it over. gen3 parts take the address in a wider pair of registers than
        // gen2 — writing the gen2 register on a gen3 device leaves the loader looking at
        // an address that was never written.
        match self.family {
            fw::Family::Ax210 | fw::Family::Be200 => {
                w32(self.regs, csr::CSR_CTXT_INFO_ADDR, ctxt_phys as u32);
                w32(self.regs, csr::CSR_CTXT_INFO_ADDR + 4, (ctxt_phys >> 32) as u32);
                w32(self.regs, csr::CSR_IML_DATA_ADDR, list_phys as u32);
                w32(self.regs, csr::CSR_IML_SIZE_ADDR, ctxt.fw.img_size);
            }
            _ => {
                // gen2 takes the address shifted right by 4, and an address with low bits
                // set is not representable at all — encoded and checked in `csr` rather
                // than shifted inline here.
                let ba = csr::ctxt_info_ba(ctxt_phys)
                    .ok_or("context info address is not representable in the gen2 doorbell")?;
                w32(self.regs, csr::CSR_CTXT_INFO_BA, ba);
            }
        }
        self.ctxt_phys = ctxt_phys;
        crate::ktrace::log_fmt(format_args!(
            "iwlwifi: context info at {ctxt_phys:#x}, {} byte runtime section at {fw_phys:#x} -- handed to the device",
            bytes.len()
        ));
        crate::ktrace::log(
            "iwlwifi",
            "no receive path yet, so firmware's alive notification cannot be observed",
        );
        Ok(ctxt_phys)
    }
}
