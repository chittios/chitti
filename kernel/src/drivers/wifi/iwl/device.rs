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
//! It carries commands and notifications, and stops there. The firmware **configuration**
//! commands a scan needs are not here — see [`super`] for why guessing their layouts would
//! be worse than their absence.

use super::{context, csr, fw, proto};

// Same API either side — see the note in [`super`] on why nothing here is arch-gated.
#[cfg(target_arch = "x86_64")]
use crate::arch::x86_64::pci::PciDevice;
#[cfg(target_arch = "aarch64")]
use crate::pci::PciDevice;

/// How long to spin on a hardware handshake before giving up. Generous — these are
/// microsecond-scale in hardware — and finite, because the alternative is a boot that
/// stops here on a machine whose radio is asleep.
const HANDSHAKE_SPINS: u32 = 500_000;

/// BAR0 window. The CSRs and the indirect-access registers all live inside it.
const MMIO_SPAN: usize = 0x2000;

/// Number of receive buffers handed to the device. Small on purpose: this path only has
/// to catch firmware's startup notifications, not carry traffic.
const NRX: usize = 16;
/// Size of each. Firmware's notifications are far smaller, but the device is told the
/// buffer order, so this has to be a real page.
const RX_BUF: usize = 4096;

/// Offset of `closed_rb_num` in the device's status block, and the mask that makes it an
/// index. The upper bits are not part of the number — used unmasked, the index runs off
/// the end of the ring immediately.
const RB_STATUS_CLOSED: usize = 0;
const RB_CLOSED_MASK: u16 = 0x0fff;

/// Slots in the command queue. A power of two, because the device wraps the index by
/// masking — and 16 descriptors of 256 bytes is exactly one page.
const NCMD: usize = 16;
/// Bytes reserved per command.
///
/// `SCAN_REQ_UMAC` v17 is a 1,940-byte command.  The original 256-byte slot
/// was enough for bring-up and NVM queries, but made the fully-built scan
/// request impossible to submit: `send_cmd` correctly refused it before the
/// radio ever saw it.  Keep a fixed, DMA-safe stride and reject anything that
/// does not fit rather than truncating a command on the air interface.
const CMD_SLOT: usize = 2048;
/// The command queue's id. Queue 0 on gen2 parts, and it appears in two unrelated places —
/// the doorbell and every sequence number — which is why it is one constant.
const CMD_QUEUE_ID: u8 = 0;

/// The receive ring, kept so notifications can be read after firmware starts.
struct Rings {
    /// Virtual addresses of the receive buffers, indexed as the device indexes them.
    bufs: [u64; NRX],
    /// Physical addresses of the same, needed to hand a buffer back after reading it.
    bufs_phys: [u64; NRX],
    /// Virtual address of the status block the device writes its progress into.
    status: u64,
    /// Next buffer this driver has not yet looked at.
    read: usize,
    /// Virtual address of the free-buffer list, and how far the host has filled it.
    free_list: u64,
    free_widx: usize,

    /// Command queue: descriptor ring, per-slot command bytes, per-slot staging buffer.
    cmd_tfd: u64,
    cmd_buf_virt: u64,
    cmd_buf_phys: u64,
    first_tb_virt: u64,
    first_tb_phys: u64,
    queue: proto::CmdQueue,
}

/// A mapped, reset Intel WiFi device.
pub struct IwlDevice {
    regs: u64,
    pub family: fw::Family,
    pub hw_rev: u32,
    /// Physical address of the context info handed to the device, once firmware is loaded.
    pub ctxt_phys: u64,
    rings: Option<Rings>,
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
        w32(
            self.regs,
            csr::HBUS_TARG_PRPH_RADDR,
            csr::prph_read_addr(addr),
        );
        r32(self.regs, csr::HBUS_TARG_PRPH_RDAT)
    }

    /// Write an indirect peripheral register.
    pub fn prph_write(&self, addr: u32, v: u32) {
        w32(
            self.regs,
            csr::HBUS_TARG_PRPH_WADDR,
            csr::prph_write_addr(addr),
        );
        w32(self.regs, csr::HBUS_TARG_PRPH_WDAT, v);
    }

    /// Take access to the MAC clock domain, which most registers need.
    ///
    /// Ask, then wait for the grant. Proceeding without the grant is not a crash — it is
    /// worse: reads return stale values and writes are dropped, so the bring-up appears
    /// to work and the device never starts.
    fn grab_nic_access(&self) -> bool {
        let v = r32(self.regs, csr::CSR_GP_CNTRL);
        w32(
            self.regs,
            csr::CSR_GP_CNTRL,
            v | csr::GP_CNTRL_MAC_ACCESS_REQ,
        );
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
        if !wait_bits(
            regs,
            csr::CSR_HW_IF_CONFIG_REG,
            csr::HW_IF_CONFIG_NIC_READY,
            0,
        ) {
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
            rings: None,
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

        // Real receive buffers, not placeholders: firmware's first act after loading is to
        // send a notification, and without somewhere to put it a successful load and a
        // failed one are indistinguishable from here.
        let mut rx_virt = [0u64; NRX];
        let mut rx_phys = [0u64; NRX];
        for i in 0..NRX {
            let (p, v) =
                crate::mm::alloc_dma(RX_BUF).ok_or("no DMA memory for a receive buffer")?;
            rx_phys[i] = p;
            rx_virt[i] = v;
        }
        let list = proto::build_rbd_list(&rx_phys).ok_or("receive buffer list rejected")?;

        let (free_rbd, free_virt) =
            crate::mm::alloc_dma(4096).ok_or("no DMA memory for the free RBD list")?;
        // SAFETY: `free_virt` is a whole owned page; the list is NRX u64s, far smaller.
        unsafe { core::ptr::copy_nonoverlapping(list.as_ptr(), free_virt as *mut u64, list.len()) };
        let (used_rbd, _) =
            crate::mm::alloc_dma(4096).ok_or("no DMA memory for the used RBD list")?;
        let (status, status_virt) =
            crate::mm::alloc_dma(4096).ok_or("no DMA memory for the status block")?;

        // The command queue: one page of descriptors, one page of command bytes, and one
        // page of first-transfer-buffer staging. Three allocations rather than one because
        // the device DMAs from all three independently and the staging buffers have their
        // own alignment.
        let (cmdq, cmdq_virt) =
            crate::mm::alloc_dma(NCMD * 256).ok_or("no DMA memory for the command queue")?;
        let (cmd_buf_phys, cmd_buf_virt) =
            crate::mm::alloc_dma(NCMD * CMD_SLOT).ok_or("no DMA memory for command payloads")?;
        let (first_tb_phys, first_tb_virt) = crate::mm::alloc_dma(NCMD * proto::FIRST_TB_ALIGN)
            .ok_or("no DMA memory for the command staging buffers")?;
        let queue =
            proto::CmdQueue::new(CMD_QUEUE_ID, NCMD).ok_or("command queue size rejected")?;

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
                cmd_queue_size: queue.size_log2(),
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
                w32(
                    self.regs,
                    csr::CSR_CTXT_INFO_ADDR + 4,
                    (ctxt_phys >> 32) as u32,
                );
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
        self.rings = Some(Rings {
            bufs: rx_virt,
            bufs_phys: rx_phys,
            status: status_virt,
            read: 0,
            free_list: free_virt,
            // Every buffer is already in the list, so the device may fill all of them.
            free_widx: NRX,
            cmd_tfd: cmdq_virt,
            cmd_buf_virt,
            cmd_buf_phys,
            first_tb_virt,
            first_tb_phys,
            queue,
        });
        crate::ktrace::log_fmt(format_args!(
            "iwlwifi: context info at {ctxt_phys:#x}, {} byte runtime section at {fw_phys:#x} -- handed to the device",
            bytes.len()
        ));
        Ok(ctxt_phys)
    }
}

/// One notification, copied out of the ring.
///
/// Owned rather than borrowed on purpose: the buffer it came from is handed straight back to
/// the device, which may overwrite it before the caller has finished looking.
#[derive(Debug, Clone)]
pub struct Notification {
    pub group_id: u8,
    pub cmd: u8,
    pub sequence: u16,
    pub payload: alloc::vec::Vec<u8>,
}

impl IwlDevice {
    /// How long to wait for firmware to say something. Firmware answers in milliseconds;
    /// this is generous and finite.
    const ALIVE_SPINS: u32 = 2_000_000;
    /// How long to wait for a command's response, once firmware is running.
    const RESPONSE_SPINS: u32 = 1_000_000;

    /// Take the next notification the device has finished writing, if any.
    ///
    /// Non-blocking, and it hands each buffer back after reading — without that the ring is
    /// a one-off budget of sixteen notifications, which is enough for startup and nowhere
    /// near enough for a scan, where every beacon is a notification. The symptom of not
    /// recycling is a driver that works during bring-up and goes deaf under traffic.
    pub fn poll_notification(&mut self) -> Option<Notification> {
        let regs = self.regs;
        let rings = self.rings.as_mut()?;
        // SAFETY: the status block is a DMA page this driver owns, and the device writes
        // the closed index into its first halfword.
        let closed = unsafe {
            core::ptr::read_volatile((rings.status + RB_STATUS_CLOSED as u64) as *const u16)
        } & RB_CLOSED_MASK;
        let closed = closed as usize % NRX;
        if rings.read == closed {
            return None;
        }
        let idx = rings.read;
        rings.read = (rings.read + 1) % NRX;

        // SAFETY: `bufs[idx]` maps a whole owned receive buffer of `RX_BUF` bytes.
        let buf = unsafe { core::slice::from_raw_parts(rings.bufs[idx] as *const u8, RX_BUF) };
        let parsed = proto::parse_rx(buf).map(|p| Notification {
            group_id: p.group_id,
            cmd: p.cmd,
            sequence: p.sequence,
            payload: p.payload.to_vec(),
        });

        // Republish the buffer and tell the device the list grew. The write index is the
        // *count* of buffers published, not a slot number, so it keeps rising and the device
        // masks it — publishing the slot index instead makes the device believe the list
        // never grows past the ring size and it stops receiving.
        let slot = rings.free_widx % NRX;
        // SAFETY: `free_list` is an owned DMA page holding NRX u64 entries.
        unsafe {
            core::ptr::write_volatile(
                (rings.free_list as *mut u64).add(slot),
                rings.bufs_phys[idx],
            )
        };
        rings.free_widx = rings.free_widx.wrapping_add(1);
        let widx = (rings.free_widx & 0xffff) as u32;
        // Written through the peripheral window, which is a different register space from
        // the transmit doorbell — see `csr::frbdcb_widx`.
        w32(
            regs,
            csr::HBUS_TARG_PRPH_WADDR,
            csr::prph_write_addr(csr::frbdcb_widx(0)),
        );
        w32(regs, csr::HBUS_TARG_PRPH_WDAT, widx);

        parsed
    }

    /// Send a command and return the sequence firmware will echo on its response.
    ///
    /// The command is placed in its slot, split across two transfer buffers the way the
    /// device fetches it, and the doorbell rung. It does **not** wait — a caller that needs
    /// the answer follows with [`Self::wait_for_response`], and a caller that does not
    /// (there are such commands) is not made to.
    pub fn send_cmd(&mut self, group: u8, cmd: u8, payload: &[u8]) -> Result<u16, &'static str> {
        let regs = self.regs;
        let rings = self
            .rings
            .as_mut()
            .ok_or("no command queue; load firmware first")?;
        // Check the size before claiming a slot, so an over-long command does not consume
        // one and leave it in flight forever.
        if proto::CMD_HEADER_WIDE_LEN + payload.len() > CMD_SLOT {
            return Err("command payload does not fit its queue slot");
        }
        // Claiming can fail, and it must be allowed to: reusing a slot whose command the
        // device has not read loses that command silently.
        let (slot, seq) = rings
            .queue
            .claim()
            .ok_or("command queue is full; a previous command never answered")?;
        // Built after claiming, because the sequence number *is* the slot — it cannot be
        // chosen before there is a slot to name.
        let bytes = proto::build_command(group, cmd, seq, payload);

        let split = proto::split_command(bytes.len());
        let first_virt = rings.first_tb_virt + (slot * proto::FIRST_TB_ALIGN) as u64;
        let first_phys = rings.first_tb_phys + (slot * proto::FIRST_TB_ALIGN) as u64;
        let rest_virt = rings.cmd_buf_virt + (slot * CMD_SLOT) as u64;
        let rest_phys = rings.cmd_buf_phys + (slot * CMD_SLOT) as u64;

        // SAFETY: both are owned DMA regions with at least this slot's stride available, and
        // the split is bounded by `bytes.len()`.
        unsafe {
            core::ptr::copy_nonoverlapping(bytes.as_ptr(), first_virt as *mut u8, split.first);
            if split.rest > 0 {
                core::ptr::copy_nonoverlapping(
                    bytes.as_ptr().add(split.first),
                    rest_virt as *mut u8,
                    split.rest,
                );
            }
        }

        let mut bufs = alloc::vec![(first_phys, split.first as u16)];
        if split.rest > 0 {
            // The remainder was copied to the start of the slot, so that is where the
            // device is pointed — not `+ first`, which is where those bytes live in the
            // command, not in memory.
            bufs.push((rest_phys, split.rest as u16));
        }
        let tfd = proto::build_tfd(&bufs).ok_or("command descriptor rejected")?;
        // SAFETY: the descriptor ring is an owned DMA region of NCMD entries and `slot` is
        // inside it; `Tfd` is exactly the layout the device reads.
        unsafe { core::ptr::write((rings.cmd_tfd as *mut proto::Tfd).add(slot), tfd) };

        let doorbell = csr::txq_doorbell(rings.queue.id, rings.queue.write_index());
        // The descriptor and the command bytes must be visible before the doorbell, or the
        // device fetches whatever the slot held last time.
        core::sync::atomic::fence(core::sync::atomic::Ordering::SeqCst);
        w32(regs, csr::HBUS_TARG_WRPTR, doorbell);
        Ok(seq)
    }

    /// Wait for the response to `seq`, returning its payload.
    ///
    /// Notifications arriving in the meantime are logged and dropped — except firmware's
    /// error notification, which is reported as such: a command that will never be answered
    /// because firmware has died should not be reported as a timeout.
    pub fn wait_for_response(
        &mut self,
        group: u8,
        cmd: u8,
        seq: u16,
    ) -> Result<alloc::vec::Vec<u8>, &'static str> {
        for _ in 0..Self::RESPONSE_SPINS {
            if let Some(n) = self.poll_notification() {
                if n.group_id == proto::GROUP_LEGACY && n.cmd == proto::UCODE_ERROR_NTFY {
                    crate::ktrace::log(
                        "iwlwifi",
                        "firmware reported an error while a command was in flight",
                    );
                    return Err("firmware failed while a command was in flight");
                }
                if n.group_id == group && n.cmd == cmd && n.sequence == seq {
                    if let Some(rings) = self.rings.as_mut() {
                        rings.queue.retire(seq);
                    }
                    return Ok(n.payload);
                }
                crate::ktrace::log_fmt(format_args!(
                    "iwlwifi: notification group {:#02x} cmd {:#02x} seq {:#06x} while awaiting {group:#02x}/{cmd:#02x}",
                    n.group_id, n.cmd, n.sequence
                ));
                continue;
            }
            core::hint::spin_loop();
        }
        Err("no response to the command")
    }

    /// Send a command and wait for its reply.
    pub fn cmd(
        &mut self,
        group: u8,
        cmd: u8,
        payload: &[u8],
    ) -> Result<alloc::vec::Vec<u8>, &'static str> {
        let seq = self.send_cmd(group, cmd, payload)?;
        self.wait_for_response(group, cmd, seq)
    }

    /// Read the device's own MAC address.
    ///
    /// Strap registers first, then OTP: either can be blank depending on how the board was
    /// provisioned, and preferring one unconditionally yields an all-zero address on half the
    /// machines. `None` when neither holds one, which is worth reporting rather than papering
    /// over — an interface with no address cannot associate, and a made-up one would fail
    /// later and further away.
    pub fn read_mac(&self) -> Option<[u8; 6]> {
        for (w0, w1) in [
            (csr::CSR_MAC_ADDR0_STRAP, csr::CSR_MAC_ADDR1_STRAP),
            (csr::CSR_MAC_ADDR0_OTP, csr::CSR_MAC_ADDR1_OTP),
        ] {
            if let Some(mac) = csr::mac_from_words(r32(self.regs, w0), r32(self.regs, w1)) {
                return Some(mac);
            }
        }
        None
    }

    /// Ask firmware for the NVM summary — the first command this driver sends.
    ///
    /// Chosen because it is **read-only**. The transport has never run against a real device,
    /// and a first command that configures something would, if any part of it is wrong,
    /// misconfigure a radio; this one can at worst be answered with an error. What comes back
    /// is only decoded as far as [`proto::NvmInfo`] goes.
    pub fn nvm_info(&mut self) -> Result<proto::NvmInfo, &'static str> {
        let payload = self.cmd(proto::GROUP_REGULATORY_NVM, proto::NVM_GET_INFO, &[])?;
        proto::NvmInfo::parse(&payload).ok_or("the NVM response did not decode as NVM information")
    }

    /// Wait for firmware's *alive* notification.
    ///
    /// This is what makes a firmware load checkable. Before it existed, a load that
    /// silently failed and one that worked produced exactly the same host-side outcome —
    /// the address was written, and nothing more could be said. Now the device either
    /// answers or it does not, and either way the log says which.
    ///
    /// The notification's own status word is checked too, because firmware that comes up
    /// *unusable* still announces itself, and taking the announcement alone as success means
    /// the next command is sent into a dead device.
    /// Start a scan.
    ///
    /// Consults the firmware's own `IWL_UCODE_TLV_CMD_VERSIONS` table first and
    /// **refuses** rather than transmitting when it names a request layout this
    /// driver does not implement — see [`super::scan_supported`]. That is the
    /// difference between an actionable error and a radio that accepts a
    /// well-formed guess, reads our fields as different ones, and reports
    /// nothing.
    ///
    /// Returns the scan's uid, which the completion notification echoes back.
    ///
    /// **The results do not come back from this command.** `SCAN_REQ_UMAC`
    /// answers with an acknowledgement; the networks arrive afterwards as
    /// ordinary beacons and probe responses on the receive path, to be fed to
    /// [`crate::drivers::wifi::scan::Scan`]. See its module docs — this surprised
    /// me, and it changes what "implement scan" means on every chipset.
    pub fn start_scan(
        &mut self,
        image: &fw::FirmwareImage,
        blob: &[u8],
        mac: &[u8; 6],
        channels: &[super::scan::Channel],
        passive: bool,
    ) -> Result<u32, &'static str> {
        let ver = super::scan_supported(image, blob).map_err(|e| match e {
            super::ScanUnsupported::Version(_) => {
                "firmware speaks a SCAN_REQ_UMAC version this driver does not implement"
            }
            super::ScanUnsupported::Unstated => "firmware does not state its SCAN_REQ_UMAC version",
        })?;
        debug_assert_eq!(ver, super::scan::VERSION);
        // The uid identifies this scan in the completion notification. It is
        // arbitrary but must be non-zero, since zero is what an uninitialised
        // notification carries.
        let uid = 1u32;
        let (req, n) =
            super::scan::build_v17(uid, mac, channels, passive).ok_or("no channels to scan")?;
        if n < channels.len() {
            crate::ktrace::log_fmt(format_args!(
                "iwl: scan truncated to {n} of {} channels (the request holds no more)",
                channels.len()
            ));
        }
        self.send_cmd(proto::GROUP_LONG, proto::SCAN_REQ_UMAC, &req)?;
        crate::ktrace::log_fmt(format_args!(
            "iwl: SCAN_REQ_UMAC v{ver} sent, uid {uid}, {n} channel(s), {} -- \
             results arrive as beacons on the receive path",
            if passive { "passive" } else { "active" }
        ));
        Ok(uid)
    }

    /// Collect scan results for up to `ms`, feeding every beacon into `out`.
    ///
    /// The networks arrive as `REPLY_RX_MPDU` notifications, each a descriptor
    /// followed by the 802.11 frame — see [`super::rx`] for why the descriptor's
    /// length is the whole risk. Anything that is not a beacon or probe response
    /// is ignored by [`crate::drivers::wifi::scan::Scan`] itself, so this loop
    /// hands over everything and lets one place decide.
    ///
    /// Bounded by time rather than by a result count: a scan of a quiet band
    /// legitimately finds nothing, and waiting for a number that never arrives
    /// would hang. Pumps `upkeep` and answers Ctrl+C, per the standing rule.
    pub fn collect_scan(&mut self, out: &mut crate::drivers::wifi::scan::Scan, ms: u64) -> usize {
        let start = crate::arch::now_ms();
        let mut frames = 0usize;
        while crate::arch::now_ms().saturating_sub(start) < ms {
            crate::shell::upkeep();
            if crate::shell::poll_interrupt() {
                crate::ktrace::log("iwl", "scan cancelled");
                break;
            }
            let Some(n) = self.poll_notification() else {
                continue;
            };
            if n.group_id != proto::GROUP_LEGACY || n.cmd != super::rx::REPLY_RX_MPDU {
                continue;
            }
            if let Some(m) = super::rx::parse(&n.payload, self.family) {
                frames += 1;
                out.on_frame(m.frame, m.rssi);
            }
        }
        crate::ktrace::log_fmt(format_args!(
            "iwl: scan collected {frames} frame(s) -> {} network(s)",
            out.len()
        ));
        frames
    }

    pub fn wait_for_alive(&mut self) -> Result<(), &'static str> {
        if self.rings.is_none() {
            return Err("no receive ring; load firmware first");
        }
        for _ in 0..Self::ALIVE_SPINS {
            let Some(n) = self.poll_notification() else {
                core::hint::spin_loop();
                continue;
            };
            if n.group_id == proto::GROUP_LEGACY && n.cmd == proto::UCODE_ERROR_NTFY {
                crate::ktrace::log("iwlwifi", "firmware reported its own failure");
                return Err("firmware reported an error instead of coming alive");
            }
            if n.group_id == proto::GROUP_LEGACY && n.cmd == proto::UCODE_ALIVE_NTFY {
                let status = if n.payload.len() >= 2 {
                    Some(u16::from_le_bytes([n.payload[0], n.payload[1]]))
                } else {
                    None
                };
                match status {
                    // A notification too short to carry a status is accepted: the field's
                    // position is stable but its presence depends on the firmware version,
                    // and refusing here would reject a working device over a missing check.
                    Some(proto::ALIVE_STATUS_OK) | None => {
                        crate::ktrace::log_fmt(format_args!(
                            "iwlwifi: firmware is alive ({} byte notification)",
                            n.payload.len()
                        ));
                        return Ok(());
                    }
                    Some(s) => {
                        crate::ktrace::log_fmt(format_args!(
                            "iwlwifi: firmware announced itself with status {s:#06x}, not OK"
                        ));
                        return Err("firmware came up but reported a bad status");
                    }
                }
            }
            crate::ktrace::log_fmt(format_args!(
                "iwlwifi: notification group {:#02x} cmd {:#02x} while waiting for alive",
                n.group_id, n.cmd
            ));
        }
        Err("firmware never sent an alive notification")
    }
}
