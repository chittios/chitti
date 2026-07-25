//! **NVMe block driver core** — arch-neutral, the modern real-hardware storage
//! path (the disk most laptops/servers and hypervisors like VirtualBox expose
//! when not using virtio). Polled admin + one I/O queue pair; read/write via
//! PRP lists; presented behind the shared [`BlockDevice`] API.
//!
//! A controller can expose several **namespaces** (VirtualBox presents each
//! attached disk as NSID 1, 2, …). NSIDs are **not necessarily contiguous**:
//! VirtualBox maps controller *port* → NSID, so a disk on port 1 with port 0
//! empty is NSID 2 with NSID 1 inactive (exactly what a VM whose install
//! medium was detached looks like). Enumeration therefore uses the spec's
//! IDENTIFY CNS=2 **Active Namespace ID List** (mandatory since NVMe 1.1),
//! not "probe NSID 1, 2, … until the first empty one". The controller is
//! brought up **once** into a global ([`CONTROLLER`]); each namespace is a
//! lightweight [`NvmeNamespace`] handle that routes its I/O through the shared
//! controller under a lock, so multiple namespaces coexist and are usable
//! simultaneously (no per-disk controller re-reset). Device discovery is
//! per-arch (each arch finds the NVMe function on its PCI bus, maps BAR0, and
//! hands the mapped MMIO base + a [`DmaAlloc`] to [`probe_namespace`]) — the
//! same one-driver-both-arches shape as the `xhci` core.

use crate::block::{BlockDevice, BlockError, Dma, DmaAlloc, BLOCK_SIZE};
use crate::mm::Locked;
use core::ptr::{read_volatile, write_volatile};
use core::sync::atomic::{fence, Ordering};

// Controller register offsets (BAR0).
const REG_CAP: u64 = 0x00; // capabilities (u64)
const REG_CC: u64 = 0x14; // controller config
const REG_CSTS: u64 = 0x1c; // controller status
const REG_AQA: u64 = 0x24; // admin queue attributes
const REG_ASQ: u64 = 0x28; // admin SQ base (u64)
const REG_ACQ: u64 = 0x30; // admin CQ base (u64)

const CC_EN: u32 = 1;
const CSTS_RDY: u32 = 1;

const QDEPTH: usize = 16; // entries per queue (small; polled)
const SQE: usize = 64; // submission queue entry size
const CQE: usize = 16; // completion queue entry size

/// Largest MMIO span the register block can reach (CAP..doorbells for a couple
/// of queues): plenty for the admin + one I/O queue this driver creates. The
/// arch wrapper maps at least this much.
pub const MMIO_SPAN: usize = 0x2000;

const DATA_MAX: usize = 64 * 1024;

/// The single, shared NVMe controller — brought up once, then every namespace's
/// I/O routes through it (serialized by this lock; the driver is polled, one
/// request at a time).
pub static CONTROLLER: Locked<Option<NvmeController>> = Locked::new(None);

/// Ring a submission (is_cq=false) tail / completion (is_cq=true) head doorbell.
unsafe fn ring_doorbell(regs: u64, dstrd: u32, qid: u32, is_cq: bool, val: u32) {
    let idx = 2 * qid + if is_cq { 1 } else { 0 };
    unsafe { w32(regs + 0x1000 + (idx * dstrd) as u64, val) };
}

unsafe fn r32(a: u64) -> u32 {
    unsafe { read_volatile(a as *const u32) }
}
unsafe fn w32(a: u64, v: u32) {
    unsafe { write_volatile(a as *mut u32, v) };
}
unsafe fn w64(a: u64, v: u64) {
    unsafe { write_volatile(a as *mut u64, v) };
}

/// A polled NVMe controller + its single admin + I/O queue pair. Owns the DMA
/// rings + a shared bounce buffer; namespace I/O is serialized through it.
pub struct NvmeController {
    regs: u64,  // mapped BAR0 (virtual)
    dstrd: u32, // doorbell stride (bytes) = 4 << CAP.DSTRD
    // Admin queue.
    asq: Dma,
    acq: Dma,
    a_sq_tail: u32,
    a_cq_head: u32,
    a_phase: u32,
    // I/O queue (qid 1).
    iosq: Dma,
    iocq: Dma,
    io_sq_tail: u32,
    io_cq_head: u32,
    io_phase: u32,
    cid: u16,
    prp_list: Dma, // one page for PRP lists on multi-page transfers
    data_buf: Dma, // 64 KiB bounce buffer (also used for Identify)
    /// Active NSIDs from IDENTIFY CNS=2 (sparse-safe enumeration; see module
    /// doc). Empty only if the controller rejected the identify — then
    /// [`probe_namespace`] falls back to the legacy contiguous NSID walk.
    active: alloc::vec::Vec<u32>,
}

impl NvmeController {
    /// Reset the controller and set up the admin + one I/O queue pair. Does NOT
    /// touch any namespace. `None` if the controller never becomes ready.
    ///
    /// # Safety
    /// `regs` must be a valid, mapped NVMe BAR0 (≥ [`MMIO_SPAN`] bytes) and
    /// `alloc` must return real physically-contiguous DMA memory.
    unsafe fn bringup(regs: u64, alloc: DmaAlloc) -> Option<NvmeController> {
        unsafe {
            let cap = (r32(regs + REG_CAP) as u64) | ((r32(regs + REG_CAP + 4) as u64) << 32);
            let dstrd = 4u32 << ((cap >> 32) & 0xf);

            // Disable, wait not-ready.
            w32(regs + REG_CC, 0);
            let mut g = 0;
            while r32(regs + REG_CSTS) & CSTS_RDY != 0 && g < 1_000_000 {
                g += 1;
            }

            // Admin queues.
            let asq = alloc(QDEPTH * SQE)?;
            let acq = alloc(QDEPTH * CQE)?;
            w32(regs + REG_AQA, (((QDEPTH - 1) as u32) << 16) | (QDEPTH - 1) as u32);
            w64(regs + REG_ASQ, asq.phys);
            w64(regs + REG_ACQ, acq.phys);
            // CC: IOSQES=6 (64B), IOCQES=4 (16B), enable.
            w32(regs + REG_CC, (6 << 16) | (4 << 20) | CC_EN);
            g = 0;
            while r32(regs + REG_CSTS) & CSTS_RDY == 0 {
                g += 1;
                if g > 1_000_000 {
                    crate::ktrace::log("nvme", "controller never became ready");
                    return None;
                }
            }

            let mut ctrl = NvmeController {
                regs,
                dstrd,
                asq,
                acq,
                a_sq_tail: 0,
                a_cq_head: 0,
                a_phase: 1,
                iosq: alloc(QDEPTH * SQE)?,
                iocq: alloc(QDEPTH * CQE)?,
                io_sq_tail: 0,
                io_cq_head: 0,
                io_phase: 1,
                cid: 0,
                prp_list: alloc(4096)?,
                data_buf: alloc(DATA_MAX)?,
                active: alloc::vec::Vec::new(),
            };

            // Create the I/O completion + submission queues (qid 1).
            // Create IO CQ: opcode 0x05; CDW10 = (size-1)<<16 | qid; CDW11 = PC(1).
            if !ctrl.admin(0x05, 0, ctrl.iocq.phys, 0, (((QDEPTH - 1) as u32) << 16) | 1, 1) {
                return None;
            }
            // Create IO SQ: opcode 0x01; CDW11 = cqid(1)<<16 | PC(1).
            if !ctrl.admin(0x01, 0, ctrl.iosq.phys, 0, (((QDEPTH - 1) as u32) << 16) | 1, (1 << 16) | 1) {
                return None;
            }
            crate::ktrace::log("nvme", "controller up (admin + 1 I/O queue)");

            // Active Namespace ID List (IDENTIFY CNS=2, NSID=0 → the 4 KiB
            // page lists ascending active NSIDs, zero-terminated). NSIDs can
            // be sparse (VirtualBox: port → NSID, empty port 0 = inactive
            // NSID 1), so this — not a walk from NSID 1 — is the enumeration.
            if ctrl.admin(0x06, 0, ctrl.data_buf.phys, 0, 2, 0) {
                let mut page = [0u8; 4096];
                core::ptr::copy_nonoverlapping(ctrl.data_buf.virt as *const u8, page.as_mut_ptr(), 4096);
                ctrl.active = parse_ns_list(&page);
                crate::ktrace::log_fmt(format_args!(
                    "nvme: {} active namespace(s){}",
                    ctrl.active.len(),
                    match ctrl.active.first() {
                        Some(first) => alloc::format!(" (first NSID {first})"),
                        None => alloc::string::String::new(),
                    }
                ));
            } else {
                // Pre-1.1 controller: fall back to the contiguous walk.
                crate::ktrace::log("nvme", "active-ns-list identify unsupported; assuming contiguous NSIDs");
            }
            Some(ctrl)
        }
    }

    /// Identify namespace `nsid` (CNS=0); return `(capacity_in_512B_sectors,
    /// lba_bytes)`, or `None` if the namespace is absent (NSZE==0) or has an
    /// unusable LBA size. Reuses the shared bounce buffer (no concurrent I/O at
    /// probe time).
    fn identify_namespace(&mut self, nsid: u32) -> Option<(u64, u32)> {
        // SAFETY: admin queue + data_buf from bringup.
        unsafe {
            let buf = self.data_buf;
            if !self.admin(0x06, nsid, buf.phys, 0, 0, 0) {
                crate::ktrace::log_fmt(format_args!("nvme: identify nsid {nsid} failed"));
                return None;
            }
            let nsze = read_volatile(buf.virt as *const u64);
            if nsze == 0 {
                return None; // namespace inactive (legacy-walk terminator)
            }
            let flbas = read_volatile((buf.virt + 26) as *const u8) & 0xf;
            let lbaf = read_volatile((buf.virt + 128 + flbas as u64 * 4) as *const u32);
            let lba_bytes = 1u32 << ((lbaf >> 16) & 0xff);
            if lba_bytes == 0 || lba_bytes % BLOCK_SIZE as u32 != 0 {
                crate::ktrace::log("nvme", "unsupported LBA size (not a multiple of 512)");
                return None;
            }
            let capacity = nsze * (lba_bytes as u64 / BLOCK_SIZE as u64);
            crate::ktrace::log_fmt(format_args!(
                "nvme: namespace {} : {} LBAs x {} B -> {} 512B-sectors ({} MiB)",
                nsid,
                nsze,
                lba_bytes,
                capacity,
                capacity * 512 / (1024 * 1024)
            ));
            Some((capacity, lba_bytes))
        }
    }

    /// Submit an admin command + poll its completion. Returns success.
    unsafe fn admin(&mut self, opcode: u8, nsid: u32, prp1: u64, prp2: u64, cdw10: u32, cdw11: u32) -> bool {
        unsafe { self.submit(false, opcode, nsid, prp1, prp2, cdw10, cdw11, 0, 0) }
    }

    /// Submit to the admin (false) or I/O (true) SQ + poll the matching CQ.
    #[allow(clippy::too_many_arguments)]
    unsafe fn submit(&mut self, io: bool, opcode: u8, nsid: u32, prp1: u64, prp2: u64, cdw10: u32, cdw11: u32, cdw12: u32, _r: u32) -> bool {
        unsafe {
            let (regs, dstrd) = (self.regs, self.dstrd);
            let (sq, tail, qid) = if io { (self.iosq, &mut self.io_sq_tail, 1u32) } else { (self.asq, &mut self.a_sq_tail, 0u32) };
            let cid = self.cid;
            self.cid = self.cid.wrapping_add(1);
            let e = sq.virt + (*tail as u64) * SQE as u64;
            // Zero the 64-byte entry, then fill the fields we use.
            core::ptr::write_bytes(e as *mut u8, 0, SQE);
            w32(e, (opcode as u32) | ((cid as u32) << 16)); // CDW0: opcode + CID
            w32(e + 4, nsid); // NSID
            w64(e + 24, prp1); // PRP1
            w64(e + 32, prp2); // PRP2
            w32(e + 40, cdw10);
            w32(e + 44, cdw11);
            w32(e + 48, cdw12);
            fence(Ordering::SeqCst);
            *tail = (*tail + 1) % QDEPTH as u32;
            ring_doorbell(regs, dstrd, qid, false, *tail);

            // Poll the completion queue for this CID.
            let (cq, head, phase) = if io { (self.iocq, &mut self.io_cq_head, &mut self.io_phase) } else { (self.acq, &mut self.a_cq_head, &mut self.a_phase) };
            let mut spins = 0u64;
            loop {
                let c = cq.virt + (*head as u64) * CQE as u64;
                let status = read_volatile((c + 12) as *const u32);
                // CQE DWORD3: CID in bits 15:0, phase tag (P) at bit 16,
                // status field (SC/SCT) in bits 31:17.
                if ((status >> 16) & 1) == *phase {
                    let sc = (status >> 17) & 0xff; // status code
                    *head = (*head + 1) % QDEPTH as u32;
                    if *head == 0 {
                        *phase ^= 1;
                    }
                    ring_doorbell(regs, dstrd, qid, true, *head);
                    return sc == 0;
                }
                spins += 1;
                if spins > 2_000_000_000 {
                    crate::ktrace::log("nvme", "command never completed (2e9 spins) -- giving up");
                    return false;
                }
                // Answer Ctrl+C, for the same reason as the AHCI completion wait:
                // the bound is many seconds on a failing device, and a machine
                // stuck in a disk read must still be escapable. `poll_interrupt`
                // only (non-blocking, pushes back non-Ctrl+C bytes) — never
                // `upkeep()`, which would re-enter the UI pump this can be called
                // from.
                if spins % 0x10_0000 == 0 && crate::shell::poll_interrupt() {
                    crate::ktrace::log("nvme", "command cancelled by Ctrl+C");
                    return false;
                }
                core::hint::spin_loop();
            }
        }
    }

    /// One I/O read (opcode 0x02) / write (0x01) on `nsid` of `len` bytes at
    /// 512-sector `index`, through the shared bounce buffer + PRP list.
    fn rw(&mut self, nsid: u32, lba_bytes: u32, write: bool, index: u64, ptr: *mut u8, len: usize) -> Result<(), BlockError> {
        let per = lba_bytes as u64 / BLOCK_SIZE as u64; // 512-sectors per NVMe LBA
        let slba = index / per;
        let nlba = (len as u64 / lba_bytes as u64).max(1);
        // SAFETY: buffers from bringup; PRP list built for the transfer.
        unsafe {
            if write {
                core::ptr::copy_nonoverlapping(ptr as *const u8, self.data_buf.virt as *mut u8, len);
            }
            let prp1 = self.data_buf.phys;
            let pages = len.div_ceil(4096);
            let prp2 = if pages <= 1 {
                0
            } else if pages == 2 {
                self.data_buf.phys + 4096
            } else {
                for i in 0..pages - 1 {
                    w64(self.prp_list.virt + i as u64 * 8, self.data_buf.phys + (i as u64 + 1) * 4096);
                }
                self.prp_list.phys
            };
            let opcode = if write { 0x01 } else { 0x02 };
            let ok = self.submit(true, opcode, nsid, prp1, prp2, slba as u32, (slba >> 32) as u32, (nlba - 1) as u32, 0);
            if !ok {
                return Err(BlockError::DeviceError);
            }
            if !write {
                core::ptr::copy_nonoverlapping(self.data_buf.virt as *const u8, ptr, len);
            }
        }
        Ok(())
    }
}

/// Parse an IDENTIFY CNS=2 Active Namespace ID List page: ascending non-zero
/// little-endian u32 NSIDs, zero-terminated (or full). Pure — unit-tested.
pub fn parse_ns_list(page: &[u8]) -> alloc::vec::Vec<u32> {
    let mut out = alloc::vec::Vec::new();
    for w in page.chunks_exact(4) {
        let nsid = u32::from_le_bytes([w[0], w[1], w[2], w[3]]);
        if nsid == 0 {
            break;
        }
        out.push(nsid);
    }
    out
}

/// Bring up the controller (once) and attach to the `n`-th **active**
/// namespace (per the IDENTIFY CNS=2 list — NSIDs may be sparse; legacy
/// fallback is NSID `n+1`). `None` once namespaces run out. Called by the
/// per-arch `probe_nth`.
///
/// # Safety
/// `regs` must be a valid mapped NVMe BAR0; `alloc` real DMA memory.
pub unsafe fn probe_namespace(regs: u64, alloc: DmaAlloc, n: usize) -> Option<NvmeNamespace> {
    CONTROLLER.with(|slot| {
        if slot.is_none() {
            // SAFETY: forwarded from the caller's contract.
            *slot = unsafe { NvmeController::bringup(regs, alloc) };
        }
        let ctrl = slot.as_mut()?;
        let nsid = match ctrl.active.get(n) {
            Some(&id) => id,
            None if ctrl.active.is_empty() => (n + 1) as u32, // legacy fallback
            None => return None, // past the last active namespace
        };
        let (capacity, lba_bytes) = ctrl.identify_namespace(nsid)?;
        Some(NvmeNamespace { nsid, capacity, lba_bytes })
    })
}

/// A single NVMe namespace behind the shared [`BlockDevice`] API. Cheap handle;
/// every operation routes through the shared [`CONTROLLER`] under its lock, so
/// several namespaces can be held and used at once.
pub struct NvmeNamespace {
    nsid: u32,
    capacity: u64,
    lba_bytes: u32,
}

impl NvmeNamespace {
    /// Run `f` on the shared controller under the lock; `DeviceError` if the
    /// controller went away (never, in practice, once brought up).
    fn with_ctrl<F: FnOnce(&mut NvmeController) -> Result<(), BlockError>>(&self, f: F) -> Result<(), BlockError> {
        CONTROLLER.with(|slot| match slot.as_mut() {
            Some(c) => f(c),
            None => Err(BlockError::DeviceError),
        })
    }
}

impl BlockDevice for NvmeNamespace {
    fn block_count(&self) -> u64 {
        self.capacity
    }
    fn read_block(&mut self, index: u64, buf: &mut [u8]) -> Result<(), BlockError> {
        if buf.len() != BLOCK_SIZE {
            return Err(BlockError::BadBufferLen);
        }
        self.read_blocks(index, buf)
    }
    fn write_block(&mut self, index: u64, buf: &[u8]) -> Result<(), BlockError> {
        if buf.len() != BLOCK_SIZE {
            return Err(BlockError::BadBufferLen);
        }
        let mut tmp = [0u8; BLOCK_SIZE];
        tmp.copy_from_slice(buf);
        self.write_blocks(index, &tmp)
    }
    fn read_blocks(&mut self, index: u64, buf: &mut [u8]) -> Result<(), BlockError> {
        if buf.len() % BLOCK_SIZE != 0 {
            return Err(BlockError::BadBufferLen);
        }
        let (nsid, lba) = (self.nsid, self.lba_bytes);
        self.with_ctrl(|c| {
            let mut off = 0usize;
            while off < buf.len() {
                let take = (buf.len() - off).min(DATA_MAX);
                c.rw(nsid, lba, false, index + (off / BLOCK_SIZE) as u64, buf[off..].as_mut_ptr(), take)?;
                off += take;
            }
            Ok(())
        })
    }
    fn write_blocks(&mut self, index: u64, buf: &[u8]) -> Result<(), BlockError> {
        if buf.len() % BLOCK_SIZE != 0 {
            return Err(BlockError::BadBufferLen);
        }
        let (nsid, lba) = (self.nsid, self.lba_bytes);
        self.with_ctrl(|c| {
            let mut off = 0usize;
            while off < buf.len() {
                let take = (buf.len() - off).min(DATA_MAX);
                c.rw(nsid, lba, true, index + (off / BLOCK_SIZE) as u64, buf[off..].as_ptr() as *mut u8, take)?;
                off += take;
            }
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::parse_ns_list;

    fn page(ids: &[u32]) -> alloc::vec::Vec<u8> {
        let mut p = alloc::vec![0u8; 4096];
        for (i, id) in ids.iter().enumerate() {
            p[i * 4..i * 4 + 4].copy_from_slice(&id.to_le_bytes());
        }
        p
    }

    #[test_case]
    fn ns_list_contiguous() {
        assert_eq!(parse_ns_list(&page(&[1, 2])), alloc::vec![1, 2]);
    }

    #[test_case]
    fn ns_list_sparse_vbox_port1_only() {
        // VirtualBox with an empty port 0: the only disk is NSID 2 — the exact
        // shape the legacy "walk from NSID 1" enumeration missed.
        assert_eq!(parse_ns_list(&page(&[2])), alloc::vec![2]);
    }

    #[test_case]
    fn ns_list_empty_and_full() {
        assert!(parse_ns_list(&page(&[])).is_empty());
        // A full page with no zero terminator yields all 1024 entries.
        let all: alloc::vec::Vec<u32> = (1..=1024).collect();
        assert_eq!(parse_ns_list(&page(&all)).len(), 1024);
    }
}
