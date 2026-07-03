//! **NVMe block driver core** — arch-neutral, the modern real-hardware storage
//! path (the disk most laptops/servers and hypervisors like VirtualBox expose
//! when not using virtio). Polled admin + one I/O queue pair; read/write via
//! PRP lists; presented behind the shared [`BlockDevice`] API.
//!
//! This module is the *logic*; device discovery is per-arch (each arch finds
//! the NVMe function on its PCI bus, maps BAR0, and hands the mapped MMIO base
//! + a [`DmaAlloc`] to [`Nvme::bringup`]). x86 (`arch::x86_64::nvme`) and
//! aarch64 (`arch::aarch64::nvme`) are thin wrappers over this one core — the
//! same one-driver-both-arches structure as the `xhci` core.

use crate::block::{BlockDevice, BlockError, Dma, DmaAlloc, BLOCK_SIZE};
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

/// A polled NVMe controller + its single I/O queue pair. Buffers are [`Dma`]
/// pairs: the CPU fills queue entries via `.virt`, the device is programmed
/// with `.phys`.
pub struct Nvme {
    regs: u64,  // mapped BAR0 (virtual)
    dstrd: u32, // doorbell stride (bytes) = 4 << CAP.DSTRD
    capacity: u64,
    lba_bytes: u32,
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
    data_buf: Dma, // 64 KiB bounce buffer
}

impl Nvme {
    /// Bring up the controller at the already-mapped MMIO base `regs`, using
    /// `alloc` for all DMA memory. `None` if the controller never readies or the
    /// namespace has an unusable LBA size.
    ///
    /// # Safety
    /// `regs` must be a valid, mapped NVMe BAR0 (at least [`MMIO_SPAN`] bytes),
    /// and `alloc` must return real physically-contiguous DMA memory.
    pub unsafe fn bringup(regs: u64, alloc: DmaAlloc) -> Option<Nvme> {
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

            let mut nv = Nvme {
                regs,
                dstrd,
                capacity: 0,
                lba_bytes: 512,
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
            };

            // Create the I/O completion + submission queues (qid 1).
            // Create IO CQ: opcode 0x05; CDW10 = (size-1)<<16 | qid; CDW11 = PC(1).
            if !nv.admin(0x05, 0, nv.iocq.phys, 0, (((QDEPTH - 1) as u32) << 16) | 1, 1) {
                return None;
            }
            // Create IO SQ: opcode 0x01; CDW11 = cqid(1)<<16 | PC(1).
            if !nv.admin(0x01, 0, nv.iosq.phys, 0, (((QDEPTH - 1) as u32) << 16) | 1, (1 << 16) | 1) {
                return None;
            }

            // Identify Namespace 1 (CNS=0) for capacity + LBA size.
            let idbuf = alloc(4096)?;
            if !nv.admin(0x06, 1, idbuf.phys, 0, 0, 0) {
                return None;
            }
            // NSZE (u64 @ 0) = number of LBAs; FLBAS (@26) selects the LBA format;
            // LBA formats start @128, each u32, LBADS (byte 2) = log2(lba bytes).
            let nsze = read_volatile(idbuf.virt as *const u64);
            let flbas = read_volatile((idbuf.virt + 26) as *const u8) & 0xf;
            let lbaf = read_volatile((idbuf.virt + 128 + flbas as u64 * 4) as *const u32);
            let lbads = (lbaf >> 16) & 0xff;
            nv.lba_bytes = 1u32 << lbads;
            nv.capacity = nsze * (nv.lba_bytes as u64 / BLOCK_SIZE as u64);
            crate::ktrace::log_fmt(format_args!(
                "nvme: up ({} LBAs x {} B) -> {} 512B-sectors ({} MiB)",
                nsze,
                nv.lba_bytes,
                nv.capacity,
                nv.capacity * 512 / (1024 * 1024)
            ));
            if nv.lba_bytes % BLOCK_SIZE as u32 != 0 {
                crate::ktrace::log("nvme", "unsupported LBA size (not a multiple of 512)");
                return None;
            }
            Some(nv)
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
                    return false;
                }
                core::hint::spin_loop();
            }
        }
    }

    /// One I/O read (opcode 0x02) or write (0x01) of `len` bytes at 512-sector
    /// `index`, through the bounce buffer + PRP list.
    fn rw(&mut self, write: bool, index: u64, ptr: *mut u8, len: usize) -> Result<(), BlockError> {
        let per = self.lba_bytes as u64 / BLOCK_SIZE as u64; // 512-sectors per NVMe LBA
        let slba = index / per;
        let nlba = (len as u64 / self.lba_bytes as u64).max(1);
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
                // PRP list: entries for pages 2..N (page 1 is PRP1).
                for i in 0..pages - 1 {
                    w64(self.prp_list.virt + i as u64 * 8, self.data_buf.phys + (i as u64 + 1) * 4096);
                }
                self.prp_list.phys
            };
            let opcode = if write { 0x01 } else { 0x02 };
            let ok = self.submit(true, opcode, 1, prp1, prp2, slba as u32, (slba >> 32) as u32, (nlba - 1) as u32, 0);
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

impl BlockDevice for Nvme {
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
        self.rw(true, index, tmp.as_mut_ptr(), BLOCK_SIZE)
    }
    fn read_blocks(&mut self, index: u64, buf: &mut [u8]) -> Result<(), BlockError> {
        if buf.len() % BLOCK_SIZE != 0 {
            return Err(BlockError::BadBufferLen);
        }
        let mut off = 0usize;
        while off < buf.len() {
            let take = (buf.len() - off).min(DATA_MAX);
            self.rw(false, index + (off / BLOCK_SIZE) as u64, buf[off..].as_mut_ptr(), take)?;
            off += take;
        }
        Ok(())
    }
    fn write_blocks(&mut self, index: u64, buf: &[u8]) -> Result<(), BlockError> {
        if buf.len() % BLOCK_SIZE != 0 {
            return Err(BlockError::BadBufferLen);
        }
        let mut off = 0usize;
        while off < buf.len() {
            let take = (buf.len() - off).min(DATA_MAX);
            self.rw(true, index + (off / BLOCK_SIZE) as u64, buf[off..].as_ptr() as *mut u8, take)?;
            off += take;
        }
        Ok(())
    }
}
