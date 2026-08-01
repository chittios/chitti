//! **USB Mass Storage** (Bulk-Only Transport + SCSI Transparent) as a
//! [`BlockDevice`].
//!
//! Class `08` / subclass `06` / protocol `50` (BOT). Built over the shared xHCI
//! bulk pair (same transport as CDC-ECM Ethernet). Pure wire packing is unit-
//! tested; device I/O is cold-plug at boot via [`probe_nth`].
//!
//! ## Scope
//! - LUN 0 only
//! - 512-byte logical blocks (refuses other sizes)
//! - **Read and write** via SCSI READ(10) / WRITE(10) over BOT
//! - One stick: `probe_nth(0)` only
//! - VFS still mounts foreign filesystems read-only by policy; the *block*
//!   device is writable (mkfs, raw tools, future FAT write)
//!
//! ## Traps
//! - CSW residual length is *untransferred* count, same as xHCI events
//! - Stall on bulk must clear halt (not done yet — fail closed on CSW error)
//! - Do not claim HID/CDC as MSC (class triple match only)
//! - Data-OUT must complete before CSW; a short OUT is a phase error

use crate::block::{BlockDevice, BlockError, BLOCK_SIZE};

/// USB class triple for MSC BOT / SCSI transparent.
pub const CLASS_MSC: u8 = 0x08;
pub const SUBCLASS_SCSI: u8 = 0x06;
pub const PROTO_BOT: u8 = 0x50;

/// CBW signature "USBC".
pub const CBW_SIG: u32 = 0x4342_5355;
/// CSW signature "USBS".
pub const CSW_SIG: u32 = 0x5342_5355;

pub const CBW_LEN: usize = 31;
pub const CSW_LEN: usize = 13;

/// Direction bit in CBW `bmCBWFlags`.
pub const CBW_FLAG_IN: u8 = 0x80;
pub const CBW_FLAG_OUT: u8 = 0x00;

/// SCSI opcodes we use.
pub const SCSI_TEST_UNIT_READY: u8 = 0x00;
pub const SCSI_INQUIRY: u8 = 0x12;
pub const SCSI_READ_CAPACITY_10: u8 = 0x25;
pub const SCSI_READ_10: u8 = 0x28;
pub const SCSI_WRITE_10: u8 = 0x2a;

/// Build a 31-byte Command Block Wrapper.
pub fn build_cbw(tag: u32, data_len: u32, flags: u8, lun: u8, cdb: &[u8]) -> [u8; CBW_LEN] {
    let mut b = [0u8; CBW_LEN];
    b[0..4].copy_from_slice(&CBW_SIG.to_le_bytes());
    b[4..8].copy_from_slice(&tag.to_le_bytes());
    b[8..12].copy_from_slice(&data_len.to_le_bytes());
    b[12] = flags;
    b[13] = lun & 0x0f;
    let clen = cdb.len().min(16) as u8;
    b[14] = clen;
    let n = clen as usize;
    b[15..15 + n].copy_from_slice(&cdb[..n]);
    b
}

/// Parsed Command Status Wrapper.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Csw {
    pub tag: u32,
    pub residue: u32,
    /// 0 = passed, 1 = failed, 2 = phase error.
    pub status: u8,
}

/// Parse a 13-byte CSW; `None` if the signature is wrong or the buffer is short.
pub fn parse_csw(buf: &[u8]) -> Option<Csw> {
    if buf.len() < CSW_LEN {
        return None;
    }
    let sig = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
    if sig != CSW_SIG {
        return None;
    }
    Some(Csw {
        tag: u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]),
        residue: u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]),
        status: buf[12],
    })
}

/// Zero-padded 16-byte CDB for TEST UNIT READY.
pub fn cdb_test_unit_ready() -> [u8; 16] {
    let mut c = [0u8; 16];
    c[0] = SCSI_TEST_UNIT_READY;
    c
}

/// INQUIRY with allocation length `alloc_len`.
pub fn cdb_inquiry(alloc_len: u8) -> [u8; 16] {
    let mut c = [0u8; 16];
    c[0] = SCSI_INQUIRY;
    c[4] = alloc_len;
    c
}

/// READ CAPACITY (10).
pub fn cdb_read_capacity_10() -> [u8; 16] {
    let mut c = [0u8; 16];
    c[0] = SCSI_READ_CAPACITY_10;
    c
}

/// READ (10): `lba` and `nblocks` (big-endian in the CDB).
pub fn cdb_read_10(lba: u32, nblocks: u16) -> [u8; 16] {
    let mut c = [0u8; 16];
    c[0] = SCSI_READ_10;
    c[2..6].copy_from_slice(&lba.to_be_bytes());
    c[7..9].copy_from_slice(&nblocks.to_be_bytes());
    c
}

/// WRITE (10): same layout as READ (10), opcode 0x2A.
pub fn cdb_write_10(lba: u32, nblocks: u16) -> [u8; 16] {
    let mut c = [0u8; 16];
    c[0] = SCSI_WRITE_10;
    c[2..6].copy_from_slice(&lba.to_be_bytes());
    c[7..9].copy_from_slice(&nblocks.to_be_bytes());
    c
}

/// Max sectors per single READ/WRITE(10) that fit the xHCI MSC bounce buffer
/// (`USB_MSC_BUF` = 4096 → 8 × 512).
pub const MAX_XFER_SECTORS: u16 = 8;

/// Decode READ CAPACITY (10) response: (last_lba, block_size).
pub fn parse_read_capacity_10(buf: &[u8]) -> Option<(u32, u32)> {
    if buf.len() < 8 {
        return None;
    }
    let last = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]);
    let bsz = u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]);
    Some((last, bsz))
}

/// Whether an interface class triple is MSC BOT + SCSI transparent.
pub fn is_msc_bot(class: u8, subclass: u8, protocol: u8) -> bool {
    class == CLASS_MSC && subclass == SUBCLASS_SCSI && protocol == PROTO_BOT
}

// ── live device ──────────────────────────────────────────────────────────

/// A USB mass-storage LUN 0 as a 512-byte [`BlockDevice`].
pub struct UsbMsc {
    /// Number of 512-byte sectors.
    sectors: u64,
    tag: u32,
}

impl UsbMsc {
    /// Probe the xHCI bulk pair if it was classified as MSC; run INQUIRY +
    /// READ CAPACITY. `n == 0` only (one stick).
    pub fn probe_nth(n: usize) -> Option<UsbMsc> {
        if n != 0 {
            return None;
        }
        if !crate::arch::usb_msc_ready() {
            return None;
        }
        let mut dev = UsbMsc { sectors: 0, tag: 1 };
        // TEST UNIT READY (ignore fail — some sticks need a few before ready).
        for _ in 0..5 {
            if dev.bot(cdb_test_unit_ready(), None, CbDir::None).is_ok() {
                break;
            }
            // Brief settle; MSC often NAKs until media is spun up.
            for _ in 0..100_000 {
                core::hint::spin_loop();
            }
        }
        // INQUIRY (optional info for the log).
        let mut inq = [0u8; 36];
        if dev.bot(cdb_inquiry(36), Some(&mut inq), CbDir::In).is_ok() {
            let vendor = core::str::from_utf8(&inq[8..16]).unwrap_or("?").trim();
            let product = core::str::from_utf8(&inq[16..32]).unwrap_or("?").trim();
            crate::ktrace::log_fmt(format_args!("usb_msc: INQUIRY '{vendor}' '{product}'"));
        }
        let mut cap = [0u8; 8];
        if dev.bot(cdb_read_capacity_10(), Some(&mut cap), CbDir::In).is_err() {
            crate::ktrace::log("usb_msc", "READ CAPACITY failed");
            return None;
        }
        let (last, bsz) = parse_read_capacity_10(&cap)?;
        if bsz != BLOCK_SIZE as u32 {
            crate::ktrace::log_fmt(format_args!(
                "usb_msc: block size {bsz} (need {}); refusing",
                BLOCK_SIZE
            ));
            return None;
        }
        // last LBA is inclusive.
        let sectors = last as u64 + 1;
        if sectors == 0 {
            return None;
        }
        dev.sectors = sectors;
        crate::ktrace::log_fmt(format_args!(
            "usb_msc: ready, {sectors} sectors ({} MiB), 512-byte, RW",
            sectors * 512 / 1024 / 1024
        ));
        Some(dev)
    }

    fn next_tag(&mut self) -> u32 {
        let t = self.tag;
        self.tag = self.tag.wrapping_add(1);
        t
    }

    fn bot(&mut self, cdb: [u8; 16], data: Option<&mut [u8]>, dir: CbDir) -> Result<(), BlockError> {
        let data_len = data.as_ref().map(|d| d.len() as u32).unwrap_or(0);
        let flags = match dir {
            CbDir::In => CBW_FLAG_IN,
            CbDir::Out => CBW_FLAG_OUT,
            CbDir::None => CBW_FLAG_OUT,
        };
        let tag = self.next_tag();
        let cbw = build_cbw(tag, data_len, flags, 0, &cdb);
        // Longer timeout on data phases: flash sticks can be slow to program.
        if !crate::arch::usb_bulk_sync_out(&cbw, 2000) {
            return Err(BlockError::DeviceError);
        }
        if let Some(buf) = data {
            match dir {
                CbDir::In => {
                    let n = crate::arch::usb_bulk_sync_in(buf, 5000).ok_or(BlockError::DeviceError)?;
                    if n < buf.len() {
                        // Short read: zero the rest so the caller does not see stale data.
                        buf[n..].fill(0);
                    }
                }
                CbDir::Out => {
                    if !crate::arch::usb_bulk_sync_out(buf, 10_000) {
                        return Err(BlockError::DeviceError);
                    }
                }
                CbDir::None => {}
            }
        }
        let mut cswb = [0u8; CSW_LEN];
        let n = crate::arch::usb_bulk_sync_in(&mut cswb, 5000).ok_or(BlockError::DeviceError)?;
        if n < CSW_LEN {
            return Err(BlockError::DeviceError);
        }
        let csw = parse_csw(&cswb).ok_or(BlockError::DeviceError)?;
        if csw.tag != tag || csw.status != 0 {
            return Err(BlockError::DeviceError);
        }
        Ok(())
    }

    /// WRITE(10) of `n` contiguous sectors starting at `index` (`buf` length =
    /// `n * 512`).
    fn write_sectors(&mut self, index: u64, buf: &[u8]) -> Result<(), BlockError> {
        if buf.len() % BLOCK_SIZE != 0 || buf.is_empty() {
            return Err(BlockError::BadBufferLen);
        }
        let n = (buf.len() / BLOCK_SIZE) as u16;
        if n > MAX_XFER_SECTORS {
            return Err(BlockError::BadBufferLen);
        }
        if index >= self.sectors || index + n as u64 > self.sectors {
            return Err(BlockError::OutOfRange);
        }
        if index > u32::MAX as u64 {
            return Err(BlockError::OutOfRange);
        }
        let cdb = cdb_write_10(index as u32, n);
        // bot wants &mut for the shared Option path; data is only read for OUT.
        let mut tmp = [0u8; BLOCK_SIZE * MAX_XFER_SECTORS as usize];
        tmp[..buf.len()].copy_from_slice(buf);
        self.bot(cdb, Some(&mut tmp[..buf.len()]), CbDir::Out)
    }

    fn read_sectors(&mut self, index: u64, buf: &mut [u8]) -> Result<(), BlockError> {
        if buf.len() % BLOCK_SIZE != 0 || buf.is_empty() {
            return Err(BlockError::BadBufferLen);
        }
        let n = (buf.len() / BLOCK_SIZE) as u16;
        if n > MAX_XFER_SECTORS {
            return Err(BlockError::BadBufferLen);
        }
        if index >= self.sectors || index + n as u64 > self.sectors {
            return Err(BlockError::OutOfRange);
        }
        if index > u32::MAX as u64 {
            return Err(BlockError::OutOfRange);
        }
        let cdb = cdb_read_10(index as u32, n);
        self.bot(cdb, Some(buf), CbDir::In)
    }
}

enum CbDir {
    In,
    Out,
    None,
}

impl BlockDevice for UsbMsc {
    fn block_count(&self) -> u64 {
        self.sectors
    }

    fn read_block(&mut self, index: u64, buf: &mut [u8]) -> Result<(), BlockError> {
        if buf.len() != BLOCK_SIZE {
            return Err(BlockError::BadBufferLen);
        }
        self.read_sectors(index, buf)
    }

    fn write_block(&mut self, index: u64, buf: &[u8]) -> Result<(), BlockError> {
        if buf.len() != BLOCK_SIZE {
            return Err(BlockError::BadBufferLen);
        }
        self.write_sectors(index, buf)
    }

    fn read_blocks(&mut self, index: u64, buf: &mut [u8]) -> Result<(), BlockError> {
        let total = buf.len() / BLOCK_SIZE;
        if total == 0 || buf.len() % BLOCK_SIZE != 0 {
            return Err(BlockError::BadBufferLen);
        }
        let mut done = 0usize;
        while done < total {
            let chunk = (total - done).min(MAX_XFER_SECTORS as usize);
            let off = done * BLOCK_SIZE;
            let end = off + chunk * BLOCK_SIZE;
            self.read_sectors(index + done as u64, &mut buf[off..end])?;
            done += chunk;
        }
        Ok(())
    }

    fn write_blocks(&mut self, index: u64, buf: &[u8]) -> Result<(), BlockError> {
        let total = buf.len() / BLOCK_SIZE;
        if total == 0 || buf.len() % BLOCK_SIZE != 0 {
            return Err(BlockError::BadBufferLen);
        }
        let mut done = 0usize;
        while done < total {
            let chunk = (total - done).min(MAX_XFER_SECTORS as usize);
            let off = done * BLOCK_SIZE;
            let end = off + chunk * BLOCK_SIZE;
            self.write_sectors(index + done as u64, &buf[off..end])?;
            done += chunk;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn is_msc_bot_matches_class_triple_only() {
        assert!(is_msc_bot(0x08, 0x06, 0x50));
        assert!(!is_msc_bot(0x08, 0x06, 0x00)); // CBI
        assert!(!is_msc_bot(0x08, 0x04, 0x50)); // UFI
        assert!(!is_msc_bot(0x03, 0x01, 0x01)); // HID
        assert!(!is_msc_bot(0x0a, 0x00, 0x00)); // CDC data
    }

    #[test_case]
    fn build_cbw_layout_matches_bot_spec() {
        let cdb = cdb_read_10(0x100, 8);
        let b = build_cbw(0x42, 4096, CBW_FLAG_IN, 0, &cdb);
        assert_eq!(u32::from_le_bytes([b[0], b[1], b[2], b[3]]), CBW_SIG);
        assert_eq!(u32::from_le_bytes([b[4], b[5], b[6], b[7]]), 0x42);
        assert_eq!(u32::from_le_bytes([b[8], b[9], b[10], b[11]]), 4096);
        assert_eq!(b[12], CBW_FLAG_IN);
        assert_eq!(b[13], 0);
        assert_eq!(b[14], 16); // we always pass 16-byte CDB pad
        assert_eq!(b[15], SCSI_READ_10);
        // LBA big-endian at CDB[2..6] → CBW[17..21]
        assert_eq!(&b[17..21], &0x100u32.to_be_bytes());
        assert_eq!(&b[22..24], &8u16.to_be_bytes());
    }

    #[test_case]
    fn parse_csw_accepts_pass_and_rejects_bad_sig() {
        let mut b = [0u8; 13];
        b[0..4].copy_from_slice(&CSW_SIG.to_le_bytes());
        b[4..8].copy_from_slice(&7u32.to_le_bytes());
        b[12] = 0;
        let c = parse_csw(&b).unwrap();
        assert_eq!(c.tag, 7);
        assert_eq!(c.status, 0);
        b[0] = 0;
        assert!(parse_csw(&b).is_none());
        assert!(parse_csw(&b[..12]).is_none());
    }

    #[test_case]
    fn parse_read_capacity_counts_inclusive_last_lba() {
        let mut b = [0u8; 8];
        b[0..4].copy_from_slice(&99u32.to_be_bytes()); // last LBA
        b[4..8].copy_from_slice(&512u32.to_be_bytes());
        let (last, bsz) = parse_read_capacity_10(&b).unwrap();
        assert_eq!(last, 99);
        assert_eq!(bsz, 512);
        // 100 sectors total
        assert_eq!(last as u64 + 1, 100);
    }

    #[test_case]
    fn cdb_read_10_is_big_endian() {
        let c = cdb_read_10(0x01020304, 0x0506);
        assert_eq!(c[0], SCSI_READ_10);
        assert_eq!(&c[2..6], &[0x01, 0x02, 0x03, 0x04]);
        assert_eq!(&c[7..9], &[0x05, 0x06]);
    }

    #[test_case]
    fn cdb_write_10_matches_read_layout_with_write_opcode() {
        let r = cdb_read_10(0x100, 4);
        let w = cdb_write_10(0x100, 4);
        assert_eq!(w[0], SCSI_WRITE_10);
        assert_eq!(&w[1..], &r[1..]);
        // CBW for a one-sector write: data length 512, OUT flag.
        let b = build_cbw(1, 512, CBW_FLAG_OUT, 0, &w);
        assert_eq!(b[12], CBW_FLAG_OUT);
        assert_eq!(u32::from_le_bytes([b[8], b[9], b[10], b[11]]), 512);
        assert_eq!(b[15], SCSI_WRITE_10);
    }
}
