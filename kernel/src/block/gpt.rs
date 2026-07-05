//! A minimal **GPT partition-table writer** for `/install`: lay down a
//! protective MBR + primary/backup GPT with two partitions — an EFI System
//! Partition (FAT32, for Limine + the kernel + the model) and a Chitti/Linux
//! data partition (ext4, our OS volume). Enough of the GPT spec to produce
//! a table firmware + `sgdisk`/Linux accept; not a general editor.

use crate::block::{BlockDevice, BlockError, BLOCK_SIZE};

/// EFI System Partition type GUID (C12A7328-F81F-11D2-BA4B-00A0C93EC93B).
const ESP_GUID: [u8; 16] =
    [0x28, 0x73, 0x2a, 0xc1, 0x1f, 0xf8, 0xd2, 0x11, 0xba, 0x4b, 0x00, 0xa0, 0xc9, 0x3e, 0xc9, 0x3b];
/// Linux filesystem data type GUID (0FC63DAF-8483-4772-8E79-3D69D8477DE4) —
/// used for the Chitti (ext4) OS partition.
const LINUX_GUID: [u8; 16] =
    [0xaf, 0x3d, 0xc6, 0x0f, 0x83, 0x84, 0x72, 0x47, 0x8e, 0x79, 0x3d, 0x69, 0xd8, 0x47, 0x7d, 0xe4];

/// A partition to create: (type GUID, first LBA, last LBA, UTF-16 name).
pub struct PartitionSpec {
    pub type_guid: [u8; 16],
    pub first_lba: u64,
    pub last_lba: u64,
    pub name: &'static str,
}

/// The result of laying out a GPT: where each partition lives.
pub struct Layout {
    pub esp_first: u64,
    pub esp_last: u64,
    pub os_first: u64,
    pub os_last: u64,
    /// A separate ext4 *data* partition for durable agent state (synapse::fs).
    /// Kept apart from the OS/model partition so the running kernel can rewrite
    /// it freely without touching the model.
    pub data_first: u64,
    pub data_last: u64,
}

// --- CRC32 (IEEE 802.3, reflected) --------------------------------------
fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xffff_ffff;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xedb8_8320 & mask);
        }
    }
    !crc
}

fn le64(buf: &mut [u8], off: usize, v: u64) {
    buf[off..off + 8].copy_from_slice(&v.to_le_bytes());
}
fn le32(buf: &mut [u8], off: usize, v: u32) {
    buf[off..off + 4].copy_from_slice(&v.to_le_bytes());
}

/// Compute the standard two-partition layout for a disk of `total_sectors`
/// 512-byte blocks: a 512 MiB ESP followed by the rest as the OS partition.
/// GPT reserves LBA0 (PMBR), LBA1 (header), LBA2..33 (entries) and the last 33
/// sectors (backup). Returns `None` if the disk is too small.
pub fn default_layout(total_sectors: u64) -> Option<Layout> {
    let first_usable = 34u64;
    let last_usable = total_sectors.checked_sub(34)?;
    // 64 MiB ESP: enough for the Limine loader, and within FAT16's cluster-count
    // range (the ext4 partition holds limine.conf + kernel + model).
    let esp_sectors = 64 * 1024 * 1024 / BLOCK_SIZE as u64;
    let esp_first = first_usable;
    let esp_last = esp_first + esp_sectors - 1;
    let os_first = esp_last + 1;
    // Carve a 256 MiB ext4 data partition off the tail for durable agent state.
    let data_sectors = 256 * 1024 * 1024 / BLOCK_SIZE as u64;
    if os_first + data_sectors + 2 >= last_usable {
        return None;
    }
    let data_last = last_usable;
    let data_first = data_last - data_sectors + 1;
    let os_last = data_first - 1;
    Some(Layout { esp_first, esp_last, os_first, os_last, data_first, data_last })
}

/// Write a protective MBR + primary and backup GPT describing `parts` to `dev`.
/// A fixed disk GUID / partition GUIDs are used (deterministic install).
pub fn write<D: BlockDevice>(dev: &mut D, parts: &[PartitionSpec]) -> Result<(), BlockError> {
    let total = dev.block_count();
    let entries_lba = 2u64;
    let n_entries = 128u32;
    let entry_size = 128u32;
    let entry_sectors = (n_entries * entry_size).div_ceil(BLOCK_SIZE as u32) as u64; // 32
    let backup_hdr = total - 1;
    let backup_entries = backup_hdr - entry_sectors;

    // --- protective MBR (LBA0) ---
    let mut mbr = [0u8; BLOCK_SIZE];
    mbr[0x1be + 4] = 0xee; // type 0xEE = GPT protective
    le32(&mut mbr, 0x1be + 8, 1); // starting LBA
    le32(&mut mbr, 0x1be + 12, (total - 1).min(0xffff_ffff) as u32);
    mbr[510] = 0x55;
    mbr[511] = 0xaa;
    dev.write_block(0, &mbr)?;

    // --- partition entry array (LBA2..) ---
    let mut entries = [0u8; 128 * 128];
    for (i, p) in parts.iter().enumerate() {
        let e = i * 128;
        entries[e..e + 16].copy_from_slice(&p.type_guid);
        // Unique partition GUID: deterministic, derived from the index.
        for b in 0..16 {
            entries[e + 16 + b] = (0xa0 + i as u8).wrapping_add(b as u8);
        }
        le64(&mut entries, e + 32, p.first_lba);
        le64(&mut entries, e + 40, p.last_lba);
        // name: UTF-16LE, up to 36 code units.
        for (k, c) in p.name.encode_utf16().take(36).enumerate() {
            entries[e + 56 + k * 2..e + 56 + k * 2 + 2].copy_from_slice(&c.to_le_bytes());
        }
    }
    let entries_crc = crc32(&entries);
    // Write the entry array (primary + backup).
    write_bytes(dev, entries_lba, &entries)?;
    write_bytes(dev, backup_entries, &entries)?;

    // --- GPT header (primary at LBA1, backup at last sector) ---
    let disk_guid: [u8; 16] = *b"CHITTI-OS-DISK01";
    let write_hdr = |dev: &mut D, my_lba: u64, alt_lba: u64, ent_lba: u64| -> Result<(), BlockError> {
        let mut h = [0u8; BLOCK_SIZE];
        h[0..8].copy_from_slice(b"EFI PART");
        le32(&mut h, 8, 0x0001_0000); // revision 1.0
        le32(&mut h, 12, 92); // header size
        le64(&mut h, 24, my_lba);
        le64(&mut h, 32, alt_lba);
        le64(&mut h, 40, 34); // first usable
        le64(&mut h, 48, total - 34); // last usable
        h[56..72].copy_from_slice(&disk_guid);
        le64(&mut h, 72, ent_lba);
        le32(&mut h, 80, n_entries);
        le32(&mut h, 84, entry_size);
        le32(&mut h, 88, entries_crc);
        // header CRC over the 92-byte header with the CRC field (16) zeroed.
        le32(&mut h, 16, 0);
        let hc = crc32(&h[0..92]);
        le32(&mut h, 16, hc);
        dev.write_block(my_lba, &h)
    };
    write_hdr(dev, 1, backup_hdr, entries_lba)?;
    write_hdr(dev, backup_hdr, 1, backup_entries)?;
    Ok(())
}

/// Write a byte buffer spanning multiple sectors, sector-aligned.
fn write_bytes<D: BlockDevice>(dev: &mut D, start_lba: u64, data: &[u8]) -> Result<(), BlockError> {
    let mut buf = [0u8; BLOCK_SIZE];
    for (i, chunk) in data.chunks(BLOCK_SIZE).enumerate() {
        buf.fill(0);
        buf[..chunk.len()].copy_from_slice(chunk);
        dev.write_block(start_lba + i as u64, &buf)?;
    }
    Ok(())
}

/// One partition read back from an on-disk GPT: `(first_lba, last_lba, name)`.
pub struct ReadPart {
    pub first_lba: u64,
    pub last_lba: u64,
    pub name: alloc::string::String,
}

/// Read a disk's GPT: `Some((is_chitti_disk, partitions))` if a valid GPT
/// header is present at LBA 1. `is_chitti_disk` = the disk GUID matches the
/// one [`write`] stamps — how `/install` detects an existing Chitti install
/// and updates the system partitions in place instead of erasing the disk.
pub fn read<D: BlockDevice>(dev: &mut D) -> Option<(bool, alloc::vec::Vec<ReadPart>)> {
    let mut h = [0u8; BLOCK_SIZE];
    dev.read_block(1, &mut h).ok()?;
    if &h[0..8] != b"EFI PART" {
        return None;
    }
    let is_chitti = &h[56..72] == b"CHITTI-OS-DISK01";
    let ent_lba = rd64(&h, 72);
    let n = rd32(&h, 80).min(128) as usize;
    let esz = rd32(&h, 84) as usize;
    if esz < 128 {
        return None;
    }
    let mut parts = alloc::vec::Vec::new();
    let mut sec = [0u8; BLOCK_SIZE];
    for i in 0..n {
        let byte_off = i * esz;
        let lba = ent_lba + (byte_off / BLOCK_SIZE) as u64;
        let off = byte_off % BLOCK_SIZE;
        if off + 128 > BLOCK_SIZE {
            continue; // entries are 128-byte aligned within sectors for esz=128
        }
        dev.read_block(lba, &mut sec).ok()?;
        let e = &sec[off..off + 128];
        if e[0..16].iter().all(|&b| b == 0) {
            continue; // unused entry
        }
        let first_lba = rd64(e, 32);
        let last_lba = rd64(e, 40);
        let mut name = alloc::string::String::new();
        for k in 0..36 {
            let c = u16::from_le_bytes([e[56 + k * 2], e[56 + k * 2 + 1]]);
            if c == 0 {
                break;
            }
            if let Some(ch) = char::from_u32(c as u32) {
                name.push(ch);
            }
        }
        parts.push(ReadPart { first_lba, last_lba, name });
    }
    Some((is_chitti, parts))
}

fn rd32(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}
fn rd64(b: &[u8], off: usize) -> u64 {
    u64::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3], b[off + 4], b[off + 5], b[off + 6], b[off + 7]])
}

/// Two-partition layout for the aarch64 install: an ESP big enough for
/// `esp_bytes` (the UEFI stub + kernel + model all live on the FAT ESP, which
/// the stub reads directly) + the rest as the ext4 data partition for durable
/// agent state. No separate ext4 OS partition — the stub needs no second copy.
pub fn esp_data_parts(total_sectors: u64, esp_bytes: u64) -> Option<[PartitionSpec; 2]> {
    let first_usable = 34u64;
    let last_usable = total_sectors.checked_sub(34)?;
    // ESP: payload + 64 MiB slack for FAT metadata/growth.
    let esp_sectors = (esp_bytes + 64 * 1024 * 1024).div_ceil(BLOCK_SIZE as u64);
    let esp_last = first_usable + esp_sectors - 1;
    let data_first = esp_last + 1;
    if data_first + 2048 >= last_usable {
        return None; // no room for a data partition
    }
    Some([
        PartitionSpec { type_guid: ESP_GUID, first_lba: first_usable, last_lba: esp_last, name: "EFI System" },
        PartitionSpec { type_guid: LINUX_GUID, first_lba: data_first, last_lba: last_usable, name: "Chitti Data" },
    ])
}

/// Build the three standard partition specs (ESP + ext4 OS/model + ext4 data).
pub fn standard_parts(layout: &Layout) -> [PartitionSpec; 3] {
    [
        PartitionSpec { type_guid: ESP_GUID, first_lba: layout.esp_first, last_lba: layout.esp_last, name: "EFI System" },
        PartitionSpec { type_guid: LINUX_GUID, first_lba: layout.os_first, last_lba: layout.os_last, name: "Chitti OS" },
        PartitionSpec { type_guid: LINUX_GUID, first_lba: layout.data_first, last_lba: layout.data_last, name: "Chitti Data" },
    ]
}
