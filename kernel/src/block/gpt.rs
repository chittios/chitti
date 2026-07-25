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
        PartitionSpec { type_guid: LINUX_GUID, first_lba: layout.os_first, last_lba: layout.os_last, name: "ChittiOS" },
        PartitionSpec { type_guid: LINUX_GUID, first_lba: layout.data_first, last_lba: layout.data_last, name: "Chitti Data" },
    ]
}

// --- installing alongside an existing OS ---------------------------------
//
// `/install` currently has exactly two behaviours: update an existing Chitti disk
// in place, or write a fresh GPT over the whole device. The second one erases
// Windows, which makes "install on the machine you already have" impossible.
//
// The pieces below are the *planning* half of installing alongside: pure
// arithmetic over the partition table, unit-tested, no device writes. They answer
// "where could ChittiOS go without touching anything that is already there?" so
// the destructive path can be replaced with a decision rather than an assumption.

/// A run of unallocated sectors, inclusive of both ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FreeExtent {
    pub first_lba: u64,
    pub last_lba: u64,
}

impl FreeExtent {
    pub fn sectors(&self) -> u64 {
        self.last_lba.saturating_sub(self.first_lba).saturating_add(1)
    }
}

/// Sectors reserved at the start of the disk (protective MBR, GPT header, and a
/// 128-entry partition array) and mirrored at the end (backup GPT).
const GPT_RESERVED_HEAD: u64 = 34;
const GPT_RESERVED_TAIL: u64 = 33;

/// Every gap between existing partitions that is at least `min_sectors` long.
///
/// Existing entries may be unsorted and, on a disk that has been repartitioned by
/// several tools, may even overlap; both are handled by sorting on `first_lba` and
/// advancing a high-water mark rather than assuming the entries tile the disk in
/// order. Zero-length / unused entries are ignored.
pub fn free_extents(parts: &[ReadPart], total_sectors: u64, min_sectors: u64) -> alloc::vec::Vec<FreeExtent> {
    let mut used: alloc::vec::Vec<(u64, u64)> = parts
        .iter()
        .filter(|p| p.last_lba >= p.first_lba && p.first_lba != 0)
        .map(|p| (p.first_lba, p.last_lba))
        .collect();
    used.sort_unstable();

    let mut out = alloc::vec::Vec::new();
    let end = total_sectors.saturating_sub(GPT_RESERVED_TAIL); // exclusive
    let mut cursor = GPT_RESERVED_HEAD;
    for (first, last) in used {
        if first > cursor {
            let gap_last = (first - 1).min(end.saturating_sub(1));
            if gap_last >= cursor {
                let e = FreeExtent { first_lba: cursor, last_lba: gap_last };
                if e.sectors() >= min_sectors {
                    out.push(e);
                }
            }
        }
        // High-water mark, so an overlapping or contained entry cannot rewind it
        // and manufacture free space inside another partition.
        cursor = cursor.max(last.saturating_add(1));
    }
    if end > cursor {
        let e = FreeExtent { first_lba: cursor, last_lba: end - 1 };
        if e.sectors() >= min_sectors {
            out.push(e);
        }
    }
    out
}

/// Where ChittiOS would go on a disk that already has an OS on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AlongsidePlan {
    /// The existing EFI System Partition to add our loader to. We *share* it
    /// rather than formatting one: a PC has exactly one ESP, and reformatting it
    /// would remove the Windows boot manager.
    pub esp_first_lba: u64,
    pub esp_last_lba: u64,
    /// Free space claimed for the ChittiOS ext4 partition.
    pub os_first_lba: u64,
    pub os_last_lba: u64,
}

/// Plan an install alongside the existing contents of a disk.
///
/// Returns `None` when it cannot be done safely, which is a real answer and not a
/// failure: no ESP to share (so nothing would boot us), or no single free extent
/// big enough. Refusing beats falling back to erasing the disk.
///
/// `esp_names` is matched case-insensitively against partition names because the
/// ESP is identified differently by different tools ("EFI System Partition",
/// "EFI system partition", "ESP"). A caller with GUID-level information should
/// prefer that; this is the name-based fallback.
pub fn plan_alongside(
    parts: &[ReadPart],
    total_sectors: u64,
    needed_sectors: u64,
    min_esp_sectors: u64,
) -> Option<AlongsidePlan> {
    // Find an existing ESP big enough to also hold our loader.
    let esp = parts.iter().find(|p| {
        let n = p.name.to_ascii_lowercase();
        (n.contains("efi") || n == "esp")
            && p.last_lba >= p.first_lba
            && (p.last_lba - p.first_lba + 1) >= min_esp_sectors
    })?;
    // Largest free extent, so we do not strand the install in a tiny gap when a
    // better one exists.
    let best = free_extents(parts, total_sectors, needed_sectors)
        .into_iter()
        .max_by_key(|e| e.sectors())?;
    Some(AlongsidePlan {
        esp_first_lba: esp.first_lba,
        esp_last_lba: esp.last_lba,
        os_first_lba: best.first_lba,
        os_last_lba: best.first_lba + needed_sectors - 1,
    })
}

#[cfg(test)]
mod alongside_tests {
    use super::*;
    use alloc::string::String;
    use alloc::vec;

    fn p(first: u64, last: u64, name: &str) -> ReadPart {
        ReadPart { first_lba: first, last_lba: last, name: String::from(name) }
    }

    /// A realistic Windows disk: MSR, the ESP, Windows itself, a recovery
    /// partition, and free space left at the end.
    fn windows_disk() -> alloc::vec::Vec<ReadPart> {
        vec![
            p(2048, 206_847, "EFI system partition"),
            p(206_848, 239_615, "Microsoft reserved partition"),
            p(239_616, 4_000_000, "Basic data partition"),
            p(4_000_001, 4_100_000, "Windows Recovery environment"),
        ]
    }

    #[test_case]
    fn free_extents_finds_the_tail_gap() {
        let parts = windows_disk();
        let free = free_extents(&parts, 8_000_000, 1000);
        // Head is consumed by the ESP starting at 2048, so the only real gap is
        // after the recovery partition.
        assert_eq!(free.len(), 2, "{free:?}");
        assert_eq!(free[0], FreeExtent { first_lba: 34, last_lba: 2047 });
        assert_eq!(free[1].first_lba, 4_100_001);
        assert_eq!(free[1].last_lba, 8_000_000 - 33 - 1);
    }

    #[test_case]
    fn free_extents_respects_the_minimum() {
        let parts = windows_disk();
        // The 2014-sector head gap must drop out when a larger minimum is asked.
        let free = free_extents(&parts, 8_000_000, 100_000);
        assert_eq!(free.len(), 1);
        assert_eq!(free[0].first_lba, 4_100_001);
    }

    #[test_case]
    fn free_extents_never_reports_space_inside_a_partition() {
        // Overlapping / contained entries appear on disks repartitioned by several
        // tools. A naive scan that resets its cursor per entry would report the
        // inside of the big partition as free — and installing there would
        // overwrite live data.
        let parts = vec![p(2048, 1_000_000, "Big"), p(4096, 8192, "Contained")];
        let free = free_extents(&parts, 2_000_000, 1);
        for e in &free {
            assert!(
                e.first_lba > 1_000_000 || e.last_lba < 2048,
                "extent {e:?} overlaps an existing partition"
            );
        }
    }

    #[test_case]
    fn free_extents_reserves_both_gpt_copies() {
        // Nothing may be handed out below LBA 34 (protective MBR + primary GPT)
        // or in the last 33 sectors (backup GPT).
        let free = free_extents(&[], 100_000, 1);
        assert_eq!(free.len(), 1);
        assert_eq!(free[0].first_lba, 34);
        assert_eq!(free[0].last_lba, 100_000 - 34);
    }

    #[test_case]
    fn plan_alongside_shares_the_existing_esp() {
        let parts = windows_disk();
        let plan = plan_alongside(&parts, 8_000_000, 1_000_000, 100_000).unwrap();
        // The ESP is *shared*, not recreated: a PC has one, and reformatting it
        // would delete the Windows boot manager.
        assert_eq!(plan.esp_first_lba, 2048);
        assert_eq!(plan.esp_last_lba, 206_847);
        // Our partition lands in the tail gap, and touches nothing existing.
        assert_eq!(plan.os_first_lba, 4_100_001);
        assert_eq!(plan.os_last_lba, 4_100_001 + 1_000_000 - 1);
        for e in &parts {
            assert!(
                plan.os_first_lba > e.last_lba || plan.os_last_lba < e.first_lba,
                "plan overlaps {}",
                e.name
            );
        }
    }

    #[test_case]
    fn plan_alongside_refuses_rather_than_guessing() {
        let parts = windows_disk();
        // Not enough contiguous free space -> None, so the caller reports it
        // instead of falling back to erasing the disk.
        assert!(plan_alongside(&parts, 8_000_000, 100_000_000, 100_000).is_none());
        // No ESP at all -> nothing could boot us, so refuse.
        let no_esp = vec![p(2048, 4_000_000, "Basic data partition")];
        assert!(plan_alongside(&no_esp, 8_000_000, 1000, 1000).is_none());
        // An ESP too small to also hold our loader is not usable either.
        let tiny = vec![p(2048, 2148, "EFI system partition"), p(3000, 4_000_000, "Basic data partition")];
        assert!(plan_alongside(&tiny, 8_000_000, 1000, 100_000).is_none());
    }
}
