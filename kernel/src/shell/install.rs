//! install
//!
//! The **self-hosting installer** carved out of the former 16k-line
//! `shell/mod.rs` monolith: `/install`, `/install plan` and
//! `/install alongside` (see `block::gpt` + `block::esp`), plus `/mkfs`
//! and `/ext4read` dev helpers. Moved verbatim; `use super::*` keeps the
//! parent's statics visible, and the parent re-imports this module's items
//! with `pub(crate) use install::*`.

use super::*;


/// Parse `/install` arguments into [`InstallArgs`].
/// Tokens in any order: an optional numeric disk index, `yes` (skip the
/// confirmation modal — for scripted use), `format` (force a full repartition
/// even when an existing Chitti install would be updated in place), and `plan`
/// (read-only: report a possible install alongside an existing OS, write
/// nothing), and `alongside` (non-destructively add our loader to the existing
/// ESP, modifying no partition).
pub(super) fn parse_install_args(arg: &str) -> InstallArgs {
    let mut a = InstallArgs::default();
    for tok in arg.split_whitespace() {
        match tok {
            "yes" => a.pre_confirmed = true,
            "format" => a.force_format = true,
            "plan" => a.plan_only = true,
            "alongside" => a.alongside = true,
            t => {
                if let Ok(n) = t.parse::<usize>() {
                    a.target = Some(n);
                }
            }
        }
    }
    a
}

/// Parsed `/install` arguments.
#[derive(Default, Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct InstallArgs {
    pre_confirmed: bool,
    force_format: bool,
    /// `plan`: report what an install alongside the existing OS would do and
    /// change **nothing**. Read-only, no modal, no confirmation — the point is to
    /// be able to ask "will ChittiOS fit next to Windows on this machine?" without
    /// risking the machine to find out.
    plan_only: bool,
    /// `alongside`: add our loader to the **existing** ESP instead of
    /// repartitioning. Non-destructive to every existing partition.
    alongside: bool,
    target: Option<usize>,
}

/// Report what installing **alongside** the existing OS on `target_idx` would do,
/// and change nothing.
///
/// This is `/install plan`. It exists because the only way to find out whether
/// ChittiOS fits next to Windows used to be to run `/install`, which writes a
/// fresh GPT over the whole disk — you had to risk the machine to learn the
/// answer. Every operation here is a read.
///
/// It reports rather than decides: the planner ([`gpt::plan_alongside`]) refuses
/// when there is no ESP to share or no single gap big enough, and this prints that
/// refusal with the free space it did find, so the reason is visible.
pub(super) fn install_plan(target_idx: usize) {
    use crate::block::{gpt, BlockDevice};
    let Some(mut dev) = crate::block::probe_disk_nth(target_idx) else {
        serial_println!("install> no disk {target_idx} (see /disks)");
        return;
    };
    let total = dev.block_count();
    let Some((is_chitti, parts)) = gpt::read(&mut dev) else {
        serial_println!("install> disk {target_idx} has no GPT ({total} sectors) -- nothing to install alongside;");
        serial_println!("install>   a plain `/install {target_idx}` would partition the whole disk.");
        return;
    };
    serial_println!("install> disk {target_idx}: {total} sectors, {} partition(s){}", parts.len(), if is_chitti { " (an existing ChittiOS disk)" } else { "" });
    for p in &parts {
        let mib = (p.last_lba.saturating_sub(p.first_lba) + 1) / 2048;
        serial_println!("install>   {:>10}..{:<10} {:>7} MiB  {}", p.first_lba, p.last_lba, mib, p.name);
    }
    // 8 GiB of ext4 for the OS + model, and an ESP with room for our loader.
    const NEED_SECTORS: u64 = 8 * 1024 * 2048;
    const MIN_ESP_SECTORS: u64 = 64 * 2048; // 64 MiB
    let free = gpt::free_extents(&parts, total, 2048);
    if free.is_empty() {
        serial_println!("install> no unallocated space -- shrink a Windows partition first (Disk Management)");
    } else {
        serial_println!("install> unallocated:");
        for e in &free {
            serial_println!("install>   {:>10}..{:<10} {:>7} MiB", e.first_lba, e.last_lba, e.sectors() / 2048);
        }
    }
    match gpt::plan_alongside(&parts, total, NEED_SECTORS, MIN_ESP_SECTORS) {
        Some(plan) => {
            serial_println!("install> PLAN: share the existing ESP at {}..{} (adds our loader; Windows' stays)", plan.esp_first_lba, plan.esp_last_lba);
            serial_println!("install>       new ChittiOS ext4 at {}..{} ({} MiB)", plan.os_first_lba, plan.os_last_lba, (plan.os_last_lba - plan.os_first_lba + 1) / 2048);
            serial_println!("install>       existing partitions: untouched");
            serial_println!("install> NOT YET EXECUTABLE: writing the loader into an existing ESP needs FAT32");
            serial_println!("install>   write-into-existing-volume support, which is not implemented. `/install`");
            serial_println!("install>   still repartitions the WHOLE disk and would erase the above.");
        }
        None => {
            serial_println!("install> cannot install alongside: need an ESP >= {} MiB to share and {} MiB contiguous free", MIN_ESP_SECTORS / 2048, NEED_SECTORS / 2048);
        }
    }
}

/// Install the ChittiOS loader into the **existing** ESP on `target_idx`, backing
/// up whatever loader is there. `/install alongside`.
///
/// This is the non-destructive counterpart to `/install`: it adds one file to the
/// EFI System Partition already on the disk and touches no partition table and no
/// existing partition. What it does *not* do is create the ChittiOS data partition
/// — so it makes a machine bootable into a ChittiOS kernel that came from the
/// install medium, not a full on-disk install. The output says so rather than
/// implying more than happened.
/// x86-only, and legitimately so rather than as a dropped feature: the payload
/// comes from a Limine module here, the fallback loader is `BOOTX64.EFI`, and the
/// scenario is coexisting with an x86 Windows install. An ARM equivalent needs
/// `BOOTAA64.EFI` and the aarch64 payload plumbing, which is separate work.
#[cfg(target_arch = "x86_64")]
pub(super) fn install_alongside(target_idx: usize, pre_confirmed: bool) {
    use crate::block::{esp::Esp, gpt, BlockDevice, Partition};
    let Some(efi) = crate::cortex::find_module("BOOTX64.EFI") else {
        serial_println!("install> no BOOTX64.EFI payload -- build the ISO with xtask");
        return;
    };
    let Some(mut dev) = crate::block::probe_disk_nth(target_idx) else {
        serial_println!("install> no disk {target_idx} (see /disks)");
        return;
    };
    let Some((_, parts)) = gpt::read(&mut dev) else {
        serial_println!("install> disk {target_idx} has no GPT -- nothing to install alongside");
        return;
    };
    // Locate the ESP by name, the same way the planner does.
    let Some(esp_part) = parts.iter().find(|p| {
        let n = p.name.to_ascii_lowercase();
        n.contains("efi") || n == "esp"
    }) else {
        serial_println!("install> no EFI System Partition on disk {target_idx}");
        return;
    };
    // Destructive-enough to confirm: it rewrites a file in another OS's ESP.
    if !pre_confirmed
        && !crate::modal::confirm(
            "Install ChittiOS loader alongside?",
            &alloc::format!(
                "Adds ChittiOS to the EXISTING EFI System Partition on disk {target_idx} (lba {}..{}). Any current \\EFI\\BOOT\\BOOTX64.EFI is renamed to BOOTX64.CHB as a backup, not deleted. No partition table or existing partition is modified. Proceed?",
                esp_part.first_lba, esp_part.last_lba
            ),
        )
    {
        serial_println!("install> aborted (not confirmed)");
        return;
    }
    let count = esp_part.last_lba.saturating_sub(esp_part.first_lba) + 1;
    let mut part = Partition::new(&mut dev, esp_part.first_lba, count);
    let mut esp = match Esp::open(&mut part) {
        Ok(e) => e,
        Err(e) => {
            serial_println!("install> cannot open the ESP filesystem: {e:?}");
            return;
        }
    };
    serial_println!("install> ESP {} MiB free", esp.free_bytes() / (1024 * 1024));
    match esp.install_loader(efi) {
        Ok(done) => {
            serial_println!("install> loader written to \\EFI\\BOOT\\BOOTX64.EFI ({} cluster(s))", done.clusters_used);
            if done.backed_up {
                serial_println!("install>   previous loader renamed to {} (restore it to undo)", crate::block::esp::BACKUP_NAME);
            } else if done.backup_preserved {
                serial_println!("install>   existing {} backup left untouched (it is the original)", crate::block::esp::BACKUP_NAME);
            }
            serial_println!("install> existing partitions: UNCHANGED");
            serial_println!("install> NB no ChittiOS data partition was created, and firmware NVRAM was");
            serial_println!("install>   not touched -- a machine that boots Windows via its own NVRAM entry");
            serial_println!("install>   still needs its boot order changed by hand.");
        }
        Err(e) => serial_println!("install> failed: {e:?} (nothing was changed)"),
    }
}

/// The `/install` human gate: a permission modal (destructive actions are
/// confirmed via the modal, not an inline `yes` token — `yes` remains only as
/// a scripted pre-confirmation). Returns true to proceed.
pub(super) fn confirm_install(pre_confirmed: bool, update: bool, disk: usize) -> bool {
    if pre_confirmed {
        return true;
    }
    let (title, msg) = if update {
        (
            "Update ChittiOS \u{2014} confirm?",
            alloc::format!(
                "Disk {} already has Chitti installed. The system partitions (boot loader, kernel, model) will be REWRITTEN; the data partition (agent state) is preserved. Add 'format' to erase everything instead. Proceed?",
                disk
            ),
        )
    } else {
        (
            "Install ChittiOS \u{2014} confirm?",
            alloc::format!("This ERASES EVERYTHING on disk {} and repartitions it (GPT: ESP + ext4). Proceed?", disk),
        )
    };
    if crate::modal::confirm(title, &msg) {
        true
    } else {
        serial_println!("install> aborted (not confirmed)");
        false
    }
}

#[cfg(target_arch = "x86_64")]
pub(super) fn disk_install(arg: &str) {
    use crate::block::{ext4::{Ext4Writer, FileSpec}, fat::FatWriter, gpt, BlockDevice, Partition};
    use alloc::string::String;
    use alloc::vec::Vec;
    let a = parse_install_args(arg);
    let (pre_confirmed, force_format, target_override) = (a.pre_confirmed, a.force_format, a.target);
    if a.plan_only {
        install_plan(target_override.unwrap_or(0));
        return;
    }
    if a.alongside {
        install_alongside(target_override.unwrap_or(0), pre_confirmed);
        return;
    }
    let (Some(efi), Some(kernel)) = (crate::cortex::find_module("BOOTX64.EFI"), crate::cortex::find_module("payload/chitti-kernel")) else {
        serial_println!("install> installer payload missing (BOOTX64.EFI / kernel modules) -- build the ISO with xtask");
        return;
    };
    // Default target: with 2+ disks, prefer disk 1 (the permanent second drive)
    // over disk 0 (usually the boot/install image). Explicit `/install N` wins.
    let target_idx = target_override.unwrap_or_else(|| {
        if crate::block::probe_disk_nth(1).is_some() {
            serial_println!(
                "install> multiple disks present — defaulting to disk 1 (use /install 0 to target the boot disk)"
            );
            1
        } else {
            0
        }
    });
    let Some(mut dev) = crate::block::probe_disk_nth(target_idx) else {
        serial_println!("install> no disk {} (see /disks; boot with a -drive)", target_idx);
        return;
    };
    // An existing Chitti install (our GPT disk GUID) is UPDATED in place: the
    // system partitions are rewritten, the data partition (durable agent
    // state) is untouched. `format` forces the old erase-everything path.
    let existing = gpt::read(&mut dev).and_then(|(chitti, parts)| {
        if !chitti || force_format {
            return None;
        }
        let find = |n: &str| parts.iter().find(|p| p.name == n).map(|p| (p.first_lba, p.last_lba));
        match (find("EFI System"), find("ChittiOS")) {
            (Some(e), Some(o)) => Some((e, o)),
            _ => None,
        }
    });
    if !confirm_install(pre_confirmed, existing.is_some(), target_idx) {
        return;
    }
    serial_println!("install> target disk {}", target_idx);
    let total = dev.block_count();

    // 1. Partitions: reuse the existing GPT on an update; otherwise write a
    //    fresh GPT (FAT ESP + ext4 OS + ext4 data).
    let (esp_range, os_range, fresh_layout) = match existing {
        Some((e, o)) => {
            serial_println!("install> existing Chitti install detected -- updating in place (data partition preserved)");
            (e, o, None)
        }
        None => {
            let Some(layout) = gpt::default_layout(total) else {
                serial_println!("install> disk too small ({} sectors)", total);
                return;
            };
            if let Err(e) = gpt::write(&mut dev, &gpt::standard_parts(&layout)) {
                serial_println!("install> GPT write failed: {:?}", e);
                return;
            }
            serial_println!("install> GPT: ESP lba {}..{}, ext4 OS lba {}..{}, ext4 data lba {}..{}", layout.esp_first, layout.esp_last, layout.os_first, layout.os_last, layout.data_first, layout.data_last);
            ((layout.esp_first, layout.esp_last), (layout.os_first, layout.os_last), Some(layout))
        }
    };

    // 2. FAT ESP: the Limine loader at /EFI/BOOT/BOOTX64.EFI, plus limine.conf
    //    + the kernel at the root, so the disk boots from FAT alone (UEFI
    //    firmware requires FAT; Limine reads its config from the boot volume).
    let esp_conf = b"timeout: 0\n\n/ChittiOS\n    protocol: limine\n    resolution: 1920x1080\n    path: boot():/chitti-kernel\n";
    {
        let mut esp = Partition::new(&mut dev, esp_range.0, esp_range.1 - esp_range.0 + 1);
        let r = FatWriter::format(&mut esp).and_then(|mut fw| {
            fw.write_efi_boot_file("BOOTX64.EFI", efi)?;
            fw.write_root_file("limine.conf", esp_conf)?;
            fw.write_root_file("chitti-kernel", kernel)?;
            Ok(())
        });
        if let Err(e) = r {
            serial_println!("install> ESP FAT write failed: {:?}", e);
            return;
        }
    }
    serial_println!("install> ESP (FAT16): BOOTX64.EFI + limine.conf + kernel written.");

    // 3. ext4 OS partition: limine.conf + kernel + model parts.
    let parts = crate::cortex::model_parts();
    let mut conf = String::from("timeout: 3\n\n/ChittiOS\n    protocol: limine\n    resolution: 1920x1080\n    path: boot():/chitti-kernel\n");
    for (name, _) in &parts {
        conf.push_str("    module_path: boot():/");
        conf.push_str(name);
        conf.push('\n');
    }
    let conf_bytes = conf.into_bytes();
    let mut files: Vec<FileSpec> = Vec::new();
    files.push(FileSpec { name: "limine.conf", data: &conf_bytes });
    files.push(FileSpec { name: "chitti-kernel", data: kernel });
    for (name, data) in &parts {
        files.push(FileSpec { name, data });
    }
    {
        let mut os = Partition::new(&mut dev, os_range.0, os_range.1 - os_range.0 + 1);
        if let Err(e) = Ext4Writer::format(&mut os, &files) {
            serial_println!("install> ext4 format/write failed: {:?}", e);
            return;
        }
    }
    serial_println!("install> ext4 OS partition written: limine.conf + kernel + {} model part(s).", parts.len());
    if let Some(layout) = fresh_layout {
        // Fresh install only: an empty ext4 data partition for durable agent
        // state (synapse::fs mounts it at boot, since it holds no *.gguf). An
        // update never touches it — the user home, agent state and user files
        // all survive `/install`. (The home's `.keep` markers are seeded by
        // `agent::home::ensure_user_home` on first boot, after the store
        // mounts this partition.)
        let mut data = Partition::new(&mut dev, layout.data_first, layout.data_last - layout.data_first + 1);
        if let Err(e) = Ext4Writer::format(&mut data, &[]) {
            serial_println!("install> ext4 data partition format failed: {:?}", e);
            return;
        }
        serial_println!("install> ext4 data partition (lba {}..{}) formatted for durable agent state.", layout.data_first, layout.data_last);
    } else {
        serial_println!("install> data partition preserved (home + agent state intact).");
    }
    serial_println!("install> DONE -- the disk now boots Chitti standalone via UEFI. Remove the ISO and reboot.");
}

/// aarch64 `/install`: make the target disk boot Chitti standalone via UEFI.
/// Layout: GPT with a FAT ESP carrying the Chitti UEFI stub (BOOTAA64.EFI) +
/// the kernel + the model — the stub reads all three off the ESP at boot — plus
/// an ext4 data partition for durable agent state. The installer payload is
/// read from the **boot ESP** this system was started from (the FAT volume
/// holding `chitti-kernel`), the aarch64 equivalent of the x86 path's Limine
/// payload modules.
#[cfg(target_arch = "aarch64")]
pub(super) fn disk_install(arg: &str) {
    use crate::block::{ext4::Ext4Writer, fat::FatWriter, fat_read::FatReader, gpt, BlockDevice, Partition};
    use crate::fs::detect::FsType;
    let a = parse_install_args(arg);
    let (pre_confirmed, force_format, target_override) = (a.pre_confirmed, a.force_format, a.target);
    if a.plan_only {
        install_plan(target_override.unwrap_or(0));
        return;
    }
    if a.alongside {
        // See `install_alongside`: the x86 path installs `BOOTX64.EFI` from a
        // Limine module. The ARM equivalent needs `BOOTAA64.EFI` and the aarch64
        // payload plumbing; the `plan` report works on both arches meanwhile.
        serial_println!("install> `alongside` is x86-only so far (ARM needs BOOTAA64.EFI); try `/install plan`");
        return;
    }
    // Identify the boot ESP (payload source): the FAT volume containing
    // `chitti-kernel`. Scan every disk's *volumes* (via the FS detector), so it
    // is found whether the ESP is a bare FAT disk (fresh `--uefi` boot) OR a GPT
    // partition (an already-installed disk — the common case). Its disk is never
    // a valid install target (we'd overwrite the payload we're reading).
    let mut esp: Option<(usize, u64, u64)> = None; // (disk, start_lba, sectors)
    'scan: for i in 0..16 {
        let Some(mut dev) = crate::block::probe_disk_nth(i) else { break };
        for v in crate::fs::detect::probe(&mut dev) {
            if matches!(v.fs, FsType::Fat16 | FsType::Fat32) {
                let mut part = Partition::new(&mut dev, v.start_lba, v.sectors);
                if let Some(mut r) = FatReader::open(&mut part) {
                    if r.exists("chitti-kernel") {
                        esp = Some((i, v.start_lba, v.sectors));
                        break 'scan;
                    }
                }
            }
        }
    }
    let Some((esp_idx, esp_lba, esp_sectors)) = esp else {
        serial_println!("install> no boot ESP found (a FAT volume with /chitti-kernel) -- boot via `--uefi` to install");
        return;
    };
    // Target: the explicit index if given, else the first non-ESP disk.
    let target_idx = match target_override {
        Some(n) => n,
        None => (0..16).find(|&i| i != esp_idx && crate::block::probe_disk_nth(i).is_some()).unwrap_or(esp_idx),
    };
    if target_idx == esp_idx {
        serial_println!("install> disk {} holds the boot ESP -- cannot install onto it (pick another; see /disks)", target_idx);
        return;
    }
    let Some(mut target) = crate::block::probe_disk_nth(target_idx) else {
        serial_println!("install> no disk {} (see /disks)", target_idx);
        return;
    };
    // Existing Chitti install on the target? Update in place: rewrite the ESP
    // (stub + kernel + model), preserve the ext4 data partition. `format`
    // forces a full repartition.
    let existing = gpt::read(&mut target).and_then(|(chitti, parts_read)| {
        if !chitti || force_format {
            return None;
        }
        parts_read.iter().find(|p| p.name == "EFI System").map(|p| (p.first_lba, p.last_lba))
    });
    if !confirm_install(pre_confirmed, existing.is_some(), target_idx) {
        return;
    }
    serial_println!("install> target disk {} (boot ESP is on disk {}, lba {})", target_idx, esp_idx, esp_lba);

    // Read the stub + kernel off the boot ESP partition. The model is NOT re-read
    // from FAT (it would not fit the 256 MiB heap): the stub already loaded it
    // into RAM at the fixed model address, so `cortex::model_module()` hands us
    // the exact bytes. (Reading the ESP disk + writing the target disk at the
    // same time is safe now that the NVMe controller is shared.)
    let Some(mut src_dev) = crate::block::probe_disk_nth(esp_idx) else { return };
    let mut esp_part = Partition::new(&mut src_dev, esp_lba, esp_sectors);
    let (stub, kernel, model_size) = {
        let Some(mut r) = FatReader::open(&mut esp_part) else {
            serial_println!("install> boot ESP unreadable");
            return;
        };
        let Some(stub) = r.read_file("EFI/BOOT/BOOTAA64.EFI") else {
            serial_println!("install> BOOTAA64.EFI missing from the boot ESP");
            return;
        };
        let Some(kernel) = r.read_file("chitti-kernel") else {
            serial_println!("install> chitti-kernel missing from the boot ESP");
            return;
        };
        (stub, kernel, r.file_size("model.gguf.000"))
    };
    // The model's bytes are already in RAM (the stub loaded them at the fixed
    // model address); `model_module()` exposes the RAM window, and the FAT
    // directory entry gives the file's true size to slice it by.
    let model: Option<&'static [u8]> = match (crate::cortex::model_module(), model_size) {
        (Some(m), Some(sz)) if (sz as usize) <= m.len() => Some(&m[..sz as usize]),
        _ => None,
    };
    let model_len = model.map(|m| m.len()).unwrap_or(0);
    serial_println!(
        "install> payload from boot ESP: stub {} B, kernel {} B, model {} B",
        stub.len(),
        kernel.len(),
        model_len
    );

    // 1. Partitions: reuse the existing ESP range on an update; otherwise
    //    write a fresh GPT (ESP sized for the payload + ext4 data).
    let total = target.block_count();
    let esp_bytes = (stub.len() + kernel.len() + model_len) as u64;
    let (esp_range, fresh_data) = match existing {
        Some((first, last)) => {
            let cap = (last - first + 1) * 512;
            if cap < esp_bytes {
                serial_println!("install> existing ESP too small ({} B for a {} B payload) -- re-run with 'format'", cap, esp_bytes);
                return;
            }
            serial_println!("install> existing Chitti install detected -- updating the ESP in place (data preserved)");
            ((first, last), None)
        }
        None => {
            let Some(parts) = gpt::esp_data_parts(total, esp_bytes) else {
                serial_println!("install> target disk too small ({} sectors for a {} B payload)", total, esp_bytes);
                return;
            };
            if let Err(e) = gpt::write(&mut target, &parts) {
                serial_println!("install> GPT write failed: {:?}", e);
                return;
            }
            serial_println!(
                "install> GPT: ESP lba {}..{}, ext4 data lba {}..{}",
                parts[0].first_lba,
                parts[0].last_lba,
                parts[1].first_lba,
                parts[1].last_lba
            );
            ((parts[0].first_lba, parts[0].last_lba), Some((parts[1].first_lba, parts[1].last_lba)))
        }
    };

    // 2. FAT ESP: the stub at /EFI/BOOT/BOOTAA64.EFI + kernel + model at the
    //    root (exactly where the stub looks).
    {
        let mut esp = Partition::new(&mut target, esp_range.0, esp_range.1 - esp_range.0 + 1);
        let r = FatWriter::format(&mut esp).and_then(|mut fw| {
            fw.write_efi_boot_file("BOOTAA64.EFI", &stub)?;
            fw.write_root_file("chitti-kernel", &kernel)?;
            if let Some(m) = model {
                fw.write_root_file("model.gguf.000", m)?;
            }
            Ok(())
        });
        if let Err(e) = r {
            serial_println!("install> ESP FAT write failed: {:?}", e);
            return;
        }
    }
    serial_println!("install> ESP (FAT): BOOTAA64.EFI + kernel{} written.", if model.is_some() { " + model" } else { "" });

    // 3. Fresh install only: an empty ext4 data partition for durable agent
    //    state. An update never touches it.
    if let Some((first, last)) = fresh_data {
        let mut data = Partition::new(&mut target, first, last - first + 1);
        if let Err(e) = Ext4Writer::format(&mut data, &[]) {
            serial_println!("install> ext4 data partition format failed: {:?}", e);
            return;
        }
        serial_println!("install> ext4 data partition formatted for durable agent state.");
    } else {
        serial_println!("install> data partition preserved (agent state intact).");
    }
    serial_println!("install> DONE -- the disk now boots Chitti standalone via UEFI. Reboot with --disk-only.");
}

pub(super) fn disk_mkext4(arg: &str) {
    use crate::block::ext4::{Ext4Writer, FileSpec};
    let a = arg.trim();
    // Destructive: confirmed via the permission modal ('yes'/'empty' inline
    // still accepted as a scripted pre-confirmation).
    if a != "yes" && a != "empty" {
        let ok = crate::modal::confirm(
            "Format disk as ext4 \u{2014} confirm?",
            "This ERASES the whole disk and formats it ext4 (with 2 test files). Proceed?",
        );
        if !ok {
            serial_println!("mkext4> aborted (not confirmed; scripted: /mkext4 yes | empty)");
            return;
        }
    }
    let Some(mut dev) = crate::block::probe_disk() else {
        serial_println!("mkext4> no block device");
        return;
    };
    if a == "empty" {
        match Ext4Writer::format(&mut dev, &[]) {
            Ok(()) => serial_println!("mkext4> formatted an empty ext4 (0 files) -- the /install data-partition case."),
            Err(e) => serial_println!("mkext4> empty ext4 format failed: {:?}", e),
        }
        return;
    }
    // A small file + a ~200 KiB file (forces single-indirect blocks).
    let hello = b"hello from Chitti's from-scratch ext4 writer\n";
    let big: alloc::vec::Vec<u8> = (0..200_000u32).map(|i| ((i.wrapping_mul(7)) & 0xff) as u8).collect();
    let files = [
        FileSpec { name: "hello.txt", data: &hello[..] },
        FileSpec { name: "big.bin", data: &big[..] },
    ];
    match Ext4Writer::format(&mut dev, &files) {
        Ok(()) => serial_println!("mkext4> formatted ext4 + wrote hello.txt (45 B) + big.bin (200000 B)."),
        Err(e) => serial_println!("mkext4> ext4 format failed: {:?}", e),
    }
}

pub(super) fn disk_ext4read() {
    use crate::block::ext4_read::Ext4Reader;
    let Some(mut dev) = crate::block::probe_disk() else {
        serial_println!("ext4read> no block device");
        return;
    };
    let Some(mut r) = Ext4Reader::open(&mut dev) else {
        serial_println!("ext4read> not an ext filesystem at LBA 0 (try /mkext4 yes first)");
        return;
    };
    serial_println!("ext4read> block_size={}", r.block_size);
    for (name, ino, is_dir) in r.list_root() {
        serial_println!("  {}{}  (inode {})", name, if is_dir { "/" } else { "" }, ino);
    }
    // Verify hello.txt round-trips.
    let mut buf = [0u8; 128];
    if let Some(n) = r.read_root_file("hello.txt", &mut buf) {
        serial_println!("ext4read> hello.txt ({} B): {}", n, core::str::from_utf8(&buf[..n]).unwrap_or("?"));
    }
    // Verify big.bin (200000 B) byte-for-byte against the /mkext4 pattern.
    if let Some(sz) = r.file_size("big.bin") {
        let mut big = alloc::vec![0u8; sz as usize];
        let n = r.read_root_file("big.bin", &mut big).unwrap_or(0);
        let ok = n == 200_000 && big.iter().enumerate().all(|(i, &b)| b == ((i as u32).wrapping_mul(7) & 0xff) as u8);
        serial_println!("ext4read> big.bin {} B, pattern match: {}", n, ok);
    }
}
