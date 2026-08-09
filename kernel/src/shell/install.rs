//! install
//!
//! The **self-hosting installer** carved out of the former 16k-line
//! `shell/mod.rs` monolith: `/install`, `/install plan` and
//! `/install alongside` (see `block::gpt` + `block::esp`), plus `/mkfs`
//! and `/ext4read` dev helpers. Moved verbatim; `use super::*` keeps the
//! parent's statics visible, and the parent re-imports this module's items
//! with `pub(crate) use install::*`.

use super::*;
// Explicit, because `use super::*` glob-imports a private `Vec` from
// `framebuffer::views` that shadows the prelude's. An explicit import wins over
// a glob, so this is what makes `Vec` mean `alloc::vec::Vec` in this file.
use alloc::string::String;
use alloc::vec::Vec;

// ── Carrying the live system onto a freshly formatted data partition ──────
//
// A **fresh** `/install` used to format the data partition with `&[]` — an empty
// volume — so everything the human had set up on the machine they were installing
// *from* was discarded: their theme, their login password, shell history, saved
// sessions, agent memory, permissions, downloads. The installed system booted
// blank, and from the outside that reads exactly as "/install overwrote my
// configs". (The *update* path never had this problem: it leaves the existing
// data partition alone.)
//
// `Ext4Store` keeps synapse keys as flat root-level ext4 files with `/`
// percent-encoded, so a fresh volume can be seeded through `Ext4Writer::format`
// by writing `key_encode(key)` names — exactly what a later `Ext4Store::mount`
// decodes back.

/// How many bytes of live store a fresh install will carry across.
///
/// A bound rather than "everything", because every carried file has to be held
/// in RAM at once while the volume is formatted, and this kernel's allocator is
/// a first-fit list sharing a heap with a loaded model. Sixteen mebibytes is far
/// above the things people actually lose (`/configs/**` and the credential are
/// kilobytes; history and sessions are small) and well below trouble.
pub(super) const CARRY_BUDGET: usize = 16 * 1024 * 1024;

/// Why a key was not carried. Reported, never silent — a migration that quietly
/// drops half your files is worse than one that refuses.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub(super) enum Dropped {
    /// `/samples/**` is re-seeded from the image on the next boot, so copying it
    /// would spend most of the budget on files the installed system recreates.
    Regenerated,
    /// Carrying it would exceed [`CARRY_BUDGET`].
    OverBudget,
}

/// Decide what a fresh install carries, given every store key and its size.
///
/// Pure, so the policy is testable without a disk. Smallest first, which is not
/// a micro-optimisation: it means the small files people care about — the theme,
/// the login record, shell history, permissions — are carried even when one huge
/// download would otherwise have eaten the whole budget. Ties break on the key so
/// two runs of the same store produce the same volume.
pub(super) fn plan_carry(
    entries: &[(String, usize)],
    budget: usize,
) -> (Vec<String>, Vec<(String, Dropped)>) {
    let mut keep = Vec::new();
    let mut drop = Vec::new();
    let mut ordered: Vec<&(String, usize)> = Vec::new();
    for e in entries {
        if e.0.starts_with("/samples/") {
            drop.push((e.0.clone(), Dropped::Regenerated));
        } else {
            ordered.push(e);
        }
    }
    ordered.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    let mut used = 0usize;
    for (key, size) in ordered {
        if used.saturating_add(*size) > budget {
            drop.push((key.clone(), Dropped::OverBudget));
            continue;
        }
        used += *size;
        keep.push(key.clone());
    }
    keep.sort();
    drop.sort();
    (keep, drop)
}

/// Read the live store into the `(encoded name, bytes)` pairs a fresh data
/// partition is formatted with, and report what was left behind.
///
/// **Holds [`CredentialAccess`] for the whole walk**, and that is the load-bearing
/// line here. `synapse::fs::list` filters the login credential record out and
/// `read` refuses it — deliberately, so no agent can enumerate or exfiltrate the
/// verifier — which means a migration written the obvious way copies everything
/// *except* the password and silently produces a machine whose owner cannot log
/// in. The guard is exactly what it is for: kernel-side code, on a path a human
/// typed, that legitimately needs the record.
fn read_carry() -> (Vec<(String, Vec<u8>)>, Vec<(String, Dropped)>) {
    let _access = crate::synapse::fs::CredentialAccess::new();
    let keys = crate::synapse::fs::list();
    let mut sized: Vec<(String, usize)> = Vec::with_capacity(keys.len());
    for k in &keys {
        let n = crate::synapse::fs::size_of(k).unwrap_or(0);
        sized.push((k.clone(), n));
    }
    let (keep, dropped) = plan_carry(&sized, CARRY_BUDGET);
    let mut out = Vec::with_capacity(keep.len());
    for k in keep {
        if let Some(bytes) = crate::synapse::fs::read(&k) {
            out.push((crate::block::ext4_store::key_encode(&k), bytes));
        }
    }
    (out, dropped)
}

/// Format `data` as the new store, carrying the live one across, and say what
/// happened. Used only on a **fresh** install; an update never reaches here.
fn format_data_with_carry<D: crate::block::BlockDevice>(data: &mut D) -> Result<(), crate::block::BlockError> {
    use crate::block::ext4::{Ext4Writer, FileSpec};
    let (carried, dropped) = read_carry();
    let bytes: usize = carried.iter().map(|(_, v)| v.len()).sum();
    let files: Vec<FileSpec> = carried
        .iter()
        .map(|(name, data)| FileSpec { name, data })
        .collect();
    Ext4Writer::format(data, &files)?;
    serial_println!(
        "install> data partition formatted, carrying {} file(s) ({} KiB) from this system \
         — theme, login password, history, sessions and agent state come with it.",
        files.len(),
        bytes / 1024
    );
    let regen = dropped.iter().filter(|(_, d)| *d == Dropped::Regenerated).count();
    if regen > 0 {
        serial_println!("install>   {regen} sample file(s) not copied — the installed system re-seeds them at boot.");
    }
    let over: Vec<&(String, Dropped)> = dropped.iter().filter(|(_, d)| *d == Dropped::OverBudget).collect();
    if !over.is_empty() {
        // Named, not counted: "3 files were too big" tells you nothing about
        // whether you just lost something you needed.
        serial_println!("install>   {} file(s) exceeded the {} MiB migration budget and were NOT copied:", over.len(), CARRY_BUDGET / 1024 / 1024);
        for (k, _) in over.iter().take(8) {
            serial_println!("install>     {k}");
        }
        if over.len() > 8 {
            serial_println!("install>     … and {} more", over.len() - 8);
        }
    }
    Ok(())
}

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
        // Fresh install only: the ext4 data partition for durable agent state
        // (synapse::fs mounts it at boot, since it holds no *.gguf). An update
        // never touches it — the user home, agent state and user files all
        // survive `/install`.
        //
        // Seeded with the live store rather than formatted empty. Installing is
        // something you do *after* setting a machine up, so an empty volume threw
        // away the theme, the login password, the history and every config of the
        // system you were installing from. (The home's `.keep` markers are still
        // seeded by `agent::home::ensure_user_home` on first boot for the case
        // where there was nothing to carry.)
        let mut data = Partition::new(&mut dev, layout.data_first, layout.data_last - layout.data_first + 1);
        if let Err(e) = format_data_with_carry(&mut data) {
            serial_println!("install> ext4 data partition format failed: {:?}", e);
            return;
        }
        serial_println!("install> data partition at lba {}..{}.", layout.data_first, layout.data_last);
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

    // 3. Fresh install only: the ext4 data partition for durable agent state,
    //    seeded with the live store rather than formatted empty (see the x86
    //    path). An update never touches it.
    if let Some((first, last)) = fresh_data {
        let mut data = Partition::new(&mut target, first, last - first + 1);
        if let Err(e) = format_data_with_carry(&mut data) {
            serial_println!("install> ext4 data partition format failed: {:?}", e);
            return;
        }
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

pub(super) fn disk_mkexfat(arg: &str) {
    use crate::block::exfat_rw;
    // `/mkexfat [<disk>] [yes|empty]` — the disk index defaults to 0 (the
    // first block device, matching /mkext4); an explicit index lets a script
    // format a *specific* attached disk (the e2e harness formats its own).
    let mut words = arg.split_whitespace();
    let first = words.next().unwrap_or("");
    let mut disk = 0usize;
    let mut confirm = first;
    if let Ok(n) = first.parse::<usize>() {
        disk = n;
        confirm = words.next().unwrap_or("");
    }
    // Destructive: confirmed via the permission modal ('yes'/'empty' inline
    // still accepted as a scripted pre-confirmation).
    if confirm != "yes" && confirm != "empty" {
        let ok = crate::modal::confirm(
            "Format disk as exFAT \u{2014} confirm?",
            "This ERASES the whole disk and formats it exFAT. Proceed?",
        );
        if !ok {
            serial_println!("mkexfat> aborted (not confirmed; scripted: /mkexfat yes | empty)");
            return;
        }
    }
    let Some(mut dev) = crate::block::probe_disk_nth(disk) else {
        serial_println!("mkexfat> no block device at disk {disk}");
        return;
    };
    // `empty` = no files written; otherwise write a hello + a larger file as a
    // formatter smoke test, mirroring /mkext4.
    let label = if confirm == "empty" { "" } else { "CHITTI" };
    match exfat_rw::format(&mut dev, label) {
        Ok(()) => {
            serial_println!("mkexfat> formatted disk {disk} exFAT (label {:?}).", if label.is_empty() { "none" } else { label });
            if confirm != "empty" {
                let big: alloc::vec::Vec<u8> = (0..200_000u32).map(|i| ((i.wrapping_mul(7)) & 0xff) as u8).collect();
                match exfat_rw::ExfatRw::open(&mut dev, true).and_then(|mut vol| {
                    vol.write("hello.txt", b"hello from Chitti's exFAT writer\n")?;
                    vol.write("big.bin", &big)
                }) {
                    Ok(()) => serial_println!("mkexfat> wrote hello.txt + big.bin (200000 B)."),
                    Err(e) => serial_println!("mkexfat> format ok, smoke-test write failed: {:?}", e),
                }
            }
        }
        Err(e) => serial_println!("mkexfat> format failed: {:?}", e),
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

#[cfg(test)]
mod tests {
    use super::*;

    fn e(k: &str, n: usize) -> (String, usize) {
        (String::from(k), n)
    }

    /// The whole point of the migration: a fresh `/install` must carry the small
    /// things a human actually configured. Before this, the data partition was
    /// formatted with `&[]` and every one of these was silently gone on the
    /// installed machine.
    #[test_case]
    fn a_fresh_install_carries_the_configs_the_credential_and_the_history() {
        let store = alloc::vec![
            e("/configs/core/ui.json", 900),
            e("/configs/core/auth.json", 300),
            e("/configs/core/display.json", 200),
            e("/configs/core/panes.json", 200),
            e("/configs/themes/mine.json", 1200),
            e("/history", 4096),
            e("/sessions/1", 20_000),
            e("/agent/9001/memory/notes", 512),
        ];
        let (keep, dropped) = plan_carry(&store, CARRY_BUDGET);
        assert!(dropped.is_empty(), "a tiny store dropped something: {dropped:?}");
        for k in [
            "/configs/core/ui.json",
            "/configs/core/auth.json",
            "/configs/themes/mine.json",
            "/history",
            "/sessions/1",
            "/agent/9001/memory/notes",
        ] {
            assert!(keep.iter().any(|x| x == k), "{k} was not carried");
        }
    }

    /// `/samples/**` is re-seeded from the image at boot, so copying it would
    /// spend the budget recreating files the installed system recreates anyway.
    #[test_case]
    fn samples_are_left_behind_because_boot_re_seeds_them() {
        let store = alloc::vec![
            e("/samples/videos/big.mp4", 8_000_000),
            e("/samples/images/fruits.jpg", 400_000),
            e("/configs/core/ui.json", 900),
        ];
        let (keep, dropped) = plan_carry(&store, CARRY_BUDGET);
        assert_eq!(keep, alloc::vec![String::from("/configs/core/ui.json")]);
        assert_eq!(dropped.len(), 2);
        assert!(dropped.iter().all(|(_, d)| *d == Dropped::Regenerated));
        // A path that merely *mentions* samples is not a sample.
        let (keep, _) = plan_carry(&alloc::vec![e("/home/chitti/samples-notes.md", 10)], CARRY_BUDGET);
        assert_eq!(keep.len(), 1, "a file outside /samples/ was treated as one");
    }

    /// Smallest-first is the load-bearing part: one oversized download must not
    /// cost the human their theme and their password.
    #[test_case]
    fn one_huge_file_cannot_crowd_out_the_small_ones() {
        let store = alloc::vec![
            e("/downloads/movie.mp4", 900),
            e("/configs/core/ui.json", 100),
            e("/configs/core/auth.json", 100),
        ];
        let (keep, dropped) = plan_carry(&store, 300);
        assert!(keep.iter().any(|k| k == "/configs/core/ui.json"));
        assert!(keep.iter().any(|k| k == "/configs/core/auth.json"));
        assert_eq!(dropped, alloc::vec![(String::from("/downloads/movie.mp4"), Dropped::OverBudget)]);
    }

    /// Nothing is dropped without being reported, and the split is exhaustive —
    /// every key ends up in exactly one of the two lists.
    #[test_case]
    fn every_key_is_either_carried_or_reported() {
        let store = alloc::vec![
            e("/a", 10),
            e("/b", 10_000_000),
            e("/samples/x", 5),
            e("/c", 10),
        ];
        let (keep, dropped) = plan_carry(&store, 100);
        assert_eq!(keep.len() + dropped.len(), store.len(), "a key vanished from both lists");
        for (k, _) in &store {
            let carried = keep.iter().any(|x| x == k);
            let reported = dropped.iter().any(|(x, _)| x == k);
            assert!(carried ^ reported, "{k} is in both lists or neither");
        }
    }

    /// Same store in, same volume out — two installs of one machine must not
    /// disagree about which files made it.
    #[test_case]
    fn the_selection_is_deterministic_including_size_ties() {
        let a = alloc::vec![e("/z", 50), e("/a", 50), e("/m", 50)];
        let b = alloc::vec![e("/m", 50), e("/z", 50), e("/a", 50)];
        // Budget fits two of the three, so the tie-break actually decides.
        assert_eq!(plan_carry(&a, 100).0, plan_carry(&b, 100).0);
        assert_eq!(plan_carry(&a, 100).0, alloc::vec![String::from("/a"), String::from("/m")]);
    }

    /// An empty store is a fresh machine, not an error.
    #[test_case]
    fn an_empty_store_carries_nothing_and_reports_nothing() {
        let (keep, dropped) = plan_carry(&[], CARRY_BUDGET);
        assert!(keep.is_empty() && dropped.is_empty());
    }

    /// **The end-to-end proof**, against a real ext4 volume rather than the
    /// selection policy alone: put files in the live store, run the exact code
    /// `/install` runs on a fresh disk, then read the resulting volume back and
    /// check every key is there under the name `Ext4Store` will decode.
    ///
    /// The credential is the case that matters. `synapse::fs::list` filters it
    /// out and `read` refuses it, so a migration written without holding
    /// `CredentialAccess` copies everything *except* the password — and produces
    /// an installed machine its owner cannot log into, with no error anywhere.
    #[test_case]
    fn a_fresh_install_volume_round_trips_the_store_including_the_credential() {
        use crate::block::ext4_rw::Ext4Rw;
        use crate::block::ext4_store::key_encode;
        use crate::block::ramdisk::RamDisk;

        // A theme, a session, and the login record — the things the user said
        // were being lost.
        crate::synapse::fs::write("/configs/core/ui.json", b"{\"theme\":\"nord\"}");
        crate::synapse::fs::write("/sessions/carry-probe", b"a saved session");
        {
            let _a = crate::synapse::fs::CredentialAccess::new();
            crate::synapse::fs::write(crate::auth::PATH, b"{\"kdf\":\"pbkdf2-hmac-sha256\"}");
        }

        // 16 MiB volume — enough for ext4 metadata plus the payload.
        let mut disk = RamDisk::new(32768);
        format_data_with_carry(&mut disk).expect("format with carry failed");

        let mut rw = Ext4Rw::open(&mut disk).expect("the written volume is not readable ext4");
        for (key, want) in [
            ("/configs/core/ui.json", &b"{\"theme\":\"nord\"}"[..]),
            ("/sessions/carry-probe", &b"a saved session"[..]),
            (crate::auth::PATH, &b"{\"kdf\":\"pbkdf2-hmac-sha256\"}"[..]),
        ] {
            let name = alloc::format!("/{}", key_encode(key));
            let got = rw
                .read(&name)
                .unwrap_or_else(|e| panic!("{key} did not survive the install ({e:?})"));
            assert_eq!(got, want, "{key} came across with the wrong bytes");
        }

        {
            let _a = crate::synapse::fs::CredentialAccess::new();
            crate::synapse::fs::delete(crate::auth::PATH);
        }
        crate::synapse::fs::delete("/configs/core/ui.json");
        crate::synapse::fs::delete("/sessions/carry-probe");
    }

    /// The names written to the new volume must be exactly what `Ext4Store`
    /// decodes back, or the installed system mounts a store full of keys nobody
    /// asked for.
    #[test_case]
    fn carried_names_round_trip_through_the_store_encoding() {
        use crate::block::ext4_store::{key_decode, key_encode};
        for k in [
            "/configs/core/auth.json",
            "/agent/9001/memory/notes",
            "/home/chitti/a%b",
            "/history",
        ] {
            assert_eq!(key_decode(&key_encode(k)), k, "{k} did not round trip");
        }
    }
}
