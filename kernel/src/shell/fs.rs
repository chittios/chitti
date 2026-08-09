//! fs
//!
//! The **storage / filesystem / channel** command surface carved out of the
//! former 16k-line `shell/mod.rs` monolith: `/mount` + `/disk` + `/fs`
//! commands over `crate::fs::{mount,vfs}`, the mount-path helpers, and the
//! `/channel` external-messaging surface (`msgchan`). Moved verbatim;
//! `use super::*` keeps the parent's statics visible, and the parent
//! re-imports this module's items with `pub(crate) use fs::*`.

use super::*;

// --- block-device / filesystem commands (via `crate::fs::{mount,vfs}`) ----

/// `/mount <disk> [vol] [/path]` — bind volume `vol` (default 0) of disk `disk`
/// to a mount path (default the first free `/mnt`, `/mnt2`, …).
pub(super) fn disk_mount(arg: &str) {
    use alloc::string::String;
    let mut disk: Option<usize> = None;
    let mut vol: usize = 0;
    let mut path: Option<String> = None;
    // Default RW for FAT/ext; `ro` forces read-only. NTFS always RO for now.
    let mut want_rw = true;
    let mut nums = 0;
    for tok in arg.split_whitespace() {
        if tok == "ro" {
            want_rw = false;
        } else if tok == "rw" {
            want_rw = true;
        } else if let Some(p) = tok.strip_prefix('/') {
            path = Some(alloc::format!("/{p}"));
        } else if let Ok(n) = tok.parse::<usize>() {
            if nums == 0 {
                disk = Some(n);
            } else {
                vol = n;
            }
            nums += 1;
        }
    }
    let Some(disk) = disk else {
        serial_println!(
            "mount> usage: /mount <disk> [vol] [/path] [rw|ro]\n\
             mount> FAT/ext/exFAT default rw; NTFS mounts read-only"
        );
        return;
    };
    match crate::fs::mount::mount(disk, vol, path.as_deref(), want_rw) {
        Ok(mt) => {
            serial_println!(
                "mount> {} -> disk {} ({}, {} MiB, label={}, {})",
                mt.path,
                mt.disk,
                mt.fs.name(),
                mt.sectors * 512 / 1024 / 1024,
                mt.label.as_deref().unwrap_or("-"),
                if mt.writable { "rw" } else { "ro" }
            );
            if want_rw && !mt.writable {
                serial_println!(
                    "mount> note: {} is mounted read-only (write support not implemented)",
                    mt.fs.name()
                );
            }
        }
        Err(crate::fs::mount::MountError::NoDisk) => {
            serial_println!("mount> no disk {} (see /disks)", disk);
        }
        Err(crate::fs::mount::MountError::NoVolume) => {
            serial_println!("mount> disk {} has no volume {} (see /disks)", disk, vol);
        }
        Err(crate::fs::mount::MountError::Busy) => {
            serial_println!("mount> path already mounted (/umount it first)");
        }
        Err(crate::fs::mount::MountError::Unsupported) => {
            serial_println!(
                "mount> unsupported filesystem (FAT/ext/exFAT: rw; NTFS: ro list+read)"
            );
        }
        Err(e) => serial_println!("mount> failed: {e:?}"),
    }
}

/// `/umount <path>` — remove a mount.
pub(super) fn disk_umount(arg: &str) {
    let path = arg.trim();
    match crate::fs::mount::umount(path) {
        Ok(()) => serial_println!("umount> {} unmounted", path),
        Err(_) => serial_println!("umount> {} not mounted (see /mounts)", path),
    }
}

/// `/mounts` — list the mount table.
pub(super) fn disk_mounts() {
    let m = crate::fs::mount::list();
    if m.is_empty() {
        serial_println!("mounts> (nothing mounted; /mount <disk> [vol] [/path])");
        return;
    }
    for mt in m.iter() {
        // A host shared folder has no disk, LBA or size, and printing the
        // sentinel index for them reads as a corrupt mount table.
        if crate::fs::host::is_host(mt) {
            serial_println!(
                "  {:<8} host folder                             {:<8} {} tag={}",
                mt.path,
                mt.fs.name(),
                if mt.writable { "rw" } else { "ro" },
                mt.label.as_deref().unwrap_or("-")
            );
            continue;
        }
        serial_println!(
            "  {:<8} disk {} lba {:<10} {:>6} MiB  {:<8} {} label={}",
            mt.path,
            mt.disk,
            mt.start_lba,
            mt.sectors * 512 / 1024 / 1024,
            mt.fs.name(),
            if mt.writable { "rw" } else { "ro" },
            mt.label.as_deref().unwrap_or("-")
        );
    }
}

/// `/encrypt <disk> [vol]` — format a **data** partition as C4VE (AES-XTS) and
/// put an empty ext4 on the payload. **Human-only**, destructive: existing
/// contents of that volume are wiped. Agent tools must not call this.
pub(super) fn disk_encrypt(arg: &str) {
    use crate::block::ext4::Ext4Writer;
    use crate::block::volcrypto::{self, DEFAULT_HDR_SECTORS, DEFAULT_ITERATIONS};
    use crate::block::Partition;

    let mut nums = arg.split_whitespace().filter_map(|t| t.parse::<usize>().ok());
    let Some(disk) = nums.next() else {
        serial_println!("encrypt> usage: /encrypt <disk> [vol]   (human-only; wipes the volume)");
        return;
    };
    let vol = nums.next().unwrap_or(0);
    if !crate::modal::confirm(
        "Encrypt volume",
        &alloc::format!("Wipe disk {disk} vol {vol} and encrypt with AES-XTS?"),
    ) {
        serial_println!("encrypt> cancelled");
        return;
    }
    let pass = crate::modal::input("Set passphrase", "New passphrase:", true);
    if pass.is_empty() {
        serial_println!("encrypt> empty passphrase refused");
        return;
    }
    let pass2 = crate::modal::input("Confirm passphrase", "Repeat:", true);
    if pass != pass2 {
        serial_println!("encrypt> passphrases do not match");
        return;
    }
    let Some(mut dev) = crate::block::probe_disk_nth(disk) else {
        serial_println!("encrypt> no disk {disk}");
        return;
    };
    let vols = crate::fs::detect::probe(&mut dev);
    let Some(v) = vols.get(vol).cloned() else {
        serial_println!("encrypt> no volume {vol} on disk {disk}");
        return;
    };
    let mut part = Partition::new(&mut dev, v.start_lba, v.sectors);
    let key = match volcrypto::format(&mut part, pass.as_bytes(), DEFAULT_ITERATIONS, DEFAULT_HDR_SECTORS)
    {
        Ok(k) => k,
        Err(e) => {
            serial_println!("encrypt> format header failed: {e:?}");
            return;
        }
    };
    // Empty ext4 on the encrypted payload.
    let mut payload =
        volcrypto::CryptoPart::encrypted(&mut dev, v.start_lba, v.sectors, DEFAULT_HDR_SECTORS, key);
    if let Err(e) = Ext4Writer::format(&mut payload, &[]) {
        serial_println!("encrypt> payload mkfs failed: {e:?}");
        return;
    }
    serial_println!(
        "encrypt> disk {disk} vol {vol} is C4VE + empty ext4 ({} MiB payload). Reboot and unlock, or /unlock.",
        (v.sectors.saturating_sub(DEFAULT_HDR_SECTORS)) * 512 / 1024 / 1024
    );
}

/// `/unlock [disk] [vol]` — unlock a C4VE data volume and adopt it as the
/// synapse store (if not already mounted). Human-only.
pub(super) fn disk_unlock(arg: &str) {
    use crate::block::ext4_store::Ext4Store;
    use crate::block::volcrypto;
    use crate::block::Partition;

    let mut nums = arg.split_whitespace().filter_map(|t| t.parse::<usize>().ok());
    let disk = nums.next().unwrap_or(0);
    let vol = nums.next().unwrap_or(0);
    let pass = crate::modal::input("Unlock volume", "Passphrase:", true);
    if pass.is_empty() {
        serial_println!("unlock> cancelled");
        return;
    }
    let Some(mut dev) = crate::block::probe_disk_nth(disk) else {
        serial_println!("unlock> no disk {disk}");
        return;
    };
    let vols = crate::fs::detect::probe(&mut dev);
    let Some(v) = vols.get(vol).cloned() else {
        serial_println!("unlock> no volume {vol}");
        return;
    };
    let start = v.start_lba;
    let count = v.sectors;
    let (key, hdr) = {
        let mut part = Partition::new(&mut dev, start, count);
        match volcrypto::unlock(&mut part, pass.as_bytes()) {
            Ok(x) => x,
            Err(_) => {
                serial_println!("unlock> wrong passphrase or not a C4VE volume");
                return;
            }
        }
    };
    drop(dev); // free the handle so mount can re-probe the same disk
    match Ext4Store::mount_encrypted(disk, start, count, key, hdr.hdr_sectors) {
        Some(store) => {
            crate::synapse::fs::mount_ext4(store);
            serial_println!(
                "unlock> adopted encrypted store (disk {disk} vol {vol}, hdr {} sectors)",
                hdr.hdr_sectors
            );
        }
        None => serial_println!("unlock> unlocked but payload is not a readable ext4"),
    }
}

/// List the root of a non-store mount via the VFS.
pub(super) fn ls_mount(path: &str) {
    if path == "/" {
        fs_ls("/");
        return;
    }
    match crate::fs::vfs::readdir(path) {
        Ok(entries) => {
            serial_println!("ls> {} ({} entries):", path, entries.len());
            for e in entries.into_iter().take(64) {
                if e.is_dir {
                    serial_println!("  {}/", e.name);
                } else if e.size > 0 {
                    serial_println!("  {} ({} bytes)", e.name, e.size);
                } else {
                    serial_println!("  {}", e.name);
                }
            }
        }
        Err(e) => serial_println!("ls> {path}: {e:?}"),
    }
}

// --- store filesystem commands (Linux-like over synapse::fs) -------------

/// Parse flags from a shell arg line. Returns `(flags, positionals)`.
pub(super) fn fs_split_flags(arg: &str) -> (alloc::vec::Vec<char>, alloc::vec::Vec<alloc::string::String>) {
    let mut flags = alloc::vec::Vec::new();
    let mut pos = alloc::vec::Vec::new();
    for tok in arg.split_whitespace() {
        if tok == "--" {
            continue;
        }
        if let Some(rest) = tok.strip_prefix('-') {
            if !rest.is_empty() && rest.chars().all(|c| c.is_ascii_alphabetic()) {
                for c in rest.chars() {
                    flags.push(c);
                }
                continue;
            }
        }
        pos.push(alloc::string::String::from(tok));
    }
    (flags, pos)
}

/// Is `path` a directory we could list — store, disk mount, or foreign VFS?
///
/// Deliberately the same three cases [`fs_ls`] treats as listable, and in the
/// same order: `/cd` refusing something `/ls` would happily show is worse than
/// the missing check it replaces.
pub(super) fn is_listable_dir(path: &str) -> bool {
    if path == "/" {
        return true;
    }
    if crate::synapse::fs::is_dir(path) {
        return true;
    }
    if crate::fs::mount::by_path(path).is_some() {
        return true;
    }
    crate::fs::vfs::readdir(path).is_ok()
        && (path_on_mount(path) || crate::fs::mount::resolve(path).is_some())
}

/// `/cd [dir]` — move the shell's working directory.
///
/// **The argument is an ordinary path and goes through `resolve_path` like every
/// other one.** It used to be handed to `set_shell_cwd` raw, which normalises but
/// does not resolve — and `vpath::normalize` leaves a bare name relative. So
/// `/cd gobreaker` inside `/home/chitti` stored the cwd as the *relative string*
/// `gobreaker`, and from then on every path command resolved against a base that
/// was itself unresolved: `/ls` reported `gobreaker: no such file or directory`
/// while the prompt claimed to be inside it. `/cd ..` was wrong the same way,
/// landing at the store root instead of the parent. Every `resolve_path` test
/// sets an absolute cwd, which is why the resolver stayed correct and the thing
/// feeding it did not.
///
/// A missing target is now **refused**, as `cd` does everywhere else. Silently
/// accepting it is what made the original failure so hard to read: `cd> gobreaker`
/// looked exactly like success, and the complaint surfaced one command later
/// against `/ls`, which was never wrong.
pub(super) fn fs_cd(arg: &str) {
    let arg = arg.trim();
    // Bare `/cd` and `/cd ~` go home, like a login shell. Everything else —
    // including `.` (stay) and `/` (the store root) — is an ordinary path.
    let target = if arg.is_empty() || arg == "~" {
        crate::agent::home::USER_HOME.to_string()
    } else {
        super::resolve_path(arg)
    };
    if !is_listable_dir(&target) {
        // A typo and a file are different mistakes; name which one it is.
        if crate::synapse::fs::is_file(&target) || crate::fs::vfs::read_mount(&target).is_ok() {
            serial_println!("cd> {target}: not a directory");
            crate::shell::status::fail1();
        } else {
            serial_println!("cd> {target}: no such file or directory");
            crate::shell::status::fail1();
        }
        return;
    }
    super::set_shell_cwd(&target);
    serial_println!("cd> {}", super::shell_cwd());
}

/// `/ls [path] [-l] [-h]` — hierarchical listing of the store (default: the pwd).
/// `-l` (or `-1`) is the long form, `-h` gives its sizes in `ls -h` units.
/// Also: `/ls <n>` lists volume *n* on disk 0; `/ls /mnt` lists a non-store mount.
pub(super) fn fs_ls(arg: &str) {
    let (flags, pos) = fs_split_flags(arg);
    let long = flags.contains(&'l') || flags.contains(&'1');
    // `-h` was parsed and then dropped on the floor, so `/ls -lah` printed raw
    // byte counts and gave no hint that the flag meant nothing.
    let human = flags.contains(&'h');
    // Numeric → legacy disk volume root listing (before pwd resolution).
    if let Some(t) = pos.first().and_then(|s| s.parse::<usize>().ok()) {
        disk_ls_volume(t);
        return;
    }
    // No path → the current directory, like `ls` (was the store root).
    let path = super::resolve_path(pos.first().map(|s| s.as_str()).unwrap_or("."));

    // A non-root disk mount (e.g. /mnt) lists the volume via the VFS.
    // `/` is the Synapse store tree (never dump percent-encoded on-disk keys).
    if path != "/" {
        if crate::fs::mount::by_path(&path).is_some() {
            ls_mount(&path);
            return;
        }
    }

    // Store hierarchical listing, then VFS fallback for mount files.
    use crate::synapse::fs as store;
    use crate::synapse::vpath::{self, EntryClass};

    match store::classify(&path) {
        None => {
            // Store miss: try foreign-mount VFS (USB /mnt/…), then file read.
            match crate::fs::vfs::readdir(&path) {
                Ok(entries) if path_on_mount(&path) || crate::fs::mount::resolve(&path).is_some() => {
                    serial_println!("ls> {}  ({} entries)", path, entries.len());
                    for e in entries.into_iter().take(64) {
                        if e.is_dir {
                            serial_println!("  {}/", e.name);
                        } else if e.size > 0 {
                            serial_println!("  {} ({} bytes)", e.name, e.size);
                        } else {
                            serial_println!("  {}", e.name);
                        }
                    }
                }
                _ => {
                    if crate::fs::vfs::read_mount(&path).is_ok() {
                        serial_println!("ls> {}: is a file (use /cat)", path);
                    } else {
                        serial_println!("ls> {}: no such file or directory", path);
                        crate::shell::status::fail1();
                    }
                }
            }
        }
        Some(EntryClass::File) => {
            let sz = store::size_of(&path).unwrap_or(0);
            serial_println!("ls> {}  ({} bytes)", path, sz);
        }
        Some(EntryClass::Dir) => {
            let entries = store::list_dir(&path);
            serial_println!("ls> {}  ({} entries)", path, entries.len());
            if entries.is_empty() {
                return;
            }
            for e in &entries {
                if long {
                    let full = if path == "/" {
                        alloc::format!("/{}", e.name)
                    } else {
                        alloc::format!("{}/{}", path, e.name)
                    };
                    let (mode, uid, mtime) = store::meta(&full)
                        .map(|m| (m.mode, m.uid, m.mtime))
                        .unwrap_or((0, 0, 0));
                    serial_println!(
                        "  {}",
                        vpath::format_long_meta_sized(e, mode, uid, mtime, human)
                    );
                } else {
                    serial_println!("  {}", vpath::format_short(e));
                }
            }
        }
    }
}

/// `/cat <path>` — print a store file (preferred) or a mounted-volume file.
pub(super) fn fs_cat(arg: &str) {
    let full = super::resolve_path(arg.trim());
    if full.is_empty() || arg.trim().is_empty() {
        serial_println!("cat> usage: /cat <path>");
        crate::shell::status::fail1();
        return;
    }
    if crate::synapse::fs::is_dir(&full) {
        serial_println!("cat> {}: is a directory", full);
        crate::shell::status::fail1();
        return;
    }
    // Store + mounts through the VFS facade.
    let data = crate::fs::vfs::read(&full);
    match data {
        Ok(bytes) => {
            serial_println!("cat> {} ({} bytes):", full, bytes.len());
            match core::str::from_utf8(&bytes) {
                Ok(s) => match crate::highlight::lang_for_path(&full) {
                    Some(lang) => {
                        let mut st = crate::highlight::State::default();
                        for line in s.lines() {
                            serial_println!("{}", crate::highlight::ansi_line(lang, line, &mut st));
                        }
                    }
                    None => serial_println!("{}", s),
                },
                Err(_) => serial_println!("(binary; {} bytes)", bytes.len()),
            }
        }
        Err(_) => {
            serial_println!("cat> {} not found (store or mounts; see /ls, /mounts)", full);
            crate::shell::status::fail1();
        }
    }
}


/// `/head [-n N|-c N] <path>` and `/tail [-n N|-c N] <path>` — the first or
/// last part of a file, over the same VFS facade `/cat` uses (store, mounted
/// volumes, and a `/host` shared folder alike).
fn fs_head_tail(arg: &str, from_end: bool) {
    let cmd = if from_end { "tail" } else { "head" };
    let (spec, raw) = match crate::shell::headtail::parse(arg, from_end) {
        Ok(v) => v,
        Err(e) => {
            serial_println!("{cmd}> {e}");
            serial_println!("{cmd}> usage: /{cmd} [-n <lines>|-c <bytes>] [path]");
            crate::shell::status::fail1();
            return;
        }
    };
    // No path (or `-`) means piped input, exactly as `head` reads stdin. With
    // no pipeline feeding this stage there is nothing to read, which is the
    // "no path given" case and belongs here rather than in the pure parser —
    // only the caller knows whether a pipeline is running.
    let (full, bytes) = if crate::shell::headtail::is_stdin(raw) {
        match crate::shell::pipeline::take_piped() {
            Some(text) => (alloc::string::String::from("(piped)"), text.into_bytes()),
            None => {
                serial_println!("{cmd}> no path given");
                serial_println!("{cmd}> usage: /{cmd} [-n <lines>|-c <bytes>] [path]");
                crate::shell::status::fail1();
                return;
            }
        }
    } else {
        let full = super::resolve_path(raw.unwrap_or(""));
        if crate::synapse::fs::is_dir(&full) {
            serial_println!("{cmd}> {full}: is a directory");
            crate::shell::status::fail1();
            return;
        }
        match crate::fs::vfs::read(&full) {
            Ok(b) => (full, b),
            Err(_) => {
                serial_println!("{cmd}> {full} not found (store or mounts; see /ls, /mounts)");
                crate::shell::status::fail1();
                return;
            }
        }
    };
    let part = crate::shell::headtail::select(&bytes, &spec);
    if spec.bytes {
        serial_println!("{cmd}> {full} ({} of {} bytes):", part.len(), bytes.len());
    } else {
        let total = crate::shell::headtail::count_lines(&bytes);
        let shown = crate::shell::headtail::count_lines(part);
        serial_println!("{cmd}> {full} ({shown} of {total} line(s)):");
    }
    match core::str::from_utf8(part) {
        Ok(s) => {
            // `head` starts at byte 0, so a highlighter's fence/string state is
            // correct. `tail` starts in the middle, where that state is unknown
            // — a view beginning inside a code fence would be coloured as if it
            // were not — so it prints plain rather than confidently wrong.
            match crate::highlight::lang_for_path(&full).filter(|_| !from_end) {
                Some(lang) => {
                    let mut st = crate::highlight::State::default();
                    for line in s.lines() {
                        serial_println!("{}", crate::highlight::ansi_line(lang, line, &mut st));
                    }
                }
                None => serial_println!("{}", s.trim_end_matches('\n')),
            }
        }
        // Byte mode can legitimately cut a multi-byte character in half, so an
        // invalid slice here is not evidence the file is binary.
        Err(_) => serial_println!("({} bytes; not valid UTF-8 at this cut)", part.len()),
    }
}

pub(super) fn fs_head(arg: &str) {
    fs_head_tail(arg, false);
}

pub(super) fn fs_tail(arg: &str) {
    fs_head_tail(arg, true);
}

/// `/pbcopy <path>` — put a file's contents on the clipboard, which syncs to
/// the host (OSC-52 over the serial terminal, and the SPICE agent when a
/// clipboard channel is attached — see `/clip` for which route is live).
///
/// Named after the macOS tool, but it takes a **path**: there is no stdin to
/// read from here. `/clip <text>` is the literal-text form.
pub(super) fn fs_pbcopy(arg: &str) {
    let raw = arg.trim();
    // Piped input, or `-` naming it explicitly.
    if raw.is_empty() || raw == "-" {
        if let Some(text) = crate::shell::pipeline::take_piped() {
            let n = text.len();
            crate::clipboard::set(text, false);
            serial_println!("pbcopy> copied {n} byte(s) from the pipe");
            return;
        }
    }
    if raw.is_empty() {
        serial_println!("pbcopy> usage: /pbcopy <path>   (for literal text, /clip <text>)");
        crate::shell::status::fail1();
        return;
    }
    let full = super::resolve_path(raw);
    if crate::synapse::fs::is_dir(&full) {
        serial_println!("pbcopy> {full}: is a directory");
        crate::shell::status::fail1();
        return;
    }
    let bytes = match crate::fs::vfs::read(&full) {
        Ok(b) => b,
        Err(_) => {
            serial_println!("pbcopy> {full} not found (store or mounts; see /ls, /mounts)");
            crate::shell::status::fail1();
            return;
        }
    };
    // Bounded on purpose. The clipboard is held as one String, and the OSC-52
    // route base64-expands it 4/3 and pushes the result down the serial line —
    // a multi-megabyte file would take minutes and look like a hang.
    const MAX: usize = crate::clipboard::vdagent::MAX_CLIPBOARD;
    if bytes.len() > MAX {
        serial_println!(
            "pbcopy> {full} is {} bytes; the clipboard holds at most {MAX} \
(use /head -c {MAX} {full} to see the start)",
            bytes.len()
        );
        return;
    }
    // The clipboard is text on every route it feeds, so a binary file is
    // refused rather than silently lossily converted — a mangled paste is
    // worse than a refusal that names the reason.
    let Ok(text) = core::str::from_utf8(&bytes) else {
        serial_println!("pbcopy> {full} is not UTF-8 text; the clipboard carries text only");
        crate::shell::status::fail1();
        return;
    };
    let n = text.len();
    crate::clipboard::set(alloc::string::String::from(text), false);
    serial_println!(
        "pbcopy> copied {n} byte(s) from {full}; {}",
        if crate::clipboard::agent_present() {
            "announced to the host agent + OSC-52 to the serial terminal"
        } else {
            "OSC-52 to the serial terminal"
        }
    );
}

/// `/grep <query> [path_glob]` — content search over the store.
pub(super) fn fs_grep(arg: &str) {
    let mut parts = arg.split_whitespace();
    let Some(query) = parts.next() else {
        serial_println!("grep> usage: /grep <query> [path_glob]");
        crate::shell::status::fail1();
        return;
    };
    let path_glob = parts.next().unwrap_or("");
    // No path and a pipeline feeding us: search what was piped, as `grep`
    // searches stdin. With a path given, the pipe is not consumed and the
    // runner reports that.
    if path_glob.is_empty() {
        if let Some(text) = crate::shell::pipeline::take_piped() {
            let files = alloc::vec![(alloc::string::String::from("(piped)"), text)];
            let hits = crate::tools::pathutil::grep_files(query, &files, 200);
            if hits.is_empty() {
                serial_println!("grep> no matches for {:?}", query);
                crate::shell::status::fail1();
                return;
            }
            serial_println!("grep> {} hit(s) for {:?}:", hits.len(), query);
            for h in hits {
                serial_println!("{}", h.text);
            }
            return;
        }
    }
    let mut paths = crate::synapse::fs::list();
    if !path_glob.is_empty() {
        let resolved = super::resolve_path(path_glob);
        paths = crate::tools::pathutil::glob_filter(&resolved, &paths);
    }
    let mut files: alloc::vec::Vec<(alloc::string::String, alloc::string::String)> = alloc::vec::Vec::new();
    for p in paths {
        if let Some(bytes) = crate::synapse::fs::read(&p) {
            files.push((p, alloc::string::String::from_utf8_lossy(&bytes).into_owned()));
        }
    }
    let hits = crate::tools::pathutil::grep_files(query, &files, 50);
    if hits.is_empty() {
        serial_println!("grep> no matches for {:?}", query);
        // `grep` reports "no match" as a failure, which is what makes
        // `/grep x f && /echo found` mean anything.
        crate::shell::status::fail1();
        return;
    }
    serial_println!("grep> {} hit(s) for {:?}:", hits.len(), query);
    for h in hits {
        serial_println!("  {}:{}:{}", h.path, h.line, h.text);
    }
}

/// `/glob <pattern>` — path glob over the store.
pub(super) fn fs_glob(arg: &str) {
    let raw = arg.trim();
    if raw.is_empty() {
        serial_println!("glob> usage: /glob <pattern>   e.g. /glob **/*.md");
        crate::shell::status::fail1();
        return;
    }
    let pattern = super::resolve_path(raw);
    let paths = crate::synapse::fs::list();
    let hits = crate::tools::pathutil::glob_filter(&pattern, &paths);
    serial_println!("glob> {} match(es) for {:?}:", hits.len(), pattern);
    for p in hits {
        serial_println!("  {}", p);
    }
}

/// True when `path` is on a **foreign** mount (`/mnt`, USB, second disk), so
/// shell FS ops should use raw VFS instead of the synapse store.
///
/// The auto-mounted data volume at `/` is **not** treated as a foreign mount:
/// it is the durable synapse store itself. Routing `/mkdir /test` through VFS
/// wrote an ext4 dir on disk that `/ls` (store view) never saw — the 01_30
/// VirtualBox screenshot. Those paths stay on `synapse::fs` so they list and
/// persist via Ext4Store.
pub(super) fn path_on_mount(path: &str) -> bool {
    let Some((mt, _rel)) = crate::fs::mount::resolve(path) else {
        return false;
    };
    mt.path != "/"
}

pub(super) fn vfs_err_msg(op: &str, path: &str, e: crate::fs::vfs::VfsError) {
    use crate::fs::vfs::VfsError;
    match e {
        VfsError::ReadOnly => serial_println!("{op}> {path}: read-only mount (remount with rw?)"),
        VfsError::Unsupported => {
            serial_println!(
                "{op}> {path}: unsupported on this volume (FAT/ext RW; NTFS is read-only)"
            )
        }
        VfsError::NotFound => serial_println!("{op}> {path}: not found"),
        VfsError::NotAFile => serial_println!("{op}> {path}: not a file"),
        VfsError::NotADir => serial_println!("{op}> {path}: not a directory"),
        VfsError::NotMounted => serial_println!("{op}> {path}: not mounted (see /mounts)"),
        VfsError::Io => serial_println!("{op}> {path}: I/O error"),
    }
}

/// `/mkdir [-p] <path>` — create a directory (store or writable mount).
pub(super) fn fs_mkdir(arg: &str) {
    let (flags, pos) = fs_split_flags(arg);
    let parents = flags.contains(&'p');
    let Some(raw) = pos.first() else {
        serial_println!("mkdir> usage: /mkdir [-p] <path>");
        crate::shell::status::fail1();
        return;
    };
    let path = super::resolve_path(raw);
    if path_on_mount(&path) {
        let norm = crate::fs::path::normalize(&path);
        if parents {
            let mut cur = alloc::string::String::new();
            for part in norm.split('/').filter(|s| !s.is_empty()) {
                cur.push('/');
                cur.push_str(part);
                if crate::fs::mount::by_path(&cur).is_some() {
                    continue; // mount root
                }
                // Best-effort: skip if already present as a file or dir.
                if crate::fs::vfs::readdir(&cur).is_ok() {
                    continue;
                }
                if let Err(e) = crate::fs::vfs::mkdir(&cur) {
                    // Exists as a file (NotAFile mapping) or already there.
                    if matches!(
                        e,
                        crate::fs::vfs::VfsError::NotAFile | crate::fs::vfs::VfsError::Io
                    ) {
                        // FatRw Exists → NotAFile; treat "already exists" as ok for -p.
                        continue;
                    }
                    vfs_err_msg("mkdir", &cur, e);
                    return;
                }
            }
            serial_println!("mkdir> {norm}");
            return;
        }
        match crate::fs::vfs::mkdir(&path) {
            Ok(()) => serial_println!("mkdir> {norm}"),
            Err(e) => vfs_err_msg("mkdir", &path, e),
        }
        return;
    }
    match crate::synapse::fs::mkdir(&path, parents) {
        Ok(()) => serial_println!("mkdir> {}", crate::synapse::vpath::normalize(&path)),
        Err(e) => serial_println!("mkdir> {}: {}", path, e),
    }
}

/// `/cp [-r] <src> <dst>` — copy file (store and/or mounted volumes).
pub(super) fn fs_cp(arg: &str) {
    let (flags, pos) = fs_split_flags(arg);
    let recursive = flags.contains(&'r') || flags.contains(&'R');
    if pos.len() < 2 {
        serial_println!("cp> usage: /cp [-r] <src> <dst>");
        crate::shell::status::fail1();
        return;
    }
    let src = super::resolve_path(&pos[0]);
    let dst = super::resolve_path(&pos[1]);
    // Volume path: single-file copy via VFS (recursive trees stay store-only).
    if path_on_mount(&src) || path_on_mount(&dst) {
        if recursive {
            serial_println!("cp> recursive copy across mounts is not supported yet");
            crate::shell::status::fail1();
            return;
        }
        match crate::fs::vfs::read(&src) {
            Ok(data) => match crate::fs::vfs::write(&dst, &data) {
                Ok(()) => serial_println!("cp> {} → {} ({} byte(s))", src, dst, data.len()),
                Err(e) => vfs_err_msg("cp", &dst, e),
            },
            Err(e) => vfs_err_msg("cp", &src, e),
        }
        return;
    }
    match crate::synapse::fs::copy(&src, &dst, recursive) {
        Ok(n) => serial_println!("cp> {} → {} ({} file(s))", src, dst, n),
        Err(e) => serial_println!("cp> {}: {}", src, e),
    }
}

/// `/mv <src> <dst>` — rename/move in the store or on a writable ext mount.
pub(super) fn fs_mv(arg: &str) {
    let (_flags, pos) = fs_split_flags(arg);
    if pos.len() < 2 {
        serial_println!("mv> usage: /mv <src> <dst>");
        crate::shell::status::fail1();
        return;
    }
    let src = super::resolve_path(&pos[0]);
    let dst = super::resolve_path(&pos[1]);
    if path_on_mount(&src) || path_on_mount(&dst) {
        match crate::fs::vfs::rename(&src, &dst) {
            Ok(()) => serial_println!("mv> {} → {}", src, dst),
            Err(e) => vfs_err_msg("mv", &src, e),
        }
        return;
    }
    match crate::synapse::fs::rename(&src, &dst) {
        Ok(n) => serial_println!("mv> {} → {} ({} file(s))", src, dst, n),
        Err(e) => serial_println!("mv> {}: {}", src, e),
    }
}

/// `/rm [-r] <path>` — remove a store file/tree or a file on a writable mount.
pub(super) fn fs_rm(arg: &str) {
    let (flags, pos) = fs_split_flags(arg);
    let recursive = flags.contains(&'r') || flags.contains(&'R');
    let Some(raw) = pos.first() else {
        serial_println!("rm> usage: /rm [-r] <path>");
        crate::shell::status::fail1();
        return;
    };
    let path = super::resolve_path(raw);
    if path_on_mount(&path) {
        if recursive {
            serial_println!("rm> recursive remove on mounts is not supported yet");
            crate::shell::status::fail1();
            return;
        }
        match crate::fs::vfs::unlink(&path) {
            Ok(()) => serial_println!("rm> {}", path),
            Err(e) => vfs_err_msg("rm", &path, e),
        }
        return;
    }
    match crate::synapse::fs::remove(&path, recursive) {
        Ok(n) => serial_println!("rm> {} ({} file(s))", path, n),
        Err(e) => serial_println!("rm> {}: {}", path, e),
    }
}

/// `/touch <path>` — create empty file or refresh existing (store or mount).
pub(super) fn fs_touch(arg: &str) {
    let path = super::resolve_path(arg.trim());
    if path.is_empty() || arg.trim().is_empty() {
        serial_println!("touch> usage: /touch <path>");
        crate::shell::status::fail1();
        return;
    }
    if path_on_mount(&path) {
        // Create empty or leave existing contents (read + rewrite).
        let data = crate::fs::vfs::read(&path).unwrap_or_default();
        match crate::fs::vfs::write(&path, &data) {
            Ok(()) => serial_println!("touch> {}", crate::fs::path::normalize(&path)),
            Err(e) => vfs_err_msg("touch", &path, e),
        }
        return;
    }
    match crate::synapse::fs::touch(&path) {
        Ok(()) => serial_println!("touch> {}", crate::synapse::vpath::normalize(&path)),
        Err(e) => serial_println!("touch> {}: {}", path, e),
    }
}

/// `/channel` — manage external messaging channels (Telegram first; generic
/// backends). OpenClaw-style: add a bot, start polling, pair/allow senders,
/// send/reply. Inbound text with `auto_agent` is answered by the shell agent.
pub(super) fn run_channel(arg: &str) {
    use crate::msgchan::{self, DmPolicy, Kind};
    let mut parts = arg.split_whitespace();
    let sub = parts.next().unwrap_or("");
    match sub {
        "" | "list" | "ls" => {
            let all = msgchan::list();
            if all.is_empty() {
                serial_println!("channel> (none) — /channel add telegram <name> <bot_token>");
                serial_println!("channel> types: {}", msgchan::types().join(", "));
                return;
            }
            serial_println!("channel> {} instance(s):", all.len());
            for i in all {
                let st = if i.running { "running" } else { "stopped" };
                let err = i
                    .last_error
                    .as_deref()
                    .map(|e| alloc::format!(" err={e}"))
                    .unwrap_or_default();
                serial_println!(
                    "  {:<12} {:<10} {:<8} policy={} allow={} auto_agent={}{err}",
                    i.name,
                    i.kind.as_str(),
                    st,
                    i.policy.as_str(),
                    i.allow_from.len(),
                    i.auto_agent
                );
            }
        }
        "types" => {
            serial_println!("channel> backends: {}", msgchan::types().join(", "));
            serial_println!("channel> add more kinds in msgchan::Kind without changing this command");
        }
        "add" => {
            // /channel add telegram <name> <token> [pairing|allowlist|open]
            let kind_s = parts.next().unwrap_or("");
            let name = parts.next().unwrap_or("");
            let token = parts.next().unwrap_or("");
            let pol_s = parts.next().unwrap_or("pairing");
            let Some(kind) = Kind::parse(kind_s) else {
                serial_println!("channel> usage: /channel add <type> <name> <token> [pairing|allowlist|open]");
                serial_println!("channel> types: {}", msgchan::types().join(", "));
                return;
            };
            let policy = DmPolicy::parse(pol_s).unwrap_or(DmPolicy::Pairing);
            match msgchan::add(name, kind, token, policy) {
                Ok(()) => serial_println!(
                    "channel> added '{}' ({}, policy={}) — /channel start {name}",
                    name,
                    kind.as_str(),
                    policy.as_str()
                ),
                Err(e) => serial_println!("channel> add failed: {e}"),
            }
        }
        "remove" | "rm" => {
            let name = parts.next().unwrap_or("");
            if name.is_empty() {
                serial_println!("channel> usage: /channel remove <name>");
                return;
            }
            match msgchan::remove(name) {
                Ok(()) => serial_println!("channel> removed '{name}'"),
                Err(e) => serial_println!("channel> {e}"),
            }
        }
        "start" => {
            let name = parts.next().unwrap_or("");
            if name.is_empty() {
                serial_println!("channel> usage: /channel start <name>");
                return;
            }
            serial_println!("channel> starting '{name}' (HTTPS to api.telegram.org; Ctrl+C cancels)…");
            match msgchan::start(name) {
                Ok(()) => {
                    serial_println!(
                        "channel> '{name}' started — polling every ~2.5s in the background"
                    );
                    serial_println!(
                        "channel> DM the bot, then /channel pair {name} <CODE> (or /channel status)"
                    );
                }
                Err(e) => serial_println!("channel> start failed: {e}"),
            }
        }
        "stop" => {
            let name = parts.next().unwrap_or("");
            if name.is_empty() {
                serial_println!("channel> usage: /channel stop <name>");
                return;
            }
            match msgchan::stop(name) {
                Ok(()) => serial_println!("channel> '{name}' stopped"),
                Err(e) => serial_println!("channel> {e}"),
            }
        }
        "status" => {
            let name = parts.next();
            let mut any = false;
            for i in msgchan::list() {
                if name.is_some_and(|n| n != i.name) {
                    continue;
                }
                any = true;
                serial_println!(
                    "channel> {}  kind={}  {}  policy={}  offset={}  allow_from={:?}",
                    i.name,
                    i.kind.as_str(),
                    if i.running { "running" } else { "stopped" },
                    i.policy.as_str(),
                    i.offset,
                    i.allow_from
                );
                if let Some(p) = &i.last_peer {
                    serial_println!("  last_peer={p}");
                }
                if let Some((code, uid, disp)) = &i.pending_pair {
                    serial_println!(
                        "  pending_pair: code={code}  from={disp} ({uid})  →  /channel pair {} {code}",
                        i.name
                    );
                }
                if let Some(e) = &i.last_error {
                    serial_println!("  last_error={e}");
                }
            }
            if !any {
                serial_println!("channel> (no matching instance)");
            }
            let q = msgchan::inbound_len();
            if q > 0 {
                serial_println!("channel> {} inbound message(s) queued for the agent", q);
            }
            if name.is_none() || any {
                serial_println!(
                    "channel> polls every ~2.5s while the prompt is idle; /channel poll [name] forces one now"
                );
            }
        }
        "allow" => {
            let name = parts.next().unwrap_or("");
            let uid = parts.next().unwrap_or("");
            if name.is_empty() || uid.is_empty() {
                serial_println!("channel> usage: /channel allow <name> <user_id|*>");
                return;
            }
            match msgchan::allow(name, uid) {
                Ok(()) => {
                    serial_println!("channel> '{name}' allows {uid}");
                    // Catch up on DMs that arrived before allow (offset may
                    // still be 0 — Telegram buffers recent updates).
                    serial_println!("channel> fetching pending updates…");
                    msgchan::poll_now(Some(name));
                }
                Err(e) => serial_println!("channel> {e}"),
            }
        }
        "pair" => {
            // /channel pair <name> <code>  — CODE is the 4 hex digits the bot
            // sends (e.g. AB12), NOT your Telegram user id.
            let name = parts.next().unwrap_or("");
            let code = parts.next().unwrap_or("");
            if name.is_empty() || code.is_empty() {
                serial_println!("channel> usage: /channel pair <name> <CODE>");
                serial_println!(
                    "channel> CODE = 4 hex digits from the bot DM (e.g. AB12), not your user id"
                );
                serial_println!(
                    "channel> if there is no code yet: DM the bot, wait a few seconds, /channel status"
                );
                return;
            }
            match msgchan::pair_approve(name, code) {
                Ok(uid) => serial_println!("channel> paired {uid} on '{name}'"),
                Err(e) => {
                    serial_println!("channel> pair failed: {e}");
                    if e == "no pending pair" {
                        serial_println!(
                            "channel> tip: use /channel allow {name} <user_id> if you already know your Telegram id"
                        );
                        serial_println!(
                            "channel> pairing only appears after a DM is *received* (polling must be running)"
                        );
                    }
                }
            }
        }
        "poll" => {
            // Force an immediate getUpdates round (debug / catch-up).
            let name = parts.next();
            serial_println!("channel> polling…");
            msgchan::poll_now(name);
            serial_println!("channel> poll done — /channel status");
        }
        "send" => {
            // /channel send <name> <peer> <text…>
            let name = parts.next().unwrap_or("");
            let peer = parts.next().unwrap_or("");
            let text: alloc::string::String = parts.collect::<alloc::vec::Vec<_>>().join(" ");
            if name.is_empty() || peer.is_empty() || text.is_empty() {
                serial_println!("channel> usage: /channel send <name> <peer_id> <text>");
                return;
            }
            match msgchan::send(name, peer, &text) {
                Ok(()) => serial_println!("channel> sent to {peer} via '{name}'"),
                Err(e) => serial_println!("channel> send failed: {e}"),
            }
        }
        "reply" => {
            let name = parts.next().unwrap_or("");
            let text: alloc::string::String = parts.collect::<alloc::vec::Vec<_>>().join(" ");
            if name.is_empty() || text.is_empty() {
                serial_println!("channel> usage: /channel reply <name> <text>");
                return;
            }
            match msgchan::reply(name, &text) {
                Ok(()) => serial_println!("channel> replied on '{name}'"),
                Err(e) => serial_println!("channel> reply failed: {e}"),
            }
        }
        "help" | _ => {
            serial_println!("channel> messaging channels (generic; Telegram first):");
            serial_println!("  /channel [list]                     list instances");
            serial_println!("  /channel types                      available backends");
            serial_println!("  /channel add telegram <name> <tok>  [pairing|allowlist|open]");
            serial_println!("  /channel start|stop|remove <name>");
            serial_println!("  /channel status [name]");
            serial_println!("  /channel allow <name> <user_id|*>");
            serial_println!("  /channel pair <name> <CODE>         approve a DM pairing");
            serial_println!("  /channel send <name> <peer> <text>");
            serial_println!("  /channel reply <name> <text>        reply to last inbound");
            serial_println!("  /channel poll [name]                force getUpdates now");
            serial_println!("  config: {}", msgchan::CONFIG_PATH);
        }
    }
}

/// Strip light markdown so Telegram gets plain text (the model often emits
/// `**bold**` despite the system prompt).
pub(super) fn strip_md_light(s: &str) -> alloc::string::String {
    let mut out = alloc::string::String::with_capacity(s.len());
    let b = s.as_bytes();
    let mut i = 0;
    while i < b.len() {
        // **bold** or *italic* or `code`
        if b[i] == b'*' || b[i] == b'`' || b[i] == b'_' {
            // skip run of the same marker
            let m = b[i];
            while i < b.len() && b[i] == m {
                i += 1;
            }
            continue;
        }
        out.push(b[i] as char);
        i += 1;
    }
    out
}

/// Drain inbound messaging-channel queue: each message becomes a shell-agent
/// turn; the reply is sent back on the same channel. Called from the interactive
/// loop (not from upkeep — inference is too heavy for the poll tick).
pub(super) fn drain_channel_inbound(
    chat: &mut Option<ChatSession>,
    session: &mut crate::agent::types::Session,
) {
    // Process a bounded number per loop so the prompt stays responsive.
    for _ in 0..3 {
        let Some(msg) = crate::msgchan::take_inbound() else {
            break;
        };
        serial_println!(
            "channel[{}] → agent: {} says: {}",
            msg.channel,
            msg.from_name,
            msg.text
        );
        // Ensure a chat session exists.
        if chat.is_none() {
            let mut spin = Spinner::new("channel");
            *chat = ChatSession::load(&mut spin);
            if let Some(c) = chat.as_mut() {
                c.hydrate_from_session(session);
            }
        }
        let Some(sess) = chat.as_mut() else {
            let _ = crate::msgchan::send(
                &msg.channel,
                &msg.peer_id,
                "Chitti: no local model loaded — cannot auto-reply. Use /channel reply from the console, or /model load.",
            );
            continue;
        };
        // Frame the turn so a small model stays on *this* message (not the
        // previous one) and uses tools for OS facts instead of inventing them.
        let user = alloc::format!(
            "Message from Telegram user {} (channel {}).\n\
             Answer ONLY the latest user message below. Do not continue an earlier topic.\n\
             If the question needs machine state (disks, files, network, time), call the right tool first; never invent those facts.\n\
             For simple math or greetings, answer directly in one short plain-text reply (no markdown).\n\
             \n\
             User message:\n{}",
            msg.from_name, msg.channel, msg.text
        );
        let reply = sess.turn(&user, session);
        let reply = strip_md_light(reply.trim());
        let reply = reply.trim();
        if reply.is_empty() {
            let _ = crate::msgchan::send(
                &msg.channel,
                &msg.peer_id,
                "(no reply — try again or check /think /model on the console)",
            );
            continue;
        }
        serial_println!("channel[{}] ← agent: {}", msg.channel, reply);
        if let Err(e) = crate::msgchan::send(&msg.channel, &msg.peer_id, reply) {
            serial_println!("channel> delivery failed: {e}");
        }
    }
}

/// Load large fallback fonts (CJK) from any disk volume and register them into
/// the system font fallback chain — OS-wide, so the console/UI and the browser
/// all render CJK. Runs at most once. Kept off the kernel binary because of the
/// font's size (~16 MB); placed on the fonts/voice disk by `cargo xtask`
/// (fetch with `cargo xtask font-assets`). A graceful no-op when absent (the
/// Indic + emoji faces are always bundled in the binary). Safe to call at boot
/// now that the block probe is idempotent.
pub(super) fn ensure_disk_fallback_fonts() {
    use core::sync::atomic::{AtomicBool, Ordering};
    static TRIED: AtomicBool = AtomicBool::new(false);
    if TRIED.swap(true, Ordering::Relaxed) {
        return;
    }
    const DISK_FALLBACKS: &[(&str, &[&str])] = &[(
        "Noto Sans CJK",
        &["NotoSansCJKsc-Regular.otf", "NotoSansCJK.otf", "cjk.otf"],
    )];
    for (name, files) in DISK_FALLBACKS {
        if crate::font_ttf::fallback_loaded(name) {
            continue;
        }
        if let Some(bytes) = find_on_disks(files) {
            // Parsing a font through fontdue churns the first-fit allocator in
            // proportion to its glyph count; a full ~16 MB CJK CFF face stalls
            // the (cooperative) kernel for minutes and freezes the shell. Cap
            // the size so an oversized font is skipped rather than hanging —
            // use a **subset** CJK face (a few thousand common glyphs, ≤ a few
            // MB) if you want CJK coverage.
            const MAX_FALLBACK_BYTES: usize = 6 * 1024 * 1024;
            if bytes.len() > MAX_FALLBACK_BYTES {
                serial_println!(
                    "font: {} is {} MiB — too large to parse in-kernel, skipped (use a subset ≤ {} MiB)",
                    name,
                    bytes.len() / (1024 * 1024),
                    MAX_FALLBACK_BYTES / (1024 * 1024)
                );
                continue;
            }
            serial_println!("font: loading {} ({} KiB)\u{2026}", name, bytes.len() / 1024);
            match crate::font_ttf::register_fallback(name, &bytes) {
                Ok(()) => crate::ktrace::log_fmt(format_args!(
                    "font: registered {} fallback ({} bytes, disk)",
                    name,
                    bytes.len()
                )),
                Err(e) => serial_println!("font: {} load failed: {}", name, e),
            }
        }
    }
}

/// Scan every disk + volume for the first readable file named one of
/// `names` (FAT or ext4; root or subpath like `brcm/foo.bin`). Independent of
/// `/mount` — see [`crate::fs::vfs::find_on_disks`].
pub(crate) fn find_on_disks(names: &[&str]) -> Option<alloc::vec::Vec<u8>> {
    crate::fs::vfs::find_on_disks(names)
}

/// Read a file at an absolute path under some active mount (or the store).
/// Shared by `/voice models load` and media open helpers.
pub(super) fn read_mounted(full: &str) -> Option<alloc::vec::Vec<u8>> {
    crate::fs::vfs::read(full).ok()
}

pub(super) fn disk_list() {
    use crate::block::BlockDevice;
    // Enumerate every block device, not just the boot disk: a machine can have
    // several (e.g. two NVMe namespaces on one controller — VirtualBox presents
    // each attached disk that way). `probe_disk_nth` walks them until absent.
    let mut found = 0usize;
    let mut d = 0usize;
    while let Some(mut dev) = crate::block::probe_disk_nth(d) {
        found += 1;
        let sectors = dev.block_count();
        serial_println!("disks> disk {}: {} sectors ({} MiB)", d, sectors, sectors * 512 / 1024 / 1024);
        let vols = crate::fs::detect::probe(&mut dev);
        if vols.is_empty() {
            serial_println!("  (no recognizable volumes -- blank or unsupported layout)");
        }
        for (i, v) in vols.iter().enumerate() {
            serial_println!(
                "  [{}] lba {:<10} {:>6} MiB  {:<8} label={}",
                i,
                v.start_lba,
                v.sectors * 512 / 1024 / 1024,
                v.fs.name(),
                v.label.as_deref().unwrap_or("-")
            );
        }
        d += 1;
        if d >= 16 {
            break; // safety bound
        }
    }
    if found == 0 {
        serial_println!(
            "disks> no block device found\n\
             disks>   expected: NVMe (VirtualBox default), AHCI/SATA, or virtio-blk\n\
             disks>   after /install: reboot from the permanent disk alone — data is\n\
             disks>   on the 'Chitti Data' GPT partition and mounts at boot"
        );
        return;
    }
    serial_println!(
        "  ({} disk(s); /mount <disk> [vol] [/path]; data partition auto-mounts as synapse store)",
        found
    );
}

/// List volume `n` on disk 0 (on-disk root; for debugging real FAT/ext4 layouts).
pub(super) fn disk_ls_volume(n: usize) {
    use crate::fs::detect::FsType;
    let Some(mut dev) = crate::block::probe_disk() else {
        serial_println!("ls> no block device");
        return;
    };
    let vols = crate::fs::detect::probe(&mut dev);
    let Some(v) = vols.get(n).cloned() else {
        serial_println!("ls> no volume {} (see /disks)", n);
        return;
    };
    match v.fs {
        FsType::Fat16 | FsType::Fat32 => {
            let mut part = crate::block::Partition::new(&mut dev, v.start_lba, v.sectors);
            match crate::block::fat_read::FatReader::open(&mut part) {
                Some(mut r) => {
                    let entries = r.list_root();
                    serial_println!("ls> {} volume {} root ({} entries):", v.fs.name(), n, entries.len());
                    for (name, size, is_dir) in entries {
                        if is_dir {
                            serial_println!("  {}/", name);
                        } else {
                            serial_println!("  {} ({} bytes)", name, size);
                        }
                    }
                }
                None => serial_println!("ls> FAT volume unreadable"),
            }
        }
        FsType::Ext2 | FsType::Ext3 | FsType::Ext4 => {
            let mut part = crate::block::Partition::new(&mut dev, v.start_lba, v.sectors);
            match crate::block::ext4_read::Ext4Reader::open(&mut part) {
                Some(mut r) => {
                    let entries = r.list_root();
                    // Data-partition keys are percent-encoded flat names; show
                    // a hierarchical store view when this volume is the live store.
                    serial_println!(
                        "ls> {} volume {} root ({} on-disk entries; use /ls / for the store tree):",
                        v.fs.name(),
                        n,
                        entries.len()
                    );
                    for (name, ino, is_dir) in entries.into_iter().take(32) {
                        let shown = crate::block::ext4_store::key_decode(&name);
                        let base = crate::synapse::vpath::basename(&shown);
                        serial_println!(
                            "  {}{}  (inode {})",
                            base,
                            if is_dir { "/" } else { "" },
                            ino
                        );
                    }
                }
                None => serial_println!("ls> ext volume unreadable"),
            }
        }
        other => serial_println!(
            "ls> volume {} is {} -- directory listing not implemented",
            n,
            other.name()
        ),
    }
}
