//! Build orchestration for Chitti OS: assembles a bootable Limine image
//! from the kernel and drives QEMU. All project commands go through
//! `cargo xtask <cmd>` (see CHITTI_OS_HANDOFF.md Part 7).

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let mut args = env::args().skip(1);
    let cmd = args.next().unwrap_or_default();
    let rest: Vec<String> = args.collect();
    let release = rest.iter().any(|a| a == "--release");

    let result = match cmd.as_str() {
        "build" => build_kernel(release).map(|_| ()),
        "image" => image(release),
        "run" => cmd_run(release),
        "test" => cmd_test(),
        // Hidden subcommand: installed as `[target.x86_64-chitti] runner` in
        // kernel/.cargo/config.toml so `cargo test` can boot each compiled
        // test binary in QEMU and translate isa-debug-exit into a real exit
        // code.
        "runner" => cmd_runner(&rest),
        _ => Err(usage()),
    };

    if let Err(e) = result {
        eprintln!("xtask error: {e}");
        std::process::exit(1);
    }
}

fn usage() -> String {
    "usage: cargo xtask <build|image|run|test> [--release]".to_string()
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask is always under <repo>/xtask")
        .to_path_buf()
}

fn kernel_dir() -> PathBuf {
    repo_root().join("kernel")
}

fn run(cmd: &mut Command) -> Result<(), String> {
    let status = cmd
        .status()
        .map_err(|e| format!("failed to spawn {cmd:?}: {e}"))?;
    if !status.success() {
        return Err(format!("command failed ({status}): {cmd:?}"));
    }
    Ok(())
}

/// Build the real (non-test) kernel binary. Returns the path to the ELF.
fn build_kernel(release: bool) -> Result<PathBuf, String> {
    let kdir = kernel_dir();
    let mut cmd = Command::new("cargo");
    cmd.current_dir(&kdir)
        .arg("build")
        .arg("--bin")
        .arg("chitti-kernel");
    if release {
        cmd.arg("--release");
    }
    run(&mut cmd)?;

    let profile = if release { "release" } else { "debug" };
    Ok(kdir
        .join("target/x86_64-chitti")
        .join(profile)
        .join("chitti-kernel"))
}

fn brew_prefix(pkg: &str) -> Result<PathBuf, String> {
    let out = Command::new("brew")
        .args(["--prefix", pkg])
        .output()
        .map_err(|e| format!("failed to run `brew --prefix {pkg}`: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "`brew --prefix {pkg}` failed; install it with `brew install {pkg}`, \
             or point CHITTI_LIMINE_SHARE/CHITTI_LIMINE_BIN at an existing install"
        ));
    }
    Ok(PathBuf::from(
        String::from_utf8_lossy(&out.stdout).trim().to_string(),
    ))
}

fn limine_share_dir() -> Result<PathBuf, String> {
    if let Ok(dir) = env::var("CHITTI_LIMINE_SHARE") {
        return Ok(PathBuf::from(dir));
    }
    Ok(brew_prefix("limine")?.join("share/limine"))
}

fn limine_bin() -> Result<PathBuf, String> {
    if let Ok(bin) = env::var("CHITTI_LIMINE_BIN") {
        return Ok(PathBuf::from(bin));
    }
    Ok(brew_prefix("limine")?.join("bin/limine"))
}

/// Assemble a hybrid BIOS/UEFI ISO around `kernel_bin`, per Limine's
/// documented `xorriso` + `limine bios-install` recipe
/// (USAGE.md#bios-uefi-hybrid-iso-creation).
fn assemble_image(kernel_bin: &Path) -> Result<PathBuf, String> {
    let root = repo_root();
    let iso_root = root.join("target/iso_root");
    if iso_root.exists() {
        fs::remove_dir_all(&iso_root)
            .map_err(|e| format!("removing stale {}: {e}", iso_root.display()))?;
    }
    fs::create_dir_all(iso_root.join("boot/limine")).map_err(|e| e.to_string())?;
    fs::create_dir_all(iso_root.join("EFI/BOOT")).map_err(|e| e.to_string())?;

    fs::copy(kernel_bin, iso_root.join("boot/chitti-kernel"))
        .map_err(|e| format!("copying kernel binary: {e}"))?;
    fs::copy(
        root.join("kernel/limine.conf"),
        iso_root.join("boot/limine/limine.conf"),
    )
    .map_err(|e| format!("copying limine.conf: {e}"))?;

    let share = limine_share_dir()?;
    for f in ["limine-bios.sys", "limine-bios-cd.bin", "limine-uefi-cd.bin"] {
        fs::copy(share.join(f), iso_root.join("boot/limine").join(f))
            .map_err(|e| format!("copying {f} from {}: {e}", share.display()))?;
    }
    for f in ["BOOTX64.EFI", "BOOTIA32.EFI"] {
        fs::copy(share.join(f), iso_root.join("EFI/BOOT").join(f))
            .map_err(|e| format!("copying {f} from {}: {e}", share.display()))?;
    }

    let iso_path = root.join("target/chitti.iso");
    run(Command::new("xorriso")
        .args([
            "-as",
            "mkisofs",
            "-R",
            "-r",
            "-J",
            "-b",
            "boot/limine/limine-bios-cd.bin",
            "-no-emul-boot",
            "-boot-load-size",
            "4",
            "-boot-info-table",
            "-hfsplus",
            "-apm-block-size",
            "2048",
            "--efi-boot",
            "boot/limine/limine-uefi-cd.bin",
            "-efi-boot-part",
            "--efi-boot-image",
            "--protective-msdos-label",
        ])
        .arg(&iso_root)
        .arg("-o")
        .arg(&iso_path))?;

    run(Command::new(limine_bin()?).arg("bios-install").arg(&iso_path))?;

    Ok(iso_path)
}

fn image(release: bool) -> Result<(), String> {
    let bin = build_kernel(release)?;
    let iso = assemble_image(&bin)?;
    println!("image: {}", iso.display());
    Ok(())
}

// `-no-shutdown` deliberately excluded: it makes QEMU pause instead of
// exit on a guest-triggered shutdown, which is exactly what writing to
// isa-debug-exit causes. The test runner needs the process to actually
// exit so it can read the isa-debug-exit status code back.
const QEMU_BASE_ARGS: &[&str] = &[
    "-M",
    "q35",
    "-m",
    "2G",
    "-device",
    "isa-debug-exit,iobase=0xf4,iosize=0x04",
    "-no-reboot",
];

fn qemu_base_cmd(iso: &Path) -> Command {
    let mut cmd = Command::new("qemu-system-x86_64");
    cmd.args(QEMU_BASE_ARGS);
    cmd.arg("-cdrom").arg(iso);
    cmd
}

/// `cargo xtask run`: boot the real kernel interactively (serial to stdio,
/// framebuffer in the QEMU display window).
fn cmd_run(release: bool) -> Result<(), String> {
    let bin = build_kernel(release)?;
    let iso = assemble_image(&bin)?;
    let mut cmd = qemu_base_cmd(&iso);
    cmd.args(["-serial", "stdio"]);
    let status = cmd
        .status()
        .map_err(|e| format!("failed to spawn qemu-system-x86_64: {e}"))?;
    eprintln!("qemu exited: {status}");
    Ok(())
}

/// `cargo xtask test`: run the in-kernel `custom_test_frameworks` test
/// suite via `cargo test --lib`, which cross-compiles each test binary and
/// hands it to the `runner` subcommand below to execute under QEMU.
fn cmd_test() -> Result<(), String> {
    let kdir = kernel_dir();
    let mut cmd = Command::new("cargo");
    cmd.current_dir(&kdir).args(["test", "--lib", "--quiet"]);
    run(&mut cmd)
}

/// Invoked by cargo (via the `runner` config) with the path to a compiled
/// test binary. Boots it in QEMU headlessly and translates the
/// isa-debug-exit exit code `(value << 1) | 1` into this process's exit
/// status, which cargo interprets as the test outcome.
fn cmd_runner(args: &[String]) -> Result<(), String> {
    let bin_path = args
        .first()
        .ok_or_else(|| "runner: missing test binary path argument".to_string())?;
    let iso = assemble_image(Path::new(bin_path))?;
    let mut cmd = qemu_base_cmd(&iso);
    cmd.args(["-serial", "stdio", "-display", "none"]);
    let status = cmd
        .status()
        .map_err(|e| format!("failed to spawn qemu-system-x86_64: {e}"))?;
    match status.code() {
        Some(33) => Ok(()), // QemuExitCode::Success (0x10) -> (0x10 << 1) | 1
        Some(other) => Err(format!(
            "QEMU exited with status {other} (expected 33 = isa-debug-exit success)"
        )),
        None => Err("QEMU was terminated by a signal".to_string()),
    }
}
