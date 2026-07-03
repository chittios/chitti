//! Build orchestration for Chitti OS: assembles a bootable Limine image
//! from the kernel and drives QEMU. All project commands go through
//! `cargo xtask <cmd>` (see CHITTI_OS_HANDOFF.md Part 7).

use std::env;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Target architecture, chosen explicitly via `-arch x86_64|aarch64` (never
/// auto-detected from the host): the same unified kernel builds for both.
#[derive(Clone, Copy, PartialEq)]
enum Arch {
    X86_64,
    Aarch64,
}

/// Which bundled model to build/run for, chosen via `-model <name>` (like
/// `-arch`). Selects the kernel memory-layout feature, the GGUF file, and the
/// aarch64 load address + guest RAM. Default is the compact 0.8B.
#[derive(Clone, Copy, PartialEq)]
enum Model {
    Qwen08B,
    Qwen9B,
}

impl Model {
    /// Cargo features that select this model's memory layout in the kernel.
    fn features(self) -> &'static [&'static str] {
        match self {
            Model::Qwen08B => &[],
            Model::Qwen9B => &["model-9b"],
        }
    }
    /// The GGUF file bundled for this model (relative to the repo root).
    fn gguf_rel(self) -> &'static str {
        match self {
            Model::Qwen08B => "assets/model.gguf",
            Model::Qwen9B => "assets/model-9b.gguf",
        }
    }
    /// aarch64 guest-physical load address (must match `cortex::model_module`).
    fn aarch64_addr(self) -> &'static str {
        match self {
            Model::Qwen08B => "0x48000000",
            Model::Qwen9B => "0x80000000",
        }
    }
    /// QEMU `-m` size big enough for the model + heap in the identity map.
    fn qemu_mem(self) -> &'static str {
        match self {
            Model::Qwen08B => "2G",
            Model::Qwen9B => "10G",
        }
    }
    fn label(self) -> &'static str {
        match self {
            Model::Qwen08B => "qwen3.5-0.8b",
            Model::Qwen9B => "qwen3.5-9b",
        }
    }
}

/// Parse `-model <value>` (or `-model=<value>`); default the 0.8B model.
fn parse_model(rest: &[String]) -> Result<Model, String> {
    let mut it = rest.iter();
    while let Some(a) = it.next() {
        let val = if let Some(v) = a.strip_prefix("-model=") {
            Some(v.to_string())
        } else if a == "-model" || a == "--model" {
            it.next().cloned()
        } else {
            None
        };
        if let Some(v) = val {
            return match v.as_str() {
                "qwen3.5-0.8b" | "qwen3.5-0.8B" | "0.8b" | "0.8B" | "qwen0.8b" | "default" => Ok(Model::Qwen08B),
                "qwen3.5-9b" | "qwen3.5-9B" | "9b" | "9B" | "qwen9b" => Ok(Model::Qwen9B),
                other => Err(format!("unknown -model '{other}' (expected qwen3.5-0.8b or qwen3.5-9b)")),
            };
        }
    }
    Ok(Model::Qwen08B)
}

/// Parse `-arch <value>` (or `-arch=<value>`) from the args; default x86_64.
fn parse_arch(rest: &[String]) -> Result<Arch, String> {
    let mut it = rest.iter();
    while let Some(a) = it.next() {
        let val = if let Some(v) = a.strip_prefix("-arch=") {
            Some(v.to_string())
        } else if a == "-arch" || a == "--arch" {
            it.next().cloned()
        } else {
            None
        };
        if let Some(v) = val {
            return match v.as_str() {
                "x86_64" | "x86-64" | "x64" => Ok(Arch::X86_64),
                "aarch64" | "arm64" => Ok(Arch::Aarch64),
                other => Err(format!("unknown -arch '{other}' (expected x86_64 or aarch64)")),
            };
        }
    }
    Ok(Arch::X86_64)
}

fn main() {
    let mut args = env::args().skip(1);
    let cmd = args.next().unwrap_or_default();
    let rest: Vec<String> = args.collect();
    let release = rest.iter().any(|a| a == "--release");
    let arch = match parse_arch(&rest) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("xtask error: {e}");
            std::process::exit(1);
        }
    };
    let model = match parse_model(&rest) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("xtask error: {e}");
            std::process::exit(1);
        }
    };

    let result = match cmd.as_str() {
        "build" => cmd_build(release, arch, model),
        "image" => image(release, model),
        "run" => cmd_run(release, arch, model),
        "test" => cmd_test(),
        // Phase 3 parity gate: build the kernel with the `refcheck` feature,
        // boot the real model, run the acceptance checks, exit pass/fail.
        "ref-check" => cmd_ref_check(),
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
    "usage: cargo xtask <build|image|run|test|ref-check> [-arch x86_64|aarch64] \
     [-model qwen3.5-0.8b|qwen3.5-9b] [--release]"
        .to_string()
}

/// `cargo xtask build [-arch ...] [-model ...]`: build the unified kernel for
/// the chosen architecture and model memory layout.
fn cmd_build(release: bool, arch: Arch, model: Model) -> Result<(), String> {
    match arch {
        Arch::X86_64 => build_kernel_with(release, model.features()).map(|_| ()),
        Arch::Aarch64 => build_kernel_aarch64(release, model.features()).map(|_| ()),
    }
}

/// Build the unified kernel for aarch64 (`targets/aarch64-chitti.json`) with
/// the given extra cargo `features`, returning the path to the resulting ELF
/// (`-M virt -kernel` bootable).
fn build_kernel_aarch64(release: bool, features: &[&str]) -> Result<PathBuf, String> {
    let kdir = kernel_dir();
    let mut cmd = Command::new("cargo");
    cmd.current_dir(&kdir).args(["build", "--target", "../targets/aarch64-chitti.json"]);
    if release {
        cmd.arg("--release");
    }
    if !features.is_empty() {
        cmd.arg("--features").arg(features.join(","));
    }
    run(&mut cmd)?;
    let profile = if release { "release" } else { "debug" };
    let elf = kdir.join(format!("target/aarch64-chitti/{profile}/chitti-kernel"));
    if !elf.exists() {
        return Err(format!("aarch64 kernel not found at {}", elf.display()));
    }
    Ok(elf)
}

/// `cargo xtask arm64`: build the standalone aarch64 kernel and boot it on
/// `qemu-system-aarch64 -M virt` with `-accel hvf`, so it runs *natively* on
/// Boot the unified kernel built for aarch64 on `qemu-system-aarch64 -M virt`
/// with `-accel hvf`, so it runs *natively* on this Apple Silicon host (no
/// cross-arch emulation) with NEON. Serial to stdio (`-nographic`).
fn cmd_run_aarch64(release: bool, model: Model) -> Result<(), String> {
    // Native inference on aarch64 is only worthwhile optimized: debug NEON is
    // ~30x slower (no inlining of intrinsics, bounds/overflow checks in the hot
    // matvec loop). So this path defaults to a release build regardless of the
    // `--release` flag; the whole point of `-arch aarch64` is native speed.
    if !release {
        eprintln!("note: building aarch64 in RELEASE (debug NEON inference is ~30x slower)");
    }
    let elf = build_kernel_aarch64(true, model.features())?;
    let mut qemu = Command::new("qemu-system-aarch64");
    // Guest RAM holds the kernel + the model (loaded at `model.aarch64_addr`) +
    // the heap; `model.qemu_mem` is sized for the chosen model's layout (2G for
    // 0.8B, 12G for 9B). `-smp 4`: four vCPUs, which under `-accel hvf` run on
    // four *native* M-series cores in parallel (unlike TCG, where extra vCPUs
    // only contend). Chitti's aarch64 SMP brings the secondaries up via PSCI
    // and splits the hot matvec across them.
    // `-device ramfb`: a simple linear framebuffer the kernel configures via
    // fw_cfg (arch::aarch64::ramfb) and renders the TUI into — the aarch64
    // equivalent of the x86 Limine framebuffer. Dropping `-nographic` lets QEMU
    // open its display window; `-serial mon:stdio` keeps the serial console (and
    // QEMU monitor) on stdio so you still type at the terminal (Ctrl-A X quits,
    // Ctrl-A C for the monitor).
    qemu.args([
        "-M", "virt", "-cpu", "host", "-accel", "hvf", "-smp", "4", "-m", model.qemu_mem(),
        "-device", "ramfb", "-device", "virtio-keyboard-device", "-serial", "mon:stdio", "-kernel",
    ]);
    qemu.arg(&elf);
    // Place the GGUF in guest RAM at the model's load address (where the aarch64
    // `cortex::model_module` looks) -- the equivalent of the x86 Limine boot
    // module, so `infer` works natively.
    let gguf = repo_root().join(model.gguf_rel());
    if gguf.exists() {
        let base = u64::from_str_radix(model.aarch64_addr().trim_start_matches("0x"), 16)
            .map_err(|e| format!("bad model addr {}: {e}", model.aarch64_addr()))?;
        for arg in model_loader_args(&gguf, base)? {
            qemu.arg("-device").arg(arg);
        }
        eprintln!("attaching {} at guest phys {}", model.gguf_rel(), model.aarch64_addr());
    } else {
        eprintln!("note: {} absent -- `infer` will report no model", model.gguf_rel());
    }
    eprintln!("booting aarch64 Chitti ({}) natively via HVF (Ctrl-A X to quit qemu)...", model.label());
    run(&mut qemu)
}

/// Build the QEMU `-device loader` argument(s) that place `gguf` in guest RAM
/// starting at `base_addr`. QEMU's generic loader maps each file as a ROM blob
/// and fails for images >= 4 GiB, so a large model is split into <= 1 GiB
/// chunk files loaded at consecutive addresses -- the guest sees one contiguous
/// blob at `base_addr` regardless. Chunks are cached under `target/` and only
/// rewritten when the model changes.
fn model_loader_args(gguf: &Path, base_addr: u64) -> Result<Vec<String>, String> {
    let size = fs::metadata(gguf).map_err(|e| format!("stat {}: {e}", gguf.display()))?.len();
    const CHUNK: u64 = 1 << 30; // 1 GiB, safely under the loader's 4 GiB limit
    if size < 4 << 30 {
        return Ok(vec![format!("loader,file={},addr={:#x},force-raw=on", gguf.display(), base_addr)]);
    }

    let dir = repo_root().join("target/model-chunks");
    let meta = dir.join("source.meta");
    let want = format!("{}:{size}", gguf.display());
    let n_chunks = size.div_ceil(CHUNK);
    let fresh = fs::read_to_string(&meta).map(|s| s.trim() == want).unwrap_or(false)
        && (0..n_chunks).all(|i| dir.join(format!("c{i}")).exists());
    if !fresh {
        eprintln!("splitting {} ({size} bytes) into {n_chunks} chunks for QEMU loader...", gguf.display());
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).map_err(|e| format!("mkdir {}: {e}", dir.display()))?;
        let mut f = fs::File::open(gguf).map_err(|e| format!("open {}: {e}", gguf.display()))?;
        let mut buf = vec![0u8; 8 << 20];
        for i in 0..n_chunks {
            let mut remaining = CHUNK.min(size - i * CHUNK);
            let mut out = fs::File::create(dir.join(format!("c{i}"))).map_err(|e| e.to_string())?;
            while remaining > 0 {
                let want = buf.len().min(remaining as usize);
                let got = f.read(&mut buf[..want]).map_err(|e| e.to_string())?;
                if got == 0 {
                    break;
                }
                out.write_all(&buf[..got]).map_err(|e| e.to_string())?;
                remaining -= got as u64;
            }
        }
        fs::write(&meta, want).map_err(|e| e.to_string())?;
    }
    Ok((0..n_chunks)
        .map(|i| {
            format!("loader,file={},addr={:#x},force-raw=on", dir.join(format!("c{i}")).display(), base_addr + i * CHUNK)
        })
        .collect())
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

/// As `build_kernel`, but with extra cargo features enabled.
fn build_kernel_with(release: bool, features: &[&str]) -> Result<PathBuf, String> {
    let kdir = kernel_dir();
    let mut cmd = Command::new("cargo");
    cmd.current_dir(&kdir)
        .arg("build")
        .arg("--bin")
        .arg("chitti-kernel");
    if release {
        cmd.arg("--release");
    }
    if !features.is_empty() {
        cmd.arg("--features").arg(features.join(","));
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

/// Assemble a hybrid BIOS/UEFI ISO around `kernel_bin` with the default (0.8B)
/// model bundled, per Limine's documented `xorriso` + `limine bios-install`
/// recipe (USAGE.md#bios-uefi-hybrid-iso-creation).
fn assemble_image(kernel_bin: &Path) -> Result<PathBuf, String> {
    assemble_image_opt(kernel_bin, Some("assets/model.gguf"))
}

/// As `assemble_image`, but bundling the model GGUF at `model_rel`.
fn assemble_image_with(kernel_bin: &Path, model_rel: &str) -> Result<PathBuf, String> {
    assemble_image_opt(kernel_bin, Some(model_rel))
}

/// `model_rel`: repo-relative path of the model GGUF to copy into the image, or
/// `None` to bundle no model. The fast test suite (`cargo xtask test`) passes
/// `None` so it never bundles or boots the model it doesn't need;
/// `run`/`image`/`ref-check` pass the selected model.
fn assemble_image_opt(kernel_bin: &Path, model_rel: Option<&str>) -> Result<PathBuf, String> {
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

    // The Phase 3 model is loaded as a Limine boot module, not compiled in.
    // It's optional at image-assembly time: the tensor-kernel unit tests
    // (`cargo xtask test`) don't need it, only end-to-end inference does.
    // `limine.conf` references it unconditionally; Limine tolerates a
    // missing module (the ModuleRequest response simply omits it).
    // Bundle the model + declare it in limine.conf ONLY when both requested
    // and present. Limine panics at boot if `module_path` names a module the
    // image lacks, so the declaration must track the actual file.
    let model = model_rel.map(|r| root.join(r)).filter(|p| p.exists());
    if let Some(model) = model {
        fs::copy(&model, iso_root.join("boot/model.gguf")).map_err(|e| format!("copying model.gguf: {e}"))?;
        let conf_path = iso_root.join("boot/limine/limine.conf");
        let mut conf = fs::read_to_string(&conf_path).map_err(|e| e.to_string())?;
        conf.push_str("    module_path: boot():/boot/model.gguf\n");
        fs::write(&conf_path, conf).map_err(|e| format!("appending module_path: {e}"))?;
    } else if let Some(r) = model_rel {
        eprintln!(
            "xtask: note: {r} not present -- run xtask/fetch-model.sh for inference; \
             the image boots without it (no module declared)."
        );
    }

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

fn image(release: bool, model: Model) -> Result<(), String> {
    let bin = build_kernel_with(release, model.features())?;
    // Bundle the selected model if present; otherwise a kernel-only bootable
    // ISO (what CI ships -- the model is fetched separately, being large).
    let iso = assemble_image_with(&bin, model.gguf_rel())?;
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
    // `-cpu max`: expose AVX2 + XSAVE under TCG so the Cortex kernels can
    // use the AVX2/FMA path (the default `qemu64` lacks them, and the kernel
    // then falls back to SSE2 -- correct, just slower).
    "-cpu",
    "max",
    // NB: SMP (`-smp 4`) is added only by the test runner (`cmd_runner`), where
    // the SMP bring-up + spinlock self-test runs. Interactive `run` and
    // `ref-check` stay single-CPU: inference is BSP-bound, and under
    // single-thread TCG extra vCPUs only cost round-robin overhead.
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

/// `cargo xtask run [-arch ...]`: boot the unified kernel. On aarch64 it runs
/// natively via QEMU + HVF; on x86 it boots the Limine image under
/// qemu-system-x86_64 (TCG on this host).
fn cmd_run(release: bool, arch: Arch, model: Model) -> Result<(), String> {
    if arch == Arch::Aarch64 {
        return cmd_run_aarch64(release, model);
    }
    let bin = build_kernel_with(release, model.features())?;
    let iso = assemble_image_with(&bin, model.gguf_rel())?;
    let disk = ensure_disk_image()?;
    let mut cmd = qemu_base_cmd(&iso);
    cmd.args(["-serial", "stdio"]);
    cmd.arg("-drive").arg(format!("file={},if=none,id=chittidisk,format=raw", disk.display()));
    cmd.args(["-device", "virtio-blk-pci,drive=chittidisk,disable-modern=on"]);
    let status = cmd
        .status()
        .map_err(|e| format!("failed to spawn qemu-system-x86_64: {e}"))?;
    eprintln!("qemu exited: {status}");
    Ok(())
}

/// Create `target/chitti-disk.img` (a 4 MiB raw disk) if it does not exist,
/// so the virtio-blk device has a backing file. Kept across runs so the
/// SimpleFS boot counter persists.
fn ensure_disk_image() -> Result<PathBuf, String> {
    let path = repo_root().join("target/chitti-disk.img");
    if !path.exists() {
        let zeros = vec![0u8; 4 * 1024 * 1024];
        std::fs::write(&path, &zeros).map_err(|e| format!("failed to create disk image: {e}"))?;
        eprintln!("created fresh 4 MiB disk image at {}", path.display());
    }
    Ok(path)
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
    // Fast test suite: exclude the model so the ISO stays small and boots
    // quickly (the tensor-kernel tests validate against baked-in NumPy
    // reference vectors, not the real model).
    let iso = assemble_image_opt(Path::new(bin_path), None)?;
    let mut cmd = qemu_base_cmd(&iso);
    // The in-kernel test suite includes the Phase 7 SMP bring-up + spinlock
    // self-test, so the harness runs with four vCPUs.
    cmd.args(["-smp", "4", "-serial", "stdio", "-display", "none"]);
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

/// `cargo xtask ref-check`: the Phase 3 reference-parity gate. Builds the
/// kernel in release with the `refcheck` feature, boots it with the real
/// model and extra RAM, and lets the in-kernel acceptance routine
/// (`cortex::run_acceptance`) exit QEMU with success (33) or failure. Serial
/// goes to stdio so the `REFCHECK:` lines are visible while it runs.
fn cmd_ref_check() -> Result<(), String> {
    let model = repo_root().join("assets/model.gguf");
    if !model.exists() {
        return Err("assets/model.gguf not present -- run xtask/fetch-model.sh first".to_string());
    }
    let bin = build_kernel_with(true, &["refcheck"])?;
    let iso = assemble_image(&bin)?;
    // CPU inference under QEMU/TCG takes minutes; the model module needs
    // headroom beyond 2 GiB.
    let mut cmd = Command::new("qemu-system-x86_64");
    cmd.args([
        "-M", "q35", "-cpu", "max", "-smp", "4", "-m", "4G", "-device",
        "isa-debug-exit,iobase=0xf4,iosize=0x04", "-no-reboot",
    ]);
    cmd.arg("-cdrom").arg(&iso);
    cmd.args(["-serial", "stdio", "-display", "none"]);
    eprintln!("ref-check: running in-kernel acceptance gate under QEMU (this takes a few minutes)...");
    let status = cmd.status().map_err(|e| format!("failed to spawn qemu-system-x86_64: {e}"))?;
    match status.code() {
        Some(33) => {
            println!("ref-check: PASS (all Phase 3 acceptance checks green)");
            Ok(())
        }
        Some(35) => Err("ref-check: FAIL (QemuExitCode::Failed -- an acceptance check did not match)".to_string()),
        Some(other) => Err(format!("ref-check: QEMU exited with unexpected status {other}")),
        None => Err("ref-check: QEMU was terminated by a signal".to_string()),
    }
}
