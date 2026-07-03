//! Real boot entry point. `cargo xtask run -arch x86_64` boots the x86 kernel
//! via Limine; `cargo xtask run -arch aarch64` boots the aarch64 kernel
//! natively via QEMU + HVF (`-M virt -kernel`, entered at `aarch64_start`).
#![no_std]
#![no_main]

use chitti_kernel::serial_println;
use core::panic::PanicInfo;

const BOOT_MSG: &str = "Chitti: boot ok";

/// Shared OS steady state (arch-independent): the Synapse + Persona demos,
/// then the interactive intent shell (never returns).
fn run_os() -> ! {
    chitti_kernel::synapse::demo();
    chitti_kernel::shell::demo();
    chitti_kernel::agent::demo();
    chitti_kernel::shell::run();
}

// --- x86_64 boot (Limine) -----------------------------------------------

#[cfg(target_arch = "x86_64")]
#[unsafe(no_mangle)]
pub extern "C" fn _start() -> ! {
    use chitti_kernel::{arch, framebuffer, limine_protocol, FRAMEBUFFER_REQUEST, MEMMAP_REQUEST};

    // Must be first: SIMD codegen is on crate-wide, so the optimizer may emit
    // SSE (XMM) instructions below; they fault until SSE is enabled.
    arch::x86_64::fpu::enable_sse();
    chitti_kernel::serial::init();

    // Framebuffer console first, so serial output is mirrored to the window.
    if let Some(fb) = FRAMEBUFFER_REQUEST.response().and_then(|r| r.framebuffers().first().copied()) {
        framebuffer::init_console(fb);
    }
    serial_println!("{}", BOOT_MSG);

    if let Some(mm) = MEMMAP_REQUEST.response() {
        let usable: u64 = mm
            .entries()
            .iter()
            .filter(|e| e.entry_type == limine_protocol::MEMMAP_USABLE)
            .map(|e| e.length)
            .sum();
        serial_println!("Chitti: {} usable memory-map bytes across {} entries", usable, mm.entries().len());
    } else {
        serial_println!("Chitti: memory map request was refused");
    }

    chitti_kernel::init();
    serial_println!(
        "Chitti: SMP: {} core(s) online (see ktrace 'smp:' lines for the spinlock self-test)",
        chitti_kernel::smp::cpu_count()
    );
    disk_demo();
    // Bring up a USB keyboard (xHCI + HID) if present, so a real USB keyboard
    // drives the shell alongside PS/2. No-op without an xHCI controller.
    chitti_kernel::arch::x86_64::xhci::init_global();

    match chitti_kernel::cortex::model_module() {
        Some(bytes) => {
            if let Ok(g) = chitti_kernel::cortex::gguf::Gguf::parse(bytes) {
                serial_println!(
                    "Chitti: model.gguf loaded ({} MiB): {} layers, dim {}, {} heads/{} kv, ffn {}, vocab {}",
                    bytes.len() / (1024 * 1024),
                    g.config.block_count,
                    g.config.embedding_length,
                    g.config.head_count,
                    g.config.head_count_kv,
                    g.config.feed_forward_length,
                    g.tokens.len(),
                );
            }
        }
        None => serial_println!("Chitti: no model.gguf boot module present"),
    }

    // Make agent state durable: if an ext4 *data* partition is present (an
    // installed system), point synapse::fs at it so runtime writes persist
    // across reboots. No-op on the live ISO (no data partition).
    mount_persistent_store();

    // `ref-check` builds run the Phase 3 acceptance gate and exit QEMU.
    #[cfg(feature = "refcheck")]
    {
        let ok = chitti_kernel::cortex::run_acceptance();
        chitti_kernel::qemu::exit_qemu(if ok {
            chitti_kernel::qemu::QemuExitCode::Success
        } else {
            chitti_kernel::qemu::QemuExitCode::Failed
        });
    }
    #[cfg(not(feature = "refcheck"))]
    run_os();

    #[allow(unreachable_code)]
    loop {
        arch::hlt();
    }
}

// --- aarch64 boot (QEMU virt + HVF; entered from arch::aarch64::boot) ----

#[cfg(target_arch = "aarch64")]
#[unsafe(no_mangle)]
pub extern "C" fn aarch64_start() -> ! {
    // The boot stub (arch::aarch64::boot) already set the stack, enabled NEON,
    // and zeroed BSS. Enable the MMU *first* -- with it off, RAM is Device
    // memory where the LL/SC exclusives that back `Locked`/atomics never
    // succeed (a spinlock would spin forever). `serial_println!` mirrors to a
    // `Locked` framebuffer console, so even the banner needs normal memory.
    chitti_kernel::arch::aarch64::mmu::init();
    chitti_kernel::serial::init();
    serial_println!("{} -- NATIVE aarch64 on Apple Silicon (QEMU + HVF)", BOOT_MSG);
    chitti_kernel::init();
    // Bring up the ramfb framebuffer TUI — the aarch64 equivalent of the x86
    // Limine framebuffer. Needs the heap (from `init()`), so it comes after.
    // Absent if QEMU was launched without `-device ramfb`; then we stay serial.
    if let Some((addr, w, h, pitch)) = unsafe { chitti_kernel::arch::aarch64::ramfb::init() } {
        chitti_kernel::framebuffer::init_console_raw(addr, w, h, pitch);
        serial_println!("Chitti: framebuffer TUI up ({}x{} ramfb) -- console mirrored to the window", w, h);
        // With a window, wire the virtio-keyboard so it accepts keystrokes too
        // (the `virt` machine has no PS/2). Absent if launched without one.
        if chitti_kernel::arch::aarch64::virtio_input::init() {
            serial_println!("Chitti: virtio-keyboard up -- type in the window or the serial terminal");
        }
    }
    run_os();
}

/// Point `synapse::fs` at an ext4 *data* partition so agent writes are durable
/// across reboots (the installed system). Chooses an ext4 volume that does NOT
/// hold the model (`*.gguf`), so it never adopts the model/OS partition. No-op
/// on the live ISO, or if there's no writable ext4 data volume.
#[cfg(target_arch = "x86_64")]
fn mount_persistent_store() {
    use chitti_kernel::block::ext4_read::Ext4Reader;
    use chitti_kernel::block::ext4_store::Ext4Store;
    use chitti_kernel::block::virtio::VirtioBlk;
    use chitti_kernel::block::Partition;
    use chitti_kernel::fs::detect::FsType;

    let Some(mut dev) = VirtioBlk::probe() else { return };
    let vols = chitti_kernel::fs::detect::probe(&mut dev);
    let mut chosen: Option<(u64, u64)> = None;
    for v in vols {
        if !matches!(v.fs, FsType::Ext2 | FsType::Ext3 | FsType::Ext4) {
            continue;
        }
        let mut part = Partition::new(&mut dev, v.start_lba, v.sectors);
        if let Some(mut r) = Ext4Reader::open(&mut part) {
            // The data partition is the ext4 that holds neither the model
            // (*.gguf) nor the boot/OS files (kernel, limine.conf) — so agent
            // state never lands on the model/OS partition even when the model
            // is absent (e.g. a --no-model install).
            let is_os_or_model = r.list_root().iter().any(|(n, _, _)| n.contains(".gguf") || n == "chitti-kernel" || n == "limine.conf");
            if !is_os_or_model {
                chosen = Some((v.start_lba, v.sectors));
                break;
            }
        }
    }
    let Some((start, count)) = chosen else {
        serial_println!("Chitti: synapse persistence -> none (no ext4 data partition; state is in-memory only)");
        return;
    };
    if let Some(store) = Ext4Store::mount(dev, start, count) {
        chitti_kernel::synapse::fs::mount_ext4(store);
        serial_println!("Chitti: synapse persistence -> ext4 data partition at lba {} ({} sectors); writes are durable", start, count);
        // Prove the round-trip: a boot counter written *through synapse::fs*.
        // It only increments if the previous boot's write was recovered from
        // ext4 on mount — i.e. runtime writes truly persisted across the reboot.
        let prior = chitti_kernel::synapse::fs::read("synapse_boots")
            .and_then(|b| core::str::from_utf8(&b).ok().and_then(|s| s.trim().parse::<u32>().ok()))
            .unwrap_or(0);
        let boots = prior + 1;
        let mut buf = [0u8; 12];
        chitti_kernel::synapse::fs::write("synapse_boots", fmt_u32(boots, &mut buf).as_bytes());
        serial_println!("Chitti: synapse.fs boot #{} (persisted via ext4); files = {:?}", boots, chitti_kernel::synapse::fs::list());
        if prior > 0 {
            serial_println!("Chitti: synapse.fs (the counter survived a reboot -- agent writes persist on ext4)");
        }
    }
}

/// Phase 7 block-device FS demo (x86 only -- the virtio-blk driver uses PCI
/// port I/O): mount SimpleFS on the disk and bump a persistent boot counter.
#[cfg(target_arch = "x86_64")]
fn disk_demo() {
    use chitti_kernel::block::virtio::VirtioBlk;
    use chitti_kernel::block::BlockDevice;
    use chitti_kernel::fs::SimpleFs;

    serial_println!("Chitti: --- Block-device filesystem (Phase 7) ---");
    let Some(dev) = VirtioBlk::probe() else {
        serial_println!("Chitti: disk> no virtio-blk device present (boot with a -drive to enable persistence)");
        return;
    };
    serial_println!("Chitti: disk> virtio-blk found: {} sectors", dev.block_count());

    // Mount ONLY -- never auto-format. A blank or foreign disk is left
    // untouched and reported; formatting is an explicit user action (`/install`
    // or `/mkfs`), like a real OS installer.
    let mut fs = match SimpleFs::mount(dev) {
        Ok(fs) => fs,
        Err(_) => {
            serial_println!("Chitti: disk> present but not a Chitti (SimpleFS) volume -- NOT auto-formatting.");
            serial_println!("Chitti: disk> use /install (bootable install) or /mkfs to set it up.");
            return;
        }
    };

    let prior = fs
        .read("boots")
        .ok()
        .and_then(|b| core::str::from_utf8(&b).ok().and_then(|s| s.trim().parse::<u32>().ok()))
        .unwrap_or(0);
    let boots = prior + 1;
    let mut buf = [0u8; 12];
    let text = fmt_u32(boots, &mut buf);
    match fs.write("boots", text.as_bytes()).and_then(|_| fs.write("banner", b"written by Chitti OS SimpleFS")) {
        Ok(()) => {
            let files = fs.list().unwrap_or_default();
            serial_println!("Chitti: disk> boot #{} (persisted on disk); files = {:?}", boots, files);
            if prior > 0 {
                serial_println!("Chitti: disk> (the counter survived a reboot -- durable storage works)");
            } else {
                serial_println!("Chitti: disk> (first boot on this Chitti disk; run again to see the counter increment)");
            }
        }
        Err(e) => serial_println!("Chitti: disk> write failed: {:?}", e),
    }
}

/// Format a `u32` into `buf` without `alloc`, returning the decimal string.
#[cfg(target_arch = "x86_64")]
fn fmt_u32(mut n: u32, buf: &mut [u8; 12]) -> &str {
    if n == 0 {
        buf[0] = b'0';
        return core::str::from_utf8(&buf[..1]).unwrap();
    }
    let mut i = buf.len();
    while n > 0 {
        i -= 1;
        buf[i] = b'0' + (n % 10) as u8;
        n /= 10;
    }
    buf.copy_within(i.., 0);
    core::str::from_utf8(&buf[..buf.len() - i]).unwrap()
}

#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    serial_println!("KERNEL PANIC: {}", info);
    loop {
        chitti_kernel::arch::hlt();
    }
}
