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
    serial_println!("{}", BOOT_MSG);

    // Capture the Limine framebuffer descriptor NOW: the response lives in
    // bootloader-reclaimable memory, so it must be copied before `init()`
    // reclaims it. The (heap-allocating) `init_console` is deferred until after
    // `init()` brings the heap up.
    let fb_info = FRAMEBUFFER_REQUEST.response().and_then(|r| r.framebuffers().first().copied());

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
    // Framebuffer console AFTER the heap is up (Screen::build allocates its
    // pane titles / status strings), using the descriptor captured before
    // `init()` reclaimed the bootloader response memory.
    if let Some(fb) = fb_info {
        framebuffer::init_console(fb);
    }
    serial_println!(
        "Chitti: SMP: {} core(s) online (see ktrace 'smp:' lines for the spinlock self-test)",
        chitti_kernel::smp::cpu_count()
    );
    disk_demo();
    // Bring up a USB keyboard (xHCI + HID) if present, so a real USB keyboard
    // drives the shell alongside PS/2. No-op without an xHCI controller.
    let usb_kbd = chitti_kernel::arch::x86_64::xhci::init_global();
    // INPUT summary (parity with aarch64): PS/2 (i8042) is always present on a
    // PC; USB is READY only if a HID keyboard enumerated on the xHCI.
    serial_println!(
        "Chitti: INPUT  ps2={}  usb-kbd={}  (serial always works)",
        "yes",
        if usb_kbd { "READY" } else { "no" }
    );

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

// --- aarch64 boot via the Limine protocol (UEFI/AAVMF, boots from disk) ---
// Enabled by the `boot-limine` feature. Limine loads the kernel higher-half
// with the MMU on (identity-mapping the first 4 GiB incl. device MMIO, plus an
// HHDM) and calls `limine_start`. We take the heap from the Limine memmap and
// otherwise reuse the exact same steady state as the `-kernel` path.
#[cfg(all(target_arch = "aarch64", feature = "boot-limine"))]
#[unsafe(no_mangle)]
pub extern "C" fn limine_start() -> ! {
    use chitti_kernel::{limine_protocol, FRAMEBUFFER_REQUEST, HHDM_REQUEST, MEMMAP_REQUEST};
    // Enable FP/SIMD (NEON) at EL1 — the tensor kernels need it.
    unsafe {
        core::arch::asm!("mrs x9, cpacr_el1", "orr x9, x9, #(3 << 20)", "msr cpacr_el1, x9", "isb", out("x9") _, options(nostack));
    }
    chitti_kernel::serial::init();
    serial_println!("{} -- NATIVE aarch64 via Limine (UEFI/AAVMF), booted from disk", BOOT_MSG);
    // Limine maps usable RAM through the HHDM (higher-half direct map), not a
    // low identity map, so the heap must be addressed at `phys + hhdm_offset`.
    // (Device MMIO — PL011, virtio-mmio — is separately low-mapped, which is
    // why serial already works.)
    let hhdm = HHDM_REQUEST.response().expect("Limine HHDM request refused").offset;
    chitti_kernel::arch::aarch64::set_hhdm(hhdm);
    let need = chitti_kernel::mm::heap::HEAP_SIZE as u64;
    let mut phys = 0u64;
    let mm = MEMMAP_REQUEST.response().expect("Limine memmap request refused");
    for e in mm.entries() {
        if e.entry_type == limine_protocol::MEMMAP_USABLE && e.length >= need {
            phys = (e.base + 0xfff) & !0xfff;
            break;
        }
    }
    assert!(phys != 0, "Limine memmap: no usable region of {} bytes for the heap", need);
    let heap_va = (phys + hhdm) as usize;
    chitti_kernel::mm::heap::init_static(heap_va, chitti_kernel::mm::heap::HEAP_SIZE);
    serial_println!("Chitti: heap {} MiB at phys {:#x} (hhdm va {:#x})", chitti_kernel::mm::heap::HEAP_SIZE / (1024 * 1024), phys, heap_va);

    chitti_kernel::sched::init();
    if let Some(fb) = FRAMEBUFFER_REQUEST.response().and_then(|r| r.framebuffers().first().copied()) {
        chitti_kernel::framebuffer::init_console(fb);
        serial_println!("Chitti: framebuffer up via Limine GOP -- console mirrored to the window");
    }
    disk_demo();
    match chitti_kernel::cortex::model_module() {
        Some(bytes) => {
            if let Ok(g) = chitti_kernel::cortex::gguf::Gguf::parse(bytes) {
                serial_println!(
                    "Chitti: model.gguf loaded ({} MiB): {} layers, dim {}, vocab {}",
                    bytes.len() / (1024 * 1024),
                    g.config.block_count,
                    g.config.embedding_length,
                    g.tokens.len(),
                );
            }
        }
        None => serial_println!("Chitti: no model.gguf (Limine module or ext4)"),
    }
    mount_persistent_store();
    run_os();
}

// --- aarch64 boot (QEMU virt + HVF; entered from arch::aarch64::boot) ----

#[cfg(all(target_arch = "aarch64", not(feature = "boot-limine")))]
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
    // Bring up the framebuffer TUI. Preferred source: the **boot-info page** the
    // UEFI stub publishes at 0x47F00000 (magic "CHITTIBI") carrying the GOP
    // framebuffer — works on ANY UEFI platform (VirtualBox-ARM, UTM, real
    // hardware). Fallback: the QEMU-only ramfb device (`-kernel` path). Either
    // needs the heap (from `init()`), so this comes after.
    // Bring up the PCIe bus from ACPI (ECAM base via MCFG), so the real virtio-
    // pci transport is available before probing disks. No-op on the `-kernel`
    // path (no boot-info RSDP) — there the virtio-mmio fallback is used.
    aarch64_pcie_init();

    // Preferred: the UEFI GOP framebuffer (with its real pixel format) from the
    // stub's boot-info page — works on real hardware / VirtualBox / UTM at the
    // monitor's native resolution. Fallback: QEMU ramfb (always XRGB8888).
    // Seed the wall clock from the UEFI stub's captured time (the reliable clock
    // on VirtualBox-ARM and other UEFI platforms). Before the shell's clock::init,
    // which then keeps this seed instead of probing the (absent) PL031.
    if let Some(secs) = bootinfo_unix() {
        chitti_kernel::clock::set_unix(secs as i64);
        serial_println!("Chitti: wall clock seeded from UEFI ({} unix)", secs);
    }

    let fb = if let Some((addr, w, h, pitch, bpp, rs, gs, bs)) = bootinfo_framebuffer() {
        chitti_kernel::framebuffer::init_console_raw_fmt(addr, w, h, pitch, bpp, rs, gs, bs);
        Some((w, h))
    } else if let Some((addr, w, h, pitch)) = unsafe { chitti_kernel::arch::aarch64::ramfb::init() } {
        chitti_kernel::framebuffer::init_console_raw(addr, w, h, pitch);
        Some((w, h))
    } else {
        None
    };
    if let Some((w, h)) = fb {
        serial_println!("Chitti: framebuffer TUI up ({}x{}) -- console mirrored to the window", w, h);
        // Bring up a USB keyboard (xHCI + HID) if present — the real-hardware
        // input path; needs the PCIe bus from aarch64_pcie_init.
        let usb_kbd = chitti_kernel::arch::aarch64::xhci::init_global();
        // A PL050 PS/2 keyboard (ARM dev boards / some hypervisors) — the ARM
        // analogue of the x86 i8042. No-op where absent (e.g. QEMU `virt`).
        let pl050 = chitti_kernel::arch::aarch64::pl050::init();
        // A PL050 PS/2 mouse (a second KMI) — the ARM PS/2 pointing device, as
        // VirtualBox-ARM presents (hidpointing=ps2mouse). No-op where absent.
        let _pl050_mouse = chitti_kernel::arch::aarch64::pl050_mouse::init();
        // Also wire the virtio-keyboard (QEMU `virt` window). Absent without one.
        let virtio_kbd = chitti_kernel::arch::aarch64::virtio_input::init();
        // A virtio pointer (tablet/mouse) for the window — the aarch64 mouse.
        let _mouse = chitti_kernel::arch::aarch64::virtio_pointer::init();
        // A single, non-scrolling INPUT summary right before the shell so the
        // discovered input path is visible on the framebuffer (the only console
        // that survives a platform whose serial/UART we don't reach). This is the
        // ground truth for "why isn't my keyboard working".
        let ecam = chitti_kernel::pci::ecam_base();
        serial_println!(
            "Chitti: INPUT  pcie-ecam={:#x}  usb-kbd={}  pl050-kbd={}  virtio-kbd={}  mouse[virtio={} ps2={}]  (serial always works)",
            ecam,
            if usb_kbd { "READY" } else { "no" },
            if pl050 { "yes" } else { "no" },
            if virtio_kbd { "yes" } else { "no" },
            if _mouse { "yes" } else { "no" },
            if _pl050_mouse { "yes" } else { "no" }
        );
    }
    // Same storage bring-up as x86: mount SimpleFS demo disk (if any) and point
    // synapse::fs at an ext4 data partition for durable agent state. No-op
    // without a `-drive`/virtio-blk-device.
    disk_demo();
    mount_persistent_store();
    // Everything is up (framebuffer, USB/input, disk, persistent store) with IRQs
    // masked. NOW begin timer-preemptive scheduling: unmask IRQs so the generic
    // timer preempts the shell. Deferred to here (not inside init()) so device
    // bring-up — the framebuffer especially — runs identically to the cooperative
    // path; a no-op where the GIC/timer wasn't available (HVF → cooperative).
    chitti_kernel::arch::aarch64::gic::start_preemption();
    run_os();
}

/// The stub's boot-info page (address passed in x1, magic "CHITTIBI"), if
/// present and within the identity map. `None` on the `-kernel` path.
#[cfg(all(target_arch = "aarch64", not(feature = "boot-limine")))]
fn bootinfo_page() -> Option<u64> {
    let map_limit = chitti_kernel::arch::aarch64::mmu::mapped_bytes();
    let bi = unsafe { core::ptr::read_volatile(core::ptr::addr_of!(chitti_kernel::arch::aarch64::boot::BOOT_X1)) };
    if bi == 0 || bi >= map_limit {
        return None;
    }
    // SAFETY: identity-mapped RAM below the map limit.
    let magic = unsafe { core::slice::from_raw_parts(bi as *const u8, 8) };
    (magic == b"CHITTIBI").then_some(bi)
}

/// Discover + bring up PCIe from the stub's ACPI RSDP (boot-info offset 40):
/// walk MCFG for the ECAM base, map it Device, init `pci`. No-op without a
/// boot-info page (the `-kernel` path uses the virtio-mmio fallback).
#[cfg(all(target_arch = "aarch64", not(feature = "boot-limine")))]
fn aarch64_pcie_init() {
    let Some(bi) = bootinfo_page() else { return };
    // SAFETY: identity-mapped; RSDP pointer is at offset 40.
    let rsdp = unsafe { core::ptr::read_volatile((bi + 40) as *const u64) };
    let Some(seg) = chitti_kernel::acpi::ecam_from_rsdp(rsdp) else {
        serial_println!("Chitti: no ACPI MCFG (RSDP {:#x}) -- PCIe unavailable, using virtio-mmio", rsdp);
        return;
    };
    chitti_kernel::arch::aarch64::mmu::map_device_gib(seg.base);
    chitti_kernel::pci::init(seg.base, seg.bus_end);
    serial_println!("Chitti: PCIe ECAM {:#x} (buses {}..{}) mapped -- virtio-pci enabled", seg.base, seg.bus_start, seg.bus_end);
}

#[cfg(all(target_arch = "aarch64", feature = "boot-limine"))]
fn aarch64_pcie_init() {}

/// Read the UEFI stub's boot-info page (magic "CHITTIBI"): the GOP framebuffer
/// `(addr, width, height, pitch, bpp_bytes, r_shift, g_shift, b_shift)` captured
/// before ExitBootServices. `None` on the `-kernel` path (no stub) or if the
/// framebuffer lies outside the identity map. The geometry is little-endian u64s
/// after the 8-byte magic; the pixel format is 4 bytes at offset 48
/// (r_shift, g_shift, b_shift, bytes-per-pixel).
#[cfg(all(target_arch = "aarch64", not(feature = "boot-limine")))]
fn bootinfo_framebuffer() -> Option<(usize, u64, u64, u64, u64, u32, u32, u32)> {
    // Identity-map coverage (the extent arch::aarch64::mmu actually mapped).
    let map_limit = chitti_kernel::arch::aarch64::mmu::mapped_bytes();
    let bi = bootinfo_page()?;
    // SAFETY: `bi` is identity-mapped RAM below the map limit; read 52 bytes.
    let page = unsafe { core::slice::from_raw_parts(bi as *const u8, 52) };
    let f = |o: usize| u64::from_le_bytes(page[o..o + 8].try_into().unwrap());
    let (addr, w, h, pitch) = (f(8), f(16), f(24), f(32));
    if addr == 0 || addr + h * pitch > map_limit {
        serial_println!("Chitti: boot-info framebuffer at {:#x} outside the identity map -- skipping", addr);
        return None;
    }
    // Pixel format at 48..52. An older stub that left these zero (all shifts 0,
    // bpp 0) is treated as the common XRGB8888 default.
    let (mut rs, mut gs, mut bs, mut bpp) = (page[48] as u32, page[49] as u32, page[50] as u32, page[51] as u64);
    if bpp == 0 {
        (rs, gs, bs, bpp) = (16, 8, 0, 4);
    }
    serial_println!("Chitti: framebuffer from UEFI boot-info (GOP {}x{} at {:#x}, shifts {}/{}/{})", w, h, addr, rs, gs, bs);
    Some((addr as usize, w, h, pitch, bpp, rs, gs, bs))
}

/// The UTC Unix time the UEFI stub captured (boot-info offset 52..60), or `None`
/// if absent/zero. This is the reliable wall clock on VirtualBox-ARM (whose
/// generic timer doesn't advance) and any UEFI platform.
#[cfg(all(target_arch = "aarch64", not(feature = "boot-limine")))]
fn bootinfo_unix() -> Option<u64> {
    let bi = bootinfo_page()?;
    // SAFETY: identity-mapped boot-info page; read the 8-byte time field.
    let page = unsafe { core::slice::from_raw_parts(bi as *const u8, 60) };
    let secs = u64::from_le_bytes(page[52..60].try_into().unwrap());
    (secs > 0).then_some(secs)
}

/// Point `synapse::fs` at an ext4 *data* partition so agent writes are durable
/// across reboots (the installed system). Chooses an ext4 volume that does NOT
/// hold the model (`*.gguf`), so it never adopts the model/OS partition. No-op
/// on the live ISO, or if there's no writable ext4 data volume. Arch-generic:
/// the disk is `block::probe_disk()` (virtio-blk over PCI on x86, over
/// virtio-mmio on aarch64).
fn mount_persistent_store() {
    use chitti_kernel::block::ext4_read::Ext4Reader;
    use chitti_kernel::block::ext4_store::Ext4Store;
    use chitti_kernel::block::Partition;
    use chitti_kernel::fs::detect::FsType;

    let Some(mut dev) = chitti_kernel::block::probe_disk() else { return };
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

/// Phase 7 block-device FS demo: mount SimpleFS on the disk and bump a
/// persistent boot counter. Arch-generic via `block::probe_disk()`.
fn disk_demo() {
    use chitti_kernel::block::BlockDevice;
    use chitti_kernel::fs::SimpleFs;

    serial_println!("Chitti: --- Block-device filesystem (Phase 7) ---");
    let Some(dev) = chitti_kernel::block::probe_disk() else {
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
