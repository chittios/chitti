//! Real boot entry point. `cargo xtask run -arch x86_64` boots the x86 kernel
//! via Limine; `cargo xtask run -arch aarch64` boots the aarch64 kernel
//! natively via QEMU + HVF (`-M virt -kernel`, entered at `aarch64_start`).
#![no_std]
#![no_main]

use chitti_kernel::serial_println;
use core::panic::PanicInfo;

const BOOT_MSG: &str = "Chitti: boot ok";

/// Shared OS steady state (arch-independent): the interactive shell agent
/// (never returns). The Synapse/persona/agent boot demos that used to run here
/// (and spawned demo agents + files on every boot) live on in the test suite;
/// the only default agent is the shell agent.
fn run_os() -> ! {
    // Install the built-in system agents (network, http, ssh) into /agent/ from
    // their bundled markdown + manifest, before the shell comes up.
    chitti_kernel::agent::system::install_all(chitti_kernel::arch::now_ms());
    // Bundled skills (L0 index only until `skill` is invoked).
    chitti_kernel::skills::bundled::install_all();
    // Agents authored on this machine (`/agents new` + `/agents install --path`).
    //
    // They need re-installing every boot for the same reason the system roster does:
    // an install registers a role and its tools in memory, and install records are
    // written to the store but never read back. Without this a local agent's files
    // would survive a reboot while the agent itself quietly would not.
    //
    // Grants come from the **recorded** grant intersected with what the manifest now
    // asks for, never from the manifest alone — the package lives in a writable store,
    // so re-reading its requests as authority would make editing a file an escalation.
    chitti_kernel::agent::local_pkg::reinstall_all(chitti_kernel::arch::now_ms());
    // The `/samples/` corpus, when this image was built with one
    // (CHITTI_SAMPLE_FILES): openable images/video/audio/PDFs on a first boot
    // with no network. Skips files that already exist, so an edited sample on an
    // installed system is never reverted; a no-op when nothing is embedded.
    chitti_kernel::samples::seed();
    // Optional allow/ask/deny tool patterns (creates a default file if missing).
    chitti_kernel::tools::permissions::ensure_default();
    chitti_kernel::tools::permissions::load();
    // External messaging channels (Telegram, …) — load after the store is up.
    chitti_kernel::msgchan::load();
    // The notification queue, so anything unread from the last session survives
    // a reboot: an unread queue that a restart silently emptied would be worse
    // than no queue at all.
    chitti_kernel::notify::load();
    // Scheduled runs. After `notify::load`, because a re-anchor on a moved clock
    // posts into the notification ring.
    chitti_kernel::schedule::load();
    // The pump/idle task, before the shell takes over: it is what lets a waiting
    // task actually sleep instead of pumping `upkeep()` itself. Inert until
    // something blocks (it is reached only via the scheduler's empty-queue
    // fallback), so the boot path is unchanged for every task that stays runnable.
    chitti_kernel::shell::start_pump();
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
        chitti_kernel::mm::set_ram_total(usable);
    } else {
        serial_println!("Chitti: memory map request was refused");
    }

    chitti_kernel::init();
    // Framebuffer console AFTER the heap is up (Screen::build allocates its
    // pane titles / status strings), using the descriptor captured before
    // `init()` reclaimed the bootloader response memory.
    #[cfg(not(feature = "server"))]
    if let Some(fb) = fb_info {
        framebuffer::init_console(fb);
    }
    #[cfg(feature = "server")]
    let _ = fb_info; // server build: serial console only
    serial_println!(
        "Chitti: SMP: {} core(s) online (see ktrace 'smp:' lines for the spinlock self-test)",
        chitti_kernel::smp::cpu_count()
    );
    // Bring up USB HID keyboard + pointer (xHCI) if present, so a real USB
    // keyboard/tablet drives the shell alongside PS/2. No-op without xHCI.
    let _usb = chitti_kernel::arch::x86_64::xhci::init_global();
    // A display driver, if this machine has one we can drive. PCI is already up on
    // x86 by this point; finding nothing keeps Limine's framebuffer.
    chitti_kernel::kms::probe();
    // INPUT summary (parity with aarch64): PS/2 (i8042) is always present on a
    // PC; USB is READY only if a HID device enumerated on the xHCI.
    serial_println!(
        "Chitti: INPUT  ps2={}  usb-kbd={}  usb-mse={}  (serial always works)",
        "yes",
        if chitti_kernel::arch::x86_64::xhci::has_keyboard() { "READY" } else { "no" },
        if chitti_kernel::arch::x86_64::xhci::has_mouse() { "READY" } else { "no" }
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
    // The user's theme and the login gate, the moment the store that holds both
    // is readable — so every line boot prints from here on is already in their
    // palette at their font scale, and nothing is shown before the console is
    // unlocked. See `shell::boot_appearance_and_gate` for why not later.
    chitti_kernel::shell::boot_appearance_and_gate();

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
    // Bring up networking (e1000 / virtio-net-pci over PCI). No-op if absent.
    #[cfg(not(feature = "refcheck"))]
    chitti_kernel::net::autodetect();
    // Bring up audio (virtio-snd) for the /voice pipeline. No-op if absent.
    #[cfg(not(feature = "refcheck"))]
    chitti_kernel::sound::autodetect();
    // Attach a host shared folder (virtio-9p) at /host. No-op if absent, which
    // is the common case — most boots have no folder shared in.
    #[cfg(not(feature = "refcheck"))]
    chitti_kernel::fs::host::attach_at_boot();
    // Host clipboard channel (SPICE agent over virtio-serial). No-op if absent.
    #[cfg(not(feature = "refcheck"))]
    chitti_kernel::clipboard::agent_init();
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
    #[cfg(not(feature = "server"))]
    if let Some(fb) = FRAMEBUFFER_REQUEST.response().and_then(|r| r.framebuffers().first().copied()) {
        chitti_kernel::framebuffer::init_console(fb);
        serial_println!("Chitti: framebuffer up via Limine GOP -- console mirrored to the window");
    }
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
    // Theme + login gate, as early as the store allows (see the x86 path).
    chitti_kernel::shell::boot_appearance_and_gate();
    // Bring up networking (e1000 / virtio-net-pci over PCI). No-op if absent.
    chitti_kernel::net::autodetect();
    // Bring up audio (virtio-snd) for the /voice pipeline. No-op if absent.
    chitti_kernel::sound::autodetect();
    // Attach a host shared folder (virtio-9p) at /host. No-op if absent, which
    // is the common case — most boots have no folder shared in.
    chitti_kernel::fs::host::attach_at_boot();
    // Host clipboard channel (SPICE agent over virtio-serial). No-op if absent.
    chitti_kernel::clipboard::agent_init();
    run_os();
}

// --- aarch64 boot (QEMU virt + HVF; entered from arch::aarch64::boot) ----

#[cfg(all(target_arch = "aarch64", not(feature = "boot-limine")))]
#[unsafe(no_mangle)]
pub extern "C" fn aarch64_start() -> ! {
    // The boot stub (arch::aarch64::boot) already set the stack, enabled FP/SIMD
    // (CPACR_EL1.FPEN + the VHE CPTR_EL2.FPEN Apple needs), and zeroed BSS. Enable
    // the MMU *first* -- with it off, RAM is Device memory where the LL/SC
    // exclusives that back `Locked`/atomics never succeed (a spinlock would spin
    // forever). `serial_println!` mirrors to a `Locked` framebuffer console, so
    // even the banner needs normal memory.
    chitti_kernel::arch::aarch64::mmu::init();
    // The FDT (m1n1's DTB) can sit in m1n1's heap *above* the /memory-reported RAM
    // top that mmu::init mapped — readable with the MMU off (detect() did), but a
    // fault once the MMU is on. Map its GiB(s) as RAM (only if unmapped) before any
    // MMU-on parse (init_uart_apple's has_compatible/reg_of_compatible). No-op on
    // QEMU/hv (FDT already within the mapped range).
    {
        let fdt = chitti_kernel::arch::aarch64::boot::boot_x0();
        chitti_kernel::arch::aarch64::mmu::map_ram_gib_if_unmapped(fdt);
        chitti_kernel::arch::aarch64::mmu::map_ram_gib_if_unmapped(fdt + (1 << 30)); // may straddle a GiB
    }
    // Select the Apple Samsung s5l console from the boot FDT (m1n1) before the
    // first print — Apple Silicon has no PL011, so otherwise the banner would
    // write into an unbacked address and nothing would appear. No-op on QEMU.
    chitti_kernel::arch::aarch64::init_uart_apple();
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
    aarch64_display_edid_init();
    // Bind a display driver if this machine has one. Must follow PCIe discovery.
    // Deliberately **bind only** — the console hand-off happens after the platform
    // framebuffer init below, because on aarch64 that init runs *after* this point:
    // taking the console over here meant KMS brought it up at one size and the
    // firmware/ramfb path then replaced it, leaving the driver's scanout orphaned.
    chitti_kernel::kms::probe_bind_only();

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

    #[cfg(feature = "server")]
    let fb: Option<(u64, u64)> = None; // server build: serial console only
    #[cfg(not(feature = "server"))]
    let fb = if let Some((addr, w, h, pitch, bpp, rs, gs, bs)) = bootinfo_framebuffer() {
        chitti_kernel::framebuffer::init_console_raw_fmt(addr, w, h, pitch, bpp, rs, gs, bs);
        Some((w, h))
    } else if let Some(f) =
        unsafe { chitti_kernel::fdt::find_framebuffer(chitti_kernel::arch::aarch64::boot::boot_x0()) }
    {
        // Apple/m1n1: the `simple-framebuffer` the bootloader set up (base is in
        // RAM, already Normal-mapped by mmu::init). QEMU `-kernel` has no such
        // FDT node, so this is skipped there and ramfb below is used instead.
        chitti_kernel::framebuffer::init_console_raw_fmt(
            f.base as usize,
            f.width as u64,
            f.height as u64,
            f.stride as u64,
            f.bpp as u64,
            f.r_shift as u32,
            f.g_shift as u32,
            f.b_shift as u32,
        );
        Some((f.width as u64, f.height as u64))
    } else if let Some((addr, w, h, pitch)) = unsafe { chitti_kernel::arch::aarch64::ramfb::init() } {
        chitti_kernel::framebuffer::init_console_raw(addr, w, h, pitch);
        Some((w, h))
    } else {
        None
    };
    // Record total physical RAM for the status bar / `/top`: the UEFI stub's
    // memory-map total (VBox / real hardware), else the fw_cfg `ramsize` the
    // launcher published (`-kernel`), else the discovered RAM span.
    {
        let ram = bootinfo_ram_bytes()
            .or_else(|| {
                if chitti_kernel::arch::aarch64::is_apple() {
                    // fw_cfg is absent on Apple (would data-abort); take the RAM
                    // size straight from the FDT `/memory` node instead.
                    unsafe { chitti_kernel::fdt::memory_region(chitti_kernel::arch::aarch64::boot::boot_x0()).map(|(_, s)| s) }
                } else {
                    unsafe { chitti_kernel::arch::aarch64::ramfb::read_ram_bytes() }
                }
            })
            .unwrap_or_else(|| chitti_kernel::arch::aarch64::mmu::ram_end().saturating_sub(0x4000_0000));
        chitti_kernel::mm::set_ram_total(ram);
    }
    // Now that the platform framebuffer (if any) is up, let a bound display driver
    // take the console only if nothing else provided one.
    chitti_kernel::kms::adopt_console_if_needed();
    if let Some((w, h)) = fb {
        serial_println!("Chitti: framebuffer TUI up ({}x{}) -- console mirrored to the window", w, h);
        // Bring up USB HID keyboard + mouse (xHCI). On Apple Silicon the
        // controller is the on-SoC dwc3 (DART + ATC-PHY, via `apple_usb`); on
        // VirtualBox / real SBSA hardware it's an xHCI over PCIe. Both feed the
        // shared xHCI core, so `has_keyboard`/`poll_key` work the same after.
        if chitti_kernel::arch::aarch64::is_apple() {
            chitti_kernel::arch::aarch64::apple_usb::init();
        } else {
            chitti_kernel::arch::aarch64::xhci::init_global();
        }
        let usb_kbd = chitti_kernel::arch::aarch64::xhci::has_keyboard();
        let usb_mse = chitti_kernel::arch::aarch64::xhci::has_mouse();
        // The PL050 (PrimeCell KMI) and virtio-mmio (0x0a00_0000) input probes
        // read fixed QEMU/SBSA addresses that are unbacked on Apple Silicon —
        // where, under m1n1's hv, the read is a fatal data abort rather than
        // harmless garbage. Skip them there; native Apple USB HID is a follow-up.
        let (pl050, _pl050_mouse, virtio_kbd, _mouse) = if chitti_kernel::arch::aarch64::is_apple() {
            (false, false, false, false)
        } else {
            // A PL050 PS/2 keyboard (ARM dev boards / some hypervisors) — the ARM
            // analogue of the x86 i8042. No-op where absent (e.g. QEMU `virt`).
            let pl050 = chitti_kernel::arch::aarch64::pl050::init();
            // A PL050 PS/2 mouse (a second KMI) — as VirtualBox-ARM presents with
            // hidpointing=ps2mouse (we force usbtablet via `make vbox`).
            let pl050_mouse = chitti_kernel::arch::aarch64::pl050_mouse::init();
            // The virtio-keyboard + pointer (QEMU `virt` window).
            let virtio_kbd = chitti_kernel::arch::aarch64::virtio_input::init();
            let mouse = chitti_kernel::arch::aarch64::virtio_pointer::init();
            (pl050, pl050_mouse, virtio_kbd, mouse)
        };
        // A single, non-scrolling INPUT summary right before the shell so the
        // discovered input path is visible on the framebuffer (the only console
        // that survives a platform whose serial/UART we don't reach). This is the
        // ground truth for "why isn't my keyboard/mouse working".
        // On VirtualBox-ARM expect: usb-kbd=READY usb-mse=READY (virtio/ps2 no).
        let ecam = chitti_kernel::pci::ecam_base();
        serial_println!(
            "Chitti: INPUT  pcie-ecam={:#x}  usb-kbd={}  usb-mse={}  pl050-kbd={}  virtio-kbd={}  mouse[virtio={} ps2={}]  (serial always works)",
            ecam,
            if usb_kbd { "READY" } else { "no" },
            if usb_mse { "READY" } else { "no" },
            if pl050 { "yes" } else { "no" },
            if virtio_kbd { "yes" } else { "no" },
            if _mouse { "yes" } else { "no" },
            if _pl050_mouse { "yes" } else { "no" }
        );
    }
    // Same storage bring-up as x86: point synapse::fs at an ext4 data partition
    // for durable agent state. No-op without a `-drive`/virtio-blk-device. The
    // disk probe reads virtio-mmio at 0x0a00_0000 (and PCIe/NVMe/AHCI), all
    // QEMU/SBSA addresses that fault on Apple; Apple's ANS2 storage is a
    // follow-up, so agent state stays in-memory there.
    if !chitti_kernel::arch::aarch64::is_apple() {
        mount_persistent_store();
    }
    // Theme + login gate, as early as the store allows (see the x86 path).
    // Outside the `is_apple` guard on purpose: Apple Silicon has no store yet, so
    // there is nothing to unlock and nothing themed to load — but the call is
    // still correct there (defaults, and a gate that finds no record is inert),
    // and a machine that skipped it would be the one divergence in this file.
    chitti_kernel::shell::boot_appearance_and_gate();
    // `ref-check` builds run the acceptance gate and power off via PSCI (the
    // aarch64 analogue of the x86 isa-debug-exit path above); the host side
    // (`cargo xtask ref-check -arch aarch64`) greps serial for `ALL PASS`.
    #[cfg(feature = "refcheck")]
    {
        let ok = chitti_kernel::cortex::run_acceptance();
        serial_println!("refcheck: {} -- powering off", if ok { "PASS" } else { "FAIL" });
        chitti_kernel::arch::aarch64::psci_system_off();
    }
    // Bring up networking (virtio-net over mmio, else a PCI NIC) so /network,
    // /ping and /wifi work. No-op if no NIC is present. On Apple Silicon the
    // QEMU virtio/PCI probes would fault (unbacked MMIO); instead we bring up
    // APCIE + the Broadcom FullMAC WiFi when `chitti.wifi` is on the bootargs.
    #[cfg(not(feature = "refcheck"))]
    if chitti_kernel::arch::aarch64::is_apple() {
        if chitti_kernel::drivers::wifi::init_apple() {
            serial_println!("Chitti: Wi-Fi radio probed (see /wifi info)");
        }
    } else {
        chitti_kernel::net::autodetect();
        // Bring up audio (virtio-snd) for the /voice pipeline. No-op if absent.
        chitti_kernel::sound::autodetect();
        // Attach a host shared folder (virtio-9p) at /host. No-op if absent.
        chitti_kernel::fs::host::attach_at_boot();
        // Host clipboard channel (SPICE agent over virtio-serial).
        chitti_kernel::clipboard::agent_init();
    }
    // Everything is up (framebuffer, USB/input, disk, persistent store) with IRQs
    // masked. NOW begin timer-preemptive scheduling: unmask IRQs so the generic
    // timer preempts the shell. Deferred to here (not inside init()) so device
    // bring-up — the framebuffer especially — runs identically to the cooperative
    // path; a no-op where the GIC/timer wasn't available (HVF → cooperative).
    #[cfg(not(feature = "refcheck"))]
    {
        chitti_kernel::arch::aarch64::gic::start_preemption();
        run_os();
    }
    #[allow(unreachable_code)]
    loop {
        chitti_kernel::arch::aarch64::hlt();
    }
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

/// Adopt the active display's EDID from the stub's boot-info page (length at
/// offset 384, the 128-byte base block at 388).
///
/// This is what lets the OS know *which* screen it is on — its identity keys the
/// per-display settings, and its name is what `/display` reports. The firmware's
/// EDID buffer is gone by the time the kernel runs, so the loader has to hand the
/// bytes over; nothing here is fatal if it didn't (a hypervisor usually has no
/// EDID at all, and the settings then key off the framebuffer size instead).
#[cfg(all(target_arch = "aarch64", not(feature = "boot-limine")))]
fn aarch64_display_edid_init() {
    let Some(bi) = bootinfo_page() else { return };
    // SAFETY: identity-mapped RAM below the map limit; the boot-info page is one
    // 4 KiB page, so 388 + 128 is well inside it.
    let page = unsafe { core::slice::from_raw_parts(bi as *const u8, 4096) };
    let len = u32::from_le_bytes(page[384..388].try_into().unwrap()) as usize;
    if len == 0 || len > chitti_kernel::edid::BASE_BLOCK_LEN {
        return;
    }
    chitti_kernel::display::set_edid(&page[388..388 + len]);
}

#[cfg(not(all(target_arch = "aarch64", not(feature = "boot-limine"))))]
fn aarch64_display_edid_init() {}

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
    if addr == 0 {
        return None;
    }
    // The RAM identity map is sized to RAM only (no unbacked over-map — see
    // mmu::init). A framebuffer that lives in a device window above RAM won't be
    // covered, so map its GiB block(s) as Device on demand rather than skipping.
    let fb_end = addr + h * pitch;
    if fb_end > map_limit {
        let mut a = addr & !((1u64 << 30) - 1);
        while a < fb_end {
            chitti_kernel::arch::aarch64::mmu::map_device_gib(a);
            a += 1 << 30;
        }
        serial_println!("Chitti: boot-info framebuffer at {:#x} above RAM map -- mapped its window (Device)", addr);
    }
    // Pixel format at 48..52. An older stub that left these zero (all shifts 0,
    // bpp 0) is treated as the common XRGB8888 default.
    let (mut rs, mut gs, mut bs, mut bpp) = (page[48] as u32, page[49] as u32, page[50] as u32, page[51] as u64);
    if bpp == 0 {
        (rs, gs, bs, bpp) = (16, 8, 0, 4);
    }
    serial_println!(
        "Chitti: framebuffer from UEFI boot-info (GOP {}x{} at {:#x}, pitch {} bytes = {} px/line, {} bpp, shifts {}/{}/{})",
        w, h, addr, pitch, if bpp > 0 { pitch / bpp } else { 0 }, bpp, rs, gs, bs
    );
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

/// Total physical RAM (bytes) the UEFI stub reported at boot-info offset 92,
/// or `None` if there's no boot-info page or the field is 0.
#[cfg(all(target_arch = "aarch64", not(feature = "boot-limine")))]
fn bootinfo_ram_bytes() -> Option<u64> {
    let bi = bootinfo_page()?;
    // SAFETY: identity-mapped boot-info page; read the 8-byte RAM-total field.
    let page = unsafe { core::slice::from_raw_parts(bi as *const u8, 100) };
    let bytes = u64::from_le_bytes(page[92..100].try_into().unwrap());
    (bytes > 0).then_some(bytes)
}

/// Point `synapse::fs` at an ext4 *data* partition so agent writes are durable
/// across reboots (the installed system).
///
/// **Memfs is only for the live ISO/image.** When a `Chitti Data` (or other
/// data) volume is present — the permanent-disk case after `/install` — this
/// *must* adopt Ext4Store. Uses the same finder as VFS auto-mount so the two
/// never disagree.
fn mount_persistent_store() {
    use chitti_kernel::block::ext4_store::Ext4Store;
    use chitti_kernel::block::volcrypto;
    use chitti_kernel::block::BlockDevice;
    use chitti_kernel::block::Partition;

    // Inventory every disk (helps diagnose empty /disks after install).
    let mut disks_seen = 0usize;
    for disk in 0..16usize {
        let Some(dev) = chitti_kernel::block::probe_disk_nth(disk) else {
            break;
        };
        disks_seen += 1;
        let sectors = dev.block_count();
        serial_println!(
            "Chitti: disk {}: {} sectors ({} MiB)",
            disk,
            sectors,
            sectors * 512 / 1024 / 1024
        );
    }

    if disks_seen == 0 {
        serial_println!(
            "Chitti: no block device — live ISO/image mode; synapse store is memfs (not durable)"
        );
        return;
    }

    let Some(v) = chitti_kernel::fs::mount::find_data_volume() else {
        serial_println!(
            "Chitti: synapse persistence -> memfs (scanned {} disk(s); no data partition — ISO/image only)",
            disks_seen
        );
        return;
    };

    if v.named {
        serial_println!(
            "Chitti: found GPT 'Chitti Data' on disk {} lba {} ({} MiB{})",
            v.disk,
            v.start_lba,
            v.sectors * 512 / 1024 / 1024,
            if v.encrypted { ", encrypted" } else { "" }
        );
    }

    let disk = v.disk;
    let start = v.start_lba;
    let count = v.sectors;
    let encrypted = v.encrypted;

    let store = if encrypted {
        serial_println!(
            "Chitti: encrypted data partition on disk {} at lba {start} — unlock required",
            disk
        );
        let pass = chitti_kernel::modal::input("Unlock data volume", "Passphrase:", true);
        if pass.is_empty() {
            serial_println!(
                "Chitti: unlock cancelled; refusing memfs on a persistence disk — store unmounted"
            );
            return;
        }
        let Some(mut dev) = chitti_kernel::block::probe_disk_nth(disk) else {
            serial_println!("Chitti: disk {disk} disappeared during unlock");
            return;
        };
        let mut part = Partition::new(&mut dev, start, count);
        match volcrypto::unlock(&mut part, pass.as_bytes()) {
            Ok((key, hdr)) => {
                serial_println!(
                    "Chitti: volume unlocked (hdr={} sectors)",
                    hdr.hdr_sectors
                );
                drop(dev);
                Ext4Store::mount_encrypted(disk, start, count, key, hdr.hdr_sectors)
            }
            Err(_) => {
                serial_println!(
                    "Chitti: wrong passphrase or corrupt header; refusing memfs on a persistence disk"
                );
                return;
            }
        }
    } else {
        Ext4Store::mount(disk, start, count)
    };

    if let Some(store) = store {
        chitti_kernel::synapse::fs::mount_ext4(store);
        serial_println!(
            "Chitti: synapse persistence -> ext4 disk {} lba {} ({} sectors{}); NOT memfs",
            disk,
            start,
            count,
            if encrypted { ", encrypted" } else { "" }
        );
        let prior = chitti_kernel::synapse::fs::read("synapse_boots")
            .and_then(|b| {
                core::str::from_utf8(&b)
                    .ok()
                    .and_then(|s| s.trim().parse::<u32>().ok())
            })
            .unwrap_or(0);
        let boots = prior + 1;
        let mut buf = [0u8; 12];
        chitti_kernel::synapse::fs::write("synapse_boots", fmt_u32(boots, &mut buf).as_bytes());
        serial_println!(
            "Chitti: synapse.fs boot #{} (backend={}); durable files survive reboot",
            boots,
            chitti_kernel::synapse::fs::backend_name()
        );
        if prior > 0 {
            serial_println!(
                "Chitti: synapse.fs (boot counter survived reboot — agent writes persist on ext4)"
            );
        }
    } else {
        // Data volume *exists* but would not open — still must not silently use memfs.
        serial_println!(
            "Chitti: ERROR: data volume on disk {disk} lba {start} would not open — store left unmounted (not memfs)"
        );
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
