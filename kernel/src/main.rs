//! Real boot entry point. `cargo xtask run` boots this binary.
#![no_std]
#![no_main]

use chitti_kernel::{
    arch, framebuffer, limine_protocol, serial, serial_println, FRAMEBUFFER_REQUEST, MEMMAP_REQUEST,
};
use core::fmt::Write as _;
use core::panic::PanicInfo;

const BOOT_MSG: &str = "Chitti: boot ok";

#[unsafe(no_mangle)]
pub extern "C" fn _start() -> ! {
    // Must be first: SIMD codegen is on crate-wide (Phase 3), so the
    // optimizer may emit SSE (XMM) instructions in any code below; they
    // fault until SSE is enabled at the hardware level. See fpu::enable_sse.
    arch::x86_64::fpu::enable_sse();

    serial::init();
    serial_println!("{}", BOOT_MSG);

    if let Some(fb_resp) = FRAMEBUFFER_REQUEST.response() {
        if let Some(fb) = fb_resp.framebuffers().first() {
            let mut writer = framebuffer::Writer::new(fb);
            let _ = write!(writer, "{}", BOOT_MSG);
        } else {
            serial_println!("Chitti: no framebuffers reported");
        }
    } else {
        serial_println!("Chitti: framebuffer request was refused");
    }

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

    // GDT/TSS, IDT + exceptions, FPU/SSE + NX, PIC/PIT/keyboard IRQs, the
    // frame allocator + kernel heap, the scheduler, then `sti`. See
    // `chitti_kernel::init`.
    chitti_kernel::init();

    // Phase 7: report the SMP bring-up result (the self-test ran inside init).
    serial_println!("Chitti: SMP: {} core(s) online (see ktrace 'smp:' lines for the spinlock self-test)", chitti_kernel::smp::cpu_count());

    // Phase 4: demonstrate the Synapse capability ABI end to end (grammar
    // validation -> capability check -> deterministic execution -> audit).
    // Fast and model-free, so it runs on every boot regardless of whether
    // the Cortex model module is present.
    chitti_kernel::synapse::demo();

    // Report the Cortex model boot module, if present (Phase 3). The model
    // is used on demand via the shell's `infer` builtin rather than in a
    // blocking boot-time demo -- inference is slow under QEMU TCG.
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

    // `ref-check` builds run the full Phase 3 acceptance gate and exit QEMU
    // with a pass/fail code, skipping the interactive shell.
    #[cfg(feature = "refcheck")]
    {
        let ok = chitti_kernel::cortex::run_acceptance();
        chitti_kernel::qemu::exit_qemu(if ok {
            chitti_kernel::qemu::QemuExitCode::Success
        } else {
            chitti_kernel::qemu::QemuExitCode::Failed
        });
    }

    // Phase 5: a fast, deterministic demonstration of the intent->plan->act
    // loop, then hand the console to the interactive intent shell (which
    // never returns -- it is the system's steady state).
    #[cfg(not(feature = "refcheck"))]
    {
        chitti_kernel::shell::demo();
        chitti_kernel::shell::run();
    }

    #[allow(unreachable_code)]
    loop {
        arch::x86_64::hlt();
    }
}

#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    serial_println!("KERNEL PANIC: {}", info);
    loop {
        arch::x86_64::hlt();
    }
}
