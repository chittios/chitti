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

    // Phase 4: demonstrate the Synapse capability ABI end to end (grammar
    // validation -> capability check -> deterministic execution -> audit).
    // Fast and model-free, so it runs on every boot regardless of whether
    // the Cortex model module is present.
    chitti_kernel::synapse::demo();

    // Phase 3: report the Cortex model boot module and run the reference
    // inference demo, if the model is present.
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
            // `ref-check` builds run the full acceptance gate and exit QEMU
            // with a pass/fail code; a normal `run` just shows the demo.
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
            run_inference_demo();
        }
        None => serial_println!("Chitti: no model.gguf boot module present"),
    }

    loop {
        arch::x86_64::hlt();
    }
}

/// Phase 3 deliverable: generate a coherent, reproducible token stream from
/// the tiny model. Logs the prompt and the model's response to both serial
/// (what `cargo xtask run` shows on stdio) and the framebuffer window, then
/// records the parity check the `REFCHECK:` line carries.
#[cfg(not(feature = "refcheck"))]
fn run_inference_demo() {
    use chitti_kernel::cortex::{self, refcheck};

    serial_println!("Chitti: --- Cortex inference ---");
    serial_println!("Chitti: prompt:   {}", refcheck::PROMPT);
    // The response streams token-by-token from run_reference_inference
    // (a "Chitti: response> ..." line) as it is generated.
    match cortex::run_reference_inference() {
        Some(result) => {
            serial_println!("Chitti: full:     {}{}", refcheck::PROMPT, result.continuation_text);
            serial_println!(
                "Chitti: (tokens={:?}, matches NumPy reference={})",
                result.continuation,
                result.matched_reference,
            );
            // Also render the prompt + response to the framebuffer window.
            if let Some(fb) = FRAMEBUFFER_REQUEST.response().and_then(|r| r.framebuffers().first().copied()) {
                let mut w = framebuffer::Writer::new(fb);
                let _ = write!(w, "\n\nprompt:   {}\nresponse: {}", refcheck::PROMPT, result.continuation_text);
            }
        }
        None => serial_println!("Chitti: inference could not run"),
    }
}

#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    serial_println!("KERNEL PANIC: {}", info);
    loop {
        arch::x86_64::hlt();
    }
}
