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
    // frame allocator + kernel heap, then `sti`. See `chitti_kernel::init`.
    chitti_kernel::init();

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
