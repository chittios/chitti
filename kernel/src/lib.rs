//! Chitti OS kernel library: shared code between the real boot binary
//! (`src/main.rs`) and the in-kernel test harness (`cargo test --lib`,
//! compiled via `custom_test_frameworks`). Everything below the
//! determinism boundary starts here — see `CHITTI_OS_HANDOFF.md` Part 2.
#![no_std]
#![cfg_attr(test, no_main)]
#![feature(custom_test_frameworks)]
#![feature(abi_x86_interrupt)]
#![test_runner(crate::test_runner)]
#![reexport_test_harness_main = "test_main"]
extern crate alloc;

pub mod arch;
pub mod ktrace;
pub mod limine_protocol;
pub mod mm;
pub mod qemu;
pub mod serial;

// The framebuffer writer pulls in `font8x8` (a normal, non-build-std
// dependency). Building it under `cfg(test)` as well hits a known
// `-Z build-std` + `cargo test` interaction where the "plain" and
// "--test" compilations of this crate end up linked against two
// non-unified copies of `core`/`alloc` ("duplicate lang item" errors) for
// any ordinary dependency shared between them. The test harness never
// draws to the framebuffer, so it's simplest to just not compile the
// module (and font8x8) into the test binary at all.
#[cfg(not(test))]
pub mod framebuffer;

// --- Limine requests -------------------------------------------------
//
// A single definition here is linked into both the real kernel binary
// (which `use`s these through the lib) and the test-harness binary (which
// *is* this lib compiled with `--test`), so there is exactly one copy of
// the wire-format request structs to audit.

#[used]
#[link_section = ".requests_start_marker"]
static _REQUESTS_START: limine_protocol::RequestsStartMarker =
    limine_protocol::RequestsStartMarker::new();

/// Base revision 3: stable across all Limine 5.x-12.x releases.
#[used]
#[link_section = ".requests"]
pub static BASE_REVISION: limine_protocol::BaseRevision = limine_protocol::BaseRevision::new(3);

#[used]
#[link_section = ".requests"]
pub static FRAMEBUFFER_REQUEST: limine_protocol::FramebufferRequest =
    limine_protocol::FramebufferRequest::new();

#[used]
#[link_section = ".requests"]
pub static MEMMAP_REQUEST: limine_protocol::MemmapRequest = limine_protocol::MemmapRequest::new();

#[used]
#[link_section = ".requests"]
pub static HHDM_REQUEST: limine_protocol::HhdmRequest = limine_protocol::HhdmRequest::new();

#[used]
#[link_section = ".requests_end_marker"]
static _REQUESTS_END: limine_protocol::RequestsEndMarker =
    limine_protocol::RequestsEndMarker::new();

/// Phase 1 bring-up: GDT/TSS, IDT + exception handlers, FPU/SSE + NX,
/// the PIC/PIT/keyboard IRQ lines, the frame allocator + kernel heap, and
/// finally `sti`. Shared by the real boot binary (`main.rs`) and the
/// `custom_test_frameworks` harness below, so every test also runs with
/// interrupts, paging extensions, and a working heap available.
pub fn init() {
    assert!(BASE_REVISION.is_supported(), "Limine did not accept base revision 3");

    arch::x86_64::gdt::init();
    arch::x86_64::idt::init();
    arch::x86_64::fpu::init();
    arch::x86_64::pic::init();
    arch::x86_64::pit::init();
    arch::x86_64::keyboard::init();

    let hhdm_offset = HHDM_REQUEST.response().expect("HHDM request refused by Limine").offset;
    let memmap = MEMMAP_REQUEST
        .response()
        .expect("memory map request refused by Limine")
        .entries();
    mm::init(memmap, hhdm_offset);

    arch::x86_64::interrupts::enable();
    ktrace::log("init", "Phase 1 bring-up complete, interrupts enabled");
}

// --- custom_test_frameworks harness -----------------------------------

/// A runnable in-kernel test. Blanket-implemented for any `Fn()`, printing
/// its (mangled) name and an ok/failed marker around the call.
pub trait Testable {
    fn run(&self);
}

impl<T: Fn()> Testable for T {
    fn run(&self) {
        serial_print!("{}...\t", core::any::type_name::<T>());
        self();
        serial_println!("[ok]");
    }
}

pub fn test_runner(tests: &[&dyn Testable]) {
    serial_println!("running {} test(s)", tests.len());
    for test in tests {
        test.run();
    }
    qemu::exit_qemu(qemu::QemuExitCode::Success);
}

pub fn test_panic_handler(info: &core::panic::PanicInfo) -> ! {
    serial_println!("[failed]");
    serial_println!("{}", info);
    qemu::exit_qemu(qemu::QemuExitCode::Failed);
}

#[cfg(test)]
#[unsafe(no_mangle)]
pub extern "C" fn _start() -> ! {
    serial::init();
    init();
    test_main();
    loop {
        arch::x86_64::hlt();
    }
}

#[cfg(test)]
#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    test_panic_handler(info)
}

// --- Phase 0 sanity tests ----------------------------------------------

#[test_case]
fn trivial_assertion() {
    assert_eq!(1 + 1, 2);
}

#[test_case]
fn base_revision_is_supported() {
    assert!(BASE_REVISION.is_supported());
}

#[test_case]
fn framebuffer_response_present() {
    assert!(FRAMEBUFFER_REQUEST.response().is_some());
}

#[test_case]
fn memmap_response_present() {
    assert!(MEMMAP_REQUEST.response().is_some());
}

// --- Phase 1 acceptance tests -------------------------------------------

/// (a) the timer increments a counter over N ticks.
#[test_case]
fn timer_ticks_advance() {
    let start = arch::x86_64::pit::ticks();
    let target = start + 5;
    // Bounded spin: with the PIT at 1kHz this resolves in single-digit
    // milliseconds. If interrupts are broken this must fail loudly rather
    // than hang the whole QEMU test run forever.
    let mut spins = 0u64;
    while arch::x86_64::pit::ticks() < target {
        arch::x86_64::hlt();
        spins += 1;
        assert!(spins < 100_000_000, "timer did not advance -- IRQ0 is not firing");
    }
    assert!(arch::x86_64::pit::ticks() >= target);
}

/// (b) heap alloc/free of varied sizes with no corruption.
#[test_case]
fn heap_alloc_dealloc_varied_sizes() {
    use alloc::boxed::Box;
    use alloc::vec::Vec;

    // A single large-then-freed allocation, interleaved with many small
    // ones, exercises both the split-on-alloc and reinsert-on-dealloc
    // paths of the linked-list allocator. `vec![elem; n]` (unlike
    // `Box::new([elem; n])`) writes straight into the heap allocation via
    // `from_elem` rather than building an `n`-byte stack temporary first
    // -- important here since the boot stack is nowhere near 64 KiB.
    let big = alloc::vec![0xabu8; 64 * 1024];
    assert!(big.iter().all(|&b| b == 0xab));
    drop(big);

    let mut values = Vec::new();
    for i in 0..500u32 {
        values.push(Box::new(i));
    }
    for (i, v) in values.iter().enumerate() {
        assert_eq!(**v, i as u32);
    }
    drop(values);

    let mut vec: Vec<u64> = Vec::new();
    for i in 0..2000u64 {
        vec.push(i * i);
    }
    assert_eq!(vec.len(), 2000);
    assert_eq!(vec[1999], 1999 * 1999);
    for i in (0..2000).step_by(2) {
        vec[i] = 0;
    }
    assert_eq!(vec[0], 0);
    assert_eq!(vec[1], 1);
}

/// (c) a deliberately triggered exception is caught and reported, not a
/// triple fault: `int3` (breakpoint) is the safest exception to trigger
/// from a running test, since the handler `iretq`s straight back to the
/// next instruction rather than requiring any fault recovery.
#[test_case]
fn breakpoint_exception_is_caught_not_triple_faulted() {
    let hits_before = arch::x86_64::idt::BREAKPOINT_HITS.load(core::sync::atomic::Ordering::SeqCst);
    // SAFETY: `int3` is a normal, trap-gate-handled exception; the
    // handler logs and returns, so execution resumes right here.
    unsafe { core::arch::asm!("int3", options(nomem, nostack)) };
    let hits_after = arch::x86_64::idt::BREAKPOINT_HITS.load(core::sync::atomic::Ordering::SeqCst);
    assert_eq!(hits_after, hits_before + 1, "breakpoint handler did not run");
    // Reaching this line at all is the real proof: a triple fault would
    // have reset the VM, and this test (and every later one) would never
    // report a result.
}
