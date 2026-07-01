//! Chitti OS kernel library: shared code between the real boot binary
//! (`src/main.rs`) and the in-kernel test harness (`cargo test --lib`,
//! compiled via `custom_test_frameworks`). Everything below the
//! determinism boundary starts here — see `CHITTI_OS_HANDOFF.md` Part 2.
#![no_std]
#![cfg_attr(test, no_main)]
#![feature(custom_test_frameworks)]
#![test_runner(crate::test_runner)]
#![reexport_test_harness_main = "test_main"]

pub mod arch;
pub mod limine_protocol;
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
#[link_section = ".requests_end_marker"]
static _REQUESTS_END: limine_protocol::RequestsEndMarker =
    limine_protocol::RequestsEndMarker::new();

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
