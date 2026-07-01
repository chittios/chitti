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
pub mod cap;
pub mod cortex;
pub mod ipc;
pub mod ktrace;
pub mod limine_protocol;
pub mod mm;
pub mod qemu;
pub mod sched;
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
#[link_section = ".requests"]
pub static MODULE_REQUEST: limine_protocol::ModuleRequest = limine_protocol::ModuleRequest::new();

#[used]
#[link_section = ".requests_end_marker"]
static _REQUESTS_END: limine_protocol::RequestsEndMarker =
    limine_protocol::RequestsEndMarker::new();

/// Phase 0-2 bring-up: GDT/TSS, IDT + exception handlers, FPU/SSE + NX,
/// the PIC/PIT/keyboard IRQ lines, the frame allocator + kernel heap, the
/// task scheduler (wrapping this call's own context as task 0), and
/// finally `sti`. Shared by the real boot binary (`main.rs`) and the
/// `custom_test_frameworks` harness below, so every test also runs with
/// interrupts, paging extensions, a working heap, and a live scheduler
/// available.
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
    sched::init();

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
    // First action: enable SSE at the hardware level before any SIMD
    // codegen runs (see `arch::x86_64::fpu::enable_sse`).
    arch::x86_64::fpu::enable_sse();
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

// --- Phase 2 acceptance tests -------------------------------------------

/// (a) 3+ tasks interleave and all make progress, via voluntary
/// `yield_now` calls -- the cooperative half of "cooperative first, then
/// timer-preemptive." Each worker yields every single iteration, so a
/// correct round-robin scheduler produces a heavily interleaved log; a
/// broken one (e.g. a `switch_to` that corrupts state) would show up as
/// either a hang (bounded spin catches it) or a log dominated by one
/// task id (the transitions assertion catches it).
#[test_case]
fn cooperative_tasks_interleave_and_progress() {
    use alloc::vec::Vec;
    use core::sync::atomic::{AtomicU64, Ordering};

    static COUNTERS: [AtomicU64; 3] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
    static LOG: mm::Locked<Vec<u8>> = mm::Locked::new(Vec::new());
    const TARGET: u64 = 50;

    extern "C" fn worker(idx: u64) {
        let counter = &COUNTERS[idx as usize];
        while counter.load(Ordering::SeqCst) < TARGET {
            counter.fetch_add(1, Ordering::SeqCst);
            LOG.with(|log| log.push(idx as u8));
            sched::yield_now();
        }
    }

    for i in 0..3u64 {
        sched::spawn("worker", worker, i);
    }

    let mut spins = 0u64;
    while COUNTERS.iter().any(|c| c.load(Ordering::SeqCst) < TARGET) {
        sched::yield_now();
        spins += 1;
        assert!(spins < 100_000_000, "not all tasks progressed -- scheduler stuck or a task starved");
    }
    for c in &COUNTERS {
        assert_eq!(c.load(Ordering::SeqCst), TARGET);
    }

    let transitions = LOG.with(|log| log.windows(2).filter(|w| w[0] != w[1]).count());
    assert!(transitions > TARGET as usize, "tasks ran sequentially instead of interleaving (only {transitions} switches)");
}

/// Timer-preemptive half: a task that never calls `yield_now` at all
/// still gets interrupted and descheduled, proving the PIT tick hook
/// (`sched::on_timer_tick`, wired from `pit::timer_handler`) actually
/// forces a switch rather than only supporting voluntary cooperation.
#[test_case]
fn timer_preempts_a_non_yielding_task() {
    use core::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);

    extern "C" fn hog(_arg: u64) {
        loop {
            COUNTER.fetch_add(1, Ordering::SeqCst);
        }
    }

    sched::spawn("hog", hog, 0);

    let mut spins = 0u64;
    while COUNTER.load(Ordering::SeqCst) < 1000 {
        arch::x86_64::hlt();
        spins += 1;
        assert!(spins < 100_000_000, "timer preemption never returned control -- hog task starved the CPU");
    }
    // Reaching this line at all is the proof: `hog` never yields, so
    // control could only have returned here via `on_timer_tick` forcing
    // a switch back to this (bootstrap) task.
}

/// (b) an IPC round-trip delivers a message: the main test task sends a
/// request on `request_ep`, a spawned responder task receives it, doubles
/// it, and sends the result back on `reply_ep`.
#[test_case]
fn ipc_round_trip_delivers_a_message() {
    // Capability convention for `responder` (documented here since a
    // task has no other way to learn its own capability-table layout):
    // slot 0 is granted an `IpcReceive(request_ep)` right, slot 1 an
    // `IpcSend(reply_ep)` right, in that order, before the task ever runs.
    extern "C" fn responder(_arg: u64) {
        let recv_cap = cap::Cap(0);
        let send_cap = cap::Cap(1);
        let request = ipc::receive(recv_cap).expect("responder: receive was denied");
        ipc::send(send_cap, request.data * 2).expect("responder: send was denied");
    }

    let request_ep = ipc::create_endpoint();
    let reply_ep = ipc::create_endpoint();

    let responder_id = sched::spawn("ipc_responder", responder, 0);
    cap::grant(responder_id, cap::Right::IpcReceive(request_ep)); // responder's slot 0
    cap::grant(responder_id, cap::Right::IpcSend(reply_ep)); // responder's slot 1

    let my_send_cap = cap::grant(sched::current_task_id(), cap::Right::IpcSend(request_ep));
    let my_recv_cap = cap::grant(sched::current_task_id(), cap::Right::IpcReceive(reply_ep));

    ipc::send(my_send_cap, 21).expect("main: send was denied");
    let reply = ipc::receive(my_recv_cap).expect("main: receive was denied");
    assert_eq!(reply.data, 42, "round trip did not deliver the expected reply");
    assert_eq!(reply.sender, responder_id, "reply reported the wrong sender");
}

/// (c) a task lacking a capability is denied the gated operation, and the
/// denial is `ktrace`d (`cap::record_denial`, asserted here via the
/// `cap::denials()` counter it also increments).
#[test_case]
fn capability_denial_is_refused_and_ktraced() {
    use core::sync::atomic::{AtomicBool, Ordering};

    static DONE: AtomicBool = AtomicBool::new(false);

    extern "C" fn unprivileged_sender(_arg: u64) {
        // This task was never granted any capability, so slot 0 doesn't
        // exist in its table: the send must be denied, not silently
        // allowed.
        let bogus_cap = cap::Cap(0);
        let result = ipc::send(bogus_cap, 0);
        assert_eq!(result, Err(ipc::IpcError::CapabilityDenied), "send without a capability was not denied");
        DONE.store(true, Ordering::SeqCst);
    }

    let denials_before = cap::denials();
    sched::spawn("unprivileged_sender", unprivileged_sender, 0);

    let mut spins = 0u64;
    while !DONE.load(Ordering::SeqCst) {
        arch::x86_64::hlt();
        spins += 1;
        assert!(spins < 100_000_000, "the unprivileged task never ran to completion");
    }
    assert_eq!(cap::denials(), denials_before + 1, "the denial was not recorded/ktrace'd");
}

/// The cooperative async executor (`sched::executor`) is a separate
/// concurrency layer from the stackful scheduler above -- this proves two
/// futures interleave (each yields once via a waker-rescheduled `Poll`)
/// entirely within one stackful task's call to `Executor::run`.
#[test_case]
fn async_executor_interleaves_two_futures() {
    use core::future::Future;
    use core::pin::Pin;
    use core::sync::atomic::{AtomicU64, Ordering};
    use core::task::{Context, Poll};

    struct YieldOnce {
        yielded: bool,
    }

    impl Future for YieldOnce {
        type Output = ();
        fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
            if self.yielded {
                Poll::Ready(())
            } else {
                self.yielded = true;
                cx.waker().wake_by_ref();
                Poll::Pending
            }
        }
    }

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    let mut executor = sched::executor::Executor::new();
    executor.spawn(async {
        YieldOnce { yielded: false }.await;
        COUNTER.fetch_add(1, Ordering::SeqCst);
    });
    executor.spawn(async {
        YieldOnce { yielded: false }.await;
        COUNTER.fetch_add(1, Ordering::SeqCst);
    });
    executor.run();

    assert_eq!(COUNTER.load(Ordering::SeqCst), 2, "not all async tasks ran to completion");
}
