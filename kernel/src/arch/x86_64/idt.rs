//! Interrupt Descriptor Table: CPU exception handlers (vectors 0-31) and
//! the hardware IRQ handlers the PIC remaps to vectors 32-47 (`pic.rs`).
//!
//! Handlers use the nightly `"x86-interrupt"` calling convention so the
//! compiler generates the `iretq`-compatible prologue/epilogue itself
//! (saving/restoring registers, popping the error code where present)
//! instead of us hand-writing per-vector asm trampolines.

use super::gdt::DOUBLE_FAULT_IST_INDEX;
use core::arch::asm;
use core::mem::size_of;

/// What the CPU pushes onto the stack before entering a handler (Intel
/// SDM Vol. 3A, 6.12.1). Identical for every vector; vectors with an error
/// code get it as an extra leading `u64` parameter instead of a struct
/// field, per the `"x86-interrupt"` ABI.
#[repr(C)]
pub struct InterruptStackFrame {
    pub instruction_pointer: u64,
    pub code_segment: u64,
    pub cpu_flags: u64,
    pub stack_pointer: u64,
    pub stack_segment: u64,
}

type Handler = extern "x86-interrupt" fn(InterruptStackFrame);
type HandlerWithErrorCode = extern "x86-interrupt" fn(InterruptStackFrame, u64);

const GATE_PRESENT: u8 = 1 << 7;
const GATE_TYPE_INTERRUPT: u8 = 0xe; // disables further interrupts on entry
const GATE_TYPE_TRAP: u8 = 0xf; // leaves IF as-is (used for breakpoint/debug)

#[repr(C, packed)]
#[derive(Clone, Copy)]
struct IdtEntry {
    offset_low: u16,
    selector: u16,
    ist: u8,
    type_attr: u8,
    offset_mid: u16,
    offset_high: u32,
    reserved: u32,
}

impl IdtEntry {
    const fn missing() -> Self {
        Self {
            offset_low: 0,
            selector: 0,
            ist: 0,
            type_attr: 0,
            offset_mid: 0,
            offset_high: 0,
            reserved: 0,
        }
    }

    fn set(&mut self, handler_addr: u64, ist_index: u8, gate_type: u8) {
        self.offset_low = handler_addr as u16;
        self.offset_mid = (handler_addr >> 16) as u16;
        self.offset_high = (handler_addr >> 32) as u32;
        self.selector = super::gdt::KERNEL_CODE_SELECTOR;
        self.ist = ist_index;
        self.type_attr = GATE_PRESENT | gate_type;
        self.reserved = 0;
    }
}

const ENTRY_COUNT: usize = 256;
static mut IDT: [IdtEntry; ENTRY_COUNT] = [IdtEntry::missing(); ENTRY_COUNT];

#[repr(C, packed)]
struct DescriptorTablePointer {
    limit: u16,
    base: u64,
}

// --- exception vectors ---------------------------------------------------

pub const DIVIDE_ERROR: u8 = 0;
pub const DEBUG: u8 = 1;
pub const NMI: u8 = 2;
pub const BREAKPOINT: u8 = 3;
pub const OVERFLOW: u8 = 4;
pub const BOUND_RANGE_EXCEEDED: u8 = 5;
pub const INVALID_OPCODE: u8 = 6;
pub const DEVICE_NOT_AVAILABLE: u8 = 7;
pub const DOUBLE_FAULT: u8 = 8;
pub const INVALID_TSS: u8 = 10;
pub const SEGMENT_NOT_PRESENT: u8 = 11;
pub const STACK_SEGMENT_FAULT: u8 = 12;
pub const GENERAL_PROTECTION_FAULT: u8 = 13;
pub const PAGE_FAULT: u8 = 14;
pub const X87_FPU_ERROR: u8 = 16;
pub const ALIGNMENT_CHECK: u8 = 17;
pub const MACHINE_CHECK: u8 = 18;
pub const SIMD_FP_EXCEPTION: u8 = 19;

/// Incremented by the breakpoint handler; `kernel::tests` asserts this to
/// prove a deliberately triggered exception was caught and reported
/// (rather than triple-faulting), per Phase 1's acceptance criteria.
pub static BREAKPOINT_HITS: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

extern "x86-interrupt" fn breakpoint_handler(frame: InterruptStackFrame) {
    BREAKPOINT_HITS.fetch_add(1, core::sync::atomic::Ordering::SeqCst);
    crate::ktrace::log_fmt(format_args!(
        "idt: breakpoint (int3) at rip={:#x}, resuming",
        frame.instruction_pointer
    ));
}

// Declared to return `()`, matching `HandlerWithErrorCode`, even though
// the trailing `loop` never actually returns: a double fault is fatal in
// Phase 1 (no stack/task recovery yet), so this halts instead.
extern "x86-interrupt" fn double_fault_handler(frame: InterruptStackFrame, error_code: u64) {
    crate::ktrace::log_fmt(format_args!(
        "idt: DOUBLE FAULT (error={error_code:#x}, rip={:#x}) -- halting, not triple-faulting",
        frame.instruction_pointer
    ));
    loop {
        super::hlt();
    }
}

extern "x86-interrupt" fn page_fault_handler(frame: InterruptStackFrame, error_code: u64) {
    let faulting_addr: u64;
    // SAFETY: CR2 holds the faulting linear address for the page fault
    // currently being handled; reading it has no side effects.
    unsafe { asm!("mov {}, cr2", out(reg) faulting_addr, options(nomem, nostack, preserves_flags)) };
    crate::ktrace::log_fmt(format_args!(
        "idt: PAGE FAULT accessing {faulting_addr:#x} (error={error_code:#x}, rip={:#x})",
        frame.instruction_pointer
    ));
    // A tenant's fault is not the kernel's. The scheduler task here is the *kernel*
    // one that called `enter_ring3` — a tenant is ring-3 code inside that task, not a
    // task of its own — so `fault_current_task` would correctly refuse to kill it and
    // we would then halt a working machine over a tenant's bad pointer. Hand control
    // back to `enter_ring3`, which reports `Exit::Fault`.
    if super::fastcall::tenant_live() {
        // SAFETY: a tenant is live, so `RESUME_SLOT` names the kernel stack that
        // entered ring 3.
        unsafe { super::fastcall::abort_tenant(error_code, faulting_addr) };
    }
    // Otherwise contain it to the faulting task: this handler runs on that task's own
    // stack (only #DF has an IST), so abandoning the task abandons this frame with
    // it. Returns only when isolation is impossible — no scheduler yet, or the
    // bootstrap task, where a fault is a kernel bug.
    crate::sched::fault_current_task("page fault");
    // Not isolatable (the shell's own task, or no scheduler yet): try the
    // shell's recovery landmark before halting. Same reasoning as the aarch64
    // dispatcher — this handler is on the faulting task's own stack (only #DF
    // uses an IST), so nothing will ever `iretq` from this frame.
    let armed = crate::fault_recovery::is_armed();
    let held = crate::mm::locks_held();
    let n = crate::fault_recovery::consecutive();
    if crate::fault_recovery::should_recover(armed, held, n) {
        // SAFETY: as above; the gate has established no `Locked` is held.
        unsafe { crate::fault_recovery::recover("page fault") };
    }
    if let Some(why) = crate::fault_recovery::refusal(armed, held, n) {
        crate::ktrace::log_fmt(format_args!("idt: cannot return to the prompt -- {why}"));
    }
    crate::ktrace::log("idt", "page fault not isolatable -- halting");
    loop {
        super::hlt();
    }
}

extern "x86-interrupt" fn general_protection_fault_handler(frame: InterruptStackFrame, error_code: u64) {
    crate::ktrace::log_fmt(format_args!(
        "idt: GENERAL PROTECTION FAULT (error={error_code:#x}, rip={:#x})",
        frame.instruction_pointer
    ));
    if super::fastcall::tenant_live() {
        // SAFETY: as in the page-fault handler above.
        unsafe { super::fastcall::abort_tenant(error_code, frame.instruction_pointer) };
    }
    crate::sched::fault_current_task("general protection fault");
    // Not isolatable (the shell's own task, or no scheduler yet): try the
    // shell's recovery landmark before halting. Same reasoning as the aarch64
    // dispatcher — this handler is on the faulting task's own stack (only #DF
    // uses an IST), so nothing will ever `iretq` from this frame.
    let armed = crate::fault_recovery::is_armed();
    let held = crate::mm::locks_held();
    let n = crate::fault_recovery::consecutive();
    if crate::fault_recovery::should_recover(armed, held, n) {
        // SAFETY: as above; the gate has established no `Locked` is held.
        unsafe { crate::fault_recovery::recover("general protection fault") };
    }
    if let Some(why) = crate::fault_recovery::refusal(armed, held, n) {
        crate::ktrace::log_fmt(format_args!("idt: cannot return to the prompt -- {why}"));
    }
    crate::ktrace::log("idt", "GP fault not isolatable -- halting");
    loop {
        super::hlt();
    }
}

macro_rules! unhandled_exception {
    ($name:ident, $vector:expr) => {
        extern "x86-interrupt" fn $name(frame: InterruptStackFrame) {
            crate::ktrace::log_fmt(format_args!(
                "idt: unhandled exception {} (rip={:#x}) -- halting, not triple-faulting",
                $vector, frame.instruction_pointer
            ));
            loop {
                super::hlt();
            }
        }
    };
}

unhandled_exception!(divide_error_handler, DIVIDE_ERROR);
unhandled_exception!(debug_handler, DEBUG);
unhandled_exception!(nmi_handler, NMI);
unhandled_exception!(overflow_handler, OVERFLOW);
unhandled_exception!(bound_range_handler, BOUND_RANGE_EXCEEDED);
unhandled_exception!(invalid_opcode_handler, INVALID_OPCODE);
unhandled_exception!(device_not_available_handler, DEVICE_NOT_AVAILABLE);
unhandled_exception!(x87_fpu_error_handler, X87_FPU_ERROR);
unhandled_exception!(machine_check_handler, MACHINE_CHECK);
unhandled_exception!(simd_fp_exception_handler, SIMD_FP_EXCEPTION);

macro_rules! unhandled_exception_with_error_code {
    ($name:ident, $vector:expr) => {
        extern "x86-interrupt" fn $name(frame: InterruptStackFrame, error_code: u64) {
            crate::ktrace::log_fmt(format_args!(
                "idt: unhandled exception {} (error={error_code:#x}, rip={:#x}) -- halting, not triple-faulting",
                $vector, frame.instruction_pointer
            ));
            loop {
                super::hlt();
            }
        }
    };
}

unhandled_exception_with_error_code!(invalid_tss_handler, INVALID_TSS);
unhandled_exception_with_error_code!(segment_not_present_handler, SEGMENT_NOT_PRESENT);
unhandled_exception_with_error_code!(stack_segment_fault_handler, STACK_SEGMENT_FAULT);
unhandled_exception_with_error_code!(alignment_check_handler, ALIGNMENT_CHECK);

fn set_handler(vector: u8, handler: Handler, ist_index: u8, gate_type: u8) {
    // SAFETY: single-threaded boot-time initialization of the IDT array,
    // completed (via `lidt`) before interrupts are ever enabled.
    unsafe { IDT[vector as usize].set(handler as u64, ist_index, gate_type) };
}

fn set_handler_with_error_code(vector: u8, handler: HandlerWithErrorCode, ist_index: u8, gate_type: u8) {
    // SAFETY: see `set_handler`.
    unsafe { IDT[vector as usize].set(handler as u64, ist_index, gate_type) };
}

/// Install an IRQ handler at `vector` (32-47, i.e. `pic::IRQ_BASE +
/// irq_line`). Used by `pic.rs`/`pit.rs`/`keyboard.rs`.
pub fn set_irq_handler(vector: u8, handler: Handler) {
    set_handler(vector, handler, 0, GATE_TYPE_INTERRUPT);
}

pub fn init() {
    set_handler(DIVIDE_ERROR, divide_error_handler, 0, GATE_TYPE_INTERRUPT);
    set_handler(DEBUG, debug_handler, 0, GATE_TYPE_TRAP);
    set_handler(NMI, nmi_handler, 0, GATE_TYPE_INTERRUPT);
    set_handler(BREAKPOINT, breakpoint_handler, 0, GATE_TYPE_TRAP);
    set_handler(OVERFLOW, overflow_handler, 0, GATE_TYPE_TRAP);
    set_handler(BOUND_RANGE_EXCEEDED, bound_range_handler, 0, GATE_TYPE_INTERRUPT);
    set_handler(INVALID_OPCODE, invalid_opcode_handler, 0, GATE_TYPE_INTERRUPT);
    set_handler(DEVICE_NOT_AVAILABLE, device_not_available_handler, 0, GATE_TYPE_INTERRUPT);
    set_handler_with_error_code(
        DOUBLE_FAULT,
        double_fault_handler,
        DOUBLE_FAULT_IST_INDEX as u8,
        GATE_TYPE_INTERRUPT,
    );
    set_handler_with_error_code(INVALID_TSS, invalid_tss_handler, 0, GATE_TYPE_INTERRUPT);
    set_handler_with_error_code(SEGMENT_NOT_PRESENT, segment_not_present_handler, 0, GATE_TYPE_INTERRUPT);
    set_handler_with_error_code(STACK_SEGMENT_FAULT, stack_segment_fault_handler, 0, GATE_TYPE_INTERRUPT);
    set_handler_with_error_code(
        GENERAL_PROTECTION_FAULT,
        general_protection_fault_handler,
        0,
        GATE_TYPE_INTERRUPT,
    );
    set_handler_with_error_code(PAGE_FAULT, page_fault_handler, 0, GATE_TYPE_INTERRUPT);
    set_handler(X87_FPU_ERROR, x87_fpu_error_handler, 0, GATE_TYPE_INTERRUPT);
    set_handler_with_error_code(ALIGNMENT_CHECK, alignment_check_handler, 0, GATE_TYPE_INTERRUPT);
    set_handler(MACHINE_CHECK, machine_check_handler, 0, GATE_TYPE_INTERRUPT);
    set_handler(SIMD_FP_EXCEPTION, simd_fp_exception_handler, 0, GATE_TYPE_INTERRUPT);

    // SAFETY: `IDT` is fully populated above before this load, and nothing
    // else touches it concurrently (single-threaded boot).
    unsafe {
        let ptr = DescriptorTablePointer {
            limit: (size_of::<[IdtEntry; ENTRY_COUNT]>() - 1) as u16,
            base: core::ptr::addr_of!(IDT) as u64,
        };
        asm!("lidt [{}]", in(reg) &ptr, options(readonly, nostack, preserves_flags));
    }
    crate::ktrace::log("idt", "IDT loaded: CPU exceptions installed");
}

/// Load the (shared, already-populated) IDT on an application processor. The
/// IDT table itself is filled once by the BSP's `init`; each core only needs
/// to point its own `IDTR` at it via `lidt`, so exceptions on that core reach
/// our handlers instead of triple-faulting.
pub fn load_ap() {
    // SAFETY: `IDT` was fully populated by `init` (which runs on the BSP
    // before any AP starts); `lidt` only reads it and sets this core's IDTR.
    unsafe {
        let ptr = DescriptorTablePointer {
            limit: (size_of::<[IdtEntry; ENTRY_COUNT]>() - 1) as u16,
            base: core::ptr::addr_of!(IDT) as u64,
        };
        asm!("lidt [{}]", in(reg) &ptr, options(readonly, nostack, preserves_flags));
    }
}
