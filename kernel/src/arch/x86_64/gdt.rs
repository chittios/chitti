//! Global Descriptor Table + Task State Segment.
//!
//! Long mode makes segmentation almost entirely vestigial (base/limit are
//! ignored for code/data segments), but two things still require a GDT:
//! reloading `CS` with a 64-bit code descriptor, and loading the Task
//! Register (`ltr`) with a TSS descriptor so the CPU can find the
//! Interrupt Stack Table (IST) — the dedicated stack the double-fault
//! handler runs on, so a stack-overflow-caused double fault doesn't also
//! fault on its own handler's prologue.

use core::arch::asm;
use core::mem::size_of;

/// One 8-byte code/data segment descriptor, built up from the fields
/// defined by the x86 segment descriptor format rather than copied as a
/// magic hex constant, so every bit's meaning stays legible.
const fn segment_descriptor(access: u8, flags: u8) -> u64 {
    // base = 0, limit = 0: ignored by the CPU for long-mode code/data
    // segments (base is only consulted for FS/GS, limit not at all).
    ((flags as u64 & 0xf) << 52) | ((access as u64) << 40)
}

const ACCESS_PRESENT: u8 = 1 << 7;
const ACCESS_SEGMENT: u8 = 1 << 4; // S=1: code/data, not a system descriptor
const ACCESS_EXECUTABLE: u8 = 1 << 3;
const ACCESS_RW: u8 = 1 << 1; // readable (code) / writable (data)
const FLAGS_LONG_MODE: u8 = 1 << 1; // L bit: this is a 64-bit code segment

const KERNEL_CODE64: u64 =
    segment_descriptor(ACCESS_PRESENT | ACCESS_SEGMENT | ACCESS_EXECUTABLE | ACCESS_RW, FLAGS_LONG_MODE);
const KERNEL_DATA: u64 = segment_descriptor(ACCESS_PRESENT | ACCESS_SEGMENT | ACCESS_RW, 0);

pub const KERNEL_CODE_SELECTOR: u16 = 1 << 3; // index 1
pub const KERNEL_DATA_SELECTOR: u16 = 2 << 3; // index 2
const TSS_SELECTOR: u16 = 3 << 3; // index 3 (occupies indices 3 and 4)

/// Index into the TSS's Interrupt Stack Table used by the double-fault
/// handler (`idt.rs`). IST indices are 1-based; 0 means "don't switch
/// stacks".
pub const DOUBLE_FAULT_IST_INDEX: u16 = 1;

const DOUBLE_FAULT_STACK_SIZE: usize = 4096 * 4;

#[repr(align(16))]
struct Stack([u8; DOUBLE_FAULT_STACK_SIZE]);
static mut DOUBLE_FAULT_STACK: Stack = Stack([0; DOUBLE_FAULT_STACK_SIZE]);

/// x86_64 Task State Segment (Intel SDM Vol. 3A, 8.7). Only `ist[0]`
/// (IST1) is populated; `rsp`/the other ISTs are unused until Phase 2
/// introduces ring transitions and per-task kernel stacks.
#[repr(C, packed)]
struct Tss {
    reserved0: u32,
    rsp: [u64; 3],
    reserved1: u64,
    ist: [u64; 7],
    reserved2: u64,
    reserved3: u16,
    iomap_base: u16,
}

impl Tss {
    const fn new() -> Self {
        Self {
            reserved0: 0,
            rsp: [0; 3],
            reserved1: 0,
            ist: [0; 7],
            reserved2: 0,
            reserved3: 0,
            // Point the I/O permission bitmap past the TSS limit: there is
            // no bitmap, so every port I/O from ring 0 is (as usual for
            // the kernel) unconditionally permitted anyway.
            iomap_base: size_of::<Tss>() as u16,
        }
    }
}

static mut TSS: Tss = Tss::new();

#[repr(C, packed)]
struct DescriptorTablePointer {
    limit: u16,
    base: u64,
}

/// 5 slots: null, kernel code, kernel data, TSS (2 slots: a 64-bit system
/// descriptor is twice the width of a code/data descriptor).
static mut GDT: [u64; 5] = [0, KERNEL_CODE64, KERNEL_DATA, 0, 0];

fn tss_descriptor(tss_addr: u64) -> (u64, u64) {
    let limit = (size_of::<Tss>() - 1) as u64;
    const ACCESS_TSS_AVAILABLE: u64 = 0x89; // present, DPL0, type=0b1001 (64-bit TSS, available)
    let low = (limit & 0xffff)
        | ((tss_addr & 0xff_ffff) << 16)
        | (ACCESS_TSS_AVAILABLE << 40)
        | (((limit >> 16) & 0xf) << 48)
        | (((tss_addr >> 24) & 0xff) << 56);
    let high = (tss_addr >> 32) & 0xffff_ffff;
    (low, high)
}

/// Build the GDT/TSS, load them, and switch `CS`/`SS`/data segments over
/// to the new kernel descriptors. Must run before `idt::init()`, since the
/// IDT's interrupt gates reference `KERNEL_CODE_SELECTOR`.
pub fn init() {
    // SAFETY: single-threaded boot-time initialization; nothing else
    // touches these statics before or concurrently with this call.
    unsafe {
        let stack_top = core::ptr::addr_of!(DOUBLE_FAULT_STACK.0) as u64 + DOUBLE_FAULT_STACK_SIZE as u64;
        TSS.ist[(DOUBLE_FAULT_IST_INDEX - 1) as usize] = stack_top;

        let tss_addr = core::ptr::addr_of!(TSS) as u64;
        let (low, high) = tss_descriptor(tss_addr);
        GDT[3] = low;
        GDT[4] = high;

        let gdt_ptr = DescriptorTablePointer {
            limit: (size_of::<[u64; 5]>() - 1) as u16,
            base: core::ptr::addr_of!(GDT) as u64,
        };
        asm!("lgdt [{}]", in(reg) &gdt_ptr, options(readonly, nostack, preserves_flags));

        // Reload CS via a far return trick (there's no direct `mov cs`),
        // then reload the data segment registers and the task register.
        //
        // `code_sel` MUST be pushed before `tmp` is computed: `tmp` is a
        // `lateout` register, which tells LLVM it's free to alias it to
        // any `in` register (here, `code_sel`'s) once that input has been
        // consumed. Computing `tmp` first previously clobbered
        // `code_sel`'s register before its `push` ever ran, silently
        // pushing the same (wrong) value twice and turning `retfq` into a
        // jump to a bogus, unmapped code selector -- an instant #GP with
        // no IDT loaded yet, i.e. a triple fault.
        asm!(
            "push {code_sel}",
            "lea {tmp}, [55f + rip]",
            "push {tmp}",
            "retfq",
            "55:",
            code_sel = in(reg) KERNEL_CODE_SELECTOR as u64,
            tmp = lateout(reg) _,
            options(preserves_flags),
        );
        asm!(
            "mov ds, {sel:x}",
            "mov es, {sel:x}",
            "mov ss, {sel:x}",
            "mov fs, {sel:x}",
            "mov gs, {sel:x}",
            sel = in(reg) KERNEL_DATA_SELECTOR,
            options(nostack, preserves_flags),
        );
        asm!("ltr {sel:x}", sel = in(reg) TSS_SELECTOR, options(nostack, preserves_flags));
    }
    crate::ktrace::log("gdt", "GDT + TSS loaded, IST1 stack ready for double-fault");
}
