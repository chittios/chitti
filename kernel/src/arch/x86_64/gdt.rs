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

/// Descriptor privilege level 3 — ring 3. Bits 46:45 of the descriptor, which is
/// bits 6:5 of the access byte.
const ACCESS_DPL3: u8 = 3 << 5;

const KERNEL_CODE64: u64 =
    segment_descriptor(ACCESS_PRESENT | ACCESS_SEGMENT | ACCESS_EXECUTABLE | ACCESS_RW, FLAGS_LONG_MODE);
const KERNEL_DATA: u64 = segment_descriptor(ACCESS_PRESENT | ACCESS_SEGMENT | ACCESS_RW, 0);
/// Ring-3 data and code. Identical to the kernel pair but for `DPL3`; in long
/// mode base/limit are ignored, so privilege is the entire difference.
const USER_DATA: u64 = segment_descriptor(ACCESS_PRESENT | ACCESS_SEGMENT | ACCESS_RW | ACCESS_DPL3, 0);
const USER_CODE64: u64 = segment_descriptor(
    ACCESS_PRESENT | ACCESS_SEGMENT | ACCESS_EXECUTABLE | ACCESS_RW | ACCESS_DPL3,
    FLAGS_LONG_MODE,
);

// **The order of these is not a style choice; `syscall`/`sysret` derive selectors
// from it arithmetically.** `syscall` loads `CS = STAR[47:32]` and
// `SS = STAR[47:32] + 8`, so kernel code must be immediately followed by kernel
// data. `sysretq` loads `CS = STAR[63:48] + 16` and `SS = STAR[63:48] + 8`, so
// from a base B the GDT must hold user *data* at B+8 and user *code* at B+16 —
// note the inversion relative to the kernel pair, which is the detail that makes
// a hand-built GDT fault on the first return to ring 3 rather than at setup.
//
// Hence: null, kernel code, kernel data, user data, user code, TSS. The TSS moved
// to the end so `STAR[63:48]` does not have to point into the middle of it.
pub const KERNEL_CODE_SELECTOR: u16 = 1 << 3; // index 1 -> 0x08
pub const KERNEL_DATA_SELECTOR: u16 = 2 << 3; // index 2 -> 0x10
/// Ring-3 selectors, **including RPL 3 in the low bits**. Loading these without
/// the RPL set is a general-protection fault, and one that looks like a bad
/// descriptor rather than a bad selector.
pub const USER_DATA_SELECTOR: u16 = (3 << 3) | 3; // index 3 -> 0x1b
pub const USER_CODE_SELECTOR: u16 = (4 << 3) | 3; // index 4 -> 0x23
const TSS_SELECTOR: u16 = 5 << 3; // index 5 (occupies indices 5 and 6)

/// The `sysret` base: `STAR[63:48]`. `sysretq` computes user CS as base+16 and
/// user SS as base+8, which with the layout above is index 4 and index 3.
const SYSRET_BASE_SELECTOR: u16 = 2 << 3; // 0x10

/// The `sysret` base selector, for `fastcall`'s `STAR` composition.
pub const fn sysret_base_selector() -> u16 {
    SYSRET_BASE_SELECTOR
}

/// Compose the `STAR` MSR: kernel CS for `syscall` in 47:32, the `sysret` base in
/// 63:48. Pure so the arithmetic that decides which selectors ring 3 gets is
/// checkable without a CPU.
pub const fn star_value(syscall_cs: u16, sysret_base: u16) -> u64 {
    ((sysret_base as u64) << 48) | ((syscall_cs as u64) << 32)
}

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

/// 7 slots: null, kernel code, kernel data, user data, user code, TSS (2 slots:
/// a 64-bit system descriptor is twice the width of a code/data descriptor).
/// See the selector constants for why user *data* precedes user *code*.
static mut GDT: [u64; 7] = [0, KERNEL_CODE64, KERNEL_DATA, USER_DATA, USER_CODE64, 0, 0];

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
        GDT[5] = low;
        GDT[6] = high;

        let gdt_ptr = DescriptorTablePointer {
            limit: (size_of::<[u64; 7]>() - 1) as u16,
            base: core::ptr::addr_of!(GDT) as u64,
        };
        asm!("lgdt [{}]", in(reg) &gdt_ptr, options(readonly, nostack, preserves_flags));
        reload_segments();
    }
    crate::ktrace::log("gdt", "GDT + TSS loaded, IST1 stack ready for double-fault");
}

/// Set `TSS.rsp0`: the stack the CPU switches to when a trap arrives **from ring
/// 3**. Must name the current task's kernel stack.
///
/// Nothing needed this while every task ran in ring 0 — a trap from ring 0 keeps
/// the current stack and never consults `rsp0`, which is why the TSS has carried
/// an all-zero `rsp` array until now. The moment a task runs in ring 3 it becomes
/// load-bearing: a zero here means the first syscall or interrupt from userspace
/// pushes its frame at address 0, which is a page fault whose own handler also
/// has no stack — a double fault, then a triple. So this is called on every switch
/// to a task that can enter ring 3, not once at boot.
pub fn set_kernel_stack(top: u64) {
    // SAFETY: `TSS` is this core's task-state segment; `rsp0` is only read by the
    // CPU on a privilege-raising trap, and writing it does not affect the
    // currently-executing (ring 0) context.
    unsafe { TSS.rsp[0] = top };
}

/// The current `TSS.rsp0`, for diagnostics and tests.
pub fn kernel_stack() -> u64 {
    // SAFETY: a plain read of this core's TSS field.
    unsafe { TSS.rsp[0] }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The descriptor's DPL field, bits 46:45.
    fn dpl(desc: u64) -> u8 {
        ((desc >> 45) & 0b11) as u8
    }
    fn present(desc: u64) -> bool {
        desc & (1 << 47) != 0
    }
    fn long_mode(desc: u64) -> bool {
        desc & (1 << 53) != 0
    }
    fn executable(desc: u64) -> bool {
        desc & (1 << 43) != 0
    }

    #[test_case]
    fn user_descriptors_are_ring_three_and_kernel_ones_are_not() {
        // A DPL that silently stayed 0 would give "userspace" full privilege
        // while every other part of the transition appeared to work — the failure
        // would be no isolation rather than a fault.
        assert_eq!(dpl(USER_CODE64), 3);
        assert_eq!(dpl(USER_DATA), 3);
        assert_eq!(dpl(KERNEL_CODE64), 0);
        assert_eq!(dpl(KERNEL_DATA), 0);
        assert!(present(USER_CODE64) && present(USER_DATA));
        // 64-bit code needs the L bit; data segments must not set it.
        assert!(long_mode(USER_CODE64) && executable(USER_CODE64));
        assert!(!long_mode(USER_DATA) && !executable(USER_DATA));
    }

    #[test_case]
    fn selectors_carry_rpl_three() {
        // Loading a ring-3 selector with RPL 0 is a #GP that reads like a bad
        // descriptor rather than a bad selector, so the RPL is part of the
        // constant rather than something each call site remembers.
        assert_eq!(USER_CODE_SELECTOR & 3, 3);
        assert_eq!(USER_DATA_SELECTOR & 3, 3);
        assert_eq!(KERNEL_CODE_SELECTOR & 3, 0);
        // Index bits still name the right GDT slots.
        assert_eq!(USER_DATA_SELECTOR >> 3, 3);
        assert_eq!(USER_CODE_SELECTOR >> 3, 4);
    }

    #[test_case]
    fn the_gdt_layout_satisfies_what_syscall_and_sysret_compute() {
        // The whole reason the order is what it is. `syscall` takes CS from
        // STAR[47:32] and SS from that +8; `sysretq` takes CS from STAR[63:48]+16
        // and SS from +8 — note the inversion. Getting this wrong faults on the
        // first *return* to ring 3, long after setup looked fine.
        let star = star_value(KERNEL_CODE_SELECTOR, SYSRET_BASE_SELECTOR);
        let syscall_cs = ((star >> 32) & 0xffff) as u16;
        let sysret_base = ((star >> 48) & 0xffff) as u16;
        assert_eq!(syscall_cs, KERNEL_CODE_SELECTOR);
        assert_eq!(syscall_cs + 8, KERNEL_DATA_SELECTOR, "syscall's SS must follow its CS");
        assert_eq!(sysret_base + 16, USER_CODE_SELECTOR & !3, "sysret CS = base + 16");
        assert_eq!(sysret_base + 8, USER_DATA_SELECTOR & !3, "sysret SS = base + 8");
        // And the descriptors those selectors name really are the user pair.
        // SAFETY: boot-time-initialised static, read-only here.
        let gdt = unsafe { &*core::ptr::addr_of!(GDT) };
        assert_eq!(gdt[(USER_CODE_SELECTOR >> 3) as usize], USER_CODE64);
        assert_eq!(gdt[(USER_DATA_SELECTOR >> 3) as usize], USER_DATA);
    }

    #[test_case]
    fn the_tss_moved_clear_of_the_sysret_base() {
        // The TSS occupies two slots; it sits after the user pair so
        // `STAR[63:48]` never has to point into the middle of it.
        assert_eq!(TSS_SELECTOR >> 3, 5);
        assert!(TSS_SELECTOR > USER_CODE_SELECTOR & !3);
    }

    #[test_case]
    fn rsp0_round_trips_and_starts_unset() {
        // Zero until a ring-3-capable task is switched to — and that zero is
        // exactly what would triple-fault on the first trap from userspace, which
        // is why `set_kernel_stack` exists and is called per switch.
        let saved = kernel_stack();
        set_kernel_stack(0xdead_beef_0000);
        assert_eq!(kernel_stack(), 0xdead_beef_0000);
        set_kernel_stack(saved);
        assert_eq!(kernel_stack(), saved);
    }
}

/// Reload `CS` (via a far return), the data segment registers, and the task
/// register to this core's freshly-`lgdt`'d GDT. Shared by the BSP `init` and
/// the per-AP `init_ap`, since both need the identical selector layout.
///
/// # Safety
/// A GDT with the standard Chitti layout (null, code, data, TSS-low, TSS-high)
/// must already be loaded via `lgdt`.
unsafe fn reload_segments() {
    // Reload CS via a far return trick (there's no direct `mov cs`), then the
    // data segment registers and the task register.
    //
    // `code_sel` MUST be pushed before `tmp` is computed: `tmp` is a
    // `lateout` register, which tells LLVM it's free to alias it to any `in`
    // register (here, `code_sel`'s) once that input has been consumed.
    // Computing `tmp` first would clobber `code_sel` before its `push` runs,
    // pushing the wrong value twice and turning `retfq` into a jump to a bogus
    // code selector -- an instant #GP / triple fault.
    unsafe {
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
}

/// Set up this application processor's own GDT + TSS. Each AP needs its *own*
/// TSS (a TSS's busy bit means one TSS descriptor can't be `ltr`'d on two
/// cores), so the GDT, TSS, and double-fault IST stack are heap-allocated and
/// leaked (they live for the core's lifetime). Must run after `mm::init`.
pub fn init_ap() {
    use alloc::boxed::Box;

    // Per-CPU double-fault IST stack (leaked).
    let df_stack: &'static mut [u8] = Box::leak(alloc::vec![0u8; DOUBLE_FAULT_STACK_SIZE].into_boxed_slice());
    let df_top = df_stack.as_ptr() as u64 + DOUBLE_FAULT_STACK_SIZE as u64;

    let tss: &'static mut Tss = Box::leak(Box::new(Tss::new()));
    tss.ist[(DOUBLE_FAULT_IST_INDEX - 1) as usize] = df_top;
    let tss_addr = tss as *const Tss as u64;
    let (low, high) = tss_descriptor(tss_addr);

    // **The same layout as the BSP's, slot for slot.** `reload_segments` below
    // `ltr`s `TSS_SELECTOR`, and every selector constant in this module is an
    // index into whichever GDT is loaded — so an AP GDT that put the TSS
    // somewhere else would `ltr` past its own limit. That is what happened when
    // the user pair was inserted and this array was left at five slots: the BSP
    // came up, the first AP died on `ltr`, and the machine reset during SMP
    // bring-up with no panic and no output after `smp: Limine reports 4 cpu(s)`.
    let gdt: &'static mut [u64; 7] =
        Box::leak(Box::new([0, KERNEL_CODE64, KERNEL_DATA, USER_DATA, USER_CODE64, low, high]));

    // SAFETY: `gdt` is a valid, correctly-laid-out GDT that outlives this
    // core; loading it and reloading segments to its selectors is the AP
    // equivalent of what `init` does on the BSP.
    unsafe {
        let gdt_ptr = DescriptorTablePointer {
            limit: (size_of::<[u64; 7]>() - 1) as u16,
            base: gdt.as_ptr() as u64,
        };
        asm!("lgdt [{}]", in(reg) &gdt_ptr, options(readonly, nostack, preserves_flags));
        reload_segments();
    }
}
