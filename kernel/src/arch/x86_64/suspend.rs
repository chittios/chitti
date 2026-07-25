//! **ACPI S3 suspend-to-RAM on x86**, including the part that makes it hard: coming
//! back.
//!
//! Sleeping is one register write. Resuming is not. Firmware wakes the CPU as though it
//! had just been reset — **real mode**, paging off, no GDT, no IDT — and jumps to
//! whatever physical address the OS left in the FACS waking vector. So resuming means a
//! trampoline that walks the CPU back up through protected mode into long mode before a
//! single instruction of Rust can run.
//!
//! ## The round trip
//!
//! 1. [`prepare`] takes a page **below 1 MiB** (firmware enters real mode, where
//!    `CS:IP` reaches no higher), copies the trampoline into it, and builds a temporary
//!    page table.
//! 2. [`suspend`] saves the state firmware will destroy, publishes the trampoline's
//!    physical address in the FACS, flushes the caches, and writes `SLP_TYP | SLP_EN`.
//! 3. Power goes away from everything but RAM. Later, something wakes the machine.
//! 4. The trampoline runs: real mode → protected mode → long mode → [`resume_entry`],
//!    which is ordinary Rust on the kernel's own stack.
//! 5. [`resume_entry`] puts back the rest of the state and returns into [`suspend`]'s
//!    caller.
//!
//! ## Three decisions that keep the resume path honest
//!
//! **Nothing is patched into the code.** The trampoline learns its own load address
//! from `CS` (firmware sets `CS = page >> 4`) and derives every other address from it
//! with displacements the assembler computed. Setup writes *data* fields only, and the
//! code is copied byte-for-byte — self-modifying setup is the classic way to get a
//! resume path that works on one machine and triple-faults on the next.
//!
//! **Offsets are never written down twice.** Every offset the assembly uses is
//! `label - chitti_s3_wake_start`, and Rust recovers the same values from the exported
//! labels rather than from a parallel list of constants. The field offsets *within* the
//! saved-state block are the one place two definitions must agree, so a test pins them
//! with `offset_of!`.
//!
//! **Paging comes back on with a temporary table, not the kernel's.** The kernel's
//! tables are not guaranteed to identity-map the low page the trampoline is executing
//! from, and enabling paging would then unmap the instruction pointer mid-stream — a
//! fault with no IDT yet, which presents as a machine that wakes and instantly dies.
//! The temporary table is the kernel's PML4 with entry 0 replaced by a 4 GiB identity
//! mapping, so the trampoline's own address *and* every kernel address are valid at
//! once; the real `CR3` goes back later, from Rust, running in the higher half where the
//! switch is safe.
//!
//! **Unverified on real hardware**, but QEMU implements S3 and a `system_wakeup` monitor
//! command, so unlike the battery this path can be made to actually happen.

use super::port;
use crate::acpi;

/// Firmware enters real mode, so the trampoline has to be reachable by `CS:IP`.
const LOW_MEMORY_LIMIT: u64 = 1 << 20;

/// `CR3` is reloaded by a 32-bit `mov` while still in protected mode, so the temporary
/// page tables have to live where a 32-bit register can name them.
const CR3_LIMIT: u64 = 1u64 << 32;

/// State firmware destroys across S3.
///
/// `#[repr(C)]` and field order are load-bearing: the assembly reads the first three
/// fields by displacement. `state_offsets_match_the_assembly` pins them.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct SavedState {
    /// Temporary page-table root the trampoline enables paging with. **Offset 0.**
    pub temp_cr3: u64,
    /// Virtual address of [`resume_entry`]. **Offset 8.**
    pub resume_entry: u64,
    /// Kernel stack pointer, which the trampoline installs before entering Rust.
    /// **Offset 16.**
    pub rsp: u64,
    /// The kernel's real `CR3`, restored from Rust once in the higher half.
    pub kernel_cr3: u64,
    pub cr0: u64,
    pub cr4: u64,
    pub efer: u64,
    /// Where [`resume_entry`] hands control back to — inside [`suspend`], just past the
    /// instruction that wrote `SLP_EN`.
    pub return_rip: u64,
    pub rbx: u64,
    pub rbp: u64,
    pub r12: u64,
    pub r13: u64,
    pub r14: u64,
    pub r15: u64,
    /// `GDTR`/`IDTR` as `lgdt`/`lidt` want them: 2-byte limit then 8-byte base.
    pub gdtr: [u8; 16],
    pub idtr: [u8; 16],
    /// `XCR0`, when `CR4.OSXSAVE` says the extended state is enabled. Restored right
    /// after `CR4`, because an `xsetbv` before `OSXSAVE` is set faults.
    pub xcr0: u64,
    /// The task register's selector. Firmware resets `TR`, and an interrupt that needs
    /// an IST stack with no TSS loaded is an unrecoverable fault.
    pub tr: u16,
    pub _pad: [u16; 3],
    /// FACS hardware signature seen before sleeping. ACPI requires an OS to refuse to
    /// resume when this changed: the machine is no longer the one that saved state.
    pub hardware_signature: u32,
    /// Set by [`resume_entry`], so [`suspend`] can tell "came back from S3" from "the
    /// sleep write returned without sleeping".
    pub resumed: u32,
}

// Offsets the assembly uses. Mirrored, not derived — hence the test.
const S_TEMP_CR3: usize = 0;
const S_RESUME_ENTRY: usize = 8;
const S_RSP: usize = 16;

core::arch::global_asm!(
    r#"
.section .text
.balign 16
.att_syntax
.global chitti_s3_wake_start
.global chitti_s3_wake_end
.global chitti_s3_gdt
.global chitti_s3_gdt_desc
.global chitti_s3_state
.set S_TEMP_CR3,     0
.set S_RESUME_ENTRY, 8
.set S_RSP,          16

.code16
chitti_s3_wake_start:
    cli
    cld
    // CS is the only thing that says where we are: firmware entered at CS = page >> 4,
    // IP = 0. Everything below is derived from it rather than patched in, which is what
    // lets these exact bytes work on any machine.
    movw    %cs, %ax
    movw    %ax, %ds
    movw    %ax, %es
    movw    %ax, %ss
    movw    $(chitti_s3_stack_top - chitti_s3_wake_start), %sp

    // ebx = linear base of this page = cs << 4. Used as the base for every later
    // reference, once segmentation is out of the way.
    xorl    %ebx, %ebx
    movw    %cs, %bx
    shll    $4, %ebx

    // The GDT pseudo-descriptor needs an absolute linear base; its limit was written by
    // `prepare`. Note the memory operands here are *segment* offsets, because ds is the
    // page — %ebx must not appear in them or the address would be counted twice.
    leal    (chitti_s3_gdt - chitti_s3_wake_start)(%ebx), %eax
    movl    %eax, chitti_s3_gdt_desc - chitti_s3_wake_start + 2
    lgdtl   chitti_s3_gdt_desc - chitti_s3_wake_start

    movl    %cr0, %eax
    orl     $1, %eax
    movl    %eax, %cr0

    // A far *return* rather than a far jump: its target is a value we compute at
    // runtime, where a far jump would need it encoded in the instruction — which is
    // exactly the patching this design avoids.
    leal    (chitti_s3_pm32 - chitti_s3_wake_start)(%ebx), %eax
    pushl   $8
    pushl   %eax
    lretl

.code32
chitti_s3_pm32:
    movw    $16, %ax
    movw    %ax, %ds
    movw    %ax, %es
    movw    %ax, %ss
    movw    %ax, %fs
    movw    %ax, %gs

    // PAE first; long mode requires it.
    movl    %cr4, %eax
    orl     $(1 << 5), %eax
    movl    %eax, %cr4

    // The temporary table: the kernel's PML4 with entry 0 identity-mapping low memory,
    // so turning paging on leaves *this* code mapped as well as the kernel.
    movl    (chitti_s3_state - chitti_s3_wake_start + S_TEMP_CR3)(%ebx), %eax
    movl    %eax, %cr3

    // EFER.LME *and* EFER.NXE. NXE is not optional here: the kernel's page tables mark
    // data pages no-execute, and with NXE clear bit 63 of a PTE is a *reserved* bit, so
    // the first touch of the kernel stack would take a reserved-bit page fault with no
    // IDT installed. Enabling long mode without it is a machine that wakes and dies on
    // its first memory access.
    movl    $0xC0000080, %ecx
    rdmsr
    orl     $((1 << 8) | (1 << 11)), %eax
    wrmsr

    // Paging on — long mode is active from here.
    movl    %cr0, %eax
    orl     $0x80000001, %eax
    movl    %eax, %cr0

    leal    (chitti_s3_lm64 - chitti_s3_wake_start)(%ebx), %eax
    pushl   $24
    pushl   %eax
    lretl

.code64
chitti_s3_lm64:
    // Install the kernel's stack before entering Rust: it is in RAM, it survived, and
    // the temporary table maps it. The trampoline's own scratch stack is 256 bytes and
    // nowhere near enough for compiled code.
    movq    (chitti_s3_state - chitti_s3_wake_start + S_RSP)(%rbx), %rsp
    movq    (chitti_s3_state - chitti_s3_wake_start + S_RESUME_ENTRY)(%rbx), %rax
    // The identity address of the page, so Rust can read the saved state before it
    // changes CR3.
    movq    %rbx, %rdi
    jmp     *%rax

.balign 16
chitti_s3_gdt:
    .quad 0
    .quad 0x00cf9b000000ffff    // 32-bit code, base 0, limit 4 GiB
    .quad 0x00cf93000000ffff    // data
    .quad 0x00af9b000000ffff    // 64-bit code (L bit, not D)
chitti_s3_gdt_desc:
    .word 0
    .long 0
.balign 16
chitti_s3_state:
    .space 224
.balign 16
    .space 256
chitti_s3_stack_top:
chitti_s3_wake_end:
.intel_syntax noprefix
"#
);

extern "C" {
    fn chitti_s3_wake_start();
    fn chitti_s3_wake_end();
    fn chitti_s3_gdt();
    fn chitti_s3_gdt_desc();
    fn chitti_s3_state();
}

/// The assembled trampoline, as bytes to copy.
fn blob() -> &'static [u8] {
    let start = chitti_s3_wake_start as usize;
    let end = chitti_s3_wake_end as usize;
    // SAFETY: both symbols bound a region of this image's `.text`, and `end` follows
    // `start`; the range is read-only kernel code.
    unsafe { core::slice::from_raw_parts(start as *const u8, end - start) }
}

/// Offset of an exported label within the blob.
///
/// Recovered from the linker rather than written down a second time: a constant that
/// disagreed with the assembly would put the GDT base, or the saved state, somewhere the
/// trampoline does not look — and the symptom is a machine that never wakes.
fn offset_of_label(label: unsafe extern "C" fn()) -> usize {
    label as usize - chitti_s3_wake_start as usize
}

/// Where the trampoline was placed.
#[derive(Debug, Clone, Copy)]
struct Trampoline {
    phys: u64,
    virt: u64,
}

static mut TRAMPOLINE: Option<Trampoline> = None;

/// Build the temporary page table the trampoline enables paging with.
///
/// The kernel's PML4 with entry 0 replaced by a 4 GiB identity mapping. Entry 0 covers
/// the low half of the address space, where the kernel has nothing — the kernel lives in
/// the higher half, whose entries are copied across intact, which is what makes it safe
/// to jump straight to a kernel-virtual address once paging is on.
fn build_temp_tables(kernel_cr3: u64) -> Option<u64> {
    let (pml4_phys, pml4_virt) = crate::mm::alloc_dma_bounded(4096, CR3_LIMIT, 0)?;
    let (pdpt_phys, pdpt_virt) = crate::mm::alloc_dma_bounded(4096, CR3_LIMIT, 0)?;

    let src = super::paging::phys_to_virt(kernel_cr3) as *const u64;
    let dst = pml4_virt as *mut u64;
    // SAFETY: both are 512-entry page tables; `src` is the live PML4 (read-only here)
    // and `dst` a freshly-allocated, zeroed frame we own.
    unsafe {
        core::ptr::copy_nonoverlapping(src, dst, 512);
    }

    // Four 1 GiB identity pages: present | writable | huge.
    const PRESENT_WRITE: u64 = 0b11;
    const HUGE: u64 = 1 << 7;
    let pdpt = pdpt_virt as *mut u64;
    for gib in 0..4u64 {
        // SAFETY: `pdpt` is a frame we own; entries 0..4 are in bounds.
        unsafe { pdpt.add(gib as usize).write(gib * (1 << 30) | PRESENT_WRITE | HUGE) };
    }
    // SAFETY: as above; entry 0 of the copy is ours to replace.
    unsafe { dst.write(pdpt_phys | PRESENT_WRITE) };
    Some(pml4_phys)
}

/// Place the trampoline and its page table. Idempotent.
pub fn prepare() -> Result<(), &'static str> {
    // SAFETY: boot/shell-path single-threaded access to a `Copy` option.
    if unsafe { TRAMPOLINE.is_some() } {
        return Ok(());
    }
    let code = blob();
    if code.len() > 4096 {
        return Err("resume trampoline does not fit in one page");
    }
    let (phys, virt) = crate::mm::alloc_dma_bounded(4096, LOW_MEMORY_LIMIT, 0)
        .ok_or("no free page below 1 MiB for the resume trampoline")?;
    // SAFETY: `virt` maps a whole owned frame and `code` is at most a page.
    unsafe { core::ptr::copy_nonoverlapping(code.as_ptr(), virt as *mut u8, code.len()) };

    // The GDT descriptor's limit; its base is filled in by the trampoline itself, which
    // is the only code that knows where the page ended up in real-mode terms.
    let gdt_desc = virt + offset_of_label(chitti_s3_gdt_desc) as u64;
    let limit = (4 * core::mem::size_of::<u64>() - 1) as u16;
    // SAFETY: `gdt_desc` is inside the frame just allocated and copied into.
    unsafe { (gdt_desc as *mut u16).write_unaligned(limit) };

    let kernel_cr3 = super::paging::active_cr3();
    let temp_cr3 = build_temp_tables(kernel_cr3).ok_or("could not build the resume page table")?;
    let st = state_ptr(virt);
    // SAFETY: `st` points at the state block inside the frame.
    unsafe {
        (*st).temp_cr3 = temp_cr3;
        (*st).resume_entry = resume_entry as usize as u64;
        (*st).kernel_cr3 = kernel_cr3;
    }
    // SAFETY: single-threaded initialisation.
    unsafe { TRAMPOLINE = Some(Trampoline { phys, virt }) };
    crate::ktrace::log_fmt(format_args!(
        "s3: trampoline at {phys:#x} ({} bytes), temp CR3 {temp_cr3:#x}",
        code.len()
    ));
    Ok(())
}

/// The saved-state block inside the trampoline page.
fn state_ptr(page_virt: u64) -> *mut SavedState {
    (page_virt + offset_of_label(chitti_s3_state) as u64) as *mut SavedState
}

/// Read a control register or MSR the resume path has to put back.
fn read_cr0() -> u64 {
    let v: u64;
    // SAFETY: reading CR0 has no side effects.
    unsafe { core::arch::asm!("mov {}, cr0", out(reg) v, options(nomem, nostack, preserves_flags)) };
    v
}

fn read_cr4() -> u64 {
    let v: u64;
    // SAFETY: reading CR4 has no side effects.
    unsafe { core::arch::asm!("mov {}, cr4", out(reg) v, options(nomem, nostack, preserves_flags)) };
    v
}

fn read_efer() -> u64 {
    let (lo, hi): (u32, u32);
    // SAFETY: `rdmsr` of EFER (0xC000_0080) has no side effects; EFER always exists on
    // a CPU already running in long mode.
    unsafe {
        core::arch::asm!(
            "rdmsr",
            in("ecx") 0xC000_0080u32,
            out("eax") lo,
            out("edx") hi,
            options(nomem, nostack, preserves_flags),
        )
    };
    (hi as u64) << 32 | lo as u64
}

/// `CR4.OSXSAVE` — whether `XCR0` exists to be read.
const CR4_OSXSAVE: u64 = 1 << 18;

fn read_xcr0() -> u64 {
    let (lo, hi): (u32, u32);
    // SAFETY: only called when `CR4.OSXSAVE` is set, which is what makes `xgetbv`
    // legal; it has no side effects.
    unsafe {
        core::arch::asm!(
            "xgetbv",
            in("ecx") 0u32,
            out("eax") lo,
            out("edx") hi,
            options(nomem, nostack, preserves_flags),
        )
    };
    (hi as u64) << 32 | lo as u64
}

fn read_tr() -> u16 {
    let v: u16;
    // SAFETY: `str` reads the task register; no side effects.
    unsafe { core::arch::asm!("str {0:x}", out(reg) v, options(nomem, nostack, preserves_flags)) };
    v
}

fn read_gdtr() -> [u8; 16] {
    let mut out = [0u8; 16];
    // SAFETY: `sgdt` writes 10 bytes at the destination; the buffer is 16.
    unsafe { core::arch::asm!("sgdt [{}]", in(reg) out.as_mut_ptr(), options(nostack, preserves_flags)) };
    out
}

fn read_idtr() -> [u8; 16] {
    let mut out = [0u8; 16];
    // SAFETY: `sidt` writes 10 bytes at the destination; the buffer is 16.
    unsafe { core::arch::asm!("sidt [{}]", in(reg) out.as_mut_ptr(), options(nostack, preserves_flags)) };
    out
}

/// Publish `phys` as the address firmware jumps to on wake.
///
/// The 32-bit vector is the one to use: it is what firmware has always implemented, and
/// what the real-mode trampoline is for. The 64-bit vector is zeroed rather than left
/// alone, because ACPI says a non-zero value there takes precedence — a stale one from
/// a previous OS would send the wake somewhere else entirely.
fn set_waking_vector(facs: &acpi::FacsInfo, phys: u64) -> Result<(), &'static str> {
    if phys >= (1 << 32) {
        return Err("trampoline is above 4 GiB, which the 32-bit waking vector cannot name");
    }
    let va = crate::mm::map_mmio(facs.addr, acpi::FACS_MIN_LEN);
    if va == 0 {
        return Err("could not map the FACS");
    }
    // SAFETY: `va` maps at least `FACS_MIN_LEN` bytes of the firmware's FACS, and both
    // offsets are inside that. Writing the waking vector is the defined interface.
    unsafe {
        ((va + acpi::FACS_FIRMWARE_WAKING_VECTOR as u64) as *mut u32).write_volatile(phys as u32);
        if facs.has_extended_waking_vector() {
            ((va + acpi::FACS_X_FIRMWARE_WAKING_VECTOR as u64) as *mut u64).write_volatile(0);
        }
    }
    Ok(())
}

/// Suspend to RAM. Returns once the machine has resumed.
///
/// `Err` means the transition was not attempted. If it *is* attempted and the machine
/// fails to come back, this does not return at all — which is why every caller goes
/// through [`crate::power::plan`] first.
pub fn suspend(sleep: &acpi::SleepInfo, facs: &acpi::FacsInfo) -> Result<(), &'static str> {
    if sleep.state != 3 {
        return Err("not an S3 sleep descriptor");
    }
    prepare()?;
    // SAFETY: set by `prepare` just above.
    let t = unsafe { TRAMPOLINE }.ok_or("trampoline missing")?;
    set_waking_vector(facs, t.phys)?;

    let st = state_ptr(t.virt);
    // SAFETY: `st` is the state block inside the trampoline page.
    unsafe {
        (*st).cr0 = read_cr0();
        (*st).cr4 = read_cr4();
        (*st).efer = read_efer();
        (*st).gdtr = read_gdtr();
        (*st).idtr = read_idtr();
        (*st).kernel_cr3 = super::paging::active_cr3();
        (*st).tr = read_tr();
        (*st).xcr0 = if (*st).cr4 & CR4_OSXSAVE != 0 { read_xcr0() } else { 0 };
        (*st).hardware_signature = facs.hardware_signature;
        (*st).resumed = 0;
    }

    // ACPI mode must be on, or the sleep-enable write lands in a register firmware
    // still owns.
    // SAFETY: reads/writes of the FADT-declared PM1 and SMI command ports.
    unsafe {
        if sleep.smi_cmd != 0 && port::inw(sleep.pm1a_cnt) & acpi::SCI_EN == 0 {
            port::outb(sleep.smi_cmd as u16, sleep.acpi_enable);
            for _ in 0..1_000_000 {
                if port::inw(sleep.pm1a_cnt) & acpi::SCI_EN != 0 {
                    break;
                }
                core::hint::spin_loop();
            }
        }
    }

    let val_a = (sleep.slp_typa as u16) << 10 | acpi::SLP_EN;
    let val_b = (sleep.slp_typb as u16) << 10 | acpi::SLP_EN;
    crate::ktrace::log_fmt(format_args!(
        "s3: entering S3 (PM1a_CNT {:#06x} <- {val_a:#06x}), waking vector {:#x}",
        sleep.pm1a_cnt, t.phys
    ));

    // Everything below the boundary has to be in RAM before the caches lose power.
    // SAFETY: `wbinvd` writes back and invalidates the caches; always sound.
    unsafe { core::arch::asm!("wbinvd", options(nostack, preserves_flags)) };

    // SAFETY: saves the callee-saved registers, the stack pointer and a resume address
    // into the state block, then performs the sleep write. On resume, `resume_entry`
    // restores these and jumps to the `3:` label, so control returns here as if the
    // block had simply run to completion. `out` writes go to the FADT-declared PM1
    // control ports.
    unsafe {
        core::arch::asm!(
            "mov [{s} + {o_rbx}], rbx",
            "mov [{s} + {o_rbp}], rbp",
            "mov [{s} + {o_r12}], r12",
            "mov [{s} + {o_r13}], r13",
            "mov [{s} + {o_r14}], r14",
            "mov [{s} + {o_r15}], r15",
            "mov [{s} + {o_rsp}], rsp",
            "lea rax, [rip + 3f]",
            "mov [{s} + {o_rip}], rax",
            // Sleep. On a machine that honours it, the `out` never completes visibly.
            "mov dx, {pm1a:x}",
            "mov ax, {va:x}",
            "out dx, ax",
            "cmp {pm1b:x}, 0",
            "je 2f",
            "mov dx, {pm1b:x}",
            "mov ax, {vb:x}",
            "out dx, ax",
            // Wait to actually go under. A sleep transition is not instantaneous, and
            // executing on past it would run the resume path's caller with the machine
            // half asleep.
            "2:",
            "hlt",
            "jmp 2b",
            "3:",
            s = in(reg) st,
            pm1a = in(reg) sleep.pm1a_cnt,
            va = in(reg) val_a,
            pm1b = in(reg) sleep.pm1b_cnt,
            vb = in(reg) val_b,
            o_rbx = const core::mem::offset_of!(SavedState, rbx),
            o_rbp = const core::mem::offset_of!(SavedState, rbp),
            o_r12 = const core::mem::offset_of!(SavedState, r12),
            o_r13 = const core::mem::offset_of!(SavedState, r13),
            o_r14 = const core::mem::offset_of!(SavedState, r14),
            o_r15 = const core::mem::offset_of!(SavedState, r15),
            o_rsp = const core::mem::offset_of!(SavedState, rsp),
            o_rip = const core::mem::offset_of!(SavedState, return_rip),
            out("rax") _,
            out("rdx") _,
        );
    }

    // SAFETY: as above.
    let resumed = unsafe { (*st).resumed } != 0;
    if resumed {
        crate::ktrace::log("s3", "resumed from S3");
        Ok(())
    } else {
        // The write returned and we are still running: firmware declined the
        // transition. Not an error we can distinguish further, but definitely not a
        // resume.
        Err("firmware did not enter S3")
    }
}

/// Where the trampoline lands. **Not** a normal function: it is entered with the
/// kernel's stack installed but the temporary page table active, and it never returns
/// to its caller — it jumps back into [`suspend`].
///
/// `page_phys` is the trampoline page's identity address, which is how the saved state
/// is reachable *before* `CR3` goes back to the kernel's.
#[no_mangle]
extern "C" fn resume_entry(page_phys: u64) -> ! {
    // Copy the state onto the stack first: after `CR3` is restored the identity mapping
    // may be gone, and everything below still needs these values.
    // SAFETY: `page_phys` is the trampoline page, identity-mapped by the temporary
    // table that is still active.
    let st = unsafe { *state_ptr(page_phys) };
    // Record the fact of the resume in the page itself, while the identity mapping is
    // still valid. `suspend` reads it to tell "came back from S3" from "the sleep write
    // returned and nothing happened", which are otherwise the same control flow.
    // SAFETY: as above.
    unsafe { (*state_ptr(page_phys)).resumed = 1 };

    // SAFETY: restores the kernel's paging, descriptor tables, task register, segment
    // selectors, callee-saved registers and stack, then jumps to the address `suspend`
    // recorded. Each step is the inverse of one taken there.
    //
    // Every value is read out of the saved-state block through one pointer rather than
    // being passed in a register each: thirteen register operands is more than the
    // allocator has, and a displacement off a single base is what the block was laid out
    // for anyway.
    unsafe {
        core::arch::asm!(
            // Order is not arbitrary. EFER first, because it carries NXE and SCE and the
            // page tables below depend on the former. Then CR4 — which is what re-enables
            // SSE (`OSFXSR`), without which the first floating-point instruction in
            // compiled Rust faults — and only then XCR0, since `xsetbv` before
            // `CR4.OSXSAVE` is itself a fault. CR3 next, and CR0 last: it carries WP and
            // the cache bits, and nothing above needs them.
            "mov ecx, 0xC0000080",
            "mov eax, [{s} + {o_efer}]",
            "mov edx, [{s} + {o_efer} + 4]",
            "wrmsr",
            "mov rax, [{s} + {o_cr4}]",
            "mov cr4, rax",
            "test rax, {osxsave}",
            "jz 3f",
            "xor ecx, ecx",
            "mov eax, [{s} + {o_xcr0}]",
            "mov edx, [{s} + {o_xcr0} + 4]",
            "xsetbv",
            "3:",
            "mov rax, [{s} + {o_cr3}]",
            "mov cr3, rax",
            "mov rax, [{s} + {o_cr0}]",
            "mov cr0, rax",
            "lgdt [{s} + {o_gdtr}]",
            "lidt [{s} + {o_idtr}]",
            // Firmware reset TR. An interrupt that needs an IST stack with no TSS loaded
            // is unrecoverable, so this goes back before interrupts can happen.
            "mov ax, [{s} + {o_tr}]",
            "ltr ax",
            // The far return is what reloads CS: the selector the trampoline left there
            // indexes the *kernel's* GDT now, where it means something else entirely.
            "lea rax, [rip + 4f]",
            "push {code_sel}",
            "push rax",
            "retfq",
            "4:",
            "mov ax, {data_sel}",
            "mov ds, ax",
            "mov es, ax",
            "mov ss, ax",
            "mov fs, ax",
            "mov gs, ax",
            // The return address goes into a register before RSP moves, so the jump does
            // not depend on the old stack still being addressable.
            "mov rax, [{s} + {o_rip}]",
            "mov rbx, [{s} + {o_rbx}]",
            "mov rbp, [{s} + {o_rbp}]",
            "mov r12, [{s} + {o_r12}]",
            "mov r13, [{s} + {o_r13}]",
            "mov r14, [{s} + {o_r14}]",
            "mov r15, [{s} + {o_r15}]",
            "mov rsp, [{s} + {o_rsp}]",
            "jmp rax",
            s = in(reg) core::ptr::addr_of!(st),
            osxsave = const CR4_OSXSAVE,
            code_sel = const KERNEL_CODE_SELECTOR,
            data_sel = const KERNEL_DATA_SELECTOR,
            o_efer = const core::mem::offset_of!(SavedState, efer),
            o_cr4 = const core::mem::offset_of!(SavedState, cr4),
            o_cr3 = const core::mem::offset_of!(SavedState, kernel_cr3),
            o_cr0 = const core::mem::offset_of!(SavedState, cr0),
            o_xcr0 = const core::mem::offset_of!(SavedState, xcr0),
            o_tr = const core::mem::offset_of!(SavedState, tr),
            o_gdtr = const core::mem::offset_of!(SavedState, gdtr),
            o_idtr = const core::mem::offset_of!(SavedState, idtr),
            o_rip = const core::mem::offset_of!(SavedState, return_rip),
            o_rbx = const core::mem::offset_of!(SavedState, rbx),
            o_rbp = const core::mem::offset_of!(SavedState, rbp),
            o_r12 = const core::mem::offset_of!(SavedState, r12),
            o_r13 = const core::mem::offset_of!(SavedState, r13),
            o_r14 = const core::mem::offset_of!(SavedState, r14),
            o_r15 = const core::mem::offset_of!(SavedState, r15),
            o_rsp = const core::mem::offset_of!(SavedState, rsp),
            options(noreturn),
        );
    }
}

/// Kernel code/data selectors, matching [`super::gdt`]'s table layout.
const KERNEL_CODE_SELECTOR: u64 = 0x08;
const KERNEL_DATA_SELECTOR: u64 = 0x10;

#[cfg(test)]
mod tests {
    use super::*;

    #[test_case]
    fn state_offsets_match_the_assembly() {
        // The one place two definitions have to agree. A mismatch makes the trampoline
        // load a page-table root from the wrong field, which is a triple fault during
        // resume with no diagnostic at all.
        assert_eq!(core::mem::offset_of!(SavedState, temp_cr3), S_TEMP_CR3);
        assert_eq!(core::mem::offset_of!(SavedState, resume_entry), S_RESUME_ENTRY);
        assert_eq!(core::mem::offset_of!(SavedState, rsp), S_RSP);
    }

    #[test_case]
    fn the_state_block_fits_the_space_reserved_for_it_in_the_blob() {
        // The assembly reserves a fixed `.space` for the state. If the struct outgrows
        // it, writing the state would run into the trampoline's scratch stack.
        assert!(
            core::mem::size_of::<SavedState>() <= 224,
            "SavedState is {} bytes; the blob reserves 224",
            core::mem::size_of::<SavedState>()
        );
    }

    #[test_case]
    fn the_blob_fits_in_one_low_page_and_its_labels_are_ordered() {
        // Everything has to fit below 1 MiB in a single page, and the data labels have
        // to follow the code rather than land inside it.
        let len = blob().len();
        assert!(len > 0 && len <= 4096, "trampoline is {len} bytes");
        let gdt = offset_of_label(chitti_s3_gdt);
        let desc = offset_of_label(chitti_s3_gdt_desc);
        let state = offset_of_label(chitti_s3_state);
        assert!(gdt < desc && desc < state, "{gdt} {desc} {state}");
        assert!(state + core::mem::size_of::<SavedState>() <= len);
        assert_eq!(gdt % 16, 0, "the GDT must be aligned");
    }

    #[test_case]
    fn the_wake_gdt_descriptors_are_the_modes_the_trampoline_switches_through() {
        // A wrong descriptor here is a triple fault during resume, so the bits are
        // checked rather than trusted: 32-bit code needs D set and L clear, 64-bit code
        // needs L set and D clear.
        let gdt_at = chitti_s3_gdt as usize;
        // SAFETY: reading four u64s of this image's `.text` at an exported label.
        let g = unsafe { core::slice::from_raw_parts(gdt_at as *const u64, 4) };
        assert_eq!(g[0], 0, "entry 0 must be null");
        let d_bit = 1u64 << 54;
        let l_bit = 1u64 << 53;
        assert!(g[1] & d_bit != 0 && g[1] & l_bit == 0, "32-bit code");
        assert!(g[2] & d_bit != 0, "data");
        assert!(g[3] & l_bit != 0 && g[3] & d_bit == 0, "64-bit code");
        // Present, DPL 0, code/data type for all three.
        for e in &g[1..4] {
            assert!(e & (1 << 47) != 0, "present");
            assert!(e & (3 << 45) == 0, "DPL 0");
        }
    }
}
