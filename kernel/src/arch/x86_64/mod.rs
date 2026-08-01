//! x86_64-specific primitives: port I/O, GDT/TSS, IDT + exceptions, the
//! legacy PIC + PIT timer + keyboard IRQs, FPU/SSE init, and 4-level
//! paging.

pub mod ac97;
pub mod ahci;
pub mod apic;
pub mod disk;
pub mod sb16;
pub mod fpu;
pub mod fastcall;
pub mod gdt;
pub mod idt;
pub mod interrupts;
pub mod i8042;
pub mod keyboard;
pub mod nvme;
pub mod hpet;
pub mod paging;
pub mod pci;
pub mod pic;
pub mod pit;
pub mod suspend;
pub mod port;
pub mod rtc;
pub mod xhci;

/// Raw cycle counter (TSC) for entropy mixing (see `arch::cycle_count`).
pub fn cycle_count() -> u64 {
    let lo: u32;
    let hi: u32;
    // SAFETY: `rdtsc` reads the timestamp counter into edx:eax; no memory or
    // flag effects. Present on every x86-64 CPU.
    unsafe { core::arch::asm!("rdtsc", out("eax") lo, out("edx") hi, options(nomem, nostack, preserves_flags)) };
    (hi as u64) << 32 | lo as u64
}

/// A hardware random word via `RDRAND` when CPUID reports it, else 0 (see
/// `arch::hw_rand`). CPUID.01H:ECX bit 30 = RDRAND.
pub fn hw_rand() -> u64 {
    let ecx: u32;
    // SAFETY: CPUID leaf 1 is universally available; `rbx` is callee-saved by
    // LLVM so we swap it out around the instruction.
    unsafe {
        core::arch::asm!(
            "mov {tmp:r}, rbx",
            "cpuid",
            "mov rbx, {tmp:r}",
            tmp = out(reg) _,
            inout("eax") 1u32 => _,
            out("ecx") ecx,
            out("edx") _,
            options(nostack, preserves_flags),
        );
    }
    if ecx & (1 << 30) == 0 {
        return 0;
    }
    let v: u64;
    let ok: u8;
    // SAFETY: RDRAND is implemented per the CPUID check; CF=1 => a valid random
    // value was returned. A not-ready result (CF=0) yields 0; the caller mixes
    // multiple samples plus the TSC.
    unsafe {
        core::arch::asm!(
            "rdrand {v}",
            "setc {ok}",
            v = out(reg) v,
            ok = out(reg_byte) ok,
            options(nomem, nostack),
        );
    }
    if ok != 0 {
        v
    } else {
        0
    }
}

/// Halt the CPU until the next interrupt.
#[inline]
pub fn hlt() {
    // SAFETY: `hlt` has no memory-safety implications; it just stops
    // instruction execution until an interrupt arrives.
    unsafe { core::arch::asm!("hlt", options(nomem, nostack, preserves_flags)) }
}

/// Power the machine off.
///
/// **ACPI S5 first, emulator exit second.** This used to write only port `0xf4` —
/// QEMU's `isa-debug-exit` device — which does nothing at all on real hardware,
/// so `/poweroff` left the machine running with the fans on and the shell gone.
/// A real soft-off is a write of `SLP_TYPa | SLP_EN` to the FADT's `PM1a_CNT`
/// (see [`crate::acpi::s5_from_rsdp`]).
///
/// Order matters: the ACPI path is attempted first because it is the one that
/// works on a physical machine, and the `isa-debug-exit` write stays as the
/// fallback so `exit` still terminates QEMU (where the device exists but the
/// firmware's `_S5_` may not be reachable).
pub fn poweroff() -> ! {
    if let Some(s) = acpi_sleep_info() {
        // SAFETY: single 16-bit writes to the firmware-declared PM1 control
        // ports. If ACPI mode is off (legacy BIOS handoff), take ownership via
        // SMI_CMD first — on a UEFI boot SCI_EN is already set and this is
        // skipped.
        unsafe {
            if s.smi_cmd != 0 && port::inw(s.pm1a_cnt) & crate::acpi::SCI_EN == 0 {
                port::outb(s.smi_cmd as u16, s.acpi_enable);
                // Wait, bounded, for the firmware to hand ACPI over.
                for _ in 0..100_000 {
                    if port::inw(s.pm1a_cnt) & crate::acpi::SCI_EN != 0 {
                        break;
                    }
                }
            }
            let a = (s.slp_typa as u16) << 10 | crate::acpi::SLP_EN;
            port::outw(s.pm1a_cnt, a);
            if s.pm1b_cnt != 0 {
                let b = (s.slp_typb as u16) << 10 | crate::acpi::SLP_EN;
                port::outw(s.pm1b_cnt, b);
            }
        }
        // The transition is not instantaneous; give it a moment before falling
        // through to the emulator path.
        for _ in 0..1_000_000 {
            core::hint::spin_loop();
        }
    }
    // SAFETY: 0xf4 is QEMU's isa-debug-exit port; harmless on real hardware
    // (an unclaimed I/O port), and exits the emulator under xtask.
    unsafe { port::outl(0xf4, 0x10) };
    loop {
        hlt();
    }
}

/// The S5 parameters from the firmware ACPI tables, or `None` if ACPI is not
/// reachable / the DSDT has no decodable `\_S5_`.
fn acpi_sleep_info() -> Option<crate::acpi::SleepInfo> {
    let rsdp = rsdp_address()?;
    crate::acpi::s5_from_rsdp(rsdp, |phys| crate::mm::map_mmio(phys, 0x4_0000))
}

/// Start of the higher half — any address at or above this is already virtual.
const HIGHER_HALF: u64 = 0xffff_8000_0000_0000;

/// Make an address the CPU can actually read.
///
/// Limine's RSDP address is **physical** on newer protocol revisions and
/// HHDM-virtual on older ones, and the two cannot be distinguished by trying
/// both: dereferencing a raw physical address in the higher-half kernel is a page
/// fault, not a garbage read. (It faulted at `0xf52e0` and halted the boot.) They
/// *can* be distinguished by range — a virtual HHDM address is in the higher half,
/// a physical one never is.
fn readable(addr: u64) -> u64 {
    if addr >= HIGHER_HALF {
        addr
    } else {
        // Not `phys_to_virt`: Limine's HHDM covers usable RAM, and the RSDP lives
        // in firmware-reserved memory outside it, so the HHDM address is unmapped
        // too. Map the page explicitly.
        crate::mm::map_mmio(addr, 0x1000)
    }
}

/// Locate the ACPI RSDP on x86.
///
/// The bootloader's pointer first (mapped via [`readable`], then signature-checked
/// so a wrong guess is rejected rather than believed), else a scan of the legacy
/// `0xE0000..0x100000` BIOS window where a non-UEFI boot leaves it.
pub fn rsdp_address() -> Option<u64> {
    if let Some(r) = crate::RSDP_REQUEST.response() {
        let v = readable(r.address());
        if crate::acpi::find_rsdp(&[v]).is_some() {
            return Some(v);
        }
    }
    // Legacy scan: the RSDP is 16-byte aligned in the BIOS ROM area. Read through
    // the HHDM — the window is reserved physical memory, not kernel-mapped.
    let mut p = 0xE_0000u64;
    while p < 0x10_0000 {
        let v = crate::mm::map_mmio(p, 0x1000);
        if crate::acpi::find_rsdp(&[v]).is_some() {
            return Some(v);
        }
        p += 16;
    }
    None
}

/// Reboot the machine. Pulse the 8042 keyboard-controller reset line (port
/// `0x64` command `0xFE`) — the standard PC soft-reset path, works on real
/// hardware and most hypervisors. Under QEMU started with `-no-reboot` the
/// emulator exits instead of looping; either way `/restart` does not hang.
pub fn reboot() -> ! {
    // SAFETY: writing 0xFE to the 8042 status port is the architected
    // keyboard-controller CPU reset pulse; it does not return on success.
    unsafe {
        // Drain the input buffer so the controller accepts the command.
        for _ in 0..100_000 {
            if port::inb(0x64) & 0x02 == 0 {
                break;
            }
        }
        port::outb(0x64, 0xFE);
    }
    // Fallback: triple-fault via a zero-limit IDT load (if the 8042 is absent).
    // SAFETY: loading a null IDT then triggering an interrupt resets the CPU.
    unsafe {
        let null_idt: [u64; 2] = [0, 0];
        core::arch::asm!(
            "lidt [{0}]",
            "int3",
            in(reg) &null_idt,
            options(nostack, noreturn),
        );
    }
}
