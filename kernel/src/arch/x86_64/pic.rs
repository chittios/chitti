//! Legacy 8259 PIC. `CHITTI_OS_HANDOFF.md` allows "APIC (or PIC fallback)"
//! for Phase 1; the PIC is used here deliberately — it needs no ACPI/MADT
//! parsing to discover I/O-APIC redirection tables, which keeps the
//! highest-risk part of this phase (interrupts working at all, not
//! triple-faulting) small and easy to reason about. Revisiting for APIC
//! (needed anyway for Phase 7 SMP, where each core needs its own LAPIC
//! timer) is future work, not a Phase 1 requirement.

use super::port::{inb, outb};

const MASTER_COMMAND: u16 = 0x20;
const MASTER_DATA: u16 = 0x21;
const SLAVE_COMMAND: u16 = 0xa0;
const SLAVE_DATA: u16 = 0xa1;

/// Where the master PIC's IRQ0-7 land in the IDT, chosen to sit right
/// after the 32 CPU exception vectors.
pub const IRQ_BASE: u8 = 32;
pub const TIMER_VECTOR: u8 = IRQ_BASE; // IRQ0
pub const KEYBOARD_VECTOR: u8 = IRQ_BASE + 1; // IRQ1

const ICW1_INIT: u8 = 0x10;
const ICW1_ICW4: u8 = 0x01;
const ICW4_8086: u8 = 0x01;

/// Remap the PIC's IRQ0-15 to vectors 32-47 (their power-on default,
/// 8-15/0x08-0x0F, collides with CPU exception vectors) and mask
/// everything except the timer (IRQ0) and keyboard (IRQ1), the only two
/// lines Phase 1 handles.
pub fn init() {
    // SAFETY: standard 8259 PIC initialization command word sequence, run
    // once at boot before interrupts are enabled.
    unsafe {
        let master_mask = inb(MASTER_DATA);
        let slave_mask = inb(SLAVE_DATA);
        let _ = (master_mask, slave_mask); // original masks not needed; we set explicit ones below

        outb(MASTER_COMMAND, ICW1_INIT | ICW1_ICW4);
        io_wait();
        outb(SLAVE_COMMAND, ICW1_INIT | ICW1_ICW4);
        io_wait();
        outb(MASTER_DATA, IRQ_BASE); // ICW2: master vector offset
        io_wait();
        outb(SLAVE_DATA, IRQ_BASE + 8); // ICW2: slave vector offset
        io_wait();
        outb(MASTER_DATA, 1 << 2); // ICW3: slave attached to master's IRQ2
        io_wait();
        outb(SLAVE_DATA, 2); // ICW3: slave's cascade identity
        io_wait();
        outb(MASTER_DATA, ICW4_8086);
        io_wait();
        outb(SLAVE_DATA, ICW4_8086);
        io_wait();

        // Mask everything except IRQ0 (timer) and IRQ1 (keyboard).
        outb(MASTER_DATA, !0b0000_0011u8);
        outb(SLAVE_DATA, 0xff);
    }
    crate::ktrace::log("pic", "8259 PIC remapped to vectors 32-47, IRQ0/IRQ1 unmasked");
}

fn io_wait() {
    // SAFETY: port 0x80 is a POST-code scratch port conventionally used
    // as a ~1us delay; writing to it has no side effect we depend on.
    unsafe { outb(0x80, 0) };
}

/// Acknowledge an IRQ so the PIC delivers further interrupts on that line.
pub fn send_eoi(irq_line: u8) {
    const EOI: u8 = 0x20;
    // SAFETY: `irq_line` is always a valid IRQ number (0-15) supplied by
    // our own IDT handlers; the slave EOI is only needed for lines >= 8.
    unsafe {
        if irq_line >= 8 {
            outb(SLAVE_COMMAND, EOI);
        }
        outb(MASTER_COMMAND, EOI);
    }
}
