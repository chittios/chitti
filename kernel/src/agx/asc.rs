//! Apple **ASC** (Apple System Coprocessor) mailbox transport — the low-level
//! FIFO the AGX GPU's control coprocessor (`gfx-asc`) speaks over. Port of
//! m1n1's `src/asc.c` (`asc_cpu_start`, `asc_send`, `asc_recv`), vendored at
//! `third_party/m1n1/src/asc.c`. One 64-bit + 32-bit message pair per FIFO slot.
//!
//! MMIO discipline (the aarch64 rule, CLAUDE.md): every register access is a
//! **single** `ldr`/`str`. The 32-bit control/CPU registers use
//! `read_volatile`/`write_volatile` (one `ldr w`/`str w`); the **64-bit** FIFO
//! data registers use inline-asm `ldr x`/`str x` so LLVM can't coalesce the pair
//! into an `ldp`/`stp` the hypervisor can't decode. The `dsb`/`dmb` fences
//! around FIFO data (m1n1's `dma_wmb`/`dma_rmb`) are load-bearing on silicon.

use core::arch::asm;
use core::ptr::{read_volatile, write_volatile};

// --- register offsets from the ASC CPU base (asc.c) ----------------------
// The mailbox FIFO window sits at cpu_base + 0x8000; m1n1 stores that as a
// second `base` and uses 0x1xx/0x8xx offsets. We fold the +0x8000 in here.
const CPU_CONTROL: usize = 0x44;
const CPU_CONTROL_START: u32 = 0x10; // bit 4

const MBOX: usize = 0x8000;
const A2I_CONTROL: usize = MBOX + 0x110;
const A2I_SEND0: usize = MBOX + 0x800;
const A2I_SEND1: usize = MBOX + 0x808;
const I2A_CONTROL: usize = MBOX + 0x114;
const I2A_RECV0: usize = MBOX + 0x830;
const I2A_RECV1: usize = MBOX + 0x838;

const CONTROL_FULL: u32 = 1 << 16; // A2I: outbox full — can't send
const CONTROL_EMPTY: u32 = 1 << 17; // I2A: inbox empty — nothing to receive
const CONTROL_ENABLE: u32 = 1 << 0; // mailbox enable (R_MBOX_CTRL.ENABLE)

/// One ASC mailbox message: a 64-bit payload + a 32-bit tag (the RTKit endpoint).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Message {
    pub msg0: u64,
    pub msg1: u32,
}

/// A handle to one ASC coprocessor at its (identity-mapped) CPU register base.
pub struct Asc {
    base: usize,
}

impl Asc {
    /// # Safety
    /// `base` must be the Device-mapped MMIO CPU base of an ASC (the GPU node's
    /// `reg[0]` "asc" window). Caller ensures the 1 GiB block is Device-mapped
    /// (`mmu::map_device_gib`) and holds exclusive access during bring-up.
    pub unsafe fn new(base: usize) -> Asc {
        Asc { base }
    }

    #[inline]
    fn r32(&self, off: usize) -> u32 {
        // SAFETY: single 32-bit MMIO read of a mapped ASC register.
        unsafe { read_volatile((self.base + off) as *const u32) }
    }
    #[inline]
    fn w32(&self, off: usize, v: u32) {
        // SAFETY: single 32-bit MMIO write of a mapped ASC register.
        unsafe { write_volatile((self.base + off) as *mut u32, v) }
    }
    #[inline]
    fn r64(&self, off: usize) -> u64 {
        let addr = self.base + off;
        let v: u64;
        // SAFETY: single `ldr x` of a mapped 64-bit FIFO register — inline asm so
        // the load is never paired into an `ldp` the hypervisor can't decode.
        unsafe { asm!("ldr {v}, [{a}]", v = out(reg) v, a = in(reg) addr, options(nostack, preserves_flags)) };
        v
    }
    #[inline]
    fn w64(&self, off: usize, v: u64) {
        let addr = self.base + off;
        // SAFETY: single `str x` of a mapped 64-bit FIFO register (see `r64`).
        unsafe { asm!("str {v}, [{a}]", v = in(reg) v, a = in(reg) addr, options(nostack, preserves_flags)) };
    }

    /// Start the coprocessor CPU (set CPU_CONTROL.START). Idempotent.
    pub fn cpu_start(&self) {
        self.w32(CPU_CONTROL, self.r32(CPU_CONTROL) | CPU_CONTROL_START);
    }
    /// Stop the coprocessor CPU (clear START).
    pub fn cpu_stop(&self) {
        self.w32(CPU_CONTROL, self.r32(CPU_CONTROL) & !CPU_CONTROL_START);
    }
    /// True once the coprocessor CPU is running (START latched).
    pub fn cpu_running(&self) -> bool {
        self.r32(CPU_CONTROL) & CPU_CONTROL_START != 0
    }

    /// Ensure both A2I/I2A mailboxes have ENABLE set (bit 0). Some firmware
    /// leaves them clear after reset; without it FULL can stick or sends drop.
    pub fn mbox_enable(&self) {
        let a2i = self.r32(A2I_CONTROL);
        if a2i & CONTROL_ENABLE == 0 {
            self.w32(A2I_CONTROL, a2i | CONTROL_ENABLE);
        }
        let i2a = self.r32(I2A_CONTROL);
        if i2a & CONTROL_ENABLE == 0 {
            self.w32(I2A_CONTROL, i2a | CONTROL_ENABLE);
        }
    }

    /// Snapshot of mailbox/CPU regs for ktrace diagnostics.
    pub fn diag(&self) -> (u32, u32, u32) {
        (self.r32(CPU_CONTROL), self.r32(A2I_CONTROL), self.r32(I2A_CONTROL))
    }

    /// True if there is a message waiting in the inbox (I2A not empty).
    pub fn can_recv(&self) -> bool {
        self.r32(I2A_CONTROL) & CONTROL_EMPTY == 0
    }
    /// True if the outbox can accept a message (A2I not full).
    pub fn can_send(&self) -> bool {
        self.r32(A2I_CONTROL) & CONTROL_FULL == 0
    }

    /// Non-blocking receive — `None` if the inbox is empty (asc.c:89-101).
    pub fn try_recv(&self) -> Option<Message> {
        if !self.can_recv() {
            return None;
        }
        let msg0 = self.r64(I2A_RECV0);
        let msg1 = self.r64(I2A_RECV1) as u32;
        dma_rmb();
        Some(Message { msg0, msg1 })
    }

    /// Send `msg`, waiting up to `timeout_ms` for the outbox to have space,
    /// pumping `pump` (returns true on Ctrl+C) between polls. `false` on
    /// timeout/abort (asc.c:118-131, cooperative).
    ///
    /// If A2I stays FULL for the whole timeout (IOP not draining — common when
    /// the coprocessor is asleep with a stale queue), still **force-writes** the
    /// slot once (proxyclient order) and returns true: some firmwares clear FULL
    /// only after the next push. Callers that need a reply will still time out
    /// if the IOP is truly dead.
    pub fn send(&self, msg: &Message, timeout_ms: u64, pump: &mut dyn FnMut() -> bool) -> bool {
        self.mbox_enable();
        let deadline = crate::arch::now_ms() + timeout_ms;
        let mut forced = false;
        while !self.can_send() {
            if crate::arch::now_ms() >= deadline || pump() {
                forced = true;
                break;
            }
            // Nudge the CPU if START cleared mid-wait.
            if !self.cpu_running() {
                self.cpu_start();
            }
            core::hint::spin_loop();
        }
        if forced {
            let (cpu, a2i, i2a) = self.diag();
            crate::ktrace::log_fmt(format_args!(
                "asc: A2I FULL timeout — force send (cpu={cpu:#x} a2i={a2i:#x} i2a={i2a:#x})"
            ));
        }
        dma_wmb();
        self.w64(A2I_SEND0, msg.msg0);
        self.w64(A2I_SEND1, msg.msg1 as u64);
        true
    }

    /// Block for one inbound message up to `timeout_ms`, pumping `pump` (returns
    /// true on Ctrl+C) between polls. `None` on timeout or abort. This is the
    /// cooperative form of m1n1's `asc_recv_timeout` — it never unbounded-spins
    /// and keeps the UI/clock/net alive during the handshake.
    pub fn recv_blocking(&self, timeout_ms: u64, pump: &mut dyn FnMut() -> bool) -> Option<Message> {
        let deadline = crate::arch::now_ms() + timeout_ms;
        loop {
            if let Some(m) = self.try_recv() {
                return Some(m);
            }
            if crate::arch::now_ms() >= deadline || pump() {
                return None;
            }
            core::hint::spin_loop();
        }
    }
}

/// Write barrier before publishing FIFO data (m1n1 `dma_wmb`).
#[inline]
fn dma_wmb() {
    // SAFETY: a data-synchronization barrier; no memory operands.
    unsafe { asm!("dsb st", options(nostack, preserves_flags)) };
}

/// Read barrier after consuming FIFO data (m1n1 `dma_rmb`).
#[inline]
fn dma_rmb() {
    // SAFETY: a data-memory barrier; no memory operands.
    unsafe { asm!("dmb ld", options(nostack, preserves_flags)) };
}
