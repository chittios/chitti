//! **xHCI USB host controller** + a **USB HID boot-keyboard** driver, so a real
//! USB keyboard (on real hardware, or QEMU's `-device qemu-xhci -device
//! usb-kbd`) can drive the shell — the modern counterpart to PS/2. This is the
//! **arch-neutral core**; the per-arch wrappers (`arch::x86_64::xhci`,
//! `arch::aarch64::xhci`) discover the controller, map its MMIO, and hand this
//! core the register base + a DMA allocator, then reuse it verbatim.
//!
//! Scope: a polled, single-device bring-up sufficient for a boot-protocol
//! keyboard — not a general USB stack. Stages:
//!   C1 controller bring-up: reset, set up the DCBAA + command ring + event
//!      ring, start the controller (from a wrapper-supplied MMIO base).
//!   C2 enumerate: detect the port, reset it, enable a device slot, address it.
//!   C3 configure: read descriptors, pick the HID boot-keyboard interface,
//!      set the configuration, arm its interrupt IN endpoint.
//!   C4 input: poll the interrupt endpoint for 8-byte boot reports, map USB HID
//!      usage codes -> ASCII, and push them into the console ring.
//!
//! DMA memory comes from the wrapper's `Alloc` (returns `(physical, virtual)`):
//! the device is handed the physical address, the CPU uses the virtual one
//! (identical on the aarch64 identity map; HHDM on x86).
// Staged bring-up: some fields/helpers are wired in later stages (C2-C4).
#![allow(dead_code)]

use core::ptr::{read_volatile, write_volatile};

/// A page-aligned DMA allocator: `bytes -> (physical_addr, virtual_addr)`.
pub type Alloc = fn(usize) -> Option<(u64, usize)>;

// --- xHCI capability register offsets (from BAR base) ------------------
const CAP_CAPLENGTH: usize = 0x00; // u8 caplength, u16 version at +2
const CAP_HCSPARAMS1: usize = 0x04;
const CAP_HCSPARAMS2: usize = 0x08;
const CAP_DBOFF: usize = 0x14;
const CAP_RTSOFF: usize = 0x18;

// --- operational register offsets (from BAR + caplength) ---------------
const OP_USBCMD: usize = 0x00;
const OP_USBSTS: usize = 0x04;
const OP_CRCR: usize = 0x18; // 64-bit
const OP_DCBAAP: usize = 0x30; // 64-bit
const OP_CONFIG: usize = 0x38;
const OP_PORTS: usize = 0x400; // PORTSC[0] at op+0x400, stride 0x10

const USBCMD_RUN: u32 = 1 << 0;
const USBCMD_HCRST: u32 = 1 << 1;
const USBSTS_HCH: u32 = 1 << 0; // HCHalted
const USBSTS_CNR: u32 = 1 << 11; // Controller Not Ready

// TRB types we use.
const TRB_LINK: u32 = 6;

/// A 16-byte Transfer Request Block.
#[repr(C, align(16))]
#[derive(Clone, Copy, Default)]
struct Trb {
    param: u64,
    status: u32,
    control: u32,
}

/// The controller + its rings. Addresses are stored both as physical (for the
/// device) and virtual (HHDM, for the CPU).
pub struct Xhci {
    mmio: usize,     // virtual base of the mapped register space
    op: usize,       // virtual base of operational registers
    rt: usize,       // virtual base of runtime registers
    db: usize,       // virtual base of the doorbell array
    max_ports: u8,
    max_slots: u8,
    // Command ring.
    cmd_ring_va: usize,
    cmd_ring_pa: u64,
    cmd_enqueue: usize,
    cmd_cycle: u32,
    // Event ring.
    evt_ring_va: usize,
    evt_ring_pa: u64,
    evt_dequeue: usize,
    evt_cycle: u32,
    // Device context base address array.
    dcbaa_va: usize,
    dcbaa_pa: u64,
    // Context structure size: 32 or 64 bytes (HCCPARAMS1.CSZ).
    ctx_size: usize,
    // Wrapper-supplied DMA allocator (arch-specific phys/virt mapping).
    alloc: Alloc,
    // The enumerated keyboard, if one was found + configured.
    kbd: Option<Kbd>,
    // The enumerated pointer (USB tablet / mouse), if one was found.
    mouse: Option<Ptr>,
    // Small ring of decoded keyboard bytes (the shared event ring is drained by
    // `pump_events`, which routes reports; `poll_key` pops from here).
    key_buf: [u8; 16],
    key_head: usize,
    key_tail: usize,
}

/// A configured HID boot keyboard.
struct Kbd {
    slot: u8,
    int_dci: u8,       // doorbell target for the interrupt IN endpoint
    int_ring_va: usize,
    int_ring_pa: u64,
    int_enqueue: usize,
    int_cycle: u32,
    report_pa: u64,    // 8-byte HID boot report buffer
    report_va: usize,
    prev: [u8; 8],     // last report, to detect newly-pressed keys
}

/// A configured HID pointer (USB tablet or mouse). The absolute-tablet report
/// (what QEMU `usb-tablet` and VirtualBox's USB pointing device emit) is
/// `[buttons, X:u16le, Y:u16le, wheel]` with X/Y in `0..=0x7fff`; a relative
/// boot mouse is `[buttons, dx:i8, dy:i8]`. `absolute` picks how `poll_mouse`
/// interprets the report.
struct Ptr {
    slot: u8,
    int_dci: u8,
    int_ring_va: usize,
    int_ring_pa: u64,
    int_enqueue: usize,
    int_cycle: u32,
    report_pa: u64,
    report_va: usize,
    report_len: u32,
    absolute: bool,
}

/// Per-device handles produced by `enumerate_common` (an addressed device with
/// its config descriptor read), consumed by `finish_keyboard`/`finish_pointer`.
struct Common {
    slot: u8,
    ep0_va: usize,
    ep0_pa: u64,
    ep0_enq: usize,
    ep0_cycle: u32,
    in_ctx_va: usize,
    in_ctx_pa: u64,
    buf_va: usize,
    total: u32,
    cfg_val: u8,
}

const RING_TRBS: usize = 64; // TRBs per ring (last is a Link on the command ring)

// TRB types.
const TRB_NORMAL: u32 = 1;
const TRB_SETUP: u32 = 2;
const TRB_DATA: u32 = 3;
const TRB_STATUS: u32 = 4;
const TRB_ENABLE_SLOT: u32 = 9;
const TRB_ADDRESS_DEVICE: u32 = 11;
const TRB_CONFIGURE_ENDPOINT: u32 = 12;
const TRB_EVALUATE_CONTEXT: u32 = 13;
// Event TRB types.
const EVT_TRANSFER: u32 = 32;
const EVT_CMD_COMPLETION: u32 = 33;
const CC_SUCCESS: u32 = 1;

// PORTSC bits.
const PORTSC_CCS: u32 = 1 << 0; // current connect status
const PORTSC_PED: u32 = 1 << 1; // port enabled
const PORTSC_PR: u32 = 1 << 4; // port reset
// RW1C/RW1CS bits to preserve (write 0) when doing a read-modify-write.
const PORTSC_RW1CS: u32 = PORTSC_PED | (0x7f << 17);

/// Full barrier between CPU writes to (cacheable) DMA memory and the following
/// device MMIO write. On aarch64 a `dsb sy` orders the Normal-memory TRB writes
/// ahead of the Device-memory doorbell (a `dmb ish` would not order across the
/// Normal/Device boundary); on x86 a `SeqCst` fence suffices.
#[inline]
fn dma_barrier() {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: `dsb sy` is a barrier with no memory-safety implications.
    unsafe {
        core::arch::asm!("dsb sy", options(nostack, preserves_flags));
    }
    #[cfg(not(target_arch = "aarch64"))]
    core::sync::atomic::fence(core::sync::atomic::Ordering::SeqCst);
}

/// Crude busy-wait (~`iters` volatile reads); used for the USB reset-recovery
/// settle where we have no timer in the arch-neutral core. Order of a few ms.
fn spin_delay(iters: u64) {
    let mut sink = 0u64;
    for i in 0..iters {
        sink = sink.wrapping_add(unsafe { read_volatile(&i) });
    }
    let _ = sink;
}

// MMIO accessors. On aarch64 these are single-instruction register-offset
// loads/stores via inline asm — the same guarantee Linux's readl()/writel()
// make. `read_volatile` lets LLVM pick the addressing mode, and in a loop it
// can emit a post-indexed `ldr` (writeback); an MMIO fault from a writeback
// access has no instruction syndrome (ESR.ISV=0), which a hypervisor cannot
// emulate — QEMU/HVF aborts with "Assertion failed: (isv)". Plain `[reg]`
// addressing always faults with ISV=1.
unsafe fn r32(addr: usize) -> u32 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let v: u32;
        core::arch::asm!("ldr {v:w}, [{a}]", v = out(reg) v, a = in(reg) addr, options(nostack, preserves_flags));
        v
    }
    #[cfg(not(target_arch = "aarch64"))]
    unsafe {
        read_volatile(addr as *const u32)
    }
}
unsafe fn w32(addr: usize, v: u32) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        core::arch::asm!("str {v:w}, [{a}]", v = in(reg) v, a = in(reg) addr, options(nostack, preserves_flags));
    }
    #[cfg(not(target_arch = "aarch64"))]
    unsafe {
        write_volatile(addr as *mut u32, v)
    };
}
unsafe fn w64(addr: usize, v: u64) {
    // xHCI 64-bit registers: write low then high dword.
    unsafe {
        w32(addr, v as u32);
        w32(addr + 4, (v >> 32) as u32);
    }
}

impl Xhci {
    /// Bring the controller up from a wrapper-supplied `mmio` register base and
    /// DMA allocator: reset, set up the DCBAA + command ring + event ring, and
    /// start it. Discovery + MMIO mapping are the wrapper's job (arch-specific).
    pub fn bringup(mmio: usize, alloc: Alloc) -> Option<Xhci> {
        // SAFETY: `mmio` is the mapped xHCI register block.
        unsafe {
            let caplen = (r32(mmio + CAP_CAPLENGTH) & 0xff) as usize;
            let op = mmio + caplen;
            let rt = mmio + (r32(mmio + CAP_RTSOFF) & !0x1f) as usize;
            let db = mmio + (r32(mmio + CAP_DBOFF) & !0x3) as usize;
            let hcs1 = r32(mmio + CAP_HCSPARAMS1);
            let max_slots = (hcs1 & 0xff) as u8;
            let max_ports = ((hcs1 >> 24) & 0xff) as u8;
            // Context size: 64 bytes if HCCPARAMS1.CSZ (bit 2), else 32.
            let ctx_size = if r32(mmio + 0x10) & (1 << 2) != 0 { 64 } else { 32 };

            // Wait for CNR to clear, then reset.
            while r32(op + OP_USBSTS) & USBSTS_CNR != 0 {}
            w32(op + OP_USBCMD, USBCMD_HCRST);
            while r32(op + OP_USBCMD) & USBCMD_HCRST != 0 {}
            while r32(op + OP_USBSTS) & USBSTS_CNR != 0 {}

            // Program enabled device slots.
            w32(op + OP_CONFIG, max_slots as u32);

            // Scratchpad buffers, if the controller wants them (HCSPARAMS2).
            let hcs2 = r32(mmio + CAP_HCSPARAMS2);
            let nscratch = (((hcs2 >> 21) & 0x1f) | ((hcs2 >> 27) & 0x1f) << 5) as usize;

            // Device Context Base Address Array: (max_slots + 1) * 8 bytes.
            let (dcbaa_pa, dcbaa_va) = alloc(4096)?;
            let dcbaa_va = dcbaa_va as usize;
            if nscratch > 0 {
                // Scratchpad buffer array + buffers (one page each).
                let (sp_arr_pa, sp_arr_va) = alloc(nscratch * 8)?;
                for i in 0..nscratch {
                    let (buf_pa, _) = alloc(4096)?;
                    write_volatile((sp_arr_va as *mut u64).add(i), buf_pa);
                }
                write_volatile(dcbaa_va as *mut u64, sp_arr_pa); // DCBAA[0] -> scratchpad array
            }
            w64(op + OP_DCBAAP, dcbaa_pa);

            // Command ring.
            let (cmd_ring_pa, cmd_ring_va) = alloc(RING_TRBS * 16)?;
            let cmd_ring_va = cmd_ring_va as usize;
            // Last TRB is a Link back to the start (keeps the ring circular).
            let link = (cmd_ring_va + (RING_TRBS - 1) * 16) as *mut Trb;
            write_volatile(
                link,
                Trb { param: cmd_ring_pa, status: 0, control: (TRB_LINK << 10) | 0x2 /*TC*/ | 0x1 /*cycle*/ },
            );
            // CRCR = ring pointer | RCS(1).
            w64(op + OP_CRCR, cmd_ring_pa | 1);

            // Event ring: one segment + a 1-entry ERST.
            let (evt_ring_pa, evt_ring_va) = alloc(RING_TRBS * 16)?;
            let evt_ring_va = evt_ring_va as usize;
            let (erst_pa, erst_va) = alloc(64)?;
            let erst_va = erst_va as usize;
            write_volatile(erst_va as *mut u64, evt_ring_pa); // segment base
            write_volatile((erst_va + 8) as *mut u32, RING_TRBS as u32); // segment size
            // Interrupter 0 registers at rt + 0x20.
            let ir0 = rt + 0x20;
            w32(ir0 + 0x08, 1); // ERSTSZ = 1
            w64(ir0 + 0x10, erst_pa); // ERSTBA -- actually ERDP first per some HCs; set ERDP then ERSTBA
            w64(ir0 + 0x18, evt_ring_pa); // ERDP
            w64(ir0 + 0x10, erst_pa); // ERSTBA (writing this arms the interrupter)

            // Run.
            let c = r32(op + OP_USBCMD);
            w32(op + OP_USBCMD, c | USBCMD_RUN);
            // Wait for HCHalted to clear.
            let mut spins = 0u32;
            while r32(op + OP_USBSTS) & USBSTS_HCH != 0 {
                spins += 1;
                if spins > 1_000_000 {
                    crate::ktrace::log("xhci", "controller did not leave halted state");
                    return None;
                }
            }

            crate::ktrace::log_fmt(format_args!(
                "xhci: running (caplen {caplen}, {max_slots} slots, {max_ports} ports, {nscratch} scratchpad)"
            ));
            Some(Xhci {
                mmio,
                op,
                rt,
                db,
                max_ports,
                max_slots,
                cmd_ring_va,
                cmd_ring_pa,
                cmd_enqueue: 0,
                cmd_cycle: 1,
                evt_ring_va,
                evt_ring_pa,
                evt_dequeue: 0,
                evt_cycle: 1,
                dcbaa_va,
                dcbaa_pa,
                ctx_size,
                alloc,
                kbd: None,
                mouse: None,
                key_buf: [0; 16],
                key_head: 0,
                key_tail: 0,
            })
        }
    }

    /// PORTSC of 1-based `port`.
    fn portsc(&self, port: u8) -> usize {
        self.op + OP_PORTS + (port as usize - 1) * 0x10
    }

    /// Ring a doorbell (slot 0 = command ring; slot N, target = endpoint DCI).
    /// A full DMA barrier first ensures the TRBs we just wrote to (cacheable)
    /// memory are globally visible before the controller sees the doorbell —
    /// required on real hardware / VirtualBox (QEMU's coherent emulation hid it).
    unsafe fn doorbell(&self, slot: u8, target: u32) {
        dma_barrier();
        unsafe { w32(self.db + slot as usize * 4, target) };
    }

    /// Push a TRB onto a command/transfer ring and advance, wrapping at the
    /// Link TRB in the last slot. Associated (no `self`) so it can borrow a
    /// ring's fields directly.
    unsafe fn ring_push(va: usize, pa: u64, enq: &mut usize, cycle: &mut u32, param: u64, status: u32, control: u32) {
        let control = (control & !1) | *cycle;
        unsafe { write_volatile((va + *enq * 16) as *mut Trb, Trb { param, status, control }) };
        *enq += 1;
        if *enq == RING_TRBS - 1 {
            let link = (TRB_LINK << 10) | 0x2 | *cycle;
            unsafe { write_volatile((va + (RING_TRBS - 1) * 16) as *mut Trb, Trb { param: pa, status: 0, control: link }) };
            *enq = 0;
            *cycle ^= 1;
        }
    }

    /// Consume the next event TRB if one is ready (cycle bit matches), updating
    /// the dequeue pointer + ERDP. Non-blocking.
    unsafe fn next_event(&mut self) -> Option<Trb> {
        let trb = unsafe { read_volatile((self.evt_ring_va + self.evt_dequeue * 16) as *const Trb) };
        if (trb.control & 1) != self.evt_cycle {
            return None;
        }
        self.evt_dequeue += 1;
        if self.evt_dequeue == RING_TRBS {
            self.evt_dequeue = 0;
            self.evt_cycle ^= 1;
        }
        let erdp = self.evt_ring_pa + (self.evt_dequeue as u64) * 16;
        unsafe { w64(self.rt + 0x20 + 0x18, erdp | (1 << 3)) }; // ERDP, EHB=1 to ack
        Some(trb)
    }

    /// Block (bounded) for the next event of `want_type`; returns its TRB.
    unsafe fn wait_event(&mut self, want_type: u32) -> Option<Trb> {
        let mut spins = 0u64;
        loop {
            if let Some(ev) = unsafe { self.next_event() } {
                if (ev.control >> 10) & 0x3f == want_type {
                    return Some(ev);
                }
            } else {
                spins += 1;
                if spins > 20_000_000 {
                    unsafe { self.dump_event_ring(want_type) };
                    return None;
                }
            }
        }
    }

    /// Diagnostic dumped when `wait_event` times out: the current dequeue
    /// position + cycle, and every event-ring TRB's (control, status) so we can
    /// tell "controller posted nothing" from "event present, cycle mismatch".
    unsafe fn dump_event_ring(&self, want_type: u32) {
        crate::ktrace::log_fmt(format_args!(
            "xhci: wait_event({want_type}) TIMEOUT deq={} cyc={} erdp-based",
            self.evt_dequeue, self.evt_cycle
        ));
        for i in 0..RING_TRBS {
            let t = unsafe { read_volatile((self.evt_ring_va + i * 16) as *const Trb) };
            // Only log non-empty slots (control != 0) to keep it short.
            if t.control != 0 || t.status != 0 {
                let ty = (t.control >> 10) & 0x3f;
                let cc = (t.status >> 24) & 0xff;
                crate::ktrace::log_fmt(format_args!("  evt[{i}] type={ty} cc={cc} c={} ctrl={:#x}", t.control & 1, t.control));
            }
        }
    }

    /// Run a command TRB; return (completion_code, slot_id).
    unsafe fn command(&mut self, param: u64, control: u32) -> Option<(u32, u8)> {
        unsafe {
            Self::ring_push(self.cmd_ring_va, self.cmd_ring_pa, &mut self.cmd_enqueue, &mut self.cmd_cycle, param, 0, control);
            self.doorbell(0, 0);
            let ev = self.wait_event(EVT_CMD_COMPLETION)?;
            Some(((ev.status >> 24) & 0xff, ((ev.control >> 24) & 0xff) as u8))
        }
    }

    /// A control transfer on EP0 of `slot` over ring `(va,pa,enq,cycle)`.
    /// `setup` is the packed 8-byte setup packet; `buf_pa`/`len` an optional
    /// data stage (`data_in` = device→host). Returns true on success.
    #[allow(clippy::too_many_arguments)]
    unsafe fn control(
        &mut self,
        slot: u8,
        va: usize,
        pa: u64,
        enq: &mut usize,
        cycle: &mut u32,
        setup: u64,
        buf_pa: u64,
        len: u32,
        data_in: bool,
    ) -> bool {
        unsafe {
            let trt = if len == 0 { 0 } else if data_in { 3 } else { 2 };
            // Setup stage (immediate data).
            Self::ring_push(va, pa, enq, cycle, setup, 8, (TRB_SETUP << 10) | (trt << 16) | (1 << 6));
            // Data stage.
            if len > 0 {
                let dir = if data_in { 1 << 16 } else { 0 };
                Self::ring_push(va, pa, enq, cycle, buf_pa, len, (TRB_DATA << 10) | dir);
            }
            // Status stage (opposite direction; IOC to get an event).
            let sdir = if data_in && len > 0 { 0 } else { 1 << 16 };
            Self::ring_push(va, pa, enq, cycle, 0, 0, (TRB_STATUS << 10) | sdir | (1 << 5));
            self.doorbell(slot, 1); // EP0 = DCI 1
            match self.wait_event(EVT_TRANSFER) {
                Some(ev) => {
                    let cc = (ev.status >> 24) & 0xff;
                    let ok = cc == CC_SUCCESS || cc == 13 /* SHORT_PACKET */;
                    if !ok {
                        crate::ktrace::log_fmt(format_args!("xhci: control transfer cc={cc} (len {len}, in={data_in})"));
                    }
                    ok
                }
                None => {
                    crate::ktrace::log_fmt(format_args!("xhci: control transfer TIMEOUT (len {len}, in={data_in})"));
                    false
                }
            }
        }
    }

    /// C2/C3: find a connected keyboard, reset its port, address the device,
    /// read its descriptors, set the config + boot protocol, and arm the
    /// interrupt IN endpoint.
    pub fn enumerate_keyboard(&mut self) -> bool {
        self.enumerate_input()
    }

    /// Enumerate the connected USB HID devices — a boot keyboard and/or a
    /// pointer (USB tablet / mouse), which may sit on separate ports.
    pub fn enumerate_input(&mut self) -> bool {
        // Fast path out: nothing connected on any port -> no retries, no delays.
        let any_port = (1..=self.max_ports).any(|p| unsafe { r32(self.portsc(p)) } & PORTSC_CCS != 0);
        if !any_port {
            crate::ktrace::log("xhci", "no USB device connected");
            return false;
        }
        // Enumerate with RETRIES. VirtualBox resets the attached VUSB device
        // *asynchronously* after our controller reset (~20 ms); a control
        // transfer submitted in that window is silently dropped. A settle +
        // fresh attempt succeeds once the device-level reset has finished. QEMU
        // is instantaneous and never needs the retry.
        for attempt in 0..3 {
            if attempt > 0 {
                spin_delay(100_000_000);
                crate::ktrace::log_fmt(format_args!("xhci: retrying enumeration (attempt {})", attempt + 1));
            }
            // SAFETY: single-threaded boot; all DMA regions are freshly allocated.
            unsafe { self.scan_ports() };
            if self.kbd.is_some() && self.mouse.is_some() {
                break;
            }
        }
        if self.kbd.is_none() {
            crate::ktrace::log("xhci", "no HID keyboard enumerated");
        }
        self.kbd.is_some() || self.mouse.is_some()
    }

    /// Whether a HID keyboard / pointer was enumerated (boot diagnostic).
    pub fn has_keyboard(&self) -> bool {
        self.kbd.is_some()
    }
    pub fn has_mouse(&self) -> bool {
        self.mouse.is_some()
    }

    /// Enumerate every connected port once, classifying each device as a boot
    /// keyboard or a pointer and configuring whichever slot we still need. A
    /// tablet on the first port no longer starves the keyboard (each port is
    /// tried, not just the first).
    unsafe fn scan_ports(&mut self) {
        for port in 1..=self.max_ports {
            if self.kbd.is_some() && self.mouse.is_some() {
                break;
            }
            if unsafe { r32(self.portsc(port)) } & PORTSC_CCS == 0 {
                continue;
            }
            let Some(mut c) = (unsafe { self.enumerate_common(port) }) else { continue };
            if self.kbd.is_none() {
                if let Some((iface, ep, mps, ivl)) = unsafe { parse_hid_keyboard(c.buf_va, c.total as usize) } {
                    if let Some(k) = unsafe { self.finish_keyboard(&mut c, iface, ep, mps, ivl) } {
                        self.kbd = Some(k);
                        continue;
                    }
                }
            }
            if self.mouse.is_none() {
                if let Some((iface, ep, mps, ivl, proto)) = unsafe { parse_hid_pointer(c.buf_va, c.total as usize) } {
                    if let Some(p) = unsafe { self.finish_pointer(&mut c, iface, ep, mps, ivl, proto) } {
                        self.mouse = Some(p);
                        continue;
                    }
                }
            }
        }
    }

    /// Steps 1–8 shared by keyboard + pointer enumeration: reset `port`, enable a
    /// slot, address the device, learn EP0 max-packet, and read its full config
    /// descriptor into a buffer. Returns the per-device handles; the caller then
    /// classifies the config and finishes as a keyboard or pointer.
    unsafe fn enumerate_common(&mut self, port: u8) -> Option<Common> {
        let psc = self.portsc(port);
        unsafe {
            // Reset the port (USB2/FullSpeed needs it; preserve RW1C status bits).
            let before = r32(psc);
            crate::ktrace::log_fmt(format_args!("xhci: resetting port {port} (portsc={before:#x})"));
            let v = before & !PORTSC_RW1CS;
            w32(psc, v | PORTSC_PR);
            // Wait for the port to come back enabled. Each read is an MMIO trap
            // (slow under a hypervisor), so the bound is in reads, not time:
            // ~2M reads is seconds of real time, far beyond a real port reset
            // (~10 ms on VirtualBox; instant on QEMU).
            let mut spins = 0u64;
            while r32(psc) & PORTSC_PED == 0 {
                spins += 1;
                if spins > 2_000_000 {
                    crate::ktrace::log_fmt(format_args!("xhci: port {port} did not re-enable after reset (portsc={:#x})", r32(psc)));
                    return None;
                }
            }
            // USB reset-recovery: a device needs time after reset before it will
            // answer control transfers (the spec allows up to 10 ms). QEMU is
            // instant, but slower/FullSpeed models (e.g. VirtualBox's HID
            // keyboard) can NAK Address Device / GET_DESCRIPTOR without this.
            spin_delay(2_000_000);
        }
        let speed = (unsafe { r32(psc) } >> 10) & 0xf;
        let ep0_mps: u32 = match speed {
            3 => 64,      // high
            4 | 5 => 512, // super/super+
            _ => 8,       // full/low
        };
        crate::ktrace::log_fmt(format_args!("xhci: port {port} connected, speed {speed}, resetting -> enabled"));

        // 2. Enable a slot.
        let (cc, slot) = unsafe { self.command(0, TRB_ENABLE_SLOT << 10) }?;
        if cc != CC_SUCCESS {
            crate::ktrace::log_fmt(format_args!("xhci: enable slot failed cc={cc}"));
            return None;
        }

        // 3. Device context + DCBAA entry.
        let (dev_ctx_pa, dev_ctx_va) = (self.alloc)(4096)?;
        let dev_ctx_va = dev_ctx_va as usize;
        unsafe { write_volatile((self.dcbaa_va as *mut u64).add(slot as usize), dev_ctx_pa) };

        // 4. EP0 transfer ring.
        let (ep0_pa, ep0_va) = (self.alloc)(RING_TRBS * 16)?;
        let ep0_va = ep0_va as usize;
        unsafe { self.init_link(ep0_va, ep0_pa) };
        let mut ep0_enq = 0usize;
        let mut ep0_cycle = 1u32;

        // 5. Input context for Address Device (add slot + EP0).
        let (in_ctx_pa, in_ctx_va) = (self.alloc)(4096)?;
        let in_ctx_va = in_ctx_va as usize;
        unsafe {
            self.build_input_slot_ep0(in_ctx_va, port, speed, ep0_mps, ep0_pa);
            // 6. Address Device.
            let (cc, _) = self.command(in_ctx_pa, (TRB_ADDRESS_DEVICE << 10) | ((slot as u32) << 24))?;
            if cc != CC_SUCCESS {
                crate::ktrace::log_fmt(format_args!("xhci: address device failed cc={cc}"));
                return None;
            }
        }
        // Diagnostic: the output device context after Address Device — slot
        // state (dword3 bits 31:27) + EP0 state (EP-context dword0 bits 2:0) —
        // so a "transfer never runs" failure can be told from EP0 not being in
        // the Running state (1). QEMU: slot=2(addressed), ep0=1(running).
        unsafe {
            let slot_state = (read_volatile((dev_ctx_va + 12) as *const u32) >> 27) & 0x1f;
            let ep0_state = read_volatile((dev_ctx_va + self.ctx_size) as *const u32) & 0x7;
            crate::ktrace::log_fmt(format_args!("xhci: post-address slot_state={slot_state} ep0_state={ep0_state}"));
            let _ = dev_ctx_pa;
        }
        crate::ktrace::log_fmt(format_args!("xhci: device addressed on slot {slot}"));
        // SET_ADDRESS recovery: the USB spec gives a device up to 2 ms after
        // being addressed before it must answer EP0 requests. QEMU replies
        // instantly, but a real-timed device model (VirtualBox) NAKs a control
        // transfer issued immediately after Address Device, so wait first.
        spin_delay(4_000_000);

        // 7. Learn the real EP0 max packet size before any multi-byte read.
        // We addressed the device assuming MPS=8 (full/low) or 64 (high). A
        // full-speed device may actually use MPS 16/32/64; reading the 9-byte
        // config descriptor with an under-sized MPS makes the device pack all 9
        // bytes into one packet → the host sees a packet > its configured MPS →
        // BABBLE, and enumeration fails (exactly what VirtualBox's full-speed
        // keyboard hit). Reading only the first 8 bytes of the DEVICE descriptor
        // is always safe (8 ≤ any MPS), and byte 7 is bMaxPacketSize0.
        let (buf_pa, buf_va) = (self.alloc)(4096)?;
        let buf_va = buf_va as usize;
        let got_dev = unsafe {
            self.control(slot, ep0_va, ep0_pa, &mut ep0_enq, &mut ep0_cycle, setup(0x80, 6, 0x0100, 0, 8), buf_pa, 8, true)
        };
        if !got_dev {
            // Diagnostic: did the controller consume the EP0 TRBs? Read the EP0
            // TR-dequeue pointer back from the output device context — if it
            // advanced past the ring base, the transfer executed but the event
            // was lost; if not, the controller never processed the doorbell.
            // Plus USBSTS (HSE = a DMA/host error reading the ring).
            unsafe {
                let deq = read_volatile((dev_ctx_va + self.ctx_size + 8) as *const u64);
                let usbsts = r32(self.op + OP_USBSTS);
                crate::ktrace::log_fmt(format_args!("xhci: EP0 stuck: tr_deq={:#x} ring_base={:#x} usbsts={:#x}", deq, ep0_pa, usbsts));
            }
        }
        if got_dev {
            let b = unsafe { read_volatile((buf_va + 7) as *const u8) } as u32;
            // full/low (speed 1/2): bMaxPacketSize0 is the byte count (8/16/32/64).
            // super-speed (4): it's an exponent (9 => 512). high (3) is always 64.
            let real_mps = match speed {
                1 | 2 => b,
                4 => 1u32 << b,
                _ => 64,
            };
            if real_mps >= 8 && real_mps != ep0_mps {
                crate::ktrace::log_fmt(format_args!("xhci: EP0 max packet {ep0_mps} -> {real_mps}; re-evaluating"));
                unsafe {
                    self.build_input_eval_mps(in_ctx_va, real_mps);
                    let _ = self.command(in_ctx_pa, (TRB_EVALUATE_CONTEXT << 10) | ((slot as u32) << 24));
                }
            }
        }

        // 8. Read the configuration descriptor: first 9 bytes for wTotalLength.
        let ok = unsafe {
            self.control(slot, ep0_va, ep0_pa, &mut ep0_enq, &mut ep0_cycle, setup(0x80, 6, 0x0200, 0, 9), buf_pa, 9, true)
        };
        if !ok {
            return None;
        }
        let total = unsafe { read_u16(buf_va + 2) } as u32;
        let cfg_val = unsafe { read_volatile((buf_va + 5) as *const u8) };
        let ok = unsafe {
            self.control(slot, ep0_va, ep0_pa, &mut ep0_enq, &mut ep0_cycle, setup(0x80, 6, 0x0200, 0, total as u16), buf_pa, total.min(4096), true)
        };
        if !ok {
            return None;
        }
        Some(Common { slot, ep0_va, ep0_pa, ep0_enq, ep0_cycle, in_ctx_va, in_ctx_pa, buf_va, total, cfg_val })
    }

    /// Set the configuration, put the interface into a protocol, and configure
    /// the interrupt IN endpoint. Shared tail of keyboard/pointer setup; returns
    /// the endpoint's DCI + ring/report allocations.
    #[allow(clippy::too_many_arguments)]
    unsafe fn finish_endpoint(
        &mut self,
        c: &mut Common,
        iface: u8,
        ep_addr: u8,
        ep_mps: u32,
        interval: u8,
        set_boot: bool,
    ) -> Option<(u8, usize, u64, u64, usize)> {
        unsafe {
            // Set configuration.
            self.control(c.slot, c.ep0_va, c.ep0_pa, &mut c.ep0_enq, &mut c.ep0_cycle, setup(0x00, 9, c.cfg_val as u16, 0, 0), 0, 0, false);
            if set_boot {
                // SET_PROTOCOL(boot=0) to the HID interface (class request).
                self.control(c.slot, c.ep0_va, c.ep0_pa, &mut c.ep0_enq, &mut c.ep0_cycle, setup(0x21, 0x0b, 0, iface as u16, 0), 0, 0, false);
            }
        }
        let (int_pa, int_va) = (self.alloc)(RING_TRBS * 16)?;
        let int_va = int_va as usize;
        unsafe { self.init_link(int_va, int_pa) };
        let (report_pa, report_va) = (self.alloc)(4096)?;
        let report_va = report_va as usize;
        let int_dci = (((ep_addr & 0x0f) as u32) * 2 + 1) as u8; // IN endpoint DCI
        unsafe {
            self.build_input_configure(c.in_ctx_va, int_dci, ep_mps, interval, int_pa);
            let (cc, _) = self.command(c.in_ctx_pa, (TRB_CONFIGURE_ENDPOINT << 10) | ((c.slot as u32) << 24))?;
            if cc != CC_SUCCESS {
                crate::ktrace::log_fmt(format_args!("xhci: configure endpoint failed cc={cc}"));
                return None;
            }
        }
        Some((int_dci, int_va, int_pa, report_pa, report_va))
    }

    /// Finish a boot keyboard: set config + boot protocol, configure its
    /// interrupt endpoint, and arm the first transfer.
    unsafe fn finish_keyboard(&mut self, c: &mut Common, iface: u8, ep_addr: u8, ep_mps: u32, interval: u8) -> Option<Kbd> {
        let (int_dci, int_va, int_pa, report_pa, report_va) =
            unsafe { self.finish_endpoint(c, iface, ep_addr, ep_mps, interval, true) }?;
        let mut kbd = Kbd {
            slot: c.slot, int_dci, int_ring_va: int_va, int_ring_pa: int_pa,
            int_enqueue: 0, int_cycle: 1, report_pa, report_va, prev: [0; 8],
        };
        unsafe { self.queue_interrupt(&mut kbd) };
        crate::ktrace::log_fmt(format_args!("xhci: HID keyboard ready (slot {}, ep {ep_addr:#x}, dci {int_dci})", c.slot));
        Some(kbd)
    }

    /// Finish a pointer: a boot mouse (proto 2, relative) gets boot protocol; a
    /// tablet (proto 0) keeps report protocol (absolute). Configure the endpoint
    /// and arm the first transfer.
    unsafe fn finish_pointer(&mut self, c: &mut Common, iface: u8, ep_addr: u8, ep_mps: u32, interval: u8, proto: u8) -> Option<Ptr> {
        let boot_mouse = proto == 2;
        let (int_dci, int_va, int_pa, report_pa, report_va) =
            unsafe { self.finish_endpoint(c, iface, ep_addr, ep_mps, interval, boot_mouse) }?;
        let mut ptr = Ptr {
            slot: c.slot, int_dci, int_ring_va: int_va, int_ring_pa: int_pa,
            int_enqueue: 0, int_cycle: 1, report_pa, report_va,
            report_len: ep_mps.clamp(4, 8),
            absolute: !boot_mouse,
        };
        unsafe {
            self.arm_int(ptr.int_ring_va, ptr.int_ring_pa, &mut ptr.int_enqueue, &mut ptr.int_cycle, ptr.report_pa, ptr.report_len, ptr.slot, ptr.int_dci);
        }
        crate::ktrace::log_fmt(format_args!(
            "xhci: HID pointer ready (slot {}, ep {ep_addr:#x}, dci {int_dci}, {})",
            c.slot,
            if boot_mouse { "boot mouse" } else { "absolute tablet" }
        ));
        Some(ptr)
    }

    /// Initialize a transfer ring's trailing Link TRB (back to its own start).
    unsafe fn init_link(&self, va: usize, pa: u64) {
        unsafe {
            write_volatile((va + (RING_TRBS - 1) * 16) as *mut Trb, Trb { param: pa, status: 0, control: (TRB_LINK << 10) | 0x2 | 1 });
        }
    }

    /// Build the input context for Address Device: add flags A0|A1, a slot
    /// context (route 0, this root port, speed, 1 context entry) and the EP0
    /// control endpoint context.
    unsafe fn build_input_slot_ep0(&self, in_ctx_va: usize, port: u8, speed: u32, ep0_mps: u32, ep0_pa: u64) {
        unsafe {
            // zero the control + slot + ep0 area
            core::ptr::write_bytes(in_ctx_va as *mut u8, 0, self.ctx_size * 3);
            w32(in_ctx_va + 4, 0b11); // Add flags: A0 (slot) | A1 (EP0)
            let slot_ctx = in_ctx_va + self.ctx_size;
            w32(slot_ctx, (1 << 27) | (speed << 20)); // ctx entries=1, speed, route=0
            w32(slot_ctx + 4, (port as u32) << 16); // root hub port number
            let ep0 = in_ctx_va + self.ctx_size * 2;
            w32(ep0 + 4, (4 << 3) | (ep0_mps << 16) | (3 << 1)); // type=Control, MPS, CErr=3
            w64(ep0 + 8, ep0_pa | 1); // TR dequeue ptr | DCS
            w32(ep0 + 16, 8); // avg TRB length
        }
    }

    /// Build the input context for Evaluate Context to update EP0's max packet
    /// size (learned from the device descriptor). Add flag A1 (EP0) only; set the
    /// EP0 context's type/MPS/CErr — the controller re-reads MPS from it.
    unsafe fn build_input_eval_mps(&self, in_ctx_va: usize, ep0_mps: u32) {
        unsafe {
            core::ptr::write_bytes(in_ctx_va as *mut u8, 0, self.ctx_size * 3);
            w32(in_ctx_va + 4, 0b10); // Add flags: A1 (EP0) only
            let ep0 = in_ctx_va + self.ctx_size * 2;
            w32(ep0 + 4, (4 << 3) | (ep0_mps << 16) | (3 << 1)); // type=Control, MPS, CErr=3
        }
    }

    /// Extend the input context for Configure Endpoint: add the interrupt IN
    /// endpoint at `dci` and bump the slot's context-entries count.
    unsafe fn build_input_configure(&self, in_ctx_va: usize, dci: u8, mps: u32, interval: u8, int_pa: u64) {
        unsafe {
            w32(in_ctx_va, 0); // drop flags
            w32(in_ctx_va + 4, 0b1 | (1 << dci)); // add A0 (slot) + the endpoint
            let slot_ctx = in_ctx_va + self.ctx_size;
            let d0 = r32(slot_ctx) & !(0x1f << 27);
            w32(slot_ctx, d0 | ((dci as u32) << 27)); // context entries = dci
            let ep = in_ctx_va + self.ctx_size * (dci as usize + 1);
            core::ptr::write_bytes(ep as *mut u8, 0, self.ctx_size);
            // Interrupt interval: xHCI encodes it as 2^(Interval) * 125us. For a
            // full-speed 1ms endpoint that's ~8; clamp bInterval into range.
            let ivl = (interval.max(1).min(16) as u32).ilog2() + 3;
            w32(ep, ivl << 16);
            w32(ep + 4, (7 << 3) | (mps << 16) | (3 << 1)); // type=Interrupt IN, MPS, CErr=3
            w64(ep + 8, int_pa | 1);
            w32(ep + 16, mps); // avg TRB length ~ max packet
        }
    }

    /// Queue one interrupt IN transfer for the keyboard's 8-byte boot report.
    unsafe fn queue_interrupt(&self, kbd: &mut Kbd) {
        unsafe {
            self.arm_int(kbd.int_ring_va, kbd.int_ring_pa, &mut kbd.int_enqueue, &mut kbd.int_cycle, kbd.report_pa, 8, kbd.slot, kbd.int_dci);
        }
    }

    /// Queue one interrupt IN transfer of `len` bytes on `(ring, slot, dci)`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn arm_int(&self, ring_va: usize, ring_pa: u64, enq: &mut usize, cycle: &mut u32, report_pa: u64, len: u32, slot: u8, dci: u8) {
        unsafe {
            Self::ring_push(ring_va, ring_pa, enq, cycle, report_pa, len, (TRB_NORMAL << 10) | (1 << 5));
            self.doorbell(slot, dci as u32);
        }
    }

    /// Drain the shared event ring once, routing each transfer event to the
    /// keyboard (→ decoded bytes into `key_buf`) or the pointer (→ `crate::mouse`)
    /// by slot + endpoint DCI, and re-arm that endpoint. Both `poll_key` and
    /// `poll_mouse` call this; a single drainer avoids the two stealing each
    /// other's events off the one ring.
    fn pump_events(&mut self) {
        let mut kbd = self.kbd.take();
        let mut mouse = self.mouse.take();
        // SAFETY: rings/buffers are live for the controller's lifetime.
        unsafe {
            while let Some(ev) = self.next_event() {
                if (ev.control >> 10) & 0x3f != EVT_TRANSFER {
                    continue;
                }
                let slot = ((ev.control >> 24) & 0xff) as u8;
                let dci = ((ev.control >> 16) & 0x1f) as u8;
                if let Some(k) = kbd.as_mut() {
                    if k.slot == slot && k.int_dci == dci {
                        let mut report = [0u8; 8];
                        for (i, b) in report.iter_mut().enumerate() {
                            *b = read_volatile((k.report_va + i) as *const u8);
                        }
                        let shift = report[0] & 0x22 != 0;
                        let ctrl = report[0] & 0x11 != 0;
                        for &usage in &report[2..8] {
                            if usage != 0 && !k.prev[2..8].contains(&usage) {
                                if let Some(a) = hid_to_ascii(usage, shift, ctrl) {
                                    self.key_push(a);
                                }
                            }
                        }
                        k.prev = report;
                        self.queue_interrupt(k);
                        continue;
                    }
                }
                if let Some(m) = mouse.as_mut() {
                    if m.slot == slot && m.int_dci == dci {
                        let n = m.report_len as usize;
                        let mut rep = [0u8; 8];
                        for (i, b) in rep.iter_mut().enumerate().take(n.min(8)) {
                            *b = read_volatile((m.report_va + i) as *const u8);
                        }
                        if m.absolute {
                            // [buttons, X:u16le, Y:u16le] in 0..=0x7fff (tablet).
                            let x = (rep[1] as u32) | ((rep[2] as u32) << 8);
                            let y = (rep[3] as u32) | ((rep[4] as u32) << 8);
                            crate::mouse::set_abs(x as i32, y as i32, 0x7fff);
                        } else {
                            // [buttons, dx:i8, dy:i8] (boot mouse, relative).
                            crate::mouse::move_rel(rep[1] as i8 as i32, rep[2] as i8 as i32);
                        }
                        crate::mouse::set_left(rep[0] & 1 != 0);
                        self.arm_int(m.int_ring_va, m.int_ring_pa, &mut m.int_enqueue, &mut m.int_cycle, m.report_pa, m.report_len, m.slot, m.int_dci);
                        continue;
                    }
                }
            }
        }
        self.kbd = kbd;
        self.mouse = mouse;
    }

    fn key_push(&mut self, b: u8) {
        let n = (self.key_head + 1) % self.key_buf.len();
        if n != self.key_tail {
            self.key_buf[self.key_head] = b;
            self.key_head = n;
        }
    }

    /// The next decoded keyboard byte, if any (drains + routes events first).
    pub fn poll_key(&mut self) -> Option<u8> {
        self.pump_events();
        if self.key_head == self.key_tail {
            None
        } else {
            let b = self.key_buf[self.key_tail];
            self.key_tail = (self.key_tail + 1) % self.key_buf.len();
            Some(b)
        }
    }

    /// Drain pending pointer reports into [`crate::mouse`] (no-op without a USB
    /// pointer). Shares the event drain with `poll_key`.
    pub fn poll_mouse(&mut self) {
        self.pump_events();
    }
}

/// Pack a USB setup packet into the little-endian u64 a Setup Stage TRB expects.
fn setup(bm_request_type: u8, b_request: u8, w_value: u16, w_index: u16, w_length: u16) -> u64 {
    (bm_request_type as u64)
        | ((b_request as u64) << 8)
        | ((w_value as u64) << 16)
        | ((w_index as u64) << 32)
        | ((w_length as u64) << 48)
}

unsafe fn read_u16(addr: usize) -> u16 {
    unsafe { (read_volatile(addr as *const u8) as u16) | ((read_volatile((addr + 1) as *const u8) as u16) << 8) }
}

/// Walk a configuration descriptor looking for a HID boot-keyboard interface
/// (class 3, subclass 1, protocol 1) and its interrupt IN endpoint. Returns
/// `(interface number, endpoint address, max packet size, bInterval)`.
unsafe fn parse_hid_keyboard(buf: usize, len: usize) -> Option<(u8, u8, u32, u8)> {
    let mut i = 0usize;
    let mut in_kbd_iface = false;
    let mut iface_num = 0u8;
    while i + 2 <= len {
        let blen = unsafe { read_volatile((buf + i) as *const u8) } as usize;
        let btype = unsafe { read_volatile((buf + i + 1) as *const u8) };
        if blen < 2 {
            break;
        }
        match btype {
            0x04 => {
                // Interface: bInterfaceNumber@2, class@5, subclass@6, protocol@7.
                iface_num = unsafe { read_volatile((buf + i + 2) as *const u8) };
                let class = unsafe { read_volatile((buf + i + 5) as *const u8) };
                let sub = unsafe { read_volatile((buf + i + 6) as *const u8) };
                let proto = unsafe { read_volatile((buf + i + 7) as *const u8) };
                in_kbd_iface = class == 3 && sub == 1 && proto == 1;
            }
            0x05 if in_kbd_iface => {
                // Endpoint: address@2, attributes@3, wMaxPacketSize@4, interval@6.
                let addr = unsafe { read_volatile((buf + i + 2) as *const u8) };
                let attr = unsafe { read_volatile((buf + i + 3) as *const u8) };
                if addr & 0x80 != 0 && attr & 0x3 == 3 {
                    let mps = unsafe { read_u16(buf + i + 4) } as u32 & 0x7ff;
                    let interval = unsafe { read_volatile((buf + i + 6) as *const u8) };
                    return Some((iface_num, addr, mps.max(8), interval));
                }
            }
            _ => {}
        }
        i += blen;
    }
    None
}

/// Walk a config descriptor for a HID **pointer** interface — a USB tablet
/// (class 3, protocol 0) or a boot mouse (class 3, protocol 2) — and its
/// interrupt IN endpoint. Returns `(interface, endpoint addr, max packet,
/// bInterval, bInterfaceProtocol)`. Skips the keyboard (protocol 1).
unsafe fn parse_hid_pointer(buf: usize, len: usize) -> Option<(u8, u8, u32, u8, u8)> {
    let mut i = 0usize;
    let mut in_ptr_iface = false;
    let mut iface_num = 0u8;
    let mut proto = 0u8;
    while i + 2 <= len {
        let blen = unsafe { read_volatile((buf + i) as *const u8) } as usize;
        let btype = unsafe { read_volatile((buf + i + 1) as *const u8) };
        if blen < 2 {
            break;
        }
        match btype {
            0x04 => {
                iface_num = unsafe { read_volatile((buf + i + 2) as *const u8) };
                let class = unsafe { read_volatile((buf + i + 5) as *const u8) };
                let p = unsafe { read_volatile((buf + i + 7) as *const u8) };
                // HID pointer: mouse (proto 2) or tablet/absolute (proto 0). Not
                // the keyboard (proto 1).
                in_ptr_iface = class == 3 && (p == 2 || p == 0);
                proto = p;
            }
            0x05 if in_ptr_iface => {
                let addr = unsafe { read_volatile((buf + i + 2) as *const u8) };
                let attr = unsafe { read_volatile((buf + i + 3) as *const u8) };
                if addr & 0x80 != 0 && attr & 0x3 == 3 {
                    let mps = unsafe { read_u16(buf + i + 4) } as u32 & 0x7ff;
                    let interval = unsafe { read_volatile((buf + i + 6) as *const u8) };
                    return Some((iface_num, addr, mps.max(4), interval, proto));
                }
            }
            _ => {}
        }
        i += blen;
    }
    None
}

/// Map a USB HID keyboard usage id to an ASCII byte (US layout). Handles
/// letters, digits, common punctuation, space/enter/tab/backspace; Shift for
/// the upper register; Ctrl+letter -> control code (Ctrl+C=3, Ctrl+D=4).
fn hid_to_ascii(usage: u8, shift: bool, ctrl: bool) -> Option<u8> {
    let base: u8 = match usage {
        0x04..=0x1d => b'a' + (usage - 0x04),         // a..z
        0x1e..=0x26 => b'1' + (usage - 0x1e),         // 1..9
        0x27 => b'0',
        0x28 => b'\r', // Enter
        0x29 => 0x1b,  // Esc
        0x2a => 0x08,  // Backspace
        0x2b => b'\t', // Tab
        0x2c => b' ',  // Space
        0x2d => b'-',
        0x2e => b'=',
        0x2f => b'[',
        0x30 => b']',
        0x31 => b'\\',
        0x33 => b';',
        0x34 => b'\'',
        0x35 => b'`',
        0x36 => b',',
        0x37 => b'.',
        0x38 => b'/',
        _ => return None,
    };
    let ch = if shift { shift_ascii(base) } else { base };
    if ctrl && ch.is_ascii_alphabetic() {
        Some(ch.to_ascii_uppercase() & 0x1f)
    } else {
        Some(ch)
    }
}

/// The shifted form of a US-layout key.
fn shift_ascii(c: u8) -> u8 {
    match c {
        b'a'..=b'z' => c.to_ascii_uppercase(),
        b'1' => b'!',
        b'2' => b'@',
        b'3' => b'#',
        b'4' => b'$',
        b'5' => b'%',
        b'6' => b'^',
        b'7' => b'&',
        b'8' => b'*',
        b'9' => b'(',
        b'0' => b')',
        b'-' => b'_',
        b'=' => b'+',
        b'[' => b'{',
        b']' => b'}',
        b'\\' => b'|',
        b';' => b':',
        b'\'' => b'"',
        b'`' => b'~',
        b',' => b'<',
        b'.' => b'>',
        b'/' => b'?',
        other => other,
    }
}

